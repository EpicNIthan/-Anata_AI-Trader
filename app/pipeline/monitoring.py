"""Rolling outcomes, health, correlation, and paper-PnL attribution for Anata V2."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from statistics import mean, pstdev
from typing import Any, Iterable

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import (
    Candle,
    Feature,
    ModelHealthSnapshot,
    ModelPredictionRecord,
    ModelVersion,
    ShadowPrediction,
    SignalHealthSnapshot,
    SignalOutcome,
    TradingSignalRecord,
)
from app.features.schema import values_from_feature
from app.pipeline.attribution import paper_pnl_attribution
from app.pipeline.domain import HealthStatus, TradingSignal


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_scale = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_scale = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    return numerator / (left_scale * right_scale) if left_scale and right_scale else None


def _ks_distance(left: Iterable[float], right: Iterable[float]) -> float | None:
    """Return the deterministic two-sample Kolmogorov-Smirnov distance."""
    left_values = sorted(_finite(value) for value in left)
    right_values = sorted(_finite(value) for value in right)
    if len(left_values) < 2 or len(right_values) < 2:
        return None
    left_index = right_index = 0
    distance = 0.0
    for value in sorted(set(left_values + right_values)):
        while left_index < len(left_values) and left_values[left_index] <= value:
            left_index += 1
        while right_index < len(right_values) and right_values[right_index] <= value:
            right_index += 1
        distance = max(
            distance,
            abs((left_index / len(left_values)) - (right_index / len(right_values))),
        )
    return min(max(distance, 0.0), 1.0)


def _positive_relative_change(current: Iterable[float], reference: Iterable[float]) -> float | None:
    current_values = [_finite(value) for value in current]
    reference_values = [_finite(value) for value in reference]
    if not current_values or not reference_values:
        return None
    current_mean, reference_mean = mean(current_values), mean(reference_values)
    scale = max(abs(reference_mean), 1e-9)
    return max((current_mean - reference_mean) / scale, 0.0)


def _positive_relative_decline(current: Iterable[float], reference: Iterable[float]) -> float | None:
    current_values = [_finite(value) for value in current]
    reference_values = [_finite(value) for value in reference]
    if not current_values or not reference_values:
        return None
    current_mean, reference_mean = mean(current_values), mean(reference_values)
    if reference_mean <= 1e-9:
        return 0.0
    return min(max((reference_mean - current_mean) / reference_mean, 0.0), 1.0)


def _setting(name: str, default: Any) -> Any:
    """Keep focused tests/backward-compatible injected settings lightweight."""
    return getattr(settings, name, default)


_Observation = tuple[SignalOutcome, TradingSignalRecord, ModelPredictionRecord]


@dataclass(frozen=True)
class HealthSummary:
    signal_family: str
    observations: int
    information_coefficient: float | None
    net_expectancy: float | None
    calibration_error: float | None
    missing_feature_rate: float
    health_status: HealthStatus
    reason_codes: tuple[str, ...]
    prediction_drift: float | None = None
    feature_drift: float | None = None
    out_of_distribution_rate: float | None = None
    transaction_cost_increase: float | None = None
    signal_correlation_increase: float | None = None
    regime_dependence: float | None = None
    capacity_decline: float | None = None
    live_shadow_divergence: float | None = None
    consecutive_errors: int = 0
    recommended_weight_multiplier: float = 1.0
    recommended_action: str = "NORMAL_WEIGHT"


@dataclass(frozen=True)
class _RollingDiagnostics:
    prediction_drift: float | None = None
    feature_drift: float | None = None
    out_of_distribution_rate: float | None = None
    transaction_cost_increase: float | None = None
    signal_correlation_increase: float | None = None
    regime_dependence: float | None = None
    capacity_decline: float | None = None
    live_shadow_divergence: float | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _HealthEvaluation:
    observations: int
    reference_observations: int
    information_coefficient: float | None
    net_expectancy: float | None
    calibration_error: float | None
    missing_feature_rate: float
    diagnostics: _RollingDiagnostics
    health_status: HealthStatus
    reason_codes: tuple[str, ...]


class RollingHealthMonitor:
    """Label matured forecasts and persist bounded rolling health evidence.

    The monitor never promotes or reactivates a model.  A suspended/retired registry
    record remains unavailable until an explicit operator action changes its policy.
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    def update_symbol(self, symbol: str, *, now: datetime | None = None) -> dict[str, Any]:
        now = _aware(now) or datetime.now(timezone.utc)
        labeled = self.label_matured_outcomes(symbol=symbol, now=now)
        health = self.record_health(symbol=symbol, now=now)
        return {"labeled_outcomes": labeled, "health_snapshots": len(health)}

    def label_matured_outcomes(
        self,
        *,
        symbol: str,
        now: datetime | None = None,
        limit: int | None = None,
    ) -> int:
        now = _aware(now) or datetime.now(timezone.utc)
        bounded = min(max(int(limit or settings.monitoring_outcome_batch_size), 1), 2_000)
        rows = list(
            self.session.execute(
                select(TradingSignalRecord, ModelPredictionRecord)
                .join(ModelPredictionRecord, ModelPredictionRecord.prediction_id == TradingSignalRecord.prediction_id)
                .outerjoin(SignalOutcome, SignalOutcome.signal_id == TradingSignalRecord.signal_id)
                .where(
                    TradingSignalRecord.symbol == symbol.upper(),
                    SignalOutcome.id.is_(None),
                )
                .order_by(TradingSignalRecord.generated_at)
                .limit(bounded)
            )
        )
        created = 0
        for signal, prediction in rows:
            generated = _aware(signal.generated_at)
            if generated is None:
                continue
            due = generated + timedelta(seconds=max(int(prediction.forecast_horizon_seconds or 0), 1))
            if due > now:
                continue
            start = self.session.scalar(
                select(Candle)
                .where(
                    Candle.symbol == signal.symbol,
                    Candle.is_closed.is_(True),
                    Candle.close_time.is_not(None),
                    Candle.close_time <= generated,
                )
                .order_by(desc(Candle.close_time))
                .limit(1)
            )
            end = self.session.scalar(
                select(Candle)
                .where(
                    Candle.symbol == signal.symbol,
                    Candle.is_closed.is_(True),
                    Candle.close_time.is_not(None),
                    Candle.close_time >= due,
                    Candle.close_time <= now,
                )
                .order_by(Candle.close_time)
                .limit(1)
            )
            if start is None or end is None or start.close <= 0:
                continue
            realized_return = (float(end.close) / float(start.close)) - 1.0
            direction = str(signal.direction or "FLAT").upper()
            hit = (
                (direction == "LONG" and realized_return > 0)
                or (direction == "SHORT" and realized_return < 0)
                or (direction == "FLAT" and abs(realized_return) <= max(signal.expected_cost, 1e-9))
            )
            realized_cost = max(_finite(signal.expected_cost), 0.0)
            signed_return = realized_return if direction != "SHORT" else -realized_return
            observed_at = _aware(end.close_time) or due
            self.session.add(
                SignalOutcome(
                    signal_id=signal.signal_id,
                    symbol=signal.symbol,
                    horizon_seconds=prediction.forecast_horizon_seconds,
                    realized_return=realized_return,
                    realized_cost=realized_cost,
                    net_return=signed_return - realized_cost,
                    directional_hit=hit,
                    observed_at=observed_at,
                    payload={
                        "start_candle_id": start.id,
                        "end_candle_id": end.id,
                        "start_price": start.close,
                        "end_price": end.close,
                        "point_in_time": True,
                    },
                )
            )
            created += 1
        if created:
            self.session.flush()
        return created

    def record_health(self, *, symbol: str | None = None, now: datetime | None = None) -> list[HealthSummary]:
        now = _aware(now) or datetime.now(timezone.utc)
        window = max(int(_setting("monitoring_health_window", 100)), 2)
        statement = (
            select(SignalOutcome, TradingSignalRecord, ModelPredictionRecord)
            .join(TradingSignalRecord, TradingSignalRecord.signal_id == SignalOutcome.signal_id)
            .join(ModelPredictionRecord, ModelPredictionRecord.prediction_id == TradingSignalRecord.prediction_id)
            .order_by(desc(SignalOutcome.observed_at))
            .limit(min(max(window * 100, window * 2), 10_000))
        )
        if symbol:
            statement = statement.where(SignalOutcome.symbol == symbol.upper())
        observations = list(self.session.execute(statement))
        grouped: dict[str, list[_Observation]] = defaultdict(list)
        model_grouped: dict[tuple[str, str, str], list[_Observation]] = defaultdict(list)
        for outcome, signal, prediction in observations:
            row = (outcome, signal, prediction)
            if len(grouped[signal.signal_family]) < window * 2:
                grouped[signal.signal_family].append(row)
            model_key = self._model_key(prediction)
            if len(model_grouped[model_key]) < window * 2:
                model_grouped[model_key].append(row)

        feature_ids = {
            prediction.feature_id
            for _, _, prediction in observations
            if prediction.feature_id is not None
        }
        feature_by_id = {
            feature.id: feature
            for feature in self.session.scalars(select(Feature).where(Feature.id.in_(feature_ids)))
        } if feature_ids else {}
        trace_ids = {prediction.decision_trace_id for _, _, prediction in observations}
        predictions_by_trace, shadow_prediction_ids = self._prediction_evidence(trace_ids)
        correlation_increase = self._family_correlation_increase(grouped, window=window)

        model_cache: dict[tuple[str, str, str], ModelVersion | None] = {}
        family_models: dict[str, list[ModelVersion]] = defaultdict(list)
        for key, model_rows in model_grouped.items():
            model = self._resolve_model(model_rows[0][2], cache=model_cache)
            if model is not None and all(existing.id != model.id for existing in family_models[model_rows[0][1].signal_family]):
                family_models[model_rows[0][1].signal_family].append(model)

        summaries: list[HealthSummary] = []
        for family, all_rows in grouped.items():
            evaluation = self._evaluate_rows(
                all_rows,
                family=family,
                symbol=symbol,
                feature_by_id=feature_by_id,
                predictions_by_trace=predictions_by_trace,
                shadow_prediction_ids=shadow_prediction_ids,
                correlation_increase=correlation_increase.get(family),
                window=window,
            )
            status, sticky_reason = self._sticky_signal_status(family, symbol, evaluation.health_status)
            reasons = list(evaluation.reason_codes)
            if sticky_reason:
                reasons.append(sticky_reason)
            consecutive_errors = max(
                (self._current_consecutive_errors(model) for model in family_models.get(family, [])),
                default=0,
            )
            weight_multiplier, action = self._health_policy(status)
            diagnostics = evaluation.diagnostics
            summary = HealthSummary(
                signal_family=family,
                observations=evaluation.observations,
                information_coefficient=evaluation.information_coefficient,
                net_expectancy=evaluation.net_expectancy,
                calibration_error=evaluation.calibration_error,
                missing_feature_rate=evaluation.missing_feature_rate,
                health_status=status,
                reason_codes=tuple(reasons),
                prediction_drift=diagnostics.prediction_drift,
                feature_drift=diagnostics.feature_drift,
                out_of_distribution_rate=diagnostics.out_of_distribution_rate,
                transaction_cost_increase=diagnostics.transaction_cost_increase,
                signal_correlation_increase=diagnostics.signal_correlation_increase,
                regime_dependence=diagnostics.regime_dependence,
                capacity_decline=diagnostics.capacity_decline,
                live_shadow_divergence=diagnostics.live_shadow_divergence,
                consecutive_errors=consecutive_errors,
                recommended_weight_multiplier=weight_multiplier,
                recommended_action=action,
            )
            summaries.append(summary)
            self.session.add(
                SignalHealthSnapshot(
                    signal_family=family,
                    symbol=symbol.upper() if symbol else None,
                    health_status=status.value,
                    rolling_information_coefficient=evaluation.information_coefficient,
                    rolling_net_expectancy=evaluation.net_expectancy,
                    calibration_error=evaluation.calibration_error,
                    prediction_drift=diagnostics.prediction_drift,
                    feature_drift=diagnostics.feature_drift,
                    ood_rate=diagnostics.out_of_distribution_rate,
                    missing_feature_rate=evaluation.missing_feature_rate,
                    live_shadow_divergence=diagnostics.live_shadow_divergence,
                    transaction_cost_increase=diagnostics.transaction_cost_increase,
                    correlation_increase=diagnostics.signal_correlation_increase,
                    regime_dependence=diagnostics.regime_dependence,
                    capacity_decline=diagnostics.capacity_decline,
                    consecutive_errors=consecutive_errors,
                    recommended_weight_multiplier=weight_multiplier,
                    recommended_action=action,
                    reason_codes=reasons,
                    observed_at=now,
                    payload={
                        "observations": evaluation.observations,
                        "reference_observations": evaluation.reference_observations,
                        "metric_evidence": diagnostics.evidence,
                    },
                )
            )

        for key, all_rows in model_grouped.items():
            model = self._resolve_model(all_rows[0][2], cache=model_cache)
            if model is None:
                continue
            family = all_rows[0][1].signal_family
            evaluation = self._evaluate_rows(
                all_rows,
                family=family,
                symbol=symbol,
                feature_by_id=feature_by_id,
                predictions_by_trace=predictions_by_trace,
                shadow_prediction_ids=shadow_prediction_ids,
                correlation_increase=correlation_increase.get(family),
                window=window,
            )
            effective = self._normalized_health(model.health_status)
            reasons = list(evaluation.reason_codes)
            if effective not in {HealthStatus.SUSPENDED, HealthStatus.RETIRED}:
                effective = evaluation.health_status
                model.health_status = effective.value
            else:
                reasons.append("TERMINAL_MODEL_HEALTH_REQUIRES_EXPLICIT_REACTIVATION")
            consecutive_errors = self._current_consecutive_errors(model)
            if consecutive_errors and effective not in {HealthStatus.SUSPENDED, HealthStatus.RETIRED}:
                if effective in {HealthStatus.HEALTHY, HealthStatus.WATCH}:
                    effective = HealthStatus.DEGRADED
                    model.health_status = effective.value
                reasons.append("CONSECUTIVE_MODEL_ERRORS_ACTIVE")
            weight_multiplier, action = self._health_policy(effective)
            diagnostics = evaluation.diagnostics
            self.session.add(
                ModelHealthSnapshot(
                    model_version_id=model.id,
                    health_status=effective.value,
                    rolling_information_coefficient=evaluation.information_coefficient,
                    rolling_net_expectancy=evaluation.net_expectancy,
                    calibration_error=evaluation.calibration_error,
                    prediction_drift=diagnostics.prediction_drift,
                    feature_drift=diagnostics.feature_drift,
                    ood_rate=diagnostics.out_of_distribution_rate,
                    missing_feature_rate=evaluation.missing_feature_rate,
                    live_shadow_divergence=diagnostics.live_shadow_divergence,
                    transaction_cost_increase=diagnostics.transaction_cost_increase,
                    signal_correlation_increase=diagnostics.signal_correlation_increase,
                    regime_dependence=diagnostics.regime_dependence,
                    capacity_decline=diagnostics.capacity_decline,
                    consecutive_errors=consecutive_errors,
                    recommended_weight_multiplier=weight_multiplier,
                    recommended_action=action,
                    reason_codes=list(dict.fromkeys(reasons)),
                    observed_at=now,
                    payload={
                        "observations": evaluation.observations,
                        "reference_observations": evaluation.reference_observations,
                        "symbol": symbol.upper() if symbol else None,
                        "signal_family": family,
                        "metric_evidence": diagnostics.evidence,
                    },
                )
            )
        if summaries:
            self.session.flush()
        return summaries

    @staticmethod
    def _model_key(prediction: ModelPredictionRecord) -> tuple[str, str, str]:
        if prediction.model_version_id is not None:
            return ("id", str(prediction.model_version_id), "")
        return ("natural", str(prediction.model_id), str(prediction.model_version))

    def _resolve_model(
        self,
        prediction: ModelPredictionRecord,
        *,
        cache: dict[tuple[str, str, str], ModelVersion | None],
    ) -> ModelVersion | None:
        key = self._model_key(prediction)
        if key in cache:
            return cache[key]
        if prediction.model_version_id is not None:
            model = self.session.get(ModelVersion, prediction.model_version_id)
        else:
            model = self.session.scalar(
                select(ModelVersion)
                .where(
                    ModelVersion.model_id == prediction.model_id,
                    ModelVersion.version == prediction.model_version,
                )
                .order_by(desc(ModelVersion.created_at))
                .limit(1)
            )
        cache[key] = model
        return model

    def _prediction_evidence(
        self,
        trace_ids: set[str],
    ) -> tuple[dict[str, list[ModelPredictionRecord]], set[str]]:
        if not trace_ids:
            return {}, set()
        rows = list(
            self.session.scalars(
                select(ModelPredictionRecord)
                .where(ModelPredictionRecord.decision_trace_id.in_(trace_ids))
                .order_by(desc(ModelPredictionRecord.generated_at))
                .limit(10_000)
            )
        )
        by_trace: dict[str, list[ModelPredictionRecord]] = defaultdict(list)
        for prediction in rows:
            by_trace[prediction.decision_trace_id].append(prediction)
        shadow_ids = {
            prediction_id
            for prediction_id in self.session.scalars(
                select(ShadowPrediction.prediction_id).where(
                    ShadowPrediction.decision_trace_id.in_(trace_ids),
                    ShadowPrediction.prediction_id.is_not(None),
                )
            )
            if prediction_id
        }
        return dict(by_trace), shadow_ids

    def _evaluate_rows(
        self,
        all_rows: list[_Observation],
        *,
        family: str,
        symbol: str | None,
        feature_by_id: dict[int, Feature],
        predictions_by_trace: dict[str, list[ModelPredictionRecord]],
        shadow_prediction_ids: set[str],
        correlation_increase: float | None,
        window: int,
    ) -> _HealthEvaluation:
        rows = all_rows[:window]
        reference_rows = all_rows[window : window * 2]
        forecasts = [_finite(signal.net_expected_return) for _, signal, _ in rows]
        realized = [_finite(outcome.net_return) for outcome, _, _ in rows]
        calibration_errors = [
            abs(_finite(signal.confidence) - (1.0 if outcome.directional_hit else 0.0))
            for outcome, signal, _ in rows
            if outcome.directional_hit is not None
        ]
        missing = [bool((prediction.payload or {}).get("missing_features")) for _, _, prediction in rows]
        calibration = mean(calibration_errors) if calibration_errors else None
        missing_rate = sum(1 for item in missing if item) / len(missing) if missing else 0.0
        ic = _pearson(forecasts, realized)
        expectancy = mean(realized) if realized else None
        diagnostics = self._rolling_diagnostics(
            rows,
            reference_rows,
            family=family,
            symbol=symbol,
            feature_by_id=feature_by_id,
            predictions_by_trace=predictions_by_trace,
            shadow_prediction_ids=shadow_prediction_ids,
            correlation_increase=correlation_increase,
        )
        status, reasons = self._health_status(
            len(rows),
            calibration,
            missing_rate,
            information_coefficient=ic,
            net_expectancy=expectancy,
            diagnostics=diagnostics,
        )
        return _HealthEvaluation(
            observations=len(rows),
            reference_observations=len(reference_rows),
            information_coefficient=ic,
            net_expectancy=expectancy,
            calibration_error=calibration,
            missing_feature_rate=missing_rate,
            diagnostics=diagnostics,
            health_status=status,
            reason_codes=tuple(reasons),
        )

    def _rolling_diagnostics(
        self,
        rows: list[_Observation],
        reference_rows: list[_Observation],
        *,
        family: str,
        symbol: str | None,
        feature_by_id: dict[int, Feature],
        predictions_by_trace: dict[str, list[ModelPredictionRecord]],
        shadow_prediction_ids: set[str],
        correlation_increase: float | None,
    ) -> _RollingDiagnostics:
        minimum_reference = max(int(_setting("health_min_reference_observations", 10)), 2)
        reference_sufficient = len(reference_rows) >= minimum_reference
        prediction_drift = None
        transaction_cost_increase = None
        capacity_decline = None
        historical_divergence = None
        if reference_sufficient:
            prediction_drift = _ks_distance(
                (prediction.expected_return for _, _, prediction in rows),
                (prediction.expected_return for _, _, prediction in reference_rows),
            )
            transaction_cost_increase = _positive_relative_change(
                (max(_finite(outcome.realized_cost), 0.0) for outcome, _, _ in rows),
                (max(_finite(outcome.realized_cost), 0.0) for outcome, _, _ in reference_rows),
            )
            capacity_decline = _positive_relative_decline(
                (self._capacity_proxy(outcome, signal) for outcome, signal, _ in rows),
                (self._capacity_proxy(outcome, signal) for outcome, signal, _ in reference_rows),
            )
            historical_divergence = _ks_distance(
                (outcome.net_return for outcome, _, _ in rows),
                (outcome.net_return for outcome, _, _ in reference_rows),
            )

        feature_drift, ood_rate, feature_evidence = self._feature_diagnostics(
            rows,
            reference_rows if reference_sufficient else [],
            feature_by_id=feature_by_id,
        )
        shadow_divergence, shadow_pairs = self._shadow_divergence(
            rows,
            family=family,
            symbol=symbol,
            predictions_by_trace=predictions_by_trace,
            shadow_prediction_ids=shadow_prediction_ids,
        )
        divergence_values = [
            value for value in (historical_divergence, shadow_divergence) if value is not None
        ]
        live_shadow_divergence = max(divergence_values) if divergence_values else None
        regime_dependence, regime_evidence = self._regime_dependence(rows)
        evidence = {
            "reference_sufficient": reference_sufficient,
            "minimum_reference_observations": minimum_reference,
            "prediction_drift_method": "two_sample_ks_expected_return",
            "feature_drift_method": "mean_top_decile_two_sample_ks",
            "ood_method": "reference_zscore_feature_fraction",
            "transaction_cost_method": "positive_relative_mean_increase",
            "capacity_method": "relative_decline_liquidity_cost_proxy",
            "historical_performance_divergence": historical_divergence,
            "paired_live_shadow_divergence": shadow_divergence,
            "live_shadow_pairs": shadow_pairs,
            "regime_method": "eta_squared_net_return",
            "regime_evidence": regime_evidence,
            **feature_evidence,
        }
        return _RollingDiagnostics(
            prediction_drift=prediction_drift,
            feature_drift=feature_drift,
            out_of_distribution_rate=ood_rate,
            transaction_cost_increase=transaction_cost_increase,
            signal_correlation_increase=correlation_increase,
            regime_dependence=regime_dependence,
            capacity_decline=capacity_decline,
            live_shadow_divergence=live_shadow_divergence,
            evidence=evidence,
        )

    @staticmethod
    def _feature_values(
        prediction: ModelPredictionRecord,
        feature_by_id: dict[int, Feature],
    ) -> dict[str, float]:
        payload = prediction.payload or {}
        required = payload.get("required_features")
        required_columns = [str(item) for item in required] if isinstance(required, list) else []
        values: dict[str, Any] = {}
        if prediction.feature_id is not None and prediction.feature_id in feature_by_id:
            feature = feature_by_id[prediction.feature_id]
            feature_payload = feature.payload or {}
            raw_values = feature_payload.get("values")
            if isinstance(raw_values, dict):
                values = raw_values
            else:
                values = values_from_feature(feature, required_columns or None)
        if not values:
            for key in ("feature_values", "features"):
                candidate = payload.get(key)
                if isinstance(candidate, dict):
                    values = candidate
                    break
        if required_columns:
            values = {key: values.get(key) for key in required_columns if key in values}
        output: dict[str, float] = {}
        for key, value in values.items():
            if isinstance(value, bool):
                output[str(key)] = float(value)
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                output[str(key)] = number
        return output

    def _feature_diagnostics(
        self,
        rows: list[_Observation],
        reference_rows: list[_Observation],
        *,
        feature_by_id: dict[int, Feature],
    ) -> tuple[float | None, float | None, dict[str, Any]]:
        recent_vectors = [
            vector
            for _, _, prediction in rows
            if (vector := self._feature_values(prediction, feature_by_id))
        ]
        reference_vectors = [
            vector
            for _, _, prediction in reference_rows
            if (vector := self._feature_values(prediction, feature_by_id))
        ]
        minimum_reference = max(int(_setting("health_min_reference_observations", 10)), 2)
        if len(recent_vectors) < 2 or len(reference_vectors) < minimum_reference:
            return None, None, {
                "recent_feature_vectors": len(recent_vectors),
                "reference_feature_vectors": len(reference_vectors),
                "feature_drift_unavailable": "INSUFFICIENT_COMPARABLE_FEATURE_VECTORS",
            }

        common_features = sorted(
            set.intersection(
                *(set(vector) for vector in [*recent_vectors, *reference_vectors])
            )
        ) if recent_vectors and reference_vectors else []
        drift_by_feature: dict[str, float] = {}
        reference_by_feature: dict[str, list[float]] = {}
        for name in common_features:
            recent = [vector[name] for vector in recent_vectors if name in vector]
            reference = [vector[name] for vector in reference_vectors if name in vector]
            if len(recent) < 2 or len(reference) < minimum_reference:
                continue
            distance = _ks_distance(recent, reference)
            if distance is not None:
                drift_by_feature[name] = distance
                reference_by_feature[name] = reference
        if not drift_by_feature:
            return None, None, {
                "recent_feature_vectors": len(recent_vectors),
                "reference_feature_vectors": len(reference_vectors),
                "comparable_features": 0,
                "feature_drift_unavailable": "NO_COMMON_NUMERIC_FEATURES",
            }

        top_count = max(1, math.ceil(len(drift_by_feature) * 0.10))
        ordered_drift = sorted(drift_by_feature.items(), key=lambda item: (-item[1], item[0]))
        feature_drift = mean(value for _, value in ordered_drift[:top_count])
        zscore_threshold = max(_finite(_setting("health_ood_zscore_threshold", 4.0), 4.0), 0.1)
        feature_fraction = min(
            max(_finite(_setting("health_ood_feature_fraction", 0.10), 0.10), 1e-9),
            1.0,
        )
        ood_rows = 0
        comparable_rows = 0
        for vector in recent_vectors:
            outliers = comparable = 0
            for name, reference in reference_by_feature.items():
                if name not in vector:
                    continue
                center = mean(reference)
                scale = pstdev(reference) if len(reference) > 1 else 0.0
                floor = max(abs(center) * 1e-6, 1e-9)
                comparable += 1
                if scale <= floor:
                    is_outlier = abs(vector[name] - center) > floor
                else:
                    is_outlier = abs(vector[name] - center) / scale > zscore_threshold
                outliers += int(is_outlier)
            if comparable:
                comparable_rows += 1
                if outliers / comparable >= feature_fraction:
                    ood_rows += 1
        ood_rate = ood_rows / comparable_rows if comparable_rows else None
        return feature_drift, ood_rate, {
            "recent_feature_vectors": len(recent_vectors),
            "reference_feature_vectors": len(reference_vectors),
            "comparable_features": len(drift_by_feature),
            "top_feature_drift": dict(ordered_drift[:10]),
            "ood_comparable_rows": comparable_rows,
            "ood_zscore_threshold": zscore_threshold,
            "ood_feature_fraction": feature_fraction,
        }

    @staticmethod
    def _shadow_divergence(
        rows: list[_Observation],
        *,
        family: str,
        symbol: str | None,
        predictions_by_trace: dict[str, list[ModelPredictionRecord]],
        shadow_prediction_ids: set[str],
    ) -> tuple[float | None, int]:
        values: list[float] = []
        normalized_symbol = symbol.upper() if symbol else None
        for _, _, live in rows:
            for candidate in predictions_by_trace.get(live.decision_trace_id, []):
                payload = candidate.payload or {}
                is_shadow = bool(payload.get("shadow_only")) or candidate.prediction_id in shadow_prediction_ids
                if not is_shadow or candidate.prediction_id == live.prediction_id:
                    continue
                if candidate.model_family != family:
                    continue
                if normalized_symbol and candidate.symbol != normalized_symbol:
                    continue
                difference = abs(_finite(live.expected_return) - _finite(candidate.expected_return))
                scale = abs(_finite(live.expected_return)) + abs(_finite(candidate.expected_return))
                values.append(min(difference / max(scale, 1e-9), 1.0))
        return (mean(values), len(values)) if values else (None, 0)

    @staticmethod
    def _regime_dependence(rows: list[_Observation]) -> tuple[float | None, dict[str, int]]:
        grouped: dict[str, list[float]] = defaultdict(list)
        for outcome, signal, prediction in rows:
            regime = str(signal.regime or prediction.regime or "").strip().lower()
            if not regime or regime == "unknown":
                continue
            grouped[regime].append(_finite(outcome.net_return))
        counts = {name: len(values) for name, values in sorted(grouped.items())}
        values = [value for group in grouped.values() for value in group]
        if len(values) < 2 or not grouped:
            return None, counts
        if len(grouped) == 1:
            return 1.0, counts
        overall = mean(values)
        total_variation = sum((value - overall) ** 2 for value in values)
        if total_variation <= 1e-18:
            return 0.0, counts
        between = sum(len(group) * (mean(group) - overall) ** 2 for group in grouped.values())
        return min(max(between / total_variation, 0.0), 1.0), counts

    @staticmethod
    def _capacity_proxy(outcome: SignalOutcome, signal: TradingSignalRecord) -> float:
        maximum_cost = max(_finite(_setting("v2_max_expected_cost_pct", 0.003), 0.003), 1e-9)
        liquidity = min(max(_finite(signal.liquidity_score), 0.0), 1.0)
        observed_cost = max(_finite(outcome.realized_cost, _finite(signal.expected_cost)), 0.0)
        cost_capacity = max(1.0 - min(observed_cost / maximum_cost, 1.0), 0.0)
        return liquidity * cost_capacity

    @staticmethod
    def _series_by_minute(rows: list[_Observation]) -> dict[str, float]:
        buckets: dict[str, list[float]] = defaultdict(list)
        for outcome, _, _ in rows:
            timestamp = _aware(outcome.observed_at)
            if timestamp is None:
                continue
            key = timestamp.replace(second=0, microsecond=0).isoformat()
            buckets[key].append(_finite(outcome.net_return))
        return {key: mean(values) for key, values in buckets.items()}

    @classmethod
    def _aligned_correlation(
        cls,
        left: list[_Observation],
        right: list[_Observation],
        *,
        minimum_points: int,
    ) -> float | None:
        left_series, right_series = cls._series_by_minute(left), cls._series_by_minute(right)
        common = sorted(set(left_series) & set(right_series))
        if len(common) < minimum_points:
            return None
        return _pearson(
            [left_series[key] for key in common],
            [right_series[key] for key in common],
        )

    @classmethod
    def _family_correlation_increase(
        cls,
        grouped: dict[str, list[_Observation]],
        *,
        window: int,
    ) -> dict[str, float | None]:
        output: dict[str, float | None] = {family: None for family in grouped}
        families = sorted(grouped)
        minimum_points = max(int(_setting("health_min_reference_observations", 10)), 2)
        for family in families:
            increases: list[float] = []
            for other in families:
                if other == family:
                    continue
                recent = cls._aligned_correlation(
                    grouped[family][:window],
                    grouped[other][:window],
                    minimum_points=minimum_points,
                )
                reference = cls._aligned_correlation(
                    grouped[family][window : window * 2],
                    grouped[other][window : window * 2],
                    minimum_points=minimum_points,
                )
                if recent is None or reference is None:
                    continue
                increases.append(max(abs(recent) - abs(reference), 0.0))
            if increases:
                output[family] = max(increases)
        return output

    @staticmethod
    def _normalized_health(value: Any) -> HealthStatus:
        try:
            return HealthStatus(str(value or "WATCH").upper())
        except ValueError:
            return HealthStatus.WATCH

    @staticmethod
    def _health_policy(status: HealthStatus) -> tuple[float, str]:
        return {
            HealthStatus.HEALTHY: (1.0, "NORMAL_WEIGHT"),
            HealthStatus.WATCH: (0.75, "VISIBLE_WARNING"),
            HealthStatus.DEGRADED: (0.35, "REDUCED_ENSEMBLE_WEIGHT"),
            HealthStatus.SUSPENDED: (0.0, "BLOCK_NEW_EXPOSURE"),
            HealthStatus.RETIRED: (0.0, "UNAVAILABLE_FOR_NEW_DECISIONS"),
        }[status]

    def _sticky_signal_status(
        self,
        family: str,
        symbol: str | None,
        proposed: HealthStatus,
    ) -> tuple[HealthStatus, str | None]:
        statement = select(SignalHealthSnapshot).where(SignalHealthSnapshot.signal_family == family)
        if symbol:
            statement = statement.where(SignalHealthSnapshot.symbol == symbol.upper())
        else:
            statement = statement.where(SignalHealthSnapshot.symbol.is_(None))
        previous = self.session.scalar(statement.order_by(desc(SignalHealthSnapshot.observed_at)).limit(1))
        if previous is None:
            return proposed, None
        current = self._normalized_health(previous.health_status)
        if current in {HealthStatus.SUSPENDED, HealthStatus.RETIRED}:
            return current, "TERMINAL_SIGNAL_HEALTH_REQUIRES_EXPLICIT_REACTIVATION"
        return proposed, None

    def _current_consecutive_errors(self, model: ModelVersion) -> int:
        previous = self.session.scalar(
            select(ModelHealthSnapshot)
            .where(ModelHealthSnapshot.model_version_id == model.id)
            .order_by(desc(ModelHealthSnapshot.observed_at), desc(ModelHealthSnapshot.id))
            .limit(1)
        )
        if previous is None:
            return 0
        consecutive = int(previous.consecutive_errors or (previous.payload or {}).get("consecutive_errors", 0) or 0)
        if consecutive <= 0:
            return 0
        latest_success = self.session.scalar(
            select(ModelPredictionRecord)
            .where(
                (ModelPredictionRecord.model_version_id == model.id)
                | (
                    (ModelPredictionRecord.model_id == model.model_id)
                    & (ModelPredictionRecord.model_version == model.version)
                )
            )
            .order_by(desc(ModelPredictionRecord.generated_at), desc(ModelPredictionRecord.id))
            .limit(1)
        )
        success_at = _aware(latest_success.generated_at) if latest_success is not None else None
        previous_at = _aware(previous.observed_at)
        if success_at is not None and previous_at is not None and success_at > previous_at:
            return 0
        return consecutive

    def latest_signal_health(self, family: str, *, symbol: str | None = None) -> HealthStatus:
        statement = select(SignalHealthSnapshot).where(SignalHealthSnapshot.signal_family == family)
        if symbol:
            # Exact symbol policy wins even if a newer global snapshot exists; a
            # global healthy row must never bypass a symbol suspension.
            row = self.session.scalar(
                statement.where(SignalHealthSnapshot.symbol == symbol.upper())
                .order_by(desc(SignalHealthSnapshot.observed_at), desc(SignalHealthSnapshot.id))
                .limit(1)
            )
            if row is None:
                row = self.session.scalar(
                    statement.where(SignalHealthSnapshot.symbol.is_(None))
                    .order_by(desc(SignalHealthSnapshot.observed_at), desc(SignalHealthSnapshot.id))
                    .limit(1)
                )
        else:
            row = self.session.scalar(
                statement.where(SignalHealthSnapshot.symbol.is_(None))
                .order_by(desc(SignalHealthSnapshot.observed_at), desc(SignalHealthSnapshot.id))
                .limit(1)
            )
        if row is None:
            return HealthStatus.HEALTHY
        try:
            return HealthStatus(str(row.health_status).upper())
        except ValueError:
            return HealthStatus.WATCH

    def recent_performance(self, families: Iterable[str], *, symbol: str | None = None) -> dict[str, float]:
        output: dict[str, float] = {}
        for family in set(families):
            statement = (
                select(SignalOutcome.net_return)
                .join(TradingSignalRecord, TradingSignalRecord.signal_id == SignalOutcome.signal_id)
                .where(TradingSignalRecord.signal_family == family, SignalOutcome.net_return.is_not(None))
                .order_by(desc(SignalOutcome.observed_at))
                .limit(settings.monitoring_health_window)
            )
            if symbol:
                statement = statement.where(SignalOutcome.symbol == symbol.upper())
            values = [_finite(value) for value in self.session.scalars(statement)]
            output[family] = max(min(mean(values), 0.2), -0.2) if values else 0.0
        return output

    def signal_correlations(self, signals: Iterable[TradingSignal]) -> dict[tuple[str, str], float]:
        current = list(signals)
        by_family: dict[str, dict[str, float]] = {}
        for family in {item.signal_family for item in current}:
            rows = list(
                self.session.execute(
                    select(SignalOutcome.observed_at, SignalOutcome.net_return)
                    .join(TradingSignalRecord, TradingSignalRecord.signal_id == SignalOutcome.signal_id)
                    .where(TradingSignalRecord.signal_family == family, SignalOutcome.net_return.is_not(None))
                    .order_by(desc(SignalOutcome.observed_at))
                    .limit(settings.monitoring_health_window)
                )
            )
            by_family[family] = {
                (_aware(timestamp) or datetime.min.replace(tzinfo=timezone.utc)).replace(second=0, microsecond=0).isoformat(): _finite(value)
                for timestamp, value in rows
            }
        result: dict[tuple[str, str], float] = {}
        for index, left in enumerate(current):
            for right in current[index + 1 :]:
                left_values, right_values = by_family.get(left.signal_family, {}), by_family.get(right.signal_family, {})
                common = sorted(set(left_values) & set(right_values))
                correlation = _pearson([left_values[key] for key in common], [right_values[key] for key in common])
                if correlation is not None:
                    result[(left.signal_id, right.signal_id)] = correlation
        return result

    def record_model_error(self, model: ModelVersion, reason: str, *, now: datetime | None = None) -> HealthStatus:
        now = _aware(now) or datetime.now(timezone.utc)
        consecutive = self._current_consecutive_errors(model) + 1
        current = self._normalized_health(model.health_status)
        reasons = ["MODEL_INFERENCE_ERROR"]
        if current in {HealthStatus.SUSPENDED, HealthStatus.RETIRED}:
            status = current
            reasons.append("TERMINAL_MODEL_HEALTH_REQUIRES_EXPLICIT_REACTIVATION")
        elif consecutive >= int(_setting("health_suspend_consecutive_errors", 5)):
            status = HealthStatus.SUSPENDED
            model.health_status = status.value
            model.suspension_reason = reason
            reasons.append("CONSECUTIVE_ERROR_SUSPENSION")
        else:
            status = HealthStatus.DEGRADED
            model.health_status = status.value
        weight_multiplier, action = self._health_policy(status)
        self.session.add(
            ModelHealthSnapshot(
                model_version_id=model.id,
                health_status=status.value,
                consecutive_errors=consecutive,
                recommended_weight_multiplier=weight_multiplier,
                recommended_action=action,
                reason_codes=reasons,
                observed_at=now,
                payload={"consecutive_errors": consecutive, "error": reason[:500]},
            )
        )
        self.session.flush()
        return status

    @staticmethod
    def _health_status(
        observations: int,
        calibration_error: float | None,
        missing_feature_rate: float,
        *,
        information_coefficient: float | None = None,
        net_expectancy: float | None = None,
        diagnostics: _RollingDiagnostics | None = None,
    ) -> tuple[HealthStatus, list[str]]:
        if observations < int(_setting("health_min_observations", 20)):
            return HealthStatus.WATCH, ["INSUFFICIENT_LIVE_OUTCOMES"]
        diagnostics = diagnostics or _RollingDiagnostics()
        high_bad = (
            ("CALIBRATION_ERROR", calibration_error, "health_watch_calibration_error", 0.25, "health_degraded_calibration_error", 0.40),
            ("MISSING_FEATURE_RATE", missing_feature_rate, "health_watch_missing_feature_rate", 0.10, "health_degraded_missing_feature_rate", 0.25),
            ("PREDICTION_DRIFT", diagnostics.prediction_drift, "health_watch_prediction_drift", 0.25, "health_degraded_prediction_drift", 0.50),
            ("FEATURE_DRIFT", diagnostics.feature_drift, "health_watch_feature_drift", 0.25, "health_degraded_feature_drift", 0.50),
            ("OOD_RATE", diagnostics.out_of_distribution_rate, "health_watch_ood_rate", 0.10, "health_degraded_ood_rate", 0.25),
            ("LIVE_SHADOW_DIVERGENCE", diagnostics.live_shadow_divergence, "health_watch_live_shadow_divergence", 0.30, "health_degraded_live_shadow_divergence", 0.60),
            ("TRANSACTION_COST_INCREASE", diagnostics.transaction_cost_increase, "health_watch_transaction_cost_increase", 0.25, "health_degraded_transaction_cost_increase", 0.75),
            ("SIGNAL_CORRELATION_INCREASE", diagnostics.signal_correlation_increase, "health_watch_correlation_increase", 0.15, "health_degraded_correlation_increase", 0.30),
            ("REGIME_DEPENDENCE", diagnostics.regime_dependence, "health_watch_regime_dependence", 0.70, "health_degraded_regime_dependence", 0.90),
            ("CAPACITY_DECLINE", diagnostics.capacity_decline, "health_watch_capacity_decline", 0.15, "health_degraded_capacity_decline", 0.35),
        )
        low_bad = (
            ("INFORMATION_COEFFICIENT", information_coefficient, "health_watch_min_information_coefficient", 0.0, "health_degraded_min_information_coefficient", -0.10),
            ("NET_EXPECTANCY", net_expectancy, "health_watch_min_net_expectancy", 0.0, "health_degraded_min_net_expectancy", -0.001),
        )
        degraded: list[str] = []
        watch: list[str] = []
        for code, value, watch_name, watch_default, degraded_name, degraded_default in high_bad:
            if value is None:
                continue
            if value >= _finite(_setting(degraded_name, degraded_default), degraded_default):
                degraded.append(f"{code}_DEGRADED")
            elif value >= _finite(_setting(watch_name, watch_default), watch_default):
                watch.append(f"{code}_WATCH")
        for code, value, watch_name, watch_default, degraded_name, degraded_default in low_bad:
            if value is None:
                continue
            if value <= _finite(_setting(degraded_name, degraded_default), degraded_default):
                degraded.append(f"{code}_DEGRADED")
            elif value <= _finite(_setting(watch_name, watch_default), watch_default):
                watch.append(f"{code}_WATCH")
        if degraded:
            return HealthStatus.DEGRADED, list(dict.fromkeys(degraded + watch))
        if watch:
            return HealthStatus.WATCH, list(dict.fromkeys(watch))
        return HealthStatus.HEALTHY, ["ROLLING_HEALTH_WITHIN_THRESHOLDS"]


# ``paper_pnl_attribution`` remains imported above for backward compatibility.
