"""Rolling outcomes, health, correlation, and paper-PnL attribution for Anata V2."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from statistics import mean
from typing import Any, Iterable

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import (
    Candle,
    EnsembleDecisionRecord,
    EnsembleSignalWeight,
    Feature,
    ModelHealthSnapshot,
    ModelPredictionRecord,
    ModelVersion,
    PaperTrade,
    SignalHealthSnapshot,
    SignalOutcome,
    ShadowPrediction,
    SimulatedFillRecord,
    TradingSignalRecord,
)
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
        statement = (
            select(SignalOutcome, TradingSignalRecord, ModelPredictionRecord)
            .join(TradingSignalRecord, TradingSignalRecord.signal_id == SignalOutcome.signal_id)
            .join(ModelPredictionRecord, ModelPredictionRecord.prediction_id == TradingSignalRecord.prediction_id)
            .order_by(desc(SignalOutcome.observed_at))
            .limit(max(settings.monitoring_health_window * 40, settings.monitoring_health_window * 2))
        )
        if symbol:
            statement = statement.where(SignalOutcome.symbol == symbol.upper())
        grouped: dict[str, list[tuple[SignalOutcome, TradingSignalRecord, ModelPredictionRecord]]] = defaultdict(list)
        for outcome, signal, prediction in self.session.execute(statement):
            if len(grouped[signal.signal_family]) < settings.monitoring_health_window * 2:
                grouped[signal.signal_family].append((outcome, signal, prediction))
        correlation_increase = self._family_correlation_increase(grouped)
        summaries: list[HealthSummary] = []
        for family, all_rows in grouped.items():
            rows = all_rows[: settings.monitoring_health_window]
            previous_rows = all_rows[settings.monitoring_health_window : settings.monitoring_health_window * 2]
            forecasts = [_finite(item[1].net_expected_return) for item in rows]
            realized = [_finite(item[0].net_return) for item in rows]
            calibration_errors = [
                abs(_finite(signal.confidence) - (1.0 if outcome.directional_hit else 0.0))
                for outcome, signal, _ in rows
                if outcome.directional_hit is not None
            ]
            missing = [
                bool((prediction.payload or {}).get("missing_features"))
                for _, _, prediction in rows
            ]
            calibration = mean(calibration_errors) if calibration_errors else None
            missing_rate = sum(1 for item in missing if item) / len(missing) if missing else 0.0
            ic = _pearson(forecasts, realized)
            expectancy = mean(realized) if realized else None
            diagnostics = self._rolling_diagnostics(
                rows,
                previous_rows,
                correlation_increase=correlation_increase.get(family),
            )
            status, reasons = self._health_status(
                len(rows),
                calibration,
                missing_rate,
                diagnostics=diagnostics,
            )
            summary = HealthSummary(
                signal_family=family,
                observations=len(rows),
                information_coefficient=ic,
                net_expectancy=expectancy,
                calibration_error=calibration,
                missing_feature_rate=missing_rate,
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
            )
            summaries.append(summary)
            self.session.add(
                SignalHealthSnapshot(
                    signal_family=family,
                    symbol=symbol.upper() if symbol else None,
                    health_status=status.value,
                    rolling_information_coefficient=ic,
                    rolling_net_expectancy=expectancy,
                    correlation_increase=diagnostics.signal_correlation_increase,
                    consecutive_errors=0,
                    reason_codes=reasons,
                    observed_at=now,
                    payload={
                        "observations": len(rows),
                        "calibration_error": calibration,
                        "missing_feature_rate": missing_rate,
                        "prediction_drift": diagnostics.prediction_drift,
                        "feature_drift": diagnostics.feature_drift,
                        "out_of_distribution_rate": diagnostics.out_of_distribution_rate,
                        "transaction_cost_increase": diagnostics.transaction_cost_increase,
                        "regime_dependence": diagnostics.regime_dependence,
                        "capacity_decline": diagnostics.capacity_decline,
                        "live_shadow_divergence": diagnostics.live_shadow_divergence,
                    },
                )
            )
            model_keys = {(prediction.model_id, prediction.model_version) for _, _, prediction in rows}
            for model_id, version in model_keys:
                model = self.session.scalar(
                    select(ModelVersion)
                    .where(ModelVersion.model_id == model_id, ModelVersion.version == version)
                    .order_by(desc(ModelVersion.created_at))
                    .limit(1)
                )
                if model is None:
                    continue
                current = str(model.health_status or "HEALTHY").upper()
                effective = HealthStatus(current) if current in HealthStatus._value2member_map_ else HealthStatus.WATCH
                if effective not in {HealthStatus.SUSPENDED, HealthStatus.RETIRED}:
                    model.health_status = status.value
                    effective = status
                self.session.add(
                    ModelHealthSnapshot(
                        model_version_id=model.id,
                        health_status=effective.value,
                        rolling_information_coefficient=ic,
                        rolling_net_expectancy=expectancy,
                        calibration_error=calibration,
                        prediction_drift=diagnostics.prediction_drift,
                        feature_drift=diagnostics.feature_drift,
                        ood_rate=diagnostics.out_of_distribution_rate,
                        missing_feature_rate=missing_rate,
                        reason_codes=reasons,
                        observed_at=now,
                        payload={
                            "observations": len(rows),
                            "symbol": symbol,
                            "transaction_cost_increase": diagnostics.transaction_cost_increase,
                            "signal_correlation_increase": diagnostics.signal_correlation_increase,
                            "regime_dependence": diagnostics.regime_dependence,
                            "capacity_decline": diagnostics.capacity_decline,
                            "live_shadow_divergence": diagnostics.live_shadow_divergence,
                        },
                    )
                )
        if summaries:
            self.session.flush()
        return summaries

    def latest_signal_health(self, family: str, *, symbol: str | None = None) -> HealthStatus:
        statement = select(SignalHealthSnapshot).where(SignalHealthSnapshot.signal_family == family)
        if symbol:
            statement = statement.where(
                (SignalHealthSnapshot.symbol == symbol.upper()) | (SignalHealthSnapshot.symbol.is_(None))
            )
        row = self.session.scalar(statement.order_by(desc(SignalHealthSnapshot.observed_at)).limit(1))
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
        previous = self.session.scalar(
            select(ModelHealthSnapshot)
            .where(ModelHealthSnapshot.model_version_id == model.id)
            .order_by(desc(ModelHealthSnapshot.observed_at))
            .limit(1)
        )
        consecutive = int((previous.payload or {}).get("consecutive_errors", 0)) + 1 if previous else 1
        current = str(model.health_status or "HEALTHY").upper()
        if current in {HealthStatus.SUSPENDED.value, HealthStatus.RETIRED.value}:
            status = HealthStatus(current)
        elif consecutive >= settings.health_suspend_consecutive_errors:
            status = HealthStatus.SUSPENDED
            model.health_status = status.value
            model.suspension_reason = reason
        else:
            status = HealthStatus.DEGRADED
            model.health_status = status.value
        self.session.add(
            ModelHealthSnapshot(
                model_version_id=model.id,
                health_status=status.value,
                reason_codes=["MODEL_INFERENCE_ERROR"],
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
    ) -> tuple[HealthStatus, list[str]]:
        reasons: list[str] = []
        if observations < settings.health_min_observations:
            reasons.append("INSUFFICIENT_LIVE_OUTCOMES")
            return HealthStatus.WATCH, reasons
        if (
            calibration_error is not None
            and calibration_error >= settings.health_degraded_calibration_error
        ) or missing_feature_rate >= settings.health_degraded_missing_feature_rate:
            if calibration_error is not None and calibration_error >= settings.health_degraded_calibration_error:
                reasons.append("CALIBRATION_ERROR_DEGRADED")
            if missing_feature_rate >= settings.health_degraded_missing_feature_rate:
                reasons.append("MISSING_FEATURE_RATE_DEGRADED")
            return HealthStatus.DEGRADED, reasons
        if (
            calibration_error is not None
            and calibration_error >= settings.health_watch_calibration_error
        ) or missing_feature_rate >= settings.health_watch_missing_feature_rate:
            reasons.append("ROLLING_HEALTH_WATCH")
            return HealthStatus.WATCH, reasons
        return HealthStatus.HEALTHY, ["ROLLING_HEALTH_WITHIN_THRESHOLDS"]


def paper_pnl_attribution(
    session: Session,
    *,
    symbol: str | None = None,
    account_id: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int = 2_000,
) -> dict[str, Any]:
    """Return an additive paper-PnL decomposition from recorded trace evidence.

    Alpha is an execution-price-adjusted estimate; ensemble/sizing/broad-market terms
    remain zero until a genuine counterfactual evaluator records them.  The residual is
    explicit so the decomposition never invents precision.
    """
    statement = select(PaperTrade).order_by(desc(PaperTrade.created_at)).limit(min(max(limit, 1), 10_000))
    if symbol:
        statement = statement.where(PaperTrade.symbol == symbol.upper())
    if account_id:
        statement = statement.where(PaperTrade.paper_account_id == account_id)
    if start:
        statement = statement.where(PaperTrade.created_at >= start)
    if end:
        statement = statement.where(PaperTrade.created_at <= end)
    trades = list(session.scalars(statement))
    order_ids = {row.simulated_order_id for row in trades if row.simulated_order_id}
    fills = {
        row.order_id: row
        for row in session.scalars(select(SimulatedFillRecord).where(SimulatedFillRecord.order_id.in_(order_ids)))
    } if order_ids else {}
    total = sum(_finite(row.realized_pnl) for row in trades)
    fees = -sum(max(_finite(row.fee), 0.0) for row in trades)
    funding = -sum(_finite(fills[row.simulated_order_id].funding) for row in trades if row.simulated_order_id in fills)
    slippage = -sum(
        abs(_finite(fills[row.simulated_order_id].slippage)) * max(_finite(fills[row.simulated_order_id].notional), 0.0)
        for row in trades
        if row.simulated_order_id in fills
    )
    gross = sum(_finite((row.raw_payload or {}).get("gross_pnl")) for row in trades)
    alpha = gross - slippage
    components = {
        "alpha_contribution": alpha,
        "ensemble_contribution": 0.0,
        "position_sizing_contribution": 0.0,
        "broad_market_exposure": 0.0,
        "execution_contribution": 0.0,
        "fees": fees,
        "slippage": slippage,
        "funding": funding,
    }
    residual = total - sum(components.values())

    by_model: dict[str, float] = defaultdict(float)
    by_signal: dict[str, float] = defaultdict(float)
    by_family: dict[str, float] = defaultdict(float)
    by_regime: dict[str, float] = defaultdict(float)
    by_symbol: dict[str, float] = defaultdict(float)
    by_external: dict[str, float] = defaultdict(float)
    for trade in trades:
        trade_alpha = _finite((trade.raw_payload or {}).get("gross_pnl"))
        by_symbol[trade.symbol] += _finite(trade.realized_pnl)
        if not trade.decision_trace_id:
            continue
        ensemble = session.scalar(
            select(EnsembleDecisionRecord)
            .where(EnsembleDecisionRecord.decision_trace_id == trade.decision_trace_id)
            .order_by(desc(EnsembleDecisionRecord.generated_at))
            .limit(1)
        )
        if ensemble is None:
            continue
        by_regime[ensemble.current_regime or "unknown"] += _finite(trade.realized_pnl)
        weights = list(
            session.scalars(
                select(EnsembleSignalWeight).where(
                    EnsembleSignalWeight.ensemble_decision_id == ensemble.ensemble_decision_id,
                    EnsembleSignalWeight.weight > 0,
                )
            )
        )
        for weight_row in weights:
            signal = session.scalar(
                select(TradingSignalRecord).where(TradingSignalRecord.signal_id == weight_row.signal_id).limit(1)
            )
            if signal is None:
                continue
            contribution = trade_alpha * _finite(weight_row.weight)
            by_signal[signal.signal_id] += contribution
            by_family[signal.signal_family] += contribution
            prediction = session.scalar(
                select(ModelPredictionRecord).where(ModelPredictionRecord.prediction_id == signal.prediction_id).limit(1)
            )
            if prediction:
                by_model[f"{prediction.model_id}:{prediction.model_version}"] += contribution
                key = "available" if prediction.external_context_available else "unavailable"
                by_external[key] += contribution
    return {
        "paper_only": True,
        "trade_count": len(trades),
        "total_paper_pnl": total,
        "components": {**components, "unexplained_residual": residual},
        "by_model": dict(by_model),
        "by_signal": dict(by_signal),
        "by_signal_family": dict(by_family),
        "by_symbol": dict(by_symbol),
        "by_regime": dict(by_regime),
        "by_external_ai_availability": dict(by_external),
        "limitations": [
            "Ensemble, sizing, and broad-market counterfactual contributions remain unassigned until a counterfactual evaluator records them.",
            "Slippage attribution is an estimate from simulated fill metadata.",
        ],
    }
