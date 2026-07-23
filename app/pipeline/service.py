"""End-to-end V2 orchestration for one paper-trading decision trace."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import case, desc, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import (
    AccountEquity,
    AiDecision,
    Candle,
    ChampionAssignment,
    DecisionTimelineEvent,
    EnsembleDecisionRecord,
    EnsembleSignalWeight,
    ModelPredictionRecord,
    ModelVersion,
    PaperSandboxAccount,
    PaperTrade,
    PortfolioTargetRecord,
    Position,
    TradingSignalRecord,
)
from app.features.feature_builder import FeatureBuilder
from app.pipeline.artifact_models import ArtifactModelError, RegisteredArtifactModel
from app.pipeline.data_quality import PointInTimeValidator
from app.pipeline.domain import HealthStatus, ModelLifecycle, ModelPrediction, SignalLifecycle, new_id
from app.pipeline.ensemble import DeterministicRegimeEnsemble
from app.pipeline.execution import ExecutionOutcome, PaperExecutionSimulator
from app.pipeline.monitoring import RollingHealthMonitor
from app.pipeline.narrow_models import BaselineCostModel, BaselineReliabilityModel, NarrowModel, classify_regime, default_narrow_models
from app.pipeline.portfolio import DeterministicPortfolioConstructor, PortfolioContext
from app.pipeline.registry import ModelRegistry
from app.pipeline.risk import MarketSnapshot, PortfolioRiskEngine, RiskInputs
from app.pipeline.signals import SignalFactory
from app.trading.paper_engine import ExecutionResult


@dataclass(frozen=True)
class PipelineRunResult:
    decision_trace_id: str
    symbol: str
    feature_id: int | None
    prediction_ids: tuple[str, ...]
    signal_ids: tuple[str, ...]
    ensemble_decision_id: str
    portfolio_target_id: str
    risk_decision_id: str
    action: str
    status: str
    message: str
    trade_id: int | None = None
    legacy_decision_id: int | None = None
    requested_exposure: float = 0.0
    approved_exposure: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _ResolvedModel:
    """One inference model and the registry policy that governs its signal."""

    model: NarrowModel
    record: ModelVersion | None
    signal_lifecycle: SignalLifecycle


class V2PipelineService:
    """Run the mandatory Data -> Models -> Signals -> Ensemble -> Portfolio -> Risk -> Execution path."""

    def __init__(self, session: Session, *, models: list[NarrowModel] | None = None) -> None:
        self.session = session
        self._explicit_models = models is not None
        self.models = list(models) if models is not None else []
        self.validator = PointInTimeValidator()
        self.monitor = RollingHealthMonitor(session)
        self.cost_model = BaselineCostModel()
        self.reliability_model = BaselineReliabilityModel()
        self.signal_factory = SignalFactory(minimum_edge=settings.v2_min_net_edge, ttl_seconds=settings.v2_signal_ttl_seconds)
        self.ensemble = DeterministicRegimeEnsemble(
            minimum_edge=settings.v2_min_net_edge,
            correlation_threshold=settings.v2_correlation_penalty_threshold,
            external_context_bound=settings.v2_external_context_max_adjustment,
            ttl_seconds=settings.v2_signal_ttl_seconds,
        )
        self.portfolio = DeterministicPortfolioConstructor(
            max_symbol_exposure=settings.v2_max_symbol_exposure_pct,
            max_gross_exposure=settings.v2_max_gross_exposure_pct,
            max_net_exposure=settings.v2_max_net_exposure_pct,
            max_cluster_exposure=settings.v2_max_cluster_exposure_pct,
            minimum_liquidity=settings.v2_min_liquidity_score,
            minimum_edge=settings.v2_min_net_edge,
        )

    def run_symbol(self, symbol: str, *, account_id: str | None = None) -> PipelineRunResult:
        """Build an auditable paper-only trace for a single symbol.

        No future training, automatic promotion, private exchange API or live-money
        action occurs here. A failed external context simply leaves its bounded input at
        zero while the local narrow-model baseline continues.
        """
        account_id = account_id or settings.v2_champion_account_id
        normalized = symbol.upper()
        trace_id = new_id("trace")
        feature = FeatureBuilder(self.session).build_for_symbol(normalized, interval=settings.paper_trade_timeframe, store=True)
        snapshot = self.validator.snapshot_from_feature(feature, decision_time=feature.as_of)
        self._timeline(trace_id, "FEATURE_SNAPSHOT", "RECORDED", ["POINT_IN_TIME_FEATURE_SNAPSHOT"], {"feature_id": feature.id, "schema": feature.schema_version})
        candles = list(
            self.session.scalars(
                select(Candle)
                .where(Candle.symbol == normalized, Candle.interval == settings.paper_trade_timeframe)
                .order_by(desc(Candle.open_time))
                .limit(60)
            )
        )
        quality = self.validator.validate_candles(
            reversed(candles),
            interval=settings.paper_trade_timeframe,
        )
        if not quality.valid:
            snapshot = snapshot.model_copy(update={"missing_required_features": [*snapshot.missing_required_features, "DATA_QUALITY"]})
        if settings.monitoring_enabled:
            try:
                monitoring = self.monitor.update_symbol(normalized)
                self._timeline(trace_id, "ROLLING_MONITORING", "RECORDED", ["OUTCOMES_AND_HEALTH_REFRESHED"], monitoring)
            except Exception as exc:  # Monitoring cannot become an execution dependency.
                self._timeline(
                    trace_id,
                    "ROLLING_MONITORING",
                    "DEGRADED",
                    ["MONITORING_REFRESH_FAILED"],
                    {"error": f"{type(exc).__name__}: {exc}"[:500]},
                )
        cost = self.cost_model.estimate(snapshot, fee_rate=settings.paper_fee_rate)
        reliability, _ = self.reliability_model.confidence(snapshot)
        resolved, resolution_reasons = self._resolve_active_models(normalized, account_id)
        prediction_inputs: list[tuple[ModelPrediction, HealthStatus, SignalLifecycle]] = []
        inference_errors: list[dict[str, Any]] = []
        for item in resolved:
            try:
                prediction = item.model.predict(snapshot)
                self._record_prediction(prediction, trace_id, feature.id)
                model_health = item.model.health_status()
                signal_health = self.monitor.latest_signal_health(prediction.model_family, symbol=normalized)
                health = self._worst_health(model_health, signal_health)
                prediction_inputs.append((prediction, health, item.signal_lifecycle))
            except Exception as exc:  # One frozen model cannot take down the baseline.
                inference_errors.append(self._record_inference_error(item, exc))
        predictions = [item[0] for item in prediction_inputs]
        prediction_reason = "NARROW_MODELS_GENERATED" if predictions else "NO_ACTIVE_MODEL_PREDICTIONS"
        self._timeline(
            trace_id,
            "MODEL_PREDICTIONS",
            "RECORDED" if predictions else "DEGRADED",
            [prediction_reason, *resolution_reasons],
            {
                "prediction_ids": [item.prediction_id for item in predictions],
                "registered_model_version_ids": [
                    item.record.id for item in resolved if item.record is not None
                ],
                "errors": inference_errors,
            },
        )
        shadow_ids, shadow_errors = self._run_shadow_models(snapshot, trace_id=trace_id, feature_id=feature.id)
        self._timeline(
            trace_id,
            "SHADOW_PREDICTIONS",
            "RECORDED" if not shadow_errors else "DEGRADED",
            ["SHADOW_MODE_NON_EXECUTING"],
            {"prediction_ids": shadow_ids, "errors": shadow_errors},
        )

        signals = [
            self.signal_factory.from_prediction(
                prediction,
                cost=cost,
                liquidity_score=cost.fill_probability,
                health_status=health,
                lifecycle_status=lifecycle,
            )
            for prediction, health, lifecycle in prediction_inputs
        ]
        for signal in signals:
            self._record_signal(signal, trace_id)
        self._timeline(trace_id, "TRADING_SIGNALS", "RECORDED", ["SIGNALS_REGISTERED"], {"signal_ids": [item.signal_id for item in signals]})

        regime = classify_regime(snapshot.values)
        external_adjustment = self._bounded_external_adjustment(snapshot)
        try:
            correlations = self.monitor.signal_correlations(signals) if settings.monitoring_enabled else {}
            recent_performance = (
                self.monitor.recent_performance((item.signal_family for item in signals), symbol=normalized)
                if settings.monitoring_enabled
                else {}
            )
        except Exception as exc:
            correlations, recent_performance = {}, {}
            self._timeline(
                trace_id,
                "ENSEMBLE_MONITORING_INPUTS",
                "DEGRADED",
                ["ROLLING_INPUTS_UNAVAILABLE"],
                {"error": f"{type(exc).__name__}: {exc}"[:500]},
            )
        ensemble_result = self.ensemble.combine(
            normalized,
            signals,
            regime=regime,
            correlations=correlations,
            external_context_score=external_adjustment,
            recent_performance=recent_performance,
        )
        ensemble = ensemble_result.decision
        self._record_ensemble(ensemble, trace_id, ensemble_result.exclusions)
        self._timeline(trace_id, "ENSEMBLE", "RECORDED", ensemble.reason_codes, {"ensemble_decision_id": ensemble.ensemble_decision_id})

        context = self._portfolio_context(account_id)
        context = PortfolioContext(
            equity=context.equity,
            exposures=context.exposures,
            cluster_by_symbol=context.cluster_by_symbol,
            liquidity_score=cost.fill_probability,
            current_gross_exposure=context.current_gross_exposure,
            current_net_exposure=context.current_net_exposure,
        )
        target = self.portfolio.construct(ensemble, context)
        self._record_target(target, trace_id, account_id)
        self._timeline(trace_id, "PORTFOLIO_TARGET", "RECORDED", ["TARGET_EXPOSURE_REQUESTED"], {"portfolio_target_id": target.portfolio_target_id})

        account = self._latest_account(account_id)
        mark, observed = self._latest_market(candles, snapshot, feature)
        risk_inputs = RiskInputs(
            account_id=account_id,
            cash_balance=account.cash_balance,
            equity=account.equity,
            # Freshness is anchored to the source candle, never to the time at
            # which FeatureBuilder happened to run.  Rebuilding a stale feature
            # therefore cannot make a stale market feed tradable.
            market=MarketSnapshot(symbol=normalized, price=mark, observed_at=observed, source="candle"),
            # Reliability is a bounded quality adjustment to ensemble confidence; it
            # does not select leverage or bypass any other risk limit.
            confidence=max(min(ensemble.combined_confidence + (reliability - 0.5) * 0.20, 1.0), 0.0),
            liquidity_score=cost.fill_probability,
            expected_cost=cost.total_cost,
            expected_volatility=ensemble.combined_expected_volatility,
            model_health=self._aggregate_health([item[1] for item in prediction_inputs]),
            signal_health=self._aggregate_health([item.health_status for item in signals]),
            required_features_missing=tuple(snapshot.missing_required_features),
            current_gross_exposure=context.gross_exposure,
            current_net_exposure=context.net_exposure,
        )
        risk = PortfolioRiskEngine(self.session).approve(target, risk_inputs, decision_trace_id=trace_id)
        self._timeline(
            trace_id,
            "RISK_DECISION",
            "APPROVED" if risk.approved else "REJECTED",
            risk.triggered_limits or risk.rejection_reasons or ["RISK_APPROVED"],
            {"risk_decision_id": risk.risk_decision_id, "approved_exposure": risk.approved_exposure},
        )
        execution: ExecutionOutcome | None = None
        if risk.approved:
            execution = PaperExecutionSimulator(self.session).submit_target(
                target=target,
                risk_decision=risk,
                market=risk_inputs.market,
                equity=account.equity,
                account_id=account_id,
                decision_trace_id=trace_id,
            )
            self._timeline(
                trace_id,
                "PAPER_EXECUTION",
                execution.result.status,
                ["PAPER_ONLY_EXECUTION"],
                {"order_id": execution.order.order_id if execution.order else None, "trade_id": execution.result.trade_id},
            )
        else:
            execution = ExecutionOutcome(None, None, ExecutionResult("REJECTED", "; ".join(risk.rejection_reasons)))

        ai_decision = self._record_legacy_decision(feature, ensemble, target, risk, execution, trace_id)
        self.session.commit()
        action = "BUY" if target.requested_delta > 0 else "SELL" if target.requested_delta < 0 else "HOLD"
        return PipelineRunResult(
            decision_trace_id=trace_id,
            symbol=normalized,
            feature_id=feature.id,
            prediction_ids=tuple(item.prediction_id for item in predictions),
            signal_ids=tuple(item.signal_id for item in signals),
            ensemble_decision_id=ensemble.ensemble_decision_id,
            portfolio_target_id=target.portfolio_target_id,
            risk_decision_id=risk.risk_decision_id,
            action=action,
            status=execution.result.status,
            message=execution.result.message,
            trade_id=execution.result.trade_id,
            legacy_decision_id=ai_decision.id,
            requested_exposure=target.requested_target_exposure,
            approved_exposure=risk.approved_exposure,
        )

    def _resolve_active_models(self, symbol: str, account_id: str) -> tuple[list[_ResolvedModel], list[str]]:
        """Resolve the only models permitted to influence this paper account.

        An active sandbox is bound to its one registered candidate.  The champion
        account uses explicit assignments when present and falls back to deterministic
        local baselines only when no assignment exists and policy allows that fallback.
        """
        sandbox: PaperSandboxAccount | None = None
        if account_id != settings.v2_champion_account_id:
            sandbox = self.session.scalar(
                select(PaperSandboxAccount)
                .where(PaperSandboxAccount.account_id == account_id)
                .limit(1)
            )
            if sandbox is None:
                raise ValueError("paper account is neither the champion account nor a registered sandbox")
            if not sandbox.active or sandbox.closed_at is not None:
                raise ValueError("paper sandbox account is closed")

        if self._explicit_models:
            lifecycle = SignalLifecycle.PAPER if sandbox is not None else SignalLifecycle.PRODUCTION
            return [
                _ResolvedModel(model=model, record=None, signal_lifecycle=lifecycle)
                for model in self.models
            ], ["EXPLICIT_TEST_MODEL_SET"]

        if not settings.v2_use_narrow_models:
            return [], ["NARROW_MODEL_INFERENCE_DISABLED"]

        if sandbox is not None:
            if sandbox.model_version_id is None:
                return [], ["SANDBOX_HAS_NO_REGISTERED_MODEL"]
            row = self.session.get(ModelVersion, sandbox.model_version_id)
            if row is None:
                return [], ["SANDBOX_MODEL_RECORD_MISSING"]
            if row.lifecycle_state != ModelLifecycle.PAPER_SANDBOX.value:
                return [], ["SANDBOX_MODEL_LIFECYCLE_MISMATCH"]
            if str(row.health_status or "").upper() in {
                HealthStatus.SUSPENDED.value,
                HealthStatus.RETIRED.value,
            }:
                return [], ["SANDBOX_MODEL_HEALTH_BLOCKED"]
            try:
                model = RegisteredArtifactModel.from_record(row)
            except Exception as exc:
                self._record_registry_model_error(row, exc)
                return [], ["SANDBOX_MODEL_LOAD_FAILED"]
            return [
                _ResolvedModel(model=model, record=row, signal_lifecycle=SignalLifecycle.PAPER)
            ], ["ISOLATED_SANDBOX_MODEL"]

        assignments = list(
            self.session.scalars(
                select(ChampionAssignment)
                .where(
                    ChampionAssignment.active_to.is_(None),
                    ChampionAssignment.symbol_scope.in_((symbol.upper(), "*")),
                )
                .order_by(
                    case((ChampionAssignment.symbol_scope == symbol.upper(), 0), else_=1),
                    desc(ChampionAssignment.active_from),
                )
            )
        )
        resolved: list[_ResolvedModel] = []
        seen_families: set[str] = set()
        reasons: list[str] = []
        for assignment in assignments:
            # An exact-scope assignment intentionally wins over a wildcard even if it
            # is unhealthy.  Falling through to an older wildcard could silently undo
            # an operator's suspension decision.
            if assignment.model_family in seen_families:
                continue
            seen_families.add(assignment.model_family)
            row = self.session.get(ModelVersion, assignment.model_version_id)
            if row is None:
                reasons.append("CHAMPION_MODEL_RECORD_MISSING")
                continue
            if row.lifecycle_state != ModelLifecycle.CHAMPION.value:
                reasons.append("CHAMPION_LIFECYCLE_MISMATCH")
                continue
            if str(row.health_status or "").upper() in {
                HealthStatus.SUSPENDED.value,
                HealthStatus.RETIRED.value,
            }:
                reasons.append("CHAMPION_HEALTH_BLOCKED")
                continue
            try:
                model = RegisteredArtifactModel.from_record(row)
            except Exception as exc:
                self._record_registry_model_error(row, exc)
                reasons.append("CHAMPION_MODEL_LOAD_FAILED")
                continue
            resolved.append(
                _ResolvedModel(model=model, record=row, signal_lifecycle=SignalLifecycle.PRODUCTION)
            )

        if assignments:
            return resolved, ["REGISTERED_CHAMPION_ASSIGNMENTS", *reasons]
        if settings.v2_require_registered_champion:
            return [], ["REGISTERED_CHAMPION_REQUIRED"]
        return [
            _ResolvedModel(model=model, record=None, signal_lifecycle=SignalLifecycle.PRODUCTION)
            for model in default_narrow_models()
        ], ["DETERMINISTIC_BASELINE_FALLBACK"]

    def _run_shadow_models(self, snapshot, *, trace_id: str, feature_id: int | None) -> tuple[list[str], list[dict[str, Any]]]:
        """Persist challenger predictions without ever constructing their signals."""
        if not settings.v2_use_narrow_models:
            return [], []
        rows = list(
            self.session.scalars(
                select(ModelVersion)
                .where(ModelVersion.lifecycle_state == ModelLifecycle.SHADOW.value)
                .order_by(desc(ModelVersion.created_at))
                .limit(50)
            )
        )
        registry = ModelRegistry(self.session)
        prediction_ids: list[str] = []
        errors: list[dict[str, Any]] = []
        for row in rows:
            if str(row.health_status or "").upper() in {
                HealthStatus.SUSPENDED.value,
                HealthStatus.RETIRED.value,
            }:
                continue
            try:
                model = RegisteredArtifactModel.from_record(row)
                prediction = model.predict(snapshot)
                prediction = prediction.model_copy(
                    update={
                        "metadata": {
                            **prediction.metadata,
                            "shadow_only": True,
                            "signal_and_execution_forbidden": True,
                        }
                    }
                )
                self._record_prediction(prediction, trace_id, feature_id)
                registry.record_shadow(
                    prediction,
                    model_version_id=row.id,
                    decision_trace_id=trace_id,
                )
                prediction_ids.append(prediction.prediction_id)
            except Exception as exc:
                self._record_registry_model_error(row, exc)
                errors.append(
                    {
                        "model_version_id": row.id,
                        "error_type": type(exc).__name__,
                        "message": str(exc)[:300],
                    }
                )
        return prediction_ids, errors

    def _record_inference_error(self, item: _ResolvedModel, exc: Exception) -> dict[str, Any]:
        if item.record is not None:
            self._record_registry_model_error(item.record, exc)
        return {
            "model_id": item.model.model_id,
            "model_version_id": item.record.id if item.record is not None else None,
            "error_type": type(exc).__name__,
            "message": str(exc)[:300],
        }

    def _record_registry_model_error(self, row: ModelVersion, exc: Exception) -> None:
        try:
            self.monitor.record_model_error(row, f"{type(exc).__name__}: {exc}")
        except Exception:
            # The trace still records the inference failure. Monitoring persistence is
            # deliberately fail-open so it cannot grant execution or break baselines.
            return

    @staticmethod
    def _worst_health(*values: HealthStatus) -> HealthStatus:
        severity = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.WATCH: 1,
            HealthStatus.DEGRADED: 2,
            HealthStatus.SUSPENDED: 3,
            HealthStatus.RETIRED: 4,
        }
        normalized: list[HealthStatus] = []
        for value in values:
            if isinstance(value, HealthStatus):
                normalized.append(value)
                continue
            try:
                normalized.append(HealthStatus(str(value).upper()))
            except ValueError:
                normalized.append(HealthStatus.WATCH)
        return max(normalized or [HealthStatus.WATCH], key=lambda item: severity[item])

    @classmethod
    def _aggregate_health(cls, values: list[HealthStatus]) -> HealthStatus:
        return cls._worst_health(*values)

    def _record_prediction(self, prediction: ModelPrediction, trace_id: str, feature_id: int | None) -> None:
        registered_id = prediction.metadata.get("registered_model_version_id") if prediction.metadata else None
        try:
            model_version_id = int(registered_id) if registered_id is not None else None
        except (TypeError, ValueError):
            model_version_id = None
        self.session.add(
            ModelPredictionRecord(
                prediction_id=prediction.prediction_id,
                decision_trace_id=trace_id,
                model_version_id=model_version_id,
                model_id=prediction.model_id,
                model_version=prediction.model_version,
                model_family=prediction.model_family,
                symbol=prediction.symbol,
                generated_at=prediction.generated_at,
                valid_from=prediction.valid_from,
                expires_at=prediction.expires_at,
                forecast_horizon_seconds=prediction.forecast_horizon_seconds,
                expected_return=prediction.expected_return,
                expected_volatility=prediction.expected_volatility,
                probability_up=prediction.probability_up,
                probability_down=prediction.probability_down,
                confidence=prediction.confidence,
                calibration_score=prediction.calibration_score,
                uncertainty=prediction.uncertainty,
                regime=prediction.regime,
                feature_schema_version=prediction.feature_schema_version,
                feature_snapshot_id=prediction.feature_snapshot_id,
                feature_id=feature_id,
                data_version=prediction.data_version,
                external_context_available=prediction.external_context_available,
                payload=prediction.metadata,
            )
        )
        self.session.flush()

    def _record_signal(self, signal, trace_id: str) -> None:
        self.session.add(
            TradingSignalRecord(
                signal_id=signal.signal_id,
                prediction_id=signal.prediction_id,
                decision_trace_id=trace_id,
                signal_family=signal.signal_family,
                symbol=signal.symbol,
                generated_at=signal.generated_at,
                valid_until=signal.valid_until,
                direction=signal.direction.value,
                strength=signal.strength,
                expected_return=signal.expected_return,
                expected_cost=signal.expected_cost,
                net_expected_return=signal.net_expected_return,
                confidence=signal.confidence,
                uncertainty=signal.uncertainty,
                regime=signal.regime,
                liquidity_score=signal.liquidity_score,
                health_status=signal.health_status.value,
                lifecycle_status=signal.lifecycle_status.value,
                reason_codes=signal.reason_codes,
                payload=signal.metadata,
            )
        )
        self.session.flush()

    def _record_ensemble(self, ensemble, trace_id: str, exclusions: dict[str, str]) -> None:
        self.session.add(
            EnsembleDecisionRecord(
                ensemble_decision_id=ensemble.ensemble_decision_id,
                decision_trace_id=trace_id,
                symbol=ensemble.symbol,
                generated_at=ensemble.generated_at,
                valid_until=ensemble.valid_until,
                combined_expected_return=ensemble.combined_expected_return,
                combined_expected_volatility=ensemble.combined_expected_volatility,
                combined_uncertainty=ensemble.combined_uncertainty,
                combined_confidence=ensemble.combined_confidence,
                current_regime=ensemble.current_regime,
                supporting_signals=ensemble.supporting_signals,
                conflicting_signals=ensemble.conflicting_signals,
                signal_weights=ensemble.signal_weights,
                correlation_penalty=ensemble.correlation_penalty,
                transaction_cost_penalty=ensemble.transaction_cost_penalty,
                regime_penalty=ensemble.regime_penalty,
                external_context_adjustment=ensemble.external_context_adjustment,
                decision_status=ensemble.decision_status.value,
                reason_codes=ensemble.reason_codes,
                payload={"exclusions": exclusions},
            )
        )
        self.session.flush()
        for signal_id, weight in ensemble.signal_weights.items():
            self.session.add(
                EnsembleSignalWeight(
                    ensemble_decision_id=ensemble.ensemble_decision_id,
                    signal_id=signal_id,
                    weight=weight,
                    exclusion_reason=exclusions.get(signal_id),
                )
            )
        self.session.flush()

    def _record_target(self, target, trace_id: str, account_id: str) -> None:
        self.session.add(
            PortfolioTargetRecord(
                portfolio_target_id=target.portfolio_target_id,
                decision_trace_id=trace_id,
                paper_account_id=account_id,
                symbol=target.symbol,
                current_exposure=target.current_exposure,
                requested_target_exposure=target.requested_target_exposure,
                requested_delta=target.requested_delta,
                expected_return=target.expected_return,
                expected_risk=target.expected_risk,
                risk_contribution=target.risk_contribution,
                urgency=target.urgency,
                source_ensemble_decision_id=target.source_ensemble_decision_id,
                created_at=target.created_at,
            )
        )
        self.session.flush()

    def _portfolio_context(self, account_id: str) -> PortfolioContext:
        account = self._latest_account(account_id)
        positions = list(self.session.scalars(select(Position).where(Position.status == "OPEN", Position.paper_account_id == account_id)))
        exposures: dict[str, float] = {}
        for position in positions:
            notional = (position.quantity * (position.current_price or position.entry_price)) if position.quantity else position.notional
            signed = abs(notional) / max(account.equity, 1e-9)
            exposures[position.symbol] = signed if position.side.upper() == "LONG" else -signed
        return PortfolioContext(equity=account.equity, exposures=exposures)

    def _latest_account(self, account_id: str) -> AccountEquity:
        row = self.session.scalar(
            select(AccountEquity).where(AccountEquity.paper_account_id == account_id).order_by(desc(AccountEquity.timestamp)).limit(1)
        )
        if row is None:
            sandbox = self.session.scalar(
                select(PaperSandboxAccount)
                .where(PaperSandboxAccount.account_id == account_id)
                .limit(1)
            )
            try:
                starting_balance = float(sandbox.starting_balance) if sandbox is not None else float(settings.paper_start_balance)
            except (TypeError, ValueError):
                starting_balance = 0.0
            if not math.isfinite(starting_balance) or starting_balance <= 0:
                starting_balance = 0.0
            row = AccountEquity(
                paper_account_id=account_id,
                cash_balance=starting_balance,
                equity=starting_balance,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                drawdown=0.0,
            )
            self.session.add(row)
            self.session.flush()
        return row

    @staticmethod
    def _mark_price(snapshot, feature) -> float:
        try:
            value = float(snapshot.values.get("last_close") or feature.payload.get("values", {}).get("last_close") or 0.0)
            return value if value > 0 else 0.0
        except (TypeError, ValueError, AttributeError):
            return 0.0

    def _latest_market(self, candles: list[Candle], snapshot, feature) -> tuple[float, datetime]:
        """Return the price and actual source observation time for risk gating.

        ``Feature.available_to_model_time`` is deliberately not a market timestamp:
        it means the derived feature was available, which may be much later than the
        underlying candle.  With no source candle the epoch sentinel makes the
        independent stale-feed gate reject an exposure increase.
        """
        if candles:
            latest = max(candles, key=lambda row: row.close_time or row.open_time)
            observed = latest.close_time or latest.open_time
            if observed.tzinfo is None:
                observed = observed.replace(tzinfo=timezone.utc)
            else:
                observed = observed.astimezone(timezone.utc)
            try:
                price = float(latest.close)
            except (TypeError, ValueError):
                price = self._mark_price(snapshot, feature)
            return price, observed
        return self._mark_price(snapshot, feature), datetime.fromtimestamp(0, tz=timezone.utc)

    @staticmethod
    def _bounded_external_adjustment(snapshot) -> float:
        # External AI is optional. It can nudge a base ensemble only if a typed caller
        # recorded a bounded numeric direction/importance value in the feature layer.
        context = snapshot.external_context
        if not context.get("external_ai_available"):
            return 0.0
        try:
            confidence = float(context.get("external_ai_confidence") or 0.0)
            direction = float(snapshot.values.get("external_ai_direction_score") or 0.0)
            return max(min(direction * confidence * 0.001, settings.v2_external_context_max_adjustment), -settings.v2_external_context_max_adjustment)
        except (TypeError, ValueError):
            return 0.0

    def _timeline(self, trace_id: str, stage: str, status: str, reasons: list[str] | tuple[str, ...], payload: dict[str, Any]) -> None:
        self.session.add(
            DecisionTimelineEvent(
                decision_trace_id=trace_id,
                stage=stage,
                status=status,
                reason_codes=list(reasons),
                payload=payload,
            )
        )
        self.session.flush()

    def _record_legacy_decision(self, feature, ensemble, target, risk, execution, trace_id: str) -> AiDecision:
        action = "BUY" if target.requested_delta > 0 else "SELL" if target.requested_delta < 0 else "HOLD"
        row = AiDecision(
            symbol=feature.symbol,
            strategy_name="anata-v2-narrow-baseline",
            source_name="v2-pipeline",
            feature_id=feature.id,
            feature_schema_version=feature.schema_version,
            action=action,
            confidence=ensemble.combined_confidence,
            reason="; ".join(ensemble.reason_codes + risk.triggered_limits + risk.rejection_reasons),
            execution_status=execution.result.status,
            execution_message=execution.result.message,
            trade_id=execution.result.trade_id,
            raw={
                "decision_trace_id": trace_id,
                "ensemble_decision_id": ensemble.ensemble_decision_id,
                "portfolio_target_id": target.portfolio_target_id,
                "risk_decision_id": risk.risk_decision_id,
                "no_model_execution_plan": True,
            },
            result={"requested_exposure": target.requested_target_exposure, "approved_exposure": risk.approved_exposure},
        )
        self.session.add(row)
        self.session.flush()
        return row
