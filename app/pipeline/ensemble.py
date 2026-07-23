"""Regime-aware, deterministic signal combination for the V2 paper baseline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable, Mapping, Protocol

from app.pipeline.domain import (
    Direction,
    EnsembleDecision,
    EnsembleStatus,
    HealthStatus,
    SignalLifecycle,
    TradingSignal,
    utc_now,
)


class EnsembleModel(Protocol):
    """Future learned meta-model contract; it receives signals, never orders."""

    def combine(self, symbol: str, signals: list[TradingSignal], **kwargs: object) -> "EnsembleResult": ...


@dataclass(frozen=True)
class EnsembleResult:
    decision: EnsembleDecision
    exclusions: dict[str, str]


def _correlation(correlations: Mapping[object, float] | None, left: str, right: str) -> float:
    if not correlations:
        return 0.0
    for key in ((left, right), (right, left), f"{left}:{right}", f"{right}:{left}"):
        value = correlations.get(key)
        if value is not None:
            try:
                return max(min(float(value), 1.0), -1.0)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


class DeterministicRegimeEnsemble:
    """Weight evidence by reliability, cost, health and incremental independence.

    It is deliberately more conservative than equal voting.  A correlated technical
    cluster gets progressively less total influence, and a degraded/suspended model
    cannot receive normal capital influence.
    """

    def __init__(
        self,
        *,
        minimum_edge: float = 0.0005,
        correlation_threshold: float = 0.70,
        external_context_bound: float = 0.10,
        ttl_seconds: int = 300,
    ) -> None:
        self.minimum_edge = max(minimum_edge, 0.0)
        self.correlation_threshold = min(max(correlation_threshold, 0.0), 1.0)
        self.external_context_bound = max(external_context_bound, 0.0)
        self.ttl_seconds = max(ttl_seconds, 1)

    def combine(
        self,
        symbol: str,
        signals: Iterable[TradingSignal],
        *,
        regime: str = "unknown",
        correlations: Mapping[object, float] | None = None,
        external_context_score: float = 0.0,
        recent_performance: Mapping[str, float] | None = None,
    ) -> EnsembleResult:
        now = utc_now()
        candidates = list(signals)
        valid: list[TradingSignal] = []
        exclusions: dict[str, str] = {}
        for signal in candidates:
            if signal.symbol != symbol.upper():
                exclusions[signal.signal_id] = "SYMBOL_MISMATCH"
            elif not signal.is_valid_at(now):
                exclusions[signal.signal_id] = "EXPIRED"
            elif signal.lifecycle_status in {SignalLifecycle.SHADOW, SignalLifecycle.SUSPENDED, SignalLifecycle.RETIRED}:
                exclusions[signal.signal_id] = f"LIFECYCLE_{signal.lifecycle_status.value}"
            elif signal.health_status in {HealthStatus.SUSPENDED, HealthStatus.RETIRED}:
                exclusions[signal.signal_id] = f"HEALTH_{signal.health_status.value}"
            elif signal.direction == Direction.FLAT:
                exclusions[signal.signal_id] = "FLAT_SIGNAL"
            else:
                valid.append(signal)

        if not valid:
            decision = EnsembleDecision(
                symbol=symbol,
                generated_at=now,
                valid_until=now + timedelta(seconds=self.ttl_seconds),
                combined_expected_return=0.0,
                combined_expected_volatility=0.0,
                combined_uncertainty=1.0,
                combined_confidence=0.0,
                current_regime=regime,
                signal_weights={item.signal_id: 0.0 for item in candidates},
                decision_status=EnsembleStatus.NEUTRAL,
                reason_codes=["NO_VALID_SIGNALS"],
            )
            return EnsembleResult(decision=decision, exclusions=exclusions)

        # Score quality before correlation penalties.  Recent OOS performance only
        # adjusts a bounded amount; it cannot reverse a signal sign on its own.
        raw_weights: dict[str, float] = {}
        ordered = sorted(valid, key=lambda row: (row.confidence * (1.0 - row.uncertainty) * row.liquidity_score), reverse=True)
        accepted: list[TradingSignal] = []
        correlation_penalties: list[float] = []
        for signal in ordered:
            health_multiplier = {
                HealthStatus.HEALTHY: 1.0,
                HealthStatus.WATCH: 0.75,
                HealthStatus.DEGRADED: 0.35,
            }.get(signal.health_status, 0.0)
            performance = 0.0
            if recent_performance:
                try:
                    performance = max(min(float(recent_performance.get(signal.signal_family, 0.0)), 0.2), -0.2)
                except (TypeError, ValueError):
                    performance = 0.0
            edge_quality = max(signal.strength, 0.05)
            base = (
                max(signal.confidence, 0.0)
                * max(1.0 - signal.uncertainty, 0.0)
                * max(signal.liquidity_score, 0.0)
                * edge_quality
            )
            base *= health_multiplier * (1.0 + performance)
            maximum_correlation = max((_correlation(correlations, signal.signal_id, prior.signal_id) for prior in accepted), default=0.0)
            # Same family is treated as correlated even if the historical estimate is
            # not yet available. This prevents RSI/MACD/momentum clones from voting as
            # independent alpha.
            if any(prior.signal_family == signal.signal_family for prior in accepted):
                maximum_correlation = max(maximum_correlation, self.correlation_threshold)
            if maximum_correlation >= self.correlation_threshold:
                penalty = min((maximum_correlation - self.correlation_threshold) / max(1.0 - self.correlation_threshold, 1e-9), 1.0)
                base *= 1.0 - (0.75 * penalty)
                correlation_penalties.append(penalty)
                if base <= 1e-10:
                    exclusions[signal.signal_id] = "CORRELATION_PENALTY"
            raw_weights[signal.signal_id] = max(base, 0.0)
            accepted.append(signal)

        total_weight = sum(raw_weights.values())
        weights = {item.signal_id: (raw_weights.get(item.signal_id, 0.0) / total_weight if total_weight else 0.0) for item in valid}
        weights.update({item.signal_id: 0.0 for item in candidates if item.signal_id not in weights})
        combined_return = sum(signal.net_expected_return * weights[signal.signal_id] for signal in valid)
        combined_volatility = sum(
            max(float(signal.metadata.get("expected_volatility", 0.0) or 0.0), 0.0) * weights[signal.signal_id]
            for signal in valid
        )
        combined_uncertainty = sum(signal.uncertainty * weights[signal.signal_id] for signal in valid)
        combined_confidence = sum(signal.confidence * weights[signal.signal_id] for signal in valid)
        cost_penalty = sum(signal.expected_cost * weights[signal.signal_id] for signal in valid)
        correlation_penalty = sum(correlation_penalties) / len(correlation_penalties) if correlation_penalties else 0.0
        regime_penalty = self._regime_penalty(regime)
        external_adjustment = max(min(external_context_score, self.external_context_bound), -self.external_context_bound)
        combined_return = combined_return * (1.0 - correlation_penalty * 0.5) * (1.0 - regime_penalty) + external_adjustment
        combined_uncertainty = min(1.0, combined_uncertainty + correlation_penalty * 0.2 + regime_penalty * 0.25)
        combined_confidence *= 1.0 - min(correlation_penalty * 0.2 + regime_penalty * 0.3, 0.6)

        supporting = [
            signal.signal_id
            for signal in valid
            if weights[signal.signal_id] > 0 and signal.net_expected_return * combined_return > 0
        ]
        conflicting = [
            signal.signal_id
            for signal in valid
            if weights[signal.signal_id] > 0 and signal.net_expected_return * combined_return < 0
        ]
        reason_codes = ["DETERMINISTIC_WEIGHTED_ENSEMBLE"]
        if correlation_penalty:
            reason_codes.append("CORRELATED_SIGNAL_WEIGHT_REDUCED")
        if cost_penalty:
            reason_codes.append("EXPECTED_COST_PENALTY_APPLIED")
        if regime_penalty:
            reason_codes.append(f"REGIME_PENALTY_{regime.upper()}")
        if external_adjustment:
            reason_codes.append("BOUNDED_EXTERNAL_CONTEXT_ADJUSTMENT")
        status = EnsembleStatus.ACTIONABLE if abs(combined_return) >= self.minimum_edge and combined_confidence > 0 else EnsembleStatus.NEUTRAL
        if status == EnsembleStatus.NEUTRAL:
            reason_codes.append("COMBINED_EDGE_BELOW_THRESHOLD")
        decision = EnsembleDecision(
            symbol=symbol,
            generated_at=now,
            valid_until=now + timedelta(seconds=self.ttl_seconds),
            combined_expected_return=combined_return,
            combined_expected_volatility=max(combined_volatility, 0.0),
            combined_uncertainty=combined_uncertainty,
            combined_confidence=max(min(combined_confidence, 1.0), 0.0),
            current_regime=regime,
            supporting_signals=supporting,
            conflicting_signals=conflicting,
            signal_weights=weights,
            correlation_penalty=correlation_penalty,
            transaction_cost_penalty=cost_penalty,
            regime_penalty=regime_penalty,
            external_context_adjustment=external_adjustment,
            decision_status=status,
            reason_codes=reason_codes,
        )
        return EnsembleResult(decision=decision, exclusions=exclusions)

    @staticmethod
    def _regime_penalty(regime: str) -> float:
        normalized = (regime or "").lower()
        if normalized in {"risk_off", "liquidity_stress", "news_shock"}:
            return 0.30
        if normalized in {"high_volatility", "crowded_market"}:
            return 0.15
        return 0.0
