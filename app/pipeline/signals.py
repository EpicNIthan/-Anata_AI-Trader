"""Convert standardized forecasts into lifecycle-managed trading signals."""

from __future__ import annotations

from datetime import timedelta
from typing import Iterable

from app.pipeline.domain import (
    Direction,
    HealthStatus,
    ModelPrediction,
    SignalLifecycle,
    TradingSignal,
    utc_now,
)
from app.pipeline.narrow_models import CostEstimate


class SignalFactory:
    """Deterministic forecast-to-signal adapter with no execution dependency."""

    def __init__(self, *, minimum_edge: float = 0.0005, ttl_seconds: int = 300) -> None:
        self.minimum_edge = max(float(minimum_edge), 0.0)
        self.ttl_seconds = max(int(ttl_seconds), 1)

    def from_prediction(
        self,
        prediction: ModelPrediction,
        *,
        cost: CostEstimate | float = 0.0,
        liquidity_score: float | None = None,
        lifecycle_status: SignalLifecycle = SignalLifecycle.PAPER,
        health_status: HealthStatus = HealthStatus.HEALTHY,
    ) -> TradingSignal:
        expected_cost = cost.total_cost if isinstance(cost, CostEstimate) else max(float(cost), 0.0)
        net_expected_return = prediction.expected_return - expected_cost if prediction.expected_return >= 0 else prediction.expected_return + expected_cost
        if abs(net_expected_return) < self.minimum_edge:
            direction = Direction.FLAT
            reason_codes = ["NET_EDGE_BELOW_THRESHOLD"]
        elif net_expected_return > 0:
            direction = Direction.LONG
            reason_codes = ["POSITIVE_NET_EXPECTED_RETURN"]
        else:
            direction = Direction.SHORT
            reason_codes = ["NEGATIVE_NET_EXPECTED_RETURN"]
        if prediction.external_context_available:
            reason_codes.append("EXTERNAL_CONTEXT_AVAILABLE")
        else:
            reason_codes.append("BASE_ENSEMBLE_ONLY")
        if health_status != HealthStatus.HEALTHY:
            reason_codes.append(f"MODEL_HEALTH_{health_status.value}")
        now = utc_now()
        valid_until = min(prediction.expires_at, now + timedelta(seconds=self.ttl_seconds))
        if valid_until <= now:
            valid_until = now + timedelta(seconds=1)
        confidence = prediction.confidence * prediction.calibration_score
        return TradingSignal(
            prediction_id=prediction.prediction_id,
            signal_family=prediction.model_family,
            symbol=prediction.symbol,
            generated_at=now,
            valid_until=valid_until,
            direction=direction,
            strength=min(abs(net_expected_return) / max(self.minimum_edge * 4.0, 1e-9), 1.0),
            expected_return=prediction.expected_return,
            expected_cost=expected_cost,
            net_expected_return=net_expected_return,
            confidence=max(min(confidence, 1.0), 0.0),
            uncertainty=prediction.uncertainty,
            regime=prediction.regime,
            liquidity_score=liquidity_score if liquidity_score is not None else (cost.fill_probability if isinstance(cost, CostEstimate) else 0.5),
            health_status=health_status,
            lifecycle_status=lifecycle_status,
            reason_codes=reason_codes,
            metadata={
                "model_id": prediction.model_id,
                "model_version": prediction.model_version,
                "calibration_score": prediction.calibration_score,
                "expected_volatility": prediction.expected_volatility,
                "external_context_available": prediction.external_context_available,
            },
        )

    def valid_signals(self, signals: Iterable[TradingSignal]) -> list[TradingSignal]:
        now = utc_now()
        return [
            signal
            for signal in signals
            if signal.is_valid_at(now)
            and signal.direction != Direction.FLAT
            and signal.health_status not in {HealthStatus.SUSPENDED, HealthStatus.RETIRED}
            and signal.lifecycle_status not in {SignalLifecycle.SUSPENDED, SignalLifecycle.RETIRED, SignalLifecycle.SHADOW}
        ]
