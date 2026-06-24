from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from app.db.models import Feature
from app.features.schema import values_from_feature


@dataclass(frozen=True)
class StrategyDecision:
    action: str
    confidence: float
    reason: str
    stop_loss: float | None = None
    take_profit: float | None = None

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class RuleBasedStrategy:
    name = "rule-based-v1"

    def decide(self, feature: Feature | dict[str, Any]) -> StrategyDecision:
        data = self._feature_to_dict(feature)
        trend = data["trend"]
        price_change = data["price_change"]
        sentiment = data["sentiment_score"]
        risk = data["risk_score"]
        volatility = data["volatility"]
        last_close = data.get("last_close")
        trader_crowd_score = float(data.get("trader_crowd_score") or 0.0)
        crowd_risk_score = float(data.get("crowd_risk_score") or 0.0)
        taker_buy_pressure = float(data.get("taker_buy_pressure") or 0.0)

        flow_score = trader_crowd_score * 0.25 + ((taker_buy_pressure - 0.5) * 0.35 if taker_buy_pressure else 0.0)
        score = price_change * 4.0 + sentiment * 0.35 + flow_score - risk * 0.45 - crowd_risk_score * 0.20
        confidence = max(0.0, min(0.95, 0.50 + abs(score) + min(volatility * 4.0, 0.15)))

        if risk >= 0.80:
            return StrategyDecision(
                action="HOLD",
                confidence=max(confidence, 0.60),
                reason="Risk score is elevated; waiting for clearer conditions.",
            )
        if crowd_risk_score >= 0.90 and abs(score) < 0.30:
            return StrategyDecision(
                action="HOLD",
                confidence=max(confidence, 0.60),
                reason="Trader-flow data looks crowded; waiting for cleaner risk/reward.",
            )
        if trend == "up" and score > 0.05:
            return StrategyDecision(
                action="BUY",
                confidence=confidence,
                reason="Price trend, news, and trader-flow features are constructive.",
                stop_loss=self._price_level(last_close, -0.02),
                take_profit=self._price_level(last_close, 0.04),
            )
        if trend == "down" and score < -0.05:
            return StrategyDecision(
                action="SELL",
                confidence=confidence,
                reason="Price trend weakened and model score is negative.",
                stop_loss=self._price_level(last_close, 0.02),
                take_profit=self._price_level(last_close, -0.03),
            )
        return StrategyDecision(
            action="HOLD",
            confidence=max(0.50, 1.0 - abs(score)),
            reason="No strong edge from current feature set.",
        )

    def _feature_to_dict(self, feature: Feature | dict[str, Any]) -> dict[str, Any]:
        return values_from_feature(
            feature,
            [
                "price_change",
                "sentiment_score",
                "risk_score",
                "volatility",
                "trader_crowd_score",
                "crowd_risk_score",
                "taker_buy_pressure",
            ],
        )

    def _price_level(self, price: float | None, offset: float) -> float | None:
        if not price:
            return None
        return round(price * (1.0 + offset), 8)


def decide_trade(feature: Feature | dict[str, Any]) -> StrategyDecision:
    return RuleBasedStrategy().decide(feature)
