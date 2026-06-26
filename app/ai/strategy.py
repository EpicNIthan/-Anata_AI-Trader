from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from app.config import settings
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
    name = "indicator-style-v2"

    def decide(self, feature: Feature | dict[str, Any]) -> StrategyDecision:
        data = self._feature_to_dict(feature)
        last_close = self._as_float(data.get("last_close"))
        candle_return_1m = self._as_float(data.get("candle_return_1m"))
        candle_return_5m = self._as_float(data.get("candle_return_5m"))
        legacy_price_change = self._as_float(data.get("price_change"))
        price_change = candle_return_5m or legacy_price_change
        volume_change = self._as_float(data.get("volume_change"))
        volatility = abs(self._as_float(data.get("volatility")))
        trend_score = self._as_float(data.get("trend_score"))
        sentiment = self._as_float(data.get("sentiment_score"))
        sentiment_confidence = self._as_float(data.get("sentiment_confidence"))
        risk = self._as_float(data.get("risk_score"))
        impact = self._as_float(data.get("impact_score"))
        trader_crowd_score = self._as_float(data.get("trader_crowd_score"))
        crowd_risk_score = self._as_float(data.get("crowd_risk_score"))
        taker_buy_pressure = self._as_float(data.get("taker_buy_pressure"))
        liquidation_imbalance = self._as_float(data.get("liquidation_imbalance_5m"))
        market_regime = self._as_float(data.get("market_regime_score"))
        candles_used = self._as_float(data.get("candles_used"))

        if candles_used and candles_used < 20:
            return StrategyDecision(
                action="HOLD",
                confidence=0.60,
                reason=f"Waiting for more candle history before paper entry ({candles_used:.0f}/20 candles).",
            )

        round_trip_fee = settings.paper_fee_rate * 2.0
        required_edge = round_trip_fee + settings.strategy_min_edge_after_fees
        directional_move = abs(candle_return_5m) + abs(candle_return_1m) * 0.45
        news_edge = abs(sentiment) * max(sentiment_confidence, 0.25) * 0.0025
        flow_edge = abs(trader_crowd_score) * 0.0015 + abs(taker_buy_pressure - 0.5) * 0.002 if taker_buy_pressure else 0.0
        expected_edge = directional_move + news_edge + flow_edge

        if risk >= 0.85:
            return StrategyDecision(
                action="HOLD",
                confidence=0.70,
                reason="High risk score; waiting instead of opening noisy paper exposure.",
            )
        if crowd_risk_score >= 0.90 and abs(trend_score) < 0.35:
            return StrategyDecision(
                action="HOLD",
                confidence=0.65,
                reason="Trader-flow looks crowded without a strong trend confirmation.",
            )
        if expected_edge < required_edge:
            return StrategyDecision(
                action="HOLD",
                confidence=0.60,
                reason=(
                    "Indicator edge is too small after fees; waiting for a cleaner setup "
                    f"({expected_edge:.4%} edge vs {required_edge:.4%} required)."
                ),
            )

        flow_bias = (taker_buy_pressure - 0.5) * 0.85 if taker_buy_pressure else 0.0
        momentum_bias = (candle_return_5m * 80.0) + (candle_return_1m * 35.0) + (trend_score * 0.70)
        news_bias = sentiment * max(sentiment_confidence, 0.35) * 0.35
        market_bias = market_regime * 0.15
        long_score = momentum_bias + news_bias + flow_bias + trader_crowd_score * 0.20 + market_bias
        short_score = -momentum_bias - news_bias - flow_bias - trader_crowd_score * 0.20 - market_bias

        # Liquidation imbalance can hint at squeeze risk: positive supports longs, negative supports shorts.
        long_score += liquidation_imbalance * 0.12
        short_score -= liquidation_imbalance * 0.12

        # News impact should matter, but not overpower price. Bad news increases short confidence; good news increases long confidence.
        long_score += max(sentiment, 0.0) * impact * 0.20
        short_score += max(-sentiment, 0.0) * impact * 0.20
        long_score -= risk * 0.18
        short_score -= risk * 0.10

        threshold = 0.22 if volatility >= 0.0008 else 0.30
        if long_score > threshold and long_score > short_score + 0.08:
            confidence = self._confidence(long_score, volatility, expected_edge, required_edge)
            stop_pct, take_pct = self._adaptive_risk_levels(volatility, long=True)
            return StrategyDecision(
                action="BUY",
                confidence=confidence,
                reason=(
                    "Indicator-style long setup: 1m/5m momentum, trend, news, and flow are aligned "
                    f"(long score {long_score:.3f}, edge {expected_edge:.4%})."
                ),
                stop_loss=self._price_level(last_close, -stop_pct),
                take_profit=self._price_level(last_close, take_pct),
            )
        if short_score > threshold and short_score > long_score + 0.08:
            confidence = self._confidence(short_score, volatility, expected_edge, required_edge)
            stop_pct, take_pct = self._adaptive_risk_levels(volatility, long=False)
            return StrategyDecision(
                action="SELL",
                confidence=confidence,
                reason=(
                    "Indicator-style short setup: downside momentum, trend, news, and flow are aligned "
                    f"(short score {short_score:.3f}, edge {expected_edge:.4%})."
                ),
                stop_loss=self._price_level(last_close, stop_pct),
                take_profit=self._price_level(last_close, -take_pct),
            )
        return StrategyDecision(
            action="HOLD",
            confidence=0.55,
            reason=(
                "No clean indicator alignment yet; waiting instead of creating flat zero-edge trades "
                f"(long {long_score:.3f}, short {short_score:.3f})."
            ),
        )

    def _feature_to_dict(self, feature: Feature | dict[str, Any]) -> dict[str, Any]:
        return values_from_feature(
            feature,
            [
                "price_change",
                "sentiment_score",
                "sentiment_confidence",
                "risk_score",
                "impact_score",
                "volatility",
                "volume_change",
                "candle_return_1m",
                "candle_return_5m",
                "trend_score",
                "trader_crowd_score",
                "crowd_risk_score",
                "taker_buy_pressure",
                "liquidation_imbalance_5m",
                "market_regime_score",
            ],
        )

    def _as_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value or default)
        except (TypeError, ValueError):
            return default

    def _confidence(self, score: float, volatility: float, expected_edge: float, required_edge: float) -> float:
        edge_bonus = max(0.0, min(0.18, (expected_edge - required_edge) * 35.0))
        volatility_bonus = max(0.0, min(0.10, volatility * 40.0))
        return max(settings.risk_min_confidence + 0.01, min(0.92, 0.55 + min(score, 1.0) * 0.22 + edge_bonus + volatility_bonus))

    def _adaptive_risk_levels(self, volatility: float, *, long: bool) -> tuple[float, float]:
        stop = max(settings.auto_default_stop_loss_pct, min(0.025, 0.006 + volatility * 8.0))
        take = max(settings.auto_default_take_profit_pct, min(0.055, stop * (2.0 if long else 1.8)))
        return stop, take

    def _price_level(self, price: float | None, offset: float) -> float | None:
        if not price:
            return None
        return round(price * (1.0 + offset), 8)


def decide_trade(feature: Feature | dict[str, Any]) -> StrategyDecision:
    return RuleBasedStrategy().decide(feature)
