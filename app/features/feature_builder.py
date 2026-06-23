from __future__ import annotations

from datetime import datetime, timedelta, timezone
from statistics import mean, pstdev
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Candle, Feature, NewsSentiment
from app.features.schema import CURRENT_FEATURE_SCHEMA_VERSION, feature_payload


def _safe_pct_change(current: float, previous: float) -> float:
    if previous == 0:
        return 0.0
    return (current - previous) / previous


def _trend_from_change(price_change: float, volatility: float) -> str:
    threshold = max(volatility * 0.5, 0.001)
    if price_change > threshold:
        return "up"
    if price_change < -threshold:
        return "down"
    return "sideways"


class FeatureBuilder:
    def __init__(self, session: Session) -> None:
        self.session = session

    def build_for_symbol(
        self,
        symbol: str,
        lookback: int = 60,
        store: bool = True,
        interval: str | None = None,
    ) -> Feature:
        normalized_symbol = symbol.upper()
        candle_interval = interval or settings.paper_trade_timeframe
        query = (
            select(Candle)
            .where(Candle.symbol == normalized_symbol)
            .order_by(desc(Candle.open_time))
            .limit(lookback)
        )
        if candle_interval:
            query = (
                select(Candle)
                .where(Candle.symbol == normalized_symbol, Candle.interval == candle_interval)
                .order_by(desc(Candle.open_time))
                .limit(lookback)
            )
        candles = list(
            self.session.scalars(query)
        )
        candles.reverse()

        closes = [candle.close for candle in candles]
        volumes = [candle.volume for candle in candles]
        returns = [_safe_pct_change(closes[i], closes[i - 1]) for i in range(1, len(closes))]
        price_change = _safe_pct_change(closes[-1], closes[0]) if len(closes) >= 2 else 0.0
        volatility = pstdev(returns) if len(returns) > 1 else 0.0

        if len(volumes) >= 4:
            midpoint = len(volumes) // 2
            earlier_volume = mean(volumes[:midpoint])
            recent_volume = mean(volumes[midpoint:])
            volume_change = _safe_pct_change(recent_volume, earlier_volume)
        else:
            volume_change = 0.0

        sentiment_score, risk_score, sentiment_count = self._recent_news_scores(normalized_symbol)
        trend = _trend_from_change(price_change, volatility)
        now = datetime.now(timezone.utc)

        values: dict[str, Any] = {
            "price_change": price_change,
            "volume_change": volume_change,
            "volatility": volatility,
            "trend": trend,
            "sentiment_score": sentiment_score,
            "risk_score": risk_score,
            "last_close": closes[-1] if closes else None,
            "candles_used": len(candles),
            "sentiment_articles_used": sentiment_count,
        }
        payload = feature_payload(
            schema_version=CURRENT_FEATURE_SCHEMA_VERSION,
            values=values,
            metadata={
                "lookback": lookback,
                "interval": candle_interval,
                "returns_used": len(returns),
                "missing_future_features_default": "0/null",
            },
            sources={
                "candles": "candles",
                "news_sentiment": "news_sentiment",
            },
        )
        feature = Feature(
            symbol=normalized_symbol,
            schema_version=CURRENT_FEATURE_SCHEMA_VERSION,
            source_name="feature_builder",
            as_of=now,
            price_change=price_change,
            volume_change=volume_change,
            volatility=volatility,
            trend=trend,
            sentiment_score=sentiment_score,
            risk_score=risk_score,
            payload=payload,
            raw_payload=payload,
        )
        if store:
            self.session.add(feature)
            self.session.commit()
            self.session.refresh(feature)
        return feature

    def _recent_news_scores(self, symbol: str) -> tuple[float, float, int]:
        since = datetime.now(timezone.utc) - timedelta(hours=48)
        sentiments = list(
            self.session.scalars(
                select(NewsSentiment)
                .where(NewsSentiment.created_at >= since)
                .order_by(desc(NewsSentiment.created_at))
                .limit(200)
            )
        )
        relevant: list[NewsSentiment] = []
        for sentiment in sentiments:
            affected_symbols = sentiment.affected_symbols or []
            if not affected_symbols or symbol in affected_symbols:
                relevant.append(sentiment)
        if not relevant:
            return 0.0, 0.0, 0
        return (
            mean(item.sentiment_score for item in relevant),
            mean(item.risk_score for item in relevant),
            len(relevant),
        )
