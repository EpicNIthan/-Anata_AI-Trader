from __future__ import annotations

from datetime import datetime, timedelta, timezone
from statistics import mean, pstdev
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Candle, Feature, NewsArticle, NewsSentiment
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


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _recency_weight(value: datetime | None, now: datetime, horizon_hours: float = 48.0) -> float:
    timestamp = _aware(value)
    if timestamp is None:
        return 0.25
    age_hours = max((now - timestamp).total_seconds() / 3600.0, 0.0)
    return _clamp(1.0 - (age_hours / horizon_hours), 0.0, 1.0)


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
        candle_return_1m = _safe_pct_change(closes[-1], closes[-2]) if len(closes) >= 2 else 0.0
        candle_return_5m = _safe_pct_change(closes[-1], closes[-6]) if len(closes) >= 6 else price_change
        volatility = pstdev(returns) if len(returns) > 1 else 0.0

        if len(volumes) >= 4:
            midpoint = len(volumes) // 2
            earlier_volume = mean(volumes[:midpoint])
            recent_volume = mean(volumes[midpoint:])
            volume_change = _safe_pct_change(recent_volume, earlier_volume)
        else:
            volume_change = 0.0

        news_features = self._recent_news_features(normalized_symbol, now=datetime.now(timezone.utc))
        sentiment_score = news_features["sentiment_score"]
        sentiment_confidence = news_features["sentiment_confidence"]
        risk_score = news_features["risk_score"]
        impact_score = news_features["impact_score"]
        recency_weight = news_features["recency_weight"]
        sentiment_count = int(news_features["sentiment_articles_used"])
        trend = _trend_from_change(price_change, volatility)
        trend_score = _clamp(candle_return_5m / max(volatility * 3.0, 0.001), -1.0, 1.0)
        now = datetime.now(timezone.utc)

        values: dict[str, Any] = {
            "price_change": price_change,
            "volume_change": volume_change,
            "volatility": volatility,
            "trend": trend,
            "trend_score": trend_score,
            "sentiment_score": sentiment_score,
            "sentiment_confidence": sentiment_confidence,
            "risk_score": risk_score,
            "impact_score": impact_score,
            "recency_weight": recency_weight,
            "btc_related": news_features["btc_related"],
            "eth_related": news_features["eth_related"],
            "macro_related": news_features["macro_related"],
            "candle_return_1m": candle_return_1m,
            "candle_return_5m": candle_return_5m,
            "last_close": closes[-1] if closes else None,
            "candles_used": len(candles),
            "sentiment_articles_used": sentiment_count,
        }
        inspector_vector = {
            key: values[key]
            for key in (
                "sentiment_score",
                "sentiment_confidence",
                "risk_score",
                "impact_score",
                "recency_weight",
                "btc_related",
                "eth_related",
                "macro_related",
                "candle_return_1m",
                "candle_return_5m",
                "volatility",
                "volume_change",
                "trend_score",
            )
        }
        values["final_ai_input"] = {
            "schema_version": CURRENT_FEATURE_SCHEMA_VERSION,
            "symbol": normalized_symbol,
            "timeframe": candle_interval,
            "vector": inspector_vector,
            "strategy_input": {
                "price_change": price_change,
                "sentiment_score": sentiment_score,
                "risk_score": risk_score,
                "volatility": volatility,
                "trend": trend,
            },
        }
        payload = feature_payload(
            schema_version=CURRENT_FEATURE_SCHEMA_VERSION,
            values=values,
            metadata={
                "lookback": lookback,
                "interval": candle_interval,
                "returns_used": len(returns),
                "missing_future_features_default": "0/null",
                "news_context": news_features["news_context"],
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

    def _recent_news_features(self, symbol: str, now: datetime) -> dict[str, Any]:
        since = datetime.now(timezone.utc) - timedelta(hours=48)
        rows = list(
            self.session.execute(
                select(NewsSentiment, NewsArticle)
                .join(NewsArticle, NewsArticle.id == NewsSentiment.article_id)
                .where(NewsSentiment.created_at >= since)
                .order_by(desc(NewsSentiment.created_at))
                .limit(200)
            )
        )
        relevant: list[tuple[NewsSentiment, NewsArticle, float]] = []
        btc_related = 0.0
        eth_related = 0.0
        macro_related = 0.0
        for sentiment, article in rows:
            affected_symbols = sentiment.affected_symbols or []
            topics = sentiment.topics or []
            text = f"{article.title or ''} {article.raw_text or ''}".lower()
            article_btc_related = "BTCUSDT" in affected_symbols or "bitcoin" in text or " btc" in f" {text}"
            article_eth_related = "ETHUSDT" in affected_symbols or "ethereum" in text or " eth" in f" {text}"
            article_macro_related = "macro" in topics or any(
                term in text for term in ("fed", "federal reserve", "inflation", "interest rate", "rates", "war", "sec")
            )
            btc_related = max(btc_related, 1.0 if article_btc_related else 0.0)
            eth_related = max(eth_related, 1.0 if article_eth_related else 0.0)
            macro_related = max(macro_related, 1.0 if article_macro_related else 0.0)
            if not affected_symbols or symbol in affected_symbols or article_macro_related:
                relevant.append((sentiment, article, _recency_weight(article.published_at or sentiment.created_at, now)))
        if not relevant:
            return {
                "sentiment_score": 0.0,
                "sentiment_confidence": 0.0,
                "risk_score": 0.0,
                "impact_score": 0.0,
                "recency_weight": 0.0,
                "btc_related": btc_related,
                "eth_related": eth_related,
                "macro_related": macro_related,
                "sentiment_articles_used": 0,
                "news_context": [],
            }

        weights = [max(weight, 0.01) * (sentiment.confidence if sentiment.confidence is not None else 0.5) for sentiment, _, weight in relevant]
        total_weight = sum(weights) or 1.0
        sentiment_score = sum(sentiment.sentiment_score * weight for (sentiment, _, _), weight in zip(relevant, weights)) / total_weight
        risk_score = sum(sentiment.risk_score * weight for (sentiment, _, _), weight in zip(relevant, weights)) / total_weight
        recency = mean(weight for _, _, weight in relevant)
        confidence = sum((sentiment.confidence if sentiment.confidence is not None else 0.5) * weight for (sentiment, _, _), weight in zip(relevant, weights)) / total_weight
        impact_score = _clamp((abs(sentiment_score) * confidence * 0.45) + (risk_score * 0.40) + (recency * 0.15), 0.0, 1.0)

        return {
            "sentiment_score": _clamp(sentiment_score, -1.0, 1.0),
            "sentiment_confidence": _clamp(confidence, 0.0, 1.0),
            "risk_score": _clamp(risk_score, 0.0, 1.0),
            "impact_score": impact_score,
            "recency_weight": _clamp(recency, 0.0, 1.0),
            "btc_related": btc_related,
            "eth_related": eth_related,
            "macro_related": macro_related,
            "sentiment_articles_used": len(relevant),
            "news_context": [
                {
                    "title": article.title,
                    "text": article.raw_text or article.title,
                    "source": article.source,
                    "provider": article.source_name,
                    "published_at": article.published_at.isoformat() if article.published_at else None,
                    "sentiment_score": sentiment.sentiment_score,
                    "sentiment_confidence": sentiment.confidence,
                    "risk_score": sentiment.risk_score,
                    "topics": sentiment.topics or [],
                    "affected_symbols": sentiment.affected_symbols or [],
                    "url": article.url,
                }
                for sentiment, article, _ in relevant[:5]
            ],
        }
