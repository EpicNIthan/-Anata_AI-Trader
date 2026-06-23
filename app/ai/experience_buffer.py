from __future__ import annotations

from datetime import datetime, timedelta, timezone
from statistics import mean
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.db.models import (
    AiDecision,
    Candle,
    ExperienceRecord,
    ExternalDataEvent,
    Feature,
    NewsArticle,
    NewsSentiment,
    PaperTrade,
)


def latest_market_state(session: Session, symbol: str) -> dict[str, Any]:
    candle = session.scalar(
        select(Candle)
        .where(Candle.symbol == symbol.upper())
        .order_by(desc(Candle.open_time))
        .limit(1)
    )
    future_events = list(
        session.scalars(
            select(ExternalDataEvent)
            .where((ExternalDataEvent.symbol == symbol.upper()) | (ExternalDataEvent.symbol.is_(None)))
            .order_by(desc(ExternalDataEvent.event_time))
            .limit(20)
        )
    )
    return {
        "symbol": symbol.upper(),
        "latest_candle": {
            "open_time": candle.open_time.isoformat() if candle else None,
            "interval": candle.interval if candle else None,
            "open": candle.open if candle else None,
            "high": candle.high if candle else None,
            "low": candle.low if candle else None,
            "close": candle.close if candle else None,
            "volume": candle.volume if candle else None,
            "source_name": candle.source_name if candle else None,
        },
        "external_events": [
            {
                "source_name": event.source_name,
                "data_type": event.data_type,
                "symbol": event.symbol,
                "event_time": event.event_time.isoformat(),
                "numeric_value": event.numeric_value,
                "payload": event.payload,
            }
            for event in future_events
        ],
    }


def latest_news_state(session: Session, symbol: str) -> dict[str, Any]:
    since = datetime.now(timezone.utc) - timedelta(hours=48)
    rows = session.execute(
        select(NewsSentiment, NewsArticle)
        .join(NewsArticle, NewsArticle.id == NewsSentiment.article_id)
        .where(NewsSentiment.created_at >= since)
        .order_by(desc(NewsSentiment.created_at))
        .limit(100)
    ).all()
    relevant: list[tuple[NewsSentiment, NewsArticle]] = []
    for sentiment, article in rows:
        affected = sentiment.affected_symbols or []
        if not affected or symbol.upper() in affected:
            relevant.append((sentiment, article))
    sentiment_values = [item[0].sentiment_score for item in relevant]
    risk_values = [item[0].risk_score for item in relevant]
    return {
        "lookback_hours": 48,
        "article_count": len(relevant),
        "avg_sentiment_score": mean(sentiment_values) if sentiment_values else 0.0,
        "avg_risk_score": mean(risk_values) if risk_values else 0.0,
        "latest_articles": [
            {
                "title": article.title,
                "source": article.source,
                "published_at": article.published_at.isoformat() if article.published_at else None,
                "sentiment_score": sentiment.sentiment_score,
                "risk_score": sentiment.risk_score,
                "affected_symbols": sentiment.affected_symbols or [],
            }
            for sentiment, article in relevant[:10]
        ],
    }


def reward_from_result(session: Session, trade_id: int | None, execution_status: str) -> float:
    if trade_id is None:
        return 0.0
    trade = session.get(PaperTrade, trade_id)
    if trade is None:
        return 0.0
    if trade.action == "SELL":
        return float(trade.realized_pnl)
    if execution_status == "FILLED":
        return 0.0
    return 0.0


def record_experience(
    session: Session,
    *,
    decision: AiDecision,
    feature: Feature | None,
    execution_result: dict[str, Any],
) -> ExperienceRecord:
    symbol = decision.symbol.upper()
    market_state = latest_market_state(session, symbol)
    news_state = latest_news_state(session, symbol)
    reward = reward_from_result(session, decision.trade_id, decision.execution_status)
    feature_payload = feature.payload if feature else None

    decision.market_state = market_state
    decision.news_state = news_state
    decision.result = execution_result
    decision.reward = reward
    decision.raw_payload = decision.raw

    record = ExperienceRecord(
        ai_decision_id=decision.id,
        feature_id=feature.id if feature else decision.feature_id,
        model_version_id=decision.model_version_id,
        symbol=symbol,
        feature_schema_version=decision.feature_schema_version or (feature.schema_version if feature else None),
        market_state=market_state,
        news_state=news_state,
        feature_payload=feature_payload,
        action=decision.action,
        confidence=decision.confidence,
        result=execution_result,
        reward=reward,
        raw_payload={
            "decision": decision.raw,
            "execution": execution_result,
        },
    )
    session.add(record)
    return record

