from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Candle, ExperienceRecord, NewsArticle, NewsSentiment


def _dt(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def timeframe_seconds(interval: str) -> int:
    unit = interval[-1:]
    try:
        amount = int(interval[:-1])
    except ValueError:
        return 60
    if unit == "m":
        return amount * 60
    if unit == "h":
        return amount * 3600
    if unit == "d":
        return amount * 86400
    return 60


def market_snapshot(session: Session, collector_state: dict[str, Any] | None = None) -> dict[str, Any]:
    symbols = settings.binance_symbols
    latest_by_symbol = []
    latest_time: datetime | None = None
    for symbol in symbols:
        candle = session.scalar(
            select(Candle)
            .where(Candle.symbol == symbol.upper())
            .order_by(desc(Candle.open_time))
            .limit(1)
        )
        if candle:
            candle_time = _aware(candle.open_time)
            latest_time = max(latest_time, candle_time) if latest_time and candle_time else candle_time
        latest_by_symbol.append(
            {
                "symbol": symbol.upper(),
                "price": candle.close if candle else None,
                "open_time": _dt(candle.open_time if candle else None),
                "close_time": _dt(candle.close_time if candle else None),
                "is_closed": candle.is_closed if candle else None,
                "source_name": candle.source_name if candle else None,
            }
        )

    candle_count = session.scalar(select(func.count(Candle.id))) or 0
    age_seconds = None
    if latest_time:
        age_seconds = (datetime.now(timezone.utc) - _aware(latest_time)).total_seconds()
    stale_after = max(timeframe_seconds(settings.paper_trade_timeframe) * 3, 300)
    return {
        "collector": collector_state or {},
        "candle_count": candle_count,
        "latest_candle_time": _dt(latest_time),
        "latest_by_symbol": latest_by_symbol,
        "latest_prices": {row["symbol"]: row["price"] for row in latest_by_symbol},
        "stale": latest_time is None or (age_seconds is not None and age_seconds > stale_after),
        "age_seconds": age_seconds,
        "stale_after_seconds": stale_after,
        "store_live_candle_updates": settings.store_live_candle_updates,
        "closed_candles_only": not settings.store_live_candle_updates,
    }


def latest_candles(session: Session, limit: int = 50) -> list[dict[str, Any]]:
    rows = session.scalars(select(Candle).order_by(desc(Candle.open_time)).limit(limit)).all()
    return [
        {
            "id": row.id,
            "symbol": row.symbol,
            "interval": row.interval,
            "open_time": _dt(row.open_time),
            "close_time": _dt(row.close_time),
            "open": row.open,
            "high": row.high,
            "low": row.low,
            "close": row.close,
            "volume": row.volume,
            "is_closed": row.is_closed,
            "source_name": row.source_name,
        }
        for row in rows
    ]


def news_snapshot(session: Session, collector_state: dict[str, Any] | None = None) -> dict[str, Any]:
    latest = session.scalar(select(NewsArticle).order_by(desc(NewsArticle.published_at)).limit(1))
    news_count = session.scalar(select(func.count(NewsArticle.id))) or 0
    sentiment_count = session.scalar(select(func.count(NewsSentiment.id))) or 0
    experience_count = session.scalar(select(func.count(ExperienceRecord.id))) or 0
    age_seconds = None
    if latest and latest.published_at:
        age_seconds = (datetime.now(timezone.utc) - _aware(latest.published_at)).total_seconds()
    missing_key = not bool(settings.news_api_key)
    warning = "NEWS_API_KEY missing or placeholder" if missing_key else None
    return {
        "collector": collector_state or {},
        "news_count": news_count,
        "sentiment_count": sentiment_count,
        "experience_count": experience_count,
        "latest_news_time": _dt(latest.published_at if latest else None),
        "latest_title": latest.title if latest else None,
        "latest_source": latest.source if latest else None,
        "stale": latest is None or (age_seconds is not None and age_seconds > 7200),
        "age_seconds": age_seconds,
        "news_api_key_configured": bool(settings.news_api_key),
        "warning": warning,
        "mock_fallback_enabled": settings.news_mock_fallback_enabled,
    }


def latest_news(session: Session, limit: int = 25) -> list[dict[str, Any]]:
    rows = session.execute(
        select(NewsArticle, NewsSentiment)
        .outerjoin(NewsSentiment, NewsSentiment.article_id == NewsArticle.id)
        .order_by(desc(NewsArticle.published_at))
        .limit(limit)
    ).all()
    return [
        {
            "id": article.id,
            "title": article.title,
            "source": article.source,
            "url": article.url,
            "published_at": _dt(article.published_at),
            "sentiment_score": sentiment.sentiment_score if sentiment else None,
            "risk_score": sentiment.risk_score if sentiment else None,
            "affected_symbols": sentiment.affected_symbols if sentiment else [],
        }
        for article, sentiment in rows
    ]
