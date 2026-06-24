from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Candle, ExperienceRecord, LiveCandleUpdate, NewsArticle, NewsSentiment


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
            .where(Candle.symbol == symbol.upper(), Candle.is_closed.is_(True))
            .order_by(desc(Candle.open_time))
            .limit(1)
        )
        live_update = session.scalar(
            select(LiveCandleUpdate)
            .where(LiveCandleUpdate.symbol == symbol.upper())
            .order_by(desc(LiveCandleUpdate.open_time))
            .limit(1)
        )
        display_price = candle.close if candle else None
        display_time = candle.open_time if candle else None
        display_source = candle.source_name if candle else None
        display_closed = candle.is_closed if candle else None
        if live_update and (display_time is None or live_update.open_time >= display_time):
            display_price = live_update.close
            display_time = live_update.open_time
            display_source = live_update.source_name
            display_closed = False
        if candle:
            candle_time = _aware(candle.open_time)
            latest_time = max(latest_time, candle_time) if latest_time and candle_time else candle_time
        latest_by_symbol.append(
            {
                "symbol": symbol.upper(),
                "price": display_price,
                "open_time": _dt(display_time),
                "close_time": _dt(candle.close_time if candle else None),
                "is_closed": display_closed,
                "source_name": display_source,
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
        "closed_candles_only": True,
        "live_updates_table": "live_candle_updates",
        "training_candles_table": "candles",
    }


def latest_candles(session: Session, limit: int = 50) -> list[dict[str, Any]]:
    rows = session.scalars(select(Candle).where(Candle.is_closed.is_(True)).order_by(desc(Candle.open_time)).limit(limit)).all()
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
    enabled_providers = {item.upper() for item in settings.news_providers}
    provider_config = {
        "rss": {
            "enabled": settings.rss_news_enabled and "RSS" in enabled_providers,
            "configured": bool(settings.rss_feeds),
            "role": "free crypto RSS feeds",
            "delayed": False,
            "fallback": False,
            "warning": None if settings.rss_feeds else "RSS_FEEDS missing",
        },
        "gdelt": {
            "enabled": settings.gdelt_enabled and "GDELT" in enabled_providers,
            "configured": True,
            "role": "global macro/world risk",
            "delayed": False,
            "fallback": False,
            "warning": None,
        },
        "newsapi": {
            "enabled": settings.newsapi_enabled and "NEWSAPI" in enabled_providers,
            "configured": bool(settings.news_api_key),
            "role": "delayed/free fallback",
            "delayed": True,
            "fallback": True,
            "warning": None if settings.news_api_key else "NEWS_API_KEY missing or placeholder",
        },
    }
    provider_details = ((collector_state or {}).get("details") or {}).get("providers") or {}
    providers: dict[str, Any] = {}
    for provider, base in provider_config.items():
        latest_provider = session.scalar(
            select(NewsArticle)
            .where(NewsArticle.source_name == provider)
            .order_by(desc(NewsArticle.published_at))
            .limit(1)
        )
        count_provider = session.scalar(
            select(func.count(NewsArticle.id)).where(NewsArticle.source_name == provider)
        ) or 0
        live = provider_details.get(provider, {})
        last_error = live.get("last_error")
        if last_error is None and base["enabled"]:
            last_error = base["warning"]
        status_label = "disabled"
        if base["enabled"] and not base["configured"]:
            status_label = "missing config"
        elif base["enabled"] and last_error:
            status_label = "warning" if _provider_warning(last_error) else "error"
        elif base["enabled"] and (collector_state or {}).get("running"):
            status_label = "running"
        elif base["enabled"]:
            status_label = "ready"
        providers[provider] = {
            **base,
            **live,
            "status": status_label,
            "article_count": count_provider,
            "latest_article_at": _dt(latest_provider.published_at if latest_provider else None),
            "latest_title": latest_provider.title if latest_provider else None,
            "last_error": last_error,
            "query_url": live.get("query_url"),
            "response_code": live.get("response_code"),
            "rows_parsed": live.get("rows_parsed", 0),
            "rows_saved": live.get("rows_saved", 0),
            "last_run_at": live.get("last_run_at"),
        }
    warning = "; ".join(
        provider["last_error"]
        for provider in providers.values()
        if provider["enabled"] and provider.get("last_error") and not provider.get("configured")
    ) or None
    return {
        "collector": collector_state or {},
        "providers": providers,
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


def _provider_warning(message: str) -> bool:
    lowered = message.lower()
    return "rate limit" in lowered or "429" in lowered or "cooling down" in lowered


def latest_news(session: Session, limit: int = 25, provider: str | None = None) -> list[dict[str, Any]]:
    query = (
        select(NewsArticle, NewsSentiment)
        .outerjoin(NewsSentiment, NewsSentiment.article_id == NewsArticle.id)
    )
    if provider:
        query = query.where(NewsArticle.source_name == provider.lower())
    rows = session.execute(query.order_by(desc(NewsArticle.published_at)).limit(limit)).all()
    return [
        {
            "id": article.id,
            "provider": article.source_name,
            "title": article.title,
            "source": article.source,
            "url": article.url,
            "published_at": _dt(article.published_at),
            "sentiment_score": sentiment.sentiment_score if sentiment else None,
            "risk_score": sentiment.risk_score if sentiment else None,
            "sentiment_label": sentiment.sentiment_label if sentiment else None,
            "confidence": sentiment.confidence if sentiment else None,
            "model_name": sentiment.model_name if sentiment else None,
            "affected_symbols": sentiment.affected_symbols if sentiment else [],
        }
        for article, sentiment in rows
    ]
