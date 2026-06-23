from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import httpx
from sqlalchemy import select

from app.ai.news_sentiment import analyze_news
from app.config import settings
from app.db.models import NewsArticle, NewsSentiment
from app.db.session import SessionLocal

logger = logging.getLogger(__name__)


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized)
    except ValueError:
        logger.debug("Could not parse news timestamp: %s", value)
        return None


class NewsCollector:
    def __init__(self, poll_seconds: int | None = None) -> None:
        self.poll_seconds = poll_seconds or settings.news_poll_seconds

    @property
    def can_collect(self) -> bool:
        return bool(settings.news_api_key) or settings.news_mock_fallback_enabled

    @property
    def unavailable_reason(self) -> str:
        return "NEWS_API_KEY missing"

    async def run(self, stop_event: asyncio.Event, state: Any | None = None) -> None:
        if not self.can_collect:
            if state:
                state.warning = self.unavailable_reason
                state.mark_error(self.unavailable_reason)
            return

        while not stop_event.is_set():
            try:
                if state:
                    state.mark_message({"provider": "mock" if not settings.news_api_key else settings.news_provider_url})
                count = await self.fetch_once()
                if state:
                    state.mark_saved(count, {"articles_stored": count, "rows_saved": count})
                    state.last_error = None
            except Exception as exc:
                logger.exception("News collector error")
                if state:
                    state.mark_error(str(exc))
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self.poll_seconds)
            except asyncio.TimeoutError:
                continue

    async def fetch_once(self) -> int:
        if not settings.news_api_key:
            if settings.news_mock_fallback_enabled:
                return self.store_mock_article()
            raise RuntimeError(self.unavailable_reason)

        params = {
            "q": settings.news_query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 50,
            "apiKey": settings.news_api_key,
        }
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(settings.news_provider_url, params=params)
            response.raise_for_status()
            payload = response.json()

        return self.store_articles(payload.get("articles", []), provider_payload=payload)

    def store_articles(self, articles: list[dict[str, Any]], provider_payload: dict[str, Any] | None = None) -> int:
        stored = 0
        with SessionLocal() as session:
            for item in articles:
                url = item.get("url")
                title = item.get("title") or ""
                if not url or not title:
                    continue
                source = (item.get("source") or {}).get("name") or "unknown"
                existing = session.scalar(select(NewsArticle).where(NewsArticle.source == source, NewsArticle.url == url))
                if existing:
                    continue

                body = item.get("content") or item.get("description") or title
                article = NewsArticle(
                    source=source,
                    source_name=source,
                    title=title,
                    url=url,
                    published_at=_parse_datetime(item.get("publishedAt")) or datetime.now(timezone.utc),
                    raw_text=body,
                    raw=item,
                    raw_payload=item,
                )
                session.add(article)
                session.flush()

                sentiment_score, risk_score, topics, affected_symbols = analyze_news(body)
                session.add(
                    NewsSentiment(
                        article_id=article.id,
                        sentiment_score=sentiment_score,
                        risk_score=risk_score,
                        topics=topics,
                        affected_symbols=affected_symbols,
                        source_name="placeholder-v1",
                        raw_payload={
                            "text_length": len(body),
                            "interface": "analyze_news",
                            "provider_status": (provider_payload or {}).get("status"),
                        },
                    )
                )
                stored += 1
            session.commit()
        return stored

    def store_mock_article(self, title: str | None = None, body: str | None = None) -> int:
        now = datetime.now(timezone.utc)
        article = {
            "source": {"name": "mock-news"},
            "title": title or f"Mock crypto market update {now.isoformat()}",
            "url": f"mock://news/{int(now.timestamp())}",
            "publishedAt": now.isoformat(),
            "description": body or "Bitcoin and Ethereum market conditions remain mixed as traders watch liquidity, regulation, and macro risk.",
            "content": body or "Bitcoin and Ethereum market conditions remain mixed as traders watch liquidity, regulation, and macro risk.",
        }
        return self.store_articles([article], provider_payload={"status": "mock"})
