from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode

import httpx
from sqlalchemy import func, or_, select

from app.ai.news_sentiment import analyze_news
from app.config import settings
from app.collectors.rss_news_collector import RssNewsCollector
from app.db.models import NewsArticle, NewsSentiment
from app.db.session import SessionLocal

logger = logging.getLogger(__name__)

_PROVIDER_NEXT_ALLOWED_AT: dict[str, datetime] = {}

SYMBOL_KEYWORDS = {
    "BTCUSDT": {"btc", "bitcoin"},
    "ETHUSDT": {"eth", "ether", "ethereum"},
    "BNBUSDT": {"bnb", "binance"},
    "SOLUSDT": {"sol", "solana"},
    "XRPUSDT": {"xrp", "ripple"},
    "USDT": {"usdt", "tether", "stablecoin"},
}


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        if value.isdigit() and len(value) == 14:
            return datetime.strptime(value, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except ValueError:
        logger.debug("Could not parse news timestamp: %s", value)
        return None


def _sanitize_error(exc: Exception) -> str:
    message = str(exc).strip() or type(exc).__name__
    for secret in (settings.news_api_key,):
        if secret:
            message = message.replace(secret, "***")
    return message.replace("apiKey=your_news_api_key", "apiKey=***")


def _provider_enabled(name: str) -> bool:
    return name.lower() in {provider.lower() for provider in settings.news_providers}


def _cooldown_remaining(provider: str) -> str | None:
    next_allowed = _PROVIDER_NEXT_ALLOWED_AT.get(provider)
    if not next_allowed:
        return None
    now = datetime.now(timezone.utc)
    if now >= next_allowed:
        _PROVIDER_NEXT_ALLOWED_AT.pop(provider, None)
        return None
    seconds = int((next_allowed - now).total_seconds())
    return f"{provider.upper()} cooling down for {seconds}s to avoid free-provider rate limits"


def _set_provider_cooldown(provider: str, seconds: int) -> None:
    if seconds > 0:
        _PROVIDER_NEXT_ALLOWED_AT[provider] = datetime.now(timezone.utc) + timedelta(seconds=seconds)


def _affected_symbols(text: str, extra_codes: list[str] | None = None) -> list[str]:
    tokens = {part.strip(".,:;!?()[]{}\"'").lower() for part in text.split()}
    affected = {
        symbol
        for symbol, words in SYMBOL_KEYWORDS.items()
        if tokens & words
    }
    for code in extra_codes or []:
        normalized = code.strip().upper()
        if normalized == "BTC":
            affected.add("BTCUSDT")
        elif normalized == "ETH":
            affected.add("ETHUSDT")
        elif normalized in {"BNB", "SOL", "XRP"}:
            affected.add(f"{normalized}USDT")
        elif normalized == "USDT":
            affected.add("USDT")
    return sorted(affected)


@dataclass
class NormalizedArticle:
    provider: str
    source: str
    title: str
    url: str
    published_at: datetime
    raw_text: str
    raw_payload: dict[str, Any]
    affected_symbols: list[str] = field(default_factory=list)


@dataclass
class ProviderStatus:
    provider: str
    enabled: bool
    configured: bool
    role: str
    delayed: bool = False
    fallback: bool = False
    running: bool = False
    last_run_at: str | None = None
    last_saved_at: str | None = None
    rows_saved: int = 0
    messages_received: int = 0
    last_error: str | None = None
    latest_article_at: str | None = None
    latest_title: str | None = None
    article_count: int = 0
    query_url: str | None = None
    response_code: int | None = None
    rows_parsed: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseNewsProvider:
    provider = "base"
    role = "news"
    delayed = False
    fallback = False

    @property
    def enabled(self) -> bool:
        return _provider_enabled(self.provider)

    @property
    def configured(self) -> bool:
        return True

    @property
    def unavailable_reason(self) -> str | None:
        return None

    @property
    def last_warning(self) -> str | None:
        return None

    @property
    def min_interval_seconds(self) -> int:
        return 0

    @property
    def last_query_url(self) -> str | None:
        return None

    @property
    def last_response_code(self) -> int | None:
        return None

    @property
    def last_rows_parsed(self) -> int:
        return 0

    async def fetch(self, client: httpx.AsyncClient) -> list[NormalizedArticle]:
        raise NotImplementedError


class RssProvider(BaseNewsProvider):
    provider = "rss"
    role = "free crypto RSS feeds"

    def __init__(self) -> None:
        self._last_warning: str | None = None
        self._last_rows_parsed = 0

    @property
    def enabled(self) -> bool:
        return settings.rss_news_enabled and _provider_enabled(self.provider)

    @property
    def configured(self) -> bool:
        return bool(settings.rss_feeds)

    @property
    def unavailable_reason(self) -> str | None:
        return None if self.configured else "RSS_FEEDS missing"

    @property
    def last_warning(self) -> str | None:
        return self._last_warning

    @property
    def last_query_url(self) -> str | None:
        return ",".join(settings.rss_feeds)

    @property
    def last_rows_parsed(self) -> int:
        return self._last_rows_parsed

    async def fetch(self, client: httpx.AsyncClient) -> list[NormalizedArticle]:
        if not self.configured:
            return []
        collector = RssNewsCollector()
        items = await collector.fetch(client, settings.rss_feeds)
        self._last_rows_parsed = len(items)
        self._last_warning = "; ".join(
            f"{feed}: {error}" for feed, error in collector.last_errors.items()
        ) or None
        articles: list[NormalizedArticle] = []
        for item in items:
            articles.append(
                NormalizedArticle(
                    provider=self.provider,
                    source=item.source,
                    title=item.title,
                    url=item.url,
                    published_at=item.published_at,
                    raw_text=item.raw_text,
                    raw_payload=item.raw_payload,
                    affected_symbols=_affected_symbols(item.raw_text),
                )
            )
        return articles


class GdeltProvider(BaseNewsProvider):
    provider = "gdelt"
    role = "global macro/world risk"

    def __init__(self) -> None:
        self._last_query_url: str | None = None
        self._last_response_code: int | None = None
        self._last_rows_parsed = 0

    @property
    def enabled(self) -> bool:
        return settings.gdelt_enabled and _provider_enabled(self.provider)

    @property
    def min_interval_seconds(self) -> int:
        return settings.gdelt_poll_interval_seconds

    @property
    def last_query_url(self) -> str | None:
        return self._last_query_url

    @property
    def last_response_code(self) -> int | None:
        return self._last_response_code

    @property
    def last_rows_parsed(self) -> int:
        return self._last_rows_parsed

    async def fetch(self, client: httpx.AsyncClient) -> list[NormalizedArticle]:
        query = (
            'bitcoin OR ethereum OR crypto OR cryptocurrency OR binance OR okx OR '
            'tether OR stablecoin OR sec OR "federal reserve" OR inflation OR "interest rates" OR war'
        )
        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": settings.gdelt_max_records,
            "sort": "HybridRel",
        }
        url = f"https://api.gdeltproject.org/api/v2/doc/doc?{urlencode(params)}"
        self._last_query_url = url
        self._last_response_code = None
        self._last_rows_parsed = 0
        response = await client.get(url)
        self._last_response_code = response.status_code
        response.raise_for_status()
        payload = response.json()
        articles: list[NormalizedArticle] = []
        for item in payload.get("articles", []):
            title = item.get("title") or ""
            article_url = item.get("url") or ""
            if not title or not article_url:
                continue
            source = item.get("domain") or item.get("sourceCommonName") or "gdelt"
            published_at = _parse_datetime(item.get("seendate")) or datetime.now(timezone.utc)
            raw_text = " ".join(
                part
                for part in [
                    title,
                    item.get("domain") or "",
                    item.get("language") or "",
                    item.get("sourcecountry") or "",
                ]
                if part
            )
            articles.append(
                NormalizedArticle(
                    provider=self.provider,
                    source=source,
                    title=title,
                    url=article_url,
                    published_at=published_at,
                    raw_text=raw_text,
                    raw_payload=item,
                    affected_symbols=_affected_symbols(raw_text),
                )
            )
        self._last_rows_parsed = len(articles)
        return articles


class NewsApiProvider(BaseNewsProvider):
    provider = "newsapi"
    role = "delayed/free fallback"
    delayed = True
    fallback = True

    def __init__(self) -> None:
        self._last_response_code: int | None = None
        self._last_rows_parsed = 0

    @property
    def enabled(self) -> bool:
        return settings.newsapi_enabled and _provider_enabled(self.provider)

    @property
    def configured(self) -> bool:
        return bool(settings.news_api_key)

    @property
    def unavailable_reason(self) -> str | None:
        return None if self.configured else "NEWS_API_KEY missing or placeholder"

    @property
    def last_query_url(self) -> str | None:
        return settings.news_provider_url

    @property
    def last_response_code(self) -> int | None:
        return self._last_response_code

    @property
    def last_rows_parsed(self) -> int:
        return self._last_rows_parsed

    async def fetch(self, client: httpx.AsyncClient) -> list[NormalizedArticle]:
        if not self.configured:
            return []
        response = await client.get(
            settings.news_provider_url,
            params={
                "q": settings.news_query,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 50,
                "apiKey": settings.news_api_key,
            },
        )
        self._last_response_code = response.status_code
        response.raise_for_status()
        payload = response.json()
        articles: list[NormalizedArticle] = []
        for item in payload.get("articles", []):
            title = item.get("title") or ""
            url = item.get("url") or ""
            if not title or not url:
                continue
            source = (item.get("source") or {}).get("name") or "newsapi"
            body = item.get("content") or item.get("description") or title
            articles.append(
                NormalizedArticle(
                    provider=self.provider,
                    source=source,
                    title=title,
                    url=url,
                    published_at=_parse_datetime(item.get("publishedAt")) or datetime.now(timezone.utc),
                    raw_text=body,
                    raw_payload=item,
                    affected_symbols=_affected_symbols(body),
                )
            )
        self._last_rows_parsed = len(articles)
        return articles


class NewsCollector:
    def __init__(self, poll_seconds: int | None = None) -> None:
        self.poll_seconds = poll_seconds or settings.news_poll_seconds
        self.providers: list[BaseNewsProvider] = [
            RssProvider(),
            GdeltProvider(),
            NewsApiProvider(),
        ]
        self.statuses: dict[str, ProviderStatus] = {
            provider.provider: ProviderStatus(
                provider=provider.provider,
                enabled=provider.enabled,
                configured=provider.configured,
                role=provider.role,
                delayed=provider.delayed,
                fallback=provider.fallback,
                last_error=provider.unavailable_reason if provider.enabled and not provider.configured else None,
            )
            for provider in self.providers
        }

    @property
    def can_collect(self) -> bool:
        return any(provider.enabled and provider.configured for provider in self.providers)

    @property
    def unavailable_reason(self) -> str:
        enabled = [provider for provider in self.providers if provider.enabled]
        if not enabled:
            return "No news providers enabled"
        return "; ".join(provider.unavailable_reason or "" for provider in enabled if not provider.configured) or "No news providers configured"

    def snapshot(self) -> dict[str, dict[str, Any]]:
        self.refresh_db_counts()
        return {name: status.as_dict() for name, status in self.statuses.items()}

    async def run(self, stop_event: asyncio.Event, state: Any | None = None) -> None:
        if not self.can_collect:
            if state:
                state.warning = self.unavailable_reason
                state.details = {"providers": self.snapshot()}
                state.mark_error(self.unavailable_reason)
            return

        while not stop_event.is_set():
            try:
                if state:
                    state.mark_message({"providers": self.snapshot()})
                result = await self.fetch_once()
                if state:
                    state.mark_saved(result["rows_saved"], {"providers": self.snapshot(), **result})
                    state.last_error = self._combined_errors()
                    state.warning = self._combined_warnings()
            except Exception as exc:
                logger.exception("News collector loop error")
                if state:
                    state.mark_error(_sanitize_error(exc))
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self.poll_seconds)
            except asyncio.TimeoutError:
                continue

    async def fetch_once(self, provider_filter: str | None = None) -> dict[str, Any]:
        total_saved = 0
        per_provider: dict[str, Any] = {}
        async with httpx.AsyncClient(timeout=20) as client:
            for provider in self.providers:
                if provider_filter and provider.provider != provider_filter.lower():
                    continue
                status = self.statuses[provider.provider]
                status.enabled = provider.enabled
                status.configured = provider.configured
                if not provider.enabled:
                    status.last_error = None
                    self._apply_provider_diagnostics(status, provider)
                    per_provider[provider.provider] = status.as_dict()
                    continue
                if not provider.configured:
                    status.last_error = provider.unavailable_reason
                    self._apply_provider_diagnostics(status, provider)
                    per_provider[provider.provider] = status.as_dict()
                    continue
                cooldown_message = _cooldown_remaining(provider.provider)
                if cooldown_message:
                    status.last_error = cooldown_message
                    self._apply_provider_diagnostics(status, provider)
                    per_provider[provider.provider] = status.as_dict()
                    continue
                status.running = True
                status.messages_received += 1
                status.last_run_at = datetime.now(timezone.utc).isoformat()
                try:
                    articles = await provider.fetch(client)
                    _set_provider_cooldown(provider.provider, provider.min_interval_seconds)
                    saved = self.store_articles(articles)
                    status.rows_saved += saved
                    if saved:
                        status.last_saved_at = datetime.now(timezone.utc).isoformat()
                    status.last_error = provider.last_warning
                    total_saved += saved
                except Exception as exc:
                    logger.exception("News provider failed: %s", provider.provider)
                    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
                        _set_provider_cooldown(provider.provider, max(provider.min_interval_seconds, 900))
                        status.last_error = (
                            f"{provider.provider.upper()} rate limit (HTTP 429). "
                            f"Cooling down for {max(provider.min_interval_seconds, 900)}s."
                        )
                    else:
                        status.last_error = _sanitize_error(exc)
                finally:
                    status.running = False
                    self._apply_provider_diagnostics(status, provider)
                    per_provider[provider.provider] = status.as_dict()
        self.refresh_db_counts()
        return {
            "rows_saved": total_saved,
            "providers": per_provider,
        }

    def store_articles(self, articles: list[NormalizedArticle]) -> int:
        stored = 0
        with SessionLocal() as session:
            for item in articles:
                existing = session.scalar(
                    select(NewsArticle).where(
                        or_(
                            (NewsArticle.source == item.source) & (NewsArticle.url == item.url),
                            (NewsArticle.source_name == item.provider)
                            & or_(
                                NewsArticle.url == item.url,
                                (NewsArticle.title == item.title) & (NewsArticle.published_at == item.published_at),
                            ),
                        ),
                    )
                )
                if existing:
                    continue
                article = NewsArticle(
                    source=item.source,
                    source_name=item.provider,
                    title=item.title,
                    url=item.url,
                    published_at=item.published_at,
                    raw_text=item.raw_text,
                    raw={**item.raw_payload, "provider": item.provider},
                    raw_payload={**item.raw_payload, "provider": item.provider},
                )
                session.add(article)
                session.flush()

                sentiment = analyze_news(item.raw_text)
                merged_symbols = sorted(set(sentiment.affected_symbols) | set(item.affected_symbols))
                session.add(
                    NewsSentiment(
                        article_id=article.id,
                        sentiment_score=sentiment.sentiment_score,
                        risk_score=sentiment.risk_score,
                        topics=sentiment.topics,
                        affected_symbols=merged_symbols,
                        model_name=sentiment.model_name,
                        sentiment_label=sentiment.label,
                        confidence=sentiment.confidence,
                        source_name=f"{item.provider}:{sentiment.model_name}",
                        raw_payload={
                            "provider": item.provider,
                            "interface": "analyze_news",
                            "text_length": len(item.raw_text),
                            **sentiment.raw_payload,
                        },
                    )
                )
                stored += 1
            session.commit()
        return stored

    def store_mock_article(self, title: str | None = None, body: str | None = None) -> int:
        now = datetime.now(timezone.utc)
        text = body or "Bitcoin and Ethereum market conditions remain mixed as traders watch liquidity, regulation, and macro risk."
        article = NormalizedArticle(
            provider="mock",
            source="mock-news",
            title=title or f"Mock crypto market update {now.isoformat()}",
            url=f"mock://news/{int(now.timestamp())}",
            published_at=now,
            raw_text=text,
            raw_payload={"status": "mock", "body": text},
            affected_symbols=_affected_symbols(text),
        )
        return self.store_articles([article])

    def refresh_db_counts(self) -> None:
        with SessionLocal() as session:
            for provider, status in self.statuses.items():
                status.article_count = int(
                    session.scalar(select(func.count(NewsArticle.id)).where(NewsArticle.source_name == provider)) or 0
                )
                latest = session.scalar(
                    select(NewsArticle)
                    .where(NewsArticle.source_name == provider)
                    .order_by(NewsArticle.published_at.desc())
                    .limit(1)
                )
                status.latest_article_at = latest.published_at.isoformat() if latest and latest.published_at else None
                status.latest_title = latest.title if latest else None

    def _apply_provider_diagnostics(self, status: ProviderStatus, provider: BaseNewsProvider) -> None:
        status.query_url = provider.last_query_url
        status.response_code = provider.last_response_code
        status.rows_parsed = provider.last_rows_parsed

    def _combined_errors(self) -> str | None:
        errors = [
            status.last_error
            for status in self.statuses.values()
            if status.enabled and status.last_error and not self._is_warning(status.last_error)
        ]
        return "; ".join(errors) if errors else None

    def _combined_warnings(self) -> str | None:
        warnings = [
            status.last_error
            for status in self.statuses.values()
            if status.enabled and status.last_error and self._is_warning(status.last_error)
        ]
        return "; ".join(warnings) if warnings else None

    @staticmethod
    def _is_warning(message: str) -> bool:
        lowered = message.lower()
        return "rate limit" in lowered or "429" in lowered or "cooling down" in lowered
