"""Database adapter for safe, optional news-intelligence enrichment.

The intelligence package deliberately has no ORM dependency.  This adapter is the
only place where a validated local/external result is written to Anata's database.
It does not import signal, portfolio, risk, or execution code.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import ExternalAIRequest, NewsArticle, StructuredNewsEvent as StructuredNewsEventRecord

from .providers import GenericOpenAICompatibleProvider, LocalStudentProvider
from .schemas import NewsDocument
from .service import IntelligenceRouter, RouterPolicy


def _utc(value: datetime | None) -> datetime:
    """Normalize a nullable database timestamp without inventing source time."""
    if value is None:
        return datetime.now(timezone.utc)
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _symbol_for_asset(asset: str) -> str:
    """Use configured spot/perpetual symbols where a base asset is unambiguous."""
    normalized = asset.upper()
    for symbol in [*settings.auto_trader_symbols, *settings.binance_symbols]:
        if symbol.upper().startswith(normalized) and symbol.upper().endswith("USDT"):
            return symbol.upper()
    return f"{normalized}USDT" if normalized and normalized not in {"USDT", "USDC"} else normalized


def build_intelligence_router() -> IntelligenceRouter:
    """Build the lightweight process-local router from explicit configuration.

    Configured Gemini, Groq, Hugging Face router, or generic OpenAI-compatible
    endpoints share the SDK-free HTTP adapter. Pricing must be explicitly configured
    before the zero-budget default permits a request. A configured local student is
    preferred locally and falls back to deterministic rules on any artifact failure.
    """
    providers = []
    local_provider = (
        LocalStudentProvider(
            settings.local_news_student_path,
            model_name=settings.local_news_student_version,
        )
        if settings.local_news_student_path
        else None
    )
    provider_specs = [
        (
            "gemini",
            settings.gemini_base_url,
            settings.gemini_api_key,
            settings.gemini_model,
            settings.gemini_input_cost_per_million_usd,
            settings.gemini_output_cost_per_million_usd,
        ),
        (
            "groq",
            settings.groq_base_url,
            settings.groq_api_key,
            settings.groq_model,
            settings.groq_input_cost_per_million_usd,
            settings.groq_output_cost_per_million_usd,
        ),
        (
            "huggingface",
            settings.huggingface_inference_base_url,
            settings.huggingface_inference_token,
            settings.huggingface_inference_model,
            settings.huggingface_input_cost_per_million_usd,
            settings.huggingface_output_cost_per_million_usd,
        ),
    ]
    for name, base_url, api_key, model, input_cost, output_cost in provider_specs:
        if api_key and model:
            providers.append(
                GenericOpenAICompatibleProvider(
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    timeout_seconds=settings.external_ai_timeout_seconds,
                    input_cost_per_million_usd=input_cost,
                    output_cost_per_million_usd=output_cost,
                    provider_name=name,
                )
            )
    if settings.generic_ai_base_url:
        providers.append(
            GenericOpenAICompatibleProvider(
                base_url=settings.generic_ai_base_url,
                api_key=settings.generic_ai_api_key,
                model=settings.external_ai_generic_model,
                timeout_seconds=settings.external_ai_timeout_seconds,
                input_cost_per_million_usd=settings.external_ai_input_cost_per_million_usd,
                output_cost_per_million_usd=settings.external_ai_output_cost_per_million_usd,
                provider_name="generic",
            )
        )
    return IntelligenceRouter(
        providers,
        local_provider=local_provider,
        policy=RouterPolicy(
            external_ai_enabled=settings.external_ai_enabled,
            provider_order=tuple(settings.external_ai_provider_order),
            daily_request_limit=settings.external_ai_daily_request_limit,
            monthly_budget_usd=settings.external_ai_monthly_budget_usd,
            timeout_seconds=settings.external_ai_timeout_seconds,
            max_retries=settings.external_ai_max_retries,
            importance_threshold=settings.external_ai_importance_threshold,
            local_uncertainty_threshold=settings.external_ai_local_uncertainty_threshold,
            cache_ttl_seconds=settings.external_ai_cache_ttl_seconds,
            circuit_failure_threshold=settings.external_ai_circuit_breaker_failures,
            circuit_open_seconds=settings.external_ai_circuit_breaker_seconds,
        ),
    )


async def enrich_article(
    session: Session,
    article: NewsArticle,
    *,
    router: IntelligenceRouter | None = None,
    force: bool = False,
) -> StructuredNewsEventRecord:
    """Analyze one stored article and persist its selected structured event.

    Existing successful work for an article/prompt version is reused unless an
    operator explicitly requests ``force``.  A disabled, quota-limited, invalid, or
    failed external provider still yields a local event and therefore cannot stop
    collection or the paper pipeline.
    """
    active_router = router or build_intelligence_router()
    prompt_version = settings.external_ai_prompt_version
    if not force:
        existing = session.scalar(
            select(StructuredNewsEventRecord)
            .where(
                StructuredNewsEventRecord.article_id == article.id,
                StructuredNewsEventRecord.prompt_version == prompt_version,
            )
            .order_by(desc(StructuredNewsEventRecord.created_at))
            .limit(1)
        )
        if existing is not None:
            return existing

    received = _utc(article.received_time or article.created_at)
    available = _utc(article.available_to_model_time or received)
    document = NewsDocument(
        title=article.title,
        content=article.raw_text or article.title,
        source=article.source_name or article.source,
        url=article.url,
        published_at=article.event_time or article.published_at,
        received_at=received,
        available_to_model_at=available,
        relevant_assets=(),
        metadata={"article_id": article.id},
    )
    result = await active_router.analyze(document, prompt_version=prompt_version)
    event = result.selected_event
    affected = list(event.affected_assets)
    primary_symbol = _symbol_for_asset(affected[0]) if affected else None
    for audit in result.request_audits:
        session.add(
            ExternalAIRequest(
                symbol=primary_symbol,
                provider=audit.provider,
                model=audit.model,
                content_hash=audit.content_hash,
                prompt_version=audit.prompt_version,
                requested_at=audit.requested_at,
                completed_at=audit.completed_at,
                status=audit.status.value.upper(),
                token_usage=audit.token_usage.model_dump() if audit.token_usage else None,
                estimated_cost_usd=audit.estimated_cost_usd,
                error_category=audit.error_category,
                retry_count=audit.retry_count,
                cache_hit=audit.cache_hit,
                payload={
                    "article_id": article.id,
                    "affected_symbols": [_symbol_for_asset(asset) for asset in affected],
                    "reason_codes": list(result.reason_codes),
                },
            )
        )

    event_available_at = max(available, _utc(event.generated_at))
    row = StructuredNewsEventRecord(
        article_id=article.id,
        primary_symbol=primary_symbol,
        event_type=event.event_type.value,
        affected_assets=affected,
        affected_entities=list(event.affected_entities),
        direction=event.direction.value,
        sentiment=event.sentiment,
        severity=event.severity,
        importance=event.importance,
        novelty=event.novelty,
        time_horizon=event.time_horizon.value,
        factual_claims=[claim.model_dump() for claim in event.factual_claims],
        confidence=event.confidence,
        source_summary=event.source_summary,
        provider=event.provider,
        model=event.model,
        prompt_version=event.prompt_version,
        validation_status=event.validation_status.value.upper(),
        published_time=article.event_time or article.published_at,
        received_time=received,
        processed_time=datetime.now(timezone.utc),
        available_to_model_time=event_available_at,
        payload={
            "document_hash": result.document_hash,
            "source_reference": event.source_reference,
            "local_event": result.local_event.model_dump(),
            "external_event": result.external_event.model_dump() if result.external_event else None,
            "external_ai_available": result.external_ai_available,
            "external_ai_missing": result.external_ai_missing,
            "external_ai_failed": result.external_ai_failed,
            "reason_codes": list(result.reason_codes),
        },
    )
    session.add(row)
    session.flush()
    return row


async def enrich_recent_articles(
    session: Session,
    *,
    limit: int = 25,
    force: bool = False,
    router: IntelligenceRouter | None = None,
) -> list[StructuredNewsEventRecord]:
    """Enrich a bounded batch for the independent enrichment worker or admin API."""
    bounded = min(max(int(limit), 1), 250)
    articles = list(session.scalars(select(NewsArticle).order_by(desc(NewsArticle.created_at)).limit(bounded)))
    active_router = router or build_intelligence_router()
    rows: list[StructuredNewsEventRecord] = []
    for article in articles:
        rows.append(await enrich_article(session, article, router=active_router, force=force))
    return rows
