"""Database adapter for safe, optional news-intelligence enrichment.

The intelligence package deliberately has no ORM dependency.  This adapter is the
only place where a validated local/external result is written to Anata's database.
It does not import signal, portfolio, risk, or execution code.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import ExternalAIRequest, NewsArticle, StructuredNewsEvent as StructuredNewsEventRecord
from app.pipeline.artifact_store import ArtifactIntegrityError, resolve_model_artifact
from app.pipeline.registry import ModelRegistry

from .providers import (
    GenericOpenAICompatibleProvider,
    IntelligenceProviderError,
    LocalStudentProvider,
    NEWS_STUDENT_MODEL_FAMILY,
)
from .schemas import NewsDocument, ProviderResponse, ProviderUsage, RequestStatus, StructuredNewsEvent
from .service import IntelligenceRouter, RouterPolicy


_HTTP_ATTEMPT_STATUSES = {RequestStatus.SUCCESS.value.upper(), RequestStatus.FAILED.value.upper()}
logger = logging.getLogger(__name__)


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


def _attempt_multiplier(row: ExternalAIRequest) -> int:
    """Count old aggregate retry rows and new one-row-per-attempt records."""

    payload = row.payload if isinstance(row.payload, Mapping) else {}
    if payload.get("http_attempt") is True:
        return 1
    if str(row.status or "").upper() not in _HTTP_ATTEMPT_STATUSES:
        return 0
    # Before per-attempt persistence, one terminal row represented all retries.
    return max(int(row.retry_count or 0) + 1, 1)


def hydrate_intelligence_router(
    session: Session,
    router: IntelligenceRouter,
    *,
    now: datetime | None = None,
) -> None:
    """Restore quota, spend, rate-limit, and circuit state from request audits."""

    now = _utc(now)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    rows = list(
        session.scalars(
            select(ExternalAIRequest)
            .where(ExternalAIRequest.requested_at >= month_start)
            .order_by(ExternalAIRequest.requested_at)
        )
    )
    daily_requests = 0
    monthly_spend = 0.0
    last_requested: dict[str, datetime] = {}
    for row in rows:
        multiplier = _attempt_multiplier(row)
        if multiplier <= 0:
            continue
        requested_at = _utc(row.requested_at)
        if requested_at.date() == now.date():
            daily_requests += multiplier
        monthly_spend += max(float(row.estimated_cost_usd or 0.0), 0.0) * multiplier
        current = last_requested.get(row.provider)
        if current is None or requested_at > current:
            last_requested[row.provider] = requested_at

    consecutive_failures: dict[str, tuple[int, datetime | None]] = {}
    history_limit = max(router.policy.circuit_failure_threshold * 4, 25)
    for provider in router.providers:
        history = list(
            session.scalars(
                select(ExternalAIRequest)
                .where(
                    ExternalAIRequest.provider == provider,
                    ExternalAIRequest.status.in_(tuple(_HTTP_ATTEMPT_STATUSES)),
                )
                .order_by(desc(ExternalAIRequest.requested_at))
                .limit(history_limit)
            )
        )
        failure_count = 0
        latest_failure_at: datetime | None = None
        for row in history:
            status = str(row.status or "").upper()
            if status == RequestStatus.SUCCESS.value.upper():
                break
            if status == RequestStatus.FAILED.value.upper():
                failure_count += _attempt_multiplier(row)
                if latest_failure_at is None:
                    latest_failure_at = _utc(row.completed_at or row.requested_at)
        if history:
            latest_attempt_at = _utc(history[0].requested_at)
            current = last_requested.get(provider)
            if current is None or latest_attempt_at > current:
                last_requested[provider] = latest_attempt_at
            consecutive_failures[provider] = (failure_count, latest_failure_at)

    router.hydrate_persistent_state(
        now=now,
        daily_requests=daily_requests,
        monthly_spend_usd=monthly_spend,
        provider_last_requested_at=last_requested,
        provider_consecutive_failures=consecutive_failures,
    )


def _hydrate_persistent_cache(
    session: Session,
    router: IntelligenceRouter,
    document: NewsDocument,
    prompt_version: str,
    *,
    now: datetime | None = None,
) -> bool:
    """Load a prior validated external event into the TTL cache by content hash."""

    now = _utc(now)
    ttl_seconds = max(int(router.policy.cache_ttl_seconds), 0)
    if ttl_seconds <= 0:
        return False
    cache_key = router.cache_key(document, prompt_version)
    if router.cache.get(cache_key, now=now) is not None:
        return True
    cutoff = now - timedelta(seconds=ttl_seconds)
    audits = list(
        session.scalars(
            select(ExternalAIRequest)
            .where(
                ExternalAIRequest.content_hash == document.content_hash,
                ExternalAIRequest.prompt_version == prompt_version,
                # Cache hits do not extend TTL; the original successful request is
                # always the expiry anchor.
                ExternalAIRequest.status == RequestStatus.SUCCESS.value.upper(),
                ExternalAIRequest.requested_at >= cutoff,
            )
            .order_by(desc(ExternalAIRequest.requested_at))
            .limit(20)
        )
    )
    for audit in audits:
        audit_payload = audit.payload if isinstance(audit.payload, Mapping) else {}
        try:
            article_id = int(audit_payload.get("article_id"))
        except (TypeError, ValueError):
            continue
        stored = session.scalar(
            select(StructuredNewsEventRecord)
            .where(
                StructuredNewsEventRecord.article_id == article_id,
                StructuredNewsEventRecord.prompt_version == prompt_version,
            )
            .order_by(desc(StructuredNewsEventRecord.created_at))
            .limit(1)
        )
        stored_payload = stored.payload if stored is not None and isinstance(stored.payload, Mapping) else {}
        external_payload = stored_payload.get("external_event")
        if stored_payload.get("document_hash") != document.content_hash or not isinstance(external_payload, Mapping):
            continue
        try:
            event = StructuredNewsEvent.from_mapping(
                external_payload,
                provider=str(external_payload.get("provider") or audit.provider),
                model=external_payload.get("model") or audit.model,
                prompt_version=prompt_version,
                source_reference=external_payload.get("source_reference") or document.source_reference,
                source_text=document.text,
            )
            usage_payload = audit.token_usage if isinstance(audit.token_usage, Mapping) else {}
            usage = ProviderUsage(
                input_tokens=usage_payload.get("input_tokens"),
                output_tokens=usage_payload.get("output_tokens"),
                estimated_cost_usd=float(
                    usage_payload.get("estimated_cost_usd", audit.estimated_cost_usd or 0.0)
                ),
            )
            response = ProviderResponse(provider=audit.provider, model=audit.model, event=event, usage=usage)
        except (TypeError, ValueError):
            continue
        cached_at = _utc(audit.completed_at or audit.requested_at)
        remaining_seconds = int(ttl_seconds - max((now - cached_at).total_seconds(), 0.0))
        if remaining_seconds <= 0:
            continue
        router.cache.put(cache_key, response, ttl_seconds=remaining_seconds, now=now)
        return True
    return False


def _active_local_student(session: Session | None) -> LocalStudentProvider | None:
    """Resolve DB champion first, then the explicit environment fallback."""
    if session is not None:
        try:
            model = ModelRegistry(session).champion(
                model_family=NEWS_STUDENT_MODEL_FAMILY,
                symbol="*",
            )
            if model is not None:
                if model.lifecycle_state != "CHAMPION" or model.status != "active":
                    raise ValueError("active news student assignment has invalid lifecycle state")
                runtime_path = resolve_model_artifact(model, session=session)
                raw_payload = model.raw_payload if isinstance(model.raw_payload, Mapping) else {}
                return LocalStudentProvider.from_artifact(
                    runtime_path,
                    model_name=model.version,
                    artifact_member=str(
                        raw_payload.get("model_member")
                        or raw_payload.get("model_file")
                        or "student_artifact.json"
                    ),
                )
        except (ArtifactIntegrityError, IntelligenceProviderError, OSError, ValueError) as exc:
            logger.warning("Active DB news student is unavailable; trying configured fallback: %s", exc)

    if settings.local_news_student_path:
        try:
            configured_version = str(settings.local_news_student_version or "").strip()
            if configured_version == "rule-v1":
                configured_version = ""
            return LocalStudentProvider.from_artifact(
                settings.local_news_student_path,
                model_name=configured_version or None,
            )
        except (ArtifactIntegrityError, IntelligenceProviderError, OSError, ValueError) as exc:
            logger.warning("Configured news student is unavailable; local rules will be used: %s", exc)
    return None


def build_intelligence_router(session: Session | None = None) -> IntelligenceRouter:
    """Build the lightweight process-local router from explicit configuration.

    Configured Gemini, Groq, Hugging Face router, or generic OpenAI-compatible
    endpoints share the SDK-free HTTP adapter. Pricing must be explicitly configured
    before the zero-budget default permits a request. A configured local student is
    preferred locally and falls back to deterministic rules on any artifact failure.
    """
    providers = []
    local_provider = _active_local_student(session)
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
            provider_min_interval_seconds=settings.external_ai_provider_min_interval_seconds,
        ),
    )


async def enrich_article(
    session: Session,
    article: NewsArticle,
    *,
    router: IntelligenceRouter | None = None,
    force: bool = False,
    hydrate_state: bool = True,
) -> StructuredNewsEventRecord:
    """Analyze one stored article and persist its selected structured event.

    Existing successful work for an article/prompt version is reused unless an
    operator explicitly requests ``force``.  A disabled, quota-limited, invalid, or
    failed external provider still yields a local event and therefore cannot stop
    collection or the paper pipeline.
    """
    active_router = router or build_intelligence_router(session)
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
    if hydrate_state:
        hydrate_intelligence_router(session, active_router)
    if not force:
        _hydrate_persistent_cache(session, active_router, document, prompt_version)
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
                    "http_attempt": audit.status in {RequestStatus.SUCCESS, RequestStatus.FAILED},
                    "attempt_number": audit.retry_count + 1
                    if audit.status in {RequestStatus.SUCCESS, RequestStatus.FAILED}
                    else None,
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
    active_router = router or build_intelligence_router(session)
    hydrate_intelligence_router(session, active_router)
    rows: list[StructuredNewsEventRecord] = []
    for article in articles:
        rows.append(
            await enrich_article(
                session,
                article,
                router=active_router,
                force=force,
                hydrate_state=False,
            )
        )
    return rows
