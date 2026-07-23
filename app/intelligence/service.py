"""Quota-aware, fail-open router for external news intelligence.

The router treats external models as an optional context overlay.  It always
returns a local structured event when a provider is disabled, unavailable,
invalid, over quota, or circuit-broken.  It has no execution-side imports.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping

from .providers import (
    GenericOpenAICompatibleProvider,
    IntelligenceProvider,
    IntelligenceProviderError,
    LocalRuleProvider,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from .schemas import (
    EventDirection,
    EventType,
    IntelligenceResult,
    NewsDocument,
    ProviderRequestAudit,
    ProviderResponse,
    ProviderUsage,
    RequestStatus,
    StructuredNewsEvent,
    TimeHorizon,
    utc_now,
)


@dataclass(frozen=True)
class RouterPolicy:
    """Explicit safe limits for an :class:`IntelligenceRouter` instance.

    A zero monthly budget allows only providers whose declared maximum request
    cost is zero.  This prevents accidentally spending on a paid endpoint just
    because a key happened to be present.
    """

    external_ai_enabled: bool = False
    provider_order: tuple[str, ...] = ()
    daily_request_limit: int = 20
    monthly_budget_usd: float = 0.0
    timeout_seconds: float = 15.0
    max_retries: int = 2
    retry_backoff_seconds: float = 0.25
    importance_threshold: float = 0.70
    local_uncertainty_threshold: float = 0.45
    critical_severity_threshold: float = 0.80
    cache_ttl_seconds: int = 86_400
    circuit_failure_threshold: int = 3
    circuit_open_seconds: int = 300
    provider_min_interval_seconds: float = 0.0
    provider_request_cost_estimates: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.daily_request_limit < 0:
            raise ValueError("daily_request_limit cannot be negative")
        if self.monthly_budget_usd < 0:
            raise ValueError("monthly_budget_usd cannot be negative")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if self.retry_backoff_seconds < 0 or self.provider_min_interval_seconds < 0:
            raise ValueError("backoff and provider interval cannot be negative")
        if self.cache_ttl_seconds < 0 or self.circuit_open_seconds < 0:
            raise ValueError("cache/circuit durations cannot be negative")
        if self.circuit_failure_threshold < 1:
            raise ValueError("circuit_failure_threshold must be at least one")
        for name, value in (
            ("importance_threshold", self.importance_threshold),
            ("local_uncertainty_threshold", self.local_uncertainty_threshold),
            ("critical_severity_threshold", self.critical_severity_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")


@dataclass
class _CacheEntry:
    response: ProviderResponse
    expires_at: datetime


class InMemoryIntelligenceCache:
    """Small TTL cache; a database-backed implementation can replace it later."""

    def __init__(self) -> None:
        self._entries: dict[str, _CacheEntry] = {}

    def get(self, key: str, *, now: datetime | None = None) -> ProviderResponse | None:
        now = now or utc_now()
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.expires_at <= now:
            self._entries.pop(key, None)
            return None
        return entry.response

    def put(self, key: str, response: ProviderResponse, *, ttl_seconds: int, now: datetime | None = None) -> None:
        if ttl_seconds <= 0:
            return
        now = now or utc_now()
        self._entries[key] = _CacheEntry(response=response, expires_at=now + timedelta(seconds=ttl_seconds))

    def clear(self) -> None:
        self._entries.clear()


class InMemoryUsageLedger:
    """Tracks requests and estimated spend without coupling to the database."""

    def __init__(self) -> None:
        self._daily_requests: dict[str, int] = {}
        self._monthly_spend: dict[str, float] = {}
        self._last_requested_at: dict[str, datetime] = {}

    @staticmethod
    def _day(now: datetime) -> str:
        return now.astimezone(timezone.utc).date().isoformat()

    @staticmethod
    def _month(now: datetime) -> str:
        current = now.astimezone(timezone.utc)
        return f"{current.year:04d}-{current.month:02d}"

    def daily_requests(self, now: datetime) -> int:
        return self._daily_requests.get(self._day(now), 0)

    def monthly_spend(self, now: datetime) -> float:
        return self._monthly_spend.get(self._month(now), 0.0)

    def last_requested_at(self, provider: str) -> datetime | None:
        return self._last_requested_at.get(provider)

    def reserve_request(self, provider: str, now: datetime) -> None:
        day = self._day(now)
        self._daily_requests[day] = self._daily_requests.get(day, 0) + 1
        self._last_requested_at[provider] = now

    def record_spend(self, amount: float, now: datetime) -> None:
        month = self._month(now)
        self._monthly_spend[month] = self._monthly_spend.get(month, 0.0) + max(0.0, float(amount))


@dataclass
class CircuitBreaker:
    """Per-provider failure state.  A local fallback is used while it is open."""

    failure_count: int = 0
    open_until: datetime | None = None

    def is_open(self, now: datetime) -> bool:
        return self.open_until is not None and now < self.open_until

    def record_success(self) -> None:
        self.failure_count = 0
        self.open_until = None

    def record_failure(self, *, now: datetime, threshold: int, open_seconds: int) -> None:
        self.failure_count += 1
        if self.failure_count >= threshold:
            self.open_until = now + timedelta(seconds=open_seconds)


def _neutral_local_event(document: NewsDocument, *, reason: str) -> StructuredNewsEvent:
    """Last-resort local result; this preserves fail-open behavior even if a custom local model fails."""

    return StructuredNewsEvent.from_mapping(
        {
            "event_type": EventType.OTHER.value,
            "affected_assets": list(document.relevant_assets),
            "affected_entities": [],
            "direction": EventDirection.NEUTRAL.value,
            "sentiment": 0.0,
            "severity": 0.0,
            "importance": 0.0,
            "novelty": 0.0,
            "time_horizon": TimeHorizon.UNKNOWN.value,
            "factual_claims": [],
            "confidence": 0.0,
            "source_summary": document.title,
            "metadata": {"level": 0, "fallback_reason": reason},
        },
        provider="local_failsafe",
        model="neutral-news-failsafe-v1",
        prompt_version="local-failsafe-v1",
        source_reference=document.source_reference,
        source_text=document.text,
    )


class IntelligenceRouter:
    """Route optional external enrichment with local fail-open behavior.

    ``analyze`` only returns :class:`IntelligenceResult` data.  Consumers must
    independently decide how, if at all, to use it downstream; this router has
    no path to execution or risk configuration.
    """

    def __init__(
        self,
        providers: Iterable[IntelligenceProvider] = (),
        *,
        local_provider: IntelligenceProvider | None = None,
        policy: RouterPolicy | None = None,
        cache: InMemoryIntelligenceCache | None = None,
        usage_ledger: InMemoryUsageLedger | None = None,
    ) -> None:
        self.policy = policy or RouterPolicy()
        self.local_provider = local_provider or LocalRuleProvider()
        provider_list = list(providers)
        self.providers: dict[str, IntelligenceProvider] = {}
        for provider in provider_list:
            if provider.name in self.providers:
                raise ValueError(f"duplicate intelligence provider name: {provider.name}")
            self.providers[provider.name] = provider
        self.cache = cache or InMemoryIntelligenceCache()
        self.usage_ledger = usage_ledger or InMemoryUsageLedger()
        self.circuit_breakers: dict[str, CircuitBreaker] = {
            name: CircuitBreaker() for name, provider in self.providers.items() if provider.is_external
        }
        self._state_lock = asyncio.Lock()

    @staticmethod
    def cache_key(document: NewsDocument, prompt_version: str) -> str:
        return f"{document.content_hash}:{prompt_version}"

    async def _local_event(self, document: NewsDocument) -> tuple[StructuredNewsEvent, tuple[str, ...]]:
        try:
            response = await self.local_provider.enrich(document, prompt_version="local-intelligence-v1")
            return response.event, ()
        except Exception:
            # A custom student can fail during artifact loading.  Never let that stop the paper loop.
            try:
                response = await LocalRuleProvider().enrich(document, prompt_version="local-rule-fallback-v1")
                return response.event, ("local_provider_failed_rule_fallback_used",)
            except Exception:
                return _neutral_local_event(document, reason="all_local_providers_failed"), (
                    "local_provider_failed_neutral_fallback_used",
                )

    def _ordered_external_providers(self) -> list[IntelligenceProvider]:
        order = self.policy.provider_order or tuple(self.providers)
        ordered: list[IntelligenceProvider] = []
        for name in order:
            provider = self.providers.get(name)
            if provider is not None and provider.is_external:
                ordered.append(provider)
        return ordered

    @staticmethod
    def _is_relevant(document: NewsDocument, event: StructuredNewsEvent) -> bool:
        if document.relevant_assets or event.affected_assets:
            return True
        words = set(document.text.lower().split())
        return bool(words & {"crypto", "cryptocurrency", "bitcoin", "ethereum", "blockchain", "token"})

    def _request_cost_estimate(self, provider: IntelligenceProvider) -> float:
        if provider.name in self.policy.provider_request_cost_estimates:
            return max(0.0, float(self.policy.provider_request_cost_estimates[provider.name]))
        # Generic provider exposes a conservative upper estimate only when explicit price data is configured.
        return max(0.0, float(getattr(provider, "estimated_request_cost_usd", 0.0)))

    async def _eligibility_audit(
        self, provider: IntelligenceProvider, document: NewsDocument, prompt_version: str, now: datetime
    ) -> ProviderRequestAudit | None:
        """Atomically determine whether one provider request may be attempted."""

        async with self._state_lock:
            breaker = self.circuit_breakers.setdefault(provider.name, CircuitBreaker())
            if breaker.is_open(now):
                return self._audit(provider, document, prompt_version, now, now, RequestStatus.CIRCUIT_OPEN)
            if self.policy.daily_request_limit == 0 or self.usage_ledger.daily_requests(now) >= self.policy.daily_request_limit:
                return self._audit(provider, document, prompt_version, now, now, RequestStatus.QUOTA_EXHAUSTED)
            last_requested_at = self.usage_ledger.last_requested_at(provider.name)
            if (
                last_requested_at is not None
                and (now - last_requested_at).total_seconds() < self.policy.provider_min_interval_seconds
            ):
                return self._audit(provider, document, prompt_version, now, now, RequestStatus.RATE_LIMITED)
            estimated_cost = self._request_cost_estimate(provider)
            projected_spend = self.usage_ledger.monthly_spend(now) + estimated_cost
            if estimated_cost > 0.0 and (
                self.policy.monthly_budget_usd <= 0.0 or projected_spend > self.policy.monthly_budget_usd
            ):
                return self._audit(provider, document, prompt_version, now, now, RequestStatus.BUDGET_EXHAUSTED)
            self.usage_ledger.reserve_request(provider.name, now)
        return None

    async def _record_success(self, provider: IntelligenceProvider, usage: ProviderUsage, now: datetime) -> None:
        async with self._state_lock:
            self.circuit_breakers.setdefault(provider.name, CircuitBreaker()).record_success()
            self.usage_ledger.record_spend(usage.estimated_cost_usd, now)

    async def _record_failure(self, provider: IntelligenceProvider, now: datetime) -> None:
        async with self._state_lock:
            self.circuit_breakers.setdefault(provider.name, CircuitBreaker()).record_failure(
                now=now,
                threshold=self.policy.circuit_failure_threshold,
                open_seconds=self.policy.circuit_open_seconds,
            )

    @staticmethod
    def _error_category(error: Exception) -> str:
        if isinstance(error, IntelligenceProviderError):
            return error.category
        if isinstance(error, asyncio.TimeoutError):
            return "timeout"
        return "unexpected_error"

    @staticmethod
    def _audit(
        provider: IntelligenceProvider,
        document: NewsDocument,
        prompt_version: str,
        requested_at: datetime,
        completed_at: datetime,
        status: RequestStatus,
        *,
        usage: ProviderUsage | None = None,
        error_category: str | None = None,
        retry_count: int = 0,
        cache_hit: bool = False,
    ) -> ProviderRequestAudit:
        return ProviderRequestAudit(
            provider=provider.name,
            model=provider.model,
            content_hash=document.content_hash or "",
            prompt_version=prompt_version,
            requested_at=requested_at,
            completed_at=completed_at,
            status=status,
            token_usage=usage,
            estimated_cost_usd=usage.estimated_cost_usd if usage else 0.0,
            error_category=error_category,
            retry_count=retry_count,
            cache_hit=cache_hit,
        )

    @staticmethod
    def _result(
        document: NewsDocument,
        local_event: StructuredNewsEvent,
        *,
        external_event: StructuredNewsEvent | None = None,
        failed: bool = False,
        reason_codes: Iterable[str] = (),
        audits: Iterable[ProviderRequestAudit] = (),
    ) -> IntelligenceResult:
        return IntelligenceResult(
            document_hash=document.content_hash or "",
            local_event=local_event,
            external_event=external_event,
            # External information is an overlay; a downstream ensemble can explicitly bound its influence.
            selected_event=external_event or local_event,
            external_ai_available=external_event is not None,
            external_ai_missing=external_event is None,
            external_ai_failed=failed,
            reason_codes=tuple(dict.fromkeys(reason_codes)),
            request_audits=tuple(audits),
        )

    async def analyze(
        self, document: NewsDocument | Mapping[str, Any], *, prompt_version: str = "external-news-v1"
    ) -> IntelligenceResult:
        """Return local intelligence plus optional external context, never raising provider errors."""

        if not isinstance(document, NewsDocument):
            document = NewsDocument.from_mapping(document)
        local_event, local_reasons = await self._local_event(document)
        reasons = list(local_reasons)
        if not self.policy.external_ai_enabled:
            return self._result(document, local_event, reason_codes=[*reasons, "external_ai_disabled"])
        if document.is_duplicate:
            return self._result(document, local_event, reason_codes=[*reasons, "duplicate_content_local_only"])
        if not self._is_relevant(document, local_event):
            return self._result(document, local_event, reason_codes=[*reasons, "irrelevant_content_local_only"])
        if local_event.importance < self.policy.importance_threshold:
            return self._result(document, local_event, reason_codes=[*reasons, "local_importance_below_threshold"])
        local_uncertainty = 1.0 - local_event.confidence
        is_critical = local_event.severity >= self.policy.critical_severity_threshold
        if local_uncertainty < self.policy.local_uncertainty_threshold and not is_critical:
            return self._result(document, local_event, reason_codes=[*reasons, "local_confidence_sufficient"])

        cache_key = self.cache_key(document, prompt_version)
        now = utc_now()
        async with self._state_lock:
            cached = self.cache.get(cache_key, now=now)
        if cached is not None:
            audit = self._audit(
                self.providers.get(cached.provider, _CachedProvider(cached.provider, cached.model)),
                document,
                prompt_version,
                now,
                now,
                RequestStatus.CACHE_HIT,
                usage=cached.usage,
                cache_hit=True,
            )
            return self._result(
                document,
                local_event,
                external_event=cached.event,
                reason_codes=[*reasons, "external_context_cache_hit"],
                audits=[audit],
            )

        external_providers = self._ordered_external_providers()
        if not external_providers:
            return self._result(document, local_event, reason_codes=[*reasons, "no_external_provider_available"])

        audits: list[ProviderRequestAudit] = []
        attempted = False
        for provider in external_providers:
            requested_at = utc_now()
            ineligible = await self._eligibility_audit(provider, document, prompt_version, requested_at)
            if ineligible is not None:
                audits.append(ineligible)
                reasons.append(f"{provider.name}_{ineligible.status.value}")
                continue
            attempted = True
            final_error: Exception | None = None
            for retry in range(self.policy.max_retries + 1):
                try:
                    response = await asyncio.wait_for(
                        provider.enrich(document, prompt_version=prompt_version), timeout=self.policy.timeout_seconds
                    )
                    if not isinstance(response, ProviderResponse):
                        raise IntelligenceProviderError("provider did not return ProviderResponse")
                    completed_at = utc_now()
                    await self._record_success(provider, response.usage, completed_at)
                    async with self._state_lock:
                        self.cache.put(cache_key, response, ttl_seconds=self.policy.cache_ttl_seconds, now=completed_at)
                    audits.append(
                        self._audit(
                            provider,
                            document,
                            prompt_version,
                            requested_at,
                            completed_at,
                            RequestStatus.SUCCESS,
                            usage=response.usage,
                            retry_count=retry,
                        )
                    )
                    return self._result(
                        document,
                        local_event,
                        external_event=response.event,
                        reason_codes=[*reasons, f"external_context_from_{provider.name}"],
                        audits=audits,
                    )
                except Exception as exc:  # Provider failures deliberately cannot escape this boundary.
                    final_error = exc
                    if retry < self.policy.max_retries:
                        await asyncio.sleep(self.policy.retry_backoff_seconds * (2**retry))
            completed_at = utc_now()
            await self._record_failure(provider, completed_at)
            audits.append(
                self._audit(
                    provider,
                    document,
                    prompt_version,
                    requested_at,
                    completed_at,
                    RequestStatus.FAILED,
                    error_category=self._error_category(final_error or IntelligenceProviderError()),
                    retry_count=self.policy.max_retries,
                )
            )
            reasons.append(f"{provider.name}_failed")
        return self._result(
            document,
            local_event,
            failed=attempted,
            reason_codes=[*reasons, "external_context_unavailable_local_used"],
            audits=audits,
        )

    async def enrich(
        self, document: NewsDocument | Mapping[str, Any], *, prompt_version: str = "external-news-v1"
    ) -> IntelligenceResult:
        """Compatibility alias for :meth:`analyze`."""

        return await self.analyze(document, prompt_version=prompt_version)


class _CachedProvider:
    """Minimal provider view used only to write a cache-hit audit record."""

    is_external = True

    def __init__(self, name: str, model: str | None) -> None:
        self.name = name
        self.model = model

    async def enrich(self, document: NewsDocument, *, prompt_version: str) -> ProviderResponse:  # pragma: no cover
        raise RuntimeError("cached providers are not callable")
