"""Restart-safe external-intelligence accounting and cache tests."""

from __future__ import annotations

import asyncio
import unittest
from datetime import datetime, timezone

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.db.models import Base, ExternalAIRequest, NewsArticle
from app.intelligence.persistence import enrich_article, hydrate_intelligence_router
from app.intelligence.providers import IntelligenceProviderError
from app.intelligence.schemas import NewsDocument, ProviderResponse, StructuredNewsEvent
from app.intelligence.service import IntelligenceRouter, RouterPolicy


class _SuccessProvider:
    name = "restart_provider"
    model = "restart-v1"
    is_external = True
    estimated_request_cost_usd = 0.5

    def __init__(self) -> None:
        self.calls = 0

    async def enrich(self, document: NewsDocument, *, prompt_version: str) -> ProviderResponse:
        self.calls += 1
        event = StructuredNewsEvent.from_mapping(
            {
                "event_type": "macro",
                "affected_assets": ["BTC"],
                "affected_entities": [],
                "direction": "bearish",
                "sentiment": -0.5,
                "severity": 0.8,
                "importance": 0.9,
                "novelty": 0.7,
                "time_horizon": "short_term",
                "factual_claims": [],
                "confidence": 0.8,
                "source_summary": "Persisted restart-safe context.",
            },
            provider=self.name,
            model=self.model,
            prompt_version=prompt_version,
            source_reference=document.source_reference,
            source_text=document.text,
        )
        return ProviderResponse(provider=self.name, model=self.model, event=event)


class _FailureProvider(_SuccessProvider):
    async def enrich(self, document: NewsDocument, *, prompt_version: str) -> ProviderResponse:
        self.calls += 1
        raise IntelligenceProviderError("provider failure contains should-not-be-persisted-secret")


def _policy(**overrides: object) -> RouterPolicy:
    values = {
        "external_ai_enabled": True,
        "importance_threshold": 0.0,
        "local_uncertainty_threshold": 0.0,
        "daily_request_limit": 20,
        "monthly_budget_usd": 20.0,
        "max_retries": 0,
        "retry_backoff_seconds": 0.0,
        "provider_min_interval_seconds": 0.0,
        "cache_ttl_seconds": 3600,
    }
    values.update(overrides)
    return RouterPolicy(**values)


class IntelligencePersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)

    def tearDown(self) -> None:
        self.engine.dispose()

    @staticmethod
    def _document(suffix: str = "one") -> NewsDocument:
        return NewsDocument(
            title=f"Bitcoin macro shock {suffix}",
            content=f"Bitcoin faces a relevant macro shock in report {suffix}.",
            source="test",
        )

    def _request(
        self,
        *,
        status: str,
        cost: float = 0.0,
        retry_count: int = 0,
        requested_at: datetime | None = None,
    ) -> ExternalAIRequest:
        now = requested_at or datetime.now(timezone.utc)
        return ExternalAIRequest(
            provider="restart_provider",
            model="restart-v1",
            content_hash="a" * 64,
            prompt_version="structured-news-v1",
            requested_at=now,
            completed_at=now,
            status=status,
            estimated_cost_usd=cost,
            retry_count=retry_count,
            cache_hit=False,
            payload={"http_attempt": True, "attempt_number": retry_count + 1},
        )

    def test_restart_hydrates_daily_quota(self) -> None:
        provider = _SuccessProvider()
        router = IntelligenceRouter([provider], policy=_policy(daily_request_limit=1))
        with Session(self.engine) as session:
            session.add(self._request(status="SUCCESS"))
            session.commit()
            hydrate_intelligence_router(session, router)

        result = asyncio.run(router.analyze(self._document("quota")))

        self.assertEqual(provider.calls, 0)
        self.assertEqual(result.request_audits[0].status.value, "quota_exhausted")

    def test_restart_hydrates_spend_rate_and_circuit_state(self) -> None:
        with Session(self.engine) as session:
            session.add(self._request(status="SUCCESS", cost=0.75))
            session.commit()

            budget_provider = _SuccessProvider()
            budget_router = IntelligenceRouter(
                [budget_provider],
                policy=_policy(monthly_budget_usd=1.0),
            )
            hydrate_intelligence_router(session, budget_router)
            budget_result = asyncio.run(budget_router.analyze(self._document("budget")))
            self.assertEqual(budget_result.request_audits[0].status.value, "budget_exhausted")
            self.assertEqual(budget_provider.calls, 0)

            rate_provider = _SuccessProvider()
            rate_router = IntelligenceRouter(
                [rate_provider],
                policy=_policy(provider_min_interval_seconds=60.0),
            )
            hydrate_intelligence_router(session, rate_router)
            rate_result = asyncio.run(rate_router.analyze(self._document("rate")))
            self.assertEqual(rate_result.request_audits[0].status.value, "rate_limited")
            self.assertEqual(rate_provider.calls, 0)

            session.add_all(
                [
                    self._request(status="FAILED", requested_at=datetime.now(timezone.utc)),
                    self._request(status="FAILED", requested_at=datetime.now(timezone.utc)),
                ]
            )
            session.commit()
            circuit_provider = _SuccessProvider()
            circuit_router = IntelligenceRouter(
                [circuit_provider],
                policy=_policy(circuit_failure_threshold=2),
            )
            hydrate_intelligence_router(session, circuit_router)
            circuit_result = asyncio.run(circuit_router.analyze(self._document("circuit")))
            self.assertEqual(circuit_result.request_audits[0].status.value, "circuit_open")
            self.assertEqual(circuit_provider.calls, 0)

    def test_content_prompt_cache_survives_restart_and_url_change(self) -> None:
        first_provider = _SuccessProvider()
        first_router = IntelligenceRouter([first_provider], policy=_policy())
        with Session(self.engine) as session:
            first_article = NewsArticle(
                source="wire-a",
                source_name="wire-a",
                title="Bitcoin macro shock",
                url="https://example.test/one",
                raw_text="Bitcoin faces a relevant macro shock.",
                received_time=datetime.now(timezone.utc),
                available_to_model_time=datetime.now(timezone.utc),
            )
            session.add(first_article)
            session.flush()
            asyncio.run(enrich_article(session, first_article, router=first_router))
            session.commit()
            self.assertEqual(first_provider.calls, 1)

            restarted_provider = _SuccessProvider()
            restarted_router = IntelligenceRouter([restarted_provider], policy=_policy())
            second_article = NewsArticle(
                source="wire-b",
                source_name="wire-b",
                title="Bitcoin macro shock",
                url="https://mirror.test/two",
                raw_text="Bitcoin faces a relevant macro shock.",
                received_time=datetime.now(timezone.utc),
                available_to_model_time=datetime.now(timezone.utc),
            )
            session.add(second_article)
            session.flush()
            row = asyncio.run(enrich_article(session, second_article, router=restarted_router))
            session.commit()

            audits = list(session.scalars(select(ExternalAIRequest).order_by(ExternalAIRequest.id)))
            self.assertEqual(restarted_provider.calls, 0)
            self.assertIn("external_context_cache_hit", (row.payload or {}).get("reason_codes", []))
            self.assertEqual([audit.status for audit in audits], ["SUCCESS", "CACHE_HIT"])

    def test_every_retry_is_persisted_as_a_sanitized_attempt(self) -> None:
        provider = _FailureProvider()
        router = IntelligenceRouter([provider], policy=_policy(max_retries=2))
        with Session(self.engine) as session:
            article = NewsArticle(
                source="wire",
                source_name="wire",
                title="Bitcoin macro shock",
                url="https://example.test/retry",
                raw_text="Bitcoin faces a relevant macro shock.",
                received_time=datetime.now(timezone.utc),
                available_to_model_time=datetime.now(timezone.utc),
            )
            session.add(article)
            session.flush()
            asyncio.run(enrich_article(session, article, router=router))
            session.commit()
            audits = list(session.scalars(select(ExternalAIRequest).order_by(ExternalAIRequest.id)))

        self.assertEqual(provider.calls, 3)
        self.assertEqual([audit.status for audit in audits], ["FAILED", "FAILED", "FAILED"])
        self.assertEqual([audit.retry_count for audit in audits], [0, 1, 2])
        self.assertEqual([audit.payload["attempt_number"] for audit in audits], [1, 2, 3])
        self.assertNotIn("should-not-be-persisted-secret", repr([(audit.error_category, audit.payload) for audit in audits]))


if __name__ == "__main__":
    unittest.main()
