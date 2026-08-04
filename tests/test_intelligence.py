"""Regression tests for the safe, non-executing intelligence boundary."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile
import unittest

from app.intelligence.providers import (
    IntelligenceProviderError,
    LocalRuleProvider,
    LocalStudentProvider,
    ProviderResponseError,
    ProviderTimeoutError,
)
from app.intelligence.schemas import IntelligenceValidationError, NewsDocument, ProviderResponse, StructuredNewsEvent
from app.intelligence.service import IntelligenceRouter, RouterPolicy


class _ExternalSuccess:
    name = "test_external"
    model = "test-v1"
    is_external = True

    async def enrich(self, document: NewsDocument, *, prompt_version: str) -> ProviderResponse:
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
                "source_summary": "A validated external overlay.",
            },
            provider=self.name,
            model=self.model,
            prompt_version=prompt_version,
            source_reference=document.source_reference,
            source_text=document.text,
        )
        return ProviderResponse(provider=self.name, model=self.model, event=event)


class _ExternalFailure:
    name = "test_failure"
    model = "test-v1"
    is_external = True

    async def enrich(self, document: NewsDocument, *, prompt_version: str) -> ProviderResponse:
        raise IntelligenceProviderError("simulated failure")


class _CountingFailure(_ExternalFailure):
    def __init__(self) -> None:
        self.calls = 0

    async def enrich(self, document: NewsDocument, *, prompt_version: str) -> ProviderResponse:
        self.calls += 1
        return await super().enrich(document, prompt_version=prompt_version)


class _CostlyExternal:
    name = "costly_external"
    model = "costly-v1"
    is_external = True
    estimated_request_cost_usd = 1.0

    def __init__(self) -> None:
        self.calls = 0

    async def enrich(self, document: NewsDocument, *, prompt_version: str) -> ProviderResponse:
        self.calls += 1
        raise AssertionError("a provider blocked by the budget must never be called")


class _ExternalTimeout:
    name = "timeout_external"
    model = "timeout-v1"
    is_external = True

    async def enrich(self, document: NewsDocument, *, prompt_version: str) -> ProviderResponse:
        raise ProviderTimeoutError("provider timed out without exposing credentials")


class _ExternalInvalid:
    name = "invalid_external"
    model = "invalid-v1"
    is_external = True

    async def enrich(self, document: NewsDocument, *, prompt_version: str) -> ProviderResponse:
        raise ProviderResponseError("invalid structured JSON")


class IntelligenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = NewsDocument(
            title="Bitcoin macro shock",
            content="Bitcoin faces a macro shock after a 3 percent move.",
            source="test",
        )

    def test_schema_rejects_out_of_range_values(self) -> None:
        payload = {
            "event_type": "macro",
            "affected_assets": ["BTC"],
            "affected_entities": [],
            "direction": "bearish",
            "sentiment": -1.2,
            "severity": 0.8,
            "importance": 0.9,
            "novelty": 0.7,
            "time_horizon": "short_term",
            "factual_claims": [],
            "confidence": 0.8,
            "source_summary": "A summary.",
        }
        with self.assertRaises(IntelligenceValidationError):
            StructuredNewsEvent.from_mapping(payload)

    def test_local_rule_emits_bounded_numeric_provenance(self) -> None:
        event = asyncio.run(LocalRuleProvider().enrich(self.document)).event

        self.assertGreaterEqual(event.source_reliability, 0.0)
        self.assertLessEqual(event.source_reliability, 1.0)
        self.assertEqual(
            set(event.affected_asset_probabilities),
            set(event.affected_assets),
        )
        self.assertTrue(
            all(0.0 <= value <= 1.0 for value in event.affected_asset_probabilities.values())
        )

    def test_legacy_student_artifact_gets_deterministic_numeric_fallbacks(self) -> None:
        artifact = {
            "artifact_type": "anata_news_student_naive_bayes_v1",
            "version": "student-numeric-v1",
            "vocabulary_size": 2,
            "tasks": {
                "sentiment": {"label_counts": {"positive": 2, "neutral": 1}},
                "event_type": {"label_counts": {"macro": 1, "other": 2}},
            },
            "asset_keywords": {"BTC": ["bitcoin"]},
        }
        with tempfile.TemporaryDirectory(prefix="anata-news-student-") as temporary:
            path = Path(temporary) / "student.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            provider = LocalStudentProvider.from_artifact(path)
            first = asyncio.run(provider.enrich(self.document)).event
            second = asyncio.run(provider.enrich(self.document)).event

        self.assertEqual(first.source_reliability, second.source_reliability)
        self.assertEqual(
            first.affected_asset_probabilities,
            second.affected_asset_probabilities,
        )
        self.assertEqual(set(first.affected_asset_probabilities), {"BTC"})
        self.assertTrue(0.0 <= first.affected_asset_probabilities["BTC"] <= 1.0)

    def test_disabled_external_provider_fails_open_to_local(self) -> None:
        result = asyncio.run(IntelligenceRouter().analyze(self.document))
        self.assertFalse(result.external_ai_available)
        self.assertFalse(result.external_ai_failed)
        self.assertEqual(result.selected_event.provider, "local_rule")

    def test_external_response_is_cached_by_content_and_prompt(self) -> None:
        router = IntelligenceRouter(
            [_ExternalSuccess()],
            policy=RouterPolicy(
                external_ai_enabled=True,
                importance_threshold=0.0,
                local_uncertainty_threshold=0.0,
                daily_request_limit=5,
            ),
        )
        first = asyncio.run(router.analyze(self.document))
        second = asyncio.run(router.analyze(self.document))
        self.assertTrue(first.external_ai_available)
        self.assertEqual(second.request_audits[0].status.value, "cache_hit")

    def test_circuit_breaker_preserves_local_result(self) -> None:
        router = IntelligenceRouter(
            [_ExternalFailure()],
            policy=RouterPolicy(
                external_ai_enabled=True,
                importance_threshold=0.0,
                local_uncertainty_threshold=0.0,
                max_retries=0,
                circuit_failure_threshold=2,
                daily_request_limit=10,
            ),
        )
        asyncio.run(router.analyze(self.document))
        asyncio.run(router.analyze(self.document))
        result = asyncio.run(router.analyze(self.document))
        self.assertFalse(result.external_ai_available)
        self.assertIn("test_failure_circuit_open", result.reason_codes)
        self.assertEqual(result.selected_event.provider, "local_rule")

    def test_exhausted_budget_skips_provider_and_fails_open_to_local(self) -> None:
        provider = _CostlyExternal()
        router = IntelligenceRouter(
            [provider],
            policy=RouterPolicy(
                external_ai_enabled=True,
                importance_threshold=0.0,
                local_uncertainty_threshold=0.0,
                monthly_budget_usd=0.0,
                daily_request_limit=10,
            ),
        )

        result = asyncio.run(router.analyze(self.document))

        self.assertEqual(provider.calls, 0)
        self.assertFalse(result.external_ai_available)
        self.assertFalse(result.external_ai_failed)
        self.assertEqual(result.selected_event.provider, "local_rule")
        self.assertIn("costly_external_budget_exhausted", result.reason_codes)
        self.assertEqual(result.request_audits[0].status.value, "budget_exhausted")

    def test_timeout_and_invalid_json_are_audited_without_stopping_local_intelligence(self) -> None:
        for provider, expected_category in (
            (_ExternalTimeout(), "timeout"),
            (_ExternalInvalid(), "invalid_response"),
        ):
            router = IntelligenceRouter(
                [provider],
                policy=RouterPolicy(
                    external_ai_enabled=True,
                    importance_threshold=0.0,
                    local_uncertainty_threshold=0.0,
                    max_retries=0,
                    daily_request_limit=5,
                ),
            )
            result = asyncio.run(router.analyze(self.document))
            self.assertFalse(result.external_ai_available)
            self.assertTrue(result.external_ai_failed)
            self.assertEqual(result.selected_event.provider, "local_rule")
            self.assertEqual(result.request_audits[0].error_category, expected_category)

    def test_provider_fallback_and_daily_quota_are_deterministic(self) -> None:
        router = IntelligenceRouter(
            [_ExternalFailure(), _ExternalSuccess()],
            policy=RouterPolicy(
                external_ai_enabled=True,
                provider_order=("test_failure", "test_external"),
                importance_threshold=0.0,
                local_uncertainty_threshold=0.0,
                max_retries=0,
                daily_request_limit=2,
            ),
        )
        first = asyncio.run(router.analyze(self.document))
        self.assertTrue(first.external_ai_available)
        self.assertEqual(first.external_event.provider, "test_external")
        self.assertEqual([audit.status.value for audit in first.request_audits], ["failed", "success"])

        second_document = NewsDocument(
            title="Ethereum macro shock",
            content="Ethereum faces a separate macro shock after another move.",
            source="test",
        )
        second = asyncio.run(router.analyze(second_document))
        self.assertFalse(second.external_ai_available)
        self.assertFalse(second.external_ai_failed)
        self.assertTrue(all(audit.status.value == "quota_exhausted" for audit in second.request_audits))

    def test_each_retry_consumes_one_daily_request(self) -> None:
        provider = _CountingFailure()
        router = IntelligenceRouter(
            [provider],
            policy=RouterPolicy(
                external_ai_enabled=True,
                importance_threshold=0.0,
                local_uncertainty_threshold=0.0,
                max_retries=3,
                retry_backoff_seconds=0.0,
                provider_min_interval_seconds=0.0,
                daily_request_limit=2,
            ),
        )

        result = asyncio.run(router.analyze(self.document))

        self.assertEqual(provider.calls, 2)
        self.assertEqual(
            [audit.status.value for audit in result.request_audits],
            ["failed", "failed", "quota_exhausted"],
        )
        self.assertEqual(router.usage_ledger.daily_requests(result.request_audits[0].requested_at), 2)

    def test_provider_interval_blocks_a_second_http_attempt(self) -> None:
        provider = _CountingFailure()
        router = IntelligenceRouter(
            [provider],
            policy=RouterPolicy(
                external_ai_enabled=True,
                importance_threshold=0.0,
                local_uncertainty_threshold=0.0,
                max_retries=0,
                provider_min_interval_seconds=60.0,
                circuit_failure_threshold=10,
                daily_request_limit=10,
            ),
        )
        asyncio.run(router.analyze(self.document))
        second = asyncio.run(
            router.analyze(
                NewsDocument(
                    title="Ethereum macro shock",
                    content="Ethereum faces a different macro shock.",
                    source="test",
                )
            )
        )

        self.assertEqual(provider.calls, 1)
        self.assertEqual(second.request_audits[0].status.value, "rate_limited")

    def test_failure_evidence_never_contains_an_unrelated_secret(self) -> None:
        secret = "unit-test-super-secret-token"
        router = IntelligenceRouter(
            [_ExternalInvalid()],
            policy=RouterPolicy(
                external_ai_enabled=True,
                importance_threshold=0.0,
                local_uncertainty_threshold=0.0,
                max_retries=0,
                daily_request_limit=2,
            ),
        )
        result = asyncio.run(router.analyze(self.document))
        self.assertNotIn(secret, repr(result))
        self.assertNotIn(secret, " ".join(result.reason_codes))


if __name__ == "__main__":
    unittest.main()
