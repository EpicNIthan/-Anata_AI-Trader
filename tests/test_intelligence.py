"""Regression tests for the safe, non-executing intelligence boundary."""

from __future__ import annotations

import asyncio
import unittest

from app.intelligence.providers import IntelligenceProviderError
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


if __name__ == "__main__":
    unittest.main()
