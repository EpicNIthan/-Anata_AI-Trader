"""Focused acceptance checks for architecture and lifecycle invariants."""

from __future__ import annotations

import ast
import asyncio
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError
from sqlalchemy import create_engine, desc, select
from sqlalchemy.orm import Session

from app.api.vision import vision_replay
from app.config import Settings
from app.db.models import Base, ChampionAssignment, DecisionTimelineEvent, ModelVersion, PromotionDecision
from app.intelligence.providers import IntelligenceProviderError
from app.intelligence.schemas import NewsDocument, ProviderResponse
from app.intelligence.service import IntelligenceRouter, RouterPolicy
from app.pipeline.domain import (
    Direction,
    EnsembleDecision,
    EnsembleStatus,
    FeatureSnapshot,
    ModelLifecycle,
    ModelPrediction,
    SignalLifecycle,
    TradingSignal,
)
from app.pipeline.ensemble import DeterministicRegimeEnsemble
from app.pipeline.narrow_models import NewsEventModel
from app.pipeline.portfolio import DeterministicPortfolioConstructor, PortfolioContext
from app.pipeline.registry import ModelRegistry
from app.pipeline.signals import SignalFactory
from app.pipeline.service import V2PipelineService
from app.research.schemas import CandidateStrategySpec, ExperimentDefinition


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _signal(*, lifecycle: SignalLifecycle = SignalLifecycle.PAPER, expired: bool = False) -> TradingSignal:
    now = _now()
    generated_at = now - timedelta(minutes=10) if expired else now
    valid_until = now - timedelta(minutes=5) if expired else now + timedelta(minutes=5)
    return TradingSignal(
        prediction_id="pred-acceptance",
        signal_family="alpha.acceptance",
        symbol="BTCUSDT",
        generated_at=generated_at,
        valid_until=valid_until,
        direction=Direction.LONG,
        strength=0.8,
        expected_return=0.01,
        expected_cost=0.001,
        net_expected_return=0.009,
        confidence=0.8,
        uncertainty=0.2,
        liquidity_score=0.9,
        lifecycle_status=lifecycle,
        metadata={"expected_volatility": 0.02},
    )


def _ensemble_decision() -> EnsembleDecision:
    return EnsembleDecision(
        symbol="BTCUSDT",
        combined_expected_return=0.01,
        combined_expected_volatility=0.02,
        combined_uncertainty=0.10,
        combined_confidence=0.90,
        supporting_signals=["sig-acceptance"],
        signal_weights={"sig-acceptance": 1.0},
        decision_status=EnsembleStatus.ACTIONABLE,
    )


class ArchitectureAndSignalAcceptanceTests(unittest.TestCase):
    def test_forecasting_model_modules_cannot_import_or_reference_paper_execution(self) -> None:
        root = Path(__file__).resolve().parents[1]
        model_sources = [
            path
            for path in (root / "app").rglob("*.py")
            if "model" in path.stem.lower() and path != root / "app" / "db" / "models.py"
        ]
        self.assertTrue(model_sources)
        forbidden_modules = {"app.trading.paper_engine", "app.pipeline.execution"}
        forbidden_names = {"PaperEngine", "PaperExecutionSimulator"}
        violations: list[str] = []
        for path in model_sources:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in forbidden_modules:
                            violations.append(f"{path.name}:{node.lineno}:import {alias.name}")
                elif isinstance(node, ast.ImportFrom) and node.module in forbidden_modules:
                    violations.append(f"{path.name}:{node.lineno}:from {node.module}")
                elif isinstance(node, ast.Name) and node.id in forbidden_names:
                    violations.append(f"{path.name}:{node.lineno}:{node.id}")
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if any(module in node.value for module in forbidden_modules):
                        violations.append(f"{path.name}:{node.lineno}:dynamic execution import")
        self.assertEqual(violations, [])

    def test_portfolio_target_requires_and_traces_an_ensemble_decision(self) -> None:
        constructor = DeterministicPortfolioConstructor()
        context = PortfolioContext(equity=10_000.0, liquidity_score=1.0)
        now = _now()
        prediction = ModelPrediction(
            model_id="acceptance-model",
            model_version="v1",
            model_family="alpha.acceptance",
            symbol="BTCUSDT",
            generated_at=now,
            valid_from=now,
            expires_at=now + timedelta(minutes=5),
            forecast_horizon_seconds=300,
            expected_return=0.01,
            expected_volatility=0.02,
            probability_up=0.6,
            probability_down=0.3,
            confidence=0.8,
            feature_schema_version="acceptance-v1",
            feature_snapshot_id="feature-acceptance",
        )

        with self.assertRaisesRegex(TypeError, "EnsembleDecision"):
            constructor.construct(prediction, context)  # type: ignore[arg-type]

        ensemble = _ensemble_decision()
        target = constructor.construct(ensemble, context)
        self.assertEqual(target.source_ensemble_decision_id, ensemble.ensemble_decision_id)

    def test_expired_signal_is_excluded_before_ensemble_weighting(self) -> None:
        signal = _signal(expired=True)
        self.assertEqual(SignalFactory().valid_signals([signal]), [])
        result = DeterministicRegimeEnsemble().combine("BTCUSDT", [signal])
        self.assertEqual(result.exclusions[signal.signal_id], "EXPIRED")
        self.assertEqual(result.decision.decision_status, EnsembleStatus.NEUTRAL)

    def test_signal_lifecycle_transitions_gate_tradable_influence(self) -> None:
        signal = _signal(lifecycle=SignalLifecycle.RESEARCH)
        factory = SignalFactory()
        ensemble = DeterministicRegimeEnsemble()
        tradable = {
            SignalLifecycle.PAPER,
            SignalLifecycle.LIMITED,
            SignalLifecycle.PRODUCTION,
            SignalLifecycle.REDUCED,
        }
        lifecycle_sequence = (
            SignalLifecycle.RESEARCH,
            SignalLifecycle.VALIDATION,
            SignalLifecycle.SHADOW,
            SignalLifecycle.PAPER,
            SignalLifecycle.LIMITED,
            SignalLifecycle.PRODUCTION,
            SignalLifecycle.REDUCED,
            SignalLifecycle.SUSPENDED,
            SignalLifecycle.RETIRED,
        )
        for lifecycle in lifecycle_sequence:
            with self.subTest(lifecycle=lifecycle.value):
                signal.lifecycle_status = lifecycle
                filtered = factory.valid_signals([signal])
                combined = ensemble.combine("BTCUSDT", [signal])
                if lifecycle in tradable:
                    self.assertEqual(filtered, [signal])
                    self.assertNotIn(signal.signal_id, combined.exclusions)
                    self.assertGreater(combined.decision.signal_weights[signal.signal_id], 0.0)
                else:
                    self.assertEqual(filtered, [])
                    self.assertEqual(combined.exclusions[signal.signal_id], f"LIFECYCLE_{lifecycle.value}")
                    self.assertEqual(combined.decision.signal_weights[signal.signal_id], 0.0)

    def test_correlated_cluster_exposure_is_capped_at_portfolio_construction(self) -> None:
        constructor = DeterministicPortfolioConstructor(
            max_symbol_exposure=0.20,
            max_gross_exposure=1.0,
            max_net_exposure=1.0,
            max_cluster_exposure=0.25,
            minimum_edge=0.0001,
        )
        ensemble = _ensemble_decision()
        independent = constructor.construct(
            ensemble,
            PortfolioContext(equity=10_000.0, liquidity_score=1.0),
        )
        clustered = constructor.construct(
            ensemble,
            PortfolioContext(
                equity=10_000.0,
                exposures={"ETHUSDT": 0.22},
                cluster_by_symbol={"BTCUSDT": "crypto-major", "ETHUSDT": "crypto-major"},
                liquidity_score=1.0,
            ),
        )
        self.assertGreater(independent.requested_target_exposure, 0.03)
        self.assertAlmostEqual(clustered.requested_target_exposure, 0.03, places=8)
        self.assertEqual(V2PipelineService._correlated_cluster("BTCUSDT"), "crypto-beta")
        self.assertEqual(V2PipelineService._correlated_cluster("ETHUSDC"), "crypto-beta")
        self.assertEqual(V2PipelineService._correlated_cluster("UNKNOWN"), "UNKNOWN")

        with patch.dict(os.environ, {"V2_MAX_CLUSTER_EXPOSURE_PCT": "1.01"}):
            with self.assertRaisesRegex(ValueError, "V2_MAX_CLUSTER_EXPOSURE_PCT"):
                Settings()

    def test_confidence_calibration_and_external_context_adjustments_are_bounded(self) -> None:
        now = _now()
        prediction_payload = {
            "model_id": "calibration-acceptance",
            "model_version": "v1",
            "model_family": "alpha.acceptance",
            "symbol": "BTCUSDT",
            "generated_at": now,
            "valid_from": now,
            "expires_at": now + timedelta(minutes=5),
            "forecast_horizon_seconds": 300,
            "expected_return": 0.01,
            "expected_volatility": 0.02,
            "probability_up": 0.6,
            "probability_down": 0.3,
            "confidence": 0.8,
            "calibration_score": 0.5,
            "feature_schema_version": "acceptance-v1",
            "feature_snapshot_id": "feature-calibration",
        }
        for invalid in (-0.01, 1.01):
            with self.subTest(calibration_score=invalid), self.assertRaises(ValidationError):
                ModelPrediction(**{**prediction_payload, "calibration_score": invalid})

        signal = _signal()
        ensemble = DeterministicRegimeEnsemble(external_context_bound=0.10)
        positive = ensemble.combine("BTCUSDT", [signal], external_context_score=999.0)
        negative = ensemble.combine("BTCUSDT", [signal], external_context_score=-999.0)
        self.assertEqual(positive.decision.external_context_adjustment, 0.10)
        self.assertEqual(negative.decision.external_context_adjustment, -0.10)
        self.assertIn("BOUNDED_EXTERNAL_CONTEXT_ADJUSTMENT", positive.decision.reason_codes)

    def test_experiment_definition_has_a_stable_reproducibility_fingerprint(self) -> None:
        candidate = CandidateStrategySpec(
            candidate_id="acceptance-candidate",
            feature_families=("price", "news"),
            target_name="future_return_5m",
            forecast_horizon=300,
            model_family="ridge",
            hyperparameters={"alpha": 1.0, "fit_intercept": True},
            random_seed=17,
        )
        first = ExperimentDefinition(
            experiment_id="experiment-one",
            candidate=candidate,
            dataset_version="dataset-sha256-a",
            feature_version="features-v1",
            code_version="commit-a",
            configuration={"purge": 2, "embargo": 2},
        )
        second = ExperimentDefinition(
            experiment_id="experiment-two",
            candidate=candidate,
            dataset_version="dataset-sha256-a",
            feature_version="features-v1",
            code_version="commit-a",
            configuration={"embargo": 2, "purge": 2},
        )
        changed = ExperimentDefinition(
            experiment_id="experiment-three",
            candidate=candidate,
            dataset_version="dataset-sha256-b",
            feature_version="features-v1",
            code_version="commit-a",
            configuration={"purge": 2, "embargo": 2},
        )
        self.assertEqual(first.reproducibility_fingerprint, second.reproducibility_fingerprint)
        self.assertNotEqual(first.reproducibility_fingerprint, changed.reproducibility_fingerprint)
        restored = ExperimentDefinition.from_dict(first.model_dump())
        self.assertEqual(restored.reproducibility_fingerprint, first.reproducibility_fingerprint)


class ChampionLifecycleAcceptanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory(prefix="anata-registry-acceptance-")
        self.temp_dir = Path(self._temporary_directory.name)
        self.engine = create_engine(f"sqlite:///{(self.temp_dir / 'registry.sqlite3').as_posix()}")
        Base.metadata.create_all(self.engine)
        self.session = Session(self.engine)
        self.registry = ModelRegistry(self.session)

    def tearDown(self) -> None:
        self.session.close()
        self.engine.dispose()
        self._temporary_directory.cleanup()

    def _register(self, name: str, *, lifecycle: ModelLifecycle = ModelLifecycle.TRAINED) -> ModelVersion:
        artifact = self.temp_dir / f"{name}.json"
        artifact.write_text(
            json.dumps({"feature_columns": ["price_change"], "coefficients": [1.0], "intercept": 0.0}),
            encoding="utf-8",
        )
        return self.registry.register(
            name=name,
            model_id=f"{name}-model",
            version="v1",
            model_family="alpha.acceptance",
            path=str(artifact),
            feature_schema_version="acceptance-v1",
            feature_columns=["price_change"],
            lifecycle=lifecycle,
        )

    def test_challenger_cannot_silently_replace_or_register_as_champion(self) -> None:
        first = self._register("first")
        self.registry.promote(first.id, model_family=first.model_family, actor="manual")
        challenger = self._register("challenger")

        with self.assertRaisesRegex(ValueError, "explicit promotion"):
            self._register("silent-champion", lifecycle=ModelLifecycle.CHAMPION)
        with patch("app.pipeline.registry.settings", SimpleNamespace(v2_auto_promote_champion=False)):
            with self.assertRaises(PermissionError):
                self.registry.promote(challenger.id, model_family=challenger.model_family, actor="scheduler")

        active = self.session.scalar(
            select(ChampionAssignment).where(ChampionAssignment.active_to.is_(None))
        )
        self.assertEqual(active.model_version_id, first.id)
        self.assertEqual(challenger.lifecycle_state, ModelLifecycle.TRAINED.value)

    def test_manual_promotion_records_an_auditable_decision(self) -> None:
        candidate = self._register("manual-candidate")
        self.registry.promote(
            candidate.id,
            model_family=candidate.model_family,
            symbol_scope="BTCUSDT",
            actor="manual",
            reason="acceptance approval",
        )
        audit = self.session.scalar(
            select(PromotionDecision).order_by(desc(PromotionDecision.created_at)).limit(1)
        )
        self.assertIsNotNone(audit)
        self.assertEqual(audit.action, "PROMOTE")
        self.assertTrue(audit.approved)
        self.assertEqual(audit.decided_by, "manual")
        self.assertEqual(audit.reason, "acceptance approval")
        self.assertTrue(audit.payload["manual"])

    def test_rollback_restores_previous_champion_and_records_audit(self) -> None:
        first = self._register("rollback-first")
        second = self._register("rollback-second")
        self.registry.promote(first.id, model_family=first.model_family, actor="manual")
        self.registry.promote(second.id, model_family=second.model_family, actor="manual")

        restored = self.registry.rollback(
            model_family=second.model_family,
            actor="manual",
            reason="acceptance rollback",
        )
        self.assertEqual(restored.id, first.id)
        self.assertEqual(self.registry.champion(model_family=first.model_family, symbol="BTCUSDT").id, first.id)
        active = list(
            self.session.scalars(
                select(ChampionAssignment).where(ChampionAssignment.active_to.is_(None))
            )
        )
        self.assertEqual([row.model_version_id for row in active], [first.id])
        audit = self.session.scalar(
            select(PromotionDecision).order_by(desc(PromotionDecision.created_at)).limit(1)
        )
        self.assertEqual(audit.action, "ROLLBACK")
        self.assertEqual(audit.model_version_id, first.id)
        self.assertEqual(audit.previous_model_version_id, second.id)
        self.assertEqual(audit.reason, "acceptance rollback")

    def test_vision_decision_replay_returns_recorded_schema_in_time_order(self) -> None:
        trace_id = "trace-acceptance-replay"
        now = _now()
        self.session.add_all(
            [
                DecisionTimelineEvent(
                    decision_trace_id=trace_id,
                    stage="RISK_DECISION",
                    occurred_at=now + timedelta(seconds=1),
                    status="APPROVED",
                    reason_codes=["RISK_APPROVED"],
                    payload={"risk_decision_id": "risk-1"},
                ),
                DecisionTimelineEvent(
                    decision_trace_id=trace_id,
                    stage="ENSEMBLE",
                    occurred_at=now,
                    status="RECORDED",
                    reason_codes=["ENSEMBLE_READY"],
                    payload={"ensemble_decision_id": "ensemble-1"},
                ),
            ]
        )
        self.session.flush()

        payload = vision_replay(trace_id, session=self.session)
        self.assertEqual(payload["trace_id"], trace_id)
        self.assertEqual(payload["source"], "v2")
        self.assertFalse(payload["partial"])
        self.assertEqual([event["stage"] for event in payload["events"]], ["ENSEMBLE", "RISK_DECISION"])
        for event in payload["events"]:
            self.assertTrue({"id", "stage", "time", "status", "reason_codes", "data"}.issubset(event))
        self.assertEqual(
            set(payload["records"]),
            {"predictions", "signals", "ensembles", "portfolio_targets", "risk_decisions", "orders", "fills"},
        )


class _FailingExternalProvider:
    name = "acceptance_external"
    model = "acceptance-v1"
    is_external = True

    async def enrich(self, document: NewsDocument, *, prompt_version: str) -> ProviderResponse:
        raise IntelligenceProviderError("deterministic acceptance failure")


class LocalIntelligenceContinuationTests(unittest.TestCase):
    def test_external_ai_failure_continues_through_local_model_signal_and_ensemble(self) -> None:
        document = NewsDocument(
            title="Bitcoin macro shock",
            content="Bitcoin faces a macro shock after a three percent move.",
            source="acceptance",
        )
        router = IntelligenceRouter(
            [_FailingExternalProvider()],
            policy=RouterPolicy(
                external_ai_enabled=True,
                importance_threshold=0.0,
                local_uncertainty_threshold=0.0,
                max_retries=0,
                daily_request_limit=2,
            ),
        )
        result = asyncio.run(router.analyze(document))
        event = result.selected_event
        now = _now()
        snapshot = FeatureSnapshot(
            symbol="BTCUSDT",
            as_of=now,
            available_to_model_time=now,
            schema_version="acceptance-v1",
            values={
                "sentiment_score": event.sentiment,
                "risk_score": event.severity,
                "impact_score": event.importance,
                "sentiment_confidence": event.confidence,
                "recency_weight": 1.0,
                "volatility": 0.02,
            },
            external_context={
                "external_ai_available": result.external_ai_available,
                "external_ai_missing": result.external_ai_missing,
                "external_ai_failed": result.external_ai_failed,
                "external_ai_provider": None,
                "local_news_model_version": event.model,
            },
        )
        prediction = NewsEventModel().predict(snapshot)
        signal = SignalFactory(minimum_edge=0.0).from_prediction(prediction)
        ensemble = DeterministicRegimeEnsemble(minimum_edge=0.0).combine("BTCUSDT", [signal])

        self.assertTrue(result.external_ai_failed)
        self.assertEqual(event.provider, "local_rule")
        self.assertFalse(prediction.external_context_available)
        self.assertTrue(prediction.metadata["external_ai_failed"])
        self.assertEqual(prediction.metadata["local_news_model_version"], event.model)
        self.assertIn(signal, SignalFactory().valid_signals([signal]))
        self.assertIn(signal.signal_id, ensemble.decision.signal_weights)


if __name__ == "__main__":
    unittest.main()
