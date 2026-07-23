"""Integration contracts for registered models and the V2 serving runtime.

The tests deliberately use only disposable SQLite state and local JSON artifacts.
They verify policy boundaries rather than model profitability.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.db.models import (
    Base,
    Candle,
    ChampionAssignment,
    ModelPredictionRecord,
    ModelVersion,
    PaperSandboxAccount,
    ShadowPrediction,
    SignalHealthSnapshot,
    SignalOutcome,
    TradingSignalRecord,
)
from app.pipeline.artifact_models import ArtifactModelError, RegisteredArtifactModel
from app.pipeline.domain import (
    Direction,
    FeatureSnapshot,
    HealthStatus,
    ModelLifecycle,
    SignalLifecycle,
    TradingSignal,
)
from app.pipeline.monitoring import RollingHealthMonitor
from app.pipeline.service import V2PipelineService


UTC = timezone.utc


class RuntimeIntegrationCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory(prefix="anata-v2-runtime-")
        self.temp_dir = Path(self._temporary_directory.name)
        database_path = self.temp_dir / "runtime.sqlite3"
        self.engine = create_engine(
            f"sqlite:///{database_path.as_posix()}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self.engine)
        self.session = Session(self.engine)

    def tearDown(self) -> None:
        self.session.close()
        self.engine.dispose()
        self._temporary_directory.cleanup()

    def _artifact(self, name: str, coefficient: float, **extra: object) -> Path:
        path = self.temp_dir / f"{name}.json"
        path.write_text(
            json.dumps(
                {
                    "feature_columns": ["momentum"],
                    "coefficients": [coefficient],
                    "intercept": 0.001,
                    **extra,
                }
            ),
            encoding="utf-8",
        )
        return path

    def _model(
        self,
        name: str,
        coefficient: float,
        *,
        family: str = "alpha.momentum",
        lifecycle: ModelLifecycle = ModelLifecycle.TRAINED,
        **artifact_fields: object,
    ) -> ModelVersion:
        row = ModelVersion(
            model_id=f"{name}-id",
            name=name,
            version="v1",
            feature_schema_version="runtime-v1",
            feature_columns=["momentum"],
            path=str(self._artifact(name, coefficient, **artifact_fields)),
            status="active" if lifecycle == ModelLifecycle.CHAMPION else "candidate",
            model_family=family,
            lifecycle_state=lifecycle.value,
            health_status=HealthStatus.HEALTHY.value,
            preprocessing_version="runtime-preprocess-v1",
            forecast_horizon_seconds=300,
            metrics={"calibration_score": 0.8, "uncertainty": 0.2},
            raw_payload={},
        )
        self.session.add(row)
        self.session.flush()
        return row

    @staticmethod
    def _snapshot(*, momentum: float = 0.01, as_of: datetime | None = None) -> FeatureSnapshot:
        as_of = as_of or datetime(2026, 1, 1, 12, tzinfo=UTC)
        return FeatureSnapshot(
            symbol="BTCUSDT",
            as_of=as_of,
            available_to_model_time=as_of,
            schema_version="runtime-v1",
            values={"momentum": momentum, "volatility": 0.02},
            data_version="runtime-test",
        )

    @staticmethod
    def _resolver_settings() -> SimpleNamespace:
        return SimpleNamespace(
            v2_champion_account_id="champion",
            v2_use_narrow_models=True,
            v2_require_registered_champion=False,
        )


class RegisteredArtifactBoundaryTests(RuntimeIntegrationCase):
    def test_artifact_emits_forecast_only_and_ignores_legacy_execution_targets(self) -> None:
        row = self._model(
            "legacy-multitarget",
            2.0,
            action="SELL",
            leverage=125,
            margin_allocation=0.99,
            stop_loss=0.50,
            sizing_coefficients=[999_999.0],
            target_columns=["future_return", "leverage", "margin_pct", "action"],
        )

        model = RegisteredArtifactModel.from_record(row)
        prediction = model.predict(self._snapshot(momentum=0.01))

        self.assertAlmostEqual(prediction.expected_return, 0.021)
        self.assertEqual(prediction.model_id, row.model_id)
        self.assertEqual(prediction.metadata["registered_model_version_id"], row.id)
        self.assertTrue(prediction.metadata["legacy_sizing_targets_ignored"])
        for forbidden in ("action", "leverage", "margin_allocation", "stop_loss", "notional"):
            self.assertNotIn(forbidden, type(prediction).model_fields)
            self.assertNotIn(forbidden, prediction.model_dump())

    def test_artifact_rejects_schema_mismatch_before_forecasting(self) -> None:
        row = self._model("schema-bound", 1.0)
        model = RegisteredArtifactModel.from_record(row)
        snapshot = self._snapshot().model_copy(update={"schema_version": "future-schema-v99"})

        with self.assertRaisesRegex(ArtifactModelError, "FEATURE_SCHEMA_MISMATCH"):
            model.predict(snapshot)


class PipelineModelPolicyTests(RuntimeIntegrationCase):
    def test_exact_symbol_champion_wins_and_only_registered_champion_is_resolved(self) -> None:
        wildcard = self._model("wildcard", 1.0, lifecycle=ModelLifecycle.CHAMPION)
        exact = self._model("btc-exact", 3.0, lifecycle=ModelLifecycle.CHAMPION)
        self.session.add_all(
            [
                ChampionAssignment(
                    model_version_id=wildcard.id,
                    model_family=wildcard.model_family,
                    symbol_scope="*",
                ),
                ChampionAssignment(
                    model_version_id=exact.id,
                    model_family=exact.model_family,
                    symbol_scope="BTCUSDT",
                ),
            ]
        )
        self.session.flush()
        service = V2PipelineService(self.session)

        with patch("app.pipeline.service.settings", self._resolver_settings()):
            btc_models, btc_reasons = service._resolve_active_models("BTCUSDT", "champion")
            eth_models, _ = service._resolve_active_models("ETHUSDT", "champion")

        self.assertEqual([item.record.id for item in btc_models], [exact.id])
        self.assertEqual([item.record.id for item in eth_models], [wildcard.id])
        self.assertEqual(btc_models[0].signal_lifecycle, SignalLifecycle.PRODUCTION)
        self.assertIn("REGISTERED_CHAMPION_ASSIGNMENTS", btc_reasons)
        self.assertAlmostEqual(btc_models[0].model.predict(self._snapshot()).expected_return, 0.031)

    def test_sandbox_account_is_bound_to_its_candidate_not_the_champion(self) -> None:
        champion = self._model("champion", 1.0, lifecycle=ModelLifecycle.CHAMPION)
        candidate = self._model("candidate", -2.0, lifecycle=ModelLifecycle.PAPER_SANDBOX)
        self.session.add_all(
            [
                ChampionAssignment(
                    model_version_id=champion.id,
                    model_family=champion.model_family,
                    symbol_scope="*",
                ),
                PaperSandboxAccount(
                    account_id="sandbox-candidate",
                    name="candidate isolation",
                    model_version_id=candidate.id,
                    starting_balance=2_500.0,
                    max_exposure_pct=0.03,
                    active=True,
                ),
            ]
        )
        self.session.flush()
        service = V2PipelineService(self.session)

        with patch("app.pipeline.service.settings", self._resolver_settings()):
            models, reasons = service._resolve_active_models("BTCUSDT", "sandbox-candidate")

        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].record.id, candidate.id)
        self.assertNotEqual(models[0].record.id, champion.id)
        self.assertEqual(models[0].signal_lifecycle, SignalLifecycle.PAPER)
        self.assertEqual(reasons, ["ISOLATED_SANDBOX_MODEL"])
        self.assertLess(models[0].model.predict(self._snapshot()).expected_return, 0.0)

    def test_shadow_prediction_is_persisted_but_never_resolved_or_signaled(self) -> None:
        champion = self._model("champion-main", 1.0, lifecycle=ModelLifecycle.CHAMPION)
        shadow = self._model("challenger-shadow", 5.0, lifecycle=ModelLifecycle.SHADOW)
        self.session.add(
            ChampionAssignment(
                model_version_id=champion.id,
                model_family=champion.model_family,
                symbol_scope="*",
            )
        )
        self.session.flush()
        service = V2PipelineService(self.session)

        with patch("app.pipeline.service.settings", self._resolver_settings()):
            active, _ = service._resolve_active_models("BTCUSDT", "champion")
            prediction_ids, errors = service._run_shadow_models(
                self._snapshot(),
                trace_id="trace-shadow-isolation",
                feature_id=None,
            )

        self.assertEqual([item.record.id for item in active], [champion.id])
        self.assertEqual(errors, [])
        self.assertEqual(len(prediction_ids), 1)
        shadow_row = self.session.scalar(select(ShadowPrediction))
        prediction_row = self.session.scalar(
            select(ModelPredictionRecord).where(ModelPredictionRecord.prediction_id == prediction_ids[0])
        )
        self.assertIsNotNone(shadow_row)
        self.assertEqual(shadow_row.model_version_id, shadow.id)
        self.assertIsNotNone(prediction_row)
        self.assertTrue(prediction_row.payload["shadow_only"])
        self.assertEqual(self.session.scalar(select(TradingSignalRecord)), None)


class RollingMonitoringTests(RuntimeIntegrationCase):
    @staticmethod
    def _monitoring_settings(**overrides: object) -> SimpleNamespace:
        values: dict[str, object] = {
            "monitoring_outcome_batch_size": 250,
            "monitoring_health_window": 20,
            "health_min_observations": 2,
            "health_watch_calibration_error": 0.30,
            "health_degraded_calibration_error": 0.50,
            "health_watch_missing_feature_rate": 0.20,
            "health_degraded_missing_feature_rate": 0.50,
            "health_suspend_consecutive_errors": 3,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def _prediction_and_signal(
        self,
        *,
        suffix: str,
        family: str,
        generated_at: datetime,
        expected_return: float = 0.01,
        confidence: float = 0.8,
        symbol: str = "BTCUSDT",
    ) -> tuple[ModelPredictionRecord, TradingSignalRecord]:
        prediction_id = f"pred-{suffix}"
        signal_id = f"sig-{suffix}"
        prediction = ModelPredictionRecord(
            prediction_id=prediction_id,
            decision_trace_id=f"trace-{suffix}",
            model_id=f"model-{family}",
            model_version="v1",
            model_family=family,
            symbol=symbol,
            generated_at=generated_at,
            valid_from=generated_at,
            expires_at=generated_at + timedelta(minutes=10),
            forecast_horizon_seconds=300,
            expected_return=expected_return,
            expected_volatility=0.02,
            probability_up=0.7,
            probability_down=0.3,
            confidence=confidence,
            calibration_score=0.8,
            uncertainty=0.2,
            feature_schema_version="runtime-v1",
            feature_snapshot_id=f"feature-{suffix}",
            data_version="runtime-test",
            external_context_available=False,
            payload={"missing_features": []},
        )
        signal = TradingSignalRecord(
            signal_id=signal_id,
            prediction_id=prediction_id,
            decision_trace_id=f"trace-{suffix}",
            signal_family=family,
            symbol=symbol,
            generated_at=generated_at,
            valid_until=generated_at + timedelta(minutes=10),
            direction=Direction.LONG.value,
            strength=0.8,
            expected_return=expected_return,
            expected_cost=0.001,
            net_expected_return=expected_return - 0.001,
            confidence=confidence,
            uncertainty=0.2,
            liquidity_score=0.9,
            health_status=HealthStatus.HEALTHY.value,
            lifecycle_status=SignalLifecycle.PRODUCTION.value,
            reason_codes=[],
            payload={},
        )
        self.session.add_all([prediction, signal])
        self.session.flush()
        return prediction, signal

    def test_outcome_label_uses_closed_point_in_time_candles_and_is_idempotent(self) -> None:
        generated = datetime(2026, 1, 1, 12, tzinfo=UTC)
        _, signal = self._prediction_and_signal(
            suffix="pit",
            family="alpha.pit",
            generated_at=generated,
        )
        candles = [
            Candle(
                exchange="test",
                symbol="BTCUSDT",
                interval="1m",
                open_time=generated - timedelta(minutes=1),
                close_time=generated,
                open=99.0,
                high=101.0,
                low=98.0,
                close=100.0,
                volume=1.0,
                is_closed=True,
            ),
            Candle(
                exchange="test",
                symbol="BTCUSDT",
                interval="1m",
                open_time=generated + timedelta(minutes=4),
                close_time=generated + timedelta(minutes=5),
                open=100.0,
                high=111.0,
                low=99.0,
                close=110.0,
                volume=1.0,
                is_closed=True,
            ),
            Candle(
                exchange="test",
                symbol="BTCUSDT",
                interval="1m",
                open_time=generated + timedelta(minutes=7),
                close_time=generated + timedelta(minutes=8),
                open=110.0,
                high=201.0,
                low=109.0,
                close=200.0,
                volume=1.0,
                is_closed=True,
            ),
        ]
        self.session.add_all(candles)
        self.session.flush()
        monitor = RollingHealthMonitor(self.session)

        with patch("app.pipeline.monitoring.settings", self._monitoring_settings()):
            created = monitor.label_matured_outcomes(
                symbol="BTCUSDT",
                now=generated + timedelta(minutes=6),
            )
            duplicate = monitor.label_matured_outcomes(
                symbol="BTCUSDT",
                now=generated + timedelta(minutes=6),
            )

        outcome = self.session.scalar(select(SignalOutcome).where(SignalOutcome.signal_id == signal.signal_id))
        self.assertEqual(created, 1)
        self.assertEqual(duplicate, 0)
        self.assertAlmostEqual(outcome.realized_return, 0.10)
        self.assertEqual(outcome.payload["start_candle_id"], candles[0].id)
        self.assertEqual(outcome.payload["end_candle_id"], candles[1].id)
        self.assertTrue(outcome.payload["point_in_time"])

    def test_health_performance_and_correlations_are_symbol_scoped(self) -> None:
        observed = datetime(2026, 1, 2, 12, tzinfo=UTC)
        current_signals: list[TradingSignal] = []
        series = [(0.01, 0.02), (0.02, 0.04), (0.03, 0.06)]
        for index, (left_return, right_return) in enumerate(series):
            at = observed + timedelta(minutes=index)
            for family, value in (("alpha.left", left_return), ("alpha.right", right_return)):
                _, signal = self._prediction_and_signal(
                    suffix=f"{family}-{index}",
                    family=family,
                    generated_at=at - timedelta(minutes=5),
                    expected_return=value + 0.001,
                )
                self.session.add(
                    SignalOutcome(
                        signal_id=signal.signal_id,
                        symbol="BTCUSDT",
                        horizon_seconds=300,
                        realized_return=value,
                        realized_cost=0.0,
                        net_return=value,
                        directional_hit=True,
                        observed_at=at,
                        payload={"point_in_time": True},
                    )
                )
            if index == 0:
                for family in ("alpha.left", "alpha.right"):
                    current_signals.append(
                        TradingSignal(
                            prediction_id=f"current-{family}",
                            signal_family=family,
                            symbol="BTCUSDT",
                            generated_at=observed,
                            valid_until=observed + timedelta(minutes=5),
                            direction=Direction.LONG,
                            strength=0.8,
                            expected_return=0.02,
                            expected_cost=0.001,
                            net_expected_return=0.019,
                            confidence=0.8,
                            uncertainty=0.2,
                            liquidity_score=0.9,
                        )
                    )

        # A large ETH result must not pollute BTC-scoped health or expectancy.
        _, eth_signal = self._prediction_and_signal(
            suffix="eth-outlier",
            family="alpha.left",
            generated_at=observed,
            expected_return=1.0,
            symbol="ETHUSDT",
        )
        self.session.add(
            SignalOutcome(
                signal_id=eth_signal.signal_id,
                symbol="ETHUSDT",
                horizon_seconds=300,
                realized_return=1.0,
                realized_cost=0.0,
                net_return=1.0,
                directional_hit=True,
                observed_at=observed + timedelta(hours=1),
                payload={"point_in_time": True},
            )
        )
        self.session.flush()
        monitor = RollingHealthMonitor(self.session)

        with patch("app.pipeline.monitoring.settings", self._monitoring_settings()):
            summaries = monitor.record_health(symbol="BTCUSDT", now=observed + timedelta(hours=2))
            performance = monitor.recent_performance(["alpha.left"], symbol="BTCUSDT")
            correlations = monitor.signal_correlations(current_signals)

        by_family = {item.signal_family: item for item in summaries}
        pair = (current_signals[0].signal_id, current_signals[1].signal_id)
        self.assertEqual(by_family["alpha.left"].observations, 3)
        self.assertEqual(by_family["alpha.left"].health_status, HealthStatus.HEALTHY)
        self.assertAlmostEqual(performance["alpha.left"], 0.02)
        self.assertAlmostEqual(correlations[pair], 1.0)
        self.assertEqual(
            self.session.scalar(
                select(SignalHealthSnapshot.health_status)
                .where(SignalHealthSnapshot.signal_family == "alpha.left")
                .order_by(SignalHealthSnapshot.observed_at.desc())
                .limit(1)
            ),
            HealthStatus.HEALTHY.value,
        )


if __name__ == "__main__":
    unittest.main()
