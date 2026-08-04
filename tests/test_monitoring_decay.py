from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.orm import Session

from app.config import Settings
from app.db.migrations import run_additive_migrations
from app.db.models import (
    Base,
    Feature,
    ModelHealthSnapshot,
    ModelPredictionRecord,
    ModelVersion,
    ShadowPrediction,
    SignalHealthSnapshot,
    SignalOutcome,
    TradingSignalRecord,
)
from app.pipeline.domain import Direction, HealthStatus, ModelLifecycle, SignalLifecycle
from app.pipeline.monitoring import RollingHealthMonitor


UTC = timezone.utc


class MonitoringDecayTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory(prefix="anata-monitoring-")
        database_path = Path(self._temporary_directory.name) / "monitoring.sqlite3"
        self.engine = create_engine(f"sqlite:///{database_path.as_posix()}")
        Base.metadata.create_all(self.engine)
        self.session = Session(self.engine)

    def tearDown(self) -> None:
        self.session.close()
        self.engine.dispose()
        self._temporary_directory.cleanup()

    @staticmethod
    def _settings(**overrides: object) -> SimpleNamespace:
        values: dict[str, object] = {
            "monitoring_health_window": 4,
            "health_min_observations": 4,
            "health_min_reference_observations": 4,
            "health_suspend_consecutive_errors": 2,
            "v2_max_expected_cost_pct": 0.003,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def _model(
        self,
        name: str,
        family: str,
        *,
        lifecycle: ModelLifecycle = ModelLifecycle.CHAMPION,
        health: HealthStatus = HealthStatus.HEALTHY,
    ) -> ModelVersion:
        model = ModelVersion(
            model_id=f"{name}-id",
            name=name,
            version="v1",
            feature_schema_version="monitor-v1",
            feature_columns=["momentum"],
            path=f"{name}.json",
            status="active",
            model_family=family,
            lifecycle_state=lifecycle.value,
            health_status=health.value,
        )
        self.session.add(model)
        self.session.flush()
        return model

    def _observation(
        self,
        *,
        model: ModelVersion,
        family: str,
        suffix: str,
        observed_at: datetime,
        expected_return: float,
        net_return: float,
        feature_value: float,
        cost: float,
        liquidity: float,
        regime: str,
        missing: bool = False,
    ) -> ModelPredictionRecord:
        generated_at = observed_at - timedelta(minutes=5)
        trace_id = f"trace-{suffix}"
        feature = Feature(
            symbol="BTCUSDT",
            schema_version="monitor-v1",
            as_of=generated_at,
            available_to_model_time=generated_at,
            payload={"schema_version": "monitor-v1", "values": {"momentum": feature_value}},
        )
        self.session.add(feature)
        self.session.flush()
        prediction = ModelPredictionRecord(
            prediction_id=f"pred-{suffix}",
            decision_trace_id=trace_id,
            model_version_id=model.id,
            model_id=model.model_id,
            model_version=model.version,
            model_family=family,
            symbol="BTCUSDT",
            generated_at=generated_at,
            valid_from=generated_at,
            expires_at=generated_at + timedelta(minutes=10),
            forecast_horizon_seconds=300,
            expected_return=expected_return,
            expected_volatility=0.02,
            probability_up=0.8,
            probability_down=0.2,
            confidence=0.9,
            calibration_score=0.8,
            uncertainty=0.2,
            regime=regime,
            feature_schema_version="monitor-v1",
            feature_snapshot_id=f"feature-{suffix}",
            feature_id=feature.id,
            data_version="monitor-test",
            external_context_available=False,
            payload={
                "required_features": ["momentum"],
                "missing_features": ["momentum"] if missing else [],
            },
        )
        signal = TradingSignalRecord(
            signal_id=f"signal-{suffix}",
            prediction_id=prediction.prediction_id,
            decision_trace_id=trace_id,
            signal_family=family,
            symbol="BTCUSDT",
            generated_at=generated_at,
            valid_until=generated_at + timedelta(minutes=10),
            direction=Direction.LONG.value,
            strength=0.8,
            expected_return=expected_return,
            expected_cost=cost,
            net_expected_return=expected_return - cost,
            confidence=0.9,
            uncertainty=0.2,
            regime=regime,
            liquidity_score=liquidity,
            health_status=HealthStatus.HEALTHY.value,
            lifecycle_status=SignalLifecycle.PRODUCTION.value,
            reason_codes=[],
            payload={},
        )
        outcome = SignalOutcome(
            signal_id=signal.signal_id,
            symbol="BTCUSDT",
            horizon_seconds=300,
            realized_return=net_return + cost,
            realized_cost=cost,
            net_return=net_return,
            directional_hit=net_return > 0,
            observed_at=observed_at,
            payload={"point_in_time": True},
        )
        self.session.add_all([prediction, signal, outcome])
        self.session.flush()
        return prediction

    def _shadow(
        self,
        *,
        model: ModelVersion,
        live: ModelPredictionRecord,
        suffix: str,
        expected_return: float,
    ) -> None:
        shadow = ModelPredictionRecord(
            prediction_id=f"shadow-pred-{suffix}",
            decision_trace_id=live.decision_trace_id,
            model_version_id=model.id,
            model_id=model.model_id,
            model_version=model.version,
            model_family=live.model_family,
            symbol=live.symbol,
            generated_at=live.generated_at,
            valid_from=live.valid_from,
            expires_at=live.expires_at,
            forecast_horizon_seconds=live.forecast_horizon_seconds,
            expected_return=expected_return,
            expected_volatility=live.expected_volatility,
            probability_up=0.2,
            probability_down=0.8,
            confidence=0.8,
            calibration_score=0.8,
            uncertainty=0.2,
            regime=live.regime,
            feature_schema_version=live.feature_schema_version,
            feature_snapshot_id=live.feature_snapshot_id,
            feature_id=live.feature_id,
            data_version=live.data_version,
            external_context_available=False,
            payload={"shadow_only": True, "required_features": ["momentum"]},
        )
        self.session.add(shadow)
        self.session.flush()
        self.session.add(
            ShadowPrediction(
                model_version_id=model.id,
                prediction_id=shadow.prediction_id,
                decision_trace_id=shadow.decision_trace_id,
                symbol=shadow.symbol,
                generated_at=shadow.generated_at,
                payload={"shadow_only": True},
            )
        )

    def test_all_decay_metrics_are_calculated_and_persisted_as_columns(self) -> None:
        alpha = self._model("alpha-live", "alpha.decay")
        beta = self._model("beta-live", "beta.peer")
        shadow = self._model("alpha-shadow", "alpha.decay", lifecycle=ModelLifecycle.SHADOW)
        start = datetime(2026, 1, 1, 12, tzinfo=UTC)
        alpha_reference = [-0.02, -0.01, 0.01, 0.02]
        beta_reference = [0.01, -0.01, -0.01, 0.01]
        alpha_recent = [0.01, 0.02, 0.03, 0.04]
        for index in range(4):
            at = start + timedelta(minutes=index)
            self._observation(
                model=alpha,
                family="alpha.decay",
                suffix=f"alpha-reference-{index}",
                observed_at=at,
                expected_return=alpha_reference[index],
                net_return=alpha_reference[index],
                feature_value=index / 10,
                cost=0.0005,
                liquidity=0.9,
                regime="trend" if index % 2 else "range",
            )
            self._observation(
                model=beta,
                family="beta.peer",
                suffix=f"beta-reference-{index}",
                observed_at=at,
                expected_return=beta_reference[index],
                net_return=beta_reference[index],
                feature_value=index / 10,
                cost=0.0005,
                liquidity=0.9,
                regime="trend" if index % 2 else "range",
            )
        for index in range(4):
            at = start + timedelta(minutes=10 + index)
            live = self._observation(
                model=alpha,
                family="alpha.decay",
                suffix=f"alpha-recent-{index}",
                observed_at=at,
                expected_return=0.10 + (index / 10),
                net_return=alpha_recent[index],
                feature_value=10.0 + index,
                cost=0.0025,
                liquidity=0.2,
                regime="trend",
                missing=index < 2,
            )
            self._shadow(
                model=shadow,
                live=live,
                suffix=str(index),
                expected_return=-(0.10 + (index / 10)),
            )
            self._observation(
                model=beta,
                family="beta.peer",
                suffix=f"beta-recent-{index}",
                observed_at=at,
                expected_return=alpha_recent[index],
                net_return=alpha_recent[index],
                feature_value=10.0 + index,
                cost=0.0025,
                liquidity=0.2,
                regime="trend",
            )
        self.session.flush()

        with patch("app.pipeline.monitoring.settings", self._settings()):
            summaries = RollingHealthMonitor(self.session).record_health(
                symbol="BTCUSDT",
                now=start + timedelta(hours=1),
            )

        summary = {item.signal_family: item for item in summaries}["alpha.decay"]
        self.assertAlmostEqual(summary.information_coefficient, 1.0)
        self.assertAlmostEqual(summary.net_expectancy, 0.025)
        self.assertAlmostEqual(summary.calibration_error, 0.1)
        self.assertAlmostEqual(summary.missing_feature_rate, 0.5)
        self.assertAlmostEqual(summary.prediction_drift, 1.0)
        self.assertAlmostEqual(summary.feature_drift, 1.0)
        self.assertAlmostEqual(summary.out_of_distribution_rate, 1.0)
        self.assertGreater(summary.transaction_cost_increase, 3.9)
        self.assertAlmostEqual(summary.signal_correlation_increase, 1.0)
        self.assertAlmostEqual(summary.regime_dependence, 1.0)
        self.assertGreater(summary.capacity_decline, 0.9)
        self.assertAlmostEqual(summary.live_shadow_divergence, 1.0)
        self.assertEqual(summary.health_status, HealthStatus.DEGRADED)
        self.assertEqual(summary.recommended_weight_multiplier, 0.35)
        self.assertEqual(summary.recommended_action, "REDUCED_ENSEMBLE_WEIGHT")

        signal_snapshot = self.session.scalar(
            select(SignalHealthSnapshot)
            .where(SignalHealthSnapshot.signal_family == "alpha.decay")
            .order_by(SignalHealthSnapshot.observed_at.desc())
            .limit(1)
        )
        model_snapshot = self.session.scalar(
            select(ModelHealthSnapshot)
            .where(ModelHealthSnapshot.model_version_id == alpha.id)
            .order_by(ModelHealthSnapshot.observed_at.desc())
            .limit(1)
        )
        self.assertAlmostEqual(signal_snapshot.prediction_drift, summary.prediction_drift)
        self.assertAlmostEqual(signal_snapshot.feature_drift, summary.feature_drift)
        self.assertAlmostEqual(signal_snapshot.ood_rate, summary.out_of_distribution_rate)
        self.assertAlmostEqual(signal_snapshot.live_shadow_divergence, summary.live_shadow_divergence)
        self.assertAlmostEqual(signal_snapshot.transaction_cost_increase, summary.transaction_cost_increase)
        self.assertAlmostEqual(signal_snapshot.correlation_increase, summary.signal_correlation_increase)
        self.assertAlmostEqual(signal_snapshot.regime_dependence, summary.regime_dependence)
        self.assertAlmostEqual(signal_snapshot.capacity_decline, summary.capacity_decline)
        self.assertEqual(signal_snapshot.recommended_action, "REDUCED_ENSEMBLE_WEIGHT")
        self.assertAlmostEqual(model_snapshot.prediction_drift, summary.prediction_drift)
        self.assertAlmostEqual(model_snapshot.signal_correlation_increase, summary.signal_correlation_increase)
        self.assertEqual(model_snapshot.recommended_weight_multiplier, 0.35)
        self.assertEqual(alpha.health_status, HealthStatus.DEGRADED.value)

    def test_consecutive_errors_suspend_and_success_never_reactivates_terminal_health(self) -> None:
        model = self._model("error-model", "alpha.errors")
        monitor = RollingHealthMonitor(self.session)
        start = datetime(2026, 1, 2, 12, tzinfo=UTC)
        with patch("app.pipeline.monitoring.settings", self._settings()):
            first = monitor.record_model_error(model, "first failure", now=start)
            second = monitor.record_model_error(model, "second failure", now=start + timedelta(seconds=1))
        snapshots = list(
            self.session.scalars(
                select(ModelHealthSnapshot)
                .where(ModelHealthSnapshot.model_version_id == model.id)
                .order_by(ModelHealthSnapshot.id)
            )
        )
        self.assertEqual(first, HealthStatus.DEGRADED)
        self.assertEqual(second, HealthStatus.SUSPENDED)
        self.assertEqual([row.consecutive_errors for row in snapshots], [1, 2])
        self.assertEqual(snapshots[-1].recommended_action, "BLOCK_NEW_EXPOSURE")
        self.assertEqual(model.health_status, HealthStatus.SUSPENDED.value)

        # A later successful inference resets the error streak, but terminal health
        # remains sticky until an explicit operator policy changes the model record.
        success = ModelPredictionRecord(
            prediction_id="successful-after-suspension",
            decision_trace_id="trace-success-after-suspension",
            model_version_id=model.id,
            model_id=model.model_id,
            model_version=model.version,
            model_family=model.model_family,
            symbol="BTCUSDT",
            generated_at=start + timedelta(seconds=2),
            valid_from=start + timedelta(seconds=2),
            expires_at=start + timedelta(minutes=5),
            forecast_horizon_seconds=300,
            expected_return=0.01,
            expected_volatility=0.01,
            probability_up=0.6,
            probability_down=0.4,
            confidence=0.7,
            calibration_score=0.7,
            uncertainty=0.3,
            feature_schema_version="monitor-v1",
            feature_snapshot_id="feature-success",
            payload={"missing_features": []},
        )
        self.session.add(success)
        self.session.flush()
        with patch("app.pipeline.monitoring.settings", self._settings()):
            terminal = monitor.record_model_error(model, "failure after success", now=start + timedelta(seconds=3))
        latest = self.session.scalar(
            select(ModelHealthSnapshot)
            .where(ModelHealthSnapshot.model_version_id == model.id)
            .order_by(ModelHealthSnapshot.id.desc())
            .limit(1)
        )
        self.assertEqual(terminal, HealthStatus.SUSPENDED)
        self.assertEqual(latest.consecutive_errors, 1)
        self.assertIn("TERMINAL_MODEL_HEALTH_REQUIRES_EXPLICIT_REACTIVATION", latest.reason_codes)
        self.assertEqual(model.health_status, HealthStatus.SUSPENDED.value)

    def test_good_window_cannot_reactivate_suspended_model_or_signal_family(self) -> None:
        model = self._model(
            "terminal-model",
            "alpha.terminal",
            health=HealthStatus.SUSPENDED,
        )
        start = datetime(2026, 1, 3, 12, tzinfo=UTC)
        self.session.add(
            SignalHealthSnapshot(
                signal_family="alpha.terminal",
                symbol="BTCUSDT",
                health_status=HealthStatus.SUSPENDED.value,
                recommended_weight_multiplier=0.0,
                recommended_action="BLOCK_NEW_EXPOSURE",
                observed_at=start - timedelta(minutes=1),
                reason_codes=["MANUAL_SUSPENSION"],
            )
        )
        for index in range(8):
            value = 0.01 + ((index % 4) * 0.01)
            self._observation(
                model=model,
                family="alpha.terminal",
                suffix=f"terminal-{index}",
                observed_at=start + timedelta(minutes=index),
                expected_return=value,
                net_return=value,
                feature_value=float(index % 4),
                cost=0.0005,
                liquidity=0.9,
                regime="trend" if index % 2 else "range",
            )
        with patch("app.pipeline.monitoring.settings", self._settings()):
            summary = RollingHealthMonitor(self.session).record_health(
                symbol="BTCUSDT",
                now=start + timedelta(hours=1),
            )[0]

        latest_model = self.session.scalar(
            select(ModelHealthSnapshot)
            .where(ModelHealthSnapshot.model_version_id == model.id)
            .order_by(ModelHealthSnapshot.id.desc())
            .limit(1)
        )
        latest_signal = self.session.scalar(
            select(SignalHealthSnapshot)
            .where(
                SignalHealthSnapshot.signal_family == "alpha.terminal",
                SignalHealthSnapshot.symbol == "BTCUSDT",
            )
            .order_by(SignalHealthSnapshot.id.desc())
            .limit(1)
        )
        self.assertEqual(summary.health_status, HealthStatus.SUSPENDED)
        self.assertEqual(summary.recommended_weight_multiplier, 0.0)
        self.assertEqual(latest_model.health_status, HealthStatus.SUSPENDED.value)
        self.assertEqual(latest_model.recommended_action, "BLOCK_NEW_EXPOSURE")
        self.assertEqual(latest_signal.health_status, HealthStatus.SUSPENDED.value)
        self.assertEqual(model.health_status, HealthStatus.SUSPENDED.value)


class MonitoringMigrationTests(unittest.TestCase):
    def test_health_metric_migration_is_additive_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory(prefix="anata-monitoring-migration-") as temporary:
            engine = create_engine(f"sqlite:///{(Path(temporary) / 'legacy.sqlite3').as_posix()}")
            try:
                with engine.begin() as connection:
                    connection.execute(
                        text(
                            "CREATE TABLE model_health_snapshots "
                            "(id INTEGER PRIMARY KEY, model_version_id INTEGER, observed_at TIMESTAMP)"
                        )
                    )
                    connection.execute(
                        text(
                            "CREATE TABLE signal_health_snapshots "
                            "(id INTEGER PRIMARY KEY, signal_family VARCHAR(128), "
                            "symbol VARCHAR(32), observed_at TIMESTAMP)"
                        )
                    )
                first = run_additive_migrations(engine)
                second = run_additive_migrations(engine)
                model_columns = {item["name"] for item in inspect(engine).get_columns("model_health_snapshots")}
                signal_columns = {item["name"] for item in inspect(engine).get_columns("signal_health_snapshots")}
                signal_indexes = {item["name"] for item in inspect(engine).get_indexes("signal_health_snapshots")}
            finally:
                engine.dispose()

        required = {
            "rolling_information_coefficient",
            "rolling_net_expectancy",
            "calibration_error",
            "prediction_drift",
            "feature_drift",
            "ood_rate",
            "missing_feature_rate",
            "live_shadow_divergence",
            "transaction_cost_increase",
            "regime_dependence",
            "capacity_decline",
            "consecutive_errors",
            "recommended_weight_multiplier",
            "recommended_action",
        }
        self.assertTrue(required <= model_columns)
        self.assertTrue(required <= signal_columns)
        self.assertIn("signal_correlation_increase", model_columns)
        self.assertIn("correlation_increase", signal_columns)
        self.assertIn("ix_signal_health_symbol_family_time", signal_indexes)
        self.assertGreater(first["added_count"], 0)
        self.assertEqual(second["added_count"], 0)

    def test_invalid_monitoring_threshold_configuration_is_rejected(self) -> None:
        with patch.dict(
            os.environ,
            {
                "MONITORING_HEALTH_WINDOW": "4",
                "HEALTH_MIN_OBSERVATIONS": "4",
                "HEALTH_MIN_REFERENCE_OBSERVATIONS": "5",
            },
        ):
            with self.assertRaisesRegex(ValueError, "HEALTH_MIN_REFERENCE_OBSERVATIONS"):
                Settings()


if __name__ == "__main__":
    unittest.main()
