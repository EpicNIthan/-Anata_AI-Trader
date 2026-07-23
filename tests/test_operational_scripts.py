from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from app.pipeline.artifact_models import RegisteredArtifactModel
from app.pipeline.domain import FeatureSnapshot
from app.research import ResearchValidationError
from scripts.manage_model_registry import _artifact_contract, _columns_from_payload, build_parser
from scripts.run_worker import normalize_role
from scripts.train_narrow_return_model import train_artifact


class OperationalScriptTests(unittest.TestCase):
    def _rows(self, count: int = 90) -> list[dict[str, object]]:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        rows: list[dict[str, object]] = []
        for index in range(count):
            timestamp = start + timedelta(minutes=index)
            feature_a = (index - 40) / 100.0
            feature_b = ((index % 11) - 5) / 10.0
            rows.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "available_to_model_time": timestamp.isoformat(),
                    "label_available_time": (timestamp + timedelta(minutes=2)).isoformat(),
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "actual_return": 0.004 * feature_a - 0.002 * feature_b + 0.0001,
                }
            )
        return rows

    def test_trainer_emits_only_return_contract_with_purged_chronological_split(self) -> None:
        artifact = train_artifact(
            rows=self._rows(),
            feature_columns=("feature_a", "feature_b"),
            target_name="actual_return",
            feature_schema_version="unit-v1",
            forecast_horizon_seconds=120,
            train_fraction=0.6,
            validation_fraction=0.2,
            alpha=0.0,
            transaction_cost=0.0,
            minimum_rows_per_split=5,
            dataset_version="unit-dataset-v1",
        )

        self.assertEqual(artifact["allowed_outputs"], ["expected_return"])
        self.assertIn("position_size", artifact["forbidden_outputs"])
        self.assertNotIn("leverage", artifact)
        self.assertLess(artifact["metrics"]["held_out_test"]["root_mean_squared_error"], 1e-10)
        # Forward labels at each split boundary are removed rather than leaking.
        self.assertLess(artifact["split"]["counts_after_purge"]["train"], 54)
        self.assertLess(artifact["split"]["counts_after_purge"]["validation"], 18)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            record = SimpleNamespace(
                id=7,
                model_id="unit-return",
                model_family="alpha.linear_return",
                version="v1",
                feature_columns=["feature_a", "feature_b"],
                path=str(path),
                metrics=artifact["metrics"],
                health_status="HEALTHY",
                forecast_horizon_seconds=120,
                artifact_checksum=None,
                preprocessing_version="raw-linear-v1",
                feature_schema_version="unit-v1",
                raw_payload={},
            )
            model = RegisteredArtifactModel.from_record(record)
            now = datetime.now(timezone.utc)
            prediction = model.predict(
                FeatureSnapshot(
                    symbol="BTCUSDT",
                    as_of=now,
                    available_to_model_time=now,
                    schema_version="unit-v1",
                    values={"feature_a": 0.4, "feature_b": -0.2},
                )
            )
        self.assertAlmostEqual(prediction.expected_return, 0.0021, places=10)
        self.assertNotIn("leverage", prediction.model_dump())

    def test_trainer_rejects_future_feature_availability(self) -> None:
        rows = self._rows()
        timestamp = datetime.fromisoformat(str(rows[10]["timestamp"]))
        rows[10]["available_to_model_time"] = (timestamp + timedelta(seconds=1)).isoformat()
        with self.assertRaises(ResearchValidationError):
            train_artifact(
                rows=rows,
                feature_columns=("feature_a", "feature_b"),
                target_name="actual_return",
                feature_schema_version="unit-v1",
                forecast_horizon_seconds=120,
                train_fraction=0.6,
                validation_fraction=0.2,
                alpha=1.0,
                transaction_cost=0.0,
                minimum_rows_per_split=5,
                dataset_version="unit-dataset-v1",
            )

    def test_trainer_rejects_position_sizing_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "future-return forecast"):
            train_artifact(
                rows=self._rows(),
                feature_columns=("feature_a", "feature_b"),
                target_name="target_best_leverage",
                feature_schema_version="unit-v1",
                forecast_horizon_seconds=120,
                train_fraction=0.6,
                validation_fraction=0.2,
                alpha=1.0,
                transaction_cost=0.0,
                minimum_rows_per_split=5,
                dataset_version="unit-dataset-v1",
            )

    def test_registry_cli_infers_json_artifact_contract(self) -> None:
        payload = {
            "feature_columns": ["a", "b"],
            "coefficients": [0.1, -0.2],
            "intercept": 0.0,
            "feature_schema_version": "unit-v1",
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(_artifact_contract(path)["feature_schema_version"], "unit-v1")
        self.assertEqual(_columns_from_payload(payload), ["a", "b"])
        parsed = build_parser().parse_args(
            [
                "register-challenger",
                "--artifact",
                "model.json",
                "--name",
                "unit",
                "--version",
                "v1",
                "--model-family",
                "alpha.linear_return",
            ]
        )
        self.assertEqual(parsed.command, "register-challenger")

    def test_worker_role_alias_is_normalized(self) -> None:
        self.assertEqual(normalize_role("paper_trader"), "paper-trader")


if __name__ == "__main__":
    unittest.main()
