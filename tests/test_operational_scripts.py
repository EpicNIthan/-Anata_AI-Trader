from __future__ import annotations

import json
import gzip
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
from scripts.research_utils import read_research_rows
from scripts.train_narrow_return_model import _dataset_schema_version, _infer_features, train_artifact
from scripts.upload_model import upload_package


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

    def test_gzip_csv_and_training_serving_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prepared.csv.gz"
            with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
                handle.write("as_of,feature_schema_version,price_change,training_feature_id,target_future_return_5m\n")
                handle.write("2026-01-01T00:00:00+00:00,price-news-market-v5,0.1,99,0.01\n")
            rows = read_research_rows(path)
        schema = _dataset_schema_version(rows, None)
        features = _infer_features(
            rows,
            "target_future_return_5m",
            allowed_features={"price_change"},
        )
        self.assertEqual(schema, "price-news-market-v5")
        self.assertEqual(features, ["price_change"])
        with self.assertRaisesRegex(ValueError, "does not match dataset schema"):
            _dataset_schema_version(rows, "price-news-market-v4")

    def test_trainer_drops_unlabeled_tail_rows(self) -> None:
        rows = self._rows()
        rows[-1]["actual_return"] = ""
        artifact = train_artifact(
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
        self.assertEqual(artifact["missing_value_policy"]["dropped_unlabeled_rows"], 1)

    def test_upload_helper_uses_header_and_accepts_candidate_only(self) -> None:
        captured: dict[str, object] = {}

        class Response:
            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            @staticmethod
            def read() -> bytes:
                return json.dumps(
                    {
                        "status": "candidate",
                        "model": {"id": 17, "status": "candidate", "lifecycle_state": "TRAINED"},
                    }
                ).encode("utf-8")

        def opener(req: object, *, timeout: float) -> Response:
            captured["request"] = req
            captured["timeout"] = timeout
            return Response()

        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / "challenger.zip"
            package.write_bytes(b"package-bytes")
            result = upload_package(
                url="https://anata.example",
                token="header-secret",
                package=package,
                timeout_seconds=9.0,
                opener=opener,
            )

        req = captured["request"]
        self.assertEqual(req.full_url, "https://anata.example/api/models/upload")
        self.assertEqual(req.get_header("X-admin-token"), "header-secret")
        self.assertNotIn(b"header-secret", req.data)
        self.assertEqual(captured["timeout"], 9.0)
        self.assertEqual(result["model"]["id"], 17)

    def test_upload_helper_rejects_unsafe_url_and_active_response(self) -> None:
        class ActiveResponse:
            def __enter__(self) -> "ActiveResponse":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            @staticmethod
            def read() -> bytes:
                return b'{"status":"active","model":{"status":"active"}}'

        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / "challenger.zip"
            package.write_bytes(b"package-bytes")
            with self.assertRaisesRegex(ValueError, "HTTP\\(S\\)"):
                upload_package(url="file:///tmp", token="secret", package=package)
            with self.assertRaisesRegex(ValueError, "non-candidate"):
                upload_package(
                    url="https://anata.example",
                    token="secret",
                    package=package,
                    opener=lambda *_args, **_kwargs: ActiveResponse(),
                )


if __name__ == "__main__":
    unittest.main()
