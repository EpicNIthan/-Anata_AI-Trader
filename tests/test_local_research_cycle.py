"""End-to-end tests for the bounded local narrow-model research cycle."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import zipfile
from unittest.mock import patch

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from app.db.models import (
    Base,
    CandidateEvaluation,
    ChampionAssignment,
    ExperimentRun,
    ModelVersion,
    StrategyCandidate,
)
from app.pipeline.artifact_models import ArtifactModelError, RegisteredArtifactModel
from app.pipeline.domain import FeatureSnapshot, ModelLifecycle
from app.pipeline.registry import ArtifactValidator, ModelRegistry
from app.research import ResearchValidationError
from app.research.training import (
    build_narrow_candidate_configs,
    discover_available_labeled_rows,
    observed_news_student_contract,
    package_narrow_artifact,
    run_narrow_research_cycle,
    verify_narrow_package,
)
from scripts.research_utils import read_research_rows
from scripts.run_local_research_cycle import build_parser, execute, persist_research_cycle


def _rows(count: int = 180) -> list[dict[str, object]]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows: list[dict[str, object]] = []
    for index in range(count):
        timestamp = start + timedelta(minutes=index)
        price_change = ((index % 17) - 8) / 1000.0
        sentiment = ((index % 9) - 4) / 4.0
        volatility = 0.005 + (index % 7) / 1000.0
        risk = (index % 5) / 5.0
        volume_change = ((index % 13) - 6) / 10.0
        target = 0.65 * price_change + 0.0008 * sentiment - 0.03 * volatility - 0.0002 * risk
        rows.append(
            {
                "timestamp": timestamp.isoformat(),
                "available_to_model_time": timestamp.isoformat(),
                "label_available_time": (timestamp + timedelta(minutes=1)).isoformat(),
                "feature_schema_version": "price-news-v1",
                "symbol": "BTCUSDT",
                "price_change": price_change,
                "volume_change": volume_change,
                "volatility": volatility,
                "sentiment_score": sentiment,
                "risk_score": risk,
                "target_future_return_5m": target,
            }
        )
    return rows


class LabeledRowTests(unittest.TestCase):
    def test_detects_only_current_labels_and_rejects_future_features(self) -> None:
        rows = _rows(30)
        rows[-1]["label_available_time"] = "2027-01-01T00:00:00+00:00"
        rows[-2]["target_future_return_5m"] = ""
        labeled, inventory = discover_available_labeled_rows(
            rows,
            target_name="target_future_return_5m",
            forecast_horizon_seconds=300,
            labels_as_of="2026-02-01T00:00:00+00:00",
        )
        self.assertEqual(len(labeled), 28)
        self.assertEqual(inventory.pending_label_rows, 1)
        self.assertEqual(inventory.missing_or_invalid_label_rows, 1)
        self.assertEqual(len(set(inventory.row_ids)), 28)

        bad = _rows(30)
        timestamp = datetime.fromisoformat(str(bad[5]["timestamp"]))
        bad[5]["available_to_model_time"] = (timestamp + timedelta(seconds=1)).isoformat()
        with self.assertRaises(ResearchValidationError):
            discover_available_labeled_rows(
                bad,
                target_name="target_future_return_5m",
                forecast_horizon_seconds=300,
                labels_as_of="2026-02-01T00:00:00+00:00",
            )

    def test_family_search_is_functional_and_bounded(self) -> None:
        rows, _ = discover_available_labeled_rows(
            _rows(40),
            target_name="target_future_return_5m",
            forecast_horizon_seconds=300,
            labels_as_of="2026-02-01T00:00:00+00:00",
        )
        candidates, unavailable = build_narrow_candidate_configs(
            rows,
            allowed_columns=("price_change", "volatility", "sentiment_score", "risk_score"),
            model_families=("short_horizon_momentum", "mean_reversion", "news_event", "linear_baseline"),
            alphas=(0.1, 1.0),
        )
        self.assertEqual(len(candidates), 8)
        self.assertFalse(unavailable)
        feature_contracts = {candidate.model_family: candidate.feature_columns for candidate in candidates}
        self.assertNotEqual(
            feature_contracts["short_horizon_momentum"],
            feature_contracts["linear_baseline"],
        )
        self.assertEqual(feature_contracts["news_event"], ("sentiment_score", "risk_score"))

    def test_cycle_rejects_position_sizing_targets(self) -> None:
        with self.assertRaisesRegex(ValueError, "future-return forecast"):
            discover_available_labeled_rows(
                _rows(30),
                target_name="target_best_leverage",
                forecast_horizon_seconds=300,
                labels_as_of="2026-02-01T00:00:00+00:00",
            )

    def test_cycle_rejects_target_leakage_in_feature_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(
            ValueError, "cannot be model features"
        ):
            run_narrow_research_cycle(
                _rows(80),
                output_dir=Path(temporary),
                target_name="target_future_return_5m",
                feature_schema_version="price-news-v1",
                allowed_feature_columns=("price_change", "target_future_return_5m"),
                forecast_horizon_seconds=300,
                dataset_version="unit",
                labels_as_of="2026-02-01T00:00:00+00:00",
                model_families=("linear_baseline",),
                alphas=(1.0,),
            )

    def test_news_student_contract_requires_one_homogeneous_student_version(self) -> None:
        contract = observed_news_student_contract(
            [
                {
                    "values": {
                        "local_news_provider": "local_student",
                        "local_news_model_version": "student-train-v3",
                    }
                },
                {
                    "values": {
                        "local_news_provider": "local_rule",
                        "local_news_model_version": "deterministic-news-rules-v1",
                    }
                },
            ]
        )
        self.assertEqual(
            contract,
            {
                "required": True,
                "version": "student-train-v3",
                "external_ai_features_optional": True,
            },
        )
        with self.assertRaisesRegex(ResearchValidationError, "multiple local news student versions"):
            observed_news_student_contract(
                [
                    {
                        "local_news_provider": "local_student",
                        "local_news_model_version": "student-a",
                    },
                    {
                        "local_news_provider": "local_student",
                        "local_news_model_version": "student-b",
                    },
                ]
            )

    def test_packaged_return_model_enforces_observed_news_student_version(self) -> None:
        news_contract = {
            "required": True,
            "version": "student-serving-v2",
            "external_ai_features_optional": True,
        }
        artifact = {
            "artifact_type": "anata_narrow_return_model_v1",
            "contract_version": 1,
            "model_family": "alpha.unit",
            "candidate_id": "candidate-unit",
            "candidate_fingerprint": "f" * 64,
            "feature_schema_version": "runtime-v1",
            "feature_columns": ["momentum"],
            "preprocessing_version": "runtime-v1",
            "training_dataset_version": "unit-data-v1",
            "target_name": "target_future_return_5m",
            "forecast_horizon_seconds": 300,
            "coefficients": [2.0],
            "intercept": 0.001,
            "metrics": {"calibration_score": 0.8},
            "training_period": {
                "start": "2026-01-01T00:00:00+00:00",
                "end": "2026-01-02T00:00:00+00:00",
            },
            "training_rows": 2,
            "missing_value_policy": {"momentum": "required"},
            "allowed_outputs": ["expected_return"],
            "paper_only": True,
            "automatic_promotion": False,
            "news_student_version": news_contract,
        }
        with tempfile.TemporaryDirectory(prefix="anata-news-contract-") as temporary:
            path = Path(temporary) / "candidate.zip"
            package_narrow_artifact(artifact, path)
            validation = verify_narrow_package(path)
            self.assertEqual(validation["news_student_version"], news_contract)
            model = RegisteredArtifactModel(
                model_id="unit-return-model",
                model_family="alpha.unit",
                version="v1",
                required_features=("momentum",),
                artifact_path=str(path),
                expected_feature_schema_version="runtime-v1",
            )
            model._load_artifact()
            now = datetime(2026, 1, 2, tzinfo=timezone.utc)
            matching = FeatureSnapshot(
                symbol="BTCUSDT",
                as_of=now,
                available_to_model_time=now,
                schema_version="runtime-v1",
                values={"momentum": 0.01},
                external_context={
                    "local_news_model_version": "student-serving-v2",
                    "external_ai_missing": True,
                },
            )
            prediction = model.predict(matching)
            self.assertEqual(
                prediction.metadata["required_news_student_version"],
                "student-serving-v2",
            )
            mismatched = matching.model_copy(
                update={
                    "external_context": {
                        "local_news_model_version": "student-other",
                        "external_ai_missing": True,
                    }
                }
            )
            with self.assertRaisesRegex(ArtifactModelError, "LOCAL_NEWS_STUDENT_VERSION_MISMATCH"):
                model.predict(mismatched)


class NarrowResearchCycleTests(unittest.TestCase):
    def test_cycle_evaluates_packages_and_registers_without_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = run_narrow_research_cycle(
                _rows(),
                output_dir=output,
                target_name="target_future_return_5m",
                feature_schema_version="price-news-v1",
                allowed_feature_columns=(
                    "price_change",
                    "volume_change",
                    "volatility",
                    "sentiment_score",
                    "risk_score",
                ),
                forecast_horizon_seconds=300,
                dataset_version="sha256:" + "a" * 64,
                labels_as_of="2026-02-01T00:00:00+00:00",
                transaction_cost=0.0,
                model_families=("short_horizon_momentum", "mean_reversion", "news_event", "linear_baseline"),
                alphas=(0.1, 1.0),
                train_size=60,
                validation_size=10,
                test_size=20,
                step_size=20,
                purge_size=2,
                embargo_size=2,
            )
            self.assertEqual(result["status"], "completed")
            self.assertFalse(result["automatic_promotion"])
            self.assertEqual(result["candidate_count"], 8)
            self.assertEqual(len(result["challenger_packages"]), 4)
            self.assertTrue(all(item["folds"] for item in result["candidates"]))
            self.assertTrue(
                any(fold["purged_indices"] for item in result["candidates"] for fold in item["folds"])
            )
            self.assertTrue(
                any(fold["embargoed_indices"] for item in result["candidates"] for fold in item["folds"])
            )

            first = result["challenger_packages"][0]
            package_path = Path(first["path"])
            self.assertTrue(verify_narrow_package(package_path)["compatible"])
            self.assertTrue(ArtifactValidator().validate(package_path, allow_legacy=False).compatible)
            expected_annualizer = 365.25 * 24 * 60 * 60 / 300
            self.assertAlmostEqual(
                float(result["candidates"][0]["metrics"]["annualization_factor"]),
                expected_annualizer,
            )
            self.assertEqual(len(result["oos_observation_artifacts"]), 4)
            for package in result["challenger_packages"]:
                descriptor = package["oos_observations"]
                oos_path = Path(descriptor["path"])
                self.assertTrue(oos_path.is_file())
                oos_rows = read_research_rows(oos_path)
                self.assertEqual(len(oos_rows), descriptor["rows"])
                self.assertTrue(oos_rows)
                self.assertTrue(all("prediction" in row and "actual_return" in row for row in oos_rows))
                self.assertTrue(
                    all(row["metadata"]["execution"]["calibrated"] is False for row in oos_rows)
                )
                with zipfile.ZipFile(package["path"]) as archive:
                    metadata = json.loads(archive.read("model_metadata.json"))
                self.assertEqual(metadata["oos_observations"]["sha256"], descriptor["sha256"])
                self.assertNotIn("path", metadata["oos_observations"])

            engine = create_engine("sqlite:///:memory:")
            Base.metadata.create_all(engine)
            with Session(engine) as session:
                package_rows: list[ModelVersion] = []
                registry = ModelRegistry(session)
                for package in result["challenger_packages"]:
                    package_rows.append(
                        registry.register(
                            name=str(package["name"]),
                            model_id=str(package["model_id"]),
                            version=str(package["version"]),
                            model_family=str(package["model_family"]),
                            path=str(package["path"]),
                            feature_schema_version=str(package["feature_schema_version"]),
                            feature_columns=list(package["feature_columns"]),
                            lifecycle=ModelLifecycle.TRAINED,
                            metrics=dict(package["metrics"]),
                            preprocessing_version=str(package["preprocessing_version"]),
                            training_dataset_version=str(package["training_dataset_version"]),
                            forecast_horizon_seconds=int(package["forecast_horizon_seconds"]),
                        )
                    )
                session.commit()
                self.assertEqual(session.scalar(select(ChampionAssignment)), None)
                self.assertTrue(all(row.lifecycle_state == "TRAINED" for row in package_rows))
                first_persistence = persist_research_cycle(session, result)
                session.commit()
                first_finished_at = {
                    row.experiment_id: row.finished_at
                    for row in session.scalars(select(ExperimentRun)).all()
                }
                second_persistence = persist_research_cycle(session, result)
                session.commit()
                self.assertEqual(len(first_persistence["research_records"]), 8)
                self.assertTrue(
                    all(item["reused"] for item in second_persistence["registered_challengers"])
                )
                self.assertEqual(
                    session.scalar(select(func.count()).select_from(StrategyCandidate)),
                    8,
                )
                self.assertEqual(
                    session.scalar(select(func.count()).select_from(ExperimentRun)),
                    8,
                )
                self.assertEqual(
                    session.scalar(select(func.count()).select_from(CandidateEvaluation)),
                    8,
                )
                self.assertEqual(
                    session.scalar(select(func.count()).select_from(ModelVersion)),
                    4,
                )
                self.assertEqual(
                    session.scalar(select(func.count()).select_from(ChampionAssignment)),
                    0,
                )
                self.assertTrue(
                    all(
                        len(row.candidate_id) <= 64
                        for row in session.scalars(select(StrategyCandidate)).all()
                    )
                )
                self.assertTrue(
                    all(
                        len(row.experiment_id) <= 64
                        for row in session.scalars(select(ExperimentRun)).all()
                    )
                )
                persisted_experiments = session.scalars(select(ExperimentRun)).all()
                self.assertTrue(
                    all(
                        row.train_period and row.validation_period and row.test_period
                        for row in persisted_experiments
                    )
                )
                self.assertTrue(
                    all(
                        row.configuration.get("fold_periods")
                        and row.validation_period.get("fold_count")
                        == len(row.configuration["fold_periods"])
                        for row in persisted_experiments
                    )
                )
                self.assertEqual(
                    {
                        row.experiment_id: row.finished_at
                        for row in session.scalars(select(ExperimentRun)).all()
                    },
                    first_finished_at,
                )
                # Loading the package proves training and serving agree on the
                # feature order and JSON coefficient contract.
                loaded = RegisteredArtifactModel.from_record(package_rows[0])
                self.assertEqual(tuple(package_rows[0].feature_columns or []), loaded.required_features)

    def test_command_state_skips_an_unchanged_labeled_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "rows.json"
            input_path.write_text(__import__("json").dumps(_rows(80)), encoding="utf-8")
            argv = [
                "--input",
                str(input_path),
                "--output-dir",
                str(root / "research"),
                "--labels-as-of",
                "2026-02-01T00:00:00+00:00",
                "--model-families",
                "linear_baseline",
                "--alphas",
                "1.0",
                "--minimum-new-rows",
                "10",
                "--train-size",
                "30",
                "--validation-size",
                "5",
                "--test-size",
                "10",
                "--step-size",
                "10",
                "--no-register",
            ]
            parser = build_parser()
            first = execute(parser.parse_args(argv))
            second = execute(parser.parse_args(argv))
            self.assertEqual(first["status"], "completed")
            self.assertEqual(second["status"], "waiting_for_labels")
            self.assertEqual(second["new_labeled_rows"], 0)
            self.assertFalse(second["automatic_promotion"])

    def test_optional_upload_fails_before_training_when_token_is_missing(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--input",
                "missing-input.json",
                "--output-dir",
                "research-output",
                "--upload-url",
                "https://anata.example",
                "--upload-token-env",
                "ANATA_TEST_UPLOAD_TOKEN",
            ]
        )
        with patch.dict("os.environ", {}, clear=True), self.assertRaisesRegex(
            ValueError, "ANATA_TEST_UPLOAD_TOKEN"
        ):
            execute(args)


if __name__ == "__main__":
    unittest.main()
