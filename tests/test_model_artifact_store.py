"""Durable cross-role artifact storage and upload-boundary integrity tests."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from fastapi import HTTPException, UploadFile
from sqlalchemy import create_engine, func, inspect, select
from sqlalchemy.orm import Session

from app.ai.model_strategy import PriceModelStrategy
from app.api.routes import upload_model
from app.db.migrations import run_additive_migrations
from app.db.models import (
    Base,
    ChampionAssignment,
    ModelArtifactBlob,
    ModelVersion,
)
from app.intelligence.persistence import build_intelligence_router
from app.intelligence.schemas import NewsDocument
from app.pipeline.artifact_models import ArtifactModelError, RegisteredArtifactModel
from app.pipeline.artifact_store import verify_package_checksum_manifest
from app.pipeline.domain import FeatureSnapshot, ModelLifecycle
from app.pipeline.registry import ModelRegistry
from scripts.package_news_student import main as package_news_student


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _full_package(path: Path, *, coefficient: float = 2.0) -> bytes:
    members = {
        "return_model.json": _json_bytes(
            {
                "feature_columns": ["momentum"],
                "coefficients": [coefficient],
                "intercept": 0.001,
            }
        ),
        "feature_schema.json": _json_bytes(
            {"feature_schema_version": "runtime-v1", "feature_columns": ["momentum"]}
        ),
        "model_metadata.json": _json_bytes(
            {
                "model_file": "return_model.json",
                "feature_schema_version": "runtime-v1",
                "feature_columns": ["momentum"],
            }
        ),
        "training_metrics.json": _json_bytes({"calibration_score": 0.8}),
        "training_period.json": _json_bytes(
            {
                "start": "2026-01-01T00:00:00+00:00",
                "end": "2026-01-02T00:00:00+00:00",
            }
        ),
        "required_features.json": _json_bytes(["momentum"]),
        "optional_features.json": _json_bytes([]),
        "missing_value_policy.json": _json_bytes({"momentum": "required"}),
        "news_student_version.json": _json_bytes({"required": False, "version": None}),
    }
    manifest = {
        "algorithm": "sha256",
        "files": {
            name: {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
            for name, content in sorted(members.items())
        },
    }
    members["checksum_manifest.json"] = _json_bytes(manifest)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(members.items()):
            archive.writestr(name, content)
    content = buffer.getvalue()
    path.write_bytes(content)
    return content


def _remote_package(*, executable_suffix: str | None = None) -> bytes:
    metadata = {
        "model_id": "remote-json-v1",
        "name": "remote-json",
        "version": "v1",
        "model_family": "alpha.remote_json",
        "model_file": "return_model.json",
        "feature_schema_version": "runtime-v1",
        "feature_columns": ["momentum"],
        "preprocessing_version": "runtime-v1",
    }
    members = {
        "metadata.json": _json_bytes(metadata),
        "return_model.json": _json_bytes(
            {
                "feature_columns": ["momentum"],
                "coefficients": [2.0],
                "intercept": 0.001,
            }
        ),
    }
    if executable_suffix is not None:
        members[f"unsafe{executable_suffix}"] = b"not-a-real-pickle"
    members["checksum_manifest.json"] = _json_bytes(
        {
            "algorithm": "sha256",
            "files": {
                name: {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
                for name, content in sorted(members.items())
            },
        }
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(members.items()):
            archive.writestr(name, content)
    return buffer.getvalue()


class DurableArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="anata-artifact-store-")
        self.root = Path(self.temporary.name)
        self.engine = create_engine(f"sqlite:///{(self.root / 'store.sqlite3').as_posix()}")
        Base.metadata.create_all(self.engine)

    def tearDown(self) -> None:
        self.engine.dispose()
        self.temporary.cleanup()

    def _register_package(self) -> tuple[int, Path]:
        package_path = self.root / "candidate.zip"
        _full_package(package_path)
        with Session(self.engine) as session:
            row = ModelRegistry(session).register(
                name="durable-candidate",
                model_id="durable-candidate",
                version="v1",
                model_family="alpha.momentum",
                path=str(package_path),
                feature_schema_version="runtime-v1",
                feature_columns=["momentum"],
                lifecycle=ModelLifecycle.TRAINED,
                preprocessing_version="runtime-v1",
            )
            model_version_id = int(row.id)
            session.commit()
        return model_version_id, package_path

    @staticmethod
    def _snapshot() -> FeatureSnapshot:
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        return FeatureSnapshot(
            symbol="BTCUSDT",
            as_of=now,
            available_to_model_time=now,
            schema_version="runtime-v1",
            values={"momentum": 0.01},
        )

    def test_new_session_loads_db_bytes_after_uploader_path_is_lost(self) -> None:
        model_version_id, package_path = self._register_package()
        package_path.unlink()

        with Session(self.engine) as runtime_session:
            row = runtime_session.get(ModelVersion, model_version_id)
            model = RegisteredArtifactModel.from_record(row)
            prediction = model.predict(self._snapshot())
            blob = runtime_session.scalar(
                select(ModelArtifactBlob).where(
                    ModelArtifactBlob.model_version_id == model_version_id
                )
            )
            self.assertIsNotNone(blob)
            self.assertEqual(blob.sha256, row.artifact_checksum)
            self.assertAlmostEqual(prediction.expected_return, 0.021)
            self.assertNotEqual(Path(model.artifact_path), package_path)
            self.assertTrue(Path(model.artifact_path).is_file())
            self.assertEqual(row.lifecycle_state, ModelLifecycle.TRAINED.value)
            self.assertEqual(
                runtime_session.scalar(select(func.count()).select_from(ChampionAssignment)),
                0,
            )

    def test_outer_checksum_tamper_is_rejected(self) -> None:
        model_version_id, package_path = self._register_package()
        package_path.unlink()
        with Session(self.engine) as session:
            blob = session.scalar(
                select(ModelArtifactBlob).where(
                    ModelArtifactBlob.model_version_id == model_version_id
                )
            )
            blob.content = bytes(blob.content) + b"tamper"
            blob.size_bytes = len(blob.content)
            session.commit()
        with Session(self.engine) as session:
            with self.assertRaisesRegex(ArtifactModelError, "ARTIFACT_CHECKSUM_MISMATCH"):
                RegisteredArtifactModel.from_record(session.get(ModelVersion, model_version_id))

    def test_internal_manifest_tamper_is_rejected_even_if_outer_sha_is_rewritten(self) -> None:
        model_version_id, package_path = self._register_package()
        package_path.unlink()
        with Session(self.engine) as session:
            row = session.get(ModelVersion, model_version_id)
            blob = session.scalar(
                select(ModelArtifactBlob).where(
                    ModelArtifactBlob.model_version_id == model_version_id
                )
            )
            with zipfile.ZipFile(io.BytesIO(bytes(blob.content))) as source:
                members = {name: source.read(name) for name in source.namelist()}
            model_payload = json.loads(members["return_model.json"])
            model_payload["intercept"] = 999.0
            members["return_model.json"] = _json_bytes(model_payload)
            output = io.BytesIO()
            with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, content in sorted(members.items()):
                    archive.writestr(name, content)
            tampered = output.getvalue()
            rewritten_sha = hashlib.sha256(tampered).hexdigest()
            blob.content = tampered
            blob.size_bytes = len(tampered)
            blob.sha256 = rewritten_sha
            row.artifact_checksum = rewritten_sha
            session.commit()
        with Session(self.engine) as session:
            with self.assertRaisesRegex(ArtifactModelError, "PACKAGE_MEMBER_CHECKSUM_MISMATCH"):
                RegisteredArtifactModel.from_record(session.get(ModelVersion, model_version_id))

    def test_registry_blob_and_model_row_rollback_together(self) -> None:
        path = self.root / "rollback.zip"
        _full_package(path)
        with Session(self.engine) as session:
            ModelRegistry(session).register(
                name="rollback",
                model_id="rollback",
                version="v1",
                model_family="alpha.rollback",
                path=str(path),
                feature_schema_version="runtime-v1",
                feature_columns=["momentum"],
            )
            session.rollback()
        with Session(self.engine) as session:
            self.assertEqual(session.scalar(select(func.count()).select_from(ModelVersion)), 0)
            self.assertEqual(session.scalar(select(func.count()).select_from(ModelArtifactBlob)), 0)


class RemoteUploadSecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="anata-upload-security-")
        self.root = Path(self.temporary.name)
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.session = Session(self.engine)

    def tearDown(self) -> None:
        self.session.close()
        self.engine.dispose()
        self.temporary.cleanup()

    def test_remote_upload_rejects_pickle_and_joblib_members(self) -> None:
        for suffix in (".pkl", ".pickle", ".joblib"):
            with self.subTest(suffix=suffix), patch(
                "app.api.routes.settings",
                SimpleNamespace(model_dir=self.root),
            ):
                upload = UploadFile(
                    filename=f"unsafe-{suffix[1:]}.zip",
                    file=io.BytesIO(_remote_package(executable_suffix=suffix)),
                )
                with self.assertRaisesRegex(HTTPException, "declarative JSON only"):
                    upload_model(upload, self.session)
                self.session.rollback()
        self.assertEqual(self.session.scalar(select(func.count()).select_from(ModelVersion)), 0)
        self.assertEqual(self.session.scalar(select(func.count()).select_from(ModelArtifactBlob)), 0)

    def test_remote_upload_rejects_non_json_package_members(self) -> None:
        with patch(
            "app.api.routes.settings",
            SimpleNamespace(model_dir=self.root),
        ):
            upload = UploadFile(
                filename="unsafe-code.zip",
                file=io.BytesIO(_remote_package(executable_suffix=".py")),
            )
            with self.assertRaisesRegex(HTTPException, "JSON members only"):
                upload_model(upload, self.session)
        self.assertEqual(self.session.scalar(select(func.count()).select_from(ModelVersion)), 0)

    def test_remote_upload_requires_checksum_manifest(self) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(
                "metadata.json",
                json.dumps(
                    {
                        "model_file": "return_model.json",
                        "feature_columns": ["momentum"],
                    }
                ),
            )
            archive.writestr(
                "return_model.json",
                json.dumps(
                    {
                        "feature_columns": ["momentum"],
                        "coefficients": [1.0],
                    }
                ),
            )
        with patch(
            "app.api.routes.settings",
            SimpleNamespace(model_dir=self.root),
        ):
            with self.assertRaisesRegex(HTTPException, "PACKAGE_CHECKSUM_MANIFEST_MISSING"):
                upload_model(
                    UploadFile(filename="missing-manifest.zip", file=io.BytesIO(buffer.getvalue())),
                    self.session,
                )
        self.assertEqual(self.session.scalar(select(func.count()).select_from(ModelVersion)), 0)

    def test_safe_json_upload_is_durable_but_not_activated(self) -> None:
        with patch(
            "app.api.routes.settings",
            SimpleNamespace(model_dir=self.root),
        ):
            upload = UploadFile(
                filename="safe-json.zip",
                file=io.BytesIO(_remote_package()),
            )
            result = upload_model(upload, self.session)
        row = self.session.get(ModelVersion, int(result["model"]["id"]))
        original_path = Path(row.path)
        self.assertEqual(result["status"], "candidate")
        self.assertEqual(row.lifecycle_state, ModelLifecycle.TRAINED.value)
        self.assertEqual(self.session.scalar(select(func.count()).select_from(ModelArtifactBlob)), 1)
        self.assertEqual(self.session.scalar(select(func.count()).select_from(ChampionAssignment)), 0)
        original_path.unlink()
        model_id = row.id
        self.session.close()
        self.session = Session(self.engine)
        runtime_row = self.session.get(ModelVersion, model_id)
        legacy_payload = PriceModelStrategy()._load_model_payload(runtime_row)
        self.assertEqual(legacy_payload["coefficients"], [2.0])
        prediction = RegisteredArtifactModel.from_record(runtime_row).predict(
            DurableArtifactTests._snapshot()
        )
        self.assertAlmostEqual(prediction.expected_return, 0.021)

    def test_packaged_news_student_can_enter_safe_trained_lifecycle(self) -> None:
        artifact_path = self.root / "news-student.json"
        artifact_path.write_text(
            json.dumps(
                {
                    "artifact_type": "anata_news_student_naive_bayes_v1",
                    "version": "news-unit-v1",
                    "training_rows": 2,
                    "training_period": {},
                    "teacher_versions": ["teacher-unit"],
                    "vocabulary_size": 1,
                    "tasks": {
                        "sentiment": {"label_counts": {"neutral": 2}},
                        "event_type": {"label_counts": {"other": 2}},
                    },
                }
            ),
            encoding="utf-8",
        )
        output_dir = self.root / "news-student-package"
        with patch(
            "sys.argv",
            [
                "package_news_student.py",
                "--artifact",
                str(artifact_path),
                "--output-dir",
                str(output_dir),
            ],
        ), redirect_stdout(io.StringIO()):
            package_news_student()
        upload_package = output_dir.with_suffix(".zip")
        self.assertTrue(upload_package.is_file())
        verified = verify_package_checksum_manifest(
            upload_package.read_bytes(),
            require_manifest=True,
        )
        self.assertEqual(verified["package_type"], "anata_local_news_student_package_v1")

        with patch(
            "app.api.routes.settings",
            SimpleNamespace(model_dir=self.root / "uploaded-news"),
        ):
            result = upload_model(
                UploadFile(
                    filename=upload_package.name,
                    file=io.BytesIO(upload_package.read_bytes()),
                ),
                self.session,
            )
        row = self.session.get(ModelVersion, int(result["model"]["id"]))
        self.assertEqual(row.model_family, "intelligence.news_student_naive_bayes")
        self.assertEqual(row.lifecycle_state, ModelLifecycle.TRAINED.value)
        self.assertEqual(self.session.scalar(select(func.count()).select_from(ModelArtifactBlob)), 1)
        self.assertEqual(self.session.scalar(select(func.count()).select_from(ChampionAssignment)), 0)

        registry = ModelRegistry(self.session)
        with self.assertRaisesRegex(ValueError, "context-only"):
            registry.start_sandbox(row.id)
        with self.assertRaisesRegex(ValueError, "context-only"):
            registry.start_shadow(row.id)
        with self.assertRaisesRegex(ValueError, "wildcard"):
            registry.promote(
                row.id,
                model_family="intelligence.news_student_naive_bayes",
                symbol_scope="BTCUSDT",
            )

        registry.promote(
            row.id,
            model_family="intelligence.news_student_naive_bayes",
            symbol_scope="*",
            actor="manual",
            reason="reviewed unit-test activation",
        )
        self.session.commit()
        local_origin = Path(row.path)
        self.assertTrue(local_origin.is_file())
        local_origin.unlink()
        model_version_id = row.id
        self.session.close()
        self.session = Session(self.engine)

        runtime_row = self.session.get(ModelVersion, model_version_id)
        router = build_intelligence_router(self.session)
        result = asyncio.run(
            router.analyze(
                NewsDocument(
                    title="Bitcoin context",
                    content="Bitcoin market context for the durable student.",
                    source="unit-test",
                )
            )
        )
        self.assertEqual(runtime_row.lifecycle_state, ModelLifecycle.CHAMPION.value)
        self.assertEqual(router.local_provider.name, "local_student")
        self.assertEqual(result.local_event.provider, "local_student")
        self.assertEqual(result.local_event.model, "news-unit-v1")
        self.assertNotEqual(Path(router.local_provider.artifact_path), local_origin)


class ArtifactMigrationTests(unittest.TestCase):
    def test_durable_artifact_table_migration_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory(prefix="anata-artifact-migration-") as temporary:
            engine = create_engine(f"sqlite:///{(Path(temporary) / 'legacy.sqlite3').as_posix()}")
            try:
                ModelVersion.__table__.create(engine)
                first = run_additive_migrations(engine)
                second = run_additive_migrations(engine)
                tables = set(inspect(engine).get_table_names())
            finally:
                engine.dispose()
        self.assertIn("model_artifact_blobs", tables)
        self.assertEqual(first["created_tables"], ["model_artifact_blobs"])
        self.assertEqual(second["created_tables"], [])


if __name__ == "__main__":
    unittest.main()
