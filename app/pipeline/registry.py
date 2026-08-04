"""Model lifecycle, champion/challenger, shadow and sandbox registry services."""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import case, desc, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import (
    ChampionAssignment,
    ModelVersion,
    PaperSandboxAccount,
    PromotionDecision,
    ShadowPrediction,
)
from app.pipeline.domain import ModelLifecycle, ModelPrediction
from app.pipeline.artifact_models import ArtifactModelError, RegisteredArtifactModel
from app.pipeline.artifact_store import (
    ArtifactIntegrityError,
    ModelArtifactStore,
    resolve_model_artifact,
    verify_artifact_bytes,
)


REQUIRED_PACKAGE_FILES = {
    "feature_schema.json",
    "model_metadata.json",
    "training_metrics.json",
    "training_period.json",
    "required_features.json",
    "optional_features.json",
    "missing_value_policy.json",
    "news_student_version.json",
    "checksum_manifest.json",
}


@dataclass(frozen=True)
class ArtifactValidation:
    compatible: bool
    checksum: str | None
    manifest: dict[str, Any]
    errors: tuple[str, ...] = ()
    legacy_compatible: bool = False


class ArtifactValidator:
    """Validate an artifact contract before a candidate reaches shadow/sandbox."""

    def validate(self, path: str | Path, *, allow_legacy: bool = True) -> ArtifactValidation:
        target = Path(path)
        if not target.exists() or not target.is_file():
            return ArtifactValidation(False, None, {}, ("ARTIFACT_NOT_FOUND",))
        checksum = self.checksum(target)
        if target.suffix.lower() == ".json":
            try:
                payload = json.loads(target.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return ArtifactValidation(False, checksum, {}, ("INVALID_ARTIFACT_JSON",))
            required_columns = payload.get("feature_columns")
            if isinstance(required_columns, list):
                return ArtifactValidation(True, checksum, {"legacy": True, "feature_columns": required_columns}, legacy_compatible=True)
            return ArtifactValidation(False, checksum, payload if isinstance(payload, dict) else {}, ("MISSING_FEATURE_COLUMNS",))
        if target.suffix.lower() == ".zip":
            try:
                content = target.read_bytes()
                package_integrity = verify_artifact_bytes(
                    content,
                    filename=target.name,
                    expected_checksum=checksum,
                    require_package_manifest=False,
                )
                with zipfile.ZipFile(target) as archive:
                    names = set(archive.namelist())
                    missing = sorted(REQUIRED_PACKAGE_FILES - names)
                    manifest: dict[str, Any] = {
                        "package_files": sorted(names),
                        "missing_required_package_files": missing,
                        "checksum_manifest_verified": package_integrity[
                            "package_checksum_manifest"
                        ]
                        is not None,
                    }
                    if "model_metadata.json" in names:
                        metadata = json.loads(archive.read("model_metadata.json").decode("utf-8"))
                        if isinstance(metadata, dict):
                            manifest["model_metadata"] = metadata
                    model_members = [
                        name
                        for name in names
                        if Path(name).suffix.lower() in {".joblib", ".pkl", ".pickle", ".json"}
                        and Path(name).name not in REQUIRED_PACKAGE_FILES
                    ]
                    if not model_members:
                        missing.append("MODEL_ARTIFACT")
                    manifest["model_artifacts"] = sorted(model_members)
                    if not missing:
                        return ArtifactValidation(True, checksum, manifest)
                    if allow_legacy and model_members:
                        return ArtifactValidation(
                            True,
                            checksum,
                            {**manifest, "legacy": True},
                            tuple(f"MISSING_PACKAGE_FILE:{name}" for name in missing),
                            legacy_compatible=True,
                        )
                    return ArtifactValidation(
                        False,
                        checksum,
                        manifest,
                        tuple(f"MISSING_PACKAGE_FILE:{name}" for name in missing),
                    )
            except ArtifactIntegrityError as exc:
                return ArtifactValidation(False, checksum, {}, (str(exc),))
            except (OSError, zipfile.BadZipFile, UnicodeDecodeError, json.JSONDecodeError):
                return ArtifactValidation(False, checksum, {}, ("INVALID_ARTIFACT_PACKAGE",))
        # Standalone joblib packages remain readable for existing deployments when
        # the ModelVersion row supplies the schema/preprocessing contract. Loading is
        # deferred until a shadow, sandbox, or champion transition is requested.
        return ArtifactValidation(
            compatible=allow_legacy,
            checksum=checksum,
            manifest={"legacy": True, "artifact_path": str(target)},
            errors=() if allow_legacy else ("MISSING_PACKAGE_MANIFEST",),
            legacy_compatible=allow_legacy,
        )

    @staticmethod
    def checksum(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


class ModelRegistry:
    """Persist model lifecycle changes without automatic champion replacement."""

    def __init__(self, session: Session, *, validator: ArtifactValidator | None = None) -> None:
        self.session = session
        self.validator = validator or ArtifactValidator()

    def register(
        self,
        *,
        name: str,
        model_id: str,
        version: str,
        model_family: str,
        path: str,
        feature_schema_version: str,
        feature_columns: list[str],
        lifecycle: ModelLifecycle = ModelLifecycle.TRAINED,
        metrics: dict[str, Any] | None = None,
        preprocessing_version: str = "v1",
        training_dataset_version: str | None = None,
        forecast_horizon_seconds: int = 300,
        parent_model_id: str | None = None,
        raw_payload: dict[str, Any] | None = None,
    ) -> ModelVersion:
        if lifecycle == ModelLifecycle.CHAMPION:
            raise ValueError("Register candidates before using the explicit promotion workflow")
        validation = self.validator.validate(path)
        if not validation.compatible:
            raise ValueError("Model artifact is incompatible: " + ", ".join(validation.errors))
        row = ModelVersion(
            name=name,
            model_id=model_id,
            version=version,
            model_family=model_family,
            path=path,
            feature_schema_version=feature_schema_version,
            feature_columns=feature_columns,
            status="candidate" if lifecycle != ModelLifecycle.CHAMPION else "active",
            lifecycle_state=lifecycle.value,
            health_status="HEALTHY",
            artifact_checksum=validation.checksum,
            preprocessing_version=preprocessing_version,
            training_dataset_version=training_dataset_version,
            forecast_horizon_seconds=forecast_horizon_seconds,
            parent_model_id=parent_model_id,
            metrics=metrics or {},
            package_manifest=validation.manifest,
            raw_payload={
                **dict(raw_payload or {}),
                "artifact_validation": {
                    "legacy_compatible": validation.legacy_compatible,
                    "errors": list(validation.errors),
                },
            },
        )
        self.session.add(row)
        self.session.flush()
        self.store_artifact(row, path)
        return row

    def store_artifact(self, model: ModelVersion, path: str | Path | None = None) -> None:
        """Idempotently attach artifact bytes to ``model`` in this transaction."""

        try:
            ModelArtifactStore(self.session).put_path(model, path or model.path)
        except ArtifactIntegrityError as exc:
            raise ValueError(f"Model artifact could not be stored safely: {exc}") from exc

    def champion(self, *, model_family: str, symbol: str) -> ModelVersion | None:
        assignment = self.session.scalar(
            select(ChampionAssignment)
            .where(
                ChampionAssignment.model_family == model_family,
                ChampionAssignment.active_to.is_(None),
                ChampionAssignment.symbol_scope.in_((symbol.upper(), "*")),
            )
            .order_by(
                case((ChampionAssignment.symbol_scope == symbol.upper(), 0), else_=1),
                desc(ChampionAssignment.active_from),
            )
            .limit(1)
        )
        return self.session.get(ModelVersion, assignment.model_version_id) if assignment else None

    def promote(self, model_version_id: int, *, model_family: str, symbol_scope: str = "*", actor: str = "manual", reason: str | None = None) -> ModelVersion:
        """Explicitly promote one technically compatible model and record rollback data."""
        if actor != "manual" and not settings.v2_auto_promote_champion:
            raise PermissionError("Automatic champion promotion is disabled by default.")
        candidate = self.session.get(ModelVersion, model_version_id)
        if candidate is None:
            raise ValueError("Model version does not exist")
        if candidate.model_family and candidate.model_family != model_family:
            raise ValueError("Promotion family must match the registered model family")
        if model_family == "intelligence.news_student_naive_bayes" and symbol_scope != "*":
            raise ValueError("The context-only news student must be activated with wildcard scope")
        if candidate.lifecycle_state in {ModelLifecycle.SUSPENDED.value, ModelLifecycle.RETIRED.value}:
            raise ValueError("Suspended or retired models cannot become champion")
        if str(candidate.health_status or "").upper() in {"SUSPENDED", "RETIRED"}:
            raise ValueError("A model with suspended or retired health cannot become champion")
        self._validate_runtime_model(candidate, action="promote")
        previous_rows = list(self.session.scalars(
            select(ChampionAssignment)
            .where(
                ChampionAssignment.model_family == model_family,
                ChampionAssignment.symbol_scope == symbol_scope,
                ChampionAssignment.active_to.is_(None),
            )
            .order_by(desc(ChampionAssignment.active_from))
        ))
        previous = previous_rows[0] if previous_rows else None
        now = datetime.now(timezone.utc)
        previous_id = previous.model_version_id if previous else None
        for assignment in previous_rows:
            assignment.active_to = now
            old = self.session.get(ModelVersion, assignment.model_version_id)
            if old:
                old.lifecycle_state = ModelLifecycle.TRAINED.value
                old.status = "candidate"
        candidate.lifecycle_state = ModelLifecycle.CHAMPION.value
        candidate.status = "active"
        self.session.add(
            ChampionAssignment(
                model_version_id=candidate.id,
                model_family=model_family,
                symbol_scope=symbol_scope,
                active_from=now,
                assigned_by=actor,
                reason=reason,
            )
        )
        self.session.add(
            PromotionDecision(
                model_version_id=candidate.id,
                previous_model_version_id=previous_id,
                action="PROMOTE",
                approved=True,
                decided_by=actor,
                reason=reason,
                payload={"manual": actor == "manual"},
            )
        )
        self.session.flush()
        return candidate

    def rollback(self, *, model_family: str, symbol_scope: str = "*", actor: str = "manual", reason: str | None = None) -> ModelVersion:
        current = self.session.scalar(
            select(ChampionAssignment)
            .where(
                ChampionAssignment.model_family == model_family,
                ChampionAssignment.symbol_scope == symbol_scope,
                ChampionAssignment.active_to.is_(None),
            )
            .order_by(desc(ChampionAssignment.active_from))
            .limit(1)
        )
        if current is None:
            raise ValueError("No champion exists to roll back")
        prior = self.session.scalar(
            select(PromotionDecision)
            .where(PromotionDecision.model_version_id == current.model_version_id, PromotionDecision.previous_model_version_id.is_not(None))
            .order_by(desc(PromotionDecision.created_at))
            .limit(1)
        )
        if prior is None or prior.previous_model_version_id is None:
            raise ValueError("No previous champion is recorded for rollback")
        current.active_to = datetime.now(timezone.utc)
        current_model = self.session.get(ModelVersion, current.model_version_id)
        if current_model:
            current_model.lifecycle_state = ModelLifecycle.TRAINED.value
            current_model.status = "candidate"
        restored = self.session.get(ModelVersion, prior.previous_model_version_id)
        if restored is None:
            raise ValueError("Previous champion model record is unavailable")
        restored.lifecycle_state = ModelLifecycle.CHAMPION.value
        restored.status = "active"
        self.session.add(
            ChampionAssignment(
                model_version_id=restored.id,
                model_family=model_family,
                symbol_scope=symbol_scope,
                assigned_by=actor,
                reason=reason or "rollback",
            )
        )
        self.session.add(
            PromotionDecision(
                model_version_id=restored.id,
                previous_model_version_id=current.model_version_id,
                action="ROLLBACK",
                approved=True,
                decided_by=actor,
                reason=reason,
            )
        )
        self.session.flush()
        return restored

    def start_sandbox(self, model_version_id: int, *, name: str | None = None, starting_balance: float | None = None) -> PaperSandboxAccount:
        """Create an isolated fake account after technical—not profitability—checks."""
        model = self.session.get(ModelVersion, model_version_id)
        if model is None:
            raise ValueError("Model version does not exist")
        if model.model_family == "intelligence.news_student_naive_bayes":
            raise ValueError("The news student is context-only and cannot enter a trading sandbox")
        if model.lifecycle_state in {ModelLifecycle.SUSPENDED.value, ModelLifecycle.RETIRED.value}:
            raise ValueError("Suspended or retired models cannot enter a sandbox")
        validation = self._validate_runtime_model(model, action="start sandbox")
        model.lifecycle_state = ModelLifecycle.PAPER_SANDBOX.value
        account = PaperSandboxAccount(
            account_id=f"sandbox-{model.id}-{uuid4().hex[:10]}",
            name=name or f"sandbox-{model.name}-{model.version}",
            model_version_id=model.id,
            starting_balance=starting_balance if starting_balance is not None else settings.paper_start_balance,
            max_exposure_pct=settings.v2_sandbox_max_exposure_pct,
            active=True,
            payload={"technical_validation": validation.manifest, "profitability_gate_required": False, "paper_only": True},
        )
        self.session.add(account)
        self.session.flush()
        return account

    def start_shadow(self, model_version_id: int) -> ModelVersion:
        """Move a technically valid challenger into non-executing shadow inference."""
        model = self.session.get(ModelVersion, model_version_id)
        if model is None:
            raise ValueError("Model version does not exist")
        if model.model_family == "intelligence.news_student_naive_bayes":
            raise ValueError("The news student is context-only and cannot enter trading-model shadow inference")
        if model.lifecycle_state in {ModelLifecycle.SUSPENDED.value, ModelLifecycle.RETIRED.value}:
            raise ValueError("Suspended or retired models cannot enter shadow mode")
        self._validate_runtime_model(model, action="start shadow")
        model.lifecycle_state = ModelLifecycle.SHADOW.value
        model.status = "candidate"
        self.session.flush()
        return model

    def record_shadow(self, prediction: ModelPrediction, *, model_version_id: int, decision_trace_id: str) -> ShadowPrediction:
        model = self.session.get(ModelVersion, model_version_id)
        if model is None:
            raise ValueError("Shadow model version does not exist")
        if model.lifecycle_state != ModelLifecycle.SHADOW.value:
            raise ValueError("Only a registered SHADOW model may write shadow predictions")
        if model.model_id and model.model_id != prediction.model_id:
            raise ValueError("Shadow prediction model_id does not match the registered model")
        if model.version != prediction.model_version:
            raise ValueError("Shadow prediction version does not match the registered model")
        if not math.isfinite(prediction.expected_return) or not math.isfinite(prediction.uncertainty):
            raise ValueError("Shadow prediction must be finite")
        row = ShadowPrediction(
            model_version_id=model_version_id,
            prediction_id=prediction.prediction_id,
            decision_trace_id=decision_trace_id,
            symbol=prediction.symbol,
            generated_at=prediction.generated_at,
            payload=prediction.model_dump(mode="json"),
        )
        self.session.add(row)
        self.session.flush()
        return row

    def _validate_runtime_model(self, model: ModelVersion, *, action: str) -> ArtifactValidation:
        try:
            runtime_path = resolve_model_artifact(model, session=self.session)
            # Lifecycle checks opportunistically backfill trusted pre-store rows
            # while their original local artifact is still available.
            self.store_artifact(model, runtime_path)
        except ArtifactIntegrityError as exc:
            raise ValueError(f"Cannot {action}: artifact integrity check failed ({exc})") from exc
        validation = self.validator.validate(runtime_path)
        if not validation.compatible:
            raise ValueError(f"Cannot {action}: incompatible model artifact ({', '.join(validation.errors)})")
        if not model.feature_schema_version:
            raise ValueError(f"Cannot {action}: feature schema version is missing")
        if not model.preprocessing_version:
            raise ValueError(f"Cannot {action}: preprocessing version is missing")
        if not model.feature_columns:
            raise ValueError(f"Cannot {action}: required feature columns are missing")
        if model.model_family == "intelligence.news_student_naive_bayes":
            try:
                from app.intelligence.providers import load_json_student_artifact

                raw_payload = model.raw_payload if isinstance(model.raw_payload, dict) else {}
                load_json_student_artifact(
                    runtime_path,
                    artifact_member=str(
                        raw_payload.get("model_member")
                        or raw_payload.get("model_file")
                        or "student_artifact.json"
                    ),
                )
            except Exception as exc:
                raise ValueError(
                    f"Cannot {action}: news student artifact load check failed ({type(exc).__name__}: {exc})"
                ) from exc
            return validation
        try:
            RegisteredArtifactModel.from_record(model)
        except ArtifactModelError as exc:
            raise ValueError(f"Cannot {action}: artifact load check failed ({exc})") from exc
        return validation
