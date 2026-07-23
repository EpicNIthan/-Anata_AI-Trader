"""Frozen registered-artifact adapters for the execution-independent model layer.

The adapter intentionally exposes only :class:`ModelPrediction` inputs through the
``NarrowModel`` contract.  Legacy packages may contain leverage, margin, stop, or exit
estimators; those files are never consulted here.  Only the declared return-forecast
artifact is loaded.
"""

from __future__ import annotations

import io
import json
import math
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING

from app.pipeline.domain import FeatureSnapshot, HealthStatus, ModelPrediction
from app.pipeline.narrow_models import NarrowModel, PredictionDistribution, _clamp, _sigmoid, _value

if TYPE_CHECKING:  # Keep the forecasting boundary usable without importing the ORM.
    from app.db.models import ModelVersion


class ArtifactModelError(RuntimeError):
    """Raised when a frozen registry artifact cannot make a valid forecast."""


def _finite(value: Any, *, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactModelError(f"{field_name} is not numeric") from exc
    if not math.isfinite(result):
        raise ArtifactModelError(f"{field_name} must be finite")
    return result


def _first_prediction(value: Any) -> float:
    """Extract one scalar from common sklearn/numpy prediction shapes."""
    try:
        first = value[0]
    except (TypeError, IndexError, KeyError) as exc:
        raise ArtifactModelError("artifact predict() returned no values") from exc
    while isinstance(first, (list, tuple)) and first:
        first = first[0]
    if hasattr(first, "item"):
        try:
            first = first.item()
        except (TypeError, ValueError):
            pass
    return _finite(first, field_name="predicted return")


@dataclass
class RegisteredArtifactModel(NarrowModel):
    """A frozen JSON/joblib return model registered as one narrow forecaster.

    ``feature_columns`` are both the required feature order and the serving contract.
    Model packages that also contain sizing targets remain compatible, but those
    targets are deliberately ignored.
    """

    artifact_path: str = ""
    expected_feature_schema_version: str = ""
    preprocessing_version: str = "v1"
    registry_model_version_id: int | None = None
    artifact_checksum: str | None = None
    artifact_member: str | None = None
    calibration_score: float = 0.50
    metrics: Mapping[str, Any] = field(default_factory=dict)
    _payload: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _estimator: Any = field(default=None, init=False, repr=False)

    @classmethod
    def from_record(cls, row: "ModelVersion") -> "RegisteredArtifactModel":
        columns = tuple(str(item) for item in (row.feature_columns or []) if str(item).strip())
        if not columns:
            raise ArtifactModelError("registered artifact has no feature column contract")
        if not row.path:
            raise ArtifactModelError("registered artifact path is empty")
        metrics = dict(row.metrics or {})
        calibration = metrics.get("calibration_score", metrics.get("calibration", 0.50))
        try:
            calibration_score = _clamp(float(calibration), 0.0, 1.0)
        except (TypeError, ValueError):
            calibration_score = 0.50
        health_name = str(row.health_status or "HEALTHY").upper()
        try:
            health = HealthStatus(health_name)
        except ValueError:
            health = HealthStatus.WATCH
        instance = cls(
            model_id=row.model_id or f"registered-{row.id}",
            model_family=row.model_family or "alpha.registered_return",
            version=row.version,
            required_features=columns,
            optional_features=(),
            forecast_horizon=max(int(row.forecast_horizon_seconds or 300), 1),
            artifact_path=row.path,
            expected_feature_schema_version=row.feature_schema_version or "",
            preprocessing_version=row.preprocessing_version or "v1",
            registry_model_version_id=row.id,
            artifact_checksum=row.artifact_checksum,
            artifact_member=(row.raw_payload or {}).get("model_file") if isinstance(row.raw_payload, dict) else None,
            calibration_score=calibration_score,
            metrics=metrics,
        )
        instance._health = health
        instance._load_artifact()
        return instance

    def validate_inputs(self, snapshot: FeatureSnapshot) -> list[str]:
        missing = super().validate_inputs(snapshot)
        if self.expected_feature_schema_version and snapshot.schema_version != self.expected_feature_schema_version:
            missing.append(
                f"FEATURE_SCHEMA_MISMATCH:{snapshot.schema_version}!={self.expected_feature_schema_version}"
            )
        for name in self.required_features:
            value = snapshot.values.get(name)
            try:
                if value is None or not math.isfinite(float(value)):
                    missing.append(name)
            except (TypeError, ValueError):
                missing.append(name)
        return sorted(set(missing))

    def predict_distribution(self, snapshot: FeatureSnapshot) -> PredictionDistribution:
        invalid = self.validate_inputs(snapshot)
        if invalid:
            raise ArtifactModelError("registered artifact input validation failed: " + ", ".join(invalid))
        vector = [_finite(snapshot.values[name], field_name=name) for name in self.required_features]
        predicted_return = self._predict_return(vector)
        volatility = max(abs(_value(snapshot.values, "volatility")), abs(predicted_return) * 0.6, 0.0001)
        probability_up = _sigmoid(predicted_return / max(volatility, 1e-9))
        uncertainty_metric = self.metrics.get("uncertainty", 1.0 - self.calibration_score)
        try:
            uncertainty = _clamp(float(uncertainty_metric), 0.05, 0.95)
        except (TypeError, ValueError):
            uncertainty = _clamp(1.0 - self.calibration_score, 0.05, 0.95)
        return PredictionDistribution(
            expected_return=predicted_return,
            expected_volatility=volatility,
            probability_up=probability_up,
            probability_down=1.0 - probability_up,
            uncertainty=uncertainty,
        )

    def predict(self, snapshot: FeatureSnapshot) -> ModelPrediction:
        prediction = super().predict(snapshot)
        return prediction.model_copy(
            update={
                "calibration_score": self.calibration_score,
                "metadata": {
                    **prediction.metadata,
                    "baseline": False,
                    "registered_model_version_id": self.registry_model_version_id,
                    "artifact_checksum": self.artifact_checksum,
                    "preprocessing_version": self.preprocessing_version,
                    "legacy_sizing_targets_ignored": True,
                },
            }
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_family": self.model_family,
            "version": self.version,
            "required_features": list(self.required_features),
            "optional_features": list(self.optional_features),
            "forecast_horizon": self.forecast_horizon,
            "baseline": False,
            "registered_model_version_id": self.registry_model_version_id,
            "preprocessing_version": self.preprocessing_version,
            "legacy_sizing_targets_ignored": True,
        }

    def _load_artifact(self) -> None:
        path = Path(self.artifact_path)
        if not path.is_file():
            raise ArtifactModelError(f"registered artifact does not exist: {path}")
        suffix = path.suffix.lower()
        if suffix == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ArtifactModelError("registered JSON artifact is invalid") from exc
            if not isinstance(payload, dict):
                raise ArtifactModelError("registered JSON artifact must contain an object")
            self._payload = payload
            self._validate_json_payload()
            return
        if suffix == ".zip":
            self._load_zip(path)
            return
        if suffix in {".joblib", ".pkl", ".pickle"}:
            self._estimator = self._joblib_load(path)
            self._validate_estimator()
            return
        raise ArtifactModelError(f"unsupported registered artifact type: {suffix or '<none>'}")

    def _load_zip(self, path: Path) -> None:
        try:
            with zipfile.ZipFile(path) as archive:
                names = set(archive.namelist())
                metadata: dict[str, Any] = {}
                if "model_metadata.json" in names:
                    raw = json.loads(archive.read("model_metadata.json").decode("utf-8"))
                    metadata = raw if isinstance(raw, dict) else {}
                declared = self.artifact_member or metadata.get("model_file") or metadata.get("artifact_path")
                candidates = []
                if declared:
                    candidates.extend([str(declared), Path(str(declared)).name])
                candidates.extend(
                    name
                    for name in sorted(names)
                    if Path(name).suffix.lower() in {".joblib", ".pkl", ".pickle", ".json"}
                    and Path(name).name not in {
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
                )
                member = next((name for name in candidates if name in names), None)
                if member is None:
                    raise ArtifactModelError("registered package contains no return-model artifact")
                raw_artifact = archive.read(member)
                member_suffix = Path(member).suffix.lower()
                if member_suffix == ".json":
                    payload = json.loads(raw_artifact.decode("utf-8"))
                    if not isinstance(payload, dict):
                        raise ArtifactModelError("packaged JSON model must contain an object")
                    self._payload = payload
                    self._validate_json_payload()
                else:
                    self._estimator = self._joblib_load(io.BytesIO(raw_artifact))
                    self._validate_estimator()
        except (OSError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
            raise ArtifactModelError("registered model package is invalid") from exc

    def _validate_json_payload(self) -> None:
        coefficients = self._payload.get("coefficients")
        if not isinstance(coefficients, list) or len(coefficients) != len(self.required_features):
            raise ArtifactModelError("JSON return model coefficients do not match feature_columns")
        for index, value in enumerate(coefficients):
            _finite(value, field_name=f"coefficients[{index}]")
        _finite(self._payload.get("intercept", 0.0), field_name="intercept")

    @staticmethod
    def _joblib_load(source: Any) -> Any:
        try:
            import joblib
        except ImportError as exc:  # pragma: no cover - production requirements include joblib.
            raise ArtifactModelError("joblib is unavailable for a registered legacy artifact") from exc
        try:
            return joblib.load(source)
        except Exception as exc:
            raise ArtifactModelError(f"registered estimator could not be loaded: {type(exc).__name__}") from exc

    def _validate_estimator(self) -> None:
        if self._estimator is None or not callable(getattr(self._estimator, "predict", None)):
            raise ArtifactModelError("registered estimator does not expose predict()")

    def _predict_return(self, vector: list[float]) -> float:
        if self._payload:
            coefficients = [_finite(item, field_name="coefficient") for item in self._payload["coefficients"]]
            intercept = _finite(self._payload.get("intercept", 0.0), field_name="intercept")
            return _finite(intercept + sum(value * weight for value, weight in zip(vector, coefficients)), field_name="predicted return")
        if self._estimator is None:
            raise ArtifactModelError("registered artifact was not loaded")
        try:
            return _first_prediction(self._estimator.predict([vector]))
        except ArtifactModelError:
            raise
        except Exception as exc:
            raise ArtifactModelError(f"registered estimator inference failed: {type(exc).__name__}") from exc
