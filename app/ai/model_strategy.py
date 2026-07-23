"""Compatibility adapter for uploaded return-forecast artifacts.

This module deliberately stops at a standardized forecast/direction representation.
It never produces leverage, margin, notional, stops, holds, orders, or execution
instructions. The V2 pipeline decides signal eligibility, portfolio exposure and risk.
"""

from __future__ import annotations

import io
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.ai.strategy import StrategyDecision
from app.ai.symbol_identity import symbol_identity_values
from app.config import settings
from app.db.models import Feature, ModelVersion
from app.features.schema import numeric_vector, values_from_feature


REGIME_ORDER = [
    "news_shock",
    "risk_off",
    "liquidity_stress",
    "crowded_market",
    "breakout_pressure",
    "mean_reversion_pressure",
    "trend_up",
    "trend_down",
    "range_low_volatility",
    "high_volatility",
]


@dataclass(frozen=True)
class ModelDecision:
    """Legacy display adapter: decision conveys direction only, never a trade plan."""

    decision: StrategyDecision
    model: ModelVersion
    prediction: dict[str, Any]


class PriceModelStrategy:
    """Load a frozen uploaded artifact and expose a return forecast for V2 migration."""

    name = "uploaded-model-prediction-adapter"

    def __init__(self) -> None:
        self.last_fallback_reason: str | None = None

    def decide(self, session: Session, feature: Feature) -> ModelDecision | None:
        self.last_fallback_reason = None
        if not settings.enable_server_inference:
            self.last_fallback_reason = "Server model inference is disabled."
            return None
        model = session.scalar(
            select(ModelVersion)
            .where(ModelVersion.status == "active")
            .order_by(desc(ModelVersion.created_at))
            .limit(1)
        )
        if model is None:
            self.last_fallback_reason = "No active model is registered."
            return None
        payload = self._load_model_payload(model)
        if payload is None:
            self.last_fallback_reason = "Active model metadata or file could not be loaded."
            return None
        feature_columns = list(payload.get("feature_columns") or model.feature_columns or [])
        if not feature_columns:
            self.last_fallback_reason = "Active model has no feature_columns metadata."
            return None
        values = values_from_feature(feature, feature_columns)
        values.update(symbol_identity_values(feature.symbol))
        vector = self._feature_vector(feature, feature_columns)
        specialist_regime, specialist_file = self._select_specialist_model_file(payload, values)
        predicted_return = self._predict_model_file(payload, specialist_file, vector) if specialist_file else None
        prediction_source = f"regime_specialist:{specialist_regime}" if predicted_return is not None and specialist_regime else "global_model"
        if predicted_return is None:
            predicted_return = self._predict(payload, vector)
        if predicted_return is None or not math.isfinite(predicted_return):
            self.last_fallback_reason = self.last_fallback_reason or "Active model type is not compatible with safe server inference."
            return None
        required_edge = settings.paper_fee_rate * 2.0 + settings.strategy_min_edge_after_fees
        confidence = self._confidence(predicted_return, required_edge)
        direction = "LONG" if predicted_return > required_edge else "SHORT" if predicted_return < -required_edge else "FLAT"
        action = "BUY" if direction == "LONG" else "SELL" if direction == "SHORT" else "HOLD"
        expected_volatility = max(abs(float(values.get("volatility") or 0.0)), abs(predicted_return) * 0.5, 0.0001)
        probability_up = self._sigmoid(predicted_return / max(expected_volatility, 1e-6))
        prediction = {
            "model_id": model.model_id,
            "model_version": model.version,
            "model_family": model.model_family or "uploaded.return_forecast",
            "feature_schema_version": model.feature_schema_version,
            "feature_columns": feature_columns,
            "predicted_return": predicted_return,
            "expected_return": predicted_return,
            "expected_volatility": expected_volatility,
            "probability_up": probability_up,
            "probability_down": 1.0 - probability_up,
            "confidence": confidence,
            "uncertainty": max(0.0, min(1.0, 1.0 - confidence)),
            "direction": direction,
            "required_edge_after_fees": required_edge,
            "prediction_source": prediction_source,
            "specialist_regime": specialist_regime,
            "legacy_sizing_targets_ignored": True,
        }
        reason = (
            f"Frozen uploaded model forecast ({prediction_source}) estimates {predicted_return:.4%} return; "
            "V2 risk and portfolio policy own all sizing and execution."
        )
        return ModelDecision(StrategyDecision(action=action, confidence=confidence, reason=reason), model, prediction)

    def _feature_vector(self, feature: Feature, feature_columns: list[str]) -> list[float]:
        if not any(column.startswith("symbol_") for column in feature_columns):
            return numeric_vector(feature, feature_columns)
        values = values_from_feature(feature, feature_columns)
        values.update(symbol_identity_values(feature.symbol))
        return [float(values.get(column, 0.0) or 0.0) for column in feature_columns]

    def _load_model_payload(self, model: ModelVersion) -> dict[str, Any] | None:
        payload = dict(model.raw_payload or {})
        path_value = payload.get("model_file") or model.path
        if payload and payload.get("coefficients") is not None:
            return payload
        if not path_value:
            return payload or None
        path = Path(str(path_value))
        if path.exists() and path.suffix.lower() == ".json":
            try:
                file_payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return payload or None
            if isinstance(file_payload, dict):
                payload.update(file_payload)
        if path.exists():
            payload.setdefault("model_file", str(path))
        return payload or None

    def _predict(self, payload: dict[str, Any], vector: list[float]) -> float | None:
        coefficients = payload.get("coefficients")
        if isinstance(coefficients, list):
            try:
                weights = [float(value or 0.0) for value in coefficients]
                if len(weights) != len(vector):
                    return None
                return float(payload.get("intercept", 0.0) or 0.0) + sum(value * weight for value, weight in zip(vector, weights))
            except (TypeError, ValueError):
                return None
        model_file = payload.get("model_file")
        return self._predict_model_file(payload, str(model_file), vector) if model_file else None

    def _predict_model_file(self, payload: dict[str, Any], model_file: str | None, vector: list[float]) -> float | None:
        if not model_file:
            return None
        try:
            import joblib
        except Exception:
            self.last_fallback_reason = "joblib is unavailable for uploaded sklearn inference."
            return None
        try:
            path = Path(model_file)
            if path.exists():
                estimator = joblib.load(path)
            else:
                package_path = Path(str(payload.get("package_path") or ""))
                if not package_path.exists():
                    return None
                with zipfile.ZipFile(package_path) as archive:
                    member = Path(model_file).name
                    if member not in archive.namelist():
                        return None
                    estimator = joblib.load(io.BytesIO(archive.read(member)))
            result = estimator.predict([vector])[0]
            return float(result[0] if isinstance(result, (list, tuple)) else result)
        except Exception as exc:
            self.last_fallback_reason = f"Model inference failed: {type(exc).__name__}."
            return None

    def _select_specialist_model_file(self, payload: dict[str, Any], values: dict[str, Any]) -> tuple[str | None, str | None]:
        files = payload.get("specialist_model_files") if isinstance(payload.get("specialist_model_files"), dict) else {}
        active = self._active_regime(values, list(payload.get("specialist_regime_order") or REGIME_ORDER))
        return active, str(files[active]) if active and active in files else None

    @staticmethod
    def _active_regime(values: dict[str, Any], order: list[str]) -> str | None:
        def value(key: str) -> float:
            try:
                return float(values.get(key, 0.0) or 0.0)
            except (TypeError, ValueError):
                return 0.0
        checks = {
            "news_shock": value("regime_news_shock_score") >= 0.45,
            "risk_off": value("regime_risk_off_score") >= 0.45,
            "liquidity_stress": value("regime_liquidity_stress_score") >= 0.35,
            "crowded_market": value("regime_crowd_pressure") >= 0.35,
            "breakout_pressure": value("regime_breakout_pressure") >= 0.35,
            "mean_reversion_pressure": value("regime_mean_reversion_pressure") >= 0.35,
            "trend_up": value("regime_trend_strength") >= 0.35 and value("regime_direction_score") > 0.10,
            "trend_down": value("regime_trend_strength") >= 0.35 and value("regime_direction_score") < -0.10,
            "range_low_volatility": value("regime_trend_strength") < 0.25 and value("regime_volatility_score") < 0.20,
            "high_volatility": value("regime_volatility_score") >= 0.35,
        }
        return next((regime for regime in order if checks.get(regime)), None)

    @staticmethod
    def _confidence(predicted_return: float, required_edge: float) -> float:
        scale = abs(predicted_return) / max(required_edge * 4.0, 1e-9)
        return max(0.0, min(0.95, 0.45 + scale * 0.35))

    @staticmethod
    def _sigmoid(value: float) -> float:
        value = max(min(value, 20.0), -20.0)
        return 1.0 / (1.0 + math.exp(-value))
