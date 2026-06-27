from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.ai.strategy import StrategyDecision
from app.config import settings
from app.db.models import Feature, ModelVersion
from app.features.schema import numeric_vector, values_from_feature


@dataclass(frozen=True)
class ModelDecision:
    decision: StrategyDecision
    model: ModelVersion
    prediction: dict[str, Any]


class PriceModelStrategy:
    name = "uploaded-model-inference"

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

        vector = numeric_vector(feature, feature_columns)
        predicted_return = self._predict(payload, vector)
        if predicted_return is None:
            self.last_fallback_reason = self.last_fallback_reason or "Active model type is not compatible with Railway inference."
            return None
        required_edge = settings.paper_fee_rate * 2.0 + settings.strategy_min_edge_after_fees
        confidence = self._confidence(predicted_return, required_edge)
        values = values_from_feature(feature, feature_columns)
        last_close = values.get("last_close")
        trade_plan = self._trade_plan(payload=payload, vector=vector, confidence=confidence, predicted_return=predicted_return)

        prediction = {
            "model_id": model.model_id,
            "model_version": model.version,
            "feature_schema_version": model.feature_schema_version,
            "feature_columns": feature_columns,
            "predicted_return": predicted_return,
            "required_edge_after_fees": required_edge,
            "confidence": confidence,
            "trade_plan": trade_plan,
            "trade_plan_source": trade_plan.get("source"),
        }

        if abs(predicted_return) < required_edge:
            return ModelDecision(
                StrategyDecision(
                    action="HOLD",
                    confidence=max(0.50, confidence),
                    reason=(
                        "Trained model edge is too small after fees "
                        f"({predicted_return:.4%} prediction vs {required_edge:.4%} required)."
                    ),
                    max_hold_seconds=int(trade_plan["max_hold_seconds"]),
                ),
                model,
                prediction,
            )

        side = "LONG" if predicted_return > 0 else "SHORT"
        action = "BUY" if side == "LONG" else "SELL"
        return ModelDecision(
            StrategyDecision(
                action=action,
                confidence=confidence,
                reason=(
                    f"Trained model trade plan: {predicted_return:.4%} {side.lower()} edge, "
                    f"{trade_plan['margin_pct']:.2%} margin, {trade_plan['leverage']:.2f}x leverage."
                ),
                stop_loss=self._price_level(
                    last_close,
                    side,
                    stop=True,
                    predicted_return=predicted_return,
                    planned_pct=float(trade_plan.get("stop_loss_pct") or 0.0),
                ),
                take_profit=self._price_level(
                    last_close,
                    side,
                    stop=False,
                    predicted_return=predicted_return,
                    planned_pct=float(trade_plan.get("take_profit_pct") or 0.0),
                ),
                margin_pct=float(trade_plan["margin_pct"]),
                leverage=float(trade_plan["leverage"]),
                max_hold_seconds=int(trade_plan["max_hold_seconds"]),
            ),
            model,
            prediction,
        )

    def _load_model_payload(self, model: ModelVersion) -> dict[str, Any] | None:
        payload = dict(model.raw_payload or {})
        path_value = payload.get("model_file") or model.path
        if payload and payload.get("coefficients") is not None:
            return payload
        if not path_value:
            return payload or None
        path = Path(str(path_value))
        if path.exists() and path.suffix.lower() == ".json":
            file_payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(file_payload, dict):
                payload.update(file_payload)
                payload.setdefault("model_file", str(path))
                return payload
        if path.exists():
            payload.setdefault("model_file", str(path))
            return payload
        return payload or None

    def _predict(self, payload: dict[str, Any], vector: list[float]) -> float | None:
        coefficients = payload.get("coefficients")
        if isinstance(coefficients, list):
            try:
                weights = [float(value or 0.0) for value in coefficients]
            except (TypeError, ValueError):
                return None
            if len(weights) != len(vector):
                return None
            intercept = float(payload.get("intercept", 0.0) or 0.0)
            return intercept + sum(value * coefficient for value, coefficient in zip(vector, weights))

        model_file = payload.get("model_file")
        if not model_file:
            return None
        return self._predict_model_file(payload, str(model_file), vector)

    def _predict_plan_target(self, payload: dict[str, Any], target: str, vector: list[float]) -> float | None:
        plan_files = payload.get("plan_model_files") if isinstance(payload.get("plan_model_files"), dict) else {}
        plan_file = plan_files.get(target)
        if not plan_file:
            return None
        return self._predict_model_file(payload, str(plan_file), vector)

    def _predict_model_file(self, payload: dict[str, Any], model_file: str, vector: list[float]) -> float | None:
        try:
            import joblib
        except Exception:
            self.last_fallback_reason = "joblib is not installed on the server, so uploaded sklearn model cannot run."
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
            prediction = estimator.predict([vector])
        except Exception as exc:
            self.last_fallback_reason = f"Model inference failed: {type(exc).__name__}: {exc}"
            return None
        try:
            first = prediction[0]
            if isinstance(first, (list, tuple)):
                first = first[0]
            return float(first)
        except (TypeError, ValueError, IndexError):
            self.last_fallback_reason = "Model prediction was not a numeric scalar."
            return None

    def _confidence(self, predicted_return: float, required_edge: float) -> float:
        scale = abs(predicted_return) / max(required_edge * 4.0, 1e-9)
        return max(settings.risk_min_confidence, min(0.95, 0.55 + scale * 0.40))

    def _trade_plan(self, *, payload: dict[str, Any], vector: list[float], confidence: float, predicted_return: float) -> dict[str, float | int | str]:
        learned = {
            "margin_pct": self._predict_plan_target(payload, "target_best_margin_pct", vector),
            "leverage": self._predict_plan_target(payload, "target_best_leverage", vector),
            "stop_loss_pct": self._predict_plan_target(payload, "target_best_stop_loss_pct", vector),
            "take_profit_pct": self._predict_plan_target(payload, "target_best_take_profit_pct", vector),
            "max_hold_seconds": self._predict_plan_target(payload, "target_best_hold_seconds", vector),
        }
        if all(value is not None for value in learned.values()):
            return {
                "margin_pct": max(0.0001, min(float(learned["margin_pct"]), 1.0)),
                "leverage": max(1.0, min(float(learned["leverage"]), settings.paper_max_leverage)),
                "stop_loss_pct": max(0.0005, min(float(learned["stop_loss_pct"]), 0.50)),
                "take_profit_pct": max(0.0005, min(float(learned["take_profit_pct"]), 1.00)),
                "max_hold_seconds": int(max(60, min(float(learned["max_hold_seconds"]), 86400))),
                "source": "learned_plan_models",
            }

        confidence_span = max(1.0 - settings.risk_min_confidence, 1e-9)
        confidence_scale = max(0.0, min(1.0, (confidence - settings.risk_min_confidence) / confidence_span))
        edge_scale = max(0.0, min(1.0, abs(predicted_return) / 0.02))
        plan_scale = max(confidence_scale, edge_scale * 0.65)
        max_margin_pct = min(max(settings.risk_max_trade_size_pct, 0.01), 1.0)
        margin_pct = max(0.0025, max_margin_pct * plan_scale)
        max_leverage = max(settings.paper_max_leverage, settings.paper_min_leverage, 1.0)
        min_leverage = min(max(settings.paper_min_leverage, 1.0), max_leverage)
        leverage = min_leverage + (max_leverage - min_leverage) * plan_scale
        max_hold_seconds = int(900 + (settings.auto_max_hold_seconds - 900) * max(0.0, 1.0 - edge_scale))
        return {
            "margin_pct": margin_pct,
            "leverage": leverage,
            "stop_loss_pct": max(settings.auto_default_stop_loss_pct, min(0.08, abs(predicted_return) * 0.75)),
            "take_profit_pct": max(settings.auto_default_take_profit_pct, min(0.20, abs(predicted_return) * 2.5)),
            "max_hold_seconds": max_hold_seconds,
            "source": "fallback_generated_plan",
        }

    def _price_level(
        self,
        price: Any,
        side: str,
        *,
        stop: bool,
        predicted_return: float = 0.0,
        planned_pct: float = 0.0,
    ) -> float | None:
        try:
            mark = float(price)
        except (TypeError, ValueError):
            return None
        if mark <= 0:
            return None
        if stop:
            offset = planned_pct if planned_pct > 0 else max(settings.auto_default_stop_loss_pct, min(0.08, abs(predicted_return) * 0.75))
            multiplier = 1.0 - offset if side == "LONG" else 1.0 + offset
            return round(mark * multiplier, 8)

        offset = planned_pct if planned_pct > 0 else max(settings.auto_default_take_profit_pct, min(0.20, abs(predicted_return) * 2.5))
        multiplier = 1.0 + offset if side == "LONG" else 1.0 - offset
        return round(mark * multiplier, 8)
