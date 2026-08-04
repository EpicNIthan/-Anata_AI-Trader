"""Deterministic local training for forecast-only narrow-model challengers.

This module is intentionally independent of the web and paper-execution loops.  It
accepts point-in-time rows, searches a bounded set of linear configurations, fits a
fresh estimator inside every walk-forward fold, and emits frozen artifacts that the
runtime :class:`RegisteredArtifactModel` adapter can load.  It has no champion
promotion function.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any
import zipfile

from app.features.schema import DEFAULT_FEATURE_VALUES
from app.research.evaluation import (
    ExecutionAssumptions,
    WalkForwardEvaluator,
    apply_execution_assumptions,
    assert_point_in_time_availability,
    observation_from_row,
    row_timestamp,
)
from app.research.schemas import EvaluationObservation, ResearchValidationError, ensure_utc, stable_fingerprint


PACKAGE_REQUIRED_FILES: frozenset[str] = frozenset(
    {
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
MODEL_MEMBER = "return_model.json"
FORBIDDEN_TARGET_TOKENS: tuple[str, ...] = (
    "action",
    "allocation",
    "exit",
    "hold",
    "leverage",
    "margin",
    "order",
    "position",
    "size",
    "stop",
    "take_profit",
    "trade_quality",
)
LEAKAGE_FEATURE_TOKENS: tuple[str, ...] = (
    "future_return",
    "hit_first",
    "label_",
    "max_drawdown",
    "max_upside",
    "next_price",
    "target_",
)

# A narrow family gets only the inputs needed for its hypothesis.  The final
# baseline deliberately uses every observed live-serving feature; it is the broad
# reference against which the specialist families are compared.
FAMILY_FEATURE_COLUMNS: dict[str, tuple[str, ...] | None] = {
    "short_horizon_momentum": (
        "price_change",
        "candle_return_1m",
        "candle_return_5m",
        "macd_pct",
        "macd_histogram_pct",
        "trend_score",
        "regime_direction_score",
    ),
    "medium_horizon_momentum": (
        "candle_return_5m",
        "trend_score",
        "sma_20_distance_pct",
        "ema_20_distance_pct",
        "adx_14",
        "spot_price_change_24h",
        "regime_trend_strength",
    ),
    "mean_reversion": (
        "rsi_14",
        "bollinger_position",
        "sma_20_distance_pct",
        "ema_20_distance_pct",
        "vwap_20_distance_pct",
        "volatility",
        "regime_mean_reversion_pressure",
    ),
    "breakout_pressure": (
        "candle_return_5m",
        "volume_change",
        "bollinger_width_pct",
        "spot_intraday_range_pct",
        "spot_activity_score",
        "spot_orderbook_imbalance",
        "regime_breakout_pressure",
    ),
    "derivatives_flow": (
        "taker_buy_pressure",
        "taker_buy_sell_ratio",
        "open_interest_change",
        "funding_rate",
        "crowd_long_short_ratio",
        "trader_crowd_score",
        "regime_crowd_pressure",
    ),
    "liquidation_pressure": (
        "liquidation_long_usd_1m",
        "liquidation_short_usd_1m",
        "liquidation_long_usd_5m",
        "liquidation_short_usd_5m",
        "liquidation_total_usd_5m",
        "liquidation_imbalance_5m",
        "liquidation_spike_score",
    ),
    "news_event": (
        "sentiment_score",
        "sentiment_confidence",
        "risk_score",
        "impact_score",
        "recency_weight",
        "macro_risk_score",
        "regulation_risk_score",
        "fed_risk_score",
        "war_risk_score",
        "exchange_hack_risk_score",
        "etf_positive_score",
    ),
    "cross_asset_context": (
        "fear_greed_value",
        "fear_greed_change_24h",
        "global_market_cap_change_24h",
        "total_volume_change_24h",
        "btc_dominance_change_24h",
        "eth_dominance",
        "stablecoin_depeg_risk",
        "stablecoin_supply_change_24h",
        "world_risk_score",
        "market_regime_score",
    ),
    "linear_baseline": None,
}

ROBUST_FAMILIES: frozenset[str] = frozenset(
    {"mean_reversion", "liquidation_pressure", "news_event", "cross_asset_context"}
)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _value(row: Mapping[str, Any], name: str) -> Any:
    if name in row:
        return row[name]
    for container_name in ("values", "features", "feature_values"):
        container = row.get(container_name)
        if isinstance(container, Mapping) and name in container:
            return container[name]
    raise KeyError(name)


def _runtime_feature_default(name: str) -> float:
    value = DEFAULT_FEATURE_VALUES.get(name, 0.0)
    try:
        return _finite(value, f"default for {name}")
    except ValueError:
        return 0.0


def observed_news_student_contract(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return the single local-student version observed by point-in-time features.

    Rule and legacy fallbacks are not trained artifacts. Mixing two actual student
    versions would make serving compatibility ambiguous, so packaging refuses it and
    asks the operator to train version-homogeneous datasets.
    """
    versions: set[str] = set()
    for row in rows:
        try:
            raw_version = _value(row, "local_news_model_version")
        except KeyError:
            continue
        version = str(raw_version or "").strip()
        if not version or version in {
            "deterministic-news-rules-v1",
            "legacy-news-sentiment-fallback",
            "rule-v1",
            "unknown-local-model",
        }:
            continue
        try:
            provider = str(_value(row, "local_news_provider") or "").strip().lower()
        except KeyError:
            provider = ""
        if provider and provider != "local_student":
            continue
        versions.add(version)
    if len(versions) > 1:
        raise ResearchValidationError(
            "training rows contain multiple local news student versions; split and train each version separately: "
            + ", ".join(sorted(versions))
        )
    version = next(iter(versions), None)
    return {
        "required": version is not None,
        "version": version,
        "external_ai_features_optional": True,
    }


def validate_return_target(target_name: str) -> str:
    """Keep research artifacts on the forecast side of the risk boundary."""

    normalized = target_name.strip()
    lowered = normalized.lower()
    if not normalized or any(token in lowered for token in FORBIDDEN_TARGET_TOKENS):
        raise ValueError(
            "target must be a future-return forecast, never sizing, leverage, stop, exit, or order output"
        )
    if "return" not in lowered and "price_change" not in lowered:
        raise ValueError("target must describe a future return or price change")
    return normalized


def validate_feature_contract(feature_columns: Sequence[str], *, target_name: str) -> tuple[str, ...]:
    columns = tuple(str(item).strip() for item in feature_columns)
    unsafe = [
        name
        for name in columns
        if not name
        or name == target_name
        or any(token in name.lower() for token in LEAKAGE_FEATURE_TOKENS)
    ]
    if unsafe:
        raise ValueError("target/forward-label columns cannot be model features: " + ", ".join(unsafe))
    if len(columns) != len(set(columns)):
        raise ValueError("feature columns must be unique")
    return columns


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def _finite_json(value: Any) -> Any:
    """Replace undefined ratio outputs with JSON ``null`` recursively."""

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_oos_observations(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write deterministic standalone-evaluator input without mutating existing data."""

    payload = b"".join(_json_bytes(_finite_json(dict(row))) for row in rows)
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        if output_path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace different OOS observations: {output_path}")
    else:
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(output_path)
    return {
        "path": str(output_path),
        "filename": output_path.name,
        "sha256": sha256_path(output_path),
        "bytes": output_path.stat().st_size,
        "rows": len(rows),
        "format": "jsonl",
        "contract": "evaluation_observation_v1",
    }


@dataclass(frozen=True, slots=True)
class LabeledRowInventory:
    total_rows: int
    finite_label_rows: int
    available_labeled_rows: int
    pending_label_rows: int
    missing_or_invalid_label_rows: int
    newest_available_label_time: datetime | None
    row_ids: tuple[str, ...]

    def model_dump(self, *, include_row_ids: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "total_rows": self.total_rows,
            "finite_label_rows": self.finite_label_rows,
            "available_labeled_rows": self.available_labeled_rows,
            "pending_label_rows": self.pending_label_rows,
            "missing_or_invalid_label_rows": self.missing_or_invalid_label_rows,
            "newest_available_label_time": (
                self.newest_available_label_time.isoformat() if self.newest_available_label_time else None
            ),
        }
        if include_row_ids:
            payload["row_ids"] = list(self.row_ids)
        return payload


def discover_available_labeled_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_name: str,
    forecast_horizon_seconds: int,
    labels_as_of: datetime | str | None = None,
    transaction_cost: float = 0.0,
) -> tuple[list[dict[str, Any]], LabeledRowInventory]:
    """Return only finite labels that were knowable by ``labels_as_of``.

    Rows without an explicit label timestamp use ``decision time + horizon``.
    Feature availability is validated before label filtering so an unlabeled bad row
    cannot silently hide a point-in-time data-quality violation.
    """

    if forecast_horizon_seconds <= 0:
        raise ValueError("forecast_horizon_seconds must be positive")
    target_name = validate_return_target(target_name)
    transaction_cost = _finite(transaction_cost, "transaction_cost")
    if transaction_cost < 0:
        raise ValueError("transaction_cost cannot be negative")
    cutoff = ensure_utc(labels_as_of, field_name="labels_as_of") if labels_as_of else datetime.now(timezone.utc)
    assert_point_in_time_availability(rows)
    normalized: list[dict[str, Any]] = []
    finite_count = 0
    pending_count = 0
    invalid_count = 0
    newest: datetime | None = None
    row_ids: list[str] = []
    for row in rows:
        try:
            label = _finite(_value(row, target_name), target_name)
        except (KeyError, ValueError):
            invalid_count += 1
            continue
        finite_count += 1
        decision_time = row_timestamp(row)
        explicit_label_time = next(
            (row.get(name) for name in ("label_available_time", "label_end_time", "label_end") if row.get(name) not in (None, "")),
            None,
        )
        label_time = (
            ensure_utc(explicit_label_time, field_name="label_available_time")
            if explicit_label_time is not None
            else decision_time + timedelta(seconds=forecast_horizon_seconds)
        )
        if label_time < decision_time:
            raise ResearchValidationError("label_available_time cannot precede the decision timestamp")
        if label_time > cutoff:
            pending_count += 1
            continue
        candidate = dict(row)
        candidate["timestamp"] = decision_time.isoformat()
        candidate.setdefault("available_to_model_time", decision_time.isoformat())
        candidate["label_available_time"] = label_time.isoformat()
        candidate["actual_return"] = label
        candidate["transaction_cost"] = transaction_cost
        identity = stable_fingerprint(
            {
                "timestamp": decision_time.isoformat(),
                "symbol": str(row.get("symbol") or "*"),
                "label_available_time": label_time.isoformat(),
                "target_name": target_name,
                "target_value": label,
            }
        )
        candidate["research_row_id"] = identity
        normalized.append(candidate)
        row_ids.append(identity)
        newest = label_time if newest is None or label_time > newest else newest
    normalized.sort(key=lambda item: (row_timestamp(item), str(item.get("symbol") or ""), item["research_row_id"]))
    inventory = LabeledRowInventory(
        total_rows=len(rows),
        finite_label_rows=finite_count,
        available_labeled_rows=len(normalized),
        pending_label_rows=pending_count,
        missing_or_invalid_label_rows=invalid_count,
        newest_available_label_time=newest,
        row_ids=tuple(sorted(row_ids)),
    )
    return normalized, inventory


def observed_numeric_features(
    rows: Sequence[Mapping[str, Any]],
    *,
    allowed_columns: Sequence[str],
) -> tuple[str, ...]:
    """Find live-contract columns containing at least one finite observed value."""

    observed: list[str] = []
    for name in dict.fromkeys(str(item) for item in allowed_columns):
        for row in rows:
            try:
                _finite(_value(row, name), name)
            except (KeyError, ValueError):
                continue
            observed.append(name)
            break
    return tuple(observed)


@dataclass(frozen=True, slots=True)
class NarrowCandidateConfig:
    candidate_id: str
    model_family: str
    feature_columns: tuple[str, ...]
    algorithm: str
    alpha: float
    huber_delta: float = 1.35

    @property
    def fingerprint(self) -> str:
        return stable_fingerprint(self.model_dump())

    def model_dump(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "model_family": self.model_family,
            "feature_columns": list(self.feature_columns),
            "algorithm": self.algorithm,
            "alpha": self.alpha,
            "huber_delta": self.huber_delta,
        }


def build_narrow_candidate_configs(
    rows: Sequence[Mapping[str, Any]],
    *,
    allowed_columns: Sequence[str],
    model_families: Sequence[str] = tuple(FAMILY_FEATURE_COLUMNS),
    alphas: Sequence[float] = (0.1, 1.0, 10.0),
    maximum_baseline_features: int = 128,
) -> tuple[tuple[NarrowCandidateConfig, ...], tuple[str, ...]]:
    """Build a bounded deterministic search grid and report unavailable families."""

    if maximum_baseline_features <= 0:
        raise ValueError("maximum_baseline_features must be positive")
    numeric = observed_numeric_features(rows, allowed_columns=allowed_columns)
    numeric_set = set(numeric)
    normalized_alphas = tuple(sorted({_finite(value, "alpha") for value in alphas}))
    if not normalized_alphas or any(value < 0 for value in normalized_alphas):
        raise ValueError("alphas must contain non-negative finite values")
    configurations: list[NarrowCandidateConfig] = []
    unavailable: list[str] = []
    for family in dict.fromkeys(str(item).strip() for item in model_families):
        declared = FAMILY_FEATURE_COLUMNS.get(family)
        if family not in FAMILY_FEATURE_COLUMNS:
            raise ValueError(f"unsupported narrow model family: {family}")
        columns = tuple(numeric[:maximum_baseline_features]) if declared is None else tuple(
            name for name in declared if name in numeric_set
        )
        if not columns:
            unavailable.append(family)
            continue
        algorithm = "huber_ridge" if family in ROBUST_FAMILIES else "ridge"
        for alpha in normalized_alphas:
            identity = stable_fingerprint(
                {"family": family, "features": columns, "algorithm": algorithm, "alpha": alpha}
            )[:12]
            configurations.append(
                NarrowCandidateConfig(
                    candidate_id=f"{family}-{algorithm}-{identity}",
                    model_family=family,
                    feature_columns=columns,
                    algorithm=algorithm,
                    alpha=alpha,
                )
            )
    return tuple(configurations), tuple(unavailable)


@dataclass(frozen=True, slots=True)
class FrozenLinearEstimator:
    feature_columns: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float
    imputed_values: int = 0

    def predict_rows(self, rows: Sequence[Mapping[str, Any]]) -> list[float]:
        predictions: list[float] = []
        for row in rows:
            vector: list[float] = []
            for name in self.feature_columns:
                try:
                    vector.append(_finite(_value(row, name), name))
                except (KeyError, ValueError):
                    vector.append(_runtime_feature_default(name))
            predictions.append(self.intercept + sum(value * weight for value, weight in zip(vector, self.coefficients)))
        return predictions


def fit_linear_estimator(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: NarrowCandidateConfig,
    target_name: str,
) -> FrozenLinearEstimator:
    """Fit deterministic ridge or Huber-ridge and return raw-space coefficients."""

    if not rows:
        raise ResearchValidationError("cannot fit a narrow model without rows")
    validate_return_target(target_name)
    validate_feature_contract(config.feature_columns, target_name=target_name)
    import numpy as np

    imputed = 0
    matrix: list[list[float]] = []
    targets: list[float] = []
    for row in rows:
        vector: list[float] = []
        for name in config.feature_columns:
            try:
                vector.append(_finite(_value(row, name), name))
            except (KeyError, ValueError):
                vector.append(_runtime_feature_default(name))
                imputed += 1
        matrix.append(vector)
        targets.append(_finite(_value(row, target_name), target_name))
    x = np.asarray(matrix, dtype=float)
    y = np.asarray(targets, dtype=float)
    means = np.mean(x, axis=0)
    scales = np.std(x, axis=0)
    scales = np.where(scales > 0.0, scales, 1.0)
    standardized = (x - means) / scales
    design = np.column_stack((np.ones(len(x), dtype=float), standardized))
    penalty = np.eye(design.shape[1], dtype=float) * config.alpha
    penalty[0, 0] = 0.0

    def solve(weights: Any) -> Any:
        weighted_design = design * weights[:, None]
        return np.linalg.pinv(design.T @ weighted_design + penalty) @ design.T @ (weights * y)

    weights = np.ones(len(y), dtype=float)
    beta = solve(weights)
    if config.algorithm == "huber_ridge":
        for _ in range(12):
            residuals = y - design @ beta
            median = float(np.median(residuals))
            mad = float(np.median(np.abs(residuals - median)))
            scale = max(1.4826 * mad, float(np.std(residuals)) * 0.1, 1e-12)
            threshold = config.huber_delta * scale
            absolute = np.abs(residuals)
            updated = np.where(absolute <= threshold, 1.0, threshold / np.maximum(absolute, 1e-12))
            next_beta = solve(updated)
            if float(np.max(np.abs(next_beta - beta))) < 1e-12:
                beta = next_beta
                break
            beta = next_beta
            weights = updated
    elif config.algorithm != "ridge":
        raise ValueError(f"unsupported training algorithm: {config.algorithm}")
    raw_coefficients = np.asarray(beta[1:], dtype=float) / scales
    raw_intercept = float(beta[0] - np.dot(means, raw_coefficients))
    if not math.isfinite(raw_intercept) or not all(math.isfinite(float(value)) for value in raw_coefficients):
        raise ResearchValidationError("training produced a non-finite linear artifact")
    return FrozenLinearEstimator(
        feature_columns=config.feature_columns,
        coefficients=tuple(float(value) for value in raw_coefficients),
        intercept=raw_intercept,
        imputed_values=imputed,
    )


def _metric_number(metrics: Mapping[str, Any], name: str, default: float) -> float:
    value = metrics.get(name)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def candidate_rank_key(result: Mapping[str, Any]) -> tuple[float, float, float, float, str]:
    """Conservative, deterministic offline rank; this is never a promotion gate."""

    metrics = result.get("metrics") if isinstance(result.get("metrics"), Mapping) else {}
    return (
        _metric_number(metrics, "information_coefficient", -math.inf),
        _metric_number(metrics, "net_expectancy", -math.inf),
        _metric_number(metrics, "directional_hit_rate", -math.inf),
        -_metric_number(result, "root_mean_squared_error", math.inf),
        str(result.get("candidate_id") or ""),
    )


def evaluate_narrow_candidate(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: NarrowCandidateConfig,
    target_name: str,
    train_size: int,
    validation_size: int,
    test_size: int,
    step_size: int,
    purge_size: int,
    embargo_size: int,
    forecast_horizon_seconds: int,
    expanding: bool = True,
    execution_assumptions: ExecutionAssumptions | None = None,
    include_oos_observations: bool = False,
) -> dict[str, Any]:
    """Refit one candidate in every chronological test fold and score only OOS rows."""

    predictions: list[float] = []
    actuals: list[float] = []
    oos_observations: list[EvaluationObservation] = []
    fold_index = 0
    assumptions = execution_assumptions or ExecutionAssumptions()

    def fit_predict(
        training_rows: Sequence[Mapping[str, Any]],
        _validation_rows: Sequence[Mapping[str, Any]],
        testing_rows: Sequence[Mapping[str, Any]],
    ) -> list[float]:
        nonlocal fold_index
        estimator = fit_linear_estimator(training_rows, config=config, target_name=target_name)
        output = estimator.predict_rows(testing_rows)
        predictions.extend(output)
        actuals.extend(_finite(_value(row, target_name), target_name) for row in testing_rows)
        if include_oos_observations:
            for row, predicted in zip(testing_rows, output):
                base = observation_from_row(row, prediction=predicted)
                oos_observations.append(
                    EvaluationObservation(
                        timestamp=base.timestamp,
                        prediction=predicted,
                        actual_return=_finite(_value(row, target_name), target_name),
                        symbol=base.symbol,
                        signal_id=config.candidate_id,
                        model_family=f"alpha.{config.model_family}",
                        regime=base.regime,
                        position=None,
                        transaction_cost=base.transaction_cost,
                        holding_seconds=float(forecast_horizon_seconds),
                        external_ai_available=base.external_ai_available,
                        available_to_model_time=base.available_to_model_time,
                        label_available_time=base.label_available_time,
                        feature_families=(config.model_family,),
                        metadata={
                            **dict(base.metadata),
                            "candidate_id": config.candidate_id,
                            "candidate_fingerprint": config.fingerprint,
                            "fold_index": fold_index,
                            "forecast_horizon_seconds": forecast_horizon_seconds,
                        },
                    )
                )
        fold_index += 1
        return output

    evaluator = WalkForwardEvaluator(
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        step_size=step_size,
        expanding=expanding,
        purge_window=purge_size,
        embargo_window=embargo_size,
        forecast_horizon_seconds=forecast_horizon_seconds,
        execution_assumptions=assumptions,
    )
    result = evaluator.evaluate(
        rows,
        fit_predict=fit_predict,
        target_getter=lambda row: _finite(_value(row, target_name), target_name),
    )
    if not predictions:
        raise ResearchValidationError("candidate produced no out-of-sample predictions")
    mean_absolute_error = sum(abs(actual - predicted) for actual, predicted in zip(actuals, predictions)) / len(predictions)
    root_mean_squared_error = math.sqrt(
        sum((actual - predicted) ** 2 for actual, predicted in zip(actuals, predictions)) / len(predictions)
    )
    payload = {
        **config.model_dump(),
        "candidate_fingerprint": config.fingerprint,
        "status": "completed",
        "metrics": _finite_json(result.evaluation.metrics),
        "mean_absolute_error": mean_absolute_error,
        "root_mean_squared_error": root_mean_squared_error,
        "fold_metrics": [_finite_json(item.metrics) for item in result.fold_evaluations],
        "folds": [fold.model_dump() for fold in result.folds],
        "oos_observation_count": len(predictions),
        "forecast_horizon_seconds": forecast_horizon_seconds,
        "annualization_basis": "forecast_horizon_seconds",
        "execution_assumptions": assumptions.model_dump(),
    }
    if include_oos_observations:
        payload["oos_observations"] = [
            observation.model_dump()
            for observation in apply_execution_assumptions(oos_observations, assumptions)
        ]
    return payload


def _adaptive_split_sizes(row_count: int) -> dict[str, int]:
    if row_count < 24:
        raise ResearchValidationError("at least 24 available labeled rows are required for walk-forward training")
    test = max(4, min(256, row_count // 6))
    validation = max(2, min(128, row_count // 10))
    train = max(12, row_count // 2)
    # Leave explicit purge + validation + embargo + test room for at least one fold.
    while train + validation + test + 2 >= row_count and train > 12:
        train -= 1
    if train + validation + test + 2 >= row_count:
        raise ResearchValidationError("not enough rows for leakage-safe train/validation/test gaps")
    return {"train_size": train, "validation_size": validation, "test_size": test, "step_size": test}


def _training_period(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    timestamps = [row_timestamp(row) for row in rows]
    return {"start": min(timestamps).isoformat(), "end": max(timestamps).isoformat()}


def train_selected_artifact(
    rows: Sequence[Mapping[str, Any]],
    *,
    selected_result: Mapping[str, Any],
    target_name: str,
    feature_schema_version: str,
    forecast_horizon_seconds: int,
    dataset_version: str,
) -> dict[str, Any]:
    config = NarrowCandidateConfig(
        candidate_id=str(selected_result["candidate_id"]),
        model_family=str(selected_result["model_family"]),
        feature_columns=tuple(str(item) for item in selected_result["feature_columns"]),
        algorithm=str(selected_result["algorithm"]),
        alpha=float(selected_result["alpha"]),
        huber_delta=float(selected_result.get("huber_delta", 1.35)),
    )
    estimator = fit_linear_estimator(rows, config=config, target_name=target_name)
    actuals = [_finite(_value(row, target_name), target_name) for row in rows]
    target_scale = math.sqrt(sum((value - sum(actuals) / len(actuals)) ** 2 for value in actuals) / len(actuals))
    oos_mae = _metric_number(selected_result, "mean_absolute_error", math.inf)
    calibration = max(0.0, min(1.0, 1.0 - oos_mae / max(target_scale, 1e-12)))
    period = _training_period(rows)
    news_student_contract = observed_news_student_contract(rows)
    trained_at = max(
        ensure_utc(row["label_available_time"], field_name="label_available_time") for row in rows
    ).isoformat()
    return {
        "artifact_type": "anata_narrow_return_forecast",
        "contract_version": 1,
        "feature_columns": list(config.feature_columns),
        "coefficients": list(estimator.coefficients),
        "intercept": estimator.intercept,
        "target_name": target_name,
        "feature_schema_version": feature_schema_version,
        "preprocessing_version": "raw-linear-v1",
        "missing_value_policy": {
            "strategy": "live_schema_default",
            "training_imputed_feature_values": estimator.imputed_values,
            "feature_defaults": {
                name: _runtime_feature_default(name) for name in config.feature_columns
            },
        },
        "forecast_horizon_seconds": forecast_horizon_seconds,
        "annualization_basis": "forecast_horizon_seconds",
        "execution_assumptions": dict(selected_result.get("execution_assumptions") or {}),
        "training_dataset_version": dataset_version,
        "model_family": f"alpha.{config.model_family}",
        "algorithm": config.algorithm,
        "hyperparameters": {"alpha": config.alpha, "huber_delta": config.huber_delta, "shuffle": False},
        "candidate_id": config.candidate_id,
        "candidate_fingerprint": config.fingerprint,
        "training_period": period,
        "training_rows": len(rows),
        "metrics": {
            "calibration_score": calibration,
            "walk_forward": dict(selected_result.get("metrics") or {}),
            "mean_absolute_error": selected_result.get("mean_absolute_error"),
            "root_mean_squared_error": selected_result.get("root_mean_squared_error"),
            "fold_metrics": list(selected_result.get("fold_metrics") or []),
        },
        "trained_at": trained_at,
        "allowed_outputs": ["expected_return"],
        "forbidden_outputs": [
            "position_size",
            "leverage",
            "margin",
            "stop_loss",
            "take_profit",
            "exit_time",
            "order_instruction",
        ],
        "paper_only": True,
        "automatic_promotion": False,
        "news_student_version": news_student_contract,
    }


def _package_payloads(artifact: Mapping[str, Any]) -> dict[str, bytes]:
    features = [str(item) for item in artifact["feature_columns"]]
    news_contract = dict(
        artifact.get("news_student_version")
        or {"required": False, "version": None, "external_ai_features_optional": True}
    )
    metadata = {
        "artifact_type": artifact["artifact_type"],
        "contract_version": artifact["contract_version"],
        "model_file": MODEL_MEMBER,
        "model_family": artifact["model_family"],
        "candidate_id": artifact["candidate_id"],
        "candidate_fingerprint": artifact["candidate_fingerprint"],
        "feature_schema_version": artifact["feature_schema_version"],
        "feature_columns": features,
        "preprocessing_version": artifact["preprocessing_version"],
        "training_dataset_version": artifact["training_dataset_version"],
        "target_name": artifact["target_name"],
        "forecast_horizon_seconds": artifact["forecast_horizon_seconds"],
        "allowed_outputs": artifact["allowed_outputs"],
        "oos_observations": artifact.get("oos_observations"),
        "execution_assumptions": artifact.get("execution_assumptions"),
        "annualization_basis": artifact.get("annualization_basis", "forecast_horizon_seconds"),
        "paper_only": True,
        "automatic_promotion": False,
        "news_student_version": news_contract,
    }
    payloads = {
        MODEL_MEMBER: _json_bytes(artifact),
        "feature_schema.json": _json_bytes(
            {
                "feature_schema_version": artifact["feature_schema_version"],
                "feature_columns": features,
                "feature_types": {name: "number" for name in features},
                "ordered": True,
                "feature_order_sha256": _sha256_bytes(_json_bytes(features)),
            }
        ),
        "model_metadata.json": _json_bytes(metadata),
        "training_metrics.json": _json_bytes(artifact["metrics"]),
        "training_period.json": _json_bytes(
            {
                **dict(artifact["training_period"]),
                "training_rows": artifact["training_rows"],
                "point_in_time": True,
                "random_time_series_shuffle": False,
            }
        ),
        "required_features.json": _json_bytes(features),
        "optional_features.json": _json_bytes([]),
        "missing_value_policy.json": _json_bytes(artifact["missing_value_policy"]),
        "news_student_version.json": _json_bytes(
            news_contract
        ),
    }
    manifest = {
        "algorithm": "sha256",
        "manifest_contract_version": 1,
        "files": {
            name: {"sha256": _sha256_bytes(content), "bytes": len(content)}
            for name, content in sorted(payloads.items())
        },
        "note": "The manifest hashes every package member except itself.",
    }
    payloads["checksum_manifest.json"] = _json_bytes(manifest)
    return payloads


def _zip_bytes(payloads: Mapping[str, bytes]) -> bytes:
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(payloads):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, payloads[name])
    return target.getvalue()


def package_narrow_artifact(
    artifact: Mapping[str, Any],
    output_path: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create an atomic, deterministic, runtime-compatible ZIP package."""

    if output_path.suffix.lower() != ".zip":
        raise ValueError("narrow model package path must end in .zip")
    payload = _zip_bytes(_package_payloads(artifact))
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        existing = output_path.read_bytes()
        if existing != payload:
            raise FileExistsError(f"refusing to replace a different model package: {output_path}")
    else:
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(output_path)
    validation = verify_narrow_package(output_path)
    return {
        "path": str(output_path),
        "sha256": sha256_path(output_path),
        "bytes": output_path.stat().st_size,
        "validation": validation,
    }


def verify_narrow_package(path: Path) -> dict[str, Any]:
    """Verify required members, declared hashes, feature order, and model shape."""

    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ResearchValidationError("model package contains duplicate member names")
            missing = sorted((PACKAGE_REQUIRED_FILES | {MODEL_MEMBER}) - set(names))
            if missing:
                raise ResearchValidationError("model package is missing: " + ", ".join(missing))
            manifest = json.loads(archive.read("checksum_manifest.json"))
            files = manifest.get("files") if isinstance(manifest, Mapping) else None
            if not isinstance(files, Mapping):
                raise ResearchValidationError("checksum manifest has no files mapping")
            expected_hashed_members = set(names) - {"checksum_manifest.json"}
            if set(files) != expected_hashed_members:
                raise ResearchValidationError("checksum manifest does not cover every non-manifest member")
            for name, descriptor in files.items():
                if name not in names or not isinstance(descriptor, Mapping):
                    raise ResearchValidationError(f"invalid checksum entry: {name}")
                content = archive.read(name)
                if descriptor.get("sha256") != _sha256_bytes(content) or descriptor.get("bytes") != len(content):
                    raise ResearchValidationError(f"checksum mismatch for package member: {name}")
            artifact = json.loads(archive.read(MODEL_MEMBER))
            required_features = json.loads(archive.read("required_features.json"))
            if artifact.get("feature_columns") != required_features:
                raise ResearchValidationError("artifact and required feature order differ")
            coefficients = artifact.get("coefficients")
            if not isinstance(coefficients, list) or len(coefficients) != len(required_features):
                raise ResearchValidationError("artifact coefficient count does not match required features")
            news_contract = json.loads(archive.read("news_student_version.json"))
            if not isinstance(news_contract, Mapping) or not isinstance(news_contract.get("required"), bool):
                raise ResearchValidationError("news_student_version.json has an invalid required flag")
            required_news_version = news_contract.get("version")
            if news_contract["required"] and (
                not isinstance(required_news_version, str) or not required_news_version.strip()
            ):
                raise ResearchValidationError("required news student version is missing")
            artifact_contract = artifact.get("news_student_version")
            if artifact_contract != news_contract:
                raise ResearchValidationError("artifact and news student version contracts differ")
            metadata = json.loads(archive.read("model_metadata.json"))
            if not isinstance(metadata, Mapping) or metadata.get("news_student_version") != news_contract:
                raise ResearchValidationError("metadata and news student version contracts differ")
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        raise ResearchValidationError(f"invalid narrow model package: {exc}") from exc
    return {
        "compatible": True,
        "required_files_present": True,
        "checksums_verified": len(files),
        "model_member": MODEL_MEMBER,
        "feature_count": len(required_features),
        "news_student_version": news_contract,
    }


def run_narrow_research_cycle(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    target_name: str,
    feature_schema_version: str,
    allowed_feature_columns: Sequence[str],
    forecast_horizon_seconds: int,
    dataset_version: str,
    labels_as_of: datetime | str | None = None,
    transaction_cost: float = 0.0008,
    model_families: Sequence[str] = tuple(FAMILY_FEATURE_COLUMNS),
    alphas: Sequence[float] = (0.1, 1.0, 10.0),
    train_size: int | None = None,
    validation_size: int | None = None,
    test_size: int | None = None,
    step_size: int | None = None,
    purge_size: int = 1,
    embargo_size: int = 1,
    expanding: bool = True,
    execution_assumptions: ExecutionAssumptions | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Search, evaluate, refit, and package one best challenger per available family."""

    target_name = validate_return_target(target_name)
    validate_feature_contract(allowed_feature_columns, target_name=target_name)
    assumptions = execution_assumptions or ExecutionAssumptions()
    labeled_rows, inventory = discover_available_labeled_rows(
        rows,
        target_name=target_name,
        forecast_horizon_seconds=forecast_horizon_seconds,
        labels_as_of=labels_as_of,
        transaction_cost=transaction_cost,
    )
    configurations, unavailable = build_narrow_candidate_configs(
        labeled_rows,
        allowed_columns=allowed_feature_columns,
        model_families=model_families,
        alphas=alphas,
    )
    if not configurations:
        raise ResearchValidationError("no requested model family has observed live-serving features")
    split = _adaptive_split_sizes(len(labeled_rows))
    split.update(
        {
            key: int(value)
            for key, value in {
                "train_size": train_size,
                "validation_size": validation_size,
                "test_size": test_size,
                "step_size": step_size,
            }.items()
            if value is not None
        }
    )
    if min(split.values()) <= 0 or min(purge_size, embargo_size) < 0:
        raise ValueError("split sizes must be positive and purge/embargo sizes cannot be negative")
    evaluations: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for config in configurations:
        try:
            evaluations.append(
                evaluate_narrow_candidate(
                    labeled_rows,
                    config=config,
                    target_name=target_name,
                    purge_size=purge_size,
                    embargo_size=embargo_size,
                    forecast_horizon_seconds=forecast_horizon_seconds,
                    expanding=expanding,
                    execution_assumptions=assumptions,
                    **split,
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "candidate_id": config.candidate_id,
                    "model_family": config.model_family,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    if not evaluations:
        raise ResearchValidationError("all narrow model candidate evaluations failed")
    selected: dict[str, dict[str, Any]] = {}
    for result in evaluations:
        family = str(result["model_family"])
        current = selected.get(family)
        if current is None or candidate_rank_key(result) > candidate_rank_key(current):
            selected[family] = result
    output_dir = output_dir.resolve()
    packages: list[dict[str, Any]] = []
    oos_artifacts: list[dict[str, Any]] = []
    dataset_token = dataset_version.removeprefix("sha256:")[:12]
    for family, selection in sorted(selected.items()):
        detailed = evaluate_narrow_candidate(
            labeled_rows,
            config=NarrowCandidateConfig(
                candidate_id=str(selection["candidate_id"]),
                model_family=str(selection["model_family"]),
                feature_columns=tuple(str(item) for item in selection["feature_columns"]),
                algorithm=str(selection["algorithm"]),
                alpha=float(selection["alpha"]),
                huber_delta=float(selection.get("huber_delta", 1.35)),
            ),
            target_name=target_name,
            purge_size=purge_size,
            embargo_size=embargo_size,
            forecast_horizon_seconds=forecast_horizon_seconds,
            expanding=expanding,
            execution_assumptions=assumptions,
            include_oos_observations=True,
            **split,
        )
        oos_rows = list(detailed.pop("oos_observations"))
        for index, evaluation in enumerate(evaluations):
            if evaluation["candidate_id"] == detailed["candidate_id"]:
                evaluations[index] = detailed
                break
        selected[family] = detailed
        selection = detailed
        artifact = train_selected_artifact(
            labeled_rows,
            selected_result=selection,
            target_name=target_name,
            feature_schema_version=feature_schema_version,
            forecast_horizon_seconds=forecast_horizon_seconds,
            dataset_version=dataset_version,
        )
        version = f"r1-{dataset_token}-{artifact['candidate_fingerprint'][:8]}"
        oos_descriptor = write_oos_observations(
            oos_rows,
            output_dir / "oos" / f"{family}-{version}.jsonl",
            overwrite=overwrite,
        )
        artifact["oos_observations"] = {
            key: value for key, value in oos_descriptor.items() if key != "path"
        }
        package_path = output_dir / "challengers" / f"{family}-{version}.zip"
        package = package_narrow_artifact(artifact, package_path, overwrite=overwrite)
        oos_artifacts.append(
            {
                **oos_descriptor,
                "candidate_id": artifact["candidate_id"],
                "candidate_fingerprint": artifact["candidate_fingerprint"],
                "model_family": artifact["model_family"],
                "forecast_horizon_seconds": forecast_horizon_seconds,
                "execution_assumptions": assumptions.model_dump(),
            }
        )
        selection["oos_observations"] = oos_descriptor
        packages.append(
            {
                **package,
                "name": f"local-{family}",
                "model_id": f"local-{family}",
                "version": version,
                "model_family": artifact["model_family"],
                "feature_schema_version": feature_schema_version,
                "feature_columns": artifact["feature_columns"],
                "preprocessing_version": artifact["preprocessing_version"],
                "training_dataset_version": dataset_version,
                "forecast_horizon_seconds": forecast_horizon_seconds,
                "metrics": artifact["metrics"],
                "candidate_id": artifact["candidate_id"],
                "candidate_fingerprint": artifact["candidate_fingerprint"],
                "training_period": artifact["training_period"],
                "oos_observations": oos_descriptor,
                "execution_assumptions": assumptions.model_dump(),
                "lifecycle_state": "TRAINED",
                "automatic_promotion": False,
            }
        )
    selected_ids = {item["candidate_id"] for item in selected.values()}
    for evaluation in evaluations:
        evaluation["selected_for_challenger"] = evaluation["candidate_id"] in selected_ids
    return {
        "status": "completed",
        "paper_only": True,
        "automatic_promotion": False,
        "dataset_version": dataset_version,
        "feature_schema_version": feature_schema_version,
        "target_name": target_name,
        "forecast_horizon_seconds": forecast_horizon_seconds,
        "annualization_basis": "forecast_horizon_seconds",
        "execution_assumptions": assumptions.model_dump(),
        "data_period": _training_period(labeled_rows),
        "label_inventory": inventory.model_dump(),
        "walk_forward": {
            **split,
            "purge_size": purge_size,
            "embargo_size": embargo_size,
            "expanding": expanding,
            "random_shuffle": False,
        },
        "candidate_count": len(configurations),
        "successful_candidate_count": len(evaluations),
        "candidate_failures": failures,
        "unavailable_model_families": list(unavailable),
        "candidates": evaluations,
        "challenger_packages": packages,
        "oos_observation_artifacts": oos_artifacts,
        "registered_challengers": [],
        "promotions": [],
        "limitations": [
            "Artifacts are deterministic linear ridge/Huber-ridge baselines, not deep or order-book models.",
            "Missing feature values use the same per-feature defaults as the live schema.",
            "Offline rank selects a challenger configuration only; it is not a champion-promotion decision.",
        ],
    }
