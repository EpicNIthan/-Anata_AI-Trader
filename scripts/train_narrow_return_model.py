"""Train one frozen, forecast-only linear return model.

The output contract contains only an expected-return estimator.  It deliberately
cannot encode position size, leverage, stops, exits, or order instructions.  Rows
are split chronologically through ``app.research`` and observations whose forward
label crosses a split boundary are purged before fitting or validation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.research import chronological_split, ensure_utc, evaluate_predictions  # noqa: E402
from app.research.evaluation import row_timestamp  # noqa: E402
from app.features.schema import model_input_columns_for_schema  # noqa: E402
from scripts.research_utils import read_research_rows, sha256_file  # noqa: E402


FORBIDDEN_TARGET_TOKENS = (
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
NON_FEATURE_NAMES = {
    "actual_return",
    "available_to_model_time",
    "decision_time",
    "event_time",
    "label",
    "label_available_time",
    "label_end",
    "label_end_time",
    "label_return",
    "prediction",
    "realized_return",
    "signal_time",
    "symbol",
    "target",
    "time",
    "timestamp",
    "as_of",
}
FORBIDDEN_FEATURE_TOKENS = (
    "future_return",
    "hit_first",
    "label_",
    "max_drawdown",
    "max_upside",
    "next_price",
)


def _nested_value(row: Mapping[str, Any], name: str) -> Any:
    if name in row:
        return row[name]
    for container_name in ("values", "features", "feature_values"):
        container = row.get(container_name)
        if isinstance(container, Mapping) and name in container:
            return container[name]
    raise KeyError(name)


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _feature_file(path: Path | None) -> list[str]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        payload = payload.get("feature_columns", payload.get("required_features", payload.get("features")))
    if not isinstance(payload, list):
        raise ValueError("features file must contain a JSON list or an object with feature_columns")
    return [str(value).strip() for value in payload if str(value).strip()]


def _infer_features(
    rows: Sequence[Mapping[str, Any]],
    target: str,
    *,
    allowed_features: set[str] | None = None,
) -> list[str]:
    if not rows:
        return []
    first = rows[0]
    values = first.get("values") if isinstance(first.get("values"), Mapping) else first
    features: list[str] = []
    for name in sorted(str(key) for key in values):
        lowered = name.lower()
        if name == target or lowered in NON_FEATURE_NAMES or lowered.startswith("target_"):
            continue
        if allowed_features is not None and name not in allowed_features:
            continue
        try:
            _finite(_nested_value(first, name), label=name)
        except (KeyError, ValueError):
            continue
        features.append(name)
    return features


def _dataset_schema_version(rows: Sequence[Mapping[str, Any]], requested: str | None) -> str:
    observed = {
        str(row.get("feature_schema_version") or row.get("schema_version") or "").strip()
        for row in rows
        if str(row.get("feature_schema_version") or row.get("schema_version") or "").strip()
    }
    if len(observed) > 1:
        raise ValueError("input contains multiple feature schema versions; train each schema separately")
    dataset_schema = next(iter(observed), None)
    if requested and dataset_schema and requested != dataset_schema:
        raise ValueError(
            f"requested feature schema {requested!r} does not match dataset schema {dataset_schema!r}"
        )
    schema = requested or dataset_schema
    if not schema:
        raise ValueError("feature schema version is missing; pass --feature-schema-version explicitly")
    return schema


def _validate_contract(features: Sequence[str], target: str) -> None:
    lowered_target = target.lower()
    if any(token in lowered_target for token in FORBIDDEN_TARGET_TOKENS):
        raise ValueError("target must be a future-return forecast, never a sizing, leverage, stop, exit, or order target")
    if "return" not in lowered_target and "price_change" not in lowered_target:
        raise ValueError("target must describe a future return or price change")
    if not features:
        raise ValueError("at least one numeric feature is required")
    normalized = [name.strip() for name in features]
    if any(not name for name in normalized) or len(normalized) != len(set(normalized)):
        raise ValueError("feature columns must be non-empty and unique")
    unsafe = [
        name
        for name in normalized
        if name == target
        or name.lower().startswith("target_")
        or name.lower() in NON_FEATURE_NAMES
        or any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if unsafe:
        raise ValueError("label/target columns cannot be model features: " + ", ".join(unsafe))


def _label_available_at(row: Mapping[str, Any], *, horizon_seconds: int) -> datetime:
    decision_time = row_timestamp(row)
    for name in ("label_available_time", "label_end_time", "label_end"):
        value = row.get(name)
        if value not in (None, ""):
            available = ensure_utc(value, field_name=name)
            if available < decision_time:
                raise ValueError(f"{name} cannot precede the decision timestamp")
            return available
    return decision_time + timedelta(seconds=horizon_seconds)


def _purge_before(
    indices: Sequence[int],
    rows: Sequence[Mapping[str, Any]],
    *,
    next_period_start: datetime,
    horizon_seconds: int,
) -> list[int]:
    return [
        index
        for index in indices
        if _label_available_at(rows[index], horizon_seconds=horizon_seconds) <= next_period_start
    ]


def _atomic_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing output: {path}; pass --overwrite")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def train_artifact(
    *,
    rows: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str],
    target_name: str,
    feature_schema_version: str,
    forecast_horizon_seconds: int,
    train_fraction: float,
    validation_fraction: float,
    alpha: float,
    transaction_cost: float,
    minimum_rows_per_split: int,
    dataset_version: str,
) -> dict[str, Any]:
    """Fit and evaluate a deterministic ridge model without time-series shuffling."""

    if forecast_horizon_seconds <= 0:
        raise ValueError("forecast horizon must be positive")
    if validation_fraction <= 0:
        raise ValueError("validation fraction must be positive for calibration and held-out testing")
    if minimum_rows_per_split < 1:
        raise ValueError("minimum rows per split must be positive")
    if alpha < 0:
        raise ValueError("ridge alpha cannot be negative")
    if transaction_cost < 0:
        raise ValueError("transaction cost cannot be negative")
    _validate_contract(feature_columns, target_name)

    normalized_rows: list[dict[str, Any]] = []
    dropped_unlabeled_rows = 0
    for row in rows:
        candidate = dict(row)
        try:
            _finite(_nested_value(candidate, target_name), label=target_name)
        except (KeyError, ValueError):
            dropped_unlabeled_rows += 1
            continue
        normalized_rows.append(candidate)
    if len(normalized_rows) < 3:
        raise ValueError("too few rows with a finite future-return label")
    split = chronological_split(
        normalized_rows,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    validation_start = min(row_timestamp(normalized_rows[index]) for index in split.validation_indices)
    test_start = min(row_timestamp(normalized_rows[index]) for index in split.test_indices)
    train_indices = _purge_before(
        split.train_indices,
        normalized_rows,
        next_period_start=validation_start,
        horizon_seconds=forecast_horizon_seconds,
    )
    validation_indices = _purge_before(
        split.validation_indices,
        normalized_rows,
        next_period_start=test_start,
        horizon_seconds=forecast_horizon_seconds,
    )
    test_indices = list(split.test_indices)
    counts = {
        "train": len(train_indices),
        "validation": len(validation_indices),
        "test": len(test_indices),
    }
    if any(count < minimum_rows_per_split for count in counts.values()):
        raise ValueError(
            "too few rows after chronological split and label-overlap purge: "
            + ", ".join(f"{name}={count}" for name, count in counts.items())
        )

    import numpy as np

    imputed_feature_values = 0

    def feature_value(row: Mapping[str, Any], name: str) -> float:
        nonlocal imputed_feature_values
        try:
            return _finite(_nested_value(row, name), label=name)
        except (KeyError, ValueError):
            imputed_feature_values += 1
            return 0.0

    def matrix(indices: Sequence[int]) -> tuple[Any, Any]:
        x = np.asarray(
            [[feature_value(normalized_rows[index], name) for name in feature_columns] for index in indices],
            dtype=float,
        )
        y = np.asarray(
            [_finite(_nested_value(normalized_rows[index], target_name), label=target_name) for index in indices],
            dtype=float,
        )
        return x, y

    x_train, y_train = matrix(train_indices)
    x_validation, y_validation = matrix(validation_indices)
    x_test, y_test = matrix(test_indices)
    feature_mean = np.mean(x_train, axis=0)
    feature_scale = np.std(x_train, axis=0)
    feature_scale = np.where(feature_scale > 0.0, feature_scale, 1.0)
    x_train_scaled = (x_train - feature_mean) / feature_scale
    target_mean = float(np.mean(y_train))
    centered_target = y_train - target_mean
    gram = x_train_scaled.T @ x_train_scaled
    penalty = np.eye(gram.shape[0], dtype=float) * alpha
    # pinv is deterministic and also handles alpha=0 with collinear features.
    standardized_coefficients = np.linalg.pinv(gram + penalty) @ x_train_scaled.T @ centered_target

    # Convert the standardized model back to raw feature space. Serving therefore
    # needs no mutable preprocessing object and can validate one immutable JSON file.
    raw_coefficients = np.asarray(standardized_coefficients, dtype=float) / feature_scale
    raw_intercept = float(target_mean - np.dot(feature_mean, raw_coefficients))
    validation_predictions = raw_intercept + x_validation @ raw_coefficients
    test_predictions = raw_intercept + x_test @ raw_coefficients
    validation_timestamps = [row_timestamp(normalized_rows[index]) for index in validation_indices]
    test_timestamps = [row_timestamp(normalized_rows[index]) for index in test_indices]

    def metrics(predictions: Any, actual: Any, timestamps: Sequence[datetime]) -> dict[str, Any]:
        research_metrics = evaluate_predictions(
            predictions.tolist(),
            actual.tolist(),
            transaction_costs=[transaction_cost] * len(actual),
            timestamps=timestamps,
        )
        return {
            **research_metrics,
            "mean_absolute_error": float(np.mean(np.abs(actual - predictions))),
            "root_mean_squared_error": float(math.sqrt(np.mean(np.square(actual - predictions)))),
        }

    validation_metrics = metrics(validation_predictions, y_validation, validation_timestamps)
    test_metrics = metrics(test_predictions, y_test, test_timestamps)
    scale = float(np.std(y_validation))
    calibration_score = max(
        0.0,
        min(1.0, 1.0 - float(validation_metrics["mean_absolute_error"]) / max(scale, 1e-12)),
    )
    trained_at = datetime.now(timezone.utc).isoformat()
    return {
        "artifact_type": "anata_narrow_return_forecast",
        "contract_version": 1,
        "feature_columns": list(feature_columns),
        "coefficients": [float(value) for value in raw_coefficients],
        "intercept": raw_intercept,
        "target_name": target_name,
        "feature_schema_version": feature_schema_version,
        "preprocessing_version": "raw-linear-v1",
        "missing_value_policy": {
            "strategy": "zero",
            "imputed_feature_values": imputed_feature_values,
            "dropped_unlabeled_rows": dropped_unlabeled_rows,
        },
        "forecast_horizon_seconds": forecast_horizon_seconds,
        "training_dataset_version": dataset_version,
        "model_family": "alpha.linear_return",
        "algorithm": "ridge_linear_regression",
        "hyperparameters": {"alpha": alpha, "shuffle": False},
        "split": {
            "method": "chronological_train_validation_test_with_forward_label_purge",
            "train_fraction": train_fraction,
            "validation_fraction": validation_fraction,
            "counts_after_purge": counts,
            "train_period": split.train_period.model_dump() if split.train_period else None,
            "validation_period": split.validation_period.model_dump() if split.validation_period else None,
            "test_period": split.test_period.model_dump() if split.test_period else None,
            "forecast_horizon_seconds": forecast_horizon_seconds,
        },
        "metrics": {
            "calibration_score": calibration_score,
            "validation": validation_metrics,
            "held_out_test": test_metrics,
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
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a leakage-resistant forecast-only JSON model artifact.")
    parser.add_argument("--input", required=True, type=Path, help="Point-in-time CSV, JSON, or JSONL feature rows.")
    parser.add_argument("--output", required=True, type=Path, help="New .json model artifact.")
    parser.add_argument("--target", default="target_future_return_5m")
    feature_group = parser.add_mutually_exclusive_group()
    feature_group.add_argument("--features", help="Ordered, comma-separated numeric features.")
    feature_group.add_argument("--features-file", type=Path, help="JSON feature list/contract.")
    parser.add_argument("--feature-schema-version", help="Defaults to the single schema recorded in the dataset.")
    parser.add_argument("--forecast-horizon-seconds", type=int, default=300)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--transaction-cost", type=float, default=0.0008)
    parser.add_argument("--minimum-rows-per-split", type=int, default=20)
    parser.add_argument("--dataset-version", help="Defaults to SHA-256 of the input file.")
    parser.add_argument("--report", type=Path, help="Optional separate metrics/report JSON.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.suffix.lower() != ".json":
        raise SystemExit("--output must end in .json for the frozen linear artifact contract")
    rows = read_research_rows(args.input)
    if not rows:
        raise SystemExit("no input rows found")
    try:
        feature_schema_version = _dataset_schema_version(rows, args.feature_schema_version)
    except ValueError as exc:
        print(json.dumps({"status": "error", "error": str(exc), "paper_only": True}, indent=2), file=sys.stderr)
        return 2
    allowed_features = set(model_input_columns_for_schema(feature_schema_version))
    features = (
        [value.strip() for value in args.features.split(",") if value.strip()]
        if args.features
        else _feature_file(args.features_file)
    )
    if not features:
        features = _infer_features(rows, args.target, allowed_features=allowed_features)
    unsupported = sorted(set(features) - allowed_features)
    if unsupported:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": "features are not served by the live schema: " + ", ".join(unsupported),
                    "paper_only": True,
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2
    dataset_version = args.dataset_version or f"sha256:{sha256_file(args.input)}"
    try:
        artifact = train_artifact(
            rows=rows,
            feature_columns=features,
            target_name=args.target,
            feature_schema_version=feature_schema_version,
            forecast_horizon_seconds=args.forecast_horizon_seconds,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            alpha=args.alpha,
            transaction_cost=args.transaction_cost,
            minimum_rows_per_split=args.minimum_rows_per_split,
            dataset_version=dataset_version,
        )
        _atomic_json(args.output, artifact, overwrite=args.overwrite)
        if args.report:
            _atomic_json(
                args.report,
                {
                    "artifact": str(args.output.resolve()),
                    "dataset_version": dataset_version,
                    "target": args.target,
                    "features": features,
                    "split": artifact["split"],
                    "metrics": artifact["metrics"],
                    "paper_only": True,
                    "automatic_promotion": False,
                },
                overwrite=args.overwrite,
            )
    except (FileExistsError, KeyError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc), "paper_only": True}, indent=2), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "completed",
                "artifact": str(args.output.resolve()),
                "target": args.target,
                "feature_count": len(features),
                "split": artifact["split"],
                "held_out_test": artifact["metrics"]["held_out_test"],
                "next_step": "Register explicitly with scripts/manage_model_registry.py register-challenger; no promotion occurred.",
                "paper_only": True,
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
