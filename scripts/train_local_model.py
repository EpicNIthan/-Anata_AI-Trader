from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


NON_FEATURE_COLUMNS = {
    "feature_id",
    "training_feature_id",
    "symbol",
    "as_of",
    "feature_schema_version",
    "trend",
    "final_ai_input",
    "target_next_price_change",
    "target_future_return_5m",
    "target_future_return_15m",
    "target_future_return_1h",
    "target_future_return_4h",
    "target_max_upside_1h",
    "target_max_drawdown_1h",
    "target_stop_loss_hit_first",
    "target_take_profit_hit_first",
    "target_direction_15m",
    "target_trade_quality_score",
}


def _dataset_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_columns(frame, target: str) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        if column in NON_FEATURE_COLUMNS or column == target:
            continue
        try:
            frame[column].astype(float)
        except (TypeError, ValueError):
            continue
        columns.append(column)
    return columns


def _make_model(model_type: str):
    if model_type == "sklearn_hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, l2_regularization=0.01)
    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
    if model_type == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except Exception as exc:
            raise SystemExit(
                "LightGBM is not installed or failed to import. Install it locally with "
                "`pip install lightgbm`, or use `--model-type sklearn_hist_gradient_boosting`."
            ) from exc
        return LGBMRegressor(n_estimators=500, learning_rate=0.03, max_depth=-1, random_state=42)
    if model_type == "xgboost":
        try:
            from xgboost import XGBRegressor
        except Exception as exc:
            raise SystemExit(
                "XGBoost is not installed or failed to import. Install it locally with "
                "`pip install xgboost`, or use `--model-type sklearn_hist_gradient_boosting`."
            ) from exc
        return XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=5, random_state=42, objective="reg:squarederror")
    raise ValueError(f"Unsupported model type: {model_type}")


def _metrics(predictions, actual) -> dict[str, float]:
    import numpy as np

    errors = predictions - actual
    return {
        "directional_accuracy": float(((predictions > 0) == (actual > 0)).mean()),
        "mse": float(np.mean(errors**2)),
        "mae": float(np.mean(np.abs(errors))),
    }


def _feature_importance(model: Any, feature_columns: list[str]) -> dict[str, float]:
    values = getattr(model, "feature_importances_", None)
    if values is None:
        return {}
    pairs = sorted(zip(feature_columns, values), key=lambda item: float(item[1]), reverse=True)
    return {name: float(value) for name, value in pairs}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a stronger model locally from an exported Anata dataset.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-type", default="sklearn_hist_gradient_boosting", choices=["sklearn_hist_gradient_boosting", "random_forest", "lightgbm", "xgboost"])
    parser.add_argument("--target", default="target_trade_quality_score")
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    parser.add_argument("--validation-size", type=float, default=0.25)
    parser.add_argument("--dry-run", action="store_true", help="Validate dataset labels/features without fitting a model.")
    args = parser.parse_args()

    import pandas as pd

    frame = pd.read_csv(args.dataset)
    if "as_of" not in frame.columns:
        raise SystemExit("Dataset is missing as_of column")
    frame["as_of"] = pd.to_datetime(frame["as_of"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["as_of"]).sort_values("as_of")
    if args.target not in frame.columns:
        raise SystemExit(f"Dataset is missing target column {args.target}")

    train_frame = frame.dropna(subset=[args.target]).copy()
    if len(train_frame) < 1000:
        print(f"WARNING: dataset has only {len(train_frame)} labeled rows. Wait for more data if possible.")
    date_span = train_frame["as_of"].max() - train_frame["as_of"].min()
    if date_span.total_seconds() < 2 * 24 * 3600:
        print(f"WARNING: dataset covers only {date_span}. Two or more days is recommended before training.")

    feature_columns = _feature_columns(train_frame, args.target)
    if not feature_columns:
        raise SystemExit("No numeric feature columns found")
    if args.dry_run:
        result = {
            "status": "ok",
            "labeled_rows": int(len(train_frame)),
            "feature_columns": len(feature_columns),
            "target": args.target,
            "dataset_days": float(date_span.total_seconds() / 86400.0),
        }
        print(json.dumps(result, indent=2))
        return

    import joblib

    split = max(1, int(len(train_frame) * (1.0 - min(max(args.validation_size, 0.05), 0.50))))
    if split >= len(train_frame):
        raise SystemExit("Not enough rows for validation split")
    training_slice = train_frame.iloc[:split].copy()
    validation_slice = train_frame.iloc[split:].copy()

    x_train = training_slice[feature_columns].fillna(0.0).astype(float)
    y_train = training_slice[args.target].astype(float)
    x_validation = validation_slice[feature_columns].fillna(0.0).astype(float)
    y_validation = validation_slice[args.target].astype(float)

    validation_model = _make_model(args.model_type)
    validation_model.fit(x_train, y_train)
    validation_predictions = validation_model.predict(x_validation)
    validation_metrics = _metrics(validation_predictions, y_validation.to_numpy())

    final_model = _make_model(args.model_type)
    x_all = train_frame[feature_columns].fillna(0.0).astype(float)
    y_all = train_frame[args.target].astype(float)
    final_model.fit(x_all, y_all)

    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / f"model_{version}.joblib"
    metadata_path = args.out_dir / f"model_{version}.json"
    joblib.dump(final_model, model_path)
    metrics = {
        "train_rows": int(len(training_slice)),
        "validation_rows": int(len(validation_slice)),
        "total_labeled_rows": int(len(train_frame)),
        "validation_directional_accuracy": validation_metrics["directional_accuracy"],
        "validation_mse": validation_metrics["mse"],
        "validation_mae": validation_metrics["mae"],
        "dataset_days": float(date_span.total_seconds() / 86400.0),
    }
    metadata = {
        "model_id": f"{args.model_type}:{version}",
        "name": args.model_type,
        "version": version,
        "model_type": args.model_type,
        "model_file": model_path.name,
        "status": "candidate",
        "activation_mode": "manual",
        "feature_schema_version": str(train_frame["feature_schema_version"].iloc[-1]) if "feature_schema_version" in train_frame else "price-news-v3",
        "feature_columns": feature_columns,
        "target": args.target,
        "training_dataset_path": str(args.dataset),
        "training_dataset_hash": _dataset_hash(args.dataset),
        "metrics": metrics,
        "feature_importance": _feature_importance(final_model, feature_columns),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"model": str(model_path), "metadata": str(metadata_path), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
