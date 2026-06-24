from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


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
        from lightgbm import LGBMRegressor

        return LGBMRegressor(n_estimators=500, learning_rate=0.03, max_depth=-1, random_state=42)
    if model_type == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=5, random_state=42, objective="reg:squarederror")
    raise ValueError(f"Unsupported model type: {model_type}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a stronger model locally from an exported Anata dataset.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-type", default="sklearn_hist_gradient_boosting", choices=["sklearn_hist_gradient_boosting", "random_forest", "lightgbm", "xgboost"])
    parser.add_argument("--target", default="target_trade_quality_score")
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    import joblib
    import pandas as pd

    frame = pd.read_csv(args.dataset).sort_values(["symbol", "as_of"])
    if args.target not in frame.columns:
        raise SystemExit(f"Dataset is missing target column {args.target}")
    feature_columns = _feature_columns(frame, args.target)
    if not feature_columns:
        raise SystemExit("No numeric feature columns found")
    train_frame = frame.dropna(subset=[args.target]).copy()
    x = train_frame[feature_columns].fillna(0.0).astype(float)
    y = train_frame[args.target].astype(float)
    model = _make_model(args.model_type)
    model.fit(x, y)

    predictions = model.predict(x)
    directional_accuracy = float(((predictions > 0) == (y.to_numpy() > 0)).mean())
    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / f"model_{version}.joblib"
    metadata_path = args.out_dir / f"model_{version}.json"
    joblib.dump(model, model_path)
    metadata = {
        "model_id": f"{args.model_type}:{version}",
        "name": args.model_type,
        "version": version,
        "model_type": args.model_type,
        "model_file": model_path.name,
        "feature_schema_version": str(train_frame["feature_schema_version"].iloc[-1]) if "feature_schema_version" in train_frame else "price-news-v3",
        "feature_columns": feature_columns,
        "target": args.target,
        "training_dataset_path": str(args.dataset),
        "training_dataset_hash": _dataset_hash(args.dataset),
        "metrics": {
            "train_rows": int(len(train_frame)),
            "directional_accuracy_in_sample": directional_accuracy,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"model": str(model_path), "metadata": str(metadata_path), "metrics": metadata["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
