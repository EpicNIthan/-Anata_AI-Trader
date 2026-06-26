from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
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
    "news_converter_model",
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


def _max_drawdown(equity_curve: list[float]) -> float:
    peak = equity_curve[0] if equity_curve else 1.0
    worst = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        if peak:
            worst = min(worst, (value - peak) / peak)
    return abs(worst)


def _feature_columns(frame, target: str) -> list[str]:
    columns = []
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

        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, l2_regularization=0.02)
    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(n_estimators=350, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1)
    if model_type == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except Exception as exc:
            raise RuntimeError("LightGBM is not installed. Skipping lightgbm. Install with: pip install lightgbm") from exc
        return LGBMRegressor(n_estimators=700, learning_rate=0.025, random_state=42)
    if model_type == "xgboost":
        try:
            from xgboost import XGBRegressor
        except Exception as exc:
            raise RuntimeError("XGBoost is not installed. Skipping xgboost. Install with: pip install xgboost") from exc
        return XGBRegressor(n_estimators=700, learning_rate=0.025, max_depth=5, random_state=42, objective="reg:squarederror")
    raise ValueError(f"Unsupported model type: {model_type}")


def _simulate(predictions, realized_returns, *, fee_rate: float, min_edge: float) -> dict[str, Any]:
    threshold = max(min_edge, fee_rate * 2)
    actions = [1 if pred > threshold else (-1 if pred < -threshold else 0) for pred in predictions]
    trade_returns = [(action * ret - (fee_rate * 2 if action else 0.0)) for action, ret in zip(actions, realized_returns)]
    equity = [1.0]
    for ret in trade_returns:
        equity.append(equity[-1] * (1.0 + float(ret)))
    traded = [ret for action, ret in zip(actions, trade_returns) if action]
    wins = [ret for ret in traded if ret > 0]
    return {
        "net_return_after_fees": float(equity[-1] - 1.0),
        "max_drawdown": float(_max_drawdown(equity)),
        "number_of_trades": int(len(traded)),
        "skipped_no_trade_count": int(len(actions) - len(traded)),
        "simulated_win_rate": float(len(wins) / len(traded)) if traded else 0.0,
        "average_return_per_trade": float(sum(traded) / len(traded)) if traded else 0.0,
        "average_realized_return_after_fees": float(sum(trade_returns) / len(trade_returns)) if trade_returns else 0.0,
    }


def _rule_based_predictions(test):
    import numpy as np

    candle = test["candle_return_5m"].to_numpy(dtype=float) if "candle_return_5m" in test else np.zeros(len(test))
    sentiment = test["sentiment_score"].to_numpy(dtype=float) if "sentiment_score" in test else np.zeros(len(test))
    risk = test["risk_score"].to_numpy(dtype=float) if "risk_score" in test else np.zeros(len(test))
    crowd = test["trader_crowd_score"].to_numpy(dtype=float) if "trader_crowd_score" in test else np.zeros(len(test))
    return candle + (sentiment * 0.002) + (crowd * 0.001) - (risk * 0.002)


def _directional_accuracy(predictions, actual) -> float:
    return float(((predictions > 0) == (actual > 0)).mean())


def _data_readiness(labeled_rows: int, dataset_days: float) -> dict[str, Any]:
    if labeled_rows < 1000 or dataset_days < 1:
        rank = "D"
        use = "pipeline test only"
    elif labeled_rows < 20000 or dataset_days < 3:
        rank = "C-"
        use = "early experiment; do not activate unless safety checks pass"
    elif labeled_rows < 75000 or dataset_days < 7:
        rank = "C/B-"
        use = "experiment model; compare against Bot carefully"
    elif labeled_rows < 250000 or dataset_days < 14:
        rank = "B"
        use = "first serious paper-test candidate"
    else:
        rank = "A"
        use = "stronger paper-test candidate, still not live-money proof"
    return {"rank": rank, "use": use, "labeled_rows": labeled_rows, "dataset_days": dataset_days}


def _failure_advice(candidates: list[dict[str, Any]], dataset_days: float) -> list[str]:
    advice = [
        "Training completed, but no model was packaged because every candidate failed safety checks.",
        "Keep Paper Runner on Bot and keep collecting more days before activating Trained AI.",
    ]
    if dataset_days < 3:
        advice.append("24 hours is enough to test the pipeline, but usually too short for a reliable trading model. Aim for 3-7 days for the next attempt.")
    if candidates:
        best = max(candidates, key=lambda item: item.get("metrics", {}).get("net_return_after_fees", -999.0))
        metrics = best.get("metrics", {})
        if metrics.get("net_return_after_fees", 0) <= 0:
            advice.append("The best candidate still had negative net return after fees, so uploading it would teach Railway to trade badly.")
        if metrics.get("max_drawdown", 0) >= 0.15:
            advice.append("Drawdown was too high; the model is overtrading or learning a weak/unstable pattern.")
        if metrics.get("number_of_trades", 0) > metrics.get("test_rows", 0) * 0.50:
            advice.append("The model traded too often. Retry later with a higher edge, for example --min-edge 0.003 or --min-edge 0.005.")
    return advice


def _feature_importance(model: Any, feature_columns: list[str]) -> dict[str, float]:
    values = getattr(model, "feature_importances_", None)
    if values is None:
        return {}
    pairs = sorted(zip(feature_columns, values), key=lambda item: float(item[1]), reverse=True)
    return {name: float(value) for name, value in pairs[:80]}


def _package_model(model_path: Path, metadata: dict[str, Any], output_dir: Path) -> Path:
    package_path = output_dir / f"model_package_{metadata['version']}.zip"
    metadata["model_file"] = model_path.name
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(model_path, model_path.name)
        archive.writestr("metadata.json", json.dumps(metadata, indent=2, default=str))
    return package_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multiple local models and package the best safe candidate.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--target", default="target_trade_quality_score")
    parser.add_argument("--return-column", default="target_future_return_15m")
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--fee-rate", type=float, default=0.0004)
    parser.add_argument("--min-edge", type=float, default=0.001)
    parser.add_argument(
        "--model-types",
        default="sklearn_hist_gradient_boosting,random_forest,lightgbm,xgboost",
        help="Comma-separated model types.",
    )
    parser.add_argument("--min-trades", type=int, default=10)
    args = parser.parse_args()

    try:
        import joblib
        import numpy as np
        import pandas as pd
    except Exception as exc:
        raise SystemExit("Install local training dependencies first: pip install -r requirements-local-training.txt") from exc

    frame = pd.read_csv(args.dataset)
    if "as_of" not in frame.columns:
        raise SystemExit("Dataset is missing as_of column.")
    frame["as_of"] = pd.to_datetime(frame["as_of"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["as_of"]).sort_values("as_of")
    if args.target not in frame.columns:
        raise SystemExit(f"Dataset is missing target column {args.target}.")
    return_column = args.return_column if args.return_column in frame.columns else args.target
    train_frame = frame.dropna(subset=[args.target, return_column]).copy()
    if len(train_frame) < 100:
        raise SystemExit(f"Only {len(train_frame)} labeled rows found. Build more labels or collect more data first.")
    feature_columns = _feature_columns(train_frame, args.target)
    if not feature_columns:
        raise SystemExit("No numeric feature columns found.")
    dataset_days = float((train_frame["as_of"].max() - train_frame["as_of"].min()).total_seconds() / 86400.0)
    split = max(1, int(len(train_frame) * (1.0 - min(max(args.test_size, 0.05), 0.50))))
    if split >= len(train_frame):
        raise SystemExit("Not enough rows for time-based test split.")
    training = train_frame.iloc[:split].copy()
    test = train_frame.iloc[split:].copy()
    x_train = training[feature_columns].fillna(0.0).astype(float)
    y_train = training[args.target].astype(float)
    x_test = test[feature_columns].fillna(0.0).astype(float)
    realized_returns = test[return_column].fillna(0.0).astype(float).to_numpy()

    baseline_predictions = _rule_based_predictions(test)
    baseline_metrics = _simulate(baseline_predictions, realized_returns, fee_rate=args.fee_rate, min_edge=args.min_edge)
    baseline_metrics["directional_accuracy"] = _directional_accuracy(baseline_predictions, realized_returns)

    candidates: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}
    for model_type in [item.strip() for item in args.model_types.split(",") if item.strip()]:
        try:
            model = _make_model(model_type)
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
        except Exception as exc:
            skipped[model_type] = str(exc)
            continue
        metrics = _simulate(predictions, realized_returns, fee_rate=args.fee_rate, min_edge=args.min_edge)
        metrics["directional_accuracy"] = _directional_accuracy(predictions, realized_returns)
        metrics["average_predicted_return"] = float(np.mean(predictions)) if len(predictions) else 0.0
        metrics["test_rows"] = int(len(test))
        metrics["model_type"] = model_type
        metrics["beats_rule_based_baseline"] = metrics["net_return_after_fees"] > baseline_metrics["net_return_after_fees"]
        warnings = []
        if metrics["directional_accuracy"] > 0.75:
            warnings.append("Unrealistically high directional accuracy; possible leakage/overfitting.")
        if metrics["number_of_trades"] < args.min_trades:
            warnings.append("Too few simulated trades.")
        if metrics["net_return_after_fees"] <= 0:
            warnings.append("Net return after fees is not positive.")
        if metrics["max_drawdown"] >= 0.15:
            warnings.append("Max drawdown is above preferred 0.15.")
        if not metrics["beats_rule_based_baseline"]:
            warnings.append("Model did not beat rule-based baseline.")
        passed = (
            metrics["directional_accuracy"] > 0.51
            and metrics["net_return_after_fees"] > 0
            and metrics["max_drawdown"] < 0.15
            and metrics["number_of_trades"] >= args.min_trades
            and metrics["beats_rule_based_baseline"]
            and metrics["directional_accuracy"] <= 0.75
        )
        candidates.append({"model_type": model_type, "metrics": metrics, "warnings": warnings, "passed": passed})

    passed_candidates = [candidate for candidate in candidates if candidate["passed"]]
    best = max(
        passed_candidates,
        key=lambda item: (
            item["metrics"]["net_return_after_fees"],
            -item["metrics"]["max_drawdown"],
            item["metrics"]["directional_accuracy"],
        ),
        default=None,
    )
    result: dict[str, Any] = {
        "status": "trained_failed_safety_checks" if candidates else "no_model_selected",
        "dataset": str(args.dataset),
        "target": args.target,
        "return_column": return_column,
        "train_rows": int(len(training)),
        "test_rows": int(len(test)),
        "total_labeled_rows": int(len(train_frame)),
        "dataset_days": dataset_days,
        "feature_columns": len(feature_columns),
        "data_readiness": _data_readiness(int(len(train_frame)), dataset_days),
        "baseline": baseline_metrics,
        "candidates": candidates,
        "skipped_models": skipped,
        "failed_safety_checks": True,
        "next_steps": _failure_advice(candidates, dataset_days),
        "message": "Training completed, but no model passed safety checks. Nothing was packaged or uploaded.",
    }
    if best is None:
        print(json.dumps(result, indent=2, default=str))
        return

    final_model = _make_model(best["model_type"])
    final_model.fit(train_frame[feature_columns].fillna(0.0).astype(float), train_frame[args.target].astype(float))
    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / f"model_{version}.joblib"
    joblib.dump(final_model, model_path)
    metadata = {
        "model_id": f"{best['model_type']}:{version}",
        "name": best["model_type"],
        "version": version,
        "model_type": best["model_type"],
        "status": "candidate",
        "activation_mode": "manual",
        "feature_schema_version": str(train_frame["feature_schema_version"].dropna().iloc[-1]) if "feature_schema_version" in train_frame else "local-raw-v1",
        "feature_columns": feature_columns,
        "target": args.target,
        "return_column": return_column,
        "training_dataset_path": str(args.dataset),
        "training_dataset_hash": _dataset_hash(args.dataset),
        "metrics": best["metrics"] | {
            "paper_test_readiness": "PASS",
            "rule_based_baseline": baseline_metrics,
            "train_rows": int(len(training)),
            "total_labeled_rows": int(len(train_frame)),
            "dataset_days": dataset_days,
        },
        "feature_importance": _feature_importance(final_model, feature_columns),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = args.out_dir / f"model_{version}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    package_path = _package_model(model_path, metadata, args.out_dir)
    result.update(
        {
            "status": "passed",
            "message": "Best model passed safety checks and was packaged as a candidate. Upload/activate explicitly.",
            "failed_safety_checks": False,
            "next_steps": ["Upload the model package, then activate it from the dashboard Training tab."],
            "best_model": best,
            "model": str(model_path),
            "metadata": str(metadata_path),
            "package": str(package_path),
            "model_id": metadata["model_id"],
        }
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
