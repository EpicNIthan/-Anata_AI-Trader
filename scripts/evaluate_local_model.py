from __future__ import annotations

import argparse
import json
from pathlib import Path


def _max_drawdown(equity_curve: list[float]) -> float:
    peak = equity_curve[0] if equity_curve else 0.0
    worst = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        if peak:
            worst = min(worst, (value - peak) / peak)
    return abs(worst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a local Anata model with a time-based split.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--fee-rate", type=float, default=0.0004)
    parser.add_argument("--test-size", type=float, default=0.25)
    args = parser.parse_args()

    import joblib
    import pandas as pd

    metadata_path = args.metadata or args.model.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(args.dataset).sort_values("as_of")
    target = metadata.get("target") or "target_trade_quality_score"
    returns = "target_future_return_15m" if "target_future_return_15m" in frame.columns else "target_next_price_change"
    feature_columns = list(metadata["feature_columns"])
    frame = frame.dropna(subset=[target]).copy()
    split = max(1, int(len(frame) * (1.0 - args.test_size)))
    test = frame.iloc[split:].copy()
    if test.empty:
        raise SystemExit("Not enough rows for test split")
    model = joblib.load(args.model)
    predictions = model.predict(test[feature_columns].fillna(0.0).astype(float))
    realized = test[returns].fillna(0.0).astype(float).to_numpy()
    actions = [1 if pred > args.fee_rate * 2 else (-1 if pred < -args.fee_rate * 2 else 0) for pred in predictions]
    trade_returns = [(action * ret - (args.fee_rate * 2 if action else 0.0)) for action, ret in zip(actions, realized)]
    equity = [1.0]
    for ret in trade_returns:
        equity.append(equity[-1] * (1.0 + ret))
    traded = [ret for action, ret in zip(actions, trade_returns) if action]
    metrics = {
        "directional_accuracy": float(((predictions > 0) == (realized > 0)).mean()),
        "net_return_after_fees": float(equity[-1] - 1.0),
        "max_drawdown": float(_max_drawdown(equity)),
        "simulated_win_rate": float(sum(1 for ret in traded if ret > 0) / len(traded)) if traded else 0.0,
        "number_of_trades": int(len(traded)),
        "test_rows": int(len(test)),
        "hold_baseline_return": 0.0,
        "rule_based_baseline": "not_run",
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
