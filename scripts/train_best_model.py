from __future__ import annotations

import argparse
import hashlib
import json
import sys
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
    "target_trade_action",
    "target_edge_threshold",
    "target_edge_aware_trade_score",
    "target_best_margin_pct",
    "target_best_leverage",
    "target_best_stop_loss_pct",
    "target_best_take_profit_pct",
    "target_best_hold_seconds",
}
PLAN_TARGETS = [
    "target_best_margin_pct",
    "target_best_leverage",
    "target_best_stop_loss_pct",
    "target_best_take_profit_pct",
    "target_best_hold_seconds",
]
REGIME_FEATURE_COLUMNS = [
    "regime_trend_strength",
    "regime_direction_score",
    "regime_volatility_score",
    "regime_news_shock_score",
    "regime_risk_off_score",
    "regime_liquidity_stress_score",
    "regime_breakout_pressure",
    "regime_mean_reversion_pressure",
    "regime_crowd_pressure",
]
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
DEMO_CONFIG: dict[str, float | int] = {
    "slippage_rate": 0.0002,
    "stress_cost_multiplier": 2.0,
    "max_demo_drawdown": 0.35,
    "max_trade_rate": 0.70,
    "min_trade_directional_accuracy": 0.50,
    "min_profitable_slices": 1,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


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


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


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


def _fit_model(model: Any, x_train, y_train, sample_weight=None) -> None:
    if sample_weight is None:
        model.fit(x_train, y_train)
        return
    try:
        model.fit(x_train, y_train, sample_weight=sample_weight)
    except TypeError:
        model.fit(x_train, y_train)


def _trade_cost(*, fee_rate: float, slippage_rate: float = 0.0, cost_multiplier: float = 1.0) -> float:
    return float(max(0.0, ((fee_rate + slippage_rate) * 2.0) * max(0.0, cost_multiplier)))


def _actions_from_predictions(predictions, *, fee_rate: float, min_edge: float):
    import numpy as np

    predictions = np.asarray(predictions, dtype=float)
    threshold = float(max(min_edge, fee_rate * 2.0))
    actions = np.where(predictions > threshold, 1, np.where(predictions < -threshold, -1, 0))
    return actions, threshold


def _longest_losing_streak(trade_returns) -> int:
    longest = 0
    current = 0
    for ret in trade_returns:
        if float(ret) <= 0.0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _raw_simulate(
    predictions,
    realized_returns,
    *,
    fee_rate: float,
    min_edge: float,
    slippage_rate: float = 0.0,
    cost_multiplier: float = 1.0,
) -> dict[str, Any]:
    import numpy as np

    predictions = np.asarray(predictions, dtype=float)
    realized_returns = np.asarray(realized_returns, dtype=float)
    actions, threshold = _actions_from_predictions(predictions, fee_rate=fee_rate, min_edge=min_edge)
    round_trip_cost = _trade_cost(fee_rate=fee_rate, slippage_rate=slippage_rate, cost_multiplier=cost_multiplier)
    trade_returns = np.where(actions != 0, (actions * realized_returns) - round_trip_cost, 0.0)

    equity = [1.0]
    for ret in trade_returns:
        equity.append(max(0.0, equity[-1] * (1.0 + float(ret))))

    traded = trade_returns[actions != 0]
    wins = traded[traded > 0]
    losses = traded[traded <= 0]
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) else 0.0
    average_win = float(wins.mean()) if len(wins) else 0.0
    average_loss = float(abs(losses.mean())) if len(losses) else 0.0
    long_count = int((actions == 1).sum())
    short_count = int((actions == -1).sum())
    trade_count = int(len(traded))

    return {
        "net_return_after_fees": float(equity[-1] - 1.0),
        "max_drawdown": float(_max_drawdown(equity)),
        "number_of_trades": trade_count,
        "skipped_no_trade_count": int((actions == 0).sum()),
        "trade_rate": float(trade_count / len(actions)) if len(actions) else 0.0,
        "long_trade_count": long_count,
        "short_trade_count": short_count,
        "action_bias": float((long_count - short_count) / trade_count) if trade_count else 0.0,
        "simulated_win_rate": float(len(wins) / trade_count) if trade_count else 0.0,
        "average_return_per_trade": float(traded.mean()) if trade_count else 0.0,
        "average_realized_return_after_fees": float(trade_returns.mean()) if len(trade_returns) else 0.0,
        "best_trade_return": float(traded.max()) if trade_count else 0.0,
        "worst_trade_return": float(traded.min()) if trade_count else 0.0,
        "average_win_return": average_win,
        "average_loss_return": average_loss,
        "payoff_ratio": float(average_win / average_loss) if average_loss > 0 else (999.0 if average_win > 0 else 0.0),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0),
        "longest_losing_streak": _longest_losing_streak(traded),
        "edge_threshold_used": threshold,
        "round_trip_cost_used": round_trip_cost,
    }


def _directional_accuracy(predictions, actual) -> float:
    import numpy as np

    predictions = np.asarray(predictions, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if len(predictions) == 0:
        return 0.0
    return float(((predictions > 0) == (actual > 0)).mean())


def _trade_directional_accuracy(predictions, realized_returns, *, fee_rate: float, min_edge: float) -> float:
    import numpy as np

    realized_returns = np.asarray(realized_returns, dtype=float)
    actions, _ = _actions_from_predictions(predictions, fee_rate=fee_rate, min_edge=min_edge)
    mask = actions != 0
    if not bool(mask.any()):
        return 0.0
    actual_direction = np.where(realized_returns[mask] > 0, 1, -1)
    return float((actions[mask] == actual_direction).mean())


def _slice_validation(
    predictions,
    realized_returns,
    *,
    fee_rate: float,
    min_edge: float,
    slippage_rate: float = 0.0,
    cost_multiplier: float = 1.0,
    slices: int = 4,
    min_rows: int = 25,
) -> dict[str, Any]:
    import numpy as np

    predictions = np.asarray(predictions, dtype=float)
    realized_returns = np.asarray(realized_returns, dtype=float)
    chunks = np.array_split(np.arange(len(predictions)), max(1, slices))
    metrics: list[dict[str, Any]] = []
    profitable = 0
    losing = 0
    evaluated = 0
    worst_net = None
    worst_drawdown = 0.0

    for index, chunk in enumerate(chunks, start=1):
        if len(chunk) < min_rows:
            metrics.append({"slice": index, "status": "too_few_rows", "rows": int(len(chunk))})
            continue
        result = _raw_simulate(
            predictions[chunk],
            realized_returns[chunk],
            fee_rate=fee_rate,
            min_edge=min_edge,
            slippage_rate=slippage_rate,
            cost_multiplier=cost_multiplier,
        )
        result["slice"] = index
        result["rows"] = int(len(chunk))
        result["status"] = "evaluated"
        metrics.append(result)
        evaluated += 1
        net_return = float(result["net_return_after_fees"])
        worst_net = net_return if worst_net is None else min(worst_net, net_return)
        worst_drawdown = max(worst_drawdown, float(result["max_drawdown"]))
        if net_return > 0.0:
            profitable += 1
        else:
            losing += 1

    return {
        "metrics": metrics,
        "summary": {
            "evaluated_slice_count": evaluated,
            "profitable_slice_count": profitable,
            "losing_slice_count": losing,
            "worst_slice_net_return": float(worst_net) if worst_net is not None else None,
            "worst_slice_drawdown": float(worst_drawdown),
            "min_rows_per_slice": min_rows,
        },
    }


def _simulate(predictions, realized_returns, *, fee_rate: float, min_edge: float) -> dict[str, Any]:
    metrics = _raw_simulate(predictions, realized_returns, fee_rate=fee_rate, min_edge=min_edge)
    metrics["trade_directional_accuracy"] = _trade_directional_accuracy(
        predictions,
        realized_returns,
        fee_rate=fee_rate,
        min_edge=min_edge,
    )
    metrics["stress_test"] = _raw_simulate(
        predictions,
        realized_returns,
        fee_rate=fee_rate,
        min_edge=min_edge,
        slippage_rate=float(DEMO_CONFIG["slippage_rate"]),
        cost_multiplier=float(DEMO_CONFIG["stress_cost_multiplier"]),
    )
    normal_slices = _slice_validation(predictions, realized_returns, fee_rate=fee_rate, min_edge=min_edge)
    stress_slices = _slice_validation(
        predictions,
        realized_returns,
        fee_rate=fee_rate,
        min_edge=min_edge,
        slippage_rate=float(DEMO_CONFIG["slippage_rate"]),
        cost_multiplier=float(DEMO_CONFIG["stress_cost_multiplier"]),
    )
    metrics["slice_summary"] = normal_slices["summary"]
    metrics["slice_metrics"] = normal_slices["metrics"]
    metrics["stress_slice_summary"] = stress_slices["summary"]
    metrics["stress_slice_metrics"] = stress_slices["metrics"]
    return metrics


def _rule_based_predictions(test):
    import numpy as np

    candle = test["candle_return_5m"].to_numpy(dtype=float) if "candle_return_5m" in test else np.zeros(len(test))
    sentiment = test["sentiment_score"].to_numpy(dtype=float) if "sentiment_score" in test else np.zeros(len(test))
    risk = test["risk_score"].to_numpy(dtype=float) if "risk_score" in test else np.zeros(len(test))
    crowd = test["trader_crowd_score"].to_numpy(dtype=float) if "trader_crowd_score" in test else np.zeros(len(test))
    return candle + (sentiment * 0.002) + (crowd * 0.001) - (risk * 0.002)


def _data_readiness(labeled_rows: int, dataset_days: float) -> dict[str, Any]:
    if labeled_rows < 1000 or dataset_days < 1:
        rank = "D"
        use = "pipeline test only"
    elif labeled_rows < 20000 or dataset_days < 3:
        rank = "C-"
        use = "early experiment; demo activation only if safety gate passes"
    elif labeled_rows < 75000 or dataset_days < 7:
        rank = "C/B-"
        use = "demo/paper experiment; compare against Bot carefully"
    elif labeled_rows < 250000 or dataset_days < 14:
        rank = "B"
        use = "first serious demo/paper candidate"
    else:
        rank = "A"
        use = "stronger demo/paper candidate, still not live-money proof"
    return {"rank": rank, "use": use, "labeled_rows": labeled_rows, "dataset_days": dataset_days}


def _failure_advice(candidates: list[dict[str, Any]], dataset_days: float) -> list[str]:
    advice = [
        "Training completed, but no model was packaged because every candidate failed demo/paper safety checks.",
        "Keep Paper Runner on Bot and keep collecting more days before activating Trained AI in the demo website.",
    ]
    if dataset_days < 3:
        advice.append("24 hours can test the pipeline, but 3-7 days gives the demo gate better evidence.")
    if candidates:
        best = max(candidates, key=lambda item: item.get("metrics", {}).get("net_return_after_fees", -999.0))
        metrics = best.get("metrics", {})
        gate = metrics.get("demo_safety_gate", {})
        failures = gate.get("hard_failures", [])
        if failures:
            advice.append("Best candidate failed demo gate checks: " + ", ".join(failures[:8]))
        if metrics.get("number_of_trades", 0) > metrics.get("test_rows", 0) * 0.70:
            advice.append("The model traded too often. Try a higher --min-edge such as 0.003 or 0.005.")
    return advice


def _feature_importance(model: Any, feature_columns: list[str]) -> dict[str, float]:
    values = getattr(model, "feature_importances_", None)
    if values is None:
        return {}
    pairs = sorted(zip(feature_columns, values), key=lambda item: float(item[1]), reverse=True)
    return {name: float(value) for name, value in pairs[:80]}


def _series(frame, column: str, default: float = 0.0):
    import pandas as pd

    if column in frame:
        return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)
    return pd.Series(default, index=frame.index, dtype="float64")


def _to_series(values, index):
    import pandas as pd

    return pd.Series(values, index=index).astype(float)


def _add_symbol_features(frame):
    if "symbol" not in frame.columns:
        return frame, []
    sys.path.insert(0, str(_repo_root()))
    try:
        from app.ai.symbol_identity import SYMBOL_FEATURE_COLUMNS, symbol_identity_values
    except Exception as exc:
        raise SystemExit(f"Could not import symbol identity features: {exc}") from exc
    import pandas as pd

    output = frame.copy()
    symbol_features = [symbol_identity_values(symbol) for symbol in output["symbol"]]
    symbol_frame = pd.DataFrame(symbol_features, index=output.index)
    for column in SYMBOL_FEATURE_COLUMNS:
        output[column] = symbol_frame[column].astype(float)
    return output, list(SYMBOL_FEATURE_COLUMNS)


def _add_regime_features(frame):
    import numpy as np

    output = frame.copy()
    trend = _series(output, "trend_score")
    candle_5m = _series(output, "candle_return_5m")
    volatility = _series(output, "volatility").abs()
    volume_change = _series(output, "volume_change")
    sentiment = _series(output, "sentiment_score")
    risk = _series(output, "risk_score")
    impact = _series(output, "impact_score")
    macro = _series(output, "macro_risk_score")
    world = _series(output, "world_risk_score")
    regulation = _series(output, "regulation_risk_score")
    fed = _series(output, "fed_risk_score")
    war = _series(output, "war_risk_score")
    security = _series(output, "security_risk_score")
    stablecoin = _series(output, "stablecoin_depeg_risk")
    liquidation = _series(output, "liquidation_spike_score")
    crowd = _series(output, "trader_crowd_score").abs()
    crowd_risk = _series(output, "crowd_risk_score")
    crowd_ratio = _series(output, "crowd_long_short_ratio", 1.0)
    open_interest_change = _series(output, "open_interest_change").abs()
    funding_rate = _series(output, "funding_rate").abs()

    trend_strength = _to_series(np.maximum(trend.abs(), (candle_5m.abs() * 80.0).clip(upper=1.0)), output.index).clip(lower=0.0, upper=1.0)
    direction_score = (trend + candle_5m * 40.0).clip(lower=-1.0, upper=1.0)
    volatility_score = _to_series(np.tanh(volatility * 120.0), output.index).clip(lower=0.0, upper=1.0)
    news_shock = _to_series(
        np.maximum.reduce([sentiment.abs() * impact.clip(lower=0.0), risk, regulation, fed, war, security]),
        output.index,
    ).clip(lower=0.0, upper=1.0)
    risk_off = _to_series(np.maximum.reduce([risk, macro, world, regulation, fed, war, security, stablecoin]), output.index).clip(lower=0.0, upper=1.0)
    liquidity_stress = _to_series(
        np.maximum.reduce([liquidation, (open_interest_change * 25.0).clip(upper=1.0), (funding_rate * 500.0).clip(upper=1.0)]),
        output.index,
    ).clip(lower=0.0, upper=1.0)
    breakout = (np.maximum(candle_5m.abs() * 70.0, volume_change.clip(lower=0.0) * 0.15) * np.maximum(0.25, trend_strength)).clip(lower=0.0, upper=1.0)
    mean_reversion = (volatility_score * (1.0 - trend_strength.clip(upper=1.0)) * (1.0 - news_shock.clip(upper=1.0))).clip(lower=0.0, upper=1.0)
    crowd_pressure = _to_series(np.maximum.reduce([crowd, crowd_risk, (crowd_ratio - 1.0).abs() * 0.25]), output.index).clip(lower=0.0, upper=1.0)

    output["regime_trend_strength"] = trend_strength
    output["regime_direction_score"] = direction_score
    output["regime_volatility_score"] = volatility_score
    output["regime_news_shock_score"] = news_shock
    output["regime_risk_off_score"] = risk_off
    output["regime_liquidity_stress_score"] = liquidity_stress
    output["regime_breakout_pressure"] = breakout
    output["regime_mean_reversion_pressure"] = mean_reversion
    output["regime_crowd_pressure"] = crowd_pressure
    return output


def _edge_threshold(frame, *, fee_rate: float, min_edge: float, hold_edge_multiplier: float):
    import numpy as np

    base = max(min_edge, fee_rate * 2.0 * hold_edge_multiplier)
    volatility = _series(frame, "volatility").abs()
    trend_strength = _series(frame, "regime_trend_strength")
    noisy_market_buffer = (volatility * 0.40).clip(lower=0.0, upper=0.006)
    weak_trend_buffer = ((1.0 - trend_strength.clip(lower=0.0, upper=1.0)) * 0.0005).clip(lower=0.0, upper=0.0005)
    return _to_series(np.maximum(base, base + noisy_market_buffer + weak_trend_buffer), frame.index).clip(lower=base, upper=0.020)


def _add_edge_aware_targets(frame, *, fee_rate: float, min_edge: float, hold_edge_multiplier: float):
    output = frame.copy()
    if "target_future_return_15m" not in output:
        return output
    ret_15m = _series(output, "target_future_return_15m")
    threshold = _edge_threshold(output, fee_rate=fee_rate, min_edge=min_edge, hold_edge_multiplier=hold_edge_multiplier)
    action = (ret_15m > threshold).astype(float) - (ret_15m < -threshold).astype(float)
    edge_score = ret_15m.copy()
    edge_score = edge_score.where(action == 0.0, ret_15m - (action * threshold))
    edge_score = edge_score.where(action != 0.0, 0.0)
    output["target_trade_action"] = action
    output["target_edge_threshold"] = threshold
    output["target_edge_aware_trade_score"] = edge_score
    return output


def _source_weights(frame, *, historical_weight: float, live_weight: float, recency_strength: float):
    import numpy as np
    import pandas as pd

    if len(frame) == 0:
        return None
    schema = frame["feature_schema_version"].astype(str).str.lower() if "feature_schema_version" in frame else pd.Series("", index=frame.index)
    final_input = frame["final_ai_input"].astype(str).str.lower() if "final_ai_input" in frame else pd.Series("", index=frame.index)
    is_historical = schema.str.contains("coingecko", na=False) | final_input.str.contains("coingecko_history", na=False)
    weights = np.where(is_historical.to_numpy(dtype=bool), historical_weight, live_weight).astype(float)
    if recency_strength > 0 and "as_of" in frame:
        order = pd.to_datetime(frame["as_of"], utc=True, errors="coerce").rank(method="first", pct=True).fillna(0.5).to_numpy(dtype=float)
        weights *= 1.0 + (recency_strength * order)
    return weights


def _regime_masks(frame) -> dict[str, Any]:
    return {
        "news_shock": _series(frame, "regime_news_shock_score") >= 0.45,
        "risk_off": _series(frame, "regime_risk_off_score") >= 0.45,
        "liquidity_stress": _series(frame, "regime_liquidity_stress_score") >= 0.35,
        "crowded_market": _series(frame, "regime_crowd_pressure") >= 0.35,
        "breakout_pressure": _series(frame, "regime_breakout_pressure") >= 0.35,
        "mean_reversion_pressure": _series(frame, "regime_mean_reversion_pressure") >= 0.35,
        "trend_up": (_series(frame, "regime_trend_strength") >= 0.35) & (_series(frame, "regime_direction_score") > 0.10),
        "trend_down": (_series(frame, "regime_trend_strength") >= 0.35) & (_series(frame, "regime_direction_score") < -0.10),
        "range_low_volatility": (_series(frame, "regime_trend_strength") < 0.25) & (_series(frame, "regime_volatility_score") < 0.20),
        "high_volatility": _series(frame, "regime_volatility_score") >= 0.35,
    }


def _row_regime_names(frame) -> list[str | None]:
    masks = _regime_masks(frame)
    output: list[str | None] = []
    for idx in frame.index:
        selected = None
        for regime in REGIME_ORDER:
            mask = masks.get(regime)
            if mask is not None and bool(mask.loc[idx]):
                selected = regime
                break
        output.append(selected)
    return output


def _regime_validation(frame, predictions, realized_returns, *, fee_rate: float, min_edge: float, min_rows: int = 25) -> dict[str, Any]:
    import numpy as np

    predictions = np.asarray(predictions, dtype=float)
    realized_returns = np.asarray(realized_returns, dtype=float)
    metrics: dict[str, Any] = {}
    profitable = 0
    losing = 0
    worst_net = None
    for regime, mask in _regime_masks(frame).items():
        mask_values = mask.to_numpy(dtype=bool)
        row_count = int(mask_values.sum())
        if row_count < min_rows:
            metrics[regime] = {"status": "too_few_rows", "rows": row_count}
            continue
        regime_metrics = _simulate(predictions[mask_values], realized_returns[mask_values], fee_rate=fee_rate, min_edge=min_edge)
        regime_metrics["directional_accuracy"] = _directional_accuracy(predictions[mask_values], realized_returns[mask_values])
        regime_metrics["rows"] = row_count
        regime_metrics["status"] = "evaluated"
        metrics[regime] = regime_metrics
        net_return = float(regime_metrics["net_return_after_fees"])
        if net_return > 0:
            profitable += 1
        else:
            losing += 1
        worst_net = net_return if worst_net is None else min(worst_net, net_return)
    evaluated = profitable + losing
    return {
        "metrics": metrics,
        "summary": {
            "evaluated_regime_count": evaluated,
            "profitable_regime_count": profitable,
            "losing_regime_count": losing,
            "worst_regime_net_return": float(worst_net) if worst_net is not None else None,
            "min_rows_per_regime": min_rows,
        },
    }


def _add_plan_targets(frame, *, max_margin_pct: float = 0.10, max_leverage: float = 125.0):
    import numpy as np

    output = frame.copy()
    if "target_future_return_15m" not in output or "target_trade_quality_score" not in output:
        return output
    ret_15m = output["target_future_return_15m"].astype(float)
    quality = output["target_trade_quality_score"].astype(float)
    upside = output["target_max_upside_1h"].astype(float) if "target_max_upside_1h" in output else ret_15m.clip(lower=0.0)
    drawdown = output["target_max_drawdown_1h"].astype(float) if "target_max_drawdown_1h" in output else ret_15m.clip(upper=0.0)
    take_first = output["target_take_profit_hit_first"].astype(float) if "target_take_profit_hit_first" in output else 0.0
    trade_action = output["target_trade_action"].astype(float).abs() if "target_trade_action" in output else 1.0
    strength = ((quality + 0.004) / 0.035).clip(lower=0.0, upper=1.0)
    edge_strength = (ret_15m.abs() / 0.025).clip(lower=0.0, upper=1.0)
    plan_strength = np.maximum(strength, edge_strength * 0.65) * trade_action
    output["target_best_margin_pct"] = (max_margin_pct * plan_strength).clip(lower=0.0, upper=max_margin_pct)
    output["target_best_leverage"] = (1.0 + (max_leverage - 1.0) * plan_strength).clip(lower=1.0, upper=max_leverage)
    stop_pct = np.maximum(drawdown.abs() * 1.15, ret_15m.abs() * 0.65).clip(lower=0.003, upper=0.08)
    take_pct = np.maximum(upside.abs() * 0.90, ret_15m.abs() * 1.8).clip(lower=0.006, upper=0.20)
    output["target_best_stop_loss_pct"] = stop_pct * np.maximum(trade_action, 0.0)
    output["target_best_take_profit_pct"] = take_pct * np.maximum(trade_action, 0.0)
    hold_scale = (1.0 - edge_strength).clip(lower=0.0, upper=1.0)
    output["target_best_hold_seconds"] = (900 + (14400 - 900) * hold_scale - (take_first * 1800)).clip(lower=300, upper=14400)
    return output


def _train_plan_models(model_type: str, train_frame, feature_columns: list[str], out_dir: Path, version: str):
    import joblib

    model_files: dict[str, str] = {}
    metrics: dict[str, Any] = {}
    for target in PLAN_TARGETS:
        if target not in train_frame:
            continue
        target_frame = train_frame.dropna(subset=[target])
        if len(target_frame) < 100:
            metrics[target] = {"status": "skipped", "rows": int(len(target_frame))}
            continue
        model = _make_model(model_type)
        x_target = target_frame[feature_columns].fillna(0.0).astype(float)
        y_target = target_frame[target].astype(float)
        model.fit(x_target, y_target)
        model_path = out_dir / f"model_{version}_{target}.joblib"
        joblib.dump(model, model_path)
        model_files[target] = model_path.name
        predicted = model.predict(x_target)
        mae = float(abs(predicted - y_target.to_numpy()).mean()) if len(target_frame) else 0.0
        metrics[target] = {"status": "trained", "rows": int(len(target_frame)), "mae_in_sample": mae}
    return model_files, metrics


def _train_specialist_models(
    *,
    model_type: str,
    training,
    test,
    feature_columns: list[str],
    target: str,
    realized_returns,
    fee_rate: float,
    min_edge: float,
    min_rows: int,
    source_weight_args: dict[str, float],
):
    import numpy as np

    predictions = np.full(len(test), np.nan, dtype=float)
    test_regimes = _row_regime_names(test)
    files_ready: dict[str, Any] = {}
    report: dict[str, Any] = {}
    for regime in REGIME_ORDER:
        train_mask = _regime_masks(training).get(regime)
        if train_mask is None:
            report[regime] = {"status": "skipped_no_mask", "train_rows": 0, "test_rows": int(test_regimes.count(regime))}
            continue
        train_rows = training[train_mask].copy()
        test_indices = [index for index, name in enumerate(test_regimes) if name == regime]
        if len(train_rows) < min_rows or len(test_indices) < 25:
            report[regime] = {"status": "skipped_too_few_rows", "train_rows": int(len(train_rows)), "test_rows": int(len(test_indices))}
            continue
        model = _make_model(model_type)
        x_train = train_rows[feature_columns].fillna(0.0).astype(float)
        y_train = train_rows[target].astype(float)
        weights = _source_weights(train_rows, **source_weight_args)
        _fit_model(model, x_train, y_train, sample_weight=weights)
        x_test = test.iloc[test_indices][feature_columns].fillna(0.0).astype(float)
        regime_predictions = model.predict(x_test)
        predictions[test_indices] = regime_predictions
        metrics = _simulate(regime_predictions, realized_returns[test_indices], fee_rate=fee_rate, min_edge=min_edge)
        metrics["directional_accuracy"] = _directional_accuracy(regime_predictions, realized_returns[test_indices])
        metrics["train_rows"] = int(len(train_rows))
        metrics["test_rows"] = int(len(test_indices))
        metrics["status"] = "trained"
        report[regime] = metrics
        files_ready[regime] = model
    return predictions, files_ready, report


def _write_specialist_models(models: dict[str, Any], out_dir: Path, version: str):
    import joblib

    files: dict[str, str] = {}
    paths: list[Path] = []
    for regime, model in models.items():
        path = out_dir / f"model_{version}_specialist_{regime}.joblib"
        joblib.dump(model, path)
        files[regime] = path.name
        paths.append(path)
    return files, paths


def _package_model(model_path: Path, metadata: dict[str, Any], output_dir: Path, extra_paths: list[Path] | None = None) -> Path:
    package_path = output_dir / f"model_package_{metadata['version']}.zip"
    metadata["model_file"] = model_path.name
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(model_path, model_path.name)
        for extra_path in extra_paths or []:
            archive.write(extra_path, extra_path.name)
        archive.writestr("metadata.json", json.dumps(metadata, indent=2, default=str))
    return package_path


def _gate_check(value: Any, passed: bool, *, requirement: str, hard: bool = True) -> dict[str, Any]:
    return {"value": value, "passed": bool(passed), "requirement": requirement, "hard": bool(hard)}


def _demo_safety_gate(metrics: dict[str, Any], min_trades: int) -> dict[str, Any]:
    regime_summary = metrics.get("regime_summary", {}) or {}
    slice_summary = metrics.get("slice_summary", {}) or {}
    stress_test = metrics.get("stress_test", {}) or {}
    evaluated_regimes = int(regime_summary.get("evaluated_regime_count", 0) or 0)
    profitable_regimes = int(regime_summary.get("profitable_regime_count", 0) or 0)
    evaluated_slices = int(slice_summary.get("evaluated_slice_count", 0) or 0)
    profitable_slices = int(slice_summary.get("profitable_slice_count", 0) or 0)
    trade_rate = float(metrics.get("trade_rate", 0.0) or 0.0)
    trade_accuracy = float(metrics.get("trade_directional_accuracy", 0.0) or 0.0)
    stress_net = float(stress_test.get("net_return_after_fees", 0.0) or 0.0)
    stress_drawdown = float(stress_test.get("max_drawdown", 0.0) or 0.0)
    max_demo_drawdown = float(DEMO_CONFIG["max_demo_drawdown"])
    stress_drawdown_limit = min(0.65, max_demo_drawdown * 1.5)

    checks = {
        "net_return_after_fees_positive": _gate_check(
            metrics.get("net_return_after_fees"),
            float(metrics.get("net_return_after_fees", 0.0) or 0.0) > 0.0,
            requirement="> 0 after normal fees",
        ),
        "stress_test_not_destroyed": _gate_check(
            stress_net,
            stress_net > -0.05,
            requirement="> -5% after stress fee/slippage test",
        ),
        "max_drawdown_demo_limit": _gate_check(
            metrics.get("max_drawdown"),
            float(metrics.get("max_drawdown", 1.0) or 1.0) <= max_demo_drawdown,
            requirement=f"<= {max_demo_drawdown:.2%} demo max drawdown",
        ),
        "stress_drawdown_demo_limit": _gate_check(
            stress_drawdown,
            stress_drawdown <= stress_drawdown_limit,
            requirement=f"<= {stress_drawdown_limit:.2%} stress max drawdown",
        ),
        "minimum_trades": _gate_check(
            metrics.get("number_of_trades"),
            int(metrics.get("number_of_trades", 0) or 0) >= min_trades,
            requirement=f">= {min_trades} simulated trades",
        ),
        "not_overtrading": _gate_check(
            trade_rate,
            trade_rate <= float(DEMO_CONFIG["max_trade_rate"]),
            requirement=f"trade rate <= {float(DEMO_CONFIG['max_trade_rate']):.0%} of test rows",
        ),
        "trade_directional_accuracy": _gate_check(
            trade_accuracy,
            trade_accuracy >= float(DEMO_CONFIG["min_trade_directional_accuracy"]),
            requirement=f">= {float(DEMO_CONFIG['min_trade_directional_accuracy']):.2%} on executed trades",
        ),
        "beats_rule_based_baseline": _gate_check(
            metrics.get("beats_rule_based_baseline"),
            bool(metrics.get("beats_rule_based_baseline", False)),
            requirement="AI net return after fees > rule-based baseline net return",
        ),
        "accuracy_not_suspiciously_high": _gate_check(
            metrics.get("directional_accuracy"),
            float(metrics.get("directional_accuracy", 0.0) or 0.0) <= 0.75,
            requirement="<= 75% total directional accuracy to reduce leakage risk",
        ),
        "trade_accuracy_not_suspiciously_high": _gate_check(
            trade_accuracy,
            trade_accuracy <= 0.85,
            requirement="<= 85% trade directional accuracy to reduce leakage risk",
        ),
        "time_slices_have_a_winner": _gate_check(
            profitable_slices,
            evaluated_slices < 2 or profitable_slices >= int(DEMO_CONFIG["min_profitable_slices"]),
            requirement=f">= {int(DEMO_CONFIG['min_profitable_slices'])} profitable chronological test slice when at least 2 slices are evaluated",
        ),
        "regimes_not_all_losing": _gate_check(
            {"profitable": profitable_regimes, "evaluated": evaluated_regimes},
            evaluated_regimes < 3 or profitable_regimes > 0,
            requirement="not 0 profitable regimes when at least 3 regimes are evaluated",
        ),
        "regime_coverage": _gate_check(
            evaluated_regimes,
            evaluated_regimes >= 3,
            requirement=">= 3 evaluated regimes preferred",
            hard=False,
        ),
    }
    hard_failures = [name for name, check in checks.items() if check["hard"] and not check["passed"]]
    soft_warnings = [name for name, check in checks.items() if not check["hard"] and not check["passed"]]
    return {
        "mode": "paper_demo_activation",
        "passed": not hard_failures,
        "hard_failures": hard_failures,
        "soft_warnings": soft_warnings,
        "checks": checks,
        "settings": dict(DEMO_CONFIG),
        "note": "PASS only means usable for the demo/paper website. It is not real-money approval.",
    }


def _candidate_score(metrics: dict[str, Any], passed: bool) -> dict[str, Any]:
    if passed:
        return {"score": 100.0, "grade": "DEMO_PASS", "components": {"passed_demo_safety_checks": 100.0}}

    trade_directional = float(metrics.get("trade_directional_accuracy", metrics.get("directional_accuracy", 0.0)) or 0.0)
    net_return = float(metrics.get("net_return_after_fees", -1.0) or 0.0)
    drawdown = float(metrics.get("max_drawdown", 1.0) or 0.0)
    trade_rate = float(metrics.get("trade_rate", 1.0) or 0.0)
    profit_factor = float(metrics.get("profit_factor", 0.0) or 0.0)
    beats_baseline = bool(metrics.get("beats_rule_based_baseline", False))
    stress_net = float((metrics.get("stress_test", {}) or {}).get("net_return_after_fees", -1.0) or 0.0)
    regime_summary = metrics.get("regime_summary", {}) or {}
    slice_summary = metrics.get("slice_summary", {}) or {}
    evaluated_regimes = max(1, int(regime_summary.get("evaluated_regime_count", 0) or 0))
    profitable_regimes = int(regime_summary.get("profitable_regime_count", 0) or 0)
    evaluated_slices = max(1, int(slice_summary.get("evaluated_slice_count", 0) or 0))
    profitable_slices = int(slice_summary.get("profitable_slice_count", 0) or 0)

    direction_score = 12.0 * _clip((trade_directional - 0.49) / 0.03)
    return_score = 24.0 * _clip((net_return + 0.05) / 0.08)
    stress_score = 12.0 * _clip((stress_net + 0.08) / 0.08)
    drawdown_score = 16.0 * _clip((0.75 - drawdown) / 0.75)
    activity_score = 8.0 * _clip(1.0 - max(0.0, trade_rate - float(DEMO_CONFIG["max_trade_rate"])) / max(0.01, 1.0 - float(DEMO_CONFIG["max_trade_rate"])))
    baseline_score = 10.0 if beats_baseline else 0.0
    regime_score = 8.0 * _clip(profitable_regimes / evaluated_regimes)
    slice_score = 10.0 * _clip(profitable_slices / evaluated_slices)
    profit_factor_score = 10.0 * _clip((profit_factor - 0.8) / 0.7)
    score = min(99.0, max(0.0, direction_score + return_score + stress_score + drawdown_score + activity_score + baseline_score + regime_score + slice_score + profit_factor_score))

    if score >= 80:
        grade = "close"
    elif score >= 60:
        grade = "improving"
    elif score >= 35:
        grade = "early"
    else:
        grade = "far"
    return {
        "score": round(score, 2),
        "grade": grade,
        "components": {
            "trade_directional_accuracy_points": round(direction_score, 2),
            "net_return_points": round(return_score, 2),
            "stress_cost_points": round(stress_score, 2),
            "drawdown_points": round(drawdown_score, 2),
            "trade_activity_points": round(activity_score, 2),
            "baseline_points": round(baseline_score, 2),
            "regime_points": round(regime_score, 2),
            "time_slice_points": round(slice_score, 2),
            "profit_factor_points": round(profit_factor_score, 2),
        },
        "score_rules": {
            "100": "model passed demo/paper safety checks and was packaged",
            "80_99": "close, but at least one demo activation condition still failed",
            "60_79": "improving, not ready for demo activation",
            "35_59": "early, needs better behavior",
            "0_34": "far from passing",
        },
    }


def _score_candidates(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        return {"score": 0.0, "grade": "no_model", "best_model_type": None}
    for candidate in candidates:
        candidate["safety_score"] = _candidate_score(candidate.get("metrics", {}), bool(candidate.get("passed")))
    best = max(candidates, key=lambda item: item.get("safety_score", {}).get("score", 0.0))
    return {
        "score": best["safety_score"]["score"],
        "grade": best["safety_score"]["grade"],
        "best_model_type": best.get("model_type"),
        "components": best["safety_score"].get("components", {}),
        "score_rules": best["safety_score"].get("score_rules", {}),
    }


def _candidate_passed(metrics: dict[str, Any], min_trades: int) -> bool:
    gate = _demo_safety_gate(metrics, min_trades)
    metrics["demo_safety_gate"] = gate
    return bool(gate["passed"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multiple local models and package the best demo/paper-safe candidate.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--target", default="target_edge_aware_trade_score")
    parser.add_argument("--return-column", default="target_future_return_15m")
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--fee-rate", type=float, default=0.0004)
    parser.add_argument("--min-edge", type=float, default=0.001)
    parser.add_argument("--model-types", default="sklearn_hist_gradient_boosting,random_forest,lightgbm,xgboost", help="Comma-separated model types.")
    parser.add_argument("--min-trades", type=int, default=10)
    parser.add_argument("--max-plan-margin-pct", type=float, default=0.10)
    parser.add_argument("--max-plan-leverage", type=float, default=125.0)
    parser.add_argument("--symbol-aware", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--edge-aware-target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hold-edge-multiplier", type=float, default=2.5)
    parser.add_argument("--historical-source-weight", type=float, default=1.0)
    parser.add_argument("--live-source-weight", type=float, default=3.0)
    parser.add_argument("--recency-weight-strength", type=float, default=0.5)
    parser.add_argument("--regime-specialists", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--specialist-min-rows", type=int, default=750)
    parser.add_argument("--demo-slippage-rate", type=float, default=0.0002)
    parser.add_argument("--demo-stress-cost-multiplier", type=float, default=2.0)
    parser.add_argument("--max-demo-drawdown", type=float, default=0.35)
    parser.add_argument("--max-demo-trade-rate", type=float, default=0.70)
    parser.add_argument("--min-demo-trade-accuracy", type=float, default=0.50)
    parser.add_argument("--min-demo-profitable-slices", type=int, default=1)
    args = parser.parse_args()

    DEMO_CONFIG.update(
        {
            "slippage_rate": args.demo_slippage_rate,
            "stress_cost_multiplier": args.demo_stress_cost_multiplier,
            "max_demo_drawdown": args.max_demo_drawdown,
            "max_trade_rate": args.max_demo_trade_rate,
            "min_trade_directional_accuracy": args.min_demo_trade_accuracy,
            "min_profitable_slices": args.min_demo_profitable_slices,
        }
    )

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

    symbol_feature_columns: list[str] = []
    if args.symbol_aware:
        frame, symbol_feature_columns = _add_symbol_features(frame)
    frame = _add_regime_features(frame)
    if args.edge_aware_target:
        frame = _add_edge_aware_targets(frame, fee_rate=args.fee_rate, min_edge=args.min_edge, hold_edge_multiplier=args.hold_edge_multiplier)
    frame = _add_plan_targets(frame, max_margin_pct=args.max_plan_margin_pct, max_leverage=args.max_plan_leverage)
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
    source_weight_args = {
        "historical_weight": args.historical_source_weight,
        "live_weight": args.live_source_weight,
        "recency_strength": args.recency_weight_strength,
    }
    train_weights = _source_weights(training, **source_weight_args)
    x_test = test[feature_columns].fillna(0.0).astype(float)
    realized_returns = test[return_column].fillna(0.0).astype(float).to_numpy()

    baseline_predictions = _rule_based_predictions(test)
    baseline_metrics = _simulate(baseline_predictions, realized_returns, fee_rate=args.fee_rate, min_edge=args.min_edge)
    baseline_metrics["directional_accuracy"] = _directional_accuracy(baseline_predictions, realized_returns)
    baseline_regime_validation = _regime_validation(test, baseline_predictions, realized_returns, fee_rate=args.fee_rate, min_edge=args.min_edge)
    baseline_metrics["regime_summary"] = baseline_regime_validation["summary"]
    baseline_metrics["regime_metrics"] = baseline_regime_validation["metrics"]

    candidates: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}
    trained_specialists: dict[str, dict[str, Any]] = {}

    for model_type in [item.strip() for item in args.model_types.split(",") if item.strip()]:
        try:
            model = _make_model(model_type)
            _fit_model(model, x_train, y_train, sample_weight=train_weights)
            global_predictions = model.predict(x_test)
        except Exception as exc:
            skipped[model_type] = str(exc)
            continue

        predictions = global_predictions
        specialist_report: dict[str, Any] = {}
        specialist_summary = {"enabled": bool(args.regime_specialists), "trained_count": 0, "used_test_rows": 0}
        if args.regime_specialists:
            specialist_predictions, specialists, specialist_report = _train_specialist_models(
                model_type=model_type,
                training=training,
                test=test,
                feature_columns=feature_columns,
                target=args.target,
                realized_returns=realized_returns,
                fee_rate=args.fee_rate,
                min_edge=args.min_edge,
                min_rows=args.specialist_min_rows,
                source_weight_args=source_weight_args,
            )
            used_mask = ~np.isnan(specialist_predictions)
            if used_mask.any():
                predictions = global_predictions.copy()
                predictions[used_mask] = specialist_predictions[used_mask]
                trained_specialists[model_type] = specialists
                specialist_summary = {
                    "enabled": True,
                    "trained_count": int(len(specialists)),
                    "used_test_rows": int(used_mask.sum()),
                    "used_test_rows_pct": float(used_mask.mean() * 100.0),
                }

        metrics = _simulate(predictions, realized_returns, fee_rate=args.fee_rate, min_edge=args.min_edge)
        metrics["directional_accuracy"] = _directional_accuracy(predictions, realized_returns)
        metrics["average_predicted_return"] = float(np.mean(predictions)) if len(predictions) else 0.0
        metrics["test_rows"] = int(len(test))
        metrics["model_type"] = model_type
        metrics["beats_rule_based_baseline"] = metrics["net_return_after_fees"] > baseline_metrics["net_return_after_fees"]
        metrics["global_only"] = _simulate(global_predictions, realized_returns, fee_rate=args.fee_rate, min_edge=args.min_edge)
        metrics["global_only"]["directional_accuracy"] = _directional_accuracy(global_predictions, realized_returns)
        metrics["specialist_summary"] = specialist_summary
        metrics["specialist_metrics"] = specialist_report
        regime_validation = _regime_validation(test, predictions, realized_returns, fee_rate=args.fee_rate, min_edge=args.min_edge)
        metrics["regime_summary"] = regime_validation["summary"]
        metrics["regime_metrics"] = regime_validation["metrics"]

        warnings = []
        if metrics["directional_accuracy"] > 0.75:
            warnings.append("Unrealistically high total directional accuracy; possible leakage/overfitting.")
        if metrics.get("trade_directional_accuracy", 0.0) > 0.85:
            warnings.append("Unrealistically high trade directional accuracy; possible leakage/overfitting.")
        if metrics["number_of_trades"] < args.min_trades:
            warnings.append("Too few simulated trades.")
        if metrics["net_return_after_fees"] <= 0:
            warnings.append("Net return after fees is not positive.")
        if metrics["max_drawdown"] > args.max_demo_drawdown:
            warnings.append("Max drawdown is above demo activation limit.")
        if metrics.get("trade_rate", 0.0) > args.max_demo_trade_rate:
            warnings.append("Model trades too often for demo activation.")
        if not metrics["beats_rule_based_baseline"]:
            warnings.append("Model did not beat rule-based baseline.")
        if metrics["regime_summary"]["evaluated_regime_count"] < 3:
            warnings.append("Too few populated regimes for strong regime validation.")
        if metrics["regime_summary"]["losing_regime_count"] > metrics["regime_summary"]["profitable_regime_count"]:
            warnings.append("Model loses in more evaluated regimes than it wins.")

        passed = _candidate_passed(metrics, args.min_trades)
        if metrics.get("demo_safety_gate", {}).get("hard_failures"):
            warnings.append("Demo gate hard failures: " + ", ".join(metrics["demo_safety_gate"]["hard_failures"]))
        candidates.append({"model_type": model_type, "metrics": metrics, "warnings": warnings, "passed": passed})

    final_score = _score_candidates(candidates)
    passed_candidates = [candidate for candidate in candidates if candidate["passed"]]
    best = max(
        passed_candidates,
        key=lambda item: (
            item["metrics"]["net_return_after_fees"],
            item["metrics"]["regime_summary"]["profitable_regime_count"],
            -item["metrics"]["max_drawdown"],
            item["metrics"].get("trade_directional_accuracy", 0.0),
        ),
        default=None,
    )

    result: dict[str, Any] = {
        "status": "trained_failed_demo_safety_checks" if candidates else "no_model_selected",
        "dataset": str(args.dataset),
        "target": args.target,
        "return_column": return_column,
        "train_rows": int(len(training)),
        "test_rows": int(len(test)),
        "total_labeled_rows": int(len(train_frame)),
        "dataset_days": dataset_days,
        "feature_columns": len(feature_columns),
        "symbol_aware": bool(args.symbol_aware),
        "edge_aware_target": bool(args.edge_aware_target),
        "regime_specialists": bool(args.regime_specialists),
        "specialist_min_rows": args.specialist_min_rows,
        "hold_edge_multiplier": args.hold_edge_multiplier,
        "demo_safety_mode": "paper_demo_activation",
        "demo_safety_settings": dict(DEMO_CONFIG),
        "source_weights": {
            "historical_source_weight": args.historical_source_weight,
            "live_source_weight": args.live_source_weight,
            "recency_weight_strength": args.recency_weight_strength,
        },
        "symbol_feature_columns": [column for column in symbol_feature_columns if column in feature_columns],
        "regime_feature_columns": [column for column in REGIME_FEATURE_COLUMNS if column in feature_columns],
        "regime_order": REGIME_ORDER,
        "plan_targets": [target for target in PLAN_TARGETS if target in train_frame],
        "data_readiness": _data_readiness(int(len(train_frame)), dataset_days),
        "baseline": baseline_metrics,
        "candidates": candidates,
        "skipped_models": skipped,
        "failed_safety_checks": True,
        "next_steps": _failure_advice(candidates, dataset_days),
        "final_score": final_score,
        "message": f"Training completed, but no model passed demo/paper safety checks. Nothing was packaged or uploaded. Final score: {final_score['score']}/100 ({final_score['grade']}).",
    }
    if best is None:
        print(json.dumps(result, indent=2, default=str))
        return

    final_model = _make_model(best["model_type"])
    _fit_model(
        final_model,
        train_frame[feature_columns].fillna(0.0).astype(float),
        train_frame[args.target].astype(float),
        sample_weight=_source_weights(train_frame, **source_weight_args),
    )
    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / f"model_{version}.joblib"
    joblib.dump(final_model, model_path)

    plan_model_files, plan_model_metrics = _train_plan_models(best["model_type"], train_frame, feature_columns, args.out_dir, version)
    plan_model_paths = [args.out_dir / filename for filename in plan_model_files.values()]
    specialist_model_files: dict[str, str] = {}
    specialist_model_paths: list[Path] = []
    if args.regime_specialists:
        specialist_frame = train_frame.copy()
        _, specialist_models, _ = _train_specialist_models(
            model_type=best["model_type"],
            training=specialist_frame,
            test=test,
            feature_columns=feature_columns,
            target=args.target,
            realized_returns=realized_returns,
            fee_rate=args.fee_rate,
            min_edge=args.min_edge,
            min_rows=args.specialist_min_rows,
            source_weight_args=source_weight_args,
        )
        specialist_model_files, specialist_model_paths = _write_specialist_models(specialist_models, args.out_dir, version)

    metadata = {
        "model_id": f"{best['model_type']}:{version}",
        "name": best["model_type"],
        "version": version,
        "model_type": best["model_type"],
        "status": "candidate",
        "activation_mode": "manual",
        "paper_demo_only": True,
        "feature_schema_version": str(train_frame["feature_schema_version"].dropna().iloc[-1]) if "feature_schema_version" in train_frame and len(train_frame["feature_schema_version"].dropna()) else "local-raw-v1",
        "feature_columns": feature_columns,
        "symbol_aware": bool(args.symbol_aware),
        "edge_aware_target": bool(args.edge_aware_target),
        "regime_specialists": bool(args.regime_specialists),
        "specialist_model_files": specialist_model_files,
        "specialist_regime_order": REGIME_ORDER,
        "specialist_min_rows": args.specialist_min_rows,
        "hold_edge_multiplier": args.hold_edge_multiplier,
        "demo_safety_mode": "paper_demo_activation",
        "demo_safety_settings": dict(DEMO_CONFIG),
        "source_weights": {
            "historical_source_weight": args.historical_source_weight,
            "live_source_weight": args.live_source_weight,
            "recency_weight_strength": args.recency_weight_strength,
        },
        "symbol_feature_columns": [column for column in symbol_feature_columns if column in feature_columns],
        "regime_feature_columns": [column for column in REGIME_FEATURE_COLUMNS if column in feature_columns],
        "target": args.target,
        "return_column": return_column,
        "plan_model_files": plan_model_files,
        "plan_targets": PLAN_TARGETS,
        "training_dataset_path": str(args.dataset),
        "training_dataset_hash": _dataset_hash(args.dataset),
        "metrics": best["metrics"]
        | {
            "paper_test_readiness": "DEMO_PASS",
            "final_score": {"score": 100.0, "grade": "DEMO_PASS"},
            "rule_based_baseline": baseline_metrics,
            "train_rows": int(len(training)),
            "total_labeled_rows": int(len(train_frame)),
            "dataset_days": dataset_days,
            "plan_model_metrics": plan_model_metrics,
        },
        "feature_importance": _feature_importance(final_model, feature_columns),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = args.out_dir / f"model_{version}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    package_path = _package_model(model_path, metadata, args.out_dir, extra_paths=plan_model_paths + specialist_model_paths)

    result.update(
        {
            "status": "passed",
            "message": "Best model passed demo/paper safety checks and was packaged. Upload/activate explicitly for the demo website only. Final score: 100/100 (DEMO_PASS).",
            "failed_safety_checks": False,
            "next_steps": ["Upload the model package, then activate it from the dashboard Training tab for demo/paper trading."],
            "best_model": best,
            "model": str(model_path),
            "plan_models": plan_model_files,
            "specialist_models": specialist_model_files,
            "metadata": str(metadata_path),
            "package": str(package_path),
            "model_id": metadata["model_id"],
            "final_score": {"score": 100.0, "grade": "DEMO_PASS", "best_model_type": best.get("model_type")},
        }
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
