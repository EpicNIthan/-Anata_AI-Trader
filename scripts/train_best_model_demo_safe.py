from __future__ import annotations

"""
Demo/paper safety wrapper for scripts/train_best_model.py.

This keeps the original trainer intact, but monkey-patches the safety simulation and
candidate gate before delegating to train_best_model.main(). A pass means the model is
usable for the demo website / paper runner only. It is not real-money approval.
"""

import argparse
import sys
from typing import Any

import train_best_model as base


CONFIG = {
    "slippage_rate": 0.0002,
    "stress_cost_multiplier": 2.0,
    "max_demo_drawdown": 0.35,
    "max_trade_rate": 0.70,
    "min_trade_directional_accuracy": 0.50,
    "min_profitable_slices": 1,
}


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _trade_cost(*, fee_rate: float, slippage_rate: float = 0.0, cost_multiplier: float = 1.0) -> float:
    return float(max(0.0, ((fee_rate + slippage_rate) * 2.0) * max(0.0, cost_multiplier)))


def _actions_from_predictions(predictions, *, fee_rate: float, min_edge: float):
    import numpy as np

    predictions = np.asarray(predictions, dtype=float)
    threshold = float(max(min_edge, fee_rate * 2.0))
    return np.where(predictions > threshold, 1, np.where(predictions < -threshold, -1, 0)), threshold


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
        "max_drawdown": float(base._max_drawdown(equity)),
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
        slippage_rate=CONFIG["slippage_rate"],
        cost_multiplier=CONFIG["stress_cost_multiplier"],
    )
    normal_slices = _slice_validation(predictions, realized_returns, fee_rate=fee_rate, min_edge=min_edge)
    stress_slices = _slice_validation(
        predictions,
        realized_returns,
        fee_rate=fee_rate,
        min_edge=min_edge,
        slippage_rate=CONFIG["slippage_rate"],
        cost_multiplier=CONFIG["stress_cost_multiplier"],
    )
    metrics["slice_summary"] = normal_slices["summary"]
    metrics["slice_metrics"] = normal_slices["metrics"]
    metrics["stress_slice_summary"] = stress_slices["summary"]
    metrics["stress_slice_metrics"] = stress_slices["metrics"]
    return metrics


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
            float(metrics.get("max_drawdown", 1.0) or 1.0) <= CONFIG["max_demo_drawdown"],
            requirement=f"<= {CONFIG['max_demo_drawdown']:.2%} demo max drawdown",
        ),
        "stress_drawdown_demo_limit": _gate_check(
            stress_drawdown,
            stress_drawdown <= min(0.65, CONFIG["max_demo_drawdown"] * 1.5),
            requirement=f"<= {min(0.65, CONFIG['max_demo_drawdown'] * 1.5):.2%} stress max drawdown",
        ),
        "minimum_trades": _gate_check(
            metrics.get("number_of_trades"),
            int(metrics.get("number_of_trades", 0) or 0) >= min_trades,
            requirement=f">= {min_trades} simulated trades",
        ),
        "not_overtrading": _gate_check(
            trade_rate,
            trade_rate <= CONFIG["max_trade_rate"],
            requirement=f"trade rate <= {CONFIG['max_trade_rate']:.0%} of test rows",
        ),
        "trade_directional_accuracy": _gate_check(
            trade_accuracy,
            trade_accuracy >= CONFIG["min_trade_directional_accuracy"],
            requirement=f">= {CONFIG['min_trade_directional_accuracy']:.2%} on executed trades",
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
            evaluated_slices < 2 or profitable_slices >= CONFIG["min_profitable_slices"],
            requirement=f">= {CONFIG['min_profitable_slices']} profitable chronological test slice when at least 2 slices are evaluated",
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
        "settings": dict(CONFIG),
        "note": "PASS only means usable for the demo/paper website. It is not real-money approval.",
    }


def _candidate_passed(metrics: dict[str, Any], min_trades: int) -> bool:
    gate = _demo_safety_gate(metrics, min_trades)
    metrics["demo_safety_gate"] = gate
    return bool(gate["passed"])


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
    activity_score = 8.0 * _clip(1.0 - max(0.0, trade_rate - CONFIG["max_trade_rate"]) / max(0.01, 1.0 - CONFIG["max_trade_rate"]))
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
    return advice


def _parse_wrapper_args() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--demo-slippage-rate", type=float, default=CONFIG["slippage_rate"])
    parser.add_argument("--demo-stress-cost-multiplier", type=float, default=CONFIG["stress_cost_multiplier"])
    parser.add_argument("--max-demo-drawdown", type=float, default=CONFIG["max_demo_drawdown"])
    parser.add_argument("--max-demo-trade-rate", type=float, default=CONFIG["max_trade_rate"])
    parser.add_argument("--min-demo-trade-accuracy", type=float, default=CONFIG["min_trade_directional_accuracy"])
    parser.add_argument("--min-demo-profitable-slices", type=int, default=CONFIG["min_profitable_slices"])
    known, remaining = parser.parse_known_args()
    CONFIG.update(
        {
            "slippage_rate": known.demo_slippage_rate,
            "stress_cost_multiplier": known.demo_stress_cost_multiplier,
            "max_demo_drawdown": known.max_demo_drawdown,
            "max_trade_rate": known.max_demo_trade_rate,
            "min_trade_directional_accuracy": known.min_demo_trade_accuracy,
            "min_profitable_slices": known.min_demo_profitable_slices,
        }
    )
    sys.argv = [sys.argv[0], *remaining]


def main() -> None:
    _parse_wrapper_args()
    base._simulate = _simulate
    base._candidate_passed = _candidate_passed
    base._candidate_score = _candidate_score
    base._failure_advice = _failure_advice
    base.main()


if __name__ == "__main__":
    main()
