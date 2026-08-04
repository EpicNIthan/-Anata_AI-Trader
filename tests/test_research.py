"""Focused regression tests for leakage-safe local research utilities."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from app.research import (
    EvaluationObservation,
    ExecutionAssumptions,
    PurgedTimeSeriesSplit,
    SignalSeries,
    WalkForwardEvaluator,
    analyze_ensemble_saturation,
    analyze_signal_independence,
    apply_execution_assumptions,
    assess_incremental_contribution,
    chronological_split,
    evaluate_observations,
    persisted_research_id,
)


def _rows(*, count: int = 10, label_delay_minutes: int = 0) -> list[dict[str, object]]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    result: list[dict[str, object]] = []
    for index in range(count):
        timestamp = start + timedelta(minutes=index)
        prediction = 0.01 if index % 2 == 0 else -0.01
        result.append(
            {
                "timestamp": timestamp.isoformat(),
                "available_to_model_time": timestamp.isoformat(),
                "label_available_time": (timestamp + timedelta(minutes=label_delay_minutes)).isoformat(),
                "prediction": prediction,
                "actual_return": prediction * 0.8,
                "signal_id": "alpha_a" if index % 2 == 0 else "alpha_b",
                "symbol": "BTCUSDT",
                "feature_families": ["price"],
            }
        )
    return result


class ResearchTests(unittest.TestCase):
    def test_chronological_split_never_shuffles(self) -> None:
        rows = list(reversed(_rows(count=10)))
        split = chronological_split(rows, train_fraction=0.6, validation_fraction=0.2)
        ordered = [rows[index]["timestamp"] for index in split.ordered_indices]
        self.assertEqual(ordered, sorted(ordered))
        self.assertLess(
            max(rows[index]["timestamp"] for index in split.train_indices),
            min(rows[index]["timestamp"] for index in split.test_indices),
        )

    def test_purged_split_removes_overlapping_label(self) -> None:
        rows = _rows(count=8, label_delay_minutes=3)
        folds = list(PurgedTimeSeriesSplit(n_splits=1, min_train_size=2, test_size=2).split(rows))
        self.assertTrue(folds)
        self.assertTrue(folds[0].purged_indices)
        self.assertTrue(set(folds[0].train_indices).isdisjoint(folds[0].test_indices))

    def test_walk_forward_metrics_and_signal_analysis(self) -> None:
        rows = _rows(count=10)
        walk = WalkForwardEvaluator(train_size=4, test_size=2, step_size=2).evaluate(rows)
        self.assertGreaterEqual(len(walk.folds), 1)
        self.assertIn("information_coefficient", walk.evaluation.metrics)
        metrics = evaluate_observations(rows).metrics
        self.assertIn("maximum_drawdown", metrics)
        analysis = analyze_signal_independence(rows, correlation_threshold=0.8)
        self.assertEqual(len(analysis.pairs), 1)

    def test_annualization_uses_forecast_horizon_instead_of_row_spacing(self) -> None:
        rows = _rows(count=6)
        result = evaluate_observations(rows, forecast_horizon_seconds=300)
        expected = 365.25 * 24 * 60 * 60 / 300
        self.assertAlmostEqual(float(result.metrics["annualization_factor"]), expected)
        self.assertAlmostEqual(
            float(result.performance_by_symbol["BTCUSDT"]["annualization_factor"]),
            expected,
        )

    def test_execution_assumptions_are_explicit_deterministic_and_uncalibrated(self) -> None:
        timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
        observations = (
            EvaluationObservation(
                timestamp=timestamp,
                prediction=1.0,
                actual_return=0.01,
                symbol="BTCUSDT",
                position=1.0,
                transaction_cost=0.0001,
                metadata={"requested_volume_participation": 0.50},
            ),
            EvaluationObservation(
                timestamp=timestamp + timedelta(minutes=5),
                prediction=1.0,
                actual_return=0.01,
                symbol="UNAVAILABLE",
                position=1.0,
                metadata={"symbol_available": False},
            ),
            EvaluationObservation(
                timestamp=timestamp + timedelta(minutes=10),
                prediction=1.0,
                actual_return=0.01,
                symbol="MISSING",
                position=1.0,
                metadata={"missing_data": True},
            ),
            EvaluationObservation(
                timestamp=timestamp + timedelta(minutes=15),
                prediction=1.0,
                actual_return=0.01,
                symbol="CHANGED",
                position=1.0,
                metadata={"coverage_changed": True},
            ),
        )
        assumptions = ExecutionAssumptions(
            fee_rate=0.001,
            spread_rate=0.002,
            slippage_rate=0.003,
            latency_seconds=2,
            latency_cost_rate_per_second=0.001,
            funding_rate_per_period=0.004,
            partial_fill_fraction=0.8,
            max_volume_participation=0.25,
            market_impact_rate=0.005,
        )
        applied = apply_execution_assumptions(observations, assumptions)
        execution = applied[0].metadata["execution"]
        self.assertEqual(applied[0].position, 0.5)
        self.assertEqual(execution["fill_fraction"], 0.5)
        self.assertIn("VOLUME_PARTICIPATION_CAPPED", execution["reason_codes"])
        self.assertIn("PARTIAL_FILL_APPLIED", execution["reason_codes"])
        self.assertAlmostEqual(applied[0].transaction_cost, 0.00735)
        self.assertFalse(assumptions.model_dump()["calibrated"])
        self.assertEqual(applied[1].position, 0.0)
        self.assertIn(
            "UNAVAILABLE_SYMBOL_SKIPPED",
            applied[1].metadata["execution"]["reason_codes"],
        )
        self.assertIn("MISSING_DATA_SKIPPED", applied[2].metadata["execution"]["reason_codes"])
        self.assertIn("DATA_COVERAGE_CHANGED", applied[3].metadata["execution"]["reason_codes"])

    def test_incremental_signal_requires_positive_equal_weight_utility(self) -> None:
        timestamps = tuple(
            datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=index)
            for index in range(8)
        )
        incumbent_pnl = (0.010, 0.020, 0.005, 0.015, 0.012, 0.018, 0.007, 0.013)
        candidate_pnl = (-0.004, -0.011, -0.007, -0.003, -0.009, -0.005, -0.012, -0.006)
        signals = {
            "incumbent": SignalSeries(
                signal_id="incumbent",
                timestamps=timestamps,
                predictions=incumbent_pnl,
                positions=(1.0,) * len(timestamps),
                pnl=incumbent_pnl,
            ),
            "candidate": SignalSeries(
                signal_id="candidate",
                timestamps=timestamps,
                predictions=candidate_pnl,
                positions=(1.0,) * len(timestamps),
                pnl=candidate_pnl,
            ),
        }
        contribution = assess_incremental_contribution(
            "candidate",
            ["incumbent"],
            signals,
            correlation_threshold=0.80,
        )
        self.assertLess(float(contribution["max_absolute_pnl_correlation"]), 0.80)
        self.assertFalse(contribution["is_incremental"])
        self.assertLess(float(contribution["marginal_utility"]["net_expectancy"]), 0.0)
        self.assertIn("NON_POSITIVE_MARGINAL_NET_EXPECTANCY", contribution["reason_codes"])
        saturation = analyze_ensemble_saturation(["incumbent", "candidate"], signals)
        self.assertEqual(saturation["saturation_point"], 2)
        self.assertEqual(saturation["diminishing_return_point"], 2)
        self.assertTrue(saturation["curve"][1]["diminishing_return"])
        standalone = assess_incremental_contribution("candidate", [], signals)
        self.assertFalse(standalone["is_incremental"])
        self.assertIn("NON_POSITIVE_MARGINAL_NET_EXPECTANCY", standalone["reason_codes"])

    def test_persisted_research_identifiers_are_stable_and_bounded(self) -> None:
        original = "candidate-" + "x" * 100
        first = persisted_research_id(original, field_name="candidate_id")
        self.assertEqual(first, persisted_research_id(original, field_name="candidate_id"))
        self.assertEqual(len(first), 64)
        self.assertNotEqual(first, persisted_research_id(original[:-1] + "y", field_name="candidate_id"))


if __name__ == "__main__":
    unittest.main()
