"""Focused regression tests for leakage-safe local research utilities."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from app.research import (
    PurgedTimeSeriesSplit,
    WalkForwardEvaluator,
    analyze_signal_independence,
    chronological_split,
    evaluate_observations,
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


if __name__ == "__main__":
    unittest.main()
