"""Analyze prediction, position, PnL, drawdown, trade, and feature overlap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_utils import ensure_new_output, read_research_rows

from app.research import analyze_signal_independence


def main() -> None:
    parser = argparse.ArgumentParser(description="Group highly correlated stored signals before ensemble allocation.")
    parser.add_argument("--input", required=True, type=Path, help="Rows must include signal_id, timestamp, prediction, actual_return.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    ensure_new_output(args.output, overwrite=args.overwrite)
    rows = read_research_rows(args.input)
    if not any(row.get("signal_id") for row in rows):
        raise SystemExit("Input needs a signal_id column/field for correlation analysis.")
    analysis = analyze_signal_independence(rows, correlation_threshold=args.threshold)
    args.output.write_text(json.dumps(analysis.model_dump(), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "completed",
                "output": str(args.output),
                "pair_count": len(analysis.pairs),
                "correlated_groups": [list(group) for group in analysis.correlated_groups],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
