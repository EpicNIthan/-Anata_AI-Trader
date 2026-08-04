"""Run a point-in-time historical evaluation of stored prediction observations."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from research_utils import ensure_new_output, load_candidate, observation_period, read_research_rows, sha256_file

from app.research import (
    ExecutionAssumptions,
    ExperimentDefinition,
    build_experiment_report,
    evaluate_observations,
    write_experiment_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate chronological stored predictions; does not train or trade.")
    parser.add_argument("--input", required=True, type=Path, help="JSONL, JSON array, or CSV prediction observations.")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--candidate", type=Path, help="Optional declarative CandidateStrategySpec JSON.")
    parser.add_argument("--dataset-version", help="Defaults to a SHA-256 fingerprint of the input.")
    parser.add_argument("--feature-version", default="unknown")
    parser.add_argument("--code-version", default="local")
    parser.add_argument("--annualization-factor", type=float)
    parser.add_argument("--execution-assumptions", type=Path, help="Optional deterministic assumptions JSON.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    ensure_new_output(args.report, overwrite=args.overwrite)
    rows = read_research_rows(args.input)
    if not rows:
        raise SystemExit("No input rows found.")
    candidate = load_candidate(args.candidate)
    assumptions = None
    if args.execution_assumptions is not None:
        payload = json.loads(args.execution_assumptions.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise SystemExit("Execution assumptions JSON must contain an object.")
        assumptions = ExecutionAssumptions.from_mapping(payload)
    dataset_version = args.dataset_version or f"sha256:{sha256_file(args.input)}"
    period = observation_period(rows)
    definition = ExperimentDefinition(
        experiment_id=f"historical-{candidate.candidate_id}-{sha256_file(args.input)[:12]}",
        candidate=candidate,
        dataset_version=dataset_version,
        feature_version=args.feature_version,
        code_version=args.code_version,
        random_seed=candidate.random_seed,
        test_period=period,
        configuration={
            "input": str(args.input),
            "mode": "historical_stored_predictions",
            "forecast_horizon_seconds": candidate.forecast_horizon,
            "execution_assumptions": assumptions.model_dump() if assumptions else "recorded_oos_costs",
        },
        notes="Point-in-time historical evaluation of already recorded predictions; no promotion decision is made.",
    )
    started_at = datetime.now(timezone.utc)
    result = evaluate_observations(
        rows,
        annualization_factor=args.annualization_factor,
        forecast_horizon_seconds=candidate.forecast_horizon,
        execution_assumptions=assumptions,
    )
    report = build_experiment_report(
        definition,
        result,
        artifacts=[args.input],
        notes=("Inputs are evaluated chronologically after availability validation.",),
        started_at=started_at,
        metadata={"input_sha256": sha256_file(args.input), "mode": "historical"},
    )
    write_experiment_report(report, args.report)
    print(json.dumps({"status": "completed", "report": str(args.report), "metrics": result.metrics}, indent=2, default=str))


if __name__ == "__main__":
    main()
