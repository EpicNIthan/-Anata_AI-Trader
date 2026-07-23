"""Run a chronological purged/embargoed walk-forward evaluation locally."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from research_utils import ensure_new_output, load_candidate, read_research_rows, sha256_file

from app.research import ExperimentDefinition, WalkForwardEvaluator, build_experiment_report, write_experiment_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate stored predictions with leakage-safe walk-forward folds.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--train-size", type=int, required=True)
    parser.add_argument("--validation-size", type=int, default=0)
    parser.add_argument("--test-size", type=int, required=True)
    parser.add_argument("--step-size", type=int)
    parser.add_argument("--purge-size", type=int, default=0)
    parser.add_argument("--embargo-size", type=int, default=0)
    parser.add_argument("--rolling", action="store_true", help="Use a rolling rather than expanding training window.")
    parser.add_argument("--dataset-version")
    parser.add_argument("--feature-version", default="unknown")
    parser.add_argument("--code-version", default="local")
    parser.add_argument("--annualization-factor", type=float)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.train_size < 1 or args.test_size < 1 or min(args.validation_size, args.purge_size, args.embargo_size) < 0:
        raise SystemExit("Train/test sizes must be positive and validation/purge/embargo sizes cannot be negative.")
    ensure_new_output(args.report, overwrite=args.overwrite)
    rows = read_research_rows(args.input)
    if not rows:
        raise SystemExit("No input rows found.")
    candidate = load_candidate(args.candidate)
    evaluator = WalkForwardEvaluator(
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        step_size=args.step_size,
        expanding=not args.rolling,
        purge_window=args.purge_size,
        embargo_window=args.embargo_size,
        annualization_factor=args.annualization_factor,
    )
    started_at = datetime.now(timezone.utc)
    walk_forward = evaluator.evaluate(rows)
    dataset_hash = sha256_file(args.input)
    definition = ExperimentDefinition(
        experiment_id=f"walk-forward-{candidate.candidate_id}-{dataset_hash[:12]}",
        candidate=candidate,
        dataset_version=args.dataset_version or f"sha256:{dataset_hash}",
        feature_version=args.feature_version,
        code_version=args.code_version,
        random_seed=candidate.random_seed,
        configuration={
            "input": str(args.input),
            "train_size": args.train_size,
            "validation_size": args.validation_size,
            "test_size": args.test_size,
            "step_size": args.step_size or args.test_size,
            "purge_size": args.purge_size,
            "embargo_size": args.embargo_size,
            "expanding": not args.rolling,
        },
        notes="Walk-forward evaluation of stored predictions; it has no model activation or trading side effect.",
    )
    report = build_experiment_report(
        definition,
        walk_forward.evaluation,
        fold_metrics=[fold.metrics for fold in walk_forward.fold_evaluations],
        artifacts=[args.input],
        notes=("Folds are chronological with configured purge and embargo windows.",),
        started_at=started_at,
        metadata={"walk_forward": walk_forward.model_dump(), "input_sha256": dataset_hash},
    )
    write_experiment_report(report, args.report)
    print(
        json.dumps(
            {
                "status": "completed",
                "report": str(args.report),
                "fold_count": len(walk_forward.folds),
                "metrics": walk_forward.evaluation.metrics,
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
