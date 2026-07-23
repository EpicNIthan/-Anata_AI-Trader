"""Run one bounded local research-scheduler pass against labeled observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from research_utils import default_candidate, read_research_rows, sha256_file

from app.research import (
    CandidateStrategySpec,
    LabeledDataSnapshot,
    ResearchScheduler,
    ResearchSchedulerConfig,
)


def _load_candidates(path: Path | None) -> tuple[CandidateStrategySpec, ...]:
    if path is None:
        return (default_candidate(),)
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        payload = payload.get("candidates", [payload.get("candidate", payload)])
    if not isinstance(payload, list):
        raise ValueError("candidate configuration must be an object or a list of objects")
    return tuple(CandidateStrategySpec.from_dict(item) for item in payload if isinstance(item, Mapping))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local candidate evaluation with durable cursor state and no auto-promotion.")
    parser.add_argument("--input", required=True, type=Path, help="Labeled prediction JSONL/JSON/CSV.")
    parser.add_argument("--state", type=Path, default=Path("research_reports/scheduler_state.json"))
    parser.add_argument("--reports-dir", type=Path, default=Path("research_reports"))
    parser.add_argument("--candidates", type=Path, help="Optional declarative candidate object/list JSON.")
    parser.add_argument("--dataset-version", help="Defaults to an input SHA-256 fingerprint.")
    parser.add_argument("--feature-version", default="unknown")
    parser.add_argument("--minimum-new-rows", type=int, default=100)
    parser.add_argument("--max-candidates", type=int, default=25)
    parser.add_argument("--code-version", default="local")
    parser.add_argument("--force", action="store_true", help="Evaluate even when fewer than --minimum-new-rows are new.")
    args = parser.parse_args()
    rows = read_research_rows(args.input)
    if not rows:
        raise SystemExit("No labeled rows found.")
    candidates = _load_candidates(args.candidates)
    if not candidates:
        raise SystemExit("No valid candidates found.")
    dataset_hash = sha256_file(args.input)
    snapshot = LabeledDataSnapshot.from_rows(
        rows,
        dataset_version=args.dataset_version or f"sha256:{dataset_hash}",
        feature_version=args.feature_version,
        metadata={"input": str(args.input), "input_sha256": dataset_hash},
    )
    scheduler = ResearchScheduler(
        snapshot_provider=lambda: snapshot,
        candidate_provider=lambda _snapshot: candidates,
        config=ResearchSchedulerConfig(
            minimum_new_labeled_rows=args.minimum_new_rows,
            max_candidates_per_run=args.max_candidates,
            reports_dir=args.reports_dir,
            state_path=args.state,
            code_version=args.code_version,
            allow_upload=False,
            allow_auto_champion_promotion=False,
        ),
    )
    result = scheduler.run_once(force=args.force)
    print(
        json.dumps(
            {
                "status": result.status,
                "new_labeled_rows": result.new_labeled_rows,
                "reports": [report.definition.experiment_id for report in result.reports],
                "registered_challengers": len(result.registered_challengers),
                "uploads": len(result.uploads),
                "promotions": len(result.promotions),
                "messages": list(result.messages),
                "safety": "Uploads and automatic champion promotion are hard-disabled in this local command.",
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
