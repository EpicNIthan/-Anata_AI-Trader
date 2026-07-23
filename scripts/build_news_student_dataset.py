"""Build a deduplicated, compact JSONL dataset for the lightweight news student."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from news_student_utils import ensure_writable_output, iter_jsonl, normalized_student_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local student dataset from validated teacher labels.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--keep-duplicates", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input does not exist: {args.input}")
    if not 0.0 <= args.min_confidence <= 1.0:
        raise SystemExit("--min-confidence must be between zero and one")
    ensure_writable_output(args.output, overwrite=args.overwrite)
    rows: list[dict[str, Any]] = []
    rejected = 0
    duplicates = 0
    seen: set[str] = set()
    for _, raw in iter_jsonl(args.input):
        try:
            row = normalized_student_row(raw)
            confidence = float(row["teacher_event"]["confidence"])
            if confidence < args.min_confidence:
                rejected += 1
                continue
            content_hash = str(row["content_hash"])
            if not args.keep_duplicates and content_hash in seen:
                duplicates += 1
                continue
            seen.add(content_hash)
            rows.append(row)
        except Exception:
            rejected += 1
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "completed",
                "input": str(args.input),
                "output": str(args.output),
                "dataset_rows": len(rows),
                "duplicate_rows_skipped": duplicates,
                "invalid_or_low_confidence_rows_skipped": rejected,
                "next_step": "Train a local student; this does not deploy or activate anything.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
