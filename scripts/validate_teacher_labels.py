"""Validate untrusted teacher JSONL against the typed structured-event schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from news_student_utils import document_from_row, ensure_writable_output, event_from_row, iter_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate teacher labels before they can train a local student.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Validated teacher-label JSONL to create.")
    parser.add_argument("--rejects", type=Path, help="Optional rejected-row JSONL to create.")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument(
        "--reject-unverified-numbers",
        action="store_true",
        help="Reject events whose claimed numeric literals are not found in the source article.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input does not exist: {args.input}")
    if not 0.0 <= args.min_confidence <= 1.0:
        raise SystemExit("--min-confidence must be between zero and one")
    ensure_writable_output(args.output, overwrite=args.overwrite)
    if args.rejects:
        ensure_writable_output(args.rejects, overwrite=args.overwrite)
    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    warnings = 0
    for line_number, row in iter_jsonl(args.input):
        try:
            document = document_from_row(row)
            event = event_from_row(row, document=document)
            source_warnings = event.validate_against_source(document.text)
            if event.confidence < args.min_confidence:
                raise ValueError(f"confidence {event.confidence:.3f} is below the configured minimum")
            if source_warnings and args.reject_unverified_numbers:
                raise ValueError("; ".join(source_warnings))
            warnings += len(source_warnings)
            valid.append(
                {
                    "document": document.model_dump(),
                    "teacher_event": event.model_dump(),
                    "teacher_provider": event.provider,
                    "teacher_model": event.model,
                    "prompt_version": event.prompt_version,
                }
            )
        except Exception as exc:
            rejected.append({"line_number": line_number, "error": type(exc).__name__, "message": str(exc)})
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in valid:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    if args.rejects:
        with args.rejects.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rejected:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "completed",
                "input": str(args.input),
                "output": str(args.output),
                "valid_rows": len(valid),
                "rejected_rows": len(rejected),
                "numeric_claim_warnings": warnings,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
