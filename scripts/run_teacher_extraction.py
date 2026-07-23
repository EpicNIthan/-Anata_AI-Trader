"""Create baseline structured teacher labels from raw-news JSONL, fully offline.

This command intentionally uses the deterministic Level-0 teacher.  A heavier
local teacher can write the same schema and then be checked with
``validate_teacher_labels.py``; it is never required by Railway.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from news_student_utils import document_from_row, ensure_writable_output, iter_jsonl

from app.intelligence.providers import LocalRuleProvider


async def _extract(input_path: Path, *, limit: int | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    provider = LocalRuleProvider()
    labeled: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for line_number, row in iter_jsonl(input_path):
        if limit is not None and len(labeled) >= limit:
            break
        try:
            document = document_from_row(row)
            response = await provider.enrich(document, prompt_version="offline-rule-teacher-v1")
            labeled.append(
                {
                    "document": document.model_dump(),
                    "teacher_event": response.event.model_dump(),
                    "teacher_provider": provider.name,
                    "teacher_model": provider.model,
                }
            )
        except Exception as exc:
            rejected.append({"line_number": line_number, "error": type(exc).__name__, "message": str(exc)})
    return labeled, rejected


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate safe offline structured teacher labels from news JSONL.")
    parser.add_argument("--input", required=True, type=Path, help="Raw news JSONL with title/content/source fields.")
    parser.add_argument("--output", required=True, type=Path, help="Teacher-label JSONL to create.")
    parser.add_argument("--rejects", type=Path, help="Optional JSONL report for invalid input rows.")
    parser.add_argument("--limit", type=int, help="Maximum source rows to process.")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly permit replacing output files.")
    args = parser.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input does not exist: {args.input}")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    ensure_writable_output(args.output, overwrite=args.overwrite)
    if args.rejects:
        ensure_writable_output(args.rejects, overwrite=args.overwrite)
    labeled, rejected = asyncio.run(_extract(args.input, limit=args.limit))
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in labeled:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    if args.rejects:
        with args.rejects.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rejected:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "completed",
                "teacher": "offline deterministic rule teacher",
                "input": str(args.input),
                "output": str(args.output),
                "labeled_rows": len(labeled),
                "rejected_rows": len(rejected),
                "next_step": "Validate this file before building a student dataset.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
