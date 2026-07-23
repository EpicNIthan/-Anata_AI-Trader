"""Evaluate a compact student against held-out structured teacher labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from news_student_utils import document_from_row, event_from_row, iter_jsonl, write_json

from app.intelligence.providers import predict_json_student_artifact, validate_json_student_artifact


def _sentiment_label(value: float) -> str:
    if value > 0.15:
        return "positive"
    if value < -0.15:
        return "negative"
    return "neutral"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a JSON news student without any external service.")
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--report", type=Path, help="Optional report JSON to create.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.artifact.is_file() or not args.dataset.is_file():
        raise SystemExit("Both --artifact and --dataset must exist.")
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    validate_json_student_artifact(artifact)
    rows = 0
    sentiment_correct = 0
    event_correct = 0
    sentiment_absolute_error = 0.0
    skipped = 0
    for _, row in iter_jsonl(args.dataset):
        try:
            document = document_from_row(row)
            event = event_from_row(row, document=document)
            predicted = predict_json_student_artifact(artifact, document.text)
            rows += 1
            sentiment_correct += int(predicted["sentiment_label"].lower() == _sentiment_label(event.sentiment))
            event_correct += int(predicted["event_type"] == event.event_type.value)
            sentiment_absolute_error += abs(float(predicted["sentiment"]) - event.sentiment)
        except Exception:
            skipped += 1
    if rows == 0:
        raise SystemExit("No valid evaluation rows were found.")
    report: dict[str, Any] = {
        "status": "evaluated",
        "artifact": str(args.artifact),
        "artifact_version": artifact.get("version"),
        "dataset": str(args.dataset),
        "evaluated_rows": rows,
        "skipped_rows": skipped,
        "sentiment_label_accuracy": sentiment_correct / rows,
        "event_type_accuracy": event_correct / rows,
        "sentiment_mean_absolute_error": sentiment_absolute_error / rows,
        "interpretation": "Imitation metrics only; they are not evidence of trading profitability.",
    }
    if args.report:
        write_json(args.report, report, overwrite=args.overwrite)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
