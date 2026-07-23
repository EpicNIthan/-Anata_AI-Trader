"""Train a compact, dependency-free Naive-Bayes news student artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from news_student_utils import ensure_writable_output, event_from_row, iter_jsonl, sha256_file

from app.intelligence.providers import tokenize_news_text
from app.intelligence.schemas import NewsDocument


def _sentiment_label(value: float) -> str:
    if value > 0.15:
        return "positive"
    if value < -0.15:
        return "negative"
    return "neutral"


def _period(rows: list[dict[str, Any]]) -> dict[str, str | None]:
    values = sorted(value for row in rows if (value := row.get("published_at") or row.get("available_to_model_at")))
    return {"first": values[0] if values else None, "last": values[-1] if values else None}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small Railway-compatible student from structured teacher labels.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="JSON student artifact to create.")
    parser.add_argument("--version", help="Optional explicit student version.")
    parser.add_argument("--max-vocabulary", type=int, default=8_000)
    parser.add_argument("--max-tokens-per-row", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.dataset.is_file():
        raise SystemExit(f"Dataset does not exist: {args.dataset}")
    if args.max_vocabulary < 50 or args.max_tokens_per_row < 10:
        raise SystemExit("Vocabulary and token limits are too small for a useful student.")
    ensure_writable_output(args.output, overwrite=args.overwrite)

    task_label_counts: dict[str, Counter[str]] = {"sentiment": Counter(), "event_type": Counter()}
    task_token_counts: dict[str, dict[str, Counter[str]]] = {
        "sentiment": defaultdict(Counter),
        "event_type": defaultdict(Counter),
    }
    vocabulary = Counter()
    numeric_sums = Counter()
    asset_token_counts: dict[str, Counter[str]] = defaultdict(Counter)
    records: list[dict[str, Any]] = []
    teacher_versions: set[str] = set()
    skipped = 0
    for _, row in iter_jsonl(args.dataset):
        try:
            text = str(row.get("text") or "")
            if not text and isinstance(row.get("document"), dict):
                document = NewsDocument.from_mapping(row["document"])
                text = document.text
            else:
                document = NewsDocument(title=text[:200] or "untitled", content=text or "missing", source="student_dataset")
            event = event_from_row(row, document=document)
            tokens = tokenize_news_text(text)[: args.max_tokens_per_row]
            if not tokens:
                raise ValueError("row has no usable text tokens")
            labels = {"sentiment": _sentiment_label(event.sentiment), "event_type": event.event_type.value}
            for task, label in labels.items():
                task_label_counts[task][label] += 1
                task_token_counts[task][label].update(tokens)
            vocabulary.update(tokens)
            for field_name in ("importance", "severity", "novelty"):
                numeric_sums[field_name] += float(getattr(event, field_name))
            for asset in event.affected_assets:
                asset_token_counts[asset].update(tokens)
            records.append(
                {
                    "published_at": row.get("published_at"),
                    "available_to_model_at": row.get("available_to_model_at"),
                }
            )
            teacher_versions.add(f"{event.provider}:{event.model or 'unknown'}")
        except Exception:
            skipped += 1
    if len(records) < 3:
        raise SystemExit("Need at least three valid labeled rows to create a student artifact.")
    chosen_vocab = {token for token, _ in vocabulary.most_common(args.max_vocabulary)}
    tasks: dict[str, Any] = {}
    for task, label_counts in task_label_counts.items():
        filtered_counts: dict[str, dict[str, int]] = {}
        totals: dict[str, int] = {}
        for label, counts in task_token_counts[task].items():
            filtered = {token: int(count) for token, count in counts.items() if token in chosen_vocab}
            filtered_counts[label] = filtered
            totals[label] = sum(filtered.values())
        tasks[task] = {
            "label_counts": {label: int(count) for label, count in label_counts.items()},
            "token_counts": filtered_counts,
            "token_totals": totals,
        }
    asset_keywords: dict[str, list[str]] = {}
    for asset, counts in asset_token_counts.items():
        # Ticker tokens are useful even when a source uses a cashtag rather than a full name.
        candidate_tokens = [asset.lower(), *[token for token, _ in counts.most_common(12)]]
        asset_keywords[asset] = list(dict.fromkeys(candidate_tokens))[:12]
    dataset_hash = sha256_file(args.dataset)
    version = args.version or f"student-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{dataset_hash[:8]}"
    artifact = {
        "artifact_type": "anata_news_student_naive_bayes_v1",
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_rows": len(records),
        "skipped_rows": skipped,
        "dataset_path": str(args.dataset),
        "dataset_sha256": dataset_hash,
        "training_period": _period(records),
        "teacher_versions": sorted(teacher_versions),
        "vocabulary_size": len(chosen_vocab),
        "tasks": tasks,
        "numeric_means": {name: float(numeric_sums[name] / len(records)) for name in ("importance", "severity", "novelty")},
        "asset_keywords": asset_keywords,
        "default_time_horizon": "short_term",
        "paper_only": True,
        "notes": "Student imitates structured teacher labels; it produces context features only and cannot execute trades.",
    }
    args.output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "trained",
                "artifact": str(args.output),
                "version": version,
                "training_rows": len(records),
                "skipped_rows": skipped,
                "vocabulary_size": len(chosen_vocab),
                "teacher_versions": sorted(teacher_versions),
                "next_step": "Evaluate and package this artifact explicitly; it is not activated automatically.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
