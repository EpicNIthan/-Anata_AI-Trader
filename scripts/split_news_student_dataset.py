"""Create deterministic chronological news train/validation/holdout splits.

The splitter consumes validated teacher rows or compact student rows without
normalizing them.  It orders solely by point-in-time availability, keeps equal
availability timestamps in one partition, and removes repeated content hashes
before selecting boundaries.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

try:
    from scripts.news_student_utils import iter_jsonl, sha256_file
except ImportError:  # Direct ``python scripts/split_news_student_dataset.py`` execution.
    from news_student_utils import iter_jsonl, sha256_file


_CONTENT_HASH = re.compile(r"[a-fA-F0-9]{64}")
_DEFAULT_FRACTIONS = (0.70, 0.15, 0.15)
_PARTITION_NAMES = ("train", "validation", "holdout")


@dataclass(frozen=True)
class PreparedRow:
    row: dict[str, Any]
    available_at: datetime
    available_at_iso: str
    content_hash: str
    content_identity: str
    canonical_hash: str
    source_line: int


@dataclass(frozen=True)
class SplitRequest:
    mode: str
    values: tuple[float | int, float | int, float | int]


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        rendered = json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row is not canonical JSON: {exc}") from exc
    return rendered.encode("utf-8")


def _nested_document(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    document = row.get("document")
    return document if isinstance(document, Mapping) else None


def _required_availability(row: Mapping[str, Any], *, line_number: int) -> tuple[datetime, str]:
    document = _nested_document(row)
    raw = row.get("available_to_model_at", row.get("available_to_model_time"))
    if raw in (None, "") and document is not None:
        raw = document.get("available_to_model_at", document.get("available_to_model_time"))
    if raw in (None, ""):
        raw = row.get("received_at", row.get("created_at"))
    if raw in (None, "") and document is not None:
        raw = document.get("received_at", document.get("created_at"))
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(
            f"line {line_number} has no available_to_model_at or received_at timestamp"
        )
    value = raw.strip()
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith(("Z", "z")) else value)
    except ValueError as exc:
        raise ValueError(f"line {line_number} has invalid available_to_model_at: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"line {line_number} available_to_model_at must include a timezone")
    utc_value = parsed.astimezone(timezone.utc)
    return utc_value, utc_value.isoformat()


def _required_content_hash(row: Mapping[str, Any], *, line_number: int) -> str:
    document = _nested_document(row)
    raw = row.get("content_hash")
    if raw in (None, "") and document is not None:
        raw = document.get("content_hash")
    value = str(raw or "").strip().lower()
    if not _CONTENT_HASH.fullmatch(value):
        raise ValueError(f"line {line_number} content_hash must be a 64-character SHA-256 digest")
    return value


def _content_identity(row: Mapping[str, Any], *, line_number: int) -> str:
    """Return source-content identity used to detect unsafe hash collisions."""

    def document_digest(title: str, content: str) -> str:
        material = f"{title.strip()}\n{content.strip()}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    document = _nested_document(row)
    if document is not None:
        title = document.get("title")
        content = document.get("content", document.get("raw_text", document.get("text")))
        if isinstance(title, str) and isinstance(content, str) and (title.strip() or content.strip()):
            return document_digest(title, content)
    text = row.get("text")
    if isinstance(text, str) and text.strip():
        # ``normalized_student_row`` renders source text as ``title\n\ncontent``.
        # Reconstruct the source hash material so teacher and student forms of
        # the same article can be safely recognized as one item.
        if "\n\n" in text:
            title, content = text.split("\n\n", 1)
            return document_digest(title, content)
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
    title = row.get("title")
    content = row.get("content", row.get("raw_text"))
    if isinstance(title, str) and isinstance(content, str) and (title.strip() or content.strip()):
        return document_digest(title, content)
    raise ValueError(f"line {line_number} has no source text with which to verify content-hash deduplication")


def prepare_rows(rows: Iterable[tuple[int, dict[str, Any]]]) -> tuple[list[PreparedRow], int]:
    """Validate, sort, and safely deduplicate rows.

    The earliest point-in-time occurrence wins.  For equal timestamps, declared
    content hash and canonical full-row hash provide an input-order-independent
    tie-break.  Differing source text under one declared hash is rejected rather
    than silently collapsed.
    """

    prepared: list[PreparedRow] = []
    for line_number, row in rows:
        available_at, available_at_iso = _required_availability(row, line_number=line_number)
        content_hash = _required_content_hash(row, line_number=line_number)
        canonical_hash = hashlib.sha256(_canonical_bytes(row)).hexdigest()
        prepared.append(
            PreparedRow(
                row=dict(row),
                available_at=available_at,
                available_at_iso=available_at_iso,
                content_hash=content_hash,
                content_identity=_content_identity(row, line_number=line_number),
                canonical_hash=canonical_hash,
                source_line=line_number,
            )
        )
    prepared.sort(
        key=lambda item: (
            item.available_at,
            item.content_hash,
            item.canonical_hash,
            item.source_line,
        )
    )
    unique: list[PreparedRow] = []
    identities: dict[str, str] = {}
    duplicate_count = 0
    for item in prepared:
        prior_identity = identities.get(item.content_hash)
        if prior_identity is None:
            identities[item.content_hash] = item.content_identity
            unique.append(item)
            continue
        if prior_identity != item.content_identity:
            raise ValueError(
                "unsafe content-hash collision: differing source text uses "
                f"content_hash {item.content_hash} (including line {item.source_line})"
            )
        duplicate_count += 1
    return unique, duplicate_count


def resolve_split_request(
    *,
    fractions: Sequence[float | None],
    counts: Sequence[int | None],
) -> SplitRequest:
    if len(fractions) != 3 or len(counts) != 3:
        raise ValueError("exactly three fraction and count slots are required")
    has_fraction = any(value is not None for value in fractions)
    has_count = any(value is not None for value in counts)
    if has_fraction and has_count:
        raise ValueError("fraction options and count options cannot be combined")
    if has_count:
        if any(value is None for value in counts):
            raise ValueError("--train-count, --validation-count, and --holdout-count must be supplied together")
        integer_values = tuple(int(value) for value in counts if value is not None)
        if any(value <= 0 for value in integer_values):
            raise ValueError("all partition counts must be positive")
        return SplitRequest("counts", integer_values)
    if not has_fraction:
        return SplitRequest("fractions", _DEFAULT_FRACTIONS)
    if any(value is None for value in fractions):
        raise ValueError("--train-fraction, --validation-fraction, and --holdout-fraction must be supplied together")
    float_values = tuple(float(value) for value in fractions if value is not None)
    if any(not 0.0 < value < 1.0 for value in float_values):
        raise ValueError("all partition fractions must be greater than zero and less than one")
    if abs(sum(float_values) - 1.0) > 1e-9:
        raise ValueError("partition fractions must sum to exactly 1.0")
    return SplitRequest("fractions", float_values)


def _timestamp_endpoints(rows: Sequence[PreparedRow]) -> list[int]:
    endpoints: list[int] = []
    for index, item in enumerate(rows, start=1):
        if index == len(rows) or rows[index].available_at != item.available_at:
            endpoints.append(index)
    return endpoints


def _fraction_boundaries(rows: Sequence[PreparedRow], values: Sequence[float | int]) -> tuple[int, int]:
    endpoints = _timestamp_endpoints(rows)
    if len(endpoints) < 3:
        raise ValueError("at least three distinct available_to_model_at timestamps are required")
    total = len(rows)
    targets = (
        total * float(values[0]),
        total * (float(values[0]) + float(values[1])),
    )
    first_options = endpoints[:-2]
    best: tuple[float, int, int] | None = None
    for first in first_options:
        minimum_second_index = bisect_left(endpoints, first + 1)
        maximum_second_index = len(endpoints) - 2
        pivot = bisect_left(endpoints, targets[1], lo=minimum_second_index, hi=maximum_second_index + 1)
        candidate_indexes = {
            max(minimum_second_index, min(maximum_second_index, pivot)),
            max(minimum_second_index, min(maximum_second_index, pivot - 1)),
        }
        for second_index in candidate_indexes:
            second = endpoints[second_index]
            if not first < second < total:
                continue
            actual = (first, second - first, total - second)
            score = sum(abs(actual[index] - total * float(values[index])) for index in range(3))
            candidate = (score, first, second)
            if best is None or candidate < best:
                best = candidate
    if best is None:  # Defensive; three timestamp cohorts guarantee a solution.
        raise ValueError("could not select non-overlapping chronological partition boundaries")
    return best[1], best[2]


def partition_rows(rows: Sequence[PreparedRow], request: SplitRequest) -> tuple[list[PreparedRow], ...]:
    total = len(rows)
    if total < 3:
        raise ValueError("at least three unique rows are required")
    endpoints = set(_timestamp_endpoints(rows))
    if request.mode == "counts":
        train_count, validation_count, holdout_count = (int(value) for value in request.values)
        if train_count + validation_count + holdout_count != total:
            raise ValueError(
                "partition counts must sum to the deduplicated row count "
                f"({total}); got {train_count + validation_count + holdout_count}"
            )
        first, second = train_count, train_count + validation_count
        if first not in endpoints or second not in endpoints:
            raise ValueError(
                "explicit counts would split rows sharing a boundary available_to_model_at timestamp; "
                "choose timestamp-cohort-aligned counts"
            )
    else:
        first, second = _fraction_boundaries(rows, request.values)
    partitions = (list(rows[:first]), list(rows[first:second]), list(rows[second:]))
    if any(not partition for partition in partitions):
        raise ValueError("train, validation, and holdout partitions must all be non-empty")
    return partitions


def _teacher_version(row: Mapping[str, Any]) -> tuple[str | None, str | None, str | None, str | None]:
    event = row.get("teacher_event")
    event = event if isinstance(event, Mapping) else {}
    metadata = event.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    provider = row.get("teacher_provider", event.get("provider"))
    model = row.get("teacher_model", event.get("model"))
    prompt = row.get("prompt_version", event.get("prompt_version"))
    revision = row.get("teacher_revision", metadata.get("teacher_revision"))
    return tuple(None if value is None else str(value) for value in (provider, model, prompt, revision))  # type: ignore[return-value]


def _teacher_versions(rows: Sequence[PreparedRow]) -> list[dict[str, str | None]]:
    return [
        {"provider": provider, "model": model, "prompt_version": prompt, "revision": revision}
        for provider, model, prompt, revision in sorted(
            {_teacher_version(item.row) for item in rows},
            key=lambda values: tuple(value or "" for value in values),
        )
    ]


def _write_partition(path: Path, rows: Sequence[PreparedRow]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in rows:
            handle.write(_canonical_bytes(item.row).decode("utf-8"))
            handle.write("\n")


def _partition_manifest(path: Path, rows: Sequence[PreparedRow]) -> dict[str, Any]:
    return {
        "file": path.name,
        "rows": len(rows),
        "period": {
            "first_available_to_model_at": rows[0].available_at_iso,
            "last_available_to_model_at": rows[-1].available_at_iso,
        },
        "sha256": sha256_file(path),
        "teacher_versions": _teacher_versions(rows),
    }


def split_news_dataset(
    *,
    input_path: Path,
    output_dir: Path,
    request: SplitRequest,
    prefix: str = "news_student",
    overwrite: bool = False,
) -> dict[str, Any]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input does not exist: {input_path}")
    if not prefix or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", prefix):
        raise ValueError("--prefix must contain only letters, digits, dot, underscore, or hyphen")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        name: output_dir / f"{prefix}_{name}.jsonl"
        for name in _PARTITION_NAMES
    }
    manifest_path = output_dir / f"{prefix}_split_manifest.json"
    all_outputs = [*outputs.values(), manifest_path]
    if input_path.resolve() in {path.resolve() for path in all_outputs}:
        raise ValueError("input must not also be one of the generated output paths")
    existing = [path for path in all_outputs if path.exists()]
    if existing and not overwrite:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing output(s): {rendered}. Pass --overwrite to replace them.")

    raw_rows = list(iter_jsonl(input_path))
    unique_rows, duplicate_count = prepare_rows(raw_rows)
    partitions = partition_rows(unique_rows, request)
    for name, rows in zip(_PARTITION_NAMES, partitions, strict=True):
        _write_partition(outputs[name], rows)

    partition_details = {
        name: _partition_manifest(outputs[name], rows)
        for name, rows in zip(_PARTITION_NAMES, partitions, strict=True)
    }
    train_rows, validation_rows, holdout_rows = partitions
    manifest = {
        "schema_version": "anata-news-chronological-split-v1",
        "input": {
            "file": input_path.name,
            "rows": len(raw_rows),
            "sha256": sha256_file(input_path),
        },
        "ordering": {
            "primary": "available_to_model_at_utc_else_received_at_utc",
            "tie_break": ["content_hash", "canonical_row_sha256", "source_line_for_identical_rows"],
            "random_shuffle": False,
            "equal_timestamp_cohorts_split": False,
        },
        "deduplication": {
            "key": "content_hash",
            "policy": "earliest_point_in_time_then_stable_tie_break",
            "unique_rows": len(unique_rows),
            "duplicate_rows_dropped": duplicate_count,
            "collision_check": "same_hash_requires_same_source_text",
        },
        "requested_split": {
            "mode": request.mode,
            "train": request.values[0],
            "validation": request.values[1],
            "holdout": request.values[2],
        },
        "partitions": partition_details,
        "teacher_versions": _teacher_versions(unique_rows),
        "integrity": {
            "content_hash_overlap": False,
            "strict_period_order": (
                train_rows[-1].available_at < validation_rows[0].available_at
                and validation_rows[-1].available_at < holdout_rows[0].available_at
            ),
            "rows_accounted_for": sum(len(rows) for rows in partitions) == len(unique_rows),
        },
        "paper_only": True,
    }
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False))
        handle.write("\n")
    return {**manifest, "manifest": str(manifest_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministically split validated news rows by point-in-time availability."
    )
    parser.add_argument("--input", required=True, type=Path, help="Validated teacher/student JSONL input.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prefix", default="news_student")
    parser.add_argument("--train-fraction", type=float)
    parser.add_argument("--validation-fraction", type=float)
    parser.add_argument("--holdout-fraction", type=float)
    parser.add_argument("--train-count", type=int)
    parser.add_argument("--validation-count", type=int)
    parser.add_argument("--holdout-count", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        request = resolve_split_request(
            fractions=(args.train_fraction, args.validation_fraction, args.holdout_fraction),
            counts=(args.train_count, args.validation_count, args.holdout_count),
        )
        result = split_news_dataset(
            input_path=args.input,
            output_dir=args.output_dir,
            request=request,
            prefix=args.prefix,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
