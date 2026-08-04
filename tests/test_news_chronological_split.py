from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import pytest

from scripts.news_student_utils import sha256_file
from scripts.split_news_student_dataset import (
    SplitRequest,
    prepare_rows,
    resolve_split_request,
    split_news_dataset,
)


def _row(
    name: str,
    available_at: datetime,
    *,
    content_hash: str | None = None,
    model: str = "teacher-a",
    lineage: str | None = None,
) -> dict[str, object]:
    text = f"{name} headline\n\n{name} body"
    return {
        "text": text,
        "content_hash": content_hash or hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "available_to_model_at": available_at.isoformat(),
        "published_at": (available_at - timedelta(minutes=5)).isoformat(),
        "teacher_provider": "local-test",
        "teacher_model": model,
        "prompt_version": "prompt-v3",
        "teacher_event": {
            "provider": "local-test",
            "model": model,
            "prompt_version": "prompt-v3",
            "metadata": {"teacher_revision": "immutable-r1"},
        },
        "lineage": {"archive_member": lineage or f"{name}.jsonl", "source_id": name},
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_split_orders_deduplicates_preserves_lineage_and_has_no_overlap(tmp_path: Path) -> None:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    original = _row("duplicate", base)
    duplicate_later = {
        "document": {
            "title": "duplicate headline",
            "content": "duplicate body",
            "source": "fixture-wire",
            "content_hash": original["content_hash"],
            "available_to_model_at": (base + timedelta(hours=5)).isoformat(),
        },
        "teacher_provider": "local-test",
        "teacher_model": "teacher-a",
        "prompt_version": "prompt-v3",
        "teacher_event": {"metadata": {"teacher_revision": "immutable-r1"}},
        "lineage": {"archive_member": "duplicate-later.jsonl", "source_id": "duplicate"},
    }
    rows = [
        _row("f", base + timedelta(hours=4), model="teacher-b"),
        _row("c", base + timedelta(hours=1)),
        duplicate_later,
        _row("d", base + timedelta(hours=2)),
        original,
        _row("b", base + timedelta(hours=1)),
        _row("e", base + timedelta(hours=3)),
    ]
    input_path = tmp_path / "validated.jsonl"
    _write_jsonl(input_path, rows)

    result = split_news_dataset(
        input_path=input_path,
        output_dir=tmp_path / "split",
        request=SplitRequest("fractions", (0.5, 0.25, 0.25)),
    )

    partitions = {
        name: _read_jsonl(tmp_path / "split" / f"news_student_{name}.jsonl")
        for name in ("train", "validation", "holdout")
    }
    combined = [row for name in ("train", "validation", "holdout") for row in partitions[name]]
    timestamps = [datetime.fromisoformat(str(row["available_to_model_at"])) for row in combined]
    assert timestamps == sorted(timestamps)
    assert len(combined) == 6
    assert result["deduplication"]["duplicate_rows_dropped"] == 1
    retained_duplicate = next(row for row in combined if row["content_hash"] == original["content_hash"])
    assert retained_duplicate["lineage"] == original["lineage"]

    hash_sets = [{str(row["content_hash"]) for row in partition} for partition in partitions.values()]
    assert hash_sets[0].isdisjoint(hash_sets[1])
    assert hash_sets[0].isdisjoint(hash_sets[2])
    assert hash_sets[1].isdisjoint(hash_sets[2])
    assert result["integrity"] == {
        "content_hash_overlap": False,
        "strict_period_order": True,
        "rows_accounted_for": True,
    }
    assert {version["model"] for version in result["teacher_versions"]} == {"teacher-a", "teacher-b"}


def test_equal_boundary_timestamps_stay_together_and_exact_counts_must_align(tmp_path: Path) -> None:
    base = datetime(2026, 2, 1, tzinfo=timezone.utc)
    rows = [
        _row("a", base),
        _row("b", base + timedelta(hours=1)),
        _row("c", base + timedelta(hours=1)),
        _row("d", base + timedelta(hours=2)),
        _row("e", base + timedelta(hours=3)),
    ]
    input_path = tmp_path / "validated.jsonl"
    _write_jsonl(input_path, list(reversed(rows)))

    result = split_news_dataset(
        input_path=input_path,
        output_dir=tmp_path / "aligned",
        request=SplitRequest("counts", (1, 2, 2)),
    )
    assert [result["partitions"][name]["rows"] for name in ("train", "validation", "holdout")] == [1, 2, 2]
    assert result["partitions"]["train"]["period"]["last_available_to_model_at"] < result["partitions"]["validation"]["period"]["first_available_to_model_at"]

    with pytest.raises(ValueError, match="sharing a boundary"):
        split_news_dataset(
            input_path=input_path,
            output_dir=tmp_path / "misaligned",
            request=SplitRequest("counts", (2, 1, 2)),
        )


def test_split_outputs_and_checksums_are_byte_deterministic(tmp_path: Path) -> None:
    base = datetime(2026, 3, 1, tzinfo=timezone.utc)
    rows = [_row(str(index), base + timedelta(hours=index)) for index in range(8)]
    input_path = tmp_path / "validated.jsonl"
    _write_jsonl(input_path, list(reversed(rows)))

    first = split_news_dataset(
        input_path=input_path,
        output_dir=tmp_path / "first",
        request=SplitRequest("fractions", (0.5, 0.25, 0.25)),
    )
    second = split_news_dataset(
        input_path=input_path,
        output_dir=tmp_path / "second",
        request=SplitRequest("fractions", (0.5, 0.25, 0.25)),
    )

    for name in ("train", "validation", "holdout"):
        assert first["partitions"][name]["sha256"] == second["partitions"][name]["sha256"]
        assert sha256_file(tmp_path / "first" / f"news_student_{name}.jsonl") == sha256_file(
            tmp_path / "second" / f"news_student_{name}.jsonl"
        )
    assert (tmp_path / "first" / "news_student_split_manifest.json").read_bytes() == (
        tmp_path / "second" / "news_student_split_manifest.json"
    ).read_bytes()


def test_duplicate_hash_with_different_source_text_is_rejected() -> None:
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    shared_hash = "a" * 64
    rows = [_row("one", base, content_hash=shared_hash), _row("two", base, content_hash=shared_hash)]

    with pytest.raises(ValueError, match="unsafe content-hash collision"):
        prepare_rows(enumerate(rows, start=1))


def test_split_refuses_overwrite_and_validates_fraction_or_count_mode(tmp_path: Path) -> None:
    base = datetime(2026, 5, 1, tzinfo=timezone.utc)
    input_path = tmp_path / "validated.jsonl"
    _write_jsonl(input_path, [_row(str(index), base + timedelta(hours=index)) for index in range(6)])
    output_dir = tmp_path / "split"
    request = resolve_split_request(fractions=(0.5, 0.25, 0.25), counts=(None, None, None))
    split_news_dataset(input_path=input_path, output_dir=output_dir, request=request)

    before = (output_dir / "news_student_train.jsonl").read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        split_news_dataset(input_path=input_path, output_dir=output_dir, request=request)
    assert (output_dir / "news_student_train.jsonl").read_bytes() == before

    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_split_request(fractions=(0.5, 0.25, 0.25), counts=(3, 1, 2))
    with pytest.raises(ValueError, match="sum to exactly 1.0"):
        resolve_split_request(fractions=(0.5, 0.3, 0.3), counts=(None, None, None))
