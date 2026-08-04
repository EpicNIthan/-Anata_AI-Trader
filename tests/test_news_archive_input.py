from __future__ import annotations

import gzip
import json
from pathlib import Path
import zipfile

import pytest

from scripts.news_student_utils import iter_jsonl


def _daily_zip(path: Path, rows: list[dict[str, object]]) -> None:
    payload = "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("news_articles.jsonl.gz", gzip.compress(payload))
        archive.writestr("ai_decisions.jsonl.gz", gzip.compress(b'{"must":"not be read"}\n'))


def test_teacher_input_reads_verified_daily_zip_directly(tmp_path: Path) -> None:
    first = tmp_path / "raw_2026-01-01.zip"
    second = tmp_path / "raw_2026-01-02.zip"
    _daily_zip(first, [{"title": "first", "content": "one", "source": "wire"}])
    _daily_zip(second, [{"title": "second", "content": "two", "source": "wire"}])

    rows = [row for _, row in iter_jsonl(tmp_path)]

    assert [row["title"] for row in rows] == ["first", "second"]
    assert all("must" not in row for row in rows)


def test_daily_zip_without_news_member_fails_explicitly(tmp_path: Path) -> None:
    archive_path = tmp_path / "raw_empty.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("manifest.json", "{}")

    with pytest.raises(ValueError, match="contains no news_articles"):
        list(iter_jsonl(archive_path))
