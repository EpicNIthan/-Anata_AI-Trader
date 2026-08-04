"""Dependency-free helpers shared by local news teacher/student commands."""

from __future__ import annotations

import hashlib
import gzip
import io
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.intelligence.schemas import IntelligenceValidationError, NewsDocument, StructuredNewsEvent  # noqa: E402


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_jsonl_lines(lines: Iterable[str], *, source: str) -> Iterator[tuple[int, dict[str, Any]]]:
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise ValueError(f"{source}:{line_number} must contain a JSON object")
        yield line_number, parsed


def _iter_jsonl_file(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = sorted(
                name
                for name in archive.namelist()
                if Path(name).name.lower() in {"news_articles.jsonl", "news_articles.jsonl.gz"}
            )
            if not members:
                raise ValueError(f"{path} contains no news_articles.jsonl(.gz) member")
            for member in members:
                with archive.open(member) as raw:
                    if member.lower().endswith(".gz"):
                        binary: Any = gzip.GzipFile(fileobj=raw)
                    else:
                        binary = raw
                    with io.TextIOWrapper(binary, encoding="utf-8") as handle:
                        yield from _parse_jsonl_lines(handle, source=f"{path}!{member}")
        return
    opener = gzip.open if path.suffix.lower() == ".gz" else Path.open
    if opener is gzip.open:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            yield from _parse_jsonl_lines(handle, source=str(path))
    else:
        with path.open("r", encoding="utf-8") as handle:
            yield from _parse_jsonl_lines(handle, source=str(path))


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield JSON objects from JSONL, gzip, a daily ZIP, or a ZIP directory.

    Directory traversal is deterministic and accepts only ``.jsonl``,
    ``.jsonl.gz``, and ``.zip`` inputs. Daily raw-data ZIPs are narrowed to their
    ``news_articles.jsonl.gz`` member, so operational JSONL files cannot
    accidentally become teacher examples.
    """

    if path.is_dir():
        inputs = sorted(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file()
            and (
                candidate.suffix.lower() in {".jsonl", ".zip"}
                or candidate.name.lower().endswith(".jsonl.gz")
            )
        )
        if not inputs:
            raise ValueError(f"{path} contains no JSONL, JSONL.GZ, or ZIP inputs")
        for candidate in inputs:
            yield from _iter_jsonl_file(candidate)
        return
    if not path.is_file():
        raise FileNotFoundError(f"Input does not exist: {path}")
    yield from _iter_jsonl_file(path)


def ensure_writable_output(path: Path, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}. Pass --overwrite to replace it.")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]], *, overwrite: bool) -> int:
    """Write a JSONL output after an explicit overwrite check."""

    ensure_writable_output(path, overwrite=overwrite)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, default=str))
            handle.write("\n")
            count += 1
    return count


def write_json(path: Path, value: Mapping[str, Any], *, overwrite: bool) -> None:
    ensure_writable_output(path, overwrite=overwrite)
    path.write_text(json.dumps(dict(value), indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8")


def document_from_row(row: Mapping[str, Any]) -> NewsDocument:
    """Accept both source-news rows and previously nested teacher records."""

    nested = row.get("document")
    value = nested if isinstance(nested, Mapping) else row
    if not isinstance(nested, Mapping) and value.get("text") and not value.get("title"):
        text = str(value["text"])
        # Compact student datasets intentionally store one text field.  Recreate
        # a valid source document without changing the original text payload.
        value = {
            **dict(value),
            "title": text.splitlines()[0][:200] or "untitled news item",
            "content": text,
            "source": value.get("source") or "student_dataset",
        }
    return NewsDocument.from_mapping(value)


def event_from_row(row: Mapping[str, Any], *, document: NewsDocument) -> StructuredNewsEvent:
    """Extract a validated teacher event from common JSONL field names."""

    candidates = ("teacher_event", "event", "structured_event", "label")
    event_payload: Any = None
    for name in candidates:
        if name in row:
            event_payload = row[name]
            break
    if event_payload is None:
        # Permit a file containing event objects with source-document fields alongside them.
        event_payload = row
    if not isinstance(event_payload, Mapping):
        raise IntelligenceValidationError("teacher event must be a JSON object")
    return StructuredNewsEvent.from_mapping(
        event_payload,
        provider=str(event_payload.get("provider") or row.get("teacher_provider") or "teacher"),
        model=event_payload.get("model") or row.get("teacher_model"),
        prompt_version=str(event_payload.get("prompt_version") or row.get("prompt_version") or "teacher-v1"),
        source_reference=document.source_reference,
        source_text=document.text,
    )


def normalized_student_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a validated teacher row to the compact student-dataset format."""

    document = document_from_row(row)
    event = event_from_row(row, document=document)
    return {
        "text": document.text,
        "content_hash": document.content_hash,
        "published_at": document.published_at.isoformat() if document.published_at else None,
        "available_to_model_at": document.available_to_model_at.isoformat() if document.available_to_model_at else None,
        "teacher_event": event.model_dump(),
        "teacher_provider": event.provider,
        "teacher_model": event.model,
        "prompt_version": event.prompt_version,
    }
