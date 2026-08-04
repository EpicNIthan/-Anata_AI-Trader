"""Streaming verification for Railway-to-laptop raw-data archives.

The verifier is intentionally dependency-free.  It validates the archive and every
table manifest before the downloader is allowed to request remote cleanup.  Checks are
performed against the compressed table bytes stored inside the ZIP so a checksum
proves the exact exported file arrived, not merely that it could be decompressed.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping


_TIME_COLUMNS: dict[str, str] = {
    "candles": "open_time",
    "live_candle_updates": "open_time",
    "market_ticks": "event_time",
    "news_articles": "created_at",
    "news_sentiment": "created_at",
    "external_data_events": "event_time",
    "features": "as_of",
    "training_features": "as_of",
    "paper_trades": "created_at",
    "positions": "opened_at",
    "ai_decisions": "created_at",
    "experience_buffer": "created_at",
    "account_equity": "timestamp",
    "model_versions": "created_at",
    "training_runs": "started_at",
}


def sha256_path(path: Path) -> str:
    """Hash a local file using bounded memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_member(archive: zipfile.ZipFile, member: str) -> str:
    digest = hashlib.sha256()
    with archive.open(member) as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _interval_seconds(value: Any) -> int | None:
    raw = str(value or "").strip().lower()
    if len(raw) < 2:
        return None
    try:
        amount = int(raw[:-1])
    except ValueError:
        return None
    multiplier = {"m": 60, "h": 3600, "d": 86_400, "w": 604_800}.get(raw[-1])
    return amount * multiplier if amount > 0 and multiplier else None


def _manifest_entries(archive: zipfile.ZipFile) -> list[tuple[str, dict[str, Any]]]:
    entries: list[tuple[str, dict[str, Any]]] = []
    for name in sorted(archive.namelist()):
        if PurePosixPath(name).name != "manifest.json":
            continue
        with archive.open(name) as handle:
            payload = json.loads(handle.read().decode("utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("tables_exported"), dict):
            entries.append((name, payload))
    return entries


def _rows(
    archive: zipfile.ZipFile,
    member: str,
    *,
    csv_mode: bool,
) -> Iterator[dict[str, Any]]:
    with archive.open(member) as compressed_member:
        with gzip.GzipFile(fileobj=compressed_member, mode="rb") as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", errors="strict", newline="") as text:
                if csv_mode:
                    yield from csv.DictReader(text)
                    return
                for line_number, line in enumerate(text, start=1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"{member}:{line_number} is not a JSON object")
                    yield value


def _local_rows(path: Path, *, csv_mode: bool) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8", errors="strict", newline="") as text:
        if csv_mode:
            yield from csv.DictReader(text)
            return
        for line_number, line in enumerate(text, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield value


def refresh_export_manifests(folder: Path) -> list[Path]:
    """Rebuild manifest facts after safe local daily-file merges.

    A merged laptop archive is a new immutable file, so retaining checksums and row
    counts from either input archive would be false evidence.  This function updates
    only derived verification fields and preserves source manifests under the existing
    merge-provenance block.
    """

    refreshed: list[Path] = []
    for manifest_path in sorted(folder.rglob("manifest.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        tables = payload.get("tables_exported") if isinstance(payload, dict) else None
        if not isinstance(tables, dict):
            continue
        counts: dict[str, int] = {}
        sizes: dict[str, int] = {}
        hashes: dict[str, str] = {}
        ranges: dict[str, dict[str, str | None]] = {}
        symbols: set[str] = set()
        for table_name, relative_name in tables.items():
            table_path = manifest_path.parent / str(relative_name)
            if not table_path.is_file():
                raise ValueError(f"manifest references missing local file {table_path}")
            count = 0
            first: datetime | None = None
            last: datetime | None = None
            time_column = _TIME_COLUMNS.get(str(table_name))
            for row in _local_rows(table_path, csv_mode=table_path.name.endswith(".csv.gz")):
                count += 1
                timestamp = _utc(row.get(time_column)) if time_column else None
                if time_column and timestamp is None:
                    raise ValueError(f"{table_path} row {count} has invalid {time_column}")
                if timestamp is not None:
                    first = timestamp if first is None or timestamp < first else first
                    last = timestamp if last is None or timestamp > last else last
                symbol = str(row.get("symbol") or "").strip().upper()
                if symbol:
                    symbols.add(symbol)
            counts[str(table_name)] = count
            sizes[str(relative_name)] = table_path.stat().st_size
            hashes[str(relative_name)] = sha256_path(table_path)
            ranges[str(table_name)] = {
                "first_timestamp": first.isoformat() if first else None,
                "last_timestamp": last.isoformat() if last else None,
            }
        payload["row_counts"] = counts
        payload["file_sizes"] = sizes
        payload["file_checksums_sha256"] = hashes
        payload["table_time_ranges"] = ranges
        payload["symbols"] = sorted(symbols) or list(payload.get("symbols") or [])
        payload["verification"] = {
            **(payload.get("verification") or {}),
            "writers_closed": True,
            "manifest_complete": True,
            "news_count": int(counts.get("news_articles", 0)),
            "derivatives_count": int(counts.get("external_data_events", 0)),
            "refreshed_after_local_merge": True,
        }
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        temporary.replace(manifest_path)
        refreshed.append(manifest_path)
    return refreshed


def _same_time(left: Any, right: datetime | None) -> bool:
    expected = _utc(left)
    if expected is None or right is None:
        return expected is right
    return math.isclose(expected.timestamp(), right.timestamp(), abs_tol=1e-6)


def verify_export_archive(path: Path, *, require_checksums: bool = True) -> dict[str, Any]:
    """Verify table bytes, rows, timestamps, and candle continuity in one archive.

    Raises no assertion errors; callers receive a structured result and can refuse
    destructive cleanup when ``valid`` is false.
    """

    path = path.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    actual_counts: dict[str, int] = defaultdict(int)
    table_ranges: dict[str, dict[str, str | None]] = {}
    files_verified = 0
    checksum_files_verified = 0
    missing_intervals = 0
    candle_gaps = 0
    candle_order_errors = 0
    candle_last: dict[tuple[str, str], datetime] = {}

    if not path.is_file():
        return {
            "valid": False,
            "errors": [f"archive does not exist: {path}"],
            "warnings": [],
            "successful_local_file_close": False,
        }

    archive_sha256 = sha256_path(path)
    try:
        with zipfile.ZipFile(path) as archive:
            corrupt = archive.testzip()
            if corrupt:
                errors.append(f"ZIP CRC validation failed for {corrupt}")
            manifests = _manifest_entries(archive)
            if not manifests:
                errors.append("no per-table manifest.json was found")

            for manifest_name, manifest in manifests:
                prefix = PurePosixPath(manifest_name).parent
                expected_counts = manifest.get("row_counts") or {}
                expected_sizes = manifest.get("file_sizes") or {}
                expected_hashes = manifest.get("file_checksums_sha256") or {}
                expected_ranges = manifest.get("table_time_ranges") or {}
                if require_checksums and not expected_hashes:
                    errors.append(f"{manifest_name} has no file_checksums_sha256 map")
                if not bool((manifest.get("verification") or {}).get("writers_closed")):
                    errors.append(f"{manifest_name} does not confirm closed export writers")

                for table_name, relative_name in (manifest.get("tables_exported") or {}).items():
                    member = (prefix / str(relative_name)).as_posix()
                    try:
                        info = archive.getinfo(member)
                    except KeyError:
                        errors.append(f"manifest references missing member {member}")
                        continue
                    expected_size = expected_sizes.get(str(relative_name))
                    if expected_size is not None and int(expected_size) != info.file_size:
                        errors.append(
                            f"{member} size mismatch: manifest={expected_size} downloaded={info.file_size}"
                        )
                    expected_hash = expected_hashes.get(str(relative_name))
                    if expected_hash:
                        actual_hash = _sha256_member(archive, member)
                        if actual_hash != str(expected_hash):
                            errors.append(f"{member} checksum mismatch")
                        else:
                            checksum_files_verified += 1

                    count = 0
                    first: datetime | None = None
                    last: datetime | None = None
                    time_column = _TIME_COLUMNS.get(str(table_name))
                    try:
                        for row in _rows(archive, member, csv_mode=str(relative_name).endswith(".csv.gz")):
                            count += 1
                            timestamp = _utc(row.get(time_column)) if time_column else None
                            if time_column and timestamp is None:
                                errors.append(f"{member} row {count} has invalid {time_column}")
                            if timestamp is not None:
                                first = timestamp if first is None or timestamp < first else first
                                last = timestamp if last is None or timestamp > last else last
                            if str(table_name) == "candles" and timestamp is not None:
                                key = (str(row.get("symbol") or "").upper(), str(row.get("interval") or ""))
                                previous = candle_last.get(key)
                                seconds = _interval_seconds(key[1])
                                if previous is not None:
                                    delta = (timestamp - previous).total_seconds()
                                    if delta <= 0:
                                        candle_order_errors += 1
                                    elif seconds and delta > seconds * 1.5:
                                        candle_gaps += 1
                                        missing_intervals += max(round(delta / seconds) - 1, 1)
                                candle_last[key] = timestamp
                    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
                        errors.append(f"cannot validate {member}: {exc}")
                        continue

                    files_verified += 1
                    expected_count = expected_counts.get(str(table_name))
                    if expected_count is None:
                        errors.append(f"{manifest_name} has no row count for {table_name}")
                    elif count != int(expected_count):
                        errors.append(
                            f"{member} row-count mismatch: manifest={expected_count} downloaded={count}"
                        )
                    actual_counts[str(table_name)] += count
                    observed_range = {
                        "first_timestamp": first.isoformat() if first else None,
                        "last_timestamp": last.isoformat() if last else None,
                    }
                    prior_range = table_ranges.get(str(table_name))
                    if prior_range:
                        prior_first = _utc(prior_range.get("first_timestamp"))
                        prior_last = _utc(prior_range.get("last_timestamp"))
                        combined_first = min(value for value in (prior_first, first) if value is not None) if prior_first or first else None
                        combined_last = max(value for value in (prior_last, last) if value is not None) if prior_last or last else None
                        table_ranges[str(table_name)] = {
                            "first_timestamp": combined_first.isoformat() if combined_first else None,
                            "last_timestamp": combined_last.isoformat() if combined_last else None,
                        }
                    else:
                        table_ranges[str(table_name)] = observed_range

                    expected_range = expected_ranges.get(str(table_name)) or {}
                    if expected_range and not _same_time(expected_range.get("first_timestamp"), first):
                        errors.append(f"{member} first timestamp differs from manifest")
                    if expected_range and not _same_time(expected_range.get("last_timestamp"), last):
                        errors.append(f"{member} last timestamp differs from manifest")

            if candle_order_errors:
                errors.append(f"candles contain {candle_order_errors} duplicate or non-monotonic intervals")
            if missing_intervals:
                warnings.append(
                    f"candles contain {candle_gaps} gaps with approximately {missing_intervals} missing intervals"
                )
    except (OSError, zipfile.BadZipFile, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"archive verification failed: {exc}")

    timestamps = [
        _utc(value)
        for item in table_ranges.values()
        for value in (item.get("first_timestamp"), item.get("last_timestamp"))
        if value
    ]
    first_timestamp = min(timestamps).isoformat() if timestamps else None
    last_timestamp = max(timestamps).isoformat() if timestamps else None
    return {
        "valid": not errors,
        "archive": str(path),
        "archive_sha256": archive_sha256,
        "archive_size_bytes": path.stat().st_size,
        "manifest_count": len(manifests) if "manifests" in locals() else 0,
        "files_verified": files_verified,
        "checksum_files_verified": checksum_files_verified,
        "row_counts": dict(sorted(actual_counts.items())),
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "table_time_ranges": table_ranges,
        "missing_interval_summary": {
            "gap_count": candle_gaps,
            "estimated_missing_intervals": missing_intervals,
            "duplicate_or_non_monotonic": candle_order_errors,
        },
        "news_count": int(actual_counts.get("news_articles", 0)),
        "derivatives_count": int(actual_counts.get("external_data_events", 0)),
        "export_manifest_verified": bool(files_verified),
        "successful_local_file_close": True,
        "errors": errors,
        "warnings": warnings,
    }
