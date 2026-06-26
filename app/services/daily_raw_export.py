from __future__ import annotations

import shutil
import zipfile
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Candle
from app.services.raw_data_export import RAW_EXPORT_ROOT, _export_range, _write_raw_folder


def _utc_day_start(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return datetime.combine(value.date(), time.min, tzinfo=timezone.utc)


def _default_range(session: Session) -> tuple[datetime, datetime]:
    earliest = session.scalar(select(func.min(Candle.open_time)).where(Candle.open_time.is_not(None)))
    start = _utc_day_start(earliest) if earliest else _utc_day_start(datetime.now(timezone.utc))
    # Use now, not tomorrow, so the current under-24h day is included but not extended into fake future time.
    end = datetime.now(timezone.utc)
    if end <= start:
        end = start + timedelta(days=1)
    return start, end


def _daily_ranges(start: datetime, end: datetime):
    current = _utc_day_start(start)
    while current < end:
        next_day = current + timedelta(days=1)
        yield current, min(next_day, end)
        current = next_day


def create_daily_raw_data_archive(session: Session, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = payload or {}
    payload = dict(payload)
    start, end, _day_name = _export_range(payload)
    if start is None and end is None:
        start, end = _default_range(session)
    elif start is None:
        start = _default_range(session)[0]
    elif end is None:
        end = datetime.now(timezone.utc)
    if end <= start:
        end = start + timedelta(days=1)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_id = f"raw_daily_export_{timestamp}.zip"
    work_root = RAW_EXPORT_ROOT / archive_id.removesuffix(".zip")
    if work_root.exists():
        shutil.rmtree(work_root)
    root_folder = work_root / "daily_raw_data"
    root_folder.mkdir(parents=True, exist_ok=True)

    day_manifests: dict[str, Any] = {}
    total_row_counts: dict[str, int] = {}
    days: list[str] = []
    for day_start, day_end in _daily_ranges(start, end):
        day_name = day_start.date().isoformat()
        days.append(day_name)
        day_folder = root_folder / day_name
        day_options = {key: value for key, value in payload.items() if key not in {"use_all_data", "date", "since_date", "until_date", "daily_split"}}
        manifest = _write_raw_folder(session, day_folder, start=day_start, end=day_end, options=day_options)
        day_manifests[day_name] = manifest
        for table_name, count in (manifest.get("row_counts") or {}).items():
            total_row_counts[table_name] = total_row_counts.get(table_name, 0) + int(count or 0)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "daily_split",
        "days": days,
        "day_count": len(days),
        "time_range": {"start": start.isoformat(), "end": end.isoformat()},
        "total_row_counts": total_row_counts,
        "day_manifests": day_manifests,
        "options": payload,
    }
    (root_folder / "daily_manifest.json").write_text(__import__("json").dumps(summary, indent=2, default=str), encoding="utf-8")

    archive_path = RAW_EXPORT_ROOT / archive_id
    if archive_path.exists():
        archive_path.unlink()
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(work_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(work_root))

    return {
        "status": "ok",
        "archive_id": archive_id,
        "archive_path": str(archive_path),
        "download_url": f"/api/raw-data/download/{archive_id}",
        "size_bytes": archive_path.stat().st_size,
        "manifest": summary,
    }
