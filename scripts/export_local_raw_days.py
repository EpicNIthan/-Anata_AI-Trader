from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path

from sqlalchemy import func, select

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.session import SessionLocal
from app.services.raw_data_export import RAW_EXPORT_ROOT, RAW_TABLES, create_raw_data_archive


def _as_utc_day(value: datetime) -> date:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).date()


def _database_day_bounds(session) -> tuple[date, date]:
    timestamps: list[datetime] = []
    for spec in RAW_TABLES:
        oldest, newest = session.execute(select(func.min(spec.time_column), func.max(spec.time_column))).one()
        if oldest is not None:
            timestamps.append(oldest)
        if newest is not None:
            timestamps.append(newest)
    if not timestamps:
        raise RuntimeError("The restored database contains no timestamped raw-data rows.")
    return min(_as_utc_day(item) for item in timestamps), max(_as_utc_day(item) for item in timestamps)


def _verify_archive(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise RuntimeError(f"Archive verification failed at {bad_member}.")
        if not any(name.endswith("manifest.json") for name in archive.namelist()):
            raise RuntimeError("Archive has no manifest.json.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create one local raw_YYYY-MM-DD.zip per UTC day from the database selected by DATABASE_URL."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/raw_days"))
    parser.add_argument("--since-date", type=date.fromisoformat, default=None, help="UTC start day, YYYY-MM-DD")
    parser.add_argument("--until-date", type=date.fromisoformat, default=None, help="UTC end day, YYYY-MM-DD")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing local daily ZIPs.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with SessionLocal() as session:
        detected_start, detected_end = _database_day_bounds(session)
        start = args.since_date or detected_start
        end = args.until_date or detected_end
        if end < start:
            raise SystemExit("--until-date must not be before --since-date.")

        results: list[dict[str, object]] = []
        current = start
        while current <= end:
            target = args.output_dir / f"raw_{current.isoformat()}.zip"
            if target.exists() and not args.overwrite:
                results.append({"day": current.isoformat(), "path": str(target), "status": "kept_existing"})
                current = date.fromordinal(current.toordinal() + 1)
                continue

            exported = create_raw_data_archive(session, {"date": current.isoformat()})
            source = RAW_EXPORT_ROOT / str(exported["archive_id"])
            _verify_archive(source)
            if target.exists():
                target.unlink()
            shutil.move(str(source), target)
            results.append(
                {
                    "day": current.isoformat(),
                    "path": str(target),
                    "status": "written",
                    "size_bytes": target.stat().st_size,
                    "row_counts": (exported.get("manifest") or {}).get("row_counts", {}),
                }
            )
            current = date.fromordinal(current.toordinal() + 1)

    print(json.dumps({"status": "ok", "detected_range": [detected_start.isoformat(), detected_end.isoformat()], "days": results}, indent=2))


if __name__ == "__main__":
    main()
