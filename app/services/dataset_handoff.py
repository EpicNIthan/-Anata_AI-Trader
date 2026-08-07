from __future__ import annotations

import csv
import gzip
import json
import shutil
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import DateTime, Integer, String, delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Mapped, Session, mapped_column

from app.db.models import Base, Candle
from app.services.raw_data_export import RAW_TABLES, _columns_for, _csv_safe, _write_raw_folder
from app.strategies import regime_models as _regime_models  # noqa: F401

HANDOFF_ROOT = Path("datasets") / "handoffs"
TIME_COLUMNS = (
    "event_time", "open_time", "as_of", "generated_at", "observed_at",
    "timestamp", "candle_close_time", "available_at", "filled_at",
    "started_at", "active_from", "created_at", "updated_at",
)
KEEP_TABLES = {
    "positions",
    "model_versions",
    "model_artifact_blobs",
    "regime_pullback_accounts",
    "regime_pullback_position_meta",
    "regime_pullback_daily_risk",
    "regime_pullback_symbol_risk",
    "regime_pullback_strategy_lock",
    "paper_sandbox_accounts",
    "champion_assignments",
    "strategy_candidates",
}
CORE_TABLES = {spec.name for spec in RAW_TABLES}


class DatasetHandoffState(Base):
    __tablename__ = "dataset_handoff_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    last_deleted_cutoff: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    latest_archive_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    latest_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    latest_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    prepared_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    download_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _state(session: Session) -> DatasetHandoffState:
    row = session.get(DatasetHandoffState, 1)
    if row is None:
        row = DatasetHandoffState(id=1)
        session.add(row)
        session.flush()
    return row


def _dataset_specs() -> list[tuple[str, Any, Any]]:
    specs: list[tuple[str, Any, Any]] = []
    seen: set[str] = set()
    for mapper in Base.registry.mappers:
        model = mapper.class_
        table_name = getattr(model, "__tablename__", None)
        if not table_name or table_name == DatasetHandoffState.__tablename__ or table_name in seen:
            continue
        chosen = None
        for name in TIME_COLUMNS:
            candidate = getattr(model, name, None)
            if candidate is not None:
                chosen = candidate
                break
        if chosen is None:
            continue
        specs.append((table_name, model, chosen))
        seen.add(table_name)
    return sorted(specs, key=lambda item: item[0])


def _earliest_time(session: Session, specs: list[tuple[str, Any, Any]]) -> datetime | None:
    values = []
    for _, model, time_column in specs:
        value = session.scalar(select(func.min(time_column)).select_from(model))
        aware = _aware(value)
        if aware:
            values.append(aware)
    return min(values) if values else None


def _json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        aware = _aware(value)
        return aware.isoformat() if aware else None
    return value


def _write_extra_table(
    session: Session,
    *,
    model: Any,
    time_column: Any,
    start: datetime,
    end: datetime,
    path: Path,
) -> dict[str, Any]:
    table = model.__table__
    query = select(model).where(time_column >= start, time_column < end).order_by(time_column)
    if model is Candle:
        query = query.where(Candle.source_name != "strategy_history_cache")
    rows = 0
    first = last = None
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in session.scalars(query).yield_per(1000):
            data = {column.name: _json_value(getattr(row, column.name)) for column in table.columns}
            handle.write(json.dumps(data, sort_keys=True, separators=(",", ":"), default=str))
            handle.write("\n")
            stamp = _aware(getattr(row, time_column.key))
            first = first or stamp
            last = stamp or last
            rows += 1
    session.expunge_all()
    return {
        "file": path.name,
        "rows": rows,
        "first_timestamp": first.isoformat() if first else None,
        "last_timestamp": last.isoformat() if last else None,
        "size_bytes": path.stat().st_size,
    }


def _overwrite_core_candles(
    session: Session,
    *,
    start: datetime,
    end: datetime,
    raw_folder: Path,
    core_manifest: dict[str, Any],
) -> None:
    path = raw_folder / "candles.csv.gz"
    columns = _columns_for(next(spec for spec in RAW_TABLES if spec.name == "candles"))
    query = (
        select(Candle)
        .where(
            Candle.open_time >= start,
            Candle.open_time < end,
            Candle.is_closed.is_(True),
            Candle.source_name != "strategy_history_cache",
        )
        .order_by(Candle.open_time)
    )
    rows = 0
    first = last = None
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in session.scalars(query).yield_per(1000):
            writer.writerow({column: _csv_safe(getattr(row, column)) for column in columns})
            stamp = _aware(row.open_time)
            first = first or stamp
            last = stamp or last
            rows += 1
    session.expunge_all()
    core_manifest.setdefault("row_counts", {})["candles"] = rows
    core_manifest.setdefault("file_sizes", {})["candles.csv.gz"] = path.stat().st_size
    core_manifest.setdefault("table_time_ranges", {})["candles"] = {
        "first_timestamp": first.isoformat() if first else None,
        "last_timestamp": last.isoformat() if last else None,
    }


def _zip_folder(source: Path, output: Path) -> None:
    if output.exists():
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source))


def prepare_handoff(session: Session) -> dict[str, Any]:
    HANDOFF_ROOT.mkdir(parents=True, exist_ok=True)
    state = _state(session)
    if state.latest_archive_id and state.deleted_at is None:
        path = HANDOFF_ROOT / state.latest_archive_id
        if path.exists():
            return {
                "status": "ready",
                "archive_id": state.latest_archive_id,
                "download_url": f"/api/data/handoff/download/{state.latest_archive_id}",
                "start": _aware(state.latest_start).isoformat() if state.latest_start else None,
                "end": _aware(state.latest_end).isoformat() if state.latest_end else None,
                "message": "Previous prepared archive is still waiting for download/delete.",
            }

    specs = _dataset_specs()
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = _aware(state.last_deleted_cutoff) or _earliest_time(session, specs)
    if start is None or start >= end:
        return {"status": "empty", "message": "No finished data is available yet."}

    stamp = end.strftime("%Y%m%d_%H%M")
    archive_id = f"raw_handoff_{stamp}.zip"
    work = HANDOFF_ROOT / archive_id.removesuffix(".zip")
    if work.exists():
        shutil.rmtree(work)
    raw_folder = work / "raw_data"
    raw_folder.mkdir(parents=True, exist_ok=True)

    core_manifest = _write_raw_folder(
        session,
        raw_folder,
        start=start,
        end=end,
        options={
            "include_market": True,
            "include_news": True,
            "include_external": True,
            "include_experience": True,
            "include_models": True,
            "symbols": None,
            "handoff": True,
        },
    )
    _overwrite_core_candles(session, start=start, end=end, raw_folder=raw_folder, core_manifest=core_manifest)

    extras: dict[str, Any] = {}
    for table_name, model, time_column in specs:
        if table_name in CORE_TABLES or table_name in KEEP_TABLES:
            continue
        extras[table_name] = _write_extra_table(
            session,
            model=model,
            time_column=time_column,
            start=start,
            end=end,
            path=raw_folder / f"{table_name}.jsonl.gz",
        )

    handoff_manifest = {
        "archive_id": archive_id,
        "format": "anata_raw_days_compatible_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "time_range": {
            "start_inclusive": start.isoformat(),
            "end_exclusive": end.isoformat(),
            "next_archive_start": end.isoformat(),
            "last_complete_1m_candle_open": (end - timedelta(minutes=1)).isoformat(),
        },
        "core_manifest": core_manifest,
        "extended_tables": extras,
        "preserved_operational_tables": sorted(KEEP_TABLES),
        "note": "Put this ZIP in raw_days. Core filenames remain compatible with the existing local training loader.",
    }
    (work / "handoff_manifest.json").write_text(json.dumps(handoff_manifest, indent=2), encoding="utf-8")
    archive_path = HANDOFF_ROOT / archive_id
    _zip_folder(work, archive_path)

    state.latest_archive_id = archive_id
    state.latest_start = start
    state.latest_end = end
    state.prepared_at = datetime.now(timezone.utc)
    state.download_requested_at = None
    state.deleted_at = None
    session.commit()
    return {
        "status": "ready",
        "archive_id": archive_id,
        "download_url": f"/api/data/handoff/download/{archive_id}",
        "size_bytes": archive_path.stat().st_size,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "next_start": end.isoformat(),
        "core_rows": core_manifest.get("row_counts", {}),
        "extended_rows": {name: info["rows"] for name, info in extras.items()},
    }


def mark_download_requested(session: Session, archive_id: str) -> Path:
    state = _state(session)
    if state.latest_archive_id != archive_id:
        raise FileNotFoundError(archive_id)
    path = HANDOFF_ROOT / Path(archive_id).name
    if path.name != archive_id or not path.exists():
        raise FileNotFoundError(archive_id)
    state.download_requested_at = datetime.now(timezone.utc)
    session.commit()
    return path


def _delete_order(specs: list[tuple[str, Any, Any]]) -> list[tuple[str, Any, Any]]:
    by_name = {spec[0]: spec for spec in specs}
    names = set(by_name)
    deps: dict[str, set[str]] = {name: set() for name in names}
    for name, model, _ in specs:
        for fk in model.__table__.foreign_keys:
            parent = fk.column.table.name
            if parent in names:
                deps[name].add(parent)
    ordered: list[str] = []
    remaining = set(names)
    while remaining:
        ready = sorted(name for name in remaining if not (deps[name] & remaining))
        if not ready:
            ready = [sorted(remaining)[0]]
        ordered.extend(ready)
        remaining.difference_update(ready)
    return [by_name[name] for name in reversed(ordered)]


def delete_latest_handoff(session: Session) -> dict[str, Any]:
    state = _state(session)
    if not state.latest_archive_id or state.deleted_at is not None:
        return {"status": "nothing_to_delete", "message": "No prepared downloaded dataset is waiting for deletion."}
    if state.download_requested_at is None:
        raise ValueError("Download the ZIP first. Deletion is locked until the download endpoint has been opened.")
    start, end = _aware(state.latest_start), _aware(state.latest_end)
    if start is None or end is None:
        raise ValueError("Prepared archive has no valid time range.")

    specs = [spec for spec in _dataset_specs() if spec[0] not in KEEP_TABLES]
    deleted: dict[str, int] = {}
    try:
        for table_name, model, time_column in _delete_order(specs):
            condition = [time_column >= start, time_column < end]
            if model is Candle:
                condition.append(Candle.source_name != "strategy_history_cache")
            count = session.execute(delete(model).where(*condition)).rowcount or 0
            deleted[table_name] = int(count)
        session.flush()
    except IntegrityError as exc:
        session.rollback()
        raise ValueError(
            "Nothing was deleted because one downloaded table is still referenced by operational state. "
            "The ZIP and handoff marker were kept so this can be fixed safely."
        ) from exc

    state = _state(session)
    state.last_deleted_cutoff = end
    state.deleted_at = datetime.now(timezone.utc)
    session.commit()

    archive_id = state.latest_archive_id
    archive_path = HANDOFF_ROOT / str(archive_id)
    work = HANDOFF_ROOT / str(archive_id).removesuffix(".zip")
    archive_path.unlink(missing_ok=True)
    if work.exists():
        shutil.rmtree(work)
    return {
        "status": "deleted",
        "archive_id": archive_id,
        "range": {"start": start.isoformat(), "end": end.isoformat()},
        "deleted_rows": deleted,
        "next_archive_start": end.isoformat(),
        "preserved_operational_tables": sorted(KEEP_TABLES),
    }


def handoff_status(session: Session) -> dict[str, Any]:
    state = _state(session)
    return {
        "latest_archive_id": state.latest_archive_id,
        "latest_start": _aware(state.latest_start).isoformat() if state.latest_start else None,
        "latest_end": _aware(state.latest_end).isoformat() if state.latest_end else None,
        "prepared_at": _aware(state.prepared_at).isoformat() if state.prepared_at else None,
        "download_requested_at": _aware(state.download_requested_at).isoformat() if state.download_requested_at else None,
        "deleted_at": _aware(state.deleted_at).isoformat() if state.deleted_at else None,
        "last_deleted_cutoff": _aware(state.last_deleted_cutoff).isoformat() if state.last_deleted_cutoff else None,
    }
