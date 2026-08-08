from __future__ import annotations

import csv
import gzip
import io
import json
import queue
import threading
import zipfile
from datetime import datetime, timezone
from typing import Any, Iterator

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Base, Candle
from app.db.session import SessionLocal
from app.services import dataset_handoff as handoff
from app.services.raw_data_export import RAW_TABLES, _columns_for, _csv_safe, _row_dict

_CHUNK = 1024 * 1024
_END = object()


class _QueueWriter(io.RawIOBase):
    """Unseekable file object that lets zipfile emit bytes while rows are still being read."""

    def __init__(self, output: queue.Queue, cancelled: threading.Event) -> None:
        super().__init__()
        self.output = output
        self.cancelled = cancelled
        self.position = 0

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def tell(self) -> int:
        return self.position

    def write(self, data: bytes | bytearray | memoryview) -> int:
        if self.cancelled.is_set():
            raise BrokenPipeError("dataset download cancelled")
        payload = bytes(data)
        if not payload:
            return 0
        while not self.cancelled.is_set():
            try:
                self.output.put(payload, timeout=0.5)
                self.position += len(payload)
                return len(payload)
            except queue.Full:
                continue
        raise BrokenPipeError("dataset download cancelled")

    def flush(self) -> None:
        return None


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        converted = _aware(value)
        return converted.isoformat() if converted else None
    return value


def _earliest_time(session: Session) -> datetime | None:
    values: list[datetime] = []
    for _, model, time_column in handoff._dataset_specs():
        value = session.scalar(select(func.min(time_column)).select_from(model))
        converted = _aware(value)
        if converted is not None:
            values.append(converted)
    return min(values) if values else None


def prepare_stream_plan(session: Session) -> dict[str, Any]:
    """Freeze one exact DB time range without building a temporary ZIP on Railway disk."""

    state = handoff._state(session)
    if state.latest_archive_id and state.deleted_at is None and state.latest_start and state.latest_end:
        # Reuse the same frozen range after an interrupted/failed browser download.
        state.download_requested_at = None
        session.commit()
        return {
            "status": "ready",
            "archive_id": state.latest_archive_id,
            "start": _aware(state.latest_start).isoformat(),
            "end": _aware(state.latest_end).isoformat(),
            "download_url": f"/api/data/handoff/download/{state.latest_archive_id}",
            "streaming": True,
        }

    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = _aware(state.last_deleted_cutoff) or _earliest_time(session)
    if start is None or start >= end:
        return {"status": "empty", "message": "No finished data is available yet."}

    archive_id = f"raw_handoff_{end.strftime('%Y%m%d_%H%M')}.zip"
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
        "start": start.isoformat(),
        "end": end.isoformat(),
        "next_start": end.isoformat(),
        "download_url": f"/api/data/handoff/download/{archive_id}",
        "streaming": True,
    }


def handoff_download_status(session: Session) -> dict[str, Any]:
    data = handoff.handoff_status(session)
    completed = data.get("download_requested_at")
    data["download_completed_at"] = completed
    data["download_ready_to_delete"] = bool(completed and not data.get("deleted_at"))
    data["delivery_mode"] = "direct_db_stream"
    return data


def _write_core_csv(
    archive: zipfile.ZipFile,
    *,
    spec: Any,
    session: Session,
    start: datetime,
    end: datetime,
) -> dict[str, Any]:
    path = f"raw_data/{spec.name}.csv.gz"
    query = select(spec.model).where(spec.time_column >= start, spec.time_column < end).order_by(spec.time_column)
    if spec.model is Candle:
        query = query.where(Candle.is_closed.is_(True), Candle.source_name != "strategy_history_cache")

    columns = _columns_for(spec)
    count = 0
    first = last = None
    with archive.open(path, "w", force_zip64=True) as member:
        with gzip.GzipFile(fileobj=member, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="", write_through=True) as text:
                writer = csv.DictWriter(text, fieldnames=columns)
                writer.writeheader()
                for row in session.scalars(query).yield_per(500):
                    writer.writerow({column: _csv_safe(getattr(row, column)) for column in columns})
                    stamp = _aware(getattr(row, spec.time_column.key))
                    first = first or stamp
                    last = stamp or last
                    count += 1
    session.expunge_all()
    return {
        "file": path,
        "rows": count,
        "first_timestamp": first.isoformat() if first else None,
        "last_timestamp": last.isoformat() if last else None,
    }


def _write_core_jsonl(
    archive: zipfile.ZipFile,
    *,
    spec: Any,
    session: Session,
    start: datetime,
    end: datetime,
) -> dict[str, Any]:
    path = f"raw_data/{spec.name}.jsonl.gz"
    query = select(spec.model).where(spec.time_column >= start, spec.time_column < end).order_by(spec.time_column)
    count = 0
    first = last = None
    with archive.open(path, "w", force_zip64=True) as member:
        with gzip.GzipFile(fileobj=member, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", write_through=True) as text:
                for row in session.scalars(query).yield_per(500):
                    text.write(json.dumps(_row_dict(row, session=session), sort_keys=True, separators=(",", ":"), default=str))
                    text.write("\n")
                    stamp = _aware(getattr(row, spec.time_column.key))
                    first = first or stamp
                    last = stamp or last
                    count += 1
    session.expunge_all()
    return {
        "file": path,
        "rows": count,
        "first_timestamp": first.isoformat() if first else None,
        "last_timestamp": last.isoformat() if last else None,
    }


def _write_extended_jsonl(
    archive: zipfile.ZipFile,
    *,
    table_name: str,
    model: Any,
    time_column: Any,
    session: Session,
    start: datetime,
    end: datetime,
) -> dict[str, Any]:
    path = f"raw_data/{table_name}.jsonl.gz"
    query = select(model).where(time_column >= start, time_column < end).order_by(time_column)
    count = 0
    first = last = None
    with archive.open(path, "w", force_zip64=True) as member:
        with gzip.GzipFile(fileobj=member, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", write_through=True) as text:
                for row in session.scalars(query).yield_per(500):
                    payload = {column.name: _json_value(getattr(row, column.name)) for column in model.__table__.columns}
                    text.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str))
                    text.write("\n")
                    stamp = _aware(getattr(row, time_column.key))
                    first = first or stamp
                    last = stamp or last
                    count += 1
    session.expunge_all()
    return {
        "file": path,
        "rows": count,
        "first_timestamp": first.isoformat() if first else None,
        "last_timestamp": last.isoformat() if last else None,
    }


def _producer(
    archive_id: str,
    output: queue.Queue,
    cancelled: threading.Event,
) -> None:
    error: BaseException | None = None
    try:
        with SessionLocal() as session:
            state = handoff._state(session)
            if state.latest_archive_id != archive_id or state.deleted_at is not None:
                raise FileNotFoundError(archive_id)
            start = _aware(state.latest_start)
            end = _aware(state.latest_end)
            if start is None or end is None:
                raise RuntimeError("download range is missing")

            writer = _QueueWriter(output, cancelled)
            with zipfile.ZipFile(writer, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                table_results: dict[str, Any] = {}
                core_names = {spec.name for spec in RAW_TABLES}

                for spec in RAW_TABLES:
                    if cancelled.is_set():
                        raise BrokenPipeError("dataset download cancelled")
                    if spec.file_format == "csv":
                        table_results[spec.name] = _write_core_csv(
                            archive, spec=spec, session=session, start=start, end=end
                        )
                    else:
                        table_results[spec.name] = _write_core_jsonl(
                            archive, spec=spec, session=session, start=start, end=end
                        )

                for table_name, model, time_column in handoff._dataset_specs():
                    if table_name in core_names or table_name in handoff.KEEP_TABLES:
                        continue
                    if cancelled.is_set():
                        raise BrokenPipeError("dataset download cancelled")
                    table_results[table_name] = _write_extended_jsonl(
                        archive,
                        table_name=table_name,
                        model=model,
                        time_column=time_column,
                        session=session,
                        start=start,
                        end=end,
                    )

                manifest = {
                    "archive_id": archive_id,
                    "format": "anata_raw_days_compatible_v3_stream",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "time_range": {
                        "start_inclusive": start.isoformat(),
                        "end_exclusive": end.isoformat(),
                        "next_archive_start": end.isoformat(),
                    },
                    "tables": table_results,
                    "preserved_operational_tables": sorted(handoff.KEEP_TABLES),
                    "note": "Put this ZIP directly in raw_days. Core filenames remain compatible with the existing local training loader.",
                }
                archive.writestr("handoff_manifest.json", json.dumps(manifest, indent=2, default=str).encode("utf-8"))
    except BaseException as exc:  # passed back to the response generator
        error = exc
    finally:
        while True:
            try:
                output.put(("error", error) if error else _END, timeout=0.5)
                break
            except queue.Full:
                if cancelled.is_set():
                    break


def stream_dataset_archive(archive_id: str) -> Iterator[bytes]:
    """Stream a ZIP as it is built so Railway never waits for a huge temporary archive before sending bytes."""

    output: queue.Queue = queue.Queue(maxsize=8)
    cancelled = threading.Event()
    worker = threading.Thread(target=_producer, args=(archive_id, output, cancelled), daemon=True)
    worker.start()
    completed = False
    try:
        while True:
            item = output.get()
            if item is _END:
                completed = True
                break
            if isinstance(item, tuple) and len(item) == 2 and item[0] == "error":
                error = item[1]
                if isinstance(error, BrokenPipeError):
                    return
                raise RuntimeError(f"dataset stream failed: {type(error).__name__}: {error}") from error
            yield item
    finally:
        cancelled.set()
        worker.join(timeout=2)
        if completed:
            with SessionLocal() as session:
                state = handoff._state(session)
                if state.latest_archive_id == archive_id and state.deleted_at is None:
                    # Existing delete logic treats this as the completed-transfer marker.
                    state.download_requested_at = datetime.now(timezone.utc)
                    session.commit()
