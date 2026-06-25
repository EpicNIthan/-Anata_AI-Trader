from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.db.models import (
    AccountEquity,
    AiDecision,
    Candle,
    ExperienceRecord,
    ExternalDataEvent,
    Feature,
    LiveCandleUpdate,
    MarketTick,
    ModelVersion,
    NewsArticle,
    NewsSentiment,
    PaperTrade,
    Position,
    TrainingFeature,
    TrainingRun,
)
from app.services.raw_data_export import (
    FINISHED_DATA_ROOT,
    RAW_EXPORT_ROOT,
    RAW_TABLES_BY_NAME,
    RawTableSpec,
    _export_range,
)


# Delete children first so PostgreSQL foreign keys do not block cleanup.
DELETE_ORDER: list[str] = [
    "experience_buffer",
    "ai_decisions",
    "training_features",
    "features",
    "news_sentiment",
    "news_articles",
    "paper_trades",
    "positions",
    "account_equity",
    "external_data_events",
    "market_ticks",
    "live_candle_updates",
    "candles",
    "training_runs",
    "model_versions",
]


MODEL_DELETE_ORDER = [
    ExperienceRecord,
    AiDecision,
    TrainingFeature,
    Feature,
    NewsSentiment,
    NewsArticle,
    PaperTrade,
    Position,
    AccountEquity,
    ExternalDataEvent,
    MarketTick,
    LiveCandleUpdate,
    Candle,
    TrainingRun,
    ModelVersion,
]


def _normalize_symbols(value: Any) -> list[str] | None:
    if isinstance(value, str):
        symbols = [item.strip().upper() for item in value.split(",") if item.strip()]
        return symbols or None
    if isinstance(value, list):
        symbols = [str(item).strip().upper() for item in value if str(item).strip()]
        return symbols or None
    return None


def _safe_remove_path(path: Path, *, root: Path) -> bool:
    root = root.resolve()
    path = path.resolve()
    if root != path and root not in path.parents:
        raise ValueError(f"Refusing to delete path outside {root}: {path}")
    if path.is_dir():
        shutil.rmtree(path)
        return True
    if path.is_file():
        path.unlink()
        return True
    return False


def _delete_archive(archive_id: str | None) -> dict[str, Any]:
    if not archive_id:
        return {"deleted": False, "reason": "archive_id not provided"}
    safe = Path(archive_id).name
    if safe != archive_id or not safe.endswith(".zip"):
        raise ValueError("Invalid archive_id")
    archive_path = RAW_EXPORT_ROOT / safe
    removed_archive = _safe_remove_path(archive_path, root=RAW_EXPORT_ROOT) if archive_path.exists() else False
    work_dir = RAW_EXPORT_ROOT / safe.removesuffix(".zip")
    removed_work_dir = _safe_remove_path(work_dir, root=RAW_EXPORT_ROOT) if work_dir.exists() else False
    return {
        "archive_id": safe,
        "deleted_archive": removed_archive,
        "deleted_work_dir": removed_work_dir,
    }


def _delete_finished_data(payload: dict[str, Any]) -> dict[str, Any]:
    deleted: list[str] = []
    missing: list[str] = []
    if payload.get("delete_all_finished_data"):
        if FINISHED_DATA_ROOT.exists():
            for child in sorted(FINISHED_DATA_ROOT.iterdir()):
                if child.is_dir():
                    _safe_remove_path(child, root=FINISHED_DATA_ROOT)
                    deleted.append(str(child))
        return {"mode": "all", "deleted": deleted, "missing": missing}

    dates: list[str] = []
    for key in ("date", "day"):
        if payload.get(key):
            dates.append(str(payload[key])[:10])
    # If a date-range export covered one exact day, _export_range gives us the day folder name.
    try:
        _start, _end, day_name = _export_range(payload)
        if day_name:
            dates.append(day_name)
    except Exception:
        pass
    dates = sorted(set(dates))
    for day in dates:
        folder = FINISHED_DATA_ROOT / day
        if folder.exists():
            _safe_remove_path(folder, root=FINISHED_DATA_ROOT)
            deleted.append(str(folder))
        else:
            missing.append(str(folder))
    return {"mode": "selected", "deleted": deleted, "missing": missing}


def _delete_rows_for_spec(
    session: Session,
    spec: RawTableSpec,
    *,
    start: datetime | None,
    end: datetime | None,
    symbols: list[str] | None,
) -> int:
    stmt = delete(spec.model)
    if start is not None:
        stmt = stmt.where(spec.time_column >= start)
    if end is not None:
        stmt = stmt.where(spec.time_column < end)
    symbol_column = getattr(spec.model, "symbol", None)
    if symbols and symbol_column is not None:
        stmt = stmt.where(symbol_column.in_(symbols))
    result = session.execute(stmt)
    return int(result.rowcount or 0)


def delete_raw_database_rows(session: Session, payload: dict[str, Any]) -> dict[str, Any]:
    start, end, _day_name = _export_range(payload)
    symbols = _normalize_symbols(payload.get("symbols"))
    deleted_rows: dict[str, int] = {}
    for name in DELETE_ORDER:
        spec = RAW_TABLES_BY_NAME[name]
        deleted_rows[name] = _delete_rows_for_spec(session, spec, start=start, end=end, symbols=symbols)
    session.commit()
    return {
        "time_range": {
            "start": start.isoformat() if start else None,
            "end": end.isoformat() if end else None,
        },
        "symbols": symbols,
        "deleted_rows": deleted_rows,
        "total_deleted_rows": sum(deleted_rows.values()),
    }


def cleanup_downloaded_raw_data(session: Session, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Delete Railway-side raw export files and, optionally, matching DB rows.

    This is intended to be called by the PC downloader only after the ZIP is fully
    written and the manifest has been verified locally.
    """

    payload = payload or {}
    result: dict[str, Any] = {
        "status": "ok",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "deleted_archive": None,
        "deleted_finished_data": None,
        "deleted_database_rows": None,
    }

    if payload.get("delete_archive", True):
        result["deleted_archive"] = _delete_archive(payload.get("archive_id"))

    if payload.get("delete_finished_data", True):
        result["deleted_finished_data"] = _delete_finished_data(payload)

    if payload.get("delete_db_rows", False):
        result["deleted_database_rows"] = delete_raw_database_rows(session, payload)

    return result
