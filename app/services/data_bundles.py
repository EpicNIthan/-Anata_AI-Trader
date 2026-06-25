from __future__ import annotations

import csv
import gzip
import json
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.db.models import (
    AccountEquity,
    AiDecision,
    Base,
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

BUNDLE_ROOT = Path("datasets") / "daily_bundles"


@dataclass(frozen=True)
class BundleTable:
    name: str
    model: Any
    time_column: Any


BUNDLE_TABLES = [
    BundleTable("candles", Candle, Candle.open_time),
    BundleTable("news_articles", NewsArticle, NewsArticle.created_at),
    BundleTable("news_sentiment", NewsSentiment, NewsSentiment.created_at),
    BundleTable("external_data_events", ExternalDataEvent, ExternalDataEvent.event_time),
    BundleTable("features", Feature, Feature.as_of),
    BundleTable("training_features", TrainingFeature, TrainingFeature.as_of),
    BundleTable("experience_buffer", ExperienceRecord, ExperienceRecord.created_at),
    BundleTable("ai_decisions", AiDecision, AiDecision.created_at),
    BundleTable("paper_trades", PaperTrade, PaperTrade.created_at),
    BundleTable("account_equity", AccountEquity, AccountEquity.timestamp),
    BundleTable("positions", Position, Position.opened_at),
    BundleTable("model_versions", ModelVersion, ModelVersion.created_at),
    BundleTable("training_runs", TrainingRun, TrainingRun.started_at),
]
BUNDLE_TABLES_BY_NAME = {table.name: table for table in BUNDLE_TABLES}
DEFAULT_TRAINING_BUNDLE_TABLES = [
    "candles",
    "news_articles",
    "news_sentiment",
    "external_data_events",
    "training_features",
    "experience_buffer",
]
ALL_BUNDLE_TABLES = [table.name for table in BUNDLE_TABLES]


def _utc_day(value: datetime) -> datetime:
    aware = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.combine(aware.date(), time.min, tzinfo=timezone.utc)


def _now_day() -> datetime:
    return _utc_day(datetime.now(timezone.utc))


def _dt(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return value


def _safe_name(value: str) -> str:
    safe = Path(value).name
    if safe != value or not safe:
        raise ValueError("invalid bundle id")
    return safe


def _bundle_dir(day: datetime, status: str) -> Path:
    return BUNDLE_ROOT / f"{day.date().isoformat()}_{status}"


def _bundle_zip_path(day: datetime, status: str) -> Path:
    return BUNDLE_ROOT / f"{day.date().isoformat()}_{status}.zip"


def _date_range(
    session: Session,
    *,
    selected_tables: list[BundleTable],
    since_date: datetime | None = None,
    days: int | None = None,
) -> list[datetime]:
    current_day = _now_day()
    if since_date:
        start_day = _utc_day(since_date)
    else:
        earliest_values = []
        for table in selected_tables:
            if table.time_column is None:
                continue
            value = session.scalar(select(func.min(table.time_column)))
            if value:
                earliest_values.append(value if value.tzinfo else value.replace(tzinfo=timezone.utc))
        start_day = _utc_day(min(earliest_values)) if earliest_values else current_day
    if days is not None:
        start_day = max(start_day, current_day - timedelta(days=max(days - 1, 0)))
    total_days = max((current_day - start_day).days, 0) + 1
    return [start_day + timedelta(days=index) for index in range(total_days)]


def _query_for_day(table: BundleTable, start: datetime, end: datetime) -> Any:
    if table.model is None or table.time_column is None:
        return None
    query = select(table.model).where(table.time_column >= start, table.time_column < end).order_by(table.time_column)
    if table.model is Candle:
        query = query.where(Candle.is_closed.is_(True))
    return query


def _write_table(session: Session, path: Path, table: BundleTable, start: datetime, end: datetime) -> dict[str, Any]:
    table_name = table.name
    columns = [column.name for column in Base.metadata.tables[table_name].columns]
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        query = _query_for_day(table, start, end)
        if query is None:
            return {"file": path.name, "rows": 0, "note": "table has no daily time column"}
        for row in session.scalars(query).yield_per(1000):
            writer.writerow({column: _json_safe(getattr(row, column)) for column in columns})
            rows_written += 1
    session.expunge_all()
    return {"file": path.name, "rows": rows_written}


def _zip_dir(source_dir: Path, output_path: Path) -> None:
    if output_path.exists():
        output_path.unlink()
    # The member CSV files are already gzip-compressed, so ZIP_STORED saves Railway CPU/memory.
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source_dir.parent))


def _selected_tables(tables: list[str] | None = None) -> list[BundleTable]:
    names = tables or DEFAULT_TRAINING_BUNDLE_TABLES
    if any(str(name).strip().lower() in {"all", "*", "database"} for name in names):
        names = ALL_BUNDLE_TABLES
    if any(str(name).strip().lower() in {"training", "useful", "default"} for name in names):
        names = DEFAULT_TRAINING_BUNDLE_TABLES
    selected = []
    for name in names:
        table = BUNDLE_TABLES_BY_NAME.get(str(name).strip())
        if table is not None:
            selected.append(table)
    return selected or [BUNDLE_TABLES_BY_NAME[name] for name in DEFAULT_TRAINING_BUNDLE_TABLES]


def build_daily_bundles(
    session: Session,
    *,
    since_date: datetime | None = None,
    days: int | None = None,
    include_unfinished: bool = True,
    tables: list[str] | None = None,
) -> dict[str, Any]:
    BUNDLE_ROOT.mkdir(parents=True, exist_ok=True)
    current_day = _now_day()
    selected_tables = _selected_tables(tables)
    bundles = []
    for day in _date_range(session, selected_tables=selected_tables, since_date=since_date, days=days):
        status = "unfinished" if day >= current_day else "finished"
        if status == "unfinished" and not include_unfinished:
            continue
        end = day + timedelta(days=1)
        bundle_dir = _bundle_dir(day, status)
        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)
        bundle_dir.mkdir(parents=True, exist_ok=True)
        table_exports = {}
        total_rows = 0
        for table in selected_tables:
            table_exports[table.name] = _write_table(session, bundle_dir / f"{table.name}.csv.gz", table, day, end)
            total_rows += int(table_exports[table.name]["rows"])
        manifest = {
            "bundle_id": f"{day.date().isoformat()}_{status}.zip",
            "day": day.date().isoformat(),
            "status": status,
            "start": day.isoformat(),
            "end": end.isoformat(),
            "total_rows": total_rows,
            "tables": table_exports,
            "table_names": [table.name for table in selected_tables],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        zip_path = _bundle_zip_path(day, status)
        _zip_dir(bundle_dir, zip_path)
        bundles.append(manifest | {"path": str(zip_path), "size_bytes": zip_path.stat().st_size})
    return {"status": "ok", "bundle_root": str(BUNDLE_ROOT), "bundles": bundles}


def list_daily_bundles() -> dict[str, Any]:
    BUNDLE_ROOT.mkdir(parents=True, exist_ok=True)
    bundles = []
    for zip_path in sorted(BUNDLE_ROOT.glob("*.zip")):
        parts = zip_path.stem.rsplit("_", 1)
        day = parts[0] if parts else zip_path.stem
        status = parts[1] if len(parts) > 1 else "unknown"
        manifest_path = BUNDLE_ROOT / zip_path.stem / "manifest.json"
        manifest = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                manifest = {}
        bundles.append(
            {
                "bundle_id": zip_path.name,
                "day": manifest.get("day") or day,
                "status": manifest.get("status") or status,
                "total_rows": manifest.get("total_rows"),
                "size_bytes": zip_path.stat().st_size,
                "created_at": manifest.get("created_at"),
                "download_url": f"/api/data/bundles/download/{zip_path.name}",
            }
        )
    return {"bundle_root": str(BUNDLE_ROOT), "bundles": bundles}


def bundle_download_path(bundle_id: str) -> Path:
    safe = _safe_name(bundle_id)
    path = BUNDLE_ROOT / safe
    if not path.exists() or not path.is_file() or path.suffix.lower() != ".zip":
        raise FileNotFoundError(safe)
    return path


def cleanup_finished_after_download(session: Session, *, keep_unfinished: bool = True) -> dict[str, Any]:
    cutoff = _now_day() if keep_unfinished else datetime.now(timezone.utc)
    old_news_article_ids = select(NewsArticle.id).where(NewsArticle.created_at < cutoff)
    deleted: dict[str, int] = {}
    deleted["experience_buffer"] = session.execute(delete(ExperienceRecord).where(ExperienceRecord.created_at < cutoff)).rowcount or 0
    deleted["training_features"] = session.execute(delete(TrainingFeature).where(TrainingFeature.as_of < cutoff)).rowcount or 0
    deleted["ai_decisions"] = session.execute(delete(AiDecision).where(AiDecision.created_at < cutoff)).rowcount or 0
    deleted["paper_trades"] = session.execute(delete(PaperTrade).where(PaperTrade.created_at < cutoff)).rowcount or 0
    deleted["account_equity"] = session.execute(delete(AccountEquity).where(AccountEquity.timestamp < cutoff)).rowcount or 0
    deleted["news_sentiment"] = session.execute(delete(NewsSentiment).where(NewsSentiment.article_id.in_(old_news_article_ids))).rowcount or 0
    deleted["news_articles"] = session.execute(delete(NewsArticle).where(NewsArticle.created_at < cutoff)).rowcount or 0
    deleted["external_data_events"] = session.execute(delete(ExternalDataEvent).where(ExternalDataEvent.event_time < cutoff)).rowcount or 0
    deleted["features"] = session.execute(delete(Feature).where(Feature.as_of < cutoff)).rowcount or 0
    deleted["candles"] = session.execute(delete(Candle).where(Candle.open_time < cutoff)).rowcount or 0
    deleted["live_candle_updates"] = session.execute(delete(LiveCandleUpdate).where(LiveCandleUpdate.updated_at < cutoff)).rowcount or 0
    deleted["market_ticks"] = session.execute(delete(MarketTick).where(MarketTick.event_time < cutoff)).rowcount or 0
    session.commit()

    deleted_bundles = []
    for zip_path in sorted(BUNDLE_ROOT.glob("*_finished.zip")):
        deleted_bundles.append(zip_path.name)
        zip_path.unlink(missing_ok=True)
        folder = BUNDLE_ROOT / zip_path.stem
        if folder.exists():
            shutil.rmtree(folder)
    return {
        "status": "ok",
        "cutoff": cutoff.isoformat(),
        "kept_unfinished_day": keep_unfinished,
        "deleted_rows": deleted,
        "deleted_finished_bundles": deleted_bundles,
        "message": "Finished Railway data was deleted after local download. Current unfinished day remains collecting.",
    }
