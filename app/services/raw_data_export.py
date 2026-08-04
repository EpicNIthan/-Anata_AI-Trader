from __future__ import annotations

import csv
import gzip
import hashlib
import json
import shutil
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
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

FINISHED_DATA_ROOT = Path("finished_data")
RAW_EXPORT_ROOT = Path("datasets") / "raw_exports"


@dataclass(frozen=True)
class RawTableSpec:
    name: str
    model: Any
    time_column: Any
    file_format: str
    group: str


RAW_TABLES: list[RawTableSpec] = [
    RawTableSpec("candles", Candle, Candle.open_time, "csv", "market"),
    RawTableSpec("live_candle_updates", LiveCandleUpdate, LiveCandleUpdate.open_time, "csv", "market"),
    RawTableSpec("market_ticks", MarketTick, MarketTick.event_time, "csv", "market"),
    RawTableSpec("news_articles", NewsArticle, NewsArticle.created_at, "jsonl", "news"),
    RawTableSpec("news_sentiment", NewsSentiment, NewsSentiment.created_at, "csv", "news"),
    RawTableSpec("external_data_events", ExternalDataEvent, ExternalDataEvent.event_time, "jsonl", "external"),
    RawTableSpec("features", Feature, Feature.as_of, "jsonl", "experience"),
    RawTableSpec("training_features", TrainingFeature, TrainingFeature.as_of, "jsonl", "experience"),
    RawTableSpec("paper_trades", PaperTrade, PaperTrade.created_at, "csv", "experience"),
    RawTableSpec("positions", Position, Position.opened_at, "csv", "experience"),
    RawTableSpec("ai_decisions", AiDecision, AiDecision.created_at, "jsonl", "experience"),
    RawTableSpec("experience_buffer", ExperienceRecord, ExperienceRecord.created_at, "jsonl", "experience"),
    RawTableSpec("account_equity", AccountEquity, AccountEquity.timestamp, "csv", "experience"),
    RawTableSpec("model_versions", ModelVersion, ModelVersion.created_at, "jsonl", "models"),
    RawTableSpec("training_runs", TrainingRun, TrainingRun.started_at, "jsonl", "models"),
]
RAW_TABLES_BY_NAME = {table.name: table for table in RAW_TABLES}


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _sha256_file(path: Path) -> str:
    """Return the digest of an export file after its writer has closed it."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _actual_time_range(
    session: Session,
    spec: RawTableSpec,
    *,
    start: datetime | None,
    end: datetime | None,
    symbols: list[str] | None,
) -> dict[str, str | None]:
    """Return the actual first/last exported timestamps, not requested bounds."""

    statement = select(func.min(spec.time_column), func.max(spec.time_column)).select_from(spec.model)
    if start is not None:
        statement = statement.where(spec.time_column >= start)
    if end is not None:
        statement = statement.where(spec.time_column < end)
    if spec.model is Candle:
        statement = statement.where(Candle.is_closed.is_(True))
    symbol_column = getattr(spec.model, "symbol", None)
    if symbols and symbol_column is not None:
        statement = statement.where(symbol_column.in_([symbol.upper() for symbol in symbols]))
    first, last = session.execute(statement).one()
    first_utc = _as_utc(first)
    last_utc = _as_utc(last)
    return {
        "first_timestamp": first_utc.isoformat() if first_utc else None,
        "last_timestamp": last_utc.isoformat() if last_utc else None,
    }


def _parse_dt(value: Any, *, until: bool = False) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return _as_utc(value)
    text = str(value).strip()
    if not text:
        return None
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        parsed_date = date.fromisoformat(text)
        parsed_time = time.max if until else time.min
        parsed = datetime.combine(parsed_date, parsed_time, tzinfo=timezone.utc)
        return parsed + timedelta(microseconds=1) if until else parsed
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return _as_utc(parsed)


def _day_bounds(day: date | str | datetime | None = None) -> tuple[datetime, datetime]:
    if day is None:
        selected = datetime.now(timezone.utc).date() - timedelta(days=1)
    elif isinstance(day, datetime):
        selected = _as_utc(day).date()  # type: ignore[union-attr]
    elif isinstance(day, date):
        selected = day
    else:
        selected = date.fromisoformat(str(day)[:10])
    start = datetime.combine(selected, time.min, tzinfo=timezone.utc)
    return start, start + timedelta(days=1)


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return _as_utc(value).isoformat() if _as_utc(value) else None
    if isinstance(value, (dict, list)):
        return value
    return value


def _csv_safe(value: Any) -> Any:
    value = _json_safe(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return value


def _row_dict(row: Any, *, session: Session | None = None) -> dict[str, Any]:
    table = Base.metadata.tables[row.__tablename__]
    data = {column.name: _json_safe(getattr(row, column.name)) for column in table.columns}
    if isinstance(row, NewsArticle) and session is not None:
        sentiment = session.scalar(select(NewsSentiment).where(NewsSentiment.article_id == row.id).limit(1))
        data["article_id"] = row.id
        data["provider"] = row.source_name or row.source
        data["affected_symbols"] = sentiment.affected_symbols if sentiment else []
        data["sentiment_model_name"] = sentiment.model_name if sentiment else None
        data["sentiment_score"] = sentiment.sentiment_score if sentiment else None
        data["risk_score"] = sentiment.risk_score if sentiment else None
        data["sentiment_topics"] = sentiment.topics if sentiment else []
    return data


def _columns_for(spec: RawTableSpec) -> list[str]:
    return [column.name for column in Base.metadata.tables[spec.name].columns]


def _query_for_spec(
    spec: RawTableSpec,
    *,
    start: datetime | None,
    end: datetime | None,
    symbols: list[str] | None,
) -> Any:
    query = select(spec.model)
    if start is not None:
        query = query.where(spec.time_column >= start)
    if end is not None:
        query = query.where(spec.time_column < end)
    if spec.model is Candle:
        query = query.where(Candle.is_closed.is_(True))
    symbol_column = getattr(spec.model, "symbol", None)
    if symbols and symbol_column is not None:
        query = query.where(symbol_column.in_([symbol.upper() for symbol in symbols]))
    return query.order_by(spec.time_column)


def _write_csv(
    session: Session,
    spec: RawTableSpec,
    path: Path,
    *,
    start: datetime | None,
    end: datetime | None,
    symbols: list[str] | None,
) -> int:
    rows_written = 0
    columns = _columns_for(spec)
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in session.scalars(_query_for_spec(spec, start=start, end=end, symbols=symbols)).yield_per(1000):
            writer.writerow({column: _csv_safe(getattr(row, column)) for column in columns})
            rows_written += 1
    session.expunge_all()
    return rows_written


def _write_jsonl(
    session: Session,
    spec: RawTableSpec,
    path: Path,
    *,
    start: datetime | None,
    end: datetime | None,
    symbols: list[str] | None,
) -> int:
    rows_written = 0
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in session.scalars(_query_for_spec(spec, start=start, end=end, symbols=symbols)).yield_per(1000):
            handle.write(json.dumps(_row_dict(row, session=session), sort_keys=True, separators=(",", ":"), default=str))
            handle.write("\n")
            rows_written += 1
    session.expunge_all()
    return rows_written


def _symbol_values(session: Session, start: datetime | None, end: datetime | None, symbols: list[str] | None) -> list[str]:
    if symbols:
        return sorted({symbol.upper() for symbol in symbols})
    query = select(Candle.symbol).where(Candle.is_closed.is_(True))
    if start is not None:
        query = query.where(Candle.open_time >= start)
    if end is not None:
        query = query.where(Candle.open_time < end)
    return sorted({value for value in session.scalars(query).all() if value})


def _select_tables(options: dict[str, Any] | None = None) -> list[RawTableSpec]:
    options = options or {}
    requested_tables = options.get("tables") or options.get("include_tables")
    if requested_tables:
        if isinstance(requested_tables, str):
            table_names = [item.strip() for item in requested_tables.split(",") if item.strip()]
        elif isinstance(requested_tables, list):
            table_names = [str(item).strip() for item in requested_tables if str(item).strip()]
        else:
            raise ValueError("tables must be a comma-separated string or list.")
        unknown = [name for name in table_names if name not in RAW_TABLES_BY_NAME]
        if unknown:
            raise ValueError(f"Unknown raw export table(s): {', '.join(unknown)}")
        return [RAW_TABLES_BY_NAME[name] for name in table_names]

    if options.get("news_only"):
        return [RAW_TABLES_BY_NAME["news_articles"], RAW_TABLES_BY_NAME["news_sentiment"]]

    include_market = bool(options.get("include_market", True))
    include_news = bool(options.get("include_news", True))
    include_external = bool(options.get("include_external", True))
    include_experience = bool(options.get("include_experience", True))
    include_models = bool(options.get("include_models", True))
    groups = {
        "market": include_market,
        "news": include_news,
        "external": include_external,
        "experience": include_experience,
        "models": include_models,
    }
    return [spec for spec in RAW_TABLES if groups.get(spec.group, False)]


def _write_raw_folder(
    session: Session,
    folder: Path,
    *,
    start: datetime | None,
    end: datetime | None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    options = options or {}
    folder.mkdir(parents=True, exist_ok=True)
    selected_tables = _select_tables(options)
    symbols = options.get("symbols")
    if isinstance(symbols, str):
        symbols = [item.strip().upper() for item in symbols.split(",") if item.strip()]
    elif isinstance(symbols, list):
        symbols = [str(item).upper() for item in symbols if str(item)]
    else:
        symbols = None

    row_counts: dict[str, int] = {}
    tables_exported: dict[str, str] = {}
    file_sizes: dict[str, int] = {}
    file_checksums: dict[str, str] = {}
    table_time_ranges: dict[str, dict[str, str | None]] = {}
    warnings: list[str] = []

    for spec in selected_tables:
        suffix = "jsonl.gz" if spec.file_format == "jsonl" else "csv.gz"
        path = folder / f"{spec.name}.{suffix}"
        if spec.file_format == "jsonl":
            rows_written = _write_jsonl(session, spec, path, start=start, end=end, symbols=symbols)
        else:
            rows_written = _write_csv(session, spec, path, start=start, end=end, symbols=symbols)
        row_counts[spec.name] = rows_written
        tables_exported[spec.name] = path.name
        file_sizes[path.name] = path.stat().st_size
        file_checksums[path.name] = _sha256_file(path)
        table_time_ranges[spec.name] = _actual_time_range(
            session,
            spec,
            start=start,
            end=end,
            symbols=symbols,
        )

    if row_counts.get("news_articles", 0) == 0 and any(spec.name == "news_articles" for spec in selected_tables):
        warnings.append("No news articles exported for this range.")
    if row_counts.get("market_ticks", 0) == 0 and not settings.store_market_ticks:
        warnings.append("market_ticks is empty because STORE_MARKET_TICKS is false.")
    if any(spec.name == "news_articles" for spec in selected_tables):
        raw_text_count = session.scalar(
            select(func.count(NewsArticle.id)).where(
                NewsArticle.raw_text.is_not(None),
                *( [NewsArticle.created_at >= start] if start is not None else [] ),
                *( [NewsArticle.created_at < end] if end is not None else [] ),
            )
        ) or 0
        if raw_text_count == 0 and row_counts.get("news_articles", 0) > 0:
            warnings.append("News rows were exported, but raw_text is empty for this range.")

    manifest = {
        "date": start.date().isoformat() if start and end and (end - start) <= timedelta(days=1, seconds=1) else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tables_exported": tables_exported,
        "row_counts": row_counts,
        "file_sizes": file_sizes,
        "file_checksums_sha256": file_checksums,
        "table_time_ranges": table_time_ranges,
        "symbols": _symbol_values(session, start, end, symbols),
        "time_range": {
            "start": start.isoformat() if start else None,
            "end": end.isoformat() if end else None,
        },
        "options": options,
        "warnings": warnings,
        "verification": {
            "writers_closed": True,
            "manifest_complete": True,
            "news_count": int(row_counts.get("news_articles", 0)),
            "derivatives_count": int(row_counts.get("external_data_events", 0)),
        },
    }
    manifest_path = folder / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    file_sizes["manifest.json"] = manifest_path.stat().st_size
    return manifest


def finish_daily_raw_data(session: Session, *, day: date | str | datetime | None = None) -> dict[str, Any]:
    start, end = _day_bounds(day)
    folder = FINISHED_DATA_ROOT / start.date().isoformat()
    if folder.exists():
        shutil.rmtree(folder)
    manifest = _write_raw_folder(
        session,
        folder,
        start=start,
        end=end,
        options={
            "include_market": True,
            "include_news": True,
            "include_external": True,
            "include_experience": True,
            "include_models": True,
            "finished_data": True,
        },
    )
    return {"status": "ok", "folder": str(folder), "manifest": manifest}


def _zip_folder(source: Path, output: Path) -> None:
    if output.exists():
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source))


def _export_range(payload: dict[str, Any]) -> tuple[datetime | None, datetime | None, str | None]:
    if payload.get("date"):
        start, end = _day_bounds(str(payload["date"]))
        return start, end, start.date().isoformat()
    if payload.get("use_all_data"):
        return None, None, None
    start = _parse_dt(payload.get("since_date"))
    end = _parse_dt(payload.get("until_date"), until=True)
    if start is None and end is None:
        start, end = _day_bounds(datetime.now(timezone.utc).date())
    return start, end, start.date().isoformat() if start and end and (end - start) <= timedelta(days=1, seconds=1) else None


def create_raw_data_archive(session: Session, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = payload or {}
    start, end, day_name = _export_range(payload)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_id = f"raw_data_export_{timestamp}.zip"
    work_root = RAW_EXPORT_ROOT / archive_id.removesuffix(".zip")
    if work_root.exists():
        shutil.rmtree(work_root)
    if day_name:
        output_folder = work_root / "finished_data" / day_name
    else:
        output_folder = work_root / "raw_data"
    manifest = _write_raw_folder(session, output_folder, start=start, end=end, options=payload)
    archive_path = RAW_EXPORT_ROOT / archive_id
    _zip_folder(work_root, archive_path)
    return {
        "status": "ok",
        "archive_id": archive_id,
        "archive_path": str(archive_path),
        "download_url": f"/api/raw-data/download/{archive_id}",
        "size_bytes": archive_path.stat().st_size,
        "manifest": manifest,
    }


def raw_data_archive_path(archive_id: str) -> Path:
    safe = Path(archive_id).name
    if safe != archive_id or not safe.endswith(".zip"):
        raise ValueError("Invalid raw data archive id")
    path = RAW_EXPORT_ROOT / safe
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(safe)
    return path
