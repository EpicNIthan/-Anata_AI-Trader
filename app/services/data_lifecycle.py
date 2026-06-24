from __future__ import annotations

import asyncio
import csv
import gzip
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import (
    AccountEquity,
    Base,
    Candle,
    ExperienceRecord,
    ExternalDataEvent,
    Feature,
    LiveCandleUpdate,
    MarketTick,
    NewsArticle,
    NewsSentiment,
    TrainingFeature,
)
from app.db.session import SessionLocal

logger = logging.getLogger(__name__)

ARCHIVE_TABLES = {
    "candles": Candle,
    "features": Feature,
    "training_features": TrainingFeature,
    "experience_buffer": ExperienceRecord,
}

TABLE_TIME_COLUMNS = {
    "candles": Candle.open_time,
    "features": Feature.as_of,
    "training_features": TrainingFeature.as_of,
    "experience_buffer": ExperienceRecord.created_at,
}


@dataclass
class LifecycleState:
    running: bool = False
    interval_seconds: int = 86400
    last_run_at: str | None = None
    last_archive_path: str | None = None
    last_cleanup: dict[str, Any] | None = None
    last_error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cutoff(days: int = 0, hours: int = 0) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=days, hours=hours)


def _rowcount(result: Any) -> int:
    return int(getattr(result, "rowcount", 0) or 0)


def cleanup_old_data(session: Session) -> dict[str, Any]:
    live_cutoff = _cutoff(hours=settings.live_update_retention_hours)
    raw_news_cutoff = _cutoff(days=settings.raw_news_retention_days)
    raw_tick_cutoff = _cutoff(days=settings.raw_tick_retention_days)
    diagnostic_cutoff = _cutoff(days=settings.diagnostic_retention_days)
    experience_cutoff = _cutoff(days=settings.experience_retention_days)
    closed_candle_cutoff = _cutoff(days=settings.closed_candle_retention_days)

    deleted_live_updates = _rowcount(
        session.execute(delete(LiveCandleUpdate).where(LiveCandleUpdate.updated_at < live_cutoff))
    )
    deleted_legacy_live_candles = _rowcount(
        session.execute(delete(Candle).where(Candle.is_closed.is_(False), Candle.updated_at < live_cutoff))
    )
    deleted_raw_ticks = _rowcount(
        session.execute(delete(MarketTick).where(MarketTick.event_time < raw_tick_cutoff))
    )
    deleted_old_closed_candles = _rowcount(
        session.execute(delete(Candle).where(Candle.is_closed.is_(True), Candle.open_time < closed_candle_cutoff))
    )
    compacted_news_articles = _rowcount(
        session.execute(
            update(NewsArticle)
            .where(NewsArticle.created_at < raw_news_cutoff)
            .values(raw=None, raw_payload=None, raw_text=None)
        )
    )
    compacted_news_sentiment = _rowcount(
        session.execute(
            update(NewsSentiment)
            .where(NewsSentiment.created_at < raw_news_cutoff)
            .values(raw_payload=None)
        )
    )
    compacted_experiences = _rowcount(
        session.execute(
            update(ExperienceRecord)
            .where(ExperienceRecord.created_at < experience_cutoff)
            .values(market_state=None, news_state=None, feature_payload=None)
        )
    )
    deleted_account_equity = _rowcount(
        session.execute(delete(AccountEquity).where(AccountEquity.timestamp < diagnostic_cutoff))
    )
    deleted_external_events = _rowcount(
        session.execute(delete(ExternalDataEvent).where(ExternalDataEvent.event_time < diagnostic_cutoff))
    )
    session.commit()
    return {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "retention": retention_config(),
        "deleted": {
            "live_candle_updates": deleted_live_updates,
            "legacy_live_candles": deleted_legacy_live_candles,
            "market_ticks": deleted_raw_ticks,
            "old_closed_candles": deleted_old_closed_candles,
            "account_equity": deleted_account_equity,
            "external_data_events": deleted_external_events,
        },
        "compacted": {
            "news_articles_raw_fields": compacted_news_articles,
            "news_sentiment_raw_payload": compacted_news_sentiment,
            "experience_buffer_raw_context": compacted_experiences,
        },
    }


def retention_config() -> dict[str, Any]:
    return {
        "live_update_retention_hours": settings.live_update_retention_hours,
        "raw_news_retention_days": settings.raw_news_retention_days,
        "raw_tick_retention_days": settings.raw_tick_retention_days,
        "diagnostic_retention_days": settings.diagnostic_retention_days,
        "experience_retention_days": settings.experience_retention_days,
        "closed_candle_retention_days": settings.closed_candle_retention_days,
        "archive_dir": str(settings.archive_dir),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _archive_table(session: Session, table_name: str, before: datetime, archive_dir: Path) -> dict[str, Any]:
    model = ARCHIVE_TABLES[table_name]
    time_column = TABLE_TIME_COLUMNS[table_name]
    rows = list(session.scalars(select(model).where(time_column < before).order_by(time_column)))
    path = archive_dir / f"{table_name}_{before.strftime('%Y%m%d_%H%M%S')}.csv.gz"
    archive_dir.mkdir(parents=True, exist_ok=True)
    columns = [column.name for column in Base.metadata.tables[table_name].columns]
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _json_safe(getattr(row, column)) for column in columns})
    return {"table": table_name, "path": str(path), "rows": len(rows)}


def archive_old_data(
    session: Session,
    *,
    before: datetime | None = None,
    tables: list[str] | None = None,
    delete_after_archive: bool = False,
) -> dict[str, Any]:
    before = before or _cutoff(days=30)
    selected_tables = tables or ["candles", "training_features", "experience_buffer"]
    archive_dir = settings.archive_dir
    exports = []
    for table_name in selected_tables:
        if table_name not in ARCHIVE_TABLES:
            continue
        exports.append(_archive_table(session, table_name, before, archive_dir))

    deleted: dict[str, int] = {}
    if delete_after_archive:
        for item in exports:
            table_name = item["table"]
            if table_name == "features":
                deleted[table_name] = 0
                continue
            model = ARCHIVE_TABLES[table_name]
            time_column = TABLE_TIME_COLUMNS[table_name]
            if table_name == "candles":
                result = session.execute(delete(Candle).where(Candle.is_closed.is_(True), Candle.open_time < before))
            else:
                result = session.execute(delete(model).where(time_column < before))
            deleted[table_name] = _rowcount(result)
        session.commit()

    return {
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "before": before.isoformat(),
        "delete_after_archive": delete_after_archive,
        "exports": exports,
        "deleted": deleted,
    }


class DataLifecycleService:
    def __init__(self, interval_seconds: int = 86400) -> None:
        self.state = LifecycleState(interval_seconds=interval_seconds)
        self._task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None

    def status(self) -> dict[str, Any]:
        return {**self.state.as_dict(), "retention": retention_config()}

    async def start(self) -> dict[str, Any]:
        if self._task and not self._task.done():
            return self.status()
        self._stop_event = asyncio.Event()
        self.state.running = True
        self._task = asyncio.create_task(self._run_loop(self._stop_event), name="data-lifecycle")
        return self.status()

    async def stop(self) -> dict[str, Any]:
        if self._stop_event:
            self._stop_event.set()
        if self._task and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=5)
            except asyncio.TimeoutError:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
        self.state.running = False
        return self.status()

    async def _run_loop(self, stop_event: asyncio.Event) -> None:
        try:
            while not stop_event.is_set():
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=self.state.interval_seconds)
                    continue
                except asyncio.TimeoutError:
                    await asyncio.to_thread(self.run_cleanup_once)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive worker boundary
            logger.exception("Data lifecycle worker stopped after unexpected error")
            self.state.last_error = str(exc)
        finally:
            self.state.running = False

    def run_cleanup_once(self) -> dict[str, Any]:
        with SessionLocal() as session:
            result = cleanup_old_data(session)
        self.state.last_run_at = datetime.now(timezone.utc).isoformat()
        self.state.last_cleanup = result
        self.state.last_error = None
        return self.status()

    def archive_once(
        self,
        *,
        before: datetime | None = None,
        tables: list[str] | None = None,
        delete_after_archive: bool = False,
    ) -> dict[str, Any]:
        with SessionLocal() as session:
            result = archive_old_data(
                session,
                before=before,
                tables=tables,
                delete_after_archive=delete_after_archive,
            )
        first_path = result["exports"][0]["path"] if result.get("exports") else None
        self.state.last_archive_path = first_path
        return result
