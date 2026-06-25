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
    AiDecision,
    Base,
    Candle,
    ExperienceRecord,
    ExternalDataEvent,
    Feature,
    LiveCandleUpdate,
    MarketTick,
    NewsArticle,
    NewsSentiment,
    PaperTrade,
    TrainingFeature,
)
from app.db.session import SessionLocal

logger = logging.getLogger(__name__)

ARCHIVE_TABLES = {
    "account_equity": AccountEquity,
    "ai_decisions": AiDecision,
    "candles": Candle,
    "external_data_events": ExternalDataEvent,
    "features": Feature,
    "live_candle_updates": LiveCandleUpdate,
    "market_ticks": MarketTick,
    "news_articles": NewsArticle,
    "news_sentiment": NewsSentiment,
    "paper_trades": PaperTrade,
    "training_features": TrainingFeature,
    "experience_buffer": ExperienceRecord,
}

TABLE_TIME_COLUMNS = {
    "account_equity": AccountEquity.timestamp,
    "ai_decisions": AiDecision.created_at,
    "candles": Candle.open_time,
    "external_data_events": ExternalDataEvent.event_time,
    "features": Feature.as_of,
    "live_candle_updates": LiveCandleUpdate.updated_at,
    "market_ticks": MarketTick.event_time,
    "news_articles": NewsArticle.created_at,
    "news_sentiment": NewsSentiment.created_at,
    "paper_trades": PaperTrade.created_at,
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


def _compact_payload(value: dict[str, Any] | None, *, keep_debug: bool = False) -> tuple[dict[str, Any] | None, bool]:
    if not isinstance(value, dict):
        return value, False
    payload = dict(value)
    changed = False
    values = dict(payload.get("values") or {})
    if "final_ai_input" in values:
        values.pop("final_ai_input", None)
        payload["values"] = values
        changed = True
    if not keep_debug:
        metadata = dict(payload.get("metadata") or {})
        for key in ("news_context", "derivatives_context", "debug", "raw_news_context"):
            if key in metadata:
                metadata.pop(key, None)
                changed = True
        payload["metadata"] = metadata
    return payload, changed


def _compact_training_features(session: Session, cutoff: datetime) -> int:
    rows = list(session.scalars(select(TrainingFeature)))
    compacted = 0
    for row in rows:
        changed = False
        values = dict(row.feature_values or {})
        if "final_ai_input" in values:
            values.pop("final_ai_input", None)
            row.feature_values = values
            changed = True
        payload, payload_changed = _compact_payload(row.payload, keep_debug=False)
        if payload_changed:
            row.payload = payload
            changed = True
        if changed:
            compacted += 1
    return compacted


def _compact_features(session: Session, cutoff: datetime) -> int:
    rows = list(session.scalars(select(Feature).where(Feature.as_of < cutoff)))
    compacted = 0
    for row in rows:
        changed = False
        if row.raw_payload is not None:
            row.raw_payload = None
            changed = True
        payload, payload_changed = _compact_payload(row.payload, keep_debug=False)
        if payload_changed:
            row.payload = payload
            changed = True
        if changed:
            compacted += 1
    return compacted


def _compact_account_equity(session: Session, cutoff: datetime) -> int:
    rows = list(session.scalars(select(AccountEquity).where(AccountEquity.timestamp < cutoff).order_by(AccountEquity.timestamp)))
    keep_buckets: set[str] = set()
    delete_ids: list[int] = []
    for row in rows:
        timestamp = row.timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        bucket = timestamp.strftime("%Y%m%d%H")
        if bucket in keep_buckets:
            delete_ids.append(row.id)
            continue
        keep_buckets.add(bucket)
    if not delete_ids:
        return 0
    return _rowcount(session.execute(delete(AccountEquity).where(AccountEquity.id.in_(delete_ids))))


def compact_database(
    session: Session,
    *,
    archive_before_delete: bool = False,
    archive_tables: list[str] | None = None,
    factory_mode: bool = False,
    keep_recent_days: int | None = None,
) -> dict[str, Any]:
    raw_payload_hours = 0 if factory_mode else settings.raw_payload_retention_hours
    live_update_hours = 0 if factory_mode else settings.live_update_retention_hours
    raw_news_text_days = 0 if factory_mode else settings.raw_news_text_retention_days
    raw_tick_days = 0 if factory_mode else settings.raw_tick_retention_days
    raw_external_days = keep_recent_days if factory_mode and keep_recent_days is not None else settings.raw_external_event_retention_days
    account_equity_days = min(1, keep_recent_days) if factory_mode and keep_recent_days is not None else settings.account_equity_retention_days
    external_data_days = keep_recent_days if factory_mode and keep_recent_days is not None else settings.external_data_retention_days
    closed_candle_days = keep_recent_days if factory_mode and keep_recent_days is not None else settings.keep_closed_candles_days
    training_feature_days = keep_recent_days if factory_mode and keep_recent_days is not None else settings.training_feature_retention_days
    experience_days = keep_recent_days if factory_mode and keep_recent_days is not None else settings.experience_retention_days
    raw_news_days = keep_recent_days if factory_mode and keep_recent_days is not None else settings.raw_news_retention_days

    raw_payload_cutoff = _cutoff(hours=raw_payload_hours)
    live_cutoff = _cutoff(hours=live_update_hours)
    raw_tick_cutoff = _cutoff(days=raw_tick_days)
    raw_external_event_cutoff = _cutoff(days=raw_external_days)
    raw_news_text_cutoff = _cutoff(days=raw_news_text_days)
    account_equity_cutoff = _cutoff(days=account_equity_days)
    external_data_cutoff = _cutoff(days=external_data_days)
    closed_candle_cutoff = _cutoff(days=closed_candle_days)
    training_feature_cutoff = _cutoff(days=training_feature_days)
    experience_cutoff = _cutoff(days=experience_days)
    raw_news_cutoff = _cutoff(days=raw_news_days)
    operational_cutoff = _cutoff(days=settings.operational_retention_days)

    archived: dict[str, Any] | None = None
    if archive_before_delete:
        archived = archive_old_data(
            session,
            before=raw_payload_cutoff,
            tables=archive_tables or [
                "live_candle_updates",
                "market_ticks",
                "news_articles",
                "external_data_events",
                "features",
                "experience_buffer",
            ],
            delete_after_archive=False,
        )

    deleted_live_updates = _rowcount(
        session.execute(delete(LiveCandleUpdate).where(LiveCandleUpdate.updated_at < live_cutoff))
    )
    deleted_legacy_live_candles = _rowcount(
        session.execute(delete(Candle).where(Candle.is_closed.is_(False), Candle.updated_at < live_cutoff))
    )
    if settings.store_market_ticks:
        deleted_raw_ticks = _rowcount(session.execute(delete(MarketTick).where(MarketTick.event_time < raw_tick_cutoff)))
    else:
        deleted_raw_ticks = _rowcount(session.execute(delete(MarketTick)))
    deleted_old_closed_candles = _rowcount(
        session.execute(delete(Candle).where(Candle.is_closed.is_(True), Candle.open_time < closed_candle_cutoff))
    )
    old_news_article_ids = select(NewsArticle.id).where(NewsArticle.created_at < raw_news_cutoff)
    deleted_old_news_sentiment = _rowcount(session.execute(delete(NewsSentiment).where(NewsSentiment.article_id.in_(old_news_article_ids))))
    deleted_old_news_articles = _rowcount(session.execute(delete(NewsArticle).where(NewsArticle.created_at < raw_news_cutoff)))
    deleted_old_training_features = _rowcount(
        session.execute(delete(TrainingFeature).where(TrainingFeature.as_of < training_feature_cutoff))
    )
    deleted_old_experiences = _rowcount(
        session.execute(delete(ExperienceRecord).where(ExperienceRecord.created_at < experience_cutoff))
    )
    compacted_candles = _rowcount(
        session.execute(
            update(Candle)
            .where(Candle.updated_at < raw_payload_cutoff)
            .values(raw=None, raw_payload=None)
        )
    )
    compacted_live_updates = _rowcount(
        session.execute(
            update(LiveCandleUpdate)
            .where(LiveCandleUpdate.updated_at < raw_payload_cutoff)
            .values(raw_payload=None)
        )
    )
    compacted_market_ticks = 0
    if settings.store_market_ticks:
        compacted_market_ticks = _rowcount(
            session.execute(
                update(MarketTick)
                .where(MarketTick.event_time < raw_payload_cutoff)
                .values(raw=None, raw_payload=None)
            )
        )
    compacted_news_articles = _rowcount(
        session.execute(
            update(NewsArticle)
            .where(NewsArticle.created_at < raw_payload_cutoff)
            .values(raw=None, raw_payload=None)
        )
    )
    compacted_news_text = _rowcount(
        session.execute(
            update(NewsArticle)
            .where(
                NewsArticle.created_at < raw_news_text_cutoff,
                NewsArticle.id.in_(select(NewsSentiment.article_id)),
            )
            .values(raw_text=None)
        )
    )
    compacted_news_sentiment = _rowcount(
        session.execute(
            update(NewsSentiment)
            .where(NewsSentiment.created_at < raw_payload_cutoff)
            .values(raw_payload=None)
        )
    )
    compacted_experiences = _rowcount(
        session.execute(
            update(ExperienceRecord)
            .where(ExperienceRecord.created_at < raw_payload_cutoff)
            .values(market_state=None, news_state=None, feature_payload=None, raw_payload=None)
        )
    )
    old_feature_ids = select(Feature.id).where(Feature.as_of < operational_cutoff)
    old_trade_ids = select(PaperTrade.id).where(PaperTrade.created_at < operational_cutoff)
    if settings.railway_data_factory_mode:
        detached_old_experience_decisions = _rowcount(
            session.execute(
                update(ExperienceRecord)
                .where(ExperienceRecord.created_at < operational_cutoff)
                .values(ai_decision_id=None)
            )
        )
        detached_old_decision_features = _rowcount(
            session.execute(
                update(AiDecision)
                .where(AiDecision.feature_id.in_(old_feature_ids))
                .values(feature_id=None)
            )
        )
        detached_old_decision_trades = _rowcount(
            session.execute(
                update(AiDecision)
                .where(AiDecision.trade_id.in_(old_trade_ids))
                .values(trade_id=None)
            )
        )
        detached_old_experience_features = _rowcount(
            session.execute(
                update(ExperienceRecord)
                .where(ExperienceRecord.feature_id.in_(old_feature_ids))
                .values(feature_id=None)
            )
        )
        detached_old_training_feature_sources = _rowcount(
            session.execute(
                update(TrainingFeature)
                .where(TrainingFeature.source_feature_id.in_(old_feature_ids))
                .values(source_feature_id=None)
            )
        )
    else:
        detached_old_experience_decisions = 0
        detached_old_decision_features = 0
        detached_old_decision_trades = 0
        detached_old_experience_features = 0
        detached_old_training_feature_sources = 0
    compacted_old_ai_decisions = _rowcount(
        session.execute(
            update(AiDecision)
            .where(AiDecision.created_at < raw_payload_cutoff)
            .values(market_state=None, news_state=None, raw=None, raw_payload=None)
        )
    )
    compacted_paper_trades = _rowcount(
        session.execute(
            update(PaperTrade)
            .where(PaperTrade.created_at < raw_payload_cutoff)
            .values(raw_payload=None)
        )
    )
    compacted_account_equity = _rowcount(
        session.execute(
            update(AccountEquity)
            .where(AccountEquity.timestamp < raw_payload_cutoff)
            .values(raw=None)
        )
    )
    deleted_account_equity = _compact_account_equity(session, account_equity_cutoff)
    compacted_external_events = _rowcount(
        session.execute(
            update(ExternalDataEvent)
            .where(ExternalDataEvent.event_time < raw_external_event_cutoff)
            .values(raw_payload=None)
        )
    )
    deleted_external_events = _rowcount(
        session.execute(delete(ExternalDataEvent).where(ExternalDataEvent.event_time < external_data_cutoff))
    )
    compacted_features = _compact_features(session, raw_payload_cutoff)
    compacted_training_features = _compact_training_features(session, training_feature_cutoff)
    if settings.railway_data_factory_mode:
        session.flush()
        session.expunge_all()
        deleted_old_ai_decisions = _rowcount(
            session.execute(
                delete(AiDecision)
                .where(AiDecision.created_at < operational_cutoff)
                .execution_options(synchronize_session=False)
            )
        )
        deleted_old_paper_trades = _rowcount(
            session.execute(
                delete(PaperTrade)
                .where(PaperTrade.created_at < operational_cutoff)
                .execution_options(synchronize_session=False)
            )
        )
        deleted_old_features = _rowcount(
            session.execute(
                delete(Feature)
                .where(Feature.as_of < operational_cutoff)
                .execution_options(synchronize_session=False)
            )
        )
    else:
        deleted_old_ai_decisions = 0
        deleted_old_paper_trades = 0
        deleted_old_features = 0
    session.commit()
    return {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "mode": "compact",
        "factory_mode": factory_mode,
        "keep_recent_days": keep_recent_days,
        "retention": retention_config()
        | {
            "effective_raw_payload_retention_hours": raw_payload_hours,
            "effective_live_update_retention_hours": live_update_hours,
            "effective_raw_news_text_retention_days": raw_news_text_days,
            "effective_closed_candle_retention_days": closed_candle_days,
            "effective_training_feature_retention_days": training_feature_days,
            "effective_experience_retention_days": experience_days,
            "effective_raw_news_retention_days": raw_news_days,
            "operational_retention_days": settings.operational_retention_days,
            "railway_data_factory_mode": settings.railway_data_factory_mode,
        },
        "archived": archived,
        "deleted": {
            "live_candle_updates": deleted_live_updates,
            "legacy_live_candles": deleted_legacy_live_candles,
            "market_ticks": deleted_raw_ticks,
            "old_closed_candles": deleted_old_closed_candles,
            "old_news_articles": deleted_old_news_articles,
            "old_news_sentiment": deleted_old_news_sentiment,
            "old_training_features": deleted_old_training_features,
            "old_experiences": deleted_old_experiences,
            "old_ai_decisions": deleted_old_ai_decisions,
            "old_paper_trades": deleted_old_paper_trades,
            "old_debug_features": deleted_old_features,
            "account_equity": deleted_account_equity,
            "external_data_events": deleted_external_events,
        },
        "compacted": {
            "candles_raw_fields": compacted_candles,
            "live_candle_updates_raw_payload": compacted_live_updates,
            "market_ticks_raw_fields": compacted_market_ticks,
            "news_articles_raw_fields": compacted_news_articles,
            "news_articles_raw_text": compacted_news_text,
            "news_sentiment_raw_payload": compacted_news_sentiment,
            "experience_buffer_raw_context": compacted_experiences,
            "ai_decisions_raw_context": compacted_old_ai_decisions,
            "paper_trades_raw_payload": compacted_paper_trades,
            "account_equity_raw": compacted_account_equity,
            "external_data_raw_payload": compacted_external_events,
            "features_debug_payload": compacted_features,
            "training_features_final_ai_input": compacted_training_features,
            "experience_ai_decision_links_detached": detached_old_experience_decisions,
            "ai_decision_feature_links_detached": detached_old_decision_features,
            "ai_decision_trade_links_detached": detached_old_decision_trades,
            "experience_feature_links_detached": detached_old_experience_features,
            "training_feature_source_links_detached": detached_old_training_feature_sources,
        },
    }


def cleanup_old_data(session: Session) -> dict[str, Any]:
    return compact_database(session)


def retention_config() -> dict[str, Any]:
    return {
        "railway_data_factory_mode": settings.railway_data_factory_mode,
        "data_lifecycle_interval_seconds": settings.data_lifecycle_interval_seconds,
        "operational_retention_days": settings.operational_retention_days,
        "raw_payload_retention_hours": settings.raw_payload_retention_hours,
        "live_update_retention_hours": settings.live_update_retention_hours,
        "account_equity_retention_days": settings.account_equity_retention_days,
        "raw_news_text_retention_days": settings.raw_news_text_retention_days,
        "keep_closed_candles_days": settings.keep_closed_candles_days,
        "keep_training_features_days": settings.keep_training_features_days,
        "keep_experience_days": settings.keep_experience_days,
        "raw_news_retention_days": settings.raw_news_retention_days,
        "raw_external_event_retention_days": settings.raw_external_event_retention_days,
        "raw_tick_retention_days": settings.raw_tick_retention_days,
        "store_market_ticks": settings.store_market_ticks,
        "diagnostic_retention_days": settings.diagnostic_retention_days,
        "external_data_retention_days": settings.external_data_retention_days,
        "training_feature_retention_days": settings.training_feature_retention_days,
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
        daily_raw_export: dict[str, Any] | None = None
        with SessionLocal() as session:
            try:
                from app.services.raw_data_export import finish_daily_raw_data

                daily_raw_export = finish_daily_raw_data(session)
            except Exception as exc:  # pragma: no cover - defensive lifecycle boundary
                logger.exception("Daily raw data export failed before compact cleanup")
                daily_raw_export = {
                    "status": "error",
                    "message": "Daily raw data export failed before compact cleanup.",
                    "error": str(exc),
                }
            if daily_raw_export.get("status") == "ok":
                result = cleanup_old_data(session)
                result["daily_raw_export"] = daily_raw_export
            else:
                result = {
                    "ran_at": datetime.now(timezone.utc).isoformat(),
                    "mode": "skipped",
                    "message": "Compact cleanup skipped because daily raw data export failed.",
                    "daily_raw_export": daily_raw_export,
                }
        self.state.last_run_at = datetime.now(timezone.utc).isoformat()
        self.state.last_cleanup = result
        self.state.last_error = None if daily_raw_export is None or daily_raw_export.get("status") == "ok" else daily_raw_export.get("error")
        return self.status()

    def compact_once(
        self,
        *,
        archive_before_delete: bool = False,
        archive_tables: list[str] | None = None,
        factory_mode: bool = False,
        keep_recent_days: int | None = None,
    ) -> dict[str, Any]:
        with SessionLocal() as session:
            result = compact_database(
                session,
                archive_before_delete=archive_before_delete,
                archive_tables=archive_tables,
                factory_mode=factory_mode,
                keep_recent_days=keep_recent_days,
            )
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
