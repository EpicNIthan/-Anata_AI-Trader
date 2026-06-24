from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import desc, func, select, text
from sqlalchemy.orm import Session

from app.db.models import Base, Candle, LiveCandleUpdate


def _dt(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _count_rows(session: Session, table_name: str) -> int:
    quoted = table_name.replace('"', '""')
    return int(session.execute(text(f'SELECT COUNT(*) FROM "{quoted}"')).scalar_one() or 0)


def _postgres_table_sizes(session: Session) -> dict[str, dict[str, float | int]]:
    rows = session.execute(
        text(
            """
            SELECT relname AS table_name, pg_total_relation_size(relid) AS bytes
            FROM pg_catalog.pg_statio_user_tables
            """
        )
    ).mappings()
    return {
        str(row["table_name"]): {
            "bytes": int(row["bytes"] or 0),
            "mb": round((int(row["bytes"] or 0) / 1024 / 1024), 4),
        }
        for row in rows
    }


def _postgres_database_size(session: Session) -> int | None:
    try:
        return int(session.execute(text("SELECT pg_database_size(current_database())")).scalar_one() or 0)
    except Exception:
        return None


def _postgres_column_size(session: Session, table_name: str, column_name: str) -> int | None:
    quoted_table = table_name.replace('"', '""')
    quoted_column = column_name.replace('"', '""')
    try:
        return int(
            session.execute(text(f'SELECT COALESCE(SUM(pg_column_size("{quoted_column}")), 0) FROM "{quoted_table}"')).scalar_one()
            or 0
        )
    except Exception:
        return None


def _json_column_estimate(session: Session, table_name: str, column_name: str, *, limit: int = 1000) -> int | None:
    table = Base.metadata.tables[table_name]
    if column_name not in table.c:
        return None
    total_rows = _count_rows(session, table_name)
    if total_rows <= 0:
        return 0
    rows = session.execute(select(table.c[column_name]).where(table.c[column_name].is_not(None)).limit(limit)).all()
    if not rows:
        return 0
    sample_bytes = 0
    for (value,) in rows:
        if value is None:
            continue
        if isinstance(value, str):
            sample_bytes += len(value.encode("utf-8"))
        else:
            sample_bytes += len(json.dumps(value, separators=(",", ":"), default=str).encode("utf-8"))
    non_null_sample = len(rows)
    if non_null_sample <= 0:
        return 0
    return int((sample_bytes / non_null_sample) * total_rows)


def _raw_payload_estimate(session: Session, table_name: str, *, size_supported: bool) -> dict[str, Any]:
    table = Base.metadata.tables[table_name]
    estimates: dict[str, int] = {}
    for column_name in ("raw_payload", "raw", "raw_text", "payload", "feature_values", "market_state", "news_state", "feature_payload", "result"):
        if column_name not in table.c:
            continue
        value = _postgres_column_size(session, table_name, column_name) if size_supported else None
        if value is None:
            value = _json_column_estimate(session, table_name, column_name)
        estimates[column_name] = int(value or 0)
    total = sum(estimates.values())
    return {
        "bytes": total,
        "mb": round(total / 1024 / 1024, 4),
        "columns": {key: {"bytes": value, "mb": round(value / 1024 / 1024, 4)} for key, value in estimates.items()},
    }


def _table_time_bounds(session: Session, table_name: str) -> dict[str, str | None]:
    table = Base.metadata.tables[table_name]
    for column_name in (
        "updated_at",
        "created_at",
        "open_time",
        "event_time",
        "published_at",
        "as_of",
        "timestamp",
        "started_at",
    ):
        if column_name not in table.c:
            continue
        oldest, newest = session.execute(select(func.min(table.c[column_name]), func.max(table.c[column_name]))).one()
        return {
            "time_column": column_name,
            "oldest": _dt(oldest),
            "newest": _dt(newest),
        }
    return {"time_column": None, "oldest": None, "newest": None}


def database_diagnostics(session: Session, *, include_raw_estimates: bool = False) -> dict[str, Any]:
    bind = session.get_bind()
    dialect = bind.dialect.name if bind is not None else "unknown"
    table_names = sorted(Base.metadata.tables.keys())
    size_supported = dialect == "postgresql"
    sizes = _postgres_table_sizes(session) if size_supported else {}
    database_total_bytes = _postgres_database_size(session) if size_supported else None
    if database_total_bytes is None and dialect == "sqlite" and bind is not None:
        database_path = getattr(bind.url, "database", None)
        if database_path and database_path not in {":memory:", ""}:
            path = Path(database_path)
            if path.exists():
                database_total_bytes = path.stat().st_size

    rows_by_table: list[dict[str, Any]] = []
    for table_name in table_names:
        count = _count_rows(session, table_name)
        size = sizes.get(table_name)
        time_bounds = _table_time_bounds(session, table_name)
        raw_payload_size = (
            _raw_payload_estimate(session, table_name, size_supported=size_supported)
            if include_raw_estimates
            else {"bytes": None, "mb": None, "columns": {}}
        )
        rows_by_table.append(
            {
                "table": table_name,
                "rows": count,
                "mb": size["mb"] if size else None,
                "bytes": size["bytes"] if size else None,
                "raw_payload_estimate": raw_payload_size,
                **time_bounds,
            }
        )

    candle_groups = session.execute(
        select(
            Candle.symbol,
            Candle.interval,
            Candle.is_closed,
            func.count(Candle.id),
            func.min(Candle.open_time),
            func.max(Candle.open_time),
            func.max(Candle.updated_at),
        )
        .group_by(Candle.symbol, Candle.interval, Candle.is_closed)
        .order_by(Candle.symbol, Candle.interval, Candle.is_closed.desc())
    ).all()
    candles_by_symbol_timeframe = [
        {
            "symbol": symbol,
            "timeframe": interval,
            "quality": "closed_training" if is_closed else "live_in_progress",
            "is_closed": bool(is_closed),
            "rows": count,
            "first_open_time": _dt(first_open),
            "latest_open_time": _dt(latest_open),
            "latest_updated_at": _dt(latest_update),
        }
        for symbol, interval, is_closed, count, first_open, latest_open, latest_update in candle_groups
    ]

    duplicate_count = func.count(Candle.id)
    duplicate_groups = session.execute(
        select(Candle.exchange, Candle.symbol, Candle.interval, Candle.open_time, func.count(Candle.id).label("rows"))
        .group_by(Candle.exchange, Candle.symbol, Candle.interval, Candle.open_time)
        .having(duplicate_count > 1)
        .order_by(duplicate_count.desc())
        .limit(20)
    ).all()
    duplicate_candles = [
        {
            "exchange": exchange,
            "symbol": symbol,
            "timeframe": interval,
            "open_time": _dt(open_time),
            "rows": rows,
        }
        for exchange, symbol, interval, open_time, rows in duplicate_groups
    ]

    closed_count = int(session.scalar(select(func.count(Candle.id)).where(Candle.is_closed.is_(True))) or 0)
    legacy_live_count = int(session.scalar(select(func.count(Candle.id)).where(Candle.is_closed.is_(False))) or 0)
    live_update_count = int(session.scalar(select(func.count(LiveCandleUpdate.id))) or 0)
    latest_candle = session.scalar(select(Candle).order_by(desc(Candle.open_time)).limit(1))
    latest_live_update = session.scalar(select(LiveCandleUpdate).order_by(desc(LiveCandleUpdate.updated_at)).limit(1))
    total_bytes = database_total_bytes
    if total_bytes is None and size_supported:
        total_bytes = sum(int(item.get("bytes") or 0) for item in sizes.values())
    top_largest_tables = sorted(
        rows_by_table,
        key=lambda item: int(item.get("bytes") or item.get("raw_payload_estimate", {}).get("bytes") or 0),
        reverse=True,
    )[:10]
    total_mb = round((total_bytes or 0) / 1024 / 1024, 4) if total_bytes is not None else None
    storage_gb = (total_bytes or 0) / 1024 / 1024 / 1024 if total_bytes is not None else None
    estimated_monthly_cost_usd = round(storage_gb * 0.25, 4) if storage_gb is not None else None

    return {
        "dialect": dialect,
        "size_supported": size_supported,
        "database_total": {
            "bytes": total_bytes,
            "mb": total_mb,
            "gb": round(storage_gb, 4) if storage_gb is not None else None,
            "warning": "DB is over 1GB. Run compact/archive and keep Railway inference-only." if total_mb and total_mb > 1024 else None,
        },
        "estimated_monthly_storage_cost_usd": estimated_monthly_cost_usd,
        "cost_note": "Rough planning estimate only; Railway billing can differ by plan/region.",
        "rows_by_table": rows_by_table,
        "top_largest_tables": top_largest_tables,
        "candles": {
            "closed_training_rows": closed_count,
            "live_update_rows": live_update_count,
            "legacy_live_rows_in_candles": legacy_live_count,
            "store_live_rows_meaning": "live rows are stored in live_candle_updates as one upserted in-progress candle per symbol/timeframe/open_time",
            "training_quality_rule": "use candles.is_closed=true for training-quality candle features",
            "latest": {
                "symbol": latest_candle.symbol if latest_candle else None,
                "timeframe": latest_candle.interval if latest_candle else None,
                "open_time": _dt(latest_candle.open_time if latest_candle else None),
                "is_closed": latest_candle.is_closed if latest_candle else None,
                "source_name": latest_candle.source_name if latest_candle else None,
            },
            "latest_live_update": {
                "symbol": latest_live_update.symbol if latest_live_update else None,
                "timeframe": latest_live_update.interval if latest_live_update else None,
                "open_time": _dt(latest_live_update.open_time if latest_live_update else None),
                "updated_at": _dt(latest_live_update.updated_at if latest_live_update else None),
                "update_count": latest_live_update.update_count if latest_live_update else None,
                "source_name": latest_live_update.source_name if latest_live_update else None,
            },
            "by_symbol_timeframe": candles_by_symbol_timeframe,
            "duplicate_open_time_groups": duplicate_candles,
            "duplicate_group_count": len(duplicate_candles),
        },
    }
