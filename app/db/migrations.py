from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


ADDITIVE_COLUMNS: dict[str, dict[str, str]] = {
    "candles": {
        "exchange": "VARCHAR(32) DEFAULT 'binance'",
        "source_name": "VARCHAR(128)",
        "symbol": "VARCHAR(32)",
        "interval": "VARCHAR(16) DEFAULT '1m'",
        "open_time": "TIMESTAMP WITH TIME ZONE",
        "close_time": "TIMESTAMP WITH TIME ZONE",
        "open": "FLOAT DEFAULT 0",
        "high": "FLOAT DEFAULT 0",
        "low": "FLOAT DEFAULT 0",
        "close": "FLOAT DEFAULT 0",
        "volume": "FLOAT DEFAULT 0",
        "quote_volume": "FLOAT DEFAULT 0",
        "trades": "INTEGER DEFAULT 0",
        "is_closed": "BOOLEAN DEFAULT FALSE",
        "raw": "JSON",
        "raw_payload": "JSON",
        "created_at": "TIMESTAMP WITH TIME ZONE",
        "updated_at": "TIMESTAMP WITH TIME ZONE",
    },
    "live_candle_updates": {
        "exchange": "VARCHAR(32) DEFAULT 'binance'",
        "source_name": "VARCHAR(128)",
        "symbol": "VARCHAR(32)",
        "interval": "VARCHAR(16) DEFAULT '1m'",
        "event_time": "TIMESTAMP WITH TIME ZONE",
        "open_time": "TIMESTAMP WITH TIME ZONE",
        "close_time": "TIMESTAMP WITH TIME ZONE",
        "open": "FLOAT DEFAULT 0",
        "high": "FLOAT DEFAULT 0",
        "low": "FLOAT DEFAULT 0",
        "close": "FLOAT DEFAULT 0",
        "volume": "FLOAT DEFAULT 0",
        "quote_volume": "FLOAT DEFAULT 0",
        "trades": "INTEGER DEFAULT 0",
        "update_count": "INTEGER DEFAULT 1",
        "raw_payload": "JSON",
        "created_at": "TIMESTAMP WITH TIME ZONE",
        "updated_at": "TIMESTAMP WITH TIME ZONE",
    },
    "market_ticks": {
        "exchange": "VARCHAR(32) DEFAULT 'binance'",
        "source_name": "VARCHAR(128)",
        "symbol": "VARCHAR(32)",
        "event_time": "TIMESTAMP WITH TIME ZONE",
        "price": "FLOAT DEFAULT 0",
        "quantity": "FLOAT DEFAULT 0",
        "raw": "JSON",
        "raw_payload": "JSON",
        "created_at": "TIMESTAMP WITH TIME ZONE",
    },
    "news_articles": {
        "source": "VARCHAR(128)",
        "source_name": "VARCHAR(128)",
        "title": "TEXT",
        "url": "TEXT",
        "published_at": "TIMESTAMP WITH TIME ZONE",
        "raw_text": "TEXT",
        "raw": "JSON",
        "raw_payload": "JSON",
        "created_at": "TIMESTAMP WITH TIME ZONE",
    },
    "news_sentiment": {
        "article_id": "INTEGER",
        "sentiment_score": "FLOAT DEFAULT 0",
        "risk_score": "FLOAT DEFAULT 0",
        "topics": "JSON",
        "affected_symbols": "JSON",
        "model_name": "VARCHAR(128) DEFAULT 'placeholder-v1'",
        "sentiment_label": "VARCHAR(32)",
        "confidence": "FLOAT",
        "source_name": "VARCHAR(128)",
        "raw_payload": "JSON",
        "created_at": "TIMESTAMP WITH TIME ZONE",
    },
    "features": {
        "symbol": "VARCHAR(32)",
        "schema_version": "VARCHAR(64)",
        "source_name": "VARCHAR(128)",
        "as_of": "TIMESTAMP WITH TIME ZONE",
        "price_change": "FLOAT DEFAULT 0",
        "volume_change": "FLOAT DEFAULT 0",
        "volatility": "FLOAT DEFAULT 0",
        "trend": "VARCHAR(32) DEFAULT 'sideways'",
        "sentiment_score": "FLOAT DEFAULT 0",
        "risk_score": "FLOAT DEFAULT 0",
        "payload": "JSON",
        "raw_payload": "JSON",
        "created_at": "TIMESTAMP WITH TIME ZONE",
    },
    "training_features": {
        "source_feature_id": "INTEGER",
        "symbol": "VARCHAR(32)",
        "schema_version": "VARCHAR(64)",
        "source_name": "VARCHAR(128)",
        "as_of": "TIMESTAMP WITH TIME ZONE",
        "feature_values": "JSON",
        "payload": "JSON",
        "created_at": "TIMESTAMP WITH TIME ZONE",
    },
    "paper_trades": {
        "symbol": "VARCHAR(32)",
        "action": "VARCHAR(16)",
        "side": "VARCHAR(16) DEFAULT 'LONG'",
        "quantity": "FLOAT DEFAULT 0",
        "price": "FLOAT DEFAULT 0",
        "notional": "FLOAT DEFAULT 0",
        "fee": "FLOAT DEFAULT 0",
        "realized_pnl": "FLOAT DEFAULT 0",
        "balance_after": "FLOAT DEFAULT 0",
        "equity_after": "FLOAT DEFAULT 0",
        "status": "VARCHAR(32) DEFAULT 'FILLED'",
        "reason": "TEXT",
        "raw_payload": "JSON",
        "created_at": "TIMESTAMP WITH TIME ZONE",
    },
    "positions": {
        "symbol": "VARCHAR(32)",
        "side": "VARCHAR(16) DEFAULT 'LONG'",
        "quantity": "FLOAT DEFAULT 0",
        "entry_price": "FLOAT DEFAULT 0",
        "current_price": "FLOAT DEFAULT 0",
        "notional": "FLOAT DEFAULT 0",
        "margin_used": "FLOAT DEFAULT 0",
        "leverage": "FLOAT DEFAULT 1",
        "stop_loss": "FLOAT",
        "take_profit": "FLOAT",
        "realized_pnl": "FLOAT DEFAULT 0",
        "unrealized_pnl": "FLOAT DEFAULT 0",
        "status": "VARCHAR(32) DEFAULT 'OPEN'",
        "opened_at": "TIMESTAMP WITH TIME ZONE",
        "closed_at": "TIMESTAMP WITH TIME ZONE",
    },
    "model_versions": {
        "model_id": "VARCHAR(128)",
        "name": "VARCHAR(128)",
        "version": "VARCHAR(64)",
        "feature_schema_version": "VARCHAR(64)",
        "feature_columns": "JSON",
        "path": "TEXT",
        "parent_model_id": "VARCHAR(128)",
        "checkpoint_path": "TEXT",
        "status": "VARCHAR(32) DEFAULT 'trained'",
        "metrics": "JSON",
        "raw_payload": "JSON",
        "created_at": "TIMESTAMP WITH TIME ZONE",
    },
    "training_runs": {
        "model_name": "VARCHAR(128)",
        "dataset_path": "TEXT",
        "model_version_id": "INTEGER",
        "feature_schema_version": "VARCHAR(64)",
        "from_checkpoint_path": "TEXT",
        "since_date": "TIMESTAMP WITH TIME ZONE",
        "use_all_data": "BOOLEAN",
        "status": "VARCHAR(32) DEFAULT 'created'",
        "metrics": "JSON",
        "started_at": "TIMESTAMP WITH TIME ZONE",
        "finished_at": "TIMESTAMP WITH TIME ZONE",
    },
    "ai_decisions": {
        "symbol": "VARCHAR(32)",
        "strategy_name": "VARCHAR(128) DEFAULT 'rule-based-v1'",
        "source_name": "VARCHAR(128)",
        "feature_id": "INTEGER",
        "model_version_id": "INTEGER",
        "feature_schema_version": "VARCHAR(64)",
        "action": "VARCHAR(16) DEFAULT 'HOLD'",
        "confidence": "FLOAT DEFAULT 0",
        "reason": "TEXT",
        "stop_loss": "FLOAT",
        "take_profit": "FLOAT",
        "execution_status": "VARCHAR(32) DEFAULT 'PENDING'",
        "execution_message": "TEXT",
        "trade_id": "INTEGER",
        "market_state": "JSON",
        "news_state": "JSON",
        "result": "JSON",
        "reward": "FLOAT",
        "raw": "JSON",
        "raw_payload": "JSON",
        "created_at": "TIMESTAMP WITH TIME ZONE",
    },
    "external_data_events": {
        "source_name": "VARCHAR(128)",
        "data_type": "VARCHAR(64)",
        "symbol": "VARCHAR(32)",
        "event_time": "TIMESTAMP WITH TIME ZONE",
        "numeric_value": "FLOAT",
        "payload": "JSON",
        "raw_payload": "JSON",
        "created_at": "TIMESTAMP WITH TIME ZONE",
    },
    "experience_buffer": {
        "ai_decision_id": "INTEGER",
        "feature_id": "INTEGER",
        "model_version_id": "INTEGER",
        "symbol": "VARCHAR(32)",
        "feature_schema_version": "VARCHAR(64)",
        "market_state": "JSON",
        "news_state": "JSON",
        "feature_payload": "JSON",
        "action": "VARCHAR(16) DEFAULT 'HOLD'",
        "confidence": "FLOAT DEFAULT 0",
        "result": "JSON",
        "reward": "FLOAT DEFAULT 0",
        "raw_payload": "JSON",
        "archived_at": "TIMESTAMP WITH TIME ZONE",
        "created_at": "TIMESTAMP WITH TIME ZONE",
    },
    "account_equity": {
        "timestamp": "TIMESTAMP WITH TIME ZONE",
        "cash_balance": "FLOAT DEFAULT 0",
        "equity": "FLOAT DEFAULT 0",
        "realized_pnl": "FLOAT DEFAULT 0",
        "unrealized_pnl": "FLOAT DEFAULT 0",
        "drawdown": "FLOAT DEFAULT 0",
        "raw": "JSON",
    },
}


def run_additive_migrations(engine: Engine) -> dict[str, Any]:
    """Apply small nullable-column migrations for deployments without Alembic yet."""
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    added_columns: list[dict[str, str]] = []
    skipped_missing_tables: list[str] = []
    with engine.begin() as conn:
        for table_name, columns in ADDITIVE_COLUMNS.items():
            if table_name not in existing_tables:
                skipped_missing_tables.append(table_name)
                continue
            existing_columns = {column["name"] for column in inspector.get_columns(table_name)}
            for column_name, column_type in columns.items():
                if column_name in existing_columns:
                    continue
                logger.info("Adding missing column %s.%s", table_name, column_name)
                conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"))
                added_columns.append({"table": table_name, "column": column_name, "type": column_type})
    refreshed_tables = set(inspect(engine).get_table_names())
    return {
        "status": "ok",
        "added_columns": added_columns,
        "added_count": len(added_columns),
        "existing_tables": sorted(refreshed_tables),
        "skipped_missing_tables": skipped_missing_tables,
    }

