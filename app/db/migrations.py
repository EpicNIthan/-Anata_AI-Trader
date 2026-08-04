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
        "event_time": "TIMESTAMP WITH TIME ZONE",
        "received_time": "TIMESTAMP WITH TIME ZONE",
        "processed_time": "TIMESTAMP WITH TIME ZONE",
        "available_to_model_time": "TIMESTAMP WITH TIME ZONE",
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
        "event_time": "TIMESTAMP WITH TIME ZONE",
        "received_time": "TIMESTAMP WITH TIME ZONE",
        "processed_time": "TIMESTAMP WITH TIME ZONE",
        "available_to_model_time": "TIMESTAMP WITH TIME ZONE",
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
        "paper_account_id": "VARCHAR(128) DEFAULT 'champion'",
        "risk_decision_id": "VARCHAR(64)",
        "simulated_order_id": "VARCHAR(64)",
        "decision_trace_id": "VARCHAR(64)",
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
        "paper_account_id": "VARCHAR(128) DEFAULT 'champion'",
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
        "model_family": "VARCHAR(128)",
        "lifecycle_state": "VARCHAR(32) DEFAULT 'TRAINED'",
        "health_status": "VARCHAR(32) DEFAULT 'HEALTHY'",
        "artifact_checksum": "VARCHAR(128)",
        "preprocessing_version": "VARCHAR(128)",
        "training_dataset_version": "VARCHAR(128)",
        "training_start_at": "TIMESTAMP WITH TIME ZONE",
        "training_end_at": "TIMESTAMP WITH TIME ZONE",
        "forecast_horizon_seconds": "INTEGER",
        "package_manifest": "JSON",
        "promotion_history": "JSON",
        "suspension_reason": "TEXT",
        "retirement_reason": "TEXT",
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
    # This V2 table may already exist on a deployment from an earlier rolling
    # release.  create_all() does not retrofit its new symbol discriminator.
    "external_ai_requests": {
        "symbol": "VARCHAR(32)",
    },
    "model_health_snapshots": {
        "rolling_information_coefficient": "FLOAT",
        "rolling_net_expectancy": "FLOAT",
        "calibration_error": "FLOAT",
        "prediction_drift": "FLOAT",
        "feature_drift": "FLOAT",
        "ood_rate": "FLOAT",
        "missing_feature_rate": "FLOAT",
        "live_shadow_divergence": "FLOAT",
        "transaction_cost_increase": "FLOAT",
        "signal_correlation_increase": "FLOAT",
        "regime_dependence": "FLOAT",
        "capacity_decline": "FLOAT",
        "consecutive_errors": "INTEGER DEFAULT 0",
        "recommended_weight_multiplier": "FLOAT DEFAULT 1.0",
        "recommended_action": "VARCHAR(64) DEFAULT 'NORMAL_WEIGHT'",
    },
    "signal_health_snapshots": {
        "rolling_information_coefficient": "FLOAT",
        "rolling_net_expectancy": "FLOAT",
        "calibration_error": "FLOAT",
        "prediction_drift": "FLOAT",
        "feature_drift": "FLOAT",
        "ood_rate": "FLOAT",
        "missing_feature_rate": "FLOAT",
        "live_shadow_divergence": "FLOAT",
        "transaction_cost_increase": "FLOAT",
        "correlation_increase": "FLOAT",
        "regime_dependence": "FLOAT",
        "capacity_decline": "FLOAT",
        "consecutive_errors": "INTEGER DEFAULT 0",
        "recommended_weight_multiplier": "FLOAT DEFAULT 1.0",
        "recommended_action": "VARCHAR(64) DEFAULT 'NORMAL_WEIGHT'",
    },
    "account_equity": {
        "timestamp": "TIMESTAMP WITH TIME ZONE",
        "paper_account_id": "VARCHAR(128) DEFAULT 'champion'",
        "cash_balance": "FLOAT DEFAULT 0",
        "equity": "FLOAT DEFAULT 0",
        "realized_pnl": "FLOAT DEFAULT 0",
        "unrealized_pnl": "FLOAT DEFAULT 0",
        "drawdown": "FLOAT DEFAULT 0",
        "raw": "JSON",
    },
}


# Index creation is additive and idempotent on SQLite and PostgreSQL.  Existing
# deployments created before V2 need these hot-path indexes because create_all() does
# not retrofit indexes onto already-created tables.
ADDITIVE_INDEXES: dict[str, tuple[str, ...]] = {
    "news_articles": (
        "CREATE INDEX IF NOT EXISTS ix_news_articles_available_to_model_time ON news_articles (available_to_model_time)",
    ),
    "features": (
        "CREATE INDEX IF NOT EXISTS ix_features_available_to_model_time ON features (available_to_model_time)",
    ),
    "paper_trades": (
        "CREATE INDEX IF NOT EXISTS ix_paper_trades_account_time ON paper_trades (paper_account_id, created_at)",
        "CREATE INDEX IF NOT EXISTS ix_paper_trades_risk_decision ON paper_trades (risk_decision_id)",
        "CREATE INDEX IF NOT EXISTS ix_paper_trades_trace ON paper_trades (decision_trace_id)",
    ),
    "positions": (
        "CREATE INDEX IF NOT EXISTS ix_positions_account_symbol_status ON positions (paper_account_id, symbol, status)",
    ),
    "account_equity": (
        "CREATE INDEX IF NOT EXISTS ix_account_equity_account_time ON account_equity (paper_account_id, timestamp)",
    ),
    "model_versions": (
        "CREATE INDEX IF NOT EXISTS ix_model_versions_lifecycle_health ON model_versions (lifecycle_state, health_status)",
    ),
    "external_ai_requests": (
        "CREATE INDEX IF NOT EXISTS ix_external_ai_requests_symbol_time ON external_ai_requests (symbol, requested_at)",
    ),
    "signal_health_snapshots": (
        "CREATE INDEX IF NOT EXISTS ix_signal_health_signal_time ON signal_health_snapshots (signal_family, observed_at)",
        "CREATE INDEX IF NOT EXISTS ix_signal_health_symbol_family_time ON signal_health_snapshots (symbol, signal_family, observed_at)",
    ),
    "model_health_snapshots": (
        "CREATE INDEX IF NOT EXISTS ix_model_health_model_time ON model_health_snapshots (model_version_id, observed_at)",
    ),
}


def run_additive_migrations(engine: Engine) -> dict[str, Any]:
    """Apply small nullable-column migrations for deployments without Alembic yet."""
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    created_tables: list[str] = []
    if "model_versions" in existing_tables and "model_artifact_blobs" not in existing_tables:
        # Existing deployments need the durable artifact table even though
        # ``create_all`` cannot be assumed for every migration entry point.
        from app.db.models import ModelArtifactBlob

        ModelArtifactBlob.__table__.create(bind=engine, checkfirst=True)
        created_tables.append("model_artifact_blobs")
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
        # Re-inspect after column additions so a partially upgraded database never
        # attempts an index against a missing legacy table.
        current_tables = set(inspect(conn).get_table_names())
        for table_name, statements in ADDITIVE_INDEXES.items():
            if table_name not in current_tables:
                continue
            for statement in statements:
                conn.execute(text(statement))
    refreshed_tables = set(inspect(engine).get_table_names())
    return {
        "status": "ok",
        "added_columns": added_columns,
        "added_count": len(added_columns),
        "created_tables": created_tables,
        "existing_tables": sorted(refreshed_tables),
        "skipped_missing_tables": skipped_missing_tables,
    }

