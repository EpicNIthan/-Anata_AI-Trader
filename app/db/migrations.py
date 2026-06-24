from __future__ import annotations

import logging

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


ADDITIVE_COLUMNS: dict[str, dict[str, str]] = {
    "candles": {
        "source_name": "VARCHAR(128)",
        "raw_payload": "JSON",
    },
    "market_ticks": {
        "source_name": "VARCHAR(128)",
        "raw_payload": "JSON",
    },
    "news_articles": {
        "source_name": "VARCHAR(128)",
        "raw_payload": "JSON",
    },
    "news_sentiment": {
        "sentiment_label": "VARCHAR(32)",
        "confidence": "FLOAT",
        "source_name": "VARCHAR(128)",
        "raw_payload": "JSON",
    },
    "features": {
        "schema_version": "VARCHAR(64)",
        "source_name": "VARCHAR(128)",
        "raw_payload": "JSON",
    },
    "paper_trades": {
        "raw_payload": "JSON",
    },
    "model_versions": {
        "model_id": "VARCHAR(128)",
        "feature_schema_version": "VARCHAR(64)",
        "feature_columns": "JSON",
        "parent_model_id": "VARCHAR(128)",
        "checkpoint_path": "TEXT",
        "raw_payload": "JSON",
    },
    "training_runs": {
        "feature_schema_version": "VARCHAR(64)",
        "from_checkpoint_path": "TEXT",
        "since_date": "TIMESTAMP WITH TIME ZONE",
        "use_all_data": "BOOLEAN",
    },
    "ai_decisions": {
        "source_name": "VARCHAR(128)",
        "model_version_id": "INTEGER",
        "feature_schema_version": "VARCHAR(64)",
        "market_state": "JSON",
        "news_state": "JSON",
        "result": "JSON",
        "reward": "FLOAT",
        "raw_payload": "JSON",
    },
}


def run_additive_migrations(engine: Engine) -> None:
    """Apply small nullable-column migrations for deployments without Alembic yet."""
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    with engine.begin() as conn:
        for table_name, columns in ADDITIVE_COLUMNS.items():
            if table_name not in existing_tables:
                continue
            existing_columns = {column["name"] for column in inspector.get_columns(table_name)}
            for column_name, column_type in columns.items():
                if column_name in existing_columns:
                    continue
                logger.info("Adding missing column %s.%s", table_name, column_name)
                conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"))
