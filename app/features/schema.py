from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.db.models import Feature

CURRENT_FEATURE_SCHEMA_VERSION = "price-news-v3"

FEATURE_COLUMNS_BY_SCHEMA: dict[str, list[str]] = {
    "price-news-v1": [
        "price_change",
        "volume_change",
        "volatility",
        "sentiment_score",
        "risk_score",
    ],
    "price-news-v2": [
        "sentiment_score",
        "sentiment_confidence",
        "risk_score",
        "impact_score",
        "recency_weight",
        "btc_related",
        "eth_related",
        "macro_related",
        "candle_return_1m",
        "candle_return_5m",
        "volatility",
        "volume_change",
        "trend_score",
    ],
    "price-news-v3": [
        "sentiment_score",
        "sentiment_confidence",
        "risk_score",
        "impact_score",
        "recency_weight",
        "btc_related",
        "eth_related",
        "macro_related",
        "candle_return_1m",
        "candle_return_5m",
        "volatility",
        "volume_change",
        "trend_score",
        "crowd_long_account_pct",
        "crowd_short_account_pct",
        "crowd_long_short_ratio",
        "top_trader_long_account_pct",
        "top_trader_position_long_pct",
        "taker_buy_pressure",
        "taker_buy_sell_ratio",
        "open_interest_value",
        "open_interest_change",
        "funding_rate",
        "trader_crowd_score",
        "crowd_risk_score",
        "derivatives_recency_weight",
    ],
}

DEFAULT_FEATURE_VALUES: dict[str, float | None] = {
    "price_change": 0.0,
    "volume_change": 0.0,
    "volatility": 0.0,
    "sentiment_score": 0.0,
    "sentiment_confidence": 0.0,
    "risk_score": 0.0,
    "impact_score": 0.0,
    "recency_weight": 0.0,
    "btc_related": 0.0,
    "eth_related": 0.0,
    "macro_related": 0.0,
    "candle_return_1m": 0.0,
    "candle_return_5m": 0.0,
    "trend_score": 0.0,
    "crowd_long_account_pct": 0.0,
    "crowd_short_account_pct": 0.0,
    "crowd_long_short_ratio": 0.0,
    "top_trader_long_account_pct": 0.0,
    "top_trader_position_long_pct": 0.0,
    "taker_buy_pressure": 0.0,
    "taker_buy_sell_ratio": 0.0,
    "open_interest_value": 0.0,
    "open_interest_change": 0.0,
    "funding_rate": 0.0,
    "trader_crowd_score": 0.0,
    "crowd_risk_score": 0.0,
    "derivatives_recency_weight": 0.0,
    "last_close": None,
    "candles_used": 0.0,
    "sentiment_articles_used": 0.0,
}


@dataclass(frozen=True)
class FeatureVector:
    schema_version: str
    values: dict[str, float | str | None]
    metadata: dict[str, Any]


def columns_for_schema(schema_version: str | None = None) -> list[str]:
    version = schema_version or CURRENT_FEATURE_SCHEMA_VERSION
    return list(FEATURE_COLUMNS_BY_SCHEMA.get(version, FEATURE_COLUMNS_BY_SCHEMA[CURRENT_FEATURE_SCHEMA_VERSION]))


def feature_payload(
    *,
    schema_version: str,
    values: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    sources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    safe_values = dict(DEFAULT_FEATURE_VALUES)
    safe_values.update(values)
    return {
        "schema_version": schema_version,
        "values": safe_values,
        "metadata": metadata or {},
        "sources": sources or {},
    }


def values_from_feature(feature: Feature | dict[str, Any], feature_columns: list[str] | None = None) -> dict[str, Any]:
    if isinstance(feature, dict):
        payload = feature
        values = payload.get("values", payload)
        schema_version = payload.get("schema_version", CURRENT_FEATURE_SCHEMA_VERSION)
    else:
        payload = feature.payload or {}
        values = payload.get("values", {})
        schema_version = feature.schema_version or payload.get("schema_version", CURRENT_FEATURE_SCHEMA_VERSION)
        legacy = {
            "price_change": feature.price_change,
            "volume_change": feature.volume_change,
            "volatility": feature.volatility,
            "trend": feature.trend,
            "sentiment_score": feature.sentiment_score,
            "risk_score": feature.risk_score,
        }
        legacy.update(values)
        values = legacy

    columns = feature_columns or columns_for_schema(schema_version)
    output: dict[str, Any] = {}
    for column in columns:
        output[column] = values.get(column, DEFAULT_FEATURE_VALUES.get(column, 0.0))
    for optional_key in (
        "trend",
        "last_close",
        "candles_used",
        "sentiment_articles_used",
        "price_change",
        "final_ai_input",
    ):
        output.setdefault(optional_key, values.get(optional_key, DEFAULT_FEATURE_VALUES.get(optional_key)))
    output["schema_version"] = schema_version
    return output


def numeric_vector(feature: Feature | dict[str, Any], feature_columns: list[str]) -> list[float]:
    values = values_from_feature(feature, feature_columns)
    vector: list[float] = []
    for column in feature_columns:
        value = values.get(column, DEFAULT_FEATURE_VALUES.get(column, 0.0))
        vector.append(float(value or 0.0))
    return vector
