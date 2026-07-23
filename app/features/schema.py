from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from app.db.models import Feature

CURRENT_FEATURE_SCHEMA_VERSION = "price-news-market-v5"

TECHNICAL_INDICATOR_COLUMNS = [
    "rsi_14",
    "macd_pct",
    "macd_signal_pct",
    "macd_histogram_pct",
    "sma_20_distance_pct",
    "ema_20_distance_pct",
    "bollinger_width_pct",
    "bollinger_position",
    "atr_14_pct",
    "vwap_20_distance_pct",
    "adx_14",
]

TIME_CONTEXT_FEATURE_COLUMNS = [
    "time_hour_utc_sin",
    "time_hour_utc_cos",
    "time_day_of_week_sin",
    "time_day_of_week_cos",
    "time_is_weekend",
    "session_asia",
    "session_london",
    "session_new_york",
]

REGIME_FEATURE_COLUMNS = [
    "regime_trend_strength",
    "regime_direction_score",
    "regime_volatility_score",
    "regime_news_shock_score",
    "regime_risk_off_score",
    "regime_liquidity_stress_score",
    "regime_breakout_pressure",
    "regime_mean_reversion_pressure",
    "regime_crowd_pressure",
]

SPOT_CONTEXT_FEATURE_COLUMNS = [
    "spot_price_change_24h",
    "spot_volume_24h",
    "spot_quote_volume_24h",
    "spot_trade_count_24h",
    "spot_weighted_avg_price",
    "spot_bid_ask_spread_pct",
    "spot_orderbook_imbalance",
    "spot_depth_bid_notional_5",
    "spot_depth_ask_notional_5",
    "spot_intraday_range_pct",
    "spot_activity_score",
]

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
    "price-news-market-v4": [
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
        *SPOT_CONTEXT_FEATURE_COLUMNS,
        "fear_greed_value",
        "fear_greed_change_1d",
        "fear_greed_change_24h",
        "fear_greed_classification",
        "total_market_cap_usd",
        "market_cap_change_24h",
        "global_market_cap_change_24h",
        "total_volume_usd",
        "total_volume_change_24h",
        "btc_dominance",
        "btc_dominance_change",
        "btc_dominance_change_24h",
        "eth_dominance",
        "liquidation_long_usd_1m",
        "liquidation_short_usd_1m",
        "liquidation_long_usd_5m",
        "liquidation_short_usd_5m",
        "liquidation_total_usd_5m",
        "liquidation_imbalance_5m",
        "liquidation_spike_score",
        "usdt_deviation",
        "usdc_deviation",
        "usdt_price_deviation",
        "usdc_price_deviation",
        "stablecoin_depeg_risk",
        "stablecoin_supply_change_1d",
        "stablecoin_supply_change_24h",
        "macro_risk_score",
        "regulation_risk_score",
        "fed_risk_score",
        "war_risk_score",
        "exchange_hack_risk_score",
        "etf_positive_score",
        "security_risk_score",
        "etf_bullish_score",
        "world_risk_score",
        "market_regime_score",
        *REGIME_FEATURE_COLUMNS,
    ],
    "price-news-market-v5": [
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
        *TECHNICAL_INDICATOR_COLUMNS,
        *TIME_CONTEXT_FEATURE_COLUMNS,
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
        *SPOT_CONTEXT_FEATURE_COLUMNS,
        "fear_greed_value",
        "fear_greed_change_1d",
        "fear_greed_change_24h",
        "fear_greed_classification",
        "total_market_cap_usd",
        "market_cap_change_24h",
        "global_market_cap_change_24h",
        "total_volume_usd",
        "total_volume_change_24h",
        "btc_dominance",
        "btc_dominance_change",
        "btc_dominance_change_24h",
        "eth_dominance",
        "liquidation_long_usd_1m",
        "liquidation_short_usd_1m",
        "liquidation_long_usd_5m",
        "liquidation_short_usd_5m",
        "liquidation_total_usd_5m",
        "liquidation_imbalance_5m",
        "liquidation_spike_score",
        "usdt_deviation",
        "usdc_deviation",
        "usdt_price_deviation",
        "usdc_price_deviation",
        "stablecoin_depeg_risk",
        "stablecoin_supply_change_1d",
        "stablecoin_supply_change_24h",
        "macro_risk_score",
        "regulation_risk_score",
        "fed_risk_score",
        "war_risk_score",
        "exchange_hack_risk_score",
        "etf_positive_score",
        "security_risk_score",
        "etf_bullish_score",
        "world_risk_score",
        "market_regime_score",
        *REGIME_FEATURE_COLUMNS,
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
    "rsi_14": 0.5,
    "macd_pct": 0.0,
    "macd_signal_pct": 0.0,
    "macd_histogram_pct": 0.0,
    "sma_20_distance_pct": 0.0,
    "ema_20_distance_pct": 0.0,
    "bollinger_width_pct": 0.0,
    "bollinger_position": 0.5,
    "atr_14_pct": 0.0,
    "vwap_20_distance_pct": 0.0,
    "adx_14": 0.0,
    "time_hour_utc_sin": 0.0,
    "time_hour_utc_cos": 0.0,
    "time_day_of_week_sin": 0.0,
    "time_day_of_week_cos": 0.0,
    "time_is_weekend": 0.0,
    "session_asia": 0.0,
    "session_london": 0.0,
    "session_new_york": 0.0,
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
    "spot_price_change_24h": 0.0,
    "spot_volume_24h": 0.0,
    "spot_quote_volume_24h": 0.0,
    "spot_trade_count_24h": 0.0,
    "spot_weighted_avg_price": 0.0,
    "spot_bid_ask_spread_pct": 0.0,
    "spot_orderbook_imbalance": 0.0,
    "spot_depth_bid_notional_5": 0.0,
    "spot_depth_ask_notional_5": 0.0,
    "spot_intraday_range_pct": 0.0,
    "spot_activity_score": 0.0,
    "fear_greed_value": 0.0,
    "fear_greed_change_1d": 0.0,
    "fear_greed_change_24h": 0.0,
    "fear_greed_classification": 0.0,
    "total_market_cap_usd": 0.0,
    "market_cap_change_24h": 0.0,
    "global_market_cap_change_24h": 0.0,
    "total_volume_usd": 0.0,
    "total_volume_change_24h": 0.0,
    "btc_dominance": 0.0,
    "btc_dominance_change": 0.0,
    "btc_dominance_change_24h": 0.0,
    "eth_dominance": 0.0,
    "liquidation_long_usd_1m": 0.0,
    "liquidation_short_usd_1m": 0.0,
    "liquidation_long_usd_5m": 0.0,
    "liquidation_short_usd_5m": 0.0,
    "liquidation_total_usd_5m": 0.0,
    "liquidation_imbalance_5m": 0.0,
    "liquidation_spike_score": 0.0,
    "usdt_deviation": 0.0,
    "usdc_deviation": 0.0,
    "usdt_price_deviation": 0.0,
    "usdc_price_deviation": 0.0,
    "stablecoin_depeg_risk": 0.0,
    "stablecoin_supply_change_1d": 0.0,
    "stablecoin_supply_change_24h": 0.0,
    "macro_risk_score": 0.0,
    "regulation_risk_score": 0.0,
    "fed_risk_score": 0.0,
    "war_risk_score": 0.0,
    "exchange_hack_risk_score": 0.0,
    "etf_positive_score": 0.0,
    "security_risk_score": 0.0,
    "etf_bullish_score": 0.0,
    "world_risk_score": 0.0,
    "market_regime_score": 0.0,
    "regime_trend_strength": 0.0,
    "regime_direction_score": 0.0,
    "regime_volatility_score": 0.0,
    "regime_news_shock_score": 0.0,
    "regime_risk_off_score": 0.0,
    "regime_liquidity_stress_score": 0.0,
    "regime_breakout_pressure": 0.0,
    "regime_mean_reversion_pressure": 0.0,
    "regime_crowd_pressure": 0.0,
    "last_close": None,
    "candles_used": 0.0,
    "sentiment_articles_used": 0.0,
}


def _float_value(values: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = values.get(key, default)
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def derived_regime_values(values: dict[str, Any]) -> dict[str, float]:
    trend = _float_value(values, "trend_score")
    candle_5m = _float_value(values, "candle_return_5m")
    volatility = abs(_float_value(values, "volatility"))
    volume_change = _float_value(values, "volume_change")
    sentiment = _float_value(values, "sentiment_score")
    risk = _float_value(values, "risk_score")
    impact = _float_value(values, "impact_score")
    macro = _float_value(values, "macro_risk_score")
    world = _float_value(values, "world_risk_score")
    regulation = _float_value(values, "regulation_risk_score")
    fed = _float_value(values, "fed_risk_score")
    war = _float_value(values, "war_risk_score")
    security = _float_value(values, "security_risk_score")
    stablecoin = _float_value(values, "stablecoin_depeg_risk")
    liquidation = _float_value(values, "liquidation_spike_score")
    crowd = abs(_float_value(values, "trader_crowd_score"))
    crowd_risk = _float_value(values, "crowd_risk_score")
    open_interest_change = abs(_float_value(values, "open_interest_change"))
    funding_rate = abs(_float_value(values, "funding_rate"))
    spot_spread = abs(_float_value(values, "spot_bid_ask_spread_pct"))
    spot_imbalance = abs(_float_value(values, "spot_orderbook_imbalance"))
    spot_range = abs(_float_value(values, "spot_intraday_range_pct"))
    spot_activity = _float_value(values, "spot_activity_score")

    trend_strength = _clip(max(abs(trend), min(abs(candle_5m) * 80.0, 1.0)))
    direction_score = _clip(trend + candle_5m * 40.0, -1.0, 1.0)
    volatility_score = _clip(max(math.tanh(volatility * 120.0), min(spot_range * 15.0, 1.0)), 0.0, 1.0)
    news_shock = _clip(max(abs(sentiment) * max(impact, 0.0), risk, regulation, fed, war, security))
    risk_off = _clip(max(risk, macro, world, regulation, fed, war, security, stablecoin))
    liquidity_stress = _clip(
        max(
            liquidation,
            min(open_interest_change * 25.0, 1.0),
            min(funding_rate * 500.0, 1.0),
            min(spot_spread * 200.0, 1.0),
        )
    )
    breakout = _clip(
        max(abs(candle_5m) * 70.0, max(volume_change, 0.0) * 0.15, spot_activity, spot_range * 10.0)
        * max(0.25, trend_strength)
    )
    mean_reversion = _clip(volatility_score * (1.0 - min(trend_strength, 1.0)) * (1.0 - min(news_shock, 1.0)))
    crowd_pressure = _clip(max(crowd, crowd_risk, spot_imbalance, abs(_float_value(values, "crowd_long_short_ratio") - 1.0) * 0.25))

    return {
        "regime_trend_strength": trend_strength,
        "regime_direction_score": direction_score,
        "regime_volatility_score": volatility_score,
        "regime_news_shock_score": news_shock,
        "regime_risk_off_score": risk_off,
        "regime_liquidity_stress_score": liquidity_stress,
        "regime_breakout_pressure": breakout,
        "regime_mean_reversion_pressure": mean_reversion,
        "regime_crowd_pressure": crowd_pressure,
    }


def _fill_missing_regime_values(values: dict[str, Any]) -> dict[str, Any]:
    output = dict(values)
    derived = derived_regime_values(output)
    for key, value in derived.items():
        if key not in values or values.get(key) in (None, ""):
            output[key] = value
    return output


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
    safe_values.update(_fill_missing_regime_values(values))
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
    safe_values = dict(DEFAULT_FEATURE_VALUES)
    safe_values.update(_fill_missing_regime_values(values))
    output: dict[str, Any] = {}
    for column in columns:
        output[column] = safe_values.get(column, DEFAULT_FEATURE_VALUES.get(column, 0.0))
    for optional_key in (
        "trend",
        "last_close",
        "candles_used",
        "sentiment_articles_used",
        "price_change",
        "final_ai_input",
        "external_ai_available",
        "external_ai_missing",
        "external_ai_failed",
        "external_ai_confidence",
        "external_ai_age_seconds",
        "external_ai_provider",
        "external_ai_prompt_version",
        "external_ai_direction_score",
    ):
        output.setdefault(optional_key, safe_values.get(optional_key, DEFAULT_FEATURE_VALUES.get(optional_key)))
    output["schema_version"] = schema_version
    return output


def numeric_vector(feature: Feature | dict[str, Any], feature_columns: list[str]) -> list[float]:
    values = values_from_feature(feature, feature_columns)
    vector: list[float] = []
    for column in feature_columns:
        value = values.get(column, DEFAULT_FEATURE_VALUES.get(column, 0.0))
        vector.append(float(value or 0.0))
    return vector
