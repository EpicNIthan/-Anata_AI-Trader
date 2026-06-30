from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from statistics import mean, pstdev
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Candle, ExternalDataEvent, Feature, NewsArticle, NewsSentiment, TrainingFeature
from app.features.schema import CURRENT_FEATURE_SCHEMA_VERSION, feature_payload


def _safe_pct_change(current: float, previous: float) -> float:
    if previous == 0:
        return 0.0
    return (current - previous) / previous


def _trend_from_change(price_change: float, volatility: float) -> str:
    threshold = max(volatility * 0.5, 0.001)
    if price_change > threshold:
        return "up"
    if price_change < -threshold:
        return "down"
    return "sideways"


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _recency_weight(value: datetime | None, now: datetime, horizon_hours: float = 48.0) -> float:
    timestamp = _aware(value)
    if timestamp is None:
        return 0.25
    age_hours = max((now - timestamp).total_seconds() / 3600.0, 0.0)
    return _clamp(1.0 - (age_hours / horizon_hours), 0.0, 1.0)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sma(values: list[float], period: int) -> float:
    window = values[-period:] if len(values) >= period else values
    return mean(window) if window else 0.0


def _ema_series(values: list[float], period: int) -> list[float]:
    if not values:
        return []
    alpha = 2.0 / (period + 1.0)
    output = [float(values[0])]
    for value in values[1:]:
        output.append((float(value) * alpha) + (output[-1] * (1.0 - alpha)))
    return output


def _rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) <= period:
        return 0.5
    deltas = [closes[i] - closes[i - 1] for i in range(len(closes) - period, len(closes))]
    gains = [max(delta, 0.0) for delta in deltas]
    losses = [abs(min(delta, 0.0)) for delta in deltas]
    average_gain = mean(gains) if gains else 0.0
    average_loss = mean(losses) if losses else 0.0
    if average_loss == 0.0:
        return 1.0 if average_gain > 0.0 else 0.5
    rs = average_gain / average_loss
    return _clamp(1.0 - (1.0 / (1.0 + rs)), 0.0, 1.0)


def _true_ranges(highs: list[float], lows: list[float], closes: list[float]) -> list[float]:
    ranges: list[float] = []
    for index, high in enumerate(highs):
        low = lows[index] if index < len(lows) else high
        if index == 0 or index - 1 >= len(closes):
            ranges.append(max(high - low, 0.0))
            continue
        previous_close = closes[index - 1]
        ranges.append(max(high - low, abs(high - previous_close), abs(low - previous_close), 0.0))
    return ranges


def _adx(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> float:
    if len(highs) <= period or len(lows) <= period or len(closes) <= period:
        return 0.0
    true_ranges = _true_ranges(highs, lows, closes)
    dx_values: list[float] = []
    start = max(1, len(highs) - (period * 2))
    for end in range(start + period, len(highs) + 1):
        plus_dm = 0.0
        minus_dm = 0.0
        tr_sum = sum(true_ranges[end - period : end])
        if tr_sum <= 0.0:
            continue
        for index in range(end - period + 1, end):
            up_move = highs[index] - highs[index - 1]
            down_move = lows[index - 1] - lows[index]
            if up_move > down_move and up_move > 0.0:
                plus_dm += up_move
            if down_move > up_move and down_move > 0.0:
                minus_dm += down_move
        plus_di = 100.0 * plus_dm / tr_sum
        minus_di = 100.0 * minus_dm / tr_sum
        total = plus_di + minus_di
        if total > 0.0:
            dx_values.append(abs(plus_di - minus_di) / total)
    return _clamp(mean(dx_values[-period:]) if dx_values else 0.0, 0.0, 1.0)


def _technical_indicator_features(candles: list[Candle]) -> dict[str, float]:
    closes = [float(candle.close or 0.0) for candle in candles]
    highs = [float(candle.high or 0.0) for candle in candles]
    lows = [float(candle.low or 0.0) for candle in candles]
    volumes = [float(candle.volume or 0.0) for candle in candles]
    last_close = closes[-1] if closes else 0.0
    if last_close <= 0.0:
        return {
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
        }

    ema_12 = _ema_series(closes, 12)
    ema_26 = _ema_series(closes, 26)
    macd_series = [(fast - slow) for fast, slow in zip(ema_12[-len(ema_26) :], ema_26)] if ema_12 and ema_26 else []
    macd_value = macd_series[-1] if macd_series else 0.0
    signal_series = _ema_series(macd_series, 9)
    signal_value = signal_series[-1] if signal_series else 0.0
    histogram_value = macd_value - signal_value

    sma_20 = _sma(closes, 20)
    ema_20_series = _ema_series(closes, 20)
    ema_20 = ema_20_series[-1] if ema_20_series else sma_20
    bollinger_window = closes[-20:] if len(closes) >= 20 else closes
    bollinger_mid = mean(bollinger_window) if bollinger_window else last_close
    bollinger_std = pstdev(bollinger_window) if len(bollinger_window) > 1 else 0.0
    bollinger_upper = bollinger_mid + (bollinger_std * 2.0)
    bollinger_lower = bollinger_mid - (bollinger_std * 2.0)
    bollinger_range = bollinger_upper - bollinger_lower
    bollinger_position = ((last_close - bollinger_lower) / bollinger_range) if bollinger_range > 0.0 else 0.5

    true_ranges = _true_ranges(highs, lows, closes)
    atr_14 = mean(true_ranges[-14:]) if true_ranges else 0.0
    vwap_window = candles[-20:] if len(candles) >= 20 else candles
    vwap_numerator = sum(((row.high + row.low + row.close) / 3.0) * max(row.volume or 0.0, 0.0) for row in vwap_window)
    vwap_denominator = sum(max(row.volume or 0.0, 0.0) for row in vwap_window)
    vwap_20 = vwap_numerator / vwap_denominator if vwap_denominator > 0.0 else last_close

    return {
        "rsi_14": _rsi(closes),
        "macd_pct": macd_value / last_close,
        "macd_signal_pct": signal_value / last_close,
        "macd_histogram_pct": histogram_value / last_close,
        "sma_20_distance_pct": _safe_pct_change(last_close, sma_20) if sma_20 else 0.0,
        "ema_20_distance_pct": _safe_pct_change(last_close, ema_20) if ema_20 else 0.0,
        "bollinger_width_pct": (bollinger_range / bollinger_mid) if bollinger_mid else 0.0,
        "bollinger_position": _clamp(bollinger_position, 0.0, 1.0),
        "atr_14_pct": atr_14 / last_close,
        "vwap_20_distance_pct": _safe_pct_change(last_close, vwap_20) if vwap_20 else 0.0,
        "adx_14": _adx(highs, lows, closes),
    }


def _time_context_features(value: datetime | None) -> dict[str, float]:
    timestamp = _aware(value) or datetime.now(timezone.utc)
    hour = timestamp.hour + (timestamp.minute / 60.0)
    day = float(timestamp.weekday())
    hour_angle = (hour / 24.0) * math.tau
    day_angle = (day / 7.0) * math.tau
    return {
        "time_hour_utc_sin": math.sin(hour_angle),
        "time_hour_utc_cos": math.cos(hour_angle),
        "time_day_of_week_sin": math.sin(day_angle),
        "time_day_of_week_cos": math.cos(day_angle),
        "time_is_weekend": 1.0 if timestamp.weekday() >= 5 else 0.0,
        "session_asia": 1.0 if 0 <= timestamp.hour < 8 else 0.0,
        "session_london": 1.0 if 7 <= timestamp.hour < 16 else 0.0,
        "session_new_york": 1.0 if 13 <= timestamp.hour < 22 else 0.0,
    }


class FeatureBuilder:
    def __init__(self, session: Session) -> None:
        self.session = session

    def build_for_symbol(
        self,
        symbol: str,
        lookback: int = 60,
        store: bool = True,
        interval: str | None = None,
    ) -> Feature:
        normalized_symbol = symbol.upper()
        candle_interval = interval or settings.paper_trade_timeframe
        query = (
            select(Candle)
            .where(Candle.symbol == normalized_symbol)
            .order_by(desc(Candle.open_time))
            .limit(lookback)
        )
        if candle_interval:
            query = (
                select(Candle)
                .where(Candle.symbol == normalized_symbol, Candle.interval == candle_interval, Candle.is_closed.is_(True))
                .order_by(desc(Candle.open_time))
                .limit(lookback)
            )
        candles = list(self.session.scalars(query))
        training_quality_candles = True
        if len(candles) < 2 and candle_interval:
            training_quality_candles = False
            fallback_query = (
                select(Candle)
                .where(Candle.symbol == normalized_symbol, Candle.interval == candle_interval)
                .order_by(desc(Candle.open_time))
                .limit(lookback)
            )
            candles = list(self.session.scalars(fallback_query))
        candles.reverse()

        closes = [candle.close for candle in candles]
        volumes = [candle.volume for candle in candles]
        returns = [_safe_pct_change(closes[i], closes[i - 1]) for i in range(1, len(closes))]
        price_change = _safe_pct_change(closes[-1], closes[0]) if len(closes) >= 2 else 0.0
        candle_return_1m = _safe_pct_change(closes[-1], closes[-2]) if len(closes) >= 2 else 0.0
        candle_return_5m = _safe_pct_change(closes[-1], closes[-6]) if len(closes) >= 6 else price_change
        volatility = pstdev(returns) if len(returns) > 1 else 0.0

        if len(volumes) >= 4:
            midpoint = len(volumes) // 2
            earlier_volume = mean(volumes[:midpoint])
            recent_volume = mean(volumes[midpoint:])
            volume_change = _safe_pct_change(recent_volume, earlier_volume)
        else:
            volume_change = 0.0

        now = datetime.now(timezone.utc)
        news_features = self._recent_news_features(normalized_symbol, now=now)
        derivatives_features = self._recent_derivatives_features(normalized_symbol, now=now)
        external_features = self._recent_external_context_features(normalized_symbol, now=now)
        technical_features = _technical_indicator_features(candles)
        time_features = _time_context_features((candles[-1].close_time or candles[-1].open_time) if candles else now)
        sentiment_score = news_features["sentiment_score"]
        sentiment_confidence = news_features["sentiment_confidence"]
        risk_score = news_features["risk_score"]
        impact_score = news_features["impact_score"]
        recency_weight = news_features["recency_weight"]
        sentiment_count = int(news_features["sentiment_articles_used"])
        trend = _trend_from_change(price_change, volatility)
        trend_score = _clamp(candle_return_5m / max(volatility * 3.0, 0.001), -1.0, 1.0)

        values: dict[str, Any] = {
            "price_change": price_change,
            "volume_change": volume_change,
            "volatility": volatility,
            "trend": trend,
            "trend_score": trend_score,
            "sentiment_score": sentiment_score,
            "sentiment_confidence": sentiment_confidence,
            "risk_score": risk_score,
            "impact_score": impact_score,
            "recency_weight": recency_weight,
            "btc_related": news_features["btc_related"],
            "eth_related": news_features["eth_related"],
            "macro_related": news_features["macro_related"],
            "candle_return_1m": candle_return_1m,
            "candle_return_5m": candle_return_5m,
            **technical_features,
            **time_features,
            "last_close": closes[-1] if closes else None,
            "candles_used": len(candles),
            "sentiment_articles_used": sentiment_count,
            "crowd_long_account_pct": derivatives_features["crowd_long_account_pct"],
            "crowd_short_account_pct": derivatives_features["crowd_short_account_pct"],
            "crowd_long_short_ratio": derivatives_features["crowd_long_short_ratio"],
            "top_trader_long_account_pct": derivatives_features["top_trader_long_account_pct"],
            "top_trader_position_long_pct": derivatives_features["top_trader_position_long_pct"],
            "taker_buy_pressure": derivatives_features["taker_buy_pressure"],
            "taker_buy_sell_ratio": derivatives_features["taker_buy_sell_ratio"],
            "open_interest_value": derivatives_features["open_interest_value"],
            "open_interest_change": derivatives_features["open_interest_change"],
            "funding_rate": derivatives_features["funding_rate"],
            "trader_crowd_score": derivatives_features["trader_crowd_score"],
            "crowd_risk_score": derivatives_features["crowd_risk_score"],
            "derivatives_recency_weight": derivatives_features["derivatives_recency_weight"],
            "spot_price_change_24h": external_features["spot_price_change_24h"],
            "spot_volume_24h": external_features["spot_volume_24h"],
            "spot_quote_volume_24h": external_features["spot_quote_volume_24h"],
            "spot_trade_count_24h": external_features["spot_trade_count_24h"],
            "spot_weighted_avg_price": external_features["spot_weighted_avg_price"],
            "spot_bid_ask_spread_pct": external_features["spot_bid_ask_spread_pct"],
            "spot_orderbook_imbalance": external_features["spot_orderbook_imbalance"],
            "spot_depth_bid_notional_5": external_features["spot_depth_bid_notional_5"],
            "spot_depth_ask_notional_5": external_features["spot_depth_ask_notional_5"],
            "spot_intraday_range_pct": external_features["spot_intraday_range_pct"],
            "spot_activity_score": external_features["spot_activity_score"],
            "fear_greed_value": external_features["fear_greed_value"],
            "fear_greed_change_1d": external_features["fear_greed_change_1d"],
            "fear_greed_change_24h": external_features["fear_greed_change_24h"],
            "fear_greed_classification": external_features["fear_greed_classification"],
            "total_market_cap_usd": external_features["total_market_cap_usd"],
            "market_cap_change_24h": external_features["market_cap_change_24h"],
            "global_market_cap_change_24h": external_features["global_market_cap_change_24h"],
            "total_volume_usd": external_features["total_volume_usd"],
            "total_volume_change_24h": external_features["total_volume_change_24h"],
            "btc_dominance": external_features["btc_dominance"],
            "btc_dominance_change": external_features["btc_dominance_change"],
            "btc_dominance_change_24h": external_features["btc_dominance_change_24h"],
            "eth_dominance": external_features["eth_dominance"],
            "liquidation_long_usd_1m": external_features["liquidation_long_usd_1m"],
            "liquidation_short_usd_1m": external_features["liquidation_short_usd_1m"],
            "liquidation_long_usd_5m": external_features["liquidation_long_usd_5m"],
            "liquidation_short_usd_5m": external_features["liquidation_short_usd_5m"],
            "liquidation_total_usd_5m": external_features["liquidation_total_usd_5m"],
            "liquidation_imbalance_5m": external_features["liquidation_imbalance_5m"],
            "liquidation_spike_score": external_features["liquidation_spike_score"],
            "usdt_deviation": external_features["usdt_deviation"],
            "usdc_deviation": external_features["usdc_deviation"],
            "usdt_price_deviation": external_features["usdt_price_deviation"],
            "usdc_price_deviation": external_features["usdc_price_deviation"],
            "stablecoin_depeg_risk": external_features["stablecoin_depeg_risk"],
            "stablecoin_supply_change_1d": external_features["stablecoin_supply_change_1d"],
            "stablecoin_supply_change_24h": external_features["stablecoin_supply_change_24h"],
            "macro_risk_score": external_features["macro_risk_score"],
            "regulation_risk_score": external_features["regulation_risk_score"],
            "fed_risk_score": external_features["fed_risk_score"],
            "war_risk_score": external_features["war_risk_score"],
            "exchange_hack_risk_score": external_features["exchange_hack_risk_score"],
            "etf_positive_score": external_features["etf_positive_score"],
            "security_risk_score": external_features["security_risk_score"],
            "etf_bullish_score": external_features["etf_bullish_score"],
            "world_risk_score": external_features["world_risk_score"],
            "market_regime_score": external_features["market_regime_score"],
        }
        inspector_vector = {
            key: values[key]
            for key in (
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
                "time_hour_utc_sin",
                "time_hour_utc_cos",
                "time_day_of_week_sin",
                "time_day_of_week_cos",
                "time_is_weekend",
                "session_asia",
                "session_london",
                "session_new_york",
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
            )
        }
        values["final_ai_input"] = {
            "schema_version": CURRENT_FEATURE_SCHEMA_VERSION,
            "symbol": normalized_symbol,
            "timeframe": candle_interval,
            "vector": inspector_vector,
            "strategy_input": {
                "price_change": price_change,
                "sentiment_score": sentiment_score,
                "risk_score": risk_score,
                "volatility": volatility,
                "trend": trend,
                "rsi_14": technical_features["rsi_14"],
                "atr_14_pct": technical_features["atr_14_pct"],
                "adx_14": technical_features["adx_14"],
                "vwap_20_distance_pct": technical_features["vwap_20_distance_pct"],
                "trader_crowd_score": derivatives_features["trader_crowd_score"],
                "crowd_risk_score": derivatives_features["crowd_risk_score"],
                "taker_buy_pressure": derivatives_features["taker_buy_pressure"],
                "spot_activity_score": external_features["spot_activity_score"],
                "spot_orderbook_imbalance": external_features["spot_orderbook_imbalance"],
                "spot_bid_ask_spread_pct": external_features["spot_bid_ask_spread_pct"],
                "market_regime_score": external_features["market_regime_score"],
                "macro_risk_score": external_features["macro_risk_score"],
                "stablecoin_depeg_risk": external_features["stablecoin_depeg_risk"],
            },
        }
        payload = feature_payload(
            schema_version=CURRENT_FEATURE_SCHEMA_VERSION,
            values=values,
            metadata={
                "lookback": lookback,
                "interval": candle_interval,
                "returns_used": len(returns),
                "missing_future_features_default": "0/null",
                "technical_indicator_periods": {
                    "rsi": 14,
                    "macd": "12/26/9",
                    "atr": 14,
                    "vwap": 20,
                    "bollinger": 20,
                    "adx": 14,
                },
                "news_context": news_features["news_context"],
                "derivatives_context": derivatives_features["derivatives_context"],
                "external_context": external_features["external_context"],
                "source_freshness": external_features["source_freshness"],
                "stale_sources": external_features["stale_sources"],
                "training_quality_candles": training_quality_candles,
                "closed_candles_used": len([candle for candle in candles if candle.is_closed]),
                "live_candles_used": len([candle for candle in candles if not candle.is_closed]),
            },
            sources={
                "candles": "candles",
                "news_sentiment": "news_sentiment",
                "derivatives": "external_data_events",
                "market_context": "external_data_events",
            },
        )
        feature = Feature(
            symbol=normalized_symbol,
            schema_version=CURRENT_FEATURE_SCHEMA_VERSION,
            source_name="feature_builder",
            as_of=now,
            price_change=price_change,
            volume_change=volume_change,
            volatility=volatility,
            trend=trend,
            sentiment_score=sentiment_score,
            risk_score=risk_score,
            payload=payload,
            raw_payload=payload,
        )
        if store:
            self.session.add(feature)
            self.session.flush()
            training_values = dict(payload.get("values", {}))
            training_values.pop("final_ai_input", None)
            training_metadata = dict(payload.get("metadata", {}))
            training_metadata.pop("news_context", None)
            training_metadata.pop("derivatives_context", None)
            training_metadata.pop("external_context", None)
            training_metadata["debug_payload"] = "full final_ai_input kept on recent features rows only"
            training_payload = {
                **payload,
                "values": training_values,
                "metadata": training_metadata,
            }
            self.session.add(
                TrainingFeature(
                    source_feature_id=feature.id,
                    symbol=feature.symbol,
                    schema_version=feature.schema_version,
                    source_name=feature.source_name,
                    as_of=feature.as_of,
                    feature_values=training_values,
                    payload=training_payload,
                )
            )
            self.session.commit()
            self.session.refresh(feature)
        return feature

    def _recent_derivatives_features(self, symbol: str, now: datetime) -> dict[str, Any]:
        since = now - timedelta(hours=24)
        rows = list(
            self.session.scalars(
                select(ExternalDataEvent)
                .where(
                    ExternalDataEvent.symbol == symbol,
                    ExternalDataEvent.source_name.like("binance_futures_%"),
                    ExternalDataEvent.event_time >= since,
                )
                .order_by(desc(ExternalDataEvent.event_time))
                .limit(100)
            )
        )
        latest_by_type: dict[str, ExternalDataEvent] = {}
        for row in rows:
            latest_by_type.setdefault(row.data_type, row)

        def payload_value(data_type: str, key: str, default: float = 0.0) -> float:
            row = latest_by_type.get(data_type)
            if row is None:
                return default
            return _float((row.payload or {}).get(key), default)

        crowd_long_account_pct = payload_value("global_long_short_account_ratio", "long_account_pct")
        crowd_short_account_pct = payload_value("global_long_short_account_ratio", "short_account_pct")
        crowd_long_short_ratio = payload_value("global_long_short_account_ratio", "long_short_ratio")
        top_trader_long_account_pct = payload_value("top_long_short_account_ratio", "long_account_pct")
        top_trader_position_long_pct = payload_value("top_long_short_position_ratio", "long_account_pct")
        taker_buy_pressure = payload_value("taker_buy_sell_volume", "buy_pressure")
        taker_buy_sell_ratio = payload_value("taker_buy_sell_volume", "buy_sell_ratio")
        open_interest_value = payload_value("open_interest_hist", "sum_open_interest_value")
        if open_interest_value == 0.0:
            open_interest_value = payload_value("open_interest", "open_interest")
        funding_rate = payload_value("funding_rate", "funding_rate")

        oi_rows = [
            row
            for row in rows
            if row.data_type in {"open_interest_hist", "open_interest"} and row.numeric_value is not None
        ]
        oi_rows.sort(key=lambda item: item.event_time)
        open_interest_change = 0.0
        if len(oi_rows) >= 2:
            open_interest_change = _safe_pct_change(float(oi_rows[-1].numeric_value or 0.0), float(oi_rows[-2].numeric_value or 0.0))

        biases: list[float] = []
        if crowd_long_account_pct or crowd_short_account_pct:
            biases.append(_clamp((crowd_long_account_pct - crowd_short_account_pct) * 2.0, -1.0, 1.0))
        if top_trader_long_account_pct:
            biases.append(_clamp((top_trader_long_account_pct - 0.5) * 2.0, -1.0, 1.0))
        if top_trader_position_long_pct:
            biases.append(_clamp((top_trader_position_long_pct - 0.5) * 2.0, -1.0, 1.0))
        if taker_buy_pressure:
            biases.append(_clamp((taker_buy_pressure - 0.5) * 2.0, -1.0, 1.0))
        trader_crowd_score = mean(biases) if biases else 0.0

        funding_pressure = _clamp(funding_rate / 0.0005, -1.0, 1.0) if funding_rate else 0.0
        crowding = max(abs(item) for item in biases) if biases else 0.0
        crowd_risk_score = _clamp((crowding * 0.45) + (abs(funding_pressure) * 0.35) + (abs(open_interest_change) * 8.0 * 0.20), 0.0, 1.0)

        recencies = [_recency_weight(row.event_time, now, horizon_hours=24.0) for row in latest_by_type.values()]
        derivatives_recency_weight = mean(recencies) if recencies else 0.0
        return {
            "crowd_long_account_pct": crowd_long_account_pct,
            "crowd_short_account_pct": crowd_short_account_pct,
            "crowd_long_short_ratio": crowd_long_short_ratio,
            "top_trader_long_account_pct": top_trader_long_account_pct,
            "top_trader_position_long_pct": top_trader_position_long_pct,
            "taker_buy_pressure": taker_buy_pressure,
            "taker_buy_sell_ratio": taker_buy_sell_ratio,
            "open_interest_value": open_interest_value,
            "open_interest_change": _clamp(open_interest_change, -1.0, 1.0),
            "funding_rate": funding_rate,
            "trader_crowd_score": _clamp(trader_crowd_score, -1.0, 1.0),
            "crowd_risk_score": crowd_risk_score,
            "derivatives_recency_weight": _clamp(derivatives_recency_weight, 0.0, 1.0),
            "derivatives_context": [
                {
                    "source_name": row.source_name,
                    "data_type": row.data_type,
                    "event_time": row.event_time.isoformat() if row.event_time else None,
                    "numeric_value": row.numeric_value,
                    "payload": row.payload,
                }
                for row in latest_by_type.values()
            ],
        }

    def _recent_external_context_features(self, symbol: str, now: datetime) -> dict[str, Any]:
        since = now - timedelta(hours=48)
        rows = list(
            self.session.scalars(
                select(ExternalDataEvent)
                .where(
                    ExternalDataEvent.event_time >= since,
                    ExternalDataEvent.source_name.in_(
                        [
                            "alternative_me_fear_greed",
                            "coingecko_global_market",
                            "binance_spot_context",
                            "binance_futures_liquidations",
                            "defillama_stablecoin_risk",
                            "macro_risk_news",
                        ]
                    ),
                )
                .order_by(desc(ExternalDataEvent.event_time))
                .limit(400)
            )
        )

        latest_by_type: dict[tuple[str, str | None], ExternalDataEvent] = {}
        for row in rows:
            key = (row.data_type, row.symbol)
            latest_by_type.setdefault(key, row)
            latest_by_type.setdefault((row.data_type, None), row)

        def value(data_type: str, default: float = 0.0, *, symbol_first: bool = False) -> float:
            row = latest_by_type.get((data_type, symbol)) if symbol_first else None
            row = row or latest_by_type.get((data_type, None))
            return float(row.numeric_value) if row and row.numeric_value is not None else default

        def latest_time(data_type: str, *, symbol_first: bool = False) -> datetime | None:
            row = latest_by_type.get((data_type, symbol)) if symbol_first else None
            row = row or latest_by_type.get((data_type, None))
            return _aware(row.event_time) if row and row.event_time else None

        def freshness(source_name: str, data_types: list[str], *, symbol_first: bool = False, stale_after_hours: float = 6.0) -> dict[str, Any]:
            times = [latest_time(data_type, symbol_first=symbol_first) for data_type in data_types]
            latest = max([item for item in times if item is not None], default=None)
            age_hours = ((now - latest).total_seconds() / 3600.0) if latest else None
            return {
                "source_name": source_name,
                "latest_at": latest.isoformat() if latest else None,
                "age_hours": age_hours,
                "stale": True if age_hours is None else age_hours > stale_after_hours,
            }

        fear_greed_value = value("fear_greed_value")
        fear_greed_change_1d = value("fear_greed_change_1d")
        fear_greed_change_24h = value("fear_greed_change_24h", fear_greed_change_1d)
        fear_greed_classification = value("fear_greed_classification", value("fear_greed_classification_score"))
        total_market_cap_usd = value("total_market_cap_usd", value("global_market_cap_usd"))
        market_cap_change_24h = value("market_cap_change_24h", value("global_market_cap_change_24h"))
        global_market_cap_change_24h = value("global_market_cap_change_24h", market_cap_change_24h)
        total_volume_usd = value("total_volume_usd")
        total_volume_change_24h = value("total_volume_change_24h")
        btc_dominance = value("btc_dominance")
        btc_dominance_change = value("btc_dominance_change")
        btc_dominance_change_24h = value("btc_dominance_change_24h", btc_dominance_change)
        eth_dominance = value("eth_dominance")
        spot_price_change_24h = value("spot_price_change_24h", symbol_first=True)
        spot_volume_24h = value("spot_volume_24h", symbol_first=True)
        spot_quote_volume_24h = value("spot_quote_volume_24h", symbol_first=True)
        spot_trade_count_24h = value("spot_trade_count_24h", symbol_first=True)
        spot_weighted_avg_price = value("spot_weighted_avg_price", symbol_first=True)
        spot_bid_ask_spread_pct = value("spot_bid_ask_spread_pct", symbol_first=True)
        spot_orderbook_imbalance = value("spot_orderbook_imbalance", symbol_first=True)
        spot_depth_bid_notional_5 = value("spot_depth_bid_notional_5", symbol_first=True)
        spot_depth_ask_notional_5 = value("spot_depth_ask_notional_5", symbol_first=True)
        spot_intraday_range_pct = value("spot_intraday_range_pct", symbol_first=True)
        spot_activity_score = value("spot_activity_score", symbol_first=True)
        liquidation_long_usd_1m = value("liquidation_long_usd_1m", symbol_first=True)
        liquidation_short_usd_1m = value("liquidation_short_usd_1m", symbol_first=True)
        liquidation_long_usd_5m = value("liquidation_long_usd_5m", symbol_first=True)
        liquidation_short_usd_5m = value("liquidation_short_usd_5m", symbol_first=True)
        liquidation_total_usd_5m = value("liquidation_total_usd_5m", symbol_first=True)
        liquidation_imbalance_5m = value("liquidation_imbalance_5m", symbol_first=True)
        liquidation_spike_score = value("liquidation_spike_score", symbol_first=True)
        usdt_price_deviation = value("usdt_price_deviation", value("usdt_deviation"))
        usdc_price_deviation = value("usdc_price_deviation", value("usdc_deviation"))
        usdt_deviation = value("usdt_deviation", usdt_price_deviation)
        usdc_deviation = value("usdc_deviation", usdc_price_deviation)
        stablecoin_depeg_risk = value("stablecoin_depeg_risk")
        stablecoin_supply_change_1d = value("stablecoin_supply_change_1d")
        stablecoin_supply_change_24h = value("stablecoin_supply_change_24h", stablecoin_supply_change_1d)
        macro_risk_score = value("macro_risk_score")
        regulation_risk_score = value("regulation_risk_score")
        fed_risk_score = value("fed_risk_score")
        war_risk_score = value("war_risk_score")
        exchange_hack_risk_score = value("exchange_hack_risk_score")
        etf_positive_score = value("etf_positive_score", value("etf_bullish_score"))
        security_risk_score = value("security_risk_score")
        etf_bullish_score = value("etf_bullish_score")
        world_risk_score = value("world_risk_score")

        fear_greed_bias = _clamp((fear_greed_value - 50.0) / 50.0, -1.0, 1.0) if fear_greed_value else 0.0
        growth_bias = _clamp(global_market_cap_change_24h * 8.0 + total_volume_change_24h * 2.0, -1.0, 1.0)
        risk_drag = _clamp(
            (stablecoin_depeg_risk * 0.35)
            + (macro_risk_score * 0.20)
            + (regulation_risk_score * 0.15)
            + (security_risk_score * 0.15)
            + (world_risk_score * 0.15),
            0.0,
            1.0,
        )
        liquidation_drag = _clamp(liquidation_spike_score / 5.0, 0.0, 1.0)
        spot_liquidity_bias = _clamp((spot_activity_score * 0.5) - (spot_bid_ask_spread_pct * 100.0), -0.5, 0.5)
        market_regime_score = _clamp(
            fear_greed_bias * 0.25
            + growth_bias * 0.25
            + spot_liquidity_bias * 0.10
            + etf_bullish_score * 0.20
            - risk_drag * 0.35
            - liquidation_drag * 0.10,
            -1.0,
            1.0,
        )
        return {
            "fear_greed_value": fear_greed_value,
            "fear_greed_change_1d": fear_greed_change_1d,
            "fear_greed_change_24h": fear_greed_change_24h,
            "fear_greed_classification": fear_greed_classification,
            "total_market_cap_usd": total_market_cap_usd,
            "market_cap_change_24h": market_cap_change_24h,
            "global_market_cap_change_24h": global_market_cap_change_24h,
            "total_volume_usd": total_volume_usd,
            "total_volume_change_24h": total_volume_change_24h,
            "btc_dominance": btc_dominance,
            "btc_dominance_change": btc_dominance_change,
            "btc_dominance_change_24h": btc_dominance_change_24h,
            "eth_dominance": eth_dominance,
            "spot_price_change_24h": spot_price_change_24h,
            "spot_volume_24h": spot_volume_24h,
            "spot_quote_volume_24h": spot_quote_volume_24h,
            "spot_trade_count_24h": spot_trade_count_24h,
            "spot_weighted_avg_price": spot_weighted_avg_price,
            "spot_bid_ask_spread_pct": spot_bid_ask_spread_pct,
            "spot_orderbook_imbalance": spot_orderbook_imbalance,
            "spot_depth_bid_notional_5": spot_depth_bid_notional_5,
            "spot_depth_ask_notional_5": spot_depth_ask_notional_5,
            "spot_intraday_range_pct": spot_intraday_range_pct,
            "spot_activity_score": spot_activity_score,
            "liquidation_long_usd_1m": liquidation_long_usd_1m,
            "liquidation_short_usd_1m": liquidation_short_usd_1m,
            "liquidation_long_usd_5m": liquidation_long_usd_5m,
            "liquidation_short_usd_5m": liquidation_short_usd_5m,
            "liquidation_total_usd_5m": liquidation_total_usd_5m,
            "liquidation_imbalance_5m": liquidation_imbalance_5m,
            "liquidation_spike_score": liquidation_spike_score,
            "usdt_deviation": usdt_deviation,
            "usdc_deviation": usdc_deviation,
            "usdt_price_deviation": usdt_price_deviation,
            "usdc_price_deviation": usdc_price_deviation,
            "stablecoin_depeg_risk": stablecoin_depeg_risk,
            "stablecoin_supply_change_1d": stablecoin_supply_change_1d,
            "stablecoin_supply_change_24h": stablecoin_supply_change_24h,
            "macro_risk_score": macro_risk_score,
            "regulation_risk_score": regulation_risk_score,
            "fed_risk_score": fed_risk_score,
            "war_risk_score": war_risk_score,
            "exchange_hack_risk_score": exchange_hack_risk_score,
            "etf_positive_score": etf_positive_score,
            "security_risk_score": security_risk_score,
            "etf_bullish_score": etf_bullish_score,
            "world_risk_score": world_risk_score,
            "market_regime_score": market_regime_score,
            "external_context": [
                {
                    "source_name": row.source_name,
                    "data_type": row.data_type,
                    "symbol": row.symbol,
                    "event_time": row.event_time.isoformat() if row.event_time else None,
                    "numeric_value": row.numeric_value,
                    "payload": row.payload,
                }
                for row in rows[:20]
            ],
            "source_freshness": {
                "fear_greed": freshness("alternative_me_fear_greed", ["fear_greed_value"], stale_after_hours=30.0),
                "global_market": freshness("coingecko_global_market", ["global_market_cap_change_24h", "market_cap_change_24h"], stale_after_hours=12.0),
                "spot_context": freshness("binance_spot_context", ["spot_activity_score"], symbol_first=True, stale_after_hours=1.0),
                "liquidations": freshness("binance_futures_liquidations", ["liquidation_total_usd_5m"], symbol_first=True, stale_after_hours=1.0),
                "stablecoin_risk": freshness("defillama_stablecoin_risk", ["stablecoin_depeg_risk"], stale_after_hours=30.0),
                "macro_risk": freshness("macro_risk_news", ["macro_risk_score"], stale_after_hours=12.0),
            },
            "stale_sources": [
                name
                for name, item in {
                    "fear_greed": freshness("alternative_me_fear_greed", ["fear_greed_value"], stale_after_hours=30.0),
                    "global_market": freshness("coingecko_global_market", ["global_market_cap_change_24h", "market_cap_change_24h"], stale_after_hours=12.0),
                    "spot_context": freshness("binance_spot_context", ["spot_activity_score"], symbol_first=True, stale_after_hours=1.0),
                    "liquidations": freshness("binance_futures_liquidations", ["liquidation_total_usd_5m"], symbol_first=True, stale_after_hours=1.0),
                    "stablecoin_risk": freshness("defillama_stablecoin_risk", ["stablecoin_depeg_risk"], stale_after_hours=30.0),
                    "macro_risk": freshness("macro_risk_news", ["macro_risk_score"], stale_after_hours=12.0),
                }.items()
                if item["stale"]
            ],
        }

    def _recent_news_features(self, symbol: str, now: datetime) -> dict[str, Any]:
        now = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
        since = now - timedelta(hours=48)
        rows = list(
            self.session.execute(
                select(NewsSentiment, NewsArticle)
                .join(NewsArticle, NewsArticle.id == NewsSentiment.article_id)
                .where(
                    NewsArticle.published_at >= since,
                    NewsArticle.published_at <= now,
                )
                .order_by(desc(NewsArticle.published_at))
                .limit(200)
            )
        )
        relevant: list[tuple[NewsSentiment, NewsArticle, float]] = []
        btc_related = 0.0
        eth_related = 0.0
        macro_related = 0.0
        for sentiment, article in rows:
            affected_symbols = sentiment.affected_symbols or []
            topics = sentiment.topics or []
            text = f"{article.title or ''} {article.raw_text or ''}".lower()
            article_btc_related = "BTCUSDT" in affected_symbols or "bitcoin" in text or " btc" in f" {text}"
            article_eth_related = "ETHUSDT" in affected_symbols or "ethereum" in text or " eth" in f" {text}"
            article_macro_related = "macro" in topics or any(
                term in text for term in ("fed", "federal reserve", "inflation", "interest rate", "rates", "war", "sec")
            )
            btc_related = max(btc_related, 1.0 if article_btc_related else 0.0)
            eth_related = max(eth_related, 1.0 if article_eth_related else 0.0)
            macro_related = max(macro_related, 1.0 if article_macro_related else 0.0)
            if not affected_symbols or symbol in affected_symbols or article_macro_related:
                relevant.append((sentiment, article, _recency_weight(article.published_at or sentiment.created_at, now)))
        if not relevant:
            return {
                "sentiment_score": 0.0,
                "sentiment_confidence": 0.0,
                "risk_score": 0.0,
                "impact_score": 0.0,
                "recency_weight": 0.0,
                "btc_related": btc_related,
                "eth_related": eth_related,
                "macro_related": macro_related,
                "sentiment_articles_used": 0,
                "news_context": [],
            }

        weights = [max(weight, 0.01) * (sentiment.confidence if sentiment.confidence is not None else 0.5) for sentiment, _, weight in relevant]
        total_weight = sum(weights) or 1.0
        sentiment_score = sum(sentiment.sentiment_score * weight for (sentiment, _, _), weight in zip(relevant, weights)) / total_weight
        risk_score = sum(sentiment.risk_score * weight for (sentiment, _, _), weight in zip(relevant, weights)) / total_weight
        recency = mean(weight for _, _, weight in relevant)
        confidence = sum((sentiment.confidence if sentiment.confidence is not None else 0.5) * weight for (sentiment, _, _), weight in zip(relevant, weights)) / total_weight
        impact_score = _clamp((abs(sentiment_score) * confidence * 0.45) + (risk_score * 0.40) + (recency * 0.15), 0.0, 1.0)

        return {
            "sentiment_score": _clamp(sentiment_score, -1.0, 1.0),
            "sentiment_confidence": _clamp(confidence, 0.0, 1.0),
            "risk_score": _clamp(risk_score, 0.0, 1.0),
            "impact_score": impact_score,
            "recency_weight": _clamp(recency, 0.0, 1.0),
            "btc_related": btc_related,
            "eth_related": eth_related,
            "macro_related": macro_related,
            "sentiment_articles_used": len(relevant),
            "news_context": [
                {
                    "title": article.title,
                    "text": article.raw_text or article.title,
                    "source": article.source,
                    "provider": article.source_name,
                    "published_at": article.published_at.isoformat() if article.published_at else None,
                    "sentiment_score": sentiment.sentiment_score,
                    "sentiment_confidence": sentiment.confidence,
                    "risk_score": sentiment.risk_score,
                    "topics": sentiment.topics or [],
                    "affected_symbols": sentiment.affected_symbols or [],
                    "url": article.url,
                }
                for sentiment, article, _ in relevant[:5]
            ],
        }
