from __future__ import annotations

import argparse
from datetime import datetime, timezone
from statistics import mean, pstdev
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Candle, TrainingFeature
from app.db.session import SessionLocal, create_db_and_tables
from app.features.schema import CURRENT_FEATURE_SCHEMA_VERSION, feature_payload


def _safe_pct_change(current: float, previous: float) -> float:
    return (current - previous) / previous if previous else 0.0


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _trend_from_change(price_change: float, volatility: float) -> str:
    threshold = max(volatility * 0.5, 0.001)
    if price_change > threshold:
        return "up"
    if price_change < -threshold:
        return "down"
    return "sideways"


def _horizon_rows(interval: str) -> dict[str, int]:
    if interval.endswith("m"):
        minutes = int(interval[:-1] or "1")
        return {
            "5m": max(1, 5 // minutes),
            "15m": max(1, 15 // minutes),
            "1h": max(1, 60 // minutes),
            "4h": max(1, 240 // minutes),
        }
    return {"5m": 5, "15m": 15, "1h": 60, "4h": 240}


def _existing_training_feature(session: Session, symbol: str, as_of: datetime) -> TrainingFeature | None:
    return session.scalar(
        select(TrainingFeature)
        .where(
            TrainingFeature.symbol == symbol,
            TrainingFeature.as_of == as_of,
            TrainingFeature.source_name == "historical_replay_builder",
        )
        .limit(1)
    )


def _build_values(
    *,
    symbol: str,
    interval: str,
    lookback: list[Candle],
    current_index: int,
    candles: list[Candle],
    horizons: dict[str, int],
) -> dict[str, Any]:
    closes = [row.close for row in lookback]
    volumes = [row.volume for row in lookback]
    returns = [_safe_pct_change(closes[index], closes[index - 1]) for index in range(1, len(closes))]
    price_change = _safe_pct_change(closes[-1], closes[0]) if len(closes) >= 2 else 0.0
    candle_return_1m = _safe_pct_change(closes[-1], closes[-2]) if len(closes) >= 2 else 0.0
    candle_return_5m = _safe_pct_change(closes[-1], closes[-6]) if len(closes) >= 6 else price_change
    volatility = pstdev(returns) if len(returns) > 1 else 0.0
    if len(volumes) >= 4:
        midpoint = len(volumes) // 2
        volume_change = _safe_pct_change(mean(volumes[midpoint:]), mean(volumes[:midpoint]))
    else:
        volume_change = 0.0
    trend = _trend_from_change(price_change, volatility)
    trend_score = _clamp(candle_return_5m / max(volatility * 3.0, 0.001), -1.0, 1.0)
    entry_price = candles[current_index].close
    future_window = candles[current_index + 1 : current_index + 1 + horizons["1h"]]
    max_future_high = max((row.high for row in future_window), default=entry_price)
    min_future_low = min((row.low for row in future_window), default=entry_price)
    stop_loss = entry_price * (1.0 - settings.auto_default_stop_loss_pct)
    take_profit = entry_price * (1.0 + settings.auto_default_take_profit_pct)
    stop_index = next((index for index, row in enumerate(future_window, start=1) if row.low <= stop_loss), None)
    take_index = next((index for index, row in enumerate(future_window, start=1) if row.high >= take_profit), None)
    if stop_index and take_index:
        first_exit = "take_profit" if take_index <= stop_index else "stop_loss"
    elif take_index:
        first_exit = "take_profit"
    elif stop_index:
        first_exit = "stop_loss"
    else:
        first_exit = "none"

    values: dict[str, Any] = {
        "price_change": price_change,
        "volume_change": volume_change,
        "volatility": volatility,
        "trend": trend,
        "trend_score": trend_score,
        "sentiment_score": 0.0,
        "sentiment_confidence": 0.0,
        "risk_score": 0.0,
        "impact_score": 0.0,
        "recency_weight": 0.0,
        "btc_related": 1.0 if symbol == "BTCUSDT" else 0.0,
        "eth_related": 1.0 if symbol == "ETHUSDT" else 0.0,
        "macro_related": 0.0,
        "candle_return_1m": candle_return_1m,
        "candle_return_5m": candle_return_5m,
        "last_close": entry_price,
        "candles_used": len(lookback),
        "sentiment_articles_used": 0,
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
        "target_future_return_5m": _future_return(candles, current_index, horizons["5m"]),
        "target_future_return_15m": _future_return(candles, current_index, horizons["15m"]),
        "target_future_return_1h": _future_return(candles, current_index, horizons["1h"]),
        "target_future_return_4h": _future_return(candles, current_index, horizons["4h"]),
        "target_max_upside_1h": _safe_pct_change(max_future_high, entry_price),
        "target_max_drawdown_1h": _safe_pct_change(min_future_low, entry_price),
        "target_stop_loss_hit_first": 1.0 if first_exit == "stop_loss" else 0.0,
        "target_take_profit_hit_first": 1.0 if first_exit == "take_profit" else 0.0,
    }
    values["final_ai_input"] = {
        "schema_version": CURRENT_FEATURE_SCHEMA_VERSION,
        "symbol": symbol,
        "timeframe": interval,
        "vector": {key: values.get(key, 0.0) for key in values if not key.startswith("target_")},
        "labels": {key: value for key, value in values.items() if key.startswith("target_")},
    }
    return values


def _future_return(candles: list[Candle], current_index: int, horizon: int) -> float:
    target_index = current_index + horizon
    if target_index >= len(candles):
        return 0.0
    return _safe_pct_change(candles[target_index].close, candles[current_index].close)


def build_training_labels(
    *,
    symbols: list[str] | None = None,
    interval: str = "1m",
    lookback: int = 60,
    stride: int = 5,
    max_rows_per_symbol: int = 5000,
) -> dict[str, Any]:
    create_db_and_tables()
    normalized_symbols = [symbol.upper() for symbol in (symbols or settings.binance_symbols)]
    horizons = _horizon_rows(interval)
    per_symbol: dict[str, int] = {}
    with SessionLocal() as session:
        for symbol in normalized_symbols:
            rows = list(
                session.scalars(
                    select(Candle)
                    .where(Candle.symbol == symbol, Candle.interval == interval, Candle.is_closed.is_(True))
                    .order_by(desc(Candle.open_time))
                    .limit(max_rows_per_symbol + lookback + horizons["4h"] + 5)
                )
            )
            candles = list(reversed(rows))
            created = 0
            last_usable_index = len(candles) - horizons["4h"] - 1
            for current_index in range(lookback, max(last_usable_index, lookback), max(stride, 1)):
                current = candles[current_index]
                as_of = current.close_time or current.open_time
                if _existing_training_feature(session, symbol, as_of):
                    continue
                lookback_rows = candles[current_index - lookback : current_index + 1]
                values = _build_values(
                    symbol=symbol,
                    interval=interval,
                    lookback=lookback_rows,
                    current_index=current_index,
                    candles=candles,
                    horizons=horizons,
                )
                payload = feature_payload(
                    schema_version=CURRENT_FEATURE_SCHEMA_VERSION,
                    values=values,
                    metadata={
                        "builder": "historical_replay_builder",
                        "interval": interval,
                        "lookback": lookback,
                        "stride": stride,
                        "label_horizons": horizons,
                        "source_candle_id": current.id,
                    },
                    sources={"candles": "candles", "news_sentiment": "zero_unavailable", "derivatives": "zero_unavailable"},
                )
                session.add(
                    TrainingFeature(
                        source_feature_id=None,
                        symbol=symbol,
                        schema_version=CURRENT_FEATURE_SCHEMA_VERSION,
                        source_name="historical_replay_builder",
                        as_of=as_of,
                        feature_values=payload["values"],
                        payload=payload,
                    )
                )
                created += 1
            per_symbol[symbol] = created
        session.commit()
    return {
        "symbols": normalized_symbols,
        "interval": interval,
        "lookback": lookback,
        "stride": stride,
        "rows_created": sum(per_symbol.values()),
        "per_symbol": per_symbol,
        "schema_version": CURRENT_FEATURE_SCHEMA_VERSION,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact labeled training features from closed candles.")
    parser.add_argument("--symbols", default=",".join(settings.binance_symbols))
    parser.add_argument("--interval", default=settings.paper_trade_timeframe)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-rows-per-symbol", type=int, default=5000)
    args = parser.parse_args()
    result = build_training_labels(
        symbols=[item.strip().upper() for item in args.symbols.split(",") if item.strip()],
        interval=args.interval,
        lookback=args.lookback,
        stride=args.stride,
        max_rows_per_symbol=args.max_rows_per_symbol,
    )
    print(result)


if __name__ == "__main__":
    main()
