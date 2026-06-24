from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from app.collectors.market_collector import BinanceMarketCollector
from app.config import settings
from app.db.session import create_db_and_tables


def _interval_ms(interval: str) -> int:
    unit = interval[-1].lower()
    amount = int(interval[:-1] or "1")
    multipliers = {"s": 1000, "m": 60_000, "h": 3_600_000, "d": 86_400_000}
    if unit not in multipliers:
        raise ValueError(f"Unsupported interval: {interval}")
    return amount * multipliers[unit]


async def backfill_historical_candles(
    *,
    symbols: list[str] | None = None,
    interval: str = "1m",
    days: int = 14,
    max_rows_per_symbol: int = 5000,
    mock: bool = False,
) -> dict[str, Any]:
    create_db_and_tables()
    normalized_symbols = [symbol.upper() for symbol in (symbols or settings.binance_symbols)]
    collector = BinanceMarketCollector(symbols=normalized_symbols, interval=interval)
    if mock:
        per_symbol: dict[str, int] = {}
        for symbol in normalized_symbols:
            rows = collector._mock_klines(symbol, max_rows_per_symbol)
            per_symbol[symbol] = collector.store_rest_klines(symbol, rows)
        return {
            "source": "mock",
            "symbols": normalized_symbols,
            "interval": interval,
            "days": days,
            "max_rows_per_symbol": max_rows_per_symbol,
            "rows_saved": sum(per_symbol.values()),
            "per_symbol": per_symbol,
        }

    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(days=days)
    step_ms = _interval_ms(interval)
    per_symbol = {}
    async with httpx.AsyncClient(timeout=30) as client:
        for symbol in normalized_symbols:
            rows_saved = 0
            next_start_ms = int(start.timestamp() * 1000)
            end_ms = int(end.timestamp() * 1000)
            while next_start_ms < end_ms and rows_saved < max_rows_per_symbol:
                limit = min(1000, max_rows_per_symbol - rows_saved)
                response = await client.get(
                    f"{settings.binance_rest_base_url.rstrip('/')}/api/v3/klines",
                    params={
                        "symbol": symbol,
                        "interval": interval,
                        "startTime": next_start_ms,
                        "endTime": end_ms,
                        "limit": limit,
                    },
                )
                response.raise_for_status()
                klines = response.json()
                if not klines:
                    break
                rows_saved += collector.store_rest_klines(symbol, klines)
                last_open_ms = int(klines[-1][0])
                next_start_ms = last_open_ms + step_ms
                if len(klines) < limit:
                    break
            per_symbol[symbol] = rows_saved
    return {
        "source": "binance_rest",
        "symbols": normalized_symbols,
        "interval": interval,
        "days": days,
        "max_rows_per_symbol": max_rows_per_symbol,
        "rows_saved": sum(per_symbol.values()),
        "per_symbol": per_symbol,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill historical closed candles.")
    parser.add_argument("--symbols", default=",".join(settings.binance_symbols))
    parser.add_argument("--interval", default=settings.paper_trade_timeframe)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--max-rows-per-symbol", type=int, default=5000)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    result = asyncio.run(
        backfill_historical_candles(
            symbols=[item.strip().upper() for item in args.symbols.split(",") if item.strip()],
            interval=args.interval,
            days=args.days,
            max_rows_per_symbol=args.max_rows_per_symbol,
            mock=args.mock,
        )
    )
    print(result)


if __name__ == "__main__":
    main()
