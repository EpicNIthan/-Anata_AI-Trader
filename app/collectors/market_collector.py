from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from sqlalchemy import select

from app.config import settings
from app.db.models import Candle, LiveCandleUpdate, MarketTick
from app.db.session import SessionLocal

try:
    import websockets
except ImportError:  # pragma: no cover - depends on deployment environment
    websockets = None

logger = logging.getLogger(__name__)


def _dt_from_ms(value: int | float) -> datetime:
    return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc)


def format_collector_error(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


class BinanceMarketCollector:
    def __init__(self, symbols: list[str] | None = None, interval: str | None = None) -> None:
        self.symbols = symbols or settings.binance_symbols
        self.interval = interval or settings.binance_interval
        self.store_live_updates = settings.store_live_candle_updates
        self.store_market_ticks = settings.store_market_ticks

    @property
    def subscribed_streams(self) -> list[str]:
        return [f"{symbol.lower()}@kline_{self.interval}" for symbol in self.symbols]

    @property
    def stream_url(self) -> str:
        streams = "/".join(self.subscribed_streams)
        return f"{settings.binance_ws_base_url.rstrip('/')}/stream?streams={streams}"

    async def run(self, stop_event: asyncio.Event, state: Any | None = None) -> None:
        if websockets is None:
            message = "websockets package is not installed"
            logger.error(message)
            if state:
                state.mark_error(message)
            return

        if state:
            state.set_subscription(streams=self.subscribed_streams, websocket_url=self.stream_url)
            state.closed_candles_only = True

        try:
            result = await self.backfill_all(limit=100)
            logger.info("Binance REST backfill completed: %s", result)
            if state:
                state.mark_saved(result["rows_saved"], {"backfill": result, "rows_saved": result["rows_saved"]})
        except Exception as exc:
            logger.exception("Binance REST backfill failed")
            if state:
                state.mark_error(f"Backfill failed: {format_collector_error(exc)}")

        backoff_seconds = 1
        while not stop_event.is_set():
            try:
                logger.info("Connecting to Binance stream: %s", self.stream_url)
                async with websockets.connect(self.stream_url, ping_interval=20, ping_timeout=20) as websocket:
                    backoff_seconds = 1
                    while not stop_event.is_set():
                        raw_message = await asyncio.wait_for(websocket.recv(), timeout=30)
                        payload = json.loads(raw_message)
                        if state:
                            state.mark_message({"websocket_url": self.stream_url})
                        details = self.store_message(payload)
                        if state:
                            state.mark_saved(int(details.get("rows_saved", 0)), details)
                            state.last_error = None
            except asyncio.TimeoutError as exc:
                message = "Binance websocket timeout; no connection/message received before timeout"
                logger.warning("%s: %s", message, self.stream_url)
                if state:
                    state.mark_error(f"{message}: {format_collector_error(exc)}")
                continue
            except Exception as exc:
                logger.exception("Binance market collector error")
                if state:
                    state.mark_error(format_collector_error(exc))
                await asyncio.sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 2, 60)
            finally:
                logger.info("Disconnected from Binance stream: %s", self.stream_url)

    def store_message(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = payload.get("data", payload)
        kline = data.get("k", {})
        if not kline:
            raise ValueError("Binance message did not contain kline data")

        symbol = str(kline["s"]).upper()
        interval = str(kline["i"])
        open_time = _dt_from_ms(kline["t"])
        close_time = _dt_from_ms(kline["T"])
        event_time = _dt_from_ms(data.get("E", kline["T"]))
        close_price = float(kline["c"])
        is_closed = bool(kline.get("x", False))
        candle_saved = False
        candle_created = False
        candle_updated = False
        live_update_saved = False
        live_update_created = False
        live_update_updated = False

        with SessionLocal() as session:
            if self.store_live_updates and not is_closed:
                live_update_saved = True
                live_update = session.scalar(
                    select(LiveCandleUpdate).where(
                        LiveCandleUpdate.exchange == "binance",
                        LiveCandleUpdate.symbol == symbol,
                        LiveCandleUpdate.interval == interval,
                        LiveCandleUpdate.open_time == open_time,
                    )
                )
                if live_update is None:
                    live_update_created = True
                    live_update = LiveCandleUpdate(
                        exchange="binance",
                        source_name="binance_kline_live",
                        symbol=symbol,
                        interval=interval,
                        event_time=event_time,
                        open_time=open_time,
                        close_time=close_time,
                        open=float(kline["o"]),
                        high=float(kline["h"]),
                        low=float(kline["l"]),
                        close=close_price,
                        volume=float(kline["v"]),
                        quote_volume=float(kline.get("q", 0.0)),
                        trades=int(kline.get("n", 0)),
                        update_count=1,
                        raw_payload=payload,
                    )
                    session.add(live_update)
                else:
                    live_update_updated = True
                    live_update.event_time = event_time
                    live_update.close_time = close_time
                    live_update.high = float(kline["h"])
                    live_update.low = float(kline["l"])
                    live_update.close = close_price
                    live_update.volume = float(kline["v"])
                    live_update.quote_volume = float(kline.get("q", 0.0))
                    live_update.trades = int(kline.get("n", 0))
                    live_update.update_count += 1
                    live_update.raw_payload = payload

            if is_closed:
                candle_saved = True
                candle = session.scalar(
                    select(Candle).where(
                        Candle.exchange == "binance",
                        Candle.symbol == symbol,
                        Candle.interval == interval,
                        Candle.open_time == open_time,
                    )
                )
                if candle is None:
                    candle_created = True
                    candle = Candle(
                        exchange="binance",
                        source_name="binance_kline_closed",
                        symbol=symbol,
                        interval=interval,
                        open_time=open_time,
                        close_time=close_time,
                        open=float(kline["o"]),
                        high=float(kline["h"]),
                        low=float(kline["l"]),
                        close=close_price,
                        volume=float(kline["v"]),
                        quote_volume=float(kline.get("q", 0.0)),
                        trades=int(kline.get("n", 0)),
                        is_closed=is_closed,
                        raw=payload,
                        raw_payload=payload,
                    )
                    session.add(candle)
                else:
                    candle_updated = True
                    candle.open = float(kline["o"])
                    candle.close_time = close_time
                    candle.high = float(kline["h"])
                    candle.low = float(kline["l"])
                    candle.close = close_price
                    candle.volume = float(kline["v"])
                    candle.quote_volume = float(kline.get("q", 0.0))
                    candle.trades = int(kline.get("n", 0))
                    candle.is_closed = True
                    candle.source_name = "binance_kline_closed"
                    candle.raw = payload
                    candle.raw_payload = payload
                live_update = session.scalar(
                    select(LiveCandleUpdate).where(
                        LiveCandleUpdate.exchange == "binance",
                        LiveCandleUpdate.symbol == symbol,
                        LiveCandleUpdate.interval == interval,
                        LiveCandleUpdate.open_time == open_time,
                    )
                )
                if live_update:
                    session.delete(live_update)

            tick_saved = False
            if self.store_market_ticks:
                tick_saved = True
                session.add(
                    MarketTick(
                        exchange="binance",
                        source_name="binance_tick",
                        symbol=symbol,
                        event_time=event_time,
                        price=close_price,
                        quantity=float(kline.get("v", 0.0)),
                        raw=payload,
                        raw_payload=payload,
                    )
                )
            session.commit()

        return {
            "symbol": symbol,
            "interval": interval,
            "price": close_price,
            "closed": is_closed,
            "candle_saved": candle_saved,
            "candle_created": candle_created,
            "candle_updated": candle_updated,
            "live_update_saved": live_update_saved,
            "live_update_created": live_update_created,
            "live_update_updated": live_update_updated,
            "live_update_upserted": live_update_saved,
            "training_quality_closed_candle": candle_saved and is_closed,
            "tick_saved": tick_saved,
            "rows_saved": int(candle_saved) + int(live_update_saved) + int(tick_saved),
            "store_live_candle_updates": self.store_live_updates,
            "store_market_ticks": self.store_market_ticks,
            "closed_candles_only": True,
            "live_updates_table": "live_candle_updates",
            "training_candles_table": "candles",
        }

    async def backfill_all(self, limit: int = 100, mock: bool = False) -> dict[str, Any]:
        total_rows = 0
        per_symbol: dict[str, int] = {}
        async with httpx.AsyncClient(timeout=20) as client:
            for symbol in self.symbols:
                if mock:
                    klines = self._mock_klines(symbol, limit)
                else:
                    url = f"{settings.binance_rest_base_url.rstrip('/')}/api/v3/klines"
                    response = await client.get(
                        url,
                        params={"symbol": symbol.upper(), "interval": self.interval, "limit": limit},
                    )
                    response.raise_for_status()
                    klines = response.json()
                saved = self.store_rest_klines(symbol.upper(), klines)
                per_symbol[symbol.upper()] = saved
                total_rows += saved
        return {
            "source": "mock" if mock else "binance_rest",
            "symbols": [symbol.upper() for symbol in self.symbols],
            "interval": self.interval,
            "limit": limit,
            "rows_saved": total_rows,
            "per_symbol": per_symbol,
            "store_live_candle_updates": self.store_live_updates,
            "closed_candles_only": True,
            "live_updates_table": "live_candle_updates",
            "training_candles_table": "candles",
        }

    def store_rest_klines(self, symbol: str, klines: list[list[Any]]) -> int:
        rows_saved = 0
        with SessionLocal() as session:
            for row in klines:
                open_time = _dt_from_ms(row[0])
                close_time = _dt_from_ms(row[6])
                candle = session.scalar(
                    select(Candle).where(
                        Candle.exchange == "binance",
                        Candle.symbol == symbol,
                        Candle.interval == self.interval,
                        Candle.open_time == open_time,
                    )
                )
                payload = {"source": "binance_rest_klines", "row": row}
                if candle is None:
                    candle = Candle(
                        exchange="binance",
                        source_name="binance_rest_klines_closed",
                        symbol=symbol,
                        interval=self.interval,
                        open_time=open_time,
                        close_time=close_time,
                        open=float(row[1]),
                        high=float(row[2]),
                        low=float(row[3]),
                        close=float(row[4]),
                        volume=float(row[5]),
                        quote_volume=float(row[7]),
                        trades=int(row[8]),
                        is_closed=True,
                        raw=payload,
                        raw_payload=payload,
                    )
                    session.add(candle)
                else:
                    candle.close_time = close_time
                    candle.open = float(row[1])
                    candle.high = float(row[2])
                    candle.low = float(row[3])
                    candle.close = float(row[4])
                    candle.volume = float(row[5])
                    candle.quote_volume = float(row[7])
                    candle.trades = int(row[8])
                    candle.is_closed = True
                    candle.source_name = "binance_rest_klines_closed"
                    candle.raw = payload
                    candle.raw_payload = payload
                rows_saved += 1
            session.commit()
        return rows_saved

    def _mock_klines(self, symbol: str, limit: int) -> list[list[Any]]:
        base = 65000.0 if symbol.upper().startswith("BTC") else 3500.0
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        rows: list[list[Any]] = []
        for index in range(limit):
            open_dt = now - timedelta(minutes=limit - index)
            close_dt = open_dt + timedelta(minutes=1) - timedelta(milliseconds=1)
            drift = index * 0.0008
            open_price = base * (1 + drift)
            close_price = open_price * (1 + ((index % 5) - 2) * 0.0004)
            high = max(open_price, close_price) * 1.001
            low = min(open_price, close_price) * 0.999
            volume = 10 + index
            rows.append(
                [
                    int(open_dt.timestamp() * 1000),
                    f"{open_price:.8f}",
                    f"{high:.8f}",
                    f"{low:.8f}",
                    f"{close_price:.8f}",
                    f"{volume:.8f}",
                    int(close_dt.timestamp() * 1000),
                    f"{volume * close_price:.8f}",
                    100 + index,
                    "0",
                    "0",
                    "0",
                ]
            )
        return rows
