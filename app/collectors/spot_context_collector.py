from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import ExternalDataEvent
from app.db.session import SessionLocal

logger = logging.getLogger(__name__)


def _now_minute() -> datetime:
    return datetime.now(timezone.utc).replace(second=0, microsecond=0)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class BinanceSpotContextCollector:
    name = "binance_spot_context"
    source_name = "binance_spot_context"

    def __init__(self, symbols: list[str] | None = None, base_url: str | None = None) -> None:
        self.symbols = [symbol.upper() for symbol in (symbols or settings.spot_context_symbols)]
        self.base_url = (base_url or settings.binance_rest_base_url).rstrip("/")
        self.interval_seconds = max(settings.spot_context_poll_interval_seconds, 60)

    async def run(self, stop_event: asyncio.Event, state: Any | None = None) -> None:
        if not settings.enable_spot_context_collector:
            if state:
                state.warning = "ENABLE_SPOT_CONTEXT_COLLECTOR=false"
                state.running = False
            return
        if state:
            state.details = {"base_url": self.base_url, "symbols": self.symbols, "interval_seconds": self.interval_seconds}
        while not stop_event.is_set():
            try:
                result = await self.fetch_once()
                if state:
                    state.mark_event(result | {"rows_saved": result.get("rows_saved", 0)})
                    state.last_error = None
            except Exception as exc:
                logger.warning("Binance spot context collector failed: %s", exc)
                if state:
                    state.mark_error(f"{type(exc).__name__}: {exc}")
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self.interval_seconds)
            except asyncio.TimeoutError:
                continue

    async def fetch_once(self, *, mock: bool = False) -> dict[str, Any]:
        rows_saved = 0
        per_symbol: dict[str, int] = {}
        urls: list[str] = []
        event_time = _now_minute()
        if mock:
            with SessionLocal() as session:
                for symbol in self.symbols:
                    saved = self._store_symbol(session, symbol, self._mock_ticker(symbol), self._mock_depth(symbol), event_time)
                    per_symbol[symbol] = saved
                    rows_saved += saved
                session.commit()
            return {"source": self.name, "mode": "mock", "symbols": self.symbols, "rows_saved": rows_saved, "per_symbol": per_symbol}

        async with httpx.AsyncClient(timeout=20) as client:
            with SessionLocal() as session:
                for symbol in self.symbols:
                    ticker_url = f"{self.base_url}/api/v3/ticker/24hr"
                    depth_url = f"{self.base_url}/api/v3/depth"
                    urls.extend([ticker_url, depth_url])
                    ticker_response = await client.get(ticker_url, params={"symbol": symbol})
                    ticker_response.raise_for_status()
                    depth_response = await client.get(depth_url, params={"symbol": symbol, "limit": 5})
                    depth_response.raise_for_status()
                    saved = self._store_symbol(session, symbol, ticker_response.json(), depth_response.json(), event_time)
                    per_symbol[symbol] = saved
                    rows_saved += saved
                session.commit()
        return {
            "source": self.name,
            "mode": "live",
            "symbols": self.symbols,
            "rows_saved": rows_saved,
            "per_symbol": per_symbol,
            "urls": urls[-20:],
        }

    def _store_symbol(self, session: Session, symbol: str, ticker: dict[str, Any], depth: dict[str, Any], event_time: datetime) -> int:
        last_price = _float(ticker.get("lastPrice"))
        open_price = _float(ticker.get("openPrice"))
        high_price = _float(ticker.get("highPrice"))
        low_price = _float(ticker.get("lowPrice"))
        volume = _float(ticker.get("volume"))
        quote_volume = _float(ticker.get("quoteVolume"))
        trade_count = _float(ticker.get("count"))
        price_change_pct = _float(ticker.get("priceChangePercent")) / 100.0
        weighted_avg_price = _float(ticker.get("weightedAvgPrice"))
        bid_price = _float((depth.get("bids") or [[0]])[0][0])
        bid_qty = _float((depth.get("bids") or [[0, 0]])[0][1])
        ask_price = _float((depth.get("asks") or [[0]])[0][0])
        ask_qty = _float((depth.get("asks") or [[0, 0]])[0][1])
        mid_price = (bid_price + ask_price) / 2.0 if bid_price and ask_price else last_price
        spread_pct = ((ask_price - bid_price) / mid_price) if mid_price and ask_price and bid_price else 0.0
        depth_bid_notional = sum(_float(price) * _float(qty) for price, qty in (depth.get("bids") or [])[:5])
        depth_ask_notional = sum(_float(price) * _float(qty) for price, qty in (depth.get("asks") or [])[:5])
        depth_total = depth_bid_notional + depth_ask_notional
        depth_imbalance = (depth_bid_notional - depth_ask_notional) / depth_total if depth_total else 0.0
        intraday_range_pct = ((high_price - low_price) / open_price) if open_price else 0.0
        spot_activity_score = _clamp((abs(price_change_pct) * 8.0) + min(quote_volume / 1_000_000_000.0, 1.0) + min(trade_count / 1_000_000.0, 1.0), 0.0, 3.0)
        values = {
            "spot_price_change_24h": price_change_pct,
            "spot_volume_24h": volume,
            "spot_quote_volume_24h": quote_volume,
            "spot_trade_count_24h": trade_count,
            "spot_weighted_avg_price": weighted_avg_price,
            "spot_bid_ask_spread_pct": spread_pct,
            "spot_orderbook_imbalance": _clamp(depth_imbalance, -1.0, 1.0),
            "spot_depth_bid_notional_5": depth_bid_notional,
            "spot_depth_ask_notional_5": depth_ask_notional,
            "spot_intraday_range_pct": intraday_range_pct,
            "spot_activity_score": spot_activity_score,
        }
        saved = 0
        for data_type, numeric_value in values.items():
            saved += int(
                self._upsert_event(
                    session,
                    data_type=data_type,
                    symbol=symbol,
                    event_time=event_time,
                    numeric_value=numeric_value,
                    payload={"mode": "live", "last_price": last_price, "mid_price": mid_price},
                    raw_payload={"ticker": ticker, "depth": depth} if settings.store_raw_external_events else None,
                )
            )
        return saved

    def _upsert_event(
        self,
        session: Session,
        *,
        data_type: str,
        symbol: str,
        event_time: datetime,
        numeric_value: float,
        payload: dict[str, Any],
        raw_payload: dict[str, Any] | None,
    ) -> bool:
        existing = session.scalar(
            select(ExternalDataEvent)
            .where(
                ExternalDataEvent.source_name == self.source_name,
                ExternalDataEvent.data_type == data_type,
                ExternalDataEvent.symbol == symbol,
                ExternalDataEvent.event_time == event_time,
            )
            .limit(1)
        )
        if existing:
            existing.numeric_value = numeric_value
            existing.payload = payload
            existing.raw_payload = raw_payload
            return False
        session.add(
            ExternalDataEvent(
                source_name=self.source_name,
                data_type=data_type,
                symbol=symbol,
                event_time=event_time,
                numeric_value=numeric_value,
                payload=payload,
                raw_payload=raw_payload,
            )
        )
        return True

    def _mock_ticker(self, symbol: str) -> dict[str, Any]:
        price = 65000.0 if symbol.startswith("BTC") else 3500.0
        return {
            "symbol": symbol,
            "lastPrice": str(price),
            "openPrice": str(price * 0.99),
            "highPrice": str(price * 1.01),
            "lowPrice": str(price * 0.985),
            "volume": "10000",
            "quoteVolume": str(price * 10000),
            "count": "500000",
            "priceChangePercent": "1.0",
            "weightedAvgPrice": str(price * 0.998),
        }

    def _mock_depth(self, symbol: str) -> dict[str, Any]:
        price = 65000.0 if symbol.startswith("BTC") else 3500.0
        return {
            "bids": [[str(price * 0.9999), "2.0"], [str(price * 0.9998), "3.0"]],
            "asks": [[str(price * 1.0001), "1.5"], [str(price * 1.0002), "2.5"]],
        }
