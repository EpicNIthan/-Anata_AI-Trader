from __future__ import annotations
import asyncio, logging
from datetime import datetime, timedelta, timezone
from typing import Any
import httpx
from sqlalchemy import delete, func, select
from app.config import settings
from app.db.models import Candle
from app.db.session import SessionLocal

logger = logging.getLogger(__name__)
CACHE_SOURCE = "strategy_history_cache"
CACHE_DAYS = 10
TARGET_ROWS = CACHE_DAYS * 24 * 60
MAX_FETCH_ROWS = 15000

def _dt(ms: int | float) -> datetime:
    return datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc)

async def ensure_strategy_history() -> dict[str, Any]:
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(days=CACHE_DAYS)
    symbols = [s.upper() for s in settings.auto_trader_symbols]
    result: dict[str, Any] = {"start": start.isoformat(), "end": end.isoformat(), "symbols": {}}
    with SessionLocal() as session:
        session.execute(delete(Candle).where(
            Candle.source_name == CACHE_SOURCE,
            Candle.interval == "1m",
            Candle.open_time < start,
        ))
        session.commit()
    async with httpx.AsyncClient(timeout=30) as client:
        for symbol in symbols:
            with SessionLocal() as session:
                count = int(session.scalar(select(func.count(Candle.id)).where(
                    Candle.symbol == symbol,
                    Candle.interval == "1m",
                    Candle.is_closed.is_(True),
                    Candle.open_time >= start,
                    Candle.open_time < end,
                )) or 0)
            if count >= TARGET_ROWS:
                result["symbols"][symbol] = {"status": "ready", "rows_available": count, "rows_inserted": 0}
                continue
            inserted = fetched = 0
            next_start_ms = int(start.timestamp() * 1000)
            end_ms = int(end.timestamp() * 1000)
            while next_start_ms < end_ms and fetched < MAX_FETCH_ROWS:
                limit = min(1000, MAX_FETCH_ROWS - fetched)
                response = await client.get(
                    f"{settings.binance_rest_base_url.rstrip('/')}/api/v3/klines",
                    params={"symbol": symbol, "interval": "1m", "startTime": next_start_ms, "endTime": end_ms, "limit": limit},
                )
                response.raise_for_status()
                rows = response.json()
                if not rows:
                    break
                fetched += len(rows)
                page_start = _dt(rows[0][0])
                page_end = _dt(rows[-1][0]) + timedelta(minutes=1)
                with SessionLocal() as session:
                    existing = set(session.scalars(select(Candle.open_time).where(
                        Candle.exchange == "binance",
                        Candle.symbol == symbol,
                        Candle.interval == "1m",
                        Candle.open_time >= page_start,
                        Candle.open_time < page_end,
                    )).all())
                    for row in rows:
                        open_time = _dt(row[0])
                        if open_time >= end or open_time in existing:
                            continue
                        session.add(Candle(
                            exchange="binance", source_name=CACHE_SOURCE, symbol=symbol, interval="1m",
                            open_time=open_time, close_time=_dt(row[6]),
                            open=float(row[1]), high=float(row[2]), low=float(row[3]), close=float(row[4]),
                            volume=float(row[5]), quote_volume=float(row[7]), trades=int(row[8]),
                            is_closed=True, raw={"source": CACHE_SOURCE}, raw_payload={"source": CACHE_SOURCE},
                        ))
                        existing.add(open_time)
                        inserted += 1
                    session.commit()
                next_start_ms = int(rows[-1][0]) + 60000
                if len(rows) < limit:
                    break
            with SessionLocal() as session:
                final_count = int(session.scalar(select(func.count(Candle.id)).where(
                    Candle.symbol == symbol,
                    Candle.interval == "1m",
                    Candle.is_closed.is_(True),
                    Candle.open_time >= start,
                    Candle.open_time < end,
                )) or 0)
            result["symbols"][symbol] = {
                "status": "ready" if final_count >= TARGET_ROWS else "partial",
                "rows_available": final_count, "rows_inserted": inserted, "rows_fetched": fetched,
            }
    return result

class StrategyHistoryBootstrapService:
    def __init__(self) -> None:
        self.task: asyncio.Task | None = None
        self.last_result: dict[str, Any] | None = None
        self.last_error: str | None = None
    async def start(self) -> None:
        if self.task and not self.task.done():
            return
        self.task = asyncio.create_task(self._run(), name="strategy-history-bootstrap")
    async def _run(self) -> None:
        try:
            self.last_result = await ensure_strategy_history()
            self.last_error = None
            logger.info("Strategy history bootstrap finished: %s", self.last_result)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.exception("Strategy history bootstrap failed")
    async def refresh_now(self) -> dict[str, Any]:
        if self.task and not self.task.done():
            await self.task
        else:
            await self._run()
        return self.last_result or {"status": "error", "error": self.last_error}
    async def stop(self) -> None:
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        self.task = None
    def status(self) -> dict[str, Any]:
        return {"running": bool(self.task and not self.task.done()), "last_result": self.last_result, "last_error": self.last_error}
