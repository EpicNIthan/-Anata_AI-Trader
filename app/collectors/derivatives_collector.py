from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import ExternalDataEvent
from app.db.session import SessionLocal

logger = logging.getLogger(__name__)

_BINANCE_FUTURES_BLOCKED_UNTIL: datetime | None = None


def _dt_from_ms(value: Any, fallback: datetime | None = None) -> datetime:
    if value in (None, ""):
        return fallback or datetime.now(timezone.utc)
    return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc)


def _float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _http_status(exc: Exception) -> int | None:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code
    return None


def _futures_blocked_message() -> str | None:
    global _BINANCE_FUTURES_BLOCKED_UNTIL
    if not _BINANCE_FUTURES_BLOCKED_UNTIL:
        return None
    now = datetime.now(timezone.utc)
    if now >= _BINANCE_FUTURES_BLOCKED_UNTIL:
        _BINANCE_FUTURES_BLOCKED_UNTIL = None
        return None
    seconds = int((_BINANCE_FUTURES_BLOCKED_UNTIL - now).total_seconds())
    return (
        "Binance Futures public API is blocked from this Railway region/IP "
        f"(HTTP 451). Cooling down for {seconds}s."
    )


def _set_futures_blocked_cooldown(seconds: int = 3600) -> str:
    global _BINANCE_FUTURES_BLOCKED_UNTIL
    _BINANCE_FUTURES_BLOCKED_UNTIL = datetime.now(timezone.utc) + timedelta(seconds=seconds)
    return _futures_blocked_message() or "Binance Futures public API is blocked from this Railway region/IP (HTTP 451)."


def _ratio_payload(raw: dict[str, Any], *, period: str) -> dict[str, Any]:
    long_account = _float(raw.get("longAccount"), 0.5) or 0.5
    short_account = _float(raw.get("shortAccount"), 0.5) or 0.5
    return {
        "period": period,
        "long_short_ratio": _float(raw.get("longShortRatio"), 1.0),
        "long_account_pct": long_account,
        "short_account_pct": short_account,
        "long_bias": max(-1.0, min(1.0, (long_account - short_account))),
    }


def _taker_payload(raw: dict[str, Any], *, period: str) -> dict[str, Any]:
    buy_volume = _float(raw.get("buyVol"), 0.0) or 0.0
    sell_volume = _float(raw.get("sellVol"), 0.0) or 0.0
    total = buy_volume + sell_volume
    buy_pressure = buy_volume / total if total > 0 else 0.5
    return {
        "period": period,
        "buy_sell_ratio": _float(raw.get("buySellRatio"), 1.0),
        "buy_volume": buy_volume,
        "sell_volume": sell_volume,
        "buy_pressure": buy_pressure,
        "taker_bias": max(-1.0, min(1.0, (buy_pressure - 0.5) * 2.0)),
    }


class BinanceDerivativesCollector:
    """Collects public aggregate futures sentiment data.

    This is not private trader data. It is public market-wide derivatives data:
    long/short ratios, taker buy/sell pressure, open interest, and funding.
    """

    name = "binance_derivatives"

    def __init__(
        self,
        symbols: list[str] | None = None,
        period: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.symbols = [symbol.upper() for symbol in (symbols or settings.derivatives_symbols)]
        self.period = period or settings.derivatives_period
        self.base_url = (base_url or settings.binance_futures_rest_base_url).rstrip("/")

    @property
    def endpoints(self) -> dict[str, str]:
        return {
            "global_long_short_account_ratio": "/futures/data/globalLongShortAccountRatio",
            "top_long_short_account_ratio": "/futures/data/topLongShortAccountRatio",
            "top_long_short_position_ratio": "/futures/data/topLongShortPositionRatio",
            "taker_buy_sell_volume": "/futures/data/takerlongshortRatio",
            "open_interest_hist": "/futures/data/openInterestHist",
            "open_interest": "/fapi/v1/openInterest",
            "funding_rate": "/fapi/v1/fundingRate",
        }

    async def run(self, stop_event: asyncio.Event, state: Any | None = None) -> None:
        if not settings.derivatives_enabled:
            if state:
                state.warning = "DERIVATIVES_ENABLED=false"
                state.running = False
            return

        if state:
            state.details = {
                "base_url": self.base_url,
                "symbols": self.symbols,
                "period": self.period,
                "endpoints": self.endpoints,
            }
        while not stop_event.is_set():
            try:
                result = await self.fetch_once()
                if state:
                    state.mark_event(result | {"rows_saved": result.get("rows_saved", 0)})
                    errors = result.get("errors") or {}
                    state.last_error = None
                    state.warning = "; ".join(list(errors.values())[:3]) if errors else None
            except Exception as exc:
                logger.exception("Binance derivatives collector failed")
                if state:
                    state.mark_error(f"{type(exc).__name__}: {exc}")
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=max(settings.derivatives_poll_interval_seconds, 30))
            except asyncio.TimeoutError:
                continue

    async def fetch_once(self, *, mock: bool = False) -> dict[str, Any]:
        rows_saved = 0
        errors: dict[str, str] = {}
        per_symbol: dict[str, int] = {}
        url_log: list[str] = []
        blocked_message = _futures_blocked_message()
        if blocked_message:
            return {
                "source": self.name,
                "mode": "live",
                "period": self.period,
                "symbols": self.symbols,
                "rows_saved": 0,
                "per_symbol": {symbol: 0 for symbol in self.symbols},
                "errors": {"binance_futures": blocked_message},
                "urls": url_log,
            }
        if mock:
            with SessionLocal() as session:
                for symbol in self.symbols:
                    saved = self._store_mock_symbol(session, symbol)
                    per_symbol[symbol] = saved
                    rows_saved += saved
                session.commit()
            return {
                "source": self.name,
                "mode": "mock",
                "period": self.period,
                "symbols": self.symbols,
                "rows_saved": rows_saved,
                "per_symbol": per_symbol,
                "errors": errors,
                "urls": url_log,
            }

        blocked = False
        async with httpx.AsyncClient(timeout=20) as client:
            with SessionLocal() as session:
                for symbol in self.symbols:
                    saved = 0
                    if blocked:
                        per_symbol[symbol] = saved
                        continue
                    for data_type, path in self.endpoints.items():
                        url = f"{self.base_url}{path}"
                        params = self._params(data_type, symbol)
                        url_log.append(f"{url}?{httpx.QueryParams(params)}")
                        try:
                            response = await client.get(url, params=params)
                            response.raise_for_status()
                            saved += self._store_response(session, symbol, data_type, response.json())
                        except Exception as exc:
                            if _http_status(exc) == 451:
                                blocked_message = _set_futures_blocked_cooldown()
                                logger.warning(blocked_message)
                                errors["binance_futures"] = blocked_message
                                blocked = True
                                break
                            logger.warning("Derivatives fetch failed for %s %s: %s", symbol, data_type, exc)
                            errors[f"{symbol}:{data_type}"] = f"{type(exc).__name__}: {exc}"
                    per_symbol[symbol] = saved
                    rows_saved += saved
                session.commit()
        return {
            "source": self.name,
            "mode": "live",
            "period": self.period,
            "symbols": self.symbols,
            "rows_saved": rows_saved,
            "per_symbol": per_symbol,
            "errors": errors,
            "urls": url_log[-20:],
        }

    def status(self, session: Session) -> dict[str, Any]:
        latest = list(
            session.scalars(
                select(ExternalDataEvent)
                .where(ExternalDataEvent.source_name.like("binance_futures_%"))
                .order_by(desc(ExternalDataEvent.event_time))
                .limit(20)
            )
        )
        counts = {
            data_type: count
            for data_type, count in session.execute(
                select(ExternalDataEvent.data_type, func.count(ExternalDataEvent.id))
                .where(ExternalDataEvent.source_name.like("binance_futures_%"))
                .group_by(ExternalDataEvent.data_type)
            ).all()
        }
        return {
            "enabled": settings.derivatives_enabled,
            "base_url": self.base_url,
            "period": self.period,
            "symbols": self.symbols,
            "counts_by_type": counts,
            "latest": [self._serialize_event(row) for row in latest],
        }

    def _params(self, data_type: str, symbol: str) -> dict[str, Any]:
        if data_type in {
            "global_long_short_account_ratio",
            "top_long_short_account_ratio",
            "top_long_short_position_ratio",
            "taker_buy_sell_volume",
            "open_interest_hist",
        }:
            return {"symbol": symbol, "period": self.period, "limit": 2}
        if data_type == "funding_rate":
            return {"symbol": symbol, "limit": 1}
        return {"symbol": symbol}

    def _store_response(self, session: Session, symbol: str, data_type: str, payload: Any) -> int:
        rows = payload if isinstance(payload, list) else [payload]
        saved = 0
        for raw in rows:
            if not isinstance(raw, dict):
                continue
            event_time = self._event_time(data_type, raw)
            source_name = f"binance_futures_{data_type}"
            parsed_payload, numeric_value = self._parse_payload(data_type, raw)
            if self._upsert_event(
                session,
                source_name=source_name,
                data_type=data_type,
                symbol=symbol,
                event_time=event_time,
                numeric_value=numeric_value,
                payload=parsed_payload,
                raw_payload=raw,
            ):
                saved += 1
        return saved

    def _store_mock_symbol(self, session: Session, symbol: str) -> int:
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        mock_rows = {
            "global_long_short_account_ratio": {
                "longShortRatio": "1.2400",
                "longAccount": "0.5535",
                "shortAccount": "0.4465",
                "timestamp": int(now.timestamp() * 1000),
            },
            "top_long_short_account_ratio": {
                "longShortRatio": "1.1100",
                "longAccount": "0.5261",
                "shortAccount": "0.4739",
                "timestamp": int(now.timestamp() * 1000),
            },
            "top_long_short_position_ratio": {
                "longShortRatio": "1.3800",
                "longAccount": "0.5798",
                "shortAccount": "0.4202",
                "timestamp": int(now.timestamp() * 1000),
            },
            "taker_buy_sell_volume": {
                "buySellRatio": "1.1800",
                "buyVol": "1180.0",
                "sellVol": "1000.0",
                "timestamp": int(now.timestamp() * 1000),
            },
            "open_interest_hist": {
                "sumOpenInterest": "20403.637",
                "sumOpenInterestValue": "150570784.07",
                "timestamp": int(now.timestamp() * 1000),
            },
            "open_interest": {
                "openInterest": "20403.637",
                "symbol": symbol,
                "time": int(now.timestamp() * 1000),
            },
            "funding_rate": {
                "symbol": symbol,
                "fundingRate": "0.0001",
                "fundingTime": int(now.timestamp() * 1000),
                "markPrice": "65000",
            },
        }
        saved = 0
        for data_type, raw in mock_rows.items():
            saved += self._store_response(session, symbol, data_type, raw)
        return saved

    def _event_time(self, data_type: str, raw: dict[str, Any]) -> datetime:
        if data_type == "funding_rate":
            return _dt_from_ms(raw.get("fundingTime"))
        if data_type == "open_interest":
            return _dt_from_ms(raw.get("time"))
        return _dt_from_ms(raw.get("timestamp"))

    def _parse_payload(self, data_type: str, raw: dict[str, Any]) -> tuple[dict[str, Any], float | None]:
        if data_type in {
            "global_long_short_account_ratio",
            "top_long_short_account_ratio",
            "top_long_short_position_ratio",
        }:
            parsed = _ratio_payload(raw, period=self.period)
            return parsed, parsed["long_short_ratio"]
        if data_type == "taker_buy_sell_volume":
            parsed = _taker_payload(raw, period=self.period)
            return parsed, parsed["buy_sell_ratio"]
        if data_type == "open_interest_hist":
            parsed = {
                "period": self.period,
                "sum_open_interest": _float(raw.get("sumOpenInterest")),
                "sum_open_interest_value": _float(raw.get("sumOpenInterestValue")),
                "cmc_circulating_supply": _float(raw.get("CMCCirculatingSupply")),
            }
            return parsed, parsed["sum_open_interest_value"] or parsed["sum_open_interest"]
        if data_type == "open_interest":
            parsed = {"open_interest": _float(raw.get("openInterest"))}
            return parsed, parsed["open_interest"]
        if data_type == "funding_rate":
            parsed = {
                "funding_rate": _float(raw.get("fundingRate")),
                "mark_price": _float(raw.get("markPrice")),
            }
            return parsed, parsed["funding_rate"]
        return {}, None

    def _upsert_event(
        self,
        session: Session,
        *,
        source_name: str,
        data_type: str,
        symbol: str,
        event_time: datetime,
        numeric_value: float | None,
        payload: dict[str, Any],
        raw_payload: dict[str, Any],
    ) -> bool:
        existing = session.scalar(
            select(ExternalDataEvent)
            .where(
                ExternalDataEvent.source_name == source_name,
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
                source_name=source_name,
                data_type=data_type,
                symbol=symbol,
                event_time=event_time,
                numeric_value=numeric_value,
                payload=payload,
                raw_payload=raw_payload,
            )
        )
        return True

    def _serialize_event(self, row: ExternalDataEvent) -> dict[str, Any]:
        return {
            "id": row.id,
            "source_name": row.source_name,
            "data_type": row.data_type,
            "symbol": row.symbol,
            "event_time": row.event_time.isoformat() if row.event_time else None,
            "numeric_value": row.numeric_value,
            "payload": row.payload,
        }
