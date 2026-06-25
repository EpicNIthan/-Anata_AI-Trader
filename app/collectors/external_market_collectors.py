from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from statistics import mean
from typing import Any

import httpx
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import ExternalDataEvent, NewsArticle, NewsSentiment
from app.db.session import SessionLocal

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _minute(value: datetime | None = None) -> datetime:
    return (value or _now()).replace(second=0, microsecond=0)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _dt_from_seconds(value: Any) -> datetime:
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return _now()


def _raw_payload(payload: dict[str, Any] | None, *, force: bool = False) -> dict[str, Any] | None:
    if force or settings.store_raw_external_events:
        return payload
    return None


def _classification_score(value: str | None) -> float:
    key = (value or "").lower().strip()
    if "extreme fear" in key:
        return 0.0
    if "fear" in key:
        return 25.0
    if "neutral" in key:
        return 50.0
    if "extreme greed" in key:
        return 100.0
    if "greed" in key:
        return 75.0
    return 50.0


def _upsert_event(
    session: Session,
    *,
    source_name: str,
    data_type: str,
    symbol: str | None,
    event_time: datetime,
    numeric_value: float | None,
    payload: dict[str, Any] | None = None,
    raw_payload: dict[str, Any] | None = None,
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
        existing.payload = payload or {}
        existing.raw_payload = raw_payload
        return False
    session.add(
        ExternalDataEvent(
            source_name=source_name,
            data_type=data_type,
            symbol=symbol,
            event_time=event_time,
            numeric_value=numeric_value,
            payload=payload or {},
            raw_payload=raw_payload,
        )
    )
    return True


def _previous_numeric(session: Session, source_name: str, data_type: str, symbol: str | None = None) -> float | None:
    row = session.scalar(
        select(ExternalDataEvent)
        .where(
            ExternalDataEvent.source_name == source_name,
            ExternalDataEvent.data_type == data_type,
            ExternalDataEvent.symbol == symbol,
        )
        .order_by(desc(ExternalDataEvent.event_time))
        .limit(1)
    )
    return float(row.numeric_value) if row and row.numeric_value is not None else None


@dataclass(frozen=True)
class ExternalCollectorResult:
    collector: str
    enabled: bool
    mode: str
    rows_saved: int
    details: dict[str, Any]
    error: str | None = None


class BaseExternalCollector:
    name = "base"
    role = "external market context"
    source_name = "external"
    data_types: tuple[str, ...] = ()

    @property
    def enabled(self) -> bool:
        return False

    async def fetch_once(self, *, mock: bool = False) -> ExternalCollectorResult:
        raise NotImplementedError

    def status(self, session: Session) -> dict[str, Any]:
        counts = {
            data_type: int(count)
            for data_type, count in session.execute(
                select(ExternalDataEvent.data_type, func.count(ExternalDataEvent.id))
                .where(ExternalDataEvent.source_name == self.source_name)
                .group_by(ExternalDataEvent.data_type)
            ).all()
        }
        latest = list(
            session.scalars(
                select(ExternalDataEvent)
                .where(ExternalDataEvent.source_name == self.source_name)
                .order_by(desc(ExternalDataEvent.event_time))
                .limit(10)
            )
        )
        return {
            "name": self.name,
            "source_name": self.source_name,
            "enabled": self.enabled,
            "role": self.role,
            "counts_by_type": counts,
            "latest": [_serialize_event(row) for row in latest],
        }


class FearGreedCollector(BaseExternalCollector):
    name = "fear_greed"
    role = "Alternative.me crypto fear/greed index"
    source_name = "alternative_me_fear_greed"
    data_types = (
        "fear_greed_value",
        "fear_greed_change_1d",
        "fear_greed_change_24h",
        "fear_greed_classification_score",
        "fear_greed_classification",
    )

    @property
    def enabled(self) -> bool:
        return settings.enable_fear_greed_collector

    async def fetch_once(self, *, mock: bool = False) -> ExternalCollectorResult:
        if mock:
            payload = {
                "data": [
                    {"value": "54", "value_classification": "Neutral", "timestamp": str(int(_now().timestamp()))},
                    {"value": "49", "value_classification": "Neutral", "timestamp": str(int((_now() - timedelta(days=1)).timestamp()))},
                ]
            }
            return self._store(payload, mode="mock")
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get("https://api.alternative.me/fng/", params={"limit": 2, "format": "json"})
            response.raise_for_status()
            return self._store(response.json(), mode="live")

    def _store(self, payload: dict[str, Any], *, mode: str) -> ExternalCollectorResult:
        rows = payload.get("data") if isinstance(payload, dict) else []
        if not rows:
            return ExternalCollectorResult(self.name, self.enabled, mode, 0, {"warning": "No fear/greed data returned"})
        latest = rows[0]
        previous = rows[1] if len(rows) > 1 else {}
        value = _float(latest.get("value"))
        previous_value = _float(previous.get("value"), value)
        change = value - previous_value
        classification = latest.get("value_classification")
        event_time = _minute(_dt_from_seconds(latest.get("timestamp")))
        saved = 0
        with SessionLocal() as session:
            event_payload = {"classification": classification, "source": "alternative.me", "mode": mode}
            for data_type, numeric in (
                ("fear_greed_value", value),
                ("fear_greed_change_1d", change),
                ("fear_greed_change_24h", change),
                ("fear_greed_classification_score", _classification_score(classification)),
                ("fear_greed_classification", _classification_score(classification)),
            ):
                saved += int(
                    _upsert_event(
                        session,
                        source_name=self.source_name,
                        data_type=data_type,
                        symbol=None,
                        event_time=event_time,
                        numeric_value=numeric,
                        payload=event_payload,
                        raw_payload=_raw_payload(payload),
                    )
                )
            session.commit()
        return ExternalCollectorResult(self.name, self.enabled, mode, saved, {"value": value, "change_1d": change})


class GlobalMarketCollector(BaseExternalCollector):
    name = "global_market"
    role = "CoinGecko public global crypto market context"
    source_name = "coingecko_global_market"
    data_types = (
        "global_market_cap_usd",
        "total_market_cap_usd",
        "global_market_cap_change_24h",
        "market_cap_change_24h",
        "total_volume_usd",
        "total_volume_change_24h",
        "btc_dominance",
        "eth_dominance",
        "btc_dominance_change",
        "btc_dominance_change_24h",
    )

    @property
    def enabled(self) -> bool:
        return settings.enable_global_market_collector

    async def fetch_once(self, *, mock: bool = False) -> ExternalCollectorResult:
        if mock:
            payload = {
                "data": {
                    "total_market_cap": {"usd": 2_500_000_000_000},
                    "total_volume": {"usd": 105_000_000_000},
                    "market_cap_change_percentage_24h_usd": 1.8,
                    "market_cap_percentage": {"btc": 58.1, "eth": 10.5},
                }
            }
            return self._store(payload, mode="mock")
        headers = {}
        if settings.coingecko_demo_api_key:
            headers["x-cg-demo-api-key"] = settings.coingecko_demo_api_key
        async with httpx.AsyncClient(timeout=20, headers=headers) as client:
            response = await client.get("https://api.coingecko.com/api/v3/global")
            response.raise_for_status()
            return self._store(response.json(), mode="live")

    def _store(self, payload: dict[str, Any], *, mode: str) -> ExternalCollectorResult:
        data = payload.get("data") if isinstance(payload, dict) else {}
        market_cap = _float((data.get("total_market_cap") or {}).get("usd"))
        volume = _float((data.get("total_volume") or {}).get("usd"))
        market_cap_change = _float(data.get("market_cap_change_percentage_24h_usd")) / 100.0
        dominance = data.get("market_cap_percentage") or {}
        btc_dominance = _float(dominance.get("btc")) / 100.0
        eth_dominance = _float(dominance.get("eth")) / 100.0
        event_time = _minute()
        with SessionLocal() as session:
            previous_volume = _previous_numeric(session, self.source_name, "total_volume_usd")
            previous_btc_dominance = _previous_numeric(session, self.source_name, "btc_dominance")
            volume_change = ((volume - previous_volume) / previous_volume) if previous_volume else 0.0
            btc_dominance_change = btc_dominance - previous_btc_dominance if previous_btc_dominance is not None else 0.0
            values = {
                "global_market_cap_usd": market_cap,
                "total_market_cap_usd": market_cap,
                "global_market_cap_change_24h": market_cap_change,
                "market_cap_change_24h": market_cap_change,
                "total_volume_usd": volume,
                "total_volume_change_24h": volume_change,
                "btc_dominance": btc_dominance,
                "eth_dominance": eth_dominance,
                "btc_dominance_change": btc_dominance_change,
                "btc_dominance_change_24h": btc_dominance_change,
            }
            saved = 0
            for data_type, numeric in values.items():
                saved += int(
                    _upsert_event(
                        session,
                        source_name=self.source_name,
                        data_type=data_type,
                        symbol=None,
                        event_time=event_time,
                        numeric_value=numeric,
                        payload={"mode": mode},
                        raw_payload=_raw_payload(payload),
                    )
                )
            session.commit()
        return ExternalCollectorResult(self.name, self.enabled, mode, saved, values)


class StablecoinRiskCollector(BaseExternalCollector):
    name = "stablecoin_risk"
    role = "DefiLlama stablecoin peg/supply context"
    source_name = "defillama_stablecoin_risk"
    data_types = (
        "usdt_price_deviation",
        "usdc_price_deviation",
        "usdt_deviation",
        "usdc_deviation",
        "stablecoin_depeg_risk",
        "stablecoin_supply_change_1d",
        "stablecoin_supply_change_24h",
    )

    @property
    def enabled(self) -> bool:
        return settings.enable_stablecoin_risk_collector

    async def fetch_once(self, *, mock: bool = False) -> ExternalCollectorResult:
        if mock:
            payload = {
                "peggedAssets": [
                    {"symbol": "USDT", "price": 0.9994, "circulating": {"peggedUSD": 112_000_000_000}, "change_1d": 0.001},
                    {"symbol": "USDC", "price": 1.0002, "circulating": {"peggedUSD": 33_000_000_000}, "change_1d": -0.0005},
                ]
            }
            return self._store(payload, mode="mock")
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get("https://stablecoins.llama.fi/stablecoins", params={"includePrices": "true"})
            response.raise_for_status()
            return self._store(response.json(), mode="live")

    def _asset(self, rows: list[dict[str, Any]], symbol: str) -> dict[str, Any]:
        normalized = symbol.upper()
        return next((row for row in rows if str(row.get("symbol") or "").upper() == normalized), {})

    def _price(self, row: dict[str, Any]) -> float:
        return _float(row.get("price") or (row.get("pegInfo") or {}).get("price"), 1.0)

    def _supply_change(self, row: dict[str, Any]) -> float:
        for key in ("change_1d", "circulatingChange1d", "supply_change_1d"):
            if row.get(key) is not None:
                return _float(row.get(key))
        return 0.0

    def _store(self, payload: dict[str, Any], *, mode: str) -> ExternalCollectorResult:
        rows = payload.get("peggedAssets") or payload.get("data") or []
        if not isinstance(rows, list):
            rows = []
        usdt = self._asset(rows, "USDT")
        usdc = self._asset(rows, "USDC")
        usdt_deviation = self._price(usdt) - 1.0 if usdt else 0.0
        usdc_deviation = self._price(usdc) - 1.0 if usdc else 0.0
        supply_change = self._supply_change(usdt) + self._supply_change(usdc)
        depeg_risk = _clamp(max(abs(usdt_deviation), abs(usdc_deviation)) / 0.01, 0.0, 1.0)
        values = {
            "usdt_price_deviation": usdt_deviation,
            "usdc_price_deviation": usdc_deviation,
            "usdt_deviation": usdt_deviation,
            "usdc_deviation": usdc_deviation,
            "stablecoin_depeg_risk": depeg_risk,
            "stablecoin_supply_change_1d": supply_change,
            "stablecoin_supply_change_24h": supply_change,
        }
        event_time = _minute()
        saved = 0
        with SessionLocal() as session:
            for data_type, numeric in values.items():
                saved += int(
                    _upsert_event(
                        session,
                        source_name=self.source_name,
                        data_type=data_type,
                        symbol=None,
                        event_time=event_time,
                        numeric_value=numeric,
                        payload={"mode": mode, "assets_found": sorted({str(row.get("symbol", "")).upper() for row in rows if row.get("symbol")})},
                        raw_payload=_raw_payload(payload),
                    )
                )
            session.commit()
        return ExternalCollectorResult(self.name, self.enabled, mode, saved, values)


class MacroRiskCollector(BaseExternalCollector):
    name = "macro_risk"
    role = "news-derived macro/regulation/security/world risk"
    source_name = "macro_risk_news"
    data_types = (
        "macro_risk_score",
        "regulation_risk_score",
        "fed_risk_score",
        "war_risk_score",
        "exchange_hack_risk_score",
        "security_risk_score",
        "etf_positive_score",
        "etf_bullish_score",
        "world_risk_score",
    )

    KEYWORDS = {
        "macro_risk_score": ("fed", "federal reserve", "cpi", "rate hike", "rate cut", "inflation", "recession", "bank crisis"),
        "regulation_risk_score": ("sec", "lawsuit", "regulation", "regulatory", "ban", "enforcement"),
        "fed_risk_score": ("fed", "federal reserve", "cpi", "rate hike", "rate cut", "inflation"),
        "war_risk_score": ("war", "geopolitical", "sanction", "missile", "conflict"),
        "exchange_hack_risk_score": ("hack", "exploit", "bridge attack", "exchange hack", "stolen", "drain"),
        "security_risk_score": ("hack", "exploit", "bridge attack", "stolen", "drain", "vulnerability"),
        "etf_positive_score": ("etf approval", "spot etf", "etf inflow", "approval"),
        "etf_bullish_score": ("etf approval", "spot etf", "etf inflow", "approval"),
        "world_risk_score": ("war", "geopolitical", "bank crisis", "recession", "sanction"),
    }

    @property
    def enabled(self) -> bool:
        return settings.enable_macro_risk_collector

    async def fetch_once(self, *, mock: bool = False) -> ExternalCollectorResult:
        with SessionLocal() as session:
            if mock:
                scores = {
                    "macro_risk_score": 0.55,
                    "regulation_risk_score": 0.35,
                    "fed_risk_score": 0.45,
                    "war_risk_score": 0.20,
                    "exchange_hack_risk_score": 0.10,
                    "security_risk_score": 0.10,
                    "etf_positive_score": 0.40,
                    "etf_bullish_score": 0.40,
                    "world_risk_score": 0.20,
                }
                saved = self._store_scores(session, scores, {"mode": "mock", "articles_used": 1})
                session.commit()
                return ExternalCollectorResult(self.name, self.enabled, "mock", saved, scores)
            since = _now() - timedelta(hours=48)
            rows = session.execute(
                select(NewsArticle, NewsSentiment)
                .outerjoin(NewsSentiment, NewsSentiment.article_id == NewsArticle.id)
                .where(NewsArticle.published_at >= since)
                .order_by(desc(NewsArticle.published_at))
                .limit(300)
            ).all()
            scores = self._score_articles(rows)
            saved = self._store_scores(session, scores, {"mode": "live", "articles_used": len(rows)})
            session.commit()
        return ExternalCollectorResult(self.name, self.enabled, "live", saved, scores | {"articles_used": len(rows)})

    def _score_articles(self, rows: list[tuple[NewsArticle, NewsSentiment | None]]) -> dict[str, float]:
        accum: dict[str, list[float]] = defaultdict(list)
        for article, sentiment in rows:
            text = f"{article.title or ''} {article.raw_text or ''}".lower()
            risk = _float(sentiment.risk_score if sentiment else 0.25)
            sentiment_score = _float(sentiment.sentiment_score if sentiment else 0.0)
            for data_type, keywords in self.KEYWORDS.items():
                matches = sum(1 for keyword in keywords if keyword in text)
                if matches <= 0:
                    continue
                if data_type in {"etf_bullish_score", "etf_positive_score"}:
                    accum[data_type].append(_clamp((0.35 + max(sentiment_score, 0.0)) * min(matches, 3) / 3, 0.0, 1.0))
                else:
                    accum[data_type].append(_clamp((0.35 + risk) * min(matches, 3) / 3, 0.0, 1.0))
        return {data_type: mean(values) if values else 0.0 for data_type, values in accum.items()} | {
            data_type: 0.0 for data_type in self.KEYWORDS if data_type not in accum
        }

    def _store_scores(self, session: Session, scores: dict[str, float], payload: dict[str, Any]) -> int:
        saved = 0
        event_time = _minute()
        for data_type, numeric in scores.items():
            saved += int(
                _upsert_event(
                    session,
                    source_name=self.source_name,
                    data_type=data_type,
                    symbol=None,
                    event_time=event_time,
                    numeric_value=numeric,
                    payload=payload,
                    raw_payload=None,
                )
            )
        return saved


class LiquidationCollector(BaseExternalCollector):
    name = "liquidations"
    role = "Binance Futures force-order liquidation rollups"
    source_name = "binance_futures_liquidations"
    data_types = (
        "liquidation_long_usd_5m",
        "liquidation_short_usd_5m",
        "liquidation_long_usd_1m",
        "liquidation_short_usd_1m",
        "liquidation_total_usd_5m",
        "liquidation_imbalance_5m",
        "liquidation_spike_score",
    )

    def __init__(self, symbols: list[str] | None = None) -> None:
        self.symbols = [symbol.upper() for symbol in (symbols or settings.liquidation_symbols)]
        self.rollup_seconds = max(settings.liquidation_rollup_seconds, 30)
        self._events: deque[dict[str, Any]] = deque(maxlen=5000)

    @property
    def enabled(self) -> bool:
        return settings.enable_liquidation_collector

    @property
    def stream_url(self) -> str:
        streams = "/".join(f"{symbol.lower()}@forceOrder" for symbol in self.symbols)
        return f"{settings.binance_futures_ws_base_url.rstrip('/')}/stream?streams={streams}"

    async def fetch_once(self, *, mock: bool = False) -> ExternalCollectorResult:
        if not mock:
            return ExternalCollectorResult(
                self.name,
                self.enabled,
                "status",
                0,
                {"message": "Liquidations are collected from websocket worker; use mock=true for smoke tests."},
            )
        now = _now()
        self._events.extend(
            [
                {"symbol": "BTCUSDT", "side": "SELL", "notional": 125_000.0, "event_time": now - timedelta(minutes=1)},
                {"symbol": "BTCUSDT", "side": "BUY", "notional": 40_000.0, "event_time": now - timedelta(minutes=2)},
                {"symbol": "ETHUSDT", "side": "SELL", "notional": 31_000.0, "event_time": now - timedelta(minutes=1)},
            ]
        )
        saved = self._store_rollups(_minute(now), mode="mock")
        return ExternalCollectorResult(self.name, self.enabled, "mock", saved, {"symbols": self.symbols})

    async def run(self, stop_event: asyncio.Event, state: Any | None = None) -> None:
        if not self.enabled:
            if state:
                state.warning = "ENABLE_LIQUIDATION_COLLECTOR=false"
                state.running = False
            return
        try:
            import websockets
        except Exception as exc:  # pragma: no cover - optional runtime dependency guard
            if state:
                state.mark_error(f"websockets unavailable: {exc}")
            return
        if state:
            state.set_subscription(streams=[f"{symbol.lower()}@forceOrder" for symbol in self.symbols], websocket_url=self.stream_url)
        last_rollup = _now()
        while not stop_event.is_set():
            try:
                async with websockets.connect(self.stream_url, ping_interval=20, ping_timeout=20) as websocket:
                    if state:
                        state.last_error = None
                    while not stop_event.is_set():
                        try:
                            raw = await asyncio.wait_for(websocket.recv(), timeout=5)
                            parsed = self._parse_force_order(json.loads(raw))
                            if parsed:
                                self._events.append(parsed)
                                if state:
                                    state.mark_message({"symbol": parsed["symbol"], "notional": parsed["notional"]})
                            if (_now() - last_rollup).total_seconds() >= self.rollup_seconds:
                                rows = self._store_rollups(_minute(), mode="live")
                                last_rollup = _now()
                                if state:
                                    state.mark_saved(rows, {"rows_saved": rows, "symbols": self.symbols})
                        except asyncio.TimeoutError:
                            if (_now() - last_rollup).total_seconds() >= self.rollup_seconds:
                                rows = self._store_rollups(_minute(), mode="live")
                                last_rollup = _now()
                                if state:
                                    state.mark_saved(rows, {"rows_saved": rows, "symbols": self.symbols})
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Liquidation websocket failed: %s", exc)
                if state:
                    state.mark_error(f"{type(exc).__name__}: {exc}")
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=10)
                except asyncio.TimeoutError:
                    continue

    def _parse_force_order(self, raw: dict[str, Any]) -> dict[str, Any] | None:
        data = raw.get("data") or raw
        order = data.get("o") or {}
        symbol = str(order.get("s") or data.get("s") or "").upper()
        if symbol not in self.symbols:
            return None
        event_time = datetime.fromtimestamp(_float(data.get("E"), _now().timestamp() * 1000) / 1000.0, tz=timezone.utc)
        side = str(order.get("S") or "").upper()
        price = _float(order.get("ap") or order.get("p"))
        quantity = _float(order.get("q"))
        notional = price * quantity
        return {"symbol": symbol, "side": side, "notional": notional, "event_time": event_time}

    def _store_rollups(self, event_time: datetime, *, mode: str) -> int:
        cutoff = event_time - timedelta(minutes=5)
        recent = [item for item in self._events if item["event_time"] >= cutoff]
        saved = 0
        with SessionLocal() as session:
            for symbol in self.symbols:
                rows = [item for item in recent if item["symbol"] == symbol]
                long_usd = sum(item["notional"] for item in rows if item["side"] == "SELL")
                short_usd = sum(item["notional"] for item in rows if item["side"] == "BUY")
                rows_1m = [item for item in rows if item["event_time"] >= event_time - timedelta(minutes=1)]
                long_usd_1m = sum(item["notional"] for item in rows_1m if item["side"] == "SELL")
                short_usd_1m = sum(item["notional"] for item in rows_1m if item["side"] == "BUY")
                total = long_usd + short_usd
                imbalance = (short_usd - long_usd) / total if total > 0 else 0.0
                previous_total = _previous_numeric(session, self.source_name, "liquidation_total_usd_5m", symbol) or 0.0
                spike = _clamp((total / max(previous_total, 50_000.0)) - 1.0, 0.0, 10.0)
                payload = {"mode": mode, "window": "5m", "events_used": len(rows)}
                values = {
                    "liquidation_long_usd_1m": long_usd_1m,
                    "liquidation_short_usd_1m": short_usd_1m,
                    "liquidation_long_usd_5m": long_usd,
                    "liquidation_short_usd_5m": short_usd,
                    "liquidation_total_usd_5m": total,
                    "liquidation_imbalance_5m": imbalance,
                    "liquidation_spike_score": spike,
                }
                for data_type, numeric in values.items():
                    saved += int(
                        _upsert_event(
                            session,
                            source_name=self.source_name,
                            data_type=data_type,
                            symbol=symbol,
                            event_time=event_time,
                            numeric_value=numeric,
                            payload=payload,
                            raw_payload={"events": rows} if settings.store_raw_liquidations else None,
                        )
                    )
            session.commit()
        return saved


class ExternalMarketCollectorManager:
    def __init__(self) -> None:
        self.collectors: dict[str, BaseExternalCollector] = {
            "fear_greed": FearGreedCollector(),
            "global_market": GlobalMarketCollector(),
            "stablecoin_risk": StablecoinRiskCollector(),
            "macro_risk": MacroRiskCollector(),
        }

    @property
    def any_enabled(self) -> bool:
        return any(collector.enabled for collector in self.collectors.values())

    async def run(self, stop_event: asyncio.Event, state: Any | None = None) -> None:
        if not self.any_enabled:
            if state:
                state.warning = "No external market collectors enabled"
                state.running = False
            return
        while not stop_event.is_set():
            result = await self.fetch_once()
            if state:
                rows = int(result.get("rows_saved", 0))
                state.mark_event({"rows_saved": rows, "collectors": result.get("collectors", {})})
                errors = [item.get("error") for item in result.get("collectors", {}).values() if item.get("error")]
                state.last_error = "; ".join(errors) if errors else None
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=max(settings.external_collector_interval_seconds, 60))
            except asyncio.TimeoutError:
                continue

    async def fetch_once(self, *, collector_name: str | None = None, mock: bool = False) -> dict[str, Any]:
        selected = self.collectors
        if collector_name:
            key = collector_name.lower()
            if key not in self.collectors:
                raise ValueError(f"Unknown external collector: {collector_name}")
            selected = {key: self.collectors[key]}
        rows_saved = 0
        collector_results: dict[str, Any] = {}
        for name, collector in selected.items():
            if not mock and not collector.enabled:
                collector_results[name] = {
                    "collector": collector.name,
                    "enabled": collector.enabled,
                    "mode": "disabled",
                    "rows_saved": 0,
                    "details": {"message": f"ENABLE_{collector.name.upper()}_COLLECTOR=false"},
                    "error": None,
                }
                continue
            try:
                result = await collector.fetch_once(mock=mock)
                rows_saved += result.rows_saved
                collector_results[name] = result.__dict__
            except Exception as exc:
                logger.exception("External collector failed: %s", name)
                collector_results[name] = {
                    "collector": collector.name,
                    "enabled": collector.enabled,
                    "mode": "mock" if mock else "live",
                    "rows_saved": 0,
                    "details": {},
                    "error": f"{type(exc).__name__}: {exc}",
                }
        return {"rows_saved": rows_saved, "collectors": collector_results}

    def status(self, session: Session) -> dict[str, Any]:
        return {
            "interval_seconds": settings.external_collector_interval_seconds,
            "collectors": {name: collector.status(session) for name, collector in self.collectors.items()},
        }


def _serialize_event(row: ExternalDataEvent) -> dict[str, Any]:
    return {
        "id": row.id,
        "source_name": row.source_name,
        "data_type": row.data_type,
        "symbol": row.symbol,
        "event_time": row.event_time.isoformat() if row.event_time else None,
        "numeric_value": row.numeric_value,
        "payload": row.payload,
    }
