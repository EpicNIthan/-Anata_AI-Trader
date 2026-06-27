from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.collectors.derivatives_collector import BinanceDerivativesCollector
from app.collectors.external_market_collectors import ExternalMarketCollectorManager, LiquidationCollector
from app.collectors.market_collector import BinanceMarketCollector
from app.collectors.news_collector import NewsCollector
from app.collectors.spot_context_collector import BinanceSpotContextCollector


@dataclass
class CollectorState:
    name: str
    running: bool = False
    messages: int = 0
    messages_received: int = 0
    rows_saved: int = 0
    last_event_at: str | None = None
    last_message_at: str | None = None
    last_saved_at: str | None = None
    last_error: str | None = None
    warning: str | None = None
    subscribed_streams: list[str] = field(default_factory=list)
    websocket_url: str | None = None
    closed_candles_only: bool = False
    details: dict[str, Any] | None = None

    def set_subscription(self, *, streams: list[str] | None = None, websocket_url: str | None = None) -> None:
        if streams is not None:
            self.subscribed_streams = streams
        if websocket_url is not None:
            self.websocket_url = websocket_url

    def mark_message(self, details: dict[str, Any] | None = None) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.messages += 1
        self.messages_received += 1
        self.last_event_at = now
        self.last_message_at = now
        self.details = details

    def mark_saved(self, rows: int = 1, details: dict[str, Any] | None = None) -> None:
        if details is not None:
            self.details = details
        self.last_event_at = datetime.now(timezone.utc).isoformat()
        if rows <= 0:
            return
        now = datetime.now(timezone.utc).isoformat()
        self.rows_saved += rows
        self.last_event_at = now
        self.last_saved_at = now

    def mark_event(self, details: dict[str, Any] | None = None) -> None:
        self.mark_message(details)
        saved_rows = 0
        if details:
            saved_rows = int(details.get("rows_saved", 0) or details.get("articles_stored", 0) or 0)
        self.mark_saved(saved_rows, details)
        self.last_error = None

    def mark_error(self, error: str) -> None:
        self.last_error = error
        self.last_event_at = datetime.now(timezone.utc).isoformat()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class WorkerManager:
    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._stop_events: dict[str, asyncio.Event] = {}
        self._states: dict[str, CollectorState] = {
            "market": CollectorState(name="market"),
            "spot_context": CollectorState(name="spot_context"),
            "news": CollectorState(name="news"),
            "derivatives": CollectorState(name="derivatives"),
            "external": CollectorState(name="external"),
            "liquidations": CollectorState(name="liquidations"),
        }

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {name: state.as_dict() for name, state in self._states.items()}

    async def start(self, name: str) -> dict[str, Any]:
        if name not in self._states:
            raise ValueError(f"Unknown worker: {name}")
        task = self._tasks.get(name)
        if task and not task.done():
            return self._states[name].as_dict()

        stop_event = asyncio.Event()
        self._stop_events[name] = stop_event
        state = self._states[name]
        state.last_error = None
        state.warning = None

        market_collector: BinanceMarketCollector | None = None
        spot_context_collector: BinanceSpotContextCollector | None = None
        news_collector: NewsCollector | None = None
        derivatives_collector: BinanceDerivativesCollector | None = None
        external_collector: ExternalMarketCollectorManager | None = None
        liquidation_collector: LiquidationCollector | None = None
        if name == "market":
            market_collector = BinanceMarketCollector()
            state.set_subscription(streams=market_collector.subscribed_streams, websocket_url=market_collector.stream_url)
            state.closed_candles_only = True
        elif name == "spot_context":
            spot_context_collector = BinanceSpotContextCollector()
            state.details = {
                "base_url": spot_context_collector.base_url,
                "symbols": spot_context_collector.symbols,
                "interval_seconds": spot_context_collector.interval_seconds,
            }
        elif name == "news":
            news_collector = NewsCollector()
            if not news_collector.can_collect:
                state.running = False
                state.warning = news_collector.unavailable_reason
                state.mark_error(news_collector.unavailable_reason)
                return state.as_dict()
        elif name == "derivatives":
            derivatives_collector = BinanceDerivativesCollector()
            state.details = {
                "base_url": derivatives_collector.base_url,
                "period": derivatives_collector.period,
                "symbols": derivatives_collector.symbols,
                "endpoints": derivatives_collector.endpoints,
            }
        elif name == "external":
            external_collector = ExternalMarketCollectorManager()
            state.details = {"collectors": list(external_collector.collectors)}
            if not external_collector.any_enabled:
                state.running = False
                state.warning = "No external market collectors enabled"
                state.mark_error(state.warning)
                return state.as_dict()
        elif name == "liquidations":
            liquidation_collector = LiquidationCollector()
            state.set_subscription(
                streams=[f"{symbol.lower()}@forceOrder" for symbol in liquidation_collector.symbols],
                websocket_url=liquidation_collector.stream_url,
            )
            if not liquidation_collector.enabled:
                state.running = False
                state.warning = "ENABLE_LIQUIDATION_COLLECTOR=false"
                state.mark_error(state.warning)
                return state.as_dict()

        state.running = True

        async def runner() -> None:
            try:
                if name == "market":
                    await (market_collector or BinanceMarketCollector()).run(stop_event, state)
                elif name == "spot_context":
                    await (spot_context_collector or BinanceSpotContextCollector()).run(stop_event, state)
                elif name == "news":
                    await (news_collector or NewsCollector()).run(stop_event, state)
                elif name == "derivatives":
                    await (derivatives_collector or BinanceDerivativesCollector()).run(stop_event, state)
                elif name == "external":
                    await (external_collector or ExternalMarketCollectorManager()).run(stop_event, state)
                elif name == "liquidations":
                    await (liquidation_collector or LiquidationCollector()).run(stop_event, state)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - defensive worker boundary
                state.mark_error(str(exc))
            finally:
                state.running = False

        self._tasks[name] = asyncio.create_task(runner(), name=f"{name}-collector")
        return state.as_dict()

    async def stop(self, name: str) -> dict[str, Any]:
        if name not in self._states:
            raise ValueError(f"Unknown worker: {name}")
        event = self._stop_events.get(name)
        if event:
            event.set()
        task = self._tasks.get(name)
        if task and not task.done():
            try:
                await asyncio.wait_for(task, timeout=5)
            except asyncio.TimeoutError:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._states[name].running = False
        return self._states[name].as_dict()

    async def stop_all(self) -> None:
        for name in list(self._states):
            await self.stop(name)
