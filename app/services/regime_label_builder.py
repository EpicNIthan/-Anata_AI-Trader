from __future__ import annotations

"""Delayed future-label maintenance, isolated from the live decision path."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Candle
from app.strategies.regime_models import RegimeDecisionRecord, RegimeFutureLabelRecord
from app.strategies.regime_pullback_v1 import CandleBar, Regime, build_future_label, dec, resample_complete_bars

HORIZONS = (15, 60, 240, 720)


class RegimeLabelBuilder:
    def __init__(self, session: Session) -> None:
        self.session = session

    def run(self, *, now: datetime | None = None, limit: int = 500) -> dict[str, Any]:
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        decisions = self.session.scalars(
            select(RegimeDecisionRecord)
            .where(RegimeDecisionRecord.candle_close_time <= now - timedelta(minutes=15))
            .order_by(RegimeDecisionRecord.candle_close_time)
            .limit(limit)
        ).all()
        inserted = 0
        unavailable = 0
        for decision in decisions:
            values = decision.indicator_values or {}
            price = float(values.get("close") or 0.0)
            atr = float(values.get("atr15m") or 0.0)
            if price <= 0 or atr <= 0:
                unavailable += 1
                continue
            direction = "SHORT" if decision.regime == Regime.SHORT.value else "LONG"
            stop = price - 1.5 * atr if direction == "LONG" else price + 1.5 * atr
            target = price + 2.5 * atr if direction == "LONG" else price - 2.5 * atr
            future = self._future_bars(decision.symbol, decision.candle_close_time, now)
            for horizon in HORIZONS:
                if decision.candle_close_time + timedelta(minutes=horizon) > now:
                    unavailable += 1
                    continue
                exists = self.session.scalar(
                    select(RegimeFutureLabelRecord.id).where(
                        RegimeFutureLabelRecord.decision_id == decision.id,
                        RegimeFutureLabelRecord.horizon_minutes == horizon,
                    )
                )
                if exists:
                    continue
                label = build_future_label(
                    decision_close=decision.candle_close_time,
                    decision_price=price,
                    direction=direction,
                    future_bars=future,
                    horizon_minutes=horizon,
                    stop_price=stop,
                    target_price=target,
                    estimated_round_trip_cost_rate=float(dec(decision.fee_rate) * Decimal("2") + dec(decision.slippage_rate) * Decimal("2") + dec(decision.spread_bps or 0) / Decimal("10000")),
                )
                if not label.available:
                    unavailable += 1
                    continue
                self.session.add(
                    RegimeFutureLabelRecord(
                        decision_id=decision.id,
                        horizon_minutes=horizon,
                        available_at=decision.candle_close_time + timedelta(minutes=horizon),
                        future_return=dec(label.future_return),
                        future_return_after_costs=dec(label.future_return_after_costs),
                        maximum_favorable_excursion=dec(label.maximum_favorable_excursion),
                        maximum_adverse_excursion=dec(label.maximum_adverse_excursion),
                        stop_reached_first=label.stop_reached_first,
                        target_reached_first=label.target_reached_first,
                        time_to_stop_seconds=label.time_to_stop_seconds,
                        time_to_target_seconds=label.time_to_target_seconds,
                        decision_regime=decision.regime,
                        data_complete=label.data_complete,
                        payload={"strategy_version": decision.strategy_version, "decision_action": decision.action, "neutral_direction_assumption": decision.regime == Regime.NEUTRAL.value},
                    )
                )
                inserted += 1
        self.session.commit()
        return {"inserted": inserted, "unavailable": unavailable, "horizons_minutes": list(HORIZONS)}

    def _future_bars(self, symbol: str, decision_close: datetime, now: datetime) -> list[CandleBar]:
        rows = self.session.scalars(
            select(Candle)
            .where(
                Candle.symbol == symbol,
                Candle.interval == "1m",
                Candle.is_closed.is_(True),
                Candle.open_time >= decision_close,
                Candle.close_time <= now,
            )
            .order_by(Candle.open_time)
        ).all()
        one_minute = [
            CandleBar(
                open_time=row.open_time,
                close_time=row.close_time or row.open_time + timedelta(minutes=1),
                open=row.open,
                high=row.high,
                low=row.low,
                close=row.close,
                volume=row.volume,
                complete=bool(row.is_closed),
            )
            for row in rows
        ]
        return resample_complete_bars(one_minute, 15)


class RegimeLabelMaintenanceService:
    """Low-frequency delayed-label worker; never called from the live decision path."""

    def __init__(self, interval_seconds: int = 300) -> None:
        import asyncio
        self.interval_seconds = max(interval_seconds, 60)
        self._task = None
        self._stop_event = None
        self.running = False
        self.last_result: dict[str, Any] | None = None
        self.last_error: str | None = None
        self._asyncio = asyncio

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop_event = self._asyncio.Event()
        self.running = True
        self._task = self._asyncio.create_task(self._loop(), name="regime-future-labels")

    async def stop(self) -> None:
        if self._stop_event:
            self._stop_event.set()
        if self._task and not self._task.done():
            try:
                await self._asyncio.wait_for(self._task, timeout=5)
            except self._asyncio.TimeoutError:
                self._task.cancel()
        self.running = False

    async def _loop(self) -> None:
        from app.db.session import SessionLocal
        try:
            while not self._stop_event.is_set():
                try:
                    with SessionLocal() as session:
                        self.last_result = await self._asyncio.to_thread(RegimeLabelBuilder(session).run)
                        self.last_error = None
                except Exception as exc:
                    self.last_error = str(exc)
                try:
                    await self._asyncio.wait_for(self._stop_event.wait(), timeout=self.interval_seconds)
                except self._asyncio.TimeoutError:
                    pass
        finally:
            self.running = False
