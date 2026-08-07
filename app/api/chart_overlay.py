from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.api.routes import (
    _aggregate_from_base_candles,
    _latest_live_update,
    _serialize_live_candle_update,
    _serialize_stored_candle,
    _timeframe_seconds,
)
from app.db.models import AiDecision, Candle, LiveCandleUpdate, PaperTrade, Position
from app.db.session import get_session
from app.security import require_admin
from app.strategies.regime_pullback_v1 import STRATEGY_NAME

router = APIRouter(prefix="/api/chart", tags=["chart-overlay"], dependencies=[Depends(require_admin)])


def _dt(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@router.get("/candles")
def chart_candles(
    session: Session = Depends(get_session),
    symbol: str = "BTCUSDT",
    timeframe: str = "1m",
    limit: int = 1440,
) -> list[dict[str, Any]]:
    """Dashboard-oriented candle history with a larger safe window than the legacy endpoint."""

    symbol = symbol.upper()
    timeframe = timeframe.strip().lower() or "1m"
    safe_limit = min(max(limit, 1), 5000)
    target_seconds = _timeframe_seconds(timeframe)
    if target_seconds is None:
        raise HTTPException(status_code=400, detail=f"Unsupported timeframe: {timeframe}")

    rows = list(
        session.scalars(
            select(Candle)
            .where(Candle.symbol == symbol, Candle.interval == timeframe, Candle.is_closed.is_(True))
            .order_by(desc(Candle.open_time))
            .limit(safe_limit)
        )
    )
    rows.reverse()
    if rows:
        output = [_serialize_stored_candle(row) for row in rows]
        live_update = _latest_live_update(session, symbol, timeframe)
        if live_update and (not rows or live_update.open_time > rows[-1].open_time):
            output.append(_serialize_live_candle_update(live_update))
        return output[-safe_limit:]

    base_interval = "1m"
    base_seconds = 60
    requested_base_ratio = max(target_seconds // base_seconds, 1)
    base_limit = min(max(safe_limit * requested_base_ratio + requested_base_ratio, 300), 60000)
    base_rows = list(
        session.scalars(
            select(Candle)
            .where(Candle.symbol == symbol, Candle.interval == base_interval, Candle.is_closed.is_(True))
            .order_by(desc(Candle.open_time))
            .limit(base_limit)
        )
    )
    base_rows.reverse()
    live_base = _latest_live_update(session, symbol, base_interval)
    if live_base and (not base_rows or live_base.open_time > base_rows[-1].open_time):
        base_rows.append(live_base)  # type: ignore[arg-type]
    if not base_rows:
        return []

    if target_seconds < base_seconds:
        fallback_rows = base_rows[-safe_limit:]
        output: list[dict[str, Any]] = []
        for row in fallback_rows:
            if isinstance(row, LiveCandleUpdate):
                output.append(_serialize_live_candle_update(row, requested_timeframe=timeframe))
            else:
                output.append(
                    _serialize_stored_candle(
                        row,
                        requested_timeframe=timeframe,
                        source_name="1m_live_fallback",
                    )
                )
        return output

    return _aggregate_from_base_candles(
        base_rows,
        symbol=symbol,
        timeframe=timeframe,
        timeframe_seconds=target_seconds,
        limit=safe_limit,
    )


@router.get("/signals")
def chart_signals(
    session: Session = Depends(get_session),
    symbol: str = "BTCUSDT",
    limit: int = 200,
) -> dict[str, Any]:
    """Return paper-only trade/position events used to annotate the dashboard chart."""

    symbol = symbol.upper()
    safe_limit = min(max(limit, 1), 1000)

    positions = list(
        session.scalars(
            select(Position)
            .where(Position.symbol == symbol)
            .order_by(desc(Position.opened_at))
            .limit(safe_limit)
        )
    )
    trades = list(
        session.scalars(
            select(PaperTrade)
            .where(PaperTrade.symbol == symbol)
            .order_by(desc(PaperTrade.created_at))
            .limit(safe_limit)
        )
    )
    decisions = list(
        session.scalars(
            select(AiDecision)
            .where(
                AiDecision.symbol == symbol,
                AiDecision.strategy_name == STRATEGY_NAME,
                AiDecision.action != "HOLD",
            )
            .order_by(desc(AiDecision.created_at))
            .limit(safe_limit)
        )
    )

    return {
        "paper_only": True,
        "strategy": STRATEGY_NAME,
        "symbol": symbol,
        "positions": [
            {
                "id": row.id,
                "side": row.side,
                "quantity": row.quantity,
                "entry_price": row.entry_price,
                "current_price": row.current_price,
                "stop_loss": row.stop_loss,
                "take_profit": row.take_profit,
                "status": row.status,
                "realized_pnl": row.realized_pnl,
                "unrealized_pnl": row.unrealized_pnl,
                "opened_at": _dt(row.opened_at),
                "closed_at": _dt(row.closed_at),
            }
            for row in positions
        ],
        "trades": [
            {
                "id": row.id,
                "action": row.action,
                "side": row.side,
                "quantity": row.quantity,
                "price": row.price,
                "fee": row.fee,
                "realized_pnl": row.realized_pnl,
                "status": row.status,
                "reason": row.reason,
                "created_at": _dt(row.created_at),
            }
            for row in trades
        ],
        "decisions": [
            {
                "id": row.id,
                "action": row.action,
                "confidence": row.confidence,
                "execution_status": row.execution_status,
                "reason": row.reason or row.execution_message,
                "stop_loss": _float_or_none(row.stop_loss),
                "take_profit": _float_or_none(row.take_profit),
                "trade_id": row.trade_id,
                "created_at": _dt(row.created_at),
            }
            for row in decisions
        ],
    }
