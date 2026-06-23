from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.ai.experience_buffer import record_experience
from app.ai.strategy import RuleBasedStrategy
from app.api.webhooks import SignalRequest
from app.collectors.market_collector import BinanceMarketCollector, format_collector_error
from app.collectors.news_collector import NewsCollector
from app.db.models import (
    AccountEquity,
    AiDecision,
    ExperienceRecord,
    ExternalDataEvent,
    ModelVersion,
    PaperTrade,
    Position,
)
from app.db.session import get_session
from app.features.feature_builder import FeatureBuilder
from app.services.collector_status import latest_candles, latest_news, market_snapshot, news_snapshot
from app.trading.paper_engine import PaperEngine

router = APIRouter(prefix="/api", tags=["lab"])


def _dt(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _worker_manager(request: Request):
    manager = getattr(request.app.state, "worker_manager", None)
    if manager is None:
        raise HTTPException(status_code=503, detail="Worker manager is not initialized")
    return manager


def _auto_trader(request: Request):
    service = getattr(request.app.state, "auto_trader", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Auto trader is not initialized")
    return service


@router.get("/status")
def status(request: Request, session: Session = Depends(get_session)) -> dict[str, Any]:
    snapshot = PaperEngine(session).snapshot()
    return {
        "account": snapshot,
        "collectors": _worker_manager(request).snapshot(),
        "auto_trader": _auto_trader(request).status(),
    }


@router.post("/auto-trader/start")
async def start_auto_trader(request: Request) -> dict[str, Any]:
    return await _auto_trader(request).start()


@router.post("/auto-trader/stop")
async def stop_auto_trader(request: Request) -> dict[str, Any]:
    return await _auto_trader(request).stop()


@router.get("/auto-trader/status")
def auto_trader_status(request: Request) -> dict[str, Any]:
    return _auto_trader(request).status()


@router.get("/collectors/status")
def collector_status(request: Request) -> dict[str, Any]:
    return _worker_manager(request).snapshot()


@router.get("/market/status")
def market_status(request: Request, session: Session = Depends(get_session)) -> dict[str, Any]:
    state = _worker_manager(request).snapshot().get("market", {})
    return market_snapshot(session, state)


@router.get("/market/latest")
def market_latest(session: Session = Depends(get_session), limit: int = 50) -> list[dict[str, Any]]:
    return latest_candles(session, limit=limit)


@router.post("/market/backfill")
async def market_backfill(
    request: Request,
    payload: dict[str, Any] | None = Body(default=None),
) -> dict[str, Any]:
    payload = payload or {}
    symbols = payload.get("symbols")
    if isinstance(symbols, str):
        symbols = [item.strip().upper() for item in symbols.split(",") if item.strip()]
    interval = payload.get("interval")
    limit = int(payload.get("limit") or 100)
    mock = bool(payload.get("mock", False))
    collector = BinanceMarketCollector(symbols=symbols, interval=interval)
    state = _worker_manager(request)._states.get("market")
    try:
        result = await collector.backfill_all(limit=limit, mock=mock)
        if state:
            state.set_subscription(streams=collector.subscribed_streams, websocket_url=collector.stream_url)
            state.closed_candles_only = not collector.store_live_updates
            state.mark_saved(result["rows_saved"], {"backfill": result, "rows_saved": result["rows_saved"]})
            state.last_error = None
    except Exception as exc:
        message = format_collector_error(exc)
        if state:
            state.set_subscription(streams=collector.subscribed_streams, websocket_url=collector.stream_url)
            state.closed_candles_only = not collector.store_live_updates
            state.mark_error(f"Backfill failed: {message}")
        raise HTTPException(status_code=502, detail=f"Market backfill failed: {message}") from exc
    return result


@router.get("/news/status")
def news_status(request: Request, session: Session = Depends(get_session)) -> dict[str, Any]:
    state = _worker_manager(request).snapshot().get("news", {})
    return news_snapshot(session, state)


@router.get("/news/latest")
def news_latest(
    session: Session = Depends(get_session),
    limit: int = 25,
    provider: str | None = None,
) -> list[dict[str, Any]]:
    return latest_news(session, limit=limit, provider=provider)


@router.post("/news/run-once")
async def news_run_once(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    payload = payload or {}
    provider = payload.get("provider")
    if provider:
        provider = str(provider).lower()
    return await NewsCollector().fetch_once(provider_filter=provider)


@router.post("/news/mock")
def news_mock(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    payload = payload or {}
    stored = NewsCollector().store_mock_article(title=payload.get("title"), body=payload.get("body"))
    return {"rows_saved": stored, "source": "mock-news"}


@router.post("/collectors/{name}/start")
async def start_collector(name: str, request: Request) -> dict[str, Any]:
    if name not in {"market", "news"}:
        raise HTTPException(status_code=404, detail="Unknown collector")
    return await _worker_manager(request).start(name)


@router.post("/collectors/{name}/stop")
async def stop_collector(name: str, request: Request) -> dict[str, Any]:
    if name not in {"market", "news"}:
        raise HTTPException(status_code=404, detail="Unknown collector")
    return await _worker_manager(request).stop(name)


@router.post("/paper-trade")
def create_paper_trade(payload: SignalRequest, session: Session = Depends(get_session)) -> dict[str, Any]:
    engine = PaperEngine(session)
    result = engine.execute_signal(
        symbol=payload.symbol,
        action=payload.action,
        confidence=payload.confidence,
        reason=payload.reason,
        stop_loss=payload.stop_loss,
        take_profit=payload.take_profit,
        price=payload.price,
        quantity=payload.quantity,
        notional=payload.notional,
    )
    execution = {
        "status": result.status,
        "message": result.message,
        "trade_id": result.trade_id,
        "balance": result.balance,
        "equity": result.equity,
    }
    ai_decision = AiDecision(
        symbol=payload.symbol,
        strategy_name=payload.source,
        source_name=payload.source,
        action=payload.action,
        confidence=payload.confidence,
        reason=payload.reason,
        stop_loss=payload.stop_loss,
        take_profit=payload.take_profit,
        execution_status=result.status,
        execution_message=result.message,
        trade_id=result.trade_id,
        raw=payload.model_dump(),
        result=execution,
    )
    session.add(ai_decision)
    session.flush()
    record_experience(session, decision=ai_decision, feature=None, execution_result=execution)
    session.commit()
    return {
        "decision_id": ai_decision.id,
        **execution,
    }


@router.post("/strategy/{symbol}/paper")
def run_strategy(symbol: str, session: Session = Depends(get_session)) -> dict[str, Any]:
    feature = FeatureBuilder(session).build_for_symbol(symbol)
    decision = RuleBasedStrategy().decide(feature)
    engine_result = PaperEngine(session).execute_signal(
        symbol=symbol,
        action=decision.action,
        confidence=decision.confidence,
        reason=decision.reason,
        stop_loss=decision.stop_loss,
        take_profit=decision.take_profit,
    )
    execution = {
        "status": engine_result.status,
        "message": engine_result.message,
        "trade_id": engine_result.trade_id,
        "balance": engine_result.balance,
        "equity": engine_result.equity,
    }
    ai_decision = AiDecision(
        symbol=symbol.upper(),
        strategy_name=RuleBasedStrategy.name,
        source_name=RuleBasedStrategy.name,
        feature_id=feature.id,
        feature_schema_version=feature.schema_version,
        action=decision.action,
        confidence=decision.confidence,
        reason=decision.reason,
        stop_loss=decision.stop_loss,
        take_profit=decision.take_profit,
        execution_status=engine_result.status,
        execution_message=engine_result.message,
        trade_id=engine_result.trade_id,
        raw=decision.model_dump(),
        result=execution,
    )
    session.add(ai_decision)
    session.flush()
    record_experience(session, decision=ai_decision, feature=feature, execution_result=execution)
    session.commit()
    return {
        "feature_id": feature.id,
        "feature_schema_version": feature.schema_version,
        "decision": decision.model_dump(),
        "execution": execution,
    }


@router.post("/data-events")
def create_data_event(payload: dict[str, Any], session: Session = Depends(get_session)) -> dict[str, Any]:
    source_name = str(payload.get("source_name") or "unknown")
    data_type = str(payload.get("data_type") or "generic")
    symbol = payload.get("symbol")
    numeric_value = payload.get("numeric_value")
    event = ExternalDataEvent(
        source_name=source_name,
        data_type=data_type,
        symbol=str(symbol).upper() if symbol else None,
        numeric_value=float(numeric_value) if numeric_value not in (None, "") else None,
        payload=payload.get("payload") if isinstance(payload.get("payload"), dict) else None,
        raw_payload=payload,
    )
    session.add(event)
    session.commit()
    session.refresh(event)
    return {
        "id": event.id,
        "source_name": event.source_name,
        "data_type": event.data_type,
        "symbol": event.symbol,
        "event_time": _dt(event.event_time),
    }


@router.get("/data-events")
def data_events(session: Session = Depends(get_session), limit: int = 50) -> list[dict[str, Any]]:
    rows = session.scalars(select(ExternalDataEvent).order_by(desc(ExternalDataEvent.event_time)).limit(limit)).all()
    return [
        {
            "id": row.id,
            "source_name": row.source_name,
            "data_type": row.data_type,
            "symbol": row.symbol,
            "event_time": _dt(row.event_time),
            "numeric_value": row.numeric_value,
            "payload": row.payload,
        }
        for row in rows
    ]


@router.get("/experiences")
def experiences(session: Session = Depends(get_session), limit: int = 50) -> list[dict[str, Any]]:
    rows = session.scalars(select(ExperienceRecord).order_by(desc(ExperienceRecord.created_at)).limit(limit)).all()
    return [
        {
            "id": row.id,
            "symbol": row.symbol,
            "feature_schema_version": row.feature_schema_version,
            "action": row.action,
            "confidence": row.confidence,
            "reward": row.reward,
            "result": row.result,
            "created_at": _dt(row.created_at),
            "archived_at": _dt(row.archived_at),
        }
        for row in rows
    ]


@router.get("/trades")
def trades(session: Session = Depends(get_session), limit: int = 50) -> list[dict[str, Any]]:
    rows = session.scalars(select(PaperTrade).order_by(desc(PaperTrade.created_at)).limit(limit)).all()
    return [
        {
            "id": row.id,
            "symbol": row.symbol,
            "action": row.action,
            "quantity": row.quantity,
            "price": row.price,
            "notional": row.notional,
            "fee": row.fee,
            "realized_pnl": row.realized_pnl,
            "status": row.status,
            "created_at": _dt(row.created_at),
        }
        for row in rows
    ]


@router.get("/positions")
def positions(session: Session = Depends(get_session)) -> list[dict[str, Any]]:
    rows = session.scalars(select(Position).order_by(desc(Position.opened_at))).all()
    return [
        {
            "id": row.id,
            "symbol": row.symbol,
            "side": row.side,
            "quantity": row.quantity,
            "entry_price": row.entry_price,
            "current_price": row.current_price,
            "unrealized_pnl": row.unrealized_pnl,
            "realized_pnl": row.realized_pnl,
            "status": row.status,
            "opened_at": _dt(row.opened_at),
            "closed_at": _dt(row.closed_at),
        }
        for row in rows
    ]


@router.get("/equity")
def equity(session: Session = Depends(get_session), limit: int = 200) -> list[dict[str, Any]]:
    rows = session.scalars(select(AccountEquity).order_by(desc(AccountEquity.timestamp)).limit(limit)).all()
    rows = list(reversed(rows))
    return [
        {
            "timestamp": _dt(row.timestamp),
            "cash_balance": row.cash_balance,
            "equity": row.equity,
            "realized_pnl": row.realized_pnl,
            "unrealized_pnl": row.unrealized_pnl,
            "drawdown": row.drawdown,
        }
        for row in rows
    ]


@router.get("/models/latest")
def latest_model(session: Session = Depends(get_session)) -> dict[str, Any]:
    model = session.scalar(select(ModelVersion).order_by(desc(ModelVersion.created_at)).limit(1))
    if model is None:
        return {"name": "none", "version": "untrained", "status": "missing"}
    return {
        "model_id": model.model_id,
        "name": model.name,
        "version": model.version,
        "feature_schema_version": model.feature_schema_version,
        "feature_columns": model.feature_columns,
        "path": model.path,
        "status": model.status,
        "metrics": model.metrics,
        "created_at": _dt(model.created_at),
    }
