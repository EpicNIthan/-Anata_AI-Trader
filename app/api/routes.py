from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.ai.experience_buffer import record_experience
from app.ai.model_strategy import PriceModelStrategy
from app.ai.news_sentiment import active_sentiment_model, analyze_news, reset_sentiment_model_cache
from app.ai.strategy import RuleBasedStrategy
from app.api.webhooks import SignalRequest
from app.config import settings
from app.collectors.derivatives_collector import BinanceDerivativesCollector
from app.collectors.market_collector import BinanceMarketCollector, format_collector_error
from app.collectors.news_collector import NewsCollector
from app.db.models import (
    AccountEquity,
    AiDecision,
    Candle,
    ExperienceRecord,
    ExternalDataEvent,
    Feature,
    LiveCandleUpdate,
    ModelVersion,
    NewsArticle,
    NewsSentiment,
    PaperTrade,
    Position,
    TrainingFeature,
    TrainingRun,
)
from app.db.session import get_session
from app.features.feature_builder import FeatureBuilder
from app.features.schema import CURRENT_FEATURE_SCHEMA_VERSION, columns_for_schema, values_from_feature
from app.security import require_admin
from app.services.collector_status import latest_candles, latest_news, market_snapshot, news_snapshot
from app.services.db_diagnostics import database_diagnostics
from app.training.dataset_accelerator import build_accelerated_dataset
from app.training.export_dataset import export_dataset, parse_since_date
from app.training.train_price_model import train_price_model
from app.trading.paper_engine import PaperEngine

router = APIRouter(prefix="/api", tags=["lab"])

INSPECTOR_FEATURE_KEYS = [
    "sentiment_score",
    "sentiment_confidence",
    "risk_score",
    "impact_score",
    "recency_weight",
    "btc_related",
    "eth_related",
    "macro_related",
    "candle_return_1m",
    "candle_return_5m",
    "volatility",
    "volume_change",
    "trend_score",
    "crowd_long_account_pct",
    "crowd_short_account_pct",
    "crowd_long_short_ratio",
    "top_trader_long_account_pct",
    "top_trader_position_long_pct",
    "taker_buy_pressure",
    "taker_buy_sell_ratio",
    "open_interest_value",
    "open_interest_change",
    "funding_rate",
    "trader_crowd_score",
    "crowd_risk_score",
    "derivatives_recency_weight",
]


def _dt(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _timeframe_seconds(value: str | None) -> int | None:
    timeframe = (value or "1m").strip().lower()
    if len(timeframe) < 2:
        return None
    try:
        amount = int(timeframe[:-1])
    except ValueError:
        return None
    if amount <= 0:
        return None
    multiplier = {"s": 1, "m": 60, "h": 3600, "d": 86400}.get(timeframe[-1])
    if multiplier is None:
        return None
    return amount * multiplier


def _serialize_candle(
    *,
    candle_id: int | None,
    symbol: str,
    timeframe: str,
    open_time: datetime | None,
    close_time: datetime | None,
    open_price: float,
    high: float,
    low: float,
    close: float,
    volume: float,
    is_closed: bool,
    source_name: str | None,
    base_interval: str | None = None,
) -> dict[str, Any]:
    return {
        "id": candle_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "base_interval": base_interval or timeframe,
        "source_name": source_name,
        "time": int(open_time.timestamp()) if open_time else None,
        "open_time": _dt(open_time),
        "close_time": _dt(close_time),
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "is_closed": is_closed,
    }


def _serialize_stored_candle(row: Candle, requested_timeframe: str | None = None, source_name: str | None = None) -> dict[str, Any]:
    return _serialize_candle(
        candle_id=row.id,
        symbol=row.symbol,
        timeframe=requested_timeframe or row.interval,
        open_time=row.open_time,
        close_time=row.close_time,
        open_price=row.open,
        high=row.high,
        low=row.low,
        close=row.close,
        volume=row.volume,
        is_closed=row.is_closed,
        source_name=source_name or row.source_name,
        base_interval=row.interval,
    )


def _serialize_live_candle_update(row: LiveCandleUpdate, requested_timeframe: str | None = None) -> dict[str, Any]:
    return _serialize_candle(
        candle_id=row.id,
        symbol=row.symbol,
        timeframe=requested_timeframe or row.interval,
        open_time=row.open_time,
        close_time=row.close_time,
        open_price=row.open,
        high=row.high,
        low=row.low,
        close=row.close,
        volume=row.volume,
        is_closed=False,
        source_name=row.source_name or "live_candle_updates",
        base_interval=row.interval,
    ) | {"update_count": row.update_count, "event_time": _dt(row.event_time)}


def _latest_live_update(session: Session, symbol: str, interval: str) -> LiveCandleUpdate | None:
    return session.scalar(
        select(LiveCandleUpdate)
        .where(LiveCandleUpdate.symbol == symbol, LiveCandleUpdate.interval == interval)
        .order_by(desc(LiveCandleUpdate.open_time))
        .limit(1)
    )


def _aggregate_from_base_candles(
    rows: list[Candle],
    *,
    symbol: str,
    timeframe: str,
    timeframe_seconds: int,
    limit: int,
) -> list[dict[str, Any]]:
    buckets: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    current_bucket_start: int | None = None

    for row in rows:
        if not row.open_time:
            continue
        row_start = int(row.open_time.timestamp())
        bucket_start = row_start - (row_start % timeframe_seconds)
        bucket_open_time = datetime.fromtimestamp(bucket_start, tz=timezone.utc)
        if current is None or bucket_start != current_bucket_start:
            if current is not None:
                buckets.append(current)
            current_bucket_start = bucket_start
            current = {
                "open_time": bucket_open_time,
                "close_time": row.close_time,
                "open": row.open,
                "high": row.high,
                "low": row.low,
                "close": row.close,
                "volume": row.volume or 0.0,
                "is_closed": bool(getattr(row, "is_closed", False)),
            }
            continue

        current["close_time"] = row.close_time or current["close_time"]
        current["high"] = max(_float(current["high"]), row.high)
        current["low"] = min(_float(current["low"]), row.low)
        current["close"] = row.close
        current["volume"] = _float(current["volume"]) + _float(row.volume)
        current["is_closed"] = bool(current["is_closed"]) and bool(getattr(row, "is_closed", False))

    if current is not None:
        buckets.append(current)

    return [
        _serialize_candle(
            candle_id=None,
            symbol=symbol,
            timeframe=timeframe,
            open_time=bucket["open_time"],
            close_time=bucket["close_time"],
            open_price=_float(bucket["open"]),
            high=_float(bucket["high"]),
            low=_float(bucket["low"]),
            close=_float(bucket["close"]),
            volume=_float(bucket["volume"]),
            is_closed=bool(bucket["is_closed"]),
            source_name="aggregated_from_1m",
            base_interval="1m",
        )
        for bucket in buckets[-limit:]
    ]


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


def _data_lifecycle(request: Request):
    service = getattr(request.app.state, "data_lifecycle", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Data lifecycle service is not initialized")
    return service


def _derivatives_snapshot(session: Session, worker_state: dict[str, Any] | None = None) -> dict[str, Any]:
    worker_state = worker_state or {}
    collector = BinanceDerivativesCollector()
    status = collector.status(session)
    return {
        **status,
        "running": bool(worker_state.get("running")),
        "messages_received": worker_state.get("messages_received", 0),
        "rows_saved": worker_state.get("rows_saved", 0),
        "last_message_at": worker_state.get("last_message_at"),
        "last_saved_at": worker_state.get("last_saved_at"),
        "last_error": worker_state.get("last_error"),
        "details": worker_state.get("details"),
    }


def _decision_source(row: AiDecision) -> str:
    raw = row.raw or row.raw_payload or {}
    source = raw.get("decision_source") if isinstance(raw, dict) else None
    if source:
        return str(source)
    if row.model_version_id or (row.source_name and "model" in row.source_name):
        return "model"
    if row.source_name and "exploration" in row.source_name:
        return "exploration"
    return "strategy"


def _trading_diagnostics(session: Session) -> dict[str, Any]:
    decisions = list(session.scalars(select(AiDecision).order_by(desc(AiDecision.created_at)).limit(500)))
    strategy_trades = 0
    exploration_trades = 0
    skipped_trades = 0
    hold_reasons: list[dict[str, Any]] = []
    for decision in decisions:
        source = _decision_source(decision)
        filled = decision.trade_id is not None and decision.execution_status == "FILLED"
        if filled and source == "exploration":
            exploration_trades += 1
        elif filled:
            strategy_trades += 1
        if not filled:
            skipped_trades += 1
        if decision.action == "HOLD" or decision.execution_status in {"HELD", "REJECTED"}:
            hold_reasons.append(
                {
                    "time": _dt(decision.created_at),
                    "symbol": decision.symbol,
                    "action": decision.action,
                    "status": decision.execution_status,
                    "confidence": decision.confidence,
                    "reason": decision.reason or decision.execution_message,
                    "decision_source": source,
                }
            )
    last_decision = decisions[0] if decisions else None
    latest_warning = None
    if last_decision and last_decision.trade_id is None and last_decision.execution_status in {"HELD", "REJECTED"}:
        latest_warning = (
            f"No paper trade opened: {last_decision.action} {last_decision.execution_status}. "
            f"{last_decision.reason or last_decision.execution_message or 'No reason recorded.'}"
        )
    return {
        "strategy_trades": strategy_trades,
        "exploration_trades": exploration_trades,
        "skipped_trades": skipped_trades,
        "hold_reasons": hold_reasons[:10],
        "latest_warning": latest_warning,
        "last_strategy_action": {
            "time": _dt(last_decision.created_at) if last_decision else None,
            "symbol": last_decision.symbol if last_decision else None,
            "action": last_decision.action if last_decision else None,
            "confidence": last_decision.confidence if last_decision else None,
            "status": last_decision.execution_status if last_decision else None,
            "reason": (last_decision.reason or last_decision.execution_message) if last_decision else None,
            "decision_source": _decision_source(last_decision) if last_decision else None,
        },
    }


@router.get("/status")
def status(request: Request, session: Session = Depends(get_session)) -> dict[str, Any]:
    snapshot = PaperEngine(session).snapshot()
    return {
        "account": snapshot,
        "collectors": _worker_manager(request).snapshot(),
        "auto_trader": _auto_trader(request).status(),
    }


@router.get("/dashboard/summary")
def dashboard_summary(request: Request, session: Session = Depends(get_session)) -> dict[str, Any]:
    account = PaperEngine(session).snapshot()
    collectors = _worker_manager(request).snapshot()
    latest_model = session.scalar(select(ModelVersion).order_by(desc(ModelVersion.created_at)).limit(1))
    latest_training_run = session.scalar(select(TrainingRun).order_by(desc(TrainingRun.started_at)).limit(1))
    latest_decision = session.scalar(select(AiDecision).order_by(desc(AiDecision.created_at)).limit(1))
    return {
        "account": account,
        "mode": "paper",
        "paper_start_balance": settings.paper_start_balance,
        "market": market_snapshot(session, collectors.get("market", {})),
        "news": news_snapshot(session, collectors.get("news", {})),
        "derivatives": _derivatives_snapshot(session, collectors.get("derivatives", {})),
        "collectors": collectors,
        "auto_trader": _auto_trader(request).status(),
        "trading": _trading_diagnostics(session),
        "sentiment_model": active_sentiment_model(),
        "model": {
            "model_id": latest_model.model_id if latest_model else None,
            "name": latest_model.name if latest_model else "none",
            "version": latest_model.version if latest_model else "untrained",
            "feature_schema_version": latest_model.feature_schema_version if latest_model else CURRENT_FEATURE_SCHEMA_VERSION,
            "status": latest_model.status if latest_model else "missing",
        },
        "training": {
            "last_run_at": _dt(latest_training_run.started_at if latest_training_run else None),
            "status": latest_training_run.status if latest_training_run else "none",
            "feature_schema_version": latest_training_run.feature_schema_version if latest_training_run else CURRENT_FEATURE_SCHEMA_VERSION,
        },
        "counts": {
            "candles": session.scalar(select(func.count(Candle.id))) or 0,
            "news": session.scalar(select(func.count(NewsArticle.id))) or 0,
            "sentiment": session.scalar(select(func.count(NewsSentiment.id))) or 0,
            "features": session.scalar(select(func.count(Feature.id))) or 0,
            "training_features": session.scalar(select(func.count(TrainingFeature.id))) or 0,
            "experiences": session.scalar(select(func.count(ExperienceRecord.id))) or 0,
            "external_data_events": session.scalar(select(func.count(ExternalDataEvent.id))) or 0,
            "trades": session.scalar(select(func.count(PaperTrade.id))) or 0,
            "open_positions": session.scalar(select(func.count(Position.id)).where(Position.status == "OPEN")) or 0,
        },
        "latest_decision": {
            "id": latest_decision.id,
            "symbol": latest_decision.symbol,
            "action": latest_decision.action,
            "confidence": latest_decision.confidence,
            "created_at": _dt(latest_decision.created_at),
        }
        if latest_decision
        else None,
        "server_time": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/auto-trader/start")
async def start_auto_trader(request: Request, _: None = Depends(require_admin)) -> dict[str, Any]:
    return await _auto_trader(request).start()


@router.post("/auto-trader/stop")
async def stop_auto_trader(request: Request, _: None = Depends(require_admin)) -> dict[str, Any]:
    return await _auto_trader(request).stop()


@router.get("/auto-trader/status")
def auto_trader_status(request: Request) -> dict[str, Any]:
    return _auto_trader(request).status()


@router.get("/collectors/status")
def collector_status(request: Request) -> dict[str, Any]:
    return _worker_manager(request).snapshot()


@router.get("/derivatives/status")
def derivatives_status(request: Request, session: Session = Depends(get_session)) -> dict[str, Any]:
    state = _worker_manager(request).snapshot().get("derivatives", {})
    return _derivatives_snapshot(session, state)


@router.get("/derivatives/latest")
def derivatives_latest(
    session: Session = Depends(get_session),
    symbol: str | None = None,
    data_type: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    query = select(ExternalDataEvent).where(ExternalDataEvent.source_name.like("binance_futures_%"))
    if symbol:
        query = query.where(ExternalDataEvent.symbol == symbol.upper())
    if data_type:
        query = query.where(ExternalDataEvent.data_type == data_type)
    rows = session.scalars(query.order_by(desc(ExternalDataEvent.event_time)).limit(min(max(limit, 1), 200))).all()
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


@router.post("/derivatives/run-once")
async def derivatives_run_once(
    payload: dict[str, Any] | None = Body(default=None),
    _: None = Depends(require_admin),
) -> dict[str, Any]:
    payload = payload or {}
    symbols = payload.get("symbols")
    if isinstance(symbols, str):
        symbols = [item.strip().upper() for item in symbols.split(",") if item.strip()]
    if symbols is not None and not isinstance(symbols, list):
        raise HTTPException(status_code=400, detail="symbols must be a list or comma-separated string")
    period = payload.get("period")
    mock = bool(payload.get("mock", False))
    collector = BinanceDerivativesCollector(symbols=symbols, period=period)
    return await collector.fetch_once(mock=mock)


@router.get("/db/diagnostics")
def db_diagnostics(session: Session = Depends(get_session)) -> dict[str, Any]:
    return database_diagnostics(session)


@router.get("/db/lifecycle/status")
def db_lifecycle_status(request: Request) -> dict[str, Any]:
    return _data_lifecycle(request).status()


@router.post("/db/cleanup")
def db_cleanup(request: Request, _: None = Depends(require_admin)) -> dict[str, Any]:
    return _data_lifecycle(request).run_cleanup_once()


@router.post("/db/archive")
def db_archive(
    request: Request,
    payload: dict[str, Any] | None = Body(default=None),
    _: None = Depends(require_admin),
) -> dict[str, Any]:
    payload = payload or {}
    tables = payload.get("tables")
    if isinstance(tables, str):
        tables = [item.strip() for item in tables.split(",") if item.strip()]
    if tables is not None and not isinstance(tables, list):
        raise HTTPException(status_code=400, detail="tables must be a list or comma-separated string")
    before = parse_since_date(payload.get("before_date") or payload.get("before"))
    delete_after_archive = bool(payload.get("delete_after_archive", False))
    return _data_lifecycle(request).archive_once(
        before=before,
        tables=tables,
        delete_after_archive=delete_after_archive,
    )


@router.get("/market/status")
def market_status(request: Request, session: Session = Depends(get_session)) -> dict[str, Any]:
    state = _worker_manager(request).snapshot().get("market", {})
    return market_snapshot(session, state)


@router.get("/market/latest")
def market_latest(
    session: Session = Depends(get_session),
    limit: int = 50,
    symbol: str | None = None,
) -> list[dict[str, Any]] | dict[str, Any]:
    if symbol:
        normalized_symbol = symbol.upper()
        candle = session.scalar(
            select(Candle)
            .where(Candle.symbol == normalized_symbol, Candle.is_closed.is_(True))
            .order_by(desc(Candle.open_time))
            .limit(1)
        )
        live_update = _latest_live_update(session, normalized_symbol, settings.binance_interval)
        if candle is None and live_update is None:
            return {}
        if live_update and (candle is None or live_update.open_time >= candle.open_time):
            return _serialize_live_candle_update(live_update)
        return {
            "id": candle.id,
            "symbol": candle.symbol,
            "timeframe": candle.interval,
            "open_time": _dt(candle.open_time),
            "close_time": _dt(candle.close_time),
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume,
            "is_closed": candle.is_closed,
        }
    return latest_candles(session, limit=limit)


@router.get("/market/candles")
def market_candles(
    session: Session = Depends(get_session),
    symbol: str = "BTCUSDT",
    timeframe: str = "1m",
    limit: int = 300,
) -> list[dict[str, Any]]:
    symbol = symbol.upper()
    timeframe = timeframe.strip().lower() or "1m"
    safe_limit = min(max(limit, 1), 1000)
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
    base_limit = min(max(safe_limit * requested_base_ratio + requested_base_ratio, 200), 6000)
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
        output = []
        for row in fallback_rows:
            if isinstance(row, LiveCandleUpdate):
                output.append(_serialize_live_candle_update(row, requested_timeframe=timeframe))
            else:
                output.append(_serialize_stored_candle(row, requested_timeframe=timeframe, source_name="1m_live_fallback"))
        return output

    return _aggregate_from_base_candles(
        base_rows,
        symbol=symbol,
        timeframe=timeframe,
        timeframe_seconds=target_seconds,
        limit=safe_limit,
    )


@router.post("/market/backfill")
async def market_backfill(
    request: Request,
    payload: dict[str, Any] | None = Body(default=None),
    _: None = Depends(require_admin),
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
            state.closed_candles_only = True
            state.mark_saved(result["rows_saved"], {"backfill": result, "rows_saved": result["rows_saved"]})
            state.last_error = None
    except Exception as exc:
        message = format_collector_error(exc)
        if state:
            state.set_subscription(streams=collector.subscribed_streams, websocket_url=collector.stream_url)
            state.closed_candles_only = True
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
async def news_run_once(
    payload: dict[str, Any] | None = Body(default=None),
    _: None = Depends(require_admin),
) -> dict[str, Any]:
    payload = payload or {}
    provider = payload.get("provider")
    if provider:
        provider = str(provider).lower()
    return await NewsCollector().fetch_once(provider_filter=provider)


@router.post("/news/mock")
def news_mock(
    payload: dict[str, Any] | None = Body(default=None),
    _: None = Depends(require_admin),
) -> dict[str, Any]:
    payload = payload or {}
    stored = NewsCollector().store_mock_article(title=payload.get("title"), body=payload.get("body"))
    return {"rows_saved": stored, "source": "mock-news"}


@router.post("/collectors/{name}/start")
async def start_collector(name: str, request: Request, _: None = Depends(require_admin)) -> dict[str, Any]:
    if name not in {"market", "news", "derivatives"}:
        raise HTTPException(status_code=404, detail="Unknown collector")
    return await _worker_manager(request).start(name)


@router.post("/collectors/{name}/stop")
async def stop_collector(name: str, request: Request, _: None = Depends(require_admin)) -> dict[str, Any]:
    if name not in {"market", "news", "derivatives"}:
        raise HTTPException(status_code=404, detail="Unknown collector")
    return await _worker_manager(request).stop(name)


@router.post("/paper-trade")
def create_paper_trade(
    payload: SignalRequest,
    session: Session = Depends(get_session),
    _: None = Depends(require_admin),
) -> dict[str, Any]:
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
def run_strategy(
    symbol: str,
    session: Session = Depends(get_session),
    _: None = Depends(require_admin),
) -> dict[str, Any]:
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
def create_data_event(
    payload: dict[str, Any],
    session: Session = Depends(get_session),
    _: None = Depends(require_admin),
) -> dict[str, Any]:
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


@router.post("/training/export")
def training_export(
    payload: dict[str, Any] | None = Body(default=None),
    session: Session = Depends(get_session),
    _: None = Depends(require_admin),
) -> dict[str, Any]:
    payload = payload or {}
    feature_schema_version = str(payload.get("feature_schema_version") or CURRENT_FEATURE_SCHEMA_VERSION)
    since_date = parse_since_date(payload.get("since_date"))
    use_all_data = bool(payload.get("use_all_data", True))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = Path("datasets") / f"features_{timestamp}.csv"
    exported_path = export_dataset(
        output_path,
        feature_schema_version=feature_schema_version,
        since_date=since_date,
        use_all_data=use_all_data,
    )
    counts = {
        "candles": session.scalar(select(func.count(Candle.id))) or 0,
        "news_articles": session.scalar(select(func.count(NewsArticle.id))) or 0,
        "news_sentiment": session.scalar(select(func.count(NewsSentiment.id))) or 0,
        "features": session.scalar(select(func.count(Feature.id))) or 0,
        "training_features": session.scalar(select(func.count(TrainingFeature.id))) or 0,
        "ai_decisions": session.scalar(select(func.count(AiDecision.id))) or 0,
        "experiences": session.scalar(select(func.count(ExperienceRecord.id))) or 0,
        "paper_trades": session.scalar(select(func.count(PaperTrade.id))) or 0,
        "open_positions": session.scalar(select(func.count(Position.id)).where(Position.status == "OPEN")) or 0,
    }
    news_by_provider = {
        provider: count
        for provider, count in session.execute(
            select(NewsArticle.source_name, func.count(NewsArticle.id)).group_by(NewsArticle.source_name)
        ).all()
    }
    features_by_symbol = {
        symbol: count
        for symbol, count in session.execute(
            select(TrainingFeature.symbol, func.count(TrainingFeature.id)).group_by(TrainingFeature.symbol)
        ).all()
    }
    latest_feature = session.scalar(select(Feature).order_by(desc(Feature.as_of)).limit(1))
    return {
        "exported_path": str(exported_path),
        "feature_schema_version": feature_schema_version,
        "since_date": since_date.isoformat() if since_date else None,
        "use_all_data": use_all_data,
        "counts": counts,
        "news_by_provider": news_by_provider,
        "features_by_symbol": features_by_symbol,
        "latest_feature_at": _dt(latest_feature.as_of if latest_feature else None),
    }


@router.post("/training/build-dataset")
async def training_build_dataset(
    payload: dict[str, Any] | None = Body(default=None),
    _: None = Depends(require_admin),
) -> dict[str, Any]:
    payload = payload or {}
    symbols = payload.get("symbols")
    if isinstance(symbols, str):
        symbols = [item.strip().upper() for item in symbols.split(",") if item.strip()]
    if symbols is not None and not isinstance(symbols, list):
        raise HTTPException(status_code=400, detail="symbols must be a list or comma-separated string")
    return await build_accelerated_dataset(
        symbols=symbols,
        interval=str(payload.get("interval") or settings.paper_trade_timeframe),
        days=min(max(int(payload.get("days") or 14), 1), 365),
        max_rows_per_symbol=min(max(int(payload.get("max_rows_per_symbol") or 5000), 100), 250_000),
        lookback=min(max(int(payload.get("lookback") or 60), 10), 500),
        stride=min(max(int(payload.get("stride") or 5), 1), 500),
        replay_limit=min(max(int(payload.get("replay_limit") or 20_000), 100), 500_000),
        backfill=bool(payload.get("backfill", True)),
        mock=bool(payload.get("mock", False)),
        export=bool(payload.get("export", True)),
    )


@router.post("/training/train-model")
async def training_train_model(
    payload: dict[str, Any] | None = Body(default=None),
    session: Session = Depends(get_session),
    _: None = Depends(require_admin),
) -> dict[str, Any]:
    payload = payload or {}
    build_dataset = bool(payload.get("build_dataset", False))
    dataset_path_value = payload.get("dataset_path")
    dataset_summary: dict[str, Any] | None = None

    if build_dataset:
        symbols = payload.get("symbols")
        if isinstance(symbols, str):
            symbols = [item.strip().upper() for item in symbols.split(",") if item.strip()]
        if symbols is not None and not isinstance(symbols, list):
            raise HTTPException(status_code=400, detail="symbols must be a list or comma-separated string")
        dataset_summary = await build_accelerated_dataset(
            symbols=symbols,
            interval=str(payload.get("interval") or settings.paper_trade_timeframe),
            days=min(max(int(payload.get("days") or 14), 1), 365),
            max_rows_per_symbol=min(max(int(payload.get("max_rows_per_symbol") or 5000), 100), 250_000),
            lookback=min(max(int(payload.get("lookback") or 60), 10), 500),
            stride=min(max(int(payload.get("stride") or 5), 1), 500),
            replay_limit=min(max(int(payload.get("replay_limit") or 20_000), 100), 500_000),
            backfill=bool(payload.get("backfill", True)),
            mock=bool(payload.get("mock", False)),
            export=True,
        )
        dataset_path_value = dataset_summary.get("exported_path")

    dataset_path = Path(dataset_path_value) if dataset_path_value else None
    from_checkpoint_value = payload.get("from_checkpoint")
    from_checkpoint: Path | None = None
    if from_checkpoint_value == "latest":
        latest = session.scalar(select(ModelVersion).where(ModelVersion.status == "trained").order_by(desc(ModelVersion.created_at)).limit(1))
        from_checkpoint = Path(latest.path) if latest else None
    elif from_checkpoint_value:
        from_checkpoint = Path(str(from_checkpoint_value))

    feature_schema_version = str(payload.get("feature_schema_version") or CURRENT_FEATURE_SCHEMA_VERSION)
    since_date = parse_since_date(payload.get("since_date"))
    use_all_data = bool(payload.get("use_all_data", True))
    epochs = min(max(int(payload.get("epochs") or 500), 1), 20_000)
    learning_rate = float(payload.get("learning_rate") or 0.05)

    model_path = await asyncio.to_thread(
        train_price_model,
        dataset_path,
        from_checkpoint=from_checkpoint,
        since_date=since_date,
        use_all_data=use_all_data,
        feature_schema_version=feature_schema_version,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    session.expire_all()
    model = session.scalar(select(ModelVersion).order_by(desc(ModelVersion.created_at)).limit(1))
    return {
        "status": "trained",
        "model_path": str(model_path),
        "dataset_path": str(dataset_path) if dataset_path else None,
        "dataset_summary": dataset_summary,
        "model": {
            "id": model.id if model else None,
            "model_id": model.model_id if model else None,
            "name": model.name if model else None,
            "version": model.version if model else None,
            "feature_schema_version": model.feature_schema_version if model else feature_schema_version,
            "feature_columns": model.feature_columns if model else None,
            "metrics": model.metrics if model else None,
            "path": model.path if model else str(model_path),
            "status": model.status if model else "trained",
            "created_at": _dt(model.created_at if model else None),
        },
        "auto_trader_use_trained_model": settings.auto_trader_use_trained_model,
    }


@router.get("/data-events")
def data_events(
    session: Session = Depends(get_session),
    limit: int = 50,
    symbol: str | None = None,
    source_name: str | None = None,
    data_type: str | None = None,
) -> list[dict[str, Any]]:
    query = select(ExternalDataEvent)
    if symbol:
        query = query.where(ExternalDataEvent.symbol == symbol.upper())
    if source_name:
        query = query.where(ExternalDataEvent.source_name == source_name)
    if data_type:
        query = query.where(ExternalDataEvent.data_type == data_type)
    rows = session.scalars(query.order_by(desc(ExternalDataEvent.event_time)).limit(min(max(limit, 1), 200))).all()
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


@router.get("/features/latest")
def latest_feature(
    session: Session = Depends(get_session),
    symbol: str = "BTCUSDT",
    refresh_if_missing: bool = True,
) -> dict[str, Any]:
    normalized_symbol = symbol.upper()
    feature = session.scalar(
        select(Feature)
        .where(Feature.symbol == normalized_symbol)
        .order_by(desc(Feature.as_of))
        .limit(1)
    )
    values = (feature.payload or {}).get("values", {}) if feature else {}
    persisted = feature is not None
    if refresh_if_missing and (feature is None or any(key not in values for key in INSPECTOR_FEATURE_KEYS)):
        feature = FeatureBuilder(session).build_for_symbol(normalized_symbol, store=False)
        values = (feature.payload or {}).get("values", {})
        persisted = False
    if feature is None:
        return {
            "symbol": normalized_symbol,
            "status": "missing",
            "message": "No feature rows found for this symbol.",
            "vector": {key: 0.0 for key in INSPECTOR_FEATURE_KEYS},
            "final_ai_input": {},
            "news_context": [],
        }

    payload = feature.payload or {}
    metadata = payload.get("metadata", {})
    vector = {key: values.get(key, 0.0) for key in INSPECTOR_FEATURE_KEYS}
    strategy_columns = [
        "price_change",
        "sentiment_score",
        "risk_score",
        "volatility",
        "trader_crowd_score",
        "crowd_risk_score",
        "taker_buy_pressure",
    ]
    strategy_values = values_from_feature(feature, strategy_columns)
    final_ai_input = values.get("final_ai_input") or {
        "schema_version": feature.schema_version or payload.get("schema_version") or CURRENT_FEATURE_SCHEMA_VERSION,
        "symbol": normalized_symbol,
        "timeframe": metadata.get("interval"),
        "feature_columns": columns_for_schema(feature.schema_version),
        "vector": vector,
        "strategy_input": {key: strategy_values.get(key, 0.0) for key in strategy_columns} | {
            "trend": strategy_values.get("trend")
        },
    }
    return {
        "id": feature.id,
        "persisted": persisted,
        "status": "ok",
        "symbol": feature.symbol,
        "schema_version": feature.schema_version,
        "as_of": _dt(feature.as_of),
        "timeframe": metadata.get("interval"),
        "vector": vector,
        "sentiment_score": vector["sentiment_score"],
        "sentiment_confidence": vector["sentiment_confidence"],
        "risk_score": vector["risk_score"],
        "impact_score": vector["impact_score"],
        "recency_weight": vector["recency_weight"],
        "btc_related": vector["btc_related"],
        "eth_related": vector["eth_related"],
        "macro_related": vector["macro_related"],
        "candle_return_1m": vector["candle_return_1m"],
        "candle_return_5m": vector["candle_return_5m"],
        "volatility": vector["volatility"],
        "volume_change": vector["volume_change"],
        "trend_score": vector["trend_score"],
        "final_ai_input": final_ai_input,
        "news_context": metadata.get("news_context", []),
        "derivatives_context": metadata.get("derivatives_context", []),
        "raw_payload": payload,
    }


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


@router.get("/ai-decisions")
def ai_decisions(session: Session = Depends(get_session), limit: int = 50) -> list[dict[str, Any]]:
    rows = session.scalars(select(AiDecision).order_by(desc(AiDecision.created_at)).limit(limit)).all()
    feature_ids = [row.feature_id for row in rows if row.feature_id]
    features = {}
    if feature_ids:
        features = {feature.id: feature for feature in session.scalars(select(Feature).where(Feature.id.in_(feature_ids))).all()}
    return [
        {
            "id": row.id,
            "time": _dt(row.created_at),
            "symbol": row.symbol,
            "action": row.action,
            "confidence": row.confidence,
            "decision_source": _decision_source(row),
            "model_version_id": row.model_version_id,
            "execution_status": row.execution_status,
            "sentiment_score": features.get(row.feature_id).sentiment_score if row.feature_id in features else None,
            "risk_score": features.get(row.feature_id).risk_score if row.feature_id in features else None,
            "strategy": row.strategy_name,
            "reward": row.reward,
            "status": row.execution_status,
            "reason": row.reason or row.execution_message,
        }
        for row in rows
    ]


@router.get("/sentiment/latest")
def sentiment_latest(session: Session = Depends(get_session), limit: int = 50) -> list[dict[str, Any]]:
    rows = session.execute(
        select(NewsSentiment, NewsArticle)
        .join(NewsArticle, NewsArticle.id == NewsSentiment.article_id)
        .order_by(desc(NewsSentiment.created_at))
        .limit(limit)
    ).all()
    return [
        {
            "id": sentiment.id,
            "article_id": article.id,
            "time": _dt(sentiment.created_at),
            "published_at": _dt(article.published_at),
            "provider": article.source_name,
            "source": article.source,
            "title": article.title,
            "url": article.url,
            "sentiment_score": sentiment.sentiment_score,
            "risk_score": sentiment.risk_score,
            "label": sentiment.sentiment_label,
            "confidence": sentiment.confidence,
            "model_name": sentiment.model_name,
            "affected_symbols": sentiment.affected_symbols or [],
            "topics": sentiment.topics or [],
        }
        for sentiment, article in rows
    ]


@router.post("/sentiment/reprocess")
def sentiment_reprocess(
    payload: dict[str, Any] | None = Body(default=None),
    session: Session = Depends(get_session),
    _: None = Depends(require_admin),
) -> dict[str, Any]:
    payload = payload or {}
    limit = min(max(int(payload.get("limit") or 100), 1), 1000)
    reset_model = bool(payload.get("reset_model", True))
    if reset_model:
        reset_sentiment_model_cache()
    articles = list(session.scalars(select(NewsArticle).order_by(desc(NewsArticle.published_at)).limit(limit)))
    processed = 0
    for article in articles:
        text = f"{article.title or ''}\n{article.raw_text or ''}".strip()
        result = analyze_news(text)
        sentiment = session.scalar(select(NewsSentiment).where(NewsSentiment.article_id == article.id).limit(1))
        if sentiment is None:
            sentiment = NewsSentiment(article_id=article.id)
            session.add(sentiment)
        sentiment.sentiment_score = result.sentiment_score
        sentiment.risk_score = result.risk_score
        sentiment.topics = result.topics
        sentiment.affected_symbols = result.affected_symbols
        sentiment.model_name = result.model_name
        sentiment.sentiment_label = result.label
        sentiment.confidence = result.confidence
        sentiment.source_name = f"{article.source_name or 'unknown'}:{result.model_name}"
        sentiment.raw_payload = result.raw_payload
        processed += 1
    session.commit()
    return {
        "processed": processed,
        "active_model": active_sentiment_model(),
    }


@router.get("/trades")
def trades(session: Session = Depends(get_session), limit: int = 50) -> list[dict[str, Any]]:
    rows = session.scalars(select(PaperTrade).order_by(desc(PaperTrade.created_at)).limit(limit)).all()
    return [
        {
            "id": row.id,
            "symbol": row.symbol,
            "action": row.action,
            "side": row.side,
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
            "notional": row.quantity * row.current_price,
            "entry_notional": row.notional,
            "margin_used": row.margin_used,
            "leverage": row.leverage,
            "unrealized_pnl": row.unrealized_pnl,
            "unrealized_pnl_pct": (row.unrealized_pnl / (row.quantity * row.entry_price)) if row.quantity and row.entry_price else 0.0,
            "realized_pnl": row.realized_pnl,
            "stop_loss": row.stop_loss,
            "take_profit": row.take_profit,
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


@router.get("/models/predict")
def model_predict(
    session: Session = Depends(get_session),
    symbol: str = "BTCUSDT",
) -> dict[str, Any]:
    feature = FeatureBuilder(session).build_for_symbol(symbol.upper(), store=False)
    model_decision = PriceModelStrategy().decide(session, feature)
    if model_decision is None:
        return {
            "status": "missing",
            "symbol": symbol.upper(),
            "message": "No trained model is available for prediction.",
        }
    return {
        "status": "ok",
        "symbol": symbol.upper(),
        "decision": model_decision.decision.model_dump(),
        "prediction": model_decision.prediction,
        "model": {
            "id": model_decision.model.id,
            "model_id": model_decision.model.model_id,
            "name": model_decision.model.name,
            "version": model_decision.model.version,
            "feature_schema_version": model_decision.model.feature_schema_version,
            "metrics": model_decision.model.metrics,
        },
    }
