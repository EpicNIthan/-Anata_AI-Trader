from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.ai.experience_buffer import record_experience
from app.ai.news_sentiment import active_sentiment_model
from app.ai.strategy import RuleBasedStrategy
from app.api.webhooks import SignalRequest
from app.config import settings
from app.collectors.market_collector import BinanceMarketCollector, format_collector_error
from app.collectors.news_collector import NewsCollector
from app.db.models import (
    AccountEquity,
    AiDecision,
    Candle,
    ExperienceRecord,
    ExternalDataEvent,
    Feature,
    ModelVersion,
    NewsArticle,
    NewsSentiment,
    PaperTrade,
    Position,
    TrainingRun,
)
from app.db.session import get_session
from app.features.feature_builder import FeatureBuilder
from app.features.schema import CURRENT_FEATURE_SCHEMA_VERSION
from app.security import require_admin
from app.services.collector_status import latest_candles, latest_news, market_snapshot, news_snapshot
from app.training.export_dataset import export_dataset, parse_since_date
from app.trading.paper_engine import PaperEngine

router = APIRouter(prefix="/api", tags=["lab"])


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
                "is_closed": bool(row.is_closed),
            }
            continue

        current["close_time"] = row.close_time or current["close_time"]
        current["high"] = max(_float(current["high"]), row.high)
        current["low"] = min(_float(current["low"]), row.low)
        current["close"] = row.close
        current["volume"] = _float(current["volume"]) + _float(row.volume)
        current["is_closed"] = bool(current["is_closed"]) and bool(row.is_closed)

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
        "collectors": collectors,
        "auto_trader": _auto_trader(request).status(),
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
            "experiences": session.scalar(select(func.count(ExperienceRecord.id))) or 0,
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
        candle = session.scalar(
            select(Candle)
            .where(Candle.symbol == symbol.upper())
            .order_by(desc(Candle.open_time))
            .limit(1)
        )
        if candle is None:
            return {}
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
            .where(Candle.symbol == symbol, Candle.interval == timeframe)
            .order_by(desc(Candle.open_time))
            .limit(safe_limit)
        )
    )
    rows.reverse()
    if rows:
        return [_serialize_stored_candle(row) for row in rows]

    base_interval = "1m"
    base_seconds = 60
    requested_base_ratio = max(target_seconds // base_seconds, 1)
    base_limit = min(max(safe_limit * requested_base_ratio + requested_base_ratio, 200), 6000)
    base_rows = list(
        session.scalars(
            select(Candle)
            .where(Candle.symbol == symbol, Candle.interval == base_interval)
            .order_by(desc(Candle.open_time))
            .limit(base_limit)
        )
    )
    base_rows.reverse()
    if not base_rows:
        return []

    if target_seconds < base_seconds:
        fallback_rows = base_rows[-safe_limit:]
        return [
            _serialize_stored_candle(row, requested_timeframe=timeframe, source_name="1m_live_fallback")
            for row in fallback_rows
        ]

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
    if name not in {"market", "news"}:
        raise HTTPException(status_code=404, detail="Unknown collector")
    return await _worker_manager(request).start(name)


@router.post("/collectors/{name}/stop")
async def stop_collector(name: str, request: Request, _: None = Depends(require_admin)) -> dict[str, Any]:
    if name not in {"market", "news"}:
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
        for symbol, count in session.execute(select(Feature.symbol, func.count(Feature.id)).group_by(Feature.symbol)).all()
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
            "notional": row.quantity * row.current_price,
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
