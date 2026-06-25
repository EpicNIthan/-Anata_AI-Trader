from __future__ import annotations

import json
import logging
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.ai.experience_buffer import record_experience
from app.ai.model_strategy import PriceModelStrategy
from app.ai.news_sentiment import active_sentiment_model, analyze_news, reset_sentiment_model_cache
from app.ai.strategy import RuleBasedStrategy
from app.api.webhooks import SignalRequest
from app.config import settings
from app.collectors.derivatives_collector import BinanceDerivativesCollector
from app.collectors.external_market_collectors import ExternalMarketCollectorManager, LiquidationCollector
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
from app.db.session import create_db_and_tables, get_session
from app.features.feature_builder import FeatureBuilder
from app.features.schema import CURRENT_FEATURE_SCHEMA_VERSION, columns_for_schema, values_from_feature
from app.security import require_admin
from app.services.collector_status import latest_candles, latest_news, market_snapshot, news_snapshot
from app.services.db_diagnostics import database_diagnostics
from app.services.training_service import SERVER_TRAINING_DISABLED_MESSAGE, train_model_job
from app.training.dataset_accelerator import build_accelerated_dataset
from app.training.export_dataset import export_dataset, parse_since_date
from app.training.label_builder import build_labels_for_existing_features, label_status
from app.trading.paper_engine import PaperEngine

router = APIRouter(prefix="/api", tags=["lab"], dependencies=[Depends(require_admin)])
logger = logging.getLogger(__name__)

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
    "fear_greed_value",
    "fear_greed_change_1d",
    "fear_greed_change_24h",
    "fear_greed_classification",
    "total_market_cap_usd",
    "market_cap_change_24h",
    "global_market_cap_change_24h",
    "total_volume_usd",
    "total_volume_change_24h",
    "btc_dominance",
    "btc_dominance_change",
    "btc_dominance_change_24h",
    "eth_dominance",
    "liquidation_long_usd_1m",
    "liquidation_short_usd_1m",
    "liquidation_long_usd_5m",
    "liquidation_short_usd_5m",
    "liquidation_total_usd_5m",
    "liquidation_imbalance_5m",
    "liquidation_spike_score",
    "usdt_deviation",
    "usdc_deviation",
    "usdt_price_deviation",
    "usdc_price_deviation",
    "stablecoin_depeg_risk",
    "stablecoin_supply_change_1d",
    "stablecoin_supply_change_24h",
    "macro_risk_score",
    "regulation_risk_score",
    "fed_risk_score",
    "war_risk_score",
    "exchange_hack_risk_score",
    "etf_positive_score",
    "security_risk_score",
    "etf_bullish_score",
    "world_risk_score",
    "market_regime_score",
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


def _training_service(request: Request):
    service = getattr(request.app.state, "training_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Training service is not initialized")
    return service


def _dataset_id_from_path(path: Path) -> str:
    return path.name


def _dataset_download_path(dataset_id: str) -> Path:
    safe_name = Path(dataset_id).name
    if safe_name != dataset_id or not safe_name:
        raise HTTPException(status_code=400, detail="Invalid dataset id")
    path = Path("datasets") / safe_name
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Dataset not found")
    return path


def _model_payload(model: ModelVersion | None) -> dict[str, Any]:
    if model is None:
        return {"name": "none", "version": "untrained", "status": "missing"}
    payload = model.raw_payload or {}
    return {
        "id": model.id,
        "model_id": model.model_id,
        "name": model.name,
        "version": model.version,
        "feature_schema_version": model.feature_schema_version,
        "feature_columns": model.feature_columns or payload.get("feature_columns"),
        "path": model.path,
        "status": model.status,
        "metrics": model.metrics or payload.get("metrics"),
        "model_type": payload.get("model_type"),
        "training_dataset_hash": payload.get("training_dataset_hash") or payload.get("dataset_hash"),
        "created_at": _dt(model.created_at),
        "raw_payload": payload,
    }


def _find_model_metadata(zip_file: zipfile.ZipFile) -> dict[str, Any]:
    exact_metadata_names = [
        name
        for name in zip_file.namelist()
        if Path(name).name.lower() in {"metadata.json", "model_metadata.json"}
    ]
    metadata_names = exact_metadata_names + [
        name
        for name in zip_file.namelist()
        if name not in exact_metadata_names and Path(name).suffix.lower() == ".json"
    ]
    if not metadata_names:
        raise HTTPException(status_code=400, detail="Model package must include metadata JSON")
    for name in metadata_names:
        try:
            data = json.loads(zip_file.read(name).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            data["_metadata_file"] = name
            return data
    raise HTTPException(status_code=400, detail="Could not parse model metadata JSON")


def _safe_zip_member_path(name: str) -> Path:
    member = Path(name)
    if member.is_absolute() or ".." in member.parts:
        raise HTTPException(status_code=400, detail=f"Unsafe model package path: {name}")
    return member


def _save_uploaded_model_package(upload: UploadFile) -> dict[str, Any]:
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    package_dir = settings.model_dir / "packages"
    package_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_filename = Path(upload.filename or f"model_package_{timestamp}.zip").name
    if not safe_filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Model package must be a .zip file")
    package_path = package_dir / f"{timestamp}_{safe_filename}"
    with package_path.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)

    try:
        with zipfile.ZipFile(package_path) as archive:
            metadata = _find_model_metadata(archive)
            model_file_hint = metadata.get("model_file") or metadata.get("model_path")
            model_names = [
                name
                for name in archive.namelist()
                if Path(name).suffix.lower() in {".joblib", ".pkl", ".json"} and name != metadata.get("_metadata_file")
            ]
            if model_file_hint:
                hinted = str(model_file_hint)
                if hinted in archive.namelist():
                    model_names.insert(0, hinted)
            model_member = next((name for name in model_names if Path(name).suffix.lower() in {".joblib", ".pkl"}), None)
            model_member = model_member or next((name for name in model_names if Path(name).suffix.lower() == ".json"), None)
            if not model_member:
                raise HTTPException(status_code=400, detail="Model package must include a .joblib, .pkl, or .json model file")
            member_path = _safe_zip_member_path(model_member)
            version = str(metadata.get("version") or timestamp)
            extract_dir = settings.model_dir / "uploaded" / version
            extract_dir.mkdir(parents=True, exist_ok=True)
            output_path = extract_dir / member_path.name
            with archive.open(model_member) as source, output_path.open("wb") as target:
                shutil.copyfileobj(source, target)
    except zipfile.BadZipFile as exc:
        package_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Invalid model package zip") from exc

    metadata["package_path"] = str(package_path)
    metadata["model_file"] = str(output_path)
    metadata.setdefault("version", timestamp)
    metadata.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    return metadata


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


def _external_snapshot(session: Session, worker_state: dict[str, Any] | None = None) -> dict[str, Any]:
    worker_state = worker_state or {}
    status = ExternalMarketCollectorManager().status(session)
    liquidation_status = LiquidationCollector().status(session)
    return {
        **status,
        "liquidations": liquidation_status
        | {
            "running": bool(worker_state.get("liquidations_running")),
        },
        "worker": worker_state,
    }


def _approx_count(session: Session, model: Any) -> int:
    return int(session.scalar(select(func.max(model.id))) or 0)


def _aware_dt(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _fast_market_snapshot(session: Session, collector_state: dict[str, Any] | None = None) -> dict[str, Any]:
    latest_by_symbol: list[dict[str, Any]] = []
    latest_time: datetime | None = None
    for symbol in settings.binance_symbols:
        normalized = symbol.upper()
        candle = session.scalar(
            select(Candle)
            .where(Candle.symbol == normalized, Candle.is_closed.is_(True))
            .order_by(desc(Candle.open_time))
            .limit(1)
        )
        live_update = session.scalar(
            select(LiveCandleUpdate)
            .where(LiveCandleUpdate.symbol == normalized)
            .order_by(desc(LiveCandleUpdate.open_time))
            .limit(1)
        )
        display_price = candle.close if candle else None
        display_time = candle.open_time if candle else None
        display_source = candle.source_name if candle else None
        display_closed = candle.is_closed if candle else None
        if live_update and (display_time is None or live_update.open_time >= display_time):
            display_price = live_update.close
            display_time = live_update.open_time
            display_source = live_update.source_name
            display_closed = False
        candle_time = _aware_dt(candle.open_time if candle else None)
        if candle_time:
            latest_time = max(latest_time, candle_time) if latest_time else candle_time
        latest_by_symbol.append(
            {
                "symbol": normalized,
                "price": display_price,
                "open_time": _dt(display_time),
                "is_closed": display_closed,
                "source_name": display_source,
            }
        )
    age_seconds = None
    if latest_time:
        age_seconds = (datetime.now(timezone.utc) - latest_time).total_seconds()
    return {
        "collector": collector_state or {},
        "candle_count": _approx_count(session, Candle),
        "latest_candle_time": _dt(latest_time),
        "latest_by_symbol": latest_by_symbol,
        "latest_prices": {row["symbol"]: row["price"] for row in latest_by_symbol},
        "stale": latest_time is None or (age_seconds is not None and age_seconds > 300),
        "age_seconds": age_seconds,
        "store_live_candle_updates": settings.store_live_candle_updates,
        "closed_candles_only": True,
    }


def _fast_news_snapshot(session: Session, collector_state: dict[str, Any] | None = None) -> dict[str, Any]:
    latest = session.scalar(select(NewsArticle).order_by(desc(NewsArticle.published_at)).limit(1))
    age_seconds = None
    if latest and latest.published_at:
        published = _aware_dt(latest.published_at)
        age_seconds = (datetime.now(timezone.utc) - published).total_seconds()
    return {
        "collector": collector_state or {},
        "providers": ((collector_state or {}).get("details") or {}).get("providers") or {},
        "news_count": _approx_count(session, NewsArticle),
        "sentiment_count": _approx_count(session, NewsSentiment),
        "experience_count": _approx_count(session, ExperienceRecord),
        "latest_news_time": _dt(latest.published_at if latest else None),
        "latest_title": latest.title if latest else None,
        "latest_source": latest.source if latest else None,
        "stale": latest is None or (age_seconds is not None and age_seconds > 7200),
        "age_seconds": age_seconds,
        "news_api_key_configured": bool(settings.news_api_key),
        "warning": (collector_state or {}).get("warning") or (collector_state or {}).get("last_error"),
        "mock_fallback_enabled": settings.news_mock_fallback_enabled,
    }


def _fast_data_coverage(session: Session) -> dict[str, Any]:
    return {
        "candles": _approx_count(session, Candle),
        "news": _approx_count(session, NewsArticle),
        "sentiment": _approx_count(session, NewsSentiment),
        "derivatives": 0,
        "liquidations": 0,
        "fear_greed": 0,
        "global_market": 0,
        "stablecoin_risk": 0,
        "macro_risk": 0,
        "labeled_rows": 0,
        "label_coverage_pct": 0.0,
        "feature_schema_version": CURRENT_FEATURE_SCHEMA_VERSION,
    }


def _data_coverage(session: Session) -> dict[str, Any]:
    label_info = label_status(session)
    external_counts = {
        source_name: int(count)
        for source_name, count in session.execute(
            select(ExternalDataEvent.source_name, func.count(ExternalDataEvent.id)).group_by(ExternalDataEvent.source_name)
        ).all()
    }
    return {
        "candles": session.scalar(select(func.count(Candle.id))) or 0,
        "news": session.scalar(select(func.count(NewsArticle.id))) or 0,
        "sentiment": session.scalar(select(func.count(NewsSentiment.id))) or 0,
        "derivatives": sum(count for source, count in external_counts.items() if source.startswith("binance_futures_") and source != "binance_futures_liquidations"),
        "liquidations": external_counts.get("binance_futures_liquidations", 0),
        "fear_greed": external_counts.get("alternative_me_fear_greed", 0),
        "global_market": external_counts.get("coingecko_global_market", 0),
        "stablecoin_risk": external_counts.get("defillama_stablecoin_risk", 0),
        "macro_risk": external_counts.get("macro_risk_news", 0),
        "labeled_rows": label_info.get("rows_with_target_trade_quality_score", 0),
        "label_coverage_pct": label_info.get("label_coverage_pct", 0.0),
        "label_warning": label_info.get("warning"),
        "feature_schema_version": CURRENT_FEATURE_SCHEMA_VERSION,
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
    decisions = list(session.scalars(select(AiDecision).order_by(desc(AiDecision.created_at)).limit(100)))
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
def dashboard_summary(
    request: Request,
    session: Session = Depends(get_session),
    fast: bool = True,
) -> dict[str, Any]:
    account = PaperEngine(session).snapshot(record=False)
    collectors = _worker_manager(request).snapshot()
    active_model = session.scalar(select(ModelVersion).where(ModelVersion.status == "active").order_by(desc(ModelVersion.created_at)).limit(1))
    latest_model = active_model or session.scalar(select(ModelVersion).order_by(desc(ModelVersion.created_at)).limit(1))
    latest_training_run = session.scalar(select(TrainingRun).order_by(desc(TrainingRun.started_at)).limit(1))
    latest_decision = session.scalar(select(AiDecision).order_by(desc(AiDecision.created_at)).limit(1))
    counts = {
        "candles": _approx_count(session, Candle),
        "news": _approx_count(session, NewsArticle),
        "sentiment": _approx_count(session, NewsSentiment),
        "features": _approx_count(session, Feature),
        "training_features": _approx_count(session, TrainingFeature),
        "experiences": _approx_count(session, ExperienceRecord),
        "external_data_events": _approx_count(session, ExternalDataEvent),
        "trades": _approx_count(session, PaperTrade),
        "open_positions": session.scalar(select(func.count(Position.id)).where(Position.status == "OPEN")) or 0,
    }
    if not fast:
        counts = {
            "candles": session.scalar(select(func.count(Candle.id))) or 0,
            "news": session.scalar(select(func.count(NewsArticle.id))) or 0,
            "sentiment": session.scalar(select(func.count(NewsSentiment.id))) or 0,
            "features": session.scalar(select(func.count(Feature.id))) or 0,
            "training_features": session.scalar(select(func.count(TrainingFeature.id))) or 0,
            "experiences": session.scalar(select(func.count(ExperienceRecord.id))) or 0,
            "external_data_events": session.scalar(select(func.count(ExternalDataEvent.id))) or 0,
            "trades": session.scalar(select(func.count(PaperTrade.id))) or 0,
            "open_positions": session.scalar(select(func.count(Position.id)).where(Position.status == "OPEN")) or 0,
        }
    return {
        "account": account,
        "mode": "paper",
        "paper_start_balance": settings.paper_start_balance,
        "market": _fast_market_snapshot(session, collectors.get("market", {})) if fast else market_snapshot(session, collectors.get("market", {})),
        "news": _fast_news_snapshot(session, collectors.get("news", {})) if fast else news_snapshot(session, collectors.get("news", {})),
        "derivatives": {} if fast else _derivatives_snapshot(session, collectors.get("derivatives", {})),
        "external": {} if fast else _external_snapshot(
            session,
            {
                "external": collectors.get("external", {}),
                "liquidations": collectors.get("liquidations", {}),
                "liquidations_running": bool(collectors.get("liquidations", {}).get("running")),
            },
        ),
        "collectors": collectors,
        "auto_trader": _auto_trader(request).status(),
        "trading": _trading_diagnostics(session),
        "coverage": _fast_data_coverage(session) if fast else _data_coverage(session),
        "sentiment_model": active_sentiment_model(),
        "model": _model_payload(latest_model)
        | {
            "server_training_enabled": settings.enable_server_training,
            "server_inference_enabled": settings.enable_server_inference,
            "training_disabled_message": SERVER_TRAINING_DISABLED_MESSAGE if not settings.enable_server_training else None,
            "candidate_count": session.scalar(select(func.count(ModelVersion.id)).where(ModelVersion.status == "candidate")) or 0,
        },
        "training": {
            "last_run_at": _dt(latest_training_run.started_at if latest_training_run else None),
            "status": latest_training_run.status if latest_training_run else "none",
            "feature_schema_version": latest_training_run.feature_schema_version if latest_training_run else CURRENT_FEATURE_SCHEMA_VERSION,
            "worker": _training_service(request).status(),
        },
        "counts": counts,
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


@router.get("/external/status")
def external_status(request: Request, session: Session = Depends(get_session)) -> dict[str, Any]:
    states = _worker_manager(request).snapshot()
    return _external_snapshot(
        session,
        {
            "external": states.get("external", {}),
            "liquidations": states.get("liquidations", {}),
            "liquidations_running": bool(states.get("liquidations", {}).get("running")),
        },
    )


@router.get("/external/latest")
def external_latest(
    session: Session = Depends(get_session),
    source_name: str | None = None,
    data_type: str | None = None,
    symbol: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    query = select(ExternalDataEvent)
    if source_name:
        query = query.where(ExternalDataEvent.source_name == source_name)
    if data_type:
        query = query.where(ExternalDataEvent.data_type == data_type)
    if symbol:
        query = query.where(ExternalDataEvent.symbol == symbol.upper())
    rows = session.scalars(query.order_by(desc(ExternalDataEvent.event_time)).limit(min(max(limit, 1), 300))).all()
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


@router.post("/external/run-once")
async def external_run_once(
    payload: dict[str, Any] | None = Body(default=None),
    _: None = Depends(require_admin),
) -> dict[str, Any]:
    payload = payload or {}
    collector_name = payload.get("collector") or payload.get("name")
    mock = bool(payload.get("mock", False))
    if collector_name and str(collector_name).lower() in {"liquidations", "liquidation"}:
        liquidation_result = await LiquidationCollector().fetch_once(mock=mock)
        return {
            "rows_saved": liquidation_result.rows_saved,
            "collectors": {
                "liquidations": liquidation_result.__dict__,
            },
        }
    result = await ExternalMarketCollectorManager().fetch_once(
        collector_name=str(collector_name).lower() if collector_name else None,
        mock=mock,
    )
    return result


@router.post("/liquidations/run-once")
async def liquidations_run_once(
    payload: dict[str, Any] | None = Body(default=None),
    _: None = Depends(require_admin),
) -> dict[str, Any]:
    payload = payload or {}
    symbols = payload.get("symbols")
    if isinstance(symbols, str):
        symbols = [item.strip().upper() for item in symbols.split(",") if item.strip()]
    if symbols is not None and not isinstance(symbols, list):
        raise HTTPException(status_code=400, detail="symbols must be a list or comma-separated string")
    collector = LiquidationCollector(symbols=symbols)
    result = await collector.fetch_once(mock=bool(payload.get("mock", False)))
    return result.__dict__


@router.get("/db/diagnostics")
def db_diagnostics(session: Session = Depends(get_session), include_raw_estimates: bool = False) -> dict[str, Any]:
    return database_diagnostics(session, include_raw_estimates=include_raw_estimates)


@router.get("/db/storage")
def db_storage(session: Session = Depends(get_session)) -> dict[str, Any]:
    return database_diagnostics(session, include_raw_estimates=True)


@router.get("/storage/status")
def storage_status(session: Session = Depends(get_session)) -> dict[str, Any]:
    return database_diagnostics(session, include_raw_estimates=True)


@router.post("/db/migrate")
def db_migrate(_: None = Depends(require_admin)) -> dict[str, Any]:
    try:
        migration = create_db_and_tables()
    except Exception as exc:
        logger.exception("Database migration failed")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Database migration failed. Check Railway logs for the full traceback.",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        ) from exc
    return {
        "status": "ok",
        "migration": migration,
        "added_columns": migration.get("added_columns", []) if isinstance(migration, dict) else [],
        "existing_tables": migration.get("existing_tables", []) if isinstance(migration, dict) else [],
    }


@router.post("/storage/cleanup/run")
def storage_cleanup_run(request: Request, _: None = Depends(require_admin)) -> dict[str, Any]:
    return _data_lifecycle(request).compact_once()


@router.post("/storage/compact")
def storage_compact(request: Request, _: None = Depends(require_admin)) -> dict[str, Any]:
    return _data_lifecycle(request).compact_once()


@router.get("/db/lifecycle/status")
def db_lifecycle_status(request: Request) -> dict[str, Any]:
    return _data_lifecycle(request).status()


@router.post("/db/cleanup")
def db_cleanup(request: Request, _: None = Depends(require_admin)) -> dict[str, Any]:
    return _data_lifecycle(request).run_cleanup_once()


@router.post("/db/compact")
def db_compact(
    request: Request,
    payload: dict[str, Any] | None = Body(default=None),
    _: None = Depends(require_admin),
) -> dict[str, Any]:
    payload = payload or {}
    tables = payload.get("archive_tables") or payload.get("tables")
    if isinstance(tables, str):
        tables = [item.strip() for item in tables.split(",") if item.strip()]
    if tables is not None and not isinstance(tables, list):
        raise HTTPException(status_code=400, detail="archive_tables must be a list or comma-separated string")
    return _data_lifecycle(request).compact_once(
        archive_before_delete=bool(payload.get("archive_before_delete", False)),
        archive_tables=tables,
    )


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
    if name not in {"market", "news", "derivatives", "external", "liquidations"}:
        raise HTTPException(status_code=404, detail="Unknown collector")
    return await _worker_manager(request).start(name)


@router.post("/collectors/{name}/stop")
async def stop_collector(name: str, request: Request, _: None = Depends(require_admin)) -> dict[str, Any]:
    if name not in {"market", "news", "derivatives", "external", "liquidations"}:
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


@router.get("/training/label-status")
def training_label_status(session: Session = Depends(get_session)) -> dict[str, Any]:
    return label_status(session)


@router.post("/training/build-labels")
def training_build_labels(
    payload: dict[str, Any] | None = Body(default=None),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    payload = payload or {}
    try:
        symbols = payload.get("symbols")
        if isinstance(symbols, str):
            symbols = [item.strip().upper() for item in symbols.split(",") if item.strip()]
        if symbols is not None and not isinstance(symbols, list):
            raise HTTPException(status_code=400, detail="symbols must be a list or comma-separated string")
        schema_version = str(payload.get("schema_version") or payload.get("feature_schema_version") or CURRENT_FEATURE_SCHEMA_VERSION)
        limit = int(payload["limit"]) if payload.get("limit") else None
        result = build_labels_for_existing_features(
            session,
            symbols=symbols,
            interval=str(payload.get("interval")) if payload.get("interval") else None,
            schema_version=schema_version,
            force=bool(payload.get("force", False)),
            limit=limit,
            sync_features=bool(payload.get("sync_features", True)),
        )
        return {
            "status": "ok",
            **result,
            "label_status": label_status(session, schema_version=schema_version),
        }
    except HTTPException:
        raise
    except Exception as exc:
        session.rollback()
        logger.exception("Training label build failed")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Training label build failed. Run POST /api/db/migrate, then retry with a small limit like 500.",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "accepted_payload": ["symbols", "limit", "force", "interval", "schema_version", "feature_schema_version"],
            },
        ) from exc


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
    output_path = Path("datasets") / f"anata_dataset_{timestamp}.csv.gz"
    exported_path = export_dataset(
        output_path,
        feature_schema_version=feature_schema_version,
        since_date=since_date,
        use_all_data=use_all_data,
        auto_build_labels=payload.get("auto_build_labels"),
    )
    parquet_path: str | None = None
    try:
        import importlib.util

        if importlib.util.find_spec("pyarrow"):
            import pandas as pd

            parquet_output = exported_path.with_suffix("").with_suffix(".parquet")
            pd.read_csv(exported_path).to_parquet(parquet_output, index=False)
            parquet_path = str(parquet_output)
    except Exception:
        parquet_path = None
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
    label_summary = label_status(session, schema_version=feature_schema_version)
    return {
        "dataset_id": _dataset_id_from_path(exported_path),
        "exported_path": str(exported_path),
        "download_url": f"/api/training/download/{_dataset_id_from_path(exported_path)}",
        "parquet_path": parquet_path,
        "feature_schema_version": feature_schema_version,
        "since_date": since_date.isoformat() if since_date else None,
        "use_all_data": use_all_data,
        "counts": counts,
        "news_by_provider": news_by_provider,
        "features_by_symbol": features_by_symbol,
        "label_summary": label_summary,
        "warning": label_summary.get("warning"),
        "latest_feature_at": _dt(latest_feature.as_of if latest_feature else None),
    }


@router.get("/training/download/{dataset_id}")
def training_download(dataset_id: str) -> FileResponse:
    path = _dataset_download_path(dataset_id)
    return FileResponse(path, filename=path.name, media_type="application/gzip" if path.suffix == ".gz" else "text/csv")


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
    result = await build_accelerated_dataset(
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
    if result.get("exported_path"):
        dataset_path = Path(str(result["exported_path"]))
        result["dataset_id"] = _dataset_id_from_path(dataset_path)
        result["download_url"] = f"/api/training/download/{result['dataset_id']}"
    return result


@router.post("/training/train-model")
async def training_train_model(
    request: Request,
    payload: dict[str, Any] | None = Body(default=None),
    _: None = Depends(require_admin),
) -> dict[str, Any]:
    payload = payload or {}
    if not settings.enable_server_training:
        return {
            "status": "disabled",
            "message": SERVER_TRAINING_DISABLED_MESSAGE,
            "allowed_server_actions": ["build_dataset", "export_dataset", "upload_model", "activate_model"],
        }
    symbols = payload.get("symbols")
    if isinstance(symbols, str):
        payload["symbols"] = [item.strip().upper() for item in symbols.split(",") if item.strip()]
    elif symbols is not None and not isinstance(symbols, list):
        raise HTTPException(status_code=400, detail="symbols must be a list or comma-separated string")
    if bool(payload.get("wait", False)):
        return await train_model_job(payload)
    status = await _training_service(request).start(payload)
    return {"status": "started", "training": status}


@router.get("/training/status")
def training_status(request: Request) -> dict[str, Any]:
    return _training_service(request).status()


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
        "external_context": metadata.get("external_context", []),
        "source_freshness": metadata.get("source_freshness", {}),
        "stale_sources": metadata.get("stale_sources", []),
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


@router.get("/sentiment/model-status")
def sentiment_model_status() -> dict[str, Any]:
    return active_sentiment_model()


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
    active_model = session.scalar(select(ModelVersion).where(ModelVersion.status == "active").order_by(desc(ModelVersion.created_at)).limit(1))
    model = active_model or session.scalar(select(ModelVersion).order_by(desc(ModelVersion.created_at)).limit(1))
    return _model_payload(model)


@router.get("/models")
def list_models(session: Session = Depends(get_session), limit: int = 50) -> list[dict[str, Any]]:
    rows = session.scalars(select(ModelVersion).order_by(desc(ModelVersion.created_at)).limit(min(max(limit, 1), 200))).all()
    return [_model_payload(row) for row in rows]


@router.get("/models/active")
def active_model(session: Session = Depends(get_session)) -> dict[str, Any]:
    model = session.scalar(select(ModelVersion).where(ModelVersion.status == "active").order_by(desc(ModelVersion.created_at)).limit(1))
    return _model_payload(model)


@router.post("/models/upload")
def upload_model(
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    metadata = _save_uploaded_model_package(file)
    version = str(metadata.get("version") or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    model_id = str(metadata.get("model_id") or f"{metadata.get('model_type') or 'uploaded-model'}:{version}")
    name = str(metadata.get("name") or metadata.get("model_type") or "uploaded-model")
    feature_columns = metadata.get("feature_columns") or metadata.get("features") or columns_for_schema(metadata.get("feature_schema_version"))
    if not isinstance(feature_columns, list):
        raise HTTPException(status_code=400, detail="Model metadata feature_columns must be a list")
    model = ModelVersion(
        model_id=model_id,
        name=name,
        version=version,
        feature_schema_version=str(metadata.get("feature_schema_version") or CURRENT_FEATURE_SCHEMA_VERSION),
        feature_columns=[str(column) for column in feature_columns],
        path=str(metadata.get("package_path") or metadata.get("model_file")),
        parent_model_id=metadata.get("parent_model_id"),
        checkpoint_path=metadata.get("checkpoint_path") or metadata.get("from_checkpoint"),
        status="candidate",
        metrics=metadata.get("metrics") if isinstance(metadata.get("metrics"), dict) else {},
        raw_payload=metadata,
    )
    session.add(model)
    session.commit()
    session.refresh(model)
    return {
        "status": "candidate",
        "message": "Model uploaded and registered as candidate. Activate it explicitly after checking metrics.",
        "model": _model_payload(model),
    }


@router.post("/models/activate")
def activate_model(
    payload: dict[str, Any] | None = Body(default=None),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    payload = payload or {}
    model: ModelVersion | None = None
    if payload.get("id") is not None:
        model = session.get(ModelVersion, int(payload["id"]))
    elif payload.get("model_id"):
        model = session.scalar(select(ModelVersion).where(ModelVersion.model_id == str(payload["model_id"])).order_by(desc(ModelVersion.created_at)).limit(1))
    elif payload.get("version"):
        model = session.scalar(select(ModelVersion).where(ModelVersion.version == str(payload["version"])).order_by(desc(ModelVersion.created_at)).limit(1))
    else:
        model = session.scalar(select(ModelVersion).where(ModelVersion.status == "candidate").order_by(desc(ModelVersion.created_at)).limit(1))
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    active_rows = session.scalars(select(ModelVersion).where(ModelVersion.status == "active", ModelVersion.id != model.id)).all()
    for row in active_rows:
        row.status = "inactive"
    model.status = "active"
    session.commit()
    session.refresh(model)
    return {"status": "active", "model": _model_payload(model)}


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
            "message": "No active compatible model is available for prediction; rule-based fallback will be used.",
            "server_inference_enabled": settings.enable_server_inference,
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
