"""Read-only API payloads for the AI Vision dashboard.

The V2 tables are deliberately looked up at runtime.  This keeps the existing
paper-trading installation usable while its additive migration is rolling out:
legacy candles, decisions, trades, news, and positions still render, and a
missing V2 table produces an empty section rather than a failed dashboard.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import desc, func, or_, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import models as db_models
from app.db.session import get_session
from app.pipeline.monitoring import paper_pnl_attribution
from app.security import require_admin

router = APIRouter(prefix="/api/vision", tags=["vision"], dependencies=[Depends(require_admin)])
logger = logging.getLogger(__name__)

_SYMBOL_RE = re.compile(r"^[A-Z0-9_.:-]{3,32}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:@-]{1,128}$")
_MAIN_ACCOUNT_IDS = {"", "main", "default", "legacy", "champion"}
_MAX_CANDLES = 1_000
_MAX_EVENTS = 250
_MAX_ROWS = 200
_DEFAULT_LIMIT = max(int(settings.vision_default_limit), 1)


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _dt(value: Any) -> str | None:
    if not isinstance(value, datetime):
        return None
    value = _as_utc(value)
    return value.isoformat() if value else None


def _epoch(value: Any) -> int | None:
    value = _as_utc(value) if isinstance(value, datetime) else None
    return int(value.timestamp()) if value else None


def _json_safe(value: Any) -> Any:
    """Convert a selected database value to a compact JSON-safe value."""
    if isinstance(value, datetime):
        return _dt(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    enum_value = getattr(value, "value", None)
    if enum_value is not None and not isinstance(value, (str, bytes, int, float, bool)):
        return _json_safe(enum_value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _list(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [item.strip() for item in value.split(",") if item.strip()]
        return parsed if isinstance(parsed, list) else [parsed]
    return []


def _value(row: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(row, dict) and name in row and row[name] is not None:
            return row[name]
        try:
            value = getattr(row, name)
        except (AttributeError, TypeError):
            continue
        if value is not None:
            return value
    return default


def _identifier(row: Any, *names: str) -> str | None:
    value = _value(row, *names)
    return str(value) if value is not None else None


def _model(name: str) -> type[Any] | None:
    """Return a mapped V2 model if this deployment has it."""
    candidate = getattr(db_models, name, None)
    if isinstance(candidate, type) and getattr(candidate, "__table__", None) is not None:
        return candidate
    return None


def _column(model: type[Any] | None, *names: str) -> Any | None:
    if model is None:
        return None
    table = getattr(model, "__table__", None)
    column_names = set(table.columns.keys()) if table is not None else set()
    for name in names:
        if name in column_names:
            return getattr(model, name)
    return None


def _safe_rows(session: Session, statement: Any) -> list[Any]:
    try:
        return list(session.scalars(statement).all())
    except Exception as exc:  # pragma: no cover - depends on rollout DB state
        # PostgreSQL marks a transaction failed after a missing-table query.
        # This endpoint is read-only, so a rollback is safe and lets legacy
        # data continue rendering during a partial migration rollout.
        session.rollback()
        logger.debug("AI Vision optional query unavailable: %s", type(exc).__name__)
        return []


def _safe_execute(session: Session, statement: Any) -> list[Any]:
    try:
        return list(session.execute(statement).all())
    except Exception as exc:  # pragma: no cover - depends on rollout DB state
        session.rollback()
        logger.debug("AI Vision optional query unavailable: %s", type(exc).__name__)
        return []


def _safe_scalar(session: Session, statement: Any) -> Any | None:
    try:
        return session.scalar(statement)
    except Exception as exc:  # pragma: no cover - depends on rollout DB state
        session.rollback()
        logger.debug("AI Vision optional query unavailable: %s", type(exc).__name__)
        return None


def _safe_get(session: Session, model: type[Any], identity: Any) -> Any | None:
    try:
        return session.get(model, identity)
    except Exception as exc:  # pragma: no cover - protects old operational DBs
        session.rollback()
        logger.debug("AI Vision optional query unavailable: %s", type(exc).__name__)
        return None


def _normal_symbol(symbol: str) -> str:
    normalized = (symbol or "").strip().upper()
    if not _SYMBOL_RE.fullmatch(normalized):
        raise HTTPException(status_code=400, detail="symbol must be 3-32 uppercase letters, digits, or ._:-")
    return normalized


def _normal_identifier(value: str | None, *, field: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not _IDENTIFIER_RE.fullmatch(normalized):
        raise HTTPException(status_code=400, detail=f"Invalid {field}")
    return normalized


def _bounded(value: int, *, default: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return min(max(parsed, 1), maximum)


def _window(start: datetime | None, end: datetime | None) -> tuple[datetime | None, datetime | None]:
    normalized_start = _as_utc(start)
    normalized_end = _as_utc(end)
    if normalized_start and normalized_end and normalized_start > normalized_end:
        raise HTTPException(status_code=400, detail="start must be earlier than end")
    return normalized_start, normalized_end


def _time_column(model: type[Any] | None, candidates: Iterable[str] = ()) -> Any | None:
    return _column(
        model,
        *tuple(candidates),
        "generated_at",
        "occurred_at",
        "created_at",
        "updated_at",
        "event_time",
        "filled_at",
        "submitted_at",
        "requested_at",
        "available_to_model_time",
        "active_from",
        "observed_at",
        "evaluated_at",
        "published_time",
        "published_at",
        "timestamp",
        "as_of",
    )


def _query_v2(
    session: Session,
    model_name: str,
    *,
    symbol: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    account_id: str | None = None,
    limit: int = _MAX_ROWS,
    time_candidates: Iterable[str] = (),
    ascending: bool = False,
) -> list[Any]:
    """Query an optional V2 entity with only columns that actually exist."""
    model = _model(model_name)
    if model is None:
        return []
    statement = select(model)
    symbol_column = _column(model, "symbol", "primary_symbol")
    if symbol and symbol_column is not None:
        statement = statement.where(func.upper(symbol_column) == symbol)
    account_column = _column(model, "paper_account_id", "account_id", "paper_sandbox_account_id")
    if account_id and account_column is not None:
        statement = statement.where(account_column == account_id)
    timestamp_column = _time_column(model, time_candidates)
    if timestamp_column is not None:
        if start is not None:
            statement = statement.where(timestamp_column >= start)
        if end is not None:
            statement = statement.where(timestamp_column <= end)
        statement = statement.order_by(timestamp_column.asc() if ascending else desc(timestamp_column))
    else:
        id_column = _column(model, "id")
        if id_column is not None:
            statement = statement.order_by(id_column.asc() if ascending else desc(id_column))
    return _safe_rows(session, statement.limit(_bounded(limit, default=_MAX_ROWS, maximum=_MAX_ROWS)))


def _query_v2_by_value(
    session: Session,
    model_name: str,
    field_names: Iterable[str],
    value: Any,
    *,
    limit: int = _MAX_ROWS,
) -> list[Any]:
    model = _model(model_name)
    column = _column(model, *tuple(field_names))
    if model is None or column is None or value is None:
        return []
    timestamp_column = _time_column(model)
    statement = select(model).where(column == value)
    if timestamp_column is not None:
        statement = statement.order_by(desc(timestamp_column))
    return _safe_rows(session, statement.limit(_bounded(limit, default=_MAX_ROWS, maximum=_MAX_ROWS)))


def _query_v2_trace(session: Session, model_name: str, trace_id: str, *, limit: int = _MAX_ROWS) -> list[Any]:
    return _query_v2_by_value(session, model_name, ("decision_trace_id", "trace_id"), trace_id, limit=limit)


def _latest_v2(
    session: Session,
    model_name: str,
    *,
    symbol: str | None = None,
    account_id: str | None = None,
    end: datetime | None = None,
) -> Any | None:
    rows = _query_v2(
        session,
        model_name,
        symbol=symbol,
        end=end,
        account_id=account_id,
        limit=1,
    )
    return rows[0] if rows else None


def _champion_assignment(session: Session, symbol: str) -> Any | None:
    """Return an active assignment scoped to the symbol or the global universe."""
    model = _model("ChampionAssignment")
    if model is None:
        return None
    scope = _column(model, "symbol_scope")
    active_from = _column(model, "active_from", "created_at")
    active_to = _column(model, "active_to")
    statement = select(model)
    if scope is not None:
        statement = statement.where(or_(scope == symbol, scope == "*"))
    if active_to is not None:
        statement = statement.where(or_(active_to.is_(None), active_to > datetime.now(timezone.utc)))
    if active_from is not None:
        statement = statement.order_by(desc(active_from))
    rows = _safe_rows(session, statement.limit(1))
    return rows[0] if rows else None


def _latest_risk_for_target(session: Session, target: Any | None, account_id: str | None) -> Any | None:
    target_id = _value(target, "portfolio_target_id", "id") if target else None
    if target_id is None:
        return _latest_v2(session, "RiskDecisionRecord", account_id=account_id)
    rows = _query_v2_by_value(
        session,
        "RiskDecisionRecord",
        ("portfolio_target_id",),
        target_id,
        limit=1,
    )
    if not rows:
        return None
    if account_id:
        row_account = _value(rows[0], "paper_account_id", "account_id")
        if row_account and str(row_account) != account_id:
            return None
    return rows[0]


def _trace_ids_for_symbol(
    session: Session,
    symbol: str,
    *,
    start: datetime | None,
    end: datetime | None,
    limit: int,
) -> set[str]:
    """Find trace ids through entities that carry a recorded symbol column."""
    trace_ids: set[str] = set()
    for model_name in (
        "ModelPredictionRecord",
        "TradingSignalRecord",
        "EnsembleDecisionRecord",
        "PortfolioTargetRecord",
        "SimulatedOrderRecord",
        "SimulatedFillRecord",
    ):
        for row in _query_v2(
            session,
            model_name,
            symbol=symbol,
            start=start,
            end=end,
            limit=limit,
        ):
            trace_id = _identifier(row, "decision_trace_id", "trace_id")
            if trace_id:
                trace_ids.add(trace_id)
    return trace_ids


def _age_seconds(value: datetime | None) -> float | None:
    value = _as_utc(value)
    if value is None:
        return None
    return max((datetime.now(timezone.utc) - value).total_seconds(), 0.0)


def _stale(value: datetime | None, *, after_seconds: int = 300) -> bool:
    age = _age_seconds(value)
    return age is None or age > after_seconds


def _legacy_trade_marker(row: Any, decision_id: int | None = None) -> dict[str, Any]:
    raw_payload = _mapping(_value(row, "raw_payload", "raw"))
    intent = str(raw_payload.get("intent") or "").lower()
    action = str(_value(row, "action", default="")).upper()
    marker_kind = "exit" if intent == "close" or action == "CLOSE" else "entry"
    created_at = _value(row, "created_at")
    return {
        "id": f"legacy-trade:{_identifier(row, 'id') or 'unknown'}",
        "trade_id": _identifier(row, "id"),
        "decision_id": decision_id,
        "source": "legacy",
        "time": _dt(created_at),
        "epoch": _epoch(created_at),
        "kind": marker_kind,
        "action": action or None,
        "side": _value(row, "side"),
        "price": _value(row, "price"),
        "quantity": _value(row, "quantity"),
        "notional": _value(row, "notional"),
        "fee": _value(row, "fee"),
        "realized_pnl": _value(row, "realized_pnl"),
        "status": _value(row, "status"),
        "reason": _value(row, "reason"),
        "intent": intent or None,
    }


def _legacy_trades(
    session: Session,
    *,
    symbol: str,
    start: datetime | None,
    end: datetime | None,
    limit: int,
    paper_account_id: str | None = None,
) -> list[Any]:
    paper_trade = db_models.PaperTrade
    statement = select(paper_trade).where(paper_trade.symbol == symbol)
    if paper_account_id:
        statement = statement.where(paper_trade.paper_account_id == paper_account_id)
    if start is not None:
        statement = statement.where(paper_trade.created_at >= start)
    if end is not None:
        statement = statement.where(paper_trade.created_at <= end)
    statement = statement.order_by(desc(paper_trade.created_at)).limit(limit)
    rows = _safe_rows(session, statement)
    return list(reversed(rows))


def _legacy_trade_decision_ids(session: Session, trade_ids: list[int]) -> dict[int, int]:
    if not trade_ids:
        return {}
    ai_decision = db_models.AiDecision
    rows = _safe_rows(session, select(ai_decision).where(ai_decision.trade_id.in_(trade_ids)))
    return {
        int(row.trade_id): int(row.id)
        for row in rows
        if _value(row, "trade_id") is not None and _value(row, "id") is not None
    }


def _serialize_prediction(row: Any) -> dict[str, Any]:
    generated_at = _value(row, "generated_at", "created_at")
    return {
        "id": _identifier(row, "prediction_id", "id"),
        "prediction_id": _identifier(row, "prediction_id", "id"),
        "decision_trace_id": _identifier(row, "decision_trace_id", "trace_id"),
        "model_id": _value(row, "model_id"),
        "model_version": _value(row, "model_version", "version"),
        "model_family": _value(row, "model_family"),
        "symbol": _value(row, "symbol"),
        "generated_at": _dt(generated_at),
        "valid_from": _dt(_value(row, "valid_from")),
        "expires_at": _dt(_value(row, "expires_at", "valid_until")),
        "forecast_horizon_seconds": _value(row, "forecast_horizon_seconds", "forecast_horizon"),
        "expected_return": _value(row, "expected_return"),
        "expected_volatility": _value(row, "expected_volatility"),
        "probability_up": _value(row, "probability_up"),
        "probability_down": _value(row, "probability_down"),
        "confidence": _value(row, "confidence"),
        "calibration_score": _value(row, "calibration_score"),
        "uncertainty": _value(row, "uncertainty"),
        "regime": _value(row, "regime"),
        "feature_schema_version": _value(row, "feature_schema_version"),
        "feature_snapshot_id": _identifier(row, "feature_snapshot_id", "feature_id"),
        "data_version": _value(row, "data_version"),
        "external_context_available": _value(row, "external_context_available"),
        "metadata": _json_safe(_mapping(_value(row, "metadata", "payload"))),
    }


def _serialize_signal(row: Any) -> dict[str, Any]:
    generated_at = _value(row, "generated_at", "created_at")
    return {
        "id": _identifier(row, "signal_id", "id"),
        "signal_id": _identifier(row, "signal_id", "id"),
        "prediction_id": _identifier(row, "prediction_id"),
        "decision_trace_id": _identifier(row, "decision_trace_id", "trace_id"),
        "signal_family": _value(row, "signal_family"),
        "model_id": _value(row, "model_id"),
        "model_version": _value(row, "model_version"),
        "symbol": _value(row, "symbol"),
        "generated_at": _dt(generated_at),
        "valid_until": _dt(_value(row, "valid_until", "expires_at")),
        "direction": _value(row, "direction"),
        "strength": _value(row, "strength"),
        "expected_return": _value(row, "expected_return"),
        "expected_cost": _value(row, "expected_cost"),
        "net_expected_return": _value(row, "net_expected_return"),
        "confidence": _value(row, "confidence"),
        "uncertainty": _value(row, "uncertainty"),
        "regime": _value(row, "regime"),
        "liquidity_score": _value(row, "liquidity_score"),
        "health_status": _value(row, "health_status"),
        "lifecycle_status": _value(row, "lifecycle_status"),
        "reason_codes": _json_safe(_list(_value(row, "reason_codes"))),
        "metadata": _json_safe(_mapping(_value(row, "metadata", "payload"))),
    }


def _serialize_ensemble(row: Any) -> dict[str, Any]:
    generated_at = _value(row, "generated_at", "created_at")
    return {
        "id": _identifier(row, "ensemble_decision_id", "id"),
        "ensemble_decision_id": _identifier(row, "ensemble_decision_id", "id"),
        "decision_trace_id": _identifier(row, "decision_trace_id", "trace_id"),
        "paper_account_id": _identifier(row, "paper_account_id", "account_id", "paper_sandbox_account_id"),
        "symbol": _value(row, "symbol"),
        "generated_at": _dt(generated_at),
        "valid_until": _dt(_value(row, "valid_until", "expires_at")),
        "combined_expected_return": _value(row, "combined_expected_return"),
        "combined_expected_volatility": _value(row, "combined_expected_volatility"),
        "combined_uncertainty": _value(row, "combined_uncertainty"),
        "combined_confidence": _value(row, "combined_confidence"),
        "current_regime": _value(row, "current_regime", "regime"),
        "supporting_signals": _json_safe(_list(_value(row, "supporting_signals"))),
        "conflicting_signals": _json_safe(_list(_value(row, "conflicting_signals"))),
        "signal_weights": _json_safe(_value(row, "signal_weights", default={})),
        "correlation_penalty": _value(row, "correlation_penalty"),
        "transaction_cost_penalty": _value(row, "transaction_cost_penalty"),
        "regime_penalty": _value(row, "regime_penalty"),
        "external_context_adjustment": _value(row, "external_context_adjustment"),
        "decision_status": _value(row, "decision_status", "status"),
        "reason_codes": _json_safe(_list(_value(row, "reason_codes"))),
    }


def _serialize_target(row: Any) -> dict[str, Any]:
    created_at = _value(row, "created_at", "generated_at")
    return {
        "id": _identifier(row, "portfolio_target_id", "id"),
        "portfolio_target_id": _identifier(row, "portfolio_target_id", "id"),
        "decision_trace_id": _identifier(row, "decision_trace_id", "trace_id"),
        "source_ensemble_decision_id": _identifier(row, "source_ensemble_decision_id", "ensemble_decision_id"),
        "symbol": _value(row, "symbol"),
        "created_at": _dt(created_at),
        "current_exposure": _value(row, "current_exposure"),
        "requested_target_exposure": _value(row, "requested_target_exposure"),
        "requested_delta": _value(row, "requested_delta"),
        "expected_return": _value(row, "expected_return"),
        "expected_risk": _value(row, "expected_risk"),
        "risk_contribution": _value(row, "risk_contribution"),
        "urgency": _value(row, "urgency"),
    }


def _serialize_risk(row: Any) -> dict[str, Any]:
    created_at = _value(row, "created_at", "generated_at")
    return {
        "id": _identifier(row, "risk_decision_id", "id"),
        "risk_decision_id": _identifier(row, "risk_decision_id", "id"),
        "decision_trace_id": _identifier(row, "decision_trace_id", "trace_id"),
        "portfolio_target_id": _identifier(row, "portfolio_target_id"),
        "paper_account_id": _identifier(row, "paper_account_id", "account_id", "paper_sandbox_account_id"),
        "symbol": _value(row, "symbol"),
        "created_at": _dt(created_at),
        "approved": _value(row, "approved"),
        "requested_exposure": _value(row, "requested_exposure"),
        "approved_exposure": _value(row, "approved_exposure"),
        "requested_leverage": _value(row, "requested_leverage"),
        "approved_leverage": _value(row, "approved_leverage"),
        "triggered_limits": _json_safe(_list(_value(row, "triggered_limits"))),
        "rejection_reasons": _json_safe(_list(_value(row, "rejection_reasons"))),
        "configuration_version": _value(row, "configuration_version"),
        "kill_switch_state": _value(row, "kill_switch_state"),
    }


def _serialize_order(row: Any) -> dict[str, Any]:
    created_at = _value(row, "submitted_at", "created_at")
    return {
        "id": _identifier(row, "order_id", "id"),
        "order_id": _identifier(row, "order_id", "id"),
        "decision_trace_id": _identifier(row, "decision_trace_id", "trace_id"),
        "risk_decision_id": _identifier(row, "risk_decision_id"),
        "portfolio_target_id": _identifier(row, "portfolio_target_id"),
        "account_id": _identifier(row, "paper_account_id", "account_id", "paper_sandbox_account_id"),
        "symbol": _value(row, "symbol"),
        "side": _value(row, "side", "direction"),
        "state": _value(row, "state", "status"),
        "quantity": _value(row, "quantity", "requested_quantity"),
        "notional": _value(row, "notional", "requested_notional"),
        "submitted_at": _dt(created_at),
        "reason_codes": _json_safe(_list(_value(row, "reason_codes"))),
    }


def _serialize_fill(row: Any) -> dict[str, Any]:
    filled_at = _value(row, "filled_at", "created_at")
    return {
        "id": _identifier(row, "fill_id", "id"),
        "fill_id": _identifier(row, "fill_id", "id"),
        "order_id": _identifier(row, "order_id"),
        "decision_trace_id": _identifier(row, "decision_trace_id", "trace_id"),
        "account_id": _identifier(row, "paper_account_id", "account_id", "paper_sandbox_account_id"),
        "symbol": _value(row, "symbol"),
        "side": _value(row, "side", "direction"),
        "state": _value(row, "state", "status"),
        "quantity": _value(row, "quantity", "filled_quantity"),
        "price": _value(row, "price", "fill_price"),
        "notional": _value(row, "notional"),
        "fee": _value(row, "fee", "fee_amount"),
        "slippage": _value(row, "slippage", "slippage_amount"),
        "funding": _value(row, "funding", "funding_amount"),
        "realized_pnl": _value(row, "realized_pnl"),
        "filled_at": _dt(filled_at),
    }


def _serialize_structured_news(row: Any) -> dict[str, Any]:
    event_time = _value(row, "event_time", "available_to_model_time", "published_time", "created_at")
    assets = _list(_value(row, "affected_assets", "symbols"))
    return {
        "id": _identifier(row, "structured_news_event_id", "id"),
        "source": "structured_news",
        "time": _dt(event_time),
        "epoch": _epoch(event_time),
        "primary_symbol": _value(row, "primary_symbol", "symbol"),
        "event_type": _value(row, "event_type"),
        "affected_assets": _json_safe(assets),
        "affected_entities": _json_safe(_list(_value(row, "affected_entities"))),
        "direction": _value(row, "direction"),
        "sentiment": _value(row, "sentiment"),
        "severity": _value(row, "severity"),
        "importance": _value(row, "importance"),
        "novelty": _value(row, "novelty"),
        "time_horizon": _value(row, "time_horizon"),
        "confidence": _value(row, "confidence"),
        "source_summary": _value(row, "source_summary"),
        "provider": _value(row, "provider"),
        "validation_status": _value(row, "validation_status", "status"),
    }


def _serialize_external_request(row: Any) -> dict[str, Any]:
    return {
        "id": _identifier(row, "external_ai_request_id", "id"),
        "symbol": _value(row, "symbol"),
        "provider": _value(row, "provider"),
        "model": _value(row, "model"),
        "status": _value(row, "status"),
        "requested_at": _dt(_value(row, "requested_at", "created_at")),
        "completed_at": _dt(_value(row, "completed_at")),
        "cache_hit": _value(row, "cache_hit"),
        "error_category": _value(row, "error_category"),
        "retry_count": _value(row, "retry_count"),
        "external_ai_available": _value(row, "external_ai_available"),
    }


def _legacy_news(
    session: Session,
    *,
    symbol: str,
    start: datetime | None,
    end: datetime | None,
    limit: int,
) -> list[dict[str, Any]]:
    article = db_models.NewsArticle
    sentiment = db_models.NewsSentiment
    statement = select(article, sentiment).outerjoin(sentiment, sentiment.article_id == article.id)
    if start is not None:
        statement = statement.where(article.published_at >= start)
    if end is not None:
        statement = statement.where(article.published_at <= end)
    rows = _safe_execute(session, statement.order_by(desc(article.published_at)).limit(min(limit * 3, 750)))
    output: list[dict[str, Any]] = []
    for news_row, sentiment_row in rows:
        affected = _list(_value(sentiment_row, "affected_symbols")) if sentiment_row else []
        affected_symbols = [str(item).upper() for item in affected if str(item).strip()]
        if affected_symbols and symbol not in affected_symbols:
            continue
        published_at = _value(news_row, "published_at", "created_at")
        output.append(
            {
                "id": f"legacy-news:{_identifier(news_row, 'id') or 'unknown'}",
                "source": "legacy_news",
                "time": _dt(published_at),
                "epoch": _epoch(published_at),
                "title": _value(news_row, "title"),
                "url": _value(news_row, "url"),
                "provider": _value(news_row, "source_name", "source"),
                "affected_symbols": affected_symbols,
                "symbol_scope": "recorded_symbol" if affected_symbols else "unclassified",
                "sentiment_score": _value(sentiment_row, "sentiment_score") if sentiment_row else None,
                "risk_score": _value(sentiment_row, "risk_score") if sentiment_row else None,
                "confidence": _value(sentiment_row, "confidence") if sentiment_row else None,
                "model_name": _value(sentiment_row, "model_name") if sentiment_row else None,
            }
        )
        if len(output) >= limit:
            break
    return list(reversed(output))


def _legacy_liquidations(
    session: Session,
    *,
    symbol: str,
    start: datetime | None,
    end: datetime | None,
    limit: int,
) -> list[dict[str, Any]]:
    event = db_models.ExternalDataEvent
    statement = select(event).where(
        or_(event.symbol == symbol, event.symbol.is_(None))
    )
    if start is not None:
        statement = statement.where(event.event_time >= start)
    if end is not None:
        statement = statement.where(event.event_time <= end)
    rows = _safe_rows(session, statement.order_by(desc(event.event_time)).limit(min(limit * 3, 750)))
    output: list[dict[str, Any]] = []
    for row in rows:
        source = str(_value(row, "source_name", default=""))
        data_type = str(_value(row, "data_type", default=""))
        if "liquidation" not in source.lower() and "liquidation" not in data_type.lower():
            continue
        event_time = _value(row, "event_time", "created_at")
        output.append(
            {
                "id": f"legacy-liquidation:{_identifier(row, 'id') or 'unknown'}",
                "source": source or "legacy_external",
                "time": _dt(event_time),
                "epoch": _epoch(event_time),
                "symbol": _value(row, "symbol"),
                "data_type": data_type or None,
                "numeric_value": _value(row, "numeric_value"),
                "payload": _json_safe(_mapping(_value(row, "payload"))),
            }
        )
        if len(output) >= limit:
            break
    return list(reversed(output))


def _weight_for_signal(weights: Any, signal: dict[str, Any]) -> float | None:
    signal_id = str(signal.get("signal_id") or signal.get("id") or "")
    prediction_id = str(signal.get("prediction_id") or "")
    if isinstance(weights, dict):
        for key in (signal_id, prediction_id, signal.get("model_id")):
            if key is None:
                continue
            value = weights.get(str(key))
            try:
                return float(value) if value is not None else None
            except (TypeError, ValueError):
                continue
    for item in _list(weights):
        if not isinstance(item, dict):
            continue
        candidate = str(item.get("signal_id") or item.get("prediction_id") or item.get("model_id") or "")
        if candidate and candidate in {signal_id, prediction_id, str(signal.get("model_id") or "")}:
            try:
                return float(item.get("weight"))
            except (TypeError, ValueError):
                return None
    return None


def _legacy_decision_payload(row: Any) -> dict[str, Any]:
    created_at = _value(row, "created_at")
    raw = _mapping(_value(row, "raw", "raw_payload"))
    return {
        "id": int(_value(row, "id") or 0),
        "trace_id": f"legacy:{_value(row, 'id')}",
        "source": "legacy",
        "symbol": _value(row, "symbol"),
        "time": _dt(created_at),
        "action": _value(row, "action"),
        "confidence": _value(row, "confidence"),
        "status": _value(row, "execution_status"),
        "reason": _value(row, "reason", "execution_message"),
        "strategy_name": _value(row, "strategy_name"),
        "source_name": _value(row, "source_name"),
        "feature_id": _identifier(row, "feature_id"),
        "trade_id": _identifier(row, "trade_id"),
        "model_version_id": _identifier(row, "model_version_id"),
        "decision_source": raw.get("decision_source") or "legacy",
    }


@router.get("/symbols")
def vision_symbols(session: Session = Depends(get_session)) -> dict[str, Any]:
    """Return the configured and recorded symbols available to the Vision page."""
    configured = list(
        dict.fromkeys(
            [
                *settings.binance_symbols,
                *settings.auto_trader_symbols,
                *settings.derivatives_symbols,
            ]
        )
    )
    latest_by_symbol: dict[str, datetime | None] = {}
    candle = db_models.Candle
    rows = _safe_execute(
        session,
        select(candle.symbol, func.max(candle.open_time)).group_by(candle.symbol).limit(_MAX_ROWS),
    )
    for symbol, timestamp in rows:
        if symbol:
            latest_by_symbol[str(symbol).upper()] = _as_utc(timestamp)

    observed: set[str] = set(latest_by_symbol)
    for model_name in ("Position", "ModelPredictionRecord", "TradingSignalRecord", "EnsembleDecisionRecord"):
        model = _model(model_name)
        symbol_column = _column(model, "symbol")
        if symbol_column is None:
            continue
        for value in _safe_rows(session, select(symbol_column).distinct().limit(_MAX_ROWS)):
            if value:
                observed.add(str(value).upper())

    ordered = [symbol for symbol in configured if symbol in observed or symbol]
    ordered.extend(sorted(observed.difference(ordered)))
    symbols = [
        {
            "symbol": symbol,
            "latest_candle_at": _dt(latest_by_symbol.get(symbol)),
            "stale": _stale(latest_by_symbol.get(symbol)),
            "configured": symbol in configured,
        }
        for symbol in ordered[:_MAX_ROWS]
    ]
    return {
        "symbols": symbols,
        "default_symbol": configured[0] if configured else (symbols[0]["symbol"] if symbols else "BTCUSDT"),
    }


@router.get("/chart")
def vision_chart(
    symbol: str,
    timeframe: str = "1m",
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int = _DEFAULT_LIMIT,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Return only stored candle data for the main Vision chart."""
    symbol = _normal_symbol(symbol)
    start, end = _window(start, end)
    timeframe = (timeframe or "1m").strip().lower()
    if not re.fullmatch(r"\d{1,4}[smhd]", timeframe):
        raise HTTPException(status_code=400, detail="Unsupported timeframe")
    safe_limit = _bounded(limit, default=_DEFAULT_LIMIT, maximum=_MAX_CANDLES)
    candle = db_models.Candle
    statement = select(candle).where(
        candle.symbol == symbol,
        candle.interval == timeframe,
        candle.is_closed.is_(True),
    )
    if start is not None:
        statement = statement.where(candle.open_time >= start)
    if end is not None:
        statement = statement.where(candle.open_time <= end)
    # Select the newest bounded slice inside the requested window, then return it
    # chronologically for chart rendering.  Ascending LIMIT would otherwise show
    # the oldest part of a 24h/7d window and make fresh data appear stale.
    rows = list(reversed(_safe_rows(session, statement.order_by(desc(candle.open_time)).limit(safe_limit))))

    output = [
        {
            "id": row.id,
            "symbol": row.symbol,
            "timeframe": row.interval,
            "time": _epoch(row.open_time),
            "open_time": _dt(row.open_time),
            "close_time": _dt(row.close_time),
            "open": row.open,
            "high": row.high,
            "low": row.low,
            "close": row.close,
            "volume": row.volume,
            "is_closed": bool(row.is_closed),
            "source_name": row.source_name,
        }
        for row in rows
    ]

    live = db_models.LiveCandleUpdate
    live_statement = select(live).where(live.symbol == symbol, live.interval == timeframe)
    if start is not None:
        live_statement = live_statement.where(live.open_time >= start)
    if end is not None:
        live_statement = live_statement.where(live.open_time <= end)
    live_row = _safe_scalar(session, live_statement.order_by(desc(live.open_time)).limit(1))
    if live_row and (not output or _epoch(live_row.open_time) > output[-1]["time"]):
        output.append(
            {
                "id": live_row.id,
                "symbol": live_row.symbol,
                "timeframe": live_row.interval,
                "time": _epoch(live_row.open_time),
                "open_time": _dt(live_row.open_time),
                "close_time": _dt(live_row.close_time),
                "open": live_row.open,
                "high": live_row.high,
                "low": live_row.low,
                "close": live_row.close,
                "volume": live_row.volume,
                "is_closed": False,
                "source_name": live_row.source_name or "live_candle_updates",
            }
        )
    latest_at = _as_utc(_value(live_row, "open_time")) if live_row and output and not output[-1]["is_closed"] else (
        _as_utc(rows[-1].open_time) if rows else None
    )
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "start": _dt(start),
        "end": _dt(end),
        "candles": output[-safe_limit:],
        "latest_at": _dt(latest_at),
        "age_seconds": _age_seconds(latest_at),
        "stale": _stale(latest_at),
        "source": "stored_candles",
    }


@router.get("/overlays")
def vision_overlays(
    symbol: str,
    start: datetime | None = None,
    end: datetime | None = None,
    account_id: str | None = None,
    limit: int = _DEFAULT_LIMIT,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Return bounded chart overlays without mixing them into candle payloads."""
    symbol = _normal_symbol(symbol)
    account_id = _normal_identifier(account_id, field="account_id")
    start, end = _window(start, end)
    safe_limit = _bounded(limit, default=_DEFAULT_LIMIT, maximum=_MAX_EVENTS)
    legacy_allowed = (account_id or "").lower() in _MAIN_ACCOUNT_IDS
    effective_account_id = account_id or "champion"

    legacy_trades = _legacy_trades(
        session,
        symbol=symbol,
        start=start,
        end=end,
        limit=safe_limit,
        paper_account_id=effective_account_id,
    ) if legacy_allowed else []
    trade_ids = [int(row.id) for row in legacy_trades if row.id is not None]
    decision_ids = _legacy_trade_decision_ids(session, trade_ids)
    trades = [_legacy_trade_marker(row, decision_ids.get(int(row.id))) for row in legacy_trades]

    prediction_rows = _query_v2(
        session,
        "ModelPredictionRecord",
        symbol=symbol,
        start=start,
        end=end,
        limit=safe_limit,
        time_candidates=("generated_at",),
        ascending=True,
    )
    ensemble_rows = _query_v2(
        session,
        "EnsembleDecisionRecord",
        symbol=symbol,
        start=start,
        end=end,
        limit=safe_limit,
        time_candidates=("generated_at",),
        ascending=True,
    )
    target_rows = _query_v2(
        session,
        "PortfolioTargetRecord",
        symbol=symbol,
        start=start,
        end=end,
        account_id=effective_account_id,
        limit=safe_limit,
        time_candidates=("created_at",),
        ascending=True,
    )
    target_ids = {
        str(_value(row, "portfolio_target_id", "id"))
        for row in target_rows
        if _value(row, "portfolio_target_id", "id") is not None
    }
    risk_rows = [
        row
        for row in _query_v2(
            session,
            "RiskDecisionRecord",
            start=start,
            end=end,
            account_id=effective_account_id,
            limit=_MAX_EVENTS,
            time_candidates=("created_at",),
            ascending=True,
        )
        if not target_ids or str(_value(row, "portfolio_target_id")) in target_ids
    ][-safe_limit:]
    order_rows = _query_v2(
        session,
        "SimulatedOrderRecord",
        symbol=symbol,
        start=start,
        end=end,
        account_id=effective_account_id,
        limit=safe_limit,
        time_candidates=("submitted_at", "created_at"),
        ascending=True,
    )
    fill_rows = _query_v2(
        session,
        "SimulatedFillRecord",
        symbol=symbol,
        start=start,
        end=end,
        account_id=effective_account_id,
        limit=safe_limit,
        time_candidates=("filled_at", "created_at"),
        ascending=True,
    )
    structured_news_rows = _query_v2(
        session,
        "StructuredNewsEvent",
        symbol=symbol,
        start=start,
        end=end,
        limit=safe_limit,
        time_candidates=("available_to_model_time", "published_time", "created_at"),
        ascending=True,
    )
    structured_news = [_serialize_structured_news(row) for row in structured_news_rows]
    legacy_news = _legacy_news(session, symbol=symbol, start=start, end=end, limit=safe_limit)
    liquidations = _legacy_liquidations(session, symbol=symbol, start=start, end=end, limit=safe_limit)

    ensembles = [_serialize_ensemble(row) for row in ensemble_rows]
    disagreements = [
        {
            "id": item["id"],
            "time": item["generated_at"],
            "epoch": _epoch(_value(row, "generated_at", "created_at")),
            "supporting_signals": item["supporting_signals"],
            "conflicting_signals": item["conflicting_signals"],
            "combined_uncertainty": item["combined_uncertainty"],
        }
        for row, item in zip(ensemble_rows, ensembles)
        if item["conflicting_signals"]
    ]
    return {
        "symbol": symbol,
        "account_id": effective_account_id,
        "start": _dt(start),
        "end": _dt(end),
        "trades": trades,
        "orders": [_serialize_order(row) for row in order_rows],
        "fills": [_serialize_fill(row) for row in fill_rows],
        "predictions": [_serialize_prediction(row) for row in prediction_rows],
        "ensembles": ensembles,
        "model_disagreements": disagreements,
        "portfolio_targets": [_serialize_target(row) for row in target_rows],
        "risk_decisions": [_serialize_risk(row) for row in risk_rows],
        "news_events": [*legacy_news, *structured_news],
        "liquidation_events": liquidations,
        "availability": {
            "legacy_main_account_visible": legacy_allowed,
            "v2_records_available": bool(
                prediction_rows or ensemble_rows or target_rows or risk_rows or order_rows or fill_rows
            ),
        },
    }


@router.get("/state")
def vision_state(
    symbol: str,
    account_id: str | None = None,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Return the latest recorded state behind a paper decision for one symbol."""
    symbol = _normal_symbol(symbol)
    account_id = _normal_identifier(account_id, field="account_id")
    legacy_allowed = (account_id or "").lower() in _MAIN_ACCOUNT_IDS
    effective_account_id = account_id or "champion"

    feature = _safe_scalar(
        session,
        select(db_models.Feature).where(db_models.Feature.symbol == symbol).order_by(desc(db_models.Feature.as_of)).limit(1),
    )
    decision = _safe_scalar(
        session,
        select(db_models.AiDecision)
        .where(db_models.AiDecision.symbol == symbol)
        .order_by(desc(db_models.AiDecision.created_at))
        .limit(1),
    )
    position = None
    account = None
    if legacy_allowed:
        position_statement = (
            select(db_models.Position)
            .where(
                db_models.Position.symbol == symbol,
                db_models.Position.status == "OPEN",
                db_models.Position.paper_account_id == effective_account_id,
            )
            .order_by(desc(db_models.Position.opened_at))
            .limit(1)
        )
        position = _safe_scalar(
            session,
            position_statement,
        )
        account = _safe_scalar(
            session,
            select(db_models.AccountEquity)
            .where(db_models.AccountEquity.paper_account_id == effective_account_id)
            .order_by(desc(db_models.AccountEquity.timestamp))
            .limit(1),
        )

    assignment = _champion_assignment(session, symbol)
    ensemble_row = _latest_v2(session, "EnsembleDecisionRecord", symbol=symbol)
    target_row = _latest_v2(session, "PortfolioTargetRecord", symbol=symbol, account_id=effective_account_id)
    risk_row = _latest_risk_for_target(session, target_row, effective_account_id)
    external_row = _latest_v2(session, "ExternalAIRequest", symbol=symbol)

    active_model = _safe_scalar(
        session,
        select(db_models.ModelVersion)
        .where(db_models.ModelVersion.status == "active")
        .order_by(desc(db_models.ModelVersion.created_at))
        .limit(1),
    )
    assigned_model = None
    if assignment and _value(assignment, "model_version_id") is not None:
        assigned_model = _safe_scalar(
            session,
            select(db_models.ModelVersion)
            .where(db_models.ModelVersion.id == _value(assignment, "model_version_id"))
            .limit(1),
        )
    latest_sentiment = _safe_scalar(
        session,
        select(db_models.NewsSentiment).order_by(desc(db_models.NewsSentiment.created_at)).limit(1),
    )
    latest_candle = _safe_scalar(
        session,
        select(db_models.Candle)
        .where(db_models.Candle.symbol == symbol, db_models.Candle.is_closed.is_(True))
        .order_by(desc(db_models.Candle.open_time))
        .limit(1),
    )

    champion = {
        "model_id": (
            _value(assignment, "model_id")
            or _value(assigned_model, "model_id")
            or _value(active_model, "model_id")
        ),
        "model_version": (
            _value(assignment, "model_version", "version")
            or _value(assigned_model, "version")
            or _value(active_model, "version")
        ),
        "model_family": _value(assignment, "model_family") or _value(assigned_model, "model_family"),
        "status": _value(assignment, "status", "lifecycle_status") if assignment else (
            "legacy_active_model" if active_model else "unavailable"
        ),
        "source": "champion_assignment" if assignment else ("legacy_model_version" if active_model else "unavailable"),
    }
    ensemble = _serialize_ensemble(ensemble_row) if ensemble_row else None
    target = _serialize_target(target_row) if target_row else None
    risk = _serialize_risk(risk_row) if risk_row else None
    legacy_decision = _legacy_decision_payload(decision) if decision else None
    legacy_feature_payload = _mapping(_value(feature, "payload")) if feature else {}
    legacy_feature_values = _mapping(legacy_feature_payload.get("values"))

    reason_codes: list[Any] = []
    if ensemble:
        reason_codes.extend(ensemble.get("reason_codes") or [])
    if risk:
        reason_codes.extend(risk.get("triggered_limits") or [])
        reason_codes.extend(risk.get("rejection_reasons") or [])
    if not reason_codes and legacy_decision and legacy_decision.get("reason"):
        reason_codes.append(legacy_decision["reason"])

    if external_row:
        external_ai = _serialize_external_request(external_row)
    else:
        external_ai = {
            "id": None,
            "symbol": symbol,
            "provider": None,
            "model": None,
            "status": "not_recorded",
            "requested_at": None,
            "completed_at": None,
            "cache_hit": None,
            "error_category": None,
            "retry_count": None,
            "external_ai_available": False,
        }
    legacy_position = None
    if position:
        legacy_position = {
            "source": "legacy",
            "symbol": position.symbol,
            "side": position.side,
            "quantity": position.quantity,
            "entry_price": position.entry_price,
            "current_price": position.current_price,
            "notional": position.notional,
            "margin_used": position.margin_used,
            "leverage": position.leverage,
            "unrealized_pnl": position.unrealized_pnl,
            "stop_loss": position.stop_loss,
            "take_profit": position.take_profit,
            "status": position.status,
            "opened_at": _dt(position.opened_at),
        }
    return {
        "symbol": symbol,
        "account_id": effective_account_id,
        "source": "v2" if any((assignment, ensemble_row, target_row, risk_row, external_row)) else "legacy",
        "champion": champion,
        "ensemble": ensemble,
        "portfolio_target": target,
        "risk_decision": risk,
        "position": legacy_position,
        "account": {
            "cash_balance": _value(account, "cash_balance"),
            "equity": _value(account, "equity"),
            "realized_pnl": _value(account, "realized_pnl"),
            "unrealized_pnl": _value(account, "unrealized_pnl"),
            "drawdown": _value(account, "drawdown"),
            "as_of": _dt(_value(account, "timestamp")),
        }
        if account
        else None,
        "external_ai": external_ai,
        "local_news_model": {
            "version": _value(latest_sentiment, "model_name"),
            "source": "legacy_news_sentiment" if latest_sentiment else "unavailable",
            "as_of": _dt(_value(latest_sentiment, "created_at")),
        },
        "legacy": {
            "latest_decision": legacy_decision,
            "feature_snapshot": {
                "id": _identifier(feature, "id"),
                "as_of": _dt(_value(feature, "as_of")),
                "schema_version": _value(feature, "schema_version"),
                "trend": _value(feature, "trend") or legacy_feature_values.get("trend"),
                "volatility": _value(feature, "volatility") or legacy_feature_values.get("volatility"),
            }
            if feature
            else None,
        },
        "reason_codes": _json_safe(reason_codes),
        "data_status": {
            "latest_candle_at": _dt(_value(latest_candle, "open_time")),
            "candle_age_seconds": _age_seconds(_value(latest_candle, "open_time")),
            "stale": _stale(_value(latest_candle, "open_time")),
        },
    }


@router.get("/models")
def vision_models(
    symbol: str,
    at: datetime | None = None,
    limit: int = _DEFAULT_LIMIT,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Return the latest narrow-model evidence, or an explicitly partial legacy view."""
    symbol = _normal_symbol(symbol)
    at = _as_utc(at)
    safe_limit = _bounded(limit, default=_DEFAULT_LIMIT, maximum=_MAX_ROWS)
    prediction_rows = _query_v2(
        session,
        "ModelPredictionRecord",
        symbol=symbol,
        end=at,
        limit=safe_limit,
        time_candidates=("generated_at",),
    )
    signal_rows = _query_v2(
        session,
        "TradingSignalRecord",
        symbol=symbol,
        end=at,
        limit=safe_limit,
        time_candidates=("generated_at",),
    )
    ensemble_row = _latest_v2(session, "EnsembleDecisionRecord", symbol=symbol, end=at)
    weights = _value(ensemble_row, "signal_weights", default={}) if ensemble_row else {}

    signals_by_prediction: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in signal_rows:
        signal = _serialize_signal(row)
        if signal["prediction_id"]:
            signals_by_prediction[str(signal["prediction_id"])].append(signal)

    records: list[dict[str, Any]] = []
    seen_models: set[str] = set()
    prediction_ids_with_records: set[str] = set()
    for row in prediction_rows:
        prediction = _serialize_prediction(row)
        if prediction.get("prediction_id"):
            prediction_ids_with_records.add(str(prediction["prediction_id"]))
        model_key = ":".join(
            [
                str(prediction.get("model_id") or "unknown"),
                str(prediction.get("model_version") or "unknown"),
                str(prediction.get("model_family") or "unknown"),
            ]
        )
        if model_key in seen_models:
            continue
        seen_models.add(model_key)
        signal = (signals_by_prediction.get(str(prediction.get("prediction_id"))) or [None])[0]
        direction = signal.get("direction") if signal else None
        if direction is None and prediction.get("expected_return") is not None:
            try:
                expected_return = float(prediction["expected_return"])
            except (TypeError, ValueError):
                expected_return = 0.0
            direction = "BULLISH" if expected_return > 0 else (
                "BEARISH" if expected_return < 0 else "NEUTRAL"
            )
        records.append(
            {
                "id": prediction["id"],
                "source": "v2",
                "model_name": prediction.get("model_id"),
                "model_id": prediction.get("model_id"),
                "model_version": prediction.get("model_version"),
                "model_family": prediction.get("model_family"),
                "view": direction,
                "expected_return": prediction.get("expected_return"),
                "confidence": signal.get("confidence") if signal else prediction.get("confidence"),
                "uncertainty": signal.get("uncertainty") if signal else prediction.get("uncertainty"),
                "weight": _weight_for_signal(weights, signal) if signal else None,
                "health": signal.get("health_status") if signal else None,
                "lifecycle": signal.get("lifecycle_status") if signal else None,
                "last_prediction_time": prediction.get("generated_at"),
                "signal": signal,
            }
        )

    # Some signal producers may not persist a separate ModelPredictionRecord.
    for row in signal_rows:
        signal = _serialize_signal(row)
        if signal.get("prediction_id") and str(signal["prediction_id"]) in prediction_ids_with_records:
            continue
        model_key = ":".join(
            [
                str(signal.get("model_id") or signal.get("signal_family") or "unknown"),
                str(signal.get("model_version") or "unknown"),
            ]
        )
        if model_key in seen_models:
            continue
        seen_models.add(model_key)
        records.append(
            {
                "id": signal["id"],
                "source": "v2_signal_only",
                "model_name": signal.get("model_id") or signal.get("signal_family"),
                "model_id": signal.get("model_id"),
                "model_version": signal.get("model_version"),
                "model_family": signal.get("signal_family"),
                "view": signal.get("direction"),
                "expected_return": signal.get("expected_return"),
                "confidence": signal.get("confidence"),
                "uncertainty": signal.get("uncertainty"),
                "weight": _weight_for_signal(weights, signal),
                "health": signal.get("health_status"),
                "lifecycle": signal.get("lifecycle_status"),
                "last_prediction_time": signal.get("generated_at"),
                "signal": signal,
            }
        )

    if not records:
        legacy = _safe_scalar(
            session,
            select(db_models.AiDecision)
            .where(db_models.AiDecision.symbol == symbol)
            .order_by(desc(db_models.AiDecision.created_at))
            .limit(1),
        )
        if legacy:
            action = str(legacy.action or "").upper()
            view = {"BUY": "BULLISH", "SELL": "BEARISH", "HOLD": "NEUTRAL", "CLOSE": "NEUTRAL"}.get(action)
            records.append(
                {
                    "id": f"legacy:{legacy.id}",
                    "source": "legacy",
                    "model_name": legacy.strategy_name or legacy.source_name or "legacy decision",
                    "model_id": None,
                    "model_version": None,
                    "model_family": "legacy",
                    "view": view,
                    "expected_return": None,
                    "confidence": legacy.confidence,
                    "uncertainty": None,
                    "weight": None,
                    "health": None,
                    "lifecycle": None,
                    "last_prediction_time": _dt(legacy.created_at),
                    "signal": None,
                    "note": "Legacy decisions did not persist narrow-model predictions or ensemble weights.",
                }
            )
    return {
        "symbol": symbol,
        "at": _dt(at),
        "models": records[:safe_limit],
        "ensemble": _serialize_ensemble(ensemble_row) if ensemble_row else None,
        "message": None if records else "No recorded model prediction or signal is available for this symbol.",
    }


@router.get("/history")
def vision_history(
    symbol: str,
    start: datetime | None = None,
    end: datetime | None = None,
    account_id: str | None = None,
    limit: int = _DEFAULT_LIMIT,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Return stored paper-trading history and clearly label unavailable cost fields."""
    symbol = _normal_symbol(symbol)
    account_id = _normal_identifier(account_id, field="account_id")
    start, end = _window(start, end)
    safe_limit = _bounded(limit, default=_DEFAULT_LIMIT, maximum=_MAX_EVENTS)
    legacy_allowed = (account_id or "").lower() in _MAIN_ACCOUNT_IDS
    effective_account_id = account_id or "champion"
    legacy_rows = _legacy_trades(
        session,
        symbol=symbol,
        start=start,
        end=end,
        limit=safe_limit,
        paper_account_id=effective_account_id,
    ) if legacy_allowed else []
    legacy_trades = [_legacy_trade_marker(row) for row in legacy_rows]
    v2_fills = [
        _serialize_fill(row)
        for row in _query_v2(
            session,
            "SimulatedFillRecord",
            symbol=symbol,
            start=start,
            end=end,
            account_id=effective_account_id,
            limit=safe_limit,
            time_candidates=("filled_at", "created_at"),
            ascending=True,
        )
    ]

    realized_values = [float(item["realized_pnl"] or 0.0) for item in legacy_trades if item["realized_pnl"] is not None]
    winners = [value for value in realized_values if value > 0]
    losers = [value for value in realized_values if value < 0]
    fees = sum(float(item["fee"] or 0.0) for item in legacy_trades if item["fee"] is not None)

    equity_rows: list[Any] = []
    if legacy_allowed:
        equity = db_models.AccountEquity
        equity_statement = select(equity).where(equity.paper_account_id == effective_account_id)
        if start is not None:
            equity_statement = equity_statement.where(equity.timestamp >= start)
        if end is not None:
            equity_statement = equity_statement.where(equity.timestamp <= end)
        equity_rows = list(reversed(_safe_rows(session, equity_statement.order_by(desc(equity.timestamp)).limit(_MAX_EVENTS))))
    drawdowns = [float(_value(row, "drawdown") or 0.0) for row in equity_rows]

    profit_factor = None
    if losers:
        profit_factor = sum(winners) / abs(sum(losers))
    metrics = {
        "trade_count": len(legacy_trades),
        "win_count": len(winners),
        "loss_count": len(losers),
        "win_rate": (len(winners) / len(realized_values)) if realized_values else None,
        "average_win": (sum(winners) / len(winners)) if winners else None,
        "average_loss": (sum(losers) / len(losers)) if losers else None,
        "net_realized_pnl": sum(realized_values) if realized_values else None,
        "profit_factor": profit_factor,
        "fees": fees if legacy_trades else None,
        "maximum_drawdown": max(drawdowns) if drawdowns else None,
        "slippage": None,
        "funding": None,
        "net_expectancy": (sum(realized_values) / len(realized_values)) if realized_values else None,
    }
    attribution = paper_pnl_attribution(
        session,
        symbol=symbol,
        account_id=effective_account_id,
        start=start,
        end=end,
        limit=safe_limit,
    )
    metrics["slippage"] = attribution["components"]["slippage"]
    metrics["funding"] = attribution["components"]["funding"]
    return {
        "symbol": symbol,
        "account_id": effective_account_id,
        "start": _dt(start),
        "end": _dt(end),
        "legacy_trades": legacy_trades,
        "simulated_fills": v2_fills,
        "equity": [
            {
                "time": _dt(_value(row, "timestamp")),
                "equity": _value(row, "equity"),
                "cash_balance": _value(row, "cash_balance"),
                "drawdown": _value(row, "drawdown"),
            }
            for row in equity_rows
        ],
        "metrics": metrics,
        "performance_by_model": [
            {"model": key, "alpha_contribution": value}
            for key, value in attribution["by_model"].items()
        ],
        "performance_by_signal": [
            {"signal": key, "alpha_contribution": value}
            for key, value in attribution["by_signal"].items()
        ],
        "performance_by_signal_family": [
            {"signal_family": key, "alpha_contribution": value}
            for key, value in attribution["by_signal_family"].items()
        ],
        "performance_by_regime": [
            {"regime": key, "paper_pnl": value}
            for key, value in attribution["by_regime"].items()
        ],
        "performance_by_external_ai": attribution["by_external_ai_availability"],
        "attribution": attribution,
        "availability": {
            "legacy_main_account_visible": legacy_allowed,
            "slippage": "estimated_for_v2_fills_only",
            "funding": "recorded_for_v2_fills_only",
            "attribution": "trace_based_with_explicit_counterfactual_limitations",
            "note": "Legacy opening fills record entry fees as realized PnL; they are not closed-trade attribution.",
        },
    }


@router.get("/decisions")
def vision_decisions(
    symbol: str,
    start: datetime | None = None,
    end: datetime | None = None,
    source: str = "all",
    limit: int = _DEFAULT_LIMIT,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """List replayable legacy and V2 decision traces."""
    symbol = _normal_symbol(symbol)
    start, end = _window(start, end)
    source = source.lower().strip()
    if source not in {"all", "legacy", "v2"}:
        raise HTTPException(status_code=400, detail="source must be all, legacy, or v2")
    safe_limit = _bounded(limit, default=_DEFAULT_LIMIT, maximum=_MAX_ROWS)
    decisions: list[dict[str, Any]] = []

    if source in {"all", "v2"}:
        symbol_trace_ids = _trace_ids_for_symbol(
            session,
            symbol,
            start=start,
            end=end,
            limit=_MAX_EVENTS,
        )
        timeline_rows = _query_v2(
            session,
            "DecisionTimelineEvent",
            start=start,
            end=end,
            limit=_MAX_EVENTS,
            time_candidates=("occurred_at", "created_at"),
        )
        grouped: dict[str, list[Any]] = defaultdict(list)
        for row in timeline_rows:
            trace_id = _identifier(row, "decision_trace_id", "trace_id")
            if trace_id and trace_id in symbol_trace_ids:
                grouped[trace_id].append(row)
        for trace_id, rows in grouped.items():
            latest = rows[0]
            occurred_at = _value(latest, "occurred_at", "created_at")
            decisions.append(
                {
                    "id": trace_id,
                    "trace_id": trace_id,
                    "source": "v2",
                    "symbol": _value(latest, "symbol") or symbol,
                    "time": _dt(occurred_at),
                    "status": _value(latest, "status"),
                    "stage": _value(latest, "stage", "event_type"),
                    "event_count": len(rows),
                    "reason_codes": _json_safe(_list(_value(latest, "reason_codes"))),
                }
            )

        # A freshly written trace may have persisted its risk decision before
        # the optional timeline writer flushes; list it as a real partial trace.
        for row in _query_v2(
            session,
            "RiskDecisionRecord",
            start=start,
            end=end,
            limit=_MAX_EVENTS,
            time_candidates=("created_at",),
        ):
            trace_id = _identifier(row, "decision_trace_id", "trace_id")
            if not trace_id or trace_id in grouped or trace_id not in symbol_trace_ids:
                continue
            risk = _serialize_risk(row)
            decisions.append(
                {
                    "id": trace_id,
                    "trace_id": trace_id,
                    "source": "v2_partial",
                    "symbol": risk["symbol"] or symbol,
                    "time": risk["created_at"],
                    "status": "recorded_without_timeline",
                    "stage": "risk_decision_recorded",
                    "event_count": 1,
                    "reason_codes": [*risk["triggered_limits"], *risk["rejection_reasons"]],
                }
            )

    if source in {"all", "legacy"}:
        ai_decision = db_models.AiDecision
        statement = select(ai_decision).where(ai_decision.symbol == symbol)
        if start is not None:
            statement = statement.where(ai_decision.created_at >= start)
        if end is not None:
            statement = statement.where(ai_decision.created_at <= end)
        rows = _safe_rows(session, statement.order_by(desc(ai_decision.created_at)).limit(safe_limit))
        decisions.extend(_legacy_decision_payload(row) for row in rows)

    decisions.sort(key=lambda item: item.get("time") or "", reverse=True)
    return {
        "symbol": symbol,
        "start": _dt(start),
        "end": _dt(end),
        "decisions": decisions[:safe_limit],
    }


def _legacy_replay(session: Session, decision_id: int) -> dict[str, Any]:
    decision = _safe_get(session, db_models.AiDecision, decision_id)
    if decision is None:
        raise HTTPException(status_code=404, detail="Legacy decision not found")
    events: list[dict[str, Any]] = []
    feature = _safe_get(session, db_models.Feature, decision.feature_id) if decision.feature_id else None
    if feature:
        events.append(
            {
                "stage": "feature_snapshot_recorded",
                "time": _dt(feature.as_of),
                "status": "recorded",
                "reason_codes": [],
                "data": {
                    "feature_id": feature.id,
                    "schema_version": feature.schema_version,
                    "symbol": feature.symbol,
                },
            }
        )
    events.append(
        {
            "stage": "legacy_decision_recorded",
            "time": _dt(decision.created_at),
            "status": decision.execution_status,
            "reason_codes": [decision.reason or decision.execution_message] if (decision.reason or decision.execution_message) else [],
            "data": _legacy_decision_payload(decision),
        }
    )
    trade = _safe_get(session, db_models.PaperTrade, decision.trade_id) if decision.trade_id else None
    if trade:
        events.append(
            {
                "stage": "paper_fill_recorded",
                "time": _dt(trade.created_at),
                "status": trade.status,
                "reason_codes": [trade.reason] if trade.reason else [],
                "data": _legacy_trade_marker(trade, decision.id),
            }
        )
    experiences = _safe_rows(
        session,
        select(db_models.ExperienceRecord)
        .where(db_models.ExperienceRecord.ai_decision_id == decision.id)
        .order_by(db_models.ExperienceRecord.created_at.asc())
        .limit(_MAX_ROWS),
    )
    for experience in experiences:
        events.append(
            {
                "stage": "experience_recorded",
                "time": _dt(experience.created_at),
                "status": "recorded",
                "reason_codes": [],
                "data": {
                    "experience_id": experience.id,
                    "reward": experience.reward,
                    "result": _json_safe(_mapping(experience.result)),
                },
            }
        )
    events.sort(key=lambda item: item.get("time") or "")
    return {
        "trace_id": f"legacy:{decision.id}",
        "source": "legacy",
        "partial": True,
        "decision": _legacy_decision_payload(decision),
        "events": events,
        "missing_stages": [
            "external_ai_request",
            "narrow_model_predictions",
            "trading_signals",
            "ensemble_decision",
            "portfolio_target",
            "independent_risk_decision",
            "simulated_order",
        ],
        "note": "Legacy rows are replayed only from recorded links; unavailable stages are not inferred.",
    }


def _timeline_event_payload(row: Any) -> dict[str, Any]:
    occurred_at = _value(row, "occurred_at", "created_at")
    return {
        "id": _identifier(row, "timeline_event_id", "id"),
        "stage": _value(row, "stage", "event_type") or "recorded_event",
        "time": _dt(occurred_at),
        "status": _value(row, "status"),
        "reason_codes": _json_safe(_list(_value(row, "reason_codes"))),
        "data": _json_safe(_mapping(_value(row, "payload", "metadata"))),
    }


@router.get("/replay/{trace_id}")
def vision_replay(trace_id: str, session: Session = Depends(get_session)) -> dict[str, Any]:
    """Return a recorded decision timeline; never generate narrative steps."""
    trace_id = _normal_identifier(trace_id, field="trace_id") or ""
    if trace_id.startswith("legacy:"):
        try:
            decision_id = int(trace_id.partition(":")[2])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid legacy trace id") from exc
        return _legacy_replay(session, decision_id)

    timeline_rows = _query_v2_trace(session, "DecisionTimelineEvent", trace_id, limit=_MAX_ROWS)
    associated = {
        "predictions": [_serialize_prediction(row) for row in _query_v2_trace(session, "ModelPredictionRecord", trace_id)],
        "signals": [_serialize_signal(row) for row in _query_v2_trace(session, "TradingSignalRecord", trace_id)],
        "ensembles": [_serialize_ensemble(row) for row in _query_v2_trace(session, "EnsembleDecisionRecord", trace_id)],
        "portfolio_targets": [_serialize_target(row) for row in _query_v2_trace(session, "PortfolioTargetRecord", trace_id)],
        "risk_decisions": [_serialize_risk(row) for row in _query_v2_trace(session, "RiskDecisionRecord", trace_id)],
        "orders": [_serialize_order(row) for row in _query_v2_trace(session, "SimulatedOrderRecord", trace_id)],
        "fills": [_serialize_fill(row) for row in _query_v2_trace(session, "SimulatedFillRecord", trace_id)],
    }
    if not timeline_rows and not any(associated.values()):
        raise HTTPException(status_code=404, detail="Decision trace not found")

    events = [_timeline_event_payload(row) for row in timeline_rows]
    # A trace saved before the timeline writer is still useful.  These entries
    # deliberately say "recorded" rather than pretending a stage executed.
    if not events:
        stage_map = {
            "predictions": "model_prediction_recorded",
            "signals": "trading_signal_recorded",
            "ensembles": "ensemble_decision_recorded",
            "portfolio_targets": "portfolio_target_recorded",
            "risk_decisions": "risk_decision_recorded",
            "orders": "simulated_order_recorded",
            "fills": "simulated_fill_recorded",
        }
        for key, rows in associated.items():
            for row in rows:
                timestamp = row.get("generated_at") or row.get("created_at") or row.get("submitted_at") or row.get("filled_at")
                events.append(
                    {
                        "id": row.get("id"),
                        "stage": stage_map[key],
                        "time": timestamp,
                        "status": "recorded_without_timeline",
                        "reason_codes": row.get("reason_codes") or [],
                        "data": {"record_id": row.get("id"), "derived_from_record": True},
                    }
                )
    events.sort(key=lambda item: item.get("time") or "")
    return {
        "trace_id": trace_id,
        "source": "v2",
        "partial": not bool(timeline_rows),
        "events": events,
        "records": associated,
        "note": (
            "Timeline rows were not yet recorded; entries were derived from persisted V2 records."
            if not timeline_rows
            else None
        ),
    }


def _research_record(row: Any, *, record_type: str) -> dict[str, Any]:
    created_at = _value(
        row,
        "created_at",
        "generated_at",
        "observed_at",
        "evaluated_at",
        "active_from",
        "requested_at",
    )
    return {
        "record_type": record_type,
        "id": _identifier(
            row,
            "id",
            "candidate_id",
            "evaluation_id",
            "promotion_decision_id",
            "assignment_id",
            "paper_sandbox_account_id",
        ),
        "symbol": _value(row, "symbol"),
        "account_id": _value(row, "paper_account_id", "account_id"),
        "candidate_id": _identifier(row, "candidate_id"),
        "name": _value(row, "name"),
        "model_id": _value(row, "model_id"),
        "model_version": _value(row, "model_version", "version"),
        "model_family": _value(row, "model_family", "signal_family"),
        "lifecycle_status": _value(row, "lifecycle_status", "lifecycle_state", "status", "action"),
        "health_status": _value(row, "health_status"),
        "approved": _value(row, "approved"),
        "reason": _value(row, "reason", "notes"),
        "created_at": _dt(created_at),
        "metrics": _json_safe(_mapping(_value(row, "metrics"))),
        "reason_codes": _json_safe(_list(_value(row, "reason_codes", "rejection_reasons"))),
    }


@router.get("/research")
def vision_research(
    symbol: str | None = None,
    limit: int = _DEFAULT_LIMIT,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Return bounded registry and health records for the research panel."""
    normalized_symbol = _normal_symbol(symbol) if symbol else None
    safe_limit = _bounded(limit, default=_DEFAULT_LIMIT, maximum=_MAX_ROWS)
    entity_names = {
        "champion_assignments": "ChampionAssignment",
        "strategy_candidates": "StrategyCandidate",
        "candidate_evaluations": "CandidateEvaluation",
        "model_health": "ModelHealthSnapshot",
        "signal_health": "SignalHealthSnapshot",
        "promotion_decisions": "PromotionDecision",
        "sandbox_accounts": "PaperSandboxAccount",
    }
    output: dict[str, list[dict[str, Any]]] = {}
    for key, model_name in entity_names.items():
        rows = _query_v2(
            session,
            model_name,
            symbol=normalized_symbol,
            limit=safe_limit,
            time_candidates=("created_at", "observed_at", "evaluated_at"),
        )
        output[key] = [_research_record(row, record_type=key) for row in rows]
    return {
        "symbol": normalized_symbol,
        **output,
        "availability": {
            "v2_registry_tables_present": any(output.values()),
            "note": "Empty sections mean no compatible persisted V2 records are available yet.",
        },
    }
