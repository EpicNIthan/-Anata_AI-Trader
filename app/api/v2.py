"""Bounded, admin-only operations for the paper-only Anata V2 pipeline.

This router deliberately exposes lifecycle and observation controls only.  It has no
exchange client, accepts no model artifact uploads, and cannot turn a paper account
into a live account.  Mount it from :mod:`app.main` as ``v2_router``.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import (
    ChampionAssignment,
    EnsembleDecisionRecord,
    ModelPredictionRecord,
    ModelVersion,
    PaperSandboxAccount,
    PromotionDecision,
    RiskControlState,
    ShadowPrediction,
)
from app.db.session import get_session
from app.pipeline.attribution import paper_pnl_attribution
from app.pipeline.domain import ModelPrediction
from app.pipeline.registry import ModelRegistry
from app.pipeline.service import V2PipelineService
from app.pipeline.monitoring import RollingHealthMonitor
from app.security import require_admin


router = APIRouter(prefix="/api/v2", tags=["anata-v2-operations"], dependencies=[Depends(require_admin)])

_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9._:-]{3,32}$")
_FAMILY_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


class _StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class PipelineRunRequest(_StrictRequest):
    symbol: str = Field(min_length=3, max_length=32)
    # The champion account is the default.  A caller may select only an active,
    # registered sandbox account; arbitrary account names are intentionally refused.
    paper_account_id: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("symbol")
    @classmethod
    def normalise_symbol(cls, value: str) -> str:
        value = value.upper()
        if not _SYMBOL_PATTERN.fullmatch(value):
            raise ValueError("symbol must contain only uppercase letters, digits, '.', '_', ':' or '-'")
        return value


class PromotionRequest(_StrictRequest):
    model_family: str | None = Field(default=None, min_length=1, max_length=128)
    symbol_scope: str = Field(default="*", min_length=1, max_length=64)
    reason: str = Field(min_length=3, max_length=500)
    confirm: bool = False

    @field_validator("model_family")
    @classmethod
    def validate_family(cls, value: str | None) -> str | None:
        if value is not None and not _FAMILY_PATTERN.fullmatch(value):
            raise ValueError("model_family contains unsupported characters")
        return value

    @field_validator("symbol_scope")
    @classmethod
    def validate_scope(cls, value: str) -> str:
        value = value.upper()
        if value != "*" and not _SYMBOL_PATTERN.fullmatch(value):
            raise ValueError("symbol_scope must be '*' or a bounded symbol")
        return value


class RollbackRequest(_StrictRequest):
    model_family: str = Field(min_length=1, max_length=128)
    symbol_scope: str = Field(default="*", min_length=1, max_length=64)
    reason: str = Field(min_length=3, max_length=500)
    confirm: bool = False

    @field_validator("model_family")
    @classmethod
    def validate_family(cls, value: str) -> str:
        if not _FAMILY_PATTERN.fullmatch(value):
            raise ValueError("model_family contains unsupported characters")
        return value

    @field_validator("symbol_scope")
    @classmethod
    def validate_scope(cls, value: str) -> str:
        value = value.upper()
        if value != "*" and not _SYMBOL_PATTERN.fullmatch(value):
            raise ValueError("symbol_scope must be '*' or a bounded symbol")
        return value


class SandboxRequest(_StrictRequest):
    name: str | None = Field(default=None, min_length=1, max_length=128)
    starting_balance: float | None = Field(default=None, gt=0.0, le=1_000_000.0)
    confirm: bool = False


class ShadowRecordRequest(_StrictRequest):
    model_version_id: int = Field(gt=0)
    prediction_id: str = Field(min_length=8, max_length=64)
    decision_trace_id: str = Field(min_length=8, max_length=64)


class ShadowStartRequest(_StrictRequest):
    reason: str = Field(min_length=3, max_length=500)
    confirm: bool = False


class KillSwitchRequest(_StrictRequest):
    enabled: bool
    reason: str = Field(min_length=3, max_length=500)
    confirm: bool = False


def _paper_only() -> None:
    """Refuse administrative actions when the application is not in paper mode."""
    if not settings.is_paper_mode:
        raise HTTPException(
            status_code=409,
            detail="Anata V2 operations are disabled outside TRADING_MODE=paper; no live execution is supported.",
        )


def _require_confirmation(confirmed: bool, action: str) -> None:
    if not confirmed:
        raise HTTPException(status_code=409, detail=f"Explicit confirm=true is required to {action}.")


def _aware(value: datetime | None) -> datetime:
    """Normalise SQLite's occasionally-naive timestamps for strict V2 schemas."""
    if value is None:
        return datetime.now(timezone.utc)
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _model_payload(row: ModelVersion) -> dict[str, Any]:
    """Return registry metadata without exposing artifact paths or loading artifacts."""
    return {
        "id": row.id,
        "model_id": row.model_id,
        "name": row.name,
        "version": row.version,
        "model_family": row.model_family,
        "status": row.status,
        "lifecycle_state": row.lifecycle_state,
        "health_status": row.health_status,
        "feature_schema_version": row.feature_schema_version,
        "feature_columns": row.feature_columns or [],
        "artifact_checksum": row.artifact_checksum,
        "preprocessing_version": row.preprocessing_version,
        "training_dataset_version": row.training_dataset_version,
        "forecast_horizon_seconds": row.forecast_horizon_seconds,
        "metrics": row.metrics or {},
        "created_at": _aware(row.created_at),
    }


def _sandbox_payload(row: PaperSandboxAccount) -> dict[str, Any]:
    return {
        "id": row.id,
        "paper_account_id": row.account_id,
        "name": row.name,
        "model_version_id": row.model_version_id,
        "starting_balance": row.starting_balance,
        "max_exposure_pct": row.max_exposure_pct,
        "active": row.active,
        "created_at": _aware(row.created_at),
        "closed_at": _aware(row.closed_at) if row.closed_at else None,
        "paper_only": True,
    }


def _promotion_payload(row: PromotionDecision) -> dict[str, Any]:
    return {
        "id": row.id,
        "model_version_id": row.model_version_id,
        "previous_model_version_id": row.previous_model_version_id,
        "action": row.action,
        "approved": row.approved,
        "decided_by": row.decided_by,
        "reason": row.reason,
        "created_at": _aware(row.created_at),
    }


def _active_account_id(session: Session, requested: str | None) -> str:
    account_id = (requested or settings.v2_champion_account_id).strip()
    if account_id == settings.v2_champion_account_id:
        return account_id
    sandbox = session.scalar(
        select(PaperSandboxAccount).where(PaperSandboxAccount.account_id == account_id).order_by(desc(PaperSandboxAccount.created_at)).limit(1)
    )
    if sandbox is None:
        raise HTTPException(status_code=404, detail="paper_account_id is not the champion account or a registered sandbox")
    if not sandbox.active:
        raise HTTPException(status_code=409, detail="paper_account_id refers to a closed sandbox")
    return sandbox.account_id


def _prediction_from_record(row: ModelPredictionRecord) -> ModelPrediction:
    """Restore a persisted forecast without accepting a client-supplied prediction."""
    return ModelPrediction(
        prediction_id=row.prediction_id,
        model_id=row.model_id,
        model_version=row.model_version,
        model_family=row.model_family,
        symbol=row.symbol,
        generated_at=_aware(row.generated_at),
        valid_from=_aware(row.valid_from),
        expires_at=_aware(row.expires_at),
        forecast_horizon_seconds=row.forecast_horizon_seconds,
        expected_return=row.expected_return,
        expected_volatility=row.expected_volatility,
        probability_up=row.probability_up,
        probability_down=row.probability_down,
        confidence=row.confidence,
        calibration_score=row.calibration_score,
        uncertainty=row.uncertainty,
        regime=row.regime or "unknown",
        feature_schema_version=row.feature_schema_version,
        feature_snapshot_id=row.feature_snapshot_id,
        data_version=row.data_version or "operational",
        external_context_available=bool(row.external_context_available),
        metadata=row.payload or {},
    )


@router.post("/pipeline/run")
def run_pipeline(
    payload: PipelineRunRequest,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Run exactly one audited V2 decision through paper execution controls."""
    _paper_only()
    account_id = _active_account_id(session, payload.paper_account_id)
    try:
        result = V2PipelineService(session).run_symbol(payload.symbol, account_id=account_id)
    except ValueError as exc:
        session.rollback()
        raise HTTPException(status_code=409, detail=f"V2 pipeline could not build a paper decision: {exc}") from exc
    return {
        "paper_only": True,
        "paper_account_id": account_id,
        "automatic_promotion_enabled": settings.v2_auto_promote_champion,
        "result": result.as_dict(),
    }


@router.get("/models")
def list_models(
    model_family: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """List registered model metadata only; artifacts are neither returned nor loaded."""
    _paper_only()
    statement = select(ModelVersion)
    if model_family:
        if not _FAMILY_PATTERN.fullmatch(model_family):
            raise HTTPException(status_code=422, detail="model_family contains unsupported characters")
        statement = statement.where(ModelVersion.model_family == model_family)
    rows = session.scalars(statement.order_by(desc(ModelVersion.created_at)).limit(limit)).all()
    return {
        "paper_only": True,
        "automatic_promotion_enabled": settings.v2_auto_promote_champion,
        "items": [_model_payload(row) for row in rows],
    }


@router.get("/registry")
def registry_state(
    limit: int = Query(default=50, ge=1, le=100),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Provide a bounded snapshot of registry, champion, sandbox and audit state."""
    _paper_only()
    models = session.scalars(select(ModelVersion).order_by(desc(ModelVersion.created_at)).limit(limit)).all()
    assignments = session.scalars(
        select(ChampionAssignment).where(ChampionAssignment.active_to.is_(None)).order_by(desc(ChampionAssignment.active_from)).limit(limit)
    ).all()
    model_by_id = {row.id: row for row in models}
    missing_champion_ids = {row.model_version_id for row in assignments} - set(model_by_id)
    if missing_champion_ids:
        model_by_id.update(
            {
                row.id: row
                for row in session.scalars(select(ModelVersion).where(ModelVersion.id.in_(missing_champion_ids))).all()
            }
        )
    sandboxes = session.scalars(select(PaperSandboxAccount).order_by(desc(PaperSandboxAccount.created_at)).limit(limit)).all()
    decisions = session.scalars(select(PromotionDecision).order_by(desc(PromotionDecision.created_at)).limit(limit)).all()
    return {
        "paper_only": True,
        "automatic_promotion_enabled": settings.v2_auto_promote_champion,
        "models": [_model_payload(row) for row in models],
        "active_champions": [
            {
                "assignment_id": row.id,
                "model_version_id": row.model_version_id,
                "model": _model_payload(model_by_id[row.model_version_id]) if row.model_version_id in model_by_id else None,
                "model_family": row.model_family,
                "symbol_scope": row.symbol_scope,
                "active_from": _aware(row.active_from),
                "assigned_by": row.assigned_by,
                "reason": row.reason,
            }
            for row in assignments
        ],
        "sandboxes": [_sandbox_payload(row) for row in sandboxes],
        "recent_promotion_decisions": [_promotion_payload(row) for row in decisions],
    }


@router.post("/models/{model_version_id}/promote")
def promote_model(
    model_version_id: int,
    payload: PromotionRequest,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Explicitly, manually promote a compatible model and write a rollback audit row."""
    _paper_only()
    _require_confirmation(payload.confirm, "promote a model")
    candidate = session.get(ModelVersion, model_version_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Model version not found")
    family = payload.model_family or candidate.model_family
    if not family:
        raise HTTPException(status_code=422, detail="model_family is required because this candidate has none")
    if candidate.model_family and family != candidate.model_family:
        raise HTTPException(status_code=409, detail="model_family must match the registered candidate family")
    try:
        # ``manual`` is intentional: ModelRegistry refuses non-manual promotion while
        # automatic promotion is disabled.  Admin authentication and the confirmation
        # above are the endpoint's explicit manual-authorisation controls.
        promoted = ModelRegistry(session).promote(
            model_version_id,
            model_family=family,
            symbol_scope=payload.symbol_scope,
            actor="manual",
            reason=payload.reason,
        )
        session.commit()
    except (PermissionError, ValueError) as exc:
        session.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "paper_only": True,
        "manual_promotion": True,
        "automatic_promotion_enabled": settings.v2_auto_promote_champion,
        "model": _model_payload(promoted),
        "message": "Model promoted manually; automatic promotion remains governed by V2_AUTO_PROMOTE_CHAMPION.",
    }


@router.post("/registry/rollback")
def rollback_model(
    payload: RollbackRequest,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Restore the prior champion for a family/scope using persisted promotion history."""
    _paper_only()
    _require_confirmation(payload.confirm, "roll back a champion")
    try:
        restored = ModelRegistry(session).rollback(
            model_family=payload.model_family,
            symbol_scope=payload.symbol_scope,
            actor="manual",
            reason=payload.reason,
        )
        session.commit()
    except ValueError as exc:
        session.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "paper_only": True,
        "manual_rollback": True,
        "model": _model_payload(restored),
        "message": "Prior champion restored from the immutable promotion decision history.",
    }


@router.post("/models/{model_version_id}/sandbox")
def start_sandbox(
    model_version_id: int,
    payload: SandboxRequest,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Start an isolated paper sandbox after compatibility checks, never a live account."""
    _paper_only()
    _require_confirmation(payload.confirm, "start a paper sandbox")
    try:
        sandbox = ModelRegistry(session).start_sandbox(
            model_version_id,
            name=payload.name,
            starting_balance=payload.starting_balance,
        )
        session.commit()
    except ValueError as exc:
        session.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "paper_only": True,
        "sandbox": _sandbox_payload(sandbox),
        "message": "Sandbox created after technical compatibility checks only; it is not a profitability approval.",
    }


@router.post("/shadow")
def record_shadow_prediction(
    payload: ShadowRecordRequest,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Attach a persisted model forecast to its candidate model's shadow ledger."""
    _paper_only()
    model = session.get(ModelVersion, payload.model_version_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model version not found")
    prediction_row = session.scalar(
        select(ModelPredictionRecord)
        .where(
            ModelPredictionRecord.prediction_id == payload.prediction_id,
            ModelPredictionRecord.decision_trace_id == payload.decision_trace_id,
        )
        .limit(1)
    )
    if prediction_row is None:
        raise HTTPException(status_code=404, detail="Persisted prediction was not found for this decision trace")
    if model.model_id != prediction_row.model_id or model.version != prediction_row.model_version:
        raise HTTPException(
            status_code=409,
            detail="Shadow records must reference a forecast generated by the selected registered model version.",
        )
    existing = session.scalar(
        select(ShadowPrediction)
        .where(
            ShadowPrediction.model_version_id == model.id,
            ShadowPrediction.prediction_id == prediction_row.prediction_id,
            ShadowPrediction.decision_trace_id == payload.decision_trace_id,
        )
        .limit(1)
    )
    if existing is not None:
        raise HTTPException(status_code=409, detail="This shadow prediction is already recorded")
    try:
        row = ModelRegistry(session).record_shadow(
            _prediction_from_record(prediction_row),
            model_version_id=model.id,
            decision_trace_id=payload.decision_trace_id,
        )
        session.commit()
    except ValueError as exc:
        session.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "paper_only": True,
        "shadow_prediction": {
            "id": row.id,
            "model_version_id": row.model_version_id,
            "prediction_id": row.prediction_id,
            "decision_trace_id": row.decision_trace_id,
            "symbol": row.symbol,
            "generated_at": _aware(row.generated_at),
        },
        "message": "Shadow observation recorded without changing ensemble, portfolio, risk or execution state.",
    }


@router.post("/models/{model_version_id}/shadow")
def start_shadow_model(
    model_version_id: int,
    payload: ShadowStartRequest,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Enable live non-executing predictions for a technically valid challenger."""
    _paper_only()
    _require_confirmation(payload.confirm, "start a shadow model")
    try:
        model = ModelRegistry(session).start_shadow(model_version_id)
        history = list(model.promotion_history or [])
        history.append(
            {
                "action": "START_SHADOW",
                "reason": payload.reason,
                "decided_by": "api-admin",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        model.promotion_history = history
        session.commit()
    except ValueError as exc:
        session.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "paper_only": True,
        "execution_allowed": False,
        "model": _model_payload(model),
        "message": "Shadow inference enabled; predictions are recorded but excluded from portfolio exposure.",
    }


@router.get("/shadow")
def list_shadow_predictions(
    model_version_id: int | None = Query(default=None, gt=0),
    symbol: str | None = Query(default=None, min_length=3, max_length=32),
    limit: int = Query(default=50, ge=1, le=200),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """List bounded shadow observations with their recorded trace context."""
    _paper_only()
    if symbol is not None:
        symbol = symbol.upper()
        if not _SYMBOL_PATTERN.fullmatch(symbol):
            raise HTTPException(status_code=422, detail="symbol contains unsupported characters")
    statement = select(ShadowPrediction)
    if model_version_id is not None:
        statement = statement.where(ShadowPrediction.model_version_id == model_version_id)
    if symbol is not None:
        statement = statement.where(ShadowPrediction.symbol == symbol)
    rows = session.scalars(statement.order_by(desc(ShadowPrediction.generated_at)).limit(limit)).all()
    model_ids = {row.model_version_id for row in rows}
    models = {row.id: row for row in session.scalars(select(ModelVersion).where(ModelVersion.id.in_(model_ids))).all()} if model_ids else {}
    ensemble_by_trace: dict[str, EnsembleDecisionRecord] = {}
    trace_ids = {row.decision_trace_id for row in rows}
    if trace_ids:
        for ensemble in session.scalars(
            select(EnsembleDecisionRecord)
            .where(EnsembleDecisionRecord.decision_trace_id.in_(trace_ids))
            .order_by(desc(EnsembleDecisionRecord.generated_at))
        ):
            ensemble_by_trace.setdefault(ensemble.decision_trace_id, ensemble)
    return {
        "paper_only": True,
        "items": [
            {
                "id": row.id,
                "model_version_id": row.model_version_id,
                "model": _model_payload(models[row.model_version_id]) if row.model_version_id in models else None,
                "prediction_id": row.prediction_id,
                "decision_trace_id": row.decision_trace_id,
                "symbol": row.symbol,
                "generated_at": _aware(row.generated_at),
                "reference_ensemble": (
                    {
                        "ensemble_decision_id": ensemble_by_trace[row.decision_trace_id].ensemble_decision_id,
                        "status": ensemble_by_trace[row.decision_trace_id].decision_status,
                        "combined_expected_return": ensemble_by_trace[row.decision_trace_id].combined_expected_return,
                        "combined_confidence": ensemble_by_trace[row.decision_trace_id].combined_confidence,
                    }
                    if row.decision_trace_id in ensemble_by_trace
                    else None
                ),
            }
            for row in rows
        ],
    }


@router.get("/risk/kill-switch")
def get_kill_switch(session: Session = Depends(get_session)) -> dict[str, Any]:
    """Read the effective global paper-risk kill switch without allowing a bypass."""
    _paper_only()
    state = session.scalar(select(RiskControlState).order_by(desc(RiskControlState.updated_at)).limit(1))
    persisted_enabled = bool(state and state.enabled)
    return {
        "paper_only": True,
        "scope": "global-paper-risk",
        "effective_enabled": bool(settings.risk_kill_switch_enabled or persisted_enabled),
        "configuration_lock_enabled": bool(settings.risk_kill_switch_enabled),
        "persisted_state": (
            {
                "enabled": persisted_enabled,
                "reason": state.reason,
                "updated_by": state.updated_by,
                "updated_at": _aware(state.updated_at),
            }
            if state
            else None
        ),
        "note": "A configuration-level kill switch cannot be disabled through this API; protective closes remain governed by risk policy.",
    }


@router.put("/risk/kill-switch")
def set_kill_switch(
    payload: KillSwitchRequest,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Append an auditable global paper-risk kill-switch state transition."""
    _paper_only()
    _require_confirmation(payload.confirm, "change the global paper-risk kill switch")
    # Append rather than overwrite so risk-state changes remain reviewable even though
    # the risk engine reads only the most recent row.
    state = RiskControlState(enabled=payload.enabled, reason=payload.reason, updated_by="api-admin")
    session.add(state)
    session.commit()
    session.refresh(state)
    return {
        "paper_only": True,
        "scope": "global-paper-risk",
        "persisted_enabled": bool(state.enabled),
        "effective_enabled": bool(settings.risk_kill_switch_enabled or state.enabled),
        "configuration_lock_enabled": bool(settings.risk_kill_switch_enabled),
        "reason": state.reason,
        "updated_by": state.updated_by,
        "updated_at": _aware(state.updated_at),
        "message": (
            "Kill switch state recorded."
            if not (settings.risk_kill_switch_enabled and not state.enabled)
            else "Disable request recorded, but configuration keeps the kill switch effectively enabled."
        ),
    }


@router.post("/monitoring/run")
def run_monitoring(
    symbol: str = Query(min_length=3, max_length=32),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Label matured paper forecasts and write bounded rolling health snapshots."""
    _paper_only()
    normalized = symbol.upper()
    if not _SYMBOL_PATTERN.fullmatch(normalized):
        raise HTTPException(status_code=422, detail="symbol contains unsupported characters")
    result = RollingHealthMonitor(session).update_symbol(normalized)
    session.commit()
    return {"paper_only": True, "symbol": normalized, **result}


@router.get("/attribution")
def get_paper_attribution(
    symbol: str | None = Query(default=None, min_length=3, max_length=32),
    paper_account_id: str | None = Query(default=None, min_length=1, max_length=128),
    start: datetime | None = Query(default=None),
    end: datetime | None = Query(default=None),
    time_period: str = Query(default="day", pattern="^(hour|day|week|month)$"),
    limit: int = Query(default=2_000, ge=1, le=10_000),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Return trace-based paper-PnL attribution without inventing counterfactual terms."""
    _paper_only()
    normalized = symbol.upper() if symbol else None
    if normalized and not _SYMBOL_PATTERN.fullmatch(normalized):
        raise HTTPException(status_code=422, detail="symbol contains unsupported characters")
    if start and end and _aware(end) < _aware(start):
        raise HTTPException(status_code=422, detail="end must be greater than or equal to start")
    return paper_pnl_attribution(
        session,
        symbol=normalized,
        account_id=paper_account_id,
        start=start,
        end=end,
        limit=limit,
        time_period=time_period,
    )
