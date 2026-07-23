"""Typed, execution-independent domain objects for Anata V2.

These schemas are intentionally strict at the model boundary.  A model prediction is
only a forecast and evidence: it cannot carry leverage, margin, notional, order action,
or a paper-engine instruction.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> datetime:
    """Return an aware UTC timestamp."""
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    """Generate a readable globally unique identifier for persisted traces."""
    return f"{prefix}_{uuid4().hex}"


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class ModelLifecycle(str, Enum):
    DRAFT = "DRAFT"
    TRAINED = "TRAINED"
    VALIDATING = "VALIDATING"
    SHADOW = "SHADOW"
    PAPER_SANDBOX = "PAPER_SANDBOX"
    CHAMPION = "CHAMPION"
    DEGRADED = "DEGRADED"
    SUSPENDED = "SUSPENDED"
    RETIRED = "RETIRED"


class SignalLifecycle(str, Enum):
    RESEARCH = "RESEARCH"
    VALIDATION = "VALIDATION"
    SHADOW = "SHADOW"
    PAPER = "PAPER"
    LIMITED = "LIMITED"
    PRODUCTION = "PRODUCTION"
    REDUCED = "REDUCED"
    SUSPENDED = "SUSPENDED"
    RETIRED = "RETIRED"


class HealthStatus(str, Enum):
    HEALTHY = "HEALTHY"
    WATCH = "WATCH"
    DEGRADED = "DEGRADED"
    SUSPENDED = "SUSPENDED"
    RETIRED = "RETIRED"


class EnsembleStatus(str, Enum):
    ACTIONABLE = "ACTIONABLE"
    NEUTRAL = "NEUTRAL"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class OrderState(str, Enum):
    CREATED = "CREATED"
    RISK_APPROVED = "RISK_APPROVED"
    SUBMITTED = "SUBMITTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCEL_PENDING = "CANCEL_PENDING"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    ERROR = "ERROR"


_TERMINAL_ORDER_STATES = {OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED, OrderState.EXPIRED, OrderState.ERROR}
_ORDER_TRANSITIONS: dict[OrderState, set[OrderState]] = {
    OrderState.CREATED: {OrderState.RISK_APPROVED, OrderState.REJECTED, OrderState.ERROR},
    OrderState.RISK_APPROVED: {OrderState.SUBMITTED, OrderState.REJECTED, OrderState.ERROR},
    OrderState.SUBMITTED: {OrderState.ACKNOWLEDGED, OrderState.REJECTED, OrderState.ERROR, OrderState.EXPIRED},
    OrderState.ACKNOWLEDGED: {OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.CANCEL_PENDING, OrderState.REJECTED, OrderState.ERROR},
    OrderState.PARTIALLY_FILLED: {OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.CANCEL_PENDING, OrderState.ERROR},
    OrderState.CANCEL_PENDING: {OrderState.CANCELLED, OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.ERROR},
}


class _StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, use_enum_values=False)

    @field_validator("symbol", check_fields=False)
    @classmethod
    def _normalise_symbol(cls, value: str) -> str:
        value = value.strip().upper()
        if not value:
            raise ValueError("symbol must not be empty")
        return value

    @field_validator("generated_at", "valid_from", "expires_at", "valid_until", "created_at", check_fields=False)
    @classmethod
    def _utc_datetime(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware UTC values")
        return value.astimezone(timezone.utc)

    @field_validator(
        "expected_return",
        "expected_volatility",
        "confidence",
        "calibration_score",
        "uncertainty",
        "strength",
        "expected_cost",
        "net_expected_return",
        "liquidity_score",
        "combined_expected_return",
        "combined_expected_volatility",
        "combined_uncertainty",
        "combined_confidence",
        "correlation_penalty",
        "transaction_cost_penalty",
        "regime_penalty",
        "external_context_adjustment",
        "current_exposure",
        "requested_target_exposure",
        "requested_delta",
        "expected_risk",
        "risk_contribution",
        "urgency",
        "requested_exposure",
        "approved_exposure",
        "requested_leverage",
        "approved_leverage",
        check_fields=False,
    )
    @classmethod
    def _finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("numeric values must be finite")
        return float(value)


class ModelPrediction(_StrictSchema):
    """A standardized forecast from one frozen narrow model.

    The forbidden-extra configuration is a safety control: execution-related attributes
    such as ``leverage``, ``margin_pct``, ``notional`` or ``action`` cannot silently
    cross the model boundary.
    """

    prediction_id: str = Field(default_factory=lambda: new_id("pred"))
    model_id: str
    model_version: str
    model_family: str
    symbol: str
    generated_at: datetime = Field(default_factory=utc_now)
    valid_from: datetime = Field(default_factory=utc_now)
    expires_at: datetime = Field(default_factory=lambda: utc_now() + timedelta(minutes=5))
    forecast_horizon_seconds: int = Field(gt=0)
    expected_return: float
    expected_volatility: float = Field(ge=0.0)
    probability_up: float = Field(ge=0.0, le=1.0)
    probability_down: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    calibration_score: float = Field(default=0.5, ge=0.0, le=1.0)
    uncertainty: float = Field(default=0.5, ge=0.0, le=1.0)
    regime: str = "unknown"
    feature_schema_version: str
    feature_snapshot_id: str
    data_version: str = "operational"
    external_context_available: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _valid_window_and_probabilities(self) -> "ModelPrediction":
        if self.expires_at <= self.valid_from:
            raise ValueError("expires_at must be after valid_from")
        probability_total = self.probability_up + self.probability_down
        if probability_total > 1.000001:
            raise ValueError("probability_up + probability_down must not exceed 1")
        return self

    def is_valid_at(self, value: datetime | None = None) -> bool:
        value = value or utc_now()
        return self.valid_from <= value <= self.expires_at


class TradingSignal(_StrictSchema):
    """Registered tradable evidence derived from a prediction, not an order."""

    signal_id: str = Field(default_factory=lambda: new_id("sig"))
    prediction_id: str
    signal_family: str
    symbol: str
    generated_at: datetime = Field(default_factory=utc_now)
    valid_until: datetime = Field(default_factory=lambda: utc_now() + timedelta(minutes=5))
    direction: Direction
    strength: float = Field(ge=0.0, le=1.0)
    expected_return: float
    expected_cost: float = Field(ge=0.0)
    net_expected_return: float
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    regime: str = "unknown"
    liquidity_score: float = Field(default=0.5, ge=0.0, le=1.0)
    health_status: HealthStatus = HealthStatus.HEALTHY
    lifecycle_status: SignalLifecycle = SignalLifecycle.PAPER
    reason_codes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _valid_window(self) -> "TradingSignal":
        if self.valid_until <= self.generated_at:
            raise ValueError("valid_until must be after generated_at")
        return self

    def is_valid_at(self, value: datetime | None = None) -> bool:
        return (value or utc_now()) <= self.valid_until


class EnsembleDecision(_StrictSchema):
    """A regime-aware combination of signals, still without execution instructions."""

    ensemble_decision_id: str = Field(default_factory=lambda: new_id("ens"))
    symbol: str
    generated_at: datetime = Field(default_factory=utc_now)
    valid_until: datetime = Field(default_factory=lambda: utc_now() + timedelta(minutes=5))
    combined_expected_return: float
    combined_expected_volatility: float = Field(ge=0.0)
    combined_uncertainty: float = Field(ge=0.0, le=1.0)
    combined_confidence: float = Field(ge=0.0, le=1.0)
    current_regime: str = "unknown"
    supporting_signals: list[str] = Field(default_factory=list)
    conflicting_signals: list[str] = Field(default_factory=list)
    signal_weights: dict[str, float] = Field(default_factory=dict)
    correlation_penalty: float = Field(default=0.0, ge=0.0, le=1.0)
    transaction_cost_penalty: float = Field(default=0.0, ge=0.0)
    regime_penalty: float = Field(default=0.0, ge=0.0, le=1.0)
    external_context_adjustment: float = 0.0
    decision_status: EnsembleStatus = EnsembleStatus.NEUTRAL
    reason_codes: list[str] = Field(default_factory=list)


class PortfolioTarget(_StrictSchema):
    """A target signed exposure fraction, never a broker or paper order."""

    portfolio_target_id: str = Field(default_factory=lambda: new_id("target"))
    symbol: str
    current_exposure: float
    requested_target_exposure: float
    requested_delta: float
    expected_return: float
    expected_risk: float = Field(ge=0.0)
    risk_contribution: float = Field(ge=0.0)
    urgency: float = Field(ge=0.0, le=1.0)
    source_ensemble_decision_id: str
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _delta_is_consistent(self) -> "PortfolioTarget":
        if not math.isclose(
            self.requested_delta,
            self.requested_target_exposure - self.current_exposure,
            rel_tol=1e-7,
            abs_tol=1e-9,
        ):
            raise ValueError("requested_delta must equal target exposure minus current exposure")
        return self


class RiskDecision(_StrictSchema):
    """An independent, immutable approval or rejection of a portfolio target."""

    risk_decision_id: str = Field(default_factory=lambda: new_id("risk"))
    portfolio_target_id: str
    approved: bool
    requested_exposure: float
    approved_exposure: float
    requested_leverage: float = Field(ge=0.0)
    approved_leverage: float = Field(ge=0.0)
    triggered_limits: list[str] = Field(default_factory=list)
    rejection_reasons: list[str] = Field(default_factory=list)
    configuration_version: str = "v2"
    kill_switch_state: bool = False
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _approval_consistency(self) -> "RiskDecision":
        if not self.approved and self.approved_exposure != 0:
            raise ValueError("rejected decisions must have zero approved_exposure")
        if self.approved and self.rejection_reasons:
            raise ValueError("approved decisions cannot include rejection reasons")
        return self


class SimulatedFill(_StrictSchema):
    """A paper-only execution fill recorded against an approved simulated order."""

    fill_id: str = Field(default_factory=lambda: new_id("fill"))
    order_id: str
    symbol: str
    side: Direction
    quantity: float = Field(gt=0.0)
    price: float = Field(gt=0.0)
    notional: float = Field(gt=0.0)
    fee: float = Field(default=0.0, ge=0.0)
    slippage: float = 0.0
    funding: float = 0.0
    filled_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SimulatedOrder(_StrictSchema):
    """A paper-only order state machine requiring a traceable risk approval."""

    order_id: str = Field(default_factory=lambda: new_id("order"))
    risk_decision_id: str
    portfolio_target_id: str
    symbol: str
    side: Direction
    order_type: str = "MARKET"
    requested_quantity: float = Field(gt=0.0)
    requested_notional: float = Field(gt=0.0)
    limit_price: float | None = Field(default=None, gt=0.0)
    state: OrderState = OrderState.CREATED
    client_order_id: str = Field(default_factory=lambda: new_id("client"))
    account_id: str = "champion"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def can_transition_to(self, state: OrderState) -> bool:
        if self.state in _TERMINAL_ORDER_STATES:
            return False
        return state in _ORDER_TRANSITIONS.get(self.state, set())

    def transition(self, state: OrderState, *, when: datetime | None = None) -> "SimulatedOrder":
        """Validate and apply a state transition in memory before persistence."""
        if not self.can_transition_to(state):
            raise ValueError(f"invalid simulated order transition {self.state} -> {state}")
        self.state = state
        self.updated_at = when or utc_now()
        return self


class FeatureSnapshot(_StrictSchema):
    """A point-in-time feature payload passed to a narrow model."""

    feature_snapshot_id: str = Field(default_factory=lambda: new_id("feature"))
    symbol: str
    as_of: datetime
    available_to_model_time: datetime
    schema_version: str
    values: dict[str, Any]
    source_freshness_seconds: dict[str, float] = Field(default_factory=dict)
    missing_required_features: list[str] = Field(default_factory=list)
    data_version: str = "operational"
    external_context: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _availability_is_point_in_time(self) -> "FeatureSnapshot":
        if self.available_to_model_time.tzinfo is None or self.as_of.tzinfo is None:
            raise ValueError("feature snapshot timestamps must be timezone-aware")
        if self.available_to_model_time.astimezone(timezone.utc) > self.as_of.astimezone(timezone.utc):
            raise ValueError("feature snapshot cannot contain data that became available after its as_of time")
        return self
