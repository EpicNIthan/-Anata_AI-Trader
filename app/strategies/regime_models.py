from __future__ import annotations

"""Additive persistence models for regime_pullback_v1.

These tables intentionally do not replace or delete legacy tables. They provide a
strict idempotent ledger while compatibility rows may still be mirrored to the old
dashboard tables.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index, Integer, JSON, Numeric, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.models import Base

MONEY = Numeric(38, 18)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RegimeDecisionRecord(Base):
    __tablename__ = "regime_pullback_decisions"
    __table_args__ = (
        UniqueConstraint("strategy_version", "symbol", "candle_close_time", name="uq_regime_decision_candle"),
        Index("ix_regime_decision_symbol_time", "symbol", "candle_close_time"),
        Index("ix_regime_decision_action_time", "action", "decision_time"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    strategy_name: Mapped[str] = mapped_column(String(128), index=True)
    strategy_version: Mapped[str] = mapped_column(String(128), index=True)
    feature_schema_version: Mapped[str] = mapped_column(String(128), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    candle_close_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    market_data_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    feature_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    decision_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    source_timestamps: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    data_fresh: Mapped[bool] = mapped_column(default=False)
    data_complete: Mapped[bool] = mapped_column(default=False)
    missing_feature_flags: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    regime: Mapped[str] = mapped_column(String(16), index=True)
    action: Mapped[str] = mapped_column(String(32), index=True)
    confidence: Mapped[float] = mapped_column(default=0.0)
    confidence_components: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    indicator_values: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    spread_bps: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)
    spread_estimated: Mapped[bool] = mapped_column(default=True)
    spread_method: Mapped[str | None] = mapped_column(String(128), nullable=True)
    fee_rate: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0.0004"))
    slippage_rate: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0.0002"))
    funding_rate: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)
    open_interest_change: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)
    taker_imbalance: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)
    rejection_codes: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    risk_result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    requested_quantity: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"))
    approved_quantity: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"))
    risk_budget: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"))
    stop_distance: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"))
    estimated_notional: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"))
    entry_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    exit_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    paper_order_id: Mapped[str | None] = mapped_column(String(96), nullable=True, index=True)
    position_id: Mapped[int | None] = mapped_column(ForeignKey("positions.id"), nullable=True, index=True)
    shadow_decision: Mapped[bool] = mapped_column(default=False, index=True)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class RegimeFutureLabelRecord(Base):
    __tablename__ = "regime_pullback_future_labels"
    __table_args__ = (
        UniqueConstraint("decision_id", "horizon_minutes", name="uq_regime_label_horizon"),
        Index("ix_regime_label_available", "available_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    decision_id: Mapped[int] = mapped_column(ForeignKey("regime_pullback_decisions.id"), index=True)
    horizon_minutes: Mapped[int] = mapped_column(Integer)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    calculated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    future_return: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)
    future_return_after_costs: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)
    maximum_favorable_excursion: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)
    maximum_adverse_excursion: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)
    stop_reached_first: Mapped[bool | None] = mapped_column(nullable=True)
    target_reached_first: Mapped[bool | None] = mapped_column(nullable=True)
    time_to_stop_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    time_to_target_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    decision_regime: Mapped[str] = mapped_column(String(16))
    data_complete: Mapped[bool] = mapped_column(default=False)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class RegimePaperAccount(Base):
    __tablename__ = "regime_pullback_accounts"

    account_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    cash_balance: Mapped[Decimal] = mapped_column(MONEY)
    realized_trading_pnl: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"))
    total_fees: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"))
    funding_cost: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"))
    starting_balance: Mapped[Decimal] = mapped_column(MONEY)
    enabled: Mapped[bool] = mapped_column(default=True)
    administrative_shutdown: Mapped[bool] = mapped_column(default=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class RegimePaperOrder(Base):
    __tablename__ = "regime_pullback_orders"
    __table_args__ = (
        UniqueConstraint("client_order_id", name="uq_regime_client_order"),
        Index("ix_regime_order_symbol_time", "symbol", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    client_order_id: Mapped[str] = mapped_column(String(96), index=True)
    decision_id: Mapped[int | None] = mapped_column(ForeignKey("regime_pullback_decisions.id"), nullable=True, index=True)
    position_id: Mapped[int | None] = mapped_column(ForeignKey("positions.id"), nullable=True, index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    order_side: Mapped[str] = mapped_column(String(8))
    position_side: Mapped[str] = mapped_column(String(8))
    quantity: Mapped[Decimal] = mapped_column(MONEY)
    reference_price: Mapped[Decimal] = mapped_column(MONEY)
    fill_price: Mapped[Decimal] = mapped_column(MONEY)
    notional: Mapped[Decimal] = mapped_column(MONEY)
    fee: Mapped[Decimal] = mapped_column(MONEY)
    slippage_cost: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"))
    spread_cost: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"))
    funding_cost: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"))
    realized_trading_pnl: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"))
    is_entry: Mapped[bool] = mapped_column(default=True)
    status: Mapped[str] = mapped_column(String(32), default="FILLED")
    reason_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)


class RegimePaperFill(Base):
    __tablename__ = "regime_pullback_fills"
    __table_args__ = (UniqueConstraint("fill_id", name="uq_regime_fill_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fill_id: Mapped[str] = mapped_column(String(96), index=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("regime_pullback_orders.id"), index=True)
    quantity: Mapped[Decimal] = mapped_column(MONEY)
    fill_price: Mapped[Decimal] = mapped_column(MONEY)
    fee: Mapped[Decimal] = mapped_column(MONEY)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class RegimePositionMeta(Base):
    __tablename__ = "regime_pullback_position_meta"
    __table_args__ = (
        UniqueConstraint("position_id", name="uq_regime_position_meta_position"),
        UniqueConstraint("close_event_id", name="uq_regime_close_event"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    position_id: Mapped[int] = mapped_column(ForeignKey("positions.id"), index=True)
    strategy_version: Mapped[str] = mapped_column(String(128), index=True)
    entry_decision_id: Mapped[int] = mapped_column(ForeignKey("regime_pullback_decisions.id"), index=True)
    initial_atr: Mapped[Decimal] = mapped_column(MONEY)
    initial_stop: Mapped[Decimal] = mapped_column(MONEY)
    initial_target: Mapped[Decimal] = mapped_column(MONEY)
    initial_risk_per_unit: Mapped[Decimal] = mapped_column(MONEY)
    entry_fee: Mapped[Decimal] = mapped_column(MONEY)
    exit_fee: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"))
    total_costs: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"))
    highest_completed_close: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)
    lowest_completed_close: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)
    bars_held: Mapped[int] = mapped_column(Integer, default=0)
    break_even_active: Mapped[bool] = mapped_column(default=False)
    trailing_active: Mapped[bool] = mapped_column(default=False)
    cooldown_until_candle: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    stale_since: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    close_event_id: Mapped[str | None] = mapped_column(String(96), nullable=True)
    closed_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class RegimeDailyRiskState(Base):
    __tablename__ = "regime_pullback_daily_risk"
    __table_args__ = (UniqueConstraint("account_id", "utc_date", name="uq_regime_daily_risk"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[str] = mapped_column(String(128), index=True)
    utc_date: Mapped[str] = mapped_column(String(10), index=True)
    starting_equity: Mapped[Decimal] = mapped_column(MONEY)
    realized_trading_pnl: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"))
    total_fees: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"))
    new_trades: Mapped[int] = mapped_column(Integer, default=0)
    consecutive_losses: Mapped[int] = mapped_column(Integer, default=0)
    circuit_breaker: Mapped[bool] = mapped_column(default=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class RegimeSymbolRiskState(Base):
    __tablename__ = "regime_pullback_symbol_risk"
    __table_args__ = (UniqueConstraint("account_id", "symbol", "utc_date", name="uq_regime_symbol_risk"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[str] = mapped_column(String(128), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    utc_date: Mapped[str] = mapped_column(String(10), index=True)
    new_trades: Mapped[int] = mapped_column(Integer, default=0)
    cooldown_until_candle: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    stale_since: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class RegimeStrategyLock(Base):
    __tablename__ = "regime_pullback_strategy_lock"

    lock_name: Mapped[str] = mapped_column(String(128), primary_key=True)
    owner_id: Mapped[str] = mapped_column(String(128), index=True)
    lease_until: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)
