"""Independent pre-trade portfolio risk policy for the V2 paper pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import AccountEquity, PaperSandboxAccount, PaperTrade, Position, RiskControlState, RiskDecisionRecord
from app.pipeline.domain import HealthStatus, PortfolioTarget, RiskDecision, utc_now


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


# A deliberately small, deterministic baseline grouping. This is a final risk backstop,
# not a claim that assets inside a group always have stable correlations. Research may
# later supply a versioned dynamic mapping, but an upstream portfolio bug must not remove
# the independent cluster cap in the meantime.
_SYMBOL_CLUSTERS: dict[str, str] = {
    "BTCUSDT": "proof_of_work_majors",
    "LTCUSDT": "proof_of_work_majors",
    "ETHUSDT": "smart_contract_platforms",
    "SOLUSDT": "smart_contract_platforms",
    "BNBUSDT": "smart_contract_platforms",
    "ADAUSDT": "smart_contract_platforms",
    "AVAXUSDT": "smart_contract_platforms",
    "XRPUSDT": "payments",
    "DOGEUSDT": "meme_assets",
    "LINKUSDT": "oracle_infrastructure",
}


def _cluster_for_symbol(symbol: str) -> str:
    return _SYMBOL_CLUSTERS.get(symbol.upper(), symbol.upper())


@dataclass(frozen=True)
class MarketSnapshot:
    symbol: str
    price: float
    observed_at: datetime
    source: str = "feature"
    bid: float | None = None
    ask: float | None = None
    available_volume: float | None = None
    funding_rate: float | None = None


@dataclass(frozen=True)
class RiskInputs:
    account_id: str
    cash_balance: float
    equity: float
    market: MarketSnapshot
    confidence: float
    liquidity_score: float
    expected_cost: float
    expected_volatility: float
    model_health: HealthStatus = HealthStatus.HEALTHY
    signal_health: HealthStatus = HealthStatus.HEALTHY
    required_features_missing: tuple[str, ...] = ()
    current_gross_exposure: float = 0.0
    current_net_exposure: float = 0.0
    now: datetime | None = None
    simulation_healthy: bool = True
    spread_pct: float | None = None


@dataclass(frozen=True)
class RiskPolicy:
    min_confidence: float
    max_position_leverage: float
    max_margin_allocation_pct: float
    max_entry_fee_pct_of_equity: float
    max_daily_loss_pct: float
    max_portfolio_drawdown_pct: float
    max_open_positions: int
    cooldown_minutes: int
    max_market_data_age_seconds: int
    max_symbol_exposure_pct: float
    max_gross_exposure_pct: float
    max_net_exposure_pct: float
    max_expected_cost_pct: float
    min_liquidity_score: float
    kill_switch_enabled: bool
    configuration_version: str
    max_cluster_exposure_pct: float = 0.25
    max_spread_pct: float = 0.005

    @classmethod
    def from_settings(cls) -> "RiskPolicy":
        return cls(
            min_confidence=max(settings.risk_min_confidence, 0.0),
            max_position_leverage=max(min(settings.v2_max_position_leverage, settings.risk_max_portfolio_leverage, settings.paper_max_leverage), 1.0),
            max_margin_allocation_pct=max(min(settings.risk_max_trade_size_pct, settings.v2_max_symbol_exposure_pct), 0.0),
            max_entry_fee_pct_of_equity=max(min(settings.risk_max_entry_fee_pct_of_equity, settings.risk_max_fee_exposure_pct), 0.0),
            max_daily_loss_pct=max(settings.risk_max_daily_loss_pct, 0.0),
            max_portfolio_drawdown_pct=max(settings.risk_max_portfolio_drawdown_pct, 0.0),
            max_open_positions=max(settings.risk_max_open_positions, 0),
            cooldown_minutes=max(settings.risk_cooldown_minutes, 0),
            max_market_data_age_seconds=max(settings.risk_max_market_data_age_seconds, 0),
            max_symbol_exposure_pct=max(settings.v2_max_symbol_exposure_pct, 0.0),
            max_gross_exposure_pct=max(settings.v2_max_gross_exposure_pct, 0.0),
            max_net_exposure_pct=max(settings.v2_max_net_exposure_pct, 0.0),
            max_expected_cost_pct=max(settings.risk_max_expected_transaction_cost_pct, 0.0),
            min_liquidity_score=max(min(settings.v2_min_liquidity_score, 1.0), 0.0),
            kill_switch_enabled=settings.risk_kill_switch_enabled,
            configuration_version=settings.risk_configuration_version,
            max_cluster_exposure_pct=max(settings.v2_max_cluster_exposure_pct, 0.0),
            max_spread_pct=max(settings.risk_max_spread_pct, 0.0),
        )


class PortfolioRiskEngine:
    """Apply all global controls to every exposure-increasing V2 proposal.

    Model outputs have no sizing authority. This engine chooses a safe leverage and
    may resize a valid target, then records an immutable decision. It has no import
    of the paper engine, so execution cannot influence risk policy.
    """

    def __init__(self, session: Session, *, policy: RiskPolicy | None = None) -> None:
        self.session = session
        self.policy = policy or RiskPolicy.from_settings()

    def approve(self, target: PortfolioTarget, inputs: RiskInputs, *, decision_trace_id: str) -> RiskDecision:
        now = _aware(inputs.now) or utc_now()
        policy, sandbox_cap = self._policy_for_account(inputs.account_id)
        requested = float(target.requested_target_exposure)
        current = float(target.current_exposure)
        increasing = self._increases_exposure(current, requested)
        triggered: list[str] = []
        rejection: list[str] = []
        approved = requested
        leverage = policy.max_position_leverage

        if increasing:
            self._apply_global_gates(inputs, now, triggered, rejection, policy=policy)
            if sandbox_cap is not None and abs(requested) > sandbox_cap:
                triggered.append("SANDBOX_EXPOSURE_CAP")
            if abs(requested) > policy.max_symbol_exposure_pct:
                triggered.append("MAX_SYMBOL_EXPOSURE")
                approved = math.copysign(policy.max_symbol_exposure_pct, requested)
            prospective_gross = inputs.current_gross_exposure - abs(current) + abs(approved)
            if prospective_gross > policy.max_gross_exposure_pct:
                triggered.append("MAX_GROSS_EXPOSURE")
                room = max(policy.max_gross_exposure_pct - (inputs.current_gross_exposure - abs(current)), 0.0)
                approved = math.copysign(min(abs(approved), room), approved)
            prospective_net = inputs.current_net_exposure - current + approved
            if abs(prospective_net) > policy.max_net_exposure_pct:
                triggered.append("MAX_NET_EXPOSURE")
                upper = policy.max_net_exposure_pct - (inputs.current_net_exposure - current)
                lower = -policy.max_net_exposure_pct - (inputs.current_net_exposure - current)
                approved = min(max(approved, lower), upper)
            approved = self._apply_cluster_cap(
                target.symbol,
                current=current,
                approved=approved,
                inputs=inputs,
                triggered=triggered,
                policy=policy,
            )
            if abs(approved) > policy.max_margin_allocation_pct:
                triggered.append("MAX_MARGIN_ALLOCATION")
                approved = math.copysign(policy.max_margin_allocation_pct, approved)
            if inputs.expected_cost > policy.max_expected_cost_pct:
                rejection.append("MAX_EXPECTED_TRANSACTION_COST")
            if inputs.equity <= 0 or inputs.cash_balance <= 0:
                rejection.append("NO_AVAILABLE_PAPER_EQUITY")
            minimum_notional = float(getattr(settings, "min_paper_trade_notional", 0.0))
            if abs(approved - current) * max(inputs.equity, 0.0) < minimum_notional and abs(approved - current) > 1e-12:
                rejection.append("BELOW_MINIMUM_PAPER_NOTIONAL")
            approved = self._apply_fee_cap(current, approved, inputs, triggered, policy=policy)
            if abs(approved - current) <= 1e-12 and abs(requested - current) > 1e-12:
                rejection.append("NO_EXPOSURE_REMAINS_AFTER_RISK_CAPS")

        is_approved = not rejection
        if not is_approved:
            approved = 0.0 if increasing else current
        decision = RiskDecision(
            portfolio_target_id=target.portfolio_target_id,
            approved=is_approved,
            requested_exposure=requested,
            approved_exposure=approved,
            requested_leverage=policy.max_position_leverage,
            approved_leverage=leverage if is_approved else 0.0,
            triggered_limits=sorted(set(triggered)),
            rejection_reasons=sorted(set(rejection)),
            configuration_version=policy.configuration_version,
            kill_switch_state=self._kill_switch_active(),
            created_at=now,
        )
        self.persist(decision, target=target, account_id=inputs.account_id, decision_trace_id=decision_trace_id, inputs=inputs)
        return decision

    def persist(
        self,
        decision: RiskDecision,
        *,
        target: PortfolioTarget,
        account_id: str,
        decision_trace_id: str,
        inputs: RiskInputs,
    ) -> RiskDecisionRecord:
        row = RiskDecisionRecord(
            risk_decision_id=decision.risk_decision_id,
            decision_trace_id=decision_trace_id,
            portfolio_target_id=decision.portfolio_target_id,
            paper_account_id=account_id,
            approved=decision.approved,
            requested_exposure=decision.requested_exposure,
            approved_exposure=decision.approved_exposure,
            requested_leverage=decision.requested_leverage,
            approved_leverage=decision.approved_leverage,
            triggered_limits=decision.triggered_limits,
            rejection_reasons=decision.rejection_reasons,
            configuration_version=decision.configuration_version,
            kill_switch_state=decision.kill_switch_state,
            created_at=decision.created_at,
            payload={
                "symbol": target.symbol,
                "cluster": _cluster_for_symbol(target.symbol),
                "market_price": inputs.market.price,
                "market_observed_at": _aware(inputs.market.observed_at).isoformat() if _aware(inputs.market.observed_at) else None,
                "cash_balance": inputs.cash_balance,
                "equity": inputs.equity,
                "expected_cost": inputs.expected_cost,
                "spread_pct": self._spread_pct(inputs),
                "simulation_healthy": inputs.simulation_healthy,
                "model_health": inputs.model_health.value,
                "signal_health": inputs.signal_health.value,
            },
        )
        self.session.add(row)
        self.session.flush()
        return row

    def _apply_global_gates(
        self,
        inputs: RiskInputs,
        now: datetime,
        triggered: list[str],
        rejection: list[str],
        *,
        policy: RiskPolicy,
    ) -> None:
        if self._kill_switch_active(policy=policy):
            triggered.append("KILL_SWITCH")
            rejection.append("KILL_SWITCH_ACTIVE")
        if inputs.confidence < policy.min_confidence:
            triggered.append("MIN_CONFIDENCE")
            rejection.append("CONFIDENCE_BELOW_MINIMUM")
        if not inputs.simulation_healthy:
            triggered.append("SIMULATION_HEALTH")
            rejection.append("PAPER_EXECUTION_SIMULATION_UNHEALTHY")
        observed = _aware(inputs.market.observed_at)
        if inputs.market.price <= 0 or observed is None:
            triggered.append("MARKET_DATA")
            rejection.append("MISSING_MARKET_DATA")
        elif policy.max_market_data_age_seconds > 0 and (now - observed).total_seconds() > policy.max_market_data_age_seconds:
            triggered.append("STALE_DATA")
            rejection.append("STALE_MARKET_DATA")
        elif observed > now + timedelta(seconds=5):
            triggered.append("FUTURE_DATA")
            rejection.append("FUTURE_MARKET_DATA")
        spread = self._spread_pct(inputs)
        if spread is None or not math.isfinite(spread) or spread < 0:
            triggered.append("SPREAD_DATA")
            rejection.append("INVALID_SPREAD_DATA")
        elif policy.max_spread_pct > 0 and spread > policy.max_spread_pct:
            triggered.append("MAX_SPREAD")
            rejection.append("MAX_SPREAD_EXCEEDED")
        if inputs.required_features_missing:
            triggered.append("MISSING_REQUIRED_FEATURES")
            rejection.append("MISSING_REQUIRED_FEATURES")
        if inputs.model_health in {HealthStatus.SUSPENDED, HealthStatus.RETIRED}:
            triggered.append("MODEL_HEALTH")
            rejection.append("MODEL_HEALTH_REJECTED")
        if inputs.signal_health in {HealthStatus.SUSPENDED, HealthStatus.RETIRED}:
            triggered.append("SIGNAL_HEALTH")
            rejection.append("SIGNAL_HEALTH_REJECTED")
        if inputs.liquidity_score < policy.min_liquidity_score:
            triggered.append("MINIMUM_LIQUIDITY")
            rejection.append("LIQUIDITY_BELOW_MINIMUM")
        daily_loss = self._daily_realized_pnl(inputs.account_id, now)
        max_daily_loss = max(inputs.equity, 0.0) * policy.max_daily_loss_pct
        if max_daily_loss > 0 and daily_loss <= -max_daily_loss:
            triggered.append("MAX_DAILY_LOSS")
            rejection.append("MAX_DAILY_LOSS_REACHED")
        if policy.max_portfolio_drawdown_pct > 0 and self._drawdown(inputs.account_id) >= policy.max_portfolio_drawdown_pct:
            triggered.append("MAX_PORTFOLIO_DRAWDOWN")
            rejection.append("MAX_PORTFOLIO_DRAWDOWN_REACHED")
        if self._cooldown_active(inputs.account_id, inputs.equity, now, policy=policy):
            triggered.append("COOLDOWN")
            rejection.append("COOLDOWN_ACTIVE")
        open_count = self.session.scalar(
            select(func.count(Position.id)).where(Position.status == "OPEN", Position.paper_account_id == inputs.account_id)
        ) or 0
        current_symbol = self.session.scalar(
            select(Position.id).where(
                Position.status == "OPEN", Position.paper_account_id == inputs.account_id, Position.symbol == inputs.market.symbol.upper()
            ).limit(1)
        )
        if current_symbol is None and open_count >= policy.max_open_positions:
            triggered.append("MAX_OPEN_POSITIONS")
            rejection.append("MAX_OPEN_POSITIONS_REACHED")

    def _apply_cluster_cap(
        self,
        symbol: str,
        *,
        current: float,
        approved: float,
        inputs: RiskInputs,
        triggered: list[str],
        policy: RiskPolicy,
    ) -> float:
        if policy.max_cluster_exposure_pct <= 0:
            return 0.0
        cluster_without_symbol = self._cluster_exposure_without_symbol(inputs.account_id, symbol, inputs.equity)
        room = max(policy.max_cluster_exposure_pct - cluster_without_symbol, 0.0)
        if abs(approved) > room:
            triggered.append("MAX_CORRELATED_CLUSTER_EXPOSURE")
            approved = math.copysign(room, approved)
        return approved

    def _cluster_exposure_without_symbol(self, account_id: str, symbol: str, equity: float) -> float:
        if equity <= 0:
            return 0.0
        normalized = symbol.upper()
        cluster = _cluster_for_symbol(normalized)
        rows = self.session.scalars(
            select(Position).where(
                Position.paper_account_id == account_id,
                Position.status == "OPEN",
                Position.symbol != normalized,
            )
        )
        total_notional = 0.0
        for row in rows:
            if _cluster_for_symbol(row.symbol) != cluster:
                continue
            notional = float(row.notional or 0.0)
            if notional <= 0:
                price = float(row.current_price or row.entry_price or 0.0)
                notional = abs(float(row.quantity or 0.0) * price)
            if math.isfinite(notional) and notional > 0:
                total_notional += abs(notional)
        return total_notional / equity

    def _spread_pct(self, inputs: RiskInputs) -> float | None:
        if inputs.spread_pct is not None:
            try:
                return float(inputs.spread_pct)
            except (TypeError, ValueError):
                return None
        bid, ask = inputs.market.bid, inputs.market.ask
        if bid is not None and ask is not None:
            try:
                bid_value, ask_value = float(bid), float(ask)
            except (TypeError, ValueError):
                return None
            midpoint = (bid_value + ask_value) / 2.0
            if midpoint <= 0 or ask_value < bid_value:
                return None
            return (ask_value - bid_value) / midpoint
        # The simulator's configured spread is the deterministic fallback when no live
        # bid/ask snapshot is available. Missing spread is never silently treated as zero.
        try:
            return float(getattr(settings, "paper_simulated_spread_pct", 0.0))
        except (TypeError, ValueError):
            return None

    def _apply_fee_cap(
        self,
        current: float,
        approved: float,
        inputs: RiskInputs,
        triggered: list[str],
        *,
        policy: RiskPolicy,
    ) -> float:
        delta = abs(approved - current)
        notional = delta * max(inputs.equity, 0.0)
        fee_rate = max(float(getattr(settings, "paper_fee_rate", 0.0)), 0.0)
        fee = notional * fee_rate
        cap = max(inputs.equity, 0.0) * policy.max_entry_fee_pct_of_equity
        if fee <= cap or fee <= 0:
            return approved
        triggered.append("MAX_FEE_EXPOSURE")
        ratio = cap / fee if fee else 0.0
        adjusted_delta = delta * max(min(ratio, 1.0), 0.0)
        return current + math.copysign(adjusted_delta, approved - current)

    def _daily_realized_pnl(self, account_id: str, now: datetime) -> float:
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return float(
            self.session.scalar(
                select(func.coalesce(func.sum(PaperTrade.realized_pnl), 0.0)).where(
                    PaperTrade.created_at >= start, PaperTrade.paper_account_id == account_id
                )
            )
            or 0.0
        )

    def _drawdown(self, account_id: str) -> float:
        latest = self.session.scalar(
            select(AccountEquity).where(AccountEquity.paper_account_id == account_id).order_by(desc(AccountEquity.timestamp)).limit(1)
        )
        return abs(float(latest.drawdown or 0.0)) if latest else 0.0

    def _cooldown_active(self, account_id: str, equity: float, now: datetime, *, policy: RiskPolicy) -> bool:
        if policy.cooldown_minutes <= 0:
            return False
        since = now - timedelta(minutes=policy.cooldown_minutes)
        loss = self.session.scalar(
            select(PaperTrade)
            .where(PaperTrade.paper_account_id == account_id, PaperTrade.created_at >= since, PaperTrade.realized_pnl < 0)
            .order_by(desc(PaperTrade.created_at))
            .limit(1)
        )
        threshold = max(equity * policy.max_daily_loss_pct * 0.5, 1.0)
        return bool(loss and abs(loss.realized_pnl) >= threshold)

    def _kill_switch_active(self, *, policy: RiskPolicy | None = None) -> bool:
        effective_policy = policy or self.policy
        if effective_policy.kill_switch_enabled:
            return True
        state = self.session.scalar(select(RiskControlState).order_by(desc(RiskControlState.updated_at)).limit(1))
        return bool(state and state.enabled)

    def _policy_for_account(self, account_id: str) -> tuple[RiskPolicy, float | None]:
        """Apply a registered sandbox's persisted exposure ceiling independently."""
        sandbox = self.session.scalar(
            select(PaperSandboxAccount)
            .where(PaperSandboxAccount.account_id == account_id)
            .limit(1)
        )
        if sandbox is None:
            return self.policy, None
        try:
            raw_cap = float(sandbox.max_exposure_pct)
        except (TypeError, ValueError):
            raw_cap = 0.0
        cap = max(min(raw_cap, 1.0), 0.0) if math.isfinite(raw_cap) and sandbox.active else 0.0
        return (
            replace(
                self.policy,
                max_margin_allocation_pct=min(self.policy.max_margin_allocation_pct, cap),
                max_symbol_exposure_pct=min(self.policy.max_symbol_exposure_pct, cap),
                max_gross_exposure_pct=min(self.policy.max_gross_exposure_pct, cap),
                max_net_exposure_pct=min(self.policy.max_net_exposure_pct, cap),
                max_cluster_exposure_pct=min(self.policy.max_cluster_exposure_pct, cap),
            ),
            cap,
        )

    @staticmethod
    def _increases_exposure(current: float, requested: float) -> bool:
        if abs(requested) <= abs(current) and (
            current == 0
            or requested == 0
            or math.copysign(1.0, requested) == math.copysign(1.0, current)
        ):
            return False
        return abs(requested) > 1e-12
