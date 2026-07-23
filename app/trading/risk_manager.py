"""Legacy-compatible paper risk controls with non-bypassable global gates.

V2 calls the richer portfolio risk engine before execution.  This class remains the
compatibility adapter used by existing API endpoints and the legacy paper engine.  It
never treats model-provided leverage or margin as authority: those values are merely
untrusted hints and all exposure-increasing requests pass the same controls.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import AccountEquity, PaperTrade, Position, RiskControlState


@dataclass(frozen=True)
class RiskResult:
    accepted: bool
    reason: str
    max_notional: float
    margin_required: float = 0.0
    leverage: float = 1.0
    allocation_pct: float = 0.0
    intent: str = "none"
    side: str | None = None
    triggered_limits: tuple[str, ...] = field(default_factory=tuple)
    rejection_reasons: tuple[str, ...] = field(default_factory=tuple)


class RiskManager:
    """Apply global limits to every new or increased paper exposure."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def evaluate(
        self,
        *,
        action: str,
        confidence: float,
        cash_balance: float,
        equity: float,
        requested_notional: float | None,
        existing_position: Position | None,
        requested_leverage: float | None = None,
        requested_margin_pct: float | None = None,
        paper_data_collection_exploration: bool = False,
        market_observed_at: datetime | None = None,
        paper_account_id: str = "champion",
        kill_switch: bool | None = None,
    ) -> RiskResult:
        """Evaluate an exposure proposal.

        ``requested_leverage`` and ``requested_margin_pct`` are retained only so old
        payloads can be read. They are validated and logged through reason codes but
        never select executable risk parameters or bypass global checks.
        """
        normalized_action = action.upper()
        intent, side = self._intent(normalized_action, existing_position)
        triggered: list[str] = []
        rejection: list[str] = []

        if intent == "hold":
            return RiskResult(True, "Hold action does not create market exposure.", 0.0, intent=intent)
        if intent == "unsupported":
            return self._reject(f"Unsupported action: {normalized_action}", intent=intent, side=side, rejection=("UNSUPPORTED_ACTION",))
        if intent == "missing_position":
            return self._reject("No open paper futures position exists to close.", intent=intent, side=side, rejection=("MISSING_POSITION",))
        # Protective closes are deliberately allowed even when a kill switch is active.
        if intent == "close":
            assert existing_position is not None
            return RiskResult(
                True,
                f"{normalized_action} reduces existing {existing_position.side.upper()} paper futures exposure.",
                existing_position.quantity * existing_position.current_price,
                leverage=existing_position.leverage or 1.0,
                intent=intent,
                side=existing_position.side.upper(),
            )

        # Every OPEN/INCREASE source, including uploaded model plans and exploration,
        # traverses this exact universal gate sequence.
        if confidence < settings.risk_min_confidence:
            triggered.append("MIN_CONFIDENCE")
            rejection.append("CONFIDENCE_BELOW_MINIMUM")
        if self._kill_switch_active(kill_switch):
            triggered.append("KILL_SWITCH")
            rejection.append("KILL_SWITCH_ACTIVE")
        if settings.risk_require_fresh_data:
            freshness_reason = self._market_freshness_reason(market_observed_at)
            if freshness_reason:
                triggered.append("STALE_DATA")
                rejection.append(freshness_reason)
        daily_pnl = self._daily_realized_pnl(paper_account_id)
        max_daily_loss = max(equity, 0.0) * settings.risk_max_daily_loss_pct
        if max_daily_loss > 0 and daily_pnl <= -max_daily_loss:
            triggered.append("MAX_DAILY_LOSS")
            rejection.append("MAX_DAILY_LOSS_REACHED")
        if self._drawdown_reached(paper_account_id):
            triggered.append("MAX_PORTFOLIO_DRAWDOWN")
            rejection.append("MAX_PORTFOLIO_DRAWDOWN_REACHED")
        if self._cooldown_active(equity, paper_account_id):
            triggered.append("COOLDOWN")
            rejection.append("COOLDOWN_ACTIVE")
        open_positions = self.session.scalar(
            select(func.count(Position.id)).where(Position.status == "OPEN", Position.paper_account_id == paper_account_id)
        ) or 0
        if existing_position is None and open_positions >= settings.risk_max_open_positions:
            triggered.append("MAX_OPEN_POSITIONS")
            rejection.append("MAX_OPEN_POSITIONS_REACHED")

        planned_leverage = self._planned_leverage(requested_leverage)
        if requested_leverage is not None:
            if planned_leverage is None:
                triggered.append("MAX_LEVERAGE")
                rejection.append("INVALID_REQUESTED_LEVERAGE")
            elif planned_leverage > self._max_approved_leverage():
                triggered.append("MAX_LEVERAGE")
                rejection.append("REQUESTED_LEVERAGE_EXCEEDS_MAXIMUM")
        requested_margin = self._planned_margin_pct(requested_margin_pct)
        if requested_margin_pct is not None:
            if requested_margin is None:
                triggered.append("MAX_MARGIN_ALLOCATION")
                rejection.append("INVALID_REQUESTED_MARGIN")
            elif requested_margin > settings.risk_max_trade_size_pct:
                # Resizing is permitted, but it is explicit and cannot bypass the cap.
                triggered.append("MAX_MARGIN_ALLOCATION")

        if rejection:
            return self._reject(
                "Risk checks rejected this exposure increase: " + ", ".join(sorted(set(rejection))).replace("_", " ").lower() + ".",
                intent=intent,
                side=side,
                triggered=triggered,
                rejection=rejection,
            )

        leverage = self._paper_leverage(confidence)
        allocation_pct = self._confidence_allocation(confidence)
        # Model request cannot raise allocation. It may only lower it after all global
        # gates, preserving deterministic conservative sizing.
        if requested_margin is not None:
            allocation_pct = min(allocation_pct, requested_margin, settings.risk_max_trade_size_pct)
        allocation_pct = min(allocation_pct, settings.risk_max_trade_size_pct)
        if allocation_pct <= 0:
            return self._reject("Signal confidence did not earn a positive position size.", intent=intent, side=side, rejection=("NO_POSITIVE_SIZE",))
        max_margin = min(cash_balance / (1.0 + settings.paper_fee_rate * leverage), equity * allocation_pct)
        max_notional = max_margin * leverage
        if requested_notional is not None:
            max_notional = min(max_notional, max(float(requested_notional), 0.0))
            max_margin = max_notional / leverage if leverage else 0.0
        max_notional, max_margin, fee_capped = self._apply_fee_cap(max_notional, leverage, equity)
        if fee_capped:
            triggered.append("MAX_FEE_EXPOSURE")
        if max_notional <= 0:
            return self._reject("No cash is available for a new paper futures trade.", intent=intent, side=side, triggered=triggered, rejection=("NO_AVAILABLE_CASH",))
        source_note = "; untrusted legacy plan hints were capped by independent risk policy" if requested_leverage is not None or requested_margin_pct is not None else ""
        if paper_data_collection_exploration:
            source_note += "; exploration used the same global risk controls"
        return RiskResult(
            True,
            f"Risk checks passed for {side} paper futures. Leverage {leverage:g}x, margin allocation {allocation_pct:.1%}{source_note}.",
            max_notional,
            margin_required=max_margin,
            leverage=leverage,
            allocation_pct=allocation_pct,
            intent=intent,
            side=side,
            triggered_limits=tuple(sorted(set(triggered))),
        )

    @staticmethod
    def _reject(
        reason: str,
        *,
        intent: str,
        side: str | None,
        triggered: tuple[str, ...] | list[str] = (),
        rejection: tuple[str, ...] | list[str] = (),
    ) -> RiskResult:
        return RiskResult(
            False,
            reason,
            0.0,
            intent=intent,
            side=side,
            triggered_limits=tuple(sorted(set(triggered))),
            rejection_reasons=tuple(sorted(set(rejection))),
        )

    def _intent(self, action: str, existing_position: Position | None) -> tuple[str, str | None]:
        if action == "HOLD":
            return "hold", None
        if action not in {"BUY", "SELL", "CLOSE"}:
            return "unsupported", None
        if action == "CLOSE":
            return ("close", existing_position.side.upper()) if existing_position else ("missing_position", None)
        requested_side = "LONG" if action == "BUY" else "SHORT"
        if existing_position is None:
            return "open", requested_side
        if requested_side == existing_position.side.upper():
            return "increase", requested_side
        return "close", existing_position.side.upper()

    def _max_approved_leverage(self) -> float:
        return max(min(settings.paper_max_leverage, settings.risk_max_portfolio_leverage, settings.v2_max_position_leverage), 1.0)

    def _paper_leverage(self, confidence: float) -> float:
        max_allowed = self._max_approved_leverage()
        if not settings.paper_confidence_leverage_enabled:
            return min(max(settings.paper_leverage, 1.0), max_allowed)
        min_allowed = min(max(settings.paper_min_leverage, 1.0), max_allowed)
        span = max(1.0 - settings.risk_min_confidence, 1e-9)
        scale = max(0.0, min(1.0, (confidence - settings.risk_min_confidence) / span))
        return min_allowed + (max_allowed - min_allowed) * scale

    def _confidence_allocation(self, confidence: float) -> float:
        if confidence <= settings.risk_min_confidence:
            return 0.0
        span = max(1.0 - settings.risk_min_confidence, 1e-9)
        scale = max(0.0, min(1.0, (confidence - settings.risk_min_confidence) / span))
        return max(0.0, min(settings.risk_max_trade_size_pct, settings.risk_max_trade_size_pct * scale))

    @staticmethod
    def _planned_leverage(value: float | None) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) and result > 0 else None

    @staticmethod
    def _planned_margin_pct(value: float | None) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) and result > 0 else None

    def _apply_fee_cap(self, notional: float, leverage: float, equity: float) -> tuple[float, float, bool]:
        margin = notional / leverage if leverage else 0.0
        fee_cap_pct = min(settings.risk_max_entry_fee_pct_of_equity, settings.risk_max_fee_exposure_pct)
        if fee_cap_pct <= 0 or settings.paper_fee_rate <= 0:
            return notional, margin, False
        fee_cap = max(equity, 0.0) * fee_cap_pct
        capped_notional = min(notional, fee_cap / settings.paper_fee_rate)
        return capped_notional, capped_notional / leverage if leverage else 0.0, capped_notional < notional

    def _market_freshness_reason(self, market_observed_at: datetime | None) -> str | None:
        if market_observed_at is None:
            return "MISSING_MARKET_DATA_TIMESTAMP"
        value = market_observed_at.replace(tzinfo=timezone.utc) if market_observed_at.tzinfo is None else market_observed_at.astimezone(timezone.utc)
        now = datetime.now(timezone.utc)
        if value > now + timedelta(seconds=5):
            return "FUTURE_MARKET_DATA"
        if settings.risk_max_market_data_age_seconds > 0 and (now - value).total_seconds() > settings.risk_max_market_data_age_seconds:
            return "STALE_MARKET_DATA"
        return None

    def _daily_realized_pnl(self, paper_account_id: str) -> float:
        now = datetime.now(timezone.utc)
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return float(
            self.session.scalar(
                select(func.coalesce(func.sum(PaperTrade.realized_pnl), 0.0)).where(
                    PaperTrade.created_at >= start, PaperTrade.paper_account_id == paper_account_id
                )
            )
            or 0.0
        )

    def _drawdown_reached(self, paper_account_id: str) -> bool:
        if settings.risk_max_portfolio_drawdown_pct <= 0:
            return False
        latest = self.session.scalar(
            select(AccountEquity).where(AccountEquity.paper_account_id == paper_account_id).order_by(desc(AccountEquity.timestamp)).limit(1)
        )
        # Accept both the conventional positive drawdown now written by PaperEngine
        # and historical rows that stored the same loss as a negative return.
        return bool(latest and abs(float(latest.drawdown or 0.0)) >= settings.risk_max_portfolio_drawdown_pct)

    def _cooldown_active(self, equity: float, paper_account_id: str) -> bool:
        if settings.risk_cooldown_minutes <= 0:
            return False
        since = datetime.now(timezone.utc) - timedelta(minutes=settings.risk_cooldown_minutes)
        latest_loss = self.session.scalar(
            select(PaperTrade)
            .where(
                PaperTrade.paper_account_id == paper_account_id,
                PaperTrade.created_at >= since,
                PaperTrade.realized_pnl < 0,
            )
            .order_by(desc(PaperTrade.created_at))
            .limit(1)
        )
        threshold = max(equity * settings.risk_max_daily_loss_pct * 0.5, 1.0)
        return bool(latest_loss and abs(latest_loss.realized_pnl) >= threshold)

    def _kill_switch_active(self, override: bool | None) -> bool:
        if override is True or settings.risk_kill_switch_enabled:
            return True
        try:
            state = self.session.scalar(select(RiskControlState).order_by(desc(RiskControlState.updated_at)).limit(1))
        except Exception:  # Legacy databases upgrading before the V2 table exists.
            return False
        return bool(state and state.enabled)
