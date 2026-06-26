from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import PaperTrade, Position


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


class RiskManager:
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
    ) -> RiskResult:
        normalized_action = action.upper()
        intent, side = self._intent(normalized_action, existing_position)
        has_ai_plan = requested_leverage is not None or requested_margin_pct is not None

        if intent == "hold":
            return RiskResult(True, "Hold action does not create market exposure.", 0.0, intent=intent)

        if intent == "unsupported":
            return RiskResult(False, f"Unsupported action: {normalized_action}", 0.0, intent=intent)

        if intent == "missing_position":
            return RiskResult(False, "No open paper futures position exists to close.", 0.0, intent=intent)

        if intent == "close":
            assert existing_position is not None
            return RiskResult(
                True,
                f"{normalized_action} reduces existing {existing_position.side.upper()} paper futures exposure.",
                existing_position.quantity * existing_position.current_price,
                margin_required=0.0,
                leverage=existing_position.leverage or self._paper_leverage(confidence),
                allocation_pct=0.0,
                intent=intent,
                side=existing_position.side.upper(),
            )

        if has_ai_plan:
            return self._evaluate_ai_plan(
                intent=intent,
                side=side,
                cash_balance=cash_balance,
                equity=equity,
                requested_notional=requested_notional,
                requested_leverage=requested_leverage,
                requested_margin_pct=requested_margin_pct,
            )

        if confidence < settings.risk_min_confidence:
            return RiskResult(False, "Signal confidence is below the configured minimum.", 0.0, intent=intent, side=side)

        daily_pnl = self._daily_realized_pnl()
        max_daily_loss = settings.paper_start_balance * settings.risk_max_daily_loss_pct
        if daily_pnl <= -max_daily_loss:
            return RiskResult(False, "Max daily loss limit reached.", 0.0, intent=intent, side=side)

        if self._cooldown_active(equity):
            return RiskResult(False, "Cooldown is active after a large recent loss.", 0.0, intent=intent, side=side)

        open_positions = self.session.scalar(select(func.count(Position.id)).where(Position.status == "OPEN")) or 0
        if existing_position is None and open_positions >= settings.risk_max_open_positions:
            return RiskResult(False, "Max open position limit reached.", 0.0, intent=intent, side=side)

        leverage = self._paper_leverage(confidence)
        allocation_pct = self._confidence_allocation(confidence)
        if allocation_pct <= 0:
            return RiskResult(False, "Signal confidence did not earn a positive position size.", 0.0, intent=intent, side=side)

        max_margin = min(
            cash_balance / (1.0 + settings.paper_fee_rate * leverage),
            equity * allocation_pct,
        )
        max_notional = max_margin * leverage
        if requested_notional is not None:
            max_notional = min(max_notional, requested_notional)
            max_margin = max_notional / leverage
        if settings.risk_max_entry_fee_pct_of_equity > 0 and settings.paper_fee_rate > 0:
            max_entry_fee = equity * settings.risk_max_entry_fee_pct_of_equity
            fee_capped_notional = max_entry_fee / settings.paper_fee_rate
            if max_notional > fee_capped_notional:
                max_notional = fee_capped_notional
                max_margin = max_notional / leverage
        if max_notional <= 0:
            return RiskResult(False, "No cash is available for a new paper futures trade.", 0.0, intent=intent, side=side)

        return RiskResult(
            True,
            (
                f"Risk checks passed for {side} paper futures. "
                f"Confidence-sized leverage {leverage:g}x, margin allocation {allocation_pct:.1%}."
            ),
            max_notional,
            margin_required=max_margin,
            leverage=leverage,
            allocation_pct=allocation_pct,
            intent=intent,
            side=side,
        )

    def _evaluate_ai_plan(
        self,
        *,
        intent: str,
        side: str | None,
        cash_balance: float,
        equity: float,
        requested_notional: float | None,
        requested_leverage: float | None,
        requested_margin_pct: float | None,
    ) -> RiskResult:
        leverage = self._planned_leverage(requested_leverage)
        if leverage is None:
            return RiskResult(False, "AI trade plan rejected: leverage must be finite and positive.", 0.0, intent=intent, side=side)
        if leverage > settings.paper_max_leverage:
            return RiskResult(
                False,
                f"AI trade plan rejected: leverage {leverage:g}x is above max executable {settings.paper_max_leverage:g}x.",
                0.0,
                intent=intent,
                side=side,
            )
        allocation_pct = self._planned_margin_pct(requested_margin_pct)
        if allocation_pct is None:
            return RiskResult(False, "AI trade plan rejected: margin_pct must be finite and positive.", 0.0, intent=intent, side=side)
        if allocation_pct > 1.0:
            return RiskResult(False, "AI trade plan rejected: margin_pct cannot exceed 100% of equity.", 0.0, intent=intent, side=side)
        max_margin = min(
            cash_balance / (1.0 + settings.paper_fee_rate * leverage),
            equity * allocation_pct,
        )
        max_notional = max_margin * leverage
        if requested_notional is not None:
            max_notional = min(max_notional, requested_notional)
            max_margin = max_notional / leverage
        if max_notional <= 0:
            return RiskResult(False, "AI trade plan rejected: no cash is available for execution.", 0.0, intent=intent, side=side)
        return RiskResult(
            True,
            (
                f"AI trade plan accepted for {side} paper futures: "
                f"{allocation_pct:.2%} margin at {leverage:g}x."
            ),
            max_notional,
            margin_required=max_margin,
            leverage=leverage,
            allocation_pct=allocation_pct,
            intent=intent,
            side=side,
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

        current_side = existing_position.side.upper()
        if requested_side == current_side:
            return "increase", requested_side
        return "close", current_side

    def _paper_leverage(self, confidence: float) -> float:
        max_allowed = max(settings.paper_max_leverage, 1.0)
        if not settings.paper_confidence_leverage_enabled:
            return min(max(settings.paper_leverage, 1.0), max_allowed)
        min_allowed = min(max(settings.paper_min_leverage, 1.0), max_allowed)
        if confidence <= settings.risk_min_confidence:
            return min_allowed
        span = max(1.0 - settings.risk_min_confidence, 1e-9)
        confidence_scale = max(0.0, min(1.0, (confidence - settings.risk_min_confidence) / span))
        return min_allowed + (max_allowed - min_allowed) * confidence_scale

    def _confidence_allocation(self, confidence: float) -> float:
        if confidence <= settings.risk_min_confidence:
            return 0.0
        span = max(1.0 - settings.risk_min_confidence, 1e-9)
        confidence_scale = max(0.0, min(1.0, (confidence - settings.risk_min_confidence) / span))
        return max(0.0, min(settings.risk_max_trade_size_pct, settings.risk_max_trade_size_pct * confidence_scale))

    def _planned_leverage(self, value: float | None) -> float | None:
        try:
            leverage = float(value if value is not None else settings.paper_leverage)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(leverage) or leverage <= 0:
            return None
        return leverage

    def _planned_margin_pct(self, value: float | None) -> float | None:
        try:
            margin_pct = float(value if value is not None else settings.risk_max_trade_size_pct)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(margin_pct) or margin_pct <= 0:
            return None
        return margin_pct

    def _daily_realized_pnl(self) -> float:
        now = datetime.now(timezone.utc)
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return float(
            self.session.scalar(
                select(func.coalesce(func.sum(PaperTrade.realized_pnl), 0.0)).where(PaperTrade.created_at >= start)
            )
            or 0.0
        )

    def _cooldown_active(self, equity: float) -> bool:
        if settings.risk_cooldown_minutes <= 0:
            return False
        since = datetime.now(timezone.utc) - timedelta(minutes=max(settings.risk_cooldown_minutes, 0))
        latest_loss = self.session.scalar(
            select(PaperTrade)
            .where(PaperTrade.created_at >= since, PaperTrade.realized_pnl < 0)
            .order_by(desc(PaperTrade.created_at))
            .limit(1)
        )
        if latest_loss is None:
            return False
        large_loss_threshold = max(equity * settings.risk_max_daily_loss_pct * 0.5, 1.0)
        return abs(latest_loss.realized_pnl) >= large_loss_threshold
