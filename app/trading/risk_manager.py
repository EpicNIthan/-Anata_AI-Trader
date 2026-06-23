from __future__ import annotations

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
    ) -> RiskResult:
        normalized_action = action.upper()
        if normalized_action == "HOLD":
            return RiskResult(True, "Hold action does not create market exposure.", 0.0)

        if normalized_action in {"SELL", "CLOSE"}:
            if existing_position is None:
                return RiskResult(False, "No open long position exists to close; shorting is disabled.", 0.0)
            return RiskResult(True, "Close action reduces existing exposure.", existing_position.quantity * existing_position.current_price)

        if normalized_action != "BUY":
            return RiskResult(False, f"Unsupported action: {normalized_action}", 0.0)

        if confidence < settings.risk_min_confidence:
            return RiskResult(False, "Signal confidence is below the configured minimum.", 0.0)

        daily_pnl = self._daily_realized_pnl()
        max_daily_loss = settings.paper_start_balance * settings.risk_max_daily_loss_pct
        if daily_pnl <= -max_daily_loss:
            return RiskResult(False, "Max daily loss limit reached.", 0.0)

        if self._cooldown_active(equity):
            return RiskResult(False, "Cooldown is active after a large recent loss.", 0.0)

        open_positions = self.session.scalar(
            select(func.count(Position.id)).where(Position.status == "OPEN")
        ) or 0
        if existing_position is None and open_positions >= settings.risk_max_open_positions:
            return RiskResult(False, "Max open position limit reached.", 0.0)

        max_notional = min(cash_balance / (1.0 + settings.paper_fee_rate), equity * settings.risk_max_trade_size_pct)
        if requested_notional is not None:
            max_notional = min(max_notional, requested_notional)
        if max_notional <= 0:
            return RiskResult(False, "No cash is available for a new paper trade.", 0.0)

        return RiskResult(True, "Risk checks passed.", max_notional)

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
        since = datetime.now(timezone.utc) - timedelta(minutes=settings.risk_cooldown_minutes)
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

