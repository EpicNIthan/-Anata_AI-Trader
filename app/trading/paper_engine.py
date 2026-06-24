from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import AccountEquity, Candle, LiveCandleUpdate, PaperTrade, Position
from app.trading.risk_manager import RiskManager


@dataclass(frozen=True)
class ExecutionResult:
    status: str
    message: str
    trade_id: int | None = None
    balance: float | None = None
    equity: float | None = None


class PaperEngine:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.risk = RiskManager(session)

    def execute_signal(
        self,
        *,
        symbol: str,
        action: str,
        confidence: float,
        reason: str | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        price: float | None = None,
        quantity: float | None = None,
        notional: float | None = None,
    ) -> ExecutionResult:
        if not settings.is_paper_mode:
            return ExecutionResult("REJECTED", "Trading mode is not paper; real order APIs are disabled.")

        normalized_symbol = symbol.upper()
        normalized_action = action.upper()
        latest_account = self._latest_account(create=True)
        mark_price = self._resolve_price(normalized_symbol, price)
        if normalized_action != "HOLD" and mark_price is None:
            return ExecutionResult("REJECTED", "No price is available for this symbol.")

        existing_position = self._open_position(normalized_symbol)
        requested_notional = notional
        if requested_notional is None and quantity is not None and mark_price:
            requested_notional = quantity * mark_price

        risk = self.risk.evaluate(
            action=normalized_action,
            confidence=confidence,
            cash_balance=latest_account.cash_balance,
            equity=latest_account.equity,
            requested_notional=requested_notional,
            existing_position=existing_position,
        )
        if not risk.accepted:
            return ExecutionResult("REJECTED", risk.reason, balance=latest_account.cash_balance, equity=latest_account.equity)

        if normalized_action == "HOLD":
            equity_row = self._record_equity(latest_account.cash_balance, price_by_symbol={normalized_symbol: mark_price})
            self.session.commit()
            return ExecutionResult("HELD", risk.reason, balance=equity_row.cash_balance, equity=equity_row.equity)

        if normalized_action == "BUY":
            return self._buy(
                symbol=normalized_symbol,
                price=mark_price or 0.0,
                max_notional=risk.max_notional,
                margin_required=risk.margin_required,
                leverage=risk.leverage,
                reason=reason,
                stop_loss=stop_loss,
                take_profit=take_profit,
                existing_position=existing_position,
            )

        if normalized_action in {"SELL", "CLOSE"}:
            return self._close(
                symbol=normalized_symbol,
                price=mark_price or 0.0,
                reason=reason,
                requested_quantity=quantity,
                existing_position=existing_position,
            )

        return ExecutionResult("REJECTED", f"Unsupported action: {normalized_action}")

    def snapshot(self) -> dict[str, float]:
        latest = self._latest_account(create=True)
        latest = self._record_equity(latest.cash_balance)
        self.session.commit()
        return {
            "cash_balance": latest.cash_balance,
            "equity": latest.equity,
            "realized_pnl": latest.realized_pnl,
            "unrealized_pnl": latest.unrealized_pnl,
            "drawdown": latest.drawdown,
        }

    def _buy(
        self,
        *,
        symbol: str,
        price: float,
        max_notional: float,
        margin_required: float,
        leverage: float,
        reason: str | None,
        stop_loss: float | None,
        take_profit: float | None,
        existing_position: Position | None,
    ) -> ExecutionResult:
        account = self._latest_account(create=True)
        notional = max_notional
        margin_required = min(margin_required or (notional / max(leverage, 1.0)), account.cash_balance)
        quantity = notional / price
        fee = notional * settings.paper_fee_rate
        if quantity <= 0:
            return ExecutionResult("REJECTED", "Computed quantity was zero.")
        if margin_required + fee > account.cash_balance:
            return ExecutionResult("REJECTED", "Not enough paper cash for margin plus fee.")

        if existing_position is None:
            stop_loss = stop_loss if stop_loss is not None else self._default_stop_loss(price)
            take_profit = take_profit if take_profit is not None else self._default_take_profit(price)
            position = Position(
                symbol=symbol,
                side="LONG",
                quantity=quantity,
                entry_price=price,
                current_price=price,
                notional=notional,
                margin_used=margin_required,
                leverage=leverage,
                stop_loss=stop_loss,
                take_profit=take_profit,
                status="OPEN",
            )
            self.session.add(position)
        else:
            combined_quantity = existing_position.quantity + quantity
            existing_margin = existing_position.margin_used or self._fallback_margin(existing_position)
            existing_position.entry_price = (
                existing_position.entry_price * existing_position.quantity + notional
            ) / combined_quantity
            existing_position.quantity = combined_quantity
            existing_position.current_price = price
            existing_position.margin_used = existing_margin + margin_required
            existing_position.notional = combined_quantity * existing_position.entry_price
            existing_position.leverage = leverage
            existing_position.stop_loss = stop_loss or existing_position.stop_loss or self._default_stop_loss(existing_position.entry_price)
            existing_position.take_profit = take_profit or existing_position.take_profit or self._default_take_profit(existing_position.entry_price)
            position = existing_position

        cash_after = account.cash_balance - margin_required - fee
        trade = PaperTrade(
            symbol=symbol,
            action="BUY",
            side="LONG",
            quantity=quantity,
            price=price,
            notional=notional,
            fee=fee,
            realized_pnl=0.0,
            status="FILLED",
            reason=reason,
            raw_payload={
                "paper_leverage": leverage,
                "margin_required": margin_required,
                "fee_rate": settings.paper_fee_rate,
            },
        )
        self.session.add(trade)
        self.session.flush()
        equity_row = self._record_equity(cash_after, price_by_symbol={symbol: price})
        trade.balance_after = equity_row.cash_balance
        trade.equity_after = equity_row.equity
        position.current_price = price
        position.unrealized_pnl = (price - position.entry_price) * position.quantity
        self.session.commit()
        self.session.refresh(trade)
        return ExecutionResult(
            "FILLED",
            f"Paper BUY filled at {leverage:g}x using ${margin_required:,.2f} margin.",
            trade_id=trade.id,
            balance=equity_row.cash_balance,
            equity=equity_row.equity,
        )

    def _close(
        self,
        *,
        symbol: str,
        price: float,
        reason: str | None,
        requested_quantity: float | None,
        existing_position: Position | None,
    ) -> ExecutionResult:
        if existing_position is None:
            account = self._latest_account(create=True)
            return ExecutionResult("REJECTED", "No open long position exists to close.", balance=account.cash_balance, equity=account.equity)

        original_quantity = existing_position.quantity
        close_quantity = min(requested_quantity or original_quantity, original_quantity)
        proceeds = close_quantity * price
        fee = proceeds * settings.paper_fee_rate
        gross_pnl = (price - existing_position.entry_price) * close_quantity
        realized_pnl = gross_pnl - fee
        account = self._latest_account(create=True)
        margin_before = existing_position.margin_used or self._fallback_margin(existing_position)
        released_margin = margin_before * (close_quantity / original_quantity) if original_quantity else margin_before
        cash_after = account.cash_balance + released_margin + gross_pnl - fee

        existing_position.quantity -= close_quantity
        existing_position.current_price = price
        existing_position.margin_used = max(margin_before - released_margin, 0.0)
        existing_position.notional = existing_position.quantity * existing_position.entry_price
        existing_position.realized_pnl += realized_pnl
        existing_position.unrealized_pnl = (price - existing_position.entry_price) * existing_position.quantity
        if existing_position.quantity <= 1e-12:
            existing_position.quantity = 0.0
            existing_position.status = "CLOSED"
            existing_position.closed_at = datetime.now(timezone.utc)

        trade = PaperTrade(
            symbol=symbol,
            action="SELL",
            side="LONG",
            quantity=close_quantity,
            price=price,
            notional=proceeds,
            fee=fee,
            realized_pnl=realized_pnl,
            status="FILLED",
            reason=reason,
            raw_payload={
                "paper_leverage": existing_position.leverage or settings.paper_leverage,
                "released_margin": released_margin,
                "gross_pnl": gross_pnl,
                "fee_rate": settings.paper_fee_rate,
            },
        )
        self.session.add(trade)
        self.session.flush()
        equity_row = self._record_equity(cash_after, price_by_symbol={symbol: price})
        trade.balance_after = equity_row.cash_balance
        trade.equity_after = equity_row.equity
        self.session.commit()
        self.session.refresh(trade)
        return ExecutionResult(
            "FILLED",
            f"Paper SELL filled; released ${released_margin:,.2f} margin.",
            trade_id=trade.id,
            balance=equity_row.cash_balance,
            equity=equity_row.equity,
        )

    def _latest_account(self, create: bool = False) -> AccountEquity:
        latest = self.session.scalar(select(AccountEquity).order_by(desc(AccountEquity.timestamp)).limit(1))
        if latest is None and create:
            latest = AccountEquity(
                cash_balance=settings.paper_start_balance,
                equity=settings.paper_start_balance,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                drawdown=0.0,
            )
            self.session.add(latest)
            self.session.commit()
            self.session.refresh(latest)
        if latest is None:
            raise RuntimeError("Account equity has not been initialized")
        return latest

    def _record_equity(self, cash_balance: float, price_by_symbol: dict[str, float | None] | None = None) -> AccountEquity:
        price_by_symbol = price_by_symbol or {}
        reserved_margin = 0.0
        unrealized_pnl = 0.0
        open_positions = list(self.session.scalars(select(Position).where(Position.status == "OPEN")))
        for position in open_positions:
            price = price_by_symbol.get(position.symbol)
            if price is None:
                price = self._resolve_price(position.symbol, None) or position.current_price or position.entry_price
            position.current_price = price
            position.unrealized_pnl = (price - position.entry_price) * position.quantity
            margin_used = position.margin_used or self._fallback_margin(position)
            position.margin_used = margin_used
            position.notional = position.quantity * position.entry_price
            reserved_margin += margin_used
            unrealized_pnl += position.unrealized_pnl

        realized_pnl = float(self.session.scalar(select(func.coalesce(func.sum(PaperTrade.realized_pnl), 0.0))) or 0.0)
        equity = cash_balance + reserved_margin + unrealized_pnl
        peak_equity = float(self.session.scalar(select(func.max(AccountEquity.equity))) or equity)
        drawdown = (equity - peak_equity) / peak_equity if peak_equity else 0.0
        row = AccountEquity(
            cash_balance=cash_balance,
            equity=equity,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            drawdown=drawdown,
            raw={"open_positions": len(open_positions), "reserved_margin": reserved_margin, "paper_leverage": settings.paper_leverage},
        )
        self.session.add(row)
        self.session.flush()
        return row

    def _open_position(self, symbol: str) -> Position | None:
        return self.session.scalar(
            select(Position).where(Position.symbol == symbol, Position.status == "OPEN").order_by(desc(Position.opened_at)).limit(1)
        )

    def _resolve_price(self, symbol: str, explicit_price: float | None) -> float | None:
        if explicit_price and explicit_price > 0:
            return explicit_price
        candle = self.session.scalar(
            select(Candle).where(Candle.symbol == symbol).order_by(desc(Candle.open_time)).limit(1)
        )
        live_update = self.session.scalar(
            select(LiveCandleUpdate).where(LiveCandleUpdate.symbol == symbol).order_by(desc(LiveCandleUpdate.open_time)).limit(1)
        )
        if live_update and (candle is None or live_update.open_time >= candle.open_time):
            return live_update.close
        return candle.close if candle else None

    def _fallback_margin(self, position: Position) -> float:
        leverage = position.leverage or settings.paper_leverage or 1.0
        return (position.quantity * position.entry_price) / max(leverage, 1.0)

    def _default_stop_loss(self, price: float) -> float | None:
        if settings.auto_default_stop_loss_pct <= 0:
            return None
        return round(price * (1.0 - settings.auto_default_stop_loss_pct), 8)

    def _default_take_profit(self, price: float) -> float | None:
        if settings.auto_default_take_profit_pct <= 0:
            return None
        return round(price * (1.0 + settings.auto_default_take_profit_pct), 8)
