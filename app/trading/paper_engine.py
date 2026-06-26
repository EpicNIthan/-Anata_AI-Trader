from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import AccountEquity, Candle, LiveCandleUpdate, PaperTrade, Position
from app.trading.risk_manager import RiskManager

BROKER_TAKER_FEE_RATE = 0.0004
DUST_QUANTITY = 1e-8
DUST_NOTIONAL = 0.01


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

        if risk.intent == "close":
            return self._close(
                symbol=normalized_symbol,
                price=mark_price or 0.0,
                reason=reason,
                requested_quantity=quantity,
                existing_position=existing_position,
                requested_action=normalized_action,
            )

        if normalized_action in {"BUY", "SELL"}:
            return self._open_or_add(
                symbol=normalized_symbol,
                side="LONG" if normalized_action == "BUY" else "SHORT",
                action=normalized_action,
                price=mark_price or 0.0,
                max_notional=risk.max_notional,
                margin_required=risk.margin_required,
                leverage=risk.leverage,
                reason=reason,
                stop_loss=stop_loss,
                take_profit=take_profit,
                existing_position=existing_position,
            )

        return ExecutionResult("REJECTED", f"Unsupported action: {normalized_action}")

    def snapshot(self, *, record: bool = True) -> dict[str, float]:
        latest = self._latest_account(create=True)
        if record:
            latest = self._record_equity(latest.cash_balance)
            self.session.commit()
        return {
            "cash_balance": latest.cash_balance,
            "equity": latest.equity,
            "realized_pnl": latest.realized_pnl,
            "unrealized_pnl": latest.unrealized_pnl,
            "drawdown": latest.drawdown,
        }

    def _open_or_add(
        self,
        *,
        symbol: str,
        side: str,
        action: str,
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
        if price <= 0:
            return ExecutionResult("REJECTED", "Execution price must be positive.", balance=account.cash_balance, equity=account.equity)
        notional = max_notional
        if notional < settings.min_paper_trade_notional:
            return ExecutionResult(
                "REJECTED",
                f"Paper trade notional ${notional:,.4f} is below the ${settings.min_paper_trade_notional:,.2f} minimum.",
                balance=account.cash_balance,
                equity=account.equity,
            )
        margin_required = min(margin_required or (notional / max(leverage, 1.0)), account.cash_balance)
        quantity = notional / price
        fee_rate = self._paper_fee_rate()
        fee = self._paper_fee(notional, fee_rate)
        realized_pnl = -fee
        if quantity <= DUST_QUANTITY:
            return ExecutionResult("REJECTED", "Computed quantity was dust-sized.", balance=account.cash_balance, equity=account.equity)
        if fee <= 0:
            return ExecutionResult("REJECTED", "Computed fee was zero; paper trade is too small.", balance=account.cash_balance, equity=account.equity)
        if margin_required + fee > account.cash_balance:
            return ExecutionResult("REJECTED", "Not enough paper cash for margin plus fee.")

        if existing_position is not None and existing_position.side.upper() != side:
            return ExecutionResult("REJECTED", f"Cannot add {side} while {existing_position.side.upper()} is open; close first.")

        if existing_position is None:
            position = Position(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=price,
                current_price=price,
                notional=notional,
                margin_used=margin_required,
                leverage=leverage,
                stop_loss=stop_loss if stop_loss is not None else self._default_stop_loss(price, side),
                take_profit=take_profit if take_profit is not None else self._default_take_profit(price, side),
                realized_pnl=realized_pnl,
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
            existing_position.realized_pnl += realized_pnl
            existing_position.stop_loss = stop_loss or existing_position.stop_loss or self._default_stop_loss(
                existing_position.entry_price,
                side,
            )
            existing_position.take_profit = take_profit or existing_position.take_profit or self._default_take_profit(
                existing_position.entry_price,
                side,
            )
            position = existing_position

        cash_after = account.cash_balance - margin_required - fee
        trade = PaperTrade(
            symbol=symbol,
            action=action,
            side=side,
            quantity=quantity,
            price=price,
            notional=notional,
            fee=fee,
            realized_pnl=realized_pnl,
            status="FILLED",
            reason=reason,
            raw_payload={
                "paper_leverage": leverage,
                "margin_required": margin_required,
                "gross_pnl": 0.0,
                "fee_rate": fee_rate,
                "fee_model": "binance_usdm_futures_taker_style",
                "intent": "open" if existing_position is None else "increase",
            },
        )
        self.session.add(trade)
        self.session.flush()
        equity_row = self._record_equity(cash_after, price_by_symbol={symbol: price})
        trade.balance_after = equity_row.cash_balance
        trade.equity_after = equity_row.equity
        position.current_price = price
        position.unrealized_pnl = self._position_unrealized(position, price)
        self.session.commit()
        self.session.refresh(trade)
        return ExecutionResult(
            "FILLED",
            f"Paper {side} {action} filled at {leverage:g}x using ${margin_required:,.2f} margin and ${fee:,.4f} fee.",
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
        requested_action: str,
    ) -> ExecutionResult:
        if existing_position is None:
            account = self._latest_account(create=True)
            return ExecutionResult("REJECTED", "No open paper futures position exists to close.", balance=account.cash_balance, equity=account.equity)

        account = self._latest_account(create=True)
        original_quantity = max(existing_position.quantity or 0.0, 0.0)
        margin_before = existing_position.margin_used or self._fallback_margin(existing_position)
        if original_quantity <= DUST_QUANTITY:
            return self._close_dust_position(
                symbol=symbol,
                price=price,
                position=existing_position,
                account=account,
                margin_before=margin_before,
            )

        close_quantity = min(requested_quantity or original_quantity, original_quantity)
        proceeds = close_quantity * price
        if close_quantity <= DUST_QUANTITY or proceeds <= DUST_NOTIONAL:
            return self._close_dust_position(
                symbol=symbol,
                price=price,
                position=existing_position,
                account=account,
                margin_before=margin_before,
            )

        fee_rate = self._paper_fee_rate()
        fee = self._paper_fee(proceeds, fee_rate)
        gross_pnl = self._gross_pnl(existing_position.side, existing_position.entry_price, price, close_quantity)
        realized_pnl = gross_pnl - fee
        released_margin = margin_before * (close_quantity / original_quantity) if original_quantity else margin_before
        cash_after = account.cash_balance + released_margin + gross_pnl - fee

        existing_position.quantity -= close_quantity
        existing_position.current_price = price
        existing_position.margin_used = max(margin_before - released_margin, 0.0)
        existing_position.notional = existing_position.quantity * existing_position.entry_price
        existing_position.realized_pnl += realized_pnl
        existing_position.unrealized_pnl = self._position_unrealized(existing_position, price)
        if existing_position.quantity <= 1e-12:
            existing_position.quantity = 0.0
            existing_position.status = "CLOSED"
            existing_position.closed_at = datetime.now(timezone.utc)

        closing_action = self._closing_trade_action(existing_position.side, requested_action)
        trade = PaperTrade(
            symbol=symbol,
            action=closing_action,
            side=existing_position.side.upper(),
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
                "fee_rate": fee_rate,
                "fee_model": "binance_usdm_futures_taker_style",
                "requested_action": requested_action,
                "intent": "close",
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
            f"Paper {existing_position.side.upper()} closed; released ${released_margin:,.2f} margin and paid ${fee:,.4f} fee.",
            trade_id=trade.id,
            balance=equity_row.cash_balance,
            equity=equity_row.equity,
        )

    def _close_dust_position(
        self,
        *,
        symbol: str,
        price: float,
        position: Position,
        account: AccountEquity,
        margin_before: float,
    ) -> ExecutionResult:
        released_margin = max(margin_before, 0.0)
        position.quantity = 0.0
        position.current_price = price
        position.margin_used = 0.0
        position.notional = 0.0
        position.unrealized_pnl = 0.0
        position.status = "CLOSED"
        position.closed_at = datetime.now(timezone.utc)
        equity_row = self._record_equity(account.cash_balance + released_margin, price_by_symbol={symbol: price})
        self.session.commit()
        return ExecutionResult(
            "HELD",
            f"Ignored dust paper position for {symbol}; no zero-quantity trade was recorded.",
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
            position.unrealized_pnl = self._position_unrealized(position, price)
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
            raw={
                "open_positions": len(open_positions),
                "reserved_margin": reserved_margin,
                "paper_leverage": settings.paper_leverage,
                "paper_max_leverage": settings.paper_max_leverage,
            },
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
        candle = self.session.scalar(select(Candle).where(Candle.symbol == symbol).order_by(desc(Candle.open_time)).limit(1))
        live_update = self.session.scalar(
            select(LiveCandleUpdate).where(LiveCandleUpdate.symbol == symbol).order_by(desc(LiveCandleUpdate.open_time)).limit(1)
        )
        if live_update and (candle is None or live_update.open_time >= candle.open_time):
            return live_update.close
        return candle.close if candle else None

    def _fallback_margin(self, position: Position) -> float:
        leverage = position.leverage or settings.paper_leverage or 1.0
        return (position.quantity * position.entry_price) / max(leverage, 1.0)

    def _position_unrealized(self, position: Position, price: float) -> float:
        return self._gross_pnl(position.side, position.entry_price, price, position.quantity)

    def _gross_pnl(self, side: str, entry_price: float, exit_price: float, quantity: float) -> float:
        if side.upper() == "SHORT":
            return (entry_price - exit_price) * quantity
        return (exit_price - entry_price) * quantity

    def _paper_fee_rate(self) -> float:
        return settings.paper_fee_rate if settings.paper_fee_rate > 0 else BROKER_TAKER_FEE_RATE

    def _paper_fee(self, notional: float, fee_rate: float) -> float:
        if notional <= 0 or fee_rate <= 0:
            return 0.0
        return round(notional * fee_rate, 10)

    def _closing_trade_action(self, side: str, requested_action: str) -> str:
        if requested_action == "CLOSE":
            return "SELL" if side.upper() == "LONG" else "BUY"
        return requested_action

    def _default_stop_loss(self, price: float, side: str) -> float | None:
        if settings.auto_default_stop_loss_pct <= 0:
            return None
        multiplier = 1.0 - settings.auto_default_stop_loss_pct if side.upper() == "LONG" else 1.0 + settings.auto_default_stop_loss_pct
        return round(price * multiplier, 8)

    def _default_take_profit(self, price: float, side: str) -> float | None:
        if settings.auto_default_take_profit_pct <= 0:
            return None
        multiplier = 1.0 + settings.auto_default_take_profit_pct if side.upper() == "LONG" else 1.0 - settings.auto_default_take_profit_pct
        return round(price * multiplier, 8)
