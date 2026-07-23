from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import (
    AccountEquity,
    Candle,
    LiveCandleUpdate,
    PaperSandboxAccount,
    PaperTrade,
    PortfolioTargetRecord,
    Position,
    RiskDecisionRecord,
    SimulatedOrderRecord,
)
from app.trading.risk_manager import RiskManager, RiskResult

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
    def __init__(self, session: Session, *, paper_account_id: str = "champion") -> None:
        self.session = session
        self.risk = RiskManager(session)
        self.paper_account_id = paper_account_id

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
        leverage: float | None = None,
        margin_pct: float | None = None,
        paper_data_collection_exploration: bool = False,
        risk_decision_id: str | None = None,
        decision_trace_id: str | None = None,
        simulated_order_id: str | None = None,
        paper_account_id: str | None = None,
    ) -> ExecutionResult:
        if not settings.is_paper_mode:
            return ExecutionResult("REJECTED", "Trading mode is not paper; real order APIs are disabled.")

        account_id = paper_account_id or self.paper_account_id
        normalized_symbol = symbol.upper()
        normalized_action = action.upper()
        latest_account = self._latest_account(create=True, paper_account_id=account_id)
        mark_price, market_observed_at = self._resolve_market(normalized_symbol, price)
        if normalized_action != "HOLD" and mark_price is None:
            return ExecutionResult("REJECTED", "No price is available for this symbol.")

        existing_position = self._open_position(normalized_symbol, paper_account_id=account_id)
        requested_notional = notional
        if requested_notional is None and quantity is not None and mark_price:
            requested_notional = quantity * mark_price

        if simulated_order_id and not risk_decision_id:
            return ExecutionResult("REJECTED", "A simulated order cannot execute without its approved risk decision.")
        if risk_decision_id:
            risk = self._approved_risk_result(
                risk_decision_id=risk_decision_id,
                account_id=account_id,
                symbol=normalized_symbol,
                action=normalized_action,
                existing_position=existing_position,
                requested_notional=requested_notional,
                requested_quantity=quantity,
                equity=latest_account.equity,
                decision_trace_id=decision_trace_id,
                simulated_order_id=simulated_order_id,
            )
        else:
            risk = self.risk.evaluate(
                action=normalized_action,
                confidence=confidence,
                cash_balance=latest_account.cash_balance,
                equity=latest_account.equity,
                requested_notional=requested_notional,
                existing_position=existing_position,
                requested_leverage=leverage,
                requested_margin_pct=margin_pct,
                paper_data_collection_exploration=paper_data_collection_exploration,
                market_observed_at=market_observed_at,
                paper_account_id=account_id,
            )
        if not risk.accepted:
            return ExecutionResult("REJECTED", risk.reason, balance=latest_account.cash_balance, equity=latest_account.equity)

        if normalized_action == "HOLD":
            equity_row = self._record_equity(latest_account.cash_balance, price_by_symbol={normalized_symbol: mark_price}, paper_account_id=account_id)
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
                paper_account_id=account_id,
                risk_decision_id=risk_decision_id,
                decision_trace_id=decision_trace_id,
                simulated_order_id=simulated_order_id,
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
                paper_account_id=account_id,
                risk_decision_id=risk_decision_id,
                decision_trace_id=decision_trace_id,
                simulated_order_id=simulated_order_id,
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

    def reset_paper_account_if_needed(self) -> dict[str, object] | None:
        if not settings.is_paper_mode:
            return None
        if not settings.paper_data_collection_mode or not settings.paper_data_collection_reset_enabled:
            return None

        latest = self._latest_account(create=True)
        starting_balance = self._account_starting_balance(self.paper_account_id)
        reset_threshold = starting_balance * max(settings.paper_data_collection_reset_equity_pct, 0.0)
        if latest.equity > reset_threshold:
            return None

        previous_equity = latest.equity
        reset_at = datetime.now(timezone.utc)
        closed_positions: list[dict[str, object]] = []
        open_positions = list(
            self.session.scalars(
                select(Position).where(Position.status == "OPEN", Position.paper_account_id == self.paper_account_id)
            )
        )
        for position in open_positions:
            mark_price = self._resolve_price(position.symbol, None) or position.current_price or position.entry_price
            if mark_price and mark_price > 0:
                result = self._close(
                    symbol=position.symbol,
                    price=mark_price,
                    reason="Paper data collection account reset: force-closing open position before balance reset.",
                    requested_quantity=None,
                    existing_position=position,
                    requested_action="CLOSE",
                    paper_account_id=self.paper_account_id,
                    risk_decision_id=None,
                    decision_trace_id=None,
                    simulated_order_id=None,
                )
                closed_positions.append(
                    {
                        "position_id": position.id,
                        "symbol": position.symbol,
                        "side": position.side,
                        "price": mark_price,
                        "status": result.status,
                        "trade_id": result.trade_id,
                        "message": result.message,
                    }
                )
                continue

            position.quantity = 0.0
            position.notional = 0.0
            position.margin_used = 0.0
            position.unrealized_pnl = 0.0
            position.status = "CLOSED"
            position.closed_at = reset_at
            closed_positions.append(
                {
                    "position_id": position.id,
                    "symbol": position.symbol,
                    "side": position.side,
                    "status": "CLOSED_WITHOUT_MARK_PRICE",
                    "message": "No mark price was available; position was marked closed during paper reset.",
                }
            )

        realized_pnl = float(
            self.session.scalar(
                select(func.coalesce(func.sum(PaperTrade.realized_pnl), 0.0)).where(
                    PaperTrade.paper_account_id == self.paper_account_id
                )
            )
            or 0.0
        )
        reset_row = AccountEquity(
            paper_account_id=self.paper_account_id,
            timestamp=reset_at,
            cash_balance=starting_balance,
            equity=starting_balance,
            realized_pnl=realized_pnl,
            unrealized_pnl=0.0,
            drawdown=0.0,
            raw={
                "paper_data_collection_reset": True,
                "previous_equity": previous_equity,
                "reset_threshold": reset_threshold,
                "timestamp": reset_at.isoformat(),
                "closed_positions": closed_positions,
            },
        )
        self.session.add(reset_row)
        self.session.commit()
        return {
            "paper_data_collection_reset": True,
            "previous_equity": previous_equity,
            "reset_threshold": reset_threshold,
            "timestamp": reset_at.isoformat(),
            "closed_positions": closed_positions,
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
        paper_account_id: str,
        risk_decision_id: str | None,
        decision_trace_id: str | None,
        simulated_order_id: str | None,
    ) -> ExecutionResult:
        account = self._latest_account(create=True, paper_account_id=paper_account_id)
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
                paper_account_id=paper_account_id,
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
            paper_account_id=paper_account_id,
            risk_decision_id=risk_decision_id,
            simulated_order_id=simulated_order_id,
            decision_trace_id=decision_trace_id,
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
        equity_row = self._record_equity(cash_after, price_by_symbol={symbol: price}, paper_account_id=paper_account_id)
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
        paper_account_id: str,
        risk_decision_id: str | None,
        decision_trace_id: str | None,
        simulated_order_id: str | None,
    ) -> ExecutionResult:
        if existing_position is None:
            account = self._latest_account(create=True, paper_account_id=paper_account_id)
            return ExecutionResult("REJECTED", "No open paper futures position exists to close.", balance=account.cash_balance, equity=account.equity)

        account = self._latest_account(create=True, paper_account_id=paper_account_id)
        original_quantity = max(existing_position.quantity or 0.0, 0.0)
        margin_before = existing_position.margin_used or self._fallback_margin(existing_position)
        if original_quantity <= DUST_QUANTITY:
            return self._close_dust_position(
                symbol=symbol,
                price=price,
                position=existing_position,
                account=account,
                margin_before=margin_before,
                paper_account_id=paper_account_id,
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
                paper_account_id=paper_account_id,
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
            paper_account_id=paper_account_id,
            risk_decision_id=risk_decision_id,
            simulated_order_id=simulated_order_id,
            decision_trace_id=decision_trace_id,
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
        equity_row = self._record_equity(cash_after, price_by_symbol={symbol: price}, paper_account_id=paper_account_id)
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
        paper_account_id: str,
    ) -> ExecutionResult:
        released_margin = max(margin_before, 0.0)
        position.quantity = 0.0
        position.current_price = price
        position.margin_used = 0.0
        position.notional = 0.0
        position.unrealized_pnl = 0.0
        position.status = "CLOSED"
        position.closed_at = datetime.now(timezone.utc)
        equity_row = self._record_equity(
            account.cash_balance + released_margin,
            price_by_symbol={symbol: price},
            paper_account_id=paper_account_id,
        )
        self.session.commit()
        return ExecutionResult(
            "HELD",
            f"Ignored dust paper position for {symbol}; no zero-quantity trade was recorded.",
            balance=equity_row.cash_balance,
            equity=equity_row.equity,
        )

    def _latest_account(self, create: bool = False, *, paper_account_id: str | None = None) -> AccountEquity:
        account_id = paper_account_id or self.paper_account_id
        latest = self.session.scalar(
            select(AccountEquity).where(AccountEquity.paper_account_id == account_id).order_by(desc(AccountEquity.timestamp)).limit(1)
        )
        if latest is None and create:
            starting_balance = self._account_starting_balance(account_id)
            latest = AccountEquity(
                paper_account_id=account_id,
                cash_balance=starting_balance,
                equity=starting_balance,
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

    def _account_starting_balance(self, account_id: str) -> float:
        sandbox = self.session.scalar(
            select(PaperSandboxAccount).where(PaperSandboxAccount.account_id == account_id).limit(1)
        )
        try:
            value = float(sandbox.starting_balance) if sandbox is not None else float(settings.paper_start_balance)
        except (TypeError, ValueError):
            value = 0.0
        return value if math.isfinite(value) and value > 0 else 0.0

    def _record_equity(
        self,
        cash_balance: float,
        price_by_symbol: dict[str, float | None] | None = None,
        *,
        paper_account_id: str | None = None,
    ) -> AccountEquity:
        account_id = paper_account_id or self.paper_account_id
        price_by_symbol = price_by_symbol or {}
        reserved_margin = 0.0
        unrealized_pnl = 0.0
        open_positions = list(
            self.session.scalars(select(Position).where(Position.status == "OPEN", Position.paper_account_id == account_id))
        )
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

        realized_pnl = float(
            self.session.scalar(
                select(func.coalesce(func.sum(PaperTrade.realized_pnl), 0.0)).where(PaperTrade.paper_account_id == account_id)
            )
            or 0.0
        )
        equity = cash_balance + reserved_margin + unrealized_pnl
        peak_equity = float(
            self.session.scalar(select(func.max(AccountEquity.equity)).where(AccountEquity.paper_account_id == account_id)) or equity
        )
        # Store drawdown as the conventional positive loss from the running peak.
        # Risk readers remain compatible with historical rows that used a negative
        # signed return.
        drawdown = max((peak_equity - equity) / peak_equity, 0.0) if peak_equity else 0.0
        row = AccountEquity(
            paper_account_id=account_id,
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

    def _open_position(self, symbol: str, *, paper_account_id: str | None = None) -> Position | None:
        account_id = paper_account_id or self.paper_account_id
        return self.session.scalar(
            select(Position)
            .where(Position.symbol == symbol, Position.status == "OPEN", Position.paper_account_id == account_id)
            .order_by(desc(Position.opened_at))
            .limit(1)
        )

    def _approved_risk_result(
        self,
        *,
        risk_decision_id: str,
        account_id: str,
        symbol: str,
        action: str,
        existing_position: Position | None,
        requested_notional: float | None,
        requested_quantity: float | None,
        equity: float,
        decision_trace_id: str | None,
        simulated_order_id: str | None,
    ) -> RiskResult:
        """Load and bind the independent V2 approval to one submitted order."""
        decision = self.session.scalar(
            select(RiskDecisionRecord).where(RiskDecisionRecord.risk_decision_id == risk_decision_id).limit(1)
        )
        if decision is None:
            return RiskResult(False, "Execution rejected: risk decision was not found.", 0.0, rejection_reasons=("MISSING_RISK_DECISION",))
        if decision.paper_account_id != account_id:
            return RiskResult(False, "Execution rejected: risk decision belongs to another paper account.", 0.0, rejection_reasons=("RISK_ACCOUNT_MISMATCH",))
        if not decision.approved:
            return RiskResult(False, "Execution rejected: risk decision is not approved.", 0.0, rejection_reasons=("RISK_NOT_APPROVED",))
        if not simulated_order_id:
            return RiskResult(False, "Execution rejected: approved V2 risk requires a submitted simulated order.", 0.0, rejection_reasons=("MISSING_SIMULATED_ORDER",))
        order = self.session.scalar(
            select(SimulatedOrderRecord).where(SimulatedOrderRecord.order_id == simulated_order_id).limit(1)
        )
        if order is None:
            return RiskResult(False, "Execution rejected: submitted simulated order was not found.", 0.0, rejection_reasons=("MISSING_SIMULATED_ORDER",))
        target = self.session.scalar(
            select(PortfolioTargetRecord).where(PortfolioTargetRecord.portfolio_target_id == decision.portfolio_target_id).limit(1)
        )
        if target is None:
            return RiskResult(False, "Execution rejected: approved portfolio target was not found.", 0.0, rejection_reasons=("MISSING_PORTFOLIO_TARGET",))
        if (
            order.risk_decision_id != decision.risk_decision_id
            or order.portfolio_target_id != decision.portfolio_target_id
            or order.paper_account_id != account_id
            or target.paper_account_id != account_id
        ):
            return RiskResult(False, "Execution rejected: simulated order does not match its risk approval.", 0.0, rejection_reasons=("ORDER_RISK_BINDING_MISMATCH",))
        if (
            not decision_trace_id
            or order.decision_trace_id != decision_trace_id
            or decision.decision_trace_id != decision_trace_id
            or target.decision_trace_id != decision_trace_id
        ):
            return RiskResult(False, "Execution rejected: simulated order trace does not match its risk approval.", 0.0, rejection_reasons=("RISK_TRACE_MISMATCH",))
        if order.symbol.upper() != symbol.upper() or target.symbol.upper() != symbol.upper():
            return RiskResult(False, "Execution rejected: simulated order symbol does not match its risk approval.", 0.0, rejection_reasons=("RISK_SYMBOL_MISMATCH",))
        if order.state != "ACKNOWLEDGED":
            return RiskResult(False, "Execution rejected: simulated order is not executable in its current state.", 0.0, rejection_reasons=("INVALID_ORDER_STATE",))
        expires_at = order.expires_at
        if expires_at is not None:
            expires_at = expires_at.replace(tzinfo=timezone.utc) if expires_at.tzinfo is None else expires_at.astimezone(timezone.utc)
            if datetime.now(timezone.utc) > expires_at:
                return RiskResult(False, "Execution rejected: simulated order has expired.", 0.0, rejection_reasons=("ORDER_EXPIRED",))
        consumed = self.session.scalar(
            select(PaperTrade.id).where(PaperTrade.simulated_order_id == simulated_order_id).limit(1)
        )
        if consumed is not None:
            return RiskResult(False, "Execution rejected: simulated order was already filled.", 0.0, rejection_reasons=("RISK_APPROVAL_REPLAY",))
        intent, side = self.risk._intent(action, existing_position)
        if intent == "close":
            if requested_quantity is None or requested_quantity <= 0 or requested_quantity > order.requested_quantity + 1e-12:
                return RiskResult(False, "Execution rejected: close quantity exceeds the submitted order.", 0.0, intent=intent, side=side, rejection_reasons=("ORDER_QUANTITY_MISMATCH",))
            return RiskResult(
                True,
                "Persisted V2 risk approval permits this protective close.",
                existing_position.quantity * existing_position.current_price if existing_position else 0.0,
                leverage=existing_position.leverage if existing_position else 1.0,
                intent="close",
                side=side,
            )
        if intent not in {"open", "increase"}:
            return RiskResult(False, "Execution rejected: V2 approval does not match order intent.", 0.0, intent=intent, side=side)
        leverage = max(float(decision.approved_leverage or 0.0), 1.0)
        notional = max(float(requested_notional or 0.0), 0.0)
        if notional <= 0:
            return RiskResult(False, "Execution rejected: approved V2 order has no notional.", 0.0, intent=intent, side=side)
        if notional > float(order.requested_notional) + max(abs(float(order.requested_notional)) * 1e-9, 1e-9):
            return RiskResult(False, "Execution rejected: order notional exceeds the submitted risk-approved order.", 0.0, intent=intent, side=side, rejection_reasons=("ORDER_NOTIONAL_MISMATCH",))
        approved_delta = abs(float(decision.approved_exposure) - float(target.current_exposure))
        approved_equity = (decision.payload or {}).get("equity")
        try:
            approved_notional = approved_delta * float(approved_equity)
        except (TypeError, ValueError):
            approved_notional = float(order.requested_notional)
        if not math.isfinite(approved_notional) or float(order.requested_notional) > approved_notional + max(abs(approved_notional) * 1e-9, 1e-9):
            return RiskResult(False, "Execution rejected: submitted notional exceeds the persisted approved exposure.", 0.0, intent=intent, side=side, rejection_reasons=("APPROVED_EXPOSURE_MISMATCH",))
        margin = notional / leverage
        if margin > max(equity, 0.0):
            return RiskResult(False, "Execution rejected: approved order exceeds paper equity.", 0.0, intent=intent, side=side)
        return RiskResult(
            True,
            "Persisted V2 risk decision approved paper execution.",
            notional,
            margin_required=margin,
            leverage=leverage,
            allocation_pct=max(min(float(decision.approved_exposure), 1.0), -1.0),
            intent=intent,
            side=side,
            triggered_limits=tuple(decision.triggered_limits or []),
        )

    def _resolve_market(self, symbol: str, explicit_price: float | None) -> tuple[float | None, datetime | None]:
        candle = self.session.scalar(select(Candle).where(Candle.symbol == symbol).order_by(desc(Candle.open_time)).limit(1))
        live_update = self.session.scalar(
            select(LiveCandleUpdate).where(LiveCandleUpdate.symbol == symbol).order_by(desc(LiveCandleUpdate.open_time)).limit(1)
        )
        if live_update and (candle is None or live_update.open_time >= candle.open_time):
            observed_at = live_update.event_time or live_update.updated_at or live_update.open_time
            return (explicit_price if explicit_price and explicit_price > 0 else live_update.close), observed_at
        if candle:
            return (explicit_price if explicit_price and explicit_price > 0 else candle.close), (candle.close_time or candle.updated_at or candle.open_time)
        return (explicit_price if explicit_price and explicit_price > 0 else None), None

    def _resolve_price(self, symbol: str, explicit_price: float | None) -> float | None:
        """Backward-compatible price helper; V2 callers use `_resolve_market`."""
        return self._resolve_market(symbol, explicit_price)[0]

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
