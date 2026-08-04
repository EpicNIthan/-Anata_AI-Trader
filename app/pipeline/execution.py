"""Paper-only execution adapter gated by persisted V2 risk decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import (
    PaperTrade,
    PortfolioTargetRecord,
    Position,
    RiskDecisionRecord,
    RiskControlState,
    SimulatedFillRecord,
    SimulatedOrderRecord,
)
from app.pipeline.domain import Direction, OrderState, PortfolioTarget, RiskDecision, SimulatedFill, SimulatedOrder, new_id
from app.pipeline.risk import MarketSnapshot
from app.trading.paper_engine import ExecutionResult, PaperEngine


@dataclass(frozen=True)
class ExecutionOutcome:
    order: SimulatedOrder | None
    fill: SimulatedFill | None
    result: ExecutionResult
    duplicate: bool = False


class PaperExecutionSimulator:
    """Reach a risk-approved target using the legacy paper ledger as the accounting core.

    This is the only V2 class allowed to import ``PaperEngine``. It requires an approved
    persisted decision before it creates an order, thereby retaining compatibility with
    existing `PaperTrade` and `Position` history without restoring a model-to-order path.
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    def submit_target(
        self,
        *,
        target: PortfolioTarget,
        risk_decision: RiskDecision,
        market: MarketSnapshot,
        equity: float,
        account_id: str,
        decision_trace_id: str,
        order_type: str = "MARKET",
        limit_price: float | None = None,
    ) -> ExecutionOutcome:
        if not settings.is_paper_mode:
            return ExecutionOutcome(None, None, ExecutionResult("REJECTED", "Paper execution is disabled outside TRADING_MODE=paper."))
        persisted = self.session.scalar(
            select(RiskDecisionRecord).where(RiskDecisionRecord.risk_decision_id == risk_decision.risk_decision_id).limit(1)
        )
        if persisted is None or not persisted.approved or not risk_decision.approved:
            return ExecutionOutcome(None, None, ExecutionResult("REJECTED", "Execution requires a persisted approved V2 risk decision."))
        persisted_target = self.session.scalar(
            select(PortfolioTargetRecord)
            .where(PortfolioTargetRecord.portfolio_target_id == persisted.portfolio_target_id)
            .limit(1)
        )
        authorization_error = self._authorization_error(
            persisted=persisted,
            persisted_target=persisted_target,
            target=target,
            risk_decision=risk_decision,
            market=market,
            equity=equity,
            account_id=account_id,
            decision_trace_id=decision_trace_id,
        )
        if authorization_error:
            return ExecutionOutcome(None, None, ExecutionResult("REJECTED", authorization_error))
        if market.price <= 0:
            return ExecutionOutcome(None, None, ExecutionResult("REJECTED", "Execution requires a positive market snapshot."))
        normalized_order_type = str(order_type or "MARKET").upper()
        if normalized_order_type not in {"MARKET", "LIMIT"}:
            return ExecutionOutcome(None, None, ExecutionResult("REJECTED", "Only MARKET and LIMIT paper orders are supported."))
        if normalized_order_type == "LIMIT" and (
            limit_price is None or not math.isfinite(float(limit_price)) or float(limit_price) <= 0
        ):
            return ExecutionOutcome(None, None, ExecutionResult("REJECTED", "A positive finite limit price is required."))

        assert persisted_target is not None  # Established by the authorization check.
        approved_equity = self._persisted_equity(persisted, fallback=equity)
        delta = float(persisted.approved_exposure) - float(persisted_target.current_exposure)
        if abs(delta) <= 1e-12:
            return ExecutionOutcome(None, None, ExecutionResult("HELD", "Approved target does not require an exposure change."))
        client_order_id = f"{persisted.decision_trace_id}:{persisted.portfolio_target_id}"
        duplicate = self.session.scalar(
            select(SimulatedOrderRecord)
            .where(
                (SimulatedOrderRecord.risk_decision_id == persisted.risk_decision_id)
                | (SimulatedOrderRecord.client_order_id == client_order_id)
            )
            .limit(1)
        )
        if duplicate is not None:
            return ExecutionOutcome(None, None, ExecutionResult("REJECTED", "Risk approval was already consumed; replay prevented."), duplicate=True)

        position = self.session.scalar(
            select(Position)
            .where(Position.paper_account_id == account_id, Position.symbol == persisted_target.symbol, Position.status == "OPEN")
            .limit(1)
        )
        side, action, requested_notional, requested_quantity = self._instruction(
            delta,
            current_exposure=float(persisted_target.current_exposure),
            requested_target_exposure=float(persisted_target.requested_target_exposure),
            position=position,
            price=market.price,
            equity=approved_equity,
        )
        execution_side = self._execution_side(action, position)
        if requested_notional <= 0 or requested_quantity <= 0:
            return ExecutionOutcome(None, None, ExecutionResult("HELD", "Approved target only requires a protective no-op."))
        order = SimulatedOrder(
            risk_decision_id=persisted.risk_decision_id,
            portfolio_target_id=persisted.portfolio_target_id,
            symbol=persisted_target.symbol,
            side=side,
            requested_quantity=requested_quantity,
            requested_notional=requested_notional,
            order_type=normalized_order_type,
            limit_price=float(limit_price) if limit_price is not None else None,
            client_order_id=client_order_id,
            account_id=account_id,
            expires_at=datetime.now(timezone.utc)
            + timedelta(
                seconds=max(
                    int(getattr(settings, "paper_simulated_order_ttl_seconds", settings.v2_signal_ttl_seconds)),
                    1,
                )
            ),
            metadata={
                "decision_trace_id": persisted.decision_trace_id,
                "paper_only": True,
                "latency_ms": max(int(getattr(settings, "paper_simulated_latency_ms", 0)), 0),
                "volume_participation": self._bounded_participation(),
                "action": action,
                "execution_side": execution_side.value,
                "reference_market_price": market.price,
                "simplified_assumptions": [
                    "latency_is_recorded_not_wall_clock_slept",
                    "market_impact_is_deterministic",
                    "funding_is_booked_at_fill_as_a_simplified_cost",
                ],
            },
        )
        row = self._persist_order(order, persisted.decision_trace_id)
        self._transition(row, OrderState.RISK_APPROVED)
        self._transition(row, OrderState.SUBMITTED)
        self._transition(row, OrderState.ACKNOWLEDGED)

        if normalized_order_type == "LIMIT" and not self._limit_is_marketable(
            float(limit_price), market, execution_side
        ):
            row.payload = {
                **(row.payload or {}),
                "resting": True,
                "remaining_quantity": requested_quantity,
                "remaining_notional": requested_notional,
            }
            self.session.commit()
            return ExecutionOutcome(
                order,
                None,
                ExecutionResult("PENDING", "Paper limit order acknowledged and resting until marketable or expired."),
            )

        return self._fill_existing_order(row, market)

    def _domain_order(self, row: SimulatedOrderRecord) -> SimulatedOrder:
        """Restore a strict order view from the persisted paper state machine."""

        created = row.created_at
        updated = row.updated_at
        expires = row.expires_at
        created = created.replace(tzinfo=timezone.utc) if created.tzinfo is None else created.astimezone(timezone.utc)
        updated = updated.replace(tzinfo=timezone.utc) if updated.tzinfo is None else updated.astimezone(timezone.utc)
        if expires is not None:
            expires = expires.replace(tzinfo=timezone.utc) if expires.tzinfo is None else expires.astimezone(timezone.utc)
        return SimulatedOrder(
            order_id=row.order_id,
            risk_decision_id=row.risk_decision_id,
            portfolio_target_id=row.portfolio_target_id,
            symbol=row.symbol,
            side=Direction(row.side),
            order_type=row.order_type,
            requested_quantity=row.requested_quantity,
            requested_notional=row.requested_notional,
            limit_price=row.limit_price,
            state=OrderState(row.state),
            client_order_id=row.client_order_id,
            account_id=row.paper_account_id,
            created_at=created,
            updated_at=updated,
            expires_at=expires,
            metadata=row.payload or {},
        )

    def _fill_existing_order(self, row: SimulatedOrderRecord, market: MarketSnapshot) -> ExecutionOutcome:
        """Fill one approved open order up to its remaining risk-authorized amount."""

        order = self._domain_order(row)
        if row.state not in {OrderState.ACKNOWLEDGED.value, OrderState.PARTIALLY_FILLED.value}:
            return ExecutionOutcome(order, None, ExecutionResult("REJECTED", "Paper order is not open for filling."))
        now = datetime.now(timezone.utc)
        expires = row.expires_at
        if expires is not None:
            expires = expires.replace(tzinfo=timezone.utc) if expires.tzinfo is None else expires.astimezone(timezone.utc)
        if expires is not None and expires <= now:
            row.state = OrderState.EXPIRED.value
            row.updated_at = now
            self.session.commit()
            return ExecutionOutcome(self._domain_order(row), None, ExecutionResult("EXPIRED", "Paper order expired before fill."))

        decision = self.session.scalar(
            select(RiskDecisionRecord).where(RiskDecisionRecord.risk_decision_id == row.risk_decision_id).limit(1)
        )
        target = self.session.scalar(
            select(PortfolioTargetRecord).where(PortfolioTargetRecord.portfolio_target_id == row.portfolio_target_id).limit(1)
        )
        if decision is None or target is None or not decision.approved:
            row.state = OrderState.REJECTED.value
            row.payload = {**(row.payload or {}), "rejection_reason": "RISK_OR_TARGET_MISSING"}
            row.updated_at = now
            self.session.commit()
            return ExecutionOutcome(self._domain_order(row), None, ExecutionResult("REJECTED", "Approved risk lineage is unavailable."))

        payload = row.payload or {}
        action = str(payload.get("action") or ("BUY" if row.side == Direction.LONG.value else "SELL")).upper()
        try:
            execution_side = Direction(str(payload.get("execution_side") or row.side).upper())
        except ValueError:
            row.state = OrderState.ERROR.value
            row.payload = {**payload, "error": "INVALID_EXECUTION_SIDE"}
            self.session.commit()
            return ExecutionOutcome(self._domain_order(row), None, ExecutionResult("ERROR", "Persisted execution side is invalid."))
        execution_market_error = self._execution_market_error(row, market, action=action, now=now)
        if execution_market_error:
            row.state = OrderState.REJECTED.value
            row.payload = {**payload, "rejection_reason": execution_market_error}
            row.updated_at = now
            self.session.commit()
            return ExecutionOutcome(self._domain_order(row), None, ExecutionResult("REJECTED", execution_market_error))
        if row.order_type == "LIMIT" and row.limit_price is not None and not self._limit_is_marketable(
            float(row.limit_price), market, execution_side
        ):
            return ExecutionOutcome(order, None, ExecutionResult("PENDING", "Paper limit order remains resting."))

        filled_quantity = float(
            self.session.scalar(
                select(func.coalesce(func.sum(SimulatedFillRecord.quantity), 0.0)).where(
                    SimulatedFillRecord.order_id == row.order_id
                )
            )
            or 0.0
        )
        filled_notional = float(
            self.session.scalar(
                select(func.coalesce(func.sum(SimulatedFillRecord.notional), 0.0)).where(
                    SimulatedFillRecord.order_id == row.order_id
                )
            )
            or 0.0
        )
        remaining_quantity = max(float(row.requested_quantity) - filled_quantity, 0.0)
        remaining_notional = max(float(row.requested_notional) - filled_notional, 0.0)
        completion_measure = remaining_quantity if action == "CLOSE" else remaining_notional
        if completion_measure <= 1e-10:
            row.state = OrderState.FILLED.value
            row.updated_at = now
            self.session.commit()
            return ExecutionOutcome(self._domain_order(row), None, ExecutionResult("FILLED", "Paper order was already fully reconciled."))

        fill_fraction = self._fill_fraction(
            remaining_quantity,
            market,
            already_partially_filled=filled_quantity > 0 or filled_notional > 0,
        )
        if fill_fraction <= 0:
            if row.order_type == "LIMIT":
                row.payload = {**payload, "last_fill_skip": "NO_SIMULATED_LIQUIDITY"}
                self.session.commit()
                return ExecutionOutcome(self._domain_order(row), None, ExecutionResult("PENDING", "No simulated liquidity is currently available."))
            row.state = OrderState.REJECTED.value
            row.payload = {**payload, "rejection_reason": "NO_SIMULATED_LIQUIDITY"}
            row.updated_at = now
            self.session.commit()
            return ExecutionOutcome(self._domain_order(row), None, ExecutionResult("REJECTED", "No simulated liquidity was available."))

        planned_quantity = remaining_quantity * fill_fraction
        planned_notional = remaining_notional * fill_fraction
        simulated_price = self._execution_price(
            market,
            execution_side,
            requested_quantity=planned_quantity,
            limit_price=float(row.limit_price) if row.limit_price is not None else None,
        )
        funding_cost = planned_notional * self._funding_rate(market)
        engine = PaperEngine(self.session, paper_account_id=row.paper_account_id)
        result = engine.execute_signal(
            symbol=row.symbol,
            action=action,
            confidence=1.0,
            reason="V2 approved paper target",
            price=simulated_price,
            quantity=planned_quantity if action == "CLOSE" else None,
            notional=planned_notional if action != "CLOSE" else None,
            risk_decision_id=row.risk_decision_id,
            decision_trace_id=row.decision_trace_id,
            simulated_order_id=row.order_id,
            paper_account_id=row.paper_account_id,
            simulated_funding_cost=funding_cost,
        )
        if result.status != "FILLED":
            row.state = OrderState.ERROR.value if filled_quantity > 0 or filled_notional > 0 else OrderState.REJECTED.value
            row.payload = {**payload, "fill_error": result.message}
            row.updated_at = now
            self.session.commit()
            return ExecutionOutcome(self._domain_order(row), None, result)

        trade = self.session.get(PaperTrade, result.trade_id) if result.trade_id else None
        if trade is None:
            row.state = OrderState.ERROR.value
            self.session.commit()
            return ExecutionOutcome(self._domain_order(row), None, ExecutionResult("ERROR", "Paper engine filled without a trade record."))
        fill = SimulatedFill(
            order_id=row.order_id,
            symbol=row.symbol,
            side=execution_side,
            quantity=trade.quantity,
            price=trade.price,
            notional=trade.notional,
            fee=trade.fee,
            slippage=abs(trade.price - market.price) / market.price if market.price else 0.0,
            funding=funding_cost,
            metadata={
                "paper_trade_id": trade.id,
                "fill_fraction_of_remainder": fill_fraction,
                "volume_participation": self._bounded_participation(),
                "funding_rate": self._funding_rate(market),
                "funding_cash_flow": -funding_cost,
                "funding_booked": True,
                "market_impact_coefficient": max(
                    float(getattr(settings, "paper_simulated_market_impact_coefficient", 0.0)), 0.0
                ),
            },
        )
        self.session.add(
            SimulatedFillRecord(
                fill_id=fill.fill_id,
                order_id=fill.order_id,
                decision_trace_id=row.decision_trace_id,
                paper_account_id=row.paper_account_id,
                symbol=fill.symbol,
                side=fill.side.value,
                quantity=fill.quantity,
                price=fill.price,
                notional=fill.notional,
                fee=fill.fee,
                slippage=fill.slippage,
                funding=fill.funding,
                filled_at=fill.filled_at,
                payload=fill.metadata,
            )
        )
        trade.risk_decision_id = row.risk_decision_id
        trade.simulated_order_id = row.order_id
        trade.decision_trace_id = row.decision_trace_id
        trade.paper_account_id = row.paper_account_id
        self.session.flush()
        total_quantity = filled_quantity + fill.quantity
        total_notional = filled_notional + fill.notional
        complete = (
            total_quantity >= float(row.requested_quantity) - 1e-10
            if action == "CLOSE"
            else total_notional >= float(row.requested_notional) - 1e-8
        )
        self._transition(row, OrderState.FILLED if complete else OrderState.PARTIALLY_FILLED)
        row.payload = {
            **payload,
            "filled_quantity": total_quantity,
            "filled_notional": total_notional,
            "remaining_quantity": max(float(row.requested_quantity) - total_quantity, 0.0),
            "remaining_notional": max(float(row.requested_notional) - total_notional, 0.0),
            "fill_attempts": int(payload.get("fill_attempts") or 0) + 1,
        }
        self.session.commit()
        return ExecutionOutcome(self._domain_order(row), fill, result)

    def process_resting_orders(
        self,
        markets: dict[str, MarketSnapshot],
        *,
        now: datetime | None = None,
        limit: int = 200,
    ) -> dict[str, object]:
        """Match bounded resting paper limits against a new public market snapshot."""

        now = now or datetime.now(timezone.utc)
        now = now.replace(tzinfo=timezone.utc) if now.tzinfo is None else now.astimezone(timezone.utc)
        safe_limit = max(min(int(limit), 1_000), 1)
        rows = list(
            self.session.scalars(
                select(SimulatedOrderRecord)
                .where(
                    SimulatedOrderRecord.order_type == "LIMIT",
                    SimulatedOrderRecord.state.in_(
                        (OrderState.ACKNOWLEDGED.value, OrderState.PARTIALLY_FILLED.value)
                    ),
                )
                .order_by(SimulatedOrderRecord.created_at)
                .limit(safe_limit)
            )
        )
        summary: dict[str, object] = {
            "paper_only": True,
            "inspected": len(rows),
            "filled": 0,
            "partially_filled": 0,
            "pending": 0,
            "expired": 0,
            "rejected_or_error": 0,
            "outcomes": [],
        }
        outcomes: list[dict[str, object]] = []
        for row in rows:
            expires = row.expires_at
            if expires is not None:
                expires = expires.replace(tzinfo=timezone.utc) if expires.tzinfo is None else expires.astimezone(timezone.utc)
            if expires is not None and expires <= now:
                row.state = OrderState.EXPIRED.value
                row.updated_at = now
                summary["expired"] = int(summary["expired"]) + 1
                outcomes.append({"order_id": row.order_id, "status": "EXPIRED"})
                continue
            market = markets.get(row.symbol.upper())
            if market is None or market.price <= 0:
                summary["pending"] = int(summary["pending"]) + 1
                outcomes.append({"order_id": row.order_id, "status": "PENDING", "reason": "MARKET_UNAVAILABLE"})
                continue
            outcome = self._fill_existing_order(row, market)
            status = outcome.order.state.value if outcome.order is not None else outcome.result.status
            if status == OrderState.FILLED.value:
                summary["filled"] = int(summary["filled"]) + 1
            elif status == OrderState.PARTIALLY_FILLED.value:
                summary["partially_filled"] = int(summary["partially_filled"]) + 1
            elif status in {OrderState.REJECTED.value, OrderState.ERROR.value}:
                summary["rejected_or_error"] = int(summary["rejected_or_error"]) + 1
            else:
                summary["pending"] = int(summary["pending"]) + 1
            outcomes.append(
                {
                    "order_id": row.order_id,
                    "status": status,
                    "fill_id": outcome.fill.fill_id if outcome.fill else None,
                    "message": outcome.result.message,
                }
            )
        self.session.commit()
        summary["outcomes"] = outcomes
        return summary

    def _execution_market_error(
        self,
        row: SimulatedOrderRecord,
        market: MarketSnapshot,
        *,
        action: str,
        now: datetime,
    ) -> str | None:
        """Recheck volatile safety facts when a resting approval is finally matched."""

        if market.symbol.upper() != row.symbol.upper():
            return "Execution market symbol does not match the approved paper order."
        protective_close = action == "CLOSE"
        kill_state = self.session.scalar(
            select(RiskControlState).order_by(RiskControlState.updated_at.desc()).limit(1)
        )
        if not protective_close and (
            bool(getattr(settings, "risk_kill_switch_enabled", False))
            or bool(kill_state and kill_state.enabled)
        ):
            return "Emergency paper-risk kill switch blocks this resting exposure increase."
        observed = market.observed_at
        observed = observed.replace(tzinfo=timezone.utc) if observed.tzinfo is None else observed.astimezone(timezone.utc)
        if observed > now + timedelta(seconds=5):
            return "Execution market snapshot is future-dated."
        if (
            not protective_close
            and bool(getattr(settings, "risk_require_fresh_data", True))
            and int(getattr(settings, "risk_max_market_data_age_seconds", 180)) > 0
            and (now - observed).total_seconds()
            > int(getattr(settings, "risk_max_market_data_age_seconds", 180))
        ):
            return "Execution market snapshot is stale."
        bid, ask = self._book_prices(market)
        spread = max(ask - bid, 0.0) / max(float(market.price), 1e-12)
        max_spread = float(getattr(settings, "risk_max_spread_pct", 0.005))
        if not protective_close and max_spread > 0 and spread > max_spread:
            return "Execution spread exceeds the independent paper-risk limit."
        return None

    def cancel_order(self, order_id: str, *, reason: str = "operator_cancel") -> SimulatedOrderRecord:
        """Cancel one non-terminal simulated order with persisted state transitions."""
        row = self.session.scalar(
            select(SimulatedOrderRecord).where(SimulatedOrderRecord.order_id == order_id).limit(1)
        )
        if row is None:
            raise ValueError("simulated order does not exist")
        terminal = {
            OrderState.FILLED.value,
            OrderState.CANCELLED.value,
            OrderState.REJECTED.value,
            OrderState.EXPIRED.value,
            OrderState.ERROR.value,
        }
        if row.state in terminal:
            raise ValueError(f"cannot cancel terminal simulated order in state {row.state}")
        row.state = OrderState.CANCEL_PENDING.value
        row.updated_at = datetime.now(timezone.utc)
        row.payload = {**(row.payload or {}), "cancel_reason": reason[:200]}
        self.session.flush()
        row.state = OrderState.CANCELLED.value
        row.updated_at = datetime.now(timezone.utc)
        self.session.commit()
        return row

    def replace_limit_order(
        self,
        order_id: str,
        *,
        new_limit_price: float,
        expires_at: datetime | None = None,
    ) -> SimulatedOrderRecord:
        """Cancel a resting/partial order and acknowledge a bounded replacement.

        The replacement inherits the already-approved paper target and can only cover
        the unfilled remainder.  It does not create a new risk amount or touch a live
        venue. A later paper cycle/recovery pass may expire or reconcile it.
        """
        try:
            limit = float(new_limit_price)
        except (TypeError, ValueError) as exc:
            raise ValueError("replacement limit price must be numeric") from exc
        if not math.isfinite(limit) or limit <= 0:
            raise ValueError("replacement limit price must be positive and finite")
        original = self.session.scalar(
            select(SimulatedOrderRecord).where(SimulatedOrderRecord.order_id == order_id).limit(1)
        )
        if original is None:
            raise ValueError("simulated order does not exist")
        if original.state not in {
            OrderState.ACKNOWLEDGED.value,
            OrderState.PARTIALLY_FILLED.value,
            OrderState.SUBMITTED.value,
        }:
            raise ValueError("only an open simulated order can be replaced")
        filled = float(
            self.session.scalar(
                select(func.coalesce(func.sum(SimulatedFillRecord.quantity), 0.0)).where(
                    SimulatedFillRecord.order_id == original.order_id
                )
            )
            or 0.0
        )
        remaining = max(float(original.requested_quantity) - filled, 0.0)
        if remaining <= 1e-12:
            raise ValueError("simulated order has no unfilled quantity to replace")
        self.cancel_order(original.order_id, reason="cancel_replace")
        now = datetime.now(timezone.utc)
        ttl = max(
            int(getattr(settings, "paper_simulated_order_ttl_seconds", settings.v2_signal_ttl_seconds)),
            1,
        )
        replacement = SimulatedOrderRecord(
            order_id=new_id("order"),
            decision_trace_id=original.decision_trace_id,
            risk_decision_id=original.risk_decision_id,
            portfolio_target_id=original.portfolio_target_id,
            paper_account_id=original.paper_account_id,
            symbol=original.symbol,
            side=original.side,
            order_type="LIMIT",
            requested_quantity=remaining,
            requested_notional=remaining * limit,
            limit_price=limit,
            state=OrderState.ACKNOWLEDGED.value,
            client_order_id=f"{original.client_order_id}:replace:{new_id('r')[-12:]}",
            created_at=now,
            updated_at=now,
            expires_at=expires_at or (now + timedelta(seconds=ttl)),
            payload={
                "paper_only": True,
                "replaces_order_id": original.order_id,
                "remaining_quantity": remaining,
                "resting": True,
            },
        )
        self.session.add(replacement)
        self.session.commit()
        return replacement

    def recover_open_orders(self, *, now: datetime | None = None) -> dict[str, int]:
        """Recover persisted orders after restart and expire stale remainders."""
        now = now or datetime.now(timezone.utc)
        now = now.replace(tzinfo=timezone.utc) if now.tzinfo is None else now.astimezone(timezone.utc)
        open_states = {
            OrderState.CREATED.value,
            OrderState.RISK_APPROVED.value,
            OrderState.SUBMITTED.value,
            OrderState.ACKNOWLEDGED.value,
            OrderState.PARTIALLY_FILLED.value,
            OrderState.CANCEL_PENDING.value,
        }
        rows = list(
            self.session.scalars(
                select(SimulatedOrderRecord).where(SimulatedOrderRecord.state.in_(open_states))
            )
        )
        summary = {"inspected": len(rows), "recovered": 0, "expired": 0, "cancelled": 0, "reconciled_filled": 0}
        for row in rows:
            expires = row.expires_at
            if expires is not None:
                expires = expires.replace(tzinfo=timezone.utc) if expires.tzinfo is None else expires.astimezone(timezone.utc)
            if expires is not None and expires <= now:
                row.state = OrderState.EXPIRED.value
                row.updated_at = now
                summary["expired"] += 1
                continue
            if row.state == OrderState.CANCEL_PENDING.value:
                row.state = OrderState.CANCELLED.value
                row.updated_at = now
                summary["cancelled"] += 1
                continue
            filled = float(
                self.session.scalar(
                    select(func.coalesce(func.sum(SimulatedFillRecord.quantity), 0.0)).where(
                        SimulatedFillRecord.order_id == row.order_id
                    )
                )
                or 0.0
            )
            if filled >= float(row.requested_quantity) - 1e-12:
                row.state = OrderState.FILLED.value
                row.updated_at = now
                summary["reconciled_filled"] += 1
                continue
            if row.state in {
                OrderState.CREATED.value,
                OrderState.RISK_APPROVED.value,
                OrderState.SUBMITTED.value,
            }:
                approval = self.session.scalar(
                    select(RiskDecisionRecord).where(
                        RiskDecisionRecord.risk_decision_id == row.risk_decision_id,
                        RiskDecisionRecord.approved.is_(True),
                    ).limit(1)
                )
                row.state = OrderState.ACKNOWLEDGED.value if approval else OrderState.REJECTED.value
                row.updated_at = now
                summary["recovered"] += int(approval is not None)
            row.payload = {
                **(row.payload or {}),
                "remaining_quantity": max(float(row.requested_quantity) - filled, 0.0),
                "restart_reconciled_at": now.isoformat(),
            }
        self.session.commit()
        return summary

    def reconcile_account(
        self,
        account_id: str,
        *,
        mark_prices: dict[str, float] | None = None,
    ) -> dict[str, object]:
        """Recompute the paper cash/equity ledger and report lineage mismatches."""
        engine = PaperEngine(self.session, paper_account_id=account_id)
        latest = engine._latest_account(create=True, paper_account_id=account_id)
        snapshot = engine._record_equity(
            latest.cash_balance,
            price_by_symbol=mark_prices or {},
            paper_account_id=account_id,
        )
        positions = list(
            self.session.scalars(
                select(Position).where(Position.paper_account_id == account_id, Position.status == "OPEN")
            )
        )
        invalid_positions = [
            row.id
            for row in positions
            if not all(
                math.isfinite(float(value)) and float(value) >= 0
                for value in (row.quantity, row.margin_used, row.notional)
            )
        ]
        filled_orders = list(
            self.session.scalars(
                select(SimulatedOrderRecord).where(
                    SimulatedOrderRecord.paper_account_id == account_id,
                    SimulatedOrderRecord.state.in_((OrderState.FILLED.value, OrderState.PARTIALLY_FILLED.value)),
                )
            )
        )
        missing_fills = [
            row.order_id
            for row in filled_orders
            if self.session.scalar(
                select(SimulatedFillRecord.id).where(SimulatedFillRecord.order_id == row.order_id).limit(1)
            )
            is None
        ]
        missing_trades = [
            row.order_id
            for row in filled_orders
            if self.session.scalar(
                select(PaperTrade.id).where(PaperTrade.simulated_order_id == row.order_id).limit(1)
            )
            is None
        ]
        expected_equity = snapshot.cash_balance + sum(
            float(row.margin_used or 0.0) + float(row.unrealized_pnl or 0.0) for row in positions
        )
        return {
            "paper_only": True,
            "account_id": account_id,
            "cash_balance": snapshot.cash_balance,
            "equity": snapshot.equity,
            "expected_equity": expected_equity,
            "cash_reconciled": math.isclose(snapshot.equity, expected_equity, rel_tol=1e-9, abs_tol=1e-8),
            "position_count": len(positions),
            "invalid_position_ids": invalid_positions,
            "orders_missing_fills": missing_fills,
            "orders_missing_trades": missing_trades,
        }

    def _instruction(
        self,
        delta: float,
        *,
        current_exposure: float,
        requested_target_exposure: float,
        position: Position | None,
        price: float,
        equity: float,
    ) -> tuple[Direction, str, float, float]:
        # Cross-side changes are intentionally decomposed into a close in this pass;
        # a subsequent target cycle may open the other side after the reduction settles.
        if position and current_exposure * requested_target_exposure < 0:
            notional = position.quantity * price
            return (Direction.LONG if position.side.upper() == "LONG" else Direction.SHORT, "CLOSE", notional, position.quantity)
        if abs(requested_target_exposure) < abs(current_exposure):
            if position is None:
                return Direction.FLAT, "CLOSE", 0.0, 0.0
            fraction = min(abs(delta) / max(abs(current_exposure), 1e-9), 1.0)
            quantity = position.quantity * fraction
            return (Direction.LONG if position.side.upper() == "LONG" else Direction.SHORT, "CLOSE", quantity * price, quantity)
        side = Direction.LONG if delta > 0 else Direction.SHORT
        # Target exposure is a signed notional/equity fraction. Approved leverage is
        # applied later only when PaperEngine computes the required margin.
        notional = abs(delta) * max(equity, 0.0)
        return side, ("BUY" if side == Direction.LONG else "SELL"), notional, notional / price

    def _authorization_error(
        self,
        *,
        persisted: RiskDecisionRecord,
        persisted_target: PortfolioTargetRecord | None,
        target: PortfolioTarget,
        risk_decision: RiskDecision,
        market: MarketSnapshot,
        equity: float,
        account_id: str,
        decision_trace_id: str,
    ) -> str | None:
        """Bind an approval to its immutable trace, target, account and amounts."""
        if persisted_target is None:
            return "Execution rejected: the approved portfolio target was not found."
        if persisted.paper_account_id != account_id or persisted_target.paper_account_id != account_id:
            return "Execution rejected: risk decision account does not match paper execution account."
        if persisted.decision_trace_id != decision_trace_id or persisted_target.decision_trace_id != decision_trace_id:
            return "Execution rejected: risk decision trace does not match the approved target trace."
        if (
            persisted.portfolio_target_id != target.portfolio_target_id
            or risk_decision.portfolio_target_id != target.portfolio_target_id
        ):
            return "Execution rejected: risk approval is bound to a different portfolio target."
        normalized_symbol = target.symbol.upper()
        if normalized_symbol != persisted_target.symbol.upper() or normalized_symbol != market.symbol.upper():
            return "Execution rejected: risk approval is bound to a different symbol."
        payload_symbol = str((persisted.payload or {}).get("symbol") or "").upper()
        if payload_symbol and payload_symbol != normalized_symbol:
            return "Execution rejected: persisted risk evidence is bound to a different symbol."
        numeric_pairs = (
            (persisted.requested_exposure, risk_decision.requested_exposure),
            (persisted.approved_exposure, risk_decision.approved_exposure),
            (persisted.requested_leverage, risk_decision.requested_leverage),
            (persisted.approved_leverage, risk_decision.approved_leverage),
            (persisted_target.current_exposure, target.current_exposure),
            (persisted_target.requested_target_exposure, target.requested_target_exposure),
            (persisted_target.requested_delta, target.requested_delta),
            (persisted.requested_exposure, persisted_target.requested_target_exposure),
        )
        if any(not self._same_number(left, right) for left, right in numeric_pairs):
            return "Execution rejected: approved risk amounts do not match the persisted portfolio target."
        persisted_equity = (persisted.payload or {}).get("equity")
        if persisted_equity is not None and not self._same_number(persisted_equity, equity):
            return "Execution rejected: execution equity does not match the persisted risk snapshot."
        created_at = persisted.created_at
        created_at = created_at.replace(tzinfo=timezone.utc) if created_at.tzinfo is None else created_at.astimezone(timezone.utc)
        if datetime.now(timezone.utc) > created_at + timedelta(seconds=max(settings.v2_signal_ttl_seconds, 1)):
            return "Execution rejected: persisted risk approval has expired."
        return None

    @staticmethod
    def _same_number(left: float, right: float) -> bool:
        try:
            left_value = float(left)
            right_value = float(right)
        except (TypeError, ValueError):
            return False
        return math.isfinite(left_value) and math.isfinite(right_value) and math.isclose(
            left_value,
            right_value,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )

    @staticmethod
    def _persisted_equity(persisted: RiskDecisionRecord, *, fallback: float) -> float:
        try:
            value = float((persisted.payload or {}).get("equity"))
        except (TypeError, ValueError):
            value = float(fallback)
        return value if math.isfinite(value) and value >= 0 else 0.0

    @staticmethod
    def _execution_side(action: str, position: Position | None) -> Direction:
        if action == "BUY":
            return Direction.LONG
        if action == "SELL":
            return Direction.SHORT
        # Closing a long is a sell; closing a short is a buy.
        if position is not None and position.side.upper() == "SHORT":
            return Direction.LONG
        return Direction.SHORT

    @staticmethod
    def _bounded_participation() -> float:
        try:
            value = float(getattr(settings, "paper_simulated_volume_participation", 1.0))
        except (TypeError, ValueError):
            value = 0.0
        return max(min(value, 1.0), 0.0) if math.isfinite(value) else 0.0

    def _fill_fraction(
        self,
        requested_quantity: float,
        market: MarketSnapshot,
        *,
        already_partially_filled: bool = False,
    ) -> float:
        fraction = (
            0.5
            if getattr(settings, "paper_simulated_partial_fill_enabled", False) and not already_partially_filled
            else 1.0
        )
        if market.available_volume is None:
            return fraction
        try:
            available = max(float(market.available_volume), 0.0)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(available) or requested_quantity <= 0:
            return 0.0
        capacity = available * self._bounded_participation()
        return max(min(fraction, capacity / requested_quantity), 0.0)

    @staticmethod
    def _book_prices(market: MarketSnapshot) -> tuple[float, float]:
        half_spread = max(float(getattr(settings, "paper_simulated_spread_pct", 0.0)), 0.0) / 2.0
        bid = float(market.bid) if market.bid is not None and float(market.bid) > 0 else market.price * (1.0 - half_spread)
        ask = float(market.ask) if market.ask is not None and float(market.ask) > 0 else market.price * (1.0 + half_spread)
        return bid, ask

    def _limit_is_marketable(self, limit_price: float, market: MarketSnapshot, side: Direction) -> bool:
        bid, ask = self._book_prices(market)
        return limit_price >= ask if side == Direction.LONG else limit_price <= bid

    def _execution_price(
        self,
        market: MarketSnapshot,
        side: Direction,
        *,
        requested_quantity: float,
        limit_price: float | None,
    ) -> float:
        bid, ask = self._book_prices(market)
        reference = ask if side == Direction.LONG else bid
        slippage = max(float(getattr(settings, "paper_simulated_slippage_pct", 0.0)), 0.0)
        participation = 0.0
        if market.available_volume is not None:
            try:
                participation = requested_quantity / max(float(market.available_volume), 1e-12)
            except (TypeError, ValueError):
                participation = 0.0
        impact_coefficient = max(
            float(getattr(settings, "paper_simulated_market_impact_coefficient", 0.0)), 0.0
        )
        impact = slippage + impact_coefficient * min(max(participation, 0.0), 1.0) ** 0.5
        price = reference * (1.0 + impact if side == Direction.LONG else 1.0 - impact)
        if limit_price is not None:
            price = min(price, limit_price) if side == Direction.LONG else max(price, limit_price)
        return max(price, 1e-12)

    @staticmethod
    def _funding_rate(market: MarketSnapshot) -> float:
        raw = market.funding_rate
        if raw is None:
            raw = getattr(settings, "paper_simulated_funding_rate", 0.0)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return 0.0
        return max(value, 0.0) if math.isfinite(value) else 0.0

    def _persist_order(self, order: SimulatedOrder, trace_id: str) -> SimulatedOrderRecord:
        row = SimulatedOrderRecord(
            order_id=order.order_id,
            decision_trace_id=trace_id,
            risk_decision_id=order.risk_decision_id,
            portfolio_target_id=order.portfolio_target_id,
            paper_account_id=order.account_id,
            symbol=order.symbol,
            side=order.side.value,
            order_type=order.order_type,
            requested_quantity=order.requested_quantity,
            requested_notional=order.requested_notional,
            limit_price=order.limit_price,
            state=order.state.value,
            client_order_id=order.client_order_id,
            created_at=order.created_at,
            updated_at=order.updated_at,
            expires_at=order.expires_at,
            payload=order.metadata,
        )
        self.session.add(row)
        self.session.flush()
        return row

    @staticmethod
    def _transition(row: SimulatedOrderRecord, state: OrderState) -> None:
        row.state = state.value
        row.updated_at = datetime.now(timezone.utc)
