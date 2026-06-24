from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import desc, select

from app.ai.experience_buffer import record_experience, update_experience_rewards
from app.ai.model_strategy import ModelDecision, PriceModelStrategy
from app.ai.strategy import RuleBasedStrategy, StrategyDecision
from app.config import settings
from app.db.models import AiDecision, Candle, Feature, LiveCandleUpdate, ModelVersion, PaperTrade, Position
from app.db.session import SessionLocal
from app.features.feature_builder import FeatureBuilder
from app.trading.paper_engine import ExecutionResult, PaperEngine

logger = logging.getLogger(__name__)


@dataclass
class AutoTraderState:
    running: bool = False
    enabled: bool = False
    interval_seconds: int = 60
    symbols: list[str] = field(default_factory=list)
    cycles: int = 0
    decisions: int = 0
    strategy_trades: int = 0
    exploration_trades: int = 0
    skipped_trades: int = 0
    exploration_enabled: bool = False
    exploration_rate: float = 0.0
    last_run_at: str | None = None
    last_error: str | None = None
    last_skip_reason: str | None = None
    last_decision: dict[str, Any] | None = None
    model_strategy_enabled: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class AutoTraderService:
    name = "auto-trader-v1"

    def __init__(
        self,
        *,
        symbols: list[str] | None = None,
        interval_seconds: int | None = None,
    ) -> None:
        self.symbols = symbols or settings.auto_trader_symbols
        self.interval_seconds = max(interval_seconds or settings.auto_trader_interval_seconds, 1)
        self.state = AutoTraderState(
            enabled=settings.auto_trader_enabled,
            interval_seconds=self.interval_seconds,
            symbols=self.symbols,
            exploration_enabled=settings.exploration_mode and settings.is_paper_mode,
            exploration_rate=settings.exploration_rate,
            model_strategy_enabled=settings.auto_trader_use_trained_model,
        )
        self._rng = random.Random()
        self._task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None

    def status(self) -> dict[str, Any]:
        return {
            **self.state.as_dict(),
            "position_management": {
                "min_hold_seconds": settings.auto_min_hold_seconds,
                "take_profit_min_hold_seconds": settings.auto_take_profit_min_hold_seconds,
                "max_hold_seconds": settings.auto_max_hold_seconds,
                "max_loss_pct_of_margin": settings.auto_position_max_loss_pct,
                "default_stop_loss_pct": settings.auto_default_stop_loss_pct,
                "default_take_profit_pct": settings.auto_default_take_profit_pct,
                "fast_profit_exit_pct": settings.auto_fast_profit_exit_pct,
                "profit_close_min_net_pct": settings.auto_close_min_net_profit_pct,
                "confidence_leverage_enabled": settings.paper_confidence_leverage_enabled,
                "min_leverage": settings.paper_min_leverage,
                "max_leverage": settings.paper_max_leverage,
                "max_margin_allocation_pct": settings.risk_max_trade_size_pct,
                "max_entry_fee_pct_of_equity": settings.risk_max_entry_fee_pct_of_equity,
            },
        }

    async def start(self) -> dict[str, Any]:
        if self._task and not self._task.done():
            return self.status()
        self._stop_event = asyncio.Event()
        self.state.running = True
        self.state.enabled = True
        self.state.last_error = None
        self._task = asyncio.create_task(self._run_loop(self._stop_event), name="auto-trader")
        return self.status()

    async def stop(self) -> dict[str, Any]:
        if self._stop_event:
            self._stop_event.set()
        if self._task and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=5)
            except asyncio.TimeoutError:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
        self.state.running = False
        self.state.enabled = False
        return self.status()

    async def _run_loop(self, stop_event: asyncio.Event) -> None:
        try:
            while not stop_event.is_set():
                await asyncio.to_thread(self.run_once)
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=self.interval_seconds)
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive worker boundary
            logger.exception("Auto trader stopped after an unexpected error")
            self.state.last_error = str(exc)
        finally:
            self.state.running = False

    def run_once(self) -> dict[str, Any]:
        if not settings.is_paper_mode:
            self.state.last_error = "Auto trader is paper-only and trading mode is not paper."
            return self.status()

        cycle_decisions: list[dict[str, Any]] = []
        with SessionLocal() as session:
            for symbol in self.symbols:
                try:
                    cycle_decisions.append(self._run_symbol(session, symbol.upper()))
                except Exception as exc:
                    logger.exception("Auto trader symbol cycle failed for %s", symbol)
                    cycle_decisions.append({"symbol": symbol.upper(), "status": "ERROR", "message": str(exc)})
                    self.state.last_error = str(exc)
            PaperEngine(session).snapshot()
            update_experience_rewards(session)

        self.state.cycles += 1
        self.state.decisions += len(cycle_decisions)
        self.state.last_run_at = datetime.now(timezone.utc).isoformat()
        self.state.last_decision = cycle_decisions[-1] if cycle_decisions else None
        if cycle_decisions:
            self.state.last_error = None
        for item in cycle_decisions:
            if item.get("trade_id"):
                if item.get("decision_source") == "exploration":
                    self.state.exploration_trades += 1
                else:
                    self.state.strategy_trades += 1
            else:
                self.state.skipped_trades += 1
                self.state.last_skip_reason = item.get("message") or item.get("reason")
        return self.status()

    def _run_symbol(self, session, symbol: str) -> dict[str, Any]:
        feature = FeatureBuilder(session).build_for_symbol(
            symbol,
            interval=settings.paper_trade_timeframe,
            store=True,
        )
        fallback_decision = RuleBasedStrategy().decide(feature)
        model_decision = PriceModelStrategy().decide(session, feature) if settings.auto_trader_use_trained_model else None
        base_decision = model_decision.decision if model_decision else fallback_decision
        base_source = "model" if model_decision else "strategy"
        decision, decision_source = self._maybe_explore(session, symbol, base_decision, base_source)
        managed_decision = self._position_management_decision(session, symbol, decision)
        if managed_decision:
            decision, decision_source = managed_decision
        decision = self._fee_aware_close_decision(session, symbol, decision)
        duplicate_result = self._duplicate_or_loss_cooldown(session, symbol, decision)
        if duplicate_result:
            execution_result = duplicate_result
        else:
            execution_result = PaperEngine(session).execute_signal(
                symbol=symbol,
                action=decision.action,
                confidence=decision.confidence,
                reason=decision.reason,
                stop_loss=decision.stop_loss,
                take_profit=decision.take_profit,
                notional=settings.min_paper_trade_notional if decision_source == "exploration" and decision.action in {"BUY", "SELL"} else None,
            )

        execution = {
            "status": execution_result.status,
            "message": execution_result.message,
            "trade_id": execution_result.trade_id,
            "balance": execution_result.balance,
            "equity": execution_result.equity,
        }
        execution["decision_source"] = decision_source
        execution["strategy_action"] = fallback_decision.action
        ai_decision = self._record_decision(
            session,
            symbol,
            feature,
            decision,
            execution_result,
            execution,
            decision_source=decision_source,
            strategy_decision=fallback_decision,
            base_decision=base_decision,
            model_decision=model_decision,
        )
        return {
            "decision_id": ai_decision.id,
            "symbol": symbol,
            "feature_id": feature.id,
            "feature_schema_version": feature.schema_version,
            "action": decision.action,
            "confidence": decision.confidence,
            "reason": decision.reason,
            "decision_source": decision_source,
            "strategy_action": fallback_decision.action,
            "model_action": model_decision.decision.action if model_decision else None,
            "model_version_id": model_decision.model.id if model_decision else None,
            **execution,
        }

    def _maybe_explore(
        self,
        session,
        symbol: str,
        base_decision: StrategyDecision,
        base_source: str,
    ) -> tuple[StrategyDecision, str]:
        if not settings.is_paper_mode or not settings.exploration_mode or settings.exploration_rate <= 0:
            return base_decision, base_source
        if self._rng.random() >= min(max(settings.exploration_rate, 0.0), 1.0):
            return base_decision, base_source

        existing_position = session.scalar(
            select(Position)
            .where(Position.symbol == symbol, Position.status == "OPEN")
            .order_by(desc(Position.opened_at))
            .limit(1)
        )
        if existing_position:
            action = self._rng.choice(["BUY", "SELL", "CLOSE"])
        else:
            action = self._rng.choice(["BUY", "SELL"])
        confidence = max(settings.risk_min_confidence, min(max(base_decision.confidence, 0.0), 0.75))
        reason = (
            f"Exploration paper action selected to collect action-result experience. "
            f"Original {base_source} wanted {base_decision.action}: {base_decision.reason}"
        )
        return StrategyDecision(action=action, confidence=confidence, reason=reason), "exploration"

    def _record_decision(
        self,
        session,
        symbol: str,
        feature: Feature,
        decision: StrategyDecision,
        execution_result: ExecutionResult,
        execution: dict[str, Any],
        decision_source: str,
        strategy_decision: StrategyDecision,
        base_decision: StrategyDecision,
        model_decision: ModelDecision | None,
    ) -> AiDecision:
        strategy_name = RuleBasedStrategy.name
        source_name = self.name
        model_version: ModelVersion | None = None
        if decision_source == "exploration":
            strategy_name = "exploration-v1"
            source_name = "auto-trader-exploration-v1"
        elif decision_source == "model":
            model_version = model_decision.model if model_decision else None
            strategy_name = model_version.name if model_version else PriceModelStrategy.name
            source_name = "auto-trader-model-v1"
        elif decision_source == "risk-exit":
            strategy_name = "position-risk-manager-v1"
            source_name = "auto-trader-risk-exit-v1"
        elif decision_source == "position-management":
            strategy_name = "position-manager-v1"
            source_name = "auto-trader-position-manager-v1"
        ai_decision = AiDecision(
            symbol=symbol,
            strategy_name=strategy_name,
            source_name=source_name,
            feature_id=feature.id,
            model_version_id=model_version.id if model_version else None,
            feature_schema_version=feature.schema_version,
            action=decision.action,
            confidence=decision.confidence,
            reason=decision.reason,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            execution_status=execution_result.status,
            execution_message=execution_result.message,
            trade_id=execution_result.trade_id,
            raw={
                **decision.model_dump(),
                "decision_source": decision_source,
                "base_decision": base_decision.model_dump(),
                "strategy_decision": strategy_decision.model_dump(),
                "model_prediction": model_decision.prediction if model_decision else None,
                "exploration": {
                    "enabled": settings.exploration_mode,
                    "rate": settings.exploration_rate,
                    "min_notional": settings.min_paper_trade_notional,
                },
            },
            result=execution,
        )
        session.add(ai_decision)
        session.flush()
        if ai_decision.trade_id:
            trade = session.get(PaperTrade, ai_decision.trade_id)
            if trade:
                trade.raw_payload = {
                    **(trade.raw_payload or {}),
                    "decision_source": decision_source,
                    "ai_decision_id": ai_decision.id,
                    "auto_trader": self.name,
                }
        record_experience(session, decision=ai_decision, feature=feature, execution_result=execution)
        session.commit()
        session.refresh(ai_decision)
        return ai_decision

    def _position_management_decision(
        self,
        session,
        symbol: str,
        strategy_decision: StrategyDecision,
    ) -> tuple[StrategyDecision, str] | None:
        position = session.scalar(
            select(Position)
            .where(Position.symbol == symbol, Position.status == "OPEN")
            .order_by(desc(Position.opened_at))
            .limit(1)
        )
        if position is None:
            return None
        mark_price = self._latest_price(session, symbol) or position.current_price or position.entry_price
        position.current_price = mark_price
        gross_pnl = self._position_gross_pnl(position, mark_price)
        close_notional = position.quantity * mark_price
        close_fee = close_notional * settings.paper_fee_rate
        net_pnl = gross_pnl - close_fee
        margin = position.margin_used or ((position.quantity * position.entry_price) / max(position.leverage or settings.paper_leverage, 1.0))
        pnl_on_margin = net_pnl / margin if margin else 0.0
        net_profit_pct = net_pnl / close_notional if close_notional else 0.0
        now = datetime.now(timezone.utc)
        opened_at = position.opened_at
        if opened_at.tzinfo is None:
            opened_at = opened_at.replace(tzinfo=timezone.utc)
        age_seconds = max((now - opened_at).total_seconds(), 0.0)

        stop_loss = position.stop_loss or self._default_stop_loss(position.entry_price, position.side)
        take_profit = position.take_profit or self._default_take_profit(position.entry_price, position.side)
        if stop_loss and self._stop_loss_hit(position.side, mark_price, stop_loss):
            return (
                StrategyDecision(
                    action="CLOSE",
                    confidence=1.0,
                    reason=f"Risk exit: stop loss hit at {mark_price:.8f} <= {stop_loss:.8f}.",
                ),
                "risk-exit",
            )
        if settings.auto_position_max_loss_pct > 0 and pnl_on_margin <= -settings.auto_position_max_loss_pct:
            return (
                StrategyDecision(
                    action="CLOSE",
                    confidence=1.0,
                    reason=(
                        "Risk exit: max position loss hit "
                        f"({pnl_on_margin:.2%} of margin, net PnL ${net_pnl:.4f})."
                    ),
                ),
                "risk-exit",
            )
        if take_profit and self._take_profit_hit(position.side, mark_price, take_profit):
            if age_seconds >= settings.auto_take_profit_min_hold_seconds:
                return (
                    StrategyDecision(
                        action="CLOSE",
                        confidence=0.95,
                        reason=f"Take profit hit at {mark_price:.8f} >= {take_profit:.8f} after {age_seconds / 60:.1f} minutes.",
                    ),
                    "position-management",
                )
            if net_profit_pct >= settings.auto_fast_profit_exit_pct:
                return (
                    StrategyDecision(
                        action="CLOSE",
                        confidence=0.95,
                        reason=(
                            "Fast profit exit: take profit hit before minimum hold, "
                            f"but net profit {net_profit_pct:.2%} is strong enough."
                        ),
                    ),
                    "position-management",
                )
        if settings.auto_max_hold_seconds > 0 and age_seconds >= settings.auto_max_hold_seconds and net_pnl >= 0:
            return (
                StrategyDecision(
                    action="CLOSE",
                    confidence=0.80,
                    reason=f"Time exit: max hold reached with non-negative net PnL (${net_pnl:.4f}).",
                ),
                "position-management",
            )
        return None

    def _fee_aware_close_decision(
        self,
        session,
        symbol: str,
        decision: StrategyDecision,
    ) -> StrategyDecision:
        if decision.action.upper() not in {"SELL", "CLOSE"}:
            return decision
        existing_position = session.scalar(
            select(Position)
            .where(Position.symbol == symbol, Position.status == "OPEN")
            .order_by(desc(Position.opened_at))
            .limit(1)
        )
        if existing_position is None:
            return decision
        mark_price = self._latest_price(session, symbol) or existing_position.current_price or existing_position.entry_price
        gross_pnl = self._position_gross_pnl(existing_position, mark_price)
        close_notional = existing_position.quantity * mark_price
        close_fee = close_notional * settings.paper_fee_rate
        min_net_profit = close_notional * settings.auto_close_min_net_profit_pct
        net_profit = gross_pnl - close_fee
        net_profit_pct = net_profit / close_notional if close_notional else 0.0
        age_seconds = self._position_age_seconds(existing_position)
        if (
            gross_pnl > 0
            and age_seconds < settings.auto_min_hold_seconds
            and net_profit_pct < settings.auto_fast_profit_exit_pct
        ):
            return StrategyDecision(
                action="HOLD",
                confidence=max(decision.confidence, settings.risk_min_confidence),
                reason=(
                    "Weak profit close skipped because minimum hold time is active "
                    f"({age_seconds:.0f}/{settings.auto_min_hold_seconds}s, net profit {net_profit_pct:.2%})."
                ),
                stop_loss=decision.stop_loss,
                take_profit=decision.take_profit,
            )
        if gross_pnl > 0 and gross_pnl <= close_fee + min_net_profit:
            return StrategyDecision(
                action="HOLD",
                confidence=max(decision.confidence, settings.risk_min_confidence),
                reason=(
                    "Close skipped because gross profit is too small after fees "
                    f"(gross ${gross_pnl:.4f}, fee ${close_fee:.4f}, required net ${min_net_profit:.4f})."
                ),
                stop_loss=decision.stop_loss,
                take_profit=decision.take_profit,
            )
        return decision

    def _latest_price(self, session, symbol: str) -> float | None:
        candle = session.scalar(
            select(Candle).where(Candle.symbol == symbol).order_by(desc(Candle.open_time)).limit(1)
        )
        live_update = session.scalar(
            select(LiveCandleUpdate).where(LiveCandleUpdate.symbol == symbol).order_by(desc(LiveCandleUpdate.open_time)).limit(1)
        )
        if live_update and (candle is None or live_update.open_time >= candle.open_time):
            return live_update.close
        return candle.close if candle else None

    def _position_age_seconds(self, position: Position) -> float:
        opened_at = position.opened_at
        if opened_at.tzinfo is None:
            opened_at = opened_at.replace(tzinfo=timezone.utc)
        return max((datetime.now(timezone.utc) - opened_at).total_seconds(), 0.0)

    def _default_stop_loss(self, price: float, side: str) -> float | None:
        if settings.auto_default_stop_loss_pct <= 0:
            return None
        multiplier = 1.0 - settings.auto_default_stop_loss_pct if side.upper() == "LONG" else 1.0 + settings.auto_default_stop_loss_pct
        return price * multiplier

    def _default_take_profit(self, price: float, side: str) -> float | None:
        if settings.auto_default_take_profit_pct <= 0:
            return None
        multiplier = 1.0 + settings.auto_default_take_profit_pct if side.upper() == "LONG" else 1.0 - settings.auto_default_take_profit_pct
        return price * multiplier

    def _duplicate_or_loss_cooldown(
        self,
        session,
        symbol: str,
        decision: StrategyDecision,
    ) -> ExecutionResult | None:
        action = decision.action.upper()
        if action not in {"BUY", "SELL"}:
            return None
        existing_position = session.scalar(
            select(Position)
            .where(Position.symbol == symbol, Position.status == "OPEN")
            .order_by(desc(Position.opened_at))
            .limit(1)
        )
        target_side = "LONG" if action == "BUY" else "SHORT"
        if existing_position and existing_position.side.upper() != target_side:
            return None

        now = datetime.now(timezone.utc)
        duplicate_since = now - timedelta(seconds=max(settings.auto_trader_interval_seconds, 60))
        recent_buy = session.scalar(
            select(PaperTrade)
            .where(
                PaperTrade.symbol == symbol,
                PaperTrade.action == action,
                PaperTrade.side == target_side,
                PaperTrade.created_at >= duplicate_since,
            )
            .order_by(desc(PaperTrade.created_at))
            .limit(1)
        )
        if existing_position and recent_buy:
            snapshot = PaperEngine(session).snapshot()
            return ExecutionResult(
                status="REJECTED",
                message=f"Duplicate {target_side.lower()} entry cooldown is active for this symbol.",
                balance=snapshot["cash_balance"],
                equity=snapshot["equity"],
            )

        if settings.risk_cooldown_minutes > 0:
            loss_since = now - timedelta(minutes=settings.risk_cooldown_minutes)
            recent_loss = session.scalar(
                select(PaperTrade)
                .where(PaperTrade.created_at >= loss_since, PaperTrade.realized_pnl < 0)
                .order_by(desc(PaperTrade.created_at))
                .limit(1)
            )
            if recent_loss:
                loss_threshold = max(settings.min_paper_trade_notional * 0.02, 1.0)
                if abs(float(recent_loss.realized_pnl or 0.0)) >= loss_threshold:
                    snapshot = PaperEngine(session).snapshot()
                    return ExecutionResult(
                        status="REJECTED",
                        message=(
                            "Auto trader cooldown is active after a meaningful recent loss "
                            f"(${recent_loss.realized_pnl:.4f})."
                        ),
                        balance=snapshot["cash_balance"],
                        equity=snapshot["equity"],
                    )

        return None

    def _position_gross_pnl(self, position: Position, mark_price: float) -> float:
        if position.side.upper() == "SHORT":
            return (position.entry_price - mark_price) * position.quantity
        return (mark_price - position.entry_price) * position.quantity

    def _stop_loss_hit(self, side: str, mark_price: float, stop_loss: float) -> bool:
        return mark_price <= stop_loss if side.upper() == "LONG" else mark_price >= stop_loss

    def _take_profit_hit(self, side: str, mark_price: float, take_profit: float) -> bool:
        return mark_price >= take_profit if side.upper() == "LONG" else mark_price <= take_profit
