from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import desc, select

from app.ai.experience_buffer import record_experience, update_experience_rewards
from app.ai.strategy import RuleBasedStrategy, StrategyDecision
from app.config import settings
from app.db.models import AiDecision, Feature, PaperTrade, Position
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
        )
        self._rng = random.Random()
        self._task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None

    def status(self) -> dict[str, Any]:
        return self.state.as_dict()

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
        strategy_decision = RuleBasedStrategy().decide(feature)
        decision, decision_source = self._maybe_explore(session, symbol, strategy_decision)
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
                notional=settings.min_paper_trade_notional if decision_source == "exploration" and decision.action == "BUY" else None,
            )

        execution = {
            "status": execution_result.status,
            "message": execution_result.message,
            "trade_id": execution_result.trade_id,
            "balance": execution_result.balance,
            "equity": execution_result.equity,
        }
        execution["decision_source"] = decision_source
        execution["strategy_action"] = strategy_decision.action
        ai_decision = self._record_decision(
            session,
            symbol,
            feature,
            decision,
            execution_result,
            execution,
            decision_source=decision_source,
            strategy_decision=strategy_decision,
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
            "strategy_action": strategy_decision.action,
            **execution,
        }

    def _maybe_explore(
        self,
        session,
        symbol: str,
        strategy_decision: StrategyDecision,
    ) -> tuple[StrategyDecision, str]:
        if not settings.is_paper_mode or not settings.exploration_mode or settings.exploration_rate <= 0:
            return strategy_decision, "strategy"
        if self._rng.random() >= min(max(settings.exploration_rate, 0.0), 1.0):
            return strategy_decision, "strategy"

        existing_position = session.scalar(
            select(Position)
            .where(Position.symbol == symbol, Position.status == "OPEN")
            .order_by(desc(Position.opened_at))
            .limit(1)
        )
        if existing_position:
            action = self._rng.choice(["BUY", "SELL", "CLOSE"])
        else:
            action = "BUY"
        confidence = max(settings.risk_min_confidence, min(max(strategy_decision.confidence, 0.0), 0.75))
        reason = (
            f"Exploration paper action selected to collect action-result experience. "
            f"Original strategy wanted {strategy_decision.action}: {strategy_decision.reason}"
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
    ) -> AiDecision:
        ai_decision = AiDecision(
            symbol=symbol,
            strategy_name=RuleBasedStrategy.name if decision_source == "strategy" else "exploration-v1",
            source_name=self.name if decision_source == "strategy" else "auto-trader-exploration-v1",
            feature_id=feature.id,
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
                "strategy_decision": decision.model_dump() if decision_source == "strategy" else strategy_decision.model_dump(),
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

    def _duplicate_or_loss_cooldown(
        self,
        session,
        symbol: str,
        decision: StrategyDecision,
    ) -> ExecutionResult | None:
        if decision.action.upper() != "BUY":
            return None

        now = datetime.now(timezone.utc)
        duplicate_since = now - timedelta(seconds=max(settings.auto_trader_interval_seconds, 60))
        existing_long = session.scalar(
            select(Position)
            .where(Position.symbol == symbol, Position.status == "OPEN", Position.side == "LONG")
            .order_by(desc(Position.opened_at))
            .limit(1)
        )
        recent_buy = session.scalar(
            select(PaperTrade)
            .where(PaperTrade.symbol == symbol, PaperTrade.action == "BUY", PaperTrade.created_at >= duplicate_since)
            .order_by(desc(PaperTrade.created_at))
            .limit(1)
        )
        if existing_long and recent_buy:
            snapshot = PaperEngine(session).snapshot()
            return ExecutionResult(
                status="REJECTED",
                message="Duplicate long entry cooldown is active for this symbol.",
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
                snapshot = PaperEngine(session).snapshot()
                return ExecutionResult(
                    status="REJECTED",
                    message="Auto trader cooldown is active after a recent loss.",
                    balance=snapshot["cash_balance"],
                    equity=snapshot["equity"],
                )

        return None
