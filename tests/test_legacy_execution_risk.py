"""Execution-boundary regressions for retained paper compatibility callers."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from app.db.models import (
    AccountEquity,
    Base,
    Candle,
    EnsembleDecisionRecord,
    PaperTrade,
    PortfolioTargetRecord,
    RiskControlState,
    RiskDecisionRecord,
)
from app.trading.paper_engine import (
    COMPATIBILITY_RISK_CONFIGURATION_VERSION,
    PaperEngine,
)


class LegacyExecutionRiskBoundaryTests(unittest.TestCase):
    """Prove hostile compatibility payloads cannot skip universal account gates."""

    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory(prefix="anata-legacy-risk-")
        database_path = Path(self._temporary_directory.name) / "risk.sqlite3"
        self.engine = create_engine(
            f"sqlite:///{database_path.as_posix()}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self.engine)
        self.session = Session(self.engine)
        self.now = datetime.now(timezone.utc)
        self.account = AccountEquity(
            paper_account_id="champion",
            timestamp=self.now - timedelta(seconds=1),
            cash_balance=10_000.0,
            equity=10_000.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            drawdown=0.0,
        )
        self.session.add_all(
            [
                self.account,
                Candle(
                    exchange="test",
                    source_name="test",
                    symbol="BTCUSDT",
                    interval="1m",
                    open_time=self.now - timedelta(minutes=1),
                    close_time=self.now,
                    open=100.0,
                    high=101.0,
                    low=99.0,
                    close=100.0,
                    volume=1_000.0,
                    quote_volume=100_000.0,
                    trades=100,
                    is_closed=True,
                ),
            ]
        )
        self.session.commit()

    def tearDown(self) -> None:
        self.session.close()
        self.engine.dispose()
        self._temporary_directory.cleanup()

    @staticmethod
    def _risk_settings(**overrides: object) -> SimpleNamespace:
        values: dict[str, object] = {
            "risk_min_confidence": 0.10,
            "risk_kill_switch_enabled": False,
            "risk_require_fresh_data": True,
            "risk_max_market_data_age_seconds": 300,
            "risk_max_daily_loss_pct": 0.05,
            "risk_max_portfolio_drawdown_pct": 0.90,
            "risk_cooldown_minutes": 0,
            "risk_max_open_positions": 10,
            "paper_max_leverage": 3.0,
            "risk_max_portfolio_leverage": 3.0,
            "v2_max_position_leverage": 3.0,
            "risk_max_trade_size_pct": 0.10,
            "paper_fee_rate": 0.0004,
            "risk_max_entry_fee_pct_of_equity": 0.10,
            "risk_max_fee_exposure_pct": 0.10,
            "paper_confidence_leverage_enabled": False,
            "paper_leverage": 2.0,
            "paper_min_leverage": 1.0,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    @staticmethod
    def _paper_settings() -> SimpleNamespace:
        return SimpleNamespace(
            is_paper_mode=True,
            paper_start_balance=10_000.0,
            min_paper_trade_notional=1.0,
            paper_fee_rate=0.0004,
            paper_leverage=2.0,
            paper_max_leverage=3.0,
            auto_default_stop_loss_pct=0.0,
            auto_default_take_profit_pct=0.0,
        )

    def _execute_hostile_plan(self, *, risk_settings: SimpleNamespace, source: str):
        before_trades = self.session.scalar(select(func.count(PaperTrade.id))) or 0
        with patch("app.trading.risk_manager.settings", risk_settings), patch(
            "app.trading.paper_engine.settings", self._paper_settings()
        ):
            result = PaperEngine(self.session).execute_signal(
                symbol="BTCUSDT",
                action="BUY",
                confidence=0.99,
                reason="hostile model plan",
                stop_loss=1.0,
                take_profit=1_000_000.0,
                price=100.0,
                notional=1_000_000_000.0,
                leverage=999.0,
                margin_pct=0.99,
                compatibility_source=source,
            )
        after_trades = self.session.scalar(select(func.count(PaperTrade.id))) or 0
        self.assertEqual(before_trades, after_trades, "a rejected risk decision created a paper fill")
        self.assertEqual(result.status, "REJECTED")
        self.assertIsNotNone(result.risk_decision_id)
        self.assertIsNotNone(result.decision_trace_id)
        decision = self.session.scalar(
            select(RiskDecisionRecord).where(RiskDecisionRecord.risk_decision_id == result.risk_decision_id)
        )
        self.assertIsNotNone(decision)
        self.assertFalse(decision.approved)
        self.assertEqual(decision.configuration_version, COMPATIBILITY_RISK_CONFIGURATION_VERSION)
        self.assertEqual(decision.payload["source"], source)
        self.assertTrue(decision.payload["compatibility_risk_audit"])
        self.assertEqual(decision.payload["untrusted_request"]["requested_leverage"], 999.0)
        self.assertEqual(
            self.session.scalar(
                select(func.count(PortfolioTargetRecord.id)).where(
                    PortfolioTargetRecord.decision_trace_id == result.decision_trace_id
                )
            ),
            1,
        )
        self.assertEqual(
            self.session.scalar(
                select(func.count(EnsembleDecisionRecord.id)).where(
                    EnsembleDecisionRecord.decision_trace_id == result.decision_trace_id
                )
            ),
            1,
        )
        return result, decision

    def test_daily_loss_cannot_be_bypassed_by_model_plan(self) -> None:
        self.session.add(
            PaperTrade(
                symbol="ETHUSDT",
                paper_account_id="champion",
                action="SELL",
                side="LONG",
                quantity=1.0,
                price=100.0,
                notional=100.0,
                realized_pnl=-600.0,
                status="FILLED",
                created_at=self.now - timedelta(minutes=1),
            )
        )
        self.session.commit()

        result, decision = self._execute_hostile_plan(
            risk_settings=self._risk_settings(risk_cooldown_minutes=0),
            source="api.paper-trade",
        )

        self.assertIn("MAX_DAILY_LOSS_REACHED", result.rejection_reasons)
        self.assertIn("MAX_DAILY_LOSS", decision.triggered_limits)

    def test_drawdown_cannot_be_bypassed_by_model_plan(self) -> None:
        self.account.drawdown = 0.25
        self.session.commit()

        result, decision = self._execute_hostile_plan(
            risk_settings=self._risk_settings(risk_max_portfolio_drawdown_pct=0.10),
            source="api.strategy-paper",
        )

        self.assertIn("MAX_PORTFOLIO_DRAWDOWN_REACHED", result.rejection_reasons)
        self.assertIn("MAX_PORTFOLIO_DRAWDOWN", decision.triggered_limits)

    def test_recent_loss_cooldown_cannot_be_bypassed_by_model_plan(self) -> None:
        self.session.add(
            PaperTrade(
                symbol="ETHUSDT",
                paper_account_id="champion",
                action="SELL",
                side="LONG",
                quantity=1.0,
                price=100.0,
                notional=100.0,
                realized_pnl=-600.0,
                status="FILLED",
                created_at=self.now - timedelta(minutes=1),
            )
        )
        self.session.commit()

        result, decision = self._execute_hostile_plan(
            risk_settings=self._risk_settings(
                risk_max_daily_loss_pct=0.10,
                risk_cooldown_minutes=30,
            ),
            source="api.signal",
        )

        self.assertNotIn("MAX_DAILY_LOSS_REACHED", result.rejection_reasons)
        self.assertIn("COOLDOWN_ACTIVE", result.rejection_reasons)
        self.assertIn("COOLDOWN", decision.triggered_limits)

    def test_persisted_kill_switch_cannot_be_bypassed_by_model_plan(self) -> None:
        self.session.add(
            RiskControlState(
                enabled=True,
                reason="operator emergency stop",
                updated_by="test",
                updated_at=self.now,
            )
        )
        self.session.commit()

        result, decision = self._execute_hostile_plan(
            risk_settings=self._risk_settings(),
            source="auto-trader.legacy.model",
        )

        self.assertIn("KILL_SWITCH_ACTIVE", result.rejection_reasons)
        self.assertIn("KILL_SWITCH", decision.triggered_limits)
        self.assertTrue(decision.kill_switch_state)

    def test_approved_compatibility_fill_links_to_persisted_audit(self) -> None:
        with patch("app.trading.risk_manager.settings", self._risk_settings()), patch(
            "app.trading.paper_engine.settings", self._paper_settings()
        ):
            result = PaperEngine(self.session).execute_signal(
                symbol="BTCUSDT",
                action="BUY",
                confidence=0.90,
                reason="bounded compatibility order",
                price=100.0,
                notional=100.0,
                compatibility_source="api.signal",
            )

        self.assertEqual(result.status, "FILLED")
        decision = self.session.scalar(
            select(RiskDecisionRecord).where(RiskDecisionRecord.risk_decision_id == result.risk_decision_id)
        )
        trade = self.session.get(PaperTrade, result.trade_id)
        self.assertIsNotNone(decision)
        self.assertTrue(decision.approved)
        self.assertIsNotNone(trade)
        self.assertEqual(trade.risk_decision_id, result.risk_decision_id)
        self.assertEqual(trade.decision_trace_id, result.decision_trace_id)

    def test_audit_persistence_failure_rejects_before_fill(self) -> None:
        with patch("app.trading.risk_manager.settings", self._risk_settings()), patch(
            "app.trading.paper_engine.settings", self._paper_settings()
        ), patch.object(PaperEngine, "_persist_compatibility_risk_audit", return_value=None):
            result = PaperEngine(self.session).execute_signal(
                symbol="BTCUSDT",
                action="BUY",
                confidence=0.90,
                price=100.0,
                notional=100.0,
                compatibility_source="api.paper-trade",
            )

        self.assertEqual(result.status, "REJECTED")
        self.assertIn("failed closed", result.message)
        self.assertIn("RISK_AUDIT_PERSISTENCE_FAILED", result.rejection_reasons)
        self.assertEqual(self.session.scalar(select(func.count(PaperTrade.id))), 0)


if __name__ == "__main__":
    unittest.main()
