"""Focused regression contracts for the paper-only Anata V2 pipeline.

These tests use a fresh SQLite database and temporary artifact directory per test.
They intentionally exercise only local deterministic code; no collector, broker, or
paid/external service is needed.
"""

from __future__ import annotations

import base64
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.orm import Session

from app.api.vision import router as vision_api_router
from app.dashboard.routes import router as dashboard_router
from app.db.models import (
    AccountEquity,
    Base,
    Candle,
    EnsembleDecisionRecord,
    ExternalAIRequest,
    ModelVersion,
    PaperSandboxAccount,
    PaperTrade,
    PortfolioTargetRecord,
    Position,
    RiskControlState,
    RiskDecisionRecord,
    SimulatedFillRecord,
    SimulatedOrderRecord,
)
from app.db.migrations import run_additive_migrations
from app.db.session import get_session
from app.pipeline.data_quality import PointInTimeValidator
from app.pipeline.domain import (
    Direction,
    ModelLifecycle,
    ModelPrediction,
    PortfolioTarget,
    RiskDecision,
    TradingSignal,
    new_id,
)
from app.pipeline.ensemble import DeterministicRegimeEnsemble
from app.pipeline.execution import PaperExecutionSimulator
from app.pipeline.registry import ModelRegistry
from app.pipeline.risk import MarketSnapshot, PortfolioRiskEngine, RiskInputs, RiskPolicy
from app.trading.paper_engine import PaperEngine
from app.trading.risk_manager import RiskManager


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class IsolatedSqliteCase(unittest.TestCase):
    """Create a disposable on-disk SQLite ledger for each V2 contract."""

    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory(prefix="anata-v2-tests-")
        self.temp_dir = Path(self._temporary_directory.name)
        database_path = self.temp_dir / "v2-contracts.sqlite3"
        self.engine = create_engine(
            f"sqlite:///{database_path.as_posix()}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self.engine)
        self.session = Session(self.engine)

    def tearDown(self) -> None:
        self.session.close()
        self.engine.dispose()
        self._temporary_directory.cleanup()

    @staticmethod
    def _target(
        *,
        requested: float = 0.25,
        current: float = 0.0,
        symbol: str = "BTCUSDT",
    ) -> PortfolioTarget:
        return PortfolioTarget(
            symbol=symbol,
            current_exposure=current,
            requested_target_exposure=requested,
            requested_delta=requested - current,
            expected_return=0.01,
            expected_risk=0.02,
            risk_contribution=0.01,
            urgency=0.5,
            source_ensemble_decision_id=new_id("ens"),
        )

    @staticmethod
    def _policy(**overrides: object) -> RiskPolicy:
        values: dict[str, object] = {
            "min_confidence": 0.10,
            "max_position_leverage": 2.0,
            "max_margin_allocation_pct": 0.90,
            "max_entry_fee_pct_of_equity": 0.50,
            "max_daily_loss_pct": 0.05,
            "max_portfolio_drawdown_pct": 0.99,
            "max_open_positions": 10,
            "cooldown_minutes": 0,
            "max_market_data_age_seconds": 300,
            "max_symbol_exposure_pct": 0.90,
            "max_gross_exposure_pct": 0.90,
            "max_net_exposure_pct": 0.90,
            "max_expected_cost_pct": 0.50,
            "min_liquidity_score": 0.10,
            "kill_switch_enabled": False,
            "configuration_version": "v2-contract-test",
        }
        values.update(overrides)
        return RiskPolicy(**values)  # type: ignore[arg-type]

    @staticmethod
    def _inputs(
        now: datetime,
        *,
        account_id: str = "champion",
        equity: float = 10_000.0,
        cash_balance: float | None = None,
        observed_at: datetime | None = None,
        **overrides: object,
    ) -> RiskInputs:
        values: dict[str, object] = {
            "account_id": account_id,
            "cash_balance": equity if cash_balance is None else cash_balance,
            "equity": equity,
            "market": MarketSnapshot(symbol="BTCUSDT", price=100.0, observed_at=observed_at or now),
            "confidence": 0.90,
            "liquidity_score": 0.90,
            "expected_cost": 0.001,
            "expected_volatility": 0.02,
            "current_gross_exposure": 0.0,
            "current_net_exposure": 0.0,
            "now": now,
        }
        values.update(overrides)
        return RiskInputs(**values)  # type: ignore[arg-type]

    def _persist_target(self, target: PortfolioTarget, *, account_id: str = "champion", trace_id: str | None = None) -> str:
        """Persist the upstream trace links that a production risk record expects."""
        trace_id = trace_id or new_id("trace")
        now = _utc_now()
        self.session.add(
            EnsembleDecisionRecord(
                ensemble_decision_id=target.source_ensemble_decision_id,
                decision_trace_id=trace_id,
                symbol=target.symbol,
                generated_at=now,
                valid_until=now + timedelta(minutes=5),
                combined_expected_return=target.expected_return,
                combined_expected_volatility=target.expected_risk,
                combined_uncertainty=0.20,
                combined_confidence=0.80,
                current_regime="test",
                supporting_signals=[],
                conflicting_signals=[],
                signal_weights={},
                correlation_penalty=0.0,
                transaction_cost_penalty=0.0,
                regime_penalty=0.0,
                external_context_adjustment=0.0,
                decision_status="ACTIONABLE",
                reason_codes=["TEST"],
                payload={"symbol": target.symbol, "equity": 10_000.0, "cash_balance": 10_000.0},
            )
        )
        self.session.add(
            PortfolioTargetRecord(
                portfolio_target_id=target.portfolio_target_id,
                decision_trace_id=trace_id,
                paper_account_id=account_id,
                symbol=target.symbol,
                current_exposure=target.current_exposure,
                requested_target_exposure=target.requested_target_exposure,
                requested_delta=target.requested_delta,
                expected_return=target.expected_return,
                expected_risk=target.expected_risk,
                risk_contribution=target.risk_contribution,
                urgency=target.urgency,
                source_ensemble_decision_id=target.source_ensemble_decision_id,
                created_at=target.created_at,
                payload={},
            )
        )
        self.session.flush()
        return trace_id

    def _persist_risk(
        self,
        decision: RiskDecision,
        *,
        target: PortfolioTarget,
        account_id: str,
        trace_id: str,
    ) -> None:
        self.session.add(
            RiskDecisionRecord(
                risk_decision_id=decision.risk_decision_id,
                decision_trace_id=trace_id,
                portfolio_target_id=target.portfolio_target_id,
                paper_account_id=account_id,
                approved=decision.approved,
                requested_exposure=decision.requested_exposure,
                approved_exposure=decision.approved_exposure,
                requested_leverage=decision.requested_leverage,
                approved_leverage=decision.approved_leverage,
                triggered_limits=decision.triggered_limits,
                rejection_reasons=decision.rejection_reasons,
                configuration_version=decision.configuration_version,
                kill_switch_state=decision.kill_switch_state,
                created_at=decision.created_at,
                payload={"symbol": target.symbol, "equity": 10_000.0, "cash_balance": 10_000.0},
            )
        )
        self.session.flush()

    def _risk_approval(
        self,
        target: PortfolioTarget,
        inputs: RiskInputs,
        policy: RiskPolicy,
        *,
        trace_id: str | None = None,
        paper_fee_rate: float = 0.0004,
    ) -> RiskDecision:
        trace_id = trace_id or self.session.scalar(
            select(PortfolioTargetRecord.decision_trace_id).where(
                PortfolioTargetRecord.portfolio_target_id == target.portfolio_target_id
            )
        )
        trace_id = trace_id or self._persist_target(target, account_id=inputs.account_id)
        risk_settings = SimpleNamespace(min_paper_trade_notional=1.0, paper_fee_rate=paper_fee_rate)
        with patch("app.pipeline.risk.settings", risk_settings):
            return PortfolioRiskEngine(self.session, policy=policy).approve(target, inputs, decision_trace_id=trace_id)


class V2DomainAndQualityTests(IsolatedSqliteCase):
    def test_model_prediction_rejects_top_level_execution_controls(self) -> None:
        now = _utc_now()
        payload = {
            "model_id": "momentum-short",
            "model_version": "v1",
            "model_family": "momentum",
            "symbol": "BTCUSDT",
            "generated_at": now,
            "valid_from": now,
            "expires_at": now + timedelta(minutes=5),
            "forecast_horizon_seconds": 300,
            "expected_return": 0.01,
            "expected_volatility": 0.02,
            "probability_up": 0.60,
            "probability_down": 0.30,
            "confidence": 0.80,
            "feature_schema_version": "test-v1",
            "feature_snapshot_id": "feature_test",
        }

        for forbidden_field, value in {
            "leverage": 125.0,
            "margin_pct": 0.99,
            "notional": 1_000_000.0,
            "order_action": "BUY",
        }.items():
            with self.subTest(forbidden_field=forbidden_field):
                self.assertNotIn(forbidden_field, ModelPrediction.model_fields)
                with self.assertRaises(ValidationError):
                    ModelPrediction(**payload, **{forbidden_field: value})

    def test_point_in_time_quality_reports_market_and_news_defects(self) -> None:
        now = _utc_now().replace(microsecond=0)
        validator = PointInTimeValidator()
        candle_report = validator.validate_candles(
            [
                {"open_time": now - timedelta(minutes=5), "open": 100, "high": 101, "low": 99, "close": 100, "volume": 2},
                {"open_time": now - timedelta(minutes=5), "open": 100, "high": 101, "low": 99, "close": 100, "volume": 2},
                {"open_time": now - timedelta(minutes=1), "open": 100, "high": 90, "low": 95, "close": 101, "volume": -1},
                {"open_time": now + timedelta(minutes=10), "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
            ],
            interval="1m",
            now=now,
        )
        codes = {issue.code for issue in candle_report.issues}
        self.assertFalse(candle_report.valid)
        self.assertTrue(
            {
                "DUPLICATE_CANDLE",
                "NON_MONOTONIC_TIMESTAMP",
                "MISSING_CANDLE_INTERVAL",
                "INVALID_OHLC",
                "NEGATIVE_VOLUME",
                "FUTURE_TIMESTAMP",
            }.issubset(codes),
            codes,
        )

        news_report = validator.validate_news_available(
            {"available_to_model_time": now + timedelta(seconds=1)},
            decision_time=now,
        )
        self.assertFalse(news_report.valid)
        self.assertEqual([issue.code for issue in news_report.issues], ["FUTURE_NEWS_LEAKAGE"])

    def test_correlated_signals_receive_less_weight_and_a_penalty(self) -> None:
        now = _utc_now()

        def signal(signal_id: str, family: str) -> TradingSignal:
            return TradingSignal(
                signal_id=signal_id,
                prediction_id=f"pred_{signal_id}",
                signal_family=family,
                symbol="BTCUSDT",
                generated_at=now,
                valid_until=now + timedelta(minutes=5),
                direction=Direction.LONG,
                strength=0.80,
                expected_return=0.02,
                expected_cost=0.001,
                net_expected_return=0.019,
                confidence=0.80,
                uncertainty=0.10,
                liquidity_score=0.90,
                metadata={"expected_volatility": 0.02},
            )

        first, second = signal("sig_a", "momentum"), signal("sig_b", "breakout")
        ensemble = DeterministicRegimeEnsemble(correlation_threshold=0.70)
        independent = ensemble.combine("BTCUSDT", [first, second])
        correlated = ensemble.combine(
            "BTCUSDT",
            [first, second],
            correlations={(first.signal_id, second.signal_id): 0.95},
        )

        self.assertEqual(independent.decision.correlation_penalty, 0.0)
        self.assertGreater(correlated.decision.correlation_penalty, 0.0)
        self.assertLess(
            correlated.decision.signal_weights[second.signal_id],
            independent.decision.signal_weights[second.signal_id],
        )
        self.assertIn("CORRELATED_SIGNAL_WEIGHT_REDUCED", correlated.decision.reason_codes)


class V2RiskControlRegressionTests(IsolatedSqliteCase):
    """Every exposure-increasing target must pass global gates, independent of model wishes."""

    def test_max_open_positions_cannot_be_bypassed(self) -> None:
        now = _utc_now()
        target = self._target()
        self._persist_target(target)
        self.session.add(Position(symbol="ETHUSDT", paper_account_id="champion", status="OPEN", quantity=1.0, entry_price=100.0))
        self.session.flush()

        decision = self._risk_approval(target, self._inputs(now), self._policy(max_open_positions=1))
        self.assertFalse(decision.approved)
        self.assertIn("MAX_OPEN_POSITIONS_REACHED", decision.rejection_reasons)

    def test_daily_loss_cannot_be_bypassed(self) -> None:
        now = _utc_now()
        target = self._target()
        self._persist_target(target)
        self.session.add(
            PaperTrade(
                symbol="ETHUSDT",
                paper_account_id="champion",
                action="SELL",
                realized_pnl=-600.0,
                created_at=now,
            )
        )
        self.session.flush()

        decision = self._risk_approval(target, self._inputs(now), self._policy(max_daily_loss_pct=0.05))
        self.assertFalse(decision.approved)
        self.assertIn("MAX_DAILY_LOSS_REACHED", decision.rejection_reasons)

    def test_cooldown_cannot_be_bypassed(self) -> None:
        now = _utc_now()
        target = self._target()
        self._persist_target(target)
        self.session.add(
            PaperTrade(
                symbol="ETHUSDT",
                paper_account_id="champion",
                action="SELL",
                realized_pnl=-300.0,
                created_at=now - timedelta(minutes=1),
            )
        )
        self.session.flush()

        decision = self._risk_approval(
            target,
            self._inputs(now, equity=1_000.0),
            self._policy(max_daily_loss_pct=0.50, cooldown_minutes=30),
        )
        self.assertFalse(decision.approved)
        self.assertIn("COOLDOWN_ACTIVE", decision.rejection_reasons)

    def test_maximum_leverage_is_selected_by_risk_not_model_metadata(self) -> None:
        now = _utc_now()
        target = self._target(requested=0.20)
        self._persist_target(target)

        decision = self._risk_approval(target, self._inputs(now), self._policy(max_position_leverage=2.0))
        self.assertTrue(decision.approved)
        self.assertEqual(decision.requested_leverage, 2.0)
        self.assertEqual(decision.approved_leverage, 2.0)

    def test_maximum_margin_allocation_is_capped(self) -> None:
        now = _utc_now()
        target = self._target(requested=0.80)
        self._persist_target(target)

        decision = self._risk_approval(
            target,
            self._inputs(now),
            self._policy(max_margin_allocation_pct=0.20),
        )
        self.assertTrue(decision.approved)
        self.assertAlmostEqual(decision.approved_exposure, 0.20)
        self.assertIn("MAX_MARGIN_ALLOCATION", decision.triggered_limits)

    def test_fee_exposure_is_resized_not_bypassed(self) -> None:
        now = _utc_now()
        target = self._target(requested=0.50)
        self._persist_target(target)

        decision = self._risk_approval(
            target,
            self._inputs(now),
            self._policy(max_entry_fee_pct_of_equity=0.0001),
            paper_fee_rate=0.01,
        )
        self.assertTrue(decision.approved)
        self.assertLess(abs(decision.approved_exposure), abs(target.requested_target_exposure))
        self.assertIn("MAX_FEE_EXPOSURE", decision.triggered_limits)

    def test_correlated_cluster_exposure_is_independently_resized_and_invalid_evidence_rejects(self) -> None:
        now = _utc_now()
        target = self._target(requested=0.10)
        self._persist_target(target)
        policy = self._policy(max_correlated_cluster_exposure_pct=0.25)

        resized = self._risk_approval(
            target,
            self._inputs(
                now,
                current_gross_exposure=0.22,
                current_net_exposure=0.22,
                current_correlated_cluster_exposure=0.22,
                correlated_cluster_id="crypto-beta",
            ),
            policy,
        )
        self.assertTrue(resized.approved)
        self.assertAlmostEqual(resized.approved_exposure, 0.03, places=8)
        self.assertIn("MAX_CORRELATED_CLUSTER_EXPOSURE", resized.triggered_limits)

        compatibility_target = self._target(requested=0.10, current=0.05, symbol="SOLUSDT")
        self._persist_target(compatibility_target)
        compatibility = self._risk_approval(
            compatibility_target,
            self._inputs(
                now,
                market=MarketSnapshot(symbol="SOLUSDT", price=100.0, observed_at=now),
            ),
            policy,
        )
        self.assertTrue(compatibility.approved)
        self.assertNotIn("INCONSISTENT_CORRELATED_CLUSTER_EXPOSURE", compatibility.rejection_reasons)

        invalid_target = self._target(requested=0.10, symbol="ETHUSDT")
        self._persist_target(invalid_target)
        invalid = self._risk_approval(
            invalid_target,
            self._inputs(
                now,
                market=MarketSnapshot(symbol="ETHUSDT", price=100.0, observed_at=now),
                current_correlated_cluster_exposure=float("nan"),
                correlated_cluster_id="crypto-beta",
            ),
            policy,
        )
        self.assertFalse(invalid.approved)
        self.assertIn("INVALID_CORRELATED_CLUSTER_EXPOSURE", invalid.rejection_reasons)
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            self._policy(max_correlated_cluster_exposure_pct=1.01)

    def test_stale_market_data_cannot_be_bypassed(self) -> None:
        now = _utc_now()
        target = self._target()
        self._persist_target(target)

        decision = self._risk_approval(
            target,
            self._inputs(now, observed_at=now - timedelta(seconds=301)),
            self._policy(max_market_data_age_seconds=300),
        )
        self.assertFalse(decision.approved)
        self.assertIn("STALE_MARKET_DATA", decision.rejection_reasons)

    def test_persisted_kill_switch_cannot_be_bypassed(self) -> None:
        now = _utc_now()
        target = self._target()
        self._persist_target(target)
        self.session.add(RiskControlState(enabled=True, reason="test kill", updated_by="test", updated_at=now))
        self.session.flush()

        decision = self._risk_approval(target, self._inputs(now), self._policy())
        self.assertFalse(decision.approved)
        self.assertTrue(decision.kill_switch_state)
        self.assertIn("KILL_SWITCH_ACTIVE", decision.rejection_reasons)

    @staticmethod
    def _legacy_risk_settings(**overrides: object) -> SimpleNamespace:
        """Minimal deterministic configuration for legacy model-plan regression coverage."""
        values: dict[str, object] = {
            "risk_min_confidence": 0.10,
            "risk_kill_switch_enabled": False,
            "risk_require_fresh_data": True,
            "risk_max_market_data_age_seconds": 60,
            "risk_max_daily_loss_pct": 0.05,
            "risk_max_portfolio_drawdown_pct": 0.90,
            "risk_cooldown_minutes": 30,
            "risk_max_open_positions": 1,
            "paper_max_leverage": 3.0,
            "risk_max_portfolio_leverage": 3.0,
            "v2_max_position_leverage": 3.0,
            "risk_max_trade_size_pct": 0.10,
            "paper_fee_rate": 0.01,
            "risk_max_entry_fee_pct_of_equity": 0.0001,
            "risk_max_fee_exposure_pct": 0.0001,
            "paper_confidence_leverage_enabled": True,
            "paper_leverage": 2.0,
            "paper_min_leverage": 1.0,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_legacy_model_plan_hints_still_hit_every_global_gate(self) -> None:
        """Regression for the former AI-plan early-return path in ``RiskManager``."""
        now = _utc_now()
        self.session.add_all(
            [
                Position(symbol="ETHUSDT", paper_account_id="champion", status="OPEN", quantity=1.0, entry_price=100.0),
                PaperTrade(
                    symbol="ETHUSDT",
                    paper_account_id="champion",
                    action="SELL",
                    realized_pnl=-1_000.0,
                    created_at=now - timedelta(minutes=1),
                ),
                RiskControlState(enabled=True, reason="test kill", updated_by="test", updated_at=now),
            ]
        )
        self.session.flush()
        with patch("app.trading.risk_manager.settings", self._legacy_risk_settings()):
            result = RiskManager(self.session).evaluate(
                action="BUY",
                confidence=0.90,
                cash_balance=10_000.0,
                equity=10_000.0,
                requested_notional=1_000_000.0,
                existing_position=None,
                requested_leverage=999.0,
                requested_margin_pct=0.99,
                market_observed_at=now - timedelta(seconds=120),
                paper_account_id="champion",
            )

        self.assertFalse(result.accepted)
        self.assertTrue(
            {
                "KILL_SWITCH_ACTIVE",
                "STALE_MARKET_DATA",
                "MAX_DAILY_LOSS_REACHED",
                "COOLDOWN_ACTIVE",
                "MAX_OPEN_POSITIONS_REACHED",
                "REQUESTED_LEVERAGE_EXCEEDS_MAXIMUM",
            }.issubset(set(result.rejection_reasons)),
            result,
        )
        self.assertIn("MAX_MARGIN_ALLOCATION", result.triggered_limits)

    def test_legacy_model_plan_hints_cannot_escape_the_fee_cap(self) -> None:
        now = _utc_now()
        with patch("app.trading.risk_manager.settings", self._legacy_risk_settings()):
            result = RiskManager(self.session).evaluate(
                action="BUY",
                confidence=0.90,
                cash_balance=10_000.0,
                equity=10_000.0,
                requested_notional=None,
                existing_position=None,
                requested_leverage=3.0,
                requested_margin_pct=0.99,
                market_observed_at=now,
                paper_account_id="champion",
            )

        self.assertTrue(result.accepted)
        self.assertLessEqual(result.leverage, 3.0)
        self.assertLessEqual(result.max_notional, 100.0)  # $1 fee cap at a 1% simulated fee rate.
        self.assertIn("MAX_MARGIN_ALLOCATION", result.triggered_limits)
        self.assertIn("MAX_FEE_EXPOSURE", result.triggered_limits)


class V2ExecutionAndIsolationTests(IsolatedSqliteCase):
    @staticmethod
    def _execution_settings() -> tuple[SimpleNamespace, SimpleNamespace]:
        execution_settings = SimpleNamespace(
            is_paper_mode=True,
            v2_signal_ttl_seconds=300,
            paper_simulated_partial_fill_enabled=False,
            paper_simulated_spread_pct=0.0,
            paper_simulated_slippage_pct=0.0,
        )
        paper_engine_settings = SimpleNamespace(
            is_paper_mode=True,
            paper_start_balance=10_000.0,
            min_paper_trade_notional=1.0,
            paper_fee_rate=0.0004,
            paper_leverage=2.0,
            paper_max_leverage=2.0,
            auto_default_stop_loss_pct=0.0,
            auto_default_take_profit_pct=0.0,
        )
        return execution_settings, paper_engine_settings

    def test_execution_requires_persisted_approved_risk_and_accepts_one(self) -> None:
        target = self._target(requested=0.10)
        trace_id = self._persist_target(target)
        now = _utc_now()
        market = MarketSnapshot(symbol=target.symbol, price=100.0, observed_at=now)
        approved = RiskDecision(
            portfolio_target_id=target.portfolio_target_id,
            approved=True,
            requested_exposure=target.requested_target_exposure,
            approved_exposure=target.requested_target_exposure,
            requested_leverage=2.0,
            approved_leverage=2.0,
        )
        rejected = RiskDecision(
            portfolio_target_id=target.portfolio_target_id,
            approved=False,
            requested_exposure=target.requested_target_exposure,
            approved_exposure=0.0,
            requested_leverage=2.0,
            approved_leverage=0.0,
            rejection_reasons=["TEST_REJECTION"],
        )
        execution_settings, paper_engine_settings = self._execution_settings()
        with patch("app.pipeline.execution.settings", execution_settings), patch(
            "app.trading.paper_engine.settings", paper_engine_settings
        ):
            simulator = PaperExecutionSimulator(self.session)
            missing = simulator.submit_target(
                target=target,
                risk_decision=approved,
                market=market,
                equity=10_000.0,
                account_id="champion",
                decision_trace_id=trace_id,
            )
            self.assertEqual(missing.result.status, "REJECTED")
            self.assertIn("persisted approved", missing.result.message.lower())

            self._persist_risk(rejected, target=target, account_id="champion", trace_id=trace_id)
            unapproved = simulator.submit_target(
                target=target,
                risk_decision=rejected,
                market=market,
                equity=10_000.0,
                account_id="champion",
                decision_trace_id=trace_id,
            )
            self.assertEqual(unapproved.result.status, "REJECTED")

            self._persist_risk(approved, target=target, account_id="champion", trace_id=trace_id)
            accepted = simulator.submit_target(
                target=target,
                risk_decision=approved,
                market=market,
                equity=10_000.0,
                account_id="champion",
                decision_trace_id=trace_id,
            )

        self.assertEqual(accepted.result.status, "FILLED")
        self.assertIsNotNone(accepted.order)
        self.assertIsNotNone(accepted.fill)
        # Exposure is a notional/equity fraction. Approved leverage reserves less
        # margin; it must not multiply the authorized notional a second time.
        self.assertAlmostEqual(accepted.fill.notional, 1_000.0)
        position = self.session.scalar(
            select(Position).where(Position.paper_account_id == "champion", Position.symbol == target.symbol, Position.status == "OPEN")
        )
        self.assertIsNotNone(position)
        self.assertAlmostEqual(position.margin_used, 500.0)
        self.assertEqual(
            self.session.scalar(select(SimulatedOrderRecord.state).where(SimulatedOrderRecord.order_id == accepted.order.order_id)),
            "FILLED",
        )

    def test_execution_rejects_cross_target_amount_tampering_and_replay(self) -> None:
        target = self._target(requested=0.10, symbol="BTCUSDT")
        trace_id = self._persist_target(target)
        approved = RiskDecision(
            portfolio_target_id=target.portfolio_target_id,
            approved=True,
            requested_exposure=0.10,
            approved_exposure=0.10,
            requested_leverage=2.0,
            approved_leverage=2.0,
        )
        self._persist_risk(approved, target=target, account_id="champion", trace_id=trace_id)
        other_target = self._target(requested=0.10, symbol="ETHUSDT")
        other_trace = self._persist_target(other_target, trace_id=trace_id)
        market = MarketSnapshot(symbol=target.symbol, price=100.0, observed_at=_utc_now())
        execution_settings, paper_engine_settings = self._execution_settings()

        with patch("app.pipeline.execution.settings", execution_settings), patch(
            "app.trading.paper_engine.settings", paper_engine_settings
        ):
            simulator = PaperExecutionSimulator(self.session)
            cross_target = simulator.submit_target(
                target=other_target,
                risk_decision=approved,
                market=MarketSnapshot(symbol=other_target.symbol, price=100.0, observed_at=_utc_now()),
                equity=10_000.0,
                account_id="champion",
                decision_trace_id=other_trace,
            )
            tampered = RiskDecision(
                risk_decision_id=approved.risk_decision_id,
                portfolio_target_id=target.portfolio_target_id,
                approved=True,
                requested_exposure=0.10,
                approved_exposure=0.50,
                requested_leverage=2.0,
                approved_leverage=2.0,
            )
            amount_mismatch = simulator.submit_target(
                target=target,
                risk_decision=tampered,
                market=market,
                equity=10_000.0,
                account_id="champion",
                decision_trace_id=trace_id,
            )
            accepted = simulator.submit_target(
                target=target,
                risk_decision=approved,
                market=market,
                equity=10_000.0,
                account_id="champion",
                decision_trace_id=trace_id,
            )
            replay = simulator.submit_target(
                target=target,
                risk_decision=approved,
                market=market,
                equity=10_000.0,
                account_id="champion",
                decision_trace_id=trace_id,
            )

        self.assertEqual(cross_target.result.status, "REJECTED")
        self.assertIn("target", cross_target.result.message.lower())
        self.assertEqual(amount_mismatch.result.status, "REJECTED")
        self.assertIn("amount", amount_mismatch.result.message.lower())
        self.assertEqual(accepted.result.status, "FILLED")
        self.assertEqual(replay.result.status, "REJECTED")
        self.assertTrue(replay.duplicate)
        self.assertIn("replay", replay.result.message.lower())
        self.assertEqual(
            len(list(self.session.scalars(select(PaperTrade).where(PaperTrade.risk_decision_id == approved.risk_decision_id)))),
            1,
        )

    def test_positive_and_legacy_signed_drawdown_reject_new_exposure(self) -> None:
        now = _utc_now()
        self.session.add(
            AccountEquity(
                paper_account_id="champion",
                timestamp=now - timedelta(seconds=1),
                cash_balance=10_000.0,
                equity=10_000.0,
                drawdown=0.0,
            )
        )
        self.session.flush()
        _, paper_engine_settings = self._execution_settings()
        with patch("app.trading.paper_engine.settings", paper_engine_settings):
            equity_row = PaperEngine(self.session)._record_equity(8_000.0, paper_account_id="champion")
        self.assertAlmostEqual(equity_row.drawdown, 0.20)

        target = self._target(requested=0.10)
        self._persist_target(target)
        decision = self._risk_approval(
            target,
            self._inputs(now, equity=8_000.0, cash_balance=8_000.0),
            self._policy(max_portfolio_drawdown_pct=0.15),
        )
        self.assertFalse(decision.approved)
        self.assertIn("MAX_PORTFOLIO_DRAWDOWN_REACHED", decision.rejection_reasons)

        equity_row.drawdown = -0.20
        self.session.flush()
        legacy_target = self._target(requested=0.10, symbol="ETHUSDT")
        self._persist_target(legacy_target)
        legacy_decision = self._risk_approval(
            legacy_target,
            self._inputs(
                now,
                equity=8_000.0,
                cash_balance=8_000.0,
                market=MarketSnapshot(symbol="ETHUSDT", price=100.0, observed_at=now),
            ),
            self._policy(max_portfolio_drawdown_pct=0.15),
        )
        self.assertFalse(legacy_decision.approved)
        self.assertIn("MAX_PORTFOLIO_DRAWDOWN_REACHED", legacy_decision.rejection_reasons)

    def test_sandbox_registration_uses_a_distinct_paper_account_without_profit_gate(self) -> None:
        artifact = self.temp_dir / "candidate.json"
        artifact.write_text(
            json.dumps({"feature_columns": ["price_change"], "coefficients": [1.0], "intercept": 0.0}),
            encoding="utf-8",
        )
        registry = ModelRegistry(self.session)
        model = registry.register(
            name="candidate",
            model_id="candidate-model",
            version="v1",
            model_family="momentum",
            path=str(artifact),
            feature_schema_version="test-v1",
            feature_columns=["price_change"],
            lifecycle=ModelLifecycle.TRAINED,
            metrics={},
        )
        registry_settings = SimpleNamespace(paper_start_balance=2_500.0, v2_sandbox_max_exposure_pct=0.03)
        with patch("app.pipeline.registry.settings", registry_settings):
            sandbox = registry.start_sandbox(model.id, name="candidate sandbox")
        self.session.flush()

        self.assertNotEqual(sandbox.account_id, "champion")
        self.assertTrue(sandbox.account_id.startswith("sandbox-"))
        self.assertEqual(sandbox.model_version_id, model.id)
        self.assertEqual(sandbox.max_exposure_pct, 0.03)
        self.assertFalse(sandbox.payload["profitability_gate_required"])
        self.assertEqual(model.lifecycle_state, ModelLifecycle.PAPER_SANDBOX.value)
        self.assertEqual(
            self.session.scalar(select(PaperSandboxAccount.account_id).where(PaperSandboxAccount.id == sandbox.id)),
            sandbox.account_id,
        )
        self.assertIsNone(self.session.scalar(select(ModelVersion.id).where(ModelVersion.lifecycle_state == ModelLifecycle.CHAMPION.value)))

        # Persisted sandbox limits, not champion defaults or caller inputs, govern
        # both the fake account balance and approved exposure.
        _, paper_engine_settings = self._execution_settings()
        with patch("app.trading.paper_engine.settings", paper_engine_settings):
            sandbox_equity = PaperEngine(self.session, paper_account_id=sandbox.account_id)._latest_account(create=True)
        self.assertEqual(sandbox_equity.cash_balance, 2_500.0)
        self.assertEqual(sandbox_equity.equity, 2_500.0)

        target = self._target(requested=0.25)
        self._persist_target(target, account_id=sandbox.account_id)
        capped = self._risk_approval(
            target,
            self._inputs(
                _utc_now(),
                account_id=sandbox.account_id,
                equity=2_500.0,
                cash_balance=2_500.0,
            ),
            self._policy(),
        )
        self.assertTrue(capped.approved)
        self.assertAlmostEqual(capped.approved_exposure, 0.03)
        self.assertIn("SANDBOX_EXPOSURE_CAP", capped.triggered_limits)

    def test_data_collection_reset_keeps_realized_pnl_account_isolated(self) -> None:
        account_id = "sandbox-reset"
        self.session.add_all(
            [
                PaperSandboxAccount(
                    account_id=account_id,
                    name="reset sandbox",
                    starting_balance=2_500.0,
                    max_exposure_pct=0.03,
                    active=True,
                ),
                AccountEquity(
                    paper_account_id=account_id,
                    cash_balance=100.0,
                    equity=100.0,
                    drawdown=0.96,
                ),
                PaperTrade(symbol="BTCUSDT", paper_account_id=account_id, action="SELL", realized_pnl=-5.0),
                PaperTrade(symbol="ETHUSDT", paper_account_id="champion", action="SELL", realized_pnl=-500.0),
            ]
        )
        self.session.flush()
        reset_settings = SimpleNamespace(
            is_paper_mode=True,
            paper_data_collection_mode=True,
            paper_data_collection_reset_enabled=True,
            paper_data_collection_reset_equity_pct=0.10,
            paper_start_balance=10_000.0,
            paper_leverage=2.0,
            paper_max_leverage=2.0,
        )
        with patch("app.trading.paper_engine.settings", reset_settings):
            result = PaperEngine(self.session, paper_account_id=account_id).reset_paper_account_if_needed()

        self.assertIsNotNone(result)
        latest = self.session.scalar(
            select(AccountEquity)
            .where(AccountEquity.paper_account_id == account_id)
            .order_by(AccountEquity.timestamp.desc())
            .limit(1)
        )
        self.assertIsNotNone(latest)
        self.assertEqual(latest.cash_balance, 2_500.0)
        self.assertEqual(latest.equity, 2_500.0)
        self.assertEqual(latest.realized_pnl, -5.0)

    def test_resting_limit_cancel_replace_and_restart_expiration(self) -> None:
        target = self._target(requested=0.10, symbol="SOLUSDT")
        trace_id = self._persist_target(target)
        approved = RiskDecision(
            portfolio_target_id=target.portfolio_target_id,
            approved=True,
            requested_exposure=0.10,
            approved_exposure=0.10,
            requested_leverage=2.0,
            approved_leverage=2.0,
        )
        self._persist_risk(approved, target=target, account_id="champion", trace_id=trace_id)
        execution_settings, paper_engine_settings = self._execution_settings()
        execution_settings.paper_simulated_order_ttl_seconds = 300
        market = MarketSnapshot(symbol=target.symbol, price=100.0, observed_at=_utc_now())
        with patch("app.pipeline.execution.settings", execution_settings), patch(
            "app.trading.paper_engine.settings", paper_engine_settings
        ):
            simulator = PaperExecutionSimulator(self.session)
            resting = simulator.submit_target(
                target=target,
                risk_decision=approved,
                market=market,
                equity=10_000.0,
                account_id="champion",
                decision_trace_id=trace_id,
                order_type="LIMIT",
                limit_price=90.0,
            )
            self.assertEqual(resting.result.status, "PENDING")
            self.assertIsNone(resting.fill)
            replacement = simulator.replace_limit_order(resting.order.order_id, new_limit_price=95.0)
            original_state = self.session.scalar(
                select(SimulatedOrderRecord.state).where(SimulatedOrderRecord.order_id == resting.order.order_id)
            )
            recovery = simulator.recover_open_orders(now=replacement.expires_at + timedelta(seconds=1))

        self.assertEqual(original_state, "CANCELLED")
        self.assertEqual(replacement.state, "EXPIRED")
        self.assertEqual(recovery["expired"], 1)
        self.assertEqual(
            self.session.scalar(select(PaperTrade.id).where(PaperTrade.simulated_order_id == replacement.order_id)),
            None,
        )

    def test_resting_limit_matches_later_and_completes_one_partial_fill(self) -> None:
        target = self._target(requested=0.10, symbol="ADAUSDT")
        trace_id = self._persist_target(target)
        approved = RiskDecision(
            portfolio_target_id=target.portfolio_target_id,
            approved=True,
            requested_exposure=0.10,
            approved_exposure=0.10,
            requested_leverage=2.0,
            approved_leverage=2.0,
        )
        self._persist_risk(approved, target=target, account_id="champion", trace_id=trace_id)
        execution_settings, paper_engine_settings = self._execution_settings()
        execution_settings.paper_simulated_partial_fill_enabled = True
        execution_settings.paper_simulated_funding_rate = 0.001
        with patch("app.pipeline.execution.settings", execution_settings), patch(
            "app.trading.paper_engine.settings", paper_engine_settings
        ):
            simulator = PaperExecutionSimulator(self.session)
            resting = simulator.submit_target(
                target=target,
                risk_decision=approved,
                market=MarketSnapshot(symbol=target.symbol, price=100.0, observed_at=_utc_now()),
                equity=10_000.0,
                account_id="champion",
                decision_trace_id=trace_id,
                order_type="LIMIT",
                limit_price=90.0,
            )
            first = simulator.process_resting_orders(
                {target.symbol: MarketSnapshot(symbol=target.symbol, price=89.0, observed_at=_utc_now())}
            )
            second = simulator.process_resting_orders(
                {target.symbol: MarketSnapshot(symbol=target.symbol, price=89.0, observed_at=_utc_now())}
            )

        self.assertEqual(resting.result.status, "PENDING")
        self.assertEqual(first["partially_filled"], 1)
        self.assertEqual(second["filled"], 1)
        fills = list(
            self.session.scalars(
                select(SimulatedFillRecord).where(SimulatedFillRecord.order_id == resting.order.order_id)
            )
        )
        self.assertEqual(len(fills), 2)
        self.assertAlmostEqual(sum(row.notional for row in fills), 1_000.0, places=6)
        self.assertTrue(all((row.payload or {}).get("funding_booked") for row in fills))
        self.assertEqual(
            self.session.scalar(
                select(SimulatedOrderRecord.state).where(SimulatedOrderRecord.order_id == resting.order.order_id)
            ),
            "FILLED",
        )

    def test_resting_limit_rechecks_stale_market_before_increasing_exposure(self) -> None:
        target = self._target(requested=0.10, symbol="SOLUSDT")
        trace_id = self._persist_target(target)
        approved = RiskDecision(
            portfolio_target_id=target.portfolio_target_id,
            approved=True,
            requested_exposure=0.10,
            approved_exposure=0.10,
            requested_leverage=2.0,
            approved_leverage=2.0,
        )
        self._persist_risk(approved, target=target, account_id="champion", trace_id=trace_id)
        execution_settings, paper_engine_settings = self._execution_settings()
        execution_settings.risk_require_fresh_data = True
        execution_settings.risk_max_market_data_age_seconds = 30
        with patch("app.pipeline.execution.settings", execution_settings), patch(
            "app.trading.paper_engine.settings", paper_engine_settings
        ):
            simulator = PaperExecutionSimulator(self.session)
            resting = simulator.submit_target(
                target=target,
                risk_decision=approved,
                market=MarketSnapshot(symbol=target.symbol, price=100.0, observed_at=_utc_now()),
                equity=10_000.0,
                account_id="champion",
                decision_trace_id=trace_id,
                order_type="LIMIT",
                limit_price=90.0,
            )
            summary = simulator.process_resting_orders(
                {
                    target.symbol: MarketSnapshot(
                        symbol=target.symbol,
                        price=89.0,
                        observed_at=_utc_now() - timedelta(minutes=5),
                    )
                }
            )

        self.assertEqual(summary["rejected_or_error"], 1)
        self.assertEqual(
            self.session.scalar(
                select(SimulatedOrderRecord.state).where(SimulatedOrderRecord.order_id == resting.order.order_id)
            ),
            "REJECTED",
        )
        self.assertIsNone(
            self.session.scalar(select(PaperTrade.id).where(PaperTrade.simulated_order_id == resting.order.order_id))
        )

    def test_volume_participation_funding_impact_and_account_reconciliation(self) -> None:
        target = self._target(requested=0.10, symbol="XRPUSDT")
        trace_id = self._persist_target(target)
        approved = RiskDecision(
            portfolio_target_id=target.portfolio_target_id,
            approved=True,
            requested_exposure=0.10,
            approved_exposure=0.10,
            requested_leverage=2.0,
            approved_leverage=2.0,
        )
        self._persist_risk(approved, target=target, account_id="champion", trace_id=trace_id)
        execution_settings, paper_engine_settings = self._execution_settings()
        execution_settings.paper_simulated_volume_participation = 0.10
        execution_settings.paper_simulated_partial_fill_enabled = False
        execution_settings.paper_simulated_funding_rate = 0.001
        execution_settings.paper_simulated_market_impact_coefficient = 0.002
        market = MarketSnapshot(
            symbol=target.symbol,
            price=100.0,
            observed_at=_utc_now(),
            available_volume=20.0,
        )
        with patch("app.pipeline.execution.settings", execution_settings), patch(
            "app.trading.paper_engine.settings", paper_engine_settings
        ):
            simulator = PaperExecutionSimulator(self.session)
            outcome = simulator.submit_target(
                target=target,
                risk_decision=approved,
                market=market,
                equity=10_000.0,
                account_id="champion",
                decision_trace_id=trace_id,
            )
            reconciliation = simulator.reconcile_account("champion", mark_prices={target.symbol: outcome.fill.price})

        self.assertEqual(outcome.result.status, "FILLED")
        self.assertIsNotNone(outcome.fill)
        self.assertGreater(outcome.fill.quantity, 1.9)
        self.assertLessEqual(outcome.fill.quantity, 2.0)
        self.assertAlmostEqual(outcome.fill.funding, outcome.fill.notional * 0.001, places=8)
        self.assertGreater(outcome.fill.fee, 0.0)
        self.assertGreater(outcome.fill.slippage, 0.0)
        self.assertGreater(outcome.fill.price, 100.0)
        self.assertEqual(
            self.session.scalar(select(SimulatedOrderRecord.state).where(SimulatedOrderRecord.order_id == outcome.order.order_id)),
            "PARTIALLY_FILLED",
        )
        self.assertTrue(reconciliation["cash_reconciled"])
        self.assertEqual(reconciliation["orders_missing_fills"], [])
        self.assertEqual(reconciliation["orders_missing_trades"], [])


class V2VisionAuthenticationTests(IsolatedSqliteCase):
    def test_vision_page_and_api_require_and_accept_basic_authentication(self) -> None:
        app = FastAPI()
        app.include_router(vision_api_router)
        app.include_router(dashboard_router)

        def isolated_session():
            session = Session(self.engine)
            try:
                yield session
            finally:
                session.close()

        app.dependency_overrides[get_session] = isolated_session
        credentials = SimpleNamespace(admin_token=None, dashboard_username="vision-user", dashboard_password="vision-pass")
        basic = base64.b64encode(b"vision-user:vision-pass").decode("ascii")
        headers = {"Authorization": f"Basic {basic}"}
        with patch("app.security.settings", credentials):
            with TestClient(app) as client:
                self.assertEqual(client.get("/vision").status_code, 401)
                self.assertEqual(client.get("/api/vision/symbols").status_code, 401)
                self.assertEqual(client.get("/vision", headers=headers).status_code, 200)
                response = client.get("/api/vision/symbols", headers=headers)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("symbols", response.json())


class V2ApiAndVisionIntegrationTests(IsolatedSqliteCase):
    def _vision_app(self) -> FastAPI:
        app = FastAPI()
        app.include_router(vision_api_router)

        def isolated_session():
            session = Session(self.engine)
            try:
                yield session
            finally:
                session.close()

        app.dependency_overrides[get_session] = isolated_session
        return app

    def test_main_mounts_v2_operations_router(self) -> None:
        from app.main import app as production_app

        route_paths = {getattr(route, "path", None) for route in production_app.routes}
        self.assertIn("/api/v2/pipeline/run", route_paths)
        self.assertIn("/api/v2/models", route_paths)

    def test_vision_page_uses_configured_refresh_and_default_limit(self) -> None:
        app = FastAPI()
        app.include_router(dashboard_router)
        dashboard_settings = SimpleNamespace(
            binance_symbols=["BTCUSDT"],
            auto_trader_symbols=["BTCUSDT"],
            derivatives_symbols=["BTCUSDT"],
            trading_mode="paper",
            vision_refresh_seconds=37,
            vision_default_limit=17,
        )
        credentials = SimpleNamespace(admin_token="vision-token", dashboard_username=None, dashboard_password=None)
        with patch("app.dashboard.routes.settings", dashboard_settings), patch("app.security.settings", credentials):
            with TestClient(app) as client:
                response = client.get("/vision", headers={"x-admin-token": "vision-token"})

        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("refreshSeconds: 37", response.text)
        self.assertIn("defaultLimit: 17", response.text)

    def test_chart_filters_symbol_and_window_then_returns_latest_slice_in_time_order(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for minute in range(5):
            timestamp = start + timedelta(minutes=minute)
            self.session.add(
                Candle(
                    exchange="binance",
                    symbol="BTCUSDT",
                    interval="1m",
                    open_time=timestamp,
                    close_time=timestamp + timedelta(minutes=1),
                    open=float(minute),
                    high=float(minute) + 1.0,
                    low=float(minute) - 1.0,
                    close=float(minute),
                    volume=1.0,
                    is_closed=True,
                )
            )
        self.session.add(
            Candle(
                exchange="binance",
                symbol="ETHUSDT",
                interval="1m",
                open_time=start + timedelta(minutes=3),
                close_time=start + timedelta(minutes=4),
                open=99.0,
                high=100.0,
                low=98.0,
                close=99.0,
                volume=1.0,
                is_closed=True,
            )
        )
        self.session.commit()

        credentials = SimpleNamespace(admin_token="vision-token", dashboard_username=None, dashboard_password=None)
        with patch("app.security.settings", credentials):
            with TestClient(self._vision_app()) as client:
                response = client.get(
                    "/api/vision/chart",
                    headers={"x-admin-token": "vision-token"},
                    params={
                        "symbol": "BTCUSDT",
                        "start": (start + timedelta(minutes=1)).isoformat(),
                        "end": (start + timedelta(minutes=3)).isoformat(),
                        "limit": 2,
                    },
                )

        self.assertEqual(response.status_code, 200, response.text)
        candles = response.json()["candles"]
        self.assertEqual([row["close"] for row in candles], [2.0, 3.0])
        self.assertEqual([row["symbol"] for row in candles], ["BTCUSDT", "BTCUSDT"])
        self.assertLess(candles[0]["time"], candles[1]["time"])

    def test_external_ai_state_never_falls_back_to_another_symbol(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.session.add_all(
            [
                ExternalAIRequest(
                    symbol="BTCUSDT",
                    provider="btc-provider",
                    model="btc-model",
                    content_hash="btc-hash",
                    prompt_version="test-v1",
                    requested_at=now,
                    status="COMPLETED",
                ),
                ExternalAIRequest(
                    symbol="ETHUSDT",
                    provider="eth-provider",
                    model="eth-model",
                    content_hash="eth-hash",
                    prompt_version="test-v1",
                    requested_at=now + timedelta(minutes=1),
                    status="COMPLETED",
                ),
            ]
        )
        self.session.commit()

        credentials = SimpleNamespace(admin_token="vision-token", dashboard_username=None, dashboard_password=None)
        with patch("app.security.settings", credentials):
            with TestClient(self._vision_app()) as client:
                btc_response = client.get(
                    "/api/vision/state",
                    headers={"x-admin-token": "vision-token"},
                    params={"symbol": "BTCUSDT"},
                )
                missing_response = client.get(
                    "/api/vision/state",
                    headers={"x-admin-token": "vision-token"},
                    params={"symbol": "SOLUSDT"},
                )

        self.assertEqual(btc_response.status_code, 200, btc_response.text)
        self.assertEqual(btc_response.json()["external_ai"]["symbol"], "BTCUSDT")
        self.assertEqual(btc_response.json()["external_ai"]["provider"], "btc-provider")
        self.assertEqual(missing_response.status_code, 200, missing_response.text)
        self.assertEqual(missing_response.json()["external_ai"]["status"], "not_recorded")
        self.assertEqual(missing_response.json()["external_ai"]["symbol"], "SOLUSDT")

    def test_additive_migration_backfills_external_ai_symbol_column_and_index(self) -> None:
        database_path = self.temp_dir / "legacy-external-ai.sqlite3"
        legacy_engine = create_engine(f"sqlite:///{database_path.as_posix()}")
        try:
            with legacy_engine.begin() as connection:
                connection.execute(
                    text(
                        "CREATE TABLE external_ai_requests "
                        "(id INTEGER PRIMARY KEY, requested_at TIMESTAMP)"
                    )
                )

            result = run_additive_migrations(legacy_engine)
            columns = {column["name"] for column in inspect(legacy_engine).get_columns("external_ai_requests")}
            indexes = {index["name"] for index in inspect(legacy_engine).get_indexes("external_ai_requests")}
        finally:
            legacy_engine.dispose()

        self.assertIn({"table": "external_ai_requests", "column": "symbol", "type": "VARCHAR(32)"}, result["added_columns"])
        self.assertIn("symbol", columns)
        self.assertIn("ix_external_ai_requests_symbol_time", indexes)
