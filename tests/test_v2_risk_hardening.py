"""Regression tests for final, independent V2 paper-risk backstops."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.models import Base, Position
from app.pipeline.domain import PortfolioTarget, new_id
from app.pipeline.risk import MarketSnapshot, PortfolioRiskEngine, RiskInputs, RiskPolicy


class V2RiskHardeningTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory(prefix="anata-risk-hardening-")
        database_path = Path(self._temporary_directory.name) / "risk.sqlite3"
        self.engine = create_engine(
            f"sqlite:///{database_path.as_posix()}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self.engine)
        self.session = Session(self.engine)
        self.now = datetime.now(timezone.utc)

    def tearDown(self) -> None:
        self.session.close()
        self.engine.dispose()
        self._temporary_directory.cleanup()

    def _policy(self, **overrides: object) -> RiskPolicy:
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
            "configuration_version": "risk-hardening-test",
            "max_cluster_exposure_pct": 0.25,
            "max_spread_pct": 0.005,
        }
        values.update(overrides)
        return RiskPolicy(**values)  # type: ignore[arg-type]

    def _target(self, symbol: str = "ETHUSDT", requested: float = 0.20) -> PortfolioTarget:
        return PortfolioTarget(
            symbol=symbol,
            current_exposure=0.0,
            requested_target_exposure=requested,
            requested_delta=requested,
            expected_return=0.01,
            expected_risk=0.02,
            risk_contribution=0.01,
            urgency=0.5,
            source_ensemble_decision_id=new_id("ens"),
        )

    def _inputs(self, symbol: str = "ETHUSDT", **overrides: object) -> RiskInputs:
        values: dict[str, object] = {
            "account_id": "champion",
            "cash_balance": 10_000.0,
            "equity": 10_000.0,
            "market": MarketSnapshot(
                symbol=symbol,
                price=100.0,
                observed_at=self.now,
                bid=99.99,
                ask=100.01,
            ),
            "confidence": 0.90,
            "liquidity_score": 0.90,
            "expected_cost": 0.001,
            "expected_volatility": 0.02,
            "current_gross_exposure": 0.0,
            "current_net_exposure": 0.0,
            "now": self.now,
        }
        values.update(overrides)
        return RiskInputs(**values)  # type: ignore[arg-type]

    def _approve(self, target: PortfolioTarget, inputs: RiskInputs, policy: RiskPolicy):
        risk_settings = SimpleNamespace(
            min_paper_trade_notional=1.0,
            paper_fee_rate=0.0004,
            paper_simulated_spread_pct=0.0002,
        )
        with patch("app.pipeline.risk.settings", risk_settings):
            return PortfolioRiskEngine(self.session, policy=policy).approve(
                target,
                inputs,
                decision_trace_id=new_id("trace"),
            )

    def test_risk_rejects_unhealthy_execution_simulation(self) -> None:
        decision = self._approve(
            self._target(),
            self._inputs(simulation_healthy=False),
            self._policy(),
        )
        self.assertFalse(decision.approved)
        self.assertIn("PAPER_EXECUTION_SIMULATION_UNHEALTHY", decision.rejection_reasons)

    def test_risk_rejects_excessive_spread(self) -> None:
        decision = self._approve(
            self._target(),
            self._inputs(spread_pct=0.02),
            self._policy(max_spread_pct=0.005),
        )
        self.assertFalse(decision.approved)
        self.assertIn("MAX_SPREAD_EXCEEDED", decision.rejection_reasons)

    def test_risk_independently_caps_correlated_cluster(self) -> None:
        self.session.add(
            Position(
                symbol="SOLUSDT",
                paper_account_id="champion",
                side="LONG",
                quantity=20.0,
                entry_price=100.0,
                current_price=100.0,
                notional=2_000.0,
                margin_used=1_000.0,
                leverage=2.0,
                status="OPEN",
            )
        )
        self.session.flush()

        decision = self._approve(
            self._target(symbol="ETHUSDT", requested=0.20),
            self._inputs(symbol="ETHUSDT", current_gross_exposure=0.20, current_net_exposure=0.20),
            self._policy(max_cluster_exposure_pct=0.25),
        )

        self.assertTrue(decision.approved)
        self.assertAlmostEqual(decision.approved_exposure, 0.05, places=6)
        self.assertIn("MAX_CORRELATED_CLUSTER_EXPOSURE", decision.triggered_limits)


if __name__ == "__main__":
    unittest.main()
