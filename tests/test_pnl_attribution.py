"""Deterministic contract tests for trace-based paper-PnL attribution."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.api.vision import vision_history, vision_overlays, vision_state
from app.db.models import (
    AccountEquity,
    Base,
    EnsembleDecisionRecord,
    EnsembleSignalWeight,
    ExternalAIRequest,
    ModelPredictionRecord,
    PaperTrade,
    PortfolioTargetRecord,
    Position,
    RiskDecisionRecord,
    SimulatedFillRecord,
    SimulatedOrderRecord,
    TradingSignalRecord,
)
from app.pipeline.attribution import MAX_ATTRIBUTION_ROWS, paper_pnl_attribution


class PaperPnlAttributionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory(prefix="anata-attribution-")
        database_path = Path(self._temporary_directory.name) / "attribution.sqlite3"
        self.engine = create_engine(
            f"sqlite:///{database_path.as_posix()}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self.engine)
        self.session = Session(self.engine)
        self.at = datetime(2026, 7, 14, 12, 30, tzinfo=timezone.utc)

    def tearDown(self) -> None:
        self.session.close()
        self.engine.dispose()
        self._temporary_directory.cleanup()

    def _prediction(
        self,
        prediction_id: str,
        *,
        trace_id: str,
        model_id: str,
        family: str,
        external: bool,
        provider: str | None,
    ) -> ModelPredictionRecord:
        return ModelPredictionRecord(
            prediction_id=prediction_id,
            decision_trace_id=trace_id,
            model_id=model_id,
            model_version="v1",
            model_family=family,
            symbol="BTCUSDT",
            generated_at=self.at,
            valid_from=self.at,
            expires_at=self.at + timedelta(minutes=5),
            forecast_horizon_seconds=300,
            expected_return=0.01,
            expected_volatility=0.02,
            probability_up=0.60,
            probability_down=0.40,
            confidence=0.80,
            calibration_score=0.70,
            uncertainty=0.20,
            regime="trend",
            feature_schema_version="price-news-v3",
            feature_snapshot_id=f"feature-{prediction_id}",
            data_version="test-v1",
            external_context_available=external,
            payload={
                "external_ai_provider": provider,
                "external_ai_prompt_version": "external-prompt-v1" if provider else None,
                "external_ai_available": external,
                "external_ai_missing": not external,
                "external_ai_failed": False,
                "local_news_model_version": "news-student-v1",
            },
        )

    def _signal(
        self,
        signal_id: str,
        prediction_id: str,
        *,
        trace_id: str,
        family: str,
    ) -> TradingSignalRecord:
        return TradingSignalRecord(
            signal_id=signal_id,
            prediction_id=prediction_id,
            decision_trace_id=trace_id,
            signal_family=family,
            symbol="BTCUSDT",
            generated_at=self.at,
            valid_until=self.at + timedelta(minutes=5),
            direction="LONG",
            strength=0.80,
            expected_return=0.01,
            expected_cost=0.001,
            net_expected_return=0.009,
            confidence=0.80,
            uncertainty=0.20,
            regime="trend",
            liquidity_score=0.90,
            health_status="HEALTHY",
            lifecycle_status="PAPER",
            reason_codes=["TEST"],
            payload={},
        )

    def _add_complete_closed_trade(self) -> int:
        trace_id = "trace-complete"
        ensemble_id = "ensemble-complete"
        target_id = "target-complete"
        risk_id = "risk-complete"
        order_id = "order-complete"
        self.session.add_all(
            [
                self._prediction(
                    "prediction-trend",
                    trace_id=trace_id,
                    model_id="trend-model",
                    family="trend",
                    external=True,
                    provider="gemini",
                ),
                self._prediction(
                    "prediction-micro",
                    trace_id=trace_id,
                    model_id="micro-model",
                    family="microstructure",
                    external=False,
                    provider=None,
                ),
            ]
        )
        self.session.flush()
        self.session.add_all(
            [
                self._signal(
                    "signal-trend",
                    "prediction-trend",
                    trace_id=trace_id,
                    family="trend",
                ),
                self._signal(
                    "signal-micro",
                    "prediction-micro",
                    trace_id=trace_id,
                    family="microstructure",
                ),
            ]
        )
        self.session.add(
            EnsembleDecisionRecord(
                ensemble_decision_id=ensemble_id,
                decision_trace_id=trace_id,
                symbol="BTCUSDT",
                generated_at=self.at,
                valid_until=self.at + timedelta(minutes=5),
                combined_expected_return=0.01,
                combined_expected_volatility=0.02,
                combined_uncertainty=0.20,
                combined_confidence=0.80,
                current_regime="trend",
                supporting_signals=["signal-trend", "signal-micro"],
                conflicting_signals=[],
                signal_weights={"signal-trend": 0.60, "signal-micro": 0.40},
                correlation_penalty=0.10,
                transaction_cost_penalty=0.001,
                regime_penalty=0.0,
                external_context_adjustment=0.01,
                decision_status="ACTIONABLE",
                reason_codes=["TEST"],
                payload={},
            )
        )
        self.session.flush()
        self.session.add_all(
            [
                EnsembleSignalWeight(
                    ensemble_decision_id=ensemble_id,
                    signal_id="signal-trend",
                    weight=0.60,
                ),
                EnsembleSignalWeight(
                    ensemble_decision_id=ensemble_id,
                    signal_id="signal-micro",
                    weight=0.40,
                ),
            ]
        )
        self.session.add(
            PortfolioTargetRecord(
                portfolio_target_id=target_id,
                decision_trace_id=trace_id,
                paper_account_id="champion",
                symbol="BTCUSDT",
                current_exposure=0.50,
                requested_target_exposure=0.0,
                requested_delta=-0.50,
                expected_return=0.01,
                expected_risk=0.02,
                risk_contribution=0.01,
                urgency=0.80,
                source_ensemble_decision_id=ensemble_id,
                created_at=self.at,
                payload={},
            )
        )
        self.session.add(
            RiskDecisionRecord(
                risk_decision_id=risk_id,
                decision_trace_id=trace_id,
                portfolio_target_id=target_id,
                paper_account_id="champion",
                approved=True,
                requested_exposure=0.50,
                approved_exposure=0.25,
                requested_leverage=2.0,
                approved_leverage=1.0,
                triggered_limits=["MAX_SYMBOL_EXPOSURE"],
                rejection_reasons=[],
                configuration_version="test-v1",
                kill_switch_state=False,
                created_at=self.at,
                payload={},
            )
        )
        self.session.add(
            SimulatedOrderRecord(
                order_id=order_id,
                decision_trace_id=trace_id,
                risk_decision_id=risk_id,
                portfolio_target_id=target_id,
                paper_account_id="champion",
                symbol="BTCUSDT",
                side="LONG",
                order_type="MARKET",
                requested_quantity=1.0,
                requested_notional=100.0,
                state="FILLED",
                client_order_id="client-complete",
                created_at=self.at,
                updated_at=self.at,
                payload={},
            )
        )
        self.session.add_all(
            [
                SimulatedFillRecord(
                    fill_id="fill-complete-1",
                    order_id=order_id,
                    decision_trace_id=trace_id,
                    paper_account_id="champion",
                    symbol="BTCUSDT",
                    side="LONG",
                    quantity=0.5,
                    price=101.0,
                    notional=50.0,
                    fee=1.0,
                    slippage=0.01,
                    funding=0.25,
                    filled_at=self.at,
                    payload={"funding_rate": 0.005},
                ),
                SimulatedFillRecord(
                    fill_id="fill-complete-2",
                    order_id=order_id,
                    decision_trace_id=trace_id,
                    paper_account_id="champion",
                    symbol="BTCUSDT",
                    side="LONG",
                    quantity=0.25,
                    price=102.0,
                    notional=25.0,
                    fee=1.0,
                    slippage=0.02,
                    funding=0.25,
                    filled_at=self.at + timedelta(seconds=1),
                    payload={"funding_rate": 0.01},
                ),
            ]
        )
        trade = PaperTrade(
            symbol="BTCUSDT",
            paper_account_id="champion",
            risk_decision_id=risk_id,
            simulated_order_id=order_id,
            decision_trace_id=trace_id,
            action="SELL",
            side="LONG",
            quantity=0.75,
            price=102.0,
            notional=75.0,
            fee=2.0,
            realized_pnl=100.0,
            status="FILLED",
            raw_payload={"intent": "close", "gross_pnl": 102.0},
            created_at=self.at,
        )
        self.session.add(trade)
        self.session.commit()
        return trade.id

    def test_complete_trace_reconciles_and_dimensions_every_requested_stage(self) -> None:
        trade_id = self._add_complete_closed_trade()

        result = paper_pnl_attribution(
            self.session,
            symbol="BTCUSDT",
            account_id="champion",
            time_period="day",
        )

        self.assertEqual(result["trade_count"], 1)
        self.assertEqual(result["closed_trade_count"], 1)
        self.assertEqual(result["closed_paper_pnl"], 100.0)
        self.assertAlmostEqual(result["components"]["alpha_contribution"], 103.0)
        self.assertAlmostEqual(result["components"]["slippage"], -1.0)
        self.assertAlmostEqual(result["components"]["fees"], -2.0)
        self.assertEqual(result["components"]["funding"], 0.0)
        self.assertAlmostEqual(
            result["component_metadata"]["funding"]["estimated_value_not_booked"],
            -0.5,
        )
        self.assertTrue(result["reconciliation"]["reconciled"])
        self.assertEqual(result["reconciliation"]["difference"], 0.0)
        self.assertIsNone(result["component_metadata"]["ensemble_contribution"]["value"])

        self.assertAlmostEqual(result["by_model"]["trend-model:v1"], 61.8)
        self.assertAlmostEqual(result["by_model"]["micro-model:v1"], 41.2)
        self.assertAlmostEqual(result["by_signal"]["signal-trend"], 61.8)
        self.assertAlmostEqual(result["by_signal_family"]["microstructure"], 41.2)
        self.assertAlmostEqual(result["by_external_ai_availability"]["available"], 61.8)
        self.assertAlmostEqual(result["by_external_ai_provider"]["gemini"], 61.8)
        self.assertAlmostEqual(result["by_external_ai_provider"]["unavailable"], 41.2)
        self.assertEqual(result["by_ensemble_weighting"], {"ensemble-complete": 100.0})
        self.assertEqual(result["by_portfolio_sizing"], {"flat": 100.0})
        self.assertEqual(result["by_risk_resizing"], {"reduced": 100.0})
        self.assertEqual(result["by_symbol"], {"BTCUSDT": 100.0})
        self.assertEqual(result["by_regime"], {"trend": 100.0})
        self.assertEqual(result["by_time_period"], {"2026-07-14": 100.0})
        self.assertAlmostEqual(result["allocation_reconciliation"]["signal_unallocated_alpha"], 0.0)
        self.assertAlmostEqual(result["allocation_reconciliation"]["model_unallocated_alpha"], 0.0)

        record = result["trades"][0]
        self.assertEqual(record["paper_trade_id"], trade_id)
        self.assertEqual(record["lineage_status"], "complete_v2")
        self.assertEqual(record["missing_evidence"], [])
        self.assertEqual(len(record["simulated_fill_ids"]), 2)
        self.assertAlmostEqual(record["ensemble_weight_sum"], 1.0)
        self.assertAlmostEqual(record["unallocated_alpha_contribution"], 0.0)
        self.assertEqual(record["risk_resizing"]["resize_category"], "reduced")
        self.assertAlmostEqual(record["risk_resizing"]["resize_ratio"], 0.5)

    def test_vision_history_exposes_bounded_provider_and_resize_dimensions(self) -> None:
        self._add_complete_closed_trade()

        result = vision_history(
            symbol="BTCUSDT",
            start=None,
            end=None,
            account_id="champion",
            limit=10,
            session=self.session,
        )

        self.assertEqual(result["performance_by_external_ai_provider"]["gemini"], 61.8)
        self.assertEqual(result["performance_by_ensemble_weighting"], {"ensemble-complete": 100.0})
        self.assertEqual(result["performance_by_risk_resizing"], {"reduced": 100.0})
        self.assertEqual(result["attribution"]["selection"]["limit"], 10)
        self.assertTrue(result["attribution"]["selection"]["bounded"])

    def test_vision_outcome_metrics_exclude_opening_ledger_events(self) -> None:
        self._add_complete_closed_trade()
        self.session.add(
            PaperTrade(
                symbol="BTCUSDT",
                paper_account_id="champion",
                decision_trace_id="trace-open",
                action="BUY",
                side="LONG",
                quantity=0.25,
                price=100.0,
                notional=25.0,
                fee=3.0,
                realized_pnl=-3.0,
                status="FILLED",
                raw_payload={"intent": "open"},
                created_at=self.at + timedelta(seconds=2),
            )
        )
        self.session.commit()

        result = vision_history(
            symbol="BTCUSDT",
            account_id="champion",
            limit=10,
            session=self.session,
        )
        metrics = result["metrics"]

        self.assertEqual(metrics["trade_count"], 1)
        self.assertEqual(metrics["closed_trade_count"], 1)
        self.assertEqual(metrics["ledger_event_count"], 2)
        self.assertEqual(metrics["win_count"], 1)
        self.assertEqual(metrics["loss_count"], 0)
        self.assertEqual(metrics["win_rate"], 1.0)
        self.assertEqual(metrics["closed_paper_pnl"], 100.0)
        self.assertEqual(metrics["net_realized_pnl"], 100.0)
        self.assertEqual(metrics["ledger_total_paper_pnl"], 97.0)
        self.assertEqual(metrics["net_expectancy"], 100.0)
        self.assertEqual(len(result["paper_ledger_events"]), 2)

    def test_sandbox_reads_exact_paper_ledger_position_and_equity(self) -> None:
        self.session.add_all(
            [
                Position(
                    symbol="BTCUSDT",
                    paper_account_id="sandbox-a",
                    side="LONG",
                    quantity=2.0,
                    entry_price=100.0,
                    current_price=105.0,
                    notional=210.0,
                    margin_used=105.0,
                    leverage=2.0,
                    unrealized_pnl=10.0,
                    status="OPEN",
                    opened_at=self.at,
                ),
                Position(
                    symbol="BTCUSDT",
                    paper_account_id="champion",
                    side="SHORT",
                    quantity=99.0,
                    entry_price=200.0,
                    current_price=190.0,
                    notional=18_810.0,
                    margin_used=9_405.0,
                    leverage=2.0,
                    unrealized_pnl=990.0,
                    status="OPEN",
                    opened_at=self.at + timedelta(seconds=1),
                ),
                AccountEquity(
                    timestamp=self.at,
                    paper_account_id="sandbox-a",
                    cash_balance=1_000.0,
                    equity=1_010.0,
                    unrealized_pnl=10.0,
                    drawdown=0.02,
                ),
                AccountEquity(
                    timestamp=self.at + timedelta(seconds=1),
                    paper_account_id="champion",
                    cash_balance=9_000.0,
                    equity=9_990.0,
                    unrealized_pnl=990.0,
                    drawdown=0.10,
                ),
                PaperTrade(
                    symbol="BTCUSDT",
                    paper_account_id="sandbox-a",
                    decision_trace_id="sandbox-trace",
                    risk_decision_id="sandbox-risk",
                    simulated_order_id="sandbox-order",
                    action="BUY",
                    side="LONG",
                    quantity=2.0,
                    price=100.0,
                    notional=200.0,
                    fee=1.0,
                    realized_pnl=-1.0,
                    status="FILLED",
                    raw_payload={"intent": "open"},
                    created_at=self.at,
                ),
                PaperTrade(
                    symbol="BTCUSDT",
                    paper_account_id="champion",
                    action="BUY",
                    side="SHORT",
                    quantity=99.0,
                    price=200.0,
                    notional=19_800.0,
                    fee=20.0,
                    realized_pnl=-20.0,
                    status="FILLED",
                    raw_payload={"intent": "open"},
                    created_at=self.at + timedelta(seconds=1),
                ),
                PaperTrade(
                    symbol="BTCUSDT",
                    paper_account_id="sandbox-a",
                    action="CLOSE",
                    side="LONG",
                    quantity=0.5,
                    price=106.0,
                    notional=53.0,
                    fee=0.5,
                    realized_pnl=2.5,
                    status="FILLED",
                    raw_payload={"intent": "close", "gross_pnl": 3.0},
                    created_at=self.at + timedelta(seconds=2),
                ),
            ]
        )
        self.session.commit()

        current = vision_state(symbol="BTCUSDT", account_id="sandbox-a", session=self.session)
        overlays = vision_overlays(
            symbol="BTCUSDT",
            account_id="sandbox-a",
            limit=10,
            session=self.session,
        )
        history = vision_history(
            symbol="BTCUSDT",
            account_id="sandbox-a",
            limit=10,
            session=self.session,
        )

        self.assertEqual(current["position"]["paper_account_id"], "sandbox-a")
        self.assertEqual(current["position"]["quantity"], 2.0)
        self.assertEqual(current["account"]["equity"], 1_010.0)
        self.assertEqual(len(overlays["trades"]), 2)
        traced = next(item for item in overlays["trades"] if item["source"] == "v2-paper-ledger")
        legacy = next(item for item in overlays["trades"] if item["source"] == "legacy")
        self.assertTrue(traced["id"].startswith("v2-paper-trade:"))
        self.assertEqual(traced["decision_trace_id"], "sandbox-trace")
        self.assertEqual(traced["paper_account_id"], "sandbox-a")
        self.assertTrue(legacy["id"].startswith("legacy-trade:"))
        self.assertIsNone(legacy["decision_trace_id"])
        self.assertTrue(overlays["availability"]["paper_ledger_account_scoped"])
        self.assertEqual(len(history["paper_ledger_events"]), 2)
        self.assertEqual(history["equity"][0]["equity"], 1_010.0)
        self.assertTrue(history["availability"]["paper_ledger_account_scoped"])

    def test_current_intelligence_uses_displayed_decision_prediction_lineage(self) -> None:
        self._add_complete_closed_trade()
        self.session.add_all(
            [
                EnsembleDecisionRecord(
                    ensemble_decision_id="ensemble-unrelated",
                    decision_trace_id="trace-unrelated",
                    symbol="BTCUSDT",
                    generated_at=self.at + timedelta(minutes=2),
                    valid_until=self.at + timedelta(minutes=7),
                    combined_expected_return=-0.5,
                    combined_expected_volatility=0.9,
                    combined_uncertainty=0.9,
                    combined_confidence=0.1,
                    current_regime="unrelated",
                    supporting_signals=[],
                    conflicting_signals=[],
                    signal_weights={},
                    correlation_penalty=0.0,
                    transaction_cost_penalty=0.0,
                    regime_penalty=0.0,
                    external_context_adjustment=0.0,
                    decision_status="NEUTRAL",
                    reason_codes=[],
                    payload={},
                ),
                ExternalAIRequest(
                    symbol="BTCUSDT",
                    provider="newer-unrelated-provider",
                    model="unrelated-model",
                    content_hash="unrelated-content",
                    prompt_version="unrelated-prompt",
                    requested_at=self.at + timedelta(minutes=3),
                    completed_at=self.at + timedelta(minutes=3, seconds=1),
                    status="SUCCEEDED",
                    retry_count=0,
                    cache_hit=False,
                    payload={},
                ),
            ]
        )
        self.session.commit()

        current = vision_state(symbol="BTCUSDT", account_id="champion", session=self.session)
        overlays = vision_overlays(
            symbol="BTCUSDT",
            account_id="champion",
            limit=10,
            session=self.session,
        )

        self.assertEqual(current["decision_trace_id"], "trace-complete")
        self.assertEqual(current["ensemble"]["ensemble_decision_id"], "ensemble-complete")
        self.assertEqual(current["external_ai"]["provider"], "gemini")
        self.assertEqual(current["external_ai"]["source"], "displayed_decision_predictions")
        self.assertTrue(current["external_ai"]["lineage_match"])
        self.assertEqual(current["external_ai"]["decision_trace_id"], "trace-complete")
        self.assertEqual(current["local_news_model"]["version"], "news-student-v1")
        self.assertTrue(current["local_news_model"]["lineage_match"])
        self.assertEqual(overlays["portfolio_targets"][0]["requested_target_exposure"], 0.0)
        self.assertEqual(overlays["risk_decisions"][0]["approved_exposure"], 0.25)

    def test_unlinked_latest_intelligence_is_explicitly_labeled_fallback(self) -> None:
        self.session.add(
            ExternalAIRequest(
                symbol="ETHUSDT",
                provider="latest-only-provider",
                model="latest-only-model",
                content_hash="latest-only-content",
                prompt_version="latest-only-prompt",
                requested_at=self.at,
                completed_at=self.at + timedelta(seconds=1),
                status="SUCCEEDED",
                retry_count=0,
                cache_hit=False,
                payload={},
            )
        )
        self.session.commit()

        current = vision_state(symbol="ETHUSDT", account_id="sandbox-b", session=self.session)

        self.assertEqual(current["external_ai"]["provider"], "latest-only-provider")
        self.assertEqual(current["external_ai"]["source"], "latest_symbol_request_fallback")
        self.assertFalse(current["external_ai"]["lineage_match"])
        self.assertIsNone(current["external_ai"]["decision_trace_id"])

    def test_dashboard_draws_requested_and_approved_target_markers(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        javascript = (project_root / "app/dashboard/static/vision.js").read_text(encoding="utf-8")
        template = (project_root / "app/dashboard/templates/vision.html").read_text(encoding="utf-8")

        self.assertIn("(overlays.portfolio_targets || []).forEach", javascript)
        self.assertIn("TARGET REQUEST", javascript)
        self.assertIn("(overlays.risk_decisions || []).forEach", javascript)
        self.assertIn("RISK APPROVED", javascript)
        self.assertIn("Requested target", template)
        self.assertIn("Approved risk target", template)

    def test_legacy_missing_evidence_is_explicit_and_goes_to_residual(self) -> None:
        self.session.add(
            PaperTrade(
                symbol="ETHUSDT",
                paper_account_id="champion",
                action="CLOSE",
                side="LONG",
                quantity=1.0,
                price=20.0,
                notional=20.0,
                fee=1.0,
                realized_pnl=7.0,
                status="FILLED",
                raw_payload=None,
                created_at=self.at,
            )
        )
        self.session.commit()

        result = paper_pnl_attribution(self.session, symbol="ETHUSDT")

        self.assertEqual(result["closed_trade_count"], 1)
        self.assertEqual(result["components"]["alpha_contribution"], 0.0)
        self.assertEqual(result["components"]["fees"], -1.0)
        self.assertEqual(result["components"]["unexplained_residual"], 8.0)
        self.assertIsNone(result["component_metadata"]["alpha_contribution"]["value"])
        self.assertTrue(result["reconciliation"]["reconciled"])
        record = result["trades"][0]
        self.assertEqual(record["lineage_status"], "legacy_untraced")
        self.assertIn("model_signal_lineage", record["missing_evidence"])
        self.assertEqual(
            record["component_metadata"]["alpha_contribution"]["method"],
            "unavailable_no_recorded_gross_pnl",
        )

    def test_explicit_components_are_used_without_backfilling_missing_counterfactuals(self) -> None:
        self.session.add(
            PaperTrade(
                symbol="SOLUSDT",
                paper_account_id="sandbox",
                action="CLOSE",
                side="SHORT",
                quantity=1.0,
                price=10.0,
                notional=10.0,
                fee=2.0,
                realized_pnl=10.0,
                status="FILLED",
                raw_payload={
                    "intent": "close",
                    "gross_pnl": 99_999.0,
                    "attribution": {
                        "alpha_contribution": 4.0,
                        "ensemble_contribution": 1.0,
                        "position_sizing_contribution": 2.0,
                        "broad_market_exposure": 2.0,
                        "execution_contribution": 1.0,
                        "slippage": -1.0,
                        "funding": 0.5,
                        "methods": {"broad_market_exposure": "recorded_benchmark_counterfactual"},
                    },
                },
                created_at=self.at,
            )
        )
        self.session.commit()

        result = paper_pnl_attribution(self.session, symbol="SOLUSDT", account_id="sandbox")

        self.assertEqual(result["components"]["alpha_contribution"], 4.0)
        self.assertEqual(result["components"]["broad_market_exposure"], 2.0)
        self.assertEqual(result["components"]["unexplained_residual"], 2.5)
        self.assertEqual(
            result["trades"][0]["component_metadata"]["broad_market_exposure"]["method"],
            "recorded_benchmark_counterfactual",
        )
        self.assertTrue(result["reconciliation"]["reconciled"])

    def test_time_filters_period_buckets_and_limit_are_bounded(self) -> None:
        for offset in range(3):
            self.session.add(
                PaperTrade(
                    symbol="XRPUSDT",
                    paper_account_id="champion",
                    action="CLOSE",
                    side="LONG",
                    quantity=1.0,
                    price=1.0,
                    notional=1.0,
                    fee=0.0,
                    realized_pnl=float(offset + 1),
                    status="FILLED",
                    raw_payload={"intent": "close", "gross_pnl": float(offset + 1)},
                    created_at=self.at + timedelta(days=offset),
                )
            )
        self.session.commit()

        result = paper_pnl_attribution(
            self.session,
            symbol="XRPUSDT",
            start=self.at + timedelta(hours=1),
            end=self.at + timedelta(days=3),
            limit=MAX_ATTRIBUTION_ROWS + 99,
            time_period="month",
        )

        self.assertEqual(result["trade_count"], 2)
        self.assertEqual(result["total_paper_pnl"], 5.0)
        self.assertEqual(result["by_time_period"], {"2026-07": 5.0})
        self.assertEqual(result["selection"]["limit"], MAX_ATTRIBUTION_ROWS)
        self.assertTrue(result["selection"]["bounded"])

    def test_invalid_period_and_window_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "time_period"):
            paper_pnl_attribution(self.session, time_period="quarter")
        with self.assertRaisesRegex(ValueError, "end"):
            paper_pnl_attribution(self.session, start=self.at, end=self.at - timedelta(seconds=1))


if __name__ == "__main__":
    unittest.main()
