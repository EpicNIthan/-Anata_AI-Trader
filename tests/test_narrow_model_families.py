from __future__ import annotations

import unittest
from datetime import datetime, timezone

from app.pipeline.domain import FeatureSnapshot
from app.pipeline.narrow_models import (
    BaselineCostModel,
    BaselineReliabilityModel,
    default_market_condition_models,
    default_narrow_models,
)


class NarrowModelFamilyTests(unittest.TestCase):
    @staticmethod
    def _snapshot() -> FeatureSnapshot:
        now = datetime.now(timezone.utc)
        return FeatureSnapshot(
            symbol="BTCUSDT",
            as_of=now,
            available_to_model_time=now,
            schema_version="price-news-market-v5",
            values={
                "candle_return_1m": 0.001,
                "candle_return_5m": 0.002,
                "price_change": 0.002,
                "trend_score": 0.4,
                "ema_20_distance_pct": 0.001,
                "rsi_14": 0.55,
                "bollinger_position": 0.6,
                "regime_mean_reversion_pressure": 0.2,
                "regime_breakout_pressure": 0.4,
                "volume_change": 0.2,
                "taker_buy_pressure": 0.55,
                "trader_crowd_score": 0.1,
                "funding_rate": 0.0001,
                "liquidation_imbalance_5m": -0.2,
                "liquidation_spike_score": 0.3,
                "sentiment_score": 0.2,
                "risk_score": 0.1,
                "impact_score": 0.4,
                "sentiment_confidence": 0.7,
                "recency_weight": 0.8,
                "market_regime_score": 0.2,
                "fear_greed_value": 55,
                "macro_risk_score": 0.1,
                "regime_trend_strength": 0.6,
                "regime_direction_score": 0.5,
                "regime_volatility_score": 0.3,
                "regime_liquidity_stress_score": 0.2,
                "regime_news_shock_score": 0.1,
                "regime_risk_off_score": 0.15,
                "crowd_risk_score": 0.2,
                "spot_bid_ask_spread_pct": 0.0002,
                "volatility": 0.003,
            },
            external_context={
                "external_ai_available": True,
                "external_ai_missing": False,
                "external_ai_failed": False,
                "external_ai_provider": "unit-provider",
                "external_ai_prompt_version": "unit-prompt-v1",
                "external_ai_confidence": 0.75,
                "local_news_model_version": "student-v1",
            },
        )

    def test_all_alpha_baselines_emit_forecasts_only_with_provider_lineage(self) -> None:
        snapshot = self._snapshot()
        models = default_narrow_models()
        self.assertEqual(len(models), 8)
        self.assertEqual(len({model.model_family for model in models}), 8)
        for model in models:
            prediction = model.predict(snapshot)
            payload = prediction.model_dump()
            self.assertNotIn("leverage", payload)
            self.assertNotIn("margin", payload)
            self.assertNotIn("order", payload)
            self.assertEqual(prediction.metadata["external_ai_provider"], "unit-provider")
            self.assertEqual(prediction.metadata["external_ai_prompt_version"], "unit-prompt-v1")

    def test_all_condition_cost_and_reliability_families_are_functional_and_bounded(self) -> None:
        snapshot = self._snapshot()
        conditions = [model.classify(snapshot) for model in default_market_condition_models()]
        self.assertEqual(len(conditions), 6)
        self.assertEqual(len({item.model_family for item in conditions}), 6)
        self.assertTrue(all(-1.0 <= item.score <= 1.0 for item in conditions))
        self.assertTrue(all(0.0 <= item.confidence <= 1.0 for item in conditions))

        cost_model = BaselineCostModel()
        cost = cost_model.estimate(snapshot, fee_rate=0.0004)
        self.assertGreater(cost.total_cost, 0.0)
        self.assertGreaterEqual(cost.fill_probability, 0.0)
        self.assertLessEqual(cost.fill_probability, 1.0)
        self.assertEqual(
            {
                cost_model.spread_model.model_family,
                cost_model.slippage_model.model_family,
                cost_model.funding_model.model_family,
                cost_model.fill_model.model_family,
                cost_model.transaction_model.model_family,
            },
            {"cost.spread", "cost.slippage", "cost.funding", "cost.fill_probability", "cost.transaction"},
        )

        reliability_model = BaselineReliabilityModel()
        estimate = reliability_model.assess(snapshot)
        confidence, uncertainty = reliability_model.confidence(snapshot)
        self.assertTrue(0.0 <= estimate.calibration <= 1.0)
        self.assertTrue(0.0 <= confidence <= 1.0)
        self.assertTrue(0.0 <= uncertainty <= 1.0)
        self.assertEqual(
            {
                reliability_model.calibration_model.model_family,
                reliability_model.uncertainty_model.model_family,
                reliability_model.drift_model.model_family,
                reliability_model.ood_model.model_family,
                reliability_model.data_quality_model.model_family,
            },
            {
                "reliability.calibration",
                "reliability.uncertainty",
                "reliability.feature_drift",
                "reliability.out_of_distribution",
                "reliability.data_quality",
            },
        )


if __name__ == "__main__":
    unittest.main()

