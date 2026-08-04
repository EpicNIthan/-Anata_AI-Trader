"""Small, deterministic narrow models used by the paper-only V2 baseline.

They are intentionally independent from portfolio sizing and execution.  Each model
emits only a standardized :class:`ModelPrediction`, allowing stronger locally trained
artifacts to replace an individual family without changing risk or execution code.
"""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from app.pipeline.domain import FeatureSnapshot, HealthStatus, ModelPrediction, utc_now


def _value(values: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        result = values.get(key, default)
        return float(default if result is None else result)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-_clamp(value, -20.0, 20.0)))


@dataclass(frozen=True)
class PredictionDistribution:
    expected_return: float
    expected_volatility: float
    probability_up: float
    probability_down: float
    uncertainty: float


@dataclass
class NarrowModel(ABC):
    """Protocol-like base class for a frozen narrow forecasting model."""

    model_id: str
    model_family: str
    version: str = "baseline-v1"
    required_features: tuple[str, ...] = ()
    optional_features: tuple[str, ...] = ()
    forecast_horizon: int = 300
    _health: HealthStatus = field(default=HealthStatus.HEALTHY, init=False, repr=False)
    _fit_metadata: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def fit(self, rows: list[FeatureSnapshot], targets: list[float] | None = None) -> "NarrowModel":
        """Record lightweight calibration metadata without mutating production behavior.

        Deterministic baseline models do not fit a high-capacity estimator.  This method
        still gives local research a common interface and records sample counts/target
        moments for provenance.
        """
        target_values = [float(item) for item in (targets or []) if math.isfinite(float(item))]
        self._fit_metadata = {
            "rows": len(rows),
            "target_rows": len(target_values),
            "target_mean": sum(target_values) / len(target_values) if target_values else None,
        }
        return self

    def validate_inputs(self, snapshot: FeatureSnapshot) -> list[str]:
        missing = list(snapshot.missing_required_features)
        missing.extend(name for name in self.required_features if snapshot.values.get(name) in (None, ""))
        return sorted(set(missing))

    @abstractmethod
    def predict_distribution(self, snapshot: FeatureSnapshot) -> PredictionDistribution:
        """Return a forecast distribution for one point-in-time feature snapshot."""

    def predict(self, snapshot: FeatureSnapshot) -> ModelPrediction:
        missing = self.validate_inputs(snapshot)
        distribution = self.predict_distribution(snapshot)
        directional_certainty = abs(distribution.probability_up - distribution.probability_down)
        confidence = _clamp(
            (0.35 + (1.0 - distribution.uncertainty) * 0.35 + directional_certainty * 0.40)
            * (1.0 - min(len(missing) * 0.15, 0.7)),
            0.0,
            0.92,
        )
        now = utc_now()
        return ModelPrediction(
            model_id=self.model_id,
            model_version=self.version,
            model_family=self.model_family,
            symbol=snapshot.symbol,
            generated_at=now,
            valid_from=now,
            expires_at=now + __import__("datetime").timedelta(seconds=max(self.forecast_horizon, 1)),
            forecast_horizon_seconds=max(self.forecast_horizon, 1),
            expected_return=distribution.expected_return,
            expected_volatility=max(distribution.expected_volatility, 0.0),
            probability_up=_clamp(distribution.probability_up, 0.0, 1.0),
            probability_down=_clamp(distribution.probability_down, 0.0, 1.0),
            confidence=confidence,
            # A deterministic baseline carries a conservative prior, not an asserted
            # historical calibration result. Local evaluation overwrites this metadata
            # before a trained candidate can become champion.
            calibration_score=0.70 if not self._fit_metadata else _clamp(0.65 + min(self._fit_metadata.get("rows", 0), 10000) / 100000, 0.0, 0.80),
            uncertainty=_clamp(distribution.uncertainty + min(len(missing) * 0.1, 0.5), 0.0, 1.0),
            regime=classify_regime(snapshot.values),
            feature_schema_version=snapshot.schema_version,
            feature_snapshot_id=snapshot.feature_snapshot_id,
            data_version=snapshot.data_version,
            external_context_available=bool(snapshot.external_context.get("external_ai_available", False)),
            metadata={
                "missing_features": missing,
                "fit_metadata": self._fit_metadata,
                # Persist point-in-time provider lineage for later attribution.  The
                # provider remains null when external context was unavailable; no
                # request is inferred from timestamps.
                "external_ai_provider": snapshot.external_context.get("external_ai_provider"),
                "external_ai_prompt_version": snapshot.external_context.get("external_ai_prompt_version"),
                "external_ai_available": bool(snapshot.external_context.get("external_ai_available", False)),
                "external_ai_missing": bool(snapshot.external_context.get("external_ai_missing", True)),
                "external_ai_failed": bool(snapshot.external_context.get("external_ai_failed", False)),
                "external_ai_confidence": snapshot.external_context.get("external_ai_confidence"),
                "local_news_provider": snapshot.external_context.get("local_news_provider"),
                "local_news_model_version": snapshot.external_context.get("local_news_model_version"),
                **self.metadata(),
            },
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_family": self.model_family,
            "version": self.version,
            "required_features": list(self.required_features),
            "optional_features": list(self.optional_features),
            "forecast_horizon": self.forecast_horizon,
            "baseline": True,
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({**self.metadata(), "fit_metadata": self._fit_metadata}, indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> dict[str, Any]:
        """Load portable metadata; trained artifact loaders may override this method."""
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def health_status(self) -> HealthStatus:
        return self._health


def _distribution(score: float, volatility: float, *, scale: float = 0.0025, uncertainty: float = 0.45) -> PredictionDistribution:
    score = _clamp(score, -2.0, 2.0)
    expected_return = score * scale
    probability_up = _sigmoid(score * 1.8)
    probability_down = _sigmoid(-score * 1.8)
    return PredictionDistribution(
        expected_return=expected_return,
        expected_volatility=max(volatility, abs(expected_return) * 0.6, 0.0001),
        probability_up=probability_up,
        probability_down=probability_down,
        uncertainty=_clamp(uncertainty + min(volatility * 30.0, 0.25), 0.05, 0.95),
    )


@dataclass
class ShortHorizonMomentumModel(NarrowModel):
    model_id: str = "baseline-short-momentum"
    model_family: str = "alpha.short_horizon_momentum"
    required_features: tuple[str, ...] = ("candle_return_1m", "candle_return_5m", "trend_score")
    optional_features: tuple[str, ...] = ("macd_histogram_pct", "volume_change", "adx_14")
    forecast_horizon: int = 300

    def predict_distribution(self, snapshot: FeatureSnapshot) -> PredictionDistribution:
        v = snapshot.values
        score = (
            _value(v, "candle_return_1m") * 120.0
            + _value(v, "candle_return_5m") * 80.0
            + _value(v, "trend_score") * 0.55
            + _value(v, "macd_histogram_pct") * 70.0
            + max(_value(v, "volume_change"), 0.0) * 0.06
        )
        return _distribution(score, abs(_value(v, "volatility")), uncertainty=0.38)


@dataclass
class MediumHorizonMomentumModel(NarrowModel):
    model_id: str = "baseline-medium-momentum"
    model_family: str = "alpha.medium_horizon_momentum"
    required_features: tuple[str, ...] = ("price_change", "trend_score", "ema_20_distance_pct")
    optional_features: tuple[str, ...] = ("adx_14", "vwap_20_distance_pct")
    forecast_horizon: int = 1800

    def predict_distribution(self, snapshot: FeatureSnapshot) -> PredictionDistribution:
        v = snapshot.values
        score = (
            _value(v, "price_change") * 45.0
            + _value(v, "trend_score") * 0.85
            + _value(v, "ema_20_distance_pct") * 55.0
            + _value(v, "vwap_20_distance_pct") * 35.0
            + _value(v, "adx_14") * (1 if _value(v, "trend_score") >= 0 else -1) * 0.2
        )
        return _distribution(score, abs(_value(v, "volatility")), scale=0.0035, uncertainty=0.42)


@dataclass
class MeanReversionModel(NarrowModel):
    model_id: str = "baseline-mean-reversion"
    model_family: str = "alpha.mean_reversion"
    required_features: tuple[str, ...] = ("rsi_14", "bollinger_position", "regime_mean_reversion_pressure")
    optional_features: tuple[str, ...] = ("vwap_20_distance_pct", "volatility")
    forecast_horizon: int = 600

    def predict_distribution(self, snapshot: FeatureSnapshot) -> PredictionDistribution:
        v = snapshot.values
        pressure = _value(v, "regime_mean_reversion_pressure")
        score = (
            (0.5 - _value(v, "rsi_14", 0.5)) * 1.25
            + (0.5 - _value(v, "bollinger_position", 0.5)) * 1.0
            - _value(v, "vwap_20_distance_pct") * 30.0
        ) * max(pressure, 0.15)
        return _distribution(score, abs(_value(v, "volatility")), scale=0.002, uncertainty=0.48)


@dataclass
class BreakoutPressureModel(NarrowModel):
    model_id: str = "baseline-breakout-pressure"
    model_family: str = "alpha.breakout_pressure"
    required_features: tuple[str, ...] = ("regime_breakout_pressure", "trend_score", "volume_change")
    optional_features: tuple[str, ...] = ("adx_14", "atr_14_pct", "candle_return_5m")
    forecast_horizon: int = 900

    def predict_distribution(self, snapshot: FeatureSnapshot) -> PredictionDistribution:
        v = snapshot.values
        pressure = _value(v, "regime_breakout_pressure")
        score = (
            _value(v, "trend_score") * 0.9
            + _value(v, "candle_return_5m") * 70.0
            + max(_value(v, "volume_change"), 0.0) * 0.08
            + _value(v, "adx_14") * 0.25
        ) * max(pressure, 0.1)
        return _distribution(score, abs(_value(v, "atr_14_pct")), scale=0.003, uncertainty=0.45)


@dataclass
class DerivativesFlowModel(NarrowModel):
    model_id: str = "baseline-derivatives-flow"
    model_family: str = "alpha.derivatives_flow"
    required_features: tuple[str, ...] = ("taker_buy_pressure", "trader_crowd_score", "funding_rate")
    optional_features: tuple[str, ...] = ("open_interest_change", "crowd_risk_score")
    forecast_horizon: int = 600

    def predict_distribution(self, snapshot: FeatureSnapshot) -> PredictionDistribution:
        v = snapshot.values
        taker = (_value(v, "taker_buy_pressure", 0.5) - 0.5) * 2.0
        crowd = _value(v, "trader_crowd_score")
        funding = _value(v, "funding_rate") * 500.0
        oi = _value(v, "open_interest_change") * 12.0
        crowd_risk = _value(v, "crowd_risk_score")
        score = taker * 0.8 + crowd * 0.25 - funding * 0.25 + oi * 0.15
        return _distribution(score, abs(_value(v, "volatility")), scale=0.0022, uncertainty=0.42 + crowd_risk * 0.25)


@dataclass
class LiquidationPressureModel(NarrowModel):
    model_id: str = "baseline-liquidation-pressure"
    model_family: str = "alpha.liquidation_pressure"
    required_features: tuple[str, ...] = ("liquidation_imbalance_5m", "liquidation_spike_score")
    optional_features: tuple[str, ...] = ("regime_liquidity_stress_score",)
    forecast_horizon: int = 300

    def predict_distribution(self, snapshot: FeatureSnapshot) -> PredictionDistribution:
        v = snapshot.values
        spike = _value(v, "liquidation_spike_score")
        score = _value(v, "liquidation_imbalance_5m") * max(spike, 0.15) * 1.5
        return _distribution(score, abs(_value(v, "volatility")), scale=0.002, uncertainty=0.52 - min(spike * 0.15, 0.12))


@dataclass
class NewsEventModel(NarrowModel):
    model_id: str = "baseline-news-event"
    model_family: str = "alpha.news_event"
    required_features: tuple[str, ...] = ("sentiment_score", "risk_score", "impact_score")
    optional_features: tuple[str, ...] = ("sentiment_confidence", "recency_weight")
    forecast_horizon: int = 1800

    def predict_distribution(self, snapshot: FeatureSnapshot) -> PredictionDistribution:
        v = snapshot.values
        confidence = max(_value(v, "sentiment_confidence"), 0.2)
        recency = max(_value(v, "recency_weight"), 0.1)
        score = (_value(v, "sentiment_score") * confidence * 1.2 - _value(v, "risk_score") * 0.75) * max(_value(v, "impact_score"), 0.1) * recency
        external_missing = bool(snapshot.external_context.get("external_ai_missing", True))
        return _distribution(score, abs(_value(v, "volatility")), scale=0.0028, uncertainty=0.46 + (0.06 if external_missing else 0.0))


@dataclass
class CrossAssetContextModel(NarrowModel):
    model_id: str = "baseline-market-context"
    model_family: str = "alpha.market_context"
    required_features: tuple[str, ...] = ("market_regime_score", "fear_greed_value", "macro_risk_score")
    optional_features: tuple[str, ...] = ("btc_dominance_change", "global_market_cap_change_24h")
    forecast_horizon: int = 3600

    def predict_distribution(self, snapshot: FeatureSnapshot) -> PredictionDistribution:
        v = snapshot.values
        fear = (_value(v, "fear_greed_value") - 50.0) / 50.0
        score = (
            _value(v, "market_regime_score") * 0.45
            + fear * 0.15
            + _value(v, "global_market_cap_change_24h") * 5.0
            - _value(v, "macro_risk_score") * 0.55
        )
        return _distribution(score, abs(_value(v, "volatility")), scale=0.0018, uncertainty=0.52)


def classify_regime(values: dict[str, Any]) -> str:
    """Return one explicit regime, in priority order, from existing feature values."""
    if _value(values, "regime_news_shock_score") >= 0.55:
        return "news_shock"
    if _value(values, "regime_risk_off_score") >= 0.55:
        return "risk_off"
    if _value(values, "regime_liquidity_stress_score") >= 0.45:
        return "liquidity_stress"
    if _value(values, "regime_breakout_pressure") >= 0.5:
        return "breakout"
    if _value(values, "regime_mean_reversion_pressure") >= 0.5:
        return "mean_reversion"
    if _value(values, "regime_trend_strength") >= 0.35:
        return "trend_up" if _value(values, "regime_direction_score") >= 0 else "trend_down"
    if _value(values, "regime_volatility_score") >= 0.5:
        return "high_volatility"
    return "range"


@dataclass(frozen=True)
class MarketConditionEstimate:
    """One bounded, explainable market-condition classification."""

    model_id: str
    model_family: str
    label: str
    score: float
    confidence: float
    reason_codes: tuple[str, ...] = ()


class MarketConditionModel(ABC):
    """Execution-independent interface for specialized regime classifiers."""

    model_id: ClassVar[str]
    model_family: ClassVar[str]
    required_features: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def classify(self, snapshot: FeatureSnapshot) -> MarketConditionEstimate:
        """Classify one point-in-time snapshot without sizing or execution authority."""

    def validate_inputs(self, snapshot: FeatureSnapshot) -> tuple[str, ...]:
        return tuple(name for name in self.required_features if snapshot.values.get(name) in (None, ""))


class TrendRegimeModel(MarketConditionModel):
    model_id = "baseline-trend-regime"
    model_family = "condition.trend_regime"
    required_features = ("regime_trend_strength", "regime_direction_score")

    def classify(self, snapshot: FeatureSnapshot) -> MarketConditionEstimate:
        strength = _clamp(abs(_value(snapshot.values, "regime_trend_strength")), 0.0, 1.0)
        direction = _value(snapshot.values, "regime_direction_score")
        label = "range" if strength < 0.35 else "trend_up" if direction >= 0 else "trend_down"
        return MarketConditionEstimate(
            self.model_id,
            self.model_family,
            label,
            _clamp(direction * strength, -1.0, 1.0),
            _clamp(0.5 + strength * 0.5, 0.0, 1.0),
            ("TREND_STRENGTH_CLASSIFIED",),
        )


class VolatilityRegimeModel(MarketConditionModel):
    model_id = "baseline-volatility-regime"
    model_family = "condition.volatility_regime"
    required_features = ("regime_volatility_score",)

    def classify(self, snapshot: FeatureSnapshot) -> MarketConditionEstimate:
        score = _clamp(
            max(
                _value(snapshot.values, "regime_volatility_score"),
                abs(_value(snapshot.values, "volatility")) * 40.0,
            ),
            0.0,
            1.0,
        )
        label = "low_volatility" if score < 0.25 else "high_volatility" if score >= 0.60 else "normal_volatility"
        return MarketConditionEstimate(
            self.model_id,
            self.model_family,
            label,
            score,
            _clamp(0.55 + abs(score - 0.45) * 0.7, 0.0, 1.0),
            ("VOLATILITY_LEVEL_CLASSIFIED",),
        )


class LiquidityRegimeModel(MarketConditionModel):
    model_id = "baseline-liquidity-regime"
    model_family = "condition.liquidity_regime"
    required_features = ("regime_liquidity_stress_score",)

    def classify(self, snapshot: FeatureSnapshot) -> MarketConditionEstimate:
        spread_stress = _value(snapshot.values, "spot_bid_ask_spread_pct") * 150.0
        score = _clamp(max(_value(snapshot.values, "regime_liquidity_stress_score"), spread_stress), 0.0, 1.0)
        label = "liquidity_stress" if score >= 0.55 else "thin_liquidity" if score >= 0.30 else "normal_liquidity"
        return MarketConditionEstimate(
            self.model_id,
            self.model_family,
            label,
            score,
            _clamp(0.6 + abs(score - 0.4) * 0.5, 0.0, 1.0),
            ("LIQUIDITY_STRESS_CLASSIFIED",),
        )


class CrowdingRegimeModel(MarketConditionModel):
    model_id = "baseline-crowding-regime"
    model_family = "condition.crowding_regime"
    required_features = ("crowd_risk_score", "funding_rate")

    def classify(self, snapshot: FeatureSnapshot) -> MarketConditionEstimate:
        score = _clamp(
            max(
                _value(snapshot.values, "crowd_risk_score"),
                abs(_value(snapshot.values, "funding_rate")) * 500.0,
            ),
            0.0,
            1.0,
        )
        label = "crowded" if score >= 0.60 else "watch" if score >= 0.35 else "balanced"
        return MarketConditionEstimate(
            self.model_id,
            self.model_family,
            label,
            score,
            _clamp(0.55 + abs(score - 0.45) * 0.6, 0.0, 1.0),
            ("CROWDING_CLASSIFIED",),
        )


class NewsShockRegimeModel(MarketConditionModel):
    model_id = "baseline-news-shock-regime"
    model_family = "condition.news_shock"
    required_features = ("regime_news_shock_score",)

    def classify(self, snapshot: FeatureSnapshot) -> MarketConditionEstimate:
        score = _clamp(
            max(
                _value(snapshot.values, "regime_news_shock_score"),
                _value(snapshot.values, "impact_score") * _value(snapshot.values, "recency_weight"),
            ),
            0.0,
            1.0,
        )
        label = "news_shock" if score >= 0.55 else "news_active" if score >= 0.25 else "news_normal"
        return MarketConditionEstimate(
            self.model_id,
            self.model_family,
            label,
            score,
            _clamp(0.55 + abs(score - 0.4) * 0.6, 0.0, 1.0),
            ("NEWS_SHOCK_CLASSIFIED",),
        )


class RiskAppetiteRegimeModel(MarketConditionModel):
    model_id = "baseline-risk-appetite-regime"
    model_family = "condition.risk_appetite"
    required_features = ("regime_risk_off_score", "market_regime_score")

    def classify(self, snapshot: FeatureSnapshot) -> MarketConditionEstimate:
        risk_off = _clamp(
            max(_value(snapshot.values, "regime_risk_off_score"), _value(snapshot.values, "macro_risk_score")),
            0.0,
            1.0,
        )
        risk_on = _clamp(max(_value(snapshot.values, "market_regime_score"), 0.0), 0.0, 1.0)
        score = _clamp(risk_on - risk_off, -1.0, 1.0)
        label = "risk_off" if score <= -0.25 else "risk_on" if score >= 0.25 else "risk_neutral"
        return MarketConditionEstimate(
            self.model_id,
            self.model_family,
            label,
            score,
            _clamp(0.55 + abs(score) * 0.4, 0.0, 1.0),
            ("RISK_APPETITE_CLASSIFIED",),
        )


def default_market_condition_models() -> list[MarketConditionModel]:
    """Return all functional lightweight condition-model families."""

    return [
        TrendRegimeModel(),
        VolatilityRegimeModel(),
        LiquidityRegimeModel(),
        CrowdingRegimeModel(),
        NewsShockRegimeModel(),
        RiskAppetiteRegimeModel(),
    ]


@dataclass(frozen=True)
class CostEstimate:
    spread: float
    slippage: float
    fee: float
    funding: float
    fill_probability: float

    @property
    def total_cost(self) -> float:
        return max(self.spread, 0.0) + max(self.slippage, 0.0) + max(self.fee, 0.0) + max(self.funding, 0.0)


class SpreadEstimateModel:
    model_id = "baseline-spread-estimate"
    model_family = "cost.spread"

    def estimate(self, snapshot: FeatureSnapshot) -> float:
        observed = _value(snapshot.values, "spot_bid_ask_spread_pct")
        fallback = min(abs(_value(snapshot.values, "volatility")) * 0.05, 0.002)
        return _clamp(observed if observed > 0 else fallback, 0.0, 0.02)


class SlippageEstimateModel:
    model_id = "baseline-slippage-estimate"
    model_family = "cost.slippage"

    def estimate(self, snapshot: FeatureSnapshot) -> float:
        volatility = abs(_value(snapshot.values, "volatility"))
        stress = _clamp(_value(snapshot.values, "regime_liquidity_stress_score"), 0.0, 1.0)
        return _clamp(0.00005 + volatility * 0.4 + stress * 0.001, 0.0, 0.01)


class FundingCostEstimateModel:
    model_id = "baseline-funding-cost"
    model_family = "cost.funding"

    def estimate(self, snapshot: FeatureSnapshot) -> float:
        # One half funding interval is the conservative default forecast-horizon proxy.
        return _clamp(abs(_value(snapshot.values, "funding_rate")) * 0.5, 0.0, 0.01)


class FillProbabilityEstimateModel:
    model_id = "baseline-fill-probability"
    model_family = "cost.fill_probability"

    def estimate(self, snapshot: FeatureSnapshot, *, spread: float | None = None) -> float:
        spread = SpreadEstimateModel().estimate(snapshot) if spread is None else max(float(spread), 0.0)
        stress = _clamp(_value(snapshot.values, "regime_liquidity_stress_score"), 0.0, 1.0)
        missing_penalty = min(len(snapshot.missing_required_features) * 0.08, 0.40)
        return _clamp(1.0 - max(spread * 150.0, stress) - missing_penalty, 0.0, 1.0)


class TransactionCostEstimateModel:
    model_id = "baseline-transaction-cost"
    model_family = "cost.transaction"

    @staticmethod
    def estimate(*, spread: float, slippage: float, fee: float, funding: float) -> float:
        return sum(max(float(value), 0.0) for value in (spread, slippage, fee, funding))


class BaselineCostModel:
    """Estimate paper transaction costs from features without emitting an order."""

    model_id: ClassVar[str] = "baseline-cost-model"
    model_family: ClassVar[str] = "cost.transaction"

    def __init__(self) -> None:
        self.spread_model = SpreadEstimateModel()
        self.slippage_model = SlippageEstimateModel()
        self.funding_model = FundingCostEstimateModel()
        self.fill_model = FillProbabilityEstimateModel()
        self.transaction_model = TransactionCostEstimateModel()

    def estimate(self, snapshot: FeatureSnapshot, *, fee_rate: float = 0.0004) -> CostEstimate:
        spread = self.spread_model.estimate(snapshot)
        slippage = self.slippage_model.estimate(snapshot)
        funding = self.funding_model.estimate(snapshot)
        liquidity = self.fill_model.estimate(snapshot, spread=spread)
        self.transaction_model.estimate(
            spread=spread,
            slippage=slippage,
            fee=max(fee_rate, 0.0),
            funding=funding,
        )
        return CostEstimate(spread=spread, slippage=slippage, fee=max(fee_rate, 0.0), funding=funding, fill_probability=liquidity)


@dataclass(frozen=True)
class ReliabilityEstimate:
    calibration: float
    uncertainty: float
    feature_drift: float
    out_of_distribution: float
    data_quality_confidence: float

    @property
    def confidence(self) -> float:
        return _clamp(
            self.calibration
            * (1.0 - self.uncertainty)
            * (1.0 - self.feature_drift)
            * (1.0 - self.out_of_distribution)
            * self.data_quality_confidence,
            0.0,
            1.0,
        )


class PredictionCalibrationModel:
    model_id = "baseline-prediction-calibration"
    model_family = "reliability.calibration"

    def estimate(self, snapshot: FeatureSnapshot) -> float:
        return _clamp(_value(snapshot.values, "historical_calibration_score", 0.70), 0.0, 1.0)


class PredictionUncertaintyModel:
    model_id = "baseline-prediction-uncertainty"
    model_family = "reliability.uncertainty"

    def estimate(self, snapshot: FeatureSnapshot) -> float:
        volatility = min(abs(_value(snapshot.values, "volatility")) * 10.0, 0.25)
        missing = min(len(snapshot.missing_required_features) * 0.15, 0.60)
        return _clamp(0.20 + volatility + missing, 0.0, 1.0)


class FeatureDriftDetectionModel:
    model_id = "baseline-feature-drift"
    model_family = "reliability.feature_drift"

    def estimate(self, snapshot: FeatureSnapshot) -> float:
        return _clamp(_value(snapshot.values, "feature_drift_score", 0.0), 0.0, 1.0)


class OutOfDistributionDetectionModel:
    model_id = "baseline-out-of-distribution"
    model_family = "reliability.out_of_distribution"

    def estimate(self, snapshot: FeatureSnapshot) -> float:
        explicit = _value(snapshot.values, "out_of_distribution_score", 0.0)
        non_finite = sum(
            1
            for value in snapshot.values.values()
            if isinstance(value, float) and not math.isfinite(value)
        )
        return _clamp(max(explicit, non_finite / max(len(snapshot.values), 1)), 0.0, 1.0)


class DataQualityConfidenceModel:
    model_id = "baseline-data-quality-confidence"
    model_family = "reliability.data_quality"

    def estimate(self, snapshot: FeatureSnapshot) -> float:
        stale = max(snapshot.source_freshness_seconds.values(), default=0.0)
        missing_penalty = min(len(snapshot.missing_required_features) * 0.15, 0.70)
        stale_penalty = min(stale / 1800.0, 0.60)
        return _clamp(1.0 - missing_penalty - stale_penalty, 0.0, 1.0)


class BaselineReliabilityModel:
    """Compose independent reliability estimates into a bounded quality score."""

    model_id: ClassVar[str] = "baseline-reliability-model"
    model_family: ClassVar[str] = "reliability.composite"

    def __init__(self) -> None:
        self.calibration_model = PredictionCalibrationModel()
        self.uncertainty_model = PredictionUncertaintyModel()
        self.drift_model = FeatureDriftDetectionModel()
        self.ood_model = OutOfDistributionDetectionModel()
        self.data_quality_model = DataQualityConfidenceModel()

    def assess(self, snapshot: FeatureSnapshot) -> ReliabilityEstimate:
        return ReliabilityEstimate(
            calibration=self.calibration_model.estimate(snapshot),
            uncertainty=self.uncertainty_model.estimate(snapshot),
            feature_drift=self.drift_model.estimate(snapshot),
            out_of_distribution=self.ood_model.estimate(snapshot),
            data_quality_confidence=self.data_quality_model.estimate(snapshot),
        )

    def confidence(self, snapshot: FeatureSnapshot) -> tuple[float, float]:
        estimate = self.assess(snapshot)
        combined_uncertainty = _clamp(
            max(
                estimate.uncertainty,
                estimate.feature_drift,
                estimate.out_of_distribution,
                1.0 - estimate.data_quality_confidence,
            ),
            0.0,
            1.0,
        )
        return estimate.confidence, combined_uncertainty


def default_narrow_models() -> list[NarrowModel]:
    """Return the independent lightweight baseline model set used with no artifact."""
    return [
        ShortHorizonMomentumModel(),
        MediumHorizonMomentumModel(),
        MeanReversionModel(),
        BreakoutPressureModel(),
        DerivativesFlowModel(),
        LiquidationPressureModel(),
        NewsEventModel(),
        CrossAssetContextModel(),
    ]
