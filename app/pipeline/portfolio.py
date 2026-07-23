"""Independent, deterministic portfolio target construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from app.pipeline.domain import EnsembleDecision, EnsembleStatus, PortfolioTarget


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class PortfolioContext:
    """Current paper portfolio represented as signed fractions of equity."""

    equity: float
    exposures: Mapping[str, float] = field(default_factory=dict)
    cluster_by_symbol: Mapping[str, str] = field(default_factory=dict)
    liquidity_score: float = 0.5
    current_gross_exposure: float | None = None
    current_net_exposure: float | None = None

    @property
    def gross_exposure(self) -> float:
        return self.current_gross_exposure if self.current_gross_exposure is not None else sum(abs(value) for value in self.exposures.values())

    @property
    def net_exposure(self) -> float:
        return self.current_net_exposure if self.current_net_exposure is not None else sum(self.exposures.values())

    def cluster_exposure(self, symbol: str) -> float:
        cluster = self.cluster_by_symbol.get(symbol, symbol)
        return sum(abs(value) for key, value in self.exposures.items() if self.cluster_by_symbol.get(key, key) == cluster)


class DeterministicPortfolioConstructor:
    """Translate an opportunity estimate into a bounded target exposure.

    This deliberately handles only exposure fractions.  It does not pick leverage,
    order type, quantity, stop, margin or broker instructions.
    """

    def __init__(
        self,
        *,
        max_symbol_exposure: float = 0.10,
        max_gross_exposure: float = 0.40,
        max_net_exposure: float = 0.25,
        max_cluster_exposure: float = 0.25,
        minimum_liquidity: float = 0.20,
        minimum_edge: float = 0.0005,
    ) -> None:
        self.max_symbol_exposure = max(max_symbol_exposure, 0.0)
        self.max_gross_exposure = max(max_gross_exposure, 0.0)
        self.max_net_exposure = max(max_net_exposure, 0.0)
        self.max_cluster_exposure = max(max_cluster_exposure, 0.0)
        self.minimum_liquidity = _clip(minimum_liquidity, 0.0, 1.0)
        self.minimum_edge = max(minimum_edge, 1e-9)

    def construct(self, ensemble: EnsembleDecision, context: PortfolioContext) -> PortfolioTarget:
        current = float(context.exposures.get(ensemble.symbol, 0.0))
        if ensemble.decision_status != EnsembleStatus.ACTIONABLE or context.equity <= 0:
            requested = 0.0
            urgency = 0.0
        else:
            liquidity = _clip(context.liquidity_score, 0.0, 1.0)
            if liquidity < self.minimum_liquidity:
                requested = 0.0
                urgency = 0.0
            else:
                edge_score = _clip(abs(ensemble.combined_expected_return) / (self.minimum_edge * 6.0), 0.0, 1.0)
                risk_adjustment = 1.0 - _clip(
                    ensemble.combined_uncertainty * 0.55 + ensemble.combined_expected_volatility * 25.0,
                    0.0,
                    0.85,
                )
                confidence_adjustment = _clip(ensemble.combined_confidence, 0.0, 1.0)
                liquidity_adjustment = _clip((liquidity - self.minimum_liquidity) / max(1.0 - self.minimum_liquidity, 1e-9), 0.0, 1.0)
                raw = self.max_symbol_exposure * edge_score * risk_adjustment * confidence_adjustment * liquidity_adjustment
                direction = 1.0 if ensemble.combined_expected_return > 0 else -1.0
                requested = direction * raw
                requested = self._apply_portfolio_caps(ensemble.symbol, requested, context)
                urgency = _clip(edge_score * confidence_adjustment, 0.0, 1.0)
        delta = requested - current
        return PortfolioTarget(
            symbol=ensemble.symbol,
            current_exposure=current,
            requested_target_exposure=requested,
            requested_delta=delta,
            expected_return=ensemble.combined_expected_return,
            expected_risk=max(ensemble.combined_uncertainty, ensemble.combined_expected_volatility),
            risk_contribution=abs(requested) * max(ensemble.combined_uncertainty, 0.01),
            urgency=urgency,
            source_ensemble_decision_id=ensemble.ensemble_decision_id,
        )

    def _apply_portfolio_caps(self, symbol: str, requested: float, context: PortfolioContext) -> float:
        requested = _clip(requested, -self.max_symbol_exposure, self.max_symbol_exposure)
        existing_without_symbol = context.gross_exposure - abs(float(context.exposures.get(symbol, 0.0)))
        gross_room = max(self.max_gross_exposure - existing_without_symbol, 0.0)
        requested = _clip(requested, -gross_room, gross_room)
        net_without_symbol = context.net_exposure - float(context.exposures.get(symbol, 0.0))
        low = -self.max_net_exposure - net_without_symbol
        high = self.max_net_exposure - net_without_symbol
        requested = _clip(requested, low, high)
        cluster_without_symbol = context.cluster_exposure(symbol) - abs(float(context.exposures.get(symbol, 0.0)))
        cluster_room = max(self.max_cluster_exposure - cluster_without_symbol, 0.0)
        return _clip(requested, -cluster_room, cluster_room)
