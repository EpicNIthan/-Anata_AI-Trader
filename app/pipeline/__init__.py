"""Paper-only quantitative decision pipeline.

The package deliberately keeps the dependency direction one-way:
data/features -> models -> signals -> ensemble -> portfolio -> risk -> execution.
Model and signal modules do not import paper execution.
"""

from app.pipeline.domain import (
    EnsembleDecision,
    ModelPrediction,
    PortfolioTarget,
    RiskDecision,
    SimulatedFill,
    SimulatedOrder,
    TradingSignal,
)

__all__ = [
    "EnsembleDecision",
    "ModelPrediction",
    "PortfolioTarget",
    "RiskDecision",
    "SimulatedFill",
    "SimulatedOrder",
    "TradingSignal",
]
