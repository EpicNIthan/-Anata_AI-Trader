"""Safe, typed news-intelligence primitives.

The package is deliberately independent from portfolio, risk, and execution code.
It can enrich a news document, but it cannot create an order or alter a risk limit.
"""

from .schemas import (
    EventDirection,
    EventType,
    IntelligenceResult,
    IntelligenceValidationError,
    NewsDocument,
    ProviderRequestAudit,
    StructuredNewsEvent,
    TimeHorizon,
)
from .providers import (
    GenericOpenAICompatibleProvider,
    IntelligenceProvider,
    LocalRuleProvider,
    LocalStudentProvider,
)
from .service import IntelligenceRouter, RouterPolicy

__all__ = [
    "EventDirection",
    "EventType",
    "GenericOpenAICompatibleProvider",
    "IntelligenceProvider",
    "IntelligenceResult",
    "IntelligenceRouter",
    "IntelligenceValidationError",
    "LocalRuleProvider",
    "LocalStudentProvider",
    "NewsDocument",
    "ProviderRequestAudit",
    "RouterPolicy",
    "StructuredNewsEvent",
    "TimeHorizon",
]
