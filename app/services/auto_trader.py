"""Compatibility import for the only automatic paper-trading engine.

The former minute-loop/model/exploration implementation was removed. Existing API
routes can continue importing ``AutoTraderService`` while startup now resolves to the
single regime_pullback_v1 scheduler.
"""

from app.services.regime_pullback_service import AutoTraderService, AutoTraderState

__all__ = ["AutoTraderService", "AutoTraderState"]
