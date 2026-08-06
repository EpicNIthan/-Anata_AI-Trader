from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.raw_data_maintenance import router as raw_data_maintenance_router
from app.api.regime_pullback import router as regime_pullback_router
from app.api.routes import router as api_router
from app.api.v2 import router as v2_router
from app.api.vision import router as vision_router
from app.api.webhooks import router as webhook_router
from app.collectors.runtime import WorkerManager
from app.config import settings
from app.dashboard.routes import router as dashboard_router
from app.db.session import create_db_and_tables, ping_database
from app.logging_config import setup_logging
from app.services.auto_trader import AutoTraderService
from app.services.data_lifecycle import DataLifecycleService
from app.services.enrichment_service import EnrichmentService
from app.services.regime_label_builder import RegimeLabelMaintenanceService
from app.services.training_service import TrainingService
from app.strategies.regime_pullback_v1 import STRATEGY_NAME, STRATEGY_VERSION

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Importing the strategy service above registers its additive SQLAlchemy models
    # before create_all. No legacy table or row is dropped or rewritten.
    create_db_and_tables()
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    manager = WorkerManager()
    auto_trader = AutoTraderService()
    label_service = RegimeLabelMaintenanceService()
    data_lifecycle = DataLifecycleService(interval_seconds=settings.data_lifecycle_interval_seconds)
    training_service = TrainingService()
    enrichment_service = EnrichmentService()
    role = settings.worker_role.replace("_", "-")
    if role not in {"all", "web", "collector", "paper-trader", "enrichment"}:
        raise RuntimeError("WORKER_ROLE must be one of: all, web, collector, paper-trader, enrichment")
    app.state.worker_manager = manager
    app.state.auto_trader = auto_trader
    app.state.regime_label_service = label_service
    app.state.data_lifecycle = data_lifecycle
    app.state.training_service = training_service
    app.state.enrichment_service = enrichment_service
    app.state.worker_role = role
    run_collectors = role in {"all", "collector"}
    run_paper_trader = role in {"all", "paper-trader"}
    run_enrichment = role in {"all", "enrichment"}
    if run_collectors:
        await data_lifecycle.start()
    if run_collectors and settings.enable_market_collector:
        await manager.start("market")
    if run_collectors and settings.enable_spot_context_collector:
        await manager.start("spot_context")
    if run_collectors and settings.enable_news_collector:
        await manager.start("news")
    if run_collectors and settings.enable_derivatives_collector:
        await manager.start("derivatives")
    if run_collectors and any(
        [
            settings.enable_fear_greed_collector,
            settings.enable_global_market_collector,
            settings.enable_stablecoin_risk_collector,
            settings.enable_macro_risk_collector,
        ]
    ):
        await manager.start("external")
    if run_collectors and settings.enable_liquidation_collector:
        await manager.start("liquidations")
    if run_enrichment and settings.enrichment_enabled:
        await enrichment_service.start()
    if run_paper_trader and settings.auto_trader_enabled:
        await auto_trader.start()
        await label_service.start()
    try:
        yield
    finally:
        await label_service.stop()
        await data_lifecycle.stop()
        await enrichment_service.stop()
        await auto_trader.stop()
        await manager.stop_all()


app = FastAPI(
    title="Anata AI Crypto Trading Lab",
    version="0.2.0",
    description="Paper-only crypto trading research and data-collection lab.",
    lifespan=lifespan,
)

static_dir = Path("app/dashboard/static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")
app.include_router(webhook_router)
app.include_router(api_router)
app.include_router(v2_router)
app.include_router(vision_router)
app.include_router(raw_data_maintenance_router)
app.include_router(regime_pullback_router)
app.include_router(dashboard_router)


@app.get("/health")
def health() -> dict[str, object]:
    try:
        database_ok = ping_database()
    except Exception as exc:
        return {
            "status": "degraded",
            "database": False,
            "error": str(exc),
            "trading_mode": settings.trading_mode,
            "paper_only": True,
            "live_trading_enabled": False,
        }
    return {
        "status": "ok",
        "database": database_ok,
        "trading_mode": settings.trading_mode,
        "paper_only": True,
        "live_trading_enabled": False,
        "strategy_name": STRATEGY_NAME,
        "strategy_version": STRATEGY_VERSION,
        "worker_role": settings.worker_role,
        "symbols": getattr(getattr(app.state, "auto_trader", None), "symbols", settings.binance_symbols),
        "auto_trader_enabled": settings.auto_trader_enabled,
        "derivatives_enabled": settings.derivatives_enabled,
        "external_collectors_enabled": {
            "spot_context": settings.enable_spot_context_collector,
            "fear_greed": settings.enable_fear_greed_collector,
            "global_market": settings.enable_global_market_collector,
            "liquidations": settings.enable_liquidation_collector,
            "stablecoin_risk": settings.enable_stablecoin_risk_collector,
            "macro_risk": settings.enable_macro_risk_collector,
        },
    }
