from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.raw_data_maintenance import router as raw_data_maintenance_router
from app.api.routes import router as api_router
from app.api.webhooks import router as webhook_router
from app.collectors.runtime import WorkerManager
from app.config import settings
from app.dashboard.routes import router as dashboard_router
from app.db.session import create_db_and_tables, ping_database
from app.logging_config import setup_logging
from app.services.auto_trader import AutoTraderService
from app.services.data_lifecycle import DataLifecycleService
from app.services.training_service import TrainingService

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    manager = WorkerManager()
    auto_trader = AutoTraderService()
    data_lifecycle = DataLifecycleService(interval_seconds=settings.data_lifecycle_interval_seconds)
    training_service = TrainingService()
    app.state.worker_manager = manager
    app.state.auto_trader = auto_trader
    app.state.data_lifecycle = data_lifecycle
    app.state.training_service = training_service
    await data_lifecycle.start()
    if settings.enable_market_collector:
        await manager.start("market")
    if settings.enable_spot_context_collector:
        await manager.start("spot_context")
    if settings.enable_news_collector:
        await manager.start("news")
    if settings.enable_derivatives_collector:
        await manager.start("derivatives")
    if any(
        [
            settings.enable_fear_greed_collector,
            settings.enable_global_market_collector,
            settings.enable_stablecoin_risk_collector,
            settings.enable_macro_risk_collector,
        ]
    ):
        await manager.start("external")
    if settings.enable_liquidation_collector:
        await manager.start("liquidations")
    if settings.auto_trader_enabled:
        await auto_trader.start()
    try:
        yield
    finally:
        await data_lifecycle.stop()
        await auto_trader.stop()
        await manager.stop_all()


app = FastAPI(
    title="Anata AI Crypto Trading Lab",
    version="0.1.0",
    description="Paper-only AI crypto trading research lab.",
    lifespan=lifespan,
)

static_dir = Path("app/dashboard/static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")
app.include_router(webhook_router)
app.include_router(api_router)
app.include_router(raw_data_maintenance_router)
app.include_router(dashboard_router)


@app.get("/health")
def health() -> dict[str, object]:
    try:
        database_ok = ping_database()
    except Exception as exc:
        return {"status": "degraded", "database": False, "error": str(exc), "trading_mode": settings.trading_mode}
    return {
        "status": "ok",
        "database": database_ok,
        "trading_mode": settings.trading_mode,
        "symbols": settings.binance_symbols,
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
