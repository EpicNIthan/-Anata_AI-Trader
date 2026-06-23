from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ["DATABASE_URL"] = "sqlite:///./_smoke_trading_lab.db"
os.environ["TRADING_MODE"] = "paper"
os.environ["ENABLE_MARKET_COLLECTOR"] = "false"
os.environ["ENABLE_NEWS_COLLECTOR"] = "false"
os.environ["AUTO_TRADER_ENABLED"] = "false"
os.environ["AUTO_TRADER_INTERVAL_SECONDS"] = "1"
os.environ["AUTO_TRADER_SYMBOLS"] = "BTCUSDT"
os.environ["PAPER_START_BALANCE"] = "10000"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from sqlalchemy import func, inspect, select

from app.db.models import AiDecision, Candle, ExperienceRecord, Feature
from app.db.session import engine
from app.db.session import SessionLocal
from app.main import app


def main() -> None:
    db_path = Path("_smoke_trading_lab.db")
    if db_path.exists():
        db_path.unlink()

    required_tables = {
        "candles",
        "news_articles",
        "news_sentiment",
        "features",
        "paper_trades",
        "positions",
        "model_versions",
        "training_runs",
        "ai_decisions",
        "account_equity",
        "experience_buffer",
        "external_data_events",
        "market_ticks",
    }

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200, health.text
        assert health.json()["status"] == "ok", health.text

        dashboard = client.get("/dashboard")
        assert dashboard.status_code == 200, dashboard.text[:500]
        assert "Anata AI Trading Lab" in dashboard.text

        trade = client.post(
            "/api/signal",
            json={
                "symbol": "BTCUSDT",
                "action": "BUY",
                "confidence": 0.90,
                "price": 50000,
                "notional": 250,
                "reason": "smoke test",
                "source": "smoke-test",
            },
        )
        assert trade.status_code == 200, trade.text
        assert trade.json()["status"] == "FILLED", trade.text
        assert trade.json()["decision_id"] > 0, trade.text

        experiences = client.get("/api/experiences")
        assert experiences.status_code == 200, experiences.text
        assert len(experiences.json()) >= 1, experiences.text

        collector_status = client.get("/api/collectors/status")
        assert collector_status.status_code == 200, collector_status.text

        market_backfill = client.post("/api/market/backfill", json={"symbols": ["BTCUSDT"], "limit": 100, "mock": True})
        assert market_backfill.status_code == 200, market_backfill.text
        assert market_backfill.json()["rows_saved"] > 0, market_backfill.text

        market_status = client.get("/api/market/status")
        assert market_status.status_code == 200, market_status.text
        assert market_status.json()["candle_count"] > 0, market_status.text

        with SessionLocal() as session:
            decisions_before = session.scalar(select(func.count(AiDecision.id))) or 0
            experiences_before = session.scalar(select(func.count(ExperienceRecord.id))) or 0
            candles_before = session.scalar(select(func.count(Candle.id))) or 0
        assert candles_before > 0

        auto_start = client.post("/api/auto-trader/start")
        assert auto_start.status_code == 200, auto_start.text
        assert auto_start.json()["running"] is True, auto_start.text

        auto_status = {}
        for _ in range(20):
            auto_status = client.get("/api/auto-trader/status").json()
            if auto_status.get("cycles", 0) >= 1:
                break
            time.sleep(0.25)
        assert auto_status.get("cycles", 0) >= 1, auto_status
        assert auto_status.get("last_run_at"), auto_status

        with SessionLocal() as session:
            decisions_after = session.scalar(select(func.count(AiDecision.id))) or 0
            experiences_after = session.scalar(select(func.count(ExperienceRecord.id))) or 0
            latest_feature = session.scalar(select(Feature).order_by(Feature.created_at.desc()).limit(1))
        assert decisions_after > decisions_before
        assert experiences_after > experiences_before
        assert latest_feature is not None
        assert (latest_feature.payload or {}).get("values", {}).get("candles_used", 0) > 0
        assert (latest_feature.payload or {}).get("values", {}).get("last_close") is not None

        auto_stop = client.post("/api/auto-trader/stop")
        assert auto_stop.status_code == 200, auto_stop.text
        assert auto_stop.json()["running"] is False, auto_stop.text

        tables = set(inspect(engine).get_table_names())
        missing = required_tables - tables
        assert not missing, f"Missing tables: {sorted(missing)}"

    engine.dispose()
    if db_path.exists():
        db_path.unlink()
    print("smoke_ok=true")


if __name__ == "__main__":
    main()
