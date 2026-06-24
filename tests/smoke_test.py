from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMOKE_FEED_PATH = ROOT / "_smoke_feed.xml"
SMOKE_FEED_PATH.write_text(
    """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Smoke Crypto Feed</title>
    <item>
      <title>Bitcoin and Ethereum liquidity improves</title>
      <link>https://example.com/smoke-bitcoin-ethereum</link>
      <pubDate>Wed, 24 Jun 2026 04:00:00 GMT</pubDate>
      <description>BTC, ETH, Solana, BNB, and XRP traders watch inflation and risk.</description>
    </item>
  </channel>
</rss>
""",
    encoding="utf-8",
)

os.environ["DATABASE_URL"] = "sqlite:///./_smoke_trading_lab.db"
os.environ["TRADING_MODE"] = "paper"
os.environ["ENABLE_MARKET_COLLECTOR"] = "false"
os.environ["ENABLE_NEWS_COLLECTOR"] = "false"
os.environ["AUTO_TRADER_ENABLED"] = "false"
os.environ["AUTO_TRADER_INTERVAL_SECONDS"] = "1"
os.environ["AUTO_TRADER_SYMBOLS"] = "BTCUSDT"
os.environ["NEWS_PROVIDER"] = "rss,gdelt,newsapi"
os.environ["RSS_NEWS_ENABLED"] = "true"
os.environ["RSS_FEEDS"] = f"file://{SMOKE_FEED_PATH}"
os.environ["GDELT_ENABLED"] = "false"
os.environ["NEWSAPI_ENABLED"] = "false"
os.environ["NEWS_API_KEY"] = ""
os.environ["ENABLE_HF_SENTIMENT"] = "false"
os.environ["PAPER_START_BALANCE"] = "10000"
os.environ["DASHBOARD_USERNAME"] = "admin"
os.environ["DASHBOARD_PASSWORD"] = "secret"
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from sqlalchemy import func, inspect, select

from app.db.models import AiDecision, Candle, ExperienceRecord, Feature, NewsArticle, NewsSentiment
from app.db.session import engine
from app.db.session import SessionLocal
from app.main import app


def main() -> None:
    db_path = Path("_smoke_trading_lab.db")
    exported_path: Path | None = None
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
        auth = ("admin", "secret")
        health = client.get("/health")
        assert health.status_code == 200, health.text
        assert health.json()["status"] == "ok", health.text

        blocked_dashboard = client.get("/dashboard")
        assert blocked_dashboard.status_code == 401, blocked_dashboard.text
        dashboard = client.get("/dashboard", auth=auth)
        assert dashboard.status_code == 200, dashboard.text[:500]
        assert "Anata AI Trader" in dashboard.text

        blocked_trade = client.post(
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
        assert blocked_trade.status_code == 401, blocked_trade.text

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
            auth=auth,
        )
        assert trade.status_code == 200, trade.text
        assert trade.json()["status"] == "FILLED", trade.text
        assert trade.json()["decision_id"] > 0, trade.text

        experiences = client.get("/api/experiences")
        assert experiences.status_code == 200, experiences.text
        assert len(experiences.json()) >= 1, experiences.text

        news_run = client.post("/api/news/run-once", auth=auth)
        assert news_run.status_code == 200, news_run.text
        assert "providers" in news_run.json(), news_run.text
        assert news_run.json()["rows_saved"] > 0, news_run.text
        rss_latest = client.get("/api/news/latest?provider=rss")
        assert rss_latest.status_code == 200, rss_latest.text
        assert rss_latest.json()[0]["provider"] == "rss", rss_latest.text
        assert "BTCUSDT" in rss_latest.json()[0]["affected_symbols"], rss_latest.text
        sentiment_latest = client.get("/api/sentiment/latest")
        assert sentiment_latest.status_code == 200, sentiment_latest.text
        assert sentiment_latest.json()[0]["model_name"], sentiment_latest.text

        mock_news = client.post(
            "/api/news/mock",
            json={
                "title": "Mock BTC macro update",
                "body": "Bitcoin and Ethereum traders watch inflation, the Fed, and liquidity risk.",
            },
            auth=auth,
        )
        assert mock_news.status_code == 200, mock_news.text
        assert mock_news.json()["rows_saved"] == 1, mock_news.text
        with SessionLocal() as session:
            news_count = session.scalar(select(func.count(NewsArticle.id))) or 0
            sentiment_count = session.scalar(select(func.count(NewsSentiment.id))) or 0
        assert news_count > 0
        assert sentiment_count > 0

        collector_status = client.get("/api/collectors/status")
        assert collector_status.status_code == 200, collector_status.text

        market_backfill = client.post(
            "/api/market/backfill",
            json={"symbols": ["BTCUSDT"], "limit": 100, "mock": True},
            auth=auth,
        )
        assert market_backfill.status_code == 200, market_backfill.text
        assert market_backfill.json()["rows_saved"] > 0, market_backfill.text

        market_status = client.get("/api/market/status")
        assert market_status.status_code == 200, market_status.text
        assert market_status.json()["candle_count"] > 0, market_status.text
        candles = client.get("/api/market/candles?symbol=BTCUSDT&timeframe=1m&limit=5")
        assert candles.status_code == 200, candles.text
        assert len(candles.json()) > 0, candles.text
        summary = client.get("/api/dashboard/summary")
        assert summary.status_code == 200, summary.text
        assert summary.json()["sentiment_model"]["active_model"], summary.text

        with SessionLocal() as session:
            decisions_before = session.scalar(select(func.count(AiDecision.id))) or 0
            experiences_before = session.scalar(select(func.count(ExperienceRecord.id))) or 0
            candles_before = session.scalar(select(func.count(Candle.id))) or 0
        assert candles_before > 0

        auto_start = client.post("/api/auto-trader/start", auth=auth)
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
        decisions = client.get("/api/ai-decisions")
        assert decisions.status_code == 200, decisions.text

        with SessionLocal() as session:
            decisions_after = session.scalar(select(func.count(AiDecision.id))) or 0
            experiences_after = session.scalar(select(func.count(ExperienceRecord.id))) or 0
            latest_feature = session.scalar(select(Feature).order_by(Feature.created_at.desc()).limit(1))
        assert decisions_after > decisions_before
        assert experiences_after > experiences_before
        assert latest_feature is not None
        assert (latest_feature.payload or {}).get("values", {}).get("candles_used", 0) > 0
        assert (latest_feature.payload or {}).get("values", {}).get("last_close") is not None

        export = client.post("/api/training/export", json={"use_all_data": True}, auth=auth)
        assert export.status_code == 200, export.text
        assert export.json()["counts"]["features"] > 0, export.text
        exported_path = Path(export.json()["exported_path"])

        auto_stop = client.post("/api/auto-trader/stop", auth=auth)
        assert auto_stop.status_code == 200, auto_stop.text
        assert auto_stop.json()["running"] is False, auto_stop.text

        tables = set(inspect(engine).get_table_names())
        missing = required_tables - tables
        assert not missing, f"Missing tables: {sorted(missing)}"

    engine.dispose()
    if db_path.exists():
        db_path.unlink()
    if SMOKE_FEED_PATH.exists():
        SMOKE_FEED_PATH.unlink()
    if exported_path and exported_path.exists():
        exported_path.unlink()
    print("smoke_ok=true")


if __name__ == "__main__":
    main()
