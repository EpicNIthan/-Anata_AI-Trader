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
os.environ["EXPLORATION_MODE"] = "true"
os.environ["EXPLORATION_RATE"] = "1.0"
os.environ["MIN_PAPER_TRADE_NOTIONAL"] = "50"
os.environ["ARCHIVE_DIR"] = "./_smoke_archives"
os.environ["DERIVATIVES_ENABLED"] = "true"
os.environ["ENABLE_DERIVATIVES_COLLECTOR"] = "false"
os.environ["DERIVATIVES_SYMBOLS"] = "BTCUSDT"
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

from app.collectors.market_collector import BinanceMarketCollector
from app.db.models import AiDecision, Candle, ExperienceRecord, ExternalDataEvent, Feature, LiveCandleUpdate, NewsArticle, NewsSentiment, TrainingFeature
from app.db.session import engine
from app.db.session import SessionLocal
from app.main import app


def main() -> None:
    db_path = Path("_smoke_trading_lab.db")
    exported_path: Path | None = None
    archive_paths: list[Path] = []
    if db_path.exists():
        db_path.unlink()

    required_tables = {
        "candles",
        "live_candle_updates",
        "news_articles",
        "news_sentiment",
        "features",
        "paper_trades",
        "positions",
        "model_versions",
        "training_runs",
        "training_features",
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
        sentiment_reprocess = client.post("/api/sentiment/reprocess", json={"limit": 5, "reset_model": True}, auth=auth)
        assert sentiment_reprocess.status_code == 200, sentiment_reprocess.text
        assert sentiment_reprocess.json()["processed"] >= 1, sentiment_reprocess.text
        assert "active_model" in sentiment_reprocess.json(), sentiment_reprocess.text

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
        assert "derivatives" in collector_status.json(), collector_status.text

        derivatives_run = client.post(
            "/api/derivatives/run-once",
            json={"symbols": ["BTCUSDT"], "mock": True},
            auth=auth,
        )
        assert derivatives_run.status_code == 200, derivatives_run.text
        assert derivatives_run.json()["rows_saved"] >= 7, derivatives_run.text
        derivatives_status = client.get("/api/derivatives/status")
        assert derivatives_status.status_code == 200, derivatives_status.text
        assert derivatives_status.json()["counts_by_type"]["funding_rate"] >= 1, derivatives_status.text
        derivatives_latest = client.get("/api/derivatives/latest?symbol=BTCUSDT")
        assert derivatives_latest.status_code == 200, derivatives_latest.text
        assert len(derivatives_latest.json()) >= 1, derivatives_latest.text

        market_backfill = client.post(
            "/api/market/backfill",
            json={"symbols": ["BTCUSDT"], "limit": 100, "mock": True},
            auth=auth,
        )
        assert market_backfill.status_code == 200, market_backfill.text
        assert market_backfill.json()["rows_saved"] > 0, market_backfill.text
        now_ms = int(time.time() // 60 * 60 * 1000)
        live_payload = {
            "data": {
                "E": now_ms + 10_000,
                "k": {
                    "s": "BTCUSDT",
                    "i": "1m",
                    "t": now_ms,
                    "T": now_ms + 59_999,
                    "o": "65000.0",
                    "h": "65100.0",
                    "l": "64900.0",
                    "c": "65050.0",
                    "v": "12.0",
                    "q": "780600.0",
                    "n": 42,
                    "x": False,
                },
            }
        }
        collector = BinanceMarketCollector(symbols=["BTCUSDT"], interval="1m")
        first_live = collector.store_message(live_payload)
        live_payload["data"]["k"]["c"] = "65075.0"
        second_live = collector.store_message(live_payload)
        assert first_live["live_update_created"] is True, first_live
        assert second_live["live_update_updated"] is True, second_live
        with SessionLocal() as session:
            live_rows = session.scalars(select(LiveCandleUpdate).where(LiveCandleUpdate.symbol == "BTCUSDT")).all()
            assert len(live_rows) == 1
            assert live_rows[0].update_count == 2

        market_status = client.get("/api/market/status")
        assert market_status.status_code == 200, market_status.text
        assert market_status.json()["candle_count"] > 0, market_status.text
        db_diagnostics = client.get("/api/db/diagnostics")
        assert db_diagnostics.status_code == 200, db_diagnostics.text
        assert db_diagnostics.json()["candles"]["closed_training_rows"] > 0, db_diagnostics.text
        assert db_diagnostics.json()["candles"]["live_update_rows"] == 1, db_diagnostics.text
        assert db_diagnostics.json()["candles"]["duplicate_group_count"] == 0, db_diagnostics.text
        candles = client.get("/api/market/candles?symbol=BTCUSDT&timeframe=1m&limit=5")
        assert candles.status_code == 200, candles.text
        assert len(candles.json()) > 0, candles.text
        candles_5m = client.get("/api/market/candles?symbol=BTCUSDT&timeframe=5m&limit=5")
        assert candles_5m.status_code == 200, candles_5m.text
        assert len(candles_5m.json()) > 0, candles_5m.text
        assert candles_5m.json()[-1]["source_name"] == "aggregated_from_1m", candles_5m.text
        candles_1s = client.get("/api/market/candles?symbol=BTCUSDT&timeframe=1s&limit=5")
        assert candles_1s.status_code == 200, candles_1s.text
        assert len(candles_1s.json()) > 0, candles_1s.text
        assert candles_1s.json()[-1]["source_name"] in {"1m_live_fallback", "binance_kline_live"}, candles_1s.text
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
        assert auto_status.get("exploration_enabled") is True, auto_status
        decisions = client.get("/api/ai-decisions")
        assert decisions.status_code == 200, decisions.text
        assert decisions.json()[0]["decision_source"] == "exploration", decisions.text
        summary_after_auto = client.get("/api/dashboard/summary")
        assert summary_after_auto.status_code == 200, summary_after_auto.text
        assert summary_after_auto.json()["trading"]["exploration_trades"] + summary_after_auto.json()["trading"]["skipped_trades"] >= 1

        with SessionLocal() as session:
            decisions_after = session.scalar(select(func.count(AiDecision.id))) or 0
            experiences_after = session.scalar(select(func.count(ExperienceRecord.id))) or 0
            latest_feature = session.scalar(select(Feature).order_by(Feature.created_at.desc()).limit(1))
        assert decisions_after > decisions_before
        assert experiences_after > experiences_before
        assert latest_feature is not None
        assert (latest_feature.payload or {}).get("values", {}).get("candles_used", 0) > 0
        assert (latest_feature.payload or {}).get("values", {}).get("last_close") is not None
        feature_latest = client.get("/api/features/latest?symbol=BTCUSDT")
        assert feature_latest.status_code == 200, feature_latest.text
        feature_payload = feature_latest.json()
        assert feature_payload["symbol"] == "BTCUSDT", feature_latest.text
        assert feature_payload["schema_version"] == "price-news-v3", feature_latest.text
        assert "sentiment_confidence" in feature_payload["vector"], feature_latest.text
        assert "candle_return_1m" in feature_payload["vector"], feature_latest.text
        assert "trader_crowd_score" in feature_payload["vector"], feature_latest.text
        assert feature_payload["vector"]["derivatives_recency_weight"] > 0, feature_latest.text
        assert len(feature_payload["derivatives_context"]) >= 1, feature_latest.text
        assert "final_ai_input" in feature_payload, feature_latest.text
        assert "strategy_input" in feature_payload["final_ai_input"], feature_latest.text
        with SessionLocal() as session:
            training_feature_count = session.scalar(select(func.count(TrainingFeature.id))) or 0
            external_event_count = session.scalar(select(func.count(ExternalDataEvent.id))) or 0
        assert training_feature_count > 0
        assert external_event_count >= 7

        cleanup = client.post("/api/db/cleanup", auth=auth)
        assert cleanup.status_code == 200, cleanup.text
        assert "retention" in cleanup.json(), cleanup.text

        archive = client.post(
            "/api/db/archive",
            json={"before_date": "2999-01-01T00:00:00+00:00", "tables": ["candles", "training_features", "experience_buffer"]},
            auth=auth,
        )
        assert archive.status_code == 200, archive.text
        assert len(archive.json()["exports"]) >= 1, archive.text
        archive_paths = [Path(item["path"]) for item in archive.json()["exports"]]

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
    for archive_path in archive_paths:
        if archive_path.exists():
            archive_path.unlink()
    archive_dir = Path("_smoke_archives")
    if archive_dir.exists():
        archive_dir.rmdir()
    print("smoke_ok=true")


if __name__ == "__main__":
    main()
