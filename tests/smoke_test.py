from __future__ import annotations

import csv
import gzip
import os
import shutil
import subprocess
import sys
import time
import json
import zipfile
from datetime import datetime, timedelta, timezone
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
os.environ["RAW_PAYLOAD_RETENTION_HOURS"] = "1"
os.environ["LIVE_UPDATE_RETENTION_HOURS"] = "1"
os.environ["ACCOUNT_EQUITY_RETENTION_DAYS"] = "7"
os.environ["RAW_NEWS_TEXT_RETENTION_DAYS"] = "1"
os.environ["KEEP_CLOSED_CANDLES_DAYS"] = "365"
os.environ["KEEP_TRAINING_FEATURES_DAYS"] = "365"
os.environ["KEEP_EXPERIENCE_DAYS"] = "365"
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
os.environ["ENABLE_SERVER_TRAINING"] = "false"
os.environ["ENABLE_SERVER_INFERENCE"] = "true"
os.environ["PAPER_START_BALANCE"] = "10000"
os.environ["MODEL_DIR"] = "./_smoke_models"
os.environ["DASHBOARD_USERNAME"] = "admin"
os.environ["DASHBOARD_PASSWORD"] = "secret"
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from sqlalchemy import func, inspect, select, text

from app.collectors.market_collector import BinanceMarketCollector
from app.db.models import AiDecision, Candle, ExperienceRecord, ExternalDataEvent, Feature, LiveCandleUpdate, ModelVersion, NewsArticle, NewsSentiment, TrainingFeature
from app.db.session import engine
from app.db.session import SessionLocal
from app.features.schema import CURRENT_FEATURE_SCHEMA_VERSION, columns_for_schema, numeric_vector
from app.main import app


def main() -> None:
    db_path = Path("_smoke_trading_lab.db")
    exported_path: Path | None = None
    accelerated_exported_path: Path | None = None
    uploaded_model_package: Path | None = None
    uploaded_model_file: Path | None = None
    sentiment_upload_path: Path | None = None
    archive_paths: list[Path] = []
    if db_path.exists():
        db_path.unlink()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE training_features (
                    id INTEGER PRIMARY KEY,
                    symbol VARCHAR(32),
                    schema_version VARCHAR(64),
                    as_of TIMESTAMP,
                    created_at TIMESTAMP
                )
                """
            )
        )

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
        migrated_training_feature_columns = {column["name"] for column in inspect(engine).get_columns("training_features")}
        for column_name in ("source_feature_id", "source_name", "feature_values", "payload"):
            assert column_name in migrated_training_feature_columns, sorted(migrated_training_feature_columns)
        manual_migrate = client.post("/api/db/migrate", auth=auth)
        assert manual_migrate.status_code == 200, manual_migrate.text
        assert manual_migrate.json()["status"] == "ok", manual_migrate.text
        assert "training_features" in manual_migrate.json()["existing_tables"], manual_migrate.text
        blocked_api = client.get("/api/dashboard/summary")
        assert blocked_api.status_code == 401, blocked_api.text

        blocked_dashboard = client.get("/dashboard")
        assert blocked_dashboard.status_code == 401, blocked_dashboard.text
        dashboard = client.get("/dashboard", auth=auth)
        assert dashboard.status_code == 200, dashboard.text[:500]
        assert "Anata AI Trader" in dashboard.text
        assert "modelsBody" in dashboard.text

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

        close_long = client.post(
            "/api/signal",
            json={
                "symbol": "BTCUSDT",
                "action": "CLOSE",
                "confidence": 0.95,
                "price": 50100,
                "reason": "smoke close long",
                "source": "smoke-test",
            },
            auth=auth,
        )
        assert close_long.status_code == 200, close_long.text
        assert close_long.json()["status"] == "FILLED", close_long.text

        short_trade = client.post(
            "/api/signal",
            json={
                "symbol": "BTCUSDT",
                "action": "SELL",
                "confidence": 0.90,
                "price": 50200,
                "notional": 250,
                "reason": "smoke short test",
                "source": "smoke-test",
            },
            auth=auth,
        )
        assert short_trade.status_code == 200, short_trade.text
        assert short_trade.json()["status"] == "FILLED", short_trade.text
        positions_after_short = client.get("/api/positions", auth=auth)
        assert positions_after_short.status_code == 200, positions_after_short.text
        open_short = [row for row in positions_after_short.json() if row["symbol"] == "BTCUSDT" and row["status"] == "OPEN"]
        assert open_short and open_short[0]["side"] == "SHORT", positions_after_short.text

        close_short = client.post(
            "/api/signal",
            json={
                "symbol": "BTCUSDT",
                "action": "CLOSE",
                "confidence": 0.95,
                "price": 50100,
                "reason": "smoke close short",
                "source": "smoke-test",
            },
            auth=auth,
        )
        assert close_short.status_code == 200, close_short.text
        assert close_short.json()["status"] == "FILLED", close_short.text

        experiences = client.get("/api/experiences", auth=auth)
        assert experiences.status_code == 200, experiences.text
        assert len(experiences.json()) >= 1, experiences.text

        news_run = client.post("/api/news/run-once", auth=auth)
        assert news_run.status_code == 200, news_run.text
        assert "providers" in news_run.json(), news_run.text
        assert news_run.json()["rows_saved"] > 0, news_run.text
        rss_latest = client.get("/api/news/latest?provider=rss", auth=auth)
        assert rss_latest.status_code == 200, rss_latest.text
        assert rss_latest.json()[0]["provider"] == "rss", rss_latest.text
        assert "BTCUSDT" in rss_latest.json()[0]["affected_symbols"], rss_latest.text
        sentiment_latest = client.get("/api/sentiment/latest", auth=auth)
        assert sentiment_latest.status_code == 200, sentiment_latest.text
        assert sentiment_latest.json()[0]["model_name"], sentiment_latest.text
        sentiment_model_status = client.get("/api/sentiment/model-status", auth=auth)
        assert sentiment_model_status.status_code == 200, sentiment_model_status.text
        assert "backend" in sentiment_model_status.json(), sentiment_model_status.text
        hf_env = os.environ.copy()
        hf_env["ENABLE_HF_SENTIMENT"] = "true"
        hf_env["HF_SENTIMENT_BACKEND"] = "api"
        hf_env["HF_API_TOKEN"] = ""
        hf_missing_token = subprocess.run(
            [
                sys.executable,
                "-c",
                "import json; from app.ai.news_sentiment import active_sentiment_model; print(json.dumps(active_sentiment_model()))",
            ],
            cwd=ROOT,
            env=hf_env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert hf_missing_token.returncode == 0, hf_missing_token.stderr
        missing_token_status = json.loads(hf_missing_token.stdout)
        assert missing_token_status["fallback"] is True, missing_token_status
        assert "HF_API_TOKEN" in (missing_token_status["hf_last_error"] or ""), missing_token_status
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
            first_article = session.scalar(select(NewsArticle).order_by(NewsArticle.id).limit(1))
        assert news_count > 0
        assert sentiment_count > 0
        assert first_article is not None
        raw_news_export = client.post("/api/news/export-raw", json={"use_all_data": True}, auth=auth)
        assert raw_news_export.status_code == 200, raw_news_export.text
        assert raw_news_export.json()["rows"] > 0, raw_news_export.text
        raw_news_download = client.get(raw_news_export.json()["download_url"], auth=auth)
        assert raw_news_download.status_code == 200, raw_news_download.text
        assert raw_news_download.content
        sentiment_upload_path = Path("_smoke_news_sentiment.jsonl.gz")
        with gzip.open(sentiment_upload_path, "wt", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "article_id": first_article.id,
                        "url": first_article.url,
                        "sentiment_score": 0.42,
                        "risk_score": 0.12,
                        "topics": ["macro", "market"],
                        "affected_symbols": ["BTCUSDT"],
                        "label": "positive",
                        "confidence": 0.88,
                        "model_name": "smoke-local-news-model",
                        "raw_payload": {"test": True},
                    }
                )
                + "\n"
            )
        with sentiment_upload_path.open("rb") as handle:
            sentiment_import = client.post(
                "/api/sentiment/import",
                files={"file": (sentiment_upload_path.name, handle, "application/gzip")},
                auth=auth,
            )
        assert sentiment_import.status_code == 200, sentiment_import.text
        assert sentiment_import.json()["imported"] == 1, sentiment_import.text
        with SessionLocal() as session:
            imported_sentiment = session.scalar(select(NewsSentiment).where(NewsSentiment.article_id == first_article.id).limit(1))
            assert imported_sentiment is not None
            assert imported_sentiment.model_name == "smoke-local-news-model"

        collector_status = client.get("/api/collectors/status", auth=auth)
        assert collector_status.status_code == 200, collector_status.text
        assert "derivatives" in collector_status.json(), collector_status.text

        derivatives_run = client.post(
            "/api/derivatives/run-once",
            json={"symbols": ["BTCUSDT"], "mock": True},
            auth=auth,
        )
        assert derivatives_run.status_code == 200, derivatives_run.text
        assert derivatives_run.json()["rows_saved"] >= 7, derivatives_run.text
        derivatives_status = client.get("/api/derivatives/status", auth=auth)
        assert derivatives_status.status_code == 200, derivatives_status.text
        assert derivatives_status.json()["counts_by_type"]["funding_rate"] >= 1, derivatives_status.text
        derivatives_latest = client.get("/api/derivatives/latest?symbol=BTCUSDT", auth=auth)
        assert derivatives_latest.status_code == 200, derivatives_latest.text
        assert len(derivatives_latest.json()) >= 1, derivatives_latest.text
        external_run = client.post("/api/external/run-once", json={"mock": True}, auth=auth)
        assert external_run.status_code == 200, external_run.text
        assert external_run.json()["rows_saved"] >= 17, external_run.text
        liquidation_run = client.post("/api/liquidations/run-once", json={"mock": True}, auth=auth)
        assert liquidation_run.status_code == 200, liquidation_run.text
        assert liquidation_run.json()["rows_saved"] >= 10, liquidation_run.text
        external_status = client.get("/api/external/status", auth=auth)
        assert external_status.status_code == 200, external_status.text
        assert external_status.json()["collectors"]["fear_greed"]["counts_by_type"]["fear_greed_value"] >= 1, external_status.text

        accelerated = client.post(
            "/api/training/build-dataset",
            json={
                "symbols": ["BTCUSDT"],
                "interval": "1m",
                "days": 1,
                "max_rows_per_symbol": 400,
                "lookback": 60,
                "stride": 20,
                "replay_limit": 1000,
                "backfill": True,
                "mock": True,
                "export": True,
            },
            auth=auth,
        )
        assert accelerated.status_code == 200, accelerated.text
        accelerated_payload = accelerated.json()
        assert accelerated_payload["labels"]["rows_created"] > 0, accelerated.text
        assert accelerated_payload["replay"]["experiences_created"] > 0, accelerated.text
        accelerated_exported_path = Path(accelerated_payload["exported_path"])
        with SessionLocal() as session:
            candles_for_label = list(
                session.scalars(
                    select(Candle)
                    .where(Candle.symbol == "BTCUSDT", Candle.interval == "1m", Candle.is_closed.is_(True))
                    .order_by(Candle.open_time)
                )
            )
            assert len(candles_for_label) > 260
            entry_candle = candles_for_label[80]
            values = {
                "price_change": 0.0,
                "volume_change": 0.0,
                "volatility": 0.0,
                "sentiment_score": 0.0,
                "risk_score": 0.0,
                "last_close": entry_candle.close,
            }
            payload = {
                "schema_version": CURRENT_FEATURE_SCHEMA_VERSION,
                "values": values,
                "metadata": {"interval": "1m", "test": "unlabeled_feature_for_label_builder"},
                "sources": {"candles": "smoke"},
            }
            feature = Feature(
                symbol="BTCUSDT",
                schema_version=CURRENT_FEATURE_SCHEMA_VERSION,
                source_name="smoke_unlabeled",
                as_of=entry_candle.close_time or entry_candle.open_time,
                payload=payload,
                raw_payload=payload,
            )
            session.add(feature)
            session.flush()
            training_feature = TrainingFeature(
                source_feature_id=feature.id,
                symbol="BTCUSDT",
                schema_version=CURRENT_FEATURE_SCHEMA_VERSION,
                source_name="smoke_unlabeled",
                as_of=feature.as_of,
                feature_values=values,
                payload=payload,
            )
            session.add(training_feature)
            session.commit()
            smoke_training_feature_id = training_feature.id
        label_status_before = client.get("/api/training/label-status", auth=auth)
        assert label_status_before.status_code == 200, label_status_before.text
        build_labels = client.post("/api/training/build-labels", json={"symbols": ["BTCUSDT"]}, auth=auth)
        assert build_labels.status_code == 200, build_labels.text
        assert build_labels.json()["labeled"] > 0, build_labels.text
        assert build_labels.json()["label_status"]["rows_with_target_trade_quality_score"] > 0, build_labels.text
        with SessionLocal() as session:
            labeled_row = session.get(TrainingFeature, smoke_training_feature_id)
            assert labeled_row is not None
            labeled_values = dict(labeled_row.feature_values or {})
            assert labeled_values.get("target_trade_quality_score") not in (None, "")

        train_model = client.post(
            "/api/training/train-model",
            json={"dataset_path": accelerated_payload["exported_path"], "use_all_data": True, "wait": True},
            auth=auth,
        )
        assert train_model.status_code == 200, train_model.text
        assert train_model.json()["status"] == "disabled", train_model.text
        assert "Download dataset and train locally" in train_model.json()["message"], train_model.text

        feature_columns = columns_for_schema(CURRENT_FEATURE_SCHEMA_VERSION)
        uploaded_model_file = Path("_smoke_mock_model.json")
        uploaded_model_package = Path("_smoke_model_package.zip")
        uploaded_model_file.write_text(
            json.dumps(
                {
                    "intercept": 0.01,
                    "coefficients": [0.0 for _ in feature_columns],
                    "feature_columns": feature_columns,
                    "feature_schema_version": CURRENT_FEATURE_SCHEMA_VERSION,
                }
            ),
            encoding="utf-8",
        )
        metadata = {
            "model_id": "smoke-linear:1",
            "name": "smoke-linear",
            "version": "smoke-1",
            "model_type": "linear_json",
            "model_file": uploaded_model_file.name,
            "feature_schema_version": CURRENT_FEATURE_SCHEMA_VERSION,
            "feature_columns": feature_columns,
            "metrics": {"directional_accuracy": 0.60, "net_return_after_fees": 0.01},
            "training_dataset_hash": "smoke",
        }
        with zipfile.ZipFile(uploaded_model_package, "w") as package:
            package.write(uploaded_model_file, uploaded_model_file.name)
            package.writestr("metadata.json", json.dumps(metadata))
        with uploaded_model_package.open("rb") as handle:
            upload_model = client.post("/api/models/upload", files={"file": (uploaded_model_package.name, handle, "application/zip")}, auth=auth)
        assert upload_model.status_code == 200, upload_model.text
        assert upload_model.json()["model"]["status"] == "candidate", upload_model.text
        models = client.get("/api/models", auth=auth)
        assert models.status_code == 200, models.text
        assert any(row["status"] == "candidate" for row in models.json()), models.text
        activate = client.post("/api/models/activate", json={"model_id": "smoke-linear:1"}, auth=auth)
        assert activate.status_code == 200, activate.text
        assert activate.json()["model"]["status"] == "active", activate.text
        active_model = client.get("/api/models/active", auth=auth)
        assert active_model.status_code == 200, active_model.text
        assert active_model.json()["status"] == "active", active_model.text
        latest_model = client.get("/api/models/latest", auth=auth)
        assert latest_model.status_code == 200, latest_model.text
        assert latest_model.json()["status"] == "active", latest_model.text
        model_prediction = client.get("/api/models/predict?symbol=BTCUSDT", auth=auth)
        assert model_prediction.status_code == 200, model_prediction.text
        assert model_prediction.json()["status"] == "ok", model_prediction.text
        assert "predicted_return" in model_prediction.json()["prediction"], model_prediction.text

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

        market_status = client.get("/api/market/status", auth=auth)
        assert market_status.status_code == 200, market_status.text
        assert market_status.json()["candle_count"] > 0, market_status.text
        db_diagnostics = client.get("/api/db/diagnostics", auth=auth)
        assert db_diagnostics.status_code == 200, db_diagnostics.text
        assert db_diagnostics.json()["candles"]["closed_training_rows"] > 0, db_diagnostics.text
        assert db_diagnostics.json()["candles"]["live_update_rows"] == 1, db_diagnostics.text
        assert db_diagnostics.json()["candles"]["duplicate_group_count"] == 0, db_diagnostics.text
        assert "top_largest_tables" in db_diagnostics.json(), db_diagnostics.text
        db_storage = client.get("/api/db/storage", auth=auth)
        assert db_storage.status_code == 200, db_storage.text
        assert "rows_by_table" in db_storage.json(), db_storage.text
        collection_report = client.get("/api/data/collection-report?include_storage=true", auth=auth)
        assert collection_report.status_code == 200, collection_report.text
        assert "improve_next" in collection_report.json(), collection_report.text
        assert collection_report.json()["counts"]["candles"] > 0, collection_report.text
        storage_status = client.get("/api/storage/status", auth=auth)
        assert storage_status.status_code == 200, storage_status.text
        assert "top_largest_tables" in storage_status.json(), storage_status.text
        storage_compact = client.post("/api/storage/compact", auth=auth)
        assert storage_compact.status_code == 200, storage_compact.text
        candles = client.get("/api/market/candles?symbol=BTCUSDT&timeframe=1m&limit=5", auth=auth)
        assert candles.status_code == 200, candles.text
        assert len(candles.json()) > 0, candles.text
        candles_5m = client.get("/api/market/candles?symbol=BTCUSDT&timeframe=5m&limit=5", auth=auth)
        assert candles_5m.status_code == 200, candles_5m.text
        assert len(candles_5m.json()) > 0, candles_5m.text
        assert candles_5m.json()[-1]["source_name"] == "aggregated_from_1m", candles_5m.text
        candles_1s = client.get("/api/market/candles?symbol=BTCUSDT&timeframe=1s&limit=5", auth=auth)
        assert candles_1s.status_code == 200, candles_1s.text
        assert len(candles_1s.json()) > 0, candles_1s.text
        assert candles_1s.json()[-1]["source_name"] in {"1m_live_fallback", "binance_kline_live"}, candles_1s.text
        summary = client.get("/api/dashboard/summary", auth=auth)
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
            auto_status = client.get("/api/auto-trader/status", auth=auth).json()
            if auto_status.get("cycles", 0) >= 1:
                break
            time.sleep(0.25)
        assert auto_status.get("cycles", 0) >= 1, auto_status
        assert auto_status.get("last_run_at"), auto_status
        assert auto_status.get("exploration_enabled") is True, auto_status
        assert auto_status.get("position_management", {}).get("min_hold_seconds", 0) > 0, auto_status
        decisions = client.get("/api/ai-decisions", auth=auth)
        assert decisions.status_code == 200, decisions.text
        assert decisions.json()[0]["decision_source"] in {"exploration", "model", "position-management", "risk-exit", "strategy"}, decisions.text
        summary_after_auto = client.get("/api/dashboard/summary", auth=auth)
        assert summary_after_auto.status_code == 200, summary_after_auto.text
        trading_counts = summary_after_auto.json()["trading"]
        assert (
            trading_counts["strategy_trades"]
            + trading_counts["exploration_trades"]
            + trading_counts["skipped_trades"]
            >= 1
        ), summary_after_auto.text

        with SessionLocal() as session:
            decisions_after = session.scalar(select(func.count(AiDecision.id))) or 0
            experiences_after = session.scalar(select(func.count(ExperienceRecord.id))) or 0
            latest_feature = session.scalar(select(Feature).order_by(Feature.created_at.desc()).limit(1))
        assert decisions_after > decisions_before
        assert experiences_after > experiences_before
        assert latest_feature is not None
        assert (latest_feature.payload or {}).get("values", {}).get("candles_used", 0) > 0
        assert (latest_feature.payload or {}).get("values", {}).get("last_close") is not None
        feature_latest = client.get("/api/features/latest?symbol=BTCUSDT", auth=auth)
        assert feature_latest.status_code == 200, feature_latest.text
        feature_payload = feature_latest.json()
        assert feature_payload["symbol"] == "BTCUSDT", feature_latest.text
        assert feature_payload["schema_version"] == "price-news-market-v4", feature_latest.text
        assert "sentiment_confidence" in feature_payload["vector"], feature_latest.text
        assert "candle_return_1m" in feature_payload["vector"], feature_latest.text
        assert "trader_crowd_score" in feature_payload["vector"], feature_latest.text
        assert feature_payload["vector"]["derivatives_recency_weight"] > 0, feature_latest.text
        assert feature_payload["vector"]["fear_greed_value"] > 0, feature_latest.text
        assert "fear_greed_change_24h" in feature_payload["vector"], feature_latest.text
        assert "market_cap_change_24h" in feature_payload["vector"], feature_latest.text
        assert "liquidation_long_usd_1m" in feature_payload["vector"], feature_latest.text
        assert "usdt_deviation" in feature_payload["vector"], feature_latest.text
        assert "stablecoin_supply_change_24h" in feature_payload["vector"], feature_latest.text
        assert "fed_risk_score" in feature_payload["vector"], feature_latest.text
        assert "market_regime_score" in feature_payload["vector"], feature_latest.text
        assert "source_freshness" in feature_payload, feature_latest.text
        assert len(feature_payload["derivatives_context"]) >= 1, feature_latest.text
        assert len(feature_payload["external_context"]) >= 1, feature_latest.text
        assert "final_ai_input" in feature_payload, feature_latest.text
        assert "strategy_input" in feature_payload["final_ai_input"], feature_latest.text
        with SessionLocal() as session:
            latest_feature_row = session.get(Feature, feature_payload["id"])
            assert latest_feature_row is not None
            v3_columns = columns_for_schema("price-news-v3")
            assert "market_regime_score" not in v3_columns
            assert len(numeric_vector(latest_feature_row, v3_columns)) == len(v3_columns)
            training_feature_count = session.scalar(select(func.count(TrainingFeature.id))) or 0
            external_event_count = session.scalar(select(func.count(ExternalDataEvent.id))) or 0
        assert training_feature_count > 0
        assert external_event_count >= 7

        auto_stop_before_compact = client.post("/api/auto-trader/stop", auth=auth)
        assert auto_stop_before_compact.status_code == 200, auto_stop_before_compact.text

        with SessionLocal() as session:
            compact_candle = session.scalar(select(Candle).where(Candle.is_closed.is_(True)).order_by(Candle.open_time).limit(1))
            assert compact_candle is not None
            compact_candle.raw_payload = {"large": "x" * 4096}
            compact_candle.raw = {"large": "x" * 4096}
            compact_candle.updated_at = datetime.now(timezone.utc) - timedelta(hours=2)
            closed_candles_before_compact = session.scalar(select(func.count(Candle.id)).where(Candle.is_closed.is_(True))) or 0
            training_features_before_compact = session.scalar(select(func.count(TrainingFeature.id))) or 0
            model_versions_before_compact = session.scalar(select(func.count(ModelVersion.id))) or 0
            compact_candle_id = compact_candle.id
            session.commit()
        compact = client.post("/api/db/compact", json={}, auth=auth)
        assert compact.status_code == 200, compact.text
        compact_payload = compact.json()["last_cleanup"]
        assert compact_payload["compacted"]["candles_raw_fields"] >= 1, compact.text
        with SessionLocal() as session:
            closed_candles_after_compact = session.scalar(select(func.count(Candle.id)).where(Candle.is_closed.is_(True))) or 0
            training_features_after_compact = session.scalar(select(func.count(TrainingFeature.id))) or 0
            model_versions_after_compact = session.scalar(select(func.count(ModelVersion.id))) or 0
            compacted_candle = session.get(Candle, compact_candle_id)
            assert compacted_candle is not None
            assert compacted_candle.raw_payload is None
            assert compacted_candle.raw is None
        assert closed_candles_after_compact == closed_candles_before_compact
        assert training_features_after_compact == training_features_before_compact
        assert model_versions_after_compact == model_versions_before_compact
        factory_compact = client.post("/api/db/compact", json={"factory_mode": True, "keep_recent_days": 365}, auth=auth)
        assert factory_compact.status_code == 200, factory_compact.text
        assert factory_compact.json()["last_cleanup"]["factory_mode"] is True, factory_compact.text

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
        assert export.json()["dataset_id"].endswith(".csv.gz"), export.text
        exported_path = Path(export.json()["exported_path"])
        download = client.get(export.json()["download_url"], auth=auth)
        assert download.status_code == 200, download.text
        assert len(download.content) > 0, export.text
        with gzip.open(exported_path, "rt", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            assert "source_freshness" in (reader.fieldnames or []), export.text
            assert "fear_greed_change_24h" in (reader.fieldnames or []), export.text
            labeled_export_rows = [
                row for row in reader if row.get("target_trade_quality_score") not in (None, "")
            ]
        assert len(labeled_export_rows) > 0, export.text
        dry_run = subprocess.run(
            [
                sys.executable,
                "scripts/train_local_model.py",
                "--dataset",
                str(exported_path),
                "--target",
                "target_trade_quality_score",
                "--dry-run",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert dry_run.returncode == 0, dry_run.stderr or dry_run.stdout
        assert '"labeled_rows": 0' not in dry_run.stdout, dry_run.stdout

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
    if accelerated_exported_path and accelerated_exported_path.exists():
        accelerated_exported_path.unlink()
    if uploaded_model_file and uploaded_model_file.exists():
        uploaded_model_file.unlink()
    if uploaded_model_package and uploaded_model_package.exists():
        uploaded_model_package.unlink()
    if sentiment_upload_path and sentiment_upload_path.exists():
        sentiment_upload_path.unlink()
    smoke_model_dir = Path("_smoke_models")
    if smoke_model_dir.exists():
        shutil.rmtree(smoke_model_dir)
    for archive_path in archive_paths:
        if archive_path.exists():
            archive_path.unlink()
    archive_dir = Path("_smoke_archives")
    if archive_dir.exists():
        archive_dir.rmdir()
    print("smoke_ok=true")


if __name__ == "__main__":
    main()
