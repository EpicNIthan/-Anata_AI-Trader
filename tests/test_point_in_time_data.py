"""Focused point-in-time and retention regressions for the Anata V2 data path."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.collectors.news_collector import NewsCollector, NormalizedArticle
from app.config import settings
from app.db.models import (
    Base,
    Feature,
    ModelPredictionRecord,
    NewsArticle,
    NewsSentiment,
    StructuredNewsEvent,
)
from app.features.feature_builder import FeatureBuilder
from app.pipeline.data_quality import PointInTimeValidator
from app.services.data_lifecycle import compact_database


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


class PointInTimeDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory(prefix="anata-pit-tests-")
        database_path = Path(self._temporary_directory.name) / "point-in-time.sqlite3"
        self.engine = create_engine(
            f"sqlite:///{database_path.as_posix()}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self.engine)
        self.session = Session(self.engine)

    def tearDown(self) -> None:
        self.session.close()
        self.engine.dispose()
        self._temporary_directory.cleanup()

    def test_candle_validation_detects_true_input_order_regression(self) -> None:
        now = datetime.now(timezone.utc).replace(microsecond=0)
        rows = [
            {"open_time": now - timedelta(minutes=3), "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
            {"open_time": now - timedelta(minutes=1), "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
            {"open_time": now - timedelta(minutes=2), "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
        ]

        report = PointInTimeValidator().validate_candles(rows, interval="1m", now=now)

        issues = [issue for issue in report.issues if issue.code == "NON_MONOTONIC_TIMESTAMP"]
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].context["index"], 2)
        self.assertNotIn(
            "NON_MONOTONIC_TIMESTAMP",
            {
                issue.code
                for issue in PointInTimeValidator().validate_candles(
                    sorted(rows, key=lambda row: row["open_time"]), interval="1m", now=now
                ).issues
            },
        )

    def test_feature_snapshot_rejects_data_unavailable_at_decision_time(self) -> None:
        decision_time = datetime.now(timezone.utc).replace(microsecond=0)
        feature = Feature(
            symbol="BTCUSDT",
            schema_version="price-news-v4",
            as_of=decision_time,
            available_to_model_time=decision_time + timedelta(seconds=1),
            payload={"values": {"price_change": 0.01}},
        )
        self.session.add(feature)
        self.session.flush()

        with self.assertRaisesRegex(ValueError, "not available to the model"):
            PointInTimeValidator().snapshot_from_feature(feature, decision_time=decision_time)

    def test_structured_external_context_ignores_future_and_unrelated_events(self) -> None:
        now = datetime.now(timezone.utc).replace(microsecond=0)
        past = StructuredNewsEvent(
            primary_symbol="BTCUSDT",
            event_type="market_move",
            affected_assets=["BTC"],
            direction="bullish",
            sentiment=0.4,
            severity=0.5,
            importance=0.8,
            confidence=0.75,
            provider="external-test",
            prompt_version="pit-v1",
            validation_status="VALID",
            available_to_model_time=now - timedelta(minutes=5),
            payload={"external_ai_available": True, "external_ai_missing": False},
        )
        future = StructuredNewsEvent(
            primary_symbol="BTCUSDT",
            event_type="market_move",
            affected_assets=["BTC"],
            direction="bearish",
            importance=1.0,
            confidence=1.0,
            provider="future-provider",
            validation_status="VALID",
            available_to_model_time=now + timedelta(minutes=1),
            payload={"external_ai_available": True},
        )
        unrelated = StructuredNewsEvent(
            primary_symbol="ETHUSDT",
            event_type="market_move",
            affected_assets=["ETH"],
            direction="bearish",
            importance=1.0,
            confidence=1.0,
            provider="unrelated-provider",
            validation_status="VALID",
            available_to_model_time=now - timedelta(minutes=1),
            payload={"external_ai_available": True},
        )
        self.session.add_all([past, future, unrelated])
        self.session.commit()

        result = FeatureBuilder(self.session)._recent_external_ai_features("BTCUSDT", now)

        self.assertTrue(result["external_ai_available"])
        self.assertEqual(result["external_ai_provider"], "external-test")
        self.assertGreater(result["external_ai_direction_score"], 0.0)
        context_ids = {item["id"] for item in result["external_ai_context"]}
        self.assertEqual(context_ids, {past.id})

    def test_news_collection_records_event_receipt_processing_and_availability(self) -> None:
        published_at = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(hours=1)
        article = NormalizedArticle(
            provider="test-provider",
            source="test-source",
            title="Bitcoin liquidity improves",
            url="https://example.test/pit-news",
            published_at=published_at,
            raw_text="Bitcoin liquidity improves with lower market risk.",
            raw_payload={"fixture": True},
            affected_symbols=["BTCUSDT"],
        )
        isolated_sessions = sessionmaker(bind=self.engine, expire_on_commit=False)

        with patch("app.collectors.news_collector.SessionLocal", isolated_sessions):
            self.assertEqual(NewsCollector().store_articles([article]), 1)

        stored = self.session.scalar(select(NewsArticle).where(NewsArticle.url == article.url))
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertEqual(_utc(stored.event_time), published_at)
        self.assertIsNotNone(stored.received_time)
        self.assertIsNotNone(stored.processed_time)
        self.assertIsNotNone(stored.available_to_model_time)
        self.assertLessEqual(_utc(stored.received_time), _utc(stored.processed_time))
        self.assertEqual(_utc(stored.processed_time), _utc(stored.available_to_model_time))

    def test_news_features_use_availability_instead_of_publication_time(self) -> None:
        now = datetime.now(timezone.utc).replace(microsecond=0)
        article = NewsArticle(
            source="delayed-source",
            source_name="test",
            title="Bitcoin article received later",
            url="https://example.test/delayed",
            published_at=now - timedelta(hours=1),
            event_time=now - timedelta(hours=1),
            received_time=now + timedelta(minutes=1),
            processed_time=now + timedelta(minutes=1),
            available_to_model_time=now + timedelta(minutes=1),
            raw_text="Bitcoin news",
        )
        self.session.add(article)
        self.session.flush()
        self.session.add(
            NewsSentiment(
                article_id=article.id,
                sentiment_score=1.0,
                risk_score=0.0,
                affected_symbols=["BTCUSDT"],
                confidence=1.0,
            )
        )
        self.session.commit()

        result = FeatureBuilder(self.session)._recent_news_features("BTCUSDT", now)

        self.assertEqual(result["sentiment_articles_used"], 0)
        self.assertEqual(result["sentiment_score"], 0.0)

    def test_feature_builder_runs_without_external_context_rows(self) -> None:
        feature = FeatureBuilder(self.session).build_for_symbol("BTCUSDT", store=False)

        values = (feature.payload or {}).get("values", {})
        self.assertFalse(values["external_ai_available"])
        self.assertTrue(values["external_ai_missing"])
        self.assertEqual(values["external_ai_direction_score"], 0.0)

    def test_lifecycle_detaches_v2_lineage_before_legacy_row_deletion(self) -> None:
        now = datetime.now(timezone.utc).replace(microsecond=0)
        old = now - timedelta(days=60)
        feature = Feature(
            symbol="BTCUSDT",
            schema_version="price-news-v4",
            as_of=old,
            available_to_model_time=old,
            payload={"values": {}},
            created_at=old,
        )
        article = NewsArticle(
            source="old-source",
            source_name="test",
            title="Old article",
            url="https://example.test/old",
            published_at=old,
            event_time=old,
            received_time=old,
            processed_time=old,
            available_to_model_time=old,
            created_at=old,
        )
        self.session.add_all([feature, article])
        self.session.flush()
        prediction = ModelPredictionRecord(
            prediction_id="pred_old_feature",
            decision_trace_id="trace_old_feature",
            model_id="test-model",
            model_version="v1",
            model_family="test",
            symbol="BTCUSDT",
            generated_at=old,
            valid_from=old,
            expires_at=old + timedelta(minutes=5),
            forecast_horizon_seconds=300,
            expected_return=0.0,
            expected_volatility=0.0,
            probability_up=0.5,
            probability_down=0.5,
            confidence=0.5,
            calibration_score=0.5,
            uncertainty=0.5,
            feature_schema_version=feature.schema_version,
            feature_snapshot_id=f"feature_db_{feature.id}",
            feature_id=feature.id,
        )
        event = StructuredNewsEvent(
            article_id=article.id,
            primary_symbol="BTCUSDT",
            event_type="other",
            direction="neutral",
            validation_status="VALID",
            available_to_model_time=old,
        )
        self.session.add_all([prediction, event])
        self.session.commit()
        feature_id, article_id = feature.id, article.id
        prediction_id, event_id = prediction.id, event.id

        lifecycle_settings = replace(
            settings,
            railway_data_factory_mode=True,
            operational_retention_days=2,
            raw_news_retention_days=30,
        )
        with patch("app.services.data_lifecycle.settings", lifecycle_settings):
            result = compact_database(self.session)

        self.session.expire_all()
        self.assertIsNone(self.session.get(Feature, feature_id))
        self.assertIsNone(self.session.get(NewsArticle, article_id))
        self.assertIsNone(self.session.get(ModelPredictionRecord, prediction_id).feature_id)
        self.assertIsNone(self.session.get(StructuredNewsEvent, event_id).article_id)
        self.assertEqual(result["compacted"]["model_prediction_feature_links_detached"], 1)
        self.assertEqual(result["compacted"]["structured_news_article_links_detached"], 1)


if __name__ == "__main__":
    unittest.main()
