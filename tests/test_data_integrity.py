from __future__ import annotations

import csv
import gzip
import json
import tempfile
import unittest
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from app.pipeline.data_quality import PointInTimeValidator
from app.services.raw_data_cleanup import cleanup_downloaded_raw_data
from scripts.sync_verification import refresh_export_manifests, verify_export_archive


UTC = timezone.utc


class DataQualityCompletenessTests(unittest.TestCase):
    def test_candle_quality_is_series_scoped_and_detects_revision_gap_stale_and_outlier(self) -> None:
        now = datetime(2026, 1, 1, 0, 20, tzinfo=UTC)
        rows = [
            {
                "symbol": "BTCUSDT",
                "interval": "1m",
                "open_time": datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
                "open": 100,
                "high": 101,
                "low": 99,
                "close": 100,
                "volume": 5,
            },
            {
                # A second symbol at the same timestamp is not a duplicate.
                "symbol": "ETHUSDT",
                "interval": "1m",
                "open_time": datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
                "open": 50,
                "high": 51,
                "low": 49,
                "close": 50,
                "volume": 2,
            },
            {
                "symbol": "BTCUSDT",
                "interval": "1m",
                "open_time": datetime(2026, 1, 1, 0, 2, tzinfo=UTC),
                "open": 100,
                "high": 201,
                "low": 99,
                "close": 200,
                "volume": 7,
            },
            {
                # Same identity but changed values proves a source revision.
                "symbol": "BTCUSDT",
                "interval": "1m",
                "open_time": datetime(2026, 1, 1, 0, 2, tzinfo=UTC),
                "open": 100,
                "high": 202,
                "low": 99,
                "close": 201,
                "volume": 7,
            },
        ]
        report = PointInTimeValidator().validate_candles(
            rows,
            interval="1m",
            now=now,
            stale_after_seconds=60,
            outlier_return_threshold=0.5,
        )
        codes = [issue.code for issue in report.issues]
        self.assertEqual(codes.count("DUPLICATE_CANDLE"), 1)
        self.assertIn("REVISED_VALUE", codes)
        self.assertIn("MISSING_CANDLE_INTERVAL", codes)
        self.assertIn("PRICE_OUTLIER", codes)
        self.assertIn("STALE_FEED", codes)

    def test_feature_and_bundle_quality_cover_schema_provider_outlier_and_completeness(self) -> None:
        validator = PointInTimeValidator()
        feature_report = validator.validate_feature_payload(
            {"known": 20.0, "unexpected": 1.0, "external_ai_provider": None},
            required_features=("known", "missing"),
            allowed_features=("known", "external_ai_provider"),
            provider_required=True,
            revised_fields=("known",),
            training_reference={"known": {"mean": 0.0, "std": 1.0}},
            maximum_z_score=5.0,
        )
        self.assertEqual(
            {issue.code for issue in feature_report.issues},
            {
                "MISSING_REQUIRED_FEATURES",
                "SCHEMA_DRIFT",
                "MISSING_PROVIDER_DATA",
                "REVISED_VALUE",
                "FEATURE_OUTLIER",
            },
        )
        bundle_report = validator.validate_bundle_manifest(
            {"mode": "daily_split", "total_row_counts": {"candles": 2}, "unfinished_days": ["2026-01-01"]}
        )
        self.assertIn("INCOMPLETE_DAILY_BUNDLE", {issue.code for issue in bundle_report.issues})


class SyncVerificationTests(unittest.TestCase):
    @staticmethod
    def _archive(root: Path) -> Path:
        folder = root / "export"
        folder.mkdir()
        candles = folder / "candles.csv.gz"
        with gzip.open(candles, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("symbol", "interval", "open_time", "open", "high", "low", "close", "volume"),
            )
            writer.writeheader()
            writer.writerow(
                {
                    "symbol": "BTCUSDT",
                    "interval": "1m",
                    "open_time": "2026-01-01T00:00:00+00:00",
                    "open": 100,
                    "high": 101,
                    "low": 99,
                    "close": 100,
                    "volume": 10,
                }
            )
            writer.writerow(
                {
                    "symbol": "BTCUSDT",
                    "interval": "1m",
                    "open_time": "2026-01-01T00:02:00+00:00",
                    "open": 100,
                    "high": 102,
                    "low": 99,
                    "close": 101,
                    "volume": 11,
                }
            )
        news = folder / "news_articles.jsonl.gz"
        with gzip.open(news, "wt", encoding="utf-8") as handle:
            handle.write(json.dumps({"created_at": "2026-01-01T00:01:00+00:00", "title": "test"}) + "\n")
        external = folder / "external_data_events.jsonl.gz"
        with gzip.open(external, "wt", encoding="utf-8") as handle:
            handle.write(json.dumps({"event_time": "2026-01-01T00:01:30+00:00", "source": "test"}) + "\n")
        (folder / "manifest.json").write_text(
            json.dumps(
                {
                    "tables_exported": {
                        "candles": candles.name,
                        "news_articles": news.name,
                        "external_data_events": external.name,
                    },
                    "row_counts": {},
                    "file_sizes": {},
                }
            ),
            encoding="utf-8",
        )
        refresh_export_manifests(folder)
        archive_path = root / "verified-export.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(folder.iterdir()):
                archive.write(path, path.name)
        return archive_path

    def test_archive_verification_proves_rows_checksums_times_and_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = self._archive(Path(temporary))
            result = verify_export_archive(archive)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["row_counts"]["candles"], 2)
        self.assertEqual(result["news_count"], 1)
        self.assertEqual(result["derivatives_count"], 1)
        self.assertEqual(result["missing_interval_summary"]["estimated_missing_intervals"], 1)
        self.assertEqual(result["checksum_files_verified"], 3)
        self.assertTrue(result["successful_local_file_close"])

    def test_remote_cleanup_requires_and_compares_the_exact_local_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = self._archive(root)
            railway_root = root / "railway"
            railway_root.mkdir()
            railway_archive = railway_root / archive.name
            railway_archive.write_bytes(archive.read_bytes())
            verification = verify_export_archive(archive)
            payload = {
                "archive_id": railway_archive.name,
                "local_manifest_verified": True,
                "local_verification": verification,
                "local_archive_sha256": verification["archive_sha256"],
                "local_archive_size_bytes": verification["archive_size_bytes"],
                "delete_archive": True,
                "delete_finished_data": False,
                "delete_db_rows": False,
            }
            with patch("app.services.raw_data_cleanup.RAW_EXPORT_ROOT", railway_root):
                result = cleanup_downloaded_raw_data(object(), payload)
            self.assertTrue(result["cleanup_confirmation"])
            self.assertFalse(railway_archive.exists())

            with patch("app.services.raw_data_cleanup.RAW_EXPORT_ROOT", railway_root):
                with self.assertRaisesRegex(ValueError, "structured local verification"):
                    cleanup_downloaded_raw_data(object(), {"archive_id": "missing.zip", "local_manifest_verified": True})


if __name__ == "__main__":
    unittest.main()

