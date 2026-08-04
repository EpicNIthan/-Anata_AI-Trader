"""Point-in-time validation shared by live inference and local research."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from app.db.models import Candle, Feature, NewsArticle
from app.features.schema import values_from_feature
from app.pipeline.domain import FeatureSnapshot, new_id


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _interval_seconds(interval: str) -> int | None:
    raw = (interval or "").strip().lower()
    if len(raw) < 2:
        return None
    try:
        amount = int(raw[:-1])
    except ValueError:
        return None
    units = {"m": 60, "h": 3600, "d": 86400, "w": 604800}
    return amount * units.get(raw[-1], 0) if amount > 0 and raw[-1] in units else None


@dataclass(frozen=True)
class DataQualityIssue:
    code: str
    severity: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DataQualityReport:
    checked_at: datetime
    issues: tuple[DataQualityIssue, ...]
    first_timestamp: datetime | None
    last_timestamp: datetime | None
    row_count: int

    @property
    def valid(self) -> bool:
        return not any(issue.severity in {"ERROR", "CRITICAL"} for issue in self.issues)

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "checked_at": self.checked_at.isoformat(),
            "row_count": self.row_count,
            "first_timestamp": self.first_timestamp.isoformat() if self.first_timestamp else None,
            "last_timestamp": self.last_timestamp.isoformat() if self.last_timestamp else None,
            "issues": [
                {"code": item.code, "severity": item.severity, "message": item.message, "context": item.context}
                for item in self.issues
            ],
        }


class PointInTimeValidator:
    """Validate market/news records without silently repairing historical data."""

    def validate_candles(
        self,
        candles: Iterable[Candle | dict[str, Any]],
        *,
        interval: str,
        now: datetime | None = None,
        stale_after_seconds: float | None = None,
        outlier_return_threshold: float = 0.50,
    ) -> DataQualityReport:
        now = _aware(now) or datetime.now(timezone.utc)
        rows = list(candles)
        issues: list[DataQualityIssue] = []
        timestamps: list[datetime] = []
        seen: set[tuple[str, str, datetime]] = set()
        last_by_series: dict[tuple[str, str], datetime] = {}
        times_by_series: dict[tuple[str, str], list[datetime]] = {}
        observed_values: dict[tuple[str, str, datetime], tuple[float, float, float, float, float]] = {}
        previous_close: dict[tuple[str, str], float] = {}
        expected_interval = _interval_seconds(interval)

        for index, row in enumerate(rows):
            value = self._value(row, "open_time")
            timestamp = _aware(value) if isinstance(value, datetime) else None
            if timestamp is None:
                issues.append(DataQualityIssue("MISSING_TIMESTAMP", "ERROR", "Candle has no valid open_time.", {"index": index}))
                continue
            symbol = str(self._value(row, "symbol") or "UNKNOWN").upper()
            row_interval = str(self._value(row, "interval") or interval).lower()
            series_key = (symbol, row_interval)
            identity = (symbol, row_interval, timestamp)
            previous_timestamp = last_by_series.get(series_key)
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                issues.append(
                    DataQualityIssue(
                        "NON_MONOTONIC_TIMESTAMP",
                        "ERROR",
                        "Candle timestamps are not strictly increasing within a symbol/interval series.",
                        {
                            "index": index,
                            "symbol": symbol,
                            "interval": row_interval,
                            "previous": previous_timestamp.isoformat(),
                            "current": timestamp.isoformat(),
                        },
                    )
                )
            last_by_series[series_key] = timestamp
            times_by_series.setdefault(series_key, []).append(timestamp)
            timestamps.append(timestamp)
            if identity in seen:
                issues.append(
                    DataQualityIssue(
                        "DUPLICATE_CANDLE",
                        "ERROR",
                        "Duplicate candle symbol/interval/open_time identity.",
                        {"symbol": symbol, "interval": row_interval, "timestamp": timestamp.isoformat()},
                    )
                )
            seen.add(identity)
            if timestamp > now + timedelta(seconds=5):
                issues.append(DataQualityIssue("FUTURE_TIMESTAMP", "ERROR", "Candle timestamp is in the future.", {"timestamp": timestamp.isoformat()}))

            open_price = self._number(row, "open")
            high = self._number(row, "high")
            low = self._number(row, "low")
            close = self._number(row, "close")
            volume = self._number(row, "volume")
            if min(open_price, high, low, close) < 0 or high < max(open_price, close) or low > min(open_price, close) or high < low:
                issues.append(DataQualityIssue("INVALID_OHLC", "ERROR", "Candle OHLC relationship is invalid.", {"index": index}))
            if volume < 0:
                issues.append(DataQualityIssue("NEGATIVE_VOLUME", "ERROR", "Candle volume cannot be negative.", {"index": index}))

            price_tuple = (open_price, high, low, close, volume)
            prior_values = observed_values.get(identity)
            if prior_values is not None and prior_values != price_tuple:
                issues.append(
                    DataQualityIssue(
                        "REVISED_VALUE",
                        "WARNING",
                        "Duplicate candle identity carries revised OHLCV values.",
                        {"symbol": symbol, "interval": row_interval, "timestamp": timestamp.isoformat()},
                    )
                )
            observed_values[identity] = price_tuple
            prior_close = previous_close.get(series_key)
            if (
                prior_close is not None
                and prior_close > 0
                and close > 0
                and outlier_return_threshold > 0
                and abs(close / prior_close - 1.0) > outlier_return_threshold
            ):
                issues.append(
                    DataQualityIssue(
                        "PRICE_OUTLIER",
                        "WARNING",
                        "Candle return exceeds the configured deterministic outlier threshold.",
                        {
                            "symbol": symbol,
                            "interval": row_interval,
                            "timestamp": timestamp.isoformat(),
                            "prior_close": prior_close,
                            "close": close,
                            "threshold": outlier_return_threshold,
                        },
                    )
                )
            if close > 0:
                previous_close[series_key] = close

        sorted_times = sorted(timestamps)
        for (symbol, row_interval), series_times in times_by_series.items():
            seconds = _interval_seconds(row_interval) or expected_interval
            ordered = sorted(series_times)
            for prior, current in zip(ordered, ordered[1:]):
                if seconds and current - prior > timedelta(seconds=seconds * 1.5):
                    missed = max(round((current - prior).total_seconds() / seconds) - 1, 1)
                    issues.append(
                        DataQualityIssue(
                            "MISSING_CANDLE_INTERVAL",
                            "WARNING",
                            "One or more expected candle intervals are absent.",
                            {
                                "symbol": symbol,
                                "interval": row_interval,
                                "from": prior.isoformat(),
                                "to": current.isoformat(),
                                "estimated_missing": missed,
                            },
                        )
                    )
        if stale_after_seconds is not None:
            if stale_after_seconds < 0:
                raise ValueError("stale_after_seconds cannot be negative")
            if not sorted_times or (now - sorted_times[-1]).total_seconds() > stale_after_seconds:
                issues.append(
                    DataQualityIssue(
                        "STALE_FEED",
                        "ERROR",
                        "Latest candle is older than the configured feed freshness limit.",
                        {
                            "last_timestamp": sorted_times[-1].isoformat() if sorted_times else None,
                            "stale_after_seconds": stale_after_seconds,
                        },
                    )
                )
        return DataQualityReport(
            checked_at=now,
            issues=tuple(issues),
            first_timestamp=sorted_times[0] if sorted_times else None,
            last_timestamp=sorted_times[-1] if sorted_times else None,
            row_count=len(rows),
        )

    def validate_feature_payload(
        self,
        values: dict[str, Any],
        *,
        required_features: Iterable[str] = (),
        allowed_features: Iterable[str] | None = None,
        provider_required: bool = False,
        provider_field: str = "external_ai_provider",
        revised_fields: Iterable[str] = (),
        training_reference: dict[str, dict[str, float]] | None = None,
        maximum_z_score: float = 6.0,
        now: datetime | None = None,
    ) -> DataQualityReport:
        """Validate schema, provider missingness, revisions and numeric feature drift."""

        checked_at = _aware(now) or datetime.now(timezone.utc)
        issues: list[DataQualityIssue] = []
        required = {str(name) for name in required_features}
        missing = sorted(name for name in required if values.get(name) in (None, ""))
        if missing:
            issues.append(
                DataQualityIssue(
                    "MISSING_REQUIRED_FEATURES",
                    "ERROR",
                    "One or more required live features are unavailable.",
                    {"features": missing},
                )
            )
        if allowed_features is not None:
            allowed = {str(name) for name in allowed_features}
            unexpected = sorted(set(values) - allowed)
            if unexpected:
                issues.append(
                    DataQualityIssue(
                        "SCHEMA_DRIFT",
                        "WARNING",
                        "Feature payload contains fields outside the registered schema.",
                        {"unexpected_features": unexpected},
                    )
                )
        if provider_required and values.get(provider_field) in (None, ""):
            issues.append(
                DataQualityIssue(
                    "MISSING_PROVIDER_DATA",
                    "WARNING",
                    "An optional provider was expected for this quality audit but is missing.",
                    {"provider_field": provider_field},
                )
            )
        revised = sorted({str(name) for name in revised_fields if str(name)})
        if revised:
            issues.append(
                DataQualityIssue(
                    "REVISED_VALUE",
                    "WARNING",
                    "One or more source values were revised after initial observation.",
                    {"features": revised},
                )
            )
        if maximum_z_score <= 0:
            raise ValueError("maximum_z_score must be positive")
        for name, reference in (training_reference or {}).items():
            try:
                value = float(values.get(name))
                expected_mean = float(reference["mean"])
                expected_std = float(reference["std"])
            except (KeyError, TypeError, ValueError):
                continue
            if not all(math.isfinite(item) for item in (value, expected_mean, expected_std)):
                issues.append(
                    DataQualityIssue(
                        "NON_FINITE_FEATURE",
                        "ERROR",
                        "A model input is NaN or infinite.",
                        {"feature": name},
                    )
                )
                continue
            if expected_std > 0 and abs(value - expected_mean) / expected_std > maximum_z_score:
                issues.append(
                    DataQualityIssue(
                        "FEATURE_OUTLIER",
                        "WARNING",
                        "Feature lies outside the configured training-reference z-score.",
                        {"feature": name, "value": value, "mean": expected_mean, "std": expected_std},
                    )
                )
        return DataQualityReport(
            checked_at=checked_at,
            issues=tuple(issues),
            first_timestamp=None,
            last_timestamp=None,
            row_count=1,
        )

    def validate_bundle_manifest(
        self,
        manifest: dict[str, Any],
        *,
        require_finished: bool = True,
        now: datetime | None = None,
    ) -> DataQualityReport:
        """Validate export completeness before a bundle can be treated as durable data."""

        checked_at = _aware(now) or datetime.now(timezone.utc)
        issues: list[DataQualityIssue] = []
        row_counts = manifest.get("row_counts", manifest.get("total_row_counts"))
        if not isinstance(row_counts, dict):
            issues.append(DataQualityIssue("MISSING_ROW_COUNTS", "ERROR", "Bundle manifest has no row-count map."))
            row_counts = {}
        checksums = manifest.get("file_checksums_sha256")
        if manifest.get("mode") != "daily_split" and not isinstance(checksums, dict):
            issues.append(DataQualityIssue("MISSING_FILE_CHECKSUMS", "ERROR", "Bundle manifest has no checksum map."))
        verification = manifest.get("verification") or {}
        if manifest.get("mode") != "daily_split" and verification.get("writers_closed") is not True:
            issues.append(DataQualityIssue("UNCONFIRMED_FILE_CLOSE", "ERROR", "Bundle writers were not confirmed closed."))
        unfinished = manifest.get("unfinished_days") or []
        if require_finished and unfinished:
            issues.append(
                DataQualityIssue(
                    "INCOMPLETE_DAILY_BUNDLE",
                    "ERROR",
                    "Bundle contains one or more unfinished UTC days.",
                    {"unfinished_days": list(unfinished)},
                )
            )
        ranges = manifest.get("table_time_ranges") or {}
        timestamps: list[datetime] = []
        for table_name, details in ranges.items():
            if not isinstance(details, dict):
                continue
            for value in (details.get("first_timestamp"), details.get("last_timestamp")):
                if not value:
                    continue
                try:
                    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                except ValueError:
                    issues.append(
                        DataQualityIssue(
                            "INVALID_MANIFEST_TIMESTAMP",
                            "ERROR",
                            "Bundle manifest contains a malformed table timestamp.",
                            {"table": str(table_name), "value": str(value)},
                        )
                    )
                    continue
                aware = _aware(parsed)
                if aware is not None:
                    timestamps.append(aware)
        return DataQualityReport(
            checked_at=checked_at,
            issues=tuple(issues),
            first_timestamp=min(timestamps) if timestamps else None,
            last_timestamp=max(timestamps) if timestamps else None,
            row_count=sum(int(value or 0) for value in row_counts.values()),
        )

    def validate_news_available(
        self,
        article: NewsArticle | dict[str, Any],
        *,
        decision_time: datetime,
    ) -> DataQualityReport:
        decision_time = _aware(decision_time) or datetime.now(timezone.utc)
        available = self._article_available_time(article)
        issues: list[DataQualityIssue] = []
        if available is None:
            issues.append(DataQualityIssue("MISSING_AVAILABILITY_TIME", "ERROR", "News has no available_to_model_time or received time.", {}))
        elif available > decision_time:
            issues.append(
                DataQualityIssue(
                    "FUTURE_NEWS_LEAKAGE",
                    "ERROR",
                    "News became available after the decision time and cannot affect it.",
                    {"available_to_model_time": available.isoformat(), "decision_time": decision_time.isoformat()},
                )
            )
        return DataQualityReport(
            checked_at=decision_time,
            issues=tuple(issues),
            first_timestamp=available,
            last_timestamp=available,
            row_count=1,
        )

    def snapshot_from_feature(
        self,
        feature: Feature,
        *,
        decision_time: datetime | None = None,
        required_features: Iterable[str] = (),
        data_version: str = "operational",
    ) -> FeatureSnapshot:
        """Convert an existing feature row into a non-leaking V2 snapshot."""
        available = _aware(feature.available_to_model_time) or _aware(feature.as_of) or _aware(feature.created_at)
        decision_time = _aware(decision_time) or available or datetime.now(timezone.utc)
        if available is not None and available > decision_time:
            raise ValueError(
                "feature was not available to the model at the requested decision time: "
                f"available_to_model_time={available.isoformat()} decision_time={decision_time.isoformat()}"
            )
        values = values_from_feature(feature)
        missing = [name for name in required_features if values.get(name) in (None, "")]
        payload = feature.payload or {}
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        source_freshness = metadata.get("source_freshness", {}) if isinstance(metadata, dict) else {}
        freshness: dict[str, float] = {}
        if isinstance(source_freshness, dict):
            for source, details in source_freshness.items():
                try:
                    if isinstance(details, dict):
                        age_hours = details.get("age_hours")
                        if age_hours is None:
                            continue
                        freshness[str(source)] = max(float(age_hours) * 3600.0, 0.0)
                    else:
                        freshness[str(source)] = max(float(details), 0.0)
                except (TypeError, ValueError):
                    continue
        external_context = {
            "external_ai_available": bool(values.get("external_ai_available", False)),
            "external_ai_missing": bool(values.get("external_ai_missing", True)),
            "external_ai_failed": bool(values.get("external_ai_failed", False)),
            "external_ai_confidence": values.get("external_ai_confidence"),
            "external_ai_age_seconds": values.get("external_ai_age_seconds"),
            "external_ai_provider": values.get("external_ai_provider"),
            "external_ai_prompt_version": values.get("external_ai_prompt_version"),
            "local_news_provider": values.get("local_news_provider")
            or metadata.get("local_news_provider"),
            "local_news_model_version": values.get("local_news_model_version")
            or metadata.get("local_news_model_version"),
        }
        return FeatureSnapshot(
            feature_snapshot_id=f"feature_db_{feature.id}" if feature.id else new_id("feature"),
            symbol=feature.symbol,
            as_of=decision_time,
            available_to_model_time=available or decision_time,
            schema_version=feature.schema_version,
            values=values,
            source_freshness_seconds=freshness,
            missing_required_features=missing,
            data_version=data_version,
            external_context=external_context,
        )

    @staticmethod
    def _value(row: Candle | dict[str, Any], key: str) -> Any:
        return row.get(key) if isinstance(row, dict) else getattr(row, key, None)

    def _number(self, row: Candle | dict[str, Any], key: str) -> float:
        try:
            return float(self._value(row, key) or 0.0)
        except (TypeError, ValueError):
            return -1.0

    def _article_available_time(self, article: NewsArticle | dict[str, Any]) -> datetime | None:
        if isinstance(article, dict):
            value = article.get("available_to_model_time") or article.get("received_time") or article.get("published_at")
        else:
            value = article.available_to_model_time or article.received_time or article.published_at
        return _aware(value) if isinstance(value, datetime) else None
