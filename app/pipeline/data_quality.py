"""Point-in-time validation shared by live inference and local research."""

from __future__ import annotations

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
    ) -> DataQualityReport:
        now = _aware(now) or datetime.now(timezone.utc)
        rows = list(candles)
        issues: list[DataQualityIssue] = []
        timestamps: list[datetime] = []
        seen: set[datetime] = set()
        expected_interval = _interval_seconds(interval)

        for index, row in enumerate(rows):
            value = self._value(row, "open_time")
            timestamp = _aware(value) if isinstance(value, datetime) else None
            if timestamp is None:
                issues.append(DataQualityIssue("MISSING_TIMESTAMP", "ERROR", "Candle has no valid open_time.", {"index": index}))
                continue
            if timestamps and timestamp <= timestamps[-1]:
                issues.append(
                    DataQualityIssue(
                        "NON_MONOTONIC_TIMESTAMP",
                        "ERROR",
                        "Candle timestamps are not strictly increasing in input order.",
                        {
                            "index": index,
                            "previous": timestamps[-1].isoformat(),
                            "current": timestamp.isoformat(),
                        },
                    )
                )
            timestamps.append(timestamp)
            if timestamp in seen:
                issues.append(DataQualityIssue("DUPLICATE_CANDLE", "ERROR", "Duplicate candle open_time.", {"timestamp": timestamp.isoformat()}))
            seen.add(timestamp)
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

        sorted_times = sorted(timestamps)
        for prior, current in zip(sorted_times, sorted_times[1:]):
            if expected_interval and current - prior > timedelta(seconds=expected_interval * 1.5):
                missed = max(round((current - prior).total_seconds() / expected_interval) - 1, 1)
                issues.append(
                    DataQualityIssue(
                        "MISSING_CANDLE_INTERVAL",
                        "WARNING",
                        "One or more expected candle intervals are absent.",
                        {"from": prior.isoformat(), "to": current.isoformat(), "estimated_missing": missed},
                    )
                )
        return DataQualityReport(
            checked_at=now,
            issues=tuple(issues),
            first_timestamp=sorted_times[0] if sorted_times else None,
            last_timestamp=sorted_times[-1] if sorted_times else None,
            row_count=len(rows),
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
