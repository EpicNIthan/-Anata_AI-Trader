from __future__ import annotations

from bisect import bisect_right
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Candle, Feature, TrainingFeature
from app.features.schema import CURRENT_FEATURE_SCHEMA_VERSION

TARGET_COLUMNS = [
    "target_future_return_5m",
    "target_future_return_15m",
    "target_future_return_1h",
    "target_future_return_4h",
    "target_max_upside_1h",
    "target_max_drawdown_1h",
    "target_stop_loss_hit_first",
    "target_take_profit_hit_first",
    "target_direction_15m",
    "target_trade_quality_score",
]


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _timestamp(value: datetime | None) -> float:
    aware = _aware(value)
    return aware.timestamp() if aware else 0.0


def _safe_pct_change(current: float, previous: float) -> float:
    return (current - previous) / previous if previous else 0.0


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _horizon_rows(interval: str) -> dict[str, int]:
    if interval.endswith("m"):
        minutes = max(int(interval[:-1] or "1"), 1)
        return {
            "5m": max(1, 5 // minutes),
            "15m": max(1, 15 // minutes),
            "1h": max(1, 60 // minutes),
            "4h": max(1, 240 // minutes),
        }
    return {"5m": 5, "15m": 15, "1h": 60, "4h": 240}


def values_for_training_feature(row: TrainingFeature | Feature) -> dict[str, Any]:
    payload = row.payload or {}
    values = dict(payload.get("values", {}))
    if isinstance(row, TrainingFeature):
        values.update(row.feature_values or {})
        values.pop("final_ai_input", None)
    return values


def _is_labeled(row: TrainingFeature) -> bool:
    value = values_for_training_feature(row).get("target_trade_quality_score")
    return value not in (None, "")


def _interval_for_feature(row: TrainingFeature) -> str:
    metadata = (row.payload or {}).get("metadata", {})
    return str(metadata.get("interval") or settings.paper_trade_timeframe or "1m")


def _label_values(candles: list[Candle], entry_index: int, horizons: dict[str, int]) -> dict[str, Any] | None:
    required_horizon = horizons["4h"]
    if entry_index < 0 or entry_index + required_horizon >= len(candles):
        return None
    entry_price = candles[entry_index].close
    if not entry_price:
        return None
    future_1h = candles[entry_index + 1 : entry_index + 1 + horizons["1h"]]
    max_future_high = max((row.high for row in future_1h), default=entry_price)
    min_future_low = min((row.low for row in future_1h), default=entry_price)
    stop_loss = entry_price * (1.0 - settings.auto_default_stop_loss_pct)
    take_profit = entry_price * (1.0 + settings.auto_default_take_profit_pct)
    stop_index = next((index for index, row in enumerate(future_1h, start=1) if row.low <= stop_loss), None)
    take_index = next((index for index, row in enumerate(future_1h, start=1) if row.high >= take_profit), None)
    if stop_index and take_index:
        first_exit = "take_profit" if take_index <= stop_index else "stop_loss"
    elif take_index:
        first_exit = "take_profit"
    elif stop_index:
        first_exit = "stop_loss"
    else:
        first_exit = "none"

    future_return_5m = _safe_pct_change(candles[entry_index + horizons["5m"]].close, entry_price)
    future_return_15m = _safe_pct_change(candles[entry_index + horizons["15m"]].close, entry_price)
    future_return_1h = _safe_pct_change(candles[entry_index + horizons["1h"]].close, entry_price)
    future_return_4h = _safe_pct_change(candles[entry_index + horizons["4h"]].close, entry_price)
    max_upside_1h = _safe_pct_change(max_future_high, entry_price)
    max_drawdown_1h = _safe_pct_change(min_future_low, entry_price)
    required_edge = settings.paper_fee_rate * 2.0 + settings.strategy_min_edge_after_fees
    if future_return_15m > required_edge:
        direction = 1.0
    elif future_return_15m < -required_edge:
        direction = -1.0
    else:
        direction = 0.0
    quality = _clamp(
        (future_return_15m / max(required_edge * 4.0, 1e-9)) * 0.45
        + (max_upside_1h / max(settings.auto_default_take_profit_pct, 1e-9)) * 0.25
        + (max_drawdown_1h / max(settings.auto_default_stop_loss_pct, 1e-9)) * 0.25
        + (0.15 if first_exit == "take_profit" else 0.0)
        - (0.20 if first_exit == "stop_loss" else 0.0),
        -1.0,
        1.0,
    )
    return {
        "target_future_return_5m": future_return_5m,
        "target_future_return_15m": future_return_15m,
        "target_future_return_1h": future_return_1h,
        "target_future_return_4h": future_return_4h,
        "target_max_upside_1h": max_upside_1h,
        "target_max_drawdown_1h": max_drawdown_1h,
        "target_stop_loss_hit_first": 1.0 if first_exit == "stop_loss" else 0.0,
        "target_take_profit_hit_first": 1.0 if first_exit == "take_profit" else 0.0,
        "target_direction_15m": direction,
        "target_trade_quality_score": quality,
    }


def _write_labels(row: TrainingFeature, labels: dict[str, Any]) -> None:
    values = values_for_training_feature(row)
    values.update(labels)
    payload = dict(row.payload or {})
    payload["schema_version"] = row.schema_version or payload.get("schema_version") or CURRENT_FEATURE_SCHEMA_VERSION
    payload["values"] = values
    metadata = dict(payload.get("metadata") or {})
    metadata["labels_built_at"] = datetime.now(timezone.utc).isoformat()
    metadata["label_source"] = "closed_candles_future_window"
    payload["metadata"] = metadata
    row.feature_values = values
    row.payload = payload


def _sync_features_to_training_features(session: Session, schema_version: str) -> int:
    created = 0
    feature_rows = session.scalars(select(Feature).where(Feature.schema_version == schema_version)).all()
    for feature in feature_rows:
        if session.scalar(select(TrainingFeature.id).where(TrainingFeature.source_feature_id == feature.id).limit(1)):
            continue
        values = dict((feature.payload or {}).get("values", {}))
        values.pop("final_ai_input", None)
        payload = dict(feature.payload or {})
        payload["values"] = values
        metadata = dict(payload.get("metadata") or {})
        metadata.pop("news_context", None)
        metadata.pop("derivatives_context", None)
        metadata["debug_payload"] = "full final_ai_input kept on recent features rows only"
        payload["metadata"] = metadata
        session.add(
            TrainingFeature(
                source_feature_id=feature.id,
                symbol=feature.symbol,
                schema_version=feature.schema_version,
                source_name=feature.source_name,
                as_of=feature.as_of,
                feature_values=values,
                payload=payload,
            )
        )
        created += 1
    if created:
        session.flush()
    return created


def build_labels_for_existing_features(
    session: Session,
    *,
    symbols: list[str] | None = None,
    interval: str | None = None,
    schema_version: str = CURRENT_FEATURE_SCHEMA_VERSION,
    force: bool = False,
    limit: int | None = None,
    sync_features: bool = True,
) -> dict[str, Any]:
    normalized_symbols = [symbol.upper() for symbol in symbols] if symbols else None
    synced_features = _sync_features_to_training_features(session, schema_version) if sync_features else 0
    query = select(TrainingFeature).where(TrainingFeature.schema_version == schema_version)
    if normalized_symbols:
        query = query.where(TrainingFeature.symbol.in_(normalized_symbols))
    if limit:
        query = query.limit(max(limit, 1))
    features = list(session.scalars(query.order_by(TrainingFeature.symbol, TrainingFeature.as_of)))
    candles_by_key: dict[tuple[str, str], list[Candle]] = {}
    times_by_key: dict[tuple[str, str], list[float]] = {}
    labeled = 0
    skipped_no_future = 0
    skipped_no_candles = 0
    skipped_existing = 0
    per_symbol: dict[str, dict[str, int]] = {}

    for feature in features:
        if not force and _is_labeled(feature):
            skipped_existing += 1
            continue
        feature_interval = interval or _interval_for_feature(feature)
        key = (feature.symbol, feature_interval)
        if key not in candles_by_key:
            candles = list(
                session.scalars(
                    select(Candle)
                    .where(
                        Candle.symbol == feature.symbol,
                        Candle.interval == feature_interval,
                        Candle.is_closed.is_(True),
                    )
                    .order_by(Candle.open_time)
                )
            )
            candles_by_key[key] = candles
            times_by_key[key] = [_timestamp(row.close_time or row.open_time) for row in candles]
        candles = candles_by_key[key]
        per_symbol.setdefault(feature.symbol, {"labeled": 0, "skipped_no_future": 0, "skipped_no_candles": 0})
        if not candles:
            skipped_no_candles += 1
            per_symbol[feature.symbol]["skipped_no_candles"] += 1
            continue
        entry_index = bisect_right(times_by_key[key], _timestamp(feature.as_of)) - 1
        if entry_index < 0:
            skipped_no_candles += 1
            per_symbol[feature.symbol]["skipped_no_candles"] += 1
            continue
        labels = _label_values(candles, entry_index, _horizon_rows(feature_interval))
        if labels is None:
            skipped_no_future += 1
            per_symbol[feature.symbol]["skipped_no_future"] += 1
            continue
        _write_labels(feature, labels)
        labeled += 1
        per_symbol[feature.symbol]["labeled"] += 1
    session.commit()
    return {
        "schema_version": schema_version,
        "symbols": normalized_symbols or "all",
        "interval": interval or "feature/default",
        "features_seen": len(features),
        "synced_features": synced_features,
        "labeled": labeled,
        "skipped_existing": skipped_existing,
        "skipped_no_future": skipped_no_future,
        "skipped_no_candles": skipped_no_candles,
        "per_symbol": per_symbol,
    }


def label_status(session: Session, *, schema_version: str = CURRENT_FEATURE_SCHEMA_VERSION) -> dict[str, Any]:
    rows = list(session.scalars(select(TrainingFeature).where(TrainingFeature.schema_version == schema_version)))
    feature_rows = list(session.scalars(select(Feature).where(Feature.schema_version == schema_version)))
    counts = {column: 0 for column in TARGET_COLUMNS}
    labeled_times: list[datetime] = []
    unlabeled_times: list[datetime] = []
    for row in rows:
        values = values_for_training_feature(row)
        for column in TARGET_COLUMNS:
            if values.get(column) not in (None, ""):
                counts[column] += 1
        if values.get("target_trade_quality_score") not in (None, ""):
            labeled_times.append(row.as_of)
        else:
            unlabeled_times.append(row.as_of)

    candle_count_by_symbol = {
        symbol: count
        for symbol, count in session.execute(
            select(Candle.symbol, func.count(Candle.id))
            .where(Candle.is_closed.is_(True))
            .group_by(Candle.symbol)
        ).all()
    }
    candles_by_symbol: dict[str, list[Candle]] = {}
    times_by_symbol: dict[str, list[float]] = {}
    for symbol in {row.symbol for row in rows}:
        candles = list(
            session.scalars(
                select(Candle)
                .where(Candle.symbol == symbol, Candle.interval == settings.paper_trade_timeframe, Candle.is_closed.is_(True))
                .order_by(Candle.open_time)
            )
        )
        candles_by_symbol[symbol] = candles
        times_by_symbol[symbol] = [_timestamp(candle.close_time or candle.open_time) for candle in candles]
    enough_future: dict[str, Any] = {}
    for horizon_name in ("5m", "15m", "1h"):
        horizon_rows = _horizon_rows(settings.paper_trade_timeframe).get(horizon_name, 1)
        possible = 0
        for row in rows:
            candles = candles_by_symbol.get(row.symbol, [])
            if not candles:
                continue
            times = times_by_symbol.get(row.symbol, [])
            entry_index = bisect_right(times, _timestamp(row.as_of)) - 1
            if entry_index >= 0 and entry_index + horizon_rows < len(candles):
                possible += 1
        enough_future[horizon_name] = {
            "rows_possible": possible,
            "enough_for_any_rows": possible > 0,
        }
    target_count = counts["target_trade_quality_score"]
    total = len(rows)
    all_times = [row.as_of for row in rows if row.as_of] + [row.as_of for row in feature_rows if row.as_of]
    symbols_covered = sorted({row.symbol for row in rows} | {row.symbol for row in feature_rows})
    return {
        "schema_version": schema_version,
        "selected_training_target": settings.model_target,
        "total_feature_rows": len(feature_rows),
        "total_training_feature_rows": total,
        "total_training_features": total,
        "unlabeled_rows": total - target_count,
        "earliest_feature_time": min(all_times).isoformat() if all_times else None,
        "latest_feature_time": max(all_times).isoformat() if all_times else None,
        "symbols_covered": symbols_covered,
        "rows_with_target_future_return_5m": counts["target_future_return_5m"],
        "rows_with_target_future_return_15m": counts["target_future_return_15m"],
        "rows_with_target_future_return_1h": counts["target_future_return_1h"],
        "rows_with_target_direction_15m": counts["target_direction_15m"],
        "rows_with_target_trade_quality_score": target_count,
        "labeled_rows_by_target": counts,
        "label_coverage_pct": (target_count / total) if total else 0.0,
        "label_readiness": "OK" if target_count > 0 else "NOT READY",
        "latest_labeled_as_of": max(labeled_times).isoformat() if labeled_times else None,
        "oldest_unlabeled_as_of": min(unlabeled_times).isoformat() if unlabeled_times else None,
        "candle_count_by_symbol": candle_count_by_symbol,
        "enough_future_candles": enough_future,
        "warning": "target_trade_quality_score labeled rows = 0" if target_count == 0 else None,
    }
