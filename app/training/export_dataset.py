from __future__ import annotations

import argparse
import csv
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import or_, select

from app.db.models import Feature, TrainingFeature
from app.db.session import SessionLocal, create_db_and_tables
from app.features.schema import CURRENT_FEATURE_SCHEMA_VERSION, columns_for_schema, values_from_feature


FEATURE_COLUMNS = columns_for_schema(CURRENT_FEATURE_SCHEMA_VERSION)
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


def parse_since_date(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def export_dataset(
    output_path: Path,
    *,
    feature_schema_version: str = CURRENT_FEATURE_SCHEMA_VERSION,
    feature_columns: list[str] | None = None,
    since_date: datetime | None = None,
    use_all_data: bool = False,
) -> Path:
    create_db_and_tables()
    feature_columns = feature_columns or columns_for_schema(feature_schema_version)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with SessionLocal() as session:
        query = select(TrainingFeature).where(
            or_(TrainingFeature.schema_version == feature_schema_version, TrainingFeature.schema_version.is_(None))
        )
        if since_date and not use_all_data:
            query = query.where(TrainingFeature.as_of >= since_date)
        features = list(session.scalars(query.order_by(TrainingFeature.symbol, TrainingFeature.as_of)))
        if not features:
            fallback_query = select(Feature).where(
                or_(Feature.schema_version == feature_schema_version, Feature.schema_version.is_(None))
            )
            if since_date and not use_all_data:
                fallback_query = fallback_query.where(Feature.as_of >= since_date)
            features = list(session.scalars(fallback_query.order_by(Feature.symbol, Feature.as_of)))

    rows: list[dict[str, object]] = []
    for index, feature in enumerate(features):
        next_feature = features[index + 1] if index + 1 < len(features) else None
        if next_feature is None or next_feature.symbol != feature.symbol:
            target = ""
        else:
            target_payload = next_feature.payload if isinstance(next_feature, TrainingFeature) else next_feature
            target = values_from_feature(target_payload, ["price_change"])["price_change"]
        values = values_from_feature(feature.payload if isinstance(feature, TrainingFeature) else feature, feature_columns)
        payload_values = (feature.payload or {}).get("values", {}) if isinstance(feature, TrainingFeature) else (feature.payload or {}).get("values", {})
        if payload_values.get("target_future_return_15m") not in (None, ""):
            target = payload_values.get("target_future_return_15m")
        rows.append(
            {
                "feature_id": feature.source_feature_id if isinstance(feature, TrainingFeature) else feature.id,
                "training_feature_id": feature.id if isinstance(feature, TrainingFeature) else "",
                "symbol": feature.symbol,
                "as_of": feature.as_of.isoformat(),
                "feature_schema_version": feature.schema_version or feature_schema_version,
                "trend": values.get("trend") or "sideways",
                "target_next_price_change": target,
                "final_ai_input": json.dumps(payload_values.get("final_ai_input") or {}, separators=(",", ":")),
                **{column: values.get(column, 0.0) for column in feature_columns},
                **{column: payload_values.get(column, "") for column in TARGET_COLUMNS},
            }
        )

    opener = gzip.open if output_path.suffix == ".gz" else open
    with opener(output_path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "feature_id",
                "training_feature_id",
                "symbol",
                "as_of",
                "feature_schema_version",
                "trend",
                "final_ai_input",
                *feature_columns,
                "target_next_price_change",
                *TARGET_COLUMNS,
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export feature rows for AI model training.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--feature-schema-version", default=CURRENT_FEATURE_SCHEMA_VERSION)
    parser.add_argument("--since-date", default=None, help="UTC date/time filter, for example 2026-06-24")
    parser.add_argument("--use-all-data", action="store_true")
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = args.output or Path("datasets") / f"anata_dataset_{timestamp}.csv.gz"
    path = export_dataset(
        output,
        feature_schema_version=args.feature_schema_version,
        since_date=parse_since_date(args.since_date),
        use_all_data=args.use_all_data,
    )
    print(f"exported={path}")


if __name__ == "__main__":
    main()
