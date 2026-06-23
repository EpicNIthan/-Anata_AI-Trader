from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sqlalchemy import desc, select

from app.db.models import ModelVersion, TrainingRun
from app.db.session import SessionLocal, create_db_and_tables
from app.training.export_dataset import export_dataset, parse_since_date


def _latest_model_path() -> Path:
    with SessionLocal() as session:
        model = session.scalar(select(ModelVersion).order_by(desc(ModelVersion.created_at)).limit(1))
        if model is None:
            raise ValueError("No model version exists yet.")
        return Path(model.path)


def evaluate_model(
    dataset_path: Path | None = None,
    model_path: Path | None = None,
    *,
    since_date: datetime | None = None,
    use_all_data: bool = False,
) -> dict[str, float]:
    create_db_and_tables()
    model_path = model_path or _latest_model_path()
    model = json.loads(model_path.read_text(encoding="utf-8"))
    feature_columns = model["feature_columns"]
    feature_schema_version = model.get("feature_schema_version", "price-news-v1")
    if dataset_path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        dataset_path = Path("datasets") / f"eval_features_{stamp}.csv"
        export_dataset(
            dataset_path,
            feature_schema_version=feature_schema_version,
            feature_columns=feature_columns,
            since_date=since_date,
            use_all_data=use_all_data,
        )
    coefficients = np.asarray([model["intercept"], *model["coefficients"]], dtype=float)

    rows: list[list[float]] = []
    targets: list[float] = []
    with dataset_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            target = row.get("target_next_price_change")
            if target in (None, ""):
                continue
            rows.append([float(row.get(column) or 0.0) for column in feature_columns])
            targets.append(float(target))
    if not rows:
        raise ValueError("Dataset has no labeled rows.")

    x = np.asarray(rows, dtype=float)
    y = np.asarray(targets, dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    predictions = design @ coefficients
    metrics = {
        "mse": float(np.mean((predictions - y) ** 2)),
        "mae": float(np.mean(np.abs(predictions - y))),
        "directional_accuracy": float(np.mean(np.sign(predictions) == np.sign(y))),
        "rows": float(len(y)),
    }

    with SessionLocal() as session:
        session.add(
            TrainingRun(
                model_name=model["name"],
                dataset_path=str(dataset_path),
                feature_schema_version=feature_schema_version,
                from_checkpoint_path=str(model_path),
                since_date=since_date,
                use_all_data=use_all_data,
                status="evaluated",
                metrics=metrics,
                finished_at=datetime.now(timezone.utc),
            )
        )
        session.commit()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved price model.")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--from-checkpoint", type=Path, default=None)
    parser.add_argument("--since-date", default=None, help="UTC date/time filter, for example 2026-06-24")
    parser.add_argument("--use-all-data", action="store_true")
    args = parser.parse_args()
    metrics = evaluate_model(
        args.dataset,
        args.model or args.from_checkpoint,
        since_date=parse_since_date(args.since_date),
        use_all_data=args.use_all_data,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
