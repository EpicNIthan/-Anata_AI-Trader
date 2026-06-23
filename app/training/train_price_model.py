from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.config import settings
from app.db.models import ModelVersion, TrainingRun
from app.db.session import SessionLocal, create_db_and_tables
from app.features.schema import CURRENT_FEATURE_SCHEMA_VERSION, columns_for_schema
from app.training.export_dataset import export_dataset, parse_since_date


def _load_dataset(path: Path, feature_columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    targets: list[float] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            target = row.get("target_next_price_change")
            if target in (None, ""):
                continue
            rows.append([float(row.get(column) or 0.0) for column in feature_columns])
            targets.append(float(target))
    if not rows:
        raise ValueError("Dataset has no labeled rows. Build more features before training.")
    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)


def _load_checkpoint(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _initial_weights(feature_columns: list[str], checkpoint: dict[str, object] | None) -> np.ndarray:
    weights = np.zeros(len(feature_columns) + 1, dtype=float)
    if not checkpoint:
        return weights
    weights[0] = float(checkpoint.get("intercept", 0.0) or 0.0)
    old_columns = list(checkpoint.get("feature_columns", []))
    old_coefficients = list(checkpoint.get("coefficients", []))
    for old_index, column in enumerate(old_columns):
        if old_index >= len(old_coefficients) or column not in feature_columns:
            continue
        new_index = feature_columns.index(column)
        weights[new_index + 1] = float(old_coefficients[old_index] or 0.0)
    return weights


def _fit_with_checkpoint(x: np.ndarray, y: np.ndarray, initial: np.ndarray, epochs: int, learning_rate: float) -> np.ndarray:
    weights = initial.copy()
    design = np.column_stack([np.ones(len(x)), x])
    for _ in range(max(epochs, 1)):
        errors = design @ weights - y
        gradient = (design.T @ errors) / len(y)
        weights -= learning_rate * gradient
    return weights


def _fit_from_zero(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return np.asarray(coefficients, dtype=float)


def train_price_model(
    dataset_path: Path | None = None,
    *,
    from_checkpoint: Path | None = None,
    since_date: datetime | None = None,
    use_all_data: bool = False,
    feature_schema_version: str = CURRENT_FEATURE_SCHEMA_VERSION,
    epochs: int = 500,
    learning_rate: float = 0.05,
) -> Path:
    create_db_and_tables()
    started = datetime.now(timezone.utc)
    checkpoint = _load_checkpoint(from_checkpoint)
    if checkpoint and checkpoint.get("feature_schema_version"):
        feature_schema_version = str(checkpoint["feature_schema_version"])
    feature_columns = list(checkpoint.get("feature_columns", [])) if checkpoint else columns_for_schema(feature_schema_version)
    if not feature_columns:
        feature_columns = columns_for_schema(feature_schema_version)
    if dataset_path is None:
        dataset_path = Path("datasets") / f"features_{started.strftime('%Y%m%d_%H%M%S')}.csv"
        export_dataset(
            dataset_path,
            feature_schema_version=feature_schema_version,
            feature_columns=feature_columns,
            since_date=since_date,
            use_all_data=use_all_data,
        )

    x, y = _load_dataset(dataset_path, feature_columns)
    design = np.column_stack([np.ones(len(x)), x])
    if checkpoint:
        coefficients = _fit_with_checkpoint(
            x,
            y,
            _initial_weights(feature_columns, checkpoint),
            epochs=epochs,
            learning_rate=learning_rate,
        )
    else:
        coefficients = _fit_from_zero(x, y)
    predictions = design @ coefficients
    mse = float(np.mean((predictions - y) ** 2))
    mae = float(np.mean(np.abs(predictions - y)))
    directional_accuracy = float(np.mean(np.sign(predictions) == np.sign(y)))

    version = started.strftime("%Y%m%d_%H%M%S")
    model_id = f"price-linear-regression:{version}"
    parent_model_id = str(checkpoint.get("model_id")) if checkpoint and checkpoint.get("model_id") else None
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = settings.model_dir / f"price_linear_{version}.json"
    payload = {
        "model_id": model_id,
        "name": "price-linear-regression",
        "version": version,
        "feature_schema_version": feature_schema_version,
        "feature_columns": feature_columns,
        "parent_model_id": parent_model_id,
        "from_checkpoint": str(from_checkpoint) if from_checkpoint else None,
        "intercept": float(coefficients[0]),
        "coefficients": [float(value) for value in coefficients[1:]],
        "metrics": {
            "mse": mse,
            "mae": mae,
            "directional_accuracy": directional_accuracy,
            "rows": int(len(y)),
            "continued_from_checkpoint": bool(checkpoint),
        },
        "created_at": started.isoformat(),
    }
    model_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with SessionLocal() as session:
        model = ModelVersion(
            model_id=model_id,
            name=payload["name"],
            version=version,
            feature_schema_version=feature_schema_version,
            feature_columns=feature_columns,
            path=str(model_path),
            parent_model_id=parent_model_id,
            checkpoint_path=str(from_checkpoint) if from_checkpoint else None,
            status="trained",
            metrics=payload["metrics"],
            raw_payload=payload,
        )
        session.add(model)
        session.flush()
        session.add(
            TrainingRun(
                model_name=payload["name"],
                dataset_path=str(dataset_path),
                model_version_id=model.id,
                feature_schema_version=feature_schema_version,
                from_checkpoint_path=str(from_checkpoint) if from_checkpoint else None,
                since_date=since_date,
                use_all_data=use_all_data,
                status="finished",
                metrics=payload["metrics"],
                started_at=started,
                finished_at=datetime.now(timezone.utc),
            )
        )
        session.commit()
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a first-pass price movement model.")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--from-checkpoint", type=Path, default=None)
    parser.add_argument("--since-date", default=None, help="UTC date/time filter, for example 2026-06-24")
    parser.add_argument("--use-all-data", action="store_true")
    parser.add_argument("--feature-schema-version", default=CURRENT_FEATURE_SCHEMA_VERSION)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    args = parser.parse_args()
    path = train_price_model(
        args.dataset,
        from_checkpoint=args.from_checkpoint,
        since_date=parse_since_date(args.since_date),
        use_all_data=args.use_all_data,
        feature_schema_version=args.feature_schema_version,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    print(f"model={path}")


if __name__ == "__main__":
    main()
