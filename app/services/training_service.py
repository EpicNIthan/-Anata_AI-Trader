from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import desc, select

from app.config import settings
from app.db.models import ModelVersion
from app.db.session import SessionLocal
from app.features.schema import CURRENT_FEATURE_SCHEMA_VERSION
from app.training.dataset_accelerator import build_accelerated_dataset
from app.training.export_dataset import parse_since_date

SERVER_TRAINING_DISABLED_MESSAGE = "Server training is disabled. Download dataset and train locally."


@dataclass
class TrainingState:
    running: bool = False
    status: str = "idle"
    started_at: str | None = None
    finished_at: str | None = None
    last_error: str | None = None
    last_result: dict[str, Any] | None = None
    latest_model: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class TrainingService:
    def __init__(self) -> None:
        self.state = TrainingState()
        self._task: asyncio.Task[None] | None = None

    def status(self) -> dict[str, Any]:
        return self.state.as_dict()

    async def start(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if not settings.enable_server_training:
            self.state.running = False
            self.state.status = "disabled"
            self.state.finished_at = datetime.now(timezone.utc).isoformat()
            self.state.last_error = None
            self.state.last_result = {
                "status": "disabled",
                "message": SERVER_TRAINING_DISABLED_MESSAGE,
                "allowed_server_actions": ["build_dataset", "export_dataset", "upload_model", "activate_model"],
            }
            return self.status()
        if self._task and not self._task.done():
            return self.status()
        self.state.running = True
        self.state.status = "running"
        self.state.started_at = datetime.now(timezone.utc).isoformat()
        self.state.finished_at = None
        self.state.last_error = None
        self._task = asyncio.create_task(self._run(payload or {}), name="training-service")
        return self.status()

    async def run_and_wait(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._task and not self._task.done():
            return self.status()
        await self.start(payload)
        if self._task:
            await self._task
        return self.status()

    async def _run(self, payload: dict[str, Any]) -> None:
        try:
            result = await train_model_job(payload)
            self.state.last_result = result
            self.state.latest_model = result.get("model")
            self.state.status = "finished"
        except Exception as exc:
            self.state.last_error = f"{type(exc).__name__}: {exc}"
            self.state.status = "failed"
        finally:
            self.state.running = False
            self.state.finished_at = datetime.now(timezone.utc).isoformat()


async def train_model_job(payload: dict[str, Any]) -> dict[str, Any]:
    if not settings.enable_server_training:
        return {
            "status": "disabled",
            "message": SERVER_TRAINING_DISABLED_MESSAGE,
            "allowed_server_actions": ["build_dataset", "export_dataset", "upload_model", "activate_model"],
        }
    build_dataset = bool(payload.get("build_dataset", False))
    dataset_path_value = payload.get("dataset_path")
    dataset_summary: dict[str, Any] | None = None

    if build_dataset:
        symbols = payload.get("symbols")
        if isinstance(symbols, str):
            symbols = [item.strip().upper() for item in symbols.split(",") if item.strip()]
        dataset_summary = await build_accelerated_dataset(
            symbols=symbols if isinstance(symbols, list) else None,
            interval=str(payload.get("interval") or settings.paper_trade_timeframe),
            days=min(max(int(payload.get("days") or 14), 1), 365),
            max_rows_per_symbol=min(max(int(payload.get("max_rows_per_symbol") or 5000), 100), 250_000),
            lookback=min(max(int(payload.get("lookback") or 60), 10), 500),
            stride=min(max(int(payload.get("stride") or 5), 1), 500),
            replay_limit=min(max(int(payload.get("replay_limit") or 20_000), 100), 500_000),
            backfill=bool(payload.get("backfill", True)),
            mock=bool(payload.get("mock", False)),
            export=True,
        )
        dataset_path_value = dataset_summary.get("exported_path")

    dataset_path = Path(dataset_path_value) if dataset_path_value else None
    from_checkpoint_value = payload.get("from_checkpoint")
    from_checkpoint: Path | None = None
    if from_checkpoint_value == "latest":
        with SessionLocal() as session:
            latest = session.scalar(
                select(ModelVersion)
                .where(ModelVersion.status.in_(["active", "candidate", "trained"]))
                .order_by(desc(ModelVersion.created_at))
                .limit(1)
            )
            from_checkpoint = Path(latest.path) if latest else None
    elif from_checkpoint_value:
        from_checkpoint = Path(str(from_checkpoint_value))

    feature_schema_version = str(payload.get("feature_schema_version") or CURRENT_FEATURE_SCHEMA_VERSION)
    since_date = parse_since_date(payload.get("since_date"))
    use_all_data = bool(payload.get("use_all_data", True))
    epochs = min(max(int(payload.get("epochs") or 500), 1), 20_000)
    learning_rate = float(payload.get("learning_rate") or 0.05)
    from app.training.train_price_model import train_price_model

    model_path = await asyncio.to_thread(
        train_price_model,
        dataset_path,
        from_checkpoint=from_checkpoint,
        since_date=since_date,
        use_all_data=use_all_data,
        feature_schema_version=feature_schema_version,
        epochs=epochs,
        learning_rate=learning_rate,
    )

    with SessionLocal() as session:
        model = session.scalar(select(ModelVersion).order_by(desc(ModelVersion.created_at)).limit(1))
        model_payload = {
            "id": model.id if model else None,
            "model_id": model.model_id if model else None,
            "name": model.name if model else None,
            "version": model.version if model else None,
            "feature_schema_version": model.feature_schema_version if model else feature_schema_version,
            "feature_columns": model.feature_columns if model else None,
            "metrics": model.metrics if model else None,
            "path": model.path if model else str(model_path),
            "status": model.status if model else "candidate",
            "created_at": model.created_at.isoformat() if model and model.created_at else None,
        }

    return {
        "status": "trained",
        "model_path": str(model_path),
        "dataset_path": str(dataset_path) if dataset_path else None,
        "dataset_summary": dataset_summary,
        "model": model_payload,
        "auto_trader_use_trained_model": settings.auto_trader_use_trained_model,
    }
