from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from app.db.session import SessionLocal
from app.features.schema import CURRENT_FEATURE_SCHEMA_VERSION
from app.training.label_builder import build_labels_for_existing_features, label_status_fast

logger = logging.getLogger(__name__)


@dataclass
class LabelMaintenanceState:
    running: bool = False
    interval_seconds: int = 900
    last_run_at: str | None = None
    last_result: dict[str, Any] | None = None
    last_status: dict[str, Any] | None = None
    last_error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class LabelMaintenanceService:
    """Build training labels automatically as future closed candles become available."""

    def __init__(self, *, interval_seconds: int = 900, batch_limit: int = 5_000) -> None:
        self.interval_seconds = max(interval_seconds, 60)
        self.batch_limit = max(batch_limit, 100)
        self.state = LabelMaintenanceState(interval_seconds=self.interval_seconds)
        self._task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None

    def status(self) -> dict[str, Any]:
        return self.state.as_dict()

    async def start(self) -> dict[str, Any]:
        if self._task and not self._task.done():
            return self.status()
        self._stop_event = asyncio.Event()
        self.state.running = True
        self.state.last_error = None
        self._task = asyncio.create_task(self._run_loop(self._stop_event), name="label-maintenance")
        return self.status()

    async def stop(self) -> dict[str, Any]:
        if self._stop_event:
            self._stop_event.set()
        if self._task and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=10)
            except asyncio.TimeoutError:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
        self.state.running = False
        return self.status()

    async def _run_loop(self, stop_event: asyncio.Event) -> None:
        try:
            while not stop_event.is_set():
                await asyncio.to_thread(self.run_once)
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=self.interval_seconds)
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive service boundary
            logger.exception("Label maintenance stopped after an unexpected error")
            self.state.last_error = f"{type(exc).__name__}: {exc}"
        finally:
            self.state.running = False

    def run_once(self) -> dict[str, Any]:
        with SessionLocal() as session:
            result = build_labels_for_existing_features(
                session,
                schema_version=CURRENT_FEATURE_SCHEMA_VERSION,
                limit=self.batch_limit,
                force=False,
                sync_features=True,
            )
            status = label_status_fast(session)
        self.state.last_run_at = datetime.now(timezone.utc).isoformat()
        self.state.last_result = result
        self.state.last_status = status
        self.state.last_error = None
        if result.get("labeled"):
            logger.info("Auto label maintenance labeled %s rows", result["labeled"])
        return self.status()