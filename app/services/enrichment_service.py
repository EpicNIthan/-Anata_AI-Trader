"""Bounded background news enrichment using local intelligence first."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from app.config import settings
from app.db.session import SessionLocal
from app.intelligence.persistence import build_intelligence_router, enrich_recent_articles

logger = logging.getLogger(__name__)


@dataclass
class EnrichmentState:
    running: bool = False
    last_started_at: str | None = None
    last_finished_at: str | None = None
    last_error: str | None = None
    rows_processed: int = 0


class EnrichmentService:
    """Periodically create structured local news events.

    External providers remain optional inside the intelligence router. A timeout,
    invalid response, quota exhaustion, or disabled provider still persists the local
    deterministic/student event and cannot stop the paper-trading loop.
    """

    def __init__(self, *, interval_seconds: int | None = None, batch_size: int | None = None) -> None:
        self.interval_seconds = max(int(interval_seconds or settings.enrichment_interval_seconds), 15)
        self.batch_size = min(max(int(batch_size or settings.enrichment_batch_size), 1), 250)
        self.state = EnrichmentState()
        # Re-resolve the manually active DB artifact for each bounded pass so a
        # separate Railway enrichment role observes explicit activations without a
        # process restart. Persistent quota/circuit state is hydrated from the DB.
        self.router = None
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop.clear()
        self.state.running = True
        self._task = asyncio.create_task(self._run(), name="news-enrichment")

    async def stop(self) -> None:
        self._stop.set()
        if self._task and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=5)
            except asyncio.TimeoutError:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
        self.state.running = False

    async def run_once(self) -> dict[str, Any]:
        self.state.last_started_at = datetime.now(timezone.utc).isoformat()
        with SessionLocal() as session:
            try:
                active_router = build_intelligence_router(session)
                rows = await enrich_recent_articles(session, limit=self.batch_size, router=active_router)
                session.commit()
                self.state.rows_processed += len(rows)
                self.state.last_error = None
            except Exception as exc:  # The external/local provider boundary must not terminate the worker.
                session.rollback()
                self.state.last_error = f"{type(exc).__name__}: {exc}"
                logger.exception("News enrichment pass failed safely")
                rows = []
        self.state.last_finished_at = datetime.now(timezone.utc).isoformat()
        return {**asdict(self.state), "rows_in_pass": len(rows)}

    def status(self) -> dict[str, Any]:
        return asdict(self.state)

    async def _run(self) -> None:
        try:
            while not self._stop.is_set():
                await self.run_once()
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self.interval_seconds)
                except asyncio.TimeoutError:
                    continue
        finally:
            self.state.running = False
