from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.db.models import Base
from app.db.session import get_session
from app.security import require_admin
from app.services import dataset_handoff as handoff
from app.services.strategy_history_bootstrap import ensure_strategy_history

router = APIRouter(
    prefix="/api/data/handoff",
    tags=["dataset-handoff"],
    dependencies=[Depends(require_admin)],
)


def _exact_dataset_specs() -> list[tuple[str, Any, Any]]:
    """Use the same timestamp for core export and deletion, then generic event time for extended ledgers."""
    core_times = {spec.name: spec.time_column for spec in handoff.RAW_TABLES}
    specs: list[tuple[str, Any, Any]] = []
    seen: set[str] = set()
    for mapper in Base.registry.mappers:
        model = mapper.class_
        table_name = getattr(model, "__tablename__", None)
        if not table_name or table_name == handoff.DatasetHandoffState.__tablename__ or table_name in seen:
            continue
        chosen = core_times.get(table_name)
        if chosen is None:
            for name in handoff.TIME_COLUMNS:
                candidate = getattr(model, name, None)
                if candidate is not None:
                    chosen = candidate
                    break
        if chosen is None:
            continue
        specs.append((table_name, model, chosen))
        seen.add(table_name)
    return sorted(specs, key=lambda item: item[0])


# All handoff service functions resolve this global at call time. Replacing it here
# keeps the export and delete boundaries identical for legacy/core dataset tables.
handoff._dataset_specs = _exact_dataset_specs


@router.get("/status")
def status(session: Session = Depends(get_session)) -> dict:
    return handoff.handoff_status(session)


@router.post("/prepare")
def prepare(session: Session = Depends(get_session)) -> dict:
    try:
        return handoff.prepare_handoff(session)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Dataset preparation failed: {type(exc).__name__}: {exc}") from exc


@router.get("/download/{archive_id}")
def download(archive_id: str, session: Session = Depends(get_session)) -> FileResponse:
    try:
        path = handoff.mark_download_requested(session, archive_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Prepared dataset archive not found") from exc
    return FileResponse(path, filename=path.name, media_type="application/zip")


@router.post("/delete-latest")
async def delete_latest(session: Session = Depends(get_session)) -> dict:
    try:
        deleted = handoff.delete_latest_handoff(session)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Dataset deletion failed: {type(exc).__name__}: {exc}") from exc

    history = await ensure_strategy_history()
    return {**deleted, "strategy_history": history, "paper_only": True}
