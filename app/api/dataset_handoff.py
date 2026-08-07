from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.db.session import get_session
from app.security import require_admin
from app.services.dataset_handoff import (
    delete_latest_handoff,
    handoff_status,
    mark_download_requested,
    prepare_handoff,
)
from app.services.strategy_history_bootstrap import ensure_strategy_history

router = APIRouter(
    prefix="/api/data/handoff",
    tags=["dataset-handoff"],
    dependencies=[Depends(require_admin)],
)


@router.get("/status")
def status(session: Session = Depends(get_session)) -> dict:
    return handoff_status(session)


@router.post("/prepare")
def prepare(session: Session = Depends(get_session)) -> dict:
    try:
        return prepare_handoff(session)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Dataset preparation failed: {type(exc).__name__}: {exc}") from exc


@router.get("/download/{archive_id}")
def download(archive_id: str, session: Session = Depends(get_session)) -> FileResponse:
    try:
        path = mark_download_requested(session, archive_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Prepared dataset archive not found") from exc
    return FileResponse(path, filename=path.name, media_type="application/zip")


@router.post("/delete-latest")
async def delete_latest(session: Session = Depends(get_session)) -> dict:
    try:
        deleted = delete_latest_handoff(session)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Dataset deletion failed: {type(exc).__name__}: {exc}") from exc

    history = await ensure_strategy_history()
    return {**deleted, "strategy_history": history, "paper_only": True}
