from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends
from sqlalchemy.orm import Session

from app.db.session import get_session
from app.security import require_admin
from app.services.daily_raw_export import create_daily_raw_data_archive
from app.services.raw_data_cleanup import cleanup_downloaded_raw_data

router = APIRouter(prefix="/api/raw-data", tags=["raw-data"], dependencies=[Depends(require_admin)])


@router.post("/export-daily")
def export_daily_raw_data_api(
    payload: dict[str, Any] | None = Body(default=None),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    return create_daily_raw_data_archive(session, payload or {})


@router.post("/cleanup-downloaded")
def raw_data_cleanup_after_verified_download(
    payload: dict[str, Any] | None = Body(default=None),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    return cleanup_downloaded_raw_data(session, payload or {})
