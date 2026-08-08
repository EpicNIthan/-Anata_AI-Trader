from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db.models import Base
from app.db.session import get_session
from app.security import require_admin
from app.services import dataset_handoff as handoff
from app.services.handoff_stream import (
    handoff_download_status,
    prepare_stream_plan,
    stream_dataset_archive,
)
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


handoff._dataset_specs = _exact_dataset_specs


def _stream_response(archive_id: str) -> StreamingResponse:
    safe_name = archive_id.replace('"', "")
    return StreamingResponse(
        stream_dataset_archive(archive_id),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}"',
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/status")
def status(session: Session = Depends(get_session)) -> dict:
    return handoff_download_status(session)


@router.post("/prepare")
def prepare(session: Session = Depends(get_session)) -> dict:
    try:
        return prepare_stream_plan(session)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Dataset preparation failed: {type(exc).__name__}: {exc}") from exc


@router.get("/download-latest")
def download_latest(session: Session = Depends(get_session)) -> StreamingResponse:
    try:
        plan = prepare_stream_plan(session)
        if plan.get("status") == "empty":
            raise HTTPException(status_code=404, detail=plan.get("message") or "No dataset is available yet")
        return _stream_response(str(plan["archive_id"]))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Dataset download failed: {type(exc).__name__}: {exc}") from exc


@router.get("/download/{archive_id}")
def download(archive_id: str, session: Session = Depends(get_session)) -> StreamingResponse:
    state = handoff._state(session)
    if state.latest_archive_id != archive_id or state.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Prepared dataset archive not found")
    return _stream_response(archive_id)


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
