from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.services import dataset_handoff as handoff

CHUNK_SIZE = 1024 * 1024


def prepare_archive_path(session: Session, archive_id: str) -> Path:
    """Validate a prepared archive and clear any stale completion marker before a new transfer."""
    state = handoff._state(session)
    if state.latest_archive_id != archive_id or state.deleted_at is not None:
        raise FileNotFoundError(archive_id)
    path = handoff.HANDOFF_ROOT / Path(archive_id).name
    if path.name != archive_id or not path.exists() or not path.is_file():
        raise FileNotFoundError(archive_id)
    state.download_requested_at = None
    session.commit()
    return path


def stream_archive(archive_id: str) -> Iterator[bytes]:
    """Stream the already-built ZIP and mark completion only after every byte was yielded."""
    path = handoff.HANDOFF_ROOT / Path(archive_id).name
    completed = False
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(CHUNK_SIZE)
                if not chunk:
                    break
                yield chunk
        completed = True
    finally:
        if completed:
            with SessionLocal() as session:
                state = handoff._state(session)
                if state.latest_archive_id == archive_id and state.deleted_at is None:
                    state.download_requested_at = datetime.now(timezone.utc)
                    session.commit()


def download_status(session: Session) -> dict[str, object]:
    data = handoff.handoff_status(session)
    completed_at = data.get("download_requested_at")
    data["download_completed_at"] = completed_at
    data["download_ready_to_delete"] = bool(completed_at and not data.get("deleted_at"))
    return data
