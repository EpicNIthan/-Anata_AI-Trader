"""Durable model-artifact bytes, integrity verification, and runtime materialization."""

from __future__ import annotations

import hashlib
import io
import json
import mimetypes
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Mapping
from uuid import uuid4
import zipfile

from sqlalchemy import select
from sqlalchemy.orm import Session, object_session

from app.db.models import ModelArtifactBlob, ModelVersion


class ArtifactIntegrityError(RuntimeError):
    """Raised when durable or local artifact bytes fail their declared contract."""


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _normalized_checksum(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower().removeprefix("sha256:")
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ArtifactIntegrityError("ARTIFACT_CHECKSUM_INVALID")
    return normalized


def _safe_package_member(name: str) -> str:
    if not name or "\\" in name:
        raise ArtifactIntegrityError("PACKAGE_MEMBER_PATH_UNSAFE")
    member = PurePosixPath(name)
    if member.is_absolute() or ".." in member.parts:
        raise ArtifactIntegrityError("PACKAGE_MEMBER_PATH_UNSAFE")
    return name


def verify_package_checksum_manifest(
    content: bytes,
    *,
    require_manifest: bool,
) -> dict[str, Any] | None:
    """Verify every non-directory ZIP member against ``checksum_manifest.json``."""

    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            names = [info.filename for info in archive.infolist() if not info.is_dir()]
            if len(names) != len(set(names)):
                raise ArtifactIntegrityError("PACKAGE_DUPLICATE_MEMBER")
            for name in names:
                _safe_package_member(name)
            if "checksum_manifest.json" not in names:
                if require_manifest:
                    raise ArtifactIntegrityError("PACKAGE_CHECKSUM_MANIFEST_MISSING")
                return None
            try:
                manifest = json.loads(archive.read("checksum_manifest.json").decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ArtifactIntegrityError("PACKAGE_CHECKSUM_MANIFEST_INVALID") from exc
            files = manifest.get("files") if isinstance(manifest, Mapping) else None
            if not isinstance(files, Mapping):
                raise ArtifactIntegrityError("PACKAGE_CHECKSUM_MANIFEST_INVALID")
            declared = {str(name) for name in files}
            actual = set(names) - {"checksum_manifest.json"}
            if declared != actual:
                raise ArtifactIntegrityError("PACKAGE_CHECKSUM_MANIFEST_COVERAGE_MISMATCH")
            for name, descriptor in files.items():
                member_name = _safe_package_member(str(name))
                if not isinstance(descriptor, Mapping):
                    raise ArtifactIntegrityError("PACKAGE_CHECKSUM_ENTRY_INVALID")
                expected = _normalized_checksum(str(descriptor.get("sha256") or ""))
                member_content = archive.read(member_name)
                if sha256_bytes(member_content) != expected:
                    raise ArtifactIntegrityError("PACKAGE_MEMBER_CHECKSUM_MISMATCH")
                declared_size = descriptor.get("bytes")
                if declared_size is not None:
                    try:
                        size = int(declared_size)
                    except (TypeError, ValueError) as exc:
                        raise ArtifactIntegrityError("PACKAGE_CHECKSUM_ENTRY_INVALID") from exc
                    if size != len(member_content):
                        raise ArtifactIntegrityError("PACKAGE_MEMBER_SIZE_MISMATCH")
            return dict(manifest)
    except ArtifactIntegrityError:
        raise
    except (OSError, zipfile.BadZipFile) as exc:
        raise ArtifactIntegrityError("ARTIFACT_ZIP_INVALID") from exc


def verify_artifact_bytes(
    content: bytes,
    *,
    filename: str,
    expected_checksum: str | None,
    require_package_manifest: bool,
) -> dict[str, Any]:
    """Verify whole-artifact identity and, for packages, every manifest member."""

    checksum = sha256_bytes(content)
    expected = _normalized_checksum(expected_checksum)
    if expected is not None and checksum != expected:
        raise ArtifactIntegrityError("ARTIFACT_CHECKSUM_MISMATCH")
    package_manifest = None
    if Path(filename).suffix.lower() == ".zip":
        package_manifest = verify_package_checksum_manifest(
            content,
            require_manifest=require_package_manifest,
        )
    return {
        "sha256": checksum,
        "size_bytes": len(content),
        "package_checksum_manifest": package_manifest,
    }


def _requires_package_manifest(model: ModelVersion, filename: str) -> bool:
    if Path(filename).suffix.lower() != ".zip":
        return False
    manifest = model.package_manifest if isinstance(model.package_manifest, Mapping) else {}
    return not bool(manifest.get("legacy") or manifest.get("legacy_upload"))


class ModelArtifactStore:
    """Persist immutable artifact bytes in the caller's SQLAlchemy transaction."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def put_path(self, model: ModelVersion, path: str | Path) -> ModelArtifactBlob:
        target = Path(path)
        if not target.is_file():
            raise ArtifactIntegrityError("ARTIFACT_NOT_FOUND")
        return self.put_bytes(model, target.read_bytes(), filename=target.name)

    def put_bytes(
        self,
        model: ModelVersion,
        content: bytes,
        *,
        filename: str,
        media_type: str | None = None,
    ) -> ModelArtifactBlob:
        if model.id is None:
            self.session.flush()
        safe_filename = Path(filename).name or "artifact.bin"
        verification = verify_artifact_bytes(
            bytes(content),
            filename=safe_filename,
            expected_checksum=model.artifact_checksum,
            require_package_manifest=_requires_package_manifest(model, safe_filename),
        )
        if model.artifact_checksum is None:
            model.artifact_checksum = str(verification["sha256"])
        existing = self.session.scalar(
            select(ModelArtifactBlob)
            .where(ModelArtifactBlob.model_version_id == model.id)
            .limit(1)
        )
        if existing is not None:
            if (
                existing.sha256 != verification["sha256"]
                or existing.size_bytes != len(content)
                or bytes(existing.content) != bytes(content)
            ):
                raise ArtifactIntegrityError("MODEL_ARTIFACT_BLOB_CONFLICT")
            return existing
        blob = ModelArtifactBlob(
            model_version_id=int(model.id),
            sha256=str(verification["sha256"]),
            filename=safe_filename,
            media_type=media_type or mimetypes.guess_type(safe_filename)[0] or "application/octet-stream",
            size_bytes=len(content),
            content=bytes(content),
        )
        self.session.add(blob)
        self.session.flush()
        return blob

    def get(self, model: ModelVersion) -> ModelArtifactBlob | None:
        if model.id is None:
            return None
        return self.session.scalar(
            select(ModelArtifactBlob)
            .where(ModelArtifactBlob.model_version_id == model.id)
            .limit(1)
        )


def resolve_model_artifact(
    model: ModelVersion,
    *,
    session: Session | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    """Return verified local bytes, materializing the DB copy after path loss."""

    origin = Path(str(model.path or ""))
    if origin.is_file():
        content = origin.read_bytes()
        verify_artifact_bytes(
            content,
            filename=origin.name,
            expected_checksum=model.artifact_checksum,
            require_package_manifest=_requires_package_manifest(model, origin.name),
        )
        return origin

    active_session = session or object_session(model)
    if active_session is None:
        raise ArtifactIntegrityError("DURABLE_ARTIFACT_SESSION_UNAVAILABLE")
    blob = ModelArtifactStore(active_session).get(model)
    if blob is None:
        raise ArtifactIntegrityError("DURABLE_ARTIFACT_NOT_FOUND")
    if model.artifact_checksum is None or blob.sha256 != _normalized_checksum(model.artifact_checksum):
        raise ArtifactIntegrityError("DURABLE_ARTIFACT_IDENTITY_MISMATCH")
    content = bytes(blob.content)
    verification = verify_artifact_bytes(
        content,
        filename=blob.filename,
        expected_checksum=model.artifact_checksum,
        require_package_manifest=_requires_package_manifest(model, blob.filename),
    )
    if blob.size_bytes != len(content) or blob.sha256 != verification["sha256"]:
        raise ArtifactIntegrityError("DURABLE_ARTIFACT_METADATA_MISMATCH")

    root = Path(cache_dir) if cache_dir is not None else Path(tempfile.gettempdir()) / "anata-model-artifacts"
    root.mkdir(parents=True, exist_ok=True)
    suffix = Path(blob.filename).suffix.lower() or ".bin"
    target = root / f"model-{model.id}-{blob.sha256}{suffix}"
    if target.is_file():
        try:
            verify_artifact_bytes(
                target.read_bytes(),
                filename=target.name,
                expected_checksum=model.artifact_checksum,
                require_package_manifest=_requires_package_manifest(model, target.name),
            )
            return target
        except ArtifactIntegrityError:
            pass
    temporary = root / f".{target.name}.{uuid4().hex}.tmp"
    temporary.write_bytes(content)
    temporary.replace(target)
    return target
