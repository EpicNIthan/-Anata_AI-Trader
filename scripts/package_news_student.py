"""Package a validated local student with compatibility metadata and checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from news_student_utils import sha256_file

from app.intelligence.providers import validate_json_student_artifact


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a portable, checksummed local-news student package.")
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true", help="Allow an existing *empty* output directory.")
    args = parser.parse_args()
    if not args.artifact.is_file():
        raise SystemExit(f"Artifact does not exist: {args.artifact}")
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    validate_json_student_artifact(artifact)
    output_dir = args.output_dir
    if output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Refusing to write into existing {output_dir}; choose a fresh directory or pass --overwrite for an empty one.")
        if not output_dir.is_dir():
            raise SystemExit(f"Output path is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise SystemExit("Refusing to delete or replace an existing package. Choose a new empty --output-dir.")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    artifact_name = "student_artifact.json"
    shutil.copy2(args.artifact, output_dir / artifact_name)
    metadata = {
        "model_id": f"local-news-student:{artifact['version']}",
        "name": "local-news-student",
        "version": artifact["version"],
        "model_family": "news_student_naive_bayes",
        "artifact": artifact_name,
        "artifact_type": artifact["artifact_type"],
        "teacher_versions": artifact.get("teacher_versions", []),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "paper_only": True,
        "execution_capability": False,
    }
    files: dict[str, Any] = {
        "feature_schema.json": {"version": "news-student-text-v1", "features": [{"name": "text", "type": "string"}]},
        "model_metadata.json": metadata,
        "training_metrics.json": {"training_rows": artifact.get("training_rows"), "note": "Run evaluate_news_student.py for held-out imitation metrics."},
        "training_period.json": artifact.get("training_period", {}),
        "required_features.json": ["text"],
        "optional_features.json": ["title", "source", "published_at", "available_to_model_at"],
        "missing_value_policy.json": {"text": "required_nonempty", "optional_fields": "explicitly_missing_not_neutral"},
        "news_student_version.json": {"version": artifact["version"], "teacher_versions": artifact.get("teacher_versions", [])},
    }
    for filename, contents in files.items():
        _write_json(output_dir / filename, contents)
    checksums = {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    manifest = {
        "package_type": "anata_local_news_student_package_v1",
        "version": artifact["version"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": checksums,
        "paper_only": True,
        "activation": "manual",
    }
    _write_json(output_dir / "checksum_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "packaged",
                "package_dir": str(output_dir),
                "version": artifact["version"],
                "files": len(manifest["files"]) + 1,
                "activation": "manual; no model was uploaded or activated",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
