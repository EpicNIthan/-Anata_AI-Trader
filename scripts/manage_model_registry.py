"""Operate the paper-only model registry from a deployment shell.

This command never imports an exchange client and never submits an order.  It is a
thin transaction wrapper around :class:`app.pipeline.registry.ModelRegistry`, so
artifact compatibility and lifecycle transition checks remain centralized.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def _json_object(path: Path | None, *, label: str) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain a JSON object")
    return dict(payload)


def _columns_from_payload(payload: Any) -> list[str]:
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, Mapping):
        values = None
        for key in ("feature_columns", "required_features", "features", "columns"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                values = candidate
                break
        if values is None:
            return []
    else:
        return []
    columns = [str(value).strip() for value in values if str(value).strip()]
    if len(columns) != len(set(columns)):
        raise ValueError("feature column contract contains duplicates")
    return columns


def _artifact_contract(path: Path) -> dict[str, Any]:
    """Read only declarative metadata; executable artifacts are loaded by the registry."""

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("JSON model artifact must contain an object")
        return dict(payload)
    if path.suffix.lower() != ".zip":
        return {}
    contract: dict[str, Any] = {}
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        for member in ("model_metadata.json", "feature_schema.json", "required_features.json"):
            if member not in names:
                continue
            payload = json.loads(archive.read(member).decode("utf-8"))
            if member == "model_metadata.json" and isinstance(payload, Mapping):
                contract.update(payload)
            elif member == "feature_schema.json" and isinstance(payload, Mapping):
                contract.update({key: value for key, value in payload.items() if key not in contract})
            elif member == "required_features.json":
                columns = _columns_from_payload(payload)
                if columns:
                    contract["feature_columns"] = columns
    return contract


def _feature_columns(args: argparse.Namespace, contract: Mapping[str, Any]) -> list[str]:
    if args.features:
        columns = [value.strip() for value in args.features.split(",") if value.strip()]
    elif args.features_file:
        payload = json.loads(args.features_file.read_text(encoding="utf-8"))
        columns = _columns_from_payload(payload)
    else:
        columns = _columns_from_payload(contract)
    if not columns:
        raise ValueError(
            "No feature columns found. Pass --features/--features-file or include feature_columns in the artifact."
        )
    if len(columns) != len(set(columns)):
        raise ValueError("feature column contract contains duplicates")
    return columns


def _identifier(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-.")
    if not normalized:
        raise ValueError("model identifier cannot be empty")
    return normalized[:128]


def _as_json(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def _model_payload(row: Any) -> dict[str, Any]:
    return {
        "id": row.id,
        "model_id": row.model_id,
        "name": row.name,
        "version": row.version,
        "model_family": row.model_family,
        "lifecycle_state": row.lifecycle_state,
        "health_status": row.health_status,
        "feature_schema_version": row.feature_schema_version,
        "feature_columns": list(row.feature_columns or []),
        "artifact_checksum": row.artifact_checksum,
        "path": row.path,
    }


def _add_common_model_id(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-version-id", required=True, type=int, help="Database id from register-challenger.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage challenger, shadow, sandbox, and champion state for paper trading only."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    register = subparsers.add_parser("register-challenger", help="Register a frozen forecast-only artifact.")
    register.add_argument("--artifact", required=True, type=Path)
    register.add_argument("--name", required=True)
    register.add_argument("--model-id", help="Stable model id; defaults to <name>-<version>.")
    register.add_argument("--version", required=True)
    register.add_argument("--model-family", required=True)
    register.add_argument("--feature-schema-version", help="May be inferred from artifact metadata.")
    feature_group = register.add_mutually_exclusive_group()
    feature_group.add_argument("--features", help="Ordered, comma-separated feature columns.")
    feature_group.add_argument("--features-file", type=Path, help="JSON list or object containing required features.")
    register.add_argument("--preprocessing-version", help="Defaults to artifact metadata or raw-linear-v1.")
    register.add_argument("--dataset-version", help="Immutable training dataset version/fingerprint.")
    register.add_argument("--forecast-horizon-seconds", type=int, help="Defaults to artifact metadata or 300.")
    register.add_argument("--parent-model-id")
    register.add_argument("--metrics", type=Path, help="Optional JSON object of offline metrics.")

    shadow = subparsers.add_parser("start-shadow", help="Run predictions without creating target exposure.")
    _add_common_model_id(shadow)

    sandbox = subparsers.add_parser("start-sandbox", help="Create an isolated fake-money account.")
    _add_common_model_id(sandbox)
    sandbox.add_argument("--name")
    sandbox.add_argument("--starting-balance", type=float)

    promote = subparsers.add_parser("promote", help="Explicitly assign a technically valid paper champion.")
    _add_common_model_id(promote)
    promote.add_argument("--model-family", required=True)
    promote.add_argument("--symbol-scope", default="*")
    promote.add_argument("--actor", default="manual")
    promote.add_argument("--reason", required=True, help="Auditable manual promotion rationale.")

    rollback = subparsers.add_parser("rollback", help="Restore the previously recorded paper champion.")
    rollback.add_argument("--model-family", required=True)
    rollback.add_argument("--symbol-scope", default="*")
    rollback.add_argument("--actor", default="manual")
    rollback.add_argument("--reason", required=True, help="Auditable rollback rationale.")
    return parser


def _register(args: argparse.Namespace, registry: Any) -> dict[str, Any]:
    artifact = args.artifact.expanduser().resolve()
    if not artifact.is_file():
        raise ValueError(f"artifact does not exist: {artifact}")
    contract = _artifact_contract(artifact)
    columns = _feature_columns(args, contract)
    schema_version = args.feature_schema_version or contract.get("feature_schema_version") or contract.get("schema_version")
    if not schema_version:
        raise ValueError("feature schema version is required (--feature-schema-version or artifact metadata)")
    horizon = args.forecast_horizon_seconds or contract.get("forecast_horizon_seconds") or contract.get("forecast_horizon") or 300
    try:
        horizon = int(horizon)
    except (TypeError, ValueError) as exc:
        raise ValueError("forecast horizon must be an integer number of seconds") from exc
    if horizon <= 0:
        raise ValueError("forecast horizon must be positive")
    metrics = _json_object(args.metrics, label="metrics")
    model_id = _identifier(args.model_id or f"{args.name}-{args.version}")
    from app.pipeline.domain import ModelLifecycle

    row = registry.register(
        name=args.name.strip(),
        model_id=model_id,
        version=args.version.strip(),
        model_family=args.model_family.strip(),
        path=str(artifact),
        feature_schema_version=str(schema_version),
        feature_columns=columns,
        lifecycle=ModelLifecycle.TRAINED,
        metrics=metrics or dict(contract.get("metrics") or {}),
        preprocessing_version=args.preprocessing_version or str(contract.get("preprocessing_version") or "raw-linear-v1"),
        training_dataset_version=args.dataset_version or contract.get("training_dataset_version") or contract.get("dataset_version"),
        forecast_horizon_seconds=horizon,
        parent_model_id=args.parent_model_id,
    )
    return {"action": "REGISTER_CHALLENGER", "model": _model_payload(row)}


def execute(args: argparse.Namespace) -> dict[str, Any]:
    """Execute one registry transaction. Kept separate for focused CLI tests."""

    from app.db.session import SessionLocal, create_db_and_tables
    from app.pipeline.registry import ModelRegistry

    create_db_and_tables()
    with SessionLocal() as session:
        registry = ModelRegistry(session)
        try:
            if args.command == "register-challenger":
                result = _register(args, registry)
            elif args.command == "start-shadow":
                row = registry.start_shadow(args.model_version_id)
                result = {"action": "START_SHADOW", "model": _model_payload(row)}
            elif args.command == "start-sandbox":
                if args.starting_balance is not None and args.starting_balance <= 0:
                    raise ValueError("starting balance must be positive")
                account = registry.start_sandbox(
                    args.model_version_id,
                    name=args.name,
                    starting_balance=args.starting_balance,
                )
                result = {
                    "action": "START_SANDBOX",
                    "sandbox": {
                        "account_id": account.account_id,
                        "name": account.name,
                        "model_version_id": account.model_version_id,
                        "starting_balance": account.starting_balance,
                        "max_exposure_pct": account.max_exposure_pct,
                        "active": account.active,
                    },
                }
            elif args.command == "promote":
                row = registry.promote(
                    args.model_version_id,
                    model_family=args.model_family.strip(),
                    symbol_scope=args.symbol_scope.upper(),
                    actor=args.actor,
                    reason=args.reason,
                )
                result = {"action": "PROMOTE", "model": _model_payload(row), "symbol_scope": args.symbol_scope.upper()}
            elif args.command == "rollback":
                row = registry.rollback(
                    model_family=args.model_family.strip(),
                    symbol_scope=args.symbol_scope.upper(),
                    actor=args.actor,
                    reason=args.reason,
                )
                result = {"action": "ROLLBACK", "model": _model_payload(row), "symbol_scope": args.symbol_scope.upper()}
            else:  # pragma: no cover - argparse prevents this branch.
                raise ValueError(f"unknown command: {args.command}")
            session.commit()
        except Exception:
            session.rollback()
            raise
    return {**result, "safety": "paper-only; no exchange order path is available to this command"}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = execute(args)
    except (OSError, ValueError, PermissionError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        print(json.dumps({"status": "error", "error": str(exc), "paper_only": True}, indent=2), file=sys.stderr)
        return 2
    print(json.dumps({"status": "ok", **result}, indent=2, default=_as_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
