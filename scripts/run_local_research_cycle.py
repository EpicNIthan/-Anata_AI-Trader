"""Run the complete local narrow-model research cycle once.

The command detects newly available labels, searches functional specialist model
families, runs purged/embargoed walk-forward evaluation, packages one challenger per
family, and registers each package in the local model registry.  It deliberately has
no champion-promotion option.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.features.schema import model_input_columns_for_schema  # noqa: E402
from app.research.evaluation import ExecutionAssumptions  # noqa: E402
from app.research.schemas import persisted_research_id, stable_fingerprint  # noqa: E402
from app.research.training import (  # noqa: E402
    FAMILY_FEATURE_COLUMNS,
    discover_available_labeled_rows,
    run_narrow_research_cycle,
)
from scripts.research_utils import read_research_rows, sha256_file  # noqa: E402


def _atomic_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing report: {path}; pass --overwrite")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"contract_version": 1, "seen_labeled_row_ids": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"research state must contain an object: {path}")
    ids = payload.get("seen_labeled_row_ids", [])
    if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
        raise ValueError(f"research state has invalid seen_labeled_row_ids: {path}")
    return dict(payload)


def _write_state(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _schema_version(rows: Sequence[Mapping[str, Any]], requested: str | None) -> str:
    observed = {
        str(row.get("feature_schema_version") or row.get("schema_version") or "").strip()
        for row in rows
        if str(row.get("feature_schema_version") or row.get("schema_version") or "").strip()
    }
    if len(observed) > 1:
        raise ValueError("input contains multiple feature schema versions; run one cycle per schema")
    dataset_schema = next(iter(observed), None)
    if requested and dataset_schema and requested != dataset_schema:
        raise ValueError(
            f"requested feature schema {requested!r} does not match dataset schema {dataset_schema!r}"
        )
    schema = requested or dataset_schema
    if not schema:
        raise ValueError("feature schema version is missing; pass --feature-schema-version")
    return schema


def _csv_values(raw: str, *, label: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError(f"{label} cannot be empty")
    return values


def _alphas(raw: str) -> list[float]:
    try:
        values = [float(value) for value in _csv_values(raw, label="alphas")]
    except ValueError as exc:
        raise ValueError("alphas must be a comma-separated list of numbers") from exc
    if any(value < 0 for value in values):
        raise ValueError("alphas cannot be negative")
    return values


def _execution_assumptions(path: Path | None) -> ExecutionAssumptions:
    if path is None:
        return ExecutionAssumptions()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("execution assumptions JSON must contain an object")
    return ExecutionAssumptions.from_mapping(payload)


def _utc_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _aggregate_fold_period(
    fold_periods: list[dict[str, Any]],
    key: str,
) -> dict[str, Any] | None:
    """Return a compact envelope while retaining exact fold periods elsewhere."""

    periods = [
        dict(item[key])
        for item in fold_periods
        if isinstance(item.get(key), Mapping)
        and item[key].get("start")
        and item[key].get("end")
    ]
    if not periods:
        return None
    return {
        "start": min(str(period["start"]) for period in periods),
        "end": max(str(period["end"]) for period in periods),
        "fold_count": len(periods),
        "semantics": "aggregate envelope; exact chronological boundaries are in configuration.fold_periods",
    }


def persist_research_cycle(
    session: Any,
    result: Mapping[str, Any],
    *,
    register_models: bool = True,
) -> dict[str, Any]:
    """Idempotently persist one cycle and its TRAINED packages in one DB transaction."""

    from sqlalchemy import select

    from app.db.models import CandidateEvaluation, ExperimentRun, ModelVersion, StrategyCandidate
    from app.pipeline.domain import ModelLifecycle
    from app.pipeline.registry import ModelRegistry

    registry = ModelRegistry(session)
    registrations: list[dict[str, Any]] = []
    packages = list(result.get("challenger_packages") or [])
    if register_models:
        for package in packages:
            model_id = str(package["model_id"])
            version = str(package["version"])
            checksum = str(package["sha256"])
            row = session.scalar(
                select(ModelVersion)
                .where(ModelVersion.model_id == model_id, ModelVersion.version == version)
                .order_by(ModelVersion.id.desc())
                .limit(1)
            )
            reused = row is not None
            if row is not None and row.artifact_checksum != checksum:
                raise ValueError(
                    f"registry already has {model_id} {version} with a different artifact checksum"
                )
            if row is None:
                row = registry.register(
                    name=str(package["name"]),
                    model_id=model_id,
                    version=version,
                    model_family=str(package["model_family"]),
                    path=str(package["path"]),
                    feature_schema_version=str(package["feature_schema_version"]),
                    feature_columns=[str(item) for item in package["feature_columns"]],
                    lifecycle=ModelLifecycle.TRAINED,
                    metrics=dict(package.get("metrics") or {}),
                    preprocessing_version=str(package["preprocessing_version"]),
                    training_dataset_version=str(package["training_dataset_version"]),
                    forecast_horizon_seconds=int(package["forecast_horizon_seconds"]),
                )
            else:
                # Backfill durable bytes for a pre-store registry row; idempotent if
                # this model version already owns the same immutable blob.
                registry.store_artifact(row, str(package["path"]))
            period = package.get("training_period") if isinstance(package.get("training_period"), Mapping) else {}
            row.training_start_at = row.training_start_at or _utc_datetime(period.get("start"))
            row.training_end_at = row.training_end_at or _utc_datetime(period.get("end"))
            registrations.append(
                {
                    "model_version_id": row.id,
                    "model_id": row.model_id,
                    "version": row.version,
                    "model_family": row.model_family,
                    "lifecycle_state": row.lifecycle_state,
                    "artifact_checksum": row.artifact_checksum,
                    "reused": reused,
                    "promoted": False,
                }
            )

    candidates_persisted: list[dict[str, Any]] = []
    dataset_version = str(result.get("dataset_version") or "unknown")
    feature_version = str(result.get("feature_schema_version") or "unknown")
    target_name = str(result.get("target_name") or "actual_return")
    horizon = int(result.get("forecast_horizon_seconds") or 300)
    walk_forward = dict(result.get("walk_forward") or {})
    execution_assumptions = dict(result.get("execution_assumptions") or {})
    package_by_candidate = {str(item.get("candidate_id")): item for item in packages}
    for evaluation in result.get("candidates") or []:
        original_candidate_id = str(evaluation["candidate_id"])
        candidate_id = persisted_research_id(original_candidate_id, field_name="candidate_id")
        candidate = session.scalar(
            select(StrategyCandidate).where(StrategyCandidate.candidate_id == candidate_id).limit(1)
        )
        selected = bool(evaluation.get("selected_for_challenger"))
        if candidate is None:
            candidate = StrategyCandidate(candidate_id=candidate_id)
            session.add(candidate)
        candidate.model_family = str(evaluation.get("model_family") or "unknown")
        candidate.feature_families = [candidate.model_family]
        candidate.target_name = target_name
        candidate.forecast_horizon_seconds = horizon
        candidate.hyperparameters = {
            "algorithm": evaluation.get("algorithm"),
            "alpha": evaluation.get("alpha"),
            "huber_delta": evaluation.get("huber_delta"),
        }
        candidate.regime_filter = None
        candidate.signal_threshold = 0.0
        candidate.confidence_policy = {"name": "bounded"}
        candidate.cost_model = str(execution_assumptions.get("version") or "recorded_cost")
        candidate.portfolio_policy = "evaluation_only"
        candidate.exit_policy_name = "forecast_horizon"
        candidate.training_window = str(walk_forward.get("train_size") or "unknown")
        candidate.validation_window = str(walk_forward.get("validation_size") or "unknown")
        candidate.random_seed = 0
        candidate.hypothesis = f"Evaluate {candidate.model_family} as a forecast-only narrow challenger."
        candidate.lifecycle_state = "TRAINED" if selected else "VALIDATED"
        candidate.payload = {
            "original_candidate_id": original_candidate_id,
            "candidate_fingerprint": evaluation.get("candidate_fingerprint"),
            "feature_columns": list(evaluation.get("feature_columns") or []),
            "selected_for_challenger": selected,
            "paper_only": True,
            "automatic_promotion": False,
        }
        session.flush()

        experiment_fingerprint = stable_fingerprint(
            {
                "candidate_fingerprint": evaluation.get("candidate_fingerprint"),
                "dataset_version": dataset_version,
                "feature_version": feature_version,
                "target_name": target_name,
                "forecast_horizon_seconds": horizon,
                "walk_forward": walk_forward,
                "execution_assumptions": execution_assumptions,
            }
        )
        experiment_id = persisted_research_id(
            f"cycle-{candidate_id}-{experiment_fingerprint[:16]}",
            field_name="experiment_id",
        )
        experiment = session.scalar(
            select(ExperimentRun).where(ExperimentRun.experiment_id == experiment_id).limit(1)
        )
        if experiment is None:
            experiment = ExperimentRun(experiment_id=experiment_id)
            session.add(experiment)
        package = package_by_candidate.get(original_candidate_id)
        oos = evaluation.get("oos_observations") if isinstance(evaluation.get("oos_observations"), Mapping) else None
        metrics = {
            **dict(evaluation.get("metrics") or {}),
            "mean_absolute_error": evaluation.get("mean_absolute_error"),
            "root_mean_squared_error": evaluation.get("root_mean_squared_error"),
            "oos_observation_count": evaluation.get("oos_observation_count"),
            "selected_for_challenger": selected,
        }
        period = dict(result.get("data_period") or {})
        raw_folds = evaluation.get("folds") if isinstance(evaluation.get("folds"), list) else []
        fold_periods = [
            {
                "fold_number": fold.get("fold_number"),
                "train_period": dict(fold.get("train_period") or {}),
                "validation_period": dict(fold.get("validation_period") or {}),
                "test_period": dict(fold.get("test_period") or {}),
            }
            for fold in raw_folds
            if isinstance(fold, Mapping)
        ]
        experiment.code_version = "local-research-cycle-v1"
        experiment.configuration = {
            "candidate": dict(candidate.payload or {}),
            "walk_forward": walk_forward,
            "fold_periods": fold_periods,
            "execution_assumptions": execution_assumptions,
            "annualization_basis": result.get("annualization_basis"),
        }
        experiment.dataset_version = dataset_version
        experiment.feature_version = feature_version
        experiment.model_version = str(package.get("version")) if package else None
        experiment.random_seed = 0
        experiment.train_period = _aggregate_fold_period(fold_periods, "train_period") or period or None
        experiment.validation_period = _aggregate_fold_period(fold_periods, "validation_period")
        experiment.test_period = _aggregate_fold_period(fold_periods, "test_period") or period or None
        if experiment.test_period is not None:
            experiment.test_period = {
                **dict(experiment.test_period),
                "oos_observations": evaluation.get("oos_observation_count"),
            }
        experiment.metrics = metrics
        experiment.artifacts = {
            "model_package": package.get("path") if package else None,
            "model_package_sha256": package.get("sha256") if package else None,
            "oos_observations": dict(oos or {}),
            "cycle_report": result.get("report"),
        }
        experiment.notes = "Forecast-only OOS research; no champion promotion was performed."
        experiment.status = "COMPLETED"
        experiment.finished_at = (
            _utc_datetime(result.get("completed_at"))
            or experiment.finished_at
            or datetime.now(timezone.utc)
        )
        session.flush()

        candidate_evaluation = session.scalar(
            select(CandidateEvaluation)
            .where(
                CandidateEvaluation.candidate_id == candidate_id,
                CandidateEvaluation.experiment_id == experiment_id,
            )
            .limit(1)
        )
        if candidate_evaluation is None:
            candidate_evaluation = CandidateEvaluation(
                candidate_id=candidate_id,
                experiment_id=experiment_id,
            )
            session.add(candidate_evaluation)
        candidate_evaluation.status = "COMPLETED"
        candidate_evaluation.technically_compatible = bool(package is not None or not selected)
        candidate_evaluation.metrics = metrics
        candidate_evaluation.notes = "Selected for TRAINED challenger packaging." if selected else "Evaluated OOS; not selected."
        session.flush()
        candidates_persisted.append(
            {
                "candidate_id": candidate_id,
                "experiment_id": experiment_id,
                "selected_for_challenger": selected,
            }
        )
    for failure in result.get("candidate_failures") or []:
        original_candidate_id = str(failure["candidate_id"])
        candidate_id = persisted_research_id(original_candidate_id, field_name="candidate_id")
        candidate = session.scalar(
            select(StrategyCandidate).where(StrategyCandidate.candidate_id == candidate_id).limit(1)
        )
        if candidate is None:
            candidate = StrategyCandidate(
                candidate_id=candidate_id,
                model_family=str(failure.get("model_family") or "unknown"),
                feature_families=[str(failure.get("model_family") or "unknown")],
                target_name=target_name,
                forecast_horizon_seconds=horizon,
            )
            session.add(candidate)
        candidate.lifecycle_state = "FAILED"
        candidate.payload = {
            "original_candidate_id": original_candidate_id,
            "error": failure.get("error"),
            "paper_only": True,
            "automatic_promotion": False,
        }
        session.flush()
        experiment_id = persisted_research_id(
            f"cycle-{candidate_id}-{stable_fingerprint({'dataset': dataset_version, 'failure': failure})[:16]}",
            field_name="experiment_id",
        )
        experiment = session.scalar(
            select(ExperimentRun).where(ExperimentRun.experiment_id == experiment_id).limit(1)
        )
        if experiment is None:
            experiment = ExperimentRun(experiment_id=experiment_id)
            session.add(experiment)
        experiment.code_version = "local-research-cycle-v1"
        experiment.configuration = {"candidate": dict(candidate.payload or {}), "walk_forward": walk_forward}
        experiment.dataset_version = dataset_version
        experiment.feature_version = feature_version
        experiment.random_seed = 0
        experiment.metrics = {}
        experiment.artifacts = {"cycle_report": result.get("report")}
        experiment.notes = str(failure.get("error") or "candidate evaluation failed")
        experiment.status = "FAILED"
        experiment.finished_at = (
            _utc_datetime(result.get("completed_at"))
            or experiment.finished_at
            or datetime.now(timezone.utc)
        )
        session.flush()
        candidate_evaluation = session.scalar(
            select(CandidateEvaluation)
            .where(
                CandidateEvaluation.candidate_id == candidate_id,
                CandidateEvaluation.experiment_id == experiment_id,
            )
            .limit(1)
        )
        if candidate_evaluation is None:
            candidate_evaluation = CandidateEvaluation(
                candidate_id=candidate_id,
                experiment_id=experiment_id,
            )
            session.add(candidate_evaluation)
        candidate_evaluation.status = "FAILED"
        candidate_evaluation.technically_compatible = False
        candidate_evaluation.metrics = {}
        candidate_evaluation.notes = str(failure.get("error") or "candidate evaluation failed")
        session.flush()
        candidates_persisted.append(
            {
                "candidate_id": candidate_id,
                "experiment_id": experiment_id,
                "selected_for_challenger": False,
                "status": "FAILED",
            }
        )
    return {
        "registered_challengers": registrations,
        "research_records": candidates_persisted,
        "automatic_promotion": False,
    }


def persist_research_cycle_to_database(result: Mapping[str, Any]) -> dict[str, Any]:
    from app.db.session import SessionLocal, create_db_and_tables

    create_db_and_tables()
    with SessionLocal() as session:
        try:
            persisted = persist_research_cycle(session, result, register_models=True)
            session.commit()
            return persisted
        except Exception:
            session.rollback()
            raise


def register_challenger_packages(packages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Backward-compatible package-only registry helper; never promotes."""

    return persist_research_cycle_to_database({"challenger_packages": list(packages)})[
        "registered_challengers"
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect labels, train/search narrow models, run leakage-safe walk-forward tests, "
            "package artifacts, and register local challengers without promotion."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Prepared point-in-time CSV/JSON/JSONL dataset.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--state", type=Path, help="Defaults to <output-dir>/research_state.json.")
    parser.add_argument("--report", type=Path, help="Defaults to an immutable report path under <output-dir>/reports.")
    parser.add_argument("--target", default="target_future_return_5m")
    parser.add_argument("--feature-schema-version")
    parser.add_argument("--forecast-horizon-seconds", type=int, default=300)
    parser.add_argument("--labels-as-of", help="UTC ISO cutoff; defaults to current UTC time.")
    parser.add_argument("--transaction-cost", type=float, default=0.0008)
    parser.add_argument(
        "--execution-assumptions",
        type=Path,
        help="Optional deterministic JSON for fees/spread/slippage/latency/funding/fills/coverage.",
    )
    parser.add_argument(
        "--model-families",
        default=",".join(FAMILY_FEATURE_COLUMNS),
        help="Controlled comma-separated family names.",
    )
    parser.add_argument("--alphas", default="0.1,1.0,10.0", help="Controlled ridge regularization grid.")
    parser.add_argument("--minimum-new-rows", type=int, default=100)
    parser.add_argument("--train-size", type=int)
    parser.add_argument("--validation-size", type=int)
    parser.add_argument("--test-size", type=int)
    parser.add_argument("--step-size", type=int)
    parser.add_argument("--purge-size", type=int, default=1)
    parser.add_argument("--embargo-size", type=int, default=1)
    parser.add_argument("--rolling", action="store_true", help="Use a rolling rather than expanding train window.")
    parser.add_argument("--force", action="store_true", help="Retrain even if fewer than --minimum-new-rows are new.")
    parser.add_argument("--overwrite", action="store_true", help="Replace deterministic artifacts/reports for this input.")
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Package only. By default successful packages are registered as TRAINED challengers.",
    )
    parser.add_argument(
        "--upload-url",
        help="Optional Railway base URL. Uploads packages as TRAINED only; never activates them.",
    )
    parser.add_argument(
        "--upload-token-env",
        default="ADMIN_TOKEN",
        help="Environment-variable name containing the upload token (default: ADMIN_TOKEN).",
    )
    return parser


def execute(args: argparse.Namespace) -> dict[str, Any]:
    if args.minimum_new_rows <= 0:
        raise ValueError("minimum-new-rows must be positive")
    upload_token: str | None = None
    if args.upload_url:
        token_env = str(args.upload_token_env or "").strip()
        if not token_env or not token_env.replace("_", "").isalnum():
            raise ValueError("upload-token-env must be an environment-variable name")
        upload_token = os.getenv(token_env)
        if not upload_token:
            raise ValueError(f"upload token environment variable is empty: {token_env}")
    rows = read_research_rows(args.input)
    if not rows:
        raise ValueError("input contains no rows")
    schema = _schema_version(rows, args.feature_schema_version)
    input_hash = sha256_file(args.input)
    dataset_version = f"sha256:{input_hash}"
    labels_as_of = args.labels_as_of or datetime.now(timezone.utc).isoformat()
    assumptions = _execution_assumptions(args.execution_assumptions)
    _labeled, inventory = discover_available_labeled_rows(
        rows,
        target_name=args.target,
        forecast_horizon_seconds=args.forecast_horizon_seconds,
        labels_as_of=labels_as_of,
        transaction_cost=args.transaction_cost,
    )
    output_dir = args.output_dir.resolve()
    state_path = (args.state or (output_dir / "research_state.json")).resolve()
    state = _read_state(state_path)
    seen = set(state.get("seen_labeled_row_ids", []))
    current_ids = set(inventory.row_ids)
    new_ids = current_ids - seen
    if not args.force and len(new_ids) < args.minimum_new_rows:
        return {
            "status": "waiting_for_labels",
            "paper_only": True,
            "automatic_promotion": False,
            "dataset_version": dataset_version,
            "label_inventory": {
                key: value for key, value in inventory.model_dump().items() if key != "row_ids"
            },
            "new_labeled_rows": len(new_ids),
            "minimum_new_labeled_rows": args.minimum_new_rows,
            "state": str(state_path),
            "registered_challengers": [],
            "promotions": [],
        }
    result = run_narrow_research_cycle(
        rows,
        output_dir=output_dir,
        target_name=args.target,
        feature_schema_version=schema,
        allowed_feature_columns=model_input_columns_for_schema(schema),
        forecast_horizon_seconds=args.forecast_horizon_seconds,
        dataset_version=dataset_version,
        labels_as_of=labels_as_of,
        transaction_cost=args.transaction_cost,
        model_families=_csv_values(args.model_families, label="model-families"),
        alphas=_alphas(args.alphas),
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        step_size=args.step_size,
        purge_size=args.purge_size,
        embargo_size=args.embargo_size,
        expanding=not args.rolling,
        execution_assumptions=assumptions,
        overwrite=args.overwrite,
    )
    result["new_labeled_rows"] = len(new_ids)
    result["input"] = str(args.input.resolve())
    result["input_sha256"] = input_hash
    result["labels_as_of"] = labels_as_of
    # Row ids are durable cursor material, not report material.
    result["label_inventory"].pop("row_ids", None)
    report_path = args.report or (
        output_dir
        / "reports"
        / f"cycle-{input_hash[:12]}-{inventory.available_labeled_rows}.json"
    )
    result["report"] = str(report_path.resolve())
    result["completed_at"] = datetime.now(timezone.utc).isoformat()
    if not args.no_register:
        persisted = persist_research_cycle_to_database(result)
        result["registered_challengers"] = persisted["registered_challengers"]
        result["research_records"] = persisted["research_records"]
    if args.upload_url:
        from scripts.upload_model import upload_package

        uploaded: list[dict[str, Any]] = []
        for package in result.get("challenger_packages", []):
            response = upload_package(
                url=str(args.upload_url),
                token=str(upload_token),
                package=Path(str(package["path"])),
            )
            model = response.get("model") if isinstance(response.get("model"), Mapping) else {}
            uploaded.append(
                {
                    "package_sha256": package.get("sha256"),
                    "model_version_id": model.get("id"),
                    "model_id": model.get("model_id"),
                    "version": model.get("version"),
                    "lifecycle_state": model.get("lifecycle_state") or "TRAINED",
                    "activated": False,
                    "promoted": False,
                }
            )
        result["uploaded_challengers"] = uploaded
    _atomic_json(report_path, result, overwrite=args.overwrite)
    _write_state(
        state_path,
        {
            "contract_version": 1,
            "seen_labeled_row_ids": sorted(seen | current_ids),
            "last_dataset_version": dataset_version,
            "last_available_labeled_rows": inventory.available_labeled_rows,
            "last_new_labeled_rows": len(new_ids),
            "last_report": str(report_path.resolve()),
            "last_status": result["status"],
            "last_completed_at": result["completed_at"],
            "automatic_promotion": False,
        },
    )
    result["state"] = str(state_path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = execute(args)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {"status": "error", "error": f"{type(exc).__name__}: {exc}", "paper_only": True},
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "new_labeled_rows": result.get("new_labeled_rows", 0),
                "candidate_count": result.get("candidate_count", 0),
                "challenger_packages": [item["path"] for item in result.get("challenger_packages", [])],
                "registered_challengers": result.get("registered_challengers", []),
                "uploaded_challengers": result.get("uploaded_challengers", []),
                "report": result.get("report"),
                "state": result.get("state"),
                "safety": "paper-only; this command has no champion-promotion or exchange-order operation",
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
