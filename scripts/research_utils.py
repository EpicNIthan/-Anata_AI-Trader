"""Small stdlib helpers for the local, read-only research commands."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.research import CandidateStrategySpec, TimeRange, observation_from_row  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _coerce_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    if stripped.lower() in {"true", "false"}:
        return stripped.lower() == "true"
    if stripped[0:1] in {"{", "["}:
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    return value


def read_research_rows(path: Path) -> list[dict[str, Any]]:
    """Read JSONL, JSON-array, or CSV observations without time-series shuffling."""

    if not path.is_file():
        raise FileNotFoundError(f"Input does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [{key: _coerce_value(value) for key, value in row.items()} for row in csv.DictReader(handle)]
    raw = path.read_text(encoding="utf-8")
    if suffix == ".json":
        parsed = json.loads(raw)
        if not isinstance(parsed, list) or not all(isinstance(item, Mapping) for item in parsed):
            raise ValueError("JSON research input must be an array of objects")
        return [dict(item) for item in parsed]
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, Mapping):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        rows.append(dict(parsed))
    return rows


def default_candidate() -> CandidateStrategySpec:
    return CandidateStrategySpec(
        candidate_id="stored-prediction-baseline",
        feature_families=("stored_features",),
        target_name="actual_return",
        forecast_horizon=900,
        model_family="stored_prediction",
        hyperparameters={},
        signal_threshold=0.0,
        confidence_policy="bounded",
        cost_model="recorded_cost",
        portfolio_policy="evaluation_only",
        exit_policy_name="recorded_horizon",
        training_window=100,
        validation_window=1,
        random_seed=0,
        hypothesis="Evaluate already-recorded predictions only; this command does not train, deploy, or trade.",
    )


def load_candidate(path: Path | None) -> CandidateStrategySpec:
    if path is None:
        return default_candidate()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping) and isinstance(payload.get("candidate"), Mapping):
        payload = payload["candidate"]
    if not isinstance(payload, Mapping):
        raise ValueError("candidate file must contain an object or {candidate: object}")
    return CandidateStrategySpec.from_dict(payload)


def observation_period(rows: Iterable[Mapping[str, Any]]) -> TimeRange | None:
    timestamps = sorted(observation_from_row(row).timestamp for row in rows)
    if not timestamps or timestamps[-1] == timestamps[0]:
        return None
    return TimeRange(start=timestamps[0], end=timestamps[-1] + timedelta(microseconds=1))


def ensure_new_output(path: Path, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing report: {path}. Pass --overwrite to replace it.")
