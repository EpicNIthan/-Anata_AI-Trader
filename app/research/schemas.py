"""Declarative, dependency-light schemas used by the local research pipeline.

The production paper-trading process intentionally does not import this module.
Research configurations are plain data, which makes them safe to persist, hash,
review, and replay without evaluating generated Python code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TypeAlias


JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
WindowSize: TypeAlias = int | timedelta


class ResearchValidationError(ValueError):
    """Raised when a declarative research object is malformed or unsafe."""


class ExperimentStatus(str, Enum):
    """Lifecycle states for a reproducible research experiment."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


DEFAULT_MODEL_FAMILIES: tuple[str, ...] = (
    "short_horizon_momentum",
    "medium_horizon_momentum",
    "mean_reversion",
    "breakout_pressure",
    "derivatives_flow",
    "liquidation_pressure",
    "news_event",
    "cross_asset_context",
    "linear_baseline",
)

_DURATION_RE = re.compile(r"^(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>s|m|h|d|w)$", re.IGNORECASE)
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_UNSAFE_PARAMETER_KEYS = {
    "code",
    "source",
    "source_code",
    "python",
    "script",
    "callable",
    "function",
    "module",
    "eval",
    "exec",
}


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


def ensure_utc(value: datetime | str, *, field_name: str = "timestamp") -> datetime:
    """Coerce an ISO timestamp to UTC and reject timezone-naive values.

    Research code must not silently interpret a local timestamp as UTC: doing so
    can move an observation across a split boundary and create subtle leakage.
    """

    if isinstance(value, str):
        raw = value.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            value = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise ResearchValidationError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if not isinstance(value, datetime):
        raise ResearchValidationError(f"{field_name} must be a datetime or ISO-8601 timestamp")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ResearchValidationError(f"{field_name} must include a timezone")
    return value.astimezone(timezone.utc)


def parse_window(value: WindowSize | str, *, field_name: str = "window") -> WindowSize:
    """Parse a positive row-count or compact duration (for example ``30d``)."""

    if isinstance(value, bool):
        raise ResearchValidationError(f"{field_name} cannot be boolean")
    if isinstance(value, int):
        if value <= 0:
            raise ResearchValidationError(f"{field_name} row count must be positive")
        return value
    if isinstance(value, timedelta):
        if value.total_seconds() <= 0:
            raise ResearchValidationError(f"{field_name} duration must be positive")
        return value
    if isinstance(value, str):
        match = _DURATION_RE.match(value.strip())
        if not match:
            raise ResearchValidationError(
                f"{field_name} must be a positive row count or duration such as '90d'"
            )
        amount = float(match.group("value"))
        unit = match.group("unit").lower()
        seconds = amount * {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[unit]
        if seconds <= 0:
            raise ResearchValidationError(f"{field_name} duration must be positive")
        return timedelta(seconds=seconds)
    raise ResearchValidationError(f"{field_name} must be an integer, timedelta, or duration string")


def window_to_value(value: WindowSize) -> int | str:
    """Return a stable JSON-friendly representation of a row or time window."""

    if isinstance(value, int):
        return value
    seconds = value.total_seconds()
    if seconds.is_integer():
        return f"{int(seconds)}s"
    return f"{seconds:g}s"


def _require_identifier(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not _IDENTIFIER_RE.fullmatch(cleaned):
        raise ResearchValidationError(
            f"{field_name} must be 1-128 characters using letters, numbers, '.', '_', ':', or '-'"
        )
    return cleaned


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ResearchValidationError(f"{field_name} must be numeric")
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ResearchValidationError(f"{field_name} must be numeric") from exc
    if not math.isfinite(converted):
        raise ResearchValidationError(f"{field_name} must be finite")
    return converted


def _json_safe(value: Any, *, field_name: str = "value") -> JSONValue:
    """Validate a value is declarative JSON and contains no executable hooks."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ResearchValidationError(f"{field_name} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        output: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ResearchValidationError(f"{field_name} mapping keys must be non-empty strings")
            if key.strip().lower() in _UNSAFE_PARAMETER_KEYS:
                raise ResearchValidationError(
                    f"{field_name}.{key} is not allowed in declarative candidate specifications"
                )
            output[key] = _json_safe(item, field_name=f"{field_name}.{key}")
        return output
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, field_name=f"{field_name}[]") for item in value]
    raise ResearchValidationError(f"{field_name} must contain JSON-compatible values only")


def jsonable(value: Any) -> JSONValue:
    """Convert supported research objects to deterministic JSON-compatible data."""

    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if isinstance(value, datetime):
        return ensure_utc(value).isoformat()
    if isinstance(value, timedelta):
        return window_to_value(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    return _json_safe(value)


def stable_json_dumps(value: Any) -> str:
    """Serialize declarative values in canonical form for checksums and reports."""

    return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def stable_fingerprint(value: Any) -> str:
    """Return a SHA-256 fingerprint of a canonical declarative object."""

    return hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class TimeRange:
    """A half-open UTC interval ``[start, end)`` used by all research splits."""

    start: datetime | str
    end: datetime | str

    def __post_init__(self) -> None:
        start = ensure_utc(self.start, field_name="start")
        end = ensure_utc(self.end, field_name="end")
        if end <= start:
            raise ResearchValidationError("TimeRange.end must be later than TimeRange.start")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    def contains(self, value: datetime | str) -> bool:
        timestamp = ensure_utc(value)
        return self.start <= timestamp < self.end

    def overlaps(self, other: "TimeRange") -> bool:
        return self.start < other.end and other.start < self.end

    @property
    def duration(self) -> timedelta:
        return self.end - self.start

    def model_dump(self, **_: Any) -> dict[str, str]:
        return {"start": self.start.isoformat(), "end": self.end.isoformat()}

    to_dict = model_dump

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TimeRange":
        return cls(start=value["start"], end=value["end"])


@dataclass(frozen=True, slots=True)
class CandidateStrategySpec:
    """A safe, declarative candidate strategy definition.

    It deliberately describes *what* to research, not arbitrary executable
    source. Model factories and feature registries decide how the declared
    family is implemented.
    """

    candidate_id: str
    feature_families: Sequence[str]
    target_name: str
    forecast_horizon: int
    model_family: str
    hyperparameters: Mapping[str, Any] = field(default_factory=dict)
    regime_filter: str | Sequence[str] | None = None
    signal_threshold: float = 0.0
    confidence_policy: str = "bounded"
    cost_model: str = "default"
    portfolio_policy: str = "deterministic_baseline"
    exit_policy_name: str = "horizon"
    training_window: WindowSize | str = 500
    validation_window: WindowSize | str = 100
    random_seed: int = 0
    hypothesis: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        candidate_id = _require_identifier(self.candidate_id, "candidate_id")
        model_family = _require_identifier(self.model_family, "model_family")
        target_name = _require_identifier(self.target_name, "target_name")
        if isinstance(self.feature_families, (str, bytes)):
            raise ResearchValidationError("feature_families must be a sequence of feature-family names, not one string")
        families = tuple(_require_identifier(str(item), "feature_families item") for item in self.feature_families)
        if not families:
            raise ResearchValidationError("feature_families cannot be empty")
        if len(set(families)) != len(families):
            raise ResearchValidationError("feature_families must not contain duplicates")
        if isinstance(self.forecast_horizon, bool) or int(self.forecast_horizon) <= 0:
            raise ResearchValidationError("forecast_horizon must be a positive number of seconds")
        if int(self.forecast_horizon) != self.forecast_horizon:
            raise ResearchValidationError("forecast_horizon must be an integer number of seconds")
        threshold = _finite_float(self.signal_threshold, "signal_threshold")
        if threshold < 0:
            raise ResearchValidationError("signal_threshold cannot be negative")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ResearchValidationError("random_seed must be an integer")
        for name in ("confidence_policy", "cost_model", "portfolio_policy", "exit_policy_name"):
            _require_identifier(str(getattr(self, name)), name)
        if not isinstance(self.hypothesis, str):
            raise ResearchValidationError("hypothesis must be text")
        regime_filter: str | tuple[str, ...] | None
        if self.regime_filter is None or self.regime_filter == "":
            regime_filter = None
        elif isinstance(self.regime_filter, str):
            regime_filter = _require_identifier(self.regime_filter, "regime_filter")
        else:
            regime_filter = tuple(_require_identifier(str(item), "regime_filter item") for item in self.regime_filter)
            if not regime_filter:
                regime_filter = None
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "model_family", model_family)
        object.__setattr__(self, "target_name", target_name)
        object.__setattr__(self, "feature_families", families)
        object.__setattr__(self, "forecast_horizon", int(self.forecast_horizon))
        object.__setattr__(self, "signal_threshold", threshold)
        object.__setattr__(self, "regime_filter", regime_filter)
        object.__setattr__(self, "training_window", parse_window(self.training_window, field_name="training_window"))
        object.__setattr__(self, "validation_window", parse_window(self.validation_window, field_name="validation_window"))
        object.__setattr__(self, "hyperparameters", _json_safe(self.hyperparameters, field_name="hyperparameters"))
        object.__setattr__(self, "metadata", _json_safe(self.metadata, field_name="metadata"))

    @property
    def fingerprint(self) -> str:
        """Stable content hash, independent of runtime object identity."""

        return stable_fingerprint(self.model_dump())

    def model_dump(self, **_: Any) -> dict[str, JSONValue]:
        regime: JSONValue
        if isinstance(self.regime_filter, tuple):
            regime = list(self.regime_filter)
        else:
            regime = self.regime_filter
        return {
            "candidate_id": self.candidate_id,
            "feature_families": list(self.feature_families),
            "target_name": self.target_name,
            "forecast_horizon": self.forecast_horizon,
            "model_family": self.model_family,
            "hyperparameters": dict(self.hyperparameters),
            "regime_filter": regime,
            "signal_threshold": self.signal_threshold,
            "confidence_policy": self.confidence_policy,
            "cost_model": self.cost_model,
            "portfolio_policy": self.portfolio_policy,
            "exit_policy_name": self.exit_policy_name,
            "training_window": window_to_value(self.training_window),
            "validation_window": window_to_value(self.validation_window),
            "random_seed": self.random_seed,
            "hypothesis": self.hypothesis,
            "metadata": dict(self.metadata),
        }

    to_dict = model_dump

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateStrategySpec":
        return cls(
            candidate_id=str(value["candidate_id"]),
            feature_families=value["feature_families"],
            target_name=str(value["target_name"]),
            forecast_horizon=int(value["forecast_horizon"]),
            model_family=str(value["model_family"]),
            hyperparameters=value.get("hyperparameters", {}),
            regime_filter=value.get("regime_filter"),
            signal_threshold=float(value.get("signal_threshold", 0.0)),
            confidence_policy=str(value.get("confidence_policy", "bounded")),
            cost_model=str(value.get("cost_model", "default")),
            portfolio_policy=str(value.get("portfolio_policy", "deterministic_baseline")),
            exit_policy_name=str(value.get("exit_policy_name", "horizon")),
            training_window=value.get("training_window", 500),
            validation_window=value.get("validation_window", 100),
            random_seed=int(value.get("random_seed", 0)),
            hypothesis=str(value.get("hypothesis", "")),
            metadata=value.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class CandidateSearchSpace:
    """A bounded, deterministic search space for declarative candidates."""

    candidate_prefix: str
    feature_family_sets: Sequence[Sequence[str]]
    target_names: Sequence[str]
    forecast_horizons: Sequence[int]
    model_families: Sequence[str]
    hyperparameter_grid: Mapping[str, Mapping[str, Sequence[Any]]] = field(default_factory=dict)
    regime_filters: Sequence[str | Sequence[str] | None] = (None,)
    signal_thresholds: Sequence[float] = (0.0,)
    confidence_policies: Sequence[str] = ("bounded",)
    cost_models: Sequence[str] = ("default",)
    portfolio_policies: Sequence[str] = ("deterministic_baseline",)
    exit_policy_names: Sequence[str] = ("horizon",)
    training_windows: Sequence[WindowSize | str] = (500,)
    validation_windows: Sequence[WindowSize | str] = (100,)
    random_seed: int = 0
    hypothesis: str = ""
    max_candidates: int = 500

    def __post_init__(self) -> None:
        _require_identifier(self.candidate_prefix, "candidate_prefix")
        if self.max_candidates <= 0:
            raise ResearchValidationError("max_candidates must be positive")
        if not self.feature_family_sets or not self.target_names or not self.forecast_horizons or not self.model_families:
            raise ResearchValidationError("search space requires feature families, targets, horizons, and model families")
        for family in self.model_families:
            _require_identifier(str(family), "model_family")
        for family, params in self.hyperparameter_grid.items():
            _require_identifier(str(family), "hyperparameter_grid model family")
            _json_safe(params, field_name=f"hyperparameter_grid.{family}")

    def _parameter_sets(self, model_family: str) -> Iterable[dict[str, JSONValue]]:
        """Yield deterministic cartesian parameter combinations for one family."""

        parameters = self.hyperparameter_grid.get(model_family, {})
        if not parameters:
            yield {}
            return
        keys = sorted(parameters)
        values: list[Sequence[Any]] = []
        for key in keys:
            options = parameters[key]
            if not isinstance(options, Sequence) or isinstance(options, (str, bytes)) or not options:
                raise ResearchValidationError(
                    f"hyperparameter_grid.{model_family}.{key} must be a non-empty sequence"
                )
            values.append(options)
        from itertools import product

        for combination in product(*values):
            yield {key: _json_safe(value, field_name=f"hyperparameters.{key}") for key, value in zip(keys, combination)}

    def iter_candidates(self) -> Iterable[CandidateStrategySpec]:
        """Generate deterministic, bounded candidates without dynamic code execution."""

        from itertools import product

        produced = 0
        for model_family in self.model_families:
            for hyperparameters in self._parameter_sets(str(model_family)):
                dimensions = product(
                    self.feature_family_sets,
                    self.target_names,
                    self.forecast_horizons,
                    self.regime_filters,
                    self.signal_thresholds,
                    self.confidence_policies,
                    self.cost_models,
                    self.portfolio_policies,
                    self.exit_policy_names,
                    self.training_windows,
                    self.validation_windows,
                )
                for (
                    feature_families,
                    target_name,
                    horizon,
                    regime_filter,
                    threshold,
                    confidence_policy,
                    cost_model,
                    portfolio_policy,
                    exit_policy_name,
                    training_window,
                    validation_window,
                ) in dimensions:
                    if produced >= self.max_candidates:
                        raise ResearchValidationError(
                            f"search space exceeds max_candidates={self.max_candidates}; narrow it explicitly"
                        )
                    identity = {
                        "prefix": self.candidate_prefix,
                        "index": produced,
                        "model_family": model_family,
                        "hyperparameters": hyperparameters,
                        "feature_families": list(feature_families),
                        "target_name": target_name,
                        "forecast_horizon": horizon,
                        "regime_filter": regime_filter,
                        "signal_threshold": threshold,
                    }
                    candidate_id = f"{self.candidate_prefix[:110]}-{produced:04d}-{stable_fingerprint(identity)[:8]}"
                    yield CandidateStrategySpec(
                        candidate_id=candidate_id,
                        feature_families=feature_families,
                        target_name=str(target_name),
                        forecast_horizon=int(horizon),
                        model_family=str(model_family),
                        hyperparameters=hyperparameters,
                        regime_filter=regime_filter,
                        signal_threshold=float(threshold),
                        confidence_policy=str(confidence_policy),
                        cost_model=str(cost_model),
                        portfolio_policy=str(portfolio_policy),
                        exit_policy_name=str(exit_policy_name),
                        training_window=training_window,
                        validation_window=validation_window,
                        random_seed=self.random_seed,
                        hypothesis=self.hypothesis,
                        metadata={"search_space": self.candidate_prefix},
                    )
                    produced += 1

    @property
    def fingerprint(self) -> str:
        return stable_fingerprint(self.model_dump())

    def model_dump(self, **_: Any) -> dict[str, JSONValue]:
        def regime_value(value: str | Sequence[str] | None) -> JSONValue:
            if value is None or isinstance(value, str):
                return value
            return [str(item) for item in value]

        return {
            "candidate_prefix": self.candidate_prefix,
            "feature_family_sets": [list(values) for values in self.feature_family_sets],
            "target_names": [str(value) for value in self.target_names],
            "forecast_horizons": [int(value) for value in self.forecast_horizons],
            "model_families": [str(value) for value in self.model_families],
            "hyperparameter_grid": _json_safe(self.hyperparameter_grid, field_name="hyperparameter_grid"),
            "regime_filters": [regime_value(value) for value in self.regime_filters],
            "signal_thresholds": [float(value) for value in self.signal_thresholds],
            "confidence_policies": [str(value) for value in self.confidence_policies],
            "cost_models": [str(value) for value in self.cost_models],
            "portfolio_policies": [str(value) for value in self.portfolio_policies],
            "exit_policy_names": [str(value) for value in self.exit_policy_names],
            "training_windows": [window_to_value(parse_window(value, field_name="training_window")) for value in self.training_windows],
            "validation_windows": [window_to_value(parse_window(value, field_name="validation_window")) for value in self.validation_windows],
            "random_seed": self.random_seed,
            "hypothesis": self.hypothesis,
            "max_candidates": self.max_candidates,
        }

    to_dict = model_dump

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateSearchSpace":
        return cls(
            candidate_prefix=str(value["candidate_prefix"]),
            feature_family_sets=value["feature_family_sets"],
            target_names=value["target_names"],
            forecast_horizons=value["forecast_horizons"],
            model_families=value["model_families"],
            hyperparameter_grid=value.get("hyperparameter_grid", {}),
            regime_filters=value.get("regime_filters", (None,)),
            signal_thresholds=value.get("signal_thresholds", (0.0,)),
            confidence_policies=value.get("confidence_policies", ("bounded",)),
            cost_models=value.get("cost_models", ("default",)),
            portfolio_policies=value.get("portfolio_policies", ("deterministic_baseline",)),
            exit_policy_names=value.get("exit_policy_names", ("horizon",)),
            training_windows=value.get("training_windows", (500,)),
            validation_windows=value.get("validation_windows", (100,)),
            random_seed=int(value.get("random_seed", 0)),
            hypothesis=str(value.get("hypothesis", "")),
            max_candidates=int(value.get("max_candidates", 500)),
        )


@dataclass(frozen=True, slots=True)
class EvaluationObservation:
    """One point-in-time prediction and its realized, subsequently available label."""

    timestamp: datetime | str
    prediction: float
    actual_return: float
    symbol: str | None = None
    signal_id: str | None = None
    model_family: str | None = None
    regime: str | None = None
    position: float | None = None
    transaction_cost: float = 0.0
    holding_seconds: float | None = None
    external_ai_available: bool | None = None
    available_to_model_time: datetime | str | None = None
    label_available_time: datetime | str | None = None
    feature_families: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        timestamp = ensure_utc(self.timestamp, field_name="timestamp")
        available = (
            ensure_utc(self.available_to_model_time, field_name="available_to_model_time")
            if self.available_to_model_time is not None
            else timestamp
        )
        if available > timestamp:
            raise ResearchValidationError(
                "available_to_model_time cannot be after the decision timestamp (point-in-time leakage)"
            )
        label_available = (
            ensure_utc(self.label_available_time, field_name="label_available_time")
            if self.label_available_time is not None
            else None
        )
        if label_available is not None and label_available < timestamp:
            raise ResearchValidationError("label_available_time cannot precede the decision timestamp")
        position = _finite_float(self.position, "position") if self.position is not None else None
        holding = _finite_float(self.holding_seconds, "holding_seconds") if self.holding_seconds is not None else None
        if holding is not None and holding < 0:
            raise ResearchValidationError("holding_seconds cannot be negative")
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "available_to_model_time", available)
        object.__setattr__(self, "label_available_time", label_available)
        object.__setattr__(self, "prediction", _finite_float(self.prediction, "prediction"))
        object.__setattr__(self, "actual_return", _finite_float(self.actual_return, "actual_return"))
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "transaction_cost", max(0.0, _finite_float(self.transaction_cost, "transaction_cost")))
        object.__setattr__(self, "holding_seconds", holding)
        feature_families = () if self.feature_families is None else self.feature_families
        if isinstance(feature_families, (str, bytes)):
            feature_families = (feature_families,)
        external_ai_available = self.external_ai_available
        if external_ai_available is not None and not isinstance(external_ai_available, bool):
            raise ResearchValidationError("external_ai_available must be true, false, or null")
        object.__setattr__(self, "external_ai_available", external_ai_available)
        object.__setattr__(self, "feature_families", tuple(str(item) for item in feature_families))
        object.__setattr__(self, "metadata", _json_safe(self.metadata, field_name="metadata"))

    def model_dump(self, **_: Any) -> dict[str, JSONValue]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "prediction": self.prediction,
            "actual_return": self.actual_return,
            "symbol": self.symbol,
            "signal_id": self.signal_id,
            "model_family": self.model_family,
            "regime": self.regime,
            "position": self.position,
            "transaction_cost": self.transaction_cost,
            "holding_seconds": self.holding_seconds,
            "external_ai_available": self.external_ai_available,
            "available_to_model_time": self.available_to_model_time.isoformat(),
            "label_available_time": self.label_available_time.isoformat() if self.label_available_time else None,
            "feature_families": list(self.feature_families),
            "metadata": dict(self.metadata),
        }

    to_dict = model_dump


@dataclass(frozen=True, slots=True)
class ExperimentDefinition:
    """Everything needed to reproduce an experiment except its generated artifact bytes."""

    experiment_id: str
    candidate: CandidateStrategySpec
    dataset_version: str
    feature_version: str
    model_version: str | None = None
    code_version: str = "unknown"
    random_seed: int | None = None
    train_period: TimeRange | None = None
    validation_period: TimeRange | None = None
    test_period: TimeRange | None = None
    configuration: Mapping[str, Any] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "experiment_id", _require_identifier(self.experiment_id, "experiment_id"))
        object.__setattr__(self, "dataset_version", _require_identifier(self.dataset_version, "dataset_version"))
        object.__setattr__(self, "feature_version", _require_identifier(self.feature_version, "feature_version"))
        object.__setattr__(self, "code_version", _require_identifier(self.code_version, "code_version"))
        if self.model_version is not None:
            object.__setattr__(self, "model_version", _require_identifier(self.model_version, "model_version"))
        if self.random_seed is not None and (isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int)):
            raise ResearchValidationError("random_seed must be an integer or None")
        if not isinstance(self.notes, str):
            raise ResearchValidationError("notes must be text")
        configuration = _json_safe(self.configuration, field_name="configuration")
        object.__setattr__(self, "configuration", configuration)
        periods = (self.train_period, self.validation_period, self.test_period)
        present = [period for period in periods if period is not None]
        for left, right in zip(present, present[1:]):
            if left.end > right.start:
                raise ResearchValidationError("experiment train, validation, and test periods must not overlap")

    @property
    def reproducibility_fingerprint(self) -> str:
        payload = self._base_dump()
        payload.pop("experiment_id", None)
        return stable_fingerprint(payload)

    def _base_dump(self) -> dict[str, JSONValue]:
        return {
            "experiment_id": self.experiment_id,
            "candidate": self.candidate.model_dump(),
            "dataset_version": self.dataset_version,
            "feature_version": self.feature_version,
            "model_version": self.model_version,
            "code_version": self.code_version,
            "random_seed": self.random_seed if self.random_seed is not None else self.candidate.random_seed,
            "train_period": self.train_period.model_dump() if self.train_period else None,
            "validation_period": self.validation_period.model_dump() if self.validation_period else None,
            "test_period": self.test_period.model_dump() if self.test_period else None,
            "configuration": dict(self.configuration),
            "notes": self.notes,
        }

    def model_dump(self, **_: Any) -> dict[str, JSONValue]:
        payload = self._base_dump()
        payload["reproducibility_fingerprint"] = self.reproducibility_fingerprint
        return payload

    to_dict = model_dump

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperimentDefinition":
        def period(name: str) -> TimeRange | None:
            raw = value.get(name)
            return TimeRange.from_dict(raw) if isinstance(raw, Mapping) else None

        return cls(
            experiment_id=str(value["experiment_id"]),
            candidate=CandidateStrategySpec.from_dict(value["candidate"]),
            dataset_version=str(value["dataset_version"]),
            feature_version=str(value["feature_version"]),
            model_version=value.get("model_version"),
            code_version=str(value.get("code_version", "unknown")),
            random_seed=value.get("random_seed"),
            train_period=period("train_period"),
            validation_period=period("validation_period"),
            test_period=period("test_period"),
            configuration=value.get("configuration", {}),
            notes=str(value.get("notes", "")),
        )


@dataclass(frozen=True, slots=True)
class ExperimentReport:
    """Persistable result of one historical or walk-forward experiment."""

    definition: ExperimentDefinition
    status: ExperimentStatus | str
    metrics: Mapping[str, Any] = field(default_factory=dict)
    fold_metrics: Sequence[Mapping[str, Any]] = ()
    artifacts: Sequence[str | Path] = ()
    notes: Sequence[str] = ()
    started_at: datetime | str | None = None
    finished_at: datetime | str | None = None
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = ExperimentStatus(self.status)
        started = ensure_utc(self.started_at, field_name="started_at") if self.started_at is not None else None
        finished = ensure_utc(self.finished_at, field_name="finished_at") if self.finished_at is not None else None
        if started and finished and finished < started:
            raise ResearchValidationError("finished_at cannot precede started_at")
        if self.error is not None and not isinstance(self.error, str):
            raise ResearchValidationError("error must be text or None")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "started_at", started)
        object.__setattr__(self, "finished_at", finished)
        object.__setattr__(self, "metrics", _json_safe(self.metrics, field_name="metrics"))
        object.__setattr__(
            self,
            "fold_metrics",
            tuple(_json_safe(item, field_name="fold_metrics") for item in self.fold_metrics),
        )
        object.__setattr__(self, "artifacts", tuple(str(item) for item in self.artifacts))
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))
        object.__setattr__(self, "metadata", _json_safe(self.metadata, field_name="metadata"))

    @property
    def report_fingerprint(self) -> str:
        return stable_fingerprint(self._base_dump())

    def _base_dump(self) -> dict[str, JSONValue]:
        return {
            "experiment": self.definition.model_dump(),
            "status": self.status.value,
            "metrics": dict(self.metrics),
            "fold_metrics": [dict(item) for item in self.fold_metrics],
            "artifacts": list(self.artifacts),
            "notes": list(self.notes),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "error": self.error,
            "metadata": dict(self.metadata),
        }

    def model_dump(self, **_: Any) -> dict[str, JSONValue]:
        payload = self._base_dump()
        payload["report_fingerprint"] = self.report_fingerprint
        return payload

    to_dict = model_dump

    def write_json(self, path: str | Path) -> Path:
        """Write a canonical JSON report and return the resolved report path."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(stable_json_dumps(self.model_dump()) + "\n", encoding="utf-8")
        temporary.replace(target)
        return target

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperimentReport":
        definition_value = value.get("experiment", value.get("definition"))
        if not isinstance(definition_value, Mapping):
            raise ResearchValidationError("experiment report requires an experiment definition")
        return cls(
            definition=ExperimentDefinition.from_dict(definition_value),
            status=value.get("status", ExperimentStatus.PENDING.value),
            metrics=value.get("metrics", {}),
            fold_metrics=value.get("fold_metrics", ()),
            artifacts=value.get("artifacts", ()),
            notes=value.get("notes", ()),
            started_at=value.get("started_at"),
            finished_at=value.get("finished_at"),
            error=value.get("error"),
            metadata=value.get("metadata", {}),
        )
