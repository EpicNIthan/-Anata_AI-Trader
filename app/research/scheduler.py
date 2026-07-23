"""Local-only orchestration for reproducible candidate research.

This module is deliberately callback-driven: it can use the repository's
existing dataset/training services without importing the paper execution loop.
It never activates a champion unless an operator explicitly sets
``allow_auto_champion_promotion=True`` *and* supplies a promotion callback.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import inspect
import json
from pathlib import Path
import threading
from typing import Any, Protocol

from app.research.evaluation import (
    EvaluationResult,
    WalkForwardResult,
    evaluate_observations,
    make_experiment_report,
)
from app.research.schemas import (
    CandidateStrategySpec,
    EvaluationObservation,
    ExperimentDefinition,
    ExperimentReport,
    ExperimentStatus,
    ResearchValidationError,
    TimeRange,
    ensure_utc,
    stable_fingerprint,
    stable_json_dumps,
    utc_now,
)


class DatasetSnapshotProvider(Protocol):
    """Protocol for a local data lake or a synchronized Railway export."""

    def __call__(self) -> "LabeledDataSnapshot": ...


class CandidateProvider(Protocol):
    """Protocol for a deterministic candidate registry or search space."""

    def __call__(self, snapshot: "LabeledDataSnapshot") -> Iterable[CandidateStrategySpec]: ...


class CandidateTrainer(Protocol):
    """Train one candidate locally and return an optional artifact descriptor."""

    def __call__(self, candidate: CandidateStrategySpec, snapshot: "LabeledDataSnapshot") -> Any: ...


class CandidateEvaluator(Protocol):
    """Evaluate one candidate locally after training or loading its artifact."""

    def __call__(self, candidate: CandidateStrategySpec, artifact: Any, snapshot: "LabeledDataSnapshot") -> Any: ...


@dataclass(frozen=True, slots=True)
class LabeledDataSnapshot:
    """A point-in-time description of locally available labeled research data."""

    dataset_version: str
    feature_version: str
    total_labeled_rows: int
    rows: Sequence[Any] = ()
    newest_label_time: datetime | str | None = None
    data_range: TimeRange | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.dataset_version or not str(self.dataset_version).strip():
            raise ResearchValidationError("dataset_version cannot be empty")
        if not self.feature_version or not str(self.feature_version).strip():
            raise ResearchValidationError("feature_version cannot be empty")
        if isinstance(self.total_labeled_rows, bool) or self.total_labeled_rows < 0:
            raise ResearchValidationError("total_labeled_rows must be a non-negative integer")
        newest = ensure_utc(self.newest_label_time, field_name="newest_label_time") if self.newest_label_time else None
        if self.rows and len(self.rows) > self.total_labeled_rows:
            raise ResearchValidationError("rows cannot exceed total_labeled_rows")
        if self.data_range is not None and not isinstance(self.data_range, TimeRange):
            raise ResearchValidationError("data_range must be a TimeRange or None")
        object.__setattr__(self, "dataset_version", str(self.dataset_version).strip())
        object.__setattr__(self, "feature_version", str(self.feature_version).strip())
        object.__setattr__(self, "newest_label_time", newest)
        object.__setattr__(self, "rows", tuple(self.rows))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[Any],
        *,
        dataset_version: str,
        feature_version: str,
        total_labeled_rows: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "LabeledDataSnapshot":
        """Build a snapshot for in-memory research scripts or deterministic tests."""

        timestamps: list[datetime] = []
        for row in rows:
            value = None
            if isinstance(row, Mapping):
                value = row.get("label_available_time") or row.get("timestamp") or row.get("as_of")
            else:
                value = (
                    getattr(row, "label_available_time", None)
                    or getattr(row, "timestamp", None)
                    or getattr(row, "as_of", None)
                )
            if value is not None:
                timestamps.append(ensure_utc(value, field_name="row timestamp"))
        data_range = None
        # TimeRange is half-open, so one microsecond preserves even a single row.
        if timestamps:
            start = min(timestamps)
            end = max(timestamps)
            data_range = TimeRange(start=start, end=end + timedelta(microseconds=1))
        return cls(
            dataset_version=dataset_version,
            feature_version=feature_version,
            total_labeled_rows=total_labeled_rows if total_labeled_rows is not None else len(rows),
            rows=rows,
            newest_label_time=max(timestamps) if timestamps else None,
            data_range=data_range,
            metadata=metadata or {},
        )

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return {
            "dataset_version": self.dataset_version,
            "feature_version": self.feature_version,
            "total_labeled_rows": self.total_labeled_rows,
            "rows_loaded": len(self.rows),
            "newest_label_time": self.newest_label_time.isoformat() if self.newest_label_time else None,
            "data_range": self.data_range.model_dump() if self.data_range else None,
            "metadata": dict(self.metadata),
        }

    to_dict = model_dump


@dataclass(frozen=True, slots=True)
class CandidateArtifact:
    """Portable result from a local train/package callback."""

    candidate_id: str
    model_version: str | None = None
    artifact_path: str | Path | None = None
    checksum: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "model_version": self.model_version,
            "artifact_path": str(self.artifact_path) if self.artifact_path is not None else None,
            "checksum": self.checksum,
            "metadata": dict(self.metadata),
        }

    to_dict = model_dump


@dataclass(frozen=True, slots=True)
class ResearchSchedulerConfig:
    """Safe operational configuration for a local research process."""

    minimum_new_labeled_rows: int = 100
    max_candidates_per_run: int = 50
    poll_interval_seconds: float = 300.0
    reports_dir: Path = Path("research_reports")
    state_path: Path = Path("research_reports/scheduler_state.json")
    code_version: str = "unknown"
    allow_upload: bool = False
    allow_auto_champion_promotion: bool = False

    def __post_init__(self) -> None:
        if self.minimum_new_labeled_rows <= 0:
            raise ResearchValidationError("minimum_new_labeled_rows must be positive")
        if self.max_candidates_per_run <= 0:
            raise ResearchValidationError("max_candidates_per_run must be positive")
        if self.poll_interval_seconds <= 0:
            raise ResearchValidationError("poll_interval_seconds must be positive")
        if not self.code_version:
            raise ResearchValidationError("code_version cannot be empty")
        object.__setattr__(self, "reports_dir", Path(self.reports_dir))
        object.__setattr__(self, "state_path", Path(self.state_path))

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return {
            "minimum_new_labeled_rows": self.minimum_new_labeled_rows,
            "max_candidates_per_run": self.max_candidates_per_run,
            "poll_interval_seconds": self.poll_interval_seconds,
            "reports_dir": str(self.reports_dir),
            "state_path": str(self.state_path),
            "code_version": self.code_version,
            "allow_upload": self.allow_upload,
            "allow_auto_champion_promotion": self.allow_auto_champion_promotion,
        }


@dataclass(frozen=True, slots=True)
class ResearchSchedulerState:
    """Small durable cursor used to detect newly labeled data."""

    processed_labeled_rows: int = 0
    dataset_version: str | None = None
    newest_label_time: datetime | str | None = None
    last_run_at: datetime | str | None = None
    last_status: str = "never_run"
    last_error: str | None = None

    def __post_init__(self) -> None:
        if self.processed_labeled_rows < 0:
            raise ResearchValidationError("processed_labeled_rows cannot be negative")
        newest = ensure_utc(self.newest_label_time, field_name="newest_label_time") if self.newest_label_time else None
        last_run = ensure_utc(self.last_run_at, field_name="last_run_at") if self.last_run_at else None
        object.__setattr__(self, "newest_label_time", newest)
        object.__setattr__(self, "last_run_at", last_run)

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return {
            "processed_labeled_rows": self.processed_labeled_rows,
            "dataset_version": self.dataset_version,
            "newest_label_time": self.newest_label_time.isoformat() if self.newest_label_time else None,
            "last_run_at": self.last_run_at.isoformat() if self.last_run_at else None,
            "last_status": self.last_status,
            "last_error": self.last_error,
        }

    to_dict = model_dump

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResearchSchedulerState":
        return cls(
            processed_labeled_rows=int(value.get("processed_labeled_rows", 0)),
            dataset_version=value.get("dataset_version"),
            newest_label_time=value.get("newest_label_time"),
            last_run_at=value.get("last_run_at"),
            last_status=str(value.get("last_status", "never_run")),
            last_error=value.get("last_error"),
        )


@dataclass(frozen=True, slots=True)
class ResearchRunResult:
    """Auditable outcome of a single scheduler poll/run."""

    status: str
    snapshot: LabeledDataSnapshot
    new_labeled_rows: int
    reports: tuple[ExperimentReport, ...] = ()
    registered_challengers: tuple[Any, ...] = ()
    uploads: tuple[Any, ...] = ()
    promotions: tuple[Any, ...] = ()
    messages: tuple[str, ...] = ()

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return {
            "status": self.status,
            "snapshot": self.snapshot.model_dump(),
            "new_labeled_rows": self.new_labeled_rows,
            "reports": [report.model_dump() for report in self.reports],
            "registered_challengers": list(self.registered_challengers),
            "uploads": list(self.uploads),
            "promotions": list(self.promotions),
            "messages": list(self.messages),
        }

    to_dict = model_dump


def detect_new_labeled_rows(snapshot: LabeledDataSnapshot, state: ResearchSchedulerState) -> int:
    """Return rows not processed by the scheduler's durable cursor.

    A new dataset version is treated as a fresh local snapshot. This protects a
    restored/exported data lake whose row count is lower than a previous one.
    """

    if state.dataset_version != snapshot.dataset_version:
        return snapshot.total_labeled_rows
    return max(0, snapshot.total_labeled_rows - state.processed_labeled_rows)


def _load_state(path: Path) -> ResearchSchedulerState:
    if not path.exists():
        return ResearchSchedulerState()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("state is not an object")
        return ResearchSchedulerState.from_dict(payload)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ResearchValidationError(f"cannot load research scheduler state {path}: {exc}") from exc


def _write_state(path: Path, state: ResearchSchedulerState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(stable_json_dumps(state.model_dump()) + "\n", encoding="utf-8")
    temporary.replace(path)


def _invoke(callback: Callable[..., Any], *args: Any) -> Any:
    """Call a documented callback while accommodating narrower test adapters."""

    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return callback(*args)
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(parameter.kind is parameter.VAR_POSITIONAL for parameter in signature.parameters.values())
    if has_varargs:
        return callback(*args)
    required = [parameter for parameter in positional if parameter.default is inspect.Parameter.empty]
    if len(required) > len(args):
        raise ResearchValidationError(
            f"callback {getattr(callback, '__name__', type(callback).__name__)} requires {len(required)} arguments"
        )
    return callback(*args[: len(positional)])


def _as_candidate_artifact(candidate: CandidateStrategySpec, value: Any) -> CandidateArtifact:
    if isinstance(value, CandidateArtifact):
        if value.candidate_id != candidate.candidate_id:
            raise ResearchValidationError("trainer returned an artifact for a different candidate")
        return value
    if value is None:
        return CandidateArtifact(candidate_id=candidate.candidate_id)
    if isinstance(value, (str, Path)):
        return CandidateArtifact(candidate_id=candidate.candidate_id, artifact_path=value)
    if isinstance(value, Mapping):
        return CandidateArtifact(
            candidate_id=str(value.get("candidate_id", candidate.candidate_id)),
            model_version=value.get("model_version"),
            artifact_path=value.get("artifact_path", value.get("path")),
            checksum=value.get("checksum"),
            metadata=value.get("metadata", {}),
        )
    return CandidateArtifact(candidate_id=candidate.candidate_id, metadata={"trainer_result_type": type(value).__name__})


class ResearchScheduler:
    """Run bounded, local candidate research with a durable data cursor.

    The scheduler deliberately makes no strategy-quality promotion decision.
    Registration, upload, and promotion are separate callbacks, with upload and
    promotion disabled by default. This preserves the champion/challenger gate.
    """

    def __init__(
        self,
        *,
        snapshot_provider: DatasetSnapshotProvider,
        candidate_provider: CandidateProvider | None = None,
        trainer: CandidateTrainer | None = None,
        evaluator: CandidateEvaluator | None = None,
        artifact_packager: Callable[[CandidateStrategySpec, CandidateArtifact, ExperimentReport, LabeledDataSnapshot], Any]
        | None = None,
        challenger_registrar: Callable[[CandidateStrategySpec, CandidateArtifact, ExperimentReport, LabeledDataSnapshot], Any]
        | None = None,
        uploader: Callable[[CandidateStrategySpec, CandidateArtifact, ExperimentReport, LabeledDataSnapshot], Any] | None = None,
        promoter: Callable[[CandidateStrategySpec, CandidateArtifact, ExperimentReport, LabeledDataSnapshot], Any] | None = None,
        config: ResearchSchedulerConfig | None = None,
    ) -> None:
        self.snapshot_provider = snapshot_provider
        self.candidate_provider = candidate_provider
        self.trainer = trainer
        self.evaluator = evaluator
        self.artifact_packager = artifact_packager
        self.challenger_registrar = challenger_registrar
        self.uploader = uploader
        self.promoter = promoter
        self.config = config or ResearchSchedulerConfig()
        self._lock = threading.Lock()
        self.state = _load_state(self.config.state_path)

    def reload_state(self) -> ResearchSchedulerState:
        """Reload the persistent cursor, useful after an operator intervention."""

        self.state = _load_state(self.config.state_path)
        return self.state

    def _report_path(self, definition: ExperimentDefinition) -> Path:
        return self.config.reports_dir / f"{definition.experiment_id}.json"

    def _experiment_definition(
        self,
        candidate: CandidateStrategySpec,
        snapshot: LabeledDataSnapshot,
    ) -> ExperimentDefinition:
        discriminator = stable_fingerprint(
            {
                "candidate": candidate.model_dump(),
                "dataset_version": snapshot.dataset_version,
                "feature_version": snapshot.feature_version,
                "code_version": self.config.code_version,
            }
        )[:12]
        prefix = candidate.candidate_id[:90]
        experiment_id = f"research-{prefix}-{discriminator}"[:128]
        return ExperimentDefinition(
            experiment_id=experiment_id,
            candidate=candidate,
            dataset_version=snapshot.dataset_version,
            feature_version=snapshot.feature_version,
            model_version=None,
            code_version=self.config.code_version,
            random_seed=candidate.random_seed,
            train_period=snapshot.data_range,
            configuration={
                "scheduler": self.config.model_dump(),
                "snapshot": snapshot.model_dump(),
            },
            notes="Generated by local research scheduler; no automatic champion activation.",
        )

    def _default_evaluate(
        self,
        candidate: CandidateStrategySpec,
        artifact: CandidateArtifact,
        snapshot: LabeledDataSnapshot,
        definition: ExperimentDefinition,
        started_at: datetime,
    ) -> ExperimentReport:
        if not snapshot.rows:
            raise ResearchValidationError(
                "no evaluator was supplied and snapshot has no loaded rows with prediction/actual_return fields"
            )
        result = evaluate_observations(snapshot.rows)
        artifacts = [artifact.artifact_path] if artifact.artifact_path else []
        return make_experiment_report(
            definition,
            result,
            artifacts=artifacts,
            notes=("Evaluated stored predictions; no retraining callback was configured.",),
            started_at=started_at,
            metadata={"candidate_artifact": artifact.model_dump()},
        )

    def _coerce_report(
        self,
        value: Any,
        *,
        definition: ExperimentDefinition,
        artifact: CandidateArtifact,
        started_at: datetime,
    ) -> ExperimentReport:
        if isinstance(value, ExperimentReport):
            # Preserve a rich evaluator report but bind it to this audited definition.
            if value.definition.experiment_id == definition.experiment_id:
                return value
            return ExperimentReport(
                definition=definition,
                status=value.status,
                metrics=value.metrics,
                fold_metrics=value.fold_metrics,
                artifacts=tuple(
                    dict.fromkeys(
                        [
                            *value.artifacts,
                            *([str(artifact.artifact_path)] if artifact.artifact_path else []),
                        ]
                    )
                ),
                notes=value.notes,
                started_at=value.started_at or started_at,
                finished_at=value.finished_at or utc_now(),
                error=value.error,
                metadata={**dict(value.metadata), "candidate_artifact": artifact.model_dump()},
            )
        if isinstance(value, WalkForwardResult):
            fold_metrics = [evaluation.metrics for evaluation in value.fold_evaluations]
            return make_experiment_report(
                definition,
                value.evaluation,
                fold_metrics=fold_metrics,
                artifacts=[artifact.artifact_path] if artifact.artifact_path else [],
                started_at=started_at,
                metadata={"candidate_artifact": artifact.model_dump(), "walk_forward": value.model_dump()},
            )
        if isinstance(value, EvaluationResult):
            return make_experiment_report(
                definition,
                value,
                artifacts=[artifact.artifact_path] if artifact.artifact_path else [],
                started_at=started_at,
                metadata={"candidate_artifact": artifact.model_dump()},
            )
        if isinstance(value, Mapping):
            return ExperimentReport(
                definition=definition,
                status=ExperimentStatus.COMPLETED,
                metrics=value,
                artifacts=[artifact.artifact_path] if artifact.artifact_path else [],
                started_at=started_at,
                finished_at=utc_now(),
                metadata={"candidate_artifact": artifact.model_dump()},
            )
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            observations = [
                item if isinstance(item, EvaluationObservation) else item
                for item in value
            ]
            result = evaluate_observations(observations)
            return make_experiment_report(
                definition,
                result,
                artifacts=[artifact.artifact_path] if artifact.artifact_path else [],
                started_at=started_at,
                metadata={"candidate_artifact": artifact.model_dump()},
            )
        raise ResearchValidationError(
            "candidate evaluator must return ExperimentReport, WalkForwardResult, EvaluationResult, metrics mapping, or rows"
        )

    def _failed_report(
        self,
        definition: ExperimentDefinition,
        error: Exception,
        started_at: datetime,
    ) -> ExperimentReport:
        return ExperimentReport(
            definition=definition,
            status=ExperimentStatus.FAILED,
            error=f"{type(error).__name__}: {error}",
            started_at=started_at,
            finished_at=utc_now(),
            notes=("Candidate failure was isolated; no champion state was changed.",),
        )

    def run_once(
        self,
        *,
        force: bool = False,
        candidates: Iterable[CandidateStrategySpec] | None = None,
    ) -> ResearchRunResult:
        """Detect data, run bounded candidates, and persist reports/state safely."""

        if not self._lock.acquire(blocking=False):
            raise ResearchValidationError("research scheduler is already running")
        try:
            snapshot = _invoke(self.snapshot_provider)
            if not isinstance(snapshot, LabeledDataSnapshot):
                raise ResearchValidationError("snapshot_provider must return LabeledDataSnapshot")
            new_rows = detect_new_labeled_rows(snapshot, self.state)
            if not force and new_rows < self.config.minimum_new_labeled_rows:
                message = (
                    f"waiting for labeled data: {new_rows} new rows; "
                    f"minimum is {self.config.minimum_new_labeled_rows}"
                )
                self.state = ResearchSchedulerState(
                    processed_labeled_rows=self.state.processed_labeled_rows,
                    dataset_version=self.state.dataset_version,
                    newest_label_time=self.state.newest_label_time,
                    last_run_at=utc_now(),
                    last_status="waiting_for_labels",
                    last_error=None,
                )
                _write_state(self.config.state_path, self.state)
                return ResearchRunResult(
                    status="waiting_for_labels",
                    snapshot=snapshot,
                    new_labeled_rows=new_rows,
                    messages=(message,),
                )

            raw_candidates = candidates if candidates is not None else (
                _invoke(self.candidate_provider, snapshot) if self.candidate_provider is not None else ()
            )
            candidate_list = list(raw_candidates)
            if not candidate_list:
                self.state = ResearchSchedulerState(
                    processed_labeled_rows=snapshot.total_labeled_rows,
                    dataset_version=snapshot.dataset_version,
                    newest_label_time=snapshot.newest_label_time,
                    last_run_at=utc_now(),
                    last_status="no_candidates",
                    last_error=None,
                )
                _write_state(self.config.state_path, self.state)
                return ResearchRunResult(
                    status="no_candidates",
                    snapshot=snapshot,
                    new_labeled_rows=new_rows,
                    messages=("No candidate specifications were supplied.",),
                )
            if len(candidate_list) > self.config.max_candidates_per_run:
                raise ResearchValidationError(
                    f"candidate run has {len(candidate_list)} entries; limit is {self.config.max_candidates_per_run}"
                )
            if any(not isinstance(candidate, CandidateStrategySpec) for candidate in candidate_list):
                raise ResearchValidationError("all candidates must be CandidateStrategySpec instances")
            fingerprints = [candidate.fingerprint for candidate in candidate_list]
            if len(fingerprints) != len(set(fingerprints)):
                raise ResearchValidationError("candidate run contains duplicate declarative specifications")

            reports: list[ExperimentReport] = []
            registrations: list[Any] = []
            uploads: list[Any] = []
            promotions: list[Any] = []
            messages: list[str] = []
            for candidate in candidate_list:
                started_at = utc_now()
                definition = self._experiment_definition(candidate, snapshot)
                try:
                    artifact = _as_candidate_artifact(
                        candidate,
                        _invoke(self.trainer, candidate, snapshot) if self.trainer is not None else None,
                    )
                    raw_result = (
                        _invoke(self.evaluator, candidate, artifact, snapshot)
                        if self.evaluator is not None
                        else None
                    )
                    report = (
                        self._coerce_report(raw_result, definition=definition, artifact=artifact, started_at=started_at)
                        if raw_result is not None
                        else self._default_evaluate(candidate, artifact, snapshot, definition, started_at)
                    )
                    if self.artifact_packager is not None and report.status is ExperimentStatus.COMPLETED:
                        package_result = _invoke(self.artifact_packager, candidate, artifact, report, snapshot)
                        package_paths: list[str] = []
                        if isinstance(package_result, (str, Path)):
                            package_paths = [str(package_result)]
                        elif isinstance(package_result, Sequence) and not isinstance(package_result, (str, bytes)):
                            package_paths = [str(item) for item in package_result]
                        elif isinstance(package_result, Mapping):
                            package_paths = [str(item) for item in package_result.get("artifacts", [])]
                        if package_paths:
                            report = ExperimentReport(
                                definition=report.definition,
                                status=report.status,
                                metrics=report.metrics,
                                fold_metrics=report.fold_metrics,
                                artifacts=tuple(dict.fromkeys([*report.artifacts, *package_paths])),
                                notes=report.notes,
                                started_at=report.started_at,
                                finished_at=report.finished_at,
                                error=report.error,
                                metadata=report.metadata,
                            )
                    report.write_json(self._report_path(definition))
                    reports.append(report)
                    if report.status is ExperimentStatus.COMPLETED and self.challenger_registrar is not None:
                        registrations.append(_invoke(self.challenger_registrar, candidate, artifact, report, snapshot))
                    if report.status is ExperimentStatus.COMPLETED and self.config.allow_upload and self.uploader is not None:
                        uploads.append(_invoke(self.uploader, candidate, artifact, report, snapshot))
                    if (
                        report.status is ExperimentStatus.COMPLETED
                        and self.config.allow_auto_champion_promotion
                        and self.promoter is not None
                    ):
                        promotions.append(_invoke(self.promoter, candidate, artifact, report, snapshot))
                except Exception as exc:  # Candidate failures must not block other local experiments.
                    report = self._failed_report(definition, exc, started_at)
                    report.write_json(self._report_path(definition))
                    reports.append(report)
                    messages.append(f"{candidate.candidate_id}: {report.error}")

            status = "completed" if any(report.status is ExperimentStatus.COMPLETED for report in reports) else "failed"
            self.state = ResearchSchedulerState(
                processed_labeled_rows=snapshot.total_labeled_rows,
                dataset_version=snapshot.dataset_version,
                newest_label_time=snapshot.newest_label_time,
                last_run_at=utc_now(),
                last_status=status,
                last_error="; ".join(messages) if messages else None,
            )
            _write_state(self.config.state_path, self.state)
            if not self.config.allow_auto_champion_promotion:
                messages.append("Automatic champion promotion remains disabled.")
            return ResearchRunResult(
                status=status,
                snapshot=snapshot,
                new_labeled_rows=new_rows,
                reports=tuple(reports),
                registered_challengers=tuple(registrations),
                uploads=tuple(uploads),
                promotions=tuple(promotions),
                messages=tuple(messages),
            )
        finally:
            self._lock.release()

    def run_forever(
        self,
        *,
        stop_event: threading.Event | None = None,
        max_iterations: int | None = None,
    ) -> None:
        """Poll locally until stopped; suitable for a CLI process, never web requests."""

        event = stop_event or threading.Event()
        iterations = 0
        while not event.is_set():
            self.run_once()
            iterations += 1
            if max_iterations is not None and iterations >= max_iterations:
                return
            event.wait(self.config.poll_interval_seconds)


LocalResearchScheduler = ResearchScheduler
