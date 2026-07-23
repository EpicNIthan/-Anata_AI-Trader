"""Leakage-aware historical evaluation and signal-independence analysis.

The module intentionally uses the standard library only.  It accepts plain
dict rows, dataclasses, or :class:`~app.research.schemas.EvaluationObservation`
instances so local research scripts do not need a dataframe dependency.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import math
from pathlib import Path
import statistics
from typing import Any, TypeAlias

from app.research.schemas import (
    EvaluationObservation,
    ExperimentDefinition,
    ExperimentReport,
    ExperimentStatus,
    JSONValue,
    ResearchValidationError,
    TimeRange,
    WindowSize,
    ensure_utc,
    parse_window,
    utc_now,
)


Record: TypeAlias = Any
TimestampGetter: TypeAlias = Callable[[Record], datetime]
ValueGetter: TypeAlias = Callable[[Record], Any]
_MISSING = object()
_YEAR_SECONDS = 365.25 * 24 * 60 * 60


def _row_value(row: Record, *names: str, default: Any = _MISSING) -> Any:
    """Read a field from a mapping, dataclass-like object, or SQLAlchemy row."""

    for name in names:
        if isinstance(row, Mapping) and name in row:
            return row[name]
        value = getattr(row, name, _MISSING)
        if value is not _MISSING:
            return value
    if default is _MISSING:
        raise ResearchValidationError(f"row is missing one of required fields: {', '.join(names)}")
    return default


def row_timestamp(row: Record) -> datetime:
    """Return the point-in-time timestamp for a generic research row.

    Explicit decision timestamps win.  A raw feature row with only ``as_of``
    and ``available_to_model_time`` is ordered by availability, because that is
    the earliest safe time it may enter a model.
    """

    explicit = _row_value(row, "timestamp", "decision_time", "signal_time", default=None)
    if explicit is not None:
        return ensure_utc(explicit, field_name="timestamp")
    available = _row_value(row, "available_to_model_time", default=None)
    if available is not None:
        return ensure_utc(available, field_name="available_to_model_time")
    value = _row_value(row, "as_of", "event_time", "time", "created_at", default=None)
    if value is None:
        raise ResearchValidationError(
            "row needs timestamp, decision_time, as_of, event_time, or available_to_model_time"
        )
    return ensure_utc(value, field_name="timestamp")


def _as_finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ResearchValidationError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ResearchValidationError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ResearchValidationError(f"{name} must be finite")
    return result


def _as_non_negative_window(value: WindowSize | str | int | timedelta, name: str) -> WindowSize | int:
    """Parse split gaps, allowing zero in addition to normal positive windows."""

    if value == 0:
        return 0
    return parse_window(value, field_name=name)


def _sort_indices(records: Sequence[Record], timestamp_getter: TimestampGetter | None = None) -> list[tuple[int, datetime]]:
    getter = timestamp_getter or row_timestamp
    ordered = [(index, ensure_utc(getter(record), field_name="timestamp")) for index, record in enumerate(records)]
    # Python's stable sort retains source order for multiple symbols at the same timestamp.
    ordered.sort(key=lambda item: item[1])
    return ordered


def assert_point_in_time_availability(
    records: Sequence[Record],
    *,
    timestamp_getter: TimestampGetter | None = None,
    availability_getter: ValueGetter | None = None,
) -> None:
    """Fail when a row claims it was available after its model decision time."""

    timestamp_getter = timestamp_getter or row_timestamp
    for position, record in enumerate(records):
        decision_time = ensure_utc(timestamp_getter(record), field_name="timestamp")
        available = (
            availability_getter(record)
            if availability_getter is not None
            else _row_value(record, "available_to_model_time", default=None)
        )
        if available is not None and ensure_utc(available, field_name="available_to_model_time") > decision_time:
            raise ResearchValidationError(
                f"row {position} has available_to_model_time after its decision timestamp; refusing leakage"
            )


def _period_for_positions(ordered: Sequence[tuple[int, datetime]], positions: Sequence[int]) -> TimeRange | None:
    if not positions:
        return None
    first = ordered[positions[0]][1]
    last = ordered[positions[-1]][1]
    # All timestamps can be identical across symbols; preserve a valid half-open range.
    return TimeRange(start=first, end=last + timedelta(microseconds=1))


def _advance_group_boundary(ordered: Sequence[tuple[int, datetime]], boundary: int) -> int:
    """Move a split edge after a same-timestamp group so it cannot straddle sets."""

    if boundary <= 0 or boundary >= len(ordered):
        return boundary
    timestamp = ordered[boundary - 1][1]
    while boundary < len(ordered) and ordered[boundary][1] == timestamp:
        boundary += 1
    return boundary


@dataclass(frozen=True, slots=True)
class ChronologicalSplit:
    """Indices for non-overlapping, ordered train/validation/test partitions."""

    ordered_indices: tuple[int, ...]
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    train_period: TimeRange | None = None
    validation_period: TimeRange | None = None
    test_period: TimeRange | None = None

    @property
    def train(self) -> tuple[int, ...]:
        return self.train_indices

    @property
    def validation(self) -> tuple[int, ...]:
        return self.validation_indices

    @property
    def test(self) -> tuple[int, ...]:
        return self.test_indices

    def materialize(self, records: Sequence[Record]) -> tuple[list[Record], list[Record], list[Record]]:
        return (
            [records[index] for index in self.train_indices],
            [records[index] for index in self.validation_indices],
            [records[index] for index in self.test_indices],
        )

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return {
            "ordered_indices": list(self.ordered_indices),
            "train_indices": list(self.train_indices),
            "validation_indices": list(self.validation_indices),
            "test_indices": list(self.test_indices),
            "train_period": self.train_period.model_dump() if self.train_period else None,
            "validation_period": self.validation_period.model_dump() if self.validation_period else None,
            "test_period": self.test_period.model_dump() if self.test_period else None,
        }


def chronological_split(
    records: Sequence[Record],
    *,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    test_fraction: float | None = None,
    timestamp_getter: TimestampGetter | None = None,
) -> ChronologicalSplit:
    """Create a stable train/validation/test split without time-series shuffling.

    Rows that share a timestamp (for example several symbols at one candle) are
    never separated across a boundary.  Fractions therefore represent targets,
    not a promise of an exact count.
    """

    if len(records) < 3:
        raise ResearchValidationError("chronological_split requires at least three rows")
    assert_point_in_time_availability(records, timestamp_getter=timestamp_getter)
    train_fraction = _as_finite_float(train_fraction, "train_fraction")
    validation_fraction = _as_finite_float(validation_fraction, "validation_fraction")
    if test_fraction is None:
        test_fraction = 1.0 - train_fraction - validation_fraction
    test_fraction = _as_finite_float(test_fraction, "test_fraction")
    if min(train_fraction, validation_fraction, test_fraction) < 0:
        raise ResearchValidationError("split fractions cannot be negative")
    if abs(train_fraction + validation_fraction + test_fraction - 1.0) > 1e-9:
        raise ResearchValidationError("train_fraction + validation_fraction + test_fraction must equal 1")
    if train_fraction <= 0 or test_fraction <= 0:
        raise ResearchValidationError("train and test fractions must be positive")

    ordered = _sort_indices(records, timestamp_getter)
    count = len(ordered)
    train_end = _advance_group_boundary(ordered, max(1, int(math.floor(count * train_fraction))))
    validation_end = _advance_group_boundary(
        ordered,
        max(train_end, int(math.floor(count * (train_fraction + validation_fraction)))),
    )
    if validation_fraction == 0:
        validation_end = train_end
    if train_end >= count or validation_end >= count:
        raise ResearchValidationError("split fractions leave no chronological test observations")
    if validation_fraction > 0 and validation_end <= train_end:
        raise ResearchValidationError("split fractions leave no validation observations after timestamp grouping")

    train_positions = tuple(range(0, train_end))
    validation_positions = tuple(range(train_end, validation_end))
    test_positions = tuple(range(validation_end, count))
    return ChronologicalSplit(
        ordered_indices=tuple(index for index, _ in ordered),
        train_indices=tuple(ordered[position][0] for position in train_positions),
        validation_indices=tuple(ordered[position][0] for position in validation_positions),
        test_indices=tuple(ordered[position][0] for position in test_positions),
        train_period=_period_for_positions(ordered, train_positions),
        validation_period=_period_for_positions(ordered, validation_positions),
        test_period=_period_for_positions(ordered, test_positions),
    )


@dataclass(frozen=True, slots=True)
class PurgedFold:
    """A chronological fold with label-overlap purges and pre-test embargo."""

    fold_number: int
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    purged_indices: tuple[int, ...] = ()
    embargoed_indices: tuple[int, ...] = ()
    train_period: TimeRange | None = None
    test_period: TimeRange | None = None

    def materialize(self, records: Sequence[Record]) -> tuple[list[Record], list[Record]]:
        return ([records[index] for index in self.train_indices], [records[index] for index in self.test_indices])

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return {
            "fold_number": self.fold_number,
            "train_indices": list(self.train_indices),
            "test_indices": list(self.test_indices),
            "purged_indices": list(self.purged_indices),
            "embargoed_indices": list(self.embargoed_indices),
            "train_period": self.train_period.model_dump() if self.train_period else None,
            "test_period": self.test_period.model_dump() if self.test_period else None,
        }


def _is_within_window(
    *,
    earlier_position: int,
    later_position: int,
    ordered: Sequence[tuple[int, datetime]],
    window: WindowSize | int,
) -> bool:
    if window == 0:
        return False
    if isinstance(window, int):
        return later_position - earlier_position <= window
    return ordered[later_position][1] - ordered[earlier_position][1] <= window


@dataclass(slots=True)
class PurgedTimeSeriesSplit:
    """Expanding, strictly-past time-series folds with purge and embargo controls.

    Training only uses rows before each test block. ``purge_window`` removes the
    rows nearest the test block; ``embargo_window`` adds a second conservative
    pre-test gap.  Rows whose label availability extends into the test period
    are always removed, even when both gaps are zero.
    """

    n_splits: int = 5
    min_train_size: int = 1
    test_size: int | None = None
    purge_window: WindowSize | str | int = 0
    embargo_window: WindowSize | str | int = 0
    timestamp_getter: TimestampGetter | None = None
    label_end_getter: ValueGetter | None = None

    def __post_init__(self) -> None:
        if self.n_splits <= 0:
            raise ResearchValidationError("n_splits must be positive")
        if self.min_train_size <= 0:
            raise ResearchValidationError("min_train_size must be positive")
        if self.test_size is not None and self.test_size <= 0:
            raise ResearchValidationError("test_size must be positive when provided")
        self.purge_window = _as_non_negative_window(self.purge_window, "purge_window")
        self.embargo_window = _as_non_negative_window(self.embargo_window, "embargo_window")

    def split(self, records: Sequence[Record]) -> Iterator[PurgedFold]:
        """Yield ordered folds. No future row can appear in a fold's training set."""

        if len(records) <= self.min_train_size:
            raise ResearchValidationError("not enough rows for requested minimum training size")
        assert_point_in_time_availability(records, timestamp_getter=self.timestamp_getter)
        ordered = _sort_indices(records, self.timestamp_getter)
        count = len(ordered)
        test_size = self.test_size or max(1, (count - self.min_train_size) // self.n_splits)
        first_test_start = count - test_size * self.n_splits
        if first_test_start < self.min_train_size:
            first_test_start = self.min_train_size

        produced = 0
        for test_start in range(first_test_start, count, test_size):
            if produced >= self.n_splits:
                break
            test_end = min(count, test_start + test_size)
            if test_end <= test_start:
                continue
            test_time = ordered[test_start][1]
            candidate_positions = range(0, test_start)
            train_positions: list[int] = []
            purged_positions: list[int] = []
            embargoed_positions: list[int] = []
            for position in candidate_positions:
                record = records[ordered[position][0]]
                label_end = (
                    self.label_end_getter(record)
                    if self.label_end_getter is not None
                    else _row_value(record, "label_available_time", "label_end_time", "label_end", default=None)
                )
                if label_end is not None and ensure_utc(label_end, field_name="label_end_time") > test_time:
                    purged_positions.append(position)
                    continue
                if _is_within_window(
                    earlier_position=position,
                    later_position=test_start,
                    ordered=ordered,
                    window=self.embargo_window,
                ):
                    embargoed_positions.append(position)
                    continue
                if _is_within_window(
                    earlier_position=position,
                    later_position=test_start,
                    ordered=ordered,
                    window=self.purge_window,
                ):
                    purged_positions.append(position)
                    continue
                train_positions.append(position)
            if len(train_positions) < self.min_train_size:
                # A fold without enough leakage-safe history is not an evaluable fold.
                continue
            test_positions = tuple(range(test_start, test_end))
            yield PurgedFold(
                fold_number=produced,
                train_indices=tuple(ordered[position][0] for position in train_positions),
                test_indices=tuple(ordered[position][0] for position in test_positions),
                purged_indices=tuple(ordered[position][0] for position in purged_positions),
                embargoed_indices=tuple(ordered[position][0] for position in embargoed_positions),
                train_period=_period_for_positions(ordered, train_positions),
                test_period=_period_for_positions(ordered, test_positions),
            )
            produced += 1

    def get_n_splits(self, records: Sequence[Record] | None = None) -> int:
        """sklearn-compatible helper; actual count may be smaller after purging."""

        return self.n_splits


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _sample_stdev(values: Sequence[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right):
        raise ResearchValidationError("correlation inputs must have equal lengths")
    if len(left) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_norm = math.sqrt(sum((a - left_mean) ** 2 for a in left))
    right_norm = math.sqrt(sum((b - right_mean) ** 2 for b in right))
    if left_norm == 0 or right_norm == 0:
        return None
    return max(-1.0, min(1.0, numerator / (left_norm * right_norm)))


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Return stable average ranks (the tie convention used by Spearman IC)."""

    sorted_indices = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(sorted_indices):
        end = cursor + 1
        value = values[sorted_indices[cursor]]
        while end < len(sorted_indices) and values[sorted_indices[end]] == value:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for index in sorted_indices[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    if not 0.0 <= probability <= 1.0:
        raise ResearchValidationError("quantile probability must be in [0, 1]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _drawdown_path(returns: Sequence[float]) -> tuple[list[float], list[float]]:
    equity = 1.0
    peak = 1.0
    equity_curve: list[float] = []
    drawdowns: list[float] = []
    for value in returns:
        # A simulated account cannot have negative equity. Values below -100%
        # are retained in raw returns but collapse the compounded path to zero.
        equity *= max(0.0, 1.0 + value)
        peak = max(peak, equity)
        drawdown = (peak - equity) / peak if peak > 0 else 0.0
        equity_curve.append(equity)
        drawdowns.append(drawdown)
    return equity_curve, drawdowns


def _infer_annualization_factor(timestamps: Sequence[datetime] | None) -> float:
    if timestamps is None or len(timestamps) < 2:
        return 1.0
    intervals = [
        (later - earlier).total_seconds()
        for earlier, later in zip(timestamps, timestamps[1:])
        if later > earlier
    ]
    if not intervals:
        return 1.0
    median_seconds = statistics.median(intervals)
    return max(1.0, _YEAR_SECONDS / median_seconds)


def _sign(value: float) -> float:
    if value > 0:
        return 1.0
    if value < 0:
        return -1.0
    return 0.0


def _average_holding_seconds(
    positions: Sequence[float], timestamps: Sequence[datetime] | None, explicit_holding_seconds: Sequence[float | None] | None,
) -> float | None:
    explicit = [value for value in (explicit_holding_seconds or []) if value is not None]
    if explicit:
        return statistics.fmean(explicit)
    if not positions:
        return None
    durations: list[float] = []
    start: int | None = None
    active_sign = 0.0
    for index, position in enumerate(positions):
        sign = _sign(position)
        if sign != 0 and (start is None or sign != active_sign):
            if start is not None:
                if timestamps is not None:
                    durations.append(max(0.0, (timestamps[index] - timestamps[start]).total_seconds()))
                else:
                    durations.append(float(index - start))
            start = index
            active_sign = sign
        elif sign == 0 and start is not None:
            if timestamps is not None:
                durations.append(max(0.0, (timestamps[index] - timestamps[start]).total_seconds()))
            else:
                durations.append(float(index - start))
            start = None
            active_sign = 0.0
    if start is not None:
        if timestamps is not None and len(timestamps) > start + 1:
            durations.append(max(0.0, (timestamps[-1] - timestamps[start]).total_seconds()))
        else:
            durations.append(float(len(positions) - start))
    return statistics.fmean(durations) if durations else None


def evaluate_predictions(
    predictions: Sequence[float],
    actual_returns: Sequence[float],
    *,
    positions: Sequence[float] | None = None,
    transaction_costs: Sequence[float] | None = None,
    timestamps: Sequence[datetime | str] | None = None,
    holding_seconds: Sequence[float | None] | None = None,
    annualization_factor: float | None = None,
) -> dict[str, float | int | None]:
    """Calculate signal and paper-simulation metrics from aligned observations.

    ``prediction`` drives direction when no explicit position is provided. Cost
    values are per-observation fractions of equity, and are subtracted from gross
    strategy returns. All ratio metrics return ``None`` when mathematically
    undefined rather than leaking NaN/Infinity into a report or database JSON.
    """

    if len(predictions) != len(actual_returns):
        raise ResearchValidationError("predictions and actual_returns must have equal lengths")
    count = len(predictions)
    if positions is not None and len(positions) != count:
        raise ResearchValidationError("positions must match predictions length")
    if transaction_costs is not None and len(transaction_costs) != count:
        raise ResearchValidationError("transaction_costs must match predictions length")
    if timestamps is not None and len(timestamps) != count:
        raise ResearchValidationError("timestamps must match predictions length")
    if holding_seconds is not None and len(holding_seconds) != count:
        raise ResearchValidationError("holding_seconds must match predictions length")
    if count == 0:
        return {
            "observation_count": 0,
            "active_observation_count": 0,
            "information_coefficient": None,
            "rank_information_coefficient": None,
            "directional_hit_rate": None,
            "gross_expectancy": None,
            "net_expectancy": None,
            "sharpe_ratio": None,
            "sortino_ratio": None,
            "maximum_drawdown": None,
        }

    normalized_predictions = [_as_finite_float(value, "prediction") for value in predictions]
    normalized_actuals = [_as_finite_float(value, "actual_return") for value in actual_returns]
    normalized_positions = (
        [_as_finite_float(value, "position") for value in positions]
        if positions is not None
        else [_sign(value) for value in normalized_predictions]
    )
    normalized_costs = (
        [max(0.0, _as_finite_float(value, "transaction_cost")) for value in transaction_costs]
        if transaction_costs is not None
        else [0.0] * count
    )
    normalized_timestamps = [ensure_utc(value, field_name="timestamp") for value in timestamps] if timestamps else None
    if normalized_timestamps is not None and any(
        later < earlier for earlier, later in zip(normalized_timestamps, normalized_timestamps[1:])
    ):
        raise ResearchValidationError("timestamps must be ordered before metric evaluation")
    if holding_seconds is not None:
        normalized_holding = [
            _as_finite_float(value, "holding_seconds") if value is not None else None for value in holding_seconds
        ]
    else:
        normalized_holding = None

    gross_returns = [position * actual for position, actual in zip(normalized_positions, normalized_actuals)]
    net_returns = [gross - cost for gross, cost in zip(gross_returns, normalized_costs)]
    active_indices = [index for index, position in enumerate(normalized_positions) if position != 0]
    active_gross = [gross_returns[index] for index in active_indices]
    active_net = [net_returns[index] for index in active_indices]
    wins = [value for value in active_net if value > 0]
    losses = [value for value in active_net if value < 0]
    total_wins = sum(wins)
    total_losses = abs(sum(losses))
    profit_factor: float | None
    if total_losses > 0:
        profit_factor = total_wins / total_losses
    else:
        profit_factor = None
    equity_curve, drawdowns = _drawdown_path(net_returns)
    gross_equity_curve, _ = _drawdown_path(gross_returns)
    max_drawdown = max(drawdowns) if drawdowns else None
    annualizer = annualization_factor if annualization_factor is not None else _infer_annualization_factor(normalized_timestamps)
    annualizer = _as_finite_float(annualizer, "annualization_factor")
    if annualizer <= 0:
        raise ResearchValidationError("annualization_factor must be positive")
    net_mean = statistics.fmean(net_returns)
    net_stdev = _sample_stdev(net_returns)
    sharpe = (net_mean / net_stdev * math.sqrt(annualizer)) if net_stdev and net_stdev > 0 else None
    downside = [min(0.0, value) for value in net_returns]
    downside_deviation = math.sqrt(statistics.fmean(value * value for value in downside)) if downside else None
    sortino = (net_mean / downside_deviation * math.sqrt(annualizer)) if downside_deviation and downside_deviation > 0 else None
    total_return = equity_curve[-1] - 1.0
    gross_total_return = gross_equity_curve[-1] - 1.0
    annualized_return: float | None
    if equity_curve[-1] <= 0:
        annualized_return = -1.0
    else:
        annualized_log_return = math.log(equity_curve[-1]) * (annualizer / count)
        # Minute data can imply an astronomically large annualized number. A
        # non-finite headline metric is less useful than an explicit undefined
        # value, while the finite total return remains available for review.
        annualized_return = math.expm1(annualized_log_return) if annualized_log_return < 709.0 else None
    calmar = (
        annualized_return / max_drawdown
        if annualized_return is not None and max_drawdown is not None and max_drawdown > 0
        else None
    )
    var_95 = _quantile(net_returns, 0.05)
    tail = [value for value in net_returns if var_95 is not None and value <= var_95]
    cvar_95 = statistics.fmean(tail) if tail else None
    turnover_total = sum(
        abs(current - previous) for previous, current in zip([0.0, *normalized_positions[:-1]], normalized_positions)
    )
    rank_ic = _pearson(_average_ranks(normalized_predictions), _average_ranks(normalized_actuals))
    hit_rate = (
        sum(_sign(normalized_predictions[index]) == _sign(normalized_actuals[index]) for index in active_indices)
        / len(active_indices)
        if active_indices
        else None
    )
    average_holding = _average_holding_seconds(normalized_positions, normalized_timestamps, normalized_holding)
    return {
        "observation_count": count,
        "active_observation_count": len(active_indices),
        "active_rate": len(active_indices) / count,
        "information_coefficient": _pearson(normalized_predictions, normalized_actuals),
        "rank_information_coefficient": rank_ic,
        "directional_hit_rate": hit_rate,
        "gross_expectancy": _mean(active_gross),
        "net_expectancy": _mean(active_net),
        "gross_return_mean": _mean(gross_returns),
        "net_return_mean": net_mean,
        "average_win": _mean(wins),
        "average_loss": _mean(losses),
        "profit_factor": profit_factor,
        "total_return": total_return,
        "gross_total_return": gross_total_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "maximum_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "turnover": turnover_total,
        "average_turnover": turnover_total / count,
        "exposure": statistics.fmean(abs(value) for value in normalized_positions),
        "average_holding_time_seconds": average_holding,
        "transaction_cost_total": sum(normalized_costs),
        "transaction_cost_mean": statistics.fmean(normalized_costs),
        "value_at_risk_95": var_95,
        "conditional_value_at_risk_95": cvar_95,
        "tail_loss_95": max(0.0, -(cvar_95 or 0.0)),
        "worst_return": min(net_returns),
        "best_return": max(net_returns),
        "annualization_factor": annualizer,
    }


# Conventional aliases make the metric API easy to discover from research notebooks.
calculate_metrics = evaluate_predictions


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Metrics plus curves and segmented views from a point-in-time evaluation."""

    metrics: Mapping[str, Any]
    equity_curve: tuple[float, ...]
    net_returns: tuple[float, ...]
    gross_returns: tuple[float, ...]
    performance_by_symbol: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    performance_by_model_family: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    performance_by_regime: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    performance_by_external_ai: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return {
            "metrics": dict(self.metrics),
            "equity_curve": list(self.equity_curve),
            "net_returns": list(self.net_returns),
            "gross_returns": list(self.gross_returns),
            "performance_by_symbol": {key: dict(value) for key, value in self.performance_by_symbol.items()},
            "performance_by_model_family": {
                key: dict(value) for key, value in self.performance_by_model_family.items()
            },
            "performance_by_regime": {key: dict(value) for key, value in self.performance_by_regime.items()},
            "performance_by_external_ai": {
                key: dict(value) for key, value in self.performance_by_external_ai.items()
            },
        }

    to_dict = model_dump


def observation_from_row(row: Record, *, prediction: float | None = None) -> EvaluationObservation:
    """Coerce a permissive research row into a validated observation.

    Accepted labels include ``actual_return``, ``realized_return``, and
    ``target_return``. This keeps old exported datasets usable while keeping the
    internal evaluator strict and point-in-time checked.
    """

    if isinstance(row, EvaluationObservation) and prediction is None:
        return row
    row_prediction = prediction if prediction is not None else _row_value(row, "prediction", "expected_return")
    actual_return = _row_value(row, "actual_return", "realized_return", "target_return", "label_return", "target")
    timestamp = row_timestamp(row)
    available = _row_value(row, "available_to_model_time", default=timestamp)
    return EvaluationObservation(
        timestamp=timestamp,
        prediction=row_prediction,
        actual_return=actual_return,
        symbol=_row_value(row, "symbol", default=None),
        signal_id=_row_value(row, "signal_id", default=None),
        model_family=_row_value(row, "model_family", default=None),
        regime=_row_value(row, "regime", "market_regime", default=None),
        position=_row_value(row, "position", "signal_position", "exposure", default=None),
        transaction_cost=_row_value(row, "transaction_cost", "expected_cost", "cost", default=0.0),
        holding_seconds=_row_value(row, "holding_seconds", "holding_time_seconds", default=None),
        external_ai_available=_row_value(row, "external_ai_available", default=None),
        available_to_model_time=available,
        label_available_time=_row_value(row, "label_available_time", "label_end_time", "label_end", default=None),
        feature_families=_row_value(row, "feature_families", default=()),
        metadata=_row_value(row, "metadata", default={}),
    )


def _group_metrics(observations: Sequence[EvaluationObservation], attr: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[EvaluationObservation]] = defaultdict(list)
    for observation in observations:
        value = getattr(observation, attr)
        if value is not None and value != "":
            grouped[str(value)].append(observation)
    return {
        key: evaluate_predictions(
            [item.prediction for item in items],
            [item.actual_return for item in items],
            positions=[item.position if item.position is not None else _sign(item.prediction) for item in items],
            transaction_costs=[item.transaction_cost for item in items],
            timestamps=[item.timestamp for item in items],
            holding_seconds=[item.holding_seconds for item in items],
        )
        for key, items in sorted(grouped.items())
    }


def evaluate_observations(
    observations: Sequence[EvaluationObservation | Mapping[str, Any] | Any],
    *,
    annualization_factor: float | None = None,
) -> EvaluationResult:
    """Evaluate point-in-time rows and produce core plus segmented metrics."""

    normalized = [observation_from_row(item) for item in observations]
    normalized.sort(key=lambda item: item.timestamp)
    assert_point_in_time_availability(normalized)
    predictions = [item.prediction for item in normalized]
    actual_returns = [item.actual_return for item in normalized]
    positions = [item.position if item.position is not None else _sign(item.prediction) for item in normalized]
    costs = [item.transaction_cost for item in normalized]
    timestamps = [item.timestamp for item in normalized]
    holdings = [item.holding_seconds for item in normalized]
    metrics = evaluate_predictions(
        predictions,
        actual_returns,
        positions=positions,
        transaction_costs=costs,
        timestamps=timestamps,
        holding_seconds=holdings,
        annualization_factor=annualization_factor,
    )
    gross_returns = [position * actual for position, actual in zip(positions, actual_returns)]
    net_returns = [gross - cost for gross, cost in zip(gross_returns, costs)]
    equity_curve, _ = _drawdown_path(net_returns)
    external_groups = {
        "available": [item for item in normalized if item.external_ai_available is True],
        "unavailable": [item for item in normalized if item.external_ai_available is False],
        "unknown": [item for item in normalized if item.external_ai_available is None],
    }
    external_metrics = {
        key: evaluate_predictions(
            [item.prediction for item in items],
            [item.actual_return for item in items],
            positions=[item.position if item.position is not None else _sign(item.prediction) for item in items],
            transaction_costs=[item.transaction_cost for item in items],
            timestamps=[item.timestamp for item in items],
            holding_seconds=[item.holding_seconds for item in items],
        )
        for key, items in external_groups.items()
        if items
    }
    return EvaluationResult(
        metrics=metrics,
        equity_curve=tuple(equity_curve),
        net_returns=tuple(net_returns),
        gross_returns=tuple(gross_returns),
        performance_by_symbol=_group_metrics(normalized, "symbol"),
        performance_by_model_family=_group_metrics(normalized, "model_family"),
        performance_by_regime=_group_metrics(normalized, "regime"),
        performance_by_external_ai=external_metrics,
    )


def transaction_cost_sensitivity(
    predictions: Sequence[float],
    actual_returns: Sequence[float],
    *,
    cost_rates: Sequence[float] = (0.0, 0.0002, 0.0005, 0.001),
    positions: Sequence[float] | None = None,
    timestamps: Sequence[datetime | str] | None = None,
) -> dict[str, dict[str, float | int | None]]:
    """Measure metric sensitivity to a per-unit-turnover transaction-cost rate."""

    normalized_positions = list(positions) if positions is not None else [_sign(float(value)) for value in predictions]
    if len(normalized_positions) != len(predictions):
        raise ResearchValidationError("positions must match predictions length")
    turnover_units = [
        abs(current - previous)
        for previous, current in zip([0.0, *normalized_positions[:-1]], normalized_positions)
    ]
    result: dict[str, dict[str, float | int | None]] = {}
    for rate in cost_rates:
        numeric_rate = _as_finite_float(rate, "cost_rate")
        if numeric_rate < 0:
            raise ResearchValidationError("cost_rate cannot be negative")
        costs = [numeric_rate * unit for unit in turnover_units]
        result[f"{numeric_rate:.8g}"] = evaluate_predictions(
            predictions,
            actual_returns,
            positions=normalized_positions,
            transaction_costs=costs,
            timestamps=timestamps,
        )
    return result


def make_experiment_report(
    definition: ExperimentDefinition,
    result: EvaluationResult,
    *,
    fold_metrics: Sequence[Mapping[str, Any]] = (),
    artifacts: Sequence[str | Path] = (),
    notes: Sequence[str] = (),
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
    status: ExperimentStatus | str = ExperimentStatus.COMPLETED,
    metadata: Mapping[str, Any] | None = None,
) -> ExperimentReport:
    """Build a persistable experiment report without making approval claims."""

    return ExperimentReport(
        definition=definition,
        status=status,
        metrics=result.model_dump(),
        fold_metrics=fold_metrics,
        artifacts=artifacts,
        notes=notes,
        started_at=started_at,
        finished_at=finished_at or utc_now(),
        metadata=metadata or {},
    )


build_experiment_report = make_experiment_report


def write_experiment_report(report: ExperimentReport, path: str | Path) -> Path:
    """Persist a canonical, reproducible JSON report."""

    return report.write_json(path)


def _positions_for_window(
    ordered: Sequence[tuple[int, datetime]],
    start: int,
    window: WindowSize | int,
) -> tuple[int, int]:
    """Return [start, end) positions for either a row or duration window."""

    if start >= len(ordered):
        return start, start
    if isinstance(window, int):
        return start, min(len(ordered), start + window)
    cutoff = ordered[start][1] + window
    end = start
    while end < len(ordered) and ordered[end][1] < cutoff:
        end += 1
    return start, end


def _window_start_before(
    ordered: Sequence[tuple[int, datetime]],
    end: int,
    window: WindowSize | int,
) -> int:
    if isinstance(window, int):
        return max(0, end - window)
    if end <= 0:
        return 0
    cutoff = ordered[end - 1][1] - window
    start = end
    while start > 0 and ordered[start - 1][1] >= cutoff:
        start -= 1
    return start


def _skip_window(
    ordered: Sequence[tuple[int, datetime]],
    start: int,
    window: WindowSize | int,
) -> int:
    if window == 0:
        return start
    return _positions_for_window(ordered, start, window)[1]


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    """One leakage-safe rolling experiment fold."""

    fold_number: int
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    purged_indices: tuple[int, ...] = ()
    embargoed_indices: tuple[int, ...] = ()
    train_period: TimeRange | None = None
    validation_period: TimeRange | None = None
    test_period: TimeRange | None = None

    def materialize(self, records: Sequence[Record]) -> tuple[list[Record], list[Record], list[Record]]:
        return (
            [records[index] for index in self.train_indices],
            [records[index] for index in self.validation_indices],
            [records[index] for index in self.test_indices],
        )

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return {
            "fold_number": self.fold_number,
            "train_indices": list(self.train_indices),
            "validation_indices": list(self.validation_indices),
            "test_indices": list(self.test_indices),
            "purged_indices": list(self.purged_indices),
            "embargoed_indices": list(self.embargoed_indices),
            "train_period": self.train_period.model_dump() if self.train_period else None,
            "validation_period": self.validation_period.model_dump() if self.validation_period else None,
            "test_period": self.test_period.model_dump() if self.test_period else None,
        }


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    """Aggregate evaluation and individual fold diagnostics."""

    folds: tuple[WalkForwardFold, ...]
    evaluation: EvaluationResult
    fold_evaluations: tuple[EvaluationResult, ...]

    @property
    def metrics(self) -> Mapping[str, Any]:
        return self.evaluation.metrics

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return {
            "metrics": self.evaluation.model_dump(),
            "folds": [fold.model_dump() for fold in self.folds],
            "fold_metrics": [evaluation.model_dump() for evaluation in self.fold_evaluations],
        }

    to_dict = model_dump


PredictionRoutine: TypeAlias = Callable[[Sequence[Record], Sequence[Record], Sequence[Record]], Any]
ModelFactory: TypeAlias = Callable[[], Any]


class WalkForwardEvaluator:
    """Create and evaluate chronological, purged walk-forward folds.

    The evaluator supports either a ``fit_predict`` routine or a lightweight
    model factory exposing ``fit(train_rows)`` and ``predict(test_rows)``. It
    never randomly shuffles data, and it recreates the model per fold.
    """

    def __init__(
        self,
        *,
        train_window: WindowSize | str | None = None,
        validation_window: WindowSize | str | int = 0,
        test_window: WindowSize | str | None = None,
        step_window: WindowSize | str | None = None,
        expanding: bool = True,
        purge_window: WindowSize | str | int = 0,
        embargo_window: WindowSize | str | int = 0,
        timestamp_getter: TimestampGetter | None = None,
        label_end_getter: ValueGetter | None = None,
        annualization_factor: float | None = None,
        # Friendly aliases for callers used to sklearn-style terminology.
        train_size: int | None = None,
        validation_size: int | None = None,
        test_size: int | None = None,
        step_size: int | None = None,
    ) -> None:
        if train_window is None:
            train_window = train_size
        elif train_size is not None:
            raise ResearchValidationError("provide only one of train_window or train_size")
        if test_window is None:
            test_window = test_size
        elif test_size is not None:
            raise ResearchValidationError("provide only one of test_window or test_size")
        if validation_size is not None:
            if validation_window != 0:
                raise ResearchValidationError("provide only one of validation_window or validation_size")
            validation_window = validation_size
        if step_size is not None:
            if step_window is not None:
                raise ResearchValidationError("provide only one of step_window or step_size")
            step_window = step_size
        if train_window is None or test_window is None:
            raise ResearchValidationError("train_window and test_window are required")
        self.train_window = parse_window(train_window, field_name="train_window")
        self.validation_window = _as_non_negative_window(validation_window, "validation_window")
        self.test_window = parse_window(test_window, field_name="test_window")
        self.step_window = parse_window(step_window, field_name="step_window") if step_window is not None else self.test_window
        self.expanding = bool(expanding)
        self.purge_window = _as_non_negative_window(purge_window, "purge_window")
        self.embargo_window = _as_non_negative_window(embargo_window, "embargo_window")
        self.timestamp_getter = timestamp_getter
        self.label_end_getter = label_end_getter
        self.annualization_factor = annualization_factor

    def split(self, records: Sequence[Record]) -> Iterator[WalkForwardFold]:
        """Yield sequential folds where model fitting never observes future rows."""

        if not records:
            return
        assert_point_in_time_availability(records, timestamp_getter=self.timestamp_getter)
        ordered = _sort_indices(records, self.timestamp_getter)
        count = len(ordered)
        _, train_end = _positions_for_window(ordered, 0, self.train_window)
        if train_end <= 0:
            return
        fold_number = 0
        while train_end < count:
            purge_start = train_end
            validation_start = _skip_window(ordered, purge_start, self.purge_window)
            validation_start = _advance_group_boundary(ordered, validation_start)
            if validation_start >= count:
                break
            validation_end = _skip_window(ordered, validation_start, self.validation_window)
            validation_end = _advance_group_boundary(ordered, validation_end)
            test_start = _skip_window(ordered, validation_end, self.embargo_window)
            test_start = _advance_group_boundary(ordered, test_start)
            if test_start >= count:
                break
            _, test_end = _positions_for_window(ordered, test_start, self.test_window)
            test_end = _advance_group_boundary(ordered, test_end)
            if test_end <= test_start:
                break

            raw_train_start = 0 if self.expanding else _window_start_before(ordered, train_end, self.train_window)
            train_positions: list[int] = list(range(raw_train_start, train_end))
            purged_positions: list[int] = list(range(purge_start, validation_start))
            embargoed_positions: list[int] = list(range(validation_end, test_start))
            test_time = ordered[test_start][1]
            safe_train_positions: list[int] = []
            for position in train_positions:
                record = records[ordered[position][0]]
                label_end = (
                    self.label_end_getter(record)
                    if self.label_end_getter is not None
                    else _row_value(record, "label_available_time", "label_end_time", "label_end", default=None)
                )
                if label_end is not None and ensure_utc(label_end, field_name="label_end_time") > test_time:
                    purged_positions.append(position)
                else:
                    safe_train_positions.append(position)
            if safe_train_positions:
                validation_positions = tuple(range(validation_start, validation_end))
                test_positions = tuple(range(test_start, test_end))
                yield WalkForwardFold(
                    fold_number=fold_number,
                    train_indices=tuple(ordered[position][0] for position in safe_train_positions),
                    validation_indices=tuple(ordered[position][0] for position in validation_positions),
                    test_indices=tuple(ordered[position][0] for position in test_positions),
                    purged_indices=tuple(ordered[position][0] for position in sorted(set(purged_positions))),
                    embargoed_indices=tuple(ordered[position][0] for position in embargoed_positions),
                    train_period=_period_for_positions(ordered, safe_train_positions),
                    validation_period=_period_for_positions(ordered, validation_positions),
                    test_period=_period_for_positions(ordered, test_positions),
                )
                fold_number += 1

            next_train_end = _skip_window(ordered, train_end, self.step_window)
            next_train_end = _advance_group_boundary(ordered, next_train_end)
            if next_train_end <= train_end:
                break
            train_end = next_train_end

    def _prediction_output(self, output: Any, expected_count: int) -> tuple[list[float], list[float] | None, list[float] | None]:
        """Normalize common prediction-routine return forms."""

        positions: list[float] | None = None
        costs: list[float] | None = None
        if isinstance(output, Mapping):
            predictions = output.get("predictions", output.get("prediction"))
            positions_value = output.get("positions", output.get("position"))
            costs_value = output.get("transaction_costs", output.get("costs"))
            if positions_value is not None:
                positions = [_as_finite_float(value, "position") for value in positions_value]
            if costs_value is not None:
                costs = [_as_finite_float(value, "transaction_cost") for value in costs_value]
        else:
            predictions = output
        if predictions is None:
            raise ResearchValidationError("prediction routine did not return predictions")
        normalized = [_as_finite_float(value, "prediction") for value in predictions]
        if len(normalized) != expected_count:
            raise ResearchValidationError(
                f"prediction routine returned {len(normalized)} predictions for {expected_count} test rows"
            )
        if positions is not None and len(positions) != expected_count:
            raise ResearchValidationError("prediction routine positions must match test rows")
        if costs is not None and len(costs) != expected_count:
            raise ResearchValidationError("prediction routine costs must match test rows")
        return normalized, positions, costs

    def evaluate(
        self,
        records: Sequence[Record],
        *,
        fit_predict: PredictionRoutine | None = None,
        model_factory: ModelFactory | None = None,
        target_getter: ValueGetter | None = None,
    ) -> WalkForwardResult:
        """Fit/recreate candidates across folds and evaluate their out-of-sample rows.

        Passing neither callback reuses the ``prediction`` field already stored
        on each test row, which is useful for evaluating shadow predictions.
        """

        if fit_predict is not None and model_factory is not None:
            raise ResearchValidationError("provide either fit_predict or model_factory, not both")
        folds = tuple(self.split(records))
        if not folds:
            raise ResearchValidationError("walk-forward configuration produced no leakage-safe folds")
        fold_results: list[EvaluationResult] = []
        all_observations: list[EvaluationObservation] = []
        for fold in folds:
            train_rows, validation_rows, test_rows = fold.materialize(records)
            if fit_predict is not None:
                output = fit_predict(train_rows, validation_rows, test_rows)
                predictions, positions, costs = self._prediction_output(output, len(test_rows))
            elif model_factory is not None:
                model = model_factory()
                fit = getattr(model, "fit", None)
                predict = getattr(model, "predict", None)
                if not callable(fit) or not callable(predict):
                    raise ResearchValidationError("model_factory must create a model with fit() and predict()")
                fit(train_rows)
                output = predict(test_rows)
                predictions, positions, costs = self._prediction_output(output, len(test_rows))
            else:
                predictions = [_as_finite_float(_row_value(row, "prediction", "expected_return"), "prediction") for row in test_rows]
                positions = None
                costs = None
            observations: list[EvaluationObservation] = []
            for index, row in enumerate(test_rows):
                observation = observation_from_row(row, prediction=predictions[index])
                actual = target_getter(row) if target_getter is not None else observation.actual_return
                observations.append(
                    EvaluationObservation(
                        timestamp=observation.timestamp,
                        prediction=predictions[index],
                        actual_return=actual,
                        symbol=observation.symbol,
                        signal_id=observation.signal_id,
                        model_family=observation.model_family,
                        regime=observation.regime,
                        position=positions[index] if positions is not None else observation.position,
                        transaction_cost=costs[index] if costs is not None else observation.transaction_cost,
                        holding_seconds=observation.holding_seconds,
                        external_ai_available=observation.external_ai_available,
                        available_to_model_time=observation.available_to_model_time,
                        label_available_time=observation.label_available_time,
                        feature_families=observation.feature_families,
                        metadata=observation.metadata,
                    )
                )
            fold_result = evaluate_observations(observations, annualization_factor=self.annualization_factor)
            fold_results.append(fold_result)
            all_observations.extend(observations)
        aggregate = evaluate_observations(all_observations, annualization_factor=self.annualization_factor)
        return WalkForwardResult(folds=folds, evaluation=aggregate, fold_evaluations=tuple(fold_results))


@dataclass(frozen=True, slots=True)
class SignalSeries:
    """Aligned prediction, position, and PnL streams for one registered signal."""

    signal_id: str
    timestamps: tuple[datetime, ...]
    predictions: tuple[float, ...]
    positions: tuple[float, ...]
    pnl: tuple[float, ...]
    feature_families: tuple[str, ...] = ()
    model_family: str | None = None

    def __post_init__(self) -> None:
        if not self.signal_id:
            raise ResearchValidationError("SignalSeries.signal_id cannot be empty")
        lengths = {len(self.timestamps), len(self.predictions), len(self.positions), len(self.pnl)}
        if len(lengths) != 1:
            raise ResearchValidationError("SignalSeries timestamps, predictions, positions, and pnl must align")
        timestamps = tuple(ensure_utc(value, field_name="timestamp") for value in self.timestamps)
        if any(later < earlier for earlier, later in zip(timestamps, timestamps[1:])):
            raise ResearchValidationError("SignalSeries timestamps must be ordered")
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "predictions", tuple(_as_finite_float(value, "prediction") for value in self.predictions))
        object.__setattr__(self, "positions", tuple(_as_finite_float(value, "position") for value in self.positions))
        object.__setattr__(self, "pnl", tuple(_as_finite_float(value, "pnl") for value in self.pnl))
        object.__setattr__(self, "feature_families", tuple(sorted({str(item) for item in self.feature_families if str(item)})))

    @classmethod
    def from_rows(cls, signal_id: str, rows: Sequence[Record]) -> "SignalSeries":
        """Build a signal stream from generic rows with prediction/actual fields."""

        observations = sorted((observation_from_row(row) for row in rows), key=lambda item: item.timestamp)
        families = sorted({family for item in observations for family in item.feature_families})
        positions = tuple(item.position if item.position is not None else _sign(item.prediction) for item in observations)
        pnl = tuple(position * item.actual_return - item.transaction_cost for position, item in zip(positions, observations))
        model_family = next((item.model_family for item in observations if item.model_family), None)
        return cls(
            signal_id=signal_id,
            timestamps=tuple(item.timestamp for item in observations),
            predictions=tuple(item.prediction for item in observations),
            positions=positions,
            pnl=pnl,
            feature_families=tuple(families),
            model_family=model_family,
        )

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "timestamps": [value.isoformat() for value in self.timestamps],
            "predictions": list(self.predictions),
            "positions": list(self.positions),
            "pnl": list(self.pnl),
            "feature_families": list(self.feature_families),
            "model_family": self.model_family,
        }


def _mean_by_timestamp(timestamps: Sequence[datetime], values: Sequence[float]) -> dict[datetime, float]:
    grouped: dict[datetime, list[float]] = defaultdict(list)
    for timestamp, value in zip(timestamps, values):
        grouped[timestamp].append(value)
    return {timestamp: statistics.fmean(values_at_time) for timestamp, values_at_time in grouped.items()}


def _aligned_values(
    left_timestamps: Sequence[datetime],
    left_values: Sequence[float],
    right_timestamps: Sequence[datetime],
    right_values: Sequence[float],
) -> tuple[list[float], list[float]]:
    left = _mean_by_timestamp(left_timestamps, left_values)
    right = _mean_by_timestamp(right_timestamps, right_values)
    common = sorted(set(left).intersection(right))
    return [left[key] for key in common], [right[key] for key in common]


def _drawdown_returns(values: Sequence[float]) -> list[float]:
    _, drawdowns = _drawdown_path(values)
    return drawdowns


@dataclass(frozen=True, slots=True)
class SignalCorrelation:
    """Pairwise independence diagnostics for two signal streams."""

    left_signal_id: str
    right_signal_id: str
    aligned_observations: int
    prediction_correlation: float | None
    position_correlation: float | None
    pnl_correlation: float | None
    drawdown_correlation: float | None
    trade_overlap: float | None
    feature_overlap: float | None

    @property
    def maximum_absolute_correlation(self) -> float | None:
        values = [
            abs(value)
            for value in (
                self.prediction_correlation,
                self.position_correlation,
                self.pnl_correlation,
                self.drawdown_correlation,
            )
            if value is not None
        ]
        return max(values) if values else None

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return {
            "left_signal_id": self.left_signal_id,
            "right_signal_id": self.right_signal_id,
            "aligned_observations": self.aligned_observations,
            "prediction_correlation": self.prediction_correlation,
            "position_correlation": self.position_correlation,
            "pnl_correlation": self.pnl_correlation,
            "drawdown_correlation": self.drawdown_correlation,
            "trade_overlap": self.trade_overlap,
            "feature_overlap": self.feature_overlap,
            "maximum_absolute_correlation": self.maximum_absolute_correlation,
        }


@dataclass(frozen=True, slots=True)
class SignalIndependenceAnalysis:
    """Correlation matrix summary and correlated-signal families."""

    correlation_threshold: float
    pairs: tuple[SignalCorrelation, ...]
    correlated_groups: tuple[tuple[str, ...], ...]

    def pair_for(self, left_signal_id: str, right_signal_id: str) -> SignalCorrelation | None:
        wanted = {left_signal_id, right_signal_id}
        return next(
            (
                pair
                for pair in self.pairs
                if {pair.left_signal_id, pair.right_signal_id} == wanted
            ),
            None,
        )

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return {
            "correlation_threshold": self.correlation_threshold,
            "pairs": [pair.model_dump() for pair in self.pairs],
            "correlated_groups": [list(group) for group in self.correlated_groups],
        }


def _signal_series_from_input(
    signals: Mapping[str, SignalSeries | Sequence[Record]] | Sequence[Record],
) -> dict[str, SignalSeries]:
    if isinstance(signals, Mapping):
        output: dict[str, SignalSeries] = {}
        for signal_id, value in signals.items():
            output[str(signal_id)] = value if isinstance(value, SignalSeries) else SignalSeries.from_rows(str(signal_id), value)
        return output
    grouped: dict[str, list[Record]] = defaultdict(list)
    for row in signals:
        signal_id = _row_value(row, "signal_id")
        grouped[str(signal_id)].append(row)
    return {signal_id: SignalSeries.from_rows(signal_id, rows) for signal_id, rows in grouped.items()}


def _pair_correlation(left: SignalSeries, right: SignalSeries) -> SignalCorrelation:
    left_prediction, right_prediction = _aligned_values(
        left.timestamps, left.predictions, right.timestamps, right.predictions
    )
    left_position, right_position = _aligned_values(left.timestamps, left.positions, right.timestamps, right.positions)
    left_pnl, right_pnl = _aligned_values(left.timestamps, left.pnl, right.timestamps, right.pnl)
    left_drawdown, right_drawdown = _aligned_values(
        left.timestamps,
        _drawdown_returns(left.pnl),
        right.timestamps,
        _drawdown_returns(right.pnl),
    )
    left_position_map = _mean_by_timestamp(left.timestamps, left.positions)
    right_position_map = _mean_by_timestamp(right.timestamps, right.positions)
    common_times = set(left_position_map).intersection(right_position_map)
    either_active = {
        timestamp
        for timestamp in set(left_position_map).union(right_position_map)
        if left_position_map.get(timestamp, 0.0) != 0 or right_position_map.get(timestamp, 0.0) != 0
    }
    both_active = {
        timestamp
        for timestamp in common_times
        if left_position_map[timestamp] != 0 and right_position_map[timestamp] != 0
    }
    trade_overlap = len(both_active) / len(either_active) if either_active else None
    left_features = set(left.feature_families)
    right_features = set(right.feature_families)
    feature_union = left_features.union(right_features)
    feature_overlap = len(left_features.intersection(right_features)) / len(feature_union) if feature_union else None
    return SignalCorrelation(
        left_signal_id=left.signal_id,
        right_signal_id=right.signal_id,
        aligned_observations=len(left_prediction),
        prediction_correlation=_pearson(left_prediction, right_prediction),
        position_correlation=_pearson(left_position, right_position),
        pnl_correlation=_pearson(left_pnl, right_pnl),
        drawdown_correlation=_pearson(left_drawdown, right_drawdown),
        trade_overlap=trade_overlap,
        feature_overlap=feature_overlap,
    )


def analyze_signal_independence(
    signals: Mapping[str, SignalSeries | Sequence[Record]] | Sequence[Record],
    *,
    correlation_threshold: float = 0.80,
) -> SignalIndependenceAnalysis:
    """Measure prediction/position/PnL dependence and group correlated signals."""

    threshold = _as_finite_float(correlation_threshold, "correlation_threshold")
    if not 0.0 < threshold <= 1.0:
        raise ResearchValidationError("correlation_threshold must be in (0, 1]")
    series = _signal_series_from_input(signals)
    signal_ids = sorted(series)
    pairs: list[SignalCorrelation] = []
    parent = {signal_id: signal_id for signal_id in signal_ids}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left_index, left_id in enumerate(signal_ids):
        for right_id in signal_ids[left_index + 1 :]:
            pair = _pair_correlation(series[left_id], series[right_id])
            pairs.append(pair)
            dependent = (pair.maximum_absolute_correlation or 0.0) >= threshold
            # Feature overlap is a weaker but useful warning where PnL history is short.
            dependent = dependent or (pair.feature_overlap is not None and pair.feature_overlap >= threshold)
            if dependent:
                union(left_id, right_id)
    grouped: dict[str, list[str]] = defaultdict(list)
    for signal_id in signal_ids:
        grouped[find(signal_id)].append(signal_id)
    groups = tuple(
        tuple(sorted(group))
        for group in sorted(grouped.values(), key=lambda members: tuple(sorted(members)))
        if len(group) > 1
    )
    return SignalIndependenceAnalysis(
        correlation_threshold=threshold,
        pairs=tuple(pairs),
        correlated_groups=groups,
    )


signal_correlation_analysis = analyze_signal_independence


def assess_incremental_contribution(
    candidate_signal_id: str,
    incumbent_signal_ids: Sequence[str],
    signals: Mapping[str, SignalSeries | Sequence[Record]] | Sequence[Record],
    *,
    correlation_threshold: float = 0.80,
) -> dict[str, Any]:
    """Return a transparent independence gate before adding a signal to an ensemble."""

    all_series = _signal_series_from_input(signals)
    if candidate_signal_id not in all_series:
        raise ResearchValidationError(f"candidate signal {candidate_signal_id!r} is not present")
    incumbents = [signal_id for signal_id in incumbent_signal_ids if signal_id in all_series and signal_id != candidate_signal_id]
    if not incumbents:
        return {
            "candidate_signal_id": candidate_signal_id,
            "incumbent_signal_ids": [],
            "is_incremental": True,
            "max_absolute_prediction_correlation": None,
            "max_absolute_pnl_correlation": None,
            "mean_pnl_difference_vs_incumbents": None,
            "reason_codes": ["NO_INCUMBENT_SIGNALS"],
        }
    candidate = all_series[candidate_signal_id]
    pair_metrics = [_pair_correlation(candidate, all_series[signal_id]) for signal_id in incumbents]
    prediction_values = [abs(item.prediction_correlation) for item in pair_metrics if item.prediction_correlation is not None]
    pnl_values = [abs(item.pnl_correlation) for item in pair_metrics if item.pnl_correlation is not None]
    candidate_mean = _mean(candidate.pnl)
    incumbent_means = [_mean(all_series[signal_id].pnl) for signal_id in incumbents]
    usable_incumbents = [value for value in incumbent_means if value is not None]
    mean_difference = candidate_mean - statistics.fmean(usable_incumbents) if candidate_mean is not None and usable_incumbents else None
    maximum = max([*prediction_values, *pnl_values], default=0.0)
    independent = maximum < correlation_threshold
    return {
        "candidate_signal_id": candidate_signal_id,
        "incumbent_signal_ids": incumbents,
        "is_incremental": independent,
        "max_absolute_prediction_correlation": max(prediction_values) if prediction_values else None,
        "max_absolute_pnl_correlation": max(pnl_values) if pnl_values else None,
        "mean_pnl_difference_vs_incumbents": mean_difference,
        "reason_codes": ["INDEPENDENT_CONTRIBUTION"] if independent else ["CORRELATED_SIGNAL_PENALTY"],
        "pairs": [item.model_dump() for item in pair_metrics],
    }
