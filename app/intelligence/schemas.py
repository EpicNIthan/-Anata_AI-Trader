"""Typed schemas for local and external news intelligence.

These dataclasses intentionally have no ORM or trading dependencies.  They are
safe to use in collectors, local research commands, and a future persistence
adapter without giving an intelligence provider any execution capability.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Mapping


class IntelligenceValidationError(ValueError):
    """Raised when untrusted intelligence data violates the accepted schema."""


class EventType(str, Enum):
    ADOPTION = "adoption"
    EXCHANGE = "exchange"
    FUNDING = "funding"
    LIQUIDATION = "liquidation"
    MACRO = "macro"
    MARKET_MOVE = "market_move"
    MARKET_STRUCTURE = "market_structure"
    PARTNERSHIP = "partnership"
    PRODUCT = "product"
    PROTOCOL = "protocol"
    REGULATORY = "regulatory"
    RUMOR = "rumor"
    SECURITY_INCIDENT = "security_incident"
    STABLECOIN = "stablecoin"
    TOKENOMICS = "tokenomics"
    OTHER = "other"


class EventDirection(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class TimeHorizon(str, Enum):
    IMMEDIATE = "immediate"
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"
    UNKNOWN = "unknown"


class ValidationStatus(str, Enum):
    VALID = "valid"
    VALID_WITH_WARNINGS = "valid_with_warnings"
    INVALID = "invalid"


class RequestStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    CACHE_HIT = "cache_hit"
    QUOTA_EXHAUSTED = "quota_exhausted"
    BUDGET_EXHAUSTED = "budget_exhausted"
    CIRCUIT_OPEN = "circuit_open"
    RATE_LIMITED = "rate_limited"


def utc_now() -> datetime:
    """Return an aware UTC timestamp."""

    return datetime.now(timezone.utc)


def as_utc(value: datetime | str | None, *, field_name: str = "timestamp") -> datetime | None:
    """Parse a timestamp and normalize it to an aware UTC value."""

    if value is None:
        return None
    if isinstance(value, str):
        candidate = value.strip().replace("Z", "+00:00")
        try:
            value = datetime.fromisoformat(candidate)
        except ValueError as exc:
            raise IntelligenceValidationError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if not isinstance(value, datetime):
        raise IntelligenceValidationError(f"{field_name} must be a datetime or ISO-8601 timestamp")
    if value.tzinfo is None:
        raise IntelligenceValidationError(f"{field_name} must include a timezone")
    return value.astimezone(timezone.utc)


def _enum(value: Any, enum_type: type[Enum], field_name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    if value is None:
        raise IntelligenceValidationError(f"{field_name} is required")
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "positive": "bullish",
        "negative": "bearish",
        "none": "neutral",
        "near_term": "short_term",
        "nearterm": "short_term",
        "mid_term": "medium_term",
        "longterm": "long_term",
    }
    normalized = aliases.get(normalized, normalized)
    try:
        return enum_type(normalized)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise IntelligenceValidationError(f"{field_name} must be one of: {allowed}") from exc


def bounded_float(value: Any, field_name: str, *, low: float, high: float) -> float:
    """Coerce a finite numeric value and enforce inclusive bounds."""

    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise IntelligenceValidationError(f"{field_name} must be numeric") from exc
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        raise IntelligenceValidationError(f"{field_name} must be finite")
    if not low <= parsed <= high:
        raise IntelligenceValidationError(f"{field_name} must be between {low} and {high}")
    return parsed


_ASSET_PATTERN = re.compile(r"^[A-Z][A-Z0-9]{1,14}$")
_NUMBER_PATTERN = re.compile(r"(?<![A-Za-z0-9_])[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?")


def normalize_asset_symbol(value: Any) -> str:
    """Normalize a base asset or exchange-style symbol while rejecting arbitrary text."""

    symbol = str(value or "").strip().upper().replace("/", "").replace("-", "")
    if not _ASSET_PATTERN.fullmatch(symbol):
        raise IntelligenceValidationError(f"Invalid affected asset symbol: {value!r}")
    return symbol


def _string_list(value: Any, field_name: str, *, max_items: int, max_length: int) -> tuple[str, ...]:
    if value is None:
        return ()
    values = [value] if isinstance(value, str) else list(value)
    if len(values) > max_items:
        raise IntelligenceValidationError(f"{field_name} cannot contain more than {max_items} entries")
    cleaned: list[str] = []
    for item in values:
        item = str(item).strip()
        if not item:
            continue
        if len(item) > max_length:
            raise IntelligenceValidationError(f"{field_name} entry exceeds {max_length} characters")
        if item not in cleaned:
            cleaned.append(item)
    return tuple(cleaned)


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_json_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class FactualClaim:
    """A short, source-attributable factual claim extracted from a document."""

    claim: str
    numbers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.claim or len(self.claim.strip()) > 1_000:
            raise IntelligenceValidationError("factual claim must contain 1 to 1000 characters")
        if len(self.numbers) > 12:
            raise IntelligenceValidationError("a factual claim may contain at most 12 extracted numbers")

    @classmethod
    def from_value(cls, value: Any) -> "FactualClaim":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(claim=value.strip(), numbers=tuple(_NUMBER_PATTERN.findall(value)))
        if not isinstance(value, Mapping):
            raise IntelligenceValidationError("factual_claims entries must be strings or objects")
        claim = value.get("claim", value.get("text", ""))
        supplied = value.get("numbers")
        numbers = _NUMBER_PATTERN.findall(str(claim)) if supplied is None else _string_list(
            supplied, "factual_claims.numbers", max_items=12, max_length=64
        )
        return cls(claim=str(claim).strip(), numbers=tuple(numbers))

    def missing_source_numbers(self, source_text: str) -> tuple[str, ...]:
        """Return claimed numeric literals that cannot be found in the source text.

        This is deliberately a warning-level helper: numbers can be normalized or
        rounded by a model, so callers can choose whether a mismatch is fatal.
        """

        source_numbers = {
            token.replace(",", "").rstrip("%") for token in _NUMBER_PATTERN.findall(source_text or "")
        }
        missing = []
        for number in self.numbers:
            normalized = number.replace(",", "").rstrip("%")
            if normalized and normalized not in source_numbers:
                missing.append(number)
        return tuple(missing)

    def model_dump(self) -> dict[str, Any]:
        return _json_safe(self)


@dataclass(frozen=True)
class NewsDocument:
    """A validated source document used as an intelligence-provider input."""

    title: str
    content: str
    source: str
    url: str | None = None
    published_at: datetime | None = None
    received_at: datetime | None = None
    available_to_model_at: datetime | None = None
    content_hash: str | None = None
    is_duplicate: bool = False
    relevant_assets: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        title = self.title.strip()
        content = self.content.strip()
        source = self.source.strip()
        if not title:
            raise IntelligenceValidationError("document title is required")
        if not content:
            raise IntelligenceValidationError("document content is required")
        if not source:
            raise IntelligenceValidationError("document source is required")
        if len(title) > 2_000 or len(content) > 200_000:
            raise IntelligenceValidationError("document title or content is too large")
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "published_at", as_utc(self.published_at, field_name="published_at"))
        object.__setattr__(self, "received_at", as_utc(self.received_at, field_name="received_at"))
        object.__setattr__(self, "available_to_model_at", as_utc(self.available_to_model_at, field_name="available_to_model_at"))
        normalized_assets = tuple(normalize_asset_symbol(asset) for asset in self.relevant_assets)
        object.__setattr__(self, "relevant_assets", tuple(dict.fromkeys(normalized_assets)))
        calculated_hash = hashlib.sha256(f"{title}\n{content}".encode("utf-8")).hexdigest()
        if self.content_hash and not re.fullmatch(r"[a-fA-F0-9]{64}", self.content_hash):
            raise IntelligenceValidationError("content_hash must be a SHA-256 hexadecimal digest")
        object.__setattr__(self, "content_hash", (self.content_hash or calculated_hash).lower())

    @property
    def text(self) -> str:
        return f"{self.title}\n\n{self.content}"

    @property
    def source_reference(self) -> str:
        return self.url or self.source

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NewsDocument":
        assets = value.get("relevant_assets", value.get("affected_assets", ())) or ()
        if isinstance(assets, str):
            assets = (assets,)
        return cls(
            title=str(value.get("title", "")),
            content=str(value.get("content", value.get("raw_text", value.get("text", "")))),
            source=str(value.get("source", value.get("source_name", "unknown"))),
            url=value.get("url"),
            published_at=value.get("published_at", value.get("published_time")),
            received_at=value.get("received_at", value.get("created_at")),
            available_to_model_at=value.get("available_to_model_at", value.get("available_to_model_time")),
            content_hash=value.get("content_hash"),
            is_duplicate=bool(value.get("is_duplicate", False)),
            relevant_assets=tuple(assets),
            metadata=dict(value.get("metadata") or {}),
        )

    def model_dump(self) -> dict[str, Any]:
        return _json_safe(self)


@dataclass(frozen=True)
class StructuredNewsEvent:
    """Validated structured enrichment from a local or external intelligence provider."""

    event_type: EventType
    affected_assets: tuple[str, ...]
    affected_entities: tuple[str, ...]
    direction: EventDirection
    sentiment: float
    severity: float
    importance: float
    novelty: float
    time_horizon: TimeHorizon
    factual_claims: tuple[FactualClaim, ...]
    confidence: float
    source_summary: str
    source_reference: str | None = None
    provider: str = "unknown"
    model: str | None = None
    prompt_version: str = "v1"
    validation_status: ValidationStatus = ValidationStatus.VALID
    validation_warnings: tuple[str, ...] = ()
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    generated_at: datetime = field(default_factory=utc_now)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.event_type, EventType):
            object.__setattr__(self, "event_type", _enum(self.event_type, EventType, "event_type"))
        if not isinstance(self.direction, EventDirection):
            object.__setattr__(self, "direction", _enum(self.direction, EventDirection, "direction"))
        if not isinstance(self.time_horizon, TimeHorizon):
            object.__setattr__(self, "time_horizon", _enum(self.time_horizon, TimeHorizon, "time_horizon"))
        if not isinstance(self.validation_status, ValidationStatus):
            object.__setattr__(
                self, "validation_status", _enum(self.validation_status, ValidationStatus, "validation_status")
            )
        assets = tuple(normalize_asset_symbol(asset) for asset in self.affected_assets)
        if len(assets) > 20:
            raise IntelligenceValidationError("affected_assets cannot contain more than 20 entries")
        object.__setattr__(self, "affected_assets", tuple(dict.fromkeys(assets)))
        object.__setattr__(
            self,
            "affected_entities",
            _string_list(self.affected_entities, "affected_entities", max_items=25, max_length=200),
        )
        claims = tuple(FactualClaim.from_value(claim) for claim in self.factual_claims)
        if len(claims) > 20:
            raise IntelligenceValidationError("factual_claims cannot contain more than 20 entries")
        object.__setattr__(self, "factual_claims", claims)
        for name, low, high in (
            ("sentiment", -1.0, 1.0),
            ("severity", 0.0, 1.0),
            ("importance", 0.0, 1.0),
            ("novelty", 0.0, 1.0),
            ("confidence", 0.0, 1.0),
        ):
            object.__setattr__(self, name, bounded_float(getattr(self, name), name, low=low, high=high))
        source_summary = self.source_summary.strip()
        if not source_summary or len(source_summary) > 4_000:
            raise IntelligenceValidationError("source_summary must contain 1 to 4000 characters")
        if not self.provider.strip() or len(self.provider.strip()) > 128:
            raise IntelligenceValidationError("provider must contain 1 to 128 characters")
        if not self.prompt_version.strip() or len(self.prompt_version.strip()) > 128:
            raise IntelligenceValidationError("prompt_version must contain 1 to 128 characters")
        object.__setattr__(self, "source_summary", source_summary)
        object.__setattr__(self, "provider", self.provider.strip())
        object.__setattr__(self, "prompt_version", self.prompt_version.strip())
        object.__setattr__(self, "generated_at", as_utc(self.generated_at, field_name="generated_at") or utc_now())
        try:
            uuid.UUID(str(self.event_id))
        except (ValueError, TypeError, AttributeError) as exc:
            raise IntelligenceValidationError("event_id must be a UUID") from exc

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        provider: str | None = None,
        model: str | None = None,
        prompt_version: str | None = None,
        source_reference: str | None = None,
        source_text: str | None = None,
        reject_unverifiable_numbers: bool = False,
    ) -> "StructuredNewsEvent":
        """Create a validated event from an untrusted provider response."""

        if not isinstance(value, Mapping):
            raise IntelligenceValidationError("provider result must be a JSON object")
        missing = [
            name
            for name in (
                "event_type",
                "affected_assets",
                "affected_entities",
                "direction",
                "sentiment",
                "severity",
                "importance",
                "novelty",
                "time_horizon",
                "factual_claims",
                "confidence",
                "source_summary",
            )
            if name not in value
        ]
        if missing:
            raise IntelligenceValidationError(f"provider result is missing required fields: {', '.join(missing)}")
        affected_assets = value.get("affected_assets") or ()
        affected_entities = value.get("affected_entities") or ()
        if isinstance(affected_assets, str):
            affected_assets = (affected_assets,)
        if isinstance(affected_entities, str):
            affected_entities = (affected_entities,)
        event = cls(
            event_type=_enum(value["event_type"], EventType, "event_type"),
            affected_assets=tuple(affected_assets),
            affected_entities=tuple(affected_entities),
            direction=_enum(value["direction"], EventDirection, "direction"),
            sentiment=bounded_float(value["sentiment"], "sentiment", low=-1.0, high=1.0),
            severity=bounded_float(value["severity"], "severity", low=0.0, high=1.0),
            importance=bounded_float(value["importance"], "importance", low=0.0, high=1.0),
            novelty=bounded_float(value["novelty"], "novelty", low=0.0, high=1.0),
            time_horizon=_enum(value["time_horizon"], TimeHorizon, "time_horizon"),
            factual_claims=tuple(FactualClaim.from_value(item) for item in value.get("factual_claims") or ()),
            confidence=bounded_float(value["confidence"], "confidence", low=0.0, high=1.0),
            source_summary=str(value["source_summary"]),
            source_reference=source_reference or value.get("source_reference"),
            provider=provider or str(value.get("provider", "unknown")),
            model=model or value.get("model"),
            prompt_version=prompt_version or str(value.get("prompt_version", "v1")),
            validation_status=ValidationStatus.VALID,
            event_id=str(value.get("event_id") or uuid.uuid4()),
            generated_at=value.get("generated_at") or utc_now(),
            metadata=dict(value.get("metadata") or {}),
        )
        warnings = event.validate_against_source(source_text) if source_text else ()
        if warnings and reject_unverifiable_numbers:
            raise IntelligenceValidationError("; ".join(warnings))
        if warnings:
            object.__setattr__(event, "validation_status", ValidationStatus.VALID_WITH_WARNINGS)
            object.__setattr__(event, "validation_warnings", warnings)
        return event

    def validate_against_source(self, source_text: str | None) -> tuple[str, ...]:
        """Check factual numeric literals against the supplied article text."""

        if not source_text:
            return ()
        warnings: list[str] = []
        for index, claim in enumerate(self.factual_claims):
            missing = claim.missing_source_numbers(source_text)
            if missing:
                warnings.append(f"factual_claims[{index}] has source-unverified numbers: {', '.join(missing)}")
        return tuple(warnings)

    def model_dump(self) -> dict[str, Any]:
        return _json_safe(self)

    def to_json(self) -> str:
        return json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class ProviderUsage:
    """Provider-reported or estimated cost and token accounting."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float = 0.0

    def __post_init__(self) -> None:
        for field_name in ("input_tokens", "output_tokens"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, int) or value < 0):
                raise IntelligenceValidationError(f"{field_name} must be a non-negative integer or None")
        object.__setattr__(
            self,
            "estimated_cost_usd",
            bounded_float(self.estimated_cost_usd, "estimated_cost_usd", low=0.0, high=1_000_000.0),
        )

    def model_dump(self) -> dict[str, Any]:
        return _json_safe(self)


@dataclass(frozen=True)
class ProviderResponse:
    """A provider response after its event payload was schema-validated."""

    provider: str
    model: str | None
    event: StructuredNewsEvent
    usage: ProviderUsage = field(default_factory=ProviderUsage)
    raw_response: Mapping[str, Any] | None = None

    def model_dump(self) -> dict[str, Any]:
        return _json_safe(self)


@dataclass(frozen=True)
class ProviderRequestAudit:
    """Persistable request outcome record; never includes a secret or raw credential."""

    provider: str
    model: str | None
    content_hash: str
    prompt_version: str
    requested_at: datetime
    completed_at: datetime
    status: RequestStatus
    token_usage: ProviderUsage | None = None
    estimated_cost_usd: float = 0.0
    error_category: str | None = None
    retry_count: int = 0
    cache_hit: bool = False

    def __post_init__(self) -> None:
        if not self.provider or len(self.provider) > 128:
            raise IntelligenceValidationError("audit provider must contain 1 to 128 characters")
        if not re.fullmatch(r"[a-fA-F0-9]{64}", self.content_hash):
            raise IntelligenceValidationError("audit content_hash must be a SHA-256 hexadecimal digest")
        object.__setattr__(self, "requested_at", as_utc(self.requested_at, field_name="requested_at") or utc_now())
        object.__setattr__(self, "completed_at", as_utc(self.completed_at, field_name="completed_at") or utc_now())
        if self.completed_at < self.requested_at:
            raise IntelligenceValidationError("completed_at cannot precede requested_at")
        if not isinstance(self.status, RequestStatus):
            object.__setattr__(self, "status", _enum(self.status, RequestStatus, "status"))
        if self.retry_count < 0:
            raise IntelligenceValidationError("retry_count cannot be negative")
        object.__setattr__(
            self,
            "estimated_cost_usd",
            bounded_float(self.estimated_cost_usd, "estimated_cost_usd", low=0.0, high=1_000_000.0),
        )

    def model_dump(self) -> dict[str, Any]:
        return _json_safe(self)


@dataclass(frozen=True)
class IntelligenceResult:
    """The local baseline plus an optional externally validated context overlay."""

    document_hash: str
    local_event: StructuredNewsEvent
    external_event: StructuredNewsEvent | None
    selected_event: StructuredNewsEvent
    external_ai_available: bool
    external_ai_missing: bool
    external_ai_failed: bool
    reason_codes: tuple[str, ...] = ()
    request_audits: tuple[ProviderRequestAudit, ...] = ()

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-fA-F0-9]{64}", self.document_hash):
            raise IntelligenceValidationError("document_hash must be a SHA-256 hexadecimal digest")
        if self.external_ai_available and self.external_event is None:
            raise IntelligenceValidationError("external_ai_available requires external_event")
        if self.external_ai_available and self.external_ai_missing:
            raise IntelligenceValidationError("external AI cannot be both available and missing")

    @property
    def external_ai_age_seconds(self) -> float | None:
        if not self.external_event:
            return None
        return max(0.0, (utc_now() - self.external_event.generated_at).total_seconds())

    def model_dump(self) -> dict[str, Any]:
        return _json_safe(self)
