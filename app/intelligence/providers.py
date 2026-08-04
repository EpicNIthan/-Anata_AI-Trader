"""Provider adapters for the bounded news-intelligence layer.

Providers return schema-validated events only.  This module intentionally does
not import strategy, portfolio, risk, or paper-execution code.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import zipfile
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Protocol, runtime_checkable

import httpx

from .schemas import (
    EventDirection,
    EventType,
    IntelligenceValidationError,
    NewsDocument,
    ProviderResponse,
    ProviderUsage,
    StructuredNewsEvent,
    TimeHorizon,
)

NEWS_STUDENT_MODEL_FAMILY = "intelligence.news_student_naive_bayes"
NEWS_STUDENT_ARTIFACT_MEMBER = "student_artifact.json"


class IntelligenceProviderError(RuntimeError):
    """Base error for safe-to-record provider failures."""

    category = "provider_error"


class ProviderUnavailableError(IntelligenceProviderError):
    category = "unavailable"


class ProviderTimeoutError(IntelligenceProviderError):
    category = "timeout"


class ProviderRateLimitError(IntelligenceProviderError):
    category = "rate_limit"


class ProviderResponseError(IntelligenceProviderError):
    category = "invalid_response"


@runtime_checkable
class IntelligenceProvider(Protocol):
    """Small async provider contract used by :class:`IntelligenceRouter`."""

    name: str
    model: str | None
    is_external: bool

    async def enrich(self, document: NewsDocument, *, prompt_version: str) -> ProviderResponse:
        """Return one validated structured event for the supplied document."""


def tokenize_news_text(text: str) -> list[str]:
    """Deterministic tokenization shared by the rule and JSON-student providers."""

    return re.findall(r"[a-z0-9]{2,}", text.lower())


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


_ASSET_KEYWORDS: dict[str, tuple[str, ...]] = {
    "BTC": ("bitcoin", "btc"),
    "ETH": ("ethereum", "ether", "eth"),
    "SOL": ("solana", "sol"),
    "BNB": ("binance", "bnb"),
    "XRP": ("ripple", "xrp"),
    "ADA": ("cardano", "ada"),
    "DOGE": ("dogecoin", "doge"),
    "AVAX": ("avalanche", "avax"),
    "LINK": ("chainlink", "link"),
    "USDT": ("tether", "usdt"),
    "USDC": ("usd coin", "usdc", "circle"),
}

_POSITIVE_WORDS = {
    "adoption",
    "approval",
    "approved",
    "bullish",
    "buyback",
    "cuts",
    "growth",
    "inflow",
    "launch",
    "partnership",
    "rally",
    "record",
    "recovery",
    "surge",
    "upgrade",
}
_NEGATIVE_WORDS = {
    "ban",
    "bankruptcy",
    "bearish",
    "crash",
    "depeg",
    "decline",
    "drop",
    "exploit",
    "hack",
    "halt",
    "lawsuit",
    "liquidation",
    "outflow",
    "selloff",
    "sues",
}
_SEVERE_WORDS = {
    "bankruptcy",
    "breach",
    "crash",
    "default",
    "depeg",
    "exploit",
    "hack",
    "liquidation",
    "sanction",
    "war",
}
_EVENT_KEYWORDS: tuple[tuple[EventType, set[str]], ...] = (
    (EventType.SECURITY_INCIDENT, {"exploit", "hack", "breach", "drained", "vulnerability"}),
    (EventType.STABLECOIN, {"depeg", "stablecoin", "tether", "usdc", "reserve"}),
    (EventType.LIQUIDATION, {"liquidation", "liquidated", "cascade"}),
    (EventType.REGULATORY, {"sec", "regulation", "regulatory", "lawsuit", "court", "ban", "etf"}),
    (EventType.MACRO, {"fed", "inflation", "rates", "payroll", "dollar", "tariff"}),
    (EventType.FUNDING, {"funding", "fundraise", "raises", "raised", "investment"}),
    (EventType.PARTNERSHIP, {"partnership", "partners", "collaboration", "integrates"}),
    (EventType.PROTOCOL, {"upgrade", "fork", "mainnet", "validator", "staking"}),
    (EventType.TOKENOMICS, {"unlock", "vesting", "burn", "airdrop", "supply"}),
    (EventType.EXCHANGE, {"exchange", "listing", "delisting", "withdrawals"}),
    (EventType.MARKET_MOVE, {"rally", "selloff", "surge", "plunge", "volatility"}),
)


def _assets_from_text(text: str, seeded: tuple[str, ...] = ()) -> tuple[str, ...]:
    lowered = text.lower()
    assets = list(seeded)
    for asset, keywords in _ASSET_KEYWORDS.items():
        if any(re.search(rf"\b{re.escape(keyword)}\b", lowered) for keyword in keywords):
            assets.append(asset)
    return tuple(dict.fromkeys(assets))


def _source_reliability(
    *,
    source: str | None,
    text: str,
    url: str | None = None,
    published_at: Any = None,
    received_at: Any = None,
    is_duplicate: bool = False,
) -> float:
    """Return a bounded evidence-completeness prior without external lookups.

    This deliberately does not claim that a named publisher is truthful.  It rewards
    locally observable provenance fields and sufficient source text, and penalizes a
    duplicate.  The same deterministic fallback is available to rule and student
    providers when an older trained artifact has no reliability head.
    """
    normalized_source = str(source or "").strip().lower()
    value = 0.30
    if normalized_source and normalized_source not in {"unknown", "untitled"}:
        value += 0.15
    if str(url or "").lower().startswith("https://"):
        value += 0.10
    if published_at is not None:
        value += 0.08
    if received_at is not None:
        value += 0.07
    value += min(len(tokenize_news_text(text)), 200) / 1_000.0
    if is_duplicate:
        value -= 0.20
    return _clamp(value, 0.05, 0.95)


def _affected_asset_probabilities(
    text: str,
    assets: tuple[str, ...] | list[str],
    *,
    seeded: tuple[str, ...] = (),
) -> dict[str, float]:
    lowered = text.lower()
    seeded_assets = {str(asset).upper() for asset in seeded}
    probabilities: dict[str, float] = {}
    for raw_asset in assets:
        asset = str(raw_asset).upper()
        keywords = _ASSET_KEYWORDS.get(asset, {asset.lower()})
        hits = sum(
            len(re.findall(rf"\b{re.escape(str(keyword).lower())}\b", lowered))
            for keyword in keywords
        )
        probability = 0.45 + min(hits, 4) * 0.10 + (0.20 if asset in seeded_assets else 0.0)
        probabilities[asset] = _clamp(probability, 0.05, 0.95)
    return probabilities


def _safe_asset_probabilities(
    value: Any,
    *,
    assets: tuple[str, ...] | list[str],
    fallback: Mapping[str, Any],
    text: str,
    seeded: tuple[str, ...] = (),
) -> dict[str, float]:
    normalized_assets = tuple(dict.fromkeys(str(asset).upper() for asset in assets))
    deterministic = _affected_asset_probabilities(text, normalized_assets, seeded=seeded)
    source = value if isinstance(value, Mapping) else fallback
    output: dict[str, float] = {}
    for asset in normalized_assets:
        raw = source.get(asset) if isinstance(source, Mapping) else None
        if raw is None and isinstance(source, Mapping):
            raw = source.get(asset.lower())
        try:
            probability = float(raw) if raw is not None else deterministic.get(asset, 0.5)
            if not math.isfinite(probability):
                raise ValueError
        except (TypeError, ValueError):
            probability = deterministic.get(asset, 0.5)
        output[asset] = _clamp(probability, 0.0, 1.0)
    return output


def _first_claims(text: str, *, limit: int = 3) -> list[dict[str, Any]]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    claims: list[dict[str, Any]] = []
    for sentence in sentences:
        numbers = re.findall(r"(?<![A-Za-z0-9_])[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?", sentence)
        if numbers:
            claims.append({"claim": sentence[:1_000], "numbers": numbers[:12]})
        if len(claims) >= limit:
            break
    return claims


class LocalRuleProvider:
    """Always-available deterministic Level-0 news analysis."""

    name = "local_rule"
    model = "deterministic-news-rules-v1"
    is_external = False

    async def enrich(self, document: NewsDocument, *, prompt_version: str = "local-rule-v1") -> ProviderResponse:
        tokens = set(tokenize_news_text(document.text))
        positive = len(tokens & _POSITIVE_WORDS)
        negative = len(tokens & _NEGATIVE_WORDS)
        severe = len(tokens & _SEVERE_WORDS)
        event_type = EventType.OTHER
        event_hits = 0
        for candidate_type, keywords in _EVENT_KEYWORDS:
            hits = len(tokens & keywords)
            if hits > event_hits:
                event_type = candidate_type
                event_hits = hits
        raw_sentiment = (positive - negative) / max(1, positive + negative)
        sentiment = _clamp(raw_sentiment, -1.0, 1.0)
        if sentiment > 0.15:
            direction = EventDirection.BULLISH
        elif sentiment < -0.15:
            direction = EventDirection.BEARISH
        elif positive and negative:
            direction = EventDirection.MIXED
        else:
            direction = EventDirection.NEUTRAL
        word_count = len(tokens)
        severity = _clamp(0.10 + severe * 0.18 + (0.08 if event_type == EventType.SECURITY_INCIDENT else 0.0), 0.0, 1.0)
        importance = _clamp(0.10 + event_hits * 0.13 + severe * 0.12 + min(word_count, 500) / 4_000, 0.0, 1.0)
        novelty = _clamp(0.10 if document.is_duplicate else 0.35 + min(word_count, 400) / 2_000, 0.0, 1.0)
        confidence = _clamp(0.35 + min(positive + negative + event_hits, 6) * 0.07, 0.0, 0.85)
        if severity >= 0.70:
            horizon = TimeHorizon.IMMEDIATE
        elif event_type in {EventType.MACRO, EventType.REGULATORY, EventType.PROTOCOL}:
            horizon = TimeHorizon.MEDIUM_TERM
        elif event_type == EventType.OTHER:
            horizon = TimeHorizon.UNKNOWN
        else:
            horizon = TimeHorizon.SHORT_TERM
        summary = document.title
        if len(summary) < 20:
            summary = document.content[:300]
        affected_assets = _assets_from_text(document.text, document.relevant_assets)
        source_reliability = _source_reliability(
            source=document.source,
            text=document.text,
            url=document.url,
            published_at=document.published_at,
            received_at=document.received_at,
            is_duplicate=document.is_duplicate,
        )
        asset_probabilities = _affected_asset_probabilities(
            document.text,
            affected_assets,
            seeded=document.relevant_assets,
        )
        event = StructuredNewsEvent.from_mapping(
            {
                "event_type": event_type.value,
                "affected_assets": affected_assets,
                "affected_asset_probabilities": asset_probabilities,
                "affected_entities": (),
                "direction": direction.value,
                "sentiment": sentiment,
                "severity": severity,
                "importance": importance,
                "novelty": novelty,
                "time_horizon": horizon.value,
                "factual_claims": _first_claims(document.text),
                "confidence": confidence,
                "source_summary": summary[:1_000],
                "source_reliability": source_reliability,
                "metadata": {
                    "level": 0,
                    "positive_keyword_count": positive,
                    "negative_keyword_count": negative,
                    "event_keyword_count": event_hits,
                    "is_duplicate": document.is_duplicate,
                    "source_reliability": source_reliability,
                    "affected_asset_probabilities": asset_probabilities,
                },
            },
            provider=self.name,
            model=self.model,
            prompt_version=prompt_version,
            source_reference=document.source_reference,
            source_text=document.text,
        )
        return ProviderResponse(provider=self.name, model=self.model, event=event)


def _softmax_log_scores(scores: Mapping[str, float]) -> tuple[str, float]:
    if not scores:
        return "other", 0.0
    maximum = max(scores.values())
    weights = {label: math.exp(score - maximum) for label, score in scores.items()}
    total = sum(weights.values()) or 1.0
    best = max(scores, key=scores.get)
    return best, weights[best] / total


def validate_json_student_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the small, dependency-free student artifact format."""

    if artifact.get("artifact_type") != "anata_news_student_naive_bayes_v1":
        raise IntelligenceValidationError("unsupported JSON student artifact type")
    tasks = artifact.get("tasks")
    if not isinstance(tasks, Mapping) or "sentiment" not in tasks or "event_type" not in tasks:
        raise IntelligenceValidationError("student artifact must include sentiment and event_type tasks")
    for name, task in tasks.items():
        if not isinstance(task, Mapping) or not isinstance(task.get("label_counts"), Mapping):
            raise IntelligenceValidationError(f"student artifact task {name!r} is invalid")


def load_json_student_artifact(
    path: str | Path,
    *,
    artifact_member: str | None = None,
    verify_package: bool = True,
) -> dict[str, Any]:
    """Load one declarative student from JSON or a checksummed registry ZIP."""
    target = Path(path)
    if not target.is_file():
        raise ProviderUnavailableError("local student artifact is not installed")
    try:
        if target.suffix.lower() == ".json":
            payload = json.loads(target.read_text(encoding="utf-8"))
        elif target.suffix.lower() == ".zip":
            if verify_package:
                # Local import keeps the provider schema independent of registry
                # initialization while applying the same durable-package verifier.
                from app.pipeline.artifact_store import verify_package_checksum_manifest

                verify_package_checksum_manifest(target.read_bytes(), require_manifest=True)
            with zipfile.ZipFile(target) as archive:
                names = archive.namelist()
                if len(names) != len(set(names)):
                    raise IntelligenceValidationError("student package contains duplicate members")
                member = artifact_member
                if not member and "model_metadata.json" in names:
                    metadata = json.loads(archive.read("model_metadata.json").decode("utf-8"))
                    if isinstance(metadata, Mapping):
                        member = str(metadata.get("model_file") or metadata.get("artifact") or "") or None
                member = member or NEWS_STUDENT_ARTIFACT_MEMBER
                matched = next(
                    (
                        name
                        for name in names
                        if name == member or Path(name).name == Path(member).name
                    ),
                    None,
                )
                if matched is None or Path(matched).suffix.lower() != ".json":
                    raise IntelligenceValidationError("student package has no declared JSON artifact")
                payload = json.loads(archive.read(matched).decode("utf-8"))
        else:
            raise IntelligenceValidationError("local student artifact must be JSON or a package ZIP")
    except IntelligenceProviderError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        raise ProviderUnavailableError("local student artifact could not be loaded") from exc
    if not isinstance(payload, dict):
        raise IntelligenceValidationError("student artifact must contain a JSON object")
    validate_json_student_artifact(payload)
    return payload


def predict_json_student_artifact(
    artifact: Mapping[str, Any],
    text: str,
    *,
    source: str | None = None,
    url: str | None = None,
    published_at: Any = None,
    received_at: Any = None,
    is_duplicate: bool = False,
    seeded_assets: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Run the compact JSON Naive-Bayes student without scikit-learn."""

    validate_json_student_artifact(artifact)
    token_counts = Counter(tokenize_news_text(text))
    vocabulary_size = max(1, int(artifact.get("vocabulary_size", 1)))
    predictions: dict[str, tuple[str, float]] = {}
    for task_name, task in artifact["tasks"].items():
        label_counts = task.get("label_counts", {})
        total_rows = max(1, sum(int(value) for value in label_counts.values()))
        per_label_tokens = task.get("token_counts", {})
        token_totals = task.get("token_totals", {})
        scores: dict[str, float] = {}
        for label, row_count in label_counts.items():
            label = str(label)
            score = math.log((int(row_count) + 1) / (total_rows + len(label_counts)))
            label_tokens = per_label_tokens.get(label, {})
            denominator = max(1, int(token_totals.get(label, 0)) + vocabulary_size)
            for token, count in token_counts.items():
                score += count * math.log((int(label_tokens.get(token, 0)) + 1) / denominator)
            scores[label] = score
        predictions[task_name] = _softmax_log_scores(scores)
    sentiment_label, sentiment_confidence = predictions.get("sentiment", ("neutral", 0.0))
    sentiment_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0, "bullish": 1.0, "bearish": -1.0}
    sentiment = sentiment_map.get(sentiment_label.lower(), 0.0)
    event_type, event_type_probability = predictions.get("event_type", (EventType.OTHER.value, 0.0))
    numeric_means = artifact.get("numeric_means", {})
    asset_keywords = artifact.get("asset_keywords", {})
    assets = [
        str(asset).upper()
        for asset, keywords in asset_keywords.items()
        if any(str(keyword).lower() in text.lower() for keyword in (keywords or ()))
    ]
    normalized_assets = tuple(dict.fromkeys([*seeded_assets, *assets]))
    asset_probabilities = _safe_asset_probabilities(
        artifact.get("affected_asset_probabilities"),
        assets=normalized_assets,
        fallback={},
        text=text,
        seeded=seeded_assets,
    )
    return {
        "sentiment": sentiment,
        "sentiment_label": sentiment_label,
        "sentiment_confidence": sentiment_confidence,
        "event_type": event_type if event_type in {item.value for item in EventType} else EventType.OTHER.value,
        "event_type_probability": event_type_probability,
        "importance": _clamp(float(numeric_means.get("importance", 0.35)), 0.0, 1.0),
        "severity": _clamp(float(numeric_means.get("severity", 0.20)), 0.0, 1.0),
        "novelty": _clamp(float(numeric_means.get("novelty", 0.40)), 0.0, 1.0),
        "confidence": _clamp((sentiment_confidence + event_type_probability) / 2, 0.0, 1.0),
        "affected_assets": normalized_assets,
        "affected_asset_probabilities": asset_probabilities,
        "source_reliability": _source_reliability(
            source=source,
            text=text,
            url=url,
            published_at=published_at,
            received_at=received_at,
            is_duplicate=is_duplicate,
        ),
        "time_horizon": str(artifact.get("default_time_horizon", TimeHorizon.SHORT_TERM.value)),
        "local_model_version": str(artifact.get("version", "unknown")),
    }


class LocalStudentProvider:
    """Level-1 provider for a small JSON or scikit-learn-compatible local artifact."""

    name = "local_student"
    is_external = False

    def __init__(
        self,
        artifact_path: str | Path | None = None,
        *,
        predictor: Callable[[str], Mapping[str, Any] | Awaitable[Mapping[str, Any]]] | None = None,
        model_name: str | None = None,
        artifact_member: str | None = None,
    ) -> None:
        self.artifact_path = Path(artifact_path) if artifact_path else None
        self.predictor = predictor
        self.artifact_member = artifact_member
        self.model = model_name or (self.artifact_path.stem if self.artifact_path else "local-student")
        self._artifact: Any = None
        self._loaded = False
        self._fallback = LocalRuleProvider()

    @classmethod
    def from_artifact(
        cls,
        artifact_path: str | Path,
        *,
        model_name: str | None = None,
        artifact_member: str | None = None,
    ) -> "LocalStudentProvider":
        """Eagerly validate one installed artifact before selecting the provider."""
        provider = cls(
            artifact_path,
            model_name=model_name,
            artifact_member=artifact_member,
        )
        artifact = load_json_student_artifact(
            artifact_path,
            artifact_member=artifact_member,
        )
        provider._artifact = artifact
        provider._loaded = True
        if not model_name:
            provider.model = str(artifact.get("version") or provider.model)
        return provider

    def _load_artifact(self) -> Any:
        if self._loaded:
            return self._artifact
        self._loaded = True
        if not self.artifact_path:
            return None
        if not self.artifact_path.exists():
            raise ProviderUnavailableError("local student artifact is not installed")
        try:
            if self.artifact_path.suffix.lower() in {".json", ".zip"}:
                self._artifact = load_json_student_artifact(
                    self.artifact_path,
                    artifact_member=self.artifact_member,
                )
            else:
                import joblib  # Existing optional lightweight dependency; imported only when used.

                self._artifact = joblib.load(self.artifact_path)
        except IntelligenceProviderError:
            raise
        except Exception as exc:  # Do not expose paths or artifact internals through a public error.
            raise ProviderUnavailableError("local student artifact could not be loaded") from exc
        return self._artifact

    async def _predict(self, document: NewsDocument) -> Mapping[str, Any]:
        if self.predictor is not None:
            outcome = self.predictor(document.text)
            if hasattr(outcome, "__await__"):
                outcome = await outcome
            if not isinstance(outcome, Mapping):
                raise ProviderResponseError("local student predictor must return an object")
            return outcome
        artifact = self._load_artifact()
        if isinstance(artifact, Mapping):
            return predict_json_student_artifact(
                artifact,
                document.text,
                source=document.source,
                url=document.url,
                published_at=document.published_at,
                received_at=document.received_at,
                is_duplicate=document.is_duplicate,
                seeded_assets=document.relevant_assets,
            )
        if artifact is None:
            raise ProviderUnavailableError("local student artifact is not installed")
        try:
            predicted = artifact.predict([document.text])
            label = str(predicted[0]) if predicted else "neutral"
            probability = 0.5
            if hasattr(artifact, "predict_proba"):
                probabilities = artifact.predict_proba([document.text])
                probability = float(max(probabilities[0]))
            return {"sentiment_label": label, "sentiment_confidence": probability}
        except Exception as exc:
            raise ProviderResponseError("local student artifact returned an invalid prediction") from exc

    async def enrich(self, document: NewsDocument, *, prompt_version: str = "local-student-v1") -> ProviderResponse:
        baseline = await self._fallback.enrich(document, prompt_version="local-rule-for-student-v1")
        predicted = dict(await self._predict(document))
        base = baseline.event.model_dump()
        sentiment = predicted.get("sentiment", predicted.get("sentiment_score"))
        if sentiment is None:
            label = str(predicted.get("sentiment_label", "neutral")).lower()
            sentiment = {"positive": 1.0, "bullish": 1.0, "negative": -1.0, "bearish": -1.0}.get(label, 0.0)
        sentiment = _clamp(float(sentiment), -1.0, 1.0)
        direction = EventDirection.BULLISH.value if sentiment > 0.15 else EventDirection.BEARISH.value if sentiment < -0.15 else EventDirection.NEUTRAL.value
        affected_assets = tuple(predicted.get("affected_assets", base["affected_assets"]) or ())
        base_metadata = dict(base.get("metadata") or {})
        source_reliability = predicted.get("source_reliability", base.get("source_reliability", 0.5))
        try:
            source_reliability = float(source_reliability)
            if not math.isfinite(source_reliability):
                raise ValueError
        except (TypeError, ValueError):
            source_reliability = float(base.get("source_reliability", 0.5))
        source_reliability = _clamp(source_reliability, 0.0, 1.0)
        asset_probabilities = _safe_asset_probabilities(
            predicted.get("affected_asset_probabilities"),
            assets=affected_assets,
            fallback=base.get("affected_asset_probabilities", {}),
            text=document.text,
            seeded=document.relevant_assets,
        )
        base.update(
            {
                "event_type": predicted.get("event_type", base["event_type"]),
                "affected_assets": affected_assets,
                "affected_asset_probabilities": asset_probabilities,
                "direction": predicted.get("direction", direction),
                "sentiment": sentiment,
                "severity": predicted.get("severity", base["severity"]),
                "importance": predicted.get("importance", base["importance"]),
                "novelty": predicted.get("novelty", base["novelty"]),
                "time_horizon": predicted.get("time_horizon", base["time_horizon"]),
                "confidence": predicted.get(
                    "confidence", predicted.get("sentiment_confidence", base["confidence"])
                ),
                "source_summary": base["source_summary"],
                "source_reliability": source_reliability,
                "metadata": {
                    **base_metadata,
                    "level": 1,
                    "event_type_probability": predicted.get("event_type_probability"),
                    "sentiment_confidence": predicted.get("sentiment_confidence"),
                    "local_model_version": predicted.get("local_model_version", self.model),
                    "source_reliability": source_reliability,
                    "affected_asset_probabilities": asset_probabilities,
                },
            }
        )
        event = StructuredNewsEvent.from_mapping(
            base,
            provider=self.name,
            model=self.model,
            prompt_version=prompt_version,
            source_reference=document.source_reference,
            source_text=document.text,
        )
        return ProviderResponse(provider=self.name, model=self.model, event=event)


class GenericOpenAICompatibleProvider:
    """OpenAI-compatible HTTP adapter with no SDK dependency and no secret logging."""

    name = "generic_openai_compatible"
    is_external = True

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        model: str,
        timeout_seconds: float = 15.0,
        max_article_characters: int = 16_000,
        max_tokens: int = 800,
        input_cost_per_million_usd: float | None = None,
        output_cost_per_million_usd: float | None = None,
        client: httpx.AsyncClient | None = None,
        extra_headers: Mapping[str, str] | None = None,
        supports_json_mode: bool = True,
        provider_name: str | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url is required")
        if not model.strip():
            raise ValueError("model is required")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or None
        self.model = model.strip()
        self.timeout_seconds = max(0.1, float(timeout_seconds))
        self.max_article_characters = max(1_000, int(max_article_characters))
        self.max_tokens = max(64, int(max_tokens))
        self.input_cost_per_million_usd = (
            max(0.0, float(input_cost_per_million_usd)) if input_cost_per_million_usd is not None else None
        )
        self.output_cost_per_million_usd = (
            max(0.0, float(output_cost_per_million_usd)) if output_cost_per_million_usd is not None else None
        )
        self.client = client
        self.extra_headers = dict(extra_headers or {})
        self.supports_json_mode = supports_json_mode
        self.name = provider_name or self.name

    @property
    def endpoint(self) -> str:
        return self.base_url if self.base_url.endswith("/chat/completions") else f"{self.base_url}/chat/completions"

    @property
    def estimated_request_cost_usd(self) -> float:
        """Conservative upper estimate used by the router's pre-request budget gate.

        Unknown pricing deliberately evaluates to infinity.  An operator must
        explicitly configure a rate or a router-specific allowance before an
        enabled provider can be called; an API key never implies a free tier.
        """

        if self.input_cost_per_million_usd is None or self.output_cost_per_million_usd is None:
            return float("inf")
        estimated_input_tokens = self.max_article_characters / 4
        return (
            estimated_input_tokens * self.input_cost_per_million_usd
            + self.max_tokens * self.output_cost_per_million_usd
        ) / 1_000_000

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You extract factual crypto-news context. Return only one JSON object with exactly these required "
            "fields: event_type (adoption|exchange|funding|liquidation|macro|market_move|market_structure|"
            "partnership|product|protocol|regulatory|rumor|security_incident|stablecoin|tokenomics|other), "
            "affected_assets (array of upper-case tickers), affected_entities (array of names), direction "
            "(bullish|bearish|neutral|mixed), sentiment (-1..1), severity (0..1), importance (0..1), "
            "novelty (0..1), time_horizon (immediate|short_term|medium_term|long_term|unknown), "
            "factual_claims (array of {claim,numbers}), confidence (0..1), and source_summary. "
            "Use only facts from the article. Do not provide trading advice, position sizes, leverage, orders, or code."
        )

    @staticmethod
    def _extract_content(payload: Mapping[str, Any]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderResponseError("provider response has no choices")
        message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
        content = message.get("content") if isinstance(message, Mapping) else None
        if isinstance(content, list):
            content = "".join(
                str(part.get("text", "")) if isinstance(part, Mapping) else str(part) for part in content
            )
        if not isinstance(content, str) or not content.strip():
            raise ProviderResponseError("provider response has no JSON content")
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE).strip()
        return content

    def _usage(self, payload: Mapping[str, Any]) -> ProviderUsage:
        usage = payload.get("usage") if isinstance(payload.get("usage"), Mapping) else {}
        input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
        output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
        try:
            input_tokens = int(input_tokens) if input_tokens is not None else None
            output_tokens = int(output_tokens) if output_tokens is not None else None
        except (TypeError, ValueError):
            input_tokens = output_tokens = None
        cost = 0.0
        if self.input_cost_per_million_usd is not None and self.output_cost_per_million_usd is not None:
            cost = (
                (input_tokens or 0) * self.input_cost_per_million_usd
                + (output_tokens or 0) * self.output_cost_per_million_usd
            ) / 1_000_000
        return ProviderUsage(input_tokens=input_tokens, output_tokens=output_tokens, estimated_cost_usd=cost)

    async def _post(self, payload: Mapping[str, Any], headers: Mapping[str, str]) -> httpx.Response:
        if self.client is not None:
            return await self.client.post(self.endpoint, json=payload, headers=headers, timeout=self.timeout_seconds)
        async with httpx.AsyncClient() as client:
            return await client.post(self.endpoint, json=payload, headers=headers, timeout=self.timeout_seconds)

    async def enrich(self, document: NewsDocument, *, prompt_version: str = "external-news-v1") -> ProviderResponse:
        if not self.api_key:
            raise ProviderUnavailableError("external provider key is not configured")
        source_payload = {
            "title": document.title,
            "source": document.source,
            "source_reference": document.source_reference,
            "published_at": document.published_at.isoformat() if document.published_at else None,
            "known_assets": list(document.relevant_assets),
            "article": document.content[: self.max_article_characters],
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": json.dumps(source_payload, ensure_ascii=False)},
            ],
        }
        if self.supports_json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json", **self.extra_headers}
        try:
            response = await self._post(payload, headers)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError("external provider timed out") from exc
        except httpx.HTTPError as exc:
            raise IntelligenceProviderError("external provider request failed") from exc
        if response.status_code == 429:
            raise ProviderRateLimitError("external provider rate limited the request")
        if response.status_code >= 500:
            raise IntelligenceProviderError("external provider server error")
        if response.status_code >= 400:
            raise ProviderResponseError("external provider rejected the request")
        try:
            response_data = response.json()
            content = self._extract_content(response_data)
            event_payload = json.loads(content)
            if isinstance(event_payload, Mapping) and isinstance(event_payload.get("event"), Mapping):
                event_payload = event_payload["event"]
            event = StructuredNewsEvent.from_mapping(
                event_payload,
                provider=self.name,
                model=str(response_data.get("model") or self.model),
                prompt_version=prompt_version,
                source_reference=document.source_reference,
                source_text=document.text,
            )
        except (ValueError, TypeError, IntelligenceValidationError) as exc:
            raise ProviderResponseError("external provider returned invalid structured JSON") from exc
        return ProviderResponse(
            provider=self.name,
            model=str(response_data.get("model") or self.model),
            event=event,
            usage=self._usage(response_data),
            raw_response=response_data,
        )
