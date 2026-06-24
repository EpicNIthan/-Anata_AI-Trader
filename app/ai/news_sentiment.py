from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

POSITIVE_WORDS = {
    "adoption",
    "approval",
    "beat",
    "bull",
    "bullish",
    "growth",
    "inflow",
    "partnership",
    "rally",
    "record",
    "surge",
    "upgrade",
}
NEGATIVE_WORDS = {
    "ban",
    "bear",
    "bearish",
    "crash",
    "decline",
    "drop",
    "exploit",
    "hack",
    "lawsuit",
    "outflow",
    "probe",
    "selloff",
}
RISK_WORDS = {
    "bankruptcy",
    "ban",
    "breach",
    "contagion",
    "crash",
    "default",
    "depeg",
    "exploit",
    "fed",
    "hack",
    "inflation",
    "lawsuit",
    "liquidation",
    "regulation",
    "sanction",
    "sec",
    "war",
}
RISK_PHRASES = {
    "withdrawal halt",
}
TOPIC_KEYWORDS = {
    "regulation": {"sec", "etf", "regulation", "lawsuit", "court", "ban"},
    "security": {"hack", "exploit", "breach", "wallet", "bridge"},
    "macro": {"fed", "inflation", "rates", "jobs", "dollar", "war"},
    "market": {"rally", "selloff", "liquidation", "volume", "volatility"},
}
SYMBOL_KEYWORDS = {
    "BTCUSDT": {"btc", "bitcoin"},
    "ETHUSDT": {"eth", "ethereum", "ether"},
    "SOLUSDT": {"sol", "solana"},
    "BNBUSDT": {"bnb", "binance"},
    "XRPUSDT": {"xrp", "ripple"},
}
LABEL_TO_SCORE = {
    "positive": 1.0,
    "neutral": 0.0,
    "negative": -1.0,
}


@dataclass(frozen=True)
class SentimentResult:
    sentiment_score: float
    risk_score: float
    topics: list[str]
    affected_symbols: list[str]
    model_name: str
    label: str
    confidence: float
    raw_payload: dict[str, Any]

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _keyword_analysis(text: str) -> tuple[float, float, list[str], list[str]]:
    tokens = _tokens(text)
    positive = len(tokens & POSITIVE_WORDS)
    negative = len(tokens & NEGATIVE_WORDS)
    risk_hits = len(tokens & RISK_WORDS)
    lowered = text.lower()
    risk_hits += sum(1 for phrase in RISK_PHRASES if phrase in lowered)

    sentiment_score = _clamp((positive - negative) / max(positive + negative, 1), -1.0, 1.0)
    risk_score = _clamp(risk_hits / 4.0, 0.0, 1.0)
    topics = [topic for topic, words in TOPIC_KEYWORDS.items() if tokens & words]
    affected_symbols = [symbol for symbol, words in SYMBOL_KEYWORDS.items() if tokens & words]
    return sentiment_score, risk_score, topics, affected_symbols


@lru_cache(maxsize=1)
def _hf_pipeline() -> Any | None:
    if not settings.enable_hf_sentiment:
        return None
    try:
        from transformers import pipeline

        logger.info("Loading Hugging Face sentiment model: %s", settings.news_sentiment_model)
        return pipeline("sentiment-analysis", model=settings.news_sentiment_model)
    except Exception:
        logger.exception("Failed to load Hugging Face sentiment model; falling back to rules")
        return None


def _hf_analysis(text: str) -> tuple[str, float, dict[str, Any]] | None:
    model = _hf_pipeline()
    if model is None:
        return None
    try:
        result = model(text[:1800], truncation=True)
    except Exception:
        logger.exception("Hugging Face sentiment inference failed; falling back to rules")
        return None
    first = result[0] if isinstance(result, list) and result else result
    if not isinstance(first, dict):
        return None
    label = str(first.get("label") or "neutral").lower()
    confidence = float(first.get("score") or 0.0)
    if "positive" in label:
        label = "positive"
    elif "negative" in label:
        label = "negative"
    else:
        label = "neutral"
    return label, confidence, {"provider": "huggingface", "result": first}


def analyze_news(text: str) -> SentimentResult:
    """Analyze news with optional HF model and rule-based fallback/risk overlay."""
    keyword_sentiment, risk_score, topics, affected_symbols = _keyword_analysis(text)
    hf_result = _hf_analysis(text)
    if hf_result:
        label, confidence, raw_model = hf_result
        sentiment_score = LABEL_TO_SCORE[label] * confidence
        model_name = settings.news_sentiment_model
    else:
        sentiment_score = keyword_sentiment
        confidence = min(abs(keyword_sentiment), 0.75) if keyword_sentiment else 0.50
        if sentiment_score > 0:
            label = "positive"
        elif sentiment_score < 0:
            label = "negative"
        else:
            label = "neutral"
        model_name = "rule-based-fallback-v1"
        raw_model = {"provider": "rules", "hf_enabled": settings.enable_hf_sentiment}

    return SentimentResult(
        sentiment_score=_clamp(sentiment_score, -1.0, 1.0),
        risk_score=risk_score,
        topics=topics,
        affected_symbols=affected_symbols,
        model_name=model_name,
        label=label,
        confidence=_clamp(confidence, 0.0, 1.0),
        raw_payload={
            **raw_model,
            "risk_overlay_words": sorted(_tokens(text) & RISK_WORDS),
            "risk_overlay_phrases": [phrase for phrase in RISK_PHRASES if phrase in text.lower()],
            "active_model": active_sentiment_model(),
        },
    )


def active_sentiment_model() -> dict[str, Any]:
    hf_loaded = _hf_pipeline() is not None if settings.enable_hf_sentiment else False
    return {
        "requested_model": settings.news_sentiment_model,
        "active_model": settings.news_sentiment_model if hf_loaded else "rule-based-fallback-v1",
        "hf_enabled": settings.enable_hf_sentiment,
        "hf_loaded": hf_loaded,
        "fallback": not hf_loaded,
        "future_models": ["ProsusAI/finbert", "ElKulako/cryptobert"],
    }
