from __future__ import annotations

import re

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
    "breach",
    "contagion",
    "default",
    "hack",
    "liquidation",
    "regulation",
    "sanction",
    "sec",
    "war",
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
}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def analyze_news(text: str) -> tuple[float, float, list[str], list[str]]:
    """Placeholder interface ready to swap for a Hugging Face model later."""
    tokens = _tokens(text)
    positive = len(tokens & POSITIVE_WORDS)
    negative = len(tokens & NEGATIVE_WORDS)
    risk_hits = len(tokens & RISK_WORDS)

    sentiment_score = _clamp((positive - negative) / max(positive + negative, 1), -1.0, 1.0)
    risk_score = _clamp(risk_hits / 4.0, 0.0, 1.0)
    topics = [topic for topic, words in TOPIC_KEYWORDS.items() if tokens & words]
    affected_symbols = [symbol for symbol, words in SYMBOL_KEYWORDS.items() if tokens & words]
    return sentiment_score, risk_score, topics, affected_symbols

