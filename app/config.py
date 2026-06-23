from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _csv(value: str | None, default: str) -> list[str]:
    raw = value if value is not None else default
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def _float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return float(value)


def _int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return int(value)


def _bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _secret(name: str) -> str | None:
    value = os.getenv(name)
    if value in (None, ""):
        return None
    cleaned = value.strip()
    placeholders = {"your_news_api_key", "your_real_key_here", "your_api_key_here", "change_me"}
    if cleaned.lower() in placeholders:
        return None
    return cleaned


def _database_url() -> str:
    url = os.getenv("DATABASE_URL", "sqlite:///./trading_lab.db")
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url.removeprefix("postgres://")
    if url.startswith("postgresql://") and "+psycopg" not in url:
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


@dataclass(frozen=True)
class Settings:
    database_url: str = field(default_factory=_database_url)
    binance_symbols: list[str] = field(
        default_factory=lambda: _csv(os.getenv("BINANCE_SYMBOLS"), "BTCUSDT,ETHUSDT")
    )
    binance_interval: str = os.getenv("BINANCE_INTERVAL", "1m")
    store_live_candle_updates: bool = field(default_factory=lambda: _bool("STORE_LIVE_CANDLE_UPDATES", True))
    binance_rest_base_url: str = os.getenv("BINANCE_REST_BASE_URL", "https://data-api.binance.vision")
    binance_ws_base_url: str = os.getenv("BINANCE_WS_BASE_URL", "wss://data-stream.binance.vision")
    news_api_key: str | None = field(default_factory=lambda: _secret("NEWS_API_KEY"))
    news_provider_url: str = os.getenv("NEWS_PROVIDER_URL", "https://newsapi.org/v2/everything")
    news_query: str = os.getenv("NEWS_QUERY", "crypto OR bitcoin OR ethereum OR macro economy")
    news_poll_seconds: int = field(default_factory=lambda: _int("NEWS_POLL_SECONDS", 300))
    news_mock_fallback_enabled: bool = field(default_factory=lambda: _bool("NEWS_MOCK_FALLBACK_ENABLED"))
    paper_start_balance: float = field(default_factory=lambda: _float("PAPER_START_BALANCE", 10000.0))
    model_dir: Path = field(default_factory=lambda: Path(os.getenv("MODEL_DIR", "./models")))
    trading_mode: str = os.getenv("TRADING_MODE", "paper").lower()
    enable_market_collector: bool = field(default_factory=lambda: _bool("ENABLE_MARKET_COLLECTOR"))
    enable_news_collector: bool = field(default_factory=lambda: _bool("ENABLE_NEWS_COLLECTOR"))
    auto_trader_enabled: bool = field(default_factory=lambda: _bool("AUTO_TRADER_ENABLED"))
    auto_trader_interval_seconds: int = field(default_factory=lambda: _int("AUTO_TRADER_INTERVAL_SECONDS", 60))
    auto_trader_symbols: list[str] = field(
        default_factory=lambda: _csv(os.getenv("AUTO_TRADER_SYMBOLS"), "BTCUSDT,ETHUSDT")
    )
    paper_trade_timeframe: str = os.getenv("PAPER_TRADE_TIMEFRAME", "1m")
    paper_fee_rate: float = field(default_factory=lambda: _float("PAPER_FEE_RATE", 0.001))
    risk_max_trade_size_pct: float = field(default_factory=lambda: _float("RISK_MAX_TRADE_SIZE_PCT", 0.10))
    risk_max_daily_loss_pct: float = field(default_factory=lambda: _float("RISK_MAX_DAILY_LOSS_PCT", 0.05))
    risk_max_open_positions: int = field(default_factory=lambda: _int("RISK_MAX_OPEN_POSITIONS", 3))
    risk_min_confidence: float = field(default_factory=lambda: _float("RISK_MIN_CONFIDENCE", 0.55))
    risk_cooldown_minutes: int = field(default_factory=lambda: _int("RISK_COOLDOWN_MINUTES", 30))
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()

    @property
    def is_paper_mode(self) -> bool:
        return self.trading_mode == "paper"


settings = Settings()
