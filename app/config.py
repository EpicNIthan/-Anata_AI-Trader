from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,AVAXUSDT,LINKUSDT,LTCUSDT"


def _csv(value: str | None, default: str) -> list[str]:
    raw = value if value is not None else default
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def _csv_raw(value: str | None, default: str) -> list[str]:
    raw = value if value is not None else default
    return [item.strip() for item in raw.split(",") if item.strip()]


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
        default_factory=lambda: _csv(os.getenv("BINANCE_SYMBOLS"), DEFAULT_SYMBOLS)
    )
    binance_interval: str = os.getenv("BINANCE_INTERVAL", "1m")
    store_live_candle_updates: bool = field(default_factory=lambda: _bool("STORE_LIVE_CANDLE_UPDATES", True))
    binance_rest_base_url: str = os.getenv("BINANCE_REST_BASE_URL", "https://data-api.binance.vision")
    binance_ws_base_url: str = os.getenv("BINANCE_WS_BASE_URL", "wss://data-stream.binance.vision")
    binance_futures_rest_base_url: str = os.getenv("BINANCE_FUTURES_REST_BASE_URL", "https://fapi.binance.com")
    derivatives_enabled: bool = field(default_factory=lambda: _bool("DERIVATIVES_ENABLED", True))
    enable_derivatives_collector: bool = field(default_factory=lambda: _bool("ENABLE_DERIVATIVES_COLLECTOR"))
    derivatives_poll_interval_seconds: int = field(default_factory=lambda: _int("DERIVATIVES_POLL_INTERVAL_SECONDS", 300))
    derivatives_period: str = os.getenv("DERIVATIVES_PERIOD", "5m")
    derivatives_symbols: list[str] = field(
        default_factory=lambda: _csv(os.getenv("DERIVATIVES_SYMBOLS"), DEFAULT_SYMBOLS)
    )
    news_api_key: str | None = field(default_factory=lambda: _secret("NEWS_API_KEY"))
    news_provider_url: str = os.getenv("NEWS_PROVIDER_URL", "https://newsapi.org/v2/everything")
    news_providers: list[str] = field(
        default_factory=lambda: _csv(os.getenv("NEWS_PROVIDER") or os.getenv("NEWS_PROVIDERS"), "rss,gdelt,newsapi")
    )
    rss_news_enabled: bool = field(default_factory=lambda: _bool("RSS_NEWS_ENABLED", True))
    rss_feeds: list[str] = field(
        default_factory=lambda: _csv_raw(
            os.getenv("RSS_FEEDS"),
            "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml,https://cointelegraph.com/rss,https://decrypt.co/feed",
        )
    )
    rss_request_user_agent: str = os.getenv("RSS_REQUEST_USER_AGENT", "AnataAITrader/1.0 RSS reader")
    news_query: str = os.getenv("NEWS_QUERY", "crypto OR bitcoin OR ethereum OR macro economy")
    news_poll_seconds: int = field(
        default_factory=lambda: _int("NEWS_POLL_INTERVAL_SECONDS", _int("NEWS_POLL_SECONDS", 300))
    )
    news_sentiment_model: str = os.getenv(
        "NEWS_SENTIMENT_MODEL",
        "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
    )
    enable_hf_sentiment: bool = field(default_factory=lambda: _bool("ENABLE_HF_SENTIMENT", False))
    gdelt_enabled: bool = field(default_factory=lambda: _bool("GDELT_ENABLED", True))
    gdelt_poll_interval_seconds: int = field(default_factory=lambda: _int("GDELT_POLL_INTERVAL_SECONDS", 900))
    gdelt_max_records: int = field(default_factory=lambda: _int("GDELT_MAX_RECORDS", 20))
    newsapi_enabled: bool = field(default_factory=lambda: _bool("NEWSAPI_ENABLED", False))
    news_mock_fallback_enabled: bool = field(default_factory=lambda: _bool("NEWS_MOCK_FALLBACK_ENABLED"))
    paper_start_balance: float = field(default_factory=lambda: _float("PAPER_START_BALANCE", 10000.0))
    model_dir: Path = field(default_factory=lambda: Path(os.getenv("MODEL_DIR", "./models")))
    trading_mode: str = os.getenv("TRADING_MODE", "paper").lower()
    enable_market_collector: bool = field(default_factory=lambda: _bool("ENABLE_MARKET_COLLECTOR"))
    enable_news_collector: bool = field(default_factory=lambda: _bool("ENABLE_NEWS_COLLECTOR"))
    auto_trader_enabled: bool = field(default_factory=lambda: _bool("AUTO_TRADER_ENABLED"))
    auto_trader_interval_seconds: int = field(default_factory=lambda: _int("AUTO_TRADER_INTERVAL_SECONDS", 60))
    auto_trader_symbols: list[str] = field(
        default_factory=lambda: _csv(os.getenv("AUTO_TRADER_SYMBOLS"), DEFAULT_SYMBOLS)
    )
    paper_trade_timeframe: str = os.getenv("PAPER_TRADE_TIMEFRAME", "1m")
    paper_fee_rate: float = field(default_factory=lambda: _float("PAPER_FEE_RATE", 0.001))
    paper_leverage: float = field(default_factory=lambda: _float("PAPER_LEVERAGE", 10.0))
    paper_max_leverage: float = field(default_factory=lambda: _float("PAPER_MAX_LEVERAGE", 20.0))
    risk_max_trade_size_pct: float = field(default_factory=lambda: _float("RISK_MAX_TRADE_SIZE_PCT", 0.50))
    risk_max_daily_loss_pct: float = field(default_factory=lambda: _float("RISK_MAX_DAILY_LOSS_PCT", 0.05))
    risk_max_open_positions: int = field(default_factory=lambda: _int("RISK_MAX_OPEN_POSITIONS", 3))
    risk_min_confidence: float = field(default_factory=lambda: _float("RISK_MIN_CONFIDENCE", 0.55))
    risk_cooldown_minutes: int = field(default_factory=lambda: _int("RISK_COOLDOWN_MINUTES", 30))
    strategy_min_edge_after_fees: float = field(default_factory=lambda: _float("STRATEGY_MIN_EDGE_AFTER_FEES", 0.001))
    auto_close_min_net_profit_pct: float = field(default_factory=lambda: _float("AUTO_CLOSE_MIN_NET_PROFIT_PCT", 0.001))
    exploration_mode: bool = field(default_factory=lambda: _bool("EXPLORATION_MODE", False))
    exploration_rate: float = field(default_factory=lambda: _float("EXPLORATION_RATE", 0.05))
    min_paper_trade_notional: float = field(default_factory=lambda: _float("MIN_PAPER_TRADE_NOTIONAL", 50.0))
    live_update_retention_hours: int = field(default_factory=lambda: _int("LIVE_UPDATE_RETENTION_HOURS", 48))
    raw_news_retention_days: int = field(default_factory=lambda: _int("RAW_NEWS_RETENTION_DAYS", 30))
    raw_tick_retention_days: int = field(default_factory=lambda: _int("RAW_TICK_RETENTION_DAYS", 7))
    diagnostic_retention_days: int = field(default_factory=lambda: _int("DIAGNOSTIC_RETENTION_DAYS", 7))
    external_data_retention_days: int = field(default_factory=lambda: _int("EXTERNAL_DATA_RETENTION_DAYS", 365))
    experience_retention_days: int = field(default_factory=lambda: _int("EXPERIENCE_RETENTION_DAYS", 365))
    closed_candle_retention_days: int = field(default_factory=lambda: _int("CLOSED_CANDLE_RETENTION_DAYS", 1095))
    archive_dir: Path = field(default_factory=lambda: Path(os.getenv("ARCHIVE_DIR", "./archives")))
    dashboard_username: str | None = field(default_factory=lambda: _secret("DASHBOARD_USERNAME"))
    dashboard_password: str | None = field(default_factory=lambda: _secret("DASHBOARD_PASSWORD"))
    admin_token: str | None = field(default_factory=lambda: _secret("ADMIN_TOKEN"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()

    @property
    def is_paper_mode(self) -> bool:
        return self.trading_mode == "paper"


settings = Settings()
