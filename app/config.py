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
    return value.strip().strip('"').strip("'").lower() in {"1", "true", "yes", "on"}


def _secret(name: str) -> str | None:
    value = os.getenv(name)
    if value in (None, ""):
        return None
    cleaned = value.strip()
    placeholders = {
        "your_news_api_key",
        "your_real_key_here",
        "your_api_key_here",
        "your_huggingface_token",
        "hf_your_token_here",
        "change_me",
    }
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
    binance_symbols: list[str] = field(default_factory=lambda: _csv(os.getenv("BINANCE_SYMBOLS"), DEFAULT_SYMBOLS))
    binance_interval: str = os.getenv("BINANCE_INTERVAL", "1m")
    store_live_candle_updates: bool = field(default_factory=lambda: _bool("STORE_LIVE_CANDLE_UPDATES", True))
    store_market_ticks: bool = field(default_factory=lambda: _bool("STORE_MARKET_TICKS", False))
    binance_rest_base_url: str = os.getenv("BINANCE_REST_BASE_URL", "https://data-api.binance.vision")
    binance_ws_base_url: str = os.getenv("BINANCE_WS_BASE_URL", "wss://data-stream.binance.vision")
    binance_futures_rest_base_url: str = os.getenv("BINANCE_FUTURES_REST_BASE_URL", "https://fapi.binance.com")
    binance_futures_ws_base_url: str = os.getenv("BINANCE_FUTURES_WS_BASE_URL", "wss://fstream.binance.com")

    # Collector defaults are production/data-factory friendly so Railway does not need a huge variable list.
    derivatives_enabled: bool = field(default_factory=lambda: _bool("DERIVATIVES_ENABLED", True))
    enable_derivatives_collector: bool = field(default_factory=lambda: _bool("ENABLE_DERIVATIVES_COLLECTOR", True))
    derivatives_poll_interval_seconds: int = field(default_factory=lambda: _int("DERIVATIVES_POLL_INTERVAL_SECONDS", 300))
    derivatives_period: str = os.getenv("DERIVATIVES_PERIOD", "5m")
    derivatives_symbols: list[str] = field(default_factory=lambda: _csv(os.getenv("DERIVATIVES_SYMBOLS"), DEFAULT_SYMBOLS))
    enable_fear_greed_collector: bool = field(default_factory=lambda: _bool("ENABLE_FEAR_GREED_COLLECTOR", True))
    enable_global_market_collector: bool = field(default_factory=lambda: _bool("ENABLE_GLOBAL_MARKET_COLLECTOR", True))
    enable_liquidation_collector: bool = field(default_factory=lambda: _bool("ENABLE_LIQUIDATION_COLLECTOR", False))
    enable_stablecoin_risk_collector: bool = field(
        default_factory=lambda: _bool("ENABLE_STABLECOIN_RISK_COLLECTOR", _bool("ENABLE_STABLECOIN_COLLECTOR", True))
    )
    enable_macro_risk_collector: bool = field(default_factory=lambda: _bool("ENABLE_MACRO_RISK_COLLECTOR", True))
    external_collector_interval_seconds: int = field(default_factory=lambda: _int("EXTERNAL_COLLECTOR_INTERVAL_SECONDS", 300))
    liquidation_rollup_seconds: int = field(default_factory=lambda: _int("LIQUIDATION_ROLLUP_SECONDS", 60))
    liquidation_symbols: list[str] = field(default_factory=lambda: _csv(os.getenv("LIQUIDATION_SYMBOLS"), "BTCUSDT,ETHUSDT"))
    store_raw_liquidations: bool = field(default_factory=lambda: _bool("STORE_RAW_LIQUIDATIONS", False))
    store_raw_external_events: bool = field(default_factory=lambda: _bool("STORE_RAW_EXTERNAL_EVENTS", False))
    coingecko_demo_api_key: str | None = field(default_factory=lambda: _secret("COINGECKO_DEMO_API_KEY"))

    news_api_key: str | None = field(default_factory=lambda: _secret("NEWS_API_KEY"))
    news_provider_url: str = os.getenv("NEWS_PROVIDER_URL", "https://newsapi.org/v2/everything")
    news_providers: list[str] = field(default_factory=lambda: _csv(os.getenv("NEWS_PROVIDER") or os.getenv("NEWS_PROVIDERS"), "rss,gdelt,newsapi"))
    rss_news_enabled: bool = field(default_factory=lambda: _bool("RSS_NEWS_ENABLED", True))
    rss_feeds: list[str] = field(
        default_factory=lambda: _csv_raw(
            os.getenv("RSS_FEEDS"),
            "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml,https://cointelegraph.com/rss,https://decrypt.co/feed",
        )
    )
    rss_request_user_agent: str = os.getenv("RSS_REQUEST_USER_AGENT", "AnataAITrader/1.0 RSS reader")
    news_query: str = os.getenv("NEWS_QUERY", "crypto OR bitcoin OR ethereum OR macro economy")
    news_poll_seconds: int = field(default_factory=lambda: _int("NEWS_POLL_INTERVAL_SECONDS", _int("NEWS_POLL_SECONDS", 120)))
    news_sentiment_model: str = os.getenv("NEWS_SENTIMENT_MODEL", "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    enable_hf_sentiment: bool = field(default_factory=lambda: _bool("ENABLE_HF_SENTIMENT", False))
    hf_sentiment_backend: str = os.getenv("HF_SENTIMENT_BACKEND", "api").lower()
    hf_api_token: str | None = (
        _secret("HF_API_TOKEN") or _secret("HUGGINGFACE_API_TOKEN") or _secret("HUGGINGFACEHUB_API_TOKEN")
    )
    hf_api_timeout_seconds: int = field(default_factory=lambda: _int("HF_API_TIMEOUT_SECONDS", 20))
    gdelt_enabled: bool = field(default_factory=lambda: _bool("GDELT_ENABLED", True))
    gdelt_poll_interval_seconds: int = field(default_factory=lambda: _int("GDELT_POLL_INTERVAL_SECONDS", 900))
    gdelt_max_records: int = field(default_factory=lambda: _int("GDELT_MAX_RECORDS", 20))
    newsapi_enabled: bool = field(default_factory=lambda: _bool("NEWSAPI_ENABLED", False))
    news_mock_fallback_enabled: bool = field(default_factory=lambda: _bool("NEWS_MOCK_FALLBACK_ENABLED", False))

    paper_start_balance: float = field(default_factory=lambda: _float("PAPER_START_BALANCE", 10000.0))
    model_dir: Path = field(default_factory=lambda: Path(os.getenv("MODEL_DIR", "./models")))
    trading_mode: str = os.getenv("TRADING_MODE", "paper").lower()
    enable_server_training: bool = field(default_factory=lambda: _bool("ENABLE_SERVER_TRAINING", False))
    enable_server_inference: bool = field(default_factory=lambda: _bool("ENABLE_SERVER_INFERENCE", True))
    model_type: str = os.getenv("MODEL_TYPE", "local-uploaded")
    model_target: str = os.getenv("MODEL_TARGET", "target_trade_quality_score")
    auto_build_labels_on_export: bool = field(default_factory=lambda: _bool("AUTO_BUILD_LABELS_ON_EXPORT", True))
    model_min_rows: int = field(default_factory=lambda: _int("MODEL_MIN_ROWS", 500))
    model_activation_mode: str = os.getenv("MODEL_ACTIVATION_MODE", "manual")
    model_min_directional_accuracy: float = field(default_factory=lambda: _float("MODEL_MIN_DIRECTIONAL_ACCURACY", 0.52))
    model_max_test_drawdown: float = field(default_factory=lambda: _float("MODEL_MAX_TEST_DRAWDOWN", 0.20))
    model_require_positive_net_return: bool = field(default_factory=lambda: _bool("MODEL_REQUIRE_POSITIVE_NET_RETURN", True))
    training_max_rows: int = field(default_factory=lambda: _int("TRAINING_MAX_ROWS", 250_000))
    training_timeout_seconds: int = field(default_factory=lambda: _int("TRAINING_TIMEOUT_SECONDS", 900))
    training_cpu_safe_mode: bool = field(default_factory=lambda: _bool("TRAINING_CPU_SAFE_MODE", True))
    training_disable_collectors_during_heavy_job: bool = field(default_factory=lambda: _bool("TRAINING_DISABLE_COLLECTORS_DURING_HEAVY_JOB", True))
    training_background_only: bool = field(default_factory=lambda: _bool("TRAINING_BACKGROUND_ONLY", True))

    # Paper bot defaults live here. Railway only needs to override these when you intentionally want a different mode.
    enable_market_collector: bool = field(default_factory=lambda: _bool("ENABLE_MARKET_COLLECTOR", True))
    enable_news_collector: bool = field(default_factory=lambda: _bool("ENABLE_NEWS_COLLECTOR", True))
    auto_trader_enabled: bool = field(default_factory=lambda: _bool("AUTO_TRADER_ENABLED", True))
    auto_trader_use_trained_model: bool = field(default_factory=lambda: _bool("AUTO_TRADER_USE_TRAINED_MODEL", False))
    auto_trader_interval_seconds: int = field(default_factory=lambda: _int("AUTO_TRADER_INTERVAL_SECONDS", 60))
    auto_trader_symbols: list[str] = field(default_factory=lambda: _csv(os.getenv("AUTO_TRADER_SYMBOLS"), DEFAULT_SYMBOLS))
    paper_trade_timeframe: str = os.getenv("PAPER_TRADE_TIMEFRAME", "1m")
    paper_fee_rate: float = field(default_factory=lambda: _float("PAPER_FEE_RATE", 0.0004))
    paper_leverage: float = field(default_factory=lambda: _float("PAPER_LEVERAGE", 10.0))
    paper_min_leverage: float = field(default_factory=lambda: _float("PAPER_MIN_LEVERAGE", 1.0))
    paper_max_leverage: float = field(default_factory=lambda: _float("PAPER_MAX_LEVERAGE", 125.0))
    paper_confidence_leverage_enabled: bool = field(default_factory=lambda: _bool("PAPER_CONFIDENCE_LEVERAGE_ENABLED", True))
    risk_max_entry_fee_pct_of_equity: float = field(default_factory=lambda: _float("RISK_MAX_ENTRY_FEE_PCT_OF_EQUITY", 0.01))
    risk_max_trade_size_pct: float = field(default_factory=lambda: _float("RISK_MAX_TRADE_SIZE_PCT", 0.10))
    risk_max_daily_loss_pct: float = field(default_factory=lambda: _float("RISK_MAX_DAILY_LOSS_PCT", 0.05))
    risk_max_open_positions: int = field(default_factory=lambda: _int("RISK_MAX_OPEN_POSITIONS", 10))
    risk_min_confidence: float = field(default_factory=lambda: _float("RISK_MIN_CONFIDENCE", 0.55))
    risk_cooldown_minutes: int = field(default_factory=lambda: _int("RISK_COOLDOWN_MINUTES", 30))
    strategy_min_edge_after_fees: float = field(default_factory=lambda: _float("STRATEGY_MIN_EDGE_AFTER_FEES", 0.001))
    auto_close_min_net_profit_pct: float = field(default_factory=lambda: _float("AUTO_CLOSE_MIN_NET_PROFIT_PCT", 0.001))
    auto_min_hold_seconds: int = field(default_factory=lambda: _int("AUTO_MIN_HOLD_SECONDS", 900))
    auto_take_profit_min_hold_seconds: int = field(default_factory=lambda: _int("AUTO_TAKE_PROFIT_MIN_HOLD_SECONDS", 0))
    auto_max_hold_seconds: int = field(default_factory=lambda: _int("AUTO_MAX_HOLD_SECONDS", 14400))
    auto_position_max_loss_pct: float = field(default_factory=lambda: _float("AUTO_POSITION_MAX_LOSS_PCT", 0.10))
    auto_default_stop_loss_pct: float = field(default_factory=lambda: _float("AUTO_DEFAULT_STOP_LOSS_PCT", 0.01))
    auto_default_take_profit_pct: float = field(default_factory=lambda: _float("AUTO_DEFAULT_TAKE_PROFIT_PCT", 0.02))
    auto_fast_profit_exit_pct: float = field(default_factory=lambda: _float("AUTO_FAST_PROFIT_EXIT_PCT", 0.006))
    # Exploration creates random paper actions. Keep it off by default so collected data is cleaner.
    exploration_mode: bool = field(default_factory=lambda: _bool("EXPLORATION_MODE", False))
    exploration_rate: float = field(default_factory=lambda: _float("EXPLORATION_RATE", 0.05))
    min_paper_trade_notional: float = field(default_factory=lambda: _float("MIN_PAPER_TRADE_NOTIONAL", 50.0))

    railway_data_factory_mode: bool = field(default_factory=lambda: _bool("RAILWAY_DATA_FACTORY_MODE", True))
    data_lifecycle_interval_seconds: int = field(default_factory=lambda: _int("DATA_LIFECYCLE_INTERVAL_SECONDS", 86400))
    operational_retention_days: int = field(default_factory=lambda: _int("OPERATIONAL_RETENTION_DAYS", 2))
    raw_payload_retention_hours: int = field(default_factory=lambda: _int("RAW_PAYLOAD_RETENTION_HOURS", 6))
    live_update_retention_hours: int = field(default_factory=lambda: _int("LIVE_UPDATE_RETENTION_HOURS", 6))
    account_equity_retention_days: int = field(default_factory=lambda: _int("ACCOUNT_EQUITY_RETENTION_DAYS", 2))
    raw_news_text_retention_days: int = field(default_factory=lambda: _int("RAW_NEWS_TEXT_RETENTION_DAYS", 3))
    keep_closed_candles_days: int = field(default_factory=lambda: _int("KEEP_CLOSED_CANDLES_DAYS", 365))
    keep_training_features_days: int = field(default_factory=lambda: _int("KEEP_TRAINING_FEATURES_DAYS", 365))
    keep_experience_days: int = field(default_factory=lambda: _int("KEEP_EXPERIENCE_DAYS", 365))
    raw_news_retention_days: int = field(default_factory=lambda: _int("RAW_NEWS_RETENTION_DAYS", 30))
    raw_external_event_retention_days: int = field(default_factory=lambda: _int("RAW_EXTERNAL_EVENT_RETENTION_DAYS", 1))
    raw_tick_retention_days: int = field(default_factory=lambda: _int("RAW_TICK_RETENTION_DAYS", 1))
    diagnostic_retention_days: int = field(default_factory=lambda: _int("DIAGNOSTIC_RETENTION_DAYS", 2))
    external_data_retention_days: int = field(default_factory=lambda: _int("EXTERNAL_DATA_RETENTION_DAYS", 365))
    training_feature_retention_days: int = field(default_factory=lambda: _int("TRAINING_FEATURE_RETENTION_DAYS", _int("KEEP_TRAINING_FEATURES_DAYS", 365)))
    experience_retention_days: int = field(default_factory=lambda: _int("EXPERIENCE_RETENTION_DAYS", _int("KEEP_EXPERIENCE_DAYS", 365)))
    closed_candle_retention_days: int = field(default_factory=lambda: _int("CLOSED_CANDLE_RETENTION_DAYS", _int("KEEP_CLOSED_CANDLES_DAYS", 365)))
    archive_dir: Path = field(default_factory=lambda: Path(os.getenv("ARCHIVE_DIR", "./archives")))
    dashboard_username: str | None = field(default_factory=lambda: _secret("DASHBOARD_USERNAME"))
    dashboard_password: str | None = field(default_factory=lambda: _secret("DASHBOARD_PASSWORD"))
    admin_token: str | None = field(default_factory=lambda: _secret("ADMIN_TOKEN"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()

    @property
    def is_paper_mode(self) -> bool:
        return self.trading_mode == "paper"


settings = Settings()
