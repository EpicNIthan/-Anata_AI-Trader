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


def _optional_float(name: str) -> float | None:
    value = os.getenv(name)
    return None if value in (None, "") else float(value)


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
    enable_spot_context_collector: bool = field(default_factory=lambda: _bool("ENABLE_SPOT_CONTEXT_COLLECTOR", True))
    spot_context_poll_interval_seconds: int = field(default_factory=lambda: _int("SPOT_CONTEXT_POLL_INTERVAL_SECONDS", 300))
    spot_context_symbols: list[str] = field(default_factory=lambda: _csv(os.getenv("SPOT_CONTEXT_SYMBOLS"), DEFAULT_SYMBOLS))
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
    paper_data_collection_mode: bool = field(default_factory=lambda: _bool("PAPER_DATA_COLLECTION_MODE", False))
    paper_data_collection_exploration_rate: float = field(default_factory=lambda: _float("PAPER_DATA_COLLECTION_EXPLORATION_RATE", 0.35))
    paper_data_collection_reset_enabled: bool = field(default_factory=lambda: _bool("PAPER_DATA_COLLECTION_RESET_ENABLED", True))
    paper_data_collection_reset_equity_pct: float = field(default_factory=lambda: _float("PAPER_DATA_COLLECTION_RESET_EQUITY_PCT", 0.10))
    paper_data_collection_min_hold_seconds: int = field(default_factory=lambda: _int("PAPER_DATA_COLLECTION_MIN_HOLD_SECONDS", 60))
    paper_data_collection_close_rate: float = field(default_factory=lambda: _float("PAPER_DATA_COLLECTION_CLOSE_RATE", 0.35))
    paper_data_collection_confidence: float = field(default_factory=lambda: _float("PAPER_DATA_COLLECTION_CONFIDENCE", 0.70))
    min_paper_trade_notional: float = field(default_factory=lambda: _float("MIN_PAPER_TRADE_NOTIONAL", 50.0))

    # V2 pipeline is paper-only and is enabled by default.  The older direct strategy
    # path remains as a compatibility adapter, never as a model-to-order bypass.
    anata_v2_enabled: bool = field(default_factory=lambda: _bool("ANATA_V2_ENABLED", True))
    v2_use_narrow_models: bool = field(default_factory=lambda: _bool("V2_USE_NARROW_MODELS", True))
    v2_require_registered_champion: bool = field(default_factory=lambda: _bool("V2_REQUIRE_REGISTERED_CHAMPION", False))
    v2_auto_promote_champion: bool = field(default_factory=lambda: _bool("V2_AUTO_PROMOTE_CHAMPION", False))
    v2_champion_account_id: str = os.getenv("V2_CHAMPION_ACCOUNT_ID", "champion")
    v2_default_forecast_horizon_seconds: int = field(default_factory=lambda: _int("V2_DEFAULT_FORECAST_HORIZON_SECONDS", 300))
    v2_signal_ttl_seconds: int = field(default_factory=lambda: _int("V2_SIGNAL_TTL_SECONDS", 300))
    v2_min_net_edge: float = field(default_factory=lambda: _float("V2_MIN_NET_EDGE", 0.0005))
    v2_max_position_leverage: float = field(default_factory=lambda: _float("V2_MAX_POSITION_LEVERAGE", 3.0))
    v2_max_symbol_exposure_pct: float = field(default_factory=lambda: _float("V2_MAX_SYMBOL_EXPOSURE_PCT", 0.10))
    v2_max_gross_exposure_pct: float = field(default_factory=lambda: _float("V2_MAX_GROSS_EXPOSURE_PCT", 0.40))
    v2_max_net_exposure_pct: float = field(default_factory=lambda: _float("V2_MAX_NET_EXPOSURE_PCT", 0.25))
    v2_max_cluster_exposure_pct: float = field(default_factory=lambda: _float("V2_MAX_CLUSTER_EXPOSURE_PCT", 0.25))
    v2_min_liquidity_score: float = field(default_factory=lambda: _float("V2_MIN_LIQUIDITY_SCORE", 0.20))
    v2_max_expected_cost_pct: float = field(default_factory=lambda: _float("V2_MAX_EXPECTED_COST_PCT", 0.003))
    v2_external_context_max_adjustment: float = field(default_factory=lambda: _float("V2_EXTERNAL_CONTEXT_MAX_ADJUSTMENT", 0.10))
    v2_correlation_penalty_threshold: float = field(default_factory=lambda: _float("V2_CORRELATION_PENALTY_THRESHOLD", 0.70))
    v2_sandbox_max_exposure_pct: float = field(default_factory=lambda: _float("V2_SANDBOX_MAX_EXPOSURE_PCT", 0.03))
    v2_model_registry_dir: Path = field(default_factory=lambda: Path(os.getenv("V2_MODEL_REGISTRY_DIR", "./model_registry")))

    # Every exposure increase, including legacy model hints, exploration and sandbox
    # activity, is subject to this independent policy.
    risk_kill_switch_enabled: bool = field(default_factory=lambda: _bool("RISK_KILL_SWITCH_ENABLED", False))
    risk_configuration_version: str = os.getenv("RISK_CONFIGURATION_VERSION", "v2-safe-defaults")
    risk_max_market_data_age_seconds: int = field(default_factory=lambda: _int("RISK_MAX_MARKET_DATA_AGE_SECONDS", 180))
    risk_max_portfolio_drawdown_pct: float = field(default_factory=lambda: _float("RISK_MAX_PORTFOLIO_DRAWDOWN_PCT", 0.15))
    risk_max_portfolio_leverage: float = field(default_factory=lambda: _float("RISK_MAX_PORTFOLIO_LEVERAGE", 3.0))
    risk_max_spread_pct: float = field(default_factory=lambda: _float("RISK_MAX_SPREAD_PCT", 0.005))
    risk_max_expected_transaction_cost_pct: float = field(default_factory=lambda: _float("RISK_MAX_EXPECTED_TRANSACTION_COST_PCT", 0.003))
    risk_max_fee_exposure_pct: float = field(default_factory=lambda: _float("RISK_MAX_FEE_EXPOSURE_PCT", 0.01))
    risk_require_fresh_data: bool = field(default_factory=lambda: _bool("RISK_REQUIRE_FRESH_DATA", True))

    # Deterministic paper-execution assumptions.  These never call an exchange.
    paper_simulated_spread_pct: float = field(default_factory=lambda: _float("PAPER_SIMULATED_SPREAD_PCT", 0.0002))
    paper_simulated_slippage_pct: float = field(default_factory=lambda: _float("PAPER_SIMULATED_SLIPPAGE_PCT", 0.0001))
    paper_simulated_latency_ms: int = field(default_factory=lambda: _int("PAPER_SIMULATED_LATENCY_MS", 0))
    paper_simulated_volume_participation: float = field(default_factory=lambda: _float("PAPER_SIMULATED_VOLUME_PARTICIPATION", 0.10))
    paper_simulated_partial_fill_enabled: bool = field(default_factory=lambda: _bool("PAPER_SIMULATED_PARTIAL_FILL_ENABLED", False))
    paper_simulated_funding_rate: float = field(default_factory=lambda: _float("PAPER_SIMULATED_FUNDING_RATE", 0.0))
    paper_simulated_market_impact_coefficient: float = field(
        default_factory=lambda: _float("PAPER_SIMULATED_MARKET_IMPACT_COEFFICIENT", 0.0002)
    )
    paper_simulated_order_ttl_seconds: int = field(default_factory=lambda: _int("PAPER_SIMULATED_ORDER_TTL_SECONDS", 300))

    # Local intelligence is always sufficient to run the base ensemble.  External
    # providers are optional, quota/budget constrained context overlays.
    external_ai_enabled: bool = field(default_factory=lambda: _bool("EXTERNAL_AI_ENABLED", False))
    external_ai_provider_order: list[str] = field(default_factory=lambda: _csv_raw(os.getenv("EXTERNAL_AI_PROVIDER_ORDER"), "gemini,groq,huggingface,generic"))
    external_ai_daily_request_limit: int = field(default_factory=lambda: _int("EXTERNAL_AI_DAILY_REQUEST_LIMIT", 20))
    external_ai_monthly_budget_usd: float = field(default_factory=lambda: _float("EXTERNAL_AI_MONTHLY_BUDGET_USD", 0.0))
    external_ai_timeout_seconds: int = field(default_factory=lambda: _int("EXTERNAL_AI_TIMEOUT_SECONDS", 15))
    external_ai_max_retries: int = field(default_factory=lambda: _int("EXTERNAL_AI_MAX_RETRIES", 2))
    external_ai_importance_threshold: float = field(default_factory=lambda: _float("EXTERNAL_AI_IMPORTANCE_THRESHOLD", 0.70))
    external_ai_local_uncertainty_threshold: float = field(default_factory=lambda: _float("EXTERNAL_AI_LOCAL_UNCERTAINTY_THRESHOLD", 0.45))
    external_ai_prompt_version: str = os.getenv("EXTERNAL_AI_PROMPT_VERSION", "structured-news-v1")
    external_ai_circuit_breaker_failures: int = field(default_factory=lambda: _int("EXTERNAL_AI_CIRCUIT_BREAKER_FAILURES", 3))
    external_ai_circuit_breaker_seconds: int = field(default_factory=lambda: _int("EXTERNAL_AI_CIRCUIT_BREAKER_SECONDS", 300))
    external_ai_cache_ttl_seconds: int = field(default_factory=lambda: _int("EXTERNAL_AI_CACHE_TTL_SECONDS", 86400))
    generic_ai_base_url: str | None = field(default_factory=lambda: os.getenv("GENERIC_AI_BASE_URL") or None)
    generic_ai_api_key: str | None = field(default_factory=lambda: _secret("GENERIC_AI_API_KEY"))
    external_ai_generic_model: str = os.getenv("EXTERNAL_AI_GENERIC_MODEL", "structured-news-context")
    external_ai_input_cost_per_million_usd: float | None = field(
        default_factory=lambda: _optional_float("EXTERNAL_AI_INPUT_COST_PER_MILLION_USD")
    )
    external_ai_output_cost_per_million_usd: float | None = field(
        default_factory=lambda: _optional_float("EXTERNAL_AI_OUTPUT_COST_PER_MILLION_USD")
    )
    gemini_api_key: str | None = field(default_factory=lambda: _secret("GEMINI_API_KEY"))
    gemini_base_url: str = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
    gemini_model: str = os.getenv("GEMINI_MODEL", "")
    gemini_input_cost_per_million_usd: float | None = field(default_factory=lambda: _optional_float("GEMINI_INPUT_COST_PER_MILLION_USD"))
    gemini_output_cost_per_million_usd: float | None = field(default_factory=lambda: _optional_float("GEMINI_OUTPUT_COST_PER_MILLION_USD"))
    groq_api_key: str | None = field(default_factory=lambda: _secret("GROQ_API_KEY"))
    groq_base_url: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    groq_model: str = os.getenv("GROQ_MODEL", "")
    groq_input_cost_per_million_usd: float | None = field(default_factory=lambda: _optional_float("GROQ_INPUT_COST_PER_MILLION_USD"))
    groq_output_cost_per_million_usd: float | None = field(default_factory=lambda: _optional_float("GROQ_OUTPUT_COST_PER_MILLION_USD"))
    huggingface_inference_token: str | None = field(default_factory=lambda: _secret("HUGGINGFACE_INFERENCE_TOKEN"))
    huggingface_inference_base_url: str = os.getenv("HUGGINGFACE_INFERENCE_BASE_URL", "https://router.huggingface.co/v1")
    huggingface_inference_model: str = os.getenv("HUGGINGFACE_INFERENCE_MODEL", "")
    huggingface_input_cost_per_million_usd: float | None = field(default_factory=lambda: _optional_float("HUGGINGFACE_INPUT_COST_PER_MILLION_USD"))
    huggingface_output_cost_per_million_usd: float | None = field(default_factory=lambda: _optional_float("HUGGINGFACE_OUTPUT_COST_PER_MILLION_USD"))
    local_news_student_path: Path | None = field(
        default_factory=lambda: Path(os.getenv("LOCAL_NEWS_STUDENT_PATH")) if os.getenv("LOCAL_NEWS_STUDENT_PATH") else None
    )
    local_news_student_version: str = os.getenv("LOCAL_NEWS_STUDENT_VERSION", "rule-v1")
    enrichment_enabled: bool = field(default_factory=lambda: _bool("ENRICHMENT_ENABLED", True))
    enrichment_interval_seconds: int = field(default_factory=lambda: _int("ENRICHMENT_INTERVAL_SECONDS", 300))
    enrichment_batch_size: int = field(default_factory=lambda: _int("ENRICHMENT_BATCH_SIZE", 25))

    vision_refresh_seconds: int = field(default_factory=lambda: _int("VISION_REFRESH_SECONDS", 15))
    vision_default_limit: int = field(default_factory=lambda: _int("VISION_DEFAULT_LIMIT", 250))
    monitoring_enabled: bool = field(default_factory=lambda: _bool("MONITORING_ENABLED", True))
    monitoring_outcome_batch_size: int = field(default_factory=lambda: _int("MONITORING_OUTCOME_BATCH_SIZE", 250))
    monitoring_health_window: int = field(default_factory=lambda: _int("MONITORING_HEALTH_WINDOW", 100))
    health_min_observations: int = field(default_factory=lambda: _int("HEALTH_MIN_OBSERVATIONS", 20))
    health_watch_calibration_error: float = field(default_factory=lambda: _float("HEALTH_WATCH_CALIBRATION_ERROR", 0.25))
    health_degraded_calibration_error: float = field(default_factory=lambda: _float("HEALTH_DEGRADED_CALIBRATION_ERROR", 0.40))
    health_watch_missing_feature_rate: float = field(default_factory=lambda: _float("HEALTH_WATCH_MISSING_FEATURE_RATE", 0.10))
    health_degraded_missing_feature_rate: float = field(default_factory=lambda: _float("HEALTH_DEGRADED_MISSING_FEATURE_RATE", 0.25))
    health_watch_prediction_drift: float = field(default_factory=lambda: _float("HEALTH_WATCH_PREDICTION_DRIFT", 2.0))
    health_degraded_prediction_drift: float = field(default_factory=lambda: _float("HEALTH_DEGRADED_PREDICTION_DRIFT", 4.0))
    health_watch_feature_drift: float = field(default_factory=lambda: _float("HEALTH_WATCH_FEATURE_DRIFT", 2.0))
    health_degraded_feature_drift: float = field(default_factory=lambda: _float("HEALTH_DEGRADED_FEATURE_DRIFT", 4.0))
    health_watch_ood_rate: float = field(default_factory=lambda: _float("HEALTH_WATCH_OOD_RATE", 0.10))
    health_degraded_ood_rate: float = field(default_factory=lambda: _float("HEALTH_DEGRADED_OOD_RATE", 0.25))
    health_watch_correlation_increase: float = field(default_factory=lambda: _float("HEALTH_WATCH_CORRELATION_INCREASE", 0.20))
    health_degraded_correlation_increase: float = field(default_factory=lambda: _float("HEALTH_DEGRADED_CORRELATION_INCREASE", 0.40))
    health_watch_transaction_cost_increase: float = field(default_factory=lambda: _float("HEALTH_WATCH_TRANSACTION_COST_INCREASE", 0.50))
    health_degraded_transaction_cost_increase: float = field(default_factory=lambda: _float("HEALTH_DEGRADED_TRANSACTION_COST_INCREASE", 1.00))
    health_watch_capacity_decline: float = field(default_factory=lambda: _float("HEALTH_WATCH_CAPACITY_DECLINE", 0.20))
    health_degraded_capacity_decline: float = field(default_factory=lambda: _float("HEALTH_DEGRADED_CAPACITY_DECLINE", 0.40))
    health_suspend_consecutive_errors: int = field(default_factory=lambda: _int("HEALTH_SUSPEND_CONSECUTIVE_ERRORS", 5))
    research_enabled: bool = field(default_factory=lambda: _bool("RESEARCH_ENABLED", False))
    research_auto_promote: bool = field(default_factory=lambda: _bool("RESEARCH_AUTO_PROMOTE", False))
    research_data_lake_dir: Path = field(default_factory=lambda: Path(os.getenv("RESEARCH_DATA_LAKE_DIR", "./local_data")))
    research_report_dir: Path = field(default_factory=lambda: Path(os.getenv("RESEARCH_REPORT_DIR", "./research_reports")))
    research_scheduler_interval_seconds: int = field(default_factory=lambda: _int("RESEARCH_SCHEDULER_INTERVAL_SECONDS", 3600))
    worker_role: str = os.getenv("WORKER_ROLE", "all").strip().lower()

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
