# Minimal Railway Variables

Most Anata AI Trader settings now live as defaults in `app/config.py`. Do not paste every default into Railway. Only keep variables that are secret or deployment-specific.

## Keep in Railway

```text
DATABASE_URL=
ADMIN_TOKEN=
```

Optional:

```text
DASHBOARD_USERNAME=
DASHBOARD_PASSWORD=
HF_API_TOKEN=
```

## Recommended V2 safety pins

The application has safe code defaults, but pin these deployment invariants so a
future default change cannot enable live-like behavior or automatic promotion:

```text
TRADING_MODE=paper
WORKER_ROLE=all
ANATA_V2_ENABLED=true
AUTO_TRADER_ENABLED=false
ENABLE_SERVER_TRAINING=false
V2_AUTO_PROMOTE_CHAMPION=false
RESEARCH_ENABLED=false
RESEARCH_AUTO_PROMOTE=false
EXTERNAL_AI_ENABLED=false
RISK_CONFIGURATION_VERSION=v2-safe-defaults
V2_MAX_POSITION_LEVERAGE=3
RISK_MAX_PORTFOLIO_LEVERAGE=3
V2_MAX_SYMBOL_EXPOSURE_PCT=0.10
V2_MAX_GROSS_EXPOSURE_PCT=0.40
V2_MAX_NET_EXPOSURE_PCT=0.25
V2_SANDBOX_MAX_EXPOSURE_PCT=0.03
PAPER_SIMULATED_SPREAD_PCT=0.0002
PAPER_SIMULATED_SLIPPAGE_PCT=0.0001
PAPER_SIMULATED_MARKET_IMPACT_COEFFICIENT=0
PAPER_SIMULATED_VOLUME_PARTICIPATION=0.10
PAPER_SIMULATED_PARTIAL_FILL_ENABLED=false
PAPER_SIMULATED_FUNDING_RATE=0
PAPER_SIMULATED_LATENCY_MS=0
PAPER_SIMULATED_ORDER_TTL_SECONDS=300
STORE_MARKET_TICKS=false
```

Enable `AUTO_TRADER_ENABLED=true` only after current candles, risk settings, the kill
switch, and the V2 trace have been checked. All execution remains simulated.

`WORKER_ROLE` accepts `all`, `web`, `collector`, `paper-trader`, or `enrichment`.
`all` is the simplest single-service deployment. For separate Railway services sharing
one PostgreSQL database, use `web` for the public service and assign the other roles to
background services. There is no Railway research role; heavy research stays local.

## Optional legacy paper data collection mode

Use this only when intentionally exercising the legacy compatibility path for
paper/demo training-data collection. The default V2 narrow pipeline does not use the
legacy exploration branch. It does not enable real-money trading and must be paired
with `TRADING_MODE=paper`.

```text
TRADING_MODE=paper
PAPER_DATA_COLLECTION_MODE=true
PAPER_DATA_COLLECTION_EXPLORATION_RATE=0.35
PAPER_DATA_COLLECTION_RESET_ENABLED=true
PAPER_DATA_COLLECTION_RESET_EQUITY_PCT=0.10
PAPER_DATA_COLLECTION_MIN_HOLD_SECONDS=60
PAPER_DATA_COLLECTION_CLOSE_RATE=0.35
PAPER_DATA_COLLECTION_CONFIDENCE=0.70
AUTO_TRADER_ENABLED=true
AUTO_TRADER_USE_TRAINED_MODEL=false
AUTO_TRADER_INTERVAL_SECONDS=60
MIN_PAPER_TRADE_NOTIONAL=50
```

The same values are available in `presets/paper_data_collection.env`.

Railway normally injects `PORT` automatically. You only need `PORT=8000` for local testing or a custom deployment.

## Optional local student and external provider

A compact local student needs only its artifact path; it does not need torch:

```text
WORKER_ROLE=enrichment
ENRICHMENT_ENABLED=true
LOCAL_NEWS_STUDENT_PATH=./models/news_student.json
LOCAL_NEWS_STUDENT_VERSION=student-version
EXTERNAL_AI_ENABLED=false
```

External AI is disabled by default. If enabled deliberately, configure a provider key,
model, declared input/output prices, daily limit, and monthly budget. The runtime
supports OpenAI-compatible Gemini, Groq, Hugging Face router, and generic endpoints.
Unknown prices fail the pre-request budget gate. Never add provider secrets to source
files or model metadata.

## Remove from Railway if you did not intentionally override them

These are already handled by code defaults:

```text
TRADING_MODE
BINANCE_SYMBOLS
BINANCE_INTERVAL
STORE_LIVE_CANDLE_UPDATES
PAPER_TRADE_TIMEFRAME
NEWS_API_KEY
ENABLE_MARKET_COLLECTOR
ENABLE_NEWS_COLLECTOR
AUTO_TRADER_ENABLED
AUTO_TRADER_INTERVAL_SECONDS
AUTO_TRADER_SYMBOLS
PAPER_START_BALANCE
MODEL_DIR
BINANCE_REST_BASE_URL
BINANCE_WS_BASE_URL
NEWS_PROVIDER
RSS_NEWS_ENABLED
GDELT_ENABLED
NEWSAPI_ENABLED
NEWS_POLL_INTERVAL_SECONDS
RSS_FEEDS
GDELT_POLL_INTERVAL_SECONDS
GDELT_MAX_RECORDS
RSS_REQUEST_USER_AGENT
NEWS_SENTIMENT_MODEL
ENABLE_HF_SENTIMENT
EXPLORATION_MODE
EXPLORATION_RATE
PAPER_DATA_COLLECTION_MODE
PAPER_DATA_COLLECTION_EXPLORATION_RATE
PAPER_DATA_COLLECTION_RESET_ENABLED
PAPER_DATA_COLLECTION_RESET_EQUITY_PCT
PAPER_DATA_COLLECTION_MIN_HOLD_SECONDS
PAPER_DATA_COLLECTION_CLOSE_RATE
PAPER_DATA_COLLECTION_CONFIDENCE
MIN_PAPER_TRADE_NOTIONAL
DERIVATIVES_ENABLED
ENABLE_DERIVATIVES_COLLECTOR
DERIVATIVES_POLL_INTERVAL_SECONDS
DERIVATIVES_PERIOD
DERIVATIVES_SYMBOLS
PAPER_LEVERAGE
PAPER_MAX_LEVERAGE
RISK_MAX_TRADE_SIZE_PCT
STRATEGY_MIN_EDGE_AFTER_FEES
AUTO_CLOSE_MIN_NET_PROFIT_PCT
AUTO_MIN_HOLD_SECONDS
AUTO_TAKE_PROFIT_MIN_HOLD_SECONDS
AUTO_MAX_HOLD_SECONDS
AUTO_POSITION_MAX_LOSS_PCT
AUTO_DEFAULT_STOP_LOSS_PCT
AUTO_DEFAULT_TAKE_PROFIT_PCT
AUTO_FAST_PROFIT_EXIT_PCT
AUTO_TRADER_USE_TRAINED_MODEL
PAPER_FEE_RATE
PAPER_MIN_LEVERAGE
PAPER_CONFIDENCE_LEVERAGE_ENABLED
RISK_MAX_ENTRY_FEE_PCT_OF_EQUITY
ENABLE_SERVER_TRAINING
STORE_MARKET_TICKS
ENABLE_SERVER_INFERENCE
ENABLE_FEAR_GREED_COLLECTOR
ENABLE_GLOBAL_MARKET_COLLECTOR
ENABLE_STABLECOIN_RISK_COLLECTOR
ENABLE_MACRO_RISK_COLLECTOR
STORE_RAW_EXTERNAL_EVENTS
STORE_RAW_LIQUIDATIONS
AUTO_BUILD_LABELS_ON_EXPORT
HF_SENTIMENT_BACKEND
RAILWAY_DATA_FACTORY_MODE
DATA_LIFECYCLE_INTERVAL_SECONDS
OPERATIONAL_RETENTION_DAYS
RAW_PAYLOAD_RETENTION_HOURS
LIVE_UPDATE_RETENTION_HOURS
ACCOUNT_EQUITY_RETENTION_DAYS
RAW_NEWS_TEXT_RETENTION_DAYS
RAW_EXTERNAL_EVENT_RETENTION_DAYS
RAW_TICK_RETENTION_DAYS
DIAGNOSTIC_RETENTION_DAYS
```

## Important defaults now in code

- Market collector: enabled by default when the role is `all` or `collector`.
- News collector: enabled by default when the role is `all` or `collector`.
- Derivatives collector: enabled by default for a collector role, with cooldown when Binance Futures returns HTTP 451 from Railway.
- Fear/Greed, global market, stablecoin risk, and macro risk collectors: enabled by default.
- Paper auto trader: enabled by the code default only for `all` or `paper-trader`; pin
  `AUTO_TRADER_ENABLED=false` during initial deployment.
- With `ANATA_V2_ENABLED=true`, the runner uses the mandatory V2 pipeline. The legacy
  Bot/Trained-AI selector is not a model-to-order bypass.
- Exploration mode is intentionally off by default because it creates random/noisy paper trades.
- Paper data collection mode is intentionally off by default. Enable it only with `TRADING_MODE=paper`.
- Server training is disabled by default; train on your PC.
- Raw ticks are disabled by default because they grow too fast.

The complete local reference is `.env.example`. Do not paste every non-secret default
into Railway; variables shown in the V2 safety block are intentional operational pins.
