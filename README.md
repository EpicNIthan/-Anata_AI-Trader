# Anata AI Crypto Trading Lab

A real paper-only AI crypto trading lab built with Python, FastAPI, SQLAlchemy, PostgreSQL, Binance market streams, public futures trader-flow data, configurable news polling, feature building, a replaceable strategy interface, and a dashboard.

No live exchange order APIs are wired. There are no real trading keys, no withdrawals, and no real-money execution.

## What Is Included

- FastAPI app with `/health`, dashboard routes, API routes, and `/api/signal`.
- SQLAlchemy models for candles, news, sentiment, features, paper trades, positions, model versions, training runs, AI decisions, and account equity.
- Binance websocket kline collector for liquid USDT symbols such as `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`, `XRPUSDT`, `ADAUSDT`, `DOGEUSDT`, `AVAXUSDT`, `LINKUSDT`, and `LTCUSDT`.
- Binance Futures public aggregate collector for long/short ratios, top-trader ratios, taker buy/sell flow, open interest, and funding rate.
- REST news collector with a configurable provider URL and API key.
- News sentiment interface with a rule-based fallback and optional Hugging Face model if you install `requirements-hf.txt`.
- Feature builder combining candle behavior and sentiment/risk scores.
- Paper futures engine with fake balance, long/short positions, leverage, fees, PnL, trade history, and no live exchange orders.
- Risk manager for max trade size, max daily loss, max open positions, cooldown after large loss, and low-confidence rejection.
- Railway-safe dataset export plus laptop/offline training scripts for stronger local models.
- Dockerfile and Railway-compatible start command.
- Versioned JSON feature schemas, model lineage metadata, and an experience replay buffer for continued training.
- Autonomous paper-trading runner with API and dashboard controls.

## Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Optional Hugging Face sentiment dependencies are intentionally separate so Railway does not need heavy packages by default:

```powershell
python -m pip install -r requirements-hf.txt
```

Powerful local training dependencies are also separate. Install them on your laptop, not Railway:

```powershell
pip install -r requirements-local-training.txt
```

Set `DATABASE_URL` in `.env` to PostgreSQL for normal use:

```env
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/anata_ai_trader
TRADING_MODE=paper
```

For quick local smoke checks only, the app can also run with SQLite:

```env
DATABASE_URL=sqlite:///./trading_lab.db
```

Run the app:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open:

- `http://localhost:8000/health`
- `http://localhost:8000/dashboard`

## Security

`/health` is public. The dashboard and all `/api/*` lab endpoints are private when either `ADMIN_TOKEN` or `DASHBOARD_USERNAME` plus `DASHBOARD_PASSWORD` is configured.

Browser login options:

- Open `/dashboard?admin_token=YOUR_TOKEN` when using `ADMIN_TOKEN`.
- Or use browser Basic Auth with `DASHBOARD_USERNAME` and `DASHBOARD_PASSWORD`.

Script/API calls can pass:

```powershell
curl -H "x-admin-token: YOUR_TOKEN" https://your-app.up.railway.app/api/dashboard/summary
```

## Collectors

Collectors are background workers and are off by default. Start them from the dashboard or API:

```powershell
curl -X POST http://localhost:8000/api/collectors/market/start
curl -X POST http://localhost:8000/api/collectors/news/start
```

You can enable them on startup:

```env
ENABLE_MARKET_COLLECTOR=true
ENABLE_NEWS_COLLECTOR=true
```

The Binance collector writes closed candles for training and upserted live candle rows for charts. Raw market ticks are optional and off by default. The news collector writes articles and sentiment rows from free-first providers:

- RSS: crypto-specific news from configured feeds such as CoinDesk, Cointelegraph, and Decrypt.
- GDELT: global/world/macroeconomic news, no API key required.
- NewsAPI: delayed/free fallback and dev source only, disabled unless `NEWSAPI_ENABLED=true`.

Collector diagnostics:

```powershell
curl http://localhost:8000/api/market/status
curl http://localhost:8000/api/market/latest
curl -X POST http://localhost:8000/api/market/backfill -H "Content-Type: application/json" -d "{\"limit\":100}"
curl http://localhost:8000/api/news/status
curl http://localhost:8000/api/news/latest
curl http://localhost:8000/api/news/latest?provider=rss
curl http://localhost:8000/api/news/latest?provider=gdelt
curl -X POST http://localhost:8000/api/news/run-once -H "Content-Type: application/json" -d "{\"provider\":\"gdelt\"}"
```

Trader-flow / derivatives diagnostics:

```powershell
curl -X POST http://localhost:8000/api/collectors/derivatives/start
curl http://localhost:8000/api/derivatives/status
curl http://localhost:8000/api/derivatives/latest?symbol=BTCUSDT
curl -X POST http://localhost:8000/api/derivatives/run-once -H "Content-Type: application/json" -d "{\"symbols\":[\"BTCUSDT\"],\"mock\":true}"
```

The derivatives collector uses public aggregate data only. It does not know individual trader win rate. It approximates current crowd behavior through account long/short ratios, taker buy/sell pressure, open interest, and funding. Those values are stored in `external_data_events` and compacted into `training_features`.

Market status exposes `last_message_at`, `last_saved_at`, `messages_received`, `rows_saved`, `last_error`, `subscribed_streams`, `websocket_url`, and whether candles are live-updated or closed-only. Binance streams are lowercase, for example `btcusdt@kline_1m`.

Use this to see live candle changes before candle close:

```env
STORE_LIVE_CANDLE_UPDATES=true
STORE_MARKET_TICKS=false
```

`STORE_MARKET_TICKS=false` is the recommended Railway default. Closed candles and compact training features are useful long-term; raw tick rows grow very quickly.

If one provider fails or is missing a token, the other providers continue. For UI/sentiment testing without a provider:

```powershell
curl -X POST http://localhost:8000/api/news/mock -H "Content-Type: application/json" -d "{\"title\":\"Mock BTC update\",\"body\":\"Bitcoin rallies as liquidity improves\"}"
```

## Paper Signals

Send a paper-only signal:

```powershell
curl -X POST http://localhost:8000/api/signal `
  -H "Content-Type: application/json" `
  -d "{\"symbol\":\"BTCUSDT\",\"action\":\"BUY\",\"confidence\":0.8,\"price\":65000,\"notional\":250,\"reason\":\"manual test\"}"
```

Supported actions:

- `BUY`
- `SELL`
- `HOLD`
- `CLOSE`

`SELL` and `CLOSE` only close existing long paper positions. Short selling is disabled.
Paper futures semantics:

- `BUY` opens/increases a long when no short is open.
- `BUY` closes an open short.
- `SELL` opens/increases a short when no long is open.
- `SELL` closes an open long.
- `CLOSE` closes the current open long or short.

## Autonomous Paper Trader

The auto trader can run the paper loop without manual `/api/signal` calls. It builds features for each configured symbol, calls the current strategy, obeys the paper risk manager, executes through the paper engine only, records `ai_decisions`, records `experience_buffer`, and updates account equity.

Environment:

```env
AUTO_TRADER_ENABLED=false
AUTO_TRADER_USE_TRAINED_MODEL=true
AUTO_TRADER_INTERVAL_SECONDS=60
AUTO_TRADER_SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,AVAXUSDT,LINKUSDT,LTCUSDT
PAPER_TRADE_TIMEFRAME=1m
PAPER_FEE_RATE=0.0004
PAPER_MIN_LEVERAGE=1
PAPER_MAX_LEVERAGE=125
PAPER_CONFIDENCE_LEVERAGE_ENABLED=true
RISK_MAX_TRADE_SIZE_PCT=0.50
RISK_MAX_ENTRY_FEE_PCT_OF_EQUITY=0.01
STRATEGY_MIN_EDGE_AFTER_FEES=0.001
AUTO_CLOSE_MIN_NET_PROFIT_PCT=0.001
AUTO_MIN_HOLD_SECONDS=900
AUTO_TAKE_PROFIT_MIN_HOLD_SECONDS=0
AUTO_MAX_HOLD_SECONDS=14400
AUTO_POSITION_MAX_LOSS_PCT=0.10
AUTO_DEFAULT_STOP_LOSS_PCT=0.01
AUTO_DEFAULT_TAKE_PROFIT_PCT=0.02
AUTO_FAST_PROFIT_EXIT_PCT=0.006
EXPLORATION_MODE=true
EXPLORATION_RATE=0.05
MIN_PAPER_TRADE_NOTIONAL=50
DERIVATIVES_ENABLED=true
ENABLE_DERIVATIVES_COLLECTOR=false
DERIVATIVES_POLL_INTERVAL_SECONDS=300
DERIVATIVES_PERIOD=5m
DERIVATIVES_SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,AVAXUSDT,LINKUSDT,LTCUSDT
RAW_PAYLOAD_RETENTION_HOURS=24
LIVE_UPDATE_RETENTION_HOURS=12
ACCOUNT_EQUITY_RETENTION_DAYS=7
RAW_NEWS_TEXT_RETENTION_DAYS=7
KEEP_CLOSED_CANDLES_DAYS=365
KEEP_TRAINING_FEATURES_DAYS=365
KEEP_EXPERIENCE_DAYS=365
RAW_NEWS_RETENTION_DAYS=30
RAW_TICK_RETENTION_DAYS=7
DIAGNOSTIC_RETENTION_DAYS=7
EXTERNAL_DATA_RETENTION_DAYS=365
EXPERIENCE_RETENTION_DAYS=365
CLOSED_CANDLE_RETENTION_DAYS=365
```

Controls:

```powershell
curl -X POST http://localhost:8000/api/auto-trader/start
curl http://localhost:8000/api/auto-trader/status
curl -X POST http://localhost:8000/api/auto-trader/stop
```

Safety behavior:

- Paper mode only.
- No live exchange order execution.
- Uses the active uploaded model when `AUTO_TRADER_USE_TRAINED_MODEL=true`; falls back to `rule-based-v1` when no active compatible model exists.
- Uses the existing risk manager.
- Uses confidence-sized paper-only leverage. With `PAPER_CONFIDENCE_LEVERAGE_ENABLED=true`, low confidence stays near `PAPER_MIN_LEVERAGE`, while very high confidence can approach `PAPER_MAX_LEVERAGE`.
- Sizes new entries from 0% to `RISK_MAX_TRADE_SIZE_PCT` of equity as margin based on confidence. With `RISK_MAX_TRADE_SIZE_PCT=0.50`, very high-confidence paper trades can use up to 50% of equity as margin.
- Caps entry fee exposure with `RISK_MAX_ENTRY_FEE_PCT_OF_EQUITY`, because high leverage creates large notional and fees are charged on notional, not margin.
- Avoids weak entries when the expected edge is smaller than round-trip paper fees plus `STRATEGY_MIN_EDGE_AFTER_FEES`.
- Keeps weak/noisy strategy exits open for at least `AUTO_MIN_HOLD_SECONDS` so the bot does not churn in and out after one minute.
- Allows take-profit exits immediately by default with `AUTO_TAKE_PROFIT_MIN_HOLD_SECONDS=0`.
- Allows fast profit exits during the minimum hold if net profit is at least `AUTO_FAST_PROFIT_EXIT_PCT`.
- Closes immediately when a stop loss or max-position-loss rule is hit, even if that means taking a small loss now.
- Avoids closing tiny green positions when the closing fee would eat the profit, unless the bot is cutting a loss.
- Adds default paper stop loss / take profit levels to new entries when the strategy did not provide them.
- Blocks rapid duplicate same-direction entries for the same symbol.
- Blocks new auto-trader entries during cooldown after a meaningful recent realized loss.
- Optional exploration mode only runs in paper mode and uses tiny fake notionals while still passing through risk checks.
- Records whether each decision came from `strategy` or `exploration`, then updates 5m/15m/1h reward diagnostics when future candles exist.

Diagnostics:

```powershell
curl http://localhost:8000/api/db/diagnostics
curl http://localhost:8000/api/db/storage
curl -X POST http://localhost:8000/api/db/compact
curl -X POST http://localhost:8000/api/db/cleanup
curl -X POST http://localhost:8000/api/db/archive
```

Reprocess existing news sentiment after enabling Hugging Face:

```powershell
curl -X POST http://localhost:8000/api/sentiment/reprocess -H "Content-Type: application/json" -d "{\"limit\":200,\"reset_model\":true}"
```

`/api/dashboard/summary` and the dashboard sentiment status show whether HF really loaded. If `hf_loaded=false`, the app is still using the rule-based fallback. This can happen if Railway cannot download the model, does not have enough memory for `transformers`/`torch`, or the variable is not enabled. Quoted booleans like `ENABLE_HF_SENTIMENT="true"` are accepted, but Railway variables should normally be entered without quotes.

Data lifecycle rules:

- `live_candle_updates` is short-term chart data and is upserted by symbol/timeframe/open time.
- `candles` stores closed candles for long-term training.
- `training_features` stores compact numeric feature rows for model training/export.
- Raw ticks, live updates, old diagnostic rows, old raw news text, and old raw payloads are cleaned or compacted by the daily lifecycle job.
- Public external data such as derivatives flow is kept longer as compact numeric/payload rows, while old raw payloads are compacted.
- Training exports read compact features first, so raw logs can be compacted after features are built and archived.
- The dashboard `DB Storage` tab shows total DB size, largest tables, raw/JSON estimates, last cleanup, and buttons for `Compact DB` or `Archive + Compact`.
- `POST /api/db/compact` strips bulky raw/debug fields but keeps closed candles, `training_features`, `experience_buffer`, `model_versions`, and `paper_trades`.

## Training Workflow

Features are stored as versioned JSON payloads. The current schema is `price-news-v3`; `price-news-v2` and `price-news-v1` remain available for older models. Missing future values default to `0` or `null` so older models keep running when new data sources are added.

`price-news-v3` adds public trader-flow features:

- crowd long/short account percentages
- top-trader account and position long percentages
- taker buy pressure and buy/sell ratio
- open interest value and open interest change
- funding rate
- combined `trader_crowd_score`
- combined `crowd_risk_score`

Railway is inference-only by default:

```env
ENABLE_SERVER_TRAINING=false
ENABLE_SERVER_INFERENCE=true
```

Railway collects data, builds/exports datasets, runs the dashboard, runs paper trading, and runs active model inference. Heavy training should happen on your laptop after downloading a dataset. If `/api/training/train-model` is called while server training is disabled, it returns: `Server training is disabled. Download dataset and train locally.`

Before training locally, check label coverage. `target_trade_quality_score` must have labeled rows:

```powershell
curl -H "x-admin-token: YOUR_ADMIN_TOKEN" https://anataai-trader-production.up.railway.app/api/training/label-status

curl -X POST https://anataai-trader-production.up.railway.app/api/training/build-labels `
  -H "x-admin-token: YOUR_ADMIN_TOKEN" `
  -H "Content-Type: application/json" `
  -d "{}"
```

`AUTO_BUILD_LABELS_ON_EXPORT=true` makes dataset export try to build labels first. Recent rows without enough future candles are skipped until more closed candles exist.

Build features through the API or internal jobs, then export:

```powershell
python -m app.training.export_dataset
```

Fast dataset accelerator:

```powershell
python -m app.training.dataset_accelerator --symbols BTCUSDT,ETHUSDT,SOLUSDT --days 30 --max-rows-per-symbol 20000 --stride 5
```

Or from the dashboard Training tab, click `Build Training Dataset`.

What it does:

- Backfills closed historical candles.
- Builds compact `training_features` at historical candle times.
- Adds future-return labels for `5m`, `15m`, `1h`, and `4h`.
- Adds max-upside/max-drawdown labels.
- Adds stop-loss/take-profit-hit-first labels.
- Adds `target_direction_15m` and `target_trade_quality_score`.
- Creates offline replay rows in `experience_buffer` for `BUY` and `HOLD`.
- Exports a gzip CSV like `datasets/anata_dataset_YYYYMMDD_HHMMSS.csv.gz`.

API:

```powershell
curl -X POST http://localhost:8000/api/training/build-dataset `
  -H "Content-Type: application/json" `
  -d "{\"symbols\":[\"BTCUSDT\",\"ETHUSDT\"],\"days\":14,\"max_rows_per_symbol\":5000,\"stride\":5,\"backfill\":true,\"export\":true}"
```

This is much faster than waiting for live paper trades. Live paper trading remains the final test; historical replay is the fast training-data factory.

Useful export filters:

```powershell
python -m app.training.export_dataset --since-date 2026-06-24
python -m app.training.export_dataset --use-all-data
```

Railway export/download API:

```powershell
curl -X POST https://your-app.up.railway.app/api/training/export `
  -H "x-admin-token: YOUR_ADMIN_TOKEN" `
  -H "Content-Type: application/json" `
  -d "{\"use_all_data\":true}"

curl -L https://your-app.up.railway.app/api/training/download/anata_dataset_YYYYMMDD_HHMMSS.csv.gz `
  -H "x-admin-token: YOUR_ADMIN_TOKEN" `
  -o datasets/anata_dataset_YYYYMMDD_HHMMSS.csv.gz
```

Laptop workflow for your Railway app:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-local-training.txt

python scripts/download_dataset.py --url https://anataai-trader-production.up.railway.app --token YOUR_ADMIN_TOKEN

python scripts/train_local_model.py --dataset datasets/latest.csv.gz --model-type sklearn_hist_gradient_boosting --target target_trade_quality_score

python scripts/train_local_model.py --dataset datasets/latest.csv.gz --model-type lightgbm --target target_trade_quality_score

python scripts/evaluate_local_model.py --dataset datasets/latest.csv.gz --model models/YOUR_MODEL.joblib

python scripts/package_model.py --model models/YOUR_MODEL.joblib

python scripts/upload_model.py --url https://anataai-trader-production.up.railway.app --token YOUR_ADMIN_TOKEN --package model_package_VERSION.zip
```

Supported local model types:

- `sklearn_hist_gradient_boosting`
- `random_forest`
- `lightgbm` if installed locally
- `xgboost` if installed locally

If LightGBM or XGBoost fails to install/import on Windows, use `--model-type sklearn_hist_gradient_boosting`; the local scripts will print a clear message and keep the sklearn path available.

Uploaded models are registered as `candidate` first. Activate only after checking metrics:

```powershell
curl -X POST https://your-app.up.railway.app/api/models/activate `
  -H "x-admin-token: YOUR_ADMIN_TOKEN" `
  -H "Content-Type: application/json" `
  -d "{\"model_id\":\"MODEL_ID_FROM_UPLOAD\"}"

python scripts/activate_model.py --url https://anataai-trader-production.up.railway.app --token YOUR_ADMIN_TOKEN --model-id MODEL_ID
```

Model endpoints:

```powershell
curl -H "x-admin-token: YOUR_ADMIN_TOKEN" https://your-app.up.railway.app/api/models
curl -H "x-admin-token: YOUR_ADMIN_TOKEN" https://your-app.up.railway.app/api/models/latest
curl -H "x-admin-token: YOUR_ADMIN_TOKEN" https://your-app.up.railway.app/api/models/active
```

Every saved model records `model_id`, `version`, `feature_schema_version`, `feature_columns`, `created_at`, metrics, model type, and training dataset hash. Old models are kept in `model_versions`; new models can use a larger feature list while old models keep reading only the columns they were trained on.

The auto trader loads only the active model. If the active model is missing or incompatible, it falls back to the rule-based strategy and reports the fallback reason in status/diagnostics.

## Replay And Future Data

Every signal-like decision writes an `ai_decisions` row and an `experience_buffer` row with:

- market state
- news state
- feature payload when available
- action and confidence
- execution result
- reward

Inspect replay data:

```powershell
curl http://localhost:8000/api/experiences
```

Future data sources can be added through `external_data_events` without changing the core candle/news tables. Examples include funding rate, open interest, whale alerts, liquidation data, fear/greed index, social sentiment, and on-chain data.

```powershell
curl -X POST http://localhost:8000/api/data-events `
  -H "Content-Type: application/json" `
  -d "{\"source_name\":\"fear_greed\",\"data_type\":\"fear_greed_index\",\"numeric_value\":72,\"payload\":{\"label\":\"greed\"}}"
```

## Railway

Railway can build from the Dockerfile. The start command is:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Set environment variables in Railway:

- `DATABASE_URL`
- `TRADING_MODE=paper`
- `ADMIN_TOKEN=long_random_secret`
- `ENABLE_SERVER_TRAINING=false`
- `ENABLE_SERVER_INFERENCE=true`
- `BINANCE_SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,AVAXUSDT,LINKUSDT,LTCUSDT`
- `AUTO_TRADER_SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,AVAXUSDT,LINKUSDT,LTCUSDT`
- `AUTO_TRADER_USE_TRAINED_MODEL=true`
- `DERIVATIVES_ENABLED=true`
- `ENABLE_DERIVATIVES_COLLECTOR=true` if you want trader-flow data to run automatically
- `PAPER_FEE_RATE=0.0004`
- `PAPER_MIN_LEVERAGE=1`
- `PAPER_MAX_LEVERAGE=125`
- `PAPER_CONFIDENCE_LEVERAGE_ENABLED=true`
- `RISK_MAX_TRADE_SIZE_PCT=0.50`
- `RISK_MAX_ENTRY_FEE_PCT_OF_EQUITY=0.01`
- `STORE_MARKET_TICKS=false`
- `STRATEGY_MIN_EDGE_AFTER_FEES=0.001`
- `AUTO_MIN_HOLD_SECONDS=900`
- `AUTO_TAKE_PROFIT_MIN_HOLD_SECONDS=0`
- `AUTO_FAST_PROFIT_EXIT_PCT=0.006`
- `AUTO_POSITION_MAX_LOSS_PCT=0.10`
- `AUTO_DEFAULT_STOP_LOSS_PCT=0.01`
- `NEWS_PROVIDER=rss,gdelt,newsapi`
- `RSS_NEWS_ENABLED=true`
- `GDELT_ENABLED=true`
- `NEWSAPI_ENABLED=false`
- `NEWS_API_KEY` only if using NewsAPI fallback
- risk settings such as `RISK_MAX_TRADE_SIZE_PCT`

## Safety Boundaries

- No real exchange order calls.
- No exchange API key fields.
- No withdrawals.
- Paper mode is the only supported execution mode.
- Risk checks run before paper entries.
- Architecture leaves room for Binance/OKX testnet adapters later, behind a separate execution interface.

## Smoke Test

```powershell
python tests/smoke_test.py
```

The smoke test verifies `/health`, dashboard rendering, private API protection, paper signal execution, dataset export/download, candidate model upload, explicit activation, active model inference/fallback safety, collector status access, and database table creation.
