# Anata AI Crypto Trading Lab

A real paper-only AI crypto trading lab built with Python, FastAPI, SQLAlchemy, PostgreSQL, Binance market streams, public futures trader-flow data, configurable news polling, feature building, a replaceable strategy interface, and a dashboard.

No live exchange order APIs are wired. There are no real trading keys, no withdrawals, and no real-money execution.

## What Is Included

- FastAPI app with `/health`, dashboard routes, API routes, and `/api/signal`.
- SQLAlchemy models for candles, news, sentiment, features, paper trades, positions, model versions, training runs, AI decisions, and account equity.
- Binance websocket kline collector for liquid USDT symbols such as `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`, `XRPUSDT`, `ADAUSDT`, `DOGEUSDT`, `AVAXUSDT`, `LINKUSDT`, and `LTCUSDT`.
- Binance Futures public aggregate collector for long/short ratios, top-trader ratios, taker buy/sell flow, open interest, and funding rate.
- REST news collector with a configurable provider URL and API key.
- Placeholder news sentiment interface shaped for a future Hugging Face model.
- Feature builder combining candle behavior and sentiment/risk scores.
- Paper engine with fake balance, fees, PnL, trade history, and long-only positions.
- Risk manager for max trade size, max daily loss, max open positions, cooldown after large loss, and low-confidence rejection.
- Training scripts for dataset export, a first-pass linear model, and evaluation.
- Dockerfile and Railway-compatible start command.
- Versioned JSON feature schemas, model lineage metadata, and an experience replay buffer for continued training.
- Autonomous paper-trading runner with API and dashboard controls.

## Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
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

The Binance collector writes candles and market ticks. The news collector writes articles and sentiment rows from free-first providers:

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
```

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

## Autonomous Paper Trader

The auto trader can run the paper loop without manual `/api/signal` calls. It builds features for each configured symbol, calls the current strategy, obeys the paper risk manager, executes through the paper engine only, records `ai_decisions`, records `experience_buffer`, and updates account equity.

Environment:

```env
AUTO_TRADER_ENABLED=false
AUTO_TRADER_INTERVAL_SECONDS=60
AUTO_TRADER_SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,AVAXUSDT,LINKUSDT,LTCUSDT
PAPER_TRADE_TIMEFRAME=1m
PAPER_LEVERAGE=10
PAPER_MAX_LEVERAGE=20
RISK_MAX_TRADE_SIZE_PCT=0.50
STRATEGY_MIN_EDGE_AFTER_FEES=0.001
AUTO_CLOSE_MIN_NET_PROFIT_PCT=0.001
AUTO_MIN_HOLD_SECONDS=900
AUTO_TAKE_PROFIT_MIN_HOLD_SECONDS=900
AUTO_MAX_HOLD_SECONDS=14400
AUTO_POSITION_MAX_LOSS_PCT=0.10
AUTO_DEFAULT_STOP_LOSS_PCT=0.01
AUTO_DEFAULT_TAKE_PROFIT_PCT=0.02
EXPLORATION_MODE=true
EXPLORATION_RATE=0.05
MIN_PAPER_TRADE_NOTIONAL=50
DERIVATIVES_ENABLED=true
ENABLE_DERIVATIVES_COLLECTOR=false
DERIVATIVES_POLL_INTERVAL_SECONDS=300
DERIVATIVES_PERIOD=5m
DERIVATIVES_SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,AVAXUSDT,LINKUSDT,LTCUSDT
LIVE_UPDATE_RETENTION_HOURS=48
RAW_NEWS_RETENTION_DAYS=30
RAW_TICK_RETENTION_DAYS=7
DIAGNOSTIC_RETENTION_DAYS=7
EXTERNAL_DATA_RETENTION_DAYS=365
EXPERIENCE_RETENTION_DAYS=365
CLOSED_CANDLE_RETENTION_DAYS=1095
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
- Uses the existing risk manager.
- Uses explicit paper-only leverage. `PAPER_LEVERAGE=10` means a $500 margin allocation creates up to $5,000 fake notional exposure. Fees are still charged on notional.
- Sizes new entries from 0% to `RISK_MAX_TRADE_SIZE_PCT` of equity as margin based on confidence. With `RISK_MAX_TRADE_SIZE_PCT=0.50`, very high-confidence paper trades can use up to 50% of equity as margin.
- Avoids weak entries when the expected edge is smaller than round-trip paper fees plus `STRATEGY_MIN_EDGE_AFTER_FEES`.
- Keeps normal/profit exits open for at least `AUTO_MIN_HOLD_SECONDS` so the bot does not churn in and out after one minute.
- Closes immediately when a stop loss or max-position-loss rule is hit, even if that means taking a small loss now.
- Avoids closing tiny green positions when the closing fee would eat the profit, unless the bot is cutting a loss.
- Adds default paper stop loss / take profit levels to new entries when the strategy did not provide them.
- Blocks rapid duplicate long entries for the same symbol.
- Blocks new auto-trader buys during cooldown after a recent realized loss.
- Optional exploration mode only runs in paper mode and uses tiny fake notionals while still passing through risk checks.
- Records whether each decision came from `strategy` or `exploration`, then updates 5m/15m/1h reward diagnostics when future candles exist.

Diagnostics:

```powershell
curl http://localhost:8000/api/db/diagnostics
curl -X POST http://localhost:8000/api/db/cleanup
curl -X POST http://localhost:8000/api/db/archive
```

Reprocess existing news sentiment after enabling Hugging Face:

```powershell
curl -X POST http://localhost:8000/api/sentiment/reprocess -H "Content-Type: application/json" -d "{\"limit\":200,\"reset_model\":true}"
```

`/api/dashboard/summary` and the dashboard sentiment status show whether HF really loaded. If `hf_loaded=false`, the app is still using the rule-based fallback. This can happen if Railway cannot download the model or does not have enough memory for `transformers`/`torch`.

Data lifecycle rules:

- `live_candle_updates` is short-term chart data and is upserted by symbol/timeframe/open time.
- `candles` stores closed candles for long-term training.
- `training_features` stores compact numeric feature rows for model training/export.
- Raw ticks, live updates, old diagnostic rows, and raw news payloads are cleaned or compacted by the daily lifecycle job.
- Public external data such as derivatives flow is kept longer as compact numeric/payload rows, while old raw payloads are compacted.
- Training exports read compact features first, so raw logs can be compacted after features are built and archived.

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

Build features through the API or internal jobs, then export:

```powershell
python -m app.training.export_dataset
```

Useful export filters:

```powershell
python -m app.training.export_dataset --since-date 2026-06-24
python -m app.training.export_dataset --use-all-data
```

Train the first model:

```powershell
python -m app.training.train_price_model --dataset datasets/features_YYYYMMDD_HHMMSS.csv
```

Continue from an existing checkpoint without starting from zero:

```powershell
python -m app.training.train_price_model --from-checkpoint models/price_linear_YYYYMMDD_HHMMSS.json --since-date 2026-06-24
```

Evaluate:

```powershell
python -m app.training.evaluate_model --dataset datasets/features_YYYYMMDD_HHMMSS.csv
python -m app.training.evaluate_model --from-checkpoint models/price_linear_YYYYMMDD_HHMMSS.json --use-all-data
```

Every saved model records `model_id`, `version`, `feature_schema_version`, `feature_columns`, `created_at`, metrics, and optional parent checkpoint metadata. Old models are kept in `model_versions`; new models can use a larger feature list while old models keep reading only the columns they were trained on.

The current model is a simple linear baseline saved under `MODEL_DIR`. The strategy interface is intentionally stable so PPO, SAC, transformer, or ensemble models can replace it later without rewriting execution.

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
- `BINANCE_SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,AVAXUSDT,LINKUSDT,LTCUSDT`
- `AUTO_TRADER_SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,AVAXUSDT,LINKUSDT,LTCUSDT`
- `DERIVATIVES_ENABLED=true`
- `ENABLE_DERIVATIVES_COLLECTOR=true` if you want trader-flow data to run automatically
- `PAPER_LEVERAGE=10`
- `RISK_MAX_TRADE_SIZE_PCT=0.50`
- `STRATEGY_MIN_EDGE_AFTER_FEES=0.001`
- `AUTO_MIN_HOLD_SECONDS=900`
- `AUTO_POSITION_MAX_LOSS_PCT=0.10`
- `AUTO_DEFAULT_STOP_LOSS_PCT=0.01`
- `NEWS_API_KEY` if using the news collector
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

The smoke test verifies `/health`, dashboard rendering, paper signal execution, collector status access, and database table creation.
