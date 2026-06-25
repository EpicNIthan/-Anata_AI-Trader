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

Optional local Hugging Face sentiment dependencies are intentionally separate so Railway does not need heavy packages by default:

```powershell
python -m pip install -r requirements-hf.txt
```

Do not install `requirements-hf.txt` on small Railway containers. Railway should use `HF_SENTIMENT_BACKEND=api` with `HF_API_TOKEN` instead, otherwise local `torch`/`transformers` can run the deployment out of memory.

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

Optional external context collectors are off by default and store compact numeric rows in `external_data_events`:

- Fear/Greed: Alternative.me crypto fear/greed index.
- Global market: CoinGecko public global endpoint.
- Liquidations: Binance Futures force-order websocket rollups.
- Stablecoin risk: DefiLlama stablecoin peg/supply data.
- Macro risk: news-derived macro/regulation/security/ETF/world-risk scores.

```env
ENABLE_FEAR_GREED_COLLECTOR=false
ENABLE_GLOBAL_MARKET_COLLECTOR=false
ENABLE_LIQUIDATION_COLLECTOR=false
ENABLE_STABLECOIN_COLLECTOR=false
ENABLE_STABLECOIN_RISK_COLLECTOR=false
ENABLE_MACRO_RISK_COLLECTOR=false
STORE_RAW_EXTERNAL_EVENTS=false
STORE_RAW_LIQUIDATIONS=false
RAW_EXTERNAL_EVENT_RETENTION_DAYS=7
```

Run a safe mock collection pass:

```powershell
curl -X POST http://localhost:8000/api/external/run-once -H "Content-Type: application/json" -d "{\"mock\":true}"
curl -X POST http://localhost:8000/api/liquidations/run-once -H "Content-Type: application/json" -d "{\"mock\":true}"
curl http://localhost:8000/api/external/status
```

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
RAW_EXTERNAL_EVENT_RETENTION_DAYS=7
RAW_TICK_RETENTION_DAYS=7
DIAGNOSTIC_RETENTION_DAYS=7
EXTERNAL_DATA_RETENTION_DAYS=365
TRAINING_FEATURE_RETENTION_DAYS=365
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
curl http://localhost:8000/api/storage/status
curl -X POST http://localhost:8000/api/db/compact
curl -X POST http://localhost:8000/api/storage/cleanup/run
curl -X POST http://localhost:8000/api/storage/compact
curl -X POST http://localhost:8000/api/db/cleanup
curl -X POST http://localhost:8000/api/db/archive
```

Reprocess existing news sentiment after enabling Hugging Face:

```powershell
curl -X POST http://localhost:8000/api/sentiment/reprocess -H "Content-Type: application/json" -d "{\"limit\":200,\"reset_model\":true}"
```

Railway-safe Hugging Face sentiment should use the hosted HF Inference API instead of loading `torch` in the Railway container:

```env
ENABLE_HF_SENTIMENT=true
HF_SENTIMENT_BACKEND=api
HF_API_TOKEN=hf_your_token_here
NEWS_SENTIMENT_MODEL=mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis
```

For Railway memory safety:

- Keep `HF_SENTIMENT_BACKEND=api`, not `local`.
- Do not add `torch`, `transformers`, `lightgbm`, or `xgboost` to Railway default requirements.
- Keep `ENABLE_SERVER_TRAINING=false`.
- Train on your laptop with `requirements-local-training.txt`.
- Default Railway requirements are intentionally lean. Uploaded sklearn `.joblib` models may need extra inference dependencies or a larger Railway plan; if unavailable, the app safely falls back to the rule-based strategy.

Then check:

```powershell
curl -H "x-admin-token: YOUR_ADMIN_TOKEN" https://your-app.up.railway.app/api/sentiment/model-status
curl -X POST https://your-app.up.railway.app/api/sentiment/reprocess `
  -H "x-admin-token: YOUR_ADMIN_TOKEN" `
  -H "Content-Type: application/json" `
  -d "{\"limit\":200,\"reset_model\":true}"
```

`/api/dashboard/summary` and the dashboard sentiment status show whether HF really loaded. If `hf_loaded=false`, the app is still using the rule-based fallback. This can happen if `HF_API_TOKEN` is missing, Hugging Face rejects/rate-limits the request, Railway cannot reach HF, or the variable is not enabled. Quoted booleans like `ENABLE_HF_SENTIMENT="true"` are accepted, but Railway variables should normally be entered without quotes.

Local laptop-only HF is still possible with:

```env
HF_SENTIMENT_BACKEND=local
```

and `pip install -r requirements-hf.txt`, but this is not recommended on small Railway containers.

Recommended heavy-news workflow:

Railway only collects raw news. Your laptop converts raw news text into smarter numeric sentiment, uploads those scores, then you rebuild/export/train.

```powershell
$Url = "https://anataai-trader-production.up.railway.app"
$Token = "YOUR_ADMIN_TOKEN"

# 1. Download raw news from Railway.
python scripts/download_raw_news.py --url $Url --token $Token --since-date 2026-06-24 --output datasets/latest_raw_news.csv.gz

# 2. On your laptop, install heavy sentiment deps if you want Hugging Face locally.
python -m pip install -r requirements-hf.txt

# 3. Score raw news locally. If transformers is unavailable, this still writes a fallback file.
python scripts/score_news_local.py --input datasets/latest_raw_news.csv.gz --output datasets/latest_news_sentiment.jsonl.gz

# 4. Upload smarter sentiment rows back to Railway.
python scripts/upload_sentiment.py --url $Url --token $Token --file datasets/latest_news_sentiment.jsonl.gz

# 5. Rebuild compact training rows so they use the uploaded sentiment scores, then export/train.
Invoke-RestMethod -Method Post -Uri "$Url/api/training/build-dataset" `
  -Headers @{"x-admin-token"=$Token;"Content-Type"="application/json"} `
  -Body '{"symbols":["BTCUSDT","ETHUSDT"],"days":14,"max_rows_per_symbol":5000,"stride":5,"backfill":false,"export":false}'

python scripts/download_dataset.py --url $Url --token $Token --since-date 2026-06-24 --no-auto-build-labels --output datasets/latest.csv.gz
python scripts/train_local_model.py --dataset datasets/latest.csv.gz --model-type sklearn_hist_gradient_boosting --target target_trade_quality_score
```

This loop can be repeated: collect more data on Railway, download raw news/dataset again, score/train on your laptop, upload a new candidate model, activate only after metrics look good.

Simple laptop data-factory workflow:

Railway should be treated as a temporary collector, not your long-term warehouse. Your PC keeps the permanent archive under `local_data/`, and Railway keeps collecting fresh data after cleanup.

```powershell
$Url = "https://anataai-trader-production.up.railway.app"
$Token = "YOUR_ADMIN_TOKEN"

# Download raw news, labeled training dataset, and a collection report into local_data/YYYYMMDD_HHMMSS/.
python scripts/sync_laptop_data.py --url $Url --token $Token --since-date 2026-06-24

# Same download, then compact Railway only after the files are written successfully.
python scripts/sync_laptop_data.py --url $Url --token $Token --since-date 2026-06-24 --clear-railway-after-download --railway-keep-days 7
```

The sync folder contains:

- `training_dataset.csv.gz` for local model training.
- `raw_news.csv.gz` for heavier local news-to-number processing.
- `collection_report_before.json` showing what Railway collected.
- `manifest.json` with file sizes, row counts, export IDs, and cleanup result.

Use `local_data/` as your laptop data lake. Keep every dated folder. When training, combine the older local datasets with the newest download so the model learns from all history while Railway only stores recent working data.

To see what Railway is collecting right now:

```powershell
Invoke-RestMethod -Uri "$Url/api/data/collection-report" -Headers @{"x-admin-token"=$Token}
```

The dashboard `DB Storage` tab also has a `Collection Report` button. It shows current rows, label coverage, news providers, sentiment models, and the next data-quality improvements.

Simplest daily bundle workflow:

Railway can package useful training data into one folder per UTC day. A complete 24-hour day is marked `finished`; the current day is marked `unfinished`. After your laptop downloads the finished bundles, Railway can delete the finished DB rows and keep only the unfinished current day.

```powershell
$Url = "https://anataai-trader-production.up.railway.app"
$Token = "YOUR_ADMIN_TOKEN"

# Download daily zip folders into local_data/daily_bundles/.
python scripts/sync_daily_bundles.py --url $Url --token $Token --days 7

# Download, then delete finished Railway data. Today's unfinished day remains collecting.
python scripts/sync_daily_bundles.py --url $Url --token $Token --days 7 --delete-finished-from-railway
```

Each daily bundle contains only useful training data:

- `candles.csv.gz` closed training-quality candles.
- `news_articles.csv.gz` raw news for local news AI.
- `news_sentiment.csv.gz` current sentiment scores.
- `external_data_events.csv.gz` trader-flow, macro, liquidation, fear/greed, stablecoin context.
- `features.csv.gz` and `training_features.csv.gz`.
- `experience_buffer.csv.gz`, `ai_decisions.csv.gz`, `paper_trades.csv.gz`, and `account_equity.csv.gz`.

Recommended normal loop:

1. Railway collects nonstop.
2. Download daily bundles to your PC.
3. Delete finished Railway data, keep unfinished day.
4. Train locally using all folders in `local_data/daily_bundles/`.
5. Upload model package as candidate.
6. Activate the model.
7. Choose `Bot` or `Trained AI` in the dashboard paper runner. Both modes still collect data for later training.

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

Features are stored as versioned JSON payloads. The current schema is `price-news-market-v4`; `price-news-v3`, `price-news-v2`, and `price-news-v1` remain available for older models. Missing future values default to `0` or `null` so older models keep running when new data sources are added.

`price-news-v3` adds public trader-flow features:

- crowd long/short account percentages
- top-trader account and position long percentages
- taker buy pressure and buy/sell ratio
- open interest value and open interest change
- funding rate
- combined `trader_crowd_score`
- combined `crowd_risk_score`

`price-news-market-v4` keeps all v3 columns and adds:

- fear/greed value and 1d change
- prompt-compatible aliases such as `fear_greed_change_24h`, `market_cap_change_24h`, `usdt_deviation`, and `stablecoin_supply_change_24h`
- global market cap and volume change
- BTC dominance and dominance change
- liquidation long/short/total/imbalance/spike rollups
- USDT/USDC peg deviation and stablecoin depeg risk
- macro, regulation, security, ETF, and world risk scores
- combined `market_regime_score`

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

PowerShell workflow for your Railway app:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-local-training.txt

$Url = "https://anataai-trader-production.up.railway.app"
$Token = "YOUR_ADMIN_TOKEN"
$Headers = @{
  "x-admin-token" = $Token
  "Content-Type" = "application/json"
}

# 1. Build labels first so export does not spend the whole request doing heavy label work.
Invoke-RestMethod -Method Post -Uri "$Url/api/training/build-labels" -Headers $Headers -Body "{}"

# 2. Export a date range and only print the dataset_id. Use a recent since_date if Railway is slow.
$DatasetId = python scripts/download_dataset.py --url $Url --token $Token --since-date 2026-06-24 --no-auto-build-labels --timeout 600 --export-only
$DatasetId

# Optional full export after compaction/labeling:
# $DatasetId = python scripts/download_dataset.py --url $Url --token $Token --use-all-data --no-auto-build-labels --timeout 900 --export-only

# 3. Download by dataset_id. This can be retried without creating another export.
python scripts/download_dataset.py --url $Url --token $Token --dataset-id $DatasetId --output datasets/latest.csv.gz --timeout 600

# 4. Train locally.
python scripts/train_local_model.py --dataset datasets/latest.csv.gz --model-type sklearn_hist_gradient_boosting --target target_trade_quality_score

python scripts/train_local_model.py --dataset datasets/latest.csv.gz --model-type lightgbm --target target_trade_quality_score

python scripts/evaluate_local_model.py --dataset datasets/latest.csv.gz --model models/YOUR_MODEL.joblib

# 5. Package, upload as candidate, then activate only after checking metrics.
python scripts/package_model.py --model models/YOUR_MODEL.joblib

python scripts/upload_model.py --url $Url --token $Token --package model_package_VERSION.zip

python scripts/activate_model.py --url $Url --token $Token --model-id MODEL_ID
```

For large Railway datasets, `scripts/download_dataset.py` supports:

```powershell
python scripts/download_dataset.py --url $Url --token $Token --since-date 2026-06-24 --timeout 600
python scripts/download_dataset.py --url $Url --token $Token --use-all-data --timeout 900
python scripts/download_dataset.py --url $Url --token $Token --since-date 2026-06-24 --export-only
python scripts/download_dataset.py --url $Url --token $Token --dataset-id anata_dataset_YYYYMMDD_HHMMSS.csv.gz --output datasets/latest.csv.gz
```

If export times out, build labels first, run DB compact from the dashboard/Data tab, or retry with a smaller `--since-date` range.

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
