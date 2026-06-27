# CoinGecko + GDELT Historical Collector

Standalone PC-only collector for building Anata raw daily files from free historical market/news sources.

This folder is intentionally isolated from the FastAPI/Railway app. It does not import `app/*`, does not start collectors, does not trade, and does not write to the database.

It writes training-compatible data under:

```text
datasets/raw_days/
```

By default it writes both:

```text
datasets/raw_days/coingecko_history/YYYY-MM-DD/*.gz
datasets/raw_days/raw_YYYY-MM-DD.zip
```

The expanded `coingecko_history/YYYY-MM-DD` folders are what `scripts/prepare_training_data.py` reads directly. The ZIP files are kept as compressed daily backups.

## What It Collects

From CoinGecko `/coins/{id}/market_chart/range`:

```text
price history
market cap history
total volume history
derived return/trend/volatility features
```

From GDELT DOC 2.0 Article List:

```text
historical crypto/news headlines
article URLs
published time
source domain
language
source country
basic raw_text made from headline + metadata
```

The collector writes each day as:

```text
candles.csv.gz
training_features.jsonl.gz
external_data_events.jsonl.gz
news_articles.jsonl.gz
manifest.json
```

## Important Limits

CoinGecko free/public historical data is useful, but it is not the same as Binance 1m exchange candles.

Important differences:

```text
CoinGecko price data is aggregated market data.
CoinGecko volume is total volume at the timestamp, not Binance candle volume.
CoinGecko free historical range is normally limited to the past 365 days.
CoinGecko granularity is automatic.
GDELT gives headlines/URLs/metadata, not full article bodies.
Some old GDELT windows can be sparse or unavailable depending on API coverage.
```

Default `--chunk-days 80` is used because CoinGecko normally returns hourly data for ranges up to 90 days. If you request more than 90 days in one API call, it normally returns daily data instead.

So for 1 year, this collector makes several smaller requests per coin to keep the data hourly when possible.

## Quick Start

From the repo root:

```powershell
python historical_collectors/coingecko_history/collect_coingecko_history.py `
  --days 365 `
  --output-dir datasets/raw_days
```

Then prepare:

```powershell
python scripts/prepare_training_data.py `
  --input datasets/raw_days `
  --output-dir datasets/processed `
  --news-converter smart
```

Then train:

```powershell
$Dataset = Get-ChildItem datasets/processed/anata_training_ready_*.csv.gz |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

python scripts/train_best_model.py `
  --dataset $Dataset.FullName `
  --out-dir models `
  --model-types sklearn_hist_gradient_boosting
```

## Safer First Test

Start with 30 days before downloading a full year:

```powershell
python historical_collectors/coingecko_history/collect_coingecko_history.py `
  --days 30 `
  --symbols BTCUSDT,ETHUSDT,SOLUSDT `
  --output-dir datasets/raw_days
```

## Market Only

If GDELT is slow or you only want market data:

```powershell
python historical_collectors/coingecko_history/collect_coingecko_history.py `
  --days 365 `
  --no-include-gdelt-news
```

## News Query

Default GDELT query:

```text
bitcoin OR btc OR ethereum OR eth OR solana OR xrp OR cardano OR dogecoin OR avalanche OR chainlink OR litecoin OR crypto OR cryptocurrency OR stablecoin OR binance OR coinbase OR etf
```

Custom query example:

```powershell
python historical_collectors/coingecko_history/collect_coingecko_history.py `
  --days 90 `
  --gdelt-query '(bitcoin OR ethereum OR crypto OR etf OR stablecoin OR binance)'
```

## Default Coins

```text
BTCUSDT -> bitcoin
ETHUSDT -> ethereum
SOLUSDT -> solana
BNBUSDT -> binancecoin
XRPUSDT -> ripple
ADAUSDT -> cardano
DOGEUSDT -> dogecoin
AVAXUSDT -> avalanche-2
LINKUSDT -> chainlink
LTCUSDT -> litecoin
```

Add custom CoinGecko coin IDs like this:

```powershell
python historical_collectors/coingecko_history/collect_coingecko_history.py `
  --days 365 `
  --symbols BTCUSDT:bitcoin,ETHUSDT:ethereum,PEPEUSDT:pepe
```

## API Key

No key is required for public/demo usage, but CoinGecko rate limits can be strict. If you have a demo key:

```powershell
$env:COINGECKO_DEMO_API_KEY="your_key"

python historical_collectors/coingecko_history/collect_coingecko_history.py --days 365
```

For a paid Pro key:

```powershell
$env:COINGECKO_API_KEY="your_key"

python historical_collectors/coingecko_history/collect_coingecko_history.py `
  --pro `
  --days 365
```

## Notes For Training

This history is good for giving the AI more market vision quickly:

```text
longer trend history
large volatility periods
market cap context
volume regime changes
older bull/bear/range periods
news headline sentiment context
macro/regulation/security keywords from old news
```

It does not replace real-time exchange data. The best setup is:

```text
CoinGecko history = fast old market context
GDELT history = old headline/news context
Railway real-time collector = cleaner live exchange candles/news/trades
PC prepare/train = combine both into training data
```
