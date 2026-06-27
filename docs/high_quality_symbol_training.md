# High-Quality Multi-Symbol Training

This workflow trains one shared model across many high-liquidity symbols while giving the model symbol identity features.

That means the AI can learn patterns like:

```text
BTC behaves differently from DOGE.
SOL behaves differently from LTC.
Meme coins behave differently from majors.
Layer 2 coins behave differently from old legacy coins.
```

But the model still learns shared market knowledge across all symbols.

## 1. Historical 365-Day Backfill

Run from the repo root on your PC:

```powershell
python historical_collectors/coingecko_history/collect_high_quality_history.py `
  --days 365 `
  --output-dir datasets/raw_days
```

This collects 30 high-quality/high-liquidity symbols by default:

```text
BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, ADAUSDT, DOGEUSDT, AVAXUSDT, LINKUSDT, LTCUSDT,
DOTUSDT, TRXUSDT, BCHUSDT, XLMUSDT, NEARUSDT, APTUSDT, ARBUSDT, OPUSDT, INJUSDT, SUIUSDT,
ATOMUSDT, FILUSDT, UNIUSDT, ETCUSDT, AAVEUSDT, ICPUSDT, SEIUSDT, RENDERUSDT, SHIBUSDT, PEPEUSDT
```

## 2. Prepare Dataset

```powershell
python scripts/prepare_training_data.py `
  --input datasets/raw_days `
  --output-dir datasets/processed `
  --news-converter smart
```

## 3. Train Normally

`train_best_model.py` now adds symbol identity features automatically by default.

```powershell
$Dataset = Get-ChildItem datasets/processed/anata_training_ready_*.csv.gz |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

python scripts/train_best_model.py `
  --dataset $Dataset.FullName `
  --out-dir models `
  --model-types sklearn_hist_gradient_boosting
```

The trainer adds columns like:

```text
symbol_is_BTCUSDT
symbol_is_DOGEUSDT
symbol_group_major
symbol_group_meme
symbol_group_layer1
symbol_group_defi
```

You can disable this only for comparison testing:

```powershell
python scripts/train_best_model.py `
  --dataset $Dataset.FullName `
  --out-dir models `
  --model-types sklearn_hist_gradient_boosting `
  --no-symbol-aware
```

## 4. Upload Normally

```powershell
$Package = Get-ChildItem models/model_package_*.zip |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

python scripts/upload_model.py `
  --url $Url `
  --token $Token `
  --package $Package.FullName
```

Then activate from the dashboard.

## 5. Real-Time Railway Collection

Use this preset when you are ready for Railway to collect the same expanded universe:

```text
presets/high_quality_symbols.env
```

Paste its variables into Railway.

Important: 30 symbols increases API calls and database rows. Use daily download + cleanup so the Railway DB plateaus instead of growing forever.
