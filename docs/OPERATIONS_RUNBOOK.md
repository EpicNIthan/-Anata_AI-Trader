# Operations runbook

This runbook is authoritative for commands present in this checkout. Anata remains
paper-only: there is no live exchange adapter, private trading-key configuration, or
withdrawal path.

## Install

Production/Railway dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Local research and tests:

```powershell
python -m pip install -r requirements-local-training.txt
python -m pip install -r requirements-dev.txt
# Optional laptop-only FinBERT/CryptoBERT preparation:
python -m pip install -r requirements-hf.txt
```

Do not install `requirements-hf.txt` on a small Railway service.

## Initialize or migrate the database

Set `DATABASE_URL`, then run the same additive create/migration function used at app
startup:

```powershell
python -c "from app.db.session import create_db_and_tables; print(create_db_and_tables())"
```

This calls SQLAlchemy `create_all()` and then the idempotent additive migration helper.
It does not intentionally drop existing tables. Back up PostgreSQL before a production
schema change. There is no Alembic command in this repository.

## Test and import checks

```powershell
python -m pytest -q
python -c "import app.main; print(app.main.app.title)"
python tests/smoke_test.py
```

Tests use paper/SQLite fixtures and do not require broker or paid-provider credentials.

## Runtime roles

Every role starts the FastAPI process; `WORKER_ROLE` decides which background jobs
start in its lifespan.

| `WORKER_ROLE` | Background work |
| --- | --- |
| `web` | None; API, dashboard, Vision only |
| `collector` | enabled public collectors plus data lifecycle |
| `paper-trader` | V2 automatic paper loop when `AUTO_TRADER_ENABLED=true` |
| `enrichment` | bounded local-first structured-news enrichment |
| `all` | all of the above; convenient locally, heavier on Railway |

`research` is not a valid runtime role. Heavy research is an explicit local CLI.

Run everything locally in one paper-only process:

```powershell
$env:TRADING_MODE="paper"
python scripts/run_worker.py --role all --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/health`, `/dashboard`, and `/vision`.

For a focused collector process:

```powershell
$env:ENABLE_MARKET_COLLECTOR="true"
$env:ENABLE_NEWS_COLLECTOR="true"
python scripts/run_worker.py --role collector --host 0.0.0.0 --port 8000
```

For the paper trader in a separately deployed service:

```powershell
$env:TRADING_MODE="paper"
$env:ANATA_V2_ENABLED="true"
$env:AUTO_TRADER_ENABLED="true"
python scripts/run_worker.py --role paper-trader --host 0.0.0.0 --port 8000
```

Alternatively start/stop it through an authenticated web/all process:

```powershell
$Headers = @{"x-admin-token"="YOUR_ADMIN_TOKEN"}
Invoke-RestMethod -Method Post http://localhost:8000/api/auto-trader/start -Headers $Headers
Invoke-RestMethod http://localhost:8000/api/auto-trader/status -Headers $Headers
Invoke-RestMethod -Method Post http://localhost:8000/api/auto-trader/stop -Headers $Headers
```

## Railway versus the local computer

Railway should run the lightweight production/paper side: PostgreSQL, web/Vision,
public collection, local-student or bounded external enrichment, frozen lightweight
inference, risk, paper execution, monitoring, and raw export. Use `WORKER_ROLE=all` for
one small service or separate `web`, `collector`, `enrichment`, and `paper-trader`
services sharing the same PostgreSQL database.

The local computer owns the permanent data lake, checksums, heavy news preparation,
model fitting, candidate search, historical/walk-forward evaluation, and report
archive. Keep `ENABLE_SERVER_TRAINING=false`, `RESEARCH_ENABLED=false`,
`RESEARCH_AUTO_PROMOTE=false`, and `V2_AUTO_PROMOTE_CHAMPION=false` on Railway.

## Synchronize Railway data

```powershell
$Url = "https://YOUR-APP.up.railway.app"
$Token = "YOUR_ADMIN_TOKEN"
python scripts/download_raw_data.py --url $Url --token $Token `
  --finished-only --daily-files --training-only `
  --output-dir local_data/raw_days
```

Only after a verified local download, optionally clean matching finished rows:

```powershell
python scripts/download_raw_data.py --url $Url --token $Token `
  --finished-only --daily-files --training-only `
  --output-dir local_data/raw_days `
  --cleanup-after-download --delete-railway-db-rows
```

The current unfinished day is protected by the finished-only cutoff. Review the
printed manifest and cleanup result.

## Prepare and train locally

Point-in-time numeric preparation:

```powershell
python scripts/prepare_training_data.py --input local_data/raw_days `
  --output-dir local_data/processed --news-converter smart
```

Run the complete multi-family forecast-only research cycle. It detects newly
available/corrected labels, searches bounded ridge/Huber baselines, refits inside
chronological walk-forward folds with label-overlap purging and embargo, packages one
challenger per available family, and registers each as `TRAINED` without promotion:

```powershell
python scripts/run_local_research_cycle.py `
  --input local_data/processed/YOUR_DATASET.csv.gz `
  --output-dir local_data/research
```

Run the same command again after synchronization; it exits with
`waiting_for_labels` until enough new labels exist. For one single-family artifact,
the lower-level trainer remains available:

```powershell
python scripts/train_narrow_return_model.py `
  --input local_data/processed/YOUR_DATASET.csv.gz `
  --output models/narrow_return_v1.json `
  --target target_future_return_5m `
  --forecast-horizon-seconds 300 `
  --report research_reports/narrow_return_v1.json
```

Artifacts emit expected return only; leverage/sizing/order targets are rejected. The
cycle writes immutable evaluation reports, checksummed packages, and stored OOS
metrics, but it never starts shadow/sandbox or promotes a champion.

The older, broader baseline comparison trainer remains available as
`scripts/train_best_model.py`, but `train_narrow_return_model.py` is the direct V2
forecast contract.

News teacher/student commands are in
[NEWS_TEACHER_STUDENT.md](NEWS_TEACHER_STUDENT.md). The shortest train command is:

```powershell
python scripts/train_news_student.py --dataset local_data/news_student_train.jsonl `
  --output models/news_student.json
```

## Historical and walk-forward evaluation

The evaluators consume chronological prediction-observation JSONL/JSON/CSV with the
fields expected by `scripts/research_utils.py`; they do not query the operational DB
or place paper trades.

```powershell
python scripts/evaluate_historical.py --input local_data/observations.jsonl `
  --report research_reports/historical.json `
  --feature-version price-news-market-v5

python scripts/run_walk_forward_evaluation.py --input local_data/observations.jsonl `
  --report research_reports/walk_forward.json `
  --train-size 2000 --validation-size 250 --test-size 250 `
  --step-size 250 --purge-size 15 --embargo-size 15
```

Run bounded declarative candidate evaluation when new labeled rows are available:

```powershell
python scripts/run_research_scheduler.py --input local_data/observations.jsonl `
  --state local_data/research_state.json --reports-dir research_reports `
  --minimum-new-rows 500 --max-candidates 5
```

This is a polling pass, not a daemon, and never promotes a champion.

## Register, shadow, sandbox, promote, and roll back

Upload a package as a challenger; the response contains its numeric model-version ID:

```powershell
python scripts/upload_model.py --url $Url --token $Token `
  --package models/model_package_VERSION.zip
```

From a trusted local/deployment shell whose `DATABASE_URL` points at the intended
database, a frozen JSON artifact can instead be registered directly:

```powershell
python scripts/manage_model_registry.py register-challenger `
  --artifact models/narrow_return_v1.json `
  --name narrow-return --version v1 --model-family alpha.linear_return
```

Use authenticated V2 lifecycle endpoints. All mutation requests require
`confirm=true` where shown:

```powershell
$Headers = @{"x-admin-token"=$Token;"Content-Type"="application/json"}
$ModelVersionId = 123

# Start non-executing shadow mode.
Invoke-RestMethod -Method Post "$Url/api/v2/models/$ModelVersionId/shadow" `
  -Headers $Headers -Body '{"reason":"reviewed technical compatibility","confirm":true}'

# Create an isolated fake account; profitability is not an admission requirement.
$Sandbox = Invoke-RestMethod -Method Post "$Url/api/v2/models/$ModelVersionId/sandbox" `
  -Headers $Headers -Body '{"name":"candidate-sandbox","starting_balance":10000,"confirm":true}'

# Run one pipeline cycle against that registered sandbox account.
Invoke-RestMethod -Method Post "$Url/api/v2/pipeline/run" -Headers $Headers `
  -Body (ConvertTo-Json @{symbol="BTCUSDT";paper_account_id=$Sandbox.sandbox.paper_account_id})

# Explicit manual champion promotion; use the family returned by registration.
Invoke-RestMethod -Method Post "$Url/api/v2/models/$ModelVersionId/promote" `
  -Headers $Headers `
  -Body '{"model_family":"alpha.linear_return","symbol_scope":"BTCUSDT","reason":"manual review completed","confirm":true}'

# Restore the prior recorded champion.
Invoke-RestMethod -Method Post "$Url/api/v2/registry/rollback" `
  -Headers $Headers `
  -Body '{"model_family":"alpha.linear_return","symbol_scope":"BTCUSDT","reason":"operator rollback","confirm":true}'
```

A champion assignment is loaded by the V2 family/symbol resolver only after artifact,
feature, preprocessing, lifecycle, and health validation. Confirm the selected model
and prediction IDs in Vision; invalid legacy artifacts fail closed rather than falling
through to an unintended older wildcard champion.

The same trusted-shell command provides direct lifecycle operations:

```powershell
python scripts/manage_model_registry.py start-shadow --model-version-id 123
python scripts/manage_model_registry.py start-sandbox --model-version-id 123 `
  --name candidate-sandbox --starting-balance 10000
python scripts/manage_model_registry.py promote --model-version-id 123 `
  --model-family alpha.linear_return --symbol-scope BTCUSDT `
  --reason "manual review completed"
python scripts/manage_model_registry.py rollback --model-family alpha.linear_return `
  --symbol-scope BTCUSDT --reason "operator rollback"
```

Run one bounded local paper cycle against the collected operational database:

```powershell
python scripts/run_paper_cycle.py --symbols BTCUSDT,ETHUSDT
```

For a sandbox, add `--paper-account-id sandbox-...`. The command refuses to run
unless `TRADING_MODE=paper`.

## Monitoring and incident checks

```powershell
Invoke-RestMethod -Method Post "$Url/api/v2/monitoring/run?symbol=BTCUSDT" -Headers @{"x-admin-token"=$Token}
Invoke-RestMethod "$Url/api/v2/attribution?symbol=BTCUSDT&paper_account_id=champion" -Headers @{"x-admin-token"=$Token}
Invoke-RestMethod "$Url/api/v2/risk/kill-switch" -Headers @{"x-admin-token"=$Token}
Invoke-RestMethod "$Url/api/vision/state?symbol=BTCUSDT" -Headers @{"x-admin-token"=$Token}
```

During an incident, enable the kill switch before debugging new exposure. Preserve
database/log evidence, inspect collector freshness, then decide whether to suspend,
roll back, or restart a paper-only worker.

## Deliberate boundaries and remaining operator wrappers

- No Alembic/dedicated migration CLI; use `create_db_and_tables()` as shown.
- No continuous local research daemon or automatic challenger uploader.
- Cancel/replace, open-order recovery, and account reconciliation exist as simulator
  service methods and tests, but have no authenticated operator REST/CLI wrapper or
  scheduled startup invocation.
- No automatic champion promotion by design.

The standalone stored-prediction evaluators consume `observations.jsonl`; the working
prepared-data train/refit/walk-forward path is `run_local_research_cycle.py`. These
boundaries should not be worked around with live-order credentials.
