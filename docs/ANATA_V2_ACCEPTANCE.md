# Anata V2 acceptance report

Acceptance date: 2026-08-04  
Scope: paper-only quantitative platform in this checkout  
Initial findings: [ANATA_V2_AUDIT.md](ANATA_V2_AUDIT.md)

## Outcome

The coherent V2 path is implemented and executable:

```text
public data -> verified archive -> point-in-time preparation
-> narrow forecasts -> signals -> regime ensemble -> portfolio target
-> independent risk -> simulated paper execution -> monitoring/attribution
-> AI Vision
```

The platform remains deliberately unable to place a live exchange order. Model
training, a positive backtest, shadow status, or sandbox status cannot bypass risk or
silently create a champion assignment.

Implementation acceptance is not model-performance acceptance. The real-data run in
this report produced negative cost-adjusted walk-forward expectancy for every narrow
family, so no candidate was promoted.

## Architecture migration

The legacy center was one `StrategyDecision` carrying forecast, leverage, margin,
stop/take, and execution intent into `PaperEngine`. An uploaded AI plan could return
early from risk checks. The V2 center is a typed, persisted dependency chain:

- narrow models output forecasts only;
- signal records have validity/lifecycle metadata;
- the ensemble must exist before a portfolio target;
- portfolio code requests exposure but cannot approve it;
- the independent risk engine applies universal gates and persists approval;
- paper execution verifies and consumes the matching persisted approval once;
- every persisted decision stage is trace-linked for replay, monitoring, and
  attribution; feature lineage is linked through the recorded snapshot ID and
  timeline evidence.

Legacy routes, collectors, tables, paper records, exports, and the original dashboard
remain available as compatibility surfaces. V2 endpoints and Vision are additive.

## Critical defects closed

- The AI-plan early-return bypass no longer escapes daily loss, drawdown, cooldown,
  position-count, leverage, margin, fee, freshness, or kill-switch controls.
- A model cannot choose notional, leverage, margin, stops, exits, or order details.
- Target exposure is no longer multiplied by leverage a second time at execution.
- Champion, shadow, and sandbox registry states are connected to runtime resolution;
  shadow predictions cannot become signals or exposure.
- Sandbox cash, positions, equity, risk caps, history, and Vision reads are partitioned
  by exact fake account ID.
- V2 simulated-order submission requires a matching persisted `APPROVED` risk
  decision and is replay-resistant. Backward-compatible manual/strategy/auto-trader
  callers now persist a labeled compatibility ensemble, portfolio target, and risk
  decision before `PaperEngine` may fill; audit persistence failure rejects execution.
- Market data is rechecked when a resting order later matches.
- Per-fill fees, spread/slippage, impact, partial fills, simplified funding cash flow,
  order recovery/expiry, cancel/replace, and reconciliation are persisted.
- Railway cleanup now requires proof of an exact closed, checksummed local archive
  before deleting finished remote rows.
- Current feature quality checks cover series-scoped ordering, gaps, duplicates,
  revisions, invalid OHLCV, outliers, stale data, schema/provider/missingness, and
  bundle completeness.

## Real archived-data execution

The acceptance run used existing local daily archives and an isolated SQLite registry;
it did not mutate a Railway database or activate a deployed model.

### Market preparation

Source: `datasets/raw_days-save/raw_2026-07-07.zip`

```text
prepared rows                         8,830
15m trade-quality labels              8,700
finite available 5m-return labels     8,790
symbols                               10
duplicate rows                        0
missing required columns              0
news coverage                         99.66%
derivatives coverage                  100%
external-context coverage             100%
future leakage detected               false
observed period                       about 15.3 hours
```

The short period is an operational proof only. The preparation report correctly warns
that two or more days are recommended before training.

### Narrow-model cycle

`run_local_research_cycle.py` detected 8,790 finite, currently available labels,
searched 27 deterministic ridge/Huber configurations, refit each configuration inside
chronological walk-forward folds, selected one challenger for each of nine available
families, verified its package, wrote compatible OOS JSONL, and registered nine
`TRAINED` versions plus durable checksum-verified artifact bytes in the isolated
registry. It transactionally stored 27 candidates, 27 evaluations, and 27 experiment
records with exact fold boundaries. It created no assignment or promotion.

Each selected family had 4,265 OOS observations. Net expectancy after the configured
0.0008 cost was negative for every family (approximately -0.00070 to -0.00099 per
observation), with very large drawdowns on this short/high-turnover sample. Those
packages are useful compatibility baselines, not champion evidence.

A second invocation against the same snapshot returned `waiting_for_labels`, reported
zero new labeled rows, trained zero candidates, and registered nothing. Corrected or
late labels change their stable row fingerprint and will be reconsidered.

### Bounded paper cycle

One BTCUSDT cycle against the isolated registry persisted one feature, eight baseline
predictions, eight signals, one ensemble decision, one portfolio target, one risk
decision, and ten timeline events. Risk rejected the requested exposure for low
confidence, missing current market data, and missing required features. It created:

```text
simulated orders    0
simulated fills     0
paper trades        0
```

This is a successful fail-closed operational result, not a trading success.

### News teacher/student cycle

The offline teacher read `news_articles.jsonl.gz` directly from two real daily ZIPs.
It produced 56 earlier training rows and 38 later holdout rows with zero schema or
numeric-claim rejections. The compact dependency-free student packaged successfully,
but held-out imitation quality was weak:

```text
sentiment-label accuracy    42.11%
event-type accuracy         34.21%
sentiment MAE                0.6316
```

The student was not uploaded or activated. These are imitation metrics, never trading
profitability metrics.

## Database and migration shape

The new coherent set is:

```text
model_predictions             trading_signals
signal_outcomes               ensemble_decisions
ensemble_signal_weights       portfolio_targets
risk_decisions                risk_control_state
simulated_orders              simulated_fills
strategy_candidates           candidate_evaluations
experiment_runs               champion_assignments
promotion_decisions           shadow_predictions
paper_sandbox_accounts        structured_news_events
external_ai_requests          model_health_snapshots
signal_health_snapshots       decision_timeline_events
model_artifact_blobs
```

`model_artifact_blobs` stores immutable package bytes and SHA-256 beside the registry
record, so separate Railway roles do not depend on the uploader container's local
filesystem. Existing legacy tables gain nullable point-in-time, account-partition,
trace, registry, and health fields plus indexes for time, symbol, model, signal,
account, lifecycle, and health queries.

Startup still runs `create_all()` followed by the idempotent additive migration helper.
The helper creates the durable artifact table for an existing deployment and adds only
missing nullable/defaulted columns and `IF NOT EXISTS` indexes. It contains no table
drop, destructive rename, truncation, or data reset.

## Railway versus laptop

Railway runs lightweight public collection, frozen inference, enrichment fallback,
risk, paper simulation, monitoring, export, APIs, and Vision. It installs only
`requirements.txt`; torch, transformers, LightGBM, and XGBoost remain outside that
file. Server training, research, external AI, and automatic promotion are disabled by
safe defaults.

The laptop owns verified permanent archives, heavy news conversion, labels, candidate
search, fitting, historical/walk-forward evaluation, packages, and manual lifecycle
operations. Provider secrets and paid APIs are optional and were not required for the
test or acceptance workflows.

## External-AI workflow

News intelligence is local-first. A deterministic provider is always available; an
installed compact student replaces it as the base local model, while provider failure
falls back to rules. External calls occur only for non-duplicate, relevant,
important/uncertain content after content-hash/prompt cache, quota, budget, rate, and
circuit checks. Every HTTP attempt—including retries—is separately bounded and audited.

UTC daily usage, UTC monthly spend, last-provider request time, consecutive failures,
and successful cache entries hydrate from `external_ai_requests` after a restart, so a
restart or another URL cannot reset the safety budget. Only typed validated structured
events reach features, and external influence is capped by
`V2_EXTERNAL_CONTEXT_MAX_ADJUSTMENT`. Keys and provider exception text are never
persisted. With all providers disabled or failing, the paper cycle continues on base
market, derivatives, regime, local-news, and approved local-alpha inputs.

The heavy teacher remains laptop-only. `run_teacher_extraction.py --teacher-mode hf`
can merge a pinned local Hugging Face sentiment teacher into the validated structured
schema; rule mode is the dependency-free baseline. Teacher outputs must pass validation
before student-dataset construction, chronological holdout evaluation, packaging,
manual upload, and manual `news.student` activation.

## AI Vision routes

- Browser page: `/vision`; compatibility alias: `/dashboard/vision`.
- Browser token exchange: `/admin/login?next=/vision` (body token to strict HTTP-only
  cookie; credentials in URL queries are rejected).
- Read APIs: `/api/vision/symbols`, `/api/vision/chart`, `/api/vision/overlays`,
  `/api/vision/state`, `/api/vision/models`, `/api/vision/history`,
  `/api/vision/decisions`, `/api/vision/replay/{trace_id}`, and
  `/api/vision/research`.

Vision reads persisted facts only. It cannot submit an order, change risk, activate a
student, or promote a trading model.

## Configuration variables added

The full defaults and descriptions are in `.env.example` and
[RAILWAY_VARIABLES.md](RAILWAY_VARIABLES.md). New groups are:

- V2/lifecycle: `ANATA_V2_ENABLED`, `V2_USE_NARROW_MODELS`,
  `V2_REQUIRE_REGISTERED_CHAMPION`, `V2_AUTO_PROMOTE_CHAMPION`,
  `V2_CHAMPION_ACCOUNT_ID`, `V2_DEFAULT_FORECAST_HORIZON_SECONDS`,
  `V2_SIGNAL_TTL_SECONDS`, `V2_MIN_NET_EDGE`, `V2_MAX_POSITION_LEVERAGE`,
  `V2_MAX_SYMBOL_EXPOSURE_PCT`, `V2_MAX_GROSS_EXPOSURE_PCT`,
  `V2_MAX_NET_EXPOSURE_PCT`, `V2_MAX_CLUSTER_EXPOSURE_PCT`,
  `V2_MIN_LIQUIDITY_SCORE`, `V2_MAX_EXPECTED_COST_PCT`,
  `V2_EXTERNAL_CONTEXT_MAX_ADJUSTMENT`, `V2_CORRELATION_PENALTY_THRESHOLD`,
  `V2_SANDBOX_MAX_EXPOSURE_PCT`, and `V2_MODEL_REGISTRY_DIR`.
- Independent risk: `RISK_KILL_SWITCH_ENABLED`, `RISK_CONFIGURATION_VERSION`,
  `RISK_MAX_MARKET_DATA_AGE_SECONDS`, `RISK_MAX_PORTFOLIO_DRAWDOWN_PCT`,
  `RISK_MAX_PORTFOLIO_LEVERAGE`, `RISK_MAX_SPREAD_PCT`,
  `RISK_MAX_EXPECTED_TRANSACTION_COST_PCT`, `RISK_MAX_FEE_EXPOSURE_PCT`, and
  `RISK_REQUIRE_FRESH_DATA`.
- Paper simulation: `PAPER_SIMULATED_SPREAD_PCT`,
  `PAPER_SIMULATED_SLIPPAGE_PCT`, `PAPER_SIMULATED_LATENCY_MS`,
  `PAPER_SIMULATED_VOLUME_PARTICIPATION`, `PAPER_SIMULATED_PARTIAL_FILL_ENABLED`,
  `PAPER_SIMULATED_FUNDING_RATE`, `PAPER_SIMULATED_MARKET_IMPACT_COEFFICIENT`, and
  `PAPER_SIMULATED_ORDER_TTL_SECONDS`.
- External/local news: `EXTERNAL_AI_ENABLED`, `EXTERNAL_AI_PROVIDER_ORDER`,
  `EXTERNAL_AI_DAILY_REQUEST_LIMIT`, `EXTERNAL_AI_MONTHLY_BUDGET_USD`,
  `EXTERNAL_AI_TIMEOUT_SECONDS`, `EXTERNAL_AI_MAX_RETRIES`,
  `EXTERNAL_AI_IMPORTANCE_THRESHOLD`, `EXTERNAL_AI_LOCAL_UNCERTAINTY_THRESHOLD`,
  `EXTERNAL_AI_PROMPT_VERSION`, `EXTERNAL_AI_CIRCUIT_BREAKER_FAILURES`,
  `EXTERNAL_AI_CIRCUIT_BREAKER_SECONDS`, `EXTERNAL_AI_CACHE_TTL_SECONDS`,
  `EXTERNAL_AI_PROVIDER_MIN_INTERVAL_SECONDS`, provider endpoint/model/pricing/key
  groups for Gemini/Groq/Hugging Face/generic, `LOCAL_NEWS_STUDENT_PATH`,
  `LOCAL_NEWS_STUDENT_VERSION`, and the `ENRICHMENT_*` controls. All example keys
  are empty.
- Vision/health/research: `VISION_*`, `MONITORING_*`, `HEALTH_*`,
  `RESEARCH_ENABLED`, `RESEARCH_AUTO_PROMOTE`, `RESEARCH_DATA_LAKE_DIR`,
  `RESEARCH_REPORT_DIR`, and `RESEARCH_SCHEDULER_INTERVAL_SECONDS`.

Railway should set only deployment-specific values and secrets, while pinning the
paper-only and no-auto-promotion invariants. Empty authentication configuration now
fails closed rather than exposing administrative APIs.

## Required command matrix

The authoritative commands, environment boundaries, and lifecycle examples are in
[OPERATIONS_RUNBOOK.md](OPERATIONS_RUNBOOK.md). The essential sequence is:

```powershell
# Install production and local/test dependencies
python -m pip install -r requirements.txt
python -m pip install -r requirements-local-training.txt
python -m pip install -r requirements-dev.txt

# Additive database creation/migration and tests
python -c "from app.db.session import create_db_and_tables; print(create_db_and_tables())"
python -m pytest -q

# Web, collectors, enrichment, and paper worker (paper mode only)
python scripts/run_worker.py --role all --host 0.0.0.0 --port 8000

# Verified Railway synchronization
python scripts/download_raw_data.py --url $Url --token $Token `
  --finished-only --daily-files --training-only --output-dir local_data/raw_days

# Prepare and run all narrow-family research locally
python scripts/prepare_training_data.py --input local_data/raw_days `
  --output-dir local_data/processed --news-converter smart
python scripts/run_local_research_cycle.py `
  --input local_data/processed/YOUR_DATASET.csv.gz --output-dir local_data/research

# One bounded paper-only decision cycle
python scripts/run_paper_cycle.py --symbols BTCUSDT,ETHUSDT
```

Separate documented commands cover historical evaluation, stored-prediction
walk-forward evaluation, teacher validation, student training/evaluation/packaging,
challenger registration/upload, shadow, isolated sandbox, explicit manual promotion,
rollback, monitoring, attribution, and Vision.

## Final validation matrix

| Check | Result |
| --- | --- |
| Application import | Pass; FastAPI application and all registered routes load |
| Clean database | Pass; 38 tables, database ping successful |
| Migration idempotence | Pass; second create/migration run adds zero tables/columns |
| Destructive migration scan | Pass; no drop, truncate, destructive rename, or reset path |
| Default execution boundary | Pass; `TRADING_MODE=live` is rejected and the auto trader is paper-only |
| Private exchange/withdrawal primitives | None found |
| Authentication defaults | Pass; admin/dashboard/API routes return `503` when credentials are absent; health/login remain public |
| Query-secret safety | Pass; URL tokens are rejected; browser login uses body to HTTP-only cookie |
| Shadow/sandbox isolation | Pass in lifecycle and exact-account regression tests |
| External provider failure | Pass; local model → signal → ensemble/paper loop continues |
| Artifact durability/integrity | Pass; path-loss materialization and outer/member tamper tests reject bad bytes |
| Remote deserialization boundary | Pass; pickle/joblib/non-JSON uploads are rejected |
| Secret scan | Pass; no high-confidence key/private-key pattern in tracked or untracked source files |
| Docker context | Pass; `.env`, DBs, models, archives, and datasets excluded |
| Railway dependencies | Pass; 10 runtime packages, no torch/transformers/LightGBM/XGBoost |
| Full automated test suite | Final count recorded below after the combined rerun |

The only test warning is the existing Starlette deprecation notice for its current
`httpx` TestClient integration; it is not an application failure.

## Remaining evidence and simulation limits

- No fresh Railway download/cleanup was attempted without the user's deployment URL
  and admin credential. The code and deterministic integrity tests cover that path;
  remote deletion remains fail-closed.
- The exercised market sample is much too short for a champion decision, and all nine
  selected baselines lost after the assumed cost.
- The news student needs more diverse chronological teacher history and stronger
  held-out accuracy before activation.
- The simulator does not model an exchange order book, queue position, outages,
  delistings, stochastic impact, or periodic directional funding. Resting-order market
  events require an explicit worker/service call.
- Monitoring thresholds and attribution are empirical paper diagnostics, not
  guarantees of significance, capacity, causality, or profitability.
- Promotion and rollback remain manual paper-registry actions by design.

Live trading remains unavailable: there is no private exchange-order adapter,
withdrawal path, live-mode default, or repository credential for real-money execution.
