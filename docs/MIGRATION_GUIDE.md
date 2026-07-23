# Migrating an existing Anata deployment to V2

V2 keeps the existing collectors, features, model records, paper trades, positions,
equity, dashboard, authentication, and data export. It adds an audited decision path
beside those records. The migration is additive; it does not intentionally drop or
rename an existing table.

## Architecture change

Legacy paper path:

```text
FeatureBuilder -> one rule/uploaded strategy decision -> legacy risk/paper engine
```

V2 path:

```text
point-in-time feature snapshot
  -> narrow predictions -> registered signals -> regime ensemble
  -> portfolio target -> independent risk decision
  -> risk-authorized paper order/fill -> monitoring and Vision
```

Legacy `ai_decisions`, `paper_trades`, `positions`, and `account_equity` remain readable.
Vision labels legacy replay as partial instead of manufacturing V2 stages.

## Database additions

`Base.metadata.create_all()` creates the new V2 tables when absent:

```text
model_predictions              trading_signals
signal_outcomes                ensemble_decisions
ensemble_signal_weights        portfolio_targets
risk_decisions                 simulated_orders
simulated_fills                strategy_candidates
candidate_evaluations          champion_assignments
promotion_decisions            shadow_predictions
paper_sandbox_accounts         structured_news_events
external_ai_requests           model_health_snapshots
signal_health_snapshots        experiment_runs
decision_timeline_events       risk_control_state
```

The additive migration helper also extends existing operational tables where needed,
including point-in-time timestamps, V2 model-registry metadata, paper account/trace/
risk/order IDs, and hot-path indexes. Existing values are preserved; nullable/default
columns support rolling upgrades.

This repository does not use Alembic and does not keep a database schema-version
ledger. The helper is idempotent by inspection, but a failed migration can be partial,
so backup and verify each deployment.

## Pre-deploy checklist

1. Stop or disable the automatic paper trader for the migration window.
2. Back up PostgreSQL and verify the backup can be restored.
3. Record current `/health`, collector status, row counts, active legacy model, open
   paper positions, latest equity, and DB size.
4. Confirm there are no live exchange keys or order adapters; V2 supports paper mode
   only.
5. Keep automatic promotion disabled.

Safe migration variables:

```env
TRADING_MODE=paper
ANATA_V2_ENABLED=true
AUTO_TRADER_ENABLED=false
V2_AUTO_PROMOTE_CHAMPION=false
RESEARCH_AUTO_PROMOTE=false
EXTERNAL_AI_ENABLED=false
RISK_KILL_SWITCH_ENABLED=true
```

The temporary kill switch blocks new exposure while still permitting protective
reductions.

## Apply the migration

After installing the new production requirements and setting `DATABASE_URL`:

```powershell
python -c "from app.db.session import create_db_and_tables; print(create_db_and_tables())"
```

The application invokes the same function during startup. Running it explicitly first
makes the added-column report visible before background workers start.

Verify import and schema creation with a disposable database before PostgreSQL:

```powershell
$env:DATABASE_URL="sqlite:///./migration_smoke.db"
python -c "from app.db.session import create_db_and_tables; print(create_db_and_tables())"
python -c "import app.main; print(app.main.app.title)"
```

Use a disposable filename and remove it only after confirming its resolved path is in
the workspace.

## Staged activation

1. Start `WORKER_ROLE=web` and verify `/health`, `/dashboard`, `/vision`, and
   authenticated `/api/v2/registry`.
2. Start `WORKER_ROLE=collector`; wait for recent closed candles and news.
3. Start `WORKER_ROLE=enrichment` with external AI disabled. Confirm local structured
   news rows and no provider failure stops collection.
4. With the kill switch still on, call one `/api/v2/pipeline/run`. Confirm prediction,
   signal, ensemble, target, rejected risk, and timeline records share one trace.
5. Review safe exposure/leverage/cost limits and disable the persisted kill switch
   explicitly if appropriate.
6. Start `WORKER_ROLE=paper-trader` or the API paper runner.
7. Verify orders/fills reference persisted risk decisions and monitor equity/health.

No challenger becomes champion during these steps. Uploaded packages are candidates
until a confirmed manual lifecycle action.

## Model and feature compatibility

- Feature schema `price-news-market-v5` is current; older schema definitions remain
  available for legacy artifacts.
- Upload records include model family, lifecycle, checksum, preprocessing version,
  training dataset version, horizon, metrics, and compatibility metadata.
- Shadow/sandbox/promotion load the artifact and require feature/preprocessing
  metadata. Legacy packages may be admitted through the explicit compatibility path.
- Active champion assignments are resolved into load-validated
  `RegisteredArtifactModel` instances by family/symbol. With no assignment, the
  deterministic narrow baselines are the fallback unless a registered champion is
  required by configuration.

Do not edit an artifact in place. Register a new version and retain checksums and the
previous champion for rollback.

## Rollback

Model rollback is explicit and does not require an application rollback:

```powershell
Invoke-RestMethod -Method Post "$Url/api/v2/registry/rollback" `
  -Headers @{"x-admin-token"=$Token;"Content-Type"="application/json"} `
  -Body '{"model_family":"alpha.short_horizon_momentum","symbol_scope":"BTCUSDT","reason":"operator rollback","confirm":true}'
```

For an application rollback:

- set `AUTO_TRADER_ENABLED=false` and enable the risk kill switch first;
- deploy the prior application version;
- leave V2 tables/columns intact so history remains recoverable;
- do not manually delete new tables as part of rollback;
- verify legacy code tolerates the additive columns and re-check paper positions/equity.

`ANATA_V2_ENABLED=false` selects the legacy auto-trader path in the current code, but
that path retains the older mixed strategy-plan design. Use it only as a temporary
paper compatibility fallback, not as the target architecture.

## Current configuration caveats

`V2_USE_NARROW_MODELS` and `V2_REQUIRE_REGISTERED_CHAMPION` are active runtime gates.
`V2_MODEL_REGISTRY_DIR`, `RESEARCH_ENABLED`, and `RESEARCH_AUTO_PROMOTE` are parsed but
do not start a research worker: research remains local CLI-driven and automatic
promotion is absent. Verify actual lifecycle and trace records rather than relying on
configuration alone.

## Post-migration acceptance

```powershell
python -m pytest -q
```

Also confirm:

- app startup and an idempotent second migration succeed;
- no existing operational table or history was intentionally destroyed;
- trading mode is paper and no private exchange credential exists;
- external AI disabled/failure still produces local context;
- shadow records do not create exposure and sandboxes use unique fake accounts;
- every V2 simulated order has a persisted approved risk decision;
- stale data and the kill switch reject new exposure;
- Vision loads actual records and replays an end-to-end trace.

See [ANATA_V2_AUDIT.md](ANATA_V2_AUDIT.md) for the original risks and
[OPERATIONS_RUNBOOK.md](OPERATIONS_RUNBOOK.md) for current commands.
