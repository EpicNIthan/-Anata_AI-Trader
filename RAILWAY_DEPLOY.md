# Railway deployment checklist

This repository builds through `Dockerfile` and `railway.json`. The image installs only
`requirements.txt`; heavy Hugging Face/news preparation and research dependencies stay
on the local computer.

## 1. Database and secrets

Add a Railway PostgreSQL service. Railway supplies `DATABASE_URL`; the app accepts the
Railway `postgresql://...` form and uses `psycopg`.

Set a strong admin credential:

```env
ADMIN_TOKEN=choose-a-long-random-secret
```

Or configure both `DASHBOARD_USERNAME` and `DASHBOARD_PASSWORD`. Do not commit real
values. `/health` is public; dashboard and `/api/*` lab/V2/Vision endpoints use admin
authentication.

## 2. Pin paper-only V2 safety

Recommended first deployment:

```env
TRADING_MODE=paper
WORKER_ROLE=all
ANATA_V2_ENABLED=true
AUTO_TRADER_ENABLED=false
ENABLE_SERVER_TRAINING=false
V2_AUTO_PROMOTE_CHAMPION=false
RESEARCH_ENABLED=false
RESEARCH_AUTO_PROMOTE=false
EXTERNAL_AI_ENABLED=false
STORE_MARKET_TICKS=false
PAPER_LEVERAGE=3
PAPER_MAX_LEVERAGE=3
V2_MAX_POSITION_LEVERAGE=3
RISK_MAX_PORTFOLIO_LEVERAGE=3
RISK_MAX_TRADE_SIZE_PCT=0.10
V2_MAX_SYMBOL_EXPOSURE_PCT=0.10
V2_MAX_GROSS_EXPOSURE_PCT=0.40
V2_MAX_NET_EXPOSURE_PCT=0.25
V2_SANDBOX_MAX_EXPOSURE_PCT=0.03
PAPER_SIMULATED_MARKET_IMPACT_COEFFICIENT=0
PAPER_SIMULATED_PARTIAL_FILL_ENABLED=false
PAPER_SIMULATED_ORDER_TTL_SECONDS=300
```

Most collector/provider defaults live in `app/config.py`; add only intentional
overrides. See `docs/RAILWAY_VARIABLES.md` for the complete policy.

## 3. Choose a process layout

For one service, keep `WORKER_ROLE=all`. It serves the UI/API and starts enabled
collectors, local-first enrichment, and (only when enabled) the automatic paper trader.

For a larger deployment, create services from the same image and PostgreSQL database:

- `WORKER_ROLE=web` for the public API/dashboard/Vision service;
- `WORKER_ROLE=collector` for public data collectors and lifecycle jobs;
- `WORKER_ROLE=enrichment` for bounded structured-news enrichment;
- `WORKER_ROLE=paper-trader` for the V2 paper loop.

Each role still starts FastAPI because that is the repository's current process entry
point. Expose only the web service publicly. Do not use a `research` role; it is not
valid and heavy research is local-only.

## 4. Start command and migrations

The Docker image runs:

```bash
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

Railway injects `PORT`. Startup runs SQLAlchemy `create_all()` and the idempotent
additive migration helper. Back up PostgreSQL before deploying a schema change; this
repository does not include Alembic.

## 5. Verify before enabling the paper loop

Open or query:

- `/health` — database, paper mode, and worker role;
- `/dashboard` — legacy operational dashboard;
- `/vision` — V2 read-only AI Vision;
- `/api/market/status` and `/api/news/status` — authenticated collector status;
- `/api/v2/registry` — authenticated model/champion/sandbox state;
- `/api/v2/risk/kill-switch` — effective paper-risk state.

Wait for a fresh closed candle, then run one authenticated
`POST /api/v2/pipeline/run` and inspect its trace in `/api/vision/replay/{trace_id}`.
Only after reviewing limits and records should you set
`AUTO_TRADER_ENABLED=true` on an `all` or `paper-trader` service.

## Optional local student or external AI

A compact JSON news student is Railway-safe:

```env
LOCAL_NEWS_STUDENT_PATH=./models/news_student.json
LOCAL_NEWS_STUDENT_VERSION=student-version
ENRICHMENT_ENABLED=true
EXTERNAL_AI_ENABLED=false
```

External providers are optional context only. If enabling one, set its key/model,
declared prices, daily limit, and monthly budget explicitly. Provider failure must
fall back to local intelligence and must not stop the paper loop.

## Safety

Anata exposes no live exchange order API, private exchange-key setting, withdrawal, or
real-money execution path. Do not add such credentials to Railway. Paper fills are a
limited deterministic simulation and are not evidence of achievable execution or
profitability.
