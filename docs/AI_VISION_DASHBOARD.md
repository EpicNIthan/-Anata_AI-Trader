# AI Vision dashboard

AI Vision is a read-only Jinja/vanilla-JavaScript dashboard over recorded market,
model, signal, portfolio, risk, paper-execution, research, and health data. It does not
submit trades, promote models, or generate a narrative with an LLM.

## Open the page

Run the FastAPI application:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then open:

```text
http://localhost:8000/vision
```

`/dashboard/vision` is a compatibility alias. When `ADMIN_TOKEN` is configured, open
`/vision?admin_token=YOUR_TOKEN` once; the route stores an HTTP-only cookie so browser
polling can authenticate. With dashboard username/password configured, use browser
Basic Auth. Do not place a production token in screenshots or shared links.

Railway uses the same path: `https://YOUR-APP.up.railway.app/vision`.

## Panels and sources

| View | Recorded source |
| --- | --- |
| Candles and live bar | `candles`, `live_candle_updates` |
| Prediction/model views | `model_predictions`, `trading_signals`, ensemble weights |
| Ensemble and disagreements | `ensemble_decisions` |
| Requested/approved exposure | `portfolio_targets`, `risk_decisions` |
| Orders, fills, trades, position/equity | V2 simulated records plus compatible legacy paper ledger |
| News and external AI | `structured_news_events`, `external_ai_requests`, legacy news rows |
| Research lifecycle | assignments, candidates, evaluations, health, promotions, sandboxes |
| Decision replay | `decision_timeline_events` and stage-linked V2 records |

The API labels legacy-only data as partial. It does not invent missing model,
ensemble, risk, slippage, funding, or attribution stages. Staleness is computed from
stored timestamps rather than the browser refresh time.

## Read-only API

All `/api/vision/*` endpoints use the same admin authentication as the dashboard:

```text
GET /api/vision/symbols
GET /api/vision/chart
GET /api/vision/overlays
GET /api/vision/state
GET /api/vision/models
GET /api/vision/history
GET /api/vision/decisions
GET /api/vision/replay/{trace_id}
GET /api/vision/research
```

Examples:

```powershell
$Headers = @{"x-admin-token"="YOUR_ADMIN_TOKEN"}

Invoke-RestMethod "http://localhost:8000/api/vision/chart?symbol=BTCUSDT&timeframe=1m&limit=250" `
  -Headers $Headers

Invoke-RestMethod "http://localhost:8000/api/vision/overlays?symbol=BTCUSDT&account_id=champion" `
  -Headers $Headers

Invoke-RestMethod "http://localhost:8000/api/vision/history?symbol=BTCUSDT&account_id=champion" `
  -Headers $Headers

Invoke-RestMethod "http://localhost:8000/api/vision/decisions?symbol=BTCUSDT&source=v2" `
  -Headers $Headers
```

`chart`, `overlays`, `history`, and `decisions` accept ISO-8601 `start` and `end`
where applicable. `chart` also accepts a validated timeframe such as `1m`; list
endpoints have bounded `limit` parameters. Use a registered sandbox account ID to
inspect its isolated records.

## Replay semantics

A V2 pipeline run persists a `decision_trace_id`. Replay returns only actual timeline
and linked records: model predictions, signals, ensemble, portfolio target, risk
decision, simulated order, and fill. If stage records exist but the timeline row is
missing, the response marks them `recorded_without_timeline` rather than claiming the
stage executed normally.

Legacy replay IDs use `legacy:DECISION_ID` and explicitly list the stages unavailable
in the old architecture.

## Refresh and bounds

```env
VISION_REFRESH_SECONDS=15
VISION_DEFAULT_LIMIT=250
```

The template clamps refresh to at least five seconds. APIs bound maximum rows/candles
server-side. Increasing the browser limit does not turn Vision into a bulk export
interface; use raw-data export for research history.

## Performance and attribution

The history endpoint calculates basic paper-ledger metrics and includes trace-based V2
attribution where records exist. It keeps an explicit unexplained residual and does
not fabricate counterfactual ensemble, sizing, or broad-market contributions. Legacy
opening fills may include entry fees in realized PnL, so Vision labels that limitation.

## Troubleshooting

- `401/403`: use the configured token or Basic Auth; confirm the browser received the
  auth cookie.
- Empty chart: start the market collector and verify `/api/market/status`; Vision does
  not synthesize candles.
- `stale=true`: inspect the latest candle time and collector error before running a
  V2 cycle. Risk should reject an exposure increase on stale data.
- Empty V2 panels: run the V2 pipeline or auto trader. Legacy history alone cannot
  reconstruct narrow-model stages.
- Empty research/health panels: register/evaluate candidates and run the monitoring
  endpoint after forecast horizons mature.

Verify endpoint schemas with:

```powershell
python -m pytest tests/test_v2_pipeline_contracts.py -q
```

See [MONITORING_AND_DECAY.md](MONITORING_AND_DECAY.md) for health/attribution caveats
and [OPERATIONS_RUNBOOK.md](OPERATIONS_RUNBOOK.md) for deployment checks.
