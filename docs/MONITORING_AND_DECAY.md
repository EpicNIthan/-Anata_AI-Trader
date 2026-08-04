# Monitoring, attribution, and decay management

V2 monitoring labels matured paper forecasts from later closed candles, calculates
bounded rolling health evidence, feeds recent health/correlation evidence back into
the deterministic ensemble, and exposes trace-based paper-PnL attribution. It does
not claim that a strategy is profitable or silently promote a replacement.

## Outcome labeling

For each signal without an outcome, `RollingHealthMonitor` waits until its registered
forecast horizon has elapsed. It then finds:

- the latest closed candle at or before signal generation; and
- the first closed candle at or after the due time, but no later than now.

It records realized return, the signal's estimated cost, signed net return,
directional hit, horizon, observation time, and source candle IDs in
`signal_outcomes`. Missing endpoints remain unlabeled; no future candle is pulled
backward to make a result available early.

Outcome work is bounded by `MONITORING_OUTCOME_BATCH_SIZE`. Each V2 pipeline run
performs a bounded refresh for its symbol before using monitoring evidence. Operators
can also trigger it explicitly after horizons mature:

```powershell
$Headers = @{"x-admin-token"="YOUR_ADMIN_TOKEN"}
Invoke-RestMethod -Method Post `
  "http://localhost:8000/api/v2/monitoring/run?symbol=BTCUSDT" `
  -Headers $Headers
```

## Rolling health

The most recent bounded observations per signal family produce:

- Pearson information coefficient between stored net forecasts and realized net
  returns;
- mean net expectancy;
- mean absolute calibration error, `|confidence - directional_hit|`;
- rate of predictions that reported missing features;
- two-sample KS drift for recent versus preceding expected-return distributions;
- top-decile feature KS drift and a reference-z-score OOD row rate;
- paired live/shadow forecast divergence and recent-versus-preceding realized-return
  divergence;
- positive relative transaction-cost increase;
- increase in maximum absolute cross-family outcome correlation;
- regime dependence using eta-squared of net returns by recorded regime; and
- decline in a bounded executable-capacity proxy based on liquidity and observed
  cost;
- observation count and explicit reason codes.

The current window contains the newest `MONITORING_HEALTH_WINDOW` matured outcomes.
The immediately preceding window is the reference population. Drift metrics are null
with an explicit evidence reason until at least
`HEALTH_MIN_REFERENCE_OBSERVATIONS` comparable reference rows exist; the monitor does
not turn missing evidence into a healthy zero.

Health transitions use configured thresholds:

```env
MONITORING_ENABLED=true
MONITORING_OUTCOME_BATCH_SIZE=250
MONITORING_HEALTH_WINDOW=100
HEALTH_MIN_OBSERVATIONS=20
HEALTH_MIN_REFERENCE_OBSERVATIONS=10
HEALTH_WATCH_MIN_INFORMATION_COEFFICIENT=0
HEALTH_DEGRADED_MIN_INFORMATION_COEFFICIENT=-0.10
HEALTH_WATCH_MIN_NET_EXPECTANCY=0
HEALTH_DEGRADED_MIN_NET_EXPECTANCY=-0.001
HEALTH_WATCH_CALIBRATION_ERROR=0.25
HEALTH_DEGRADED_CALIBRATION_ERROR=0.40
HEALTH_WATCH_MISSING_FEATURE_RATE=0.10
HEALTH_DEGRADED_MISSING_FEATURE_RATE=0.25
HEALTH_WATCH_PREDICTION_DRIFT=0.25
HEALTH_DEGRADED_PREDICTION_DRIFT=0.50
HEALTH_WATCH_FEATURE_DRIFT=0.25
HEALTH_DEGRADED_FEATURE_DRIFT=0.50
HEALTH_WATCH_OOD_RATE=0.10
HEALTH_DEGRADED_OOD_RATE=0.25
HEALTH_OOD_ZSCORE_THRESHOLD=4
HEALTH_OOD_FEATURE_FRACTION=0.10
HEALTH_WATCH_LIVE_SHADOW_DIVERGENCE=0.30
HEALTH_DEGRADED_LIVE_SHADOW_DIVERGENCE=0.60
HEALTH_WATCH_TRANSACTION_COST_INCREASE=0.25
HEALTH_DEGRADED_TRANSACTION_COST_INCREASE=0.75
HEALTH_WATCH_CORRELATION_INCREASE=0.15
HEALTH_DEGRADED_CORRELATION_INCREASE=0.30
HEALTH_WATCH_REGIME_DEPENDENCE=0.70
HEALTH_DEGRADED_REGIME_DEPENDENCE=0.90
HEALTH_WATCH_CAPACITY_DECLINE=0.15
HEALTH_DEGRADED_CAPACITY_DECLINE=0.35
HEALTH_SUSPEND_CONSECUTIVE_ERRORS=5
```

Too little history is `WATCH`, not `HEALTHY`. Threshold breaches become `WATCH` or
`DEGRADED`. `HEALTHY`, `WATCH`, and `DEGRADED` persist recommended weight multipliers
of `1.0`, `0.75`, and `0.35`, matching the deterministic ensemble. Repeated recorded
model inference errors suspend a model after the configured count and persist a zero
weight plus `BLOCK_NEW_EXPOSURE`. A suspended or retired model or signal-family health
state is sticky: a later good window or successful inference resets evidence counters
but never reactivates terminal health without an explicit operator policy.

Snapshots are append-only evidence in `signal_health_snapshots` and
`model_health_snapshots`. Every tracked metric, consecutive-error count, recommended
weight multiplier, and recommended action has an explicit database column; method and
coverage details remain in `payload.metric_evidence`. The current status is also
reflected on a matching registered model unless it is already suspended/retired.

## Use in the ensemble and risk path

The V2 pipeline reads the latest family health, recent bounded net performance, and
pairwise outcome correlations. The ensemble:

- reduces `WATCH` and `DEGRADED` weights;
- excludes suspended/retired evidence;
- bounds recent-performance influence to ±20%;
- reduces highly correlated signals and treats duplicate family evidence as
  correlated even before sufficient history exists.

Risk independently rejects suspended/retired model or signal health supplied to it.
Monitoring cannot relax a risk limit, create exposure, or promote a challenger.

The correlation estimator aligns family outcomes at minute resolution and requires
overlapping observations in both the recent and preceding windows. With little overlap
there is no estimated increase; the ensemble's same-family penalty still applies.

## Paper-PnL attribution

Read bounded trace-based attribution:

```powershell
Invoke-RestMethod `
  "http://localhost:8000/api/v2/attribution?symbol=BTCUSDT&paper_account_id=champion&time_period=day&limit=2000" `
  -Headers @{"x-admin-token"="YOUR_ADMIN_TOKEN"}
```

The endpoint also accepts ISO-8601 `start` and `end` filters. `time_period` is bounded
to `hour`, `day`, `week`, or `month`, and `limit` cannot exceed 10,000 rows. AI Vision
uses the same reader with its smaller event limit.

The response includes every selected PnL event, its available trace IDs, an additive
decomposition, and grouped values by model, signal, family, exact ensemble, portfolio
sizing bucket, risk-resize category, symbol, regime, external-AI availability/provider,
and UTC time period. Signal/model/provider alpha is allocated only by persisted
ensemble weights. `coverage` reports both trade-count and absolute-PnL coverage for
each lineage stage.

Read `component_metadata` before interpreting numeric `components`. Counterfactual
ensemble, position-sizing, broad-market, and general execution values stay null unless
explicitly recorded; their numeric reconciliation value is zero and the difference is
reported as `unexplained_residual`. Legacy trades remain in `trades` with explicit
missing-evidence labels. New simulator fills record and book a signed funding cash
flow; legacy funding fields without that signed evidence remain separate unbooked
estimates.

AI Vision exposes the same attribution and research/health rows through
`/api/vision/history` and `/api/vision/research`.

## Operator response to decay

1. Confirm the data feed and point-in-time feature availability before blaming a
   model.
2. Inspect observation count, calibration error, missing-feature rate, IC, net
   expectancy, correlation changes, and regime coverage.
3. Use the kill switch for an incident that should block all new exposure.
4. Suspend/degrade a model explicitly if evidence warrants it; do not edit artifacts
   in place.
5. Train a new version locally, then use shadow or an isolated sandbox.
6. Promote only after manual review; keep the prior champion available for rollback.

## Limits and interpretation

- Health windows are empirical paper evidence, not statistical guarantees.
- KS and z-score diagnostics are deterministic lightweight baselines, not learned
  multivariate drift or OOD models. Stronger offline detectors can replace them while
  preserving the persisted contract.
- Capacity is a relative liquidity/cost proxy because the paper ledger does not claim
  exchange-level market capacity.
- New simulator fills book a simplified non-negative funding cost per fill. Legacy
  records without `funding_cash_flow` remain separate unbooked estimates; this is not
  directional or periodic venue settlement.
- Monitoring is symbol-local during pipeline refresh; sparse symbols may remain
  `WATCH` for a long time.
- No threshold proves profitability, and a few recent wins never cause promotion.

See [CHAMPION_CHALLENGER.md](CHAMPION_CHALLENGER.md) for lifecycle controls and
[PAPER_EXECUTION.md](PAPER_EXECUTION.md) for simulator limitations.
