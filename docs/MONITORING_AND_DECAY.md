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
- observation count and explicit reason codes.

Health transitions use configured thresholds:

```env
MONITORING_ENABLED=true
MONITORING_OUTCOME_BATCH_SIZE=250
MONITORING_HEALTH_WINDOW=100
HEALTH_MIN_OBSERVATIONS=20
HEALTH_WATCH_CALIBRATION_ERROR=0.25
HEALTH_DEGRADED_CALIBRATION_ERROR=0.40
HEALTH_WATCH_MISSING_FEATURE_RATE=0.10
HEALTH_DEGRADED_MISSING_FEATURE_RATE=0.25
HEALTH_SUSPEND_CONSECUTIVE_ERRORS=5
```

Too little history is `WATCH`, not `HEALTHY`. Threshold breaches become `WATCH` or
`DEGRADED`. Repeated recorded model inference errors can suspend a model after the
configured count. A suspended or retired registry record is not automatically
reactivated by a later good window.

Snapshots are append-only evidence in `signal_health_snapshots` and
`model_health_snapshots`. The current status is also reflected on a matching registered
model unless it is already suspended/retired.

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
overlapping observations. With little overlap there is no estimated pairwise value;
the ensemble's same-family penalty still applies. This baseline does not yet calculate
position, drawdown, or feature-overlap correlation automatically.

## Paper-PnL attribution

Read bounded trace-based attribution:

```powershell
Invoke-RestMethod `
  "http://localhost:8000/api/v2/attribution?symbol=BTCUSDT&paper_account_id=champion" `
  -Headers @{"x-admin-token"="YOUR_ADMIN_TOKEN"}
```

The response includes total recorded paper PnL; fees, simulated slippage and funding;
and grouped values by model, signal, family, symbol, regime, and external-AI
availability where trace links exist. It assigns signal/model alpha in proportion to
recorded ensemble weights.

Counterfactual ensemble, position-sizing, and broad-market contributions are currently
left at zero and the difference is reported as `unexplained_residual`. This is
intentional: the system does not manufacture precision it cannot support. Legacy
trades and incomplete traces reduce attribution coverage.

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
- Current prediction/feature drift and OOD fields are persisted but the rolling
  monitor does not yet calculate population-stability or learned OOD scores.
- Funding attribution stays zero until the simulator records funding cash flows.
- Monitoring is symbol-local during pipeline refresh; sparse symbols may remain
  `WATCH` for a long time.
- No threshold proves profitability, and a few recent wins never cause promotion.

See [CHAMPION_CHALLENGER.md](CHAMPION_CHALLENGER.md) for lifecycle controls and
[PAPER_EXECUTION.md](PAPER_EXECUTION.md) for simulator limitations.
