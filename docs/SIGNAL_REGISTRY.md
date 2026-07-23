# Signal registry

A trading signal is registered evidence derived from one model prediction. It is not an
order and cannot set margin, leverage, quantity, stop loss, or execution timing.

## Signal contract

`TradingSignal` and `trading_signals` record:

- predecessor `prediction_id`, signal family, symbol, generation time, expiry;
- long/short/flat direction, strength, expected return/cost/net return;
- confidence, uncertainty, regime, liquidity, health, lifecycle, and reason codes;
- model/calibration/volatility and external-context metadata.

Lifecycle values are `RESEARCH`, `VALIDATION`, `SHADOW`, `PAPER`, `LIMITED`,
`PRODUCTION`, `REDUCED`, `SUSPENDED`, and `RETIRED`. Health values match the model
registry's health vocabulary.

## Admission rules

`SignalFactory` subtracts the cost estimate, emits `FLAT` below the configured net-edge
threshold, and bounds TTL by the predecessor prediction expiry. The ensemble excludes:

- expired, flat, symbol-mismatched signals;
- `SHADOW`, `SUSPENDED`, or `RETIRED` lifecycle signals;
- `SUSPENDED` or `RETIRED` health signals.

Shadow predictions are written to `shadow_predictions`; they do not create a
portfolio target. Signal outcomes can be recorded later in `signal_outcomes` for health
and research analysis.

## Inspect recorded signal evidence

```powershell
$Token = "YOUR_ADMIN_TOKEN"
Invoke-RestMethod "http://localhost:8000/api/vision/models?symbol=BTCUSDT" `
  -Headers @{"x-admin-token"=$Token}
Invoke-RestMethod "http://localhost:8000/api/vision/overlays?symbol=BTCUSDT" `
  -Headers @{"x-admin-token"=$Token}
```

The first endpoint shows model/signal evidence and current ensemble weights; the second
returns bounded chart overlays. Empty results mean evidence was not persisted yet, not
that a signal was silently neutral.
