# Paper execution simulator

Anata has no live broker or exchange execution adapter. V2 execution writes simulated
orders and fills, then uses the existing paper ledger for fake cash, positions, fees,
and PnL. Exchange API keys, withdrawals, and live-money order calls are unavailable.

## Authorization boundary

`PaperExecutionSimulator` is the only V2 component that imports `PaperEngine`. It
requires both the typed in-memory `RiskDecision` and its persisted, approved
`risk_decisions` row. Before submitting, it verifies that the approval is bound to the
same:

- portfolio target and decision trace;
- paper account and symbol;
- requested/approved exposure and leverage;
- persisted current/target exposure;
- paper equity snapshot;
- unexpired V2 signal lifetime.

Any mismatch fails closed. A unique client order ID and risk-decision lookup make each
approval single-use, so replaying an approved request cannot create another order.

### Compatibility entry points

The retained `/api/paper-trade`, `/api/strategy/{symbol}/paper`, `/api/signal`, and
V2-disabled auto-trader paths all enter `PaperEngine` through the same legacy
compatibility boundary. Before an exposure-increasing fill, that boundary independently
checks confidence, the database/configuration kill switch, market freshness, daily
loss, portfolio drawdown, recent-loss cooldown, open-position count, leverage, margin,
cash, fee exposure, and position sizing. Leverage, margin, notional, stops, and targets
from an API or model plan remain untrusted request data and cannot suppress those gates.
Protective reductions remain available while the kill switch is active.

Every compatibility evaluation, including a rejection or hold, is committed before
execution as a clearly labelled compatibility `ensemble_decisions` placeholder,
`portfolio_targets` row, and `risk_decisions` row. The risk payload records the fixed
entry-point source, account, symbol, requested values, evaluated cash/equity/market
state, triggered limits, rejection reasons, and configuration version
`legacy-compatibility-universal-risk-v1`. An approved fill stores that risk ID and trace
ID on `paper_trades`; the same IDs are also returned as additive response fields. If
the audit chain cannot be persisted, the session is rolled back and the request is
rejected before a fill.

This compatibility approval is deliberately scoped to one synchronous
`PaperEngine.execute_signal()` call. It does not create a `simulated_orders` row and
cannot be replayed through the stricter V2 order-consumption path. New V2 submissions
continue to require their persisted V2 target, approval, and simulated order.

## State and records

The typed order contract supports:

```text
CREATED -> RISK_APPROVED -> SUBMITTED -> ACKNOWLEDGED
  -> FILLED | PARTIALLY_FILLED | CANCEL_PENDING | EXPIRED | REJECTED | ERROR
CANCEL_PENDING -> CANCELLED | PARTIALLY_FILLED | FILLED | EXPIRED | ERROR
```

The default V2 pipeline exercises the create/approve/submit/acknowledge and
market-fill/reject paths. The simulator also implements programmatic resting limit
orders, cancel/replace, restart recovery/expiry, and account reconciliation. It
persists:

- `simulated_orders`: order, target, risk, trace and account IDs, requested quantity
  and notional, state, client ID, timestamps, and paper-only metadata;
- `simulated_fills`: fill/order/trace/account IDs, quantity, price, notional, fee,
  measured slippage, funding field, and fill time;
- legacy-compatible `paper_trades`, `positions`, and `account_equity` records.

Crossing from long to short or short to long is decomposed: the current side is closed
in one cycle, and a later independently approved cycle may open the other side.

## Fill and cost assumptions

For a market-style fill, the simulator uses supplied bid/ask or derives a book from the
configured spread, then adds deterministic slippage and optional square-root market
impact. `PaperEngine` applies the configured paper fee and approved leverage and
updates the fake account. When partial fills are enabled, the baseline cap is 50% of
requested quantity/notional. If a market snapshot supplies volume, the fill is also
capped by configured participation. A funding scenario cost is recorded on the fill
for attribution. New V2 fills also persist a signed `funding_cash_flow` and immediately
book that simplified cost into fake cash and realized PnL. Legacy fills without the
signed field remain explicitly labeled as unbooked estimates.

```env
TRADING_MODE=paper
PAPER_FEE_RATE=0.0004
PAPER_SIMULATED_SPREAD_PCT=0.0002
PAPER_SIMULATED_SLIPPAGE_PCT=0.0001
PAPER_SIMULATED_PARTIAL_FILL_ENABLED=false
PAPER_SIMULATED_VOLUME_PARTICIPATION=0.10
PAPER_SIMULATED_MARKET_IMPACT_COEFFICIENT=0
PAPER_SIMULATED_FUNDING_RATE=0
PAPER_SIMULATED_LATENCY_MS=0
PAPER_SIMULATED_ORDER_TTL_SECONDS=300
```

All values are deterministic. They are scenario assumptions, not a calibrated claim
about fill quality on any venue.

## Run and inspect one V2 cycle

Start the app, ensure recent closed candles exist, then call the authenticated paper
endpoint:

```powershell
$Headers = @{"x-admin-token"="YOUR_ADMIN_TOKEN";"Content-Type"="application/json"}
Invoke-RestMethod -Method Post http://localhost:8000/api/v2/pipeline/run `
  -Headers $Headers -Body '{"symbol":"BTCUSDT"}'
```

Inspect the exact recorded trace through AI Vision:

```powershell
Invoke-RestMethod http://localhost:8000/api/vision/decisions?symbol=BTCUSDT `
  -Headers @{"x-admin-token"="YOUR_ADMIN_TOKEN"}

Invoke-RestMethod http://localhost:8000/api/vision/replay/TRACE_ID `
  -Headers @{"x-admin-token"="YOUR_ADMIN_TOKEN"}
```

An approved target may still result in `HELD` when it requests no exposure change.
A stale source candle is normally rejected by risk before execution.

## Restart behavior and reconciliation

Order, fill, trade, position, and equity rows survive a process restart because they
are stored in the database. Duplicate consumption checks therefore also survive a
restart. `recover_open_orders()` expires stale orders, checks cancelled approvals and
recorded fills, and marks recovered evidence. `reconcile_account()` compares open
positions, marks, equity, and the latest account snapshot. `cancel_order()` and
`replace_limit_order()` enforce terminal-state and remaining-quantity constraints.

These recovery/cancel/replace/reconciliation methods are currently service APIs used
by tests; there is no authenticated REST or scheduled startup wrapper for operators.

## Simulation limitations

Latency is recorded as metadata rather than slept in wall-clock time. The default V2
runtime does not currently populate bid/ask, available volume, or observed funding in
its `MarketSnapshot`, so configured fallbacks are used and participation/impact cannot
be calibrated from the default candle-only path. There is no order-book depth, queue
position, stochastic impact, venue outage, time-varying spread process, scheduled
resting-order event loop, or delisting engine. `process_resting_orders()` can consume
later market snapshots and complete partial fills, but an operator/worker must invoke
it; it is not a venue feed. Funding is a non-negative per-fill scenario cost rather
than directional, periodic exchange settlement. For these reasons the simulator is
useful for architecture, accounting, and deterministic risk tests, but it is not a
full market simulator and does not establish profitability.

See [PORTFOLIO_AND_RISK.md](PORTFOLIO_AND_RISK.md) for approval controls and
[AI_VISION_DASHBOARD.md](AI_VISION_DASHBOARD.md) for trace replay.
