# Portfolio construction and independent risk

The V2 ensemble estimates opportunity; it does not emit an order. Portfolio
construction converts that estimate into a signed target exposure, and an independent
risk engine either rejects or resizes the target before paper execution can consume it.

```text
EnsembleDecision
  -> PortfolioTarget (requested signed exposure / equity)
  -> RiskDecision (approved exposure and leverage)
  -> persisted risk_decisions row
  -> PaperExecutionSimulator
```

No model prediction contains leverage, margin, notional, or order instructions.

## Deterministic portfolio baseline

`DeterministicPortfolioConstructor` uses current signed exposures, paper equity,
expected return, confidence, uncertainty, expected volatility, liquidity, and current
gross/net exposure. It scales the requested target and clips it against:

- per-symbol exposure;
- gross portfolio exposure;
- net portfolio exposure;
- configured correlated-cluster exposure;
- minimum liquidity and minimum net edge.

The output is a `PortfolioTarget`, including current exposure, requested target and
delta, expected risk, risk contribution, urgency, and the source ensemble decision.
Non-actionable ensembles or nonpositive equity target zero exposure.

This is a deterministic fallback, not mean-variance optimization. The runtime V2
service currently does not estimate a covariance matrix. Unless a caller supplies a
`cluster_by_symbol` mapping, each symbol behaves as its own cluster, so the cluster
cap is not yet a portfolio-wide crypto-beta constraint.

## Risk policy

`PortfolioRiskEngine` applies the same global gates to every exposure increase,
including the champion account and a registered sandbox:

- configuration or persisted kill switch;
- minimum ensemble confidence;
- positive, non-future, sufficiently fresh source-candle price;
- no missing required features;
- model and signal health not suspended/retired;
- minimum liquidity;
- maximum daily realized loss;
- maximum recorded portfolio drawdown;
- cooldown after a sufficiently large recent loss;
- maximum number of open positions;
- symbol, gross, net, margin-allocation, and sandbox exposure caps;
- expected transaction-cost ceiling;
- positive paper cash/equity and minimum paper notional;
- entry-fee exposure cap.

Limits that can be satisfied by reducing size are recorded in `triggered_limits` and
resize the exposure. Hard failures are recorded in `rejection_reasons`; execution then
receives no approved increase. Risk-reducing moves use the protective path and are not
blocked by entry gates such as the kill switch.

The V2 policy chooses leverage from the minimum of the V2 position limit, portfolio
limit, and paper-engine maximum. Model-requested leverage and margin are not inputs.
Every decision records `RISK_CONFIGURATION_VERSION`, requested/approved exposure and
leverage, kill-switch state, market/equity evidence, account, target, and trace IDs.

## Sandbox isolation

A sandbox is a persisted `paper_sandbox_accounts` row with a unique account ID,
starting balance, and maximum exposure. Risk looks up this row independently and
tightens symbol, gross, net, and margin caps to
`V2_SANDBOX_MAX_EXPOSURE_PCT`. Supplying an arbitrary account name to the V2 API is
rejected; it must be the champion account or an active registered sandbox.

Sandbox admission is technical rather than profitability-based. It does not waive the
kill switch, stale-data gate, fee cap, or any other system-integrity control.

## Configuration

Safe paper defaults are included in `.env.example`. The main V2 controls are:

```env
TRADING_MODE=paper
V2_MAX_POSITION_LEVERAGE=3
V2_MAX_SYMBOL_EXPOSURE_PCT=0.10
V2_MAX_GROSS_EXPOSURE_PCT=0.40
V2_MAX_NET_EXPOSURE_PCT=0.25
V2_MAX_CLUSTER_EXPOSURE_PCT=0.25
V2_SANDBOX_MAX_EXPOSURE_PCT=0.03
RISK_MAX_TRADE_SIZE_PCT=0.10
RISK_MAX_DAILY_LOSS_PCT=0.05
RISK_MAX_PORTFOLIO_DRAWDOWN_PCT=0.15
RISK_MAX_OPEN_POSITIONS=3
RISK_MAX_MARKET_DATA_AGE_SECONDS=180
RISK_MAX_EXPECTED_TRANSACTION_COST_PCT=0.003
RISK_MAX_ENTRY_FEE_PCT_OF_EQUITY=0.01
RISK_MAX_FEE_EXPOSURE_PCT=0.01
RISK_KILL_SWITCH_ENABLED=false
RISK_CONFIGURATION_VERSION=v2-safe-defaults
```

Changing a limit changes future decisions only; it does not rewrite recorded history.
Keep `V2_AUTO_PROMOTE_CHAMPION=false` and `RESEARCH_AUTO_PROMOTE=false`.

## Kill-switch operations

Read the effective state:

```powershell
$Headers = @{"x-admin-token"="YOUR_ADMIN_TOKEN"}
Invoke-RestMethod http://localhost:8000/api/v2/risk/kill-switch -Headers $Headers
```

Enable it with an auditable append-only state change:

```powershell
$Headers = @{"x-admin-token"="YOUR_ADMIN_TOKEN";"Content-Type"="application/json"}
Invoke-RestMethod -Method Put http://localhost:8000/api/v2/risk/kill-switch `
  -Headers $Headers `
  -Body '{"enabled":true,"reason":"operator incident response","confirm":true}'
```

A configuration-level `RISK_KILL_SWITCH_ENABLED=true` cannot be disabled through the
API. Protective reductions remain available.

## Known limits

- `RISK_MAX_SPREAD_PCT` and `RISK_REQUIRE_FRESH_DATA` are configuration fields, but
  the V2 risk implementation currently gates the cost estimate and source-candle age
  directly; it does not independently consume a measured bid/ask spread or the
  freshness toggle.
- Portfolio cluster maps and covariance estimates are not populated by the default
  pipeline.
- Health gates depend on recorded monitoring snapshots and are conservative baseline
  controls, not proof that a model is sound.
- Paper equity and drawdown are simulator records, not broker reconciliations.

See [PAPER_EXECUTION.md](PAPER_EXECUTION.md) for how an approval is consumed and
[MONITORING_AND_DECAY.md](MONITORING_AND_DECAY.md) for health inputs.
