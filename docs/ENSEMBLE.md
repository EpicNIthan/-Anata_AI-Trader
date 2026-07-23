# Regime-aware ensemble

`DeterministicRegimeEnsemble` is the current production baseline. It combines
standardized signals into an `EnsembleDecision`, but it never produces a broker/paper
order. A future learned meta-model must implement the same execution-independent
interface.

## Weighting policy

Before normalization, a signal's influence combines:

- calibrated confidence and inverse uncertainty;
- liquidity/fill-probability score and signal strength;
- health multiplier (`HEALTHY` 1.0, `WATCH` 0.75, `DEGRADED` 0.35);
- a bounded recent out-of-sample-performance adjustment;
- an incremental correlation penalty.

Signals in the same family are treated as correlated even before enough history exists.
At or above `V2_CORRELATION_PENALTY_THRESHOLD` (default `0.70`), their influence is
reduced. This prevents variations of the same technical information from receiving
independent full weight.

The combined return is then adjusted for correlation, expected transaction costs,
regime penalty, and a bounded optional external-context value. The external adjustment
is capped by `V2_EXTERNAL_CONTEXT_MAX_ADJUSTMENT`; a missing provider contributes zero,
not an invented opinion.

## Regime treatment

`risk_off`, `liquidity_stress`, and `news_shock` carry a 0.30 regime penalty;
`high_volatility` and `crowded_market` carry a 0.15 penalty. Other classifications are
currently unpenalized. The decision is actionable only when the absolute combined edge
meets `V2_MIN_NET_EDGE` and confidence is positive.

Every persisted ensemble records supporting and conflicting signal IDs, final weights,
exclusions, correlation/cost/regime penalties, external adjustment, status, and reason
codes. This is the evidence shown by Vision replay.

## Analyze independence locally

Use stored signal history before adding a candidate to a production family:

```powershell
python scripts/analyze_signal_correlation.py `
  --input local_data/signals.jsonl `
  --output research_reports/signal_independence.json `
  --threshold 0.80
```

Input rows need `signal_id`, timestamp, prediction, and actual return; optional
position/PnL/feature-family fields improve the analysis. The report measures prediction,
position, PnL, drawdown, trade overlap, and feature overlap rather than relying on one
correlation number.
