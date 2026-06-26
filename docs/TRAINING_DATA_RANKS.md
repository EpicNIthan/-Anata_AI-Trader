# Training Data Ranks

Database size is only a rough signal. A 321 MB Railway database can be useful, but model quality depends more on labeled training rows, days covered, market regimes, and clean paper-trade outcomes.

Use `/api/data/collection-report` and the dashboard Data / Storage tab to check the real training signal.

## What Counts Most

1. Closed 1m candles across all symbols.
2. Training features built from candles, news, sentiment, and external risk context.
3. Labels built from future closed candles, especially `target_trade_quality_score` and `target_direction_15m`.
4. Clean paper trades with open/close outcomes, fees, realized PnL, holding time, stop loss, and take profit.
5. Multiple market regimes: pump, dump, sideways, high volatility, low volatility, news shock.

## Practical Ranks

| Rank | Data Target | Expected Usefulness |
| --- | --- | --- |
| F | Less than 1 day, few labels | Not useful. Only tests that the pipeline runs. |
| D | 1-2 days, at least 500 labeled rows | Can train a toy model. Do not let it trade automatically. |
| C | 3-5 days, 5k-20k labeled rows, some paper trades | Useful for experiments. May match Bot sometimes, but not reliably smarter. |
| B | 7-10 days, 25k-75k labeled rows, clean bot paper trades | First serious model. Can test against Bot in paper mode. |
| A | 14-30 days, 100k-250k labeled rows, many clean paper outcomes | Can be useful if validation metrics beat Bot after fees and drawdown. |
| Pro | 45-90+ days, 500k+ labeled rows, multiple regimes, high-quality paper trade history | Strong candidate. Still needs walk-forward testing before trusting. |

## Rough Storage Expectations

These are approximate because PostgreSQL indexes, JSON payloads, raw news, and retained diagnostics can change size a lot.

| Collection Time | Railway DB Size | Dataset Export Size | Meaning |
| --- | --- | --- | --- |
| 1 day | 100-300 MB | 10-50 MB | Good pipeline test, weak model. |
| 3 days | 300-700 MB | 40-150 MB | Early training dataset. |
| 7-10 days | 700 MB-1.5 GB | 150-500 MB | First serious local training run. |
| 14-30 days | 1.5-5 GB | 500 MB-2 GB | Better model evaluation. |

## When To Use Trained AI

Only activate Trained AI for paper trading when all are true:

- Label coverage is not zero.
- Training dataset has enough rows for the rank you want.
- Validation metrics beat the Bot baseline.
- Test drawdown is acceptable.
- Paper mode confirms it avoids fee churn and dust trades.

Until then, keep Paper Runner in Bot mode to collect cleaner examples.
