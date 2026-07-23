# Backtesting methodology

The V2 evaluator is event/time ordered and point-in-time aware. It evaluates stored
predictions and returns; it is not a claim that any strategy is profitable.

## Required input discipline

- Use UTC decision timestamps in chronological order.
- Include `available_to_model_time`; it must not be after the decision timestamp.
- Include `label_available_time` for overlapping forward labels; it must not precede
  the decision timestamp.
- Keep train, validation, and test periods distinct. Do not random-shuffle rows.
- Include a per-observation transaction cost whenever a simulated position could trade.
- Record missing external-AI availability explicitly so with/without-context performance
  can be compared.

## Evaluator outputs

`evaluate_predictions()` calculates information coefficient/rank IC, directional hit
rate, gross/net expectancy, average win/loss, profit factor, total and annualized
return, Sharpe, Sortino, drawdown, Calmar, turnover, exposure, average holding time,
VaR/CVaR/tail loss, and cost totals. `evaluate_observations()` also segments metrics by
symbol, model family, regime, and external-AI availability.

Use net expectancy and risk-adjusted performance as decision inputs; win rate is only a
diagnostic.

## Historical evaluation

```powershell
python scripts/evaluate_historical.py `
  --input local_data/observations.jsonl `
  --candidate local_data/candidate.json `
  --dataset-version raw-2026-07-23 --feature-version price-news-market-v5 `
  --report research_reports/historical.json
```

## Walk-forward evaluation

`WalkForwardEvaluator` uses expanding training history by default; `--rolling` limits
the training window. It applies configured purge and embargo windows around validation
and test blocks, plus label-overlap protection.

```powershell
python scripts/run_walk_forward_evaluation.py `
  --input local_data/observations.jsonl `
  --candidate local_data/candidate.json `
  --report research_reports/walk_forward.json `
  --train-size 2000 --validation-size 250 --test-size 250 --step-size 250 `
  --purge-size 15 --embargo-size 15 --rolling
```

## Simulation limits

The research evaluator accepts recorded cost inputs, but it does not infer real market
microstructure. The current paper simulator applies configurable spread/slippage and an
optional 50% partial fill. Funding is stored as a field but defaults to zero. Treat
results as a controlled paper/research estimate, not execution evidence.

Before promotion, inspect regime slices, cost sensitivity, correlation with existing
signals, data coverage changes, and the difference between shadow/sandbox and historical
results.
