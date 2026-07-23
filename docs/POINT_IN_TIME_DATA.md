# Point-in-time data policy

Historical research may use only information that was available to the model at the
decision timestamp. A correct event time alone is not enough: delayed news, revisions,
and post-processing can otherwise leak future information.

## Time fields

| Field | Meaning | Safe use |
| --- | --- | --- |
| `event_time` / `published_at` | When the underlying event reportedly happened | Descriptive only until availability is known |
| `received_time` | When Anata received the record | Candidate availability evidence |
| `processed_time` | When Anata completed processing | Audits pipeline delay |
| `available_to_model_time` | Earliest time the model could consume the row | Required research availability bound |
| feature `as_of` | Feature observation time | Must not imply that late source rows were known |
| `label_available_time` | When a supervised target is mature | Must be after the decision time |

`PointInTimeValidator` creates a `FeatureSnapshot` from a stored feature and carries
`available_to_model_time`, source freshness, missing-required features, schema, and
external-context missingness into the model boundary.

## Enforced checks

- Candle validation rejects invalid OHLC, negative volume, duplicate timestamps, and
  timestamps materially ahead of UTC now; gaps are reported as warnings.
- `validate_news_available()` rejects a news row whose availability is after the
  decision time.
- Research observations require timezone-aware UTC values and reject a label whose
  availability precedes the decision timestamp.
- Evaluation sorts chronologically; it does not random-shuffle time series.
- V2 risk rejects stale/future market data and missing required features for exposure
  increases.

## Safe historical row shape

Use an explicit availability timestamp when producing input for
`scripts/evaluate_historical.py` or walk-forward evaluation:

```json
{"timestamp":"2026-07-01T00:00:00+00:00","available_to_model_time":"2026-07-01T00:00:00+00:00","label_available_time":"2026-07-01T00:05:00+00:00","symbol":"BTCUSDT","prediction":0.0012,"actual_return":0.0008,"transaction_cost":0.0004}
```

The evaluator accepts `actual_return`/`realized_return`/`target_return`, but it does
not make a delayed label available early. Use a nonzero purge and embargo whenever the
forecast labels overlap.

```powershell
python scripts/run_walk_forward_evaluation.py `
  --input local_data/observations.jsonl `
  --report research_reports/walk_forward.json `
  --train-size 2000 --validation-size 250 --test-size 250 --step-size 250 `
  --purge-size 15 --embargo-size 15
```

## Operating guidance

1. Store raw source payloads and availability times before feature aggregation.
2. Treat missing external AI as a feature state, never as a neutral external answer.
3. Keep a feature-schema and data-version value with each artifact and experiment.
4. Do not backfill a historical feature with a later-received article or revised
   external event unless its original availability is demonstrably before the decision.
5. Reject or explicitly report rows with unknown availability rather than silently
   treating `created_at` as a safe substitute.

The current feature builder records availability and freshness metadata. Older legacy
rows can remain readable, but a legacy row without sufficient timing evidence should
be excluded from point-in-time claims.
