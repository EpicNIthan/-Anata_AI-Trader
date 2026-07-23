# Feature registry

The current feature schema is `price-news-market-v5`, defined in
`app/features/schema.py`. `FeatureBuilder` writes the complete vector in
`Feature.payload["values"]` and retains compact top-level fields for backwards
compatibility. Schema versions are explicit so an old model can keep selecting the
columns it was trained on.

## Feature families

| Family | Examples | Primary source |
| --- | --- | --- |
| Price and activity | `price_change`, `candle_return_1m`, `volatility`, `volume_change`, `trend_score` | closed candles |
| Technical | RSI, MACD, SMA/EMA distance, Bollinger metrics, ATR, VWAP, ADX | closed candles |
| Time context | UTC hour/day sin/cos, weekend, Asia/London/New York sessions | candle close time |
| News | sentiment, confidence, risk, impact, recency, BTC/ETH/macro relevance | articles and sentiment |
| Derivatives | crowd ratios, taker pressure, OI, funding, crowd risk | public futures events |
| Spot/context | bid/ask spread, order-book imbalance, activity | public spot context |
| Broad market/risk | fear/greed, dominance, liquidations, stablecoin, macro and security risk | external context |
| Regime | trend, direction, volatility, news shock, risk-off, liquidity stress, breakout, mean-reversion, crowd pressure | deterministic derived values |

Use code rather than a hand-maintained column list when inspecting the registry:

```powershell
python -c "from app.features.schema import CURRENT_FEATURE_SCHEMA_VERSION, columns_for_schema; print(CURRENT_FEATURE_SCHEMA_VERSION); print('`n'.join(columns_for_schema(CURRENT_FEATURE_SCHEMA_VERSION)))"
```

## Contract at the model boundary

`FeatureSnapshot` contains:

- `feature_snapshot_id`, symbol, schema version, values, and data version;
- `as_of` and `available_to_model_time` as timezone-aware UTC timestamps;
- source freshness and missing-required-feature lists;
- explicit external-AI availability, missingness, failure, provider, prompt version,
  confidence, and age metadata.

Each narrow model declares `required_features`, `optional_features`, and a forecast
horizon. Missing required fields lower model confidence and are passed to risk, which
rejects exposure-increasing targets with missing required features.

## Compatibility rules

- Never reorder a model's recorded `feature_columns` without retraining it.
- Add a new schema version when feature meaning, scale, preprocessing, or missing-value
  policy changes.
- Preserve previous schemas in `FEATURE_COLUMNS_BY_SCHEMA`; they are needed to load
  legacy artifacts safely.
- Store preprocessing/missing-value policy with an artifact, not only in code.
- Do not let a default `0` hide data absence in a claim about a feature's predictive
  value; use freshness and missingness metadata in evaluation.

Fetch a latest feature for inspection (it is protected when auth is configured):

```powershell
Invoke-RestMethod "http://localhost:8000/api/features/latest?symbol=BTCUSDT" `
  -Headers @{"x-admin-token"="YOUR_ADMIN_TOKEN"}
```

This endpoint can build an in-memory feature only when no stored row is present. It is
useful for diagnosis, not evidence that the resulting row belongs in a historical test.
