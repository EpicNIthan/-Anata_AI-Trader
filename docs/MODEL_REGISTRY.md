# Model registry

The registry extends `model_versions` rather than replacing the existing uploaded-model
history. It records artifact identity, schema compatibility, lifecycle, health, and
promotion lineage for individual narrow-model families.

## Lifecycle

```text
DRAFT -> TRAINED -> VALIDATING -> SHADOW -> PAPER_SANDBOX -> CHAMPION
                                                     |             |
                                                     +--> DEGRADED / SUSPENDED / RETIRED
```

The allowed values are `DRAFT`, `TRAINED`, `VALIDATING`, `SHADOW`, `PAPER_SANDBOX`,
`CHAMPION`, `DEGRADED`, `SUSPENDED`, and `RETIRED`. Health is separately recorded as
`HEALTHY`, `WATCH`, `DEGRADED`, `SUSPENDED`, or `RETIRED`.

The legacy `status` field remains for compatibility (`candidate`/`active`); V2 lifecycle
and health fields are the authoritative new metadata.

## Required registry evidence

For each row, retain at least:

- `model_id`, version, family, artifact origin path, checksum, feature schema and columns;
- preprocessing and training-dataset versions, training time range, horizon, metrics;
- package manifest, parent/challenger relationship, lifecycle and health;
- promotion history and any suspension/retirement reason.

`ArtifactValidator` computes a SHA-256 checksum and checks artifact loadability before
registration/promotion. Registration also writes the exact artifact bytes to
`model_artifact_blobs` in the same database transaction as `ModelVersion`. The path is
therefore an origin/audit hint, not a cross-service storage dependency. If another
Railway role cannot see that path, it verifies the stored BLOB against
`ModelVersion.artifact_checksum`, materializes a content-addressed runtime cache file,
and loads that verified copy.

For ZIP packages, runtime verification also requires every non-manifest member to be
covered by `checksum_manifest.json`; each declared SHA-256 and byte length is checked
before model parsing. A local-path checksum mismatch, a changed BLOB, an incomplete
manifest, or a changed package member fails closed. Current trusted local legacy
JSON/joblib/ZIP artifacts remain permitted through a compatibility path when their
feature contract is available; a full local package
should additionally include `feature_schema.json`, `model_metadata.json`,
`training_metrics.json`, `training_period.json`, `required_features.json`,
`optional_features.json`, and `missing_value_policy.json`.

## Register a challenger

The V2 operations API/script is the preferred route once deployed. Until then, this
direct local operation is the supported low-level interface; run it only against the
intended database and commit only after reviewing the result.

```powershell
@'
from app.db.session import SessionLocal
from app.pipeline.registry import ModelRegistry

with SessionLocal() as session:
    row = ModelRegistry(session).register(
        name="btc-short-momentum",
        model_id="btc-short-momentum",
        version="2026-07-23-candidate",
        model_family="alpha.short_horizon_momentum",
        path="models/candidate.json",
        feature_schema_version="price-news-market-v5",
        feature_columns=["candle_return_1m", "candle_return_5m", "trend_score"],
    )
    session.commit()
    print(row.id, row.lifecycle_state, row.artifact_checksum)
'@ | python -
```

Registration does not activate a model. A model should first be evaluated, shadowed, or
run in its isolated sandbox. See [CHAMPION_CHALLENGER.md](CHAMPION_CHALLENGER.md).

The authenticated `/api/models/upload` boundary is stricter than trusted local
registration: every non-directory member in an uploaded ZIP must be declarative JSON.
`.pkl`, `.pickle`, and `.joblib` members are rejected even when a JSON model is also
present, because deserializing remotely supplied pickle data can execute code. A
complete `checksum_manifest.json` is mandatory at this boundary. The
uploaded ZIP bytes—not an extracted executable file—are stored durably and enter only
the `TRAINED` candidate lifecycle. Upload never activates or promotes the model.

## What a champion means

Promotion creates a `champion_assignments` row scoped to a model family and symbol
(`*` is the default scope), closes the previous assignment, and writes a
`promotion_decisions` audit record. `V2_AUTO_PROMOTE_CHAMPION` is false by default;
automatic promotion remains disabled unless an operator explicitly changes policy.
