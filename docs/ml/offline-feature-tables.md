# BigQuery Offline Feature Tables

Version: 1.0  
Date: 2025-08-12

## Dataset
- Name: `market_regime_{env}` (e.g., `market_regime_dev`)
- Location: US
- Partitioning: `DATE(ts_minute)` or `date` (DATE)
- Clustering: `symbol`, `dte`

## Tables
- `c1_features`, `c2_features`, ..., `c8_features`
- `training_dataset` (joined/denormalized for ML)

### Common Columns
- symbol STRING
- ts_minute TIMESTAMP (UTC, truncated to minute)
- date DATE (extracted from ts_minute)
- dte INT64
- zone_name STRING

### Example: c2_features (partial)
```
CREATE TABLE `project.market_regime_{env}.c2_features` (
  symbol STRING,
  ts_minute TIMESTAMP,
  date DATE,
  dte INT64,
  zone_name STRING,
  c2_gamma_exposure FLOAT64,
  c2_sentiment_level INT64,
  c2_pin_risk_score FLOAT64,
  c2_vanna FLOAT64,
  c2_charm FLOAT64,
  c2_volga FLOAT64
)
PARTITION BY date
CLUSTER BY symbol, dte;
```

### training_dataset View/Table
- Join across `c{n}_features` on (symbol, ts_minute, dte)
- Label columns (if supervised): `regime_label`, `transition_label`, `gap_direction_label` (for model training in Epic 3)

### Data Population
- Source: GCS Parquet → Arrow/RAPIDS transform → BigQuery load (CustomJob)
- Validation: row counts, null checks, schema compliance; results logged in `mr_load_audit`

## Governance
- Retention: 90 days (table expiration), longer for training snapshots
- Access: read for ML, write via service account used by CustomJobs/Pipelines



