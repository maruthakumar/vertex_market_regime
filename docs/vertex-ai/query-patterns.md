# BigQuery Query Patterns for Market Regime Feature Store

Version: 1.1  
Date: 2025-08-13
Last Updated: Story 2.2 Implementation

## Overview
This document provides optimized query patterns for accessing features from BigQuery tables and Vertex AI Feature Store.

## Table Structure
- **Dataset**: `market_regime_{env}` where env = dev/staging/prod
- **Tables**: `c1_features` through `c8_features`, `training_dataset`
- **Partitioning**: By `date` (DATE type)
- **Clustering**: By `symbol`, `dte`

## Common Query Patterns

### 1. Point-in-Time Feature Retrieval
Retrieve features for a specific point in time without data leakage:

```sql
-- Get latest features for a specific symbol and DTE
SELECT *
FROM `arched-bot-269016.market_regime_{env}.training_dataset`
WHERE symbol = @symbol
  AND dte = @dte
  AND ts_minute <= @request_timestamp
  AND ts_minute >= TIMESTAMP_SUB(@request_timestamp, INTERVAL 60 MINUTE)
ORDER BY ts_minute DESC
LIMIT 1;
```

### 2. Online Feature Subset
Retrieve only the 32 online features for Feature Store ingestion:

```sql
SELECT 
  -- Entity ID construction
  CONCAT(symbol, '_', FORMAT_TIMESTAMP('%Y%m%d%H%M', ts_minute), '_', CAST(dte AS STRING)) as entity_id,
  ts_minute as feature_timestamp,
  
  -- Online features only
  c1_momentum_score, c1_vol_compression, c1_breakout_probability,
  c2_gamma_exposure, c2_sentiment_level, c2_pin_risk_score,
  c3_institutional_flow_score, c3_divergence_type, c3_range_expansion_score,
  c4_skew_bias_score, c4_term_structure_signal, c4_iv_regime_level,
  c5_momentum_score, c5_volatility_regime_score, c5_confluence_score,
  c6_correlation_agreement_score, c6_breakdown_alert, c6_system_stability_score,
  c7_level_strength_score, c7_breakout_probability,
  c8_component_agreement_score, c8_integration_confidence, c8_transition_probability_hint,
  zone_name, symbol, dte
  
FROM `arched-bot-269016.market_regime_{env}.training_dataset`
WHERE date = CURRENT_DATE()  -- Use partition
  AND symbol IN ('NIFTY', 'BANKNIFTY')  -- Use clustering
  AND dte IN (0, 3, 7, 14)  -- Use clustering
```

### 3. Training Data Window
Retrieve training data for a specific time window:

```sql
SELECT *
FROM `arched-bot-269016.market_regime_{env}.training_dataset`
WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY) AND CURRENT_DATE()
  AND symbol = 'NIFTY'
  AND dte <= 30
  AND MOD(EXTRACT(MINUTE FROM ts_minute), 5) = 0  -- Sample every 5 minutes
```

### 4. Component Feature Join
Efficiently join specific component features:

```sql
WITH base_features AS (
  SELECT 
    symbol, ts_minute, date, dte, zone_name,
    c1_momentum_score, c1_vol_compression
  FROM `arched-bot-269016.market_regime_{env}.c1_features`
  WHERE date = CURRENT_DATE()
    AND symbol = 'NIFTY'
)
SELECT 
  b.*,
  c2.c2_gamma_exposure,
  c2.c2_sentiment_level,
  c3.c3_institutional_flow_score
FROM base_features b
JOIN `arched-bot-269016.market_regime_{env}.c2_features` c2
  USING (symbol, ts_minute, dte)
JOIN `arched-bot-269016.market_regime_{env}.c3_features` c3
  USING (symbol, ts_minute, dte)
```

### 5. Feature Aggregation by Zone
Aggregate features by trading zone:

```sql
SELECT 
  zone_name,
  symbol,
  dte,
  AVG(c1_momentum_score) as avg_momentum,
  AVG(c2_gamma_exposure) as avg_gamma,
  MAX(c3_institutional_flow_score) as max_institutional_flow,
  APPROX_QUANTILES(c4_iv_regime_level, 100)[OFFSET(50)] as median_iv_regime,
  COUNT(*) as sample_count
FROM `arched-bot-269016.market_regime_{env}.training_dataset`
WHERE date = CURRENT_DATE()
GROUP BY zone_name, symbol, dte
ORDER BY zone_name, symbol, dte
```

### 6. Regime Detection Query
Identify current market regime:

```sql
WITH latest_features AS (
  SELECT 
    symbol,
    c8_regime_classification,
    c8_regime_probability_trending,
    c8_regime_probability_volatile,
    c8_regime_probability_ranging,
    c8_regime_probability_breakout,
    c8_integration_confidence,
    ts_minute
  FROM `arched-bot-269016.market_regime_{env}.c8_features`
  WHERE date = CURRENT_DATE()
    AND symbol IN ('NIFTY', 'BANKNIFTY')
    AND ts_minute = (
      SELECT MAX(ts_minute) 
      FROM `arched-bot-269016.market_regime_{env}.c8_features`
      WHERE date = CURRENT_DATE()
    )
)
SELECT 
  symbol,
  CASE c8_regime_classification
    WHEN 1 THEN 'Trending'
    WHEN 2 THEN 'Volatile'
    WHEN 3 THEN 'Ranging'
    WHEN 4 THEN 'Breakout'
    ELSE 'Unknown'
  END as regime,
  ROUND(c8_integration_confidence, 3) as confidence,
  ts_minute as last_update
FROM latest_features
```

### 7. Feature Quality Check
Monitor feature quality and completeness:

```sql
SELECT 
  'c1_features' as table_name,
  COUNT(*) as row_count,
  COUNTIF(c1_momentum_score IS NULL) as null_momentum_score,
  COUNTIF(c1_vol_compression IS NULL) as null_vol_compression,
  MIN(ts_minute) as earliest_timestamp,
  MAX(ts_minute) as latest_timestamp,
  COUNT(DISTINCT symbol) as unique_symbols,
  COUNT(DISTINCT dte) as unique_dtes
FROM `arched-bot-269016.market_regime_{env}.c1_features`
WHERE date = CURRENT_DATE()

UNION ALL

-- Repeat for other component tables
SELECT 'c2_features', COUNT(*), COUNTIF(c2_gamma_exposure IS NULL), 
       COUNTIF(c2_sentiment_level IS NULL), MIN(ts_minute), MAX(ts_minute),
       COUNT(DISTINCT symbol), COUNT(DISTINCT dte)
FROM `arched-bot-269016.market_regime_{env}.c2_features`
WHERE date = CURRENT_DATE()
```

### 8. Incremental Data Load
Load only new data since last checkpoint:

```sql
DECLARE last_checkpoint TIMESTAMP DEFAULT (
  SELECT MAX(ts_minute) 
  FROM `arched-bot-269016.market_regime_{env}.feature_ingestion_checkpoint`
);

INSERT INTO `arched-bot-269016.market_regime_{env}.training_dataset`
SELECT * 
FROM `arched-bot-269016.market_regime_{env}.training_dataset_staging`
WHERE ts_minute > last_checkpoint
  AND ts_minute <= CURRENT_TIMESTAMP();

-- Update checkpoint
INSERT INTO `arched-bot-269016.market_regime_{env}.feature_ingestion_checkpoint`
VALUES (CURRENT_TIMESTAMP(), 'incremental_load', @@row_count);
```

### 9. Feature Store Batch Export
Export features for Vertex AI Feature Store ingestion:

```sql
EXPORT DATA OPTIONS(
  uri='gs://vertex-mr-data/feature_store_export/dt=*/features-*.parquet',
  format='PARQUET',
  overwrite=true
) AS
SELECT 
  CONCAT(symbol, '_', FORMAT_TIMESTAMP('%Y%m%d%H%M', ts_minute), '_', CAST(dte AS STRING)) as entity_id,
  ts_minute,
  -- All online features
  * EXCEPT(created_at, updated_at)
FROM `arched-bot-269016.market_regime_{env}.training_dataset`
WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 2 DAY)
  AND MOD(EXTRACT(MINUTE FROM ts_minute), 1) = 0;  -- Every minute
```

### 10. Performance Monitoring
Monitor query performance and costs:

```sql
SELECT 
  user_email,
  query,
  total_bytes_processed,
  total_slot_ms,
  ROUND(total_bytes_processed / POW(10, 12) * 5, 4) as estimated_cost_usd,
  creation_time,
  end_time,
  TIMESTAMP_DIFF(end_time, creation_time, MILLISECOND) as duration_ms
FROM `arched-bot-269016`.`region-us`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
WHERE creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
  AND statement_type = 'SELECT'
  AND state = 'DONE'
ORDER BY total_bytes_processed DESC
LIMIT 20;
```

## Query Optimization Tips

### 1. Always Use Partitioning
```sql
-- Good: Uses partition pruning
WHERE date = '2025-08-12'

-- Bad: Scans all partitions
WHERE DATE(ts_minute) = '2025-08-12'
```

### 2. Leverage Clustering
```sql
-- Good: Uses clustering
WHERE symbol = 'NIFTY' AND dte = 7

-- Less optimal: Only partial clustering benefit
WHERE dte = 7  -- Symbol should come first
```

### 3. Minimize Data Scanned
```sql
-- Good: Select only needed columns
SELECT symbol, c1_momentum_score, c2_gamma_exposure

-- Bad: Select all columns when not needed
SELECT *
```

### 4. Use Approximate Functions
```sql
-- Good: For large datasets
APPROX_QUANTILES(value, 100)[OFFSET(50)] as median

-- Expensive: Exact calculation
PERCENTILE_CONT(value, 0.5) OVER() as median
```

### 5. Cache Repeated Queries
```sql
-- Enable query cache
SET @@query_cache_max_results = 100;
```

## Cost Estimation

### Storage Costs
- Active storage: $0.02 per GB per month
- Long-term storage (90+ days): $0.01 per GB per month

### Query Costs
- On-demand: $5 per TB processed
- Flat-rate: Starting at $2,000/month for 500 slots

### Estimated Monthly Costs (Production)
- Storage: ~$94/month (47GB active, updated for 872 total features)
- Queries: ~$500/month (100k queries/day)
- Total: ~$594/month

### Feature Count Summary (Story 2.2)
- Component 1: 120 features (triple straddle)
- Component 2: 98 features (Greeks sentiment, γ=1.5)
- Component 3: 105 features (OI-PA trending)
- Component 4: 87 features (IV skew percentile)
- Component 5: 94 features (ATR-EMA-CPR)
- Component 6: 200 features (correlation & predictive)
- Component 7: 120 features (support/resistance - 72 base + 48 advanced)
- Component 8: 48 features (master integration)
- **Total: 872 features** (32 online, 840 offline)

## Best Practices

1. **Use materialized views** for frequently accessed aggregations
2. **Implement incremental refresh** for large tables
3. **Set table expiration** for old partitions
4. **Monitor costs** using INFORMATION_SCHEMA
5. **Use BigQuery BI Engine** for dashboard queries
6. **Enable result caching** for repeated queries
7. **Partition by date** and cluster by frequently filtered columns
8. **Use STRUCT** types for nested data
9. **Batch small queries** to reduce overhead
10. **Schedule heavy queries** during off-peak hours

## Troubleshooting

### Query Too Slow
- Check if partitioning and clustering are being used
- Verify no unnecessary JOINs or subqueries
- Consider materialized views

### Query Too Expensive
- Reduce columns selected
- Add partition filters
- Use sampling for development
- Consider flat-rate pricing

### Data Freshness Issues
- Check ingestion pipeline status
- Verify Feature Store sync frequency
- Monitor data quality metrics

## Related Documentation
- [Feature Store Specification](../ml/feature-store-spec.md)
- [Offline Feature Tables](../ml/offline-feature-tables.md)
- [BigQuery DDL Files](../../vertex_market_regime/src/bigquery/ddl/)