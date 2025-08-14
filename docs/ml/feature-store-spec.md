# Vertex AI Feature Store Specification (Market Regime)

Version: 1.0  
Date: 2025-08-12

## Purpose
Define entity types, feature definitions, TTLs, and ingestion approach for online features required for low-latency regime analysis.

## Entity Types

### instrument_minute (minute-level online serving)
- entity_id: `${symbol}_${yyyymmddHHMM}_${dte}` (STRING)
- keys (for reference):
  - symbol: STRING (e.g., NIFTY)
  - ts_minute: TIMESTAMP (UTC, minute truncated)
  - dte: INT64 (days to expiry)
- TTL: 48h
- Online Store: Vertex AI Feature Store
- Offline Store: BigQuery (`market_regime_{env}.training_dataset`)

### instrument_day (daily aggregates, optional)
- entity_id: `${symbol}_${yyyymmdd}` (STRING)
- TTL: 30d

## Online Core Feature Set (≈32 features)
Minimal online subset to support Epic 3 serving; all other features remain offline in BigQuery.

- Component 1 (Straddle):
  - c1_momentum_score (FLOAT32)
  - c1_vol_compression (FLOAT32)
  - c1_breakout_probability (FLOAT32)
- Component 2 (Greeks):
  - c2_gamma_exposure (FLOAT32)
  - c2_sentiment_level (INT32)
  - c2_pin_risk_score (FLOAT32)
- Component 3 (OI-PA):
  - c3_institutional_flow_score (FLOAT32)
  - c3_divergence_type (INT32)
  - c3_range_expansion_score (FLOAT32)
- Component 4 (IV Skew/Percentiles):
  - c4_skew_bias_score (FLOAT32)
  - c4_term_structure_signal (INT32)
  - c4_iv_regime_level (INT32)
- Component 5 (ATR-EMA-CPR):
  - c5_momentum_score (FLOAT32)
  - c5_volatility_regime_score (INT32)
  - c5_confluence_score (FLOAT32)
- Component 6 (Correlation/Prediction):
  - c6_correlation_agreement_score (FLOAT32)
  - c6_breakdown_alert (INT32)
  - c6_system_stability_score (FLOAT32)
- Component 7 (S/R):
  - c7_level_strength_score (FLOAT32)
  - c7_breakout_probability (FLOAT32)
- Component 8 (Integration features):
  - c8_component_agreement_score (FLOAT32)
  - c8_integration_confidence (FLOAT32)
  - c8_transition_probability_hint (FLOAT32)

- Context features:
  - zone_name (STRING) — OPEN/MID_MORN/LUNCH/AFTERNOON/CLOSE
  - ts_minute (TIMESTAMP)
  - symbol (STRING)
  - dte (INT64)

Note: Types align with schema registry. Enumerations use INTs. All feature names are snake_case and prefixed `c{n}_`.

## Offline Features
- All remaining features (774+ total across components) materialized to BigQuery tables (`c1_features`..`c8_features`, `training_dataset`).

## Ingestion
- Batch: Vertex AI CustomJob reads Parquet (GCS) → computes/validates → writes BigQuery; then registers selected features to FS
- Online: FS ingestion job materializes the ~32 online features per minute entity

## Parity & Versioning
- Feature version captured in schema registry; FS feature description includes version and provenance
- Train/serve parity validated via parity harness; sampled comparisons logged to BigQuery

## Ownership
- Product Owner: PO
- Platform: Vertex AI/Infra
- Data: FE owners per component



