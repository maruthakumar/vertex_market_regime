# Epic 1: Feature Engineering Foundation
**Duration:** Weeks 1-2  
**Status:** Ready for Development  
**Priority:** CRITICAL - Foundation for all subsequent components

## Epic Goal
Implement the full feature engineering pipeline for all 8 components using the local Parquet → Arrow → GPU path, establishing stable feature schemas, tests, and performance baselines. Cloud infrastructure (BigQuery/Feature Store/Endpoints) will be provisioned in the next epic after feature definitions are validated.

## Epic Description

### Execution Mode
- Local: Parquet → Arrow → RAPIDS/cuDF on a local GPU host (no BigQuery in request path)
- Deliver reproducible, versioned feature transforms with training/serving parity guaranteed by a test harness

### Deliverables
- Feature Engineering Framework (shared utilities, schema registry, caching)
- Component feature sets implemented and validated (total 774 features across 8 components)
- Parity harness and unit tests for deterministic feature outputs
- Performance baselines per component and overall

### Component logic and regime input contracts
For each component, implement the logic signals and output contracts required for 8-regime formation, including component-level confidence and transition thresholds. Produce the following per component:
- component_score ∈ [-1.0, 1.0]
- component_confidence ∈ [0.0, 1.0]
- transition_thresholds: named thresholds that trigger regime transitions (documented in schema registry)

Contracts by component:
- Component 1 (Triple Straddle): straddle_trend_score, vol_compression_score, breakout_probability, transition_thresholds for CPR width and momentum shifts
- Component 2 (Greeks Sentiment): gamma_exposure_score (γ=1.5), vanna/charm/volga_aggregate, pin_risk_score, sentiment_level, thresholds for sentiment flips and pin-risk proximity
- Component 3 (OI-PA Trending): institutional_flow_score, divergence_type, range_expansion_score, thresholds for flow reversals and divergence strength
- Component 4 (IV Skew): skew_bias_score, term_structure_signal, iv_regime_level, thresholds for skew inversion and percentile bands
- Component 5 (ATR-EMA-CPR): momentum_score, volatility_regime_score, confluence_score, thresholds for ATR band breaks and EMA confluence
- Component 6 (Correlation): correlation_agreement_score, breakdown_alert, system_stability_score, thresholds for correlation breakdown/onset
- Component 7 (Support/Resistance): level_strength_score, breakout_probability, thresholds for breakout/failed-break reclassifications
- Component 8 (Master Integration inputs only in Epic 1): integration_feature_staging, proposed_transition_thresholds, component_weight_hints (final integration in Epic 3)

All outputs must be defined in the schema registry with names, types, ranges, and calibration notes.

### References
- PRD: [docs/prd.md](../prd.md)
- Architecture: [docs/architecture.md](../architecture.md)
- Component 1: [mr_tripple_rolling_straddle_component1.md](../market_regime/mr_tripple_rolling_straddle_component1.md)
- Component 2: [mr_greeks_sentiment_analysis_component2.md](../market_regime/mr_greeks_sentiment_analysis_component2.md)
- Component 3: [mr_oi_pa_trending_analysis_component3.md](../market_regime/mr_oi_pa_trending_analysis_component3.md)
- Component 4: [mr_iv_skew_analysis_component4.md](../market_regime/mr_iv_skew_analysis_component4.md)
- Component 5: [mr_atr_ema_cpr_component5.md](../market_regime/mr_atr_ema_cpr_component5.md)
- Component 6: [mr_correlation_noncorelation_component6.md](../market_regime/mr_correlation_noncorelation_component6.md)
- Component 7: [mr_support_resistance_component7.md](../market_regime/mr_support_resistance_component7.md)
- Component 8: [mr_dte_adaptive_overlay_component8.md](../market_regime/mr_dte_adaptive_overlay_component8.md)

### Success Criteria
- Per-component feature counts match specification (120/98/105/87/94/150/72/48)
- Parity tests pass across train/infer paths for all components
- Performance: per-component budgets met locally; overall baseline recorded

## Epic Stories

### Story 1: Feature Engineering Framework Scaffolding (CRITICAL)
**As a** platform engineer  
**I want** a unified FE framework with schema registry, caching, and reproducible transforms  
**So that** all components share consistent, versioned feature definitions

**Acceptance Criteria:**
- [ ] Schema registry with versioned feature definitions per component
- [ ] Common utilities for Arrow/RAPIDS transforms
- [ ] Deterministic transform functions (pure, side-effect free)
- [ ] Local feature cache with TTL controls for iterative runs

---

### Story 2: Component 1 Feature Engineering (120 Features)
- Rolling straddle price overlays (EMA/VWAP/Pivots) on straddle prices, multi-timeframe integration
- Target: <150ms

**Acceptance Criteria:**
- [ ] 120 features materialized and validated
- [ ] Unit tests verifying counts and value ranges
- [ ] Performance <150ms on representative sample

---

### Story 3: Component 2 Feature Engineering (98 Features, Gamma Fix)
- Volume-weighted Greeks; gamma weight = 1.5; second-order Greeks (vanna/charm/volga)
- Target: <120ms

**Acceptance Criteria:**
- [ ] 98 features materialized and validated
- [ ] Gamma weight 1.5 used in all relevant transforms
- [ ] Parity checks for sentiment scoring

---

### Story 4: Component 3 Feature Engineering (105 Features)
- Cumulative ATM ±7 strikes; institutional flow; divergence types  
- Target: <200ms

**Acceptance Criteria:**
- [ ] 105 features materialized and validated
- [ ] Multi-timeframe rollups verified

---

### Story 5: Component 4 Feature Engineering (87 Features) - Enhanced IV Percentile Analysis
- Individual DTE tracking (dte=0...dte=58); 7-regime IV percentile classification; 4-timeframe momentum analysis; sophisticated IVP+IVR implementation
- Target: <350ms (Enhanced for precision IV percentile analysis)

**Acceptance Criteria:**
- [ ] 87 features materialized and validated with enhanced IV percentile sophistication
- [ ] Individual DTE-level IV percentile tracking (dte=0, dte=1...dte=58)
- [ ] 7-regime classification system (Extremely Low to Extremely High IV regimes)
- [ ] 4-timeframe momentum analysis (5min/15min/30min/1hour)
- [ ] 4-zone intraday analysis (MID_MORN/LUNCH/AFTERNOON/CLOSE) per production schema
- [ ] Advanced IVP + IVR integration with historical ranking system

---

### Story 6: Component 5 Feature Engineering (94 Features)
- Dual asset analysis (straddle vs underlying); ATR/EMA/CPR  
- Target: <200ms

**Acceptance Criteria:**
- [ ] 94 features materialized and validated
- [ ] Cross-asset confluence checks

---

### Story 7: Component 6 Feature Engineering (150 Features)
- 30x30 correlation matrix (expert-optimized 774 set backbone)  
- Target: <180ms

**Acceptance Criteria:**
- [ ] 150 correlation features materialized and validated
- [ ] Progressive computation stability (no explosion)

---

### Story 8: Component 7 Feature Engineering (72 Features)
- Multi-method levels; dual asset confluence; level strength  
- Target: <150ms

**Acceptance Criteria:**
- [ ] 72 features materialized and validated
- [ ] Strength scoring monotonicity tests

---

### Story 9: Component 8 Feature Engineering (48 Features)
- Master integration feature staging; DTE-adaptive weighting; structure change signals; confidence and transition thresholds  
- Target: <100ms

**Acceptance Criteria:**
- [ ] 48 features materialized and validated
- [ ] Inputs sourced from prior component outputs only
- [ ] component_confidence and proposed transition_thresholds defined and documented

---

### Story 10: Training/Serving Parity Harness
**As a** QA engineer  
**I want** a parity harness that runs train-time and infer-time transforms with identical inputs  
**So that** feature outputs match exactly across contexts

**Acceptance Criteria:**
- [ ] Harness produces matching hashes for train vs infer pipelines
- [ ] Reports discrepancies with diffs and thresholds

---

### Story 11: Performance Baseline Measurement
**As a** performance engineer  
**I want** a baseline report of per-component and overall FE times and memory  
**So that** we can track improvements and regressions

**Acceptance Criteria:**
- [ ] Report includes per-component times, memory usage, and totals
- [ ] Stored in `docs/performance/fe-baseline.md`

### Story 12: Output Contracts and Calibration
**As a** quantitative engineer  
**I want** formal output contracts and calibration across all components  
**So that** master integration can rely on consistent signals and thresholds

**Acceptance Criteria:**
- [ ] For each component: component_score, component_confidence, transition_thresholds defined in schema registry
- [ ] Calibration procedure documented (binning, isotonic/Platt if needed)
- [ ] Sample calibration curves and validation metrics stored under `docs/validation/component-calibration/`

## Epic Dependencies & Validation Gates

### Prerequisites
- [ ] Representative Parquet datasets available locally
- [ ] Development environment with GPU access

### Validation Gates (Go/No-Go Decision Points)
- [ ] All 8 component feature sets implemented with correct counts
- [ ] Parity harness passes on samples for all components
- [ ] Performance baselines recorded; budgets met or risks flagged

### Cross-Epic Handoff
- Handoff artifacts: schema registry, feature specs, tests, baseline report
- Next Epic (2): Cloud infrastructure setup (BigQuery tables, Vertex AI Feature Store, endpoints) using these finalized feature specs

## Definition of Done
- [ ] Feature engineering framework complete and versioned
- [ ] All 8 component feature sets complete with tests passing
- [ ] Parity harness green across all components
- [ ] Performance baseline documented and accepted
- [ ] Handoff package ready for Epic 2 provisioning
