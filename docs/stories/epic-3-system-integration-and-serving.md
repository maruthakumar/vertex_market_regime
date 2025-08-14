# Epic 3: System Integration and Serving
**Duration:** Weeks 3-4  
**Status:** Planned  
**Priority:** CRITICAL - End-to-end runtime path

## Epic Goal
Integrate the 8-component analysis with master integration, wire online feature serving, and expose API v2 endpoints that meet base performance targets (<800ms total). This epic turns validated feature engineering (Epic 1) and provisioned infra (Epic 2) into a cohesive, low-latency runtime service.

## Epic Description
- Orchestrate all 8 components in parallel with strict per-component budgets and fallbacks
- Perform master integration and 8-regime classification using adaptive weights
- Retrieve online features from Vertex AI Feature Store with training/serving parity
- Provide API v2 endpoints with authentication, monitoring, and structured logs

### Scope refinement for 95% win rate
- Success metric is precision-at-threshold ≥95% with selective inference. Coverage (percent of minutes classified) is tuned via thresholds and reported. Accuracy alone is not the target.

## Epic Stories

### Story 0: Label Spec and Generation (CRITICAL)
**As a** data scientist  
**I want** deterministic, no-lookahead labels for 8 regimes and transition events in BigQuery  
**So that** models learn correct targets and downstream calibration is meaningful

**Acceptance Criteria:**
- [ ] Label spec documented in `docs/ml/label-spec.md` (sources, windows, tie-breaks, no-lookahead rules)
- [ ] BigQuery SQL creates `labels_minute` with (symbol, ts_minute, dte, regime_label, transition_label)
- [ ] Tests verifying no future leakage (purge/embargo windows defined)

### Story 1: Training Dataset Build (BQ)
**Acceptance Criteria:**
- [ ] `training_dataset` view/table joining `c1..c8` features on (symbol, ts_minute, dte) + labels
- [ ] Time-based split config (train/val/test) with anchored windows
- [ ] Data quality report (row counts, nulls, distributions) stored in `docs/ml/`

### Story 2: Baseline Model Training and Registration
**Acceptance Criteria:**
- [ ] Train a strong single model (XGBoost or TabNet) for 8-regime classification on joined features
- [ ] Train a short-sequence transition forecaster (baseline LSTM/TFT) for transition probabilities
- [ ] Log metrics to Vertex AI Experiments; register models in Model Registry (no endpoint yet)

### Story 3: Probability Calibration and Threshold Gating (CRITICAL)
**Acceptance Criteria:**
- [ ] Calibrate probabilities (Platt or isotonic) on validation set
- [ ] Compute PR curves and select thresholds to achieve precision ≥95%; document resulting coverage
- [ ] Produce gating config file (thresholds per regime/transition) versioned in repo

### Story 4: Serving Integration with Selective Inference
**Acceptance Criteria:**
- [ ] API v2 enforces thresholds; returns regime only when confidence ≥ threshold, else emits "no-signal"
- [ ] Response includes regime, confidence, transition probabilities, reasons/timings
- [ ] Online features pulled from Feature Store; sampled parity checks recorded

### Story 5: Optional Ensemble Experiment (Offline)
**Acceptance Criteria:**
- [ ] If baseline fails precision/coverage targets, run stacking experiment offline (AutoML/TabTransformer/XGBoost, etc.)
- [ ] If ensemble improves PR curve materially, distill to a compact serving model; else document no-go

## Validation Gates (Go/No-Go)
- [ ] Label generation validated (no-lookahead), dataset joins complete
- [ ] Calibrated thresholds achieve precision ≥95% on holdout; coverage target documented
- [ ] Total latency (P95) ≤ 800ms (base); memory ≤ 3.7GB
- [ ] API v2 selective inference live in staging; parity checks green

## Success Metrics
- Total P95 latency ≤ 800ms (stretch ≤ 600ms after optimization)
- Memory peak ≤ 3.7GB (stretch ≤ 2.5GB)
- Precision@threshold ≥ 95%; coverage target agreed and tracked
- Error budget policy defined and tracked

## Definition of Done
- [ ] Orchestration service, master integration, and API v2 implemented and tested
- [ ] Online feature serving integrated with parity checks
- [ ] Calibration and gating in place; thresholds versioned
- [ ] Observability and alerts active; runbooks drafted
- [ ] Performance baselines documented; gates passed
- [ ] Handoff to Epic 4 (production readiness) with risks noted
