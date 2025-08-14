# Testing Strategy (BMAD-Aligned)

Version: 1.0  
Date: 2025-08-10

## Purpose
Define test layers, tooling, and validation gates to ensure the Market Regime Master Framework meets functional and non-functional requirements with evidence.

## Test Layers

### 1. Unit Tests
Scope: Pure functions and feature transforms in the FE framework and components.
- Tools: pytest, hypothesis (optional), numpy/pandas/cuDF asserts
- Requirements:
  - Deterministic outputs for fixed inputs
  - Strict type/shape checks via pydantic/dataclasses where applicable
  - Edge cases (missing values, extreme DTE, zero volume)

### 2. Component Integration Tests
Scope: Full component FE pipeline over representative Parquet samples.
- Tools: pytest + local GPU (RAPIDS/cuDF), Arrow memory pools
- Requirements:
  - Feature count matches spec (120/98/105/87/94/150/72/48 or updated counts per epic)
  - Runtime within per-component budgets
  - Outputs include component_score, component_confidence, transition_thresholds

### 3. Parity Harness Tests
Scope: Training vs serving feature parity for all components.
- Tools: Custom parity harness; hash-based or tolerance-based comparisons
- Requirements:
  - Same transforms executed in train and infer contexts
  - Matching outputs (exact or ε tolerance) with discrepancies reported
  - Schema versioning and backfill checks

### 4. System Integration Tests (Epic 3)
Scope: Orchestration, master integration, API v2, online feature serving.
- Tools: pytest + httpx/requests; load scenarios; mocks for Feature Store
- Requirements:
  - End-to-end latency <800ms base (P95)
  - Correct 8-regime outputs and agreement/confidence metrics
  - Robust fallbacks and error paths validated

### 5. Performance and Load Tests
Scope: Per-component and end-to-end under realistic load.
- Tools: pytest-benchmark/Locust/k6, custom timers, memory profilers
- Requirements:
  - Budgets enforced; memory stable, no leaks
  - Cold start mitigation validated

### 6. Security and Resilience Tests
Scope: API authN/authZ, input validation, circuit breakers, rollback.
- Tools: OWASP checks, fuzzing, fault injection
- Requirements:
  - No P1 security findings; resilient to component timeouts/Cloud failures

## Test Data Management
- Parquet samples checked into a separate data bucket; small, representative subsets
- Synthetic data generators for edge cases where needed
- Versioned test datasets; link versions to schema registry entries

## CI Integration
- Run unit + component tests on commit; parity harness nightly
- Performance smoke tests weekly or pre-release
- Security scans in CI/CD; block on P1 findings

## Validation Gates (Evidence)
- Gate 1 (end of Epic 1):
  - Evidence: FE counts per component, parity reports, performance baseline
- Gate 2 (end of Epic 2):
  - Evidence: Infra deployed, BQ tables/FS entity types created, smoke tests
- Gate 3 (end of Epic 3):
  - Evidence: E2E latency <800ms P95, API v2 integration tests, online feature parity checks
- Gate 4 (end of Epic 4):
  - Evidence: Sustained SLOs, security review, rollback drills, production runbooks

## Reporting
- Store reports under `docs/validation/` and `docs/performance/`
- Summarize key metrics in epic documents; link to artifacts

## Ownership
- FE/unit tests: Component owners
- Parity/system/performance: Platform & QA
- Security/resilience: Security/Platform

## Change Control
- Test updates must accompany schema/feature changes
- Keep parity harness specs versioned with feature definitions

