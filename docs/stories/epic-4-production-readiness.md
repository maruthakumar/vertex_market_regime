# Epic 4: Production Readiness
**Duration:** Weeks 5-6  
**Status:** Planned  
**Priority:** CRITICAL - Launch quality

## Epic Goal
Harden the system for production with performance tuning, security/compliance, observability, runbooks, rollback drills, and final validation to meet or exceed base targets (stretch where possible).

## SLOs (production)
- Precision@threshold ≥ 95% (per regime), measured on rolling window
- Coverage ≥ agreed target (e.g., ≥30–50%), reported alongside precision
- Latency P95 ≤ 800ms (stretch ≤ 600ms)
- Online feature lookup P95 ≤ 50ms; Feature Store availability SLO met

## Epic Stories

### Story 1: Observability and SLO Dashboards
**Acceptance Criteria:**
- [ ] Dashboards for precision@thr, coverage, latency, online feature P95, accuracy drift
- [ ] Alerts on precision drop, coverage collapse, latency breach, feature parity issues

### Story 2: Shadow/Canary and Rollback
**Acceptance Criteria:**
- [ ] Shadow deploy vs v1; canary with guardrails
- [ ] One-click rollback with safe defaults and documented runbook

### Story 3: Drift Detection and Retraining Cadence
**Acceptance Criteria:**
- [ ] Drift signals (feature/label) computed daily; thresholds and actions documented
- [ ] Retrain schedule defined (e.g., weekly/bi-weekly) with artifact/versioning checklist

### Story 4: Security/Compliance Hardening
**Acceptance Criteria:**
- [ ] IAM least-privilege verified; secrets rotation; audit logging enabled
- [ ] Pen-test checklist passed; PII review complete (if applicable)

## Validation Gates (Go/No-Go)
- [ ] SLO dashboards live; alerts tested
- [ ] Canary passed SLOs; rollback drill completed successfully
- [ ] Drift detection producing signals; retrain pipeline tested end-to-end

## Definition of Done
- [ ] All SLOs enforced and monitored
- [ ] Incident playbooks/runbooks committed
- [ ] Final BMAD validation gates passed
