# Gate 2 Validation — Epic 2: Vertex AI Foundation
Date: 2025-08-12  
Status: PENDING  
Owner: PO / Platform / QA

## Pass Criteria
- [ ] BigQuery dataset `market_regime_{env}` exists; ≥1 `c*_features` table populated
- [ ] Feature Store created with entity types; ~32 online features registered; sample read <50ms P95
- [ ] CustomJob completed on a sample day; logs stored; audit table updated
- [ ] Training pipeline ran end-to-end; baseline model registered in Model Registry
- [ ] IAM/Artifact Registry/budgets validated; monitoring dashboards accessible

## Artifacts
- Feature Store Spec: `docs/ml/feature-store-spec.md`
- Offline Features Spec: `docs/ml/offline-feature-tables.md`
- Training Pipeline Spec: `docs/ml/training-pipeline-spec.md`
- Setup Guide: `docs/ml/vertex-ai-setup.md`
- Epic Doc: `docs/stories/epic-2-data-pipeline-modernization.md`

## Decision
- [ ] GO to Epic 3
- [ ] NO-GO (list blocking issues)

## Risks/Notes
- Online feature cardinality/TTL costs
- Model registry without endpoint until Epic 3



