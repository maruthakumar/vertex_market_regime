# Epic 2: Data Pipeline Modernization

> Note (2025-08-10): This epic is sequenced after `docs/stories/epic-1-feature-engineering-foundation.md` (BMAD-aligned). Cloud infrastructure provisioning (BigQuery/Feature Store/Endpoints) proceeds here using finalized feature schemas from Epic 1.

**Duration:** Weeks 1-2  
**Status:** In Progress (2025-08-12)  
**Priority:** CRITICAL - Foundation for all subsequent components

## Epic Goal
Transform the legacy HeavyDB-based data processing pipeline into a modern Parquet → Arrow → GPU architecture while establishing cloud integration foundation, ensuring zero disruption to existing trading operations.

## Epic Description

### Vertex AI Foundation (Trimmed Scope)
This epic is trimmed to ONLY deliver Vertex AI ML foundations required for market regime formation. HeavyDB is deprecated and out of scope. Canonical data: GCS Parquet (local for dev), BigQuery for offline features, Vertex AI Feature Store for online features.

#### Goals
- BigQuery offline feature tables per component (C1..C8) + unified training dataset
- Vertex AI Feature Store with entity types and a minimal online feature set for low-latency serving
- Containerized FE/ingestion jobs runnable as Vertex AI CustomJobs; optional Pipelines orchestration
- Artifact Registry for images; Experiments/Model Registry for training outputs
- IAM/budgets/monitoring configured for the above

#### Trimmed Epic 2 Stories (Vertex AI only)
1) Feature Store Design & Spec (CRITICAL)
- Define entity types and feature defs derived from schema registry (Epic 1)
- Keys: entity_id = `${symbol}_${yyyymmddHHMM}_${dte}` for minute-level; alt daily entity for aggregates
- TTL for online features (e.g., 48h); offline store = BigQuery tables
- Acceptance:
  - [ ] Spec doc with entity types, features, TTLs, data types, owners
  - [ ] Mapping from schema registry → FS names (snake_case)

2) BigQuery Offline Feature Tables
- Dataset `market_regime_{env}` with partitioned, clustered tables: `c1_features`..`c8_features` and `training_dataset`
- Partition: by `date` (DATE) or `ts_minute` (TIMESTAMP); Cluster: `symbol`, `dte`
- Acceptance:
  - [ ] DDLs defined in docs; dry-run queries validated
  - [ ] Minimum one populated sample table from Parquet for smoke test

3) FE Container + CustomJob
- Containerize FE ingestion job reading Parquet → produce BigQuery tables
- Submit Vertex AI CustomJob (GPU optional) to process a 1-day slice
- Acceptance:
  - [ ] Dockerfile + run script documented (Artifact Registry path)
  - [ ] CustomJob spec YAML/Python sample; job runs successfully on sample slice

4) Training Pipeline Skeleton + Experiments/Registry
- Define KFP v2 (Vertex AI Pipelines) skeleton: BQ dataset creation → train → eval → register
- Baseline models (not tuned):
  - Regime classifier (TabNet/XGBoost/Transformer tabular)
  - Transition forecaster (LSTM/Temporal Fusion transformer candidate)
- Acceptance:
  - [ ] Pipeline spec documented with IO contracts
  - [ ] Model registered in Vertex AI Model Registry (no endpoint yet)

5) IAM, Artifact Registry, Budgets/Monitoring
- Service accounts + least-priv roles: Vertex AI, BQ JobUser/DataEditor, GCS viewer, AR writer
- Create Artifact Registry repo (region-scoped)
- Budgets and alerting rules
- Acceptance:
  - [ ] IAM matrix documented; AR path set; budgets in place

6) Minimal Online Feature Registration
- Register a small core online feature set (≈32) across components for Epic 3 serving path
- Acceptance:
  - [ ] FS created; entity types exist; features registered; sample ingestion verified

#### Validation Gates (Go/No-Go for Epic 3)
- [ ] BigQuery dataset exists with ≥1 offline feature table populated
- [ ] Feature Store created; core online features registered; sample read latency validated
- [ ] CustomJob executed successfully on sample Parquet slice; logs stored
- [ ] Training pipeline skeleton completed; baseline model registered in Model Registry
- [ ] IAM/AR/budgets configured and validated

#### Out of Scope (deferred to Epic 3)
- Endpoint deployment and API v2 integration
- Full-scale training/tuning; online inference wiring

### Existing System Context
- **Current System:** HeavyDB-based CSV processing in backtester_v2
- **Technology Stack:** Python 3.8+, HeavyDB, Pandas/cuDF, Excel configuration
- **Architecture:** Monolithic data processing with ~3000ms latency
- **Integration Points:** market_regime_strategy.py, Excel parameter system, REST API endpoints

### Enhancement Details
- **What's being added:** Modern Parquet → Arrow → GPU data pipeline with cloud integration
- **How it integrates:** Parallel implementation with gradual migration from HeavyDB
- **Success criteria:** <800ms processing time, 100% API compatibility, zero downtime migration

## Epic Stories

### Story 1: Database Migration Planning & Validation (CRITICAL FIX)
**As a** system administrator  
**I want** comprehensive database migration procedures with validation checkpoints  
**So that** the HeavyDB → Parquet migration can proceed safely without data loss

**Acceptance Criteria:**
- [ ] Complete schema mapping from HeavyDB to Parquet format documented
- [ ] Data migration scripts with rollback procedures created
- [ ] Validation checkpoints defined for each migration stage
- [ ] Data integrity verification procedures established
- [ ] Rollback testing procedures documented
- [ ] Migration timeline with Go/No-Go decision points defined

**Technical Requirements:**
- [ ] HeavyDB schema analysis and documentation
- [ ] Parquet schema design with Arrow optimization
- [ ] ETL pipeline specification for data conversion
- [ ] Data validation algorithms for integrity checking
- [ ] Automated rollback procedures tested

**Risk Mitigation:**
- **Primary Risk:** Data loss or corruption during migration
- **Mitigation:** Comprehensive validation at each step, parallel system operation
- **Rollback:** Complete HeavyDB restoration procedures with data verification

### Story 2: GCP Infrastructure Setup (CRITICAL FIX)
**As a** cloud administrator  
**I want** Infrastructure as Code (IaC) for complete GCP resource provisioning  
**So that** Vertex AI integration can be deployed consistently with cost controls

**Acceptance Criteria:**
- [ ] Terraform scripts for Vertex AI, BigQuery, Cloud Storage provisioning
- [ ] Service account configuration with least-privilege IAM roles
- [ ] VPN setup between local HeavyDB and GCP resources
- [ ] Cost monitoring with budget alerts and resource optimization
- [ ] Network security configuration with firewall rules
- [ ] Automated deployment and teardown procedures

**Technical Requirements:**
- [ ] Terraform modules for each GCP service
- [ ] Service account key management and rotation
- [ ] Network configuration with private connectivity
- [ ] Monitoring and logging setup
- [ ] Cost optimization policies and alerts
- [ ] Security hardening and compliance checks

**Resource Specifications:**
```hcl
# Vertex AI Configuration
vertex_ai = {
  region = "us-central1"
  machine_type = "n1-standard-4"
  gpu_type = "nvidia-tesla-t4"
  gpu_count = 1
}

# BigQuery Configuration  
bigquery = {
  dataset_location = "US"
  table_expiration_ms = 7776000000 # 90 days
}

# Cloud Storage Configuration
storage = {
  location = "US-CENTRAL1"
  storage_class = "STANDARD"
}
```

### Story 3: Excel → YAML Conversion System
**As a** configuration manager  
**I want** automatic Excel to YAML conversion with validation  
**So that** existing 600+ parameters are preserved while enabling cloud-native configuration

**Acceptance Criteria:**
- [ ] All Excel sheets automatically parsed and converted to YAML
- [ ] Parameter validation and error handling implemented
- [ ] Change detection triggers automatic YAML updates
- [ ] Backward compatibility with existing parameter names maintained
- [ ] Configuration versioning and rollback capability

**Technical Implementation:**
```python
# Excel Parser Enhancement
class EnhancedExcelParser:
    def __init__(self, excel_path: str, yaml_output_path: str):
        self.excel_path = excel_path
        self.yaml_output_path = yaml_output_path
        self.validation_rules = self._load_validation_rules()
    
    def convert_to_yaml(self) -> Dict[str, Any]:
        # Parse all Excel sheets
        # Validate parameters
        # Convert to structured YAML
        # Apply versioning
        pass
```

### Story 4: Parquet Data Processing Pipeline
**As a** quantitative analyst  
**I want** high-performance Parquet-based data processing with GPU acceleration  
**So that** large datasets can be analyzed efficiently with >10x performance improvement

**Acceptance Criteria:**
- [ ] Parquet format processing operational with Apache Arrow integration
- [ ] GPU optimization using RAPIDS cuDF functional
- [ ] Processing performance >10x faster than existing CSV processing
- [ ] Seamless integration with existing data sources maintained
- [ ] Memory usage optimized for <3.7GB total system constraint

**Technical Architecture:**
```python
# Modern Data Pipeline
class ParquetArrowPipeline:
    def __init__(self):
        self.arrow_context = pa.default_memory_pool()
        self.cudf_context = cudf.get_option('default_gpu_memory_pool')
    
    def process_market_data(self, data_source: str) -> cudf.DataFrame:
        # Load from Parquet
        # Convert to Arrow format
        # Process with GPU acceleration
        # Return cuDF DataFrame
        pass
```

### Story 5: Vertex AI Connection Establishment
**As a** ML engineer  
**I want** basic Vertex AI integration with authentication and monitoring  
**So that** the foundation for adaptive learning components is established

**Acceptance Criteria:**
- [ ] Vertex AI client connection with authentication working
- [ ] Basic model serving endpoint operational
- [ ] Connection monitoring and health checks implemented
- [ ] Error handling with fallback to local processing
- [ ] Latency monitoring and circuit breaker pattern

**Integration Pattern:**
```python
# Vertex AI Client
class VertexAIClient:
    def __init__(self):
        self.client = aiplatform.gapic.PredictionServiceClient()
        self.circuit_breaker = CircuitBreaker(failure_threshold=3)
    
    @circuit_breaker
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        # Vertex AI prediction with fallback
        pass
```

### Story 6: API Documentation System (HIGH PRIORITY FIX)
**As a** developer  
**I want** automated API documentation generation with v1/v2 compatibility guides  
**So that** integration teams can seamlessly adopt enhanced endpoints

**Acceptance Criteria:**
- [ ] OpenAPI 3.0 specifications for all v2 endpoints
- [ ] Interactive documentation with Swagger UI
- [ ] Migration guides from v1 to v2 endpoints  
- [ ] Code examples for each endpoint
- [ ] Integration testing documentation
- [ ] Developer onboarding updated

**Documentation Structure:**
```yaml
api_documentation:
  v2_endpoints:
    - /api/v2/regime/analyze
    - /api/v2/regime/weights  
    - /api/v2/regime/health
  migration_guides:
    - v1_to_v2_compatibility.md
    - endpoint_mapping.md
    - breaking_changes.md
  integration_examples:
    - python_client_example.py
    - javascript_integration.js
    - curl_examples.sh
```

## Epic Dependencies & Validation Gates

### Prerequisites
- [ ] Existing backtester_v2 system analysis completed
- [ ] HeavyDB schema documentation current
- [ ] Google Cloud project provisioned with billing
- [ ] Development environment with GPU access available

### Validation Gates (Go/No-Go Decision Points)

#### Week 1 Checkpoint
**Criteria for proceeding to Week 2:**
- [ ] Database migration plan validated with dry-run testing
- [ ] GCP infrastructure provisioned and accessible
- [ ] Excel → YAML conversion working for sample configurations
- [ ] No regression in existing system functionality

#### Week 2 Checkpoint (Epic Completion)
**Criteria for proceeding to Epic 2:**
- [ ] Parquet pipeline processing real market data successfully
- [ ] Vertex AI connection established with basic health checks
- [ ] API documentation published and accessible
- [ ] Performance baseline established (current vs new pipeline)
- [ ] All integration points tested and verified

### Cross-Epic Coordination

**Epic 1 → Epic 2 Handoff Requirements:**
- [ ] Modern data pipeline operational and performance-validated
- [ ] Cloud infrastructure ready for component deployment
- [ ] Configuration system supports component parameter management
- [ ] Monitoring infrastructure can track component performance
- [ ] API framework ready for component integration

## Risk Assessment & Mitigation

### High-Risk Areas

1. **Database Migration Complexity**
   - **Risk:** Data corruption or extended downtime
   - **Mitigation:** Parallel system operation, comprehensive validation
   - **Contingency:** Complete rollback procedures tested

2. **Cloud Integration Security**
   - **Risk:** Unauthorized access or data breach
   - **Mitigation:** Least-privilege IAM, VPN connectivity, audit logging
   - **Contingency:** Immediate access revocation procedures

3. **Performance Degradation**
   - **Risk:** New pipeline slower than expected
   - **Mitigation:** Benchmark testing, GPU optimization, caching
   - **Contingency:** Automatic fallback to HeavyDB processing

## Success Metrics

### Technical KPIs
- **Processing Time:** <800ms (target: 73% improvement from 3000ms)
- **Memory Usage:** <3.7GB total system memory
- **API Latency:** <100ms for configuration endpoints
- **Data Integrity:** 100% validation pass rate during migration

### Business KPIs  
- **System Uptime:** >99.5% during migration period
- **API Compatibility:** 100% backward compatibility maintained
- **Cost Efficiency:** GCP costs <$500/month operational
- **Developer Velocity:** Documentation reduces onboarding time by 50%

## Definition of Done

### Epic Completion Criteria
- [ ] All 6 stories completed with acceptance criteria met
- [ ] Database migration executed successfully with validation
- [ ] GCP infrastructure operational with cost monitoring
- [ ] Modern data pipeline processing live market data
- [ ] API documentation published and developer-tested
- [ ] No regression in existing backtester functionality
- [ ] Performance targets achieved and sustained
- [ ] Epic 2 prerequisites satisfied and validated

This epic establishes the critical foundation for the entire Market Regime Master Framework enhancement while addressing all critical gaps identified in the PO validation report.
