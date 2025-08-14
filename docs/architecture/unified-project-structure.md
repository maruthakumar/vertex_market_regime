# Unified Project Structure (BMAD-Aligned)

Version: 1.1  
Date: 2025-08-12  
Last Updated: Added ML documentation directory structure

## Purpose
Define a consistent, BMAD-aligned structure for documents, code, infrastructure, and stories so planning and development agents can navigate and execute tasks predictably.

## High-Level Layout
```
/Users/maruth/projects/market_regime/
├── backtester_v2/                       # Legacy system (brownfield context)
│   └── ui-centralized/strategies/market_regime/
├── vertex_market_regime/                # New modular implementation (cloud-native)
│   ├── src/
│   ├── configs/
│   └── tests/
├── docs/
│   ├── prd.md                           # Product Requirements (canonical)
│   ├── architecture.md                  # Architecture (canonical)
│   ├── market_regime/                   # Component specifications (1..8)
│   ├── ml/                              # ML/Vertex AI specifications (CRITICAL)
│   │   ├── feature-store-spec.md        # Feature Store entity types, online features, TTLs
│   │   ├── offline-feature-tables.md    # BigQuery DDLs, table structures, partitioning
│   │   ├── training-pipeline-spec.md    # ML pipeline specifications
│   │   └── vertex-ai-setup.md           # GCP project config, IAM, setup commands
│   ├── stories/                         # Epics and stories (BMAD naming)
│   │   ├── epic-1-feature-engineering-foundation.md
│   │   ├── epic-2-data-pipeline-modernization.md
│   │   ├── epic-3-system-integration-and-serving.md
│   │   └── epic-4-production-readiness.md
│   ├── validation/                      # Evidence from gates/tests
│   ├── performance/                     # Baselines and reports
│   └── architecture/                    # Architecture sub-sections
│       ├── unified-project-structure.md # (this file)
│       └── testing-strategy.md          # Test strategy and gates
├── infrastructure/
│   ├── terraform/                       # IaC for GCP (Vertex AI, BQ, GCS)
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── terraform.tfvars.example
│   └── deployment-guide.md
└── web-bundles/                         # (If used) web agent bundles
```

## BMAD Document Conventions
- Canonical docs:
  - PRD at `docs/prd.md`
  - Architecture at `docs/architecture.md`
- Component specs under `docs/market_regime/` (1..8)
- ML/Vertex AI specs under `docs/ml/` (CRITICAL for Epic 2-3 implementation)
- Stories and epics under `docs/stories/` with lowercase, kebab-case filenames:
  - `epic-n-<name>.md`, `story-n-m-<name>.md` (if further sharding)
- Validation evidence under `docs/validation/`
- Performance baselines under `docs/performance/`

## ML Documentation (`docs/ml/`)
This directory contains critical specifications for Vertex AI and BigQuery implementation:

### Core ML Specifications
- **`feature-store-spec.md`**: Defines Vertex AI Feature Store configuration
  - Entity types: `instrument_minute` (minute-level) and `instrument_day` (daily aggregates)
  - Online feature set: 32 core features for low-latency serving (<50ms)
  - TTL policies: 48h for minute-level, 30d for daily
  - Entity ID format: `${symbol}_${yyyymmddHHMM}_${dte}`

- **`offline-feature-tables.md`**: BigQuery offline feature storage
  - Dataset: `market_regime_{env}` (dev/staging/prod)
  - Tables: `c1_features` through `c8_features` plus `training_dataset`
  - Partitioning: `DATE(ts_minute)` for efficient querying
  - Clustering: `symbol, dte` for performance optimization
  - Contains DDL examples and table structures

- **`vertex-ai-setup.md`**: GCP infrastructure configuration
  - Project ID: `arched-bot-269016`
  - Primary region: `us-central1`
  - Artifact Registry repo: `mr-ml`
  - IAM roles and service account requirements
  - Setup commands and deployment procedures

- **`training-pipeline-spec.md`**: ML pipeline specifications
  - KFP v2 (Vertex AI Pipelines) definitions
  - Model training and registration workflows
  - Reserved for Epic 3 implementation

### Usage Guidelines
- All Epic 2 stories MUST reference these specifications
- Developers should use these as source of truth for ML infrastructure
- Any updates to ML architecture must be reflected here
- These specs bridge Epic 1 (feature engineering) with Epic 2-3 (ML infrastructure)

## Sharding Guidelines
- Level-2 (`##`) headings can be sharded by agents when needed.
- Architecture subtopics reside in `docs/architecture/`. Examples:
  - `unified-project-structure.md` (this)
  - `testing-strategy.md`
  - Additional topics may include: `tech-stack.md`, `source-tree.md` (optional)
- Keep sections small and single-purpose to support SM/Dev task focus.

## Naming Standards
- Files (docs): lowercase, kebab-case: `epic-1-feature-engineering-foundation.md`
- Python modules: snake_case for files and identifiers
- Directories: snake_case or kebab-case consistently within a tree
- Components: prefix with `component_0n_` in code; human docs use "Component n: <name>"

## Source of Truth
- `docs/prd.md` and `docs/architecture.md` are the only canonical planning documents.
- Older or duplicate architecture files must be marked "superseded" or archived.

## Dual-Environment Abstraction
- Local path: Parquet → Arrow → RAPIDS (GPU)
- Cloud path: Parquet (GCS) → BigQuery (offline) + Vertex AI Feature Store (online) → Vertex AI (GPU)
- A single data access layer should abstract environment-specific differences.

## Status and Gates
- Epics define validation gates. Gate evidence is stored under `docs/validation/` with links from the relevant epic.

## Cross-Referencing
- Each epic should link to:
  - PRD/Architecture
  - Relevant component documents (1..8)
  - Test strategy and validation gates sections

## Change Management
- When structure changes, update this file and reference from `docs/architecture.md`.
- Prefer additive changes; mark deprecated paths explicitly.



