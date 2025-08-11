# Source Tree Structure Documentation

## Overview
This document defines the comprehensive source tree structure for the Market Regime Master Framework, including existing architecture, new modular organization, and integration guidelines.

## Project Architecture Paradigm

The Market Regime Master Framework follows a **modular vertex_market_regime architecture** with clean separation from legacy systems while maintaining backward compatibility.

```
┌─────────────────────────────────────────────────────────────┐
│              Source Tree Architecture v2.0                  │
│           Modular • Scalable • Cloud-Native                │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
      ┌─────────────────┐ ┌─────────────┐ ┌─────────────────┐
      │  Legacy Bridge  │ │  Core New   │ │  Cloud Deploy   │
      │                 │ │  System     │ │                 │
      │ • backtester_v2 │ │ • vertex_mr │ │ • Infrastructure│
      │ • Preserved     │ │ • 8 Comps   │ │ • Monitoring    │
      │ • Compatible    │ │ • Clean     │ │ • Automation    │
      └─────────────────┘ └─────────────┘ └─────────────────┘
```

## Root Directory Structure

```
/Users/maruth/projects/market_regime/
├── 📁 vertex_market_regime/           # NEW: Core new system (clean implementation)
│   ├── 📁 src/                        # Source code
│   ├── 📁 configs/                    # Configuration management
│   ├── 📁 tests/                      # Test suites
│   ├── 📁 deployment/                 # Cloud deployment
│   ├── 📁 docs/                       # Technical documentation
│   └── 📁 scripts/                    # Utility scripts
├── 📁 backtester_v2/                  # EXISTING: Legacy system (preserved)
│   ├── 📁 ui-centralized/             # Existing UI system
│   └── 📁 docs/                       # Existing documentation
├── 📁 docs/                           # ENHANCED: Master documentation
│   ├── 📁 architecture/               # Architecture specifications
│   ├── 📁 stories/                    # Implementation stories
│   └── 📁 market_regime/              # Component specifications
├── 📁 infrastructure/                 # NEW: Infrastructure as code
│   ├── 📁 terraform/                  # GCP resource definitions
│   └── 📁 deployment-guide.md         # Deployment documentation
└── 📁 web-bundles/                    # EXISTING: Agent automation
    ├── 📁 agents/                     # BMAD agents
    └── 📁 teams/                      # Agent teams
```

## Core New System: vertex_market_regime/

This is the primary implementation directory for the new 8-component adaptive learning system.

```
vertex_market_regime/
├── 📄 README.md                       # Project overview and setup
├── 📄 setup.py                        # Python package configuration
├── 📄 requirements.txt                # Python dependencies
├── 📄 pyproject.toml                  # Modern Python project config
├── 📁 src/                            # Main source code
│   ├── 📁 vertex_market_regime/       # Main Python package
│   │   ├── 📄 __init__.py
│   │   ├── 📁 components/             # 8-component system
│   │   │   ├── 📄 __init__.py
│   │   │   ├── 📄 base_component.py   # Abstract base class
│   │   │   ├── 📁 component_01_triple_straddle/
│   │   │   │   ├── 📄 __init__.py
│   │   │   │   ├── 📄 straddle_analyzer.py
│   │   │   │   ├── 📄 correlation_matrix.py
│   │   │   │   └── 📄 dte_optimizer.py
│   │   │   ├── 📁 component_02_greeks_sentiment/
│   │   │   │   ├── 📄 __init__.py
│   │   │   │   ├── 📄 greeks_analyzer.py       # CRITICAL: gamma_weight=1.5
│   │   │   │   ├── 📄 sentiment_classifier.py
│   │   │   │   └── 📄 volume_weighter.py
│   │   │   ├── 📁 component_03_oi_pa_trending/
│   │   │   │   ├── 📄 __init__.py
│   │   │   │   ├── 📄 oi_analyzer.py
│   │   │   │   ├── 📄 institutional_flow.py
│   │   │   │   └── 📄 divergence_detector.py
│   │   │   ├── 📁 component_04_iv_skew/
│   │   │   │   ├── 📄 __init__.py
│   │   │   │   ├── 📄 skew_analyzer.py
│   │   │   │   ├── 📄 dual_dte_framework.py
│   │   │   │   └── 📄 regime_classifier.py
│   │   │   ├── 📁 component_05_atr_ema_cpr/
│   │   │   │   ├── 📄 __init__.py
│   │   │   │   ├── 📄 dual_asset_analyzer.py
│   │   │   │   ├── 📄 technical_indicators.py
│   │   │   │   └── 📄 confluence_detector.py
│   │   │   ├── 📁 component_06_correlation/
│   │   │   │   ├── 📄 __init__.py
│   │   │   │   ├── 📄 correlation_matrix_30x30.py  # GPU required
│   │   │   │   ├── 📄 breakdown_detector.py
│   │   │   │   └── 📄 feature_optimizer.py        # 774 features
│   │   │   ├── 📁 component_07_support_resistance/
│   │   │   │   ├── 📄 __init__.py
│   │   │   │   ├── 📄 level_detector.py
│   │   │   │   ├── 📄 confluence_analyzer.py
│   │   │   │   └── 📄 breakout_predictor.py
│   │   │   └── 📁 component_08_master_integration/
│   │   │       ├── 📄 __init__.py
│   │   │       ├── 📄 regime_orchestrator.py
│   │   │       ├── 📄 weight_optimizer.py
│   │   │       └── 📄 classification_engine.py
│   │   ├── 📁 data/                   # Data processing pipeline
│   │   │   ├── 📄 __init__.py
│   │   │   ├── 📄 parquet_manager.py  # Primary data interface
│   │   │   ├── 📄 arrow_processor.py  # Memory-efficient processing
│   │   │   ├── 📄 gpu_accelerator.py  # RAPIDS cuDF integration
│   │   │   └── 📁 schemas/             # Data schemas
│   │   │       ├── 📄 market_data.py
│   │   │       ├── 📄 options_data.py
│   │   │       └── 📄 analysis_results.py
│   │   ├── 📁 cloud/                  # Google Cloud integration
│   │   │   ├── 📄 __init__.py
│   │   │   ├── 📄 vertex_ai_client.py # Vertex AI integration
│   │   │   ├── 📄 gcs_manager.py      # Cloud Storage
│   │   │   ├── 📄 bigquery_client.py  # Analytics only
│   │   │   └── 📄 monitoring.py       # Cloud monitoring
│   │   ├── 📁 ml/                     # Machine learning pipeline
│   │   │   ├── 📄 __init__.py
│   │   │   ├── 📄 feature_engineering.py  # 774 features
│   │   │   ├── 📄 model_training.py       # Vertex AI training
│   │   │   ├── 📄 prediction_pipeline.py  # Inference pipeline
│   │   │   └── 📄 adaptive_weights.py     # Weight optimization
│   │   ├── 📁 api/                    # REST API endpoints
│   │   │   ├── 📄 __init__.py
│   │   │   ├── 📄 main.py             # FastAPI application
│   │   │   ├── 📄 v2_endpoints.py     # Enhanced endpoints
│   │   │   ├── 📄 health.py           # Health monitoring
│   │   │   └── 📄 middleware.py       # Authentication, logging
│   │   ├── 📁 legacy_bridge/          # Integration with existing system
│   │   │   ├── 📄 __init__.py
│   │   │   ├── 📄 backtester_integration.py
│   │   │   ├── 📄 excel_config_bridge.py
│   │   │   └── 📄 compatibility_layer.py
│   │   └── 📁 utils/                  # Utility functions
│   │       ├── 📄 __init__.py
│   │       ├── 📄 performance_monitor.py
│   │       ├── 📄 logging_config.py
│   │       └── 📄 validation.py
│   └── 📁 __pycache__/                # Python cache (ignored)
├── 📁 configs/                        # Configuration management
│   ├── 📄 README.md
│   ├── 📁 excel/                      # Excel configuration files
│   │   ├── 📄 MR_CONFIG_REGIME_1.0.0.xlsx
│   │   ├── 📄 MR_CONFIG_STRATEGY_1.0.0.xlsx
│   │   ├── 📄 MR_CONFIG_OPTIMIZATION_1.0.0.xlsx
│   │   ├── 📄 MR_CONFIG_PORTFOLIO_1.0.0.xlsx
│   │   └── 📄 excel_parser.py
│   ├── 📁 yaml/                       # YAML configurations
│   │   ├── 📄 components.yaml         # Component configurations
│   │   ├── 📄 performance.yaml        # Performance targets
│   │   ├── 📄 vertex_ai.yaml          # ML configurations
│   │   └── 📄 deployment.yaml         # Deployment settings
│   └── 📁 templates/                  # Configuration templates
│       ├── 📄 component_template.yaml
│       └── 📄 deployment_template.yaml
├── 📁 tests/                          # Comprehensive test suite
│   ├── 📄 README.md
│   ├── 📄 conftest.py                 # Pytest configuration
│   ├── 📁 unit/                       # Unit tests
│   │   ├── 📁 components/             # Component-specific tests
│   │   │   ├── 📄 test_component_01.py
│   │   │   ├── 📄 test_component_02.py  # Gamma weight validation
│   │   │   ├── 📄 test_component_03.py
│   │   │   ├── 📄 test_component_04.py
│   │   │   ├── 📄 test_component_05.py
│   │   │   ├── 📄 test_component_06.py  # GPU testing
│   │   │   ├── 📄 test_component_07.py
│   │   │   └── 📄 test_component_08.py
│   │   ├── 📁 data/
│   │   │   ├── 📄 test_parquet_manager.py
│   │   │   ├── 📄 test_arrow_processor.py
│   │   │   └── 📄 test_gpu_accelerator.py
│   │   ├── 📁 ml/
│   │   │   ├── 📄 test_feature_engineering.py  # 774 features
│   │   │   ├── 📄 test_model_training.py
│   │   │   └── 📄 test_adaptive_weights.py
│   │   └── 📁 api/
│   │       ├── 📄 test_v2_endpoints.py
│   │       └── 📄 test_health.py
│   ├── 📁 integration/                # Integration tests
│   │   ├── 📄 test_8_component_pipeline.py
│   │   ├── 📄 test_vertex_ai_integration.py
│   │   ├── 📄 test_performance_targets.py
│   │   └── 📄 test_legacy_compatibility.py
│   └── 📁 fixtures/                   # Test data and fixtures
│       ├── 📄 market_data_samples.py
│       ├── 📄 component_configs.py
│       └── 📁 sample_data/
│           ├── 📄 NIFTY_sample.parquet
│           └── 📄 BANKNIFTY_sample.parquet
├── 📁 deployment/                     # Cloud deployment resources
│   ├── 📄 README.md
│   ├── 📁 docker/                     # Container definitions
│   │   ├── 📄 Dockerfile.api          # API service container
│   │   ├── 📄 Dockerfile.worker       # ML worker container
│   │   ├── 📄 Dockerfile.gpu          # GPU processing container
│   │   └── 📄 docker-compose.yml      # Local development
│   ├── 📁 kubernetes/                 # K8s manifests
│   │   ├── 📄 namespace.yaml
│   │   ├── 📄 api-deployment.yaml
│   │   ├── 📄 worker-deployment.yaml
│   │   ├── 📄 gpu-deployment.yaml
│   │   ├── 📄 configmap.yaml
│   │   ├── 📄 secrets.yaml
│   │   └── 📄 ingress.yaml
│   ├── 📁 terraform/                  # Infrastructure as code
│   │   ├── 📄 main.tf                 # Main Terraform config
│   │   ├── 📄 variables.tf            # Variable definitions
│   │   ├── 📄 outputs.tf              # Output values
│   │   ├── 📄 vertex_ai.tf            # Vertex AI resources
│   │   ├── 📄 gcs.tf                  # Cloud Storage
│   │   ├── 📄 gke.tf                  # Kubernetes cluster
│   │   └── 📄 monitoring.tf           # Monitoring setup
│   └── 📁 helm/                       # Helm charts
│       ├── 📄 Chart.yaml
│       ├── 📄 values.yaml
│       └── 📁 templates/
│           ├── 📄 deployment.yaml
│           ├── 📄 service.yaml
│           └── 📄 ingress.yaml
├── 📁 docs/                           # Technical documentation
│   ├── 📄 README.md
│   ├── 📁 api/                        # API documentation
│   │   ├── 📄 endpoints.md
│   │   ├── 📄 authentication.md
│   │   └── 📄 examples.md
│   ├── 📁 architecture/               # Architecture documentation
│   │   ├── 📄 overview.md
│   │   ├── 📄 data_pipeline.md
│   │   ├── 📄 ml_architecture.md
│   │   └── 📄 performance.md
│   ├── 📁 components/                 # Component documentation
│   │   ├── 📄 component_01_guide.md
│   │   ├── 📄 component_02_guide.md   # Gamma weight documentation
│   │   ├── 📄 component_03_guide.md
│   │   ├── 📄 component_04_guide.md
│   │   ├── 📄 component_05_guide.md
│   │   ├── 📄 component_06_guide.md   # GPU requirements
│   │   ├── 📄 component_07_guide.md
│   │   └── 📄 component_08_guide.md
│   ├── 📁 deployment/                 # Deployment guides
│   │   ├── 📄 local_setup.md
│   │   ├── 📄 cloud_deployment.md
│   │   └── 📄 monitoring_setup.md
│   └── 📁 migration/                  # Migration documentation
│       ├── 📄 heavydb_to_parquet.md
│       └── 📄 legacy_integration.md
├── 📁 monitoring/                     # Monitoring and observability
│   ├── 📁 grafana/                    # Grafana dashboards
│   │   ├── 📄 performance_dashboard.json
│   │   ├── 📄 component_health.json
│   │   └── 📄 ml_metrics.json
│   ├── 📁 prometheus/                 # Prometheus configuration
│   │   ├── 📄 prometheus.yml
│   │   ├── 📄 alerts.yml
│   │   └── 📄 recording_rules.yml
│   └── 📁 logging/                    # Logging configuration
│       ├── 📄 fluentd.conf
│       └── 📄 log_parsing.yaml
├── 📁 scripts/                        # Utility and automation scripts
│   ├── 📄 README.md
│   ├── 📄 setup_environment.sh        # Environment setup
│   ├── 📄 validate_structure.py       # Structure validation
│   ├── 📄 migrate_data.py             # Data migration
│   ├── 📄 deploy.sh                   # Deployment automation
│   └── 📄 benchmark.py                # Performance benchmarking
└── 📄 validation_report.md            # Project validation report
```

## Legacy System Integration: backtester_v2/

The existing backtester system is preserved and enhanced with integration points to the new system.

```
backtester_v2/                        # EXISTING: Preserved legacy system
├── 📁 ui-centralized/                 # Existing UI system
│   ├── 📁 strategies/
│   │   ├── 📁 market_regime/          # Legacy market regime implementation
│   │   │   ├── 📄 market_regime_strategy.py    # Existing implementation
│   │   │   ├── 📄 enhanced_integration.py      # NEW: Integration bridge
│   │   │   └── 📁 vertex_integration/          # NEW: Bridge to new system
│   │   │       ├── 📄 __init__.py
│   │   │       ├── 📄 api_bridge.py            # API integration
│   │   │       ├── 📄 data_bridge.py           # Data transformation
│   │   │       └── 📄 compatibility_layer.py   # Backward compatibility
│   │   ├── 📁 configurations/         # Configuration management system
│   │   │   ├── 📁 excel_integration/  # Excel bridge to new system
│   │   │   │   ├── 📄 excel_to_vertex.py
│   │   │   │   └── 📄 parameter_mapper.py
│   │   │   └── 📁 vertex_configs/     # NEW: Vertex AI configurations
│   │   │       ├── 📄 component_configs.yaml
│   │   │       └── 📄 ml_hyperparams.yaml
│   │   └── 📁 api/                    # Enhanced API layer
│   │       ├── 📄 enhanced_endpoints.py        # NEW: Enhanced endpoints
│   │       └── 📄 vertex_proxy.py              # NEW: Proxy to new system
│   └── 📁 enhanced_monitoring/        # NEW: Enhanced monitoring
│       ├── 📄 performance_tracker.py
│       └── 📄 component_health.py
└── 📁 docs/                          # Existing comprehensive documentation
    ├── 📄 README.md                  # Updated with integration info
    └── 📁 vertex_integration/        # NEW: Integration documentation
        ├── 📄 migration_guide.md
        └── 📄 api_integration.md
```

## Documentation Structure: docs/

Enhanced documentation structure with comprehensive architecture and component specifications.

```
docs/                                  # ENHANCED: Master documentation
├── 📁 architecture/                   # NEW: Architecture specifications
│   ├── 📄 coding-standards.md         # Coding standards and conventions
│   ├── 📄 tech-stack.md               # Technology stack documentation
│   └── 📄 source-tree.md              # This document
├── 📁 stories/                        # Implementation stories and planning
│   ├── 📄 testing-strategy.md         # Testing strategy and requirements
│   ├── 📄 epic-1-feature-engineering-foundation.md
│   ├── 📄 epic-2-data-pipeline-modernization.md
│   ├── 📄 epic-3-system-integration-and-serving.md
│   └── 📄 epic-4-production-readiness.md
├── 📁 market_regime/                  # Component specifications
│   ├── 📄 mr_master_v1.md             # Master framework specification
│   ├── 📄 mr_tripple_rolling_straddle_component1.md
│   ├── 📄 mr_greeks_sentiment_analysis_component2.md
│   ├── 📄 mr_oi_pa_trending_analysis_component3.md
│   ├── 📄 mr_iv_skew_analysis_component4.md
│   ├── 📄 mr_atr_ema_cpr_component5.md
│   ├── 📄 mr_correlation_noncorelation_component6.md
│   ├── 📄 mr_support_resistance_component7.md
│   └── 📄 mr_dte_adaptive_overlay_component8.md
├── 📄 architecture.md                 # Main architecture document
├── 📄 prd.md                         # Product requirements document
└── 📄 MASTER_ARCHITECTURE_v2.md       # Definitive architecture specification
```

## Infrastructure Structure: infrastructure/

Infrastructure as code and deployment automation.

```
infrastructure/                       # NEW: Infrastructure management
├── 📁 terraform/                     # Terraform configurations
│   ├── 📄 main.tf                    # Main infrastructure
│   ├── 📄 variables.tf               # Variable definitions
│   ├── 📄 outputs.tf                 # Output definitions
│   ├── 📄 terraform.tfvars.example   # Example variables
│   ├── 📁 modules/                   # Terraform modules
│   │   ├── 📁 vertex_ai/             # Vertex AI module
│   │   ├── 📁 gcs/                   # Cloud Storage module
│   │   ├── 📁 gke/                   # Kubernetes module
│   │   └── 📁 monitoring/            # Monitoring module
│   └── 📁 environments/              # Environment-specific configs
│       ├── 📄 dev.tfvars
│       ├── 📄 staging.tfvars
│       └── 📄 production.tfvars
└── 📄 deployment-guide.md            # Deployment documentation
```

## Integration Guidelines

### File Naming Conventions
- **snake_case**: All Python files and directories
- **kebab-case**: Documentation and configuration files
- **Descriptive prefixes**: `component_`, `vertex_`, `ml_`, `gpu_`
- **Version suffixes**: `_v1`, `_v2` for versioned implementations

### Import Organization Standards
```python
# Standard library imports
import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

# Third-party imports
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from google.cloud import aiplatform
from sklearn.base import BaseEstimator

# Local application imports - absolute imports preferred
from vertex_market_regime.components.base_component import BaseComponent
from vertex_market_regime.data.parquet_manager import ParquetManager
from vertex_market_regime.ml.feature_engineering import FeatureEngineer
from vertex_market_regime.utils.performance_monitor import PerformanceMonitor

# Legacy bridge imports - clearly separated
from backtester_v2.ui_centralized.strategies.market_regime import market_regime_strategy
```

### Directory Organization Principles
1. **Separation of Concerns**: Each directory has a single, well-defined responsibility
2. **Modular Design**: Components are independently deployable and testable
3. **Clean Architecture**: Dependencies flow from outer layers to inner layers
4. **Configuration Centralization**: All configuration in dedicated directories
5. **Test Co-location**: Tests mirror the source structure for clarity

### Integration Patterns
- **Bridge Pattern**: Legacy system integration through dedicated bridge modules
- **Facade Pattern**: Simplified interfaces for complex subsystems
- **Strategy Pattern**: Pluggable component implementations
- **Observer Pattern**: Event-driven communication between components

### Deployment Structure
```python
DEPLOYMENT_STRUCTURE = {
    "local_development": {
        "primary_directory": "vertex_market_regime/",
        "data_source": "local_parquet_files",
        "gpu_acceleration": "optional",
        "services": ["api", "components", "monitoring"]
    },
    "cloud_deployment": {
        "primary_directory": "vertex_market_regime/",
        "data_source": "gcs_parquet",
        "gpu_acceleration": "required",
        "services": ["vertex_ai", "gke", "monitoring", "storage"]
    },
    "hybrid_deployment": {
        "legacy_system": "backtester_v2/",
        "new_system": "vertex_market_regime/", 
        "integration": "bridge_modules",
        "migration_strategy": "gradual_cutover"
    }
}
```

## Performance Considerations

### Directory Access Patterns
- **Hot paths**: Frequently accessed code in optimized locations
- **Cold storage**: Archived data and logs in cost-effective storage
- **Caching strategy**: Intermediate results cached appropriately
- **Memory mapping**: Large datasets memory-mapped when possible

### Scalability Architecture
```python
SCALABILITY_STRUCTURE = {
    "horizontal_scaling": {
        "component_level": "independent_scaling",
        "service_level": "kubernetes_pods",
        "data_level": "partitioned_parquet"
    },
    "vertical_scaling": {
        "compute_intensive": "gpu_nodes",
        "memory_intensive": "high_memory_nodes",
        "io_intensive": "ssd_storage"
    }
}
```

## Security Structure

### Security Boundaries
```
┌─────────────────────────────────────────────────────────────┐
│                    Security Architecture                     │
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   Public API    │  │  Internal Svc   │  │  Data Layer     │  │
│  │                 │  │                 │  │                 │  │
│  │ • Authentication│  │ • Service Mesh  │  │ • Encryption    │  │
│  │ • Rate Limiting │  │ • mTLS          │  │ • Access Control│  │
│  │ • Input Valid   │  │ • Network Pol   │  │ • Audit Logs    │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Secrets Management
- **Environment Variables**: Non-sensitive configuration
- **Google Secret Manager**: Sensitive credentials and API keys
- **Kubernetes Secrets**: Container-level secrets
- **Encryption at Rest**: All persistent data encrypted

This source tree structure provides a comprehensive, scalable, and maintainable foundation for the Market Regime Master Framework while preserving existing system investments and enabling smooth migration to the new cloud-native architecture.