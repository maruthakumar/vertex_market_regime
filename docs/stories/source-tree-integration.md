# Source Tree Integration

## Existing Project Structure
```
/Users/maruth/projects/market_regime/
├── backtester_v2/
│   ├── ui-centralized/
│   │   ├── strategies/
│   │   │   ├── market_regime/                    # Main enhancement location
│   │   │   │   ├── comprehensive_modules/        # Existing implementation  
│   │   │   │   ├── enhanced_modules/            # Current enhanced modules
│   │   │   │   ├── core/                        # Existing core logic
│   │   │   │   ├── indicators/                  # Existing indicators
│   │   │   │   └── config/                      # Excel configuration system
│   │   │   └── other_strategies/                # Other trading strategies
│   │   ├── configurations/                      # Configuration management
│   │   └── api/                                 # Existing API layer
│   └── docs/                                    # Comprehensive documentation
├── docs/
│   └── market_regime/                           # Master framework docs
└── web-bundles/                                 # Deployment automation
```

## New File Organization
```
/Users/maruth/projects/market_regime/
├── backtester_v2/
│   ├── ui-centralized/
│   │   ├── strategies/
│   │   │   ├── market_regime/
│   │   │   │   ├── adaptive_learning/           # NEW: 8-component system
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── orchestrator.py          # AdaptiveLearningOrchestrator
│   │   │   │   │   ├── components/              # 8 adaptive components
│   │   │   │   │   │   ├── component1_triple_straddle.py
│   │   │   │   │   │   ├── component2_greeks_sentiment.py  
│   │   │   │   │   │   ├── component3_oi_pa_trending.py
│   │   │   │   │   │   ├── component4_iv_skew.py
│   │   │   │   │   │   ├── component5_atr_ema_cpr.py
│   │   │   │   │   │   ├── component6_correlation_framework.py
│   │   │   │   │   │   ├── component7_support_resistance.py
│   │   │   │   │   │   └── component8_dte_master.py
│   │   │   │   │   ├── ml_integration/          # Vertex AI integration
│   │   │   │   │   │   ├── vertex_ai_client.py
│   │   │   │   │   │   ├── feature_engineering.py
│   │   │   │   │   │   ├── model_training.py
│   │   │   │   │   │   └── prediction_pipeline.py
│   │   │   │   │   ├── learning/                # Adaptive learning logic
│   │   │   │   │   │   ├── weight_optimizer.py
│   │   │   │   │   │   ├── performance_tracker.py
│   │   │   │   │   │   └── market_structure_detector.py
│   │   │   │   │   └── data_models/             # New data models
│   │   │   │   │       ├── analysis_results.py
│   │   │   │   │       ├── adaptive_weights.py  
│   │   │   │   │       └── master_classification.py
│   │   │   │   ├── enhanced_api/                # NEW: Enhanced API layer
│   │   │   │   │   ├── v2_endpoints.py
│   │   │   │   │   ├── health_monitoring.py
│   │   │   │   │   └── weight_management.py
│   │   │   │   ├── comprehensive_modules/       # Existing (preserved)
│   │   │   │   ├── enhanced_modules/            # Existing (preserved)
│   │   │   │   └── core/                        # Enhanced existing
│   │   │   └── configurations/                  # Enhanced config system
│   │   │       ├── vertex_ai/                   # NEW: Vertex AI configs
│   │   │       └── adaptive_learning/           # NEW: ML hyperparameters
│   └── cloud_deployment/                        # NEW: Cloud deployment
│       ├── terraform/                           # Infrastructure as code
│       ├── kubernetes/                          # Container orchestration
│       └── monitoring/                          # Cloud monitoring
```

## Integration Guidelines

- **File Naming:** Follow existing snake_case convention with descriptive prefixes (adaptive_, component_, ml_)
- **Folder Organization:** Mirror existing structure with new directories for enhanced functionality, preserve all existing paths
- **Import/Export Patterns:** Maintain existing import patterns, add new module imports with clear namespacing (from adaptive_learning.components import ...)
