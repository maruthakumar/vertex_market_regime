# Market Regime Master Framework Brownfield Enhancement Architecture

> Decision Update (2025-08-10): HeavyDB is deprecated. End-state data platform is Parquet → Arrow → GPU. HeavyDB may be used read-only during migration and will be removed after cutover.

### Performance Targets Alignment
- Base (Brownfield): <800ms total processing, <3.7GB memory, >85–87% accuracy (aligned with PRD)
- Stretch (Cloud-native): <600ms total processing, <2.5GB memory, >87% accuracy (per `MASTER_ARCHITECTURE_v2.md`)

## Introduction

This document outlines the architectural approach for enhancing the Market Regime Master Framework project with an 8-component adaptive learning system integrated with Google Vertex AI. Its primary goal is to serve as the guiding architectural blueprint for AI-driven development of new features while ensuring seamless integration with the existing backtester_v2 system.

**Relationship to Existing Architecture:**
This document supplements the existing project architecture by defining how the new 8-component adaptive learning framework will integrate with the current HeavyDB-based backtester system. Where conflicts arise between new and existing patterns, this document provides guidance on maintaining consistency while implementing enhancements.

### Existing Project Analysis

**Current Project State:**
- **Primary Purpose:** Quantitative trading system with market regime classification for options strategies
- **Current Tech Stack:** Python 3.8+, HeavyDB, Pandas/cuDF, Excel-based configuration, REST APIs
- **Architecture Style:** Modular monolithic architecture with plugin-based strategy system
- **Deployment Method:** Local deployment with SSH server integration and HeavyDB infrastructure

**Available Documentation:**
- Market Regime Master Framework v1.0 specification with 8-component adaptive system
- Vertex AI PRD for cloud migration and ML enhancement
- BMAD orchestration documentation for deployment automation
- HeavyDB connection guides and performance optimization documentation
- Comprehensive strategy testing documentation across 31 Excel configuration sheets

**Identified Constraints:**
- Performance requirement: <800ms total processing time for 8-component analysis
- Memory constraint: <3.7GB total system memory usage
- Accuracy target: >85% regime classification accuracy
- Existing HeavyDB infrastructure must be preserved and integrated
- 600+ Excel configuration parameters must be maintained and mapped to ML hyperparameters
- Zero-downtime migration requirement for production trading systems

### Change Log

| Change | Date | Version | Description | Author |
|--------|------|---------|-------------|---------|
| Initial Architecture | 2025-08-10 | 1.0 | Created comprehensive brownfield architecture for 8-component adaptive learning system | Claude Code |

## Enhancement Scope and Integration Strategy

**Enhancement Type:** Major system enhancement with ML integration and cloud deployment
**Scope:** Integration of 8-component adaptive learning framework with existing backtester_v2 system and HeavyDB infrastructure
**Integration Impact:** Medium - Significant new functionality while preserving existing system operations

### Integration Approach

**Code Integration Strategy:** Modular enhancement with backward compatibility - new 8-component framework will be implemented as enhanced modules within the existing `strategies/market_regime/` directory structure, maintaining all existing API contracts while adding new ML-enhanced capabilities.

**Database Integration:** Parquet-first architecture on GCS with Arrow in-memory processing and RAPIDS/cuDF. HeavyDB is deprecated and allowed only as a temporary, read-only migration source; end-state removes HeavyDB entirely. Optional BigQuery is used for analytics/reporting.

**API Integration:** Extension of existing REST API framework with new endpoints for 8-component analysis while maintaining full backward compatibility with current trading system integrations.

**UI Integration:** Enhancement of existing UI framework in `backtester_v2/ui-centralized/` with new dashboards for component monitoring and adaptive learning visualization, preserving all existing strategy configuration interfaces.

### Compatibility Requirements

- **Existing API Compatibility:** 100% backward compatibility maintained for all existing endpoints
- **Database Schema Compatibility:** HeavyDB schema preserved with optional new tables for ML metadata
- **UI/UX Consistency:** Enhanced dashboards follow existing design patterns and navigation structure
- **Performance Impact:** New system must not degrade existing backtester performance (<3 second regime analysis maintained)

## Tech Stack Alignment

### Existing Technology Stack

| Category | Current Technology | Version | Usage in Enhancement | Notes |
|----------|-------------------|---------|---------------------|-------|
| Database | GCS Parquet + Arrow | Latest | Primary storage + in-memory processing | HeavyDB deprecated (migration-only) |
| Language | Python | 3.8+ | All component implementations | Leverage existing codebase |
| Data Processing | RAPIDS cuDF + Apache Arrow | Latest | Feature engineering + analysis | GPU acceleration primary |
| Web Framework | FastAPI/Flask | Latest | API endpoints | Extend existing REST framework |
| Configuration | Excel + YAML | N/A | Parameter management | Excel sheets mapped to ML hyperparameters |
| Monitoring | Custom logging | N/A | Performance tracking | Enhanced with ML metrics |
| Deployment | SSH + tmux | N/A | Local orchestration | Augmented with BMAD automation |

### New Technology Additions

| Technology | Version | Purpose | Rationale | Integration Method |
|------------|---------|---------|-----------|-------------------|
| Google Vertex AI | Latest | ML model training/serving | Scalable ML infrastructure for adaptive learning | API integration with existing system |
| Google BigQuery | Latest | ML data warehouse | Structured data for model training | ETL pipeline from HeavyDB |
| Google Cloud Storage | Latest | Model artifacts + data pipeline | Durable storage for ML assets | Batch data transfer and streaming |
| scikit-learn | 1.3+ | Feature engineering + validation | Proven ML library for ensemble models | Python integration |
| TensorFlow/PyTorch | Latest | Deep learning components | Advanced pattern recognition in market regimes | Optional for neural network components |

## Data Models and Schema Changes

### New Data Models

#### **ComponentAnalysisResult**
**Purpose:** Store results from individual component analysis for adaptive learning feedback
**Integration:** BigQuery table in dataset `market_regime_{environment}`, with reference keys to options_chain (migration-only). End-state uses GCS Parquet → Vertex AI Pipelines for feature engineering and Vertex AI Feature Store for online features.

**Key Attributes:**
- `analysis_id`: UUID - Unique identifier for each analysis run
- `component_id`: INT - Component identifier (1-8)
- `timestamp`: TIMESTAMP - Analysis execution time
- `regime_prediction`: STRING - Individual component regime prediction
- `confidence_score`: FLOAT - Component confidence (0.0-1.0)
- `processing_time_ms`: INT - Component processing time
- `weight_factor`: FLOAT - Current adaptive weight for this component

**Relationships:**
- **With Existing:** Links to existing options_chain data via trade_date and symbol
- **With New:** Aggregated by MasterRegimeAnalysis for final classification

#### **AdaptiveLearningWeights**
**Purpose:** Store historical weight evolution for each component's adaptive learning system
**Integration:** BigQuery time-series table for weight optimization and performance tracking

**Key Attributes:**
- `weight_id`: UUID - Unique weight record identifier
- `component_id`: INT - Component being weighted
- `dte_bucket`: STRING - DTE range (0-7, 8-30, 31+)
- `regime_context`: STRING - Market regime when weight was applied
- `weight_value`: FLOAT - Adaptive weight value
- `performance_metric`: FLOAT - Historical performance score
- `last_updated`: TIMESTAMP - Weight update timestamp

**Relationships:**
- **With Existing:** Performance metrics calculated against actual market moves
- **With New:** Used by ComponentAnalysisResult for real-time weight application

#### **MasterRegimeAnalysis**
**Purpose:** Final integrated analysis results from all 8 components with master classification
**Integration:** BigQuery primary results table for the enhanced 8-component system

**Key Attributes:**
- `analysis_id`: UUID - Links to individual component analyses
- `symbol`: STRING - Asset symbol (NIFTY, BANKNIFTY, etc.)
- `timestamp`: TIMESTAMP - Analysis timestamp
- `master_regime`: STRING - Final regime classification (LVLD, HVC, VCPE, etc.)
- `master_confidence`: FLOAT - Overall system confidence
- `component_agreement`: FLOAT - Inter-component correlation score
- `processing_time_total_ms`: INT - Total 8-component processing time
- `vertex_ai_model_version`: STRING - ML model version used

**Relationships:**
- **With Existing:** Replaces/enhances current regime classification in trading decisions
- **With New:** Aggregates all ComponentAnalysisResult records per analysis cycle

### Schema Integration Strategy

**Database Changes Required:**
- **New Tables:** component_analysis_results, adaptive_learning_weights, master_regime_analysis, ml_model_metadata
- **Modified Tables:** Enhanced options_chain with regime_analysis_id foreign key (optional)
- **New Indexes:** Time-based indexes on component results, symbol-based clustering for performance
- **Migration Strategy:** Progressive schema deployment with fallback compatibility

**Backward Compatibility:**
- All existing tables and schemas remain unchanged
- New tables use separate namespace to avoid conflicts  
- Existing regime classification API maintains current response format with optional enhancement fields

## Component Architecture

### Component Specification Links
- [Component 1: Triple Rolling Straddle](market_regime/mr_tripple_rolling_straddle_component1.md)
- [Component 2: Greeks Sentiment Analysis](market_regime/mr_greeks_sentiment_analysis_component2.md)
- [Component 3: OI-PA Trending Analysis](market_regime/mr_oi_pa_trending_analysis_component3.md)
- [Component 4: IV Skew Analysis](market_regime/mr_iv_skew_analysis_component4.md)
- [Component 5: ATR-EMA-CPR Integration](market_regime/mr_atr_ema_cpr_component5.md)
- [Component 6: Correlation & Non-Correlation Framework](market_regime/mr_correlation_noncorelation_component6.md)
- [Component 7: Support/Resistance Formation Logic](market_regime/mr_support_resistance_component7.md)
- [Component 8: DTE-Adaptive Master Integration](market_regime/mr_dte_adaptive_overlay_component8.md)

#### Execution Environments
- Local: Arrow + RAPIDS/cuDF on a local GPU host (no BigQuery in request path)
- Cloud: Vertex AI CustomJobs (GPU) or GKE GPU nodes run Arrow + RAPIDS for feature engineering and component processing; BigQuery is used for offline analytics/training datasets; online features come from Vertex AI Feature Store

#### Training/Serving Parity & Feature Versioning
- All feature transforms packaged in the model/runtime environment
- Vertex AI Feature Store provides online features with the same definitions as training
- Feature definitions and schemas are versioned; backfills are required on changes

#### Vertex AI Feature Engineering (All Components) - REQUIRED
- Engineer all 774 features via Vertex AI Pipelines
- Store/serve features via Vertex AI Feature Store with strict training/serving parity
- Data source: GCS Parquet; processing: Apache Arrow + RAPIDS cuDF
- Maintain feature schema versioning and lineage for training and inference

### New Components

#### **AdaptiveLearningOrchestrator**
**Responsibility:** Coordinates the 8-component analysis pipeline, manages adaptive weight updates, and ensures performance targets are met
**Integration Points:** Integrates with existing market_regime_strategy.py and HeavyDB data pipeline

**Key Interfaces:**
- `analyze_market_regime(market_data) -> MasterRegimeResult`
- `update_component_weights(performance_feedback) -> WeightUpdateResult` 
- `get_system_health_metrics() -> SystemHealthMetrics`

**Dependencies:**
- **Existing Components:** HeavyDB connection, market data pipeline, existing regime classification
- **New Components:** All 8 adaptive components, Vertex AI integration layer

**Technology Stack:** Python 3.8+, asyncio for concurrent component processing, Redis for caching

#### **Component 1: Enhanced 10-Component Triple Rolling Straddle System** 
**Responsibility:** Advanced symmetric straddle analysis (ATM/ITM1/OTM1) with rolling straddle overlay technical analysis and DTE-specific historical learning
**Integration Points:** Revolutionary application of EMA/VWAP/Pivot analysis to ROLLING STRADDLE PRICES (not underlying prices)

**Key Features:**
- **🚨 CRITICAL PARADIGM SHIFT**: EMA/VWAP/Pivots applied to ROLLING STRADDLE PRICES, not underlying
- **10-Component Dynamic Weighting**: ATM/ITM1/OTM1 straddles + individual CE/PE + correlation analysis
- **DTE-Specific Historical Learning**: Separate optimization for each DTE (0-90) with dual learning modes
- **Multi-Timeframe Integration**: 3min/5min/10min/15min analysis with dynamic weight optimization
- **120 Expert Features**: Comprehensive technical analysis applied to options pricing

**Key Interfaces:**
- `analyze_symmetric_straddles(market_data, dte) -> TripleStraddleResult`
- `apply_rolling_straddle_technical_analysis(straddle_prices) -> TechnicalScores`
- `learn_dte_specific_weights(dte, performance_history) -> OptimalWeights`
- `integrate_multi_timeframe_analysis(timeframe_data) -> WeightedScore`

**Performance Targets:**
- Processing Time: <150ms
- Memory Usage: <350MB
- Feature Count: 120 features
- Accuracy Target: >85% regime classification contribution

**Technology Stack:** Rolling straddle price extraction, EMA/VWAP/Pivot analysis on straddle prices, DTE-specific weight learning, multi-timeframe correlation

#### **Component 2: Advanced Greeks Sentiment Analysis System**
**Responsibility:** Volume-weighted Greeks analysis with second-order Greeks integration and adaptive sentiment threshold learning
**Integration Points:** Revolutionary Greeks behavioral fingerprint analysis for institutional sentiment detection

**🚨 CRITICAL FIX IMPLEMENTED**: 
- **Gamma Weight CORRECTED**: From 0.0 (WRONG) → 1.5 (CORRECT) - Highest priority for pin risk detection
- **Second-Order Greeks**: Vanna, Charm, Volga integration for institutional-grade analysis
- **Volume-Weighted Analysis**: Symbol-specific volume threshold learning (NIFTY: 15k+, BANKNIFTY: 22k+)
- **DTE-Specific Adjustments**: Granular DTE=0,1,2... specific multipliers for precise regime detection

**Key Features:**
- **Adaptive Greeks Weight Learning**: Historical performance-based weight optimization
- **Symbol-Specific Calibration**: Different behavior patterns for NIFTY vs BANKNIFTY vs Stocks
- **7-Level Sentiment Classification**: Complete sentiment spectrum with adaptive thresholds
- **Institutional Flow Detection**: Volume-weighted Greeks for large player identification
- **98 Expert Features**: Comprehensive first and second-order Greeks analysis

**Key Interfaces:**
- `analyze_volume_weighted_greeks(options_data, straddle_strikes) -> GreeksAnalysis`
- `learn_optimal_greeks_weights(symbol, dte, learning_mode) -> AdaptiveWeights`
- `classify_sentiment_with_adaptive_thresholds(greeks_score) -> SentimentLevel`
- `detect_institutional_flow(volume_data, oi_data) -> FlowAnalysis`

**Performance Targets:**
- Processing Time: <120ms  
- Memory Usage: <250MB
- Feature Count: 98 features
- Sentiment Accuracy: >88%
- Pin Risk Detection: >92% (DTE 0-3)

**Technology Stack:** Advanced Greeks calculations, volume weighting algorithms, adaptive threshold learning, second-order Greeks estimation

#### **Component 3: OI-PA Trending Analysis with Institutional Intelligence**
**Responsibility:** Cumulative ATM ±7 strikes analysis with rolling timeframe institutional flow detection and adaptive threshold learning
**Integration Points:** Revolutionary multi-strike cumulative analysis for institutional position detection

**Key Features:**
- **Cumulative ATM ±7 Analysis**: ATM, ±1, ±2, ±3, ±4, ±5, ±6, ±7 strikes cumulative OI/Volume tracking
- **Rolling Timeframe Analysis**: 5min (35%) + 15min (20%) + other timeframes with adaptive weighting
- **Institutional Flow Detection**: CE vs PE flow patterns with learned thresholds
- **Strike Range Expansion**: Dynamic expansion from ±7 to ±15 based on volatility and performance
- **5 Divergence Types**: Bull/Bear/Hidden/Regular/Extreme divergence classification
- **105 Expert Features**: Comprehensive OI and PA pattern recognition

**Key Interfaces:**
- `analyze_cumulative_atm_strikes(oi_data, strike_range=7) -> CumulativeAnalysis`
- `detect_institutional_flow(ce_flow, pe_flow, timeframes) -> FlowSignal`
- `classify_divergence_patterns(oi_trends, price_trends) -> DivergenceTypes`
- `adapt_strike_range(volatility, performance) -> OptimalRange`

**Performance Targets:**
- Processing Time: <200ms
- Memory Usage: <400MB 
- Feature Count: 105 features
- Institutional Detection: >82%
- Flow Prediction Accuracy: >85%

**Technology Stack:** Multi-strike cumulative analysis, rolling timeframe processing, institutional flow algorithms, adaptive range optimization

#### **Component 4: IV Skew Analysis with Dual DTE Framework**
**Responsibility:** Advanced IV skew analysis with specific DTE + DTE range learning and 7-level volatility regime classification
**Integration Points:** Dual DTE framework combining granular and categorical volatility analysis

**Key Features:**
- **Dual DTE Framework**: Specific DTE (dte=0,1,2...) AND DTE ranges (weekly/monthly/far) analysis
- **7-Level IV Regime Classification**: Complete volatility regime spectrum with adaptive thresholds
- **Put/Call Skew Analysis**: Separate analysis of put skew vs call skew patterns
- **Term Structure Intelligence**: Cross-DTE volatility structure analysis
- **Percentile Optimization**: Dynamic percentile thresholds based on market conditions
- **87 Expert Features**: Comprehensive volatility pattern recognition

**Key Interfaces:**
- `analyze_dual_dte_iv_skew(iv_surface, specific_dte, dte_range) -> SkewAnalysis`
- `classify_iv_regime_7_levels(skew_data) -> IVRegimeLevel`
- `calculate_put_call_skew_differential(put_iv, call_iv) -> SkewDifferential`
- `optimize_percentile_thresholds(market_condition) -> AdaptiveThresholds`

**Performance Targets:**
- Processing Time: <200ms
- Memory Usage: <300MB
- Feature Count: 87 features  
- IV Pattern Recognition: >85%
- Regime Classification: >88%

**Technology Stack:** IV surface analysis, dual DTE learning, skew differential calculations, percentile optimization algorithms

#### **Component 5: ATR-EMA-CPR Integration with Dual Asset Analysis**
**Responsibility:** Revolutionary dual asset technical analysis comparing straddle prices vs underlying with adaptive period optimization
**Integration Points:** Advanced technical indicator integration with options-specific enhancements

**Key Features:**
- **Dual Asset Analysis**: Parallel analysis of straddle prices AND underlying prices
- **ATR Period Optimization**: Dynamic 14/21/50 period weighting based on performance
- **EMA Timeframe Intelligence**: Daily/weekly/monthly EMA importance adaptation
- **CPR Method Selection**: Standard/Fibonacci/Camarilla pivot method optimization
- **Multi-Timeframe Coordination**: Synchronized analysis across multiple timeframes
- **94 Expert Features**: Comprehensive dual asset technical analysis

**Key Interfaces:**
- `analyze_dual_asset_atr_ema(straddle_data, underlying_data) -> DualAnalysis`
- `optimize_atr_periods(volatility_regime) -> OptimalPeriods`
- `calculate_ema_confluence_zones(price_data) -> ConfluenceZones`
- `select_optimal_cpr_method(market_condition) -> CPRMethod`

**Performance Targets:**
- Processing Time: <200ms
- Memory Usage: <500MB
- Feature Count: 94 features
- Trend Detection: >87%
- Volatility Accuracy: >85%

**Technology Stack:** Dual asset processing, ATR optimization, EMA confluence analysis, CPR method selection algorithms

#### **Component 6: Ultra-Comprehensive Correlation Framework (30x30 Matrix)**
**Responsibility:** Advanced 30x30 correlation matrix with 774 expert-optimized features and intelligent breakdown detection
**Integration Points:** Cross-component validation system with correlation breakdown alert system

**Key Features:**
- **30x30 Correlation Matrix**: Comprehensive cross-component correlation analysis
- **774 Expert-Optimized Features**: Intelligently reduced from 940 naive implementation
- **Hierarchical Implementation**: 10x10 → 18x18 → 24x24 → 30x30 progressive validation
- **Correlation Breakdown Detection**: Early warning system for regime instability
- **Graph Neural Networks**: Advanced relationship modeling architecture
- **Reinforcement Learning**: PPO-based regime classification optimization
- **150 Expert Features**: Ultra-comprehensive correlation intelligence

**Performance Phases:**
- **Phase 1 (18x18)**: 80-82% accuracy, 464 features
- **Phase 2 (24x24)**: 82-85% accuracy, 656 features
- **Phase 3 (30x30)**: 85-88% accuracy, 774 features
- **Phase 4 (RL)**: 88-92% accuracy, reinforcement learning optimized

**Key Interfaces:**
- `calculate_progressive_correlation_matrix(phase) -> CorrelationMatrix`
- `detect_correlation_breakdowns(threshold) -> BreakdownAlerts`
- `validate_cross_component_agreement() -> ValidationScore`
- `optimize_with_reinforcement_learning() -> RLOptimizedWeights`

**Performance Targets:**
- Processing Time: <180ms
- Memory Usage: <350MB
- Feature Count: 150 features (774 total correlation features)
- Breakdown Detection: >90%
- Cross-Validation Accuracy: >92%

**Technology Stack:** Graph neural networks, hierarchical clustering, transformer attention, reinforcement learning (PPO), Vertex AI integration

#### **Component 7: Dynamic Support/Resistance Logic with Multi-Method Confluence**
**Responsibility:** Advanced level detection using 5 proven methods with dual asset analysis and multi-timeframe confluence
**Integration Points:** Dynamic level detection for both straddle prices and underlying with strength scoring

**Key Features:**
- **5 Proven Detection Methods**: Pivot/Volume/Psychological/Fibonacci/Technical confluence
- **Dual Asset Level Analysis**: Support/resistance for both straddle prices AND underlying
- **Multi-Timeframe Confluence**: Daily/weekly/monthly level significance analysis
- **Dynamic Level Strength**: Touch count, hold success, volume confirmation scoring
- **Breakout Probability**: Advanced breakout prediction with confidence scoring
- **72 Expert Features**: Comprehensive level detection and strength analysis

**Key Interfaces:**
- `detect_multi_method_levels(price_data, methods) -> LevelAnalysis`
- `calculate_dual_asset_confluence(straddle_levels, underlying_levels) -> ConfluenceScore`
- `predict_breakout_probability(level, current_price, volume) -> BreakoutPrediction`
- `score_level_strength(touch_count, hold_success, volume) -> StrengthScore`

**Performance Targets:**
- Processing Time: <150ms
- Memory Usage: <600MB
- Feature Count: 72 features
- Level Prediction: >88%
- Breakout Accuracy: >85%

**Technology Stack:** Multi-method level detection, dual asset analysis, confluence scoring, breakout prediction algorithms

#### **Component 8: DTE-Adaptive Master Integration with Regime Classification**
**Responsibility:** Master orchestration of all 7 components with DTE-specific optimization and final 8-regime classification
**Integration Points:** Intelligent component integration with market structure change detection and regime mapping

**Key Features:**
- **Master Component Integration**: Intelligent weighting of all 7 components based on performance
- **DTE-Specific Optimization**: Different component importance by expiry proximity
- **8-Regime Classification**: Final classification into LVLD/HVC/VCPE/TBVE/TBVS/SCGS/PSED/CBV
- **Market Structure Detection**: Automatic adaptation when market conditions change
- **18→8 Regime Mapping**: Advanced mapping from detailed to strategic classifications
- **48 Expert Features**: Master integration and regime classification features

**8-Regime Strategic Classification:**
- **LVLD**: Low Volatility Low Delta - Stable market conditions
- **HVC**: High Volatility Contraction - Volatility declining from high levels
- **VCPE**: Volatility Contraction Price Expansion - Low vol with strong directional moves
- **TBVE**: Trend Breaking Volatility Expansion - Trend reversal with increasing volatility
- **TBVS**: Trend Breaking Volatility Suppression - Trend change with controlled volatility
- **SCGS**: Strong Correlation Good Sentiment - High component agreement with positive sentiment
- **PSED**: Poor Sentiment Elevated Divergence - Negative sentiment with component disagreement
- **CBV**: Choppy Breakout Volatility - Sideways market with periodic volatility spikes

**Key Interfaces:**
- `integrate_all_components(component_results) -> MasterIntegration`
- `classify_8_regime_system(integrated_score) -> RegimeClassification`
- `adapt_dte_specific_component_weights(dte) -> ComponentWeights`
- `detect_market_structure_changes(performance_history) -> StructureAlert`

**Performance Targets:**
- Processing Time: <100ms
- Memory Usage: <1000MB
- Feature Count: 48 features
- Regime Classification: >88%
- Market Structure Detection: <24 hours response

**Technology Stack:** Ensemble modeling, regime classification algorithms, market structure detection, DTE-specific optimization

### Component Interaction Diagram

```mermaid
graph TD    subgraph Local
      A1[Parquet (Local)] --> B1[Arrow → GPU]
      B1 --> C1a[8-Component Analysis]
    end
    
    subgraph Cloud
      A2[Parquet (GCS)] --> B2[BigQuery Processing]
      B2 --> B2a[Arrow → GPU]
      B2a --> C1b[8-Component Analysis]
    end
    
    C1a --> VX[Vertex AI]
    C1b --> VX
    VX --> C8[DTE Master Integration]
    
    C8 --> API[Enhanced API Layer]
    API --> UI[Backtester UI]
    API --> EXT[External Trading Systems]
    
    C8 --> METRICS[Performance Monitoring]
```

## API Design and Integration

### API Integration Strategy

**API Integration Strategy:** Extension pattern - new ML-enhanced endpoints added alongside existing API framework with version-controlled progressive enhancement

**Authentication:** Leverage existing API key and session management system used by backtester_v2

**Versioning:** Semantic versioning with backward compatibility (/api/v1/ maintained, /api/v2/ for enhanced features)

### New API Endpoints

#### **Enhanced Regime Analysis**
- **Method:** POST
- **Endpoint:** `/api/v2/regime/analyze`
- **Purpose:** 8-component adaptive learning analysis with ML enhancement
- **Integration:** Extends existing regime analysis with component-level insights

**Request:**
```json
{
  "symbol": "NIFTY",
  "timestamp": "2025-08-10T14:30:00Z",
  "dte_filter": [0, 7, 30],
  "component_weights": {
    "triple_straddle": 0.15,
    "greeks_sentiment": 0.14,
    "oi_pa_trending": 0.13,
    "iv_skew": 0.12,
    "atr_ema_cpr": 0.11,
    "correlation": 0.10,
    "support_resistance": 0.10,
    "master_integration": 0.15
  },
  "use_adaptive_weights": true,
  "enable_ml_enhancement": true
}
```

**Response:**
```json
{
  "analysis_id": "uuid-12345",
  "symbol": "NIFTY",
  "timestamp": "2025-08-10T14:30:03.250Z",
  "master_regime": "VCPE",
  "regime_name": "Volatility Contraction Price Expansion",
  "master_confidence": 0.89,
  "processing_time_ms": 750,
  "component_results": {
    "triple_straddle": {
      "regime_contribution": "BULLISH_EXPANSION",
      "confidence": 0.85,
      "adaptive_weight": 0.16,
      "correlation_matrix": "10x10_matrix_data"
    },
    "greeks_sentiment": {
      "sentiment_level": "MODERATELY_BULLISH", 
      "confidence": 0.78,
      "delta_sentiment": 0.65,
      "gamma_exposure": 1.23
    }
  },
  "ml_enhancement": {
    "vertex_ai_prediction": "VCPE",
    "ml_confidence": 0.91,
    "model_version": "mr-adaptive-v1.2.3",
    "feature_importance": "top_10_features"
  },
  "system_health": {
    "component_agreement": 0.82,
    "correlation_breakdown_alerts": [],
    "performance_within_targets": true
  }
}
```

#### **Adaptive Weight Management**
- **Method:** PUT  
- **Endpoint:** `/api/v2/regime/weights`
- **Purpose:** Update and monitor adaptive weight evolution across all components
- **Integration:** New capability for real-time weight optimization

**Request:**
```json
{
  "component_id": "triple_straddle",
  "dte_bucket": "0-7",
  "performance_feedback": {
    "actual_regime": "VCPE",
    "predicted_regime": "VCPE", 
    "accuracy_score": 0.95,
    "timing_accuracy_minutes": 3.2
  },
  "update_strategy": "exponential_smoothing"
}
```

**Response:**
```json
{
  "component_id": "triple_straddle",
  "old_weight": 0.15,
  "new_weight": 0.162,
  "weight_change": 0.012,
  "performance_improvement": 0.03,
  "update_timestamp": "2025-08-10T14:35:00Z",
  "learning_convergence": "improving"
}
```

#### **Component Health Monitoring**
- **Method:** GET
- **Endpoint:** `/api/v2/regime/health`  
- **Purpose:** Real-time monitoring of all 8 components and system health
- **Integration:** New monitoring capability for system reliability

**Response:**
```json
{
  "system_status": "HEALTHY",
  "total_processing_time_ms": 750,
  "memory_usage_gb": 2.1,
  "component_status": {
    "triple_straddle": {
      "status": "HEALTHY",
      "last_update": "2025-08-10T14:30:00Z",
      "processing_time_ms": 140,
      "accuracy_7d": 0.87
    },
    "correlation_framework": {
      "status": "WARNING", 
      "issue": "correlation_breakdown_detected",
      "affected_pairs": ["greeks-oi_pa", "iv_skew-atr"],
      "processing_time_ms": 185
    }
  },
  "adaptive_learning": {
    "weights_updated_last_hour": 12,
    "convergence_status": "stable",
    "market_structure_changes": 0
  },
  "vertex_ai_integration": {
    "model_serving_status": "online",
    "prediction_latency_ms": 45,
    "model_drift_detected": false
  }
}
```

## External API Integration

### **Google Vertex AI API**
- **Purpose:** ML model training, serving, and hyperparameter optimization for adaptive learning system
- **Documentation:** https://cloud.google.com/vertex-ai/docs
- **Base URL:** https://us-central1-aiplatform.googleapis.com
- **Authentication:** Service Account with Vertex AI permissions
- **Integration Method:** REST API calls from Python backend with connection pooling

**Key Endpoints Used:**
- `POST /v1/projects/{project}/locations/{location}/endpoints/{endpoint}:predict` - Real-time ML predictions
- `POST /v1/projects/{project}/locations/{location}/trainingPipelines` - Automated model retraining

**Error Handling:** Graceful fallback to rule-based regime classification if Vertex AI unavailable, with automatic retry logic and circuit breaker pattern

### **Google BigQuery API**
- **Purpose:** ML data warehouse for historical pattern analysis and feature engineering
- **Documentation:** https://cloud.google.com/bigquery/docs/reference/rest
- **Base URL:** https://bigquery.googleapis.com
- **Authentication:** Service Account with BigQuery read/write permissions  
- **Integration Method:** Python BigQuery client with connection pooling and query optimization

**Key Endpoints Used:**
- `POST /bigquery/v2/projects/{project}/jobs` - Execute feature engineering queries
- `GET /bigquery/v2/projects/{project}/datasets/{dataset}/tables/{table}/data` - Historical data retrieval

**Error Handling:** Automatic query retry with exponential backoff, fallback to HeavyDB for critical data needs

## Source Tree Integration

### Existing Project Structure
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

### New File Organization
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

### Integration Guidelines

- **File Naming:** Follow existing snake_case convention with descriptive prefixes (adaptive_, component_, ml_)
- **Folder Organization:** Mirror existing structure with new directories for enhanced functionality, preserve all existing paths
- **Import/Export Patterns:** Maintain existing import patterns, add new module imports with clear namespacing (from adaptive_learning.components import ...)

## Infrastructure and Deployment Integration

### Existing Infrastructure
**Current Deployment:** SSH-based deployment with tmux session management and manual orchestration
**Infrastructure Tools:** HeavyDB cluster, Python virtual environments, systemd services for automation
**Environments:** Local development, staging server, production trading environment

### Enhancement Deployment Strategy

**Deployment Approach:** Hybrid cloud-local deployment maintaining HeavyDB infrastructure while adding Google Cloud ML services

**Infrastructure Changes:**
- Google Cloud project setup with Vertex AI, BigQuery, and Cloud Storage
- VPN connection between local HeavyDB and Google Cloud for secure data transfer
- Container registry for ML model artifacts and component deployments
- Enhanced monitoring with Google Cloud Monitoring integrated with existing systems

**Pipeline Integration:** 
- Extend existing BMAD orchestration system to include Vertex AI model deployment
- Automated data pipeline for HeavyDB → BigQuery synchronization  
- CI/CD pipeline for ML model updates and component deployments
- Blue-green deployment strategy for zero-downtime model updates

### Rollback Strategy

**Rollback Method:** Automated rollback to existing system with feature flags for gradual component activation

**Risk Mitigation:** 
- All new components have fallback to existing implementations
- Performance monitoring with automatic rollback triggers if latency >800ms
- Database transactions ensure data consistency during rollback
- Canary deployments for gradual traffic migration

**Monitoring:** 
- Real-time performance monitoring of all 8 components
- ML model drift detection with automatic retraining triggers
- Business metric tracking (accuracy, latency, throughput)
- Alert system for critical failures with automatic escalation

## Coding Standards and Conventions

### Existing Standards Compliance

**Code Style:** PEP 8 compliance with existing project conventions (120 character line limit, descriptive variable names)
**Linting Rules:** flake8 with existing project exclusions, black code formatting
**Testing Patterns:** pytest framework with existing test structure, 90%+ code coverage requirement
**Documentation Style:** Google-style docstrings with type hints, comprehensive inline documentation

### Enhancement-Specific Standards

- **Adaptive Component Interface:** All 8 components must implement `AdaptiveComponent` base class with standardized methods
- **ML Integration Pattern:** Vertex AI calls wrapped in retry logic with circuit breaker pattern
- **Performance Logging:** Mandatory sub-component timing logs for <800ms total target
- **Error Handling:** All ML service calls must have graceful fallback to existing algorithms

### Critical Integration Rules

- **Existing API Compatibility:** All new endpoints maintain backward compatibility, existing endpoints unchanged
- **Database Integration:** New tables only, no modification of existing HeavyDB schema
- **Error Handling:** All adaptive learning failures must gracefully fall back to existing regime classification
- **Logging Consistency:** All new components use existing logging framework with structured JSON output

## Testing Strategy

### Integration with Existing Tests

**Existing Test Framework:** pytest with fixtures for HeavyDB data, comprehensive Excel configuration testing
**Test Organization:** Unit tests per component, integration tests per module, end-to-end workflow testing  
**Coverage Requirements:** Maintain existing 90% coverage while adding comprehensive adaptive learning tests

### New Testing Requirements

#### Unit Tests for New Components
- **Framework:** pytest with mock objects for external ML services
- **Location:** `backtester_v2/ui-centralized/strategies/market_regime/adaptive_learning/tests/`
- **Coverage Target:** 95% coverage for all 8 adaptive components
- **Integration with Existing:** Extend existing test fixtures with adaptive learning data

#### Integration Tests
- **Scope:** End-to-end testing of 8-component pipeline with real HeavyDB data
- **Existing System Verification:** Ensure no regression in existing regime classification accuracy
- **New Feature Testing:** Validate <800ms performance target and >85% accuracy improvement

#### Regression Testing  
- **Existing Feature Verification:** Automated testing of all existing backtester functionality with new components disabled
- **Automated Regression Suite:** Daily regression tests against historical market data  
- **Manual Testing Requirements:** Monthly review of trading signal quality by quantitative analysts

## Security Integration

### Existing Security Measures
**Authentication:** API key-based authentication for backtester endpoints
**Authorization:** Role-based access (admin, trader, analyst) with session management
**Data Protection:** Local data encryption at rest, TLS for API communications  
**Security Tools:** Input validation, SQL injection protection for HeavyDB queries

### Enhancement Security Requirements

**New Security Measures:**
- Google Cloud IAM integration with service accounts for Vertex AI access
- API key rotation for enhanced endpoints, OAuth 2.0 for interactive features
- Data pipeline encryption for HeavyDB → BigQuery transfers
- ML model artifact signing and verification

**Integration Points:**
- Single sign-on integration between existing auth and Google Cloud Identity
- Audit logging for all ML model predictions and weight updates
- Data lineage tracking for compliance and debugging

**Compliance Requirements:**
- SOC 2 Type II readiness for cloud components
- Financial industry data retention (7 years) for ML training data
- GDPR compliance for any personal data in model training

### Security Testing
**Existing Security Tests:** Penetration testing of API endpoints, SQL injection testing
**New Security Test Requirements:** Vertex AI API security testing, data pipeline encryption validation, ML model poisoning attack prevention
**Penetration Testing:** Quarterly security assessment of cloud integration points and API attack surface

## Next Steps

### Story Manager Handoff

**Prompt for Story Manager:**

"Implement the Market Regime Master Framework 8-Component Adaptive Learning System based on the comprehensive brownfield architecture document. Key integration requirements validated with the existing system:

- **Performance Constraint**: Total system processing must be <800ms with <3.7GB memory usage
- **Accuracy Target**: >85% regime classification accuracy while maintaining existing system performance  
- **Integration Points**: Seamlessly integrate with existing HeavyDB infrastructure at `strategies/market_regime/`
- **Backward Compatibility**: Maintain 100% API compatibility with existing trading system integrations

**First Story Implementation**: Begin with Component 1 (Triple Rolling Straddle System) integration, implementing the adaptive weight learning engine while preserving existing straddle analysis functionality. Include comprehensive integration checkpoints to validate no performance degradation in the existing backtester system.

**System Integrity Priority**: Throughout implementation, existing system functionality must remain unimpacted. All new adaptive learning components should run in parallel with existing systems initially, with gradual migration based on validation results."

### Developer Handoff

**Prompt for Developers:**

"Begin implementing the 8-Component Adaptive Learning System following the brownfield architecture specifications. Reference the comprehensive architecture document for detailed technical decisions based on real project constraints.

**Integration Requirements**: 
- Extend existing `backtester_v2/ui-centralized/strategies/market_regime/` structure
- Maintain all existing Excel parameter mapping (600+ parameters → ML hyperparameters)  
- Preserve HeavyDB integration patterns and connection pooling
- Follow existing code style with PEP 8 compliance and Google-style docstrings

**Implementation Sequence**:
1. **Component 1**: Implement adaptive triple straddle system with 10-component weighting
2. **ML Pipeline**: Set up Vertex AI integration with fallback to existing algorithms
3. **Performance Monitoring**: Implement real-time component performance tracking
4. **Integration Testing**: Validate each component against existing system accuracy

**Critical Verification Steps**:
- Each component must have graceful fallback to existing implementations
- All database changes must be additive only (no existing schema modifications)
- API changes must maintain backward compatibility with version-controlled enhancement
- Performance testing must validate <800ms total processing time for all 8 components

**Existing System Compatibility**: Before proceeding with any implementation, thoroughly test existing system functionality to establish baseline performance metrics. All enhancements must preserve existing trading system reliability.

---

## **🚨 CRITICAL SYSTEM FIXES & PERFORMANCE SPECIFICATIONS 🚨**

### **CRITICAL FIXES IMPLEMENTED:**

#### **1. Component 2 - Greeks Sentiment Analysis CRITICAL FIX:**
- **❌ WRONG**: `gamma_weight: float = 0.0` (COMPLETELY INCORRECT)
- **✅ CORRECTED**: `gamma_weight: float = 1.5` (HIGHEST WEIGHT - Critical for pin risk detection)
- **Impact**: This fix is game-changing for regime transition detection at expiry

#### **2. Component 1 - Rolling Straddle Paradigm Shift:**
- **❌ TRADITIONAL**: EMA/VWAP/Pivots applied to underlying prices (inferior)  
- **✅ REVOLUTIONARY**: EMA/VWAP/Pivots applied to ROLLING STRADDLE PRICES (superior)
- **Exception**: CPR analysis remains on underlying for regime classification

#### **3. 774-Feature Expert Optimization:**
- **Original Naive**: 940 features (combinatorial explosion risk)
- **Expert Optimized**: 774 features (20% reduction, 95% intelligence retained)
- **Implementation**: Hierarchical 10x10→18x18→24x24→30x30 progressive validation

### **SYSTEM-WIDE PERFORMANCE SPECIFICATIONS**

#### **Component-Level Performance Targets**

| Component | Processing Time | Memory Usage | Feature Count | Accuracy Target |
|-----------|----------------|---------------|---------------|----------------|
| **Component 1** | <150ms | <350MB | 120 | >85% |
| **Component 2** | <120ms | <250MB | 98 | >88% |
| **Component 3** | <200ms | <400MB | 105 | >82% |
| **Component 4** | <200ms | <300MB | 87 | >85% |
| **Component 5** | <200ms | <500MB | 94 | >87% |
| **Component 6** | <180ms | <350MB | 150 | >90% |
| **Component 7** | <150ms | <600MB | 72 | >88% |
| **Component 8** | <100ms | <1000MB | 48 | >88% |
| **🎯 TOTAL SYSTEM** | **<800ms** | **<3.7GB** | **774** | **>87%** |

### **774-Feature Engineering Framework**

```python
FEATURE_BREAKDOWN = {
    "Component_1_Triple_Straddle": {
        "features": 120,
        "categories": [
            "Rolling straddle EMA analysis (40 features)",
            "Rolling straddle VWAP analysis (30 features)", 
            "Rolling straddle pivot analysis (25 features)",
            "Multi-timeframe integration (25 features)"
        ]
    },
    "Component_2_Greeks_Sentiment": {
        "features": 98,
        "categories": [
            "Volume-weighted first-order Greeks (35 features)",
            "Second-order Greeks (Vanna, Charm, Volga) (25 features)",
            "DTE-specific adjustments (20 features)",
            "7-level sentiment classification (18 features)"
        ]
    },
    "Component_3_OI_PA_Trending": {
        "features": 105,
        "categories": [
            "Cumulative ATM ±7 strikes analysis (45 features)",
            "Rolling timeframe analysis (25 features)",
            "Institutional flow detection (20 features)",
            "5 divergence pattern classification (15 features)"
        ]
    },
    "Component_4_IV_Skew": {
        "features": 87,
        "categories": [
            "Dual DTE framework analysis (35 features)",
            "Put/call skew differential (25 features)",
            "7-level IV regime classification (15 features)",
            "Term structure intelligence (12 features)"
        ]
    },
    "Component_5_ATR_EMA_CPR": {
        "features": 94,
        "categories": [
            "Dual asset technical analysis (40 features)",
            "ATR period optimization (20 features)",
            "EMA timeframe intelligence (20 features)",
            "CPR method selection (14 features)"
        ]
    },
    "Component_6_Correlation": {
        "features": 150,
        "categories": [
            "30x30 correlation matrix (90 features)",
            "Correlation breakdown detection (25 features)",
            "Cross-component validation (20 features)",
            "Graph neural network features (15 features)"
        ]
    },
    "Component_7_Support_Resistance": {
        "features": 72,
        "categories": [
            "Multi-method level detection (30 features)",
            "Dual asset confluence (20 features)",
            "Level strength scoring (15 features)",
            "Breakout probability (7 features)"
        ]
    },
    "Component_8_Master_Integration": {
        "features": 48,
        "categories": [
            "8-regime classification (20 features)",
            "DTE-adaptive weighting (15 features)",
            "Market structure detection (8 features)",
            "Component integration (5 features)"
        ]
    }
}

# TOTAL EXPERT FEATURES: 774 (Optimized from 940 naive implementation)
```

---

**🚀 ARCHITECTURE UPDATED: Ready for Production Implementation with Critical Fixes Applied 🚀**

The Market Regime Master Framework architecture is now properly aligned with the detailed component specifications and includes all critical fixes for production deployment.