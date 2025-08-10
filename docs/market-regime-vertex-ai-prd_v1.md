# Product Requirements Document
## Market Regime Analysis - Vertex AI Migration

**Document Version:** 1.0  
**Date:** August 4, 2025  
**Owner:** Market Regime Team  
**Status:** Draft

---

## 1. Executive Summary

### 1.1 Overview
Migrate the existing Market Regime Analysis system from local CSV processing to Google Cloud Vertex AI while preserving all current functionality, performance targets, and the sophisticated 18-regime classification system.

### 1.2 Business Objectives
- **Scale**: Handle 32GB+ historical data and real-time processing
- **Performance**: Maintain <3 second response time for regime classification
- **Reliability**: Achieve 99.9% uptime for production trading decisions
- **Cost Optimization**: Reduce infrastructure costs vs current setup
- **ML Enhancement**: Add predictive capabilities using historical patterns

### 1.3 Success Metrics
- **Accuracy**: ≥95% regime classification accuracy vs current system
- **Latency**: <3 seconds for real-time regime determination
- **Uptime**: 99.9% availability during market hours
- **Cost**: <$500/month operational costs
- **Migration**: Zero downtime during transition

---

## 2. Current State Analysis

### 2.1 Existing Architecture
```
SSH Server (dbmt_gpu_server_001) → CSV Processing → 18-Regime Classification → Trading Signals
    ↓
32GB Historical Data (2018-2025)
47 columns: OHLC, Greeks, IV, OI, Volume
Assets: NIFTY, BANKNIFTY, SENSEX, Commodities
```

### 2.2 Current System Components & File Locations

#### 2.2.1 Data Storage Infrastructure
- **Primary Storage**: `/nvme0n1-disk/market_data_processing/standardized_output/`
- **Data Structure**:
  ```
  /nvme0n1-disk/market_data_processing/standardized_output/
  ├── nifty/           (2,847 files, ~11.3GB)
  ├── banknifty/       (2,847 files, ~11.8GB) 
  ├── sensex/          (1,205 files, ~3.2GB)
  ├── crudeoil/        (1,680 files, ~3.1GB)
  ├── naturalgas/      (1,456 files, ~2.6GB)
  └── stocks/          (12 subdirs, variable sizes)
  ```
- **File Format**: Daily CSV files: `YYYY-MM-DD_[asset]_processed.csv`
- **Schema**: 47 columns per file, ~4,120 rows per trading day

#### 2.2.2 Processing Engine Components
**Base Directory**: `/Users/maruth/projects/backtester-stable/backtester_v2/ui-centralized/strategies/market_regime/`

**Core Processing Modules**:
- `market_regime_processor.py` - Main processing engine
- `unified_enhanced_market_regime_engine.py` - Unified orchestrator (497 lines)
- `heavydb_integrated_regime_analyzer.py` - Database integration (796 lines)

**Configuration Management**:
- `config/` - Configuration files and reports
  - `SAMPLE_CONFIG_INFO.json` - 150+ parameters summary
  - `market_regime_config.json` - Runtime configuration
  - `optimized_regime_config.json` - Performance-tuned parameters
- Excel configuration system (31 sheets, 600+ parameters)

#### 2.2.3 Feature Engineering Module Locations

**Primary Indicators Directory**: `indicators/`
```
indicators/
├── __init__.py
├── greek_sentiment_v2.py              (554 lines) - Dual weighting system
├── oi_pa_analysis_v2.py               (262 lines) - OI/Price action analysis
├── greek_sentiment/                    - Greek analysis modules
│   ├── delta_analyzer.py
│   ├── gamma_analyzer.py  
│   ├── theta_analyzer.py
│   └── vega_analyzer.py
├── iv_analytics/                       - IV analysis suite
│   ├── iv_surface_analyzer.py
│   ├── iv_percentile_analyzer.py
│   └── iv_skew_analyzer.py
├── oi_pa_analysis/                     - OI/PA modules
│   ├── oi_pa_analyzer.py
│   ├── divergence_detector.py
│   └── institutional_flow_analyzer.py
├── straddle_analysis/                  - Triple straddle system
│   ├── atm_straddle_analyzer.py
│   ├── itm_straddle_analyzer.py
│   └── otm_straddle_analyzer.py
├── technical_indicators/               - Technical analysis
│   ├── atr_analyzer.py
│   ├── volume_profile_analyzer.py
│   └── market_breadth_analyzer.py
├── volume_profile/                     - Volume analysis
└── correlation_analysis/               - Cross-asset correlation
```

**Enhanced Production Modules**:
- `production_deployment_features.py` (1,628 lines) - Production optimizations
- `enhanced_greek_sentiment_integration.py` - Cloud-ready Greek analysis
- `enhanced_module_integration_manager.py` - Module orchestration
- `excel_configuration_mapper.py` - Parameter mapping

**Base Classes & Architecture**:
```
base/
├── base_indicator.py                   - Abstract indicator interface
├── strike_selector_base.py             - Strike selection strategies
└── option_data_manager.py              - Data access layer
```

#### 2.2.4 Core Processing Architecture Files

**Engine Components**:
- `core/engine.py` - Main orchestration engine
- `core/analyzer.py` - Core analysis logic
- `core/regime_classifier.py` - 18-regime classification system
- `core/regime_detector.py` - Regime detection logic

**Data Access Layer**:
```
data/
├── heavydb_data_provider.py           (430 lines) - Database interface
├── loaders.py                         - Data loading utilities  
└── processors.py                      - Data preprocessing
```

**Integration Layer**:
```
integration/
├── api_routes.py                      - REST API endpoints
├── ui_bridge.py                       - UI integration
└── consolidator.py                    - Strategy consolidation
```

### 2.3 Current Performance Specifications
- **Processing Time**: <3 seconds per regime analysis
- **Data Volume**: ~4,120 rows per daily file (47 columns)
- **Memory Usage**: <8GB during processing
- **Update Frequency**: Real-time during market hours (9:15 AM - 3:30 PM IST)
- **Accuracy**: 95%+ regime classification accuracy (production-tested)
- **Throughput**: 500+ predictions per minute
- **Availability**: 99.5% uptime during market hours

---

## 3. Product Vision & Strategy

### 3.1 Vision Statement
Transform the market regime analysis system into a cloud-native, ML-enhanced platform that maintains existing accuracy while adding predictive capabilities and scalability.

### 3.2 Strategic Goals
1. **Preserve Excellence**: Maintain all current functionality and performance
2. **Add Intelligence**: Enhance with ML-based pattern recognition
3. **Scale Efficiently**: Handle growing data volumes cost-effectively
4. **Future-Proof**: Enable advanced analytics and real-time streaming

### 3.3 Migration Philosophy
- **Evolutionary, Not Revolutionary**: Enhance existing proven logic
- **Parameter Preservation**: All 600+ Excel parameters become ML hyperparameters
- **Feature Reuse**: Existing indicators become ML features
- **Gradual Transition**: Hybrid approach during migration

---

## 4. Technical Requirements

### 4.1 Data Migration Requirements

#### 4.1.1 Data Transfer Specifications
- **Source**: `/nvme0n1-disk/market_data_processing/standardized_output/`
- **Target**: Google Cloud Storage + BigQuery
- **Volume**: 32GB initial + 500MB daily growth
- **Format**: Preserve existing 47-column CSV schema
- **Transfer Method**: 
  - Initial bulk transfer via `gsutil -m cp -r` with parallel processing
  - Ongoing sync via automated pipeline every 30 minutes
- **Validation**: MD5 checksums for each file transfer
- **Compression**: GZIP compression for 60% storage reduction

#### 4.1.2 Detailed Asset Data Mapping
```
Source Directory Structure → GCS Bucket Structure
/nvme0n1-disk/market_data_processing/standardized_output/
├── nifty/           → gs://PROJECT_ID-raw/market_data/nifty/
│   ├── 2018-01-01_nifty_processed.csv (1.4MB, 4,120 rows)
│   ├── 2018-01-02_nifty_processed.csv (1.8MB, 5,204 rows)
│   └── ... (2,847 files total)
├── banknifty/       → gs://PROJECT_ID-raw/market_data/banknifty/
│   ├── 2018-01-01_banknifty_processed.csv (1.6MB, 4,432 rows)
│   └── ... (2,847 files total)
├── sensex/          → gs://PROJECT_ID-raw/market_data/sensex/
├── crudeoil/        → gs://PROJECT_ID-raw/market_data/crudeoil/
├── naturalgas/      → gs://PROJECT_ID-raw/market_data/naturalgas/
└── stocks/          → gs://PROJECT_ID-raw/market_data/stocks/
    ├── reliance/    → gs://PROJECT_ID-raw/market_data/stocks/reliance/
    ├── tcs/         → gs://PROJECT_ID-raw/market_data/stocks/tcs/
    └── ... (12 stock directories)
```

#### 4.1.3 Data Validation Framework
- **File Integrity**: 
  - Row count validation: Expected vs actual per file
  - Column schema validation: All 47 columns present and correct types
  - Data range validation: Spot prices, strikes, Greeks within expected bounds
- **Completeness Checks**:
  - No missing trading days (excluding holidays)
  - All expected expiry dates present
  - Complete options chain for each timestamp
- **Quality Metrics**:
  - Maximum 0.1% data anomalies per file
  - Zero files with corrupted timestamps
  - IV values within 0-200% range
  - Greeks calculations mathematically consistent

#### 4.1.2 Schema Mapping
```sql
-- BigQuery schema matching existing CSV structure
CREATE TABLE market_regime.options_chain (
  trade_date DATE,
  trade_time TIME,
  expiry_date DATE,
  index_name STRING,
  spot FLOAT64,
  atm_strike FLOAT64,
  strike FLOAT64,
  dte INT64,
  expiry_bucket STRING,
  zone_id INT64,
  zone_name STRING,
  call_strike_type STRING,
  put_strike_type STRING,
  -- CE data (13 columns)
  ce_symbol STRING,
  ce_open FLOAT64,
  ce_high FLOAT64,
  ce_low FLOAT64,
  ce_close FLOAT64,
  ce_volume INT64,
  ce_oi INT64,
  ce_coi FLOAT64,
  ce_iv FLOAT64,
  ce_delta FLOAT64,
  ce_gamma FLOAT64,
  ce_theta FLOAT64,
  ce_vega FLOAT64,
  ce_rho FLOAT64,
  -- PE data (13 columns) - same structure as CE
  pe_symbol STRING,
  pe_open FLOAT64,
  pe_high FLOAT64,
  pe_low FLOAT64,
  pe_close FLOAT64,
  pe_volume INT64,
  pe_oi INT64,
  pe_coi FLOAT64,
  pe_iv FLOAT64,
  pe_delta FLOAT64,
  pe_gamma FLOAT64,
  pe_theta FLOAT64,
  pe_vega FLOAT64,
  pe_rho FLOAT64,
  -- Futures data (7 columns)
  future_open FLOAT64,
  future_high FLOAT64,
  future_low FLOAT64,
  future_close FLOAT64,
  future_volume INT64,
  future_oi INT64,
  future_coi FLOAT64
)
PARTITION BY trade_date
CLUSTER BY index_name, expiry_date;
```

### 4.2 ML Model Requirements

#### 4.2.1 Feature Engineering Module Specifications

**Primary Indicators with Detailed Locations:**

1. **Greek Sentiment V2 Module**
   - **Location**: `indicators/greek_sentiment_v2.py` (554 lines)
   - **Dependencies**: 
     - `base/base_indicator.py` - Abstract indicator interface
     - `base/strike_selector_base.py` - Strike selection strategies
     - `base/option_data_manager.py` - Data access layer
   - **Sub-modules**:
     - `indicators/greek_sentiment/delta_analyzer.py` - Delta sentiment calculation
     - `indicators/greek_sentiment/gamma_analyzer.py` - Gamma exposure analysis
     - `indicators/greek_sentiment/theta_analyzer.py` - Time decay analysis
     - `indicators/greek_sentiment/vega_analyzer.py` - Volatility sensitivity
   - **Features Generated**: 
     - `net_delta_sentiment` (range: -1.0 to 1.0)
     - `gamma_exposure_ratio` (range: 0.0 to 5.0)
     - `theta_decay_pressure` (range: -100.0 to 0.0)
     - `vega_volatility_risk` (range: 0.0 to 2.0)
     - `institutional_flow_indicator` (categorical: BULLISH/BEARISH/NEUTRAL)
   - **Configuration Parameters** (from Excel):
     - `delta_weight`: 0.3 (default), range: 0.1-0.5
     - `gamma_threshold`: 0.05, range: 0.01-0.1
     - `oi_volume_ratio`: 0.7, range: 0.5-0.9
   - **Processing Time**: <500ms per analysis
   - **Memory Usage**: <2GB
   - **Validation Criteria**:
     - Output values within expected ranges
     - Correlation with actual market moves >0.6
     - Consistency across different market conditions

2. **OI/PA Analysis V2 Module**
   - **Location**: `indicators/oi_pa_analysis_v2.py` (262 lines)
   - **Dependencies**:
     - `indicators/oi_pa_analysis/oi_pa_analyzer.py` - Core analyzer
     - `indicators/oi_pa_analysis/divergence_detector.py` - Pattern detection
     - `indicators/oi_pa_analysis/institutional_flow_analyzer.py` - Flow analysis
   - **Features Generated**:
     - `oi_price_divergence` (range: -1.0 to 1.0)
     - `institutional_vs_retail` (ratio: 0.0 to 10.0)
     - `oi_momentum` (range: -5.0 to 5.0)
     - `max_pain_distance` (range: -500.0 to 500.0 points)
     - `put_call_oi_ratio` (range: 0.1 to 3.0)
   - **Divergence Types Detected**:
     - Type 1: Bullish price, bearish OI
     - Type 2: Bearish price, bullish OI  
     - Type 3: Sideways price, trending OI
     - Type 4: Trending price, sideways OI
     - Type 5: Contradictory multi-strike signals
   - **Configuration Parameters**:
     - `divergence_threshold`: 0.15, range: 0.05-0.3
     - `institutional_threshold`: 5000, range: 1000-20000
     - `correlation_window`: 20, range: 10-50
   - **Processing Time**: <800ms per analysis
   - **Validation Criteria**:
     - Divergence detection accuracy >85%
     - False positive rate <10%
     - Correlation with price movements >0.5

3. **IV Analytics Suite**
   - **Location**: `indicators/iv_analytics/`
   - **Components**:
     - `iv_surface_analyzer.py` - IV surface construction and analysis
     - `iv_percentile_analyzer.py` - Historical percentile ranking
     - `iv_skew_analyzer.py` - Volatility skew analysis
   - **Features Generated**:
     - `iv_rank_percentile` (range: 0-100)
     - `iv_skew_ratio` (range: -2.0 to 2.0)
     - `term_structure_slope` (range: -1.0 to 1.0)
     - `atm_iv_level` (range: 0.05 to 2.0)
     - `iv_smile_curvature` (range: -0.5 to 0.5)
   - **Configuration Parameters**:
     - `percentile_lookback`: 252, range: 60-500
     - `skew_calculation_method`: 'log_moneyness' or 'strike_distance'
     - `surface_smoothing_factor`: 0.1, range: 0.01-0.3
   - **Processing Time**: <600ms per analysis
   - **Memory Usage**: <1.5GB
   - **Validation Criteria**:
     - IV calculations match market standards
     - Surface fitting R² >0.95
     - Skew patterns consistent with market conditions

4. **Extended Straddle Analysis Suite with 10×10 Correlation Matrix**
   - **Location**: `indicators/straddle_analysis/` + `correlation_matrix_engine.py`
   - **Components**:
     - `atm_straddle_analyzer.py` - At-the-money straddle analysis
     - `itm_straddle_analyzer.py` - ITM1 systematic straddle strategies
     - `otm_straddle_analyzer.py` - OTM1 systematic straddle strategies
     - `correlation_matrix_engine.py` - 10×10 correlation matrix processor
   - **10×10 Correlation Matrix Elements**:
     1. **ATM Straddle** - At-the-money combined position
     2. **ITM1 Systematic Straddle** - In-the-money Level 1 systematic position  
     3. **OTM1 Systematic Straddle** - Out-of-the-money Level 1 systematic position
     4. **ATM PE** - At-the-money Put option
     5. **ATM CE** - At-the-money Call option
     6. **ITM CE** - In-the-money Call option
     7. **ITM PE** - In-the-money Put option
     8. **OTM CE** - Out-of-the-money Call option
     9. **OTM PE** - Out-of-the-money Put option
     10. **Combined Straddle Analysis** - Composite analysis of all straddle positions
   - **Multi-Timeframe Analysis**:
     - **Intraday Trading**: 3min, 5min, 10min, 15min charts
     - **Positional Trading**: 5min, 15min, 30min, 1hour charts
   - **Indicators Applied to ALL Matrix Elements**:
     - **EMAs**: 20, 50, 100 period exponential moving averages
     - **VWAP**: Current day volume-weighted average price
     - **Previous Day VWAP**: Prior trading session VWAP reference
     - **Proven Pivot Points**: Selected methods from existing implementation
   - **Features Generated**:
     - `correlation_matrix_10x10_intraday` (100 correlations for intraday timeframes)
     - `correlation_matrix_10x10_positional` (100 correlations for positional timeframes)
     - `cross_timeframe_correlation_strength` (correlation between intraday/positional)
     - `regime_dependent_correlations` (correlation shifts by market regime)
     - `support_resistance_confluence_score` (S&R level agreement across elements)
   - **Processing Time**: <2000ms per analysis (dual timeframe complexity)
   - **Memory Usage**: <4GB (multiple correlation matrices)

5. **Support & Resistance Analysis Suite**
   - **Location**: `dynamic_support_resistance_engine.py` + `indicators/straddle_analysis/`
   - **Proven Pivot Methods** (from Excel StraddleAnalysisConfig):
     - **Classic Pivots**: Weight 0.3 (range: 0.1-0.5) - Standard pivot calculations
     - **Fibonacci Pivots**: Weight 0.25 (range: 0.1-0.4) - Fibonacci retracement levels
     - **Camarilla Pivots**: Weight 0.2 (range: 0.1-0.4) - Camarilla equation-based levels
     - **Woodie Pivots**: Weight 0.15 (range: 0.05-0.3) - Woodie formula calculations
     - **DeMark Pivots**: Weight 0.1 (range: 0.05-0.3) - Tom DeMark methodology
   - **Multi-Timeframe S&R Analysis**:
     - **Intraday S&R**: 3min, 5min, 10min, 15min chart levels
     - **Positional S&R**: 5min, 15min, 30min, 1hour chart levels
   - **Features Generated**:
     - `pivot_confluence_score_intraday` (weighted combination intraday timeframes)
     - `pivot_confluence_score_positional` (weighted combination positional timeframes)
     - `breakout_probability_multi_tf` (breakout probability across timeframes)
     - `support_resistance_strength_matrix` (strength per timeframe per method)
     - `confluence_zone_detection` (areas where multiple methods agree)
   - **Configuration Parameters**:
     - `breakout_threshold`: 0.002 (range: 0.001-0.005)
     - `consolidation_band`: 0.003 (range: 0.001-0.005)
     - `pivot_confluence_bonus`: 1.5 (range: 1.2-2.0)
     - `intraday_pivot_reset`: TRUE (boolean)
     - `pivot_time_decay`: 0.95 (range: 0.9-1.0)
     - `near_pivot_sensitivity`: 1.3 (range: 1.1-1.5)
   - **Processing Time**: <1000ms per analysis
   - **Validation Criteria**:
     - Pivot calculations match standard formulas for each method
     - Multi-timeframe confluence detection accuracy >85%
     - S&R breakout signals correlation with price moves >0.6

6. **Enhanced IV Analytics Suite**
   - **Location**: `iv_percentile_analyzer.py` + `iv_skew_analyzer.py`
   - **IV Percentile Analysis**:
     - **7-Level IV Regime Classification**: EXTREMELY_LOW (0-10%) → EXTREMELY_HIGH (90-100%)
     - **DTE-Specific Calculations**: Separate percentiles for different expiry buckets
     - **Historical IV Ranking**: Rolling percentile calculations with lookback periods
     - **Multi-Timeframe IV Analysis**: IV percentiles across all trading timeframes
   - **IV Skew Analysis**:
     - **7-Level Sentiment Classification**: EXTREMELY_BEARISH → EXTREMELY_BULLISH
     - **Put-Call IV Skew**: Strike-based skew calculations and analysis
     - **Skew Regime Detection**: Dynamic skew pattern recognition
     - **Cross-Strike Skew Analysis**: Complete options chain skew profiling
   - **Features Generated**:
     - `iv_percentile_by_dte` (percentile rankings per expiry bucket)
     - `iv_regime_classification` (7-level IV regime assignment)
     - `iv_skew_sentiment_score` (7-level sentiment from skew analysis)
     - `premium_percentile_ranking` (options premium percentile analysis)
     - `volatility_regime_confidence` (confidence in IV-based regime classification)
   - **Processing Time**: <800ms per analysis
   - **Memory Usage**: <2GB
   - **Validation Criteria**:
     - IV percentile calculations mathematically accurate
     - Skew analysis correlation with market sentiment >0.7
     - Regime classification consistency across market conditions >90%

5. **Technical Indicators Suite**
   - **Location**: `indicators/technical_indicators/`
   - **Components**:
     - `atr_analyzer.py` - Average True Range analysis
     - `volume_profile_analyzer.py` - Volume at price analysis
     - `market_breadth_analyzer.py` - Market breadth indicators
   - **ATR Analysis Features**:
     - `atr_percentile` (range: 0-100)
     - `volatility_regime` (categorical: LOW/MEDIUM/HIGH)
     - `breakout_probability` (range: 0.0 to 1.0)
   - **Volume Profile Features**:
     - `poc_distance` (range: -1000 to 1000 points)
     - `volume_imbalance` (range: -1.0 to 1.0)
     - `value_area_high_low` (2-element array)
   - **Configuration Parameters**:
     - `atr_periods`: [14, 21, 50]
     - `volume_profile_bins`: 50, range: 20-100
     - `breadth_calculation_method`: 'advance_decline' or 'new_highs_lows'
   - **Processing Time**: <400ms per analysis
   - **Validation Criteria**:
     - ATR calculations match standard formulas
     - Volume profile POC accuracy >90%
     - Breadth indicators correlation with market indices >0.7

#### 4.2.2 Enhanced Production Modules

**Production Deployment Features Module**:
- **Location**: `production_deployment_features.py` (1,628 lines)
- **Key Classes**:
  - `OIFlowData` - Open Interest flow data structure
  - `ProductionDeploymentEngine` - Main orchestrator
  - `CrossStrikeOIFlowAnalyzer` - Advanced OI analysis
  - `OISkewAnalyzer` - OI skew detection
  - `ProductionOptimizer` - Performance optimization
- **Performance Targets**:
  - Real-time OI flow analysis: <200ms latency
  - OI skew detection: 95% accuracy
  - Production uptime: 99.9%
- **Memory Optimization**:
  - Circular buffers for streaming data
  - Lazy loading of historical data
  - Connection pooling for database access
- **Validation Framework**:
  - Unit tests for each analyzer class
  - Integration tests with real market data
  - Performance benchmarks under load

**Enhanced Integration Modules**:
- **Location**: `enhanced_module_integration_manager.py`
- **Functionality**:
  - Module lifecycle management
  - Dependency resolution and loading
  - Configuration parameter injection
  - Performance monitoring and optimization
- **Integration Priorities**: 
  - CRITICAL: Greek Sentiment, OI/PA Analysis
  - HIGH: IV Analytics, Straddle Analysis
  - MEDIUM: Technical Indicators, Volume Profile
- **Validation**:
  - Module loading success rate: 100%
  - Parameter injection accuracy: 100%
  - Integration test coverage: >95%

#### 4.2.3 Configuration System Details

**Excel Configuration Mapping**:
- **Location**: `excel_configuration_mapper.py`
- **31 Excel Sheets Mapped**:
  1. `MasterConfiguration` → Core system parameters
  2. `StabilityConfiguration` → Performance tuning
  3. `TransitionManagement` → Regime transition rules
  4. `NoiseFiltering` → Signal filtering parameters
  5. `IndicatorConfiguration` → Indicator weights and thresholds
  6. `GreekSentimentConfig` → Greek analysis parameters
  7. `TrendingOIPAConfig` → OI/PA analysis settings
  8. `StraddleAnalysisConfig` → Straddle strategy parameters
  9. `IVSurfaceConfig` → IV analysis configuration
  10. `ATRIndicatorsConfig` → Technical indicator settings
  ... (21 additional sheets with specific parameter mappings)

**Parameter Validation Framework**:
- **Range Validation**: All numeric parameters within acceptable bounds
- **Type Validation**: Correct data types (float, int, string, boolean)
- **Dependency Validation**: Parameter combinations consistency
- **Business Logic Validation**: Financially meaningful parameter values
- **Performance Impact Validation**: Parameters don't degrade system performance

#### 4.2.4 ML Model Architecture Specifications

**Model Components**:
```python
# Ensemble Model Architecture
ensemble_config = {
    'base_models': {
        'random_forest': {
            'n_estimators': excel_config['rf_trees'],  # Default: 100
            'max_depth': excel_config['rf_depth'],     # Default: 10
            'min_samples_split': excel_config['rf_min_split'],  # Default: 5
            'random_state': 42
        },
        'gradient_boosting': {
            'n_estimators': excel_config['gb_trees'],  # Default: 100
            'learning_rate': excel_config['gb_learning_rate'],  # Default: 0.1
            'max_depth': excel_config['gb_depth'],     # Default: 6
            'random_state': 42
        },
        'neural_network': {
            'hidden_layers': excel_config['nn_layers'],  # Default: [64, 32, 16]
            'activation': excel_config['nn_activation'], # Default: 'relu'
            'dropout_rate': excel_config['nn_dropout'],  # Default: 0.2
            'epochs': excel_config['nn_epochs']         # Default: 100
        }
    },
    'ensemble_method': excel_config['ensemble_method'],  # Default: 'voting'
    'model_weights': excel_config['model_weights']       # Default: [0.4, 0.4, 0.2]
}
```

**Feature Vector Specifications**:
```python
# Feature vector structure (285+ features due to multi-timeframe correlation analysis)
feature_vector = {
    'greek_sentiment_features': 5,                    # From GreekSentimentV2
    'oi_pa_features': 5,                             # From OI/PA Analysis
    'iv_analytics_features': 15,                     # Enhanced IV Suite (percentile + skew + premium)
    'straddle_correlation_features_intraday': 113,   # 10×10 correlations + derived (intraday timeframes)
    'straddle_correlation_features_positional': 113, # 10×10 correlations + derived (positional timeframes)
    'support_resistance_features_intraday': 15,      # 5 Pivot methods × 3 derived features (intraday)
    'support_resistance_features_positional': 15,    # 5 Pivot methods × 3 derived features (positional)
    'technical_features': 6,                         # From Technical Indicators
    'vwap_features': 4,                              # Day VWAP, Previous Day VWAP, deviations, cross-timeframe
    'ema_features': 6,                               # EMA 20/50/100 × 2 timeframe sets
    'time_features': 3,                              # Time-based features
    'volatility_features': 6,                        # Volatility-related features
    'momentum_features': 4,                          # Momentum indicators
    'sentiment_features': 3,                         # Market sentiment
    'macro_features': 2                              # Macro economic indicators
}
# Total: 285 features (significantly increased due to dual timeframe analysis)
```

**Hyperparameter Mapping**:
```python
# Excel parameters → ML hyperparameters mapping
hyperparameter_mapping = {
    # Model architecture parameters
    'excel_config.adaptive_learning_rate': 'model.learning_rate',
    'excel_config.ensemble_size': 'model.n_estimators',
    'excel_config.tree_depth': 'model.max_depth',
    'excel_config.min_samples': 'model.min_samples_split',
    
    # Feature engineering parameters
    'excel_config.volatility_threshold': 'features.volatility_regime_threshold',
    'excel_config.trend_threshold': 'features.trend_classification_threshold',
    'excel_config.delta_weight': 'features.greek_sentiment_delta_weight',
    'excel_config.oi_threshold': 'features.oi_significance_threshold',
    
    # Regime classification parameters
    'excel_config.regime_confidence_threshold': 'classifier.confidence_threshold',
    'excel_config.transition_smoothing': 'classifier.transition_smoothing_factor',
    'excel_config.noise_filter_strength': 'classifier.noise_filter_coefficient',
    
    # Performance parameters
    'excel_config.processing_timeout': 'system.max_processing_time_ms',
    'excel_config.cache_expiry': 'system.cache_expiry_seconds',
    'excel_config.batch_size': 'system.batch_processing_size'
}
```

### 4.3 Performance Requirements
- **Latency**: <3 seconds for regime classification
- **Throughput**: Handle 1000+ concurrent requests
- **Availability**: 99.9% uptime during market hours (9:15 AM - 3:30 PM IST)
- **Accuracy**: ≥95% classification accuracy vs historical performance
- **Scalability**: Auto-scale based on demand

### 4.4 Integration Requirements
- **API Compatibility**: Maintain existing API contracts
- **Data Pipeline**: Real-time streaming from market data sources
- **Monitoring**: Comprehensive observability and alerting
- **Backup/Recovery**: Automated backup and disaster recovery

---

## 5. Functional Requirements

### 5.1 Core Features

#### 5.1.1 Data Processing Pipeline
```
Market Data → Feature Engineering → ML Model → 18-Regime Classification → Trading Signals
     ↓              ↓                  ↓              ↓                    ↓
  BigQuery    Existing Indicators  Vertex AI    Confidence Score    API Response
```

#### 5.1.2 Feature Engineering Module
- **Greek Sentiment V2**: Dual weighting, ITM analysis, 7-level classification
- **OI/PA Analysis V2**: 5-type divergence detection, institutional flow analysis
- **IV Analytics**: Surface analysis, percentile tracking, skew detection
- **Straddle Analysis**: Triple straddle strategies with correlation matrix
- **Technical Indicators**: ATR, volume profile, market breadth

#### 5.1.3 ML Model Features
- **Training**: Automated retraining on new data
- **Prediction**: Real-time regime classification
- **Confidence**: Uncertainty quantification for decisions
- **Drift Detection**: Model performance monitoring
- **A/B Testing**: Compare model versions

### 5.2 API Specifications

#### 5.2.1 Real-time Prediction API
```python
POST /api/v1/regime/predict
{
    "market_data": {
        "timestamp": "2025-08-04T14:30:00Z",
        "index_name": "nifty",
        "spot": 24500.0,
        "options_chain": [...],
        "futures_data": {...}
    },
    "config_overrides": {
        "volatility_threshold": 0.25,
        "greek_weights": {...}
    }
}

Response:
{
    "regime": "STRONG_BULLISH_LOW_VOLATILE",
    "confidence": 0.92,
    "sub_regimes": {
        "volatility_regime": "LOW",
        "trend_regime": "BULLISH", 
        "structure_regime": "WEAK"
    },
    "indicators": {
        "greek_sentiment": 0.75,
        "oi_divergence": -0.15,
        "iv_percentile": 25.0
    },
    "timestamp": "2025-08-04T14:30:03Z",
    "processing_time_ms": 2850
}
```

#### 5.2.2 Batch Processing API
```python
POST /api/v1/regime/batch
{
    "date_range": {
        "start": "2025-08-01",
        "end": "2025-08-04"
    },
    "assets": ["nifty", "banknifty"],
    "output_format": "csv"
}
```

#### 5.2.3 Configuration Management API
```python
GET /api/v1/config/parameters
PUT /api/v1/config/parameters
POST /api/v1/config/validate
```

### 5.3 User Interface Requirements

#### 5.3.1 Monitoring Dashboard
- **Real-time Metrics**: Latency, accuracy, throughput
- **Model Performance**: Drift detection, accuracy trends
- **System Health**: Infrastructure status, error rates
- **Configuration**: Parameter management interface

#### 5.3.2 Analytics Interface
- **Historical Analysis**: Regime transition patterns
- **Performance Reports**: Model vs actual market moves
- **Parameter Impact**: Sensitivity analysis for configuration changes

---

## 6. Non-Functional Requirements

### 6.1 Performance Requirements

#### 6.1.1 Latency Requirements
- **End-to-End Regime Prediction**: <3 seconds (95th percentile)
- **Feature Engineering Processing**: <2 seconds (mean)
- **ML Model Inference**: <1 second (95th percentile)
- **API Response Time**: <3.5 seconds including network overhead
- **Batch Processing**: 1 million records/hour minimum throughput

#### 6.1.2 Throughput Requirements
- **Concurrent Requests**: 1000+ simultaneous prediction requests
- **Peak Load Handling**: 5000 requests/minute during market open
- **Data Ingestion**: Process 500MB daily data within 30 minutes
- **Model Training**: Complete retraining within 6 hours
- **Feature Store Updates**: <10 seconds for parameter changes

#### 6.1.3 Resource Utilization
- **Memory Usage**: <16GB per Vertex AI endpoint instance
- **CPU Utilization**: <80% average during peak hours
- **Storage IOPS**: Minimum 1000 IOPS for BigQuery operations
- **Network Bandwidth**: <100Mbps per prediction endpoint
- **GPU Usage**: Optional for neural network components (<50% utilization)

### 6.2 Scalability Requirements

#### 6.2.1 Auto-scaling Configuration
```yaml
# Vertex AI Endpoint Auto-scaling
min_replica_count: 2
max_replica_count: 10
target_utilization_percentage: 70
scale_up_threshold: 80%  # CPU/Memory
scale_down_threshold: 30%
cool_down_period: 300s   # 5 minutes
```

#### 6.2.2 Load Distribution
- **Geographic Distribution**: Multi-region deployment (us-central1, asia-south1)
- **Traffic Splitting**: Blue-green deployment for model updates
- **Circuit Breaker**: Fail-fast mechanism for overloaded services
- **Rate Limiting**: 100 requests/minute per API key (configurable)

#### 6.2.3 Data Volume Scaling
- **Current**: 32GB historical data, 500MB daily growth
- **Projected**: 100GB within 2 years, 1GB daily growth
- **BigQuery Scaling**: Partition by date, cluster by asset
- **Storage Scaling**: Automatic lifecycle management for cost optimization

### 6.3 Reliability & Availability

#### 6.3.1 Service Level Objectives (SLOs)
- **Availability**: 99.9% during market hours (9:15 AM - 3:30 PM IST)
- **Error Rate**: <0.1% for API requests
- **Mean Time To Recovery (MTTR)**: <15 minutes for critical issues
- **Mean Time Between Failures (MTBF)**: >30 days
- **Data Durability**: 99.999999999% (11 9's) using GCS

#### 6.3.2 Fault Tolerance
- **Multi-Zone Deployment**: Services deployed across 3 availability zones
- **Database Replication**: BigQuery automatic replication
- **Backup Strategy**: 
  - Hourly incremental backups during market hours
  - Daily full backups with 30-day retention
  - Weekly backups with 1-year retention
- **Disaster Recovery**: RTO <2 hours, RPO <15 minutes

#### 6.3.3 Health Monitoring
```python
# Health check endpoints
/health/liveness    # Service is running
/health/readiness   # Service is ready to accept traffic  
/health/model       # ML model is loaded and functional
/health/data        # Data pipeline is operational
/health/features    # Feature engineering is working
```

### 6.4 Security Requirements

#### 6.4.1 Authentication & Authorization
- **API Authentication**: 
  - Primary: API keys with rotation every 90 days
  - Secondary: OAuth 2.0 for interactive applications
  - Service-to-service: Google Service Account tokens
- **Authorization Model**:
  - Admin: Full system access, configuration management
  - Trader: Prediction requests, historical data access
  - Analyst: Read-only access to reports and metrics
  - System: Automated processes and monitoring

#### 6.4.2 Data Security
- **Encryption at Rest**: AES-256 encryption for all stored data
- **Encryption in Transit**: TLS 1.3 for all API communications
- **Key Management**: Google Cloud KMS with automatic key rotation
- **Data Classification**:
  - Public: API documentation, system status
  - Internal: Feature engineering logic, model architecture
  - Confidential: Trading data, model parameters
  - Restricted: Client-specific configurations

#### 6.4.3 Compliance & Auditing
- **Audit Logging**: All API requests, configuration changes, system events
- **Data Retention**: 7 years for trading-related data (regulatory compliance)
- **Access Logging**: Complete audit trail for data access
- **Compliance Standards**: SOC 2 Type II, ISO 27001 readiness

### 6.5 Monitoring & Observability

#### 6.5.1 Metrics Collection
**Business Metrics**:
- `regime_prediction_accuracy`: Percentage of correct predictions
- `regime_transition_detection`: Timing accuracy of regime changes
- `model_confidence_distribution`: Distribution of confidence scores
- `feature_importance_drift`: Changes in feature importance over time

**Technical Metrics**:
- `api_request_duration_seconds`: Request processing time
- `ml_model_inference_duration_seconds`: Model prediction time
- `feature_engineering_duration_seconds`: Feature calculation time
- `data_pipeline_lag_seconds`: Data freshness lag
- `memory_usage_bytes`: Memory consumption per service
- `cpu_utilization_percentage`: CPU usage across services

**Infrastructure Metrics**:
- `vertex_ai_endpoint_replica_count`: Number of active replicas
- `bigquery_query_duration_seconds`: Query performance
- `cloud_storage_api_request_count`: Storage API usage
- `network_bytes_sent/received`: Network traffic

#### 6.5.2 Alerting Framework
```yaml
# Critical Alerts (PagerDuty integration)
- name: "High API Error Rate"
  condition: "error_rate > 1% for 5 minutes"
  severity: "critical"
  
- name: "Prediction Latency Breach"  
  condition: "p95_latency > 3 seconds for 3 minutes"
  severity: "critical"

- name: "Model Accuracy Degradation"
  condition: "accuracy < 90% for 1 hour"  
  severity: "critical"

# Warning Alerts (Slack integration)
- name: "Feature Pipeline Delay"
  condition: "pipeline_lag > 30 minutes"
  severity: "warning"
  
- name: "Unusual Traffic Pattern"
  condition: "request_rate > 2x baseline for 10 minutes"
  severity: "warning"
```

#### 6.5.3 Logging Strategy
```python
# Structured logging format
{
    "timestamp": "2025-08-04T14:30:15.123Z",
    "service": "regime-predictor",
    "level": "INFO",
    "correlation_id": "req_123456789",
    "user_id": "trader_001",
    "operation": "predict_regime",
    "input": {
        "asset": "nifty",
        "timestamp": "2025-08-04T14:30:00Z"
    },
    "output": {
        "regime": "STRONG_BULLISH_LOW_VOLATILE",
        "confidence": 0.92
    },
    "metrics": {
        "processing_time_ms": 2850,
        "feature_count": 51,
        "model_version": "v1.2.3"
    }
}
```

#### 6.5.4 Distributed Tracing
- **Trace Propagation**: OpenTelemetry standard across all services
- **Span Details**:
  - `data_ingestion`: BigQuery data retrieval
  - `feature_engineering`: Indicator calculations  
  - `ml_inference`: Model prediction
  - `response_formatting`: Output preparation
- **Trace Sampling**: 1% for production, 100% for staging
- **Trace Storage**: 30 days retention in Cloud Trace

### 6.6 Data Quality & Validation

#### 6.6.1 Data Quality Framework
**Input Data Validation**:
```python
# Real-time data validation rules
data_quality_rules = {
    'spot_price': {
        'min_value': 1000,      # Minimum spot price
        'max_value': 50000,     # Maximum spot price
        'null_tolerance': 0,     # No null values allowed
        'change_threshold': 0.1  # Max 10% change between updates
    },
    'implied_volatility': {
        'min_value': 0.01,      # 1% minimum IV
        'max_value': 2.0,       # 200% maximum IV
        'null_tolerance': 0.05,  # 5% null values acceptable
        'outlier_detection': 'z_score_3'
    },
    'greek_values': {
        'delta_range': [-1.0, 1.0],
        'gamma_min': 0.0,
        'theta_max': 0.0,
        'consistency_check': True  # Greeks must be mathematically consistent
    }
}
```

**Feature Validation**:
```python
# Feature quality validation
feature_validation = {
    'greek_sentiment_score': {
        'range': [-1.0, 1.0],
        'distribution_check': 'normal',
        'correlation_threshold': 0.3  # Min correlation with price moves
    },
    'oi_divergence_signal': {
        'range': [-1.0, 1.0], 
        'stability_check': True,      # No excessive volatility
        'lag_correlation': 0.4        # Predictive power check
    },
    'regime_confidence': {
        'range': [0.0, 1.0],
        'calibration_check': True,    # Well-calibrated probabilities
        'threshold_validation': 0.7   # Minimum confidence for decisions
    }
}
```

#### 6.6.2 Model Quality Assurance
**Model Performance Monitoring**:
- **Accuracy Tracking**: Daily accuracy reports vs actual market moves
- **Confusion Matrix**: Multi-class classification performance analysis
- **Precision/Recall**: Per-regime performance metrics
- **F1 Scores**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the curve for each regime class

**Model Drift Detection**:
```python
# Statistical drift detection
drift_detection = {
    'feature_drift': {
        'method': 'kolmogorov_smirnov',
        'threshold': 0.05,           # p-value threshold
        'window_size': 1000,         # Samples for comparison
        'alert_threshold': 0.02      # Alert if p < 0.02
    },
    'prediction_drift': {
        'method': 'population_stability_index',
        'threshold': 0.2,            # PSI threshold
        'monitoring_window': '7_days',
        'baseline_window': '30_days'
    },
    'performance_drift': {
        'accuracy_threshold': 0.05,  # 5% accuracy drop
        'confidence_threshold': 0.1, # 10% confidence drop
        'monitoring_frequency': 'hourly'
    }
}
```

#### 6.6.3 Business Logic Validation
**Regime Classification Validation**:
- **Transition Logic**: Validate regime transitions follow business rules
- **Consistency Checks**: Ensure regime components align (volatility + trend + structure)
- **Historical Validation**: Verify regime assignments match historical market conditions
- **Edge Case Testing**: Handle extreme market conditions (circuit breakers, expiry days)

**Parameter Validation**:
```python
# Configuration parameter validation
parameter_validation = {
    'business_logic': {
        'volatility_thresholds': 'ascending_order',  # Low < Med < High
        'trend_sensitivity': 'positive_correlation', # Higher = more sensitive
        'greek_weights': 'sum_to_one',              # Weights must sum to 1.0
        'time_decay': 'decreasing_function'         # Theta always decreasing
    },
    'financial_constraints': {
        'max_position_size': 'risk_management_limits',
        'correlation_bounds': '[-1.0, 1.0]',
        'probability_bounds': '[0.0, 1.0]',
        'return_calculations': 'mathematically_consistent'
    }
}
```

### 6.7 Testing Requirements

#### 6.7.1 Unit Testing Framework
**Coverage Requirements**:
- **Code Coverage**: Minimum 90% line coverage
- **Branch Coverage**: Minimum 85% branch coverage
- **Function Coverage**: 100% function coverage
- **Critical Path Coverage**: 100% for regime classification logic

**Test Categories**:
```python
# Unit test structure
test_categories = {
    'feature_engineering': {
        'greek_sentiment_tests': 25,      # Test cases for Greek analysis
        'oi_pa_analysis_tests': 20,       # OI/PA analysis test cases
        'iv_analytics_tests': 18,         # IV analysis test cases
        'straddle_analysis_tests': 15,    # Straddle strategy tests
        'technical_indicator_tests': 12   # Technical analysis tests
    },
    'model_logic': {  
        'prediction_tests': 30,           # Model prediction tests
        'ensemble_tests': 15,             # Ensemble logic tests
        'hyperparameter_tests': 20,       # Parameter mapping tests
        'validation_tests': 10            # Input validation tests
    },
    'integration': {
        'api_tests': 25,                  # API endpoint tests
        'database_tests': 15,             # BigQuery integration tests
        'pipeline_tests': 20,             # Data pipeline tests
        'configuration_tests': 12        # Config management tests
    }
}
```

#### 6.7.2 Integration Testing
**Data Pipeline Testing**:
- **End-to-End Data Flow**: CSV → GCS → BigQuery → Feature Engineering → ML Model
- **Data Transformation Validation**: Verify transformations preserve data integrity  
- **Error Handling**: Test graceful handling of malformed data
- **Performance Testing**: Validate processing times under various data volumes

**API Integration Testing**:
```python
# API test scenarios
api_test_scenarios = [
    {
        'name': 'successful_prediction',
        'input': 'valid_market_data.json',
        'expected_status': 200,
        'expected_response_time': '<3s',
        'validation': 'regime_format_valid'
    },
    {
        'name': 'invalid_input_handling',
        'input': 'malformed_data.json', 
        'expected_status': 400,
        'expected_error': 'validation_error'
    },
    {
        'name': 'high_load_testing',
        'concurrent_requests': 1000,
        'duration': '5_minutes',
        'success_threshold': '99%'
    }
]
```

#### 6.7.3 Performance Testing
**Load Testing Specifications**:
- **Baseline Load**: 100 requests/minute for 1 hour
- **Peak Load**: 1000 requests/minute for 30 minutes  
- **Stress Test**: 2000 requests/minute until failure
- **Endurance Test**: 500 requests/minute for 8 hours (full trading day)

**Performance Benchmarks**:
```python
# Performance test targets
performance_benchmarks = {
    'response_time': {
        'p50': '<2.0s',     # 50th percentile
        'p95': '<3.0s',     # 95th percentile  
        'p99': '<4.0s',     # 99th percentile
        'max': '<5.0s'      # Maximum acceptable
    },
    'throughput': {
        'sustained': '500 rps',      # Requests per second
        'peak': '1000 rps',
        'burst': '2000 rps'
    },
    'resource_usage': {
        'cpu_average': '<70%',
        'memory_peak': '<14GB',
        'disk_io': '<1000 IOPS'
    }
}
```

#### 6.7.4 Model Validation Testing
**Historical Backtesting**:
- **Time Period**: 2018-2025 (7 years of data)
- **Walk-Forward Analysis**: 1-year training, 3-month testing windows
- **Out-of-Sample Testing**: 20% holdout data never used in training
- **Cross-Validation**: 5-fold time-series cross-validation

**Model Comparison Testing**:
```python
# Model validation metrics
model_validation_metrics = {
    'classification_metrics': {
        'accuracy': '>95%',
        'precision_per_regime': '>90%',
        'recall_per_regime': '>85%',
        'f1_score_weighted': '>92%'
    },
    'calibration_metrics': {
        'brier_score': '<0.1',
        'reliability_diagram': 'well_calibrated',
        'confidence_accuracy_correlation': '>0.8'
    },
    'business_metrics': {
        'regime_transition_timing': '<5_minutes_delay',
        'false_signal_rate': '<10%',
        'missed_transition_rate': '<5%'
    }
}
```

#### 6.7.5 User Acceptance Testing
**Functional Testing Scenarios**:
- **Real-time Prediction**: Test with live market data during trading hours
- **Historical Analysis**: Verify batch processing of historical periods
- **Configuration Changes**: Test parameter updates and their effects
- **Error Recovery**: Validate system behavior during outages or errors

**User Experience Testing**:
- **API Usability**: Verify API contracts match existing system
- **Response Format**: Ensure output format matches user expectations
- **Documentation Accuracy**: Validate all API documentation with actual behavior
- **Migration Testing**: Seamless transition from existing system

---

## 7. Technical Architecture

### 7.1 System Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │  Google Cloud   │    │   Vertex AI     │
│                 │    │                 │    │                 │
│ • SSH Server    │───▶│ • Cloud Storage │───▶│ • ML Models     │
│ • Market Data   │    │ • BigQuery      │    │ • Predictions   │
│ • CSV Files     │    │ • Pub/Sub       │    │ • Monitoring    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Data Pipeline  │    │   Feature Eng   │    │   API Gateway   │
│                 │    │                 │    │                 │
│ • ETL Jobs      │    │ • Greek Sent.   │    │ • Rate Limiting │
│ • Validation    │    │ • OI/PA Analysis│    │ • Auth/Auth     │
│ • Monitoring    │    │ • IV Analytics  │    │ • Load Balancer │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 7.2 Component Details

#### 7.2.1 Data Layer
- **Cloud Storage**: Raw CSV files, model artifacts
- **BigQuery**: Structured data warehouse with partitioning
- **Pub/Sub**: Real-time data streaming
- **Cloud SQL**: Configuration and metadata storage

#### 7.2.2 Processing Layer
- **Dataflow**: ETL pipelines for data transformation
- **Cloud Functions**: Event-driven processing
- **Vertex AI Pipelines**: ML workflow orchestration
- **Cloud Run**: Containerized microservices

#### 7.2.3 ML Layer
- **Vertex AI Training**: Model training and hyperparameter tuning
- **Vertex AI Endpoints**: Model serving and prediction
- **Model Registry**: Version control and artifact management
- **Feature Store**: Feature management and serving

#### 7.2.4 API Layer
- **API Gateway**: Request routing and management
- **Cloud Load Balancer**: Traffic distribution
- **Cloud Armor**: DDoS protection and security
- **Identity & Access Management**: Authentication and authorization

### 7.3 Data Flow Architecture
```
1. Data Ingestion: SSH Server → Cloud Storage → BigQuery
2. Feature Engineering: BigQuery → Existing Indicators → Feature Vectors
3. ML Processing: Features → Vertex AI Model → Predictions
4. API Response: Predictions → API Gateway → Client Applications
5. Monitoring: All components → Cloud Monitoring → Alerts
```

---

## 8. Implementation Plan

### 8.1 Phase 1: Foundation Setup (Weeks 1-2)
**Objectives**: Establish Google Cloud infrastructure and data migration

**Tasks**:
- Set up Google Cloud project and enable required APIs
- Create service accounts and configure IAM permissions
- Set up Cloud Storage buckets with appropriate lifecycle policies
- Design and create BigQuery dataset with optimized schema
- Implement data transfer pipeline from SSH server to GCS
- Validate data integrity and completeness post-migration

**Deliverables**:
- Fully configured Google Cloud environment
- 32GB historical data successfully migrated to BigQuery
- Data validation reports confirming 100% accuracy
- Automated data pipeline for ongoing CSV file ingestion

**Success Criteria**:
- All historical data (2018-2025) accessible in BigQuery
- Query performance <10 seconds for typical regime analysis queries
- Zero data loss or corruption during migration

### 8.2 Phase 2: Feature Engineering Integration (Weeks 3-4)
**Objectives**: Adapt existing feature engineering modules for cloud deployment

**Tasks**:
- Containerize existing Python feature engineering modules
- Create cloud-compatible versions of indicator classes
- Implement BigQuery integration for real-time feature calculation
- Deploy feature engineering services on Cloud Run
- Create feature store for preprocessing and serving
- Build parameter mapping system (Excel → ML hyperparameters)

**Deliverables**:
- Dockerized feature engineering services
- Cloud-native versions of all existing indicators
- Feature store with 50+ engineered features
- Parameter configuration service
- Performance benchmarks matching current <3s target

**Success Criteria**:
- Feature engineering latency <2 seconds
- 100% parity with existing indicator calculations
- All 600+ Excel parameters successfully mapped to ML hyperparameters

### 8.3 Phase 3: ML Model Development (Weeks 5-6)
**Objectives**: Develop and train ML models using existing feature engineering

**Tasks**:
- Design ensemble model architecture (Random Forest + Gradient Boosting)
- Implement training pipeline with hyperparameter optimization
- Train models on 7 years of historical data
- Validate model performance against existing regime classification
- Implement model versioning and A/B testing framework
- Create model monitoring and drift detection systems

**Deliverables**:
- Trained ML models achieving ≥95% accuracy
- Vertex AI endpoints for real-time prediction
- Model evaluation reports and performance benchmarks
- Automated retraining pipeline
- Model monitoring dashboard

**Success Criteria**:
- Model accuracy ≥95% vs existing system on test data
- Prediction latency <1 second
- Confidence scores properly calibrated
- Model artifacts stored and versioned in Model Registry

### 8.4 Phase 4: API Development (Weeks 7-8)
**Objectives**: Build production-ready APIs for real-time and batch processing

**Tasks**:
- Implement REST APIs for regime prediction and configuration
- Create batch processing endpoints for historical analysis
- Build authentication and authorization system
- Implement rate limiting and request validation
- Create comprehensive API documentation
- Deploy API gateway with load balancing

**Deliverables**:
- Production-ready REST APIs
- API documentation and client SDKs
- Authentication and security implementation
- Load testing results and performance optimization
- API monitoring and analytics dashboard

**Success Criteria**:
- API response time <3 seconds end-to-end
- Handle 1000+ concurrent requests
- 99.9% API availability
- Comprehensive error handling and logging

### 8.5 Phase 5: Integration & Testing (Weeks 9-10)
**Objectives**: Integrate all components and conduct comprehensive testing

**Tasks**:
- End-to-end integration testing
- Performance testing under production load
- Security testing and vulnerability assessment
- User acceptance testing with existing workflows
- Create monitoring dashboards and alerting rules
- Prepare production deployment procedures

**Deliverables**:
- Fully integrated system ready for production
- Comprehensive test results and performance benchmarks
- Security assessment report
- Monitoring and alerting configuration
- Production deployment runbook

**Success Criteria**:
- All functional requirements met and tested
- Performance targets achieved under load testing
- Security requirements satisfied
- Zero critical or high-priority bugs

### 8.6 Phase 6: Production Deployment (Weeks 11-12)
**Objectives**: Deploy to production with gradual rollout and monitoring

**Tasks**:
- Deploy to production environment
- Implement gradual traffic migration (canary deployment)
- Monitor system performance and business metrics
- Fine-tune configuration based on production data
- Complete knowledge transfer and documentation
- Establish ongoing maintenance procedures

**Deliverables**:
- Live production system serving real traffic
- Performance monitoring and alerting in place
- Complete documentation and operational procedures
- Team training and knowledge transfer materials
- Post-deployment review and lessons learned

**Success Criteria**:
- Successful zero-downtime production deployment
- All SLAs met in production environment
- User satisfaction with system performance
- Smooth transition from existing system

---

## 9. Success Criteria & KPIs

### 9.1 Technical KPIs
- **Accuracy**: ≥95% regime classification accuracy vs existing system
- **Latency**: <3 seconds for real-time regime prediction
- **Availability**: 99.9% uptime during market hours (9:15 AM - 3:30 PM IST)
- **Throughput**: Handle 1000+ concurrent prediction requests
- **Data Processing**: Process daily CSV files within 30 minutes of arrival

### 9.2 Business KPIs
- **Cost Optimization**: <$500/month operational costs
- **Feature Parity**: 100% of existing functionality preserved
- **Parameter Support**: All 600+ Excel parameters functional
- **Migration Success**: Zero data loss during transition
- **User Satisfaction**: >90% user satisfaction with new system

### 9.3 Operational KPIs
- **Deployment Success**: Zero-downtime production deployment
- **Mean Time to Recovery**: <15 minutes for system issues
- **Error Rate**: <0.1% API error rate
- **Model Drift**: Detect performance degradation within 24 hours
- **Security**: Zero security incidents or data breaches

---

## 10. Risk Assessment & Mitigation

### 10.1 Technical Risks

#### 10.1.1 Performance Degradation
**Risk**: ML model predictions slower than existing system
**Impact**: High - Affects trading decisions
**Probability**: Medium
**Mitigation**: 
- Extensive performance testing during development
- Caching layer for frequently requested predictions
- Fallback to existing system if latency exceeds threshold

#### 10.1.2 Model Accuracy Issues
**Risk**: ML model less accurate than existing rule-based system
**Impact**: High - Incorrect regime classification affects trading
**Probability**: Low
**Mitigation**:
- Use existing feature engineering proven in production
- Ensemble approach combining multiple model types
- A/B testing to validate performance before full rollout

#### 10.1.3 Data Migration Issues
**Risk**: Data corruption or loss during migration
**Impact**: High - Loss of historical analysis capability
**Probability**: Low
**Mitigation**:
- Comprehensive data validation checks
- Parallel systems during migration period
- Complete backup of original data before migration

### 10.2 Operational Risks

#### 10.2.1 Cloud Provider Dependency
**Risk**: Vendor lock-in with Google Cloud
**Impact**: Medium - Future flexibility concerns
**Probability**: High
**Mitigation**:
- Use standard APIs and containerized deployments
- Document migration procedures to other providers
- Maintain multi-cloud disaster recovery option

#### 10.2.2 Cost Overruns
**Risk**: Cloud costs exceed budget expectations
**Impact**: Medium - Financial impact on project
**Probability**: Medium
**Mitigation**:
- Detailed cost modeling and monitoring
- Implement cost controls and budgets
- Regular cost optimization reviews

#### 10.2.3 Team Knowledge Gap
**Risk**: Team lacks ML and cloud expertise
**Impact**: Medium - Project delays and quality issues
**Probability**: Medium
**Mitigation**:
- Comprehensive training program for team
- Engage cloud specialists for complex implementations
- Detailed documentation and knowledge transfer

### 10.3 Business Risks

#### 10.3.1 Regulatory Compliance
**Risk**: New system doesn't meet regulatory requirements
**Impact**: High - Legal and compliance issues
**Probability**: Low
**Mitigation**:
- Early engagement with compliance team
- Maintain audit trails and data lineage
- Ensure all regulatory requirements are documented

#### 10.3.2 User Adoption Issues
**Risk**: Users resistant to change from existing system
**Impact**: Medium - Reduced system utilization
**Probability**: Medium
**Mitigation**:
- Maintain existing API contracts for smooth transition
- Comprehensive user training and support
- Gradual rollout with feedback incorporation

---

## 11. Budget & Resource Requirements

### 11.1 Google Cloud Costs (Monthly)

#### 11.1.1 Storage Costs
- **Cloud Storage**: $30/month (32GB data + growth)
- **BigQuery Storage**: $40/month (compressed data + queries)
- **Backup Storage**: $20/month (disaster recovery)

#### 11.1.2 Compute Costs
- **Vertex AI Training**: $100/month (model retraining)
- **Vertex AI Endpoints**: $150/month (2 instances for HA)
- **Cloud Run**: $50/month (feature engineering services)
- **Cloud Functions**: $20/month (data processing)

#### 11.1.3 Network & Services
- **API Gateway**: $30/month (request processing)
- **Load Balancer**: $25/month (traffic distribution)
- **Monitoring**: $25/month (logging and metrics)

**Total Monthly Cost**: ~$490/month (within $500 budget)

### 11.2 Development Resources

#### 11.2.1 Team Composition
- **Technical Lead**: 1 FTE (12 weeks)
- **ML Engineer**: 