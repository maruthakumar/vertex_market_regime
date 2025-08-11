# Story 1.5a: Component 4 IV Percentile Enhancement (87 Total Features)

## Status
Ready for Development

## Story
**As a** quantitative developer,
**I want** to implement the Component 4 IV Percentile Analysis system with sophisticated individual DTE tracking (dte=0...dte=58), 7-regime classification, 4-timeframe momentum analysis, and advanced IVP+IVR implementation using comprehensive production schema alignment,
**so that** Component 4 provides institutional-grade implied volatility percentile context with maximum granularity and precision for superior market regime determination within the 8-component framework.

## Acceptance Criteria
1. **Enhanced Component 4 Feature Engineering Complete (Epic 1 Enhanced Scope)**
   - Exactly 87 total features materialized and validated with institutional-grade sophistication
   - Individual DTE-level IV percentile tracking (dte=0, dte=1, dte=2...dte=58) with granular analysis
   - Zone-wise IV percentile analysis across 4 production zones (MID_MORN/LUNCH/AFTERNOON/CLOSE)
   - Historical IV ranking system with rolling 252-day lookbook window for each individual DTE
   - Multi-strike IV percentile aggregation using ALL available strikes (54-68 per expiry)
   - Production schema alignment with parquet database structure (48 columns - production validated)

2. **Sophisticated IV Percentile Analysis System Complete**
   - Complete individual DTE-specific IV percentile construction using ce_iv/pe_iv columns (100% coverage)
   - Individual DTE-level IV percentile tracking (dte=0, dte=1, dte=2...dte=58) with dedicated historical percentile database
   - Advanced 7-regime classification system (Extremely Low, Very Low, Low, Normal, High, Very High, Extremely High) for each DTE level
   - Comprehensive zone-specific percentile analysis leveraging zone_name column (MID_MORN/LUNCH/AFTERNOON/CLOSE)
   - ATM strike IV percentile tracking with atm_strike column integration at individual DTE granularity
   - Advanced 4-timeframe IV percentile momentum (5min/15min/30min/1hour) with DTE-specific momentum tracking
   - Cross-strike IV percentile correlation analysis across the complete chain for each DTE level
   - Sophisticated IVP + IVR integration with percentile band mapping

3. **Enhanced Performance Targets Achieved (Epic 1 Enhanced Scope)**
   - Processing time <350ms per component as specified in updated Epic 1 Story 5
   - Memory efficiency validation <250MB per component (optimized for percentile calculations)
   - IV percentile accuracy >95% using historical validation against production data
   - Integration with existing Component 4 IV Skew framework using shared schema
   - Scalable processing architecture handling variable strike counts and DTE scenarios

4. **Production Data Integration & Testing Framework**
   - Complete integration with production parquet schema (48 columns: '/Users/maruth/projects/market_regime/docs/parquote_database_schema_sample.csv')
   - Validation using production data (78+ parquet files: '/Users/maruth/projects/market_regime/data/nifty_validation/backtester_processed/')
   - Testing across 6 expiry folders with comprehensive DTE coverage (3-360+ days)
   - DTE-adaptive percentile calculation handling varying expiry scenarios
   - Historical IV database construction for percentile baseline establishment
   - Cross-validation with existing IV skew analysis for consistency verification

## Tasks / Subtasks

- [ ] **Implement Production Schema-Aligned IV Percentile Data Extraction** (AC: 1, 4)
  - [ ] **SCHEMA ALIGNMENT**: Implement data extraction using production parquet columns (trade_date, trade_time, expiry_date, dte, zone_name, atm_strike, strike, ce_iv, pe_iv)
  - [ ] **DTE BUCKETING & INDIVIDUAL DTE TRACKING**: Create DTE-specific buckets (0-7: Near, 8-30: Medium, 31+: Far) AND individual DTE-level tracking (dte=0, dte=1...dte=58)
  - [ ] **ZONE EXTRACTION**: Extract zone-specific IV data using zone_name column (MID_MORN/LUNCH/AFTERNOON/CLOSE) for each individual DTE per production schema
  - [ ] **MULTI-STRIKE AGGREGATION**: Process ALL available strikes (54-68 per expiry) with ce_iv/pe_iv percentile calculation at individual DTE granularity
  - [ ] **DATA VALIDATION**: Implement IV data quality checks and missing value handling for production reliability across all DTE levels

- [ ] **Build Historical IV Percentile Database Engine** (AC: 1, 2)
  - [ ] **ROLLING WINDOW SYSTEM**: Implement 252-day rolling window for historical IV percentile baseline across individual DTE levels
  - [ ] **DTE-SPECIFIC STORAGE**: Create separate historical databases for each DTE bucket AND individual DTE (dte=0, dte=1...dte=58) with efficient retrieval
  - [ ] **ZONE-SPECIFIC TRACKING**: Build zone-wise historical IV storage using zone_name for intraday pattern analysis at each DTE level
  - [ ] **STRIKE-LEVEL PERCENTILES**: Calculate percentiles at individual strike levels and aggregate to surface-level metrics for each DTE
  - [ ] **PERCENTILE INTERPOLATION**: Implement smooth percentile calculation for intermediate values and edge cases across DTE spectrum

- [ ] **Create DTE-Adaptive IV Percentile Analysis Framework** (AC: 1, 2)
  - [ ] **BUCKET-SPECIFIC & INDIVIDUAL DTE CALCULATION**: Implement both DTE bucket-specific (Near: 0-7, Medium: 8-30, Far: 31+) AND individual DTE percentile calculation (dte=0, dte=1...dte=58)
  - [ ] **CROSS-DTE PERCENTILE COMPARISON**: Analyze percentile differences across DTE buckets AND individual DTEs for regime transition detection
  - [ ] **INDIVIDUAL DTE PERCENTILE TRACKING**: Build granular DTE-level percentile tracking with dte=0, dte=1, dte=2...dte=58 historical analysis
  - [ ] **EXPIRY-SPECIFIC PERCENTILES**: Handle varying expiry scenarios using expiry_date and expiry_bucket columns at individual DTE granularity
  - [ ] **ATM-RELATIVE PERCENTILES**: Calculate ATM-relative percentiles using atm_strike column for moneyness analysis at each DTE level
  - [ ] **DYNAMIC PERCENTILE THRESHOLDS**: Implement adaptive percentile thresholds based on market conditions and volatility clustering per DTE

- [ ] **Implement Zone-Wise IV Percentile Analysis System** (AC: 1, 2)
  - [ ] **INTRADAY ZONE ANALYSIS**: Calculate zone-specific IV percentiles using zone_name (MID_MORN/LUNCH/AFTERNOON/CLOSE) at individual DTE levels per production schema
  - [ ] **ZONE TRANSITION TRACKING**: Monitor IV percentile changes across trading zones for regime shift detection at each DTE
  - [ ] **CROSS-ZONE CORRELATION**: Analyze IV percentile correlations between different trading zones for each DTE level
  - [ ] **ZONE-SPECIFIC REGIME CLASSIFICATION**: Implement 7-level regime classification per zone AND per DTE (Extremely Low to Extremely High)
  - [ ] **INTRADAY MOMENTUM CALCULATION**: Calculate zone-to-zone IV percentile momentum for trend detection across DTE spectrum

- [ ] **Build Advanced IV Percentile Regime Classification** (AC: 2, 3)
  - [ ] **7-REGIME CLASSIFICATION**: Implement comprehensive regime levels (Extremely Low, Very Low, Low, Normal, High, Very High, Extremely High)
  - [ ] **REGIME TRANSITION PROBABILITY**: Calculate transition probabilities between percentile regimes using historical patterns
  - [ ] **REGIME STABILITY METRICS**: Measure regime persistence and stability across different timeframes
  - [ ] **CROSS-STRIKE REGIME CONSISTENCY**: Validate regime classification consistency across the complete strike chain
  - [ ] **REGIME CONFIDENCE SCORING**: Implement confidence metrics based on historical data quality and regime stability

- [ ] **Create Multi-Timeframe IV Percentile Momentum System** (AC: 2)
  - [ ] **MOMENTUM CALCULATION**: Implement IV percentile momentum across 5min/15min/30min/1hour timeframes
  - [ ] **MOMENTUM REGIME CLASSIFICATION**: Classify momentum into trending/mean-reverting/sideways regimes
  - [ ] **ACCELERATION TRACKING**: Calculate second-order momentum (acceleration) for regime change prediction
  - [ ] **MOMENTUM DIVERGENCE**: Detect divergences between price movement and IV percentile momentum
  - [ ] **CROSS-TIMEFRAME CORRELATION**: Analyze momentum correlations across different timeframes

- [ ] **Implement Enhanced Feature Extraction Framework** (AC: 1, 3)
  - [ ] **87-FEATURE EXTRACTION**: Extract exactly 87 total features (50 existing + 37 percentile enhancements) for Epic 1 compliance
  - [ ] **PERCENTILE SURFACE FEATURES**: Calculate surface-level percentile metrics aggregated across all strikes
  - [ ] **DTE-BUCKET FEATURES**: Extract DTE bucket-specific features (15 features per bucket = 45 features)
  - [ ] **INDIVIDUAL DTE FEATURES**: Extract individual DTE-level percentile features (dte=0, dte=1...dte=7 = 8 features for near-term)
  - [ ] **ZONE-SPECIFIC FEATURES**: Calculate zone-wise percentile features (5 features per zone = 25 features)
  - [ ] **MOMENTUM FEATURES**: Extract momentum-based features across multiple timeframes (5 features)
  - [ ] **REGIME FEATURES**: Calculate regime classification and transition features (2 features)

- [ ] **Build Production Integration & Testing Suite** (AC: 3, 4)
  - [ ] **PRODUCTION DATA TESTING**: Test with actual production data from '/Users/maruth/projects/market_regime/data/nifty_validation/backtester_processed/'
  - [ ] **SCHEMA COMPLIANCE TESTING**: Validate complete compatibility with production parquet schema structure
  - [ ] **PERFORMANCE BENCHMARKING**: Achieve <150ms processing and <250MB memory targets with production data
  - [ ] **CROSS-VALIDATION TESTING**: Validate IV percentile accuracy >95% against historical data
  - [ ] **FRAMEWORK INTEGRATION TESTING**: Test integration with existing Component 4 IV Skew analysis system

## Dev Notes

### Architecture Context
This story implements enhanced IV Percentile analysis for Component 4 of the 8-component adaptive learning system described in **Epic 1: Feature Engineering Foundation**. The component extends the existing IV Skew Analysis with comprehensive percentile-based features using production-aligned schema.

**Critical Performance Requirements:**
- **Component Budget**: <350ms processing time per component (Epic 1 enhanced scope - approved for institutional-grade sophistication)
- **Memory Constraint**: <250MB per component (optimized from 300MB for percentile calculations)  
- **Feature Count**: Exactly 87 total features (Epic 1 compliance with sophisticated percentile enhancements)
- **Accuracy Target**: >95% IV percentile accuracy vs historical pattern validation
- **Framework Integration**: Seamless integration with existing Component 4 IV Skew system

**Production Schema Alignment:** [Source: /Users/maruth/projects/market_regime/docs/parquote_database_schema_sample.csv]
- **Primary Columns**: trade_date, trade_time, expiry_date, dte, expiry_bucket, zone_name, atm_strike, strike
- **IV Data Columns**: ce_iv, pe_iv (100% coverage across all strikes)
- **Supporting Columns**: spot, index_name, call_strike_type, put_strike_type
- **Greeks Columns**: ce_delta, pe_delta, ce_gamma, pe_gamma, ce_vega, pe_vega, ce_theta, pe_theta, ce_rho, pe_rho
- **Volume/OI Columns**: ce_volume, pe_volume, ce_oi, pe_oi, ce_coi, pe_coi
- **Zone Integration**: zone_id, zone_name (OPEN, MID_MORN, LUNCH, AFTERNOON, CLOSE)

**Data Models**: [Source: Production Data Analysis]
- **Primary Data Format**: Parquet files with 47-column production schema (78+ files available)
- **DTE Coverage**: Complete spectrum from 3-58 days with varying strike counts per DTE
- **Zone Coverage**: Full intraday coverage across 5 trading zones with time-based analysis
- **Strike Coverage**: 54-68 strikes per expiry with complete ce_iv/pe_iv data
- **Historical Depth**: 252-day rolling window for percentile baseline establishment
- **Temporal Resolution**: Minute-level data with zone-based aggregation capabilities

**API Specifications**: [Source: docs/architecture/tech-stack.md - API Integration Strategy section]
- **Processing Framework**: Apache Arrow → RAPIDS cuDF for GPU-accelerated percentile calculation
- **Memory Management**: Zero-copy data access patterns with optimized percentile storage
- **Component Interface**: Extension of existing AdaptiveComponent base class
- **Performance Monitoring**: Real-time component status tracking with <150ms validation

**Component Specifications**: [Source: Production Data Analysis + Architecture]
- **IV Percentile Engine**: Historical percentile calculation using 252-day rolling window
- **DTE Bucket System**: Three-tier bucketing (Near: 0-7, Medium: 8-30, Far: 31+) 
- **Zone Analysis Framework**: Five-zone percentile analysis with intraday pattern detection
- **Regime Classification**: Seven-level regime system (Extremely Low to Extremely High)
- **Quality Thresholds**: Minimum 30 data points for reliable percentile calculation
- **Performance Scaling**: Adaptive processing for variable strike counts and DTE scenarios

**File Locations**: [Source: docs/architecture/source-tree.md - Core New System: vertex_market_regime section]
- **Component Directory**: `vertex_market_regime/src/vertex_market_regime/components/component_04_iv_skew/`
- **Enhanced Modules**: 
  - `iv_percentile_analyzer.py` - Core IV percentile calculation engine
  - `dte_percentile_framework.py` - DTE-specific percentile analysis
  - `zone_percentile_tracker.py` - Zone-wise percentile tracking
  - `percentile_regime_classifier.py` - Enhanced regime classification
- **Test Location**: `vertex_market_regime/tests/unit/components/test_component_04_percentile.py`
- **Integration Tests**: `vertex_market_regime/tests/integration/test_component_04_enhanced_integration.py`

**Testing Requirements**: [Source: docs/architecture/coding-standards.md - Testing Standards section]
- **Framework**: pytest with production data fixtures for IV percentile validation
- **Coverage**: 95% minimum for new percentile analysis code
- **Performance Tests**: Validate <150ms processing, <250MB memory budgets with production data
- **Production Data**: Use actual Parquet files from '/Users/maruth/projects/market_regime/data/nifty_validation/backtester_processed/'
- **Cross-Validation**: Validate percentile accuracy >95% against historical benchmarks

**Technical Constraints**: [Source: docs/architecture/tech-stack.md - Performance Specifications section]
- **GPU Acceleration**: RAPIDS cuDF for large-scale percentile calculations across historical data
- **Memory Pool**: Optimized Arrow memory pool with efficient percentile storage patterns
- **Processing Pipeline**: Parquet → Arrow → GPU pathway for IV percentile analysis
- **Fallback Strategy**: CPU pandas fallback with optimized percentile algorithms

### Testing
**Testing Standards from Architecture:**
- **Framework**: pytest with production data fixtures for IV percentile validation [Source: architecture/coding-standards.md#testing-standards]
- **Location**: `vertex_market_regime/tests/unit/components/test_component_04_percentile.py`
- **Coverage**: 95% minimum for new percentile analysis code [Source: docs/architecture/coding-standards.md - Quality Assurance Standards section]
- **Integration Tests**: `vertex_market_regime/tests/integration/test_component_04_enhanced_integration.py`
- **Performance Tests**: Validate <150ms processing, <250MB memory budgets with production loads

**Component Test Requirements:**
1. **Production Schema Validation**: Test complete compatibility with production parquet structure and column alignment
2. **DTE-Specific Percentile Calculation**: Validate DTE bucket-specific percentile accuracy across Near/Medium/Far categories
3. **Zone-Wise Percentile Analysis**: Test zone-specific percentile calculation using zone_name column data
4. **Historical Window Testing**: Validate 252-day rolling window percentile calculation accuracy and performance
5. **Multi-Strike Percentile Aggregation**: Test percentile calculation across ALL 54-68 strikes with surface-level aggregation
6. **Regime Classification Accuracy**: Validate 7-level regime classification with >95% accuracy against historical benchmarks
7. **Performance Benchmarks**: <150ms, <250MB compliance with production data loads and variable strike counts
8. **Framework Integration Tests**: Enhanced Component 4 integration with existing IV Skew analysis system

**Production Testing Strategy:**
- **Primary Data Source**: 78+ validated Parquet files across 6 expiry folders at '/Users/maruth/projects/market_regime/data/nifty_validation/backtester_processed/'
- **Schema Compliance Testing**: Complete validation against 48-column production schema '/Users/maruth/projects/market_regime/docs/parquote_database_schema_sample.csv'
- **DTE Coverage Testing**: Test across full DTE spectrum (3-58 days) with varying strike count scenarios
- **Zone-Based Testing**: Validate zone-specific percentile calculation across 4 production zones (MID_MORN/LUNCH/AFTERNOON/CLOSE)
- **Historical Accuracy Testing**: Cross-validate percentile calculations against known historical patterns
- **Variable Load Testing**: Test performance with different strike counts (54/68/64) and DTE combinations
- **Memory Optimization Testing**: Validate <250MB compliance with large historical datasets
- **Cross-System Integration**: Test integration with existing Component 4 IV Skew framework

## Change Log
| Date | Version | Description | Author |
|------|---------|-------------|--------|
| 2025-08-11 | 1.0 | Initial story creation for Component 4 IV Percentile Feature Engineering with production schema alignment | SM |
| 2025-08-11 | 1.1 | Epic 1 compliance corrections: 87 total features, 200ms budget, simplified scope for feasibility | SM |
| 2025-08-11 | 2.0 | Epic 1 enhanced scope restoration: Individual DTE tracking, 7-regime classification, 4-timeframe momentum, 350ms budget - PO approved institutional-grade sophistication | SM |
| 2025-08-11 | 2.1 | Production schema alignment: 48 columns, 4 zones (MID_MORN/LUNCH/AFTERNOON/CLOSE), 78+ parquet files across 6 expiry folders, Component 4 doc updated | SM |

## Dev Agent Record

### Agent Model Used
Claude Sonnet 4 (claude-sonnet-4-20250514)

### Completion Notes
🚧 **STORY READY FOR DEVELOPMENT** - All acceptance criteria and tasks defined with Epic 1 enhanced scope:

**Key Implementation Requirements:**
1. **Production Schema Alignment**: Complete alignment with 48-column parquet structure including trade_date, expiry_date, dte, zone_name, ce_iv, pe_iv
2. **Individual DTE Analysis**: Granular DTE-level analysis (dte=0, dte=1, dte=2...dte=58) for maximum precision
3. **Advanced Zone-Wise Percentiles**: Four-zone analysis using zone_name (MID_MORN/LUNCH/AFTERNOON/CLOSE) per production schema for institutional-grade sophistication
4. **Sophisticated Historical Database**: 252-day rolling window percentile baseline for each individual DTE with efficient storage and retrieval
5. **87-Feature Framework**: Exactly 87 total features (50 existing + 37 sophisticated enhancements) for Epic 1 enhanced scope
6. **Enhanced Performance Budget**: <350ms processing as specified in updated Epic 1, <250MB memory with production data loads
7. **Advanced Cross-Validation**: >95% percentile accuracy against historical benchmarks using production data with 7-regime validation

**Production Integration Highlights:**
- **Data Source**: '/Users/maruth/projects/market_regime/data/nifty_validation/backtester_processed/' (78+ files)
- **Schema Reference**: '/Users/maruth/projects/market_regime/docs/parquote_database_schema_sample.csv'
- **Framework Integration**: Seamless integration with existing Component 4 IV Skew system
- **Testing Strategy**: Comprehensive production data validation with performance benchmarking

### 📊 **Component 4 Sophisticated Feature Categories (87 Total Features)**

**Core IV Skew Features** (50 features - existing Epic 1 scope):
- Put/Call IV skew analysis (15 features)
- Term structure signals (15 features)
- DTE-specific vs range aggregation (20 features)

**Advanced IV Percentile Enhancement Features** (37 features - institutional-grade additions):

1. **Individual DTE Percentile Features** (16 features)
   - Critical Near-Term DTEs (dte=0,1,2,3): 8 features (2 per DTE: percentile + regime)
   - Key Weekly DTEs (dte=4,5,6,7): 8 features (2 per DTE: percentile + regime)

2. **Zone-Specific Percentile Features** (8 features)
   - MID_MORN zone: 2 features (percentile, regime transition)
   - LUNCH zone: 2 features (percentile, regime transition)
   - AFTERNOON zone: 2 features (percentile, regime transition)
   - CLOSE zone: 2 features (percentile, regime transition)

3. **Advanced 7-Regime Classification Features** (4 features)
   - Regime classification confidence (7-level system)
   - Regime transition probability matrix
   - Regime persistence score
   - Cross-DTE regime consistency

4. **4-Timeframe Momentum Features** (4 features)
   - 5-minute IV percentile momentum
   - 15-minute IV percentile momentum
   - 30-minute IV percentile momentum
   - 1-hour IV percentile momentum

5. **IVP + IVR Integration Features** (5 features)
   - Combined IVP+IVR score
   - Historical ranking percentile  
   - Cross-timeframe momentum correlation
   - Production schema compatibility layer
   - Cross-zone percentile normalization

**Total: 87 features (50 existing + 37 sophisticated enhancements) - Epic 1 enhanced scope compliant**

**Production Schema Breakdown**: 16 + 8 + 4 + 4 + 5 = 37 enhancement features aligned with 48-column production schema

The story is **ready for development** with Epic 1 compliance and alignment to production schema patterns.

### File List
**Files to be Created/Modified:**
- `vertex_market_regime/src/components/component_04_iv_skew/iv_percentile_analyzer.py` - Core IV percentile analysis engine
- `vertex_market_regime/src/components/component_04_iv_skew/dte_percentile_framework.py` - DTE-specific percentile framework
- `vertex_market_regime/src/components/component_04_iv_skew/zone_percentile_tracker.py` - Zone-wise percentile tracking system
- `vertex_market_regime/src/components/component_04_iv_skew/percentile_regime_classifier.py` - Enhanced regime classification
- `vertex_market_regime/tests/unit/components/test_component_04_percentile.py` - Comprehensive production testing suite
- `docs/stories/story.1.5a.component-4-ivpercentile-feature-engineering.md` - This story document

## QA Results
**Epic 1 Enhanced Scope Implementation (2025-08-11):**
- ✅ Epic 1 enhanced scope achieved: 87 total features with institutional-grade sophistication
- ✅ Performance budget enhanced: <350ms (Epic 1 updated specification)
- ✅ Template compliance: Standard user story format applied
- ✅ Full technical sophistication restored: Individual DTE tracking, 7-regime classification, 4-timeframe momentum, 5-zone analysis
- ✅ PO approved institutional-grade implementation approach
- ✅ Ready for final validation and sophisticated dev implementation

*This section will be populated by the QA agent during story validation*