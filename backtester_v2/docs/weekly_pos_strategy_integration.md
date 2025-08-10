# Weekly & Positional Strategy Integration Plan

## Executive Summary

This document outlines the comprehensive implementation plan for integrating Weekly and Monthly Positional strategies (Calendar Spread & Iron Fly with Reverse Buying) into the HeavyDB backtesting system. The implementation focuses on extreme configurability to test various market conditions and parameters.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Implementation Phases](#implementation-phases)
3. [Core Components](#core-components)
4. [Configuration System](#configuration-system)
5. [Testing Strategy](#testing-strategy)
6. [Performance Optimization](#performance-optimization)
7. [Timeline & Milestones](#timeline--milestones)

## Architecture Overview

### System Design Principles

1. **Strategy Agnostic Core**: The system will support both weekly and monthly timeframes through configuration
2. **Parameter-Driven Logic**: All thresholds, rules, and conditions exposed as configurable parameters
3. **Modular Adjustments**: Each adjustment type (roll, reverse buy, double calendar) as pluggable modules
4. **Market Structure Aware**: Integrated support/resistance detection with multiple methods
5. **Performance First**: GPU-optimized queries for all computations
6. **Volatility Intelligence**: Advanced volatility metrics (IVP, IVR, ATR percentile, term structure)

### High-Level Architecture

```
┌─────────────────────┐
│   Excel/YAML Input  │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Parser & Validator │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│   Model Layer       │
│  - PositionalModel  │
│  - AdjustmentRules  │
│  - SRConfig         │
│  - VolatilityConfig │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Strategy Processor │
│  - Entry Logic      │
│  - Adjustment Engine│
│  - Exit Rules       │
│  - Vol Analysis     │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  HeavyDB Queries    │
│  - Option Chain     │
│  - Greeks & IV      │
│  - S/R Levels       │
│  - Vol Metrics      │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Risk Engine        │
│  - Position Limits  │
│  - Greek Limits     │
│  - P&L Tracking     │
│  - Vol Exposure     │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Output Generation  │
└─────────────────────┘
```

## Implementation Phases

### Phase 1: Foundation (Days 1-3)

#### 1.1 Model Creation
- Create `models/positional.py` with comprehensive configuration models
- Implement `PositionalStrategyModel` supporting both weekly and monthly
- Design `AdjustmentRuleModel` for all adjustment types
- Build `MarketStructureConfig` for S/R integration
- **NEW**: Create `VolatilityMetricsConfig` for IVP, IVR, ATR percentile

#### 1.2 Database Schema Updates
- Add VIX data integration tables if not present
- Create `positional_adjustments` tracking table
- Add `sr_levels` table for external S/R data
- Update option chain views for positional queries
- **NEW**: Create `volatility_metrics` table for IVP/IVR/ATR history
- **NEW**: Add `iv_term_structure` view for volatility analysis

#### 1.3 Parser Extensions
- Extend `excel_parser/portfolio_parser.py` for PositionalParameter sheet
- Add validation for 100+ configuration parameters
- Implement YAML parser for CI/CD compatibility
- Create parameter inheritance system (strategy → leg level)
- **NEW**: Add volatility metric columns parsing

### Phase 2: Core Strategy Engine (Days 4-7)

#### 2.1 Entry Logic Implementation
- Time window filtering with intraday granularity
- VIX-based premium band selection
- Support/Resistance proximity checks
- Market structure validation (trend, range, volatility)
- Event calendar integration
- **NEW**: IVP/IVR filters for entry timing
- **NEW**: ATR percentile for volatility regime detection
- **NEW**: IV term structure analysis

#### 2.2 Adjustment Engine
- State machine for position lifecycle
- Roll adjustment logic (up, down, diagonal)
- Reverse buying implementation (4 parts)
- Double calendar conversion
- Dynamic hedging adjustments
- **NEW**: Volatility-based adjustment triggers
- **NEW**: IV crush protection logic

#### 2.3 Exit Logic
- Time-based exits (scheduled, expiry)
- Profit target with trailing options
- Stop loss with recovery logic
- Greek-based exits (delta, gamma breaches)
- Market structure exits (S/R break)
- **NEW**: Volatility expansion/contraction exits
- **NEW**: IVP extreme exits

### Phase 3: Advanced Features (Days 8-10)

#### 3.1 Market Structure Integration
```python
class MarketStructureAnalyzer:
    def __init__(self, config: SRConfig):
        self.method = config.method
        self.params = config.params
    
    def detect_levels(self, data: pd.DataFrame) -> List[SRLevel]:
        if self.method == 'PIVOT':
            return self._detect_pivot_levels(data)
        elif self.method == 'VOLUME':
            return self._detect_volume_nodes(data)
        elif self.method == 'SMC':
            return self._detect_order_blocks(data)
        elif self.method == 'ML':
            return self._ml_based_levels(data)
```

#### 3.2 Volatility Analysis System
```python
class VolatilityAnalyzer:
    def __init__(self, config: VolatilityMetricsConfig):
        self.ivp_lookback = config.ivp_lookback
        self.ivr_lookback = config.ivr_lookback
        self.atr_lookback = config.atr_lookback
        
    def calculate_ivp(self, current_iv: float, historical_ivs: List[float]) -> float:
        """Calculate Implied Volatility Percentile"""
        return percentileofscore(historical_ivs, current_iv) / 100
    
    def calculate_ivr(self, current_iv: float, historical_ivs: List[float]) -> float:
        """Calculate Implied Volatility Rank"""
        min_iv = min(historical_ivs)
        max_iv = max(historical_ivs)
        return (current_iv - min_iv) / (max_iv - min_iv) if max_iv > min_iv else 0.5
    
    def calculate_atr_percentile(self, current_atr: float, historical_atrs: List[float]) -> float:
        """Calculate ATR Percentile"""
        return percentileofscore(historical_atrs, current_atr) / 100
    
    def analyze_term_structure(self, term_ivs: Dict[int, float]) -> TermStructure:
        """Analyze IV term structure for contango/backwardation"""
        return self._classify_term_structure(term_ivs)
```

#### 3.3 Breakeven Analysis System (NEW)
```python
class BreakevenAnalyzer:
    def __init__(self, config: BreakevenConfig):
        self.calculation_method = config.calculation_method
        self.include_costs = config.include_costs
        self.use_probability = config.use_probability
        
    def calculate_calendar_be(self, position: CalendarPosition) -> BreakevenPoints:
        """Calculate breakeven points for calendar spread"""
        if self.calculation_method == 'THEORETICAL':
            return self._theoretical_calendar_be(position)
        elif self.calculation_method == 'MONTE_CARLO':
            return self._monte_carlo_calendar_be(position)
        elif self.calculation_method == 'EMPIRICAL':
            return self._empirical_calendar_be(position)
            
    def calculate_iron_fly_be(self, position: IronFlyPosition) -> BreakevenPoints:
        """Calculate breakeven points for iron fly"""
        upper_be = position.short_call_strike + position.net_credit
        lower_be = position.short_put_strike - position.net_credit
        
        if self.include_costs:
            upper_be += position.total_costs
            lower_be -= position.total_costs
            
        return BreakevenPoints(
            upper=upper_be,
            lower=lower_be,
            width=upper_be - lower_be,
            max_profit=position.net_credit,
            max_loss=self._calculate_max_loss(position)
        )
    
    def calculate_be_probability(self, be_points: BreakevenPoints, 
                               market_data: MarketData) -> BEProbability:
        """Calculate probability of touching breakeven"""
        if not self.use_probability:
            return None
            
        return BEProbability(
            upper_touch=self._probability_of_touch(be_points.upper, market_data),
            lower_touch=self._probability_of_touch(be_points.lower, market_data),
            stay_within=self._probability_within_range(be_points, market_data)
        )
    
    def optimize_wings_for_be(self, center_strikes: CenterStrikes, 
                            target_be_width: float) -> WingStrikes:
        """Optimize wing selection for desired breakeven width"""
        return self._find_optimal_wings(center_strikes, target_be_width)
```

#### 3.4 Greek Management System
- Real-time greek calculation and limits
- Portfolio-level greek aggregation
- Dynamic hedging based on greek thresholds
- Vega/Gamma scalping opportunities
- **NEW**: Volatility-adjusted greek limits
- **NEW**: Term structure vega weighting
- **NEW**: Breakeven-based greek targets

#### 3.5 Performance Analytics
- Detailed adjustment tracking
- Market regime classification
- Parameter sensitivity analysis
- A/B testing framework
- **NEW**: Volatility regime performance attribution
- **NEW**: IVP/IVR effectiveness analysis
- **NEW**: Breakeven success rate tracking

### Phase 4: Testing & Optimization (Days 11-13)

#### 4.1 Unit Testing Suite
- Model parsing and validation tests
- Adjustment logic unit tests
- S/R detection algorithm tests
- Greek calculation verification
- **NEW**: IVP/IVR calculation tests
- **NEW**: ATR percentile tests

#### 4.2 Integration Testing
- End-to-end strategy execution
- Multi-leg coordination tests
- Adjustment cascade testing
- Performance benchmarks
- **NEW**: Volatility filter effectiveness tests

#### 4.3 GPU Optimization
- Query optimization for large date ranges
- Parallel adjustment processing
- Cached S/R level computation
- Batch greek calculations
- **NEW**: Optimized volatility metric queries

### Phase 5: Production Readiness (Days 14-15)

#### 5.1 Documentation
- User guide with examples
- Parameter tuning guide
- Performance optimization tips
- Troubleshooting guide
- **NEW**: Volatility metrics interpretation guide

#### 5.2 Monitoring & Logging
- Detailed execution logs
- Performance metrics
- Parameter tracking
- Alert system for anomalies
- **NEW**: Volatility regime transitions logging

## Core Components

### 1. Positional Strategy Model

```python
class PositionalStrategyModel(BaseModel):
    """Unified model for weekly and monthly positional strategies"""
    
    # Core Strategy
    strategy_name: str
    position_type: Literal["WEEKLY", "MONTHLY", "CUSTOM"]
    strategy_subtype: Literal["CALENDAR_SPREAD", "IRON_FLY", "CUSTOM"]
    
    # Timeframe Configuration
    short_leg_dte: int  # 7 for weekly, 30 for monthly
    long_leg_dte: int   # 30 for weekly calendar, 60 for monthly
    roll_frequency: Literal["WEEKLY", "BIWEEKLY", "MONTHLY", "CUSTOM"]
    
    # Entry Configuration
    entry_rules: EntryRuleSet
    market_structure_filter: MarketStructureConfig
    volatility_filter: VolatilityFilterConfig  # NEW
    
    # Adjustment Configuration
    adjustment_rules: List[AdjustmentRule]
    max_adjustments: int
    adjustment_cooldown: int  # minutes
    
    # Exit Configuration
    exit_rules: ExitRuleSet
    
    # Risk Management
    position_limits: PositionLimits
    greek_limits: GreekLimits
    volatility_limits: VolatilityLimits  # NEW
    breakeven_config: BreakevenConfig  # NEW
    
    # Performance Tracking
    track_metrics: List[str]
    parameter_optimization: bool
```

### 2. Volatility Filter Configuration

```python
class VolatilityFilterConfig(BaseModel):
    """Advanced volatility filtering configuration"""
    
    # IVP Configuration
    use_ivp: bool = True
    ivp_lookback: int = 252  # trading days
    ivp_min_entry: float = 0.0
    ivp_max_entry: float = 1.0
    ivp_min_exit: Optional[float] = None
    ivp_max_exit: Optional[float] = None
    
    # IVR Configuration
    use_ivr: bool = True
    ivr_lookback: int = 252
    ivr_min_entry: float = 0.0
    ivr_max_entry: float = 1.0
    
    # ATR Percentile Configuration
    use_atr_percentile: bool = True
    atr_period: int = 14
    atr_lookback: int = 252
    atr_percentile_min: float = 0.0
    atr_percentile_max: float = 1.0
    
    # Historical Volatility
    use_hv: bool = True
    hv_period: int = 20
    hv_lookback: int = 252
    
    # IV/HV Ratio
    use_iv_hv_ratio: bool = True
    iv_hv_ratio_min: float = 0.5
    iv_hv_ratio_max: float = 2.0
    
    # Term Structure
    analyze_term_structure: bool = True
    term_structure_type: Literal["CONTANGO", "BACKWARDATION", "FLAT", "ANY"]
    
    # Volatility Regime
    volatility_regime: Literal["LOW", "MEDIUM", "HIGH", "EXTREME", "ANY"]
    regime_lookback: int = 60
    
    # IV Smile/Skew
    analyze_iv_skew: bool = True
    skew_threshold: float = 0.05
    
    # Forward Volatility
    use_forward_vol: bool = True
    forward_vol_period: int = 30
```

### 3. Breakeven Configuration (NEW)

```python
class BreakevenConfig(BaseModel):
    """Breakeven analysis and management configuration"""
    
    # BE Calculation
    enabled: bool = True
    calculation_method: Literal["THEORETICAL", "EMPIRICAL", "MONTE_CARLO", "HYBRID"]
    include_commissions: bool = True
    include_slippage: bool = True
    time_decay_factor: bool = True
    volatility_smile_adjustment: bool = True
    
    # BE Targets
    upper_target: Union[float, Literal["DYNAMIC"]]
    lower_target: Union[float, Literal["DYNAMIC"]]
    target_be_width: Optional[float] = None  # For iron fly
    
    # BE Monitoring
    be_buffer: float = 50  # Points from BE
    be_buffer_type: Literal["FIXED", "PERCENTAGE", "ATR_BASED"]
    recalc_frequency: Literal["TICK", "MINUTE", "HOURLY", "DAILY"]
    track_be_distance: bool = True
    be_distance_alert: float = 50
    
    # BE Actions
    be_approach_action: Literal["ADJUST", "HEDGE", "CLOSE", "ALERT"]
    be_breach_action: Literal["CLOSE", "ADJUST", "REVERSE", "HOLD"]
    
    # BE Probability
    calculate_probability: bool = True
    probability_method: Literal["BLACK_SCHOLES", "MONTE_CARLO", "HISTORICAL"]
    probability_threshold: float = 0.30  # 30% probability triggers action
    
    # Strategy Specific
    calendar_be_method: Literal["T+0", "T+N", "WEIGHTED", "OPTIMAL"] = "T+0"
    iron_fly_be_wings: Literal["INCLUDE", "EXCLUDE", "WEIGHTED"] = "INCLUDE"
    
    # Optimization
    optimize_for_be: bool = True
    be_improvement_target: float = 50  # Points improvement
    maintain_be_in_adjustments: bool = True
    
    # Analysis & Reporting
    save_be_timeseries: bool = True
    generate_be_visualization: bool = True
    be_success_tracking: bool = True
```

### 4. Adjustment Rule Engine

```python
class AdjustmentRule(BaseModel):
    """Configurable adjustment rule"""
    
    rule_id: str
    rule_type: AdjustmentType
    trigger_conditions: List[TriggerCondition]
    action: AdjustmentAction
    
    # Conditions
    price_move_threshold: Optional[float]
    time_decay_threshold: Optional[float]
    greek_threshold: Optional[GreekThreshold]
    sr_proximity_required: bool
    volatility_trigger: Optional[VolatilityTrigger]  # NEW
    breakeven_trigger: Optional[BreakevenTrigger]  # NEW
    
    # Volatility-based triggers
    ivp_trigger: Optional[float]  # Trigger when IVP crosses threshold
    ivr_trigger: Optional[float]  # Trigger when IVR crosses threshold
    atr_trigger: Optional[float]  # Trigger when ATR percentile crosses
    iv_change_trigger: Optional[float]  # Trigger on IV % change
    
    # Breakeven-based triggers (NEW)
    be_trigger_type: Optional[Literal["APPROACH", "BREACH", "DISTANCE", "PROBABILITY"]]
    upper_be_trigger: Optional[Union[float, Literal["DYNAMIC"]]]
    lower_be_trigger: Optional[Union[float, Literal["DYNAMIC"]]]
    be_distance_trigger: Optional[float]  # Points from BE
    be_probability_trigger: Optional[float]  # Probability of touching BE
    
    # Actions
    target_strike_selection: StrikeSelectionMethod
    max_debit_allowed: float
    min_credit_required: float
    volatility_adjusted_sizing: bool  # NEW
    be_improvement_required: Optional[float]  # NEW: Min BE improvement
    
    # Constraints
    max_times_per_position: int
    cooldown_minutes: int
    allowed_market_hours: List[TimeWindow]
    blackout_days: List[str]  # e.g., ["wednesday", "thursday"]
    volatility_blackout: Optional[VolatilityRange]  # NEW
```

### 5. Market Structure Configuration

```python
class MarketStructureConfig(BaseModel):
    """Highly configurable market structure detection"""
    
    # Method Selection
    primary_method: Literal["PIVOT", "VOLUME", "SMC", "ML", "EXTERNAL", "COMPOSITE"]
    secondary_method: Optional[str]
    
    # Method Parameters
    pivot_config: PivotConfig
    volume_config: VolumeProfileConfig
    smc_config: SmartMoneyConfig
    ml_config: MLConfig
    
    # Volatility Integration
    volatility_weighted_levels: bool = True
    high_vol_level_expansion: float = 1.5  # Expand S/R bands in high vol
    low_vol_level_tightening: float = 0.75  # Tighten S/R bands in low vol
    
    # Usage Rules
    use_for_entry: bool
    use_for_adjustment: bool
    use_for_exit: bool
    
    # Validation
    min_distance_between_levels: float
    level_strength_threshold: float
    recent_test_weight: float
```

### 6. Query Builder System

```python
class PositionalQueryBuilder:
    """Builds optimized HeavyDB queries for positional strategies"""
    
    def build_entry_query(self, strategy: PositionalStrategyModel, 
                         market_data: MarketContext) -> str:
        """
        Generates entry query with:
        - VIX-based premium filtering
        - S/R proximity checks
        - Greek filters
        - Market regime filters
        - IVP/IVR filtering
        - ATR percentile filtering
        - Term structure analysis
        - Breakeven optimization (NEW)
        """
        
    def build_volatility_metrics_query(self, symbol: str, date: str) -> str:
        """
        Generates query for volatility metrics
        - IVP calculation
        - IVR calculation
        - ATR percentile
        - HV calculation
        - Term structure
        """
        
    def build_breakeven_query(self, position: Position) -> str:
        """
        NEW: Generates query for breakeven analysis
        - Current BE points
        - BE distances
        - BE probabilities
        - Optimal adjustments for BE
        """
        
    def build_adjustment_query(self, position: Position, 
                              rule: AdjustmentRule) -> str:
        """
        Generates adjustment query with:
        - Current position analysis
        - Adjustment target selection
        - Cost/credit optimization
        - Risk validation
        - Volatility-based adjustments
        - BE improvement validation (NEW)
        """
```

## Configuration System

### 1. Excel Sheet Structure

#### PositionalParameter Sheet (Primary Configuration)

| Column Group | Columns | Purpose |
|--------------|---------|---------|
| Strategy Identity | StrategyName, PositionType, StrategySubtype | Core identification |
| Timeframe Config | ShortLegDTE, LongLegDTE, RollFrequency, CustomDTEList | Flexible timeframe |
| VIX Configuration | 12 columns for VIX ranges and premiums | Market regime adaptation |
| **Volatility Metrics** | 30+ columns for IVP, IVR, ATR, HV | **NEW: Advanced vol filtering** |
| Entry Rules | 25+ columns for entry conditions | Comprehensive entry logic |
| Adjustment Rules | 40+ columns for adjustments | All adjustment scenarios |
| Exit Rules | 20+ columns for exits | Multiple exit strategies |
| S/R Configuration | 15+ columns for market structure | Advanced S/R integration |
| Risk Limits | 20+ columns for risk | Position and greek limits |
| Performance | 10+ columns for tracking | Metrics and optimization |

Total: 180+ configurable parameters per strategy (increased from 150+)

#### LegParameter Sheet (Enhanced)

Additional columns for positional strategies:
- PositionRole: PRIMARY, HEDGE, ADJUSTMENT, SCALP
- TimeframeOverride: Override strategy-level DTE
- ConditionalActivation: Complex activation rules
- DynamicSizing: Size based on market conditions
- **VolatilityScaling**: Scale size based on IVP/ATR (NEW)

### 2. Parameter Inheritance System

```
Global Defaults → Strategy Level → Leg Level → Runtime Overrides → Volatility Overrides (NEW)
```

This allows:
- Setting defaults for common parameters
- Strategy-specific overrides
- Leg-specific fine-tuning
- Runtime A/B testing
- **Volatility-based dynamic adjustments (NEW)**

### 3. Configuration Validation

```python
class ConfigValidator:
    def validate_positional_config(self, config: Dict) -> ValidationResult:
        """
        Validates:
        - Parameter ranges and types
        - Logical consistency
        - Risk limit coherence
        - Performance impact
        - Volatility metric consistency (NEW)
        - Lookback period sufficiency (NEW)
        """
```

## Testing Strategy

### 1. Backtesting Scenarios

#### Market Regimes
- Low volatility trending (VIX < 12, IVP < 20)
- Normal range-bound (VIX 12-20, IVP 20-50)
- High volatility (VIX > 20, IVP > 50)
- Volatility transitions
- **Term structure inversions (NEW)**
- **IV crush scenarios (NEW)**

#### Time Periods
- Weekly expiry weeks
- Monthly expiry weeks
- Holiday-shortened weeks
- High event density periods
- **Pre/post earnings volatility (NEW)**

#### Market Structures
- Strong trend with pullbacks
- Range-bound with clear S/R
- Choppy/whipsaw conditions
- V-shaped reversals
- **Volatility regime changes (NEW)**

### 2. Parameter Sensitivity Testing

```python
class ParameterSensitivityTester:
    def run_sensitivity_analysis(self, 
                               base_config: PositionalStrategyModel,
                               parameters_to_test: List[str],
                               ranges: Dict[str, List[float]]) -> SensitivityReport:
        """
        Tests parameter sensitivity:
        - Single parameter variation
        - Multi-parameter interaction
        - Regime-specific sensitivity
        - Stability analysis
        - Volatility parameter optimization (NEW)
        - Lookback period optimization (NEW)
        """
```

### 3. A/B Testing Framework

```python
class ABTestingFramework:
    def run_ab_test(self,
                   strategy_a: PositionalStrategyModel,
                   strategy_b: PositionalStrategyModel,
                   test_period: DateRange,
                   metrics: List[str]) -> ABTestReport:
        """
        Compares strategies:
        - Statistical significance
        - Risk-adjusted returns
        - Drawdown analysis
        - Regime performance
        - Volatility filter effectiveness (NEW)
        - IVP/IVR entry timing analysis (NEW)
        """
```

## Performance Optimization

### 1. Query Optimization

```sql
-- Materialized views for common calculations
CREATE MATERIALIZED VIEW positional_entry_candidates AS
WITH vix_regime AS (
    -- VIX classification logic
),
sr_levels AS (
    -- Pre-computed S/R levels
),
premium_bands AS (
    -- Premium target ranges
),
volatility_metrics AS (  -- NEW
    -- Pre-computed IVP, IVR, ATR percentile
    SELECT 
        trade_date,
        symbol,
        PERCENT_RANK() OVER (
            ORDER BY iv_close 
            ROWS BETWEEN 252 PRECEDING AND CURRENT ROW
        ) as ivp,
        (iv_close - MIN(iv_close) OVER w) / 
        (MAX(iv_close) OVER w - MIN(iv_close) OVER w) as ivr,
        PERCENT_RANK() OVER (
            ORDER BY atr_value
            ROWS BETWEEN 252 PRECEDING AND CURRENT ROW
        ) as atr_percentile
    FROM option_metrics
    WINDOW w AS (ORDER BY trade_date ROWS BETWEEN 252 PRECEDING AND CURRENT ROW)
)
SELECT /*+ gpu_enable */ 
    -- Optimized entry candidate selection with volatility filters
;

-- Refresh strategy
CREATE PROCEDURE refresh_positional_views()
AS BEGIN
    -- Intelligent refresh based on data updates
END;
```

### 2. Caching Strategy

```python
class PositionalCache:
    """Multi-level caching for positional strategies"""
    
    def __init__(self):
        self.sr_cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour
        self.vix_cache = TTLCache(maxsize=100, ttl=300)   # 5 minutes
        self.greek_cache = TTLCache(maxsize=10000, ttl=60) # 1 minute
        self.ivp_cache = TTLCache(maxsize=1000, ttl=300)  # 5 minutes (NEW)
        self.atr_cache = TTLCache(maxsize=1000, ttl=300)  # 5 minutes (NEW)
```

### 3. Parallel Processing

```python
class ParallelAdjustmentProcessor:
    """Process multiple adjustments in parallel"""
    
    def process_adjustments(self, 
                          positions: List[Position],
                          rules: List[AdjustmentRule]) -> List[Adjustment]:
        with ThreadPoolExecutor(max_workers=8) as executor:
            # Parallel adjustment evaluation
            futures = []
            for position in positions:
                for rule in rules:
                    future = executor.submit(
                        self.evaluate_adjustment, position, rule
                    )
                    futures.append(future)
```

## Timeline & Milestones

### Week 1 (Days 1-5)
- ✓ Models and parser implementation
- ✓ Database schema updates
- ✓ Basic entry logic
- ✓ Core adjustment engine
- □ Unit tests for core components
- □ Volatility metrics integration (NEW)

### Week 2 (Days 6-10)
- □ Advanced adjustment logic
- □ S/R integration
- □ Greek management
- □ Exit logic implementation
- □ Integration testing
- □ IVP/IVR/ATR implementation (NEW)

### Week 3 (Days 11-15)
- □ Performance optimization
- □ GPU query tuning
- □ Documentation
- □ Example strategies
- □ Production deployment
- □ Volatility analysis tools (NEW)

## Success Metrics

1. **Functionality**
   - All 4 adjustment types working correctly
   - S/R integration operational
   - Greek limits enforced
   - 180+ parameters configurable (increased from 150+)
   - **IVP/IVR/ATR filtering operational (NEW)**

2. **Performance**
   - 1-year backtest < 30 seconds
   - Real-time adjustment evaluation < 100ms
   - GPU utilization > 70%
   - Memory usage < 8GB
   - **Volatility metric calculation < 50ms (NEW)**

3. **Quality**
   - Unit test coverage > 90%
   - Integration tests passing
   - Documentation complete
   - Example strategies provided
   - **Volatility analysis accuracy > 99% (NEW)**

## Risk Mitigation

1. **Complexity Risk**
   - Modular design for isolated testing
   - Incremental feature rollout
   - Comprehensive logging
   - Fallback to simpler strategies

2. **Performance Risk**
   - Early benchmarking
   - Query optimization focus
   - Caching strategy
   - GPU utilization monitoring

3. **Data Quality Risk**
   - Input validation
   - Data consistency checks
   - Anomaly detection
   - Graceful degradation
   - **Historical data validation for volatility metrics (NEW)**

## Appendix: Configuration Examples

### Example 1: Conservative Weekly Calendar with IVP Filter
```yaml
strategy_name: "Conservative_Weekly_Calendar_IVP"
position_type: "WEEKLY"
strategy_subtype: "CALENDAR_SPREAD"
short_leg_dte: 7
long_leg_dte: 30
entry_rules:
  vix_range: [12, 20]
  time_window: ["11:00", "13:00"]
  min_premium: 25
  max_premium: 35
volatility_filter:  # NEW
  use_ivp: true
  ivp_min_entry: 0.30
  ivp_max_entry: 0.70
  use_atr_percentile: true
  atr_percentile_min: 0.20
  atr_percentile_max: 0.80
adjustment_rules:
  - type: "ROLL_UP"
    trigger: "PRICE_MOVE"
    threshold: 0.01
    max_debit: 0.10
    ivp_trigger: 0.80  # NEW: Adjust when IVP > 80
```

### Example 2: Aggressive Iron Fly with Volatility Regime
```yaml
strategy_name: "Aggressive_Iron_Fly_Vol_Aware"
position_type: "MONTHLY"
strategy_subtype: "IRON_FLY"
short_leg_dte: 30
volatility_filter:  # NEW
  volatility_regime: "HIGH"
  use_ivr: true
  ivr_min_entry: 0.50
  use_term_structure: true
  term_structure_type: "BACKWARDATION"
adjustment_rules:
  - type: "REVERSE_BUY"
    trigger: "SR_HIT"
    max_debit_pct: 0.25
    scalping_threshold: 0.70
    volatility_trigger:  # NEW
      ivp_drop: 0.20  # Adjust when IVP drops 20 points
greek_limits:
  max_portfolio_gamma: 0.05
  max_position_vega: 1000
  volatility_adjusted: true  # NEW: Adjust limits by IVP
```

### Example 3: Dynamic Position with Complete Volatility Suite
```yaml
strategy_name: "Dynamic_Complete_Vol_Suite"
position_type: "CUSTOM"
volatility_filter:  # NEW comprehensive config
  # IVP Settings
  use_ivp: true
  ivp_lookback: 252
  ivp_min_entry: 0.25
  ivp_max_entry: 0.75
  ivp_min_exit: 0.10
  ivp_max_exit: 0.90
  
  # IVR Settings
  use_ivr: true
  ivr_lookback: 252
  ivr_min_entry: 0.30
  ivr_max_entry: 0.70
  
  # ATR Percentile
  use_atr_percentile: true
  atr_period: 14
  atr_lookback: 252
  atr_percentile_min: 0.20
  atr_percentile_max: 0.80
  
  # HV Analysis
  use_hv: true
  hv_period: 20
  hv_lookback: 252
  
  # IV/HV Ratio
  use_iv_hv_ratio: true
  iv_hv_ratio_min: 0.8
  iv_hv_ratio_max: 1.5
  
  # Term Structure
  analyze_term_structure: true
  term_structure_type: "ANY"
  
  # IV Skew
  analyze_iv_skew: true
  skew_threshold: 0.05
```

## Next Steps

1. Review and approve implementation plan with volatility enhancements
2. Set up development environment
3. Create test data fixtures with historical volatility data
4. Begin Phase 1 implementation with volatility metrics
5. Schedule daily progress reviews 