"""
Enhanced POS Strategy Models with all 200+ columns support
Implements complete specification from column_mapping_ml_pos_updated.md
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Union, Any
from datetime import date, time, datetime
from enum import Enum
import json


# Enums for all column types
class PositionType(str, Enum):
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    CUSTOM = "CUSTOM"


class StrategySubtype(str, Enum):
    CALENDAR_SPREAD = "CALENDAR_SPREAD"
    IRON_FLY = "IRON_FLY"
    IRON_CONDOR = "IRON_CONDOR"
    BUTTERFLY = "BUTTERFLY"
    CUSTOM = "CUSTOM"


class RollFrequency(str, Enum):
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    BIWEEKLY = "BIWEEKLY"
    MONTHLY = "MONTHLY"
    CUSTOM = "CUSTOM"


class ExpiryType(str, Enum):
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    NEAREST = "NEAREST"
    CURRENT_WEEK = "CURRENT_WEEK"
    NEXT_WEEK = "NEXT_WEEK"
    CURRENT_MONTH = "CURRENT_MONTH"
    NEXT_MONTH = "NEXT_MONTH"


class VixMethod(str, Enum):
    SPOT = "SPOT"
    FUTURES = "FUTURES"
    TERM_STRUCTURE = "TERM_STRUCTURE"
    CUSTOM = "CUSTOM"


class PremiumType(str, Enum):
    ABSOLUTE = "ABSOLUTE"
    PERCENTAGE = "PERCENTAGE"
    ATM_RATIO = "ATM_RATIO"


class BECalculationMethod(str, Enum):
    THEORETICAL = "THEORETICAL"
    EMPIRICAL = "EMPIRICAL"
    MONTE_CARLO = "MONTE_CARLO"
    HYBRID = "HYBRID"


class BufferType(str, Enum):
    FIXED = "FIXED"
    PERCENTAGE = "PERCENTAGE"
    ATR_BASED = "ATR_BASED"


class Frequency(str, Enum):
    TICK = "TICK"
    MINUTE = "MINUTE"
    HOURLY = "HOURLY"
    DAILY = "DAILY"


class BEAction(str, Enum):
    ADJUST = "ADJUST"
    HEDGE = "HEDGE"
    CLOSE = "CLOSE"
    ALERT = "ALERT"
    REVERSE = "REVERSE"
    HOLD = "HOLD"


class InstrumentType(str, Enum):
    CALL = "CALL"
    PUT = "PUT"
    FUT = "FUT"
    STOCK = "STOCK"
    CE = "CALL"  # Alias
    PE = "PUT"   # Alias


class TransactionType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class PositionRole(str, Enum):
    PRIMARY = "PRIMARY"
    HEDGE = "HEDGE"
    ADJUSTMENT = "ADJUSTMENT"
    SCALP = "SCALP"
    PROTECTION = "PROTECTION"


class StrikeMethod(str, Enum):
    ATM = "ATM"
    ITM = "ITM"
    OTM = "OTM"
    FIXED = "FIXED"
    PREMIUM = "PREMIUM"
    DELTA = "DELTA"
    BE_OPTIMIZED = "BE_OPTIMIZED"
    STRIKE_PRICE = "STRIKE_PRICE"
    # Add numbered variants
    ITM1 = "ITM1"
    ITM2 = "ITM2"
    ITM3 = "ITM3"
    ITM4 = "ITM4"
    ITM5 = "ITM5"
    OTM1 = "OTM1"
    OTM2 = "OTM2"
    OTM3 = "OTM3"
    OTM4 = "OTM4"
    OTM5 = "OTM5"


class BEContribution(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"


class BERole(str, Enum):
    MAINTAIN = "MAINTAIN"
    IMPROVE = "IMPROVE"
    IGNORE = "IGNORE"


class StopLossType(str, Enum):
    PERCENTAGE = "PERCENTAGE"
    POINTS = "POINTS"
    PREMIUM = "PREMIUM"
    BE_BASED = "BE_BASED"
    NONE = "NONE"


class AdjustmentTrigger(str, Enum):
    TIME_BASED = "TIME_BASED"
    PRICE_BASED = "PRICE_BASED"
    GREEK_BASED = "GREEK_BASED"
    PNL_BASED = "PNL_BASED"
    BE_BASED = "BE_BASED"
    VOLATILITY_BASED = "VOLATILITY_BASED"
    REGIME_BASED = "REGIME_BASED"


class AdjustmentAction(str, Enum):
    ROLL_STRIKE = "ROLL_STRIKE"
    ROLL_EXPIRY = "ROLL_EXPIRY"
    ADD_HEDGE = "ADD_HEDGE"
    REMOVE_HEDGE = "REMOVE_HEDGE"
    CLOSE_POSITION = "CLOSE_POSITION"
    REVERSE_POSITION = "REVERSE_POSITION"
    SCALE_IN = "SCALE_IN"
    SCALE_OUT = "SCALE_OUT"
    CONVERT_STRATEGY = "CONVERT_STRATEGY"


class MarketRegime(str, Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGE_BOUND = "RANGE_BOUND"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    BREAKOUT = "BREAKOUT"
    BREAKDOWN = "BREAKDOWN"


# VIX Range Configuration
class VixRange(BaseModel):
    min: float
    max: float


class VixConfiguration(BaseModel):
    """Complete VIX configuration with all ranges"""
    method: VixMethod = VixMethod.SPOT
    low: VixRange = Field(default_factory=lambda: VixRange(min=9, max=12))
    medium: VixRange = Field(default_factory=lambda: VixRange(min=13, max=20))
    high: VixRange = Field(default_factory=lambda: VixRange(min=20, max=30))
    extreme: VixRange = Field(default_factory=lambda: VixRange(min=30, max=100))
    custom_ranges: Optional[Dict[str, VixRange]] = None


# Premium Target Configuration
class PremiumTargets(BaseModel):
    """Premium targets by VIX regime"""
    low: Optional[str] = None  # Can be range like "22-25" or single value
    medium: Optional[str] = None
    high: Optional[str] = None
    extreme: Optional[str] = None
    premium_type: PremiumType = PremiumType.ABSOLUTE
    min_acceptable: Optional[float] = None
    max_acceptable: Optional[float] = None
    differential: Optional[Union[float, str]] = None  # Can be number or percentage


# Breakeven Analysis Configuration
class BreakevenConfig(BaseModel):
    """Complete breakeven analysis configuration"""
    enabled: bool = False
    calculation_method: BECalculationMethod = BECalculationMethod.THEORETICAL
    upper_target: Union[float, str] = "DYNAMIC"  # Can be number or "DYNAMIC"
    lower_target: Union[float, str] = "DYNAMIC"
    buffer: float = 50
    buffer_type: BufferType = BufferType.FIXED
    dynamic_adjustment: bool = False
    recalc_frequency: Frequency = Frequency.HOURLY
    include_commissions: bool = True
    include_slippage: bool = True
    time_decay_factor: bool = True
    volatility_smile_be: bool = False
    spot_price_threshold: float = 0.02  # 2% from BE
    approach_action: BEAction = BEAction.ADJUST
    breach_action: BEAction = BEAction.CLOSE
    track_distance: bool = True
    distance_alert: float = 100


# Volatility Metrics Configuration
class VolatilityFilter(BaseModel):
    """Volatility-based filtering and analysis"""
    use_ivp: bool = False
    ivp_lookback: int = 252
    ivp_min_entry: float = 0.30
    ivp_max_entry: float = 0.70
    use_ivr: bool = False
    ivr_lookback: int = 252
    ivr_min_entry: float = 0.20
    ivr_max_entry: float = 0.80
    use_atr_percentile: bool = False
    atr_period: int = 14
    atr_lookback: int = 252
    atr_min_percentile: float = 0.20
    atr_max_percentile: float = 0.80


# Entry Configuration
class EntryConfig(BaseModel):
    """Entry timing and conditions"""
    days: List[str] = Field(default_factory=list)
    time_start: Optional[time] = None
    time_end: Optional[time] = None
    preferred_time: Optional[time] = None
    avoid_first_minutes: int = 0
    avoid_last_minutes: int = 0
    # Advanced entry conditions
    min_volume: Optional[int] = None
    min_oi: Optional[int] = None
    max_spread: Optional[float] = None
    require_trend_confirmation: bool = False
    trend_lookback_periods: int = 20


# Risk Management Configuration
class RiskManagement(BaseModel):
    """Complete risk management parameters"""
    max_position_size: int = 10
    max_portfolio_risk: float = 0.02  # 2%
    max_daily_loss: Optional[float] = None
    profit_target: Optional[float] = None
    stop_loss: Optional[float] = None
    be_risk_management: bool = False
    max_be_exposure: Optional[float] = None
    # Advanced risk parameters
    use_kelly_criterion: bool = False
    kelly_fraction: float = 0.25
    max_correlation_risk: float = 0.7
    sector_limits: Optional[Dict[str, float]] = None
    use_var: bool = False
    var_confidence: float = 0.95
    var_lookback: int = 252


# Enhanced Portfolio Model
class EnhancedPortfolioModel(BaseModel):
    """Enhanced portfolio configuration with all parameters"""
    portfolio_name: str
    start_date: date
    end_date: date
    index_name: str = "NIFTY"
    multiplier: int = 1
    slippage_percent: float = 0.1
    is_tick_bt: bool = False
    enabled: bool = True
    portfolio_stoploss: float = 0
    portfolio_target: float = 0
    # Additional portfolio parameters
    initial_capital: float = 1000000
    position_size_method: str = "FIXED"  # FIXED, PERCENT, VOLATILITY_BASED
    position_size_value: float = 100000
    max_open_positions: int = 5
    correlation_limit: float = 0.7
    rebalance_frequency: Optional[str] = None
    transaction_costs: float = 0.001
    # Advanced portfolio settings
    use_margin: bool = False
    margin_requirement: float = 0.2
    maintenance_margin: float = 0.15
    margin_call_action: str = "CLOSE_NEWEST"
    compound_profits: bool = False
    reinvestment_ratio: float = 0.5


# Enhanced Positional Strategy Model
class EnhancedPositionalStrategy(BaseModel):
    """Complete positional strategy with all 200+ parameters"""
    # Strategy Identity
    strategy_name: str
    position_type: PositionType
    strategy_subtype: StrategySubtype
    enabled: bool = True
    priority: int = 1
    
    # Timeframe Configuration
    short_leg_dte: int
    long_leg_dte: Optional[int] = None
    roll_frequency: Optional[RollFrequency] = None
    custom_dte_list: Optional[List[int]] = None
    min_dte_to_enter: Optional[int] = None
    max_dte_to_enter: Optional[int] = None
    preferred_expiry: ExpiryType = ExpiryType.WEEKLY
    avoid_expiry_week: bool = False
    
    # VIX Configuration
    vix_config: VixConfiguration = Field(default_factory=VixConfiguration)
    
    # Premium Targets
    premium_targets: PremiumTargets = Field(default_factory=PremiumTargets)
    
    # Breakeven Analysis
    breakeven_config: BreakevenConfig = Field(default_factory=BreakevenConfig)
    
    # Volatility Metrics
    volatility_filter: VolatilityFilter = Field(default_factory=VolatilityFilter)
    
    # Entry Configuration
    entry_config: EntryConfig = Field(default_factory=EntryConfig)
    
    # Risk Management
    risk_management: RiskManagement = Field(default_factory=RiskManagement)
    
    # Additional strategy parameters
    max_loss_per_day: Optional[float] = None
    max_loss_per_week: Optional[float] = None
    max_consecutive_losses: int = 3
    pause_after_max_losses: bool = True
    pause_duration_days: int = 2
    
    # Performance tracking
    track_metrics: bool = True
    metrics_to_track: List[str] = Field(default_factory=lambda: [
        "sharpe_ratio", "max_drawdown", "win_rate", "profit_factor"
    ])
    
    # Market conditions
    trade_only_in_market_hours: bool = True
    avoid_holidays: bool = True
    avoid_events: List[str] = Field(default_factory=list)
    
    # Execution preferences
    use_limit_orders: bool = True
    limit_order_offset: float = 0.05
    max_order_retries: int = 3
    partial_fill_handling: str = "ACCEPT"  # ACCEPT, CANCEL, WAIT
    
    class Config:
        use_enum_values = True


# Enhanced Leg Model with all parameters
class EnhancedLegModel(BaseModel):
    """Complete leg configuration with all parameters"""
    # Core Configuration
    leg_id: str
    leg_name: str
    is_active: bool = True
    leg_priority: int = 1
    
    # Position Configuration
    instrument: InstrumentType
    transaction: TransactionType
    position_role: PositionRole = PositionRole.PRIMARY
    is_weekly_leg: bool = False
    is_protective_leg: bool = False
    
    # Strike Selection Enhanced
    strike_method: StrikeMethod
    strike_value: Optional[float] = None
    strike_delta: Optional[float] = None
    strike_premium: Optional[float] = None
    optimize_for_be: bool = False
    target_be_distance: Optional[float] = None
    
    # Size Management
    lots: int = 1
    dynamic_sizing: bool = False
    size_multiplier: float = 1.0
    max_lots: Optional[int] = None
    min_lots: int = 1
    
    # Volatility-Based Sizing
    ivp_sizing: bool = False
    ivp_size_min: float = 0.5
    ivp_size_max: float = 1.5
    atr_sizing: bool = False
    atr_size_factor: float = 1.0
    vol_regime_sizing: bool = False
    vol_regime_sizes: Dict[str, float] = Field(default_factory=lambda: {
        "low": 1.0, "medium": 0.8, "high": 0.6, "extreme": 0.4
    })
    
    # Breakeven Tracking
    track_leg_be: bool = False
    leg_be_contribution: BEContribution = BEContribution.NEUTRAL
    leg_be_weight: float = 1.0
    be_adjustment_role: BERole = BERole.MAINTAIN
    min_be_improvement: Optional[float] = None
    
    # Risk Parameters
    stop_loss_type: StopLossType = StopLossType.NONE
    stop_loss_value: Optional[float] = None
    target_type: StopLossType = StopLossType.NONE
    target_value: Optional[float] = None
    trailing_stop: bool = False
    trailing_stop_distance: Optional[float] = None
    
    # Greeks limits (leg level)
    max_delta: Optional[float] = None
    max_gamma: Optional[float] = None
    max_theta: Optional[float] = None
    max_vega: Optional[float] = None
    
    # Execution parameters
    execution_priority: int = 1
    fill_or_kill: bool = False
    all_or_none: bool = False
    iceberg_order: bool = False
    iceberg_visible_lots: Optional[int] = None
    
    # Adjustment parameters
    adjustable: bool = True
    adjustment_cooldown: int = 0  # minutes
    max_adjustments: Optional[int] = None
    adjustment_cost_limit: Optional[float] = None
    
    class Config:
        use_enum_values = True


# Adjustment Rule Model
class AdjustmentRule(BaseModel):
    """Single adjustment rule with all 90+ parameters"""
    rule_id: str
    rule_name: str
    enabled: bool = True
    priority: int = 1
    
    # Trigger Configuration
    trigger_type: AdjustmentTrigger
    trigger_value: float
    trigger_comparison: str  # "GREATER", "LESS", "EQUAL", "BETWEEN"
    trigger_value2: Optional[float] = None  # For BETWEEN
    
    # Condition checks
    check_time: bool = True
    min_time_in_position: int = 0  # minutes
    max_time_in_position: Optional[int] = None
    check_pnl: bool = True
    min_pnl: Optional[float] = None
    max_pnl: Optional[float] = None
    check_underlying_move: bool = True
    underlying_move_percent: Optional[float] = None
    
    # Action Configuration
    action_type: AdjustmentAction
    action_leg_id: Optional[str] = None
    new_strike_method: Optional[StrikeMethod] = None
    new_strike_offset: Optional[float] = None
    roll_to_dte: Optional[int] = None
    
    # Risk checks before adjustment
    check_cost: bool = True
    max_adjustment_cost: Optional[float] = None
    check_be_improvement: bool = True
    min_be_improvement_required: Optional[float] = None
    check_risk_reduction: bool = True
    min_risk_reduction_percent: Optional[float] = None
    
    # Greeks-based triggers
    delta_trigger: Optional[float] = None
    gamma_trigger: Optional[float] = None
    theta_trigger: Optional[float] = None
    vega_trigger: Optional[float] = None
    delta_neutral_band: Optional[float] = None
    
    # Market condition filters
    require_market_regime: Optional[MarketRegime] = None
    avoid_market_regime: Optional[MarketRegime] = None
    vix_min: Optional[float] = None
    vix_max: Optional[float] = None
    
    # Execution parameters
    use_limit_order: bool = True
    limit_price_offset: float = 0.05
    max_retries: int = 3
    timeout_seconds: int = 60
    
    # Post-adjustment actions
    update_stops: bool = True
    update_targets: bool = True
    send_alert: bool = True
    alert_message: Optional[str] = None
    
    # Additional parameters
    backtest_win_rate: Optional[float] = None
    backtest_avg_improvement: Optional[float] = None
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    success_count: int = 0
    
    class Config:
        use_enum_values = True


# Market Structure Configuration
class MarketStructureConfig(BaseModel):
    """Market structure detection and analysis"""
    enabled: bool = False
    
    # Trend detection
    use_moving_averages: bool = True
    ma_periods: List[int] = Field(default_factory=lambda: [20, 50, 200])
    trend_strength_threshold: float = 0.7
    
    # Support/Resistance
    detect_sr_levels: bool = True
    sr_lookback_periods: int = 100
    sr_touch_threshold: float = 0.002  # 0.2%
    sr_strength_min_touches: int = 3
    
    # Volume analysis
    analyze_volume: bool = True
    volume_ma_period: int = 20
    unusual_volume_threshold: float = 2.0  # 2x average
    
    # Market breadth
    use_advance_decline: bool = False
    use_up_down_volume: bool = False
    breadth_threshold: float = 0.6
    
    # Volatility regime
    detect_volatility_regime: bool = True
    volatility_lookback: int = 30
    regime_change_threshold: float = 0.2
    
    # Pattern detection
    detect_chart_patterns: bool = False
    patterns_to_detect: List[str] = Field(default_factory=lambda: [
        "double_top", "double_bottom", "triangle", "flag"
    ])
    
    # Market microstructure
    analyze_bid_ask_spread: bool = True
    analyze_order_flow: bool = False
    detect_large_orders: bool = True
    large_order_threshold: int = 100  # lots


# Greek Limits Configuration
class GreekLimitsConfig(BaseModel):
    """Portfolio and position level Greek limits"""
    enabled: bool = False
    
    # Portfolio level limits
    portfolio_max_delta: Optional[float] = None
    portfolio_max_gamma: Optional[float] = None
    portfolio_max_theta: Optional[float] = None
    portfolio_max_vega: Optional[float] = None
    portfolio_max_rho: Optional[float] = None
    
    # Position level limits
    position_max_delta: Optional[float] = None
    position_max_gamma: Optional[float] = None
    position_max_theta: Optional[float] = None
    position_max_vega: Optional[float] = None
    
    # Delta-neutral bands
    maintain_delta_neutral: bool = False
    delta_neutral_threshold: float = 100  # absolute delta
    auto_hedge_delta: bool = False
    hedge_instrument: str = "FUTURES"
    
    # Gamma scalping
    enable_gamma_scalping: bool = False
    gamma_scalp_threshold: float = 50
    scalp_size: int = 1
    
    # Vega management
    vega_hedge_enabled: bool = False
    vega_hedge_threshold: float = 1000
    vega_hedge_method: str = "CALENDAR"  # CALENDAR, DIAGONAL, RATIO
    
    # Risk parity
    use_risk_parity: bool = False
    risk_parity_target: str = "EQUAL_RISK"  # EQUAL_RISK, EQUAL_PREMIUM
    rebalance_frequency: str = "DAILY"


# Complete POS Strategy Model
class CompletePOSStrategy(BaseModel):
    """Complete POS strategy with all components"""
    portfolio: EnhancedPortfolioModel
    strategy: EnhancedPositionalStrategy
    legs: List[EnhancedLegModel]
    adjustment_rules: Optional[List[AdjustmentRule]] = None
    market_structure: Optional[MarketStructureConfig] = None
    greek_limits: Optional[GreekLimitsConfig] = None
    
    # Validation
    @validator('legs')
    def validate_legs(cls, v, values):
        if len(v) == 0:
            raise ValueError("At least one leg is required")
        return v
    
    @validator('adjustment_rules')
    def validate_adjustment_rules(cls, v):
        if v:
            # Check for unique rule IDs
            rule_ids = [rule.rule_id for rule in v]
            if len(rule_ids) != len(set(rule_ids)):
                raise ValueError("Adjustment rule IDs must be unique")
        return v
    
    def get_leg_by_id(self, leg_id: str) -> Optional[EnhancedLegModel]:
        """Get leg by ID"""
        for leg in self.legs:
            if leg.leg_id == leg_id:
                return leg
        return None
    
    def get_active_legs(self) -> List[EnhancedLegModel]:
        """Get only active legs"""
        return [leg for leg in self.legs if leg.is_active]
    
    def get_adjustment_rules_by_trigger(self, trigger_type: AdjustmentTrigger) -> List[AdjustmentRule]:
        """Get adjustment rules by trigger type"""
        if not self.adjustment_rules:
            return []
        return [rule for rule in self.adjustment_rules if rule.trigger_type == trigger_type]
    
    class Config:
        use_enum_values = True