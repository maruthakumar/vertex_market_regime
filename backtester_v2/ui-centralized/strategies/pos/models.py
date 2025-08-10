"""
POS (Positional) Strategy Data Models
Supports complex multi-leg options strategies with adjustments
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import date, time
from enum import Enum
from pydantic import BaseModel, Field, validator


class AdjustmentTrigger(str, Enum):
    """Types of triggers for strategy adjustments"""
    PRICE_BASED = "price_based"
    TIME_BASED = "time_based"
    GREEK_BASED = "greek_based"
    PNL_BASED = "pnl_based"
    VOLATILITY_BASED = "volatility_based"
    CORRELATION_BASED = "correlation_based"
    TECHNICAL_BASED = "technical_based"


class PositionType(str, Enum):
    """Position types for options"""
    BUY = "BUY"
    SELL = "SELL"


class OptionType(str, Enum):
    """Option types"""
    CALL = "CE"
    PUT = "PE"


class StrikeSelection(str, Enum):
    """Strike selection methods"""
    ATM = "ATM"
    ITM = "ITM"
    OTM = "OTM"
    STRIKE_PRICE = "STRIKE_PRICE"
    DELTA_BASED = "DELTA_BASED"
    PERCENTAGE_BASED = "PERCENTAGE_BASED"


class AdjustmentAction(str, Enum):
    """Types of adjustment actions"""
    ADD_LEG = "ADD_LEG"
    REMOVE_LEG = "REMOVE_LEG"
    ROLL_UP = "ROLL_UP"
    ROLL_DOWN = "ROLL_DOWN"
    ROLL_OUT = "ROLL_OUT"
    HEDGE = "HEDGE"
    CLOSE_POSITION = "CLOSE_POSITION"
    MODIFY_QUANTITY = "MODIFY_QUANTITY"


class MarketRegime(str, Enum):
    """Market regime classifications"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"


class StrategyType(str, Enum):
    """Strategy types for positional trading"""
    IRON_CONDOR = "IRON_CONDOR"
    IRON_BUTTERFLY = "IRON_BUTTERFLY"
    CALENDAR_SPREAD = "CALENDAR_SPREAD"
    DIAGONAL_SPREAD = "DIAGONAL_SPREAD"
    RATIO_SPREAD = "RATIO_SPREAD"
    CUSTOM = "CUSTOM"


@dataclass
class GreekLimits:
    """Greek risk limits for portfolio"""
    max_delta: Optional[float] = None
    min_delta: Optional[float] = None
    max_gamma: Optional[float] = None
    min_gamma: Optional[float] = None
    max_theta: Optional[float] = None
    min_theta: Optional[float] = None
    max_vega: Optional[float] = None
    min_vega: Optional[float] = None
    max_rho: Optional[float] = None
    min_rho: Optional[float] = None


@dataclass
class AdjustmentRule:
    """Rules for dynamic position adjustments"""
    rule_id: str
    trigger_type: AdjustmentTrigger
    trigger_condition: str  # e.g., "spot_price > entry_price * 1.02"
    trigger_value: float
    action_type: AdjustmentAction
    action_params: Dict[str, Any]
    max_adjustments: int = 3
    cooldown_period: int = 60  # minutes
    priority: int = 1  # Higher priority rules execute first


@dataclass
class VIXFilter:
    """VIX-based filtering conditions"""
    min_vix: Optional[float] = None
    max_vix: Optional[float] = None
    vix_percentile_min: Optional[float] = None
    vix_percentile_max: Optional[float] = None
    vix_term_structure: Optional[str] = None  # CONTANGO/BACKWARDATION


@dataclass
class MarketRegimeFilter:
    """Market regime filtering conditions"""
    vix_filter: Optional[VIXFilter] = None
    trend_filter: Optional[MarketRegime] = None
    volatility_regime: Optional[str] = None
    correlation_threshold: Optional[float] = None
    market_breadth_min: Optional[float] = None
    put_call_ratio_range: Optional[tuple] = None


class POSLegModel(BaseModel):
    """Individual leg configuration for positional strategy"""
    leg_id: int = Field(..., description="Unique identifier for the leg")
    leg_name: str = Field(..., description="Descriptive name for the leg")
    option_type: OptionType = Field(..., description="CALL or PUT")
    position_type: PositionType = Field(..., description="BUY or SELL")
    
    # Strike selection
    strike_selection: StrikeSelection = Field(..., description="Method for strike selection")
    strike_offset: Optional[float] = Field(None, description="Offset from ATM in points")
    strike_price: Optional[float] = Field(None, description="Specific strike price")
    delta_target: Optional[float] = Field(None, description="Target delta for selection")
    percentage_offset: Optional[float] = Field(None, description="Percentage offset from spot")
    
    # Quantity and timing
    lots: int = Field(1, description="Number of lots")
    lot_size: int = Field(50, description="Lot size for the instrument")
    entry_time: time = Field(..., description="Entry time")
    exit_time: time = Field(..., description="Exit time")
    
    # Entry/Exit conditions
    entry_conditions: List[str] = Field(default_factory=list)
    exit_conditions: List[str] = Field(default_factory=list)
    
    # Risk management
    stop_loss: Optional[float] = Field(None, description="Stop loss percentage")
    take_profit: Optional[float] = Field(None, description="Take profit percentage")
    trailing_stop: Optional[float] = Field(None, description="Trailing stop percentage")
    
    # Adjustments
    adjustment_rules: List[AdjustmentRule] = Field(default_factory=list)
    hedge_ratio: Optional[float] = Field(None, description="Hedge ratio for the leg")
    
    # Advanced features
    entry_buffer_time: int = Field(0, description="Buffer time in minutes for entry")
    exit_buffer_time: int = Field(0, description="Buffer time in minutes for exit")
    skip_expiry_day: bool = Field(False, description="Skip trading on expiry day")
    
    @validator('strike_price')
    def validate_strike_price(cls, v, values):
        if values.get('strike_selection') == StrikeSelection.STRIKE_PRICE and v is None:
            raise ValueError("strike_price required when strike_selection is STRIKE_PRICE")
        return v
    
    @validator('delta_target')
    def validate_delta_target(cls, v, values):
        if values.get('strike_selection') == StrikeSelection.DELTA_BASED and v is None:
            raise ValueError("delta_target required when strike_selection is DELTA_BASED")
        return v


class POSPortfolioModel(BaseModel):
    """Portfolio-level configuration for positional strategies"""
    portfolio_name: str = Field(..., description="Name of the portfolio")
    strategy_name: str = Field(..., description="Name of the strategy")
    strategy_type: str = Field(..., description="Type: CALENDAR/IRON_CONDOR/IRON_FLY/CUSTOM")
    
    # Date range
    start_date: date = Field(..., description="Start date for backtesting")
    end_date: date = Field(..., description="End date for backtesting")
    
    # Instrument configuration
    index_name: str = Field("NIFTY", description="Index to trade")
    underlying_price_type: str = Field("SPOT", description="SPOT or FUTURE")
    
    # Position sizing
    position_sizing: str = Field("FIXED", description="FIXED/KELLY/VOLATILITY_BASED")
    max_positions: int = Field(1, description="Maximum concurrent positions")
    position_size_value: float = Field(100000, description="Position size in currency")
    
    # Risk management
    max_portfolio_risk: float = Field(0.02, description="Maximum portfolio risk per trade")
    max_daily_loss: Optional[float] = Field(None, description="Maximum daily loss limit")
    max_drawdown: Optional[float] = Field(None, description="Maximum drawdown limit")
    greek_limits: Optional[GreekLimits] = Field(None, description="Portfolio Greek limits")
    
    # Market filters
    market_regime_filter: Optional[MarketRegimeFilter] = Field(None)
    trade_only_on: Optional[List[str]] = Field(None, description="Days to trade (MON,TUE,etc)")
    avoid_events: bool = Field(True, description="Avoid trading during major events")
    
    # Rebalancing
    rebalance_frequency: str = Field("DAILY", description="DAILY/WEEKLY/MONTHLY/NEVER")
    rebalance_threshold: float = Field(0.05, description="Rebalance when deviation > threshold")
    
    # Costs
    transaction_costs: float = Field(0.0005, description="Transaction costs as decimal")
    slippage_model: str = Field("FIXED", description="FIXED/PERCENTAGE/MARKET_IMPACT")
    slippage_value: float = Field(0.0001, description="Slippage value")
    
    # Advanced features
    use_intraday_data: bool = Field(True, description="Use intraday data for calculations")
    calculate_greeks: bool = Field(True, description="Calculate and monitor Greeks")
    enable_adjustments: bool = Field(True, description="Enable dynamic adjustments")
    backtest_mode: str = Field("REALISTIC", description="REALISTIC/OPTIMISTIC/PESSIMISTIC")


class POSStrategyModel(BaseModel):
    """Complete strategy configuration combining portfolio and legs"""
    portfolio: POSPortfolioModel
    legs: List[POSLegModel]
    
    # Strategy-specific parameters
    entry_logic: str = Field("ALL", description="ALL legs together or SEQUENTIAL")
    exit_logic: str = Field("ALL", description="ALL legs together or INDIVIDUAL")
    
    # Breakeven management
    manage_breakeven: bool = Field(False, description="Enable breakeven management")
    breakeven_target: Optional[float] = Field(None, description="Target for breakeven")
    
    # Calendar spread specific
    roll_strategy: Optional[str] = Field(None, description="For calendar: AGGRESSIVE/CONSERVATIVE")
    
    # Iron condor/fly specific
    wing_width_rule: Optional[str] = Field(None, description="Rule for wing width selection")
    adjustment_zone: Optional[float] = Field(None, description="Zone for adjustments")
    
    @validator('legs')
    def validate_legs(cls, v):
        if len(v) < 1:
            raise ValueError("At least one leg required")
        if len(v) > 20:
            raise ValueError("Maximum 20 legs supported")
        
        # Validate unique leg IDs
        leg_ids = [leg.leg_id for leg in v]
        if len(leg_ids) != len(set(leg_ids)):
            raise ValueError("Leg IDs must be unique")
        
        return v
    
    @validator('breakeven_target')
    def validate_breakeven(cls, v, values):
        if values.get('manage_breakeven') and v is None:
            raise ValueError("breakeven_target required when manage_breakeven is True")
        return v


# Predefined strategy templates
class POSTemplates:
    """Predefined templates for common strategies"""
    
    @staticmethod
    def iron_condor(spot_price: float, expiry_date: date) -> List[POSLegModel]:
        """Create Iron Condor template"""
        return [
            POSLegModel(
                leg_id=1,
                leg_name="Short Put",
                option_type=OptionType.PUT,
                position_type=PositionType.SELL,
                strike_selection=StrikeSelection.OTM,
                strike_offset=100,
                lots=1,
                entry_time=time(9, 20),
                exit_time=time(15, 20)
            ),
            POSLegModel(
                leg_id=2,
                leg_name="Long Put", 
                option_type=OptionType.PUT,
                position_type=PositionType.BUY,
                strike_selection=StrikeSelection.OTM,
                strike_offset=200,
                lots=1,
                entry_time=time(9, 20),
                exit_time=time(15, 20)
            ),
            POSLegModel(
                leg_id=3,
                leg_name="Short Call",
                option_type=OptionType.CALL,
                position_type=PositionType.SELL,
                strike_selection=StrikeSelection.OTM,
                strike_offset=100,
                lots=1,
                entry_time=time(9, 20),
                exit_time=time(15, 20)
            ),
            POSLegModel(
                leg_id=4,
                leg_name="Long Call",
                option_type=OptionType.CALL,
                position_type=PositionType.BUY,
                strike_selection=StrikeSelection.OTM,
                strike_offset=200,
                lots=1,
                entry_time=time(9, 20),
                exit_time=time(15, 20)
            )
        ]
    
    @staticmethod
    def calendar_spread(spot_price: float) -> List[POSLegModel]:
        """Create Calendar Spread template"""
        return [
            POSLegModel(
                leg_id=1,
                leg_name="Short Near Month",
                option_type=OptionType.CALL,
                position_type=PositionType.SELL,
                strike_selection=StrikeSelection.ATM,
                strike_offset=0,
                lots=1,
                entry_time=time(9, 20),
                exit_time=time(15, 20)
            ),
            POSLegModel(
                leg_id=2,
                leg_name="Long Far Month",
                option_type=OptionType.CALL,
                position_type=PositionType.BUY,
                strike_selection=StrikeSelection.ATM,
                strike_offset=0,
                lots=1,
                entry_time=time(9, 20),
                exit_time=time(15, 20)
            )
        ]