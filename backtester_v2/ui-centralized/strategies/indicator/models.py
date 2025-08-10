"""
Technical Indicator Strategy Data Models
Supports technical analysis with 200+ TA-Lib indicators, Smart Money Concepts, and candlestick patterns
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from datetime import date, time, datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
import pandas as pd


class IndicatorType(str, Enum):
    """Categories of technical indicators"""
    TALIB = "TALIB"
    SMC = "SMC"  # Smart Money Concepts
    CANDLESTICK = "CANDLESTICK"
    VOLUME = "VOLUME"
    CUSTOM = "CUSTOM"
    ML_SIGNAL = "ML_SIGNAL"


class TALibIndicator(str, Enum):
    """Supported TA-Lib indicators"""
    # Overlap Studies
    SMA = "SMA"
    EMA = "EMA"
    WMA = "WMA"
    DEMA = "DEMA"
    TEMA = "TEMA"
    TRIMA = "TRIMA"
    KAMA = "KAMA"
    MAMA = "MAMA"
    T3 = "T3"
    BBANDS = "BBANDS"
    SAR = "SAR"
    
    # Momentum Indicators
    RSI = "RSI"
    STOCH = "STOCH"
    STOCHF = "STOCHF"
    STOCHRSI = "STOCHRSI"
    MACD = "MACD"
    MACDEXT = "MACDEXT"
    ADX = "ADX"
    ADXR = "ADXR"
    APO = "APO"
    PPO = "PPO"
    MOM = "MOM"
    ROC = "ROC"
    CCI = "CCI"
    CMO = "CMO"
    MFI = "MFI"
    WILLR = "WILLR"
    ULTOSC = "ULTOSC"
    
    # Volume Indicators
    AD = "AD"
    ADOSC = "ADOSC"
    OBV = "OBV"
    
    # Volatility Indicators
    ATR = "ATR"
    NATR = "NATR"
    TRANGE = "TRANGE"
    
    # Price Transform
    AVGPRICE = "AVGPRICE"
    MEDPRICE = "MEDPRICE"
    TYPPRICE = "TYPPRICE"
    WCLPRICE = "WCLPRICE"
    
    # Pattern Recognition
    CDLDOJI = "CDLDOJI"
    CDLHAMMER = "CDLHAMMER"
    CDLENGULFING = "CDLENGULFING"
    CDLMORNINGSTAR = "CDLMORNINGSTAR"
    CDLEVENINGSTAR = "CDLEVENINGSTAR"


class SMCIndicator(str, Enum):
    """Smart Money Concepts indicators"""
    BOS = "BOS"  # Break of Structure
    CHOCH = "CHOCH"  # Change of Character
    MSS = "MSS"  # Market Structure Shift
    ORDER_BLOCK = "ORDER_BLOCK"
    FVG = "FVG"  # Fair Value Gap
    LIQUIDITY_POOL = "LIQUIDITY_POOL"
    LIQUIDITY_GRAB = "LIQUIDITY_GRAB"
    MITIGATION_BLOCK = "MITIGATION_BLOCK"
    BREAKER_BLOCK = "BREAKER_BLOCK"
    OTE = "OTE"  # Optimal Trade Entry
    KILL_ZONE = "KILL_ZONE"
    SILVER_BULLET = "SILVER_BULLET"
    ASIAN_RANGE = "ASIAN_RANGE"
    LONDON_RANGE = "LONDON_RANGE"
    NY_RANGE = "NY_RANGE"


class ComparisonOperator(str, Enum):
    """Comparison operators for signals"""
    GREATER = ">"
    LESS = "<"
    EQUAL = "="
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    NOT_EQUAL = "!="
    CROSSES_ABOVE = "CROSSES_ABOVE"
    CROSSES_BELOW = "CROSSES_BELOW"
    BETWEEN = "BETWEEN"
    OUTSIDE = "OUTSIDE"


class SignalLogic(str, Enum):
    """Logic for combining signals"""
    AND = "AND"
    OR = "OR"
    WEIGHTED = "WEIGHTED"
    ML_BASED = "ML_BASED"
    CUSTOM = "CUSTOM"


class MLModelType(str, Enum):
    """Supported ML model types"""
    XGBOOST = "XGBOOST"
    LIGHTGBM = "LIGHTGBM"
    CATBOOST = "CATBOOST"
    RANDOM_FOREST = "RANDOM_FOREST"
    NEURAL_NET = "NEURAL_NET"
    LSTM = "LSTM"
    ENSEMBLE = "ENSEMBLE"


class Timeframe(str, Enum):
    """Supported timeframes"""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


@dataclass
class IndicatorConfig:
    """Configuration for a single indicator"""
    indicator_name: str
    indicator_type: IndicatorType
    parameters: Dict[str, Any]
    timeframe: Timeframe
    lookback_period: int
    normalization: Optional[str] = None  # ZSCORE/MINMAX/PERCENTILE
    cache_enabled: bool = True
    
    def get_cache_key(self) -> str:
        """Generate cache key for this indicator"""
        params_str = "_".join(f"{k}={v}" for k, v in sorted(self.parameters.items()))
        return f"{self.indicator_name}_{self.timeframe}_{params_str}"


@dataclass
class SMCConfig:
    """Smart Money Concepts configuration"""
    detect_bos: bool = True
    detect_choch: bool = True
    detect_order_blocks: bool = True
    detect_fvg: bool = True
    detect_liquidity: bool = True
    detect_mitigation: bool = False
    detect_breaker: bool = False
    
    # Session settings
    use_kill_zones: bool = True
    london_session: tuple = (7, 16)  # Hours in UTC
    ny_session: tuple = (12, 21)
    asia_session: tuple = (23, 8)
    
    # Sensitivity settings
    structure_lookback: int = 20
    order_block_lookback: int = 50
    fvg_min_size: float = 0.001  # Minimum FVG size as percentage
    liquidity_threshold: float = 2.0  # Standard deviations


@dataclass
class VolumeProfileConfig:
    """Volume Profile configuration"""
    profile_type: str = "FIXED_RANGE"  # FIXED_RANGE/COMPOSITE/DELTA
    num_bins: int = 24
    value_area_percentage: float = 0.70
    show_poc: bool = True
    show_vah_val: bool = True
    delta_enabled: bool = True
    imbalance_threshold: float = 0.6


@dataclass
class MLFeatureConfig:
    """ML feature engineering configuration"""
    price_features: bool = True
    technical_features: bool = True
    microstructure_features: bool = False
    sentiment_features: bool = False
    
    # Price feature settings
    return_periods: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 50])
    volatility_windows: List[int] = field(default_factory=lambda: [10, 20, 50])
    
    # Technical feature settings
    indicator_periods: List[int] = field(default_factory=lambda: [14, 20, 50])
    
    # Feature selection
    use_feature_importance: bool = True
    min_feature_importance: float = 0.01
    max_features: int = 100


@dataclass
class MLModelConfig:
    """ML model configuration"""
    model_type: MLModelType
    features: List[str]
    target_variable: str = "future_return"
    
    # Training settings
    training_window: int = 252  # days
    validation_split: float = 0.2
    test_split: float = 0.1
    retraining_frequency: int = 20  # days
    
    # Model parameters
    model_params: Dict[str, Any] = field(default_factory=dict)
    
    # Prediction settings
    prediction_horizon: int = 5  # minutes
    confidence_threshold: float = 0.6
    use_probability: bool = True
    
    # Ensemble settings
    ensemble_method: Optional[str] = None  # VOTING/STACKING/BLENDING
    ensemble_weights: Optional[List[float]] = None


@dataclass
class SignalCondition:
    """Condition for generating signals"""
    indicator_name: str
    condition_type: ComparisonOperator
    threshold_value: Optional[float] = None
    threshold_indicator: Optional[str] = None  # For dynamic thresholds
    secondary_value: Optional[float] = None  # For BETWEEN/OUTSIDE
    weight: float = 1.0  # For weighted signals
    enabled: bool = True
    
    def evaluate(self, value: float, threshold: float = None) -> bool:
        """Evaluate if condition is met"""
        if not self.enabled:
            return True
            
        threshold = threshold or self.threshold_value
        if threshold is None:
            return False
            
        if self.condition_type == ComparisonOperator.GREATER:
            return value > threshold
        elif self.condition_type == ComparisonOperator.LESS:
            return value < threshold
        elif self.condition_type == ComparisonOperator.EQUAL:
            return abs(value - threshold) < 1e-6
        elif self.condition_type == ComparisonOperator.BETWEEN:
            return threshold <= value <= self.secondary_value
        elif self.condition_type == ComparisonOperator.OUTSIDE:
            return value < threshold or value > self.secondary_value
        else:
            return False


class MLLegModel(BaseModel):
    """Leg configuration for ML-based strategies"""
    leg_id: int = Field(..., description="Unique identifier for the leg")
    leg_name: str = Field(..., description="Descriptive name for the leg")
    option_type: str = Field(..., description="CE or PE")
    position_type: str = Field(..., description="BUY or SELL")
    
    # Strike selection based on signals
    strike_selection: str = Field("ATM", description="Strike selection method")
    strike_offset: Optional[float] = Field(None, description="Offset from ATM")
    
    # Quantity
    lots: int = Field(1, description="Number of lots")
    lot_size: int = Field(50, description="Lot size")
    
    # Signal-based entry/exit
    entry_signal_group: str = Field("default", description="Signal group for entry")
    exit_signal_group: str = Field("default", description="Signal group for exit")
    
    # Risk management
    stop_loss: Optional[float] = Field(None, description="Stop loss percentage")
    take_profit: Optional[float] = Field(None, description="Take profit percentage")
    trailing_stop: Optional[float] = Field(None, description="Trailing stop percentage")
    
    # Timing
    entry_time_start: time = Field(time(9, 20), description="Earliest entry time")
    entry_time_end: time = Field(time(14, 30), description="Latest entry time")
    exit_time: time = Field(time(15, 20), description="Exit time")
    
    # ML-specific
    min_signal_confidence: float = Field(0.6, description="Minimum ML confidence")
    use_ml_exit: bool = Field(True, description="Use ML for exit signals")


class RiskManagementConfig(BaseModel):
    """Risk management configuration"""
    position_sizing: str = Field("FIXED", description="FIXED/KELLY/VOLATILITY/ML_BASED")
    max_position_size: float = Field(100000, description="Maximum position size")
    max_portfolio_risk: float = Field(0.02, description="Maximum portfolio risk")
    
    # Stop loss settings
    stop_loss_type: str = Field("PERCENTAGE", description="PERCENTAGE/ATR/VOLATILITY")
    stop_loss_value: float = Field(0.02, description="Stop loss value")
    
    # Take profit settings
    take_profit_type: str = Field("PERCENTAGE", description="PERCENTAGE/ATR/R_MULTIPLE")
    take_profit_value: float = Field(0.03, description="Take profit value")
    
    # Advanced risk management
    use_trailing_stop: bool = Field(False)
    trailing_stop_activation: float = Field(0.01, description="Profit level to activate")
    trailing_stop_distance: float = Field(0.005, description="Trailing distance")
    
    # Portfolio limits
    max_concurrent_positions: int = Field(5)
    max_correlation: float = Field(0.7, description="Maximum correlation between positions")
    max_sector_exposure: float = Field(0.3, description="Maximum exposure to one sector")


class ExecutionConfig(BaseModel):
    """Execution configuration"""
    execution_mode: str = Field("MARKET", description="MARKET/LIMIT/STOP")
    
    # Slippage model
    slippage_model: str = Field("FIXED", description="FIXED/PERCENTAGE/VOLATILITY")
    slippage_value: float = Field(0.0001)
    
    # Order management
    use_iceberg_orders: bool = Field(False)
    iceberg_display_size: Optional[float] = Field(None)
    
    # Smart execution
    use_vwap_execution: bool = Field(False)
    vwap_participation_rate: float = Field(0.1)
    
    # Timing
    avoid_first_minutes: int = Field(5, description="Avoid trading in first N minutes")
    avoid_last_minutes: int = Field(5, description="Avoid trading in last N minutes")


class IndicatorPortfolioModel(BaseModel):
    """Portfolio configuration for Technical Indicator strategies"""
    portfolio_name: str = Field(..., description="Name of the portfolio")
    strategy_name: str = Field(..., description="Name of the strategy")
    
    # Date range
    start_date: date = Field(..., description="Start date for backtesting")
    end_date: date = Field(..., description="End date for backtesting")
    
    # Instrument configuration
    index_name: str = Field("NIFTY", description="Index to trade")
    underlying_price_type: str = Field("SPOT", description="SPOT or FUTURE")
    
    # Risk and execution
    risk_config: RiskManagementConfig = Field(default_factory=RiskManagementConfig)
    execution_config: ExecutionConfig = Field(default_factory=ExecutionConfig)
    
    # Costs
    transaction_costs: float = Field(0.0005, description="Transaction costs as decimal")
    
    # ML settings
    use_walk_forward: bool = Field(True, description="Use walk-forward optimization")
    walk_forward_window: int = Field(60, description="Walk-forward window in days")
    
    # Performance tracking
    track_feature_importance: bool = Field(True)
    track_signal_accuracy: bool = Field(True)
    save_predictions: bool = Field(False)


@dataclass
class IndicatorSignal:
    """Trading signal generated by technical indicators"""
    timestamp: pd.Timestamp
    symbol: str
    direction: str  # LONG/SHORT/EXIT
    confidence: float
    predicted_move: float = 0.0
    indicators: Dict[str, Any] = field(default_factory=dict)
    signal_source: str = "technical_indicators"


@dataclass
class IndicatorTrade:
    """Executed trade based on indicator signal"""
    signal: IndicatorSignal
    entry_price: float
    position_size: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    exit_reason: Optional[str] = None
    direction: str = field(init=False)

    def __post_init__(self):
        self.direction = self.signal.direction


# Alias for backward compatibility
IndicatorPortfolioConfig = IndicatorPortfolioModel
MLIndicatorPortfolioModel = IndicatorPortfolioModel  # Keep for backward compatibility
MLIndicatorConfig = IndicatorConfig  # Keep for backward compatibility
MLSignal = IndicatorSignal  # Keep for backward compatibility
MLTrade = IndicatorTrade  # Keep for backward compatibility


class IndicatorStrategyModel(BaseModel):
    """Complete Technical Indicator strategy configuration"""
    portfolio: IndicatorPortfolioModel
    
    # Indicators
    indicators: List[IndicatorConfig]
    smc_config: Optional[SMCConfig] = Field(None)
    volume_profile_config: Optional[VolumeProfileConfig] = Field(None)
    
    # ML configuration
    ml_config: Optional[MLModelConfig] = Field(None)
    ml_feature_config: Optional[MLFeatureConfig] = Field(None)
    
    # Signals
    entry_signals: List[SignalCondition]
    exit_signals: List[SignalCondition]
    signal_logic: SignalLogic = Field(SignalLogic.AND)
    
    # Multi-leg support
    legs: List[MLLegModel] = Field(default_factory=list)
    position_type: str = Field("LONG", description="LONG/SHORT/BOTH")
    
    # Advanced features
    use_multi_timeframe: bool = Field(False)
    timeframes: List[Timeframe] = Field(default_factory=lambda: [Timeframe.M5])
    
    # Market filters
    trade_sessions: List[str] = Field(default_factory=lambda: ["REGULAR"])
    avoid_news_events: bool = Field(True)
    min_volume_filter: Optional[float] = Field(None)
    
    @validator('indicators')
    def validate_indicators(cls, v):
        if len(v) < 1:
            raise ValueError("At least one indicator required")
        return v
    
    @validator('entry_signals')
    def validate_entry_signals(cls, v):
        if len(v) < 1:
            raise ValueError("At least one entry signal required")
        return v


# Predefined indicator sets
class IndicatorPresets:
    """Predefined indicator configurations"""
    
    @staticmethod
    def trend_following() -> List[IndicatorConfig]:
        """Trend following indicator set"""
        return [
            IndicatorConfig(
                indicator_name="EMA",
                indicator_type=IndicatorType.TALIB,
                parameters={"timeperiod": 20},
                timeframe=Timeframe.M5,
                lookback_period=20
            ),
            IndicatorConfig(
                indicator_name="EMA",
                indicator_type=IndicatorType.TALIB,
                parameters={"timeperiod": 50},
                timeframe=Timeframe.M5,
                lookback_period=50
            ),
            IndicatorConfig(
                indicator_name="ADX",
                indicator_type=IndicatorType.TALIB,
                parameters={"timeperiod": 14},
                timeframe=Timeframe.M5,
                lookback_period=14
            ),
            IndicatorConfig(
                indicator_name="MACD",
                indicator_type=IndicatorType.TALIB,
                parameters={"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
                timeframe=Timeframe.M5,
                lookback_period=26
            )
        ]
    
    @staticmethod
    def mean_reversion() -> List[IndicatorConfig]:
        """Mean reversion indicator set"""
        return [
            IndicatorConfig(
                indicator_name="RSI",
                indicator_type=IndicatorType.TALIB,
                parameters={"timeperiod": 14},
                timeframe=Timeframe.M5,
                lookback_period=14
            ),
            IndicatorConfig(
                indicator_name="BBANDS",
                indicator_type=IndicatorType.TALIB,
                parameters={"timeperiod": 20, "nbdevup": 2, "nbdevdn": 2},
                timeframe=Timeframe.M5,
                lookback_period=20
            ),
            IndicatorConfig(
                indicator_name="CCI",
                indicator_type=IndicatorType.TALIB,
                parameters={"timeperiod": 20},
                timeframe=Timeframe.M5,
                lookback_period=20
            )
        ]