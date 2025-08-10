"""
Market Regime Models and Data Structures

This module defines all the data models and structures used in the
market regime detection system.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime, date
import pandas as pd
import numpy as np

class RegimeType(Enum):
    """Market regime classification types"""
    STRONG_BULLISH = "STRONG_BULLISH"
    MODERATE_BULLISH = "MODERATE_BULLISH"
    WEAK_BULLISH = "WEAK_BULLISH"
    NEUTRAL = "NEUTRAL"
    SIDEWAYS = "SIDEWAYS"
    WEAK_BEARISH = "WEAK_BEARISH"
    MODERATE_BEARISH = "MODERATE_BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"
    TRANSITION = "TRANSITION"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"

class IndicatorCategory(Enum):
    """Indicator category types"""
    PRICE_TREND = "PRICE_TREND"
    MOMENTUM = "MOMENTUM"
    VOLATILITY = "VOLATILITY"
    VOLUME = "VOLUME"
    OPTIONS_FLOW = "OPTIONS_FLOW"
    GREEK_SENTIMENT = "GREEK_SENTIMENT"
    IV_ANALYSIS = "IV_ANALYSIS"
    PREMIUM_ANALYSIS = "PREMIUM_ANALYSIS"
    MARKET_BREADTH = "MARKET_BREADTH"
    SENTIMENT = "SENTIMENT"

class IndicatorConfig(BaseModel):
    """Configuration for a single indicator"""
    id: str = Field(..., description="Unique identifier for the indicator")
    name: str = Field(..., description="Human-readable name")
    category: IndicatorCategory = Field(..., description="Indicator category")
    indicator_type: str = Field(..., description="Specific indicator type")
    base_weight: float = Field(0.1, ge=0.0, le=1.0, description="Base weight in regime calculation")
    min_weight: float = Field(0.01, ge=0.0, le=1.0, description="Minimum allowed weight")
    max_weight: float = Field(0.5, ge=0.0, le=1.0, description="Maximum allowed weight")
    enabled: bool = Field(True, description="Whether indicator is enabled")
    adaptive: bool = Field(True, description="Whether weight adapts based on performance")
    lookback_periods: int = Field(20, ge=1, description="Lookback periods for calculation")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Indicator-specific parameters")
    
    @validator('max_weight')
    def validate_weight_range(cls, v, values):
        if 'min_weight' in values and v < values['min_weight']:
            raise ValueError('max_weight must be >= min_weight')
        return v
    
    @validator('base_weight')
    def validate_base_weight_range(cls, v, values):
        if 'min_weight' in values and 'max_weight' in values:
            if v < values['min_weight'] or v > values['max_weight']:
                raise ValueError('base_weight must be between min_weight and max_weight')
        return v

class TimeframeConfig(BaseModel):
    """Configuration for timeframe analysis"""
    timeframe_minutes: int = Field(..., ge=1, description="Timeframe in minutes")
    weight: float = Field(1.0, ge=0.0, description="Weight for this timeframe")
    enabled: bool = Field(True, description="Whether timeframe is enabled")

class RegimeConfig(BaseModel):
    """Complete market regime configuration"""
    strategy_name: str = Field("MarketRegime", description="Strategy name")
    symbol: str = Field("NIFTY", description="Symbol to analyze")
    indicators: List[IndicatorConfig] = Field(..., description="List of indicators")
    timeframes: List[TimeframeConfig] = Field(
        default_factory=lambda: [
            TimeframeConfig(timeframe_minutes=1, weight=0.1),
            TimeframeConfig(timeframe_minutes=5, weight=0.3),
            TimeframeConfig(timeframe_minutes=15, weight=0.4),
            TimeframeConfig(timeframe_minutes=30, weight=0.2)
        ],
        description="Timeframes for analysis"
    )
    lookback_days: int = Field(252, ge=1, description="Historical lookback days")
    update_frequency: str = Field("MINUTE", description="Update frequency")
    performance_window: int = Field(100, ge=10, description="Performance tracking window")
    learning_rate: float = Field(0.01, ge=0.001, le=0.1, description="Adaptive learning rate")
    confidence_threshold: float = Field(0.6, ge=0.0, le=1.0, description="Minimum confidence threshold")
    regime_smoothing: int = Field(3, ge=1, description="Regime smoothing periods")
    enable_gpu: bool = Field(True, description="Enable GPU acceleration")
    enable_caching: bool = Field(True, description="Enable result caching")

class RegimeClassification(BaseModel):
    """Single regime classification result"""
    timestamp: datetime = Field(..., description="Classification timestamp")
    symbol: str = Field(..., description="Symbol")
    regime_type: RegimeType = Field(..., description="Classified regime type")
    regime_score: float = Field(..., ge=-2.0, le=2.0, description="Regime score (-2 to 2)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")
    component_scores: Dict[str, float] = Field(default_factory=dict, description="Individual indicator scores")
    timeframe_scores: Dict[int, float] = Field(default_factory=dict, description="Timeframe-specific scores")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class PerformanceMetrics(BaseModel):
    """Performance tracking metrics"""
    indicator_id: str = Field(..., description="Indicator identifier")
    evaluation_date: date = Field(..., description="Evaluation date")
    hit_rate: float = Field(..., ge=0.0, le=1.0, description="Prediction hit rate")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    information_ratio: float = Field(..., description="Information ratio")
    current_weight: float = Field(..., ge=0.0, le=1.0, description="Current weight")
    performance_score: float = Field(..., description="Overall performance score")
    trade_count: int = Field(0, ge=0, description="Number of trades")
    win_rate: float = Field(0.0, ge=0.0, le=1.0, description="Win rate")
    avg_return: float = Field(0.0, description="Average return")
    max_drawdown: float = Field(0.0, le=0.0, description="Maximum drawdown")

class RegimeTransition(BaseModel):
    """Regime transition event"""
    timestamp: datetime = Field(..., description="Transition timestamp")
    symbol: str = Field(..., description="Symbol")
    from_regime: RegimeType = Field(..., description="Previous regime")
    to_regime: RegimeType = Field(..., description="New regime")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Transition confidence")
    trigger_indicators: List[str] = Field(default_factory=list, description="Indicators that triggered transition")
    
class RegimeAlert(BaseModel):
    """Regime-based alert"""
    timestamp: datetime = Field(..., description="Alert timestamp")
    symbol: str = Field(..., description="Symbol")
    alert_type: str = Field(..., description="Alert type")
    regime_type: RegimeType = Field(..., description="Current regime")
    message: str = Field(..., description="Alert message")
    priority: str = Field("MEDIUM", description="Alert priority")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional alert data")

class RegimeSummary(BaseModel):
    """Summary of regime analysis"""
    symbol: str = Field(..., description="Symbol")
    analysis_date: date = Field(..., description="Analysis date")
    current_regime: RegimeType = Field(..., description="Current regime")
    regime_confidence: float = Field(..., ge=0.0, le=1.0, description="Current confidence")
    regime_duration_minutes: int = Field(0, ge=0, description="Duration in current regime")
    daily_transitions: int = Field(0, ge=0, description="Number of transitions today")
    dominant_indicators: List[str] = Field(default_factory=list, description="Most influential indicators")
    performance_summary: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    
class RegimeBacktestResult(BaseModel):
    """Backtest results for regime strategy"""
    strategy_name: str = Field(..., description="Strategy name")
    start_date: date = Field(..., description="Backtest start date")
    end_date: date = Field(..., description="Backtest end date")
    total_return: float = Field(..., description="Total return")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown")
    win_rate: float = Field(..., ge=0.0, le=1.0, description="Win rate")
    total_trades: int = Field(0, ge=0, description="Total number of trades")
    regime_accuracy: float = Field(..., ge=0.0, le=1.0, description="Regime prediction accuracy")
    regime_distribution: Dict[str, int] = Field(default_factory=dict, description="Regime distribution")
