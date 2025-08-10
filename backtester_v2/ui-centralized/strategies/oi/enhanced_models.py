"""Enhanced OI strategy models with dynamic weightage support."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime

class OiThresholdType(Enum):
    """OI threshold type enumeration."""
    ABSOLUTE = "ABSOLUTE"
    PERCENTAGE = "PERCENTAGE"
    PERCENTILE = "PERCENTILE"

class StrikeRangeType(Enum):
    """Strike range type enumeration."""
    FIXED = "FIXED"
    DYNAMIC = "DYNAMIC"
    ATM_BASED = "ATM_BASED"

class NormalizationMethod(Enum):
    """Normalization method enumeration."""
    ZSCORE = "ZSCORE"
    MINMAX = "MINMAX"
    NONE = "NONE"

class OutlierHandling(Enum):
    """Outlier handling method enumeration."""
    WINSORIZE = "WINSORIZE"
    CLIP = "CLIP"
    NONE = "NONE"

@dataclass
class EnhancedOIConfig:
    """Enhanced OI configuration with dynamic weightage support."""
    
    # Core Strategy Parameters
    strategy_name: str
    underlying: str = 'SPOT'
    index: str = 'NIFTY'
    dte: int = 0
    timeframe: int = 3
    start_time: str = '091600'
    end_time: str = '152000'
    last_entry_time: str = '150000'
    strike_selection_time: str = '091530'
    max_open_positions: int = 2
    oi_threshold: int = 800000
    strike_count: int = 10
    weekdays: str = '1,2,3,4,5'
    strategy_profit: float = 0
    strategy_loss: float = 0
    
    # OI-Specific Parameters
    oi_method: str = 'MAXOI_1'
    coi_based_on: str = 'YESTERDAY_CLOSE'
    oi_recheck_interval: int = 180
    oi_threshold_type: OiThresholdType = OiThresholdType.ABSOLUTE
    oi_concentration_threshold: float = 0.3
    oi_distribution_analysis: bool = True
    strike_range_type: StrikeRangeType = StrikeRangeType.FIXED
    strike_range_value: int = 10
    oi_momentum_period: int = 5
    oi_trend_analysis: bool = True
    oi_seasonal_adjustment: bool = False
    oi_volume_correlation: bool = True
    oi_liquidity_filter: bool = True
    oi_anomaly_detection: bool = True
    oi_signal_confirmation: bool = True
    
    # Dynamic Weightage Parameters
    enable_dynamic_weights: bool = True
    weight_adjustment_period: int = 20
    learning_rate: float = 0.01
    performance_window: int = 100
    min_weight: float = 0.05
    max_weight: float = 0.50
    weight_decay_factor: float = 0.95
    correlation_threshold: float = 0.7
    diversification_bonus: float = 1.1
    regime_adjustment: bool = True
    volatility_adjustment: bool = True
    liquidity_adjustment: bool = True
    trend_adjustment: bool = True
    momentum_adjustment: bool = True
    seasonal_adjustment: bool = False

@dataclass
class EnhancedLegConfig:
    """Enhanced leg configuration with OI and Greek parameters."""
    
    # Basic Leg Parameters
    strategy_name: str
    leg_id: str
    instrument: str
    transaction: str
    expiry: str = 'current'
    strike_method: str = 'MAXOI_1'
    strike_value: float = 0
    lots: int = 1
    sl_type: str = 'percentage'
    sl_value: float = 30
    tgt_type: str = 'percentage'
    tgt_value: float = 50
    trail_sl_type: str = 'percentage'
    sl_trail_at: float = 25
    sl_trail_by: float = 10
    
    # OI-Specific Leg Parameters
    leg_oi_threshold: int = 800000
    leg_oi_weight: float = 0.4
    leg_coi_weight: float = 0.3
    leg_oi_rank: int = 1
    leg_oi_concentration: float = 0.3
    leg_oi_momentum: float = 0.2
    leg_oi_trend: float = 0.15
    leg_oi_liquidity: float = 0.1
    leg_oi_anomaly: float = 0.05
    leg_oi_confirmation: bool = True
    
    # Greek-Based Parameters
    delta_weight: float = 0.25
    gamma_weight: float = 0.20
    theta_weight: float = 0.15
    vega_weight: float = 0.10
    delta_threshold: float = 0.5
    gamma_threshold: float = 0.1
    theta_threshold: float = -0.05
    vega_threshold: float = 0.2
    greek_rebalance_freq: int = 300
    greek_risk_limit: float = 10000

@dataclass
class DynamicWeightConfig:
    """Dynamic weight configuration for factor management."""
    
    # Base Factor Weights
    oi_factor_weight: float = 0.35
    coi_factor_weight: float = 0.25
    greek_factor_weight: float = 0.20
    market_factor_weight: float = 0.15
    performance_factor_weight: float = 0.05
    
    # OI Sub-Factor Weights
    current_oi_weight: float = 0.30
    oi_concentration_weight: float = 0.20
    oi_distribution_weight: float = 0.15
    oi_momentum_weight: float = 0.15
    oi_trend_weight: float = 0.10
    oi_seasonal_weight: float = 0.05
    oi_liquidity_weight: float = 0.03
    oi_anomaly_weight: float = 0.02
    
    # Adjustment Parameters
    weight_learning_rate: float = 0.01
    weight_decay_factor: float = 0.95
    weight_smoothing_factor: float = 0.15
    min_factor_weight: float = 0.05
    max_factor_weight: float = 0.50
    weight_rebalance_freq: int = 300
    performance_threshold: float = 0.6
    correlation_threshold: float = 0.7
    diversification_bonus: float = 1.1
    volatility_adjustment: float = 1.15
    trend_adjustment: float = 1.1
    regime_adjustment: float = 1.2

@dataclass
class FactorConfig:
    """Individual factor configuration."""
    
    factor_name: str
    factor_type: str
    base_weight: float
    min_weight: float = 0.05
    max_weight: float = 0.50
    lookback_period: int = 5
    smoothing_factor: float = 0.2
    threshold_type: str = 'ABSOLUTE'
    threshold_value: float = 0
    normalization_method: NormalizationMethod = NormalizationMethod.ZSCORE
    outlier_handling: OutlierHandling = OutlierHandling.WINSORIZE
    seasonal_adjustment: bool = False
    volatility_adjustment: bool = True
    trend_adjustment: bool = True
    regime_adjustment: bool = True
    performance_tracking: bool = True

@dataclass
class PortfolioConfig:
    """Portfolio-level configuration."""
    
    start_date: str
    end_date: str
    is_tick_bt: bool = False
    enabled: bool = True
    portfolio_name: str = 'OI_Enhanced_Portfolio'
    portfolio_target: float = 0
    portfolio_stoploss: float = 0
    portfolio_trailing_type: str = 'Lock Minimum Profit'
    pnl_cal_time: str = '151500'
    lock_percent: float = 0
    trail_percent: float = 0
    sq_off1_time: str = '000000'
    sq_off1_percent: float = 0
    sq_off2_time: str = '000000'
    sq_off2_percent: float = 0
    profit_reaches: float = 0
    lock_min_profit_at: float = 0
    increase_in_profit: float = 0
    trail_min_profit_by: float = 0
    multiplier: float = 1
    slippage_percent: float = 0.1

@dataclass
class StrategyConfig:
    """Strategy-level configuration linking to files."""
    
    enabled: bool
    portfolio_name: str
    strategy_type: str
    strategy_excel_file_path: str

@dataclass
class LegacyConfig:
    """Legacy configuration for backward compatibility."""
    
    # Legacy bt_setting.xlsx format
    enabled: str
    stgyno: int
    stgyfilepath: str
    strategy_name: str = ''
    
    # Legacy input_maxoi.xlsx format
    id: str = ''
    underlyingname: str = 'NIFTY'
    startdate: int = 241201
    enddate: int = 300101
    entrytime: int = 92200
    lastentrytime: int = 152400
    exittime: int = 152500
    dte: int = 0
    lot: int = 1
    expiry: str = 'current'
    noofstrikeeachside: int = 40
    striketotrade: str = 'maxoi1'
    slippagepercent: float = 0.1
    strategymaxprofit: float = 0
    strategymaxloss: float = 0

@dataclass
class PerformanceMetrics:
    """Performance tracking metrics."""
    
    timestamp: datetime
    strategy_name: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_profit: float = 0.0
    max_loss: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    hit_rate: float = 0.0
    avg_profit: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    # Dynamic weight specific metrics
    weight_adjustments: int = 0
    avg_weight_change: float = 0.0
    weight_stability: float = 0.0
    factor_performance: Dict[str, float] = field(default_factory=dict)
    
    def calculate_metrics(self, trades_df: pd.DataFrame) -> None:
        """Calculate performance metrics from trades DataFrame."""
        if trades_df.empty:
            return
            
        self.total_trades = len(trades_df)

        # Handle different column names for PnL
        pnl_column = None
        if 'Net PNL' in trades_df.columns:
            pnl_column = 'Net PNL'
        elif 'pnl' in trades_df.columns:
            pnl_column = 'pnl'
        elif 'PnL' in trades_df.columns:
            pnl_column = 'PnL'
        else:
            # Default to first numeric column if no standard PnL column found
            numeric_cols = trades_df.select_dtypes(include=[np.number]).columns
            pnl_column = numeric_cols[0] if len(numeric_cols) > 0 else None

        if pnl_column:
            self.winning_trades = len(trades_df[trades_df[pnl_column] > 0])
            self.losing_trades = len(trades_df[trades_df[pnl_column] < 0])
            self.total_pnl = trades_df[pnl_column].sum()
            self.max_profit = trades_df[pnl_column].max()
            self.max_loss = trades_df[pnl_column].min()
        else:
            self.winning_trades = 0
            self.losing_trades = 0
            self.total_pnl = 0.0
            self.max_profit = 0.0
            self.max_loss = 0.0
        
        if self.total_trades > 0:
            self.hit_rate = self.winning_trades / self.total_trades
            
        if pnl_column and self.winning_trades > 0:
            self.avg_profit = trades_df[trades_df[pnl_column] > 0][pnl_column].mean()

        if pnl_column and self.losing_trades > 0:
            self.avg_loss = trades_df[trades_df[pnl_column] < 0][pnl_column].mean()

        # Calculate Sharpe ratio (simplified)
        if pnl_column and len(trades_df) > 1:
            returns = trades_df[pnl_column]
            self.sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0

        # Calculate max drawdown
        if pnl_column:
            cumulative_pnl = trades_df[pnl_column].cumsum()
            running_max = cumulative_pnl.expanding().max()
            drawdown = cumulative_pnl - running_max
            self.max_drawdown = drawdown.min()

            # Calculate profit factor
            gross_profit = trades_df[trades_df[pnl_column] > 0][pnl_column].sum()
            gross_loss = abs(trades_df[trades_df[pnl_column] < 0][pnl_column].sum())
            self.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
