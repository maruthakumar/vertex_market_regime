"""
Common calculation utilities for market regime system

Provides reusable calculation functions to avoid code duplication.

Author: Market Regime System Optimizer
Date: 2025-07-07
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Union, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_iv_skew(option_chain: pd.DataFrame, 
                     call_col: str = 'option_type',
                     iv_col: str = 'implied_volatility') -> float:
    """
    Calculate IV skew from option chain
    
    Args:
        option_chain: DataFrame with option data
        call_col: Column name for option type
        iv_col: Column name for implied volatility
        
    Returns:
        IV skew value (put IV - call IV) / average IV
    """
    try:
        calls = option_chain[option_chain[call_col] == 'CE']
        puts = option_chain[option_chain[call_col] == 'PE']
        
        if len(calls) > 0 and len(puts) > 0:
            call_iv = calls[iv_col].mean()
            put_iv = puts[iv_col].mean()
            avg_iv = (call_iv + put_iv) / 2
            
            if avg_iv > 0:
                return (put_iv - call_iv) / avg_iv
        return 0.0
        
    except Exception as e:
        logger.warning(f"Error calculating IV skew: {e}")
        return 0.0


def calculate_call_put_ratio(option_chain: pd.DataFrame,
                           volume_col: str = 'volume',
                           type_col: str = 'option_type') -> float:
    """
    Calculate call/put volume ratio
    
    Returns:
        Call/put ratio (1.0 if no puts)
    """
    try:
        call_volume = option_chain[option_chain[type_col] == 'CE'][volume_col].sum()
        put_volume = option_chain[option_chain[type_col] == 'PE'][volume_col].sum()
        
        if put_volume > 0:
            return call_volume / put_volume
        return 1.0
        
    except Exception as e:
        logger.warning(f"Error calculating call/put ratio: {e}")
        return 1.0


def calculate_net_delta(option_chain: pd.DataFrame,
                       delta_col: str = 'delta',
                       volume_col: str = 'volume') -> float:
    """
    Calculate net delta exposure
    
    Returns:
        Net delta (volume-weighted)
    """
    try:
        if delta_col in option_chain.columns and volume_col in option_chain.columns:
            # Calculate volume-weighted delta
            option_chain['weighted_delta'] = option_chain[delta_col] * option_chain[volume_col]
            total_volume = option_chain[volume_col].sum()
            
            if total_volume > 0:
                return option_chain['weighted_delta'].sum() / total_volume
        return 0.0
        
    except Exception as e:
        logger.warning(f"Error calculating net delta: {e}")
        return 0.0


def normalize_score(value: float, min_val: float, max_val: float,
                   clip: bool = True) -> float:
    """
    Normalize value to 0-1 range
    
    Args:
        value: Value to normalize
        min_val: Minimum expected value
        max_val: Maximum expected value
        clip: Whether to clip to [0, 1]
        
    Returns:
        Normalized value
    """
    if max_val == min_val:
        return 0.5
        
    normalized = (value - min_val) / (max_val - min_val)
    
    if clip:
        return np.clip(normalized, 0.0, 1.0)
    return normalized


def calculate_weighted_average(values: List[float], 
                             weights: List[float]) -> float:
    """
    Calculate weighted average
    
    Args:
        values: List of values
        weights: List of weights (same length as values)
        
    Returns:
        Weighted average
    """
    if not values or not weights:
        return 0.0
        
    if len(values) != len(weights):
        raise ValueError("Values and weights must have same length")
        
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
        
    return sum(v * w for v, w in zip(values, weights)) / total_weight


def calculate_ema(series: Union[pd.Series, List[float]], 
                 period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average
    
    Args:
        series: Price series
        period: EMA period
        
    Returns:
        EMA series
    """
    if isinstance(series, list):
        series = pd.Series(series)
        
    return series.ewm(span=period, adjust=False).mean()


def calculate_rsi(series: Union[pd.Series, List[float]], 
                 period: int = 14) -> float:
    """
    Calculate Relative Strength Index
    
    Args:
        series: Price series
        period: RSI period (default 14)
        
    Returns:
        RSI value (0-100)
    """
    if isinstance(series, list):
        series = pd.Series(series)
        
    if len(series) < period + 1:
        return 50.0  # Neutral if insufficient data
        
    # Calculate price changes
    delta = series.diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    # Calculate average gains and losses
    avg_gains = gains.rolling(period).mean()
    avg_losses = losses.rolling(period).mean()
    
    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    # Return last value
    last_rsi = rsi.iloc[-1]
    return last_rsi if not pd.isna(last_rsi) else 50.0


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                 period: int = 14) -> float:
    """
    Calculate Average True Range
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: ATR period
        
    Returns:
        ATR value
    """
    try:
        # Calculate True Range
        high_low = high - low
        high_close = abs(high - close.shift(1))
        low_close = abs(low - close.shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = tr.rolling(period).mean()
        
        # Return last value
        last_atr = atr.iloc[-1]
        return last_atr if not pd.isna(last_atr) else 0.0
        
    except Exception as e:
        logger.warning(f"Error calculating ATR: {e}")
        return 0.0


def calculate_bollinger_bands(series: pd.Series, period: int = 20,
                            std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands
    
    Args:
        series: Price series
        period: Moving average period
        std_dev: Number of standard deviations
        
    Returns:
        Tuple of (middle_band, upper_band, lower_band)
    """
    middle_band = series.rolling(period).mean()
    std = series.rolling(period).std()
    
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    return middle_band, upper_band, lower_band


def calculate_correlation(series1: pd.Series, series2: pd.Series,
                        window: Optional[int] = None) -> float:
    """
    Calculate correlation between two series
    
    Args:
        series1: First series
        series2: Second series
        window: Rolling window size (None for full correlation)
        
    Returns:
        Correlation coefficient
    """
    try:
        if window:
            corr = series1.rolling(window).corr(series2).iloc[-1]
        else:
            corr = series1.corr(series2)
            
        return corr if not pd.isna(corr) else 0.0
        
    except Exception as e:
        logger.warning(f"Error calculating correlation: {e}")
        return 0.0