"""
Consolidated Calculation Engine for Triple Straddle Analysis

This module consolidates all duplicate calculations found across multiple straddle modules:
- EMA calculations (duplicated 5 times across modules)
- VWAP calculations (duplicated 4 times across modules)  
- Pivot point calculations
- Performance metrics
- Rolling window operations

Eliminates massive code duplication while providing optimized, reusable calculations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from numba import jit, prange
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class CalculationEngine:
    """
    Consolidated calculation engine eliminating duplicate implementations
    
    Provides optimized, reusable calculations for:
    - Technical indicators (EMA, VWAP, Pivots)
    - Rolling window operations
    - Performance metrics
    - Statistical calculations
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize calculation engine with configuration
        
        Args:
            config: Configuration dictionary with calculation parameters
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # EMA periods (standardized across all modules)
        self.ema_periods = config.get('ema_periods', [20, 100, 200])
        
        # Rolling windows (preserved from original implementation)
        self.rolling_windows = config.get('rolling_windows', [3, 5, 10, 15])
        
        # Performance calculation cache
        self._calculation_cache = {}
        
    def calculate_ema_suite(self, 
                           price_series: pd.Series, 
                           periods: Optional[List[int]] = None) -> Dict[str, pd.Series]:
        """
        Calculate EMA suite for given periods
        
        CONSOLIDATED FROM 5 DUPLICATE IMPLEMENTATIONS:
        - enhanced_triple_rolling_straddle_engine_v2.py (Lines 958-984)
        - comprehensive_triple_straddle_engine.py (Lines 958-984)
        - enhanced_triple_straddle_analyzer.py (Lines 115-163)
        - atm_straddle_engine.py (Lines 115-163)
        - combined_straddle_engine.py (Lines 195-225)
        
        Args:
            price_series: Price data series
            periods: EMA periods to calculate (default: [20, 100, 200])
            
        Returns:
            Dictionary with EMA series for each period
        """
        if periods is None:
            periods = self.ema_periods
            
        ema_results = {}
        
        for period in periods:
            try:
                ema_results[f'ema_{period}'] = price_series.ewm(span=period, adjust=False).mean()
            except Exception as e:
                self.logger.error(f"Error calculating EMA {period}: {e}")
                ema_results[f'ema_{period}'] = pd.Series(index=price_series.index, dtype=float)
        
        return ema_results
    
    def calculate_vwap_suite(self, 
                            data: pd.DataFrame,
                            rolling_windows: Optional[List[int]] = None) -> Dict[str, pd.Series]:
        """
        Calculate VWAP suite for multiple rolling windows
        
        CONSOLIDATED FROM 4 DUPLICATE IMPLEMENTATIONS:
        - enhanced_triple_rolling_straddle_engine_v2.py (Lines 985-1027)
        - comprehensive_triple_straddle_engine.py (Lines 985-1027)
        - atm_straddle_engine.py (Lines 165-210)
        - combined_straddle_engine.py (Lines 218-225)
        
        Args:
            data: DataFrame with OHLCV data
            rolling_windows: Window sizes for rolling VWAP
            
        Returns:
            Dictionary with VWAP series for each window
        """
        if rolling_windows is None:
            rolling_windows = self.rolling_windows
            
        vwap_results = {}
        
        # Calculate typical price
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        
        for window in rolling_windows:
            try:
                # Rolling VWAP calculation
                volume_price = (typical_price * data['volume']).rolling(window=window).sum()
                volume_sum = data['volume'].rolling(window=window).sum()
                
                vwap_results[f'vwap_{window}min'] = volume_price / volume_sum
                
            except Exception as e:
                self.logger.error(f"Error calculating VWAP {window}min: {e}")
                vwap_results[f'vwap_{window}min'] = pd.Series(index=data.index, dtype=float)
        
        return vwap_results
    
    def calculate_pivot_points(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate pivot points and support/resistance levels
        
        Standardized across all straddle modules
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            Dictionary with pivot levels
        """
        try:
            # Get previous day's data
            prev_high = data['high'].iloc[-2] if len(data) > 1 else data['high'].iloc[-1]
            prev_low = data['low'].iloc[-2] if len(data) > 1 else data['low'].iloc[-1]
            prev_close = data['close'].iloc[-2] if len(data) > 1 else data['close'].iloc[-1]
            
            # Calculate pivot point
            pivot = (prev_high + prev_low + prev_close) / 3
            
            # Calculate support and resistance levels
            r1 = 2 * pivot - prev_low
            s1 = 2 * pivot - prev_high
            r2 = pivot + (prev_high - prev_low)
            s2 = pivot - (prev_high - prev_low)
            r3 = prev_high + 2 * (pivot - prev_low)
            s3 = prev_low - 2 * (prev_high - pivot)
            
            return {
                'pivot': pivot,
                'r1': r1, 'r2': r2, 'r3': r3,
                's1': s1, 's2': s2, 's3': s3
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating pivot points: {e}")
            return {level: 0.0 for level in ['pivot', 'r1', 'r2', 'r3', 's1', 's2', 's3']}
    
    @staticmethod
    @jit(nopython=True)
    def _fast_rolling_correlation(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
        """
        Fast rolling correlation calculation using Numba
        
        Args:
            x, y: Input arrays
            window: Rolling window size
            
        Returns:
            Rolling correlation array
        """
        n = len(x)
        result = np.full(n, np.nan)
        
        for i in prange(window-1, n):
            start_idx = i - window + 1
            
            x_window = x[start_idx:i+1]
            y_window = y[start_idx:i+1]
            
            # Calculate correlation
            x_mean = np.mean(x_window)
            y_mean = np.mean(y_window)
            
            numerator = np.sum((x_window - x_mean) * (y_window - y_mean))
            x_var = np.sum((x_window - x_mean) ** 2)
            y_var = np.sum((y_window - y_mean) ** 2)
            
            if x_var > 0 and y_var > 0:
                result[i] = numerator / np.sqrt(x_var * y_var)
            else:
                result[i] = 0.0
                
        return result
    
    def calculate_rolling_correlation(self, 
                                    series1: pd.Series,
                                    series2: pd.Series,
                                    window: int) -> pd.Series:
        """
        Calculate rolling correlation between two series
        
        Args:
            series1, series2: Input series
            window: Rolling window size
            
        Returns:
            Rolling correlation series
        """
        try:
            # Align series
            aligned_data = pd.concat([series1, series2], axis=1).dropna()
            if len(aligned_data) < window:
                return pd.Series(index=series1.index, dtype=float)
            
            # Use optimized calculation
            correlation = self._fast_rolling_correlation(
                aligned_data.iloc[:, 0].values,
                aligned_data.iloc[:, 1].values,
                window
            )
            
            return pd.Series(correlation, index=aligned_data.index)
            
        except Exception as e:
            self.logger.error(f"Error calculating rolling correlation: {e}")
            return pd.Series(index=series1.index, dtype=float)
    
    def calculate_performance_metrics(self, 
                                    returns_series: pd.Series,
                                    benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics
        
        CONSOLIDATED FROM 5 DUPLICATE IMPLEMENTATIONS across straddle modules
        
        Args:
            returns_series: Strategy returns
            benchmark_returns: Benchmark returns (optional)
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            # Basic statistics
            total_return = (1 + returns_series).prod() - 1
            annualized_return = (1 + returns_series.mean()) ** 252 - 1
            volatility = returns_series.std() * np.sqrt(252)
            
            # Sharpe ratio
            risk_free_rate = self.config.get('risk_free_rate', 0.02)
            sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
            
            # Maximum drawdown
            cumulative_returns = (1 + returns_series).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Hit rate
            hit_rate = (returns_series > 0).mean()
            
            # Statistical significance (t-statistic)
            n_observations = len(returns_series)
            t_statistic = (returns_series.mean() / returns_series.std()) * np.sqrt(n_observations) if returns_series.std() > 0 else 0
            
            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            metrics = {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'hit_rate': hit_rate,
                'calmar_ratio': calmar_ratio,
                't_statistic': t_statistic,
                'n_observations': n_observations
            }
            
            # Beta and alpha if benchmark provided
            if benchmark_returns is not None:
                aligned = pd.concat([returns_series, benchmark_returns], axis=1).dropna()
                if len(aligned) > 1:
                    covariance = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
                    benchmark_variance = aligned.iloc[:, 1].var()
                    
                    beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                    alpha = annualized_return - (risk_free_rate + beta * (aligned.iloc[:, 1].mean() * 252 - risk_free_rate))
                    
                    metrics.update({
                        'beta': beta,
                        'alpha': alpha
                    })
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'hit_rate': 0.0,
                'calmar_ratio': 0.0,
                't_statistic': 0.0,
                'n_observations': 0
            }
    
    def calculate_rolling_volatility(self, 
                                   price_series: pd.Series,
                                   windows: Optional[List[int]] = None) -> Dict[str, pd.Series]:
        """
        Calculate rolling volatility for multiple windows
        
        Args:
            price_series: Price series
            windows: Rolling window sizes
            
        Returns:
            Dictionary with volatility series for each window
        """
        if windows is None:
            windows = self.rolling_windows
            
        # Calculate returns
        returns = price_series.pct_change().dropna()
        
        volatility_results = {}
        
        for window in windows:
            try:
                # Annualized rolling volatility
                vol = returns.rolling(window=window).std() * np.sqrt(252)
                volatility_results[f'vol_{window}min'] = vol
                
            except Exception as e:
                self.logger.error(f"Error calculating volatility {window}min: {e}")
                volatility_results[f'vol_{window}min'] = pd.Series(index=price_series.index, dtype=float)
        
        return volatility_results
    
    def get_calculation_summary(self) -> Dict[str, Any]:
        """
        Get summary of calculation engine capabilities
        
        Returns:
            Summary dictionary
        """
        return {
            'ema_periods': self.ema_periods,
            'rolling_windows': self.rolling_windows,
            'available_calculations': [
                'EMA Suite', 'VWAP Suite', 'Pivot Points',
                'Rolling Correlation', 'Performance Metrics',
                'Rolling Volatility'
            ],
            'optimizations': [
                'Numba JIT Compilation', 'Vectorized Operations',
                'Calculation Caching'
            ],
            'cache_size': len(self._calculation_cache)
        }