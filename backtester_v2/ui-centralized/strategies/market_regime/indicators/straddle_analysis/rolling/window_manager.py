"""
Rolling Window Manager for Triple Straddle Analysis

Manages the standardized [3,5,10,15] minute rolling windows preserved from
the original comprehensive triple straddle implementation. Provides efficient
rolling operations across all 6 components and straddle combinations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
import logging
from collections import deque
from dataclasses import dataclass
from numba import jit
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class RollingWindow:
    """Rolling window data structure"""
    size: int
    data: deque
    timestamps: deque
    
    def __post_init__(self):
        if not hasattr(self, 'data') or self.data is None:
            self.data = deque(maxlen=self.size)
        if not hasattr(self, 'timestamps') or self.timestamps is None:
            self.timestamps = deque(maxlen=self.size)


class RollingWindowManager:
    """
    Rolling Window Manager for [3,5,10,15] minute windows
    
    Preserves the original rolling window configuration from comprehensive
    triple straddle engine while optimizing for performance and memory usage.
    
    Key Features:
    - Fixed window sizes: [3, 5, 10, 15] minutes (PRESERVED from original)
    - Efficient buffer management with deque structures
    - Multi-component rolling analysis support
    - Optimized rolling calculations with Numba acceleration
    - Memory-efficient data storage
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize rolling window manager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Fixed rolling windows (PRESERVED from original implementation)
        self.window_sizes = config.get('rolling_windows', [3, 5, 10, 15])
        
        # Initialize rolling windows for each size
        self.windows = {}
        for size in self.window_sizes:
            self.windows[size] = RollingWindow(
                size=size,
                data=deque(maxlen=size),
                timestamps=deque(maxlen=size)
            )
        
        # Component tracking
        self.components = ['atm_ce', 'atm_pe', 'itm1_ce', 'itm1_pe', 'otm1_ce', 'otm1_pe']
        
        # Initialize component-specific windows
        self.component_windows = {}
        for component in self.components:
            self.component_windows[component] = {}
            for size in self.window_sizes:
                self.component_windows[component][size] = RollingWindow(
                    size=size,
                    data=deque(maxlen=size),
                    timestamps=deque(maxlen=size)
                )
        
        # Rolling calculation cache
        self._calculation_cache = {}
        
        self.logger.info(f"Rolling Window Manager initialized with windows: {self.window_sizes}")
    
    def add_data_point(self, 
                      component: str,
                      timestamp: pd.Timestamp,
                      data_point: Dict[str, float]) -> None:
        """
        Add new data point to rolling windows for a component
        
        Args:
            component: Component name (atm_ce, atm_pe, etc.)
            timestamp: Data timestamp
            data_point: Dictionary with OHLCV and derived data
        """
        if component not in self.component_windows:
            self.logger.warning(f"Unknown component: {component}")
            return
        
        # Add to all window sizes for this component
        for window_size in self.window_sizes:
            window = self.component_windows[component][window_size]
            window.data.append(data_point.copy())
            window.timestamps.append(timestamp)
        
        # Clear cache when new data is added
        self._clear_calculation_cache()
    
    def get_window_data(self, 
                       component: str,
                       window_size: int) -> Tuple[List[Dict], List[pd.Timestamp]]:
        """
        Get current window data for component and window size
        
        Args:
            component: Component name
            window_size: Window size in minutes
            
        Returns:
            Tuple of (data_list, timestamps_list)
        """
        if component not in self.component_windows:
            return [], []
        
        if window_size not in self.component_windows[component]:
            return [], []
        
        window = self.component_windows[component][window_size]
        return list(window.data), list(window.timestamps)
    
    def calculate_rolling_statistic(self,
                                  component: str,
                                  window_size: int,
                                  field: str,
                                  statistic: str = 'mean') -> Optional[float]:
        """
        Calculate rolling statistic for component field
        
        Args:
            component: Component name
            window_size: Window size in minutes
            field: Data field name (e.g., 'close', 'volume')
            statistic: Statistic to calculate ('mean', 'std', 'min', 'max', 'sum')
            
        Returns:
            Calculated statistic value or None if insufficient data
        """
        cache_key = f"{component}_{window_size}_{field}_{statistic}"
        if cache_key in self._calculation_cache:
            return self._calculation_cache[cache_key]
        
        data_list, _ = self.get_window_data(component, window_size)
        
        if len(data_list) < window_size:
            return None
        
        # Extract field values
        values = []
        for data_point in data_list:
            if field in data_point:
                values.append(data_point[field])
        
        if not values:
            return None
        
        # Calculate statistic
        try:
            values_array = np.array(values)
            
            if statistic == 'mean':
                result = np.mean(values_array)
            elif statistic == 'std':
                result = np.std(values_array)
            elif statistic == 'min':
                result = np.min(values_array)
            elif statistic == 'max':
                result = np.max(values_array)
            elif statistic == 'sum':
                result = np.sum(values_array)
            else:
                self.logger.warning(f"Unknown statistic: {statistic}")
                return None
            
            # Cache result
            self._calculation_cache[cache_key] = result
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating {statistic} for {component}.{field}: {e}")
            return None
    
    def calculate_rolling_ema(self,
                             component: str,
                             window_size: int,
                             field: str = 'close',
                             ema_period: int = 20) -> Optional[float]:
        """
        Calculate rolling EMA for component
        
        Args:
            component: Component name
            window_size: Window size in minutes
            field: Price field for EMA calculation
            ema_period: EMA period
            
        Returns:
            Current EMA value or None if insufficient data
        """
        data_list, _ = self.get_window_data(component, window_size)
        
        if len(data_list) < max(ema_period, window_size):
            return None
        
        # Extract price values
        prices = []
        for data_point in data_list:
            if field in data_point:
                prices.append(data_point[field])
        
        if len(prices) < ema_period:
            return None
        
        try:
            # Calculate EMA using pandas for consistency
            price_series = pd.Series(prices)
            ema_series = price_series.ewm(span=ema_period, adjust=False).mean()
            return ema_series.iloc[-1]
            
        except Exception as e:
            self.logger.error(f"Error calculating EMA for {component}: {e}")
            return None
    
    def calculate_rolling_vwap(self,
                              component: str,
                              window_size: int) -> Optional[float]:
        """
        Calculate rolling VWAP for component
        
        Args:
            component: Component name
            window_size: Window size in minutes
            
        Returns:
            Current VWAP value or None if insufficient data
        """
        data_list, _ = self.get_window_data(component, window_size)
        
        if len(data_list) < window_size:
            return None
        
        try:
            total_volume_price = 0.0
            total_volume = 0.0
            
            for data_point in data_list:
                if 'high' in data_point and 'low' in data_point and 'close' in data_point and 'volume' in data_point:
                    typical_price = (data_point['high'] + data_point['low'] + data_point['close']) / 3
                    volume = data_point['volume']
                    
                    total_volume_price += typical_price * volume
                    total_volume += volume
            
            if total_volume > 0:
                return total_volume_price / total_volume
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error calculating VWAP for {component}: {e}")
            return None
    
    def calculate_rolling_correlation(self,
                                    component1: str,
                                    component2: str,
                                    window_size: int,
                                    field: str = 'close') -> Optional[float]:
        """
        Calculate rolling correlation between two components
        
        Args:
            component1, component2: Component names
            window_size: Window size in minutes
            field: Field for correlation calculation
            
        Returns:
            Correlation coefficient or None if insufficient data
        """
        data1, _ = self.get_window_data(component1, window_size)
        data2, _ = self.get_window_data(component2, window_size)
        
        if len(data1) < window_size or len(data2) < window_size:
            return None
        
        try:
            # Extract field values
            values1 = [dp[field] for dp in data1 if field in dp]
            values2 = [dp[field] for dp in data2 if field in dp]
            
            if len(values1) != len(values2) or len(values1) < 2:
                return None
            
            # Calculate correlation
            correlation = np.corrcoef(values1, values2)[0, 1]
            
            # Handle NaN case
            if np.isnan(correlation):
                return 0.0
            
            return correlation
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation {component1}-{component2}: {e}")
            return None
    
    def get_all_rolling_metrics(self, 
                               component: str) -> Dict[str, Dict[str, float]]:
        """
        Get comprehensive rolling metrics for a component across all windows
        
        Args:
            component: Component name
            
        Returns:
            Dictionary with metrics for each window size
        """
        metrics = {}
        
        for window_size in self.window_sizes:
            window_metrics = {}
            
            # Basic statistics
            for field in ['close', 'volume', 'high', 'low']:
                for stat in ['mean', 'std', 'min', 'max']:
                    value = self.calculate_rolling_statistic(component, window_size, field, stat)
                    if value is not None:
                        window_metrics[f"{field}_{stat}"] = value
            
            # Technical indicators
            ema_20 = self.calculate_rolling_ema(component, window_size, 'close', 20)
            if ema_20 is not None:
                window_metrics['ema_20'] = ema_20
            
            vwap = self.calculate_rolling_vwap(component, window_size)
            if vwap is not None:
                window_metrics['vwap'] = vwap
            
            metrics[f"{window_size}min"] = window_metrics
        
        return metrics
    
    def get_cross_component_correlations(self, 
                                       window_size: int) -> Dict[str, float]:
        """
        Get correlation matrix for all component pairs at specific window size
        
        Args:
            window_size: Window size in minutes
            
        Returns:
            Dictionary with correlation values for all component pairs
        """
        correlations = {}
        
        for i, comp1 in enumerate(self.components):
            for j, comp2 in enumerate(self.components):
                if i < j:  # Avoid duplicate pairs
                    corr = self.calculate_rolling_correlation(comp1, comp2, window_size)
                    if corr is not None:
                        correlations[f"{comp1}_{comp2}"] = corr
        
        return correlations
    
    def is_window_ready(self, component: str, window_size: int) -> bool:
        """
        Check if window has sufficient data for calculations
        
        Args:
            component: Component name
            window_size: Window size in minutes
            
        Returns:
            True if window is ready for calculations
        """
        data_list, _ = self.get_window_data(component, window_size)
        return len(data_list) >= window_size
    
    def get_window_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all rolling windows
        
        Returns:
            Status dictionary for all components and windows
        """
        status = {}
        
        for component in self.components:
            component_status = {}
            for window_size in self.window_sizes:
                data_list, timestamps = self.get_window_data(component, window_size)
                component_status[f"{window_size}min"] = {
                    'data_points': len(data_list),
                    'capacity': window_size,
                    'ready': len(data_list) >= window_size,
                    'latest_timestamp': timestamps[-1] if timestamps else None
                }
            status[component] = component_status
        
        return status
    
    def _clear_calculation_cache(self):
        """Clear calculation cache when new data is added"""
        self._calculation_cache.clear()
    
    def reset_windows(self):
        """Reset all rolling windows"""
        for component in self.components:
            for window_size in self.window_sizes:
                window = self.component_windows[component][window_size]
                window.data.clear()
                window.timestamps.clear()
        
        self._clear_calculation_cache()
        self.logger.info("All rolling windows reset")