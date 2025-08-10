"""
Base Component Analyzer for Triple Straddle System

Provides common functionality and interface for all 6 component analyzers:
ATM_CE, ATM_PE, ITM1_CE, ITM1_PE, OTM1_CE, OTM1_PE

Each component analyzer inherits from this base class to ensure consistent
rolling analysis, technical indicators, and performance tracking.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from ..core.calculation_engine import CalculationEngine
from ..rolling.window_manager import RollingWindowManager
from ..config.excel_reader import StraddleConfig
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class ComponentAnalysisResult:
    """Result of component analysis"""
    component_name: str
    timestamp: pd.Timestamp
    
    # Rolling metrics for each window
    rolling_metrics: Dict[str, Dict[str, float]]
    
    # Technical indicators
    technical_indicators: Dict[str, float]
    
    # Performance metrics
    performance_metrics: Dict[str, float]
    
    # Regime contribution
    regime_indicators: Dict[str, float]
    
    # Price data
    current_price: float
    price_change: float
    price_change_percent: float
    
    # Volume data
    current_volume: float
    volume_ratio: float
    
    # Component-specific metrics
    component_metrics: Dict[str, Any]


class BaseComponentAnalyzer(ABC):
    """
    Base class for all component analyzers
    
    Provides standardized rolling analysis, technical indicators,
    and performance tracking for each option component.
    
    Each component analyzer must implement:
    - Component-specific price extraction
    - Component-specific metrics calculation
    - Regime contribution calculation
    """
    
    def __init__(self, 
                 component_name: str,
                 config: StraddleConfig,
                 calculation_engine: CalculationEngine,
                 window_manager: RollingWindowManager):
        """
        Initialize base component analyzer
        
        Args:
            component_name: Name of the component (e.g., 'atm_ce')
            config: Straddle configuration
            calculation_engine: Shared calculation engine
            window_manager: Rolling window manager
        """
        self.component_name = component_name
        self.config = config
        self.calculation_engine = calculation_engine
        self.window_manager = window_manager
        
        self.logger = logging.getLogger(f"{__name__}.{component_name}")
        
        # Component configuration
        self.component_weight = config.component_weights.get(component_name, 0.0)
        self.rolling_windows = config.rolling_windows
        
        # Performance tracking
        self.performance_history = []
        self.max_history_length = 1000
        
        # Component-specific settings
        self.option_type = self._determine_option_type()
        self.strike_type = self._determine_strike_type()
        
        self.logger.info(f"Component analyzer initialized: {component_name} ({self.option_type} {self.strike_type})")
    
    def _determine_option_type(self) -> str:
        """Determine if component is CE or PE"""
        return 'CE' if '_ce' in self.component_name.lower() else 'PE'
    
    def _determine_strike_type(self) -> str:
        """Determine strike type (ATM, ITM1, OTM1)"""
        if 'atm' in self.component_name.lower():
            return 'ATM'
        elif 'itm' in self.component_name.lower():
            return 'ITM1'
        elif 'otm' in self.component_name.lower():
            return 'OTM1'
        else:
            return 'UNKNOWN'
    
    @abstractmethod
    def extract_component_price(self, data: Dict[str, Any]) -> Optional[float]:
        """
        Extract component-specific price from market data
        
        Args:
            data: Market data dictionary
            
        Returns:
            Component price or None if not available
        """
        pass
    
    @abstractmethod
    def calculate_component_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate component-specific metrics
        
        Args:
            data: Market data dictionary
            
        Returns:
            Dictionary with component-specific metrics
        """
        pass
    
    @abstractmethod
    def calculate_regime_contribution(self, analysis_result: ComponentAnalysisResult) -> Dict[str, float]:
        """
        Calculate component's contribution to regime formation
        
        Args:
            analysis_result: Component analysis result
            
        Returns:
            Dictionary with regime indicators
        """
        pass
    
    def analyze(self, data: Dict[str, Any], timestamp: pd.Timestamp) -> Optional[ComponentAnalysisResult]:
        """
        Perform comprehensive component analysis
        
        Args:
            data: Market data dictionary
            timestamp: Current timestamp
            
        Returns:
            ComponentAnalysisResult or None if insufficient data
        """
        try:
            # Extract component price
            current_price = self.extract_component_price(data)
            if current_price is None:
                return None
            
            # Add data to rolling windows
            data_point = {
                'timestamp': timestamp,
                'close': current_price,
                'volume': data.get('volume', 0),
                'high': data.get('high', current_price),
                'low': data.get('low', current_price),
                'open': data.get('open', current_price)
            }
            
            self.window_manager.add_data_point(self.component_name, timestamp, data_point)
            
            # Calculate rolling metrics for all windows
            rolling_metrics = self._calculate_all_rolling_metrics()
            
            # Calculate technical indicators
            technical_indicators = self._calculate_technical_indicators()
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics()
            
            # Calculate component-specific metrics
            component_metrics = self.calculate_component_metrics(data)
            
            # Calculate price changes
            price_change, price_change_percent = self._calculate_price_changes(current_price)
            
            # Calculate volume metrics
            current_volume = data.get('volume', 0)
            volume_ratio = self._calculate_volume_ratio(current_volume)
            
            # Create analysis result
            result = ComponentAnalysisResult(
                component_name=self.component_name,
                timestamp=timestamp,
                rolling_metrics=rolling_metrics,
                technical_indicators=technical_indicators,
                performance_metrics=performance_metrics,
                regime_indicators={},  # Will be filled below
                current_price=current_price,
                price_change=price_change,
                price_change_percent=price_change_percent,
                current_volume=current_volume,
                volume_ratio=volume_ratio,
                component_metrics=component_metrics
            )
            
            # Calculate regime contribution
            result.regime_indicators = self.calculate_regime_contribution(result)
            
            # Update performance history
            self._update_performance_history(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in component analysis: {e}")
            return None
    
    def _calculate_all_rolling_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate rolling metrics for all window sizes"""
        return self.window_manager.get_all_rolling_metrics(self.component_name)
    
    def _calculate_technical_indicators(self) -> Dict[str, float]:
        """Calculate technical indicators for component"""
        indicators = {}
        
        # Calculate EMAs for different windows
        for window_size in self.rolling_windows:
            for ema_period in self.config.ema_periods:
                ema_value = self.window_manager.calculate_rolling_ema(
                    self.component_name, window_size, 'close', ema_period
                )
                if ema_value is not None:
                    indicators[f'ema_{ema_period}_{window_size}min'] = ema_value
        
        # Calculate VWAPs for different windows
        for window_size in self.rolling_windows:
            vwap_value = self.window_manager.calculate_rolling_vwap(
                self.component_name, window_size
            )
            if vwap_value is not None:
                indicators[f'vwap_{window_size}min'] = vwap_value
        
        # Get latest data for pivot calculations
        latest_data = self._get_latest_ohlcv_data()
        if latest_data is not None:
            pivot_points = self.calculation_engine.calculate_pivot_points(latest_data)
            indicators.update({f'pivot_{k}': v for k, v in pivot_points.items()})
        
        return indicators
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics for component"""
        metrics = {}
        
        # Get price series for return calculation
        for window_size in self.rolling_windows:
            data_list, _ = self.window_manager.get_window_data(self.component_name, window_size)
            
            if len(data_list) >= 2:
                prices = [dp['close'] for dp in data_list]
                returns = pd.Series(prices).pct_change().dropna()
                
                if len(returns) > 0:
                    # Calculate performance metrics using calculation engine
                    perf_metrics = self.calculation_engine.calculate_performance_metrics(returns)
                    
                    # Add window-specific prefix
                    for key, value in perf_metrics.items():
                        metrics[f'{key}_{window_size}min'] = value
        
        return metrics
    
    def _calculate_price_changes(self, current_price: float) -> Tuple[float, float]:
        """Calculate price changes from previous period"""
        try:
            # Get previous price from shortest window
            data_list, _ = self.window_manager.get_window_data(self.component_name, self.rolling_windows[0])
            
            if len(data_list) >= 2:
                previous_price = data_list[-2]['close']
                price_change = current_price - previous_price
                price_change_percent = (price_change / previous_price) * 100 if previous_price != 0 else 0.0
                return price_change, price_change_percent
            
        except Exception as e:
            self.logger.warning(f"Error calculating price changes: {e}")
        
        return 0.0, 0.0
    
    def _calculate_volume_ratio(self, current_volume: float) -> float:
        """Calculate volume ratio compared to average"""
        try:
            # Get average volume from shortest window
            avg_volume = self.window_manager.calculate_rolling_statistic(
                self.component_name, self.rolling_windows[0], 'volume', 'mean'
            )
            
            if avg_volume and avg_volume > 0:
                return current_volume / avg_volume
            
        except Exception as e:
            self.logger.warning(f"Error calculating volume ratio: {e}")
        
        return 1.0
    
    def _get_latest_ohlcv_data(self) -> Optional[pd.DataFrame]:
        """Get latest OHLCV data for calculations"""
        try:
            data_list, timestamps = self.window_manager.get_window_data(
                self.component_name, self.rolling_windows[-1]  # Use longest window
            )
            
            if not data_list:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data_list)
            df['timestamp'] = timestamps
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.warning(f"Error getting OHLCV data: {e}")
            return None
    
    def _update_performance_history(self, result: ComponentAnalysisResult):
        """Update performance history with latest result"""
        self.performance_history.append(result)
        
        # Maintain history size
        if len(self.performance_history) > self.max_history_length:
            self.performance_history.pop(0)
    
    def get_component_status(self) -> Dict[str, Any]:
        """Get component status information"""
        return {
            'component_name': self.component_name,
            'option_type': self.option_type,
            'strike_type': self.strike_type,
            'component_weight': self.component_weight,
            'rolling_windows': self.rolling_windows,
            'performance_history_length': len(self.performance_history),
            'window_status': self.window_manager.get_window_status().get(self.component_name, {}),
            'latest_analysis_time': self.performance_history[-1].timestamp if self.performance_history else None
        }
    
    def get_historical_performance(self, periods: int = 100) -> List[ComponentAnalysisResult]:
        """Get historical performance data"""
        return self.performance_history[-periods:] if self.performance_history else []
    
    def is_ready_for_analysis(self) -> bool:
        """Check if component has sufficient data for analysis"""
        # Check if at least one window is ready
        return any(
            self.window_manager.is_window_ready(self.component_name, window_size)
            for window_size in self.rolling_windows
        )