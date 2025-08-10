"""
Excel Configuration Reader for Triple Straddle Analysis

Reads configuration from the master Excel file StraddleAnalysisConfig sheet
and other related configuration sheets to eliminate hardcoded parameters
across all triple straddle modules.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
import os
from pathlib import Path
import openpyxl
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class StraddleConfig:
    """Straddle analysis configuration structure"""
    
    # Component weights (6 components)
    component_weights: Dict[str, float]
    
    # Straddle weights (3 straddles)  
    straddle_weights: Dict[str, float]
    
    # Rolling window configuration
    rolling_windows: List[int]
    window_weights: Dict[int, float]
    
    # Technical analysis parameters
    ema_periods: List[int]
    ema_weights: Dict[int, float]
    vwap_periods: List[int]
    pivot_calculation_method: str
    
    # Correlation analysis
    correlation_thresholds: Dict[str, float]
    correlation_windows: List[int]
    
    # Weight optimization
    weight_optimization_enabled: bool
    vix_thresholds: Dict[str, float]
    dte_adjustments: Dict[str, float]
    performance_feedback_rate: float
    
    # Performance tracking
    performance_windows: List[int]
    benchmark_symbol: str
    rebalancing_frequency: str
    
    # Regime formation parameters
    regime_formation_method: str
    regime_threshold_volatility: float
    regime_threshold_correlation: float
    
    # Risk management
    max_position_size: float
    stop_loss_percentage: float
    take_profit_percentage: float
    
    # Advanced settings
    calculation_precision: int
    cache_enabled: bool
    parallel_processing: bool


class StraddleExcelReader:
    """
    Excel Configuration Reader for Triple Straddle Analysis
    
    Reads comprehensive configuration from Excel sheets to eliminate
    hardcoded parameters and provide flexible configuration management.
    
    Supported Sheets:
    - StraddleAnalysisConfig: Main straddle configuration
    - WeightOptimization: Dynamic weight adjustment settings  
    - TimeframeSettings: Multi-timeframe analysis configuration
    - TechnicalAnalysis: EMA/VWAP/Pivot parameters
    - PerformanceTracking: Performance monitoring settings
    - RegimeThresholds: Market regime classification parameters
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Excel configuration reader
        
        Args:
            config_path: Path to Excel configuration file
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Configuration file path
        if config_path:
            self.config_path = Path(config_path)
        else:
            # Default path to master configuration
            self.config_path = Path("/srv/samba/shared/bt/backtester_stable/BTRUN/input_sheets/market_regime/PHASE2_ENHANCED_ULTIMATE_UNIFIED_MARKET_REGIME_CONFIG_20250627_195625_20250628_104335.xlsx")
        
        # Cached configuration
        self._config_cache = None
        self._file_modified_time = None
        
        self.logger.info(f"Excel Reader initialized for: {self.config_path}")
    
    def read_configuration(self, force_reload: bool = False) -> StraddleConfig:
        """
        Read complete configuration from Excel file
        
        Args:
            force_reload: Force reload even if cached version exists
            
        Returns:
            StraddleConfig object with all parameters
        """
        # Check if reload is needed
        if not force_reload and self._config_cache is not None:
            if self.config_path.exists():
                current_modified = self.config_path.stat().st_mtime
                if current_modified == self._file_modified_time:
                    return self._config_cache
        
        try:
            self.logger.info(f"Reading configuration from: {self.config_path}")
            
            # Check if file exists
            if not self.config_path.exists():
                self.logger.warning(f"Configuration file not found: {self.config_path}")
                return self._get_default_configuration()
            
            # Read Excel file
            with pd.ExcelFile(self.config_path) as excel_file:
                available_sheets = excel_file.sheet_names
                self.logger.info(f"Available sheets: {available_sheets}")
                
                # Read configuration from different sheets
                config_data = {}
                
                # Main straddle configuration
                config_data.update(self._read_straddle_analysis_config(excel_file))
                
                # Weight optimization settings
                config_data.update(self._read_weight_optimization_config(excel_file))
                
                # Timeframe settings
                config_data.update(self._read_timeframe_settings(excel_file))
                
                # Technical analysis parameters
                config_data.update(self._read_technical_analysis_config(excel_file))
                
                # Performance tracking
                config_data.update(self._read_performance_tracking_config(excel_file))
                
                # Regime thresholds
                config_data.update(self._read_regime_thresholds_config(excel_file))
            
            # Create StraddleConfig object
            config = self._create_straddle_config(config_data)
            
            # Cache configuration
            self._config_cache = config
            if self.config_path.exists():
                self._file_modified_time = self.config_path.stat().st_mtime
            
            self.logger.info("Configuration loaded successfully")
            return config
            
        except Exception as e:
            self.logger.error(f"Error reading Excel configuration: {e}")
            self.logger.info("Falling back to default configuration")
            return self._get_default_configuration()
    
    def _read_straddle_analysis_config(self, excel_file: pd.ExcelFile) -> Dict[str, Any]:
        """Read main straddle analysis configuration"""
        config = {}
        
        try:
            # Try to read StraddleAnalysisConfig sheet
            if 'StraddleAnalysisConfig' in excel_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name='StraddleAnalysisConfig')
                
                # Component weights (6 components)
                component_weights = {
                    'atm_ce': self._safe_float(df, 'ATM_CE_Weight', 0.20),
                    'atm_pe': self._safe_float(df, 'ATM_PE_Weight', 0.20),
                    'itm1_ce': self._safe_float(df, 'ITM1_CE_Weight', 0.15),
                    'itm1_pe': self._safe_float(df, 'ITM1_PE_Weight', 0.15),
                    'otm1_ce': self._safe_float(df, 'OTM1_CE_Weight', 0.15),
                    'otm1_pe': self._safe_float(df, 'OTM1_PE_Weight', 0.15)
                }
                
                # Straddle weights (3 straddles)
                straddle_weights = {
                    'atm': self._safe_float(df, 'ATM_Straddle_Weight', 0.50),
                    'itm1': self._safe_float(df, 'ITM1_Straddle_Weight', 0.30),
                    'otm1': self._safe_float(df, 'OTM1_Straddle_Weight', 0.20)
                }
                
                config.update({
                    'component_weights': component_weights,
                    'straddle_weights': straddle_weights
                })
                
                self.logger.info("StraddleAnalysisConfig sheet loaded")
            
        except Exception as e:
            self.logger.warning(f"Error reading StraddleAnalysisConfig: {e}")
        
        return config
    
    def _read_weight_optimization_config(self, excel_file: pd.ExcelFile) -> Dict[str, Any]:
        """Read weight optimization configuration"""
        config = {}
        
        try:
            if 'WeightOptimization' in excel_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name='WeightOptimization')
                
                config.update({
                    'weight_optimization_enabled': self._safe_bool(df, 'Optimization_Enabled', True),
                    'vix_thresholds': {
                        'low': self._safe_float(df, 'VIX_Low_Threshold', 15.0),
                        'high': self._safe_float(df, 'VIX_High_Threshold', 25.0)
                    },
                    'dte_adjustments': {
                        'short_dte': self._safe_float(df, 'Short_DTE_Adjustment', 1.2),
                        'medium_dte': self._safe_float(df, 'Medium_DTE_Adjustment', 1.0),
                        'long_dte': self._safe_float(df, 'Long_DTE_Adjustment', 0.8)
                    },
                    'performance_feedback_rate': self._safe_float(df, 'Feedback_Rate', 0.1)
                })
                
                self.logger.info("WeightOptimization sheet loaded")
                
        except Exception as e:
            self.logger.warning(f"Error reading WeightOptimization: {e}")
        
        return config
    
    def _read_timeframe_settings(self, excel_file: pd.ExcelFile) -> Dict[str, Any]:
        """Read timeframe settings configuration"""
        config = {}
        
        try:
            if 'TimeframeSettings' in excel_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name='TimeframeSettings')
                
                # Rolling windows (preserved [3,5,10,15])
                rolling_windows = [
                    int(self._safe_float(df, '3min_Window', 3)),
                    int(self._safe_float(df, '5min_Window', 5)),
                    int(self._safe_float(df, '10min_Window', 10)),
                    int(self._safe_float(df, '15min_Window', 15))
                ]
                
                # Window weights
                window_weights = {
                    3: self._safe_float(df, '3min_Weight', 0.40),
                    5: self._safe_float(df, '5min_Weight', 0.30),
                    10: self._safe_float(df, '10min_Weight', 0.20),
                    15: self._safe_float(df, '15min_Weight', 0.10)
                }
                
                config.update({
                    'rolling_windows': rolling_windows,
                    'window_weights': window_weights
                })
                
                self.logger.info("TimeframeSettings sheet loaded")
                
        except Exception as e:
            self.logger.warning(f"Error reading TimeframeSettings: {e}")
        
        return config
    
    def _read_technical_analysis_config(self, excel_file: pd.ExcelFile) -> Dict[str, Any]:
        """Read technical analysis configuration"""
        config = {}
        
        try:
            if 'TechnicalAnalysis' in excel_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name='TechnicalAnalysis')
                
                # EMA periods and weights
                ema_periods = [
                    int(self._safe_float(df, 'EMA_Period_1', 20)),
                    int(self._safe_float(df, 'EMA_Period_2', 100)),
                    int(self._safe_float(df, 'EMA_Period_3', 200))
                ]
                
                ema_weights = {
                    20: self._safe_float(df, 'EMA_20_Weight', 0.5),
                    100: self._safe_float(df, 'EMA_100_Weight', 0.3),
                    200: self._safe_float(df, 'EMA_200_Weight', 0.2)
                }
                
                # VWAP periods
                vwap_periods = rolling_windows = [3, 5, 10, 15]  # Use same as rolling windows
                
                config.update({
                    'ema_periods': ema_periods,
                    'ema_weights': ema_weights,
                    'vwap_periods': vwap_periods,
                    'pivot_calculation_method': self._safe_string(df, 'Pivot_Method', 'standard')
                })
                
                self.logger.info("TechnicalAnalysis sheet loaded")
                
        except Exception as e:
            self.logger.warning(f"Error reading TechnicalAnalysis: {e}")
        
        return config
    
    def _read_performance_tracking_config(self, excel_file: pd.ExcelFile) -> Dict[str, Any]:
        """Read performance tracking configuration"""
        config = {}
        
        try:
            if 'PerformanceTracking' in excel_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name='PerformanceTracking')
                
                config.update({
                    'performance_windows': [1, 5, 10, 20],  # Days
                    'benchmark_symbol': self._safe_string(df, 'Benchmark_Symbol', 'NIFTY'),
                    'rebalancing_frequency': self._safe_string(df, 'Rebalancing_Frequency', 'daily')
                })
                
                self.logger.info("PerformanceTracking sheet loaded")
                
        except Exception as e:
            self.logger.warning(f"Error reading PerformanceTracking: {e}")
        
        return config
    
    def _read_regime_thresholds_config(self, excel_file: pd.ExcelFile) -> Dict[str, Any]:
        """Read regime thresholds configuration"""
        config = {}
        
        try:
            if 'RegimeThresholds' in excel_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name='RegimeThresholds')
                
                config.update({
                    'correlation_thresholds': {
                        'high_correlation': self._safe_float(df, 'High_Correlation_Threshold', 0.8),
                        'medium_correlation': self._safe_float(df, 'Medium_Correlation_Threshold', 0.5),
                        'low_correlation': self._safe_float(df, 'Low_Correlation_Threshold', 0.2)
                    },
                    'correlation_windows': [3, 5, 10, 15],
                    'regime_formation_method': self._safe_string(df, 'Regime_Formation_Method', 'correlation_based'),
                    'regime_threshold_volatility': self._safe_float(df, 'Volatility_Threshold', 0.02),
                    'regime_threshold_correlation': self._safe_float(df, 'Correlation_Threshold', 0.6)
                })
                
                self.logger.info("RegimeThresholds sheet loaded")
                
        except Exception as e:
            self.logger.warning(f"Error reading RegimeThresholds: {e}")
        
        return config
    
    def _create_straddle_config(self, config_data: Dict[str, Any]) -> StraddleConfig:
        """Create StraddleConfig object from parsed data"""
        
        # Apply defaults for missing values
        defaults = self._get_default_config_dict()
        
        # Merge with defaults
        for key, default_value in defaults.items():
            if key not in config_data:
                config_data[key] = default_value
        
        # Create StraddleConfig object
        return StraddleConfig(
            component_weights=config_data['component_weights'],
            straddle_weights=config_data['straddle_weights'],
            rolling_windows=config_data['rolling_windows'],
            window_weights=config_data['window_weights'],
            ema_periods=config_data['ema_periods'],
            ema_weights=config_data['ema_weights'],
            vwap_periods=config_data['vwap_periods'],
            pivot_calculation_method=config_data['pivot_calculation_method'],
            correlation_thresholds=config_data['correlation_thresholds'],
            correlation_windows=config_data['correlation_windows'],
            weight_optimization_enabled=config_data['weight_optimization_enabled'],
            vix_thresholds=config_data['vix_thresholds'],
            dte_adjustments=config_data['dte_adjustments'],
            performance_feedback_rate=config_data['performance_feedback_rate'],
            performance_windows=config_data['performance_windows'],
            benchmark_symbol=config_data['benchmark_symbol'],
            rebalancing_frequency=config_data['rebalancing_frequency'],
            regime_formation_method=config_data['regime_formation_method'],
            regime_threshold_volatility=config_data['regime_threshold_volatility'],
            regime_threshold_correlation=config_data['regime_threshold_correlation'],
            max_position_size=config_data.get('max_position_size', 1.0),
            stop_loss_percentage=config_data.get('stop_loss_percentage', 0.05),
            take_profit_percentage=config_data.get('take_profit_percentage', 0.10),
            calculation_precision=config_data.get('calculation_precision', 6),
            cache_enabled=config_data.get('cache_enabled', True),
            parallel_processing=config_data.get('parallel_processing', True)
        )
    
    def _get_default_configuration(self) -> StraddleConfig:
        """Get default configuration when Excel file is not available"""
        config_data = self._get_default_config_dict()
        return self._create_straddle_config(config_data)
    
    def _get_default_config_dict(self) -> Dict[str, Any]:
        """Get default configuration dictionary"""
        return {
            'component_weights': {
                'atm_ce': 0.20, 'atm_pe': 0.20,
                'itm1_ce': 0.15, 'itm1_pe': 0.15,
                'otm1_ce': 0.15, 'otm1_pe': 0.15
            },
            'straddle_weights': {
                'atm': 0.50, 'itm1': 0.30, 'otm1': 0.20
            },
            'rolling_windows': [3, 5, 10, 15],
            'window_weights': {3: 0.40, 5: 0.30, 10: 0.20, 15: 0.10},
            'ema_periods': [20, 100, 200],
            'ema_weights': {20: 0.5, 100: 0.3, 200: 0.2},
            'vwap_periods': [3, 5, 10, 15],
            'pivot_calculation_method': 'standard',
            'correlation_thresholds': {
                'high_correlation': 0.8,
                'medium_correlation': 0.5,
                'low_correlation': 0.2
            },
            'correlation_windows': [3, 5, 10, 15],
            'weight_optimization_enabled': True,
            'vix_thresholds': {'low': 15.0, 'high': 25.0},
            'dte_adjustments': {
                'short_dte': 1.2, 'medium_dte': 1.0, 'long_dte': 0.8
            },
            'performance_feedback_rate': 0.1,
            'performance_windows': [1, 5, 10, 20],
            'benchmark_symbol': 'NIFTY',
            'rebalancing_frequency': 'daily',
            'regime_formation_method': 'correlation_based',
            'regime_threshold_volatility': 0.02,
            'regime_threshold_correlation': 0.6
        }
    
    # Helper methods for safe data extraction
    def _safe_float(self, df: pd.DataFrame, column: str, default: float) -> float:
        """Safely extract float value from DataFrame"""
        try:
            if column in df.columns:
                value = df[column].iloc[0] if len(df) > 0 else default
                return float(value) if pd.notna(value) else default
        except:
            pass
        return default
    
    def _safe_bool(self, df: pd.DataFrame, column: str, default: bool) -> bool:
        """Safely extract boolean value from DataFrame"""
        try:
            if column in df.columns:
                value = df[column].iloc[0] if len(df) > 0 else default
                if pd.notna(value):
                    return bool(value) if isinstance(value, bool) else str(value).lower() in ['true', '1', 'yes']
        except:
            pass
        return default
    
    def _safe_string(self, df: pd.DataFrame, column: str, default: str) -> str:
        """Safely extract string value from DataFrame"""
        try:
            if column in df.columns:
                value = df[column].iloc[0] if len(df) > 0 else default
                return str(value) if pd.notna(value) else default
        except:
            pass
        return default
    
    def validate_configuration(self, config: StraddleConfig) -> Dict[str, Any]:
        """
        Validate configuration for consistency and completeness
        
        Args:
            config: StraddleConfig object to validate
            
        Returns:
            Validation result dictionary
        """
        issues = []
        warnings = []
        
        # Validate component weights sum
        component_weight_sum = sum(config.component_weights.values())
        if abs(component_weight_sum - 1.0) > 0.01:
            issues.append(f"Component weights sum to {component_weight_sum:.3f}, should be 1.0")
        
        # Validate straddle weights sum
        straddle_weight_sum = sum(config.straddle_weights.values())
        if abs(straddle_weight_sum - 1.0) > 0.01:
            issues.append(f"Straddle weights sum to {straddle_weight_sum:.3f}, should be 1.0")
        
        # Validate rolling windows
        if config.rolling_windows != [3, 5, 10, 15]:
            warnings.append("Rolling windows differ from standard [3,5,10,15] configuration")
        
        # Validate EMA periods
        if not all(period > 0 for period in config.ema_periods):
            issues.append("All EMA periods must be positive")
        
        # Validate correlation thresholds
        thresholds = config.correlation_thresholds
        if not (0 <= thresholds['low_correlation'] <= thresholds['medium_correlation'] <= thresholds['high_correlation'] <= 1):
            issues.append("Correlation thresholds must be ordered: low <= medium <= high and in range [0,1]")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'component_weight_sum': component_weight_sum,
            'straddle_weight_sum': straddle_weight_sum
        }
    
    def get_configuration_summary(self, config: StraddleConfig) -> Dict[str, Any]:
        """
        Get summary of configuration
        
        Args:
            config: StraddleConfig object
            
        Returns:
            Configuration summary
        """
        return {
            'source_file': str(self.config_path),
            'file_exists': self.config_path.exists(),
            'component_weights': config.component_weights,
            'straddle_weights': config.straddle_weights,
            'rolling_windows': config.rolling_windows,
            'ema_periods': config.ema_periods,
            'optimization_enabled': config.weight_optimization_enabled,
            'regime_method': config.regime_formation_method,
            'cache_enabled': config.cache_enabled,
            'parallel_processing': config.parallel_processing
        }