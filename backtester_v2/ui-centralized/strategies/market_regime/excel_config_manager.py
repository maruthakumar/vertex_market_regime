"""
Market Regime Excel Configuration Manager

This module provides Excel-based configuration management for the market regime
detection system, following the same patterns as other backtester_v2 systems.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
from enum import Enum

# Import Enhanced18RegimeType with fallback handling
try:
    from .archive_enhanced_modules_do_not_use.enhanced_regime_detector import Enhanced18RegimeType
except ImportError:
    try:
        from archive_enhanced_modules_do_not_use.enhanced_regime_detector import Enhanced18RegimeType
    except ImportError:
        # Fallback enum for standalone usage
        from enum import Enum
        class Enhanced18RegimeType(Enum):
            """18 Enhanced Market Regime Types"""
            HIGH_VOLATILE_STRONG_BULLISH = "High_Volatile_Strong_Bullish"
            NORMAL_VOLATILE_STRONG_BULLISH = "Normal_Volatile_Strong_Bullish"
            LOW_VOLATILE_STRONG_BULLISH = "Low_Volatile_Strong_Bullish"
            HIGH_VOLATILE_MILD_BULLISH = "High_Volatile_Mild_Bullish"
            NORMAL_VOLATILE_MILD_BULLISH = "Normal_Volatile_Mild_Bullish"
            LOW_VOLATILE_MILD_BULLISH = "Low_Volatile_Mild_Bullish"
            HIGH_VOLATILE_NEUTRAL = "High_Volatile_Neutral"
            NORMAL_VOLATILE_NEUTRAL = "Normal_Volatile_Neutral"
            LOW_VOLATILE_NEUTRAL = "Low_Volatile_Neutral"
            HIGH_VOLATILE_SIDEWAYS = "High_Volatile_Sideways"
            NORMAL_VOLATILE_SIDEWAYS = "Normal_Volatile_Sideways"
            LOW_VOLATILE_SIDEWAYS = "Low_Volatile_Sideways"
            HIGH_VOLATILE_MILD_BEARISH = "High_Volatile_Mild_Bearish"
            NORMAL_VOLATILE_MILD_BEARISH = "Normal_Volatile_Mild_Bearish"
            LOW_VOLATILE_MILD_BEARISH = "Low_Volatile_Mild_Bearish"
            HIGH_VOLATILE_STRONG_BEARISH = "High_Volatile_Strong_Bearish"
            NORMAL_VOLATILE_STRONG_BEARISH = "Normal_Volatile_Strong_Bearish"
            LOW_VOLATILE_STRONG_BEARISH = "Low_Volatile_Strong_Bearish"

logger = logging.getLogger(__name__)

class RegimeConfigType(Enum):
    """Types of regime configuration"""
    DETECTION_PARAMETERS = "detection_parameters"
    REGIME_ADJUSTMENTS = "regime_adjustments"
    STRATEGY_MAPPINGS = "strategy_mappings"
    LIVE_SETTINGS = "live_settings"

class MarketRegimeExcelManager:
    """
    Excel-based configuration manager for market regime system
    
    Provides Excel configuration management following backtester_v2 patterns:
    - Multiple sheet configuration files
    - Template generation
    - Validation and parsing
    - Dynamic parameter updates
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Excel configuration manager
        
        Args:
            config_path (str, optional): Path to configuration Excel file
        """
        self.config_path = config_path
        self.config_data = {}
        self.template_structure = self._define_template_structure()
        
        if config_path and Path(config_path).exists():
            self.load_configuration()
        
        logger.info("MarketRegimeExcelManager initialized")
    
    def _define_template_structure(self) -> Dict[str, Dict[str, Any]]:
        """Define the Excel template structure"""
        return {
            'RegimeDetectionConfig': {
                'description': 'Core regime detection parameters',
                'columns': [
                    'Parameter', 'Value', 'Description', 'DataType', 'MinValue', 'MaxValue'
                ],
                'default_data': [
                    ['ConfidenceThreshold', 0.6, 'Minimum confidence for regime classification', 'float', 0.0, 1.0],
                    ['RegimeSmoothing', 3, 'Number of periods for regime smoothing', 'int', 1, 10],
                    ['IndicatorWeightGreek', 0.35, 'Weight for Greek sentiment indicators', 'float', 0.0, 1.0],
                    ['IndicatorWeightOI', 0.25, 'Weight for OI analysis indicators', 'float', 0.0, 1.0],
                    ['IndicatorWeightPrice', 0.20, 'Weight for price action indicators', 'float', 0.0, 1.0],
                    ['IndicatorWeightTechnical', 0.15, 'Weight for technical indicators', 'float', 0.0, 1.0],
                    ['IndicatorWeightVolatility', 0.05, 'Weight for volatility measures', 'float', 0.0, 1.0],
                    ['DirectionalThresholdStrongBullish', 0.50, 'Threshold for strong bullish classification', 'float', 0.0, 1.0],
                    ['DirectionalThresholdMildBullish', 0.20, 'Threshold for mild bullish classification', 'float', 0.0, 1.0],
                    ['DirectionalThresholdNeutral', 0.10, 'Threshold for neutral classification', 'float', -0.5, 0.5],
                    ['DirectionalThresholdSideways', 0.05, 'Threshold for sideways classification', 'float', -0.2, 0.2],
                    ['DirectionalThresholdMildBearish', -0.20, 'Threshold for mild bearish classification', 'float', -1.0, 0.0],
                    ['DirectionalThresholdStrongBearish', -0.50, 'Threshold for strong bearish classification', 'float', -1.0, 0.0],
                    ['VolatilityThresholdHigh', 0.20, 'Threshold for high volatility classification', 'float', 0.0, 1.0],
                    ['VolatilityThresholdNormalHigh', 0.15, 'Threshold for normal-high volatility', 'float', 0.0, 1.0],
                    ['VolatilityThresholdNormalLow', 0.10, 'Threshold for normal-low volatility', 'float', 0.0, 1.0],
                    ['VolatilityThresholdLow', 0.05, 'Threshold for low volatility classification', 'float', 0.0, 1.0],
                    # ENHANCED: New technical indicators configuration
                    ['EnableIVPercentile', 'YES', 'Enable IV Percentile analysis', 'bool', 'YES', 'NO'],
                    ['EnableIVSkew', 'YES', 'Enable IV Skew analysis', 'bool', 'YES', 'NO'],
                    ['EnableEnhancedATR', 'YES', 'Enable Enhanced ATR indicators', 'bool', 'YES', 'NO'],
                    ['IVPercentileWeight', 0.15, 'IV Percentile analysis weight in regime formation', 'float', 0.0, 0.5],
                    ['IVSkewWeight', 0.15, 'IV Skew analysis weight in regime formation', 'float', 0.0, 0.5],
                    ['EnhancedATRWeight', 0.10, 'Enhanced ATR indicators weight in regime formation', 'float', 0.0, 0.3],
                    ['IVPercentileMinDataPoints', 30, 'Minimum data points for IV percentile calculation', 'int', 10, 100],
                    ['IVSkewMinStrikes', 5, 'Minimum strikes required for IV skew analysis', 'int', 3, 20],
                    ['ATRShortPeriod', 14, 'Short-term ATR calculation period', 'int', 5, 30],
                    ['ATRMediumPeriod', 21, 'Medium-term ATR calculation period', 'int', 10, 50],
                    ['ATRLongPeriod', 50, 'Long-term ATR calculation period', 'int', 20, 100]
                ]
            },
            'TechnicalIndicatorsConfig': {
                'description': 'Enhanced technical indicators configuration',
                'columns': [
                    'IndicatorType', 'Parameter', 'Value', 'Description', 'DataType', 'MinValue', 'MaxValue'
                ],
                'default_data': [
                    # IV Percentile Configuration
                    ['IVPercentile', 'EnableAnalysis', 'YES', 'Enable IV Percentile analysis', 'bool', 'YES', 'NO'],
                    ['IVPercentile', 'MinDataPoints', 30, 'Minimum historical data points required', 'int', 10, 100],
                    ['IVPercentile', 'ConfidenceWeight', 0.4, 'Weight for data quality in confidence calculation', 'float', 0.0, 1.0],
                    ['IVPercentile', 'ExtremelyLowThreshold', 10, 'Percentile threshold for extremely low IV', 'int', 0, 25],
                    ['IVPercentile', 'VeryLowThreshold', 25, 'Percentile threshold for very low IV', 'int', 10, 40],
                    ['IVPercentile', 'LowThreshold', 40, 'Percentile threshold for low IV', 'int', 25, 50],
                    ['IVPercentile', 'HighThreshold', 75, 'Percentile threshold for high IV', 'int', 60, 85],
                    ['IVPercentile', 'VeryHighThreshold', 90, 'Percentile threshold for very high IV', 'int', 80, 95],

                    # IV Skew Configuration
                    ['IVSkew', 'EnableAnalysis', 'YES', 'Enable IV Skew analysis', 'bool', 'YES', 'NO'],
                    ['IVSkew', 'MinStrikes', 5, 'Minimum strikes required for analysis', 'int', 3, 20],
                    ['IVSkew', 'StrikeRangePercent', 0.10, 'Strike range percentage from ATM', 'float', 0.05, 0.20],
                    ['IVSkew', 'ExtremelyBearishThreshold', -0.15, 'Threshold for extremely bearish skew', 'float', -0.30, -0.05],
                    ['IVSkew', 'VeryBearishThreshold', -0.10, 'Threshold for very bearish skew', 'float', -0.20, -0.05],
                    ['IVSkew', 'ModeratelyBearishThreshold', -0.05, 'Threshold for moderately bearish skew', 'float', -0.10, -0.02],
                    ['IVSkew', 'NeutralLowerThreshold', -0.02, 'Lower bound for neutral skew', 'float', -0.05, 0.0],
                    ['IVSkew', 'NeutralUpperThreshold', 0.02, 'Upper bound for neutral skew', 'float', 0.0, 0.05],
                    ['IVSkew', 'ModeratelyBullishThreshold', 0.05, 'Threshold for moderately bullish skew', 'float', 0.02, 0.10],
                    ['IVSkew', 'VeryBullishThreshold', 0.10, 'Threshold for very bullish skew', 'float', 0.05, 0.20],
                    ['IVSkew', 'ExtremelyBullishThreshold', 0.15, 'Threshold for extremely bullish skew', 'float', 0.05, 0.30],

                    # Enhanced ATR Configuration
                    ['EnhancedATR', 'EnableAnalysis', 'YES', 'Enable Enhanced ATR analysis', 'bool', 'YES', 'NO'],
                    ['EnhancedATR', 'ShortPeriod', 14, 'Short-term ATR period', 'int', 5, 30],
                    ['EnhancedATR', 'MediumPeriod', 21, 'Medium-term ATR period', 'int', 10, 50],
                    ['EnhancedATR', 'LongPeriod', 50, 'Long-term ATR period', 'int', 20, 100],
                    ['EnhancedATR', 'BandPeriods', 20, 'Volatility band calculation period', 'int', 10, 50],
                    ['EnhancedATR', 'BandStdMultiplier', 2.0, 'Standard deviation multiplier for bands', 'float', 1.0, 3.0],
                    ['EnhancedATR', 'BreakoutThreshold', 1.5, 'ATR multiplier for breakout detection', 'float', 1.0, 3.0],
                    ['EnhancedATR', 'ExtremelyLowThreshold', 10, 'Percentile threshold for extremely low ATR', 'int', 0, 25],
                    ['EnhancedATR', 'LowThreshold', 25, 'Percentile threshold for low ATR', 'int', 10, 40],
                    ['EnhancedATR', 'HighThreshold', 75, 'Percentile threshold for high ATR', 'int', 60, 85],
                    ['EnhancedATR', 'ExtremelyHighThreshold', 90, 'Percentile threshold for extremely high ATR', 'int', 80, 95]
                ]
            },
            'RegimeAdjustments': {
                'description': 'Strategy adjustments for each regime type',
                'columns': [
                    'RegimeType', 'EnableRegimeFilter', 'PositionSizeMultiplier', 'StopLossMultiplier', 
                    'TakeProfitMultiplier', 'UrgencyFactor', 'RiskTolerance', 'Description'
                ],
                'default_data': self._generate_regime_adjustments_data()
            },
            'StrategyMappings': {
                'description': 'Strategy-specific regime configurations',
                'columns': [
                    'StrategyType', 'RegimeType', 'EnableStrategy', 'WeightMultiplier', 
                    'CustomParameters', 'Notes'
                ],
                'default_data': self._generate_strategy_mappings_data()
            },
            'LiveTradingConfig': {
                'description': 'Live trading and streaming configuration',
                'columns': [
                    'Parameter', 'Value', 'Description', 'DataType', 'Category'
                ],
                'default_data': [
                    ['EnableLiveTrading', 'YES', 'Enable live trading integration', 'bool', 'Trading'],
                    ['StreamingIntervalMs', 100, 'Market data streaming interval in milliseconds', 'int', 'Streaming'],
                    ['RegimeUpdateFreqSec', 60, 'Regime detection update frequency in seconds', 'int', 'Detection'],
                    ['EnableAlgobobaIntegration', 'YES', 'Enable Algobaba order management', 'bool', 'Trading'],
                    ['MaxDailyOrders', 100, 'Maximum orders per day', 'int', 'Risk'],
                    ['MaxRegimeExposure', 0.5, 'Maximum exposure per regime (0.0-1.0)', 'float', 'Risk'],
                    ['EnableRegimeAlerts', 'YES', 'Enable regime change alerts', 'bool', 'Alerts'],
                    ['AlertChannels', 'EMAIL,TELEGRAM', 'Alert delivery channels', 'str', 'Alerts'],
                    ['RegimeHistoryLimit', 1000, 'Number of regime records to keep in memory', 'int', 'Memory'],
                    ['EnablePerformanceTracking', 'YES', 'Enable regime performance tracking', 'bool', 'Analytics'],
                    ['PerformanceWindowDays', 30, 'Performance analysis window in days', 'int', 'Analytics'],
                    ['EnableAutoRestart', 'YES', 'Enable automatic component restart on failure', 'bool', 'System'],
                    ['HealthCheckIntervalSec', 30, 'System health check interval in seconds', 'int', 'System']
                ]
            }
        }
    
    def _generate_regime_adjustments_data(self) -> List[List[Any]]:
        """Generate default regime adjustment data"""
        adjustments = []
        
        # Define adjustment patterns for each regime type
        regime_configs = {
            # Strong Bullish Regimes
            'HIGH_VOLATILE_STRONG_BULLISH': [1.5, 0.8, 1.3, 0.7, 'HIGH'],
            'NORMAL_VOLATILE_STRONG_BULLISH': [1.3, 0.9, 1.2, 0.8, 'MEDIUM_HIGH'],
            'LOW_VOLATILE_STRONG_BULLISH': [1.4, 1.0, 1.1, 0.9, 'MEDIUM'],
            
            # Mild Bullish Regimes
            'HIGH_VOLATILE_MILD_BULLISH': [1.1, 0.8, 1.1, 0.8, 'MEDIUM'],
            'NORMAL_VOLATILE_MILD_BULLISH': [1.0, 1.0, 1.0, 1.0, 'MEDIUM'],
            'LOW_VOLATILE_MILD_BULLISH': [1.2, 1.1, 1.0, 1.1, 'LOW'],
            
            # Neutral Regimes
            'HIGH_VOLATILE_NEUTRAL': [0.8, 0.7, 0.9, 0.9, 'MEDIUM'],
            'NORMAL_VOLATILE_NEUTRAL': [0.9, 1.0, 1.0, 1.0, 'MEDIUM'],
            'LOW_VOLATILE_NEUTRAL': [1.0, 1.2, 0.9, 1.2, 'LOW'],
            
            # Sideways Regimes
            'HIGH_VOLATILE_SIDEWAYS': [0.6, 0.6, 0.8, 0.8, 'HIGH'],
            'NORMAL_VOLATILE_SIDEWAYS': [0.7, 0.8, 0.9, 1.0, 'MEDIUM'],
            'LOW_VOLATILE_SIDEWAYS': [0.8, 1.0, 0.8, 1.3, 'LOW'],
            
            # Bearish Regimes
            'HIGH_VOLATILE_MILD_BEARISH': [1.1, 0.8, 1.1, 0.8, 'MEDIUM'],
            'NORMAL_VOLATILE_MILD_BEARISH': [1.0, 1.0, 1.0, 1.0, 'MEDIUM'],
            'LOW_VOLATILE_MILD_BEARISH': [1.2, 1.1, 1.0, 1.1, 'LOW'],
            'HIGH_VOLATILE_STRONG_BEARISH': [1.5, 0.8, 1.3, 0.7, 'HIGH'],
            'NORMAL_VOLATILE_STRONG_BEARISH': [1.3, 0.9, 1.2, 0.8, 'MEDIUM_HIGH'],
            'LOW_VOLATILE_STRONG_BEARISH': [1.4, 1.0, 1.1, 0.9, 'MEDIUM']
        }
        
        for regime_type, config in regime_configs.items():
            pos_mult, sl_mult, tp_mult, urgency, risk_tol = config
            adjustments.append([
                regime_type, 'YES', pos_mult, sl_mult, tp_mult, urgency, risk_tol,
                f'Optimized parameters for {regime_type.replace("_", " ").title()} market conditions'
            ])
        
        return adjustments
    
    def _generate_strategy_mappings_data(self) -> List[List[Any]]:
        """Generate default strategy mapping data"""
        mappings = []
        
        strategy_types = ['TBS', 'TV', 'OI', 'ORB', 'POS', 'ML_INDICATOR']
        regime_types = [regime.value for regime in Enhanced18RegimeType]
        
        # Define strategy preferences for different regimes
        strategy_preferences = {
            'TBS': {
                'HIGH_VOLATILE': 0.8,  # TBS works well in high volatility
                'STRONG_BULLISH': 1.2,
                'STRONG_BEARISH': 1.2,
                'SIDEWAYS': 1.5  # TBS excels in sideways markets
            },
            'TV': {
                'HIGH_VOLATILE': 1.3,  # TV signals work well in volatile markets
                'STRONG_BULLISH': 1.4,
                'STRONG_BEARISH': 1.4,
                'NEUTRAL': 0.7
            },
            'OI': {
                'HIGH_VOLATILE': 1.1,
                'STRONG_BULLISH': 1.3,
                'STRONG_BEARISH': 1.3,
                'SIDEWAYS': 0.8
            },
            'ORB': {
                'HIGH_VOLATILE': 1.4,  # ORB thrives in volatile breakouts
                'STRONG_BULLISH': 1.2,
                'STRONG_BEARISH': 1.2,
                'LOW_VOLATILE': 0.6
            },
            'POS': {
                'LOW_VOLATILE': 1.3,  # Positional strategies prefer stable markets
                'SIDEWAYS': 1.2,
                'HIGH_VOLATILE': 0.7
            },
            'ML_INDICATOR': {
                'HIGH_VOLATILE': 1.1,
                'NORMAL_VOLATILE': 1.2,
                'LOW_VOLATILE': 1.0
            }
        }
        
        for strategy_type in strategy_types:
            for regime_type in regime_types:
                # Calculate weight multiplier based on preferences
                weight_mult = 1.0
                prefs = strategy_preferences.get(strategy_type, {})
                
                for pattern, multiplier in prefs.items():
                    if pattern in regime_type:
                        weight_mult *= multiplier
                
                # Normalize weight multiplier
                weight_mult = max(0.1, min(2.0, weight_mult))
                
                # Determine if strategy should be enabled for this regime
                enable_strategy = 'YES' if weight_mult >= 0.5 else 'NO'
                
                mappings.append([
                    strategy_type, regime_type, enable_strategy, round(weight_mult, 2),
                    f'Auto-generated for {strategy_type}', 
                    f'Weight based on {strategy_type} performance in {regime_type} conditions'
                ])
        
        return mappings
    
    def generate_excel_template(self, output_path: str, regime_mode: str = "18") -> str:
        """
        Generate Excel configuration template

        Args:
            output_path (str): Path where to save the template
            regime_mode (str): Regime mode - "8", "12", or "18" (default: "18")

        Returns:
            str: Path to generated template file
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Adjust template structure based on regime mode
            template_structure = self._get_regime_specific_template_structure(regime_mode)

            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                for sheet_name, sheet_config in template_structure.items():
                    # Create DataFrame from template data
                    df = pd.DataFrame(
                        sheet_config['default_data'],
                        columns=sheet_config['columns']
                    )
                    
                    # Write to Excel
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Add description as a comment (if supported)
                    worksheet = writer.sheets[sheet_name]
                    try:
                        from openpyxl.comments import Comment
                        worksheet.cell(1, 1).comment = Comment(sheet_config['description'], 'System')
                    except:
                        # Skip comment if not supported
                        pass
            
            logger.info(f"Excel template generated: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Error generating Excel template: {e}")
            raise

    def _get_regime_specific_template_structure(self, regime_mode: str) -> Dict[str, Any]:
        """
        Get template structure adjusted for specific regime mode

        Args:
            regime_mode (str): Regime mode - "8", "12", or "18"

        Returns:
            Dict: Template structure for the specified regime mode
        """
        # Start with base template structure
        template_structure = self.template_structure.copy()

        # Adjust regime adjustments based on mode
        if regime_mode == "8":
            # 8-regime mode: simplified regime types
            regime_types = [
                'STRONG_BULLISH', 'BULLISH', 'MILD_BULLISH', 'NEUTRAL',
                'MILD_BEARISH', 'BEARISH', 'STRONG_BEARISH', 'VOLATILE_NEUTRAL'
            ]
        elif regime_mode == "12":
            # 12-regime mode: balanced complexity
            regime_types = [
                'LOW_DIRECTIONAL_TRENDING', 'LOW_DIRECTIONAL_RANGE', 'LOW_NONDIRECTIONAL_TRENDING',
                'LOW_NONDIRECTIONAL_RANGE', 'MODERATE_DIRECTIONAL_TRENDING', 'MODERATE_DIRECTIONAL_RANGE',
                'MODERATE_NONDIRECTIONAL_TRENDING', 'MODERATE_NONDIRECTIONAL_RANGE',
                'HIGH_DIRECTIONAL_TRENDING', 'HIGH_DIRECTIONAL_RANGE',
                'HIGH_NONDIRECTIONAL_TRENDING', 'HIGH_NONDIRECTIONAL_RANGE'
            ]
        else:  # regime_mode == "18"
            # 18-regime mode: full complexity
            regime_types = [
                'STRONG_BULLISH', 'BULLISH', 'MILD_BULLISH', 'NEUTRAL', 'MILD_BEARISH', 'BEARISH', 'STRONG_BEARISH',
                'HIGH_VOLATILE_BULLISH', 'HIGH_VOLATILE_BEARISH', 'HIGH_VOLATILE_NEUTRAL',
                'NORMAL_VOLATILE_BULLISH', 'NORMAL_VOLATILE_BEARISH', 'NORMAL_VOLATILE_NEUTRAL',
                'LOW_VOLATILE_BULLISH', 'LOW_VOLATILE_BEARISH', 'LOW_VOLATILE_NEUTRAL',
                'HIGH_VOLATILE_SIDEWAYS', 'LOW_VOLATILE_SIDEWAYS'
            ]

        # Update regime adjustments with appropriate regime types
        template_structure['RegimeAdjustments']['default_data'] = [
            [regime_type, 'YES', 1.0, 1.0, 1.0, 1.0, 'MEDIUM', f'Auto-generated for {regime_type}']
            for regime_type in regime_types
        ]

        # Update strategy mappings with appropriate regime types
        strategy_types = ['TBS', 'TV', 'ORB', 'OI', 'POS', 'INDICATOR']
        strategy_mappings = []
        for strategy_type in strategy_types:
            for regime_type in regime_types:
                strategy_mappings.append([
                    strategy_type, regime_type, 'YES', 1.0, '', f'Auto-generated for {strategy_type} in {regime_type}'
                ])

        template_structure['StrategyMappings']['default_data'] = strategy_mappings

        return template_structure
    
    def load_configuration(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from Excel file
        
        Args:
            config_path (str, optional): Path to configuration file
            
        Returns:
            Dict: Loaded configuration data
        """
        try:
            if config_path:
                self.config_path = config_path
            
            if not self.config_path or not Path(self.config_path).exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            # Load all sheets
            excel_file = pd.ExcelFile(self.config_path)
            
            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(self.config_path, sheet_name=sheet_name)
                    self.config_data[sheet_name] = df
                    logger.debug(f"Loaded sheet: {sheet_name} with {len(df)} rows")
                except Exception as e:
                    logger.warning(f"Failed to load sheet {sheet_name}: {e}")
            
            logger.info(f"Configuration loaded from: {self.config_path}")
            return self.config_data
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def get_detection_parameters(self) -> Dict[str, Any]:
        """Get regime detection parameters"""
        try:
            if 'RegimeDetectionConfig' not in self.config_data:
                return self._get_default_detection_parameters()
            
            df = self.config_data['RegimeDetectionConfig']
            params = {}
            
            for _, row in df.iterrows():
                param_name = row['Parameter']
                param_value = row['Value']
                data_type = row.get('DataType', 'str')
                
                # Convert to appropriate data type
                if data_type == 'float':
                    param_value = float(param_value)
                elif data_type == 'int':
                    param_value = int(param_value)
                elif data_type == 'bool':
                    param_value = str(param_value).upper() in ['YES', 'TRUE', '1']
                
                params[param_name] = param_value
            
            return params
            
        except Exception as e:
            logger.error(f"Error getting detection parameters: {e}")
            return self._get_default_detection_parameters()
    
    def get_regime_adjustments(self, regime_type: Optional[str] = None) -> Dict[str, Any]:
        """Get regime-specific adjustments"""
        try:
            if 'RegimeAdjustments' not in self.config_data:
                return self._get_default_regime_adjustments()
            
            df = self.config_data['RegimeAdjustments']
            
            if regime_type:
                # Get adjustments for specific regime
                regime_df = df[df['RegimeType'] == regime_type]
                if regime_df.empty:
                    logger.warning(f"No adjustments found for regime: {regime_type}")
                    return {}
                
                row = regime_df.iloc[0]
                return {
                    'enable_regime_filter': row.get('EnableRegimeFilter', 'YES') == 'YES',
                    'position_size_multiplier': float(row.get('PositionSizeMultiplier', 1.0)),
                    'stop_loss_multiplier': float(row.get('StopLossMultiplier', 1.0)),
                    'take_profit_multiplier': float(row.get('TakeProfitMultiplier', 1.0)),
                    'urgency_factor': float(row.get('UrgencyFactor', 1.0)),
                    'risk_tolerance': row.get('RiskTolerance', 'MEDIUM')
                }
            else:
                # Get all regime adjustments
                adjustments = {}
                for _, row in df.iterrows():
                    regime = row['RegimeType']
                    adjustments[regime] = {
                        'enable_regime_filter': row.get('EnableRegimeFilter', 'YES') == 'YES',
                        'position_size_multiplier': float(row.get('PositionSizeMultiplier', 1.0)),
                        'stop_loss_multiplier': float(row.get('StopLossMultiplier', 1.0)),
                        'take_profit_multiplier': float(row.get('TakeProfitMultiplier', 1.0)),
                        'urgency_factor': float(row.get('UrgencyFactor', 1.0)),
                        'risk_tolerance': row.get('RiskTolerance', 'MEDIUM')
                    }
                
                return adjustments
                
        except Exception as e:
            logger.error(f"Error getting regime adjustments: {e}")
            return self._get_default_regime_adjustments()
    
    def get_strategy_mappings(self, strategy_type: Optional[str] = None) -> Dict[str, Any]:
        """Get strategy-regime mappings"""
        try:
            if 'StrategyMappings' not in self.config_data:
                return {}
            
            df = self.config_data['StrategyMappings']
            
            if strategy_type:
                # Get mappings for specific strategy
                strategy_df = df[df['StrategyType'] == strategy_type]
                mappings = {}
                
                for _, row in strategy_df.iterrows():
                    regime = row['RegimeType']
                    mappings[regime] = {
                        'enable_strategy': row.get('EnableStrategy', 'YES') == 'YES',
                        'weight_multiplier': float(row.get('WeightMultiplier', 1.0)),
                        'custom_parameters': row.get('CustomParameters', ''),
                        'notes': row.get('Notes', '')
                    }
                
                return mappings
            else:
                # Get all strategy mappings
                mappings = {}
                for _, row in df.iterrows():
                    strategy = row['StrategyType']
                    regime = row['RegimeType']
                    
                    if strategy not in mappings:
                        mappings[strategy] = {}
                    
                    mappings[strategy][regime] = {
                        'enable_strategy': row.get('EnableStrategy', 'YES') == 'YES',
                        'weight_multiplier': float(row.get('WeightMultiplier', 1.0)),
                        'custom_parameters': row.get('CustomParameters', ''),
                        'notes': row.get('Notes', '')
                    }
                
                return mappings
                
        except Exception as e:
            logger.error(f"Error getting strategy mappings: {e}")
            return {}
    
    def get_live_trading_config(self) -> Dict[str, Any]:
        """Get live trading configuration"""
        try:
            if 'LiveTradingConfig' not in self.config_data:
                return self._get_default_live_config()

            df = self.config_data['LiveTradingConfig']
            config = {}

            for _, row in df.iterrows():
                param_name = row['Parameter']
                param_value = row['Value']
                data_type = row.get('DataType', 'str')

                # Convert to appropriate data type
                if data_type == 'bool':
                    param_value = str(param_value).upper() in ['YES', 'TRUE', '1']
                elif data_type == 'int':
                    param_value = int(param_value)
                elif data_type == 'float':
                    param_value = float(param_value)

                config[param_name] = param_value

            return config

        except Exception as e:
            logger.error(f"Error getting live trading config: {e}")
            return self._get_default_live_config()

    def get_technical_indicators_config(self) -> Dict[str, Dict[str, Any]]:
        """Get enhanced technical indicators configuration"""
        try:
            if 'TechnicalIndicatorsConfig' not in self.config_data:
                return self._get_default_technical_indicators_config()

            df = self.config_data['TechnicalIndicatorsConfig']
            config = {}

            # Group parameters by indicator type
            for _, row in df.iterrows():
                indicator_type = row['IndicatorType']
                param_name = row['Parameter']
                param_value = row['Value']
                data_type = row.get('DataType', 'str')

                # Convert to appropriate data type
                if data_type == 'bool':
                    param_value = str(param_value).upper() in ['YES', 'TRUE', '1']
                elif data_type == 'int':
                    param_value = int(param_value)
                elif data_type == 'float':
                    param_value = float(param_value)

                # Initialize indicator config if not exists
                if indicator_type not in config:
                    config[indicator_type] = {}

                config[indicator_type][param_name] = param_value

            return config

        except Exception as e:
            logger.error(f"Error getting technical indicators config: {e}")
            return self._get_default_technical_indicators_config()
    
    def _get_default_detection_parameters(self) -> Dict[str, Any]:
        """Get default detection parameters"""
        return {
            'ConfidenceThreshold': 0.6,
            'RegimeSmoothing': 3,
            'IndicatorWeightGreek': 0.35,
            'IndicatorWeightOI': 0.25,
            'IndicatorWeightPrice': 0.20,
            'IndicatorWeightTechnical': 0.15,
            'IndicatorWeightVolatility': 0.05
        }
    
    def _get_default_regime_adjustments(self) -> Dict[str, Any]:
        """Get default regime adjustments"""
        return {
            'NORMAL_VOLATILE_MILD_BULLISH': {
                'enable_regime_filter': True,
                'position_size_multiplier': 1.0,
                'stop_loss_multiplier': 1.0,
                'take_profit_multiplier': 1.0,
                'urgency_factor': 1.0,
                'risk_tolerance': 'MEDIUM'
            }
        }
    
    def _get_default_live_config(self) -> Dict[str, Any]:
        """Get default live trading configuration"""
        return {
            'EnableLiveTrading': False,
            'StreamingIntervalMs': 100,
            'RegimeUpdateFreqSec': 60,
            'EnableAlgobobaIntegration': False,
            'MaxDailyOrders': 100,
            'MaxRegimeExposure': 0.5
        }

    def _get_default_technical_indicators_config(self) -> Dict[str, Dict[str, Any]]:
        """Get default technical indicators configuration"""
        return {
            'IVPercentile': {
                'EnableAnalysis': True,
                'MinDataPoints': 30,
                'ConfidenceWeight': 0.4,
                'ExtremelyLowThreshold': 10,
                'VeryLowThreshold': 25,
                'LowThreshold': 40,
                'HighThreshold': 75,
                'VeryHighThreshold': 90
            },
            'IVSkew': {
                'EnableAnalysis': True,
                'MinStrikes': 5,
                'StrikeRangePercent': 0.10,
                'ExtremelyBearishThreshold': -0.15,
                'VeryBearishThreshold': -0.10,
                'ModeratelyBearishThreshold': -0.05,
                'NeutralLowerThreshold': -0.02,
                'NeutralUpperThreshold': 0.02,
                'ModeratelyBullishThreshold': 0.05,
                'VeryBullishThreshold': 0.10,
                'ExtremelyBullishThreshold': 0.15
            },
            'EnhancedATR': {
                'EnableAnalysis': True,
                'ShortPeriod': 14,
                'MediumPeriod': 21,
                'LongPeriod': 50,
                'BandPeriods': 20,
                'BandStdMultiplier': 2.0,
                'BreakoutThreshold': 1.5,
                'ExtremelyLowThreshold': 10,
                'LowThreshold': 25,
                'HighThreshold': 75,
                'ExtremelyHighThreshold': 90
            }
        }
    
    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """
        Validate loaded configuration
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []
        
        try:
            # Validate detection parameters
            detection_params = self.get_detection_parameters()
            
            # Check required parameters
            required_params = ['ConfidenceThreshold', 'RegimeSmoothing']
            for param in required_params:
                if param not in detection_params:
                    errors.append(f"Missing required parameter: {param}")
            
            # Validate ranges
            if 'ConfidenceThreshold' in detection_params:
                conf_threshold = detection_params['ConfidenceThreshold']
                if not 0.0 <= conf_threshold <= 1.0:
                    errors.append(f"ConfidenceThreshold must be between 0.0 and 1.0, got: {conf_threshold}")
            
            # Validate regime adjustments
            regime_adjustments = self.get_regime_adjustments()
            for regime, adjustments in regime_adjustments.items():
                if 'position_size_multiplier' in adjustments:
                    pos_mult = adjustments['position_size_multiplier']
                    if not 0.1 <= pos_mult <= 5.0:
                        errors.append(f"Invalid position_size_multiplier for {regime}: {pos_mult}")
            
            # Validate live trading config
            live_config = self.get_live_trading_config()
            if 'StreamingIntervalMs' in live_config:
                interval = live_config['StreamingIntervalMs']
                if not 10 <= interval <= 10000:
                    errors.append(f"StreamingIntervalMs should be between 10-10000ms, got: {interval}")

            # Validate technical indicators config
            tech_indicators_config = self.get_technical_indicators_config()
            for indicator_type, config in tech_indicators_config.items():
                if indicator_type == 'IVPercentile':
                    if 'MinDataPoints' in config:
                        min_data = config['MinDataPoints']
                        if not 10 <= min_data <= 100:
                            errors.append(f"IVPercentile MinDataPoints should be 10-100, got: {min_data}")

                    # Validate percentile thresholds are in ascending order
                    thresholds = ['ExtremelyLowThreshold', 'VeryLowThreshold', 'LowThreshold', 'HighThreshold', 'VeryHighThreshold']
                    prev_value = 0
                    for threshold in thresholds:
                        if threshold in config:
                            current_value = config[threshold]
                            if current_value <= prev_value:
                                errors.append(f"IVPercentile thresholds must be in ascending order: {threshold}")
                            prev_value = current_value

                elif indicator_type == 'IVSkew':
                    if 'MinStrikes' in config:
                        min_strikes = config['MinStrikes']
                        if not 3 <= min_strikes <= 20:
                            errors.append(f"IVSkew MinStrikes should be 3-20, got: {min_strikes}")

                    # Validate skew thresholds are in ascending order
                    skew_thresholds = [
                        'ExtremelyBearishThreshold', 'VeryBearishThreshold', 'ModeratelyBearishThreshold',
                        'NeutralLowerThreshold', 'NeutralUpperThreshold', 'ModeratelyBullishThreshold',
                        'VeryBullishThreshold', 'ExtremelyBullishThreshold'
                    ]
                    prev_value = -1.0
                    for threshold in skew_thresholds:
                        if threshold in config:
                            current_value = config[threshold]
                            if current_value <= prev_value:
                                errors.append(f"IVSkew thresholds must be in ascending order: {threshold}")
                            prev_value = current_value

                elif indicator_type == 'EnhancedATR':
                    # Validate ATR periods are in ascending order
                    if all(period in config for period in ['ShortPeriod', 'MediumPeriod', 'LongPeriod']):
                        short = config['ShortPeriod']
                        medium = config['MediumPeriod']
                        long = config['LongPeriod']
                        if not (short < medium < long):
                            errors.append("EnhancedATR periods must be in ascending order: Short < Medium < Long")

            is_valid = len(errors) == 0

            if is_valid:
                logger.info("Configuration validation passed")
            else:
                logger.warning(f"Configuration validation failed with {len(errors)} errors")

            return is_valid, errors
            
        except Exception as e:
            logger.error(f"Error during configuration validation: {e}")
            return False, [f"Validation error: {str(e)}"]
    
    def update_parameter(self, sheet_name: str, parameter: str, value: Any) -> bool:
        """
        Update a specific parameter in the configuration
        
        Args:
            sheet_name (str): Name of the Excel sheet
            parameter (str): Parameter name to update
            value (Any): New value for the parameter
            
        Returns:
            bool: True if update successful
        """
        try:
            if sheet_name not in self.config_data:
                logger.error(f"Sheet not found: {sheet_name}")
                return False
            
            df = self.config_data[sheet_name]
            
            if sheet_name in ['RegimeDetectionConfig', 'LiveTradingConfig']:
                # Update parameter-value format
                mask = df['Parameter'] == parameter
                if mask.any():
                    df.loc[mask, 'Value'] = value
                    logger.info(f"Updated {parameter} = {value} in {sheet_name}")
                    return True
                else:
                    logger.error(f"Parameter not found: {parameter} in {sheet_name}")
                    return False
            
            return False

        except Exception as e:
            logger.error(f"Error updating parameter: {e}")
            return False

    def hot_reload_configuration(self) -> bool:
        """
        Hot-reload configuration from Excel file without restarting the system

        Returns:
            bool: True if reload successful
        """
        try:
            if not self.config_path or not Path(self.config_path).exists():
                logger.error("Configuration file not found for hot reload")
                return False

            # Store current configuration as backup
            backup_config = self.config_data.copy()

            # Attempt to reload configuration
            new_config = self.load_configuration()

            # Validate new configuration
            is_valid, errors = self.validate_configuration()

            if not is_valid:
                # Restore backup configuration
                self.config_data = backup_config
                logger.error(f"Hot reload failed validation: {errors}")
                return False

            logger.info("Configuration hot-reloaded successfully")
            return True

        except Exception as e:
            logger.error(f"Error during hot reload: {e}")
            # Restore backup if available
            if 'backup_config' in locals():
                self.config_data = backup_config
            return False

    def get_indicator_analyzer_config(self, indicator_type: str) -> Dict[str, Any]:
        """
        Get configuration for specific technical indicator analyzer

        Args:
            indicator_type (str): Type of indicator ('IVPercentile', 'IVSkew', 'EnhancedATR')

        Returns:
            Dict: Configuration for the specified indicator
        """
        try:
            tech_config = self.get_technical_indicators_config()

            if indicator_type not in tech_config:
                logger.warning(f"Configuration not found for indicator: {indicator_type}")
                return self._get_default_technical_indicators_config().get(indicator_type, {})

            return tech_config[indicator_type]

        except Exception as e:
            logger.error(f"Error getting indicator config for {indicator_type}: {e}")
            return {}

    def update_indicator_parameter(self, indicator_type: str, parameter: str, value: Any) -> bool:
        """
        Update a specific parameter for a technical indicator

        Args:
            indicator_type (str): Type of indicator
            parameter (str): Parameter name
            value (Any): New parameter value

        Returns:
            bool: True if update successful
        """
        try:
            if 'TechnicalIndicatorsConfig' not in self.config_data:
                logger.error("TechnicalIndicatorsConfig sheet not found")
                return False

            df = self.config_data['TechnicalIndicatorsConfig']

            # Find the row to update
            mask = (df['IndicatorType'] == indicator_type) & (df['Parameter'] == parameter)

            if mask.any():
                df.loc[mask, 'Value'] = value
                logger.info(f"Updated {indicator_type}.{parameter} = {value}")
                return True
            else:
                logger.error(f"Parameter not found: {indicator_type}.{parameter}")
                return False

        except Exception as e:
            logger.error(f"Error updating indicator parameter: {e}")
            return False

    def save_configuration(self, output_path: Optional[str] = None) -> bool:
        """
        Save current configuration back to Excel file

        Args:
            output_path (str, optional): Path to save configuration

        Returns:
            bool: True if save successful
        """
        try:
            save_path = output_path or self.config_path

            if not save_path:
                logger.error("No save path specified")
                return False

            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
                for sheet_name, df in self.config_data.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

            logger.info(f"Configuration saved to: {save_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def save_configuration(self, output_path: Optional[str] = None) -> bool:
        """
        Save current configuration to Excel file
        
        Args:
            output_path (str, optional): Path to save configuration
            
        Returns:
            bool: True if save successful
        """
        try:
            save_path = output_path or self.config_path
            
            if not save_path:
                raise ValueError("No output path specified")
            
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
                for sheet_name, df in self.config_data.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            logger.info(f"Configuration saved to: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
