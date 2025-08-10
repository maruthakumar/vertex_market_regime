"""
Enhanced Configurable Excel Manager for Market Regime Formation

This module provides highly configurable Excel-based configuration management
for market regime formation, focusing on indicator configuration, dynamic
weightage systems, historical performance-based optimization, and individual
user regime customization.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class IndicatorCategory(Enum):
    """Indicator categories for market regime formation"""
    GREEK_SENTIMENT = "greek_sentiment"
    OI_ANALYSIS = "oi_analysis"
    PRICE_ACTION = "price_action"
    TECHNICAL_INDICATORS = "technical_indicators"
    VOLATILITY_MEASURES = "volatility_measures"
    MOMENTUM_INDICATORS = "momentum_indicators"
    VOLUME_INDICATORS = "volume_indicators"
    OPTIONS_SPECIFIC = "options_specific"
    MARKET_BREADTH = "market_breadth"
    STRADDLE_ANALYSIS = "straddle_analysis"

class EnhancedConfigurableExcelManager:
    """
    Enhanced Excel configuration manager for highly configurable market regime formation

    This class provides comprehensive configuration management for:
    - All indicators with individual configuration
    - Dynamic weightage systems with historical performance
    - Custom regime definitions
    - Time-series regime storage
    - Individual user regime configurations
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Enhanced Configurable Excel Manager

        Args:
            config_path (str, optional): Path to configuration Excel file
        """
        self.config_path = config_path
        self.config_data = {}
        self.template_structure = self._define_enhanced_template_structure()

        if config_path and Path(config_path).exists():
            self.load_configuration()

        logger.info("EnhancedConfigurableExcelManager initialized")

    def _define_enhanced_template_structure(self) -> Dict[str, Dict[str, Any]]:
        """Define the enhanced Excel template structure for regime formation"""
        return {
            'IndicatorRegistry': {
                'description': 'Complete registry of all available indicators',
                'columns': [
                    'IndicatorID', 'IndicatorName', 'Category', 'SubCategory',
                    'DataSource', 'CalculationMethod', 'Parameters', 'Enabled',
                    'BaseWeight', 'MinWeight', 'MaxWeight', 'PerformanceWeight',
                    'TimeframeSensitive', 'DTESensitive', 'Description'
                ],
                'default_data': self._generate_indicator_registry_data()
            },
            'DynamicWeightageConfig': {
                'description': 'Dynamic weightage configuration and historical performance',
                'columns': [
                    'IndicatorID', 'CurrentWeight', 'HistoricalPerformance',
                    'PerformanceWindow', 'LearningRate', 'WeightAdjustmentMethod',
                    'MinPerformanceThreshold', 'MaxPerformanceThreshold',
                    'LastUpdated', 'UpdateFrequency', 'AutoAdjust'
                ],
                'default_data': self._generate_dynamic_weightage_data()
            },
            'RegimeDefinitionConfig': {
                'description': 'Custom market regime definitions and thresholds',
                'columns': [
                    'RegimeID', 'RegimeName', 'RegimeType', 'DirectionalThreshold',
                    'VolatilityThreshold', 'ConfidenceThreshold', 'MinDuration',
                    'TransitionRules', 'IndicatorCombination', 'CustomLogic',
                    'HistoricalAccuracy', 'Enabled', 'Description'
                ],
                'default_data': self._generate_regime_definition_data()
            },
            'ConfidenceScoreConfig': {
                'description': 'Confidence score calculation and regime change thresholds',
                'columns': [
                    'ScoreComponent', 'Weight', 'CalculationMethod', 'Threshold',
                    'HistoricalAccuracy', 'AdjustmentFactor', 'TimeDecay',
                    'VolatilityAdjustment', 'MarketConditionSensitive', 'Enabled'
                ],
                'default_data': self._generate_confidence_score_data()
            },
            'TimeframeConfig': {
                'description': 'Multi-timeframe analysis configuration',
                'columns': [
                    'Timeframe', 'Weight', 'LookbackPeriod', 'SmoothingFactor',
                    'RegimeStability', 'TransitionSensitivity', 'Enabled',
                    'PrimaryTimeframe', 'ConfirmationRequired', 'Description'
                ],
                'default_data': self._generate_timeframe_config_data()
            },
            'HistoricalPerformanceConfig': {
                'description': 'Historical performance analysis and optimization',
                'columns': [
                    'AnalysisType', 'LookbackPeriod', 'PerformanceMetric',
                    'BenchmarkType', 'OptimizationMethod', 'RebalanceFrequency',
                    'MinSampleSize', 'StatisticalSignificance', 'Enabled'
                ],
                'default_data': self._generate_historical_performance_data()
            },
            'UserRegimeProfiles': {
                'description': 'Individual user regime configuration profiles',
                'columns': [
                    'UserID', 'ProfileName', 'RegimePreferences', 'RiskTolerance',
                    'TimeHorizon', 'CustomWeights', 'ExcludedIndicators',
                    'CustomThresholds', 'CreatedDate', 'LastModified', 'Active'
                ],
                'default_data': self._generate_user_profile_data()
            }
        }

    def _generate_indicator_registry_data(self) -> List[List[Any]]:
        """Generate comprehensive indicator registry data"""
        indicators = [
            # Greek Sentiment Indicators
            ['DELTA_SENTIMENT', 'Delta Sentiment', 'GREEK_SENTIMENT', 'DELTA', 'OPTIONS_DATA',
             'WEIGHTED_AVERAGE', 'window=20,method=volume_weighted', True, 0.15, 0.05, 0.30, 0.85,
             True, True, 'Volume-weighted delta sentiment across strikes'],

            ['GAMMA_EXPOSURE', 'Gamma Exposure', 'GREEK_SENTIMENT', 'GAMMA', 'OPTIONS_DATA',
             'EXPOSURE_CALCULATION', 'strikes=ATM±5,weighting=distance', True, 0.12, 0.03, 0.25, 0.78,
             True, True, 'Market maker gamma exposure analysis'],

            ['THETA_DECAY', 'Theta Decay Impact', 'GREEK_SENTIMENT', 'THETA', 'OPTIONS_DATA',
             'TIME_DECAY_ANALYSIS', 'dte_sensitivity=high', True, 0.08, 0.02, 0.20, 0.72,
             False, True, 'Time decay impact on option premiums'],

            ['VEGA_SKEW', 'Vega Skew Analysis', 'GREEK_SENTIMENT', 'VEGA', 'OPTIONS_DATA',
             'SKEW_CALCULATION', 'call_put_ratio=true,iv_surface=true', True, 0.10, 0.03, 0.22, 0.80,
             True, False, 'Volatility skew through vega analysis'],

            # OI Analysis Indicators
            ['OI_MOMENTUM', 'OI Momentum', 'OI_ANALYSIS', 'MOMENTUM', 'OPTIONS_DATA',
             'MOMENTUM_CALCULATION', 'period=5,smoothing=ema', True, 0.18, 0.08, 0.35, 0.88,
             True, False, 'Open interest momentum analysis'],

            ['OI_CONCENTRATION', 'OI Concentration', 'OI_ANALYSIS', 'CONCENTRATION', 'OPTIONS_DATA',
             'CONCENTRATION_INDEX', 'strikes=ATM±10,method=herfindahl', True, 0.15, 0.05, 0.30, 0.82,
             False, False, 'Open interest concentration at key strikes'],

            ['PCR_ANALYSIS', 'Put-Call Ratio Analysis', 'OI_ANALYSIS', 'PCR', 'OPTIONS_DATA',
             'RATIO_CALCULATION', 'volume_oi_combined=true,smoothing=3', True, 0.20, 0.10, 0.40, 0.90,
             True, False, 'Put-call ratio with volume and OI'],

            # Price Action Indicators
            ['PRICE_MOMENTUM', 'Price Momentum', 'PRICE_ACTION', 'MOMENTUM', 'PRICE_DATA',
             'MOMENTUM_OSCILLATOR', 'period=14,smoothing=3', True, 0.16, 0.08, 0.30, 0.85,
             True, False, 'Price momentum with multiple timeframes'],

            ['SUPPORT_RESISTANCE', 'Support/Resistance', 'PRICE_ACTION', 'LEVELS', 'PRICE_DATA',
             'LEVEL_DETECTION', 'method=pivot_points,sensitivity=medium', True, 0.14, 0.06, 0.28, 0.79,
             True, False, 'Dynamic support and resistance levels'],

            ['BREAKOUT_ANALYSIS', 'Breakout Analysis', 'PRICE_ACTION', 'BREAKOUT', 'PRICE_DATA',
             'BREAKOUT_DETECTION', 'volume_confirmation=true,false_breakout_filter=true', True, 0.12, 0.05, 0.25, 0.83,
             True, False, 'Breakout detection with volume confirmation'],

            # Technical Indicators
            ['RSI_DIVERGENCE', 'RSI Divergence', 'TECHNICAL_INDICATORS', 'OSCILLATOR', 'PRICE_DATA',
             'DIVERGENCE_ANALYSIS', 'period=14,divergence_periods=5', True, 0.10, 0.03, 0.20, 0.76,
             True, False, 'RSI with divergence analysis'],

            ['MACD_SIGNAL', 'MACD Signal Analysis', 'TECHNICAL_INDICATORS', 'TREND', 'PRICE_DATA',
             'MACD_CALCULATION', 'fast=12,slow=26,signal=9,histogram=true', True, 0.12, 0.04, 0.22, 0.81,
             True, False, 'MACD with histogram analysis'],

            ['BOLLINGER_SQUEEZE', 'Bollinger Band Squeeze', 'TECHNICAL_INDICATORS', 'VOLATILITY', 'PRICE_DATA',
             'SQUEEZE_DETECTION', 'period=20,std_dev=2,squeeze_threshold=0.1', True, 0.08, 0.02, 0.18, 0.74,
             True, False, 'Bollinger band squeeze detection'],

            # Volatility Measures
            ['REALIZED_VOL', 'Realized Volatility', 'VOLATILITY_MEASURES', 'HISTORICAL', 'PRICE_DATA',
             'VOLATILITY_CALCULATION', 'period=20,annualized=true', True, 0.15, 0.08, 0.30, 0.87,
             True, False, 'Historical realized volatility'],

            ['IMPLIED_VOL_SURFACE', 'IV Surface Analysis', 'VOLATILITY_MEASURES', 'IMPLIED', 'OPTIONS_DATA',
             'SURFACE_ANALYSIS', 'strikes=ATM±20,dte_range=7-45', True, 0.18, 0.10, 0.35, 0.89,
             True, True, 'Implied volatility surface analysis'],

            ['VOL_SKEW', 'Volatility Skew', 'VOLATILITY_MEASURES', 'SKEW', 'OPTIONS_DATA',
             'SKEW_CALCULATION', 'call_skew=true,put_skew=true,atm_reference=true', True, 0.13, 0.05, 0.25, 0.84,
             False, True, 'Call and put volatility skew'],

            # Straddle Analysis
            ['STRADDLE_MOMENTUM', 'Straddle Momentum', 'STRADDLE_ANALYSIS', 'MOMENTUM', 'OPTIONS_DATA',
             'STRADDLE_CALCULATION', 'points=3,spacing=50,weighting=volume', True, 0.20, 0.12, 0.40, 0.91,
             True, True, 'Multi-point straddle momentum analysis'],

            ['STRADDLE_SKEW', 'Straddle Skew Analysis', 'STRADDLE_ANALYSIS', 'SKEW', 'OPTIONS_DATA',
             'SKEW_ANALYSIS', 'call_put_combined=true,time_decay_adjusted=true', True, 0.16, 0.08, 0.32, 0.86,
             True, True, 'Straddle-based skew analysis']
        ]

        return indicators

    def _generate_dynamic_weightage_data(self) -> List[List[Any]]:
        """Generate dynamic weightage configuration data"""
        weightage_data = []

        # Get all indicators from registry
        indicators = self._generate_indicator_registry_data()

        for indicator in indicators:
            indicator_id = indicator[0]
            base_weight = indicator[8]  # BaseWeight from registry
            performance_weight = indicator[11]  # PerformanceWeight from registry

            weightage_data.append([
                indicator_id,
                base_weight,  # CurrentWeight
                performance_weight,  # HistoricalPerformance (0.0-1.0)
                30,  # PerformanceWindow (days)
                0.1,  # LearningRate
                'PERFORMANCE_BASED',  # WeightAdjustmentMethod
                0.3,  # MinPerformanceThreshold
                0.95,  # MaxPerformanceThreshold
                datetime.now().strftime('%Y-%m-%d'),  # LastUpdated
                'DAILY',  # UpdateFrequency
                True  # AutoAdjust
            ])

        return weightage_data

    def _generate_regime_definition_data(self) -> List[List[Any]]:
        """Generate custom regime definition data"""
        regimes = [
            # High Volatility Regimes
            ['HV_STRONG_BULL', 'High Vol Strong Bullish', 'DIRECTIONAL', 0.70, 0.25, 0.80, 5,
             'momentum_confirmation=true', 'PRICE_ACTION+OI_ANALYSIS+VOLATILITY', 'custom_bull_logic', 0.85, True,
             'High volatility with strong bullish momentum'],

            ['HV_MILD_BULL', 'High Vol Mild Bullish', 'DIRECTIONAL', 0.35, 0.25, 0.70, 3,
             'volume_confirmation=true', 'GREEK_SENTIMENT+TECHNICAL', 'mild_bull_logic', 0.78, True,
             'High volatility with mild bullish bias'],

            ['HV_NEUTRAL', 'High Vol Neutral', 'NEUTRAL', 0.15, 0.25, 0.65, 2,
             'range_bound=true', 'VOLATILITY+STRADDLE_ANALYSIS', 'neutral_logic', 0.72, True,
             'High volatility with neutral direction'],

            ['HV_SIDEWAYS', 'High Vol Sideways', 'SIDEWAYS', 0.10, 0.25, 0.60, 4,
             'consolidation=true', 'SUPPORT_RESISTANCE+OI_CONCENTRATION', 'sideways_logic', 0.80, True,
             'High volatility sideways movement'],

            ['HV_MILD_BEAR', 'High Vol Mild Bearish', 'DIRECTIONAL', -0.35, 0.25, 0.70, 3,
             'volume_confirmation=true', 'GREEK_SENTIMENT+TECHNICAL', 'mild_bear_logic', 0.78, True,
             'High volatility with mild bearish bias'],

            ['HV_STRONG_BEAR', 'High Vol Strong Bearish', 'DIRECTIONAL', -0.70, 0.25, 0.80, 5,
             'momentum_confirmation=true', 'PRICE_ACTION+OI_ANALYSIS+VOLATILITY', 'custom_bear_logic', 0.85, True,
             'High volatility with strong bearish momentum'],

            # Normal Volatility Regimes
            ['NV_STRONG_BULL', 'Normal Vol Strong Bullish', 'DIRECTIONAL', 0.60, 0.15, 0.75, 4,
             'trend_confirmation=true', 'PRICE_MOMENTUM+OI_MOMENTUM', 'normal_bull_logic', 0.82, True,
             'Normal volatility with strong bullish trend'],

            ['NV_MILD_BULL', 'Normal Vol Mild Bullish', 'DIRECTIONAL', 0.30, 0.15, 0.65, 3,
             'gradual_trend=true', 'TECHNICAL+GREEK_SENTIMENT', 'gradual_bull_logic', 0.75, True,
             'Normal volatility with mild bullish trend'],

            ['NV_NEUTRAL', 'Normal Vol Neutral', 'NEUTRAL', 0.12, 0.15, 0.60, 2,
             'balanced=true', 'ALL_INDICATORS', 'balanced_logic', 0.70, True,
             'Normal volatility with neutral sentiment'],

            ['NV_SIDEWAYS', 'Normal Vol Sideways', 'SIDEWAYS', 0.08, 0.15, 0.55, 3,
             'range_trading=true', 'SUPPORT_RESISTANCE+STRADDLE', 'range_logic', 0.77, True,
             'Normal volatility range-bound trading'],

            ['NV_MILD_BEAR', 'Normal Vol Mild Bearish', 'DIRECTIONAL', -0.30, 0.15, 0.65, 3,
             'gradual_trend=true', 'TECHNICAL+GREEK_SENTIMENT', 'gradual_bear_logic', 0.75, True,
             'Normal volatility with mild bearish trend'],

            ['NV_STRONG_BEAR', 'Normal Vol Strong Bearish', 'DIRECTIONAL', -0.60, 0.15, 0.75, 4,
             'trend_confirmation=true', 'PRICE_MOMENTUM+OI_MOMENTUM', 'normal_bear_logic', 0.82, True,
             'Normal volatility with strong bearish trend'],

            # Low Volatility Regimes
            ['LV_STRONG_BULL', 'Low Vol Strong Bullish', 'DIRECTIONAL', 0.50, 0.08, 0.70, 6,
             'steady_trend=true', 'PRICE_ACTION+MOMENTUM', 'steady_bull_logic', 0.80, True,
             'Low volatility with steady bullish trend'],

            ['LV_MILD_BULL', 'Low Vol Mild Bullish', 'DIRECTIONAL', 0.25, 0.08, 0.60, 4,
             'slow_grind=true', 'TECHNICAL+SUPPORT_RESISTANCE', 'slow_bull_logic', 0.73, True,
             'Low volatility with slow bullish grind'],

            ['LV_NEUTRAL', 'Low Vol Neutral', 'NEUTRAL', 0.10, 0.08, 0.55, 2,
             'quiet_market=true', 'MINIMAL_INDICATORS', 'quiet_logic', 0.68, True,
             'Low volatility quiet market conditions'],

            ['LV_SIDEWAYS', 'Low Vol Sideways', 'SIDEWAYS', 0.06, 0.08, 0.50, 5,
             'tight_range=true', 'SUPPORT_RESISTANCE', 'tight_range_logic', 0.75, True,
             'Low volatility tight range trading'],

            ['LV_MILD_BEAR', 'Low Vol Mild Bearish', 'DIRECTIONAL', -0.25, 0.08, 0.60, 4,
             'slow_grind=true', 'TECHNICAL+SUPPORT_RESISTANCE', 'slow_bear_logic', 0.73, True,
             'Low volatility with slow bearish grind'],

            ['LV_STRONG_BEAR', 'Low Vol Strong Bearish', 'DIRECTIONAL', -0.50, 0.08, 0.70, 6,
             'steady_trend=true', 'PRICE_ACTION+MOMENTUM', 'steady_bear_logic', 0.80, True,
             'Low volatility with steady bearish trend']
        ]

        return regimes

    def _generate_confidence_score_data(self) -> List[List[Any]]:
        """Generate confidence score configuration data"""
        return [
            ['INDICATOR_AGREEMENT', 0.40, 'WEIGHTED_CONSENSUS', 0.70, 0.85, 1.0, 0.95, True, True, True],
            ['HISTORICAL_ACCURACY', 0.25, 'PERFORMANCE_BASED', 0.60, 0.80, 1.1, 0.90, True, False, True],
            ['SIGNAL_STRENGTH', 0.20, 'MAGNITUDE_BASED', 0.50, 0.75, 1.0, 0.85, False, True, True],
            ['MARKET_CONDITION', 0.10, 'VOLATILITY_ADJUSTED', 0.40, 0.70, 0.8, 0.80, True, True, True],
            ['TIME_CONSISTENCY', 0.05, 'TEMPORAL_STABILITY', 0.30, 0.65, 0.9, 0.75, True, False, True]
        ]

    def _generate_timeframe_config_data(self) -> List[List[Any]]:
        """Generate timeframe configuration data"""
        return [
            ['1min', 0.15, 20, 0.3, 0.2, 0.8, True, False, False, 'High frequency regime detection'],
            ['5min', 0.25, 50, 0.5, 0.4, 0.6, True, True, True, 'Primary timeframe for regime analysis'],
            ['15min', 0.30, 100, 0.7, 0.6, 0.4, True, False, True, 'Medium-term regime confirmation'],
            ['30min', 0.20, 150, 0.8, 0.7, 0.3, True, False, True, 'Longer-term regime validation'],
            ['1hour', 0.10, 200, 0.9, 0.8, 0.2, True, False, False, 'Long-term regime context']
        ]

    def _generate_historical_performance_data(self) -> List[List[Any]]:
        """Generate historical performance configuration data"""
        return [
            ['REGIME_ACCURACY', 30, 'CLASSIFICATION_ACCURACY', 'RANDOM_BASELINE', 'BAYESIAN_OPTIMIZATION', 'DAILY', 100, 0.95, True],
            ['INDICATOR_PERFORMANCE', 7, 'SHARPE_RATIO', 'MARKET_RETURN', 'GENETIC_ALGORITHM', 'WEEKLY', 50, 0.90, True],
            ['WEIGHT_OPTIMIZATION', 14, 'INFORMATION_RATIO', 'EQUAL_WEIGHT', 'GRADIENT_DESCENT', 'WEEKLY', 75, 0.85, True],
            ['REGIME_STABILITY', 60, 'TRANSITION_ACCURACY', 'PERSISTENCE_MODEL', 'ENSEMBLE_METHOD', 'MONTHLY', 200, 0.80, True],
            ['CONFIDENCE_CALIBRATION', 21, 'BRIER_SCORE', 'UNIFORM_CONFIDENCE', 'ISOTONIC_REGRESSION', 'WEEKLY', 150, 0.88, True]
        ]

    def _generate_user_profile_data(self) -> List[List[Any]]:
        """Generate user profile configuration data"""
        return [
            ['USER_001', 'Conservative Regime Profile', 'LOW_VOLATILITY_PREFERRED', 'LOW', 'LONG_TERM',
             'VOLATILITY:0.4,MOMENTUM:0.3,TECHNICAL:0.3', 'STRADDLE_ANALYSIS', 'CONFIDENCE:0.8,VOLATILITY:0.15',
             datetime.now().strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d'), True],

            ['USER_002', 'Aggressive Regime Profile', 'HIGH_VOLATILITY_PREFERRED', 'HIGH', 'SHORT_TERM',
             'MOMENTUM:0.5,VOLATILITY:0.3,GREEK:0.2', 'NONE', 'CONFIDENCE:0.6,VOLATILITY:0.25',
             datetime.now().strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d'), True],

            ['USER_003', 'Balanced Regime Profile', 'ALL_REGIMES', 'MEDIUM', 'MEDIUM_TERM',
             'EQUAL_WEIGHTS', 'NONE', 'DEFAULT_THRESHOLDS',
             datetime.now().strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d'), True]
        ]

    def generate_enhanced_excel_template(self, output_path: str) -> str:
        """
        Generate enhanced Excel template with all configuration sheets

        Args:
            output_path (str): Path for the generated template

        Returns:
            str: Path to generated template
        """
        try:
            import pandas as pd

            logger.info(f"Generating enhanced Excel template: {output_path}")

            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Generate each sheet
                for sheet_name, sheet_config in self.template_structure.items():
                    # Create DataFrame from configuration
                    df = pd.DataFrame(
                        sheet_config['default_data'],
                        columns=sheet_config['columns']
                    )

                    # Write to Excel
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

                    # Add description as a comment or separate row
                    worksheet = writer.sheets[sheet_name]
                    worksheet.insert_rows(1)
                    worksheet['A1'] = f"Description: {sheet_config['description']}"

                    logger.info(f"Generated sheet: {sheet_name} with {len(df)} rows")

            logger.info(f"✅ Enhanced Excel template generated: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error generating enhanced Excel template: {e}")
            raise

    def load_configuration(self) -> bool:
        """Load configuration from Excel file"""
        try:
            if not self.config_path or not Path(self.config_path).exists():
                logger.error(f"Configuration file not found: {self.config_path}")
                return False

            import pandas as pd

            # Read all sheets
            excel_data = pd.read_excel(self.config_path, sheet_name=None)

            # Process each sheet
            for sheet_name, df in excel_data.items():
                if sheet_name in self.template_structure:
                    # Skip description row if it exists
                    if len(df) > 0 and str(df.iloc[0, 0]).startswith('Description:'):
                        df = df.iloc[1:].reset_index(drop=True)

                    self.config_data[sheet_name] = df
                    logger.info(f"Loaded {sheet_name}: {len(df)} rows")

            logger.info(f"✅ Configuration loaded from: {self.config_path}")
            return True

        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return False

    def get_indicator_configuration(self) -> pd.DataFrame:
        """Get indicator registry configuration"""
        if 'IndicatorRegistry' in self.config_data:
            return self.config_data['IndicatorRegistry'].copy()
        else:
            import pandas as pd
            return pd.DataFrame(
                self.template_structure['IndicatorRegistry']['default_data'],
                columns=self.template_structure['IndicatorRegistry']['columns']
            )

    def get_dynamic_weightage_configuration(self) -> pd.DataFrame:
        """Get dynamic weightage configuration"""
        if 'DynamicWeightageConfig' in self.config_data:
            return self.config_data['DynamicWeightageConfig'].copy()
        else:
            import pandas as pd
            return pd.DataFrame(
                self.template_structure['DynamicWeightageConfig']['default_data'],
                columns=self.template_structure['DynamicWeightageConfig']['columns']
            )

    def get_regime_definitions(self) -> pd.DataFrame:
        """Get custom regime definitions"""
        if 'RegimeDefinitionConfig' in self.config_data:
            return self.config_data['RegimeDefinitionConfig'].copy()
        else:
            import pandas as pd
            return pd.DataFrame(
                self.template_structure['RegimeDefinitionConfig']['default_data'],
                columns=self.template_structure['RegimeDefinitionConfig']['columns']
            )

    def get_confidence_score_configuration(self) -> pd.DataFrame:
        """Get confidence score configuration"""
        if 'ConfidenceScoreConfig' in self.config_data:
            return self.config_data['ConfidenceScoreConfig'].copy()
        else:
            import pandas as pd
            return pd.DataFrame(
                self.template_structure['ConfidenceScoreConfig']['default_data'],
                columns=self.template_structure['ConfidenceScoreConfig']['columns']
            )

    def get_timeframe_configuration(self) -> pd.DataFrame:
        """Get timeframe configuration"""
        if 'TimeframeConfig' in self.config_data:
            return self.config_data['TimeframeConfig'].copy()
        else:
            import pandas as pd
            return pd.DataFrame(
                self.template_structure['TimeframeConfig']['default_data'],
                columns=self.template_structure['TimeframeConfig']['columns']
            )

    def get_historical_performance_configuration(self) -> pd.DataFrame:
        """Get historical performance configuration"""
        if 'HistoricalPerformanceConfig' in self.config_data:
            return self.config_data['HistoricalPerformanceConfig'].copy()
        else:
            import pandas as pd
            return pd.DataFrame(
                self.template_structure['HistoricalPerformanceConfig']['default_data'],
                columns=self.template_structure['HistoricalPerformanceConfig']['columns']
            )

    def get_user_profiles(self) -> pd.DataFrame:
        """Get user regime profiles"""
        if 'UserRegimeProfiles' in self.config_data:
            return self.config_data['UserRegimeProfiles'].copy()
        else:
            import pandas as pd
            return pd.DataFrame(
                self.template_structure['UserRegimeProfiles']['default_data'],
                columns=self.template_structure['UserRegimeProfiles']['columns']
            )

    def update_indicator_weight(self, indicator_id: str, new_weight: float) -> bool:
        """Update indicator weight in dynamic weightage configuration"""
        try:
            if 'DynamicWeightageConfig' not in self.config_data:
                logger.error("Dynamic weightage configuration not loaded")
                return False

            df = self.config_data['DynamicWeightageConfig']
            mask = df['IndicatorID'] == indicator_id

            if mask.any():
                df.loc[mask, 'CurrentWeight'] = new_weight
                df.loc[mask, 'LastUpdated'] = datetime.now().strftime('%Y-%m-%d')
                logger.info(f"Updated weight for {indicator_id}: {new_weight}")
                return True
            else:
                logger.error(f"Indicator {indicator_id} not found")
                return False

        except Exception as e:
            logger.error(f"Error updating indicator weight: {e}")
            return False

    def update_regime_threshold(self, regime_id: str, threshold_type: str, new_value: float) -> bool:
        """Update regime threshold in regime definition configuration"""
        try:
            if 'RegimeDefinitionConfig' not in self.config_data:
                logger.error("Regime definition configuration not loaded")
                return False

            df = self.config_data['RegimeDefinitionConfig']
            mask = df['RegimeID'] == regime_id

            if mask.any() and threshold_type in df.columns:
                df.loc[mask, threshold_type] = new_value
                logger.info(f"Updated {threshold_type} for {regime_id}: {new_value}")
                return True
            else:
                logger.error(f"Regime {regime_id} or threshold {threshold_type} not found")
                return False

        except Exception as e:
            logger.error(f"Error updating regime threshold: {e}")
            return False

    def save_configuration(self, output_path: str = None) -> bool:
        """Save current configuration to Excel file"""
        try:
            save_path = output_path or self.config_path
            if not save_path:
                logger.error("No output path specified")
                return False

            import pandas as pd

            with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
                for sheet_name, df in self.config_data.items():
                    # Add description row
                    description = self.template_structure[sheet_name]['description']

                    # Create new DataFrame with description
                    desc_row = pd.DataFrame([['Description: ' + description] + [''] * (len(df.columns) - 1)],
                                          columns=df.columns)
                    final_df = pd.concat([desc_row, df], ignore_index=True)

                    final_df.to_excel(writer, sheet_name=sheet_name, index=False)

            logger.info(f"✅ Configuration saved to: {save_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False

    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """Validate the loaded configuration"""
        try:
            errors = []

            # Check if all required sheets are present
            required_sheets = list(self.template_structure.keys())
            missing_sheets = [sheet for sheet in required_sheets if sheet not in self.config_data]

            if missing_sheets:
                errors.extend([f"Missing sheet: {sheet}" for sheet in missing_sheets])

            # Validate indicator weights sum to 1.0
            if 'DynamicWeightageConfig' in self.config_data:
                df = self.config_data['DynamicWeightageConfig']
                enabled_indicators = df[df.get('AutoAdjust', True) == True]
                total_weight = enabled_indicators['CurrentWeight'].sum()

                if abs(total_weight - 1.0) > 0.01:
                    errors.append(f"Indicator weights sum to {total_weight:.3f}, should be 1.0")

            # Validate regime thresholds
            if 'RegimeDefinitionConfig' in self.config_data:
                df = self.config_data['RegimeDefinitionConfig']
                for _, row in df.iterrows():
                    if row['ConfidenceThreshold'] < 0 or row['ConfidenceThreshold'] > 1:
                        errors.append(f"Invalid confidence threshold for {row['RegimeID']}: {row['ConfidenceThreshold']}")

            # Validate timeframe weights
            if 'TimeframeConfig' in self.config_data:
                df = self.config_data['TimeframeConfig']
                enabled_timeframes = df[df['Enabled'] == True]
                total_weight = enabled_timeframes['Weight'].sum()

                if abs(total_weight - 1.0) > 0.01:
                    errors.append(f"Timeframe weights sum to {total_weight:.3f}, should be 1.0")

            is_valid = len(errors) == 0

            if is_valid:
                logger.info("✅ Configuration validation passed")
            else:
                logger.warning(f"Configuration validation failed: {len(errors)} errors")

            return is_valid, errors

        except Exception as e:
            logger.error(f"Error validating configuration: {e}")
            return False, [f"Validation error: {str(e)}"]

    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration"""
        try:
            summary = {
                'config_path': self.config_path,
                'sheets_loaded': list(self.config_data.keys()),
                'total_indicators': 0,
                'enabled_indicators': 0,
                'total_regimes': 0,
                'enabled_regimes': 0,
                'user_profiles': 0,
                'last_loaded': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            # Count indicators
            if 'IndicatorRegistry' in self.config_data:
                df = self.config_data['IndicatorRegistry']
                summary['total_indicators'] = len(df)
                summary['enabled_indicators'] = len(df[df['Enabled'] == True])

            # Count regimes
            if 'RegimeDefinitionConfig' in self.config_data:
                df = self.config_data['RegimeDefinitionConfig']
                summary['total_regimes'] = len(df)
                summary['enabled_regimes'] = len(df[df['Enabled'] == True])

            # Count user profiles
            if 'UserRegimeProfiles' in self.config_data:
                df = self.config_data['UserRegimeProfiles']
                summary['user_profiles'] = len(df[df['Active'] == True])

            return summary

        except Exception as e:
            logger.error(f"Error getting configuration summary: {e}")
            return {'error': str(e)}