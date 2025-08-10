"""
Actual System Excel Manager

This module creates Excel configuration based on the ACTUAL existing system
at /srv/samba/shared/enhanced-market-regime-optimizer-final-package-updated/
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
import configparser

logger = logging.getLogger(__name__)

class ActualSystemExcelManager:
    """
    Excel configuration manager based on the ACTUAL existing system
    
    This creates Excel configuration sheets that match the existing
    comprehensive_indicator_config.ini and dynamic weightage system
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize based on actual system configuration"""
        self.config_path = config_path
        self.config_data = {}
        
        # Load existing INI configuration
        self.ini_config = self._load_existing_ini_config()
        
        # Define Excel structure based on actual system
        self.excel_structure = self._define_actual_excel_structure()
        
        logger.info("ActualSystemExcelManager initialized")
    
    def _load_existing_ini_config(self) -> configparser.ConfigParser:
        """Load the existing INI configuration"""
        config = configparser.ConfigParser()
        
        # Try to load from the actual system path
        ini_path = "/srv/samba/shared/enhanced-market-regime-optimizer-final-package-updated/config/comprehensive_indicator_config.ini"
        
        if Path(ini_path).exists():
            config.read(ini_path)
            logger.info(f"Loaded existing INI config from: {ini_path}")
        else:
            logger.warning("Could not load existing INI config, using defaults")
        
        return config
    
    def _define_actual_excel_structure(self) -> Dict[str, Dict[str, Any]]:
        """Define Excel structure based on ACTUAL existing system"""
        return {
            'IndicatorConfiguration': {
                'description': 'Configuration for all existing indicators from the actual system',
                'columns': [
                    'IndicatorSystem', 'Enabled', 'BaseWeight', 'PerformanceTracking', 
                    'AdaptiveWeight', 'ConfigSection', 'Parameters', 'Description'
                ],
                'data': self._generate_actual_indicator_config()
            },
            'StraddleAnalysisConfig': {
                'description': 'ATM/ITM/OTM straddle analysis with EMA/VWAP integration',
                'columns': [
                    'StraddleType', 'Enabled', 'Weight', 'EMAEnabled', 'EMAPeriods',
                    'VWAPEnabled', 'VWAPTypes', 'PreviousDayVWAP', 'Timeframes', 'Description'
                ],
                'data': self._generate_straddle_config()
            },
            'DynamicWeightageConfig': {
                'description': 'Dynamic weightage system with historical performance',
                'columns': [
                    'SystemName', 'CurrentWeight', 'HistoricalPerformance', 'LearningRate',
                    'MinWeight', 'MaxWeight', 'PerformanceWindow', 'UpdateFrequency', 'AutoAdjust'
                ],
                'data': self._generate_dynamic_weightage_config()
            },
            'MultiTimeframeConfig': {
                'description': 'Multi-timeframe analysis configuration (3-5-10-15 min)',
                'columns': [
                    'Timeframe', 'Enabled', 'Weight', 'Primary', 'ConfirmationRequired',
                    'RegimeStability', 'TransitionSensitivity', 'Description'
                ],
                'data': self._generate_timeframe_config()
            },
            'GreekSentimentConfig': {
                'description': 'Greek sentiment analysis configuration from actual system',
                'columns': [
                    'Parameter', 'Value', 'Description', 'Type', 'MinValue', 'MaxValue'
                ],
                'data': self._generate_greek_sentiment_config()
            },
            'RegimeFormationConfig': {
                'description': 'Market regime formation rules and thresholds (18 regime types)',
                'columns': [
                    'RegimeType', 'DirectionalThreshold', 'VolatilityThreshold',
                    'ConfidenceThreshold', 'MinDuration', 'Enabled', 'Description'
                ],
                'data': self._generate_regime_formation_config()
            },
            'RegimeComplexityConfig': {
                'description': 'Regime complexity configuration and mapping',
                'columns': [
                    'Setting', 'Value', 'Options', 'Description', 'Impact'
                ],
                'data': self._generate_regime_complexity_config()
            }
        }
    
    def _generate_actual_indicator_config(self) -> List[List[Any]]:
        """Generate indicator configuration based on actual system"""
        indicators = [
            # Existing systems from actual implementation
            ['greek_sentiment', True, 0.20, True, True, 'greek_sentiment', 
             'lookback_period=20,gamma_threshold=0.7,delta_threshold=0.6', 
             'Greek sentiment analysis from existing implementation'],
            
            ['trending_oi_pa', True, 0.18, True, True, 'trending_oi_pa',
             'oi_lookback=10,price_lookback=5,divergence_threshold=0.1',
             'Trending OI with PA from existing implementation'],
            
            ['ema_indicators', True, 0.15, True, True, 'ema_indicators',
             'ema_periods=20,50,100,200,timeframes=5m,15m,30m,60m',
             'EMA indicators from existing implementation'],
            
            ['vwap_indicators', True, 0.12, True, True, 'vwap_indicators',
             'current_day=true,previous_day=true,weekly=true,monthly=true',
             'VWAP indicators from existing implementation'],
            
            ['iv_skew', True, 0.10, True, True, 'iv_skew',
             'atm_range=3,otm_range=5,calculation_method=polynomial_fit',
             'IV skew from existing implementation'],
            
            ['iv_indicators', True, 0.08, True, True, 'iv_indicators',
             'iv_percentile_window=20,iv_rank_calculation=true',
             'IV indicators from existing implementation'],
            
            ['premium_indicators', True, 0.07, True, True, 'premium_indicators',
             'atm_straddle_premium_percentile=true,premium_decay_analysis=true',
             'Premium indicators from existing implementation'],
            
            ['atr_indicators', True, 0.06, True, True, 'atr_indicators',
             'period=14,ema_period=10,percentile_window=20',
             'ATR indicators from existing implementation'],
            
            ['straddle_analysis', True, 0.25, True, True, 'straddle_analysis',
             'atm_weight=0.4,itm_weight=0.3,otm_weight=0.3',
             'Enhanced straddle analysis with existing systems'],
            
            ['multi_timeframe_analysis', True, 0.04, True, True, 'multi_timeframe_analysis',
             'primary_timeframes=1m,5m,15m,30m,60m',
             'Multi-timeframe analysis from existing implementation']
        ]
        
        return indicators
    
    def _generate_straddle_config(self) -> List[List[Any]]:
        """Generate straddle configuration based on actual system"""
        straddles = [
            # ATM Straddle with EMA/VWAP integration
            ['ATM_STRADDLE', True, 0.4, True, '20,50,200', True, 
             'current_day,previous_day,weekly', True, '3m,5m,10m,15m',
             'ATM straddle with EMA and VWAP analysis'],
            
            ['ITM1_STRADDLE', True, 0.15, True, '20,50', True,
             'current_day,previous_day', True, '5m,15m',
             'ITM1 straddle analysis'],
            
            ['ITM2_STRADDLE', True, 0.10, True, '20,50', True,
             'current_day,previous_day', False, '5m,15m',
             'ITM2 straddle analysis'],
            
            ['ITM3_STRADDLE', True, 0.05, False, '', True,
             'current_day', False, '15m',
             'ITM3 straddle analysis'],
            
            ['OTM1_STRADDLE', True, 0.15, True, '20,50', True,
             'current_day,previous_day', True, '5m,15m',
             'OTM1 straddle analysis'],
            
            ['OTM2_STRADDLE', True, 0.10, True, '20,50', True,
             'current_day,previous_day', False, '5m,15m',
             'OTM2 straddle analysis'],
            
            ['OTM3_STRADDLE', True, 0.05, False, '', True,
             'current_day', False, '15m',
             'OTM3 straddle analysis']
        ]
        
        return straddles
    
    def _generate_dynamic_weightage_config(self) -> List[List[Any]]:
        """Generate dynamic weightage configuration"""
        systems = [
            ['greek_sentiment', 0.20, 0.85, 0.01, 0.05, 0.50, 30, 'daily', True],
            ['trending_oi_pa', 0.18, 0.78, 0.01, 0.05, 0.45, 30, 'daily', True],
            ['ema_indicators', 0.15, 0.82, 0.01, 0.05, 0.40, 30, 'daily', True],
            ['vwap_indicators', 0.12, 0.80, 0.01, 0.05, 0.35, 30, 'daily', True],
            ['iv_skew', 0.10, 0.75, 0.01, 0.03, 0.30, 30, 'daily', True],
            ['iv_indicators', 0.08, 0.73, 0.01, 0.03, 0.25, 30, 'daily', True],
            ['premium_indicators', 0.07, 0.77, 0.01, 0.02, 0.20, 30, 'daily', True],
            ['atr_indicators', 0.06, 0.79, 0.01, 0.02, 0.20, 30, 'daily', True],
            ['straddle_analysis', 0.25, 0.88, 0.01, 0.10, 0.60, 30, 'daily', True],
            ['multi_timeframe_analysis', 0.04, 0.72, 0.01, 0.02, 0.15, 30, 'daily', True]
        ]
        
        return systems
    
    def _generate_timeframe_config(self) -> List[List[Any]]:
        """Generate timeframe configuration for 3-5-10-15 min analysis"""
        timeframes = [
            ['3min', True, 0.15, False, False, 0.2, 0.8, 'High frequency regime detection'],
            ['5min', True, 0.35, True, True, 0.4, 0.6, 'Primary timeframe for regime analysis'],
            ['10min', True, 0.25, False, True, 0.6, 0.4, 'Medium-term regime confirmation'],
            ['15min', True, 0.20, False, True, 0.7, 0.3, 'Longer-term regime validation'],
            ['30min', True, 0.05, False, False, 0.8, 0.2, 'Long-term regime context']
        ]
        
        return timeframes
    
    def _generate_greek_sentiment_config(self) -> List[List[Any]]:
        """Generate Greek sentiment configuration from actual system"""
        config = [
            ['lookback_period', 20, 'Lookback period for Greek analysis', 'int', 5, 50],
            ['gamma_threshold', 0.7, 'Gamma threshold for sentiment', 'float', 0.1, 1.0],
            ['delta_threshold', 0.6, 'Delta threshold for sentiment', 'float', 0.1, 1.0],
            ['sentiment_weight', 0.4, 'Weight for sentiment calculation', 'float', 0.0, 1.0],
            ['current_week_weight', 0.7, 'Weight for current week expiry', 'float', 0.0, 1.0],
            ['next_week_weight', 0.2, 'Weight for next week expiry', 'float', 0.0, 1.0],
            ['current_month_weight', 0.1, 'Weight for current month expiry', 'float', 0.0, 1.0],
            ['vega_weight', 1.0, 'Vega weight in calculation', 'float', 0.0, 2.0],
            ['delta_weight', 1.0, 'Delta weight in calculation', 'float', 0.0, 2.0],
            ['theta_weight', 0.5, 'Theta weight in calculation', 'float', 0.0, 2.0],
            ['gamma_weight', 0.3, 'Gamma weight in calculation', 'float', 0.0, 2.0],
            ['max_delta', 0.5, 'Maximum delta for analysis', 'float', 0.1, 1.0],
            ['min_delta', 0.1, 'Minimum delta for analysis', 'float', 0.0, 0.5]
        ]
        
        return config
    
    def _generate_regime_formation_config(self, regime_mode: str = "18_REGIME") -> List[List[Any]]:
        """Generate regime formation configuration with support for 8/12/18 regime types"""
        regimes = [
            # Configuration header row
            ['REGIME_COMPLEXITY', regime_mode, 'N/A', 'N/A', 'N/A', True, f'Regime complexity: 8_REGIME, 12_REGIME, or 18_REGIME'],
        ]

        if regime_mode == "12_REGIME":
            # 12 Regime Types (Volatility × Trend × Structure = 12 regimes)
            regimes.extend([
                # Low Volatility Regimes
                ['LOW_DIRECTIONAL_TRENDING', 0.25, 0.30, 0.70, 5, True, 'Low volatility directional trending market'],
                ['LOW_DIRECTIONAL_RANGE', 0.25, 0.30, 0.50, 5, True, 'Low volatility directional range-bound market'],
                ['LOW_NONDIRECTIONAL_TRENDING', 0.25, 0.15, 0.70, 5, True, 'Low volatility non-directional trending market'],
                ['LOW_NONDIRECTIONAL_RANGE', 0.25, 0.15, 0.50, 5, True, 'Low volatility non-directional range-bound market'],

                # Moderate Volatility Regimes
                ['MODERATE_DIRECTIONAL_TRENDING', 0.50, 0.50, 0.70, 4, True, 'Moderate volatility directional trending market'],
                ['MODERATE_DIRECTIONAL_RANGE', 0.50, 0.50, 0.50, 4, True, 'Moderate volatility directional range-bound market'],
                ['MODERATE_NONDIRECTIONAL_TRENDING', 0.50, 0.20, 0.70, 4, True, 'Moderate volatility non-directional trending market'],
                ['MODERATE_NONDIRECTIONAL_RANGE', 0.50, 0.20, 0.50, 4, True, 'Moderate volatility non-directional range-bound market'],

                # High Volatility Regimes
                ['HIGH_DIRECTIONAL_TRENDING', 0.80, 0.70, 0.70, 3, True, 'High volatility directional trending market'],
                ['HIGH_DIRECTIONAL_RANGE', 0.80, 0.70, 0.50, 3, True, 'High volatility directional range-bound market'],
                ['HIGH_NONDIRECTIONAL_TRENDING', 0.80, 0.25, 0.70, 3, True, 'High volatility non-directional trending market'],
                ['HIGH_NONDIRECTIONAL_RANGE', 0.80, 0.25, 0.50, 3, True, 'High volatility non-directional range-bound market'],
            ])
        elif regime_mode == "18_REGIME":
            # 18 Regime Types (High/Normal/Low Volatile × Strong/Mild Bullish/Bearish/Neutral/Sideways)
            regimes.extend([
                # Strong Bullish Regimes
                ['HIGH_VOLATILE_STRONG_BULLISH', 0.70, 0.30, 0.85, 3, True, 'High volatility strong bullish regime'],
                ['NORMAL_VOLATILE_STRONG_BULLISH', 0.70, 0.20, 0.80, 5, True, 'Normal volatility strong bullish regime'],
                ['LOW_VOLATILE_STRONG_BULLISH', 0.70, 0.15, 0.75, 7, True, 'Low volatility strong bullish regime'],

                # Mild Bullish Regimes
                ['HIGH_VOLATILE_MILD_BULLISH', 0.35, 0.30, 0.75, 2, True, 'High volatility mild bullish regime'],
                ['NORMAL_VOLATILE_MILD_BULLISH', 0.35, 0.20, 0.70, 3, True, 'Normal volatility mild bullish regime'],
                ['LOW_VOLATILE_MILD_BULLISH', 0.35, 0.15, 0.65, 5, True, 'Low volatility mild bullish regime'],

                # Neutral Regimes
                ['HIGH_VOLATILE_NEUTRAL', 0.15, 0.30, 0.70, 2, True, 'High volatility neutral regime'],
                ['NORMAL_VOLATILE_NEUTRAL', 0.15, 0.20, 0.60, 3, True, 'Normal volatility neutral regime'],
                ['LOW_VOLATILE_NEUTRAL', 0.15, 0.15, 0.55, 4, True, 'Low volatility neutral regime'],

                # Sideways Regimes
                ['HIGH_VOLATILE_SIDEWAYS', 0.10, 0.30, 0.65, 2, True, 'High volatility sideways regime'],
                ['NORMAL_VOLATILE_SIDEWAYS', 0.10, 0.20, 0.55, 4, True, 'Normal volatility sideways regime'],
                ['LOW_VOLATILE_SIDEWAYS', 0.10, 0.15, 0.50, 6, True, 'Low volatility sideways regime'],

                # Mild Bearish Regimes
                ['HIGH_VOLATILE_MILD_BEARISH', -0.35, 0.30, 0.75, 2, True, 'High volatility mild bearish regime'],
                ['NORMAL_VOLATILE_MILD_BEARISH', -0.35, 0.20, 0.70, 3, True, 'Normal volatility mild bearish regime'],
                ['LOW_VOLATILE_MILD_BEARISH', -0.35, 0.15, 0.65, 5, True, 'Low volatility mild bearish regime'],

                # Strong Bearish Regimes
                ['HIGH_VOLATILE_STRONG_BEARISH', -0.70, 0.30, 0.85, 3, True, 'High volatility strong bearish regime'],
                ['NORMAL_VOLATILE_STRONG_BEARISH', -0.70, 0.20, 0.80, 5, True, 'Normal volatility strong bearish regime'],
                ['LOW_VOLATILE_STRONG_BEARISH', -0.70, 0.15, 0.75, 7, True, 'Low volatility strong bearish regime']
            ])
        else:  # 8_REGIME mode
            regimes.extend([
                # 8 Regime Types (simplified)
                ['HIGH_VOLATILE_BULLISH', 0.50, 0.30, 0.75, 3, True, 'High volatility bullish regime'],
                ['LOW_VOLATILE_BULLISH', 0.50, 0.15, 0.70, 5, True, 'Low volatility bullish regime'],
                ['HIGH_VOLATILE_NEUTRAL', 0.10, 0.30, 0.65, 3, True, 'High volatility neutral regime'],
                ['LOW_VOLATILE_NEUTRAL', 0.10, 0.15, 0.55, 5, True, 'Low volatility neutral regime'],
                ['HIGH_VOLATILE_SIDEWAYS', 0.05, 0.30, 0.60, 3, True, 'High volatility sideways regime'],
                ['LOW_VOLATILE_SIDEWAYS', 0.05, 0.15, 0.50, 5, True, 'Low volatility sideways regime'],
                ['HIGH_VOLATILE_BEARISH', -0.50, 0.30, 0.75, 3, True, 'High volatility bearish regime'],
                ['LOW_VOLATILE_BEARISH', -0.50, 0.15, 0.70, 5, True, 'Low volatility bearish regime']
            ])

        return regimes

    def _generate_regime_complexity_config(self) -> List[List[Any]]:
        """Generate regime complexity configuration with 12-regime support"""
        config = [
            ['REGIME_COMPLEXITY', '18_REGIME', '8_REGIME,12_REGIME,18_REGIME', 'Choose regime complexity level', 'Determines number of regime types'],
            ['VOLATILITY_LEVELS', '3', '2,3', 'Number of volatility levels (High/Normal/Low or High/Low)', 'Affects regime granularity'],
            ['DIRECTIONAL_LEVELS', '6', '2,4,6', 'Number of directional levels', 'Directional/Non-directional (12-regime) or Strong/Mild Bullish/Bearish/Neutral/Sideways (18-regime)'],
            ['STRUCTURE_LEVELS', '2', '1,2', 'Number of structure levels (12-regime only)', 'Trending/Range-bound structure analysis'],
            ['AUTO_SIMPLIFY', 'False', 'True,False', 'Auto-simplify to 8 regimes if needed', 'Fallback for performance'],
            ['REGIME_MAPPING_8', 'ENABLED', 'ENABLED,DISABLED', 'Enable 8-regime mapping', 'Maps 18/12 regimes to 8 for compatibility'],
            ['REGIME_MAPPING_12', 'ENABLED', 'ENABLED,DISABLED', 'Enable 12-regime mapping', 'Maps 18 regimes to 12 for optimization'],
            ['CONFIDENCE_BOOST_18', '0.05', '0.0,0.1', 'Confidence boost for 18-regime mode', 'Higher granularity bonus'],
            ['CONFIDENCE_BOOST_12', '0.03', '0.0,0.1', 'Confidence boost for 12-regime mode', 'Balanced granularity bonus'],
            ['TRANSITION_SMOOTHING', 'ENHANCED', 'BASIC,ENHANCED', 'Regime transition smoothing', 'Reduces false regime changes'],
            ['PERFORMANCE_TRACKING', 'PER_REGIME', 'AGGREGATE,PER_REGIME', 'Performance tracking granularity', 'Individual vs combined tracking'],
            ['TRIPLE_STRADDLE_WEIGHT', '0.35', '0.2,0.5', '12-regime Triple Straddle weight allocation', 'Weight for Triple Straddle Analysis in 12-regime system']
        ]

        return config
    
    def generate_excel_template(self, output_path: str) -> str:
        """Generate Excel template based on actual system"""
        try:
            logger.info(f"Generating Excel template based on actual system: {output_path}")
            
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                for sheet_name, sheet_config in self.excel_structure.items():
                    # Create DataFrame
                    df = pd.DataFrame(
                        sheet_config['data'],
                        columns=sheet_config['columns']
                    )
                    
                    # Write to Excel
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Add description
                    worksheet = writer.sheets[sheet_name]
                    worksheet.insert_rows(1)
                    worksheet['A1'] = f"Description: {sheet_config['description']}"
                    
                    logger.info(f"Generated sheet: {sheet_name} with {len(df)} rows")
            
            logger.info(f"✅ Excel template generated: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating Excel template: {e}")
            raise
    
    def load_configuration(self, config_path: str = None) -> bool:
        """Load configuration from Excel file"""
        try:
            if config_path:
                self.config_path = config_path
            
            if not self.config_path or not Path(self.config_path).exists():
                logger.error(f"Configuration file not found: {self.config_path}")
                return False
            
            # Read all sheets
            excel_data = pd.read_excel(self.config_path, sheet_name=None)
            
            # Process each sheet
            for sheet_name, df in excel_data.items():
                if sheet_name in self.excel_structure:
                    # Handle description row properly
                    if len(df) > 0 and str(df.columns[0]).startswith('Description:'):
                        # The description is in the header, actual data starts from row 0
                        # Set proper column names from the first row
                        if len(df) > 0:
                            new_columns = df.iloc[0].tolist()
                            df = df.iloc[1:].reset_index(drop=True)
                            df.columns = new_columns
                    elif len(df) > 0 and str(df.iloc[0, 0]).startswith('Description:'):
                        # The description is in the first row
                        df = df.iloc[1:].reset_index(drop=True)

                    self.config_data[sheet_name] = df
                    logger.info(f"Loaded {sheet_name}: {len(df)} rows")
            
            logger.info(f"✅ Configuration loaded from: {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return False
    
    def get_indicator_configuration(self) -> pd.DataFrame:
        """Get indicator configuration"""
        if 'IndicatorConfiguration' in self.config_data:
            return self.config_data['IndicatorConfiguration'].copy()
        else:
            return pd.DataFrame(
                self.excel_structure['IndicatorConfiguration']['data'],
                columns=self.excel_structure['IndicatorConfiguration']['columns']
            )
    
    def get_straddle_configuration(self) -> pd.DataFrame:
        """Get straddle configuration"""
        if 'StraddleAnalysisConfig' in self.config_data:
            return self.config_data['StraddleAnalysisConfig'].copy()
        else:
            return pd.DataFrame(
                self.excel_structure['StraddleAnalysisConfig']['data'],
                columns=self.excel_structure['StraddleAnalysisConfig']['columns']
            )
    
    def get_dynamic_weightage_configuration(self) -> pd.DataFrame:
        """Get dynamic weightage configuration"""
        if 'DynamicWeightageConfig' in self.config_data:
            return self.config_data['DynamicWeightageConfig'].copy()
        else:
            return pd.DataFrame(
                self.excel_structure['DynamicWeightageConfig']['data'],
                columns=self.excel_structure['DynamicWeightageConfig']['columns']
            )
    
    def get_timeframe_configuration(self) -> pd.DataFrame:
        """Get timeframe configuration"""
        if 'MultiTimeframeConfig' in self.config_data:
            return self.config_data['MultiTimeframeConfig'].copy()
        else:
            return pd.DataFrame(
                self.excel_structure['MultiTimeframeConfig']['data'],
                columns=self.excel_structure['MultiTimeframeConfig']['columns']
            )
    
    def get_greek_sentiment_configuration(self) -> pd.DataFrame:
        """Get Greek sentiment configuration"""
        if 'GreekSentimentConfig' in self.config_data:
            return self.config_data['GreekSentimentConfig'].copy()
        else:
            return pd.DataFrame(
                self.excel_structure['GreekSentimentConfig']['data'],
                columns=self.excel_structure['GreekSentimentConfig']['columns']
            )
    
    def get_regime_formation_configuration(self) -> pd.DataFrame:
        """Get regime formation configuration"""
        if 'RegimeFormationConfig' in self.config_data:
            return self.config_data['RegimeFormationConfig'].copy()
        else:
            return pd.DataFrame(
                self.excel_structure['RegimeFormationConfig']['data'],
                columns=self.excel_structure['RegimeFormationConfig']['columns']
            )

    def get_regime_complexity_configuration(self) -> pd.DataFrame:
        """Get regime complexity configuration"""
        if 'RegimeComplexityConfig' in self.config_data:
            return self.config_data['RegimeComplexityConfig'].copy()
        else:
            return pd.DataFrame(
                self.excel_structure['RegimeComplexityConfig']['data'],
                columns=self.excel_structure['RegimeComplexityConfig']['columns']
            )
    
    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """Validate the loaded configuration"""
        try:
            errors = []
            
            # Check if all required sheets are present
            required_sheets = list(self.excel_structure.keys())
            missing_sheets = [sheet for sheet in required_sheets if sheet not in self.config_data]
            
            if missing_sheets:
                errors.extend([f"Missing sheet: {sheet}" for sheet in missing_sheets])
            
            # Validate dynamic weights sum to 1.0
            if 'DynamicWeightageConfig' in self.config_data:
                df = self.config_data['DynamicWeightageConfig']
                auto_adjust_systems = df[df.get('AutoAdjust', True) == True]
                total_weight = auto_adjust_systems['CurrentWeight'].sum()
                
                if abs(total_weight - 1.0) > 0.01:
                    errors.append(f"Dynamic weights sum to {total_weight:.3f}, should be 1.0")
            
            # Validate timeframe weights
            if 'MultiTimeframeConfig' in self.config_data:
                df = self.config_data['MultiTimeframeConfig']
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
    
    def save_configuration(self, output_path: str = None) -> bool:
        """Save current configuration to Excel file"""
        try:
            save_path = output_path or self.config_path
            if not save_path:
                logger.error("No output path specified")
                return False
            
            with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
                for sheet_name, df in self.config_data.items():
                    # Add description row
                    description = self.excel_structure[sheet_name]['description']
                    
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
