"""
Market Regime Configuration Parser

This module handles parsing of Excel configuration files for market regime
detection, following the backtester_v2 patterns for configuration management.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
from .models import (
    RegimeConfig, IndicatorConfig, TimeframeConfig, 
    IndicatorCategory, RegimeType
)

logger = logging.getLogger(__name__)

class RegimeConfigParser:
    """Parse market regime configuration from Excel files"""
    
    def __init__(self):
        """Initialize the parser"""
        self.required_sheets = [
            'Indicator_Registry',
            'Timeframe_Configuration', 
            'General_Settings',
            'Performance_Settings'
        ]
        
    def parse(self, file_path: str) -> RegimeConfig:
        """
        Parse Excel configuration file
        
        Args:
            file_path (str): Path to Excel configuration file
            
        Returns:
            RegimeConfig: Parsed configuration
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {file_path}")
            
            # Read all sheets
            sheets = pd.read_excel(file_path, sheet_name=None)
            
            # Validate required sheets
            self._validate_sheets(sheets)
            
            # Parse each section
            indicators = self._parse_indicators(sheets['Indicator_Registry'])
            timeframes = self._parse_timeframes(sheets['Timeframe_Configuration'])
            general_settings = self._parse_general_settings(sheets['General_Settings'])
            performance_settings = self._parse_performance_settings(sheets['Performance_Settings'])
            
            # Combine into configuration
            config_dict = {
                'indicators': indicators,
                'timeframes': timeframes,
                **general_settings,
                **performance_settings
            }
            
            config = RegimeConfig(**config_dict)
            logger.info(f"Successfully parsed configuration with {len(indicators)} indicators")
            
            return config
            
        except Exception as e:
            logger.error(f"Error parsing configuration file {file_path}: {e}")
            raise
    
    def _validate_sheets(self, sheets: Dict[str, pd.DataFrame]):
        """Validate that all required sheets are present"""
        missing_sheets = [sheet for sheet in self.required_sheets if sheet not in sheets]
        if missing_sheets:
            raise ValueError(f"Missing required sheets: {missing_sheets}")
    
    def _parse_indicators(self, df: pd.DataFrame) -> List[IndicatorConfig]:
        """Parse indicator configuration"""
        indicators = []
        
        # Validate required columns
        required_cols = ['Indicator_ID', 'Indicator_Name', 'Category', 'Type', 'Enabled']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in Indicator_Registry: {missing_cols}")
        
        for _, row in df.iterrows():
            if not row.get('Enabled', True):
                continue
                
            try:
                # Map category string to enum
                category_str = str(row['Category']).upper().replace(' ', '_')
                category = IndicatorCategory(category_str)
                
                # Create indicator config
                indicator_config = IndicatorConfig(
                    id=str(row['Indicator_ID']),
                    name=str(row['Indicator_Name']),
                    category=category,
                    indicator_type=str(row['Type']),
                    base_weight=float(row.get('Base_Weight', 0.1)),
                    min_weight=float(row.get('Min_Weight', 0.01)),
                    max_weight=float(row.get('Max_Weight', 0.5)),
                    enabled=bool(row.get('Enabled', True)),
                    adaptive=bool(row.get('Adaptive', True)),
                    lookback_periods=int(row.get('Lookback_Periods', 20)),
                    parameters=self._parse_parameters(row.get('Parameters', ''))
                )
                
                indicators.append(indicator_config)
                
            except Exception as e:
                logger.warning(f"Error parsing indicator {row.get('Indicator_ID', 'unknown')}: {e}")
                continue
        
        if not indicators:
            raise ValueError("No valid indicators found in configuration")
        
        return indicators
    
    def _parse_timeframes(self, df: pd.DataFrame) -> List[TimeframeConfig]:
        """Parse timeframe configuration"""
        timeframes = []
        
        # Validate required columns
        required_cols = ['Timeframe_Minutes', 'Weight', 'Enabled']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in Timeframe_Configuration: {missing_cols}")
        
        for _, row in df.iterrows():
            if not row.get('Enabled', True):
                continue
                
            try:
                timeframe_config = TimeframeConfig(
                    timeframe_minutes=int(row['Timeframe_Minutes']),
                    weight=float(row['Weight']),
                    enabled=bool(row.get('Enabled', True))
                )
                
                timeframes.append(timeframe_config)
                
            except Exception as e:
                logger.warning(f"Error parsing timeframe {row.get('Timeframe_Minutes', 'unknown')}: {e}")
                continue
        
        if not timeframes:
            # Default timeframes
            timeframes = [
                TimeframeConfig(timeframe_minutes=1, weight=0.1),
                TimeframeConfig(timeframe_minutes=5, weight=0.3),
                TimeframeConfig(timeframe_minutes=15, weight=0.4),
                TimeframeConfig(timeframe_minutes=30, weight=0.2)
            ]
            logger.info("Using default timeframe configuration")
        
        return timeframes
    
    def _parse_general_settings(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse general settings"""
        settings = {}
        
        # Convert DataFrame to dictionary
        for _, row in df.iterrows():
            setting_name = str(row.get('Setting', '')).strip()
            setting_value = row.get('Value', '')
            
            if not setting_name:
                continue
            
            # Map setting names to config fields
            setting_map = {
                'strategy_name': 'strategy_name',
                'symbol': 'symbol',
                'lookback_days': 'lookback_days',
                'update_frequency': 'update_frequency',
                'confidence_threshold': 'confidence_threshold',
                'regime_smoothing': 'regime_smoothing',
                'enable_gpu': 'enable_gpu',
                'enable_caching': 'enable_caching'
            }
            
            if setting_name.lower() in setting_map:
                config_field = setting_map[setting_name.lower()]
                
                # Type conversion
                if config_field in ['lookback_days', 'regime_smoothing']:
                    settings[config_field] = int(setting_value)
                elif config_field in ['confidence_threshold']:
                    settings[config_field] = float(setting_value)
                elif config_field in ['enable_gpu', 'enable_caching']:
                    settings[config_field] = bool(setting_value)
                else:
                    settings[config_field] = str(setting_value)
        
        return settings
    
    def _parse_performance_settings(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse performance tracking settings"""
        settings = {}
        
        # Convert DataFrame to dictionary
        for _, row in df.iterrows():
            setting_name = str(row.get('Setting', '')).strip()
            setting_value = row.get('Value', '')
            
            if not setting_name:
                continue
            
            # Map setting names to config fields
            setting_map = {
                'performance_window': 'performance_window',
                'learning_rate': 'learning_rate'
            }
            
            if setting_name.lower() in setting_map:
                config_field = setting_map[setting_name.lower()]
                
                # Type conversion
                if config_field == 'performance_window':
                    settings[config_field] = int(setting_value)
                elif config_field == 'learning_rate':
                    settings[config_field] = float(setting_value)
        
        return settings
    
    def _parse_parameters(self, param_str: str) -> Dict[str, Any]:
        """Parse parameter string into dictionary"""
        parameters = {}
        
        if not param_str or pd.isna(param_str):
            return parameters
        
        try:
            # Handle JSON-like parameter strings
            if param_str.strip().startswith('{'):
                import json
                parameters = json.loads(param_str)
            else:
                # Handle key=value pairs separated by semicolons
                pairs = param_str.split(';')
                for pair in pairs:
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Try to convert to appropriate type
                        try:
                            # Try integer
                            parameters[key] = int(value)
                        except ValueError:
                            try:
                                # Try float
                                parameters[key] = float(value)
                            except ValueError:
                                # Try boolean
                                if value.lower() in ['true', 'false']:
                                    parameters[key] = value.lower() == 'true'
                                else:
                                    # Keep as string
                                    parameters[key] = value
        
        except Exception as e:
            logger.warning(f"Error parsing parameters '{param_str}': {e}")
        
        return parameters
    
    def create_template(self, output_path: str):
        """Create a template Excel configuration file"""
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Indicator Registry sheet
                indicator_data = {
                    'Indicator_ID': ['ema_trend', 'vwap_trend', 'greek_sentiment', 'iv_skew', 'premium_analysis'],
                    'Indicator_Name': ['EMA Trend', 'VWAP Trend', 'Greek Sentiment', 'IV Skew', 'Premium Analysis'],
                    'Category': ['PRICE_TREND', 'PRICE_TREND', 'GREEK_SENTIMENT', 'IV_ANALYSIS', 'PREMIUM_ANALYSIS'],
                    'Type': ['ema', 'vwap', 'greek', 'iv_skew', 'premium'],
                    'Base_Weight': [0.25, 0.20, 0.20, 0.15, 0.20],
                    'Min_Weight': [0.05, 0.05, 0.05, 0.05, 0.05],
                    'Max_Weight': [0.50, 0.50, 0.50, 0.50, 0.50],
                    'Enabled': [True, True, True, True, True],
                    'Adaptive': [True, True, True, True, True],
                    'Lookback_Periods': [20, 20, 20, 20, 20],
                    'Parameters': ['periods=5,10,20', 'bands=2', '', 'lookback=30', 'strikes=3']
                }
                pd.DataFrame(indicator_data).to_excel(writer, sheet_name='Indicator_Registry', index=False)
                
                # Timeframe Configuration sheet
                timeframe_data = {
                    'Timeframe_Minutes': [1, 5, 15, 30],
                    'Weight': [0.1, 0.3, 0.4, 0.2],
                    'Enabled': [True, True, True, True]
                }
                pd.DataFrame(timeframe_data).to_excel(writer, sheet_name='Timeframe_Configuration', index=False)
                
                # General Settings sheet
                general_data = {
                    'Setting': ['strategy_name', 'symbol', 'lookback_days', 'update_frequency', 'confidence_threshold', 'regime_smoothing', 'enable_gpu', 'enable_caching'],
                    'Value': ['MarketRegime', 'NIFTY', 252, 'MINUTE', 0.6, 3, True, True],
                    'Description': ['Strategy name', 'Symbol to analyze', 'Historical lookback days', 'Update frequency', 'Minimum confidence threshold', 'Regime smoothing periods', 'Enable GPU acceleration', 'Enable result caching']
                }
                pd.DataFrame(general_data).to_excel(writer, sheet_name='General_Settings', index=False)
                
                # Performance Settings sheet
                performance_data = {
                    'Setting': ['performance_window', 'learning_rate'],
                    'Value': [100, 0.01],
                    'Description': ['Performance tracking window', 'Adaptive learning rate']
                }
                pd.DataFrame(performance_data).to_excel(writer, sheet_name='Performance_Settings', index=False)
            
            logger.info(f"Template configuration file created: {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating template file: {e}")
            raise

    def validate_config(self, config: RegimeConfig) -> Dict[str, List[str]]:
        """
        Validate configuration for common issues

        Args:
            config (RegimeConfig): Configuration to validate

        Returns:
            Dict[str, List[str]]: Validation results with warnings and errors
        """
        validation_results = {
            'errors': [],
            'warnings': []
        }

        # Check indicator weights sum
        total_weight = sum(ind.base_weight for ind in config.indicators)
        if abs(total_weight - 1.0) > 0.1:
            validation_results['warnings'].append(
                f"Indicator weights sum to {total_weight:.3f}, not 1.0. Weights will be normalized."
            )

        # Check timeframe weights
        timeframe_weight = sum(tf.weight for tf in config.timeframes if tf.enabled)
        if timeframe_weight == 0:
            validation_results['errors'].append("No enabled timeframes found")

        # Check for duplicate indicator IDs
        indicator_ids = [ind.id for ind in config.indicators]
        if len(indicator_ids) != len(set(indicator_ids)):
            validation_results['errors'].append("Duplicate indicator IDs found")

        # Check performance settings
        if config.learning_rate <= 0 or config.learning_rate > 0.1:
            validation_results['warnings'].append(
                f"Learning rate {config.learning_rate} may be too extreme"
            )

        if config.performance_window < 50:
            validation_results['warnings'].append(
                f"Performance window {config.performance_window} may be too small"
            )

        return validation_results
