"""
Indicator configuration parser

Handles the 4-sheet Excel configuration structure for Technical Indicator Strategy
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import logging

from .excel_parser import ExcelParser
from ..core.exceptions import ParsingError

logger = logging.getLogger(__name__)

class IndicatorParser(ExcelParser):
    """
    Specialized parser for Indicator configuration files
    
    This parser handles the 4-sheet structure of Indicator configurations:
    - IndicatorConfiguration: Indicator definitions and parameters
    - SignalConditions: Entry/exit logic and conditions
    - RiskManagement: Position sizing and risk parameters
    - TimeframeSettings: Multi-timeframe configuration
    """
    
    def __init__(self):
        """Initialize Indicator parser"""
        super().__init__()
        
        # Define expected sheet structure
        self.sheet_mapping = {
            'indicatorconfiguration': 'indicators',
            'signalconditions': 'signal_conditions',
            'riskmanagement': 'risk_management',
            'timeframesettings': 'timeframe_settings'
        }
        
        # Valid indicator types
        self.valid_indicators = {
            'TALIB': ['RSI', 'MACD', 'EMA', 'SMA', 'STOCH', 'CCI', 'WILLIAMS_R', 
                      'ADX', 'BBANDS', 'ATR', 'OBV', 'AD'],
            'SMC': ['BOS_DETECTION', 'CHOCH_DETECTION', 'ORDER_BLOCK', 
                    'FAIR_VALUE_GAP', 'LIQUIDITY_SWEEP'],
            'PATTERN': ['DOJI', 'HAMMER', 'ENGULFING', 'SHOOTING_STAR', 
                       'MORNING_STAR', 'EVENING_STAR']
        }
        
        # Valid signal types
        self.signal_types = [
            'OVERBOUGHT_OVERSOLD', 'CROSSOVER', 'PRICE_CROSSOVER', 
            'TREND_STRENGTH', 'BAND_SQUEEZE', 'VOLATILITY_FILTER', 
            'VOLUME_CONFIRMATION', 'STRUCTURE_BREAK', 'STRUCTURE_CHANGE', 
            'SUPPORT_RESISTANCE', 'REVERSAL_PATTERN'
        ]
        
        # Risk parameters
        self.risk_parameters = {
            'PositionSizingMethod': {
                'type': 'enum',
                'values': ['FIXED_AMOUNT', 'PERCENTAGE_RISK', 'ATR_BASED', 'VOLATILITY_ADJUSTED'],
                'default': 'FIXED_AMOUNT'
            },
            'FixedPositionSize': {
                'type': 'decimal',
                'min': 1000,
                'max': 10000000,
                'default': 100000
            },
            'RiskPercentage': {
                'type': 'decimal',
                'min': 0.1,
                'max': 10.0,
                'default': 2.0
            },
            'MaxPositions': {
                'type': 'integer',
                'min': 1,
                'max': 10,
                'default': 3
            },
            'StopLossMethod': {
                'type': 'enum',
                'values': ['ATR_BASED', 'PERCENTAGE', 'FIXED_POINTS', 'SWING_BASED'],
                'default': 'ATR_BASED'
            },
            'StopLossMultiplier': {
                'type': 'decimal',
                'min': 0.5,
                'max': 5.0,
                'default': 2.0
            },
            'TakeProfitMethod': {
                'type': 'enum',
                'values': ['ATR_BASED', 'RISK_REWARD', 'FIXED_POINTS', 'RESISTANCE_BASED'],
                'default': 'RISK_REWARD'
            },
            'RiskRewardRatio': {
                'type': 'decimal',
                'min': 0.5,
                'max': 10.0,
                'default': 2.0
            }
        }
    
    def parse(self, file_path: str) -> Dict[str, Any]:
        """
        Parse Indicator configuration file
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            Dictionary containing parsed configuration data
        """
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            raw_data = {}
            
            # Parse each sheet
            for sheet_name in excel_file.sheet_names:
                # Skip metadata sheets
                if sheet_name.lower() in ['metadata', 'readme', 'instructions']:
                    continue
                
                # Read sheet
                df = pd.read_excel(excel_file, sheet_name=sheet_name, header=0)
                
                # Skip empty sheets
                if df.empty:
                    continue
                
                # Parse sheet based on type
                normalized_name = self._normalize_sheet_name(sheet_name)
                
                if 'indicator' in normalized_name and 'configuration' in normalized_name:
                    sheet_data = self._parse_indicator_sheet(df)
                elif 'signal' in normalized_name:
                    sheet_data = self._parse_signal_sheet(df)
                elif 'risk' in normalized_name:
                    sheet_data = self._parse_risk_sheet(df)
                elif 'timeframe' in normalized_name:
                    sheet_data = self._parse_timeframe_sheet(df)
                else:
                    # Default parsing
                    sheet_data = self._parse_sheet(df, sheet_name)
                
                if sheet_data:
                    # Map to standard structure
                    for key, mapped in self.sheet_mapping.items():
                        if key in normalized_name:
                            raw_data[mapped] = sheet_data
                            break
                    else:
                        raw_data[normalized_name] = sheet_data
            
            # Validate the parsed data
            if not self._validate_indicator_structure(raw_data):
                raise ParsingError("Invalid Indicator configuration structure", 
                                 file_path=file_path, errors=self.errors)
            
            # Add metadata
            raw_data['_metadata'] = {
                'strategy_type': 'indicator',
                'version': '1.0',
                'sheet_count': len(raw_data) - 1
            }
            
            return raw_data
            
        except Exception as e:
            raise ParsingError(f"Failed to parse Indicator file: {str(e)}", 
                             file_path=file_path)
    
    def _parse_indicator_sheet(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Parse indicator configuration sheet"""
        indicators = []
        
        # Expected columns
        expected_cols = ['IndicatorName', 'IndicatorType', 'Category', 
                        'Timeframe', 'Period', 'Enabled', 'Weight', 
                        'Threshold_Upper', 'Threshold_Lower', 'Signal_Type']
        
        # Clean column names
        df.columns = [self._clean_column_name(col) for col in df.columns]
        
        for _, row in df.iterrows():
            indicator = {}
            
            # Map columns
            indicator['name'] = str(row.get('indicatorname', '')).strip()
            indicator['type'] = str(row.get('indicatortype', 'TALIB')).upper()
            indicator['category'] = str(row.get('category', 'MOMENTUM')).upper()
            indicator['timeframe'] = str(row.get('timeframe', '5m'))
            indicator['period'] = self._convert_to_int(row.get('period', 14))
            indicator['enabled'] = self._convert_to_bool(row.get('enabled', True))
            indicator['weight'] = self._convert_to_float(row.get('weight', 0.1))
            indicator['threshold_upper'] = self._convert_to_float(row.get('threshold_upper', 0))
            indicator['threshold_lower'] = self._convert_to_float(row.get('threshold_lower', 0))
            indicator['signal_type'] = str(row.get('signal_type', 'OVERBOUGHT_OVERSOLD')).upper()
            
            # Validate indicator
            if indicator['name'] and self._validate_indicator(indicator):
                indicators.append(indicator)
        
        return indicators
    
    def _parse_signal_sheet(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Parse signal conditions sheet"""
        conditions = []
        
        # Clean column names
        df.columns = [self._clean_column_name(col) for col in df.columns]
        
        for _, row in df.iterrows():
            condition = {}
            
            # Map columns
            condition['condition_id'] = str(row.get('conditionid', '')).strip()
            condition['condition_type'] = str(row.get('conditiontype', 'ENTRY')).upper()
            condition['logic_operator'] = str(row.get('logic', 'AND')).upper()
            condition['indicator1_name'] = str(row.get('indicator1', '')).strip()
            condition['operator1'] = str(row.get('operator1', 'GREATER_THAN')).upper()
            condition['value1'] = self._convert_to_float(row.get('value1', 0))
            condition['indicator2_name'] = str(row.get('indicator2', '')).strip() or None
            condition['operator2'] = str(row.get('operator2', '')).upper() or None
            condition['value2'] = self._convert_to_float(row.get('value2')) if row.get('value2') is not None else None
            condition['weight'] = self._convert_to_float(row.get('weight', 0.5))
            condition['enabled'] = self._convert_to_bool(row.get('enabled', True))
            
            # Generate ID if missing
            if not condition['condition_id']:
                condition['condition_id'] = f"{condition['condition_type']}_{len(conditions)+1:03d}"
            
            # Validate condition
            if condition['indicator1_name']:
                conditions.append(condition)
        
        return conditions
    
    def _parse_risk_sheet(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse risk management sheet"""
        risk_params = {}
        
        # Handle key-value format
        if self._is_key_value_sheet(df):
            for _, row in df.iterrows():
                param_name = str(row.iloc[0]).strip()
                param_value = row.iloc[1]
                
                if param_name in self.risk_parameters:
                    # Validate and convert based on type
                    param_config = self.risk_parameters[param_name]
                    
                    if param_config['type'] == 'enum':
                        value = str(param_value).upper()
                        if value in param_config['values']:
                            risk_params[self._normalize_key(param_name)] = value
                        else:
                            risk_params[self._normalize_key(param_name)] = param_config['default']
                    
                    elif param_config['type'] == 'decimal':
                        value = self._convert_to_float(param_value)
                        if param_config['min'] <= value <= param_config['max']:
                            risk_params[self._normalize_key(param_name)] = value
                        else:
                            risk_params[self._normalize_key(param_name)] = param_config['default']
                    
                    elif param_config['type'] == 'integer':
                        value = self._convert_to_int(param_value)
                        if param_config['min'] <= value <= param_config['max']:
                            risk_params[self._normalize_key(param_name)] = value
                        else:
                            risk_params[self._normalize_key(param_name)] = param_config['default']
        else:
            # Handle table format
            for _, row in df.iterrows():
                if 'Parameter' in df.columns and 'Value' in df.columns:
                    param_name = str(row['Parameter']).strip()
                    param_value = row['Value']
                    
                    if param_name in self.risk_parameters:
                        # Same validation logic as above
                        param_config = self.risk_parameters[param_name]
                        # ... (same validation code)
        
        return risk_params
    
    def _parse_timeframe_sheet(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse timeframe settings sheet"""
        timeframe_config = {
            'enabled_timeframes': [],
            'primary_timeframe': '5m',
            'timeframe_weights': {},
            'consensus_method': 'WEIGHTED_AVERAGE'
        }
        
        # Parse timeframe configuration
        for _, row in df.iterrows():
            if 'Timeframe' in df.columns:
                tf = str(row.get('Timeframe', '')).strip()
                enabled = self._convert_to_bool(row.get('Enabled', False))
                weight = self._convert_to_float(row.get('Weight', 0.0))
                
                if tf and enabled:
                    timeframe_config['enabled_timeframes'].append(tf)
                    timeframe_config['timeframe_weights'][tf] = weight
                    
                    if self._convert_to_bool(row.get('Primary', False)):
                        timeframe_config['primary_timeframe'] = tf
        
        return timeframe_config
    
    def _validate_indicator(self, indicator: Dict[str, Any]) -> bool:
        """Validate individual indicator configuration"""
        # Check indicator type
        if indicator['type'] not in self.valid_indicators:
            logger.warning(f"Unknown indicator type: {indicator['type']}")
            indicator['type'] = 'TALIB'  # Default
        
        # Check indicator name
        valid_names = self.valid_indicators.get(indicator['type'], [])
        if indicator['name'].upper() not in valid_names:
            logger.warning(f"Unknown {indicator['type']} indicator: {indicator['name']}")
            return False
        
        # Validate period
        if indicator['name'] in ['RSI', 'STOCH', 'CCI', 'WILLIAMS_R', 'ADX', 'ATR']:
            if not 2 <= indicator['period'] <= 100:
                logger.warning(f"Invalid period for {indicator['name']}: {indicator['period']}")
                indicator['period'] = 14  # Default
        
        # Validate thresholds
        if indicator['name'] == 'RSI':
            if not 0 <= indicator['threshold_lower'] <= indicator['threshold_upper'] <= 100:
                logger.warning(f"Invalid RSI thresholds: {indicator['threshold_lower']}-{indicator['threshold_upper']}")
                indicator['threshold_lower'] = 30
                indicator['threshold_upper'] = 70
        
        # Validate signal type
        if indicator['signal_type'] not in self.signal_types:
            logger.warning(f"Unknown signal type: {indicator['signal_type']}")
            indicator['signal_type'] = 'OVERBOUGHT_OVERSOLD'
        
        return True
    
    def _validate_indicator_structure(self, data: Dict[str, Any]) -> bool:
        """Validate complete indicator configuration structure"""
        # Check required sections
        required_sections = ['indicators', 'signal_conditions', 'risk_management']
        
        for section in required_sections:
            if section not in data:
                self.add_error(f"Missing required section: {section}")
                return False
        
        # Validate indicators
        indicators = data.get('indicators', [])
        if not indicators:
            self.add_error("No indicators configured")
            return False
        
        # Validate signal conditions
        conditions = data.get('signal_conditions', [])
        if not conditions:
            self.add_error("No signal conditions configured")
            return False
        
        # Cross-validate condition indicators
        indicator_names = [ind['name'] for ind in indicators if ind.get('enabled', True)]
        
        for condition in conditions:
            if condition['indicator1_name'] not in indicator_names:
                self.add_error(f"Signal condition references unknown indicator: {condition['indicator1_name']}")
                return False
            
            if condition.get('indicator2_name') and condition['indicator2_name'] not in indicator_names:
                self.add_error(f"Signal condition references unknown indicator: {condition['indicator2_name']}")
                return False
        
        # Validate weights
        total_indicator_weight = sum(ind.get('weight', 0) for ind in indicators if ind.get('enabled', True))
        if abs(total_indicator_weight - 1.0) > 0.1:  # Allow some tolerance
            logger.warning(f"Indicator weights sum to {total_indicator_weight}, expected ~1.0")
        
        return True
    
    def _convert_to_bool(self, value: Any) -> bool:
        """Convert various boolean representations to bool"""
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.upper() in ['YES', 'TRUE', '1', 'ON', 'ENABLED']
        return False
    
    def _convert_to_int(self, value: Any) -> int:
        """Convert to integer with validation"""
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return 0
    
    def _convert_to_float(self, value: Any) -> float:
        """Convert to float with validation"""
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def get_enabled_indicators(self, data: Dict[str, Any]) -> List[str]:
        """Get list of enabled indicator names"""
        indicators = data.get('indicators', [])
        return [ind['name'] for ind in indicators if ind.get('enabled', True)]
    
    def get_signal_conditions(self, data: Dict[str, Any], condition_type: str) -> List[Dict[str, Any]]:
        """Get signal conditions by type"""
        conditions = data.get('signal_conditions', [])
        return [c for c in conditions if c.get('condition_type') == condition_type and c.get('enabled', True)]
    
    def __repr__(self) -> str:
        return "IndicatorParser(sheets=4, indicators=12+, timeframes=6)"