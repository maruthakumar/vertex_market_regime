#!/usr/bin/env python3
"""
TV Excel to YAML Converter
Converts TV strategy Excel configuration files to YAML format while preserving 6-file hierarchy
"""

import pandas as pd
import numpy as np
import yaml
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, date, time
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TVExcelToYAMLConverter:
    """Converts TV Excel configuration files to YAML format"""
    
    def __init__(self):
        """Initialize converter"""
        self.yaml_schema = self._define_yaml_schema()
        
    def convert_tv_master_to_yaml(self, excel_path: str) -> Dict[str, Any]:
        """
        Convert TV master configuration Excel to YAML format
        
        Args:
            excel_path: Path to TV master Excel file
            
        Returns:
            Dictionary in YAML-ready format
        """
        try:
            # Read Excel file
            df = pd.read_excel(excel_path, sheet_name='Setting')
            
            if len(df) == 0:
                raise ValueError("TV master config is empty")
            
            # Get first row (assuming single configuration)
            config_row = df.iloc[0]
            
            # Convert to YAML structure
            yaml_config = {
                'tv_configuration': {
                    'metadata': {
                        'conversion_timestamp': datetime.now().isoformat(),
                        'source_file': Path(excel_path).name,
                        'converter_version': '1.0.0'
                    },
                    'master_settings': {
                        'name': str(config_row['Name']),
                        'enabled': self._convert_boolean(config_row['Enabled']),
                        'date_range': {
                            'start': self._convert_date_to_string(config_row['StartDate']),
                            'end': self._convert_date_to_string(config_row['EndDate'])
                        }
                    },
                    'signal_configuration': {
                        'file_path': str(config_row['SignalFilePath']),
                        'date_format': str(config_row['SignalDateFormat']),
                        'time_adjustments': {
                            'entry_offset_seconds': int(config_row.get('IncreaseEntrySignalTimeBy', 0)),
                            'exit_offset_seconds': int(config_row.get('IncreaseExitSignalTimeBy', 0)),
                            'first_trade_entry_time': self._convert_time_to_string(config_row.get('FirstTradeEntryTime', 0))
                        }
                    },
                    'portfolio_mappings': {
                        'long': str(config_row['LongPortfolioFilePath']),
                        'short': str(config_row['ShortPortfolioFilePath']),
                        'manual': str(config_row['ManualPortfolioFilePath'])
                    },
                    'execution_settings': {
                        'intraday_squareoff': self._convert_boolean(config_row['IntradaySqOffApplicable']),
                        'exit_time': self._convert_time_to_string(config_row['IntradayExitTime']),
                        'tv_exit_applicable': self._convert_boolean(config_row['TvExitApplicable']),
                        'slippage_percent': float(config_row['SlippagePercent']),
                        'use_db_exit_timing': self._convert_boolean(config_row['UseDbExitTiming']),
                        'exit_search_interval': int(config_row['ExitSearchInterval']),
                        'exit_price_source': str(config_row['ExitPriceSource'])
                    },
                    'rollover_settings': {
                        'do_rollover': self._convert_boolean(config_row['DoRollover']),
                        'rollover_time': self._convert_time_to_string(config_row.get('RolloverTime', 0)),
                        'expiry_day_exit_time': self._convert_time_to_string(config_row.get('ExpiryDayExitTime', 0))
                    },
                    'manual_trade_settings': {
                        'manual_trade_entry_time': self._convert_time_to_string(config_row.get('ManualTradeEntryTime', 0)),
                        'manual_trade_lots': int(config_row.get('ManualTradeLots', 0))
                    },
                    'timing_settings': {
                        'entry_offset_seconds': int(config_row.get('IncreaseEntrySignalTimeBy', 0)),
                        'exit_offset_seconds': int(config_row.get('IncreaseExitSignalTimeBy', 0))
                    }
                }
            }
            
            return yaml_config
            
        except Exception as e:
            logger.error(f"Failed to convert TV master config to YAML: {e}")
            raise
    
    def convert_signals_to_yaml(self, excel_path: str) -> Dict[str, Any]:
        """
        Convert TV signals Excel to YAML format
        
        Args:
            excel_path: Path to signals Excel file
            
        Returns:
            Dictionary in YAML-ready format
        """
        try:
            # Read signals file
            df = pd.read_excel(excel_path, sheet_name='List of trades')
            
            signals_list = []
            
            for _, row in df.iterrows():
                signal = {
                    'trade_no': str(row['Trade #']),
                    'type': str(row['Type']),
                    'datetime': self._convert_datetime_to_string(row['Date/Time']),
                    'contracts': int(row['Contracts'])
                }
                signals_list.append(signal)
            
            yaml_config = {
                'signals': signals_list,
                'metadata': {
                    'total_signals': len(signals_list),
                    'trade_count': len(set(signal['trade_no'] for signal in signals_list)),
                    'conversion_timestamp': datetime.now().isoformat(),
                    'source_file': Path(excel_path).name
                }
            }
            
            return yaml_config
            
        except Exception as e:
            logger.error(f"Failed to convert signals to YAML: {e}")
            raise
    
    def convert_portfolio_to_yaml(self, excel_path: str) -> Dict[str, Any]:
        """
        Convert portfolio Excel to YAML format
        
        Args:
            excel_path: Path to portfolio Excel file
            
        Returns:
            Dictionary in YAML-ready format
        """
        try:
            # Read portfolio settings
            df_portfolio = pd.read_excel(excel_path, sheet_name='PortfolioSetting')
            df_strategy = pd.read_excel(excel_path, sheet_name='StrategySetting')
            
            # Convert portfolio settings
            portfolio_row = df_portfolio.iloc[0]
            portfolio_settings = {
                'capital': int(portfolio_row['Capital']),
                'max_risk': int(portfolio_row['MaxRisk']),
                'max_positions': int(portfolio_row['MaxPositions']),
                'risk_per_trade': int(portfolio_row['RiskPerTrade']),
                'use_kelly_criterion': self._convert_boolean(portfolio_row['UseKellyCriterion']),
                'rebalance_frequency': str(portfolio_row['RebalanceFrequency'])
            }
            
            # Convert strategy settings
            strategy_settings = []
            for _, row in df_strategy.iterrows():
                strategy = {
                    'strategy_name': str(row['StrategyName']),
                    'strategy_file': str(row['StrategyExcelFilePath']),
                    'enabled': self._convert_boolean(row['Enabled']),
                    'priority': int(row['Priority']),
                    'allocation_percent': int(row['AllocationPercent'])
                }
                strategy_settings.append(strategy)
            
            yaml_config = {
                'portfolio_configuration': {
                    'metadata': {
                        'conversion_timestamp': datetime.now().isoformat(),
                        'source_file': Path(excel_path).name
                    },
                    'portfolio_settings': portfolio_settings,
                    'strategy_settings': strategy_settings
                }
            }
            
            return yaml_config
            
        except Exception as e:
            logger.error(f"Failed to convert portfolio to YAML: {e}")
            raise
    
    def convert_tbs_strategy_to_yaml(self, excel_path: str) -> Dict[str, Any]:
        """
        Convert TBS strategy Excel to YAML format
        
        Args:
            excel_path: Path to TBS strategy Excel file
            
        Returns:
            Dictionary in YAML-ready format
        """
        try:
            # Read general and leg parameters
            df_general = pd.read_excel(excel_path, sheet_name='GeneralParameter')
            df_legs = pd.read_excel(excel_path, sheet_name='LegParameter')
            
            # Convert general parameters
            general_row = df_general.iloc[0]
            general_params = {
                'strategy_name': str(general_row['StrategyName']),
                'underlying': str(general_row['Underlying']),
                'index': str(general_row['Index']),
                'weekdays': self._convert_weekdays(general_row['Weekdays']),
                'dte': int(general_row['DTE']),
                'strike_selection_time': self._convert_time_to_string(general_row['StrikeSelectionTime']),
                'start_time': self._convert_time_to_string(general_row['StartTime']),
                'end_time': self._convert_time_to_string(general_row['EndTime']),
                'strategy_profit': float(general_row.get('StrategyProfit', 0)),
                'strategy_loss': float(general_row.get('StrategyLoss', 0)),
                'move_sl_to_cost': self._convert_boolean(general_row.get('MoveSlToCost', 'NO'))
            }
            
            # Convert leg parameters
            leg_params = []
            for _, row in df_legs.iterrows():
                leg = {
                    'leg_id': str(row['LegID']),
                    'instrument': str(row['Instrument']),
                    'transaction': str(row['Transaction']),
                    'expiry': str(row['Expiry']),
                    'strike_method': str(row['StrikeMethod']),
                    'strike_value': float(row['StrikeValue']),
                    'sl_type': str(row['SLType']),
                    'sl_value': float(row['SLValue']),
                    'tgt_type': str(row.get('TGTType', '')),
                    'tgt_value': float(row.get('TGTValue', 0)),
                    'lots': int(row['Lots']),
                    'enabled': not self._convert_boolean(row.get('IsIdle', 'no'))
                }
                leg_params.append(leg)
            
            yaml_config = {
                'tbs_strategy_configuration': {
                    'metadata': {
                        'conversion_timestamp': datetime.now().isoformat(),
                        'source_file': Path(excel_path).name
                    },
                    'general_parameters': general_params,
                    'leg_parameters': leg_params
                }
            }
            
            return yaml_config
            
        except Exception as e:
            logger.error(f"Failed to convert TBS strategy to YAML: {e}")
            raise
    
    def convert_complete_hierarchy_to_yaml(self, config_files: Dict[str, Path]) -> Dict[str, Any]:
        """
        Convert complete 6-file hierarchy to unified YAML
        
        Args:
            config_files: Dictionary with file paths for all 6 files
            
        Returns:
            Complete unified YAML configuration
        """
        try:
            # Convert each component
            tv_master = self.convert_tv_master_to_yaml(str(config_files['tv_master']))
            signals = self.convert_signals_to_yaml(str(config_files['signals']))
            portfolio_long = self.convert_portfolio_to_yaml(str(config_files['portfolio_long']))
            portfolio_short = self.convert_portfolio_to_yaml(str(config_files['portfolio_short']))
            portfolio_manual = self.convert_portfolio_to_yaml(str(config_files['portfolio_manual']))
            tbs_strategy = self.convert_tbs_strategy_to_yaml(str(config_files['strategy']))
            
            # Combine into unified structure
            unified_config = {
                'tv_complete_configuration': {
                    'metadata': {
                        'conversion_timestamp': datetime.now().isoformat(),
                        'converter_version': '1.0.0',
                        'description': 'Complete TV strategy configuration with 6-file hierarchy',
                        'source_files': {
                            'tv_master': config_files['tv_master'].name,
                            'signals': config_files['signals'].name,
                            'portfolio_long': config_files['portfolio_long'].name,
                            'portfolio_short': config_files['portfolio_short'].name,
                            'portfolio_manual': config_files['portfolio_manual'].name,
                            'tbs_strategy': config_files['strategy'].name
                        }
                    },
                    'tv_master': tv_master['tv_configuration'],
                    'signals': signals,
                    'portfolio_long': portfolio_long['portfolio_configuration'],
                    'portfolio_short': portfolio_short['portfolio_configuration'],
                    'portfolio_manual': portfolio_manual['portfolio_configuration'],
                    'tbs_strategy': tbs_strategy['tbs_strategy_configuration']
                }
            }
            
            return unified_config
            
        except Exception as e:
            logger.error(f"Failed to convert complete hierarchy to YAML: {e}")
            raise
    
    def save_yaml_to_file(self, yaml_data: Dict[str, Any], output_path: str) -> None:
        """
        Save YAML data to file
        
        Args:
            yaml_data: YAML data dictionary
            output_path: Output file path
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(
                    yaml_data,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    indent=2,
                    sort_keys=False
                )
            logger.info(f"YAML saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save YAML to file: {e}")
            raise
    
    def get_yaml_schema(self) -> Dict[str, Any]:
        """Get the YAML schema definition"""
        return self.yaml_schema
    
    def _define_yaml_schema(self) -> Dict[str, Any]:
        """Define the YAML schema structure"""
        return {
            'tv_configuration': {
                'required': ['master_settings', 'signal_configuration', 'portfolio_mappings', 'execution_settings'],
                'master_settings': {
                    'required': ['name', 'enabled', 'date_range'],
                    'date_range': {
                        'required': ['start', 'end']
                    }
                },
                'signal_configuration': {
                    'required': ['file_path', 'date_format', 'time_adjustments']
                },
                'portfolio_mappings': {
                    'required': ['long', 'short', 'manual']
                },
                'execution_settings': {
                    'required': ['intraday_squareoff', 'exit_time', 'slippage_percent']
                }
            }
        }
    
    def _convert_boolean(self, value: Any) -> bool:
        """Convert Excel boolean values to Python boolean"""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.upper() in ['YES', 'TRUE', '1']
        if isinstance(value, (int, float)):
            return bool(value)
        return False
    
    def _convert_date_to_string(self, value: Any) -> str:
        """Convert Excel date to string format"""
        if isinstance(value, str):
            # Handle DD_MM_YYYY format
            if '_' in value:
                parts = value.split('_')
                if len(parts) == 3:
                    day, month, year = parts
                    return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            return value
        elif isinstance(value, datetime):
            return value.strftime('%Y-%m-%d')
        elif isinstance(value, date):
            return value.strftime('%Y-%m-%d')
        else:
            return str(value)
    
    def _convert_time_to_string(self, value: Any) -> str:
        """Convert Excel time to string format"""
        if isinstance(value, (int, float)) and value > 0:
            # Handle HHMMSS format (e.g., 91600 -> 09:16:00)
            time_str = str(int(value)).zfill(6)
            hours = time_str[:2]
            minutes = time_str[2:4]
            seconds = time_str[4:6]
            return f"{hours}:{minutes}:{seconds}"
        elif isinstance(value, time):
            return value.strftime('%H:%M:%S')
        elif isinstance(value, str):
            return value
        else:
            return '00:00:00'
    
    def _convert_datetime_to_string(self, value: Any) -> str:
        """Convert Excel datetime to string format"""
        if isinstance(value, str):
            # Handle YYYYMMDD HHMMSS format
            if ' ' in value:
                date_part, time_part = value.split(' ')
                if len(date_part) == 8 and len(time_part) == 6:
                    year = date_part[:4]
                    month = date_part[4:6]
                    day = date_part[6:8]
                    hour = time_part[:2]
                    minute = time_part[2:4]
                    second = time_part[4:6]
                    return f"{year}-{month}-{day} {hour}:{minute}:{second}"
            return value
        elif isinstance(value, datetime):
            return value.strftime('%Y-%m-%d %H:%M:%S')
        else:
            return str(value)
    
    def _convert_weekdays(self, value: Any) -> List[int]:
        """Convert weekdays string to list of integers"""
        if isinstance(value, str):
            # Handle '1,2,3,4,5' format
            if ',' in value:
                return [int(x.strip()) for x in value.split(',')]
            else:
                return [int(value)]
        elif isinstance(value, (int, float)):
            return [int(value)]
        else:
            return [1, 2, 3, 4, 5]  # Default to all weekdays


class TVYAMLValidator:
    """Validator for TV YAML configurations"""
    
    def __init__(self):
        """Initialize validator"""
        pass
    
    def validate_yaml_structure(self, yaml_data: Dict[str, Any]) -> bool:
        """
        Validate YAML structure against schema
        
        Args:
            yaml_data: YAML data dictionary
            
        Returns:
            True if valid, raises exception if invalid
        """
        if 'tv_configuration' not in yaml_data:
            raise ValueError("Missing 'tv_configuration' root element")
        
        tv_config = yaml_data['tv_configuration']
        
        # Validate required sections
        required_sections = [
            'master_settings', 'signal_configuration', 
            'portfolio_mappings', 'execution_settings'
        ]
        
        for section in required_sections:
            if section not in tv_config:
                raise ValueError(f"Missing required section: {section}")
        
        # Validate master settings
        master_settings = tv_config['master_settings']
        if not isinstance(master_settings.get('enabled'), bool):
            raise ValueError("master_settings.enabled must be boolean")
        
        # Validate date range
        date_range = master_settings.get('date_range', {})
        if 'start' not in date_range or 'end' not in date_range:
            raise ValueError("date_range must have start and end dates")
        
        return True
    
    def validate_file_references(self, yaml_data: Dict[str, Any], base_path: str) -> bool:
        """
        Validate that file references in YAML exist
        
        Args:
            yaml_data: YAML data dictionary
            base_path: Base directory path for relative file references
            
        Returns:
            True if all files exist, raises exception if missing
        """
        base_dir = Path(base_path)
        tv_config = yaml_data['tv_configuration']
        
        # Check signal file
        signal_file = tv_config['signal_configuration']['file_path']
        signal_path = base_dir / signal_file
        if not signal_path.exists():
            raise FileNotFoundError(f"Signal file not found: {signal_file}")
        
        # Check portfolio files
        portfolio_mappings = tv_config['portfolio_mappings']
        for portfolio_type, portfolio_file in portfolio_mappings.items():
            portfolio_path = base_dir / portfolio_file
            if not portfolio_path.exists():
                raise FileNotFoundError(f"{portfolio_type} portfolio file not found: {portfolio_file}")
        
        return True