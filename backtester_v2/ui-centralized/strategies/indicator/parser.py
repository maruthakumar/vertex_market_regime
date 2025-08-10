"""
Parser for Technical Indicator Strategy Excel files
Handles portfolio settings, indicator configurations, and signal settings
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, time, date
import logging
from pathlib import Path
import json

from .models import (
    IndicatorStrategyModel,
    IndicatorPortfolioModel,
    IndicatorConfig,
    SMCConfig,
    SignalCondition,
    RiskManagementConfig,
    ExecutionConfig,
    IndicatorType,
    TALibIndicator,
    SMCIndicator,
    ComparisonOperator,
    SignalLogic,
    Timeframe
)
from .constants import (
    TALIB_PARAMS,
    SMC_PARAMS,
    ML_MODEL_PARAMS,
    ERROR_MESSAGES,
    INDICATOR_CATEGORIES
)

logger = logging.getLogger(__name__)


class IndicatorParser:
    """Parser for Technical Indicator strategy Excel files"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        
    def parse_input(self,
                   portfolio_file: str,
                   indicator_file: str,
                   ml_model_file: Optional[str] = None,
                   signal_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse all input files for ML Indicator strategy
        
        Args:
            portfolio_file: Path to portfolio Excel file
            indicator_file: Path to indicator configuration Excel file
            ml_model_file: Optional path to ML model configuration
            signal_file: Optional path to signal conditions file
            
        Returns:
            Dictionary containing parsed strategy configuration
        """
        try:
            # Parse portfolio settings
            portfolio_data = self.parse_portfolio_excel(portfolio_file)
            
            # Parse indicator configurations
            indicator_data = self.parse_indicator_excel(indicator_file)
            
            # Parse ML model if provided
            ml_config = None
            if ml_model_file:
                ml_config = self.parse_ml_model_config(ml_model_file)
                
            # Parse signal conditions
            signal_data = {}
            if signal_file:
                signal_data = self.parse_signal_conditions(signal_file)
            elif 'signals' in indicator_data:
                signal_data = indicator_data['signals']
                
            # Combine all data
            result = {
                "portfolio": portfolio_data,
                "indicators": indicator_data.get("indicators", []),
                "smc_config": indicator_data.get("smc_config"),
                "ml_config": ml_config,
                "signals": signal_data,
                "errors": self.errors,
                "warnings": self.warnings
            }
            
            # Create and validate complete model
            if not self.errors:
                strategy_model = self._create_strategy_model(result)
                result["model"] = strategy_model
                
            return result
            
        except Exception as e:
            logger.error(f"Error parsing ML Indicator input files: {str(e)}")
            self.errors.append(f"Parser error: {str(e)}")
            return {
                "errors": self.errors,
                "warnings": self.warnings
            }
    
    def parse_portfolio_excel(self, excel_path: str) -> Dict[str, Any]:
        """Parse portfolio settings from Excel"""
        try:
            df = pd.read_excel(excel_path, sheet_name='PortfolioSetting')
            
            portfolio_data = {}
            
            # Parse basic settings
            for idx, row in df.iterrows():
                if pd.notna(row.get('Parameter')) and pd.notna(row.get('Value')):
                    param = str(row['Parameter']).strip()
                    value = row['Value']
                    
                    if param == 'PortfolioName':
                        portfolio_data['portfolio_name'] = str(value)
                    elif param == 'StrategyName':
                        portfolio_data['strategy_name'] = str(value)
                    elif param == 'StartDate':
                        portfolio_data['start_date'] = pd.to_datetime(value).date()
                    elif param == 'EndDate':
                        portfolio_data['end_date'] = pd.to_datetime(value).date()
                    elif param == 'IndexName':
                        portfolio_data['index_name'] = str(value).upper()
                    elif param == 'UnderlyingPriceType':
                        portfolio_data['underlying_price_type'] = str(value).upper()
                    elif param == 'TransactionCosts':
                        portfolio_data['transaction_costs'] = float(value)
                    elif param == 'UseWalkForward':
                        portfolio_data['use_walk_forward'] = bool(value)
                    elif param == 'WalkForwardWindow':
                        portfolio_data['walk_forward_window'] = int(value)
                    elif param == 'TrackFeatureImportance':
                        portfolio_data['track_feature_importance'] = bool(value)
                    elif param == 'TrackSignalAccuracy':
                        portfolio_data['track_signal_accuracy'] = bool(value)
                    elif param == 'SavePredictions':
                        portfolio_data['save_predictions'] = bool(value)
            
            # Parse risk management settings
            portfolio_data['risk_config'] = self._parse_risk_config(df)
            
            # Parse execution settings
            portfolio_data['execution_config'] = self._parse_execution_config(df)
            
            # Validate required fields
            required_fields = ['portfolio_name', 'strategy_name', 'start_date', 'end_date']
            for field in required_fields:
                if field not in portfolio_data:
                    self.errors.append(f"Missing required portfolio field: {field}")
                    
            return portfolio_data
            
        except Exception as e:
            logger.error(f"Error parsing portfolio Excel: {str(e)}")
            self.errors.append(f"Portfolio parsing error: {str(e)}")
            return {}
    
    def parse_indicator_excel(self, excel_path: str) -> Dict[str, Any]:
        """Parse indicator configurations from Excel"""
        try:
            excel_file = pd.ExcelFile(excel_path)
            result = {}
            
            # Parse indicator settings
            if 'Indicators' in excel_file.sheet_names:
                indicators_df = pd.read_excel(excel_path, sheet_name='Indicators')
                result['indicators'] = self._parse_indicators(indicators_df)
                
            # Parse SMC configuration
            if 'SMC' in excel_file.sheet_names:
                smc_df = pd.read_excel(excel_path, sheet_name='SMC')
                result['smc_config'] = self._parse_smc_config(smc_df)
                
            # Parse signal conditions
            if 'Signals' in excel_file.sheet_names:
                signals_df = pd.read_excel(excel_path, sheet_name='Signals')
                result['signals'] = self._parse_signals(signals_df)
                
            # Parse multi-leg configuration if present
            if 'Legs' in excel_file.sheet_names:
                legs_df = pd.read_excel(excel_path, sheet_name='Legs')
                result['legs'] = self._parse_legs(legs_df)
                
            # Validate at least one indicator is configured
            if not result.get('indicators') and not result.get('smc_config'):
                self.errors.append("No indicators configured")
                
            return result
            
        except Exception as e:
            logger.error(f"Error parsing indicator Excel: {str(e)}")
            self.errors.append(f"Indicator parsing error: {str(e)}")
            return {}
    
    def _parse_indicators(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Parse individual indicator configurations"""
        indicators = []
        
        for idx, row in df.iterrows():
            if pd.notna(row.get('IndicatorName')):
                try:
                    indicator_name = str(row['IndicatorName']).upper()
                    indicator_type = self._determine_indicator_type(indicator_name)
                    
                    # Parse parameters
                    params = {}
                    if pd.notna(row.get('Parameters')):
                        # Try to parse as JSON first
                        try:
                            params = json.loads(row['Parameters'])
                        except:
                            # Parse as key=value pairs
                            param_str = str(row['Parameters'])
                            for pair in param_str.split(','):
                                if '=' in pair:
                                    key, value = pair.split('=')
                                    params[key.strip()] = self._parse_param_value(value.strip())
                    else:
                        # Use default parameters if available
                        if indicator_name in TALIB_PARAMS:
                            params = TALIB_PARAMS[indicator_name].copy()
                    
                    # Create indicator config
                    config = {
                        "indicator_name": indicator_name,
                        "indicator_type": indicator_type,
                        "parameters": params,
                        "timeframe": Timeframe(row.get('Timeframe', '5m')),
                        "lookback_period": int(row.get('LookbackPeriod', 
                                               params.get('timeperiod', 20))),
                        "normalization": row.get('Normalization') if pd.notna(row.get('Normalization')) else None,
                        "cache_enabled": bool(row.get('CacheEnabled', True))
                    }
                    
                    indicators.append(config)
                    
                except Exception as e:
                    logger.error(f"Error parsing indicator row {idx}: {str(e)}")
                    self.warnings.append(f"Indicator {idx} parsing warning: {str(e)}")
                    
        return indicators
    
    def _determine_indicator_type(self, indicator_name: str) -> IndicatorType:
        """Determine indicator type from name"""
        indicator_name = indicator_name.upper()
        
        # Check if it's a TA-Lib indicator
        try:
            TALibIndicator(indicator_name)
            return IndicatorType.TALIB
        except:
            pass
            
        # Check if it's an SMC indicator
        try:
            SMCIndicator(indicator_name)
            return IndicatorType.SMC
        except:
            pass
            
        # Check if it's a candlestick pattern
        if indicator_name.startswith('CDL'):
            return IndicatorType.CANDLESTICK
            
        # Check if it's a volume indicator
        if indicator_name in ['VOLUME_PROFILE', 'CVD', 'DELTA']:
            return IndicatorType.VOLUME
            
        # Default to custom
        return IndicatorType.CUSTOM
    
    def _parse_param_value(self, value: str) -> Any:
        """Parse parameter value to appropriate type"""
        value = value.strip()
        
        # Try to parse as number
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except:
            pass
            
        # Try to parse as boolean
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
            
        # Return as string
        return value
    
    def _parse_smc_config(self, df: pd.DataFrame) -> SMCConfig:
        """Parse Smart Money Concepts configuration"""
        config = {}
        
        for idx, row in df.iterrows():
            if pd.notna(row.get('Setting')) and pd.notna(row.get('Value')):
                setting = str(row['Setting']).strip()
                value = row['Value']
                
                if setting == 'DetectBOS':
                    config['detect_bos'] = bool(value)
                elif setting == 'DetectCHOCH':
                    config['detect_choch'] = bool(value)
                elif setting == 'DetectOrderBlocks':
                    config['detect_order_blocks'] = bool(value)
                elif setting == 'DetectFVG':
                    config['detect_fvg'] = bool(value)
                elif setting == 'DetectLiquidity':
                    config['detect_liquidity'] = bool(value)
                elif setting == 'UseKillZones':
                    config['use_kill_zones'] = bool(value)
                elif setting == 'StructureLookback':
                    config['structure_lookback'] = int(value)
                elif setting == 'OrderBlockLookback':
                    config['order_block_lookback'] = int(value)
                elif setting == 'FVGMinSize':
                    config['fvg_min_size'] = float(value)
                elif setting == 'LiquidityThreshold':
                    config['liquidity_threshold'] = float(value)
                    
        return SMCConfig(**config)
    
    def _parse_signals(self, df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        """Parse signal conditions"""
        entry_signals = []
        exit_signals = []
        signal_logic = SignalLogic.AND
        
        for idx, row in df.iterrows():
            if pd.notna(row.get('SignalType')) and pd.notna(row.get('IndicatorName')):
                try:
                    signal = {
                        "indicator_name": str(row['IndicatorName']),
                        "condition_type": ComparisonOperator(row['Condition']),
                        "threshold_value": float(row['ThresholdValue']) if pd.notna(row.get('ThresholdValue')) else None,
                        "threshold_indicator": row.get('ThresholdIndicator') if pd.notna(row.get('ThresholdIndicator')) else None,
                        "secondary_value": float(row['SecondaryValue']) if pd.notna(row.get('SecondaryValue')) else None,
                        "weight": float(row.get('Weight', 1.0)),
                        "enabled": bool(row.get('Enabled', True))
                    }
                    
                    signal_type = str(row['SignalType']).upper()
                    if signal_type == 'ENTRY':
                        entry_signals.append(signal)
                    elif signal_type == 'EXIT':
                        exit_signals.append(signal)
                        
                except Exception as e:
                    logger.error(f"Error parsing signal row {idx}: {str(e)}")
                    self.warnings.append(f"Signal {idx} parsing warning: {str(e)}")
        
        # Parse signal logic if present
        logic_mask = df['SignalType'].str.contains('LOGIC', case=False, na=False)
        if logic_mask.any():
            signal_logic = SignalLogic(df.loc[logic_mask, 'IndicatorName'].iloc[0])
            
        return {
            "entry_signals": entry_signals,
            "exit_signals": exit_signals,
            "signal_logic": signal_logic
        }
    
    def _parse_legs(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Parse multi-leg configuration"""
        legs = []
        
        for idx, row in df.iterrows():
            if pd.notna(row.get('LegID')):
                try:
                    leg = {
                        "leg_id": int(row['LegID']),
                        "leg_name": str(row.get('LegName', f'Leg_{row["LegID"]}')),
                        "option_type": str(row['OptionType']).upper(),
                        "position_type": str(row['PositionType']).upper(),
                        "strike_selection": str(row.get('StrikeSelection', 'ATM')).upper(),
                        "strike_offset": float(row['StrikeOffset']) if pd.notna(row.get('StrikeOffset')) else None,
                        "lots": int(row.get('Lots', 1)),
                        "lot_size": int(row.get('LotSize', 50)),
                        "entry_signal_group": str(row.get('EntrySignalGroup', 'default')),
                        "exit_signal_group": str(row.get('ExitSignalGroup', 'default')),
                        "stop_loss": float(row['StopLoss']) if pd.notna(row.get('StopLoss')) else None,
                        "take_profit": float(row['TakeProfit']) if pd.notna(row.get('TakeProfit')) else None,
                        "min_signal_confidence": float(row.get('MinConfidence', 0.6)),
                        "use_ml_exit": bool(row.get('UseMLExit', True))
                    }
                    
                    # Parse times
                    if pd.notna(row.get('EntryTimeStart')):
                        leg['entry_time_start'] = self._parse_time(row['EntryTimeStart'])
                    if pd.notna(row.get('EntryTimeEnd')):
                        leg['entry_time_end'] = self._parse_time(row['EntryTimeEnd'])
                    if pd.notna(row.get('ExitTime')):
                        leg['exit_time'] = self._parse_time(row['ExitTime'])
                        
                    legs.append(leg)
                    
                except Exception as e:
                    logger.error(f"Error parsing leg row {idx}: {str(e)}")
                    self.warnings.append(f"Leg {idx} parsing warning: {str(e)}")
                    
        return legs
    
    def _parse_risk_config(self, df: pd.DataFrame) -> RiskManagementConfig:
        """Parse risk management configuration"""
        config = {}
        
        risk_params = {
            'PositionSizing': 'position_sizing',
            'MaxPositionSize': 'max_position_size',
            'MaxPortfolioRisk': 'max_portfolio_risk',
            'StopLossType': 'stop_loss_type',
            'StopLossValue': 'stop_loss_value',
            'TakeProfitType': 'take_profit_type',
            'TakeProfitValue': 'take_profit_value',
            'UseTrailingStop': 'use_trailing_stop',
            'TrailingStopActivation': 'trailing_stop_activation',
            'TrailingStopDistance': 'trailing_stop_distance',
            'MaxConcurrentPositions': 'max_concurrent_positions',
            'MaxCorrelation': 'max_correlation',
            'MaxSectorExposure': 'max_sector_exposure'
        }
        
        for excel_param, model_param in risk_params.items():
            value = self._get_parameter_value(df, excel_param)
            if value is not None:
                if model_param in ['position_sizing', 'stop_loss_type', 'take_profit_type']:
                    config[model_param] = str(value)
                elif model_param in ['use_trailing_stop']:
                    config[model_param] = bool(value)
                elif model_param in ['max_concurrent_positions']:
                    config[model_param] = int(value)
                else:
                    config[model_param] = float(value)
                    
        return RiskManagementConfig(**config)
    
    def _parse_execution_config(self, df: pd.DataFrame) -> ExecutionConfig:
        """Parse execution configuration"""
        config = {}
        
        exec_params = {
            'ExecutionMode': 'execution_mode',
            'SlippageModel': 'slippage_model',
            'SlippageValue': 'slippage_value',
            'UseIcebergOrders': 'use_iceberg_orders',
            'IcebergDisplaySize': 'iceberg_display_size',
            'UseVWAPExecution': 'use_vwap_execution',
            'VWAPParticipationRate': 'vwap_participation_rate',
            'AvoidFirstMinutes': 'avoid_first_minutes',
            'AvoidLastMinutes': 'avoid_last_minutes'
        }
        
        for excel_param, model_param in exec_params.items():
            value = self._get_parameter_value(df, excel_param)
            if value is not None:
                if model_param in ['execution_mode', 'slippage_model']:
                    config[model_param] = str(value)
                elif model_param in ['use_iceberg_orders', 'use_vwap_execution']:
                    config[model_param] = bool(value)
                elif model_param in ['avoid_first_minutes', 'avoid_last_minutes']:
                    config[model_param] = int(value)
                else:
                    config[model_param] = float(value)
                    
        return ExecutionConfig(**config)
    
    def _get_parameter_value(self, df: pd.DataFrame, parameter: str) -> Any:
        """Get parameter value from dataframe"""
        mask = df['Parameter'].str.contains(parameter, case=False, na=False)
        if mask.any():
            return df.loc[mask, 'Value'].iloc[0]
        return None
    
    def _parse_time(self, time_value: Any) -> time:
        """Parse time from various formats"""
        if isinstance(time_value, time):
            return time_value
        elif isinstance(time_value, str):
            try:
                return datetime.strptime(time_value, "%H:%M:%S").time()
            except:
                try:
                    return datetime.strptime(time_value, "%H:%M").time()
                except:
                    logger.warning(f"Invalid time format: {time_value}, using default")
                    return time(9, 20)
        else:
            return time(9, 20)

    def parse_indicator_config(self, config_path: str) -> Dict[str, Any]:
        """Parse additional indicator configuration from file"""
        try:
            if config_path.endswith('.json'):
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
            elif config_path.endswith('.xlsx'):
                df = pd.read_excel(config_path, sheet_name='IndicatorConfig')
                config_data = {}
                for _, row in df.iterrows():
                    if pd.notna(row.get('Parameter')) and pd.notna(row.get('Value')):
                        config_data[row['Parameter']] = row['Value']
            else:
                raise ValueError(f"Unsupported indicator config format: {config_path}")

            return config_data

        except Exception as e:
            logger.error(f"Error parsing indicator config: {str(e)}")
            self.warnings.append(f"Indicator config parsing warning: {str(e)}")
            return {}
    
    def parse_signal_conditions(self, signal_path: str) -> Dict[str, List[SignalCondition]]:
        """Parse signal conditions from separate file"""
        try:
            df = pd.read_excel(signal_path, sheet_name='SignalConditions')
            return self._parse_signals(df)
            
        except Exception as e:
            logger.error(f"Error parsing signal conditions: {str(e)}")
            self.warnings.append(f"Signal parsing warning: {str(e)}")
            return {"entry_signals": [], "exit_signals": []}
    
    def _create_strategy_model(self, parsed_data: Dict[str, Any]) -> IndicatorStrategyModel:
        """Create validated strategy model from parsed data"""
        # Create portfolio model
        portfolio = IndicatorPortfolioModel(**parsed_data['portfolio'])

        # Create indicator configs
        indicators = []
        for ind_data in parsed_data.get('indicators', []):
            indicators.append(IndicatorConfig(**ind_data))

        # Create signal conditions
        entry_signals = []
        for sig_data in parsed_data.get('signals', {}).get('entry_signals', []):
            entry_signals.append(SignalCondition(**sig_data))

        exit_signals = []
        for sig_data in parsed_data.get('signals', {}).get('exit_signals', []):
            exit_signals.append(SignalCondition(**sig_data))

        # Create strategy model
        strategy = IndicatorStrategyModel(
            portfolio=portfolio,
            indicators=indicators,
            smc_config=parsed_data.get('smc_config'),
            entry_signals=entry_signals,
            exit_signals=exit_signals,
            signal_logic=parsed_data.get('signals', {}).get('signal_logic', SignalLogic.AND)
        )

        return strategy
    
    def validate_parsed_data(self, parsed_data: Dict[str, Any]) -> List[str]:
        """Validate parsed data for consistency and completeness"""
        validation_errors = []
        
        # Validate date range
        if parsed_data['portfolio'].get('start_date') >= parsed_data['portfolio'].get('end_date'):
            validation_errors.append("End date must be after start date")
            
        # Validate indicators
        if not parsed_data.get('indicators') and not parsed_data.get('smc_config'):
            validation_errors.append("At least one indicator or SMC must be configured")
            
        # Validate signals
        if not parsed_data.get('signals', {}).get('entry_signals'):
            validation_errors.append("At least one entry signal required")
            
        # Validate ML config if ML-based signals
        signal_logic = parsed_data.get('signals', {}).get('signal_logic')
        if signal_logic == SignalLogic.ML_BASED and not parsed_data.get('ml_config'):
            validation_errors.append("ML configuration required for ML_BASED signal logic")
            
        # Validate indicator references in signals
        configured_indicators = set(ind['indicator_name'] for ind in parsed_data.get('indicators', []))
        for signal in parsed_data.get('signals', {}).get('entry_signals', []):
            if signal['indicator_name'] not in configured_indicators:
                validation_errors.append(f"Signal references undefined indicator: {signal['indicator_name']}")
                
        return validation_errors


    def parse_6_sheet_template(self, input_file: str) -> Dict[str, Any]:
        """
        Parse 6-sheet Technical Indicator strategy template

        Args:
            input_file: Path to Excel file with 6-sheet template

        Returns:
            Parsed configuration dictionary
        """
        try:
            self.errors = []
            self.warnings = []

            # Read Excel file
            excel_data = pd.ExcelFile(input_file)

            # Validate sheet structure
            required_sheets = [
                'IndicatorConfiguration', 'SignalConditions', 'RiskManagement',
                'TimeframeSettings', 'PortfolioSettings', 'StrategySettings'
            ]

            missing_sheets = [sheet for sheet in required_sheets if sheet not in excel_data.sheet_names]
            if missing_sheets:
                raise ValueError(f"Missing required sheets: {missing_sheets}")

            # Parse each sheet with comprehensive validation
            parsed_data = {
                'indicator_configuration': self._parse_indicator_configuration_sheet(excel_data),
                'signal_conditions': self._parse_signal_conditions_sheet(excel_data),
                'risk_management': self._parse_risk_management_sheet(excel_data),
                'timeframe_settings': self._parse_timeframe_settings_sheet(excel_data),
                'portfolio_settings': self._parse_portfolio_settings_sheet(excel_data),
                'strategy_settings': self._parse_strategy_settings_sheet(excel_data)
            }

            # Cross-sheet validation
            self._validate_cross_sheet_references(parsed_data)

            return parsed_data

        except Exception as e:
            logger.error(f"Error parsing 6-sheet template: {str(e)}")
            self.errors.append(f"Failed to parse 6-sheet template: {str(e)}")
            raise ValueError(f"Failed to parse 6-sheet template: {str(e)}")

    def _parse_indicator_configuration_sheet(self, excel_data: pd.ExcelFile) -> List[Dict[str, Any]]:
        """Parse IndicatorConfiguration sheet"""
        try:
            df = pd.read_excel(excel_data, sheet_name='IndicatorConfiguration')

            indicators = []
            for _, row in df.iterrows():
                if pd.isna(row.get('IndicatorName')):
                    continue

                indicator = {
                    'indicator_name': str(row['IndicatorName']).strip(),
                    'indicator_type': self._validate_enum(row.get('IndicatorType', 'TALIB'),
                                                         ['TALIB', 'SMC', 'PATTERN', 'CUSTOM'], 'TALIB'),
                    'category': self._validate_enum(row.get('Category', 'MOMENTUM'),
                                                   ['MOMENTUM', 'TREND', 'VOLATILITY', 'VOLUME', 'STRUCTURE', 'CANDLESTICK'], 'MOMENTUM'),
                    'timeframe': self._validate_enum(row.get('Timeframe', '5m'),
                                                    ['1m', '5m', '15m', '1h', '4h', '1d'], '5m'),
                    'period': self._validate_integer(row.get('Period', 14), 1, 200, 14),
                    'enabled': self._validate_boolean(row.get('Enabled', 'YES')),
                    'weight': self._validate_decimal(row.get('Weight', 0.1), 0.0001, 1.0, 0.1),
                    'threshold_upper': self._validate_decimal(row.get('Threshold_Upper', 0), -999999, 999999, 0),
                    'threshold_lower': self._validate_decimal(row.get('Threshold_Lower', 0), -999999, 999999, 0),
                    'signal_type': str(row.get('Signal_Type', 'OVERBOUGHT_OVERSOLD')).strip()
                }

                # Validate indicator-specific parameters
                self._validate_indicator_parameters(indicator)
                indicators.append(indicator)

            return indicators

        except Exception as e:
            self.errors.append(f"Error parsing IndicatorConfiguration sheet: {str(e)}")
            return []

    def _parse_signal_conditions_sheet(self, excel_data: pd.ExcelFile) -> List[Dict[str, Any]]:
        """Parse SignalConditions sheet"""
        try:
            df = pd.read_excel(excel_data, sheet_name='SignalConditions')

            conditions = []
            for _, row in df.iterrows():
                if pd.isna(row.get('ConditionID')):
                    continue

                condition = {
                    'condition_id': str(row['ConditionID']).strip(),
                    'condition_type': self._validate_enum(row.get('ConditionType', 'ENTRY'),
                                                         ['ENTRY', 'EXIT', 'STOP_LOSS', 'TAKE_PROFIT'], 'ENTRY'),
                    'logic_operator': self._validate_enum(row.get('Logic', 'AND'),
                                                         ['AND', 'OR', 'NOT'], 'AND'),
                    'indicator1_name': str(row.get('Indicator1', '')).strip(),
                    'operator1': str(row.get('Operator1', 'GREATER_THAN')).strip(),
                    'value1': self._validate_decimal(row.get('Value1', 0), -999999, 999999, 0),
                    'indicator2_name': str(row.get('Indicator2', '')).strip() if pd.notna(row.get('Indicator2')) else None,
                    'operator2': str(row.get('Operator2', '')).strip() if pd.notna(row.get('Operator2')) else None,
                    'value2': self._validate_decimal(row.get('Value2', 0), -999999, 999999, 0) if pd.notna(row.get('Value2')) else None,
                    'condition_weight': self._validate_decimal(row.get('Weight', 0.5), 0.0001, 1.0, 0.5),
                    'condition_enabled': self._validate_boolean(row.get('Enabled', 'YES'))
                }

                conditions.append(condition)

            return conditions

        except Exception as e:
            self.errors.append(f"Error parsing SignalConditions sheet: {str(e)}")
            return []

    def _validate_enum(self, value, valid_values, default):
        """Validate enum field with fallback to default"""
        if pd.isna(value):
            return default

        str_value = str(value).upper().strip()
        for valid_val in valid_values:
            if str_value == valid_val.upper():
                return valid_val

        self.warnings.append(f"Invalid enum value '{value}', using default '{default}'")
        return default

    def _validate_boolean(self, value):
        """Validate boolean field"""
        if pd.isna(value):
            return True

        if isinstance(value, bool):
            return value

        if isinstance(value, (int, float)):
            return bool(value)

        if isinstance(value, str):
            return value.upper().strip() in ['YES', 'TRUE', '1', 'Y', 'T']

        return True

    def _validate_decimal(self, value, min_val, max_val, default):
        """Validate decimal field with range checking and robust NaN handling"""
        # Handle None, NaN, empty string, and other null-like values
        if value is None or pd.isna(value) or value == '' or str(value).upper() in ['NULL', 'NAN', 'NONE']:
            return default

        try:
            # Convert to float and check for NaN again
            decimal_value = float(value)

            # Explicit NaN check after conversion
            if pd.isna(decimal_value) or np.isnan(decimal_value):
                self.warnings.append(f"NaN value encountered, using default {default}")
                return default

            # Check for infinity
            if np.isinf(decimal_value):
                self.warnings.append(f"Infinite value encountered, using default {default}")
                return default

            # Range validation
            if decimal_value < min_val:
                self.warnings.append(f"Value {decimal_value} below minimum {min_val}, clamping")
                return min_val

            if decimal_value > max_val:
                self.warnings.append(f"Value {decimal_value} above maximum {max_val}, clamping")
                return max_val

            return decimal_value

        except (ValueError, TypeError, OverflowError) as e:
            self.warnings.append(f"Invalid decimal value '{value}' ({type(value).__name__}): {e}, using default {default}")
            return default

    def _validate_integer(self, value, min_val, max_val, default):
        """Validate integer field with range checking"""
        if pd.isna(value):
            return default

        try:
            int_value = int(float(value))

            if int_value < min_val:
                self.warnings.append(f"Value {int_value} below minimum {min_val}, clamping")
                return min_val

            if int_value > max_val:
                self.warnings.append(f"Value {int_value} above maximum {max_val}, clamping")
                return max_val

            return int_value

        except (ValueError, TypeError):
            self.warnings.append(f"Invalid integer value '{value}', using default {default}")
            return default

    def _validate_indicator_parameters(self, indicator):
        """Validate indicator-specific parameters"""
        indicator_name = indicator['indicator_name']

        # RSI-specific validation
        if indicator_name == 'RSI':
            if not (0 <= indicator['threshold_lower'] <= indicator['threshold_upper'] <= 100):
                self.warnings.append(f"RSI thresholds invalid, adjusting to 30/70")
                indicator['threshold_lower'] = 30
                indicator['threshold_upper'] = 70

        # MACD-specific validation
        elif indicator_name == 'MACD':
            if indicator['period'] < 12:
                self.warnings.append(f"MACD period too small, adjusting to 12")
                indicator['period'] = 12

        # Add more indicator-specific validations as needed

    def _parse_risk_management_sheet(self, excel_data: pd.ExcelFile) -> Dict[str, Any]:
        """Parse RiskManagement sheet"""
        try:
            df = pd.read_excel(excel_data, sheet_name='RiskManagement')

            risk_params = {}
            for _, row in df.iterrows():
                if pd.isna(row.get('Parameter')):
                    continue

                param_name = str(row['Parameter']).strip()
                param_value = row.get('Value', '')

                # Type-specific validation
                if param_name in ['FixedPositionSize', 'MaxDailyLoss']:
                    risk_params[param_name] = self._validate_decimal(param_value, 100, 10000000, 100000)
                elif param_name in ['RiskPercentage', 'MaxDrawdown']:
                    risk_params[param_name] = self._validate_decimal(param_value, 0.1, 50.0, 2.0)
                elif param_name in ['MaxPositions']:
                    risk_params[param_name] = self._validate_integer(param_value, 1, 10, 3)
                elif param_name in ['StopLossValue', 'TakeProfitValue', 'TrailingStopDistance', 'BreakevenTrigger']:
                    risk_params[param_name] = self._validate_decimal(param_value, 0.1, 10.0, 1.5)
                elif param_name in ['TrailingStopEnabled', 'BreakevenEnabled']:
                    risk_params[param_name] = self._validate_boolean(param_value)
                else:
                    risk_params[param_name] = str(param_value).strip()

            return risk_params

        except Exception as e:
            self.errors.append(f"Error parsing RiskManagement sheet: {str(e)}")
            return {}

    def _parse_timeframe_settings_sheet(self, excel_data: pd.ExcelFile) -> List[Dict[str, Any]]:
        """Parse TimeframeSettings sheet"""
        try:
            df = pd.read_excel(excel_data, sheet_name='TimeframeSettings')

            timeframes = []
            for _, row in df.iterrows():
                if pd.isna(row.get('Timeframe')):
                    continue

                timeframe = {
                    'timeframe': self._validate_enum(row['Timeframe'], ['1m', '5m', '15m', '1h', '4h', '1d'], '5m'),
                    'enabled': self._validate_boolean(row.get('Enabled', 'YES')),
                    'weight': self._validate_decimal(row.get('Weight', 0.1), 0.0001, 1.0, 0.1),
                    'purpose': self._validate_enum(row.get('Purpose', 'PRIMARY_SIGNALS'),
                                                  ['ENTRY_TIMING', 'PRIMARY_SIGNALS', 'TREND_CONFIRMATION', 'MAJOR_TREND', 'LONG_TERM_BIAS', 'MARKET_STRUCTURE'], 'PRIMARY_SIGNALS'),
                    'indicator_set': self._validate_enum(row.get('IndicatorSet', 'FULL'),
                                                        ['FULL', 'TREND', 'MOMENTUM', 'VOLATILITY', 'VOLUME', 'SMC', 'PATTERN', 'BASIC', 'SCALPING'], 'FULL'),
                    'priority': self._validate_integer(row.get('Priority', 5), 1, 10, 5)
                }

                timeframes.append(timeframe)

            # Auto-normalize weight distribution for enabled timeframes
            enabled_timeframes = [tf for tf in timeframes if tf['enabled']]
            if enabled_timeframes:
                total_weight = sum(tf['weight'] for tf in enabled_timeframes)
                if abs(total_weight - 1.0) > 0.0001:
                    self.warnings.append(f"Timeframe weights sum to {total_weight}, auto-normalizing to 1.0")

                    # Auto-normalize weights
                    if total_weight > 0:
                        normalization_factor = 1.0 / total_weight
                        for tf in enabled_timeframes:
                            tf['weight'] = tf['weight'] * normalization_factor

                        # Update the original timeframes list
                        for i, tf in enumerate(timeframes):
                            if tf['enabled']:
                                for enabled_tf in enabled_timeframes:
                                    if tf['timeframe'] == enabled_tf['timeframe']:
                                        timeframes[i] = enabled_tf
                                        break
                    else:
                        # If all weights are 0, distribute equally
                        equal_weight = 1.0 / len(enabled_timeframes)
                        for tf in enabled_timeframes:
                            tf['weight'] = equal_weight
                        self.warnings.append(f"All timeframe weights were 0, distributed equally: {equal_weight}")

            return timeframes

        except Exception as e:
            self.errors.append(f"Error parsing TimeframeSettings sheet: {str(e)}")
            return []

    def _parse_portfolio_settings_sheet(self, excel_data: pd.ExcelFile) -> Dict[str, Any]:
        """Parse PortfolioSettings sheet"""
        try:
            df = pd.read_excel(excel_data, sheet_name='PortfolioSettings')

            portfolio_params = {}
            for _, row in df.iterrows():
                if pd.isna(row.get('Parameter')):
                    continue

                param_name = str(row['Parameter']).strip()
                param_value = row.get('Value', '')

                # Type-specific validation
                if param_name in ['InitialCapital']:
                    portfolio_params[param_name] = self._validate_decimal(param_value, 10000, 100000000, 1000000)
                elif param_name in ['MaxCapitalRisk', 'CorrelationLimit', 'CommissionRate', 'SlippageRate', 'MarginRequirement']:
                    portfolio_params[param_name] = self._validate_decimal(param_value, 0.0001, 1.0, 0.05)
                elif param_name in ['LeverageLimit']:
                    portfolio_params[param_name] = self._validate_decimal(param_value, 1.0, 20.0, 5.0)
                else:
                    portfolio_params[param_name] = str(param_value).strip()

            return portfolio_params

        except Exception as e:
            self.errors.append(f"Error parsing PortfolioSettings sheet: {str(e)}")
            return {}

    def _parse_strategy_settings_sheet(self, excel_data: pd.ExcelFile) -> List[Dict[str, Any]]:
        """Parse StrategySettings sheet"""
        try:
            df = pd.read_excel(excel_data, sheet_name='StrategySettings')

            strategies = []
            for _, row in df.iterrows():
                if pd.isna(row.get('StrategyName')):
                    continue

                strategy = {
                    'strategy_name': str(row['StrategyName']).strip(),
                    'enabled': self._validate_boolean(row.get('Enabled', 'YES')),
                    'weight': self._validate_decimal(row.get('Weight', 0.25), 0.0001, 1.0, 0.25),
                    'max_positions': self._validate_integer(row.get('MaxPositions', 2), 1, 10, 2),
                    'indicator_set': self._validate_enum(row.get('IndicatorSet', 'FULL'),
                                                        ['FULL', 'TREND', 'MOMENTUM', 'VOLATILITY', 'VOLUME', 'SMC', 'PATTERN'], 'FULL'),
                    'timeframe': self._validate_enum(row.get('Timeframe', '5m'), ['1m', '5m', '15m', '1h', '4h', '1d'], '5m')
                }

                strategies.append(strategy)

            # Auto-normalize weight distribution for enabled strategies
            enabled_strategies = [s for s in strategies if s['enabled']]
            if enabled_strategies:
                total_weight = sum(s['weight'] for s in enabled_strategies)
                if abs(total_weight - 1.0) > 0.0001:
                    self.warnings.append(f"Strategy weights sum to {total_weight}, auto-normalizing to 1.0")

                    # Auto-normalize weights
                    if total_weight > 0:
                        normalization_factor = 1.0 / total_weight
                        for s in enabled_strategies:
                            s['weight'] = s['weight'] * normalization_factor

                        # Update the original strategies list
                        for i, s in enumerate(strategies):
                            if s['enabled']:
                                for enabled_s in enabled_strategies:
                                    if s['strategy_name'] == enabled_s['strategy_name']:
                                        strategies[i] = enabled_s
                                        break
                    else:
                        # If all weights are 0, distribute equally
                        equal_weight = 1.0 / len(enabled_strategies)
                        for s in enabled_strategies:
                            s['weight'] = equal_weight
                        self.warnings.append(f"All strategy weights were 0, distributed equally: {equal_weight}")

            return strategies

        except Exception as e:
            self.errors.append(f"Error parsing StrategySettings sheet: {str(e)}")
            return []

    def _validate_cross_sheet_references(self, parsed_data):
        """Validate references between sheets with enhanced error recovery"""
        # Get available indicators
        available_indicators = [ind['indicator_name'] for ind in parsed_data['indicator_configuration']]

        # Add common derived indicators that might be referenced
        derived_indicators = []
        for ind_name in available_indicators:
            if ind_name == 'BBANDS':
                derived_indicators.extend(['BBANDS_UPPER', 'BBANDS_MIDDLE', 'BBANDS_LOWER'])
            elif ind_name.startswith('EMA'):
                derived_indicators.append(f"{ind_name}_SIGNAL")
            elif ind_name == 'MACD':
                derived_indicators.extend(['MACD_SIGNAL', 'MACD_HISTOGRAM'])

        all_available_indicators = available_indicators + derived_indicators

        # Validate signal condition references with error recovery
        conditions_to_remove = []
        for i, condition in enumerate(parsed_data['signal_conditions']):
            has_error = False

            # Check indicator1
            if condition['indicator1_name'] and condition['indicator1_name'] not in all_available_indicators:
                self.warnings.append(f"Signal condition '{condition['condition_id']}' references unknown indicator: {condition['indicator1_name']}")
                has_error = True

            # Check indicator2
            if condition['indicator2_name'] and condition['indicator2_name'] not in all_available_indicators:
                self.warnings.append(f"Signal condition '{condition['condition_id']}' references unknown indicator: {condition['indicator2_name']}")
                has_error = True

            # If condition has errors, disable it instead of failing
            if has_error:
                condition['condition_enabled'] = False
                self.warnings.append(f"Disabled signal condition '{condition['condition_id']}' due to invalid references")

        # Validate that we have at least one enabled entry condition
        enabled_entry_conditions = [c for c in parsed_data['signal_conditions']
                                   if c['condition_type'] == 'ENTRY' and c['condition_enabled']]

        if not enabled_entry_conditions:
            self.warnings.append("No enabled entry conditions found, strategy may not generate signals")

        # Validate timeframe references in strategy settings
        available_timeframes = [tf['timeframe'] for tf in parsed_data['timeframe_settings'] if tf['enabled']]

        for strategy in parsed_data['strategy_settings']:
            if strategy['enabled'] and strategy['timeframe'] not in available_timeframes:
                self.warnings.append(f"Strategy '{strategy['strategy_name']}' references disabled timeframe: {strategy['timeframe']}")
                # Auto-assign to first available timeframe
                if available_timeframes:
                    strategy['timeframe'] = available_timeframes[0]
                    self.warnings.append(f"Auto-assigned strategy '{strategy['strategy_name']}' to timeframe: {available_timeframes[0]}")


# Backward compatibility alias
MLIndicatorParser = IndicatorParser