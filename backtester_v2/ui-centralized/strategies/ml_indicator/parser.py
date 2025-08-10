"""
Parser for ML Indicator Strategy Excel files
Handles portfolio settings, indicator configurations, and ML model settings
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, time, date
import logging
from pathlib import Path
import json

from .models import (
    MLIndicatorStrategyModel,
    MLIndicatorPortfolioModel,
    MLLegModel,
    IndicatorConfig,
    SMCConfig,
    MLModelConfig,
    MLFeatureConfig,
    SignalCondition,
    RiskManagementConfig,
    ExecutionConfig,
    IndicatorType,
    TALibIndicator,
    SMCIndicator,
    ComparisonOperator,
    SignalLogic,
    MLModelType,
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


class MLIndicatorParser:
    """Parser for ML Indicator strategy Excel files"""
    
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
    
    def parse_ml_model_config(self, config_path: str) -> MLModelConfig:
        """Parse ML model configuration from file"""
        try:
            if config_path.endswith('.json'):
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
            elif config_path.endswith('.xlsx'):
                df = pd.read_excel(config_path, sheet_name='MLConfig')
                config_data = {}
                for idx, row in df.iterrows():
                    if pd.notna(row.get('Parameter')) and pd.notna(row.get('Value')):
                        config_data[row['Parameter']] = row['Value']
            else:
                raise ValueError(f"Unsupported ML config format: {config_path}")
                
            # Parse ML configuration
            ml_config = {
                "model_type": MLModelType(config_data.get('model_type', 'XGBOOST')),
                "features": config_data.get('features', []),
                "target_variable": config_data.get('target_variable', 'future_return'),
                "training_window": int(config_data.get('training_window', 252)),
                "validation_split": float(config_data.get('validation_split', 0.2)),
                "test_split": float(config_data.get('test_split', 0.1)),
                "retraining_frequency": int(config_data.get('retraining_frequency', 20)),
                "prediction_horizon": int(config_data.get('prediction_horizon', 5)),
                "confidence_threshold": float(config_data.get('confidence_threshold', 0.6)),
                "use_probability": bool(config_data.get('use_probability', True))
            }
            
            # Parse model parameters
            if 'model_params' in config_data:
                ml_config['model_params'] = config_data['model_params']
            else:
                # Use defaults
                ml_config['model_params'] = ML_MODEL_PARAMS.get(
                    ml_config['model_type'], {}
                ).copy()
                
            # Parse ensemble settings
            if config_data.get('ensemble_method'):
                ml_config['ensemble_method'] = config_data['ensemble_method']
                ml_config['ensemble_weights'] = config_data.get('ensemble_weights')
                
            return MLModelConfig(**ml_config)
            
        except Exception as e:
            logger.error(f"Error parsing ML model config: {str(e)}")
            self.warnings.append(f"ML config parsing warning: {str(e)}")
            return None
    
    def parse_signal_conditions(self, signal_path: str) -> Dict[str, List[SignalCondition]]:
        """Parse signal conditions from separate file"""
        try:
            df = pd.read_excel(signal_path, sheet_name='SignalConditions')
            return self._parse_signals(df)
            
        except Exception as e:
            logger.error(f"Error parsing signal conditions: {str(e)}")
            self.warnings.append(f"Signal parsing warning: {str(e)}")
            return {"entry_signals": [], "exit_signals": []}
    
    def _create_strategy_model(self, parsed_data: Dict[str, Any]) -> MLIndicatorStrategyModel:
        """Create validated strategy model from parsed data"""
        # Create portfolio model
        portfolio = MLIndicatorPortfolioModel(**parsed_data['portfolio'])
        
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
            
        # Create legs if present
        legs = []
        for leg_data in parsed_data.get('indicators', {}).get('legs', []):
            legs.append(MLLegModel(**leg_data))
            
        # Create strategy model
        strategy = MLIndicatorStrategyModel(
            portfolio=portfolio,
            indicators=indicators,
            smc_config=parsed_data.get('smc_config'),
            ml_config=parsed_data.get('ml_config'),
            ml_feature_config=parsed_data.get('ml_feature_config'),
            entry_signals=entry_signals,
            exit_signals=exit_signals,
            signal_logic=parsed_data.get('signals', {}).get('signal_logic', SignalLogic.AND),
            legs=legs
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