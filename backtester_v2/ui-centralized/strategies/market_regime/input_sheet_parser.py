"""
Market Regime Input Sheet Parser

This module provides parsing functionality for market regime input sheets,
following the same patterns as other backtester_v2 systems (TBS, OI, ORB, etc.).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path

# Import MarketRegimeExcelManager with fallback handling
try:
    from .excel_config_manager import MarketRegimeExcelManager
except ImportError:
    try:
        from excel_config_manager import MarketRegimeExcelManager
    except ImportError:
        # Create a minimal fallback class for standalone usage
        class MarketRegimeExcelManager:
            def __init__(self, config_path=None):
                self.config_path = config_path
                self.config_data = {}
            
            def load_configuration(self):
                return {}
            
            def get_detection_parameters(self):
                return {}
            
            def get_regime_adjustments(self):
                return {}
            
            def get_live_trading_config(self):
                return {'EnableLiveTrading': False}

logger = logging.getLogger(__name__)

class MarketRegimeInputSheetParser:
    """
    Input sheet parser for market regime detection following BT patterns
    
    This class handles parsing of Excel input sheets with PortfolioSetting,
    StrategySetting, and regime-specific configuration sheets, following
    the same patterns as TBS, OI, ORB, and other backtester_v2 systems.
    """
    
    def __init__(self):
        """Initialize Market Regime Input Sheet Parser"""
        self.excel_manager = None
        self.parsed_portfolios = []
        self.parsed_strategies = []
        self.regime_config = {}
        
        logger.info("MarketRegimeInputSheetParser initialized")
    
    def parse_input_sheets(self, input_sheet_path: str) -> Dict[str, Any]:
        """
        Parse market regime input sheets following BT patterns
        
        Args:
            input_sheet_path (str): Path to Excel input sheet
            
        Returns:
            Dict: Parsed configuration with portfolios and strategies
        """
        try:
            logger.info(f"Parsing market regime input sheets: {input_sheet_path}")
            
            if not Path(input_sheet_path).exists():
                raise FileNotFoundError(f"Input sheet not found: {input_sheet_path}")
            
            # Initialize Excel manager
            self.excel_manager = MarketRegimeExcelManager(input_sheet_path)
            
            # Parse standard BT sheets
            portfolios = self._parse_portfolio_setting()
            strategies = self._parse_strategy_setting()
            
            # Parse regime-specific configuration
            regime_config = self._parse_regime_configuration()
            
            # Parse live trading settings
            live_config = self._parse_live_trading_configuration()
            
            # Validate configuration
            is_valid, errors = self._validate_parsed_configuration(
                portfolios, strategies, regime_config
            )
            
            parsed_result = {
                'portfolios': portfolios,
                'strategies': strategies,
                'regime_configuration': regime_config,
                'live_trading_config': live_config,
                'input_sheet_path': input_sheet_path,
                'parsing_timestamp': datetime.now().isoformat(),
                'validation_status': {
                    'is_valid': is_valid,
                    'errors': errors
                }
            }
            
            # Store parsed data
            self.parsed_portfolios = portfolios
            self.parsed_strategies = strategies
            self.regime_config = regime_config
            
            logger.info(f"Successfully parsed: {len(portfolios)} portfolios, {len(strategies)} strategies")
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"Error parsing input sheets: {e}")
            raise
    
    def _parse_portfolio_setting(self) -> List[Dict[str, Any]]:
        """Parse PortfolioSetting sheet (standard BT pattern)"""
        try:
            # Try to read PortfolioSetting sheet
            try:
                portfolio_df = pd.read_excel(
                    self.excel_manager.config_path, 
                    sheet_name='PortfolioSetting'
                )
            except:
                # Create default portfolio if sheet doesn't exist
                portfolio_df = pd.DataFrame([{
                    'PortfolioName': 'MarketRegime_Portfolio_1',
                    'StartDate': '01_01_2024',
                    'EndDate': '31_12_2024',
                    'Multiplier': 1,
                    'IsTickBT': 'NO',
                    'SlippagePercent': 0.1
                }])
                logger.info("Created default PortfolioSetting")
            
            portfolios = []
            for _, row in portfolio_df.iterrows():
                portfolio = {
                    'portfolio_name': str(row.get('PortfolioName', 'Default_Portfolio')),
                    'start_date': self._parse_date_string(row.get('StartDate', '01_01_2024')),
                    'end_date': self._parse_date_string(row.get('EndDate', '31_12_2024')),
                    'multiplier': float(row.get('Multiplier', 1)),
                    'is_tick_bt': str(row.get('IsTickBT', 'NO')).upper() == 'YES',
                    'slippage_percent': float(row.get('SlippagePercent', 0.1)),
                    'regime_enabled': True,  # Always enabled for regime detection
                    'live_trading_enabled': self._check_live_trading_enabled()
                }
                portfolios.append(portfolio)
            
            logger.info(f"Parsed {len(portfolios)} portfolios from PortfolioSetting")
            return portfolios
            
        except Exception as e:
            logger.error(f"Error parsing PortfolioSetting: {e}")
            # Return default portfolio
            return [{
                'portfolio_name': 'Default_MarketRegime_Portfolio',
                'start_date': '2024-01-01',
                'end_date': '2024-12-31',
                'multiplier': 1.0,
                'is_tick_bt': False,
                'slippage_percent': 0.1,
                'regime_enabled': True,
                'live_trading_enabled': False
            }]
    
    def _parse_strategy_setting(self) -> List[Dict[str, Any]]:
        """Parse StrategySetting sheet (standard BT pattern)"""
        try:
            # Try to read StrategySetting sheet
            try:
                strategy_df = pd.read_excel(
                    self.excel_manager.config_path, 
                    sheet_name='StrategySetting'
                )
            except:
                # Create default strategy if sheet doesn't exist
                strategy_df = pd.DataFrame([{
                    'Enabled': 'YES',
                    'PortfolioName': 'MarketRegime_Portfolio_1',
                    'StrategyType': 'MARKET_REGIME',
                    'StrategyExcelFilePath': self.excel_manager.config_path
                }])
                logger.info("Created default StrategySetting")
            
            strategies = []
            for _, row in strategy_df.iterrows():
                if str(row.get('Enabled', 'NO')).upper() == 'YES':
                    strategy = {
                        'enabled': True,
                        'portfolio_name': str(row.get('PortfolioName', 'Default_Portfolio')),
                        'strategy_type': str(row.get('StrategyType', 'MARKET_REGIME')),
                        'strategy_excel_file_path': str(row.get('StrategyExcelFilePath', '')),
                        'regime_detection_enabled': True,
                        'live_trading_enabled': self._check_live_trading_enabled(),
                        'strategy_id': f"regime_{len(strategies) + 1}"
                    }
                    strategies.append(strategy)
            
            logger.info(f"Parsed {len(strategies)} enabled strategies from StrategySetting")
            return strategies
            
        except Exception as e:
            logger.error(f"Error parsing StrategySetting: {e}")
            # Return default strategy
            return [{
                'enabled': True,
                'portfolio_name': 'Default_MarketRegime_Portfolio',
                'strategy_type': 'MARKET_REGIME',
                'strategy_excel_file_path': self.excel_manager.config_path if self.excel_manager else '',
                'regime_detection_enabled': True,
                'live_trading_enabled': False,
                'strategy_id': 'regime_1'
            }]
    
    def _parse_regime_configuration(self) -> Dict[str, Any]:
        """Parse regime-specific configuration from Excel"""
        try:
            if not self.excel_manager:
                return self._get_default_regime_config()
            
            # Get detection parameters
            detection_params = self.excel_manager.get_detection_parameters()
            
            # Get regime adjustments
            regime_adjustments = self.excel_manager.get_regime_adjustments()
            
            # Get strategy mappings
            strategy_mappings = self.excel_manager.get_strategy_mappings()
            
            regime_config = {
                'detection_parameters': {
                    'confidence_threshold': detection_params.get('ConfidenceThreshold', 0.6),
                    'regime_smoothing': int(detection_params.get('RegimeSmoothing', 3)),
                    'indicator_weights': {
                        'greek_sentiment': float(detection_params.get('IndicatorWeightGreek', 0.35)),
                        'oi_analysis': float(detection_params.get('IndicatorWeightOI', 0.25)),
                        'price_action': float(detection_params.get('IndicatorWeightPrice', 0.20)),
                        'technical_indicators': float(detection_params.get('IndicatorWeightTechnical', 0.15)),
                        'volatility_measures': float(detection_params.get('IndicatorWeightVolatility', 0.05))
                    },
                    'directional_thresholds': {
                        'strong_bullish': float(detection_params.get('DirectionalThresholdStrongBullish', 0.50)),
                        'mild_bullish': float(detection_params.get('DirectionalThresholdMildBullish', 0.20)),
                        'neutral': float(detection_params.get('DirectionalThresholdNeutral', 0.10)),
                        'sideways': float(detection_params.get('DirectionalThresholdSideways', 0.05)),
                        'mild_bearish': float(detection_params.get('DirectionalThresholdMildBearish', -0.20)),
                        'strong_bearish': float(detection_params.get('DirectionalThresholdStrongBearish', -0.50))
                    },
                    'volatility_thresholds': {
                        'high': float(detection_params.get('VolatilityThresholdHigh', 0.20)),
                        'normal_high': float(detection_params.get('VolatilityThresholdNormalHigh', 0.15)),
                        'normal_low': float(detection_params.get('VolatilityThresholdNormalLow', 0.10)),
                        'low': float(detection_params.get('VolatilityThresholdLow', 0.05))
                    }
                },
                'regime_adjustments': regime_adjustments,
                'strategy_mappings': strategy_mappings,
                'raw_detection_params': detection_params
            }
            
            logger.info("Successfully parsed regime configuration from Excel")
            return regime_config
            
        except Exception as e:
            logger.error(f"Error parsing regime configuration: {e}")
            return self._get_default_regime_config()
    
    def _parse_live_trading_configuration(self) -> Dict[str, Any]:
        """Parse live trading configuration from Excel"""
        try:
            if not self.excel_manager:
                return self._get_default_live_config()
            
            live_config = self.excel_manager.get_live_trading_config()
            
            parsed_live_config = {
                'enable_live_trading': self._parse_boolean(live_config.get('EnableLiveTrading', 'NO')),
                'streaming_interval_ms': int(live_config.get('StreamingIntervalMs', 100)),
                'regime_update_freq_sec': int(live_config.get('RegimeUpdateFreqSec', 60)),
                'enable_algobaba_integration': self._parse_boolean(live_config.get('EnableAlgobobaIntegration', 'NO')),
                'max_daily_orders': int(live_config.get('MaxDailyOrders', 100)),
                'max_regime_exposure': float(live_config.get('MaxRegimeExposure', 0.5)),
                'enable_regime_alerts': self._parse_boolean(live_config.get('EnableRegimeAlerts', 'YES')),
                'alert_channels': str(live_config.get('AlertChannels', 'EMAIL')).split(','),
                'regime_history_limit': int(live_config.get('RegimeHistoryLimit', 1000)),
                'enable_performance_tracking': self._parse_boolean(live_config.get('EnablePerformanceTracking', 'YES')),
                'performance_window_days': int(live_config.get('PerformanceWindowDays', 30)),
                'enable_auto_restart': self._parse_boolean(live_config.get('EnableAutoRestart', 'YES')),
                'health_check_interval_sec': int(live_config.get('HealthCheckIntervalSec', 30))
            }
            
            logger.info("Successfully parsed live trading configuration")
            return parsed_live_config
            
        except Exception as e:
            logger.error(f"Error parsing live trading configuration: {e}")
            return self._get_default_live_config()
    
    def _parse_date_string(self, date_str: str) -> str:
        """Parse date string to standard YYYY-MM-DD format"""
        try:
            if isinstance(date_str, str):
                # Handle DD_MM_YYYY format (common in BT systems)
                if '_' in date_str:
                    parts = date_str.split('_')
                    if len(parts) == 3:
                        day, month, year = parts
                        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                
                # Handle other formats
                return pd.to_datetime(date_str).strftime('%Y-%m-%d')
            
            return str(date_str)
            
        except Exception as e:
            logger.error(f"Error parsing date {date_str}: {e}")
            return '2024-01-01'
    
    def _parse_boolean(self, value: Any) -> bool:
        """Parse boolean value from Excel"""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.upper() in ['YES', 'TRUE', '1', 'Y']
        if isinstance(value, (int, float)):
            return value > 0
        return False
    
    def _check_live_trading_enabled(self) -> bool:
        """Check if live trading is enabled in configuration"""
        try:
            if self.excel_manager:
                live_config = self.excel_manager.get_live_trading_config()
                return self._parse_boolean(live_config.get('EnableLiveTrading', 'NO'))
            return False
            
        except Exception as e:
            logger.error(f"Error checking live trading status: {e}")
            return False
    
    def _validate_parsed_configuration(self, portfolios: List[Dict[str, Any]], 
                                     strategies: List[Dict[str, Any]], 
                                     regime_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate parsed configuration"""
        try:
            errors = []
            
            # Validate portfolios
            if not portfolios:
                errors.append("No portfolios found")
            else:
                for i, portfolio in enumerate(portfolios):
                    if not portfolio.get('portfolio_name'):
                        errors.append(f"Portfolio {i}: Missing portfolio name")
                    
                    if not portfolio.get('start_date') or not portfolio.get('end_date'):
                        errors.append(f"Portfolio {i}: Missing start or end date")
            
            # Validate strategies
            if not strategies:
                errors.append("No enabled strategies found")
            else:
                portfolio_names = {p['portfolio_name'] for p in portfolios}
                for i, strategy in enumerate(strategies):
                    if not strategy.get('portfolio_name'):
                        errors.append(f"Strategy {i}: Missing portfolio name")
                    elif strategy['portfolio_name'] not in portfolio_names:
                        errors.append(f"Strategy {i}: Portfolio '{strategy['portfolio_name']}' not found")
            
            # Validate regime configuration
            if 'detection_parameters' in regime_config:
                detection_params = regime_config['detection_parameters']
                
                # Check confidence threshold
                confidence = detection_params.get('confidence_threshold', 0.6)
                if not 0.0 <= confidence <= 1.0:
                    errors.append(f"Invalid confidence threshold: {confidence}")
                
                # Check indicator weights
                weights = detection_params.get('indicator_weights', {})
                total_weight = sum(weights.values())
                if abs(total_weight - 1.0) > 0.01:
                    errors.append(f"Indicator weights sum to {total_weight:.3f}, should be 1.0")
            
            is_valid = len(errors) == 0
            
            if is_valid:
                logger.info("Configuration validation passed")
            else:
                logger.warning(f"Configuration validation failed with {len(errors)} errors")
            
            return is_valid, errors
            
        except Exception as e:
            logger.error(f"Error validating configuration: {e}")
            return False, [f"Validation error: {str(e)}"]
    
    def _get_default_regime_config(self) -> Dict[str, Any]:
        """Get default regime configuration"""
        return {
            'detection_parameters': {
                'confidence_threshold': 0.6,
                'regime_smoothing': 3,
                'indicator_weights': {
                    'greek_sentiment': 0.35,
                    'oi_analysis': 0.25,
                    'price_action': 0.20,
                    'technical_indicators': 0.15,
                    'volatility_measures': 0.05
                },
                'directional_thresholds': {
                    'strong_bullish': 0.50,
                    'mild_bullish': 0.20,
                    'neutral': 0.10,
                    'sideways': 0.05,
                    'mild_bearish': -0.20,
                    'strong_bearish': -0.50
                },
                'volatility_thresholds': {
                    'high': 0.20,
                    'normal_high': 0.15,
                    'normal_low': 0.10,
                    'low': 0.05
                }
            },
            'regime_adjustments': {},
            'strategy_mappings': {},
            'raw_detection_params': {}
        }
    
    def _get_default_live_config(self) -> Dict[str, Any]:
        """Get default live trading configuration"""
        return {
            'enable_live_trading': False,
            'streaming_interval_ms': 100,
            'regime_update_freq_sec': 60,
            'enable_algobaba_integration': False,
            'max_daily_orders': 100,
            'max_regime_exposure': 0.5,
            'enable_regime_alerts': True,
            'alert_channels': ['EMAIL'],
            'regime_history_limit': 1000,
            'enable_performance_tracking': True,
            'performance_window_days': 30,
            'enable_auto_restart': True,
            'health_check_interval_sec': 30
        }
    
    def get_parsed_portfolios(self) -> List[Dict[str, Any]]:
        """Get parsed portfolios"""
        return self.parsed_portfolios
    
    def get_parsed_strategies(self) -> List[Dict[str, Any]]:
        """Get parsed strategies"""
        return self.parsed_strategies
    
    def get_regime_configuration(self) -> Dict[str, Any]:
        """Get regime configuration"""
        return self.regime_config
