"""
Actual System Integrator

This module integrates the Excel configuration with the ACTUAL existing system
at /srv/samba/shared/enhanced-market-regime-optimizer-final-package-updated/
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
import configparser

# Add the actual system path
sys.path.append('/srv/samba/shared/enhanced-market-regime-optimizer-final-package-updated')

try:
    # Import from the actual existing system
    from core.dynamic_weightage_integration import DynamicWeightageIntegrator
    from core.comprehensive_indicator_engine import ComprehensiveIndicatorEngine
    from utils.feature_engineering.greek_sentiment.greek_sentiment_analysis import GreekSentimentAnalysis
    from utils.feature_engineering.trending_oi_pa.trending_oi_pa_analysis import TrendingOIWithPAAnalysis
    from utils.feature_engineering.ema_indicators.ema_indicators import EMAIndicators
    from utils.feature_engineering.vwap_indicators.vwap_indicators import VWAPIndicators
    from utils.feature_engineering.iv_skew.iv_skew_analysis import IVSkewAnalysis
    from utils.feature_engineering.atr_indicators import ATRIndicators
except ImportError as e:
    logging.warning(f"Could not import from actual system: {e}")

from .actual_system_excel_manager import ActualSystemExcelManager

logger = logging.getLogger(__name__)

class ActualSystemIntegrator:
    """
    Integrator that connects Excel configuration with the actual existing system
    
    This class:
    1. Loads Excel configuration
    2. Initializes the actual existing indicator systems
    3. Applies dynamic weightage from Excel
    4. Provides unified interface for market regime calculation
    """
    
    def __init__(self, excel_config_path: str = None):
        """Initialize integrator with Excel configuration"""
        self.excel_config_path = excel_config_path
        self.excel_manager = ActualSystemExcelManager(excel_config_path)
        
        # Initialize actual system components
        self.dynamic_integrator = None
        self.comprehensive_engine = None
        self.indicator_systems = {}
        
        # Configuration data
        self.indicator_config = None
        self.straddle_config = None
        self.dynamic_weights_config = None
        self.timeframe_config = None
        self.greek_config = None
        self.regime_config = None
        
        # Load configuration if provided
        if excel_config_path:
            self.load_excel_configuration()
        
        # Initialize actual systems
        self._initialize_actual_systems()
        
        logger.info("ActualSystemIntegrator initialized")
    
    def load_excel_configuration(self) -> bool:
        """Load configuration from Excel file"""
        try:
            if not self.excel_manager.load_configuration():
                logger.error("Failed to load Excel configuration")
                return False
            
            # Load all configuration sheets
            self.indicator_config = self.excel_manager.get_indicator_configuration()
            self.straddle_config = self.excel_manager.get_straddle_configuration()
            self.dynamic_weights_config = self.excel_manager.get_dynamic_weightage_configuration()
            self.timeframe_config = self.excel_manager.get_timeframe_configuration()
            self.greek_config = self.excel_manager.get_greek_sentiment_configuration()
            self.regime_config = self.excel_manager.get_regime_formation_configuration()
            
            # Validate configuration
            is_valid, errors = self.excel_manager.validate_configuration()
            if not is_valid:
                logger.warning(f"Configuration validation issues: {errors}")
            
            logger.info("✅ Excel configuration loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading Excel configuration: {e}")
            return False
    
    def _initialize_actual_systems(self):
        """Initialize the actual existing indicator systems"""
        try:
            # Create INI config from Excel configuration
            ini_config_path = self._create_ini_from_excel()
            
            # Initialize dynamic weightage integrator
            self.dynamic_integrator = DynamicWeightageIntegrator(ini_config_path)
            
            # Initialize comprehensive indicator engine
            self.comprehensive_engine = ComprehensiveIndicatorEngine(ini_config_path)
            
            # Initialize individual systems based on Excel config
            if self.indicator_config is not None:
                enabled_systems = self.indicator_config[self.indicator_config['Enabled'] == True]
                
                for _, row in enabled_systems.iterrows():
                    system_name = row['IndicatorSystem']
                    
                    try:
                        if system_name == 'greek_sentiment':
                            config = self._get_greek_config_dict()
                            self.indicator_systems[system_name] = GreekSentimentAnalysis(config)
                        
                        elif system_name == 'trending_oi_pa':
                            config = self._get_oi_pa_config_dict()
                            self.indicator_systems[system_name] = TrendingOIWithPAAnalysis(config)
                        
                        elif system_name == 'ema_indicators':
                            config = self._get_ema_config_dict()
                            self.indicator_systems[system_name] = EMAIndicators(config)
                        
                        elif system_name == 'vwap_indicators':
                            config = self._get_vwap_config_dict()
                            self.indicator_systems[system_name] = VWAPIndicators(config)
                        
                        elif system_name == 'iv_skew':
                            config = self._get_iv_skew_config_dict()
                            self.indicator_systems[system_name] = IVSkewAnalysis(config)
                        
                        elif system_name == 'atr_indicators':
                            config = self._get_atr_config_dict()
                            self.indicator_systems[system_name] = ATRIndicators(config)
                        
                        logger.info(f"Initialized {system_name}")
                        
                    except Exception as e:
                        logger.error(f"Error initializing {system_name}: {e}")
            
            logger.info(f"✅ Initialized {len(self.indicator_systems)} actual systems")
            
        except Exception as e:
            logger.error(f"Error initializing actual systems: {e}")
    
    def _create_ini_from_excel(self) -> str:
        """Create INI configuration file from Excel configuration"""
        try:
            config = configparser.ConfigParser()
            
            # Add indicator categories section
            config['indicator_categories'] = {}
            if self.indicator_config is not None:
                for _, row in self.indicator_config.iterrows():
                    config['indicator_categories'][row['IndicatorSystem']] = str(row['Enabled']).lower()
            
            # Add individual indicator sections
            if self.indicator_config is not None:
                for _, row in self.indicator_config.iterrows():
                    system_name = row['IndicatorSystem']
                    config[system_name] = {
                        'enabled': str(row['Enabled']).lower(),
                        'base_weight': str(row['BaseWeight']),
                        'performance_tracking': str(row['PerformanceTracking']).lower(),
                        'adaptive_weight': str(row['AdaptiveWeight']).lower()
                    }
                    
                    # Parse parameters
                    if pd.notna(row['Parameters']) and row['Parameters']:
                        params = row['Parameters'].split(',')
                        for param in params:
                            if '=' in param:
                                key, value = param.split('=', 1)
                                config[system_name][key.strip()] = value.strip()
            
            # Add straddle analysis configuration
            if self.straddle_config is not None:
                config['straddle_analysis'] = {
                    'enabled': 'true',
                    'base_weight': '0.25',
                    'performance_tracking': 'true',
                    'adaptive_weight': 'true'
                }
                
                # Add individual straddle configurations
                for _, row in self.straddle_config.iterrows():
                    section_name = f"straddle_analysis.{row['StraddleType'].lower()}"
                    config[section_name] = {
                        'enabled': str(row['Enabled']).lower(),
                        'weight': str(row['Weight']),
                        'ema_enabled': str(row['EMAEnabled']).lower(),
                        'ema_periods': str(row['EMAPeriods']) if pd.notna(row['EMAPeriods']) else '',
                        'vwap_enabled': str(row['VWAPEnabled']).lower(),
                        'vwap_types': str(row['VWAPTypes']) if pd.notna(row['VWAPTypes']) else '',
                        'previous_day_vwap': str(row['PreviousDayVWAP']).lower(),
                        'timeframes': str(row['Timeframes']) if pd.notna(row['Timeframes']) else ''
                    }
            
            # Add dynamic weightage configuration
            if self.dynamic_weights_config is not None:
                config['dynamic_weightage'] = {
                    'enabled': 'true',
                    'learning_rate': '0.01',
                    'min_weight': '0.02',
                    'max_weight': '0.60'
                }
            
            # Add Greek sentiment configuration
            if self.greek_config is not None:
                if 'greek_sentiment' not in config:
                    config['greek_sentiment'] = {}
                
                for _, row in self.greek_config.iterrows():
                    config['greek_sentiment'][row['Parameter']] = str(row['Value'])
            
            # Save INI file
            ini_path = 'temp_config_from_excel.ini'
            with open(ini_path, 'w') as f:
                config.write(f)
            
            logger.info(f"Created INI config from Excel: {ini_path}")
            return ini_path
            
        except Exception as e:
            logger.error(f"Error creating INI from Excel: {e}")
            return None
    
    def _get_greek_config_dict(self) -> Dict[str, Any]:
        """Get Greek sentiment configuration as dictionary"""
        config = {}
        if self.greek_config is not None:
            for _, row in self.greek_config.iterrows():
                param = row['Parameter']
                value = row['Value']
                
                # Convert to appropriate type
                if row['Type'] == 'int':
                    config[param] = int(value)
                elif row['Type'] == 'float':
                    config[param] = float(value)
                else:
                    config[param] = value
        
        return config
    
    def _get_oi_pa_config_dict(self) -> Dict[str, Any]:
        """Get OI PA configuration as dictionary"""
        return {
            'oi_lookback': 10,
            'price_lookback': 5,
            'divergence_threshold': 0.1,
            'accumulation_threshold': 0.2,
            'use_percentile': True,
            'percentile_window': 20
        }
    
    def _get_ema_config_dict(self) -> Dict[str, Any]:
        """Get EMA configuration as dictionary"""
        return {
            'ema_periods': [20, 50, 100, 200],
            'timeframes': ['5m', '15m', '30m', '60m'],
            'price_columns': ['price', 'close', 'atm_straddle'],
            'crossover_signals': True,
            'slope_analysis': True,
            'distance_analysis': True
        }
    
    def _get_vwap_config_dict(self) -> Dict[str, Any]:
        """Get VWAP configuration as dictionary"""
        return {
            'timeframes': ['5m', '15m', '30m', '60m'],
            'current_day': True,
            'previous_day': True,
            'weekly': True,
            'monthly': True,
            'rolling_20': True,
            'rolling_50': True,
            'anchored_vwap': True,
            'deviation_bands': [1, 2, 3],
            'band_penetration_signals': True
        }
    
    def _get_iv_skew_config_dict(self) -> Dict[str, Any]:
        """Get IV skew configuration as dictionary"""
        return {
            'atm_range': 3,
            'otm_range': 5,
            'calculation_method': 'polynomial_fit',
            'skew_term_structure': True,
            'skew_momentum': True,
            'skew_percentile_ranking': True
        }
    
    def _get_atr_config_dict(self) -> Dict[str, Any]:
        """Get ATR configuration as dictionary"""
        return {
            'period': 14,
            'ema_period': 10,
            'percentile_window': 20,
            'percentile_bins': 4,
            'ema_smoothing': True,
            'atr_percentile_ranking': True,
            'atr_ema_ratio': True,
            'volatility_regime_detection': True
        }
    
    def calculate_market_regime(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate market regime using the actual existing system with Excel configuration
        
        Args:
            data (pd.DataFrame): Market data
            **kwargs: Additional arguments
            
        Returns:
            pd.DataFrame: Market regime results
        """
        try:
            logger.info("Calculating market regime using actual system with Excel config")
            
            # Use the comprehensive engine if available
            if self.comprehensive_engine:
                # Calculate all indicators
                indicator_results = self.comprehensive_engine.calculate_all_indicators(data, **kwargs)
                
                # Calculate market regime
                market_regime = self.comprehensive_engine.calculate_market_regime(indicator_results)
                
                # Apply Excel-based weight adjustments
                if self.dynamic_weights_config is not None:
                    market_regime = self._apply_excel_weight_adjustments(market_regime)
                
                # Add Excel configuration metadata
                market_regime['excel_config_applied'] = True
                market_regime['config_timestamp'] = datetime.now()
                
                logger.info(f"✅ Market regime calculated: {len(market_regime)} points")
                return market_regime
            
            # Fallback to dynamic integrator
            elif self.dynamic_integrator:
                # Calculate all indicators
                indicator_results = self.dynamic_integrator.calculate_all_indicators(data, **kwargs)
                
                # Calculate market regime
                market_regime = self.dynamic_integrator.calculate_market_regime(indicator_results)
                
                # Apply Excel-based adjustments
                if self.dynamic_weights_config is not None:
                    market_regime = self._apply_excel_weight_adjustments(market_regime)
                
                logger.info(f"✅ Market regime calculated (fallback): {len(market_regime)} points")
                return market_regime
            
            else:
                logger.error("No actual system components available")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error calculating market regime: {e}")
            return pd.DataFrame()
    
    def _apply_excel_weight_adjustments(self, market_regime: pd.DataFrame) -> pd.DataFrame:
        """Apply weight adjustments from Excel configuration"""
        try:
            if self.dynamic_weights_config is None:
                return market_regime
            
            # Get current weights from Excel
            excel_weights = {}
            for _, row in self.dynamic_weights_config.iterrows():
                if row['AutoAdjust']:
                    excel_weights[row['SystemName']] = row['CurrentWeight']
            
            # Apply weight adjustments to existing signals
            for system_name, weight in excel_weights.items():
                signal_col = f'{system_name}_signal'
                weight_col = f'{system_name}_weight'
                
                if signal_col in market_regime.columns:
                    # Update weight column
                    market_regime[weight_col] = weight
                    
                    # Recalculate contribution
                    contribution_col = f'{system_name}_contribution'
                    if contribution_col in market_regime.columns:
                        market_regime[contribution_col] = market_regime[signal_col] * weight
            
            # Recalculate overall regime score
            signal_cols = [col for col in market_regime.columns if col.endswith('_signal')]
            weight_cols = [col for col in market_regime.columns if col.endswith('_weight')]
            
            if signal_cols and weight_cols:
                # Calculate weighted average
                weighted_sum = pd.Series(0.0, index=market_regime.index)
                total_weight = pd.Series(0.0, index=market_regime.index)
                
                for signal_col in signal_cols:
                    system_name = signal_col.replace('_signal', '')
                    weight_col = f'{system_name}_weight'
                    
                    if weight_col in market_regime.columns:
                        weighted_sum += market_regime[signal_col] * market_regime[weight_col]
                        total_weight += market_regime[weight_col]
                
                # Update regime score
                market_regime['Market_Regime_Score'] = weighted_sum / (total_weight + 1e-10)
                
                # Update regime label
                market_regime['Market_Regime_Label'] = self._score_to_label(market_regime['Market_Regime_Score'])
            
            logger.info("Applied Excel weight adjustments")
            return market_regime
            
        except Exception as e:
            logger.error(f"Error applying Excel weight adjustments: {e}")
            return market_regime
    
    def _score_to_label(self, scores: pd.Series) -> pd.Series:
        """Convert regime scores to labels"""
        labels = pd.Series('Neutral', index=scores.index)
        
        labels[scores >= 1.0] = 'Strong_Bullish'
        labels[(scores >= 0.5) & (scores < 1.0)] = 'Bullish'
        labels[(scores >= 0.2) & (scores < 0.5)] = 'Mild_Bullish'
        labels[(scores > -0.2) & (scores < 0.2)] = 'Neutral'
        labels[(scores > -0.5) & (scores <= -0.2)] = 'Mild_Bearish'
        labels[(scores > -1.0) & (scores <= -0.5)] = 'Bearish'
        labels[scores <= -1.0] = 'Strong_Bearish'
        
        return labels
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all systems"""
        status = {
            'excel_config_loaded': self.excel_config_path is not None,
            'excel_config_path': self.excel_config_path,
            'comprehensive_engine_available': self.comprehensive_engine is not None,
            'dynamic_integrator_available': self.dynamic_integrator is not None,
            'individual_systems_count': len(self.indicator_systems),
            'individual_systems': list(self.indicator_systems.keys())
        }
        
        # Add configuration status
        if self.indicator_config is not None:
            enabled_systems = self.indicator_config[self.indicator_config['Enabled'] == True]
            status['enabled_indicators'] = enabled_systems['IndicatorSystem'].tolist()
            status['total_base_weight'] = enabled_systems['BaseWeight'].sum()
        
        if self.dynamic_weights_config is not None:
            auto_adjust_systems = self.dynamic_weights_config[self.dynamic_weights_config['AutoAdjust'] == True]
            status['auto_adjust_systems'] = auto_adjust_systems['SystemName'].tolist()
            status['total_dynamic_weight'] = auto_adjust_systems['CurrentWeight'].sum()
        
        return status
    
    def update_weights_from_performance(self, performance_data: Dict[str, float]) -> bool:
        """Update weights based on performance data"""
        try:
            if self.dynamic_weights_config is None:
                logger.warning("No dynamic weights configuration available")
                return False
            
            # Update weights in Excel configuration
            for system_name, performance in performance_data.items():
                mask = self.dynamic_weights_config['SystemName'] == system_name
                if mask.any():
                    current_weight = self.dynamic_weights_config.loc[mask, 'CurrentWeight'].iloc[0]
                    learning_rate = self.dynamic_weights_config.loc[mask, 'LearningRate'].iloc[0]
                    min_weight = self.dynamic_weights_config.loc[mask, 'MinWeight'].iloc[0]
                    max_weight = self.dynamic_weights_config.loc[mask, 'MaxWeight'].iloc[0]
                    
                    # Calculate new weight
                    adjustment = learning_rate * (performance - 0.5)  # 0.5 is neutral
                    new_weight = current_weight + adjustment
                    new_weight = max(min_weight, min(max_weight, new_weight))
                    
                    # Update in configuration
                    self.dynamic_weights_config.loc[mask, 'CurrentWeight'] = new_weight
                    self.dynamic_weights_config.loc[mask, 'HistoricalPerformance'] = performance
            
            # Normalize weights
            auto_adjust_mask = self.dynamic_weights_config['AutoAdjust'] == True
            if auto_adjust_mask.any():
                total_weight = self.dynamic_weights_config.loc[auto_adjust_mask, 'CurrentWeight'].sum()
                if total_weight > 0:
                    self.dynamic_weights_config.loc[auto_adjust_mask, 'CurrentWeight'] *= (1.0 / total_weight)
            
            # Update actual systems if available
            if self.dynamic_integrator:
                self.dynamic_integrator.update_weights_based_on_performance(performance_data)
            
            logger.info("✅ Weights updated from performance data")
            return True
            
        except Exception as e:
            logger.error(f"Error updating weights from performance: {e}")
            return False
    
    def save_updated_configuration(self, output_path: str = None) -> bool:
        """Save updated configuration back to Excel"""
        try:
            # Update Excel manager with current configuration
            self.excel_manager.config_data['DynamicWeightageConfig'] = self.dynamic_weights_config
            
            # Save to file
            return self.excel_manager.save_configuration(output_path)
            
        except Exception as e:
            logger.error(f"Error saving updated configuration: {e}")
            return False
