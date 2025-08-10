#!/usr/bin/env python3
"""
Intraday Optimizer for Market Regime Analysis
=============================================

Specialized optimizer for intraday trading mode that optimizes parameters
for short-term trading strategies with focus on quick execution and
high-frequency regime changes.

Features:
- Short timeframe optimization (1-15 minutes)
- High-frequency regime detection
- Quick response parameter tuning
- Volatility-based adjustments
- Risk management for intraday trading
- Performance tracking for short-term strategies

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

from ..common_utils import MathUtils, ErrorHandler, PerformanceTimer

logger = logging.getLogger(__name__)


class IntradayOptimizer:
    """
    Specialized optimizer for intraday trading regime formation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Intraday Optimizer"""
        self.config = config
        
        # Intraday-specific configuration
        self.target_timeframes = ['3min', '5min', '10min', '15min']
        self.primary_timeframe = config.get('primary_timeframe', '5min')
        self.max_holding_period_minutes = config.get('max_holding_period', 60)
        self.min_regime_duration_minutes = config.get('min_regime_duration', 5)
        
        # Optimization parameters
        self.volatility_multiplier = config.get('volatility_multiplier', 1.2)
        self.response_sensitivity = config.get('response_sensitivity', 0.8)
        self.confidence_threshold = config.get('confidence_threshold', 0.65)
        
        # Risk management
        self.max_risk_per_trade = config.get('max_risk_per_trade', 0.02)
        self.position_sizing_method = config.get('position_sizing', 'volatility_adjusted')
        
        # Initialize utilities
        self.math_utils = MathUtils()
        self.error_handler = ErrorHandler()
        self.performance_timer = PerformanceTimer()
        
        # Performance tracking
        self.optimization_history = []
        self.performance_metrics = {}
        
        logger.info(f"Intraday Optimizer initialized with primary timeframe: {self.primary_timeframe}")
    
    def optimize_regime_parameters(
        self,
        historical_data: pd.DataFrame,
        current_market_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize regime detection parameters for intraday trading
        
        Args:
            historical_data: Historical market data for optimization
            current_market_conditions: Current market state
            
        Returns:
            Dictionary with optimized parameters
        """
        try:
            logger.info("Starting intraday regime parameter optimization")
            
            # Analyze market conditions for intraday suitability
            market_analysis = self._analyze_intraday_conditions(historical_data)
            
            # Calculate volatility-based adjustments
            volatility_adjustments = self._calculate_volatility_adjustments(
                historical_data, current_market_conditions
            )
            
            # Optimize timeframe weights for intraday
            timeframe_weights = self._optimize_intraday_timeframes(
                historical_data, market_analysis
            )
            
            # Optimize threshold parameters
            threshold_parameters = self._optimize_intraday_thresholds(
                historical_data, volatility_adjustments
            )
            
            # Optimize indicator weights for quick response
            indicator_weights = self._optimize_intraday_indicators(
                historical_data, market_analysis
            )
            
            # Calculate risk management parameters
            risk_parameters = self._calculate_intraday_risk_parameters(
                historical_data, current_market_conditions
            )
            
            # Compile optimized parameters
            optimized_parameters = {
                'trading_mode': 'intraday',
                'optimization_timestamp': datetime.now().isoformat(),
                'timeframe_configuration': {
                    'primary_timeframe': self.primary_timeframe,
                    'active_timeframes': self.target_timeframes,
                    'timeframe_weights': timeframe_weights
                },
                'threshold_parameters': threshold_parameters,
                'indicator_weights': indicator_weights,
                'volatility_adjustments': volatility_adjustments,
                'risk_management': risk_parameters,
                'market_analysis': market_analysis,
                'performance_target': {
                    'max_holding_period_minutes': self.max_holding_period_minutes,
                    'min_regime_duration_minutes': self.min_regime_duration_minutes,
                    'target_win_rate': 0.60,
                    'target_profit_factor': 1.5
                }
            }
            
            # Validate optimized parameters
            validation_result = self._validate_intraday_parameters(optimized_parameters)
            optimized_parameters['validation'] = validation_result
            
            # Track optimization
            self._track_optimization(optimized_parameters, market_analysis)
            
            logger.info("Intraday regime parameter optimization completed")
            return {
                'success': True,
                'optimized_parameters': optimized_parameters,
                'optimization_summary': self._create_optimization_summary(optimized_parameters)
            }
            
        except Exception as e:
            error_msg = f"Error optimizing intraday parameters: {e}"
            self.error_handler.handle_error(error_msg, e)
            return {
                'success': False,
                'error': error_msg
            }
    
    def _analyze_intraday_conditions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market conditions for intraday trading suitability
        """
        try:
            analysis = {
                'data_points': len(data),
                'analysis_timestamp': datetime.now()
            }
            
            if len(data) < 20:
                analysis['warning'] = 'Insufficient data for comprehensive analysis'
                return analysis
            
            # Calculate intraday volatility
            if 'close' in data.columns:
                returns = data['close'].pct_change().dropna()
                analysis['intraday_volatility'] = float(returns.std())
                analysis['avg_return'] = float(returns.mean())
                
                # Calculate intraday range
                if 'high' in data.columns and 'low' in data.columns:
                    intraday_range = (data['high'] - data['low']) / data['close']
                    analysis['avg_intraday_range'] = float(intraday_range.mean())
                    analysis['range_volatility'] = float(intraday_range.std())
            
            # Calculate volume patterns
            if 'volume' in data.columns:
                analysis['avg_volume'] = float(data['volume'].mean())
                analysis['volume_volatility'] = float(data['volume'].std())
                
                # Detect volume spikes (good for intraday)
                volume_threshold = analysis['avg_volume'] + 2 * analysis['volume_volatility']
                volume_spikes = (data['volume'] > volume_threshold).sum()
                analysis['volume_spike_frequency'] = float(volume_spikes / len(data))
            
            # Calculate regime change frequency
            if 'regime_name' in data.columns:
                regime_changes = (data['regime_name'] != data['regime_name'].shift(1)).sum()
                analysis['regime_change_frequency'] = float(regime_changes / len(data))
                analysis['avg_regime_duration'] = float(len(data) / max(regime_changes, 1))
            
            # Determine intraday suitability
            suitability_score = 0
            
            if analysis.get('intraday_volatility', 0) > 0.015:  # Good volatility for intraday
                suitability_score += 0.3
            
            if analysis.get('volume_spike_frequency', 0) > 0.1:  # Frequent volume spikes
                suitability_score += 0.2
            
            if analysis.get('regime_change_frequency', 0) > 0.1:  # Frequent regime changes
                suitability_score += 0.3
            
            if analysis.get('avg_intraday_range', 0) > 0.01:  # Good intraday movement
                suitability_score += 0.2
            
            analysis['intraday_suitability_score'] = suitability_score
            
            if suitability_score >= 0.7:
                analysis['suitability_level'] = 'excellent'
            elif suitability_score >= 0.5:
                analysis['suitability_level'] = 'good'
            elif suitability_score >= 0.3:
                analysis['suitability_level'] = 'fair'
            else:
                analysis['suitability_level'] = 'poor'
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing intraday conditions: {e}")
            return {
                'analysis_timestamp': datetime.now(),
                'error': str(e)
            }
    
    def _calculate_volatility_adjustments(self, data: pd.DataFrame, current_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate volatility-based parameter adjustments for intraday trading
        """
        try:
            adjustments = {}
            
            if 'close' in data.columns and len(data) > 10:
                # Calculate recent volatility
                recent_returns = data['close'].pct_change().tail(20).dropna()
                current_volatility = float(recent_returns.std())
                
                # Calculate historical volatility baseline
                all_returns = data['close'].pct_change().dropna()
                historical_volatility = float(all_returns.std())
                
                # Volatility ratio
                volatility_ratio = current_volatility / max(historical_volatility, 0.001)
                
                adjustments['current_volatility'] = current_volatility
                adjustments['historical_volatility'] = historical_volatility
                adjustments['volatility_ratio'] = volatility_ratio
                
                # Adjust parameters based on volatility
                if volatility_ratio > 1.5:  # High volatility period
                    adjustments['threshold_multiplier'] = 0.8  # Lower thresholds for quicker detection
                    adjustments['confidence_adjustment'] = -0.05  # Lower confidence requirement
                    adjustments['regime_duration_multiplier'] = 0.7  # Shorter minimum duration
                elif volatility_ratio < 0.7:  # Low volatility period
                    adjustments['threshold_multiplier'] = 1.2  # Higher thresholds to avoid noise
                    adjustments['confidence_adjustment'] = 0.05  # Higher confidence requirement
                    adjustments['regime_duration_multiplier'] = 1.3  # Longer minimum duration
                else:  # Normal volatility
                    adjustments['threshold_multiplier'] = 1.0
                    adjustments['confidence_adjustment'] = 0.0
                    adjustments['regime_duration_multiplier'] = 1.0
                
                # Position sizing adjustment
                adjustments['position_size_multiplier'] = min(1.0, 1.0 / max(volatility_ratio, 0.5))
                
            else:
                # Default adjustments
                adjustments = {
                    'threshold_multiplier': 1.0,
                    'confidence_adjustment': 0.0,
                    'regime_duration_multiplier': 1.0,
                    'position_size_multiplier': 1.0
                }
            
            return adjustments
            
        except Exception as e:
            logger.error(f"Error calculating volatility adjustments: {e}")
            return {
                'threshold_multiplier': 1.0,
                'confidence_adjustment': 0.0,
                'regime_duration_multiplier': 1.0,
                'position_size_multiplier': 1.0,
                'error': str(e)
            }
    
    def _optimize_intraday_timeframes(self, data: pd.DataFrame, market_analysis: Dict[str, Any]) -> Dict[str, float]:
        """
        Optimize timeframe weights for intraday trading
        """
        try:
            # Base intraday weights (favor shorter timeframes)
            base_weights = {
                '3min': 0.35,   # High weight for quick response
                '5min': 0.30,   # Primary timeframe
                '10min': 0.20,  # Medium-term confirmation
                '15min': 0.15   # Trend context
            }
            
            # Adjust based on market conditions
            volatility_level = market_analysis.get('intraday_volatility', 0.01)
            regime_change_freq = market_analysis.get('regime_change_frequency', 0.1)
            
            adjusted_weights = base_weights.copy()
            
            # High volatility: favor very short timeframes
            if volatility_level > 0.02:
                adjusted_weights['3min'] *= 1.3
                adjusted_weights['5min'] *= 1.1
                adjusted_weights['10min'] *= 0.9
                adjusted_weights['15min'] *= 0.7
            
            # Low volatility: slightly favor longer timeframes for stability
            elif volatility_level < 0.008:
                adjusted_weights['3min'] *= 0.8
                adjusted_weights['5min'] *= 0.9
                adjusted_weights['10min'] *= 1.2
                adjusted_weights['15min'] *= 1.3
            
            # High regime change frequency: favor shorter timeframes
            if regime_change_freq > 0.15:
                adjusted_weights['3min'] *= 1.2
                adjusted_weights['5min'] *= 1.1
            
            # Normalize weights
            total_weight = sum(adjusted_weights.values())
            if total_weight > 0:
                adjusted_weights = {tf: w/total_weight for tf, w in adjusted_weights.items()}
            
            return adjusted_weights
            
        except Exception as e:
            logger.error(f"Error optimizing intraday timeframes: {e}")
            # Return default weights
            return {
                '3min': 0.35,
                '5min': 0.30,
                '10min': 0.20,
                '15min': 0.15
            }
    
    def _optimize_intraday_thresholds(self, data: pd.DataFrame, volatility_adj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize detection thresholds for intraday trading
        """
        try:
            # Base intraday thresholds (more sensitive for quick detection)
            base_thresholds = {
                'directional_thresholds': {
                    'strong_bullish': 0.40,    # Lower than standard for quicker detection
                    'mild_bullish': 0.15,
                    'neutral': 0.05,
                    'mild_bearish': -0.15,
                    'strong_bearish': -0.40
                },
                'volatility_thresholds': {
                    'high': 0.65,             # Lower for intraday volatility patterns
                    'normal_high': 0.40,
                    'normal_low': 0.20,
                    'low': 0.10
                },
                'confidence_threshold': self.confidence_threshold
            }
            
            # Apply volatility adjustments
            threshold_multiplier = volatility_adj.get('threshold_multiplier', 1.0)
            confidence_adjustment = volatility_adj.get('confidence_adjustment', 0.0)
            
            # Adjust directional thresholds
            adjusted_thresholds = {
                'directional_thresholds': {},
                'volatility_thresholds': {},
                'confidence_threshold': max(0.5, min(0.9, base_thresholds['confidence_threshold'] + confidence_adjustment))
            }
            
            # Adjust directional thresholds
            for direction, threshold in base_thresholds['directional_thresholds'].items():
                if threshold > 0:
                    adjusted_thresholds['directional_thresholds'][direction] = threshold * threshold_multiplier
                else:
                    adjusted_thresholds['directional_thresholds'][direction] = threshold * threshold_multiplier
            
            # Adjust volatility thresholds
            for vol_level, threshold in base_thresholds['volatility_thresholds'].items():
                adjusted_thresholds['volatility_thresholds'][vol_level] = threshold * threshold_multiplier
            
            # Add intraday-specific parameters
            adjusted_thresholds['min_regime_duration_minutes'] = max(
                3, int(self.min_regime_duration_minutes * volatility_adj.get('regime_duration_multiplier', 1.0))
            )
            adjusted_thresholds['max_regime_duration_minutes'] = self.max_holding_period_minutes
            adjusted_thresholds['quick_response_mode'] = True
            
            return adjusted_thresholds
            
        except Exception as e:
            logger.error(f"Error optimizing intraday thresholds: {e}")
            return {
                'directional_thresholds': {
                    'strong_bullish': 0.40,
                    'mild_bullish': 0.15,
                    'neutral': 0.05,
                    'mild_bearish': -0.15,
                    'strong_bearish': -0.40
                },
                'volatility_thresholds': {
                    'high': 0.65,
                    'normal_high': 0.40,
                    'normal_low': 0.20,
                    'low': 0.10
                },
                'confidence_threshold': 0.65,
                'error': str(e)
            }
    
    def _optimize_intraday_indicators(self, data: pd.DataFrame, market_analysis: Dict[str, Any]) -> Dict[str, float]:
        """
        Optimize indicator weights for intraday trading
        """
        try:
            # Base intraday indicator weights (favor price action and volume)
            base_weights = {
                'price_action': 0.30,        # High weight for immediate price signals
                'volume_analysis': 0.25,     # Volume is crucial for intraday
                'technical_indicators': 0.20, # Technical signals for entry/exit
                'greek_sentiment': 0.15,     # Options flow for market direction
                'oi_analysis': 0.10          # Lower weight for longer-term OI signals
            }
            
            # Adjust based on market analysis
            volume_spike_freq = market_analysis.get('volume_spike_frequency', 0.1)
            volatility_level = market_analysis.get('intraday_volatility', 0.01)
            
            adjusted_weights = base_weights.copy()
            
            # High volume activity: increase volume analysis weight
            if volume_spike_freq > 0.15:
                adjusted_weights['volume_analysis'] *= 1.3
                adjusted_weights['price_action'] *= 0.9
            
            # High volatility: increase technical indicators weight
            if volatility_level > 0.02:
                adjusted_weights['technical_indicators'] *= 1.2
                adjusted_weights['greek_sentiment'] *= 1.1
            
            # Low volatility: increase price action weight
            elif volatility_level < 0.008:
                adjusted_weights['price_action'] *= 1.2
                adjusted_weights['volume_analysis'] *= 1.1
            
            # Normalize weights
            total_weight = sum(adjusted_weights.values())
            if total_weight > 0:
                adjusted_weights = {indicator: w/total_weight for indicator, w in adjusted_weights.items()}
            
            return adjusted_weights
            
        except Exception as e:
            logger.error(f"Error optimizing intraday indicators: {e}")
            # Return default weights
            return {
                'price_action': 0.30,
                'volume_analysis': 0.25,
                'technical_indicators': 0.20,
                'greek_sentiment': 0.15,
                'oi_analysis': 0.10
            }
    
    def _calculate_intraday_risk_parameters(self, data: pd.DataFrame, current_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate risk management parameters for intraday trading
        """
        try:
            risk_params = {
                'max_risk_per_trade': self.max_risk_per_trade,
                'position_sizing_method': self.position_sizing_method
            }
            
            if 'close' in data.columns and len(data) > 10:
                # Calculate ATR for position sizing
                if 'high' in data.columns and 'low' in data.columns:
                    true_range = np.maximum(
                        data['high'] - data['low'],
                        np.maximum(
                            abs(data['high'] - data['close'].shift(1)),
                            abs(data['low'] - data['close'].shift(1))
                        )
                    )
                    atr = true_range.rolling(window=14).mean().iloc[-1]
                    risk_params['atr'] = float(atr)
                    risk_params['atr_multiple_stop'] = 1.5  # Tighter stops for intraday
                
                # Calculate recent volatility for position sizing
                recent_returns = data['close'].pct_change().tail(20).dropna()
                if len(recent_returns) > 0:
                    volatility = recent_returns.std()
                    risk_params['recent_volatility'] = float(volatility)
                    risk_params['volatility_adjusted_position_size'] = min(1.0, 0.02 / max(volatility, 0.005))
            
            # Intraday-specific risk parameters
            risk_params.update({
                'max_trades_per_day': 8,      # Limit to avoid overtrading
                'max_drawdown_stop': 0.03,    # Daily drawdown limit
                'profit_target_multiple': 2.0, # Risk-reward ratio
                'trailing_stop_activation': 1.0, # Activate trailing stop at 1:1
                'time_based_exit': True,      # Exit before market close
                'exit_time_minutes_before_close': 15
            })
            
            return risk_params
            
        except Exception as e:
            logger.error(f"Error calculating intraday risk parameters: {e}")
            return {
                'max_risk_per_trade': self.max_risk_per_trade,
                'position_sizing_method': self.position_sizing_method,
                'error': str(e)
            }
    
    def _validate_intraday_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate optimized intraday parameters
        """
        validation = {
            'timestamp': datetime.now().isoformat(),
            'overall_valid': True,
            'issues': [],
            'warnings': []
        }
        
        try:
            # Validate timeframe weights
            timeframe_weights = parameters.get('timeframe_configuration', {}).get('timeframe_weights', {})
            if timeframe_weights:
                weight_sum = sum(timeframe_weights.values())
                if abs(weight_sum - 1.0) > 0.01:
                    validation['issues'].append(f"Timeframe weights sum to {weight_sum:.3f}, should be 1.0")
                    validation['overall_valid'] = False
            
            # Validate thresholds
            directional_thresholds = parameters.get('threshold_parameters', {}).get('directional_thresholds', {})
            if directional_thresholds:
                if directional_thresholds.get('strong_bullish', 0) <= directional_thresholds.get('mild_bullish', 0):
                    validation['issues'].append("Strong bullish threshold should be greater than mild bullish")
                    validation['overall_valid'] = False
            
            # Validate confidence threshold
            confidence = parameters.get('threshold_parameters', {}).get('confidence_threshold', 0.5)
            if confidence < 0.5 or confidence > 0.9:
                validation['warnings'].append(f"Confidence threshold {confidence:.2f} is outside recommended range [0.5, 0.9]")
            
            # Validate indicator weights
            indicator_weights = parameters.get('indicator_weights', {})
            if indicator_weights:
                weight_sum = sum(indicator_weights.values())
                if abs(weight_sum - 1.0) > 0.01:
                    validation['issues'].append(f"Indicator weights sum to {weight_sum:.3f}, should be 1.0")
                    validation['overall_valid'] = False
            
            # Validate risk parameters
            risk_params = parameters.get('risk_management', {})
            max_risk = risk_params.get('max_risk_per_trade', 0.02)
            if max_risk > 0.05:
                validation['warnings'].append(f"Max risk per trade {max_risk:.1%} is high for intraday trading")
            
        except Exception as e:
            validation['overall_valid'] = False
            validation['issues'].append(f"Validation error: {e}")
        
        return validation
    
    def _track_optimization(self, parameters: Dict[str, Any], market_analysis: Dict[str, Any]):
        """
        Track optimization for performance analysis
        """
        optimization_record = {
            'timestamp': datetime.now(),
            'parameters': parameters.copy(),
            'market_conditions': market_analysis.copy()
        }
        
        self.optimization_history.append(optimization_record)
        
        # Keep only recent history
        if len(self.optimization_history) > 50:
            self.optimization_history = self.optimization_history[-50:]
    
    def _create_optimization_summary(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create summary of optimization results
        """
        return {
            'trading_mode': 'intraday',
            'optimization_timestamp': parameters.get('optimization_timestamp'),
            'primary_timeframe': parameters.get('timeframe_configuration', {}).get('primary_timeframe'),
            'active_timeframes': len(parameters.get('timeframe_configuration', {}).get('active_timeframes', [])),
            'confidence_threshold': parameters.get('threshold_parameters', {}).get('confidence_threshold'),
            'validation_status': parameters.get('validation', {}).get('overall_valid', False),
            'optimization_count': len(self.optimization_history)
        }
    
    def get_optimizer_status(self) -> Dict[str, Any]:
        """
        Get current status of the intraday optimizer
        """
        return {
            'optimizer_type': 'intraday',
            'target_timeframes': self.target_timeframes,
            'primary_timeframe': self.primary_timeframe,
            'max_holding_period_minutes': self.max_holding_period_minutes,
            'min_regime_duration_minutes': self.min_regime_duration_minutes,
            'confidence_threshold': self.confidence_threshold,
            'optimization_history_count': len(self.optimization_history),
            'last_optimization': self.optimization_history[-1]['timestamp'].isoformat() if self.optimization_history else None
        }


class IntradayOptimizationError(Exception):
    """Custom exception for intraday optimization errors"""
    pass