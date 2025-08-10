#!/usr/bin/env python3
"""
Timeframe Coordinator for Market Regime Analysis
================================================

Integrates the existing AdaptiveTimeframeManager into the modular structure
with enhanced coordination capabilities for different trading modes.

Features:
- Integration with existing AdaptiveTimeframeManager
- Enhanced mode switching and coordination
- Dynamic timeframe weight optimization
- Performance-based timeframe selection
- Multi-mode support (intraday/positional/hybrid)
- Excel configuration integration

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Import existing adaptive timeframe manager
from ...adaptive_timeframe_manager import AdaptiveTimeframeManager
from ..common_utils import ErrorHandler, PerformanceTimer, MathUtils

logger = logging.getLogger(__name__)


class TimeframeCoordinator:
    """
    Enhanced coordinator that integrates AdaptiveTimeframeManager into modular structure
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Timeframe Coordinator"""
        self.config = config
        self.current_mode = config.get('trading_mode', 'hybrid')
        
        # Initialize existing adaptive timeframe manager
        self.timeframe_manager = AdaptiveTimeframeManager(
            mode=self.current_mode,
            config=config.get('timeframe_config')
        )
        
        # Initialize utilities
        self.error_handler = ErrorHandler()
        self.performance_timer = PerformanceTimer()
        self.math_utils = MathUtils()
        
        # Enhanced coordination features
        self.mode_performance_history = {}
        self.dynamic_optimization_enabled = config.get('dynamic_optimization', True)
        self.performance_window = config.get('performance_window', 100)
        self.min_confidence_threshold = config.get('min_confidence_threshold', 0.65)
        
        # Mode transition tracking
        self.mode_transitions = []
        self.last_mode_switch = None
        self.mode_switch_cooldown = config.get('mode_switch_cooldown_minutes', 30)
        
        logger.info(f"Timeframe Coordinator initialized with mode: {self.current_mode}")
    
    def optimize_for_trading_mode(
        self,
        target_mode: str,
        market_data: pd.DataFrame,
        performance_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Optimize timeframe configuration for specific trading mode
        
        Args:
            target_mode: Target trading mode ('intraday', 'positional', 'hybrid')
            market_data: Historical market data for optimization
            performance_metrics: Performance metrics for current configuration
            
        Returns:
            Dictionary with optimization results
        """
        try:
            logger.info(f"Optimizing timeframe configuration for {target_mode} mode")
            
            # Check if mode switch is allowed
            if not self._can_switch_mode(target_mode):
                return {
                    'success': False,
                    'reason': 'Mode switch not allowed due to cooldown period',
                    'current_mode': self.current_mode,
                    'cooldown_remaining': self._get_cooldown_remaining()
                }
            
            # Analyze market conditions for optimal timeframe selection
            market_analysis = self._analyze_market_conditions(market_data)
            
            # Get suggested mode based on market analysis
            suggested_mode = self._suggest_optimal_mode(market_analysis, target_mode)
            
            # Update performance history if metrics provided
            if performance_metrics:
                self._update_performance_history(self.current_mode, performance_metrics)
            
            # Switch to target mode
            switch_result = self._execute_mode_switch(suggested_mode, market_analysis)
            
            if switch_result['success']:
                # Optimize weights based on market conditions
                if self.dynamic_optimization_enabled:
                    optimization_result = self._optimize_timeframe_weights(
                        market_analysis, performance_metrics
                    )
                    switch_result['weight_optimization'] = optimization_result
                
                # Track mode transition
                self._track_mode_transition(self.current_mode, suggested_mode, market_analysis)
                
                # Update current mode
                self.current_mode = suggested_mode
                self.last_mode_switch = datetime.now()
            
            return switch_result
            
        except Exception as e:
            error_msg = f"Error optimizing for trading mode {target_mode}: {e}"
            self.error_handler.handle_error(error_msg, e)
            return {
                'success': False,
                'error': error_msg,
                'current_mode': self.current_mode
            }
    
    def get_optimal_timeframes(
        self,
        mode: Optional[str] = None,
        market_conditions: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get optimal timeframes for current or specified mode
        
        Args:
            mode: Trading mode (defaults to current mode)
            market_conditions: Current market conditions for optimization
            
        Returns:
            Dictionary with optimal timeframe configuration
        """
        try:
            target_mode = mode or self.current_mode
            
            # Get active timeframes from timeframe manager
            active_timeframes = self.timeframe_manager.get_active_timeframes()
            timeframe_weights = self.timeframe_manager.get_timeframe_weights()
            mode_parameters = self.timeframe_manager.get_mode_parameters()
            
            # Enhance with market condition adjustments
            if market_conditions:
                adjusted_weights = self._adjust_weights_for_conditions(
                    timeframe_weights, market_conditions
                )
                timeframe_weights = adjusted_weights
            
            return {
                'mode': target_mode,
                'active_timeframes': active_timeframes,
                'timeframe_weights': timeframe_weights,
                'mode_parameters': mode_parameters,
                'optimization_enabled': self.dynamic_optimization_enabled,
                'last_optimization': self.last_mode_switch.isoformat() if self.last_mode_switch else None
            }
            
        except Exception as e:
            error_msg = f"Error getting optimal timeframes: {e}"
            self.error_handler.handle_error(error_msg, e)
            return {
                'mode': self.current_mode,
                'active_timeframes': [],
                'timeframe_weights': {},
                'error': error_msg
            }
    
    def update_timeframe_performance(
        self,
        timeframe_performance: Dict[str, float],
        update_weights: bool = True
    ) -> Dict[str, Any]:
        """
        Update timeframe performance metrics and optionally adjust weights
        
        Args:
            timeframe_performance: Dictionary of timeframe -> performance score
            update_weights: Whether to update weights based on performance
            
        Returns:
            Dictionary with update results
        """
        try:
            # Update performance in timeframe manager
            self.timeframe_manager.optimize_weights_from_performance(timeframe_performance)
            
            # Store performance history
            performance_entry = {
                'timestamp': datetime.now(),
                'mode': self.current_mode,
                'performance': timeframe_performance.copy(),
                'weights_updated': update_weights
            }
            
            if self.current_mode not in self.mode_performance_history:
                self.mode_performance_history[self.current_mode] = []
            
            self.mode_performance_history[self.current_mode].append(performance_entry)
            
            # Keep only recent history
            if len(self.mode_performance_history[self.current_mode]) > self.performance_window:
                self.mode_performance_history[self.current_mode] = \
                    self.mode_performance_history[self.current_mode][-self.performance_window:]
            
            # Calculate performance summary
            performance_summary = self._calculate_performance_summary(timeframe_performance)
            
            logger.info(f"Updated timeframe performance for {self.current_mode} mode")
            
            return {
                'success': True,
                'mode': self.current_mode,
                'performance_summary': performance_summary,
                'weights_updated': update_weights,
                'current_weights': self.timeframe_manager.get_timeframe_weights()
            }
            
        except Exception as e:
            error_msg = f"Error updating timeframe performance: {e}"
            self.error_handler.handle_error(error_msg, e)
            return {
                'success': False,
                'error': error_msg
            }
    
    def validate_timeframe_data(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Validate timeframe data sufficiency for current mode
        
        Args:
            data: Dictionary of timeframe -> DataFrame
            
        Returns:
            Dictionary with validation results
        """
        try:
            validation_results = {
                'mode': self.current_mode,
                'timeframe_validations': {},
                'overall_valid': True,
                'warnings': []
            }
            
            active_timeframes = self.timeframe_manager.get_active_timeframes()
            
            for timeframe in active_timeframes:
                if timeframe in data:
                    timeframe_data = data[timeframe]
                    data_points = len(timeframe_data)
                    
                    # Validate using timeframe manager
                    is_valid = self.timeframe_manager.validate_timeframe_data(
                        timeframe, data_points
                    )
                    
                    validation_results['timeframe_validations'][timeframe] = {
                        'valid': is_valid,
                        'data_points': data_points,
                        'required_points': self.timeframe_manager.get_timeframe_config(timeframe).min_data_points if self.timeframe_manager.get_timeframe_config(timeframe) else 1
                    }
                    
                    if not is_valid:
                        validation_results['overall_valid'] = False
                        validation_results['warnings'].append(
                            f"Insufficient data for {timeframe}: {data_points} points"
                        )
                else:
                    validation_results['timeframe_validations'][timeframe] = {
                        'valid': False,
                        'data_points': 0,
                        'error': 'No data provided'
                    }
                    validation_results['overall_valid'] = False
                    validation_results['warnings'].append(f"No data provided for {timeframe}")
            
            return validation_results
            
        except Exception as e:
            error_msg = f"Error validating timeframe data: {e}"
            self.error_handler.handle_error(error_msg, e)
            return {
                'overall_valid': False,
                'error': error_msg
            }
    
    def _analyze_market_conditions(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market conditions to suggest optimal timeframe configuration
        """
        try:
            analysis = {
                'timestamp': datetime.now(),
                'data_points': len(market_data)
            }
            
            if len(market_data) < 10:
                analysis['warning'] = 'Insufficient data for analysis'
                return analysis
            
            # Calculate volatility
            if 'close' in market_data.columns:
                returns = market_data['close'].pct_change().dropna()
                analysis['volatility'] = float(returns.std())
                analysis['avg_return'] = float(returns.mean())
            
            # Calculate trading activity
            if 'volume' in market_data.columns:
                analysis['avg_volume'] = float(market_data['volume'].mean())
                analysis['volume_volatility'] = float(market_data['volume'].std())
            
            # Determine market regime characteristics
            if 'volatility' in analysis:
                if analysis['volatility'] > 0.02:
                    analysis['volatility_regime'] = 'high'
                elif analysis['volatility'] > 0.01:
                    analysis['volatility_regime'] = 'medium'
                else:
                    analysis['volatility_regime'] = 'low'
            
            # Suggest holding period based on volatility
            if analysis.get('volatility_regime') == 'high':
                analysis['suggested_holding_period'] = 'short'  # Intraday
            elif analysis.get('volatility_regime') == 'low':
                analysis['suggested_holding_period'] = 'long'   # Positional
            else:
                analysis['suggested_holding_period'] = 'medium' # Hybrid
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {e}")
            return {
                'timestamp': datetime.now(),
                'error': str(e)
            }
    
    def _suggest_optimal_mode(self, market_analysis: Dict[str, Any], target_mode: str) -> str:
        """
        Suggest optimal trading mode based on market analysis and target
        """
        try:
            # If market analysis suggests a specific mode, consider it
            if 'suggested_holding_period' in market_analysis:
                suggested_period = market_analysis['suggested_holding_period']
                
                if suggested_period == 'short':
                    market_suggested = 'intraday'
                elif suggested_period == 'long':
                    market_suggested = 'positional'
                else:
                    market_suggested = 'hybrid'
                
                # Use market suggestion if target is 'auto' or if it aligns
                if target_mode == 'auto':
                    return market_suggested
                elif target_mode == market_suggested:
                    return target_mode
                else:
                    # Check if target mode conflicts with market conditions
                    volatility_regime = market_analysis.get('volatility_regime', 'medium')
                    
                    if target_mode == 'intraday' and volatility_regime == 'low':
                        logger.warning("Intraday mode requested but low volatility detected")
                    elif target_mode == 'positional' and volatility_regime == 'high':
                        logger.warning("Positional mode requested but high volatility detected")
                    
                    return target_mode
            
            # Default to target mode if no market suggestion
            return target_mode if target_mode != 'auto' else 'hybrid'
            
        except Exception as e:
            logger.error(f"Error suggesting optimal mode: {e}")
            return 'hybrid'
    
    def _can_switch_mode(self, target_mode: str) -> bool:
        """
        Check if mode switch is allowed based on cooldown period
        """
        if target_mode == self.current_mode:
            return True  # Same mode, no switch needed
        
        if not self.last_mode_switch:
            return True  # First switch
        
        time_since_switch = datetime.now() - self.last_mode_switch
        cooldown_period = timedelta(minutes=self.mode_switch_cooldown)
        
        return time_since_switch >= cooldown_period
    
    def _get_cooldown_remaining(self) -> Optional[int]:
        """
        Get remaining cooldown time in minutes
        """
        if not self.last_mode_switch:
            return None
        
        time_since_switch = datetime.now() - self.last_mode_switch
        cooldown_period = timedelta(minutes=self.mode_switch_cooldown)
        
        if time_since_switch >= cooldown_period:
            return None
        
        remaining = cooldown_period - time_since_switch
        return int(remaining.total_seconds() / 60)
    
    def _execute_mode_switch(self, target_mode: str, market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute mode switch using timeframe manager
        """
        try:
            # Prepare custom configuration if needed
            custom_config = None
            if target_mode == 'custom' and 'timeframe_config' in self.config:
                custom_config = self.config['timeframe_config']
            
            # Execute switch
            switch_success = self.timeframe_manager.switch_mode(target_mode, custom_config)
            
            if switch_success:
                return {
                    'success': True,
                    'previous_mode': self.current_mode,
                    'new_mode': target_mode,
                    'switch_timestamp': datetime.now().isoformat(),
                    'market_conditions': market_analysis,
                    'new_configuration': self.timeframe_manager.get_mode_parameters()
                }
            else:
                return {
                    'success': False,
                    'reason': 'Timeframe manager failed to switch mode',
                    'current_mode': self.current_mode
                }
                
        except Exception as e:
            logger.error(f"Error executing mode switch: {e}")
            return {
                'success': False,
                'error': str(e),
                'current_mode': self.current_mode
            }
    
    def _optimize_timeframe_weights(self, market_analysis: Dict[str, Any], performance_metrics: Optional[Dict[str, float]]) -> Dict[str, Any]:
        """
        Optimize timeframe weights based on market conditions and performance
        """
        try:
            current_weights = self.timeframe_manager.get_timeframe_weights()
            
            # If performance metrics provided, use them for optimization
            if performance_metrics:
                self.timeframe_manager.optimize_weights_from_performance(performance_metrics)
                optimized_weights = self.timeframe_manager.get_timeframe_weights()
                
                return {
                    'method': 'performance_based',
                    'original_weights': current_weights,
                    'optimized_weights': optimized_weights,
                    'performance_metrics': performance_metrics
                }
            
            # Otherwise, use market condition-based optimization
            volatility_regime = market_analysis.get('volatility_regime', 'medium')
            
            # Simple volatility-based weight adjustment
            adjusted_weights = current_weights.copy()
            
            if volatility_regime == 'high':
                # Favor shorter timeframes for high volatility
                for tf in ['3min', '5min']:
                    if tf in adjusted_weights:
                        adjusted_weights[tf] *= 1.2
            elif volatility_regime == 'low':
                # Favor longer timeframes for low volatility
                for tf in ['15min', '30min', '1hr']:
                    if tf in adjusted_weights:
                        adjusted_weights[tf] *= 1.2
            
            # Normalize weights
            total_weight = sum(adjusted_weights.values())
            if total_weight > 0:
                adjusted_weights = {tf: w/total_weight for tf, w in adjusted_weights.items()}
                
                # Apply adjusted weights
                for tf, weight in adjusted_weights.items():
                    self.timeframe_manager.update_timeframe_weight(tf, weight)
            
            return {
                'method': 'market_condition_based',
                'original_weights': current_weights,
                'optimized_weights': adjusted_weights,
                'volatility_regime': volatility_regime
            }
            
        except Exception as e:
            logger.error(f"Error optimizing timeframe weights: {e}")
            return {
                'method': 'error',
                'error': str(e)
            }
    
    def _adjust_weights_for_conditions(self, weights: Dict[str, float], conditions: Dict[str, Any]) -> Dict[str, float]:
        """
        Adjust timeframe weights based on current market conditions
        """
        adjusted_weights = weights.copy()
        
        try:
            volatility = conditions.get('volatility', 0.01)
            
            # Adjust based on volatility
            if volatility > 0.025:  # High volatility
                # Increase weight for shorter timeframes
                for tf in ['3min', '5min']:
                    if tf in adjusted_weights:
                        adjusted_weights[tf] *= 1.3
            elif volatility < 0.005:  # Low volatility
                # Increase weight for longer timeframes
                for tf in ['30min', '1hr']:
                    if tf in adjusted_weights:
                        adjusted_weights[tf] *= 1.3
            
            # Normalize weights
            total = sum(adjusted_weights.values())
            if total > 0:
                adjusted_weights = {tf: w/total for tf, w in adjusted_weights.items()}
            
            return adjusted_weights
            
        except Exception as e:
            logger.error(f"Error adjusting weights for conditions: {e}")
            return weights
    
    def _update_performance_history(self, mode: str, performance_metrics: Dict[str, float]):
        """
        Update performance history for the given mode
        """
        if mode not in self.mode_performance_history:
            self.mode_performance_history[mode] = []
        
        entry = {
            'timestamp': datetime.now(),
            'performance': performance_metrics.copy()
        }
        
        self.mode_performance_history[mode].append(entry)
        
        # Keep only recent history
        if len(self.mode_performance_history[mode]) > self.performance_window:
            self.mode_performance_history[mode] = self.mode_performance_history[mode][-self.performance_window:]
    
    def _track_mode_transition(self, from_mode: str, to_mode: str, market_analysis: Dict[str, Any]):
        """
        Track mode transition for analysis
        """
        transition = {
            'timestamp': datetime.now(),
            'from_mode': from_mode,
            'to_mode': to_mode,
            'market_conditions': market_analysis.copy()
        }
        
        self.mode_transitions.append(transition)
        
        # Keep only recent transitions
        if len(self.mode_transitions) > 50:
            self.mode_transitions = self.mode_transitions[-50:]
    
    def _calculate_performance_summary(self, performance: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate performance summary statistics
        """
        if not performance:
            return {}
        
        values = list(performance.values())
        
        return {
            'average_performance': float(np.mean(values)),
            'best_timeframe': max(performance.keys(), key=lambda k: performance[k]),
            'worst_timeframe': min(performance.keys(), key=lambda k: performance[k]),
            'performance_spread': float(max(values) - min(values)),
            'total_timeframes': len(performance)
        }
    
    def get_coordinator_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of the timeframe coordinator
        """
        return {
            'current_mode': self.current_mode,
            'last_mode_switch': self.last_mode_switch.isoformat() if self.last_mode_switch else None,
            'cooldown_remaining': self._get_cooldown_remaining(),
            'dynamic_optimization_enabled': self.dynamic_optimization_enabled,
            'active_timeframes': self.timeframe_manager.get_active_timeframes(),
            'current_weights': self.timeframe_manager.get_timeframe_weights(),
            'mode_transitions_count': len(self.mode_transitions),
            'performance_history_modes': list(self.mode_performance_history.keys()),
            'timeframe_manager_config': self.timeframe_manager.get_mode_parameters()
        }


class TimeframeCoordinationError(Exception):
    """Custom exception for timeframe coordination errors"""
    pass