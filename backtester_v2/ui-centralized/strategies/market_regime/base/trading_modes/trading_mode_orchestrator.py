#!/usr/bin/env python3
"""
Trading Mode Orchestrator for Market Regime Analysis
====================================================

Central orchestrator for all trading modes, coordinating intraday,
positional, and hybrid optimizers with the timeframe coordinator.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from .intraday_optimizer import IntradayOptimizer
from .positional_optimizer import PositionalOptimizer
from .hybrid_mode_manager import HybridModeManager
from .timeframe_coordinator import TimeframeCoordinator
from ..common_utils import ErrorHandler, PerformanceTimer

logger = logging.getLogger(__name__)


class TradingModeOrchestrator:
    """
    Central orchestrator for all trading modes and timeframe coordination
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Trading Mode Orchestrator"""
        self.config = config
        self.current_mode = config.get('trading_mode', 'hybrid')
        
        # Initialize all optimizers
        self.intraday_optimizer = IntradayOptimizer(config)
        self.positional_optimizer = PositionalOptimizer(config)
        self.hybrid_manager = HybridModeManager(config)
        self.timeframe_coordinator = TimeframeCoordinator(config)
        
        # Initialize utilities
        self.error_handler = ErrorHandler()
        self.performance_timer = PerformanceTimer()
        
        # Orchestration tracking
        self.optimization_history = []
        self.mode_switches = []
        
        logger.info(f"Trading Mode Orchestrator initialized with mode: {self.current_mode}")
    
    def optimize_for_current_mode(
        self,
        historical_data: pd.DataFrame,
        market_data: pd.DataFrame,
        performance_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Optimize parameters for current trading mode
        """
        try:
            with self.performance_timer.measure('mode_optimization'):
                # Get current market conditions
                current_conditions = self._analyze_current_conditions(market_data)
                
                # Optimize timeframe configuration first
                timeframe_result = self.timeframe_coordinator.optimize_for_trading_mode(
                    self.current_mode, historical_data, performance_metrics
                )
                
                # Optimize mode-specific parameters
                if self.current_mode == 'intraday':
                    mode_result = self.intraday_optimizer.optimize_regime_parameters(
                        historical_data, current_conditions
                    )
                elif self.current_mode == 'positional':
                    mode_result = self.positional_optimizer.optimize_regime_parameters(
                        historical_data, current_conditions
                    )
                elif self.current_mode == 'hybrid':
                    mode_result = self.hybrid_manager.optimize_hybrid_parameters(
                        historical_data, performance_metrics
                    )
                else:
                    raise ValueError(f"Unknown trading mode: {self.current_mode}")
                
                # Combine results
                combined_result = self._combine_optimization_results(
                    timeframe_result, mode_result, current_conditions
                )
                
                # Track optimization
                self._track_optimization(combined_result)
                
                return combined_result
                
        except Exception as e:
            error_msg = f"Error optimizing for mode {self.current_mode}: {e}"
            self.error_handler.handle_error(error_msg, e)
            return {
                'success': False,
                'error': error_msg,
                'current_mode': self.current_mode
            }
    
    def switch_trading_mode(
        self,
        target_mode: str,
        historical_data: pd.DataFrame,
        force_switch: bool = False
    ) -> Dict[str, Any]:
        """
        Switch to a different trading mode
        """
        try:
            logger.info(f"Attempting to switch from {self.current_mode} to {target_mode}")
            
            if target_mode == self.current_mode and not force_switch:
                return {
                    'success': True,
                    'message': f"Already in {target_mode} mode",
                    'current_mode': self.current_mode
                }
            
            # Validate target mode
            valid_modes = ['intraday', 'positional', 'hybrid']
            if target_mode not in valid_modes:
                return {
                    'success': False,
                    'error': f"Invalid mode {target_mode}. Valid modes: {valid_modes}",
                    'current_mode': self.current_mode
                }
            
            # Check if switch is advisable based on market conditions
            market_conditions = self._analyze_current_conditions(historical_data)
            switch_recommendation = self._evaluate_mode_switch(
                self.current_mode, target_mode, market_conditions
            )
            
            if not switch_recommendation['advisable'] and not force_switch:
                return {
                    'success': False,
                    'reason': switch_recommendation['reason'],
                    'recommendation': switch_recommendation,
                    'current_mode': self.current_mode
                }
            
            # Execute mode switch
            previous_mode = self.current_mode
            self.current_mode = target_mode
            
            # Update timeframe coordinator
            timeframe_update = self.timeframe_coordinator.optimize_for_trading_mode(
                target_mode, historical_data
            )
            
            # Track mode switch
            self._track_mode_switch(previous_mode, target_mode, market_conditions)
            
            logger.info(f"Successfully switched from {previous_mode} to {target_mode}")
            
            return {
                'success': True,
                'previous_mode': previous_mode,
                'new_mode': target_mode,
                'switch_timestamp': datetime.now().isoformat(),
                'timeframe_update': timeframe_update,
                'market_conditions': market_conditions
            }
            
        except Exception as e:
            error_msg = f"Error switching to mode {target_mode}: {e}"
            self.error_handler.handle_error(error_msg, e)
            return {
                'success': False,
                'error': error_msg,
                'current_mode': self.current_mode
            }
    
    def get_optimal_configuration(self) -> Dict[str, Any]:
        """
        Get current optimal configuration for active mode
        """
        try:
            # Get timeframe configuration
            timeframe_config = self.timeframe_coordinator.get_optimal_timeframes(self.current_mode)
            
            # Get mode-specific status
            if self.current_mode == 'intraday':
                mode_status = self.intraday_optimizer.get_optimizer_status()
            elif self.current_mode == 'positional':
                mode_status = self.positional_optimizer.get_optimizer_status()
            elif self.current_mode == 'hybrid':
                mode_status = self.hybrid_manager.get_manager_status()
            else:
                mode_status = {'error': f'Unknown mode: {self.current_mode}'}
            
            return {
                'current_mode': self.current_mode,
                'timeframe_configuration': timeframe_config,
                'mode_specific_status': mode_status,
                'orchestrator_status': self.get_orchestrator_status()
            }
            
        except Exception as e:
            error_msg = f"Error getting optimal configuration: {e}"
            self.error_handler.handle_error(error_msg, e)
            return {
                'current_mode': self.current_mode,
                'error': error_msg
            }
    
    def _analyze_current_conditions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze current market conditions"""
        conditions = {'timestamp': datetime.now()}
        
        if len(data) > 10 and 'close' in data.columns:
            recent_data = data.tail(20)
            returns = recent_data['close'].pct_change().dropna()
            
            conditions.update({
                'volatility': float(returns.std()),
                'trend': float(returns.mean()),
                'data_points': len(data)
            })
            
            # Add volume analysis if available
            if 'volume' in recent_data.columns:
                conditions['avg_volume'] = float(recent_data['volume'].mean())
        
        return conditions
    
    def _combine_optimization_results(
        self,
        timeframe_result: Dict[str, Any],
        mode_result: Dict[str, Any],
        conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine timeframe and mode optimization results"""
        if not timeframe_result.get('success', False) or not mode_result.get('success', False):
            return {
                'success': False,
                'timeframe_result': timeframe_result,
                'mode_result': mode_result,
                'error': 'One or both optimization steps failed'
            }
        
        # Extract parameters from both results
        optimized_params = mode_result.get('optimized_parameters', {})
        
        # Update with timeframe coordinator results
        if 'new_configuration' in timeframe_result:
            optimized_params['timeframe_configuration'].update(
                timeframe_result['new_configuration']
            )
        
        return {
            'success': True,
            'trading_mode': self.current_mode,
            'optimization_timestamp': datetime.now().isoformat(),
            'optimized_parameters': optimized_params,
            'timeframe_optimization': timeframe_result,
            'mode_optimization': mode_result,
            'market_conditions': conditions,
            'performance_metrics': self.performance_timer.get_stats()
        }
    
    def _evaluate_mode_switch(self, current: str, target: str, conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate if mode switch is advisable"""
        recommendation = {
            'advisable': True,
            'reason': 'Mode switch approved',
            'confidence': 0.5
        }
        
        volatility = conditions.get('volatility', 0.01)
        
        # Evaluate based on market conditions
        if target == 'intraday' and volatility < 0.005:
            recommendation.update({
                'advisable': False,
                'reason': 'Low volatility not suitable for intraday trading',
                'confidence': 0.2
            })
        elif target == 'positional' and volatility > 0.03:
            recommendation.update({
                'advisable': False,
                'reason': 'High volatility may not be suitable for positional trading',
                'confidence': 0.3
            })
        
        return recommendation
    
    def _track_optimization(self, result: Dict[str, Any]):
        """Track optimization results"""
        self.optimization_history.append({
            'timestamp': datetime.now(),
            'mode': self.current_mode,
            'result': result.copy()
        })
        
        if len(self.optimization_history) > 100:
            self.optimization_history = self.optimization_history[-100:]
    
    def _track_mode_switch(self, from_mode: str, to_mode: str, conditions: Dict[str, Any]):
        """Track mode switches"""
        self.mode_switches.append({
            'timestamp': datetime.now(),
            'from_mode': from_mode,
            'to_mode': to_mode,
            'conditions': conditions.copy()
        })
        
        if len(self.mode_switches) > 50:
            self.mode_switches = self.mode_switches[-50:]
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator status"""
        return {
            'current_mode': self.current_mode,
            'optimization_count': len(self.optimization_history),
            'mode_switches_count': len(self.mode_switches),
            'last_optimization': self.optimization_history[-1]['timestamp'].isoformat() if self.optimization_history else None,
            'last_mode_switch': self.mode_switches[-1]['timestamp'].isoformat() if self.mode_switches else None,
            'available_modes': ['intraday', 'positional', 'hybrid'],
            'performance_stats': self.performance_timer.get_stats()
        }


class TradingModeOrchestrationError(Exception):
    """Custom exception for trading mode orchestration errors"""
    pass