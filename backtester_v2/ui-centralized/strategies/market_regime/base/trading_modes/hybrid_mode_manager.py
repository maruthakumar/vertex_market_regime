#!/usr/bin/env python3
"""
Hybrid Mode Manager for Market Regime Analysis
==============================================

Manages hybrid trading mode that combines intraday and positional
strategies based on market conditions and performance metrics.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from ..common_utils import MathUtils, ErrorHandler

logger = logging.getLogger(__name__)


class HybridModeManager:
    """
    Manages hybrid trading mode with dynamic switching between intraday and positional
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Hybrid Mode Manager"""
        self.config = config
        self.current_sub_mode = 'balanced'  # balanced, intraday_bias, positional_bias
        self.dynamic_switching_enabled = config.get('dynamic_switching', True)
        self.performance_window = config.get('performance_window', 20)
        
        self.math_utils = MathUtils()
        self.error_handler = ErrorHandler()
        self.mode_history = []
        
        logger.info("Hybrid Mode Manager initialized")
    
    def optimize_hybrid_parameters(self, historical_data: pd.DataFrame, performance_metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Optimize parameters for hybrid trading mode
        """
        try:
            # Analyze market conditions to determine optimal hybrid configuration
            market_analysis = self._analyze_hybrid_conditions(historical_data)
            
            # Determine sub-mode based on conditions
            optimal_sub_mode = self._determine_optimal_sub_mode(market_analysis, performance_metrics)
            
            # Configure timeframes based on sub-mode
            timeframe_config = self._configure_hybrid_timeframes(optimal_sub_mode)
            
            # Configure thresholds
            threshold_config = self._configure_hybrid_thresholds(optimal_sub_mode)
            
            # Configure indicator weights
            indicator_config = self._configure_hybrid_indicators(optimal_sub_mode)
            
            optimized_parameters = {
                'trading_mode': 'hybrid',
                'sub_mode': optimal_sub_mode,
                'optimization_timestamp': datetime.now().isoformat(),
                'timeframe_configuration': timeframe_config,
                'threshold_parameters': threshold_config,
                'indicator_weights': indicator_config,
                'market_analysis': market_analysis,
                'dynamic_switching_enabled': self.dynamic_switching_enabled
            }
            
            self._track_mode_change(optimal_sub_mode, market_analysis)
            self.current_sub_mode = optimal_sub_mode
            
            return {
                'success': True,
                'optimized_parameters': optimized_parameters
            }
            
        except Exception as e:
            error_msg = f"Error optimizing hybrid parameters: {e}"
            self.error_handler.handle_error(error_msg, e)
            return {'success': False, 'error': error_msg}
    
    def _analyze_hybrid_conditions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market conditions for hybrid mode optimization"""
        analysis = {'analysis_timestamp': datetime.now()}
        
        if len(data) < 20:
            return analysis
        
        if 'close' in data.columns:
            # Calculate multiple metrics for hybrid decision
            returns = data['close'].pct_change().dropna()
            analysis['volatility'] = float(returns.std())
            analysis['trend_strength'] = float(abs(returns.mean()))
            
            # Calculate regime stability
            if 'regime_name' in data.columns:
                regime_changes = (data['regime_name'] != data['regime_name'].shift(1)).sum()
                analysis['regime_stability'] = float(1 - (regime_changes / len(data)))
            
            # Determine market character
            if analysis['volatility'] > 0.02 and analysis.get('regime_stability', 0.5) < 0.7:
                analysis['market_character'] = 'volatile_changing'
                analysis['recommended_bias'] = 'intraday_bias'
            elif analysis['volatility'] < 0.01 and analysis.get('regime_stability', 0.5) > 0.8:
                analysis['market_character'] = 'stable_trending'
                analysis['recommended_bias'] = 'positional_bias'
            else:
                analysis['market_character'] = 'mixed_conditions'
                analysis['recommended_bias'] = 'balanced'
        
        return analysis
    
    def _determine_optimal_sub_mode(self, market_analysis: Dict[str, Any], performance_metrics: Optional[Dict[str, float]]) -> str:
        """Determine optimal sub-mode for current conditions"""
        # Start with market-based recommendation
        recommended_bias = market_analysis.get('recommended_bias', 'balanced')
        
        # Adjust based on performance if available
        if performance_metrics and self.dynamic_switching_enabled:
            intraday_performance = performance_metrics.get('intraday_performance', 0.5)
            positional_performance = performance_metrics.get('positional_performance', 0.5)
            
            if intraday_performance > positional_performance + 0.1:
                return 'intraday_bias'
            elif positional_performance > intraday_performance + 0.1:
                return 'positional_bias'
        
        return recommended_bias
    
    def _configure_hybrid_timeframes(self, sub_mode: str) -> Dict[str, Any]:
        """Configure timeframes based on sub-mode"""
        base_config = {
            'primary_timeframe': '15min',
            'active_timeframes': ['5min', '15min', '30min', '1hr']
        }
        
        if sub_mode == 'intraday_bias':
            timeframe_weights = {
                '5min': 0.30,
                '15min': 0.35,
                '30min': 0.25,
                '1hr': 0.10
            }
        elif sub_mode == 'positional_bias':
            timeframe_weights = {
                '5min': 0.10,
                '15min': 0.25,
                '30min': 0.35,
                '1hr': 0.30
            }
        else:  # balanced
            timeframe_weights = {
                '5min': 0.20,
                '15min': 0.30,
                '30min': 0.30,
                '1hr': 0.20
            }
        
        base_config['timeframe_weights'] = timeframe_weights
        return base_config
    
    def _configure_hybrid_thresholds(self, sub_mode: str) -> Dict[str, Any]:
        """Configure thresholds based on sub-mode"""
        if sub_mode == 'intraday_bias':
            return {
                'directional_thresholds': {
                    'strong_bullish': 0.42,
                    'mild_bullish': 0.16,
                    'neutral': 0.06,
                    'mild_bearish': -0.16,
                    'strong_bearish': -0.42
                },
                'confidence_threshold': 0.70
            }
        elif sub_mode == 'positional_bias':
            return {
                'directional_thresholds': {
                    'strong_bullish': 0.48,
                    'mild_bullish': 0.19,
                    'neutral': 0.09,
                    'mild_bearish': -0.19,
                    'strong_bearish': -0.48
                },
                'confidence_threshold': 0.78
            }
        else:  # balanced
            return {
                'directional_thresholds': {
                    'strong_bullish': 0.45,
                    'mild_bullish': 0.18,
                    'neutral': 0.08,
                    'mild_bearish': -0.18,
                    'strong_bearish': -0.45
                },
                'confidence_threshold': 0.75
            }
    
    def _configure_hybrid_indicators(self, sub_mode: str) -> Dict[str, float]:
        """Configure indicator weights based on sub-mode"""
        if sub_mode == 'intraday_bias':
            return {
                'price_action': 0.28,
                'volume_analysis': 0.22,
                'technical_indicators': 0.20,
                'greek_sentiment': 0.18,
                'oi_analysis': 0.12
            }
        elif sub_mode == 'positional_bias':
            return {
                'oi_analysis': 0.28,
                'greek_sentiment': 0.24,
                'technical_indicators': 0.22,
                'price_action': 0.16,
                'volume_analysis': 0.10
            }
        else:  # balanced
            return {
                'greek_sentiment': 0.25,
                'oi_analysis': 0.22,
                'technical_indicators': 0.20,
                'price_action': 0.18,
                'volume_analysis': 0.15
            }
    
    def _track_mode_change(self, new_sub_mode: str, market_analysis: Dict[str, Any]):
        """Track sub-mode changes"""
        self.mode_history.append({
            'timestamp': datetime.now(),
            'sub_mode': new_sub_mode,
            'market_conditions': market_analysis.copy()
        })
        
        if len(self.mode_history) > 100:
            self.mode_history = self.mode_history[-100:]
    
    def get_manager_status(self) -> Dict[str, Any]:
        """Get current status"""
        return {
            'manager_type': 'hybrid',
            'current_sub_mode': self.current_sub_mode,
            'dynamic_switching_enabled': self.dynamic_switching_enabled,
            'mode_history_count': len(self.mode_history),
            'last_mode_change': self.mode_history[-1]['timestamp'].isoformat() if self.mode_history else None
        }


class HybridModeError(Exception):
    """Custom exception for hybrid mode errors"""
    pass