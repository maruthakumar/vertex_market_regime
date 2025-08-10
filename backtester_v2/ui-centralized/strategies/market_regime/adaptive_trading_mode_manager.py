#!/usr/bin/env python3
"""
Adaptive Trading Mode Manager for Market Regime System
Integrates adaptive timeframe management with regime-based trading logic

This module provides the bridge between trading mode selection and strategy
execution, adapting all market regime parameters based on the selected mode.

Features:
1. Trading mode-aware parameter adjustment
2. Dynamic risk management based on mode
3. Strategy weight optimization per mode
4. Mode-specific regime interpretation
5. Automatic mode switching based on market conditions

Author: The Augster
Date: 2025-01-10
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
import json

# Import related modules
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from strategies.market_regime.adaptive_timeframe_manager import AdaptiveTimeframeManager
    from strategies.market_regime.timeframe_regime_extractor import TimeframeRegimeExtractor
except ImportError:
    # For direct execution
    from adaptive_timeframe_manager import AdaptiveTimeframeManager
    from timeframe_regime_extractor import TimeframeRegimeExtractor

logger = logging.getLogger(__name__)

@dataclass
class TradingModeParameters:
    """Parameters specific to each trading mode"""
    mode_name: str
    holding_period_range: Tuple[int, int]  # Min and max minutes
    stop_loss_multiplier: float
    target_profit_multiplier: float
    position_size_factor: float
    max_positions: int
    regime_sensitivity: float
    entry_confidence_threshold: float
    exit_urgency_factor: float
    
@dataclass
class RegimeTradingAdjustment:
    """Adjustments to regime parameters based on trading mode"""
    regime_id: int
    mode: str
    risk_adjustment: float
    position_size_adjustment: float
    stop_loss_adjustment: float
    target_adjustment: float
    strategy_weight_adjustments: Dict[str, float]

class AdaptiveTradingModeManager:
    """
    Manages adaptive trading mode integration with market regime system
    """
    
    # Default trading mode parameters
    MODE_PARAMETERS = {
        'intraday': TradingModeParameters(
            mode_name='intraday',
            holding_period_range=(5, 240),  # 5 minutes to 4 hours
            stop_loss_multiplier=0.8,       # Tighter stops
            target_profit_multiplier=0.9,    # Smaller targets
            position_size_factor=0.7,        # Smaller positions
            max_positions=5,
            regime_sensitivity=1.2,          # More sensitive to regime changes
            entry_confidence_threshold=0.65,
            exit_urgency_factor=1.5         # Quicker exits
        ),
        'positional': TradingModeParameters(
            mode_name='positional',
            holding_period_range=(240, 10080),  # 4 hours to 7 days
            stop_loss_multiplier=1.5,          # Wider stops
            target_profit_multiplier=2.0,       # Larger targets
            position_size_factor=1.2,           # Larger positions
            max_positions=3,
            regime_sensitivity=0.8,             # Less sensitive to regime changes
            entry_confidence_threshold=0.85,
            exit_urgency_factor=0.7            # Slower exits
        ),
        'hybrid': TradingModeParameters(
            mode_name='hybrid',
            holding_period_range=(30, 1440),   # 30 minutes to 1 day
            stop_loss_multiplier=1.0,
            target_profit_multiplier=1.2,
            position_size_factor=1.0,
            max_positions=4,
            regime_sensitivity=1.0,
            entry_confidence_threshold=0.75,
            exit_urgency_factor=1.0
        )
    }
    
    # Strategy preferences by mode and regime
    STRATEGY_MODE_PREFERENCES = {
        'intraday': {
            'high_volatility': ['ML_INDICATOR', 'MARKET_REGIME', 'ORB'],
            'medium_volatility': ['TBS', 'ML_INDICATOR', 'ORB'],
            'low_volatility': ['TBS', 'POS', 'OI'],
            'trending': ['TBS', 'TV', 'ML_INDICATOR'],
            'mean_reverting': ['POS', 'OI', 'MARKET_REGIME']
        },
        'positional': {
            'high_volatility': ['MARKET_REGIME', 'POS', 'OI'],
            'medium_volatility': ['POS', 'TV', 'ML_INDICATOR'],
            'low_volatility': ['POS', 'OI', 'TBS'],
            'trending': ['TV', 'MARKET_REGIME', 'TBS'],
            'mean_reverting': ['POS', 'OI', 'ML_INDICATOR']
        },
        'hybrid': {
            'high_volatility': ['MARKET_REGIME', 'ML_INDICATOR', 'POS'],
            'medium_volatility': ['TBS', 'POS', 'TV'],
            'low_volatility': ['TBS', 'OI', 'POS'],
            'trending': ['TBS', 'TV', 'MARKET_REGIME'],
            'mean_reverting': ['POS', 'OI', 'ML_INDICATOR']
        }
    }
    
    def __init__(self, initial_mode='hybrid'):
        """
        Initialize adaptive trading mode manager
        
        Args:
            initial_mode: Starting trading mode
        """
        self.current_mode = initial_mode
        self.mode_parameters = self.MODE_PARAMETERS[initial_mode]
        
        # Initialize managers
        self.timeframe_manager = AdaptiveTimeframeManager(mode=initial_mode)
        self.timeframe_extractor = TimeframeRegimeExtractor(trading_mode=initial_mode)
        
        # Performance tracking
        self.mode_performance = {
            'intraday': {'trades': 0, 'win_rate': 0.0, 'avg_pnl': 0.0},
            'positional': {'trades': 0, 'win_rate': 0.0, 'avg_pnl': 0.0},
            'hybrid': {'trades': 0, 'win_rate': 0.0, 'avg_pnl': 0.0}
        }
        
        # Mode switching history
        self.mode_history = []
        self.last_mode_switch = datetime.now()
        
        logger.info(f"AdaptiveTradingModeManager initialized with {initial_mode} mode")
    
    def switch_mode(self, new_mode: str, reason: str = "Manual switch") -> bool:
        """
        Switch to a different trading mode
        
        Args:
            new_mode: Target trading mode
            reason: Reason for mode switch
            
        Returns:
            Success status
        """
        try:
            if new_mode not in self.MODE_PARAMETERS:
                logger.error(f"Invalid mode: {new_mode}")
                return False
            
            # Record mode switch
            self.mode_history.append({
                'from_mode': self.current_mode,
                'to_mode': new_mode,
                'timestamp': datetime.now(),
                'reason': reason
            })
            
            # Update parameters
            self.current_mode = new_mode
            self.mode_parameters = self.MODE_PARAMETERS[new_mode]
            
            # Update connected managers
            self.timeframe_manager.switch_mode(new_mode)
            self.timeframe_extractor.switch_trading_mode(new_mode)
            
            self.last_mode_switch = datetime.now()
            
            logger.info(f"Switched to {new_mode} mode. Reason: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Error switching mode: {e}")
            return False
    
    def get_regime_adjustments(self, regime_id: int) -> RegimeTradingAdjustment:
        """
        Get trading adjustments for specific regime based on current mode
        
        Args:
            regime_id: Market regime ID (0-17)
            
        Returns:
            RegimeTradingAdjustment object
        """
        # Base adjustments from mode parameters
        base_adjustment = RegimeTradingAdjustment(
            regime_id=regime_id,
            mode=self.current_mode,
            risk_adjustment=1.0,
            position_size_adjustment=self.mode_parameters.position_size_factor,
            stop_loss_adjustment=self.mode_parameters.stop_loss_multiplier,
            target_adjustment=self.mode_parameters.target_profit_multiplier,
            strategy_weight_adjustments={}
        )
        
        # Adjust based on regime characteristics
        volatility_level = self._get_volatility_level(regime_id)
        market_structure = self._get_market_structure(regime_id)
        
        # Mode-specific adjustments
        if self.current_mode == 'intraday':
            if volatility_level == 'high':
                base_adjustment.risk_adjustment *= 0.7
                base_adjustment.position_size_adjustment *= 0.8
            elif volatility_level == 'low':
                base_adjustment.risk_adjustment *= 1.2
                base_adjustment.position_size_adjustment *= 1.1
        
        elif self.current_mode == 'positional':
            if volatility_level == 'high':
                base_adjustment.stop_loss_adjustment *= 1.3
                base_adjustment.target_adjustment *= 1.5
            elif market_structure == 'trending':
                base_adjustment.position_size_adjustment *= 1.2
        
        # Get strategy preferences
        strategy_key = f"{volatility_level}_volatility" if market_structure == 'neutral' else market_structure
        preferred_strategies = self.STRATEGY_MODE_PREFERENCES[self.current_mode].get(
            strategy_key, ['TBS', 'POS', 'ML_INDICATOR']
        )
        
        # Set strategy weight adjustments
        for i, strategy in enumerate(preferred_strategies):
            weight = 0.5 - (i * 0.15)  # Decreasing weights
            base_adjustment.strategy_weight_adjustments[strategy] = max(0.2, weight)
        
        return base_adjustment
    
    def should_switch_mode(self, market_data: Dict[str, Any]) -> Optional[str]:
        """
        Determine if mode should be switched based on market conditions
        
        Args:
            market_data: Current market data
            
        Returns:
            Recommended mode or None if no switch needed
        """
        # Don't switch too frequently
        time_since_switch = datetime.now() - self.last_mode_switch
        if time_since_switch < timedelta(minutes=30):
            return None
        
        # Analyze market conditions
        volatility = market_data.get('volatility_percentile', 0.5)
        volume = market_data.get('volume_ratio', 1.0)
        trend_strength = market_data.get('trend_strength', 0.0)
        
        # High volatility + high volume = intraday opportunities
        if volatility > 0.7 and volume > 1.5:
            if self.current_mode != 'intraday':
                return 'intraday'
        
        # Low volatility + strong trend = positional opportunities
        elif volatility < 0.3 and abs(trend_strength) > 0.6:
            if self.current_mode != 'positional':
                return 'positional'
        
        # Mixed conditions = hybrid mode
        elif self.current_mode != 'hybrid':
            if 0.3 <= volatility <= 0.7:
                return 'hybrid'
        
        return None
    
    def optimize_parameters_for_mode(self, regime_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize all parameters based on current mode and regime
        
        Args:
            regime_data: Current regime analysis data
            
        Returns:
            Optimized parameters dictionary
        """
        regime_id = regime_data.get('regime_id', 8)  # Default to neutral
        adjustments = self.get_regime_adjustments(regime_id)
        
        # Get base parameters from mode
        optimized_params = {
            'mode': self.current_mode,
            'holding_period_range': self.mode_parameters.holding_period_range,
            'max_positions': self.mode_parameters.max_positions,
            'entry_confidence_threshold': self.mode_parameters.entry_confidence_threshold,
            'exit_urgency_factor': self.mode_parameters.exit_urgency_factor,
            
            # Adjusted parameters
            'risk_adjustment': adjustments.risk_adjustment,
            'position_size_multiplier': adjustments.position_size_adjustment,
            'stop_loss_multiplier': adjustments.stop_loss_adjustment,
            'target_multiplier': adjustments.target_adjustment,
            'strategy_weights': adjustments.strategy_weight_adjustments,
            
            # Timeframe configuration
            'active_timeframes': self.timeframe_manager.get_active_timeframes(),
            'timeframe_weights': self.timeframe_manager.get_timeframe_weights(),
            
            # Regime-specific
            'regime_sensitivity': self.mode_parameters.regime_sensitivity,
            'regime_transition_threshold': self.timeframe_manager.active_config.transition_threshold
        }
        
        return optimized_params
    
    def update_performance_metrics(self, trade_result: Dict[str, Any]):
        """
        Update performance metrics for current mode
        
        Args:
            trade_result: Trade result data
        """
        mode = trade_result.get('mode', self.current_mode)
        
        if mode in self.mode_performance:
            perf = self.mode_performance[mode]
            perf['trades'] += 1
            
            # Update win rate
            if trade_result.get('pnl', 0) > 0:
                current_wins = perf['win_rate'] * (perf['trades'] - 1)
                perf['win_rate'] = (current_wins + 1) / perf['trades']
            else:
                current_wins = perf['win_rate'] * (perf['trades'] - 1)
                perf['win_rate'] = current_wins / perf['trades']
            
            # Update average P&L
            current_total_pnl = perf['avg_pnl'] * (perf['trades'] - 1)
            perf['avg_pnl'] = (current_total_pnl + trade_result.get('pnl', 0)) / perf['trades']
    
    def get_mode_recommendation(self, strategy_stats: Dict[str, Any]) -> str:
        """
        Get mode recommendation based on strategy statistics
        
        Args:
            strategy_stats: Strategy performance statistics
            
        Returns:
            Recommended trading mode
        """
        avg_holding_period = strategy_stats.get('avg_holding_period_minutes', 60)
        trade_frequency = strategy_stats.get('trades_per_day', 5)
        win_rate = strategy_stats.get('win_rate', 0.5)
        
        # Intraday indicators
        if avg_holding_period < 60 and trade_frequency > 10:
            return 'intraday'
        
        # Positional indicators
        elif avg_holding_period > 240 and trade_frequency < 3:
            return 'positional'
        
        # Default to hybrid
        else:
            return 'hybrid'
    
    def _get_volatility_level(self, regime_id: int) -> str:
        """Get volatility level from regime ID"""
        if regime_id < 6:
            return 'low'
        elif regime_id < 12:
            return 'medium'
        else:
            return 'high'
    
    def _get_market_structure(self, regime_id: int) -> str:
        """Get market structure from regime ID"""
        # Structure pattern: trending/mean-rev alternates
        if regime_id % 2 == 0:
            return 'trending'
        else:
            return 'mean_reverting'
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of trading mode manager"""
        return {
            'current_mode': self.current_mode,
            'mode_parameters': {
                'holding_period': self.mode_parameters.holding_period_range,
                'stop_loss_mult': self.mode_parameters.stop_loss_multiplier,
                'target_mult': self.mode_parameters.target_profit_multiplier,
                'position_size': self.mode_parameters.position_size_factor,
                'max_positions': self.mode_parameters.max_positions
            },
            'active_timeframes': self.timeframe_manager.get_active_timeframes(),
            'performance': self.mode_performance,
            'last_switch': self.last_mode_switch.isoformat(),
            'switch_history': len(self.mode_history)
        }
    
    def export_configuration(self) -> Dict[str, Any]:
        """Export current configuration"""
        return {
            'mode': self.current_mode,
            'mode_parameters': self.mode_parameters.__dict__,
            'timeframe_config': self.timeframe_manager.export_configuration(),
            'performance_metrics': self.mode_performance,
            'mode_history': self.mode_history[-10:]  # Last 10 switches
        }

# Example usage
if __name__ == "__main__":
    # Initialize manager
    manager = AdaptiveTradingModeManager(initial_mode='hybrid')
    
    print("Adaptive Trading Mode Manager Status:")
    print(json.dumps(manager.get_status(), indent=2))
    
    # Test regime adjustments
    print("\nRegime Adjustments for Different Modes:")
    
    # Test intraday mode
    manager.switch_mode('intraday')
    adjustments = manager.get_regime_adjustments(12)  # High volatility regime
    print(f"\nIntraday mode adjustments for regime 12:")
    print(f"  Risk adjustment: {adjustments.risk_adjustment}")
    print(f"  Position size: {adjustments.position_size_adjustment}")
    print(f"  Strategy weights: {adjustments.strategy_weight_adjustments}")
    
    # Test positional mode
    manager.switch_mode('positional')
    adjustments = manager.get_regime_adjustments(0)  # Low volatility regime
    print(f"\nPositional mode adjustments for regime 0:")
    print(f"  Risk adjustment: {adjustments.risk_adjustment}")
    print(f"  Stop loss mult: {adjustments.stop_loss_adjustment}")
    print(f"  Strategy weights: {adjustments.strategy_weight_adjustments}")
    
    # Test mode recommendation
    test_stats = {
        'avg_holding_period_minutes': 30,
        'trades_per_day': 15,
        'win_rate': 0.65
    }
    
    recommendation = manager.get_mode_recommendation(test_stats)
    print(f"\nMode recommendation for stats: {recommendation}")
    
    # Test market condition analysis
    market_data = {
        'volatility_percentile': 0.8,
        'volume_ratio': 2.0,
        'trend_strength': 0.3
    }
    
    should_switch = manager.should_switch_mode(market_data)
    print(f"\nShould switch mode based on market data: {should_switch}")
    
    # Test parameter optimization
    regime_data = {'regime_id': 6, 'confidence': 0.85}
    optimized = manager.optimize_parameters_for_mode(regime_data)
    print(f"\nOptimized parameters:")
    print(json.dumps(optimized, indent=2, default=str))