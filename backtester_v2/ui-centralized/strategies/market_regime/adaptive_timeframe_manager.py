#!/usr/bin/env python3
"""
Adaptive Timeframe Manager for Market Regime Formation System
Manages dynamic timeframe selection based on trading mode (Intraday/Positional/Hybrid)

This module provides flexible timeframe configuration supporting both intraday 
and positional trading strategies with Excel-based configuration.

Features:
1. Multiple trading modes with preset timeframe configurations
2. Dynamic weight adjustment based on mode selection
3. Custom timeframe configuration support
4. Performance-based timeframe optimization
5. Validation and error handling

Author: The Augster
Date: 2025-01-10
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

@dataclass
class TimeframeConfig:
    """Configuration for a single timeframe"""
    name: str
    enabled: bool
    weight: float
    min_data_points: int
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'enabled': self.enabled,
            'weight': self.weight,
            'min_data_points': self.min_data_points,
            'description': self.description
        }

@dataclass
class TradingModeConfig:
    """Configuration for a trading mode"""
    mode_name: str
    description: str
    timeframes: Dict[str, TimeframeConfig]
    risk_multiplier: float = 1.0
    transition_threshold: float = 0.75
    regime_stability_window: int = 15
    
    def get_enabled_timeframes(self) -> List[str]:
        """Get list of enabled timeframes"""
        return [tf for tf, config in self.timeframes.items() if config.enabled]
    
    def get_timeframe_weights(self) -> Dict[str, float]:
        """Get normalized weights for enabled timeframes"""
        enabled_weights = {
            tf: config.weight 
            for tf, config in self.timeframes.items() 
            if config.enabled
        }
        
        # Normalize weights to sum to 1.0
        total_weight = sum(enabled_weights.values())
        if total_weight > 0:
            return {tf: w/total_weight for tf, w in enabled_weights.items()}
        return {}

class AdaptiveTimeframeManager:
    """
    Manages adaptive timeframe selection for market regime analysis
    """
    
    # Available timeframes in the system
    AVAILABLE_TIMEFRAMES = ['3min', '5min', '10min', '15min', '30min', '1hr', '4hr']
    
    # Default configurations for each trading mode
    DEFAULT_MODES = {
        'intraday': {
            'description': 'Optimized for intraday trading (minutes to hours)',
            'timeframes': {
                '3min': {'enabled': True, 'weight': 0.30, 'min_points': 20, 
                        'desc': 'Ultra short-term scalping'},
                '5min': {'enabled': True, 'weight': 0.30, 'min_points': 12, 
                        'desc': 'Short-term entry signals'},
                '10min': {'enabled': True, 'weight': 0.25, 'min_points': 6, 
                         'desc': 'Trend confirmation'},
                '15min': {'enabled': True, 'weight': 0.15, 'min_points': 4, 
                         'desc': 'Intraday context'},
                '30min': {'enabled': False, 'weight': 0.0, 'min_points': 2, 
                         'desc': 'Medium-term reference'},
                '1hr': {'enabled': False, 'weight': 0.0, 'min_points': 1, 
                       'desc': 'Hourly reference'},
                '4hr': {'enabled': False, 'weight': 0.0, 'min_points': 1, 
                       'desc': 'Not used for intraday'}
            },
            'risk_multiplier': 0.8,
            'transition_threshold': 0.65,
            'regime_stability_window': 5
        },
        'positional': {
            'description': 'Optimized for positional trading (hours to days)',
            'timeframes': {
                '3min': {'enabled': False, 'weight': 0.0, 'min_points': 20, 
                        'desc': 'Too granular for positional'},
                '5min': {'enabled': False, 'weight': 0.0, 'min_points': 12, 
                        'desc': 'Too granular for positional'},
                '10min': {'enabled': False, 'weight': 0.0, 'min_points': 6, 
                         'desc': 'Not used for positional'},
                '15min': {'enabled': True, 'weight': 0.15, 'min_points': 4, 
                         'desc': 'Entry fine-tuning'},
                '30min': {'enabled': True, 'weight': 0.25, 'min_points': 2, 
                         'desc': 'Primary signal timeframe'},
                '1hr': {'enabled': True, 'weight': 0.35, 'min_points': 1, 
                       'desc': 'Trend confirmation'},
                '4hr': {'enabled': True, 'weight': 0.25, 'min_points': 1, 
                       'desc': 'Multi-day context'}
            },
            'risk_multiplier': 1.5,
            'transition_threshold': 0.85,
            'regime_stability_window': 30
        },
        'hybrid': {
            'description': 'Balanced approach for flexible trading',
            'timeframes': {
                '3min': {'enabled': False, 'weight': 0.0, 'min_points': 20, 
                        'desc': 'Optional for scalping'},
                '5min': {'enabled': True, 'weight': 0.20, 'min_points': 12, 
                        'desc': 'Short-term signals'},
                '10min': {'enabled': False, 'weight': 0.0, 'min_points': 6, 
                         'desc': 'Optional confirmation'},
                '15min': {'enabled': True, 'weight': 0.30, 'min_points': 4, 
                         'desc': 'Primary timeframe'},
                '30min': {'enabled': True, 'weight': 0.30, 'min_points': 2, 
                         'desc': 'Medium-term trend'},
                '1hr': {'enabled': True, 'weight': 0.20, 'min_points': 1, 
                       'desc': 'Broader context'},
                '4hr': {'enabled': False, 'weight': 0.0, 'min_points': 1, 
                       'desc': 'Optional for swing'}
            },
            'risk_multiplier': 1.0,
            'transition_threshold': 0.75,
            'regime_stability_window': 15
        }
    }
    
    def __init__(self, mode: str = 'hybrid', config: Optional[Dict[str, Any]] = None):
        """
        Initialize adaptive timeframe manager
        
        Args:
            mode: Trading mode ('intraday', 'positional', 'hybrid', 'custom')
            config: Optional custom configuration
        """
        self.current_mode = mode
        self.custom_config = config
        self.mode_configs = {}
        self.performance_metrics = {}
        
        # Initialize default mode configurations
        self._initialize_default_modes()
        
        # Load custom configuration if provided
        if mode == 'custom' and config:
            self._load_custom_config(config)
        
        # Set active configuration
        self.active_config = self._get_mode_config(mode)
        
        logger.info(f"AdaptiveTimeframeManager initialized with mode: {mode}")
    
    def _initialize_default_modes(self):
        """Initialize default trading mode configurations"""
        for mode_name, mode_data in self.DEFAULT_MODES.items():
            timeframes = {}
            for tf_name in self.AVAILABLE_TIMEFRAMES:
                if tf_name in mode_data['timeframes']:
                    tf_data = mode_data['timeframes'][tf_name]
                    timeframes[tf_name] = TimeframeConfig(
                        name=tf_name,
                        enabled=tf_data['enabled'],
                        weight=tf_data['weight'],
                        min_data_points=tf_data['min_points'],
                        description=tf_data['desc']
                    )
            
            self.mode_configs[mode_name] = TradingModeConfig(
                mode_name=mode_name,
                description=mode_data['description'],
                timeframes=timeframes,
                risk_multiplier=mode_data.get('risk_multiplier', 1.0),
                transition_threshold=mode_data.get('transition_threshold', 0.75),
                regime_stability_window=mode_data.get('regime_stability_window', 15)
            )
    
    def _load_custom_config(self, config: Dict[str, Any]):
        """Load custom configuration from dictionary or Excel data"""
        try:
            timeframes = {}
            
            # Parse timeframe configurations
            for tf_name in self.AVAILABLE_TIMEFRAMES:
                if tf_name in config.get('timeframes', {}):
                    tf_config = config['timeframes'][tf_name]
                    timeframes[tf_name] = TimeframeConfig(
                        name=tf_name,
                        enabled=tf_config.get('enabled', False),
                        weight=tf_config.get('weight', 0.0),
                        min_data_points=tf_config.get('min_data_points', 1),
                        description=tf_config.get('description', '')
                    )
            
            # Create custom mode configuration
            self.mode_configs['custom'] = TradingModeConfig(
                mode_name='custom',
                description=config.get('description', 'Custom trading mode'),
                timeframes=timeframes,
                risk_multiplier=config.get('risk_multiplier', 1.0),
                transition_threshold=config.get('transition_threshold', 0.75),
                regime_stability_window=config.get('regime_stability_window', 15)
            )
            
            logger.info("Custom configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading custom configuration: {e}")
            # Fall back to hybrid mode
            self.current_mode = 'hybrid'
    
    def _get_mode_config(self, mode: str) -> TradingModeConfig:
        """Get configuration for specified mode"""
        if mode in self.mode_configs:
            return self.mode_configs[mode]
        else:
            logger.warning(f"Unknown mode {mode}, defaulting to hybrid")
            return self.mode_configs['hybrid']
    
    def switch_mode(self, new_mode: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Switch to a different trading mode
        
        Args:
            new_mode: Target mode name
            config: Optional configuration for custom mode
            
        Returns:
            Success status
        """
        try:
            if new_mode == 'custom' and config:
                self._load_custom_config(config)
            
            if new_mode in self.mode_configs:
                self.current_mode = new_mode
                self.active_config = self.mode_configs[new_mode]
                logger.info(f"Switched to {new_mode} mode")
                return True
            else:
                logger.error(f"Invalid mode: {new_mode}")
                return False
                
        except Exception as e:
            logger.error(f"Error switching mode: {e}")
            return False
    
    def get_active_timeframes(self) -> List[str]:
        """Get list of currently active timeframes"""
        return self.active_config.get_enabled_timeframes()
    
    def get_timeframe_weights(self) -> Dict[str, float]:
        """Get normalized weights for active timeframes"""
        return self.active_config.get_timeframe_weights()
    
    def get_timeframe_config(self, timeframe: str) -> Optional[TimeframeConfig]:
        """Get configuration for specific timeframe"""
        return self.active_config.timeframes.get(timeframe)
    
    def validate_timeframe_data(self, timeframe: str, data_points: int) -> bool:
        """
        Validate if sufficient data is available for timeframe
        
        Args:
            timeframe: Timeframe to validate
            data_points: Number of available data points
            
        Returns:
            True if data is sufficient
        """
        config = self.get_timeframe_config(timeframe)
        if config and config.enabled:
            return data_points >= config.min_data_points
        return False
    
    def update_timeframe_weight(self, timeframe: str, new_weight: float) -> bool:
        """
        Update weight for specific timeframe
        
        Args:
            timeframe: Timeframe to update
            new_weight: New weight value
            
        Returns:
            Success status
        """
        try:
            if timeframe in self.active_config.timeframes:
                self.active_config.timeframes[timeframe].weight = new_weight
                logger.info(f"Updated {timeframe} weight to {new_weight}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error updating timeframe weight: {e}")
            return False
    
    def enable_timeframe(self, timeframe: str, enable: bool = True) -> bool:
        """
        Enable or disable a timeframe
        
        Args:
            timeframe: Timeframe to modify
            enable: Enable flag
            
        Returns:
            Success status
        """
        try:
            if timeframe in self.active_config.timeframes:
                self.active_config.timeframes[timeframe].enabled = enable
                logger.info(f"{'Enabled' if enable else 'Disabled'} {timeframe}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error enabling/disabling timeframe: {e}")
            return False
    
    def get_mode_parameters(self) -> Dict[str, Any]:
        """Get all parameters for current mode"""
        return {
            'mode': self.current_mode,
            'description': self.active_config.description,
            'active_timeframes': self.get_active_timeframes(),
            'timeframe_weights': self.get_timeframe_weights(),
            'risk_multiplier': self.active_config.risk_multiplier,
            'transition_threshold': self.active_config.transition_threshold,
            'regime_stability_window': self.active_config.regime_stability_window
        }
    
    def suggest_mode_from_strategy(self, holding_period_minutes: int, 
                                  trade_frequency: str) -> str:
        """
        Suggest optimal mode based on strategy characteristics
        
        Args:
            holding_period_minutes: Average holding period in minutes
            trade_frequency: 'high', 'medium', 'low'
            
        Returns:
            Suggested mode name
        """
        if holding_period_minutes < 60 and trade_frequency == 'high':
            return 'intraday'
        elif holding_period_minutes > 240:
            return 'positional'
        else:
            return 'hybrid'
    
    def optimize_weights_from_performance(self, performance_data: Dict[str, float]):
        """
        Optimize timeframe weights based on performance metrics
        
        Args:
            performance_data: Dictionary of timeframe -> accuracy scores
        """
        try:
            # Only optimize enabled timeframes
            enabled_timeframes = self.get_active_timeframes()
            
            if not enabled_timeframes:
                return
            
            # Calculate performance-based weights
            total_performance = sum(
                performance_data.get(tf, 0.5) 
                for tf in enabled_timeframes
            )
            
            if total_performance > 0:
                for tf in enabled_timeframes:
                    perf_score = performance_data.get(tf, 0.5)
                    new_weight = perf_score / total_performance
                    self.update_timeframe_weight(tf, new_weight)
                
                logger.info("Timeframe weights optimized based on performance")
                
        except Exception as e:
            logger.error(f"Error optimizing weights: {e}")
    
    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """
        Validate current configuration
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check minimum enabled timeframes
        enabled = self.get_active_timeframes()
        if len(enabled) < 2:
            errors.append("At least 2 timeframes must be enabled")
        
        # Check weight sum
        weights = self.get_timeframe_weights()
        if weights and abs(sum(weights.values()) - 1.0) > 0.001:
            errors.append("Timeframe weights must sum to 1.0")
        
        # Check valid timeframe names
        for tf in enabled:
            if tf not in self.AVAILABLE_TIMEFRAMES:
                errors.append(f"Invalid timeframe: {tf}")
        
        return len(errors) == 0, errors
    
    def export_configuration(self) -> Dict[str, Any]:
        """Export current configuration for saving"""
        config = {
            'mode': self.current_mode,
            'description': self.active_config.description,
            'timeframes': {},
            'risk_multiplier': self.active_config.risk_multiplier,
            'transition_threshold': self.active_config.transition_threshold,
            'regime_stability_window': self.active_config.regime_stability_window
        }
        
        for tf_name, tf_config in self.active_config.timeframes.items():
            config['timeframes'][tf_name] = tf_config.to_dict()
        
        return config
    
    def import_configuration(self, config: Dict[str, Any]) -> bool:
        """Import configuration from dictionary"""
        try:
            if config.get('mode') == 'custom':
                return self.switch_mode('custom', config)
            else:
                mode = config.get('mode', 'hybrid')
                if self.switch_mode(mode):
                    # Apply any custom modifications
                    for tf_name, tf_data in config.get('timeframes', {}).items():
                        if 'enabled' in tf_data:
                            self.enable_timeframe(tf_name, tf_data['enabled'])
                        if 'weight' in tf_data:
                            self.update_timeframe_weight(tf_name, tf_data['weight'])
                    return True
            return False
        except Exception as e:
            logger.error(f"Error importing configuration: {e}")
            return False

# Example usage and testing
if __name__ == "__main__":
    # Initialize manager with intraday mode
    manager = AdaptiveTimeframeManager(mode='intraday')
    
    print("Intraday Mode Configuration:")
    print(f"Active timeframes: {manager.get_active_timeframes()}")
    print(f"Timeframe weights: {manager.get_timeframe_weights()}")
    print(f"Mode parameters: {json.dumps(manager.get_mode_parameters(), indent=2)}")
    
    # Switch to positional mode
    print("\n" + "="*50 + "\n")
    manager.switch_mode('positional')
    print("Positional Mode Configuration:")
    print(f"Active timeframes: {manager.get_active_timeframes()}")
    print(f"Timeframe weights: {manager.get_timeframe_weights()}")
    
    # Custom configuration example
    print("\n" + "="*50 + "\n")
    custom_config = {
        'description': 'My custom scalping setup',
        'timeframes': {
            '3min': {'enabled': True, 'weight': 0.5},
            '5min': {'enabled': True, 'weight': 0.3},
            '10min': {'enabled': True, 'weight': 0.2}
        },
        'risk_multiplier': 0.5,
        'transition_threshold': 0.6
    }
    
    manager.switch_mode('custom', custom_config)
    print("Custom Mode Configuration:")
    print(f"Active timeframes: {manager.get_active_timeframes()}")
    print(f"Timeframe weights: {manager.get_timeframe_weights()}")
    
    # Validate configuration
    is_valid, errors = manager.validate_configuration()
    print(f"\nConfiguration valid: {is_valid}")
    if errors:
        print(f"Errors: {errors}")