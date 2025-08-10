#!/usr/bin/env python3
"""
Adaptive Window Integration Module
Integration layer for Adaptive Rolling Window Optimizer with existing Enhanced Market Regime Framework V2.0

This module provides seamless integration of the adaptive rolling window system
with the existing comprehensive triple straddle engine while maintaining
backward compatibility.

Author: The Augster
Date: June 24, 2025
Version: 1.0.0 - Adaptive Window Integration
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging
import asyncio
from dataclasses import dataclass

# Import existing components
try:
    from .archive_comprehensive_modules_do_not_use.comprehensive_triple_straddle_engine import StraddleAnalysisEngine
    from .archive_enhanced_modules_do_not_use.enhanced_market_regime_engine import EnhancedMarketRegimeEngine
    from .adaptive_rolling_window_optimizer import AdaptiveRollingWindowOptimizer, AdaptiveWindowConfig
except ImportError:
    # Fallback for testing
    class StraddleAnalysisEngine:
        def __init__(self): pass
    class EnhancedMarketRegimeEngine:
        def __init__(self): pass
    from adaptive_rolling_window_optimizer import AdaptiveRollingWindowOptimizer, AdaptiveWindowConfig

logger = logging.getLogger(__name__)

@dataclass
class AdaptiveIntegrationConfig:
    """Configuration for adaptive window integration"""
    enable_adaptive_windows: bool = True
    fallback_to_static: bool = True
    performance_monitoring: bool = True
    adaptation_logging: bool = True
    validation_enabled: bool = True
    max_adaptation_frequency: int = 50  # Maximum adaptations per hour

class EnhancedMarketRegimeEngineWithAdaptiveWindows(EnhancedMarketRegimeEngine):
    """
    Enhanced Market Regime Engine with Adaptive Rolling Windows
    
    Extends the existing EnhancedMarketRegimeEngine to include adaptive
    rolling window optimization while maintaining full backward compatibility.
    """
    
    def __init__(self, adaptive_config: Optional[AdaptiveIntegrationConfig] = None):
        """Initialize enhanced engine with adaptive windows"""
        # Initialize parent class
        super().__init__()
        
        # Adaptive window configuration
        self.adaptive_config = adaptive_config or AdaptiveIntegrationConfig()
        
        # Initialize adaptive window optimizer
        if self.adaptive_config.enable_adaptive_windows:
            window_config = AdaptiveWindowConfig(
                base_windows=[3, 5, 10, 15],  # Preserve existing windows
                extended_windows=[1, 2, 4, 7, 12, 20, 30],
                adaptation_frequency=100,
                confidence_threshold=0.7
            )
            self.adaptive_optimizer = AdaptiveRollingWindowOptimizer(window_config)
        else:
            self.adaptive_optimizer = None
        
        # Performance tracking
        self.adaptation_history = []
        self.performance_comparison = {
            'static_performance': [],
            'adaptive_performance': [],
            'improvement_metrics': []
        }
        
        # Integration state
        self.current_adaptive_windows = None
        self.current_adaptive_weights = None
        self.last_adaptation_time = None
        self.adaptation_count = 0
        
        logger.info("Enhanced Market Regime Engine with Adaptive Windows initialized")
        logger.info(f"Adaptive windows enabled: {self.adaptive_config.enable_adaptive_windows}")
    
    async def analyze_comprehensive_market_regime_adaptive(self, 
                                                         market_data: Dict[str, Any],
                                                         current_dte: int = 0,
                                                         current_vix: float = 20.0) -> Dict[str, Any]:
        """
        Enhanced market regime analysis with adaptive rolling windows
        
        Args:
            market_data: Complete market data including all option prices and volumes
            current_dte: Current days to expiry for dynamic adjustments
            current_vix: Current VIX level for dynamic adjustments
            
        Returns:
            Complete market regime analysis results with adaptive window optimization
        """
        try:
            start_time = datetime.now()
            
            # Step 1: Optimize rolling windows if adaptive mode is enabled
            if self.adaptive_config.enable_adaptive_windows and self.adaptive_optimizer:
                window_optimization = await self._optimize_rolling_windows(market_data)
                
                # Update timeframes with optimized windows
                if window_optimization and window_optimization.confidence_score >= 0.7:
                    self._update_timeframes_with_adaptive_windows(window_optimization)
                    adaptation_applied = True
                else:
                    adaptation_applied = False
            else:
                window_optimization = None
                adaptation_applied = False
            
            # Step 2: Run comprehensive analysis with current timeframes
            comprehensive_results = await self._run_comprehensive_analysis(
                market_data, current_dte, current_vix
            )
            
            # Step 3: Add adaptive window information to results
            comprehensive_results['adaptive_window_info'] = {
                'adaptive_enabled': self.adaptive_config.enable_adaptive_windows,
                'adaptation_applied': adaptation_applied,
                'window_optimization': window_optimization.__dict__ if window_optimization else None,
                'current_windows': self.current_adaptive_windows,
                'current_weights': self.current_adaptive_weights,
                'adaptation_count': self.adaptation_count
            }
            
            # Step 4: Performance tracking and validation
            if self.adaptive_config.performance_monitoring:
                await self._track_performance(comprehensive_results, window_optimization)
            
            # Step 5: Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            comprehensive_results['performance_metrics']['adaptive_processing_time'] = processing_time
            
            # Step 6: Log adaptation if enabled
            if self.adaptive_config.adaptation_logging and adaptation_applied:
                self._log_adaptation(window_optimization, comprehensive_results)
            
            return comprehensive_results
            
        except Exception as e:
            logger.error(f"Error in adaptive market regime analysis: {e}")
            
            # Fallback to static analysis
            if self.adaptive_config.fallback_to_static:
                logger.info("Falling back to static window analysis")
                return await super().analyze_comprehensive_market_regime(
                    market_data, current_dte, current_vix
                )
            else:
                raise e
    
    async def _optimize_rolling_windows(self, market_data: Dict[str, Any]):
        """Optimize rolling windows based on market conditions"""
        try:
            # Check adaptation frequency limits
            if not self._should_attempt_adaptation():
                return None
            
            # Run window optimization
            window_optimization = await self.adaptive_optimizer.optimize_windows(market_data)
            
            # Validate optimization result
            if self.adaptive_config.validation_enabled:
                is_valid = self._validate_window_optimization(window_optimization)
                if not is_valid:
                    logger.warning("Window optimization validation failed, using static windows")
                    return None
            
            return window_optimization
            
        except Exception as e:
            logger.error(f"Error optimizing rolling windows: {e}")
            return None
    
    def _should_attempt_adaptation(self) -> bool:
        """Check if adaptation should be attempted based on frequency limits"""
        if not self.last_adaptation_time:
            return True
        
        # Check maximum adaptation frequency (per hour)
        time_since_last = (datetime.now() - self.last_adaptation_time).total_seconds()
        min_interval = 3600 / self.adaptive_config.max_adaptation_frequency  # seconds
        
        return time_since_last >= min_interval
    
    def _validate_window_optimization(self, optimization) -> bool:
        """Validate window optimization result"""
        try:
            # Check if windows are reasonable
            if not optimization.optimal_windows:
                return False
            
            # Check window range
            for window in optimization.optimal_windows:
                if window < 1 or window > 60:  # 1 minute to 1 hour
                    return False
            
            # Check weights
            if not optimization.optimal_weights:
                return False
            
            if len(optimization.optimal_weights) != len(optimization.optimal_windows):
                return False
            
            # Check weight sum (should be close to 1.0)
            weight_sum = sum(optimization.optimal_weights)
            if abs(weight_sum - 1.0) > 0.1:
                return False
            
            # Check confidence score
            if optimization.confidence_score < 0.5:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating window optimization: {e}")
            return False
    
    def _update_timeframes_with_adaptive_windows(self, optimization):
        """Update timeframes configuration with adaptive windows"""
        try:
            # Store current adaptive configuration
            self.current_adaptive_windows = optimization.optimal_windows
            self.current_adaptive_weights = optimization.optimal_weights
            self.last_adaptation_time = datetime.now()
            self.adaptation_count += 1
            
            # Update timeframes dictionary
            new_timeframes = {}
            for i, (window, weight) in enumerate(zip(optimization.optimal_windows, optimization.optimal_weights)):
                timeframe_key = f'{window}min'
                new_timeframes[timeframe_key] = {
                    'weight': weight,
                    'periods': window
                }
            
            # Update the timeframes in the parent class
            self.timeframes = new_timeframes
            
            # Update comprehensive engine timeframes if available
            if hasattr(self, 'comprehensive_engine'):
                self.comprehensive_engine.timeframes = new_timeframes
            
            logger.info(f"Updated timeframes with adaptive windows: {optimization.optimal_windows}")
            logger.info(f"Adaptive weights: {optimization.optimal_weights}")
            
        except Exception as e:
            logger.error(f"Error updating timeframes with adaptive windows: {e}")
    
    async def _run_comprehensive_analysis(self, market_data: Dict[str, Any], 
                                        current_dte: int, current_vix: float) -> Dict[str, Any]:
        """Run comprehensive analysis with current timeframe configuration"""
        # Use parent class method with current (possibly adaptive) timeframes
        return await super().analyze_comprehensive_market_regime(market_data, current_dte, current_vix)
    
    async def _track_performance(self, results: Dict[str, Any], optimization):
        """Track performance of adaptive vs static windows"""
        try:
            # Extract performance metrics
            accuracy = results.get('performance_metrics', {}).get('accuracy_estimate', 0.0)
            confidence = results.get('performance_metrics', {}).get('regime_confidence', 0.0)
            processing_time = results.get('performance_metrics', {}).get('total_processing_time', 0.0)
            
            # Store performance data
            performance_entry = {
                'timestamp': datetime.now(),
                'accuracy': accuracy,
                'confidence': confidence,
                'processing_time': processing_time,
                'adaptive_applied': optimization is not None,
                'windows_used': self.current_adaptive_windows if optimization else [3, 5, 10, 15],
                'weights_used': self.current_adaptive_weights if optimization else [0.15, 0.25, 0.30, 0.30]
            }
            
            if optimization:
                self.performance_comparison['adaptive_performance'].append(performance_entry)
            else:
                self.performance_comparison['static_performance'].append(performance_entry)
            
            # Calculate improvement metrics periodically
            if len(self.performance_comparison['adaptive_performance']) % 10 == 0:
                self._calculate_improvement_metrics()
            
        except Exception as e:
            logger.error(f"Error tracking performance: {e}")
    
    def _calculate_improvement_metrics(self):
        """Calculate improvement metrics comparing adaptive vs static performance"""
        try:
            adaptive_perf = self.performance_comparison['adaptive_performance']
            static_perf = self.performance_comparison['static_performance']
            
            if not adaptive_perf or not static_perf:
                return
            
            # Calculate recent averages (last 20 entries)
            recent_adaptive = adaptive_perf[-20:]
            recent_static = static_perf[-20:]
            
            adaptive_accuracy = np.mean([p['accuracy'] for p in recent_adaptive])
            static_accuracy = np.mean([p['accuracy'] for p in recent_static])
            
            adaptive_confidence = np.mean([p['confidence'] for p in recent_adaptive])
            static_confidence = np.mean([p['confidence'] for p in recent_static])
            
            # Calculate improvements
            accuracy_improvement = adaptive_accuracy - static_accuracy
            confidence_improvement = adaptive_confidence - static_confidence
            
            improvement_entry = {
                'timestamp': datetime.now(),
                'accuracy_improvement': accuracy_improvement,
                'confidence_improvement': confidence_improvement,
                'adaptive_sample_size': len(recent_adaptive),
                'static_sample_size': len(recent_static)
            }
            
            self.performance_comparison['improvement_metrics'].append(improvement_entry)
            
            logger.info(f"Performance improvement - Accuracy: {accuracy_improvement:.3f}, Confidence: {confidence_improvement:.3f}")
            
        except Exception as e:
            logger.error(f"Error calculating improvement metrics: {e}")
    
    def _log_adaptation(self, optimization, results: Dict[str, Any]):
        """Log adaptation details for monitoring and debugging"""
        try:
            adaptation_log = {
                'timestamp': datetime.now(),
                'optimization_method': optimization.optimization_method,
                'optimal_windows': optimization.optimal_windows,
                'optimal_weights': optimization.optimal_weights,
                'confidence_score': optimization.confidence_score,
                'performance_improvement': optimization.performance_improvement,
                'market_condition': optimization.market_condition.__dict__,
                'regime_result': {
                    'accuracy': results.get('performance_metrics', {}).get('accuracy_estimate', 0.0),
                    'confidence': results.get('performance_metrics', {}).get('regime_confidence', 0.0)
                }
            }
            
            self.adaptation_history.append(adaptation_log)
            
            # Keep only recent history (last 1000 adaptations)
            if len(self.adaptation_history) > 1000:
                self.adaptation_history = self.adaptation_history[-1000:]
            
            logger.info(f"Adaptation applied: {optimization.optimization_method} - Windows: {optimization.optimal_windows}")
            
        except Exception as e:
            logger.error(f"Error logging adaptation: {e}")
    
    def get_adaptive_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary for adaptive window system"""
        try:
            # Basic statistics
            total_adaptations = self.adaptation_count
            adaptive_enabled = self.adaptive_config.enable_adaptive_windows
            
            # Performance comparison
            improvement_metrics = self.performance_comparison['improvement_metrics']
            if improvement_metrics:
                latest_improvement = improvement_metrics[-1]
                avg_accuracy_improvement = np.mean([m['accuracy_improvement'] for m in improvement_metrics[-10:]])
                avg_confidence_improvement = np.mean([m['confidence_improvement'] for m in improvement_metrics[-10:]])
            else:
                latest_improvement = None
                avg_accuracy_improvement = 0.0
                avg_confidence_improvement = 0.0
            
            # Adaptive optimizer statistics
            optimizer_stats = {}
            if self.adaptive_optimizer:
                optimizer_stats = self.adaptive_optimizer.get_performance_statistics()
            
            return {
                'adaptive_enabled': adaptive_enabled,
                'total_adaptations': total_adaptations,
                'current_windows': self.current_adaptive_windows,
                'current_weights': self.current_adaptive_weights,
                'last_adaptation_time': self.last_adaptation_time,
                'performance_improvement': {
                    'accuracy_improvement': avg_accuracy_improvement,
                    'confidence_improvement': avg_confidence_improvement,
                    'latest_improvement': latest_improvement
                },
                'optimizer_statistics': optimizer_stats,
                'adaptation_history_size': len(self.adaptation_history)
            }
            
        except Exception as e:
            logger.error(f"Error generating performance summary: {e}")
            return {'error': str(e)}
    
    def reset_adaptive_system(self):
        """Reset adaptive system to default state"""
        try:
            # Reset to default timeframes
            self.timeframes = {
                '3min': {'weight': 0.15, 'periods': 3},
                '5min': {'weight': 0.25, 'periods': 5},
                '10min': {'weight': 0.30, 'periods': 10},
                '15min': {'weight': 0.30, 'periods': 15}
            }
            
            # Reset adaptive state
            self.current_adaptive_windows = None
            self.current_adaptive_weights = None
            self.last_adaptation_time = None
            self.adaptation_count = 0
            
            # Clear history
            self.adaptation_history.clear()
            self.performance_comparison = {
                'static_performance': [],
                'adaptive_performance': [],
                'improvement_metrics': []
            }
            
            logger.info("Adaptive system reset to default state")
            
        except Exception as e:
            logger.error(f"Error resetting adaptive system: {e}")

# Convenience function for easy integration
def create_adaptive_market_regime_engine(enable_adaptive: bool = True) -> EnhancedMarketRegimeEngineWithAdaptiveWindows:
    """
    Create an enhanced market regime engine with adaptive windows
    
    Args:
        enable_adaptive: Whether to enable adaptive window optimization
        
    Returns:
        EnhancedMarketRegimeEngineWithAdaptiveWindows instance
    """
    config = AdaptiveIntegrationConfig(
        enable_adaptive_windows=enable_adaptive,
        fallback_to_static=True,
        performance_monitoring=True,
        adaptation_logging=True,
        validation_enabled=True
    )
    
    return EnhancedMarketRegimeEngineWithAdaptiveWindows(config)
