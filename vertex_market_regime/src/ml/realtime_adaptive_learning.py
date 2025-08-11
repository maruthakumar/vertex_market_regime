"""
Real-Time Adaptive Learning Enhancement for Market Regime Components

Advanced adaptive learning system that continuously updates component weights
based on real-time performance feedback, market conditions, and streaming data.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import logging
from enum import Enum
import threading
import time
from abc import ABC, abstractmethod


class LearningMode(Enum):
    """Real-time learning modes"""
    PASSIVE = "passive"           # Learn from historical data only
    ACTIVE = "active"             # Learn from real-time feedback
    AGGRESSIVE = "aggressive"     # Fast adaptation to new patterns
    CONSERVATIVE = "conservative" # Slow, stable adaptation
    ADAPTIVE = "adaptive"         # Mode switches based on market conditions


class MarketRegime(Enum):
    """Market regime types for regime-specific learning"""
    TRENDING_BULLISH = "trending_bullish"
    TRENDING_BEARISH = "trending_bearish"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    UNKNOWN = "unknown"


@dataclass
class PerformanceFeedback:
    """Real-time performance feedback structure"""
    timestamp: datetime
    component_id: int
    predicted_regime: MarketRegime
    actual_regime: MarketRegime
    accuracy: float
    confidence: float
    prediction_error: float
    market_conditions: Dict[str, float]
    processing_time_ms: float
    
    def is_correct_prediction(self) -> bool:
        """Check if prediction was correct"""
        return self.predicted_regime == self.actual_regime
    
    def get_regime_transition(self) -> Optional[Tuple[MarketRegime, MarketRegime]]:
        """Get regime transition if any"""
        if self.predicted_regime != self.actual_regime:
            return (self.predicted_regime, self.actual_regime)
        return None


@dataclass
class AdaptiveLearningState:
    """Current state of adaptive learning system"""
    component_weights: Dict[str, float] = field(default_factory=dict)
    learning_rate: float = 0.01
    momentum: float = 0.9
    decay_rate: float = 0.99
    performance_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    regime_performance: Dict[MarketRegime, float] = field(default_factory=dict)
    weight_gradients: Dict[str, float] = field(default_factory=dict)
    last_update: datetime = field(default_factory=datetime.utcnow)
    update_count: int = 0
    
    def add_feedback(self, feedback: PerformanceFeedback):
        """Add performance feedback to history"""
        self.performance_history.append(feedback)
        self.last_update = datetime.utcnow()
    
    def get_recent_performance(self, minutes: int = 30) -> List[PerformanceFeedback]:
        """Get performance feedback from last N minutes"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        return [f for f in self.performance_history if f.timestamp >= cutoff_time]


class RealTimeLearningStrategy(ABC):
    """Abstract base class for real-time learning strategies"""
    
    @abstractmethod
    async def update_weights(self, 
                           state: AdaptiveLearningState,
                           feedback: PerformanceFeedback) -> Dict[str, float]:
        """Update component weights based on feedback"""
        pass
    
    @abstractmethod
    def get_learning_rate(self, state: AdaptiveLearningState) -> float:
        """Get current learning rate"""
        pass


class GradientDescentStrategy(RealTimeLearningStrategy):
    """Gradient descent based weight updating"""
    
    def __init__(self, base_learning_rate: float = 0.01, momentum: float = 0.9):
        self.base_learning_rate = base_learning_rate
        self.momentum = momentum
    
    async def update_weights(self, 
                           state: AdaptiveLearningState,
                           feedback: PerformanceFeedback) -> Dict[str, float]:
        """Update weights using gradient descent with momentum"""
        
        # Calculate prediction error gradient
        error = 1.0 - feedback.accuracy
        learning_rate = self.get_learning_rate(state)
        
        updated_weights = {}
        
        for weight_name, current_weight in state.component_weights.items():
            # Calculate gradient based on performance feedback
            gradient = self._calculate_gradient(weight_name, feedback, error)
            
            # Apply momentum
            momentum_gradient = state.weight_gradients.get(weight_name, 0.0)
            new_gradient = self.momentum * momentum_gradient + (1 - self.momentum) * gradient
            state.weight_gradients[weight_name] = new_gradient
            
            # Update weight
            weight_update = learning_rate * new_gradient
            new_weight = current_weight - weight_update
            
            # Ensure weights stay in valid range [0, 2]
            new_weight = max(0.0, min(2.0, new_weight))
            updated_weights[weight_name] = new_weight
        
        return updated_weights
    
    def _calculate_gradient(self, weight_name: str, feedback: PerformanceFeedback, error: float) -> float:
        """Calculate gradient for specific weight"""
        # Simplified gradient calculation based on prediction error
        # In practice, this would be more sophisticated
        base_gradient = error * feedback.confidence
        
        # Adjust gradient based on weight type
        if 'gamma' in weight_name.lower():
            # Gamma weights are critical for pin risk detection
            return base_gradient * 1.2
        elif 'delta' in weight_name.lower():
            # Delta weights for directional bias
            return base_gradient * 1.0
        elif 'vega' in weight_name.lower():
            # Vega weights for volatility sensitivity
            return base_gradient * 0.8
        elif 'theta' in weight_name.lower():
            # Theta weights for time decay
            return base_gradient * 0.6
        
        return base_gradient
    
    def get_learning_rate(self, state: AdaptiveLearningState) -> float:
        """Get adaptive learning rate based on recent performance"""
        recent_performance = state.get_recent_performance(15)  # Last 15 minutes
        
        if not recent_performance:
            return self.base_learning_rate
        
        # Increase learning rate if performance is poor
        avg_accuracy = np.mean([f.accuracy for f in recent_performance])
        
        if avg_accuracy < 0.7:  # Poor performance
            return self.base_learning_rate * 1.5
        elif avg_accuracy > 0.9:  # Excellent performance
            return self.base_learning_rate * 0.5
        else:
            return self.base_learning_rate


class RegimeAwareLearningStrategy(RealTimeLearningStrategy):
    """Regime-aware learning that adapts based on market conditions"""
    
    def __init__(self, base_learning_rate: float = 0.01):
        self.base_learning_rate = base_learning_rate
        self.regime_learning_rates = {
            MarketRegime.HIGH_VOLATILITY: 0.02,    # Learn faster in volatile markets
            MarketRegime.LOW_VOLATILITY: 0.005,    # Learn slower in stable markets
            MarketRegime.BREAKOUT: 0.03,           # Fastest learning during breakouts
            MarketRegime.RANGING: 0.01,            # Standard learning in ranges
            MarketRegime.TRENDING_BULLISH: 0.015,  # Moderate learning in trends
            MarketRegime.TRENDING_BEARISH: 0.015,
            MarketRegime.REVERSAL: 0.025,          # Fast learning during reversals
            MarketRegime.UNKNOWN: 0.005            # Conservative learning when uncertain
        }
    
    async def update_weights(self, 
                           state: AdaptiveLearningState,
                           feedback: PerformanceFeedback) -> Dict[str, float]:
        """Update weights based on current market regime"""
        
        # Determine current regime
        current_regime = feedback.actual_regime
        learning_rate = self.get_learning_rate(state)
        
        # Get regime-specific performance
        regime_performance = state.regime_performance.get(current_regime, 0.8)
        
        updated_weights = {}
        
        for weight_name, current_weight in state.component_weights.items():
            # Calculate regime-aware weight adjustment
            weight_adjustment = self._calculate_regime_adjustment(
                weight_name, current_regime, feedback, regime_performance
            )
            
            # Apply learning rate
            new_weight = current_weight + (learning_rate * weight_adjustment)
            
            # Apply regime-specific constraints
            new_weight = self._apply_regime_constraints(weight_name, new_weight, current_regime)
            
            updated_weights[weight_name] = new_weight
        
        # Update regime performance tracking
        state.regime_performance[current_regime] = (
            0.9 * regime_performance + 0.1 * feedback.accuracy
        )
        
        return updated_weights
    
    def _calculate_regime_adjustment(self, 
                                   weight_name: str, 
                                   regime: MarketRegime,
                                   feedback: PerformanceFeedback,
                                   regime_performance: float) -> float:
        """Calculate regime-specific weight adjustment"""
        
        # Base adjustment based on prediction accuracy
        base_adjustment = (feedback.accuracy - 0.5) * 2  # Scale to [-1, 1]
        
        # Regime-specific adjustments
        regime_multipliers = {
            MarketRegime.HIGH_VOLATILITY: {
                'gamma': 1.5,  # Increase gamma weight in volatile markets
                'vega': 1.3,   # Increase vega weight
                'delta': 0.8,  # Decrease delta weight
                'theta': 1.0
            },
            MarketRegime.LOW_VOLATILITY: {
                'gamma': 0.7,  # Decrease gamma weight in stable markets
                'vega': 0.6,   # Decrease vega weight
                'delta': 1.2,  # Increase delta weight
                'theta': 1.1   # Slight increase in theta
            },
            MarketRegime.TRENDING_BULLISH: {
                'gamma': 1.0,
                'vega': 0.9,
                'delta': 1.4,  # Strong delta emphasis in trends
                'theta': 0.8
            },
            MarketRegime.TRENDING_BEARISH: {
                'gamma': 1.0,
                'vega': 0.9,
                'delta': 1.4,  # Strong delta emphasis in trends
                'theta': 0.8
            },
            MarketRegime.BREAKOUT: {
                'gamma': 1.8,  # Maximum gamma weight for pin risk
                'vega': 1.5,   # High vega for volatility expansion
                'delta': 1.2,
                'theta': 0.5   # Lower theta weight
            }
        }
        
        # Get weight type
        weight_type = None
        for greek in ['gamma', 'delta', 'vega', 'theta']:
            if greek in weight_name.lower():
                weight_type = greek
                break
        
        if weight_type and regime in regime_multipliers:
            multiplier = regime_multipliers[regime].get(weight_type, 1.0)
            return base_adjustment * multiplier
        
        return base_adjustment
    
    def _apply_regime_constraints(self, weight_name: str, weight: float, regime: MarketRegime) -> float:
        """Apply regime-specific weight constraints"""
        
        # Ensure gamma weight never goes below 1.0 (critical for Component 2)
        if 'gamma' in weight_name.lower():
            weight = max(1.0, min(2.0, weight))
        else:
            weight = max(0.1, min(2.0, weight))
        
        # Regime-specific constraints
        if regime == MarketRegime.HIGH_VOLATILITY:
            if 'vega' in weight_name.lower():
                weight = max(0.8, weight)  # Minimum vega weight in volatile markets
        elif regime == MarketRegime.LOW_VOLATILITY:
            if 'theta' in weight_name.lower():
                weight = min(1.5, weight)  # Maximum theta weight in stable markets
        
        return weight
    
    def get_learning_rate(self, state: AdaptiveLearningState) -> float:
        """Get regime-aware learning rate"""
        recent_feedback = state.get_recent_performance(10)
        
        if not recent_feedback:
            return self.base_learning_rate
        
        # Get most recent regime
        latest_regime = recent_feedback[-1].actual_regime
        return self.regime_learning_rates.get(latest_regime, self.base_learning_rate)


class RealTimeAdaptiveLearningEngine:
    """
    Real-time adaptive learning engine for market regime components
    
    Continuously updates component weights based on streaming performance
    feedback with regime-aware learning strategies.
    """
    
    def __init__(self, 
                 learning_strategy: RealTimeLearningStrategy,
                 learning_mode: LearningMode = LearningMode.ADAPTIVE):
        """
        Initialize real-time adaptive learning engine
        
        Args:
            learning_strategy: Strategy for weight updates
            learning_mode: Learning mode configuration
        """
        self.learning_strategy = learning_strategy
        self.learning_mode = learning_mode
        self.logger = logging.getLogger(__name__)
        
        # Learning state for each component
        self.component_states: Dict[int, AdaptiveLearningState] = {}
        
        # Real-time feedback queue
        self.feedback_queue = asyncio.Queue()
        
        # Background learning task
        self.learning_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Performance callbacks
        self.performance_callbacks: List[Callable] = []
        
        self.logger.info(f"RealTimeAdaptiveLearningEngine initialized with {learning_strategy.__class__.__name__}")
    
    def initialize_component(self, 
                           component_id: int,
                           initial_weights: Dict[str, float],
                           learning_config: Optional[Dict[str, Any]] = None) -> AdaptiveLearningState:
        """
        Initialize adaptive learning for a component
        
        Args:
            component_id: Component identifier
            initial_weights: Initial weight configuration
            learning_config: Optional learning configuration
            
        Returns:
            AdaptiveLearningState for the component
        """
        config = learning_config or {}
        
        state = AdaptiveLearningState(
            component_weights=initial_weights.copy(),
            learning_rate=config.get('learning_rate', 0.01),
            momentum=config.get('momentum', 0.9),
            decay_rate=config.get('decay_rate', 0.99)
        )
        
        self.component_states[component_id] = state
        self.logger.info(f"Initialized adaptive learning for component {component_id}")
        
        return state
    
    async def start_learning(self):
        """Start the real-time learning process"""
        if self.is_running:
            self.logger.warning("Learning engine already running")
            return
        
        self.is_running = True
        self.learning_task = asyncio.create_task(self._learning_loop())
        self.logger.info("Real-time learning engine started")
    
    async def stop_learning(self):
        """Stop the real-time learning process"""
        self.is_running = False
        
        if self.learning_task:
            self.learning_task.cancel()
            try:
                await self.learning_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Real-time learning engine stopped")
    
    async def submit_feedback(self, feedback: PerformanceFeedback):
        """
        Submit performance feedback for learning
        
        Args:
            feedback: Performance feedback from component
        """
        await self.feedback_queue.put(feedback)
    
    def get_current_weights(self, component_id: int) -> Optional[Dict[str, float]]:
        """
        Get current weights for a component
        
        Args:
            component_id: Component identifier
            
        Returns:
            Current weights or None if component not initialized
        """
        if component_id in self.component_states:
            return self.component_states[component_id].component_weights.copy()
        return None
    
    def register_performance_callback(self, callback: Callable):
        """Register callback for performance updates"""
        self.performance_callbacks.append(callback)
    
    async def _learning_loop(self):
        """Main learning loop that processes feedback"""
        self.logger.info("Starting real-time learning loop")
        
        try:
            while self.is_running:
                try:
                    # Wait for feedback with timeout
                    feedback = await asyncio.wait_for(
                        self.feedback_queue.get(), 
                        timeout=1.0
                    )
                    
                    # Process the feedback
                    await self._process_feedback(feedback)
                    
                except asyncio.TimeoutError:
                    # No feedback received, continue
                    continue
                except Exception as e:
                    self.logger.error(f"Error processing feedback: {e}")
                    continue
                
        except asyncio.CancelledError:
            self.logger.info("Learning loop cancelled")
            raise
    
    async def _process_feedback(self, feedback: PerformanceFeedback):
        """Process a single feedback instance"""
        component_id = feedback.component_id
        
        if component_id not in self.component_states:
            self.logger.warning(f"No learning state for component {component_id}")
            return
        
        state = self.component_states[component_id]
        
        # Add feedback to state
        state.add_feedback(feedback)
        
        # Update weights using the learning strategy
        try:
            updated_weights = await self.learning_strategy.update_weights(state, feedback)
            
            # Apply weight updates
            old_weights = state.component_weights.copy()
            state.component_weights.update(updated_weights)
            state.update_count += 1
            
            # Log significant weight changes
            self._log_weight_changes(component_id, old_weights, updated_weights)
            
            # Notify performance callbacks
            await self._notify_performance_callbacks(component_id, feedback, updated_weights)
            
        except Exception as e:
            self.logger.error(f"Weight update failed for component {component_id}: {e}")
    
    def _log_weight_changes(self, 
                           component_id: int, 
                           old_weights: Dict[str, float],
                           new_weights: Dict[str, float]):
        """Log significant weight changes"""
        significant_changes = []
        
        for weight_name in new_weights:
            old_value = old_weights.get(weight_name, 0.0)
            new_value = new_weights[weight_name]
            
            if abs(new_value - old_value) > 0.05:  # 5% threshold
                change_pct = ((new_value - old_value) / old_value) * 100 if old_value != 0 else 0
                significant_changes.append(f"{weight_name}: {old_value:.3f} -> {new_value:.3f} ({change_pct:+.1f}%)")
        
        if significant_changes:
            self.logger.info(f"Component {component_id} weight updates: {', '.join(significant_changes)}")
    
    async def _notify_performance_callbacks(self,
                                          component_id: int,
                                          feedback: PerformanceFeedback,
                                          updated_weights: Dict[str, float]):
        """Notify registered performance callbacks"""
        for callback in self.performance_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(component_id, feedback, updated_weights)
                else:
                    callback(component_id, feedback, updated_weights)
            except Exception as e:
                self.logger.error(f"Performance callback failed: {e}")
    
    def get_learning_statistics(self, component_id: int) -> Dict[str, Any]:
        """Get learning statistics for a component"""
        if component_id not in self.component_states:
            return {}
        
        state = self.component_states[component_id]
        recent_performance = state.get_recent_performance(60)  # Last hour
        
        if not recent_performance:
            return {'status': 'no_recent_data'}
        
        accuracies = [f.accuracy for f in recent_performance]
        
        return {
            'component_id': component_id,
            'update_count': state.update_count,
            'last_update': state.last_update.isoformat(),
            'recent_feedback_count': len(recent_performance),
            'avg_accuracy': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'current_weights': state.component_weights.copy(),
            'regime_performance': dict(state.regime_performance),
            'learning_rate': self.learning_strategy.get_learning_rate(state)
        }


# Factory function for creating learning engines
def create_realtime_learning_engine(
    strategy_type: str = "gradient_descent",
    learning_mode: LearningMode = LearningMode.ADAPTIVE,
    **kwargs
) -> RealTimeAdaptiveLearningEngine:
    """
    Factory function to create real-time learning engines
    
    Args:
        strategy_type: Type of learning strategy ('gradient_descent', 'regime_aware')
        learning_mode: Learning mode configuration
        **kwargs: Additional strategy-specific parameters
        
    Returns:
        Configured RealTimeAdaptiveLearningEngine
    """
    
    if strategy_type == "gradient_descent":
        strategy = GradientDescentStrategy(
            base_learning_rate=kwargs.get('learning_rate', 0.01),
            momentum=kwargs.get('momentum', 0.9)
        )
    elif strategy_type == "regime_aware":
        strategy = RegimeAwareLearningStrategy(
            base_learning_rate=kwargs.get('learning_rate', 0.01)
        )
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    return RealTimeAdaptiveLearningEngine(strategy, learning_mode)


# Example usage and integration
async def example_integration():
    """Example of how to integrate real-time learning with Component 2"""
    
    # Create learning engine
    engine = create_realtime_learning_engine(
        strategy_type="regime_aware",
        learning_mode=LearningMode.ACTIVE,
        learning_rate=0.015
    )
    
    # Initialize Component 2 learning
    component_2_weights = {
        'gamma_weight': 1.5,  # Critical gamma weight
        'delta_weight': 1.0,
        'theta_weight': 0.8,
        'vega_weight': 1.2
    }
    
    engine.initialize_component(2, component_2_weights)
    
    # Start learning
    await engine.start_learning()
    
    # Simulate feedback (in practice, this comes from component analysis)
    feedback = PerformanceFeedback(
        timestamp=datetime.utcnow(),
        component_id=2,
        predicted_regime=MarketRegime.HIGH_VOLATILITY,
        actual_regime=MarketRegime.HIGH_VOLATILITY,
        accuracy=0.85,
        confidence=0.9,
        prediction_error=0.15,
        market_conditions={'vix': 25.0, 'trend_strength': 0.8},
        processing_time_ms=95.0
    )
    
    await engine.submit_feedback(feedback)
    
    # Get updated weights
    updated_weights = engine.get_current_weights(2)
    print(f"Updated weights: {updated_weights}")
    
    # Get learning statistics
    stats = engine.get_learning_statistics(2)
    print(f"Learning statistics: {stats}")
    
    # Stop learning
    await engine.stop_learning()


if __name__ == "__main__":
    # Run example
    asyncio.run(example_integration())