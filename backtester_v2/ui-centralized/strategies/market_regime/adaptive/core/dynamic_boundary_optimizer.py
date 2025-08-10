"""
Dynamic Boundary Optimizer

This module optimizes regime boundaries based on historical performance and
real-time feedback, enabling adaptive regime boundary adjustment.

Key Features:
- Multi-objective optimization for regime boundaries
- Historical performance integration
- Boundary stability constraints
- Real-time adaptation based on prediction accuracy
- Hysteresis-based transition management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from collections import deque, defaultdict
import logging
from dataclasses import dataclass, field
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class RegimeBoundary:
    """Represents a regime boundary definition"""
    regime_id: int
    volatility_bounds: Tuple[float, float]
    trend_bounds: Tuple[float, float]
    volume_bounds: Tuple[float, float]
    confidence_threshold: float
    hysteresis_factor: float
    last_updated: datetime
    performance_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Results from boundary optimization"""
    optimized_boundaries: Dict[int, RegimeBoundary]
    objective_value: float
    convergence_status: bool
    iterations: int
    improvement: float
    optimization_time: float


@dataclass
class BoundaryUpdate:
    """Represents a boundary update event"""
    regime_id: int
    old_boundary: RegimeBoundary
    new_boundary: RegimeBoundary
    reason: str
    performance_delta: float
    timestamp: datetime


class DynamicBoundaryOptimizer:
    """
    Optimizes regime boundaries dynamically based on performance
    """
    
    def __init__(self, regime_count: int = 12,
                 optimization_window: int = 100,
                 update_frequency: int = 50):
        """
        Initialize boundary optimizer
        
        Args:
            regime_count: Number of regimes
            optimization_window: Window for performance evaluation
            update_frequency: How often to optimize boundaries
        """
        self.regime_count = regime_count
        self.optimization_window = optimization_window
        self.update_frequency = update_frequency
        
        # Boundary definitions
        self.current_boundaries: Dict[int, RegimeBoundary] = {}
        self.boundary_history = deque(maxlen=1000)
        
        # Performance tracking
        self.performance_buffer = deque(maxlen=optimization_window)
        self.regime_accuracy: Dict[int, float] = {i: 0.5 for i in range(regime_count)}
        self.transition_accuracy: Dict[Tuple[int, int], float] = {}
        
        # Optimization configuration
        self.optimization_method = 'differential_evolution'
        self.max_iterations = 100
        self.convergence_tolerance = 1e-4
        self.boundary_change_limit = 0.2  # Max 20% change per update
        
        # Hysteresis parameters
        self.base_hysteresis = 0.05
        self.adaptive_hysteresis = True
        self.hysteresis_history: Dict[int, deque] = {
            i: deque(maxlen=50) for i in range(regime_count)
        }
        
        # Initialize default boundaries
        self._initialize_boundaries()
        
        # Metrics
        self.total_optimizations = 0
        self.successful_optimizations = 0
        self.average_improvement = 0.0
        
        logger.info(f"DynamicBoundaryOptimizer initialized for {regime_count} regimes")
    
    def _initialize_boundaries(self):
        """Initialize default regime boundaries"""
        # Create evenly spaced boundaries as starting point
        for i in range(self.regime_count):
            # Volatility bounds (low to high)
            vol_low = i / self.regime_count * 0.4
            vol_high = (i + 1) / self.regime_count * 0.4
            
            # Trend bounds (bearish to bullish)
            trend_center = (i - self.regime_count/2) / (self.regime_count/2) * 0.02
            trend_width = 0.01
            trend_low = trend_center - trend_width
            trend_high = trend_center + trend_width
            
            # Volume bounds (normalized)
            volume_low = 0.5 + i / self.regime_count * 1.0
            volume_high = 0.5 + (i + 1) / self.regime_count * 1.0
            
            self.current_boundaries[i] = RegimeBoundary(
                regime_id=i,
                volatility_bounds=(vol_low, vol_high),
                trend_bounds=(trend_low, trend_high),
                volume_bounds=(volume_low, volume_high),
                confidence_threshold=0.6,
                hysteresis_factor=self.base_hysteresis,
                last_updated=datetime.now(),
                performance_score=0.5
            )
        
        logger.info("Default boundaries initialized")
    
    def optimize_boundaries(self, performance_data: List[Dict[str, Any]],
                          market_conditions: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        """
        Optimize regime boundaries based on performance data
        
        Args:
            performance_data: Recent performance metrics
            market_conditions: Current market conditions
            
        Returns:
            Optimization results
        """
        start_time = datetime.now()
        logger.info("Starting boundary optimization...")
        
        try:
            # Update performance metrics
            self._update_performance_metrics(performance_data)
            
            # Define optimization objectives
            objectives = self._define_objectives(market_conditions)
            
            # Perform optimization
            if self.optimization_method == 'differential_evolution':
                result = self._optimize_differential_evolution(objectives)
            else:
                result = self._optimize_scipy(objectives)
            
            # Apply constraints and validate
            validated_boundaries = self._validate_boundaries(result['boundaries'])
            
            # Calculate improvement
            improvement = self._calculate_improvement(validated_boundaries)
            
            # Update boundaries if improvement is significant
            if improvement > 0.01:  # 1% improvement threshold
                self._update_boundaries(validated_boundaries, "performance_optimization")
                self.successful_optimizations += 1
            
            self.total_optimizations += 1
            self.average_improvement = (
                self.average_improvement * 0.9 + improvement * 0.1
            )
            
            optimization_time = (datetime.now() - start_time).total_seconds()
            
            return OptimizationResult(
                optimized_boundaries=validated_boundaries,
                objective_value=result['objective'],
                convergence_status=result['converged'],
                iterations=result['iterations'],
                improvement=improvement,
                optimization_time=optimization_time
            )
            
        except Exception as e:
            logger.error(f"Error in boundary optimization: {e}")
            # Return current boundaries on error
            return OptimizationResult(
                optimized_boundaries=self.current_boundaries.copy(),
                objective_value=float('inf'),
                convergence_status=False,
                iterations=0,
                improvement=0.0,
                optimization_time=0.0
            )
    
    def _update_performance_metrics(self, performance_data: List[Dict[str, Any]]):
        """Update performance metrics from recent data"""
        for record in performance_data:
            self.performance_buffer.append(record)
        
        # Update regime accuracy
        regime_predictions = defaultdict(list)
        regime_actuals = defaultdict(list)
        
        for record in self.performance_buffer:
            predicted_regime = record.get('predicted_regime')
            actual_regime = record.get('actual_regime')
            
            if predicted_regime is not None and actual_regime is not None:
                regime_predictions[actual_regime].append(predicted_regime)
                regime_actuals[actual_regime].append(actual_regime)
        
        # Calculate accuracy for each regime
        for regime_id in range(self.regime_count):
            if regime_id in regime_predictions:
                predictions = regime_predictions[regime_id]
                actuals = regime_actuals[regime_id]
                correct = sum(1 for p, a in zip(predictions, actuals) if p == a)
                self.regime_accuracy[regime_id] = correct / len(predictions) if predictions else 0.5
    
    def _define_objectives(self, market_conditions: Optional[Dict[str, Any]]) -> Dict[str, Callable]:
        """Define optimization objectives"""
        objectives = {}
        
        # Primary objective: Maximize prediction accuracy
        def accuracy_objective(boundaries: np.ndarray) -> float:
            """Minimize negative accuracy (maximize accuracy)"""
            boundary_dict = self._array_to_boundaries(boundaries)
            accuracy_scores = []
            
            for regime_id, boundary in boundary_dict.items():
                # Simulate accuracy with new boundaries
                simulated_accuracy = self._simulate_accuracy(boundary)
                accuracy_scores.append(simulated_accuracy)
            
            return -np.mean(accuracy_scores)  # Negative for minimization
        
        objectives['accuracy'] = accuracy_objective
        
        # Secondary objective: Minimize boundary overlap
        def overlap_objective(boundaries: np.ndarray) -> float:
            """Minimize regime overlap"""
            boundary_dict = self._array_to_boundaries(boundaries)
            total_overlap = 0.0
            
            for i in range(self.regime_count):
                for j in range(i + 1, self.regime_count):
                    overlap = self._calculate_overlap(
                        boundary_dict[i], boundary_dict[j]
                    )
                    total_overlap += overlap
            
            return total_overlap
        
        objectives['overlap'] = overlap_objective
        
        # Tertiary objective: Maintain stability
        def stability_objective(boundaries: np.ndarray) -> float:
            """Minimize boundary changes"""
            boundary_dict = self._array_to_boundaries(boundaries)
            total_change = 0.0
            
            for regime_id, new_boundary in boundary_dict.items():
                old_boundary = self.current_boundaries[regime_id]
                change = self._calculate_boundary_change(old_boundary, new_boundary)
                total_change += change
            
            return total_change
        
        objectives['stability'] = stability_objective
        
        # Market-adaptive objective
        if market_conditions:
            def market_objective(boundaries: np.ndarray) -> float:
                """Adapt to current market conditions"""
                boundary_dict = self._array_to_boundaries(boundaries)
                market_score = 0.0
                
                current_vol = market_conditions.get('volatility', 0.2)
                current_trend = market_conditions.get('trend', 0.0)
                
                # Favor boundaries that better separate current conditions
                for regime_id, boundary in boundary_dict.items():
                    if (boundary.volatility_bounds[0] <= current_vol <= boundary.volatility_bounds[1] and
                        boundary.trend_bounds[0] <= current_trend <= boundary.trend_bounds[1]):
                        # Current regime should have high accuracy
                        market_score += self.regime_accuracy.get(regime_id, 0.5)
                
                return -market_score  # Negative for minimization
            
            objectives['market'] = market_objective
        
        return objectives
    
    def _optimize_differential_evolution(self, objectives: Dict[str, Callable]) -> Dict[str, Any]:
        """Perform optimization using differential evolution"""
        # Combine objectives with weights
        weights = {
            'accuracy': 0.5,
            'overlap': 0.2,
            'stability': 0.2,
            'market': 0.1
        }
        
        def combined_objective(x):
            total = 0.0
            for name, obj_func in objectives.items():
                weight = weights.get(name, 0.1)
                total += weight * obj_func(x)
            return total
        
        # Define bounds for optimization variables
        bounds = self._get_optimization_bounds()
        
        # Perform optimization
        result = differential_evolution(
            combined_objective,
            bounds,
            maxiter=self.max_iterations,
            tol=self.convergence_tolerance,
            seed=42,
            workers=1
        )
        
        return {
            'boundaries': self._array_to_boundaries(result.x),
            'objective': result.fun,
            'converged': result.success,
            'iterations': result.nit
        }
    
    def _optimize_scipy(self, objectives: Dict[str, Callable]) -> Dict[str, Any]:
        """Perform optimization using scipy minimize"""
        # Similar to differential evolution but using gradient-based method
        weights = {
            'accuracy': 0.5,
            'overlap': 0.2,
            'stability': 0.2,
            'market': 0.1
        }
        
        def combined_objective(x):
            total = 0.0
            for name, obj_func in objectives.items():
                weight = weights.get(name, 0.1)
                total += weight * obj_func(x)
            return total
        
        # Initial guess from current boundaries
        x0 = self._boundaries_to_array(self.current_boundaries)
        bounds = self._get_optimization_bounds()
        
        # Perform optimization
        result = minimize(
            combined_objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': self.max_iterations,
                'ftol': self.convergence_tolerance
            }
        )
        
        return {
            'boundaries': self._array_to_boundaries(result.x),
            'objective': result.fun,
            'converged': result.success,
            'iterations': result.nit
        }
    
    def _get_optimization_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds for optimization variables"""
        bounds = []
        
        for i in range(self.regime_count):
            # Volatility bounds (6 values per regime: vol_low, vol_high, trend_low, trend_high, volume_low, volume_high)
            bounds.extend([
                (0.0, 0.5),   # vol_low
                (0.0, 0.5),   # vol_high
                (-0.05, 0.05),  # trend_low
                (-0.05, 0.05),  # trend_high
                (0.0, 3.0),   # volume_low
                (0.0, 3.0)    # volume_high
            ])
        
        return bounds
    
    def _boundaries_to_array(self, boundaries: Dict[int, RegimeBoundary]) -> np.ndarray:
        """Convert boundary dictionary to optimization array"""
        arr = []
        
        for i in range(self.regime_count):
            boundary = boundaries[i]
            arr.extend([
                boundary.volatility_bounds[0],
                boundary.volatility_bounds[1],
                boundary.trend_bounds[0],
                boundary.trend_bounds[1],
                boundary.volume_bounds[0],
                boundary.volume_bounds[1]
            ])
        
        return np.array(arr)
    
    def _array_to_boundaries(self, arr: np.ndarray) -> Dict[int, RegimeBoundary]:
        """Convert optimization array to boundary dictionary"""
        boundaries = {}
        
        for i in range(self.regime_count):
            idx = i * 6
            boundaries[i] = RegimeBoundary(
                regime_id=i,
                volatility_bounds=(arr[idx], arr[idx + 1]),
                trend_bounds=(arr[idx + 2], arr[idx + 3]),
                volume_bounds=(arr[idx + 4], arr[idx + 5]),
                confidence_threshold=self.current_boundaries[i].confidence_threshold,
                hysteresis_factor=self.current_boundaries[i].hysteresis_factor,
                last_updated=datetime.now(),
                performance_score=self.current_boundaries[i].performance_score
            )
        
        return boundaries
    
    def _simulate_accuracy(self, boundary: RegimeBoundary) -> float:
        """Simulate accuracy for a given boundary"""
        # This is a simplified simulation
        # In practice, would test against historical data
        
        # Base accuracy from current performance
        base_accuracy = self.regime_accuracy.get(boundary.regime_id, 0.5)
        
        # Adjust based on boundary characteristics
        # Tighter boundaries generally improve accuracy but reduce coverage
        vol_range = boundary.volatility_bounds[1] - boundary.volatility_bounds[0]
        trend_range = boundary.trend_bounds[1] - boundary.trend_bounds[0]
        
        # Optimal ranges (learned from data)
        optimal_vol_range = 0.05
        optimal_trend_range = 0.01
        
        # Penalize deviation from optimal
        vol_penalty = abs(vol_range - optimal_vol_range) / optimal_vol_range
        trend_penalty = abs(trend_range - optimal_trend_range) / optimal_trend_range
        
        # Simulated accuracy
        simulated = base_accuracy * (1 - 0.1 * vol_penalty) * (1 - 0.1 * trend_penalty)
        
        return np.clip(simulated, 0.0, 1.0)
    
    def _calculate_overlap(self, boundary1: RegimeBoundary, 
                         boundary2: RegimeBoundary) -> float:
        """Calculate overlap between two boundaries"""
        # Check overlap in each dimension
        vol_overlap = self._range_overlap(
            boundary1.volatility_bounds, boundary2.volatility_bounds
        )
        trend_overlap = self._range_overlap(
            boundary1.trend_bounds, boundary2.trend_bounds
        )
        volume_overlap = self._range_overlap(
            boundary1.volume_bounds, boundary2.volume_bounds
        )
        
        # Total overlap is product of overlaps (all dimensions must overlap)
        total_overlap = vol_overlap * trend_overlap * volume_overlap
        
        return total_overlap
    
    def _range_overlap(self, range1: Tuple[float, float], 
                      range2: Tuple[float, float]) -> float:
        """Calculate overlap between two ranges"""
        overlap_start = max(range1[0], range2[0])
        overlap_end = min(range1[1], range2[1])
        
        if overlap_start >= overlap_end:
            return 0.0
        
        overlap_size = overlap_end - overlap_start
        range1_size = range1[1] - range1[0]
        range2_size = range2[1] - range2[0]
        
        # Normalized overlap
        if range1_size > 0 and range2_size > 0:
            return overlap_size / min(range1_size, range2_size)
        
        return 0.0
    
    def _calculate_boundary_change(self, old: RegimeBoundary, 
                                 new: RegimeBoundary) -> float:
        """Calculate change between boundaries"""
        changes = []
        
        # Volatility change
        vol_change = (
            abs(new.volatility_bounds[0] - old.volatility_bounds[0]) +
            abs(new.volatility_bounds[1] - old.volatility_bounds[1])
        ) / 2.0
        changes.append(vol_change)
        
        # Trend change
        trend_change = (
            abs(new.trend_bounds[0] - old.trend_bounds[0]) +
            abs(new.trend_bounds[1] - old.trend_bounds[1])
        ) / 2.0
        changes.append(trend_change)
        
        # Volume change
        volume_change = (
            abs(new.volume_bounds[0] - old.volume_bounds[0]) +
            abs(new.volume_bounds[1] - old.volume_bounds[1])
        ) / 2.0
        changes.append(volume_change)
        
        return np.mean(changes)
    
    def _validate_boundaries(self, boundaries: Dict[int, RegimeBoundary]) -> Dict[int, RegimeBoundary]:
        """Validate and adjust boundaries to ensure consistency"""
        validated = {}
        
        for regime_id, boundary in boundaries.items():
            # Ensure bounds are properly ordered
            vol_bounds = (
                min(boundary.volatility_bounds),
                max(boundary.volatility_bounds)
            )
            trend_bounds = (
                min(boundary.trend_bounds),
                max(boundary.trend_bounds)
            )
            volume_bounds = (
                min(boundary.volume_bounds),
                max(boundary.volume_bounds)
            )
            
            # Ensure minimum range
            min_range = 0.01
            if vol_bounds[1] - vol_bounds[0] < min_range:
                vol_bounds = (vol_bounds[0], vol_bounds[0] + min_range)
            if trend_bounds[1] - trend_bounds[0] < min_range:
                center = (trend_bounds[0] + trend_bounds[1]) / 2
                trend_bounds = (center - min_range/2, center + min_range/2)
            
            # Apply change limits
            old_boundary = self.current_boundaries[regime_id]
            
            # Limit changes to boundary_change_limit
            def limit_change(new_val, old_val):
                max_change = old_val * self.boundary_change_limit
                return np.clip(new_val, old_val - max_change, old_val + max_change)
            
            vol_bounds = (
                limit_change(vol_bounds[0], old_boundary.volatility_bounds[0]),
                limit_change(vol_bounds[1], old_boundary.volatility_bounds[1])
            )
            
            validated[regime_id] = RegimeBoundary(
                regime_id=regime_id,
                volatility_bounds=vol_bounds,
                trend_bounds=trend_bounds,
                volume_bounds=volume_bounds,
                confidence_threshold=boundary.confidence_threshold,
                hysteresis_factor=boundary.hysteresis_factor,
                last_updated=boundary.last_updated,
                performance_score=boundary.performance_score
            )
        
        return validated
    
    def _calculate_improvement(self, new_boundaries: Dict[int, RegimeBoundary]) -> float:
        """Calculate improvement from new boundaries"""
        improvements = []
        
        for regime_id, new_boundary in new_boundaries.items():
            old_score = self.current_boundaries[regime_id].performance_score
            new_score = self._simulate_accuracy(new_boundary)
            improvement = new_score - old_score
            improvements.append(improvement)
        
        return np.mean(improvements)
    
    def _update_boundaries(self, new_boundaries: Dict[int, RegimeBoundary], reason: str):
        """Update current boundaries and record history"""
        for regime_id, new_boundary in new_boundaries.items():
            old_boundary = self.current_boundaries[regime_id]
            
            # Calculate performance delta
            performance_delta = new_boundary.performance_score - old_boundary.performance_score
            
            # Record update
            update = BoundaryUpdate(
                regime_id=regime_id,
                old_boundary=old_boundary,
                new_boundary=new_boundary,
                reason=reason,
                performance_delta=performance_delta,
                timestamp=datetime.now()
            )
            self.boundary_history.append(update)
            
            # Update current boundary
            self.current_boundaries[regime_id] = new_boundary
        
        logger.info(f"Boundaries updated: {reason}")
    
    def update_hysteresis(self, transition_data: List[Dict[str, Any]]):
        """Update hysteresis factors based on transition patterns"""
        if not self.adaptive_hysteresis:
            return
        
        # Analyze false transitions (quick reversals)
        for record in transition_data:
            from_regime = record.get('from_regime')
            to_regime = record.get('to_regime')
            duration = record.get('duration', 0)
            reversed_quickly = record.get('reversed_quickly', False)
            
            if from_regime is not None and reversed_quickly:
                # Increase hysteresis for regimes with false transitions
                self.hysteresis_history[from_regime].append(1)
                
                # Update hysteresis factor
                false_rate = np.mean(list(self.hysteresis_history[from_regime]))
                new_hysteresis = self.base_hysteresis * (1 + false_rate)
                
                self.current_boundaries[from_regime].hysteresis_factor = min(
                    new_hysteresis, 0.15  # Cap at 15%
                )
    
    def get_regime_boundaries(self, regime_id: int) -> Optional[RegimeBoundary]:
        """Get current boundaries for a specific regime"""
        return self.current_boundaries.get(regime_id)
    
    def check_regime_transition(self, current_regime: int, 
                              market_data: Dict[str, Any]) -> Tuple[int, float]:
        """
        Check if regime transition should occur with hysteresis
        
        Args:
            current_regime: Current regime ID
            market_data: Current market conditions
            
        Returns:
            Tuple of (new_regime_id, confidence)
        """
        current_vol = market_data.get('volatility', 0.0)
        current_trend = market_data.get('trend', 0.0)
        current_volume = market_data.get('volume_ratio', 1.0)
        
        # Check all regimes
        regime_scores = {}
        
        for regime_id, boundary in self.current_boundaries.items():
            # Calculate distance from boundary center
            vol_center = (boundary.volatility_bounds[0] + boundary.volatility_bounds[1]) / 2
            trend_center = (boundary.trend_bounds[0] + boundary.trend_bounds[1]) / 2
            volume_center = (boundary.volume_bounds[0] + boundary.volume_bounds[1]) / 2
            
            # Normalized distances
            vol_dist = abs(current_vol - vol_center) / (boundary.volatility_bounds[1] - boundary.volatility_bounds[0])
            trend_dist = abs(current_trend - trend_center) / (boundary.trend_bounds[1] - boundary.trend_bounds[0])
            volume_dist = abs(current_volume - volume_center) / (boundary.volume_bounds[1] - boundary.volume_bounds[0])
            
            # Check if within bounds
            in_bounds = (
                boundary.volatility_bounds[0] <= current_vol <= boundary.volatility_bounds[1] and
                boundary.trend_bounds[0] <= current_trend <= boundary.trend_bounds[1] and
                boundary.volume_bounds[0] <= current_volume <= boundary.volume_bounds[1]
            )
            
            if in_bounds:
                # Score based on distance from center (closer = higher score)
                score = 1.0 - (vol_dist + trend_dist + volume_dist) / 3.0
                
                # Apply hysteresis if transitioning away from current regime
                if regime_id != current_regime:
                    hysteresis = self.current_boundaries[current_regime].hysteresis_factor
                    score *= (1.0 - hysteresis)
                
                regime_scores[regime_id] = score
        
        # Find best regime
        if regime_scores:
            best_regime = max(regime_scores.items(), key=lambda x: x[1])
            return best_regime[0], best_regime[1]
        
        # No valid regime found, stay in current
        return current_regime, 0.5
    
    def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get optimization performance metrics"""
        success_rate = (
            self.successful_optimizations / self.total_optimizations
            if self.total_optimizations > 0 else 0.0
        )
        
        # Recent boundary changes
        recent_changes = []
        for update in list(self.boundary_history)[-10:]:
            recent_changes.append({
                'regime_id': update.regime_id,
                'reason': update.reason,
                'performance_delta': update.performance_delta,
                'timestamp': update.timestamp
            })
        
        return {
            'total_optimizations': self.total_optimizations,
            'successful_optimizations': self.successful_optimizations,
            'success_rate': success_rate,
            'average_improvement': self.average_improvement,
            'regime_accuracy': self.regime_accuracy.copy(),
            'recent_changes': recent_changes,
            'current_hysteresis': {
                i: b.hysteresis_factor for i, b in self.current_boundaries.items()
            }
        }
    
    def export_boundaries(self, filepath: str):
        """Export current boundaries to file"""
        export_data = {
            'regime_count': self.regime_count,
            'boundaries': {},
            'metadata': {
                'last_optimization': max(
                    (b.last_updated for b in self.current_boundaries.values()),
                    default=datetime.now()
                ).isoformat(),
                'total_optimizations': self.total_optimizations,
                'average_improvement': self.average_improvement
            }
        }
        
        for regime_id, boundary in self.current_boundaries.items():
            export_data['boundaries'][regime_id] = {
                'volatility_bounds': boundary.volatility_bounds,
                'trend_bounds': boundary.trend_bounds,
                'volume_bounds': boundary.volume_bounds,
                'confidence_threshold': boundary.confidence_threshold,
                'hysteresis_factor': boundary.hysteresis_factor,
                'performance_score': boundary.performance_score
            }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Boundaries exported to {filepath}")


# Example usage
if __name__ == "__main__":
    # Create optimizer
    optimizer = DynamicBoundaryOptimizer(regime_count=8)
    
    # Generate sample performance data
    performance_data = []
    for i in range(100):
        performance_data.append({
            'predicted_regime': np.random.randint(0, 8),
            'actual_regime': np.random.randint(0, 8),
            'timestamp': datetime.now() - timedelta(minutes=i*5)
        })
    
    # Current market conditions
    market_conditions = {
        'volatility': 0.15,
        'trend': 0.002,
        'volume_ratio': 1.1
    }
    
    # Optimize boundaries
    result = optimizer.optimize_boundaries(performance_data, market_conditions)
    
    print("\nOptimization Results:")
    print(f"Converged: {result.convergence_status}")
    print(f"Iterations: {result.iterations}")
    print(f"Improvement: {result.improvement:.2%}")
    print(f"Time: {result.optimization_time:.2f}s")
    
    # Test regime transition
    current_regime = 3
    new_regime, confidence = optimizer.check_regime_transition(
        current_regime, market_conditions
    )
    print(f"\nRegime transition check:")
    print(f"Current: {current_regime}, Suggested: {new_regime}, Confidence: {confidence:.2f}")
    
    # Get metrics
    metrics = optimizer.get_optimization_metrics()
    print(f"\nOptimization metrics:")
    print(f"Success rate: {metrics['success_rate']:.2%}")
    print(f"Average improvement: {metrics['average_improvement']:.2%}")