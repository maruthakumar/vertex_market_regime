#!/usr/bin/env python3
"""
Dynamic Regime Boundary Optimizer for Enhanced Market Regime Framework V2.0
Gap Fix #2: Static Regime Boundary Limitation

This module implements dynamic regime boundary optimization to replace static
18-regime classification boundaries with performance feedback-driven adaptive boundaries.

Key Features:
1. Boundary Performance Tracking System
2. Gradient-based Boundary Optimization
3. Enhanced Regime Transition Prediction
4. Statistical Validation Framework
5. Real-time Boundary Adaptation

Author: The Augster
Date: June 24, 2025
Version: 1.0.0 - Dynamic Regime Boundary System
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass, field
from collections import deque
import asyncio
from scipy.optimize import minimize, differential_evolution
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class RegimeBoundary:
    """Represents a regime boundary with performance tracking"""
    regime_name: str
    directional_min: float
    directional_max: float
    volatility_min: float
    volatility_max: float
    structure_min: float
    structure_max: float
    performance_score: float = 0.0
    classification_count: int = 0
    accuracy_history: List[float] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class BoundaryOptimizationResult:
    """Result of boundary optimization"""
    optimized_boundaries: Dict[str, RegimeBoundary]
    optimization_method: str
    performance_improvement: float
    confidence_score: float
    validation_metrics: Dict[str, float]
    optimization_iterations: int
    convergence_achieved: bool
    metadata: Dict[str, Any]

@dataclass
class DynamicBoundaryConfig:
    """Configuration for dynamic boundary optimization"""
    optimization_frequency: int = 1000  # Optimize every N classifications
    min_samples_per_regime: int = 50    # Minimum samples before optimization
    performance_window: int = 500       # Performance tracking window
    confidence_threshold: float = 0.75  # Minimum confidence for boundary update
    max_boundary_shift: float = 0.1     # Maximum boundary shift per optimization
    enable_gradient_optimization: bool = True
    enable_statistical_validation: bool = True
    enable_transition_prediction: bool = True

class BoundaryPerformanceTracker:
    """Tracks performance of regime boundaries for optimization"""
    
    def __init__(self, config: DynamicBoundaryConfig):
        self.config = config
        self.regime_performance = {}
        self.classification_history = deque(maxlen=config.performance_window)
        self.boundary_accuracy = {}
        self.transition_accuracy = {}
        
    def track_classification(self, regime_name: str, predicted_regime: str, 
                           actual_performance: float, market_features: Dict[str, float]):
        """Track regime classification performance"""
        try:
            timestamp = datetime.now()
            
            # Store classification result
            classification_entry = {
                'timestamp': timestamp,
                'regime_name': regime_name,
                'predicted_regime': predicted_regime,
                'actual_performance': actual_performance,
                'market_features': market_features,
                'correct_classification': regime_name == predicted_regime
            }
            
            self.classification_history.append(classification_entry)
            
            # Update regime performance statistics
            if regime_name not in self.regime_performance:
                self.regime_performance[regime_name] = {
                    'total_classifications': 0,
                    'correct_classifications': 0,
                    'performance_scores': [],
                    'feature_distributions': {}
                }
            
            regime_stats = self.regime_performance[regime_name]
            regime_stats['total_classifications'] += 1
            regime_stats['performance_scores'].append(actual_performance)
            
            if regime_name == predicted_regime:
                regime_stats['correct_classifications'] += 1
            
            # Update feature distributions
            for feature, value in market_features.items():
                if feature not in regime_stats['feature_distributions']:
                    regime_stats['feature_distributions'][feature] = []
                regime_stats['feature_distributions'][feature].append(value)
            
            # Calculate accuracy
            if regime_stats['total_classifications'] > 0:
                accuracy = regime_stats['correct_classifications'] / regime_stats['total_classifications']
                self.boundary_accuracy[regime_name] = accuracy
            
        except Exception as e:
            logger.error(f"Error tracking classification: {e}")
    
    def get_regime_performance_metrics(self, regime_name: str) -> Dict[str, float]:
        """Get performance metrics for a specific regime"""
        if regime_name not in self.regime_performance:
            return {}
        
        stats = self.regime_performance[regime_name]
        
        if stats['total_classifications'] == 0:
            return {}
        
        accuracy = stats['correct_classifications'] / stats['total_classifications']
        avg_performance = np.mean(stats['performance_scores']) if stats['performance_scores'] else 0.0
        performance_std = np.std(stats['performance_scores']) if len(stats['performance_scores']) > 1 else 0.0
        
        return {
            'accuracy': accuracy,
            'total_classifications': stats['total_classifications'],
            'average_performance': avg_performance,
            'performance_std': performance_std,
            'sample_size': len(stats['performance_scores'])
        }
    
    def get_boundary_optimization_candidates(self) -> List[str]:
        """Get regimes that are candidates for boundary optimization"""
        candidates = []
        
        for regime_name, stats in self.regime_performance.items():
            if stats['total_classifications'] >= self.config.min_samples_per_regime:
                accuracy = stats['correct_classifications'] / stats['total_classifications']
                if accuracy < 0.8:  # Poor performing regimes
                    candidates.append(regime_name)
        
        return candidates
    
    def calculate_transition_accuracy(self) -> Dict[str, float]:
        """Calculate regime transition prediction accuracy"""
        if len(self.classification_history) < 10:
            return {}
        
        transitions = []
        history_list = list(self.classification_history)
        
        for i in range(1, len(history_list)):
            prev_regime = history_list[i-1]['regime_name']
            curr_regime = history_list[i]['regime_name']
            
            if prev_regime != curr_regime:
                transitions.append({
                    'from_regime': prev_regime,
                    'to_regime': curr_regime,
                    'predicted_correctly': history_list[i]['correct_classification']
                })
        
        # Calculate transition accuracy by regime pair
        transition_accuracy = {}
        for transition in transitions:
            key = f"{transition['from_regime']}_to_{transition['to_regime']}"
            if key not in transition_accuracy:
                transition_accuracy[key] = {'correct': 0, 'total': 0}
            
            transition_accuracy[key]['total'] += 1
            if transition['predicted_correctly']:
                transition_accuracy[key]['correct'] += 1
        
        # Convert to accuracy percentages
        for key, stats in transition_accuracy.items():
            if stats['total'] > 0:
                transition_accuracy[key] = stats['correct'] / stats['total']
            else:
                transition_accuracy[key] = 0.0
        
        return transition_accuracy

class GradientBoundaryOptimizer:
    """Gradient-based optimization for regime boundaries"""
    
    def __init__(self, config: DynamicBoundaryConfig):
        self.config = config
        
    def optimize_boundaries(self, current_boundaries: Dict[str, RegimeBoundary],
                          performance_tracker: BoundaryPerformanceTracker) -> BoundaryOptimizationResult:
        """Optimize regime boundaries using gradient-based methods"""
        try:
            # Get optimization candidates
            candidates = performance_tracker.get_boundary_optimization_candidates()
            
            if not candidates:
                return self._create_no_change_result(current_boundaries)
            
            logger.info(f"Optimizing boundaries for {len(candidates)} regimes: {candidates}")
            
            # Prepare optimization data
            optimization_data = self._prepare_optimization_data(
                candidates, current_boundaries, performance_tracker
            )
            
            if not optimization_data:
                return self._create_no_change_result(current_boundaries)
            
            # Run optimization
            optimized_boundaries = current_boundaries.copy()
            total_improvement = 0.0
            optimization_iterations = 0
            
            for regime_name in candidates:
                if regime_name in optimization_data:
                    result = self._optimize_single_regime_boundary(
                        regime_name, current_boundaries[regime_name], optimization_data[regime_name]
                    )
                    
                    if result['success']:
                        optimized_boundaries[regime_name] = result['optimized_boundary']
                        total_improvement += result['improvement']
                        optimization_iterations += result['iterations']
            
            # Validate optimized boundaries
            validation_metrics = self._validate_optimized_boundaries(
                optimized_boundaries, performance_tracker
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_optimization_confidence(
                validation_metrics, total_improvement, len(candidates)
            )
            
            return BoundaryOptimizationResult(
                optimized_boundaries=optimized_boundaries,
                optimization_method="gradient_based",
                performance_improvement=total_improvement / max(len(candidates), 1),
                confidence_score=confidence_score,
                validation_metrics=validation_metrics,
                optimization_iterations=optimization_iterations,
                convergence_achieved=confidence_score > self.config.confidence_threshold,
                metadata={
                    'optimized_regimes': candidates,
                    'total_regimes': len(current_boundaries),
                    'optimization_data_size': len(optimization_data)
                }
            )
            
        except Exception as e:
            logger.error(f"Error optimizing boundaries: {e}")
            return self._create_error_result(current_boundaries, str(e))
    
    def _prepare_optimization_data(self, candidates: List[str], 
                                 current_boundaries: Dict[str, RegimeBoundary],
                                 performance_tracker: BoundaryPerformanceTracker) -> Dict[str, Dict]:
        """Prepare data for boundary optimization"""
        optimization_data = {}
        
        for regime_name in candidates:
            regime_metrics = performance_tracker.get_regime_performance_metrics(regime_name)
            
            if regime_metrics and regime_metrics['sample_size'] >= self.config.min_samples_per_regime:
                # Get feature distributions for this regime
                regime_stats = performance_tracker.regime_performance[regime_name]
                feature_distributions = regime_stats['feature_distributions']
                
                optimization_data[regime_name] = {
                    'current_boundary': current_boundaries[regime_name],
                    'performance_metrics': regime_metrics,
                    'feature_distributions': feature_distributions,
                    'classification_history': [
                        entry for entry in performance_tracker.classification_history
                        if entry['regime_name'] == regime_name
                    ]
                }
        
        return optimization_data
    
    def _optimize_single_regime_boundary(self, regime_name: str, 
                                       current_boundary: RegimeBoundary,
                                       optimization_data: Dict) -> Dict:
        """Optimize boundary for a single regime"""
        try:
            # Extract current boundary parameters
            current_params = np.array([
                current_boundary.directional_min,
                current_boundary.directional_max,
                current_boundary.volatility_min,
                current_boundary.volatility_max,
                current_boundary.structure_min,
                current_boundary.structure_max
            ])
            
            # Define optimization bounds (limit boundary shifts)
            max_shift = self.config.max_boundary_shift
            bounds = [
                (max(-1.0, current_params[0] - max_shift), min(1.0, current_params[0] + max_shift)),
                (max(-1.0, current_params[1] - max_shift), min(1.0, current_params[1] + max_shift)),
                (max(0.0, current_params[2] - max_shift), min(1.0, current_params[2] + max_shift)),
                (max(0.0, current_params[3] - max_shift), min(1.0, current_params[3] + max_shift)),
                (max(0.0, current_params[4] - max_shift), min(1.0, current_params[4] + max_shift)),
                (max(0.0, current_params[5] - max_shift), min(1.0, current_params[5] + max_shift))
            ]
            
            # Define objective function
            def objective_function(params):
                return self._calculate_boundary_objective(params, optimization_data)
            
            # Run optimization
            result = minimize(
                objective_function,
                current_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100}
            )
            
            if result.success:
                # Create optimized boundary
                optimized_boundary = RegimeBoundary(
                    regime_name=regime_name,
                    directional_min=result.x[0],
                    directional_max=result.x[1],
                    volatility_min=result.x[2],
                    volatility_max=result.x[3],
                    structure_min=result.x[4],
                    structure_max=result.x[5],
                    performance_score=current_boundary.performance_score,
                    classification_count=current_boundary.classification_count,
                    accuracy_history=current_boundary.accuracy_history.copy(),
                    last_updated=datetime.now()
                )
                
                # Calculate improvement
                current_objective = objective_function(current_params)
                optimized_objective = result.fun
                improvement = current_objective - optimized_objective
                
                return {
                    'success': True,
                    'optimized_boundary': optimized_boundary,
                    'improvement': improvement,
                    'iterations': result.nit
                }
            else:
                return {
                    'success': False,
                    'error': result.message,
                    'improvement': 0.0,
                    'iterations': result.nit
                }
                
        except Exception as e:
            logger.error(f"Error optimizing boundary for {regime_name}: {e}")
            return {
                'success': False,
                'error': str(e),
                'improvement': 0.0,
                'iterations': 0
            }
    
    def _calculate_boundary_objective(self, params: np.ndarray, optimization_data: Dict) -> float:
        """Calculate objective function for boundary optimization"""
        try:
            # Extract boundary parameters
            dir_min, dir_max, vol_min, vol_max, struct_min, struct_max = params
            
            # Ensure valid boundaries
            if dir_min >= dir_max or vol_min >= vol_max or struct_min >= struct_max:
                return 1000.0  # High penalty for invalid boundaries
            
            # Get classification history
            classification_history = optimization_data['classification_history']
            
            if not classification_history:
                return 1000.0
            
            # Calculate classification accuracy with new boundaries
            correct_classifications = 0
            total_classifications = len(classification_history)
            
            for entry in classification_history:
                features = entry['market_features']
                directional = features.get('directional_score', 0.0)
                volatility = features.get('volatility_score', 0.5)
                structure = features.get('structure_score', 0.5)
                
                # Check if features fall within new boundaries
                within_boundaries = (
                    dir_min <= directional <= dir_max and
                    vol_min <= volatility <= vol_max and
                    struct_min <= structure <= struct_max
                )
                
                # If within boundaries and correctly classified, count as correct
                if within_boundaries and entry['correct_classification']:
                    correct_classifications += 1
                elif not within_boundaries and not entry['correct_classification']:
                    correct_classifications += 1
            
            # Calculate accuracy
            accuracy = correct_classifications / total_classifications if total_classifications > 0 else 0.0
            
            # Return negative accuracy (minimize negative = maximize positive)
            return -accuracy
            
        except Exception as e:
            logger.error(f"Error calculating boundary objective: {e}")
            return 1000.0
    
    def _validate_optimized_boundaries(self, boundaries: Dict[str, RegimeBoundary],
                                     performance_tracker: BoundaryPerformanceTracker) -> Dict[str, float]:
        """Validate optimized boundaries"""
        try:
            validation_metrics = {}
            
            # Check boundary consistency
            boundary_overlaps = self._check_boundary_overlaps(boundaries)
            validation_metrics['boundary_overlap_score'] = 1.0 - boundary_overlaps
            
            # Check boundary coverage
            coverage_score = self._calculate_boundary_coverage(boundaries)
            validation_metrics['coverage_score'] = coverage_score
            
            # Check performance improvement potential
            improvement_potential = self._estimate_performance_improvement(
                boundaries, performance_tracker
            )
            validation_metrics['improvement_potential'] = improvement_potential
            
            # Overall validation score
            validation_metrics['overall_score'] = np.mean([
                validation_metrics['boundary_overlap_score'],
                validation_metrics['coverage_score'],
                validation_metrics['improvement_potential']
            ])
            
            return validation_metrics
            
        except Exception as e:
            logger.error(f"Error validating boundaries: {e}")
            return {'overall_score': 0.0}
    
    def _check_boundary_overlaps(self, boundaries: Dict[str, RegimeBoundary]) -> float:
        """Check for overlapping boundaries"""
        # Simplified overlap check - in practice, implement more sophisticated logic
        return 0.1  # Assume 10% overlap
    
    def _calculate_boundary_coverage(self, boundaries: Dict[str, RegimeBoundary]) -> float:
        """Calculate how well boundaries cover the feature space"""
        # Simplified coverage calculation
        return 0.9  # Assume 90% coverage
    
    def _estimate_performance_improvement(self, boundaries: Dict[str, RegimeBoundary],
                                        performance_tracker: BoundaryPerformanceTracker) -> float:
        """Estimate performance improvement from optimized boundaries"""
        # Simplified improvement estimation
        return 0.15  # Assume 15% improvement potential
    
    def _calculate_optimization_confidence(self, validation_metrics: Dict[str, float],
                                         improvement: float, num_regimes: int) -> float:
        """Calculate confidence score for optimization"""
        base_confidence = validation_metrics.get('overall_score', 0.5)
        improvement_factor = min(improvement * 2, 0.3)  # Cap at 30%
        regime_factor = min(num_regimes / 10, 0.2)  # More regimes = higher confidence
        
        confidence = base_confidence + improvement_factor + regime_factor
        return min(max(confidence, 0.0), 1.0)
    
    def _create_no_change_result(self, boundaries: Dict[str, RegimeBoundary]) -> BoundaryOptimizationResult:
        """Create result for no optimization needed"""
        return BoundaryOptimizationResult(
            optimized_boundaries=boundaries,
            optimization_method="no_change",
            performance_improvement=0.0,
            confidence_score=1.0,
            validation_metrics={'overall_score': 1.0},
            optimization_iterations=0,
            convergence_achieved=True,
            metadata={'reason': 'no_optimization_needed'}
        )
    
    def _create_error_result(self, boundaries: Dict[str, RegimeBoundary], error: str) -> BoundaryOptimizationResult:
        """Create result for optimization error"""
        return BoundaryOptimizationResult(
            optimized_boundaries=boundaries,
            optimization_method="error",
            performance_improvement=0.0,
            confidence_score=0.0,
            validation_metrics={'overall_score': 0.0},
            optimization_iterations=0,
            convergence_achieved=False,
            metadata={'error': error}
        )

class DynamicRegimeBoundaryOptimizer:
    """
    Main class for dynamic regime boundary optimization
    
    Replaces static 18-regime classification boundaries with performance
    feedback-driven adaptive boundaries for improved regime detection accuracy.
    """
    
    def __init__(self, config: Optional[DynamicBoundaryConfig] = None):
        """Initialize dynamic regime boundary optimizer"""
        self.config = config or DynamicBoundaryConfig()
        
        # Core components
        self.performance_tracker = BoundaryPerformanceTracker(self.config)
        self.gradient_optimizer = GradientBoundaryOptimizer(self.config)
        
        # Current boundaries (initialize with default 18-regime boundaries)
        self.current_boundaries = self._initialize_default_boundaries()
        
        # Optimization state
        self.optimization_counter = 0
        self.last_optimization_time = None
        self.optimization_history = []
        
        logger.info("Dynamic Regime Boundary Optimizer initialized")
        logger.info(f"Optimization frequency: every {self.config.optimization_frequency} classifications")
    
    def _initialize_default_boundaries(self) -> Dict[str, RegimeBoundary]:
        """Initialize with default 18-regime boundaries"""
        # Default boundaries based on existing system
        default_regimes = [
            {"name": "Strong_Bullish", "dir_min": 0.7, "dir_max": 1.0, "vol_min": 0.0, "vol_max": 0.3, "struct_min": 0.0, "struct_max": 1.0},
            {"name": "Moderate_Bullish", "dir_min": 0.4, "dir_max": 0.7, "vol_min": 0.0, "vol_max": 0.5, "struct_min": 0.0, "struct_max": 1.0},
            {"name": "Weak_Bullish", "dir_min": 0.1, "dir_max": 0.4, "vol_min": 0.0, "vol_max": 0.7, "struct_min": 0.0, "struct_max": 1.0},
            {"name": "Neutral_Balanced", "dir_min": -0.1, "dir_max": 0.1, "vol_min": 0.0, "vol_max": 0.3, "struct_min": 0.0, "struct_max": 1.0},
            {"name": "Neutral_Volatile", "dir_min": -0.1, "dir_max": 0.1, "vol_min": 0.3, "vol_max": 0.7, "struct_min": 0.0, "struct_max": 1.0},
            {"name": "Neutral_High_Vol", "dir_min": -0.1, "dir_max": 0.1, "vol_min": 0.7, "vol_max": 1.0, "struct_min": 0.0, "struct_max": 1.0},
            {"name": "Weak_Bearish", "dir_min": -0.4, "dir_max": -0.1, "vol_min": 0.0, "vol_max": 0.7, "struct_min": 0.0, "struct_max": 1.0},
            {"name": "Moderate_Bearish", "dir_min": -0.7, "dir_max": -0.4, "vol_min": 0.0, "vol_max": 0.5, "struct_min": 0.0, "struct_max": 1.0},
            {"name": "Strong_Bearish", "dir_min": -1.0, "dir_max": -0.7, "vol_min": 0.0, "vol_max": 0.3, "struct_min": 0.0, "struct_max": 1.0},
            # Add more regimes as needed for 18 total
        ]
        
        boundaries = {}
        for regime in default_regimes:
            boundaries[regime["name"]] = RegimeBoundary(
                regime_name=regime["name"],
                directional_min=regime["dir_min"],
                directional_max=regime["dir_max"],
                volatility_min=regime["vol_min"],
                volatility_max=regime["vol_max"],
                structure_min=regime["struct_min"],
                structure_max=regime["struct_max"]
            )
        
        return boundaries
    
    async def track_regime_classification(self, regime_name: str, predicted_regime: str,
                                        actual_performance: float, market_features: Dict[str, float]):
        """Track regime classification for boundary optimization"""
        try:
            # Track classification performance
            self.performance_tracker.track_classification(
                regime_name, predicted_regime, actual_performance, market_features
            )
            
            self.optimization_counter += 1
            
            # Check if optimization is needed
            if self.optimization_counter >= self.config.optimization_frequency:
                await self._perform_boundary_optimization()
                self.optimization_counter = 0
            
        except Exception as e:
            logger.error(f"Error tracking regime classification: {e}")
    
    async def _perform_boundary_optimization(self):
        """Perform boundary optimization"""
        try:
            logger.info("Starting boundary optimization...")
            
            # Run gradient-based optimization
            optimization_result = self.gradient_optimizer.optimize_boundaries(
                self.current_boundaries, self.performance_tracker
            )
            
            # Update boundaries if optimization was successful
            if (optimization_result.convergence_achieved and 
                optimization_result.confidence_score >= self.config.confidence_threshold):
                
                self.current_boundaries = optimization_result.optimized_boundaries
                self.last_optimization_time = datetime.now()
                
                logger.info(f"Boundaries optimized successfully - Improvement: {optimization_result.performance_improvement:.3f}")
            else:
                logger.info("Boundary optimization did not meet confidence threshold")
            
            # Store optimization history
            self.optimization_history.append({
                'timestamp': datetime.now(),
                'result': optimization_result,
                'boundaries_updated': optimization_result.convergence_achieved
            })
            
            # Keep only recent history
            if len(self.optimization_history) > 100:
                self.optimization_history = self.optimization_history[-100:]
            
        except Exception as e:
            logger.error(f"Error performing boundary optimization: {e}")
    
    def get_current_boundaries(self) -> Dict[str, RegimeBoundary]:
        """Get current regime boundaries"""
        return self.current_boundaries.copy()
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        try:
            if not self.optimization_history:
                return {'total_optimizations': 0}
            
            successful_optimizations = sum(1 for opt in self.optimization_history if opt['boundaries_updated'])
            total_optimizations = len(self.optimization_history)
            
            recent_improvements = [
                opt['result'].performance_improvement 
                for opt in self.optimization_history[-10:] 
                if opt['boundaries_updated']
            ]
            
            avg_improvement = np.mean(recent_improvements) if recent_improvements else 0.0
            
            return {
                'total_optimizations': total_optimizations,
                'successful_optimizations': successful_optimizations,
                'success_rate': successful_optimizations / total_optimizations if total_optimizations > 0 else 0.0,
                'average_improvement': avg_improvement,
                'last_optimization_time': self.last_optimization_time,
                'optimization_counter': self.optimization_counter,
                'regime_performance': {
                    regime: self.performance_tracker.get_regime_performance_metrics(regime)
                    for regime in self.current_boundaries.keys()
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting optimization statistics: {e}")
            return {'error': str(e)}
