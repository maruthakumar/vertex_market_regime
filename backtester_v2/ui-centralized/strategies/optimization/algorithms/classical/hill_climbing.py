"""
Hill Climbing Optimization Algorithm

Migrated from enhanced-market-regime-optimizer-final-package-updated and adapted
to the new BaseOptimizer interface.
"""

import numpy as np
import random
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
from ...base.base_optimizer import BaseOptimizer, OptimizationResult

logger = logging.getLogger(__name__)

class HillClimbingOptimizer(BaseOptimizer):
    """
    Hill Climbing optimization algorithm with multiple restarts
    
    Hill climbing is a local search algorithm that iteratively moves toward 
    better solutions by making small changes to the current solution.
    Multiple restarts help avoid getting stuck in local optima.
    """
    
    def __init__(self, 
                 param_space: Dict[str, Tuple[float, float]],
                 objective_function: callable,
                 maximize: bool = True,
                 step_size: float = 0.1,
                 n_restarts: int = 5,
                 **kwargs):
        """
        Initialize Hill Climbing optimizer
        
        Args:
            param_space: Dictionary mapping parameter names to (min, max) bounds
            objective_function: Function to optimize
            maximize: Whether to maximize (True) or minimize (False)
            step_size: Step size for neighborhood generation (0.0 to 1.0)
            n_restarts: Number of random restarts
        """
        super().__init__(param_space, objective_function, maximize, **kwargs)
        
        self.step_size = step_size
        self.n_restarts = n_restarts
        
        # Validate step size
        if not 0.0 < step_size <= 1.0:
            raise ValueError("Step size must be between 0.0 and 1.0")
            
        if n_restarts < 1:
            raise ValueError("Number of restarts must be at least 1")
        
        logger.info(f"Initialized Hill Climbing with step_size={step_size}, restarts={n_restarts}")
    
    def _generate_neighbor(self, current_params: Dict[str, float]) -> Dict[str, float]:
        """
        Generate a neighbor solution by modifying one random parameter
        
        Args:
            current_params: Current parameter values
            
        Returns:
            Dictionary of neighbor parameter values
        """
        neighbor = current_params.copy()
        
        # Select random parameter to modify
        param_name = random.choice(list(self.param_space.keys()))
        min_val, max_val = self.param_space[param_name]
        
        # Calculate step size for this parameter
        param_range = max_val - min_val
        delta = random.uniform(-self.step_size, self.step_size) * param_range
        
        # Apply modification and clip to bounds
        new_value = current_params[param_name] + delta
        neighbor[param_name] = np.clip(new_value, min_val, max_val)
        
        return neighbor
    
    def _hill_climb_single_run(self, 
                              max_iterations: int,
                              callback: Optional[Callable] = None) -> Tuple[Dict[str, float], float, int]:
        """
        Perform single hill climbing run
        
        Args:
            max_iterations: Maximum iterations for this run
            callback: Optional callback function
            
        Returns:
            Tuple of (best_params, best_score, iterations)
        """
        # Initialize random starting point
        current_params = self._random_params()
        current_score = self.objective_function(current_params)
        
        iterations = 0
        stagnation_count = 0
        
        while iterations < max_iterations:
            # Generate neighbor
            neighbor_params = self._generate_neighbor(current_params)
            neighbor_score = self.objective_function(neighbor_params)
            
            # Check if neighbor is better
            is_better = (neighbor_score > current_score if self.maximize 
                        else neighbor_score < current_score)
            
            if is_better:
                current_params = neighbor_params
                current_score = neighbor_score
                stagnation_count = 0
                
                logger.debug(f"Iteration {iterations}: Improved to {current_score:.6f}")
            else:
                stagnation_count += 1
            
            # Track iteration
            self.iteration_history.append(current_score)
            iterations += 1
            
            # Call callback if provided
            if callback:
                callback(iterations, current_params, current_score)
            
            # Check for convergence
            if self._check_convergence(current_score):
                logger.info(f"Hill climbing converged after {iterations} iterations")
                break
        
        return current_params, current_score, iterations
    
    def optimize(self, 
                 n_iterations: int = 1000,
                 callback: Optional[callable] = None,
                 **kwargs) -> OptimizationResult:
        """
        Run hill climbing optimization with multiple restarts
        
        Args:
            n_iterations: Maximum number of iterations
            callback: Optional callback function called each iteration
            **kwargs: Additional algorithm-specific parameters
            
        Returns:
            OptimizationResult with best parameters and metadata
        """
        self.start_time = time.time()
        
        logger.info(f"Starting Hill Climbing optimization with {self.n_restarts} restarts")
        logger.info(f"Max iterations per restart: {n_iterations}")
        
        iterations_per_restart = n_iterations // self.n_restarts
        total_iterations = 0
        all_runs = []
        
        # Run multiple restarts
        for restart in range(self.n_restarts):
            logger.info(f"Hill Climbing restart {restart + 1}/{self.n_restarts}")
            
            # Run single hill climbing
            params, score, iterations = self._hill_climb_single_run(
                iterations_per_restart, callback
            )
            
            total_iterations += iterations
            all_runs.append({
                'restart': restart,
                'params': params,
                'score': score,
                'iterations': iterations
            })
            
            # Update global best
            self._update_best(params, score)
            
            logger.info(f"Restart {restart + 1} completed: score={score:.6f}")
        
        # Find best run
        best_run = max(all_runs, key=lambda x: x['score'] if self.maximize else -x['score'])
        
        # Create result
        result = self._create_result(total_iterations)
        result.metadata.update({
            'algorithm_params': {
                'step_size': self.step_size,
                'n_restarts': self.n_restarts,
                'total_restarts': len(all_runs)
            },
            'restart_scores': [run['score'] for run in all_runs],
            'best_restart': best_run['restart'],
            'convergence_iterations': total_iterations
        })
        
        logger.info(f"Hill Climbing completed. Best score: {self.best_score:.6f}")
        return result
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return algorithm metadata for registry"""
        return {
            'name': 'Hill Climbing',
            'category': 'classical',
            'description': 'Local search algorithm with multiple restarts',
            'supports_gpu': False,
            'supports_parallel': True,
            'complexity': 'O(n * m * r)',  # n iterations, m neighbors, r restarts
            'best_for': [
                'local optimization', 
                'discrete spaces',
                'quick solutions',
                'parameter tuning'
            ],
            'hyperparameters': [
                {
                    'name': 'step_size',
                    'type': 'float',
                    'range': (0.0, 1.0),
                    'default': 0.1,
                    'description': 'Size of steps in parameter space'
                },
                {
                    'name': 'n_restarts', 
                    'type': 'int',
                    'range': (1, 100),
                    'default': 5,
                    'description': 'Number of random restarts'
                }
            ],
            'references': [
                'Russell, S. & Norvig, P. (2020). Artificial Intelligence: A Modern Approach',
                'Selman, B. & Gomes, C. (2006). Hill-climbing search'
            ]
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], **kwargs) -> 'HillClimbingOptimizer':
        """Create optimizer from configuration dictionary"""
        hill_config = config.get('hill_climbing', {})
        
        return cls(
            step_size=hill_config.get('step_size', 0.1),
            n_restarts=hill_config.get('restarts', 5),
            **kwargs
        )