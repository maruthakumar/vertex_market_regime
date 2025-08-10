"""
Grid Search Optimization Algorithm

Exhaustive search through a grid of parameter combinations.
"""

import numpy as np
import logging
import time
import itertools
from typing import Dict, List, Tuple, Optional, Any, Callable
from ...base.base_optimizer import BaseOptimizer, OptimizationResult

logger = logging.getLogger(__name__)

class GridSearchOptimizer(BaseOptimizer):
    """
    Grid Search optimization algorithm
    
    Exhaustively evaluates all combinations of parameters on a predefined grid.
    Guarantees finding the global optimum within the grid resolution.
    """
    
    def __init__(self, 
                 param_space: Dict[str, Tuple[float, float]],
                 objective_function: callable,
                 maximize: bool = True,
                 grid_points: int = 10,
                 custom_grids: Optional[Dict[str, List[float]]] = None,
                 **kwargs):
        """
        Initialize Grid Search optimizer
        
        Args:
            param_space: Dictionary mapping parameter names to (min, max) bounds
            objective_function: Function to optimize
            maximize: Whether to maximize (True) or minimize (False)
            grid_points: Number of points per dimension (if custom_grids not provided)
            custom_grids: Custom grid points for each parameter
        """
        super().__init__(param_space, objective_function, maximize, **kwargs)
        
        self.grid_points = grid_points
        self.custom_grids = custom_grids or {}
        
        if grid_points < 2:
            raise ValueError("Grid points must be at least 2")
        
        # Generate parameter grids
        self.parameter_grids = self._create_parameter_grids()
        self.total_combinations = self._calculate_total_combinations()
        
        logger.info(f"Initialized Grid Search with {self.total_combinations:,} total combinations")
        logger.info(f"Grid points per dimension: {grid_points}")
    
    def _create_parameter_grids(self) -> Dict[str, List[float]]:
        """
        Create parameter grids for each dimension
        
        Returns:
            Dictionary mapping parameter names to grid point lists
        """
        grids = {}
        
        for param_name, (min_val, max_val) in self.param_space.items():
            if param_name in self.custom_grids:
                # Use custom grid if provided
                grid = self.custom_grids[param_name]
                # Validate custom grid is within bounds
                if min(grid) < min_val or max(grid) > max_val:
                    logger.warning(f"Custom grid for {param_name} extends beyond bounds")
                grids[param_name] = grid
            else:
                # Create uniform grid
                grids[param_name] = np.linspace(min_val, max_val, self.grid_points).tolist()
        
        return grids
    
    def _calculate_total_combinations(self) -> int:
        """Calculate total number of parameter combinations"""
        total = 1
        for grid in self.parameter_grids.values():
            total *= len(grid)
        return total
    
    def _generate_all_combinations(self) -> List[Dict[str, float]]:
        """
        Generate all parameter combinations
        
        Returns:
            List of all parameter combinations
        """
        param_names = list(self.parameter_grids.keys())
        grid_values = [self.parameter_grids[name] for name in param_names]
        
        combinations = []
        for combo in itertools.product(*grid_values):
            param_dict = {param_names[i]: combo[i] for i in range(len(param_names))}
            combinations.append(param_dict)
        
        return combinations
    
    def optimize(self, 
                 n_iterations: int = None,
                 callback: Optional[callable] = None,
                 batch_size: Optional[int] = None,
                 **kwargs) -> OptimizationResult:
        """
        Run grid search optimization
        
        Args:
            n_iterations: Ignored for grid search (evaluates all combinations)
            callback: Optional callback function called each iteration
            batch_size: Number of combinations to evaluate in each batch
            **kwargs: Additional algorithm-specific parameters
            
        Returns:
            OptimizationResult with best parameters and metadata
        """
        self.start_time = time.time()
        
        if n_iterations is not None and n_iterations < self.total_combinations:
            logger.warning(f"n_iterations ({n_iterations}) < total combinations "
                         f"({self.total_combinations}). Will evaluate all combinations.")
        
        logger.info(f"Starting Grid Search optimization")
        logger.info(f"Total combinations to evaluate: {self.total_combinations:,}")
        
        # Generate all combinations
        all_combinations = self._generate_all_combinations()
        
        # Set default batch size
        if batch_size is None:
            batch_size = min(1000, self.total_combinations)
        
        evaluated_count = 0
        batch_count = 0
        
        # Process in batches to manage memory and provide progress updates
        for i in range(0, len(all_combinations), batch_size):
            batch = all_combinations[i:i + batch_size]
            batch_count += 1
            
            logger.debug(f"Evaluating batch {batch_count} ({len(batch)} combinations)")
            
            # Evaluate each combination in batch
            for combo in batch:
                score = self.objective_function(combo)
                
                # Update best if improved
                self._update_best(combo, score)
                
                # Track progress
                self.iteration_history.append(score)
                evaluated_count += 1
                
                # Call callback if provided
                if callback:
                    callback(evaluated_count, combo, score)
                
                # Log progress
                if evaluated_count % 1000 == 0:
                    progress_pct = (evaluated_count / self.total_combinations) * 100
                    logger.info(f"Progress: {evaluated_count:,}/{self.total_combinations:,} "
                              f"({progress_pct:.1f}%). Best score: {self.best_score:.6f}")
        
        # Create result
        result = self._create_result(evaluated_count)
        result.metadata.update({
            'algorithm_params': {
                'grid_points': self.grid_points,
                'total_combinations': self.total_combinations,
                'batch_size': batch_size,
                'total_batches': batch_count,
                'exhaustive_search': True
            },
            'grid_info': {
                'parameter_grids': {k: len(v) for k, v in self.parameter_grids.items()},
                'custom_grids_used': list(self.custom_grids.keys())
            },
            'coverage': {
                'combinations_evaluated': evaluated_count,
                'coverage_percentage': 100.0  # Grid search is exhaustive
            }
        })
        
        logger.info(f"Grid Search completed. Best score: {self.best_score:.6f}")
        logger.info(f"Evaluated all {evaluated_count:,} combinations")
        
        return result
    
    def get_estimated_runtime(self, avg_eval_time: float) -> float:
        """
        Estimate total runtime based on average evaluation time
        
        Args:
            avg_eval_time: Average time per function evaluation (seconds)
            
        Returns:
            Estimated total runtime in seconds
        """
        return self.total_combinations * avg_eval_time
    
    def get_memory_estimate(self, param_size_bytes: int = 8) -> float:
        """
        Estimate memory usage for storing all combinations
        
        Args:
            param_size_bytes: Bytes per parameter value
            
        Returns:
            Estimated memory usage in MB
        """
        total_params = self.total_combinations * len(self.param_space)
        bytes_needed = total_params * param_size_bytes
        return bytes_needed / (1024 * 1024)  # Convert to MB
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return algorithm metadata for registry"""
        return {
            'name': 'Grid Search',
            'category': 'classical',
            'description': 'Exhaustive search through parameter grid',
            'supports_gpu': True,  # Can benefit from parallel evaluation
            'supports_parallel': True,
            'complexity': 'O(k^n)',  # k grid points, n dimensions
            'best_for': [
                'small parameter spaces',
                'guaranteed global optimum',
                'comprehensive analysis',
                'baseline comparison'
            ],
            'hyperparameters': [
                {
                    'name': 'grid_points',
                    'type': 'int',
                    'range': (2, 100),
                    'default': 10,
                    'description': 'Number of points per dimension'
                },
                {
                    'name': 'custom_grids',
                    'type': 'dict',
                    'default': None,
                    'description': 'Custom grid points for specific parameters'
                }
            ],
            'advantages': [
                'Guaranteed to find global optimum within grid',
                'Comprehensive coverage of parameter space',
                'Highly parallelizable',
                'Reproducible results'
            ],
            'disadvantages': [
                'Exponential growth with dimensions',
                'Can be computationally expensive',
                'Limited by grid resolution',
                'Curse of dimensionality'
            ],
            'warnings': [
                'Computational cost grows exponentially with dimensions',
                'May require significant memory for large grids',
                'Consider using coarse grid first for exploration'
            ],
            'references': [
                'Larson, J. et al. (2019). Derivative-free optimization methods',
                'Nocedal, J. & Wright, S. (2006). Numerical Optimization'
            ]
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], **kwargs) -> 'GridSearchOptimizer':
        """Create optimizer from configuration dictionary"""
        grid_config = config.get('grid_search', {})
        
        return cls(
            grid_points=grid_config.get('grid_points', 10),
            custom_grids=grid_config.get('custom_grids'),
            **kwargs
        )