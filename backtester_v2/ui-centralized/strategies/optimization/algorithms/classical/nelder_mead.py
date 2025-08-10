"""
Nelder-Mead Simplex Algorithm Implementation

A derivative-free optimization algorithm that uses a simplex approach
to find the minimum of a function. Well-suited for non-linear optimization
problems where gradients are not available.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import time

from ...base.base_optimizer import BaseOptimizer, OptimizationResult

logger = logging.getLogger(__name__)

@dataclass
class NelderMeadConfig:
    """Configuration for Nelder-Mead algorithm"""
    alpha: float = 1.0      # Reflection coefficient
    gamma: float = 2.0      # Expansion coefficient
    rho: float = 0.5        # Contraction coefficient
    sigma: float = 0.5      # Shrinkage coefficient
    tolerance: float = 1e-6  # Convergence tolerance
    max_iterations: int = 1000
    adaptive: bool = True    # Use adaptive parameters
    
class NelderMeadOptimizer(BaseOptimizer):
    """
    Nelder-Mead Simplex Optimization Algorithm
    
    A robust, derivative-free optimization method that maintains a simplex
    (n+1 points in n-dimensional space) and iteratively transforms it to
    converge to the minimum.
    
    Features:
    - Derivative-free optimization
    - Robust convergence for non-linear functions
    - Adaptive parameter tuning
    - Automatic simplex initialization
    - Comprehensive convergence criteria
    """
    
    def __init__(self,
                 param_space: Dict[str, Tuple[float, float]],
                 objective_function: Callable[[Dict[str, float]], float],
                 config: Optional[NelderMeadConfig] = None,
                 **kwargs):
        """
        Initialize Nelder-Mead optimizer
        
        Args:
            param_space: Parameter space definition
            objective_function: Function to minimize
            config: Algorithm configuration
            **kwargs: Additional parameters
        """
        super().__init__(param_space, objective_function)
        
        self.config = config or NelderMeadConfig()
        self.dimensions = len(param_space)
        self.param_names = list(param_space.keys())
        self.bounds = np.array(list(param_space.values()))
        
        # Algorithm state
        self.simplex = None
        self.simplex_values = None
        self.centroid = None
        self.best_point = None
        self.best_value = float('inf')
        
        # Performance tracking
        self.function_evaluations = 0
        self.iterations = 0
        self.convergence_history = []
        
        # Adaptive parameters
        if self.config.adaptive:
            self._setup_adaptive_parameters()
        
        logger.info(f"NelderMead optimizer initialized for {self.dimensions}D problem")
    
    def optimize(self, 
                n_iterations: Optional[int] = None,
                callback: Optional[Callable] = None) -> OptimizationResult:
        """
        Execute Nelder-Mead optimization
        
        Args:
            n_iterations: Maximum iterations (overrides config)
            callback: Progress callback function
            
        Returns:
            Optimization result
        """
        start_time = time.time()
        max_iterations = n_iterations or self.config.max_iterations
        
        logger.info(f"Starting Nelder-Mead optimization (max_iterations={max_iterations})")
        
        # Initialize simplex
        self._initialize_simplex()
        
        # Main optimization loop
        for iteration in range(max_iterations):
            self.iterations = iteration + 1
            
            # Sort simplex by function values
            self._sort_simplex()
            
            # Check convergence
            if self._check_convergence():
                logger.info(f"Converged after {iteration + 1} iterations")
                break
            
            # Nelder-Mead iteration
            self._nelder_mead_iteration()
            
            # Update best point
            self._update_best_point()
            
            # Track convergence
            self.convergence_history.append({
                'iteration': iteration + 1,
                'best_value': self.best_value,
                'simplex_std': np.std(self.simplex_values),
                'function_evaluations': self.function_evaluations
            })
            
            # Progress callback
            if callback:
                callback(iteration + 1, max_iterations, self.best_value)
        
        execution_time = time.time() - start_time
        
        # Create result
        best_params = self._array_to_dict(self.best_point)
        
        result = OptimizationResult(
            best_parameters=best_params,
            best_objective_value=self.best_value,
            iterations=self.iterations,
            function_evaluations=self.function_evaluations,
            execution_time=execution_time,
            convergence_status='converged' if self._check_convergence() else 'max_iterations',
            optimization_history=self.convergence_history,
            metadata={
                'algorithm': 'nelder_mead',
                'dimensions': self.dimensions,
                'config': {
                    'alpha': self.config.alpha,
                    'gamma': self.config.gamma,
                    'rho': self.config.rho,
                    'sigma': self.config.sigma,
                    'tolerance': self.config.tolerance,
                    'adaptive': self.config.adaptive
                },
                'simplex_final_std': np.std(self.simplex_values),
                'convergence_rate': self._calculate_convergence_rate()
            }
        )
        
        logger.info(f"Optimization completed: {result.convergence_status}")
        logger.info(f"Best value: {result.best_objective_value:.6f}")
        logger.info(f"Function evaluations: {result.function_evaluations}")
        
        return result
    
    def _initialize_simplex(self):
        """Initialize the simplex with n+1 points"""
        self.simplex = np.zeros((self.dimensions + 1, self.dimensions))
        self.simplex_values = np.zeros(self.dimensions + 1)
        
        # First point: random point in parameter space
        self.simplex[0] = self._random_point()
        self.simplex_values[0] = self._evaluate_point(self.simplex[0])
        
        # Remaining points: perturb first point along each dimension
        for i in range(1, self.dimensions + 1):
            self.simplex[i] = self.simplex[0].copy()
            
            # Adaptive step size based on parameter range
            dim_idx = i - 1
            param_range = self.bounds[dim_idx, 1] - self.bounds[dim_idx, 0]
            step_size = param_range * 0.1  # 10% of range
            
            # Perturb along dimension
            self.simplex[i, dim_idx] += step_size
            
            # Ensure bounds
            self.simplex[i] = self._enforce_bounds(self.simplex[i])
            
            # Evaluate
            self.simplex_values[i] = self._evaluate_point(self.simplex[i])
        
        logger.debug(f"Initialized simplex with {len(self.simplex)} points")
    
    def _nelder_mead_iteration(self):
        """Perform one iteration of Nelder-Mead algorithm"""
        # Sort simplex (best to worst)
        self._sort_simplex()
        
        # Calculate centroid (excluding worst point)
        self.centroid = np.mean(self.simplex[:-1], axis=0)
        
        # Get worst point
        worst_point = self.simplex[-1]
        worst_value = self.simplex_values[-1]
        
        # 1. Reflection
        reflected_point = self._reflect(worst_point, self.centroid)
        reflected_value = self._evaluate_point(reflected_point)
        
        best_value = self.simplex_values[0]
        second_worst_value = self.simplex_values[-2]
        
        if best_value <= reflected_value < second_worst_value:
            # Accept reflection
            self.simplex[-1] = reflected_point
            self.simplex_values[-1] = reflected_value
            return
        
        # 2. Expansion
        if reflected_value < best_value:
            expanded_point = self._expand(reflected_point, self.centroid)
            expanded_value = self._evaluate_point(expanded_point)
            
            if expanded_value < reflected_value:
                # Accept expansion
                self.simplex[-1] = expanded_point
                self.simplex_values[-1] = expanded_value
            else:
                # Accept reflection
                self.simplex[-1] = reflected_point
                self.simplex_values[-1] = reflected_value
            return
        
        # 3. Contraction
        if reflected_value < worst_value:
            # Outside contraction
            contracted_point = self._contract_outside(reflected_point, self.centroid)
            contracted_value = self._evaluate_point(contracted_point)
            
            if contracted_value <= reflected_value:
                self.simplex[-1] = contracted_point
                self.simplex_values[-1] = contracted_value
                return
        else:
            # Inside contraction
            contracted_point = self._contract_inside(worst_point, self.centroid)
            contracted_value = self._evaluate_point(contracted_point)
            
            if contracted_value < worst_value:
                self.simplex[-1] = contracted_point
                self.simplex_values[-1] = contracted_value
                return
        
        # 4. Shrinkage
        self._shrink()
    
    def _reflect(self, point: np.ndarray, centroid: np.ndarray) -> np.ndarray:
        """Reflect point through centroid"""
        reflected = centroid + self.config.alpha * (centroid - point)
        return self._enforce_bounds(reflected)
    
    def _expand(self, point: np.ndarray, centroid: np.ndarray) -> np.ndarray:
        """Expand point away from centroid"""
        expanded = centroid + self.config.gamma * (point - centroid)
        return self._enforce_bounds(expanded)
    
    def _contract_outside(self, point: np.ndarray, centroid: np.ndarray) -> np.ndarray:
        """Contract point toward centroid (outside)"""
        contracted = centroid + self.config.rho * (point - centroid)
        return self._enforce_bounds(contracted)
    
    def _contract_inside(self, point: np.ndarray, centroid: np.ndarray) -> np.ndarray:
        """Contract point toward centroid (inside)"""
        contracted = centroid + self.config.rho * (point - centroid)
        return self._enforce_bounds(contracted)
    
    def _shrink(self):
        """Shrink all points toward best point"""
        best_point = self.simplex[0]
        
        for i in range(1, len(self.simplex)):
            self.simplex[i] = best_point + self.config.sigma * (self.simplex[i] - best_point)
            self.simplex[i] = self._enforce_bounds(self.simplex[i])
            self.simplex_values[i] = self._evaluate_point(self.simplex[i])
    
    def _sort_simplex(self):
        """Sort simplex by function values (best to worst)"""
        indices = np.argsort(self.simplex_values)
        self.simplex = self.simplex[indices]
        self.simplex_values = self.simplex_values[indices]
    
    def _check_convergence(self) -> bool:
        """Check if algorithm has converged"""
        if len(self.simplex_values) < 2:
            return False
        
        # Standard deviation of function values
        std_values = np.std(self.simplex_values)
        
        # Standard deviation of simplex points
        std_points = np.std(self.simplex, axis=0).max()
        
        # Check both criteria
        value_converged = std_values < self.config.tolerance
        point_converged = std_points < self.config.tolerance
        
        return value_converged and point_converged
    
    def _update_best_point(self):
        """Update best point if improved"""
        current_best_idx = np.argmin(self.simplex_values)
        current_best_value = self.simplex_values[current_best_idx]
        
        if current_best_value < self.best_value:
            self.best_value = current_best_value
            self.best_point = self.simplex[current_best_idx].copy()
    
    def _setup_adaptive_parameters(self):
        """Setup adaptive parameters based on dimension"""
        # Adaptive parameters for higher dimensions
        if self.dimensions > 10:
            self.config.alpha = 1.0
            self.config.gamma = 1.0 + 2.0 / self.dimensions
            self.config.rho = 0.75 - 1.0 / (2.0 * self.dimensions)
            self.config.sigma = 1.0 - 1.0 / self.dimensions
        
        logger.debug(f"Adaptive parameters: α={self.config.alpha}, γ={self.config.gamma}, "
                    f"ρ={self.config.rho}, σ={self.config.sigma}")
    
    def _calculate_convergence_rate(self) -> float:
        """Calculate convergence rate from history"""
        if len(self.convergence_history) < 2:
            return 0.0
        
        # Calculate improvement rate
        initial_value = self.convergence_history[0]['best_value']
        final_value = self.convergence_history[-1]['best_value']
        
        if initial_value == final_value:
            return 0.0
        
        improvement = abs(initial_value - final_value)
        iterations = len(self.convergence_history)
        
        return improvement / iterations
    
    def _random_point(self) -> np.ndarray:
        """Generate random point in parameter space"""
        point = np.random.uniform(0, 1, self.dimensions)
        
        # Scale to bounds
        for i in range(self.dimensions):
            point[i] = self.bounds[i, 0] + point[i] * (self.bounds[i, 1] - self.bounds[i, 0])
        
        return point
    
    def _enforce_bounds(self, point: np.ndarray) -> np.ndarray:
        """Enforce parameter bounds"""
        bounded_point = point.copy()
        
        for i in range(self.dimensions):
            bounded_point[i] = np.clip(bounded_point[i], self.bounds[i, 0], self.bounds[i, 1])
        
        return bounded_point
    
    def _evaluate_point(self, point: np.ndarray) -> float:
        """Evaluate objective function at point"""
        params = self._array_to_dict(point)
        value = self.objective_function(params)
        self.function_evaluations += 1
        return value
    
    def _array_to_dict(self, point: np.ndarray) -> Dict[str, float]:
        """Convert numpy array to parameter dictionary"""
        return {name: point[i] for i, name in enumerate(self.param_names)}
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get algorithm information"""
        return {
            'name': 'Nelder-Mead',
            'type': 'Classical',
            'category': 'Derivative-free',
            'description': 'Simplex-based optimization algorithm',
            'parameters': {
                'alpha': self.config.alpha,
                'gamma': self.config.gamma,
                'rho': self.config.rho,
                'sigma': self.config.sigma,
                'tolerance': self.config.tolerance,
                'adaptive': self.config.adaptive
            },
            'suitable_for': [
                'Non-linear optimization',
                'Derivative-free problems',
                'Continuous variables',
                'Small to medium dimensions'
            ],
            'strengths': [
                'No gradient required',
                'Robust convergence',
                'Simple implementation',
                'Handles non-smooth functions'
            ],
            'limitations': [
                'Slow convergence for high dimensions',
                'Can get stuck in local minima',
                'Requires many function evaluations'
            ]
        }