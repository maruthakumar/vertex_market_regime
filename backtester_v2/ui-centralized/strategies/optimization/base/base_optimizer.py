"""
Base Optimizer Class

Abstract base class that all optimization algorithms must inherit from.
Provides common interface and shared functionality.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
import time
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """Container for optimization results"""
    best_params: Dict[str, float]
    best_score: float
    n_iterations: int
    convergence_history: List[float]
    execution_time: float
    algorithm_name: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary"""
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_iterations': self.n_iterations,
            'convergence_history': self.convergence_history,
            'execution_time': self.execution_time,
            'algorithm_name': self.algorithm_name,
            'metadata': self.metadata
        }

class BaseOptimizer(ABC):
    """
    Abstract base class for all optimization algorithms
    
    Provides common interface and functionality that all optimizers share:
    - Parameter space handling
    - Objective function management  
    - Result tracking
    - GPU acceleration hooks
    - Robustness integration points
    """
    
    def __init__(self, 
                 param_space: Dict[str, Tuple[float, float]],
                 objective_function: callable,
                 maximize: bool = True,
                 use_gpu: bool = False,
                 random_seed: Optional[int] = None,
                 convergence_tolerance: float = 1e-6,
                 max_stagnation: int = 100,
                 **kwargs):
        """
        Initialize base optimizer
        
        Args:
            param_space: Dictionary mapping parameter names to (min, max) bounds
            objective_function: Function to optimize that takes params dict and returns float
            maximize: Whether to maximize (True) or minimize (False) the objective
            use_gpu: Whether to use GPU acceleration if available
            random_seed: Random seed for reproducibility
            convergence_tolerance: Tolerance for convergence detection
            max_stagnation: Maximum iterations without improvement before stopping
        """
        self.param_space = param_space
        self.objective_function = objective_function
        self.maximize = maximize
        self.use_gpu = use_gpu
        self.random_seed = random_seed
        self.convergence_tolerance = convergence_tolerance
        self.max_stagnation = max_stagnation
        
        # Initialize tracking variables
        self.iteration_history = []
        self.best_score = float('-inf') if maximize else float('inf')
        self.best_params = {}
        self.stagnation_count = 0
        self.start_time = None
        
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
            
        # Validate parameter space
        self._validate_param_space()
        
        logger.info(f"Initialized {self.__class__.__name__} with {len(param_space)} parameters")
    
    def _validate_param_space(self):
        """Validate parameter space configuration"""
        if not self.param_space:
            raise ValueError("Parameter space cannot be empty")
            
        for param_name, bounds in self.param_space.items():
            if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
                raise ValueError(f"Parameter {param_name} bounds must be (min, max) tuple")
                
            min_val, max_val = bounds
            if min_val >= max_val:
                raise ValueError(f"Parameter {param_name}: min ({min_val}) must be < max ({max_val})")
    
    def _check_convergence(self, current_score: float) -> bool:
        """Check if optimization has converged"""
        if len(self.iteration_history) < 2:
            return False
            
        # Check for improvement
        is_better = (current_score > self.best_score if self.maximize 
                    else current_score < self.best_score)
        
        if is_better:
            improvement = abs(current_score - self.best_score)
            if improvement < self.convergence_tolerance:
                return True
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1
            
        # Check for stagnation
        return self.stagnation_count >= self.max_stagnation
    
    def _update_best(self, params: Dict[str, float], score: float):
        """Update best parameters and score if improved"""
        is_better = (score > self.best_score if self.maximize 
                    else score < self.best_score)
        
        if is_better or not self.best_params:
            self.best_score = score
            self.best_params = params.copy()
            logger.debug(f"New best score: {score:.6f}")
    
    def _validate_params(self, params: Dict[str, float]) -> bool:
        """Validate that parameters are within bounds"""
        for param_name, value in params.items():
            if param_name not in self.param_space:
                return False
                
            min_val, max_val = self.param_space[param_name]
            if not (min_val <= value <= max_val):
                return False
                
        return True
    
    def _clip_params(self, params: Dict[str, float]) -> Dict[str, float]:
        """Clip parameters to valid bounds"""
        clipped = {}
        for param_name, value in params.items():
            if param_name in self.param_space:
                min_val, max_val = self.param_space[param_name]
                clipped[param_name] = np.clip(value, min_val, max_val)
            else:
                clipped[param_name] = value
        return clipped
    
    def _random_params(self) -> Dict[str, float]:
        """Generate random parameters within bounds"""
        params = {}
        for param_name, (min_val, max_val) in self.param_space.items():
            params[param_name] = np.random.uniform(min_val, max_val)
        return params
    
    @abstractmethod
    def optimize(self, 
                 n_iterations: int = 1000,
                 callback: Optional[callable] = None,
                 **kwargs) -> OptimizationResult:
        """
        Main optimization method that must be implemented by subclasses
        
        Args:
            n_iterations: Maximum number of iterations
            callback: Optional callback function called each iteration
            **kwargs: Algorithm-specific parameters
            
        Returns:
            OptimizationResult with best parameters and metadata
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Return algorithm metadata for registry
        
        Returns:
            Dictionary containing algorithm information:
            - name: Algorithm name
            - category: Algorithm category (classical, evolutionary, etc.)
            - supports_gpu: Whether GPU acceleration is supported
            - supports_parallel: Whether parallel execution is supported
            - complexity: Big-O complexity description
            - best_for: List of problem types this algorithm excels at
            - hyperparameters: Available hyperparameters
        """
        pass
    
    def _create_result(self, n_iterations: int) -> OptimizationResult:
        """Create optimization result object"""
        execution_time = time.time() - self.start_time if self.start_time else 0
        
        return OptimizationResult(
            best_params=self.best_params,
            best_score=self.best_score,
            n_iterations=n_iterations,
            convergence_history=self.iteration_history.copy(),
            execution_time=execution_time,
            algorithm_name=self.__class__.__name__,
            metadata=self.get_metadata()
        )
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(params={len(self.param_space)}, maximize={self.maximize})"