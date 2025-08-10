"""
Simulated Annealing Optimization Algorithm

Physics-inspired optimization based on the annealing process in metallurgy.
"""

import numpy as np
import random
import logging
import time
import math
from typing import Dict, List, Tuple, Optional, Any, Callable
from ...base.base_optimizer import BaseOptimizer, OptimizationResult

logger = logging.getLogger(__name__)

class SimulatedAnnealingOptimizer(BaseOptimizer):
    """
    Simulated Annealing optimization algorithm
    
    Physics-inspired optimization that accepts worse solutions with 
    probability that decreases over time (temperature cooling).
    """
    
    def __init__(self, 
                 param_space: Dict[str, Tuple[float, float]],
                 objective_function: callable,
                 maximize: bool = True,
                 initial_temperature: float = 100.0,
                 final_temperature: float = 0.01,
                 cooling_schedule: str = 'exponential',
                 cooling_rate: float = 0.95,
                 neighborhood_size: float = 0.1,
                 **kwargs):
        """
        Initialize Simulated Annealing optimizer
        
        Args:
            param_space: Dictionary mapping parameter names to (min, max) bounds
            objective_function: Function to optimize
            maximize: Whether to maximize (True) or minimize (False)
            initial_temperature: Starting temperature
            final_temperature: Ending temperature
            cooling_schedule: Cooling schedule ('linear', 'exponential', 'logarithmic')
            cooling_rate: Cooling rate for exponential schedule
            neighborhood_size: Size of neighborhood for generating new solutions
        """
        super().__init__(param_space, objective_function, maximize, **kwargs)
        
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.cooling_schedule = cooling_schedule
        self.cooling_rate = cooling_rate
        self.neighborhood_size = neighborhood_size
        
        # Validate parameters
        if initial_temperature <= 0:
            raise ValueError("Initial temperature must be positive")
        if final_temperature <= 0:
            raise ValueError("Final temperature must be positive")
        if initial_temperature <= final_temperature:
            raise ValueError("Initial temperature must be greater than final temperature")
        if cooling_schedule not in ['linear', 'exponential', 'logarithmic']:
            raise ValueError("Cooling schedule must be 'linear', 'exponential', or 'logarithmic'")
        if not 0.0 < cooling_rate < 1.0:
            raise ValueError("Cooling rate must be between 0.0 and 1.0")
        if not 0.0 < neighborhood_size <= 1.0:
            raise ValueError("Neighborhood size must be between 0.0 and 1.0")
        
        self.current_temperature = initial_temperature
        self.accepted_moves = 0
        self.rejected_moves = 0
        
        logger.info(f"Initialized Simulated Annealing with {cooling_schedule} cooling")
        logger.info(f"Temperature: {initial_temperature} -> {final_temperature}")
    
    def _calculate_temperature(self, iteration: int, max_iterations: int) -> float:
        """
        Calculate temperature based on cooling schedule
        
        Args:
            iteration: Current iteration
            max_iterations: Maximum iterations
            
        Returns:
            Current temperature
        """
        if self.cooling_schedule == 'linear':
            # Linear cooling
            progress = iteration / max_iterations
            temperature = self.initial_temperature * (1 - progress) + self.final_temperature * progress
            
        elif self.cooling_schedule == 'exponential':
            # Exponential cooling
            temperature = self.initial_temperature * (self.cooling_rate ** iteration)
            temperature = max(temperature, self.final_temperature)
            
        elif self.cooling_schedule == 'logarithmic':
            # Logarithmic cooling
            temperature = self.initial_temperature / (1 + math.log(1 + iteration))
            temperature = max(temperature, self.final_temperature)
        
        return temperature
    
    def _generate_neighbor(self, current_solution: Dict[str, float]) -> Dict[str, float]:
        """
        Generate neighbor solution by perturbing current solution
        
        Args:
            current_solution: Current solution
            
        Returns:
            Neighbor solution
        """
        neighbor = current_solution.copy()
        
        # Perturb each parameter with Gaussian noise
        for param_name, (min_val, max_val) in self.param_space.items():
            param_range = max_val - min_val
            noise_std = self.neighborhood_size * param_range
            
            current_value = current_solution[param_name]
            new_value = current_value + random.gauss(0, noise_std)
            
            # Clip to bounds
            neighbor[param_name] = np.clip(new_value, min_val, max_val)
        
        return neighbor
    
    def _acceptance_probability(self, current_fitness: float, neighbor_fitness: float, 
                              temperature: float) -> float:
        """
        Calculate probability of accepting a move
        
        Args:
            current_fitness: Current solution fitness
            neighbor_fitness: Neighbor solution fitness
            temperature: Current temperature
            
        Returns:
            Acceptance probability
        """
        if temperature <= 0:
            return 0.0
        
        if self.maximize:
            # For maximization, accept if neighbor is better
            if neighbor_fitness >= current_fitness:
                return 1.0
            else:
                # Accept worse solution with probability
                delta = (neighbor_fitness - current_fitness)
                return math.exp(delta / temperature)
        else:
            # For minimization, accept if neighbor is better (lower)
            if neighbor_fitness <= current_fitness:
                return 1.0
            else:
                # Accept worse solution with probability
                delta = (current_fitness - neighbor_fitness)
                return math.exp(delta / temperature)
    
    def optimize(self, 
                 n_iterations: int = 1000,
                 callback: Optional[callable] = None,
                 **kwargs) -> OptimizationResult:
        """
        Run Simulated Annealing optimization
        
        Args:
            n_iterations: Number of iterations
            callback: Optional callback function
            **kwargs: Additional parameters
            
        Returns:
            OptimizationResult with best parameters and metadata
        """
        self.start_time = time.time()
        
        logger.info(f"Starting Simulated Annealing optimization")
        logger.info(f"Iterations: {n_iterations}")
        
        # Initialize with random solution
        current_solution = self._random_params()
        current_fitness = self.objective_function(current_solution)
        
        # Track best solution
        self._update_best(current_solution, current_fitness)
        
        # Track statistics
        temperature_history = []
        acceptance_rate_history = []
        fitness_history = []
        
        # Optimization loop
        for iteration in range(n_iterations):
            # Calculate current temperature
            self.current_temperature = self._calculate_temperature(iteration, n_iterations)
            temperature_history.append(self.current_temperature)
            
            # Generate neighbor solution
            neighbor_solution = self._generate_neighbor(current_solution)
            neighbor_fitness = self.objective_function(neighbor_solution)
            
            # Calculate acceptance probability
            accept_prob = self._acceptance_probability(
                current_fitness, neighbor_fitness, self.current_temperature)
            
            # Accept or reject move
            if random.random() < accept_prob:
                # Accept move
                current_solution = neighbor_solution
                current_fitness = neighbor_fitness
                self.accepted_moves += 1
                
                # Update best if improved
                self._update_best(current_solution, current_fitness)
            else:
                # Reject move
                self.rejected_moves += 1
            
            # Track progress
            fitness_history.append(current_fitness)
            self.iteration_history.append(self.best_score)
            
            # Calculate acceptance rate for this window
            total_moves = self.accepted_moves + self.rejected_moves
            acceptance_rate = self.accepted_moves / total_moves if total_moves > 0 else 0.0
            acceptance_rate_history.append(acceptance_rate)
            
            # Call callback if provided
            if callback:
                callback(iteration + 1, current_solution, current_fitness)
            
            # Log progress
            if iteration % 100 == 0:
                logger.info(f"Iteration {iteration}: T={self.current_temperature:.4f}, "
                          f"Current={current_fitness:.6f}, Best={self.best_score:.6f}, "
                          f"Accept Rate={acceptance_rate:.3f}")
            
            # Check convergence
            if self._check_convergence(self.best_score):
                logger.info(f"Simulated Annealing converged after {iteration + 1} iterations")
                break
            
            # Early termination if temperature is too low
            if self.current_temperature < self.final_temperature:
                logger.info(f"Reached final temperature after {iteration + 1} iterations")
                break
        
        # Create result
        result = self._create_result(iteration + 1)
        result.metadata.update({
            'algorithm_params': {
                'initial_temperature': self.initial_temperature,
                'final_temperature': self.final_temperature,
                'cooling_schedule': self.cooling_schedule,
                'cooling_rate': self.cooling_rate,
                'neighborhood_size': self.neighborhood_size,
                'iterations_completed': iteration + 1
            },
            'annealing_stats': {
                'final_temperature': self.current_temperature,
                'accepted_moves': self.accepted_moves,
                'rejected_moves': self.rejected_moves,
                'final_acceptance_rate': acceptance_rate,
                'temperature_history': temperature_history,
                'acceptance_rate_history': acceptance_rate_history,
                'fitness_history': fitness_history,
                'convergence_iteration': iteration + 1 if self.stagnation_count >= self.max_stagnation else None
            }
        })
        
        logger.info(f"Simulated Annealing completed. Best fitness: {self.best_score:.6f}")
        logger.info(f"Final acceptance rate: {acceptance_rate:.3f}")
        
        return result
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return algorithm metadata for registry"""
        return {
            'name': 'Simulated Annealing',
            'category': 'physics_inspired',
            'description': 'Temperature-based optimization inspired by metallurgical annealing',
            'supports_gpu': False,
            'supports_parallel': False,  # Sequential by nature
            'complexity': 'O(n)',  # n iterations
            'best_for': [
                'escaping local optima',
                'discrete optimization',
                'traveling salesman problems',
                'scheduling problems'
            ],
            'hyperparameters': [
                {
                    'name': 'initial_temperature',
                    'type': 'float',
                    'range': (1.0, 1000.0),
                    'default': 100.0,
                    'description': 'Starting temperature'
                },
                {
                    'name': 'final_temperature',
                    'type': 'float',
                    'range': (0.001, 10.0),
                    'default': 0.01,
                    'description': 'Ending temperature'
                },
                {
                    'name': 'cooling_schedule',
                    'type': 'categorical',
                    'choices': ['linear', 'exponential', 'logarithmic'],
                    'default': 'exponential',
                    'description': 'Temperature cooling schedule'
                },
                {
                    'name': 'neighborhood_size',
                    'type': 'float',
                    'range': (0.01, 1.0),
                    'default': 0.1,
                    'description': 'Size of neighborhood for moves'
                }
            ],
            'advantages': [
                'Can escape local optima',
                'Simple to implement',
                'Guaranteed convergence (theoretical)',
                'Works well for discrete problems'
            ],
            'disadvantages': [
                'Slow convergence',
                'Many parameters to tune',
                'No guarantee of global optimum',
                'Temperature schedule affects performance'
            ],
            'references': [
                'Kirkpatrick, S. et al. (1983). Optimization by simulated annealing',
                'Černý, V. (1985). Thermodynamical approach to the traveling salesman problem',
                'Van Laarhoven, P. J. M. & Aarts, E. H. L. (1987). Simulated Annealing'
            ]
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], **kwargs) -> 'SimulatedAnnealingOptimizer':
        """Create optimizer from configuration dictionary"""
        sa_config = config.get('simulated_annealing', {})
        
        return cls(
            initial_temperature=sa_config.get('initial_temperature', 100.0),
            final_temperature=sa_config.get('final_temperature', 0.01),
            cooling_schedule=sa_config.get('cooling_schedule', 'exponential'),
            cooling_rate=sa_config.get('cooling_rate', 0.95),
            neighborhood_size=sa_config.get('neighborhood_size', 0.1),
            **kwargs
        )