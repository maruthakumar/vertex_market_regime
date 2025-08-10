"""
Particle Swarm Optimization Algorithm

Swarm intelligence optimization inspired by bird flocking behavior.
"""

import numpy as np
import random
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
from ...base.base_optimizer import BaseOptimizer, OptimizationResult

logger = logging.getLogger(__name__)

class Particle:
    """Represents a particle in the swarm"""
    
    def __init__(self, param_space: Dict[str, Tuple[float, float]]):
        self.param_space = param_space
        self.param_names = list(param_space.keys())
        
        # Initialize position randomly
        self.position = {}
        for param_name, (min_val, max_val) in param_space.items():
            self.position[param_name] = random.uniform(min_val, max_val)
        
        # Initialize velocity
        self.velocity = {}
        for param_name, (min_val, max_val) in param_space.items():
            max_velocity = 0.1 * (max_val - min_val)  # 10% of parameter range
            self.velocity[param_name] = random.uniform(-max_velocity, max_velocity)
        
        # Personal best
        self.best_position = self.position.copy()
        self.best_fitness = None
        self.fitness = None
    
    def update_personal_best(self, fitness: float, maximize: bool):
        """Update personal best if current position is better"""
        if (self.best_fitness is None or 
            (maximize and fitness > self.best_fitness) or
            (not maximize and fitness < self.best_fitness)):
            self.best_position = self.position.copy()
            self.best_fitness = fitness
    
    def update_velocity(self, global_best_position: Dict[str, float], 
                       inertia: float, cognitive: float, social: float):
        """Update particle velocity using PSO equation"""
        for param_name in self.param_names:
            # Cognitive component (personal best)
            r1 = random.random()
            cognitive_component = cognitive * r1 * (self.best_position[param_name] - self.position[param_name])
            
            # Social component (global best)
            r2 = random.random()
            social_component = social * r2 * (global_best_position[param_name] - self.position[param_name])
            
            # Update velocity
            self.velocity[param_name] = (inertia * self.velocity[param_name] + 
                                       cognitive_component + social_component)
            
            # Limit velocity
            min_val, max_val = self.param_space[param_name]
            max_velocity = 0.2 * (max_val - min_val)
            self.velocity[param_name] = np.clip(self.velocity[param_name], 
                                              -max_velocity, max_velocity)
    
    def update_position(self):
        """Update particle position based on velocity"""
        for param_name in self.param_names:
            self.position[param_name] += self.velocity[param_name]
            
            # Ensure position stays within bounds
            min_val, max_val = self.param_space[param_name]
            self.position[param_name] = np.clip(self.position[param_name], min_val, max_val)

class ParticleSwarmOptimizer(BaseOptimizer):
    """
    Particle Swarm Optimization (PSO) algorithm
    
    Swarm intelligence algorithm where particles move through the search space
    influenced by their own best position and the global best position.
    """
    
    def __init__(self, 
                 param_space: Dict[str, Tuple[float, float]],
                 objective_function: callable,
                 maximize: bool = True,
                 swarm_size: int = 30,
                 inertia_weight: float = 0.7,
                 cognitive_weight: float = 1.5,
                 social_weight: float = 1.5,
                 inertia_decay: float = 0.99,
                 **kwargs):
        """
        Initialize Particle Swarm optimizer
        
        Args:
            param_space: Dictionary mapping parameter names to (min, max) bounds
            objective_function: Function to optimize
            maximize: Whether to maximize (True) or minimize (False)
            swarm_size: Number of particles in swarm
            inertia_weight: Inertia weight (w)
            cognitive_weight: Cognitive acceleration coefficient (c1)
            social_weight: Social acceleration coefficient (c2)
            inertia_decay: Decay factor for inertia weight
        """
        super().__init__(param_space, objective_function, maximize, **kwargs)
        
        self.swarm_size = swarm_size
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.inertia_decay = inertia_decay
        
        # Validate parameters
        if swarm_size < 2:
            raise ValueError("Swarm size must be at least 2")
        if inertia_weight < 0:
            raise ValueError("Inertia weight must be non-negative")
        if cognitive_weight < 0:
            raise ValueError("Cognitive weight must be non-negative")
        if social_weight < 0:
            raise ValueError("Social weight must be non-negative")
        
        # Swarm storage
        self.swarm = []
        self.global_best_position = {}
        self.global_best_fitness = float('-inf') if maximize else float('inf')
        self.iteration = 0
        
        logger.info(f"Initialized PSO with swarm_size={swarm_size}")
        logger.info(f"w={inertia_weight}, c1={cognitive_weight}, c2={social_weight}")
    
    def _initialize_swarm(self) -> List[Particle]:
        """Initialize random swarm"""
        swarm = []
        
        for _ in range(self.swarm_size):
            particle = Particle(self.param_space)
            
            # Evaluate initial position
            particle.fitness = self.objective_function(particle.position)
            particle.update_personal_best(particle.fitness, self.maximize)
            
            # Update global best
            self._update_global_best(particle)
            self._update_best(particle.position, particle.fitness)
            
            swarm.append(particle)
        
        return swarm
    
    def _update_global_best(self, particle: Particle):
        """Update global best position"""
        if (particle.fitness is not None and
            ((self.maximize and particle.fitness > self.global_best_fitness) or
             (not self.maximize and particle.fitness < self.global_best_fitness))):
            self.global_best_position = particle.position.copy()
            self.global_best_fitness = particle.fitness
    
    def _update_swarm(self, swarm: List[Particle], current_inertia: float):
        """Update velocities and positions of all particles"""
        for particle in swarm:
            # Update velocity
            particle.update_velocity(
                self.global_best_position,
                current_inertia,
                self.cognitive_weight,
                self.social_weight
            )
            
            # Update position
            particle.update_position()
            
            # Evaluate new position
            particle.fitness = self.objective_function(particle.position)
            
            # Update personal best
            particle.update_personal_best(particle.fitness, self.maximize)
            
            # Update global best
            self._update_global_best(particle)
            self._update_best(particle.position, particle.fitness)
    
    def optimize(self, 
                 n_iterations: int = 200,
                 callback: Optional[callable] = None,
                 **kwargs) -> OptimizationResult:
        """
        Run Particle Swarm Optimization
        
        Args:
            n_iterations: Number of iterations
            callback: Optional callback function
            **kwargs: Additional parameters
            
        Returns:
            OptimizationResult with best parameters and metadata
        """
        self.start_time = time.time()
        
        logger.info(f"Starting Particle Swarm Optimization")
        logger.info(f"Iterations: {n_iterations}, Swarm size: {self.swarm_size}")
        
        # Initialize swarm
        self.swarm = self._initialize_swarm()
        
        # Track statistics
        best_fitness_history = []
        avg_fitness_history = []
        diversity_history = []
        inertia_history = []
        
        current_inertia = self.inertia_weight
        
        # Optimization loop
        for iteration in range(n_iterations):
            self.iteration = iteration
            
            # Get swarm statistics
            fitnesses = [p.fitness for p in self.swarm]
            best_fitness = max(fitnesses) if self.maximize else min(fitnesses)
            avg_fitness = np.mean(fitnesses)
            diversity = self._calculate_diversity()
            
            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)
            diversity_history.append(diversity)
            inertia_history.append(current_inertia)
            
            self.iteration_history.append(best_fitness)
            
            # Call callback if provided
            if callback:
                callback(iteration + 1, self.global_best_position, self.global_best_fitness)
            
            # Log progress
            if iteration % 20 == 0:
                logger.info(f"Iteration {iteration}: Best={best_fitness:.6f}, "
                          f"Avg={avg_fitness:.6f}, Diversity={diversity:.4f}")
            
            # Check convergence
            if self._check_convergence(best_fitness):
                logger.info(f"PSO converged after {iteration + 1} iterations")
                break
            
            # Update swarm
            if iteration < n_iterations - 1:
                self._update_swarm(self.swarm, current_inertia)
                
                # Decay inertia weight
                current_inertia *= self.inertia_decay
        
        # Create result
        result = self._create_result(iteration + 1)
        result.metadata.update({
            'algorithm_params': {
                'swarm_size': self.swarm_size,
                'inertia_weight': self.inertia_weight,
                'cognitive_weight': self.cognitive_weight,
                'social_weight': self.social_weight,
                'inertia_decay': self.inertia_decay,
                'iterations_completed': iteration + 1
            },
            'swarm_stats': {
                'best_fitness_history': best_fitness_history,
                'avg_fitness_history': avg_fitness_history,
                'diversity_history': diversity_history,
                'inertia_history': inertia_history,
                'final_diversity': diversity,
                'convergence_iteration': iteration + 1 if self.stagnation_count >= self.max_stagnation else None
            }
        })
        
        logger.info(f"PSO completed. Best fitness: {self.best_score:.6f}")
        return result
    
    def _calculate_diversity(self) -> float:
        """Calculate swarm diversity as average distance from centroid"""
        if len(self.swarm) < 2:
            return 0.0
        
        # Calculate centroid
        centroid = {}
        for param_name in self.param_space.keys():
            centroid[param_name] = np.mean([p.position[param_name] for p in self.swarm])
        
        # Calculate average distance to centroid
        distances = []
        for particle in self.swarm:
            dist = 0.0
            for param_name in self.param_space.keys():
                min_val, max_val = self.param_space[param_name]
                param_range = max_val - min_val
                
                normalized_dist = abs(particle.position[param_name] - centroid[param_name]) / param_range
                dist += normalized_dist ** 2
            
            distances.append(np.sqrt(dist))
        
        return np.mean(distances)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return algorithm metadata for registry"""
        return {
            'name': 'Particle Swarm Optimization',
            'category': 'swarm',
            'description': 'Swarm intelligence optimization inspired by bird flocking',
            'supports_gpu': True,  # Particle evaluation can be parallelized
            'supports_parallel': True,
            'complexity': 'O(i * s * n)',  # i iterations, s swarm size, n evaluations
            'best_for': [
                'continuous optimization',
                'multimodal problems',
                'real-time adaptation',
                'moderate dimensions'
            ],
            'hyperparameters': [
                {
                    'name': 'swarm_size',
                    'type': 'int',
                    'range': (10, 200),
                    'default': 30,
                    'description': 'Number of particles in swarm'
                },
                {
                    'name': 'inertia_weight',
                    'type': 'float',
                    'range': (0.0, 1.0),
                    'default': 0.7,
                    'description': 'Inertia weight for velocity update'
                },
                {
                    'name': 'cognitive_weight',
                    'type': 'float',
                    'range': (0.0, 4.0),
                    'default': 1.5,
                    'description': 'Cognitive acceleration coefficient'
                },
                {
                    'name': 'social_weight',
                    'type': 'float',
                    'range': (0.0, 4.0),
                    'default': 1.5,
                    'description': 'Social acceleration coefficient'
                }
            ],
            'advantages': [
                'Simple implementation',
                'Fast convergence',
                'Good for continuous optimization',
                'Few parameters to tune'
            ],
            'disadvantages': [
                'Can converge prematurely',
                'Sensitive to parameter settings',
                'Poor performance on discrete problems',
                'May get trapped in local optima'
            ],
            'references': [
                'Kennedy, J. & Eberhart, R. (1995). Particle swarm optimization',
                'Clerc, M. & Kennedy, J. (2002). The particle swarm',
                'Poli, R. et al. (2007). Particle swarm optimization: An overview'
            ]
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], **kwargs) -> 'ParticleSwarmOptimizer':
        """Create optimizer from configuration dictionary"""
        pso_config = config.get('particle_swarm', {})
        
        return cls(
            swarm_size=pso_config.get('swarm_size', 30),
            inertia_weight=pso_config.get('inertia_weight', 0.7),
            cognitive_weight=pso_config.get('cognitive_weight', 1.5),
            social_weight=pso_config.get('social_weight', 1.5),
            inertia_decay=pso_config.get('inertia_decay', 0.99),
            **kwargs
        )