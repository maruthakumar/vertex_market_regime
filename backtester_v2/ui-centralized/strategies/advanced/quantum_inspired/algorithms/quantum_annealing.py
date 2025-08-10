"""
Quantum Annealing Optimizer

Simulated quantum annealing for optimization problems in trading.
Particularly effective for escaping local optima in parameter optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


class QuantumAnnealingOptimizer:
    """
    Quantum annealing optimizer for trading parameter optimization.
    """
    
    def __init__(
        self,
        temperature_schedule: str = 'exponential',
        cooling_rate: float = 0.95,
        initial_temperature: float = 100.0,
        final_temperature: float = 0.01,
        max_iterations: int = 10000
    ):
        self.temperature_schedule = temperature_schedule
        self.cooling_rate = cooling_rate
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.max_iterations = max_iterations
        
        self.optimization_history = []
        self.temperature_history = []
        self.best_solution = None
        self.best_energy = float('inf')
        
        logger.info(f"Initialized Quantum Annealing with {temperature_schedule} schedule")
    
    def optimize_portfolio(
        self,
        returns: np.ndarray,
        constraints: Dict[str, Any]
    ) -> np.ndarray:
        """Optimize portfolio using quantum annealing."""
        n_assets = len(returns)
        
        # Initialize random solution
        current_solution = np.random.random(n_assets)
        current_solution /= np.sum(current_solution)
        
        current_energy = self._calculate_portfolio_energy(current_solution, returns)
        
        temperature = self.initial_temperature
        
        for iteration in range(self.max_iterations):
            # Generate neighbor solution
            neighbor_solution = self._generate_neighbor(current_solution, constraints)
            neighbor_energy = self._calculate_portfolio_energy(neighbor_solution, returns)
            
            # Accept or reject based on Boltzmann probability
            if self._accept_solution(current_energy, neighbor_energy, temperature):
                current_solution = neighbor_solution
                current_energy = neighbor_energy
            
            # Update best solution
            if current_energy < self.best_energy:
                self.best_energy = current_energy
                self.best_solution = current_solution.copy()
            
            # Update temperature
            temperature = self._update_temperature(temperature, iteration)
            
            # Store history
            self.optimization_history.append(current_energy)
            self.temperature_history.append(temperature)
            
            # Check termination
            if temperature < self.final_temperature:
                break
        
        return self.best_solution
    
    def _calculate_portfolio_energy(self, weights: np.ndarray, returns: np.ndarray) -> float:
        """Calculate portfolio energy (negative utility)."""
        expected_return = np.dot(weights, returns)
        concentration_penalty = np.sum(weights**2)  # Diversification penalty
        return -expected_return + 0.5 * concentration_penalty
    
    def _generate_neighbor(self, solution: np.ndarray, constraints: Dict[str, Any]) -> np.ndarray:
        """Generate neighbor solution with small random perturbation."""
        neighbor = solution.copy()
        
        # Random perturbation
        perturbation = np.random.normal(0, 0.01, len(solution))
        neighbor += perturbation
        
        # Apply constraints
        neighbor = np.maximum(neighbor, constraints.get('min_weight', 0.0))
        neighbor = np.minimum(neighbor, constraints.get('max_weight', 1.0))
        
        # Renormalize
        neighbor /= np.sum(neighbor)
        
        return neighbor
    
    def _accept_solution(self, current_energy: float, neighbor_energy: float, temperature: float) -> bool:
        """Accept or reject solution based on quantum annealing criteria."""
        if neighbor_energy < current_energy:
            return True
        
        if temperature <= 0:
            return False
        
        # Quantum tunneling probability
        energy_diff = neighbor_energy - current_energy
        probability = np.exp(-energy_diff / temperature)
        
        return np.random.random() < probability
    
    def _update_temperature(self, current_temp: float, iteration: int) -> float:
        """Update temperature according to cooling schedule."""
        if self.temperature_schedule == 'exponential':
            return current_temp * self.cooling_rate
        elif self.temperature_schedule == 'linear':
            return self.initial_temperature * (1 - iteration / self.max_iterations)
        elif self.temperature_schedule == 'logarithmic':
            return self.initial_temperature / np.log(iteration + 2)
        else:
            return current_temp * self.cooling_rate
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get optimization summary."""
        return {
            'temperature_schedule': self.temperature_schedule,
            'cooling_rate': self.cooling_rate,
            'best_energy': self.best_energy,
            'total_iterations': len(self.optimization_history),
            'final_temperature': self.temperature_history[-1] if self.temperature_history else 0,
            'convergence_achieved': self.temperature_history[-1] < self.final_temperature if self.temperature_history else False
        }
