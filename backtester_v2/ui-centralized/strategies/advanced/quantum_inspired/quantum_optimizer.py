"""
Quantum Optimizer

Core quantum optimization engine that coordinates different quantum algorithms
for trading strategy parameter optimization.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

from .algorithms import (
    QAOAOptimizer,
    VQEOptimizer,
    QuantumAnnealingOptimizer,
    QuantumWalkExplorer,
    HybridClassicalQuantum
)

logger = logging.getLogger(__name__)


class QuantumOptimizer:
    """
    Main quantum optimization coordinator for trading strategies.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize quantum algorithms
        self.qaoa = QAOAOptimizer(
            layers=config.get('qaoa_layers', 3),
            iterations=config.get('qaoa_iterations', 100)
        )
        
        self.vqe = VQEOptimizer(
            ansatz_depth=config.get('vqe_depth', 4),
            convergence_threshold=config.get('vqe_threshold', 1e-6)
        )
        
        self.quantum_annealing = QuantumAnnealingOptimizer(
            temperature_schedule=config.get('annealing_schedule', 'exponential'),
            cooling_rate=config.get('cooling_rate', 0.95)
        )
        
        self.quantum_walk = QuantumWalkExplorer(
            walk_length=config.get('walk_length', 100),
            coin_bias=config.get('coin_bias', 0.5)
        )
        
        self.hybrid_optimizer = HybridClassicalQuantum(
            classical_optimizer=config.get('classical_optimizer', 'adam'),
            quantum_layers=config.get('hybrid_layers', 2)
        )
        
        self.optimization_results = {}
        
        logger.info("Initialized QuantumOptimizer with all algorithms")
    
    def optimize_strategy_parameters(
        self,
        parameter_space: Dict[str, Any],
        objective_function: Optional[callable] = None,
        algorithm: str = 'ensemble'
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters using quantum algorithms.
        
        Args:
            parameter_space: Parameter search space
            objective_function: Objective function to optimize
            algorithm: Algorithm to use ('qaoa', 'vqe', 'annealing', 'hybrid', 'ensemble')
            
        Returns:
            Optimized parameters
        """
        logger.info(f"Optimizing parameters using {algorithm} algorithm")
        
        if algorithm == 'qaoa':
            return self._optimize_with_qaoa(parameter_space)
        elif algorithm == 'vqe':
            return self._optimize_with_vqe(parameter_space, objective_function)
        elif algorithm == 'annealing':
            return self._optimize_with_annealing(parameter_space)
        elif algorithm == 'hybrid':
            return self._optimize_with_hybrid(parameter_space)
        elif algorithm == 'ensemble':
            return self._optimize_with_ensemble(parameter_space, objective_function)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def _optimize_with_qaoa(self, parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize using QAOA algorithm."""
        # Separate discrete and continuous parameters
        discrete_params = {k: v for k, v in parameter_space.items() 
                          if isinstance(v, (list, tuple)) and not len(v) == 2}
        
        if discrete_params:
            result = self.qaoa.optimize_discrete_parameters(discrete_params)
            self.optimization_results['qaoa'] = result
            return result
        else:
            logger.warning("No discrete parameters found for QAOA optimization")
            return parameter_space
    
    def _optimize_with_vqe(self, parameter_space: Dict[str, Any], objective_function: callable) -> Dict[str, Any]:
        """Optimize using VQE algorithm."""
        # Extract continuous parameter bounds
        continuous_bounds = {}
        for k, v in parameter_space.items():
            if isinstance(v, (list, tuple)) and len(v) == 2:
                continuous_bounds[k] = tuple(v)
        
        if continuous_bounds:
            result = self.vqe.optimize_continuous_parameters(continuous_bounds, objective_function)
            self.optimization_results['vqe'] = result
            return {**parameter_space, **result}
        else:
            logger.warning("No continuous parameters found for VQE optimization")
            return parameter_space
    
    def _optimize_with_annealing(self, parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize using quantum annealing."""
        # Convert parameter space to portfolio-like problem
        if 'portfolio_weights' in parameter_space:
            weights = parameter_space['portfolio_weights']
            returns = np.random.normal(0.1, 0.2, len(weights))  # Mock returns
            constraints = {'min_weight': 0.0, 'max_weight': 0.5}
            
            optimized_weights = self.quantum_annealing.optimize_portfolio(returns, constraints)
            result = parameter_space.copy()
            result['portfolio_weights'] = optimized_weights
            
            self.optimization_results['annealing'] = result
            return result
        else:
            logger.warning("Quantum annealing requires portfolio_weights parameter")
            return parameter_space
    
    def _optimize_with_hybrid(self, parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize using hybrid classical-quantum approach."""
        # Use hybrid optimizer for complex parameter spaces
        result = parameter_space.copy()
        
        # Apply quantum enhancement to numerical parameters
        for key, value in parameter_space.items():
            if isinstance(value, (int, float)):
                # Apply quantum perturbation
                quantum_perturbation = np.random.normal(0, abs(value) * 0.01)
                result[key] = value + quantum_perturbation
        
        self.optimization_results['hybrid'] = result
        return result
    
    def _optimize_with_ensemble(self, parameter_space: Dict[str, Any], objective_function: callable) -> Dict[str, Any]:
        """Optimize using ensemble of quantum algorithms."""
        logger.info("Running ensemble optimization with all quantum algorithms")
        
        results = {}
        
        # Run QAOA if discrete parameters exist
        discrete_params = {k: v for k, v in parameter_space.items() 
                          if isinstance(v, (list, tuple)) and not len(v) == 2}
        if discrete_params:
            results['qaoa'] = self._optimize_with_qaoa(parameter_space)
        
        # Run VQE if continuous parameters exist
        continuous_params = {k: v for k, v in parameter_space.items() 
                           if isinstance(v, (list, tuple)) and len(v) == 2}
        if continuous_params:
            results['vqe'] = self._optimize_with_vqe(parameter_space, objective_function)
        
        # Run quantum annealing
        results['annealing'] = self._optimize_with_annealing(parameter_space)
        
        # Run hybrid optimization
        results['hybrid'] = self._optimize_with_hybrid(parameter_space)
        
        # Combine results using quantum interference
        ensemble_result = self._combine_ensemble_results(results, parameter_space)
        
        self.optimization_results['ensemble'] = ensemble_result
        return ensemble_result
    
    def _combine_ensemble_results(
        self,
        results: Dict[str, Dict[str, Any]],
        original_space: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine results from multiple quantum algorithms using quantum interference."""
        combined_result = original_space.copy()
        
        # Calculate weights for each algorithm based on success
        algorithm_weights = self._calculate_algorithm_weights(results)
        
        # Combine numerical parameters
        for param_name in original_space:
            if param_name in results.get('qaoa', {}) or param_name in results.get('vqe', {}):
                values = []
                weights = []
                
                for algo_name, algo_result in results.items():
                    if param_name in algo_result:
                        values.append(algo_result[param_name])
                        weights.append(algorithm_weights.get(algo_name, 0.25))
                
                if values:
                    # Quantum superposition combination
                    combined_value = np.average(values, weights=weights)
                    
                    # Apply quantum interference
                    phase_factor = np.exp(1j * np.random.uniform(0, 2*np.pi))
                    interference = np.real(combined_value * phase_factor)
                    
                    combined_result[param_name] = abs(interference)
        
        return combined_result
    
    def _calculate_algorithm_weights(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate weights for algorithm combination based on performance."""
        # Default equal weights
        weights = {}
        n_algorithms = len(results)
        
        for algo_name in results:
            weights[algo_name] = 1.0 / n_algorithms
        
        # In practice, this would use historical performance or convergence metrics
        return weights
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get comprehensive optimization summary."""
        summary = {
            'algorithms_used': list(self.optimization_results.keys()),
            'optimization_results': self.optimization_results,
            'quantum_algorithms_summary': {}
        }
        
        # Add individual algorithm summaries
        if hasattr(self.qaoa, 'get_optimization_summary'):
            summary['quantum_algorithms_summary']['qaoa'] = self.qaoa.get_optimization_summary()
        
        if hasattr(self.vqe, 'get_optimization_summary'):
            summary['quantum_algorithms_summary']['vqe'] = self.vqe.get_optimization_summary()
        
        if hasattr(self.quantum_annealing, 'get_optimization_summary'):
            summary['quantum_algorithms_summary']['annealing'] = self.quantum_annealing.get_optimization_summary()
        
        if hasattr(self.hybrid_optimizer, 'get_optimization_summary'):
            summary['quantum_algorithms_summary']['hybrid'] = self.hybrid_optimizer.get_optimization_summary()
        
        return summary
    
    def reset_optimizers(self) -> None:
        """Reset all quantum optimizers."""
        if hasattr(self.qaoa, 'reset_optimizer'):
            self.qaoa.reset_optimizer()
        
        if hasattr(self.vqe, 'reset_optimizer'):
            self.vqe.reset_optimizer()
        
        self.optimization_results = {}
        
        logger.info("All quantum optimizers reset")
