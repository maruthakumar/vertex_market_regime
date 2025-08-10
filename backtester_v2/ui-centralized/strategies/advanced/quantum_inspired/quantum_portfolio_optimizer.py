"""
Quantum Portfolio Optimizer

Specialized quantum optimization for portfolio construction and risk management
using quantum-inspired algorithms for superior diversification and risk-return optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging

from .algorithms import (
    QAOAOptimizer,
    VQEOptimizer,
    QuantumAnnealingOptimizer,
    HybridClassicalQuantum
)
from .quantum_models import QuantumPortfolio, QuantumParameters

logger = logging.getLogger(__name__)


class QuantumPortfolioOptimizer:
    """
    Quantum-enhanced portfolio optimization system.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize quantum algorithms for portfolio optimization
        self.qaoa = QAOAOptimizer(
            layers=config.get('qaoa_layers', 3),
            iterations=config.get('qaoa_iterations', 200)
        )
        
        self.vqe = VQEOptimizer(
            ansatz_depth=config.get('vqe_depth', 6),
            convergence_threshold=config.get('vqe_threshold', 1e-8)
        )
        
        self.quantum_annealing = QuantumAnnealingOptimizer(
            temperature_schedule='exponential',
            cooling_rate=config.get('cooling_rate', 0.98),
            max_iterations=config.get('annealing_iterations', 5000)
        )
        
        self.hybrid_optimizer = HybridClassicalQuantum(
            quantum_weight=config.get('quantum_weight', 0.4)
        )
        
        # Portfolio state
        self.current_portfolio = None
        self.optimization_history = []
        self.quantum_metrics = {}
        
        logger.info("Initialized QuantumPortfolioOptimizer")
    
    def optimize_portfolio(
        self,
        assets: List[str],
        returns: pd.DataFrame,
        risk_model: Optional[np.ndarray] = None,
        constraints: Optional[Dict[str, Any]] = None,
        objective: str = 'sharpe_ratio'
    ) -> QuantumPortfolio:
        """
        Optimize portfolio using quantum algorithms.
        
        Args:
            assets: List of asset symbols
            returns: Historical returns DataFrame
            risk_model: Risk model matrix (optional)
            constraints: Portfolio constraints
            objective: Optimization objective
            
        Returns:
            Optimized quantum portfolio
        """
        logger.info(f"Optimizing portfolio with {len(assets)} assets using quantum methods")
        
        # Prepare data
        returns_matrix = returns.values
        correlation_matrix = returns.corr().values
        
        if risk_model is None:
            risk_model = correlation_matrix
        
        # Set default constraints
        if constraints is None:
            constraints = self._get_default_constraints(len(assets))
        
        # Create quantum entanglement matrix from correlations
        entanglement_matrix = self._create_entanglement_matrix(correlation_matrix)
        
        # Optimize using different quantum algorithms
        optimization_results = {}
        
        # QAOA optimization for discrete allocation
        if self.config.get('use_qaoa', True):
            qaoa_weights = self._optimize_with_qaoa(
                returns_matrix, correlation_matrix, constraints
            )
            optimization_results['qaoa'] = qaoa_weights
        
        # VQE optimization for continuous allocation
        if self.config.get('use_vqe', True):
            vqe_weights = self._optimize_with_vqe(
                returns_matrix, entanglement_matrix, constraints
            )
            optimization_results['vqe'] = vqe_weights
        
        # Quantum annealing for global optimization
        if self.config.get('use_annealing', True):
            annealing_weights = self._optimize_with_annealing(
                returns_matrix, constraints
            )
            optimization_results['annealing'] = annealing_weights
        
        # Hybrid classical-quantum optimization
        if self.config.get('use_hybrid', True):
            hybrid_weights = self._optimize_with_hybrid(
                returns_matrix, correlation_matrix
            )
            optimization_results['hybrid'] = hybrid_weights
        
        # Combine results using quantum ensemble
        optimal_weights = self._quantum_ensemble_optimization(
            optimization_results, correlation_matrix
        )
        
        # Create quantum portfolio
        quantum_portfolio = self._create_quantum_portfolio(
            assets, optimal_weights, entanglement_matrix, returns
        )
        
        # Update state
        self.current_portfolio = quantum_portfolio
        self.optimization_history.append({
            'timestamp': pd.Timestamp.now(),
            'weights': optimal_weights,
            'objective_value': self._calculate_objective_value(optimal_weights, returns_matrix),
            'quantum_metrics': self.quantum_metrics.copy()
        })
        
        return quantum_portfolio
    
    def _optimize_with_qaoa(
        self,
        returns: np.ndarray,
        correlation_matrix: np.ndarray,
        constraints: Dict[str, Any]
    ) -> Dict[str, float]:
        """Optimize portfolio using QAOA algorithm."""
        logger.debug("Running QAOA portfolio optimization")
        
        expected_returns = np.mean(returns, axis=0)
        risk_aversion = self.config.get('risk_aversion', 1.0)
        
        # QAOA optimization
        weights_array = self.qaoa.optimize_portfolio(
            expected_returns,
            correlation_matrix,
            risk_aversion,
            constraints
        )
        
        return {f'asset_{i}': weight for i, weight in enumerate(weights_array)}
    
    def _optimize_with_vqe(
        self,
        returns: np.ndarray,
        entanglement_matrix: np.ndarray,
        constraints: Dict[str, Any]
    ) -> Dict[str, float]:
        """Optimize portfolio using VQE algorithm."""
        logger.debug("Running VQE portfolio optimization")
        
        expected_returns = np.mean(returns, axis=0)
        risk_tolerance = self.config.get('risk_tolerance', 0.1)
        
        # VQE optimization
        weights_array = self.vqe.optimize_portfolio(
            expected_returns,
            entanglement_matrix,
            risk_tolerance
        )
        
        return {f'asset_{i}': weight for i, weight in enumerate(weights_array)}
    
    def _optimize_with_annealing(
        self,
        returns: np.ndarray,
        constraints: Dict[str, Any]
    ) -> Dict[str, float]:
        """Optimize portfolio using quantum annealing."""
        logger.debug("Running quantum annealing portfolio optimization")
        
        expected_returns = np.mean(returns, axis=0)
        
        # Quantum annealing optimization
        weights_array = self.quantum_annealing.optimize_portfolio(
            expected_returns,
            constraints
        )
        
        return {f'asset_{i}': weight for i, weight in enumerate(weights_array)}
    
    def _optimize_with_hybrid(
        self,
        returns: np.ndarray,
        correlation_matrix: np.ndarray
    ) -> Dict[str, float]:
        """Optimize portfolio using hybrid approach."""
        logger.debug("Running hybrid classical-quantum portfolio optimization")
        
        expected_returns = np.mean(returns, axis=0)
        
        # Hybrid optimization
        weights_array = self.hybrid_optimizer.optimize_portfolio(
            expected_returns,
            correlation_matrix
        )
        
        return {f'asset_{i}': weight for i, weight in enumerate(weights_array)}
    
    def _quantum_ensemble_optimization(
        self,
        optimization_results: Dict[str, Dict[str, float]],
        correlation_matrix: np.ndarray
    ) -> Dict[str, float]:
        """Combine optimization results using quantum ensemble methods."""
        logger.debug("Combining optimization results using quantum ensemble")
        
        if not optimization_results:
            raise ValueError("No optimization results to combine")
        
        # Get number of assets
        first_result = list(optimization_results.values())[0]
        n_assets = len(first_result)
        
        # Calculate quantum superposition weights
        ensemble_weights = np.zeros(n_assets)
        total_amplitude = 0.0
        
        for algo_name, weights_dict in optimization_results.items():
            # Convert to array
            weights_array = np.array(list(weights_dict.values()))
            
            # Calculate algorithm amplitude based on performance
            performance_score = self._calculate_algorithm_performance(
                algo_name, weights_array, correlation_matrix
            )
            amplitude = np.sqrt(performance_score)
            
            # Apply quantum phase
            phase = self._get_algorithm_phase(algo_name)
            quantum_contribution = amplitude * np.exp(1j * phase) * weights_array
            
            # Add real part (quantum interference)
            ensemble_weights += quantum_contribution.real
            total_amplitude += amplitude**2
        
        # Normalize
        if total_amplitude > 0:
            ensemble_weights = np.abs(ensemble_weights)
            ensemble_weights /= np.sum(ensemble_weights)
        else:
            ensemble_weights = np.ones(n_assets) / n_assets
        
        # Update quantum metrics
        self.quantum_metrics.update({
            'ensemble_coherence': total_amplitude,
            'algorithm_count': len(optimization_results),
            'quantum_advantage': self._calculate_quantum_advantage(ensemble_weights, correlation_matrix)
        })
        
        return {f'asset_{i}': weight for i, weight in enumerate(ensemble_weights)}
    
    def _create_entanglement_matrix(self, correlation_matrix: np.ndarray) -> np.ndarray:
        """Create quantum entanglement matrix from asset correlations."""
        entanglement_strength = self.config.get('entanglement_strength', 0.8)
        
        # Convert correlations to entanglement strengths
        entanglement_matrix = np.abs(correlation_matrix) * entanglement_strength
        
        # Zero diagonal (no self-entanglement)
        np.fill_diagonal(entanglement_matrix, 0)
        
        return entanglement_matrix
    
    def _create_quantum_portfolio(
        self,
        assets: List[str],
        weights: Dict[str, float],
        entanglement_matrix: np.ndarray,
        returns: pd.DataFrame
    ) -> QuantumPortfolio:
        """Create quantum portfolio object."""
        # Calculate quantum risk
        weights_array = np.array(list(weights.values()))
        quantum_risk = self._calculate_quantum_risk(weights_array, entanglement_matrix)
        
        # Calculate coherence measure
        coherence_measure = self._calculate_coherence_measure(entanglement_matrix)
        
        # Create quantum state representation
        quantum_state = {
            'amplitudes': weights_array / np.linalg.norm(weights_array),
            'phases': np.random.uniform(0, 2*np.pi, len(weights_array)),
            'entanglement_matrix': entanglement_matrix,
            'coherence_time': self.config.get('coherence_time', 100)
        }
        
        return QuantumPortfolio(
            assets=assets,
            weights=weights,
            entanglement_matrix=entanglement_matrix,
            quantum_risk=quantum_risk,
            coherence_measure=coherence_measure,
            total_value=self.config.get('portfolio_value', 1000000),
            quantum_state=quantum_state
        )
    
    def _calculate_algorithm_performance(
        self,
        algo_name: str,
        weights: np.ndarray,
        correlation_matrix: np.ndarray
    ) -> float:
        """Calculate performance score for algorithm."""
        # Simple performance metric based on diversification and expected return
        diversification_score = 1.0 / np.sum(weights**2)  # Inverse concentration
        correlation_penalty = np.sum(weights.T @ correlation_matrix @ weights)
        
        performance = diversification_score / (1 + correlation_penalty)
        
        # Algorithm-specific adjustments
        algo_factors = {
            'qaoa': 1.1,  # Bonus for discrete optimization
            'vqe': 1.2,   # Bonus for continuous optimization
            'annealing': 0.9,  # Slight penalty for stochastic nature
            'hybrid': 1.0  # Neutral
        }
        
        return performance * algo_factors.get(algo_name, 1.0)
    
    def _get_algorithm_phase(self, algo_name: str) -> float:
        """Get quantum phase for algorithm."""
        phases = {
            'qaoa': 0.0,
            'vqe': np.pi / 4,
            'annealing': np.pi / 2,
            'hybrid': 3 * np.pi / 4
        }
        return phases.get(algo_name, 0.0)
    
    def _calculate_quantum_risk(
        self,
        weights: np.ndarray,
        entanglement_matrix: np.ndarray
    ) -> float:
        """Calculate quantum-enhanced risk measure."""
        # Classical risk
        classical_risk = np.sum(weights**2)
        
        # Quantum entanglement risk
        entanglement_risk = weights.T @ entanglement_matrix @ weights
        
        # Combined quantum risk
        quantum_risk = classical_risk + 0.5 * entanglement_risk
        
        return quantum_risk
    
    def _calculate_coherence_measure(self, entanglement_matrix: np.ndarray) -> float:
        """Calculate quantum coherence measure."""
        # Use entanglement entropy as coherence measure
        eigenvalues = np.linalg.eigvals(entanglement_matrix)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove near-zero eigenvalues
        
        if len(eigenvalues) == 0:
            return 0.0
        
        # Normalize eigenvalues
        eigenvalues = eigenvalues / np.sum(eigenvalues)
        
        # Calculate entropy
        coherence = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-10))
        
        return coherence
    
    def _calculate_quantum_advantage(
        self,
        quantum_weights: np.ndarray,
        correlation_matrix: np.ndarray
    ) -> float:
        """Calculate quantum advantage over classical optimization."""
        # Classical equal-weight portfolio
        n_assets = len(quantum_weights)
        classical_weights = np.ones(n_assets) / n_assets
        
        # Calculate Sharpe-like ratios
        quantum_risk = quantum_weights.T @ correlation_matrix @ quantum_weights
        classical_risk = classical_weights.T @ correlation_matrix @ classical_weights
        
        # Quantum advantage = classical_risk / quantum_risk
        if quantum_risk > 0:
            return classical_risk / quantum_risk
        else:
            return 1.0
    
    def _calculate_objective_value(
        self,
        weights: Dict[str, float],
        returns: np.ndarray
    ) -> float:
        """Calculate objective value for optimization."""
        weights_array = np.array(list(weights.values()))
        expected_returns = np.mean(returns, axis=0)
        
        return np.dot(weights_array, expected_returns)
    
    def _get_default_constraints(self, n_assets: int) -> Dict[str, Any]:
        """Get default portfolio constraints."""
        return {
            'max_weight': self.config.get('max_weight', 0.3),
            'min_weight': self.config.get('min_weight', 0.0),
            'sum_constraint': 1.0,
            'max_positions': min(n_assets, self.config.get('max_positions', 10))
        }
    
    def rebalance_portfolio(
        self,
        new_returns: pd.DataFrame,
        transaction_costs: float = 0.001
    ) -> QuantumPortfolio:
        """Rebalance existing portfolio using quantum optimization."""
        if self.current_portfolio is None:
            raise ValueError("No current portfolio to rebalance")
        
        logger.info("Rebalancing portfolio using quantum optimization")
        
        # Get current assets
        assets = self.current_portfolio.assets
        
        # Optimize new weights
        new_portfolio = self.optimize_portfolio(
            assets,
            new_returns,
            constraints=self._get_rebalancing_constraints(transaction_costs)
        )
        
        return new_portfolio
    
    def _get_rebalancing_constraints(self, transaction_costs: float) -> Dict[str, Any]:
        """Get constraints for portfolio rebalancing."""
        base_constraints = self._get_default_constraints(len(self.current_portfolio.assets))
        
        # Add transaction cost considerations
        base_constraints['transaction_costs'] = transaction_costs
        base_constraints['turnover_limit'] = self.config.get('turnover_limit', 0.5)
        
        return base_constraints
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary."""
        if self.current_portfolio is None:
            return {'error': 'No portfolio optimized yet'}
        
        portfolio_dict = self.current_portfolio.to_dict()
        
        summary = {
            'portfolio': portfolio_dict,
            'quantum_metrics': self.quantum_metrics,
            'optimization_history': self.optimization_history,
            'quantum_algorithms_summary': {
                'qaoa': self.qaoa.get_optimization_summary(),
                'vqe': self.vqe.get_optimization_summary(),
                'annealing': self.quantum_annealing.get_optimization_summary(),
                'hybrid': self.hybrid_optimizer.get_optimization_summary()
            }
        }
        
        return summary
