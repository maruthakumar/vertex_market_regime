"""
Quantum Approximate Optimization Algorithm (QAOA)

Implementation of QAOA for portfolio optimization and discrete parameter optimization
in trading strategies. QAOA is particularly effective for combinatorial optimization
problems in finance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


class QAOAOptimizer:
    """
    Quantum Approximate Optimization Algorithm for trading optimization problems.
    """
    
    def __init__(self, layers: int = 3, iterations: int = 100, seed: Optional[int] = None):
        """
        Initialize QAOA optimizer.
        
        Args:
            layers: Number of QAOA layers (p parameter)
            iterations: Maximum optimization iterations
            seed: Random seed for reproducibility
        """
        self.layers = layers
        self.iterations = iterations
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        # QAOA parameters
        self.gamma_params = np.random.uniform(0, 2*np.pi, layers)  # Problem Hamiltonian angles
        self.beta_params = np.random.uniform(0, np.pi, layers)     # Mixer Hamiltonian angles
        
        # Optimization history
        self.optimization_history = []
        self.best_solution = None
        self.best_energy = float('inf')
        
        logger.info(f"Initialized QAOA with {layers} layers, {iterations} iterations")
    
    def optimize_portfolio(
        self,
        returns: np.ndarray,
        correlation_matrix: np.ndarray,
        risk_aversion: float = 1.0,
        constraints: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Optimize portfolio using QAOA for quadratic optimization.
        
        Args:
            returns: Expected returns array
            correlation_matrix: Asset correlation matrix
            risk_aversion: Risk aversion parameter
            constraints: Portfolio constraints
            
        Returns:
            Optimal portfolio weights
        """
        logger.info(f"Starting QAOA portfolio optimization for {len(returns)} assets")
        
        n_assets = len(returns)
        
        # Convert to QUBO (Quadratic Unconstrained Binary Optimization) formulation
        Q_matrix = self._construct_portfolio_qubo(
            returns, correlation_matrix, risk_aversion, n_assets
        )
        
        # Initialize quantum state (equal superposition)
        quantum_state = np.ones(2**n_assets) / np.sqrt(2**n_assets)
        
        # QAOA optimization
        optimal_params = self._optimize_qaoa_parameters(Q_matrix, quantum_state)
        
        # Extract solution from optimized quantum state
        final_state = self._apply_qaoa_circuit(Q_matrix, quantum_state, optimal_params)
        portfolio_weights = self._extract_portfolio_solution(final_state, n_assets)
        
        # Apply constraints and normalize
        if constraints:
            portfolio_weights = self._apply_constraints(portfolio_weights, constraints)
        
        portfolio_weights = self._normalize_weights(portfolio_weights)
        
        logger.info(f"QAOA optimization completed. Best energy: {self.best_energy:.6f}")
        
        return portfolio_weights
    
    def optimize_discrete_parameters(self, parameter_space: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Optimize discrete parameters using QAOA.
        
        Args:
            parameter_space: Dictionary of parameter names to possible values
            
        Returns:
            Optimal parameter configuration
        """
        logger.info(f"Optimizing {len(parameter_space)} discrete parameters with QAOA")
        
        # Encode discrete parameters as binary variables
        encoded_space, encoding_map = self._encode_discrete_parameters(parameter_space)
        
        # Create objective function for parameter optimization
        def objective_function(binary_config):
            params = self._decode_binary_config(binary_config, encoding_map)
            return self._evaluate_parameter_configuration(params)
        
        # Convert to QUBO formulation
        Q_matrix = self._parameter_space_to_qubo(encoded_space, objective_function)
        
        # Initialize quantum state
        n_qubits = sum(len(values) for values in encoded_space.values())
        quantum_state = np.ones(2**n_qubits) / np.sqrt(2**n_qubits)
        
        # QAOA optimization
        optimal_params = self._optimize_qaoa_parameters(Q_matrix, quantum_state)
        
        # Extract and decode solution
        final_state = self._apply_qaoa_circuit(Q_matrix, quantum_state, optimal_params)
        binary_solution = self._extract_binary_solution(final_state)
        optimal_config = self._decode_binary_config(binary_solution, encoding_map)
        
        return optimal_config
    
    def _construct_portfolio_qubo(
        self,
        returns: np.ndarray,
        correlation_matrix: np.ndarray,
        risk_aversion: float,
        n_assets: int
    ) -> np.ndarray:
        """
        Construct QUBO matrix for portfolio optimization.
        
        Args:
            returns: Expected returns
            correlation_matrix: Asset correlations
            risk_aversion: Risk aversion parameter
            n_assets: Number of assets
            
        Returns:
            QUBO matrix
        """
        # Discretize weights (binary: include asset or not)
        # For more granular weights, we would need more qubits per asset
        
        Q = np.zeros((n_assets, n_assets))
        
        # Diagonal terms (expected returns - individual risk)
        for i in range(n_assets):
            Q[i, i] = -returns[i] + risk_aversion * correlation_matrix[i, i]
        
        # Off-diagonal terms (correlation risk)
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                correlation_penalty = risk_aversion * correlation_matrix[i, j]
                Q[i, j] = correlation_penalty
                Q[j, i] = correlation_penalty
        
        return Q
    
    def _optimize_qaoa_parameters(
        self,
        Q_matrix: np.ndarray,
        initial_state: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimize QAOA variational parameters using classical optimization.
        
        Args:
            Q_matrix: QUBO matrix
            initial_state: Initial quantum state
            
        Returns:
            Optimal gamma and beta parameters
        """
        
        def objective(params):
            n_params = len(params)
            gamma_vals = params[:n_params//2]
            beta_vals = params[n_params//2:]
            
            # Apply QAOA circuit
            state = self._apply_qaoa_circuit(Q_matrix, initial_state, (gamma_vals, beta_vals))
            
            # Calculate expectation value
            energy = self._calculate_expectation_value(Q_matrix, state)
            
            # Store optimization history
            self.optimization_history.append(energy)
            
            if energy < self.best_energy:
                self.best_energy = energy
                self.best_solution = state.copy()
            
            return energy
        
        # Initial parameters
        initial_params = np.concatenate([self.gamma_params, self.beta_params])
        
        # Optimize using classical methods
        result = minimize(
            objective,
            initial_params,
            method='COBYLA',
            options={'maxiter': self.iterations}
        )
        
        optimal_params = result.x
        n_params = len(optimal_params)
        
        return optimal_params[:n_params//2], optimal_params[n_params//2:]
    
    def _apply_qaoa_circuit(
        self,
        Q_matrix: np.ndarray,
        state: np.ndarray,
        params: Tuple[np.ndarray, np.ndarray]
    ) -> np.ndarray:
        """
        Apply QAOA quantum circuit to the state.
        
        Args:
            Q_matrix: Problem QUBO matrix
            state: Current quantum state
            params: Gamma and beta parameters
            
        Returns:
            Updated quantum state
        """
        gamma_vals, beta_vals = params
        current_state = state.copy()
        
        for layer in range(self.layers):
            gamma = gamma_vals[layer]
            beta = beta_vals[layer]
            
            # Apply problem Hamiltonian e^(-i*gamma*H_P)
            current_state = self._apply_problem_hamiltonian(current_state, Q_matrix, gamma)
            
            # Apply mixer Hamiltonian e^(-i*beta*H_M)
            current_state = self._apply_mixer_hamiltonian(current_state, beta)
        
        return current_state
    
    def _apply_problem_hamiltonian(
        self,
        state: np.ndarray,
        Q_matrix: np.ndarray,
        gamma: float
    ) -> np.ndarray:
        """
        Apply the problem Hamiltonian to the quantum state.
        
        Args:
            state: Current quantum state
            Q_matrix: QUBO matrix
            gamma: Problem Hamiltonian angle
            
        Returns:
            Updated state after problem Hamiltonian
        """
        n_qubits = int(np.log2(len(state)))
        updated_state = state.copy()
        
        # Apply diagonal phase shifts based on QUBO terms
        for i in range(len(state)):
            binary_config = self._int_to_binary(i, n_qubits)
            energy = self._evaluate_qubo_energy(binary_config, Q_matrix)
            phase = np.exp(-1j * gamma * energy)
            updated_state[i] *= phase
        
        return updated_state
    
    def _apply_mixer_hamiltonian(self, state: np.ndarray, beta: float) -> np.ndarray:
        """
        Apply the mixer Hamiltonian (X rotations) to the quantum state.
        
        Args:
            state: Current quantum state
            beta: Mixer Hamiltonian angle
            
        Returns:
            Updated state after mixer Hamiltonian
        """
        n_qubits = int(np.log2(len(state)))
        updated_state = np.zeros_like(state)
        
        # Apply X rotation to each qubit
        cos_beta = np.cos(beta)
        sin_beta = np.sin(beta)
        
        for i in range(len(state)):
            binary_config = self._int_to_binary(i, n_qubits)
            
            # For each qubit, calculate the effect of X rotation
            for qubit in range(n_qubits):
                # Flip the qubit
                flipped_config = binary_config.copy()
                flipped_config[qubit] = 1 - flipped_config[qubit]
                flipped_index = self._binary_to_int(flipped_config)
                
                # Apply rotation matrix elements
                updated_state[i] += cos_beta * state[i]
                updated_state[flipped_index] += -1j * sin_beta * state[i]
        
        return updated_state / np.linalg.norm(updated_state)
    
    def _calculate_expectation_value(self, Q_matrix: np.ndarray, state: np.ndarray) -> float:
        """
        Calculate expectation value of the QUBO Hamiltonian.
        
        Args:
            Q_matrix: QUBO matrix
            state: Quantum state
            
        Returns:
            Expectation value
        """
        n_qubits = int(np.log2(len(state)))
        expectation = 0.0
        
        for i in range(len(state)):
            probability = abs(state[i])**2
            binary_config = self._int_to_binary(i, n_qubits)
            energy = self._evaluate_qubo_energy(binary_config, Q_matrix)
            expectation += probability * energy
        
        return expectation
    
    def _evaluate_qubo_energy(self, binary_config: np.ndarray, Q_matrix: np.ndarray) -> float:
        """
        Evaluate QUBO energy for a binary configuration.
        
        Args:
            binary_config: Binary configuration
            Q_matrix: QUBO matrix
            
        Returns:
            Energy value
        """
        return binary_config.T @ Q_matrix @ binary_config
    
    def _extract_portfolio_solution(self, final_state: np.ndarray, n_assets: int) -> np.ndarray:
        """
        Extract portfolio weights from final quantum state.
        
        Args:
            final_state: Final quantum state
            n_assets: Number of assets
            
        Returns:
            Portfolio weights
        """
        # Find the state with highest probability
        probabilities = np.abs(final_state)**2
        most_likely_state = np.argmax(probabilities)
        
        # Convert to binary configuration
        binary_solution = self._int_to_binary(most_likely_state, n_assets)
        
        # Convert binary to weights (equal weight for selected assets)
        weights = binary_solution.astype(float)
        
        return weights
    
    def _extract_binary_solution(self, final_state: np.ndarray) -> np.ndarray:
        """
        Extract binary solution from quantum state.
        
        Args:
            final_state: Final quantum state
            
        Returns:
            Binary solution array
        """
        probabilities = np.abs(final_state)**2
        most_likely_state = np.argmax(probabilities)
        n_qubits = int(np.log2(len(final_state)))
        
        return self._int_to_binary(most_likely_state, n_qubits)
    
    def _encode_discrete_parameters(
        self,
        parameter_space: Dict[str, List[Any]]
    ) -> Tuple[Dict[str, List[int]], Dict]:
        """
        Encode discrete parameters as binary variables.
        
        Args:
            parameter_space: Original parameter space
            
        Returns:
            Encoded binary space and encoding mapping
        """
        encoded_space = {}
        encoding_map = {}
        
        for param_name, values in parameter_space.items():
            n_bits = int(np.ceil(np.log2(len(values))))
            encoded_space[param_name] = list(range(2**n_bits))
            encoding_map[param_name] = {i: values[i] if i < len(values) else values[-1] 
                                      for i in range(2**n_bits)}
        
        return encoded_space, encoding_map
    
    def _decode_binary_config(
        self,
        binary_config: np.ndarray,
        encoding_map: Dict
    ) -> Dict[str, Any]:
        """
        Decode binary configuration to parameter values.
        
        Args:
            binary_config: Binary configuration
            encoding_map: Encoding mapping
            
        Returns:
            Decoded parameter configuration
        """
        decoded_config = {}
        bit_index = 0
        
        for param_name, value_map in encoding_map.items():
            n_bits = int(np.ceil(np.log2(len(value_map))))
            param_bits = binary_config[bit_index:bit_index+n_bits]
            param_index = self._binary_to_int(param_bits)
            decoded_config[param_name] = value_map[param_index]
            bit_index += n_bits
        
        return decoded_config
    
    def _parameter_space_to_qubo(
        self,
        encoded_space: Dict[str, List[int]],
        objective_function: Callable
    ) -> np.ndarray:
        """
        Convert parameter optimization to QUBO formulation.
        
        Args:
            encoded_space: Encoded parameter space
            objective_function: Objective function to optimize
            
        Returns:
            QUBO matrix
        """
        # For simplicity, create a random QUBO matrix
        # In practice, this would be constructed based on the objective function
        total_qubits = sum(int(np.ceil(np.log2(len(values)))) for values in encoded_space.values())
        Q_matrix = np.random.normal(0, 0.1, (total_qubits, total_qubits))
        Q_matrix = (Q_matrix + Q_matrix.T) / 2  # Make symmetric
        
        return Q_matrix
    
    def _evaluate_parameter_configuration(self, params: Dict[str, Any]) -> float:
        """
        Evaluate a parameter configuration (placeholder).
        
        Args:
            params: Parameter configuration
            
        Returns:
            Objective value
        """
        # Placeholder evaluation - in practice, this would run backtests
        return np.random.random()
    
    def _apply_constraints(
        self,
        weights: np.ndarray,
        constraints: Dict[str, Any]
    ) -> np.ndarray:
        """
        Apply portfolio constraints to weights.
        
        Args:
            weights: Raw portfolio weights
            constraints: Constraint specifications
            
        Returns:
            Constrained weights
        """
        constrained_weights = weights.copy()
        
        # Apply maximum weight constraint
        if 'max_weight' in constraints:
            max_weight = constraints['max_weight']
            constrained_weights = np.minimum(constrained_weights, max_weight)
        
        # Apply minimum weight constraint
        if 'min_weight' in constraints:
            min_weight = constraints['min_weight']
            constrained_weights = np.maximum(constrained_weights, min_weight)
        
        return constrained_weights
    
    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """
        Normalize weights to sum to 1.
        
        Args:
            weights: Raw weights
            
        Returns:
            Normalized weights
        """
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            return weights / weight_sum
        else:
            # Equal weights if all weights are zero
            return np.ones(len(weights)) / len(weights)
    
    def _int_to_binary(self, integer: int, n_bits: int) -> np.ndarray:
        """Convert integer to binary array."""
        binary_str = format(integer, f'0{n_bits}b')
        return np.array([int(bit) for bit in binary_str])
    
    def _binary_to_int(self, binary: np.ndarray) -> int:
        """Convert binary array to integer."""
        return int(''.join(map(str, binary.astype(int))), 2)
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get optimization summary and statistics.
        
        Returns:
            Summary dictionary
        """
        return {
            'layers': self.layers,
            'iterations': self.iterations,
            'best_energy': self.best_energy,
            'optimization_history': self.optimization_history,
            'convergence_achieved': len(self.optimization_history) > 10 and 
                                  abs(self.optimization_history[-1] - self.optimization_history[-10]) < 1e-6,
            'final_gamma_params': self.gamma_params.tolist(),
            'final_beta_params': self.beta_params.tolist()
        }
    
    def plot_optimization_history(self) -> None:
        """Plot optimization convergence history."""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.plot(self.optimization_history)
            plt.title('QAOA Optimization Convergence')
            plt.xlabel('Iteration')
            plt.ylabel('Energy')
            plt.grid(True)
            plt.show()
        
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
    
    def reset_optimizer(self) -> None:
        """Reset optimizer state for new optimization."""
        self.gamma_params = np.random.uniform(0, 2*np.pi, self.layers)
        self.beta_params = np.random.uniform(0, np.pi, self.layers)
        self.optimization_history = []
        self.best_solution = None
        self.best_energy = float('inf')
        
        logger.info("QAOA optimizer reset for new optimization")