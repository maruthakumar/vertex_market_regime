"""
Variational Quantum Eigensolver (VQE)

Implementation of VQE for risk optimization and continuous parameter optimization.
VQE is particularly effective for finding optimal solutions to eigenvalue problems
in portfolio optimization and risk management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


class VQEOptimizer:
    """
    Variational Quantum Eigensolver for risk optimization and parameter tuning.
    """
    
    def __init__(
        self,
        ansatz_depth: int = 4,
        convergence_threshold: float = 1e-6,
        max_iterations: int = 1000,
        seed: Optional[int] = None
    ):
        """
        Initialize VQE optimizer.
        
        Args:
            ansatz_depth: Depth of variational ansatz circuit
            convergence_threshold: Convergence threshold for optimization
            max_iterations: Maximum number of iterations
            seed: Random seed for reproducibility
        """
        self.ansatz_depth = ansatz_depth
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        # VQE parameters
        self.theta_params = None
        self.phi_params = None
        
        # Optimization tracking
        self.optimization_history = []
        self.eigenvalue_history = []
        self.best_eigenvalue = float('inf')
        self.best_eigenvector = None
        
        logger.info(f"Initialized VQE with depth {ansatz_depth}, threshold {convergence_threshold}")
    
    def optimize_portfolio(
        self,
        returns: np.ndarray,
        entanglement_matrix: np.ndarray,
        risk_tolerance: float = 0.1
    ) -> np.ndarray:
        """
        Optimize portfolio using VQE for risk minimization.
        
        Args:
            returns: Expected returns array
            entanglement_matrix: Asset entanglement matrix
            risk_tolerance: Risk tolerance parameter
            
        Returns:
            Optimal portfolio weights
        """
        logger.info(f"Starting VQE portfolio optimization for {len(returns)} assets")
        
        n_assets = len(returns)
        
        # Construct risk Hamiltonian
        risk_hamiltonian = self._construct_risk_hamiltonian(
            returns, entanglement_matrix, risk_tolerance
        )
        
        # Initialize variational parameters
        self._initialize_parameters(n_assets)
        
        # VQE optimization loop
        optimal_params = self._optimize_vqe_parameters(risk_hamiltonian, n_assets)
        
        # Extract portfolio weights from optimal state
        optimal_state = self._construct_ansatz_state(optimal_params, n_assets)
        portfolio_weights = self._extract_portfolio_weights(optimal_state, n_assets)
        
        # Normalize weights
        portfolio_weights = portfolio_weights / np.sum(portfolio_weights)
        
        logger.info(f"VQE optimization completed. Best eigenvalue: {self.best_eigenvalue:.6f}")
        
        return portfolio_weights
    
    def optimize_continuous_parameters(
        self,
        parameter_bounds: Dict[str, Tuple[float, float]],
        objective_function: Optional[Callable] = None
    ) -> Dict[str, float]:
        """
        Optimize continuous parameters using VQE.
        
        Args:
            parameter_bounds: Dictionary of parameter names to (min, max) bounds
            objective_function: Custom objective function (optional)
            
        Returns:
            Optimal parameter values
        """
        logger.info(f"Optimizing {len(parameter_bounds)} continuous parameters with VQE")
        
        # Encode continuous parameters as quantum amplitudes
        param_names = list(parameter_bounds.keys())
        n_params = len(param_names)
        
        # Create parameter Hamiltonian
        param_hamiltonian = self._construct_parameter_hamiltonian(
            parameter_bounds, objective_function
        )
        
        # Initialize parameters for VQE
        self._initialize_parameters(n_params)
        
        # VQE optimization
        optimal_vqe_params = self._optimize_vqe_parameters(param_hamiltonian, n_params)
        
        # Extract parameter values
        optimal_state = self._construct_ansatz_state(optimal_vqe_params, n_params)
        parameter_values = self._extract_parameter_values(
            optimal_state, param_names, parameter_bounds
        )
        
        return parameter_values
    
    def _construct_risk_hamiltonian(
        self,
        returns: np.ndarray,
        entanglement_matrix: np.ndarray,
        risk_tolerance: float
    ) -> np.ndarray:
        """
        Construct risk Hamiltonian for portfolio optimization.
        
        Args:
            returns: Expected returns
            entanglement_matrix: Asset entanglement relationships
            risk_tolerance: Risk tolerance parameter
            
        Returns:
            Risk Hamiltonian matrix
        """
        n_assets = len(returns)
        hamiltonian = np.zeros((2**n_assets, 2**n_assets))
        
        # Add return terms (diagonal)
        for i in range(2**n_assets):
            binary_config = self._int_to_binary(i, n_assets)
            expected_return = np.dot(binary_config, returns)
            hamiltonian[i, i] = -expected_return  # Negative for maximization
        
        # Add risk terms (off-diagonal entanglement)
        for i in range(2**n_assets):
            for j in range(i + 1, 2**n_assets):
                binary_i = self._int_to_binary(i, n_assets)
                binary_j = self._int_to_binary(j, n_assets)
                
                # Calculate entanglement-based risk
                risk_coupling = self._calculate_risk_coupling(
                    binary_i, binary_j, entanglement_matrix
                )
                
                hamiltonian[i, j] = risk_tolerance * risk_coupling
                hamiltonian[j, i] = risk_tolerance * risk_coupling
        
        return hamiltonian
    
    def _construct_parameter_hamiltonian(
        self,
        parameter_bounds: Dict[str, Tuple[float, float]],
        objective_function: Optional[Callable] = None
    ) -> np.ndarray:
        """
        Construct Hamiltonian for parameter optimization.
        
        Args:
            parameter_bounds: Parameter bounds
            objective_function: Custom objective function
            
        Returns:
            Parameter Hamiltonian matrix
        """
        n_params = len(parameter_bounds)
        hamiltonian_size = 2**n_params
        hamiltonian = np.zeros((hamiltonian_size, hamiltonian_size))
        
        # Default quadratic objective if none provided
        if objective_function is None:
            objective_function = lambda x: np.sum(x**2)  # Minimize sum of squares
        
        # Fill diagonal with objective function values
        param_names = list(parameter_bounds.keys())
        
        for i in range(hamiltonian_size):
            binary_config = self._int_to_binary(i, n_params)
            
            # Map binary config to parameter values
            param_values = {}
            for j, param_name in enumerate(param_names):
                min_val, max_val = parameter_bounds[param_name]
                # Map binary to continuous value
                param_values[param_name] = min_val + binary_config[j] * (max_val - min_val)
            
            # Evaluate objective function
            try:
                objective_value = objective_function(param_values)
                hamiltonian[i, i] = objective_value
            except Exception as e:
                logger.warning(f"Error evaluating objective function: {e}")
                hamiltonian[i, i] = 1.0  # Default penalty
        
        return hamiltonian
    
    def _initialize_parameters(self, n_qubits: int) -> None:
        """
        Initialize variational parameters for VQE ansatz.
        
        Args:
            n_qubits: Number of qubits in the system
        """
        # Initialize rotation angles for ansatz layers
        total_params = self.ansatz_depth * n_qubits * 2  # 2 parameters per qubit per layer
        
        self.theta_params = np.random.uniform(0, 2*np.pi, (self.ansatz_depth, n_qubits))
        self.phi_params = np.random.uniform(0, 2*np.pi, (self.ansatz_depth, n_qubits))
        
        logger.debug(f"Initialized {total_params} variational parameters")
    
    def _optimize_vqe_parameters(
        self,
        hamiltonian: np.ndarray,
        n_qubits: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimize VQE variational parameters using classical optimization.
        
        Args:
            hamiltonian: Problem Hamiltonian
            n_qubits: Number of qubits
            
        Returns:
            Optimal theta and phi parameters
        """
        
        def objective(params):
            # Reshape parameters
            n_params_per_layer = n_qubits * 2
            theta_flat = params[:self.ansatz_depth * n_qubits].reshape(self.ansatz_depth, n_qubits)
            phi_flat = params[self.ansatz_depth * n_qubits:].reshape(self.ansatz_depth, n_qubits)
            
            # Construct ansatz state
            state = self._construct_ansatz_state((theta_flat, phi_flat), n_qubits)
            
            # Calculate expectation value
            eigenvalue = np.real(state.conj().T @ hamiltonian @ state)
            
            # Store optimization history
            self.optimization_history.append(np.linalg.norm(params))
            self.eigenvalue_history.append(eigenvalue)
            
            if eigenvalue < self.best_eigenvalue:
                self.best_eigenvalue = eigenvalue
                self.best_eigenvector = state.copy()
            
            return eigenvalue
        
        # Flatten parameters for optimization
        initial_params = np.concatenate([
            self.theta_params.flatten(),
            self.phi_params.flatten()
        ])
        
        # Optimize using L-BFGS-B
        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            options={
                'maxiter': self.max_iterations,
                'ftol': self.convergence_threshold,
                'gtol': self.convergence_threshold
            }
        )
        
        # Extract optimal parameters
        optimal_params = result.x
        n_theta_params = self.ansatz_depth * n_qubits
        
        optimal_theta = optimal_params[:n_theta_params].reshape(self.ansatz_depth, n_qubits)
        optimal_phi = optimal_params[n_theta_params:].reshape(self.ansatz_depth, n_qubits)
        
        logger.info(f"VQE converged after {result.nit} iterations")
        
        return optimal_theta, optimal_phi
    
    def _construct_ansatz_state(
        self,
        params: Tuple[np.ndarray, np.ndarray],
        n_qubits: int
    ) -> np.ndarray:
        """
        Construct variational ansatz state.
        
        Args:
            params: Theta and phi parameters
            n_qubits: Number of qubits
            
        Returns:
            Quantum state vector
        """
        theta_params, phi_params = params
        
        # Start with |0...0⟩ state
        state = np.zeros(2**n_qubits)
        state[0] = 1.0
        
        # Apply ansatz layers
        for layer in range(self.ansatz_depth):
            # Apply parameterized rotations
            state = self._apply_rotation_layer(
                state, theta_params[layer], phi_params[layer], n_qubits
            )
            
            # Apply entangling gates
            state = self._apply_entangling_layer(state, n_qubits)
        
        return state / np.linalg.norm(state)
    
    def _apply_rotation_layer(
        self,
        state: np.ndarray,
        theta_vals: np.ndarray,
        phi_vals: np.ndarray,
        n_qubits: int
    ) -> np.ndarray:
        """
        Apply parameterized rotation layer to quantum state.
        
        Args:
            state: Current quantum state
            theta_vals: Theta rotation angles
            phi_vals: Phi rotation angles
            n_qubits: Number of qubits
            
        Returns:
            Updated quantum state
        """
        updated_state = state.copy()
        
        # Apply RY and RZ rotations to each qubit
        for qubit in range(n_qubits):
            theta = theta_vals[qubit]
            phi = phi_vals[qubit]
            
            # Apply RY(theta) followed by RZ(phi)
            updated_state = self._apply_single_qubit_rotation(
                updated_state, qubit, theta, phi, n_qubits
            )
        
        return updated_state
    
    def _apply_single_qubit_rotation(
        self,
        state: np.ndarray,
        qubit: int,
        theta: float,
        phi: float,
        n_qubits: int
    ) -> np.ndarray:
        """
        Apply single-qubit rotation to specific qubit.
        
        Args:
            state: Current state
            qubit: Target qubit index
            theta: Y-rotation angle
            phi: Z-rotation angle
            n_qubits: Total number of qubits
            
        Returns:
            Updated state
        """
        # Construct rotation matrix
        cos_half_theta = np.cos(theta / 2)
        sin_half_theta = np.sin(theta / 2)
        exp_phi = np.exp(1j * phi)
        
        rotation_matrix = np.array([
            [cos_half_theta, -sin_half_theta],
            [sin_half_theta * exp_phi, cos_half_theta * exp_phi]
        ])
        
        # Apply rotation to the specified qubit
        updated_state = np.zeros_like(state)
        
        for i in range(len(state)):
            binary_config = self._int_to_binary(i, n_qubits)
            
            # Extract qubit value
            qubit_val = binary_config[qubit]
            
            # Apply rotation
            for new_qubit_val in [0, 1]:
                new_config = binary_config.copy()
                new_config[qubit] = new_qubit_val
                new_index = self._binary_to_int(new_config)
                
                updated_state[new_index] += rotation_matrix[new_qubit_val, qubit_val] * state[i]
        
        return updated_state
    
    def _apply_entangling_layer(self, state: np.ndarray, n_qubits: int) -> np.ndarray:
        """
        Apply entangling layer (CNOT gates) to quantum state.
        
        Args:
            state: Current quantum state
            n_qubits: Number of qubits
            
        Returns:
            Updated quantum state with entanglement
        """
        updated_state = state.copy()
        
        # Apply CNOT gates between adjacent qubits
        for i in range(n_qubits - 1):
            control_qubit = i
            target_qubit = i + 1
            updated_state = self._apply_cnot(updated_state, control_qubit, target_qubit, n_qubits)
        
        return updated_state
    
    def _apply_cnot(
        self,
        state: np.ndarray,
        control: int,
        target: int,
        n_qubits: int
    ) -> np.ndarray:
        """
        Apply CNOT gate to quantum state.
        
        Args:
            state: Current state
            control: Control qubit index
            target: Target qubit index
            n_qubits: Total number of qubits
            
        Returns:
            Updated state after CNOT
        """
        updated_state = np.zeros_like(state)
        
        for i in range(len(state)):
            binary_config = self._int_to_binary(i, n_qubits)
            
            # Apply CNOT logic
            if binary_config[control] == 1:
                # Flip target qubit
                new_config = binary_config.copy()
                new_config[target] = 1 - new_config[target]
                new_index = self._binary_to_int(new_config)
                updated_state[new_index] = state[i]
            else:
                # No change
                updated_state[i] = state[i]
        
        return updated_state
    
    def _calculate_risk_coupling(
        self,
        config_i: np.ndarray,
        config_j: np.ndarray,
        entanglement_matrix: np.ndarray
    ) -> float:
        """
        Calculate risk coupling between two configurations.
        
        Args:
            config_i: First binary configuration
            config_j: Second binary configuration
            entanglement_matrix: Asset entanglement matrix
            
        Returns:
            Risk coupling value
        """
        coupling = 0.0
        
        for i in range(len(config_i)):
            for j in range(len(config_j)):
                if i != j:
                    coupling += (config_i[i] * config_j[j] * 
                               entanglement_matrix[i, j])
        
        return coupling
    
    def _extract_portfolio_weights(self, state: np.ndarray, n_assets: int) -> np.ndarray:
        """
        Extract portfolio weights from quantum state.
        
        Args:
            state: Final quantum state
            n_assets: Number of assets
            
        Returns:
            Portfolio weights
        """
        # Calculate probability distribution over asset configurations
        probabilities = np.abs(state)**2
        
        # Calculate expected weights
        weights = np.zeros(n_assets)
        
        for i, prob in enumerate(probabilities):
            binary_config = self._int_to_binary(i, n_assets)
            weights += prob * binary_config
        
        return weights
    
    def _extract_parameter_values(
        self,
        state: np.ndarray,
        param_names: List[str],
        parameter_bounds: Dict[str, Tuple[float, float]]
    ) -> Dict[str, float]:
        """
        Extract parameter values from quantum state.
        
        Args:
            state: Final quantum state
            param_names: Parameter names
            parameter_bounds: Parameter bounds
            
        Returns:
            Dictionary of parameter values
        """
        n_params = len(param_names)
        probabilities = np.abs(state)**2
        
        # Calculate expected parameter values
        param_values = {}
        
        for param_idx, param_name in enumerate(param_names):
            min_val, max_val = parameter_bounds[param_name]
            expected_value = 0.0
            
            for i, prob in enumerate(probabilities):
                binary_config = self._int_to_binary(i, n_params)
                param_binary_val = binary_config[param_idx]
                param_val = min_val + param_binary_val * (max_val - min_val)
                expected_value += prob * param_val
            
            param_values[param_name] = expected_value
        
        return param_values
    
    def _int_to_binary(self, integer: int, n_bits: int) -> np.ndarray:
        """Convert integer to binary array."""
        binary_str = format(integer, f'0{n_bits}b')
        return np.array([int(bit) for bit in binary_str])
    
    def _binary_to_int(self, binary: np.ndarray) -> int:
        """Convert binary array to integer."""
        return int(''.join(map(str, binary.astype(int))), 2)
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get VQE optimization summary.
        
        Returns:
            Summary dictionary
        """
        return {
            'ansatz_depth': self.ansatz_depth,
            'convergence_threshold': self.convergence_threshold,
            'best_eigenvalue': self.best_eigenvalue,
            'optimization_history': self.optimization_history,
            'eigenvalue_history': self.eigenvalue_history,
            'convergence_achieved': (
                len(self.eigenvalue_history) > 10 and
                abs(self.eigenvalue_history[-1] - self.eigenvalue_history[-10]) < self.convergence_threshold
            ),
            'total_iterations': len(self.optimization_history)
        }
    
    def calculate_fidelity(self, target_state: np.ndarray) -> float:
        """
        Calculate fidelity between best eigenvector and target state.
        
        Args:
            target_state: Target quantum state
            
        Returns:
            Fidelity measure
        """
        if self.best_eigenvector is None:
            return 0.0
        
        return abs(np.vdot(self.best_eigenvector, target_state))**2
    
    def reset_optimizer(self) -> None:
        """Reset VQE optimizer for new optimization."""
        self.theta_params = None
        self.phi_params = None
        self.optimization_history = []
        self.eigenvalue_history = []
        self.best_eigenvalue = float('inf')
        self.best_eigenvector = None
        
        logger.info("VQE optimizer reset for new optimization")