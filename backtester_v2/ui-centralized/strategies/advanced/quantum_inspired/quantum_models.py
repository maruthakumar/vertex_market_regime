"""
Quantum Strategy Data Models

This module defines data models for quantum-inspired trading strategies,
including quantum signals, trades, and parameters.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime


@dataclass
class QuantumSignal:
    """
    Quantum-enhanced trading signal with quantum state information.
    """
    timestamp: datetime
    symbol: str
    direction: str  # 'LONG', 'SHORT', 'EXIT'
    confidence: float
    quantum_state: Dict[str, Any]
    entanglement_measure: float
    coherence_time: float
    measurement_basis: str
    predicted_move: float = 0.0
    quantum_features: Dict[str, float] = field(default_factory=dict)
    algorithm_source: str = 'quantum_ensemble'
    phase_information: float = 0.0
    superposition_weights: List[float] = field(default_factory=list)
    decoherence_factor: float = 1.0
    
    def __post_init__(self):
        """Validate quantum signal parameters."""
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        
        if not 0 <= self.entanglement_measure <= 1:
            raise ValueError("Entanglement measure must be between 0 and 1")
        
        if self.coherence_time <= 0:
            raise ValueError("Coherence time must be positive")
    
    def apply_decoherence(self, time_elapsed: float) -> float:
        """
        Apply quantum decoherence effects to signal confidence.
        
        Args:
            time_elapsed: Time elapsed since signal generation
            
        Returns:
            Decoherence-adjusted confidence
        """
        decoherence_factor = np.exp(-time_elapsed / self.coherence_time)
        self.decoherence_factor = decoherence_factor
        return self.confidence * decoherence_factor
    
    def measure_quantum_state(self, basis: str = None) -> Dict[str, float]:
        """
        Perform quantum measurement of the signal's quantum state.
        
        Args:
            basis: Measurement basis ('computational', 'hadamard', 'custom')
            
        Returns:
            Measurement outcomes
        """
        if basis is None:
            basis = self.measurement_basis
        
        amplitudes = self.quantum_state.get('amplitudes', [self.confidence])
        phases = self.quantum_state.get('phases', [0.0])
        
        if basis == 'computational':
            # Standard computational basis measurement
            probabilities = [abs(amp)**2 for amp in amplitudes]
        elif basis == 'hadamard':
            # Hadamard basis measurement
            probabilities = [0.5 * (1 + np.real(amp * np.exp(1j * phase))) 
                           for amp, phase in zip(amplitudes, phases)]
        else:
            # Custom basis - use confidence as probability
            probabilities = [self.confidence]
        
        return {
            f'outcome_{i}': prob for i, prob in enumerate(probabilities)
        }
    
    def calculate_quantum_advantage(self, classical_confidence: float) -> float:
        """
        Calculate quantum advantage over classical signal.
        
        Args:
            classical_confidence: Classical signal confidence
            
        Returns:
            Quantum advantage ratio
        """
        if classical_confidence == 0:
            return float('inf') if self.confidence > 0 else 1.0
        
        return self.confidence / classical_confidence
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert quantum signal to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'direction': self.direction,
            'confidence': self.confidence,
            'entanglement_measure': self.entanglement_measure,
            'coherence_time': self.coherence_time,
            'measurement_basis': self.measurement_basis,
            'predicted_move': self.predicted_move,
            'quantum_features': self.quantum_features,
            'algorithm_source': self.algorithm_source,
            'phase_information': self.phase_information,
            'superposition_weights': self.superposition_weights,
            'decoherence_factor': self.decoherence_factor
        }


@dataclass
class QuantumTrade:
    """
    Quantum-enhanced trade execution with quantum optimization.
    """
    signal: QuantumSignal
    entry_price: float
    position_size: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None
    exit_reason: Optional[str] = None
    quantum_optimization: Dict[str, Any] = field(default_factory=dict)
    risk_quantum_state: Dict[str, float] = field(default_factory=dict)
    portfolio_entanglement: Optional[np.ndarray] = None
    
    @property
    def direction(self) -> str:
        """Get trade direction from signal."""
        return self.signal.direction
    
    @property
    def is_open(self) -> bool:
        """Check if trade is still open."""
        return self.exit_price is None
    
    @property
    def duration(self) -> Optional[float]:
        """Get trade duration in hours."""
        if self.exit_time is None:
            return None
        return (self.exit_time - self.entry_time).total_seconds() / 3600
    
    def calculate_quantum_risk(self) -> Dict[str, float]:
        """
        Calculate quantum-enhanced risk metrics.
        
        Returns:
            Dictionary of quantum risk measures
        """
        # Quantum risk based on entanglement and coherence
        entanglement_risk = self.signal.entanglement_measure * 0.5
        coherence_risk = (1 - self.signal.decoherence_factor) * 0.3
        
        # Position sizing risk using quantum superposition
        position_risk = abs(self.position_size) / (self.entry_price * 1000) * 0.2
        
        total_quantum_risk = entanglement_risk + coherence_risk + position_risk
        
        return {
            'entanglement_risk': entanglement_risk,
            'coherence_risk': coherence_risk,
            'position_risk': position_risk,
            'total_quantum_risk': total_quantum_risk,
            'risk_adjusted_confidence': self.signal.confidence * (1 - total_quantum_risk)
        }
    
    def optimize_exit_strategy(self, market_state: Dict[str, float]) -> Dict[str, float]:
        """
        Use quantum optimization for exit strategy.
        
        Args:
            market_state: Current market conditions
            
        Returns:
            Optimized exit parameters
        """
        # Quantum-inspired exit optimization
        current_price = market_state.get('current_price', self.entry_price)
        volatility = market_state.get('volatility', 0.2)
        
        # Apply quantum superposition to exit levels
        superposition_factor = 0.1
        
        # Quantum tunneling for stop loss adjustment
        quantum_stop_adjustment = np.random.normal(0, volatility * superposition_factor)
        optimized_stop = self.stop_loss * (1 + quantum_stop_adjustment)
        
        # Quantum interference for take profit optimization
        quantum_tp_adjustment = np.random.normal(0, volatility * superposition_factor * 0.5)
        optimized_tp = self.take_profit * (1 + quantum_tp_adjustment)
        
        return {
            'optimized_stop_loss': optimized_stop,
            'optimized_take_profit': optimized_tp,
            'quantum_exit_probability': self._calculate_exit_probability(current_price),
            'recommended_action': self._get_quantum_recommendation(current_price, market_state)
        }
    
    def _calculate_exit_probability(self, current_price: float) -> float:
        """Calculate quantum exit probability."""
        if self.direction == 'LONG':
            price_movement = (current_price - self.entry_price) / self.entry_price
        else:
            price_movement = (self.entry_price - current_price) / self.entry_price
        
        # Quantum measurement probability
        exit_probability = 1 / (1 + np.exp(-price_movement * 10))
        
        # Apply decoherence
        time_factor = 0.1  # Simplified time factor
        decoherence = np.exp(-time_factor / self.signal.coherence_time)
        
        return exit_probability * decoherence
    
    def _get_quantum_recommendation(self, current_price: float, market_state: Dict[str, float]) -> str:
        """Get quantum-based trade recommendation."""
        exit_prob = self._calculate_exit_probability(current_price)
        
        if exit_prob > 0.8:
            return 'STRONG_EXIT'
        elif exit_prob > 0.6:
            return 'CONSIDER_EXIT'
        elif exit_prob < 0.3:
            return 'HOLD'
        else:
            return 'MONITOR'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert quantum trade to dictionary."""
        return {
            'signal': self.signal.to_dict(),
            'entry_price': self.entry_price,
            'position_size': self.position_size,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'entry_time': self.entry_time.isoformat(),
            'exit_price': self.exit_price,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'pnl': self.pnl,
            'exit_reason': self.exit_reason,
            'direction': self.direction,
            'is_open': self.is_open,
            'duration': self.duration,
            'quantum_optimization': self.quantum_optimization,
            'risk_quantum_state': self.risk_quantum_state
        }


@dataclass
class QuantumParameters:
    """
    Configuration parameters for quantum trading strategies.
    """
    # Core quantum parameters
    coherence_time: float = 100.0
    entanglement_strength: float = 0.8
    superposition_factor: float = 0.6
    measurement_basis: str = 'computational'
    decoherence_rate: float = 0.01
    
    # Algorithm parameters
    qaoa_layers: int = 3
    qaoa_iterations: int = 100
    vqe_depth: int = 4
    vqe_threshold: float = 1e-6
    annealing_schedule: str = 'exponential'
    cooling_rate: float = 0.95
    walk_length: int = 100
    coin_bias: float = 0.5
    hybrid_layers: int = 2
    
    # Trading parameters
    signal_threshold: float = 0.7
    risk_aversion: float = 1.0
    max_weight: float = 0.3
    min_weight: float = 0.0
    risk_budget: float = 0.15
    confidence_threshold: float = 0.6
    
    # Optimization parameters
    optimization_iterations: int = 1000
    convergence_tolerance: float = 1e-8
    max_function_evaluations: int = 10000
    
    # Portfolio parameters
    max_positions: int = 10
    max_correlated_positions: int = 3
    position_sizing: str = 'quantum_kelly'
    rebalancing_frequency: str = 'daily'
    
    # Risk management
    max_portfolio_risk: float = 0.2
    max_individual_risk: float = 0.05
    stop_loss_method: str = 'quantum_adaptive'
    take_profit_method: str = 'quantum_interference'
    
    def __post_init__(self):
        """Validate quantum parameters."""
        self._validate_ranges()
        self._validate_algorithm_params()
    
    def _validate_ranges(self):
        """Validate parameter ranges."""
        if not 0 < self.coherence_time <= 1000:
            raise ValueError("Coherence time must be between 0 and 1000")
        
        if not 0 <= self.entanglement_strength <= 1:
            raise ValueError("Entanglement strength must be between 0 and 1")
        
        if not 0 <= self.superposition_factor <= 1:
            raise ValueError("Superposition factor must be between 0 and 1")
        
        if not 0 <= self.signal_threshold <= 1:
            raise ValueError("Signal threshold must be between 0 and 1")
    
    def _validate_algorithm_params(self):
        """Validate algorithm-specific parameters."""
        if self.qaoa_layers < 1:
            raise ValueError("QAOA layers must be at least 1")
        
        if self.vqe_depth < 1:
            raise ValueError("VQE depth must be at least 1")
        
        if self.cooling_rate <= 0 or self.cooling_rate >= 1:
            raise ValueError("Cooling rate must be between 0 and 1")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary."""
        return {
            'coherence_time': self.coherence_time,
            'entanglement_strength': self.entanglement_strength,
            'superposition_factor': self.superposition_factor,
            'measurement_basis': self.measurement_basis,
            'decoherence_rate': self.decoherence_rate,
            'qaoa_layers': self.qaoa_layers,
            'qaoa_iterations': self.qaoa_iterations,
            'vqe_depth': self.vqe_depth,
            'vqe_threshold': self.vqe_threshold,
            'annealing_schedule': self.annealing_schedule,
            'cooling_rate': self.cooling_rate,
            'walk_length': self.walk_length,
            'coin_bias': self.coin_bias,
            'hybrid_layers': self.hybrid_layers,
            'signal_threshold': self.signal_threshold,
            'risk_aversion': self.risk_aversion,
            'max_weight': self.max_weight,
            'min_weight': self.min_weight,
            'risk_budget': self.risk_budget,
            'confidence_threshold': self.confidence_threshold
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuantumParameters':
        """Create parameters from dictionary."""
        return cls(**data)
    
    def create_quantum_superposition(self, base_params: Dict[str, float]) -> Dict[str, List[float]]:
        """
        Create quantum superposition of parameters for optimization.
        
        Args:
            base_params: Base parameter values
            
        Returns:
            Dictionary with superposition parameter ranges
        """
        superposition_params = {}
        
        for param, value in base_params.items():
            if isinstance(value, (int, float)):
                variance = abs(value) * self.superposition_factor
                superposition_params[param] = [
                    value - variance,
                    value,
                    value + variance
                ]
            else:
                superposition_params[param] = [value]
        
        return superposition_params
    
    def optimize_for_market_regime(self, regime: str) -> 'QuantumParameters':
        """
        Optimize quantum parameters for specific market regime.
        
        Args:
            regime: Market regime ('trending', 'ranging', 'volatile', 'calm')
            
        Returns:
            Optimized quantum parameters
        """
        optimized = QuantumParameters(**self.to_dict())
        
        if regime == 'trending':
            optimized.entanglement_strength = min(0.9, self.entanglement_strength * 1.2)
            optimized.superposition_factor = max(0.3, self.superposition_factor * 0.8)
            optimized.coherence_time = self.coherence_time * 1.1
        
        elif regime == 'ranging':
            optimized.entanglement_strength = max(0.5, self.entanglement_strength * 0.8)
            optimized.superposition_factor = min(0.8, self.superposition_factor * 1.2)
            optimized.coherence_time = self.coherence_time * 0.9
        
        elif regime == 'volatile':
            optimized.decoherence_rate = self.decoherence_rate * 1.5
            optimized.superposition_factor = min(0.9, self.superposition_factor * 1.3)
            optimized.signal_threshold = min(0.8, self.signal_threshold * 1.1)
        
        elif regime == 'calm':
            optimized.decoherence_rate = self.decoherence_rate * 0.7
            optimized.coherence_time = self.coherence_time * 1.2
            optimized.signal_threshold = max(0.5, self.signal_threshold * 0.9)
        
        return optimized


@dataclass
class QuantumPortfolio:
    """
    Quantum-enhanced portfolio representation.
    """
    assets: List[str]
    weights: Dict[str, float]
    entanglement_matrix: np.ndarray
    quantum_risk: float
    coherence_measure: float
    total_value: float = 0.0
    quantum_state: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate portfolio construction."""
        if abs(sum(self.weights.values()) - 1.0) > 1e-6:
            raise ValueError("Portfolio weights must sum to 1.0")
        
        if len(self.assets) != len(self.weights):
            raise ValueError("Number of assets must match number of weights")
    
    def calculate_quantum_diversification(self) -> float:
        """
        Calculate quantum diversification benefit.
        
        Returns:
            Quantum diversification ratio
        """
        # Use entanglement matrix to calculate diversification
        n_assets = len(self.assets)
        
        if n_assets <= 1:
            return 1.0
        
        # Classical diversification
        equal_weight_variance = 1.0 / n_assets
        
        # Quantum diversification using entanglement
        entanglement_effect = np.mean(np.abs(self.entanglement_matrix))
        quantum_variance = equal_weight_variance * (1 - entanglement_effect)
        
        return equal_weight_variance / quantum_variance
    
    def measure_portfolio_state(self) -> Dict[str, float]:
        """
        Perform quantum measurement of portfolio state.
        
        Returns:
            Portfolio measurement outcomes
        """
        measurements = {}
        
        for i, asset in enumerate(self.assets):
            weight = self.weights[asset]
            
            # Quantum measurement based on weight amplitudes
            amplitude = np.sqrt(weight)
            probability = amplitude**2
            
            # Apply entanglement effects
            entanglement_sum = np.sum(np.abs(self.entanglement_matrix[i]))
            adjusted_probability = probability * (1 + entanglement_sum * 0.1)
            
            measurements[asset] = min(adjusted_probability, 1.0)
        
        return measurements
    
    def optimize_quantum_rebalancing(self, target_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Optimize portfolio rebalancing using quantum algorithms.
        
        Args:
            target_weights: Target portfolio weights
            
        Returns:
            Optimal rebalancing trades
        """
        rebalancing_trades = {}
        
        for asset in self.assets:
            current_weight = self.weights.get(asset, 0.0)
            target_weight = target_weights.get(asset, 0.0)
            
            trade_amount = target_weight - current_weight
            
            # Apply quantum tunneling for small adjustments
            if abs(trade_amount) < 0.01:
                quantum_tunneling = np.random.normal(0, 0.005)
                trade_amount += quantum_tunneling
            
            rebalancing_trades[asset] = trade_amount
        
        return rebalancing_trades
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert quantum portfolio to dictionary."""
        return {
            'assets': self.assets,
            'weights': self.weights,
            'entanglement_matrix': self.entanglement_matrix.tolist() if self.entanglement_matrix is not None else None,
            'quantum_risk': self.quantum_risk,
            'coherence_measure': self.coherence_measure,
            'total_value': self.total_value,
            'quantum_state': self.quantum_state,
            'quantum_diversification': self.calculate_quantum_diversification()
        }