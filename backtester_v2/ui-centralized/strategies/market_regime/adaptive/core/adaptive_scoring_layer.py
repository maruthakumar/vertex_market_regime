"""
Adaptive Scoring Layer (ASL)

This module implements the ASL-inspired adaptive weight system that dynamically
adjusts component weights based on prediction accuracy and market conditions.

Key Features:
- Dynamic weight evolution based on performance
- Component score calculation with adaptive weights
- Performance-based weight updates using gradient descent
- Real-time adaptation to market conditions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import deque
import logging
from dataclasses import dataclass, field
import json

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ComponentScore:
    """Represents a score from a single component"""
    component_name: str
    raw_score: float
    weighted_score: float
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ASLConfiguration:
    """Configuration for Adaptive Scoring Layer"""
    learning_rate: float = 0.05
    decay_factor: float = 0.95
    min_weight: float = 0.05
    max_weight: float = 0.8
    performance_window: int = 100
    update_frequency: int = 10
    gradient_clip: float = 0.1
    momentum: float = 0.9
    adaptive_lr: bool = True
    weight_regularization: float = 0.01


class AdaptiveScoringLayer:
    """
    Implements adaptive scoring with dynamic weight adjustment
    """
    
    def __init__(self, config: Optional[ASLConfiguration] = None,
                 initial_weights: Optional[Dict[str, float]] = None):
        """
        Initialize Adaptive Scoring Layer
        
        Args:
            config: ASL configuration
            initial_weights: Initial component weights
        """
        self.config = config or ASLConfiguration()
        
        # Default components and weights
        self.components = [
            'triple_straddle',
            'greek_sentiment', 
            'oi_analysis',
            'technical',
            'ml_ensemble'
        ]
        
        # Initialize weights
        if initial_weights:
            self.weights = initial_weights.copy()
        else:
            # Equal weights by default
            equal_weight = 1.0 / len(self.components)
            self.weights = {comp: equal_weight for comp in self.components}
        
        # Performance tracking
        self.performance_history = deque(maxlen=self.config.performance_window)
        self.weight_history = deque(maxlen=1000)
        self.gradient_history = {comp: deque(maxlen=20) for comp in self.components}
        
        # Momentum for gradient descent
        self.momentum_buffer = {comp: 0.0 for comp in self.components}
        
        # Adaptive learning rate
        self.current_lr = self.config.learning_rate
        self.lr_scheduler_step = 0
        
        # Performance metrics
        self.total_predictions = 0
        self.correct_predictions = 0
        self.last_update_time = datetime.now()
        
        logger.info(f"AdaptiveScoringLayer initialized with weights: {self.weights}")
    
    def calculate_regime_scores(self, market_data: Dict[str, Any], 
                              historical_context: Optional[Dict[str, Any]] = None) -> Dict[int, float]:
        """
        Calculate adaptive regime scores with evolving weights
        
        Args:
            market_data: Current market data
            historical_context: Historical data for context
            
        Returns:
            Dictionary of regime scores
        """
        try:
            # Calculate component scores
            component_scores = self._calculate_component_scores(market_data, historical_context)
            
            # Apply adaptive weights
            weighted_scores = self._apply_adaptive_weights(component_scores)
            
            # Aggregate to regime scores
            regime_scores = self._aggregate_to_regime_scores(weighted_scores, market_data)
            
            # Record for performance tracking
            self._record_prediction(component_scores, weighted_scores, regime_scores)
            
            return regime_scores
            
        except Exception as e:
            logger.error(f"Error calculating regime scores: {e}")
            # Return uniform scores as fallback
            return {i: 1.0/8 for i in range(8)}  # Assuming 8 regimes
    
    def _calculate_component_scores(self, market_data: Dict[str, Any],
                                  historical_context: Optional[Dict[str, Any]]) -> Dict[str, ComponentScore]:
        """
        Calculate raw scores from each component
        
        Args:
            market_data: Market data
            historical_context: Historical context
            
        Returns:
            Component scores
        """
        scores = {}
        
        # Triple Straddle Score
        scores['triple_straddle'] = ComponentScore(
            component_name='triple_straddle',
            raw_score=self._calculate_triple_straddle_score(market_data),
            weighted_score=0.0,  # Will be set later
            confidence=0.8,
            timestamp=datetime.now()
        )
        
        # Greek Sentiment Score
        scores['greek_sentiment'] = ComponentScore(
            component_name='greek_sentiment',
            raw_score=self._calculate_greek_sentiment_score(market_data),
            weighted_score=0.0,
            confidence=0.75,
            timestamp=datetime.now()
        )
        
        # OI Analysis Score
        scores['oi_analysis'] = ComponentScore(
            component_name='oi_analysis',
            raw_score=self._calculate_oi_analysis_score(market_data),
            weighted_score=0.0,
            confidence=0.7,
            timestamp=datetime.now()
        )
        
        # Technical Indicators Score
        scores['technical'] = ComponentScore(
            component_name='technical',
            raw_score=self._calculate_technical_score(market_data),
            weighted_score=0.0,
            confidence=0.65,
            timestamp=datetime.now()
        )
        
        # ML Ensemble Score
        scores['ml_ensemble'] = ComponentScore(
            component_name='ml_ensemble',
            raw_score=self._calculate_ml_ensemble_score(market_data, historical_context),
            weighted_score=0.0,
            confidence=0.85,
            timestamp=datetime.now()
        )
        
        return scores
    
    def _calculate_triple_straddle_score(self, market_data: Dict[str, Any]) -> float:
        """Calculate Triple Straddle component score"""
        try:
            # Simplified calculation - would integrate with actual Triple Straddle
            straddle_value = market_data.get('triple_straddle_value', 0.0)
            
            # Normalize to 0-1 range
            normalized = np.tanh(straddle_value / 100.0) * 0.5 + 0.5
            
            return float(normalized)
            
        except Exception as e:
            logger.error(f"Error in triple straddle calculation: {e}")
            return 0.5
    
    def _calculate_greek_sentiment_score(self, market_data: Dict[str, Any]) -> float:
        """Calculate Greek Sentiment component score"""
        try:
            # Extract Greek values
            delta_total = market_data.get('total_delta', 0.0)
            gamma_total = market_data.get('total_gamma', 0.0)
            vega_total = market_data.get('total_vega', 0.0)
            
            # Sentiment calculation
            delta_sentiment = np.tanh(delta_total / 1000.0)
            gamma_sentiment = -np.tanh(gamma_total / 500.0)  # Negative gamma is bullish
            vega_sentiment = np.tanh(vega_total / 1000.0)
            
            # Weighted combination
            sentiment = (
                delta_sentiment * 0.5 +
                gamma_sentiment * 0.3 +
                vega_sentiment * 0.2
            )
            
            # Normalize to 0-1
            return float((sentiment + 1.0) / 2.0)
            
        except Exception as e:
            logger.error(f"Error in Greek sentiment calculation: {e}")
            return 0.5
    
    def _calculate_oi_analysis_score(self, market_data: Dict[str, Any]) -> float:
        """Calculate OI Analysis component score"""
        try:
            # OI metrics
            call_oi = market_data.get('call_open_interest', 1.0)
            put_oi = market_data.get('put_open_interest', 1.0)
            
            # PCR calculation
            pcr = put_oi / max(call_oi, 1.0)
            
            # OI trend
            oi_change = market_data.get('oi_change_percent', 0.0)
            
            # Combined score
            pcr_score = 1.0 / (1.0 + np.exp(-2.0 * (pcr - 1.0)))  # Sigmoid around 1.0
            trend_score = np.tanh(oi_change / 10.0) * 0.5 + 0.5
            
            return float(pcr_score * 0.7 + trend_score * 0.3)
            
        except Exception as e:
            logger.error(f"Error in OI analysis calculation: {e}")
            return 0.5
    
    def _calculate_technical_score(self, market_data: Dict[str, Any]) -> float:
        """Calculate Technical Indicators component score"""
        try:
            # Technical indicators
            rsi = market_data.get('rsi', 50.0)
            macd_signal = market_data.get('macd_signal', 0.0)
            bb_position = market_data.get('bb_position', 0.5)  # Position in Bollinger Bands
            
            # RSI score (oversold/overbought)
            rsi_score = 1.0 - abs(rsi - 50.0) / 50.0
            
            # MACD score
            macd_score = np.tanh(macd_signal) * 0.5 + 0.5
            
            # Bollinger Bands score
            bb_score = 1.0 - abs(bb_position - 0.5) * 2.0
            
            # Combined score
            return float(rsi_score * 0.4 + macd_score * 0.4 + bb_score * 0.2)
            
        except Exception as e:
            logger.error(f"Error in technical score calculation: {e}")
            return 0.5
    
    def _calculate_ml_ensemble_score(self, market_data: Dict[str, Any],
                                   historical_context: Optional[Dict[str, Any]]) -> float:
        """Calculate ML Ensemble component score"""
        try:
            # ML predictions (simplified - would use actual ML models)
            ml_prediction = market_data.get('ml_regime_prediction', 0.5)
            ml_confidence = market_data.get('ml_confidence', 0.5)
            
            # Historical accuracy adjustment
            if historical_context:
                historical_accuracy = historical_context.get('ml_accuracy', 0.7)
                ml_prediction *= historical_accuracy
            
            # Confidence-weighted score
            return float(ml_prediction * ml_confidence)
            
        except Exception as e:
            logger.error(f"Error in ML ensemble calculation: {e}")
            return 0.5
    
    def _apply_adaptive_weights(self, component_scores: Dict[str, ComponentScore]) -> Dict[str, float]:
        """
        Apply adaptive weights to component scores
        
        Args:
            component_scores: Raw component scores
            
        Returns:
            Weighted scores
        """
        weighted_scores = {}
        
        for component, score in component_scores.items():
            weight = self.weights.get(component, 0.0)
            score.weighted_score = score.raw_score * weight
            weighted_scores[component] = score.weighted_score
        
        return weighted_scores
    
    def _aggregate_to_regime_scores(self, weighted_scores: Dict[str, float],
                                  market_data: Dict[str, Any]) -> Dict[int, float]:
        """
        Aggregate weighted component scores to regime scores
        
        Args:
            weighted_scores: Weighted component scores
            market_data: Market data for context
            
        Returns:
            Regime scores
        """
        # Get regime count from configuration
        regime_count = market_data.get('regime_count', 8)
        
        # Initialize regime scores
        regime_scores = {}
        
        # Calculate aggregate score
        total_score = sum(weighted_scores.values())
        
        # Map to regimes based on score ranges
        # This is simplified - actual implementation would use more sophisticated mapping
        for regime_id in range(regime_count):
            # Create regime-specific adjustments
            regime_factor = self._get_regime_factor(regime_id, total_score, market_data)
            
            # Base score with adjustments
            base_score = total_score * regime_factor
            
            # Add regime-specific biases
            if regime_id < regime_count // 3:  # Bullish regimes
                base_score *= (1.0 + weighted_scores.get('greek_sentiment', 0.0))
            elif regime_id > 2 * regime_count // 3:  # Bearish regimes
                base_score *= (1.0 - weighted_scores.get('greek_sentiment', 0.0))
            
            # Normalize
            regime_scores[regime_id] = max(0.0, min(1.0, base_score))
        
        # Normalize to sum to 1
        total = sum(regime_scores.values())
        if total > 0:
            regime_scores = {k: v/total for k, v in regime_scores.items()}
        else:
            regime_scores = {i: 1.0/regime_count for i in range(regime_count)}
        
        return regime_scores
    
    def _get_regime_factor(self, regime_id: int, total_score: float,
                         market_data: Dict[str, Any]) -> float:
        """
        Get regime-specific adjustment factor
        
        Args:
            regime_id: Regime identifier
            total_score: Total weighted score
            market_data: Market data
            
        Returns:
            Adjustment factor
        """
        # Volatility-based adjustments
        volatility = market_data.get('volatility', 0.2)
        
        regime_count = market_data.get('regime_count', 8)
        regimes_per_vol = regime_count // 3
        
        # Low volatility regimes
        if regime_id < regimes_per_vol:
            factor = 1.0 - volatility
        # Medium volatility regimes
        elif regime_id < 2 * regimes_per_vol:
            factor = 1.0 - abs(volatility - 0.5) * 2
        # High volatility regimes
        else:
            factor = volatility
        
        return max(0.1, factor)
    
    def update_weights_based_on_performance(self, predicted_scores: Dict[int, float],
                                          actual_outcome: Any):
        """
        Update weights based on prediction performance
        
        Args:
            predicted_scores: Predicted regime scores
            actual_outcome: Actual regime or performance metric
        """
        try:
            self.total_predictions += 1
            
            # Calculate prediction error
            error = self._calculate_prediction_error(predicted_scores, actual_outcome)
            
            # Record performance
            self.performance_history.append({
                'timestamp': datetime.now(),
                'error': error,
                'predicted': predicted_scores,
                'actual': actual_outcome
            })
            
            # Update weights if needed
            if self.total_predictions % self.config.update_frequency == 0:
                self._perform_weight_update()
                
        except Exception as e:
            logger.error(f"Error updating weights: {e}")
    
    def _calculate_prediction_error(self, predicted_scores: Dict[int, float],
                                  actual_outcome: Any) -> float:
        """
        Calculate prediction error
        
        Args:
            predicted_scores: Predicted regime scores
            actual_outcome: Actual outcome
            
        Returns:
            Error value
        """
        if isinstance(actual_outcome, int):
            # Actual regime known
            predicted_regime = max(predicted_scores.items(), key=lambda x: x[1])[0]
            error = 0.0 if predicted_regime == actual_outcome else 1.0
            
            # Add confidence penalty
            confidence = predicted_scores.get(actual_outcome, 0.0)
            error += (1.0 - confidence) * 0.5
            
        elif isinstance(actual_outcome, dict):
            # Performance-based outcome
            # Calculate based on profitability or other metrics
            performance = actual_outcome.get('performance', 0.0)
            error = max(0.0, 1.0 - performance)
        else:
            error = 0.5  # Default error
        
        return error
    
    def _perform_weight_update(self):
        """
        Perform weight update using gradient descent
        """
        if len(self.performance_history) < 10:
            return
        
        try:
            # Calculate gradients for each component
            gradients = self._calculate_gradients()
            
            # Update learning rate if adaptive
            if self.config.adaptive_lr:
                self._update_learning_rate()
            
            # Update weights with momentum
            for component in self.components:
                if component in gradients:
                    # Momentum update
                    self.momentum_buffer[component] = (
                        self.config.momentum * self.momentum_buffer[component] +
                        (1 - self.config.momentum) * gradients[component]
                    )
                    
                    # Gradient clipping
                    clipped_gradient = np.clip(
                        self.momentum_buffer[component],
                        -self.config.gradient_clip,
                        self.config.gradient_clip
                    )
                    
                    # Weight update
                    self.weights[component] -= self.current_lr * clipped_gradient
                    
                    # L2 regularization
                    self.weights[component] -= (
                        self.config.weight_regularization * 
                        self.weights[component]
                    )
            
            # Enforce constraints
            self._enforce_weight_constraints()
            
            # Normalize weights
            self._normalize_weights()
            
            # Record weight history
            self.weight_history.append({
                'timestamp': datetime.now(),
                'weights': self.weights.copy(),
                'learning_rate': self.current_lr
            })
            
            logger.debug(f"Weights updated: {self.weights}")
            
        except Exception as e:
            logger.error(f"Error in weight update: {e}")
    
    def _calculate_gradients(self) -> Dict[str, float]:
        """
        Calculate gradients for weight updates
        
        Returns:
            Gradients for each component
        """
        gradients = {comp: 0.0 for comp in self.components}
        
        # Use recent performance history
        recent_history = list(self.performance_history)[-20:]
        
        for record in recent_history:
            error = record['error']
            
            # Approximate gradient calculation
            # In practice, this would use backpropagation or policy gradient
            for component in self.components:
                # Higher weight components get more blame for errors
                component_contribution = self.weights[component]
                gradient_contribution = error * component_contribution
                
                # Add to gradient
                gradients[component] += gradient_contribution
        
        # Average gradients
        num_samples = len(recent_history)
        if num_samples > 0:
            gradients = {k: v/num_samples for k, v in gradients.items()}
        
        # Store gradient history
        for comp, grad in gradients.items():
            self.gradient_history[comp].append(grad)
        
        return gradients
    
    def _update_learning_rate(self):
        """Update learning rate using decay schedule"""
        self.lr_scheduler_step += 1
        
        # Exponential decay
        self.current_lr = self.config.learning_rate * (
            self.config.decay_factor ** (self.lr_scheduler_step / 100)
        )
        
        # Minimum learning rate
        self.current_lr = max(self.current_lr, self.config.learning_rate * 0.01)
    
    def _enforce_weight_constraints(self):
        """Enforce min/max weight constraints"""
        for component in self.components:
            self.weights[component] = np.clip(
                self.weights[component],
                self.config.min_weight,
                self.config.max_weight
            )
    
    def _normalize_weights(self):
        """Normalize weights to sum to 1"""
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
    
    def _record_prediction(self, component_scores: Dict[str, ComponentScore],
                         weighted_scores: Dict[str, float],
                         regime_scores: Dict[int, float]):
        """Record prediction for analysis"""
        # This would integrate with performance tracking system
        pass
    
    def get_weight_history(self) -> pd.DataFrame:
        """
        Get weight history as DataFrame
        
        Returns:
            DataFrame with weight history
        """
        if not self.weight_history:
            return pd.DataFrame()
        
        records = []
        for record in self.weight_history:
            row = {'timestamp': record['timestamp'], 'learning_rate': record['learning_rate']}
            row.update(record['weights'])
            records.append(row)
        
        return pd.DataFrame(records)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics
        
        Returns:
            Performance metrics dictionary
        """
        if not self.performance_history:
            return {}
        
        recent_errors = [r['error'] for r in list(self.performance_history)[-100:]]
        
        metrics = {
            'total_predictions': self.total_predictions,
            'recent_avg_error': np.mean(recent_errors) if recent_errors else 0.0,
            'recent_std_error': np.std(recent_errors) if recent_errors else 0.0,
            'current_weights': self.weights.copy(),
            'current_learning_rate': self.current_lr,
            'weight_variance': np.var(list(self.weights.values()))
        }
        
        return metrics
    
    def reset_adaptation(self):
        """Reset adaptive weights to initial state"""
        equal_weight = 1.0 / len(self.components)
        self.weights = {comp: equal_weight for comp in self.components}
        self.momentum_buffer = {comp: 0.0 for comp in self.components}
        self.current_lr = self.config.learning_rate
        self.lr_scheduler_step = 0
        self.performance_history.clear()
        self.weight_history.clear()
        
        logger.info("Adaptive weights reset to initial state")
    
    def save_state(self, filepath: str):
        """Save ASL state to file"""
        state = {
            'weights': self.weights,
            'momentum_buffer': self.momentum_buffer,
            'current_lr': self.current_lr,
            'lr_scheduler_step': self.lr_scheduler_step,
            'total_predictions': self.total_predictions,
            'performance_history': list(self.performance_history)[-100:],  # Last 100
            'weight_history': list(self.weight_history)[-100:]  # Last 100
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"ASL state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load ASL state from file"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.weights = state.get('weights', self.weights)
            self.momentum_buffer = state.get('momentum_buffer', self.momentum_buffer)
            self.current_lr = state.get('current_lr', self.current_lr)
            self.lr_scheduler_step = state.get('lr_scheduler_step', 0)
            self.total_predictions = state.get('total_predictions', 0)
            
            # Restore histories
            perf_history = state.get('performance_history', [])
            for record in perf_history:
                self.performance_history.append(record)
            
            weight_history = state.get('weight_history', [])
            for record in weight_history:
                self.weight_history.append(record)
            
            logger.info(f"ASL state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading ASL state: {e}")


# Example usage
if __name__ == "__main__":
    # Create ASL with custom configuration
    config = ASLConfiguration(
        learning_rate=0.05,
        decay_factor=0.95,
        performance_window=100
    )
    
    asl = AdaptiveScoringLayer(config)
    
    # Example market data
    market_data = {
        'regime_count': 8,
        'triple_straddle_value': 120.5,
        'total_delta': 1500,
        'total_gamma': -300,
        'total_vega': 800,
        'call_open_interest': 50000,
        'put_open_interest': 45000,
        'oi_change_percent': 5.2,
        'rsi': 58,
        'macd_signal': 0.5,
        'bb_position': 0.7,
        'ml_regime_prediction': 0.75,
        'ml_confidence': 0.82,
        'volatility': 0.25
    }
    
    # Calculate regime scores
    scores = asl.calculate_regime_scores(market_data)
    print("\nRegime Scores:")
    for regime_id, score in sorted(scores.items()):
        print(f"Regime {regime_id}: {score:.4f}")
    
    # Simulate performance update
    asl.update_weights_based_on_performance(scores, actual_outcome=2)
    
    # Get performance metrics
    metrics = asl.get_performance_metrics()
    print("\nPerformance Metrics:")
    print(f"Current weights: {metrics['current_weights']}")
    print(f"Learning rate: {metrics['current_learning_rate']:.6f}")