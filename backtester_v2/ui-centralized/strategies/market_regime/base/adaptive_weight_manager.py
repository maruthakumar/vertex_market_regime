"""
Adaptive Weight Manager for Market Regime Indicators
===================================================

Manages dynamic weight optimization using ML models and statistical analysis.

Author: Market Regime Refactoring Team
Date: 2025-07-06
Version: 2.0.0 - Refactored Architecture
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize
import logging
import json

logger = logging.getLogger(__name__)

@dataclass
class WeightOptimizationConfig:
    """Configuration for weight optimization"""
    learning_rate: float = 0.01
    decay_factor: float = 0.95
    min_weight: float = 0.01
    max_weight: float = 3.0
    window_size: int = 100
    confidence_threshold: float = 0.05
    ml_update_frequency: int = 50  # Updates before retraining ML models
    statistical_significance_threshold: float = 0.05

@dataclass
class WeightUpdateRecord:
    """Record of weight updates"""
    timestamp: datetime
    indicator_name: str
    old_weight: float
    new_weight: float
    reason: str
    performance_delta: float
    confidence: float

class AdaptiveWeightManager:
    """
    Adaptive weight management system using ML and statistical optimization
    
    Combines multiple approaches:
    - Exponential weighted moving averages
    - Random Forest for pattern recognition
    - Linear regression for trend analysis
    - Bayesian optimization for parameter tuning
    """
    
    def __init__(self, 
                 config: Optional[WeightOptimizationConfig] = None,
                 performance_tracker=None):
        """Initialize adaptive weight manager"""
        self.config = config or WeightOptimizationConfig()
        self.performance_tracker = performance_tracker
        
        # Current weights
        self.current_weights: Dict[str, float] = {}
        self.weight_history: Dict[str, List[WeightUpdateRecord]] = {}
        
        # ML Models
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.lr_model = LinearRegression()
        self.scaler = StandardScaler()
        
        # Model state
        self.models_trained = False
        self.last_ml_update = 0
        self.feature_names = []
        
        # Performance tracking
        self.performance_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self.baseline_performance: Dict[str, float] = {}
        
        # Optimization state
        self.optimization_state = {
            'total_updates': 0,
            'successful_updates': 0,
            'failed_updates': 0,
            'last_optimization': None
        }
        
        logger.info("AdaptiveWeightManager initialized")
    
    def initialize_weights(self, indicator_names: List[str], equal_weights: bool = True):
        """
        Initialize weights for indicators
        
        Args:
            indicator_names: List of indicator names
            equal_weights: Whether to use equal weights initially
        """
        if equal_weights:
            weight = 1.0 / len(indicator_names)
            for name in indicator_names:
                self.current_weights[name] = weight
        else:
            # Use performance-based initial weights if available
            for name in indicator_names:
                if self.performance_tracker:
                    metrics = self.performance_tracker.calculate_performance_metrics(name)
                    initial_weight = max(self.config.min_weight, metrics.accuracy)
                else:
                    initial_weight = 1.0 / len(indicator_names)
                
                self.current_weights[name] = initial_weight
        
        # Normalize weights
        self._normalize_weights()
        
        # Initialize history
        for name in indicator_names:
            self.weight_history[name] = []
            self.performance_history[name] = []
        
        logger.info(f"Initialized weights for {len(indicator_names)} indicators")
    
    def update_weights_from_performance(self, 
                                      performance_data: Dict[str, float],
                                      market_conditions: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Update weights based on recent performance
        
        Args:
            performance_data: Dictionary of indicator -> performance score
            market_conditions: Current market conditions for context
            
        Returns:
            Dict[str, float]: Updated weights
        """
        try:
            timestamp = datetime.now()
            updates_made = False
            
            for indicator_name, performance in performance_data.items():
                if indicator_name not in self.current_weights:
                    continue
                
                # Update performance history
                self.performance_history[indicator_name].append((timestamp, performance))
                
                # Keep only recent history
                cutoff = timestamp - timedelta(days=30)
                self.performance_history[indicator_name] = [
                    (ts, perf) for ts, perf in self.performance_history[indicator_name]
                    if ts > cutoff
                ]
                
                # Calculate weight adjustment
                old_weight = self.current_weights[indicator_name]
                new_weight = self._calculate_performance_based_weight(
                    indicator_name, performance, market_conditions
                )
                
                # Apply weight update if significant change
                if abs(new_weight - old_weight) > self.config.confidence_threshold:
                    self.current_weights[indicator_name] = new_weight
                    updates_made = True
                    
                    # Record update
                    update_record = WeightUpdateRecord(
                        timestamp=timestamp,
                        indicator_name=indicator_name,
                        old_weight=old_weight,
                        new_weight=new_weight,
                        reason="performance_based",
                        performance_delta=performance - self.baseline_performance.get(indicator_name, 0.5),
                        confidence=1.0
                    )
                    
                    self.weight_history[indicator_name].append(update_record)
                    
                    logger.info(f"Updated weight for {indicator_name}: {old_weight:.3f} -> {new_weight:.3f}")
            
            if updates_made:
                self._normalize_weights()
                self.optimization_state['total_updates'] += 1
                self.optimization_state['successful_updates'] += 1
            
            return self.current_weights.copy()
            
        except Exception as e:
            logger.error(f"Error updating weights from performance: {e}")
            self.optimization_state['failed_updates'] += 1
            return self.current_weights.copy()
    
    def update_weights_ml_optimization(self, 
                                     feature_data: pd.DataFrame,
                                     target_performance: pd.Series) -> Dict[str, float]:
        """
        Update weights using ML optimization
        
        Args:
            feature_data: DataFrame with features for ML models
            target_performance: Target performance values
            
        Returns:
            Dict[str, float]: Optimized weights
        """
        try:
            if len(feature_data) < self.config.window_size:
                logger.warning("Insufficient data for ML optimization")
                return self.current_weights.copy()
            
            # Prepare training data
            X = self._prepare_feature_matrix(feature_data)
            y = target_performance.values
            
            # Train or update models
            if not self.models_trained or self.last_ml_update > self.config.ml_update_frequency:
                self._train_ml_models(X, y)
                self.last_ml_update = 0
            
            # Get predictions from both models
            rf_weights = self._predict_optimal_weights_rf(X[-1:])  # Use latest features
            lr_weights = self._predict_optimal_weights_lr(X[-1:])
            
            # Ensemble the predictions
            ensemble_weights = self._ensemble_weight_predictions(rf_weights, lr_weights)
            
            # Apply constraints and normalization
            constrained_weights = self._apply_weight_constraints(ensemble_weights)
            
            # Update current weights
            old_weights = self.current_weights.copy()
            for indicator, weight in constrained_weights.items():
                if indicator in self.current_weights:
                    self.current_weights[indicator] = weight
            
            self._normalize_weights()
            self.last_ml_update += 1
            
            # Log significant changes
            for indicator in self.current_weights:
                old_w = old_weights.get(indicator, 0)
                new_w = self.current_weights[indicator]
                if abs(new_w - old_w) > 0.1:
                    logger.info(f"ML optimization: {indicator} weight {old_w:.3f} -> {new_w:.3f}")
            
            return self.current_weights.copy()
            
        except Exception as e:
            logger.error(f"Error in ML optimization: {e}")
            return self.current_weights.copy()
    
    def optimize_weights_bayesian(self, 
                                objective_function: callable,
                                bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, float]:
        """
        Optimize weights using Bayesian optimization
        
        Args:
            objective_function: Function to optimize (higher is better)
            bounds: Weight bounds for each indicator
            
        Returns:
            Dict[str, float]: Optimized weights
        """
        try:
            indicators = list(self.current_weights.keys())
            
            # Set default bounds
            if bounds is None:
                bounds = {ind: (self.config.min_weight, self.config.max_weight) for ind in indicators}
            
            # Define optimization function
            def objective(weights_array):
                weights_dict = dict(zip(indicators, weights_array))
                # Normalize weights
                total = sum(weights_dict.values())
                if total > 0:
                    weights_dict = {k: v/total for k, v in weights_dict.items()}
                
                return -objective_function(weights_dict)  # Minimize negative
            
            # Set up constraints (weights sum to 1)
            constraints = {
                'type': 'eq',
                'fun': lambda x: np.sum(x) - 1.0
            }
            
            # Set up bounds
            bounds_list = [bounds.get(ind, (self.config.min_weight, self.config.max_weight)) 
                          for ind in indicators]
            
            # Initial guess (current weights)
            x0 = np.array([self.current_weights[ind] for ind in indicators])
            
            # Optimize
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds_list,
                constraints=constraints,
                options={'maxiter': 100}
            )
            
            if result.success:
                # Update weights
                optimized_weights = dict(zip(indicators, result.x))
                
                for indicator, weight in optimized_weights.items():
                    self.current_weights[indicator] = weight
                
                self._normalize_weights()
                
                logger.info(f"Bayesian optimization successful. Objective value: {-result.fun:.4f}")
                self.optimization_state['last_optimization'] = datetime.now()
                
            else:
                logger.warning(f"Bayesian optimization failed: {result.message}")
            
            return self.current_weights.copy()
            
        except Exception as e:
            logger.error(f"Error in Bayesian optimization: {e}")
            return self.current_weights.copy()
    
    def get_weight_recommendations(self, 
                                 market_context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Get weight recommendations based on current context
        
        Args:
            market_context: Current market conditions
            
        Returns:
            Dict with recommendations for each indicator
        """
        recommendations = {}
        
        try:
            for indicator in self.current_weights:
                current_weight = self.current_weights[indicator]
                
                # Get performance metrics
                if self.performance_tracker:
                    metrics = self.performance_tracker.calculate_performance_metrics(indicator)
                else:
                    metrics = None
                
                # Calculate recommended weight
                if metrics and metrics.sample_size > 10:
                    # Performance-based recommendation
                    if metrics.accuracy > 0.7:
                        recommended_weight = min(current_weight * 1.2, self.config.max_weight)
                        reason = "High accuracy"
                        confidence = metrics.statistical_significance
                    elif metrics.accuracy < 0.4:
                        recommended_weight = max(current_weight * 0.8, self.config.min_weight)
                        reason = "Low accuracy"
                        confidence = metrics.statistical_significance
                    else:
                        recommended_weight = current_weight
                        reason = "Stable performance"
                        confidence = 0.5
                else:
                    recommended_weight = current_weight
                    reason = "Insufficient data"
                    confidence = 0.1
                
                recommendations[indicator] = {
                    'current_weight': current_weight,
                    'recommended_weight': recommended_weight,
                    'reason': reason,
                    'confidence': confidence,
                    'weight_change': recommended_weight - current_weight,
                    'performance_metrics': metrics.__dict__ if metrics else None
                }
        
        except Exception as e:
            logger.error(f"Error generating weight recommendations: {e}")
        
        return recommendations
    
    def _calculate_performance_based_weight(self, 
                                          indicator_name: str,
                                          current_performance: float,
                                          market_conditions: Optional[Dict[str, Any]] = None) -> float:
        """Calculate new weight based on performance"""
        current_weight = self.current_weights[indicator_name]
        
        # Get baseline performance
        baseline = self.baseline_performance.get(indicator_name, 0.5)
        
        # Calculate performance ratio
        performance_ratio = current_performance / max(baseline, 0.1)
        
        # Apply exponential adjustment
        adjustment_factor = self.config.learning_rate * (performance_ratio - 1.0)
        
        # Apply decay to prevent over-adjustment
        adjustment_factor *= self.config.decay_factor
        
        # Calculate new weight
        new_weight = current_weight * (1.0 + adjustment_factor)
        
        # Apply bounds
        new_weight = np.clip(new_weight, self.config.min_weight, self.config.max_weight)
        
        return new_weight
    
    def _prepare_feature_matrix(self, feature_data: pd.DataFrame) -> np.ndarray:
        """Prepare feature matrix for ML models"""
        # Select relevant features
        feature_columns = [col for col in feature_data.columns 
                          if col not in ['timestamp', 'indicator_name']]
        
        X = feature_data[feature_columns].values
        
        # Scale features
        if not hasattr(self.scaler, 'scale_') or self.scaler.scale_ is None:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        self.feature_names = feature_columns
        return X_scaled
    
    def _train_ml_models(self, X: np.ndarray, y: np.ndarray):
        """Train ML models"""
        try:
            # Train Random Forest
            self.rf_model.fit(X, y)
            
            # Train Linear Regression
            self.lr_model.fit(X, y)
            
            self.models_trained = True
            
            # Log training results
            rf_score = self.rf_model.score(X, y)
            lr_score = self.lr_model.score(X, y)
            
            logger.info(f"ML models trained - RF R²: {rf_score:.3f}, LR R²: {lr_score:.3f}")
            
        except Exception as e:
            logger.error(f"Error training ML models: {e}")
            self.models_trained = False
    
    def _predict_optimal_weights_rf(self, X: np.ndarray) -> Dict[str, float]:
        """Predict optimal weights using Random Forest"""
        if not self.models_trained:
            return self.current_weights.copy()
        
        try:
            prediction = self.rf_model.predict(X)[0]
            
            # Convert prediction to weight distribution
            # This is a simplified approach - in practice, you'd train on actual weight distributions
            indicators = list(self.current_weights.keys())
            n_indicators = len(indicators)
            
            # Use feature importance to weight indicators
            if hasattr(self.rf_model, 'feature_importances_'):
                importances = self.rf_model.feature_importances_
                # Map importances to indicators (simplified)
                weights = importances[:n_indicators] if len(importances) >= n_indicators else [1.0] * n_indicators
            else:
                weights = [1.0] * n_indicators
            
            return dict(zip(indicators, weights))
            
        except Exception as e:
            logger.error(f"Error predicting with RF: {e}")
            return self.current_weights.copy()
    
    def _predict_optimal_weights_lr(self, X: np.ndarray) -> Dict[str, float]:
        """Predict optimal weights using Linear Regression"""
        if not self.models_trained:
            return self.current_weights.copy()
        
        try:
            prediction = self.lr_model.predict(X)[0]
            
            # Similar approach as RF
            indicators = list(self.current_weights.keys())
            n_indicators = len(indicators)
            
            # Use coefficients to weight indicators
            if hasattr(self.lr_model, 'coef_'):
                coeffs = np.abs(self.lr_model.coef_)  # Use absolute values
                weights = coeffs[:n_indicators] if len(coeffs) >= n_indicators else [1.0] * n_indicators
            else:
                weights = [1.0] * n_indicators
            
            return dict(zip(indicators, weights))
            
        except Exception as e:
            logger.error(f"Error predicting with LR: {e}")
            return self.current_weights.copy()
    
    def _ensemble_weight_predictions(self, 
                                   rf_weights: Dict[str, float],
                                   lr_weights: Dict[str, float]) -> Dict[str, float]:
        """Ensemble weight predictions from multiple models"""
        ensemble_weights = {}
        
        for indicator in self.current_weights:
            rf_weight = rf_weights.get(indicator, self.current_weights[indicator])
            lr_weight = lr_weights.get(indicator, self.current_weights[indicator])
            
            # Simple average ensemble
            ensemble_weights[indicator] = 0.6 * rf_weight + 0.4 * lr_weight
        
        return ensemble_weights
    
    def _apply_weight_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply weight constraints"""
        constrained_weights = {}
        
        for indicator, weight in weights.items():
            constrained_weights[indicator] = np.clip(
                weight, self.config.min_weight, self.config.max_weight
            )
        
        return constrained_weights
    
    def _normalize_weights(self):
        """Normalize weights to sum to 1.0"""
        total_weight = sum(self.current_weights.values())
        
        if total_weight > 0:
            for indicator in self.current_weights:
                self.current_weights[indicator] /= total_weight
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current weights"""
        return self.current_weights.copy()
    
    def get_weight_history(self, indicator_name: str) -> List[WeightUpdateRecord]:
        """Get weight history for an indicator"""
        return self.weight_history.get(indicator_name, []).copy()
    
    def reset_weights(self, equal_weights: bool = True):
        """Reset all weights"""
        if equal_weights:
            n_indicators = len(self.current_weights)
            weight = 1.0 / n_indicators if n_indicators > 0 else 0.0
            
            for indicator in self.current_weights:
                self.current_weights[indicator] = weight
        
        # Clear history
        for indicator in self.weight_history:
            self.weight_history[indicator].clear()
            self.performance_history[indicator].clear()
        
        logger.info("Weights reset")
    
    def export_optimization_state(self) -> Dict[str, Any]:
        """Export current optimization state"""
        return {
            'current_weights': self.current_weights.copy(),
            'optimization_state': self.optimization_state.copy(),
            'models_trained': self.models_trained,
            'config': {
                'learning_rate': self.config.learning_rate,
                'decay_factor': self.config.decay_factor,
                'min_weight': self.config.min_weight,
                'max_weight': self.config.max_weight
            },
            'timestamp': datetime.now().isoformat()
        }