"""
Advanced Dynamic Weighting Engine with ML-based Optimization

This module implements machine learning-based weight optimization for the
Triple Rolling Straddle Market Regime system, adapting weights based on
market regime performance and historical accuracy with DTE-based analysis.

Features:
1. ML-based weight optimization using historical performance
2. DTE (Days to Expiry) based weight adjustment
3. Market regime performance tracking
4. Adaptive weight optimization algorithms
5. Historical accuracy analysis
6. Real-time weight adjustment
7. Performance-based weight evolution
8. Regime-specific weight optimization

Author: The Augster
Date: 2025-06-18
Version: 1.0.0
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import json

logger = logging.getLogger(__name__)

@dataclass
class WeightOptimizationResult:
    """Weight optimization result"""
    optimized_weights: Dict[str, float]
    performance_improvement: float
    confidence: float
    optimization_method: str
    historical_accuracy: float
    dte_factor: float
    regime_performance: Dict[str, float]
    optimization_time: float

@dataclass
class PerformanceMetrics:
    """Performance metrics for weight optimization"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    regime_consistency: float
    prediction_confidence: float
    dte_effectiveness: float

class AdvancedDynamicWeightingEngine:
    """
    Advanced Dynamic Weighting Engine with ML-based Optimization
    
    Implements sophisticated weight optimization using machine learning
    algorithms to adapt weights based on historical performance and DTE analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Advanced Dynamic Weighting Engine
        
        Args:
            config (Dict, optional): Configuration parameters
        """
        self.config = config or self._get_default_config()
        
        # Base weight configuration
        self.base_weights = {
            'triple_straddle': 0.35,
            'regime_components': 0.65
        }
        
        # Component weight breakdown
        self.component_weights = {
            'atm_straddle': 0.50,  # 50% of triple_straddle
            'itm1_straddle': 0.30,  # 30% of triple_straddle
            'otm1_straddle': 0.20,  # 20% of triple_straddle
            'volatility_analysis': 0.40,  # 40% of regime_components
            'directional_analysis': 0.35,  # 35% of regime_components
            'correlation_analysis': 0.25   # 25% of regime_components
        }
        
        # DTE-based weight adjustments
        self.dte_weight_factors = {
            'very_short': {'min_dte': 0, 'max_dte': 7, 'factor': 1.2},    # 0-7 DTE: Higher weight
            'short': {'min_dte': 8, 'max_dte': 21, 'factor': 1.0},       # 8-21 DTE: Normal weight
            'medium': {'min_dte': 22, 'max_dte': 45, 'factor': 0.9},     # 22-45 DTE: Slightly lower
            'long': {'min_dte': 46, 'max_dte': 90, 'factor': 0.8},       # 46-90 DTE: Lower weight
            'very_long': {'min_dte': 91, 'max_dte': 365, 'factor': 0.7}  # 91+ DTE: Lowest weight
        }
        
        # ML models for optimization
        self.ml_models = {
            'weight_optimizer': RandomForestRegressor(n_estimators=100, random_state=42),
            'performance_predictor': LinearRegression(),
            'regime_classifier': RandomForestRegressor(n_estimators=50, random_state=42)
        }
        
        # Historical performance tracking
        self.performance_history = {
            'regime_accuracy': [],
            'weight_performance': [],
            'dte_effectiveness': [],
            'optimization_results': []
        }
        
        # Optimization parameters
        self.optimization_params = {
            'learning_rate': 0.01,
            'momentum': 0.9,
            'min_samples_for_optimization': 50,
            'max_weight_change': 0.1,  # Maximum 10% weight change per optimization
            'optimization_frequency': 100  # Optimize every 100 predictions
        }
        
        # Feature scalers
        self.scalers = {
            'features': StandardScaler(),
            'targets': StandardScaler()
        }
        
        logger.info("✅ Advanced Dynamic Weighting Engine initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for dynamic weighting"""
        return {
            'optimization': {
                'enable_ml_optimization': True,
                'enable_dte_adjustment': True,
                'enable_regime_adaptation': True,
                'optimization_interval': 100,
                'min_history_size': 50
            },
            'ml_config': {
                'model_type': 'random_forest',
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'random_state': 42
            },
            'weight_constraints': {
                'min_triple_straddle_weight': 0.20,
                'max_triple_straddle_weight': 0.50,
                'min_component_weight': 0.05,
                'max_component_weight': 0.80,
                'weight_sum_tolerance': 0.01
            },
            'performance_targets': {
                'min_accuracy': 0.75,
                'min_confidence': 0.70,
                'target_improvement': 0.05,
                'optimization_threshold': 0.02
            }
        }
    
    def optimize_weights_ml_based(self, historical_data: List[Dict[str, Any]], 
                                 current_dte: int = 30) -> WeightOptimizationResult:
        """
        Optimize weights using ML-based analysis of historical performance
        
        Args:
            historical_data (List[Dict]): Historical performance data
            current_dte (int): Current days to expiry
            
        Returns:
            WeightOptimizationResult: ML-optimized weights with performance metrics
        """
        try:
            start_time = time.time()
            
            # Validate input data
            if len(historical_data) < self.config['optimization']['min_history_size']:
                logger.warning(f"Insufficient historical data: {len(historical_data)} < {self.config['optimization']['min_history_size']}")
                return self._get_fallback_optimization_result(current_dte)
            
            # Prepare features and targets
            features, targets = self._prepare_ml_features(historical_data)
            
            if features.empty or targets.empty:
                logger.warning("Failed to prepare ML features")
                return self._get_fallback_optimization_result(current_dte)
            
            # Train ML models
            ml_performance = self._train_ml_models(features, targets)
            
            # Generate optimized weights
            optimized_weights = self._generate_ml_optimized_weights(features, current_dte)
            
            # Apply DTE-based adjustments
            dte_adjusted_weights = self._apply_dte_adjustments(optimized_weights, current_dte)
            
            # Validate and normalize weights
            final_weights = self._validate_and_normalize_weights(dte_adjusted_weights)
            
            # Calculate performance metrics
            performance_improvement = self._calculate_performance_improvement(
                final_weights, historical_data
            )
            
            # Calculate historical accuracy
            historical_accuracy = self._calculate_historical_accuracy(historical_data)
            
            # Calculate DTE factor
            dte_factor = self._calculate_dte_factor(current_dte)
            
            # Calculate regime performance
            regime_performance = self._calculate_regime_performance(historical_data)
            
            optimization_time = time.time() - start_time
            
            # Create optimization result
            result = WeightOptimizationResult(
                optimized_weights=final_weights,
                performance_improvement=performance_improvement,
                confidence=ml_performance.get('confidence', 0.7),
                optimization_method="ML_BASED_WITH_DTE",
                historical_accuracy=historical_accuracy,
                dte_factor=dte_factor,
                regime_performance=regime_performance,
                optimization_time=optimization_time
            )
            
            # Update performance history
            self._update_performance_history(result, historical_data)
            
            logger.info(f"ML weight optimization: improvement={performance_improvement:.3f}, time={optimization_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in ML-based weight optimization: {e}")
            return self._get_fallback_optimization_result(current_dte)
    
    def _prepare_ml_features(self, historical_data: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare features and targets for ML training"""
        try:
            features_list = []
            targets_list = []
            
            for data_point in historical_data:
                # Extract features
                features = {
                    'regime_confidence': data_point.get('regime_confidence', 0.5),
                    'triple_straddle_score': data_point.get('triple_straddle_score', 0.5),
                    'volatility_level': data_point.get('volatility_level', 0.5),
                    'correlation_strength': data_point.get('correlation_strength', 0.5),
                    'market_momentum': data_point.get('market_momentum', 0.5),
                    'dte': data_point.get('dte', 30),
                    'iv_percentile': data_point.get('iv_percentile', 0.5),
                    'volume_profile': data_point.get('volume_profile', 0.5),
                    'time_of_day': data_point.get('timestamp', datetime.now()).hour,
                    'day_of_week': data_point.get('timestamp', datetime.now()).weekday()
                }
                
                # Extract targets (performance metrics)
                targets = {
                    'accuracy': data_point.get('accuracy', 0.5),
                    'regime_consistency': data_point.get('regime_consistency', 0.5),
                    'prediction_confidence': data_point.get('prediction_confidence', 0.5)
                }
                
                features_list.append(features)
                targets_list.append(targets)
            
            features_df = pd.DataFrame(features_list)
            targets_df = pd.DataFrame(targets_list)
            
            # Handle missing values
            features_df.fillna(features_df.mean(), inplace=True)
            targets_df.fillna(targets_df.mean(), inplace=True)
            
            return features_df, targets_df
            
        except Exception as e:
            logger.error(f"Error preparing ML features: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def _train_ml_models(self, features: pd.DataFrame, targets: pd.DataFrame) -> Dict[str, Any]:
        """Train ML models for weight optimization"""
        try:
            # Scale features
            features_scaled = self.scalers['features'].fit_transform(features)
            
            # Train weight optimizer
            weight_optimizer_target = targets['accuracy'].values
            self.ml_models['weight_optimizer'].fit(features_scaled, weight_optimizer_target)
            
            # Train performance predictor
            performance_target = targets['regime_consistency'].values
            self.ml_models['performance_predictor'].fit(features_scaled, performance_target)
            
            # Calculate model performance
            weight_pred = self.ml_models['weight_optimizer'].predict(features_scaled)
            performance_pred = self.ml_models['performance_predictor'].predict(features_scaled)
            
            weight_r2 = r2_score(weight_optimizer_target, weight_pred)
            performance_r2 = r2_score(performance_target, performance_pred)
            
            ml_performance = {
                'weight_optimizer_r2': weight_r2,
                'performance_predictor_r2': performance_r2,
                'confidence': (weight_r2 + performance_r2) / 2,
                'models_trained': True
            }
            
            logger.debug(f"ML models trained: weight_r2={weight_r2:.3f}, performance_r2={performance_r2:.3f}")
            
            return ml_performance
            
        except Exception as e:
            logger.error(f"Error training ML models: {e}")
            return {'confidence': 0.5, 'models_trained': False}
    
    def _generate_ml_optimized_weights(self, features: pd.DataFrame, current_dte: int) -> Dict[str, float]:
        """Generate ML-optimized weights"""
        try:
            # Create current feature vector
            current_features = {
                'regime_confidence': features['regime_confidence'].mean(),
                'triple_straddle_score': features['triple_straddle_score'].mean(),
                'volatility_level': features['volatility_level'].mean(),
                'correlation_strength': features['correlation_strength'].mean(),
                'market_momentum': features['market_momentum'].mean(),
                'dte': current_dte,
                'iv_percentile': features['iv_percentile'].mean(),
                'volume_profile': features['volume_profile'].mean(),
                'time_of_day': datetime.now().hour,
                'day_of_week': datetime.now().weekday()
            }
            
            current_features_df = pd.DataFrame([current_features])
            current_features_scaled = self.scalers['features'].transform(current_features_df)
            
            # Predict optimal performance
            predicted_performance = self.ml_models['weight_optimizer'].predict(current_features_scaled)[0]
            
            # Generate optimized weights based on prediction
            base_triple_straddle_weight = self.base_weights['triple_straddle']
            
            # Adjust based on predicted performance
            performance_factor = np.clip(predicted_performance, 0.5, 1.5)
            optimized_triple_straddle = base_triple_straddle_weight * performance_factor
            
            # Apply constraints
            optimized_triple_straddle = np.clip(
                optimized_triple_straddle,
                self.config['weight_constraints']['min_triple_straddle_weight'],
                self.config['weight_constraints']['max_triple_straddle_weight']
            )
            
            optimized_weights = {
                'triple_straddle': optimized_triple_straddle,
                'regime_components': 1.0 - optimized_triple_straddle
            }
            
            return optimized_weights
            
        except Exception as e:
            logger.error(f"Error generating ML-optimized weights: {e}")
            return self.base_weights.copy()
    
    def _apply_dte_adjustments(self, weights: Dict[str, float], current_dte: int) -> Dict[str, float]:
        """Apply DTE-based weight adjustments"""
        try:
            # Determine DTE category
            dte_category = self._get_dte_category(current_dte)
            dte_factor = self.dte_weight_factors[dte_category]['factor']
            
            # Apply DTE adjustment to triple straddle weight
            adjusted_weights = weights.copy()
            adjusted_weights['triple_straddle'] *= dte_factor
            
            # Ensure weights sum to 1.0
            total_weight = sum(adjusted_weights.values())
            if total_weight > 0:
                for key in adjusted_weights:
                    adjusted_weights[key] /= total_weight
            
            logger.debug(f"DTE adjustment: category={dte_category}, factor={dte_factor:.3f}")
            
            return adjusted_weights
            
        except Exception as e:
            logger.error(f"Error applying DTE adjustments: {e}")
            return weights
    
    def _get_dte_category(self, dte: int) -> str:
        """Get DTE category for weight adjustment"""
        for category, params in self.dte_weight_factors.items():
            if params['min_dte'] <= dte <= params['max_dte']:
                return category
        return 'medium'  # Default category

    def _validate_and_normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Validate and normalize weights to ensure they sum to 1.0"""
        try:
            # Apply constraints
            constrained_weights = {}

            for key, value in weights.items():
                if key == 'triple_straddle':
                    constrained_weights[key] = np.clip(
                        value,
                        self.config['weight_constraints']['min_triple_straddle_weight'],
                        self.config['weight_constraints']['max_triple_straddle_weight']
                    )
                else:
                    constrained_weights[key] = np.clip(
                        value,
                        self.config['weight_constraints']['min_component_weight'],
                        self.config['weight_constraints']['max_component_weight']
                    )

            # Normalize to sum to 1.0
            total_weight = sum(constrained_weights.values())
            if total_weight > 0:
                for key in constrained_weights:
                    constrained_weights[key] /= total_weight
            else:
                # Fallback to base weights
                constrained_weights = self.base_weights.copy()

            # Validate sum
            weight_sum = sum(constrained_weights.values())
            if abs(weight_sum - 1.0) > self.config['weight_constraints']['weight_sum_tolerance']:
                logger.warning(f"Weight sum validation failed: {weight_sum:.3f}")
                return self.base_weights.copy()

            return constrained_weights

        except Exception as e:
            logger.error(f"Error validating and normalizing weights: {e}")
            return self.base_weights.copy()

    def _calculate_performance_improvement(self, optimized_weights: Dict[str, float],
                                         historical_data: List[Dict[str, Any]]) -> float:
        """Calculate expected performance improvement from weight optimization"""
        try:
            if not historical_data:
                return 0.0

            # Calculate baseline performance with current weights
            baseline_performance = np.mean([
                data.get('accuracy', 0.5) for data in historical_data[-10:]
            ])

            # Estimate performance with optimized weights
            weight_improvement_factor = optimized_weights['triple_straddle'] / self.base_weights['triple_straddle']
            estimated_performance = baseline_performance * weight_improvement_factor

            # Calculate improvement
            improvement = estimated_performance - baseline_performance

            return np.clip(improvement, -0.5, 0.5)  # Limit improvement range

        except Exception as e:
            logger.error(f"Error calculating performance improvement: {e}")
            return 0.0

    def _calculate_historical_accuracy(self, historical_data: List[Dict[str, Any]]) -> float:
        """Calculate historical accuracy from performance data"""
        try:
            if not historical_data:
                return 0.5

            accuracies = [data.get('accuracy', 0.5) for data in historical_data]
            return np.mean(accuracies)

        except Exception as e:
            logger.error(f"Error calculating historical accuracy: {e}")
            return 0.5

    def _calculate_dte_factor(self, current_dte: int) -> float:
        """Calculate DTE factor for current expiry"""
        try:
            dte_category = self._get_dte_category(current_dte)
            return self.dte_weight_factors[dte_category]['factor']

        except Exception as e:
            logger.error(f"Error calculating DTE factor: {e}")
            return 1.0

    def _calculate_regime_performance(self, historical_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate regime-specific performance metrics"""
        try:
            regime_performance = {}

            # Group data by regime
            regime_groups = {}
            for data in historical_data:
                regime = data.get('regime_id', 'UNKNOWN')
                if regime not in regime_groups:
                    regime_groups[regime] = []
                regime_groups[regime].append(data)

            # Calculate performance for each regime
            for regime, regime_data in regime_groups.items():
                if regime_data:
                    regime_accuracy = np.mean([d.get('accuracy', 0.5) for d in regime_data])
                    regime_performance[regime] = regime_accuracy
                else:
                    regime_performance[regime] = 0.5

            return regime_performance

        except Exception as e:
            logger.error(f"Error calculating regime performance: {e}")
            return {}

    def _update_performance_history(self, optimization_result: WeightOptimizationResult,
                                  historical_data: List[Dict[str, Any]]):
        """Update performance history with optimization results"""
        try:
            # Update regime accuracy history
            self.performance_history['regime_accuracy'].append(optimization_result.historical_accuracy)

            # Update weight performance history
            weight_performance = {
                'weights': optimization_result.optimized_weights,
                'improvement': optimization_result.performance_improvement,
                'timestamp': datetime.now()
            }
            self.performance_history['weight_performance'].append(weight_performance)

            # Update DTE effectiveness
            dte_effectiveness = {
                'dte_factor': optimization_result.dte_factor,
                'performance': optimization_result.historical_accuracy,
                'timestamp': datetime.now()
            }
            self.performance_history['dte_effectiveness'].append(dte_effectiveness)

            # Update optimization results
            self.performance_history['optimization_results'].append({
                'method': optimization_result.optimization_method,
                'improvement': optimization_result.performance_improvement,
                'confidence': optimization_result.confidence,
                'timestamp': datetime.now()
            })

            # Keep only last 1000 entries
            for history_key in self.performance_history:
                if len(self.performance_history[history_key]) > 1000:
                    self.performance_history[history_key] = self.performance_history[history_key][-1000:]

        except Exception as e:
            logger.error(f"Error updating performance history: {e}")

    def _get_fallback_optimization_result(self, current_dte: int) -> WeightOptimizationResult:
        """Get fallback optimization result when ML optimization fails"""
        # Apply simple DTE-based adjustment to base weights
        dte_adjusted_weights = self._apply_dte_adjustments(self.base_weights.copy(), current_dte)

        return WeightOptimizationResult(
            optimized_weights=dte_adjusted_weights,
            performance_improvement=0.0,
            confidence=0.5,
            optimization_method="FALLBACK_DTE_BASED",
            historical_accuracy=0.5,
            dte_factor=self._calculate_dte_factor(current_dte),
            regime_performance={},
            optimization_time=0.001
        )

    def get_current_optimized_weights(self, market_data: Dict[str, Any],
                                    historical_performance: List[Dict[str, Any]]) -> Dict[str, float]:
        """Get current optimized weights for market data"""
        try:
            current_dte = market_data.get('dte', 30)

            # Perform ML-based optimization
            optimization_result = self.optimize_weights_ml_based(historical_performance, current_dte)

            return optimization_result.optimized_weights

        except Exception as e:
            logger.error(f"Error getting current optimized weights: {e}")
            return self.base_weights.copy()

    def validate_weight_optimization_performance(self, test_scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate weight optimization performance"""
        try:
            validation_start = time.time()

            optimization_results = []
            performance_improvements = []

            for i, scenario in enumerate(test_scenarios):
                scenario_start = time.time()

                # Generate historical data for scenario
                historical_data = self._generate_test_historical_data(scenario)

                # Perform optimization
                optimization_result = self.optimize_weights_ml_based(
                    historical_data,
                    scenario.get('dte', 30)
                )

                scenario_time = time.time() - scenario_start

                optimization_results.append({
                    'scenario_id': i,
                    'optimized_weights': optimization_result.optimized_weights,
                    'performance_improvement': optimization_result.performance_improvement,
                    'confidence': optimization_result.confidence,
                    'optimization_time': scenario_time,
                    'method': optimization_result.optimization_method
                })

                performance_improvements.append(optimization_result.performance_improvement)

            total_validation_time = time.time() - validation_start

            # Calculate validation metrics
            avg_improvement = np.mean(performance_improvements)
            avg_confidence = np.mean([r['confidence'] for r in optimization_results])
            avg_optimization_time = np.mean([r['optimization_time'] for r in optimization_results])

            # Assess optimization effectiveness
            optimization_effective = (
                avg_improvement >= self.config['performance_targets']['target_improvement'] and
                avg_confidence >= self.config['performance_targets']['min_confidence']
            )

            validation_result = {
                'optimization_effective': optimization_effective,
                'total_scenarios': len(test_scenarios),
                'avg_performance_improvement': avg_improvement,
                'avg_confidence': avg_confidence,
                'avg_optimization_time': avg_optimization_time,
                'total_validation_time': total_validation_time,
                'optimization_results': optimization_results,
                'performance_targets_met': {
                    'improvement_target': avg_improvement >= self.config['performance_targets']['target_improvement'],
                    'confidence_target': avg_confidence >= self.config['performance_targets']['min_confidence'],
                    'time_efficiency': avg_optimization_time < 1.0  # 1 second target
                }
            }

            logger.info(f"Weight optimization validation: improvement={avg_improvement:.3f}, confidence={avg_confidence:.3f}")

            return validation_result

        except Exception as e:
            logger.error(f"Error validating weight optimization performance: {e}")
            return {
                'optimization_effective': False,
                'error': str(e)
            }

    def _generate_test_historical_data(self, scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate test historical data for validation"""
        try:
            historical_data = []

            for i in range(100):  # Generate 100 historical points
                data_point = {
                    'regime_confidence': 0.5 + np.random.normal(0, 0.2),
                    'triple_straddle_score': 0.5 + np.random.normal(0, 0.15),
                    'volatility_level': scenario.get('volatility_level', 0.5) + np.random.normal(0, 0.1),
                    'correlation_strength': 0.5 + np.random.normal(0, 0.1),
                    'market_momentum': 0.5 + np.random.normal(0, 0.1),
                    'dte': scenario.get('dte', 30) + np.random.randint(-5, 5),
                    'iv_percentile': scenario.get('iv_percentile', 0.5) + np.random.normal(0, 0.1),
                    'volume_profile': 0.5 + np.random.normal(0, 0.1),
                    'accuracy': 0.7 + np.random.normal(0, 0.1),
                    'regime_consistency': 0.6 + np.random.normal(0, 0.1),
                    'prediction_confidence': 0.65 + np.random.normal(0, 0.1),
                    'regime_id': f"REGIME_{np.random.randint(1, 13)}",
                    'timestamp': datetime.now() - timedelta(hours=i)
                }

                # Clip values to valid ranges
                for key, value in data_point.items():
                    if isinstance(value, (int, float)) and key != 'dte':
                        data_point[key] = np.clip(value, 0.0, 1.0)

                historical_data.append(data_point)

            return historical_data

        except Exception as e:
            logger.error(f"Error generating test historical data: {e}")
            return []

    def get_optimization_performance_summary(self) -> Dict[str, Any]:
        """Get optimization performance summary"""
        try:
            if not self.performance_history['optimization_results']:
                return {'status': 'No optimization data available'}

            optimization_results = self.performance_history['optimization_results']
            weight_performance = self.performance_history['weight_performance']

            # Calculate summary metrics
            avg_improvement = np.mean([r['improvement'] for r in optimization_results])
            avg_confidence = np.mean([r['confidence'] for r in optimization_results])

            # Method distribution
            methods = [r['method'] for r in optimization_results]
            method_counts = {method: methods.count(method) for method in set(methods)}

            # Recent performance trend
            recent_improvements = [r['improvement'] for r in optimization_results[-20:]]
            performance_trend = np.mean(recent_improvements) if recent_improvements else 0.0

            return {
                'optimization_summary': {
                    'total_optimizations': len(optimization_results),
                    'avg_improvement': avg_improvement,
                    'avg_confidence': avg_confidence,
                    'performance_trend': performance_trend,
                    'method_distribution': method_counts
                },
                'weight_evolution': {
                    'total_weight_updates': len(weight_performance),
                    'current_weights': weight_performance[-1]['weights'] if weight_performance else self.base_weights,
                    'base_weights': self.base_weights
                },
                'performance_assessment': {
                    'optimization_grade': self._grade_optimization_performance(avg_improvement, avg_confidence),
                    'trend_direction': 'IMPROVING' if performance_trend > 0 else 'DECLINING' if performance_trend < 0 else 'STABLE',
                    'ml_effectiveness': avg_confidence
                }
            }

        except Exception as e:
            logger.error(f"Error getting optimization performance summary: {e}")
            return {'status': 'Error calculating optimization summary'}

    def _grade_optimization_performance(self, avg_improvement: float, avg_confidence: float) -> str:
        """Grade optimization performance"""
        try:
            # Combined score
            combined_score = (avg_improvement * 0.6 + avg_confidence * 0.4)

            if combined_score >= 0.8:
                return "EXCELLENT"
            elif combined_score >= 0.6:
                return "VERY_GOOD"
            elif combined_score >= 0.4:
                return "GOOD"
            elif combined_score >= 0.2:
                return "ACCEPTABLE"
            else:
                return "NEEDS_IMPROVEMENT"

        except Exception as e:
            logger.error(f"Error grading optimization performance: {e}")
            return "UNKNOWN"
