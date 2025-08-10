"""
Historical Optimizer - Adaptive Historical Performance Optimization
================================================================

Optimizes market regime detection parameters based on historical performance.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from scipy.optimize import minimize, differential_evolution
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Import base utilities
from ...base.common_utils import MathUtils, TimeUtils, ErrorHandler, CacheUtils

logger = logging.getLogger(__name__)


class HistoricalOptimizer:
    """Historical performance optimization for market regime parameters"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Historical Optimizer"""
        self.optimization_window = config.get('optimization_window', 252)  # 1 year
        self.lookback_periods = config.get('lookback_periods', [30, 60, 90, 180])
        self.optimization_method = config.get('optimization_method', 'differential_evolution')
        self.performance_metrics = config.get('performance_metrics', ['sharpe_ratio', 'sortino_ratio', 'max_drawdown'])
        
        # Optimization constraints
        self.parameter_bounds = config.get('parameter_bounds', {
            'volatility_threshold': (0.1, 0.8),
            'trend_threshold': (0.2, 0.9), 
            'volume_threshold': (0.1, 0.7),
            'momentum_threshold': (0.15, 0.85)
        })
        
        # Performance tracking
        self.optimization_history = {
            'parameters': [],
            'performance_scores': [],
            'optimization_dates': [],
            'convergence_info': []
        }
        
        # Caching for expensive calculations
        self.cache = CacheUtils(max_size=1000)
        
        # Mathematical utilities
        self.math_utils = MathUtils()
        self.time_utils = TimeUtils()
        
        logger.info("HistoricalOptimizer initialized with advanced optimization algorithms")
    
    def optimize_parameters(self, 
                          historical_data: pd.DataFrame,
                          current_parameters: Dict[str, float],
                          target_metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Optimize parameters based on historical performance
        
        Args:
            historical_data: Historical market data for optimization
            current_parameters: Current parameter configuration
            target_metrics: Optional target performance metrics
            
        Returns:
            Dict with optimized parameters and performance analysis
        """
        try:
            if historical_data.empty:
                return self._get_default_optimization_result(current_parameters)
            
            # Prepare optimization data
            optimization_data = self._prepare_optimization_data(historical_data)
            
            # Define objective function
            objective_function = self._create_objective_function(optimization_data, target_metrics)
            
            # Run optimization
            optimization_result = self._run_optimization(objective_function, current_parameters)
            
            # Validate optimized parameters
            validation_result = self._validate_optimized_parameters(
                optimization_result['optimized_parameters'], historical_data
            )
            
            # Calculate performance improvement
            performance_improvement = self._calculate_performance_improvement(
                current_parameters, optimization_result['optimized_parameters'], historical_data
            )
            
            # Generate optimization report
            optimization_report = self._generate_optimization_report(
                optimization_result, validation_result, performance_improvement
            )
            
            # Update optimization history
            self._update_optimization_history(optimization_result, optimization_report)
            
            return {
                'optimized_parameters': optimization_result['optimized_parameters'],
                'optimization_report': optimization_report,
                'validation_result': validation_result,
                'performance_improvement': performance_improvement,
                'convergence_info': optimization_result['convergence_info']
            }
            
        except Exception as e:
            logger.error(f"Error in parameter optimization: {e}")
            return self._get_default_optimization_result(current_parameters)
    
    def _prepare_optimization_data(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for optimization"""
        try:
            # Calculate returns and volatility
            if 'close' in historical_data.columns:
                returns = historical_data['close'].pct_change().dropna()
            else:
                returns = pd.Series([0.001] * len(historical_data))
            
            # Calculate rolling metrics
            rolling_volatility = returns.rolling(window=20).std() * np.sqrt(252)
            rolling_sharpe = returns.rolling(window=60).mean() / returns.rolling(window=60).std() * np.sqrt(252)
            
            # Volume metrics
            if 'volume' in historical_data.columns:
                volume_ma = historical_data['volume'].rolling(window=20).mean()
                volume_ratio = historical_data['volume'] / volume_ma
            else:
                volume_ratio = pd.Series([1.0] * len(historical_data))
            
            # Price momentum
            if 'close' in historical_data.columns:
                momentum_10 = historical_data['close'].pct_change(10)
                momentum_20 = historical_data['close'].pct_change(20)
            else:
                momentum_10 = pd.Series([0.001] * len(historical_data))
                momentum_20 = pd.Series([0.001] * len(historical_data))
            
            return {
                'returns': returns,
                'rolling_volatility': rolling_volatility,
                'rolling_sharpe': rolling_sharpe,
                'volume_ratio': volume_ratio,
                'momentum_10': momentum_10,
                'momentum_20': momentum_20,
                'price_data': historical_data.get('close', pd.Series([100] * len(historical_data)))
            }
            
        except Exception as e:
            logger.error(f"Error preparing optimization data: {e}")
            return {}
    
    def _create_objective_function(self, 
                                 optimization_data: Dict[str, Any], 
                                 target_metrics: Optional[Dict[str, float]]) -> callable:
        """Create objective function for optimization"""
        def objective(params_array):
            try:
                # Convert parameter array to dict
                param_names = list(self.parameter_bounds.keys())
                parameters = dict(zip(param_names, params_array))
                
                # Generate regime signals based on parameters
                regime_signals = self._generate_regime_signals(optimization_data, parameters)
                
                # Calculate performance metrics
                performance = self._calculate_performance_metrics(
                    optimization_data['returns'], regime_signals
                )
                
                # Calculate objective score
                objective_score = self._calculate_objective_score(performance, target_metrics)
                
                # Return negative score for minimization
                return -objective_score
                
            except Exception as e:
                logger.error(f"Error in objective function: {e}")
                return 1e6  # Large penalty for errors
        
        return objective
    
    def _generate_regime_signals(self, 
                               optimization_data: Dict[str, Any], 
                               parameters: Dict[str, float]) -> pd.Series:
        """Generate regime signals based on parameters"""
        try:
            volatility_threshold = parameters.get('volatility_threshold', 0.3)
            trend_threshold = parameters.get('trend_threshold', 0.5)
            volume_threshold = parameters.get('volume_threshold', 0.4)
            momentum_threshold = parameters.get('momentum_threshold', 0.3)
            
            # Get data
            volatility = optimization_data['rolling_volatility'].fillna(0.2)
            volume_ratio = optimization_data['volume_ratio'].fillna(1.0)
            momentum = optimization_data['momentum_10'].fillna(0.0)
            
            # Generate signals
            regime_signals = pd.Series([0] * len(volatility), index=volatility.index)
            
            # High volatility regime
            high_vol_mask = volatility > volatility.quantile(1 - volatility_threshold)
            regime_signals[high_vol_mask] = 1
            
            # Trending regime
            strong_momentum_mask = abs(momentum) > momentum.quantile(1 - momentum_threshold)
            regime_signals[strong_momentum_mask] = 2
            
            # High volume regime
            high_vol_ratio_mask = volume_ratio > volume_ratio.quantile(1 - volume_threshold)
            regime_signals[high_vol_ratio_mask] = 3
            
            # Trend + volume regime
            trend_volume_mask = strong_momentum_mask & high_vol_ratio_mask
            regime_signals[trend_volume_mask] = 4
            
            return regime_signals
            
        except Exception as e:
            logger.error(f"Error generating regime signals: {e}")
            return pd.Series([0] * len(optimization_data.get('returns', [])))
    
    def _calculate_performance_metrics(self, 
                                     returns: pd.Series, 
                                     regime_signals: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics for regime-based strategy"""
        try:
            # Align data
            aligned_returns = returns.reindex(regime_signals.index).fillna(0)
            
            # Calculate regime-based returns (simple strategy: go long in regime 2 and 4)
            strategy_returns = aligned_returns.copy()
            strategy_returns[regime_signals.isin([0, 1, 3])] *= 0.5  # Reduce exposure in other regimes
            
            # Performance metrics
            total_return = (1 + strategy_returns).prod() - 1
            volatility = strategy_returns.std() * np.sqrt(252)
            sharpe_ratio = self.math_utils.safe_divide(strategy_returns.mean() * 252, volatility, 0)
            
            # Sortino ratio
            downside_returns = strategy_returns[strategy_returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
            sortino_ratio = self.math_utils.safe_divide(strategy_returns.mean() * 252, downside_volatility, 0)
            
            # Maximum drawdown
            cumulative_returns = (1 + strategy_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
            
            # Regime quality metrics
            regime_consistency = self._calculate_regime_consistency(regime_signals)
            
            return {
                'total_return': float(total_return),
                'volatility': float(volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'sortino_ratio': float(sortino_ratio),
                'max_drawdown': float(max_drawdown),
                'regime_consistency': float(regime_consistency)
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {'sharpe_ratio': 0, 'sortino_ratio': 0, 'max_drawdown': 1}
    
    def _calculate_regime_consistency(self, regime_signals: pd.Series) -> float:
        """Calculate regime consistency score"""
        try:
            if len(regime_signals) == 0:
                return 0.0
            
            # Calculate regime persistence
            regime_changes = (regime_signals != regime_signals.shift(1)).sum()
            persistence_score = 1.0 - (regime_changes / len(regime_signals))
            
            # Calculate regime distribution balance
            regime_counts = regime_signals.value_counts()
            regime_distribution = regime_counts / len(regime_signals)
            balance_score = 1.0 - regime_distribution.std()
            
            # Combined consistency score
            consistency = (persistence_score + balance_score) / 2
            return max(min(consistency, 1.0), 0.0)
            
        except Exception as e:
            logger.error(f"Error calculating regime consistency: {e}")
            return 0.0
    
    def _calculate_objective_score(self, 
                                 performance: Dict[str, float], 
                                 target_metrics: Optional[Dict[str, float]]) -> float:
        """Calculate objective score for optimization"""
        try:
            if target_metrics:
                # Score based on target achievement
                score = 0.0
                for metric, target in target_metrics.items():
                    if metric in performance:
                        actual = performance[metric]
                        if metric == 'max_drawdown':
                            # Lower is better for drawdown
                            achievement = max(0, 1 - actual / target) if target > 0 else 0
                        else:
                            # Higher is better for other metrics
                            achievement = min(actual / target, 2.0) if target > 0 else 0
                        score += achievement
                
                return score / len(target_metrics)
            else:
                # Default scoring: weighted combination
                sharpe_weight = 0.4
                sortino_weight = 0.3
                drawdown_weight = 0.2
                consistency_weight = 0.1
                
                sharpe_score = min(performance.get('sharpe_ratio', 0) / 2.0, 1.0)  # Normalize to 2.0 Sharpe
                sortino_score = min(performance.get('sortino_ratio', 0) / 2.5, 1.0)  # Normalize to 2.5 Sortino
                drawdown_score = max(0, 1 - performance.get('max_drawdown', 1) / 0.2)  # Penalize >20% drawdown
                consistency_score = performance.get('regime_consistency', 0)
                
                total_score = (sharpe_weight * sharpe_score + 
                             sortino_weight * sortino_score +
                             drawdown_weight * drawdown_score +
                             consistency_weight * consistency_score)
                
                return total_score
                
        except Exception as e:
            logger.error(f"Error calculating objective score: {e}")
            return 0.0
    
    def _run_optimization(self, 
                        objective_function: callable, 
                        current_parameters: Dict[str, float]) -> Dict[str, Any]:
        """Run parameter optimization"""
        try:
            # Prepare bounds
            param_names = list(self.parameter_bounds.keys())
            bounds = [self.parameter_bounds[name] for name in param_names]
            
            # Initial guess from current parameters
            x0 = [current_parameters.get(name, np.mean(self.parameter_bounds[name])) for name in param_names]
            
            # Run optimization
            if self.optimization_method == 'differential_evolution':
                result = differential_evolution(
                    objective_function,
                    bounds,
                    seed=42,
                    maxiter=100,
                    popsize=15
                )
            else:
                result = minimize(
                    objective_function,
                    x0,
                    bounds=bounds,
                    method='L-BFGS-B'
                )
            
            # Extract optimized parameters
            optimized_params = dict(zip(param_names, result.x))
            
            return {
                'optimized_parameters': optimized_params,
                'convergence_info': {
                    'success': bool(result.success),
                    'function_value': float(-result.fun),  # Convert back from negative
                    'iterations': int(getattr(result, 'nit', 0)),
                    'message': str(result.message)
                }
            }
            
        except Exception as e:
            logger.error(f"Error running optimization: {e}")
            return {
                'optimized_parameters': current_parameters,
                'convergence_info': {'success': False, 'message': str(e)}
            }
    
    def _validate_optimized_parameters(self, 
                                     optimized_parameters: Dict[str, float], 
                                     historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate optimized parameters"""
        try:
            validation = {
                'is_valid': True,
                'validation_errors': [],
                'parameter_analysis': {}
            }
            
            # Check parameter bounds
            for param, value in optimized_parameters.items():
                if param in self.parameter_bounds:
                    min_val, max_val = self.parameter_bounds[param]
                    if not (min_val <= value <= max_val):
                        validation['is_valid'] = False
                        validation['validation_errors'].append(
                            f"Parameter {param} = {value:.4f} is out of bounds [{min_val}, {max_val}]"
                        )
                    
                    # Parameter analysis
                    validation['parameter_analysis'][param] = {
                        'value': float(value),
                        'bounds': [min_val, max_val],
                        'relative_position': (value - min_val) / (max_val - min_val)
                    }
            
            # Check parameter relationships
            vol_threshold = optimized_parameters.get('volatility_threshold', 0.3)
            trend_threshold = optimized_parameters.get('trend_threshold', 0.5)
            
            if vol_threshold > trend_threshold + 0.3:
                validation['validation_errors'].append(
                    "Volatility threshold significantly higher than trend threshold - may cause regime conflicts"
                )
            
            # Stability check
            stability_score = self._check_parameter_stability(optimized_parameters)
            validation['stability_score'] = stability_score
            
            if stability_score < 0.5:
                validation['validation_errors'].append(
                    f"Low parameter stability score: {stability_score:.3f}"
                )
            
            return validation
            
        except Exception as e:
            logger.error(f"Error validating optimized parameters: {e}")
            return {'is_valid': False, 'validation_errors': [str(e)]}
    
    def _check_parameter_stability(self, parameters: Dict[str, float]) -> float:
        """Check stability of optimized parameters"""
        try:
            if len(self.optimization_history['parameters']) < 3:
                return 1.0  # Not enough history
            
            # Compare with recent parameter sets
            recent_params = self.optimization_history['parameters'][-3:]
            stability_scores = []
            
            for past_params in recent_params:
                param_differences = []
                for param, value in parameters.items():
                    if param in past_params:
                        past_value = past_params[param]
                        if param in self.parameter_bounds:
                            param_range = self.parameter_bounds[param][1] - self.parameter_bounds[param][0]
                            normalized_diff = abs(value - past_value) / param_range
                            param_differences.append(normalized_diff)
                
                if param_differences:
                    avg_difference = np.mean(param_differences)
                    stability_scores.append(max(0, 1 - avg_difference))
            
            return np.mean(stability_scores) if stability_scores else 1.0
            
        except Exception as e:
            logger.error(f"Error checking parameter stability: {e}")
            return 0.5
    
    def _calculate_performance_improvement(self, 
                                         current_parameters: Dict[str, float],
                                         optimized_parameters: Dict[str, float],
                                         historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance improvement from optimization"""
        try:
            # Prepare data
            optimization_data = self._prepare_optimization_data(historical_data)
            
            # Current performance
            current_signals = self._generate_regime_signals(optimization_data, current_parameters)
            current_performance = self._calculate_performance_metrics(
                optimization_data['returns'], current_signals
            )
            
            # Optimized performance
            optimized_signals = self._generate_regime_signals(optimization_data, optimized_parameters)
            optimized_performance = self._calculate_performance_metrics(
                optimization_data['returns'], optimized_signals
            )
            
            # Calculate improvements
            improvements = {}
            for metric in current_performance.keys():
                current_val = current_performance[metric]
                optimized_val = optimized_performance[metric]
                
                if metric == 'max_drawdown':
                    # Lower is better for drawdown
                    improvement = ((current_val - optimized_val) / current_val * 100) if current_val != 0 else 0
                else:
                    # Higher is better for other metrics
                    improvement = ((optimized_val - current_val) / current_val * 100) if current_val != 0 else 0
                
                improvements[f"{metric}_improvement"] = float(improvement)
            
            return {
                'current_performance': current_performance,
                'optimized_performance': optimized_performance,
                'improvements': improvements,
                'overall_improvement_score': np.mean(list(improvements.values()))
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance improvement: {e}")
            return {'overall_improvement_score': 0.0}
    
    def _generate_optimization_report(self, 
                                    optimization_result: Dict[str, Any],
                                    validation_result: Dict[str, Any],
                                    performance_improvement: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        try:
            return {
                'optimization_timestamp': datetime.now(),
                'optimization_summary': {
                    'method': self.optimization_method,
                    'convergence_success': optimization_result['convergence_info']['success'],
                    'final_score': optimization_result['convergence_info'].get('function_value', 0),
                    'iterations': optimization_result['convergence_info'].get('iterations', 0)
                },
                'parameter_changes': optimization_result['optimized_parameters'],
                'validation_status': validation_result['is_valid'],
                'validation_errors': validation_result.get('validation_errors', []),
                'performance_impact': {
                    'overall_improvement': performance_improvement.get('overall_improvement_score', 0),
                    'key_improvements': performance_improvement.get('improvements', {})
                },
                'recommendation': self._generate_optimization_recommendation(
                    validation_result, performance_improvement
                )
            }
            
        except Exception as e:
            logger.error(f"Error generating optimization report: {e}")
            return {'optimization_timestamp': datetime.now(), 'status': 'error'}
    
    def _generate_optimization_recommendation(self, 
                                            validation_result: Dict[str, Any],
                                            performance_improvement: Dict[str, Any]) -> str:
        """Generate optimization recommendation"""
        try:
            if not validation_result.get('is_valid', False):
                return "REJECT - Parameters failed validation"
            
            improvement_score = performance_improvement.get('overall_improvement_score', 0)
            
            if improvement_score > 10:
                return "STRONG ACCEPT - Significant performance improvement"
            elif improvement_score > 5:
                return "ACCEPT - Moderate performance improvement"
            elif improvement_score > 0:
                return "WEAK ACCEPT - Minor performance improvement"
            elif improvement_score > -5:
                return "NEUTRAL - No significant change"
            else:
                return "REJECT - Performance deterioration"
                
        except Exception as e:
            logger.error(f"Error generating recommendation: {e}")
            return "NEUTRAL - Unable to determine"
    
    def _update_optimization_history(self, 
                                   optimization_result: Dict[str, Any],
                                   optimization_report: Dict[str, Any]):
        """Update optimization history"""
        try:
            self.optimization_history['parameters'].append(optimization_result['optimized_parameters'])
            self.optimization_history['performance_scores'].append(
                optimization_result['convergence_info'].get('function_value', 0)
            )
            self.optimization_history['optimization_dates'].append(datetime.now())
            self.optimization_history['convergence_info'].append(optimization_result['convergence_info'])
            
            # Trim history to reasonable size
            max_history = 50
            for key in self.optimization_history.keys():
                if len(self.optimization_history[key]) > max_history:
                    self.optimization_history[key] = self.optimization_history[key][-max_history:]
                    
        except Exception as e:
            logger.error(f"Error updating optimization history: {e}")
    
    def _get_default_optimization_result(self, current_parameters: Dict[str, float]) -> Dict[str, Any]:
        """Get default optimization result when optimization fails"""
        return {
            'optimized_parameters': current_parameters,
            'optimization_report': {
                'optimization_timestamp': datetime.now(),
                'optimization_summary': {'convergence_success': False},
                'recommendation': 'NEUTRAL - Optimization failed'
            },
            'validation_result': {'is_valid': True},
            'performance_improvement': {'overall_improvement_score': 0.0},
            'convergence_info': {'success': False, 'message': 'Optimization failed'}
        }
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization system"""
        try:
            if not self.optimization_history['parameters']:
                return {'status': 'no_optimization_history'}
            
            return {
                'total_optimizations': len(self.optimization_history['parameters']),
                'average_performance_score': np.mean(self.optimization_history['performance_scores']),
                'recent_performance_trend': self._calculate_performance_trend(),
                'parameter_stability': self._analyze_parameter_stability(),
                'optimization_config': {
                    'optimization_window': self.optimization_window,
                    'optimization_method': self.optimization_method,
                    'parameter_bounds': self.parameter_bounds
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting optimization summary: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _calculate_performance_trend(self) -> str:
        """Calculate recent performance trend"""
        try:
            if len(self.optimization_history['performance_scores']) < 3:
                return 'insufficient_data'
            
            recent_scores = self.optimization_history['performance_scores'][-5:]
            if len(recent_scores) >= 3:
                trend_slope = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
                
                if trend_slope > 0.01:
                    return 'improving'
                elif trend_slope < -0.01:
                    return 'declining'
                else:
                    return 'stable'
            
            return 'stable'
            
        except Exception as e:
            logger.error(f"Error calculating performance trend: {e}")
            return 'unknown'
    
    def _analyze_parameter_stability(self) -> Dict[str, float]:
        """Analyze parameter stability across optimizations"""
        try:
            if len(self.optimization_history['parameters']) < 2:
                return {}
            
            stability_analysis = {}
            param_names = list(self.parameter_bounds.keys())
            
            for param in param_names:
                param_values = []
                for param_set in self.optimization_history['parameters']:
                    if param in param_set:
                        param_values.append(param_set[param])
                
                if len(param_values) >= 2:
                    param_std = np.std(param_values)
                    param_range = self.parameter_bounds[param][1] - self.parameter_bounds[param][0]
                    stability_score = max(0, 1 - (param_std / param_range))
                    stability_analysis[param] = float(stability_score)
            
            return stability_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing parameter stability: {e}")
            return {}