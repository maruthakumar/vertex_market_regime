"""
Weight Validator - Adaptive Weight Validation System
=================================================

Validates and optimizes component weights for market regime analysis.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import base utilities
from ...base.common_utils import MathUtils, DataValidator, ErrorHandler

logger = logging.getLogger(__name__)


class WeightValidator:
    """Advanced weight validation and optimization system"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Weight Validator"""
        self.validation_window = config.get('validation_window', 100)
        self.weight_bounds = config.get('weight_bounds', {
            'min_weight': 0.0,
            'max_weight': 1.0,
            'min_total_weight': 0.95,
            'max_total_weight': 1.05
        })
        
        # Component categories and their allowed weight ranges
        self.component_categories = config.get('component_categories', {
            'straddle_analysis': {'min': 0.15, 'max': 0.35},
            'oi_pa_analysis': {'min': 0.10, 'max': 0.30},
            'greek_sentiment': {'min': 0.10, 'max': 0.25},
            'market_breadth': {'min': 0.15, 'max': 0.35},
            'iv_analytics': {'min': 0.05, 'max': 0.20},
            'technical_indicators': {'min': 0.05, 'max': 0.20}
        })
        
        # Validation history
        self.validation_history = {
            'weight_sets': [],
            'validation_scores': [],
            'performance_metrics': [],
            'timestamps': []
        }
        
        # Mathematical utilities
        self.math_utils = MathUtils()
        self.data_validator = DataValidator()
        
        logger.info("WeightValidator initialized with comprehensive weight validation")
    
    def validate_weights(self, 
                        weights: Dict[str, float],
                        component_scores: Dict[str, pd.Series],
                        target_output: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Comprehensive weight validation
        
        Args:
            weights: Component weights to validate
            component_scores: Historical component scores
            target_output: Optional target output for optimization
            
        Returns:
            Dict with validation results and recommendations
        """
        try:
            if not weights:
                return self._get_default_validation_result()
            
            # Basic weight validation
            basic_validation = self._validate_basic_weight_constraints(weights)
            
            # Category weight validation
            category_validation = self._validate_category_weights(weights)
            
            # Historical performance validation
            performance_validation = self._validate_historical_performance(weights, component_scores, target_output)
            
            # Weight stability validation
            stability_validation = self._validate_weight_stability(weights)
            
            # Correlation analysis
            correlation_analysis = self._analyze_weight_correlations(weights, component_scores)
            
            # Diversification metrics
            diversification_metrics = self._calculate_diversification_metrics(weights)
            
            # Risk analysis
            risk_analysis = self._analyze_weight_risks(weights, component_scores)
            
            # Generate optimization recommendations
            optimization_recommendations = self._generate_optimization_recommendations(
                basic_validation, category_validation, performance_validation
            )
            
            # Calculate overall validation score
            overall_score = self._calculate_overall_validation_score(
                basic_validation, category_validation, performance_validation, stability_validation
            )
            
            # Update validation history
            self._update_validation_history(weights, overall_score, performance_validation)
            
            return {
                'validation_timestamp': datetime.now(),
                'overall_score': overall_score,
                'basic_validation': basic_validation,
                'category_validation': category_validation,
                'performance_validation': performance_validation,
                'stability_validation': stability_validation,
                'correlation_analysis': correlation_analysis,
                'diversification_metrics': diversification_metrics,
                'risk_analysis': risk_analysis,
                'optimization_recommendations': optimization_recommendations,
                'validation_status': 'PASS' if overall_score > 0.7 else 'FAIL' if overall_score < 0.4 else 'WARNING'
            }
            
        except Exception as e:
            logger.error(f"Error in weight validation: {e}")
            return self._get_default_validation_result()
    
    def _validate_basic_weight_constraints(self, weights: Dict[str, float]) -> Dict[str, Any]:
        """Validate basic weight constraints"""
        try:
            validation = {
                'is_valid': True,
                'errors': [],
                'warnings': [],
                'weight_analysis': {}
            }
            
            # Check individual weight bounds
            for component, weight in weights.items():
                weight_info = {
                    'value': float(weight),
                    'within_bounds': True,
                    'issues': []
                }
                
                # Check basic bounds
                if weight < self.weight_bounds['min_weight']:
                    validation['errors'].append(f"{component}: weight {weight:.4f} below minimum {self.weight_bounds['min_weight']}")
                    weight_info['within_bounds'] = False
                    weight_info['issues'].append('below_minimum')
                
                if weight > self.weight_bounds['max_weight']:
                    validation['errors'].append(f"{component}: weight {weight:.4f} above maximum {self.weight_bounds['max_weight']}")
                    weight_info['within_bounds'] = False
                    weight_info['issues'].append('above_maximum')
                
                # Check for extreme values
                if weight < 0.01:
                    validation['warnings'].append(f"{component}: very low weight {weight:.4f} may be ineffective")
                    weight_info['issues'].append('very_low')
                
                if weight > 0.8:
                    validation['warnings'].append(f"{component}: very high weight {weight:.4f} may cause over-reliance")
                    weight_info['issues'].append('very_high')
                
                validation['weight_analysis'][component] = weight_info
            
            # Check total weight
            total_weight = sum(weights.values())
            validation['total_weight'] = float(total_weight)
            
            if total_weight < self.weight_bounds['min_total_weight']:
                validation['errors'].append(f"Total weight {total_weight:.4f} below minimum {self.weight_bounds['min_total_weight']}")
                validation['is_valid'] = False
            
            if total_weight > self.weight_bounds['max_total_weight']:
                validation['errors'].append(f"Total weight {total_weight:.4f} above maximum {self.weight_bounds['max_total_weight']}")
                validation['is_valid'] = False
            
            # Weight distribution analysis
            weight_values = list(weights.values())
            validation['weight_statistics'] = {
                'mean': float(np.mean(weight_values)),
                'std': float(np.std(weight_values)),
                'min': float(np.min(weight_values)),
                'max': float(np.max(weight_values)),
                'coefficient_of_variation': float(np.std(weight_values) / np.mean(weight_values)) if np.mean(weight_values) > 0 else 0
            }
            
            if validation['errors']:
                validation['is_valid'] = False
            
            return validation
            
        except Exception as e:
            logger.error(f"Error validating basic weight constraints: {e}")
            return {'is_valid': False, 'errors': [str(e)]}
    
    def _validate_category_weights(self, weights: Dict[str, float]) -> Dict[str, Any]:
        """Validate weights by component category"""
        try:
            validation = {
                'is_valid': True,
                'category_analysis': {},
                'errors': [],
                'warnings': []
            }
            
            # Group weights by category
            category_weights = {}
            unmatched_components = []
            
            for component, weight in weights.items():
                matched_category = None
                for category in self.component_categories.keys():
                    if category in component.lower() or any(cat_part in component.lower() for cat_part in category.split('_')):
                        matched_category = category
                        break
                
                if matched_category:
                    if matched_category not in category_weights:
                        category_weights[matched_category] = 0
                    category_weights[matched_category] += weight
                else:
                    unmatched_components.append(component)
            
            # Validate category weights
            for category, bounds in self.component_categories.items():
                category_weight = category_weights.get(category, 0)
                
                category_info = {
                    'total_weight': float(category_weight),
                    'min_bound': bounds['min'],
                    'max_bound': bounds['max'],
                    'within_bounds': bounds['min'] <= category_weight <= bounds['max'],
                    'issues': []
                }
                
                if category_weight < bounds['min']:
                    validation['errors'].append(f"Category {category}: total weight {category_weight:.4f} below minimum {bounds['min']}")
                    category_info['issues'].append('below_minimum')
                    validation['is_valid'] = False
                
                if category_weight > bounds['max']:
                    validation['errors'].append(f"Category {category}: total weight {category_weight:.4f} above maximum {bounds['max']}")
                    category_info['issues'].append('above_maximum')
                    validation['is_valid'] = False
                
                # Warning for missing categories
                if category_weight == 0:
                    validation['warnings'].append(f"Category {category}: no weight assigned")
                    category_info['issues'].append('missing')
                
                validation['category_analysis'][category] = category_info
            
            # Handle unmatched components
            if unmatched_components:
                total_unmatched_weight = sum(weights[comp] for comp in unmatched_components)
                validation['warnings'].append(f"Unmatched components: {unmatched_components} (total weight: {total_unmatched_weight:.4f})")
                validation['category_analysis']['unmatched'] = {
                    'components': unmatched_components,
                    'total_weight': float(total_unmatched_weight)
                }
            
            return validation
            
        except Exception as e:
            logger.error(f"Error validating category weights: {e}")
            return {'is_valid': False, 'errors': [str(e)]}
    
    def _validate_historical_performance(self, 
                                       weights: Dict[str, float],
                                       component_scores: Dict[str, pd.Series],
                                       target_output: Optional[pd.Series]) -> Dict[str, Any]:
        """Validate weights against historical performance"""
        try:
            validation = {
                'performance_score': 0.0,
                'metrics': {},
                'component_contributions': {},
                'recommendations': []
            }
            
            if not component_scores:
                validation['recommendations'].append("No historical data available for performance validation")
                return validation
            
            # Align all component scores
            aligned_scores = self._align_component_scores(component_scores)
            
            if aligned_scores.empty:
                validation['recommendations'].append("No aligned historical data available")
                return validation
            
            # Calculate weighted composite score
            composite_score = self._calculate_weighted_composite(aligned_scores, weights)
            
            # Performance metrics
            if target_output is not None:
                aligned_target = target_output.reindex(composite_score.index).fillna(0)
                
                # Calculate correlation
                correlation = composite_score.corr(aligned_target)
                validation['metrics']['correlation_with_target'] = float(correlation) if not np.isnan(correlation) else 0
                
                # Calculate tracking error
                tracking_error = (composite_score - aligned_target).std()
                validation['metrics']['tracking_error'] = float(tracking_error)
                
                # Calculate R-squared
                r_squared = correlation ** 2 if not np.isnan(correlation) else 0
                validation['metrics']['r_squared'] = float(r_squared)
                
                # Mean absolute error
                mae = mean_absolute_error(aligned_target, composite_score)
                validation['metrics']['mean_absolute_error'] = float(mae)
            
            # Component contribution analysis
            for component in weights.keys():
                if component in aligned_scores.columns:
                    component_series = aligned_scores[component]
                    contribution = weights[component] * component_series.std()
                    validation['component_contributions'][component] = {
                        'weighted_volatility_contribution': float(contribution),
                        'correlation_with_composite': float(component_series.corr(composite_score)) if not component_series.empty else 0,
                        'individual_volatility': float(component_series.std())
                    }
            
            # Calculate overall performance score
            if target_output is not None:
                base_score = validation['metrics'].get('r_squared', 0) * 0.6
                correlation_score = abs(validation['metrics'].get('correlation_with_target', 0)) * 0.4
                validation['performance_score'] = float(base_score + correlation_score)
            else:
                # Score based on diversification and stability
                volatility_score = max(0, 1 - composite_score.std() / 0.5)  # Penalize high volatility
                consistency_score = self._calculate_score_consistency(composite_score)
                validation['performance_score'] = float((volatility_score + consistency_score) / 2)
            
            # Generate recommendations
            if validation['performance_score'] < 0.3:
                validation['recommendations'].append("Poor historical performance - consider weight rebalancing")
            elif validation['performance_score'] < 0.6:
                validation['recommendations'].append("Moderate performance - monitor weight effectiveness")
            else:
                validation['recommendations'].append("Good historical performance")
            
            return validation
            
        except Exception as e:
            logger.error(f"Error validating historical performance: {e}")
            return {'performance_score': 0.0, 'recommendations': [str(e)]}
    
    def _align_component_scores(self, component_scores: Dict[str, pd.Series]) -> pd.DataFrame:
        """Align component scores by timestamp"""
        try:
            if not component_scores:
                return pd.DataFrame()
            
            # Find common index
            common_index = None
            for series in component_scores.values():
                if common_index is None:
                    common_index = series.index
                else:
                    common_index = common_index.intersection(series.index)
            
            if common_index.empty:
                return pd.DataFrame()
            
            # Align all series
            aligned_data = {}
            for component, series in component_scores.items():
                aligned_series = series.reindex(common_index).fillna(series.mean())
                aligned_data[component] = aligned_series
            
            return pd.DataFrame(aligned_data)
            
        except Exception as e:
            logger.error(f"Error aligning component scores: {e}")
            return pd.DataFrame()
    
    def _calculate_weighted_composite(self, aligned_scores: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
        """Calculate weighted composite score"""
        try:
            composite = pd.Series([0.0] * len(aligned_scores), index=aligned_scores.index)
            
            for component, weight in weights.items():
                if component in aligned_scores.columns:
                    component_contribution = aligned_scores[component] * weight
                    composite += component_contribution
            
            return composite
            
        except Exception as e:
            logger.error(f"Error calculating weighted composite: {e}")
            return pd.Series([0.0])
    
    def _calculate_score_consistency(self, scores: pd.Series) -> float:
        """Calculate consistency score"""
        try:
            if len(scores) < 2:
                return 0.0
            
            # Calculate rolling standard deviation
            rolling_std = scores.rolling(window=min(20, len(scores)//4)).std()
            
            # Consistency is inverse of volatility
            avg_volatility = rolling_std.mean()
            consistency = max(0, 1 - avg_volatility / 0.3)  # Normalize to 0.3 std
            
            return float(consistency)
            
        except Exception as e:
            logger.error(f"Error calculating score consistency: {e}")
            return 0.0
    
    def _validate_weight_stability(self, weights: Dict[str, float]) -> Dict[str, Any]:
        """Validate weight stability compared to historical weights"""
        try:
            validation = {
                'stability_score': 1.0,
                'stability_analysis': {},
                'recommendations': []
            }
            
            if len(self.validation_history['weight_sets']) < 2:
                validation['recommendations'].append("Insufficient historical data for stability analysis")
                return validation
            
            # Compare with recent weight sets
            recent_weights = self.validation_history['weight_sets'][-3:]  # Last 3 weight sets
            stability_scores = []
            
            for past_weights in recent_weights:
                weight_changes = []
                for component, current_weight in weights.items():
                    if component in past_weights:
                        past_weight = past_weights[component]
                        weight_change = abs(current_weight - past_weight)
                        weight_changes.append(weight_change)
                
                if weight_changes:
                    avg_change = np.mean(weight_changes)
                    stability_score = max(0, 1 - avg_change / 0.2)  # Penalize changes > 20%
                    stability_scores.append(stability_score)
            
            if stability_scores:
                validation['stability_score'] = float(np.mean(stability_scores))
                
                # Individual component stability
                for component, current_weight in weights.items():
                    component_changes = []
                    for past_weights in recent_weights:
                        if component in past_weights:
                            change = abs(current_weight - past_weights[component])
                            component_changes.append(change)
                    
                    if component_changes:
                        avg_change = np.mean(component_changes)
                        component_stability = max(0, 1 - avg_change / 0.15)
                        validation['stability_analysis'][component] = {
                            'stability_score': float(component_stability),
                            'avg_change': float(avg_change),
                            'max_change': float(max(component_changes))
                        }
            
            # Generate recommendations
            if validation['stability_score'] < 0.5:
                validation['recommendations'].append("High weight volatility detected - consider gradual transitions")
            elif validation['stability_score'] < 0.7:
                validation['recommendations'].append("Moderate weight changes - monitor stability")
            else:
                validation['recommendations'].append("Weight changes are stable")
            
            return validation
            
        except Exception as e:
            logger.error(f"Error validating weight stability: {e}")
            return {'stability_score': 1.0, 'recommendations': [str(e)]}
    
    def _analyze_weight_correlations(self, 
                                   weights: Dict[str, float],
                                   component_scores: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Analyze correlations between weighted components"""
        try:
            analysis = {
                'correlation_matrix': {},
                'high_correlations': [],
                'diversification_score': 0.0
            }
            
            # Align component scores
            aligned_scores = self._align_component_scores(component_scores)
            
            if aligned_scores.empty or len(aligned_scores.columns) < 2:
                analysis['note'] = "Insufficient data for correlation analysis"
                return analysis
            
            # Calculate correlation matrix
            correlation_matrix = aligned_scores.corr()
            
            # Convert to serializable format
            for i, col1 in enumerate(correlation_matrix.columns):
                analysis['correlation_matrix'][col1] = {}
                for j, col2 in enumerate(correlation_matrix.columns):
                    corr_value = correlation_matrix.iloc[i, j]
                    analysis['correlation_matrix'][col1][col2] = float(corr_value) if not np.isnan(corr_value) else 0
            
            # Identify high correlations
            high_corr_threshold = 0.7
            for i, col1 in enumerate(correlation_matrix.columns):
                for j, col2 in enumerate(correlation_matrix.columns):
                    if i < j:  # Avoid duplicates
                        corr_value = correlation_matrix.iloc[i, j]
                        if abs(corr_value) > high_corr_threshold:
                            analysis['high_correlations'].append({
                                'component1': col1,
                                'component2': col2,
                                'correlation': float(corr_value),
                                'combined_weight': weights.get(col1, 0) + weights.get(col2, 0)
                            })
            
            # Calculate diversification score
            # Lower average absolute correlation = better diversification
            correlations = []
            for i, col1 in enumerate(correlation_matrix.columns):
                for j, col2 in enumerate(correlation_matrix.columns):
                    if i != j:
                        corr_value = correlation_matrix.iloc[i, j]
                        if not np.isnan(corr_value):
                            correlations.append(abs(corr_value))
            
            if correlations:
                avg_abs_correlation = np.mean(correlations)
                analysis['diversification_score'] = float(max(0, 1 - avg_abs_correlation))
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing weight correlations: {e}")
            return {'diversification_score': 0.0}
    
    def _calculate_diversification_metrics(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate diversification metrics"""
        try:
            metrics = {}
            
            weight_values = list(weights.values())
            if not weight_values:
                return metrics
            
            # Herfindahl-Hirschman Index (concentration)
            hhi = sum(w**2 for w in weight_values)
            metrics['herfindahl_index'] = float(hhi)
            
            # Effective number of components
            effective_components = 1 / hhi if hhi > 0 else 0
            metrics['effective_components'] = float(effective_components)
            
            # Diversification ratio
            max_possible_components = len(weight_values)
            diversification_ratio = effective_components / max_possible_components if max_possible_components > 0 else 0
            metrics['diversification_ratio'] = float(diversification_ratio)
            
            # Weight entropy (Shannon entropy)
            entropy = -sum(w * np.log(w) for w in weight_values if w > 0)
            max_entropy = np.log(len(weight_values))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            metrics['weight_entropy'] = float(normalized_entropy)
            
            # Gini coefficient (inequality measure)
            sorted_weights = sorted(weight_values)
            n = len(sorted_weights)
            cumsum = np.cumsum(sorted_weights)
            gini = (n + 1 - 2 * sum((n + 1 - i) * w for i, w in enumerate(sorted_weights, 1))) / (n * sum(sorted_weights))
            metrics['gini_coefficient'] = float(gini)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating diversification metrics: {e}")
            return {}
    
    def _analyze_weight_risks(self, 
                            weights: Dict[str, float],
                            component_scores: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Analyze risks associated with current weights"""
        try:
            risk_analysis = {
                'concentration_risk': 0.0,
                'volatility_risk': 0.0,
                'correlation_risk': 0.0,
                'overall_risk_score': 0.0,
                'risk_factors': []
            }
            
            # Concentration risk
            weight_values = list(weights.values())
            max_weight = max(weight_values) if weight_values else 0
            concentration_risk = max_weight  # Higher max weight = higher concentration risk
            risk_analysis['concentration_risk'] = float(concentration_risk)
            
            if max_weight > 0.4:
                risk_analysis['risk_factors'].append(f"High concentration in single component: {max_weight:.3f}")
            
            # Volatility risk (if historical data available)
            if component_scores:
                aligned_scores = self._align_component_scores(component_scores)
                if not aligned_scores.empty:
                    composite_score = self._calculate_weighted_composite(aligned_scores, weights)
                    volatility = composite_score.std()
                    volatility_risk = min(volatility / 0.3, 1.0)  # Normalize to 0.3 std
                    risk_analysis['volatility_risk'] = float(volatility_risk)
                    
                    if volatility > 0.25:
                        risk_analysis['risk_factors'].append(f"High composite volatility: {volatility:.3f}")
            
            # Correlation risk
            correlation_analysis = self._analyze_weight_correlations(weights, component_scores)
            high_correlations = correlation_analysis.get('high_correlations', [])
            correlation_risk = min(len(high_correlations) / 5, 1.0)  # Normalize to 5 high correlations
            risk_analysis['correlation_risk'] = float(correlation_risk)
            
            if high_correlations:
                risk_analysis['risk_factors'].append(f"High correlations detected: {len(high_correlations)} pairs")
            
            # Overall risk score
            risk_weights = {'concentration': 0.4, 'volatility': 0.3, 'correlation': 0.3}
            overall_risk = (risk_weights['concentration'] * concentration_risk +
                          risk_weights['volatility'] * risk_analysis['volatility_risk'] +
                          risk_weights['correlation'] * correlation_risk)
            risk_analysis['overall_risk_score'] = float(overall_risk)
            
            # Risk level classification
            if overall_risk > 0.7:
                risk_analysis['risk_level'] = 'HIGH'
            elif overall_risk > 0.4:
                risk_analysis['risk_level'] = 'MEDIUM'
            else:
                risk_analysis['risk_level'] = 'LOW'
            
            return risk_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing weight risks: {e}")
            return {'overall_risk_score': 0.5, 'risk_level': 'UNKNOWN'}
    
    def _generate_optimization_recommendations(self, 
                                             basic_validation: Dict[str, Any],
                                             category_validation: Dict[str, Any],
                                             performance_validation: Dict[str, Any]) -> List[str]:
        """Generate weight optimization recommendations"""
        try:
            recommendations = []
            
            # Basic validation recommendations
            if not basic_validation.get('is_valid', False):
                recommendations.append("Fix basic weight constraint violations before proceeding")
                for error in basic_validation.get('errors', []):
                    recommendations.append(f"  - {error}")
            
            # Category validation recommendations
            if not category_validation.get('is_valid', False):
                recommendations.append("Rebalance category weights to meet allocation targets")
                for error in category_validation.get('errors', []):
                    recommendations.append(f"  - {error}")
            
            # Performance-based recommendations
            performance_score = performance_validation.get('performance_score', 0)
            if performance_score < 0.4:
                recommendations.append("Poor performance validation - consider major weight restructuring")
            elif performance_score < 0.6:
                recommendations.append("Moderate performance - fine-tune weights for better alignment")
            
            # Specific component recommendations
            component_contributions = performance_validation.get('component_contributions', {})
            for component, contrib in component_contributions.items():
                correlation = contrib.get('correlation_with_composite', 0)
                if abs(correlation) < 0.2:
                    recommendations.append(f"Component {component} has low contribution - consider reducing weight")
                elif abs(correlation) > 0.9:
                    recommendations.append(f"Component {component} is highly correlated - check for redundancy")
            
            # Weight distribution recommendations
            weight_stats = basic_validation.get('weight_statistics', {})
            cv = weight_stats.get('coefficient_of_variation', 0)
            if cv > 1.0:
                recommendations.append("High weight variation - consider more balanced distribution")
            elif cv < 0.3:
                recommendations.append("Very even weight distribution - ensure differentiation is intentional")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating optimization recommendations: {e}")
            return ["Error generating recommendations"]
    
    def _calculate_overall_validation_score(self, 
                                          basic_validation: Dict[str, Any],
                                          category_validation: Dict[str, Any],
                                          performance_validation: Dict[str, Any],
                                          stability_validation: Dict[str, Any]) -> float:
        """Calculate overall validation score"""
        try:
            # Scoring weights
            basic_weight = 0.3
            category_weight = 0.3
            performance_weight = 0.3
            stability_weight = 0.1
            
            # Basic validation score
            basic_score = 1.0 if basic_validation.get('is_valid', False) else 0.0
            
            # Category validation score
            category_score = 1.0 if category_validation.get('is_valid', False) else 0.0
            
            # Performance validation score
            performance_score = performance_validation.get('performance_score', 0.0)
            
            # Stability validation score
            stability_score = stability_validation.get('stability_score', 1.0)
            
            # Calculate weighted average
            overall_score = (basic_weight * basic_score +
                           category_weight * category_score +
                           performance_weight * performance_score +
                           stability_weight * stability_score)
            
            return float(max(min(overall_score, 1.0), 0.0))
            
        except Exception as e:
            logger.error(f"Error calculating overall validation score: {e}")
            return 0.5
    
    def _update_validation_history(self, 
                                 weights: Dict[str, float],
                                 validation_score: float,
                                 performance_validation: Dict[str, Any]):
        """Update validation history"""
        try:
            self.validation_history['weight_sets'].append(weights.copy())
            self.validation_history['validation_scores'].append(validation_score)
            self.validation_history['performance_metrics'].append(performance_validation.get('metrics', {}))
            self.validation_history['timestamps'].append(datetime.now())
            
            # Trim history to reasonable size
            max_history = 50
            for key in self.validation_history.keys():
                if len(self.validation_history[key]) > max_history:
                    self.validation_history[key] = self.validation_history[key][-max_history:]
                    
        except Exception as e:
            logger.error(f"Error updating validation history: {e}")
    
    def _get_default_validation_result(self) -> Dict[str, Any]:
        """Get default validation result when validation fails"""
        return {
            'validation_timestamp': datetime.now(),
            'overall_score': 0.0,
            'basic_validation': {'is_valid': False},
            'category_validation': {'is_valid': False},
            'performance_validation': {'performance_score': 0.0},
            'stability_validation': {'stability_score': 0.0},
            'correlation_analysis': {},
            'diversification_metrics': {},
            'risk_analysis': {'overall_risk_score': 1.0, 'risk_level': 'HIGH'},
            'optimization_recommendations': ['Validation failed - check weight configuration'],
            'validation_status': 'FAIL'
        }
    
    def optimize_weights(self, 
                        current_weights: Dict[str, float],
                        component_scores: Dict[str, pd.Series],
                        target_output: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Optimize weights using historical performance"""
        try:
            if not component_scores:
                return {'optimized_weights': current_weights, 'optimization_success': False}
            
            # Align component scores
            aligned_scores = self._align_component_scores(component_scores)
            
            if aligned_scores.empty:
                return {'optimized_weights': current_weights, 'optimization_success': False}
            
            # Define optimization objective
            def objective(weight_array):
                component_names = list(current_weights.keys())
                weights_dict = dict(zip(component_names, weight_array))
                
                # Calculate weighted composite
                composite = self._calculate_weighted_composite(aligned_scores, weights_dict)
                
                if target_output is not None:
                    aligned_target = target_output.reindex(composite.index).fillna(0)
                    # Minimize mean squared error with target
                    mse = mean_squared_error(aligned_target, composite)
                    return mse
                else:
                    # Maximize Sharpe ratio of composite
                    if composite.std() > 0:
                        sharpe = composite.mean() / composite.std()
                        return -sharpe  # Minimize negative Sharpe
                    else:
                        return 1.0
            
            # Set up constraints and bounds
            component_names = list(current_weights.keys())
            n_components = len(component_names)
            
            # Bounds for each weight
            bounds = [(self.weight_bounds['min_weight'], self.weight_bounds['max_weight']) for _ in range(n_components)]
            
            # Constraint: weights sum to approximately 1
            constraints = [{
                'type': 'eq',
                'fun': lambda x: sum(x) - 1.0
            }]
            
            # Initial guess
            x0 = [current_weights[comp] for comp in component_names]
            
            # Run optimization
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimized_weights = dict(zip(component_names, result.x))
                
                # Validate optimized weights
                validation_result = self.validate_weights(optimized_weights, component_scores, target_output)
                
                return {
                    'optimized_weights': optimized_weights,
                    'optimization_success': True,
                    'optimization_result': {
                        'objective_value': float(result.fun),
                        'iterations': int(result.nit),
                        'message': str(result.message)
                    },
                    'validation_result': validation_result
                }
            else:
                return {
                    'optimized_weights': current_weights,
                    'optimization_success': False,
                    'optimization_result': {
                        'message': str(result.message)
                    }
                }
                
        except Exception as e:
            logger.error(f"Error optimizing weights: {e}")
            return {'optimized_weights': current_weights, 'optimization_success': False, 'error': str(e)}
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of weight validation system"""
        try:
            if not self.validation_history['validation_scores']:
                return {'status': 'no_validation_history'}
            
            return {
                'total_validations': len(self.validation_history['validation_scores']),
                'average_validation_score': float(np.mean(self.validation_history['validation_scores'])),
                'recent_validation_trend': self._calculate_validation_trend(),
                'weight_stability_trend': self._analyze_weight_stability_trend(),
                'validation_config': {
                    'validation_window': self.validation_window,
                    'weight_bounds': self.weight_bounds,
                    'component_categories': self.component_categories
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting validation summary: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _calculate_validation_trend(self) -> str:
        """Calculate recent validation score trend"""
        try:
            if len(self.validation_history['validation_scores']) < 3:
                return 'insufficient_data'
            
            recent_scores = self.validation_history['validation_scores'][-5:]
            if len(recent_scores) >= 3:
                trend_slope = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
                
                if trend_slope > 0.05:
                    return 'improving'
                elif trend_slope < -0.05:
                    return 'declining'
                else:
                    return 'stable'
            
            return 'stable'
            
        except Exception as e:
            logger.error(f"Error calculating validation trend: {e}")
            return 'unknown'
    
    def _analyze_weight_stability_trend(self) -> Dict[str, float]:
        """Analyze overall weight stability trend"""
        try:
            if len(self.validation_history['weight_sets']) < 3:
                return {}
            
            # Calculate average weight changes over time
            weight_changes = []
            recent_weights = self.validation_history['weight_sets'][-5:]
            
            for i in range(1, len(recent_weights)):
                current_weights = recent_weights[i]
                previous_weights = recent_weights[i-1]
                
                changes = []
                for component in current_weights.keys():
                    if component in previous_weights:
                        change = abs(current_weights[component] - previous_weights[component])
                        changes.append(change)
                
                if changes:
                    weight_changes.append(np.mean(changes))
            
            if weight_changes:
                return {
                    'average_change': float(np.mean(weight_changes)),
                    'change_volatility': float(np.std(weight_changes)),
                    'max_change': float(max(weight_changes))
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error analyzing weight stability trend: {e}")
            return {}