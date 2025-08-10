"""
Result Aggregator - Advanced Result Aggregation and Analysis
=========================================================

Aggregates and analyzes results from all market regime components.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod

# Import base utilities
from ..base.common_utils import MathUtils, ErrorHandler

logger = logging.getLogger(__name__)


class AggregationStrategy(ABC):
    """Base class for aggregation strategies"""
    
    @abstractmethod
    def aggregate(self, results: Dict[str, Any], weights: Dict[str, float]) -> Dict[str, Any]:
        """Aggregate results using this strategy"""
        pass


class WeightedAverageStrategy(AggregationStrategy):
    """Weighted average aggregation strategy"""
    
    def aggregate(self, results: Dict[str, Any], weights: Dict[str, float]) -> Dict[str, Any]:
        try:
            weighted_sum = 0.0
            total_weight = 0.0
            component_contributions = {}
            
            for component, result in results.items():
                if component in weights and isinstance(result, dict):
                    score = self._extract_score(result)
                    weight = weights[component]
                    
                    weighted_sum += score * weight
                    total_weight += weight
                    
                    component_contributions[component] = {
                        'score': float(score),
                        'weight': float(weight),
                        'contribution': float(score * weight)
                    }
            
            aggregated_score = weighted_sum / total_weight if total_weight > 0 else 0.0
            
            return {
                'aggregated_score': float(aggregated_score),
                'aggregation_method': 'weighted_average',
                'total_weight': float(total_weight),
                'component_contributions': component_contributions
            }
            
        except Exception as e:
            logger.error(f"Error in weighted average aggregation: {e}")
            return {'aggregated_score': 0.0, 'error': str(e)}
    
    def _extract_score(self, result: Dict[str, Any]) -> float:
        """Extract score from result"""
        score_keys = ['composite_score', 'overall_score', 'score', 'regime_score']
        
        for key in score_keys:
            if key in result:
                return float(result[key])
        
        return 0.5  # Default neutral score


class EnsembleStrategy(AggregationStrategy):
    """Ensemble aggregation strategy with voting"""
    
    def aggregate(self, results: Dict[str, Any], weights: Dict[str, float]) -> Dict[str, Any]:
        try:
            # Extract regime classifications
            regime_votes = {}
            score_sum = 0.0
            total_weight = 0.0
            
            for component, result in results.items():
                if component in weights and isinstance(result, dict):
                    weight = weights[component]
                    score = self._extract_score(result)
                    regime = self._extract_regime(result)
                    
                    # Voting
                    if regime not in regime_votes:
                        regime_votes[regime] = 0.0
                    regime_votes[regime] += weight
                    
                    # Score aggregation
                    score_sum += score * weight
                    total_weight += weight
            
            # Determine winning regime
            winning_regime = max(regime_votes.items(), key=lambda x: x[1])[0] if regime_votes else 'neutral'
            regime_confidence = regime_votes.get(winning_regime, 0.0) / total_weight if total_weight > 0 else 0.0
            
            aggregated_score = score_sum / total_weight if total_weight > 0 else 0.0
            
            return {
                'aggregated_score': float(aggregated_score),
                'regime_classification': winning_regime,
                'regime_confidence': float(regime_confidence),
                'regime_votes': {k: float(v) for k, v in regime_votes.items()},
                'aggregation_method': 'ensemble_voting',
                'total_weight': float(total_weight)
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble aggregation: {e}")
            return {'aggregated_score': 0.0, 'error': str(e)}
    
    def _extract_score(self, result: Dict[str, Any]) -> float:
        """Extract score from result"""
        score_keys = ['composite_score', 'overall_score', 'score', 'regime_score']
        
        for key in score_keys:
            if key in result:
                return float(result[key])
        
        return 0.5
    
    def _extract_regime(self, result: Dict[str, Any]) -> str:
        """Extract regime classification from result"""
        regime_keys = ['regime_classification', 'primary_regime', 'regime', 'classification']
        
        for key in regime_keys:
            if key in result:
                regime_value = result[key]
                if isinstance(regime_value, dict):
                    return regime_value.get('primary_regime', 'neutral')
                else:
                    return str(regime_value)
        
        return 'neutral'


class ResultAggregator:
    """Advanced result aggregation and analysis system"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Result Aggregator"""
        self.config = config
        self.aggregation_strategies = {}
        self.default_strategy = config.get('default_strategy', 'weighted_average')
        
        # Component weights
        self.component_weights = config.get('component_weights', {
            'straddle_analysis': 0.25,
            'oi_pa_analysis': 0.20,
            'greek_sentiment': 0.15,
            'market_breadth': 0.25,
            'iv_analytics': 0.10,
            'technical_indicators': 0.05
        })
        
        # Aggregation configuration
        self.confidence_thresholds = config.get('confidence_thresholds', {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        })
        
        # Historical tracking
        self.aggregation_history = {
            'results': [],
            'timestamps': [],
            'strategies_used': [],
            'confidence_scores': []
        }
        
        # Performance metrics
        self.aggregation_metrics = {
            'total_aggregations': 0,
            'successful_aggregations': 0,
            'failed_aggregations': 0,
            'avg_aggregation_time': 0.0,
            'strategy_usage': {}
        }
        
        # Initialize strategies
        self._initialize_strategies()
        
        # Utilities
        self.math_utils = MathUtils()
        self.error_handler = ErrorHandler()
        
        logger.info("ResultAggregator initialized with advanced aggregation strategies")
    
    def _initialize_strategies(self):
        """Initialize aggregation strategies"""
        try:
            self.aggregation_strategies['weighted_average'] = WeightedAverageStrategy()
            self.aggregation_strategies['ensemble'] = EnsembleStrategy()
            
        except Exception as e:
            logger.error(f"Error initializing aggregation strategies: {e}")
    
    def aggregate_results(self,
                         component_results: Dict[str, Any],
                         aggregation_strategy: Optional[str] = None,
                         custom_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Aggregate results from multiple components
        
        Args:
            component_results: Results from individual components
            aggregation_strategy: Strategy to use for aggregation
            custom_weights: Custom weights for components
            
        Returns:
            Dict with aggregated results and analysis
        """
        start_time = datetime.now()
        
        try:
            strategy_name = aggregation_strategy or self.default_strategy
            weights = custom_weights or self.component_weights
            
            # Validate inputs
            if not self._validate_inputs(component_results, weights):
                return self._get_default_aggregation_result()
            
            # Filter valid results
            valid_results = self._filter_valid_results(component_results)
            
            if not valid_results:
                logger.warning("No valid component results for aggregation")
                return self._get_default_aggregation_result()
            
            # Perform primary aggregation
            primary_aggregation = self._perform_aggregation(valid_results, weights, strategy_name)
            
            # Calculate confidence metrics
            confidence_analysis = self._calculate_confidence_metrics(valid_results, primary_aggregation)
            
            # Generate consensus analysis
            consensus_analysis = self._analyze_consensus(valid_results, weights)
            
            # Detect anomalies and outliers
            anomaly_detection = self._detect_anomalies(valid_results)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(valid_results, primary_aggregation)
            
            # Generate insights and recommendations
            insights = self._generate_insights(valid_results, primary_aggregation, consensus_analysis)
            
            # Compile final result
            aggregated_result = {
                'aggregation_timestamp': datetime.now(),
                'primary_aggregation': primary_aggregation,
                'confidence_analysis': confidence_analysis,
                'consensus_analysis': consensus_analysis,
                'anomaly_detection': anomaly_detection,
                'quality_metrics': quality_metrics,
                'insights': insights,
                'component_summary': self._generate_component_summary(valid_results),
                'aggregation_metadata': self._generate_aggregation_metadata(start_time, strategy_name, len(valid_results))
            }
            
            # Update tracking
            self._update_aggregation_history(aggregated_result, strategy_name)
            self._update_aggregation_metrics(start_time, True, strategy_name)
            
            return aggregated_result
            
        except Exception as e:
            logger.error(f"Error aggregating results: {e}")
            self._update_aggregation_metrics(start_time, False, strategy_name or 'unknown')
            return self._get_default_aggregation_result()
    
    def _validate_inputs(self, component_results: Dict[str, Any], weights: Dict[str, float]) -> bool:
        """Validate aggregation inputs"""
        try:
            if not component_results:
                logger.error("No component results provided")
                return False
            
            if not weights:
                logger.error("No weights provided")
                return False
            
            # Check weight validity
            weight_sum = sum(weights.values())
            if abs(weight_sum - 1.0) > 0.1:
                logger.warning(f"Weights sum to {weight_sum:.3f}, not 1.0")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating inputs: {e}")
            return False
    
    def _filter_valid_results(self, component_results: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out invalid or error results"""
        try:
            valid_results = {}
            
            for component, result in component_results.items():
                if not result:
                    continue
                
                if isinstance(result, dict):
                    # Check for error status
                    if result.get('status') == 'error':
                        logger.warning(f"Component {component} returned error result")
                        continue
                    
                    # Check for required fields
                    if self._has_required_fields(result):
                        valid_results[component] = result
                    else:
                        logger.warning(f"Component {component} missing required fields")
                else:
                    logger.warning(f"Component {component} returned non-dict result")
            
            return valid_results
            
        except Exception as e:
            logger.error(f"Error filtering valid results: {e}")
            return {}
    
    def _has_required_fields(self, result: Dict[str, Any]) -> bool:
        """Check if result has required fields for aggregation"""
        try:
            # Check for at least one score field
            score_keys = ['composite_score', 'overall_score', 'score', 'regime_score']
            has_score = any(key in result for key in score_keys)
            
            return has_score
            
        except Exception as e:
            logger.error(f"Error checking required fields: {e}")
            return False
    
    def _perform_aggregation(self,
                           results: Dict[str, Any],
                           weights: Dict[str, float],
                           strategy_name: str) -> Dict[str, Any]:
        """Perform aggregation using specified strategy"""
        try:
            if strategy_name not in self.aggregation_strategies:
                logger.warning(f"Unknown strategy {strategy_name}, using weighted_average")
                strategy_name = 'weighted_average'
            
            strategy = self.aggregation_strategies[strategy_name]
            return strategy.aggregate(results, weights)
            
        except Exception as e:
            logger.error(f"Error performing aggregation: {e}")
            return {'aggregated_score': 0.0, 'error': str(e)}
    
    def _calculate_confidence_metrics(self,
                                    results: Dict[str, Any],
                                    primary_aggregation: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate confidence metrics for aggregation"""
        try:
            confidence = {
                'overall_confidence': 0.0,
                'component_agreement': 0.0,
                'data_quality_confidence': 0.0,
                'aggregation_confidence': 0.0,
                'confidence_level': 'low'
            }
            
            # Extract component scores
            component_scores = []
            component_confidences = []
            
            for component, result in results.items():
                score = self._extract_score_from_result(result)
                comp_confidence = result.get('confidence', 0.5)
                
                component_scores.append(score)
                component_confidences.append(comp_confidence)
            
            # Component agreement (lower variance = higher agreement)
            if len(component_scores) > 1:
                score_variance = np.var(component_scores)
                confidence['component_agreement'] = float(max(0, 1 - score_variance * 2))
            else:
                confidence['component_agreement'] = 1.0
            
            # Data quality confidence
            if component_confidences:
                confidence['data_quality_confidence'] = float(np.mean(component_confidences))
            
            # Aggregation confidence (based on weights and component count)
            total_weight = primary_aggregation.get('total_weight', 0.0)
            component_count = len(results)
            
            aggregation_conf = min(total_weight, 1.0) * min(component_count / 6, 1.0)  # 6 is ideal component count
            confidence['aggregation_confidence'] = float(aggregation_conf)
            
            # Overall confidence
            confidence['overall_confidence'] = float(np.mean([
                confidence['component_agreement'],
                confidence['data_quality_confidence'],
                confidence['aggregation_confidence']
            ]))
            
            # Confidence level
            overall_conf = confidence['overall_confidence']
            if overall_conf >= self.confidence_thresholds['high']:
                confidence['confidence_level'] = 'high'
            elif overall_conf >= self.confidence_thresholds['medium']:
                confidence['confidence_level'] = 'medium'
            else:
                confidence['confidence_level'] = 'low'
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating confidence metrics: {e}")
            return {'overall_confidence': 0.0, 'confidence_level': 'low'}
    
    def _analyze_consensus(self,
                         results: Dict[str, Any],
                         weights: Dict[str, float]) -> Dict[str, Any]:
        """Analyze consensus among components"""
        try:
            consensus = {
                'consensus_score': 0.0,
                'majority_view': 'neutral',
                'dissenting_components': [],
                'consensus_strength': 'weak',
                'signal_distribution': {}
            }
            
            # Extract signals and classifications
            component_signals = {}
            component_regimes = {}
            
            for component, result in results.items():
                signals = self._extract_signals_from_result(result)
                regime = self._extract_regime_from_result(result)
                
                component_signals[component] = signals
                component_regimes[component] = regime
            
            # Analyze regime consensus
            regime_votes = {}
            for component, regime in component_regimes.items():
                weight = weights.get(component, 0.0)
                if regime not in regime_votes:
                    regime_votes[regime] = 0.0
                regime_votes[regime] += weight
            
            if regime_votes:
                consensus['majority_view'] = max(regime_votes.items(), key=lambda x: x[1])[0]
                max_vote = max(regime_votes.values())
                total_votes = sum(regime_votes.values())
                consensus['consensus_score'] = float(max_vote / total_votes) if total_votes > 0 else 0.0
            
            # Identify dissenting components
            majority_regime = consensus['majority_view']
            for component, regime in component_regimes.items():
                if regime != majority_regime:
                    consensus['dissenting_components'].append({
                        'component': component,
                        'regime': regime,
                        'weight': weights.get(component, 0.0)
                    })
            
            # Consensus strength
            consensus_score = consensus['consensus_score']
            if consensus_score >= 0.8:
                consensus['consensus_strength'] = 'very_strong'
            elif consensus_score >= 0.6:
                consensus['consensus_strength'] = 'strong'
            elif consensus_score >= 0.4:
                consensus['consensus_strength'] = 'moderate'
            else:
                consensus['consensus_strength'] = 'weak'
            
            # Signal distribution
            all_signals = []
            for signals in component_signals.values():
                all_signals.extend(signals)
            
            signal_counts = {}
            for signal in all_signals:
                signal_counts[signal] = signal_counts.get(signal, 0) + 1
            
            consensus['signal_distribution'] = signal_counts
            
            return consensus
            
        except Exception as e:
            logger.error(f"Error analyzing consensus: {e}")
            return {'consensus_score': 0.0, 'majority_view': 'neutral'}
    
    def _detect_anomalies(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies and outliers in component results"""
        try:
            anomalies = {
                'outlier_components': [],
                'anomaly_score': 0.0,
                'anomaly_types': [],
                'recommendations': []
            }
            
            # Extract component scores
            component_scores = {}
            for component, result in results.items():
                score = self._extract_score_from_result(result)
                component_scores[component] = score
            
            if len(component_scores) < 3:
                return anomalies  # Need at least 3 components for outlier detection
            
            scores = list(component_scores.values())
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            # Detect statistical outliers
            for component, score in component_scores.items():
                z_score = abs(score - mean_score) / std_score if std_score > 0 else 0
                
                if z_score > 2.0:  # 2 standard deviations
                    anomalies['outlier_components'].append({
                        'component': component,
                        'score': float(score),
                        'z_score': float(z_score),
                        'deviation_type': 'high' if score > mean_score else 'low'
                    })
            
            # Calculate anomaly score
            anomalies['anomaly_score'] = float(len(anomalies['outlier_components']) / len(component_scores))
            
            # Classify anomaly types
            if anomalies['outlier_components']:
                high_deviations = sum(1 for outlier in anomalies['outlier_components'] if outlier['deviation_type'] == 'high')
                low_deviations = len(anomalies['outlier_components']) - high_deviations
                
                if high_deviations > 0:
                    anomalies['anomaly_types'].append('high_score_outliers')
                if low_deviations > 0:
                    anomalies['anomaly_types'].append('low_score_outliers')
            
            # Generate recommendations
            if anomalies['anomaly_score'] > 0.3:
                anomalies['recommendations'].append("High number of outlier components - review component configurations")
            
            if len(anomalies['outlier_components']) == 1:
                outlier = anomalies['outlier_components'][0]
                anomalies['recommendations'].append(f"Single outlier detected in {outlier['component']} - investigate component health")
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return {'outlier_components': [], 'anomaly_score': 0.0}
    
    def _calculate_quality_metrics(self,
                                 results: Dict[str, Any],
                                 primary_aggregation: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate quality metrics for aggregation"""
        try:
            quality = {
                'overall_quality': 0.0,
                'component_quality': {},
                'aggregation_quality': 0.0,
                'data_completeness': 0.0,
                'quality_grade': 'C'
            }
            
            quality_scores = []
            
            # Component-level quality
            for component, result in results.items():
                comp_quality = result.get('data_quality', {}).get('overall_score', 0.5)
                quality['component_quality'][component] = float(comp_quality)
                quality_scores.append(comp_quality)
            
            # Data completeness
            expected_components = len(self.component_weights)
            actual_components = len(results)
            quality['data_completeness'] = float(actual_components / expected_components)
            
            # Aggregation quality
            agg_score = primary_aggregation.get('aggregated_score', 0.0)
            total_weight = primary_aggregation.get('total_weight', 0.0)
            
            # Quality based on weight coverage and score reasonableness
            weight_coverage = min(total_weight, 1.0)
            score_reasonableness = 1.0 if 0.0 <= agg_score <= 1.0 else 0.5
            
            quality['aggregation_quality'] = float((weight_coverage + score_reasonableness) / 2)
            
            # Overall quality
            quality_factors = [
                np.mean(quality_scores) if quality_scores else 0.5,
                quality['data_completeness'],
                quality['aggregation_quality']
            ]
            
            quality['overall_quality'] = float(np.mean(quality_factors))
            
            # Quality grade
            overall_q = quality['overall_quality']
            if overall_q >= 0.9:
                quality['quality_grade'] = 'A'
            elif overall_q >= 0.8:
                quality['quality_grade'] = 'B'
            elif overall_q >= 0.7:
                quality['quality_grade'] = 'C'
            elif overall_q >= 0.6:
                quality['quality_grade'] = 'D'
            else:
                quality['quality_grade'] = 'F'
            
            return quality
            
        except Exception as e:
            logger.error(f"Error calculating quality metrics: {e}")
            return {'overall_quality': 0.0, 'quality_grade': 'F'}
    
    def _generate_insights(self,
                         results: Dict[str, Any],
                         primary_aggregation: Dict[str, Any],
                         consensus_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights and recommendations"""
        try:
            insights = {
                'key_insights': [],
                'recommendations': [],
                'risk_factors': [],
                'opportunities': []
            }
            
            agg_score = primary_aggregation.get('aggregated_score', 0.0)
            consensus_score = consensus_analysis.get('consensus_score', 0.0)
            majority_view = consensus_analysis.get('majority_view', 'neutral')
            
            # Key insights
            insights['key_insights'].append(f"Aggregated regime score: {agg_score:.3f}")
            insights['key_insights'].append(f"Component consensus: {consensus_score:.1%} on {majority_view}")
            insights['key_insights'].append(f"Active components: {len(results)}")
            
            # Regime-specific insights
            if agg_score > 0.7:
                insights['key_insights'].append("Strong bullish regime indicated")
                insights['opportunities'].append("Consider bullish strategies")
            elif agg_score < 0.3:
                insights['key_insights'].append("Strong bearish regime indicated")
                insights['risk_factors'].append("High downside risk environment")
            else:
                insights['key_insights'].append("Neutral/mixed regime signals")
                insights['recommendations'].append("Monitor for regime transition signals")
            
            # Consensus insights
            if consensus_score > 0.8:
                insights['key_insights'].append("High component agreement")
                insights['opportunities'].append("Strong signal confidence")
            elif consensus_score < 0.4:
                insights['risk_factors'].append("Low component agreement")
                insights['recommendations'].append("Review individual component signals")
            
            # Component-specific insights
            component_contributions = primary_aggregation.get('component_contributions', {})
            if component_contributions:
                top_contributor = max(component_contributions.items(), key=lambda x: x[1].get('contribution', 0))
                insights['key_insights'].append(f"Top contributor: {top_contributor[0]}")
            
            # Risk factors
            dissenting_components = consensus_analysis.get('dissenting_components', [])
            if len(dissenting_components) > 2:
                insights['risk_factors'].append("Multiple dissenting component views")
            
            # Recommendations
            if len(results) < len(self.component_weights):
                insights['recommendations'].append("Consider enabling additional components for better coverage")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {'key_insights': [], 'recommendations': [], 'risk_factors': [], 'opportunities': []}
    
    def _generate_component_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of component contributions"""
        try:
            summary = {
                'total_components': len(results),
                'component_status': {},
                'score_distribution': {},
                'signal_summary': {}
            }
            
            scores = []
            all_signals = []
            
            for component, result in results.items():
                # Component status
                status = result.get('status', 'unknown')
                summary['component_status'][component] = status
                
                # Score collection
                score = self._extract_score_from_result(result)
                scores.append(score)
                
                # Signal collection
                signals = self._extract_signals_from_result(result)
                all_signals.extend(signals)
            
            # Score distribution
            if scores:
                summary['score_distribution'] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores)),
                    'median': float(np.median(scores))
                }
            
            # Signal summary
            signal_counts = {}
            for signal in all_signals:
                signal_counts[signal] = signal_counts.get(signal, 0) + 1
            
            summary['signal_summary'] = {
                'total_signals': len(all_signals),
                'unique_signals': len(signal_counts),
                'top_signals': sorted(signal_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating component summary: {e}")
            return {'total_components': 0}
    
    def _generate_aggregation_metadata(self,
                                     start_time: datetime,
                                     strategy_name: str,
                                     component_count: int) -> Dict[str, Any]:
        """Generate aggregation metadata"""
        try:
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            return {
                'aggregation_timestamp': end_time,
                'processing_time_seconds': float(processing_time),
                'strategy_used': strategy_name,
                'components_processed': component_count,
                'aggregator_version': '2.0.0'
            }
            
        except Exception as e:
            logger.error(f"Error generating aggregation metadata: {e}")
            return {'aggregation_timestamp': datetime.now()}
    
    def _extract_score_from_result(self, result: Dict[str, Any]) -> float:
        """Extract score from component result"""
        try:
            score_keys = ['composite_score', 'overall_score', 'score', 'regime_score']
            
            for key in score_keys:
                if key in result:
                    return float(result[key])
            
            return 0.5  # Default neutral score
            
        except Exception as e:
            logger.error(f"Error extracting score: {e}")
            return 0.5
    
    def _extract_signals_from_result(self, result: Dict[str, Any]) -> List[str]:
        """Extract signals from component result"""
        try:
            signals = []
            signal_keys = ['regime_signals', 'signals', 'indicators', 'patterns']
            
            for key in signal_keys:
                if key in result:
                    signal_data = result[key]
                    if isinstance(signal_data, list):
                        signals.extend([str(s) for s in signal_data])
                    elif isinstance(signal_data, dict):
                        signals.extend([f"{k}:{v}" for k, v in signal_data.items()])
                    elif isinstance(signal_data, str):
                        signals.append(signal_data)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error extracting signals: {e}")
            return []
    
    def _extract_regime_from_result(self, result: Dict[str, Any]) -> str:
        """Extract regime classification from component result"""
        try:
            regime_keys = ['regime_classification', 'primary_regime', 'regime', 'classification']
            
            for key in regime_keys:
                if key in result:
                    regime_value = result[key]
                    if isinstance(regime_value, dict):
                        return regime_value.get('primary_regime', 'neutral')
                    else:
                        return str(regime_value)
            
            # Infer from score
            score = self._extract_score_from_result(result)
            if score > 0.7:
                return 'bullish'
            elif score < 0.3:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.error(f"Error extracting regime: {e}")
            return 'neutral'
    
    def _update_aggregation_history(self, result: Dict[str, Any], strategy_name: str):
        """Update aggregation history"""
        try:
            self.aggregation_history['results'].append(result)
            self.aggregation_history['timestamps'].append(datetime.now())
            self.aggregation_history['strategies_used'].append(strategy_name)
            
            confidence_score = result.get('confidence_analysis', {}).get('overall_confidence', 0.0)
            self.aggregation_history['confidence_scores'].append(confidence_score)
            
            # Trim history
            max_history = 100
            for key in self.aggregation_history.keys():
                if len(self.aggregation_history[key]) > max_history:
                    self.aggregation_history[key] = self.aggregation_history[key][-max_history:]
                    
        except Exception as e:
            logger.error(f"Error updating aggregation history: {e}")
    
    def _update_aggregation_metrics(self, start_time: datetime, success: bool, strategy_name: str):
        """Update aggregation metrics"""
        try:
            self.aggregation_metrics['total_aggregations'] += 1
            
            if success:
                self.aggregation_metrics['successful_aggregations'] += 1
            else:
                self.aggregation_metrics['failed_aggregations'] += 1
            
            # Update execution time
            processing_time = (datetime.now() - start_time).total_seconds()
            current_avg = self.aggregation_metrics['avg_aggregation_time']
            total_aggregations = self.aggregation_metrics['total_aggregations']
            
            # Running average
            self.aggregation_metrics['avg_aggregation_time'] = (
                (current_avg * (total_aggregations - 1) + processing_time) / total_aggregations
            )
            
            # Update strategy usage
            if strategy_name not in self.aggregation_metrics['strategy_usage']:
                self.aggregation_metrics['strategy_usage'][strategy_name] = 0
            self.aggregation_metrics['strategy_usage'][strategy_name] += 1
            
        except Exception as e:
            logger.error(f"Error updating aggregation metrics: {e}")
    
    def _get_default_aggregation_result(self) -> Dict[str, Any]:
        """Get default aggregation result when aggregation fails"""
        return {
            'aggregation_timestamp': datetime.now(),
            'primary_aggregation': {'aggregated_score': 0.0, 'aggregation_method': 'default'},
            'confidence_analysis': {'overall_confidence': 0.0, 'confidence_level': 'low'},
            'consensus_analysis': {'consensus_score': 0.0, 'majority_view': 'neutral'},
            'anomaly_detection': {'outlier_components': [], 'anomaly_score': 0.0},
            'quality_metrics': {'overall_quality': 0.0, 'quality_grade': 'F'},
            'insights': {'key_insights': ['Aggregation failed'], 'recommendations': ['Check component results']},
            'component_summary': {'total_components': 0},
            'aggregation_metadata': {'aggregation_timestamp': datetime.now(), 'status': 'failed'}
        }
    
    def get_aggregator_status(self) -> Dict[str, Any]:
        """Get comprehensive aggregator status"""
        try:
            return {
                'aggregator_status': 'operational',
                'strategies_available': list(self.aggregation_strategies.keys()),
                'default_strategy': self.default_strategy,
                'component_weights': self.component_weights.copy(),
                'aggregation_metrics': self.aggregation_metrics.copy(),
                'history_length': len(self.aggregation_history['results']),
                'average_confidence': float(np.mean(self.aggregation_history['confidence_scores'])) if self.aggregation_history['confidence_scores'] else 0.0,
                'success_rate': (self.aggregation_metrics['successful_aggregations'] / self.aggregation_metrics['total_aggregations']) if self.aggregation_metrics['total_aggregations'] > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error getting aggregator status: {e}")
            return {'aggregator_status': 'error', 'error': str(e)}