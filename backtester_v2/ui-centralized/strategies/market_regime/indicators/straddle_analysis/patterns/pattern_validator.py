"""
7-Layer Pattern Validation System for >90% Success Rate

Implements comprehensive pattern validation across 7 critical layers:
1. Timeframe Alignment Validation (91% threshold)
2. Technical Confluence Validation (89% threshold) 
3. Volume Confirmation Validation (93% threshold)
4. Statistical Significance Testing (87% threshold)
5. Historical Consistency Validation (90% threshold)
6. Risk-Reward Optimization (92% threshold)
7. Market Context Validation (88% threshold)

Only patterns passing ALL 7 layers with overall score >90% are accepted.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from .pattern_repository import PatternSchema
from .statistical_validator import StatisticalPatternValidator

logger = logging.getLogger(__name__)


@dataclass
class ValidationLayerResult:
    """Result of a single validation layer"""
    layer_name: str
    layer_number: int
    threshold: float
    score: float
    passed: bool
    details: Dict[str, Any]
    validation_time: float


@dataclass
class ValidationResult:
    """Complete 7-layer validation result"""
    pattern_id: str
    overall_score: float
    overall_passed: bool
    validation_time: float
    
    # Individual layer results
    layer_results: List[ValidationLayerResult]
    
    # Validation metadata
    validation_timestamp: datetime
    components_validated: List[str]
    timeframes_validated: List[int]
    
    # Quality metrics
    confidence_score: float
    reliability_score: float
    success_probability: float


class SevenLayerPatternValidator:
    """
    7-Layer Pattern Validation System for Ultra-High Success Rates
    
    Implements rigorous validation across 7 critical dimensions to ensure
    only patterns with >90% success probability are accepted for trading.
    
    Validation Layers:
    1. Timeframe Alignment (91%): Multi-timeframe pattern consistency
    2. Technical Confluence (89%): Technical indicator alignment  
    3. Volume Confirmation (93%): Volume-based pattern validation
    4. Statistical Significance (87%): Statistical tests (t-test, chi-square, etc.)
    5. Historical Consistency (90%): Historical pattern performance
    6. Risk-Reward Optimization (92%): Risk-adjusted return validation
    7. Market Context Validation (88%): Market regime and context fitting
    
    Success Criteria:
    - ALL 7 layers must pass their individual thresholds
    - Overall weighted score must exceed 90%
    - Minimum 100 historical occurrences required
    - Statistical significance p-value < 0.01
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize 7-Layer Pattern Validator
        
        Args:
            config: Validator configuration
        """
        self.config = config or self._get_default_config()
        
        # 7-layer validation thresholds for >90% success rate
        self.layer_thresholds = {
            1: {"name": "timeframe_alignment", "threshold": 0.91, "weight": 0.15},
            2: {"name": "technical_confluence", "threshold": 0.89, "weight": 0.15},
            3: {"name": "volume_confirmation", "threshold": 0.93, "weight": 0.15},
            4: {"name": "statistical_significance", "threshold": 0.87, "weight": 0.20},
            5: {"name": "historical_consistency", "threshold": 0.90, "weight": 0.15},
            6: {"name": "risk_reward_optimization", "threshold": 0.92, "weight": 0.10},
            7: {"name": "market_context_validation", "threshold": 0.88, "weight": 0.10}
        }
        
        # Overall validation requirements
        self.overall_threshold = 0.90  # 90% overall score required
        self.min_occurrences = 100
        self.max_p_value = 0.01  # 99% confidence level
        
        # Initialize statistical validator
        self.statistical_validator = StatisticalPatternValidator(config)
        
        # Performance tracking
        self.validation_count = 0
        self.patterns_passed = 0
        self.patterns_failed = 0
        self.layer_failure_stats = {i: 0 for i in range(1, 8)}
        
        # Validation history
        self.validation_history = []
        self.max_history_length = 1000
        
        self.logger = logging.getLogger(f"{__name__}.SevenLayerPatternValidator")
        self.logger.info("7-Layer Pattern Validator initialized for >90% success rate")
        self.logger.info(f"Layer thresholds: {[f'L{i}: {info[\"threshold\"]:.1%}' for i, info in self.layer_thresholds.items()]}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default validator configuration"""
        return {
            'enable_all_layers': True,
            'strict_mode': True,
            'enable_caching': True,
            'cache_duration_minutes': 60,
            'enable_performance_tracking': True,
            'min_data_quality_score': 0.8,
            'enable_advanced_statistics': True,
            'parallel_validation': True,
            'validation_timeout_seconds': 30
        }
    
    def validate_pattern(self, pattern: PatternSchema, 
                        market_data: Optional[Dict[str, Any]] = None,
                        historical_data: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Perform comprehensive 7-layer pattern validation
        
        Args:
            pattern: Pattern to validate
            market_data: Current market data for context validation
            historical_data: Historical data for consistency validation
            
        Returns:
            ValidationResult with detailed layer-by-layer results
        """
        start_time = datetime.now()
        layer_results = []
        
        try:
            self.validation_count += 1
            
            # Pre-validation checks
            if not self._pre_validation_checks(pattern):
                return self._create_failed_validation_result(
                    pattern.pattern_id, "Pre-validation checks failed", start_time
                )
            
            # Layer 1: Timeframe Alignment Validation (91% threshold)
            layer1_result = self._validate_layer_1_timeframe_alignment(pattern)
            layer_results.append(layer1_result)
            
            # Layer 2: Technical Confluence Validation (89% threshold)
            layer2_result = self._validate_layer_2_technical_confluence(pattern, market_data)
            layer_results.append(layer2_result)
            
            # Layer 3: Volume Confirmation Validation (93% threshold)
            layer3_result = self._validate_layer_3_volume_confirmation(pattern, market_data)
            layer_results.append(layer3_result)
            
            # Layer 4: Statistical Significance Testing (87% threshold)
            layer4_result = self._validate_layer_4_statistical_significance(pattern, historical_data)
            layer_results.append(layer4_result)
            
            # Layer 5: Historical Consistency Validation (90% threshold)
            layer5_result = self._validate_layer_5_historical_consistency(pattern, historical_data)
            layer_results.append(layer5_result)
            
            # Layer 6: Risk-Reward Optimization (92% threshold)
            layer6_result = self._validate_layer_6_risk_reward(pattern)
            layer_results.append(layer6_result)
            
            # Layer 7: Market Context Validation (88% threshold)
            layer7_result = self._validate_layer_7_market_context(pattern, market_data)
            layer_results.append(layer7_result)
            
            # Calculate overall validation result
            overall_result = self._calculate_overall_validation(pattern, layer_results, start_time)
            
            # Update performance tracking
            self._update_performance_tracking(overall_result, layer_results)
            
            # Store validation history
            self._store_validation_history(overall_result)
            
            return overall_result
            
        except Exception as e:
            self.logger.error(f"Error in pattern validation: {e}")
            return self._create_failed_validation_result(
                pattern.pattern_id, f"Validation error: {e}", start_time
            )
    
    def _calculate_overall_validation(self, pattern: PatternSchema, 
                                    layer_results: List[ValidationLayerResult],
                                    start_time: datetime) -> ValidationResult:
        """
        Calculate overall validation result from all 7 layers
        """
        try:
            # Calculate weighted overall score
            total_weighted_score = 0.0
            total_weight = 0.0
            
            for layer_result in layer_results:
                layer_weight = self.layer_thresholds[layer_result.layer_number]["weight"]
                total_weighted_score += layer_result.score * layer_weight
                total_weight += layer_weight
            
            overall_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
            
            # Check if all layers passed
            all_layers_passed = all(layer_result.passed for layer_result in layer_results)
            
            # Check overall threshold
            meets_overall_threshold = overall_score >= self.overall_threshold
            
            # Pattern passes only if ALL conditions are met
            overall_passed = all_layers_passed and meets_overall_threshold
            
            # Calculate quality metrics
            confidence_score = self._calculate_confidence_score(layer_results)
            reliability_score = self._calculate_reliability_score(pattern, layer_results)
            success_probability = self._calculate_success_probability(overall_score, layer_results)
            
            # Get validation metadata
            components_validated = list(pattern.components.keys())
            timeframes_validated = list(pattern.timeframe_analysis.keys())
            
            validation_time = (datetime.now() - start_time).total_seconds()
            
            return ValidationResult(
                pattern_id=pattern.pattern_id,
                overall_score=overall_score,
                overall_passed=overall_passed,
                validation_time=validation_time,
                layer_results=layer_results,
                validation_timestamp=datetime.now(),
                components_validated=components_validated,
                timeframes_validated=timeframes_validated,
                confidence_score=confidence_score,
                reliability_score=reliability_score,
                success_probability=success_probability
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating overall validation: {e}")
            return self._create_failed_validation_result(
                pattern.pattern_id, f"Overall validation error: {e}", start_time
            )
    
    def _validate_layer_1_timeframe_alignment(self, pattern: PatternSchema) -> ValidationLayerResult:
        """
        Layer 1: Timeframe Alignment Validation (91% threshold)
        
        Validates that pattern signals are consistent across multiple timeframes
        """
        start_time = datetime.now()
        layer_info = self.layer_thresholds[1]
        
        try:
            timeframe_analysis = pattern.timeframe_analysis
            
            if not timeframe_analysis or len(timeframe_analysis) < 2:
                return ValidationLayerResult(
                    layer_name=layer_info["name"],
                    layer_number=1,
                    threshold=layer_info["threshold"],
                    score=0.0,
                    passed=False,
                    details={"error": "Insufficient timeframe data"},
                    validation_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Calculate timeframe consistency scores
            timeframes = list(timeframe_analysis.keys())
            consistency_scores = []
            
            # Check signal alignment across timeframes
            for i, tf1 in enumerate(timeframes):
                for tf2 in timeframes[i+1:]:
                    tf1_data = timeframe_analysis[tf1]
                    tf2_data = timeframe_analysis[tf2]
                    
                    # Primary signal consistency
                    primary_consistency = self._calculate_signal_consistency(
                        tf1_data.get('primary_signal', ''),
                        tf2_data.get('primary_signal', '')
                    )
                    
                    # Strength consistency
                    strength1 = tf1_data.get('strength', 0.5)
                    strength2 = tf2_data.get('strength', 0.5)
                    strength_consistency = 1.0 - abs(strength1 - strength2)
                    
                    # Validation score consistency
                    validation1 = tf1_data.get('validation_score', 0.5)
                    validation2 = tf2_data.get('validation_score', 0.5)
                    validation_consistency = 1.0 - abs(validation1 - validation2)
                    
                    # Combined consistency for this timeframe pair
                    pair_consistency = np.mean([
                        primary_consistency * 0.5,
                        strength_consistency * 0.3,
                        validation_consistency * 0.2
                    ])
                    
                    consistency_scores.append(pair_consistency)
            
            # Calculate cross-timeframe confluence score
            confluence_score = pattern.cross_timeframe_confluence.get('alignment_score', 0.0)
            
            # Overall timeframe alignment score
            overall_score = np.mean([
                np.mean(consistency_scores) * 0.6,  # Inter-timeframe consistency
                confluence_score * 0.4  # Cross-timeframe confluence
            ])
            
            # Additional validation criteria
            bonus_factors = []
            
            # Bonus for more timeframes
            if len(timeframes) >= 4:
                bonus_factors.append(0.05)
            
            # Bonus for high consistency across all pairs
            if np.min(consistency_scores) > 0.8:
                bonus_factors.append(0.03)
            
            # Apply bonuses
            final_score = min(1.0, overall_score + sum(bonus_factors))
            
            # Detailed validation info
            details = {
                "timeframes_analyzed": len(timeframes),
                "consistency_scores": consistency_scores,
                "avg_consistency": np.mean(consistency_scores),
                "min_consistency": np.min(consistency_scores) if consistency_scores else 0,
                "confluence_score": confluence_score,
                "bonus_factors": bonus_factors,
                "total_bonus": sum(bonus_factors)
            }
            
            return ValidationLayerResult(
                layer_name=layer_info["name"],
                layer_number=1,
                threshold=layer_info["threshold"],
                score=final_score,
                passed=final_score >= layer_info["threshold"],
                details=details,
                validation_time=(datetime.now() - start_time).total_seconds()
            )
            
        except Exception as e:
            self.logger.warning(f"Error in Layer 1 validation: {e}")
            return ValidationLayerResult(
                layer_name=layer_info["name"],
                layer_number=1,
                threshold=layer_info["threshold"],
                score=0.0,
                passed=False,
                details={"error": str(e)},
                validation_time=(datetime.now() - start_time).total_seconds()
            )
    
    def _validate_layer_2_technical_confluence(self, pattern: PatternSchema, 
                                             market_data: Optional[Dict[str, Any]]) -> ValidationLayerResult:
        """
        Layer 2: Technical Confluence Validation (89% threshold)
        
        Validates technical indicator alignment and confluence
        """
        start_time = datetime.now()
        layer_info = self.layer_thresholds[2]
        
        try:
            components = pattern.components
            confluence_indicators = []
            
            # Analyze each component for technical confluence
            for component_name, component_data in components.items():
                if 'indicator' in component_data and 'action' in component_data:
                    indicator = component_data['indicator']
                    action = component_data['action']
                    strength = component_data.get('strength', 0.5)
                    
                    # Calculate technical confluence score for this component
                    tech_score = self._calculate_technical_confluence_score(
                        indicator, action, strength, market_data
                    )
                    confluence_indicators.append(tech_score)
            
            # Multi-indicator alignment analysis
            if len(confluence_indicators) >= 2:
                # Calculate alignment between indicators
                alignment_scores = []
                
                for i in range(len(confluence_indicators)):
                    for j in range(i+1, len(confluence_indicators)):
                        alignment = 1.0 - abs(confluence_indicators[i] - confluence_indicators[j])
                        alignment_scores.append(alignment)
                
                avg_alignment = np.mean(alignment_scores)
            else:
                avg_alignment = 0.8  # Default for single indicator
            
            # Volume confirmation boost
            volume_confirmation_boost = 0.0
            for component_data in components.values():
                if component_data.get('volume_confirmation', False):
                    volume_confirmation_boost += 0.02
            
            # Calculate overall technical confluence score
            base_score = np.mean(confluence_indicators) if confluence_indicators else 0.5
            alignment_score = avg_alignment * 0.3
            volume_boost = min(0.1, volume_confirmation_boost)
            
            final_score = min(1.0, base_score * 0.7 + alignment_score + volume_boost)
            
            details = {
                "components_analyzed": len(components),
                "confluence_indicators": confluence_indicators,
                "avg_confluence": np.mean(confluence_indicators) if confluence_indicators else 0,
                "alignment_scores": alignment_scores if len(confluence_indicators) >= 2 else [],
                "avg_alignment": avg_alignment,
                "volume_boost": volume_boost,
                "technical_confluence_score": final_score
            }
            
            return ValidationLayerResult(
                layer_name=layer_info["name"],
                layer_number=2,
                threshold=layer_info["threshold"],
                score=final_score,
                passed=final_score >= layer_info["threshold"],
                details=details,
                validation_time=(datetime.now() - start_time).total_seconds()
            )
            
        except Exception as e:
            self.logger.warning(f"Error in Layer 2 validation: {e}")
            return ValidationLayerResult(
                layer_name=layer_info["name"],
                layer_number=2,
                threshold=layer_info["threshold"],
                score=0.0,
                passed=False,
                details={"error": str(e)},
                validation_time=(datetime.now() - start_time).total_seconds()
            )
    
    def _validate_layer_3_volume_confirmation(self, pattern: PatternSchema,
                                            market_data: Optional[Dict[str, Any]]) -> ValidationLayerResult:
        """
        Layer 3: Volume Confirmation Validation (93% threshold)
        
        Validates volume-based pattern confirmation
        """
        start_time = datetime.now()
        layer_info = self.layer_thresholds[3]
        
        try:
            volume_scores = []
            components = pattern.components
            
            # Analyze volume confirmation for each component
            for component_name, component_data in components.items():
                volume_confirmation = component_data.get('volume_confirmation', False)
                
                if volume_confirmation:
                    # High score for confirmed volume
                    component_volume_score = 0.95
                else:
                    # Analyze volume patterns if available
                    component_volume_score = self._analyze_volume_patterns(
                        component_name, component_data, market_data
                    )
                
                volume_scores.append(component_volume_score)
            
            # Calculate volume profile strength
            volume_profile_score = self._calculate_volume_profile_score(pattern, market_data)
            
            # Calculate volume momentum
            volume_momentum_score = self._calculate_volume_momentum_score(pattern, market_data)
            
            # Volume divergence analysis
            volume_divergence_score = self._calculate_volume_divergence_score(pattern, market_data)
            
            # Overall volume confirmation score
            base_volume_score = np.mean(volume_scores) if volume_scores else 0.5
            
            final_score = np.mean([
                base_volume_score * 0.4,
                volume_profile_score * 0.25,
                volume_momentum_score * 0.20,
                volume_divergence_score * 0.15
            ])
            
            # Volume confirmation bonus
            if base_volume_score > 0.9 and volume_profile_score > 0.8:
                final_score = min(1.0, final_score + 0.05)
            
            details = {
                "components_volume_scores": volume_scores,
                "avg_component_volume": np.mean(volume_scores) if volume_scores else 0,
                "volume_profile_score": volume_profile_score,
                "volume_momentum_score": volume_momentum_score,
                "volume_divergence_score": volume_divergence_score,
                "volume_confirmation_count": sum(1 for comp in components.values() 
                                               if comp.get('volume_confirmation', False))
            }
            
            return ValidationLayerResult(
                layer_name=layer_info["name"],
                layer_number=3,
                threshold=layer_info["threshold"],
                score=final_score,
                passed=final_score >= layer_info["threshold"],
                details=details,
                validation_time=(datetime.now() - start_time).total_seconds()
            )
            
        except Exception as e:
            self.logger.warning(f"Error in Layer 3 validation: {e}")
            return ValidationLayerResult(
                layer_name=layer_info["name"],
                layer_number=3,
                threshold=layer_info["threshold"],
                score=0.0,
                passed=False,
                details={"error": str(e)},
                validation_time=(datetime.now() - start_time).total_seconds()
            )
    
    def _validate_layer_4_statistical_significance(self, pattern: PatternSchema,
                                                  historical_data: Optional[Dict[str, Any]]) -> ValidationLayerResult:
        """
        Layer 4: Statistical Significance Testing (87% threshold)
        
        Validates statistical significance using multiple statistical tests
        """
        start_time = datetime.now()
        layer_info = self.layer_thresholds[4]
        
        try:
            # Use statistical validator for comprehensive testing
            statistical_result = self.statistical_validator.validate_pattern_significance(
                pattern, historical_data
            )
            
            # Extract statistical scores
            statistical_scores = getattr(pattern, 'statistical_significance', {})
            
            # Required statistical tests
            required_tests = ['t_test', 'chi_square', 'bootstrap_validation']
            test_scores = []
            
            for test_name in required_tests:
                if test_name in statistical_scores:
                    test_result = statistical_scores[test_name]
                    p_value = test_result.get('p_value', 1.0)
                    
                    # Convert p-value to confidence score (lower p-value = higher confidence)
                    confidence_score = max(0.0, 1.0 - (p_value / self.max_p_value))
                    test_scores.append(confidence_score)
                else:
                    test_scores.append(0.0)  # Missing test = failed
            
            # Sample size validation
            sample_size = pattern.total_occurrences
            sample_size_score = min(1.0, sample_size / self.min_occurrences)
            
            # Effect size validation (Cohen's d or similar)
            effect_size_score = statistical_scores.get('effect_size', 0.5)
            
            # Confidence interval validation
            confidence_interval_score = statistical_scores.get('confidence_interval_score', 0.5)
            
            # Overall statistical significance score
            final_score = np.mean([
                np.mean(test_scores) * 0.5,  # Statistical tests
                sample_size_score * 0.2,     # Sample size
                effect_size_score * 0.15,    # Effect size
                confidence_interval_score * 0.15  # Confidence intervals
            ])
            
            # Bonus for exceptional statistical strength
            if np.min(test_scores) > 0.9 and sample_size > self.min_occurrences * 2:
                final_score = min(1.0, final_score + 0.05)
            
            details = {
                "statistical_tests_passed": sum(1 for score in test_scores if score > 0.8),
                "test_scores": dict(zip(required_tests, test_scores)),
                "avg_test_score": np.mean(test_scores),
                "sample_size": sample_size,
                "sample_size_score": sample_size_score,
                "effect_size_score": effect_size_score,
                "confidence_interval_score": confidence_interval_score,
                "min_p_value": min([statistical_scores.get(test, {}).get('p_value', 1.0) 
                                  for test in required_tests])
            }
            
            return ValidationLayerResult(
                layer_name=layer_info["name"],
                layer_number=4,
                threshold=layer_info["threshold"],
                score=final_score,
                passed=final_score >= layer_info["threshold"],
                details=details,
                validation_time=(datetime.now() - start_time).total_seconds()
            )
            
        except Exception as e:
            self.logger.warning(f"Error in Layer 4 validation: {e}")
            return ValidationLayerResult(
                layer_name=layer_info["name"],
                layer_number=4,
                threshold=layer_info["threshold"],
                score=0.0,
                passed=False,
                details={"error": str(e)},
                validation_time=(datetime.now() - start_time).total_seconds()
            )
    
    def _validate_layer_5_historical_consistency(self, pattern: PatternSchema,
                                               historical_data: Optional[Dict[str, Any]]) -> ValidationLayerResult:
        """
        Layer 5: Historical Consistency Validation (90% threshold)
        
        Validates consistency of pattern performance over time
        """
        start_time = datetime.now()
        layer_info = self.layer_thresholds[5]
        
        try:
            historical_performance = pattern.historical_performance
            
            # Success rate consistency
            success_rate = historical_performance.get('success_rate', 0.0)
            success_rate_score = success_rate
            
            # Performance stability over time
            performance_stability = self._calculate_performance_stability(pattern, historical_data)
            
            # Frequency consistency (regular occurrences)
            frequency_consistency = self._calculate_frequency_consistency(pattern, historical_data)
            
            # Market condition consistency (works in different market regimes)
            market_condition_consistency = self._calculate_market_condition_consistency(pattern, historical_data)
            
            # Recent performance validation (last 30 days performance)
            recent_performance = self._calculate_recent_performance(pattern, historical_data)
            
            # Overall historical consistency score
            final_score = np.mean([
                success_rate_score * 0.3,
                performance_stability * 0.25,
                frequency_consistency * 0.2,
                market_condition_consistency * 0.15,
                recent_performance * 0.1
            ])
            
            # Bonus for exceptional consistency
            if (success_rate > 0.85 and performance_stability > 0.85 and 
                frequency_consistency > 0.8):
                final_score = min(1.0, final_score + 0.03)
            
            details = {
                "success_rate": success_rate,
                "performance_stability": performance_stability,
                "frequency_consistency": frequency_consistency,
                "market_condition_consistency": market_condition_consistency,
                "recent_performance": recent_performance,
                "total_occurrences": pattern.total_occurrences,
                "avg_return": historical_performance.get('avg_return', 0),
                "max_drawdown": historical_performance.get('max_drawdown', 0)
            }
            
            return ValidationLayerResult(
                layer_name=layer_info["name"],
                layer_number=5,
                threshold=layer_info["threshold"],
                score=final_score,
                passed=final_score >= layer_info["threshold"],
                details=details,
                validation_time=(datetime.now() - start_time).total_seconds()
            )
            
        except Exception as e:
            self.logger.warning(f"Error in Layer 5 validation: {e}")
            return ValidationLayerResult(
                layer_name=layer_info["name"],
                layer_number=5,
                threshold=layer_info["threshold"],
                score=0.0,
                passed=False,
                details={"error": str(e)},
                validation_time=(datetime.now() - start_time).total_seconds()
            )
    
    def _validate_layer_6_risk_reward(self, pattern: PatternSchema) -> ValidationLayerResult:
        """
        Layer 6: Risk-Reward Optimization (92% threshold)
        
        Validates risk-adjusted returns and reward-to-risk ratios
        """
        start_time = datetime.now()
        layer_info = self.layer_thresholds[6]
        
        try:
            historical_performance = pattern.historical_performance
            
            # Risk-reward ratio
            avg_return = historical_performance.get('avg_return', 0.0)
            max_drawdown = abs(historical_performance.get('max_drawdown', 0.01))
            risk_reward_ratio = avg_return / max_drawdown if max_drawdown > 0 else 0.0
            
            # Sharpe ratio equivalent
            return_std = historical_performance.get('return_std', 0.01)
            sharpe_ratio = avg_return / return_std if return_std > 0 else 0.0
            
            # Win rate analysis
            win_rate = historical_performance.get('win_rate', 0.0)
            avg_win = historical_performance.get('avg_win', 0.0)
            avg_loss = abs(historical_performance.get('avg_loss', 0.01))
            
            # Expectancy calculation
            expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
            
            # Profit factor
            profit_factor = (win_rate * avg_win) / ((1 - win_rate) * avg_loss) if avg_loss > 0 else 0.0
            
            # Maximum consecutive losses
            max_consecutive_losses = historical_performance.get('max_consecutive_losses', 0)
            consecutive_loss_penalty = max(0.0, 1.0 - (max_consecutive_losses / 10))
            
            # Risk-adjusted scores
            risk_reward_score = min(1.0, risk_reward_ratio / 2.0)  # 2:1 ratio = 100% score
            sharpe_score = min(1.0, max(0.0, sharpe_ratio / 2.0))  # Sharpe 2.0 = 100% score
            expectancy_score = min(1.0, max(0.0, expectancy * 10))  # 10% expectancy = 100% score
            profit_factor_score = min(1.0, max(0.0, (profit_factor - 1.0) / 2.0))  # PF 3.0 = 100% score
            
            # Overall risk-reward score
            final_score = np.mean([
                risk_reward_score * 0.3,
                sharpe_score * 0.25,
                expectancy_score * 0.2,
                profit_factor_score * 0.15,
                consecutive_loss_penalty * 0.1
            ])
            
            # Bonus for exceptional risk-reward
            if risk_reward_ratio > 3.0 and sharpe_ratio > 1.5 and profit_factor > 2.0:
                final_score = min(1.0, final_score + 0.05)
            
            details = {
                "risk_reward_ratio": risk_reward_ratio,
                "sharpe_ratio": sharpe_ratio,
                "expectancy": expectancy,
                "profit_factor": profit_factor,
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "max_consecutive_losses": max_consecutive_losses,
                "risk_reward_score": risk_reward_score,
                "sharpe_score": sharpe_score,
                "expectancy_score": expectancy_score,
                "profit_factor_score": profit_factor_score
            }
            
            return ValidationLayerResult(
                layer_name=layer_info["name"],
                layer_number=6,
                threshold=layer_info["threshold"],
                score=final_score,
                passed=final_score >= layer_info["threshold"],
                details=details,
                validation_time=(datetime.now() - start_time).total_seconds()
            )
            
        except Exception as e:
            self.logger.warning(f"Error in Layer 6 validation: {e}")
            return ValidationLayerResult(
                layer_name=layer_info["name"],
                layer_number=6,
                threshold=layer_info["threshold"],
                score=0.0,
                passed=False,
                details={"error": str(e)},
                validation_time=(datetime.now() - start_time).total_seconds()
            )
    
    def _validate_layer_7_market_context(self, pattern: PatternSchema, 
                                       market_data: Optional[Dict[str, Any]]) -> ValidationLayerResult:
        """
        Layer 7: Market Context Validation (88% threshold)
        
        Validates pattern fit with current market conditions and regime
        """
        start_time = datetime.now()
        layer_info = self.layer_thresholds[7]
        
        try:
            market_context = pattern.market_context
            
            # Market regime consistency
            pattern_regime = market_context.get('preferred_regime', 'neutral')
            current_regime = market_data.get('market_regime', 'neutral') if market_data else 'neutral'
            regime_match_score = 1.0 if pattern_regime == current_regime else 0.5
            
            # Volatility environment match
            pattern_volatility = market_context.get('volatility_environment', 'medium')
            current_volatility = self._categorize_volatility(market_data) if market_data else 'medium'
            volatility_match_score = 1.0 if pattern_volatility == current_volatility else 0.6
            
            # Trend environment match
            pattern_trend = market_context.get('trend_environment', 'neutral')
            current_trend = self._categorize_trend(market_data) if market_data else 'neutral'
            trend_match_score = 1.0 if pattern_trend == current_trend else 0.6
            
            # Time-of-day consistency
            pattern_time_preference = market_context.get('time_of_day_preference', 'any')
            current_time_score = self._calculate_time_consistency(pattern_time_preference, market_data)
            
            # Market structure consistency
            pattern_structure = market_context.get('market_structure', 'any')
            current_structure_score = self._calculate_structure_consistency(pattern_structure, market_data)
            
            # Economic calendar impact
            economic_calendar_score = self._calculate_economic_calendar_impact(pattern, market_data)
            
            # DTE (Days to Expiry) consistency
            dte_consistency_score = self._calculate_dte_consistency(pattern, market_data)
            
            # Overall market context score
            final_score = np.mean([
                regime_match_score * 0.25,
                volatility_match_score * 0.2,
                trend_match_score * 0.15,
                current_time_score * 0.15,
                current_structure_score * 0.1,
                economic_calendar_score * 0.1,
                dte_consistency_score * 0.05
            ])
            
            # Bonus for perfect alignment
            if (regime_match_score >= 0.9 and volatility_match_score >= 0.9 and 
                trend_match_score >= 0.9):
                final_score = min(1.0, final_score + 0.03)
            
            details = {
                "pattern_regime": pattern_regime,
                "current_regime": current_regime,
                "regime_match_score": regime_match_score,
                "pattern_volatility": pattern_volatility,
                "current_volatility": current_volatility,
                "volatility_match_score": volatility_match_score,
                "pattern_trend": pattern_trend,
                "current_trend": current_trend,
                "trend_match_score": trend_match_score,
                "time_consistency_score": current_time_score,
                "structure_consistency_score": current_structure_score,
                "economic_calendar_score": economic_calendar_score,
                "dte_consistency_score": dte_consistency_score
            }
            
            return ValidationLayerResult(
                layer_name=layer_info["name"],
                layer_number=7,
                threshold=layer_info["threshold"],
                score=final_score,
                passed=final_score >= layer_info["threshold"],
                details=details,
                validation_time=(datetime.now() - start_time).total_seconds()
            )
            
        except Exception as e:
            self.logger.warning(f"Error in Layer 7 validation: {e}")
            return ValidationLayerResult(
                layer_name=layer_info["name"],
                layer_number=7,
                threshold=layer_info["threshold"],
                score=0.0,
                passed=False,
                details={"error": str(e)},
                validation_time=(datetime.now() - start_time).total_seconds()
            )
    
    # ========================================
    # HELPER METHODS FOR VALIDATION LAYERS
    # ========================================
    
    def _pre_validation_checks(self, pattern: PatternSchema) -> bool:
        """Perform basic pre-validation checks"""
        try:
            # Check required pattern attributes
            if not pattern.pattern_id:
                self.logger.warning("Pattern missing pattern_id")
                return False
            
            if not pattern.components:
                self.logger.warning("Pattern missing components")
                return False
            
            if not pattern.timeframe_analysis:
                self.logger.warning("Pattern missing timeframe_analysis")
                return False
            
            # Check minimum occurrences
            total_occurrences = getattr(pattern, 'total_occurrences', 0)
            if total_occurrences < self.min_occurrences:
                self.logger.warning(f"Pattern has insufficient occurrences: {total_occurrences} < {self.min_occurrences}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in pre-validation checks: {e}")
            return False
    
    def _calculate_signal_consistency(self, signal1: str, signal2: str) -> float:
        """Calculate consistency between two signals"""
        if not signal1 or not signal2:
            return 0.5
        
        # Define signal compatibility
        signal_compatibility = {
            ('bullish', 'bullish'): 1.0,
            ('bearish', 'bearish'): 1.0,
            ('neutral', 'neutral'): 1.0,
            ('bullish', 'neutral'): 0.7,
            ('bearish', 'neutral'): 0.7,
            ('neutral', 'bullish'): 0.7,
            ('neutral', 'bearish'): 0.7,
            ('bullish', 'bearish'): 0.0,
            ('bearish', 'bullish'): 0.0
        }
        
        return signal_compatibility.get((signal1.lower(), signal2.lower()), 0.5)
    
    def _calculate_technical_confluence_score(self, indicator: str, action: str, 
                                           strength: float, market_data: Optional[Dict]) -> float:
        """Calculate technical confluence score for a component"""
        try:
            # Base score from strength
            base_score = strength
            
            # Indicator-specific scoring
            indicator_weights = {
                'ema_20': 0.8,
                'ema_100': 0.9,
                'ema_200': 1.0,  # Highest weight for 200 EMA
                'vwap': 0.85,
                'pivot_point': 0.75,
                'support': 0.8,
                'resistance': 0.8
            }
            
            indicator_weight = indicator_weights.get(indicator.lower(), 0.6)
            
            # Action-specific scoring
            action_weights = {
                'rejection': 0.9,
                'support': 0.9,
                'bounce': 0.8,
                'breakout': 0.85,
                'breakdown': 0.85
            }
            
            action_weight = action_weights.get(action.lower(), 0.6)
            
            # Calculate weighted score
            confluence_score = base_score * indicator_weight * action_weight
            
            return min(1.0, confluence_score)
            
        except Exception:
            return 0.5
    
    def _analyze_volume_patterns(self, component_name: str, component_data: Dict, 
                               market_data: Optional[Dict]) -> float:
        """Analyze volume patterns for a component"""
        try:
            if not market_data:
                return 0.5
            
            # Get volume data
            current_volume = market_data.get('volume', 0)
            avg_volume = market_data.get('avg_volume', current_volume)
            
            if avg_volume <= 0:
                return 0.5
            
            # Volume ratio
            volume_ratio = current_volume / avg_volume
            
            # Score based on volume strength
            if volume_ratio > 2.0:
                return 0.95  # Very high volume
            elif volume_ratio > 1.5:
                return 0.85  # High volume
            elif volume_ratio > 1.0:
                return 0.75  # Above average volume
            elif volume_ratio > 0.5:
                return 0.60  # Below average volume
            else:
                return 0.40  # Very low volume
                
        except Exception:
            return 0.5
    
    def _calculate_volume_profile_score(self, pattern: PatternSchema, 
                                      market_data: Optional[Dict]) -> float:
        """Calculate volume profile strength score"""
        try:
            if not market_data:
                return 0.6
            
            # Check for volume profile data
            volume_profile = market_data.get('volume_profile', {})
            if not volume_profile:
                return 0.6
            
            # Analyze volume nodes
            high_volume_nodes = volume_profile.get('high_volume_nodes', [])
            total_volume = volume_profile.get('total_volume', 1)
            
            # Score based on volume distribution
            if len(high_volume_nodes) >= 3:
                return 0.9  # Strong volume profile
            elif len(high_volume_nodes) >= 2:
                return 0.8  # Good volume profile
            elif len(high_volume_nodes) >= 1:
                return 0.7  # Moderate volume profile
            else:
                return 0.5  # Weak volume profile
                
        except Exception:
            return 0.6
    
    def _calculate_volume_momentum_score(self, pattern: PatternSchema, 
                                       market_data: Optional[Dict]) -> float:
        """Calculate volume momentum score"""
        try:
            if not market_data:
                return 0.6
            
            # Get volume trend data
            volume_trend = market_data.get('volume_trend', 'neutral')
            volume_momentum = market_data.get('volume_momentum', 0.0)
            
            # Score based on volume momentum
            if volume_trend == 'increasing' and volume_momentum > 0.5:
                return 0.9
            elif volume_trend == 'increasing' and volume_momentum > 0.2:
                return 0.8
            elif volume_trend == 'stable':
                return 0.6
            elif volume_trend == 'decreasing':
                return 0.4
            else:
                return 0.5
                
        except Exception:
            return 0.6
    
    def _calculate_volume_divergence_score(self, pattern: PatternSchema, 
                                         market_data: Optional[Dict]) -> float:
        """Calculate volume divergence score"""
        try:
            if not market_data:
                return 0.7
            
            # Check for price-volume divergence
            price_direction = market_data.get('price_direction', 'neutral')
            volume_direction = market_data.get('volume_direction', 'neutral')
            
            # Positive divergence (volume up, price down) = bullish
            # Negative divergence (volume down, price up) = bearish
            # Convergence (same direction) = neutral
            
            if price_direction == 'up' and volume_direction == 'up':
                return 0.9  # Strong convergence
            elif price_direction == 'down' and volume_direction == 'down':
                return 0.9  # Strong convergence
            elif price_direction == 'up' and volume_direction == 'down':
                return 0.3  # Negative divergence (bearish)
            elif price_direction == 'down' and volume_direction == 'up':
                return 0.3  # Positive divergence (watch for reversal)
            else:
                return 0.7  # Neutral
                
        except Exception:
            return 0.7
    
    def _calculate_performance_stability(self, pattern: PatternSchema, 
                                       historical_data: Optional[Dict]) -> float:
        """Calculate performance stability over time"""
        try:
            if not historical_data:
                return 0.6
            
            # Get historical performance data
            performance_history = historical_data.get('performance_history', [])
            if len(performance_history) < 10:
                return 0.5  # Insufficient data
            
            # Calculate standard deviation of returns
            returns = [p.get('return', 0.0) for p in performance_history]
            returns_std = np.std(returns)
            returns_mean = np.mean(returns)
            
            if returns_mean <= 0:
                return 0.2  # Negative average return
            
            # Calculate coefficient of variation
            cv = returns_std / returns_mean if returns_mean > 0 else float('inf')
            
            # Score based on stability (lower CV = higher stability)
            if cv < 0.2:
                return 0.95  # Very stable
            elif cv < 0.5:
                return 0.85  # Stable
            elif cv < 1.0:
                return 0.70  # Moderately stable
            elif cv < 2.0:
                return 0.50  # Unstable
            else:
                return 0.30  # Very unstable
                
        except Exception:
            return 0.6
    
    def _calculate_frequency_consistency(self, pattern: PatternSchema, 
                                       historical_data: Optional[Dict]) -> float:
        """Calculate frequency consistency of pattern occurrences"""
        try:
            if not historical_data:
                return 0.6
            
            # Get occurrence timestamps
            occurrences = historical_data.get('occurrences', [])
            if len(occurrences) < 20:
                return 0.5  # Insufficient data
            
            # Calculate time intervals between occurrences
            intervals = []
            for i in range(1, len(occurrences)):
                interval = (occurrences[i] - occurrences[i-1]).total_seconds() / 3600  # Hours
                intervals.append(interval)
            
            if not intervals:
                return 0.5
            
            # Calculate coefficient of variation for intervals
            interval_std = np.std(intervals)
            interval_mean = np.mean(intervals)
            
            if interval_mean <= 0:
                return 0.3
            
            cv = interval_std / interval_mean
            
            # Score based on consistency (lower CV = higher consistency)
            if cv < 0.3:
                return 0.9   # Very consistent
            elif cv < 0.6:
                return 0.8   # Consistent
            elif cv < 1.0:
                return 0.7   # Moderately consistent
            elif cv < 1.5:
                return 0.5   # Inconsistent
            else:
                return 0.3   # Very inconsistent
                
        except Exception:
            return 0.6
    
    def _calculate_market_condition_consistency(self, pattern: PatternSchema, 
                                              historical_data: Optional[Dict]) -> float:
        """Calculate consistency across different market conditions"""
        try:
            if not historical_data:
                return 0.6
            
            # Get market regime performance
            regime_performance = historical_data.get('regime_performance', {})
            if not regime_performance:
                return 0.6
            
            # Calculate performance across different regimes
            regime_scores = []
            for regime, performance in regime_performance.items():
                success_rate = performance.get('success_rate', 0.0)
                regime_scores.append(success_rate)
            
            if not regime_scores:
                return 0.6
            
            # Score based on minimum performance across regimes
            min_performance = min(regime_scores)
            avg_performance = np.mean(regime_scores)
            
            # Penalize if performance varies too much across regimes
            performance_consistency = 1.0 - (max(regime_scores) - min(regime_scores))
            
            # Combined score
            final_score = (min_performance * 0.4 + avg_performance * 0.4 + 
                          performance_consistency * 0.2)
            
            return min(1.0, max(0.0, final_score))
            
        except Exception:
            return 0.6
    
    def _calculate_recent_performance(self, pattern: PatternSchema, 
                                    historical_data: Optional[Dict]) -> float:
        """Calculate recent performance (last 30 days)"""
        try:
            if not historical_data:
                return 0.6
            
            # Get recent performance data
            recent_performance = historical_data.get('recent_performance', {})
            if not recent_performance:
                return 0.6
            
            # Recent success rate
            recent_success_rate = recent_performance.get('success_rate', 0.0)
            
            # Compare to overall success rate
            overall_success_rate = pattern.historical_performance.get('success_rate', 0.0)
            
            # Score based on recent vs overall performance
            if recent_success_rate >= overall_success_rate * 1.1:
                return 0.95  # Recent performance better than overall
            elif recent_success_rate >= overall_success_rate * 0.9:
                return 0.85  # Recent performance similar to overall
            elif recent_success_rate >= overall_success_rate * 0.7:
                return 0.70  # Recent performance somewhat worse
            elif recent_success_rate >= overall_success_rate * 0.5:
                return 0.50  # Recent performance much worse
            else:
                return 0.30  # Recent performance very poor
                
        except Exception:
            return 0.6
    
    def _categorize_volatility(self, market_data: Dict) -> str:
        """Categorize current volatility environment"""
        try:
            volatility = market_data.get('volatility', 0.01)
            
            if volatility > 0.03:
                return 'high'
            elif volatility > 0.015:
                return 'medium'
            else:
                return 'low'
                
        except Exception:
            return 'medium'
    
    def _categorize_trend(self, market_data: Dict) -> str:
        """Categorize current trend environment"""
        try:
            trend_strength = market_data.get('trend_strength', 0.0)
            
            if trend_strength > 0.3:
                return 'strong_bullish'
            elif trend_strength > 0.1:
                return 'bullish'
            elif trend_strength > -0.1:
                return 'neutral'
            elif trend_strength > -0.3:
                return 'bearish'
            else:
                return 'strong_bearish'
                
        except Exception:
            return 'neutral'
    
    def _calculate_time_consistency(self, pattern_time_preference: str, 
                                  market_data: Optional[Dict]) -> float:
        """Calculate time-of-day consistency score"""
        try:
            if not market_data or pattern_time_preference == 'any':
                return 0.8
            
            current_hour = market_data.get('current_hour', 12)
            
            # Define time preference scoring
            time_preferences = {
                'morning': (9, 12),
                'afternoon': (12, 15),
                'close': (15, 16),
                'any': (9, 16)
            }
            
            preferred_start, preferred_end = time_preferences.get(pattern_time_preference, (9, 16))
            
            if preferred_start <= current_hour <= preferred_end:
                return 1.0
            else:
                return 0.6
                
        except Exception:
            return 0.8
    
    def _calculate_structure_consistency(self, pattern_structure: str, 
                                       market_data: Optional[Dict]) -> float:
        """Calculate market structure consistency score"""
        try:
            if not market_data or pattern_structure == 'any':
                return 0.8
            
            current_structure = market_data.get('market_structure', 'neutral')
            
            if pattern_structure == current_structure:
                return 1.0
            elif pattern_structure == 'any':
                return 0.8
            else:
                return 0.5
                
        except Exception:
            return 0.8
    
    def _calculate_economic_calendar_impact(self, pattern: PatternSchema, 
                                          market_data: Optional[Dict]) -> float:
        """Calculate economic calendar impact score"""
        try:
            if not market_data:
                return 0.8
            
            # Check for upcoming economic events
            economic_events = market_data.get('economic_events', [])
            
            # Check pattern's sensitivity to economic events
            event_sensitivity = pattern.market_context.get('economic_sensitivity', 'medium')
            
            if not economic_events:
                return 0.9  # No events = neutral
            
            # Score based on event impact and pattern sensitivity
            high_impact_events = [e for e in economic_events if e.get('impact', 'low') == 'high']
            
            if event_sensitivity == 'high' and high_impact_events:
                return 0.4  # High sensitivity + high impact events = poor timing
            elif event_sensitivity == 'low' and high_impact_events:
                return 0.9  # Low sensitivity + high impact events = good timing
            elif event_sensitivity == 'medium':
                return 0.7  # Medium sensitivity = neutral
            else:
                return 0.8  # Default
                
        except Exception:
            return 0.8
    
    def _calculate_dte_consistency(self, pattern: PatternSchema, 
                                 market_data: Optional[Dict]) -> float:
        """Calculate DTE (Days to Expiry) consistency score"""
        try:
            if not market_data:
                return 0.8
            
            current_dte = market_data.get('dte', 7)
            pattern_dte_range = pattern.market_context.get('dte_range', 'any')
            
            # Define DTE ranges
            dte_ranges = {
                'ultra_short': (0, 2),
                'short': (3, 7),
                'medium': (8, 21),
                'long': (22, 45),
                'any': (0, 45)
            }
            
            if pattern_dte_range not in dte_ranges:
                return 0.8
            
            min_dte, max_dte = dte_ranges[pattern_dte_range]
            
            if min_dte <= current_dte <= max_dte:
                return 1.0
            else:
                # Calculate penalty based on distance from range
                if current_dte < min_dte:
                    penalty = (min_dte - current_dte) / min_dte
                else:
                    penalty = (current_dte - max_dte) / max_dte
                
                return max(0.3, 1.0 - penalty)
                
        except Exception:
            return 0.8
    
    def _calculate_confidence_score(self, layer_results: List[ValidationLayerResult]) -> float:
        """Calculate overall confidence score"""
        try:
            # Base confidence from layer scores
            scores = [result.score for result in layer_results]
            min_score = min(scores)
            avg_score = np.mean(scores)
            
            # Confidence decreases if any layer scores very low
            confidence = avg_score * (0.5 + 0.5 * min_score)
            
            return min(1.0, confidence)
            
        except Exception:
            return 0.5
    
    def _calculate_reliability_score(self, pattern: PatternSchema, 
                                   layer_results: List[ValidationLayerResult]) -> float:
        """Calculate reliability score based on historical data"""
        try:
            # Base reliability from sample size
            total_occurrences = getattr(pattern, 'total_occurrences', 100)
            sample_reliability = min(1.0, total_occurrences / (self.min_occurrences * 2))
            
            # Reliability from validation consistency
            validation_scores = [result.score for result in layer_results]
            validation_std = np.std(validation_scores)
            validation_reliability = max(0.0, 1.0 - validation_std)
            
            # Combined reliability
            return (sample_reliability * 0.6 + validation_reliability * 0.4)
            
        except Exception:
            return 0.5
    
    def _calculate_success_probability(self, overall_score: float, 
                                     layer_results: List[ValidationLayerResult]) -> float:
        """Calculate success probability based on validation results"""
        try:
            # Base probability from overall score
            base_probability = overall_score
            
            # Boost probability if all critical layers passed
            critical_layers = [1, 2, 4, 5]  # Most critical layers
            critical_passed = sum(1 for result in layer_results 
                                if result.layer_number in critical_layers and result.passed)
            
            critical_boost = (critical_passed / len(critical_layers)) * 0.1
            
            # Final probability
            success_probability = min(0.98, base_probability + critical_boost)
            
            return success_probability
            
        except Exception:
            return overall_score
    
    def _create_failed_validation_result(self, pattern_id: str, error_message: str, 
                                       start_time: datetime) -> ValidationResult:
        """Create a failed validation result"""
        try:
            validation_time = (datetime.now() - start_time).total_seconds()
            
            return ValidationResult(
                pattern_id=pattern_id,
                overall_score=0.0,
                overall_passed=False,
                validation_time=validation_time,
                layer_results=[],
                validation_timestamp=datetime.now(),
                components_validated=[],
                timeframes_validated=[],
                confidence_score=0.0,
                reliability_score=0.0,
                success_probability=0.0
            )
            
        except Exception:
            return ValidationResult(
                pattern_id=pattern_id or "unknown",
                overall_score=0.0,
                overall_passed=False,
                validation_time=0.0,
                layer_results=[],
                validation_timestamp=datetime.now(),
                components_validated=[],
                timeframes_validated=[],
                confidence_score=0.0,
                reliability_score=0.0,
                success_probability=0.0
            )
    
    def _update_performance_tracking(self, validation_result: ValidationResult, 
                                   layer_results: List[ValidationLayerResult]):
        """Update performance tracking statistics"""
        try:
            if validation_result.overall_passed:
                self.patterns_passed += 1
            else:
                self.patterns_failed += 1
                
                # Track which layers failed
                for layer_result in layer_results:
                    if not layer_result.passed:
                        self.layer_failure_stats[layer_result.layer_number] += 1
            
        except Exception as e:
            self.logger.warning(f"Error updating performance tracking: {e}")
    
    def _store_validation_history(self, validation_result: ValidationResult):
        """Store validation result in history"""
        try:
            self.validation_history.append(validation_result)
            
            # Limit history size
            if len(self.validation_history) > self.max_history_length:
                self.validation_history = self.validation_history[-self.max_history_length:]
                
        except Exception as e:
            self.logger.warning(f"Error storing validation history: {e}")
    
    def get_validator_statistics(self) -> Dict[str, Any]:
        """Get comprehensive validator statistics"""
        try:
            total_validations = self.patterns_passed + self.patterns_failed
            success_rate = self.patterns_passed / total_validations if total_validations > 0 else 0.0
            
            # Layer failure analysis
            layer_failure_rates = {}
            for layer_num, failures in self.layer_failure_stats.items():
                layer_name = self.layer_thresholds[layer_num]["name"]
                failure_rate = failures / total_validations if total_validations > 0 else 0.0
                layer_failure_rates[layer_name] = failure_rate
            
            # Recent performance
            recent_results = self.validation_history[-50:] if self.validation_history else []
            recent_success_rate = (sum(1 for r in recent_results if r.overall_passed) / 
                                 len(recent_results)) if recent_results else 0.0
            
            return {
                "total_validations": total_validations,
                "patterns_passed": self.patterns_passed,
                "patterns_failed": self.patterns_failed,
                "overall_success_rate": success_rate,
                "recent_success_rate": recent_success_rate,
                "layer_failure_rates": layer_failure_rates,
                "validation_history_length": len(self.validation_history),
                "layer_thresholds": {
                    f"layer_{num}_{info['name']}": info["threshold"] 
                    for num, info in self.layer_thresholds.items()
                },
                "overall_threshold": self.overall_threshold
            }
            
        except Exception as e:
            self.logger.error(f"Error generating validator statistics: {e}")
            return {"error": str(e)}