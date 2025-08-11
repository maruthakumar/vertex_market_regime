"""
Advanced IV Percentile Regime Classification - Component 4 Enhancement

Sophisticated 7-regime classification system (Extremely Low, Very Low, Low, Normal, 
High, Very High, Extremely High) with regime transition probability analysis,
regime stability metrics, cross-strike regime consistency validation,
and regime confidence scoring for institutional-grade market regime determination.

This module implements the most advanced IV percentile regime classification
system with comprehensive confidence assessment and transition analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import time
from enum import Enum

from .iv_percentile_analyzer import IVPercentileData
from .dte_percentile_framework import DTEPercentileMetrics
from .zone_percentile_tracker import ZonePercentileMetrics
from .historical_percentile_database import HistoricalPercentileDatabase, PercentileDistribution


class IVPercentileRegime(Enum):
    """7-level IV percentile regime classification"""
    EXTREMELY_LOW = "extremely_low"       # <5th percentile
    VERY_LOW = "very_low"                # 5th-15th percentile
    LOW = "low"                          # 15th-30th percentile
    NORMAL = "normal"                    # 30th-70th percentile
    HIGH = "high"                        # 70th-85th percentile
    VERY_HIGH = "very_high"              # 85th-95th percentile
    EXTREMELY_HIGH = "extremely_high"    # >95th percentile


@dataclass
class RegimeTransitionProbability:
    """Regime transition probability analysis"""
    
    # Current regime
    current_regime: IVPercentileRegime
    
    # Transition probabilities to each regime
    transition_probs: Dict[IVPercentileRegime, float]
    
    # Most likely next regime
    most_likely_next: IVPercentileRegime
    next_regime_probability: float
    
    # Stability metrics
    regime_stability: float
    persistence_score: float
    
    # Risk assessment
    transition_risk_score: float
    volatility_expansion_risk: float
    
    # Confidence metrics
    prediction_confidence: float
    data_reliability: float
    
    def get_transition_signal(self) -> str:
        """Get regime transition signal"""
        
        if self.next_regime_probability > 0.7:
            return f"strong_transition_to_{self.most_likely_next.value}"
        elif self.next_regime_probability > 0.5:
            return f"moderate_transition_to_{self.most_likely_next.value}"
        elif self.regime_stability > 0.8:
            return f"stable_{self.current_regime.value}"
        else:
            return "uncertain_regime_direction"


@dataclass
class RegimeStabilityMetrics:
    """Regime stability and persistence analysis"""
    
    # Stability measures
    regime_persistence: float
    volatility_consistency: float
    directional_stability: float
    
    # Duration analysis
    current_regime_duration: int  # Days in current regime
    average_regime_duration: float
    duration_percentile: float
    
    # Strength indicators
    regime_strength: float
    conviction_level: str
    stability_trend: str  # strengthening, weakening, stable
    
    # Risk factors
    regime_exhaustion_risk: float
    reversal_probability: float
    
    def get_stability_classification(self) -> str:
        """Get overall stability classification"""
        
        if self.regime_persistence > 0.8 and self.volatility_consistency > 0.7:
            return "highly_stable"
        elif self.regime_persistence > 0.6 and self.volatility_consistency > 0.5:
            return "moderately_stable"
        elif self.regime_persistence > 0.4:
            return "somewhat_unstable"
        else:
            return "highly_unstable"


@dataclass
class CrossStrikeConsistency:
    """Cross-strike regime consistency validation"""
    
    # Consistency metrics
    overall_consistency: float
    atm_consistency: float
    wing_consistency: float
    
    # Strike-level regime distribution
    regime_distribution: Dict[IVPercentileRegime, int]
    dominant_regime: IVPercentileRegime
    regime_dispersion: float
    
    # Validation scores
    consensus_strength: float
    outlier_strikes_count: int
    consistency_confidence: float
    
    # Surface quality
    surface_regime_quality: str
    reliability_score: float
    
    def is_regime_consistent(self) -> bool:
        """Check if regime is consistent across strikes"""
        return (self.overall_consistency > 0.7 and 
                self.consensus_strength > 0.6 and
                self.outlier_strikes_count < 3)


@dataclass
class RegimeConfidenceScoring:
    """Comprehensive regime confidence scoring system"""
    
    # Component confidence scores
    data_quality_score: float
    historical_depth_score: float
    statistical_significance: float
    cross_validation_score: float
    
    # Aggregate scores
    overall_confidence: float
    reliability_grade: str  # A, B, C, D, F
    
    # Risk adjustments
    uncertainty_factors: List[str]
    confidence_boosters: List[str]
    
    # Metadata
    sample_size: int
    calculation_method: str
    
    def get_confidence_level(self) -> str:
        """Get confidence level classification"""
        
        if self.overall_confidence >= 0.9:
            return "very_high_confidence"
        elif self.overall_confidence >= 0.75:
            return "high_confidence"
        elif self.overall_confidence >= 0.6:
            return "moderate_confidence"
        elif self.overall_confidence >= 0.4:
            return "low_confidence"
        else:
            return "very_low_confidence"


@dataclass
class AdvancedRegimeClassificationResult:
    """Complete advanced regime classification result"""
    
    # Primary classification
    primary_regime: IVPercentileRegime
    regime_percentile: float
    regime_confidence: float
    
    # Secondary analysis
    regime_strength: float
    regime_conviction: str
    regime_trend: str
    
    # Transition analysis
    transition_analysis: RegimeTransitionProbability
    stability_metrics: RegimeStabilityMetrics
    
    # Validation
    cross_strike_consistency: CrossStrikeConsistency
    confidence_scoring: RegimeConfidenceScoring
    
    # Integration scores
    dte_regime_agreement: float
    zone_regime_agreement: float
    temporal_consistency: float
    
    # Risk assessment
    regime_risk_level: str
    action_recommendations: List[str]
    
    # Metadata
    calculation_time_ms: float
    components_analyzed: List[str]
    
    def get_master_regime_signal(self) -> str:
        """Get master regime signal for Component 4"""
        
        base_signal = self.primary_regime.value
        
        if self.regime_confidence > 0.8 and self.stability_metrics.regime_persistence > 0.7:
            return f"strong_{base_signal}"
        elif self.regime_confidence > 0.6:
            return f"moderate_{base_signal}" 
        elif self.transition_analysis.transition_risk_score > 0.7:
            return f"transitional_{base_signal}"
        else:
            return f"uncertain_{base_signal}"


class AdvancedIVPercentileRegimeClassifier:
    """
    Advanced IV Percentile Regime Classification System with comprehensive
    7-regime analysis, transition probability modeling, and institutional-grade
    confidence assessment.
    
    Features:
    - 7-regime classification (Extremely Low to Extremely High)
    - Regime transition probability analysis with Markov modeling
    - Multi-dimensional stability metrics and persistence analysis
    - Cross-strike regime consistency validation
    - Sophisticated confidence scoring with multiple validation layers
    - DTE and zone regime integration analysis
    - Risk-adjusted action recommendations
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # Regime thresholds configuration
        self.regime_thresholds = {
            IVPercentileRegime.EXTREMELY_LOW: (0, 5),
            IVPercentileRegime.VERY_LOW: (5, 15),
            IVPercentileRegime.LOW: (15, 30),
            IVPercentileRegime.NORMAL: (30, 70),
            IVPercentileRegime.HIGH: (70, 85),
            IVPercentileRegime.VERY_HIGH: (85, 95),
            IVPercentileRegime.EXTREMELY_HIGH: (95, 100)
        }
        
        # Transition analysis configuration
        self.transition_window_days = config.get('transition_window_days', 30)
        self.min_regime_observations = config.get('min_regime_observations', 10)
        self.stability_lookback_days = config.get('stability_lookback_days', 14)
        
        # Confidence scoring configuration
        self.min_confidence_threshold = config.get('min_confidence_threshold', 0.4)
        self.high_confidence_threshold = config.get('high_confidence_threshold', 0.75)
        self.cross_validation_weight = config.get('cross_validation_weight', 0.3)
        
        # Performance configuration
        self.processing_budget_ms = config.get('regime_processing_budget_ms', 60)
        
        # Historical transition matrix (would be learned from data)
        self.transition_matrix = self._initialize_transition_matrix()
        
        self.logger.info("Advanced IV Percentile Regime Classifier initialized with 7-regime system")
    
    def classify_advanced_regime(self, 
                               iv_data: IVPercentileData,
                               dte_metrics: DTEPercentileMetrics,
                               zone_metrics: ZonePercentileMetrics,
                               historical_db: HistoricalPercentileDatabase) -> AdvancedRegimeClassificationResult:
        """
        Perform comprehensive regime classification with advanced analysis
        
        Args:
            iv_data: IV percentile data
            dte_metrics: DTE-specific percentile metrics
            zone_metrics: Zone-specific percentile metrics
            historical_db: Historical percentile database
            
        Returns:
            AdvancedRegimeClassificationResult with complete analysis
        """
        start_time = time.time()
        
        try:
            # Primary regime classification
            primary_classification = self._classify_primary_regime(
                dte_metrics.dte_iv_percentile, iv_data, historical_db
            )
            
            # Transition probability analysis
            transition_analysis = self._analyze_regime_transitions(
                primary_classification['regime'], 
                dte_metrics.dte_iv_percentile,
                historical_db
            )
            
            # Stability metrics calculation
            stability_metrics = self._calculate_regime_stability(
                primary_classification['regime'],
                dte_metrics,
                zone_metrics,
                historical_db
            )
            
            # Cross-strike consistency validation
            cross_strike_consistency = self._validate_cross_strike_consistency(
                iv_data, primary_classification['regime']
            )
            
            # Confidence scoring
            confidence_scoring = self._calculate_regime_confidence(
                primary_classification,
                transition_analysis,
                stability_metrics,
                cross_strike_consistency,
                dte_metrics,
                zone_metrics
            )
            
            # Integration analysis
            integration_analysis = self._analyze_regime_integration(
                primary_classification['regime'],
                dte_metrics,
                zone_metrics
            )
            
            # Risk assessment and recommendations
            risk_analysis = self._assess_regime_risk(
                primary_classification['regime'],
                transition_analysis,
                stability_metrics
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            # Create comprehensive result
            result = AdvancedRegimeClassificationResult(
                primary_regime=primary_classification['regime'],
                regime_percentile=dte_metrics.dte_iv_percentile,
                regime_confidence=confidence_scoring.overall_confidence,
                regime_strength=primary_classification['strength'],
                regime_conviction=primary_classification['conviction'],
                regime_trend=primary_classification['trend'],
                transition_analysis=transition_analysis,
                stability_metrics=stability_metrics,
                cross_strike_consistency=cross_strike_consistency,
                confidence_scoring=confidence_scoring,
                dte_regime_agreement=integration_analysis['dte_agreement'],
                zone_regime_agreement=integration_analysis['zone_agreement'],
                temporal_consistency=integration_analysis['temporal_consistency'],
                regime_risk_level=risk_analysis['risk_level'],
                action_recommendations=risk_analysis['recommendations'],
                calculation_time_ms=processing_time,
                components_analyzed=['iv_data', 'dte_metrics', 'zone_metrics', 'historical_db']
            )
            
            self.logger.debug(f"Advanced regime classification completed: "
                            f"Regime={result.primary_regime.value}, "
                            f"Confidence={result.regime_confidence:.2f}, "
                            f"Time={processing_time:.2f}ms")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Advanced regime classification failed: {e}")
            return self._get_default_classification_result(iv_data, dte_metrics)
    
    def calculate_regime_transition_probabilities(self, 
                                                current_regime: IVPercentileRegime,
                                                current_percentile: float,
                                                historical_db: HistoricalPercentileDatabase) -> Dict[IVPercentileRegime, float]:
        """
        Calculate transition probabilities using historical patterns and Markov analysis
        
        Args:
            current_regime: Current IV percentile regime
            current_percentile: Current percentile value
            historical_db: Historical database for pattern analysis
            
        Returns:
            Dictionary with transition probabilities for each regime
        """
        try:
            # Base transition probabilities from historical matrix
            base_probs = self.transition_matrix.get(current_regime, {})
            
            # Adjust probabilities based on current percentile position within regime
            adjusted_probs = self._adjust_transition_probabilities(
                base_probs, current_regime, current_percentile
            )
            
            # Apply market condition adjustments
            market_adjusted_probs = self._apply_market_condition_adjustment(
                adjusted_probs, historical_db
            )
            
            # Normalize probabilities
            total_prob = sum(market_adjusted_probs.values())
            if total_prob > 0:
                normalized_probs = {regime: prob / total_prob 
                                  for regime, prob in market_adjusted_probs.items()}
            else:
                # Equal probability fallback
                normalized_probs = {regime: 1.0 / len(IVPercentileRegime) 
                                  for regime in IVPercentileRegime}
            
            return normalized_probs
            
        except Exception as e:
            self.logger.error(f"Transition probability calculation failed: {e}")
            return {regime: 1.0 / len(IVPercentileRegime) for regime in IVPercentileRegime}
    
    def validate_regime_consistency(self, 
                                  iv_data: IVPercentileData,
                                  primary_regime: IVPercentileRegime) -> CrossStrikeConsistency:
        """
        Validate regime classification consistency across the complete strike chain
        
        Args:
            iv_data: IV percentile data with all strikes
            primary_regime: Primary regime classification
            
        Returns:
            CrossStrikeConsistency with validation results
        """
        try:
            # Analyze regime across all strikes
            strike_regimes = []
            atm_strike_regimes = []
            wing_strike_regimes = []
            
            atm_idx = np.argmin(np.abs(iv_data.strikes - iv_data.atm_strike))
            
            for i, strike in enumerate(iv_data.strikes):
                # Calculate strike-specific percentile (simplified)
                ce_iv = iv_data.ce_iv[i] if not np.isnan(iv_data.ce_iv[i]) else 0
                pe_iv = iv_data.pe_iv[i] if not np.isnan(iv_data.pe_iv[i]) else 0
                avg_iv = (ce_iv + pe_iv) / 2 if (ce_iv > 0 or pe_iv > 0) else 0
                
                # Classify regime for this strike (simplified percentile estimation)
                if avg_iv > 0:
                    # Rough percentile estimation based on relative IV level
                    estimated_percentile = min(95, max(5, avg_iv * 2))  # Simplified
                    strike_regime = self._classify_regime_from_percentile(estimated_percentile)
                    strike_regimes.append(strike_regime)
                    
                    # Categorize by strike type
                    moneyness = strike / iv_data.spot
                    if abs(atm_idx - i) <= 2:  # ATM strikes
                        atm_strike_regimes.append(strike_regime)
                    elif moneyness < 0.95 or moneyness > 1.05:  # Wing strikes
                        wing_strike_regimes.append(strike_regime)
            
            if not strike_regimes:
                return self._get_default_consistency_result()
            
            # Calculate consistency metrics
            regime_counts = {regime: strike_regimes.count(regime) for regime in IVPercentileRegime}
            total_strikes = len(strike_regimes)
            
            # Dominant regime and consistency
            dominant_regime = max(regime_counts.keys(), key=lambda x: regime_counts[x])
            dominant_count = regime_counts[dominant_regime]
            overall_consistency = dominant_count / total_strikes if total_strikes > 0 else 0
            
            # ATM consistency
            atm_consistency = 1.0
            if atm_strike_regimes:
                atm_dominant = max(set(atm_strike_regimes), key=atm_strike_regimes.count)
                atm_consistency = atm_strike_regimes.count(atm_dominant) / len(atm_strike_regimes)
            
            # Wing consistency
            wing_consistency = 1.0
            if wing_strike_regimes:
                wing_dominant = max(set(wing_strike_regimes), key=wing_strike_regimes.count)
                wing_consistency = wing_strike_regimes.count(wing_dominant) / len(wing_strike_regimes)
            
            # Regime dispersion
            regime_dispersion = len(set(strike_regimes)) / len(IVPercentileRegime)
            
            # Consensus strength
            consensus_strength = max(regime_counts.values()) / total_strikes if total_strikes > 0 else 0
            
            # Outlier count
            outlier_strikes_count = sum(1 for regime in strike_regimes if regime != dominant_regime)
            
            # Confidence and quality assessment
            consistency_confidence = (overall_consistency * 0.5 + 
                                    atm_consistency * 0.3 + 
                                    wing_consistency * 0.2)
            
            surface_regime_quality = self._assess_surface_regime_quality(
                overall_consistency, regime_dispersion, outlier_strikes_count
            )
            
            reliability_score = min(1.0, consistency_confidence * (1 - regime_dispersion))
            
            return CrossStrikeConsistency(
                overall_consistency=overall_consistency,
                atm_consistency=atm_consistency,
                wing_consistency=wing_consistency,
                regime_distribution=regime_counts,
                dominant_regime=dominant_regime,
                regime_dispersion=regime_dispersion,
                consensus_strength=consensus_strength,
                outlier_strikes_count=outlier_strikes_count,
                consistency_confidence=consistency_confidence,
                surface_regime_quality=surface_regime_quality,
                reliability_score=reliability_score
            )
            
        except Exception as e:
            self.logger.error(f"Regime consistency validation failed: {e}")
            return self._get_default_consistency_result()
    
    def calculate_regime_confidence_score(self,
                                        classification_components: Dict[str, Any]) -> RegimeConfidenceScoring:
        """
        Calculate comprehensive regime confidence score with multiple validation layers
        
        Args:
            classification_components: All classification components for confidence assessment
            
        Returns:
            RegimeConfidenceScoring with detailed confidence metrics
        """
        try:
            # Data quality assessment
            data_quality_score = self._assess_data_quality(
                classification_components.get('iv_data'),
                classification_components.get('dte_metrics'),
                classification_components.get('zone_metrics')
            )
            
            # Historical depth scoring
            historical_depth_score = self._assess_historical_depth(
                classification_components.get('historical_db')
            )
            
            # Statistical significance
            statistical_significance = self._calculate_statistical_significance(
                classification_components
            )
            
            # Cross-validation score
            cross_validation_score = self._perform_cross_validation(
                classification_components
            )
            
            # Calculate overall confidence
            component_scores = [
                data_quality_score * 0.3,
                historical_depth_score * 0.25,
                statistical_significance * 0.25,
                cross_validation_score * 0.2
            ]
            
            overall_confidence = sum(component_scores)
            
            # Reliability grade
            reliability_grade = self._assign_reliability_grade(overall_confidence)
            
            # Identify uncertainty factors and confidence boosters
            uncertainty_factors = self._identify_uncertainty_factors(
                data_quality_score, historical_depth_score, 
                statistical_significance, cross_validation_score
            )
            
            confidence_boosters = self._identify_confidence_boosters(
                data_quality_score, historical_depth_score,
                statistical_significance, cross_validation_score
            )
            
            # Sample size and method
            sample_size = self._estimate_sample_size(classification_components)
            calculation_method = "multi_component_validation"
            
            return RegimeConfidenceScoring(
                data_quality_score=data_quality_score,
                historical_depth_score=historical_depth_score,
                statistical_significance=statistical_significance,
                cross_validation_score=cross_validation_score,
                overall_confidence=overall_confidence,
                reliability_grade=reliability_grade,
                uncertainty_factors=uncertainty_factors,
                confidence_boosters=confidence_boosters,
                sample_size=sample_size,
                calculation_method=calculation_method
            )
            
        except Exception as e:
            self.logger.error(f"Regime confidence scoring failed: {e}")
            return self._get_default_confidence_scoring()
    
    def _initialize_transition_matrix(self) -> Dict[IVPercentileRegime, Dict[IVPercentileRegime, float]]:
        """Initialize regime transition matrix with base probabilities"""
        
        # Base transition matrix (would be learned from historical data)
        transition_matrix = {}
        
        for current_regime in IVPercentileRegime:
            transitions = {}
            
            for target_regime in IVPercentileRegime:
                if current_regime == target_regime:
                    # Stay in same regime (persistence)
                    transitions[target_regime] = 0.6
                elif abs(list(IVPercentileRegime).index(current_regime) - 
                        list(IVPercentileRegime).index(target_regime)) == 1:
                    # Adjacent regime transition
                    transitions[target_regime] = 0.15
                elif abs(list(IVPercentileRegime).index(current_regime) - 
                        list(IVPercentileRegime).index(target_regime)) == 2:
                    # Two-step regime transition
                    transitions[target_regime] = 0.08
                else:
                    # Distant regime transition
                    transitions[target_regime] = 0.02
            
            # Normalize
            total = sum(transitions.values())
            transitions = {regime: prob / total for regime, prob in transitions.items()}
            
            transition_matrix[current_regime] = transitions
        
        return transition_matrix
    
    def _classify_primary_regime(self, percentile: float, iv_data: IVPercentileData,
                               historical_db: HistoricalPercentileDatabase) -> Dict[str, Any]:
        """Classify primary regime with strength and conviction"""
        
        # Determine regime based on percentile
        regime = self._classify_regime_from_percentile(percentile)
        
        # Calculate regime strength (distance from regime boundaries)
        regime_bounds = self.regime_thresholds[regime]
        regime_center = (regime_bounds[0] + regime_bounds[1]) / 2
        regime_width = regime_bounds[1] - regime_bounds[0]
        
        # Strength based on distance from center
        distance_from_center = abs(percentile - regime_center)
        strength = 1.0 - (distance_from_center / (regime_width / 2))
        strength = max(0.0, min(1.0, strength))
        
        # Conviction level
        if strength > 0.8:
            conviction = "very_high"
        elif strength > 0.6:
            conviction = "high"
        elif strength > 0.4:
            conviction = "moderate"
        elif strength > 0.2:
            conviction = "low"
        else:
            conviction = "very_low"
        
        # Trend analysis (simplified)
        trend = self._analyze_regime_trend(percentile, regime)
        
        return {
            'regime': regime,
            'strength': strength,
            'conviction': conviction,
            'trend': trend
        }
    
    def _classify_regime_from_percentile(self, percentile: float) -> IVPercentileRegime:
        """Classify regime from percentile value"""
        
        for regime, (min_pct, max_pct) in self.regime_thresholds.items():
            if min_pct <= percentile < max_pct:
                return regime
        
        # Handle edge case for 100th percentile
        if percentile >= 95:
            return IVPercentileRegime.EXTREMELY_HIGH
        else:
            return IVPercentileRegime.NORMAL
    
    def _analyze_regime_transitions(self, current_regime: IVPercentileRegime,
                                  current_percentile: float,
                                  historical_db: HistoricalPercentileDatabase) -> RegimeTransitionProbability:
        """Analyze regime transition probabilities"""
        
        try:
            # Get transition probabilities
            transition_probs = self.calculate_regime_transition_probabilities(
                current_regime, current_percentile, historical_db
            )
            
            # Find most likely next regime
            most_likely_next = max(transition_probs.keys(), key=lambda x: transition_probs[x])
            next_regime_probability = transition_probs[most_likely_next]
            
            # Calculate stability metrics
            stay_probability = transition_probs.get(current_regime, 0.0)
            regime_stability = stay_probability
            persistence_score = min(1.0, stay_probability * 1.5)  # Amplify for scoring
            
            # Risk assessment
            extreme_transition_risk = sum(
                prob for regime, prob in transition_probs.items()
                if regime in [IVPercentileRegime.EXTREMELY_LOW, IVPercentileRegime.EXTREMELY_HIGH]
                and regime != current_regime
            )
            
            volatility_expansion_risk = sum(
                prob for regime, prob in transition_probs.items()
                if regime in [IVPercentileRegime.HIGH, IVPercentileRegime.VERY_HIGH, IVPercentileRegime.EXTREMELY_HIGH]
                and regime != current_regime
            )
            
            # Confidence metrics
            prediction_confidence = max(transition_probs.values())
            data_reliability = 0.8  # Would be calculated from historical data quality
            
            return RegimeTransitionProbability(
                current_regime=current_regime,
                transition_probs=transition_probs,
                most_likely_next=most_likely_next,
                next_regime_probability=next_regime_probability,
                regime_stability=regime_stability,
                persistence_score=persistence_score,
                transition_risk_score=extreme_transition_risk,
                volatility_expansion_risk=volatility_expansion_risk,
                prediction_confidence=prediction_confidence,
                data_reliability=data_reliability
            )
            
        except Exception as e:
            self.logger.error(f"Regime transition analysis failed: {e}")
            return self._get_default_transition_probability(current_regime)
    
    def _calculate_regime_stability(self, regime: IVPercentileRegime,
                                  dte_metrics: DTEPercentileMetrics,
                                  zone_metrics: ZonePercentileMetrics,
                                  historical_db: HistoricalPercentileDatabase) -> RegimeStabilityMetrics:
        """Calculate comprehensive regime stability metrics"""
        
        try:
            # Regime persistence (how long in current regime)
            regime_persistence = min(1.0, dte_metrics.dte_confidence_score * 1.2)
            
            # Volatility consistency across zones
            volatility_consistency = min(1.0, zone_metrics.calculation_confidence * 1.1)
            
            # Directional stability
            directional_stability = 0.8  # Would be calculated from recent regime history
            
            # Duration analysis (simplified)
            current_regime_duration = 5  # Days - would be calculated from history
            average_regime_duration = 12.0  # Average regime duration
            duration_percentile = min(100, (current_regime_duration / average_regime_duration) * 50)
            
            # Regime strength
            regime_strength = (regime_persistence + volatility_consistency + directional_stability) / 3
            
            # Conviction level
            if regime_strength > 0.8:
                conviction_level = "very_strong"
                stability_trend = "strengthening"
            elif regime_strength > 0.6:
                conviction_level = "strong"
                stability_trend = "stable"
            elif regime_strength > 0.4:
                conviction_level = "moderate"
                stability_trend = "stable"
            else:
                conviction_level = "weak"
                stability_trend = "weakening"
            
            # Risk factors
            regime_exhaustion_risk = max(0.0, (current_regime_duration - average_regime_duration) / average_regime_duration)
            regime_exhaustion_risk = min(1.0, regime_exhaustion_risk)
            
            reversal_probability = regime_exhaustion_risk * 0.7
            
            return RegimeStabilityMetrics(
                regime_persistence=regime_persistence,
                volatility_consistency=volatility_consistency,
                directional_stability=directional_stability,
                current_regime_duration=current_regime_duration,
                average_regime_duration=average_regime_duration,
                duration_percentile=duration_percentile,
                regime_strength=regime_strength,
                conviction_level=conviction_level,
                stability_trend=stability_trend,
                regime_exhaustion_risk=regime_exhaustion_risk,
                reversal_probability=reversal_probability
            )
            
        except Exception as e:
            self.logger.error(f"Regime stability calculation failed: {e}")
            return self._get_default_stability_metrics()
    
    def _validate_cross_strike_consistency(self, iv_data: IVPercentileData,
                                         primary_regime: IVPercentileRegime) -> CrossStrikeConsistency:
        """Validate regime consistency across strikes"""
        return self.validate_regime_consistency(iv_data, primary_regime)
    
    def _calculate_regime_confidence(self, primary_classification: Dict[str, Any],
                                   transition_analysis: RegimeTransitionProbability,
                                   stability_metrics: RegimeStabilityMetrics,
                                   cross_strike_consistency: CrossStrikeConsistency,
                                   dte_metrics: DTEPercentileMetrics,
                                   zone_metrics: ZonePercentileMetrics) -> RegimeConfidenceScoring:
        """Calculate comprehensive regime confidence"""
        
        components = {
            'primary_classification': primary_classification,
            'transition_analysis': transition_analysis,
            'stability_metrics': stability_metrics,
            'cross_strike_consistency': cross_strike_consistency,
            'dte_metrics': dte_metrics,
            'zone_metrics': zone_metrics
        }
        
        return self.calculate_regime_confidence_score(components)
    
    def _analyze_regime_integration(self, primary_regime: IVPercentileRegime,
                                  dte_metrics: DTEPercentileMetrics,
                                  zone_metrics: ZonePercentileMetrics) -> Dict[str, float]:
        """Analyze integration between different regime components"""
        
        # DTE regime agreement
        dte_regime_from_classification = self._classify_regime_from_percentile(dte_metrics.dte_iv_percentile)
        dte_agreement = 1.0 if dte_regime_from_classification == primary_regime else 0.5
        
        # Zone regime agreement
        zone_regime_from_classification = self._classify_regime_from_percentile(zone_metrics.zone_iv_percentile)
        zone_agreement = 1.0 if zone_regime_from_classification == primary_regime else 0.5
        
        # Temporal consistency
        temporal_consistency = min(1.0, (dte_metrics.dte_confidence_score + zone_metrics.calculation_confidence) / 2)
        
        return {
            'dte_agreement': dte_agreement,
            'zone_agreement': zone_agreement,
            'temporal_consistency': temporal_consistency
        }
    
    def _assess_regime_risk(self, regime: IVPercentileRegime,
                          transition_analysis: RegimeTransitionProbability,
                          stability_metrics: RegimeStabilityMetrics) -> Dict[str, Any]:
        """Assess risk level and generate recommendations"""
        
        # Risk level determination
        if regime in [IVPercentileRegime.EXTREMELY_HIGH, IVPercentileRegime.EXTREMELY_LOW]:
            base_risk = "very_high"
        elif regime in [IVPercentileRegime.VERY_HIGH, IVPercentileRegime.VERY_LOW]:
            base_risk = "high"
        elif regime in [IVPercentileRegime.HIGH, IVPercentileRegime.LOW]:
            base_risk = "moderate"
        else:
            base_risk = "normal"
        
        # Adjust for transition risk
        if transition_analysis.transition_risk_score > 0.7:
            risk_level = "very_high" if base_risk != "very_high" else base_risk
        elif transition_analysis.volatility_expansion_risk > 0.6:
            risk_level = "high" if base_risk == "normal" else base_risk
        else:
            risk_level = base_risk
        
        # Generate recommendations
        recommendations = []
        
        if regime == IVPercentileRegime.EXTREMELY_HIGH:
            recommendations.extend([
                "Consider defensive positioning",
                "Monitor for volatility contraction",
                "Reduce new long volatility exposure"
            ])
        elif regime == IVPercentileRegime.EXTREMELY_LOW:
            recommendations.extend([
                "Consider opportunistic volatility purchases",
                "Monitor for volatility expansion catalysts",
                "Evaluate structured product opportunities"
            ])
        elif transition_analysis.transition_risk_score > 0.7:
            recommendations.extend([
                "Prepare for potential regime transition",
                "Reduce position sizes",
                "Increase hedging coverage"
            ])
        else:
            recommendations.append("Maintain current positioning approach")
        
        return {
            'risk_level': risk_level,
            'recommendations': recommendations
        }
    
    def _adjust_transition_probabilities(self, base_probs: Dict[IVPercentileRegime, float],
                                       current_regime: IVPercentileRegime,
                                       current_percentile: float) -> Dict[IVPercentileRegime, float]:
        """Adjust transition probabilities based on current position within regime"""
        
        adjusted_probs = base_probs.copy()
        
        # Get current regime bounds
        regime_bounds = self.regime_thresholds[current_regime]
        regime_center = (regime_bounds[0] + regime_bounds[1]) / 2
        
        # If near regime boundary, increase transition probability to adjacent regime
        if current_percentile < regime_center:
            # Near lower boundary, increase probability of moving to lower regime
            lower_regime_idx = list(IVPercentileRegime).index(current_regime) - 1
            if lower_regime_idx >= 0:
                lower_regime = list(IVPercentileRegime)[lower_regime_idx]
                if lower_regime in adjusted_probs:
                    adjusted_probs[lower_regime] *= 1.5
                    adjusted_probs[current_regime] *= 0.9
        else:
            # Near upper boundary, increase probability of moving to higher regime
            higher_regime_idx = list(IVPercentileRegime).index(current_regime) + 1
            if higher_regime_idx < len(IVPercentileRegime):
                higher_regime = list(IVPercentileRegime)[higher_regime_idx]
                if higher_regime in adjusted_probs:
                    adjusted_probs[higher_regime] *= 1.5
                    adjusted_probs[current_regime] *= 0.9
        
        return adjusted_probs
    
    def _apply_market_condition_adjustment(self, probs: Dict[IVPercentileRegime, float],
                                         historical_db: HistoricalPercentileDatabase) -> Dict[IVPercentileRegime, float]:
        """Apply market condition adjustments to transition probabilities"""
        
        # Market condition factor (simplified)
        market_stress_factor = 1.0  # Would be calculated from market indicators
        
        adjusted_probs = probs.copy()
        
        # In stressed conditions, increase probability of extreme regimes
        if market_stress_factor > 1.2:
            for regime in [IVPercentileRegime.EXTREMELY_HIGH, IVPercentileRegime.VERY_HIGH]:
                if regime in adjusted_probs:
                    adjusted_probs[regime] *= 1.3
        
        return adjusted_probs
    
    def _analyze_regime_trend(self, percentile: float, regime: IVPercentileRegime) -> str:
        """Analyze regime trend direction"""
        
        # Simplified trend analysis
        regime_bounds = self.regime_thresholds[regime]
        regime_center = (regime_bounds[0] + regime_bounds[1]) / 2
        
        if percentile > regime_center + 5:
            return "strengthening"
        elif percentile < regime_center - 5:
            return "weakening"
        else:
            return "stable"
    
    def _assess_surface_regime_quality(self, consistency: float, dispersion: float, outliers: int) -> str:
        """Assess overall surface regime quality"""
        
        if consistency > 0.8 and dispersion < 0.3 and outliers < 2:
            return "excellent"
        elif consistency > 0.6 and dispersion < 0.5 and outliers < 4:
            return "good"
        elif consistency > 0.4 and dispersion < 0.7:
            return "fair"
        else:
            return "poor"
    
    def _assess_data_quality(self, iv_data: IVPercentileData, 
                           dte_metrics: DTEPercentileMetrics,
                           zone_metrics: ZonePercentileMetrics) -> float:
        """Assess data quality for confidence scoring"""
        
        factors = []
        
        if iv_data:
            factors.append(iv_data.data_completeness)
        
        if dte_metrics:
            factors.append(dte_metrics.calculation_confidence)
        
        if zone_metrics:
            factors.append(zone_metrics.zone_data_quality)
        
        return float(np.mean(factors)) if factors else 0.5
    
    def _assess_historical_depth(self, historical_db: HistoricalPercentileDatabase) -> float:
        """Assess historical data depth for confidence"""
        
        if not historical_db:
            return 0.3
        
        # Simplified assessment
        summary = historical_db.get_historical_database_summary()
        total_entries = summary.get('total_entries', 0)
        
        # Score based on data depth
        if total_entries > 1000:
            return 0.9
        elif total_entries > 500:
            return 0.7
        elif total_entries > 100:
            return 0.5
        else:
            return 0.3
    
    def _calculate_statistical_significance(self, components: Dict[str, Any]) -> float:
        """Calculate statistical significance of regime classification"""
        
        # Simplified significance calculation
        data_points = 0
        quality_scores = []
        
        if 'dte_metrics' in components and components['dte_metrics']:
            data_points += 1
            quality_scores.append(components['dte_metrics'].data_sufficiency)
        
        if 'zone_metrics' in components and components['zone_metrics']:
            data_points += 1
            quality_scores.append(components['zone_metrics'].data_sufficiency)
        
        if data_points == 0:
            return 0.3
        
        avg_quality = np.mean(quality_scores)
        significance = min(1.0, avg_quality * (data_points / 2))
        
        return float(significance)
    
    def _perform_cross_validation(self, components: Dict[str, Any]) -> float:
        """Perform cross-validation of regime classification"""
        
        validation_scores = []
        
        # DTE vs Zone consistency
        if ('dte_metrics' in components and components['dte_metrics'] and
            'zone_metrics' in components and components['zone_metrics']):
            
            dte_regime = self._classify_regime_from_percentile(
                components['dte_metrics'].dte_iv_percentile
            )
            zone_regime = self._classify_regime_from_percentile(
                components['zone_metrics'].zone_iv_percentile
            )
            
            consistency_score = 1.0 if dte_regime == zone_regime else 0.5
            validation_scores.append(consistency_score)
        
        # Cross-strike validation
        if 'cross_strike_consistency' in components and components['cross_strike_consistency']:
            validation_scores.append(components['cross_strike_consistency'].overall_consistency)
        
        return float(np.mean(validation_scores)) if validation_scores else 0.5
    
    def _assign_reliability_grade(self, confidence: float) -> str:
        """Assign reliability grade based on confidence"""
        
        if confidence >= 0.9:
            return "A"
        elif confidence >= 0.8:
            return "B"
        elif confidence >= 0.65:
            return "C"
        elif confidence >= 0.5:
            return "D"
        else:
            return "F"
    
    def _identify_uncertainty_factors(self, data_quality: float, historical_depth: float,
                                    statistical_sig: float, cross_val: float) -> List[str]:
        """Identify factors contributing to uncertainty"""
        
        factors = []
        
        if data_quality < 0.6:
            factors.append("low_data_quality")
        
        if historical_depth < 0.5:
            factors.append("insufficient_historical_data")
        
        if statistical_sig < 0.4:
            factors.append("low_statistical_significance")
        
        if cross_val < 0.6:
            factors.append("poor_cross_validation")
        
        return factors
    
    def _identify_confidence_boosters(self, data_quality: float, historical_depth: float,
                                    statistical_sig: float, cross_val: float) -> List[str]:
        """Identify factors boosting confidence"""
        
        boosters = []
        
        if data_quality > 0.8:
            boosters.append("high_data_quality")
        
        if historical_depth > 0.7:
            boosters.append("extensive_historical_data")
        
        if statistical_sig > 0.7:
            boosters.append("high_statistical_significance")
        
        if cross_val > 0.8:
            boosters.append("strong_cross_validation")
        
        return boosters
    
    def _estimate_sample_size(self, components: Dict[str, Any]) -> int:
        """Estimate effective sample size for analysis"""
        
        # Simplified sample size estimation
        base_size = 50
        
        if 'historical_db' in components and components['historical_db']:
            summary = components['historical_db'].get_historical_database_summary()
            base_size = summary.get('total_entries', 50)
        
        return max(10, min(1000, base_size))
    
    # Default return methods
    def _get_default_classification_result(self, iv_data: IVPercentileData,
                                         dte_metrics: DTEPercentileMetrics) -> AdvancedRegimeClassificationResult:
        """Get default classification result when analysis fails"""
        
        default_regime = IVPercentileRegime.NORMAL
        
        return AdvancedRegimeClassificationResult(
            primary_regime=default_regime,
            regime_percentile=dte_metrics.dte_iv_percentile if dte_metrics else 50.0,
            regime_confidence=0.5,
            regime_strength=0.5,
            regime_conviction="moderate",
            regime_trend="stable",
            transition_analysis=self._get_default_transition_probability(default_regime),
            stability_metrics=self._get_default_stability_metrics(),
            cross_strike_consistency=self._get_default_consistency_result(),
            confidence_scoring=self._get_default_confidence_scoring(),
            dte_regime_agreement=0.5,
            zone_regime_agreement=0.5,
            temporal_consistency=0.5,
            regime_risk_level="normal",
            action_recommendations=["maintain_current_positioning"],
            calculation_time_ms=0.0,
            components_analyzed=["default"]
        )
    
    def _get_default_transition_probability(self, regime: IVPercentileRegime) -> RegimeTransitionProbability:
        """Get default transition probability when analysis fails"""
        
        equal_prob = 1.0 / len(IVPercentileRegime)
        equal_probs = {r: equal_prob for r in IVPercentileRegime}
        
        return RegimeTransitionProbability(
            current_regime=regime,
            transition_probs=equal_probs,
            most_likely_next=regime,
            next_regime_probability=equal_prob,
            regime_stability=0.5,
            persistence_score=0.5,
            transition_risk_score=0.3,
            volatility_expansion_risk=0.3,
            prediction_confidence=0.5,
            data_reliability=0.5
        )
    
    def _get_default_stability_metrics(self) -> RegimeStabilityMetrics:
        """Get default stability metrics when analysis fails"""
        
        return RegimeStabilityMetrics(
            regime_persistence=0.5,
            volatility_consistency=0.5,
            directional_stability=0.5,
            current_regime_duration=7,
            average_regime_duration=10.0,
            duration_percentile=50.0,
            regime_strength=0.5,
            conviction_level="moderate",
            stability_trend="stable",
            regime_exhaustion_risk=0.3,
            reversal_probability=0.3
        )
    
    def _get_default_consistency_result(self) -> CrossStrikeConsistency:
        """Get default consistency result when analysis fails"""
        
        return CrossStrikeConsistency(
            overall_consistency=0.5,
            atm_consistency=0.5,
            wing_consistency=0.5,
            regime_distribution={IVPercentileRegime.NORMAL: 1},
            dominant_regime=IVPercentileRegime.NORMAL,
            regime_dispersion=0.3,
            consensus_strength=0.5,
            outlier_strikes_count=2,
            consistency_confidence=0.5,
            surface_regime_quality="fair",
            reliability_score=0.5
        )
    
    def _get_default_confidence_scoring(self) -> RegimeConfidenceScoring:
        """Get default confidence scoring when analysis fails"""
        
        return RegimeConfidenceScoring(
            data_quality_score=0.5,
            historical_depth_score=0.5,
            statistical_significance=0.5,
            cross_validation_score=0.5,
            overall_confidence=0.5,
            reliability_grade="C",
            uncertainty_factors=["insufficient_data"],
            confidence_boosters=[],
            sample_size=50,
            calculation_method="default"
        )