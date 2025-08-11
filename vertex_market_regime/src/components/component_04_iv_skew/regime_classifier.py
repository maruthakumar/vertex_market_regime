"""
IV Skew Regime Classification Engine for Component 4

Advanced IV skew classification and regime detection using complete volatility
surface characteristics, put skew dominance analysis, tail risk quantification,
and institutional flow detection for 8-regime market classification.

🚨 COMPREHENSIVE REGIME CLASSIFICATION:
- Complete smile analysis using volatility surface shape, skew, and curvature
- Put skew dominance leveraging asymmetric coverage (-21% range) for fear/greed detection
- Tail risk quantification using far OTM strikes for crash probability assessment
- Institutional flow detection via unusual surface changes and positioning analysis
- 8-regime classification using complete surface characteristics and evolution patterns
- Surface arbitrage signals from inconsistencies across strikes/DTEs
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging
from scipy import stats
import warnings

from .skew_analyzer import IVSkewData, VolatilitySurfaceResult, AdvancedIVMetrics
from .dual_dte_framework import TermStructureResult, DTESpecificMetrics

warnings.filterwarnings('ignore')


class MarketRegime(Enum):
    """8-regime classification for IV skew analysis"""
    EXTREME_FEAR = "EXTREME_FEAR"           # Very high put skew, crash fear
    HIGH_FEAR = "HIGH_FEAR"                 # High put skew, defensive positioning  
    MODERATE_FEAR = "MODERATE_FEAR"         # Moderate put bias, cautious sentiment
    NEUTRAL = "NEUTRAL"                     # Balanced skew, stable conditions
    MODERATE_GREED = "MODERATE_GREED"       # Slight call bias, optimistic sentiment
    HIGH_GREED = "HIGH_GREED"               # Strong call bias, risk-seeking
    EXTREME_GREED = "EXTREME_GREED"         # Very high call skew, euphoria
    TRANSITION = "TRANSITION"               # Mixed signals, regime changing


class SkewPattern(Enum):
    """Volatility skew pattern types"""
    STEEP_PUT_SKEW = "STEEP_PUT_SKEW"
    MODERATE_PUT_SKEW = "MODERATE_PUT_SKEW"
    FLAT_SKEW = "FLAT_SKEW"
    REVERSE_SKEW = "REVERSE_SKEW"
    IRREGULAR_SKEW = "IRREGULAR_SKEW"


class SmileCharacteristic(Enum):
    """Volatility smile characteristics"""
    SYMMETRIC_SMILE = "SYMMETRIC_SMILE"
    ASYMMETRIC_PUT_WING = "ASYMMETRIC_PUT_WING"
    ASYMMETRIC_CALL_WING = "ASYMMETRIC_CALL_WING"
    FLAT_SMILE = "FLAT_SMILE"
    INVERTED_SMILE = "INVERTED_SMILE"


@dataclass
class RegimeClassificationInput:
    """Input data for regime classification"""
    skew_data: IVSkewData
    surface_result: VolatilitySurfaceResult
    advanced_metrics: AdvancedIVMetrics
    term_structure_result: Optional[TermStructureResult]
    dte_metrics: Optional[DTESpecificMetrics]


@dataclass
class SkewPatternAnalysis:
    """Detailed skew pattern analysis"""
    primary_pattern: SkewPattern
    pattern_strength: float
    pattern_confidence: float
    
    # Skew metrics
    put_skew_steepness: float
    call_skew_steepness: float
    skew_asymmetry: float
    
    # Pattern characteristics
    pattern_features: Dict[str, float]
    unusual_characteristics: List[str]
    
    # Institutional indicators
    institutional_signature: float
    flow_imbalance_score: float


@dataclass
class SmileAnalysisResult:
    """Complete volatility smile analysis"""
    smile_characteristic: SmileCharacteristic
    smile_quality: float
    smile_stability: float
    
    # Smile metrics
    curvature_score: float
    wing_asymmetry: float
    atm_iv_level: float
    
    # Risk indicators
    tail_risk_indicator: float
    crash_risk_probability: float
    
    # Arbitrage indicators
    arbitrage_opportunities: List[Dict[str, float]]
    surface_consistency: float


@dataclass  
class InstitutionalFlowAnalysis:
    """Institutional flow detection analysis"""
    flow_detection_confidence: float
    flow_direction: str  # 'bullish', 'bearish', 'neutral', 'mixed'
    flow_magnitude: float
    
    # Flow characteristics
    unusual_surface_changes: Dict[str, float]
    volume_flow_indicators: Dict[str, float]
    oi_positioning_signals: Dict[str, float]
    
    # Timing indicators
    flow_persistence: float
    flow_acceleration: float
    
    # Risk assessment
    institutional_risk_appetite: float
    hedging_vs_speculation: str


@dataclass
class TailRiskAssessment:
    """Comprehensive tail risk assessment"""
    overall_tail_risk_score: float
    
    # Put tail (crash risk)
    put_tail_risk_score: float
    crash_probability: float
    crash_magnitude_estimate: float
    
    # Call tail (melt-up risk)
    call_tail_risk_score: float
    melt_up_probability: float
    melt_up_magnitude_estimate: float
    
    # Risk factors
    risk_concentration_areas: List[Dict[str, Any]]
    tail_hedging_activity: float
    
    # Market stress indicators
    stress_level: float
    liquidity_concerns: float


@dataclass
class RegimeClassificationResult:
    """Complete regime classification result"""
    # Primary classification
    primary_regime: MarketRegime
    regime_confidence: float
    regime_stability: float
    
    # Secondary indicators
    secondary_regimes: List[Tuple[MarketRegime, float]]  # (regime, probability)
    regime_transition_probability: float
    
    # Detailed analysis components
    skew_pattern_analysis: SkewPatternAnalysis
    smile_analysis: SmileAnalysisResult
    institutional_flow: InstitutionalFlowAnalysis
    tail_risk_assessment: TailRiskAssessment
    
    # Integration metrics
    component_agreement_score: float
    overall_consistency: float
    
    # Actionable insights
    trading_signals: List[Dict[str, Any]]
    risk_warnings: List[str]
    opportunity_signals: List[str]
    
    # Metadata
    analysis_timestamp: datetime
    processing_time_ms: float
    data_quality_score: float


class SkewPatternAnalyzer:
    """Analyze volatility skew patterns"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # Pattern recognition thresholds
        self.steep_skew_threshold = config.get('steep_skew_threshold', 0.015)
        self.moderate_skew_threshold = config.get('moderate_skew_threshold', 0.008)
        self.flat_skew_threshold = config.get('flat_skew_threshold', 0.003)
        
    def analyze_skew_pattern(self, surface_result: VolatilitySurfaceResult,
                           skew_data: IVSkewData) -> SkewPatternAnalysis:
        """
        Analyze volatility skew patterns
        
        Args:
            surface_result: Volatility surface analysis result
            skew_data: IV skew data
            
        Returns:
            SkewPatternAnalysis with complete pattern analysis
        """
        try:
            # Calculate skew steepness metrics
            put_steepness = self._calculate_put_skew_steepness(surface_result, skew_data)
            call_steepness = self._calculate_call_skew_steepness(surface_result, skew_data)
            
            # Determine primary pattern
            primary_pattern = self._classify_primary_skew_pattern(put_steepness, call_steepness, surface_result)
            
            # Calculate pattern strength and confidence
            pattern_strength = self._calculate_pattern_strength(primary_pattern, put_steepness, call_steepness)
            pattern_confidence = self._calculate_pattern_confidence(surface_result, primary_pattern)
            
            # Analyze asymmetry
            skew_asymmetry = self._calculate_skew_asymmetry(surface_result, skew_data)
            
            # Extract pattern features
            pattern_features = self._extract_pattern_features(surface_result, skew_data, primary_pattern)
            
            # Detect unusual characteristics
            unusual_characteristics = self._detect_unusual_pattern_characteristics(surface_result, primary_pattern)
            
            # Analyze institutional signatures
            institutional_signature, flow_imbalance = self._analyze_institutional_signatures(skew_data, surface_result)
            
            return SkewPatternAnalysis(
                primary_pattern=primary_pattern,
                pattern_strength=float(pattern_strength),
                pattern_confidence=float(pattern_confidence),
                put_skew_steepness=float(put_steepness),
                call_skew_steepness=float(call_steepness),
                skew_asymmetry=float(skew_asymmetry),
                pattern_features=pattern_features,
                unusual_characteristics=unusual_characteristics,
                institutional_signature=float(institutional_signature),
                flow_imbalance_score=float(flow_imbalance)
            )
            
        except Exception as e:
            self.logger.error(f"Skew pattern analysis failed: {e}")
            raise
    
    def _calculate_put_skew_steepness(self, surface_result: VolatilitySurfaceResult, 
                                    skew_data: IVSkewData) -> float:
        """Calculate put skew steepness"""
        
        # Use risk reversal and skew slope metrics
        put_skew_indicators = [
            abs(surface_result.risk_reversal_25d),
            abs(surface_result.risk_reversal_10d),
            surface_result.skew_slope_25d if surface_result.skew_slope_25d > 0 else 0,
            surface_result.skew_slope_10d if surface_result.skew_slope_10d > 0 else 0,
            surface_result.put_skew_dominance
        ]
        
        # Weighted average with emphasis on put dominance
        weights = [0.2, 0.2, 0.25, 0.25, 0.1]
        put_steepness = np.average(put_skew_indicators, weights=weights)
        
        return float(put_steepness)
    
    def _calculate_call_skew_steepness(self, surface_result: VolatilitySurfaceResult,
                                     skew_data: IVSkewData) -> float:
        """Calculate call skew steepness"""
        
        # Reverse skew indicators
        call_skew_indicators = [
            abs(surface_result.risk_reversal_25d) if surface_result.risk_reversal_25d < 0 else 0,
            abs(surface_result.risk_reversal_10d) if surface_result.risk_reversal_10d < 0 else 0,
            abs(surface_result.skew_slope_25d) if surface_result.skew_slope_25d < 0 else 0,
            abs(surface_result.skew_slope_10d) if surface_result.skew_slope_10d < 0 else 0,
            surface_result.call_skew_strength
        ]
        
        weights = [0.2, 0.2, 0.25, 0.25, 0.1]
        call_steepness = np.average(call_skew_indicators, weights=weights)
        
        return float(call_steepness)
    
    def _classify_primary_skew_pattern(self, put_steepness: float, call_steepness: float,
                                     surface_result: VolatilitySurfaceResult) -> SkewPattern:
        """Classify primary skew pattern"""
        
        # Determine dominant direction
        if put_steepness > call_steepness:
            # Put skew dominance
            if put_steepness > self.steep_skew_threshold:
                return SkewPattern.STEEP_PUT_SKEW
            elif put_steepness > self.moderate_skew_threshold:
                return SkewPattern.MODERATE_PUT_SKEW
            else:
                return SkewPattern.FLAT_SKEW
        
        elif call_steepness > put_steepness:
            # Call skew (reverse skew)
            if call_steepness > self.moderate_skew_threshold:
                return SkewPattern.REVERSE_SKEW
            else:
                return SkewPattern.FLAT_SKEW
        
        else:
            # Similar steepness
            if max(put_steepness, call_steepness) < self.flat_skew_threshold:
                return SkewPattern.FLAT_SKEW
            else:
                return SkewPattern.IRREGULAR_SKEW
    
    def _calculate_pattern_strength(self, pattern: SkewPattern, put_steepness: float, 
                                  call_steepness: float) -> float:
        """Calculate pattern strength"""
        
        if pattern == SkewPattern.STEEP_PUT_SKEW:
            strength = min(1.0, put_steepness / self.steep_skew_threshold)
        elif pattern == SkewPattern.MODERATE_PUT_SKEW:
            strength = min(1.0, put_steepness / self.moderate_skew_threshold)
        elif pattern == SkewPattern.REVERSE_SKEW:
            strength = min(1.0, call_steepness / self.moderate_skew_threshold)
        elif pattern == SkewPattern.FLAT_SKEW:
            strength = 1.0 - max(put_steepness, call_steepness) / self.flat_skew_threshold
        else:
            strength = 0.5  # Irregular patterns get medium strength
        
        return max(0.0, min(1.0, strength))
    
    def _calculate_pattern_confidence(self, surface_result: VolatilitySurfaceResult,
                                    pattern: SkewPattern) -> float:
        """Calculate pattern classification confidence"""
        
        confidence_factors = []
        
        # Surface quality factor
        confidence_factors.append(surface_result.surface_quality_score)
        
        # Data completeness factor
        confidence_factors.append(surface_result.data_completeness)
        
        # Interpolation quality factor
        confidence_factors.append(surface_result.interpolation_quality)
        
        # Outlier penalty
        outlier_penalty = surface_result.outlier_count / max(len(surface_result.surface_strikes), 1)
        confidence_factors.append(1.0 - min(1.0, outlier_penalty))
        
        return float(np.mean(confidence_factors))
    
    def _calculate_skew_asymmetry(self, surface_result: VolatilitySurfaceResult,
                                skew_data: IVSkewData) -> float:
        """Calculate overall skew asymmetry"""
        
        # Combine multiple asymmetry measures
        asymmetry_measures = [
            surface_result.smile_asymmetry,
            surface_result.put_skew_dominance - surface_result.call_skew_strength,
            (surface_result.risk_reversal_25d + surface_result.risk_reversal_10d) / 2
        ]
        
        return float(np.mean(asymmetry_measures))
    
    def _extract_pattern_features(self, surface_result: VolatilitySurfaceResult,
                                skew_data: IVSkewData, pattern: SkewPattern) -> Dict[str, float]:
        """Extract detailed pattern features"""
        
        features = {
            # Basic skew features
            'skew_steepness': float(surface_result.skew_steepness),
            'skew_convexity': float(surface_result.skew_convexity),
            'smile_curvature': float(surface_result.smile_curvature),
            
            # Wing characteristics
            'put_wing_steepness': float(surface_result.put_skew_dominance),
            'call_wing_steepness': float(surface_result.call_skew_strength),
            
            # Risk reversal features
            'rr_25d': float(surface_result.risk_reversal_25d),
            'rr_10d': float(surface_result.risk_reversal_10d),
            
            # Surface quality features
            'surface_smoothness': float(surface_result.surface_quality_score),
            'data_coverage': float(surface_result.data_completeness),
            
            # Pattern-specific features
            'pattern_consistency': self._calculate_pattern_consistency(pattern, surface_result),
            'pattern_extremity': self._calculate_pattern_extremity(pattern, surface_result)
        }
        
        return features
    
    def _calculate_pattern_consistency(self, pattern: SkewPattern, 
                                     surface_result: VolatilitySurfaceResult) -> float:
        """Calculate pattern internal consistency"""
        
        # Check if different skew measures agree
        skew_measures = [
            surface_result.skew_slope_25d,
            surface_result.skew_slope_10d,
            surface_result.risk_reversal_25d,
            surface_result.risk_reversal_10d
        ]
        
        # Normalize signs for consistency check
        normalized_signs = [np.sign(measure) for measure in skew_measures]
        
        # Consistency = how many measures agree on direction
        if len(set(normalized_signs)) == 1:
            consistency = 1.0
        elif len(set(normalized_signs)) == 2:
            consistency = 0.6
        else:
            consistency = 0.3
        
        return float(consistency)
    
    def _calculate_pattern_extremity(self, pattern: SkewPattern,
                                   surface_result: VolatilitySurfaceResult) -> float:
        """Calculate pattern extremity level"""
        
        extremity_indicators = [
            abs(surface_result.risk_reversal_25d) / 0.2,  # Normalize to typical range
            abs(surface_result.risk_reversal_10d) / 0.15,
            surface_result.skew_steepness / 0.05,
            abs(surface_result.smile_asymmetry) / 0.3
        ]
        
        return float(min(1.0, np.mean(extremity_indicators)))
    
    def _detect_unusual_pattern_characteristics(self, surface_result: VolatilitySurfaceResult,
                                              pattern: SkewPattern) -> List[str]:
        """Detect unusual pattern characteristics"""
        
        unusual_characteristics = []
        
        # Check for extreme values
        if abs(surface_result.risk_reversal_25d) > 0.15:
            unusual_characteristics.append("extreme_risk_reversal")
        
        if surface_result.skew_steepness > 0.03:
            unusual_characteristics.append("exceptionally_steep_skew")
        
        if surface_result.skew_convexity > 0.01:
            unusual_characteristics.append("high_convexity")
        
        if abs(surface_result.smile_curvature) > 0.15:
            unusual_characteristics.append("extreme_smile_curvature")
        
        # Pattern-specific checks
        if pattern == SkewPattern.REVERSE_SKEW:
            unusual_characteristics.append("reverse_skew_pattern")
        
        if pattern == SkewPattern.IRREGULAR_SKEW:
            unusual_characteristics.append("irregular_surface_pattern")
        
        # Surface quality issues
        if surface_result.surface_quality_score < 0.6:
            unusual_characteristics.append("poor_surface_quality")
        
        if surface_result.outlier_count > len(surface_result.surface_strikes) * 0.15:
            unusual_characteristics.append("high_outlier_count")
        
        return unusual_characteristics
    
    def _analyze_institutional_signatures(self, skew_data: IVSkewData,
                                        surface_result: VolatilitySurfaceResult) -> Tuple[float, float]:
        """Analyze institutional flow signatures in skew patterns"""
        
        # Volume-based institutional indicators
        total_call_volume = np.sum(skew_data.call_volumes)
        total_put_volume = np.sum(skew_data.put_volumes)
        total_volume = total_call_volume + total_put_volume
        
        if total_volume > 0:
            volume_imbalance = (total_put_volume - total_call_volume) / total_volume
        else:
            volume_imbalance = 0.0
        
        # OI-based institutional indicators
        total_call_oi = np.sum(skew_data.call_oi)
        total_put_oi = np.sum(skew_data.put_oi)
        total_oi = total_call_oi + total_put_oi
        
        if total_oi > 0:
            oi_imbalance = (total_put_oi - total_call_oi) / total_oi
        else:
            oi_imbalance = 0.0
        
        # Surface characteristics indicating institutional activity
        institutional_indicators = [
            abs(volume_imbalance) * 0.3,
            abs(oi_imbalance) * 0.3,
            min(1.0, abs(surface_result.smile_asymmetry) / 0.2) * 0.2,  # Unusual asymmetry
            min(1.0, surface_result.skew_steepness / 0.02) * 0.2       # Steep skew
        ]
        
        institutional_signature = np.sum(institutional_indicators)
        flow_imbalance_score = (abs(volume_imbalance) + abs(oi_imbalance)) / 2
        
        return institutional_signature, flow_imbalance_score


class SmileAnalyzer:
    """Analyze volatility smile characteristics"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # Smile analysis thresholds
        self.symmetric_threshold = config.get('symmetric_threshold', 0.1)
        self.curvature_threshold = config.get('curvature_threshold', 0.05)
        
    def analyze_volatility_smile(self, surface_result: VolatilitySurfaceResult,
                               skew_data: IVSkewData,
                               advanced_metrics: AdvancedIVMetrics) -> SmileAnalysisResult:
        """
        Analyze complete volatility smile characteristics
        
        Args:
            surface_result: Volatility surface analysis result
            skew_data: IV skew data
            advanced_metrics: Advanced IV metrics
            
        Returns:
            SmileAnalysisResult with complete smile analysis
        """
        try:
            # Classify smile characteristic
            smile_characteristic = self._classify_smile_characteristic(surface_result)
            
            # Calculate smile quality and stability
            smile_quality = self._calculate_smile_quality(surface_result)
            smile_stability = self._calculate_smile_stability(surface_result, advanced_metrics)
            
            # Analyze curvature
            curvature_score = self._analyze_smile_curvature(surface_result)
            
            # Calculate wing asymmetry
            wing_asymmetry = self._calculate_wing_asymmetry(surface_result)
            
            # Extract ATM IV level
            atm_iv_level = float(surface_result.smile_atm_iv)
            
            # Assess tail risk
            tail_risk_indicator = self._calculate_tail_risk_indicator(surface_result, advanced_metrics)
            crash_risk_probability = self._calculate_crash_risk_probability(surface_result, advanced_metrics)
            
            # Detect arbitrage opportunities
            arbitrage_opportunities = self._detect_smile_arbitrage_opportunities(surface_result, skew_data)
            
            # Assess surface consistency
            surface_consistency = self._assess_surface_consistency(surface_result)
            
            return SmileAnalysisResult(
                smile_characteristic=smile_characteristic,
                smile_quality=float(smile_quality),
                smile_stability=float(smile_stability),
                curvature_score=float(curvature_score),
                wing_asymmetry=float(wing_asymmetry),
                atm_iv_level=atm_iv_level,
                tail_risk_indicator=float(tail_risk_indicator),
                crash_risk_probability=float(crash_risk_probability),
                arbitrage_opportunities=arbitrage_opportunities,
                surface_consistency=float(surface_consistency)
            )
            
        except Exception as e:
            self.logger.error(f"Volatility smile analysis failed: {e}")
            raise
    
    def _classify_smile_characteristic(self, surface_result: VolatilitySurfaceResult) -> SmileCharacteristic:
        """Classify primary smile characteristic"""
        
        asymmetry = surface_result.smile_asymmetry
        curvature = surface_result.smile_curvature
        
        # Check for flat smile first
        if abs(curvature) < 0.01 and abs(asymmetry) < 0.05:
            return SmileCharacteristic.FLAT_SMILE
        
        # Check for inverted smile (negative curvature)
        if curvature < -0.02:
            return SmileCharacteristic.INVERTED_SMILE
        
        # Check asymmetry patterns
        if abs(asymmetry) > self.symmetric_threshold:
            if asymmetry > 0:
                # Higher IV on put side
                return SmileCharacteristic.ASYMMETRIC_PUT_WING
            else:
                # Higher IV on call side
                return SmileCharacteristic.ASYMMETRIC_CALL_WING
        
        # Default to symmetric
        return SmileCharacteristic.SYMMETRIC_SMILE
    
    def _calculate_smile_quality(self, surface_result: VolatilitySurfaceResult) -> float:
        """Calculate overall smile quality"""
        
        quality_factors = []
        
        # Surface fitting quality
        quality_factors.append(surface_result.surface_quality_score)
        
        # Data completeness
        quality_factors.append(surface_result.data_completeness)
        
        # Interpolation quality
        quality_factors.append(surface_result.interpolation_quality)
        
        # Outlier penalty
        outlier_ratio = surface_result.outlier_count / max(len(surface_result.surface_strikes), 1)
        outlier_penalty = 1.0 - min(1.0, outlier_ratio * 2)
        quality_factors.append(outlier_penalty)
        
        # Smoothness factor (moderate curvature is good)
        if abs(surface_result.smile_curvature) > 0.2:
            smoothness_factor = 0.7  # Penalize extreme curvature
        else:
            smoothness_factor = 1.0
        quality_factors.append(smoothness_factor)
        
        return np.mean(quality_factors)
    
    def _calculate_smile_stability(self, surface_result: VolatilitySurfaceResult,
                                 advanced_metrics: AdvancedIVMetrics) -> float:
        """Calculate smile stability across time"""
        
        # Use surface stability score from advanced metrics
        base_stability = advanced_metrics.surface_stability_score
        
        # Adjust for smile-specific factors
        stability_adjustments = []
        
        # Penalize extreme curvature changes
        if abs(surface_result.smile_curvature) > 0.1:
            stability_adjustments.append(0.9)
        else:
            stability_adjustments.append(1.0)
        
        # Penalize high volatility clustering
        clustering_penalty = 1.0 - (advanced_metrics.volatility_clustering_score * 0.2)
        stability_adjustments.append(clustering_penalty)
        
        # Reward consistent surface quality
        if surface_result.surface_quality_score > 0.8:
            stability_adjustments.append(1.1)
        else:
            stability_adjustments.append(1.0)
        
        # Calculate final stability
        adjustment_factor = np.mean(stability_adjustments)
        final_stability = min(1.0, base_stability * adjustment_factor)
        
        return final_stability
    
    def _analyze_smile_curvature(self, surface_result: VolatilitySurfaceResult) -> float:
        """Analyze smile curvature characteristics"""
        
        base_curvature = surface_result.smile_curvature
        
        # Normalize curvature score (typical range: -0.1 to 0.2)
        if base_curvature > 0:
            # Positive curvature (normal smile)
            curvature_score = min(1.0, base_curvature / 0.15)
        else:
            # Negative curvature (inverted smile)
            curvature_score = max(-1.0, base_curvature / 0.1)
        
        return curvature_score
    
    def _calculate_wing_asymmetry(self, surface_result: VolatilitySurfaceResult) -> float:
        """Calculate wing asymmetry score"""
        
        # Primary asymmetry measure
        primary_asymmetry = surface_result.smile_asymmetry
        
        # Secondary asymmetry measures
        skew_asymmetry = (surface_result.put_skew_dominance - surface_result.call_skew_strength) / 2
        risk_reversal_asymmetry = (surface_result.risk_reversal_25d + surface_result.risk_reversal_10d) / 2
        
        # Weighted average
        weights = [0.5, 0.25, 0.25]
        wing_asymmetry = np.average([primary_asymmetry, skew_asymmetry, risk_reversal_asymmetry], weights=weights)
        
        return wing_asymmetry
    
    def _calculate_tail_risk_indicator(self, surface_result: VolatilitySurfaceResult,
                                     advanced_metrics: AdvancedIVMetrics) -> float:
        """Calculate tail risk indicator from smile characteristics"""
        
        tail_risk_components = []
        
        # Put tail risk (from advanced metrics)
        tail_risk_components.append(advanced_metrics.tail_risk_put)
        
        # Call tail risk (from advanced metrics)
        tail_risk_components.append(advanced_metrics.tail_risk_call)
        
        # Skew steepness contribution to tail risk
        skew_tail_risk = min(1.0, surface_result.skew_steepness / 0.02)
        tail_risk_components.append(skew_tail_risk)
        
        # Wing asymmetry contribution
        asymmetry_tail_risk = min(1.0, abs(surface_result.smile_asymmetry) / 0.3)
        tail_risk_components.append(asymmetry_tail_risk)
        
        # Weighted average
        weights = [0.4, 0.3, 0.2, 0.1]
        tail_risk_indicator = np.average(tail_risk_components, weights=weights)
        
        return tail_risk_indicator
    
    def _calculate_crash_risk_probability(self, surface_result: VolatilitySurfaceResult,
                                        advanced_metrics: AdvancedIVMetrics) -> float:
        """Calculate crash risk probability from smile and skew characteristics"""
        
        # Base crash probability from advanced metrics
        base_crash_prob = advanced_metrics.crash_probability
        
        # Smile-specific crash indicators
        crash_indicators = []
        
        # Steep put skew indicates crash concern
        if surface_result.put_skew_dominance > 0.1:
            crash_indicators.append(surface_result.put_skew_dominance)
        else:
            crash_indicators.append(0.0)
        
        # High risk reversal indicates crash hedging
        rr_crash_indicator = max(0, (surface_result.risk_reversal_25d + surface_result.risk_reversal_10d) / 2)
        crash_indicators.append(rr_crash_indicator)
        
        # Extreme smile asymmetry
        if surface_result.smile_asymmetry > 0.2:
            crash_indicators.append(surface_result.smile_asymmetry)
        else:
            crash_indicators.append(0.0)
        
        # Combine indicators
        smile_crash_prob = np.mean(crash_indicators)
        
        # Final crash probability (weighted average)
        final_crash_prob = 0.6 * base_crash_prob + 0.4 * smile_crash_prob
        
        return min(1.0, final_crash_prob)
    
    def _detect_smile_arbitrage_opportunities(self, surface_result: VolatilitySurfaceResult,
                                            skew_data: IVSkewData) -> List[Dict[str, float]]:
        """Detect arbitrage opportunities from smile characteristics"""
        
        arbitrage_opportunities = []
        
        # Butterfly arbitrage (excessive smile curvature)
        if abs(surface_result.smile_curvature) > 0.15:
            arbitrage_opportunities.append({
                'type': 'butterfly_arbitrage',
                'magnitude': float(abs(surface_result.smile_curvature)),
                'confidence': 0.7,
                'expected_profit': float(abs(surface_result.smile_curvature) * 100)
            })
        
        # Risk reversal arbitrage (excessive asymmetry)
        if abs(surface_result.smile_asymmetry) > 0.25:
            arbitrage_opportunities.append({
                'type': 'risk_reversal_arbitrage',
                'magnitude': float(abs(surface_result.smile_asymmetry)),
                'confidence': 0.6,
                'expected_profit': float(abs(surface_result.smile_asymmetry) * 50)
            })
        
        # Put spread arbitrage (steep put skew)
        if surface_result.put_skew_dominance > 0.15:
            arbitrage_opportunities.append({
                'type': 'put_spread_arbitrage',
                'magnitude': float(surface_result.put_skew_dominance),
                'confidence': 0.8,
                'expected_profit': float(surface_result.put_skew_dominance * 75)
            })
        
        # Surface inconsistency arbitrage
        if surface_result.surface_quality_score < 0.6:
            arbitrage_opportunities.append({
                'type': 'surface_inconsistency',
                'magnitude': float(1.0 - surface_result.surface_quality_score),
                'confidence': 0.5,
                'expected_profit': float((1.0 - surface_result.surface_quality_score) * 30)
            })
        
        return arbitrage_opportunities
    
    def _assess_surface_consistency(self, surface_result: VolatilitySurfaceResult) -> float:
        """Assess overall surface consistency"""
        
        consistency_factors = []
        
        # R-squared consistency
        consistency_factors.append(surface_result.surface_r_squared)
        
        # Quality score consistency
        consistency_factors.append(surface_result.surface_quality_score)
        
        # Data completeness consistency
        consistency_factors.append(surface_result.data_completeness)
        
        # Outlier consistency (fewer outliers = more consistent)
        outlier_ratio = surface_result.outlier_count / max(len(surface_result.surface_strikes), 1)
        outlier_consistency = 1.0 - min(1.0, outlier_ratio)
        consistency_factors.append(outlier_consistency)
        
        # Interpolation consistency
        consistency_factors.append(surface_result.interpolation_quality)
        
        return np.mean(consistency_factors)


class InstitutionalFlowDetector:
    """Detect institutional flow patterns in volatility surface"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # Flow detection thresholds
        self.significant_flow_threshold = config.get('significant_flow_threshold', 0.15)
        self.volume_threshold_multiplier = config.get('volume_threshold_multiplier', 2.0)
        
    def detect_institutional_flow(self, skew_data: IVSkewData, 
                                surface_result: VolatilitySurfaceResult,
                                advanced_metrics: AdvancedIVMetrics) -> InstitutionalFlowAnalysis:
        """
        Detect institutional flow patterns
        
        Args:
            skew_data: IV skew data
            surface_result: Volatility surface result
            advanced_metrics: Advanced IV metrics
            
        Returns:
            InstitutionalFlowAnalysis with complete flow analysis
        """
        try:
            # Analyze volume and OI patterns
            volume_indicators = self._analyze_volume_flow_indicators(skew_data)
            oi_indicators = self._analyze_oi_positioning_signals(skew_data)
            
            # Detect unusual surface changes
            surface_change_indicators = self._detect_unusual_surface_changes(surface_result, advanced_metrics)
            
            # Determine flow direction and magnitude
            flow_direction, flow_magnitude = self._determine_flow_direction_and_magnitude(
                volume_indicators, oi_indicators, surface_change_indicators
            )
            
            # Calculate flow detection confidence
            flow_confidence = self._calculate_flow_detection_confidence(
                volume_indicators, oi_indicators, surface_change_indicators
            )
            
            # Analyze flow persistence and acceleration
            flow_persistence, flow_acceleration = self._analyze_flow_dynamics(
                volume_indicators, oi_indicators
            )
            
            # Assess institutional risk appetite
            risk_appetite = self._assess_institutional_risk_appetite(skew_data, surface_result)
            
            # Classify hedging vs speculation
            hedging_vs_speculation = self._classify_hedging_vs_speculation(
                volume_indicators, oi_indicators, surface_result
            )
            
            return InstitutionalFlowAnalysis(
                flow_detection_confidence=float(flow_confidence),
                flow_direction=flow_direction,
                flow_magnitude=float(flow_magnitude),
                unusual_surface_changes=surface_change_indicators,
                volume_flow_indicators=volume_indicators,
                oi_positioning_signals=oi_indicators,
                flow_persistence=float(flow_persistence),
                flow_acceleration=float(flow_acceleration),
                institutional_risk_appetite=float(risk_appetite),
                hedging_vs_speculation=hedging_vs_speculation
            )
            
        except Exception as e:
            self.logger.error(f"Institutional flow detection failed: {e}")
            raise
    
    def _analyze_volume_flow_indicators(self, skew_data: IVSkewData) -> Dict[str, float]:
        """Analyze volume-based flow indicators"""
        
        total_call_volume = np.sum(skew_data.call_volumes)
        total_put_volume = np.sum(skew_data.put_volumes)
        total_volume = total_call_volume + total_put_volume
        
        indicators = {}
        
        if total_volume > 0:
            # Volume imbalance
            indicators['volume_imbalance'] = (total_put_volume - total_call_volume) / total_volume
            
            # Volume concentration
            max_call_volume = np.max(skew_data.call_volumes) if len(skew_data.call_volumes) > 0 else 0
            max_put_volume = np.max(skew_data.put_volumes) if len(skew_data.put_volumes) > 0 else 0
            
            indicators['call_volume_concentration'] = max_call_volume / (total_call_volume + 1e-10)
            indicators['put_volume_concentration'] = max_put_volume / (total_put_volume + 1e-10)
            
            # Volume intensity
            avg_volume = total_volume / len(skew_data.strikes) if len(skew_data.strikes) > 0 else 0
            indicators['volume_intensity'] = min(2.0, total_volume / (avg_volume * 10 + 1e-10))
            
            # Unusual volume patterns
            volume_cv = np.std(np.concatenate([skew_data.call_volumes, skew_data.put_volumes])) / (avg_volume + 1e-10)
            indicators['volume_dispersion'] = min(3.0, volume_cv)
        else:
            indicators = {
                'volume_imbalance': 0.0,
                'call_volume_concentration': 0.0,
                'put_volume_concentration': 0.0,
                'volume_intensity': 0.0,
                'volume_dispersion': 0.0
            }
        
        return indicators
    
    def _analyze_oi_positioning_signals(self, skew_data: IVSkewData) -> Dict[str, float]:
        """Analyze open interest positioning signals"""
        
        total_call_oi = np.sum(skew_data.call_oi)
        total_put_oi = np.sum(skew_data.put_oi)
        total_oi = total_call_oi + total_put_oi
        
        indicators = {}
        
        if total_oi > 0:
            # OI imbalance
            indicators['oi_imbalance'] = (total_put_oi - total_call_oi) / total_oi
            
            # OI concentration
            max_call_oi = np.max(skew_data.call_oi) if len(skew_data.call_oi) > 0 else 0
            max_put_oi = np.max(skew_data.put_oi) if len(skew_data.put_oi) > 0 else 0
            
            indicators['call_oi_concentration'] = max_call_oi / (total_call_oi + 1e-10)
            indicators['put_oi_concentration'] = max_put_oi / (total_put_oi + 1e-10)
            
            # OI buildup intensity
            avg_oi = total_oi / len(skew_data.strikes) if len(skew_data.strikes) > 0 else 0
            indicators['oi_buildup_intensity'] = min(2.0, total_oi / (avg_oi * 20 + 1e-10))
            
            # Put/Call OI ratio
            indicators['put_call_oi_ratio'] = total_put_oi / (total_call_oi + 1e-10)
        else:
            indicators = {
                'oi_imbalance': 0.0,
                'call_oi_concentration': 0.0,
                'put_oi_concentration': 0.0,
                'oi_buildup_intensity': 0.0,
                'put_call_oi_ratio': 1.0
            }
        
        return indicators
    
    def _detect_unusual_surface_changes(self, surface_result: VolatilitySurfaceResult,
                                      advanced_metrics: AdvancedIVMetrics) -> Dict[str, float]:
        """Detect unusual surface changes indicating institutional activity"""
        
        surface_changes = {}
        
        # Surface quality deterioration
        surface_changes['quality_deterioration'] = max(0.0, 1.0 - surface_result.surface_quality_score)
        
        # Unusual skew changes
        surface_changes['skew_steepness_change'] = min(1.0, surface_result.skew_steepness / 0.03)
        
        # Smile asymmetry changes
        surface_changes['asymmetry_change'] = min(1.0, abs(surface_result.smile_asymmetry) / 0.3)
        
        # Risk reversal extremes
        rr_extreme = max(abs(surface_result.risk_reversal_25d), abs(surface_result.risk_reversal_10d))
        surface_changes['risk_reversal_extreme'] = min(1.0, rr_extreme / 0.15)
        
        # Institutional flow score from advanced metrics
        surface_changes['institutional_flow_score'] = advanced_metrics.institutional_flow_score
        
        # Volatility clustering changes
        surface_changes['clustering_change'] = advanced_metrics.volatility_clustering_score
        
        return surface_changes
    
    def _determine_flow_direction_and_magnitude(self, volume_indicators: Dict[str, float],
                                              oi_indicators: Dict[str, float],
                                              surface_indicators: Dict[str, float]) -> Tuple[str, float]:
        """Determine overall flow direction and magnitude"""
        
        # Collect directional indicators
        bullish_indicators = []
        bearish_indicators = []
        
        # Volume-based signals
        if volume_indicators['volume_imbalance'] > 0:
            bearish_indicators.append(abs(volume_indicators['volume_imbalance']))
        else:
            bullish_indicators.append(abs(volume_indicators['volume_imbalance']))
        
        # OI-based signals
        if oi_indicators['oi_imbalance'] > 0:
            bearish_indicators.append(abs(oi_indicators['oi_imbalance']))
        else:
            bullish_indicators.append(abs(oi_indicators['oi_imbalance']))
        
        # Surface-based signals
        if surface_indicators.get('skew_steepness_change', 0) > 0.5:
            bearish_indicators.append(surface_indicators['skew_steepness_change'])
        
        if surface_indicators.get('risk_reversal_extreme', 0) > 0.5:
            bearish_indicators.append(surface_indicators['risk_reversal_extreme'])
        
        # Calculate net direction
        bullish_score = np.mean(bullish_indicators) if bullish_indicators else 0.0
        bearish_score = np.mean(bearish_indicators) if bearish_indicators else 0.0
        
        # Determine direction
        net_score = bearish_score - bullish_score
        magnitude = max(bullish_score, bearish_score)
        
        if abs(net_score) < 0.1:
            direction = 'neutral'
        elif net_score > 0.1:
            direction = 'bearish'
        elif net_score < -0.1:
            direction = 'bullish'
        else:
            direction = 'mixed'
        
        return direction, magnitude
    
    def _calculate_flow_detection_confidence(self, volume_indicators: Dict[str, float],
                                           oi_indicators: Dict[str, float],
                                           surface_indicators: Dict[str, float]) -> float:
        """Calculate confidence in flow detection"""
        
        confidence_factors = []
        
        # Volume signal strength
        volume_strength = max(
            abs(volume_indicators.get('volume_imbalance', 0)),
            volume_indicators.get('volume_intensity', 0) / 2,
            volume_indicators.get('volume_dispersion', 0) / 3
        )
        confidence_factors.append(min(1.0, volume_strength))
        
        # OI signal strength
        oi_strength = max(
            abs(oi_indicators.get('oi_imbalance', 0)),
            oi_indicators.get('oi_buildup_intensity', 0) / 2,
            abs(oi_indicators.get('put_call_oi_ratio', 1.0) - 1.0)
        )
        confidence_factors.append(min(1.0, oi_strength))
        
        # Surface signal strength
        surface_strength = max(
            surface_indicators.get('institutional_flow_score', 0),
            surface_indicators.get('skew_steepness_change', 0),
            surface_indicators.get('risk_reversal_extreme', 0)
        )
        confidence_factors.append(min(1.0, surface_strength))
        
        return np.mean(confidence_factors)
    
    def _analyze_flow_dynamics(self, volume_indicators: Dict[str, float],
                             oi_indicators: Dict[str, float]) -> Tuple[float, float]:
        """Analyze flow persistence and acceleration"""
        
        # Mock flow dynamics analysis (would use time series in production)
        
        # Flow persistence based on OI buildup
        oi_buildup = oi_indicators.get('oi_buildup_intensity', 0)
        flow_persistence = min(1.0, oi_buildup / 1.5)
        
        # Flow acceleration based on volume intensity
        volume_intensity = volume_indicators.get('volume_intensity', 0)
        flow_acceleration = min(1.0, volume_intensity / 1.8)
        
        return flow_persistence, flow_acceleration
    
    def _assess_institutional_risk_appetite(self, skew_data: IVSkewData,
                                          surface_result: VolatilitySurfaceResult) -> float:
        """Assess institutional risk appetite"""
        
        risk_appetite_indicators = []
        
        # High call activity indicates risk appetite
        total_call_volume = np.sum(skew_data.call_volumes)
        total_put_volume = np.sum(skew_data.put_volumes)
        total_volume = total_call_volume + total_put_volume
        
        if total_volume > 0:
            call_ratio = total_call_volume / total_volume
            risk_appetite_indicators.append(call_ratio)
        
        # Low put skew indicates comfort with risk
        put_skew_comfort = 1.0 - min(1.0, surface_result.put_skew_dominance / 0.2)
        risk_appetite_indicators.append(put_skew_comfort)
        
        # Low crash probability indicates risk appetite
        if hasattr(surface_result, 'crash_probability'):
            crash_comfort = 1.0 - getattr(surface_result, 'crash_probability', 0.2)
            risk_appetite_indicators.append(crash_comfort)
        
        return np.mean(risk_appetite_indicators) if risk_appetite_indicators else 0.5
    
    def _classify_hedging_vs_speculation(self, volume_indicators: Dict[str, float],
                                       oi_indicators: Dict[str, float],
                                       surface_result: VolatilitySurfaceResult) -> str:
        """Classify institutional activity as hedging vs speculation"""
        
        hedging_indicators = []
        speculation_indicators = []
        
        # High put activity suggests hedging
        if oi_indicators.get('oi_imbalance', 0) > 0.1:
            hedging_indicators.append(oi_indicators['oi_imbalance'])
        
        # High put skew suggests hedging demand
        if surface_result.put_skew_dominance > 0.1:
            hedging_indicators.append(surface_result.put_skew_dominance)
        
        # High volume intensity might suggest speculation
        if volume_indicators.get('volume_intensity', 0) > 1.5:
            speculation_indicators.append(volume_indicators['volume_intensity'] / 2)
        
        # Extreme smile characteristics might suggest speculation
        if abs(surface_result.smile_asymmetry) > 0.2:
            speculation_indicators.append(abs(surface_result.smile_asymmetry))
        
        hedging_score = np.mean(hedging_indicators) if hedging_indicators else 0.0
        speculation_score = np.mean(speculation_indicators) if speculation_indicators else 0.0
        
        if hedging_score > speculation_score * 1.2:
            return 'hedging'
        elif speculation_score > hedging_score * 1.2:
            return 'speculation'
        else:
            return 'mixed'


class TailRiskAnalyzer:
    """Comprehensive tail risk analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # Tail risk thresholds
        self.high_tail_risk_threshold = config.get('high_tail_risk_threshold', 0.7)
        self.crash_probability_threshold = config.get('crash_probability_threshold', 0.3)
        
    def analyze_tail_risk(self, skew_data: IVSkewData,
                         surface_result: VolatilitySurfaceResult,
                         advanced_metrics: AdvancedIVMetrics) -> TailRiskAssessment:
        """
        Comprehensive tail risk analysis
        
        Args:
            skew_data: IV skew data
            surface_result: Volatility surface result
            advanced_metrics: Advanced IV metrics
            
        Returns:
            TailRiskAssessment with complete tail risk analysis
        """
        try:
            # Analyze put tail risk (crash risk)
            put_tail_analysis = self._analyze_put_tail_risk(skew_data, surface_result, advanced_metrics)
            
            # Analyze call tail risk (melt-up risk)
            call_tail_analysis = self._analyze_call_tail_risk(skew_data, surface_result, advanced_metrics)
            
            # Identify risk concentration areas
            risk_concentration_areas = self._identify_risk_concentration_areas(skew_data, surface_result)
            
            # Analyze tail hedging activity
            tail_hedging_activity = self._analyze_tail_hedging_activity(skew_data, surface_result)
            
            # Assess market stress level
            stress_level = self._assess_market_stress_level(surface_result, advanced_metrics)
            
            # Evaluate liquidity concerns
            liquidity_concerns = self._evaluate_liquidity_concerns(skew_data, surface_result)
            
            # Calculate overall tail risk score
            overall_tail_risk = self._calculate_overall_tail_risk_score(
                put_tail_analysis, call_tail_analysis, stress_level
            )
            
            return TailRiskAssessment(
                overall_tail_risk_score=float(overall_tail_risk),
                put_tail_risk_score=float(put_tail_analysis['risk_score']),
                crash_probability=float(put_tail_analysis['crash_probability']),
                crash_magnitude_estimate=float(put_tail_analysis['magnitude_estimate']),
                call_tail_risk_score=float(call_tail_analysis['risk_score']),
                melt_up_probability=float(call_tail_analysis['melt_up_probability']),
                melt_up_magnitude_estimate=float(call_tail_analysis['magnitude_estimate']),
                risk_concentration_areas=risk_concentration_areas,
                tail_hedging_activity=float(tail_hedging_activity),
                stress_level=float(stress_level),
                liquidity_concerns=float(liquidity_concerns)
            )
            
        except Exception as e:
            self.logger.error(f"Tail risk analysis failed: {e}")
            raise
    
    def _analyze_put_tail_risk(self, skew_data: IVSkewData,
                             surface_result: VolatilitySurfaceResult,
                             advanced_metrics: AdvancedIVMetrics) -> Dict[str, float]:
        """Analyze put tail (crash) risk"""
        
        put_tail_risk = {}
        
        # Base put tail risk from advanced metrics
        base_put_tail_risk = advanced_metrics.tail_risk_put
        
        # Put skew steepness contribution
        skew_contribution = min(1.0, surface_result.put_skew_dominance / 0.2)
        
        # Risk reversal contribution
        rr_contribution = max(0, surface_result.risk_reversal_25d) / 0.15
        
        # Far OTM put activity
        spot = skew_data.spot
        far_otm_put_strikes = skew_data.put_strikes[skew_data.put_strikes < spot * 0.85]
        
        if len(far_otm_put_strikes) > 0:
            # Analyze activity in far OTM puts
            far_otm_volumes = []
            far_otm_ois = []
            
            for strike in far_otm_put_strikes:
                idx = np.where(skew_data.strikes == strike)[0]
                if len(idx) > 0:
                    far_otm_volumes.append(skew_data.put_volumes[idx[0]])
                    far_otm_ois.append(skew_data.put_oi[idx[0]])
            
            if far_otm_volumes:
                far_otm_activity = np.sum(far_otm_volumes) / (np.sum(skew_data.put_volumes) + 1e-10)
            else:
                far_otm_activity = 0.0
        else:
            far_otm_activity = 0.0
        
        # Combine risk factors
        risk_factors = [base_put_tail_risk, skew_contribution, rr_contribution, far_otm_activity]
        weights = [0.4, 0.3, 0.2, 0.1]
        
        put_tail_risk['risk_score'] = np.average(risk_factors, weights=weights)
        put_tail_risk['crash_probability'] = advanced_metrics.crash_probability
        
        # Crash magnitude estimate (based on skew steepness)
        magnitude_estimate = min(0.3, surface_result.skew_steepness / 0.02 * 0.15)  # Up to 30% crash
        put_tail_risk['magnitude_estimate'] = magnitude_estimate
        
        return put_tail_risk
    
    def _analyze_call_tail_risk(self, skew_data: IVSkewData,
                              surface_result: VolatilitySurfaceResult,
                              advanced_metrics: AdvancedIVMetrics) -> Dict[str, float]:
        """Analyze call tail (melt-up) risk"""
        
        call_tail_risk = {}
        
        # Base call tail risk from advanced metrics
        base_call_tail_risk = advanced_metrics.tail_risk_call
        
        # Call skew strength contribution
        call_skew_contribution = min(1.0, surface_result.call_skew_strength / 0.15)
        
        # Reverse risk reversal contribution (negative RR indicates call tail risk)
        reverse_rr_contribution = max(0, -surface_result.risk_reversal_25d) / 0.1
        
        # Far OTM call activity
        spot = skew_data.spot
        far_otm_call_strikes = skew_data.call_strikes[skew_data.call_strikes > spot * 1.15]
        
        if len(far_otm_call_strikes) > 0:
            # Analyze activity in far OTM calls
            far_otm_volumes = []
            far_otm_ois = []
            
            for strike in far_otm_call_strikes:
                idx = np.where(skew_data.strikes == strike)[0]
                if len(idx) > 0:
                    far_otm_volumes.append(skew_data.call_volumes[idx[0]])
                    far_otm_ois.append(skew_data.call_oi[idx[0]])
            
            if far_otm_volumes:
                far_otm_activity = np.sum(far_otm_volumes) / (np.sum(skew_data.call_volumes) + 1e-10)
            else:
                far_otm_activity = 0.0
        else:
            far_otm_activity = 0.0
        
        # Combine risk factors
        risk_factors = [base_call_tail_risk, call_skew_contribution, reverse_rr_contribution, far_otm_activity]
        weights = [0.4, 0.3, 0.2, 0.1]
        
        call_tail_risk['risk_score'] = np.average(risk_factors, weights=weights)
        
        # Melt-up probability (generally lower than crash probability)
        call_tail_risk['melt_up_probability'] = min(0.2, call_tail_risk['risk_score'] * 0.8)
        
        # Melt-up magnitude estimate
        magnitude_estimate = min(0.25, surface_result.call_skew_strength / 0.1 * 0.12)  # Up to 25% melt-up
        call_tail_risk['magnitude_estimate'] = magnitude_estimate
        
        return call_tail_risk
    
    def _identify_risk_concentration_areas(self, skew_data: IVSkewData,
                                         surface_result: VolatilitySurfaceResult) -> List[Dict[str, Any]]:
        """Identify areas of risk concentration"""
        
        risk_areas = []
        spot = skew_data.spot
        
        # High OI concentration areas
        total_call_oi = np.sum(skew_data.call_oi)
        total_put_oi = np.sum(skew_data.put_oi)
        
        for i, strike in enumerate(skew_data.strikes):
            call_oi_ratio = skew_data.call_oi[i] / (total_call_oi + 1e-10)
            put_oi_ratio = skew_data.put_oi[i] / (total_put_oi + 1e-10)
            
            # Identify high concentration strikes
            if call_oi_ratio > 0.15 or put_oi_ratio > 0.15:
                risk_areas.append({
                    'strike': float(strike),
                    'strike_type': 'high_oi_concentration',
                    'distance_from_spot_pct': float((strike - spot) / spot * 100),
                    'call_oi_concentration': float(call_oi_ratio),
                    'put_oi_concentration': float(put_oi_ratio),
                    'risk_level': float(max(call_oi_ratio, put_oi_ratio))
                })
        
        # Round number concentrations
        round_numbers = [
            int(spot / 100) * 100,
            (int(spot / 100) + 1) * 100,
            int(spot / 50) * 50,
            (int(spot / 50) + 1) * 50
        ]
        
        for round_strike in round_numbers:
            if round_strike in skew_data.strikes:
                idx = np.where(skew_data.strikes == round_strike)[0][0]
                total_oi_at_strike = skew_data.call_oi[idx] + skew_data.put_oi[idx]
                
                if total_oi_at_strike > np.mean(skew_data.call_oi + skew_data.put_oi) * 2:
                    risk_areas.append({
                        'strike': float(round_strike),
                        'strike_type': 'round_number_concentration',
                        'distance_from_spot_pct': float((round_strike - spot) / spot * 100),
                        'total_oi': float(total_oi_at_strike),
                        'risk_level': float(min(1.0, total_oi_at_strike / (np.mean(skew_data.call_oi + skew_data.put_oi) * 5)))
                    })
        
        return risk_areas
    
    def _analyze_tail_hedging_activity(self, skew_data: IVSkewData,
                                     surface_result: VolatilitySurfaceResult) -> float:
        """Analyze tail hedging activity level"""
        
        hedging_indicators = []
        spot = skew_data.spot
        
        # Far OTM put buying (defensive hedging)
        far_otm_put_strikes = skew_data.put_strikes[skew_data.put_strikes < spot * 0.9]
        if len(far_otm_put_strikes) > 0:
            far_otm_put_volume = np.sum([skew_data.put_volumes[i] for i, strike in enumerate(skew_data.strikes) 
                                        if strike in far_otm_put_strikes])
            total_put_volume = np.sum(skew_data.put_volumes)
            
            if total_put_volume > 0:
                far_otm_put_ratio = far_otm_put_volume / total_put_volume
                hedging_indicators.append(far_otm_put_ratio)
        
        # Put skew steepness as hedging indicator
        hedging_indicators.append(min(1.0, surface_result.put_skew_dominance / 0.2))
        
        # Risk reversal as hedging indicator
        if surface_result.risk_reversal_25d > 0:
            hedging_indicators.append(min(1.0, surface_result.risk_reversal_25d / 0.15))
        
        return np.mean(hedging_indicators) if hedging_indicators else 0.3
    
    def _assess_market_stress_level(self, surface_result: VolatilitySurfaceResult,
                                  advanced_metrics: AdvancedIVMetrics) -> float:
        """Assess overall market stress level"""
        
        stress_indicators = []
        
        # IV level stress
        atm_iv = surface_result.smile_atm_iv
        if atm_iv > 0.25:
            stress_indicators.append(min(1.0, (atm_iv - 0.2) / 0.2))
        else:
            stress_indicators.append(0.0)
        
        # Skew steepness stress
        stress_indicators.append(min(1.0, surface_result.skew_steepness / 0.03))
        
        # Surface instability stress
        stress_indicators.append(1.0 - advanced_metrics.surface_stability_score)
        
        # Volatility clustering stress
        stress_indicators.append(advanced_metrics.volatility_clustering_score)
        
        # Risk reversal extremes
        rr_extreme = max(abs(surface_result.risk_reversal_25d), abs(surface_result.risk_reversal_10d))
        stress_indicators.append(min(1.0, rr_extreme / 0.15))
        
        return np.mean(stress_indicators)
    
    def _evaluate_liquidity_concerns(self, skew_data: IVSkewData,
                                   surface_result: VolatilitySurfaceResult) -> float:
        """Evaluate liquidity concerns"""
        
        liquidity_factors = []
        
        # Volume dispersion (low dispersion = poor liquidity)
        all_volumes = np.concatenate([skew_data.call_volumes, skew_data.put_volumes])
        if len(all_volumes) > 0 and np.mean(all_volumes) > 0:
            volume_cv = np.std(all_volumes) / np.mean(all_volumes)
            if volume_cv < 0.5:  # Low variation = poor liquidity
                liquidity_factors.append(1.0 - volume_cv / 0.5)
            else:
                liquidity_factors.append(0.0)
        
        # Surface quality degradation
        liquidity_factors.append(1.0 - surface_result.surface_quality_score)
        
        # High outlier count
        outlier_ratio = surface_result.outlier_count / max(len(surface_result.surface_strikes), 1)
        liquidity_factors.append(min(1.0, outlier_ratio * 3))
        
        # Data completeness issues
        liquidity_factors.append(1.0 - surface_result.data_completeness)
        
        return np.mean(liquidity_factors)
    
    def _calculate_overall_tail_risk_score(self, put_tail_analysis: Dict[str, float],
                                         call_tail_analysis: Dict[str, float],
                                         stress_level: float) -> float:
        """Calculate overall tail risk score"""
        
        # Weight put tail risk higher (crashes more common than melt-ups)
        put_weight = 0.6
        call_weight = 0.3
        stress_weight = 0.1
        
        overall_score = (
            put_weight * put_tail_analysis['risk_score'] +
            call_weight * call_tail_analysis['risk_score'] +
            stress_weight * stress_level
        )
        
        return min(1.0, overall_score)


class IVSkewRegimeClassifier:
    """
    Main IV Skew Regime Classification Engine
    
    Integrates all analysis components for comprehensive 8-regime market classification
    using complete volatility surface characteristics and institutional flow patterns.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # Initialize sub-analyzers
        self.skew_analyzer = SkewPatternAnalyzer(config)
        self.smile_analyzer = SmileAnalyzer(config)
        self.flow_detector = InstitutionalFlowDetector(config)
        self.tail_risk_analyzer = TailRiskAnalyzer(config)
        
        # Regime classification thresholds
        self.regime_thresholds = self._initialize_regime_thresholds()
        
        self.logger.info("IV Skew Regime Classifier initialized with 8-regime classification system")
    
    def _initialize_regime_thresholds(self) -> Dict[MarketRegime, Dict[str, float]]:
        """Initialize regime classification thresholds"""
        
        thresholds = {
            MarketRegime.EXTREME_FEAR: {
                'put_skew_min': 0.20,
                'crash_prob_min': 0.4,
                'tail_risk_min': 0.7,
                'institutional_flow_min': 0.6
            },
            MarketRegime.HIGH_FEAR: {
                'put_skew_min': 0.15,
                'crash_prob_min': 0.25,
                'tail_risk_min': 0.5,
                'institutional_flow_min': 0.4
            },
            MarketRegime.MODERATE_FEAR: {
                'put_skew_min': 0.08,
                'crash_prob_min': 0.15,
                'tail_risk_min': 0.3,
                'institutional_flow_min': 0.2
            },
            MarketRegime.NEUTRAL: {
                'put_skew_max': 0.08,
                'call_skew_max': 0.05,
                'tail_risk_max': 0.3,
                'flow_balance_threshold': 0.1
            },
            MarketRegime.MODERATE_GREED: {
                'call_skew_min': 0.05,
                'put_skew_max': 0.05,
                'risk_appetite_min': 0.6
            },
            MarketRegime.HIGH_GREED: {
                'call_skew_min': 0.10,
                'put_skew_max': 0.03,
                'risk_appetite_min': 0.7
            },
            MarketRegime.EXTREME_GREED: {
                'call_skew_min': 0.15,
                'put_skew_max': 0.02,
                'risk_appetite_min': 0.8
            },
            MarketRegime.TRANSITION: {
                'consistency_threshold': 0.6,
                'stability_threshold': 0.7
            }
        }
        
        return thresholds
    
    def classify_regime(self, classification_input: RegimeClassificationInput) -> RegimeClassificationResult:
        """
        Main regime classification method
        
        Args:
            classification_input: Complete input data for classification
            
        Returns:
            RegimeClassificationResult with comprehensive regime analysis
        """
        try:
            start_time = datetime.utcnow()
            processing_start = time.time()
            
            # Step 1: Analyze skew patterns
            skew_pattern_analysis = self.skew_analyzer.analyze_skew_pattern(
                classification_input.surface_result,
                classification_input.skew_data
            )
            
            # Step 2: Analyze volatility smile
            smile_analysis = self.smile_analyzer.analyze_volatility_smile(
                classification_input.surface_result,
                classification_input.skew_data,
                classification_input.advanced_metrics
            )
            
            # Step 3: Detect institutional flow
            institutional_flow = self.flow_detector.detect_institutional_flow(
                classification_input.skew_data,
                classification_input.surface_result,
                classification_input.advanced_metrics
            )
            
            # Step 4: Analyze tail risk
            tail_risk_assessment = self.tail_risk_analyzer.analyze_tail_risk(
                classification_input.skew_data,
                classification_input.surface_result,
                classification_input.advanced_metrics
            )
            
            # Step 5: Primary regime classification
            primary_regime, regime_confidence = self._classify_primary_regime(
                skew_pattern_analysis, smile_analysis, institutional_flow, tail_risk_assessment
            )
            
            # Step 6: Secondary regime analysis
            secondary_regimes = self._analyze_secondary_regimes(
                skew_pattern_analysis, smile_analysis, institutional_flow, tail_risk_assessment, primary_regime
            )
            
            # Step 7: Regime stability and transition analysis
            regime_stability, transition_probability = self._analyze_regime_stability_and_transitions(
                classification_input, primary_regime
            )
            
            # Step 8: Component agreement analysis
            component_agreement = self._calculate_component_agreement(
                skew_pattern_analysis, smile_analysis, institutional_flow
            )
            
            # Step 9: Generate trading signals and insights
            trading_signals = self._generate_trading_signals(
                primary_regime, tail_risk_assessment, institutional_flow
            )
            
            risk_warnings = self._generate_risk_warnings(
                tail_risk_assessment, primary_regime
            )
            
            opportunity_signals = self._generate_opportunity_signals(
                smile_analysis, institutional_flow, primary_regime
            )
            
            # Step 10: Calculate overall consistency
            overall_consistency = self._calculate_overall_consistency(
                component_agreement, regime_stability, classification_input
            )
            
            # Step 11: Assess data quality
            data_quality_score = self._assess_data_quality(classification_input)
            
            # Calculate processing time
            processing_time = (time.time() - processing_start) * 1000
            
            return RegimeClassificationResult(
                primary_regime=primary_regime,
                regime_confidence=float(regime_confidence),
                regime_stability=float(regime_stability),
                secondary_regimes=secondary_regimes,
                regime_transition_probability=float(transition_probability),
                skew_pattern_analysis=skew_pattern_analysis,
                smile_analysis=smile_analysis,
                institutional_flow=institutional_flow,
                tail_risk_assessment=tail_risk_assessment,
                component_agreement_score=float(component_agreement),
                overall_consistency=float(overall_consistency),
                trading_signals=trading_signals,
                risk_warnings=risk_warnings,
                opportunity_signals=opportunity_signals,
                analysis_timestamp=start_time,
                processing_time_ms=float(processing_time),
                data_quality_score=float(data_quality_score)
            )
            
        except Exception as e:
            self.logger.error(f"Regime classification failed: {e}")
            raise
    
    def _classify_primary_regime(self, skew_analysis: SkewPatternAnalysis,
                               smile_analysis: SmileAnalysisResult,
                               flow_analysis: InstitutionalFlowAnalysis,
                               tail_risk: TailRiskAssessment) -> Tuple[MarketRegime, float]:
        """Classify primary market regime"""
        
        # Calculate regime scores
        regime_scores = {}
        
        for regime in MarketRegime:
            regime_scores[regime] = self._calculate_regime_score(
                regime, skew_analysis, smile_analysis, flow_analysis, tail_risk
            )
        
        # Find highest scoring regime
        primary_regime = max(regime_scores, key=regime_scores.get)
        regime_confidence = regime_scores[primary_regime]
        
        # Ensure minimum confidence threshold
        if regime_confidence < 0.5:
            primary_regime = MarketRegime.TRANSITION
            regime_confidence = 0.5
        
        return primary_regime, regime_confidence
    
    def _calculate_regime_score(self, regime: MarketRegime,
                              skew_analysis: SkewPatternAnalysis,
                              smile_analysis: SmileAnalysisResult,
                              flow_analysis: InstitutionalFlowAnalysis,
                              tail_risk: TailRiskAssessment) -> float:
        """Calculate score for specific regime"""
        
        if regime == MarketRegime.EXTREME_FEAR:
            return self._score_extreme_fear(skew_analysis, smile_analysis, flow_analysis, tail_risk)
        elif regime == MarketRegime.HIGH_FEAR:
            return self._score_high_fear(skew_analysis, smile_analysis, flow_analysis, tail_risk)
        elif regime == MarketRegime.MODERATE_FEAR:
            return self._score_moderate_fear(skew_analysis, smile_analysis, flow_analysis, tail_risk)
        elif regime == MarketRegime.NEUTRAL:
            return self._score_neutral(skew_analysis, smile_analysis, flow_analysis, tail_risk)
        elif regime == MarketRegime.MODERATE_GREED:
            return self._score_moderate_greed(skew_analysis, smile_analysis, flow_analysis, tail_risk)
        elif regime == MarketRegime.HIGH_GREED:
            return self._score_high_greed(skew_analysis, smile_analysis, flow_analysis, tail_risk)
        elif regime == MarketRegime.EXTREME_GREED:
            return self._score_extreme_greed(skew_analysis, smile_analysis, flow_analysis, tail_risk)
        elif regime == MarketRegime.TRANSITION:
            return self._score_transition(skew_analysis, smile_analysis, flow_analysis, tail_risk)
        else:
            return 0.0
    
    def _score_extreme_fear(self, skew_analysis: SkewPatternAnalysis,
                          smile_analysis: SmileAnalysisResult,
                          flow_analysis: InstitutionalFlowAnalysis,
                          tail_risk: TailRiskAssessment) -> float:
        """Score EXTREME_FEAR regime"""
        
        score_components = []
        
        # Very high put skew steepness
        if skew_analysis.put_skew_steepness > 0.2:
            score_components.append(1.0)
        elif skew_analysis.put_skew_steepness > 0.15:
            score_components.append(0.8)
        else:
            score_components.append(0.0)
        
        # High crash probability
        if tail_risk.crash_probability > 0.4:
            score_components.append(1.0)
        elif tail_risk.crash_probability > 0.25:
            score_components.append(0.7)
        else:
            score_components.append(0.0)
        
        # Strong bearish institutional flow
        if flow_analysis.flow_direction == 'bearish' and flow_analysis.flow_magnitude > 0.6:
            score_components.append(1.0)
        elif flow_analysis.flow_direction == 'bearish':
            score_components.append(0.6)
        else:
            score_components.append(0.0)
        
        # High tail risk
        if tail_risk.overall_tail_risk_score > 0.7:
            score_components.append(1.0)
        else:
            score_components.append(tail_risk.overall_tail_risk_score / 0.7)
        
        return np.mean(score_components)
    
    def _score_high_fear(self, skew_analysis: SkewPatternAnalysis,
                       smile_analysis: SmileAnalysisResult,
                       flow_analysis: InstitutionalFlowAnalysis,
                       tail_risk: TailRiskAssessment) -> float:
        """Score HIGH_FEAR regime"""
        
        score_components = []
        
        # High put skew steepness
        if 0.1 < skew_analysis.put_skew_steepness <= 0.2:
            score_components.append(1.0)
        elif skew_analysis.put_skew_steepness > 0.2:
            score_components.append(0.7)  # Too extreme for HIGH_FEAR
        elif skew_analysis.put_skew_steepness > 0.05:
            score_components.append(0.6)
        else:
            score_components.append(0.0)
        
        # Moderate to high crash probability
        if 0.2 < tail_risk.crash_probability <= 0.4:
            score_components.append(1.0)
        elif tail_risk.crash_probability > 0.4:
            score_components.append(0.8)
        else:
            score_components.append(tail_risk.crash_probability / 0.2)
        
        # Bearish flow
        if flow_analysis.flow_direction == 'bearish':
            score_components.append(0.8)
        elif flow_analysis.flow_direction == 'mixed':
            score_components.append(0.4)
        else:
            score_components.append(0.0)
        
        return np.mean(score_components)
    
    def _score_moderate_fear(self, skew_analysis: SkewPatternAnalysis,
                           smile_analysis: SmileAnalysisResult,
                           flow_analysis: InstitutionalFlowAnalysis,
                           tail_risk: TailRiskAssessment) -> float:
        """Score MODERATE_FEAR regime"""
        
        score_components = []
        
        # Moderate put skew
        if 0.05 < skew_analysis.put_skew_steepness <= 0.1:
            score_components.append(1.0)
        elif skew_analysis.put_skew_steepness <= 0.15:
            score_components.append(0.7)
        else:
            score_components.append(0.3)
        
        # Moderate crash probability
        if 0.1 < tail_risk.crash_probability <= 0.25:
            score_components.append(1.0)
        else:
            score_components.append(max(0.0, 1.0 - abs(tail_risk.crash_probability - 0.175) / 0.175))
        
        # Mixed or slightly bearish flow
        if flow_analysis.flow_direction in ['mixed', 'bearish']:
            score_components.append(0.7)
        else:
            score_components.append(0.3)
        
        return np.mean(score_components)
    
    def _score_neutral(self, skew_analysis: SkewPatternAnalysis,
                     smile_analysis: SmileAnalysisResult,
                     flow_analysis: InstitutionalFlowAnalysis,
                     tail_risk: TailRiskAssessment) -> float:
        """Score NEUTRAL regime"""
        
        score_components = []
        
        # Low skew steepness
        put_call_diff = abs(skew_analysis.put_skew_steepness - skew_analysis.call_skew_steepness)
        if put_call_diff < 0.03:
            score_components.append(1.0)
        elif put_call_diff < 0.05:
            score_components.append(0.8)
        else:
            score_components.append(0.3)
        
        # Low crash probability
        if tail_risk.crash_probability < 0.15:
            score_components.append(1.0)
        elif tail_risk.crash_probability < 0.25:
            score_components.append(0.6)
        else:
            score_components.append(0.0)
        
        # Neutral flow
        if flow_analysis.flow_direction == 'neutral':
            score_components.append(1.0)
        elif flow_analysis.flow_direction == 'mixed':
            score_components.append(0.7)
        else:
            score_components.append(0.3)
        
        # Balanced tail risks
        tail_balance = 1.0 - abs(tail_risk.put_tail_risk_score - tail_risk.call_tail_risk_score)
        score_components.append(tail_balance)
        
        return np.mean(score_components)
    
    def _score_moderate_greed(self, skew_analysis: SkewPatternAnalysis,
                            smile_analysis: SmileAnalysisResult,
                            flow_analysis: InstitutionalFlowAnalysis,
                            tail_risk: TailRiskAssessment) -> float:
        """Score MODERATE_GREED regime"""
        
        score_components = []
        
        # Slight call bias
        if skew_analysis.call_skew_steepness > skew_analysis.put_skew_steepness:
            call_dominance = skew_analysis.call_skew_steepness - skew_analysis.put_skew_steepness
            if 0.02 < call_dominance <= 0.08:
                score_components.append(1.0)
            else:
                score_components.append(0.6)
        else:
            score_components.append(0.2)
        
        # High risk appetite
        if flow_analysis.institutional_risk_appetite > 0.6:
            score_components.append(1.0)
        else:
            score_components.append(flow_analysis.institutional_risk_appetite)
        
        # Low crash risk
        if tail_risk.crash_probability < 0.1:
            score_components.append(1.0)
        else:
            score_components.append(max(0.0, 1.0 - tail_risk.crash_probability / 0.2))
        
        return np.mean(score_components)
    
    def _score_high_greed(self, skew_analysis: SkewPatternAnalysis,
                        smile_analysis: SmileAnalysisResult,
                        flow_analysis: InstitutionalFlowAnalysis,
                        tail_risk: TailRiskAssessment) -> float:
        """Score HIGH_GREED regime"""
        
        score_components = []
        
        # Strong call bias
        call_dominance = skew_analysis.call_skew_steepness - skew_analysis.put_skew_steepness
        if call_dominance > 0.08:
            score_components.append(1.0)
        elif call_dominance > 0.05:
            score_components.append(0.8)
        else:
            score_components.append(0.0)
        
        # Very high risk appetite
        if flow_analysis.institutional_risk_appetite > 0.75:
            score_components.append(1.0)
        elif flow_analysis.institutional_risk_appetite > 0.6:
            score_components.append(0.8)
        else:
            score_components.append(0.3)
        
        # Bullish flow
        if flow_analysis.flow_direction == 'bullish':
            score_components.append(1.0)
        else:
            score_components.append(0.4)
        
        return np.mean(score_components)
    
    def _score_extreme_greed(self, skew_analysis: SkewPatternAnalysis,
                           smile_analysis: SmileAnalysisResult,
                           flow_analysis: InstitutionalFlowAnalysis,
                           tail_risk: TailRiskAssessment) -> float:
        """Score EXTREME_GREED regime"""
        
        score_components = []
        
        # Very strong call bias
        call_dominance = skew_analysis.call_skew_steepness - skew_analysis.put_skew_steepness
        if call_dominance > 0.12:
            score_components.append(1.0)
        elif call_dominance > 0.08:
            score_components.append(0.7)
        else:
            score_components.append(0.0)
        
        # Extreme risk appetite
        if flow_analysis.institutional_risk_appetite > 0.85:
            score_components.append(1.0)
        else:
            score_components.append(0.0)
        
        # Strong bullish flow
        if flow_analysis.flow_direction == 'bullish' and flow_analysis.flow_magnitude > 0.7:
            score_components.append(1.0)
        else:
            score_components.append(0.0)
        
        # Very low put skew
        if skew_analysis.put_skew_steepness < 0.02:
            score_components.append(1.0)
        else:
            score_components.append(0.3)
        
        return np.mean(score_components)
    
    def _score_transition(self, skew_analysis: SkewPatternAnalysis,
                        smile_analysis: SmileAnalysisResult,
                        flow_analysis: InstitutionalFlowAnalysis,
                        tail_risk: TailRiskAssessment) -> float:
        """Score TRANSITION regime"""
        
        transition_indicators = []
        
        # Pattern inconsistency
        if skew_analysis.pattern_confidence < 0.6:
            transition_indicators.append(1.0 - skew_analysis.pattern_confidence)
        
        # Surface instability
        if smile_analysis.smile_stability < 0.7:
            transition_indicators.append(1.0 - smile_analysis.smile_stability)
        
        # Mixed flow signals
        if flow_analysis.flow_direction == 'mixed':
            transition_indicators.append(0.8)
        
        # Low component agreement (would calculate if available)
        transition_indicators.append(0.5)  # Default
        
        return np.mean(transition_indicators) if transition_indicators else 0.3
    
    def _analyze_secondary_regimes(self, skew_analysis: SkewPatternAnalysis,
                                 smile_analysis: SmileAnalysisResult,
                                 flow_analysis: InstitutionalFlowAnalysis,
                                 tail_risk: TailRiskAssessment,
                                 primary_regime: MarketRegime) -> List[Tuple[MarketRegime, float]]:
        """Analyze secondary regime probabilities"""
        
        regime_scores = {}
        
        for regime in MarketRegime:
            if regime != primary_regime:
                regime_scores[regime] = self._calculate_regime_score(
                    regime, skew_analysis, smile_analysis, flow_analysis, tail_risk
                )
        
        # Sort by score and return top 3
        sorted_regimes = sorted(regime_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_regimes[:3]
    
    def _analyze_regime_stability_and_transitions(self, classification_input: RegimeClassificationInput,
                                                primary_regime: MarketRegime) -> Tuple[float, float]:
        """Analyze regime stability and transition probability"""
        
        stability_factors = []
        transition_factors = []
        
        # Surface stability
        stability_factors.append(classification_input.advanced_metrics.surface_stability_score)
        
        # Smile stability
        # (Would need to calculate from smile analysis)
        stability_factors.append(0.8)  # Mock value
        
        # Pattern consistency
        # (Would use from skew analysis)
        stability_factors.append(0.7)  # Mock value
        
        # Transition factors
        if classification_input.term_structure_result:
            transition_factors.append(classification_input.term_structure_result.regime_transition_probability)
        
        # Add other transition indicators
        transition_factors.append(1.0 - np.mean(stability_factors))
        
        regime_stability = np.mean(stability_factors)
        transition_probability = np.mean(transition_factors) if transition_factors else 0.2
        
        return regime_stability, transition_probability
    
    def _calculate_component_agreement(self, skew_analysis: SkewPatternAnalysis,
                                     smile_analysis: SmileAnalysisResult,
                                     flow_analysis: InstitutionalFlowAnalysis) -> float:
        """Calculate agreement between analysis components"""
        
        agreement_factors = []
        
        # Pattern-flow agreement
        if skew_analysis.institutional_signature > 0.5 and flow_analysis.flow_detection_confidence > 0.5:
            agreement_factors.append(0.9)
        else:
            agreement_factors.append(0.5)
        
        # Smile-skew agreement
        agreement_factors.append(skew_analysis.pattern_confidence * smile_analysis.smile_quality)
        
        # Surface consistency
        agreement_factors.append(smile_analysis.surface_consistency)
        
        return np.mean(agreement_factors)
    
    def _generate_trading_signals(self, primary_regime: MarketRegime,
                                tail_risk: TailRiskAssessment,
                                flow_analysis: InstitutionalFlowAnalysis) -> List[Dict[str, Any]]:
        """Generate actionable trading signals"""
        
        signals = []
        
        # Regime-specific signals
        if primary_regime in [MarketRegime.HIGH_FEAR, MarketRegime.EXTREME_FEAR]:
            signals.append({
                'signal_type': 'protective_puts',
                'strength': 'strong' if primary_regime == MarketRegime.EXTREME_FEAR else 'moderate',
                'rationale': f'High crash probability ({tail_risk.crash_probability:.1%})',
                'action': 'buy_protective_puts'
            })
        
        elif primary_regime in [MarketRegime.HIGH_GREED, MarketRegime.EXTREME_GREED]:
            signals.append({
                'signal_type': 'call_overwriting',
                'strength': 'strong' if primary_regime == MarketRegime.EXTREME_GREED else 'moderate',
                'rationale': 'High risk appetite, elevated call premiums',
                'action': 'sell_covered_calls'
            })
        
        # Flow-based signals
        if flow_analysis.flow_detection_confidence > 0.7:
            signals.append({
                'signal_type': 'institutional_follow',
                'strength': 'moderate',
                'rationale': f'{flow_analysis.flow_direction.title()} institutional flow detected',
                'action': f'follow_{flow_analysis.flow_direction}_flow'
            })
        
        # Tail risk signals
        if tail_risk.overall_tail_risk_score > 0.7:
            signals.append({
                'signal_type': 'tail_hedge',
                'strength': 'strong',
                'rationale': 'Elevated tail risk across multiple measures',
                'action': 'increase_tail_hedging'
            })
        
        return signals
    
    def _generate_risk_warnings(self, tail_risk: TailRiskAssessment,
                              primary_regime: MarketRegime) -> List[str]:
        """Generate risk warnings"""
        
        warnings = []
        
        if tail_risk.crash_probability > 0.3:
            warnings.append(f"HIGH CRASH RISK: {tail_risk.crash_probability:.1%} probability")
        
        if tail_risk.stress_level > 0.7:
            warnings.append("MARKET STRESS: High stress indicators detected")
        
        if tail_risk.liquidity_concerns > 0.6:
            warnings.append("LIQUIDITY CONCERNS: Reduced market liquidity")
        
        if primary_regime == MarketRegime.EXTREME_FEAR:
            warnings.append("EXTREME FEAR REGIME: Consider defensive positioning")
        elif primary_regime == MarketRegime.EXTREME_GREED:
            warnings.append("EXTREME GREED REGIME: Elevated crash risk from euphoria")
        
        if len(tail_risk.risk_concentration_areas) > 3:
            warnings.append("CONCENTRATION RISK: Multiple high-risk strike areas")
        
        return warnings
    
    def _generate_opportunity_signals(self, smile_analysis: SmileAnalysisResult,
                                    flow_analysis: InstitutionalFlowAnalysis,
                                    primary_regime: MarketRegime) -> List[str]:
        """Generate opportunity signals"""
        
        opportunities = []
        
        # Arbitrage opportunities
        if len(smile_analysis.arbitrage_opportunities) > 0:
            for arb in smile_analysis.arbitrage_opportunities:
                if arb['confidence'] > 0.7:
                    opportunities.append(f"ARBITRAGE: {arb['type']} opportunity detected")
        
        # Regime-specific opportunities
        if primary_regime == MarketRegime.EXTREME_FEAR:
            opportunities.append("OPPORTUNITY: Oversold conditions, consider contrarian plays")
        elif primary_regime == MarketRegime.EXTREME_GREED:
            opportunities.append("OPPORTUNITY: Overextended conditions, consider short strategies")
        
        # Flow-based opportunities
        if flow_analysis.hedging_vs_speculation == 'hedging' and flow_analysis.flow_magnitude > 0.6:
            opportunities.append("OPPORTUNITY: Heavy hedging activity, consider volatility strategies")
        
        # Surface inefficiencies
        if smile_analysis.surface_consistency < 0.6:
            opportunities.append("OPPORTUNITY: Surface inconsistencies, consider relative value trades")
        
        return opportunities
    
    def _calculate_overall_consistency(self, component_agreement: float,
                                     regime_stability: float,
                                     classification_input: RegimeClassificationInput) -> float:
        """Calculate overall analysis consistency"""
        
        consistency_factors = []
        
        # Component agreement
        consistency_factors.append(component_agreement)
        
        # Regime stability
        consistency_factors.append(regime_stability)
        
        # Surface quality
        consistency_factors.append(classification_input.surface_result.surface_quality_score)
        
        # Data completeness
        consistency_factors.append(classification_input.surface_result.data_completeness)
        
        return np.mean(consistency_factors)
    
    def _assess_data_quality(self, classification_input: RegimeClassificationInput) -> float:
        """Assess overall data quality for classification"""
        
        quality_factors = []
        
        # Surface data quality
        quality_factors.append(classification_input.surface_result.surface_quality_score)
        quality_factors.append(classification_input.surface_result.data_completeness)
        quality_factors.append(classification_input.surface_result.interpolation_quality)
        
        # Raw data quality indicators
        strike_count_score = min(1.0, classification_input.skew_data.strike_count / 50)
        quality_factors.append(strike_count_score)
        
        # Advanced metrics quality
        quality_factors.append(classification_input.advanced_metrics.surface_stability_score)
        
        return np.mean(quality_factors)