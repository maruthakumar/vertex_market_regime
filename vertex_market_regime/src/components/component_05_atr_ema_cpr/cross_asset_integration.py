"""
Component 5: Cross-Asset Integration and Validation System

Advanced cross-asset analysis system that validates and integrates insights between
straddle price analysis and underlying price analysis, providing enhanced accuracy
through multi-asset confirmation and dynamic weighting adjustments.

Features:
- Trend direction cross-validation between straddle and underlying analysis
- Volatility regime cross-validation across both asset types  
- Support/resistance level validation using CPR analysis
- Confidence scoring with cross-asset validation boosts/penalties
- Dynamic weighting system (60% straddle, 40% underlying) with adaptive adjustments
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import logging
import asyncio
import time
import warnings
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .dual_asset_data_extractor import StraddlePriceData, UnderlyingPriceData
from .straddle_atr_ema_cpr_engine import StraddleAnalysisResult
from .underlying_atr_ema_cpr_engine import UnderlyingAnalysisResult
from .dual_dte_framework import DTEIntegratedResult

warnings.filterwarnings('ignore')


@dataclass
class TrendDirectionValidation:
    """Trend direction validation between assets"""
    agreement_score: np.ndarray
    disagreement_patterns: Dict[str, np.ndarray]
    trend_strength_correlation: float
    directional_consistency: float
    validation_confidence: np.ndarray
    conflicting_signals: np.ndarray
    metadata: Dict[str, Any]


@dataclass
class VolatilityRegimeValidation:
    """Volatility regime validation between assets"""
    regime_agreement_score: np.ndarray
    cross_asset_volatility_correlation: float
    regime_transition_alignment: float
    straddle_vs_underlying_regimes: Dict[str, np.ndarray]
    regime_validation_confidence: np.ndarray
    volatility_divergence_signals: np.ndarray
    metadata: Dict[str, Any]


@dataclass
class SupportResistanceValidation:
    """Support/resistance level validation between assets"""
    level_agreement_score: np.ndarray
    cross_asset_level_correlation: float
    breakout_confirmation_rate: float
    level_strength_validation: Dict[str, np.ndarray]
    confluence_zones: np.ndarray
    validated_levels: Dict[str, np.ndarray]
    metadata: Dict[str, Any]


@dataclass
class CrossAssetConfidenceResult:
    """Cross-asset confidence scoring result"""
    base_confidence: np.ndarray
    validation_boost: np.ndarray
    conflict_penalty: np.ndarray
    final_confidence: np.ndarray
    confidence_breakdown: Dict[str, np.ndarray]
    high_confidence_zones: np.ndarray
    low_confidence_warnings: np.ndarray
    metadata: Dict[str, Any]


@dataclass
class DynamicWeightingResult:
    """Dynamic weighting result with adaptive adjustments"""
    base_weights: Dict[str, float]
    adaptive_adjustments: Dict[str, np.ndarray]
    final_weights: Dict[str, np.ndarray]
    weight_change_reasons: Dict[str, List[str]]
    performance_based_adjustments: Dict[str, float]
    cross_validation_impact: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass
class CrossAssetIntegrationResult:
    """Complete cross-asset integration result"""
    trend_validation: TrendDirectionValidation
    volatility_validation: VolatilityRegimeValidation
    level_validation: SupportResistanceValidation
    confidence_result: CrossAssetConfidenceResult
    weighting_result: DynamicWeightingResult
    integrated_signals: Dict[str, np.ndarray]
    cross_asset_regime_classification: np.ndarray
    validation_summary: Dict[str, float]
    processing_time_ms: float
    metadata: Dict[str, Any]


class TrendDirectionValidator:
    """Validates trend direction between straddle and underlying analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # Validation thresholds
        self.agreement_threshold = config.get('trend_agreement_threshold', 0.7)
        self.strength_correlation_threshold = config.get('strength_correlation_threshold', 0.5)

    async def validate_trend_directions(self, straddle_result: StraddleAnalysisResult,
                                      underlying_result: UnderlyingAnalysisResult) -> TrendDirectionValidation:
        """Validate trend directions between straddle and underlying analysis"""
        
        # Extract trend directions
        straddle_trend = straddle_result.ema_result.trend_direction
        underlying_trend = underlying_result.ema_result.trend_directions.get('daily', np.array([]))
        
        # Ensure same length for comparison
        min_length = min(len(straddle_trend), len(underlying_trend))
        if min_length == 0:
            return self._create_empty_trend_validation()
        
        straddle_trend = straddle_trend[:min_length]
        underlying_trend = underlying_trend[:min_length]
        
        # Calculate agreement scores
        agreement_score = self._calculate_trend_agreement(straddle_trend, underlying_trend)
        
        # Identify disagreement patterns
        disagreement_patterns = self._identify_disagreement_patterns(straddle_trend, underlying_trend)
        
        # Calculate trend strength correlation
        straddle_strength = straddle_result.ema_result.trend_strength[:min_length]
        underlying_strength = underlying_result.ema_result.trend_strengths.get('daily', np.array([]))[:min_length]
        
        strength_correlation = self._calculate_strength_correlation(straddle_strength, underlying_strength)
        
        # Calculate directional consistency
        directional_consistency = self._calculate_directional_consistency(straddle_trend, underlying_trend)
        
        # Calculate validation confidence
        validation_confidence = self._calculate_trend_validation_confidence(
            agreement_score, strength_correlation, straddle_strength, underlying_strength
        )
        
        # Identify conflicting signals
        conflicting_signals = self._identify_conflicting_signals(
            straddle_trend, underlying_trend, straddle_strength, underlying_strength
        )
        
        return TrendDirectionValidation(
            agreement_score=agreement_score,
            disagreement_patterns=disagreement_patterns,
            trend_strength_correlation=strength_correlation,
            directional_consistency=directional_consistency,
            validation_confidence=validation_confidence,
            conflicting_signals=conflicting_signals,
            metadata={
                'comparison_length': min_length,
                'agreement_threshold': self.agreement_threshold,
                'validation_method': 'directional_comparison'
            }
        )

    def _calculate_trend_agreement(self, straddle_trend: np.ndarray, underlying_trend: np.ndarray) -> np.ndarray:
        """Calculate trend agreement scores"""
        
        agreement = np.full(len(straddle_trend), 0.0)
        
        for i in range(len(straddle_trend)):
            if straddle_trend[i] == underlying_trend[i]:
                if straddle_trend[i] == 0:
                    agreement[i] = 0.7  # Neutral agreement (moderate)
                else:
                    agreement[i] = 1.0  # Directional agreement (strong)
            elif (straddle_trend[i] == 0 and underlying_trend[i] != 0) or (straddle_trend[i] != 0 and underlying_trend[i] == 0):
                agreement[i] = 0.4  # Partial agreement (one neutral, one directional)
            else:
                agreement[i] = 0.0  # Disagreement (opposite directions)
        
        return agreement

    def _identify_disagreement_patterns(self, straddle_trend: np.ndarray, underlying_trend: np.ndarray) -> Dict[str, np.ndarray]:
        """Identify patterns in trend disagreements"""
        
        patterns = {}
        
        # Bull-Bear disagreement (straddle bullish, underlying bearish)
        patterns['bull_bear_conflict'] = np.logical_and(straddle_trend == 1, underlying_trend == -1).astype(int)
        
        # Bear-Bull disagreement (straddle bearish, underlying bullish)
        patterns['bear_bull_conflict'] = np.logical_and(straddle_trend == -1, underlying_trend == 1).astype(int)
        
        # Neutral-Directional disagreement
        patterns['neutral_directional_conflict'] = np.logical_or(
            np.logical_and(straddle_trend == 0, np.abs(underlying_trend) == 1),
            np.logical_and(np.abs(straddle_trend) == 1, underlying_trend == 0)
        ).astype(int)
        
        # Persistent disagreement (multiple consecutive disagreements)
        disagreement = (straddle_trend != underlying_trend).astype(int)
        patterns['persistent_disagreement'] = self._detect_persistent_patterns(disagreement, min_length=3)
        
        return patterns

    def _detect_persistent_patterns(self, signal: np.ndarray, min_length: int = 3) -> np.ndarray:
        """Detect persistent patterns in signal"""
        
        persistent = np.zeros_like(signal)
        
        count = 0
        for i in range(len(signal)):
            if signal[i] == 1:
                count += 1
                if count >= min_length:
                    persistent[i-count+1:i+1] = 1
            else:
                count = 0
        
        return persistent

    def _calculate_strength_correlation(self, straddle_strength: np.ndarray, underlying_strength: np.ndarray) -> float:
        """Calculate correlation between trend strengths"""
        
        # Remove NaN values
        valid_mask = ~(np.isnan(straddle_strength) | np.isnan(underlying_strength))
        
        if np.sum(valid_mask) < 10:  # Need at least 10 points for meaningful correlation
            return 0.0
        
        straddle_clean = straddle_strength[valid_mask]
        underlying_clean = underlying_strength[valid_mask]
        
        try:
            correlation = np.corrcoef(straddle_clean, underlying_clean)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0

    def _calculate_directional_consistency(self, straddle_trend: np.ndarray, underlying_trend: np.ndarray) -> float:
        """Calculate overall directional consistency"""
        
        agreement = self._calculate_trend_agreement(straddle_trend, underlying_trend)
        return np.mean(agreement)

    def _calculate_trend_validation_confidence(self, agreement_score: np.ndarray, 
                                             strength_correlation: float,
                                             straddle_strength: np.ndarray,
                                             underlying_strength: np.ndarray) -> np.ndarray:
        """Calculate confidence in trend validation"""
        
        base_confidence = agreement_score.copy()
        
        # Boost confidence where both assets show strong trends
        for i in range(len(base_confidence)):
            if not np.isnan(straddle_strength[i]) and not np.isnan(underlying_strength[i]):
                combined_strength = (straddle_strength[i] + underlying_strength[i]) / 2
                strength_boost = min(combined_strength * 0.3, 0.3)  # Max 30% boost
                base_confidence[i] = min(base_confidence[i] + strength_boost, 1.0)
        
        # Apply correlation adjustment
        correlation_adjustment = max(-0.2, min(0.2, strength_correlation * 0.2))
        base_confidence = np.clip(base_confidence + correlation_adjustment, 0.0, 1.0)
        
        return base_confidence

    def _identify_conflicting_signals(self, straddle_trend: np.ndarray, underlying_trend: np.ndarray,
                                    straddle_strength: np.ndarray, underlying_strength: np.ndarray) -> np.ndarray:
        """Identify high-conviction conflicting signals"""
        
        conflicts = np.zeros(len(straddle_trend))
        
        for i in range(len(straddle_trend)):
            # Strong disagreement with high conviction on both sides
            if (straddle_trend[i] != underlying_trend[i] and 
                straddle_trend[i] != 0 and underlying_trend[i] != 0):
                
                # Check if both have high strength (high conviction)
                if (not np.isnan(straddle_strength[i]) and not np.isnan(underlying_strength[i]) and
                    straddle_strength[i] > 0.6 and underlying_strength[i] > 0.6):
                    conflicts[i] = 1
        
        return conflicts

    def _create_empty_trend_validation(self) -> TrendDirectionValidation:
        """Create empty trend validation result"""
        
        return TrendDirectionValidation(
            agreement_score=np.array([]),
            disagreement_patterns={},
            trend_strength_correlation=0.0,
            directional_consistency=0.0,
            validation_confidence=np.array([]),
            conflicting_signals=np.array([]),
            metadata={'validation_method': 'empty_fallback'}
        )


class VolatilityRegimeValidator:
    """Validates volatility regimes between straddle and underlying analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # Validation parameters
        self.regime_tolerance = config.get('regime_tolerance', 1)  # Allow ±1 regime difference

    async def validate_volatility_regimes(self, straddle_result: StraddleAnalysisResult,
                                        underlying_result: UnderlyingAnalysisResult) -> VolatilityRegimeValidation:
        """Validate volatility regimes between assets"""
        
        # Extract volatility regimes
        straddle_regimes = straddle_result.atr_result.volatility_regime
        underlying_regimes = underlying_result.atr_result.volatility_regimes.get('daily', np.array([]))
        
        # Ensure same length
        min_length = min(len(straddle_regimes), len(underlying_regimes))
        if min_length == 0:
            return self._create_empty_volatility_validation()
        
        straddle_regimes = straddle_regimes[:min_length]
        underlying_regimes = underlying_regimes[:min_length]
        
        # Calculate regime agreement
        regime_agreement_score = self._calculate_regime_agreement(straddle_regimes, underlying_regimes)
        
        # Calculate cross-asset volatility correlation
        straddle_atr = straddle_result.atr_result.atr_14[:min_length]
        underlying_atr = underlying_result.atr_result.daily_atr.get('atr_14', np.array([]))[:min_length]
        
        volatility_correlation = self._calculate_volatility_correlation(straddle_atr, underlying_atr)
        
        # Analyze regime transition alignment
        transition_alignment = self._analyze_regime_transition_alignment(straddle_regimes, underlying_regimes)
        
        # Create regime comparison data
        regime_comparison = self._create_regime_comparison(straddle_regimes, underlying_regimes)
        
        # Calculate validation confidence
        validation_confidence = self._calculate_regime_validation_confidence(
            regime_agreement_score, volatility_correlation, straddle_atr, underlying_atr
        )
        
        # Identify volatility divergence signals
        divergence_signals = self._identify_volatility_divergence(
            straddle_regimes, underlying_regimes, straddle_atr, underlying_atr
        )
        
        return VolatilityRegimeValidation(
            regime_agreement_score=regime_agreement_score,
            cross_asset_volatility_correlation=volatility_correlation,
            regime_transition_alignment=transition_alignment,
            straddle_vs_underlying_regimes=regime_comparison,
            regime_validation_confidence=validation_confidence,
            volatility_divergence_signals=divergence_signals,
            metadata={
                'comparison_length': min_length,
                'regime_tolerance': self.regime_tolerance,
                'validation_method': 'regime_comparison'
            }
        )

    def _calculate_regime_agreement(self, straddle_regimes: np.ndarray, underlying_regimes: np.ndarray) -> np.ndarray:
        """Calculate regime agreement with tolerance"""
        
        agreement = np.full(len(straddle_regimes), 0.0)
        
        for i in range(len(straddle_regimes)):
            regime_diff = abs(straddle_regimes[i] - underlying_regimes[i])
            
            if regime_diff == 0:
                agreement[i] = 1.0  # Perfect agreement
            elif regime_diff <= self.regime_tolerance:
                agreement[i] = 0.7  # Close agreement within tolerance
            elif regime_diff <= self.regime_tolerance * 2:
                agreement[i] = 0.4  # Moderate disagreement
            else:
                agreement[i] = 0.0  # Strong disagreement
        
        return agreement

    def _calculate_volatility_correlation(self, straddle_atr: np.ndarray, underlying_atr: np.ndarray) -> float:
        """Calculate correlation between ATR values"""
        
        # Remove NaN values
        valid_mask = ~(np.isnan(straddle_atr) | np.isnan(underlying_atr))
        
        if np.sum(valid_mask) < 10:
            return 0.0
        
        straddle_clean = straddle_atr[valid_mask]
        underlying_clean = underlying_atr[valid_mask]
        
        try:
            correlation = np.corrcoef(straddle_clean, underlying_clean)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0

    def _analyze_regime_transition_alignment(self, straddle_regimes: np.ndarray, underlying_regimes: np.ndarray) -> float:
        """Analyze how well regime transitions align between assets"""
        
        if len(straddle_regimes) < 2:
            return 0.5
        
        # Detect transitions
        straddle_transitions = np.diff(straddle_regimes) != 0
        underlying_transitions = np.diff(underlying_regimes) != 0
        
        if len(straddle_transitions) == 0:
            return 0.5
        
        # Calculate alignment of transitions
        aligned_transitions = np.logical_and(straddle_transitions, underlying_transitions)
        total_transitions = np.logical_or(straddle_transitions, underlying_transitions)
        
        if np.sum(total_transitions) == 0:
            return 1.0  # No transitions in either - perfect alignment
        
        alignment_rate = np.sum(aligned_transitions) / np.sum(total_transitions)
        return alignment_rate

    def _create_regime_comparison(self, straddle_regimes: np.ndarray, underlying_regimes: np.ndarray) -> Dict[str, np.ndarray]:
        """Create regime comparison data"""
        
        return {
            'straddle_regimes': straddle_regimes.copy(),
            'underlying_regimes': underlying_regimes.copy(),
            'regime_difference': straddle_regimes - underlying_regimes,
            'regime_agreement': self._calculate_regime_agreement(straddle_regimes, underlying_regimes)
        }

    def _calculate_regime_validation_confidence(self, regime_agreement: np.ndarray,
                                              volatility_correlation: float,
                                              straddle_atr: np.ndarray,
                                              underlying_atr: np.ndarray) -> np.ndarray:
        """Calculate confidence in regime validation"""
        
        base_confidence = regime_agreement.copy()
        
        # Boost confidence based on volatility correlation
        correlation_boost = max(0, volatility_correlation) * 0.2
        base_confidence = np.clip(base_confidence + correlation_boost, 0.0, 1.0)
        
        # Adjust based on ATR values (higher ATR = more confident in volatility assessment)
        for i in range(len(base_confidence)):
            if not np.isnan(straddle_atr[i]) and not np.isnan(underlying_atr[i]):
                avg_atr_percentile = min((straddle_atr[i] + underlying_atr[i]) / 2, 1.0)
                atr_boost = avg_atr_percentile * 0.15
                base_confidence[i] = min(base_confidence[i] + atr_boost, 1.0)
        
        return base_confidence

    def _identify_volatility_divergence(self, straddle_regimes: np.ndarray, underlying_regimes: np.ndarray,
                                      straddle_atr: np.ndarray, underlying_atr: np.ndarray) -> np.ndarray:
        """Identify significant volatility divergence between assets"""
        
        divergence = np.zeros(len(straddle_regimes))
        
        for i in range(len(straddle_regimes)):
            # Large regime difference
            regime_diff = abs(straddle_regimes[i] - underlying_regimes[i])
            
            # ATR divergence
            atr_divergence = 0
            if not np.isnan(straddle_atr[i]) and not np.isnan(underlying_atr[i]):
                atr_ratio = max(straddle_atr[i], underlying_atr[i]) / max(min(straddle_atr[i], underlying_atr[i]), 1e-10)
                if atr_ratio > 2.0:  # One ATR is more than 2x the other
                    atr_divergence = 1
            
            # Combined divergence signal
            if regime_diff >= 2 or atr_divergence:
                divergence[i] = 1
        
        return divergence

    def _create_empty_volatility_validation(self) -> VolatilityRegimeValidation:
        """Create empty volatility validation result"""
        
        return VolatilityRegimeValidation(
            regime_agreement_score=np.array([]),
            cross_asset_volatility_correlation=0.0,
            regime_transition_alignment=0.0,
            straddle_vs_underlying_regimes={},
            regime_validation_confidence=np.array([]),
            volatility_divergence_signals=np.array([]),
            metadata={'validation_method': 'empty_fallback'}
        )


class SupportResistanceLevelValidator:
    """Validates support/resistance levels between straddle and underlying CPR analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # Level validation parameters
        self.level_tolerance_pct = config.get('level_tolerance_pct', 2.0)  # 2% tolerance

    async def validate_support_resistance_levels(self, straddle_result: StraddleAnalysisResult,
                                               underlying_result: UnderlyingAnalysisResult) -> SupportResistanceValidation:
        """Validate support/resistance levels between assets"""
        
        # Extract CPR levels
        straddle_pivot = straddle_result.cpr_result.pivot_points.get('standard', np.array([]))
        underlying_pivot = underlying_result.cpr_result.daily_cpr.get('standard', {}).get('pivot', np.array([]))
        
        # Ensure same length
        min_length = min(len(straddle_pivot), len(underlying_pivot))
        if min_length == 0:
            return self._create_empty_level_validation()
        
        straddle_pivot = straddle_pivot[:min_length]
        underlying_pivot = underlying_pivot[:min_length]
        
        # Calculate level agreement
        level_agreement_score = self._calculate_level_agreement(straddle_pivot, underlying_pivot)
        
        # Calculate level correlation
        level_correlation = self._calculate_level_correlation(straddle_pivot, underlying_pivot)
        
        # Analyze breakout confirmations
        breakout_confirmation_rate = self._analyze_breakout_confirmations(straddle_result, underlying_result, min_length)
        
        # Validate level strengths
        level_strength_validation = self._validate_level_strengths(straddle_result, underlying_result, min_length)
        
        # Identify confluence zones
        confluence_zones = self._identify_confluence_zones(straddle_pivot, underlying_pivot)
        
        # Create validated levels
        validated_levels = self._create_validated_levels(straddle_result, underlying_result, min_length)
        
        return SupportResistanceValidation(
            level_agreement_score=level_agreement_score,
            cross_asset_level_correlation=level_correlation,
            breakout_confirmation_rate=breakout_confirmation_rate,
            level_strength_validation=level_strength_validation,
            confluence_zones=confluence_zones,
            validated_levels=validated_levels,
            metadata={
                'comparison_length': min_length,
                'tolerance_pct': self.level_tolerance_pct,
                'validation_method': 'level_comparison'
            }
        )

    def _calculate_level_agreement(self, straddle_pivot: np.ndarray, underlying_pivot: np.ndarray) -> np.ndarray:
        """Calculate agreement between pivot levels"""
        
        agreement = np.full(len(straddle_pivot), 0.0)
        
        for i in range(len(straddle_pivot)):
            if not np.isnan(straddle_pivot[i]) and not np.isnan(underlying_pivot[i]):
                # Calculate percentage difference
                pct_diff = abs(straddle_pivot[i] - underlying_pivot[i]) / max(underlying_pivot[i], 1e-10) * 100
                
                if pct_diff <= self.level_tolerance_pct:
                    agreement[i] = 1.0  # Strong agreement within tolerance
                elif pct_diff <= self.level_tolerance_pct * 2:
                    agreement[i] = 0.6  # Moderate agreement
                elif pct_diff <= self.level_tolerance_pct * 3:
                    agreement[i] = 0.3  # Weak agreement
                else:
                    agreement[i] = 0.0  # Poor agreement
        
        return agreement

    def _calculate_level_correlation(self, straddle_pivot: np.ndarray, underlying_pivot: np.ndarray) -> float:
        """Calculate correlation between pivot levels"""
        
        # Remove NaN values
        valid_mask = ~(np.isnan(straddle_pivot) | np.isnan(underlying_pivot))
        
        if np.sum(valid_mask) < 10:
            return 0.0
        
        straddle_clean = straddle_pivot[valid_mask]
        underlying_clean = underlying_pivot[valid_mask]
        
        try:
            correlation = np.corrcoef(straddle_clean, underlying_clean)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0

    def _analyze_breakout_confirmations(self, straddle_result: StraddleAnalysisResult,
                                      underlying_result: UnderlyingAnalysisResult,
                                      min_length: int) -> float:
        """Analyze breakout confirmation rate between assets"""
        
        # Extract breakout signals
        straddle_resistance_breakout = straddle_result.cpr_result.breakout_signals.get('resistance_breakout', np.array([]))
        underlying_resistance_breakout = underlying_result.cpr_result.breakout_signals.get('daily', {}).get('resistance_breakout', np.array([]))
        
        if len(straddle_resistance_breakout) == 0 or len(underlying_resistance_breakout) == 0:
            return 0.5  # Default moderate confirmation
        
        # Truncate to min_length
        straddle_breakout = straddle_resistance_breakout[:min_length]
        underlying_breakout = underlying_resistance_breakout[:min_length]
        
        # Calculate confirmation rate
        confirmed_breakouts = np.logical_and(straddle_breakout, underlying_breakout)
        total_breakouts = np.logical_or(straddle_breakout, underlying_breakout)
        
        if np.sum(total_breakouts) == 0:
            return 1.0  # No breakouts - perfect confirmation (vacuously true)
        
        confirmation_rate = np.sum(confirmed_breakouts) / np.sum(total_breakouts)
        return confirmation_rate

    def _validate_level_strengths(self, straddle_result: StraddleAnalysisResult,
                                underlying_result: UnderlyingAnalysisResult,
                                min_length: int) -> Dict[str, np.ndarray]:
        """Validate level strengths between assets"""
        
        validation = {}
        
        # Get level strengths
        straddle_strength = straddle_result.cpr_result.level_strength.get('standard', np.array([]))
        underlying_strength = underlying_result.cpr_result.support_resistance_validation.get('support_strength', np.array([]))
        
        if len(straddle_strength) > 0 and len(underlying_strength) > 0:
            straddle_strength = straddle_strength[:min_length]
            underlying_strength = underlying_strength[:min_length]
            
            # Calculate strength agreement
            strength_agreement = np.zeros(min_length)
            for i in range(min_length):
                if not np.isnan(straddle_strength[i]) and not np.isnan(underlying_strength[i]):
                    strength_diff = abs(straddle_strength[i] - underlying_strength[i])
                    strength_agreement[i] = max(0, 1 - strength_diff * 2)  # Normalize to 0-1
            
            validation['strength_agreement'] = strength_agreement
            validation['straddle_strength'] = straddle_strength
            validation['underlying_strength'] = underlying_strength
        
        return validation

    def _identify_confluence_zones(self, straddle_pivot: np.ndarray, underlying_pivot: np.ndarray) -> np.ndarray:
        """Identify zones where levels from both assets converge"""
        
        confluence = np.zeros(len(straddle_pivot))
        
        for i in range(len(straddle_pivot)):
            if not np.isnan(straddle_pivot[i]) and not np.isnan(underlying_pivot[i]):
                pct_diff = abs(straddle_pivot[i] - underlying_pivot[i]) / max(underlying_pivot[i], 1e-10) * 100
                
                if pct_diff <= self.level_tolerance_pct * 0.5:  # Very close levels
                    confluence[i] = 1.0
                elif pct_diff <= self.level_tolerance_pct:
                    confluence[i] = 0.7
                elif pct_diff <= self.level_tolerance_pct * 1.5:
                    confluence[i] = 0.4
        
        return confluence

    def _create_validated_levels(self, straddle_result: StraddleAnalysisResult,
                               underlying_result: UnderlyingAnalysisResult,
                               min_length: int) -> Dict[str, np.ndarray]:
        """Create validated support/resistance levels"""
        
        validated = {}
        
        # Validated pivot levels (where both assets agree)
        straddle_pivot = straddle_result.cpr_result.pivot_points.get('standard', np.array([]))[:min_length]
        underlying_pivot = underlying_result.cpr_result.daily_cpr.get('standard', {}).get('pivot', np.array([]))[:min_length]
        
        if len(straddle_pivot) > 0 and len(underlying_pivot) > 0:
            agreement = self._calculate_level_agreement(straddle_pivot, underlying_pivot)
            
            # Only include levels where agreement > 0.6
            validated_mask = agreement > 0.6
            
            validated['validated_pivots'] = np.where(validated_mask, 
                                                   (straddle_pivot + underlying_pivot) / 2,  # Average where validated
                                                   np.nan)  # NaN where not validated
            
            validated['validation_strength'] = agreement
        
        return validated

    def _create_empty_level_validation(self) -> SupportResistanceValidation:
        """Create empty level validation result"""
        
        return SupportResistanceValidation(
            level_agreement_score=np.array([]),
            cross_asset_level_correlation=0.0,
            breakout_confirmation_rate=0.5,
            level_strength_validation={},
            confluence_zones=np.array([]),
            validated_levels={},
            metadata={'validation_method': 'empty_fallback'}
        )


class CrossAssetConfidenceEngine:
    """Engine for calculating cross-asset confidence with validation boosts and penalties"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # Confidence parameters
        self.max_validation_boost = config.get('max_validation_boost', 0.3)
        self.max_conflict_penalty = config.get('max_conflict_penalty', 0.4)

    async def calculate_cross_asset_confidence(self, straddle_result: StraddleAnalysisResult,
                                             underlying_result: UnderlyingAnalysisResult,
                                             trend_validation: TrendDirectionValidation,
                                             volatility_validation: VolatilityRegimeValidation,
                                             level_validation: SupportResistanceValidation) -> CrossAssetConfidenceResult:
        """Calculate comprehensive cross-asset confidence scores"""
        
        # Get base confidence from individual analyses
        base_confidence = self._get_base_confidence(straddle_result, underlying_result)
        
        # Calculate validation boost
        validation_boost = self._calculate_validation_boost(
            trend_validation, volatility_validation, level_validation, len(base_confidence)
        )
        
        # Calculate conflict penalty
        conflict_penalty = self._calculate_conflict_penalty(
            trend_validation, volatility_validation, level_validation, len(base_confidence)
        )
        
        # Calculate final confidence
        final_confidence = np.clip(base_confidence + validation_boost - conflict_penalty, 0.1, 0.95)
        
        # Create confidence breakdown
        confidence_breakdown = self._create_confidence_breakdown(
            base_confidence, validation_boost, conflict_penalty,
            trend_validation, volatility_validation, level_validation
        )
        
        # Identify high/low confidence zones
        high_confidence_zones = (final_confidence > 0.8).astype(int)
        low_confidence_warnings = (final_confidence < 0.3).astype(int)
        
        return CrossAssetConfidenceResult(
            base_confidence=base_confidence,
            validation_boost=validation_boost,
            conflict_penalty=conflict_penalty,
            final_confidence=final_confidence,
            confidence_breakdown=confidence_breakdown,
            high_confidence_zones=high_confidence_zones,
            low_confidence_warnings=low_confidence_warnings,
            metadata={
                'confidence_method': 'cross_asset_validation',
                'max_boost': self.max_validation_boost,
                'max_penalty': self.max_conflict_penalty
            }
        )

    def _get_base_confidence(self, straddle_result: StraddleAnalysisResult,
                           underlying_result: UnderlyingAnalysisResult) -> np.ndarray:
        """Get base confidence from individual analyses"""
        
        # Use straddle confidence as primary (60% weight)
        straddle_confidence = straddle_result.confidence_scores
        
        # Get underlying confidence (40% weight)
        underlying_confidence = underlying_result.cross_timeframe_confidence
        
        # Ensure same length
        min_length = min(len(straddle_confidence), len(underlying_confidence))
        if min_length == 0:
            return np.array([0.5])  # Default confidence
        
        straddle_conf = straddle_confidence[:min_length]
        underlying_conf = underlying_confidence[:min_length]
        
        # Weighted average (60% straddle, 40% underlying)
        base_confidence = straddle_conf * 0.6 + underlying_conf * 0.4
        
        return base_confidence

    def _calculate_validation_boost(self, trend_validation: TrendDirectionValidation,
                                  volatility_validation: VolatilityRegimeValidation,
                                  level_validation: SupportResistanceValidation,
                                  length: int) -> np.ndarray:
        """Calculate validation boost from cross-asset agreement"""
        
        validation_boost = np.zeros(length)
        
        # Trend validation boost
        if len(trend_validation.agreement_score) >= length:
            trend_boost = trend_validation.agreement_score[:length] * 0.15  # Max 15% boost from trend
            validation_boost += trend_boost
        
        # Volatility validation boost
        if len(volatility_validation.regime_agreement_score) >= length:
            vol_boost = volatility_validation.regime_agreement_score[:length] * 0.1  # Max 10% boost from volatility
            validation_boost += vol_boost
        
        # Level validation boost
        if len(level_validation.level_agreement_score) >= length:
            level_boost = level_validation.level_agreement_score[:length] * 0.05  # Max 5% boost from levels
            validation_boost += level_boost
        
        # Cap the total boost
        validation_boost = np.clip(validation_boost, 0, self.max_validation_boost)
        
        return validation_boost

    def _calculate_conflict_penalty(self, trend_validation: TrendDirectionValidation,
                                  volatility_validation: VolatilityRegimeValidation,
                                  level_validation: SupportResistanceValidation,
                                  length: int) -> np.ndarray:
        """Calculate penalty for cross-asset conflicts"""
        
        conflict_penalty = np.zeros(length)
        
        # Trend conflict penalty
        if len(trend_validation.conflicting_signals) >= length:
            trend_penalty = trend_validation.conflicting_signals[:length] * 0.2  # 20% penalty for trend conflicts
            conflict_penalty += trend_penalty
        
        # Volatility divergence penalty
        if len(volatility_validation.volatility_divergence_signals) >= length:
            vol_penalty = volatility_validation.volatility_divergence_signals[:length] * 0.15  # 15% penalty for vol divergence
            conflict_penalty += vol_penalty
        
        # Level disagreement penalty (based on poor agreement)
        if len(level_validation.level_agreement_score) >= length:
            level_disagreement = 1 - level_validation.level_agreement_score[:length]
            level_penalty = level_disagreement * 0.05  # Max 5% penalty for level disagreement
            conflict_penalty += level_penalty
        
        # Cap the total penalty
        conflict_penalty = np.clip(conflict_penalty, 0, self.max_conflict_penalty)
        
        return conflict_penalty

    def _create_confidence_breakdown(self, base_confidence: np.ndarray,
                                   validation_boost: np.ndarray,
                                   conflict_penalty: np.ndarray,
                                   trend_validation: TrendDirectionValidation,
                                   volatility_validation: VolatilityRegimeValidation,
                                   level_validation: SupportResistanceValidation) -> Dict[str, np.ndarray]:
        """Create detailed confidence breakdown"""
        
        breakdown = {
            'base_confidence': base_confidence,
            'validation_boost': validation_boost,
            'conflict_penalty': conflict_penalty,
            'net_adjustment': validation_boost - conflict_penalty
        }
        
        # Individual validation contributions
        length = len(base_confidence)
        
        if len(trend_validation.agreement_score) >= length:
            breakdown['trend_validation_contribution'] = trend_validation.agreement_score[:length] * 0.15
        
        if len(volatility_validation.regime_agreement_score) >= length:
            breakdown['volatility_validation_contribution'] = volatility_validation.regime_agreement_score[:length] * 0.1
        
        if len(level_validation.level_agreement_score) >= length:
            breakdown['level_validation_contribution'] = level_validation.level_agreement_score[:length] * 0.05
        
        return breakdown


class DynamicWeightingEngine:
    """Engine for dynamic weighting with cross-asset validation adjustments"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # Base weights (60% straddle, 40% underlying)
        self.base_weights = {
            'straddle': config.get('base_straddle_weight', 0.6),
            'underlying': config.get('base_underlying_weight', 0.4)
        }
        
        # Adjustment parameters
        self.max_weight_adjustment = config.get('max_weight_adjustment', 0.2)

    async def calculate_dynamic_weights(self, straddle_result: StraddleAnalysisResult,
                                      underlying_result: UnderlyingAnalysisResult,
                                      confidence_result: CrossAssetConfidenceResult) -> DynamicWeightingResult:
        """Calculate dynamic weights with adaptive adjustments"""
        
        data_length = len(confidence_result.final_confidence)
        
        # Start with base weights
        straddle_weights = np.full(data_length, self.base_weights['straddle'])
        underlying_weights = np.full(data_length, self.base_weights['underlying'])
        
        # Calculate adaptive adjustments
        adaptive_adjustments = self._calculate_adaptive_adjustments(
            straddle_result, underlying_result, confidence_result
        )
        
        # Apply adjustments
        straddle_weights += adaptive_adjustments['straddle']
        underlying_weights += adaptive_adjustments['underlying']
        
        # Ensure weights sum to 1 and stay within bounds
        straddle_weights = np.clip(straddle_weights, 0.2, 0.8)
        underlying_weights = 1 - straddle_weights  # Ensure sum = 1
        
        final_weights = {
            'straddle': straddle_weights,
            'underlying': underlying_weights
        }
        
        # Document weight change reasons
        weight_change_reasons = self._document_weight_changes(adaptive_adjustments)
        
        # Calculate performance-based adjustments
        performance_adjustments = self._calculate_performance_adjustments(straddle_result, underlying_result)
        
        # Calculate cross-validation impact
        cross_validation_impact = self._calculate_cross_validation_impact(confidence_result)
        
        return DynamicWeightingResult(
            base_weights=self.base_weights,
            adaptive_adjustments=adaptive_adjustments,
            final_weights=final_weights,
            weight_change_reasons=weight_change_reasons,
            performance_based_adjustments=performance_adjustments,
            cross_validation_impact=cross_validation_impact,
            metadata={
                'weighting_method': 'dynamic_cross_asset',
                'base_straddle_weight': self.base_weights['straddle'],
                'base_underlying_weight': self.base_weights['underlying']
            }
        )

    def _calculate_adaptive_adjustments(self, straddle_result: StraddleAnalysisResult,
                                      underlying_result: UnderlyingAnalysisResult,
                                      confidence_result: CrossAssetConfidenceResult) -> Dict[str, np.ndarray]:
        """Calculate adaptive weight adjustments"""
        
        data_length = len(confidence_result.final_confidence)
        adjustments = {
            'straddle': np.zeros(data_length),
            'underlying': np.zeros(data_length)
        }
        
        # Adjust based on individual confidence levels
        straddle_confidence = straddle_result.confidence_scores[:data_length]
        underlying_confidence = underlying_result.cross_timeframe_confidence[:data_length]
        
        for i in range(data_length):
            if not np.isnan(straddle_confidence[i]) and not np.isnan(underlying_confidence[i]):
                # Relative confidence adjustment
                confidence_diff = straddle_confidence[i] - underlying_confidence[i]
                
                # Adjust towards higher confidence asset
                adjustment = confidence_diff * 0.1  # Max 10% adjustment from confidence
                adjustment = np.clip(adjustment, -self.max_weight_adjustment, self.max_weight_adjustment)
                
                adjustments['straddle'][i] += adjustment
                adjustments['underlying'][i] -= adjustment
        
        # Adjust based on cross-asset validation
        validation_boost = confidence_result.validation_boost
        conflict_penalty = confidence_result.conflict_penalty
        
        # When cross-validation is strong, balance weights more evenly
        for i in range(data_length):
            net_validation = validation_boost[i] - conflict_penalty[i]
            
            if net_validation > 0.1:  # Strong cross-validation
                # Move towards more balanced weighting
                current_imbalance = abs(0.5 - (self.base_weights['straddle'] + adjustments['straddle'][i]))
                balance_adjustment = current_imbalance * 0.3  # 30% of imbalance
                
                if adjustments['straddle'][i] > 0:
                    adjustments['straddle'][i] -= balance_adjustment
                    adjustments['underlying'][i] += balance_adjustment
                else:
                    adjustments['straddle'][i] += balance_adjustment
                    adjustments['underlying'][i] -= balance_adjustment
        
        return adjustments

    def _document_weight_changes(self, adaptive_adjustments: Dict[str, np.ndarray]) -> Dict[str, List[str]]:
        """Document reasons for weight changes"""
        
        reasons = {
            'straddle_increases': [],
            'straddle_decreases': [],
            'balancing_adjustments': []
        }
        
        straddle_adj = adaptive_adjustments['straddle']
        
        # Analyze adjustment patterns
        large_increases = np.sum(straddle_adj > 0.1)
        large_decreases = np.sum(straddle_adj < -0.1)
        
        if large_increases > 0:
            reasons['straddle_increases'].append(f"High straddle confidence in {large_increases} periods")
        
        if large_decreases > 0:
            reasons['straddle_decreases'].append(f"Low straddle confidence in {large_decreases} periods")
        
        # Cross-validation adjustments
        balanced_periods = np.sum(np.abs(straddle_adj) < 0.05)
        if balanced_periods > len(straddle_adj) * 0.3:
            reasons['balancing_adjustments'].append("Strong cross-validation promoting balanced weighting")
        
        return reasons

    def _calculate_performance_adjustments(self, straddle_result: StraddleAnalysisResult,
                                         underlying_result: UnderlyingAnalysisResult) -> Dict[str, float]:
        """Calculate performance-based weight adjustments"""
        
        adjustments = {}
        
        # Processing time performance
        straddle_time = straddle_result.processing_time_ms
        underlying_time = underlying_result.processing_time_ms
        
        if straddle_time > 0 and underlying_time > 0:
            time_ratio = underlying_time / straddle_time
            if time_ratio > 1.5:  # Underlying much slower
                adjustments['performance_penalty_underlying'] = -0.05
            elif time_ratio < 0.67:  # Straddle much slower
                adjustments['performance_penalty_straddle'] = -0.05
        
        return adjustments

    def _calculate_cross_validation_impact(self, confidence_result: CrossAssetConfidenceResult) -> Dict[str, float]:
        """Calculate impact of cross-validation on weighting"""
        
        impact = {}
        
        # Average validation boost impact
        avg_boost = np.mean(confidence_result.validation_boost)
        impact['avg_validation_boost'] = avg_boost
        
        # Average conflict penalty impact
        avg_penalty = np.mean(confidence_result.conflict_penalty)
        impact['avg_conflict_penalty'] = avg_penalty
        
        # Net cross-validation effect
        impact['net_cross_validation_effect'] = avg_boost - avg_penalty
        
        # High confidence zone percentage
        high_conf_pct = np.mean(confidence_result.high_confidence_zones)
        impact['high_confidence_zone_percentage'] = high_conf_pct
        
        return impact


class CrossAssetIntegrationEngine:
    """Main engine for cross-asset integration and validation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # Initialize validators and engines
        self.trend_validator = TrendDirectionValidator(config)
        self.volatility_validator = VolatilityRegimeValidator(config)
        self.level_validator = SupportResistanceLevelValidator(config)
        self.confidence_engine = CrossAssetConfidenceEngine(config)
        self.weighting_engine = DynamicWeightingEngine(config)

    async def integrate_cross_asset_analysis(self, straddle_result: StraddleAnalysisResult,
                                           underlying_result: UnderlyingAnalysisResult,
                                           dte_integrated_result: DTEIntegratedResult) -> CrossAssetIntegrationResult:
        """Complete cross-asset integration with validation and weighting"""
        
        start_time = time.time()
        
        try:
            # Run validations concurrently
            trend_validation_task = self.trend_validator.validate_trend_directions(straddle_result, underlying_result)
            volatility_validation_task = self.volatility_validator.validate_volatility_regimes(straddle_result, underlying_result)
            level_validation_task = self.level_validator.validate_support_resistance_levels(straddle_result, underlying_result)
            
            trend_validation, volatility_validation, level_validation = await asyncio.gather(
                trend_validation_task, volatility_validation_task, level_validation_task
            )
            
            # Calculate cross-asset confidence
            confidence_result = await self.confidence_engine.calculate_cross_asset_confidence(
                straddle_result, underlying_result, trend_validation, volatility_validation, level_validation
            )
            
            # Calculate dynamic weights
            weighting_result = await self.weighting_engine.calculate_dynamic_weights(
                straddle_result, underlying_result, confidence_result
            )
            
            # Generate integrated signals
            integrated_signals = self._generate_integrated_signals(
                straddle_result, underlying_result, weighting_result,
                trend_validation, volatility_validation, level_validation
            )
            
            # Generate cross-asset regime classification
            cross_asset_regime = self._generate_cross_asset_regime_classification(
                straddle_result, underlying_result, weighting_result,
                trend_validation, volatility_validation
            )
            
            # Create validation summary
            validation_summary = self._create_validation_summary(
                trend_validation, volatility_validation, level_validation, confidence_result
            )
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            return CrossAssetIntegrationResult(
                trend_validation=trend_validation,
                volatility_validation=volatility_validation,
                level_validation=level_validation,
                confidence_result=confidence_result,
                weighting_result=weighting_result,
                integrated_signals=integrated_signals,
                cross_asset_regime_classification=cross_asset_regime,
                validation_summary=validation_summary,
                processing_time_ms=processing_time_ms,
                metadata={
                    'integration_engine': 'cross_asset_comprehensive',
                    'validation_methods': ['trend', 'volatility', 'levels'],
                    'confidence_method': 'multi_factor',
                    'weighting_method': 'dynamic_adaptive'
                }
            )
            
        except Exception as e:
            self.logger.error(f"Cross-asset integration failed: {str(e)}")
            raise

    def _generate_integrated_signals(self, straddle_result: StraddleAnalysisResult,
                                   underlying_result: UnderlyingAnalysisResult,
                                   weighting_result: DynamicWeightingResult,
                                   trend_validation: TrendDirectionValidation,
                                   volatility_validation: VolatilityRegimeValidation,
                                   level_validation: SupportResistanceValidation) -> Dict[str, np.ndarray]:
        """Generate integrated signals using cross-asset validation"""
        
        data_length = len(weighting_result.final_weights['straddle'])
        signals = {}
        
        # Confirmed trend signals (both assets agree on trend direction)
        if len(trend_validation.agreement_score) >= data_length:
            straddle_trend = straddle_result.ema_result.trend_direction[:data_length]
            underlying_trend = underlying_result.ema_result.trend_directions.get('daily', np.array([]))[:data_length]
            
            # Strong bullish signal (both assets bullish + high agreement)
            signals['confirmed_bullish'] = np.logical_and(
                np.logical_and(straddle_trend == 1, underlying_trend == 1),
                trend_validation.agreement_score[:data_length] > 0.8
            ).astype(int)
            
            # Strong bearish signal (both assets bearish + high agreement)
            signals['confirmed_bearish'] = np.logical_and(
                np.logical_and(straddle_trend == -1, underlying_trend == -1),
                trend_validation.agreement_score[:data_length] > 0.8
            ).astype(int)
        
        # Confirmed volatility expansion
        if len(volatility_validation.regime_agreement_score) >= data_length:
            straddle_vol_regime = straddle_result.atr_result.volatility_regime[:data_length]
            underlying_vol_regime = underlying_result.atr_result.volatility_regimes.get('daily', np.array([]))[:data_length]
            
            signals['confirmed_vol_expansion'] = np.logical_and(
                np.logical_and(straddle_vol_regime >= 1, underlying_vol_regime >= 1),
                volatility_validation.regime_agreement_score[:data_length] > 0.7
            ).astype(int)
        
        # Validated breakout signals
        if len(level_validation.confluence_zones) >= data_length:
            straddle_breakout = straddle_result.cpr_result.breakout_signals.get('resistance_breakout', np.zeros(data_length))
            underlying_breakout = underlying_result.cpr_result.breakout_signals.get('daily', {}).get('resistance_breakout', np.zeros(data_length))
            
            signals['validated_breakout'] = np.logical_and(
                np.logical_or(straddle_breakout, underlying_breakout),
                level_validation.confluence_zones[:data_length] > 0.7
            ).astype(int)
        
        return signals

    def _generate_cross_asset_regime_classification(self, straddle_result: StraddleAnalysisResult,
                                                  underlying_result: UnderlyingAnalysisResult,
                                                  weighting_result: DynamicWeightingResult,
                                                  trend_validation: TrendDirectionValidation,
                                                  volatility_validation: VolatilityRegimeValidation) -> np.ndarray:
        """Generate cross-asset regime classification"""
        
        data_length = len(weighting_result.final_weights['straddle'])
        cross_asset_regime = np.full(data_length, 0)
        
        straddle_regime = straddle_result.regime_classification[:data_length]
        underlying_regime = underlying_result.combined_regime_classification[:data_length]
        
        straddle_weights = weighting_result.final_weights['straddle']
        underlying_weights = weighting_result.final_weights['underlying']
        
        for i in range(data_length):
            # Weighted average of regimes
            weighted_regime = (straddle_regime[i] * straddle_weights[i] + 
                             underlying_regime[i] * underlying_weights[i])
            
            # Apply validation adjustments
            if (len(trend_validation.agreement_score) > i and 
                trend_validation.agreement_score[i] > 0.8):
                # High agreement - use weighted regime
                cross_asset_regime[i] = int(round(weighted_regime))
            elif (len(volatility_validation.regime_agreement_score) > i and
                  volatility_validation.regime_agreement_score[i] < 0.3):
                # High disagreement - use neutral regime
                cross_asset_regime[i] = 7  # Neutral/uncertain
            else:
                # Moderate agreement - use weighted regime
                cross_asset_regime[i] = int(round(weighted_regime))
        
        return cross_asset_regime

    def _create_validation_summary(self, trend_validation: TrendDirectionValidation,
                                 volatility_validation: VolatilityRegimeValidation,
                                 level_validation: SupportResistanceValidation,
                                 confidence_result: CrossAssetConfidenceResult) -> Dict[str, float]:
        """Create validation summary statistics"""
        
        summary = {}
        
        # Trend validation summary
        if len(trend_validation.agreement_score) > 0:
            summary['avg_trend_agreement'] = np.mean(trend_validation.agreement_score)
            summary['trend_directional_consistency'] = trend_validation.directional_consistency
            summary['trend_strength_correlation'] = trend_validation.trend_strength_correlation
        
        # Volatility validation summary
        if len(volatility_validation.regime_agreement_score) > 0:
            summary['avg_volatility_agreement'] = np.mean(volatility_validation.regime_agreement_score)
            summary['volatility_correlation'] = volatility_validation.cross_asset_volatility_correlation
            summary['regime_transition_alignment'] = volatility_validation.regime_transition_alignment
        
        # Level validation summary
        if len(level_validation.level_agreement_score) > 0:
            summary['avg_level_agreement'] = np.mean(level_validation.level_agreement_score)
            summary['level_correlation'] = level_validation.cross_asset_level_correlation
            summary['breakout_confirmation_rate'] = level_validation.breakout_confirmation_rate
        
        # Overall confidence summary
        summary['avg_final_confidence'] = np.mean(confidence_result.final_confidence)
        summary['avg_validation_boost'] = np.mean(confidence_result.validation_boost)
        summary['avg_conflict_penalty'] = np.mean(confidence_result.conflict_penalty)
        summary['high_confidence_percentage'] = np.mean(confidence_result.high_confidence_zones) * 100
        
        return summary