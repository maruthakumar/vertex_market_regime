"""
Multi-Timeframe IV Percentile Momentum System - Component 4 Enhancement

Advanced IV percentile momentum analysis across 5min/15min/30min/1hour timeframes
with momentum regime classification, acceleration tracking, momentum divergence
detection, and cross-timeframe correlation analysis for sophisticated momentum-based
regime detection and trading signal generation.

This module provides institutional-grade multi-timeframe momentum analysis
with comprehensive divergence detection and signal generation capabilities.
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
from .historical_percentile_database import HistoricalPercentileDatabase


class MomentumTimeframe(Enum):
    """Momentum analysis timeframes"""
    FIVE_MIN = "5min"
    FIFTEEN_MIN = "15min"  
    THIRTY_MIN = "30min"
    ONE_HOUR = "1hour"


class MomentumRegime(Enum):
    """Momentum regime classification"""
    STRONG_TRENDING_UP = "strong_trending_up"
    MODERATE_TRENDING_UP = "moderate_trending_up"
    WEAK_TRENDING_UP = "weak_trending_up"
    SIDEWAYS = "sideways"
    WEAK_TRENDING_DOWN = "weak_trending_down"
    MODERATE_TRENDING_DOWN = "moderate_trending_down"
    STRONG_TRENDING_DOWN = "strong_trending_down"
    MEAN_REVERTING = "mean_reverting"


@dataclass
class TimeframeMomentumMetrics:
    """Single timeframe momentum metrics"""
    
    # Timeframe identification
    timeframe: MomentumTimeframe
    lookback_periods: int
    
    # Core momentum metrics
    momentum_value: float
    momentum_percentile: float
    momentum_strength: str
    
    # Momentum characteristics
    momentum_direction: str  # up, down, neutral
    momentum_acceleration: float
    momentum_consistency: float
    
    # Signal quality
    signal_strength: float
    signal_confidence: float
    noise_level: float
    
    # Regime classification
    momentum_regime: MomentumRegime
    regime_confidence: float
    
    # Relative metrics
    relative_momentum: float  # vs other timeframes
    momentum_rank: int       # rank among timeframes
    
    # Quality indicators
    data_completeness: float
    calculation_time_ms: float


@dataclass
class MomentumDivergenceAnalysis:
    """Momentum divergence detection analysis"""
    
    # Divergence identification
    has_divergence: bool
    divergence_type: str  # bullish, bearish, none
    divergence_strength: float
    
    # Divergence components
    price_momentum_direction: str
    iv_percentile_momentum_direction: str
    divergence_duration: int
    
    # Signal characteristics
    divergence_significance: float
    reversal_probability: float
    signal_reliability: float
    
    # Timeframe analysis
    primary_divergence_timeframe: MomentumTimeframe
    supporting_timeframes: List[MomentumTimeframe]
    conflicting_timeframes: List[MomentumTimeframe]
    
    # Risk assessment
    divergence_risk_score: float
    false_signal_probability: float
    
    def get_divergence_signal(self) -> str:
        """Get divergence signal classification"""
        
        if not self.has_divergence:
            return "no_divergence"
        
        if self.divergence_strength > 0.7 and self.signal_reliability > 0.6:
            return f"strong_{self.divergence_type}_divergence"
        elif self.divergence_strength > 0.5:
            return f"moderate_{self.divergence_type}_divergence"
        else:
            return f"weak_{self.divergence_type}_divergence"


@dataclass
class CrossTimeframeCorrelation:
    """Cross-timeframe momentum correlation analysis"""
    
    # Correlation matrix
    correlation_matrix: Dict[MomentumTimeframe, Dict[MomentumTimeframe, float]]
    
    # Correlation characteristics
    average_correlation: float
    correlation_stability: float
    correlation_regime: str
    
    # Dominant relationships
    strongest_correlation: Tuple[MomentumTimeframe, MomentumTimeframe, float]
    weakest_correlation: Tuple[MomentumTimeframe, MomentumTimeframe, float]
    
    # Regime alignment
    aligned_timeframes: List[MomentumTimeframe]
    divergent_timeframes: List[MomentumTimeframe]
    alignment_score: float
    
    # Signal consensus
    consensus_direction: str
    consensus_strength: float
    consensus_reliability: float
    
    def get_correlation_regime(self) -> str:
        """Get correlation regime classification"""
        
        if self.average_correlation > 0.7:
            return "high_correlation"
        elif self.average_correlation > 0.4:
            return "moderate_correlation"
        elif self.average_correlation > 0.1:
            return "low_correlation"
        else:
            return "no_correlation"


@dataclass
class MomentumAccelerationAnalysis:
    """Momentum acceleration (second-order) analysis"""
    
    # Acceleration metrics
    momentum_acceleration: float
    acceleration_trend: str  # accelerating, decelerating, stable
    acceleration_strength: float
    
    # Rate of change analysis
    first_derivative: float   # momentum change rate
    second_derivative: float  # acceleration change rate
    momentum_curvature: float
    
    # Regime change prediction
    regime_change_probability: float
    predicted_momentum_direction: str
    prediction_confidence: float
    
    # Critical points
    inflection_point_proximity: float
    momentum_exhaustion_risk: float
    reversal_signal_strength: float
    
    def get_acceleration_signal(self) -> str:
        """Get acceleration-based signal"""
        
        if self.momentum_acceleration > 0.3:
            if self.acceleration_trend == "accelerating":
                return "strong_momentum_building"
            else:
                return "momentum_decelerating"
        elif self.momentum_acceleration < -0.3:
            if self.acceleration_trend == "accelerating":
                return "strong_reversal_building"
            else:
                return "reversal_decelerating"
        else:
            return "stable_momentum"


@dataclass
class ComprehensiveMomentumResult:
    """Comprehensive multi-timeframe momentum analysis result"""
    
    # Timeframe-specific results
    timeframe_metrics: Dict[MomentumTimeframe, TimeframeMomentumMetrics]
    
    # Cross-timeframe analysis
    divergence_analysis: MomentumDivergenceAnalysis
    correlation_analysis: CrossTimeframeCorrelation
    acceleration_analysis: MomentumAccelerationAnalysis
    
    # Aggregate signals
    dominant_momentum_timeframe: MomentumTimeframe
    overall_momentum_direction: str
    momentum_regime_classification: MomentumRegime
    
    # Confidence and quality
    analysis_confidence: float
    signal_quality_score: float
    data_reliability: float
    
    # Trading signals
    primary_signal: str
    signal_strength: float
    risk_level: str
    
    # Performance metrics
    total_processing_time_ms: float
    components_analyzed: List[str]
    
    def get_master_momentum_signal(self) -> str:
        """Get master momentum signal for Component 4"""
        
        base_signal = self.overall_momentum_direction
        regime = self.momentum_regime_classification.value
        
        if self.analysis_confidence > 0.8 and self.signal_quality_score > 0.7:
            return f"strong_{regime}_{base_signal}"
        elif self.analysis_confidence > 0.6:
            return f"moderate_{regime}_{base_signal}"
        else:
            return f"weak_{regime}_{base_signal}"


class MultiTimeframeMomentumSystem:
    """
    Advanced Multi-Timeframe IV Percentile Momentum System with comprehensive
    momentum analysis, divergence detection, and cross-timeframe correlation.
    
    Features:
    - Multi-timeframe momentum calculation (5min/15min/30min/1hour)
    - Momentum regime classification (trending/mean-reverting/sideways)
    - Second-order momentum (acceleration) tracking
    - Momentum divergence detection with signal reliability
    - Cross-timeframe correlation analysis
    - Comprehensive signal generation and confidence scoring
    - Risk-adjusted momentum regime determination
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # Timeframe configuration
        self.timeframes = {
            MomentumTimeframe.FIVE_MIN: {"periods": 5, "lookback": 20, "weight": 0.4},
            MomentumTimeframe.FIFTEEN_MIN: {"periods": 15, "lookback": 12, "weight": 0.3},
            MomentumTimeframe.THIRTY_MIN: {"periods": 30, "lookback": 8, "weight": 0.2},
            MomentumTimeframe.ONE_HOUR: {"periods": 60, "lookback": 6, "weight": 0.1}
        }
        
        # Momentum calculation configuration
        self.momentum_lookback_multiplier = config.get('momentum_lookback_multiplier', 1.5)
        self.momentum_smoothing = config.get('momentum_smoothing', True)
        self.min_data_points = config.get('min_momentum_data_points', 10)
        
        # Divergence detection configuration  
        self.divergence_threshold = config.get('divergence_threshold', 0.3)
        self.divergence_min_duration = config.get('divergence_min_duration', 3)
        self.divergence_confirmation_periods = config.get('divergence_confirmation_periods', 2)
        
        # Correlation analysis configuration
        self.correlation_window = config.get('correlation_window', 50)
        self.min_correlation_significance = config.get('min_correlation_significance', 0.05)
        
        # Performance configuration
        self.processing_budget_ms = config.get('momentum_processing_budget_ms', 100)
        
        self.logger.info("Multi-Timeframe Momentum System initialized with 4 timeframe analysis")
    
    def analyze_multi_timeframe_momentum(self,
                                       current_iv_data: IVPercentileData,
                                       historical_iv_data: List[IVPercentileData],
                                       historical_db: HistoricalPercentileDatabase) -> ComprehensiveMomentumResult:
        """
        Analyze IV percentile momentum across multiple timeframes
        
        Args:
            current_iv_data: Current IV percentile data
            historical_iv_data: Historical IV data for momentum calculation
            historical_db: Historical database for percentile context
            
        Returns:
            ComprehensiveMomentumResult with complete momentum analysis
        """
        start_time = time.time()
        
        try:
            # Calculate momentum for each timeframe
            timeframe_metrics = {}
            
            for timeframe in MomentumTimeframe:
                timeframe_config = self.timeframes[timeframe]
                
                momentum_metrics = self._calculate_timeframe_momentum(
                    timeframe, current_iv_data, historical_iv_data, 
                    timeframe_config, historical_db
                )
                
                timeframe_metrics[timeframe] = momentum_metrics
            
            # Cross-timeframe divergence analysis
            divergence_analysis = self._analyze_momentum_divergence(
                timeframe_metrics, current_iv_data, historical_iv_data
            )
            
            # Cross-timeframe correlation analysis
            correlation_analysis = self._analyze_cross_timeframe_correlation(
                timeframe_metrics, historical_iv_data
            )
            
            # Acceleration analysis
            acceleration_analysis = self._analyze_momentum_acceleration(
                timeframe_metrics, historical_iv_data
            )
            
            # Determine dominant momentum and overall direction
            dominant_analysis = self._determine_dominant_momentum(timeframe_metrics)
            
            # Generate comprehensive signals
            signal_analysis = self._generate_momentum_signals(
                timeframe_metrics, divergence_analysis, 
                correlation_analysis, acceleration_analysis
            )
            
            # Calculate confidence and quality scores
            confidence_quality = self._calculate_confidence_quality(
                timeframe_metrics, divergence_analysis, correlation_analysis
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            # Create comprehensive result
            result = ComprehensiveMomentumResult(
                timeframe_metrics=timeframe_metrics,
                divergence_analysis=divergence_analysis,
                correlation_analysis=correlation_analysis,
                acceleration_analysis=acceleration_analysis,
                dominant_momentum_timeframe=dominant_analysis['timeframe'],
                overall_momentum_direction=dominant_analysis['direction'],
                momentum_regime_classification=dominant_analysis['regime'],
                analysis_confidence=confidence_quality['confidence'],
                signal_quality_score=confidence_quality['quality'],
                data_reliability=confidence_quality['reliability'],
                primary_signal=signal_analysis['primary'],
                signal_strength=signal_analysis['strength'],
                risk_level=signal_analysis['risk_level'],
                total_processing_time_ms=processing_time,
                components_analyzed=['timeframe_momentum', 'divergence', 'correlation', 'acceleration']
            )
            
            self.logger.debug(f"Multi-timeframe momentum analysis completed: "
                            f"Dominant={result.dominant_momentum_timeframe.value}, "
                            f"Direction={result.overall_momentum_direction}, "
                            f"Confidence={result.analysis_confidence:.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Multi-timeframe momentum analysis failed: {e}")
            return self._get_default_momentum_result()
    
    def calculate_momentum_regime_classification(self,
                                               timeframe_momentum: Dict[MomentumTimeframe, float],
                                               momentum_strength: Dict[MomentumTimeframe, float]) -> Dict[MomentumTimeframe, MomentumRegime]:
        """
        Classify momentum into trending/mean-reverting/sideways regimes
        
        Args:
            timeframe_momentum: Momentum values by timeframe
            momentum_strength: Momentum strength by timeframe
            
        Returns:
            Dictionary with momentum regime for each timeframe
        """
        try:
            regime_classifications = {}
            
            for timeframe in MomentumTimeframe:
                momentum = timeframe_momentum.get(timeframe, 0.0)
                strength = momentum_strength.get(timeframe, 0.0)
                
                # Classify momentum regime
                regime = self._classify_momentum_regime(momentum, strength)
                regime_classifications[timeframe] = regime
            
            return regime_classifications
            
        except Exception as e:
            self.logger.error(f"Momentum regime classification failed: {e}")
            return {tf: MomentumRegime.SIDEWAYS for tf in MomentumTimeframe}
    
    def detect_momentum_divergences(self,
                                  iv_percentile_momentum: Dict[MomentumTimeframe, float],
                                  price_momentum: Dict[MomentumTimeframe, float]) -> MomentumDivergenceAnalysis:
        """
        Detect divergences between price movement and IV percentile momentum
        
        Args:
            iv_percentile_momentum: IV percentile momentum by timeframe
            price_momentum: Price momentum by timeframe (for comparison)
            
        Returns:
            MomentumDivergenceAnalysis with divergence detection results
        """
        try:
            # Analyze divergences across timeframes
            divergences = []
            
            for timeframe in MomentumTimeframe:
                iv_momentum = iv_percentile_momentum.get(timeframe, 0.0)
                px_momentum = price_momentum.get(timeframe, 0.0)
                
                # Check for divergence
                divergence_strength = self._calculate_divergence_strength(iv_momentum, px_momentum)
                
                if abs(divergence_strength) > self.divergence_threshold:
                    divergence_type = "bullish" if divergence_strength > 0 else "bearish"
                    divergences.append({
                        'timeframe': timeframe,
                        'type': divergence_type,
                        'strength': abs(divergence_strength)
                    })
            
            # Determine primary divergence
            if divergences:
                primary_divergence = max(divergences, key=lambda x: x['strength'])
                has_divergence = True
                divergence_type = primary_divergence['type']
                divergence_strength = primary_divergence['strength']
                primary_timeframe = primary_divergence['timeframe']
            else:
                has_divergence = False
                divergence_type = "none"
                divergence_strength = 0.0
                primary_timeframe = MomentumTimeframe.FIFTEEN_MIN
            
            # Supporting and conflicting timeframes
            supporting_timeframes = []
            conflicting_timeframes = []
            
            if has_divergence:
                for div in divergences:
                    if div['type'] == divergence_type and div['timeframe'] != primary_timeframe:
                        supporting_timeframes.append(div['timeframe'])
                
                # Find conflicting signals
                for timeframe in MomentumTimeframe:
                    if timeframe not in [d['timeframe'] for d in divergences]:
                        iv_mom = iv_percentile_momentum.get(timeframe, 0.0)
                        px_mom = price_momentum.get(timeframe, 0.0)
                        
                        # Check if momentum directions align (no divergence)
                        if (iv_mom > 0 and px_mom > 0) or (iv_mom < 0 and px_mom < 0):
                            conflicting_timeframes.append(timeframe)
            
            # Calculate signal characteristics
            divergence_significance = self._calculate_divergence_significance(
                divergence_strength, len(supporting_timeframes)
            )
            
            reversal_probability = min(1.0, divergence_strength * 0.8) if has_divergence else 0.0
            
            signal_reliability = self._calculate_signal_reliability(
                divergence_strength, len(supporting_timeframes), len(conflicting_timeframes)
            )
            
            # Risk assessment
            divergence_risk_score = divergence_strength * 0.6 if has_divergence else 0.1
            false_signal_probability = max(0.1, 1.0 - signal_reliability)
            
            return MomentumDivergenceAnalysis(
                has_divergence=has_divergence,
                divergence_type=divergence_type,
                divergence_strength=divergence_strength,
                price_momentum_direction=self._get_momentum_direction(
                    sum(price_momentum.values()) / len(price_momentum)
                ),
                iv_percentile_momentum_direction=self._get_momentum_direction(
                    sum(iv_percentile_momentum.values()) / len(iv_percentile_momentum)
                ),
                divergence_duration=len(divergences),  # Simplified
                divergence_significance=divergence_significance,
                reversal_probability=reversal_probability,
                signal_reliability=signal_reliability,
                primary_divergence_timeframe=primary_timeframe,
                supporting_timeframes=supporting_timeframes,
                conflicting_timeframes=conflicting_timeframes,
                divergence_risk_score=divergence_risk_score,
                false_signal_probability=false_signal_probability
            )
            
        except Exception as e:
            self.logger.error(f"Momentum divergence detection failed: {e}")
            return self._get_default_divergence_analysis()
    
    def analyze_cross_timeframe_correlations(self,
                                           timeframe_momentum: Dict[MomentumTimeframe, List[float]]) -> CrossTimeframeCorrelation:
        """
        Analyze momentum correlations across different timeframes
        
        Args:
            timeframe_momentum: Historical momentum values by timeframe
            
        Returns:
            CrossTimeframeCorrelation with correlation analysis
        """
        try:
            # Build correlation matrix
            correlation_matrix = {}
            all_correlations = []
            
            for tf1 in MomentumTimeframe:
                correlation_matrix[tf1] = {}
                momentum1 = timeframe_momentum.get(tf1, [])
                
                for tf2 in MomentumTimeframe:
                    momentum2 = timeframe_momentum.get(tf2, [])
                    
                    if tf1 == tf2:
                        correlation = 1.0
                    elif len(momentum1) >= 10 and len(momentum2) >= 10:
                        # Calculate correlation with equal length arrays
                        min_len = min(len(momentum1), len(momentum2))
                        corr = np.corrcoef(momentum1[:min_len], momentum2[:min_len])[0, 1]
                        correlation = float(corr) if not np.isnan(corr) else 0.0
                    else:
                        correlation = 0.0
                    
                    correlation_matrix[tf1][tf2] = correlation
                    
                    if tf1 != tf2:
                        all_correlations.append(abs(correlation))
            
            # Calculate aggregate metrics
            average_correlation = float(np.mean(all_correlations)) if all_correlations else 0.0
            correlation_stability = 1.0 - float(np.std(all_correlations)) if all_correlations else 0.5
            
            # Find strongest and weakest correlations
            max_corr = -1.0
            min_corr = 1.0
            strongest_pair = (MomentumTimeframe.FIVE_MIN, MomentumTimeframe.FIFTEEN_MIN, 0.0)
            weakest_pair = (MomentumTimeframe.FIVE_MIN, MomentumTimeframe.FIFTEEN_MIN, 0.0)
            
            for tf1 in MomentumTimeframe:
                for tf2 in MomentumTimeframe:
                    if tf1 != tf2:
                        corr = correlation_matrix[tf1][tf2]
                        if abs(corr) > max_corr:
                            max_corr = abs(corr)
                            strongest_pair = (tf1, tf2, corr)
                        if abs(corr) < min_corr:
                            min_corr = abs(corr)
                            weakest_pair = (tf1, tf2, corr)
            
            # Determine regime alignment
            current_momentum_directions = {}
            for tf in MomentumTimeframe:
                momentum_data = timeframe_momentum.get(tf, [])
                if momentum_data:
                    current_momentum_directions[tf] = "up" if momentum_data[-1] > 0 else "down"
                else:
                    current_momentum_directions[tf] = "neutral"
            
            # Find aligned and divergent timeframes
            momentum_directions = list(current_momentum_directions.values())
            dominant_direction = max(set(momentum_directions), key=momentum_directions.count)
            
            aligned_timeframes = [tf for tf, direction in current_momentum_directions.items() 
                                if direction == dominant_direction]
            divergent_timeframes = [tf for tf, direction in current_momentum_directions.items() 
                                  if direction != dominant_direction]
            
            alignment_score = len(aligned_timeframes) / len(MomentumTimeframe)
            
            # Signal consensus
            consensus_direction = dominant_direction
            consensus_strength = alignment_score
            consensus_reliability = min(1.0, average_correlation * alignment_score)
            
            # Correlation regime
            correlation_regime = self._classify_correlation_regime(average_correlation)
            
            return CrossTimeframeCorrelation(
                correlation_matrix=correlation_matrix,
                average_correlation=average_correlation,
                correlation_stability=correlation_stability,
                correlation_regime=correlation_regime,
                strongest_correlation=strongest_pair,
                weakest_correlation=weakest_pair,
                aligned_timeframes=aligned_timeframes,
                divergent_timeframes=divergent_timeframes,
                alignment_score=alignment_score,
                consensus_direction=consensus_direction,
                consensus_strength=consensus_strength,
                consensus_reliability=consensus_reliability
            )
            
        except Exception as e:
            self.logger.error(f"Cross-timeframe correlation analysis failed: {e}")
            return self._get_default_correlation_analysis()
    
    def _calculate_timeframe_momentum(self,
                                    timeframe: MomentumTimeframe,
                                    current_iv_data: IVPercentileData,
                                    historical_iv_data: List[IVPercentileData],
                                    timeframe_config: Dict[str, Any],
                                    historical_db: HistoricalPercentileDatabase) -> TimeframeMomentumMetrics:
        """Calculate momentum metrics for specific timeframe"""
        
        start_time = time.time()
        
        try:
            lookback_periods = timeframe_config['lookback']
            weight = timeframe_config['weight']
            
            # Extract IV percentile values for momentum calculation
            iv_percentiles = []
            
            # Add current data point
            current_atm_iv = self._calculate_atm_iv(current_iv_data)
            current_distribution = historical_db.get_dte_percentile_distribution(current_iv_data.dte)
            
            if current_distribution:
                current_percentile = current_distribution.calculate_percentile_rank(current_atm_iv)
                iv_percentiles.append(current_percentile)
            
            # Add historical data points
            for hist_data in historical_iv_data[-lookback_periods:]:
                hist_atm_iv = self._calculate_atm_iv(hist_data)
                hist_distribution = historical_db.get_dte_percentile_distribution(hist_data.dte)
                
                if hist_distribution:
                    hist_percentile = hist_distribution.calculate_percentile_rank(hist_atm_iv)
                    iv_percentiles.append(hist_percentile)
            
            if len(iv_percentiles) < 3:
                return self._get_default_timeframe_metrics(timeframe)
            
            # Calculate momentum
            momentum_value = self._calculate_momentum_value(iv_percentiles, lookback_periods)
            
            # Calculate momentum percentile (vs historical momentum values)
            momentum_percentile = self._calculate_momentum_percentile(momentum_value, timeframe)
            
            # Classify momentum strength
            momentum_strength = self._classify_momentum_strength(momentum_value)
            
            # Determine momentum direction
            momentum_direction = self._get_momentum_direction(momentum_value)
            
            # Calculate acceleration
            momentum_acceleration = self._calculate_acceleration(iv_percentiles)
            
            # Calculate consistency
            momentum_consistency = self._calculate_momentum_consistency(iv_percentiles)
            
            # Signal quality assessment
            signal_strength = min(1.0, abs(momentum_value) * 2)  # Amplify for signal
            signal_confidence = momentum_consistency * weight
            noise_level = 1.0 - momentum_consistency
            
            # Momentum regime classification
            momentum_regime = self._classify_momentum_regime(momentum_value, signal_strength)
            regime_confidence = signal_confidence
            
            # Relative metrics (simplified - would compare with other timeframes)
            relative_momentum = momentum_value  # Would be adjusted relative to other timeframes
            momentum_rank = 1  # Would be ranked among timeframes
            
            # Quality indicators
            data_completeness = min(1.0, len(iv_percentiles) / lookback_periods)
            
            processing_time = (time.time() - start_time) * 1000
            
            return TimeframeMomentumMetrics(
                timeframe=timeframe,
                lookback_periods=lookback_periods,
                momentum_value=momentum_value,
                momentum_percentile=momentum_percentile,
                momentum_strength=momentum_strength,
                momentum_direction=momentum_direction,
                momentum_acceleration=momentum_acceleration,
                momentum_consistency=momentum_consistency,
                signal_strength=signal_strength,
                signal_confidence=signal_confidence,
                noise_level=noise_level,
                momentum_regime=momentum_regime,
                regime_confidence=regime_confidence,
                relative_momentum=relative_momentum,
                momentum_rank=momentum_rank,
                data_completeness=data_completeness,
                calculation_time_ms=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Timeframe momentum calculation failed for {timeframe.value}: {e}")
            return self._get_default_timeframe_metrics(timeframe)
    
    def _calculate_atm_iv(self, iv_data: IVPercentileData) -> float:
        """Calculate ATM IV from IV data"""
        
        atm_idx = np.argmin(np.abs(iv_data.strikes - iv_data.atm_strike))
        atm_ce_iv = iv_data.ce_iv[atm_idx] if not np.isnan(iv_data.ce_iv[atm_idx]) else 0.0
        atm_pe_iv = iv_data.pe_iv[atm_idx] if not np.isnan(iv_data.pe_iv[atm_idx]) else 0.0
        
        return float((atm_ce_iv + atm_pe_iv) / 2) if (atm_ce_iv > 0 or atm_pe_iv > 0) else 0.0
    
    def _calculate_momentum_value(self, values: List[float], lookback: int) -> float:
        """Calculate momentum value from time series"""
        
        if len(values) < 2:
            return 0.0
        
        # Simple momentum: current vs lookback average
        current = values[0]  # Most recent
        historical_avg = np.mean(values[1:min(len(values), lookback+1)])
        
        if historical_avg == 0:
            return 0.0
        
        momentum = (current - historical_avg) / historical_avg
        return float(momentum)
    
    def _calculate_momentum_percentile(self, momentum: float, timeframe: MomentumTimeframe) -> float:
        """Calculate momentum percentile (simplified)"""
        
        # Mock percentile calculation - would use historical momentum distribution
        # For now, convert momentum to percentile scale
        normalized_momentum = np.tanh(momentum * 2)  # Bound between -1 and 1
        percentile = (normalized_momentum + 1) * 50  # Convert to 0-100 scale
        
        return float(max(0, min(100, percentile)))
    
    def _classify_momentum_strength(self, momentum: float) -> str:
        """Classify momentum strength"""
        
        abs_momentum = abs(momentum)
        
        if abs_momentum > 0.15:
            return "very_strong"
        elif abs_momentum > 0.10:
            return "strong"
        elif abs_momentum > 0.05:
            return "moderate"
        elif abs_momentum > 0.02:
            return "weak"
        else:
            return "very_weak"
    
    def _get_momentum_direction(self, momentum: float) -> str:
        """Get momentum direction"""
        
        if momentum > 0.02:
            return "up"
        elif momentum < -0.02:
            return "down"
        else:
            return "neutral"
    
    def _calculate_acceleration(self, values: List[float]) -> float:
        """Calculate momentum acceleration (second derivative)"""
        
        if len(values) < 3:
            return 0.0
        
        # Calculate first differences (velocity)
        velocities = [values[i] - values[i+1] for i in range(len(values)-1)]
        
        if len(velocities) < 2:
            return 0.0
        
        # Calculate second differences (acceleration)
        acceleration = velocities[0] - velocities[1]
        
        return float(acceleration)
    
    def _calculate_momentum_consistency(self, values: List[float]) -> float:
        """Calculate momentum consistency score"""
        
        if len(values) < 3:
            return 0.5
        
        # Calculate changes between consecutive values
        changes = [values[i] - values[i+1] for i in range(len(values)-1)]
        
        if not changes:
            return 0.5
        
        # Consistency based on directional agreement
        positive_changes = sum(1 for change in changes if change > 0)
        negative_changes = sum(1 for change in changes if change < 0)
        
        max_directional = max(positive_changes, negative_changes)
        consistency = max_directional / len(changes) if changes else 0.5
        
        return float(consistency)
    
    def _classify_momentum_regime(self, momentum: float, strength: float) -> MomentumRegime:
        """Classify momentum regime"""
        
        abs_momentum = abs(momentum)
        
        if momentum > 0.1 and strength > 0.7:
            return MomentumRegime.STRONG_TRENDING_UP
        elif momentum > 0.05 and strength > 0.5:
            return MomentumRegime.MODERATE_TRENDING_UP
        elif momentum > 0.02:
            return MomentumRegime.WEAK_TRENDING_UP
        elif momentum < -0.1 and strength > 0.7:
            return MomentumRegime.STRONG_TRENDING_DOWN
        elif momentum < -0.05 and strength > 0.5:
            return MomentumRegime.MODERATE_TRENDING_DOWN
        elif momentum < -0.02:
            return MomentumRegime.WEAK_TRENDING_DOWN
        elif abs_momentum < 0.02 and strength < 0.3:
            return MomentumRegime.MEAN_REVERTING
        else:
            return MomentumRegime.SIDEWAYS
    
    def _analyze_momentum_divergence(self,
                                   timeframe_metrics: Dict[MomentumTimeframe, TimeframeMomentumMetrics],
                                   current_iv_data: IVPercentileData,
                                   historical_iv_data: List[IVPercentileData]) -> MomentumDivergenceAnalysis:
        """Analyze momentum divergence across timeframes"""
        
        # Extract momentum values
        iv_momentum = {tf: metrics.momentum_value for tf, metrics in timeframe_metrics.items()}
        
        # Mock price momentum (would come from price data)
        price_momentum = {tf: metrics.momentum_value * 0.8 for tf, metrics in timeframe_metrics.items()}
        
        return self.detect_momentum_divergences(iv_momentum, price_momentum)
    
    def _analyze_cross_timeframe_correlation(self,
                                           timeframe_metrics: Dict[MomentumTimeframe, TimeframeMomentumMetrics],
                                           historical_iv_data: List[IVPercentileData]) -> CrossTimeframeCorrelation:
        """Analyze cross-timeframe correlations"""
        
        # Extract momentum time series (simplified)
        timeframe_momentum = {}
        
        for timeframe, metrics in timeframe_metrics.items():
            # Mock time series - would be calculated from historical data
            momentum_series = [metrics.momentum_value + np.random.normal(0, 0.1) for _ in range(20)]
            timeframe_momentum[timeframe] = momentum_series
        
        return self.analyze_cross_timeframe_correlations(timeframe_momentum)
    
    def _analyze_momentum_acceleration(self,
                                     timeframe_metrics: Dict[MomentumTimeframe, TimeframeMomentumMetrics],
                                     historical_iv_data: List[IVPercentileData]) -> MomentumAccelerationAnalysis:
        """Analyze momentum acceleration across timeframes"""
        
        # Calculate aggregate acceleration
        accelerations = [metrics.momentum_acceleration for metrics in timeframe_metrics.values()]
        avg_acceleration = float(np.mean(accelerations)) if accelerations else 0.0
        
        # Determine acceleration trend
        if avg_acceleration > 0.05:
            acceleration_trend = "accelerating"
        elif avg_acceleration < -0.05:
            acceleration_trend = "decelerating" 
        else:
            acceleration_trend = "stable"
        
        acceleration_strength = min(1.0, abs(avg_acceleration) * 10)
        
        # Calculate derivatives (simplified)
        momentum_values = [metrics.momentum_value for metrics in timeframe_metrics.values()]
        first_derivative = float(np.mean(np.diff(momentum_values))) if len(momentum_values) > 1 else 0.0
        second_derivative = avg_acceleration
        momentum_curvature = abs(second_derivative)
        
        # Prediction metrics
        regime_change_probability = min(1.0, acceleration_strength * 0.8)
        
        if avg_acceleration > 0:
            predicted_direction = "up"
        elif avg_acceleration < 0:
            predicted_direction = "down"
        else:
            predicted_direction = "neutral"
        
        prediction_confidence = acceleration_strength
        
        # Critical points analysis
        inflection_point_proximity = min(1.0, momentum_curvature * 2)
        momentum_exhaustion_risk = max(0.1, acceleration_strength * 0.6)
        reversal_signal_strength = inflection_point_proximity * 0.8
        
        return MomentumAccelerationAnalysis(
            momentum_acceleration=avg_acceleration,
            acceleration_trend=acceleration_trend,
            acceleration_strength=acceleration_strength,
            first_derivative=first_derivative,
            second_derivative=second_derivative,
            momentum_curvature=momentum_curvature,
            regime_change_probability=regime_change_probability,
            predicted_momentum_direction=predicted_direction,
            prediction_confidence=prediction_confidence,
            inflection_point_proximity=inflection_point_proximity,
            momentum_exhaustion_risk=momentum_exhaustion_risk,
            reversal_signal_strength=reversal_signal_strength
        )
    
    def _determine_dominant_momentum(self, 
                                   timeframe_metrics: Dict[MomentumTimeframe, TimeframeMomentumMetrics]) -> Dict[str, Any]:
        """Determine dominant momentum timeframe and characteristics"""
        
        # Find dominant timeframe by signal strength
        dominant_timeframe = max(timeframe_metrics.keys(), 
                               key=lambda tf: timeframe_metrics[tf].signal_strength)
        
        dominant_metrics = timeframe_metrics[dominant_timeframe]
        
        # Determine overall direction
        weighted_momentum = sum(
            metrics.momentum_value * self.timeframes[tf]['weight']
            for tf, metrics in timeframe_metrics.items()
        )
        
        overall_direction = self._get_momentum_direction(weighted_momentum)
        
        # Determine regime
        regime = dominant_metrics.momentum_regime
        
        return {
            'timeframe': dominant_timeframe,
            'direction': overall_direction,
            'regime': regime,
            'weighted_momentum': weighted_momentum
        }
    
    def _generate_momentum_signals(self,
                                 timeframe_metrics: Dict[MomentumTimeframe, TimeframeMomentumMetrics],
                                 divergence_analysis: MomentumDivergenceAnalysis,
                                 correlation_analysis: CrossTimeframeCorrelation,
                                 acceleration_analysis: MomentumAccelerationAnalysis) -> Dict[str, Any]:
        """Generate comprehensive momentum signals"""
        
        # Primary signal based on dominant momentum
        dominant_direction = max(
            ['up', 'down', 'neutral'],
            key=lambda direction: sum(1 for metrics in timeframe_metrics.values() 
                                    if metrics.momentum_direction == direction)
        )
        
        # Signal strength from correlation and consistency
        signal_strength = (correlation_analysis.consensus_strength * 0.5 + 
                          correlation_analysis.alignment_score * 0.5)
        
        # Risk level assessment
        if divergence_analysis.has_divergence and divergence_analysis.divergence_strength > 0.6:
            risk_level = "high"
        elif acceleration_analysis.momentum_exhaustion_risk > 0.7:
            risk_level = "elevated"
        elif correlation_analysis.average_correlation < 0.3:
            risk_level = "moderate"
        else:
            risk_level = "normal"
        
        # Primary signal generation
        if signal_strength > 0.7 and dominant_direction != 'neutral':
            primary_signal = f"strong_momentum_{dominant_direction}"
        elif signal_strength > 0.5:
            primary_signal = f"moderate_momentum_{dominant_direction}"
        elif divergence_analysis.has_divergence:
            primary_signal = divergence_analysis.get_divergence_signal()
        else:
            primary_signal = "weak_momentum_neutral"
        
        return {
            'primary': primary_signal,
            'strength': float(signal_strength),
            'risk_level': risk_level
        }
    
    def _calculate_confidence_quality(self,
                                    timeframe_metrics: Dict[MomentumTimeframe, TimeframeMomentumMetrics],
                                    divergence_analysis: MomentumDivergenceAnalysis,
                                    correlation_analysis: CrossTimeframeCorrelation) -> Dict[str, float]:
        """Calculate confidence and quality scores"""
        
        # Aggregate confidence from timeframe metrics
        timeframe_confidences = [metrics.signal_confidence for metrics in timeframe_metrics.values()]
        avg_timeframe_confidence = float(np.mean(timeframe_confidences)) if timeframe_confidences else 0.5
        
        # Quality based on correlation and consistency
        correlation_quality = correlation_analysis.average_correlation
        alignment_quality = correlation_analysis.alignment_score
        
        # Overall confidence
        analysis_confidence = (avg_timeframe_confidence * 0.4 +
                             correlation_quality * 0.3 +
                             alignment_quality * 0.3)
        
        # Signal quality
        signal_quality = (correlation_quality * 0.5 + alignment_quality * 0.5)
        
        # Data reliability
        data_completeness = [metrics.data_completeness for metrics in timeframe_metrics.values()]
        data_reliability = float(np.mean(data_completeness)) if data_completeness else 0.5
        
        return {
            'confidence': float(analysis_confidence),
            'quality': float(signal_quality),
            'reliability': data_reliability
        }
    
    def _calculate_divergence_strength(self, iv_momentum: float, price_momentum: float) -> float:
        """Calculate divergence strength between IV and price momentum"""
        
        if abs(iv_momentum) < 0.01 or abs(price_momentum) < 0.01:
            return 0.0  # No meaningful momentum
        
        # Normalize momentums
        iv_direction = 1 if iv_momentum > 0 else -1
        price_direction = 1 if price_momentum > 0 else -1
        
        # Divergence exists when directions are opposite
        if iv_direction != price_direction:
            # Strength based on magnitude
            strength = (abs(iv_momentum) + abs(price_momentum)) / 2
            return float(min(1.0, strength * 5))  # Amplify for signal
        
        return 0.0
    
    def _calculate_divergence_significance(self, strength: float, supporting_count: int) -> float:
        """Calculate divergence significance"""
        
        base_significance = strength
        support_factor = min(1.0, (supporting_count + 1) / 3)  # Max 3 supporting timeframes
        
        return float(base_significance * support_factor)
    
    def _calculate_signal_reliability(self, strength: float, supporting: int, conflicting: int) -> float:
        """Calculate signal reliability"""
        
        if supporting + conflicting == 0:
            return strength
        
        support_ratio = supporting / (supporting + conflicting)
        reliability = strength * support_ratio
        
        return float(max(0.1, min(1.0, reliability)))
    
    def _classify_correlation_regime(self, avg_correlation: float) -> str:
        """Classify correlation regime"""
        
        if avg_correlation > 0.7:
            return "high_correlation"
        elif avg_correlation > 0.4:
            return "moderate_correlation"
        elif avg_correlation > 0.1:
            return "low_correlation"
        else:
            return "no_correlation"
    
    # Default return methods
    def _get_default_momentum_result(self) -> ComprehensiveMomentumResult:
        """Get default momentum result when analysis fails"""
        
        default_timeframe_metrics = {
            tf: self._get_default_timeframe_metrics(tf) for tf in MomentumTimeframe
        }
        
        return ComprehensiveMomentumResult(
            timeframe_metrics=default_timeframe_metrics,
            divergence_analysis=self._get_default_divergence_analysis(),
            correlation_analysis=self._get_default_correlation_analysis(),
            acceleration_analysis=self._get_default_acceleration_analysis(),
            dominant_momentum_timeframe=MomentumTimeframe.FIFTEEN_MIN,
            overall_momentum_direction="neutral",
            momentum_regime_classification=MomentumRegime.SIDEWAYS,
            analysis_confidence=0.5,
            signal_quality_score=0.5,
            data_reliability=0.5,
            primary_signal="weak_momentum_neutral",
            signal_strength=0.3,
            risk_level="normal",
            total_processing_time_ms=0.0,
            components_analyzed=["default"]
        )
    
    def _get_default_timeframe_metrics(self, timeframe: MomentumTimeframe) -> TimeframeMomentumMetrics:
        """Get default timeframe metrics when calculation fails"""
        
        return TimeframeMomentumMetrics(
            timeframe=timeframe,
            lookback_periods=10,
            momentum_value=0.0,
            momentum_percentile=50.0,
            momentum_strength="weak",
            momentum_direction="neutral",
            momentum_acceleration=0.0,
            momentum_consistency=0.5,
            signal_strength=0.3,
            signal_confidence=0.5,
            noise_level=0.5,
            momentum_regime=MomentumRegime.SIDEWAYS,
            regime_confidence=0.5,
            relative_momentum=0.0,
            momentum_rank=2,
            data_completeness=0.5,
            calculation_time_ms=0.0
        )
    
    def _get_default_divergence_analysis(self) -> MomentumDivergenceAnalysis:
        """Get default divergence analysis when detection fails"""
        
        return MomentumDivergenceAnalysis(
            has_divergence=False,
            divergence_type="none",
            divergence_strength=0.0,
            price_momentum_direction="neutral",
            iv_percentile_momentum_direction="neutral",
            divergence_duration=0,
            divergence_significance=0.0,
            reversal_probability=0.0,
            signal_reliability=0.5,
            primary_divergence_timeframe=MomentumTimeframe.FIFTEEN_MIN,
            supporting_timeframes=[],
            conflicting_timeframes=[],
            divergence_risk_score=0.1,
            false_signal_probability=0.5
        )
    
    def _get_default_correlation_analysis(self) -> CrossTimeframeCorrelation:
        """Get default correlation analysis when analysis fails"""
        
        # Default correlation matrix with neutral correlations
        correlation_matrix = {}
        for tf1 in MomentumTimeframe:
            correlation_matrix[tf1] = {}
            for tf2 in MomentumTimeframe:
                if tf1 == tf2:
                    correlation_matrix[tf1][tf2] = 1.0
                else:
                    correlation_matrix[tf1][tf2] = 0.3  # Weak positive correlation
        
        return CrossTimeframeCorrelation(
            correlation_matrix=correlation_matrix,
            average_correlation=0.3,
            correlation_stability=0.5,
            correlation_regime="low_correlation",
            strongest_correlation=(MomentumTimeframe.FIVE_MIN, MomentumTimeframe.FIFTEEN_MIN, 0.5),
            weakest_correlation=(MomentumTimeframe.FIVE_MIN, MomentumTimeframe.ONE_HOUR, 0.1),
            aligned_timeframes=[MomentumTimeframe.FIVE_MIN, MomentumTimeframe.FIFTEEN_MIN],
            divergent_timeframes=[MomentumTimeframe.THIRTY_MIN, MomentumTimeframe.ONE_HOUR],
            alignment_score=0.5,
            consensus_direction="neutral",
            consensus_strength=0.5,
            consensus_reliability=0.5
        )
    
    def _get_default_acceleration_analysis(self) -> MomentumAccelerationAnalysis:
        """Get default acceleration analysis when calculation fails"""
        
        return MomentumAccelerationAnalysis(
            momentum_acceleration=0.0,
            acceleration_trend="stable",
            acceleration_strength=0.0,
            first_derivative=0.0,
            second_derivative=0.0,
            momentum_curvature=0.0,
            regime_change_probability=0.3,
            predicted_momentum_direction="neutral",
            prediction_confidence=0.5,
            inflection_point_proximity=0.2,
            momentum_exhaustion_risk=0.3,
            reversal_signal_strength=0.2
        )