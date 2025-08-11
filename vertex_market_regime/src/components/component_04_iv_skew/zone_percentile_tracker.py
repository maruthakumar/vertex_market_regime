"""
Zone-Wise IV Percentile Analysis System - Component 4 Enhancement

Advanced zone-specific percentile analysis using zone_name column 
(MID_MORN/LUNCH/AFTERNOON/CLOSE) with intraday pattern analysis,
zone transition tracking, cross-zone correlation analysis, and 
zone-specific regime classification for institutional-grade intraday
IV percentile pattern recognition.

This module provides sophisticated intraday zone analysis with
comprehensive percentile tracking across all trading zones.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time
import logging
import time
from enum import Enum

from .iv_percentile_analyzer import IVPercentileData
from .historical_percentile_database import HistoricalPercentileDatabase, PercentileDistribution


class TradingZone(Enum):
    """Trading zone classification per production schema"""
    MID_MORN = "MID_MORN"     # Mid-morning trading
    LUNCH = "LUNCH"           # Lunch time trading
    AFTERNOON = "AFTERNOON"   # Afternoon trading
    CLOSE = "CLOSE"          # Close trading


@dataclass
class ZonePercentileMetrics:
    """Zone-specific percentile metrics"""
    
    # Zone identification
    zone_name: str
    zone_id: int
    zone_enum: TradingZone
    
    # Core zone percentiles
    zone_iv_percentile: float
    zone_atm_percentile: float
    zone_surface_percentile: float
    
    # Intraday position analysis
    session_position: float  # 0.0 = start, 1.0 = end
    time_weight: float       # Importance weight based on time
    
    # Zone-specific regime classification
    zone_regime: str
    zone_regime_confidence: float
    zone_volatility_level: str
    
    # Historical context
    zone_historical_rank: int
    zone_percentile_band: str
    data_sufficiency: float
    
    # Quality metrics
    calculation_confidence: float
    zone_data_quality: float


@dataclass
class ZoneTransitionAnalysis:
    """Zone transition analysis results"""
    
    # Transition information
    from_zone: str
    to_zone: str
    transition_time: datetime
    
    # IV change metrics
    iv_change_absolute: float
    iv_change_percentage: float
    percentile_change: float
    
    # Transition characteristics
    transition_magnitude: str  # small, medium, large
    transition_direction: str  # increasing, decreasing, stable
    transition_significance: float
    
    # Pattern recognition
    is_typical_pattern: bool
    pattern_deviation: float
    historical_frequency: float
    
    # Risk assessment
    transition_risk_score: float
    volatility_impact: float


@dataclass
class CrossZoneCorrelationResult:
    """Cross-zone correlation analysis result"""
    
    # Correlation matrix
    zone_correlation_matrix: Dict[str, Dict[str, float]]
    
    # Dominant correlations
    strongest_correlation: Tuple[str, str, float]
    weakest_correlation: Tuple[str, str, float]
    
    # Regime synchronization
    regime_synchronization_score: float
    divergence_zones: List[str]
    convergence_zones: List[str]
    
    # Statistical significance
    correlation_significance: Dict[str, float]
    sample_size_adequacy: float
    
    def get_zone_relationship(self, zone1: str, zone2: str) -> str:
        """Get relationship classification between two zones"""
        
        if zone1 not in self.zone_correlation_matrix or zone2 not in self.zone_correlation_matrix[zone1]:
            return "insufficient_data"
        
        correlation = self.zone_correlation_matrix[zone1][zone2]
        
        if correlation >= 0.7:
            return "strong_positive"
        elif correlation >= 0.3:
            return "moderate_positive"
        elif correlation >= -0.3:
            return "weak_correlation"
        elif correlation >= -0.7:
            return "moderate_negative"
        else:
            return "strong_negative"


@dataclass
class IntradayPatternAnalysis:
    """Intraday IV percentile pattern analysis"""
    
    # Pattern identification
    pattern_type: str
    pattern_strength: float
    pattern_persistence: float
    
    # Zone progression analysis
    zone_progression: List[Tuple[str, float, float]]  # (zone, percentile, timestamp)
    peak_zone: str
    trough_zone: str
    
    # Volatility clustering
    clustering_zones: List[str]
    clustering_strength: float
    
    # Anomaly detection
    anomalous_zones: List[str]
    anomaly_severity: float
    
    # Predictive metrics
    next_zone_prediction: str
    confidence_level: float
    
    def get_pattern_classification(self) -> str:
        """Get overall pattern classification"""
        
        if self.pattern_strength >= 0.8:
            return f"strong_{self.pattern_type}"
        elif self.pattern_strength >= 0.6:
            return f"moderate_{self.pattern_type}"
        elif self.pattern_strength >= 0.4:
            return f"weak_{self.pattern_type}"
        else:
            return "no_clear_pattern"


class ZonePercentileTracker:
    """
    Advanced Zone-Wise IV Percentile Analysis System with comprehensive
    intraday pattern recognition and zone transition tracking.
    
    Features:
    - Zone-specific percentile calculation (MID_MORN/LUNCH/AFTERNOON/CLOSE)
    - Intraday IV pattern analysis and regime detection
    - Zone transition tracking with historical pattern matching
    - Cross-zone correlation analysis for regime synchronization
    - Zone-specific regime classification with confidence scoring
    - Anomaly detection for unusual intraday IV behavior
    - Predictive zone progression analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # Zone configuration per production schema
        self.valid_zones = {
            TradingZone.MID_MORN: {"id": 2, "time_weight": 0.8, "session_pos": 0.2},
            TradingZone.LUNCH: {"id": 3, "time_weight": 0.6, "session_pos": 0.5},
            TradingZone.AFTERNOON: {"id": 4, "time_weight": 0.9, "session_pos": 0.8},
            TradingZone.CLOSE: {"id": 5, "time_weight": 1.0, "session_pos": 1.0}
        }
        
        # Analysis configuration
        self.min_zone_data_points = config.get('min_zone_data_points', 25)
        self.transition_threshold = config.get('transition_threshold', 5.0)  # Percentile points
        self.correlation_window = config.get('correlation_window_days', 30)
        
        # Pattern detection configuration
        self.pattern_detection_enabled = config.get('enable_pattern_detection', True)
        self.anomaly_detection_threshold = config.get('anomaly_threshold', 2.0)  # Standard deviations
        
        # Performance configuration
        self.processing_budget_ms = config.get('zone_processing_budget_ms', 80)
        
        self.logger.info("Zone Percentile Tracker initialized with 4-zone analysis capability")
    
    def analyze_zone_specific_percentiles(self, iv_data: IVPercentileData,
                                        historical_db: HistoricalPercentileDatabase) -> ZonePercentileMetrics:
        """
        Calculate zone-specific IV percentiles using zone_name column
        for intraday pattern analysis
        
        Args:
            iv_data: IV percentile data with zone information
            historical_db: Historical percentile database
            
        Returns:
            ZonePercentileMetrics with comprehensive zone analysis
        """
        start_time = time.time()
        
        try:
            current_zone = iv_data.zone_name
            zone_id = iv_data.zone_id
            
            # Validate zone
            zone_enum = self._validate_zone(current_zone)
            if zone_enum is None:
                self.logger.warning(f"Invalid zone: {current_zone}, using CLOSE as default")
                zone_enum = TradingZone.CLOSE
                current_zone = "CLOSE"
            
            # Get historical zone distribution
            zone_distribution = historical_db.get_zone_percentile_distribution(current_zone)
            
            if zone_distribution is None:
                self.logger.warning(f"No historical distribution for zone {current_zone}")
                return self._get_default_zone_metrics(current_zone, zone_id, zone_enum)
            
            # Calculate current IV metrics for zone analysis
            current_atm_iv = self._calculate_atm_iv(iv_data)
            current_surface_iv = self._calculate_surface_iv(iv_data)
            
            # Calculate zone-specific percentiles
            zone_iv_percentile = zone_distribution.calculate_percentile_rank(current_atm_iv)
            zone_atm_percentile = zone_distribution.calculate_percentile_rank(current_atm_iv)
            zone_surface_percentile = zone_distribution.calculate_percentile_rank(current_surface_iv)
            
            # Intraday position analysis
            zone_config = self.valid_zones[zone_enum]
            session_position = zone_config["session_pos"]
            time_weight = zone_config["time_weight"]
            
            # Zone regime classification
            zone_regime_analysis = self._classify_zone_regime(
                zone_iv_percentile, current_zone, zone_distribution
            )
            
            # Historical context
            zone_historical_rank = int(zone_distribution.count * (zone_iv_percentile / 100))
            zone_percentile_band = self._classify_zone_percentile_band(zone_iv_percentile)
            
            # Data sufficiency and quality
            data_sufficiency = min(1.0, zone_distribution.count / self.min_zone_data_points)
            zone_data_quality = iv_data.data_completeness
            calculation_confidence = self._calculate_zone_confidence(
                zone_distribution, data_sufficiency, zone_data_quality
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            self.logger.debug(f"Zone analysis completed for {current_zone}: "
                            f"Percentile={zone_iv_percentile:.1f}%, "
                            f"Regime={zone_regime_analysis['regime']}, "
                            f"Time={processing_time:.2f}ms")
            
            return ZonePercentileMetrics(
                zone_name=current_zone,
                zone_id=zone_id,
                zone_enum=zone_enum,
                zone_iv_percentile=zone_iv_percentile,
                zone_atm_percentile=zone_atm_percentile,
                zone_surface_percentile=zone_surface_percentile,
                session_position=session_position,
                time_weight=time_weight,
                zone_regime=zone_regime_analysis['regime'],
                zone_regime_confidence=zone_regime_analysis['confidence'],
                zone_volatility_level=zone_regime_analysis['volatility_level'],
                zone_historical_rank=zone_historical_rank,
                zone_percentile_band=zone_percentile_band,
                data_sufficiency=data_sufficiency,
                calculation_confidence=calculation_confidence,
                zone_data_quality=zone_data_quality
            )
            
        except Exception as e:
            self.logger.error(f"Zone-specific percentile analysis failed: {e}")
            zone_enum = self._validate_zone(iv_data.zone_name) or TradingZone.CLOSE
            return self._get_default_zone_metrics(iv_data.zone_name, iv_data.zone_id, zone_enum)
    
    def analyze_zone_transitions(self, current_zone_metrics: ZonePercentileMetrics,
                               previous_zone_data: Optional[Dict[str, Any]],
                               historical_db: HistoricalPercentileDatabase) -> ZoneTransitionAnalysis:
        """
        Monitor IV percentile changes across trading zones for regime shift detection
        
        Args:
            current_zone_metrics: Current zone percentile metrics
            previous_zone_data: Previous zone data for transition analysis
            historical_db: Historical database for pattern matching
            
        Returns:
            ZoneTransitionAnalysis with transition metrics
        """
        start_time = time.time()
        
        try:
            if previous_zone_data is None:
                return self._get_default_transition_analysis(current_zone_metrics.zone_name)
            
            # Extract previous zone information
            prev_zone = previous_zone_data.get('zone_name', 'UNKNOWN')
            prev_percentile = previous_zone_data.get('zone_iv_percentile', 50.0)
            prev_atm_iv = previous_zone_data.get('atm_iv', 0.0)
            current_atm_iv = self._calculate_current_atm_iv(current_zone_metrics)
            
            # Calculate transition metrics
            iv_change_absolute = current_atm_iv - prev_atm_iv
            iv_change_percentage = (iv_change_absolute / prev_atm_iv * 100) if prev_atm_iv > 0 else 0.0
            percentile_change = current_zone_metrics.zone_iv_percentile - prev_percentile
            
            # Classify transition characteristics
            transition_magnitude = self._classify_transition_magnitude(abs(percentile_change))
            transition_direction = self._classify_transition_direction(percentile_change)
            
            # Calculate transition significance
            transition_significance = self._calculate_transition_significance(
                percentile_change, current_zone_metrics, prev_zone
            )
            
            # Pattern recognition
            pattern_analysis = self._analyze_transition_pattern(
                prev_zone, current_zone_metrics.zone_name, percentile_change, historical_db
            )
            
            # Risk assessment
            risk_analysis = self._assess_transition_risk(
                transition_magnitude, transition_direction, pattern_analysis
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            self.logger.debug(f"Zone transition analyzed: {prev_zone} -> {current_zone_metrics.zone_name}, "
                            f"Change={percentile_change:.2f}%, "
                            f"Magnitude={transition_magnitude}")
            
            return ZoneTransitionAnalysis(
                from_zone=prev_zone,
                to_zone=current_zone_metrics.zone_name,
                transition_time=datetime.utcnow(),
                iv_change_absolute=iv_change_absolute,
                iv_change_percentage=iv_change_percentage,
                percentile_change=percentile_change,
                transition_magnitude=transition_magnitude,
                transition_direction=transition_direction,
                transition_significance=transition_significance,
                is_typical_pattern=pattern_analysis['is_typical'],
                pattern_deviation=pattern_analysis['deviation'],
                historical_frequency=pattern_analysis['frequency'],
                transition_risk_score=risk_analysis['risk_score'],
                volatility_impact=risk_analysis['volatility_impact']
            )
            
        except Exception as e:
            self.logger.error(f"Zone transition analysis failed: {e}")
            return self._get_default_transition_analysis(current_zone_metrics.zone_name)
    
    def analyze_cross_zone_correlations(self, zone_history: List[ZonePercentileMetrics],
                                      historical_db: HistoricalPercentileDatabase) -> CrossZoneCorrelationResult:
        """
        Analyze IV percentile correlations between different trading zones
        for regime synchronization detection
        
        Args:
            zone_history: Historical zone percentile data
            historical_db: Historical database for extended correlation analysis
            
        Returns:
            CrossZoneCorrelationResult with correlation analysis
        """
        start_time = time.time()
        
        try:
            # Build correlation matrix
            correlation_matrix = self._build_zone_correlation_matrix(zone_history)
            
            # Find strongest and weakest correlations
            correlations = []
            for zone1, zone1_correlations in correlation_matrix.items():
                for zone2, correlation in zone1_correlations.items():
                    if zone1 != zone2:
                        correlations.append((zone1, zone2, correlation))
            
            if correlations:
                strongest_correlation = max(correlations, key=lambda x: abs(x[2]))
                weakest_correlation = min(correlations, key=lambda x: abs(x[2]))
            else:
                strongest_correlation = ("NONE", "NONE", 0.0)
                weakest_correlation = ("NONE", "NONE", 0.0)
            
            # Regime synchronization analysis
            sync_analysis = self._analyze_regime_synchronization(correlation_matrix, zone_history)
            
            # Statistical significance assessment
            significance_analysis = self._assess_correlation_significance(correlation_matrix, zone_history)
            
            processing_time = (time.time() - start_time) * 1000
            
            self.logger.debug(f"Cross-zone correlation analysis completed: "
                            f"Strongest={strongest_correlation[0]}-{strongest_correlation[1]} ({strongest_correlation[2]:.2f}), "
                            f"Sync={sync_analysis['synchronization_score']:.2f}")
            
            return CrossZoneCorrelationResult(
                zone_correlation_matrix=correlation_matrix,
                strongest_correlation=strongest_correlation,
                weakest_correlation=weakest_correlation,
                regime_synchronization_score=sync_analysis['synchronization_score'],
                divergence_zones=sync_analysis['divergence_zones'],
                convergence_zones=sync_analysis['convergence_zones'],
                correlation_significance=significance_analysis['significance_scores'],
                sample_size_adequacy=significance_analysis['sample_adequacy']
            )
            
        except Exception as e:
            self.logger.error(f"Cross-zone correlation analysis failed: {e}")
            return self._get_default_correlation_result()
    
    def analyze_intraday_patterns(self, zone_sequence: List[ZonePercentileMetrics]) -> IntradayPatternAnalysis:
        """
        Analyze intraday IV percentile patterns across all trading zones
        
        Args:
            zone_sequence: Sequence of zone percentile metrics throughout the day
            
        Returns:
            IntradayPatternAnalysis with pattern recognition results
        """
        start_time = time.time()
        
        try:
            if len(zone_sequence) < 2:
                return self._get_default_pattern_analysis()
            
            # Extract zone progression
            zone_progression = []
            for i, zone_metrics in enumerate(zone_sequence):
                zone_progression.append((
                    zone_metrics.zone_name,
                    zone_metrics.zone_iv_percentile,
                    float(i)  # Simplified timestamp
                ))
            
            # Identify pattern type
            pattern_analysis = self._identify_intraday_pattern(zone_progression)
            
            # Find peak and trough zones
            percentiles = [z.zone_iv_percentile for z in zone_sequence]
            peak_idx = np.argmax(percentiles)
            trough_idx = np.argmin(percentiles)
            
            peak_zone = zone_sequence[peak_idx].zone_name
            trough_zone = zone_sequence[trough_idx].zone_name
            
            # Volatility clustering analysis
            clustering_analysis = self._analyze_volatility_clustering(zone_sequence)
            
            # Anomaly detection
            anomaly_analysis = self._detect_zone_anomalies(zone_sequence)
            
            # Predictive analysis
            prediction_analysis = self._predict_next_zone_behavior(zone_sequence)
            
            processing_time = (time.time() - start_time) * 1000
            
            self.logger.debug(f"Intraday pattern analysis completed: "
                            f"Pattern={pattern_analysis['type']}, "
                            f"Strength={pattern_analysis['strength']:.2f}")
            
            return IntradayPatternAnalysis(
                pattern_type=pattern_analysis['type'],
                pattern_strength=pattern_analysis['strength'],
                pattern_persistence=pattern_analysis['persistence'],
                zone_progression=zone_progression,
                peak_zone=peak_zone,
                trough_zone=trough_zone,
                clustering_zones=clustering_analysis['zones'],
                clustering_strength=clustering_analysis['strength'],
                anomalous_zones=anomaly_analysis['zones'],
                anomaly_severity=anomaly_analysis['severity'],
                next_zone_prediction=prediction_analysis['zone'],
                confidence_level=prediction_analysis['confidence']
            )
            
        except Exception as e:
            self.logger.error(f"Intraday pattern analysis failed: {e}")
            return self._get_default_pattern_analysis()
    
    def calculate_zone_regime_classification(self, zone_metrics: ZonePercentileMetrics) -> Dict[str, Any]:
        """
        Implement zone-specific regime classification per DTE with 7-level system
        (Extremely Low, Very Low, Low, Normal, High, Very High, Extremely High)
        
        Args:
            zone_metrics: Zone percentile metrics
            
        Returns:
            Dictionary with detailed regime classification
        """
        try:
            percentile = zone_metrics.zone_iv_percentile
            zone_name = zone_metrics.zone_name
            
            # Zone-specific thresholds (different zones have different sensitivities)
            zone_thresholds = self._get_zone_specific_thresholds(zone_name)
            
            # 7-level regime classification
            if percentile >= zone_thresholds['extremely_high']:
                regime_level = "extremely_high"
                risk_level = "very_high"
                action_recommendation = "defensive"
            elif percentile >= zone_thresholds['very_high']:
                regime_level = "very_high" 
                risk_level = "high"
                action_recommendation = "cautious"
            elif percentile >= zone_thresholds['high']:
                regime_level = "high"
                risk_level = "elevated"
                action_recommendation = "selective"
            elif percentile >= zone_thresholds['normal_high']:
                regime_level = "normal"
                risk_level = "normal"
                action_recommendation = "neutral"
            elif percentile >= zone_thresholds['low']:
                regime_level = "low"
                risk_level = "below_normal"
                action_recommendation = "opportunistic"
            elif percentile >= zone_thresholds['very_low']:
                regime_level = "very_low"
                risk_level = "low"
                action_recommendation = "aggressive"
            else:
                regime_level = "extremely_low"
                risk_level = "very_low"
                action_recommendation = "maximum_opportunity"
            
            # Confidence calculation based on zone characteristics
            base_confidence = zone_metrics.calculation_confidence
            zone_weight = zone_metrics.time_weight
            data_quality = zone_metrics.zone_data_quality
            
            regime_confidence = (base_confidence * 0.5 + zone_weight * 0.3 + data_quality * 0.2)
            
            # Regime persistence estimation
            persistence_score = self._calculate_regime_persistence(zone_metrics, regime_level)
            
            return {
                'regime_level': regime_level,
                'regime_confidence': float(regime_confidence),
                'risk_level': risk_level,
                'action_recommendation': action_recommendation,
                'persistence_score': persistence_score,
                'zone_specific_factors': {
                    'zone_sensitivity': zone_thresholds.get('sensitivity_factor', 1.0),
                    'time_weight_influence': zone_weight,
                    'intraday_position_factor': zone_metrics.session_position
                },
                'classification_metadata': {
                    'percentile_used': percentile,
                    'zone_name': zone_name,
                    'threshold_set': zone_thresholds,
                    'data_sufficiency': zone_metrics.data_sufficiency
                }
            }
            
        except Exception as e:
            self.logger.error(f"Zone regime classification failed: {e}")
            return self._get_default_regime_classification()
    
    def _validate_zone(self, zone_name: str) -> Optional[TradingZone]:
        """Validate and convert zone name to enum"""
        
        zone_mapping = {
            'MID_MORN': TradingZone.MID_MORN,
            'LUNCH': TradingZone.LUNCH,
            'AFTERNOON': TradingZone.AFTERNOON,
            'CLOSE': TradingZone.CLOSE
        }
        
        return zone_mapping.get(zone_name)
    
    def _calculate_atm_iv(self, iv_data: IVPercentileData) -> float:
        """Calculate ATM IV from IV data"""
        
        atm_idx = np.argmin(np.abs(iv_data.strikes - iv_data.atm_strike))
        atm_ce_iv = iv_data.ce_iv[atm_idx] if not np.isnan(iv_data.ce_iv[atm_idx]) else 0.0
        atm_pe_iv = iv_data.pe_iv[atm_idx] if not np.isnan(iv_data.pe_iv[atm_idx]) else 0.0
        
        return float((atm_ce_iv + atm_pe_iv) / 2) if (atm_ce_iv > 0 or atm_pe_iv > 0) else 0.0
    
    def _calculate_surface_iv(self, iv_data: IVPercentileData) -> float:
        """Calculate surface average IV"""
        
        valid_ivs = np.concatenate([
            iv_data.ce_iv[~np.isnan(iv_data.ce_iv) & (iv_data.ce_iv > 0)],
            iv_data.pe_iv[~np.isnan(iv_data.pe_iv) & (iv_data.pe_iv > 0)]
        ])
        
        return float(np.mean(valid_ivs)) if len(valid_ivs) > 0 else 0.0
    
    def _classify_zone_regime(self, percentile: float, zone_name: str, 
                            distribution: PercentileDistribution) -> Dict[str, Any]:
        """Classify zone-specific regime with confidence"""
        
        # Base regime classification
        if percentile >= 90:
            regime = "extremely_high_vol"
            volatility_level = "extreme"
        elif percentile >= 75:
            regime = "high_vol"
            volatility_level = "high"
        elif percentile >= 60:
            regime = "elevated_vol"
            volatility_level = "elevated"
        elif percentile >= 40:
            regime = "normal_vol"
            volatility_level = "normal"
        elif percentile >= 25:
            regime = "low_vol"
            volatility_level = "low"
        elif percentile >= 10:
            regime = "very_low_vol"
            volatility_level = "very_low"
        else:
            regime = "extremely_low_vol"
            volatility_level = "extreme_low"
        
        # Zone-specific adjustments
        zone_factor = self._get_zone_regime_factor(zone_name)
        
        # Confidence based on distribution quality
        confidence = min(1.0, distribution.count / 50) * zone_factor
        
        return {
            'regime': regime,
            'confidence': float(confidence),
            'volatility_level': volatility_level,
            'zone_factor': zone_factor
        }
    
    def _classify_zone_percentile_band(self, percentile: float) -> str:
        """Classify percentile into bands"""
        
        if percentile >= 95:
            return "extreme_high"
        elif percentile >= 85:
            return "very_high"
        elif percentile >= 70:
            return "high"
        elif percentile >= 55:
            return "above_average"
        elif percentile >= 45:
            return "average"
        elif percentile >= 30:
            return "below_average"
        elif percentile >= 15:
            return "low"
        elif percentile >= 5:
            return "very_low"
        else:
            return "extreme_low"
    
    def _calculate_zone_confidence(self, distribution: PercentileDistribution,
                                 data_sufficiency: float, data_quality: float) -> float:
        """Calculate zone-specific confidence score"""
        
        factors = [data_sufficiency, data_quality]
        
        # Add distribution quality factor
        if distribution.std > 0 and distribution.mean > 0:
            cv = distribution.std / distribution.mean
            quality_factor = 1.0 / (1.0 + cv)
            factors.append(quality_factor)
        
        return float(np.mean(factors))
    
    def _calculate_current_atm_iv(self, zone_metrics: ZonePercentileMetrics) -> float:
        """Extract current ATM IV from zone metrics (simplified)"""
        # This would normally extract from the underlying IV data
        # For now, use a derived estimate from percentile
        return zone_metrics.zone_iv_percentile * 0.5  # Simplified approximation
    
    def _classify_transition_magnitude(self, percentile_change: float) -> str:
        """Classify transition magnitude"""
        
        abs_change = abs(percentile_change)
        
        if abs_change >= 15:
            return "large"
        elif abs_change >= 7:
            return "medium"
        elif abs_change >= 3:
            return "small"
        else:
            return "minimal"
    
    def _classify_transition_direction(self, percentile_change: float) -> str:
        """Classify transition direction"""
        
        if percentile_change > 2:
            return "increasing"
        elif percentile_change < -2:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_transition_significance(self, percentile_change: float,
                                         current_metrics: ZonePercentileMetrics,
                                         prev_zone: str) -> float:
        """Calculate statistical significance of transition"""
        
        # Base significance from magnitude
        magnitude_sig = min(1.0, abs(percentile_change) / 20)
        
        # Zone context significance
        zone_factor = self._get_zone_transition_factor(prev_zone, current_metrics.zone_name)
        
        # Data quality impact
        quality_factor = current_metrics.calculation_confidence
        
        return float((magnitude_sig * 0.5 + zone_factor * 0.3 + quality_factor * 0.2))
    
    def _analyze_transition_pattern(self, from_zone: str, to_zone: str, change: float,
                                  historical_db: HistoricalPercentileDatabase) -> Dict[str, Any]:
        """Analyze transition pattern against historical norms"""
        
        # Simplified pattern analysis
        transition_key = f"{from_zone}_to_{to_zone}"
        
        # Get historical frequency (would query database in production)
        historical_frequency = self._get_historical_transition_frequency(transition_key)
        
        # Calculate deviation from typical pattern
        typical_change = self._get_typical_transition_change(transition_key)
        deviation = abs(change - typical_change) / (abs(typical_change) + 1e-6)
        
        is_typical = deviation < 1.0  # Within 1 standard deviation
        
        return {
            'is_typical': is_typical,
            'deviation': float(deviation),
            'frequency': historical_frequency,
            'typical_change': typical_change
        }
    
    def _assess_transition_risk(self, magnitude: str, direction: str, pattern: Dict[str, Any]) -> Dict[str, float]:
        """Assess risk associated with zone transition"""
        
        # Risk scoring based on transition characteristics
        magnitude_risk = {'minimal': 0.1, 'small': 0.3, 'medium': 0.6, 'large': 0.9}.get(magnitude, 0.5)
        
        direction_risk = {'stable': 0.2, 'increasing': 0.4, 'decreasing': 0.6}.get(direction, 0.4)
        
        pattern_risk = 0.8 if not pattern['is_typical'] else 0.3
        
        # Combined risk score
        risk_score = (magnitude_risk * 0.4 + direction_risk * 0.3 + pattern_risk * 0.3)
        
        # Volatility impact assessment
        volatility_impact = magnitude_risk * 0.7  # Simplified
        
        return {
            'risk_score': float(risk_score),
            'volatility_impact': float(volatility_impact)
        }
    
    def _build_zone_correlation_matrix(self, zone_history: List[ZonePercentileMetrics]) -> Dict[str, Dict[str, float]]:
        """Build correlation matrix between zones"""
        
        # Group data by zone
        zone_data = {}
        for zone_metric in zone_history:
            zone_name = zone_metric.zone_name
            if zone_name not in zone_data:
                zone_data[zone_name] = []
            zone_data[zone_name].append(zone_metric.zone_iv_percentile)
        
        # Calculate correlations
        correlation_matrix = {}
        zone_names = list(zone_data.keys())
        
        for zone1 in zone_names:
            correlation_matrix[zone1] = {}
            for zone2 in zone_names:
                if zone1 == zone2:
                    correlation_matrix[zone1][zone2] = 1.0
                elif len(zone_data[zone1]) >= 3 and len(zone_data[zone2]) >= 3:
                    # Calculate correlation if sufficient data
                    try:
                        corr = np.corrcoef(zone_data[zone1][:len(zone_data[zone2])], 
                                         zone_data[zone2][:len(zone_data[zone1])])[0, 1]
                        correlation_matrix[zone1][zone2] = float(corr) if not np.isnan(corr) else 0.0
                    except:
                        correlation_matrix[zone1][zone2] = 0.0
                else:
                    correlation_matrix[zone1][zone2] = 0.0
        
        return correlation_matrix
    
    def _analyze_regime_synchronization(self, correlation_matrix: Dict[str, Dict[str, float]],
                                      zone_history: List[ZonePercentileMetrics]) -> Dict[str, Any]:
        """Analyze regime synchronization across zones"""
        
        # Calculate overall synchronization score
        all_correlations = []
        for zone1_corrs in correlation_matrix.values():
            for zone2, corr in zone1_corrs.items():
                if corr != 1.0:  # Exclude self-correlations
                    all_correlations.append(abs(corr))
        
        sync_score = float(np.mean(all_correlations)) if all_correlations else 0.0
        
        # Identify divergence and convergence zones
        zone_percentiles = {}
        for zone_metric in zone_history[-4:]:  # Last 4 zones
            zone_percentiles[zone_metric.zone_name] = zone_metric.zone_iv_percentile
        
        if len(zone_percentiles) >= 2:
            percentiles = list(zone_percentiles.values())
            std_dev = np.std(percentiles)
            mean_percentile = np.mean(percentiles)
            
            divergence_zones = []
            convergence_zones = []
            
            for zone, percentile in zone_percentiles.items():
                deviation = abs(percentile - mean_percentile)
                if deviation > std_dev:
                    divergence_zones.append(zone)
                else:
                    convergence_zones.append(zone)
        else:
            divergence_zones = []
            convergence_zones = list(zone_percentiles.keys())
        
        return {
            'synchronization_score': sync_score,
            'divergence_zones': divergence_zones,
            'convergence_zones': convergence_zones
        }
    
    def _assess_correlation_significance(self, correlation_matrix: Dict[str, Dict[str, float]],
                                       zone_history: List[ZonePercentileMetrics]) -> Dict[str, Any]:
        """Assess statistical significance of correlations"""
        
        significance_scores = {}
        sample_size = len(zone_history)
        
        for zone1, correlations in correlation_matrix.items():
            for zone2, correlation in correlations.items():
                if zone1 != zone2:
                    # Simplified significance test
                    t_stat = abs(correlation) * np.sqrt((sample_size - 2) / (1 - correlation**2 + 1e-10))
                    # Convert to p-value approximation
                    significance = min(1.0, t_stat / 3.0)  # Simplified
                    significance_scores[f"{zone1}_{zone2}"] = float(significance)
        
        sample_adequacy = min(1.0, sample_size / 30)  # Need at least 30 samples for reliability
        
        return {
            'significance_scores': significance_scores,
            'sample_adequacy': float(sample_adequacy)
        }
    
    def _identify_intraday_pattern(self, zone_progression: List[Tuple[str, float, float]]) -> Dict[str, Any]:
        """Identify intraday IV percentile pattern"""
        
        if len(zone_progression) < 3:
            return {'type': 'insufficient_data', 'strength': 0.0, 'persistence': 0.0}
        
        percentiles = [p[1] for p in zone_progression]
        
        # Pattern identification
        if self._is_trending_up(percentiles):
            pattern_type = "trending_up"
            strength = self._calculate_trend_strength(percentiles, "up")
        elif self._is_trending_down(percentiles):
            pattern_type = "trending_down"
            strength = self._calculate_trend_strength(percentiles, "down")
        elif self._is_u_shaped(percentiles):
            pattern_type = "u_shaped"
            strength = self._calculate_u_shape_strength(percentiles)
        elif self._is_inverted_u(percentiles):
            pattern_type = "inverted_u"
            strength = self._calculate_inverted_u_strength(percentiles)
        else:
            pattern_type = "sideways"
            strength = 1.0 - (np.std(percentiles) / (np.mean(percentiles) + 1e-6))
        
        # Persistence calculation
        persistence = self._calculate_pattern_persistence(percentiles)
        
        return {
            'type': pattern_type,
            'strength': float(max(0.0, min(1.0, strength))),
            'persistence': float(max(0.0, min(1.0, persistence)))
        }
    
    def _analyze_volatility_clustering(self, zone_sequence: List[ZonePercentileMetrics]) -> Dict[str, Any]:
        """Analyze volatility clustering across zones"""
        
        clustering_zones = []
        percentiles = [z.zone_iv_percentile for z in zone_sequence]
        
        if len(percentiles) < 2:
            return {'zones': clustering_zones, 'strength': 0.0}
        
        mean_percentile = np.mean(percentiles)
        std_percentile = np.std(percentiles)
        
        # Identify zones with similar volatility levels (clustering)
        for i, zone_metric in enumerate(zone_sequence):
            if abs(zone_metric.zone_iv_percentile - mean_percentile) < std_percentile * 0.5:
                clustering_zones.append(zone_metric.zone_name)
        
        # Clustering strength based on how many zones are clustered
        clustering_strength = len(clustering_zones) / len(zone_sequence) if zone_sequence else 0.0
        
        return {
            'zones': clustering_zones,
            'strength': float(clustering_strength)
        }
    
    def _detect_zone_anomalies(self, zone_sequence: List[ZonePercentileMetrics]) -> Dict[str, Any]:
        """Detect anomalous zones using statistical methods"""
        
        anomalous_zones = []
        percentiles = [z.zone_iv_percentile for z in zone_sequence]
        
        if len(percentiles) < 3:
            return {'zones': anomalous_zones, 'severity': 0.0}
        
        mean_percentile = np.mean(percentiles)
        std_percentile = np.std(percentiles)
        
        severity_scores = []
        
        for zone_metric in zone_sequence:
            # Z-score anomaly detection
            z_score = abs(zone_metric.zone_iv_percentile - mean_percentile) / (std_percentile + 1e-6)
            
            if z_score > self.anomaly_detection_threshold:
                anomalous_zones.append(zone_metric.zone_name)
                severity_scores.append(z_score)
        
        avg_severity = float(np.mean(severity_scores)) if severity_scores else 0.0
        
        return {
            'zones': anomalous_zones,
            'severity': avg_severity
        }
    
    def _predict_next_zone_behavior(self, zone_sequence: List[ZonePercentileMetrics]) -> Dict[str, Any]:
        """Predict next zone behavior based on pattern"""
        
        if len(zone_sequence) < 2:
            return {'zone': 'UNKNOWN', 'confidence': 0.0}
        
        # Simple prediction based on last zone and pattern
        last_zone = zone_sequence[-1].zone_name
        zone_order = ['MID_MORN', 'LUNCH', 'AFTERNOON', 'CLOSE']
        
        try:
            current_idx = zone_order.index(last_zone)
            if current_idx < len(zone_order) - 1:
                next_zone = zone_order[current_idx + 1]
                confidence = 0.8  # High confidence for sequential progression
            else:
                next_zone = 'END_OF_SESSION'
                confidence = 1.0
        except ValueError:
            next_zone = 'UNKNOWN'
            confidence = 0.0
        
        return {
            'zone': next_zone,
            'confidence': float(confidence)
        }
    
    # Helper methods for pattern recognition
    def _is_trending_up(self, percentiles: List[float]) -> bool:
        """Check if percentiles show upward trend"""
        if len(percentiles) < 3:
            return False
        
        increases = sum(1 for i in range(1, len(percentiles)) if percentiles[i] > percentiles[i-1])
        return increases >= len(percentiles) * 0.6
    
    def _is_trending_down(self, percentiles: List[float]) -> bool:
        """Check if percentiles show downward trend"""
        if len(percentiles) < 3:
            return False
        
        decreases = sum(1 for i in range(1, len(percentiles)) if percentiles[i] < percentiles[i-1])
        return decreases >= len(percentiles) * 0.6
    
    def _is_u_shaped(self, percentiles: List[float]) -> bool:
        """Check if percentiles form U-shape"""
        if len(percentiles) < 3:
            return False
        
        mid_idx = len(percentiles) // 2
        return percentiles[0] > percentiles[mid_idx] and percentiles[-1] > percentiles[mid_idx]
    
    def _is_inverted_u(self, percentiles: List[float]) -> bool:
        """Check if percentiles form inverted U-shape"""
        if len(percentiles) < 3:
            return False
        
        mid_idx = len(percentiles) // 2
        return percentiles[0] < percentiles[mid_idx] and percentiles[-1] < percentiles[mid_idx]
    
    def _calculate_trend_strength(self, percentiles: List[float], direction: str) -> float:
        """Calculate trend strength"""
        if len(percentiles) < 2:
            return 0.0
        
        total_change = percentiles[-1] - percentiles[0]
        max_possible_change = 100  # Maximum percentile change
        
        strength = abs(total_change) / max_possible_change
        return min(1.0, strength * 2)  # Amplify for stronger signal
    
    def _calculate_u_shape_strength(self, percentiles: List[float]) -> float:
        """Calculate U-shape pattern strength"""
        if len(percentiles) < 3:
            return 0.0
        
        mid_idx = len(percentiles) // 2
        dip_depth = (percentiles[0] + percentiles[-1]) / 2 - percentiles[mid_idx]
        
        return min(1.0, abs(dip_depth) / 25)  # Normalize by typical range
    
    def _calculate_inverted_u_strength(self, percentiles: List[float]) -> float:
        """Calculate inverted U-shape pattern strength"""
        if len(percentiles) < 3:
            return 0.0
        
        mid_idx = len(percentiles) // 2
        peak_height = percentiles[mid_idx] - (percentiles[0] + percentiles[-1]) / 2
        
        return min(1.0, abs(peak_height) / 25)  # Normalize by typical range
    
    def _calculate_pattern_persistence(self, percentiles: List[float]) -> float:
        """Calculate pattern persistence score"""
        if len(percentiles) < 3:
            return 0.0
        
        # Measure how consistent the pattern is
        changes = [percentiles[i] - percentiles[i-1] for i in range(1, len(percentiles))]
        consistency = 1.0 - (np.std(changes) / (np.mean(np.abs(changes)) + 1e-6))
        
        return max(0.0, min(1.0, consistency))
    
    def _get_zone_regime_factor(self, zone_name: str) -> float:
        """Get zone-specific regime factor"""
        
        zone_factors = {
            'MID_MORN': 0.8,   # Moderate confidence
            'LUNCH': 0.6,      # Lower confidence (lunch effect)
            'AFTERNOON': 0.9,  # High confidence
            'CLOSE': 1.0       # Highest confidence
        }
        
        return zone_factors.get(zone_name, 0.7)
    
    def _get_zone_specific_thresholds(self, zone_name: str) -> Dict[str, float]:
        """Get zone-specific percentile thresholds for regime classification"""
        
        # Different zones have different sensitivities
        base_thresholds = {
            'MID_MORN': {
                'extremely_high': 92, 'very_high': 82, 'high': 68,
                'normal_high': 45, 'low': 32, 'very_low': 18,
                'sensitivity_factor': 0.9
            },
            'LUNCH': {
                'extremely_high': 88, 'very_high': 78, 'high': 65,
                'normal_high': 45, 'low': 35, 'very_low': 22,
                'sensitivity_factor': 0.7
            },
            'AFTERNOON': {
                'extremely_high': 90, 'very_high': 80, 'high': 70,
                'normal_high': 45, 'low': 30, 'very_low': 20,
                'sensitivity_factor': 1.0
            },
            'CLOSE': {
                'extremely_high': 95, 'very_high': 85, 'high': 72,
                'normal_high': 45, 'low': 28, 'very_low': 15,
                'sensitivity_factor': 1.2
            }
        }
        
        return base_thresholds.get(zone_name, base_thresholds['CLOSE'])
    
    def _calculate_regime_persistence(self, zone_metrics: ZonePercentileMetrics, regime_level: str) -> float:
        """Calculate regime persistence score"""
        
        # Base persistence based on regime level
        persistence_base = {
            'extremely_high': 0.3, 'very_high': 0.4, 'high': 0.6,
            'normal': 0.8, 'low': 0.6, 'very_low': 0.4, 'extremely_low': 0.3
        }
        
        base_score = persistence_base.get(regime_level, 0.5)
        
        # Adjust for zone characteristics
        zone_factor = zone_metrics.time_weight
        confidence_factor = zone_metrics.calculation_confidence
        
        return float(base_score * zone_factor * confidence_factor)
    
    def _get_zone_transition_factor(self, from_zone: str, to_zone: str) -> float:
        """Get transition significance factor between zones"""
        
        # Sequential transitions are less significant
        sequential_transitions = [
            ('MID_MORN', 'LUNCH'), ('LUNCH', 'AFTERNOON'), ('AFTERNOON', 'CLOSE')
        ]
        
        if (from_zone, to_zone) in sequential_transitions:
            return 0.5  # Normal transition
        else:
            return 0.8  # Non-sequential transition (more significant)
    
    def _get_historical_transition_frequency(self, transition_key: str) -> float:
        """Get historical frequency of specific transition (simplified)"""
        
        # Mock historical frequencies
        common_transitions = {
            'MID_MORN_to_LUNCH': 0.9,
            'LUNCH_to_AFTERNOON': 0.9,
            'AFTERNOON_to_CLOSE': 0.9,
            'MID_MORN_to_AFTERNOON': 0.3,
            'MID_MORN_to_CLOSE': 0.1,
            'LUNCH_to_CLOSE': 0.4
        }
        
        return common_transitions.get(transition_key, 0.1)
    
    def _get_typical_transition_change(self, transition_key: str) -> float:
        """Get typical percentile change for transition (simplified)"""
        
        # Mock typical changes
        typical_changes = {
            'MID_MORN_to_LUNCH': -2.0,    # Slight decrease
            'LUNCH_to_AFTERNOON': 1.0,    # Slight increase
            'AFTERNOON_to_CLOSE': 3.0,    # Moderate increase
            'MID_MORN_to_AFTERNOON': 0.5, # Minimal change
            'MID_MORN_to_CLOSE': 2.0,     # Moderate increase
            'LUNCH_to_CLOSE': 4.0         # Larger increase
        }
        
        return typical_changes.get(transition_key, 0.0)
    
    # Default return methods
    def _get_default_zone_metrics(self, zone_name: str, zone_id: int, zone_enum: TradingZone) -> ZonePercentileMetrics:
        """Get default zone metrics when analysis fails"""
        
        return ZonePercentileMetrics(
            zone_name=zone_name,
            zone_id=zone_id,
            zone_enum=zone_enum,
            zone_iv_percentile=50.0,
            zone_atm_percentile=50.0,
            zone_surface_percentile=50.0,
            session_position=0.5,
            time_weight=0.7,
            zone_regime="normal_vol",
            zone_regime_confidence=0.5,
            zone_volatility_level="normal",
            zone_historical_rank=0,
            zone_percentile_band="average",
            data_sufficiency=0.0,
            calculation_confidence=0.5,
            zone_data_quality=0.5
        )
    
    def _get_default_transition_analysis(self, zone_name: str) -> ZoneTransitionAnalysis:
        """Get default transition analysis when data is insufficient"""
        
        return ZoneTransitionAnalysis(
            from_zone="UNKNOWN",
            to_zone=zone_name,
            transition_time=datetime.utcnow(),
            iv_change_absolute=0.0,
            iv_change_percentage=0.0,
            percentile_change=0.0,
            transition_magnitude="minimal",
            transition_direction="stable",
            transition_significance=0.0,
            is_typical_pattern=True,
            pattern_deviation=0.0,
            historical_frequency=0.5,
            transition_risk_score=0.3,
            volatility_impact=0.2
        )
    
    def _get_default_correlation_result(self) -> CrossZoneCorrelationResult:
        """Get default correlation result when analysis fails"""
        
        return CrossZoneCorrelationResult(
            zone_correlation_matrix={},
            strongest_correlation=("NONE", "NONE", 0.0),
            weakest_correlation=("NONE", "NONE", 0.0),
            regime_synchronization_score=0.5,
            divergence_zones=[],
            convergence_zones=[],
            correlation_significance={},
            sample_size_adequacy=0.0
        )
    
    def _get_default_pattern_analysis(self) -> IntradayPatternAnalysis:
        """Get default pattern analysis when data is insufficient"""
        
        return IntradayPatternAnalysis(
            pattern_type="insufficient_data",
            pattern_strength=0.0,
            pattern_persistence=0.0,
            zone_progression=[],
            peak_zone="UNKNOWN",
            trough_zone="UNKNOWN",
            clustering_zones=[],
            clustering_strength=0.0,
            anomalous_zones=[],
            anomaly_severity=0.0,
            next_zone_prediction="UNKNOWN",
            confidence_level=0.0
        )
    
    def _get_default_regime_classification(self) -> Dict[str, Any]:
        """Get default regime classification when analysis fails"""
        
        return {
            'regime_level': 'normal',
            'regime_confidence': 0.5,
            'risk_level': 'normal',
            'action_recommendation': 'neutral',
            'persistence_score': 0.5,
            'zone_specific_factors': {
                'zone_sensitivity': 1.0,
                'time_weight_influence': 0.7,
                'intraday_position_factor': 0.5
            },
            'classification_metadata': {
                'percentile_used': 50.0,
                'zone_name': 'UNKNOWN',
                'threshold_set': {},
                'data_sufficiency': 0.0
            }
        }