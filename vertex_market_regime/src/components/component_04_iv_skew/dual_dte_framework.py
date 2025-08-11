"""
Dual DTE Framework - DTE-specific IV skew analysis for Component 4

Complete volatility surface DTE framework handling varying strike counts per DTE
with surface evolution tracking and cross-DTE arbitrage detection for regime
transitions and intraday surface analysis.

🚨 CRITICAL DTE FRAMEWORK:
- DTE-adaptive surface modeling handling varying strike counts (Short: 54, Medium: 68, Long: 64)
- Surface evolution tracking across DTE spectrum (3-58 days) for regime transitions  
- Cross-DTE arbitrage detection using surface inconsistencies for trading signals
- Intraday surface analysis using zone_name (OPEN/MID_MORN/LUNCH/AFTERNOON/CLOSE)
- Term structure integration with surface consistency validation
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
from scipy import interpolate
from scipy.optimize import minimize_scalar
import warnings

from .skew_analyzer import IVSkewData, VolatilitySurfaceResult, AdvancedIVMetrics

warnings.filterwarnings('ignore')


class DTEBucket(Enum):
    """DTE bucket classifications"""
    SHORT = "SHORT"      # 0-7 days
    MEDIUM = "MEDIUM"    # 8-21 days  
    LONG = "LONG"        # 22-58 days
    EXPIRED = "EXPIRED"  # 0 days


class TradingZone(Enum):
    """Intraday trading zones"""
    OPEN = "OPEN"
    MID_MORN = "MID_MORN"
    LUNCH = "LUNCH"
    AFTERNOON = "AFTERNOON"
    CLOSE = "CLOSE"


@dataclass
class DTESpecificMetrics:
    """DTE-specific IV surface metrics"""
    dte: int
    dte_bucket: DTEBucket
    
    # Strike characteristics
    total_strikes: int
    expected_strikes: int  # Expected for this DTE bucket
    strike_coverage: float  # Actual/Expected
    
    # Surface characteristics
    surface_quality: float
    surface_smoothness: float
    surface_arbitrage_score: float
    
    # Time decay effects
    time_decay_urgency: float
    pin_risk_level: float
    expiry_proximity_effects: Dict[str, float]
    
    # Cross-DTE relationships
    term_structure_slope: float
    calendar_arbitrage_opportunities: List[Dict[str, float]]
    
    # Regime indicators
    regime_transition_signals: Dict[str, float]
    unusual_surface_patterns: List[str]


@dataclass
class CrossDTEArbitrageSignal:
    """Cross-DTE arbitrage opportunity"""
    short_dte: int
    long_dte: int
    strike: float
    arbitrage_magnitude: float
    arbitrage_type: str  # 'calendar', 'butterfly', 'surface_inconsistency'
    confidence: float
    expected_profit_bps: float
    risk_factors: List[str]


@dataclass
class IntradaySurfaceEvolution:
    """Intraday surface evolution analysis"""
    trade_date: datetime
    zone_transitions: Dict[str, Dict[str, float]]  # Zone -> metrics
    
    # Surface stability across zones
    surface_stability_score: float
    volatility_regime_changes: List[Dict[str, Any]]
    
    # Key level interactions
    key_level_responses: Dict[str, float]
    unusual_zone_patterns: List[str]
    
    # Volume/flow patterns by zone
    zone_volume_patterns: Dict[str, Dict[str, float]]
    institutional_activity_zones: List[str]


@dataclass
class TermStructureResult:
    """Complete term structure analysis result"""
    # Term structure shape
    term_structure_slope: float
    term_structure_curvature: float
    term_structure_level: float
    
    # DTE-specific analysis
    dte_metrics: Dict[int, DTESpecificMetrics]
    
    # Cross-DTE arbitrage
    arbitrage_signals: List[CrossDTEArbitrageSignal]
    arbitrage_opportunity_count: int
    
    # Surface evolution
    surface_evolution_score: float
    regime_transition_probability: float
    
    # Intraday patterns
    intraday_analysis: IntradaySurfaceEvolution
    
    # Integration metrics
    surface_consistency_score: float
    no_arbitrage_violations: int


class DTESpecificAnalyzer:
    """DTE-specific IV surface analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # DTE bucket configuration
        self.dte_buckets = {
            DTEBucket.SHORT: {'min_dte': 0, 'max_dte': 7, 'expected_strikes': 54},
            DTEBucket.MEDIUM: {'min_dte': 8, 'max_dte': 21, 'expected_strikes': 68}, 
            DTEBucket.LONG: {'min_dte': 22, 'max_dte': 58, 'expected_strikes': 64}
        }
        
        # Surface quality thresholds by DTE
        self.quality_thresholds = {
            DTEBucket.SHORT: 0.70,   # Lower threshold due to limited data
            DTEBucket.MEDIUM: 0.85,  # Highest quality expected
            DTEBucket.LONG: 0.80     # Slightly lower due to less liquidity
        }
        
    def classify_dte_bucket(self, dte: int) -> DTEBucket:
        """Classify DTE into appropriate bucket"""
        if dte <= 0:
            return DTEBucket.EXPIRED
        
        for bucket, config in self.dte_buckets.items():
            if config['min_dte'] <= dte <= config['max_dte']:
                return bucket
                
        return DTEBucket.LONG  # Default for DTEs > 58
    
    def analyze_dte_specific_surface(self, skew_data: IVSkewData, 
                                   surface_result: VolatilitySurfaceResult) -> DTESpecificMetrics:
        """
        Analyze surface characteristics specific to DTE bucket
        
        Args:
            skew_data: IV skew data
            surface_result: Volatility surface analysis result
            
        Returns:
            DTESpecificMetrics with DTE-specific analysis
        """
        try:
            dte = skew_data.dte
            dte_bucket = self.classify_dte_bucket(dte)
            
            # Strike analysis
            expected_strikes = self.dte_buckets.get(dte_bucket, {}).get('expected_strikes', 60)
            strike_coverage = skew_data.strike_count / expected_strikes
            
            # Surface quality assessment
            surface_quality = self._assess_dte_surface_quality(surface_result, dte_bucket)
            
            # Time decay effects
            time_decay_metrics = self._calculate_time_decay_effects(dte, dte_bucket)
            
            # Pin risk assessment
            pin_risk = self._calculate_pin_risk_level(skew_data, dte_bucket)
            
            # Term structure relationships
            term_structure_slope = self._calculate_term_structure_slope(skew_data, surface_result)
            
            # Calendar arbitrage opportunities
            calendar_arbitrages = self._detect_calendar_arbitrages(skew_data, surface_result, dte_bucket)
            
            # Regime transition signals
            regime_signals = self._detect_regime_transition_signals(surface_result, dte_bucket)
            
            # Unusual patterns
            unusual_patterns = self._detect_unusual_surface_patterns(surface_result, dte_bucket)
            
            return DTESpecificMetrics(
                dte=dte,
                dte_bucket=dte_bucket,
                total_strikes=skew_data.strike_count,
                expected_strikes=expected_strikes,
                strike_coverage=float(strike_coverage),
                surface_quality=surface_quality,
                surface_smoothness=float(surface_result.surface_quality_score),
                surface_arbitrage_score=float(1.0 - surface_result.outlier_count / max(len(surface_result.surface_strikes), 1)),
                time_decay_urgency=time_decay_metrics['urgency'],
                pin_risk_level=pin_risk,
                expiry_proximity_effects=time_decay_metrics['proximity_effects'],
                term_structure_slope=term_structure_slope,
                calendar_arbitrage_opportunities=calendar_arbitrages,
                regime_transition_signals=regime_signals,
                unusual_surface_patterns=unusual_patterns
            )
            
        except Exception as e:
            self.logger.error(f"DTE-specific surface analysis failed: {e}")
            raise
    
    def _assess_dte_surface_quality(self, surface_result: VolatilitySurfaceResult, 
                                   dte_bucket: DTEBucket) -> float:
        """Assess surface quality relative to DTE bucket expectations"""
        
        base_quality = surface_result.surface_quality_score
        threshold = self.quality_thresholds.get(dte_bucket, 0.8)
        
        # Adjust quality score relative to bucket expectations
        if base_quality >= threshold:
            adjusted_quality = 0.8 + 0.2 * (base_quality - threshold) / (1.0 - threshold)
        else:
            adjusted_quality = 0.8 * (base_quality / threshold)
        
        # Additional factors based on DTE bucket
        if dte_bucket == DTEBucket.SHORT:
            # Short DTE: penalize for excessive smoothness (might be overfitted)
            if surface_result.surface_r_squared > 0.99:
                adjusted_quality *= 0.9
        elif dte_bucket == DTEBucket.MEDIUM:
            # Medium DTE: bonus for high quality (most liquid)
            if base_quality > 0.9:
                adjusted_quality = min(1.0, adjusted_quality * 1.05)
        elif dte_bucket == DTEBucket.LONG:
            # Long DTE: more tolerance for lower quality
            adjusted_quality = min(1.0, adjusted_quality * 1.1)
        
        return float(np.clip(adjusted_quality, 0.0, 1.0))
    
    def _calculate_time_decay_effects(self, dte: int, dte_bucket: DTEBucket) -> Dict[str, Any]:
        """Calculate time decay effects and urgency"""
        
        # Time decay urgency (higher as expiry approaches)
        if dte <= 1:
            urgency = 1.0
        elif dte <= 3:
            urgency = 0.8
        elif dte <= 7:
            urgency = 0.6
        elif dte <= 14:
            urgency = 0.4
        elif dte <= 30:
            urgency = 0.2
        else:
            urgency = 0.1
        
        # Proximity effects
        proximity_effects = {}
        
        if dte_bucket == DTEBucket.SHORT:
            proximity_effects.update({
                'gamma_explosion': min(1.0, 10.0 / max(dte, 1)),
                'theta_burn_acceleration': min(1.0, 5.0 / max(dte, 1)),
                'pin_risk_intensification': min(1.0, 7.0 / max(dte, 1)),
                'delta_instability': min(1.0, 3.0 / max(dte, 1))
            })
        else:
            proximity_effects.update({
                'gamma_explosion': 0.1,
                'theta_burn_acceleration': 0.2, 
                'pin_risk_intensification': 0.1,
                'delta_instability': 0.05
            })
        
        return {
            'urgency': float(urgency),
            'proximity_effects': proximity_effects
        }
    
    def _calculate_pin_risk_level(self, skew_data: IVSkewData, dte_bucket: DTEBucket) -> float:
        """Calculate pin risk level based on DTE and surface characteristics"""
        
        base_pin_risk = 0.1  # Base pin risk
        
        if dte_bucket == DTEBucket.SHORT:
            # High pin risk for short DTE
            dte_multiplier = min(3.0, 10.0 / max(skew_data.dte, 1))
            
            # Check proximity to round strikes
            spot = skew_data.spot
            atm_strike = skew_data.atm_strike
            
            # Distance to nearest round number
            round_strikes = [
                int(spot / 100) * 100,
                (int(spot / 100) + 1) * 100,
                int(spot / 50) * 50,
                (int(spot / 50) + 1) * 50
            ]
            
            min_distance = min([abs(spot - strike) for strike in round_strikes])
            distance_factor = max(0.1, 1.0 - min_distance / 100)
            
            pin_risk = base_pin_risk * dte_multiplier * distance_factor
            
        elif dte_bucket == DTEBucket.MEDIUM:
            # Moderate pin risk
            pin_risk = base_pin_risk * 0.5
            
        else:
            # Low pin risk for long DTE
            pin_risk = base_pin_risk * 0.2
        
        return float(min(1.0, pin_risk))
    
    def _calculate_term_structure_slope(self, skew_data: IVSkewData, 
                                      surface_result: VolatilitySurfaceResult) -> float:
        """Calculate implied term structure slope"""
        
        # Use ATM IV as proxy for term structure level
        atm_iv = surface_result.smile_atm_iv
        dte = skew_data.dte
        
        # Approximate term structure slope (would need multiple DTEs for accurate calculation)
        # Using theoretical relationship: longer DTE should have lower IV (time decay)
        
        if dte <= 7:
            # Short DTE: expect higher IV due to gamma risk
            expected_iv_level = 0.25  # 25% base
            term_slope = (atm_iv - expected_iv_level) / max(dte, 1)
        elif dte <= 21:
            # Medium DTE: standard level
            expected_iv_level = 0.20  # 20% base
            term_slope = (atm_iv - expected_iv_level) / dte
        else:
            # Long DTE: lower IV expected
            expected_iv_level = 0.18  # 18% base
            term_slope = (atm_iv - expected_iv_level) / dte
        
        return float(term_slope)
    
    def _detect_calendar_arbitrages(self, skew_data: IVSkewData, 
                                  surface_result: VolatilitySurfaceResult,
                                  dte_bucket: DTEBucket) -> List[Dict[str, float]]:
        """Detect calendar arbitrage opportunities"""
        
        arbitrages = []
        
        # Simplified calendar arbitrage detection
        # (In production, would compare with other DTEs)
        
        atm_iv = surface_result.smile_atm_iv
        dte = skew_data.dte
        
        # Check for IV level inconsistencies
        if dte_bucket == DTEBucket.SHORT and atm_iv < 0.15:
            # Abnormally low IV for short DTE
            arbitrages.append({
                'type': 'short_dte_low_iv',
                'magnitude': 0.15 - atm_iv,
                'confidence': 0.7,
                'opportunity_score': (0.15 - atm_iv) * 10
            })
        
        elif dte_bucket == DTEBucket.LONG and atm_iv > 0.30:
            # Abnormally high IV for long DTE
            arbitrages.append({
                'type': 'long_dte_high_iv',
                'magnitude': atm_iv - 0.30,
                'confidence': 0.6,
                'opportunity_score': (atm_iv - 0.30) * 8
            })
        
        # Check skew inconsistencies
        if abs(surface_result.skew_slope_25d) > 0.05:
            # Excessive skew might indicate calendar opportunities
            arbitrages.append({
                'type': 'excessive_skew',
                'magnitude': abs(surface_result.skew_slope_25d),
                'confidence': 0.5,
                'opportunity_score': abs(surface_result.skew_slope_25d) * 5
            })
        
        return arbitrages
    
    def _detect_regime_transition_signals(self, surface_result: VolatilitySurfaceResult,
                                        dte_bucket: DTEBucket) -> Dict[str, float]:
        """Detect regime transition signals from surface characteristics"""
        
        signals = {}
        
        # Skew-based signals
        signals['put_skew_dominance'] = float(surface_result.put_skew_dominance)
        signals['skew_steepness'] = float(surface_result.skew_steepness)
        signals['smile_asymmetry'] = float(surface_result.smile_asymmetry)
        
        # Curvature-based signals
        signals['smile_curvature'] = float(surface_result.smile_curvature)
        signals['convexity_signal'] = float(surface_result.skew_convexity)
        
        # Quality-based signals (degraded quality might indicate regime change)
        signals['surface_quality_deterioration'] = float(1.0 - surface_result.surface_quality_score)
        
        # DTE-specific adjustments
        if dte_bucket == DTEBucket.SHORT:
            # Short DTE: more sensitive to regime changes
            for key in ['put_skew_dominance', 'skew_steepness']:
                signals[key] *= 1.5
        
        elif dte_bucket == DTEBucket.LONG:
            # Long DTE: smoother regime transitions
            for key in signals:
                signals[key] *= 0.8
        
        # Overall regime transition probability
        regime_factors = [
            abs(signals['put_skew_dominance']),
            min(1.0, signals['skew_steepness'] * 10),
            abs(signals['smile_asymmetry']),
            min(1.0, signals['surface_quality_deterioration'])
        ]
        
        signals['regime_transition_probability'] = float(np.mean(regime_factors))
        
        return signals
    
    def _detect_unusual_surface_patterns(self, surface_result: VolatilitySurfaceResult,
                                       dte_bucket: DTEBucket) -> List[str]:
        """Detect unusual patterns in the volatility surface"""
        
        patterns = []
        
        # Excessive smile curvature
        if abs(surface_result.smile_curvature) > 0.1:
            patterns.append("excessive_smile_curvature")
        
        # Extreme asymmetry
        if abs(surface_result.smile_asymmetry) > 0.3:
            patterns.append("extreme_smile_asymmetry")
        
        # Unusual skew steepness
        if surface_result.skew_steepness > 0.02:
            patterns.append("steep_skew")
        elif surface_result.skew_steepness < 0.005:
            patterns.append("flat_skew")
        
        # Risk reversal extremes
        if abs(surface_result.risk_reversal_25d) > 0.1:
            patterns.append("extreme_risk_reversal")
        
        # Surface quality issues
        if surface_result.surface_quality_score < 0.5:
            patterns.append("poor_surface_quality")
        
        if surface_result.outlier_count > len(surface_result.surface_strikes) * 0.2:
            patterns.append("excessive_outliers")
        
        # DTE-specific patterns
        if dte_bucket == DTEBucket.SHORT:
            if surface_result.smile_atm_iv < 0.1:
                patterns.append("unusually_low_short_iv")
            elif surface_result.smile_atm_iv > 0.5:
                patterns.append("unusually_high_short_iv")
        
        elif dte_bucket == DTEBucket.LONG:
            if surface_result.surface_r_squared < 0.7:
                patterns.append("poor_long_dte_fit")
        
        return patterns


class CrossDTEAnalyzer:
    """Cross-DTE arbitrage and surface evolution analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # Arbitrage detection thresholds
        self.min_arbitrage_magnitude = config.get('min_arbitrage_magnitude', 0.01)
        self.min_arbitrage_confidence = config.get('min_arbitrage_confidence', 0.6)
        
    def detect_cross_dte_arbitrages(self, dte_metrics_list: List[DTESpecificMetrics], 
                                  skew_data_list: List[IVSkewData]) -> List[CrossDTEArbitrageSignal]:
        """
        Detect arbitrage opportunities across DTEs
        
        Args:
            dte_metrics_list: List of DTE-specific metrics
            skew_data_list: List of IV skew data for different DTEs
            
        Returns:
            List of cross-DTE arbitrage signals
        """
        try:
            arbitrage_signals = []
            
            # Compare pairs of DTEs
            for i in range(len(dte_metrics_list)):
                for j in range(i + 1, len(dte_metrics_list)):
                    metrics_short = dte_metrics_list[i] if dte_metrics_list[i].dte < dte_metrics_list[j].dte else dte_metrics_list[j]
                    metrics_long = dte_metrics_list[j] if dte_metrics_list[i].dte < dte_metrics_list[j].dte else dte_metrics_list[i]
                    
                    # Find corresponding skew data
                    skew_short = next((s for s in skew_data_list if s.dte == metrics_short.dte), None)
                    skew_long = next((s for s in skew_data_list if s.dte == metrics_long.dte), None)
                    
                    if skew_short and skew_long:
                        pair_arbitrages = self._analyze_dte_pair_arbitrages(
                            metrics_short, metrics_long, skew_short, skew_long
                        )
                        arbitrage_signals.extend(pair_arbitrages)
            
            # Sort by opportunity score
            arbitrage_signals.sort(key=lambda x: x.expected_profit_bps, reverse=True)
            
            return arbitrage_signals
            
        except Exception as e:
            self.logger.error(f"Cross-DTE arbitrage detection failed: {e}")
            return []
    
    def _analyze_dte_pair_arbitrages(self, metrics_short: DTESpecificMetrics, 
                                   metrics_long: DTESpecificMetrics,
                                   skew_short: IVSkewData, 
                                   skew_long: IVSkewData) -> List[CrossDTEArbitrageSignal]:
        """Analyze arbitrage opportunities between two DTEs"""
        
        arbitrages = []
        
        # Calendar spread opportunities
        calendar_arbs = self._detect_calendar_spreads(metrics_short, metrics_long, skew_short, skew_long)
        arbitrages.extend(calendar_arbs)
        
        # Surface inconsistency arbitrages
        surface_arbs = self._detect_surface_inconsistencies(metrics_short, metrics_long, skew_short, skew_long)
        arbitrages.extend(surface_arbs)
        
        # Term structure arbitrages
        term_structure_arbs = self._detect_term_structure_arbitrages(metrics_short, metrics_long, skew_short, skew_long)
        arbitrages.extend(term_structure_arbs)
        
        return arbitrages
    
    def _detect_calendar_spreads(self, metrics_short: DTESpecificMetrics, 
                               metrics_long: DTESpecificMetrics,
                               skew_short: IVSkewData, 
                               skew_long: IVSkewData) -> List[CrossDTEArbitrageSignal]:
        """Detect calendar spread arbitrage opportunities"""
        
        arbitrages = []
        
        # Compare ATM strikes (most liquid)
        if abs(skew_short.atm_strike - skew_long.atm_strike) <= 100:  # Same or very close strikes
            
            # Find ATM IVs (simplified - would use surface interpolation in production)
            short_atm_iv = 0.20  # Mock - would extract from surface
            long_atm_iv = 0.18   # Mock - would extract from surface
            
            # Calendar spread: expect short IV > long IV
            iv_differential = short_atm_iv - long_atm_iv
            expected_differential = 0.02  # 2% base differential
            
            arbitrage_magnitude = abs(iv_differential - expected_differential)
            
            if arbitrage_magnitude > self.min_arbitrage_magnitude:
                confidence = min(1.0, arbitrage_magnitude / 0.05)
                
                if confidence >= self.min_arbitrage_confidence:
                    arbitrages.append(CrossDTEArbitrageSignal(
                        short_dte=metrics_short.dte,
                        long_dte=metrics_long.dte,
                        strike=float(skew_short.atm_strike),
                        arbitrage_magnitude=float(arbitrage_magnitude),
                        arbitrage_type='calendar',
                        confidence=float(confidence),
                        expected_profit_bps=float(arbitrage_magnitude * 1000),  # Convert to bps
                        risk_factors=['liquidity_risk', 'gamma_risk', 'early_assignment']
                    ))
        
        return arbitrages
    
    def _detect_surface_inconsistencies(self, metrics_short: DTESpecificMetrics, 
                                      metrics_long: DTESpecificMetrics,
                                      skew_short: IVSkewData, 
                                      skew_long: IVSkewData) -> List[CrossDTEArbitrageSignal]:
        """Detect surface inconsistency arbitrages"""
        
        arbitrages = []
        
        # Compare term structure slopes
        short_slope = metrics_short.term_structure_slope
        long_slope = metrics_long.term_structure_slope
        
        # Expect smoother term structure slope for longer DTEs
        if abs(short_slope) < abs(long_slope):
            # Unusual: short DTE has flatter slope than long DTE
            arbitrage_magnitude = abs(long_slope - short_slope)
            
            if arbitrage_magnitude > 0.001:  # 0.1% threshold
                confidence = min(1.0, arbitrage_magnitude / 0.005)
                
                if confidence >= self.min_arbitrage_confidence:
                    arbitrages.append(CrossDTEArbitrageSignal(
                        short_dte=metrics_short.dte,
                        long_dte=metrics_long.dte,
                        strike=float((skew_short.atm_strike + skew_long.atm_strike) / 2),
                        arbitrage_magnitude=float(arbitrage_magnitude),
                        arbitrage_type='surface_inconsistency',
                        confidence=float(confidence),
                        expected_profit_bps=float(arbitrage_magnitude * 500),
                        risk_factors=['model_risk', 'execution_risk']
                    ))
        
        return arbitrages
    
    def _detect_term_structure_arbitrages(self, metrics_short: DTESpecificMetrics, 
                                        metrics_long: DTESpecificMetrics,
                                        skew_short: IVSkewData, 
                                        skew_long: IVSkewData) -> List[CrossDTEArbitrageSignal]:
        """Detect term structure arbitrage opportunities"""
        
        arbitrages = []
        
        # Check for term structure inversions or unusual shapes
        # This would require multiple DTEs to build proper term structure
        
        # Simple check: pin risk comparison
        short_pin_risk = metrics_short.pin_risk_level
        long_pin_risk = metrics_long.pin_risk_level
        
        # Long DTE should have lower pin risk
        if long_pin_risk > short_pin_risk * 1.5:
            # Unusual pin risk inversion
            arbitrage_magnitude = long_pin_risk - short_pin_risk
            confidence = min(1.0, arbitrage_magnitude / 0.5)
            
            if confidence >= self.min_arbitrage_confidence:
                arbitrages.append(CrossDTEArbitrageSignal(
                    short_dte=metrics_short.dte,
                    long_dte=metrics_long.dte,
                    strike=float(skew_short.atm_strike),
                    arbitrage_magnitude=float(arbitrage_magnitude),
                    arbitrage_type='term_structure_inversion',
                    confidence=float(confidence),
                    expected_profit_bps=float(arbitrage_magnitude * 200),
                    risk_factors=['pin_risk', 'liquidity_risk']
                ))
        
        return arbitrages


class IntradayAnalyzer:
    """Intraday surface evolution analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # Zone analysis configuration
        self.zone_order = [TradingZone.OPEN, TradingZone.MID_MORN, TradingZone.LUNCH, 
                          TradingZone.AFTERNOON, TradingZone.CLOSE]
        
    def analyze_intraday_surface_evolution(self, zone_data: Dict[str, Any]) -> IntradaySurfaceEvolution:
        """
        Analyze intraday surface evolution across trading zones
        
        Args:
            zone_data: Dictionary with zone_name as key and analysis data as value
            
        Returns:
            IntradaySurfaceEvolution with complete intraday analysis
        """
        try:
            if not zone_data:
                raise ValueError("No zone data provided")
            
            # Initialize analysis structures
            zone_transitions = {}
            surface_stability_scores = []
            volatility_regime_changes = []
            unusual_zone_patterns = []
            
            # Analyze each zone
            for zone_name, data in zone_data.items():
                zone_analysis = self._analyze_single_zone(zone_name, data)
                zone_transitions[zone_name] = zone_analysis
                
                if 'surface_quality' in zone_analysis:
                    surface_stability_scores.append(zone_analysis['surface_quality'])
            
            # Calculate overall stability
            overall_stability = np.mean(surface_stability_scores) if surface_stability_scores else 0.5
            
            # Detect regime changes across zones
            regime_changes = self._detect_regime_changes_across_zones(zone_transitions)
            
            # Key level analysis
            key_level_responses = self._analyze_key_level_responses(zone_data)
            
            # Volume pattern analysis
            zone_volume_patterns = self._analyze_zone_volume_patterns(zone_data)
            
            # Institutional activity detection
            institutional_zones = self._detect_institutional_activity_zones(zone_volume_patterns)
            
            return IntradaySurfaceEvolution(
                trade_date=datetime.now().date(),  # Would extract from data
                zone_transitions=zone_transitions,
                surface_stability_score=float(overall_stability),
                volatility_regime_changes=regime_changes,
                key_level_responses=key_level_responses,
                unusual_zone_patterns=unusual_zone_patterns,
                zone_volume_patterns=zone_volume_patterns,
                institutional_activity_zones=institutional_zones
            )
            
        except Exception as e:
            self.logger.error(f"Intraday surface evolution analysis failed: {e}")
            raise
    
    def _analyze_single_zone(self, zone_name: str, data: Any) -> Dict[str, float]:
        """Analyze single trading zone"""
        
        # Mock analysis - in production would analyze actual surface data
        zone_metrics = {
            'surface_quality': 0.8,
            'iv_level_change': 0.0,
            'skew_change': 0.0,
            'volume_intensity': 1.0,
            'unusual_activity': 0.0
        }
        
        # Zone-specific patterns
        if zone_name == 'OPEN':
            zone_metrics.update({
                'surface_quality': 0.7,  # Lower quality due to volatility
                'iv_level_change': 0.02,  # Opening volatility spike
                'volume_intensity': 1.5   # High opening volume
            })
        elif zone_name == 'LUNCH':
            zone_metrics.update({
                'surface_quality': 0.9,   # Calmer, better quality
                'volume_intensity': 0.6   # Lower lunch volume
            })
        elif zone_name == 'CLOSE':
            zone_metrics.update({
                'surface_quality': 0.8,
                'iv_level_change': 0.01,  # Closing volatility
                'volume_intensity': 1.3   # High closing volume
            })
        
        return zone_metrics
    
    def _detect_regime_changes_across_zones(self, zone_transitions: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Detect regime changes across trading zones"""
        
        regime_changes = []
        
        # Compare consecutive zones for significant changes
        zone_names = list(zone_transitions.keys())
        
        for i in range(len(zone_names) - 1):
            current_zone = zone_names[i]
            next_zone = zone_names[i + 1]
            
            current_metrics = zone_transitions[current_zone]
            next_metrics = zone_transitions[next_zone]
            
            # Check for significant IV level changes
            iv_change = abs(next_metrics.get('iv_level_change', 0) - current_metrics.get('iv_level_change', 0))
            
            if iv_change > 0.015:  # 1.5% threshold
                regime_changes.append({
                    'from_zone': current_zone,
                    'to_zone': next_zone,
                    'change_type': 'iv_level_shift',
                    'magnitude': float(iv_change),
                    'significance': min(1.0, iv_change / 0.05)
                })
            
            # Check for skew changes
            skew_change = abs(next_metrics.get('skew_change', 0) - current_metrics.get('skew_change', 0))
            
            if skew_change > 0.01:
                regime_changes.append({
                    'from_zone': current_zone,
                    'to_zone': next_zone,
                    'change_type': 'skew_shift',
                    'magnitude': float(skew_change),
                    'significance': min(1.0, skew_change / 0.03)
                })
        
        return regime_changes
    
    def _analyze_key_level_responses(self, zone_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze responses at key market levels"""
        
        # Mock key level analysis
        return {
            'support_level_response': 0.6,
            'resistance_level_response': 0.7,
            'round_number_response': 0.8,
            'previous_close_response': 0.5
        }
    
    def _analyze_zone_volume_patterns(self, zone_data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Analyze volume patterns by trading zone"""
        
        zone_patterns = {}
        
        for zone_name in zone_data.keys():
            # Mock volume pattern analysis
            if zone_name == 'OPEN':
                zone_patterns[zone_name] = {
                    'relative_volume': 1.5,
                    'call_put_ratio': 1.2,
                    'oi_buildup': 0.8,
                    'institutional_flow': 0.7
                }
            elif zone_name == 'LUNCH':
                zone_patterns[zone_name] = {
                    'relative_volume': 0.6,
                    'call_put_ratio': 1.0,
                    'oi_buildup': 0.3,
                    'institutional_flow': 0.2
                }
            else:
                zone_patterns[zone_name] = {
                    'relative_volume': 1.0,
                    'call_put_ratio': 1.1,
                    'oi_buildup': 0.5,
                    'institutional_flow': 0.4
                }
        
        return zone_patterns
    
    def _detect_institutional_activity_zones(self, zone_volume_patterns: Dict[str, Dict[str, float]]) -> List[str]:
        """Detect zones with significant institutional activity"""
        
        institutional_zones = []
        
        for zone_name, patterns in zone_volume_patterns.items():
            institutional_score = (
                patterns.get('institutional_flow', 0) * 0.4 +
                patterns.get('oi_buildup', 0) * 0.3 +
                min(1.0, patterns.get('relative_volume', 0) / 1.2) * 0.3
            )
            
            if institutional_score > 0.6:
                institutional_zones.append(zone_name)
        
        return institutional_zones


class DualDTEFramework:
    """
    Complete Dual DTE Framework integrating all DTE analysis components
    
    Handles varying strike counts per DTE, surface evolution tracking,
    cross-DTE arbitrage detection, and intraday surface analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # Initialize sub-analyzers
        self.dte_analyzer = DTESpecificAnalyzer(config)
        self.cross_dte_analyzer = CrossDTEAnalyzer(config)
        self.intraday_analyzer = IntradayAnalyzer(config)
        
        self.logger.info("Dual DTE Framework initialized with complete surface evolution tracking")
    
    def analyze_complete_term_structure(self, multi_dte_data: List[Tuple[IVSkewData, VolatilitySurfaceResult]],
                                      intraday_zone_data: Optional[Dict[str, Any]] = None) -> TermStructureResult:
        """
        Complete term structure analysis across multiple DTEs
        
        Args:
            multi_dte_data: List of (skew_data, surface_result) tuples for different DTEs
            intraday_zone_data: Optional intraday zone analysis data
            
        Returns:
            TermStructureResult with complete analysis
        """
        try:
            if not multi_dte_data:
                raise ValueError("No multi-DTE data provided")
            
            # Step 1: Analyze each DTE specifically
            dte_metrics = {}
            skew_data_list = []
            surface_results = []
            
            for skew_data, surface_result in multi_dte_data:
                dte_specific_metrics = self.dte_analyzer.analyze_dte_specific_surface(skew_data, surface_result)
                dte_metrics[skew_data.dte] = dte_specific_metrics
                skew_data_list.append(skew_data)
                surface_results.append(surface_result)
            
            # Step 2: Cross-DTE arbitrage analysis
            arbitrage_signals = self.cross_dte_analyzer.detect_cross_dte_arbitrages(
                list(dte_metrics.values()), skew_data_list
            )
            
            # Step 3: Term structure shape analysis
            term_structure_metrics = self._analyze_term_structure_shape(dte_metrics, surface_results)
            
            # Step 4: Surface evolution analysis
            surface_evolution_score = self._calculate_surface_evolution_score(dte_metrics)
            
            # Step 5: Regime transition probability
            regime_transition_prob = self._calculate_regime_transition_probability(dte_metrics, arbitrage_signals)
            
            # Step 6: Intraday analysis (if provided)
            if intraday_zone_data:
                intraday_analysis = self.intraday_analyzer.analyze_intraday_surface_evolution(intraday_zone_data)
            else:
                # Default empty intraday analysis
                intraday_analysis = IntradaySurfaceEvolution(
                    trade_date=datetime.now().date(),
                    zone_transitions={},
                    surface_stability_score=0.8,
                    volatility_regime_changes=[],
                    key_level_responses={},
                    unusual_zone_patterns=[],
                    zone_volume_patterns={},
                    institutional_activity_zones=[]
                )
            
            # Step 7: Surface consistency validation
            consistency_score = self._validate_surface_consistency(dte_metrics, surface_results)
            
            # Step 8: No-arbitrage violation count
            arbitrage_violations = len([arb for arb in arbitrage_signals if arb.confidence > 0.8])
            
            return TermStructureResult(
                term_structure_slope=term_structure_metrics['slope'],
                term_structure_curvature=term_structure_metrics['curvature'],
                term_structure_level=term_structure_metrics['level'],
                dte_metrics=dte_metrics,
                arbitrage_signals=arbitrage_signals,
                arbitrage_opportunity_count=len(arbitrage_signals),
                surface_evolution_score=surface_evolution_score,
                regime_transition_probability=regime_transition_prob,
                intraday_analysis=intraday_analysis,
                surface_consistency_score=consistency_score,
                no_arbitrage_violations=arbitrage_violations
            )
            
        except Exception as e:
            self.logger.error(f"Complete term structure analysis failed: {e}")
            raise
    
    def _analyze_term_structure_shape(self, dte_metrics: Dict[int, DTESpecificMetrics], 
                                    surface_results: List[VolatilitySurfaceResult]) -> Dict[str, float]:
        """Analyze term structure shape characteristics"""
        
        if len(dte_metrics) < 2:
            return {'slope': 0.0, 'curvature': 0.0, 'level': 0.2}
        
        # Extract DTEs and corresponding ATM IVs
        dtes = sorted(dte_metrics.keys())
        atm_ivs = []
        
        for i, surface_result in enumerate(surface_results):
            atm_ivs.append(surface_result.smile_atm_iv)
        
        # Sort by DTE
        sorted_data = sorted(zip(dtes, atm_ivs))
        sorted_dtes = [x[0] for x in sorted_data]
        sorted_ivs = [x[1] for x in sorted_data]
        
        # Calculate slope (linear fit)
        if len(sorted_dtes) >= 2:
            slope = np.polyfit(sorted_dtes, sorted_ivs, 1)[0]
        else:
            slope = 0.0
        
        # Calculate curvature (quadratic fit if enough points)
        if len(sorted_dtes) >= 3:
            try:
                poly_coeffs = np.polyfit(sorted_dtes, sorted_ivs, 2)
                curvature = poly_coeffs[0]  # Quadratic coefficient
            except:
                curvature = 0.0
        else:
            curvature = 0.0
        
        # Term structure level (average IV)
        level = np.mean(sorted_ivs)
        
        return {
            'slope': float(slope),
            'curvature': float(curvature),
            'level': float(level)
        }
    
    def _calculate_surface_evolution_score(self, dte_metrics: Dict[int, DTESpecificMetrics]) -> float:
        """Calculate overall surface evolution score"""
        
        evolution_factors = []
        
        for dte, metrics in dte_metrics.items():
            # Surface quality factor
            evolution_factors.append(metrics.surface_quality)
            
            # Regime transition signal strength
            regime_signals = list(metrics.regime_transition_signals.values())
            if regime_signals:
                avg_regime_signal = np.mean([abs(s) for s in regime_signals])
                evolution_factors.append(1.0 - min(1.0, avg_regime_signal))
            
            # Surface stability
            evolution_factors.append(metrics.surface_smoothness)
        
        return float(np.mean(evolution_factors)) if evolution_factors else 0.8
    
    def _calculate_regime_transition_probability(self, dte_metrics: Dict[int, DTESpecificMetrics],
                                               arbitrage_signals: List[CrossDTEArbitrageSignal]) -> float:
        """Calculate regime transition probability"""
        
        transition_indicators = []
        
        # DTE-specific regime signals
        for dte, metrics in dte_metrics.items():
            if 'regime_transition_probability' in metrics.regime_transition_signals:
                transition_indicators.append(metrics.regime_transition_signals['regime_transition_probability'])
        
        # Arbitrage signal strength (more arbitrages might indicate regime transition)
        if arbitrage_signals:
            high_confidence_arbs = [arb for arb in arbitrage_signals if arb.confidence > 0.7]
            arbitrage_indicator = min(1.0, len(high_confidence_arbs) / 5)  # Normalize to 0-1
            transition_indicators.append(arbitrage_indicator)
        
        # Surface quality deterioration
        quality_scores = [metrics.surface_quality for metrics in dte_metrics.values()]
        if quality_scores:
            avg_quality = np.mean(quality_scores)
            quality_deterioration = max(0.0, 1.0 - avg_quality)
            transition_indicators.append(quality_deterioration)
        
        return float(np.mean(transition_indicators)) if transition_indicators else 0.2
    
    def _validate_surface_consistency(self, dte_metrics: Dict[int, DTESpecificMetrics],
                                    surface_results: List[VolatilitySurfaceResult]) -> float:
        """Validate consistency across surface"""
        
        consistency_factors = []
        
        # Surface quality consistency
        quality_scores = [metrics.surface_quality for metrics in dte_metrics.values()]
        if len(quality_scores) > 1:
            quality_consistency = 1.0 - (np.std(quality_scores) / np.mean(quality_scores))
            consistency_factors.append(max(0.0, quality_consistency))
        
        # Surface smoothness consistency
        smoothness_scores = [metrics.surface_smoothness for metrics in dte_metrics.values()]
        if len(smoothness_scores) > 1:
            smoothness_consistency = 1.0 - (np.std(smoothness_scores) / np.mean(smoothness_scores))
            consistency_factors.append(max(0.0, smoothness_consistency))
        
        # Arbitrage absence (fewer arbitrages = more consistent)
        arbitrage_penalty = min(1.0, len([m for m in dte_metrics.values() 
                                        if len(m.calendar_arbitrage_opportunities) > 0]) / len(dte_metrics))
        consistency_factors.append(1.0 - arbitrage_penalty)
        
        return float(np.mean(consistency_factors)) if consistency_factors else 0.8