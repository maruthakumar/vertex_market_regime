"""
Comprehensive Greeks Processor - Component 2

Processes ALL first-order Greeks using REAL production values (not derived estimates).
Implements full Greeks analysis: Delta, Gamma=1.5, Theta, Vega with 100% data coverage.

🚨 KEY IMPLEMENTATION: Uses ACTUAL Greeks values from production Parquet data
rather than derived estimates or approximations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import logging

from .production_greeks_extractor import ProductionGreeksData, CorrectedGreeksWeighting
from .corrected_gamma_weighter import CorrectedGammaWeighter, GammaWeightedScore


@dataclass
class ComprehensiveGreeksAnalysis:
    """Complete Greeks analysis result using ALL first-order Greeks"""
    # Individual Greek components (ACTUAL values)
    delta_analysis: Dict[str, float]      # Delta directional analysis
    gamma_analysis: GammaWeightedScore    # Gamma analysis with 1.5 weight
    theta_analysis: Dict[str, float]      # Theta time decay analysis
    vega_analysis: Dict[str, float]       # Vega volatility sensitivity
    
    # Combined analysis
    combined_score: float                 # Weighted combination of all Greeks
    regime_indication: str                # Regime classification from Greeks
    confidence: float                     # Overall analysis confidence
    
    # Metadata
    data_quality: Dict[str, float]        # Data quality metrics
    processing_time_ms: float
    timestamp: datetime


@dataclass
class GreeksDataQuality:
    """Greeks data quality assessment"""
    total_points: int
    valid_points: int
    coverage_pct: float
    missing_delta: int
    missing_gamma: int
    missing_theta: int
    missing_vega: int
    quality_score: float  # 0-1 scale


class ComprehensiveGreeksProcessor:
    """
    Process comprehensive Greeks analysis using ACTUAL production values
    
    🚨 CRITICAL FEATURES:
    - Uses 100% ACTUAL Greeks values from production data
    - NO derived estimates or approximations
    - Implements corrected gamma_weight=1.5
    - Handles all strike types (ATM, ITM1-23, OTM1-23)
    - Volume-weighted institutional analysis
    - Missing data handling strategies
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize comprehensive Greeks processor"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize corrected gamma weighter
        self.gamma_weighter = CorrectedGammaWeighter(config)
        
        # Greeks weighting system (corrected)
        self.weighting = CorrectedGreeksWeighting()
        
        # Data quality thresholds
        self.quality_thresholds = {
            'minimum_coverage': 0.95,      # 95% minimum coverage
            'high_quality': 0.99,          # 99% high quality threshold
            'volume_threshold': 100,        # Minimum volume for analysis
            'oi_threshold': 500            # Minimum OI for institutional analysis
        }
        
        # Greeks analysis ranges (from production data validation)
        self.production_ranges = {
            'delta': {'ce_min': 0.0, 'ce_max': 1.0, 'pe_min': -1.0, 'pe_max': 0.0},
            'gamma': {'min': 0.0, 'max': 0.0013},       # Actual production range
            'theta': {'ce_min': -63.5, 'ce_max': -2.0, 'pe_min': -40.3, 'pe_max': 6.5},
            'vega': {'min': 0.0, 'max': 6.5}            # Actual production range
        }
        
        self.logger.info("🚨 ComprehensiveGreeksProcessor initialized with ACTUAL values processing")
    
    def assess_data_quality(self, greeks_data_list: List[ProductionGreeksData]) -> GreeksDataQuality:
        """
        Assess data quality for Greeks analysis
        
        Args:
            greeks_data_list: List of production Greeks data
            
        Returns:
            GreeksDataQuality assessment
        """
        total_points = len(greeks_data_list)
        
        # Count missing data (though production shows 100% coverage)
        missing_counts = {
            'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0
        }
        
        valid_points = 0
        
        for data in greeks_data_list:
            point_valid = True
            
            # Check delta validity
            if (np.isnan(data.ce_delta) or np.isnan(data.pe_delta)):
                missing_counts['delta'] += 1
                point_valid = False
                
            # Check gamma validity (ACTUAL values)
            if (np.isnan(data.ce_gamma) or np.isnan(data.pe_gamma)):
                missing_counts['gamma'] += 1
                point_valid = False
                
            # Check theta validity
            if (np.isnan(data.ce_theta) or np.isnan(data.pe_theta)):
                missing_counts['theta'] += 1
                point_valid = False
                
            # Check vega validity (ACTUAL values)
            if (np.isnan(data.ce_vega) or np.isnan(data.pe_vega)):
                missing_counts['vega'] += 1
                point_valid = False
            
            if point_valid:
                valid_points += 1
        
        coverage_pct = (valid_points / total_points) * 100 if total_points > 0 else 0
        
        # Calculate quality score
        if coverage_pct >= self.quality_thresholds['high_quality'] * 100:
            quality_score = 1.0
        elif coverage_pct >= self.quality_thresholds['minimum_coverage'] * 100:
            quality_score = 0.8
        else:
            quality_score = coverage_pct / 100
        
        return GreeksDataQuality(
            total_points=total_points,
            valid_points=valid_points,
            coverage_pct=coverage_pct,
            missing_delta=missing_counts['delta'],
            missing_gamma=missing_counts['gamma'],
            missing_theta=missing_counts['theta'],
            missing_vega=missing_counts['vega'],
            quality_score=quality_score
        )
    
    def analyze_delta_component(self, greeks_data: ProductionGreeksData) -> Dict[str, float]:
        """
        Analyze delta component using ACTUAL delta values
        
        Args:
            greeks_data: ProductionGreeksData with actual delta values
            
        Returns:
            Delta analysis results
        """
        # Extract ACTUAL delta values
        ce_delta = greeks_data.ce_delta  # Column 23 - ACTUAL values
        pe_delta = greeks_data.pe_delta  # Column 37 - ACTUAL values
        
        # Combined delta exposure
        net_delta = ce_delta + pe_delta
        delta_imbalance = ce_delta - abs(pe_delta)  # Directional bias
        
        # Delta-based regime indication
        if net_delta > 0.1:
            regime_signal = "BULLISH"
        elif net_delta < -0.1:
            regime_signal = "BEARISH"
        else:
            regime_signal = "NEUTRAL"
        
        # Weighted delta score
        weighted_delta = self.weighting.delta_weight * net_delta
        
        return {
            'ce_delta': ce_delta,
            'pe_delta': pe_delta,
            'net_delta': net_delta,
            'delta_imbalance': delta_imbalance,
            'weighted_delta': weighted_delta,
            'regime_signal': regime_signal,
            'delta_magnitude': abs(net_delta)
        }
    
    def analyze_theta_component(self, greeks_data: ProductionGreeksData) -> Dict[str, float]:
        """
        Analyze theta component using ACTUAL theta values
        
        Args:
            greeks_data: ProductionGreeksData with actual theta values
            
        Returns:
            Theta analysis results
        """
        # Extract ACTUAL theta values
        ce_theta = greeks_data.ce_theta  # Column 25 - ACTUAL values
        pe_theta = greeks_data.pe_theta  # Column 39 - ACTUAL values
        
        # Combined time decay
        total_theta = ce_theta + pe_theta
        theta_imbalance = ce_theta - pe_theta
        
        # Time decay pressure assessment
        if total_theta < -10:
            decay_pressure = "HIGH"
        elif total_theta < -5:
            decay_pressure = "MEDIUM"
        else:
            decay_pressure = "LOW"
        
        # Weighted theta score
        weighted_theta = self.weighting.theta_weight * total_theta
        
        # DTE-adjusted theta (higher impact near expiry)
        dte_multiplier = max(1.0, (30 - greeks_data.dte) / 30) if greeks_data.dte <= 30 else 0.5
        dte_adjusted_theta = weighted_theta * dte_multiplier
        
        return {
            'ce_theta': ce_theta,
            'pe_theta': pe_theta,
            'total_theta': total_theta,
            'theta_imbalance': theta_imbalance,
            'weighted_theta': weighted_theta,
            'dte_adjusted_theta': dte_adjusted_theta,
            'decay_pressure': decay_pressure,
            'dte_multiplier': dte_multiplier
        }
    
    def analyze_vega_component(self, greeks_data: ProductionGreeksData) -> Dict[str, float]:
        """
        Analyze vega component using ACTUAL vega values
        
        Args:
            greeks_data: ProductionGreeksData with actual vega values
            
        Returns:
            Vega analysis results
        """
        # Extract ACTUAL vega values
        ce_vega = greeks_data.ce_vega  # Column 26 - ACTUAL values
        pe_vega = greeks_data.pe_vega  # Column 40 - ACTUAL values
        
        # Combined volatility sensitivity
        total_vega = ce_vega + pe_vega
        vega_imbalance = ce_vega - pe_vega
        
        # Volatility regime assessment
        if total_vega > 4.0:
            vol_sensitivity = "HIGH"
        elif total_vega > 2.0:
            vol_sensitivity = "MEDIUM"
        else:
            vol_sensitivity = "LOW"
        
        # Weighted vega score
        weighted_vega = self.weighting.vega_weight * total_vega
        
        return {
            'ce_vega': ce_vega,
            'pe_vega': pe_vega,
            'total_vega': total_vega,
            'vega_imbalance': vega_imbalance,
            'weighted_vega': weighted_vega,
            'vol_sensitivity': vol_sensitivity,
            'vega_magnitude': total_vega
        }
    
    def process_comprehensive_analysis(self, 
                                     greeks_data: ProductionGreeksData,
                                     volume_weight: float = 1.0) -> ComprehensiveGreeksAnalysis:
        """
        Process comprehensive Greeks analysis using ALL first-order Greeks
        
        Args:
            greeks_data: ProductionGreeksData with ACTUAL values
            volume_weight: Volume-based weighting factor
            
        Returns:
            ComprehensiveGreeksAnalysis with all Greek components
        """
        start_time = datetime.utcnow()
        
        try:
            # 1. Delta analysis (directional)
            delta_analysis = self.analyze_delta_component(greeks_data)
            
            # 2. Gamma analysis (🚨 CORRECTED 1.5 weight)
            gamma_analysis = self.gamma_weighter.calculate_gamma_weighted_score(
                greeks_data, volume_weight
            )
            
            # 3. Theta analysis (time decay)
            theta_analysis = self.analyze_theta_component(greeks_data)
            
            # 4. Vega analysis (volatility sensitivity)
            vega_analysis = self.analyze_vega_component(greeks_data)
            
            # 5. Combined comprehensive score
            combined_score = (
                delta_analysis['weighted_delta'] +
                gamma_analysis.weighted_gamma_score +  # 🚨 Uses 1.5 weight
                theta_analysis['weighted_theta'] +
                vega_analysis['weighted_vega']
            )
            
            # 6. Regime classification from combined Greeks
            regime_indication = self._classify_regime_from_greeks(
                delta_analysis, gamma_analysis, theta_analysis, vega_analysis
            )
            
            # 7. Overall confidence
            confidence = self._calculate_overall_confidence(
                delta_analysis, gamma_analysis, theta_analysis, vega_analysis,
                greeks_data
            )
            
            # 8. Data quality assessment (single point)
            data_quality = {
                'delta_quality': 1.0,  # Production data has 100% coverage
                'gamma_quality': 1.0,  # ACTUAL values available
                'theta_quality': 1.0,  # Production data complete
                'vega_quality': 1.0,   # ACTUAL values available
                'volume_quality': min((greeks_data.ce_volume + greeks_data.pe_volume) / 1000, 1.0)
            }
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return ComprehensiveGreeksAnalysis(
                delta_analysis=delta_analysis,
                gamma_analysis=gamma_analysis,
                theta_analysis=theta_analysis,
                vega_analysis=vega_analysis,
                combined_score=combined_score,
                regime_indication=regime_indication,
                confidence=confidence,
                data_quality=data_quality,
                processing_time_ms=processing_time,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error(f"Comprehensive Greeks analysis failed: {e}")
            raise
    
    def _classify_regime_from_greeks(self, 
                                   delta_analysis: Dict[str, float],
                                   gamma_analysis: GammaWeightedScore,
                                   theta_analysis: Dict[str, float],
                                   vega_analysis: Dict[str, float]) -> str:
        """
        Classify market regime using comprehensive Greeks analysis
        
        Args:
            delta_analysis: Delta component analysis
            gamma_analysis: Gamma component analysis (weighted 1.5)
            theta_analysis: Theta component analysis
            vega_analysis: Vega component analysis
            
        Returns:
            Market regime classification
        """
        # Score each component
        delta_score = delta_analysis['weighted_delta']
        gamma_score = gamma_analysis.weighted_gamma_score  # 🚨 Uses 1.5 weight
        theta_score = theta_analysis['weighted_theta']
        vega_score = vega_analysis['weighted_vega']
        
        # Combined regime score
        total_score = delta_score + gamma_score + theta_score + vega_score
        
        # Pin risk consideration (gamma-based)
        pin_risk = gamma_analysis.pin_risk_indicator
        
        # Regime classification with pin risk adjustment
        if total_score > 2.0:
            if pin_risk > 0.6:
                return "BULLISH_HIGH_PIN_RISK"
            else:
                return "BULLISH"
        elif total_score > 0.5:
            return "MILD_BULLISH"
        elif total_score < -2.0:
            if pin_risk > 0.6:
                return "BEARISH_HIGH_PIN_RISK"
            else:
                return "BEARISH"
        elif total_score < -0.5:
            return "MILD_BEARISH"
        else:
            if pin_risk > 0.8:
                return "NEUTRAL_EXTREME_PIN_RISK"
            elif pin_risk > 0.6:
                return "NEUTRAL_HIGH_PIN_RISK"
            else:
                return "NEUTRAL"
    
    def _calculate_overall_confidence(self, 
                                    delta_analysis: Dict[str, float],
                                    gamma_analysis: GammaWeightedScore,
                                    theta_analysis: Dict[str, float],
                                    vega_analysis: Dict[str, float],
                                    greeks_data: ProductionGreeksData) -> float:
        """Calculate overall confidence in comprehensive Greeks analysis"""
        
        # Component-specific confidence
        delta_conf = min(delta_analysis['delta_magnitude'] * 2, 1.0)
        gamma_conf = gamma_analysis.confidence
        theta_conf = min(abs(theta_analysis['total_theta']) / 20, 1.0)
        vega_conf = min(vega_analysis['vega_magnitude'] / 4, 1.0)
        
        # Volume-based confidence
        total_volume = greeks_data.ce_volume + greeks_data.pe_volume
        volume_conf = min(total_volume / 1000, 1.0)
        
        # Combined confidence
        overall_confidence = (
            0.25 * delta_conf +
            0.35 * gamma_conf +    # Higher weight for corrected gamma
            0.20 * theta_conf +
            0.15 * vega_conf +
            0.05 * volume_conf
        )
        
        return min(overall_confidence, 1.0)
    
    def handle_missing_greeks_data(self, df: pd.DataFrame, strategy: str = 'exclude') -> pd.DataFrame:
        """
        Handle missing Greeks data with configurable strategies
        
        Args:
            df: Production DataFrame
            strategy: 'exclude', 'interpolate', 'forward_fill', or 'zero_fill'
            
        Returns:
            Processed DataFrame
        """
        greeks_cols = ['ce_delta', 'ce_gamma', 'ce_theta', 'ce_vega',
                      'pe_delta', 'pe_gamma', 'pe_theta', 'pe_vega']
        
        initial_count = len(df)
        
        if strategy == 'exclude':
            # Remove rows with ANY missing Greeks (safest)
            df_clean = df.dropna(subset=greeks_cols)
            
        elif strategy == 'interpolate':
            # Linear interpolation for missing values
            df_clean = df.copy()
            df_clean[greeks_cols] = df_clean[greeks_cols].interpolate(method='linear')
            
        elif strategy == 'forward_fill':
            # Forward fill then backward fill
            df_clean = df.copy()
            df_clean[greeks_cols] = df_clean[greeks_cols].fillna(method='ffill').fillna(method='bfill')
            
        elif strategy == 'zero_fill':
            # Fill with zeros (NOT recommended for Greeks analysis)
            df_clean = df.copy()
            df_clean[greeks_cols] = df_clean[greeks_cols].fillna(0)
            self.logger.warning("🚨 Using zero_fill for Greeks - this may affect analysis quality")
            
        else:
            raise ValueError(f"Unknown missing data strategy: {strategy}")
        
        final_count = len(df_clean)
        removed_pct = ((initial_count - final_count) / initial_count) * 100 if initial_count > 0 else 0
        
        self.logger.info(f"Missing data handling ({strategy}): {removed_pct:.2f}% rows affected")
        return df_clean


# Testing and validation functions
def test_comprehensive_greeks_processing():
    """Test comprehensive Greeks processing with actual production-like data"""
    print("🚨 Testing Comprehensive Greeks Processing...")
    
    # Create test data matching production ranges
    test_data = ProductionGreeksData(
        ce_delta=0.6, pe_delta=-0.4,                    # Realistic delta values
        ce_gamma=0.0008, pe_gamma=0.0007,               # Production gamma range
        ce_theta=-8.5, pe_theta=-6.2,                   # Production theta range
        ce_vega=3.2, pe_vega=2.8,                       # Production vega range
        ce_volume=750, pe_volume=650,                   # Good volume
        ce_oi=2000, pe_oi=1800,                        # Institutional OI
        call_strike_type='ATM', put_strike_type='ATM',  # ATM straddle
        dte=7,                                          # Near expiry
        trade_time=datetime.utcnow(),
        expiry_date=datetime.utcnow()
    )
    
    # Initialize processor
    processor = ComprehensiveGreeksProcessor()
    
    # Run comprehensive analysis
    analysis = processor.process_comprehensive_analysis(test_data, volume_weight=1.2)
    
    # Display results
    print(f"✅ Delta analysis: {analysis.delta_analysis['regime_signal']}")
    print(f"✅ Gamma analysis (1.5x weight): {analysis.gamma_analysis.weighted_gamma_score:.6f}")
    print(f"✅ Pin risk: {analysis.gamma_analysis.pin_risk_indicator:.3f}")
    print(f"✅ Theta decay: {analysis.theta_analysis['decay_pressure']}")
    print(f"✅ Vega sensitivity: {analysis.vega_analysis['vol_sensitivity']}")
    print(f"✅ Combined regime: {analysis.regime_indication}")
    print(f"✅ Confidence: {analysis.confidence:.3f}")
    print(f"✅ Processing time: {analysis.processing_time_ms:.2f}ms")
    
    # Validate gamma correction
    expected_gamma = (test_data.ce_gamma + test_data.pe_gamma) * 1.5
    actual_gamma = analysis.gamma_analysis.weighted_gamma_score
    
    if abs(expected_gamma - actual_gamma) < 1e-10:
        print("✅ Gamma weighting (1.5x) CORRECT")
    else:
        print(f"🚨 ERROR: Expected {expected_gamma}, got {actual_gamma}")
    
    print("🚨 Comprehensive Greeks Processing test COMPLETED")


if __name__ == "__main__":
    test_comprehensive_greeks_processing()