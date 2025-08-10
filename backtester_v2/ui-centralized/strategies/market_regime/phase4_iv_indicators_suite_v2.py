#!/usr/bin/env python3
"""
Phase 4 IV Indicators Suite Enhancement V2.0
Market Regime Gaps Implementation V2.0 - Phase 4 Implementation

This module implements the Phase 4 enhancement for IV Indicators Suite with:
1. Comprehensive IV Surface Integration - Enhanced IV surface with regime classification
2. IV-Based Market Fear/Greed Analysis - IV expansion/contraction prediction for regime transitions
3. Dynamic IV Threshold Optimization - Adaptive IV thresholds with confidence scoring

Key Features:
- Enhanced IV surface analysis with 7-level sentiment classification
- IV expansion/contraction prediction for regime transitions
- Dynamic threshold optimization based on VIX and market conditions
- Integration with existing 6×6 correlation framework
- Regime transition logic with hysteresis and stability

Performance Targets:
- IV Surface Analysis: <200ms for complete surface analysis
- Sentiment Classification: >92% accuracy across 7 levels
- Threshold Optimization: <30ms for dynamic adjustment
- Memory Usage: <600MB additional allocation

Author: Senior Quantitative Trading Expert
Date: June 2025
Version: 2.0.4 - Phase 4 IV Indicators Suite Enhancement
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class IVSurfaceConfig:
    """Configuration for comprehensive IV surface integration"""
    surface_regimes: List[str] = None
    sentiment_levels: int = 7  # Extremely bearish to extremely bullish
    percentile_window: int = 252  # 1 year of trading days
    term_structure_points: List[int] = None  # DTE points for term structure

@dataclass
class IVSentimentConfig:
    """Configuration for IV-based market fear/greed analysis"""
    fear_greed_thresholds: Dict[str, float] = None
    expansion_prediction_window: int = 20
    contraction_sensitivity: float = 0.15
    stress_test_scenarios: List[str] = None

@dataclass
class IVThresholdConfig:
    """Configuration for dynamic IV threshold optimization"""
    base_thresholds: Dict[str, float] = None
    vix_adjustment_factors: Dict[str, float] = None
    confidence_scoring_weights: Dict[str, float] = None
    hysteresis_factor: float = 0.1

class ComprehensiveIVSurfaceIntegration:
    """Enhanced IV surface with regime classification"""
    
    def __init__(self, config: IVSurfaceConfig):
        self.config = config
        
        # IV surface regimes
        self.surface_regimes = config.surface_regimes or [
            'normal_skew', 'inverted_skew', 'smile_pattern', 'flat_surface',
            'steep_skew', 'volatility_smile', 'term_structure_inversion'
        ]
        
        # 7-level sentiment classification
        self.sentiment_levels = [
            'extremely_bearish', 'very_bearish', 'bearish', 'neutral',
            'bullish', 'very_bullish', 'extremely_bullish'
        ]
        
        # IV percentile tracking
        self.iv_percentile_history = deque(maxlen=config.percentile_window)
        self.surface_analysis_history = deque(maxlen=500)
        
        # Term structure analysis points
        self.term_structure_points = config.term_structure_points or [7, 14, 30, 60, 90, 180]
        
        # Performance tracking
        self.analysis_times = deque(maxlen=100)
        
        logger.info("ComprehensiveIVSurfaceIntegration initialized")
        logger.info(f"Surface regimes: {len(self.surface_regimes)}")
        logger.info(f"Sentiment levels: {config.sentiment_levels}")
        logger.info(f"Term structure points: {self.term_structure_points}")
    
    def analyze_iv_surface_regime(self, iv_data: Dict[str, Any],
                                market_data: Dict[str, Any],
                                underlying_price: float) -> Dict[str, Any]:
        """Analyze IV surface and classify regime"""
        start_time = time.time()
        
        try:
            # Extract IV data
            call_iv = iv_data.get('call_iv', {})
            put_iv = iv_data.get('put_iv', {})
            
            # Calculate IV surface metrics
            surface_metrics = self._calculate_surface_metrics(call_iv, put_iv, underlying_price)
            
            # Classify surface regime
            surface_regime = self._classify_surface_regime(surface_metrics, market_data)
            
            # Calculate IV percentiles
            iv_percentiles = self._calculate_iv_percentiles(call_iv, put_iv)
            
            # Analyze term structure
            term_structure_analysis = self._analyze_term_structure(iv_data, underlying_price)
            
            # Calculate 7-level sentiment
            sentiment_classification = self._calculate_sentiment_classification(
                surface_metrics, iv_percentiles, term_structure_analysis
            )
            
            analysis_time = time.time() - start_time
            self.analysis_times.append(analysis_time)
            
            # Store analysis record
            analysis_record = {
                'timestamp': datetime.now(),
                'surface_metrics': surface_metrics,
                'surface_regime': surface_regime,
                'sentiment_classification': sentiment_classification,
                'underlying_price': underlying_price
            }
            self.surface_analysis_history.append(analysis_record)
            
            return {
                'surface_metrics': surface_metrics,
                'surface_regime': surface_regime,
                'iv_percentiles': iv_percentiles,
                'term_structure_analysis': term_structure_analysis,
                'sentiment_classification': sentiment_classification,
                'analysis_time_ms': analysis_time * 1000,
                'performance_target_met': analysis_time < 0.2  # <200ms target
            }
            
        except Exception as e:
            logger.error(f"Error analyzing IV surface regime: {e}")
            return {'error': str(e)}
    
    def _calculate_surface_metrics(self, call_iv: Dict, put_iv: Dict, 
                                 underlying_price: float) -> Dict[str, Any]:
        """Calculate comprehensive IV surface metrics"""
        try:
            # Convert strikes to float and sort
            call_strikes = sorted([float(strike) for strike in call_iv.keys()])
            put_strikes = sorted([float(strike) for strike in put_iv.keys()])
            
            # ATM IV calculation
            atm_call_iv = self._get_atm_iv(call_iv, underlying_price, 'call')
            atm_put_iv = self._get_atm_iv(put_iv, underlying_price, 'put')
            atm_iv = (atm_call_iv + atm_put_iv) / 2
            
            # IV skew calculations
            call_skew = self._calculate_iv_skew(call_iv, underlying_price, 'call')
            put_skew = self._calculate_iv_skew(put_iv, underlying_price, 'put')
            
            # IV smile detection
            smile_metrics = self._detect_iv_smile(call_iv, put_iv, underlying_price)
            
            # IV surface curvature
            surface_curvature = self._calculate_surface_curvature(call_iv, put_iv, underlying_price)
            
            # IV range and volatility
            iv_range = self._calculate_iv_range(call_iv, put_iv)
            
            return {
                'atm_iv': float(atm_iv),
                'atm_call_iv': float(atm_call_iv),
                'atm_put_iv': float(atm_put_iv),
                'call_skew': call_skew,
                'put_skew': put_skew,
                'smile_metrics': smile_metrics,
                'surface_curvature': surface_curvature,
                'iv_range': iv_range,
                'iv_spread': float(abs(atm_call_iv - atm_put_iv))
            }
            
        except Exception as e:
            logger.error(f"Error calculating surface metrics: {e}")
            return {}
    
    def _classify_surface_regime(self, surface_metrics: Dict[str, Any],
                               market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Classify IV surface regime"""
        try:
            call_skew = surface_metrics.get('call_skew', {}).get('skew_ratio', 0)
            put_skew = surface_metrics.get('put_skew', {}).get('skew_ratio', 0)
            smile_strength = surface_metrics.get('smile_metrics', {}).get('smile_strength', 0)
            surface_curvature = surface_metrics.get('surface_curvature', {}).get('overall_curvature', 0)
            
            # Regime classification logic
            if abs(call_skew) > 0.3 or abs(put_skew) > 0.3:
                if call_skew < -0.2:
                    regime_type = 'steep_call_skew'
                elif put_skew > 0.2:
                    regime_type = 'steep_put_skew'
                else:
                    regime_type = 'normal_skew'
            elif smile_strength > 0.4:
                regime_type = 'volatility_smile'
            elif abs(surface_curvature) > 0.3:
                regime_type = 'smile_pattern'
            elif abs(call_skew) < 0.1 and abs(put_skew) < 0.1:
                regime_type = 'flat_surface'
            else:
                regime_type = 'normal_skew'
            
            # Regime strength calculation
            regime_strength = max(abs(call_skew), abs(put_skew), smile_strength, abs(surface_curvature))
            
            # Regime confidence
            regime_confidence = min(1.0, regime_strength * 2)  # Scale to 0-1
            
            return {
                'regime_type': regime_type,
                'regime_strength': float(regime_strength),
                'regime_confidence': float(regime_confidence),
                'regime_characteristics': {
                    'call_skew_dominant': abs(call_skew) > abs(put_skew),
                    'smile_present': smile_strength > 0.2,
                    'high_curvature': abs(surface_curvature) > 0.2
                }
            }
            
        except Exception as e:
            logger.error(f"Error classifying surface regime: {e}")
            return {'regime_type': 'normal_skew', 'regime_strength': 0.0}
    
    def _calculate_iv_percentiles(self, call_iv: Dict, put_iv: Dict) -> Dict[str, Any]:
        """Calculate IV percentiles for historical context"""
        try:
            # Current IV levels
            current_ivs = list(call_iv.values()) + list(put_iv.values())
            current_avg_iv = np.mean(current_ivs) if current_ivs else 0.0
            
            # Store in history
            self.iv_percentile_history.append({
                'timestamp': datetime.now(),
                'avg_iv': current_avg_iv,
                'iv_values': current_ivs
            })
            
            if len(self.iv_percentile_history) < 10:
                return {'percentile_rank': 50.0, 'insufficient_history': True}
            
            # Calculate percentile rank
            historical_ivs = [record['avg_iv'] for record in self.iv_percentile_history]
            percentile_rank = (sum(1 for iv in historical_ivs if iv < current_avg_iv) / len(historical_ivs)) * 100
            
            # IV regime classification based on percentiles
            if percentile_rank > 80:
                iv_regime = 'high_iv'
            elif percentile_rank < 20:
                iv_regime = 'low_iv'
            else:
                iv_regime = 'normal_iv'
            
            return {
                'current_avg_iv': float(current_avg_iv),
                'percentile_rank': float(percentile_rank),
                'iv_regime': iv_regime,
                'historical_samples': len(self.iv_percentile_history)
            }
            
        except Exception as e:
            logger.error(f"Error calculating IV percentiles: {e}")
            return {'percentile_rank': 50.0, 'error': str(e)}
    
    def _analyze_term_structure(self, iv_data: Dict[str, Any], 
                              underlying_price: float) -> Dict[str, Any]:
        """Analyze IV term structure"""
        try:
            # Extract term structure data (simplified - would use actual DTE data in production)
            term_structure = {}
            
            # Simulate term structure analysis
            for dte in self.term_structure_points:
                # In production, would extract actual IV for specific DTE
                # For now, simulate based on typical term structure patterns
                base_iv = 0.20  # Base IV
                time_decay_factor = np.sqrt(dte / 30)  # Square root of time
                term_structure[dte] = base_iv * time_decay_factor
            
            # Calculate term structure slope
            short_term_iv = term_structure.get(7, 0.15)
            long_term_iv = term_structure.get(90, 0.25)
            term_structure_slope = (long_term_iv - short_term_iv) / (90 - 7)
            
            # Detect term structure inversion
            inversion_detected = short_term_iv > long_term_iv
            
            # Term structure regime
            if term_structure_slope > 0.002:
                ts_regime = 'normal_contango'
            elif term_structure_slope < -0.001:
                ts_regime = 'backwardation'
            else:
                ts_regime = 'flat_structure'
            
            return {
                'term_structure': term_structure,
                'term_structure_slope': float(term_structure_slope),
                'inversion_detected': inversion_detected,
                'ts_regime': ts_regime,
                'short_term_iv': float(short_term_iv),
                'long_term_iv': float(long_term_iv)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing term structure: {e}")
            return {}
    
    def _calculate_sentiment_classification(self, surface_metrics: Dict[str, Any],
                                          iv_percentiles: Dict[str, Any],
                                          term_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate 7-level sentiment classification"""
        try:
            # Extract key metrics
            atm_iv = surface_metrics.get('atm_iv', 0.2)
            percentile_rank = iv_percentiles.get('percentile_rank', 50.0)
            call_skew = surface_metrics.get('call_skew', {}).get('skew_ratio', 0)
            put_skew = surface_metrics.get('put_skew', {}).get('skew_ratio', 0)
            ts_slope = term_structure.get('term_structure_slope', 0)
            
            # Calculate sentiment score (-3 to +3)
            sentiment_score = 0.0
            
            # IV level contribution
            if percentile_rank > 80:
                sentiment_score -= 1.5  # High IV = bearish
            elif percentile_rank < 20:
                sentiment_score += 1.0   # Low IV = bullish
            
            # Skew contribution
            if put_skew > 0.2:  # Put skew = bearish
                sentiment_score -= 1.0
            elif call_skew < -0.2:  # Call skew = bullish
                sentiment_score += 0.8
            
            # Term structure contribution
            if ts_slope < -0.001:  # Backwardation = bearish
                sentiment_score -= 0.5
            elif ts_slope > 0.002:  # Normal contango = neutral to bullish
                sentiment_score += 0.3
            
            # Classify into 7 levels
            if sentiment_score <= -2.5:
                sentiment_level = 'extremely_bearish'
                sentiment_index = 0
            elif sentiment_score <= -1.5:
                sentiment_level = 'very_bearish'
                sentiment_index = 1
            elif sentiment_score <= -0.5:
                sentiment_level = 'bearish'
                sentiment_index = 2
            elif sentiment_score <= 0.5:
                sentiment_level = 'neutral'
                sentiment_index = 3
            elif sentiment_score <= 1.5:
                sentiment_level = 'bullish'
                sentiment_index = 4
            elif sentiment_score <= 2.5:
                sentiment_level = 'very_bullish'
                sentiment_index = 5
            else:
                sentiment_level = 'extremely_bullish'
                sentiment_index = 6
            
            # Sentiment confidence
            sentiment_confidence = min(1.0, abs(sentiment_score) / 3.0)
            
            return {
                'sentiment_score': float(sentiment_score),
                'sentiment_level': sentiment_level,
                'sentiment_index': sentiment_index,
                'sentiment_confidence': float(sentiment_confidence),
                'contributing_factors': {
                    'iv_percentile_impact': percentile_rank,
                    'skew_impact': max(abs(call_skew), abs(put_skew)),
                    'term_structure_impact': abs(ts_slope)
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating sentiment classification: {e}")
            return {'sentiment_level': 'neutral', 'sentiment_index': 3}
    
    def _get_atm_iv(self, iv_data: Dict, underlying_price: float, option_type: str) -> float:
        """Get ATM implied volatility"""
        try:
            if not iv_data:
                return 0.20  # Default IV
            
            # Find closest strike to ATM
            strikes = [float(strike) for strike in iv_data.keys()]
            closest_strike = min(strikes, key=lambda x: abs(x - underlying_price))
            
            return iv_data.get(str(int(closest_strike)), 0.20)
            
        except Exception as e:
            logger.error(f"Error getting ATM IV: {e}")
            return 0.20
    
    def _calculate_iv_skew(self, iv_data: Dict, underlying_price: float, option_type: str) -> Dict[str, float]:
        """Calculate IV skew metrics"""
        try:
            if len(iv_data) < 3:
                return {'skew_ratio': 0.0, 'skew_strength': 0.0}
            
            # Separate ITM and OTM options
            itm_ivs = []
            otm_ivs = []
            
            for strike, iv in iv_data.items():
                strike_float = float(strike)
                if option_type == 'call':
                    if strike_float < underlying_price:
                        itm_ivs.append(iv)
                    else:
                        otm_ivs.append(iv)
                else:  # put
                    if strike_float > underlying_price:
                        itm_ivs.append(iv)
                    else:
                        otm_ivs.append(iv)
            
            # Calculate skew
            avg_itm_iv = np.mean(itm_ivs) if itm_ivs else 0.20
            avg_otm_iv = np.mean(otm_ivs) if otm_ivs else 0.20
            
            skew_ratio = (avg_otm_iv - avg_itm_iv) / (avg_itm_iv + 1e-8)
            skew_strength = abs(skew_ratio)
            
            return {
                'skew_ratio': float(skew_ratio),
                'skew_strength': float(skew_strength),
                'avg_itm_iv': float(avg_itm_iv),
                'avg_otm_iv': float(avg_otm_iv)
            }
            
        except Exception as e:
            logger.error(f"Error calculating IV skew: {e}")
            return {'skew_ratio': 0.0, 'skew_strength': 0.0}
    
    def _detect_iv_smile(self, call_iv: Dict, put_iv: Dict, underlying_price: float) -> Dict[str, Any]:
        """Detect IV smile patterns"""
        try:
            # Combine call and put IVs
            all_ivs = {}
            for strike, iv in call_iv.items():
                all_ivs[float(strike)] = iv
            for strike, iv in put_iv.items():
                all_ivs[float(strike)] = iv
            
            if len(all_ivs) < 5:
                return {'smile_detected': False, 'smile_strength': 0.0}
            
            # Sort by strike
            sorted_strikes = sorted(all_ivs.keys())
            sorted_ivs = [all_ivs[strike] for strike in sorted_strikes]
            
            # Find ATM index
            atm_index = min(range(len(sorted_strikes)), 
                           key=lambda i: abs(sorted_strikes[i] - underlying_price))
            
            # Check for smile pattern (higher IV at wings)
            if atm_index > 1 and atm_index < len(sorted_ivs) - 2:
                atm_iv = sorted_ivs[atm_index]
                left_wing_iv = np.mean(sorted_ivs[:atm_index-1]) if atm_index > 1 else atm_iv
                right_wing_iv = np.mean(sorted_ivs[atm_index+2:]) if atm_index < len(sorted_ivs)-2 else atm_iv
                
                wing_premium = (left_wing_iv + right_wing_iv) / 2 - atm_iv
                smile_strength = max(0, wing_premium / atm_iv)
                smile_detected = smile_strength > 0.1
            else:
                smile_strength = 0.0
                smile_detected = False
            
            return {
                'smile_detected': smile_detected,
                'smile_strength': float(smile_strength),
                'atm_index': atm_index,
                'wing_premium': float(wing_premium) if 'wing_premium' in locals() else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error detecting IV smile: {e}")
            return {'smile_detected': False, 'smile_strength': 0.0}
    
    def _calculate_surface_curvature(self, call_iv: Dict, put_iv: Dict, 
                                   underlying_price: float) -> Dict[str, Any]:
        """Calculate IV surface curvature"""
        try:
            # Simplified curvature calculation
            all_strikes = sorted(set([float(s) for s in call_iv.keys()] + [float(s) for s in put_iv.keys()]))
            
            if len(all_strikes) < 3:
                return {'overall_curvature': 0.0}
            
            # Calculate second derivative approximation
            curvatures = []
            for i in range(1, len(all_strikes) - 1):
                strike_prev = all_strikes[i-1]
                strike_curr = all_strikes[i]
                strike_next = all_strikes[i+1]
                
                # Get IVs (prefer calls for OTM, puts for ITM)
                iv_prev = call_iv.get(str(int(strike_prev)), put_iv.get(str(int(strike_prev)), 0.2))
                iv_curr = call_iv.get(str(int(strike_curr)), put_iv.get(str(int(strike_curr)), 0.2))
                iv_next = call_iv.get(str(int(strike_next)), put_iv.get(str(int(strike_next)), 0.2))
                
                # Second derivative approximation
                curvature = (iv_next - 2*iv_curr + iv_prev) / ((strike_next - strike_curr) * (strike_curr - strike_prev))
                curvatures.append(curvature)
            
            overall_curvature = np.mean(curvatures) if curvatures else 0.0
            
            return {
                'overall_curvature': float(overall_curvature),
                'max_curvature': float(max(curvatures)) if curvatures else 0.0,
                'min_curvature': float(min(curvatures)) if curvatures else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error calculating surface curvature: {e}")
            return {'overall_curvature': 0.0}
    
    def _calculate_iv_range(self, call_iv: Dict, put_iv: Dict) -> Dict[str, float]:
        """Calculate IV range metrics"""
        try:
            all_ivs = list(call_iv.values()) + list(put_iv.values())
            
            if not all_ivs:
                return {'iv_range': 0.0, 'iv_std': 0.0}
            
            iv_range = max(all_ivs) - min(all_ivs)
            iv_std = np.std(all_ivs)
            iv_mean = np.mean(all_ivs)
            
            return {
                'iv_range': float(iv_range),
                'iv_std': float(iv_std),
                'iv_mean': float(iv_mean),
                'iv_coefficient_variation': float(iv_std / iv_mean) if iv_mean > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error calculating IV range: {e}")
            return {'iv_range': 0.0, 'iv_std': 0.0}

class IVBasedMarketFearGreedAnalysis:
    """IV expansion/contraction prediction for regime transitions"""

    def __init__(self, config: IVSentimentConfig):
        self.config = config

        # Fear/Greed thresholds
        self.fear_greed_thresholds = config.fear_greed_thresholds or {
            'extreme_fear': 0.15,      # IV below 15%
            'fear': 0.20,              # IV below 20%
            'neutral_low': 0.25,       # IV 20-25%
            'neutral_high': 0.35,      # IV 25-35%
            'greed': 0.40,             # IV above 35%
            'extreme_greed': 0.50      # IV above 50%
        }

        # Expansion/contraction tracking
        self.expansion_history = deque(maxlen=config.expansion_prediction_window)
        self.contraction_signals = deque(maxlen=100)

        # Stress test scenarios
        self.stress_scenarios = config.stress_test_scenarios or [
            'vix_spike', 'earnings_event', 'fomc_meeting', 'geopolitical_shock'
        ]

        # Performance tracking
        self.prediction_accuracy = deque(maxlen=200)
        self.analysis_times = deque(maxlen=100)

        logger.info("IVBasedMarketFearGreedAnalysis initialized")
        logger.info(f"Fear/Greed thresholds: {len(self.fear_greed_thresholds)} levels")
        logger.info(f"Expansion prediction window: {config.expansion_prediction_window}")

    def analyze_fear_greed_regime(self, iv_surface_results: Dict[str, Any],
                                market_data: Dict[str, Any],
                                current_vix: float) -> Dict[str, Any]:
        """Analyze IV-based fear/greed regime"""
        start_time = time.time()

        try:
            # Extract IV metrics
            atm_iv = iv_surface_results.get('surface_metrics', {}).get('atm_iv', 0.2)
            iv_percentile = iv_surface_results.get('iv_percentiles', {}).get('percentile_rank', 50.0)

            # Calculate fear/greed level
            fear_greed_level = self._calculate_fear_greed_level(atm_iv, iv_percentile, current_vix)

            # Predict IV expansion/contraction
            expansion_prediction = self._predict_iv_expansion_contraction(
                iv_surface_results, market_data, current_vix
            )

            # Analyze regime transition probability
            regime_transition_analysis = self._analyze_regime_transition_probability(
                fear_greed_level, expansion_prediction, iv_surface_results
            )

            # Stress test analysis
            stress_test_results = self._perform_iv_stress_testing(
                atm_iv, current_vix, market_data
            )

            analysis_time = time.time() - start_time
            self.analysis_times.append(analysis_time)

            # Store expansion record
            expansion_record = {
                'timestamp': datetime.now(),
                'atm_iv': atm_iv,
                'vix': current_vix,
                'fear_greed_level': fear_greed_level,
                'expansion_prediction': expansion_prediction
            }
            self.expansion_history.append(expansion_record)

            return {
                'fear_greed_level': fear_greed_level,
                'expansion_prediction': expansion_prediction,
                'regime_transition_analysis': regime_transition_analysis,
                'stress_test_results': stress_test_results,
                'analysis_time_ms': analysis_time * 1000,
                'iv_regime_strength': self._calculate_iv_regime_strength(fear_greed_level, expansion_prediction)
            }

        except Exception as e:
            logger.error(f"Error analyzing fear/greed regime: {e}")
            return {'error': str(e)}

    def _calculate_fear_greed_level(self, atm_iv: float, iv_percentile: float,
                                  current_vix: float) -> Dict[str, Any]:
        """Calculate fear/greed level from IV metrics"""
        try:
            # Primary classification based on ATM IV
            if atm_iv <= self.fear_greed_thresholds['extreme_fear']:
                primary_level = 'extreme_greed'  # Low IV = complacency/greed
                fear_greed_score = 2.0
            elif atm_iv <= self.fear_greed_thresholds['fear']:
                primary_level = 'greed'
                fear_greed_score = 1.0
            elif atm_iv <= self.fear_greed_thresholds['neutral_low']:
                primary_level = 'neutral_bullish'
                fear_greed_score = 0.5
            elif atm_iv <= self.fear_greed_thresholds['neutral_high']:
                primary_level = 'neutral'
                fear_greed_score = 0.0
            elif atm_iv <= self.fear_greed_thresholds['greed']:
                primary_level = 'fear'
                fear_greed_score = -1.0
            else:
                primary_level = 'extreme_fear'  # High IV = fear
                fear_greed_score = -2.0

            # Adjust based on IV percentile
            if iv_percentile > 80:
                fear_greed_score -= 0.5  # High percentile = more fear
            elif iv_percentile < 20:
                fear_greed_score += 0.5  # Low percentile = more greed

            # VIX confirmation
            vix_fear_factor = max(0, (current_vix - 20) / 20)  # Normalized VIX fear
            fear_greed_score -= vix_fear_factor * 0.3

            # Final classification
            if fear_greed_score >= 1.5:
                final_level = 'extreme_greed'
            elif fear_greed_score >= 0.5:
                final_level = 'greed'
            elif fear_greed_score >= -0.5:
                final_level = 'neutral'
            elif fear_greed_score >= -1.5:
                final_level = 'fear'
            else:
                final_level = 'extreme_fear'

            return {
                'primary_level': primary_level,
                'final_level': final_level,
                'fear_greed_score': float(fear_greed_score),
                'iv_component': float(atm_iv),
                'percentile_component': float(iv_percentile),
                'vix_component': float(vix_fear_factor),
                'regime_intensity': float(abs(fear_greed_score))
            }

        except Exception as e:
            logger.error(f"Error calculating fear/greed level: {e}")
            return {'final_level': 'neutral', 'fear_greed_score': 0.0}

    def _predict_iv_expansion_contraction(self, iv_surface_results: Dict[str, Any],
                                        market_data: Dict[str, Any],
                                        current_vix: float) -> Dict[str, Any]:
        """Predict IV expansion or contraction"""
        try:
            if len(self.expansion_history) < 5:
                return {'prediction': 'insufficient_data', 'confidence': 0.0}

            # Analyze recent IV trends
            recent_ivs = [record['atm_iv'] for record in list(self.expansion_history)[-5:]]
            recent_vix = [record['vix'] for record in list(self.expansion_history)[-5:]]

            # Calculate IV momentum
            iv_momentum = (recent_ivs[-1] - recent_ivs[0]) / len(recent_ivs)
            vix_momentum = (recent_vix[-1] - recent_vix[0]) / len(recent_vix)

            # Term structure analysis
            term_structure = iv_surface_results.get('term_structure_analysis', {})
            ts_slope = term_structure.get('term_structure_slope', 0)

            # Surface regime analysis
            surface_regime = iv_surface_results.get('surface_regime', {})
            regime_strength = surface_regime.get('regime_strength', 0)

            # Prediction logic
            expansion_signals = 0
            contraction_signals = 0

            # IV momentum signals
            if iv_momentum > 0.01:  # Rising IV
                expansion_signals += 1
            elif iv_momentum < -0.01:  # Falling IV
                contraction_signals += 1

            # VIX momentum signals
            if vix_momentum > 1.0:  # Rising VIX
                expansion_signals += 1
            elif vix_momentum < -1.0:  # Falling VIX
                contraction_signals += 1

            # Term structure signals
            if ts_slope < -0.001:  # Backwardation suggests expansion
                expansion_signals += 1
            elif ts_slope > 0.002:  # Normal contango suggests contraction
                contraction_signals += 1

            # Surface regime signals
            if regime_strength > 0.3:  # Strong regime suggests continuation
                if 'fear' in surface_regime.get('regime_type', ''):
                    expansion_signals += 1
                else:
                    contraction_signals += 1

            # Make prediction
            if expansion_signals > contraction_signals:
                prediction = 'expansion'
                confidence = (expansion_signals - contraction_signals) / 4.0
            elif contraction_signals > expansion_signals:
                prediction = 'contraction'
                confidence = (contraction_signals - expansion_signals) / 4.0
            else:
                prediction = 'stable'
                confidence = 0.5

            # Time horizon prediction
            if confidence > 0.7:
                time_horizon = 'short_term'  # 1-3 days
            elif confidence > 0.4:
                time_horizon = 'medium_term'  # 3-7 days
            else:
                time_horizon = 'long_term'  # 1-2 weeks

            return {
                'prediction': prediction,
                'confidence': float(confidence),
                'time_horizon': time_horizon,
                'expansion_signals': expansion_signals,
                'contraction_signals': contraction_signals,
                'contributing_factors': {
                    'iv_momentum': float(iv_momentum),
                    'vix_momentum': float(vix_momentum),
                    'term_structure_slope': float(ts_slope),
                    'regime_strength': float(regime_strength)
                }
            }

        except Exception as e:
            logger.error(f"Error predicting IV expansion/contraction: {e}")
            return {'prediction': 'stable', 'confidence': 0.0}

    def _analyze_regime_transition_probability(self, fear_greed_level: Dict[str, Any],
                                             expansion_prediction: Dict[str, Any],
                                             iv_surface_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze probability of regime transition"""
        try:
            current_level = fear_greed_level.get('final_level', 'neutral')
            regime_intensity = fear_greed_level.get('regime_intensity', 0)
            prediction_confidence = expansion_prediction.get('confidence', 0)

            # Base transition probability
            base_transition_prob = 0.1  # 10% base probability

            # Intensity factor (extreme levels more likely to transition)
            if regime_intensity > 1.5:
                intensity_factor = 0.3
            elif regime_intensity > 1.0:
                intensity_factor = 0.2
            else:
                intensity_factor = 0.1

            # Prediction factor
            prediction_factor = prediction_confidence * 0.2

            # Surface regime factor
            surface_regime = iv_surface_results.get('surface_regime', {})
            regime_confidence = surface_regime.get('regime_confidence', 0)
            surface_factor = (1 - regime_confidence) * 0.15  # Low confidence = higher transition prob

            # Calculate total transition probability
            transition_probability = min(0.8, base_transition_prob + intensity_factor +
                                       prediction_factor + surface_factor)

            # Determine most likely transition direction
            if current_level in ['extreme_fear', 'fear']:
                likely_transition = 'fear_to_neutral'
                transition_catalyst = 'volatility_normalization'
            elif current_level in ['extreme_greed', 'greed']:
                likely_transition = 'greed_to_fear'
                transition_catalyst = 'volatility_spike'
            else:
                likely_transition = 'neutral_to_directional'
                transition_catalyst = 'market_event'

            return {
                'transition_probability': float(transition_probability),
                'likely_transition': likely_transition,
                'transition_catalyst': transition_catalyst,
                'transition_timeframe': 'short_term' if transition_probability > 0.6 else 'medium_term',
                'contributing_factors': {
                    'regime_intensity': float(intensity_factor),
                    'prediction_confidence': float(prediction_factor),
                    'surface_instability': float(surface_factor)
                }
            }

        except Exception as e:
            logger.error(f"Error analyzing regime transition probability: {e}")
            return {'transition_probability': 0.1}

    def _perform_iv_stress_testing(self, atm_iv: float, current_vix: float,
                                 market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform stress testing for IV scenarios"""
        try:
            stress_results = {}

            for scenario in self.stress_scenarios:
                if scenario == 'vix_spike':
                    # Simulate VIX spike to 40+
                    stressed_vix = current_vix * 2.0
                    stressed_iv = atm_iv * 1.5
                    impact_severity = 'high'

                elif scenario == 'earnings_event':
                    # Simulate earnings volatility crush
                    stressed_vix = current_vix * 0.7
                    stressed_iv = atm_iv * 0.6
                    impact_severity = 'medium'

                elif scenario == 'fomc_meeting':
                    # Simulate FOMC uncertainty
                    stressed_vix = current_vix * 1.3
                    stressed_iv = atm_iv * 1.2
                    impact_severity = 'medium'

                elif scenario == 'geopolitical_shock':
                    # Simulate geopolitical crisis
                    stressed_vix = current_vix * 2.5
                    stressed_iv = atm_iv * 1.8
                    impact_severity = 'extreme'

                # Calculate scenario impact
                iv_change_percent = ((stressed_iv - atm_iv) / atm_iv) * 100
                vix_change_percent = ((stressed_vix - current_vix) / current_vix) * 100

                stress_results[scenario] = {
                    'stressed_iv': float(stressed_iv),
                    'stressed_vix': float(stressed_vix),
                    'iv_change_percent': float(iv_change_percent),
                    'vix_change_percent': float(vix_change_percent),
                    'impact_severity': impact_severity,
                    'recovery_timeframe': self._estimate_recovery_time(impact_severity)
                }

            return stress_results

        except Exception as e:
            logger.error(f"Error performing IV stress testing: {e}")
            return {}

    def _calculate_iv_regime_strength(self, fear_greed_level: Dict[str, Any],
                                    expansion_prediction: Dict[str, Any]) -> float:
        """Calculate overall IV regime strength"""
        try:
            regime_intensity = fear_greed_level.get('regime_intensity', 0)
            prediction_confidence = expansion_prediction.get('confidence', 0)

            # Combine factors
            regime_strength = (regime_intensity * 0.6 + prediction_confidence * 0.4)

            return float(min(1.0, regime_strength))

        except Exception as e:
            logger.error(f"Error calculating IV regime strength: {e}")
            return 0.0

    def _estimate_recovery_time(self, impact_severity: str) -> str:
        """Estimate recovery timeframe for stress scenarios"""
        recovery_times = {
            'low': '1-2 days',
            'medium': '3-7 days',
            'high': '1-2 weeks',
            'extreme': '2-4 weeks'
        }
        return recovery_times.get(impact_severity, '1 week')

class DynamicIVThresholdOptimization:
    """Adaptive IV thresholds with confidence scoring"""

    def __init__(self, config: IVThresholdConfig):
        self.config = config

        # Base thresholds
        self.base_thresholds = config.base_thresholds or {
            'low_iv': 0.15,
            'normal_iv_lower': 0.20,
            'normal_iv_upper': 0.30,
            'high_iv': 0.35,
            'extreme_iv': 0.50
        }

        # VIX adjustment factors
        self.vix_adjustment_factors = config.vix_adjustment_factors or {
            'low_vix': {'multiplier': 0.8, 'threshold': 15},
            'normal_vix': {'multiplier': 1.0, 'threshold_range': (15, 25)},
            'high_vix': {'multiplier': 1.3, 'threshold': 25}
        }

        # Confidence scoring weights
        self.confidence_weights = config.confidence_scoring_weights or {
            'historical_accuracy': 0.4,
            'market_regime_consistency': 0.3,
            'volatility_environment': 0.2,
            'time_stability': 0.1
        }

        # Threshold history and performance tracking
        self.threshold_history = deque(maxlen=1000)
        self.performance_tracking = deque(maxlen=500)
        self.optimization_times = deque(maxlen=100)

        # Hysteresis tracking
        self.current_thresholds = self.base_thresholds.copy()
        self.last_adjustment_time = datetime.now()

        logger.info("DynamicIVThresholdOptimization initialized")
        logger.info(f"Base thresholds: {self.base_thresholds}")
        logger.info(f"Hysteresis factor: {config.hysteresis_factor}")

    def optimize_iv_thresholds(self, iv_surface_results: Dict[str, Any],
                             fear_greed_results: Dict[str, Any],
                             market_data: Dict[str, Any],
                             current_vix: float) -> Dict[str, Any]:
        """Optimize IV thresholds dynamically"""
        start_time = time.time()

        try:
            # Calculate VIX-based adjustments
            vix_adjustments = self._calculate_vix_adjustments(current_vix)

            # Calculate market regime adjustments
            regime_adjustments = self._calculate_regime_adjustments(
                iv_surface_results, fear_greed_results
            )

            # Calculate historical performance adjustments
            performance_adjustments = self._calculate_performance_adjustments()

            # Apply hysteresis to prevent oscillation
            hysteresis_adjustments = self._apply_hysteresis(
                vix_adjustments, regime_adjustments, performance_adjustments
            )

            # Calculate optimized thresholds
            optimized_thresholds = self._calculate_optimized_thresholds(
                vix_adjustments, regime_adjustments, performance_adjustments, hysteresis_adjustments
            )

            # Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(
                optimized_thresholds, iv_surface_results, market_data
            )

            # Validate and finalize thresholds
            final_thresholds = self._validate_and_finalize_thresholds(
                optimized_thresholds, confidence_scores
            )

            optimization_time = time.time() - start_time
            self.optimization_times.append(optimization_time)

            # Store optimization record
            optimization_record = {
                'timestamp': datetime.now(),
                'original_thresholds': self.current_thresholds.copy(),
                'optimized_thresholds': final_thresholds,
                'vix': current_vix,
                'confidence_scores': confidence_scores,
                'adjustments_applied': {
                    'vix_adjustments': vix_adjustments,
                    'regime_adjustments': regime_adjustments,
                    'performance_adjustments': performance_adjustments
                }
            }
            self.threshold_history.append(optimization_record)

            # Update current thresholds
            self.current_thresholds = final_thresholds
            self.last_adjustment_time = datetime.now()

            return {
                'optimized_thresholds': final_thresholds,
                'confidence_scores': confidence_scores,
                'threshold_adjustments': {
                    'vix_adjustments': vix_adjustments,
                    'regime_adjustments': regime_adjustments,
                    'performance_adjustments': performance_adjustments,
                    'hysteresis_applied': hysteresis_adjustments
                },
                'optimization_time_ms': optimization_time * 1000,
                'performance_target_met': optimization_time < 0.03,  # <30ms target
                'threshold_stability': self._calculate_threshold_stability()
            }

        except Exception as e:
            logger.error(f"Error optimizing IV thresholds: {e}")
            return {'error': str(e)}

    def _calculate_vix_adjustments(self, current_vix: float) -> Dict[str, float]:
        """Calculate VIX-based threshold adjustments"""
        try:
            # Determine VIX regime
            if current_vix < self.vix_adjustment_factors['low_vix']['threshold']:
                vix_regime = 'low_vix'
                multiplier = self.vix_adjustment_factors['low_vix']['multiplier']
            elif current_vix > self.vix_adjustment_factors['high_vix']['threshold']:
                vix_regime = 'high_vix'
                multiplier = self.vix_adjustment_factors['high_vix']['multiplier']
            else:
                vix_regime = 'normal_vix'
                multiplier = self.vix_adjustment_factors['normal_vix']['multiplier']

            # Calculate adjustments for each threshold
            vix_adjustments = {}
            for threshold_name, base_value in self.base_thresholds.items():
                adjusted_value = base_value * multiplier
                vix_adjustments[threshold_name] = adjusted_value

            return {
                'vix_regime': vix_regime,
                'multiplier': multiplier,
                'adjusted_thresholds': vix_adjustments,
                'vix_level': current_vix
            }

        except Exception as e:
            logger.error(f"Error calculating VIX adjustments: {e}")
            return {'adjusted_thresholds': self.base_thresholds.copy()}

    def _calculate_regime_adjustments(self, iv_surface_results: Dict[str, Any],
                                    fear_greed_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate market regime-based adjustments"""
        try:
            # Extract regime information
            surface_regime = iv_surface_results.get('surface_regime', {}).get('regime_type', 'normal_skew')
            fear_greed_level = fear_greed_results.get('fear_greed_level', {}).get('final_level', 'neutral')
            expansion_prediction = fear_greed_results.get('expansion_prediction', {}).get('prediction', 'stable')

            # Base adjustment factor
            regime_adjustment_factor = 1.0

            # Surface regime adjustments
            if 'steep' in surface_regime:
                regime_adjustment_factor *= 1.1  # Increase thresholds for steep regimes
            elif 'flat' in surface_regime:
                regime_adjustment_factor *= 0.9  # Decrease thresholds for flat regimes

            # Fear/greed adjustments
            if fear_greed_level in ['extreme_fear', 'fear']:
                regime_adjustment_factor *= 1.2  # Higher thresholds in fear regimes
            elif fear_greed_level in ['extreme_greed', 'greed']:
                regime_adjustment_factor *= 0.8  # Lower thresholds in greed regimes

            # Expansion/contraction adjustments
            if expansion_prediction == 'expansion':
                regime_adjustment_factor *= 1.15  # Anticipate higher IV
            elif expansion_prediction == 'contraction':
                regime_adjustment_factor *= 0.85  # Anticipate lower IV

            # Apply adjustments
            regime_adjustments = {}
            for threshold_name, base_value in self.base_thresholds.items():
                regime_adjustments[threshold_name] = base_value * regime_adjustment_factor

            return {
                'regime_adjustment_factor': regime_adjustment_factor,
                'surface_regime': surface_regime,
                'fear_greed_level': fear_greed_level,
                'expansion_prediction': expansion_prediction,
                'adjusted_thresholds': regime_adjustments
            }

        except Exception as e:
            logger.error(f"Error calculating regime adjustments: {e}")
            return {'adjusted_thresholds': self.base_thresholds.copy()}

    def _calculate_performance_adjustments(self) -> Dict[str, Any]:
        """Calculate performance-based threshold adjustments"""
        try:
            if len(self.performance_tracking) < 10:
                return {
                    'performance_factor': 1.0,
                    'adjusted_thresholds': self.base_thresholds.copy(),
                    'insufficient_history': True
                }

            # Calculate recent performance metrics
            recent_performance = list(self.performance_tracking)[-10:]
            accuracy_scores = [p.get('accuracy', 0.5) for p in recent_performance]
            avg_accuracy = np.mean(accuracy_scores)

            # Performance-based adjustment
            if avg_accuracy > 0.8:
                performance_factor = 1.0  # Good performance, no adjustment
            elif avg_accuracy > 0.6:
                performance_factor = 0.95  # Slight tightening
            else:
                performance_factor = 0.9   # More aggressive tightening

            # Apply performance adjustments
            performance_adjustments = {}
            for threshold_name, base_value in self.base_thresholds.items():
                performance_adjustments[threshold_name] = base_value * performance_factor

            return {
                'performance_factor': performance_factor,
                'avg_accuracy': avg_accuracy,
                'adjusted_thresholds': performance_adjustments,
                'performance_samples': len(recent_performance)
            }

        except Exception as e:
            logger.error(f"Error calculating performance adjustments: {e}")
            return {'adjusted_thresholds': self.base_thresholds.copy()}

    def _apply_hysteresis(self, vix_adjustments: Dict[str, Any],
                         regime_adjustments: Dict[str, Any],
                         performance_adjustments: Dict[str, Any]) -> Dict[str, Any]:
        """Apply hysteresis to prevent threshold oscillation"""
        try:
            # Calculate time since last adjustment
            time_since_adjustment = (datetime.now() - self.last_adjustment_time).total_seconds()

            # Hysteresis factor based on time and stability
            if time_since_adjustment < 300:  # Less than 5 minutes
                hysteresis_factor = self.config.hysteresis_factor * 2  # Strong hysteresis
            elif time_since_adjustment < 900:  # Less than 15 minutes
                hysteresis_factor = self.config.hysteresis_factor  # Normal hysteresis
            else:
                hysteresis_factor = self.config.hysteresis_factor * 0.5  # Weak hysteresis

            # Calculate proposed changes
            proposed_thresholds = {}
            for threshold_name in self.base_thresholds.keys():
                vix_value = vix_adjustments.get('adjusted_thresholds', {}).get(threshold_name, self.base_thresholds[threshold_name])
                regime_value = regime_adjustments.get('adjusted_thresholds', {}).get(threshold_name, self.base_thresholds[threshold_name])
                perf_value = performance_adjustments.get('adjusted_thresholds', {}).get(threshold_name, self.base_thresholds[threshold_name])

                # Weighted average of adjustments
                proposed_value = (vix_value * 0.4 + regime_value * 0.4 + perf_value * 0.2)
                proposed_thresholds[threshold_name] = proposed_value

            # Apply hysteresis
            hysteresis_thresholds = {}
            for threshold_name, proposed_value in proposed_thresholds.items():
                current_value = self.current_thresholds.get(threshold_name, self.base_thresholds[threshold_name])
                change_magnitude = abs(proposed_value - current_value) / current_value

                if change_magnitude > hysteresis_factor:
                    # Apply change with dampening
                    dampening_factor = 0.7  # Reduce change by 30%
                    hysteresis_value = current_value + (proposed_value - current_value) * dampening_factor
                else:
                    # No change due to hysteresis
                    hysteresis_value = current_value

                hysteresis_thresholds[threshold_name] = hysteresis_value

            return {
                'hysteresis_factor': hysteresis_factor,
                'time_since_adjustment': time_since_adjustment,
                'proposed_thresholds': proposed_thresholds,
                'hysteresis_thresholds': hysteresis_thresholds,
                'changes_dampened': any(abs(proposed_thresholds[k] - hysteresis_thresholds[k]) > 0.001
                                      for k in proposed_thresholds.keys())
            }

        except Exception as e:
            logger.error(f"Error applying hysteresis: {e}")
            return {'hysteresis_thresholds': self.current_thresholds.copy()}

    def _calculate_optimized_thresholds(self, vix_adjustments: Dict[str, Any],
                                      regime_adjustments: Dict[str, Any],
                                      performance_adjustments: Dict[str, Any],
                                      hysteresis_adjustments: Dict[str, Any]) -> Dict[str, float]:
        """Calculate final optimized thresholds"""
        try:
            # Use hysteresis-adjusted thresholds as base
            optimized_thresholds = hysteresis_adjustments.get('hysteresis_thresholds', self.current_thresholds).copy()

            # Ensure logical ordering of thresholds
            threshold_order = ['low_iv', 'normal_iv_lower', 'normal_iv_upper', 'high_iv', 'extreme_iv']

            # Sort and validate threshold ordering
            for i in range(len(threshold_order) - 1):
                current_threshold = threshold_order[i]
                next_threshold = threshold_order[i + 1]

                if optimized_thresholds[current_threshold] >= optimized_thresholds[next_threshold]:
                    # Fix ordering by adjusting the higher threshold
                    optimized_thresholds[next_threshold] = optimized_thresholds[current_threshold] + 0.01

            # Apply bounds checking
            for threshold_name, value in optimized_thresholds.items():
                # Ensure thresholds stay within reasonable bounds
                optimized_thresholds[threshold_name] = max(0.05, min(1.0, value))

            return optimized_thresholds

        except Exception as e:
            logger.error(f"Error calculating optimized thresholds: {e}")
            return self.current_thresholds.copy()

    def _calculate_confidence_scores(self, optimized_thresholds: Dict[str, float],
                                   iv_surface_results: Dict[str, Any],
                                   market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence scores for optimized thresholds"""
        try:
            confidence_scores = {}

            # Historical accuracy confidence
            if len(self.performance_tracking) >= 5:
                recent_accuracy = np.mean([p.get('accuracy', 0.5) for p in list(self.performance_tracking)[-5:]])
                historical_confidence = recent_accuracy
            else:
                historical_confidence = 0.5

            # Market regime consistency confidence
            surface_regime = iv_surface_results.get('surface_regime', {})
            regime_confidence = surface_regime.get('regime_confidence', 0.5)

            # Volatility environment confidence
            iv_percentiles = iv_surface_results.get('iv_percentiles', {})
            percentile_rank = iv_percentiles.get('percentile_rank', 50.0)
            # Higher confidence when IV is not at extremes
            vol_confidence = 1.0 - abs(percentile_rank - 50.0) / 50.0

            # Time stability confidence
            if len(self.threshold_history) >= 3:
                recent_changes = []
                for i in range(1, min(4, len(self.threshold_history))):
                    prev_thresholds = self.threshold_history[-i-1]['optimized_thresholds']
                    curr_thresholds = self.threshold_history[-i]['optimized_thresholds']

                    change_magnitude = np.mean([
                        abs(curr_thresholds[k] - prev_thresholds[k]) / prev_thresholds[k]
                        for k in curr_thresholds.keys()
                    ])
                    recent_changes.append(change_magnitude)

                avg_change = np.mean(recent_changes)
                time_confidence = max(0.1, 1.0 - avg_change * 10)  # Lower confidence for high volatility
            else:
                time_confidence = 0.5

            # Calculate weighted overall confidence
            overall_confidence = (
                historical_confidence * self.confidence_weights['historical_accuracy'] +
                regime_confidence * self.confidence_weights['market_regime_consistency'] +
                vol_confidence * self.confidence_weights['volatility_environment'] +
                time_confidence * self.confidence_weights['time_stability']
            )

            confidence_scores = {
                'overall_confidence': float(overall_confidence),
                'historical_accuracy': float(historical_confidence),
                'market_regime_consistency': float(regime_confidence),
                'volatility_environment': float(vol_confidence),
                'time_stability': float(time_confidence),
                'confidence_level': 'high' if overall_confidence > 0.8 else
                                  'medium' if overall_confidence > 0.6 else 'low'
            }

            return confidence_scores

        except Exception as e:
            logger.error(f"Error calculating confidence scores: {e}")
            return {'overall_confidence': 0.5}

    def _validate_and_finalize_thresholds(self, optimized_thresholds: Dict[str, float],
                                        confidence_scores: Dict[str, float]) -> Dict[str, float]:
        """Validate and finalize optimized thresholds"""
        try:
            # If confidence is too low, revert to more conservative thresholds
            overall_confidence = confidence_scores.get('overall_confidence', 0.5)

            if overall_confidence < 0.4:
                # Low confidence - blend with base thresholds
                blend_factor = 0.3  # 30% optimized, 70% base
                final_thresholds = {}
                for threshold_name in self.base_thresholds.keys():
                    optimized_value = optimized_thresholds.get(threshold_name, self.base_thresholds[threshold_name])
                    base_value = self.base_thresholds[threshold_name]
                    final_thresholds[threshold_name] = (
                        optimized_value * blend_factor + base_value * (1 - blend_factor)
                    )
            else:
                # High confidence - use optimized thresholds
                final_thresholds = optimized_thresholds.copy()

            return final_thresholds

        except Exception as e:
            logger.error(f"Error validating thresholds: {e}")
            return self.current_thresholds.copy()

    def _calculate_threshold_stability(self) -> Dict[str, Any]:
        """Calculate threshold stability metrics"""
        try:
            if len(self.threshold_history) < 5:
                return {'stability_score': 0.5, 'insufficient_history': True}

            # Calculate threshold volatility over recent history
            recent_history = list(self.threshold_history)[-5:]
            threshold_volatilities = {}

            for threshold_name in self.base_thresholds.keys():
                values = [record['optimized_thresholds'][threshold_name] for record in recent_history]
                volatility = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
                threshold_volatilities[threshold_name] = volatility

            # Overall stability score (lower volatility = higher stability)
            avg_volatility = np.mean(list(threshold_volatilities.values()))
            stability_score = max(0.0, 1.0 - avg_volatility * 10)

            return {
                'stability_score': float(stability_score),
                'threshold_volatilities': {k: float(v) for k, v in threshold_volatilities.items()},
                'avg_volatility': float(avg_volatility),
                'stability_level': 'high' if stability_score > 0.8 else
                                 'medium' if stability_score > 0.6 else 'low'
            }

        except Exception as e:
            logger.error(f"Error calculating threshold stability: {e}")
            return {'stability_score': 0.5}
