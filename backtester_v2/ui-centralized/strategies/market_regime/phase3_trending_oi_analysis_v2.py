#!/usr/bin/env python3
"""
Phase 3 Trending OI Analysis Enhancement V2.0
Market Regime Gaps Implementation V2.0 - Phase 3 Implementation

This module implements the Phase 3 enhancement for Trending OI Analysis with:
1. Advanced OI Flow Analysis - Institutional flow detection with smart money tracking
2. Max Pain and Positioning Analysis - Real-time max pain calculation with positioning insights
3. Multi-Strike OI Regime Formation - Dynamic strike range analysis with regime confirmation

Key Features:
- Enhanced institutional flow detection with volume-weighted analysis
- Real-time max pain calculation with <50ms update frequency
- Multi-strike OI regime formation with ATM ±7 strikes dynamic expansion
- OI-price correlation analysis for trend strength measurement
- Integration with existing production monitoring and alerting

Performance Targets:
- OI Analysis Latency: <150ms for multi-strike analysis
- Institutional Detection Accuracy: >85% classification accuracy
- Max Pain Calculation: <50ms update frequency
- Memory Usage: <800MB additional allocation

Author: Senior Quantitative Trading Expert
Date: June 2025
Version: 2.0.3 - Phase 3 Trending OI Analysis Enhancement
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
class OIFlowConfig:
    """Configuration for advanced OI flow analysis"""
    institutional_threshold: float = 1000000  # $1M threshold for institutional flow
    volume_weight_factor: float = 0.7
    flow_detection_window: int = 20  # 20 data points
    smart_money_confidence: float = 0.8

@dataclass
class MaxPainConfig:
    """Configuration for max pain and positioning analysis"""
    strike_range_multiplier: float = 1.5  # ATM ±1.5x for max pain calculation
    update_frequency_ms: int = 50  # 50ms target update frequency
    positioning_threshold: float = 0.6  # 60% threshold for significant positioning
    gamma_exposure_weight: float = 0.4

@dataclass
class MultiStrikeOIConfig:
    """Configuration for multi-strike OI regime formation"""
    base_strike_range: int = 7  # ATM ±7 strikes
    volatility_expansion_factor: float = 1.3  # Expand range in high volatility
    regime_confirmation_threshold: float = 0.75
    divergence_detection_sensitivity: float = 0.6

class AdvancedOIFlowAnalysis:
    """Enhanced institutional flow detection with smart money tracking"""

    def __init__(self, config: OIFlowConfig):
        self.config = config

        # Flow tracking components
        self.institutional_flows = deque(maxlen=config.flow_detection_window)
        self.retail_flows = deque(maxlen=config.flow_detection_window)
        self.smart_money_indicators = deque(maxlen=100)

        # Flow classification thresholds
        self.flow_thresholds = {
            'large_institutional': config.institutional_threshold * 5,  # $5M+
            'institutional': config.institutional_threshold,            # $1M+
            'retail': config.institutional_threshold * 0.1             # $100K+
        }

        # Volume-weighted analysis
        self.volume_weights = {
            'call_volume': 0.5,
            'put_volume': 0.5,
            'call_oi': 0.3,
            'put_oi': 0.3,
            'net_flow': 0.4
        }

        # Performance tracking
        self.analysis_times = deque(maxlen=50)
        self.detection_accuracy = deque(maxlen=100)

        logger.info("AdvancedOIFlowAnalysis initialized")
        logger.info(f"Institutional threshold: ${config.institutional_threshold:,.0f}")
        logger.info(f"Flow detection window: {config.flow_detection_window} periods")

    def analyze_institutional_flows(self, oi_data: Dict[str, Any],
                                  price_data: Dict[str, float]) -> Dict[str, Any]:
        """Analyze institutional flows with smart money tracking"""
        start_time = time.time()

        try:
            # Extract OI and volume data
            call_oi = oi_data.get('call_oi', {})
            put_oi = oi_data.get('put_oi', {})
            call_volume = oi_data.get('call_volume', {})
            put_volume = oi_data.get('put_volume', {})

            # Calculate institutional flow indicators
            institutional_metrics = self._calculate_institutional_metrics(
                call_oi, put_oi, call_volume, put_volume, price_data
            )

            # Detect smart money activity
            smart_money_signals = self._detect_smart_money_activity(
                institutional_metrics, price_data
            )

            # Calculate flow sentiment
            flow_sentiment = self._calculate_flow_sentiment(
                institutional_metrics, smart_money_signals
            )

            # Volume-weighted analysis
            volume_weighted_metrics = self._calculate_volume_weighted_metrics(
                call_oi, put_oi, call_volume, put_volume
            )

            # Store flow data
            flow_record = {
                'timestamp': datetime.now(),
                'institutional_metrics': institutional_metrics,
                'smart_money_signals': smart_money_signals,
                'flow_sentiment': flow_sentiment
            }

            if flow_sentiment['flow_type'] == 'institutional':
                self.institutional_flows.append(flow_record)
            else:
                self.retail_flows.append(flow_record)

            analysis_time = time.time() - start_time
            self.analysis_times.append(analysis_time)

            return {
                'institutional_metrics': institutional_metrics,
                'smart_money_signals': smart_money_signals,
                'flow_sentiment': flow_sentiment,
                'volume_weighted_metrics': volume_weighted_metrics,
                'flow_classification': self._classify_flow_type(institutional_metrics),
                'analysis_time_ms': analysis_time * 1000,
                'performance_target_met': analysis_time < 0.15  # <150ms target
            }

        except Exception as e:
            logger.error(f"Error analyzing institutional flows: {e}")
            return {'error': str(e)}

    def _calculate_institutional_metrics(self, call_oi: Dict, put_oi: Dict,
                                       call_volume: Dict, put_volume: Dict,
                                       price_data: Dict[str, float]) -> Dict[str, Any]:
        """Calculate institutional flow metrics"""
        try:
            underlying_price = price_data.get('underlying_price', 100.0)

            # Calculate total notional values
            total_call_notional = sum(
                oi * underlying_price * 100 for oi in call_oi.values()  # 100 shares per contract
            )
            total_put_notional = sum(
                oi * underlying_price * 100 for oi in put_oi.values()
            )

            # Calculate volume-weighted average prices
            call_vwap = self._calculate_vwap(call_volume, price_data, 'call')
            put_vwap = self._calculate_vwap(put_volume, price_data, 'put')

            # Calculate flow concentration (Herfindahl index)
            call_concentration = self._calculate_concentration_index(call_oi)
            put_concentration = self._calculate_concentration_index(put_oi)

            # Calculate net flow
            net_call_flow = sum(call_volume.values()) - sum(call_oi.values())
            net_put_flow = sum(put_volume.values()) - sum(put_oi.values())

            return {
                'total_call_notional': total_call_notional,
                'total_put_notional': total_put_notional,
                'net_notional': total_call_notional - total_put_notional,
                'call_vwap': call_vwap,
                'put_vwap': put_vwap,
                'call_concentration': call_concentration,
                'put_concentration': put_concentration,
                'net_call_flow': net_call_flow,
                'net_put_flow': net_put_flow,
                'flow_ratio': (net_call_flow / (net_put_flow + 1e-8)) if net_put_flow != 0 else 0
            }

        except Exception as e:
            logger.error(f"Error calculating institutional metrics: {e}")
            return {}

    def _detect_smart_money_activity(self, institutional_metrics: Dict[str, Any],
                                   price_data: Dict[str, float]) -> Dict[str, Any]:
        """Detect smart money activity patterns"""
        try:
            net_notional = institutional_metrics.get('net_notional', 0)
            flow_ratio = institutional_metrics.get('flow_ratio', 0)
            call_concentration = institutional_metrics.get('call_concentration', 0)
            put_concentration = institutional_metrics.get('put_concentration', 0)

            # Smart money indicators
            large_position_indicator = abs(net_notional) > self.flow_thresholds['large_institutional']
            concentrated_flow_indicator = max(call_concentration, put_concentration) > 0.7
            contrarian_indicator = (flow_ratio > 2.0 and net_notional < 0) or (flow_ratio < 0.5 and net_notional > 0)

            # Calculate smart money confidence
            confidence_factors = [
                large_position_indicator,
                concentrated_flow_indicator,
                contrarian_indicator
            ]
            smart_money_confidence = sum(confidence_factors) / len(confidence_factors)

            # Determine smart money direction
            if smart_money_confidence > self.config.smart_money_confidence:
                if net_notional > 0:
                    smart_money_direction = 'bullish'
                else:
                    smart_money_direction = 'bearish'
            else:
                smart_money_direction = 'neutral'

            smart_money_signal = {
                'confidence': smart_money_confidence,
                'direction': smart_money_direction,
                'large_position': large_position_indicator,
                'concentrated_flow': concentrated_flow_indicator,
                'contrarian_signal': contrarian_indicator,
                'signal_strength': smart_money_confidence * abs(net_notional) / self.flow_thresholds['institutional']
            }

            # Store in history
            self.smart_money_indicators.append({
                'timestamp': datetime.now(),
                'signal': smart_money_signal,
                'metrics': institutional_metrics
            })

            return smart_money_signal

        except Exception as e:
            logger.error(f"Error detecting smart money activity: {e}")
            return {}

    def _calculate_flow_sentiment(self, institutional_metrics: Dict[str, Any],
                                smart_money_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall flow sentiment"""
        try:
            net_notional = institutional_metrics.get('net_notional', 0)
            flow_ratio = institutional_metrics.get('flow_ratio', 0)
            smart_money_confidence = smart_money_signals.get('confidence', 0)

            # Determine flow type
            if abs(net_notional) > self.flow_thresholds['institutional']:
                flow_type = 'institutional'
            elif abs(net_notional) > self.flow_thresholds['retail']:
                flow_type = 'retail'
            else:
                flow_type = 'minimal'

            # Calculate sentiment score (-1 to 1)
            sentiment_score = np.tanh(net_notional / self.flow_thresholds['institutional'])

            # Adjust sentiment based on smart money signals
            if smart_money_confidence > 0.7:
                smart_money_direction = smart_money_signals.get('direction', 'neutral')
                if smart_money_direction == 'bullish':
                    sentiment_score = max(sentiment_score, 0.3)
                elif smart_money_direction == 'bearish':
                    sentiment_score = min(sentiment_score, -0.3)

            # Sentiment classification
            if sentiment_score > 0.5:
                sentiment_class = 'strongly_bullish'
            elif sentiment_score > 0.2:
                sentiment_class = 'bullish'
            elif sentiment_score > -0.2:
                sentiment_class = 'neutral'
            elif sentiment_score > -0.5:
                sentiment_class = 'bearish'
            else:
                sentiment_class = 'strongly_bearish'

            return {
                'sentiment_score': float(sentiment_score),
                'sentiment_class': sentiment_class,
                'flow_type': flow_type,
                'confidence': float(smart_money_confidence),
                'flow_strength': float(abs(sentiment_score))
            }

        except Exception as e:
            logger.error(f"Error calculating flow sentiment: {e}")
            return {}

    def _calculate_volume_weighted_metrics(self, call_oi: Dict, put_oi: Dict,
                                         call_volume: Dict, put_volume: Dict) -> Dict[str, Any]:
        """Calculate volume-weighted OI metrics"""
        try:
            # Volume-weighted OI ratios
            total_call_volume = sum(call_volume.values())
            total_put_volume = sum(put_volume.values())
            total_call_oi = sum(call_oi.values())
            total_put_oi = sum(put_oi.values())

            # Volume-to-OI ratios
            call_volume_oi_ratio = total_call_volume / (total_call_oi + 1e-8)
            put_volume_oi_ratio = total_put_volume / (total_put_oi + 1e-8)

            # Put-call ratios
            pc_volume_ratio = total_put_volume / (total_call_volume + 1e-8)
            pc_oi_ratio = total_put_oi / (total_call_oi + 1e-8)

            return {
                'call_volume_oi_ratio': float(call_volume_oi_ratio),
                'put_volume_oi_ratio': float(put_volume_oi_ratio),
                'pc_volume_ratio': float(pc_volume_ratio),
                'pc_oi_ratio': float(pc_oi_ratio),
                'total_volume': total_call_volume + total_put_volume,
                'total_oi': total_call_oi + total_put_oi,
                'volume_oi_divergence': abs(call_volume_oi_ratio - put_volume_oi_ratio)
            }

        except Exception as e:
            logger.error(f"Error calculating volume-weighted metrics: {e}")
            return {}

    def _calculate_vwap(self, volume_data: Dict, price_data: Dict, option_type: str) -> float:
        """Calculate volume-weighted average price"""
        try:
            total_volume = 0
            total_value = 0

            for strike, volume in volume_data.items():
                if volume > 0:
                    # Get option price (simplified)
                    option_price = price_data.get(f'{option_type}_{strike}', 1.0)
                    total_value += volume * option_price
                    total_volume += volume

            return total_value / total_volume if total_volume > 0 else 0.0

        except Exception as e:
            logger.error(f"Error calculating VWAP: {e}")
            return 0.0

    def _calculate_concentration_index(self, oi_data: Dict) -> float:
        """Calculate Herfindahl concentration index for OI"""
        try:
            total_oi = sum(oi_data.values())
            if total_oi == 0:
                return 0.0

            # Herfindahl index: sum of squared market shares
            concentration = sum((oi / total_oi) ** 2 for oi in oi_data.values())
            return concentration

        except Exception as e:
            logger.error(f"Error calculating concentration index: {e}")
            return 0.0

    def _classify_flow_type(self, institutional_metrics: Dict[str, Any]) -> str:
        """Classify the type of flow based on metrics"""
        net_notional = abs(institutional_metrics.get('net_notional', 0))

        if net_notional > self.flow_thresholds['large_institutional']:
            return 'large_institutional'
        elif net_notional > self.flow_thresholds['institutional']:
            return 'institutional'
        elif net_notional > self.flow_thresholds['retail']:
            return 'retail'
        else:
            return 'minimal'

class MaxPainPositioningAnalysis:
    """Real-time max pain calculation with positioning insights"""

    def __init__(self, config: MaxPainConfig):
        self.config = config

        # Max pain tracking
        self.max_pain_history = deque(maxlen=1000)
        self.positioning_analysis = deque(maxlen=500)

        # Performance tracking
        self.calculation_times = deque(maxlen=100)

        # Gamma exposure tracking
        self.gamma_exposure_levels = {
            'extreme_negative': -1000000,  # -$1M gamma exposure
            'high_negative': -500000,      # -$500K gamma exposure
            'neutral': 0,
            'high_positive': 500000,       # $500K gamma exposure
            'extreme_positive': 1000000    # $1M gamma exposure
        }

        logger.info("MaxPainPositioningAnalysis initialized")
        logger.info(f"Update frequency target: {config.update_frequency_ms}ms")
        logger.info(f"Positioning threshold: {config.positioning_threshold}")

    def calculate_max_pain_analysis(self, oi_data: Dict[str, Any],
                                  price_data: Dict[str, float],
                                  underlying_price: float) -> Dict[str, Any]:
        """Calculate comprehensive max pain analysis"""
        start_time = time.time()

        try:
            # Extract OI data
            call_oi = oi_data.get('call_oi', {})
            put_oi = oi_data.get('put_oi', {})

            # Calculate max pain level
            max_pain_level = self._calculate_max_pain_level(call_oi, put_oi, underlying_price)

            # Calculate positioning metrics
            positioning_metrics = self._calculate_positioning_metrics(
                call_oi, put_oi, underlying_price, max_pain_level
            )

            # Calculate gamma exposure
            gamma_exposure = self._calculate_gamma_exposure(
                call_oi, put_oi, price_data, underlying_price
            )

            # Analyze OI-price correlation
            oi_price_correlation = self._analyze_oi_price_correlation(
                call_oi, put_oi, underlying_price
            )

            # Calculate distance-based regime modification
            distance_metrics = self._calculate_distance_metrics(
                max_pain_level, underlying_price, positioning_metrics
            )

            calculation_time = time.time() - start_time
            self.calculation_times.append(calculation_time)

            # Store in history
            analysis_record = {
                'timestamp': datetime.now(),
                'max_pain_level': max_pain_level,
                'underlying_price': underlying_price,
                'positioning_metrics': positioning_metrics,
                'gamma_exposure': gamma_exposure,
                'distance_metrics': distance_metrics
            }
            self.max_pain_history.append(analysis_record)

            return {
                'max_pain_level': max_pain_level,
                'positioning_metrics': positioning_metrics,
                'gamma_exposure': gamma_exposure,
                'oi_price_correlation': oi_price_correlation,
                'distance_metrics': distance_metrics,
                'calculation_time_ms': calculation_time * 1000,
                'performance_target_met': calculation_time < 0.05,  # <50ms target
                'max_pain_distance': abs(underlying_price - max_pain_level),
                'max_pain_direction': 'above' if underlying_price > max_pain_level else 'below'
            }

        except Exception as e:
            logger.error(f"Error calculating max pain analysis: {e}")
            return {'error': str(e)}

    def _calculate_max_pain_level(self, call_oi: Dict, put_oi: Dict,
                                underlying_price: float) -> float:
        """Calculate max pain level using traditional method"""
        try:
            # Get all available strikes
            all_strikes = set(call_oi.keys()) | set(put_oi.keys())
            if not all_strikes:
                return underlying_price

            # Convert strikes to float and sort
            strikes = sorted([float(strike) for strike in all_strikes])

            # Calculate total pain for each strike
            min_pain = float('inf')
            max_pain_strike = underlying_price

            for strike in strikes:
                total_pain = 0

                # Calculate call pain
                for call_strike, call_oi_value in call_oi.items():
                    call_strike_float = float(call_strike)
                    if strike > call_strike_float:
                        total_pain += (strike - call_strike_float) * call_oi_value

                # Calculate put pain
                for put_strike, put_oi_value in put_oi.items():
                    put_strike_float = float(put_strike)
                    if strike < put_strike_float:
                        total_pain += (put_strike_float - strike) * put_oi_value

                # Track minimum pain
                if total_pain < min_pain:
                    min_pain = total_pain
                    max_pain_strike = strike

            return max_pain_strike

        except Exception as e:
            logger.error(f"Error calculating max pain level: {e}")
            return underlying_price

    def _calculate_positioning_metrics(self, call_oi: Dict, put_oi: Dict,
                                     underlying_price: float, max_pain_level: float) -> Dict[str, Any]:
        """Calculate institutional positioning metrics"""
        try:
            # Calculate total OI
            total_call_oi = sum(call_oi.values())
            total_put_oi = sum(put_oi.values())
            total_oi = total_call_oi + total_put_oi

            # Calculate OI concentration around max pain
            max_pain_range = underlying_price * 0.05  # ±5% range
            concentrated_oi = 0

            for strike, oi in {**call_oi, **put_oi}.items():
                strike_float = float(strike)
                if abs(strike_float - max_pain_level) <= max_pain_range:
                    concentrated_oi += oi

            concentration_ratio = concentrated_oi / total_oi if total_oi > 0 else 0

            # Calculate put-call OI ratio
            pc_oi_ratio = total_put_oi / (total_call_oi + 1e-8)

            # Calculate skew metrics
            atm_call_oi = call_oi.get(str(int(underlying_price)), 0)
            atm_put_oi = put_oi.get(str(int(underlying_price)), 0)

            # OI skew (ITM vs OTM)
            itm_call_oi = sum(oi for strike, oi in call_oi.items() if float(strike) < underlying_price)
            otm_call_oi = sum(oi for strike, oi in call_oi.items() if float(strike) > underlying_price)
            itm_put_oi = sum(oi for strike, oi in put_oi.items() if float(strike) > underlying_price)
            otm_put_oi = sum(oi for strike, oi in put_oi.items() if float(strike) < underlying_price)

            call_skew = (otm_call_oi - itm_call_oi) / (otm_call_oi + itm_call_oi + 1e-8)
            put_skew = (itm_put_oi - otm_put_oi) / (itm_put_oi + otm_put_oi + 1e-8)

            return {
                'total_oi': total_oi,
                'pc_oi_ratio': float(pc_oi_ratio),
                'concentration_ratio': float(concentration_ratio),
                'max_pain_concentration': concentration_ratio > self.config.positioning_threshold,
                'call_skew': float(call_skew),
                'put_skew': float(put_skew),
                'atm_call_oi': atm_call_oi,
                'atm_put_oi': atm_put_oi,
                'positioning_strength': float(concentration_ratio * abs(pc_oi_ratio - 1.0))
            }

        except Exception as e:
            logger.error(f"Error calculating positioning metrics: {e}")
            return {}

    def _calculate_gamma_exposure(self, call_oi: Dict, put_oi: Dict,
                                price_data: Dict[str, float], underlying_price: float) -> Dict[str, Any]:
        """Calculate gamma exposure levels"""
        try:
            total_gamma_exposure = 0
            call_gamma_exposure = 0
            put_gamma_exposure = 0

            # Simplified gamma calculation (would use actual Greeks in production)
            for strike, oi in call_oi.items():
                strike_float = float(strike)
                # Simplified gamma approximation
                gamma_approx = max(0, 1 - abs(strike_float - underlying_price) / underlying_price)
                call_gamma_exposure += oi * gamma_approx * underlying_price * 100  # $100 per point

            for strike, oi in put_oi.items():
                strike_float = float(strike)
                # Simplified gamma approximation (negative for puts)
                gamma_approx = max(0, 1 - abs(strike_float - underlying_price) / underlying_price)
                put_gamma_exposure -= oi * gamma_approx * underlying_price * 100  # Negative for puts

            total_gamma_exposure = call_gamma_exposure + put_gamma_exposure

            # Classify gamma exposure level
            gamma_level = 'neutral'
            for level, threshold in self.gamma_exposure_levels.items():
                if total_gamma_exposure <= threshold:
                    gamma_level = level
                    break

            return {
                'total_gamma_exposure': float(total_gamma_exposure),
                'call_gamma_exposure': float(call_gamma_exposure),
                'put_gamma_exposure': float(put_gamma_exposure),
                'gamma_level': gamma_level,
                'gamma_imbalance': float(abs(call_gamma_exposure + put_gamma_exposure)),
                'gamma_weighted_exposure': float(total_gamma_exposure * self.config.gamma_exposure_weight)
            }

        except Exception as e:
            logger.error(f"Error calculating gamma exposure: {e}")
            return {}

    def _analyze_oi_price_correlation(self, call_oi: Dict, put_oi: Dict,
                                    underlying_price: float) -> Dict[str, Any]:
        """Analyze OI-price correlation for trend strength"""
        try:
            if len(self.max_pain_history) < 10:
                return {'correlation': 0.0, 'trend_strength': 'insufficient_data'}

            # Extract recent price and OI data
            recent_data = list(self.max_pain_history)[-10:]
            prices = [record['underlying_price'] for record in recent_data]
            max_pains = [record['max_pain_level'] for record in recent_data]

            # Calculate correlation
            price_changes = np.diff(prices)
            max_pain_changes = np.diff(max_pains)

            if len(price_changes) > 1:
                correlation = np.corrcoef(price_changes, max_pain_changes)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
            else:
                correlation = 0.0

            # Determine trend strength
            if abs(correlation) > 0.7:
                trend_strength = 'strong'
            elif abs(correlation) > 0.4:
                trend_strength = 'moderate'
            else:
                trend_strength = 'weak'

            return {
                'correlation': float(correlation),
                'trend_strength': trend_strength,
                'correlation_direction': 'positive' if correlation > 0 else 'negative',
                'trend_confidence': float(abs(correlation))
            }

        except Exception as e:
            logger.error(f"Error analyzing OI-price correlation: {e}")
            return {'correlation': 0.0, 'trend_strength': 'error'}

    def _calculate_distance_metrics(self, max_pain_level: float, underlying_price: float,
                                  positioning_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate distance-based regime modification metrics"""
        try:
            # Distance calculations
            absolute_distance = abs(underlying_price - max_pain_level)
            relative_distance = absolute_distance / underlying_price

            # Distance-based regime signals
            close_to_max_pain = relative_distance < 0.02  # Within 2%
            far_from_max_pain = relative_distance > 0.05  # Beyond 5%

            # Positioning strength factor
            positioning_strength = positioning_metrics.get('positioning_strength', 0)

            # Regime modification strength
            if close_to_max_pain and positioning_strength > 0.6:
                regime_modification = 'strong_gravitational_pull'
            elif close_to_max_pain:
                regime_modification = 'moderate_gravitational_pull'
            elif far_from_max_pain and positioning_strength > 0.6:
                regime_modification = 'strong_breakout_potential'
            elif far_from_max_pain:
                regime_modification = 'moderate_breakout_potential'
            else:
                regime_modification = 'neutral'

            return {
                'absolute_distance': float(absolute_distance),
                'relative_distance': float(relative_distance),
                'close_to_max_pain': close_to_max_pain,
                'far_from_max_pain': far_from_max_pain,
                'regime_modification': regime_modification,
                'gravitational_strength': float(1.0 / (1.0 + relative_distance))  # Inverse distance
            }

        except Exception as e:
            logger.error(f"Error calculating distance metrics: {e}")
            return {}

class MultiStrikeOIRegimeFormation:
    """Dynamic strike range analysis with regime confirmation"""

    def __init__(self, config: MultiStrikeOIConfig):
        self.config = config

        # Strike range management
        self.current_strike_range = config.base_strike_range
        self.strike_range_history = deque(maxlen=100)

        # Regime formation tracking
        self.regime_signals = deque(maxlen=200)
        self.divergence_signals = deque(maxlen=150)

        # Performance tracking
        self.analysis_times = deque(maxlen=50)

        logger.info("MultiStrikeOIRegimeFormation initialized")
        logger.info(f"Base strike range: ATM ±{config.base_strike_range}")
        logger.info(f"Volatility expansion factor: {config.volatility_expansion_factor}")

    def analyze_multi_strike_regime(self, oi_data: Dict[str, Any],
                                  market_data: Dict[str, Any],
                                  underlying_price: float,
                                  current_vix: float) -> Dict[str, Any]:
        """Analyze multi-strike OI regime formation"""
        start_time = time.time()

        try:
            # Calculate dynamic strike range
            dynamic_range = self._calculate_dynamic_strike_range(current_vix, underlying_price)

            # Extract relevant strikes
            relevant_strikes = self._extract_relevant_strikes(
                oi_data, underlying_price, dynamic_range
            )

            # Analyze OI distribution patterns
            oi_distribution = self._analyze_oi_distribution(
                relevant_strikes, underlying_price
            )

            # Detect regime formation signals
            regime_formation_signals = self._detect_regime_formation_signals(
                oi_distribution, market_data
            )

            # Analyze OI divergence
            divergence_analysis = self._analyze_oi_divergence(
                relevant_strikes, underlying_price, market_data
            )

            # Calculate regime confirmation
            regime_confirmation = self._calculate_regime_confirmation(
                regime_formation_signals, divergence_analysis, oi_distribution
            )

            analysis_time = time.time() - start_time
            self.analysis_times.append(analysis_time)

            # Store analysis record
            analysis_record = {
                'timestamp': datetime.now(),
                'dynamic_range': dynamic_range,
                'regime_formation_signals': regime_formation_signals,
                'regime_confirmation': regime_confirmation,
                'underlying_price': underlying_price,
                'vix': current_vix
            }
            self.regime_signals.append(analysis_record)

            return {
                'dynamic_strike_range': dynamic_range,
                'relevant_strikes': relevant_strikes,
                'oi_distribution': oi_distribution,
                'regime_formation_signals': regime_formation_signals,
                'divergence_analysis': divergence_analysis,
                'regime_confirmation': regime_confirmation,
                'analysis_time_ms': analysis_time * 1000,
                'performance_target_met': analysis_time < 0.15  # <150ms target
            }

        except Exception as e:
            logger.error(f"Error analyzing multi-strike regime: {e}")
            return {'error': str(e)}

    def _calculate_dynamic_strike_range(self, vix: float, underlying_price: float) -> Dict[str, Any]:
        """Calculate dynamic strike range based on volatility"""
        try:
            # Base range
            base_range = self.config.base_strike_range

            # Volatility-based expansion
            if vix > 25:  # High volatility
                expansion_factor = self.config.volatility_expansion_factor
            elif vix < 15:  # Low volatility
                expansion_factor = 1.0 / self.config.volatility_expansion_factor
            else:  # Normal volatility
                expansion_factor = 1.0

            # Calculate expanded range
            expanded_range = int(base_range * expansion_factor)
            expanded_range = max(5, min(15, expanded_range))  # Constrain between 5-15

            # Calculate strike boundaries
            strike_interval = 50 if underlying_price > 1000 else 25 if underlying_price > 500 else 10
            atm_strike = round(underlying_price / strike_interval) * strike_interval

            lower_bound = atm_strike - (expanded_range * strike_interval)
            upper_bound = atm_strike + (expanded_range * strike_interval)

            self.current_strike_range = expanded_range

            range_info = {
                'base_range': base_range,
                'expanded_range': expanded_range,
                'expansion_factor': expansion_factor,
                'atm_strike': atm_strike,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'strike_interval': strike_interval,
                'total_strikes': expanded_range * 2 + 1
            }

            self.strike_range_history.append({
                'timestamp': datetime.now(),
                'range_info': range_info,
                'vix': vix,
                'underlying_price': underlying_price
            })

            return range_info

        except Exception as e:
            logger.error(f"Error calculating dynamic strike range: {e}")
            return {'expanded_range': self.config.base_strike_range}

    def _extract_relevant_strikes(self, oi_data: Dict[str, Any],
                                underlying_price: float, dynamic_range: Dict[str, Any]) -> Dict[str, Any]:
        """Extract OI data for relevant strikes"""
        try:
            call_oi = oi_data.get('call_oi', {})
            put_oi = oi_data.get('put_oi', {})
            call_volume = oi_data.get('call_volume', {})
            put_volume = oi_data.get('put_volume', {})

            lower_bound = dynamic_range.get('lower_bound', underlying_price - 100)
            upper_bound = dynamic_range.get('upper_bound', underlying_price + 100)

            # Filter strikes within range
            relevant_call_oi = {
                strike: oi for strike, oi in call_oi.items()
                if lower_bound <= float(strike) <= upper_bound
            }
            relevant_put_oi = {
                strike: oi for strike, oi in put_oi.items()
                if lower_bound <= float(strike) <= upper_bound
            }
            relevant_call_volume = {
                strike: vol for strike, vol in call_volume.items()
                if lower_bound <= float(strike) <= upper_bound
            }
            relevant_put_volume = {
                strike: vol for strike, vol in put_volume.items()
                if lower_bound <= float(strike) <= upper_bound
            }

            return {
                'call_oi': relevant_call_oi,
                'put_oi': relevant_put_oi,
                'call_volume': relevant_call_volume,
                'put_volume': relevant_put_volume,
                'strike_count': len(set(relevant_call_oi.keys()) | set(relevant_put_oi.keys())),
                'range_bounds': (lower_bound, upper_bound)
            }

        except Exception as e:
            logger.error(f"Error extracting relevant strikes: {e}")
            return {}

    def _analyze_oi_distribution(self, relevant_strikes: Dict[str, Any],
                               underlying_price: float) -> Dict[str, Any]:
        """Analyze OI distribution patterns"""
        try:
            call_oi = relevant_strikes.get('call_oi', {})
            put_oi = relevant_strikes.get('put_oi', {})

            # Calculate distribution metrics
            total_call_oi = sum(call_oi.values())
            total_put_oi = sum(put_oi.values())

            # ATM concentration
            atm_strikes = [strike for strike in call_oi.keys()
                          if abs(float(strike) - underlying_price) <= 25]
            atm_call_oi = sum(call_oi.get(strike, 0) for strike in atm_strikes)
            atm_put_oi = sum(put_oi.get(strike, 0) for strike in atm_strikes)

            atm_concentration = (atm_call_oi + atm_put_oi) / (total_call_oi + total_put_oi + 1e-8)

            # Wing concentration (OTM options)
            otm_call_strikes = [strike for strike in call_oi.keys()
                               if float(strike) > underlying_price + 50]
            otm_put_strikes = [strike for strike in put_oi.keys()
                              if float(strike) < underlying_price - 50]

            wing_call_oi = sum(call_oi.get(strike, 0) for strike in otm_call_strikes)
            wing_put_oi = sum(put_oi.get(strike, 0) for strike in otm_put_strikes)
            wing_concentration = (wing_call_oi + wing_put_oi) / (total_call_oi + total_put_oi + 1e-8)

            # Skew analysis
            call_skew = self._calculate_strike_skew(call_oi, underlying_price, 'call')
            put_skew = self._calculate_strike_skew(put_oi, underlying_price, 'put')

            # Distribution shape
            distribution_shape = self._classify_distribution_shape(
                atm_concentration, wing_concentration, call_skew, put_skew
            )

            return {
                'total_call_oi': total_call_oi,
                'total_put_oi': total_put_oi,
                'atm_concentration': float(atm_concentration),
                'wing_concentration': float(wing_concentration),
                'call_skew': call_skew,
                'put_skew': put_skew,
                'distribution_shape': distribution_shape,
                'pc_oi_ratio': total_put_oi / (total_call_oi + 1e-8)
            }

        except Exception as e:
            logger.error(f"Error analyzing OI distribution: {e}")
            return {}

    def _detect_regime_formation_signals(self, oi_distribution: Dict[str, Any],
                                       market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect regime formation signals from OI patterns"""
        try:
            atm_concentration = oi_distribution.get('atm_concentration', 0)
            wing_concentration = oi_distribution.get('wing_concentration', 0)
            pc_oi_ratio = oi_distribution.get('pc_oi_ratio', 1)
            distribution_shape = oi_distribution.get('distribution_shape', 'neutral')

            # Signal detection
            signals = {}

            # Consolidation signal (high ATM concentration)
            signals['consolidation'] = atm_concentration > 0.6

            # Breakout signal (high wing concentration)
            signals['breakout_potential'] = wing_concentration > 0.4

            # Directional bias signal
            if pc_oi_ratio > 1.3:
                signals['directional_bias'] = 'bearish'
            elif pc_oi_ratio < 0.7:
                signals['directional_bias'] = 'bullish'
            else:
                signals['directional_bias'] = 'neutral'

            # Volatility regime signal
            if distribution_shape == 'concentrated':
                signals['volatility_regime'] = 'low_vol_expected'
            elif distribution_shape == 'dispersed':
                signals['volatility_regime'] = 'high_vol_expected'
            else:
                signals['volatility_regime'] = 'neutral'

            # Regime strength
            regime_strength = (
                (atm_concentration if signals['consolidation'] else wing_concentration) * 0.4 +
                abs(pc_oi_ratio - 1.0) * 0.3 +
                (0.3 if distribution_shape != 'neutral' else 0.1) * 0.3
            )

            return {
                'signals': signals,
                'regime_strength': float(regime_strength),
                'primary_signal': self._determine_primary_signal(signals),
                'signal_confidence': float(min(1.0, regime_strength))
            }

        except Exception as e:
            logger.error(f"Error detecting regime formation signals: {e}")
            return {}

    def _analyze_oi_divergence(self, relevant_strikes: Dict[str, Any],
                             underlying_price: float, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze OI divergence for regime transition signals"""
        try:
            if len(self.regime_signals) < 5:
                return {'divergence_detected': False, 'reason': 'insufficient_history'}

            # Get recent OI data
            recent_signals = list(self.regime_signals)[-5:]

            # Calculate OI momentum
            call_oi_momentum = []
            put_oi_momentum = []

            for i in range(1, len(recent_signals)):
                prev_call_oi = sum(recent_signals[i-1].get('oi_distribution', {}).get('call_oi', {}).values())
                curr_call_oi = sum(recent_signals[i].get('oi_distribution', {}).get('call_oi', {}).values())
                prev_put_oi = sum(recent_signals[i-1].get('oi_distribution', {}).get('put_oi', {}).values())
                curr_put_oi = sum(recent_signals[i].get('oi_distribution', {}).get('put_oi', {}).values())

                call_oi_momentum.append((curr_call_oi - prev_call_oi) / (prev_call_oi + 1e-8))
                put_oi_momentum.append((curr_put_oi - prev_put_oi) / (prev_put_oi + 1e-8))

            # Detect divergence
            avg_call_momentum = np.mean(call_oi_momentum) if call_oi_momentum else 0
            avg_put_momentum = np.mean(put_oi_momentum) if put_oi_momentum else 0

            # Price momentum (simplified)
            price_momentum = (underlying_price - recent_signals[0]['underlying_price']) / recent_signals[0]['underlying_price']

            # Divergence detection
            call_divergence = (price_momentum > 0 and avg_call_momentum < -0.05) or (price_momentum < 0 and avg_call_momentum > 0.05)
            put_divergence = (price_momentum < 0 and avg_put_momentum < -0.05) or (price_momentum > 0 and avg_put_momentum > 0.05)

            divergence_detected = call_divergence or put_divergence
            divergence_strength = abs(price_momentum - avg_call_momentum) + abs(price_momentum - avg_put_momentum)

            # Store divergence signal
            if divergence_detected:
                self.divergence_signals.append({
                    'timestamp': datetime.now(),
                    'divergence_type': 'call' if call_divergence else 'put',
                    'divergence_strength': divergence_strength,
                    'price_momentum': price_momentum,
                    'oi_momentum': {'call': avg_call_momentum, 'put': avg_put_momentum}
                })

            return {
                'divergence_detected': divergence_detected,
                'divergence_type': 'call' if call_divergence else 'put' if put_divergence else 'none',
                'divergence_strength': float(divergence_strength),
                'price_momentum': float(price_momentum),
                'call_oi_momentum': float(avg_call_momentum),
                'put_oi_momentum': float(avg_put_momentum),
                'regime_transition_signal': divergence_detected and divergence_strength > self.config.divergence_detection_sensitivity
            }

        except Exception as e:
            logger.error(f"Error analyzing OI divergence: {e}")
            return {'divergence_detected': False, 'error': str(e)}

    def _calculate_regime_confirmation(self, regime_formation_signals: Dict[str, Any],
                                     divergence_analysis: Dict[str, Any],
                                     oi_distribution: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall regime confirmation"""
        try:
            # Extract key metrics
            signal_confidence = regime_formation_signals.get('signal_confidence', 0)
            regime_strength = regime_formation_signals.get('regime_strength', 0)
            divergence_detected = divergence_analysis.get('divergence_detected', False)
            divergence_strength = divergence_analysis.get('divergence_strength', 0)

            # Calculate confirmation score
            base_confirmation = signal_confidence * 0.6 + regime_strength * 0.4

            # Adjust for divergence
            if divergence_detected:
                if divergence_strength > 0.3:
                    confirmation_adjustment = -0.3  # Strong divergence reduces confirmation
                else:
                    confirmation_adjustment = -0.1  # Weak divergence slightly reduces confirmation
            else:
                confirmation_adjustment = 0.1  # No divergence slightly increases confirmation

            final_confirmation = max(0.0, min(1.0, base_confirmation + confirmation_adjustment))

            # Determine confirmation level
            if final_confirmation > self.config.regime_confirmation_threshold:
                confirmation_level = 'confirmed'
            elif final_confirmation > 0.5:
                confirmation_level = 'likely'
            elif final_confirmation > 0.3:
                confirmation_level = 'possible'
            else:
                confirmation_level = 'unlikely'

            # Primary regime signal
            primary_signal = regime_formation_signals.get('primary_signal', 'neutral')

            return {
                'confirmation_score': float(final_confirmation),
                'confirmation_level': confirmation_level,
                'primary_regime_signal': primary_signal,
                'regime_confirmed': final_confirmation > self.config.regime_confirmation_threshold,
                'confidence_factors': {
                    'signal_confidence': float(signal_confidence),
                    'regime_strength': float(regime_strength),
                    'divergence_impact': float(confirmation_adjustment)
                }
            }

        except Exception as e:
            logger.error(f"Error calculating regime confirmation: {e}")
            return {'confirmation_score': 0.0, 'confirmation_level': 'error'}

    def _calculate_strike_skew(self, oi_data: Dict, underlying_price: float, option_type: str) -> Dict[str, float]:
        """Calculate strike-based skew metrics"""
        try:
            if option_type == 'call':
                itm_oi = sum(oi for strike, oi in oi_data.items() if float(strike) < underlying_price)
                otm_oi = sum(oi for strike, oi in oi_data.items() if float(strike) > underlying_price)
            else:  # put
                itm_oi = sum(oi for strike, oi in oi_data.items() if float(strike) > underlying_price)
                otm_oi = sum(oi for strike, oi in oi_data.items() if float(strike) < underlying_price)

            total_oi = itm_oi + otm_oi
            skew_ratio = (otm_oi - itm_oi) / (total_oi + 1e-8)

            return {
                'skew_ratio': float(skew_ratio),
                'itm_oi': itm_oi,
                'otm_oi': otm_oi,
                'total_oi': total_oi
            }

        except Exception as e:
            logger.error(f"Error calculating strike skew: {e}")
            return {'skew_ratio': 0.0}

    def _classify_distribution_shape(self, atm_concentration: float, wing_concentration: float,
                                   call_skew: Dict, put_skew: Dict) -> str:
        """Classify OI distribution shape"""
        try:
            if atm_concentration > 0.6:
                return 'concentrated'
            elif wing_concentration > 0.4:
                return 'dispersed'
            elif abs(call_skew.get('skew_ratio', 0)) > 0.3 or abs(put_skew.get('skew_ratio', 0)) > 0.3:
                return 'skewed'
            else:
                return 'neutral'

        except Exception as e:
            logger.error(f"Error classifying distribution shape: {e}")
            return 'neutral'

    def _determine_primary_signal(self, signals: Dict[str, Any]) -> str:
        """Determine the primary regime signal"""
        try:
            if signals.get('consolidation', False):
                return 'consolidation'
            elif signals.get('breakout_potential', False):
                return 'breakout'
            elif signals.get('directional_bias') != 'neutral':
                return f"directional_{signals['directional_bias']}"
            elif signals.get('volatility_regime') != 'neutral':
                return signals['volatility_regime']
            else:
                return 'neutral'

        except Exception as e:
            logger.error(f"Error determining primary signal: {e}")
            return 'neutral'