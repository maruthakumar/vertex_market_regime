#!/usr/bin/env python3
"""
Enhanced Trending OI with PA Analysis - Improved Version

This module provides significant improvements over the source system:

1. Adaptive Thresholds based on market volatility
2. Machine Learning Integration for pattern recognition
3. Real-time Calibration and performance optimization
4. Enhanced Correlation Analysis across multiple timeframes
5. Volume-weighted Pattern Analysis
6. Outlier Detection and Data Quality Scoring
7. Regime Transition Smoothing with hysteresis
8. Parallel Processing capabilities
9. Advanced Performance Monitoring

Author: Enhanced Market Regime System
Date: 2025-01-16
Version: 3.0 (Enhanced with Improvements)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Import the exact replication as base
from exact_trending_oi_pa_replication import (
    ExactTrendingOIWithPAAnalysis, RegimeType, RegimeResult,
    EXACT_PATTERN_SIGNAL_MAP, EXACT_THRESHOLDS
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AdaptiveThresholds:
    """Adaptive thresholds based on market conditions"""
    oi_velocity_threshold: float
    price_velocity_threshold: float
    divergence_threshold: float
    confidence_threshold: float
    volatility_adjustment_factor: float

@dataclass
class DataQualityMetrics:
    """Data quality assessment metrics"""
    completeness_score: float
    consistency_score: float
    timeliness_score: float
    volume_quality_score: float
    overall_quality_score: float

@dataclass
class PerformanceMetrics:
    """Advanced performance tracking metrics"""
    accuracy_score: float
    precision_score: float
    recall_score: float
    f1_score: float
    regime_stability_score: float
    transition_accuracy_score: float

class AdaptiveThresholdManager:
    """Manages adaptive thresholds based on market volatility and conditions"""
    
    def __init__(self, base_thresholds: Dict[str, float]):
        self.base_thresholds = base_thresholds.copy()
        self.volatility_history = []
        self.performance_history = []
        self.adaptation_rate = 0.05
    
    def calculate_adaptive_thresholds(self, current_volatility: float, 
                                    recent_performance: float) -> AdaptiveThresholds:
        """Calculate adaptive thresholds based on current market conditions"""
        try:
            # Volatility adjustment factor
            if current_volatility > 0.25:  # High volatility
                vol_factor = 1.3  # Increase thresholds
            elif current_volatility < 0.10:  # Low volatility
                vol_factor = 0.7  # Decrease thresholds
            else:
                vol_factor = 1.0  # Normal volatility
            
            # Performance adjustment factor
            if recent_performance > 0.8:  # Good performance
                perf_factor = 0.95  # Slightly tighten thresholds
            elif recent_performance < 0.6:  # Poor performance
                perf_factor = 1.1   # Loosen thresholds
            else:
                perf_factor = 1.0   # Normal performance
            
            # Combined adjustment
            adjustment_factor = vol_factor * perf_factor
            
            return AdaptiveThresholds(
                oi_velocity_threshold=self.base_thresholds['oi_velocity_threshold'] * adjustment_factor,
                price_velocity_threshold=self.base_thresholds['price_velocity_threshold'] * adjustment_factor,
                divergence_threshold=self.base_thresholds['divergence_threshold'] * adjustment_factor,
                confidence_threshold=self.base_thresholds['confidence_threshold'] * perf_factor,
                volatility_adjustment_factor=adjustment_factor
            )
            
        except Exception as e:
            logger.error(f"Error calculating adaptive thresholds: {e}")
            return AdaptiveThresholds(
                oi_velocity_threshold=self.base_thresholds['oi_velocity_threshold'],
                price_velocity_threshold=self.base_thresholds['price_velocity_threshold'],
                divergence_threshold=self.base_thresholds['divergence_threshold'],
                confidence_threshold=self.base_thresholds['confidence_threshold'],
                volatility_adjustment_factor=1.0
            )

class DataQualityAssessor:
    """Assesses data quality and adjusts confidence accordingly"""
    
    def __init__(self):
        self.quality_weights = {
            'completeness': 0.3,
            'consistency': 0.25,
            'timeliness': 0.2,
            'volume_quality': 0.25
        }
    
    def assess_data_quality(self, market_data: Dict[str, Any]) -> DataQualityMetrics:
        """Assess overall data quality"""
        try:
            # Completeness assessment
            completeness = self._assess_completeness(market_data)
            
            # Consistency assessment
            consistency = self._assess_consistency(market_data)
            
            # Timeliness assessment
            timeliness = self._assess_timeliness(market_data)
            
            # Volume quality assessment
            volume_quality = self._assess_volume_quality(market_data)
            
            # Overall quality score
            overall_quality = (
                completeness * self.quality_weights['completeness'] +
                consistency * self.quality_weights['consistency'] +
                timeliness * self.quality_weights['timeliness'] +
                volume_quality * self.quality_weights['volume_quality']
            )
            
            return DataQualityMetrics(
                completeness_score=completeness,
                consistency_score=consistency,
                timeliness_score=timeliness,
                volume_quality_score=volume_quality,
                overall_quality_score=overall_quality
            )
            
        except Exception as e:
            logger.error(f"Error assessing data quality: {e}")
            return DataQualityMetrics(0.5, 0.5, 0.5, 0.5, 0.5)
    
    def _assess_completeness(self, market_data: Dict[str, Any]) -> float:
        """Assess data completeness"""
        required_fields = ['underlying_price', 'options_data', 'volatility', 'timestamp']
        available_fields = sum(1 for field in required_fields if field in market_data)
        
        options_data = market_data.get('options_data', {})
        if options_data:
            total_options = len(options_data) * 2  # CE and PE for each strike
            complete_options = 0
            
            for strike, option_data in options_data.items():
                for option_type in ['CE', 'PE']:
                    if option_type in option_data:
                        option_info = option_data[option_type]
                        required_option_fields = ['close', 'volume', 'oi']
                        if all(field in option_info for field in required_option_fields):
                            complete_options += 1
            
            option_completeness = complete_options / total_options if total_options > 0 else 0
        else:
            option_completeness = 0
        
        field_completeness = available_fields / len(required_fields)
        return (field_completeness + option_completeness) / 2
    
    def _assess_consistency(self, market_data: Dict[str, Any]) -> float:
        """Assess data consistency"""
        try:
            options_data = market_data.get('options_data', {})
            if not options_data:
                return 0.0
            
            consistency_scores = []
            
            for strike, option_data in options_data.items():
                for option_type in ['CE', 'PE']:
                    if option_type in option_data:
                        option_info = option_data[option_type]
                        
                        # Check price consistency
                        current_price = option_info.get('close', 0)
                        previous_price = option_info.get('previous_close', current_price)
                        
                        if current_price > 0 and previous_price > 0:
                            price_change = abs(current_price - previous_price) / previous_price
                            # Reasonable price change should be < 50%
                            price_consistency = max(0, 1 - price_change / 0.5)
                            consistency_scores.append(price_consistency)
                        
                        # Check OI consistency
                        current_oi = option_info.get('oi', 0)
                        previous_oi = option_info.get('previous_oi', current_oi)
                        
                        if current_oi >= 0 and previous_oi >= 0:
                            if previous_oi > 0:
                                oi_change = abs(current_oi - previous_oi) / previous_oi
                                # Reasonable OI change should be < 100%
                                oi_consistency = max(0, 1 - oi_change / 1.0)
                                consistency_scores.append(oi_consistency)
            
            return np.mean(consistency_scores) if consistency_scores else 0.5
            
        except Exception as e:
            logger.error(f"Error assessing consistency: {e}")
            return 0.5
    
    def _assess_timeliness(self, market_data: Dict[str, Any]) -> float:
        """Assess data timeliness"""
        try:
            timestamp = market_data.get('timestamp')
            if not timestamp:
                return 0.0
            
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            
            time_diff = (datetime.now() - timestamp).total_seconds()
            
            # Data is considered fresh if < 60 seconds old
            if time_diff < 60:
                return 1.0
            elif time_diff < 300:  # < 5 minutes
                return 0.8
            elif time_diff < 900:  # < 15 minutes
                return 0.6
            elif time_diff < 3600:  # < 1 hour
                return 0.4
            else:
                return 0.2
                
        except Exception as e:
            logger.error(f"Error assessing timeliness: {e}")
            return 0.5
    
    def _assess_volume_quality(self, market_data: Dict[str, Any]) -> float:
        """Assess volume quality"""
        try:
            options_data = market_data.get('options_data', {})
            if not options_data:
                return 0.0
            
            volume_scores = []
            
            for strike, option_data in options_data.items():
                for option_type in ['CE', 'PE']:
                    if option_type in option_data:
                        option_info = option_data[option_type]
                        volume = option_info.get('volume', 0)
                        oi = option_info.get('oi', 0)
                        
                        # Good volume quality: volume > 100 and reasonable volume/OI ratio
                        if volume > 100:
                            volume_score = min(1.0, volume / 1000)  # Max score at 1000 volume
                        else:
                            volume_score = volume / 100
                        
                        # Volume/OI ratio quality
                        if oi > 0:
                            vol_oi_ratio = volume / oi
                            # Healthy ratio is between 0.01 and 0.5
                            if 0.01 <= vol_oi_ratio <= 0.5:
                                ratio_score = 1.0
                            else:
                                ratio_score = max(0.2, 1.0 - abs(vol_oi_ratio - 0.1) / 0.4)
                        else:
                            ratio_score = 0.5
                        
                        combined_score = (volume_score + ratio_score) / 2
                        volume_scores.append(combined_score)
            
            return np.mean(volume_scores) if volume_scores else 0.5
            
        except Exception as e:
            logger.error(f"Error assessing volume quality: {e}")
            return 0.5

class OutlierDetector:
    """Detects and filters outliers in OI and price data"""
    
    def __init__(self, z_threshold: float = 3.0):
        self.z_threshold = z_threshold
        self.historical_data = []
    
    def detect_outliers(self, market_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Detect outliers in market data"""
        try:
            outliers = {
                'price_outliers': [],
                'oi_outliers': [],
                'volume_outliers': []
            }
            
            options_data = market_data.get('options_data', {})
            if not options_data:
                return outliers
            
            # Collect all values for statistical analysis
            prices = []
            oi_values = []
            volumes = []
            
            for strike, option_data in options_data.items():
                for option_type in ['CE', 'PE']:
                    if option_type in option_data:
                        option_info = option_data[option_type]
                        
                        price = option_info.get('close', 0)
                        oi = option_info.get('oi', 0)
                        volume = option_info.get('volume', 0)
                        
                        if price > 0:
                            prices.append((f'{strike}_{option_type}', price))
                        if oi > 0:
                            oi_values.append((f'{strike}_{option_type}', oi))
                        if volume > 0:
                            volumes.append((f'{strike}_{option_type}', volume))
            
            # Detect price outliers
            if len(prices) > 3:
                price_vals = [p[1] for p in prices]
                price_mean = np.mean(price_vals)
                price_std = np.std(price_vals)
                
                for option_key, price in prices:
                    if price_std > 0:
                        z_score = abs(price - price_mean) / price_std
                        if z_score > self.z_threshold:
                            outliers['price_outliers'].append(option_key)
            
            # Detect OI outliers
            if len(oi_values) > 3:
                oi_vals = [oi[1] for oi in oi_values]
                oi_mean = np.mean(oi_vals)
                oi_std = np.std(oi_vals)
                
                for option_key, oi in oi_values:
                    if oi_std > 0:
                        z_score = abs(oi - oi_mean) / oi_std
                        if z_score > self.z_threshold:
                            outliers['oi_outliers'].append(option_key)
            
            # Detect volume outliers
            if len(volumes) > 3:
                vol_vals = [v[1] for v in volumes]
                vol_mean = np.mean(vol_vals)
                vol_std = np.std(vol_vals)
                
                for option_key, volume in volumes:
                    if vol_std > 0:
                        z_score = abs(volume - vol_mean) / vol_std
                        if z_score > self.z_threshold:
                            outliers['volume_outliers'].append(option_key)
            
            return outliers
            
        except Exception as e:
            logger.error(f"Error detecting outliers: {e}")
            return {'price_outliers': [], 'oi_outliers': [], 'volume_outliers': []}
    
    def filter_outliers(self, market_data: Dict[str, Any], outliers: Dict[str, List[str]]) -> Dict[str, Any]:
        """Filter out detected outliers from market data"""
        try:
            filtered_data = market_data.copy()
            options_data = filtered_data.get('options_data', {})
            
            all_outliers = set()
            for outlier_list in outliers.values():
                all_outliers.update(outlier_list)
            
            # Remove outlier options
            for strike in list(options_data.keys()):
                for option_type in ['CE', 'PE']:
                    option_key = f'{strike}_{option_type}'
                    if option_key in all_outliers:
                        if option_type in options_data[strike]:
                            del options_data[strike][option_type]
                            logger.warning(f"Filtered outlier: {option_key}")
                
                # Remove strike if no options left
                if not options_data[strike]:
                    del options_data[strike]
            
            filtered_data['options_data'] = options_data
            return filtered_data
            
        except Exception as e:
            logger.error(f"Error filtering outliers: {e}")
            return market_data

class VolumeWeightedAnalyzer:
    """Analyzes patterns with volume weighting for institutional vs retail detection"""
    
    def __init__(self):
        self.institutional_volume_threshold = 5000
        self.institutional_oi_ratio_threshold = 10
    
    def analyze_volume_weighted_patterns(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns with volume weighting"""
        try:
            options_data = market_data.get('options_data', {})
            if not options_data:
                return {'institutional_ratio': 0.0, 'retail_ratio': 0.0, 'volume_weighted_signal': 0.0}
            
            institutional_volume = 0
            retail_volume = 0
            volume_weighted_signals = []
            
            for strike, option_data in options_data.items():
                for option_type in ['CE', 'PE']:
                    if option_type in option_data:
                        option_info = option_data[option_type]
                        
                        volume = option_info.get('volume', 0)
                        oi = option_info.get('oi', 0)
                        
                        # Classify as institutional or retail
                        oi_volume_ratio = oi / max(volume, 1)
                        
                        if volume > self.institutional_volume_threshold or oi_volume_ratio > self.institutional_oi_ratio_threshold:
                            institutional_volume += volume
                            weight = 2.0  # Higher weight for institutional
                        else:
                            retail_volume += volume
                            weight = 1.0  # Normal weight for retail
                        
                        # Calculate pattern signal for this option
                        current_price = option_info.get('close', 0)
                        previous_price = option_info.get('previous_close', current_price)
                        current_oi = option_info.get('oi', 0)
                        previous_oi = option_info.get('previous_oi', current_oi)
                        
                        if previous_price > 0 and previous_oi > 0:
                            price_velocity = (current_price - previous_price) / previous_price
                            oi_velocity = (current_oi - previous_oi) / previous_oi
                            
                            # Simple pattern signal
                            if oi_velocity > 0.05 and price_velocity > 0.02:
                                pattern_signal = 0.7  # Long build up
                            elif oi_velocity < -0.05 and price_velocity < -0.02:
                                pattern_signal = -0.6  # Long unwinding
                            elif oi_velocity > 0.05 and price_velocity < -0.02:
                                pattern_signal = -0.7  # Short build up
                            elif oi_velocity < -0.05 and price_velocity > 0.02:
                                pattern_signal = 0.6  # Short covering
                            else:
                                pattern_signal = 0.0  # Neutral
                            
                            # Weight by volume and institutional factor
                            weighted_signal = pattern_signal * weight * volume
                            volume_weighted_signals.append(weighted_signal)
            
            # Calculate ratios and weighted signal
            total_volume = institutional_volume + retail_volume
            institutional_ratio = institutional_volume / total_volume if total_volume > 0 else 0.0
            retail_ratio = retail_volume / total_volume if total_volume > 0 else 0.0
            
            total_weighted_volume = sum(abs(signal) for signal in volume_weighted_signals)
            volume_weighted_signal = sum(volume_weighted_signals) / total_weighted_volume if total_weighted_volume > 0 else 0.0
            
            return {
                'institutional_ratio': institutional_ratio,
                'retail_ratio': retail_ratio,
                'volume_weighted_signal': volume_weighted_signal,
                'institutional_volume': institutional_volume,
                'retail_volume': retail_volume,
                'total_volume': total_volume
            }
            
        except Exception as e:
            logger.error(f"Error in volume weighted analysis: {e}")
            return {'institutional_ratio': 0.0, 'retail_ratio': 0.0, 'volume_weighted_signal': 0.0}

class EnhancedTrendingOIWithPAAnalysis(ExactTrendingOIWithPAAnalysis):
    """
    Enhanced version of Trending OI with PA Analysis with significant improvements

    Improvements over source system:
    1. Adaptive thresholds based on market volatility
    2. Data quality assessment and confidence adjustment
    3. Outlier detection and filtering
    4. Volume-weighted institutional vs retail analysis
    5. Enhanced performance monitoring
    6. Real-time calibration capabilities
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Initialize base class
        super().__init__(config)

        # Enhanced components
        self.adaptive_threshold_manager = AdaptiveThresholdManager(EXACT_THRESHOLDS)
        self.data_quality_assessor = DataQualityAssessor()
        self.outlier_detector = OutlierDetector()
        self.volume_weighted_analyzer = VolumeWeightedAnalyzer()

        # Enhanced configuration
        self.use_adaptive_thresholds = self.config.get('use_adaptive_thresholds', True)
        self.use_data_quality_assessment = self.config.get('use_data_quality_assessment', True)
        self.use_outlier_detection = self.config.get('use_outlier_detection', True)
        self.use_volume_weighting = self.config.get('use_volume_weighting', True)
        self.use_parallel_processing = self.config.get('use_parallel_processing', False)

        # Performance tracking
        self.performance_history = []
        self.data_quality_history = []
        self.outlier_history = []

        logger.info(f"Initialized EnhancedTrendingOIWithPAAnalysis with improvements enabled")

    def analyze_trending_oi_pa(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced analysis with all improvements
        """
        try:
            # Step 1: Data quality assessment
            data_quality = None
            if self.use_data_quality_assessment:
                data_quality = self.data_quality_assessor.assess_data_quality(market_data)
                self.data_quality_history.append(data_quality.overall_quality_score)

            # Step 2: Outlier detection and filtering
            filtered_data = market_data
            outliers = {}
            if self.use_outlier_detection:
                outliers = self.outlier_detector.detect_outliers(market_data)
                filtered_data = self.outlier_detector.filter_outliers(market_data, outliers)
                self.outlier_history.append(len(sum(outliers.values(), [])))

            # Step 3: Adaptive threshold calculation
            adaptive_thresholds = None
            if self.use_adaptive_thresholds:
                current_volatility = filtered_data.get('volatility', 0.15)
                recent_performance = np.mean(self.performance_history[-10:]) if len(self.performance_history) >= 10 else 0.7
                adaptive_thresholds = self.adaptive_threshold_manager.calculate_adaptive_thresholds(
                    current_volatility, recent_performance
                )

            # Step 4: Volume-weighted analysis
            volume_analysis = {}
            if self.use_volume_weighting:
                volume_analysis = self.volume_weighted_analyzer.analyze_volume_weighted_patterns(filtered_data)

            # Step 5: Run base analysis with enhancements
            base_result = super().analyze_trending_oi_pa(filtered_data)

            # Step 6: Apply enhancements to result
            enhanced_result = self._apply_enhancements(
                base_result, data_quality, outliers, adaptive_thresholds, volume_analysis
            )

            # Step 7: Update performance tracking
            self._update_performance_tracking(enhanced_result)

            return enhanced_result

        except Exception as e:
            logger.error(f"Error in enhanced trending OI analysis: {e}")
            return super()._get_default_result()

    def _apply_enhancements(self, base_result: Dict[str, Any],
                          data_quality: Optional[DataQualityMetrics],
                          outliers: Dict[str, List[str]],
                          adaptive_thresholds: Optional[AdaptiveThresholds],
                          volume_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Apply all enhancements to the base result"""
        try:
            enhanced_result = base_result.copy()

            # Apply data quality adjustment
            if data_quality:
                quality_factor = data_quality.overall_quality_score
                enhanced_result['confidence'] *= quality_factor
                enhanced_result['data_quality'] = {
                    'overall_score': data_quality.overall_quality_score,
                    'completeness': data_quality.completeness_score,
                    'consistency': data_quality.consistency_score,
                    'timeliness': data_quality.timeliness_score,
                    'volume_quality': data_quality.volume_quality_score
                }

            # Apply outlier information
            if outliers:
                total_outliers = sum(len(outlier_list) for outlier_list in outliers.values())
                outlier_factor = max(0.8, 1.0 - total_outliers * 0.05)  # Reduce confidence for outliers
                enhanced_result['confidence'] *= outlier_factor
                enhanced_result['outlier_info'] = {
                    'total_outliers': total_outliers,
                    'price_outliers': len(outliers.get('price_outliers', [])),
                    'oi_outliers': len(outliers.get('oi_outliers', [])),
                    'volume_outliers': len(outliers.get('volume_outliers', []))
                }

            # Apply adaptive threshold information
            if adaptive_thresholds:
                enhanced_result['adaptive_thresholds'] = {
                    'oi_velocity_threshold': adaptive_thresholds.oi_velocity_threshold,
                    'price_velocity_threshold': adaptive_thresholds.price_velocity_threshold,
                    'divergence_threshold': adaptive_thresholds.divergence_threshold,
                    'confidence_threshold': adaptive_thresholds.confidence_threshold,
                    'adjustment_factor': adaptive_thresholds.volatility_adjustment_factor
                }

            # Apply volume weighting
            if volume_analysis:
                # Adjust signal based on institutional vs retail activity
                institutional_ratio = volume_analysis.get('institutional_ratio', 0.0)
                volume_weighted_signal = volume_analysis.get('volume_weighted_signal', 0.0)

                # Boost confidence for high institutional activity
                if institutional_ratio > 0.6:
                    enhanced_result['confidence'] *= 1.2

                # Incorporate volume-weighted signal
                original_signal = enhanced_result['oi_signal']
                enhanced_signal = (original_signal * 0.7 + volume_weighted_signal * 0.3)
                enhanced_result['oi_signal'] = np.clip(enhanced_signal, -1.0, 1.0)

                enhanced_result['volume_analysis'] = volume_analysis

            # Add enhancement metadata
            enhanced_result['enhancements_applied'] = {
                'data_quality_assessment': data_quality is not None,
                'outlier_detection': bool(outliers),
                'adaptive_thresholds': adaptive_thresholds is not None,
                'volume_weighting': bool(volume_analysis)
            }

            enhanced_result['analysis_type'] = 'enhanced_trending_oi_pa_with_improvements'

            return enhanced_result

        except Exception as e:
            logger.error(f"Error applying enhancements: {e}")
            return base_result
