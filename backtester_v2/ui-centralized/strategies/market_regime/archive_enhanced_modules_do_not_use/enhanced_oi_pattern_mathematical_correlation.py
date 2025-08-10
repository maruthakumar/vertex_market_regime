#!/usr/bin/env python3
"""
Enhanced OI Pattern Recognition with Mathematical Correlation
Enhanced Triple Straddle Rolling Analysis Framework v2.0

Author: The Augster
Date: 2025-06-20
Version: 2.0.0 (Mathematical Correlation Enhancement)

This module enhances the existing advanced Trending OI with PA Analysis system
by adding mathematical correlation-based pattern recognition while preserving
all existing advanced features.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from datetime import datetime
from scipy.stats import pearsonr
from dataclasses import dataclass

# Import existing advanced system
from .enhanced_modules.enhanced_trending_oi_pa_analysis import (
    EnhancedTrendingOIWithPAAnalysis, 
    OIAnalysisResult, 
    OIPattern,
    DivergenceType
)

logger = logging.getLogger(__name__)

@dataclass
class PatternCorrelationResult:
    """Result structure for pattern correlation analysis"""
    correlation_score: float
    similarity_threshold_met: bool
    time_decay_weight: float
    pattern_confidence: float
    mathematical_accuracy: float

@dataclass
class HistoricalPatternMatch:
    """Structure for historical pattern matching"""
    pattern_id: str
    correlation_score: float
    time_distance: int
    weighted_similarity: float
    pattern_outcome: str
    success_probability: float

class MathematicalPatternCorrelationEngine:
    """
    Mathematical correlation engine for OI pattern recognition
    
    Enhances existing advanced pattern recognition with:
    1. Pearson correlation-based similarity (>0.80 threshold)
    2. Time-decay weighting: exp(-λ × (T-t))
    3. Mathematical validation with ±0.001 tolerance
    4. Historical pattern matching with correlation scoring
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize mathematical correlation engine"""
        
        self.config = config or self._get_default_config()
        self.pattern_history = []
        self.correlation_cache = {}
        
        logger.info("🔬 Mathematical Pattern Correlation Engine initialized")
        logger.info(f"✅ Correlation threshold: {self.config['correlation_threshold']}")
        logger.info(f"✅ Time decay factor: {self.config['time_decay_factor']}")
        logger.info(f"✅ Mathematical tolerance: ±{self.config['mathematical_tolerance']}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for correlation engine"""
        
        return {
            # Correlation parameters
            'correlation_threshold': 0.80,      # >0.80 similarity requirement
            'time_decay_factor': 0.1,           # λ = 0.1 for exp(-λ × (T-t))
            'mathematical_tolerance': 0.001,    # ±0.001 tolerance
            
            # Pattern matching parameters
            'pattern_lookback_period': 20,      # 20 periods for pattern matching
            'min_pattern_length': 5,            # Minimum pattern length
            'max_pattern_age': 100,             # Maximum pattern age in periods
            
            # Feature extraction parameters
            'feature_components': [
                'oi_change_rate',
                'volume_change_rate', 
                'price_change_rate',
                'put_call_oi_ratio',
                'strike_distribution_skew'
            ],
            
            # Validation parameters
            'enable_mathematical_validation': True,
            'enable_correlation_caching': True,
            'cache_size_limit': 1000
        }
    
    def calculate_pattern_correlation(self, current_pattern: List[float], 
                                    historical_patterns: List[List[float]]) -> List[PatternCorrelationResult]:
        """
        Calculate Pearson correlation-based pattern similarity with time-decay weighting
        
        Args:
            current_pattern: Current pattern feature vector
            historical_patterns: List of historical pattern feature vectors
            
        Returns:
            List of correlation results with mathematical validation
        """
        
        logger.info("🔍 Calculating pattern correlations with mathematical validation...")
        
        correlation_results = []
        
        for i, historical_pattern in enumerate(historical_patterns):
            try:
                # Calculate Pearson correlation
                correlation_score, p_value = pearsonr(current_pattern, historical_pattern)
                
                # Handle NaN correlations
                if np.isnan(correlation_score):
                    correlation_score = 0.0
                
                # Calculate time decay weight
                time_distance = len(historical_patterns) - i
                time_decay_weight = np.exp(-self.config['time_decay_factor'] * time_distance)
                
                # Calculate weighted similarity
                weighted_similarity = correlation_score * time_decay_weight
                
                # Check similarity threshold
                similarity_threshold_met = correlation_score >= self.config['correlation_threshold']
                
                # Calculate pattern confidence
                pattern_confidence = min(1.0, abs(correlation_score) * time_decay_weight)
                
                # Mathematical accuracy validation
                mathematical_accuracy = self._validate_correlation_calculation(
                    current_pattern, historical_pattern, correlation_score
                )
                
                correlation_result = PatternCorrelationResult(
                    correlation_score=correlation_score,
                    similarity_threshold_met=similarity_threshold_met,
                    time_decay_weight=time_decay_weight,
                    pattern_confidence=pattern_confidence,
                    mathematical_accuracy=mathematical_accuracy
                )
                
                correlation_results.append(correlation_result)
                
                logger.debug(f"   Pattern {i}: Correlation={correlation_score:.3f}, "
                           f"Threshold_Met={similarity_threshold_met}, "
                           f"Time_Weight={time_decay_weight:.3f}")
                
            except Exception as e:
                logger.error(f"Error calculating correlation for pattern {i}: {e}")
                # Add default result for failed calculation
                correlation_results.append(PatternCorrelationResult(
                    correlation_score=0.0,
                    similarity_threshold_met=False,
                    time_decay_weight=0.0,
                    pattern_confidence=0.0,
                    mathematical_accuracy=0.0
                ))
        
        logger.info(f"   ✅ Calculated correlations for {len(correlation_results)} patterns")
        
        return correlation_results
    
    def extract_pattern_features(self, market_data: Dict[str, Any]) -> List[float]:
        """
        Extract pattern features for correlation analysis
        
        Args:
            market_data: Market data containing OI, volume, price information
            
        Returns:
            Feature vector for pattern matching
        """
        
        try:
            features = []
            
            # OI change rate
            oi_change_rate = self._calculate_oi_change_rate(market_data)
            features.append(oi_change_rate)
            
            # Volume change rate
            volume_change_rate = self._calculate_volume_change_rate(market_data)
            features.append(volume_change_rate)
            
            # Price change rate
            price_change_rate = self._calculate_price_change_rate(market_data)
            features.append(price_change_rate)
            
            # Put/Call OI ratio
            put_call_oi_ratio = self._calculate_put_call_oi_ratio(market_data)
            features.append(put_call_oi_ratio)
            
            # Strike distribution skew
            strike_distribution_skew = self._calculate_strike_distribution_skew(market_data)
            features.append(strike_distribution_skew)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting pattern features: {e}")
            return [0.0] * len(self.config['feature_components'])
    
    def find_historical_pattern_matches(self, current_features: List[float]) -> List[HistoricalPatternMatch]:
        """
        Find historical pattern matches using correlation-based similarity
        
        Args:
            current_features: Current pattern feature vector
            
        Returns:
            List of historical pattern matches with correlation scores
        """
        
        logger.info("🔍 Finding historical pattern matches...")
        
        pattern_matches = []
        
        for i, historical_entry in enumerate(self.pattern_history):
            try:
                historical_features = historical_entry['features']
                
                # Calculate correlation
                correlation_score, _ = pearsonr(current_features, historical_features)
                
                # Handle NaN correlations
                if np.isnan(correlation_score):
                    continue
                
                # Check correlation threshold
                if correlation_score >= self.config['correlation_threshold']:
                    # Calculate time distance and decay weight
                    time_distance = len(self.pattern_history) - i
                    time_decay_weight = np.exp(-self.config['time_decay_factor'] * time_distance)
                    
                    # Calculate weighted similarity
                    weighted_similarity = correlation_score * time_decay_weight
                    
                    # Get pattern outcome and success probability
                    pattern_outcome = historical_entry.get('outcome', 'unknown')
                    success_probability = historical_entry.get('success_probability', 0.5)
                    
                    pattern_match = HistoricalPatternMatch(
                        pattern_id=f"pattern_{i}",
                        correlation_score=correlation_score,
                        time_distance=time_distance,
                        weighted_similarity=weighted_similarity,
                        pattern_outcome=pattern_outcome,
                        success_probability=success_probability
                    )
                    
                    pattern_matches.append(pattern_match)
            
            except Exception as e:
                logger.error(f"Error matching historical pattern {i}: {e}")
                continue
        
        # Sort by weighted similarity (highest first)
        pattern_matches.sort(key=lambda x: x.weighted_similarity, reverse=True)
        
        logger.info(f"   ✅ Found {len(pattern_matches)} pattern matches above threshold")
        
        return pattern_matches
    
    def _validate_correlation_calculation(self, pattern1: List[float], 
                                        pattern2: List[float], 
                                        calculated_correlation: float) -> float:
        """
        Validate correlation calculation with ±0.001 tolerance
        
        Args:
            pattern1: First pattern
            pattern2: Second pattern
            calculated_correlation: Calculated correlation score
            
        Returns:
            Mathematical accuracy score (1.0 = perfect, 0.0 = failed)
        """
        
        if not self.config['enable_mathematical_validation']:
            return 1.0
        
        try:
            # Manual correlation calculation for validation
            mean1 = np.mean(pattern1)
            mean2 = np.mean(pattern2)
            
            numerator = np.sum((np.array(pattern1) - mean1) * (np.array(pattern2) - mean2))
            denominator = np.sqrt(np.sum((np.array(pattern1) - mean1)**2) * 
                                np.sum((np.array(pattern2) - mean2)**2))
            
            if denominator == 0:
                manual_correlation = 0.0
            else:
                manual_correlation = numerator / denominator
            
            # Calculate difference
            difference = abs(calculated_correlation - manual_correlation)
            
            # Check tolerance
            if difference <= self.config['mathematical_tolerance']:
                return 1.0  # Perfect accuracy
            else:
                # Proportional accuracy based on tolerance violation
                return max(0.0, 1.0 - (difference / self.config['mathematical_tolerance']))
        
        except Exception as e:
            logger.error(f"Error validating correlation calculation: {e}")
            return 0.0
    
    def _calculate_oi_change_rate(self, market_data: Dict[str, Any]) -> float:
        """Calculate OI change rate"""
        try:
            current_oi = market_data.get('total_oi', 0)
            previous_oi = market_data.get('previous_total_oi', current_oi)
            
            if previous_oi == 0:
                return 0.0
            
            return (current_oi - previous_oi) / previous_oi
        except:
            return 0.0
    
    def _calculate_volume_change_rate(self, market_data: Dict[str, Any]) -> float:
        """Calculate volume change rate"""
        try:
            current_volume = market_data.get('total_volume', 0)
            previous_volume = market_data.get('previous_total_volume', current_volume)
            
            if previous_volume == 0:
                return 0.0
            
            return (current_volume - previous_volume) / previous_volume
        except:
            return 0.0
    
    def _calculate_price_change_rate(self, market_data: Dict[str, Any]) -> float:
        """Calculate price change rate"""
        try:
            current_price = market_data.get('underlying_price', 0)
            previous_price = market_data.get('previous_underlying_price', current_price)
            
            if previous_price == 0:
                return 0.0
            
            return (current_price - previous_price) / previous_price
        except:
            return 0.0
    
    def _calculate_put_call_oi_ratio(self, market_data: Dict[str, Any]) -> float:
        """Calculate Put/Call OI ratio"""
        try:
            call_oi = market_data.get('total_call_oi', 0)
            put_oi = market_data.get('total_put_oi', 0)
            
            if call_oi == 0:
                return 1.0 if put_oi > 0 else 0.0
            
            return put_oi / call_oi
        except:
            return 1.0
    
    def _calculate_strike_distribution_skew(self, market_data: Dict[str, Any]) -> float:
        """Calculate strike distribution skew"""
        try:
            options_data = market_data.get('options_data', {})
            
            if not options_data:
                return 0.0
            
            # Calculate OI-weighted strike distribution
            strikes = []
            weights = []
            
            for strike, option_data in options_data.items():
                total_oi = 0
                if 'CE' in option_data:
                    total_oi += option_data['CE'].get('oi', 0)
                if 'PE' in option_data:
                    total_oi += option_data['PE'].get('oi', 0)
                
                if total_oi > 0:
                    strikes.append(strike)
                    weights.append(total_oi)
            
            if len(strikes) < 3:
                return 0.0
            
            # Calculate weighted skewness
            strikes = np.array(strikes)
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Normalize weights
            
            weighted_mean = np.sum(strikes * weights)
            weighted_var = np.sum(weights * (strikes - weighted_mean)**2)
            weighted_std = np.sqrt(weighted_var)
            
            if weighted_std == 0:
                return 0.0
            
            weighted_skew = np.sum(weights * ((strikes - weighted_mean) / weighted_std)**3)
            
            return weighted_skew
            
        except:
            return 0.0
    
    def update_pattern_history(self, features: List[float], outcome: str, success_probability: float):
        """Update pattern history for future correlation analysis"""
        
        pattern_entry = {
            'timestamp': datetime.now(),
            'features': features,
            'outcome': outcome,
            'success_probability': success_probability
        }
        
        self.pattern_history.append(pattern_entry)
        
        # Limit history size
        if len(self.pattern_history) > self.config['max_pattern_age']:
            self.pattern_history = self.pattern_history[-self.config['max_pattern_age']:]
        
        logger.debug(f"Updated pattern history: {len(self.pattern_history)} patterns stored")

class EnhancedOIPatternWithMathematicalCorrelation(EnhancedTrendingOIWithPAAnalysis):
    """
    Enhanced OI Pattern Recognition with Mathematical Correlation
    
    Extends the existing advanced Trending OI with PA Analysis system
    by adding mathematical correlation-based pattern recognition while
    preserving all existing advanced features.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize enhanced OI pattern analysis with mathematical correlation"""
        
        # Initialize parent class (preserves all existing advanced features)
        super().__init__(config)
        
        # Initialize mathematical correlation engine
        correlation_config = self.config.get('correlation_engine_config', {})
        self.correlation_engine = MathematicalPatternCorrelationEngine(correlation_config)
        
        # Enhanced configuration
        self.enable_mathematical_correlation = self.config.get('enable_mathematical_correlation', True)
        self.correlation_weight = self.config.get('correlation_weight', 0.3)  # 30% weight for correlation
        
        logger.info("🔬 Enhanced OI Pattern Analysis with Mathematical Correlation initialized")
        logger.info(f"✅ Mathematical correlation: {'ENABLED' if self.enable_mathematical_correlation else 'DISABLED'}")
        logger.info(f"✅ Correlation weight: {self.correlation_weight}")
    
    def analyze_trending_oi_pa(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced analysis with mathematical correlation while preserving existing features
        
        Args:
            market_data: Comprehensive market data
            
        Returns:
            Enhanced analysis results with correlation metrics
        """
        
        # Step 1: Run existing advanced analysis (preserves all features)
        base_result = super().analyze_trending_oi_pa(market_data)
        
        # Step 2: Add mathematical correlation analysis if enabled
        if self.enable_mathematical_correlation:
            correlation_analysis = self._perform_mathematical_correlation_analysis(market_data)
            
            # Step 3: Integrate correlation results with existing analysis
            enhanced_result = self._integrate_correlation_with_existing_analysis(
                base_result, correlation_analysis
            )
            
            return enhanced_result
        
        return base_result
    
    def _perform_mathematical_correlation_analysis(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform mathematical correlation analysis"""
        
        try:
            # Extract pattern features
            current_features = self.correlation_engine.extract_pattern_features(market_data)
            
            # Find historical pattern matches
            pattern_matches = self.correlation_engine.find_historical_pattern_matches(current_features)
            
            # Calculate overall correlation confidence
            if pattern_matches:
                avg_correlation = np.mean([match.correlation_score for match in pattern_matches])
                avg_weighted_similarity = np.mean([match.weighted_similarity for match in pattern_matches])
                correlation_confidence = min(1.0, avg_weighted_similarity)
            else:
                avg_correlation = 0.0
                avg_weighted_similarity = 0.0
                correlation_confidence = 0.5  # Neutral confidence when no matches
            
            return {
                'current_features': current_features,
                'pattern_matches': pattern_matches,
                'avg_correlation': avg_correlation,
                'avg_weighted_similarity': avg_weighted_similarity,
                'correlation_confidence': correlation_confidence,
                'matches_above_threshold': len(pattern_matches)
            }
            
        except Exception as e:
            logger.error(f"Error in mathematical correlation analysis: {e}")
            return {
                'current_features': [],
                'pattern_matches': [],
                'avg_correlation': 0.0,
                'avg_weighted_similarity': 0.0,
                'correlation_confidence': 0.5,
                'matches_above_threshold': 0
            }
    
    def _integrate_correlation_with_existing_analysis(self, base_result: Dict[str, Any], 
                                                    correlation_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate correlation analysis with existing advanced analysis"""
        
        try:
            # Preserve all existing results
            enhanced_result = base_result.copy()
            
            # Add correlation analysis results
            enhanced_result['mathematical_correlation'] = correlation_analysis
            
            # Enhance confidence with correlation confidence
            original_confidence = base_result.get('confidence', 0.5)
            correlation_confidence = correlation_analysis.get('correlation_confidence', 0.5)
            
            # Weighted confidence integration
            enhanced_confidence = (
                (1 - self.correlation_weight) * original_confidence +
                self.correlation_weight * correlation_confidence
            )
            
            enhanced_result['confidence'] = enhanced_confidence
            enhanced_result['correlation_enhanced'] = True
            enhanced_result['correlation_weight_applied'] = self.correlation_weight
            
            # Add mathematical validation metrics
            enhanced_result['mathematical_validation'] = {
                'correlation_threshold_met': correlation_analysis.get('matches_above_threshold', 0) > 0,
                'avg_correlation_score': correlation_analysis.get('avg_correlation', 0.0),
                'pattern_matches_found': correlation_analysis.get('matches_above_threshold', 0)
            }
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Error integrating correlation with existing analysis: {e}")
            return base_result

def main():
    """Main function for testing enhanced OI pattern correlation"""
    
    logger.info("🚀 Testing Enhanced OI Pattern Recognition with Mathematical Correlation")
    
    # Initialize enhanced system
    enhanced_oi_analyzer = EnhancedOIPatternWithMathematicalCorrelation()
    
    # Generate sample market data
    sample_market_data = {
        'underlying_price': 22000,
        'total_oi': 1000000,
        'previous_total_oi': 950000,
        'total_volume': 500000,
        'previous_total_volume': 480000,
        'total_call_oi': 600000,
        'total_put_oi': 400000,
        'options_data': {
            21900: {'CE': {'oi': 10000}, 'PE': {'oi': 15000}},
            22000: {'CE': {'oi': 25000}, 'PE': {'oi': 30000}},
            22100: {'CE': {'oi': 20000}, 'PE': {'oi': 12000}}
        },
        'timestamp': datetime.now()
    }
    
    # Perform enhanced analysis
    analysis_result = enhanced_oi_analyzer.analyze_trending_oi_pa(sample_market_data)
    
    logger.info("🎯 Enhanced OI Pattern Recognition Testing Complete")
    
    return analysis_result

if __name__ == "__main__":
    main()
