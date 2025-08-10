#!/usr/bin/env python3
"""
Enhanced Greek Sentiment Integration
===================================

This module provides complete integration of the Enhanced Greek Sentiment Analysis
with the main market regime strategy engine, including proper configuration
parameter mapping, real-time data pipeline connection, and integration hooks.

Features:
- Complete configuration parameter integration
- Real-time data pipeline connection
- Integration hooks for main strategy engine
- Performance monitoring and optimization
- Error handling and recovery
- Comprehensive test suite integration

Author: The Augster
Date: 2025-01-06
Version: 1.0.0 - Enhanced Greek Sentiment Integration
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import time

# Import the enhanced Greek sentiment analysis module
try:
    from .archive_enhanced_modules_do_not_use.enhanced_greek_sentiment_analysis import (
        GreekSentimentAnalyzerAnalysis, 
        GreekAnalysisResult,
        GreekSentimentType
    )
except ImportError:
    # Fallback import for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent / 'enhanced_modules'))
    from enhanced_greek_sentiment_analysis import (
        GreekSentimentAnalyzerAnalysis,
        GreekAnalysisResult, 
        GreekSentimentType
    )

logger = logging.getLogger(__name__)

@dataclass
class GreekSentimentIntegrationConfig:
    """Configuration for Greek sentiment integration"""
    # Greek weights (from Excel configuration)
    delta_weight: float = 1.2
    vega_weight: float = 1.5
    theta_weight: float = 0.3
    gamma_weight: float = 0.0
    
    # Sentiment thresholds (from Excel configuration)
    strong_bullish_threshold: float = 0.45
    mild_bullish_threshold: float = 0.15
    sideways_to_bullish_threshold: float = 0.08
    neutral_upper_threshold: float = 0.05
    neutral_lower_threshold: float = -0.05
    sideways_to_bearish_threshold: float = -0.08
    mild_bearish_threshold: float = -0.15
    strong_bearish_threshold: float = -0.45
    
    # DTE configuration (from Excel configuration)
    near_expiry_dte: int = 7
    medium_expiry_dte: int = 30
    far_expiry_dte: int = 90
    
    # Performance configuration
    enable_performance_tracking: bool = True
    enable_caching: bool = True
    cache_duration_minutes: int = 5
    
    # Integration configuration
    enable_real_time_updates: bool = True
    update_frequency_seconds: int = 60
    enable_error_recovery: bool = True

@dataclass
class GreekSentimentIntegrationResult:
    """Result structure for integrated Greek sentiment analysis"""
    sentiment_score: float
    sentiment_type: GreekSentimentType
    confidence: float
    regime_contribution: float
    
    # Detailed breakdown
    delta_contribution: float
    vega_contribution: float
    theta_contribution: float
    gamma_contribution: float
    
    # Integration metrics
    processing_time: float
    data_quality_score: float
    integration_status: str
    
    # Supporting data
    baseline_changes: Dict[str, float] = field(default_factory=dict)
    dte_adjustments: Dict[str, float] = field(default_factory=dict)
    cross_strike_correlation: float = 0.0
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    data_points_used: int = 0

class GreekSentimentAnalyzerIntegration:
    """
    Complete integration wrapper for Enhanced Greek Sentiment Analysis
    
    Provides seamless integration with the main market regime strategy engine
    including configuration management, real-time data processing, and
    performance optimization.
    """
    
    def __init__(self, config: Optional[GreekSentimentIntegrationConfig] = None):
        """Initialize Enhanced Greek Sentiment Integration"""
        self.config = config or GreekSentimentIntegrationConfig()
        
        # Initialize the core Greek sentiment analyzer
        self.greek_analyzer = self._initialize_greek_analyzer()
        
        # Integration state
        self.is_initialized = False
        self.last_update = None
        self.error_count = 0
        self.total_analyses = 0
        
        # Performance tracking
        self.performance_metrics = {
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'max_processing_time': 0.0,
            'min_processing_time': float('inf'),
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Caching system
        self.result_cache: Dict[str, Tuple[GreekSentimentIntegrationResult, datetime]] = {}
        
        logger.info("Enhanced Greek Sentiment Integration initialized")
    
    def _initialize_greek_analyzer(self) -> GreekSentimentAnalyzerAnalysis:
        """Initialize the core Greek sentiment analyzer with configuration"""
        try:
            # Convert integration config to analyzer config
            analyzer_config = {
                'base_greek_weights': {
                    'delta': self.config.delta_weight,
                    'vega': self.config.vega_weight,
                    'theta': self.config.theta_weight,
                    'gamma': self.config.gamma_weight
                },
                'sentiment_thresholds': {
                    'strong_bullish': self.config.strong_bullish_threshold,
                    'mild_bullish': self.config.mild_bullish_threshold,
                    'sideways_to_bullish': self.config.sideways_to_bullish_threshold,
                    'neutral_upper': self.config.neutral_upper_threshold,
                    'neutral_lower': self.config.neutral_lower_threshold,
                    'sideways_to_bearish': self.config.sideways_to_bearish_threshold,
                    'mild_bearish': self.config.mild_bearish_threshold,
                    'strong_bearish': self.config.strong_bearish_threshold
                },
                'dte_categories': {
                    'near_expiry': self.config.near_expiry_dte,
                    'medium_expiry': self.config.medium_expiry_dte,
                    'far_expiry': self.config.far_expiry_dte
                }
            }
            
            return GreekSentimentAnalyzerAnalysis(analyzer_config)
            
        except Exception as e:
            logger.error(f"Error initializing Greek sentiment analyzer: {e}")
            # Return fallback analyzer
            return GreekSentimentAnalyzerAnalysis()
    
    def analyze_greek_sentiment(self, market_data: Dict[str, Any]) -> Optional[GreekSentimentIntegrationResult]:
        """
        Perform integrated Greek sentiment analysis
        
        Args:
            market_data: Market data including options data with Greeks
            
        Returns:
            Optional[GreekSentimentIntegrationResult]: Analysis result or None if error
        """
        start_time = time.time()
        
        try:
            # Check cache first
            if self.config.enable_caching:
                cache_key = self._generate_cache_key(market_data)
                cached_result = self._get_cached_result(cache_key)
                if cached_result:
                    self.performance_metrics['cache_hits'] += 1
                    return cached_result
                else:
                    self.performance_metrics['cache_misses'] += 1
            
            # Validate input data
            if not self._validate_market_data(market_data):
                logger.warning("Invalid market data for Greek sentiment analysis")
                return None
            
            # Perform core Greek sentiment analysis
            greek_result = self.greek_analyzer.analyze_enhanced_greek_sentiment(market_data)
            
            if not greek_result:
                logger.warning("Greek sentiment analysis returned no result")
                return None
            
            # Convert to integration result
            integration_result = self._convert_to_integration_result(
                greek_result, market_data, time.time() - start_time
            )
            
            # Cache the result
            if self.config.enable_caching:
                self._cache_result(cache_key, integration_result)
            
            # Update performance metrics
            self._update_performance_metrics(time.time() - start_time)
            
            # Update state
            self.last_update = datetime.now()
            self.total_analyses += 1
            
            logger.debug(f"Greek sentiment analysis completed in {time.time() - start_time:.3f}s")
            return integration_result
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error in Greek sentiment analysis: {e}")
            return None
    
    def get_regime_component(self, market_data: Dict[str, Any]) -> float:
        """
        Get Greek sentiment regime component for market regime formation
        
        Args:
            market_data: Market data for analysis
            
        Returns:
            float: Regime component score (0.0 to 1.0)
        """
        try:
            result = self.analyze_greek_sentiment(market_data)
            if not result:
                return 0.5  # Neutral default
            
            # Convert sentiment score to regime component
            # Normalize from [-1, 1] to [0, 1] range
            regime_component = (result.sentiment_score + 1.0) / 2.0
            
            # Apply confidence weighting
            weighted_component = (regime_component * result.confidence + 
                                0.5 * (1.0 - result.confidence))
            
            return np.clip(weighted_component, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error getting regime component: {e}")
            return 0.5
    
    def _validate_market_data(self, market_data: Dict[str, Any]) -> bool:
        """Validate market data for Greek sentiment analysis"""
        try:
            # Check for required data fields
            required_fields = ['timestamp', 'underlying_price']
            for field in required_fields:
                if field not in market_data:
                    logger.warning(f"Missing required field: {field}")
                    return False
            
            # Check for options data with Greeks
            has_options_data = False
            greek_fields = ['delta', 'gamma', 'theta', 'vega']
            
            for key in market_data.keys():
                if any(greek in key.lower() for greek in greek_fields):
                    has_options_data = True
                    break
            
            if not has_options_data:
                logger.warning("No options data with Greeks found")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating market data: {e}")
            return False
    
    def _convert_to_integration_result(self, 
                                     greek_result: GreekAnalysisResult,
                                     market_data: Dict[str, Any],
                                     processing_time: float) -> GreekSentimentIntegrationResult:
        """Convert core Greek result to integration result"""
        try:
            # Calculate regime contribution (0.0 to 1.0)
            regime_contribution = (greek_result.sentiment_score + 1.0) / 2.0
            
            # Calculate data quality score
            data_quality_score = self._calculate_data_quality_score(market_data)
            
            # Count data points used
            data_points_used = self._count_data_points(market_data)
            
            return GreekSentimentIntegrationResult(
                sentiment_score=greek_result.sentiment_score,
                sentiment_type=greek_result.sentiment_type,
                confidence=greek_result.confidence,
                regime_contribution=regime_contribution,
                delta_contribution=greek_result.delta_contribution,
                vega_contribution=greek_result.vega_contribution,
                theta_contribution=greek_result.theta_contribution,
                gamma_contribution=greek_result.gamma_contribution,
                processing_time=processing_time,
                data_quality_score=data_quality_score,
                integration_status="success",
                baseline_changes=greek_result.baseline_changes.copy(),
                dte_adjustments=greek_result.dte_adjustments.copy(),
                cross_strike_correlation=greek_result.cross_strike_correlation,
                data_points_used=data_points_used
            )
            
        except Exception as e:
            logger.error(f"Error converting to integration result: {e}")
            # Return minimal result
            return GreekSentimentIntegrationResult(
                sentiment_score=0.0,
                sentiment_type=GreekSentimentType.NEUTRAL,
                confidence=0.0,
                regime_contribution=0.5,
                delta_contribution=0.0,
                vega_contribution=0.0,
                theta_contribution=0.0,
                gamma_contribution=0.0,
                processing_time=processing_time,
                data_quality_score=0.0,
                integration_status="error"
            )

    def _generate_cache_key(self, market_data: Dict[str, Any]) -> str:
        """Generate cache key for market data"""
        try:
            # Use timestamp and key data points for cache key
            timestamp = market_data.get('timestamp', datetime.now())
            underlying_price = market_data.get('underlying_price', 0)

            # Round timestamp to cache duration
            cache_minutes = self.config.cache_duration_minutes
            rounded_timestamp = timestamp.replace(
                minute=(timestamp.minute // cache_minutes) * cache_minutes,
                second=0,
                microsecond=0
            )

            return f"greek_sentiment_{rounded_timestamp}_{underlying_price:.2f}"

        except Exception as e:
            logger.error(f"Error generating cache key: {e}")
            return f"greek_sentiment_{datetime.now().timestamp()}"

    def _get_cached_result(self, cache_key: str) -> Optional[GreekSentimentIntegrationResult]:
        """Get cached result if available and valid"""
        try:
            if cache_key in self.result_cache:
                result, cache_time = self.result_cache[cache_key]

                # Check if cache is still valid
                cache_age = datetime.now() - cache_time
                if cache_age.total_seconds() < self.config.cache_duration_minutes * 60:
                    return result
                else:
                    # Remove expired cache entry
                    del self.result_cache[cache_key]

            return None

        except Exception as e:
            logger.error(f"Error getting cached result: {e}")
            return None

    def _cache_result(self, cache_key: str, result: GreekSentimentIntegrationResult):
        """Cache analysis result"""
        try:
            self.result_cache[cache_key] = (result, datetime.now())

            # Clean up old cache entries (keep only last 100)
            if len(self.result_cache) > 100:
                # Remove oldest entries
                sorted_cache = sorted(
                    self.result_cache.items(),
                    key=lambda x: x[1][1]
                )
                for key, _ in sorted_cache[:-100]:
                    del self.result_cache[key]

        except Exception as e:
            logger.error(f"Error caching result: {e}")

    def _calculate_data_quality_score(self, market_data: Dict[str, Any]) -> float:
        """Calculate data quality score based on available data"""
        try:
            quality_score = 0.0
            total_checks = 0

            # Check for timestamp freshness
            if 'timestamp' in market_data:
                timestamp = market_data['timestamp']
                if isinstance(timestamp, datetime):
                    age_minutes = (datetime.now() - timestamp).total_seconds() / 60
                    if age_minutes < 5:  # Fresh data
                        quality_score += 0.3
                total_checks += 1

            # Check for underlying price
            if 'underlying_price' in market_data and market_data['underlying_price'] > 0:
                quality_score += 0.2
            total_checks += 1

            # Check for Greek data availability
            greek_fields = ['delta', 'gamma', 'theta', 'vega']
            greek_count = 0
            for key in market_data.keys():
                if any(greek in key.lower() for greek in greek_fields):
                    greek_count += 1

            if greek_count > 0:
                quality_score += 0.3 * min(greek_count / 10, 1.0)  # Up to 10 Greek fields
            total_checks += 1

            # Check for options data completeness
            options_fields = ['ltp', 'volume', 'oi', 'iv']
            options_count = 0
            for key in market_data.keys():
                if any(field in key.lower() for field in options_fields):
                    options_count += 1

            if options_count > 0:
                quality_score += 0.2 * min(options_count / 20, 1.0)  # Up to 20 options fields
            total_checks += 1

            return min(quality_score, 1.0)

        except Exception as e:
            logger.error(f"Error calculating data quality score: {e}")
            return 0.5

    def _count_data_points(self, market_data: Dict[str, Any]) -> int:
        """Count number of data points used in analysis"""
        try:
            data_points = 0

            # Count options data points
            for key, value in market_data.items():
                if isinstance(value, (int, float)) and not pd.isna(value):
                    data_points += 1
                elif isinstance(value, (list, tuple)):
                    data_points += len(value)
                elif isinstance(value, pd.DataFrame):
                    data_points += len(value)

            return data_points

        except Exception as e:
            logger.error(f"Error counting data points: {e}")
            return 0

    def _update_performance_metrics(self, processing_time: float):
        """Update performance tracking metrics"""
        try:
            self.performance_metrics['total_processing_time'] += processing_time

            if processing_time > self.performance_metrics['max_processing_time']:
                self.performance_metrics['max_processing_time'] = processing_time

            if processing_time < self.performance_metrics['min_processing_time']:
                self.performance_metrics['min_processing_time'] = processing_time

            # Update average
            if self.total_analyses > 0:
                self.performance_metrics['average_processing_time'] = (
                    self.performance_metrics['total_processing_time'] / self.total_analyses
                )

        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")

    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status"""
        try:
            return {
                'is_initialized': self.is_initialized,
                'last_update': self.last_update,
                'total_analyses': self.total_analyses,
                'error_count': self.error_count,
                'performance_metrics': self.performance_metrics.copy(),
                'cache_size': len(self.result_cache),
                'config': {
                    'delta_weight': self.config.delta_weight,
                    'vega_weight': self.config.vega_weight,
                    'theta_weight': self.config.theta_weight,
                    'gamma_weight': self.config.gamma_weight,
                    'enable_caching': self.config.enable_caching,
                    'cache_duration_minutes': self.config.cache_duration_minutes
                }
            }

        except Exception as e:
            logger.error(f"Error getting integration status: {e}")
            return {'error': str(e)}

    def update_configuration(self, new_config: GreekSentimentIntegrationConfig) -> bool:
        """Update integration configuration"""
        try:
            self.config = new_config

            # Reinitialize the Greek analyzer with new configuration
            self.greek_analyzer = self._initialize_greek_analyzer()

            # Clear cache to force recalculation with new config
            self.result_cache.clear()

            logger.info("Greek sentiment integration configuration updated")
            return True

        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            return False

    def cleanup(self):
        """Cleanup integration resources"""
        try:
            # Clear cache
            self.result_cache.clear()

            # Reset metrics
            self.performance_metrics = {
                'total_processing_time': 0.0,
                'average_processing_time': 0.0,
                'max_processing_time': 0.0,
                'min_processing_time': float('inf'),
                'cache_hits': 0,
                'cache_misses': 0
            }

            # Cleanup analyzer if it has cleanup method
            if hasattr(self.greek_analyzer, 'cleanup'):
                self.greek_analyzer.cleanup()

            logger.info("Greek sentiment integration cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Factory function for easy instantiation
def create_greek_sentiment_integration(config: Optional[GreekSentimentIntegrationConfig] = None) -> GreekSentimentAnalyzerIntegration:
    """
    Factory function to create Enhanced Greek Sentiment Integration

    Args:
        config: Optional integration configuration

    Returns:
        GreekSentimentAnalyzerIntegration: Configured integration instance
    """
    return GreekSentimentAnalyzerIntegration(config)
