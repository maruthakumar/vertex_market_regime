"""
Base class for all regime detectors providing common functionality

This abstract base class provides shared functionality for all regime detectors
including caching, performance monitoring, data validation, and common utilities.
Concrete implementations (12-regime, 18-regime) should inherit from this base.

Author: Market Regime System Optimizer
Date: 2025-07-07
Version: 1.0.0
"""

import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class RegimeType(Enum):
    """Standard regime types used across all detectors"""
    STRONG_BULLISH = "STRONG_BULLISH"
    MODERATE_BULLISH = "MODERATE_BULLISH"
    WEAK_BULLISH = "WEAK_BULLISH"
    NEUTRAL = "NEUTRAL"
    SIDEWAYS = "SIDEWAYS"
    WEAK_BEARISH = "WEAK_BEARISH"
    MODERATE_BEARISH = "MODERATE_BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"
    
    # Extended regime types for 12 and 18 regime systems
    VOLATILE_BULLISH = "VOLATILE_BULLISH"
    VOLATILE_BEARISH = "VOLATILE_BEARISH"
    TRENDING_BULLISH = "TRENDING_BULLISH"
    TRENDING_BEARISH = "TRENDING_BEARISH"
    RANGE_BOUND_HIGH = "RANGE_BOUND_HIGH"
    RANGE_BOUND_LOW = "RANGE_BOUND_LOW"
    BREAKOUT_BULLISH = "BREAKOUT_BULLISH"
    BREAKOUT_BEARISH = "BREAKOUT_BEARISH"
    REVERSAL_BULLISH = "REVERSAL_BULLISH"
    REVERSAL_BEARISH = "REVERSAL_BEARISH"


@dataclass
class RegimeClassification:
    """Base classification result structure"""
    regime_id: str
    regime_name: str
    confidence: float
    timestamp: datetime
    volatility_score: float
    directional_score: float
    alternative_regimes: List[Tuple[str, float]]
    metadata: Dict[str, Any]


class PerformanceMonitor:
    """Performance monitoring for regime detection operations"""
    
    def __init__(self):
        self.metrics = {}
        self.operation_times = {}
        
    def start_operation(self, operation: str):
        """Start timing an operation"""
        self.operation_times[operation] = time.time()
        
    def end_operation(self, operation: str):
        """End timing an operation and record metrics"""
        if operation in self.operation_times:
            duration = time.time() - self.operation_times[operation]
            if operation not in self.metrics:
                self.metrics[operation] = []
            self.metrics[operation].append(duration)
            del self.operation_times[operation]
            return duration
        return 0
        
    def get_average_time(self, operation: str) -> float:
        """Get average time for an operation"""
        if operation in self.metrics and self.metrics[operation]:
            return np.mean(self.metrics[operation])
        return 0
        
    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all metrics"""
        summary = {}
        for operation, times in self.metrics.items():
            if times:
                summary[operation] = {
                    'count': len(times),
                    'average': np.mean(times),
                    'min': min(times),
                    'max': max(times),
                    'total': sum(times)
                }
        return summary


class CacheManager:
    """Simple cache manager for regime detection results"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.cache = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.access_count = {}
        self.hit_count = 0
        self.miss_count = 0
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.ttl_seconds):
                self.hit_count += 1
                self.access_count[key] = self.access_count.get(key, 0) + 1
                return value
            else:
                # Expired
                del self.cache[key]
        self.miss_count += 1
        return None
        
    def set(self, key: str, value: Any):
        """Set value in cache"""
        # Evict least accessed items if cache is full
        if len(self.cache) >= self.max_size:
            # Find least accessed key
            if self.access_count:
                least_accessed = min(self.access_count.items(), key=lambda x: x[1])[0]
                if least_accessed in self.cache:
                    del self.cache[least_accessed]
                    del self.access_count[least_accessed]
                    
        self.cache[key] = (value, datetime.now())
        
    def clear(self):
        """Clear the cache"""
        self.cache.clear()
        self.access_count.clear()
        self.hit_count = 0
        self.miss_count = 0
        
    def get_hit_rate(self) -> float:
        """Get cache hit rate"""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0


class RegimeDetectorBase(ABC):
    """
    Abstract base class for all regime detectors
    
    Provides common functionality:
    - Performance monitoring
    - Caching mechanisms
    - Data validation
    - Logging utilities
    - Common calculations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize base regime detector"""
        self.config = config or {}
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Caching
        cache_config = self.config.get('cache', {})
        self.cache_enabled = cache_config.get('enabled', True)
        self.cache_manager = CacheManager(
            max_size=cache_config.get('max_size', 1000),
            ttl_seconds=cache_config.get('ttl_seconds', 300)
        )
        
        # Regime configuration
        self.confidence_threshold = float(self.config.get('confidence_threshold', 0.6))
        self.regime_smoothing = self.config.get('regime_smoothing', True)
        self.smoothing_window = int(self.config.get('smoothing_window', 3))
        
        # Regime history for smoothing
        self.regime_history = []
        self.max_history_size = 100
        
        # Initialize specific detector
        self._initialize_detector()
        
        logger.info(f"{self.__class__.__name__} initialized with config: {self.config}")
        
    @abstractmethod
    def _initialize_detector(self):
        """Initialize specific detector implementation"""
        pass
        
    @abstractmethod
    def _calculate_regime_internal(self, market_data: Dict[str, Any]) -> RegimeClassification:
        """Internal regime calculation - must be implemented by subclasses"""
        pass
        
    @abstractmethod
    def get_regime_count(self) -> int:
        """Get number of regimes supported by this detector"""
        pass
        
    @abstractmethod
    def get_regime_mapping(self) -> Dict[str, str]:
        """Get mapping of regime IDs to descriptions"""
        pass
        
    def validate_market_data(self, market_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate input market data
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        self.performance_monitor.start_operation('data_validation')
        
        try:
            # Check required fields
            required_fields = ['timestamp', 'underlying_price']
            for field in required_fields:
                if field not in market_data:
                    return False, f"Missing required field: {field}"
                    
            # Validate timestamp
            if not isinstance(market_data['timestamp'], (datetime, pd.Timestamp)):
                return False, "timestamp must be datetime or pd.Timestamp"
                
            # Validate price data
            price = market_data['underlying_price']
            if not isinstance(price, (int, float)) or price <= 0:
                return False, "underlying_price must be positive number"
                
            # Validate option chain if present
            if 'option_chain' in market_data:
                option_chain = market_data['option_chain']
                if not isinstance(option_chain, pd.DataFrame):
                    return False, "option_chain must be pandas DataFrame"
                    
                # Check required columns
                required_columns = ['strike_price', 'option_type', 'last_price']
                missing_columns = set(required_columns) - set(option_chain.columns)
                if missing_columns:
                    return False, f"option_chain missing columns: {missing_columns}"
                    
            return True, None
            
        finally:
            self.performance_monitor.end_operation('data_validation')
            
    def calculate_regime(self, market_data: Dict[str, Any]) -> RegimeClassification:
        """
        Calculate market regime with caching and performance monitoring
        
        Args:
            market_data: Dictionary containing market data
            
        Returns:
            RegimeClassification result
        """
        self.performance_monitor.start_operation('total_calculation')
        
        try:
            # Validate input data
            is_valid, error_msg = self.validate_market_data(market_data)
            if not is_valid:
                logger.error(f"Invalid market data: {error_msg}")
                raise ValueError(f"Invalid market data: {error_msg}")
                
            # Generate cache key
            cache_key = self._generate_cache_key(market_data)
            
            # Check cache if enabled
            if self.cache_enabled and cache_key:
                cached_result = self.cache_manager.get(cache_key)
                if cached_result is not None:
                    logger.debug("Returning cached regime result")
                    return cached_result
                    
            # Calculate regime
            self.performance_monitor.start_operation('regime_calculation')
            result = self._calculate_regime_internal(market_data)
            self.performance_monitor.end_operation('regime_calculation')
            
            # Apply smoothing if enabled
            if self.regime_smoothing:
                result = self._apply_regime_smoothing(result)
                
            # Update history
            self._update_regime_history(result)
            
            # Cache result if enabled
            if self.cache_enabled and cache_key:
                self.cache_manager.set(cache_key, result)
                
            return result
            
        finally:
            duration = self.performance_monitor.end_operation('total_calculation')
            logger.debug(f"Regime calculation completed in {duration:.3f}s")
            
    def _generate_cache_key(self, market_data: Dict[str, Any]) -> Optional[str]:
        """Generate cache key from market data"""
        try:
            # Use timestamp and key price points
            timestamp = market_data['timestamp']
            price = market_data['underlying_price']
            
            # Convert timestamp to string
            if isinstance(timestamp, datetime):
                ts_str = timestamp.isoformat()
            else:
                ts_str = str(timestamp)
                
            # Create key
            key_parts = [
                ts_str,
                f"price_{price:.2f}",
                f"detector_{self.__class__.__name__}"
            ]
            
            # Add option chain summary if present
            if 'option_chain' in market_data:
                chain = market_data['option_chain']
                key_parts.append(f"chain_rows_{len(chain)}")
                
            return "_".join(key_parts)
            
        except Exception as e:
            logger.warning(f"Failed to generate cache key: {e}")
            return None
            
    def _apply_regime_smoothing(self, result: RegimeClassification) -> RegimeClassification:
        """Apply regime smoothing to reduce noise"""
        if len(self.regime_history) < self.smoothing_window:
            return result
            
        # Get recent regimes
        recent_regimes = [r.regime_id for r in self.regime_history[-self.smoothing_window:]]
        recent_regimes.append(result.regime_id)
        
        # Find most common regime
        regime_counts = {}
        for regime in recent_regimes:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
        smoothed_regime = max(regime_counts.items(), key=lambda x: x[1])[0]
        
        # Update result if different
        if smoothed_regime != result.regime_id:
            logger.debug(f"Regime smoothed from {result.regime_id} to {smoothed_regime}")
            result.regime_id = smoothed_regime
            result.regime_name = self.get_regime_mapping().get(smoothed_regime, smoothed_regime)
            result.metadata['smoothed'] = True
            
        return result
        
    def _update_regime_history(self, result: RegimeClassification):
        """Update regime history for smoothing"""
        self.regime_history.append(result)
        
        # Limit history size
        if len(self.regime_history) > self.max_history_size:
            self.regime_history = self.regime_history[-self.max_history_size:]
            
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            'performance': self.performance_monitor.get_metrics_summary(),
            'cache': {
                'enabled': self.cache_enabled,
                'hit_rate': self.cache_manager.get_hit_rate(),
                'size': len(self.cache_manager.cache)
            },
            'regime_history_size': len(self.regime_history)
        }
        
    def reset_cache(self):
        """Reset cache"""
        self.cache_manager.clear()
        logger.info(f"Cache cleared for {self.__class__.__name__}")
        
    def reset_history(self):
        """Reset regime history"""
        self.regime_history.clear()
        logger.info(f"Regime history cleared for {self.__class__.__name__}")