"""
Redis Caching Layer for Market Regime System

Provides distributed caching for correlation matrices, resistance levels,
and regime calculations to improve performance across multiple instances.

Features:
- Automatic serialization/deserialization of numpy arrays
- TTL-based cache expiration
- Compression for large matrices
- Connection pooling
- Fallback to local cache if Redis unavailable

Author: Market Regime System Optimizer
Date: 2025-07-07
Version: 1.0.0
"""

import redis
import numpy as np
import pandas as pd
import json
import pickle
import zlib
import hashlib
from typing import Dict, Any, Optional, Union, List
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
from functools import wraps
import time

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Redis cache configuration"""
    host: str = 'localhost'
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    decode_responses: bool = False
    max_connections: int = 50
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    connection_pool_kwargs: Optional[Dict] = None
    
    # Cache behavior
    default_ttl: int = 300  # 5 minutes
    compression_threshold: int = 1024  # Compress if larger than 1KB
    compression_level: int = 6
    key_prefix: str = 'market_regime'
    
    # Fallback options
    use_local_fallback: bool = True
    local_cache_size: int = 1000


class LocalCache:
    """Simple LRU local cache as fallback"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = {}
        self.access_times = {}
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from local cache"""
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
        
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in local cache"""
        # Evict oldest if full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
            
        self.cache[key] = value
        self.access_times[key] = time.time()
        
    def delete(self, key: str):
        """Delete from local cache"""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        
    def clear(self):
        """Clear local cache"""
        self.cache.clear()
        self.access_times.clear()


class RedisMatrixCache:
    """
    Redis-based caching for market regime matrices and calculations
    
    Handles serialization of numpy arrays and pandas DataFrames
    with automatic compression for large data.
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """Initialize Redis cache"""
        self.config = config or CacheConfig()
        self.local_cache = LocalCache(self.config.local_cache_size)
        self._connection_pool = None
        self._redis_client = None
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'errors': 0,
            'compressions': 0,
            'local_hits': 0,
            'local_misses': 0
        }
        
        # Initialize connection
        self._initialize_connection()
        
    def _initialize_connection(self):
        """Initialize Redis connection with pooling"""
        try:
            # Create connection pool
            pool_kwargs = {
                'host': self.config.host,
                'port': self.config.port,
                'db': self.config.db,
                'password': self.config.password,
                'decode_responses': self.config.decode_responses,
                'max_connections': self.config.max_connections,
                'socket_timeout': self.config.socket_timeout,
                'socket_connect_timeout': self.config.socket_connect_timeout
            }
            
            if self.config.connection_pool_kwargs:
                pool_kwargs.update(self.config.connection_pool_kwargs)
                
            self._connection_pool = redis.ConnectionPool(**pool_kwargs)
            self._redis_client = redis.Redis(connection_pool=self._connection_pool)
            
            # Test connection
            self._redis_client.ping()
            logger.info(f"Redis connection established to {self.config.host}:{self.config.port}")
            
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            if not self.config.use_local_fallback:
                raise
            logger.info("Using local cache fallback")
            self._redis_client = None
            
    def _generate_key(self, prefix: str, identifier: str) -> str:
        """Generate cache key with prefix"""
        return f"{self.config.key_prefix}:{prefix}:{identifier}"
        
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage"""
        # Handle different types
        if isinstance(value, np.ndarray):
            data = {
                '_type': 'numpy',
                'data': value.tolist(),
                'dtype': str(value.dtype),
                'shape': value.shape
            }
        elif isinstance(value, pd.DataFrame):
            data = {
                '_type': 'dataframe',
                'data': value.to_dict('records'),
                'columns': list(value.columns),
                'index': value.index.tolist()
            }
        elif isinstance(value, pd.Series):
            data = {
                '_type': 'series',
                'data': value.tolist(),
                'index': value.index.tolist(),
                'name': value.name
            }
        else:
            data = {'_type': 'generic', 'data': value}
            
        # Serialize
        serialized = pickle.dumps(data)
        
        # Compress if needed
        if len(serialized) > self.config.compression_threshold:
            serialized = zlib.compress(serialized, self.config.compression_level)
            self.stats['compressions'] += 1
            
        return serialized
        
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage"""
        try:
            # Try decompression first
            try:
                data = zlib.decompress(data)
            except:
                pass  # Not compressed
                
            # Deserialize
            obj = pickle.loads(data)
            
            # Reconstruct based on type
            if obj.get('_type') == 'numpy':
                return np.array(obj['data'], dtype=obj['dtype']).reshape(obj['shape'])
            elif obj.get('_type') == 'dataframe':
                return pd.DataFrame(obj['data'], columns=obj['columns'], index=obj['index'])
            elif obj.get('_type') == 'series':
                return pd.Series(obj['data'], index=obj['index'], name=obj['name'])
            else:
                return obj.get('data', obj)
                
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            return None
            
    def get_correlation_matrix(self, symbol: str, timeframe: int, 
                             timestamp: datetime) -> Optional[np.ndarray]:
        """Get correlation matrix from cache"""
        # Generate key
        ts_str = timestamp.strftime('%Y%m%d_%H%M')
        key = self._generate_key('corr_matrix', f"{symbol}_{timeframe}_{ts_str}")
        
        # Try local cache first
        if self.config.use_local_fallback:
            value = self.local_cache.get(key)
            if value is not None:
                self.stats['local_hits'] += 1
                return value
            self.stats['local_misses'] += 1
            
        # Try Redis
        if self._redis_client:
            try:
                data = self._redis_client.get(key)
                if data:
                    self.stats['hits'] += 1
                    value = self._deserialize_value(data)
                    # Update local cache
                    if self.config.use_local_fallback:
                        self.local_cache.set(key, value)
                    return value
                else:
                    self.stats['misses'] += 1
                    
            except Exception as e:
                logger.error(f"Redis get error: {e}")
                self.stats['errors'] += 1
                
        return None
        
    def set_correlation_matrix(self, symbol: str, timeframe: int,
                             timestamp: datetime, matrix: np.ndarray,
                             ttl: Optional[int] = None):
        """Store correlation matrix in cache"""
        # Generate key
        ts_str = timestamp.strftime('%Y%m%d_%H%M')
        key = self._generate_key('corr_matrix', f"{symbol}_{timeframe}_{ts_str}")
        
        # Update local cache
        if self.config.use_local_fallback:
            self.local_cache.set(key, matrix)
            
        # Update Redis
        if self._redis_client:
            try:
                serialized = self._serialize_value(matrix)
                ttl = ttl or self.config.default_ttl
                self._redis_client.setex(key, ttl, serialized)
                
            except Exception as e:
                logger.error(f"Redis set error: {e}")
                self.stats['errors'] += 1
                
    def get_resistance_levels(self, symbol: str, component: str,
                            timestamp: datetime) -> Optional[Dict[str, List[float]]]:
        """Get resistance levels from cache"""
        ts_str = timestamp.strftime('%Y%m%d_%H%M')
        key = self._generate_key('resistance', f"{symbol}_{component}_{ts_str}")
        
        # Try local first
        if self.config.use_local_fallback:
            value = self.local_cache.get(key)
            if value is not None:
                return value
                
        # Try Redis
        if self._redis_client:
            try:
                data = self._redis_client.get(key)
                if data:
                    return self._deserialize_value(data)
            except Exception as e:
                logger.error(f"Redis get error: {e}")
                
        return None
        
    def set_resistance_levels(self, symbol: str, component: str,
                            timestamp: datetime, levels: Dict[str, List[float]],
                            ttl: Optional[int] = None):
        """Store resistance levels in cache"""
        ts_str = timestamp.strftime('%Y%m%d_%H%M')
        key = self._generate_key('resistance', f"{symbol}_{component}_{ts_str}")
        
        # Update caches
        if self.config.use_local_fallback:
            self.local_cache.set(key, levels)
            
        if self._redis_client:
            try:
                serialized = self._serialize_value(levels)
                ttl = ttl or self.config.default_ttl
                self._redis_client.setex(key, ttl, serialized)
            except Exception as e:
                logger.error(f"Redis set error: {e}")
                
    def cache_decorator(self, prefix: str, ttl: Optional[int] = None):
        """Decorator for caching function results"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key from function arguments
                key_data = f"{func.__name__}_{args}_{kwargs}"
                key_hash = hashlib.md5(str(key_data).encode()).hexdigest()
                cache_key = self._generate_key(prefix, key_hash)
                
                # Try to get from cache
                if self._redis_client:
                    try:
                        cached = self._redis_client.get(cache_key)
                        if cached:
                            self.stats['hits'] += 1
                            return self._deserialize_value(cached)
                    except:
                        pass
                        
                # Calculate result
                result = func(*args, **kwargs)
                
                # Cache result
                if self._redis_client:
                    try:
                        serialized = self._serialize_value(result)
                        cache_ttl = ttl or self.config.default_ttl
                        self._redis_client.setex(cache_key, cache_ttl, serialized)
                    except:
                        pass
                        
                self.stats['misses'] += 1
                return result
                
            return wrapper
        return decorator
        
    def batch_get(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values in one operation"""
        results = {}
        
        # Check local cache first
        if self.config.use_local_fallback:
            for key in keys:
                value = self.local_cache.get(key)
                if value is not None:
                    results[key] = value
                    
        # Get remaining from Redis
        if self._redis_client and len(results) < len(keys):
            remaining_keys = [k for k in keys if k not in results]
            try:
                values = self._redis_client.mget(remaining_keys)
                for key, value in zip(remaining_keys, values):
                    if value:
                        deserialized = self._deserialize_value(value)
                        results[key] = deserialized
                        # Update local cache
                        if self.config.use_local_fallback:
                            self.local_cache.set(key, deserialized)
            except Exception as e:
                logger.error(f"Redis batch get error: {e}")
                
        return results
        
    def clear_pattern(self, pattern: str):
        """Clear all keys matching pattern"""
        if self._redis_client:
            try:
                # Use SCAN to avoid blocking
                cursor = 0
                while True:
                    cursor, keys = self._redis_client.scan(
                        cursor, match=f"{self.config.key_prefix}:{pattern}*"
                    )
                    if keys:
                        self._redis_client.delete(*keys)
                    if cursor == 0:
                        break
            except Exception as e:
                logger.error(f"Redis clear pattern error: {e}")
                
        # Clear local cache
        if self.config.use_local_fallback:
            self.local_cache.clear()
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = self.stats.copy()
        
        # Calculate hit rate
        total = stats['hits'] + stats['misses']
        stats['hit_rate'] = stats['hits'] / total if total > 0 else 0
        
        # Add Redis info if available
        if self._redis_client:
            try:
                info = self._redis_client.info()
                stats['redis_memory'] = info.get('used_memory_human', 'N/A')
                stats['redis_clients'] = info.get('connected_clients', 0)
            except:
                pass
                
        return stats
        
    def close(self):
        """Close Redis connection"""
        if self._connection_pool:
            self._connection_pool.disconnect()
            logger.info("Redis connection closed")