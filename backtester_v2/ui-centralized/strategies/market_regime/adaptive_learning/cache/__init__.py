"""
Local Feature Cache System

Provides TTL-based caching for iterative development runs with configurable
policies per component. Optimized for the 8-component adaptive learning system
with development-friendly cache management utilities.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Import cache components
try:
    from .local_cache import LocalFeatureCache, CachePolicy, CacheStats
    CACHE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Cache system not fully available: {str(e)}")
    CACHE_AVAILABLE = False

try:
    from .cache_manager import CacheManager, get_cache_manager
    CACHE_MANAGER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Cache manager not available: {str(e)}")
    CACHE_MANAGER_AVAILABLE = False


def get_cache_status() -> Dict[str, Any]:
    """
    Get status of cache system components.
    
    Returns:
        Dictionary with cache component status
    """
    return {
        "local_cache": CACHE_AVAILABLE,
        "cache_manager": CACHE_MANAGER_AVAILABLE,
        "cache_enabled": CACHE_AVAILABLE and CACHE_MANAGER_AVAILABLE
    }


# Default cache configuration for components
DEFAULT_CACHE_CONFIGS = {
    "component_01_triple_straddle": {
        "ttl_minutes": 15,
        "max_size_mb": 64,
        "enable_persistence": True
    },
    "component_02_greeks_sentiment": {
        "ttl_minutes": 10,
        "max_size_mb": 48,
        "enable_persistence": True
    },
    "component_03_oi_pa_trending": {
        "ttl_minutes": 20,
        "max_size_mb": 56,
        "enable_persistence": True
    },
    "component_04_iv_skew": {
        "ttl_minutes": 12,
        "max_size_mb": 40,
        "enable_persistence": True
    },
    "component_05_atr_ema_cpr": {
        "ttl_minutes": 18,
        "max_size_mb": 52,
        "enable_persistence": True
    },
    "component_06_correlation": {
        "ttl_minutes": 25,  # Longer for expensive correlation calculations
        "max_size_mb": 96,
        "enable_persistence": True
    },
    "component_07_support_resistance": {
        "ttl_minutes": 15,
        "max_size_mb": 36,
        "enable_persistence": True
    },
    "component_08_master_integration": {
        "ttl_minutes": 8,   # Shorter for master integration
        "max_size_mb": 24,
        "enable_persistence": True
    }
}


def get_component_cache_config(component_id: str) -> Dict[str, Any]:
    """
    Get default cache configuration for component.
    
    Args:
        component_id: Component identifier
        
    Returns:
        Cache configuration dictionary
    """
    return DEFAULT_CACHE_CONFIGS.get(component_id, {
        "ttl_minutes": 15,
        "max_size_mb": 48,
        "enable_persistence": True
    })


def create_component_cache(component_id: str, **overrides) -> Optional['LocalFeatureCache']:
    """
    Create cache instance for component with default configuration.
    
    Args:
        component_id: Component identifier
        **overrides: Configuration overrides
        
    Returns:
        LocalFeatureCache instance or None if not available
    """
    if not CACHE_AVAILABLE:
        return None
    
    config = get_component_cache_config(component_id)
    config.update(overrides)
    
    return LocalFeatureCache(component_id=component_id, **config)


# Export based on availability
__all__ = [
    "get_cache_status", 
    "DEFAULT_CACHE_CONFIGS",
    "get_component_cache_config",
    "create_component_cache"
]

if CACHE_AVAILABLE:
    __all__.extend(["LocalFeatureCache", "CachePolicy", "CacheStats"])

if CACHE_MANAGER_AVAILABLE:
    __all__.extend(["CacheManager", "get_cache_manager"])