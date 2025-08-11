"""
Adaptive Learning Framework for Market Regime Master Framework

This module provides the foundation infrastructure for the 8-component adaptive learning system,
including schema registry, transform utilities, deterministic transforms, and caching.

Performance Requirements:
- Framework overhead: <50ms
- Total system budget: <800ms processing time
- Memory constraint: <3.7GB total system memory usage
- Component support: 774 features across 8 components (120/98/105/87/94/150/72/48)
"""

from typing import Dict, Any, List, Optional
import logging
import time
from pathlib import Path

# Configure logging for adaptive learning framework
logger = logging.getLogger(__name__)

# Framework version and metadata
FRAMEWORK_VERSION = "1.0.0"
PERFORMANCE_BUDGET_MS = 50  # Framework overhead budget
TOTAL_SYSTEM_BUDGET_MS = 800  # Total system processing budget
MEMORY_CONSTRAINT_GB = 3.7  # Total memory constraint

# Component feature breakdown (774 total features)
COMPONENT_FEATURE_COUNT = {
    "component_01_triple_straddle": 120,
    "component_02_greeks_sentiment": 98,
    "component_03_oi_pa_trending": 105,
    "component_04_iv_skew": 87,
    "component_05_atr_ema_cpr": 94,
    "component_06_correlation": 150,
    "component_07_support_resistance": 72,
    "component_08_master_integration": 48
}

# Verify total feature count
TOTAL_FEATURES = sum(COMPONENT_FEATURE_COUNT.values())
assert TOTAL_FEATURES == 774, f"Feature count mismatch: {TOTAL_FEATURES} != 774"


class AdaptiveFrameworkException(Exception):
    """Base exception for adaptive learning framework errors."""
    pass


class PerformanceBudgetExceeded(AdaptiveFrameworkException):
    """Raised when component exceeds allocated performance budget."""
    pass


class SchemaValidationError(AdaptiveFrameworkException):
    """Raised when feature schema validation fails."""
    pass


class GPUMemoryError(AdaptiveFrameworkException):
    """Raised when GPU memory operations fail."""
    pass


class CacheError(AdaptiveFrameworkException):
    """Raised when cache operations fail."""
    pass


def get_framework_info() -> Dict[str, Any]:
    """
    Get framework information and configuration.
    
    Returns:
        Dictionary containing framework metadata and configuration
    """
    return {
        "version": FRAMEWORK_VERSION,
        "performance_budget_ms": PERFORMANCE_BUDGET_MS,
        "total_system_budget_ms": TOTAL_SYSTEM_BUDGET_MS,
        "memory_constraint_gb": MEMORY_CONSTRAINT_GB,
        "total_features": TOTAL_FEATURES,
        "component_feature_breakdown": COMPONENT_FEATURE_COUNT.copy()
    }


def validate_performance_budget(start_time: float, budget_ms: int, operation_name: str) -> None:
    """
    Validate that operation completed within performance budget.
    
    Args:
        start_time: Operation start time from time.time()
        budget_ms: Budget in milliseconds
        operation_name: Name of operation for logging
        
    Raises:
        PerformanceBudgetExceeded: If operation exceeded budget
    """
    elapsed_ms = (time.time() - start_time) * 1000
    
    # Warn if approaching budget (>90% usage)
    warning_threshold = budget_ms * 0.9
    if elapsed_ms > warning_threshold:
        if elapsed_ms > budget_ms:
            error_msg = f"{operation_name} exceeded budget: {elapsed_ms:.2f}ms > {budget_ms}ms (system constraint: <{TOTAL_SYSTEM_BUDGET_MS}ms)"
            logger.error(error_msg)
            raise PerformanceBudgetExceeded(error_msg)
        else:
            logger.warning(f"{operation_name} approaching budget limit: {elapsed_ms:.2f}ms (>{warning_threshold:.1f}ms threshold)")
    else:
        logger.debug(f"{operation_name} completed within budget: {elapsed_ms:.2f}ms")


# Export key classes and functions
__all__ = [
    "FRAMEWORK_VERSION",
    "PERFORMANCE_BUDGET_MS", 
    "TOTAL_SYSTEM_BUDGET_MS",
    "MEMORY_CONSTRAINT_GB",
    "COMPONENT_FEATURE_COUNT",
    "TOTAL_FEATURES",
    "AdaptiveFrameworkException",
    "PerformanceBudgetExceeded",
    "SchemaValidationError", 
    "GPUMemoryError",
    "CacheError",
    "get_framework_info",
    "validate_performance_budget"
]