"""
Utilities Module for Adaptive Learning Framework

Provides common utilities for:
- Arrow to cuDF conversion helpers  
- GPU memory management with automatic cleanup
- Multi-timeframe aggregation functions
- Performance monitoring and optimization tools
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Import key utility modules
try:
    from .transforms import (
        ArrowToCuDFConverter,
        MultiTimeframeAggregator,
        get_optimal_chunk_size
    )
    TRANSFORMS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Transform utilities not fully available: {str(e)}")
    TRANSFORMS_AVAILABLE = False

try:
    from .gpu_memory import GPUMemoryManager
    GPU_UTILS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"GPU utilities not available: {str(e)}")
    GPU_UTILS_AVAILABLE = False

try:
    from .performance import PerformanceProfiler, TimingContext
    PERFORMANCE_UTILS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Performance utilities not available: {str(e)}")
    PERFORMANCE_UTILS_AVAILABLE = False


def get_available_utilities() -> Dict[str, bool]:
    """
    Get status of available utility modules.
    
    Returns:
        Dictionary indicating which utility modules are available
    """
    return {
        "transforms": TRANSFORMS_AVAILABLE,
        "gpu_memory": GPU_UTILS_AVAILABLE,
        "performance": PERFORMANCE_UTILS_AVAILABLE
    }


def check_dependencies() -> Dict[str, Any]:
    """
    Check status of optional dependencies.
    
    Returns:
        Dictionary with dependency status and versions
    """
    deps = {}
    
    # Check RAPIDS/cuDF
    try:
        import cudf
        deps["cudf"] = {
            "available": True,
            "version": cudf.__version__
        }
    except ImportError:
        deps["cudf"] = {
            "available": False,
            "version": None
        }
    
    # Check Apache Arrow
    try:
        import pyarrow as pa
        deps["pyarrow"] = {
            "available": True,
            "version": pa.__version__
        }
    except ImportError:
        deps["pyarrow"] = {
            "available": False,
            "version": None
        }
    
    # Check pandas
    try:
        import pandas as pd
        deps["pandas"] = {
            "available": True,
            "version": pd.__version__
        }
    except ImportError:
        deps["pandas"] = {
            "available": False,
            "version": None
        }
    
    return deps


# Export based on availability
__all__ = ["get_available_utilities", "check_dependencies"]

if TRANSFORMS_AVAILABLE:
    __all__.extend([
        "ArrowToCuDFConverter",
        "MultiTimeframeAggregator", 
        "get_optimal_chunk_size"
    ])

if GPU_UTILS_AVAILABLE:
    __all__.extend(["GPUMemoryManager"])

if PERFORMANCE_UTILS_AVAILABLE:
    __all__.extend(["PerformanceProfiler", "TimingContext"])