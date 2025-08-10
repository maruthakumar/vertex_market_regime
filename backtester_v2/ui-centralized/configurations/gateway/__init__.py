"""
Unified Configuration Gateway

Provides a unified interface for configuration management that orchestrates
all the underlying systems while preserving backward compatibility.
"""

from .unified_gateway import UnifiedConfigurationGateway
from .strategy_detector import StrategyDetector
from .batch_processor import BatchProcessor

__all__ = [
    'UnifiedConfigurationGateway',
    'StrategyDetector', 
    'BatchProcessor'
]