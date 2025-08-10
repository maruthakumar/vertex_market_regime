"""
Integration Layer - Market Regime Integration System
=================================================

Main integration layer that orchestrates all market regime components.
Provides both new architecture and backward compatibility.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0
"""

from .integrated_engine import IntegratedMarketRegimeEngine
from .market_regime_orchestrator import MarketRegimeOrchestrator
from .component_manager import ComponentManager
from .data_pipeline import DataPipeline
from .result_aggregator import ResultAggregator

# Legacy compatibility
from .legacy_adapter import LegacyIndicatorAdapter
from .configuration_migrator import ConfigurationMigrator

# Comprehensive integration manager for all 9 components
from .comprehensive_integration_manager import (
    ComprehensiveIntegrationManager,
    create_integration_manager
)

__version__ = "2.0.0"
__author__ = "Market Regime Refactoring Team"

__all__ = [
    'MarketRegimeOrchestrator',
    'ComponentManager',
    'DataPipeline',
    'ResultAggregator',
    'IntegratedMarketRegimeEngine',
    'LegacyIndicatorAdapter',
    'ConfigurationMigrator',
    'ComprehensiveIntegrationManager',
    'create_integration_manager'
]