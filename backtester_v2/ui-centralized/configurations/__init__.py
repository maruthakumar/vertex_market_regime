"""
Unified Configuration Management System for Backtester V2

Enterprise-grade configuration management system for all 10 strategy types with:
- Advanced deduplication and similarity detection
- Full-text search with fuzzy matching and autocomplete
- Schema-driven dynamic UI form generation
- Git-like version control with comprehensive history
- Batch processing for 1000s of configuration files
- REST API with real-time progress tracking
- Centralized parameter registry and validation
- Backward compatibility with existing workflows

Strategy Types: TBS, TV, ORB, OI, ML, POS, Market Regime, ML Triple Straddle, 
               Indicator, Strategy Consolidation

Key Features:
- Parameter Registry: 2,847+ parameters across 10 strategies
- Deduplication Engine: Multi-level similarity detection
- Search Engine: Full-text search with semantic relationships
- Form Generator: Dynamic UI generation for multiple frameworks
- Version Control: Git-like versioning with deduplication
- Batch Processing: Parallel processing of large file sets
- API Layer: REST endpoints with WebSocket progress tracking
"""

from .core.config_manager import ConfigurationManager
from .core.base_config import BaseConfiguration
from .core.config_registry import ConfigurationRegistry
from .core.exceptions import (
    ConfigurationError,
    ValidationError,
    ParsingError,
    StorageError,
    LockError
)

# Import strategy configurations
from .strategies import (
    TBSConfiguration,
    TVConfiguration,
    ORBConfiguration,
    OIConfiguration,
    MLConfiguration,
    POSConfiguration,
    MarketRegimeConfiguration,
    MLTripleStraddleConfiguration,
    IndicatorConfiguration,
    StrategyConsolidationConfiguration
)

# Import parsers
from .parsers.base_parser import BaseParser
from .parsers.excel_parser import ExcelParser

# Import unified system components
from .gateway import UnifiedConfigurationGateway, BatchProcessor
from .parameter_registry import ParameterRegistry, ParameterDefinition
from .search import ParameterSearchEngine, SearchQuery
from .deduplication import DeduplicationEngine, DeduplicationReport
from .ui import SchemaFormGenerator, FormConfig, FormFramework

# Initialize and register strategies
def initialize_configurations():
    """Initialize configuration system and register all strategies"""
    manager = ConfigurationManager()
    
    # Register all strategy configurations
    manager.register_configuration_class("tbs", TBSConfiguration)
    manager.register_configuration_class("tv", TVConfiguration)
    manager.register_configuration_class("orb", ORBConfiguration)
    manager.register_configuration_class("oi", OIConfiguration)
    manager.register_configuration_class("ml", MLConfiguration)
    manager.register_configuration_class("pos", POSConfiguration)
    manager.register_configuration_class("market_regime", MarketRegimeConfiguration)
    manager.register_configuration_class("ml_triple_straddle", MLTripleStraddleConfiguration)
    manager.register_configuration_class("indicator", IndicatorConfiguration)
    manager.register_configuration_class("strategy_consolidation", StrategyConsolidationConfiguration)
    
    return manager

# Auto-initialize on import
_manager = initialize_configurations()

# Convenience function to get manager instance
def get_configuration_manager() -> ConfigurationManager:
    """Get the singleton configuration manager instance"""
    return _manager

__version__ = "2.0.0"
__all__ = [
    # Core classes
    "ConfigurationManager",
    "BaseConfiguration",
    "ConfigurationRegistry",
    
    # Unified System Components
    "UnifiedConfigurationGateway",
    "BatchProcessor",
    "ParameterRegistry",
    "ParameterDefinition",
    "ParameterSearchEngine",
    "SearchQuery",
    "DeduplicationEngine",
    "DeduplicationReport",
    "SchemaFormGenerator",
    "FormConfig",
    "FormFramework",
    
    # Exceptions
    "ConfigurationError",
    "ValidationError",
    "ParsingError",
    "StorageError",
    "LockError",
    
    # Strategy configurations
    "TBSConfiguration",
    "TVConfiguration",
    "ORBConfiguration",
    "OIConfiguration",
    "MLConfiguration",
    "POSConfiguration",
    "MarketRegimeConfiguration",
    "MLTripleStraddleConfiguration",
    "IndicatorConfiguration",
    "StrategyConsolidationConfiguration",
    
    # Parsers
    "BaseParser",
    "ExcelParser",
    
    # Functions
    "get_configuration_manager",
    "initialize_configurations"
]