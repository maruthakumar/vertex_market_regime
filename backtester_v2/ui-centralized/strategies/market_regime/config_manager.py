"""
Configuration Management System for Market Regime Integration

This module provides centralized configuration management to replace hardcoded paths
and provide environment-specific configurations.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class PathConfig:
    """Path configuration for different components"""
    base_path: str = "/srv/samba/shared/bt/backtester_stable/BTRUN"
    input_sheets_dir: str = "input_sheets/market_regime"
    backtester_v2_dir: str = "backtester_v2"
    strategies_dir: str = "backtester_v2/strategies/market_regime"
    config_dir: str = "backtester_v2/strategies/market_regime/config"
    output_dir: str = "backtester_v2/strategies/market_regime/output"
    temp_dir: str = "backtester_v2/strategies/market_regime/temp"
    
    def get_full_path(self, relative_path: str) -> str:
        """Get full path from relative path"""
        return os.path.join(self.base_path, relative_path)
    
    def get_input_sheets_path(self) -> str:
        """Get full input sheets path"""
        return self.get_full_path(self.input_sheets_dir)
    
    def get_strategies_path(self) -> str:
        """Get full strategies path"""
        return self.get_full_path(self.strategies_dir)

@dataclass
class DatabaseConfig:
    """Database configuration"""
    heavydb_host: str = "localhost"
    heavydb_port: int = 6274
    heavydb_user: str = "admin"
    heavydb_password: str = "HyperInteractive"
    heavydb_database: str = "heavyai"
    heavydb_table: str = "nifty_option_chain"
    connection_pool_size: int = 8
    query_timeout: int = 30
    
@dataclass
class PerformanceConfig:
    """Performance configuration"""
    max_processing_time_ms: float = 200.0
    cache_ttl_seconds: int = 300
    max_memory_mb: int = 2048
    batch_size: int = 1000
    parallel_workers: int = 4
    
@dataclass
class RegimeConfig:
    """Regime detection configuration"""
    regime_mode: str = "18_REGIME"
    timeframes: list = None
    correlation_threshold: float = 0.7
    confidence_threshold: float = 0.6
    min_data_points: int = 100
    
    def __post_init__(self):
        if self.timeframes is None:
            self.timeframes = [3, 5, 10, 15]

class ConfigurationManager:
    """
    Centralized configuration management for Market Regime system
    
    Handles:
    - Environment-specific configurations
    - Path resolution
    - Configuration validation
    - Dynamic configuration updates
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize configuration manager"""
        if not self._initialized:
            self.environment = os.getenv('MARKET_REGIME_ENV', 'development')
            self.config_file = os.getenv('MARKET_REGIME_CONFIG', None)
            
            # Initialize configurations
            self.paths = PathConfig()
            self.database = DatabaseConfig()
            self.performance = PerformanceConfig()
            self.regime = RegimeConfig()
            
            # Load environment-specific overrides
            self._load_environment_config()
            
            # Load from config file if specified
            if self.config_file and os.path.exists(self.config_file):
                self._load_config_file(self.config_file)
            
            # Create necessary directories
            self._ensure_directories()
            
            self._initialized = True
            logger.info(f"Configuration Manager initialized for environment: {self.environment}")
    
    def _load_environment_config(self):
        """Load environment-specific configuration"""
        env_configs = {
            'development': {
                'database': {
                    'heavydb_host': 'localhost',
                    'connection_pool_size': 4
                },
                'performance': {
                    'max_processing_time_ms': 500.0,
                    'parallel_workers': 2
                }
            },
            'staging': {
                'database': {
                    'heavydb_host': os.getenv('HEAVYDB_HOST', 'localhost'),
                    'connection_pool_size': 8
                },
                'performance': {
                    'max_processing_time_ms': 300.0,
                    'parallel_workers': 4
                }
            },
            'production': {
                'database': {
                    'heavydb_host': os.getenv('HEAVYDB_HOST', 'localhost'),
                    'connection_pool_size': 16
                },
                'performance': {
                    'max_processing_time_ms': 200.0,
                    'parallel_workers': 8
                }
            }
        }
        
        if self.environment in env_configs:
            env_config = env_configs[self.environment]
            
            # Update database config
            if 'database' in env_config:
                for key, value in env_config['database'].items():
                    setattr(self.database, key, value)
            
            # Update performance config
            if 'performance' in env_config:
                for key, value in env_config['performance'].items():
                    setattr(self.performance, key, value)
    
    def _load_config_file(self, config_file: str):
        """Load configuration from file"""
        try:
            file_ext = os.path.splitext(config_file)[1].lower()
            
            with open(config_file, 'r') as f:
                if file_ext == '.json':
                    config = json.load(f)
                elif file_ext in ['.yml', '.yaml']:
                    config = yaml.safe_load(f)
                else:
                    logger.warning(f"Unsupported config file type: {file_ext}")
                    return
            
            # Update configurations
            if 'paths' in config:
                for key, value in config['paths'].items():
                    if hasattr(self.paths, key):
                        setattr(self.paths, key, value)
            
            if 'database' in config:
                for key, value in config['database'].items():
                    if hasattr(self.database, key):
                        setattr(self.database, key, value)
            
            if 'performance' in config:
                for key, value in config['performance'].items():
                    if hasattr(self.performance, key):
                        setattr(self.performance, key, value)
            
            if 'regime' in config:
                for key, value in config['regime'].items():
                    if hasattr(self.regime, key):
                        setattr(self.regime, key, value)
            
            logger.info(f"Loaded configuration from: {config_file}")
            
        except Exception as e:
            logger.error(f"Error loading config file {config_file}: {e}")
    
    def _ensure_directories(self):
        """Ensure necessary directories exist"""
        directories = [
            self.paths.get_full_path(self.paths.config_dir),
            self.paths.get_full_path(self.paths.output_dir),
            self.paths.get_full_path(self.paths.temp_dir)
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_excel_config_path(self, filename: str = None) -> str:
        """Get path to Excel configuration file"""
        if filename:
            return os.path.join(self.paths.get_input_sheets_path(), filename)
        
        # Default to the Phase 2 config if no filename specified
        default_config = "PHASE2_ENHANCED_ULTIMATE_UNIFIED_MARKET_REGIME_CONFIG_20250627_195625_20250628_104335.xlsx"
        return os.path.join(self.paths.get_input_sheets_path(), default_config)
    
    def get_output_path(self, filename: str) -> str:
        """Get output file path"""
        return os.path.join(self.paths.get_full_path(self.paths.output_dir), filename)
    
    def get_temp_path(self, filename: str) -> str:
        """Get temporary file path"""
        return os.path.join(self.paths.get_full_path(self.paths.temp_dir), filename)
    
    def get_database_connection_params(self) -> Dict[str, Any]:
        """Get database connection parameters"""
        return {
            'host': self.database.heavydb_host,
            'port': self.database.heavydb_port,
            'user': self.database.heavydb_user,
            'password': self.database.heavydb_password,
            'dbname': self.database.heavydb_database,
            'protocol': 'binary'
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'environment': self.environment,
            'paths': asdict(self.paths),
            'database': asdict(self.database),
            'performance': asdict(self.performance),
            'regime': asdict(self.regime)
        }
    
    def save_config(self, filepath: str):
        """Save current configuration to file"""
        config = self.to_dict()
        
        file_ext = os.path.splitext(filepath)[1].lower()
        with open(filepath, 'w') as f:
            if file_ext == '.json':
                json.dump(config, f, indent=2)
            elif file_ext in ['.yml', '.yaml']:
                yaml.dump(config, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
        
        logger.info(f"Configuration saved to: {filepath}")
    
    def validate_configuration(self) -> tuple[bool, list[str]]:
        """Validate current configuration"""
        errors = []
        
        # Check paths exist
        if not os.path.exists(self.paths.base_path):
            errors.append(f"Base path does not exist: {self.paths.base_path}")
        
        # Check database connectivity (optional)
        # This would require actual connection test
        
        # Check performance limits
        if self.performance.max_processing_time_ms <= 0:
            errors.append("Max processing time must be positive")
        
        if self.performance.batch_size <= 0:
            errors.append("Batch size must be positive")
        
        # Check regime configuration
        if not self.regime.timeframes:
            errors.append("No timeframes configured")
        
        if self.regime.correlation_threshold < 0 or self.regime.correlation_threshold > 1:
            errors.append("Correlation threshold must be between 0 and 1")
        
        return len(errors) == 0, errors

# Convenience function to get singleton instance
def get_config_manager() -> ConfigurationManager:
    """Get the configuration manager instance"""
    return ConfigurationManager()

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Get configuration manager
    config = get_config_manager()
    
    # Print current configuration
    print("Current Configuration:")
    print(json.dumps(config.to_dict(), indent=2))
    
    # Validate configuration
    is_valid, errors = config.validate_configuration()
    if is_valid:
        print("\nConfiguration is valid")
    else:
        print("\nConfiguration errors:")
        for error in errors:
            print(f"  - {error}")
    
    # Example: Get Excel config path
    excel_path = config.get_excel_config_path()
    print(f"\nExcel config path: {excel_path}")
    
    # Example: Save configuration
    config.save_config("market_regime_config.json")