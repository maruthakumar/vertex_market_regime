"""
Environment Configuration Management for Market Regime Components

Centralized environment variable management for production data paths,
cloud configurations, and runtime parameters with validation and fallbacks.
"""

import os
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum


class EnvironmentType(Enum):
    """Environment type enumeration"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class DataPathConfig:
    """Data path configuration structure"""
    production_data_path: str
    backup_data_path: Optional[str]
    output_data_path: str
    cache_data_path: str
    temp_data_path: str
    
    def validate_paths(self) -> bool:
        """Validate that required paths exist and are accessible"""
        try:
            # Check production data path
            if not Path(self.production_data_path).exists():
                logging.warning(f"Production data path does not exist: {self.production_data_path}")
                return False
            
            # Check if output path is writable
            output_path = Path(self.output_data_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Check if cache path is writable
            cache_path = Path(self.cache_data_path)
            cache_path.mkdir(parents=True, exist_ok=True)
            
            # Check if temp path is writable
            temp_path = Path(self.temp_data_path)
            temp_path.mkdir(parents=True, exist_ok=True)
            
            return True
            
        except Exception as e:
            logging.error(f"Path validation failed: {e}")
            return False


@dataclass
class CloudConfig:
    """Cloud configuration structure"""
    project_id: str
    region: str
    storage_bucket: str
    vertex_ai_endpoint: Optional[str]
    bigquery_dataset: Optional[str]
    secret_manager_enabled: bool
    
    
@dataclass
class PerformanceConfig:
    """Performance configuration structure"""
    processing_budget_ms: int
    memory_budget_mb: int
    gpu_enabled: bool
    max_workers: int
    batch_size: int


class EnvironmentConfigManager:
    """
    Centralized environment configuration manager
    
    Manages all environment variables, data paths, and configuration
    with proper validation, defaults, and environment-specific overrides.
    """
    
    def __init__(self, env_type: Optional[EnvironmentType] = None):
        """
        Initialize environment configuration manager
        
        Args:
            env_type: Environment type (auto-detected if not provided)
        """
        self.logger = logging.getLogger(__name__)
        
        # Determine environment type
        self.env_type = env_type or self._detect_environment_type()
        self.logger.info(f"Initializing environment configuration for: {self.env_type.value}")
        
        # Load configurations
        self.data_paths = self._load_data_path_config()
        self.cloud_config = self._load_cloud_config()
        self.performance_config = self._load_performance_config()
        
        # Validate configurations
        self._validate_configuration()
        
    def _detect_environment_type(self) -> EnvironmentType:
        """Auto-detect environment type from environment variables"""
        env_name = os.getenv('MARKET_REGIME_ENV', 'development').lower()
        
        env_mapping = {
            'dev': EnvironmentType.DEVELOPMENT,
            'development': EnvironmentType.DEVELOPMENT,
            'staging': EnvironmentType.STAGING,
            'prod': EnvironmentType.PRODUCTION,
            'production': EnvironmentType.PRODUCTION,
            'test': EnvironmentType.TESTING,
            'testing': EnvironmentType.TESTING
        }
        
        return env_mapping.get(env_name, EnvironmentType.DEVELOPMENT)
    
    def _load_data_path_config(self) -> DataPathConfig:
        """Load data path configuration from environment variables"""
        
        # Default paths based on environment type
        if self.env_type == EnvironmentType.PRODUCTION:
            default_prod_path = "/data/production/market_regime/parquet"
            default_output_path = "/data/production/market_regime/output"
        elif self.env_type == EnvironmentType.STAGING:
            default_prod_path = "/data/staging/market_regime/parquet"
            default_output_path = "/data/staging/market_regime/output"
        else:
            # Development/Testing defaults
            default_prod_path = "/Users/maruth/projects/market_regime/data/nifty_validation/backtester_processed"
            default_output_path = "/Users/maruth/projects/market_regime/output"
        
        return DataPathConfig(
            production_data_path=os.getenv(
                'MARKET_REGIME_DATA_PATH', 
                default_prod_path
            ),
            backup_data_path=os.getenv('MARKET_REGIME_BACKUP_DATA_PATH'),
            output_data_path=os.getenv(
                'MARKET_REGIME_OUTPUT_PATH',
                default_output_path
            ),
            cache_data_path=os.getenv(
                'MARKET_REGIME_CACHE_PATH',
                f"{default_output_path}/cache"
            ),
            temp_data_path=os.getenv(
                'MARKET_REGIME_TEMP_PATH',
                f"{default_output_path}/temp"
            )
        )
    
    def _load_cloud_config(self) -> CloudConfig:
        """Load cloud configuration from environment variables"""
        
        # Default project based on environment
        default_project_mapping = {
            EnvironmentType.PRODUCTION: "arched-bot-269016",
            EnvironmentType.STAGING: "arched-bot-269016-staging", 
            EnvironmentType.DEVELOPMENT: "arched-bot-269016-dev",
            EnvironmentType.TESTING: "arched-bot-269016-test"
        }
        
        return CloudConfig(
            project_id=os.getenv(
                'GCP_PROJECT_ID',
                default_project_mapping.get(self.env_type, "arched-bot-269016-dev")
            ),
            region=os.getenv('GCP_REGION', 'us-central1'),
            storage_bucket=os.getenv(
                'GCS_BUCKET',
                f"vertex-mr-{self.env_type.value}-data"
            ),
            vertex_ai_endpoint=os.getenv('VERTEX_AI_ENDPOINT'),
            bigquery_dataset=os.getenv(
                'BIGQUERY_DATASET',
                f"market_regime_{self.env_type.value}"
            ),
            secret_manager_enabled=os.getenv('USE_SECRET_MANAGER', 'true').lower() == 'true'
        )
    
    def _load_performance_config(self) -> PerformanceConfig:
        """Load performance configuration from environment variables"""
        
        # Environment-specific defaults
        performance_defaults = {
            EnvironmentType.PRODUCTION: {
                'processing_budget_ms': 120,
                'memory_budget_mb': 280,
                'max_workers': 8,
                'batch_size': 1000
            },
            EnvironmentType.STAGING: {
                'processing_budget_ms': 150,
                'memory_budget_mb': 320,
                'max_workers': 4,
                'batch_size': 500
            },
            EnvironmentType.DEVELOPMENT: {
                'processing_budget_ms': 200,
                'memory_budget_mb': 400,
                'max_workers': 2,
                'batch_size': 100
            },
            EnvironmentType.TESTING: {
                'processing_budget_ms': 500,
                'memory_budget_mb': 512,
                'max_workers': 1,
                'batch_size': 50
            }
        }
        
        defaults = performance_defaults[self.env_type]
        
        return PerformanceConfig(
            processing_budget_ms=int(os.getenv(
                'COMPONENT_PROCESSING_BUDGET_MS', 
                defaults['processing_budget_ms']
            )),
            memory_budget_mb=int(os.getenv(
                'COMPONENT_MEMORY_BUDGET_MB',
                defaults['memory_budget_mb']
            )),
            gpu_enabled=os.getenv('GPU_ENABLED', 'false').lower() == 'true',
            max_workers=int(os.getenv(
                'MAX_WORKERS',
                defaults['max_workers']
            )),
            batch_size=int(os.getenv(
                'BATCH_SIZE',
                defaults['batch_size']
            ))
        )
    
    def _validate_configuration(self):
        """Validate all loaded configurations"""
        try:
            # Validate data paths
            if not self.data_paths.validate_paths():
                self.logger.warning("Data path validation failed - some paths may be inaccessible")
            
            # Validate cloud config
            if not self.cloud_config.project_id:
                raise ValueError("GCP Project ID is required")
            
            # Validate performance config
            if self.performance_config.processing_budget_ms <= 0:
                raise ValueError("Processing budget must be positive")
            
            if self.performance_config.memory_budget_mb <= 0:
                raise ValueError("Memory budget must be positive")
                
            self.logger.info("✅ Environment configuration validation passed")
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            raise
    
    def get_production_data_path(self) -> str:
        """Get validated production data path"""
        return self.data_paths.production_data_path
    
    def get_output_path(self) -> str:
        """Get validated output path"""
        return self.data_paths.output_data_path
    
    def get_cache_path(self) -> str:
        """Get validated cache path"""
        return self.data_paths.cache_data_path
    
    def get_component_config(self, component_id: int) -> Dict[str, Any]:
        """
        Get environment-aware configuration for a specific component
        
        Args:
            component_id: Component ID (1-8)
            
        Returns:
            Component-specific configuration dictionary
        """
        return {
            'component_id': component_id,
            'environment': self.env_type.value,
            'data_path': self.data_paths.production_data_path,
            'output_path': self.data_paths.output_data_path,
            'cache_path': self.data_paths.cache_data_path,
            'temp_path': self.data_paths.temp_data_path,
            'processing_budget_ms': self.performance_config.processing_budget_ms,
            'memory_budget_mb': self.performance_config.memory_budget_mb,
            'gpu_enabled': self.performance_config.gpu_enabled,
            'batch_size': self.performance_config.batch_size,
            'project_id': self.cloud_config.project_id,
            'region': self.cloud_config.region,
            'storage_bucket': self.cloud_config.storage_bucket,
            'max_workers': self.performance_config.max_workers,
            'cloud_enabled': True
        }
    
    def get_all_paths(self) -> Dict[str, str]:
        """Get all configured paths"""
        return {
            'production_data': self.data_paths.production_data_path,
            'backup_data': self.data_paths.backup_data_path,
            'output': self.data_paths.output_data_path,
            'cache': self.data_paths.cache_data_path,
            'temp': self.data_paths.temp_data_path
        }
    
    def update_environment_variable(self, key: str, value: str, persist: bool = False):
        """
        Update environment variable at runtime
        
        Args:
            key: Environment variable key
            value: New value
            persist: Whether to persist to .env file
        """
        os.environ[key] = value
        self.logger.info(f"Updated environment variable: {key}")
        
        if persist:
            self._persist_to_env_file(key, value)
    
    def _persist_to_env_file(self, key: str, value: str):
        """Persist environment variable to .env file"""
        try:
            env_file_path = Path(".env")
            
            # Read existing content
            lines = []
            if env_file_path.exists():
                with open(env_file_path, 'r') as f:
                    lines = f.readlines()
            
            # Update or add the key
            key_found = False
            for i, line in enumerate(lines):
                if line.startswith(f"{key}="):
                    lines[i] = f"{key}={value}\n"
                    key_found = True
                    break
            
            if not key_found:
                lines.append(f"{key}={value}\n")
            
            # Write back to file
            with open(env_file_path, 'w') as f:
                f.writelines(lines)
                
            self.logger.info(f"Persisted {key} to .env file")
            
        except Exception as e:
            self.logger.error(f"Failed to persist {key} to .env file: {e}")
    
    def print_configuration_summary(self):
        """Print comprehensive configuration summary"""
        print(f"\n{'='*60}")
        print(f"MARKET REGIME ENVIRONMENT CONFIGURATION")
        print(f"{'='*60}")
        print(f"Environment Type: {self.env_type.value.upper()}")
        print(f"\nDATA PATHS:")
        print(f"  Production Data: {self.data_paths.production_data_path}")
        print(f"  Output Path:     {self.data_paths.output_data_path}")
        print(f"  Cache Path:      {self.data_paths.cache_data_path}")
        print(f"  Temp Path:       {self.data_paths.temp_data_path}")
        if self.data_paths.backup_data_path:
            print(f"  Backup Path:     {self.data_paths.backup_data_path}")
        
        print(f"\nCLOUD CONFIG:")
        print(f"  Project ID:      {self.cloud_config.project_id}")
        print(f"  Region:          {self.cloud_config.region}")
        print(f"  Storage Bucket:  {self.cloud_config.storage_bucket}")
        print(f"  BigQuery Dataset: {self.cloud_config.bigquery_dataset}")
        
        print(f"\nPERFORMANCE CONFIG:")
        print(f"  Processing Budget: {self.performance_config.processing_budget_ms}ms")
        print(f"  Memory Budget:     {self.performance_config.memory_budget_mb}MB")
        print(f"  GPU Enabled:       {self.performance_config.gpu_enabled}")
        print(f"  Max Workers:       {self.performance_config.max_workers}")
        print(f"  Batch Size:        {self.performance_config.batch_size}")
        print(f"{'='*60}\n")


# Global environment manager instance
_env_manager: Optional[EnvironmentConfigManager] = None


def get_environment_manager(refresh: bool = False) -> EnvironmentConfigManager:
    """
    Get global environment configuration manager instance
    
    Args:
        refresh: Whether to refresh the configuration
        
    Returns:
        EnvironmentConfigManager instance
    """
    global _env_manager
    
    if _env_manager is None or refresh:
        _env_manager = EnvironmentConfigManager()
    
    return _env_manager


def get_production_data_path() -> str:
    """Convenience function to get production data path"""
    return get_environment_manager().get_production_data_path()


def get_component_config(component_id: int) -> Dict[str, Any]:
    """Convenience function to get component configuration"""
    return get_environment_manager().get_component_config(component_id)


# Example environment variables that can be set:
EXAMPLE_ENV_VARS = """
# Environment Configuration Examples

# Environment type
export MARKET_REGIME_ENV=production  # development, staging, production, testing

# Data paths
export MARKET_REGIME_DATA_PATH="/data/production/market_regime/parquet"
export MARKET_REGIME_OUTPUT_PATH="/data/production/market_regime/output" 
export MARKET_REGIME_CACHE_PATH="/data/production/market_regime/cache"
export MARKET_REGIME_BACKUP_DATA_PATH="/backup/market_regime/data"

# Cloud configuration
export GCP_PROJECT_ID="arched-bot-269016"
export GCP_REGION="us-central1"
export GCS_BUCKET="vertex-mr-production-data"
export BIGQUERY_DATASET="market_regime_production"
export USE_SECRET_MANAGER=true

# Performance configuration
export COMPONENT_PROCESSING_BUDGET_MS=120
export COMPONENT_MEMORY_BUDGET_MB=280
export GPU_ENABLED=true
export MAX_WORKERS=8
export BATCH_SIZE=1000
"""