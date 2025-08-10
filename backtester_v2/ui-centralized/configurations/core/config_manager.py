"""
Central configuration management system
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Type, Any, Callable
from datetime import datetime
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib

from .base_config import BaseConfiguration
from .config_registry import ConfigurationRegistry
from .exceptions import (
    ConfigurationError,
    ValidationError,
    StorageError,
    LockError
)

logger = logging.getLogger(__name__)

class ConfigurationManager:
    """
    Singleton configuration manager for all strategy configurations
    
    This class provides centralized management for loading, validating,
    caching, and hot-reloading configurations across all strategy types.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Ensure singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize configuration manager"""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._configs = {}  # {strategy_type: {config_name: config_instance}}
            self._cache = {}    # {cache_key: cached_data}
            self._locks = {}    # {config_key: threading.Lock}
            self._watchers = {} # {config_key: [callback_functions]}
            self._registry = ConfigurationRegistry()
            self._executor = ThreadPoolExecutor(max_workers=5)
            self._storage_backend = 'file'  # Can be 'file', 'database', etc.
            self._base_path = Path("/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/configurations/data")
            self._base_path.mkdir(parents=True, exist_ok=True)
            
            # Configuration change callbacks
            self._global_callbacks = []
            
            logger.info("ConfigurationManager initialized")
    
    def register_configuration_class(self, strategy_type: str, config_class: Type[BaseConfiguration]) -> None:
        """
        Register a configuration class for a strategy type
        
        Args:
            strategy_type: Type of strategy (tbs, tv, orb, etc.)
            config_class: Configuration class for the strategy
        """
        self._registry.register(strategy_type, config_class)
        logger.info(f"Registered configuration class for {strategy_type}")
    
    def load_configuration(self, strategy_type: str, file_path: str, 
                         config_name: Optional[str] = None) -> BaseConfiguration:
        """
        Load a configuration from file
        
        Args:
            strategy_type: Type of strategy
            file_path: Path to configuration file
            config_name: Optional name for the configuration
            
        Returns:
            Loaded configuration instance
        """
        strategy_type = strategy_type.lower()
        
        # Get configuration class
        config_class = self._registry.get_class(strategy_type)
        if not config_class:
            raise ConfigurationError(f"No configuration class registered for {strategy_type}")
        
        # Generate config name if not provided
        if not config_name:
            config_name = Path(file_path).stem
        
        # Create configuration instance
        config = config_class(strategy_type, config_name)
        
        # Load based on file type
        file_path = Path(file_path)
        if file_path.suffix.lower() in ['.xlsx', '.xls']:
            # Load from Excel
            from ..parsers import get_parser
            parser = get_parser(strategy_type)
            data = parser.parse(str(file_path))
            config.from_dict({"data": data})
        elif file_path.suffix.lower() == '.json':
            # Load from JSON
            config.load_from_file(file_path)
        else:
            raise ConfigurationError(f"Unsupported file type: {file_path.suffix}")
        
        # Validate configuration
        config.validate()
        
        # Store in manager
        self._store_configuration(config)
        
        logger.info(f"Loaded configuration: {strategy_type}/{config_name}")
        return config
    
    def save_configuration(self, config: BaseConfiguration, file_path: Optional[str] = None) -> str:
        """
        Save a configuration to file
        
        Args:
            config: Configuration to save
            file_path: Optional file path (uses default if not provided)
            
        Returns:
            Path where configuration was saved
        """
        if not file_path:
            file_path = self._get_default_path(config.strategy_type, config.strategy_name)
        
        file_path = Path(file_path)
        
        # Validate before saving
        config.validate()
        
        # Update modified timestamp
        config.modified_at = datetime.now()
        
        # Save to file
        config.save_to_file(file_path)
        
        # Update stored configuration
        self._store_configuration(config)
        
        # Notify watchers
        self._notify_watchers(config.strategy_type, config.strategy_name, 'saved')
        
        return str(file_path)
    
    def get_configuration(self, strategy_type: str, config_name: str) -> Optional[BaseConfiguration]:
        """
        Get a configuration by type and name
        
        Args:
            strategy_type: Type of strategy
            config_name: Name of configuration
            
        Returns:
            Configuration instance or None if not found
        """
        strategy_type = strategy_type.lower()
        
        if strategy_type in self._configs:
            return self._configs[strategy_type].get(config_name)
        
        return None
    
    def list_configurations(self, strategy_type: Optional[str] = None) -> Dict[str, List[str]]:
        """
        List all configurations
        
        Args:
            strategy_type: Optional filter by strategy type
            
        Returns:
            Dict mapping strategy types to configuration names
        """
        if strategy_type:
            strategy_type = strategy_type.lower()
            if strategy_type in self._configs:
                return {strategy_type: list(self._configs[strategy_type].keys())}
            else:
                return {strategy_type: []}
        
        result = {}
        for st, configs in self._configs.items():
            result[st] = list(configs.keys())
        
        return result
    
    def delete_configuration(self, strategy_type: str, config_name: str) -> bool:
        """
        Delete a configuration
        
        Args:
            strategy_type: Type of strategy
            config_name: Name of configuration
            
        Returns:
            True if deleted, False if not found
        """
        strategy_type = strategy_type.lower()
        
        if strategy_type in self._configs and config_name in self._configs[strategy_type]:
            # Check if locked
            if self._is_locked(strategy_type, config_name):
                raise LockError(f"Configuration {strategy_type}/{config_name} is locked")
            
            # Remove from storage
            del self._configs[strategy_type][config_name]
            
            # Remove file
            file_path = self._get_default_path(strategy_type, config_name)
            if file_path.exists():
                file_path.unlink()
            
            # Clear cache
            self._clear_cache(strategy_type, config_name)
            
            # Notify watchers
            self._notify_watchers(strategy_type, config_name, 'deleted')
            
            logger.info(f"Deleted configuration: {strategy_type}/{config_name}")
            return True
        
        return False
    
    def validate_configuration(self, config: BaseConfiguration) -> Dict[str, Any]:
        """
        Validate a configuration
        
        Args:
            config: Configuration to validate
            
        Returns:
            Validation result with errors if any
        """
        try:
            config.validate()
            return {
                "valid": True,
                "errors": {},
                "warnings": []
            }
        except ValidationError as e:
            return {
                "valid": False,
                "errors": e.errors,
                "warnings": []
            }
    
    def reload_configuration(self, strategy_type: str, config_name: str) -> BaseConfiguration:
        """
        Reload a configuration from disk
        
        Args:
            strategy_type: Type of strategy
            config_name: Name of configuration
            
        Returns:
            Reloaded configuration
        """
        strategy_type = strategy_type.lower()
        
        # Get file path
        file_path = self._get_default_path(strategy_type, config_name)
        
        if not file_path.exists():
            raise ConfigurationError(f"Configuration file not found: {file_path}")
        
        # Load configuration
        config = self.load_configuration(strategy_type, str(file_path), config_name)
        
        # Notify watchers
        self._notify_watchers(strategy_type, config_name, 'reloaded')
        
        return config
    
    def watch_configuration(self, strategy_type: str, config_name: str, 
                          callback: Callable[[str, str, str], None]) -> str:
        """
        Watch a configuration for changes
        
        Args:
            strategy_type: Type of strategy
            config_name: Name of configuration
            callback: Function to call on changes (strategy_type, config_name, event)
            
        Returns:
            Watch ID for unregistering
        """
        key = f"{strategy_type}/{config_name}"
        
        if key not in self._watchers:
            self._watchers[key] = []
        
        watch_id = hashlib.md5(f"{key}_{len(self._watchers[key])}_{datetime.now()}".encode()).hexdigest()
        self._watchers[key].append((watch_id, callback))
        
        logger.debug(f"Added watcher for {key}: {watch_id}")
        return watch_id
    
    def unwatch_configuration(self, watch_id: str) -> bool:
        """
        Remove a configuration watcher
        
        Args:
            watch_id: Watch ID returned by watch_configuration
            
        Returns:
            True if removed, False if not found
        """
        for key, watchers in self._watchers.items():
            for i, (wid, _) in enumerate(watchers):
                if wid == watch_id:
                    del watchers[i]
                    logger.debug(f"Removed watcher {watch_id}")
                    return True
        
        return False
    
    def export_configuration(self, config: BaseConfiguration, format: str = 'json', 
                           file_path: Optional[str] = None) -> str:
        """
        Export configuration to different formats
        
        Args:
            config: Configuration to export
            format: Export format (json, yaml, excel)
            file_path: Optional output path
            
        Returns:
            Path to exported file
        """
        if not file_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{config.strategy_type}_{config.strategy_name}_{timestamp}.{format}"
            file_path = self._base_path / "exports" / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_path = Path(file_path)
        
        if format == 'json':
            config.save_to_file(file_path)
        elif format == 'yaml':
            from ..converters import excel_to_yaml
            yaml_data = excel_to_yaml.convert(config.to_dict())
            with open(file_path, 'w') as f:
                f.write(yaml_data)
        elif format == 'excel':
            from ..converters import json_to_excel
            json_to_excel.convert(config.to_dict(), str(file_path))
        else:
            raise ConfigurationError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported configuration to {file_path}")
        return str(file_path)
    
    def import_configuration(self, file_path: str, strategy_type: str, 
                           config_name: Optional[str] = None) -> BaseConfiguration:
        """
        Import configuration from various formats
        
        Args:
            file_path: Path to import from
            strategy_type: Type of strategy
            config_name: Optional configuration name
            
        Returns:
            Imported configuration
        """
        return self.load_configuration(strategy_type, file_path, config_name)
    
    def clone_configuration(self, strategy_type: str, config_name: str, 
                          new_name: str) -> BaseConfiguration:
        """
        Clone an existing configuration
        
        Args:
            strategy_type: Type of strategy
            config_name: Name of configuration to clone
            new_name: Name for the cloned configuration
            
        Returns:
            Cloned configuration
        """
        original = self.get_configuration(strategy_type, config_name)
        
        if not original:
            raise ConfigurationError(f"Configuration not found: {strategy_type}/{config_name}")
        
        # Create clone
        cloned = original.clone()
        cloned.strategy_name = new_name
        
        # Save and store
        self.save_configuration(cloned)
        
        logger.info(f"Cloned configuration: {strategy_type}/{config_name} -> {new_name}")
        return cloned
    
    def merge_configurations(self, config1: BaseConfiguration, config2: BaseConfiguration, 
                           merged_name: str, strategy: str = 'override') -> BaseConfiguration:
        """
        Merge two configurations
        
        Args:
            config1: First configuration
            config2: Second configuration
            merged_name: Name for merged configuration
            strategy: Merge strategy
            
        Returns:
            Merged configuration
        """
        if config1.strategy_type != config2.strategy_type:
            raise ConfigurationError("Cannot merge configurations of different types")
        
        # Clone first configuration
        merged = config1.clone()
        merged.strategy_name = merged_name
        
        # Merge second configuration
        merged.merge(config2, strategy)
        
        # Validate and save
        merged.validate()
        self.save_configuration(merged)
        
        logger.info(f"Merged configurations: {config1.strategy_name} + {config2.strategy_name} -> {merged_name}")
        return merged
    
    def _store_configuration(self, config: BaseConfiguration) -> None:
        """Store configuration in memory"""
        if config.strategy_type not in self._configs:
            self._configs[config.strategy_type] = {}
        
        self._configs[config.strategy_type][config.strategy_name] = config
    
    def _get_default_path(self, strategy_type: str, config_name: str) -> Path:
        """Get default file path for configuration"""
        return self._base_path / strategy_type / f"{config_name}.json"
    
    def _is_locked(self, strategy_type: str, config_name: str) -> bool:
        """Check if configuration is locked"""
        key = f"{strategy_type}/{config_name}"
        return key in self._locks and self._locks[key].locked()
    
    def _clear_cache(self, strategy_type: str, config_name: str) -> None:
        """Clear cache for configuration"""
        prefix = f"{strategy_type}/{config_name}"
        keys_to_remove = [k for k in self._cache.keys() if k.startswith(prefix)]
        for key in keys_to_remove:
            del self._cache[key]
    
    def _notify_watchers(self, strategy_type: str, config_name: str, event: str) -> None:
        """Notify watchers of configuration changes"""
        key = f"{strategy_type}/{config_name}"
        
        if key in self._watchers:
            for watch_id, callback in self._watchers[key]:
                try:
                    callback(strategy_type, config_name, event)
                except Exception as e:
                    logger.error(f"Error in watcher callback {watch_id}: {e}")
        
        # Notify global callbacks
        for callback in self._global_callbacks:
            try:
                callback(strategy_type, config_name, event)
            except Exception as e:
                logger.error(f"Error in global callback: {e}")
    
    def add_global_callback(self, callback: Callable[[str, str, str], None]) -> None:
        """
        Add a global callback for all configuration changes
        
        Args:
            callback: Function to call on any configuration change
        """
        self._global_callbacks.append(callback)
    
    def remove_global_callback(self, callback: Callable[[str, str, str], None]) -> None:
        """
        Remove a global callback
        
        Args:
            callback: Callback function to remove
        """
        if callback in self._global_callbacks:
            self._global_callbacks.remove(callback)
    
    async def reload_all_async(self) -> Dict[str, Dict[str, bool]]:
        """
        Asynchronously reload all configurations
        
        Returns:
            Dict mapping configurations to reload status
        """
        results = {}
        
        async def reload_single(strategy_type: str, config_name: str) -> tuple:
            try:
                self.reload_configuration(strategy_type, config_name)
                return (strategy_type, config_name, True)
            except Exception as e:
                logger.error(f"Failed to reload {strategy_type}/{config_name}: {e}")
                return (strategy_type, config_name, False)
        
        tasks = []
        for strategy_type, configs in self._configs.items():
            results[strategy_type] = {}
            for config_name in configs:
                task = reload_single(strategy_type, config_name)
                tasks.append(task)
        
        completed = await asyncio.gather(*tasks)
        
        for strategy_type, config_name, success in completed:
            results[strategy_type][config_name] = success
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get configuration manager statistics
        
        Returns:
            Dict containing statistics
        """
        stats = {
            "total_configurations": sum(len(configs) for configs in self._configs.values()),
            "configurations_by_type": {st: len(configs) for st, configs in self._configs.items()},
            "cache_size": len(self._cache),
            "active_watchers": sum(len(watchers) for watchers in self._watchers.values()),
            "global_callbacks": len(self._global_callbacks),
            "storage_backend": self._storage_backend,
            "base_path": str(self._base_path)
        }
        
        return stats
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return f"ConfigurationManager(configs={stats['total_configurations']}, cache={stats['cache_size']})"