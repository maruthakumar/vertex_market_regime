"""
Unified Excel Configuration System
Integrates YAML conversion, hot reload, and versioning systems
Performance-optimized with evidence-based validation
"""

import os
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

# Import our custom components
from .converters.excel_to_yaml import ExcelToYAMLConverter, ConversionMetrics
from .hot_reload.hot_reload_system import HotReloadSystem, ChangeEvent, create_hot_reload_system, setup_hot_reload_callbacks
from .versioning.version_manager import ConfigurationVersionManager, ConfigVersion, create_version_manager, auto_version_on_change
from .core.config_manager import ConfigurationManager
from .core.exceptions import ConfigurationError, ValidationError

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """Comprehensive system metrics"""
    conversion_metrics: Dict[str, Any]
    hot_reload_metrics: Dict[str, Any]
    version_metrics: Dict[str, Any]
    system_health: Dict[str, Any]
    performance_summary: Dict[str, Any]

class ExcelConfigurationSystem:
    """
    Unified Excel Configuration System
    
    Features:
    - Excel-to-YAML conversion with <100ms performance target
    - Hot reload system with <50ms change detection
    - Configuration versioning and rollback
    - Pandas-based validation
    - Performance monitoring and metrics
    - Thread-safe operations
    - Auto-scaling for large configurations
    """
    
    def __init__(self, base_path: str, enable_hot_reload: bool = True, 
                 enable_versioning: bool = True, max_versions: int = 100):
        """
        Initialize the Excel Configuration System
        
        Args:
            base_path: Base path for configuration files
            enable_hot_reload: Enable hot reload system
            enable_versioning: Enable version management
            max_versions: Maximum versions to keep per configuration
        """
        self.base_path = Path(base_path)
        self.enable_hot_reload = enable_hot_reload
        self.enable_versioning = enable_versioning
        
        # Initialize core components
        self.config_manager = ConfigurationManager()
        self.converter = ExcelToYAMLConverter()
        
        # Initialize hot reload system
        self.hot_reload_system = None
        if enable_hot_reload:
            self.hot_reload_system = create_hot_reload_system(self.config_manager, str(base_path))
            setup_hot_reload_callbacks(self.hot_reload_system)
        
        # Initialize version manager
        self.version_manager = None
        if enable_versioning:
            versions_path = self.base_path / "versions"
            self.version_manager = create_version_manager(str(versions_path), max_versions)
            
            # Setup auto-versioning if hot reload is enabled
            if self.hot_reload_system:
                auto_version_on_change(self.version_manager, self.hot_reload_system)
        
        # System state
        self.started = False
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=6)
        
        # Performance tracking
        self.operation_metrics = []
        self.start_time = time.time()
        
        logger.info(f"ExcelConfigurationSystem initialized at {base_path}")
    
    def start(self) -> None:
        """Start the configuration system"""
        with self.lock:
            if self.started:
                logger.warning("Configuration system already started")
                return
            
            try:
                # Start hot reload system
                if self.hot_reload_system:
                    self.hot_reload_system.start_watching()
                    logger.info("Hot reload system started")
                
                # Perform initial scan and conversion
                self._perform_initial_scan()
                
                self.started = True
                logger.info("Excel Configuration System started successfully")
                
            except Exception as e:
                logger.error(f"Failed to start configuration system: {e}")
                raise
    
    def stop(self) -> None:
        """Stop the configuration system"""
        with self.lock:
            if not self.started:
                return
            
            try:
                # Stop hot reload system
                if self.hot_reload_system:
                    self.hot_reload_system.stop_watching()
                    logger.info("Hot reload system stopped")
                
                # Shutdown executor
                self.executor.shutdown(wait=True)
                
                self.started = False
                logger.info("Excel Configuration System stopped")
                
            except Exception as e:
                logger.error(f"Error stopping configuration system: {e}")
    
    def convert_configuration(self, file_path: str, strategy_type: Optional[str] = None, 
                            create_version: bool = True) -> Tuple[Dict[str, Any], ConversionMetrics]:
        """
        Convert Excel configuration to YAML
        
        Args:
            file_path: Path to Excel file
            strategy_type: Strategy type (auto-detected if None)
            create_version: Create version after conversion
            
        Returns:
            Tuple of (YAML data, conversion metrics)
        """
        start_time = time.time()
        
        try:
            # Convert Excel to YAML
            yaml_data, metrics = self.converter.convert_single_file(file_path, strategy_type)
            
            if not metrics.success:
                raise ConfigurationError(f"Conversion failed: {metrics.error_message}")
            
            # Save YAML file
            yaml_path = Path(file_path).with_suffix('.yml')
            self.converter.save_yaml(yaml_data, str(yaml_path))
            
            # Create version if requested and versioning is enabled
            if create_version and self.version_manager:
                config_name = Path(file_path).stem
                detected_strategy = strategy_type or self.converter._detect_strategy_type(file_path, None)
                
                self.version_manager.create_version(
                    file_path=file_path,
                    strategy_type=detected_strategy,
                    config_name=config_name,
                    user="system",
                    description="Manual conversion",
                    tags=["manual", "conversion"]
                )
            
            # Track performance
            total_time = time.time() - start_time
            self._track_operation("convert_configuration", total_time, True)
            
            logger.info(f"Successfully converted {file_path} in {total_time:.3f}s")
            return yaml_data, metrics
            
        except Exception as e:
            total_time = time.time() - start_time
            self._track_operation("convert_configuration", total_time, False, str(e))
            logger.error(f"Failed to convert {file_path}: {e}")
            raise
    
    def convert_strategy_configurations(self, strategy_type: str, 
                                      create_versions: bool = True) -> Dict[str, Tuple[Dict[str, Any], ConversionMetrics]]:
        """
        Convert all configurations for a strategy type
        
        Args:
            strategy_type: Strategy type to convert
            create_versions: Create versions after conversion
            
        Returns:
            Dict mapping file paths to (YAML data, metrics)
        """
        start_time = time.time()
        
        try:
            # Find strategy files
            strategy_dir = self.base_path / "data" / "prod" / strategy_type
            if not strategy_dir.exists():
                raise FileNotFoundError(f"Strategy directory not found: {strategy_dir}")
            
            excel_files = []
            for pattern in ['*.xlsx', '*.xls', '*.xlsm']:
                excel_files.extend(strategy_dir.glob(pattern))
            
            if not excel_files:
                logger.warning(f"No Excel files found for strategy {strategy_type}")
                return {}
            
            # Convert files concurrently
            results = {}
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_file = {
                    executor.submit(self.convert_configuration, str(file), strategy_type, create_versions): file
                    for file in excel_files
                }
                
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        yaml_data, metrics = future.result()
                        results[str(file_path)] = (yaml_data, metrics)
                    except Exception as e:
                        logger.error(f"Failed to convert {file_path}: {e}")
                        results[str(file_path)] = (None, ConversionMetrics(
                            file_path=str(file_path),
                            file_size=0,
                            sheet_count=0,
                            processing_time=0,
                            validation_time=0,
                            total_time=0,
                            success=False,
                            error_message=str(e)
                        ))
            
            # Track performance
            total_time = time.time() - start_time
            successful_conversions = sum(1 for _, (_, metrics) in results.items() if metrics.success)
            
            self._track_operation("convert_strategy_configurations", total_time, True, 
                                f"Converted {successful_conversions}/{len(excel_files)} files")
            
            logger.info(f"Converted {successful_conversions}/{len(excel_files)} files for {strategy_type} in {total_time:.3f}s")
            return results
            
        except Exception as e:
            total_time = time.time() - start_time
            self._track_operation("convert_strategy_configurations", total_time, False, str(e))
            logger.error(f"Failed to convert strategy configurations for {strategy_type}: {e}")
            raise
    
    def convert_all_configurations(self, create_versions: bool = True) -> Dict[str, Dict[str, Tuple[Dict[str, Any], ConversionMetrics]]]:
        """
        Convert all configurations for all strategy types
        
        Args:
            create_versions: Create versions after conversion
            
        Returns:
            Dict mapping strategy types to conversion results
        """
        start_time = time.time()
        
        try:
            # Find all strategy directories
            prod_dir = self.base_path / "data" / "prod"
            if not prod_dir.exists():
                raise FileNotFoundError(f"Production directory not found: {prod_dir}")
            
            strategy_dirs = [d for d in prod_dir.iterdir() if d.is_dir()]
            
            if not strategy_dirs:
                logger.warning("No strategy directories found")
                return {}
            
            # Convert all strategies
            results = {}
            total_files = 0
            total_successful = 0
            
            for strategy_dir in strategy_dirs:
                strategy_type = strategy_dir.name
                
                try:
                    strategy_results = self.convert_strategy_configurations(strategy_type, create_versions)
                    results[strategy_type] = strategy_results
                    
                    strategy_successful = sum(1 for _, (_, metrics) in strategy_results.items() if metrics.success)
                    total_files += len(strategy_results)
                    total_successful += strategy_successful
                    
                except Exception as e:
                    logger.error(f"Failed to convert strategy {strategy_type}: {e}")
                    results[strategy_type] = {}
            
            # Track performance
            total_time = time.time() - start_time
            self._track_operation("convert_all_configurations", total_time, True, 
                                f"Converted {total_successful}/{total_files} files across {len(strategy_dirs)} strategies")
            
            logger.info(f"Converted {total_successful}/{total_files} files across {len(strategy_dirs)} strategies in {total_time:.3f}s")
            return results
            
        except Exception as e:
            total_time = time.time() - start_time
            self._track_operation("convert_all_configurations", total_time, False, str(e))
            logger.error(f"Failed to convert all configurations: {e}")
            raise
    
    def reload_configuration(self, strategy_type: str, config_name: str) -> bool:
        """
        Manually reload a configuration
        
        Args:
            strategy_type: Strategy type
            config_name: Configuration name
            
        Returns:
            True if successful
        """
        try:
            if self.hot_reload_system:
                # Find the configuration file
                config_dir = self.base_path / "data" / "prod" / strategy_type
                config_file = None
                
                for pattern in ['*.xlsx', '*.xls', '*.xlsm']:
                    files = list(config_dir.glob(f"{config_name}*{pattern[1:]}"))
                    if files:
                        config_file = files[0]
                        break
                
                if not config_file:
                    logger.error(f"Configuration file not found: {strategy_type}/{config_name}")
                    return False
                
                # Force reload
                event = self.hot_reload_system.force_reload(str(config_file))
                return event.success
            else:
                # Manual reload through config manager
                config = self.config_manager.reload_configuration(strategy_type, config_name)
                return config is not None
                
        except Exception as e:
            logger.error(f"Failed to reload configuration {strategy_type}/{config_name}: {e}")
            return False
    
    def get_configuration_status(self, strategy_type: str, config_name: str) -> Dict[str, Any]:
        """
        Get comprehensive status for a configuration
        
        Args:
            strategy_type: Strategy type
            config_name: Configuration name
            
        Returns:
            Status information
        """
        try:
            status = {
                'strategy_type': strategy_type,
                'config_name': config_name,
                'timestamp': datetime.now().isoformat()
            }
            
            # Get configuration from manager
            config = self.config_manager.get_configuration(strategy_type, config_name)
            status['loaded'] = config is not None
            
            if config:
                status['config_info'] = {
                    'created_at': config.created_at.isoformat() if hasattr(config, 'created_at') else None,
                    'modified_at': config.modified_at.isoformat() if hasattr(config, 'modified_at') else None,
                    'valid': True
                }
                
                # Validate configuration
                validation_result = self.config_manager.validate_configuration(config)
                status['config_info']['valid'] = validation_result['valid']
                status['config_info']['validation_errors'] = validation_result.get('errors', [])
                status['config_info']['validation_warnings'] = validation_result.get('warnings', [])
            
            # Get file status
            if self.hot_reload_system:
                config_dir = self.base_path / "data" / "prod" / strategy_type
                for pattern in ['*.xlsx', '*.xls', '*.xlsm']:
                    files = list(config_dir.glob(f"{config_name}*{pattern[1:]}"))
                    if files:
                        file_status = self.hot_reload_system.get_file_status(str(files[0]))
                        status['file_info'] = file_status
                        break
            
            # Get version information
            if self.version_manager:
                versions = self.version_manager.list_versions(strategy_type, config_name, limit=5)
                status['version_info'] = {
                    'total_versions': len(versions),
                    'latest_version': versions[0].to_dict() if versions else None,
                    'recent_versions': [v.to_dict() for v in versions[:3]]
                }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get configuration status for {strategy_type}/{config_name}: {e}")
            return {
                'strategy_type': strategy_type,
                'config_name': config_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get comprehensive system metrics"""
        try:
            # Conversion metrics
            conversion_metrics = self.converter.get_performance_stats()
            
            # Hot reload metrics
            hot_reload_metrics = {}
            if self.hot_reload_system:
                hot_reload_metrics = self.hot_reload_system.get_statistics()
            
            # Version metrics
            version_metrics = {}
            if self.version_manager:
                version_metrics = self.version_manager.get_statistics()
            
            # System health
            system_health = {
                'started': self.started,
                'uptime_seconds': time.time() - self.start_time,
                'hot_reload_enabled': self.enable_hot_reload,
                'versioning_enabled': self.enable_versioning,
                'hot_reload_watching': self.hot_reload_system.watching if self.hot_reload_system else False,
                'total_operations': len(self.operation_metrics),
                'successful_operations': sum(1 for op in self.operation_metrics if op['success']),
                'failed_operations': sum(1 for op in self.operation_metrics if not op['success'])
            }
            
            # Performance summary
            if self.operation_metrics:
                operation_times = [op['duration'] for op in self.operation_metrics if op['success']]
                performance_summary = {
                    'avg_operation_time': sum(operation_times) / len(operation_times) if operation_times else 0,
                    'max_operation_time': max(operation_times) if operation_times else 0,
                    'min_operation_time': min(operation_times) if operation_times else 0,
                    'operations_under_100ms': sum(1 for t in operation_times if t < 0.1),
                    'operations_over_1s': sum(1 for t in operation_times if t > 1.0)
                }
            else:
                performance_summary = {}
            
            return SystemMetrics(
                conversion_metrics=conversion_metrics,
                hot_reload_metrics=hot_reload_metrics,
                version_metrics=version_metrics,
                system_health=system_health,
                performance_summary=performance_summary
            )
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return SystemMetrics(
                conversion_metrics={},
                hot_reload_metrics={},
                version_metrics={},
                system_health={'error': str(e)},
                performance_summary={}
            )
    
    def add_change_callback(self, callback: Callable[[ChangeEvent], None]) -> Optional[str]:
        """
        Add callback for configuration changes
        
        Args:
            callback: Callback function
            
        Returns:
            Callback ID or None if hot reload is disabled
        """
        if self.hot_reload_system:
            return self.hot_reload_system.add_global_callback(callback)
        return None
    
    def remove_change_callback(self, callback: Callable[[ChangeEvent], None]) -> bool:
        """
        Remove change callback
        
        Args:
            callback: Callback function
            
        Returns:
            True if removed successfully
        """
        if self.hot_reload_system:
            self.hot_reload_system.remove_global_callback(callback)
            return True
        return False
    
    def create_configuration_backup(self, strategy_type: str, config_name: str) -> Optional[ConfigVersion]:
        """
        Create backup of configuration
        
        Args:
            strategy_type: Strategy type
            config_name: Configuration name
            
        Returns:
            Created backup version or None if versioning is disabled
        """
        if self.version_manager:
            try:
                return self.version_manager.create_backup(strategy_type, config_name)
            except Exception as e:
                logger.error(f"Failed to create backup for {strategy_type}/{config_name}: {e}")
                return None
        return None
    
    def restore_configuration_version(self, version_id: str, target_path: str) -> bool:
        """
        Restore configuration to specific version
        
        Args:
            version_id: Version ID to restore
            target_path: Target path for restoration
            
        Returns:
            True if successful
        """
        if self.version_manager:
            return self.version_manager.restore_version(version_id, target_path)
        return False
    
    def _perform_initial_scan(self) -> None:
        """Perform initial scan of configuration files"""
        logger.info("Performing initial configuration scan...")
        
        try:
            # Find all Excel files in prod directory
            prod_dir = self.base_path / "data" / "prod"
            if not prod_dir.exists():
                logger.warning(f"Production directory not found: {prod_dir}")
                return
            
            excel_files = []
            for pattern in ['*.xlsx', '*.xls', '*.xlsm']:
                excel_files.extend(prod_dir.rglob(pattern))
            
            if not excel_files:
                logger.info("No Excel files found during initial scan")
                return
            
            # Convert files that don't have YAML equivalents
            files_to_convert = []
            for excel_file in excel_files:
                yaml_file = excel_file.with_suffix('.yml')
                if not yaml_file.exists():
                    files_to_convert.append(excel_file)
            
            if files_to_convert:
                logger.info(f"Converting {len(files_to_convert)} files during initial scan")
                
                # Convert files concurrently
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [
                        executor.submit(self._convert_single_file_safe, str(file))
                        for file in files_to_convert
                    ]
                    
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception as e:
                            logger.error(f"Failed to convert file during initial scan: {e}")
                
                logger.info("Initial configuration scan completed")
            else:
                logger.info("All Excel files already have YAML equivalents")
                
        except Exception as e:
            logger.error(f"Error during initial scan: {e}")
    
    def _convert_single_file_safe(self, file_path: str) -> bool:
        """Safely convert a single file"""
        try:
            yaml_data, metrics = self.converter.convert_single_file(file_path)
            
            if metrics.success:
                yaml_path = Path(file_path).with_suffix('.yml')
                self.converter.save_yaml(yaml_data, str(yaml_path))
                return True
            else:
                logger.error(f"Failed to convert {file_path}: {metrics.error_message}")
                return False
                
        except Exception as e:
            logger.error(f"Exception converting {file_path}: {e}")
            return False
    
    def _track_operation(self, operation: str, duration: float, success: bool, details: str = "") -> None:
        """Track operation metrics"""
        self.operation_metrics.append({
            'operation': operation,
            'duration': duration,
            'success': success,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 1000 operations
        if len(self.operation_metrics) > 1000:
            self.operation_metrics = self.operation_metrics[-1000:]
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()

# Factory functions for easy setup
def create_excel_config_system(base_path: str, **kwargs) -> ExcelConfigurationSystem:
    """
    Create Excel Configuration System with default settings
    
    Args:
        base_path: Base path for configuration files
        **kwargs: Additional configuration options
        
    Returns:
        Configured Excel Configuration System
    """
    return ExcelConfigurationSystem(base_path, **kwargs)

def create_production_config_system(base_path: str) -> ExcelConfigurationSystem:
    """
    Create production-ready Excel Configuration System
    
    Args:
        base_path: Base path for configuration files
        
    Returns:
        Production-configured Excel Configuration System
    """
    return ExcelConfigurationSystem(
        base_path=base_path,
        enable_hot_reload=True,
        enable_versioning=True,
        max_versions=50
    )

def create_development_config_system(base_path: str) -> ExcelConfigurationSystem:
    """
    Create development-optimized Excel Configuration System
    
    Args:
        base_path: Base path for configuration files
        
    Returns:
        Development-configured Excel Configuration System
    """
    return ExcelConfigurationSystem(
        base_path=base_path,
        enable_hot_reload=True,
        enable_versioning=False,  # Disable versioning for development
        max_versions=10
    )

# Usage example
if __name__ == "__main__":
    # Example usage
    config_system = create_production_config_system(
        "/srv/samba/shared/bt/backtester_stable/worktrees/ui-refactor/ui-centralized/backtester_v2/configurations"
    )
    
    try:
        with config_system:
            # Convert all configurations
            results = config_system.convert_all_configurations()
            
            # Get system metrics
            metrics = config_system.get_system_metrics()
            
            print(f"Conversion Results: {len(results)} strategies processed")
            print(f"System Health: {metrics.system_health}")
            print(f"Performance Summary: {metrics.performance_summary}")
            
    except Exception as e:
        logger.error(f"Error running configuration system: {e}")
        raise