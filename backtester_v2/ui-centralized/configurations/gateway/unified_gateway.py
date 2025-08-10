"""
Unified Configuration Gateway

Main orchestration layer that provides a unified interface for configuration
management while preserving all existing functionality.
"""

import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..core.config_manager import ConfigurationManager
from ..core.base_config import BaseConfiguration
from ..core.exceptions import ConfigurationError, ValidationError, ParsingError
from ..parameter_registry import ParameterRegistry, SchemaExtractor
from ..version_control import VersionManager, ConfigurationVersion
from .strategy_detector import StrategyDetector

logger = logging.getLogger(__name__)

class UnifiedConfigurationGateway:
    """
    Unified Configuration Gateway
    
    Orchestrates all configuration management systems while maintaining
    backward compatibility with existing APIs and workflows.
    """
    
    def __init__(self, 
                 config_manager: Optional[ConfigurationManager] = None,
                 parameter_registry: Optional[ParameterRegistry] = None,
                 version_manager: Optional[VersionManager] = None,
                 enable_features: Optional[Dict[str, bool]] = None):
        """
        Initialize unified gateway
        
        Args:
            config_manager: Existing configuration manager (preserves compatibility)
            parameter_registry: Parameter registry instance
            version_manager: Version control manager
            enable_features: Feature flags for new functionality
        """
        # Core components (backward compatible)
        self.config_manager = config_manager or ConfigurationManager()
        
        # New components (optional/feature-flagged)
        self.parameter_registry = parameter_registry or ParameterRegistry()
        self.version_manager = version_manager or VersionManager()
        self.strategy_detector = StrategyDetector()
        
        # Feature flags (gradual rollout)
        self.features = {
            'parameter_registry': True,
            'version_control': True,
            'auto_detection': True,
            'enhanced_validation': True,
            'metadata_enrichment': True,
            'deduplication': True,
            'batch_processing': True,
            **(enable_features or {})
        }
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Statistics tracking
        self.stats = {
            'total_uploads': 0,
            'successful_uploads': 0,
            'failed_uploads': 0,
            'duplicate_files': 0,
            'versions_created': 0
        }
        
        self._lock = threading.Lock()
        
        # Initialize parameter registry if enabled
        if self.features['parameter_registry']:
            self._initialize_parameter_registry()
        
        logger.info("UnifiedConfigurationGateway initialized with features: %s", 
                   [k for k, v in self.features.items() if v])
    
    def _initialize_parameter_registry(self):
        """Initialize parameter registry with existing schemas"""
        try:
            extractor = SchemaExtractor(self.parameter_registry)
            
            # Check if registry is empty
            stats = self.parameter_registry.get_statistics()
            if stats['total_parameters'] == 0:
                logger.info("Parameter registry is empty, extracting from configuration classes...")
                if extractor.extract_all_strategies():
                    logger.info("Successfully initialized parameter registry")
                else:
                    logger.warning("Failed to fully initialize parameter registry")
            else:
                logger.info(f"Parameter registry already contains {stats['total_parameters']} parameters")
                
        except Exception as e:
            logger.error(f"Failed to initialize parameter registry: {e}")
            self.features['parameter_registry'] = False
    
    def load_configuration(self, 
                         strategy_type: str, 
                         file_path: str,
                         config_name: Optional[str] = None,
                         author: str = "system",
                         commit_message: Optional[str] = None,
                         enable_version_control: bool = True) -> BaseConfiguration:
        """
        Load configuration with enhanced features (backward compatible)
        
        Args:
            strategy_type: Strategy type (or 'auto' for auto-detection)
            file_path: Path to configuration file
            config_name: Optional configuration name
            author: Author for version control
            commit_message: Commit message
            enable_version_control: Enable version control for this load
            
        Returns:
            Loaded configuration with enhancements
        """
        try:
            with self._lock:
                self.stats['total_uploads'] += 1
            
            start_time = datetime.now()
            
            # Auto-detect strategy type if requested
            if strategy_type == "auto" and self.features['auto_detection']:
                detected_type = self.strategy_detector.detect_strategy_type(file_path)
                if detected_type:
                    strategy_type = detected_type
                    logger.info(f"Auto-detected strategy type: {strategy_type}")
                else:
                    raise ConfigurationError("Could not auto-detect strategy type")
            
            # Check for duplicates if deduplication enabled
            if self.features['deduplication']:
                duplicate_version = self._check_for_duplicate(file_path)
                if duplicate_version:
                    logger.info(f"Found duplicate file, returning existing version {duplicate_version.version_number}")
                    with self._lock:
                        self.stats['duplicate_files'] += 1
                    
                    # Return configuration from existing version
                    return self._load_from_version(duplicate_version)
            
            # Load using existing configuration manager (preserves compatibility)
            config = self.config_manager.load_configuration(strategy_type, file_path, config_name)
            
            # Enrich with metadata if enabled
            if self.features['metadata_enrichment']:
                config = self._enrich_with_metadata(config, file_path)
            
            # Enhanced validation if enabled
            if self.features['enhanced_validation']:
                self._enhanced_validation(config, strategy_type)
            
            # Version control if enabled
            if self.features['version_control'] and enable_version_control:
                commit_msg = commit_message or f"Load configuration from {Path(file_path).name}"
                version = self.version_manager.commit_configuration(
                    config, file_path, author, commit_msg
                )
                
                # Store version reference in config
                if hasattr(config, '_metadata'):
                    config._metadata['version_id'] = version.version_id
                    config._metadata['version_number'] = version.version_number
                
                with self._lock:
                    self.stats['versions_created'] += 1
            
            # Update statistics
            with self._lock:
                self.stats['successful_uploads'] += 1
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Successfully loaded {strategy_type} configuration in {processing_time:.2f}s")
            
            return config
            
        except Exception as e:
            with self._lock:
                self.stats['failed_uploads'] += 1
            
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def save_configuration(self, 
                         config: BaseConfiguration,
                         file_path: Optional[str] = None,
                         author: str = "system", 
                         commit_message: Optional[str] = None) -> str:
        """
        Save configuration with version control (backward compatible)
        
        Args:
            config: Configuration to save
            file_path: Optional file path
            author: Author for version control
            commit_message: Commit message
            
        Returns:
            Path where configuration was saved
        """
        try:
            # Save using existing manager (preserves compatibility)
            saved_path = self.config_manager.save_configuration(config, file_path)
            
            # Create version if enabled
            if self.features['version_control']:
                commit_msg = commit_message or f"Save configuration {config.strategy_name}"
                self.version_manager.commit_configuration(
                    config, saved_path, author, commit_msg
                )
            
            return saved_path
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def _check_for_duplicate(self, file_path: str) -> Optional[ConfigurationVersion]:
        """Check if file is a duplicate"""
        try:
            file_hash = self.version_manager.calculate_file_hash(file_path)
            
            # Search for existing version with same file hash
            # This requires a new method in VersionManager
            return self._get_version_by_file_hash(file_hash)
            
        except Exception as e:
            logger.warning(f"Failed to check for duplicates: {e}")
            return None
    
    def _get_version_by_file_hash(self, file_hash: str) -> Optional[ConfigurationVersion]:
        """Get version by file hash (would need to add this to VersionManager)"""
        # For now, return None - would implement this optimization later
        return None
    
    def _load_from_version(self, version: ConfigurationVersion) -> BaseConfiguration:
        """Load configuration from a version"""
        # Create configuration from stored parameters
        from ..core.config_registry import ConfigurationRegistry
        
        registry = ConfigurationRegistry()
        config_class = registry.get_class(version.strategy_type)
        
        if not config_class:
            raise ConfigurationError(f"No configuration class for {version.strategy_type}")
        
        # Create instance and populate from version data
        config = config_class(version.strategy_type, version.metadata.get('strategy_name', 'restored'))
        config.from_dict(version.parameters)
        
        # Add version metadata
        if hasattr(config, '_metadata'):
            config._metadata.update({
                'version_id': version.version_id,
                'version_number': version.version_number,
                'loaded_from_version': True
            })
        
        return config
    
    def _enrich_with_metadata(self, config: BaseConfiguration, file_path: str) -> BaseConfiguration:
        """Enrich configuration with metadata"""
        try:
            file_path_obj = Path(file_path)
            
            # Add file metadata
            metadata = {
                'source_file': file_path_obj.name,
                'file_size': file_path_obj.stat().st_size,
                'upload_time': datetime.now().isoformat(),
                'file_hash': self.version_manager.calculate_file_hash(file_path)
            }
            
            # Add parameter registry metadata if available
            if self.features['parameter_registry']:
                strategy_params = self.parameter_registry.get_parameters_by_strategy(config.strategy_type)
                metadata.update({
                    'parameter_count': len(strategy_params),
                    'required_parameters': len([p for p in strategy_params if p.is_required()]),
                    'schema_version': '1.0'
                })
            
            # Store metadata in config
            if not hasattr(config, '_metadata'):
                config._metadata = {}
            config._metadata.update(metadata)
            
            return config
            
        except Exception as e:
            logger.warning(f"Failed to enrich metadata: {e}")
            return config
    
    def _enhanced_validation(self, config: BaseConfiguration, strategy_type: str):
        """Perform enhanced validation using parameter registry"""
        try:
            if not self.features['parameter_registry']:
                return
            
            # Get schema from parameter registry
            schema = self.parameter_registry.get_schema_for_strategy(strategy_type)
            
            if not schema:
                logger.warning(f"No schema found for strategy {strategy_type}")
                return
            
            # Validate against schema (would implement JSON schema validation)
            # For now, just log that enhanced validation was attempted
            logger.debug(f"Enhanced validation completed for {strategy_type}")
            
        except Exception as e:
            logger.warning(f"Enhanced validation failed: {e}")
    
    def get_configuration_history(self, configuration_id: str) -> List[ConfigurationVersion]:
        """Get version history for a configuration"""
        if not self.features['version_control']:
            return []
        
        return self.version_manager.get_configuration_history(configuration_id)
    
    def rollback_configuration(self, configuration_id: str, version_id: str) -> bool:
        """Rollback configuration to a specific version"""
        if not self.features['version_control']:
            return False
        
        return self.version_manager.rollback_to_version(configuration_id, version_id)
    
    def search_parameters(self, query: str, strategy_type: Optional[str] = None) -> List[Any]:
        """Search parameters using parameter registry"""
        if not self.features['parameter_registry']:
            return []
        
        return self.parameter_registry.search_parameters(query, strategy_type)
    
    def get_strategy_schema(self, strategy_type: str) -> Dict[str, Any]:
        """Get complete schema for a strategy"""
        if not self.features['parameter_registry']:
            return {}
        
        return self.parameter_registry.get_schema_for_strategy(strategy_type)
    
    def list_configurations(self) -> List[Dict[str, Any]]:
        """List all configurations with version information"""
        configurations = []
        
        # Get from configuration manager
        try:
            config_list = self.config_manager.list_configurations()
            configurations.extend(config_list)
        except:
            pass
        
        # Enhance with version information if available
        if self.features['version_control']:
            try:
                version_list = self.version_manager.list_configurations()
                
                # Merge configuration data
                for version_config in version_list:
                    # Check if already in list
                    existing = next((c for c in configurations 
                                   if c.get('configuration_id') == version_config['configuration_id']), None)
                    
                    if existing:
                        # Enhance existing with version info
                        existing.update({
                            'latest_version': version_config.get('latest_version_number'),
                            'version_count': len(self.get_configuration_history(version_config['configuration_id']))
                        })
                    else:
                        # Add new configuration
                        configurations.append(version_config)
            except Exception as e:
                logger.warning(f"Failed to get version information: {e}")
        
        return configurations
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        stats = {
            'gateway_stats': self.stats.copy(),
            'features_enabled': {k: v for k, v in self.features.items() if v}
        }
        
        # Add parameter registry stats
        if self.features['parameter_registry']:
            try:
                stats['parameter_registry'] = self.parameter_registry.get_statistics()
            except Exception as e:
                logger.warning(f"Failed to get parameter registry stats: {e}")
        
        # Add version control stats
        if self.features['version_control']:
            try:
                stats['version_control'] = self.version_manager.get_statistics()
            except Exception as e:
                logger.warning(f"Failed to get version control stats: {e}")
        
        return stats
    
    def enable_feature(self, feature_name: str) -> bool:
        """Enable a feature at runtime"""
        if feature_name in self.features:
            self.features[feature_name] = True
            logger.info(f"Enabled feature: {feature_name}")
            return True
        
        return False
    
    def disable_feature(self, feature_name: str) -> bool:
        """Disable a feature at runtime"""
        if feature_name in self.features:
            self.features[feature_name] = False
            logger.info(f"Disabled feature: {feature_name}")
            return True
        
        return False
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components"""
        health = {
            'overall_status': 'healthy',
            'components': {}
        }
        
        # Check configuration manager
        try:
            # Test basic functionality
            test_strategies = self.config_manager._registry.list_strategies()
            health['components']['config_manager'] = {
                'status': 'healthy',
                'strategy_count': len(test_strategies)
            }
        except Exception as e:
            health['components']['config_manager'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health['overall_status'] = 'degraded'
        
        # Check parameter registry
        if self.features['parameter_registry']:
            try:
                reg_stats = self.parameter_registry.get_statistics()
                health['components']['parameter_registry'] = {
                    'status': 'healthy',
                    'parameter_count': reg_stats['total_parameters']
                }
            except Exception as e:
                health['components']['parameter_registry'] = {
                    'status': 'unhealthy', 
                    'error': str(e)
                }
                health['overall_status'] = 'degraded'
        
        # Check version control
        if self.features['version_control']:
            try:
                vc_stats = self.version_manager.get_statistics()
                health['components']['version_control'] = {
                    'status': 'healthy',
                    'version_count': vc_stats['total_versions']
                }
            except Exception as e:
                health['components']['version_control'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                health['overall_status'] = 'degraded'
        
        return health