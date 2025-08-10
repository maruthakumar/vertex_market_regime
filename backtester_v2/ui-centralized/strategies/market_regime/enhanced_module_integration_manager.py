#!/usr/bin/env python3
"""
Enhanced Module Integration Manager
==================================

This module provides centralized integration management for enhanced market regime modules,
ensuring proper configuration parameter injection, module lifecycle management, and
systematic integration with the comprehensive market regime system.

Features:
- Module discovery and registration system
- Configuration parameter injection framework
- Module lifecycle management (init, configure, execute, cleanup)
- Dependency resolution and loading order optimization
- Integration health monitoring and error handling
- Performance tracking and optimization

Author: The Augster
Date: 2025-01-06
Version: 1.0.0 - Enhanced Module Integration Manager
"""

import logging
import importlib
import inspect
from typing import Dict, List, Any, Optional, Type, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import time
from datetime import datetime
import threading
from collections import defaultdict, deque
import traceback

logger = logging.getLogger(__name__)

class ModuleStatus(Enum):
    """Module status enumeration"""
    UNREGISTERED = "unregistered"
    REGISTERED = "registered"
    CONFIGURED = "configured"
    INITIALIZED = "initialized"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"

class IntegrationPriority(Enum):
    """Integration priority levels"""
    CRITICAL = 1    # Core functionality modules
    HIGH = 2        # Important analysis modules
    MEDIUM = 3      # Supporting modules
    LOW = 4         # Optional enhancement modules

@dataclass
class ModuleRegistration:
    """Module registration information"""
    name: str
    module_class: Type
    config_requirements: List[str]
    dependencies: List[str]
    priority: IntegrationPriority
    status: ModuleStatus = ModuleStatus.UNREGISTERED
    instance: Optional[Any] = None
    last_error: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    configuration: Dict[str, Any] = field(default_factory=dict)

@dataclass
class IntegrationMetrics:
    """Integration performance and health metrics"""
    total_modules: int = 0
    active_modules: int = 0
    error_modules: int = 0
    average_init_time: float = 0.0
    average_execution_time: float = 0.0
    total_integrations: int = 0
    successful_integrations: int = 0
    last_health_check: Optional[datetime] = None

class EnhancedModuleIntegrationManager:
    """
    Centralized integration manager for enhanced market regime modules
    
    Manages the complete lifecycle of enhanced modules including discovery,
    registration, configuration, initialization, and execution coordination.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Enhanced Module Integration Manager"""
        self.config = config or {}
        
        # Module registry
        self.registered_modules: Dict[str, ModuleRegistration] = {}
        self.module_dependencies: Dict[str, List[str]] = {}
        self.loading_order: List[str] = []
        
        # Integration state
        self.integration_active = False
        self.integration_lock = threading.RLock()
        self.metrics = IntegrationMetrics()
        
        # Performance tracking
        self.execution_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.error_history: deque = deque(maxlen=50)
        
        # Configuration management
        self.global_config: Dict[str, Any] = {}
        self.config_change_callbacks: List[Callable] = []
        
        logger.info("Enhanced Module Integration Manager initialized")
    
    def register_enhanced_module(self, 
                                name: str,
                                module_class: Type,
                                config_requirements: List[str] = None,
                                dependencies: List[str] = None,
                                priority: IntegrationPriority = IntegrationPriority.MEDIUM) -> bool:
        """
        Register an enhanced module for integration
        
        Args:
            name: Unique module name
            module_class: Module class to instantiate
            config_requirements: List of required configuration parameters
            dependencies: List of module dependencies
            priority: Integration priority level
            
        Returns:
            bool: True if registration successful
        """
        try:
            with self.integration_lock:
                if name in self.registered_modules:
                    logger.warning(f"Module {name} already registered, updating registration")
                
                # Validate module class
                if not self._validate_module_class(module_class):
                    logger.error(f"Module class {module_class} failed validation")
                    return False
                
                # Create registration
                registration = ModuleRegistration(
                    name=name,
                    module_class=module_class,
                    config_requirements=config_requirements or [],
                    dependencies=dependencies or [],
                    priority=priority,
                    status=ModuleStatus.REGISTERED
                )
                
                self.registered_modules[name] = registration
                self.module_dependencies[name] = dependencies or []
                
                # Update loading order
                self._update_loading_order()
                
                logger.info(f"Successfully registered enhanced module: {name}")
                return True
                
        except Exception as e:
            logger.error(f"Error registering module {name}: {e}")
            return False
    
    def configure_modules(self, global_config: Dict[str, Any]) -> bool:
        """
        Configure all registered modules with global configuration
        
        Args:
            global_config: Global configuration dictionary
            
        Returns:
            bool: True if all modules configured successfully
        """
        try:
            with self.integration_lock:
                self.global_config = global_config.copy()
                success_count = 0
                
                for name, registration in self.registered_modules.items():
                    if registration.status == ModuleStatus.ERROR:
                        continue
                    
                    # Extract module-specific configuration
                    module_config = self._extract_module_config(name, global_config)
                    
                    # Validate required configuration
                    if not self._validate_module_config(registration, module_config):
                        registration.status = ModuleStatus.ERROR
                        registration.last_error = "Configuration validation failed"
                        continue
                    
                    # Store configuration
                    registration.configuration = module_config
                    registration.status = ModuleStatus.CONFIGURED
                    success_count += 1
                    
                    logger.debug(f"Configured module: {name}")
                
                logger.info(f"Configured {success_count}/{len(self.registered_modules)} modules")
                return success_count == len(self.registered_modules)
                
        except Exception as e:
            logger.error(f"Error configuring modules: {e}")
            return False
    
    def initialize_modules(self) -> bool:
        """
        Initialize all configured modules in dependency order
        
        Returns:
            bool: True if all modules initialized successfully
        """
        try:
            with self.integration_lock:
                success_count = 0
                
                # Initialize in dependency order
                for name in self.loading_order:
                    registration = self.registered_modules.get(name)
                    if not registration or registration.status != ModuleStatus.CONFIGURED:
                        continue
                    
                    # Initialize module
                    start_time = time.time()
                    if self._initialize_single_module(registration):
                        init_time = time.time() - start_time
                        registration.performance_metrics['init_time'] = init_time
                        registration.status = ModuleStatus.INITIALIZED
                        success_count += 1
                        logger.debug(f"Initialized module: {name} ({init_time:.3f}s)")
                    else:
                        registration.status = ModuleStatus.ERROR
                        logger.error(f"Failed to initialize module: {name}")
                
                # Update metrics
                self.metrics.total_modules = len(self.registered_modules)
                self.metrics.active_modules = success_count
                self.metrics.error_modules = len(self.registered_modules) - success_count
                
                if success_count > 0:
                    avg_init_time = sum(r.performance_metrics.get('init_time', 0) 
                                      for r in self.registered_modules.values()) / success_count
                    self.metrics.average_init_time = avg_init_time
                
                logger.info(f"Initialized {success_count}/{len(self.registered_modules)} modules")
                return success_count == len(self.registered_modules)
                
        except Exception as e:
            logger.error(f"Error initializing modules: {e}")
            return False
    
    def activate_integration(self) -> bool:
        """
        Activate the integration system
        
        Returns:
            bool: True if integration activated successfully
        """
        try:
            with self.integration_lock:
                if self.integration_active:
                    logger.warning("Integration already active")
                    return True
                
                # Verify all modules are initialized
                initialized_count = sum(1 for r in self.registered_modules.values() 
                                      if r.status == ModuleStatus.INITIALIZED)
                
                if initialized_count == 0:
                    logger.error("No modules initialized, cannot activate integration")
                    return False
                
                # Activate modules
                for registration in self.registered_modules.values():
                    if registration.status == ModuleStatus.INITIALIZED:
                        registration.status = ModuleStatus.ACTIVE
                
                self.integration_active = True
                self.metrics.last_health_check = datetime.now()
                
                logger.info(f"Integration activated with {initialized_count} active modules")
                return True
                
        except Exception as e:
            logger.error(f"Error activating integration: {e}")
            return False
    
    def execute_module_analysis(self, module_name: str, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Execute analysis for a specific module
        
        Args:
            module_name: Name of module to execute
            market_data: Market data for analysis
            
        Returns:
            Optional[Dict]: Analysis results or None if error
        """
        try:
            registration = self.registered_modules.get(module_name)
            if not registration or registration.status != ModuleStatus.ACTIVE:
                logger.warning(f"Module {module_name} not active for execution")
                return None
            
            # Execute module analysis
            start_time = time.time()
            
            # Call appropriate analysis method based on module type
            result = self._execute_module_method(registration, market_data)
            
            execution_time = time.time() - start_time
            
            # Track performance
            self.execution_history[module_name].append(execution_time)
            registration.performance_metrics['last_execution_time'] = execution_time
            registration.performance_metrics['total_executions'] = \
                registration.performance_metrics.get('total_executions', 0) + 1
            
            return result
            
        except Exception as e:
            error_msg = f"Error executing module {module_name}: {e}"
            logger.error(error_msg)
            self.error_history.append({
                'timestamp': datetime.now(),
                'module': module_name,
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            return None
    
    def get_integration_status(self) -> Dict[str, Any]:
        """
        Get comprehensive integration status
        
        Returns:
            Dict: Integration status and metrics
        """
        with self.integration_lock:
            status = {
                'integration_active': self.integration_active,
                'metrics': {
                    'total_modules': self.metrics.total_modules,
                    'active_modules': self.metrics.active_modules,
                    'error_modules': self.metrics.error_modules,
                    'average_init_time': self.metrics.average_init_time,
                    'average_execution_time': self.metrics.average_execution_time,
                    'last_health_check': self.metrics.last_health_check
                },
                'modules': {}
            }
            
            for name, registration in self.registered_modules.items():
                status['modules'][name] = {
                    'status': registration.status.value,
                    'priority': registration.priority.value,
                    'last_error': registration.last_error,
                    'performance_metrics': registration.performance_metrics.copy(),
                    'dependencies': registration.dependencies.copy()
                }
            
            return status

    def _validate_module_class(self, module_class: Type) -> bool:
        """Validate that module class meets integration requirements"""
        try:
            # Check if class is instantiable
            if not inspect.isclass(module_class):
                return False

            # Check for required methods (at least one analysis method)
            required_methods = ['analyze', 'process', 'calculate', 'detect']
            has_analysis_method = any(hasattr(module_class, method) for method in required_methods)

            if not has_analysis_method:
                logger.warning(f"Module class {module_class} lacks analysis methods")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating module class: {e}")
            return False

    def _extract_module_config(self, module_name: str, global_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract module-specific configuration from global config"""
        module_config = {}

        # Look for module-specific config section
        if module_name in global_config:
            module_config.update(global_config[module_name])

        # Look for config by module type
        module_type = module_name.lower().replace('_', '')
        for key, value in global_config.items():
            if module_type in key.lower().replace('_', ''):
                if isinstance(value, dict):
                    module_config.update(value)
                else:
                    module_config[key] = value

        # Add global parameters that all modules might need
        global_params = ['symbol', 'timeframe', 'lookback_days', 'confidence_threshold']
        for param in global_params:
            if param in global_config:
                module_config[param] = global_config[param]

        return module_config

    def _validate_module_config(self, registration: ModuleRegistration, config: Dict[str, Any]) -> bool:
        """Validate module configuration against requirements"""
        try:
            # Check required configuration parameters
            for required_param in registration.config_requirements:
                if required_param not in config:
                    logger.error(f"Module {registration.name} missing required config: {required_param}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating config for {registration.name}: {e}")
            return False

    def _update_loading_order(self):
        """Update module loading order based on dependencies"""
        try:
            # Topological sort for dependency resolution
            visited = set()
            temp_visited = set()
            loading_order = []

            def visit(module_name: str):
                if module_name in temp_visited:
                    raise ValueError(f"Circular dependency detected involving {module_name}")
                if module_name in visited:
                    return

                temp_visited.add(module_name)

                # Visit dependencies first
                for dependency in self.module_dependencies.get(module_name, []):
                    if dependency in self.registered_modules:
                        visit(dependency)

                temp_visited.remove(module_name)
                visited.add(module_name)
                loading_order.append(module_name)

            # Sort by priority first, then resolve dependencies
            modules_by_priority = sorted(
                self.registered_modules.keys(),
                key=lambda name: self.registered_modules[name].priority.value
            )

            for module_name in modules_by_priority:
                if module_name not in visited:
                    visit(module_name)

            self.loading_order = loading_order
            logger.debug(f"Updated loading order: {self.loading_order}")

        except Exception as e:
            logger.error(f"Error updating loading order: {e}")
            # Fallback to priority-based ordering
            self.loading_order = sorted(
                self.registered_modules.keys(),
                key=lambda name: self.registered_modules[name].priority.value
            )

    def _initialize_single_module(self, registration: ModuleRegistration) -> bool:
        """Initialize a single module instance"""
        try:
            # Create module instance
            module_instance = registration.module_class(registration.configuration)

            # Verify instance has required methods
            if not self._verify_module_instance(module_instance):
                logger.error(f"Module instance {registration.name} failed verification")
                return False

            registration.instance = module_instance
            return True

        except Exception as e:
            registration.last_error = str(e)
            logger.error(f"Error initializing module {registration.name}: {e}")
            return False

    def _verify_module_instance(self, instance: Any) -> bool:
        """Verify module instance has required interface"""
        try:
            # Check for at least one analysis method
            analysis_methods = ['analyze', 'process', 'calculate', 'detect']
            has_method = any(hasattr(instance, method) and callable(getattr(instance, method))
                           for method in analysis_methods)

            return has_method

        except Exception as e:
            logger.error(f"Error verifying module instance: {e}")
            return False

    def _execute_module_method(self, registration: ModuleRegistration, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute appropriate analysis method on module instance"""
        try:
            instance = registration.instance
            if not instance:
                return None

            # Try different analysis method names
            method_names = ['analyze', 'process', 'calculate', 'detect']

            for method_name in method_names:
                if hasattr(instance, method_name):
                    method = getattr(instance, method_name)
                    if callable(method):
                        # Call method with market data
                        result = method(market_data)
                        return result if isinstance(result, dict) else {'result': result}

            logger.warning(f"No suitable analysis method found for module {registration.name}")
            return None

        except Exception as e:
            logger.error(f"Error executing method for module {registration.name}: {e}")
            return None

    def shutdown_integration(self):
        """Shutdown integration system and cleanup resources"""
        try:
            with self.integration_lock:
                logger.info("Shutting down Enhanced Module Integration Manager")

                # Cleanup module instances
                for registration in self.registered_modules.values():
                    if registration.instance and hasattr(registration.instance, 'cleanup'):
                        try:
                            registration.instance.cleanup()
                        except Exception as e:
                            logger.warning(f"Error cleaning up module {registration.name}: {e}")

                    registration.status = ModuleStatus.DISABLED
                    registration.instance = None

                self.integration_active = False
                logger.info("Integration shutdown complete")

        except Exception as e:
            logger.error(f"Error during integration shutdown: {e}")


# Factory function for easy instantiation
def create_integration_manager(config: Optional[Dict[str, Any]] = None) -> EnhancedModuleIntegrationManager:
    """
    Factory function to create Enhanced Module Integration Manager

    Args:
        config: Optional configuration dictionary

    Returns:
        EnhancedModuleIntegrationManager: Configured integration manager instance
    """
    return EnhancedModuleIntegrationManager(config)
