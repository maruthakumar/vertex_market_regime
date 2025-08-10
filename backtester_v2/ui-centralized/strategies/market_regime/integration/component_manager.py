"""
Component Manager - Dynamic Component Lifecycle Management
=======================================================

Manages the lifecycle of all market regime analysis components.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Type, Union
from datetime import datetime, timedelta
import logging
import importlib
import inspect
from abc import ABC, abstractmethod

# Import base utilities
from ..base.common_utils import ErrorHandler, CacheUtils

logger = logging.getLogger(__name__)


class ComponentInterface(ABC):
    """Base interface for all market regime components"""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize component with configuration"""
        pass
    
    @abstractmethod
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform analysis on provided data"""
        pass
    
    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """Get component health status"""
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get component metadata"""
        pass


class ComponentManager:
    """Dynamic component lifecycle management system"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Component Manager"""
        self.config = config
        self.component_registry = {}
        self.component_instances = {}
        self.component_health = {}
        self.component_dependencies = {}
        
        # Component loading configuration
        self.auto_load_components = config.get('auto_load_components', True)
        self.component_timeout = config.get('component_timeout', 30.0)
        self.health_check_interval = config.get('health_check_interval', 300)  # 5 minutes
        
        # Component categories
        self.component_categories = {
            'core_analysis': [
                'straddle_analysis',
                'oi_pa_analysis', 
                'greek_sentiment',
                'market_breadth',
                'iv_analytics',
                'technical_indicators'
            ],
            'optimization': [
                'historical_optimizer',
                'performance_evaluator',
                'weight_validator'
            ],
            'ml_models': [
                'random_forest_optimizer',
                'linear_regression_optimizer',
                'ensemble_optimizer'
            ],
            'utilities': [
                'data_pipeline',
                'result_aggregator',
                'cache_manager'
            ]
        }
        
        # Performance tracking
        self.component_metrics = {
            'load_times': {},
            'execution_times': {},
            'success_rates': {},
            'error_counts': {},
            'last_health_check': {}
        }
        
        # Error handling
        self.error_handler = ErrorHandler()
        self.cache = CacheUtils(max_size=50)
        
        # Initialize components if auto-load is enabled
        if self.auto_load_components:
            self._auto_load_components()
        
        logger.info("ComponentManager initialized with dynamic lifecycle management")
    
    def register_component(self,
                         component_name: str,
                         component_class: Type,
                         category: str = 'custom',
                         dependencies: Optional[List[str]] = None,
                         config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Register a component with the manager
        
        Args:
            component_name: Unique name for the component
            component_class: Component class type
            category: Component category
            dependencies: List of required dependencies
            config: Component-specific configuration
            
        Returns:
            bool: True if registration successful
        """
        try:
            # Validate component class
            if not self._validate_component_class(component_class):
                logger.error(f"Component class {component_class} does not implement required interface")
                return False
            
            # Register component
            self.component_registry[component_name] = {
                'class': component_class,
                'category': category,
                'dependencies': dependencies or [],
                'config': config or {},
                'registered_at': datetime.now(),
                'status': 'registered'
            }
            
            # Initialize component metrics
            self.component_metrics['load_times'][component_name] = []
            self.component_metrics['execution_times'][component_name] = []
            self.component_metrics['success_rates'][component_name] = 1.0
            self.component_metrics['error_counts'][component_name] = 0
            
            logger.info(f"Component {component_name} registered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error registering component {component_name}: {e}")
            return False
    
    def load_component(self,
                      component_name: str,
                      force_reload: bool = False) -> bool:
        """
        Load and initialize a component
        
        Args:
            component_name: Name of component to load
            force_reload: Force reload even if already loaded
            
        Returns:
            bool: True if loading successful
        """
        try:
            start_time = datetime.now()
            
            # Check if component is already loaded
            if component_name in self.component_instances and not force_reload:
                logger.info(f"Component {component_name} already loaded")
                return True
            
            # Check if component is registered
            if component_name not in self.component_registry:
                logger.error(f"Component {component_name} not registered")
                return False
            
            # Check dependencies
            if not self._check_dependencies(component_name):
                logger.error(f"Dependencies not met for component {component_name}")
                return False
            
            # Get component info
            component_info = self.component_registry[component_name]
            component_class = component_info['class']
            component_config = component_info['config']
            
            # Merge with global config
            merged_config = self._merge_configs(component_config, self.config.get(f'{component_name}_config', {}))
            
            # Instantiate component
            component_instance = component_class(merged_config)
            
            # Initialize component
            if hasattr(component_instance, 'initialize'):
                init_success = component_instance.initialize(merged_config)
                if not init_success:
                    logger.error(f"Component {component_name} initialization failed")
                    return False
            
            # Store instance
            self.component_instances[component_name] = component_instance
            
            # Update registry status
            self.component_registry[component_name]['status'] = 'loaded'
            self.component_registry[component_name]['loaded_at'] = datetime.now()
            
            # Record load time
            load_time = (datetime.now() - start_time).total_seconds()
            self.component_metrics['load_times'][component_name].append(load_time)
            
            # Initial health check
            self._perform_health_check(component_name)
            
            logger.info(f"Component {component_name} loaded successfully in {load_time:.3f}s")
            return True
            
        except Exception as e:
            logger.error(f"Error loading component {component_name}: {e}")
            self.component_metrics['error_counts'][component_name] += 1
            return False
    
    def unload_component(self, component_name: str) -> bool:
        """
        Unload a component
        
        Args:
            component_name: Name of component to unload
            
        Returns:
            bool: True if unloading successful
        """
        try:
            if component_name not in self.component_instances:
                logger.warning(f"Component {component_name} not loaded")
                return True
            
            # Check dependents
            dependents = self._get_dependents(component_name)
            if dependents:
                logger.warning(f"Component {component_name} has dependents: {dependents}")
                # Optionally unload dependents first
                for dependent in dependents:
                    self.unload_component(dependent)
            
            # Get component instance
            component_instance = self.component_instances[component_name]
            
            # Call cleanup if available
            if hasattr(component_instance, 'cleanup'):
                component_instance.cleanup()
            
            # Remove from instances
            del self.component_instances[component_name]
            
            # Update registry status
            if component_name in self.component_registry:
                self.component_registry[component_name]['status'] = 'unloaded'
                self.component_registry[component_name]['unloaded_at'] = datetime.now()
            
            # Remove from health tracking
            if component_name in self.component_health:
                del self.component_health[component_name]
            
            logger.info(f"Component {component_name} unloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading component {component_name}: {e}")
            return False
    
    def execute_component(self,
                         component_name: str,
                         data: Dict[str, Any],
                         timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Execute analysis on a component
        
        Args:
            component_name: Name of component to execute
            data: Input data for analysis
            timeout: Execution timeout in seconds
            
        Returns:
            Dict with analysis results
        """
        try:
            start_time = datetime.now()
            
            # Check if component is loaded
            if component_name not in self.component_instances:
                if not self.load_component(component_name):
                    return self._get_default_execution_result(component_name, 'component_not_loaded')
            
            # Get component instance
            component_instance = self.component_instances[component_name]
            
            # Check component health
            if not self._is_component_healthy(component_name):
                logger.warning(f"Component {component_name} health check failed")
                return self._get_default_execution_result(component_name, 'unhealthy_component')
            
            # Execute analysis
            execution_timeout = timeout or self.component_timeout
            
            try:
                # Use timeout wrapper for execution
                result = self.error_handler.retry_on_failure(
                    lambda: component_instance.analyze(data),
                    max_retries=2,
                    delay=1.0
                )
                
                # Validate result
                if not self._validate_execution_result(result):
                    logger.warning(f"Invalid result from component {component_name}")
                    result = self._get_default_execution_result(component_name, 'invalid_result')
                
            except Exception as e:
                logger.error(f"Execution error in component {component_name}: {e}")
                result = self._get_default_execution_result(component_name, 'execution_error')
                self.component_metrics['error_counts'][component_name] += 1
            
            # Record execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            self.component_metrics['execution_times'][component_name].append(execution_time)
            
            # Update success rate
            self._update_success_rate(component_name, result.get('status') != 'error')
            
            # Add execution metadata
            result['execution_metadata'] = {
                'component_name': component_name,
                'execution_time': execution_time,
                'timestamp': datetime.now(),
                'manager_version': '2.0.0'
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing component {component_name}: {e}")
            self.component_metrics['error_counts'][component_name] += 1
            return self._get_default_execution_result(component_name, 'manager_error')
    
    def get_component_status(self, component_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get status of component(s)
        
        Args:
            component_name: Specific component name, or None for all components
            
        Returns:
            Dict with component status information
        """
        try:
            if component_name:
                return self._get_single_component_status(component_name)
            else:
                return self._get_all_components_status()
                
        except Exception as e:
            logger.error(f"Error getting component status: {e}")
            return {'error': str(e)}
    
    def perform_health_checks(self) -> Dict[str, Any]:
        """
        Perform health checks on all loaded components
        
        Returns:
            Dict with health check results
        """
        try:
            health_results = {
                'overall_health': 'healthy',
                'component_health': {},
                'unhealthy_components': [],
                'health_check_timestamp': datetime.now()
            }
            
            unhealthy_count = 0
            
            for component_name in self.component_instances.keys():
                try:
                    health_status = self._perform_health_check(component_name)
                    health_results['component_health'][component_name] = health_status
                    
                    if health_status.get('status') != 'healthy':
                        unhealthy_count += 1
                        health_results['unhealthy_components'].append(component_name)
                        
                except Exception as e:
                    logger.error(f"Health check failed for component {component_name}: {e}")
                    health_results['component_health'][component_name] = {
                        'status': 'error',
                        'error': str(e)
                    }
                    unhealthy_count += 1
                    health_results['unhealthy_components'].append(component_name)
            
            # Determine overall health
            total_components = len(self.component_instances)
            if total_components == 0:
                health_results['overall_health'] = 'no_components'
            elif unhealthy_count == 0:
                health_results['overall_health'] = 'healthy'
            elif unhealthy_count / total_components < 0.3:
                health_results['overall_health'] = 'degraded'
            else:
                health_results['overall_health'] = 'unhealthy'
            
            return health_results
            
        except Exception as e:
            logger.error(f"Error performing health checks: {e}")
            return {'overall_health': 'error', 'error': str(e)}
    
    def get_component_metrics(self) -> Dict[str, Any]:
        """Get comprehensive component metrics"""
        try:
            metrics = {
                'component_counts': {
                    'registered': len(self.component_registry),
                    'loaded': len(self.component_instances),
                    'categories': {}
                },
                'performance_metrics': {},
                'health_summary': {},
                'resource_usage': {}
            }
            
            # Category counts
            for category, components in self.component_categories.items():
                loaded_in_category = sum(1 for comp in components if comp in self.component_instances)
                metrics['component_counts']['categories'][category] = {
                    'total': len(components),
                    'loaded': loaded_in_category
                }
            
            # Performance metrics
            for component_name in self.component_instances.keys():
                load_times = self.component_metrics['load_times'].get(component_name, [])
                exec_times = self.component_metrics['execution_times'].get(component_name, [])
                
                metrics['performance_metrics'][component_name] = {
                    'avg_load_time': float(np.mean(load_times)) if load_times else 0.0,
                    'avg_execution_time': float(np.mean(exec_times)) if exec_times else 0.0,
                    'success_rate': float(self.component_metrics['success_rates'].get(component_name, 1.0)),
                    'error_count': self.component_metrics['error_counts'].get(component_name, 0),
                    'execution_count': len(exec_times)
                }
            
            # Health summary
            healthy_components = sum(1 for health in self.component_health.values() 
                                   if health.get('status') == 'healthy')
            metrics['health_summary'] = {
                'healthy_components': healthy_components,
                'total_components': len(self.component_instances),
                'health_percentage': (healthy_components / len(self.component_instances) * 100) if self.component_instances else 0
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting component metrics: {e}")
            return {'error': str(e)}
    
    def optimize_component_configuration(self, component_name: str) -> Dict[str, Any]:
        """
        Optimize component configuration based on performance metrics
        
        Args:
            component_name: Name of component to optimize
            
        Returns:
            Dict with optimization recommendations
        """
        try:
            if component_name not in self.component_instances:
                return {'error': 'Component not loaded'}
            
            optimization = {
                'component_name': component_name,
                'current_performance': {},
                'recommendations': [],
                'estimated_improvements': {}
            }
            
            # Analyze current performance
            exec_times = self.component_metrics['execution_times'].get(component_name, [])
            success_rate = self.component_metrics['success_rates'].get(component_name, 1.0)
            error_count = self.component_metrics['error_counts'].get(component_name, 0)
            
            optimization['current_performance'] = {
                'avg_execution_time': float(np.mean(exec_times)) if exec_times else 0.0,
                'success_rate': float(success_rate),
                'error_count': error_count
            }
            
            # Generate recommendations
            if exec_times and np.mean(exec_times) > 5.0:
                optimization['recommendations'].append("Consider enabling caching for this component")
                optimization['estimated_improvements']['caching'] = 'Reduce execution time by 30-50%'
            
            if success_rate < 0.9:
                optimization['recommendations'].append("Review component configuration and error handling")
                optimization['estimated_improvements']['error_handling'] = 'Improve success rate to >95%'
            
            if error_count > 10:
                optimization['recommendations'].append("Component may need debugging or reconfiguration")
                optimization['estimated_improvements']['debugging'] = 'Reduce error frequency significantly'
            
            # Configuration-specific recommendations
            component_instance = self.component_instances[component_name]
            if hasattr(component_instance, 'get_optimization_recommendations'):
                component_recommendations = component_instance.get_optimization_recommendations()
                optimization['recommendations'].extend(component_recommendations)
            
            return optimization
            
        except Exception as e:
            logger.error(f"Error optimizing component configuration: {e}")
            return {'error': str(e)}
    
    def _auto_load_components(self):
        """Automatically load core components"""
        try:
            # Load core analysis components
            core_components = self.component_categories.get('core_analysis', [])
            
            for component_name in core_components:
                try:
                    # Dynamic import and registration
                    self._register_core_component(component_name)
                    self.load_component(component_name)
                except Exception as e:
                    logger.warning(f"Failed to auto-load component {component_name}: {e}")
            
        except Exception as e:
            logger.error(f"Error in auto-loading components: {e}")
    
    def _register_core_component(self, component_name: str):
        """Register a core component dynamically"""
        try:
            # Component mapping
            component_mappings = {
                'straddle_analysis': ('indicators.straddle_analysis.straddle_engine', 'StraddleAnalysisEngine'),
                'oi_pa_analysis': ('indicators.oi_pa_analysis.oi_pa_analyzer', 'OIPAAnalyzer'),
                'greek_sentiment': ('indicators.greek_sentiment.greek_sentiment_analyzer', 'GreekSentimentAnalyzer'),
                'market_breadth': ('indicators.market_breadth.market_breadth_analyzer', 'MarketBreadthAnalyzer'),
                'iv_analytics': ('indicators.iv_analytics.iv_analytics_analyzer', 'IVAnalyticsAnalyzer'),
                'technical_indicators': ('indicators.technical_indicators.technical_indicators_analyzer', 'TechnicalIndicatorsAnalyzer')
            }
            
            if component_name in component_mappings:
                module_path, class_name = component_mappings[component_name]
                
                # Dynamic import
                module = importlib.import_module(f"..{module_path}", __name__)
                component_class = getattr(module, class_name)
                
                # Register component
                self.register_component(
                    component_name,
                    component_class,
                    'core_analysis',
                    config=self.config.get(f'{component_name}_config', {})
                )
                
        except Exception as e:
            logger.error(f"Error registering core component {component_name}: {e}")
    
    def _validate_component_class(self, component_class: Type) -> bool:
        """Validate that component class implements required interface"""
        try:
            # Check if class has required methods
            required_methods = ['analyze']
            
            for method_name in required_methods:
                if not hasattr(component_class, method_name):
                    return False
                
                method = getattr(component_class, method_name)
                if not callable(method):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating component class: {e}")
            return False
    
    def _check_dependencies(self, component_name: str) -> bool:
        """Check if component dependencies are satisfied"""
        try:
            if component_name not in self.component_registry:
                return False
            
            dependencies = self.component_registry[component_name].get('dependencies', [])
            
            for dependency in dependencies:
                if dependency not in self.component_instances:
                    # Try to load dependency
                    if not self.load_component(dependency):
                        logger.error(f"Failed to load dependency {dependency} for {component_name}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking dependencies for {component_name}: {e}")
            return False
    
    def _get_dependents(self, component_name: str) -> List[str]:
        """Get list of components that depend on the given component"""
        try:
            dependents = []
            
            for comp_name, comp_info in self.component_registry.items():
                dependencies = comp_info.get('dependencies', [])
                if component_name in dependencies:
                    dependents.append(comp_name)
            
            return dependents
            
        except Exception as e:
            logger.error(f"Error getting dependents for {component_name}: {e}")
            return []
    
    def _merge_configs(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configuration dictionaries"""
        try:
            merged = config1.copy()
            
            for key, value in config2.items():
                if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                    merged[key] = self._merge_configs(merged[key], value)
                else:
                    merged[key] = value
            
            return merged
            
        except Exception as e:
            logger.error(f"Error merging configs: {e}")
            return config1
    
    def _perform_health_check(self, component_name: str) -> Dict[str, Any]:
        """Perform health check on a specific component"""
        try:
            if component_name not in self.component_instances:
                return {'status': 'not_loaded', 'timestamp': datetime.now()}
            
            component_instance = self.component_instances[component_name]
            
            # Basic health check
            health_status = {
                'status': 'healthy',
                'timestamp': datetime.now(),
                'checks': {}
            }
            
            # Check if component has health method
            if hasattr(component_instance, 'get_health_status'):
                component_health = component_instance.get_health_status()
                health_status.update(component_health)
            else:
                # Basic checks
                health_status['checks']['instance_exists'] = True
                health_status['checks']['has_analyze_method'] = hasattr(component_instance, 'analyze')
            
            # Performance-based health indicators
            success_rate = self.component_metrics['success_rates'].get(component_name, 1.0)
            error_count = self.component_metrics['error_counts'].get(component_name, 0)
            
            health_status['checks']['success_rate'] = success_rate
            health_status['checks']['error_count'] = error_count
            
            # Determine overall status
            if success_rate < 0.7 or error_count > 20:
                health_status['status'] = 'unhealthy'
            elif success_rate < 0.9 or error_count > 5:
                health_status['status'] = 'degraded'
            
            # Store health status
            self.component_health[component_name] = health_status
            self.component_metrics['last_health_check'][component_name] = datetime.now()
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error performing health check for {component_name}: {e}")
            error_status = {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now()
            }
            self.component_health[component_name] = error_status
            return error_status
    
    def _is_component_healthy(self, component_name: str) -> bool:
        """Check if component is healthy"""
        try:
            # Check if recent health check exists
            last_check = self.component_metrics['last_health_check'].get(component_name)
            
            if not last_check or (datetime.now() - last_check).total_seconds() > self.health_check_interval:
                # Perform new health check
                self._perform_health_check(component_name)
            
            health_status = self.component_health.get(component_name, {})
            return health_status.get('status') in ['healthy', 'degraded']
            
        except Exception as e:
            logger.error(f"Error checking component health for {component_name}: {e}")
            return False
    
    def _validate_execution_result(self, result: Dict[str, Any]) -> bool:
        """Validate execution result format"""
        try:
            if not isinstance(result, dict):
                return False
            
            # Check for required fields (basic validation)
            if 'status' in result and result['status'] == 'error':
                return True  # Error results are valid
            
            # Should have some analysis output
            return len(result) > 0
            
        except Exception as e:
            logger.error(f"Error validating execution result: {e}")
            return False
    
    def _update_success_rate(self, component_name: str, success: bool):
        """Update component success rate"""
        try:
            current_rate = self.component_metrics['success_rates'].get(component_name, 1.0)
            exec_times = self.component_metrics['execution_times'].get(component_name, [])
            
            # Calculate new success rate using exponential moving average
            alpha = 0.1  # Smoothing factor
            new_success = 1.0 if success else 0.0
            
            if len(exec_times) == 1:  # First execution
                self.component_metrics['success_rates'][component_name] = new_success
            else:
                self.component_metrics['success_rates'][component_name] = (
                    alpha * new_success + (1 - alpha) * current_rate
                )
                
        except Exception as e:
            logger.error(f"Error updating success rate for {component_name}: {e}")
    
    def _get_single_component_status(self, component_name: str) -> Dict[str, Any]:
        """Get status for a single component"""
        try:
            status = {
                'component_name': component_name,
                'registration_status': 'not_registered',
                'load_status': 'not_loaded',
                'health_status': {},
                'performance_metrics': {},
                'metadata': {}
            }
            
            # Registration status
            if component_name in self.component_registry:
                status['registration_status'] = 'registered'
                reg_info = self.component_registry[component_name]
                status['registration_info'] = {
                    'category': reg_info.get('category'),
                    'dependencies': reg_info.get('dependencies', []),
                    'registered_at': reg_info.get('registered_at'),
                    'status': reg_info.get('status')
                }
            
            # Load status
            if component_name in self.component_instances:
                status['load_status'] = 'loaded'
                
                # Get metadata
                component_instance = self.component_instances[component_name]
                if hasattr(component_instance, 'get_metadata'):
                    status['metadata'] = component_instance.get_metadata()
            
            # Health status
            if component_name in self.component_health:
                status['health_status'] = self.component_health[component_name]
            
            # Performance metrics
            status['performance_metrics'] = {
                'avg_load_time': float(np.mean(self.component_metrics['load_times'].get(component_name, [0]))),
                'avg_execution_time': float(np.mean(self.component_metrics['execution_times'].get(component_name, [0]))),
                'success_rate': float(self.component_metrics['success_rates'].get(component_name, 1.0)),
                'error_count': self.component_metrics['error_counts'].get(component_name, 0)
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting status for component {component_name}: {e}")
            return {'component_name': component_name, 'error': str(e)}
    
    def _get_all_components_status(self) -> Dict[str, Any]:
        """Get status for all components"""
        try:
            all_status = {
                'manager_status': 'operational',
                'total_registered': len(self.component_registry),
                'total_loaded': len(self.component_instances),
                'components': {}
            }
            
            # Get status for all registered components
            for component_name in self.component_registry.keys():
                all_status['components'][component_name] = self._get_single_component_status(component_name)
            
            # Add summary statistics
            healthy_count = sum(1 for comp_status in all_status['components'].values()
                              if comp_status.get('health_status', {}).get('status') == 'healthy')
            
            all_status['health_summary'] = {
                'healthy_components': healthy_count,
                'total_components': len(self.component_instances),
                'health_percentage': (healthy_count / len(self.component_instances) * 100) if self.component_instances else 0
            }
            
            return all_status
            
        except Exception as e:
            logger.error(f"Error getting all components status: {e}")
            return {'manager_status': 'error', 'error': str(e)}
    
    def _get_default_execution_result(self, component_name: str, error_type: str) -> Dict[str, Any]:
        """Get default execution result for failed executions"""
        return {
            'status': 'error',
            'error_type': error_type,
            'component_name': component_name,
            'timestamp': datetime.now(),
            'message': f"Component {component_name} execution failed: {error_type}",
            'composite_score': 0.0,
            'confidence': 0.0,
            'regime_signals': [],
            'insights': [f"Component {component_name} unavailable"],
            'risk_indicators': ['Component failure risk']
        }