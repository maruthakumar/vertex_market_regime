"""
Algorithm Registry - Auto-Discovery System

Automatically discovers, registers, and manages optimization algorithms
with dynamic loading, validation, and intelligent caching.
"""

import logging
import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Type, Set
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

from ..base.base_optimizer import BaseOptimizer
from .algorithm_metadata import (
    AlgorithmMetadata, AlgorithmMetadataManager, AlgorithmCategory,
    ProblemType, ComplexityLevel, ResourceRequirements, AlgorithmCapabilities
)

logger = logging.getLogger(__name__)

class AlgorithmLoadError(Exception):
    """Raised when algorithm loading fails"""
    pass

class AlgorithmValidationError(Exception):
    """Raised when algorithm validation fails"""
    pass

class AlgorithmRegistry:
    """
    Comprehensive algorithm registry with auto-discovery
    
    Automatically discovers, validates, and manages all optimization algorithms
    with intelligent caching, lazy loading, and performance tracking.
    """
    
    def __init__(self,
                 algorithms_package: str = "strategies.optimization.algorithms",
                 metadata_file: Optional[str] = None,
                 enable_parallel_discovery: bool = True,
                 cache_algorithms: bool = True):
        """
        Initialize algorithm registry
        
        Args:
            algorithms_package: Package containing algorithm implementations
            metadata_file: Path to metadata persistence file
            enable_parallel_discovery: Enable multi-threaded discovery
            cache_algorithms: Cache loaded algorithm instances
        """
        self.algorithms_package = algorithms_package
        self.enable_parallel_discovery = enable_parallel_discovery
        self.cache_algorithms = cache_algorithms
        
        # Registry storage
        self.algorithm_classes: Dict[str, Type[BaseOptimizer]] = {}
        self.algorithm_instances: Dict[str, BaseOptimizer] = {}
        self.algorithm_modules: Dict[str, str] = {}  # algorithm_name -> module_path
        
        # Metadata management
        self.metadata_manager = AlgorithmMetadataManager(metadata_file)
        
        # Discovery state
        self.discovery_completed = False
        self.discovery_lock = threading.Lock()
        self.discovery_errors: List[Tuple[str, Exception]] = []
        
        # Performance tracking
        self.load_times: Dict[str, float] = {}
        self.usage_counts: Dict[str, int] = {}
        
        logger.info("AlgorithmRegistry initialized")
    
    def discover_algorithms(self, force_rediscovery: bool = False) -> Dict[str, Any]:
        """
        Discover all available algorithms
        
        Args:
            force_rediscovery: Force rediscovery even if already completed
            
        Returns:
            Discovery summary with statistics
        """
        if self.discovery_completed and not force_rediscovery:
            logger.info("Algorithm discovery already completed")
            return self._get_discovery_summary()
        
        with self.discovery_lock:
            if self.discovery_completed and not force_rediscovery:
                return self._get_discovery_summary()
            
            start_time = time.time()
            logger.info("Starting algorithm discovery")
            
            # Clear existing registry if rediscovering
            if force_rediscovery:
                self._clear_registry()
            
            try:
                # Get algorithm package path
                package_path = self._get_package_path()
                
                if self.enable_parallel_discovery:
                    discovery_summary = self._parallel_discovery(package_path)
                else:
                    discovery_summary = self._sequential_discovery(package_path)
                
                # Create default metadata for algorithms without it
                self._create_default_metadata()
                
                self.discovery_completed = True
                discovery_time = time.time() - start_time
                
                logger.info(f"Algorithm discovery completed in {discovery_time:.2f}s")
                logger.info(f"Discovered {len(self.algorithm_classes)} algorithms")
                
                discovery_summary.update({
                    'discovery_time': discovery_time,
                    'discovery_completed': True,
                    'total_algorithms': len(self.algorithm_classes)
                })
                
                return discovery_summary
                
            except Exception as e:
                logger.error(f"Error during algorithm discovery: {e}")
                self.discovery_errors.append(("discovery", e))
                raise AlgorithmLoadError(f"Algorithm discovery failed: {e}")
    
    def get_algorithm(self, algorithm_name: str, **kwargs) -> BaseOptimizer:
        """
        Get algorithm instance by name
        
        Args:
            algorithm_name: Name of the algorithm
            **kwargs: Parameters for algorithm initialization
            
        Returns:
            Algorithm instance
        """
        # Ensure discovery has been completed
        if not self.discovery_completed:
            self.discover_algorithms()
        
        if algorithm_name not in self.algorithm_classes:
            available = list(self.algorithm_classes.keys())
            raise ValueError(f"Algorithm '{algorithm_name}' not found. Available: {available}")
        
        # Check cache first
        cache_key = f"{algorithm_name}_{hash(frozenset(kwargs.items()) if kwargs else frozenset())}"
        
        if self.cache_algorithms and cache_key in self.algorithm_instances:
            self.usage_counts[algorithm_name] = self.usage_counts.get(algorithm_name, 0) + 1
            return self.algorithm_instances[cache_key]
        
        # Load algorithm instance
        start_time = time.time()
        
        try:
            algorithm_class = self.algorithm_classes[algorithm_name]
            algorithm_instance = self._create_algorithm_instance(algorithm_class, **kwargs)
            
            # Cache if enabled
            if self.cache_algorithms:
                self.algorithm_instances[cache_key] = algorithm_instance
            
            # Track performance
            load_time = time.time() - start_time
            self.load_times[algorithm_name] = load_time
            self.usage_counts[algorithm_name] = self.usage_counts.get(algorithm_name, 0) + 1
            
            logger.debug(f"Loaded algorithm '{algorithm_name}' in {load_time:.3f}s")
            
            return algorithm_instance
            
        except Exception as e:
            logger.error(f"Error loading algorithm '{algorithm_name}': {e}")
            raise AlgorithmLoadError(f"Failed to load algorithm '{algorithm_name}': {e}")
    
    def list_algorithms(self, 
                       category: Optional[AlgorithmCategory] = None,
                       supports_gpu: Optional[bool] = None,
                       supports_parallel: Optional[bool] = None) -> List[str]:
        """
        List available algorithms with optional filtering
        
        Args:
            category: Filter by algorithm category
            supports_gpu: Filter by GPU support
            supports_parallel: Filter by parallel support
            
        Returns:
            List of algorithm names
        """
        if not self.discovery_completed:
            self.discover_algorithms()
        
        algorithms = list(self.algorithm_classes.keys())
        
        # Apply filters using metadata
        if category is not None:
            algorithms = [
                name for name in algorithms
                if self._get_algorithm_category(name) == category
            ]
        
        if supports_gpu is not None:
            algorithms = [
                name for name in algorithms
                if self._algorithm_supports_gpu(name) == supports_gpu
            ]
        
        if supports_parallel is not None:
            algorithms = [
                name for name in algorithms
                if self._algorithm_supports_parallel(name) == supports_parallel
            ]
        
        return algorithms
    
    def get_algorithm_info(self, algorithm_name: str) -> Dict[str, Any]:
        """Get comprehensive information about an algorithm"""
        if algorithm_name not in self.algorithm_classes:
            raise ValueError(f"Algorithm '{algorithm_name}' not found")
        
        algorithm_class = self.algorithm_classes[algorithm_name]
        metadata = self.metadata_manager.get_algorithm_metadata(algorithm_name)
        
        info = {
            'name': algorithm_name,
            'class_name': algorithm_class.__name__,
            'module': self.algorithm_modules.get(algorithm_name, 'unknown'),
            'docstring': inspect.getdoc(algorithm_class),
            'load_time': self.load_times.get(algorithm_name, 0.0),
            'usage_count': self.usage_counts.get(algorithm_name, 0),
        }
        
        if metadata:
            info.update({
                'category': metadata.category.value,
                'description': metadata.description,
                'version': metadata.version,
                'capabilities': {
                    'problem_types': [pt.value for pt in metadata.capabilities.problem_types],
                    'supports_gpu': metadata.capabilities.supports_gpu,
                    'supports_parallel': metadata.capabilities.supports_parallel,
                    'supports_constraints': metadata.capabilities.supports_constraints
                },
                'resource_requirements': {
                    'memory_mb': metadata.resource_requirements.memory_mb,
                    'complexity': metadata.resource_requirements.complexity.value
                },
                'performance': {
                    'total_runs': metadata.performance_profile.total_runs,
                    'success_rate': metadata.performance_profile.convergence_rate,
                    'average_improvement': metadata.performance_profile.average_improvement
                }
            })
        
        return info
    
    def recommend_algorithms(self,
                           problem_characteristics: Dict[str, Any],
                           resource_constraints: Optional[Dict[str, Any]] = None,
                           max_recommendations: int = 5) -> List[Tuple[str, float]]:
        """
        Recommend algorithms for a specific problem
        
        Args:
            problem_characteristics: Problem-specific requirements
            resource_constraints: Available computational resources
            max_recommendations: Maximum number of recommendations
            
        Returns:
            List of (algorithm_name, score) tuples
        """
        if not self.discovery_completed:
            self.discover_algorithms()
        
        recommendations = self.metadata_manager.recommend_algorithms(
            problem_characteristics, resource_constraints
        )
        
        # Filter to only include discovered algorithms
        available_recommendations = [
            (name, score) for name, score in recommendations
            if name in self.algorithm_classes
        ]
        
        return available_recommendations[:max_recommendations]
    
    def validate_algorithm(self, algorithm_class: Type) -> bool:
        """
        Validate that a class implements the BaseOptimizer interface correctly
        
        Args:
            algorithm_class: Class to validate
            
        Returns:
            True if valid, raises AlgorithmValidationError if not
        """
        # Check inheritance
        if not issubclass(algorithm_class, BaseOptimizer):
            raise AlgorithmValidationError(
                f"{algorithm_class.__name__} does not inherit from BaseOptimizer"
            )
        
        # Check required methods
        required_methods = ['optimize']
        for method_name in required_methods:
            if not hasattr(algorithm_class, method_name):
                raise AlgorithmValidationError(
                    f"{algorithm_class.__name__} missing required method: {method_name}"
                )
            
            method = getattr(algorithm_class, method_name)
            if not callable(method):
                raise AlgorithmValidationError(
                    f"{algorithm_class.__name__}.{method_name} is not callable"
                )
        
        # Check constructor signature
        try:
            sig = inspect.signature(algorithm_class.__init__)
            # Should at least accept param_space and objective_function
            required_params = ['param_space', 'objective_function']
            param_names = list(sig.parameters.keys())[1:]  # Skip 'self'
            
            for required_param in required_params:
                if required_param not in param_names:
                    # Check if there's **kwargs to accept it
                    has_kwargs = any(
                        p.kind == inspect.Parameter.VAR_KEYWORD 
                        for p in sig.parameters.values()
                    )
                    if not has_kwargs:
                        raise AlgorithmValidationError(
                            f"{algorithm_class.__name__} constructor missing required parameter: {required_param}"
                        )
        
        except Exception as e:
            raise AlgorithmValidationError(
                f"Error validating {algorithm_class.__name__} constructor: {e}"
            )
        
        return True
    
    def get_registry_statistics(self) -> Dict[str, Any]:
        """Get comprehensive registry statistics"""
        if not self.discovery_completed:
            self.discover_algorithms()
        
        # Category breakdown
        category_counts = {}
        for category in AlgorithmCategory:
            category_counts[category.value] = len(self.list_algorithms(category=category))
        
        # Performance statistics
        total_usage = sum(self.usage_counts.values())
        avg_load_time = sum(self.load_times.values()) / len(self.load_times) if self.load_times else 0
        
        # Most popular algorithms
        popular_algorithms = sorted(
            self.usage_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        return {
            'discovery_status': {
                'completed': self.discovery_completed,
                'total_algorithms': len(self.algorithm_classes),
                'discovery_errors': len(self.discovery_errors)
            },
            'category_breakdown': category_counts,
            'performance_stats': {
                'total_usage': total_usage,
                'average_load_time': avg_load_time,
                'cached_instances': len(self.algorithm_instances),
                'popular_algorithms': popular_algorithms
            },
            'capability_stats': {
                'gpu_enabled': len(self.list_algorithms(supports_gpu=True)),
                'parallel_enabled': len(self.list_algorithms(supports_parallel=True))
            }
        }
    
    def clear_cache(self):
        """Clear algorithm instance cache"""
        self.algorithm_instances.clear()
        logger.info("Algorithm cache cleared")
    
    def reload_algorithm(self, algorithm_name: str):
        """Reload a specific algorithm"""
        if algorithm_name not in self.algorithm_modules:
            raise ValueError(f"Algorithm '{algorithm_name}' not found")
        
        module_path = self.algorithm_modules[algorithm_name]
        
        try:
            # Remove from cache
            keys_to_remove = [k for k in self.algorithm_instances.keys() if k.startswith(algorithm_name)]
            for key in keys_to_remove:
                del self.algorithm_instances[key]
            
            # Reload module
            if module_path in sys.modules:
                importlib.reload(sys.modules[module_path])
            
            # Re-import algorithm class
            module = importlib.import_module(module_path)
            algorithm_class = self._find_optimizer_class_in_module(module)
            
            if algorithm_class:
                self.validate_algorithm(algorithm_class)
                self.algorithm_classes[algorithm_name] = algorithm_class
                logger.info(f"Reloaded algorithm: {algorithm_name}")
            else:
                raise AlgorithmLoadError(f"No valid optimizer class found in {module_path}")
                
        except Exception as e:
            logger.error(f"Error reloading algorithm '{algorithm_name}': {e}")
            raise AlgorithmLoadError(f"Failed to reload algorithm '{algorithm_name}': {e}")
    
    # Private methods
    
    def _get_package_path(self) -> Path:
        """Get the file system path for the algorithms package"""
        try:
            # Import the package to get its path
            package = importlib.import_module(self.algorithms_package)
            package_path = Path(package.__file__).parent
            return package_path
        except ImportError as e:
            raise AlgorithmLoadError(f"Cannot import algorithms package '{self.algorithms_package}': {e}")
    
    def _parallel_discovery(self, package_path: Path) -> Dict[str, Any]:
        """Discover algorithms using parallel processing"""
        logger.info("Using parallel discovery")
        
        # Find all Python modules in the package
        modules_to_scan = []
        for category_dir in package_path.iterdir():
            if category_dir.is_dir() and not category_dir.name.startswith('__'):
                for module_file in category_dir.glob('*.py'):
                    if not module_file.name.startswith('__'):
                        module_path = f"{self.algorithms_package}.{category_dir.name}.{module_file.stem}"
                        modules_to_scan.append((module_path, category_dir.name))
        
        discovered_algorithms = {}
        discovery_errors = []
        
        # Process modules in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._scan_module, module_path, category): (module_path, category)
                for module_path, category in modules_to_scan
            }
            
            for future in as_completed(futures):
                module_path, category = futures[future]
                try:
                    result = future.result()
                    if result:
                        algorithm_name, algorithm_class = result
                        discovered_algorithms[algorithm_name] = algorithm_class
                        self.algorithm_modules[algorithm_name] = module_path
                except Exception as e:
                    discovery_errors.append((module_path, e))
                    logger.warning(f"Error scanning module {module_path}: {e}")
        
        self.algorithm_classes.update(discovered_algorithms)
        self.discovery_errors.extend(discovery_errors)
        
        return {
            'discovered_algorithms': list(discovered_algorithms.keys()),
            'scanned_modules': len(modules_to_scan),
            'discovery_errors': len(discovery_errors)
        }
    
    def _sequential_discovery(self, package_path: Path) -> Dict[str, Any]:
        """Discover algorithms sequentially"""
        logger.info("Using sequential discovery")
        
        discovered_algorithms = {}
        scanned_modules = 0
        discovery_errors = []
        
        for category_dir in package_path.iterdir():
            if category_dir.is_dir() and not category_dir.name.startswith('__'):
                for module_file in category_dir.glob('*.py'):
                    if not module_file.name.startswith('__'):
                        module_path = f"{self.algorithms_package}.{category_dir.name}.{module_file.stem}"
                        scanned_modules += 1
                        
                        try:
                            result = self._scan_module(module_path, category_dir.name)
                            if result:
                                algorithm_name, algorithm_class = result
                                discovered_algorithms[algorithm_name] = algorithm_class
                                self.algorithm_modules[algorithm_name] = module_path
                        except Exception as e:
                            discovery_errors.append((module_path, e))
                            logger.warning(f"Error scanning module {module_path}: {e}")
        
        self.algorithm_classes.update(discovered_algorithms)
        self.discovery_errors.extend(discovery_errors)
        
        return {
            'discovered_algorithms': list(discovered_algorithms.keys()),
            'scanned_modules': scanned_modules,
            'discovery_errors': len(discovery_errors)
        }
    
    def _scan_module(self, module_path: str, category: str) -> Optional[Tuple[str, Type[BaseOptimizer]]]:
        """Scan a module for optimizer classes"""
        try:
            module = importlib.import_module(module_path)
            algorithm_class = self._find_optimizer_class_in_module(module)
            
            if algorithm_class:
                # Validate the algorithm
                self.validate_algorithm(algorithm_class)
                
                # Generate algorithm name from class name
                algorithm_name = self._generate_algorithm_name(algorithm_class.__name__, category)
                
                logger.debug(f"Discovered algorithm: {algorithm_name} in {module_path}")
                return algorithm_name, algorithm_class
            
        except Exception as e:
            logger.debug(f"Error scanning {module_path}: {e}")
            raise
        
        return None
    
    def _find_optimizer_class_in_module(self, module) -> Optional[Type[BaseOptimizer]]:
        """Find the optimizer class in a module"""
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Skip imported classes (must be defined in this module)
            if obj.__module__ != module.__name__:
                continue
            
            # Check if it's a BaseOptimizer subclass
            if (issubclass(obj, BaseOptimizer) and 
                obj != BaseOptimizer and 
                not inspect.isabstract(obj)):
                return obj
        
        return None
    
    def _generate_algorithm_name(self, class_name: str, category: str) -> str:
        """Generate a standardized algorithm name"""
        # Remove common suffixes
        name = class_name
        suffixes = ['Optimizer', 'Algorithm', 'Optimization']
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                break
        
        # Convert CamelCase to snake_case
        import re
        name = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
        
        return name
    
    def _create_algorithm_instance(self, algorithm_class: Type[BaseOptimizer], **kwargs) -> BaseOptimizer:
        """Create an instance of an algorithm class"""
        # Default parameters for testing
        default_param_space = {'x': (-1.0, 1.0), 'y': (-1.0, 1.0)}
        default_objective = lambda params: sum(v**2 for v in params.values())
        
        # Use provided parameters or defaults
        param_space = kwargs.get('param_space', default_param_space)
        objective_function = kwargs.get('objective_function', default_objective)
        
        # Remove these from kwargs to avoid duplicate arguments
        algorithm_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['param_space', 'objective_function']}
        
        return algorithm_class(
            param_space=param_space,
            objective_function=objective_function,
            **algorithm_kwargs
        )
    
    def _create_default_metadata(self):
        """Create default metadata for algorithms that don't have it"""
        for algorithm_name, algorithm_class in self.algorithm_classes.items():
            if not self.metadata_manager.get_algorithm_metadata(algorithm_name):
                # Determine category from module path
                module_path = self.algorithm_modules.get(algorithm_name, '')
                category = self._determine_category_from_path(module_path)
                
                # Create default metadata
                metadata = AlgorithmMetadata(
                    name=algorithm_name,
                    category=category,
                    description=inspect.getdoc(algorithm_class) or f"{algorithm_name} optimization algorithm",
                    capabilities=self._infer_capabilities(algorithm_class),
                    resource_requirements=self._infer_resource_requirements(algorithm_class)
                )
                
                self.metadata_manager.register_algorithm_metadata(algorithm_name, metadata)
    
    def _determine_category_from_path(self, module_path: str) -> AlgorithmCategory:
        """Determine algorithm category from module path"""
        if 'classical' in module_path:
            return AlgorithmCategory.CLASSICAL
        elif 'evolutionary' in module_path:
            return AlgorithmCategory.EVOLUTIONARY
        elif 'swarm' in module_path:
            return AlgorithmCategory.SWARM
        elif 'physics' in module_path:
            return AlgorithmCategory.PHYSICS_INSPIRED
        elif 'quantum' in module_path:
            return AlgorithmCategory.QUANTUM
        else:
            return AlgorithmCategory.CLASSICAL  # Default
    
    def _infer_capabilities(self, algorithm_class: Type[BaseOptimizer]) -> AlgorithmCapabilities:
        """Infer algorithm capabilities from class analysis"""
        capabilities = AlgorithmCapabilities()
        
        # Default problem type support
        capabilities.problem_types.add(ProblemType.CONTINUOUS)
        
        # Check for GPU support hints
        class_source = inspect.getsource(algorithm_class)
        if any(keyword in class_source.lower() for keyword in ['cupy', 'gpu', 'cuda']):
            capabilities.supports_gpu = True
        
        # Check for parallel support hints
        if any(keyword in class_source.lower() for keyword in ['parallel', 'thread', 'multiprocess']):
            capabilities.supports_parallel = True
        
        return capabilities
    
    def _infer_resource_requirements(self, algorithm_class: Type[BaseOptimizer]) -> ResourceRequirements:
        """Infer resource requirements from algorithm analysis"""
        requirements = ResourceRequirements()
        
        # Set complexity based on algorithm type
        class_name = algorithm_class.__name__.lower()
        
        if any(keyword in class_name for keyword in ['grid', 'random', 'hill']):
            requirements.complexity = ComplexityLevel.LOW
        elif any(keyword in class_name for keyword in ['genetic', 'evolution', 'swarm']):
            requirements.complexity = ComplexityLevel.HIGH
        elif any(keyword in class_name for keyword in ['quantum']):
            requirements.complexity = ComplexityLevel.VERY_HIGH
        else:
            requirements.complexity = ComplexityLevel.MEDIUM
        
        # Adjust memory based on complexity
        complexity_memory_map = {
            ComplexityLevel.LOW: 50,
            ComplexityLevel.MEDIUM: 100,
            ComplexityLevel.HIGH: 200,
            ComplexityLevel.VERY_HIGH: 500
        }
        requirements.memory_mb = complexity_memory_map[requirements.complexity]
        
        return requirements
    
    def _get_algorithm_category(self, algorithm_name: str) -> AlgorithmCategory:
        """Get algorithm category from metadata"""
        metadata = self.metadata_manager.get_algorithm_metadata(algorithm_name)
        return metadata.category if metadata else AlgorithmCategory.CLASSICAL
    
    def _algorithm_supports_gpu(self, algorithm_name: str) -> bool:
        """Check if algorithm supports GPU"""
        metadata = self.metadata_manager.get_algorithm_metadata(algorithm_name)
        return metadata.capabilities.supports_gpu if metadata else False
    
    def _algorithm_supports_parallel(self, algorithm_name: str) -> bool:
        """Check if algorithm supports parallel processing"""
        metadata = self.metadata_manager.get_algorithm_metadata(algorithm_name)
        return metadata.capabilities.supports_parallel if metadata else False
    
    def _get_discovery_summary(self) -> Dict[str, Any]:
        """Get current discovery summary"""
        return {
            'discovery_completed': self.discovery_completed,
            'total_algorithms': len(self.algorithm_classes),
            'discovered_algorithms': list(self.algorithm_classes.keys()),
            'discovery_errors': len(self.discovery_errors)
        }
    
    def _clear_registry(self):
        """Clear all registry data"""
        self.algorithm_classes.clear()
        self.algorithm_instances.clear()
        self.algorithm_modules.clear()
        self.load_times.clear()
        self.usage_counts.clear()
        self.discovery_errors.clear()
        self.discovery_completed = False