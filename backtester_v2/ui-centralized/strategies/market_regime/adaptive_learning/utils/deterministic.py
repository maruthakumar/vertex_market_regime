"""
Deterministic Transform Framework

Provides pure function architecture with no side effects for reproducible
feature engineering across all 8 components. Ensures consistent results
for backtesting, validation, and production environments.
"""

import hashlib
import logging
import random
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass
from functools import wraps
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import GPU libraries for deterministic operations
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


@dataclass
class DeterministicConfig:
    """Configuration for deterministic operations."""
    base_seed: int = 42
    component_seed_offset: int = 1000
    operation_seed_offset: int = 100
    enable_validation: bool = True
    strict_mode: bool = True  # Fail on any non-deterministic operations


class SeedManager:
    """
    Manages random seeds for reproducible operations across components.
    
    Provides hierarchical seed management:
    - Base seed for overall reproducibility
    - Component-specific seeds for isolation
    - Operation-specific seeds for fine-grained control
    """
    
    def __init__(self, config: DeterministicConfig = None):
        """
        Initialize seed manager.
        
        Args:
            config: Deterministic configuration
        """
        self.config = config or DeterministicConfig()
        self._seed_history = []
        self._operation_counter = 0
        
        # Set initial global seeds
        self._set_global_seeds(self.config.base_seed)
        
        logger.info(f"SeedManager initialized with base seed {self.config.base_seed}")
    
    def _set_global_seeds(self, seed: int) -> None:
        """Set seeds for all random number generators."""
        # Python random
        random.seed(seed)
        
        # NumPy
        np.random.seed(seed)
        
        # GPU random state if available
        if GPU_AVAILABLE:
            try:
                cp.random.seed(seed)
            except Exception as e:
                logger.warning(f"Could not set GPU random seed: {str(e)}")
        
        self._seed_history.append({
            "seed": seed,
            "timestamp": np.datetime64('now'),
            "operation": "global_seed_set"
        })
    
    def get_component_seed(self, component_id: str) -> int:
        """
        Get deterministic seed for a specific component.
        
        Args:
            component_id: Component identifier
            
        Returns:
            Deterministic seed for component
        """
        # Hash component ID for consistent seed derivation
        component_hash = hashlib.sha256(component_id.encode()).digest()
        component_offset = int.from_bytes(component_hash[:4], byteorder='big') % 10000
        
        seed = self.config.base_seed + self.config.component_seed_offset + component_offset
        return seed
    
    def get_operation_seed(self, component_id: str, operation_name: str) -> int:
        """
        Get deterministic seed for a specific operation within a component.
        
        Args:
            component_id: Component identifier  
            operation_name: Operation name
            
        Returns:
            Deterministic seed for operation
        """
        # Combine component and operation for unique seed
        operation_key = f"{component_id}::{operation_name}"
        operation_hash = hashlib.sha256(operation_key.encode()).digest()
        operation_offset = int.from_bytes(operation_hash[:4], byteorder='big') % 1000
        
        component_seed = self.get_component_seed(component_id)
        seed = component_seed + self.config.operation_seed_offset + operation_offset
        
        self._operation_counter += 1
        
        return seed
    
    def create_operation_context(self, component_id: str, operation_name: str):
        """
        Create context manager for deterministic operations.
        
        Args:
            component_id: Component identifier
            operation_name: Operation name
            
        Returns:
            Context manager that sets and restores random state
        """
        return DeterministicContext(self, component_id, operation_name)


class DeterministicContext:
    """Context manager for deterministic operations."""
    
    def __init__(self, seed_manager: SeedManager, component_id: str, operation_name: str):
        """
        Initialize deterministic context.
        
        Args:
            seed_manager: Seed manager instance
            component_id: Component identifier
            operation_name: Operation name
        """
        self.seed_manager = seed_manager
        self.component_id = component_id
        self.operation_name = operation_name
        self.operation_seed = None
        
        # Store original random states
        self._original_python_state = None
        self._original_numpy_state = None
        self._original_gpu_state = None
    
    def __enter__(self):
        """Enter deterministic context."""
        # Store original states
        self._original_python_state = random.getstate()
        self._original_numpy_state = np.random.get_state()
        
        if GPU_AVAILABLE:
            try:
                self._original_gpu_state = cp.random.get_random_state()
            except Exception:
                pass
        
        # Set operation-specific seed
        self.operation_seed = self.seed_manager.get_operation_seed(
            self.component_id, self.operation_name
        )
        self.seed_manager._set_global_seeds(self.operation_seed)
        
        logger.debug(f"Deterministic context entered: {self.component_id}::{self.operation_name} seed={self.operation_seed}")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit deterministic context and restore random states."""
        # Restore original states
        if self._original_python_state:
            random.setstate(self._original_python_state)
        
        if self._original_numpy_state:
            np.random.set_state(self._original_numpy_state)
        
        if GPU_AVAILABLE and self._original_gpu_state:
            try:
                cp.random.set_random_state(self._original_gpu_state)
            except Exception:
                pass
        
        logger.debug(f"Deterministic context exited: {self.component_id}::{self.operation_name}")


class DeterministicTransform:
    """
    Base class for deterministic transforms with input validation and type checking.
    """
    
    def __init__(self, component_id: str, seed_manager: SeedManager = None):
        """
        Initialize deterministic transform.
        
        Args:
            component_id: Component identifier
            seed_manager: Seed manager (creates default if None)
        """
        self.component_id = component_id
        self.seed_manager = seed_manager or SeedManager()
        self._validation_enabled = True
        self._strict_mode = True
    
    def validate_inputs(self, **kwargs) -> Dict[str, Any]:
        """
        Validate and sanitize input parameters.
        
        Args:
            **kwargs: Input parameters to validate
            
        Returns:
            Validated and sanitized parameters
            
        Raises:
            ValueError: If validation fails
        """
        validated = {}
        
        for key, value in kwargs.items():
            # Basic type validation
            if value is None:
                if self._strict_mode:
                    raise ValueError(f"Parameter {key} cannot be None in strict mode")
                validated[key] = value
                continue
            
            # DataFrame validation
            if isinstance(value, (pd.DataFrame, )):
                if GPU_AVAILABLE:
                    try:
                        import cudf
                        if isinstance(value, cudf.DataFrame):
                            # Validate cuDF DataFrame
                            if len(value) == 0:
                                raise ValueError(f"DataFrame {key} is empty")
                    except ImportError:
                        pass
                
                # Validate pandas DataFrame
                if isinstance(value, pd.DataFrame):
                    if len(value) == 0:
                        raise ValueError(f"DataFrame {key} is empty")
                    
                    # Check for NaN values in strict mode
                    if self._strict_mode and value.isnull().any().any():
                        raise ValueError(f"DataFrame {key} contains NaN values")
                
                validated[key] = value
            
            # Numeric validation
            elif isinstance(value, (int, float, np.number)):
                if np.isnan(value) or np.isinf(value):
                    if self._strict_mode:
                        raise ValueError(f"Parameter {key} has invalid numeric value: {value}")
                    # Replace with default value
                    validated[key] = 0.0 if isinstance(value, float) else 0
                else:
                    validated[key] = value
            
            # Array validation  
            elif isinstance(value, (list, np.ndarray)):
                if len(value) == 0:
                    if self._strict_mode:
                        raise ValueError(f"Array parameter {key} is empty")
                
                # Check for invalid values in arrays
                if isinstance(value, np.ndarray):
                    if np.isnan(value).any() or np.isinf(value).any():
                        if self._strict_mode:
                            raise ValueError(f"Array {key} contains invalid values")
                
                validated[key] = value
            
            else:
                # Pass through other types
                validated[key] = value
        
        return validated
    
    def transform_deterministic(
        self, 
        operation_name: str, 
        transform_func: Callable,
        **kwargs
    ) -> Any:
        """
        Apply deterministic transform with proper seed management.
        
        Args:
            operation_name: Name of transform operation
            transform_func: Transform function to apply
            **kwargs: Parameters for transform function
            
        Returns:
            Transform result
        """
        # Validate inputs
        validated_kwargs = self.validate_inputs(**kwargs)
        
        # Apply transform in deterministic context
        with self.seed_manager.create_operation_context(self.component_id, operation_name):
            try:
                result = transform_func(**validated_kwargs)
                
                # Validate output if enabled
                if self._validation_enabled:
                    self._validate_output(result, operation_name)
                
                return result
                
            except Exception as e:
                logger.error(f"Deterministic transform failed: {self.component_id}::{operation_name}: {str(e)}")
                raise
    
    def _validate_output(self, result: Any, operation_name: str) -> None:
        """Validate transform output."""
        if result is None:
            if self._strict_mode:
                raise ValueError(f"Transform {operation_name} returned None")
            return
        
        # DataFrame output validation
        if isinstance(result, (pd.DataFrame, )):
            if GPU_AVAILABLE:
                try:
                    import cudf
                    if isinstance(result, cudf.DataFrame):
                        if len(result) == 0 and self._strict_mode:
                            raise ValueError(f"Transform {operation_name} returned empty cuDF DataFrame")
                except ImportError:
                    pass
            
            if isinstance(result, pd.DataFrame):
                if len(result) == 0 and self._strict_mode:
                    raise ValueError(f"Transform {operation_name} returned empty DataFrame")
                
                # Check for invalid values
                if self._strict_mode:
                    numeric_cols = result.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        if result[col].isnull().any():
                            raise ValueError(f"Transform {operation_name} result contains NaN in column {col}")
        
        # Numeric output validation
        elif isinstance(result, (int, float, np.number)):
            if np.isnan(result) or np.isinf(result):
                if self._strict_mode:
                    raise ValueError(f"Transform {operation_name} returned invalid numeric value: {result}")
        
        # Array output validation
        elif isinstance(result, np.ndarray):
            if np.isnan(result).any() or np.isinf(result).any():
                if self._strict_mode:
                    raise ValueError(f"Transform {operation_name} returned array with invalid values")


def deterministic_operation(
    component_id: str,
    operation_name: str = None,
    seed_manager: SeedManager = None,
    validate_inputs: bool = True,
    validate_outputs: bool = True
):
    """
    Decorator for deterministic operations.
    
    Args:
        component_id: Component identifier
        operation_name: Operation name (uses function name if None)
        seed_manager: Seed manager instance
        validate_inputs: Enable input validation
        validate_outputs: Enable output validation
    """
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or func.__name__
        transform = DeterministicTransform(component_id, seed_manager)
        
        # Configure validation
        transform._validation_enabled = validate_outputs
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Convert args to kwargs for validation
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            return transform.transform_deterministic(
                op_name, 
                func, 
                **bound_args.arguments
            )
        
        return wrapper
    return decorator


def create_reproducible_hash(data: Union[str, bytes, Dict, List]) -> str:
    """
    Create reproducible hash of data for caching and validation.
    
    Args:
        data: Data to hash
        
    Returns:
        Hexadecimal hash string
    """
    if isinstance(data, str):
        data_bytes = data.encode('utf-8')
    elif isinstance(data, bytes):
        data_bytes = data
    elif isinstance(data, (dict, list)):
        # Convert to sorted JSON string for consistency
        import json
        json_str = json.dumps(data, sort_keys=True, ensure_ascii=True)
        data_bytes = json_str.encode('utf-8')
    else:
        # Convert to string representation
        data_bytes = str(data).encode('utf-8')
    
    return hashlib.sha256(data_bytes).hexdigest()


def validate_reproducibility(
    func: Callable,
    test_inputs: List[Dict[str, Any]],
    component_id: str = "test_component",
    num_runs: int = 3
) -> bool:
    """
    Validate that function produces reproducible results.
    
    Args:
        func: Function to test
        test_inputs: List of test input dictionaries
        component_id: Component ID for seeding
        num_runs: Number of test runs
        
    Returns:
        True if function is reproducible
    """
    logger.info(f"Testing reproducibility for {func.__name__} with {num_runs} runs")
    
    for input_idx, inputs in enumerate(test_inputs):
        reference_result = None
        
        for run in range(num_runs):
            # Create fresh seed manager for each run
            seed_manager = SeedManager()
            
            with seed_manager.create_operation_context(component_id, func.__name__):
                try:
                    result = func(**inputs)
                    
                    if run == 0:
                        reference_result = result
                    else:
                        # Compare with reference
                        if not _results_equal(result, reference_result):
                            logger.error(
                                f"Reproducibility test failed for {func.__name__} "
                                f"input {input_idx}, run {run}"
                            )
                            return False
                
                except Exception as e:
                    logger.error(f"Test run failed: {str(e)}")
                    return False
    
    logger.info(f"Reproducibility test passed for {func.__name__}")
    return True


def _results_equal(result1: Any, result2: Any, tolerance: float = 1e-10) -> bool:
    """Compare two results for equality with tolerance for floating point."""
    if type(result1) != type(result2):
        return False
    
    if isinstance(result1, (pd.DataFrame, )):
        try:
            # Check if cuDF DataFrame
            import cudf
            if isinstance(result1, cudf.DataFrame):
                return result1.equals(result2)
        except ImportError:
            pass
        
        if isinstance(result1, pd.DataFrame):
            try:
                pd.testing.assert_frame_equal(result1, result2, atol=tolerance)
                return True
            except AssertionError:
                return False
    
    elif isinstance(result1, np.ndarray):
        return np.allclose(result1, result2, atol=tolerance)
    
    elif isinstance(result1, (int, float)):
        if isinstance(result1, float):
            return abs(result1 - result2) <= tolerance
        else:
            return result1 == result2
    
    else:
        return result1 == result2