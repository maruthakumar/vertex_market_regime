"""
Parameter Space Management

Utilities for handling optimization parameter spaces, bounds, and sampling.
"""

from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

class ParameterType(Enum):
    """Types of optimization parameters"""
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"

@dataclass
class Parameter:
    """Definition of a single optimization parameter"""
    name: str
    type: ParameterType
    bounds: Union[Tuple[float, float], List[Any]]
    default: Optional[Any] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        """Validate parameter definition"""
        if self.type in [ParameterType.CONTINUOUS, ParameterType.INTEGER]:
            if not isinstance(self.bounds, (tuple, list)) or len(self.bounds) != 2:
                raise ValueError(f"Parameter {self.name}: bounds must be (min, max) for {self.type}")
            if self.bounds[0] >= self.bounds[1]:
                raise ValueError(f"Parameter {self.name}: min must be < max")
        elif self.type == ParameterType.CATEGORICAL:
            if not isinstance(self.bounds, list) or len(self.bounds) < 2:
                raise ValueError(f"Parameter {self.name}: bounds must be list of choices for categorical")
        elif self.type == ParameterType.BOOLEAN:
            self.bounds = [True, False]

class ParameterSpace:
    """
    Manages optimization parameter space with type handling and sampling
    """
    
    def __init__(self, parameters: List[Parameter]):
        """
        Initialize parameter space
        
        Args:
            parameters: List of Parameter objects defining the space
        """
        self.parameters = {p.name: p for p in parameters}
        self.param_names = list(self.parameters.keys())
        self._validate_space()
        
    def _validate_space(self):
        """Validate parameter space consistency"""
        if not self.parameters:
            raise ValueError("Parameter space cannot be empty")
            
        for param in self.parameters.values():
            if param.default is not None:
                if not self.is_valid_value(param.name, param.default):
                    raise ValueError(f"Default value for {param.name} is outside bounds")
    
    def is_valid_value(self, param_name: str, value: Any) -> bool:
        """Check if value is valid for parameter"""
        if param_name not in self.parameters:
            return False
            
        param = self.parameters[param_name]
        
        if param.type == ParameterType.CONTINUOUS:
            return isinstance(value, (int, float)) and param.bounds[0] <= value <= param.bounds[1]
        elif param.type == ParameterType.INTEGER:
            return isinstance(value, (int, np.integer)) and param.bounds[0] <= value <= param.bounds[1]
        elif param.type == ParameterType.CATEGORICAL:
            return value in param.bounds
        elif param.type == ParameterType.BOOLEAN:
            return isinstance(value, bool)
            
        return False
    
    def clip_value(self, param_name: str, value: Any) -> Any:
        """Clip value to valid range for parameter"""
        if param_name not in self.parameters:
            return value
            
        param = self.parameters[param_name]
        
        if param.type == ParameterType.CONTINUOUS:
            return np.clip(value, param.bounds[0], param.bounds[1])
        elif param.type == ParameterType.INTEGER:
            clipped = np.clip(value, param.bounds[0], param.bounds[1])
            return int(round(clipped))
        elif param.type == ParameterType.CATEGORICAL:
            # Return closest valid category
            if value in param.bounds:
                return value
            # For numeric categories, find closest
            if all(isinstance(x, (int, float)) for x in param.bounds):
                closest_idx = np.argmin([abs(value - x) for x in param.bounds])
                return param.bounds[closest_idx]
            return param.bounds[0]  # Default to first option
        elif param.type == ParameterType.BOOLEAN:
            return bool(value)
            
        return value
    
    def random_sample(self, n_samples: int = 1) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Generate random samples from parameter space
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Single dict if n_samples=1, list of dicts otherwise
        """
        samples = []
        
        for _ in range(n_samples):
            sample = {}
            for param_name, param in self.parameters.items():
                if param.type == ParameterType.CONTINUOUS:
                    sample[param_name] = np.random.uniform(param.bounds[0], param.bounds[1])
                elif param.type == ParameterType.INTEGER:
                    sample[param_name] = np.random.randint(param.bounds[0], param.bounds[1] + 1)
                elif param.type == ParameterType.CATEGORICAL:
                    sample[param_name] = np.random.choice(param.bounds)
                elif param.type == ParameterType.BOOLEAN:
                    sample[param_name] = np.random.choice([True, False])
            samples.append(sample)
        
        return samples[0] if n_samples == 1 else samples
    
    def latin_hypercube_sample(self, n_samples: int) -> List[Dict[str, Any]]:
        """
        Generate Latin Hypercube samples for better space coverage
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            List of parameter dictionaries
        """
        try:
            from scipy.stats import qmc
            
            # Get continuous parameters for LHS
            continuous_params = [(name, param) for name, param in self.parameters.items() 
                               if param.type in [ParameterType.CONTINUOUS, ParameterType.INTEGER]]
            
            if not continuous_params:
                # No continuous parameters, fall back to random sampling
                return self.random_sample(n_samples)
            
            # Generate LHS samples for continuous parameters
            sampler = qmc.LatinHypercube(d=len(continuous_params))
            unit_samples = sampler.random(n_samples)
            
            samples = []
            for i in range(n_samples):
                sample = {}
                
                # Handle continuous parameters with LHS
                for j, (param_name, param) in enumerate(continuous_params):
                    unit_val = unit_samples[i, j]
                    if param.type == ParameterType.CONTINUOUS:
                        sample[param_name] = param.bounds[0] + unit_val * (param.bounds[1] - param.bounds[0])
                    elif param.type == ParameterType.INTEGER:
                        range_size = param.bounds[1] - param.bounds[0] + 1
                        sample[param_name] = param.bounds[0] + int(unit_val * range_size)
                        if sample[param_name] > param.bounds[1]:
                            sample[param_name] = param.bounds[1]
                
                # Handle categorical and boolean parameters randomly
                for param_name, param in self.parameters.items():
                    if param_name not in sample:
                        if param.type == ParameterType.CATEGORICAL:
                            sample[param_name] = np.random.choice(param.bounds)
                        elif param.type == ParameterType.BOOLEAN:
                            sample[param_name] = np.random.choice([True, False])
                
                samples.append(sample)
            
            return samples
            
        except ImportError:
            logger.warning("scipy not available, falling back to random sampling")
            return self.random_sample(n_samples)
    
    def grid_sample(self, n_points_per_dim: int = 10) -> List[Dict[str, Any]]:
        """
        Generate grid samples across parameter space
        
        Args:
            n_points_per_dim: Number of points per dimension
            
        Returns:
            List of parameter dictionaries
        """
        samples = []
        
        # Create grid points for each parameter
        param_grids = {}
        for param_name, param in self.parameters.items():
            if param.type == ParameterType.CONTINUOUS:
                param_grids[param_name] = np.linspace(param.bounds[0], param.bounds[1], n_points_per_dim)
            elif param.type == ParameterType.INTEGER:
                range_size = param.bounds[1] - param.bounds[0] + 1
                n_points = min(n_points_per_dim, range_size)
                param_grids[param_name] = np.linspace(param.bounds[0], param.bounds[1], n_points, dtype=int)
            elif param.type == ParameterType.CATEGORICAL:
                param_grids[param_name] = param.bounds
            elif param.type == ParameterType.BOOLEAN:
                param_grids[param_name] = [True, False]
        
        # Generate all combinations
        import itertools
        grid_combinations = itertools.product(*param_grids.values())
        
        for combination in grid_combinations:
            sample = {}
            for i, param_name in enumerate(self.param_names):
                sample[param_name] = combination[i]
            samples.append(sample)
        
        return samples
    
    def to_dict(self) -> Dict[str, Tuple[Any, Any]]:
        """Convert to simple dict format for backward compatibility"""
        result = {}
        for param_name, param in self.parameters.items():
            if param.type in [ParameterType.CONTINUOUS, ParameterType.INTEGER]:
                result[param_name] = param.bounds
            elif param.type == ParameterType.CATEGORICAL:
                # Use first and last for categorical (not ideal but maintains compatibility)
                result[param_name] = (param.bounds[0], param.bounds[-1])
            elif param.type == ParameterType.BOOLEAN:
                result[param_name] = (False, True)
        return result
    
    @classmethod
    def from_dict(cls, param_dict: Dict[str, Tuple[float, float]], 
                  param_types: Optional[Dict[str, ParameterType]] = None) -> 'ParameterSpace':
        """Create ParameterSpace from simple dict format"""
        parameters = []
        
        for param_name, bounds in param_dict.items():
            param_type = param_types.get(param_name, ParameterType.CONTINUOUS) if param_types else ParameterType.CONTINUOUS
            parameters.append(Parameter(
                name=param_name,
                type=param_type,
                bounds=bounds
            ))
        
        return cls(parameters)
    
    def get_bounds_dict(self) -> Dict[str, Tuple[float, float]]:
        """Get bounds in simple dict format"""
        return self.to_dict()
    
    def __len__(self) -> int:
        """Return number of parameters"""
        return len(self.parameters)
    
    def __repr__(self) -> str:
        return f"ParameterSpace({len(self.parameters)} parameters)"