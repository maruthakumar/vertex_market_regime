"""
Objective Function Utilities

Common objective functions and wrappers for strategy optimization.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
import numpy as np
import pandas as pd
from functools import wraps
import time
import logging

logger = logging.getLogger(__name__)

class ObjectiveFunction(ABC):
    """Base class for objective functions"""
    
    def __init__(self, 
                 name: str,
                 maximize: bool = True,
                 cache_results: bool = True):
        self.name = name
        self.maximize = maximize
        self.cache_results = cache_results
        self._cache = {} if cache_results else None
        self._evaluation_count = 0
        
    @abstractmethod
    def evaluate(self, params: Dict[str, float]) -> float:
        """Evaluate objective function with given parameters"""
        pass
    
    def __call__(self, params: Dict[str, float]) -> float:
        """Make objective function callable"""
        self._evaluation_count += 1
        
        # Check cache if enabled
        if self.cache_results:
            param_key = tuple(sorted(params.items()))
            if param_key in self._cache:
                return self._cache[param_key]
        
        # Evaluate function
        result = self.evaluate(params)
        
        # Store in cache if enabled
        if self.cache_results:
            self._cache[param_key] = result
            
        return result
    
    def get_evaluation_count(self) -> int:
        """Get number of function evaluations"""
        return self._evaluation_count
    
    def clear_cache(self):
        """Clear evaluation cache"""
        if self._cache is not None:
            self._cache.clear()
            
    def get_cache_size(self) -> int:
        """Get cache size"""
        return len(self._cache) if self._cache else 0

class SharpeRatioObjective(ObjectiveFunction):
    """Sharpe ratio objective function for strategy optimization"""
    
    def __init__(self, 
                 returns_data: pd.Series,
                 risk_free_rate: float = 0.02,
                 **kwargs):
        super().__init__("SharpeRatio", maximize=True, **kwargs)
        self.returns_data = returns_data
        self.risk_free_rate = risk_free_rate
        
    def evaluate(self, params: Dict[str, float]) -> float:
        """Calculate Sharpe ratio for given parameters"""
        try:
            # Apply parameters to generate strategy returns
            strategy_returns = self._apply_strategy(params)
            
            if len(strategy_returns) < 2:
                return -np.inf
                
            excess_returns = strategy_returns - self.risk_free_rate / 252
            
            if excess_returns.std() == 0:
                return 0.0
                
            sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
            return sharpe
            
        except Exception as e:
            logger.warning(f"Error evaluating Sharpe ratio: {e}")
            return -np.inf
    
    def _apply_strategy(self, params: Dict[str, float]) -> pd.Series:
        """Apply strategy parameters to generate returns"""
        # This is a placeholder - implement actual strategy logic
        return self.returns_data

class MaxDrawdownObjective(ObjectiveFunction):
    """Maximum drawdown objective function (minimize)"""
    
    def __init__(self, 
                 returns_data: pd.Series,
                 **kwargs):
        super().__init__("MaxDrawdown", maximize=False, **kwargs)
        self.returns_data = returns_data
        
    def evaluate(self, params: Dict[str, float]) -> float:
        """Calculate maximum drawdown for given parameters"""
        try:
            strategy_returns = self._apply_strategy(params)
            
            if len(strategy_returns) < 2:
                return np.inf
                
            cumulative = (1 + strategy_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            
            max_dd = drawdown.min()
            return abs(max_dd)  # Return positive value to minimize
            
        except Exception as e:
            logger.warning(f"Error evaluating max drawdown: {e}")
            return np.inf
    
    def _apply_strategy(self, params: Dict[str, float]) -> pd.Series:
        """Apply strategy parameters to generate returns"""
        return self.returns_data

class ProfitFactorObjective(ObjectiveFunction):
    """Profit factor objective function (gross profit / gross loss)"""
    
    def __init__(self, 
                 returns_data: pd.Series,
                 **kwargs):
        super().__init__("ProfitFactor", maximize=True, **kwargs)
        self.returns_data = returns_data
        
    def evaluate(self, params: Dict[str, float]) -> float:
        """Calculate profit factor for given parameters"""
        try:
            strategy_returns = self._apply_strategy(params)
            
            if len(strategy_returns) < 2:
                return 0.0
                
            profits = strategy_returns[strategy_returns > 0].sum()
            losses = abs(strategy_returns[strategy_returns < 0].sum())
            
            if losses == 0:
                return profits if profits > 0 else 0.0
                
            return profits / losses
            
        except Exception as e:
            logger.warning(f"Error evaluating profit factor: {e}")
            return 0.0
    
    def _apply_strategy(self, params: Dict[str, float]) -> pd.Series:
        """Apply strategy parameters to generate returns"""
        return self.returns_data

class MultiObjectiveFunction(ObjectiveFunction):
    """Combine multiple objectives with weights"""
    
    def __init__(self,
                 objectives: List[ObjectiveFunction],
                 weights: Optional[List[float]] = None,
                 **kwargs):
        super().__init__("MultiObjective", maximize=True, **kwargs)
        self.objectives = objectives
        self.weights = weights or [1.0] * len(objectives)
        
        if len(self.weights) != len(self.objectives):
            raise ValueError("Number of weights must match number of objectives")
    
    def evaluate(self, params: Dict[str, float]) -> float:
        """Evaluate weighted combination of objectives"""
        total_score = 0.0
        
        for obj, weight in zip(self.objectives, self.weights):
            score = obj(params)
            
            # Convert to maximization if needed
            if not obj.maximize:
                score = -score
                
            total_score += weight * score
            
        return total_score

def gpu_accelerated(func: Callable) -> Callable:
    """Decorator to add GPU acceleration to objective functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if GPU acceleration is available and beneficial
        try:
            import cupy as cp
            # Convert numpy arrays to cupy if beneficial
            return func(*args, **kwargs)
        except ImportError:
            return func(*args, **kwargs)
    return wrapper

def timed_evaluation(func: Callable) -> Callable:
    """Decorator to time objective function evaluations"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        evaluation_time = time.time() - start_time
        
        if not hasattr(self, '_evaluation_times'):
            self._evaluation_times = []
        self._evaluation_times.append(evaluation_time)
        
        return result
    return wrapper

def robust_evaluation(func: Callable) -> Callable:
    """Decorator to add robustness to objective function evaluation"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            result = func(self, *args, **kwargs)
            if np.isnan(result) or np.isinf(result):
                logger.warning("Objective function returned NaN/Inf, returning worst score")
                return float('-inf') if self.maximize else float('inf')
            return result
        except Exception as e:
            logger.error(f"Error in objective function: {e}")
            return float('-inf') if self.maximize else float('inf')
    return wrapper