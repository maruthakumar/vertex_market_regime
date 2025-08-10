"""
Base Indicator Class for Market Regime Indicators
================================================

Provides common interface and functionality for all market regime indicators
with performance tracking, configuration management, and state handling.

Author: Market Regime Refactoring Team
Date: 2025-07-06
Version: 2.0.0 - Refactored Architecture
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class IndicatorState(Enum):
    """Indicator operational states"""
    UNINITIALIZED = "uninitialized"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    DISABLED = "disabled"

@dataclass
class IndicatorConfig:
    """Configuration for indicators"""
    name: str
    enabled: bool = True
    weight: float = 1.0
    parameters: Dict[str, Any] = field(default_factory=dict)
    strike_selection_strategy: str = "dynamic_range"
    dte_specific_weights: Dict[int, float] = field(default_factory=dict)
    performance_tracking: bool = True
    adaptive_weights: bool = True
    confidence_threshold: float = 0.5
    max_computation_time: float = 5.0  # seconds

@dataclass
class IndicatorOutput:
    """Standardized indicator output"""
    value: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    computation_time: float = 0.0
    data_quality: float = 1.0

class BaseIndicator(ABC):
    """
    Base class for all market regime indicators
    
    Provides common functionality including:
    - Configuration management
    - Performance tracking
    - State management
    - Error handling
    - Standardized interface
    """
    
    def __init__(self, config: IndicatorConfig):
        """Initialize base indicator"""
        self.config = config
        self.state = IndicatorState.UNINITIALIZED
        self.current_weight = config.weight
        self.weight_history: List[Tuple[datetime, float]] = []
        
        # Performance tracking
        self.performance_metrics = {
            'accuracy': 0.5,
            'precision': 0.5,
            'recall': 0.5,
            'confidence': 0.5,
            'computation_time': 0.0,
            'error_count': 0,
            'success_count': 0
        }
        
        # Internal state
        self._last_computation = None
        self._error_log: List[Tuple[datetime, str]] = []
        
        logger.info(f"Initialized {self.__class__.__name__} with config: {config.name}")
    
    @abstractmethod
    def analyze(self, market_data: pd.DataFrame, **kwargs) -> IndicatorOutput:
        """
        Main analysis method - must be implemented by subclasses
        
        Args:
            market_data: Market data DataFrame with required columns
            **kwargs: Additional parameters specific to indicator
            
        Returns:
            IndicatorOutput: Standardized output with value and metadata
        """
        pass
    
    @abstractmethod
    def validate_data(self, market_data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate input data - must be implemented by subclasses
        
        Args:
            market_data: Market data to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        pass
    
    @abstractmethod
    def get_required_columns(self) -> List[str]:
        """
        Get list of required DataFrame columns
        
        Returns:
            List[str]: Required column names
        """
        pass
    
    def execute_analysis(self, market_data: pd.DataFrame, **kwargs) -> IndicatorOutput:
        """
        Execute analysis with error handling and performance tracking
        
        Args:
            market_data: Market data DataFrame
            **kwargs: Additional parameters
            
        Returns:
            IndicatorOutput: Analysis result with metadata
        """
        start_time = datetime.now()
        
        try:
            # Update state
            self.state = IndicatorState.PROCESSING
            
            # Validate data
            is_valid, errors = self.validate_data(market_data)
            if not is_valid:
                self.state = IndicatorState.ERROR
                self._log_error(f"Data validation failed: {errors}")
                return self._get_error_output(f"Data validation failed: {errors[0] if errors else 'Unknown error'}")
            
            # Perform analysis
            result = self.analyze(market_data, **kwargs)
            
            # Update performance metrics
            computation_time = (datetime.now() - start_time).total_seconds()
            result.computation_time = computation_time
            
            self._update_performance_metrics(result, computation_time)
            
            # Update state
            self.state = IndicatorState.READY
            self._last_computation = datetime.now()
            self.performance_metrics['success_count'] += 1
            
            logger.debug(f"{self.config.name} analysis completed in {computation_time:.3f}s")
            
            return result
            
        except Exception as e:
            self.state = IndicatorState.ERROR
            self._log_error(f"Analysis failed: {str(e)}")
            self.performance_metrics['error_count'] += 1
            
            return self._get_error_output(f"Analysis failed: {str(e)}")
    
    def update_weight(self, new_weight: float, reason: str = "manual"):
        """
        Update indicator weight
        
        Args:
            new_weight: New weight value
            reason: Reason for weight update
        """
        old_weight = self.current_weight
        self.current_weight = np.clip(new_weight, 0.01, 3.0)
        self.weight_history.append((datetime.now(), self.current_weight))
        
        logger.info(f"{self.config.name} weight updated: {old_weight:.3f} -> {self.current_weight:.3f} ({reason})")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary
        
        Returns:
            Dict[str, Any]: Performance metrics and statistics
        """
        total_computations = self.performance_metrics['success_count'] + self.performance_metrics['error_count']
        success_rate = self.performance_metrics['success_count'] / max(total_computations, 1)
        
        return {
            'indicator_name': self.config.name,
            'state': self.state.value,
            'current_weight': self.current_weight,
            'success_rate': success_rate,
            'performance_metrics': self.performance_metrics.copy(),
            'last_computation': self._last_computation,
            'error_count': len(self._error_log),
            'recent_errors': [error[1] for error in self._error_log[-5:]]  # Last 5 errors
        }
    
    def reset_performance(self):
        """Reset performance tracking"""
        self.performance_metrics = {
            'accuracy': 0.5,
            'precision': 0.5,
            'recall': 0.5,
            'confidence': 0.5,
            'computation_time': 0.0,
            'error_count': 0,
            'success_count': 0
        }
        self._error_log.clear()
        logger.info(f"{self.config.name} performance tracking reset")
    
    def is_healthy(self) -> bool:
        """
        Check if indicator is healthy
        
        Returns:
            bool: True if indicator is operating normally
        """
        total_computations = self.performance_metrics['success_count'] + self.performance_metrics['error_count']
        
        if total_computations == 0:
            return True  # No data yet
        
        success_rate = self.performance_metrics['success_count'] / total_computations
        avg_computation_time = self.performance_metrics['computation_time']
        
        return (
            self.state != IndicatorState.ERROR and
            success_rate > 0.8 and
            avg_computation_time < self.config.max_computation_time
        )
    
    def _update_performance_metrics(self, result: IndicatorOutput, computation_time: float):
        """Update performance metrics based on result"""
        # Update computation time (exponential moving average)
        alpha = 0.1
        current_time = self.performance_metrics['computation_time']
        self.performance_metrics['computation_time'] = alpha * computation_time + (1 - alpha) * current_time
        
        # Update confidence (exponential moving average)
        current_confidence = self.performance_metrics['confidence']
        self.performance_metrics['confidence'] = alpha * result.confidence + (1 - alpha) * current_confidence
    
    def _log_error(self, error_message: str):
        """Log error with timestamp"""
        self._error_log.append((datetime.now(), error_message))
        
        # Keep only last 50 errors
        if len(self._error_log) > 50:
            self._error_log = self._error_log[-50:]
        
        logger.error(f"{self.config.name}: {error_message}")
    
    def _get_error_output(self, error_message: str) -> IndicatorOutput:
        """Get standardized error output"""
        return IndicatorOutput(
            value=0.0,
            confidence=0.0,
            metadata={
                'error': True,
                'error_message': error_message,
                'state': self.state.value
            },
            data_quality=0.0
        )
    
    def __str__(self):
        return f"{self.__class__.__name__}(name={self.config.name}, weight={self.current_weight:.3f}, state={self.state.value})"
    
    def __repr__(self):
        return self.__str__()