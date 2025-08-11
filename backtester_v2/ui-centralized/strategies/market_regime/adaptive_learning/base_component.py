"""
AdaptiveComponent Base Class

Defines the standardized interface that all 8 components must implement for the
adaptive learning system. Provides common functionality for performance monitoring,
memory management, and schema validation.
"""

import time
import logging
import psutil
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import pandas as pd

from . import (
    PerformanceBudgetExceeded,
    SchemaValidationError,
    validate_performance_budget
)


@dataclass
class ComponentConfig:
    """Configuration structure for adaptive components."""
    component_id: str
    feature_count: int
    processing_budget_ms: int
    memory_budget_mb: int
    enable_gpu: bool = True
    enable_cache: bool = True
    cache_ttl_minutes: int = 15
    fallback_enabled: bool = True


@dataclass
class AnalysisResult:
    """Standardized analysis result structure."""
    component_id: str
    features: Dict[str, Union[float, int, str]]
    metadata: Dict[str, Any]
    processing_time_ms: float
    memory_usage_mb: float
    timestamp: float
    confidence_score: float
    success: bool
    error_message: Optional[str] = None


class AdaptiveComponent(ABC):
    """
    Abstract base class for all adaptive learning components.
    
    Provides standardized interface and common functionality for:
    - Performance monitoring and budget enforcement
    - Memory usage tracking and GPU management
    - Schema validation and feature consistency
    - Caching and fallback mechanisms
    """
    
    def __init__(self, config: ComponentConfig):
        """
        Initialize adaptive component.
        
        Args:
            config: Component configuration including budgets and settings
        """
        self.config = config
        self.logger = logging.getLogger(f"adaptive_learning.{config.component_id}")
        self._schema = None
        self._cache = None
        self._gpu_available = False
        self._process = psutil.Process()
        
        # Validate configuration
        self._validate_config()
        
        # Initialize component-specific resources
        self._initialize_resources()
        
        self.logger.info(f"Initialized {config.component_id} with {config.feature_count} features")
    
    def _validate_config(self) -> None:
        """Validate component configuration."""
        if self.config.feature_count <= 0:
            raise ValueError(f"Invalid feature count: {self.config.feature_count}")
        
        if self.config.processing_budget_ms <= 0:
            raise ValueError(f"Invalid processing budget: {self.config.processing_budget_ms}")
        
        if self.config.memory_budget_mb <= 0:
            raise ValueError(f"Invalid memory budget: {self.config.memory_budget_mb}")
    
    def _initialize_resources(self) -> None:
        """Initialize component-specific resources (GPU, cache, etc.)."""
        # Check GPU availability
        try:
            import cudf  # Check if RAPIDS is available
            self._gpu_available = True
            self.logger.info("GPU acceleration available")
        except ImportError:
            self._gpu_available = False
            self.logger.info("GPU acceleration not available, using CPU fallback")
        
        # Initialize cache if enabled
        if self.config.enable_cache:
            from .cache.local_cache import LocalFeatureCache
            self._cache = LocalFeatureCache(
                component_id=self.config.component_id,
                ttl_minutes=self.config.cache_ttl_minutes
            )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        memory_info = self._process.memory_info()
        return memory_info.rss / (1024 * 1024)  # Convert bytes to MB
    
    def _validate_memory_budget(self) -> None:
        """Check if memory usage is within budget."""
        current_memory = self._get_memory_usage()
        if current_memory > self.config.memory_budget_mb:
            error_msg = f"Memory budget exceeded: {current_memory:.2f}MB > {self.config.memory_budget_mb}MB"
            self.logger.error(error_msg)
            
            # Raise exception for critical memory threshold (>125% of budget)
            critical_threshold = self.config.memory_budget_mb * 1.25
            if current_memory > critical_threshold:
                from . import GPUMemoryError
                raise GPUMemoryError(f"Critical memory threshold exceeded: {current_memory:.2f}MB > {critical_threshold:.2f}MB")
            else:
                self.logger.warning("Memory budget exceeded - consider optimization")
    
    @abstractmethod
    def analyze(self, market_data: pd.DataFrame, **kwargs) -> AnalysisResult:
        """
        Perform component-specific analysis on market data.
        
        Args:
            market_data: Market data DataFrame
            **kwargs: Additional component-specific parameters
            
        Returns:
            AnalysisResult with features and metadata
            
        Raises:
            PerformanceBudgetExceeded: If processing exceeds budget
            SchemaValidationError: If output schema is invalid
        """
        pass
    
    @abstractmethod
    def optimize_weights(self, performance_feedback: Dict[str, float]) -> Dict[str, float]:
        """
        Optimize component weights based on historical performance.
        
        Args:
            performance_feedback: Historical performance metrics
            
        Returns:
            Dictionary of optimized weights
        """
        pass
    
    @abstractmethod
    def get_health_metrics(self) -> Dict[str, Any]:
        """
        Get component health and performance metrics.
        
        Returns:
            Dictionary containing health metrics
        """
        pass
    
    @abstractmethod
    def get_feature_schema(self) -> Dict[str, Any]:
        """
        Get component's feature schema definition.
        
        Returns:
            Dictionary containing feature schema
        """
        pass
    
    def analyze_with_monitoring(self, market_data: pd.DataFrame, **kwargs) -> AnalysisResult:
        """
        Wrapper for analyze() with performance monitoring and error handling.
        
        Args:
            market_data: Market data DataFrame
            **kwargs: Additional parameters
            
        Returns:
            AnalysisResult with monitoring data
        """
        start_time = time.time()
        initial_memory = self._get_memory_usage()
        
        try:
            # Pre-analysis validation
            self._validate_memory_budget()
            
            # Perform analysis
            result = self.analyze(market_data, **kwargs)
            
            # Post-analysis validation
            final_memory = self._get_memory_usage()
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Update result with monitoring data
            result.processing_time_ms = processing_time_ms
            result.memory_usage_mb = final_memory - initial_memory
            result.timestamp = start_time
            
            # Validate performance budget
            validate_performance_budget(
                start_time, 
                self.config.processing_budget_ms, 
                f"{self.config.component_id}.analyze"
            )
            
            # Validate memory usage
            self._validate_memory_budget()
            
            self.logger.debug(
                f"Analysis completed: {processing_time_ms:.2f}ms, "
                f"{result.memory_usage_mb:.2f}MB, "
                f"confidence={result.confidence_score:.3f}"
            )
            
            return result
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            final_memory = self._get_memory_usage()
            
            self.logger.error(f"Analysis failed after {processing_time_ms:.2f}ms: {str(e)}")
            
            # Return error result
            return AnalysisResult(
                component_id=self.config.component_id,
                features={},
                metadata={"error": str(e)},
                processing_time_ms=processing_time_ms,
                memory_usage_mb=final_memory - initial_memory,
                timestamp=start_time,
                confidence_score=0.0,
                success=False,
                error_message=str(e)
            )
    
    def cleanup(self) -> None:
        """Cleanup component resources."""
        if self._cache:
            self._cache.cleanup()
        
        # GPU memory cleanup if available
        if self._gpu_available:
            try:
                import cudf
                # Force garbage collection for GPU memory
                import gc
                gc.collect()
                self.logger.debug("GPU memory cleanup completed")
            except Exception as e:
                self.logger.warning(f"GPU cleanup warning: {str(e)}")
        
        self.logger.info(f"Component {self.config.component_id} cleanup completed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()