"""
GPU Memory Management Utilities

Provides automatic GPU memory cleanup, monitoring, and optimization for the
8-component adaptive learning system. Ensures efficient GPU memory usage
within the <3.7GB total system memory constraint.
"""

import time
import logging
import threading
from typing import Dict, Any, Optional, Callable, ContextManager
from dataclasses import dataclass
from contextlib import contextmanager

from .. import GPUMemoryError

logger = logging.getLogger(__name__)

# Try to import GPU libraries
try:
    import cupy as cp
    import cudf
    GPU_AVAILABLE = True
    logger.info("GPU libraries available (CuPy/cuDF)")
except ImportError:
    GPU_AVAILABLE = False
    logger.info("GPU libraries not available")


@dataclass
class MemoryStats:
    """GPU memory statistics."""
    total_bytes: int
    used_bytes: int
    free_bytes: int
    utilization_percent: float
    timestamp: float


class GPUMemoryManager:
    """
    GPU memory manager with automatic cleanup and monitoring.
    
    Provides:
    - Automatic memory cleanup and garbage collection
    - Memory usage monitoring and alerts
    - Context managers for memory-safe operations
    - Memory pool management for RAPIDS
    """
    
    def __init__(
        self, 
        memory_limit_mb: int = 1500,  # Conservative limit for multi-component system
        cleanup_threshold: float = 0.85,  # Cleanup when 85% full
        enable_monitoring: bool = True
    ):
        """
        Initialize GPU memory manager.
        
        Args:
            memory_limit_mb: Soft memory limit in MB
            cleanup_threshold: Cleanup threshold (0.0-1.0)
            enable_monitoring: Enable background monitoring
        """
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self.cleanup_threshold = cleanup_threshold
        self.enable_monitoring = enable_monitoring
        self.gpu_available = GPU_AVAILABLE
        
        self._memory_pool = None
        self._monitor_thread = None
        self._monitoring = False
        self._stats_history = []
        
        if self.gpu_available:
            self._initialize_gpu_resources()
        else:
            logger.warning("GPU not available - memory management disabled")
    
    def _initialize_gpu_resources(self) -> None:
        """Initialize GPU resources and memory pools."""
        try:
            # Initialize memory pool
            if hasattr(cp, 'get_default_memory_pool'):
                self._memory_pool = cp.get_default_memory_pool()
                logger.info(f"GPU memory pool initialized with {self.memory_limit_bytes / (1024*1024):.0f}MB limit")
            
            # Set memory limit if supported
            if hasattr(cp.cuda, 'MemoryPool'):
                try:
                    self._memory_pool.set_limit(size=self.memory_limit_bytes)
                except Exception as e:
                    logger.warning(f"Could not set memory limit: {str(e)}")
            
            # Start monitoring if enabled
            if self.enable_monitoring:
                self._start_monitoring()
                
        except Exception as e:
            logger.error(f"GPU resource initialization failed: {str(e)}")
            self.gpu_available = False
    
    def _start_monitoring(self) -> None:
        """Start background memory monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
        self._monitor_thread.start()
        logger.info("GPU memory monitoring started")
    
    def _monitor_memory(self) -> None:
        """Background memory monitoring loop."""
        while self._monitoring:
            try:
                stats = self.get_memory_stats()
                if stats:
                    self._stats_history.append(stats)
                    
                    # Keep only last 100 samples
                    if len(self._stats_history) > 100:
                        self._stats_history.pop(0)
                    
                    # Check for cleanup trigger
                    if stats.utilization_percent > self.cleanup_threshold * 100:
                        logger.warning(
                            f"GPU memory utilization high: {stats.utilization_percent:.1f}% - "
                            f"triggering cleanup"
                        )
                        self.cleanup_memory()
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {str(e)}")
                time.sleep(10)  # Back off on error
    
    def get_memory_stats(self) -> Optional[MemoryStats]:
        """
        Get current GPU memory statistics.
        
        Returns:
            MemoryStats object or None if GPU not available
        """
        if not self.gpu_available or not self._memory_pool:
            return None
        
        try:
            used_bytes = self._memory_pool.used_bytes()
            total_bytes = self._memory_pool.total_bytes()
            free_bytes = total_bytes - used_bytes if total_bytes > 0 else 0
            
            utilization = (used_bytes / total_bytes * 100) if total_bytes > 0 else 0
            
            return MemoryStats(
                total_bytes=total_bytes,
                used_bytes=used_bytes,
                free_bytes=free_bytes,
                utilization_percent=utilization,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Failed to get memory stats: {str(e)}")
            return None
    
    def cleanup_memory(self, force: bool = False) -> bool:
        """
        Cleanup GPU memory.
        
        Args:
            force: Force aggressive cleanup
            
        Returns:
            True if cleanup was successful
        """
        if not self.gpu_available:
            return True
        
        try:
            # Standard garbage collection
            import gc
            gc.collect()
            
            # RAPIDS-specific cleanup
            if self._memory_pool:
                if force:
                    self._memory_pool.free_all_blocks()
                else:
                    # Free unused blocks only
                    if hasattr(self._memory_pool, 'free_all_unused_blocks'):
                        self._memory_pool.free_all_unused_blocks()
            
            # Additional cuDF cleanup
            if 'cudf' in globals():
                try:
                    # Clear any cached data
                    cudf.core.buffer.cuda_buffer.BUFFER_POOL.clear()
                except Exception:
                    pass  # Method might not exist in all versions
            
            logger.debug("GPU memory cleanup completed")
            return True
            
        except Exception as e:
            logger.error(f"GPU memory cleanup failed: {str(e)}")
            return False
    
    def check_memory_available(self, required_bytes: int) -> bool:
        """
        Check if sufficient GPU memory is available.
        
        Args:
            required_bytes: Required memory in bytes
            
        Returns:
            True if sufficient memory is available
        """
        if not self.gpu_available:
            return False
        
        stats = self.get_memory_stats()
        if not stats:
            return False
        
        # Add safety margin (10%)
        required_with_margin = required_bytes * 1.1
        return stats.free_bytes >= required_with_margin
    
    def estimate_dataframe_memory(self, rows: int, cols: int, dtype_size: int = 8) -> int:
        """
        Estimate GPU memory required for DataFrame.
        
        Args:
            rows: Number of rows
            cols: Number of columns
            dtype_size: Average bytes per value (default 8 for float64)
            
        Returns:
            Estimated memory in bytes
        """
        # Base memory for data
        data_memory = rows * cols * dtype_size
        
        # Add overhead (index, metadata, etc.) - roughly 20%
        overhead = data_memory * 0.2
        
        return int(data_memory + overhead)
    
    @contextmanager
    def memory_context(
        self, 
        operation_name: str = "gpu_operation",
        required_memory_mb: Optional[int] = None,
        cleanup_after: bool = True
    ) -> ContextManager[Optional[MemoryStats]]:
        """
        Context manager for GPU memory-safe operations.
        
        Args:
            operation_name: Name of operation for logging
            required_memory_mb: Expected memory requirement in MB
            cleanup_after: Cleanup memory after operation
            
        Yields:
            MemoryStats before operation or None if GPU not available
            
        Raises:
            GPUMemoryError: If insufficient memory available
        """
        if not self.gpu_available:
            yield None
            return
        
        # Pre-operation checks
        initial_stats = self.get_memory_stats()
        
        if required_memory_mb and initial_stats:
            required_bytes = required_memory_mb * 1024 * 1024
            if not self.check_memory_available(required_bytes):
                # Try cleanup and check again
                self.cleanup_memory()
                if not self.check_memory_available(required_bytes):
                    raise GPUMemoryError(
                        f"Insufficient GPU memory for {operation_name}: "
                        f"need {required_memory_mb}MB, have {initial_stats.free_bytes / (1024*1024):.1f}MB"
                    )
        
        start_time = time.time()
        logger.debug(f"Starting GPU operation: {operation_name}")
        
        try:
            yield initial_stats
            
        except Exception as e:
            logger.error(f"GPU operation {operation_name} failed: {str(e)}")
            raise
            
        finally:
            # Post-operation cleanup and monitoring
            operation_time = time.time() - start_time
            
            if cleanup_after:
                self.cleanup_memory()
            
            final_stats = self.get_memory_stats()
            if initial_stats and final_stats:
                memory_delta = final_stats.used_bytes - initial_stats.used_bytes
                logger.debug(
                    f"GPU operation {operation_name} completed in {operation_time:.2f}s, "
                    f"memory delta: {memory_delta / (1024*1024):.1f}MB"
                )
    
    def get_memory_recommendations(self) -> Dict[str, Any]:
        """
        Get memory optimization recommendations.
        
        Returns:
            Dictionary with recommendations
        """
        if not self.gpu_available:
            return {"gpu_available": False}
        
        stats = self.get_memory_stats()
        if not stats:
            return {"stats_available": False}
        
        recommendations = {
            "current_utilization": stats.utilization_percent,
            "memory_limit_mb": self.memory_limit_bytes / (1024 * 1024),
            "recommendations": []
        }
        
        # High utilization
        if stats.utilization_percent > 80:
            recommendations["recommendations"].append(
                "High GPU memory utilization - consider reducing batch sizes"
            )
        
        # Low utilization
        if stats.utilization_percent < 20:
            recommendations["recommendations"].append(
                "Low GPU memory utilization - can increase batch sizes for better performance"
            )
        
        # History analysis
        if len(self._stats_history) > 10:
            recent_avg = sum(s.utilization_percent for s in self._stats_history[-10:]) / 10
            recommendations["recent_avg_utilization"] = recent_avg
            
            if recent_avg > 70:
                recommendations["recommendations"].append(
                    "Consistently high memory usage - consider memory optimization"
                )
        
        return recommendations
    
    def shutdown(self) -> None:
        """Shutdown memory manager and cleanup resources."""
        self._monitoring = False
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        
        # Final cleanup
        self.cleanup_memory(force=True)
        
        logger.info("GPU memory manager shutdown completed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


# Global memory manager instance
_gpu_memory_manager = None


def get_gpu_memory_manager() -> GPUMemoryManager:
    """Get global GPU memory manager instance."""
    global _gpu_memory_manager
    if _gpu_memory_manager is None:
        _gpu_memory_manager = GPUMemoryManager()
    return _gpu_memory_manager


def gpu_memory_context(
    operation_name: str = "gpu_operation", 
    required_memory_mb: Optional[int] = None
):
    """Convenience function for GPU memory context."""
    manager = get_gpu_memory_manager()
    return manager.memory_context(operation_name, required_memory_mb)


# Decorator for GPU memory management
def gpu_memory_managed(
    operation_name: Optional[str] = None,
    required_memory_mb: Optional[int] = None,
    cleanup_after: bool = True
):
    """
    Decorator for GPU memory-managed functions.
    
    Args:
        operation_name: Name of operation (uses function name if None)
        required_memory_mb: Expected memory requirement
        cleanup_after: Cleanup memory after operation
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            manager = get_gpu_memory_manager()
            
            with manager.memory_context(op_name, required_memory_mb, cleanup_after):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator