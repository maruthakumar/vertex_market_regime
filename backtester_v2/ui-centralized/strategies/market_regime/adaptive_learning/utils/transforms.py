"""
Transform Utilities for Arrow/RAPIDS Integration

Provides efficient conversion helpers and aggregation functions for the
Parquet → Arrow → GPU pipeline, with fallback to CPU processing.
"""

import time
import logging
import warnings
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd

from .. import GPUMemoryError, validate_performance_budget

logger = logging.getLogger(__name__)

# Try to import RAPIDS components
try:
    import cudf
    import cupy as cp
    GPU_AVAILABLE = True
    logger.info("GPU acceleration available via RAPIDS cuDF")
except ImportError:
    GPU_AVAILABLE = False
    logger.info("GPU acceleration not available, using CPU fallback")

# Try to import Apache Arrow
try:
    import pyarrow as pa
    import pyarrow.compute as pc
    ARROW_AVAILABLE = True
    logger.info("Apache Arrow available for zero-copy operations")
except ImportError:
    ARROW_AVAILABLE = False
    logger.warning("Apache Arrow not available")


class ArrowToCuDFConverter:
    """
    Efficient converter for Arrow → cuDF with memory management and fallback.
    
    Provides zero-copy conversion where possible and handles memory constraints
    for the 8-component system with <3.7GB total memory budget.
    """
    
    def __init__(self, enable_gpu: bool = True, chunk_size: Optional[int] = None):
        """
        Initialize converter.
        
        Args:
            enable_gpu: Enable GPU acceleration if available
            chunk_size: Maximum chunk size for processing (auto-detect if None)
        """
        self.enable_gpu = enable_gpu and GPU_AVAILABLE
        self.chunk_size = chunk_size or self._get_optimal_chunk_size()
        self._memory_pressure = False
        
        logger.info(f"ArrowToCuDFConverter initialized: GPU={self.enable_gpu}, chunk_size={self.chunk_size}")
    
    def _get_optimal_chunk_size(self) -> int:
        """Calculate optimal chunk size based on available memory."""
        # Conservative chunk size for 3.7GB constraint
        # Assume each row uses ~1KB on average, reserve 1GB for other components
        available_memory_gb = 2.7  # Reserve space for other components
        optimal_rows = int(available_memory_gb * 1024 * 1024)  # Conservative estimate
        return min(optimal_rows, 1_000_000)  # Cap at 1M rows per chunk
    
    def convert_arrow_to_cudf(
        self, 
        arrow_table: 'pa.Table', 
        memory_budget_mb: int = 512
    ) -> Union['cudf.DataFrame', pd.DataFrame]:
        """
        Convert Arrow table to cuDF DataFrame with memory management.
        
        Args:
            arrow_table: PyArrow table to convert
            memory_budget_mb: Memory budget for conversion
            
        Returns:
            cuDF DataFrame if GPU available, pandas DataFrame otherwise
            
        Raises:
            GPUMemoryError: If GPU memory allocation fails
        """
        start_time = time.time()
        
        if not ARROW_AVAILABLE:
            raise ValueError("Apache Arrow not available")
        
        try:
            # Check table size
            table_rows = len(arrow_table)
            logger.debug(f"Converting Arrow table with {table_rows} rows")
            
            if self.enable_gpu and table_rows > 0:
                # Try GPU conversion
                try:
                    if table_rows <= self.chunk_size:
                        # Small table - direct conversion
                        cudf_df = cudf.from_arrow(arrow_table)
                        logger.debug(f"Direct Arrow→cuDF conversion: {table_rows} rows")
                        return cudf_df
                    else:
                        # Large table - chunked conversion
                        return self._chunked_arrow_to_cudf(arrow_table, memory_budget_mb)
                
                except Exception as e:
                    logger.warning(f"GPU conversion failed, falling back to CPU: {str(e)}")
                    self._memory_pressure = True
            
            # Fallback to pandas
            pandas_df = arrow_table.to_pandas()
            logger.debug(f"Arrow→pandas fallback conversion: {table_rows} rows")
            
            # Validate conversion time
            validate_performance_budget(start_time, 100, "arrow_to_dataframe_conversion")
            
            return pandas_df
            
        except Exception as e:
            logger.error(f"Arrow conversion failed: {str(e)}")
            raise GPUMemoryError(f"Failed to convert Arrow table: {str(e)}")
    
    def _chunked_arrow_to_cudf(
        self, 
        arrow_table: 'pa.Table', 
        memory_budget_mb: int
    ) -> 'cudf.DataFrame':
        """Convert large Arrow table using chunked processing."""
        total_rows = len(arrow_table)
        chunks = []
        
        logger.info(f"Chunked conversion for {total_rows} rows with budget {memory_budget_mb}MB")
        
        for start_idx in range(0, total_rows, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, total_rows)
            chunk_table = arrow_table.slice(start_idx, end_idx - start_idx)
            
            try:
                chunk_cudf = cudf.from_arrow(chunk_table)
                chunks.append(chunk_cudf)
                
                # Monitor GPU memory
                if hasattr(cp, 'get_default_memory_pool'):
                    pool = cp.get_default_memory_pool()
                    used_bytes = pool.used_bytes()
                    if used_bytes > memory_budget_mb * 1024 * 1024:
                        logger.warning(f"GPU memory pressure: {used_bytes / (1024*1024):.1f}MB")
                        # Force cleanup
                        cp.get_default_memory_pool().free_all_blocks()
                
            except Exception as e:
                logger.error(f"Chunk conversion failed at rows {start_idx}-{end_idx}: {str(e)}")
                raise GPUMemoryError(f"Chunked conversion failed: {str(e)}")
        
        # Concatenate chunks
        if chunks:
            result_df = cudf.concat(chunks, ignore_index=True)
            logger.info(f"Successfully concatenated {len(chunks)} chunks")
            return result_df
        else:
            raise GPUMemoryError("No chunks successfully processed")
    
    def convert_pandas_to_cudf(
        self, 
        pandas_df: pd.DataFrame, 
        memory_budget_mb: int = 512
    ) -> Union['cudf.DataFrame', pd.DataFrame]:
        """
        Convert pandas DataFrame to cuDF with memory management.
        
        Args:
            pandas_df: Pandas DataFrame to convert
            memory_budget_mb: Memory budget for conversion
            
        Returns:
            cuDF DataFrame if successful, original pandas DataFrame on fallback
        """
        if not self.enable_gpu or pandas_df.empty:
            return pandas_df
        
        start_time = time.time()
        
        try:
            # Check memory requirements
            memory_usage_mb = pandas_df.memory_usage(deep=True).sum() / (1024 * 1024)
            
            if memory_usage_mb > memory_budget_mb:
                logger.warning(
                    f"DataFrame too large for GPU: {memory_usage_mb:.1f}MB > {memory_budget_mb}MB"
                )
                return pandas_df
            
            # Convert to cuDF
            cudf_df = cudf.from_pandas(pandas_df)
            
            validate_performance_budget(start_time, 50, "pandas_to_cudf_conversion")
            
            logger.debug(f"Pandas→cuDF conversion successful: {len(pandas_df)} rows")
            return cudf_df
            
        except Exception as e:
            logger.warning(f"Pandas→cuDF conversion failed: {str(e)}")
            return pandas_df
    
    def ensure_dataframe_type(
        self, 
        df: Union[pd.DataFrame, 'cudf.DataFrame'], 
        prefer_gpu: bool = None
    ) -> Union[pd.DataFrame, 'cudf.DataFrame']:
        """
        Ensure DataFrame is in preferred format (GPU or CPU).
        
        Args:
            df: Input DataFrame
            prefer_gpu: Prefer GPU format if available
            
        Returns:
            DataFrame in preferred format
        """
        if prefer_gpu is None:
            prefer_gpu = self.enable_gpu
        
        # Check current type
        is_cudf = GPU_AVAILABLE and isinstance(df, cudf.DataFrame)
        is_pandas = isinstance(df, pd.DataFrame)
        
        if prefer_gpu and not is_cudf and is_pandas:
            # Convert pandas to cuDF
            return self.convert_pandas_to_cudf(df)
        elif not prefer_gpu and is_cudf:
            # Convert cuDF to pandas
            return df.to_pandas()
        else:
            # Already in correct format
            return df
    
    def cleanup_gpu_memory(self) -> None:
        """Force cleanup of GPU memory."""
        if GPU_AVAILABLE:
            try:
                import gc
                gc.collect()
                if hasattr(cp, 'get_default_memory_pool'):
                    cp.get_default_memory_pool().free_all_blocks()
                logger.debug("GPU memory cleanup completed")
            except Exception as e:
                logger.warning(f"GPU memory cleanup failed: {str(e)}")


class MultiTimeframeAggregator:
    """
    Multi-timeframe aggregation functions for component analysis.
    
    Supports the standard timeframes used across components:
    - 3min, 5min, 10min, 15min analysis
    - Daily, weekly, monthly aggregations
    - Rolling window calculations
    """
    
    STANDARD_TIMEFRAMES = {
        '3min': '3T',
        '5min': '5T', 
        '10min': '10T',
        '15min': '15T',
        '30min': '30T',
        '1hour': '1H',
        '1day': '1D'
    }
    
    def __init__(self, converter: Optional[ArrowToCuDFConverter] = None):
        """
        Initialize aggregator.
        
        Args:
            converter: Optional converter for GPU acceleration
        """
        self.converter = converter or ArrowToCuDFConverter()
        self.gpu_available = self.converter.enable_gpu
    
    def aggregate_timeframes(
        self,
        df: Union[pd.DataFrame, 'cudf.DataFrame'],
        timestamp_col: str = 'timestamp',
        value_cols: List[str] = None,
        timeframes: List[str] = None,
        agg_functions: List[str] = None
    ) -> Dict[str, Union[pd.DataFrame, 'cudf.DataFrame']]:
        """
        Aggregate data across multiple timeframes.
        
        Args:
            df: Input DataFrame with timestamp column
            timestamp_col: Name of timestamp column
            value_cols: Columns to aggregate (all numeric if None)
            timeframes: Timeframes to aggregate ('5min', '15min', etc.)
            agg_functions: Aggregation functions ('mean', 'std', 'min', 'max')
            
        Returns:
            Dictionary of aggregated DataFrames by timeframe
        """
        start_time = time.time()
        
        if timeframes is None:
            timeframes = ['5min', '15min']
        
        if agg_functions is None:
            agg_functions = ['mean', 'std', 'min', 'max']
        
        if value_cols is None:
            # Auto-detect numeric columns
            if self.gpu_available and isinstance(df, cudf.DataFrame):
                value_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            else:
                value_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        results = {}
        
        # Ensure timestamp column is datetime
        df_work = df.copy()
        if self.gpu_available and isinstance(df_work, cudf.DataFrame):
            df_work[timestamp_col] = cudf.to_datetime(df_work[timestamp_col])
        else:
            df_work[timestamp_col] = pd.to_datetime(df_work[timestamp_col])
        
        # Set timestamp as index for resampling
        df_work = df_work.set_index(timestamp_col)
        
        for timeframe in timeframes:
            if timeframe not in self.STANDARD_TIMEFRAMES:
                logger.warning(f"Unknown timeframe: {timeframe}")
                continue
            
            resample_rule = self.STANDARD_TIMEFRAMES[timeframe]
            
            try:
                # Perform resampling and aggregation
                resampled = df_work[value_cols].resample(resample_rule)
                
                # Apply aggregation functions
                agg_result = {}
                for func in agg_functions:
                    if hasattr(resampled, func):
                        agg_data = getattr(resampled, func)()
                        # Rename columns to include aggregation function
                        agg_data.columns = [f"{col}_{func}" for col in agg_data.columns]
                        agg_result.update(agg_data.to_dict())
                
                # Convert back to DataFrame
                if self.gpu_available and isinstance(df, cudf.DataFrame):
                    timeframe_df = cudf.DataFrame(agg_result)
                else:
                    timeframe_df = pd.DataFrame(agg_result)
                
                results[timeframe] = timeframe_df
                
                logger.debug(f"Aggregated {timeframe}: {len(timeframe_df)} periods")
                
            except Exception as e:
                logger.error(f"Aggregation failed for {timeframe}: {str(e)}")
                continue
        
        validate_performance_budget(start_time, 200, "multi_timeframe_aggregation")
        
        return results
    
    def rolling_aggregation(
        self,
        df: Union[pd.DataFrame, 'cudf.DataFrame'],
        window_sizes: List[int] = None,
        value_cols: List[str] = None
    ) -> Union[pd.DataFrame, 'cudf.DataFrame']:
        """
        Apply rolling window aggregations.
        
        Args:
            df: Input DataFrame
            window_sizes: Rolling window sizes (default: [5, 20, 50])
            value_cols: Columns to process
            
        Returns:
            DataFrame with rolling aggregation columns added
        """
        if window_sizes is None:
            window_sizes = [5, 20, 50]
        
        if value_cols is None:
            # Auto-detect numeric columns
            if self.gpu_available and isinstance(df, cudf.DataFrame):
                value_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            else:
                value_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        df_result = df.copy()
        
        for col in value_cols:
            for window in window_sizes:
                # Rolling mean
                df_result[f"{col}_rolling_{window}_mean"] = df[col].rolling(window).mean()
                # Rolling std
                df_result[f"{col}_rolling_{window}_std"] = df[col].rolling(window).std()
        
        return df_result


def get_optimal_chunk_size(available_memory_gb: float = 2.0) -> int:
    """
    Calculate optimal chunk size for data processing.
    
    Args:
        available_memory_gb: Available memory in GB
        
    Returns:
        Optimal chunk size in rows
    """
    # Conservative estimate: 1KB per row average
    bytes_per_row = 1024
    available_bytes = available_memory_gb * 1024 * 1024 * 1024
    optimal_rows = int(available_bytes / bytes_per_row)
    
    # Cap at reasonable limits
    return min(max(optimal_rows, 10_000), 2_000_000)


# Convenience function for quick conversion
def quick_arrow_to_gpu(arrow_table: 'pa.Table') -> Union['cudf.DataFrame', pd.DataFrame]:
    """Quick conversion from Arrow to GPU DataFrame."""
    converter = ArrowToCuDFConverter()
    return converter.convert_arrow_to_cudf(arrow_table)