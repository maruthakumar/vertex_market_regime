"""
Production Parquet Data Pipeline for Component 1 Triple Rolling Straddle

High-performance Parquet file loading with 49-column schema validation,
GCS → Arrow → GPU memory mapping, and multi-expiry handling with
nearest DTE selection for rolling straddle calculations.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, date
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor

# GPU acceleration imports
try:
    import cudf
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logging.warning("GPU libraries not available. Install with: pip install cudf-cu12 cupy-cuda12x")

# Google Cloud Storage imports
try:
    from google.cloud import storage
    import gcsfs
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    logging.warning("GCS libraries not available. Install with: pip install google-cloud-storage gcsfs")


@dataclass
class ParquetLoadResult:
    """Result from Parquet file loading"""
    data: Union[pd.DataFrame, 'cudf.DataFrame']
    file_path: str
    row_count: int
    column_count: int
    processing_time_ms: float
    memory_usage_mb: float
    expiry_date: str
    dte_range: Tuple[int, int]
    timestamp_range: Tuple[str, str]
    schema_valid: bool


@dataclass
class SchemaValidationResult:
    """Schema validation result"""
    is_valid: bool
    expected_columns: int
    actual_columns: int
    missing_columns: List[str]
    extra_columns: List[str]
    column_type_mismatches: Dict[str, Tuple[str, str]]


class ProductionParquetLoader:
    """
    Production Parquet Data Pipeline with GPU acceleration and GCS integration
    
    Key Features:
    - 49-column schema validation
    - GCS → Arrow → GPU memory mapping
    - Multi-expiry handling with nearest DTE selection
    - Performance optimized (<150ms per file)
    - Memory efficient (<512MB per component)
    """
    
    # Expected 49-column production schema
    EXPECTED_SCHEMA = [
        'trade_date', 'trade_time', 'expiry_date', 'index_name', 'spot', 'atm_strike', 
        'strike', 'dte', 'expiry_bucket', 'zone_id', 'zone_name', 'call_strike_type', 
        'put_strike_type', 'ce_symbol', 'ce_open', 'ce_high', 'ce_low', 'ce_close', 
        'ce_volume', 'ce_oi', 'ce_coi', 'ce_iv', 'ce_delta', 'ce_gamma', 'ce_theta', 
        'ce_vega', 'ce_rho', 'pe_symbol', 'pe_open', 'pe_high', 'pe_low', 'pe_close', 
        'pe_volume', 'pe_oi', 'pe_coi', 'pe_iv', 'pe_delta', 'pe_gamma', 'pe_theta', 
        'pe_vega', 'pe_rho', 'future_open', 'future_high', 'future_low', 'future_close', 
        'future_volume', 'future_oi', 'future_coi', 'dte_bucket'
    ]
    
    # Key columns for rolling straddle calculation
    STRADDLE_COLUMNS = [
        'trade_time', 'call_strike_type', 'put_strike_type', 'ce_close', 'pe_close',
        'ce_volume', 'pe_volume', 'dte', 'expiry_date', 'spot', 'atm_strike'
    ]
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Production Parquet Loader
        
        Args:
            config: Configuration dictionary with loader parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Performance configuration
        self.processing_budget_ms = config.get('processing_budget_ms', 150)
        self.memory_budget_mb = config.get('memory_budget_mb', 512)
        self.use_gpu = config.get('use_gpu', GPU_AVAILABLE)
        
        # Data source configuration
        self.data_root = config.get('data_root', '/Users/maruth/projects/market_regime/data/nifty_validation/backtester_processed/')
        self.use_gcs = config.get('use_gcs', False)
        self.gcs_bucket = config.get('gcs_bucket', 'vertex-mr-data')
        
        # Multi-expiry handling
        self.nearest_dte_only = config.get('nearest_dte_only', True)
        self.max_dte = config.get('max_dte', 30)
        
        # Threading configuration
        self.max_workers = config.get('max_workers', 4)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Initialize storage client if using GCS
        if self.use_gcs and GCS_AVAILABLE:
            try:
                self.storage_client = storage.Client()
                self.gcs_filesystem = gcsfs.GCSFileSystem()
                self.logger.info("GCS client initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize GCS client: {e}")
                self.use_gcs = False
        else:
            self.storage_client = None
            self.gcs_filesystem = None
            
        self.logger.info(f"ParquetLoader initialized: GPU={self.use_gpu}, GCS={self.use_gcs}")
        
    async def load_parquet_file(self, file_path: str) -> ParquetLoadResult:
        """
        Load single Parquet file with performance optimization
        
        Args:
            file_path: Path to Parquet file (local or GCS)
            
        Returns:
            ParquetLoadResult with loaded data and metadata
        """
        start_time = time.time()
        
        try:
            # Load data using appropriate method
            if self.use_gcs and file_path.startswith('gs://'):
                data = await self._load_from_gcs(file_path)
            else:
                data = await self._load_from_local(file_path)
            
            # Schema validation
            schema_result = self._validate_schema(data)
            if not schema_result.is_valid:
                self.logger.warning(f"Schema validation failed for {file_path}: {schema_result}")
            
            # Extract metadata
            row_count = len(data)
            column_count = len(data.columns)
            
            # Get expiry and DTE information
            expiry_date = str(data['expiry_date'].iloc[0])
            dte_range = (int(data['dte'].min()), int(data['dte'].max()))
            timestamp_range = (str(data['trade_time'].min()), str(data['trade_time'].max()))
            
            # Calculate processing time and memory usage
            processing_time = (time.time() - start_time) * 1000
            memory_usage = self._calculate_memory_usage(data)
            
            # Performance validation
            if processing_time > self.processing_budget_ms:
                self.logger.warning(f"Processing time {processing_time:.2f}ms exceeded budget {self.processing_budget_ms}ms")
            
            if memory_usage > self.memory_budget_mb:
                self.logger.warning(f"Memory usage {memory_usage:.2f}MB exceeded budget {self.memory_budget_mb}MB")
            
            return ParquetLoadResult(
                data=data,
                file_path=file_path,
                row_count=row_count,
                column_count=column_count,
                processing_time_ms=processing_time,
                memory_usage_mb=memory_usage,
                expiry_date=expiry_date,
                dte_range=dte_range,
                timestamp_range=timestamp_range,
                schema_valid=schema_result.is_valid
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.logger.error(f"Failed to load {file_path}: {e}")
            raise
    
    async def _load_from_local(self, file_path: str) -> Union[pd.DataFrame, 'cudf.DataFrame']:
        """Load Parquet file from local filesystem"""
        loop = asyncio.get_event_loop()
        
        def _load_sync():
            # Read using PyArrow for memory efficiency
            table = pq.read_table(file_path)
            
            # Convert to GPU DataFrame if available and configured
            if self.use_gpu and GPU_AVAILABLE:
                try:
                    df = cudf.DataFrame.from_arrow(table)
                    self.logger.debug(f"Loaded {file_path} to GPU memory")
                    return df
                except Exception as e:
                    self.logger.warning(f"GPU loading failed, falling back to CPU: {e}")
            
            # Fallback to pandas DataFrame
            df = table.to_pandas()
            return df
        
        return await loop.run_in_executor(self.executor, _load_sync)
    
    async def _load_from_gcs(self, gcs_path: str) -> Union[pd.DataFrame, 'cudf.DataFrame']:
        """Load Parquet file from Google Cloud Storage"""
        if not self.use_gcs or not GCS_AVAILABLE:
            raise ValueError("GCS not configured or available")
        
        loop = asyncio.get_event_loop()
        
        def _load_gcs_sync():
            # Read using gcsfs and PyArrow
            table = pq.read_table(gcs_path, filesystem=self.gcs_filesystem)
            
            # Convert to GPU DataFrame if available
            if self.use_gpu and GPU_AVAILABLE:
                try:
                    df = cudf.DataFrame.from_arrow(table)
                    self.logger.debug(f"Loaded {gcs_path} from GCS to GPU")
                    return df
                except Exception as e:
                    self.logger.warning(f"GPU loading failed, falling back to CPU: {e}")
            
            # Fallback to pandas
            df = table.to_pandas()
            return df
        
        return await loop.run_in_executor(self.executor, _load_gcs_sync)
    
    def _validate_schema(self, data: Union[pd.DataFrame, 'cudf.DataFrame']) -> SchemaValidationResult:
        """Validate 49-column production schema"""
        actual_columns = list(data.columns)
        expected_columns = self.EXPECTED_SCHEMA
        
        # Check column count
        is_valid = len(actual_columns) == len(expected_columns)
        
        # Find missing and extra columns
        missing_columns = [col for col in expected_columns if col not in actual_columns]
        extra_columns = [col for col in actual_columns if col not in expected_columns]
        
        # Update validity
        is_valid = is_valid and not missing_columns and not extra_columns
        
        # Check column types (basic validation)
        column_type_mismatches = {}
        # Note: Detailed type validation could be added here
        
        return SchemaValidationResult(
            is_valid=is_valid,
            expected_columns=len(expected_columns),
            actual_columns=len(actual_columns),
            missing_columns=missing_columns,
            extra_columns=extra_columns,
            column_type_mismatches=column_type_mismatches
        )
    
    def _calculate_memory_usage(self, data: Union[pd.DataFrame, 'cudf.DataFrame']) -> float:
        """Calculate DataFrame memory usage in MB"""
        try:
            if self.use_gpu and hasattr(data, 'memory_usage'):
                # cuDF memory usage
                return data.memory_usage(deep=True).sum() / 1024 / 1024
            else:
                # Pandas memory usage
                return data.memory_usage(deep=True).sum() / 1024 / 1024
        except:
            # Fallback estimation
            return len(data) * len(data.columns) * 8 / 1024 / 1024  # 8 bytes per float64
    
    async def load_multi_expiry_data(self, 
                                   date_filter: Optional[str] = None,
                                   expiry_filter: Optional[List[str]] = None) -> Dict[str, ParquetLoadResult]:
        """
        Load multiple expiry files with nearest DTE selection
        
        Args:
            date_filter: Optional date filter (e.g., '2024-01-01')
            expiry_filter: Optional expiry list (e.g., ['04012024', '11012024'])
            
        Returns:
            Dictionary mapping expiry to ParquetLoadResult
        """
        start_time = time.time()
        
        # Discover available files
        available_files = self._discover_parquet_files(date_filter, expiry_filter)
        
        if not available_files:
            raise ValueError("No Parquet files found matching criteria")
        
        self.logger.info(f"Loading {len(available_files)} Parquet files")
        
        # Load files concurrently
        load_tasks = []
        for file_path in available_files[:10]:  # Limit for initial implementation
            task = self.load_parquet_file(file_path)
            load_tasks.append(task)
        
        # Execute loads
        results = await asyncio.gather(*load_tasks, return_exceptions=True)
        
        # Process results
        expiry_data = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to load file {available_files[i]}: {result}")
                continue
            
            expiry_data[result.expiry_date] = result
        
        total_time = (time.time() - start_time) * 1000
        self.logger.info(f"Loaded {len(expiry_data)} expiry datasets in {total_time:.2f}ms")
        
        return expiry_data
    
    def _discover_parquet_files(self, 
                               date_filter: Optional[str] = None,
                               expiry_filter: Optional[List[str]] = None) -> List[str]:
        """Discover available Parquet files"""
        files = []
        
        if self.use_gcs:
            # GCS file discovery
            # Implementation would go here for production
            pass
        else:
            # Local file discovery
            data_path = Path(self.data_root)
            if not data_path.exists():
                self.logger.error(f"Data root path does not exist: {data_path}")
                return []
            
            # Scan expiry directories
            for expiry_dir in data_path.iterdir():
                if expiry_dir.is_dir() and expiry_dir.name.startswith('expiry='):
                    expiry = expiry_dir.name.replace('expiry=', '')
                    
                    # Apply expiry filter
                    if expiry_filter and expiry not in expiry_filter:
                        continue
                    
                    # Scan Parquet files in expiry directory
                    for parquet_file in expiry_dir.glob('*.parquet'):
                        # Apply date filter if specified
                        if date_filter and date_filter not in parquet_file.name:
                            continue
                        
                        files.append(str(parquet_file))
        
        return sorted(files)
    
    async def get_nearest_dte_data(self, 
                                 multi_expiry_data: Dict[str, ParquetLoadResult],
                                 target_timestamp: Optional[str] = None) -> ParquetLoadResult:
        """
        Select nearest DTE data for rolling straddle calculation
        
        Args:
            multi_expiry_data: Dictionary of expiry data
            target_timestamp: Optional target timestamp
            
        Returns:
            ParquetLoadResult with nearest DTE data
        """
        if not multi_expiry_data:
            raise ValueError("No expiry data provided")
        
        # Find nearest DTE expiry
        min_dte = float('inf')
        nearest_expiry = None
        
        for expiry, data in multi_expiry_data.items():
            avg_dte = (data.dte_range[0] + data.dte_range[1]) / 2
            if avg_dte < min_dte and avg_dte <= self.max_dte:
                min_dte = avg_dte
                nearest_expiry = expiry
        
        if nearest_expiry is None:
            # Fallback to first available
            nearest_expiry = next(iter(multi_expiry_data.keys()))
            self.logger.warning(f"No expiry within max_dte={self.max_dte}, using {nearest_expiry}")
        
        self.logger.info(f"Selected nearest DTE expiry: {nearest_expiry} (DTE={min_dte:.1f})")
        return multi_expiry_data[nearest_expiry]
    
    async def filter_for_rolling_straddle(self, data: ParquetLoadResult) -> Union[pd.DataFrame, 'cudf.DataFrame']:
        """
        Filter data for rolling straddle calculation with required columns
        
        Args:
            data: ParquetLoadResult from load operation
            
        Returns:
            Filtered DataFrame with straddle-specific columns
        """
        # Filter to required columns
        try:
            filtered_data = data.data[self.STRADDLE_COLUMNS].copy()
            
            # Sort by trade_time for time-series processing
            filtered_data = filtered_data.sort_values('trade_time')
            
            # Filter to relevant strike types (ATM, ITM1, OTM1)
            straddle_strikes = ['ATM', 'ITM1', 'OTM1']
            straddle_mask = (
                filtered_data['call_strike_type'].isin(straddle_strikes) | 
                filtered_data['put_strike_type'].isin(straddle_strikes)
            )
            filtered_data = filtered_data[straddle_mask]
            
            self.logger.info(f"Filtered to {len(filtered_data)} rows for rolling straddle calculation")
            return filtered_data
            
        except Exception as e:
            self.logger.error(f"Failed to filter data for rolling straddle: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        
        self.logger.info("ParquetLoader cleanup completed")


# Factory function for easy instantiation
def create_parquet_loader(config: Dict[str, Any]) -> ProductionParquetLoader:
    """Create and configure ProductionParquetLoader instance"""
    return ProductionParquetLoader(config)