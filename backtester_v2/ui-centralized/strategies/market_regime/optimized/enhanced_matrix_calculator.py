"""
Enhanced Matrix Calculator with Performance Optimizations

Provides optimized calculations for 10×10 correlation and resistance matrices
using vectorization, numba JIT compilation, sparse matrices, and GPU acceleration.

Key Optimizations:
1. Vectorized numpy operations
2. Numba JIT compilation for hot paths
3. Sparse matrix support where applicable
4. Incremental correlation updates
5. Memory pooling and reuse
6. Optional GPU acceleration with CuPy

Author: Market Regime System Optimizer
Date: 2025-07-07
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from numba import jit, prange, float64, int32
from scipy import sparse
import time
from collections import deque

# Optional GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MatrixConfig:
    """Configuration for matrix calculations"""
    use_gpu: bool = False
    use_sparse: bool = True
    use_incremental: bool = True
    cache_size: int = 1000
    num_threads: int = 4
    precision: str = 'float32'  # float32 or float64


class MemoryPool:
    """Memory pool for matrix allocations"""
    
    def __init__(self, max_matrices: int = 100, matrix_size: int = 10):
        self.max_matrices = max_matrices
        self.matrix_size = matrix_size
        self.pool = []
        self.in_use = set()
        
        # Pre-allocate matrices
        for i in range(max_matrices):
            matrix = np.zeros((matrix_size, matrix_size), dtype=np.float32)
            self.pool.append(matrix)
            
    def get_matrix(self):
        """Get a matrix from the pool"""
        for i, matrix in enumerate(self.pool):
            if i not in self.in_use:
                self.in_use.add(i)
                return matrix, i
        # Pool exhausted, allocate new
        return np.zeros((self.matrix_size, self.matrix_size), dtype=np.float32), -1
        
    def return_matrix(self, idx: int):
        """Return a matrix to the pool"""
        if idx >= 0:
            self.in_use.discard(idx)


@jit(nopython=True, parallel=True, cache=True)
def fast_correlation_matrix(data: np.ndarray) -> np.ndarray:
    """
    Fast correlation matrix calculation using numba
    
    Args:
        data: 2D array of shape (n_samples, n_features)
        
    Returns:
        Correlation matrix of shape (n_features, n_features)
    """
    n_samples, n_features = data.shape
    
    # Compute means
    means = np.zeros(n_features)
    for j in prange(n_features):
        means[j] = np.mean(data[:, j])
    
    # Center the data
    centered = np.zeros_like(data)
    for i in prange(n_samples):
        for j in range(n_features):
            centered[i, j] = data[i, j] - means[j]
    
    # Compute covariance matrix
    cov = np.zeros((n_features, n_features))
    for i in prange(n_features):
        for j in range(i, n_features):
            dot_product = 0.0
            for k in range(n_samples):
                dot_product += centered[k, i] * centered[k, j]
            cov[i, j] = dot_product / (n_samples - 1)
            cov[j, i] = cov[i, j]
    
    # Compute standard deviations
    stds = np.zeros(n_features)
    for j in prange(n_features):
        stds[j] = np.sqrt(cov[j, j])
    
    # Compute correlation matrix
    corr = np.zeros((n_features, n_features))
    for i in prange(n_features):
        for j in range(n_features):
            if stds[i] > 0 and stds[j] > 0:
                corr[i, j] = cov[i, j] / (stds[i] * stds[j])
            else:
                corr[i, j] = 0.0 if i != j else 1.0
                
    return corr


@jit(nopython=True, cache=True)
def incremental_correlation_update(
    prev_corr: np.ndarray,
    prev_means: np.ndarray,
    prev_vars: np.ndarray,
    prev_n: int,
    new_data: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Incrementally update correlation matrix with new data
    
    Uses Welford's online algorithm for numerical stability
    """
    n_features = prev_corr.shape[0]
    n_new = new_data.shape[0]
    n_total = prev_n + n_new
    
    # Update means
    new_means = np.zeros(n_features)
    for j in range(n_features):
        sum_new = 0.0
        for i in range(n_new):
            sum_new += new_data[i, j]
        new_means[j] = (prev_means[j] * prev_n + sum_new) / n_total
    
    # Update variances
    new_vars = np.zeros(n_features)
    for j in range(n_features):
        sum_sq = prev_vars[j] * (prev_n - 1)
        for i in range(n_new):
            delta = new_data[i, j] - new_means[j]
            sum_sq += delta * delta
        new_vars[j] = sum_sq / (n_total - 1)
    
    # Update correlations
    new_corr = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(i, n_features):
            if i == j:
                new_corr[i, j] = 1.0
            else:
                # Approximate incremental correlation
                cov_update = 0.0
                for k in range(n_new):
                    cov_update += (new_data[k, i] - new_means[i]) * \
                                 (new_data[k, j] - new_means[j])
                
                prev_cov = prev_corr[i, j] * np.sqrt(prev_vars[i] * prev_vars[j])
                new_cov = (prev_cov * (prev_n - 1) + cov_update) / (n_total - 1)
                
                std_i = np.sqrt(new_vars[i])
                std_j = np.sqrt(new_vars[j])
                
                if std_i > 0 and std_j > 0:
                    new_corr[i, j] = new_cov / (std_i * std_j)
                    new_corr[j, i] = new_corr[i, j]
                    
    return new_corr, new_means, new_vars


class Enhanced10x10MatrixCalculator:
    """
    Enhanced calculator for 10×10 correlation and resistance matrices
    
    Provides multiple optimization strategies:
    - CPU vectorization with numpy
    - JIT compilation with numba
    - GPU acceleration with CuPy
    - Sparse matrix operations
    - Incremental updates
    """
    
    def __init__(self, config: Optional[MatrixConfig] = None):
        """Initialize enhanced matrix calculator"""
        self.config = config or MatrixConfig()
        self.memory_pool = MemoryPool(max_matrices=100, matrix_size=10)
        
        # Component names for 10×10 matrix
        self.components = [
            'ATM_CE', 'ATM_PE', 'ITM1_CE', 'ITM1_PE', 'OTM1_CE', 'OTM1_PE',
            'ATM_STRADDLE', 'ITM1_STRADDLE', 'OTM1_STRADDLE', 'COMBINED_TRIPLE'
        ]
        
        # Cache for incremental updates
        self.correlation_cache = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'updates': 0
        }
        
        # GPU setup if available
        if self.config.use_gpu and GPU_AVAILABLE:
            self.gpu_memory_pool = cp.get_default_memory_pool()
            logger.info("GPU acceleration enabled for matrix calculations")
        else:
            self.config.use_gpu = False
            if self.config.use_gpu:
                logger.warning("GPU requested but CuPy not available")
                
        logger.info(f"Enhanced 10×10 Matrix Calculator initialized")
        logger.info(f"Config: GPU={self.config.use_gpu}, Sparse={self.config.use_sparse}, "
                   f"Incremental={self.config.use_incremental}")
        
    def calculate_correlation_matrix(self, data: pd.DataFrame, 
                                   method: str = 'auto') -> np.ndarray:
        """
        Calculate 10×10 correlation matrix with automatic optimization
        
        Args:
            data: DataFrame with columns for each component
            method: 'auto', 'numpy', 'numba', 'gpu', 'sparse'
            
        Returns:
            10×10 correlation matrix
        """
        start_time = time.time()
        
        # Convert to numpy array
        if isinstance(data, pd.DataFrame):
            data_array = data[self.components].values.astype(
                np.float64 if self.config.precision == 'float64' else np.float32
            )
        else:
            data_array = data
            
        # Choose method
        if method == 'auto':
            if self.config.use_gpu and GPU_AVAILABLE:
                method = 'gpu'
            elif data_array.shape[0] > 1000:
                method = 'numba'
            else:
                method = 'numpy'
                
        # Calculate correlation
        if method == 'gpu':
            corr_matrix = self._calculate_correlation_gpu(data_array)
        elif method == 'numba':
            corr_matrix = fast_correlation_matrix(data_array)
        elif method == 'sparse':
            corr_matrix = self._calculate_correlation_sparse(data_array)
        else:
            corr_matrix = self._calculate_correlation_numpy(data_array)
            
        calc_time = time.time() - start_time
        logger.debug(f"Correlation matrix calculated in {calc_time:.3f}s using {method}")
        
        return corr_matrix
        
    def _calculate_correlation_numpy(self, data: np.ndarray) -> np.ndarray:
        """Standard numpy correlation calculation"""
        # Get matrix from pool
        corr_matrix, pool_idx = self.memory_pool.get_matrix()
        
        try:
            # Vectorized correlation
            data_centered = data - data.mean(axis=0)
            cov = np.dot(data_centered.T, data_centered) / (data.shape[0] - 1)
            stds = np.sqrt(np.diag(cov))
            
            # Avoid division by zero
            stds[stds == 0] = 1.0
            
            # Calculate correlation
            corr_matrix[:] = cov / np.outer(stds, stds)
            np.fill_diagonal(corr_matrix, 1.0)
            
            return corr_matrix.copy()
            
        finally:
            self.memory_pool.return_matrix(pool_idx)
            
    def _calculate_correlation_gpu(self, data: np.ndarray) -> np.ndarray:
        """GPU-accelerated correlation calculation"""
        if not GPU_AVAILABLE:
            return self._calculate_correlation_numpy(data)
            
        # Transfer to GPU
        data_gpu = cp.asarray(data)
        
        # Calculate on GPU
        data_centered = data_gpu - cp.mean(data_gpu, axis=0)
        cov = cp.dot(data_centered.T, data_centered) / (data.shape[0] - 1)
        stds = cp.sqrt(cp.diag(cov))
        stds[stds == 0] = 1.0
        
        corr_gpu = cov / cp.outer(stds, stds)
        cp.fill_diagonal(corr_gpu, 1.0)
        
        # Transfer back to CPU
        return cp.asnumpy(corr_gpu)
        
    def _calculate_correlation_sparse(self, data: np.ndarray) -> np.ndarray:
        """Sparse matrix correlation for efficiency"""
        # Check sparsity
        sparsity = np.sum(data == 0) / data.size
        
        if sparsity > 0.5:
            # Use sparse operations
            data_sparse = sparse.csr_matrix(data)
            data_centered = data_sparse - data_sparse.mean(axis=0)
            
            # Sparse covariance
            cov = (data_centered.T @ data_centered) / (data.shape[0] - 1)
            cov_dense = cov.toarray()
            
            stds = np.sqrt(np.diag(cov_dense))
            stds[stds == 0] = 1.0
            
            return cov_dense / np.outer(stds, stds)
        else:
            # Fall back to dense
            return self._calculate_correlation_numpy(data)
            
    def calculate_incremental_correlation(self, new_data: pd.DataFrame,
                                        cache_key: str) -> np.ndarray:
        """
        Calculate correlation with incremental updates
        
        Args:
            new_data: New data to incorporate
            cache_key: Key for caching previous state
            
        Returns:
            Updated correlation matrix
        """
        new_array = new_data[self.components].values
        
        if cache_key in self.correlation_cache and self.config.use_incremental:
            # Incremental update
            cache_entry = self.correlation_cache[cache_key]
            
            new_corr, new_means, new_vars = incremental_correlation_update(
                cache_entry['corr'],
                cache_entry['means'],
                cache_entry['vars'],
                cache_entry['n'],
                new_array
            )
            
            # Update cache
            self.correlation_cache[cache_key] = {
                'corr': new_corr,
                'means': new_means,
                'vars': new_vars,
                'n': cache_entry['n'] + new_array.shape[0]
            }
            
            self.cache_stats['updates'] += 1
            return new_corr
            
        else:
            # Full calculation
            corr = self.calculate_correlation_matrix(new_data)
            
            # Initialize cache
            self.correlation_cache[cache_key] = {
                'corr': corr,
                'means': new_array.mean(axis=0),
                'vars': new_array.var(axis=0),
                'n': new_array.shape[0]
            }
            
            self.cache_stats['misses'] += 1
            return corr
            
    def calculate_resistance_matrix(self, price_data: pd.DataFrame,
                                  volume_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Calculate support/resistance levels for all 10 components
        
        Returns dictionary with resistance levels and confluence zones
        """
        resistance_data = {}
        
        for component in self.components:
            if component in price_data.columns:
                levels = self._calculate_component_resistance(
                    price_data[component],
                    volume_data[component] if volume_data is not None else None
                )
                resistance_data[component] = levels
                
        # Find confluence zones across components
        confluence = self._find_confluence_zones(resistance_data)
        
        return {
            'component_levels': resistance_data,
            'confluence_zones': confluence,
            'calculation_time': time.time()
        }
        
    @jit(nopython=True)
    def _calculate_component_resistance(prices: np.ndarray, 
                                      volumes: Optional[np.ndarray] = None) -> Dict[str, List[float]]:
        """Calculate support/resistance for a single component"""
        # Simplified S/R calculation for demonstration
        # In practice, would use more sophisticated algorithms
        
        # Find local extrema
        support_levels = []
        resistance_levels = []
        
        for i in range(1, len(prices) - 1):
            if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                support_levels.append(prices[i])
            elif prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                resistance_levels.append(prices[i])
                
        return {
            'support': sorted(set(support_levels))[-5:],  # Top 5 levels
            'resistance': sorted(set(resistance_levels))[:5]
        }
        
    def _find_confluence_zones(self, resistance_data: Dict[str, Any],
                              tolerance: float = 0.005) -> List[Dict[str, Any]]:
        """Find price levels where multiple components have S/R"""
        all_levels = []
        
        # Collect all levels with component info
        for component, levels in resistance_data.items():
            for level in levels.get('support', []):
                all_levels.append({'price': level, 'type': 'support', 'component': component})
            for level in levels.get('resistance', []):
                all_levels.append({'price': level, 'type': 'resistance', 'component': component})
                
        # Sort by price
        all_levels.sort(key=lambda x: x['price'])
        
        # Find confluences
        confluences = []
        i = 0
        while i < len(all_levels):
            cluster = [all_levels[i]]
            base_price = all_levels[i]['price']
            
            # Find nearby levels
            j = i + 1
            while j < len(all_levels) and (all_levels[j]['price'] - base_price) / base_price < tolerance:
                cluster.append(all_levels[j])
                j += 1
                
            # If multiple components at this level
            if len(set(level['component'] for level in cluster)) >= 3:
                confluences.append({
                    'price': np.mean([level['price'] for level in cluster]),
                    'strength': len(cluster),
                    'components': list(set(level['component'] for level in cluster)),
                    'types': list(set(level['type'] for level in cluster))
                })
                
            i = j
            
        return confluences
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'cache_stats': self.cache_stats,
            'memory_pool_usage': len(self.memory_pool.in_use),
            'gpu_enabled': self.config.use_gpu and GPU_AVAILABLE,
            'precision': self.config.precision
        }
        
    def clear_cache(self):
        """Clear all caches"""
        self.correlation_cache.clear()
        self.cache_stats = {'hits': 0, 'misses': 0, 'updates': 0}
        
        if self.config.use_gpu and GPU_AVAILABLE:
            self.gpu_memory_pool.free_all_blocks()