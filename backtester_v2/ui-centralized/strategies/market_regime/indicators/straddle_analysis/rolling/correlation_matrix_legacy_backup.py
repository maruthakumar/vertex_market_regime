"""
6×6 Correlation Matrix for Triple Straddle Components

Manages rolling correlation analysis between all 6 option components:
ATM_CE, ATM_PE, ITM1_CE, ITM1_PE, OTM1_CE, OTM1_PE

Provides comprehensive correlation tracking across multiple timeframes
for market regime formation and component relationship analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
from numba import jit, prange
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class CorrelationResult:
    """Correlation analysis result"""
    matrix: np.ndarray
    component_names: List[str]
    timestamp: pd.Timestamp
    window_size: int
    avg_correlation: float
    max_correlation: float
    min_correlation: float


class CorrelationMatrix:
    """
    6×6 Rolling Correlation Matrix Manager
    
    Tracks correlations between all 6 option components across multiple
    timeframes for comprehensive market regime analysis.
    
    Component Matrix:
    - ATM_CE, ATM_PE, ITM1_CE, ITM1_PE, OTM1_CE, OTM1_PE
    - 15 unique correlation pairs (6×6 symmetric matrix)
    - Rolling analysis across [3,5,10,15] minute windows
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize correlation matrix manager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Component definitions (6 components total)
        self.components = ['atm_ce', 'atm_pe', 'itm1_ce', 'itm1_pe', 'otm1_ce', 'otm1_pe']
        self.n_components = len(self.components)
        
        # Rolling window sizes
        self.window_sizes = config.get('rolling_windows', [3, 5, 10, 15])
        
        # Correlation thresholds for regime formation
        self.correlation_thresholds = config.get('correlation_thresholds', {
            'high_correlation': 0.8,
            'medium_correlation': 0.5,
            'low_correlation': 0.2
        })
        
        # Storage for correlation matrices by window size
        self.correlation_matrices = {}
        self.correlation_history = {}
        
        # Initialize storage for each window size
        for window_size in self.window_sizes:
            self.correlation_matrices[window_size] = np.eye(self.n_components)
            self.correlation_history[window_size] = []
        
        # Component pair mapping for efficient access
        self.component_pairs = []
        for i in range(self.n_components):
            for j in range(i + 1, self.n_components):
                self.component_pairs.append((self.components[i], self.components[j]))
        
        self.logger.info(f"6×6 Correlation Matrix initialized for components: {self.components}")
        self.logger.info(f"Tracking {len(self.component_pairs)} unique correlation pairs")
    
    @staticmethod
    @jit(nopython=True)
    def _fast_correlation_matrix(data_matrix: np.ndarray) -> np.ndarray:
        """
        Fast correlation matrix calculation using Numba
        
        Args:
            data_matrix: 2D array with shape (n_observations, n_components)
            
        Returns:
            Correlation matrix (n_components × n_components)
        """
        n_obs, n_comp = data_matrix.shape
        corr_matrix = np.eye(n_comp)
        
        for i in prange(n_comp):
            for j in prange(i + 1, n_comp):
                x = data_matrix[:, i]
                y = data_matrix[:, j]
                
                # Calculate means
                x_mean = np.mean(x)
                y_mean = np.mean(y)
                
                # Calculate correlation
                numerator = np.sum((x - x_mean) * (y - y_mean))
                x_var = np.sum((x - x_mean) ** 2)
                y_var = np.sum((y - y_mean) ** 2)
                
                if x_var > 0 and y_var > 0:
                    correlation = numerator / np.sqrt(x_var * y_var)
                else:
                    correlation = 0.0
                
                # Fill symmetric matrix
                corr_matrix[i, j] = correlation
                corr_matrix[j, i] = correlation
        
        return corr_matrix
    
    def calculate_correlation_matrix(self,
                                   component_data: Dict[str, List[float]],
                                   window_size: int,
                                   timestamp: pd.Timestamp) -> Optional[CorrelationResult]:
        """
        Calculate 6×6 correlation matrix for current window
        
        Args:
            component_data: Dictionary with data for each component
            window_size: Rolling window size in minutes
            timestamp: Current timestamp
            
        Returns:
            CorrelationResult object or None if insufficient data
        """
        try:
            # Validate input data
            if not all(comp in component_data for comp in self.components):
                self.logger.warning("Missing component data for correlation calculation")
                return None
            
            # Check data length consistency
            data_lengths = [len(component_data[comp]) for comp in self.components]
            if not all(length >= window_size for length in data_lengths):
                return None
            
            # Create data matrix (observations × components)
            min_length = min(data_lengths)
            data_matrix = np.zeros((min_length, self.n_components))
            
            for i, component in enumerate(self.components):
                data_matrix[:, i] = component_data[component][-min_length:]
            
            # Calculate correlation matrix using optimized function
            correlation_matrix = self._fast_correlation_matrix(data_matrix)
            
            # Calculate summary statistics
            # Extract upper triangular (excluding diagonal) for statistics
            upper_tri_indices = np.triu_indices(self.n_components, k=1)
            correlations = correlation_matrix[upper_tri_indices]
            
            avg_correlation = np.mean(correlations)
            max_correlation = np.max(correlations)
            min_correlation = np.min(correlations)
            
            # Create result object
            result = CorrelationResult(
                matrix=correlation_matrix,
                component_names=self.components.copy(),
                timestamp=timestamp,
                window_size=window_size,
                avg_correlation=avg_correlation,
                max_correlation=max_correlation,
                min_correlation=min_correlation
            )
            
            # Update stored matrices
            self.correlation_matrices[window_size] = correlation_matrix.copy()
            
            # Maintain history (keep last 100 observations per window)
            if len(self.correlation_history[window_size]) >= 100:
                self.correlation_history[window_size].pop(0)
            self.correlation_history[window_size].append(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation matrix: {e}")
            return None
    
    def get_correlation_matrix(self, window_size: int) -> Optional[np.ndarray]:
        """
        Get current correlation matrix for window size
        
        Args:
            window_size: Window size in minutes
            
        Returns:
            6×6 correlation matrix or None if not available
        """
        if window_size in self.correlation_matrices:
            return self.correlation_matrices[window_size].copy()
        return None
    
    def get_component_correlation(self,
                                component1: str,
                                component2: str,
                                window_size: int) -> Optional[float]:
        """
        Get correlation between two specific components
        
        Args:
            component1, component2: Component names
            window_size: Window size in minutes
            
        Returns:
            Correlation coefficient or None if not available
        """
        if component1 not in self.components or component2 not in self.components:
            return None
        
        matrix = self.get_correlation_matrix(window_size)
        if matrix is None:
            return None
        
        idx1 = self.components.index(component1)
        idx2 = self.components.index(component2)
        
        return matrix[idx1, idx2]
    
    def get_correlation_summary(self, window_size: int) -> Optional[Dict[str, float]]:
        """
        Get correlation summary statistics for window size
        
        Args:
            window_size: Window size in minutes
            
        Returns:
            Dictionary with correlation statistics
        """
        matrix = self.get_correlation_matrix(window_size)
        if matrix is None:
            return None
        
        # Extract upper triangular correlations (excluding diagonal)
        upper_tri_indices = np.triu_indices(self.n_components, k=1)
        correlations = matrix[upper_tri_indices]
        
        return {
            'avg_correlation': np.mean(correlations),
            'max_correlation': np.max(correlations),
            'min_correlation': np.min(correlations),
            'std_correlation': np.std(correlations),
            'median_correlation': np.median(correlations),
            'high_corr_count': np.sum(correlations > self.correlation_thresholds['high_correlation']),
            'medium_corr_count': np.sum(correlations > self.correlation_thresholds['medium_correlation']),
            'low_corr_count': np.sum(correlations < self.correlation_thresholds['low_correlation'])
        }
    
    def get_strongest_correlations(self, 
                                 window_size: int,
                                 top_n: int = 5) -> List[Tuple[str, str, float]]:
        """
        Get strongest correlations for window size
        
        Args:
            window_size: Window size in minutes
            top_n: Number of top correlations to return
            
        Returns:
            List of (component1, component2, correlation) tuples
        """
        matrix = self.get_correlation_matrix(window_size)
        if matrix is None:
            return []
        
        correlations = []
        
        for i in range(self.n_components):
            for j in range(i + 1, self.n_components):
                correlation = matrix[i, j]
                correlations.append((
                    self.components[i],
                    self.components[j],
                    abs(correlation)  # Use absolute value for ranking
                ))
        
        # Sort by absolute correlation value (descending)
        correlations.sort(key=lambda x: x[2], reverse=True)
        
        return correlations[:top_n]
    
    def detect_correlation_regime(self, window_size: int) -> Dict[str, Any]:
        """
        Detect market regime based on correlation patterns
        
        Args:
            window_size: Window size in minutes
            
        Returns:
            Dictionary with regime classification and characteristics
        """
        summary = self.get_correlation_summary(window_size)
        if summary is None:
            return {'regime': 'UNKNOWN', 'confidence': 0.0}
        
        avg_corr = summary['avg_correlation']
        high_corr_count = summary['high_corr_count']
        std_corr = summary['std_correlation']
        
        # Regime classification logic
        if avg_corr > 0.7 and high_corr_count >= 8:
            regime = 'HIGH_CORRELATION_STRESS'
            confidence = min(avg_corr + 0.2, 1.0)
        elif avg_corr > 0.5 and high_corr_count >= 5:
            regime = 'MEDIUM_CORRELATION_TRENDING'
            confidence = avg_corr
        elif avg_corr < 0.2 and std_corr > 0.3:
            regime = 'LOW_CORRELATION_RANGING'
            confidence = 1.0 - avg_corr
        elif std_corr > 0.4:
            regime = 'MIXED_CORRELATION_VOLATILE'
            confidence = std_corr
        else:
            regime = 'NORMAL_CORRELATION'
            confidence = 0.5
        
        return {
            'regime': regime,
            'confidence': confidence,
            'avg_correlation': avg_corr,
            'high_corr_pairs': high_corr_count,
            'correlation_volatility': std_corr,
            'window_size': window_size
        }
    
    def get_multi_timeframe_regime(self) -> Dict[str, Any]:
        """
        Get regime classification across all timeframes
        
        Returns:
            Multi-timeframe regime analysis
        """
        regimes = {}
        
        for window_size in self.window_sizes:
            regime_info = self.detect_correlation_regime(window_size)
            regimes[f"{window_size}min"] = regime_info
        
        # Aggregate across timeframes
        all_regimes = [info['regime'] for info in regimes.values() if info['regime'] != 'UNKNOWN']
        all_confidences = [info['confidence'] for info in regimes.values() if info['regime'] != 'UNKNOWN']
        
        if not all_regimes:
            return {'overall_regime': 'UNKNOWN', 'confidence': 0.0, 'timeframe_regimes': regimes}
        
        # Find most common regime
        from collections import Counter
        regime_counts = Counter(all_regimes)
        most_common_regime = regime_counts.most_common(1)[0][0]
        
        # Calculate average confidence for most common regime
        regime_confidences = [
            conf for regime, conf in zip(all_regimes, all_confidences)
            if regime == most_common_regime
        ]
        avg_confidence = np.mean(regime_confidences) if regime_confidences else 0.0
        
        return {
            'overall_regime': most_common_regime,
            'confidence': avg_confidence,
            'regime_consistency': regime_counts[most_common_regime] / len(all_regimes),
            'timeframe_regimes': regimes
        }
    
    def get_correlation_trend(self, 
                            window_size: int,
                            lookback_periods: int = 10) -> Dict[str, float]:
        """
        Analyze correlation trend over recent periods
        
        Args:
            window_size: Window size in minutes
            lookback_periods: Number of recent periods to analyze
            
        Returns:
            Dictionary with trend analysis
        """
        if window_size not in self.correlation_history:
            return {}
        
        history = self.correlation_history[window_size]
        if len(history) < 2:
            return {}
        
        recent_history = history[-min(lookback_periods, len(history)):]
        
        # Extract average correlations over time
        avg_correlations = [result.avg_correlation for result in recent_history]
        
        if len(avg_correlations) < 2:
            return {}
        
        # Calculate trend
        x = np.arange(len(avg_correlations))
        slope, intercept = np.polyfit(x, avg_correlations, 1)
        
        return {
            'trend_slope': slope,
            'current_correlation': avg_correlations[-1],
            'correlation_change': avg_correlations[-1] - avg_correlations[0],
            'periods_analyzed': len(avg_correlations),
            'trend_direction': 'INCREASING' if slope > 0.01 else 'DECREASING' if slope < -0.01 else 'STABLE'
        }
    
    def export_correlation_data(self, window_size: int) -> pd.DataFrame:
        """
        Export correlation matrix as DataFrame for analysis
        
        Args:
            window_size: Window size in minutes
            
        Returns:
            DataFrame with correlation matrix
        """
        matrix = self.get_correlation_matrix(window_size)
        if matrix is None:
            return pd.DataFrame()
        
        return pd.DataFrame(
            matrix,
            index=self.components,
            columns=self.components
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status
        
        Returns:
            System status dictionary
        """
        status = {
            'components': self.components,
            'n_components': self.n_components,
            'n_correlation_pairs': len(self.component_pairs),
            'window_sizes': self.window_sizes,
            'correlation_thresholds': self.correlation_thresholds,
            'matrices_available': list(self.correlation_matrices.keys()),
            'history_lengths': {
                window_size: len(history) 
                for window_size, history in self.correlation_history.items()
            }
        }
        
        return status