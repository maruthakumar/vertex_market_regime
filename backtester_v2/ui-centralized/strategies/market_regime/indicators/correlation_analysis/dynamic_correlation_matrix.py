"""
Dynamic Correlation Matrix
==========================

Maintains and updates a dynamic correlation matrix between multiple
assets and market components for real-time regime detection.

Author: Market Regime Refactoring Team
Date: 2025-07-08
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set
import logging
from datetime import datetime, timedelta
from collections import deque
import json

logger = logging.getLogger(__name__)


class DynamicCorrelationMatrix:
    """
    Maintains a dynamic correlation matrix that updates in real-time
    
    Features:
    - Sliding window correlation updates
    - Exponential weighting for recent data
    - Correlation stability tracking
    - Anomaly detection in correlation changes
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the dynamic correlation matrix"""
        self.config = config or {}
        
        # Configuration
        self.window_size = self.config.get('window_size', 252)  # 1 year default
        self.min_periods = self.config.get('min_periods', 20)
        self.decay_factor = self.config.get('decay_factor', 0.94)  # For EWMA
        self.update_frequency = self.config.get('update_frequency', 60)  # Every 60 data points
        self.stability_threshold = self.config.get('stability_threshold', 0.1)
        
        # Data storage
        self.asset_returns = {}  # Dict of deques for each asset
        self.correlation_matrix = None
        self.previous_matrix = None
        self.correlation_history = deque(maxlen=100)
        
        # Tracking
        self.last_update = None
        self.update_count = 0
        self.stability_scores = {}
        self.correlation_changes = {}
        
        # Asset management
        self.active_assets = set()
        self.asset_metadata = {}
        
        logger.info(f"DynamicCorrelationMatrix initialized with {self.window_size}-period window")
    
    def add_asset_data(self, asset_name: str, returns: Union[float, List[float]]):
        """
        Add return data for an asset
        
        Args:
            asset_name: Name of the asset
            returns: Single return value or list of returns
        """
        if asset_name not in self.asset_returns:
            self.asset_returns[asset_name] = deque(maxlen=self.window_size)
            self.active_assets.add(asset_name)
            logger.info(f"Added new asset to correlation matrix: {asset_name}")
        
        if isinstance(returns, (int, float)):
            self.asset_returns[asset_name].append(returns)
        else:
            self.asset_returns[asset_name].extend(returns)
        
        # Check if update needed
        self.update_count += 1
        if self.update_count >= self.update_frequency:
            self.update_correlation_matrix()
    
    def update_correlation_matrix(self, force: bool = False):
        """
        Update the correlation matrix
        
        Args:
            force: Force update regardless of frequency
        """
        if not force and self.update_count < self.update_frequency:
            return
        
        try:
            # Prepare data
            returns_df = self._prepare_returns_dataframe()
            
            if returns_df.empty or len(returns_df) < self.min_periods:
                logger.warning("Insufficient data for correlation calculation")
                return
            
            # Store previous matrix
            if self.correlation_matrix is not None:
                self.previous_matrix = self.correlation_matrix.copy()
            
            # Calculate new correlation matrix
            if self.decay_factor < 1.0:
                # Use exponentially weighted correlation
                self.correlation_matrix = self._calculate_ewm_correlation(returns_df)
            else:
                # Standard correlation
                self.correlation_matrix = returns_df.corr().values
            
            # Track changes
            if self.previous_matrix is not None:
                self._track_correlation_changes()
            
            # Update stability scores
            self._update_stability_scores()
            
            # Record in history
            self.correlation_history.append({
                'timestamp': datetime.now(),
                'matrix': self.correlation_matrix.copy(),
                'stability_score': self._calculate_overall_stability()
            })
            
            # Reset counter
            self.update_count = 0
            self.last_update = datetime.now()
            
            logger.debug(f"Correlation matrix updated for {len(self.active_assets)} assets")
            
        except Exception as e:
            logger.error(f"Error updating correlation matrix: {e}")
    
    def _prepare_returns_dataframe(self) -> pd.DataFrame:
        """Prepare returns data as DataFrame"""
        if not self.asset_returns:
            return pd.DataFrame()
        
        # Find minimum length
        min_length = min(len(returns) for returns in self.asset_returns.values())
        
        if min_length < self.min_periods:
            return pd.DataFrame()
        
        # Create DataFrame
        data = {}
        for asset, returns in self.asset_returns.items():
            # Use most recent data up to min_length
            data[asset] = list(returns)[-min_length:]
        
        return pd.DataFrame(data)
    
    def _calculate_ewm_correlation(self, returns_df: pd.DataFrame) -> np.ndarray:
        """Calculate exponentially weighted correlation matrix"""
        # Calculate EWMA correlation
        ewm = returns_df.ewm(alpha=1-self.decay_factor, min_periods=self.min_periods)
        
        # Manual correlation calculation with exponential weights
        n_assets = len(returns_df.columns)
        corr_matrix = np.eye(n_assets)
        
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                # Calculate weighted correlation
                x = returns_df.iloc[:, i].values
                y = returns_df.iloc[:, j].values
                
                # Apply exponential weights
                weights = self._get_exponential_weights(len(x))
                
                # Weighted correlation
                corr = self._weighted_correlation(x, y, weights)
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
        
        return corr_matrix
    
    def _get_exponential_weights(self, n: int) -> np.ndarray:
        """Get exponential weights for n observations"""
        weights = np.array([(1 - self.decay_factor) * (self.decay_factor ** i) 
                           for i in range(n-1, -1, -1)])
        return weights / weights.sum()
    
    def _weighted_correlation(self, x: np.ndarray, y: np.ndarray, 
                            weights: np.ndarray) -> float:
        """Calculate weighted correlation coefficient"""
        # Weighted means
        mean_x = np.average(x, weights=weights)
        mean_y = np.average(y, weights=weights)
        
        # Weighted covariance and standard deviations
        cov = np.average((x - mean_x) * (y - mean_y), weights=weights)
        std_x = np.sqrt(np.average((x - mean_x)**2, weights=weights))
        std_y = np.sqrt(np.average((y - mean_y)**2, weights=weights))
        
        if std_x > 0 and std_y > 0:
            return cov / (std_x * std_y)
        return 0.0
    
    def _track_correlation_changes(self):
        """Track changes in correlations"""
        if self.previous_matrix is None:
            return
        
        changes = self.correlation_matrix - self.previous_matrix
        
        # Track significant changes
        assets = list(self.active_assets)
        self.correlation_changes = {}
        
        for i in range(len(assets)):
            for j in range(i+1, len(assets)):
                change = changes[i, j]
                if abs(change) > self.stability_threshold:
                    pair = f"{assets[i]}-{assets[j]}"
                    self.correlation_changes[pair] = {
                        'change': change,
                        'previous': self.previous_matrix[i, j],
                        'current': self.correlation_matrix[i, j]
                    }
    
    def _update_stability_scores(self):
        """Update correlation stability scores"""
        if len(self.correlation_history) < 2:
            return
        
        assets = list(self.active_assets)
        
        for i, asset in enumerate(assets):
            # Calculate stability as inverse of correlation variance
            recent_correlations = []
            
            for hist in list(self.correlation_history)[-10:]:
                if i < len(hist['matrix']):
                    # Average correlation with other assets
                    avg_corr = np.mean([hist['matrix'][i, j] 
                                       for j in range(len(hist['matrix'])) 
                                       if i != j])
                    recent_correlations.append(avg_corr)
            
            if recent_correlations:
                stability = 1 / (1 + np.std(recent_correlations))
                self.stability_scores[asset] = stability
    
    def _calculate_overall_stability(self) -> float:
        """Calculate overall matrix stability score"""
        if not self.stability_scores:
            return 1.0
        
        return np.mean(list(self.stability_scores.values()))
    
    def get_correlation(self, asset1: str, asset2: str) -> Optional[float]:
        """
        Get correlation between two assets
        
        Args:
            asset1: First asset name
            asset2: Second asset name
            
        Returns:
            Correlation coefficient or None if not available
        """
        if self.correlation_matrix is None:
            return None
        
        assets = list(self.active_assets)
        
        if asset1 not in assets or asset2 not in assets:
            return None
        
        i = assets.index(asset1)
        j = assets.index(asset2)
        
        return self.correlation_matrix[i, j]
    
    def get_asset_correlations(self, asset: str) -> Dict[str, float]:
        """
        Get all correlations for a specific asset
        
        Args:
            asset: Asset name
            
        Returns:
            Dict of correlations with other assets
        """
        if self.correlation_matrix is None or asset not in self.active_assets:
            return {}
        
        assets = list(self.active_assets)
        i = assets.index(asset)
        
        correlations = {}
        for j, other_asset in enumerate(assets):
            if i != j:
                correlations[other_asset] = self.correlation_matrix[i, j]
        
        return correlations
    
    def get_regime_signal(self) -> float:
        """
        Generate regime signal from correlation patterns
        
        Returns:
            float: Signal between -1 and 1
        """
        if self.correlation_matrix is None:
            return 0.0
        
        # Average absolute correlation
        n = len(self.correlation_matrix)
        if n <= 1:
            return 0.0
        
        mask = ~np.eye(n, dtype=bool)
        avg_corr = np.mean(np.abs(self.correlation_matrix[mask]))
        
        # Stability factor
        stability = self._calculate_overall_stability()
        
        # High stable correlation = trending market (+1)
        # Low stable correlation = ranging market (0)
        # Unstable correlation = regime transition (-1)
        
        if stability > 0.8:
            # Stable correlations
            signal = (avg_corr - 0.5) * 2
        else:
            # Unstable correlations indicate regime change
            signal = -0.5 - (1 - stability) * 0.5
        
        return np.clip(signal, -1.0, 1.0)
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of the correlation matrix"""
        if self.correlation_matrix is None:
            return {}
        
        n = len(self.correlation_matrix)
        mask = ~np.eye(n, dtype=bool)
        correlations = self.correlation_matrix[mask]
        
        return {
            'n_assets': len(self.active_assets),
            'avg_correlation': np.mean(correlations),
            'std_correlation': np.std(correlations),
            'min_correlation': np.min(correlations),
            'max_correlation': np.max(correlations),
            'overall_stability': self._calculate_overall_stability(),
            'last_update': self.last_update,
            'significant_changes': len(self.correlation_changes),
            'update_count': len(self.correlation_history)
        }
    
    def export_correlation_matrix(self) -> Dict[str, Any]:
        """Export correlation matrix with metadata"""
        if self.correlation_matrix is None:
            return {}
        
        assets = list(self.active_assets)
        
        return {
            'assets': assets,
            'matrix': self.correlation_matrix.tolist(),
            'timestamp': datetime.now().isoformat(),
            'stability_scores': self.stability_scores,
            'summary': self.get_summary_statistics()
        }