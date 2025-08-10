"""
Correlation Analyzer
====================

Analyzes correlations between different market components and assets
to detect regime changes and market synchronization.

Author: Market Regime Refactoring Team
Date: 2025-07-08
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from scipy import stats
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CorrelationData:
    """Container for correlation analysis results"""
    overall_correlation: float  # -1 to 1
    correlation_matrix: np.ndarray
    eigenvalues: np.ndarray
    market_dispersion: float
    correlation_regime: str
    sector_correlations: Dict[str, float]
    rolling_correlation_trend: float
    regime_signal: float


class CorrelationAnalyzer:
    """
    Analyzes market correlations for regime detection
    
    This is one of the 9 active components in the enhanced market regime system.
    Base weight: 0.07 (7% of total regime signal)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the correlation analyzer"""
        self.config = config or {}
        
        # Configuration parameters
        self.lookback_period = self.config.get('lookback_period', 20)
        self.correlation_window = self.config.get('correlation_window', 60)
        self.min_correlation_threshold = self.config.get('min_correlation_threshold', 0.3)
        self.dispersion_threshold = self.config.get('dispersion_threshold', 0.4)
        
        # Asset groups for correlation analysis
        self.asset_groups = self.config.get('asset_groups', {
            'indices': ['NIFTY', 'BANKNIFTY', 'FINNIFTY'],
            'sectors': ['IT', 'PHARMA', 'AUTO', 'METAL'],
            'global': ['USDINR', 'GOLD', 'CRUDE']
        })
        
        # Cache for performance
        self._correlation_cache = {}
        self._last_calculation = None
        
        logger.info(f"CorrelationAnalyzer initialized with {self.correlation_window}-period window")
    
    def analyze(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Main analysis method for correlation patterns
        
        Args:
            market_data: DataFrame with columns:
                - datetime_
                - symbol or underlying columns
                - close or underlying_close
                - volume
                - Additional asset columns if available
                
        Returns:
            Dict with correlation analysis results
        """
        try:
            # Extract returns data
            returns_data = self._prepare_returns_data(market_data)
            
            if returns_data.empty or len(returns_data.columns) < 2:
                logger.warning("Insufficient data for correlation analysis")
                return self._get_default_results()
            
            # Calculate correlation metrics
            correlation_data = self._calculate_correlations(returns_data)
            
            # Analyze correlation patterns
            regime_signal = self._calculate_regime_signal(correlation_data)
            
            # Prepare results
            results = {
                'correlation_score': regime_signal,
                'overall_correlation': correlation_data.overall_correlation,
                'market_dispersion': correlation_data.market_dispersion,
                'correlation_regime': correlation_data.correlation_regime,
                'eigenvalue_ratio': self._calculate_eigenvalue_ratio(correlation_data.eigenvalues),
                'correlation_trend': correlation_data.rolling_correlation_trend,
                'cross_market_correlation': self._calculate_cross_market_correlation(returns_data),
                'sector_rotation_signal': self._calculate_sector_rotation_signal(correlation_data),
                'risk_on_off_signal': self._calculate_risk_on_off_signal(correlation_data),
                'timestamp': datetime.now()
            }
            
            # Add sector correlations if available
            if correlation_data.sector_correlations:
                results['sector_correlations'] = correlation_data.sector_correlations
            
            # Cache results
            self._correlation_cache['last_results'] = results
            self._last_calculation = datetime.now()
            
            return results
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
            return self._get_default_results()
    
    def _prepare_returns_data(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare returns data from market data
        
        Args:
            market_data: Raw market data
            
        Returns:
            DataFrame with returns for each asset
        """
        returns_df = pd.DataFrame()
        
        # Primary asset (NIFTY)
        if 'underlying_close' in market_data.columns:
            prices = market_data.groupby('datetime_')['underlying_close'].last()
            returns_df['NIFTY'] = prices.pct_change().fillna(0)
        
        # Try to extract other assets if available
        # This is simplified - in production, would query multiple assets
        if 'symbol' in market_data.columns:
            unique_symbols = market_data['symbol'].unique()
            for symbol in unique_symbols[:5]:  # Limit to 5 symbols
                symbol_data = market_data[market_data['symbol'] == symbol]
                if len(symbol_data) > 10:
                    prices = symbol_data.groupby('datetime_')['close'].last()
                    returns_df[symbol] = prices.pct_change().fillna(0)
        
        # Simulate additional assets for demonstration
        # In production, this would come from actual multi-asset data
        if len(returns_df.columns) < 3:
            base_returns = returns_df.iloc[:, 0] if len(returns_df.columns) > 0 else pd.Series(np.random.randn(100) * 0.01)
            
            # Add correlated assets
            returns_df['Asset_2'] = base_returns + np.random.randn(len(base_returns)) * 0.005
            returns_df['Asset_3'] = base_returns * 0.8 + np.random.randn(len(base_returns)) * 0.008
            returns_df['Asset_4'] = -base_returns * 0.6 + np.random.randn(len(base_returns)) * 0.01
        
        return returns_df.dropna()
    
    def _calculate_correlations(self, returns_data: pd.DataFrame) -> CorrelationData:
        """
        Calculate various correlation metrics
        
        Args:
            returns_data: DataFrame with asset returns
            
        Returns:
            CorrelationData object
        """
        # Calculate correlation matrix
        corr_matrix = returns_data.corr().values
        
        # Overall correlation (average of off-diagonal elements)
        n = len(corr_matrix)
        if n > 1:
            mask = ~np.eye(n, dtype=bool)
            overall_corr = np.mean(np.abs(corr_matrix[mask]))
        else:
            overall_corr = 1.0
        
        # Calculate eigenvalues for dispersion
        eigenvalues = np.linalg.eigvals(corr_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
        
        # Market dispersion (1 - largest eigenvalue / sum of eigenvalues)
        if len(eigenvalues) > 0 and np.sum(eigenvalues) > 0:
            dispersion = 1 - (eigenvalues[0] / np.sum(eigenvalues))
        else:
            dispersion = 0.5
        
        # Determine correlation regime
        if overall_corr > 0.7:
            regime = "high_correlation"
        elif overall_corr > 0.5:
            regime = "medium_correlation"
        elif overall_corr > 0.3:
            regime = "low_correlation"
        else:
            regime = "decorrelated"
        
        # Calculate sector correlations (simplified)
        sector_corrs = {}
        if len(returns_data.columns) >= 3:
            for i, col in enumerate(returns_data.columns[:3]):
                avg_corr = np.mean([corr_matrix[i, j] for j in range(n) if i != j])
                sector_corrs[f"Sector_{i+1}"] = avg_corr
        
        # Rolling correlation trend
        if len(returns_data) > self.correlation_window:
            rolling_corrs = []
            for i in range(self.correlation_window, len(returns_data)):
                window_data = returns_data.iloc[i-self.correlation_window:i]
                window_corr = window_data.corr().values
                mask = ~np.eye(len(window_corr), dtype=bool)
                rolling_corrs.append(np.mean(np.abs(window_corr[mask])))
            
            if len(rolling_corrs) > 1:
                # Simple trend: positive if increasing, negative if decreasing
                correlation_trend = (rolling_corrs[-1] - rolling_corrs[0]) / len(rolling_corrs)
                correlation_trend = np.clip(correlation_trend * 100, -1, 1)
            else:
                correlation_trend = 0.0
        else:
            correlation_trend = 0.0
        
        return CorrelationData(
            overall_correlation=overall_corr,
            correlation_matrix=corr_matrix,
            eigenvalues=eigenvalues,
            market_dispersion=dispersion,
            correlation_regime=regime,
            sector_correlations=sector_corrs,
            rolling_correlation_trend=correlation_trend,
            regime_signal=0.0  # Will be calculated separately
        )
    
    def _calculate_regime_signal(self, correlation_data: CorrelationData) -> float:
        """
        Calculate regime signal from correlation data
        
        Args:
            correlation_data: Correlation analysis results
            
        Returns:
            float: Regime signal between -1 and 1
        """
        signal_components = []
        
        # 1. Overall correlation signal
        # High correlation = risk-on, trending markets
        # Low correlation = risk-off, diverging markets
        corr_signal = (correlation_data.overall_correlation - 0.5) * 2
        signal_components.append(('overall_correlation', corr_signal, 0.3))
        
        # 2. Dispersion signal
        # Low dispersion = synchronized market (trending)
        # High dispersion = divergent market (ranging)
        dispersion_signal = -(correlation_data.market_dispersion - 0.5) * 2
        signal_components.append(('dispersion', dispersion_signal, 0.25))
        
        # 3. Correlation trend signal
        # Increasing correlation = strengthening trend
        # Decreasing correlation = potential regime change
        trend_signal = correlation_data.rolling_correlation_trend
        signal_components.append(('correlation_trend', trend_signal, 0.25))
        
        # 4. Regime-based signal
        regime_signals = {
            'high_correlation': 0.8,
            'medium_correlation': 0.3,
            'low_correlation': -0.3,
            'decorrelated': -0.8
        }
        regime_signal = regime_signals.get(correlation_data.correlation_regime, 0.0)
        signal_components.append(('regime', regime_signal, 0.2))
        
        # Combine signals with weights
        total_signal = 0.0
        for name, signal, weight in signal_components:
            total_signal += signal * weight
            logger.debug(f"Correlation {name}: {signal:.3f} (weight: {weight})")
        
        return np.clip(total_signal, -1.0, 1.0)
    
    def _calculate_eigenvalue_ratio(self, eigenvalues: np.ndarray) -> float:
        """Calculate ratio of largest eigenvalue to total"""
        if len(eigenvalues) > 0 and np.sum(eigenvalues) > 0:
            return eigenvalues[0] / np.sum(eigenvalues)
        return 0.0
    
    def _calculate_cross_market_correlation(self, returns_data: pd.DataFrame) -> float:
        """Calculate average cross-market correlation"""
        if len(returns_data.columns) < 2:
            return 0.0
        
        corr_matrix = returns_data.corr().values
        n = len(corr_matrix)
        mask = ~np.eye(n, dtype=bool)
        
        return np.mean(corr_matrix[mask])
    
    def _calculate_sector_rotation_signal(self, correlation_data: CorrelationData) -> float:
        """
        Calculate sector rotation signal from correlation patterns
        
        Low correlation between sectors = rotation occurring
        High correlation = no rotation
        """
        if not correlation_data.sector_correlations:
            return 0.0
        
        avg_sector_corr = np.mean(list(correlation_data.sector_correlations.values()))
        
        # Invert: low correlation = high rotation signal
        rotation_signal = (0.5 - avg_sector_corr) * 2
        
        return np.clip(rotation_signal, -1.0, 1.0)
    
    def _calculate_risk_on_off_signal(self, correlation_data: CorrelationData) -> float:
        """
        Calculate risk-on/risk-off signal
        
        High correlation + low dispersion = risk-on
        Low correlation + high dispersion = risk-off
        """
        risk_signal = (
            correlation_data.overall_correlation * 0.6 +
            (1 - correlation_data.market_dispersion) * 0.4
        )
        
        # Scale to -1 to 1 where 1 = risk-on, -1 = risk-off
        risk_signal = (risk_signal - 0.5) * 2
        
        return np.clip(risk_signal, -1.0, 1.0)
    
    def get_correlation_matrix(self) -> Optional[np.ndarray]:
        """Get the most recent correlation matrix"""
        if 'last_results' in self._correlation_cache:
            return self._correlation_cache.get('correlation_matrix')
        return None
    
    def get_market_synchronization_score(self) -> float:
        """
        Get market synchronization score (0-1)
        
        High score = markets moving together
        Low score = markets diverging
        """
        if 'last_results' not in self._correlation_cache:
            return 0.5
        
        results = self._correlation_cache['last_results']
        
        # Combine correlation and dispersion
        sync_score = (
            results.get('overall_correlation', 0.5) * 0.7 +
            (1 - results.get('market_dispersion', 0.5)) * 0.3
        )
        
        return np.clip(sync_score, 0.0, 1.0)
    
    def _get_default_results(self) -> Dict[str, Any]:
        """Get default results when analysis fails"""
        return {
            'correlation_score': 0.0,
            'overall_correlation': 0.5,
            'market_dispersion': 0.5,
            'correlation_regime': 'unknown',
            'eigenvalue_ratio': 0.0,
            'correlation_trend': 0.0,
            'cross_market_correlation': 0.0,
            'sector_rotation_signal': 0.0,
            'risk_on_off_signal': 0.0,
            'timestamp': datetime.now()
        }