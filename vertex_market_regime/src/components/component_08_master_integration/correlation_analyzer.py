"""
Cross-Component Correlation Analyzer

Analyzes correlations and relationships between component signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """
    Analyzes correlations between component signals for feature engineering
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the correlation analyzer
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Correlation parameters
        self.correlation_params = self.config.get('correlation_params', {
            'min_correlation': -1.0,
            'max_correlation': 1.0,
            'rolling_window': 20,
            'decay_factor': 0.95
        })
        
        # Component pairs (7 components = 21 unique pairs)
        self.component_pairs = self._generate_component_pairs()
        
        # Historical correlation tracking
        self.correlation_history = []
        
        logger.info(f"Initialized CorrelationAnalyzer with {len(self.component_pairs)} pairs")
    
    def _generate_component_pairs(self) -> List[Tuple[str, str]]:
        """Generate all unique component pairs"""
        components = [
            'component_01', 'component_02', 'component_03', 'component_04',
            'component_05', 'component_06', 'component_07'
        ]
        
        pairs = []
        for i, comp1 in enumerate(components):
            for comp2 in components[i+1:]:
                pairs.append((comp1, comp2))
        
        return pairs
    
    def analyze_correlations(
        self,
        aggregated_components: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze correlations between all component pairs
        
        Args:
            aggregated_components: Aggregated component data
            
        Returns:
            Dictionary containing correlation analysis results
        """
        # Calculate pairwise correlations
        pairwise_correlations = self._calculate_pairwise_correlations(
            aggregated_components
        )
        
        # Build correlation matrix
        correlation_matrix = self._build_correlation_matrix(
            pairwise_correlations
        )
        
        # Calculate rolling correlations
        rolling_correlations = self._calculate_rolling_correlations(
            aggregated_components
        )
        
        # Detect correlation breakdowns
        breakdown_analysis = self._detect_correlation_breakdowns(
            pairwise_correlations,
            rolling_correlations
        )
        
        # Calculate coherence metrics
        coherence_metrics = self._calculate_coherence_metrics(
            correlation_matrix
        )
        
        return {
            'pairwise_correlations': pairwise_correlations,
            'correlation_matrix': correlation_matrix,
            'rolling_correlations': rolling_correlations,
            'breakdown_analysis': breakdown_analysis,
            'coherence_metrics': coherence_metrics
        }
    
    def _calculate_pairwise_correlations(
        self,
        aggregated_components: Dict[str, Dict[str, Any]]
    ) -> Dict[Tuple[str, str], float]:
        """
        Calculate correlation for each component pair
        
        Args:
            aggregated_components: Aggregated component data
            
        Returns:
            Dictionary of pairwise correlations
        """
        correlations = {}
        
        for comp1_name, comp2_name in self.component_pairs:
            if comp1_name in aggregated_components and comp2_name in aggregated_components:
                comp1_signals = aggregated_components[comp1_name].get('signals', {})
                comp2_signals = aggregated_components[comp2_name].get('signals', {})
                
                # Get primary signals
                signals1 = list(comp1_signals.values())[:3] if comp1_signals else [0.0]
                signals2 = list(comp2_signals.values())[:3] if comp2_signals else [0.0]
                
                # Ensure same length
                min_len = min(len(signals1), len(signals2))
                if min_len > 0:
                    signals1 = signals1[:min_len]
                    signals2 = signals2[:min_len]
                    
                    # Calculate correlation
                    if len(signals1) > 1:
                        correlation = np.corrcoef(signals1, signals2)[0, 1]
                    else:
                        # Single value correlation
                        correlation = 1.0 if signals1[0] * signals2[0] > 0 else -1.0
                    
                    # Handle NaN
                    if np.isnan(correlation):
                        correlation = 0.0
                else:
                    correlation = 0.0
                
                correlations[(comp1_name, comp2_name)] = float(correlation)
            else:
                correlations[(comp1_name, comp2_name)] = 0.0
        
        return correlations
    
    def _build_correlation_matrix(
        self,
        pairwise_correlations: Dict[Tuple[str, str], float]
    ) -> np.ndarray:
        """
        Build full correlation matrix from pairwise correlations
        
        Args:
            pairwise_correlations: Dictionary of pairwise correlations
            
        Returns:
            7x7 correlation matrix
        """
        components = [
            'component_01', 'component_02', 'component_03', 'component_04',
            'component_05', 'component_06', 'component_07'
        ]
        
        n = len(components)
        matrix = np.eye(n)  # Start with identity matrix
        
        # Fill in correlations
        for i, comp1 in enumerate(components):
            for j, comp2 in enumerate(components):
                if i < j:
                    pair = (comp1, comp2)
                    if pair in pairwise_correlations:
                        correlation = pairwise_correlations[pair]
                        matrix[i, j] = correlation
                        matrix[j, i] = correlation  # Symmetric
        
        return matrix
    
    def _calculate_rolling_correlations(
        self,
        aggregated_components: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate rolling correlations with historical data
        
        Args:
            aggregated_components: Aggregated component data
            
        Returns:
            Dictionary of rolling correlation metrics
        """
        # Extract current correlations
        current_correlations = []
        
        for comp1_name, comp2_name in self.component_pairs:
            if comp1_name in aggregated_components and comp2_name in aggregated_components:
                comp1_signals = aggregated_components[comp1_name].get('signals', {})
                comp2_signals = aggregated_components[comp2_name].get('signals', {})
                
                if comp1_signals and comp2_signals:
                    # Simple correlation proxy
                    signal1 = list(comp1_signals.values())[0] if comp1_signals else 0.0
                    signal2 = list(comp2_signals.values())[0] if comp2_signals else 0.0
                    
                    correlation = signal1 * signal2  # Simple product as correlation proxy
                    current_correlations.append(correlation)
        
        # Update history
        self.correlation_history.append(current_correlations)
        
        # Keep only recent history
        window = self.correlation_params['rolling_window']
        if len(self.correlation_history) > window:
            self.correlation_history = self.correlation_history[-window:]
        
        # Calculate rolling metrics
        if self.correlation_history:
            history_array = np.array(self.correlation_history)
            
            # Apply exponential decay to weights
            weights = np.array([
                self.correlation_params['decay_factor'] ** (len(self.correlation_history) - i - 1)
                for i in range(len(self.correlation_history))
            ])
            weights = weights / weights.sum()
            
            # Weighted rolling statistics
            rolling_mean = float(np.average(history_array.mean(axis=1), weights=weights))
            rolling_std = float(np.average(history_array.std(axis=1), weights=weights))
            rolling_trend = float(np.polyfit(range(len(self.correlation_history)), 
                                            history_array.mean(axis=1), 1)[0]) if len(self.correlation_history) > 1 else 0.0
        else:
            rolling_mean = 0.0
            rolling_std = 0.0
            rolling_trend = 0.0
        
        return {
            'rolling_mean': rolling_mean,
            'rolling_std': rolling_std,
            'rolling_trend': rolling_trend,
            'history_length': len(self.correlation_history)
        }
    
    def _detect_correlation_breakdowns(
        self,
        pairwise_correlations: Dict[Tuple[str, str], float],
        rolling_correlations: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Detect correlation breakdowns and anomalies
        
        Args:
            pairwise_correlations: Current pairwise correlations
            rolling_correlations: Rolling correlation metrics
            
        Returns:
            Breakdown analysis results
        """
        # Calculate correlation statistics
        correlation_values = list(pairwise_correlations.values())
        
        if correlation_values:
            current_mean = np.mean(correlation_values)
            current_std = np.std(correlation_values)
            
            # Detect breakdown based on deviation from rolling mean
            if rolling_correlations['rolling_mean'] != 0:
                deviation = abs(current_mean - rolling_correlations['rolling_mean'])
                breakdown_score = deviation / (rolling_correlations['rolling_std'] + 1e-10)
            else:
                breakdown_score = 0.0
            
            # Find most divergent pairs
            divergent_pairs = []
            for pair, correlation in pairwise_correlations.items():
                if abs(correlation) < 0.2:  # Low correlation indicates divergence
                    divergent_pairs.append({
                        'pair': pair,
                        'correlation': correlation
                    })
            
            # Sort by absolute correlation (most divergent first)
            divergent_pairs.sort(key=lambda x: abs(x['correlation']))
            
            # Calculate stability metrics
            stability = 1.0 / (1.0 + current_std)
            
            # Detect regime change
            regime_change_probability = min(1.0, breakdown_score / 3.0)
            
        else:
            current_mean = 0.0
            current_std = 0.0
            breakdown_score = 0.0
            divergent_pairs = []
            stability = 0.5
            regime_change_probability = 0.0
        
        return {
            'breakdown_score': float(breakdown_score),
            'divergent_pairs': divergent_pairs[:3],  # Top 3 divergent pairs
            'stability': float(stability),
            'regime_change_probability': float(regime_change_probability),
            'current_mean_correlation': float(current_mean),
            'current_std_correlation': float(current_std)
        }
    
    def _calculate_coherence_metrics(
        self,
        correlation_matrix: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate system coherence metrics from correlation matrix
        
        Args:
            correlation_matrix: Component correlation matrix
            
        Returns:
            Dictionary of coherence metrics
        """
        # Calculate eigenvalues for principal component analysis
        try:
            eigenvalues = np.linalg.eigvalsh(correlation_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
            
            # First eigenvalue represents system coherence
            if len(eigenvalues) > 0:
                total_variance = np.sum(eigenvalues)
                if total_variance > 0:
                    coherence_index = eigenvalues[0] / total_variance
                    
                    # Calculate effective number of factors
                    cumsum = np.cumsum(eigenvalues) / total_variance
                    effective_factors = np.searchsorted(cumsum, 0.95) + 1
                else:
                    coherence_index = 0.0
                    effective_factors = len(eigenvalues)
            else:
                coherence_index = 0.0
                effective_factors = 0
                
        except np.linalg.LinAlgError:
            logger.warning("Failed to calculate eigenvalues, using defaults")
            coherence_index = 0.5
            effective_factors = 3
            eigenvalues = np.ones(7) / 7
        
        # Calculate average absolute correlation
        n = correlation_matrix.shape[0]
        upper_triangle = np.triu(correlation_matrix, k=1)
        avg_correlation = np.sum(np.abs(upper_triangle)) / (n * (n - 1) / 2)
        
        # Calculate correlation concentration
        abs_corr = np.abs(correlation_matrix)
        np.fill_diagonal(abs_corr, 0)  # Exclude diagonal
        max_correlation = np.max(abs_corr)
        
        # System integration score
        integration_score = avg_correlation * coherence_index
        
        return {
            'coherence_index': float(coherence_index),
            'effective_factors': int(effective_factors),
            'average_correlation': float(avg_correlation),
            'max_correlation': float(max_correlation),
            'integration_score': float(integration_score),
            'eigenvalue_ratio': float(eigenvalues[0] / (eigenvalues[-1] + 1e-10)) if len(eigenvalues) > 0 else 1.0
        }
    
    def get_correlation_features(
        self,
        correlation_analysis: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Extract features from correlation analysis for ML consumption
        
        Args:
            correlation_analysis: Full correlation analysis results
            
        Returns:
            Dictionary of correlation features
        """
        features = {}
        
        # Pairwise correlation features
        pairwise = correlation_analysis['pairwise_correlations']
        if pairwise:
            corr_values = list(pairwise.values())
            features['correlation_mean'] = float(np.mean(corr_values))
            features['correlation_std'] = float(np.std(corr_values))
            features['correlation_min'] = float(np.min(corr_values))
            features['correlation_max'] = float(np.max(corr_values))
        else:
            features['correlation_mean'] = 0.0
            features['correlation_std'] = 0.0
            features['correlation_min'] = 0.0
            features['correlation_max'] = 0.0
        
        # Rolling correlation features
        rolling = correlation_analysis['rolling_correlations']
        features['rolling_correlation_mean'] = rolling['rolling_mean']
        features['rolling_correlation_std'] = rolling['rolling_std']
        features['rolling_correlation_trend'] = rolling['rolling_trend']
        
        # Breakdown features
        breakdown = correlation_analysis['breakdown_analysis']
        features['breakdown_score'] = breakdown['breakdown_score']
        features['correlation_stability'] = breakdown['stability']
        features['regime_change_prob'] = breakdown['regime_change_probability']
        
        # Coherence features
        coherence = correlation_analysis['coherence_metrics']
        features['coherence_index'] = coherence['coherence_index']
        features['effective_factors'] = float(coherence['effective_factors'])
        features['integration_score'] = coherence['integration_score']
        
        return features