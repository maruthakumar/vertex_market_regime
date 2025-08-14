"""
Volume-OI Divergence Analyzer Module

Analyzes divergence patterns between volume flows and OI changes to detect:
- Institutional positioning vs retail activity
- Smart money accumulation/distribution
- Hidden buying/selling pressure
- Divergence-based regime transitions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class DivergenceType(Enum):
    """Types of volume-OI divergence patterns"""
    BULLISH_DIVERGENCE = "bullish_divergence"  # Volume up, OI down - short covering
    BEARISH_DIVERGENCE = "bearish_divergence"  # Volume up, OI up with price down
    ACCUMULATION = "accumulation"  # Low volume, high OI increase
    DISTRIBUTION = "distribution"  # High volume, OI decrease
    NO_DIVERGENCE = "no_divergence"
    COMPLEX_DIVERGENCE = "complex_divergence"


@dataclass
class DivergenceMetrics:
    """Volume-OI divergence analysis metrics"""
    volume_oi_correlation: float
    ce_divergence_score: float
    pe_divergence_score: float
    net_divergence_score: float
    divergence_type: DivergenceType
    divergence_strength: float
    institutional_activity_score: float
    retail_activity_score: float
    smart_money_indicator: float
    hidden_flow_score: float
    divergence_persistence: float
    regime_transition_probability: float
    volume_concentration: float
    oi_concentration: float
    divergence_volatility: float


class VolumeOIDivergenceAnalyzer:
    """
    Analyzes divergence patterns between volume and OI flows
    """
    
    def __init__(self, correlation_windows: List[int] = None):
        """
        Initialize Volume-OI Divergence Analyzer
        
        Args:
            correlation_windows: Windows for correlation analysis
        """
        self.correlation_windows = correlation_windows or [5, 10, 20, 50]
        self.divergence_history = []
        logger.info(f"Initialized VolumeOIDivergenceAnalyzer with windows: {self.correlation_windows}")
    
    def analyze_divergence_patterns(self, df: pd.DataFrame) -> Dict[str, DivergenceMetrics]:
        """
        Analyze volume-OI divergence across multiple timeframes
        
        Args:
            df: Production data with volume and OI columns
            
        Returns:
            Divergence metrics by timeframe
        """
        metrics_by_window = {}
        
        for window in self.correlation_windows:
            metrics = self._calculate_window_divergence(df, window)
            metrics_by_window[f"window_{window}"] = metrics
        
        # Calculate composite metrics
        composite_metrics = self._calculate_composite_divergence(metrics_by_window)
        metrics_by_window["composite"] = composite_metrics
        
        return metrics_by_window
    
    def _calculate_window_divergence(self, df: pd.DataFrame, window: int) -> DivergenceMetrics:
        """Calculate divergence metrics for a specific window"""
        
        # Calculate volume-OI correlation
        volume_oi_corr = self._calculate_volume_oi_correlation(df, window)
        
        # Calculate CE divergence
        ce_divergence = self._calculate_ce_divergence(df, window)
        
        # Calculate PE divergence
        pe_divergence = self._calculate_pe_divergence(df, window)
        
        # Net divergence
        net_divergence = (ce_divergence + pe_divergence) / 2
        
        # Classify divergence type
        divergence_type = self._classify_divergence_type(df, window, net_divergence)
        
        # Calculate divergence strength
        divergence_strength = self._calculate_divergence_strength(ce_divergence, pe_divergence, volume_oi_corr)
        
        # Institutional vs retail activity
        institutional_score = self._calculate_institutional_activity(df, window)
        retail_score = 1.0 - institutional_score  # Inverse relationship
        
        # Smart money indicator
        smart_money = self._calculate_smart_money_indicator(df, window, divergence_type)
        
        # Hidden flow detection
        hidden_flow = self._detect_hidden_flows(df, window)
        
        # Divergence persistence
        persistence = self._calculate_divergence_persistence(net_divergence)
        
        # Regime transition probability
        regime_transition = self._calculate_regime_transition_probability(divergence_type, divergence_strength)
        
        # Concentration metrics
        volume_concentration = self._calculate_volume_concentration(df, window)
        oi_concentration = self._calculate_oi_concentration(df, window)
        
        # Divergence volatility
        divergence_vol = self._calculate_divergence_volatility(df, window)
        
        return DivergenceMetrics(
            volume_oi_correlation=volume_oi_corr,
            ce_divergence_score=ce_divergence,
            pe_divergence_score=pe_divergence,
            net_divergence_score=net_divergence,
            divergence_type=divergence_type,
            divergence_strength=divergence_strength,
            institutional_activity_score=institutional_score,
            retail_activity_score=retail_score,
            smart_money_indicator=smart_money,
            hidden_flow_score=hidden_flow,
            divergence_persistence=persistence,
            regime_transition_probability=regime_transition,
            volume_concentration=volume_concentration,
            oi_concentration=oi_concentration,
            divergence_volatility=divergence_vol
        )
    
    def _calculate_volume_oi_correlation(self, df: pd.DataFrame, window: int) -> float:
        """Calculate correlation between volume and OI changes"""
        
        if 'ce_volume' not in df.columns or 'ce_oi' not in df.columns:
            return 0.0
        
        # Calculate changes
        volume_changes = (df['ce_volume'] + df.get('pe_volume', 0)).diff(window)
        oi_changes = (df['ce_oi'] + df.get('pe_oi', 0)).diff(window)
        
        # Remove NaN values
        valid_mask = ~(volume_changes.isna() | oi_changes.isna())
        
        if valid_mask.sum() < 2:
            return 0.0
        
        # Calculate correlation
        correlation = volume_changes[valid_mask].corr(oi_changes[valid_mask])
        
        return correlation if not np.isnan(correlation) else 0.0
    
    def _calculate_ce_divergence(self, df: pd.DataFrame, window: int) -> float:
        """Calculate CE side divergence score"""
        
        if 'ce_volume' not in df.columns or 'ce_oi' not in df.columns:
            return 0.0
        
        # Normalize volume and OI changes
        volume_change = df['ce_volume'].pct_change(window).mean()
        oi_change = df['ce_oi'].pct_change(window).mean()
        
        # Divergence occurs when they move in opposite directions
        if np.sign(volume_change) != np.sign(oi_change):
            divergence = abs(volume_change - oi_change)
        else:
            # Or when magnitudes are significantly different
            divergence = abs(abs(volume_change) - abs(oi_change))
        
        # Normalize to [-1, 1]
        return np.tanh(divergence)
    
    def _calculate_pe_divergence(self, df: pd.DataFrame, window: int) -> float:
        """Calculate PE side divergence score"""
        
        if 'pe_volume' not in df.columns or 'pe_oi' not in df.columns:
            return 0.0
        
        volume_change = df['pe_volume'].pct_change(window).mean()
        oi_change = df['pe_oi'].pct_change(window).mean()
        
        if np.sign(volume_change) != np.sign(oi_change):
            divergence = abs(volume_change - oi_change)
        else:
            divergence = abs(abs(volume_change) - abs(oi_change))
        
        return np.tanh(divergence)
    
    def _classify_divergence_type(self, df: pd.DataFrame, window: int, 
                                 net_divergence: float) -> DivergenceType:
        """Classify the type of divergence pattern"""
        
        if abs(net_divergence) < 0.1:
            return DivergenceType.NO_DIVERGENCE
        
        # Check for specific patterns
        if 'ce_volume' in df.columns and 'ce_oi' in df.columns:
            volume_trend = df['ce_volume'].diff(window).mean()
            oi_trend = df['ce_oi'].diff(window).mean()
            
            # Price trend if available
            price_trend = 0
            if 'spot' in df.columns:
                price_trend = df['spot'].pct_change(window).mean()
            
            # Bullish divergence: Volume up, OI down (short covering)
            if volume_trend > 0 and oi_trend < 0 and price_trend > 0:
                return DivergenceType.BULLISH_DIVERGENCE
            
            # Bearish divergence: Volume up, OI up, price down
            if volume_trend > 0 and oi_trend > 0 and price_trend < 0:
                return DivergenceType.BEARISH_DIVERGENCE
            
            # Accumulation: Low volume, high OI increase
            if abs(volume_trend) < df['ce_volume'].std() * 0.5 and oi_trend > df['ce_oi'].std():
                return DivergenceType.ACCUMULATION
            
            # Distribution: High volume, OI decrease
            if volume_trend > df['ce_volume'].std() and oi_trend < 0:
                return DivergenceType.DISTRIBUTION
        
        return DivergenceType.COMPLEX_DIVERGENCE
    
    def _calculate_divergence_strength(self, ce_div: float, pe_div: float, 
                                      correlation: float) -> float:
        """Calculate overall divergence strength"""
        
        # Average divergence
        avg_divergence = (abs(ce_div) + abs(pe_div)) / 2
        
        # Inverse correlation adds to strength
        correlation_factor = 1.0 - abs(correlation)
        
        # Combined strength
        strength = avg_divergence * 0.6 + correlation_factor * 0.4
        
        return np.clip(strength, 0.0, 1.0)
    
    def _calculate_institutional_activity(self, df: pd.DataFrame, window: int) -> float:
        """Calculate institutional activity score based on volume-OI patterns"""
        
        institutional_score = 0.0
        factors = []
        
        # Large volume with OI changes indicates institutional
        if 'ce_volume' in df.columns and 'ce_oi' in df.columns:
            volume_mean = df['ce_volume'].rolling(window).mean().mean()
            volume_std = df['ce_volume'].rolling(window).std().mean()
            
            # Check for volume spikes
            volume_spikes = ((df['ce_volume'] > volume_mean + 2 * volume_std).sum() / len(df))
            factors.append(volume_spikes)
            
            # OI concentration
            oi_concentration = df['ce_oi'].std() / (df['ce_oi'].mean() + 1)
            factors.append(min(oi_concentration, 1.0))
        
        # Block trade detection (large single trades)
        if 'ce_volume' in df.columns:
            volume_percentile = df['ce_volume'].quantile(0.95)
            block_trades = (df['ce_volume'] > volume_percentile).sum() / len(df)
            factors.append(block_trades * 2)  # Weight block trades higher
        
        if factors:
            institutional_score = np.mean(factors)
        
        return np.clip(institutional_score, 0.0, 1.0)
    
    def _calculate_smart_money_indicator(self, df: pd.DataFrame, window: int,
                                        divergence_type: DivergenceType) -> float:
        """Calculate smart money activity indicator"""
        
        smart_money_score = 0.0
        
        # Accumulation and distribution patterns indicate smart money
        if divergence_type == DivergenceType.ACCUMULATION:
            smart_money_score = 0.7
        elif divergence_type == DivergenceType.DISTRIBUTION:
            smart_money_score = 0.6
        
        # Low volume with significant OI changes
        if 'ce_volume' in df.columns and 'ce_oi' in df.columns:
            volume_percentile = df['ce_volume'].rolling(window).mean().quantile(0.3)
            oi_change = abs(df['ce_oi'].pct_change(window).mean())
            
            if df['ce_volume'].mean() < volume_percentile and oi_change > 0.05:
                smart_money_score += 0.3
        
        return np.clip(smart_money_score, 0.0, 1.0)
    
    def _detect_hidden_flows(self, df: pd.DataFrame, window: int) -> float:
        """Detect hidden institutional flows"""
        
        hidden_flow_score = 0.0
        
        if 'ce_volume' not in df.columns or 'ce_oi' not in df.columns:
            return 0.0
        
        # Calculate volume-weighted OI changes
        volume_weights = df['ce_volume'] / df['ce_volume'].sum()
        oi_changes = df['ce_oi'].diff()
        
        weighted_oi_change = (volume_weights * oi_changes).sum()
        unweighted_oi_change = oi_changes.mean()
        
        # Hidden flow when weighted and unweighted differ significantly
        if abs(weighted_oi_change - unweighted_oi_change) > abs(unweighted_oi_change) * 0.5:
            hidden_flow_score = abs(weighted_oi_change - unweighted_oi_change) / (abs(unweighted_oi_change) + 1)
        
        # Check for consistent but small OI changes (stealth accumulation)
        oi_consistency = 1.0 - (df['ce_oi'].diff().std() / (abs(df['ce_oi'].diff().mean()) + 1))
        if oi_consistency > 0.7:
            hidden_flow_score += 0.3
        
        return np.clip(hidden_flow_score, 0.0, 1.0)
    
    def _calculate_divergence_persistence(self, divergence_score: float) -> float:
        """Calculate how persistent the divergence pattern is"""
        
        self.divergence_history.append(divergence_score)
        
        if len(self.divergence_history) < 3:
            return 0.0
        
        # Keep recent history
        if len(self.divergence_history) > 20:
            self.divergence_history = self.divergence_history[-20:]
        
        # Check consistency of divergence direction
        recent_divergences = self.divergence_history[-5:]
        signs = [np.sign(d) for d in recent_divergences]
        
        # Persistence is high if signs are consistent
        sign_consistency = abs(np.mean(signs))
        
        # Also check magnitude consistency
        magnitude_consistency = 1.0 - (np.std(recent_divergences) / (np.mean(np.abs(recent_divergences)) + 0.01))
        
        persistence = sign_consistency * 0.6 + magnitude_consistency * 0.4
        
        return np.clip(persistence, 0.0, 1.0)
    
    def _calculate_regime_transition_probability(self, divergence_type: DivergenceType,
                                                strength: float) -> float:
        """Calculate probability of regime transition based on divergence"""
        
        base_probability = 0.0
        
        # Certain divergence types indicate higher transition probability
        if divergence_type == DivergenceType.BULLISH_DIVERGENCE:
            base_probability = 0.6
        elif divergence_type == DivergenceType.BEARISH_DIVERGENCE:
            base_probability = 0.6
        elif divergence_type == DivergenceType.COMPLEX_DIVERGENCE:
            base_probability = 0.4
        elif divergence_type in [DivergenceType.ACCUMULATION, DivergenceType.DISTRIBUTION]:
            base_probability = 0.5
        else:
            base_probability = 0.1
        
        # Adjust by strength
        transition_probability = base_probability * (0.5 + 0.5 * strength)
        
        return np.clip(transition_probability, 0.0, 1.0)
    
    def _calculate_volume_concentration(self, df: pd.DataFrame, window: int) -> float:
        """Calculate volume concentration metric"""
        
        if 'ce_volume' not in df.columns:
            return 0.0
        
        # Calculate Herfindahl index for volume concentration
        volume_window = df['ce_volume'].rolling(window)
        total_volume = volume_window.sum().mean()
        
        if total_volume == 0:
            return 0.0
        
        # Volume concentration in top periods
        volume_sorted = df['ce_volume'].nlargest(int(len(df) * 0.1))
        top_concentration = volume_sorted.sum() / df['ce_volume'].sum() if df['ce_volume'].sum() > 0 else 0
        
        return top_concentration
    
    def _calculate_oi_concentration(self, df: pd.DataFrame, window: int) -> float:
        """Calculate OI concentration metric"""
        
        if 'ce_oi' not in df.columns:
            return 0.0
        
        # Check concentration across strikes if available
        if 'strike' in df.columns:
            strike_oi = df.groupby('strike')['ce_oi'].sum()
            total_oi = strike_oi.sum()
            
            if total_oi > 0:
                # Calculate Herfindahl index
                shares = strike_oi / total_oi
                herfindahl = (shares ** 2).sum()
                return herfindahl
        
        # Fallback to time-based concentration
        oi_std = df['ce_oi'].std()
        oi_mean = df['ce_oi'].mean()
        
        if oi_mean > 0:
            return oi_std / oi_mean
        
        return 0.0
    
    def _calculate_divergence_volatility(self, df: pd.DataFrame, window: int) -> float:
        """Calculate volatility of divergence patterns"""
        
        if 'ce_volume' not in df.columns or 'ce_oi' not in df.columns:
            return 0.0
        
        # Calculate rolling divergence
        volume_changes = df['ce_volume'].pct_change(window)
        oi_changes = df['ce_oi'].pct_change(window)
        
        divergence_series = volume_changes - oi_changes
        divergence_vol = divergence_series.std()
        
        # Normalize
        return np.tanh(divergence_vol * 10)
    
    def _calculate_composite_divergence(self, window_metrics: Dict[str, DivergenceMetrics]) -> DivergenceMetrics:
        """Calculate composite divergence metrics across all windows"""
        
        # Extract metrics from all windows
        correlations = []
        net_divergences = []
        strengths = []
        
        for key, metrics in window_metrics.items():
            if key != "composite":
                correlations.append(metrics.volume_oi_correlation)
                net_divergences.append(metrics.net_divergence_score)
                strengths.append(metrics.divergence_strength)
        
        # Weighted average (shorter windows get higher weight)
        weights = [0.4, 0.3, 0.2, 0.1][:len(correlations)]
        weights = weights / np.sum(weights)
        
        avg_correlation = np.average(correlations, weights=weights)
        avg_divergence = np.average(net_divergences, weights=weights)
        avg_strength = np.average(strengths, weights=weights)
        
        # Get most common divergence type
        divergence_types = [m.divergence_type for _, m in window_metrics.items() if _ != "composite"]
        most_common_type = max(set(divergence_types), key=divergence_types.count)
        
        # Create composite metrics
        return DivergenceMetrics(
            volume_oi_correlation=avg_correlation,
            ce_divergence_score=avg_divergence * 0.5,
            pe_divergence_score=avg_divergence * 0.5,
            net_divergence_score=avg_divergence,
            divergence_type=most_common_type,
            divergence_strength=avg_strength,
            institutional_activity_score=np.mean([m.institutional_activity_score for _, m in window_metrics.items() if _ != "composite"]),
            retail_activity_score=np.mean([m.retail_activity_score for _, m in window_metrics.items() if _ != "composite"]),
            smart_money_indicator=np.mean([m.smart_money_indicator for _, m in window_metrics.items() if _ != "composite"]),
            hidden_flow_score=np.mean([m.hidden_flow_score for _, m in window_metrics.items() if _ != "composite"]),
            divergence_persistence=self._calculate_divergence_persistence(avg_divergence),
            regime_transition_probability=np.mean([m.regime_transition_probability for _, m in window_metrics.items() if _ != "composite"]),
            volume_concentration=np.mean([m.volume_concentration for _, m in window_metrics.items() if _ != "composite"]),
            oi_concentration=np.mean([m.oi_concentration for _, m in window_metrics.items() if _ != "composite"]),
            divergence_volatility=np.std(net_divergences)
        )
    
    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract divergence features for Component 3
        
        Args:
            df: Production data
            
        Returns:
            Array of divergence features
        """
        features = []
        
        # Calculate divergence metrics
        metrics = self.analyze_divergence_patterns(df)
        
        # Extract features from composite metrics
        composite = metrics.get("composite")
        if composite:
            features.extend([
                composite.volume_oi_correlation,
                composite.net_divergence_score,
                composite.divergence_strength,
                composite.institutional_activity_score,
                composite.retail_activity_score,
                composite.smart_money_indicator,
                composite.hidden_flow_score,
                composite.divergence_persistence,
                composite.regime_transition_probability,
                composite.volume_concentration,
                composite.oi_concentration,
                composite.divergence_volatility
            ])
        
        return np.array(features)