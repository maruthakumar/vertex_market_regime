"""
Historical Regime Analyzer

This module analyzes historical market data to identify regime patterns,
transition dynamics, and characteristic features for adaptive regime formation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from collections import defaultdict
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class RegimePattern:
    """Represents a detected regime pattern"""
    regime_id: int
    volatility_range: Tuple[float, float]
    trend_range: Tuple[float, float]
    volume_profile: Dict[str, float]
    average_duration: float
    transition_probabilities: Dict[int, float]
    characteristic_features: List[str]
    stability_score: float


@dataclass
class TransitionDynamics:
    """Represents regime transition dynamics"""
    from_regime: int
    to_regime: int
    probability: float
    average_duration_before: float
    average_duration_after: float
    trigger_conditions: List[str]
    stability_score: float


class HistoricalRegimeAnalyzer:
    """
    Analyzes historical market data to extract regime patterns and dynamics
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize analyzer with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.lookback_days = config.get('historical_lookback_days', 90)
        self.regime_count = config.get('regime_count', 12)
        self.intraday_window = config.get('intraday_window', '5min')
        self.clustering_algorithm = config.get('clustering_algorithm', 'kmeans')
        
        # Feature engineering parameters
        self.feature_windows = [5, 10, 20, 50]  # Multiple timeframes
        self.min_regime_duration = config.get('min_regime_duration', 15)
        
        # Analysis results
        self.regime_patterns: Dict[int, RegimePattern] = {}
        self.transition_matrix: pd.DataFrame = pd.DataFrame()
        self.feature_importance: Dict[str, float] = {}
        
        logger.info(f"HistoricalRegimeAnalyzer initialized with {self.regime_count} regimes")
    
    def analyze_historical_patterns(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract regime patterns from historical data
        
        Args:
            historical_data: DataFrame with market data
            
        Returns:
            Dictionary containing analysis results
        """
        logger.info(f"Analyzing {len(historical_data)} historical data points")
        
        try:
            # Validate and prepare data
            prepared_data = self._prepare_data(historical_data)
            
            # Extract features
            features = self._extract_features(prepared_data)
            
            # Perform clustering to identify regimes
            regime_clusters = self._perform_clustering(features)
            
            # Analyze each regime
            self.regime_patterns = self._analyze_regime_characteristics(
                prepared_data, regime_clusters
            )
            
            # Build transition matrix
            self.transition_matrix = self._build_transition_matrix(regime_clusters)
            
            # Analyze transition dynamics
            transition_dynamics = self._analyze_transition_dynamics(
                prepared_data, regime_clusters
            )
            
            # Calculate stability metrics
            stability_metrics = self._calculate_stability_metrics(regime_clusters)
            
            # Feature importance analysis
            self.feature_importance = self._analyze_feature_importance(
                features, regime_clusters
            )
            
            return {
                'regime_patterns': self.regime_patterns,
                'transition_matrix': self.transition_matrix,
                'transition_dynamics': transition_dynamics,
                'stability_metrics': stability_metrics,
                'feature_importance': self.feature_importance,
                'cluster_quality': self._evaluate_clustering_quality(features, regime_clusters)
            }
            
        except Exception as e:
            logger.error(f"Error in historical pattern analysis: {e}")
            raise
    
    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare and validate historical data
        
        Args:
            data: Raw historical data
            
        Returns:
            Prepared DataFrame
        """
        # Ensure required columns
        required_columns = ['timestamp', 'price', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Sort by timestamp
        data = data.sort_values('timestamp').copy()
        
        # Handle missing values
        data = data.ffill().bfill()
        
        # Add basic technical indicators
        data['returns'] = data['price'].pct_change()
        data['log_returns'] = np.log(data['price'] / data['price'].shift(1))
        data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        
        # Add volatility measures
        for window in [5, 10, 20]:
            data[f'volatility_{window}'] = data['returns'].rolling(window).std()
            data[f'realized_vol_{window}'] = np.sqrt(252) * data[f'volatility_{window}']
        
        # Add price-based features
        data['price_ma_5'] = data['price'].rolling(5).mean()
        data['price_ma_20'] = data['price'].rolling(20).mean()
        data['price_ma_50'] = data['price'].rolling(50).mean()
        
        # Trend indicators
        data['ema_12'] = data['price'].ewm(span=12).mean()
        data['ema_26'] = data['price'].ewm(span=26).mean()
        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        
        # Support/Resistance levels
        data['resistance'] = data['price'].rolling(20).max()
        data['support'] = data['price'].rolling(20).min()
        data['price_position'] = (data['price'] - data['support']) / (data['resistance'] - data['support'])
        
        # Volume indicators
        data['volume_ma'] = data['volume'].rolling(20).mean()
        data['volume_std'] = data['volume'].rolling(20).std()
        data['volume_zscore'] = (data['volume'] - data['volume_ma']) / data['volume_std']
        
        # Drop NaN values from rolling calculations
        data = data.dropna()
        
        logger.info(f"Data prepared: {len(data)} rows with {len(data.columns)} features")
        
        return data
    
    def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract features for clustering
        
        Args:
            data: Prepared market data
            
        Returns:
            Feature matrix
        """
        feature_columns = []
        
        # Price-based features
        price_features = [
            'returns',
            'log_returns',
            'price_position',
            'macd',
            'macd_signal'
        ]
        
        # Volatility features
        volatility_features = [
            f'volatility_{w}' for w in [5, 10, 20]
        ] + [
            f'realized_vol_{w}' for w in [5, 10, 20]
        ]
        
        # Volume features
        volume_features = [
            'volume_ratio',
            'volume_zscore'
        ]
        
        # Trend features
        trend_features = []
        for col in ['price_ma_5', 'price_ma_20', 'price_ma_50']:
            if col in data.columns:
                data[f'{col}_slope'] = data[col].diff(5) / data[col].shift(5)
                trend_features.append(f'{col}_slope')
        
        # Combine all features
        all_features = price_features + volatility_features + volume_features + trend_features
        available_features = [f for f in all_features if f in data.columns]
        
        # Extract feature matrix
        feature_matrix = data[available_features].values
        
        # Remove any remaining NaN values
        if np.any(np.isnan(feature_matrix)):
            logger.warning("Found NaN values in feature matrix, removing...")
            # Replace NaN with column mean
            col_mean = np.nanmean(feature_matrix, axis=0)
            inds = np.where(np.isnan(feature_matrix))
            feature_matrix[inds] = np.take(col_mean, inds[1])
        
        # Standardize features
        scaler = StandardScaler()
        feature_matrix = scaler.fit_transform(feature_matrix)
        
        # Apply PCA if specified
        if len(available_features) > 10:
            pca = PCA(n_components=0.95)  # Keep 95% variance
            feature_matrix = pca.fit_transform(feature_matrix)
            logger.info(f"PCA reduced features from {len(available_features)} to {feature_matrix.shape[1]}")
        
        logger.info(f"Extracted {feature_matrix.shape[1]} features for clustering")
        
        return feature_matrix
    
    def _perform_clustering(self, features: np.ndarray) -> np.ndarray:
        """
        Perform clustering to identify regimes
        
        Args:
            features: Feature matrix
            
        Returns:
            Cluster labels
        """
        logger.info(f"Performing {self.clustering_algorithm} clustering for {self.regime_count} regimes")
        
        if self.clustering_algorithm == 'kmeans':
            # K-means clustering
            kmeans = KMeans(
                n_clusters=self.regime_count,
                random_state=42,
                n_init=10,
                max_iter=300
            )
            labels = kmeans.fit_predict(features)
            
        elif self.clustering_algorithm == 'dbscan':
            # DBSCAN for density-based clustering
            dbscan = DBSCAN(
                eps=0.5,
                min_samples=5,
                metric='euclidean'
            )
            labels = dbscan.fit_predict(features)
            
            # Map DBSCAN labels to regime count
            unique_labels = np.unique(labels[labels != -1])
            if len(unique_labels) != self.regime_count:
                logger.warning(f"DBSCAN found {len(unique_labels)} clusters, remapping to {self.regime_count}")
                # Fallback to k-means
                kmeans = KMeans(n_clusters=self.regime_count, random_state=42)
                labels = kmeans.fit_predict(features)
                
        elif self.clustering_algorithm == 'hierarchical':
            # Agglomerative clustering
            agg_clustering = AgglomerativeClustering(
                n_clusters=self.regime_count,
                linkage='ward'
            )
            labels = agg_clustering.fit_predict(features)
            
        else:
            raise ValueError(f"Unknown clustering algorithm: {self.clustering_algorithm}")
        
        # Post-process labels to ensure minimum duration
        labels = self._enforce_minimum_duration(labels)
        
        logger.info(f"Clustering complete. Regime distribution: {np.bincount(labels)}")
        
        return labels
    
    def _enforce_minimum_duration(self, labels: np.ndarray) -> np.ndarray:
        """
        Enforce minimum regime duration constraint
        
        Args:
            labels: Raw cluster labels
            
        Returns:
            Adjusted labels
        """
        min_samples = self.min_regime_duration  # Assuming 1 sample = 1 minute for simplicity
        
        adjusted_labels = labels.copy()
        current_regime = labels[0]
        regime_start = 0
        
        for i in range(1, len(labels)):
            if labels[i] != current_regime:
                # Check if regime duration is sufficient
                regime_duration = i - regime_start
                
                if regime_duration < min_samples:
                    # Regime too short, merge with previous
                    adjusted_labels[regime_start:i] = adjusted_labels[regime_start - 1] if regime_start > 0 else current_regime
                
                current_regime = labels[i]
                regime_start = i
        
        return adjusted_labels
    
    def _analyze_regime_characteristics(self, data: pd.DataFrame, 
                                      regime_labels: np.ndarray) -> Dict[int, RegimePattern]:
        """
        Analyze characteristics of each regime
        
        Args:
            data: Market data
            regime_labels: Regime assignments
            
        Returns:
            Dictionary of regime patterns
        """
        regime_patterns = {}
        
        # Add regime labels to data
        data = data.copy()
        data['regime'] = regime_labels[:len(data)]
        
        for regime_id in range(self.regime_count):
            regime_data = data[data['regime'] == regime_id]
            
            if len(regime_data) < 10:
                logger.warning(f"Regime {regime_id} has insufficient data points: {len(regime_data)}")
                continue
            
            # Calculate regime characteristics
            volatility_range = (
                regime_data['volatility_20'].quantile(0.1),
                regime_data['volatility_20'].quantile(0.9)
            )
            
            trend_range = (
                regime_data['returns'].mean() - regime_data['returns'].std(),
                regime_data['returns'].mean() + regime_data['returns'].std()
            )
            
            volume_profile = {
                'mean': regime_data['volume'].mean(),
                'std': regime_data['volume'].std(),
                'skew': regime_data['volume'].skew(),
                'relative': regime_data['volume_ratio'].mean()
            }
            
            # Calculate average duration
            regime_durations = []
            current_duration = 0
            
            for i in range(1, len(data)):
                if data['regime'].iloc[i] == regime_id:
                    if data['regime'].iloc[i-1] == regime_id:
                        current_duration += 1
                    else:
                        current_duration = 1
                else:
                    if data['regime'].iloc[i-1] == regime_id and current_duration > 0:
                        regime_durations.append(current_duration)
                        current_duration = 0
            
            average_duration = np.mean(regime_durations) if regime_durations else 0
            
            # Identify characteristic features
            characteristic_features = []
            
            if volatility_range[1] < 0.15:
                characteristic_features.append('low_volatility')
            elif volatility_range[1] > 0.35:
                characteristic_features.append('high_volatility')
            
            if regime_data['returns'].mean() > 0.001:
                characteristic_features.append('bullish_bias')
            elif regime_data['returns'].mean() < -0.001:
                characteristic_features.append('bearish_bias')
            
            if regime_data['volume_ratio'].mean() > 1.2:
                characteristic_features.append('high_volume')
            
            if regime_data['price_position'].mean() > 0.7:
                characteristic_features.append('near_resistance')
            elif regime_data['price_position'].mean() < 0.3:
                characteristic_features.append('near_support')
            
            # Calculate stability score
            stability_score = self._calculate_regime_stability(regime_data)
            
            # Create regime pattern
            regime_patterns[regime_id] = RegimePattern(
                regime_id=regime_id,
                volatility_range=volatility_range,
                trend_range=trend_range,
                volume_profile=volume_profile,
                average_duration=average_duration,
                transition_probabilities={},  # Will be filled by transition matrix
                characteristic_features=characteristic_features,
                stability_score=stability_score
            )
        
        return regime_patterns
    
    def _calculate_regime_stability(self, regime_data: pd.DataFrame) -> float:
        """
        Calculate stability score for a regime
        
        Args:
            regime_data: Data points belonging to the regime
            
        Returns:
            Stability score (0-1)
        """
        if len(regime_data) < 5:
            return 0.0
        
        # Factors contributing to stability
        stability_factors = []
        
        # 1. Volatility consistency
        vol_consistency = 1.0 - regime_data['volatility_20'].std() / regime_data['volatility_20'].mean()
        stability_factors.append(max(0, vol_consistency))
        
        # 2. Return consistency
        return_consistency = 1.0 - abs(regime_data['returns'].std())
        stability_factors.append(max(0, return_consistency))
        
        # 3. Volume consistency
        volume_consistency = 1.0 - regime_data['volume_ratio'].std()
        stability_factors.append(max(0, min(1, volume_consistency)))
        
        # Average stability
        stability_score = np.mean(stability_factors)
        
        return float(np.clip(stability_score, 0, 1))
    
    def _build_transition_matrix(self, regime_labels: np.ndarray) -> pd.DataFrame:
        """
        Build regime transition probability matrix
        
        Args:
            regime_labels: Sequence of regime labels
            
        Returns:
            Transition probability matrix
        """
        # Count transitions
        transition_counts = defaultdict(lambda: defaultdict(int))
        
        for i in range(1, len(regime_labels)):
            from_regime = regime_labels[i-1]
            to_regime = regime_labels[i]
            transition_counts[from_regime][to_regime] += 1
        
        # Convert to probability matrix
        transition_matrix = pd.DataFrame(
            index=range(self.regime_count),
            columns=range(self.regime_count),
            data=0.0
        )
        
        for from_regime in range(self.regime_count):
            total_transitions = sum(transition_counts[from_regime].values())
            if total_transitions > 0:
                for to_regime in range(self.regime_count):
                    count = transition_counts[from_regime][to_regime]
                    transition_matrix.loc[from_regime, to_regime] = count / total_transitions
        
        # Update regime patterns with transition probabilities
        for regime_id, pattern in self.regime_patterns.items():
            pattern.transition_probabilities = dict(transition_matrix.loc[regime_id])
        
        logger.info("Transition matrix calculated")
        
        return transition_matrix
    
    def _analyze_transition_dynamics(self, data: pd.DataFrame, 
                                   regime_labels: np.ndarray) -> List[TransitionDynamics]:
        """
        Analyze dynamics of regime transitions
        
        Args:
            data: Market data
            regime_labels: Regime assignments
            
        Returns:
            List of transition dynamics
        """
        transition_dynamics = []
        data = data.copy()
        data['regime'] = regime_labels[:len(data)]
        
        # Identify all transitions
        transitions = []
        for i in range(1, len(data)):
            if data['regime'].iloc[i] != data['regime'].iloc[i-1]:
                transitions.append({
                    'index': i,
                    'from_regime': data['regime'].iloc[i-1],
                    'to_regime': data['regime'].iloc[i],
                    'timestamp': data.index[i] if isinstance(data.index, pd.DatetimeIndex) else i
                })
        
        # Analyze common transitions
        transition_pairs = defaultdict(list)
        for trans in transitions:
            pair = (trans['from_regime'], trans['to_regime'])
            transition_pairs[pair].append(trans)
        
        # Process each transition pair
        for (from_regime, to_regime), occurrences in transition_pairs.items():
            if len(occurrences) < 3:  # Need minimum occurrences
                continue
            
            # Calculate average durations
            durations_before = []
            durations_after = []
            
            for occ in occurrences:
                idx = occ['index']
                
                # Duration before transition
                before_start = idx - 1
                while before_start > 0 and data['regime'].iloc[before_start] == from_regime:
                    before_start -= 1
                durations_before.append(idx - before_start - 1)
                
                # Duration after transition
                after_end = idx
                while after_end < len(data) - 1 and data['regime'].iloc[after_end] == to_regime:
                    after_end += 1
                durations_after.append(after_end - idx)
            
            # Identify trigger conditions
            trigger_conditions = self._identify_trigger_conditions(
                data, occurrences, from_regime, to_regime
            )
            
            # Calculate stability
            probability = len(occurrences) / len(transitions)
            stability = self._calculate_transition_stability(
                durations_before, durations_after
            )
            
            dynamics = TransitionDynamics(
                from_regime=from_regime,
                to_regime=to_regime,
                probability=probability,
                average_duration_before=np.mean(durations_before),
                average_duration_after=np.mean(durations_after),
                trigger_conditions=trigger_conditions,
                stability_score=stability
            )
            
            transition_dynamics.append(dynamics)
        
        return transition_dynamics
    
    def _identify_trigger_conditions(self, data: pd.DataFrame, 
                                   occurrences: List[Dict],
                                   from_regime: int, to_regime: int) -> List[str]:
        """
        Identify common trigger conditions for transitions
        
        Args:
            data: Market data
            occurrences: List of transition occurrences
            from_regime: Source regime
            to_regime: Target regime
            
        Returns:
            List of trigger conditions
        """
        triggers = []
        
        # Analyze conditions at transition points
        volatility_changes = []
        volume_changes = []
        trend_changes = []
        
        for occ in occurrences:
            idx = occ['index']
            if idx < 5 or idx >= len(data) - 5:
                continue
            
            # Look at changes leading to transition
            vol_before = data['volatility_20'].iloc[idx-5:idx].mean()
            vol_at = data['volatility_20'].iloc[idx]
            volatility_changes.append((vol_at - vol_before) / vol_before)
            
            vol_before = data['volume_ratio'].iloc[idx-5:idx].mean()
            vol_at = data['volume_ratio'].iloc[idx]
            volume_changes.append((vol_at - vol_before) / vol_before)
            
            trend_before = data['macd'].iloc[idx-5:idx].mean()
            trend_at = data['macd'].iloc[idx]
            trend_changes.append(trend_at - trend_before)
        
        # Identify significant patterns
        if volatility_changes:
            avg_vol_change = np.mean(volatility_changes)
            if avg_vol_change > 0.2:
                triggers.append('volatility_spike')
            elif avg_vol_change < -0.2:
                triggers.append('volatility_drop')
        
        if volume_changes:
            avg_volume_change = np.mean(volume_changes)
            if avg_volume_change > 0.3:
                triggers.append('volume_surge')
        
        if trend_changes:
            avg_trend_change = np.mean(trend_changes)
            if avg_trend_change > 0:
                triggers.append('momentum_positive')
            else:
                triggers.append('momentum_negative')
        
        return triggers
    
    def _calculate_transition_stability(self, durations_before: List[float],
                                      durations_after: List[float]) -> float:
        """
        Calculate stability score for transitions
        
        Args:
            durations_before: Durations in source regime
            durations_after: Durations in target regime
            
        Returns:
            Stability score (0-1)
        """
        if not durations_before or not durations_after:
            return 0.0
        
        # Consistency of durations
        before_cv = np.std(durations_before) / np.mean(durations_before) if np.mean(durations_before) > 0 else 1
        after_cv = np.std(durations_after) / np.mean(durations_after) if np.mean(durations_after) > 0 else 1
        
        # Lower CV means more stable
        stability = 1.0 - (before_cv + after_cv) / 2
        
        return float(np.clip(stability, 0, 1))
    
    def _calculate_stability_metrics(self, regime_labels: np.ndarray) -> Dict[str, float]:
        """
        Calculate overall stability metrics
        
        Args:
            regime_labels: Regime assignments
            
        Returns:
            Dictionary of stability metrics
        """
        # Count regime changes
        regime_changes = np.sum(np.diff(regime_labels) != 0)
        total_points = len(regime_labels)
        
        # Calculate regime durations
        durations = []
        current_regime = regime_labels[0]
        current_duration = 1
        
        for i in range(1, len(regime_labels)):
            if regime_labels[i] == current_regime:
                current_duration += 1
            else:
                durations.append(current_duration)
                current_regime = regime_labels[i]
                current_duration = 1
        
        if current_duration > 0:
            durations.append(current_duration)
        
        # Calculate metrics
        metrics = {
            'total_transitions': int(regime_changes),
            'transition_rate': regime_changes / total_points,
            'average_regime_duration': np.mean(durations) if durations else 0,
            'regime_duration_std': np.std(durations) if durations else 0,
            'regime_persistence': 1.0 - (regime_changes / total_points),
            'min_regime_duration': min(durations) if durations else 0,
            'max_regime_duration': max(durations) if durations else 0
        }
        
        return metrics
    
    def _analyze_feature_importance(self, features: np.ndarray, 
                                  regime_labels: np.ndarray) -> Dict[str, float]:
        """
        Analyze feature importance for regime classification
        
        Args:
            features: Feature matrix
            regime_labels: Regime assignments
            
        Returns:
            Dictionary of feature importance scores
        """
        # Use simple variance-based importance
        importance_scores = {}
        
        # Calculate variance ratio for each feature across regimes
        for feature_idx in range(features.shape[1]):
            feature_values = features[:, feature_idx]
            
            # Between-regime variance
            regime_means = []
            for regime in range(self.regime_count):
                regime_mask = regime_labels == regime
                if np.sum(regime_mask) > 0:
                    regime_means.append(np.mean(feature_values[regime_mask]))
            
            if len(regime_means) > 1:
                between_variance = np.var(regime_means)
                total_variance = np.var(feature_values)
                
                if total_variance > 0:
                    importance = between_variance / total_variance
                else:
                    importance = 0.0
            else:
                importance = 0.0
            
            importance_scores[f'feature_{feature_idx}'] = float(importance)
        
        # Normalize scores
        total_importance = sum(importance_scores.values())
        if total_importance > 0:
            importance_scores = {
                k: v / total_importance for k, v in importance_scores.items()
            }
        
        return importance_scores
    
    def _evaluate_clustering_quality(self, features: np.ndarray, 
                                   labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate clustering quality metrics
        
        Args:
            features: Feature matrix
            labels: Cluster labels
            
        Returns:
            Dictionary of quality metrics
        """
        quality_metrics = {}
        
        # Silhouette score
        if len(np.unique(labels)) > 1:
            silhouette = silhouette_score(features, labels)
            quality_metrics['silhouette_score'] = float(silhouette)
        else:
            quality_metrics['silhouette_score'] = 0.0
        
        # Inertia (within-cluster sum of squares) for k-means
        if self.clustering_algorithm == 'kmeans':
            kmeans = KMeans(n_clusters=self.regime_count, random_state=42)
            kmeans.fit(features)
            quality_metrics['inertia'] = float(kmeans.inertia_)
        
        # Cluster sizes
        cluster_sizes = np.bincount(labels)
        quality_metrics['min_cluster_size'] = int(np.min(cluster_sizes))
        quality_metrics['max_cluster_size'] = int(np.max(cluster_sizes))
        quality_metrics['cluster_size_ratio'] = float(np.max(cluster_sizes) / np.min(cluster_sizes))
        
        return quality_metrics
    
    def identify_regime_characteristics(self, data: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        """
        Identify key characteristics of each regime
        
        Args:
            data: Market data with regime labels
            
        Returns:
            Dictionary of regime characteristics
        """
        if 'regime' not in data.columns:
            raise ValueError("Data must contain 'regime' column")
        
        characteristics = {}
        
        for regime_id in range(self.regime_count):
            regime_data = data[data['regime'] == regime_id]
            
            if len(regime_data) < 10:
                continue
            
            # Statistical characteristics
            chars = {
                'count': len(regime_data),
                'percentage': len(regime_data) / len(data) * 100,
                
                # Returns
                'mean_return': float(regime_data['returns'].mean()),
                'std_return': float(regime_data['returns'].std()),
                'sharpe_ratio': float(regime_data['returns'].mean() / regime_data['returns'].std()) if regime_data['returns'].std() > 0 else 0,
                
                # Volatility
                'avg_volatility': float(regime_data['volatility_20'].mean()),
                'volatility_range': (float(regime_data['volatility_20'].min()), 
                                   float(regime_data['volatility_20'].max())),
                
                # Volume
                'avg_volume_ratio': float(regime_data['volume_ratio'].mean()),
                'volume_volatility': float(regime_data['volume_ratio'].std()),
                
                # Trend
                'trend_strength': float(abs(regime_data['macd'].mean())),
                'trend_direction': 'bullish' if regime_data['macd'].mean() > 0 else 'bearish',
                
                # Market structure
                'avg_price_position': float(regime_data['price_position'].mean()),
                'near_support': regime_data['price_position'].mean() < 0.3,
                'near_resistance': regime_data['price_position'].mean() > 0.7,
            }
            
            characteristics[regime_id] = chars
        
        return characteristics
    
    def get_regime_summary(self) -> pd.DataFrame:
        """
        Get summary DataFrame of all regimes
        
        Returns:
            Summary DataFrame
        """
        summary_data = []
        
        for regime_id, pattern in self.regime_patterns.items():
            summary_data.append({
                'regime_id': regime_id,
                'avg_duration': pattern.average_duration,
                'volatility_low': pattern.volatility_range[0],
                'volatility_high': pattern.volatility_range[1],
                'trend_low': pattern.trend_range[0],
                'trend_high': pattern.trend_range[1],
                'volume_mean': pattern.volume_profile['mean'],
                'stability_score': pattern.stability_score,
                'features': ', '.join(pattern.characteristic_features)
            })
        
        return pd.DataFrame(summary_data)


# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'historical_lookback_days': 90,
        'regime_count': 12,
        'intraday_window': '5min',
        'clustering_algorithm': 'kmeans'
    }
    
    # Create analyzer
    analyzer = HistoricalRegimeAnalyzer(config)
    
    # Generate sample data
    dates = pd.date_range('2024-01-01', periods=10000, freq='5min')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'price': 100 + np.random.randn(10000).cumsum() * 0.5,
        'volume': np.random.randint(1000, 10000, 10000)
    })
    
    # Analyze patterns
    results = analyzer.analyze_historical_patterns(sample_data)
    
    print("\nRegime Summary:")
    print(analyzer.get_regime_summary())
    
    print("\nTransition Matrix:")
    print(results['transition_matrix'])
    
    print("\nStability Metrics:")
    for k, v in results['stability_metrics'].items():
        print(f"{k}: {v:.4f}")