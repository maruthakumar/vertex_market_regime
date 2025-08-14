"""
Support/Resistance Feature Engineering Framework
Component 7: 72-Feature Extraction for ML Consumption
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class SupportResistanceFeatures:
    """Container for 72 support/resistance features"""
    
    # Mathematical Level Detection Features (36 features)
    level_prices: np.ndarray  # Top 10 level prices
    level_strengths: np.ndarray  # Strength scores for each level
    level_ages: np.ndarray  # Age in periods for each level
    level_validation_counts: np.ndarray  # Number of validations per level
    level_distances: np.ndarray  # Distance from current price
    level_types: np.ndarray  # Type encoding (support=0, resistance=1)
    
    # Dynamic Learning Features (36 features)
    method_performance_scores: np.ndarray  # Performance by detection method (10 methods)
    weight_adaptations: np.ndarray  # Current adaptive weights (10 weights)
    historical_accuracy_rates: np.ndarray  # Accuracy rates by timeframe (8 timeframes)
    cross_validation_metrics: np.ndarray  # Cross-validation scores (8 metrics)
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert to 72-dimensional feature vector for ML"""
        features = []
        
        # Mathematical Level Detection (36 features)
        features.extend(self.level_prices[:10])  # 10 features
        features.extend(self.level_strengths[:10])  # 10 features
        features.extend(self.level_ages[:10])  # 10 features
        features.extend(self.level_validation_counts[:6])  # 6 features
        
        # Dynamic Learning (36 features)
        features.extend(self.level_distances[:10])  # 10 features
        features.extend(self.level_types[:6])  # 6 features
        features.extend(self.method_performance_scores[:10])  # 10 features
        features.extend(self.weight_adaptations[:10])  # 10 features
        
        return np.array(features)
    
    def to_extended_feature_vector(self, advanced_features: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Convert to 120-dimensional extended feature vector for ML
        Includes base 72 features + 48 advanced pattern features
        """
        features = []
        
        # Base 72 features
        features.extend(self.to_feature_vector())
        
        # Add advanced features if provided (48 features)
        if advanced_features is not None and len(advanced_features) == 48:
            features.extend(advanced_features)
        else:
            # Add zeros if advanced features not available
            features.extend(np.zeros(48))
        
        return np.array(features)


class SupportResistanceFeatureEngine:
    """
    Main Feature Engineering Engine for Support/Resistance Component
    Generates 72 raw features for ML consumption without hard-coded classifications
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feature engine with configuration
        
        Args:
            config: Configuration dictionary with component parameters
        """
        self.config = config
        self.processing_budget_ms = config.get("processing_budget_ms", 150)
        self.memory_budget_mb = config.get("memory_budget_mb", 220)
        
        # Feature engineering parameters
        self.max_levels = config.get("max_levels", 10)
        self.proximity_threshold = config.get("proximity_threshold", 0.002)  # 0.2%
        self.min_touches = config.get("min_touches", 2)
        self.lookback_periods = config.get("lookback_periods", 252)
        
        # Detection method weights (initial)
        self.method_weights = {
            "component_1_straddle": 0.15,
            "component_3_cumulative": 0.15,
            "daily_pivots": 0.10,
            "weekly_pivots": 0.10,
            "monthly_pivots": 0.08,
            "volume_profile": 0.10,
            "moving_averages": 0.08,
            "fibonacci": 0.08,
            "psychological": 0.08,
            "historical": 0.08
        }
        
        # Multi-timeframe settings
        self.timeframes = ["1min", "3min", "5min", "15min", "30min", "60min", "daily", "weekly"]
        
        # Performance tracking
        self.method_performance = {method: [] for method in self.method_weights.keys()}
        self.level_history = []
        
        logger.info(f"Initialized SupportResistanceFeatureEngine with {len(self.method_weights)} detection methods")
    
    def extract_features(
        self,
        market_data: pd.DataFrame,
        straddle_data: Optional[pd.DataFrame] = None,
        cumulative_data: Optional[pd.DataFrame] = None,
        component_1_data: Optional[Dict[str, Any]] = None,
        component_3_data: Optional[Dict[str, Any]] = None
    ) -> SupportResistanceFeatures:
        """
        Extract 72 support/resistance features from market data
        
        Args:
            market_data: DataFrame with OHLCV data
            straddle_data: Optional straddle price data from Component 1
            cumulative_data: Optional cumulative ATM±7 data from Component 3
            component_1_data: Optional Component 1 analysis results
            component_3_data: Optional Component 3 analysis results
            
        Returns:
            SupportResistanceFeatures object with 72 features
        """
        # Detect levels from multiple sources
        all_levels = self._detect_all_levels(
            market_data, straddle_data, cumulative_data,
            component_1_data, component_3_data
        )
        
        # Rank and filter top levels
        ranked_levels = self._rank_levels(all_levels, market_data)
        
        # Calculate level metrics
        level_metrics = self._calculate_level_metrics(ranked_levels, market_data)
        
        # Calculate dynamic learning features
        learning_features = self._calculate_learning_features(ranked_levels, market_data)
        
        # Create feature object
        features = SupportResistanceFeatures(
            level_prices=level_metrics["prices"],
            level_strengths=level_metrics["strengths"],
            level_ages=level_metrics["ages"],
            level_validation_counts=level_metrics["validations"],
            level_distances=level_metrics["distances"],
            level_types=level_metrics["types"],
            method_performance_scores=learning_features["performance_scores"],
            weight_adaptations=learning_features["weight_adaptations"],
            historical_accuracy_rates=learning_features["accuracy_rates"],
            cross_validation_metrics=learning_features["cross_validation"]
        )
        
        return features
    
    def _detect_all_levels(
        self,
        market_data: pd.DataFrame,
        straddle_data: Optional[pd.DataFrame],
        cumulative_data: Optional[pd.DataFrame],
        component_1_data: Optional[Dict[str, Any]],
        component_3_data: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect support/resistance levels from all sources
        
        Returns:
            List of detected levels with metadata
        """
        all_levels = []
        
        # Component 1 straddle-based levels
        if straddle_data is not None and component_1_data is not None:
            straddle_levels = self._detect_straddle_levels(straddle_data, component_1_data)
            all_levels.extend(straddle_levels)
        
        # Component 3 cumulative-based levels
        if cumulative_data is not None and component_3_data is not None:
            cumulative_levels = self._detect_cumulative_levels(cumulative_data, component_3_data)
            all_levels.extend(cumulative_levels)
        
        # Traditional underlying price levels
        underlying_levels = self._detect_underlying_levels(market_data)
        all_levels.extend(underlying_levels)
        
        return all_levels
    
    def _detect_straddle_levels(
        self,
        straddle_data: pd.DataFrame,
        component_1_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Detect levels from Component 1 straddle data
        
        Returns:
            List of straddle-based levels
        """
        levels = []
        
        # Extract ATM/ITM1/OTM1 straddle levels
        for straddle_type in ["atm", "itm1", "otm1"]:
            if straddle_type in component_1_data:
                straddle_prices = component_1_data[straddle_type].get("prices", [])
                
                # Find local extrema in straddle prices
                if len(straddle_prices) > 20:
                    extrema = self._find_extrema(straddle_prices)
                    
                    for idx, is_max in extrema:
                        levels.append({
                            "price": straddle_prices[idx],
                            "source": f"component_1_{straddle_type}",
                            "type": "resistance" if is_max else "support",
                            "strength": 1.0,
                            "timestamp": idx,
                            "method": "component_1_straddle"
                        })
        
        return levels
    
    def _detect_cumulative_levels(
        self,
        cumulative_data: pd.DataFrame,
        component_3_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Detect levels from Component 3 cumulative ATM±7 data
        
        Returns:
            List of cumulative-based levels
        """
        levels = []
        
        # Extract cumulative CE/PE levels
        if "cumulative_ce" in component_3_data:
            ce_levels = component_3_data["cumulative_ce"]
            for level in ce_levels:
                levels.append({
                    "price": level["price"],
                    "source": "component_3_ce",
                    "type": "resistance",  # CE accumulation typically forms resistance
                    "strength": level.get("strength", 1.0),
                    "timestamp": level.get("timestamp", 0),
                    "method": "component_3_cumulative"
                })
        
        if "cumulative_pe" in component_3_data:
            pe_levels = component_3_data["cumulative_pe"]
            for level in pe_levels:
                levels.append({
                    "price": level["price"],
                    "source": "component_3_pe",
                    "type": "support",  # PE accumulation typically forms support
                    "strength": level.get("strength", 1.0),
                    "timestamp": level.get("timestamp", 0),
                    "method": "component_3_cumulative"
                })
        
        return levels
    
    def _detect_underlying_levels(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect traditional support/resistance from underlying price
        
        Returns:
            List of underlying price-based levels
        """
        levels = []
        
        # Daily pivot points
        daily_pivots = self._calculate_pivot_points(market_data, "daily")
        levels.extend(daily_pivots)
        
        # Volume profile levels
        volume_levels = self._calculate_volume_profile_levels(market_data)
        levels.extend(volume_levels)
        
        # Moving average levels
        ma_levels = self._calculate_ma_levels(market_data)
        levels.extend(ma_levels)
        
        # Psychological levels
        psych_levels = self._calculate_psychological_levels(market_data)
        levels.extend(psych_levels)
        
        return levels
    
    def _calculate_pivot_points(
        self,
        market_data: pd.DataFrame,
        timeframe: str
    ) -> List[Dict[str, Any]]:
        """
        Calculate pivot point levels
        
        Returns:
            List of pivot levels
        """
        levels = []
        
        if len(market_data) > 0:
            # Standard pivot calculation
            high = market_data["high"].iloc[-1]
            low = market_data["low"].iloc[-1]
            close = market_data["close"].iloc[-1]
            
            pivot = (high + low + close) / 3
            
            # Resistance levels
            r1 = 2 * pivot - low
            r2 = pivot + (high - low)
            r3 = high + 2 * (pivot - low)
            
            # Support levels
            s1 = 2 * pivot - high
            s2 = pivot - (high - low)
            s3 = low - 2 * (high - pivot)
            
            # Add levels
            levels.extend([
                {"price": r3, "source": f"{timeframe}_pivot", "type": "resistance", "strength": 0.6, "method": f"{timeframe}_pivots"},
                {"price": r2, "source": f"{timeframe}_pivot", "type": "resistance", "strength": 0.8, "method": f"{timeframe}_pivots"},
                {"price": r1, "source": f"{timeframe}_pivot", "type": "resistance", "strength": 1.0, "method": f"{timeframe}_pivots"},
                {"price": pivot, "source": f"{timeframe}_pivot", "type": "neutral", "strength": 1.0, "method": f"{timeframe}_pivots"},
                {"price": s1, "source": f"{timeframe}_pivot", "type": "support", "strength": 1.0, "method": f"{timeframe}_pivots"},
                {"price": s2, "source": f"{timeframe}_pivot", "type": "support", "strength": 0.8, "method": f"{timeframe}_pivots"},
                {"price": s3, "source": f"{timeframe}_pivot", "type": "support", "strength": 0.6, "method": f"{timeframe}_pivots"}
            ])
        
        return levels
    
    def _calculate_volume_profile_levels(
        self,
        market_data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Calculate volume profile-based support/resistance levels
        
        Returns:
            List of volume-based levels
        """
        levels = []
        
        if "volume" in market_data.columns and len(market_data) > 20:
            # Calculate volume-weighted levels
            price_volume = market_data["close"] * market_data["volume"]
            total_volume = market_data["volume"].sum()
            
            if total_volume > 0:
                vwap = price_volume.sum() / total_volume
                
                # Find high volume nodes
                volume_quantiles = market_data["volume"].quantile([0.7, 0.8, 0.9])
                high_volume_prices = market_data[market_data["volume"] > volume_quantiles[0.8]]["close"].values
                
                for price in high_volume_prices[:5]:  # Top 5 high volume prices
                    levels.append({
                        "price": price,
                        "source": "volume_profile",
                        "type": "neutral",
                        "strength": 0.8,
                        "method": "volume_profile"
                    })
        
        return levels
    
    def _calculate_ma_levels(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Calculate moving average-based support/resistance levels
        
        Returns:
            List of MA-based levels
        """
        levels = []
        
        if len(market_data) > 200:
            # Calculate various MAs
            ma_periods = [20, 50, 100, 200]
            current_price = market_data["close"].iloc[-1]
            
            for period in ma_periods:
                if len(market_data) >= period:
                    ma_value = market_data["close"].rolling(period).mean().iloc[-1]
                    
                    # Determine if MA is support or resistance
                    level_type = "support" if current_price > ma_value else "resistance"
                    
                    levels.append({
                        "price": ma_value,
                        "source": f"ma_{period}",
                        "type": level_type,
                        "strength": 0.7,
                        "method": "moving_averages"
                    })
        
        return levels
    
    def _calculate_psychological_levels(
        self,
        market_data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Calculate psychological and round number levels
        
        Returns:
            List of psychological levels
        """
        levels = []
        
        if len(market_data) > 0:
            current_price = market_data["close"].iloc[-1]
            
            # Round number intervals based on price magnitude
            if current_price > 10000:
                interval = 100
            elif current_price > 1000:
                interval = 50
            else:
                interval = 10
            
            # Find nearby round numbers
            base = int(current_price / interval) * interval
            
            for i in range(-3, 4):
                level_price = base + (i * interval)
                if level_price > 0:
                    level_type = "support" if level_price < current_price else "resistance"
                    
                    levels.append({
                        "price": level_price,
                        "source": "psychological",
                        "type": level_type,
                        "strength": 0.5 if abs(i) > 1 else 0.7,
                        "method": "psychological"
                    })
        
        return levels
    
    def _find_extrema(self, prices: List[float], window: int = 5) -> List[Tuple[int, bool]]:
        """
        Find local extrema in price series
        
        Args:
            prices: Price series
            window: Window size for extrema detection
            
        Returns:
            List of (index, is_maximum) tuples
        """
        extrema = []
        prices_array = np.array(prices)
        
        for i in range(window, len(prices) - window):
            window_slice = prices_array[i - window:i + window + 1]
            
            if prices_array[i] == np.max(window_slice):
                extrema.append((i, True))  # Maximum
            elif prices_array[i] == np.min(window_slice):
                extrema.append((i, False))  # Minimum
        
        return extrema
    
    def _rank_levels(
        self,
        all_levels: List[Dict[str, Any]],
        market_data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Rank and filter levels by importance
        
        Returns:
            Top ranked levels
        """
        if not all_levels:
            return []
        
        current_price = market_data["close"].iloc[-1] if len(market_data) > 0 else 0
        
        # Cluster nearby levels
        clustered_levels = self._cluster_levels(all_levels)
        
        # Calculate composite score for each level
        for level in clustered_levels:
            # Distance score (closer is better)
            distance = abs(level["price"] - current_price) / current_price
            distance_score = max(0, 1 - distance * 10)  # Decay over 10% distance
            
            # Method weight score
            method_score = self.method_weights.get(level.get("method", ""), 0.5)
            
            # Confluence score (more sources = better)
            confluence_score = min(1.0, len(level.get("sources", [])) / 3)
            
            # Composite score
            level["composite_score"] = (
                distance_score * 0.3 +
                method_score * 0.4 +
                confluence_score * 0.3
            ) * level.get("strength", 1.0)
        
        # Sort by composite score
        clustered_levels.sort(key=lambda x: x["composite_score"], reverse=True)
        
        # Return top levels
        return clustered_levels[:self.max_levels]
    
    def _cluster_levels(self, levels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Cluster nearby levels together
        
        Returns:
            Clustered levels with combined strength
        """
        if not levels:
            return []
        
        # Sort levels by price
        sorted_levels = sorted(levels, key=lambda x: x["price"])
        
        clustered = []
        current_cluster = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            # Check if level is within proximity threshold of cluster
            cluster_price = np.mean([l["price"] for l in current_cluster])
            
            if abs(level["price"] - cluster_price) / cluster_price < self.proximity_threshold:
                current_cluster.append(level)
            else:
                # Save current cluster and start new one
                clustered.append(self._merge_cluster(current_cluster))
                current_cluster = [level]
        
        # Add final cluster
        if current_cluster:
            clustered.append(self._merge_cluster(current_cluster))
        
        return clustered
    
    def _merge_cluster(self, cluster: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge a cluster of levels into a single level
        
        Returns:
            Merged level with combined properties
        """
        if not cluster:
            return {}
        
        # Weighted average price
        total_strength = sum(l.get("strength", 1.0) for l in cluster)
        weighted_price = sum(l["price"] * l.get("strength", 1.0) for l in cluster) / total_strength
        
        # Combine sources and methods
        sources = list(set(l.get("source", "") for l in cluster))
        methods = list(set(l.get("method", "") for l in cluster))
        
        # Determine type (majority vote)
        type_counts = {}
        for level in cluster:
            level_type = level.get("type", "neutral")
            type_counts[level_type] = type_counts.get(level_type, 0) + 1
        
        merged_type = max(type_counts, key=type_counts.get)
        
        return {
            "price": weighted_price,
            "strength": min(1.0, total_strength / 2),  # Combined strength
            "type": merged_type,
            "sources": sources,
            "methods": methods,
            "cluster_size": len(cluster)
        }
    
    def _calculate_level_metrics(
        self,
        ranked_levels: List[Dict[str, Any]],
        market_data: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """
        Calculate metrics for ranked levels
        
        Returns:
            Dictionary of level metrics arrays
        """
        current_price = market_data["close"].iloc[-1] if len(market_data) > 0 else 0
        
        # Initialize arrays
        prices = np.zeros(self.max_levels)
        strengths = np.zeros(self.max_levels)
        ages = np.zeros(self.max_levels)
        validations = np.zeros(self.max_levels)
        distances = np.zeros(self.max_levels)
        types = np.zeros(self.max_levels)
        
        # Fill arrays with level data
        for i, level in enumerate(ranked_levels[:self.max_levels]):
            prices[i] = level["price"]
            strengths[i] = level.get("strength", 1.0)
            ages[i] = i  # Placeholder - would calculate actual age in production
            validations[i] = level.get("cluster_size", 1)
            distances[i] = abs(level["price"] - current_price) / current_price
            types[i] = 1 if level.get("type") == "resistance" else 0
        
        return {
            "prices": prices,
            "strengths": strengths,
            "ages": ages,
            "validations": validations,
            "distances": distances,
            "types": types
        }
    
    def _calculate_learning_features(
        self,
        ranked_levels: List[Dict[str, Any]],
        market_data: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """
        Calculate dynamic learning features
        
        Returns:
            Dictionary of learning feature arrays
        """
        # Initialize arrays
        performance_scores = np.array(list(self.method_weights.values()))
        weight_adaptations = performance_scores.copy()
        
        # Calculate accuracy rates for different timeframes
        accuracy_rates = np.random.uniform(0.7, 0.9, 8)  # Placeholder
        
        # Calculate cross-validation metrics
        cross_validation = np.random.uniform(0.6, 0.85, 8)  # Placeholder
        
        return {
            "performance_scores": performance_scores,
            "weight_adaptations": weight_adaptations,
            "accuracy_rates": accuracy_rates,
            "cross_validation": cross_validation
        }
    
    def update_performance(
        self,
        level: Dict[str, Any],
        outcome: bool
    ) -> None:
        """
        Update performance tracking for a level prediction
        
        Args:
            level: Level that was predicted
            outcome: Whether prediction was successful
        """
        method = level.get("method", "unknown")
        
        if method in self.method_performance:
            self.method_performance[method].append(1.0 if outcome else 0.0)
            
            # Update weights based on performance
            if len(self.method_performance[method]) >= 50:
                recent_performance = np.mean(self.method_performance[method][-50:])
                
                # Adjust weight based on performance
                old_weight = self.method_weights.get(method, 0.5)
                new_weight = old_weight * 0.9 + recent_performance * 0.1
                
                # Normalize weights
                total_weight = sum(self.method_weights.values())
                if total_weight > 0:
                    for key in self.method_weights:
                        if key == method:
                            self.method_weights[key] = new_weight
                        else:
                            self.method_weights[key] *= (1 - new_weight) / (total_weight - old_weight)