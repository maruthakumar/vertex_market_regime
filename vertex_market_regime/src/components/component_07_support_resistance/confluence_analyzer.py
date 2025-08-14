"""
Confluence Analyzer for Support/Resistance Levels
Cross-source validation and multi-timeframe agreement scoring
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from collections import defaultdict

logger = logging.getLogger(__name__)


class ConfluenceAnalyzer:
    """
    Analyzes confluence between different S&R detection sources
    Validates levels across straddle-based and underlying-based methods
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize confluence analyzer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Confluence parameters
        self.proximity_threshold = config.get("proximity_threshold", 0.002)  # 0.2%
        self.min_confluence_sources = config.get("min_confluence_sources", 2)
        self.timeframe_weights = {
            "1min": 0.05,
            "3min": 0.08,
            "5min": 0.15,
            "15min": 0.20,
            "30min": 0.15,
            "60min": 0.12,
            "daily": 0.10,
            "weekly": 0.08,
            "monthly": 0.07
        }
        
        # Source weights for confluence scoring
        self.source_weights = {
            "component_1_straddle": 0.20,
            "component_3_cumulative": 0.20,
            "daily_pivots": 0.10,
            "weekly_pivots": 0.08,
            "monthly_pivots": 0.07,
            "volume_profile": 0.10,
            "moving_averages": 0.08,
            "fibonacci": 0.07,
            "psychological": 0.05,
            "historical": 0.05
        }
        
        logger.info("Initialized ConfluenceAnalyzer")
    
    def analyze_confluence(
        self,
        straddle_levels: List[Dict[str, Any]],
        underlying_levels: List[Dict[str, Any]],
        component_1_levels: Optional[List[Dict[str, Any]]] = None,
        component_3_levels: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Analyze confluence between all S&R sources
        
        Args:
            straddle_levels: Levels from straddle-based detection
            underlying_levels: Levels from underlying price detection
            component_1_levels: Optional Component 1 specific levels
            component_3_levels: Optional Component 3 specific levels
            
        Returns:
            Dictionary with confluence analysis results
        """
        # Combine all levels
        all_levels = []
        all_levels.extend(straddle_levels)
        all_levels.extend(underlying_levels)
        
        if component_1_levels:
            all_levels.extend(component_1_levels)
        if component_3_levels:
            all_levels.extend(component_3_levels)
        
        # Measure straddle vs underlying confluence
        straddle_underlying_confluence = self.measure_straddle_underlying_confluence(
            straddle_levels, underlying_levels
        )
        
        # Measure Component 1 vs Component 3 validation
        comp1_comp3_validation = None
        if component_1_levels and component_3_levels:
            comp1_comp3_validation = self.validate_component_levels(
                component_1_levels, component_3_levels
            )
        
        # Score multi-timeframe agreement
        timeframe_agreement = self.score_timeframe_agreement(all_levels)
        
        # Combine levels with weighting
        combined_levels = self.combine_weighted_levels(all_levels)
        
        return {
            "straddle_underlying_confluence": straddle_underlying_confluence,
            "component_validation": comp1_comp3_validation,
            "timeframe_agreement": timeframe_agreement,
            "combined_levels": combined_levels,
            "confluence_statistics": self._calculate_confluence_statistics(all_levels)
        }
    
    def measure_straddle_underlying_confluence(
        self,
        straddle_levels: List[Dict[str, Any]],
        underlying_levels: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Measure confluence between straddle and underlying levels
        
        Returns:
            Dictionary with confluence measurements
        """
        confluence_pairs = []
        
        for s_level in straddle_levels:
            s_price = s_level["price"]
            
            for u_level in underlying_levels:
                u_price = u_level["price"]
                
                # Check if levels are within proximity threshold
                if abs(s_price - u_price) / s_price < self.proximity_threshold:
                    confluence_pairs.append({
                        "straddle_level": s_level,
                        "underlying_level": u_level,
                        "price_difference": abs(s_price - u_price),
                        "relative_difference": abs(s_price - u_price) / s_price,
                        "confluence_score": self._calculate_pair_confluence(s_level, u_level)
                    })
        
        # Calculate overall confluence metrics
        if confluence_pairs:
            avg_confluence = np.mean([p["confluence_score"] for p in confluence_pairs])
            max_confluence = max([p["confluence_score"] for p in confluence_pairs])
            confluence_ratio = len(confluence_pairs) / max(len(straddle_levels), len(underlying_levels))
        else:
            avg_confluence = 0
            max_confluence = 0
            confluence_ratio = 0
        
        return {
            "confluence_pairs": confluence_pairs,
            "average_confluence": avg_confluence,
            "max_confluence": max_confluence,
            "confluence_ratio": confluence_ratio,
            "total_pairs": len(confluence_pairs)
        }
    
    def validate_component_levels(
        self,
        component_1_levels: List[Dict[str, Any]],
        component_3_levels: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate Component 1 vs Component 3 levels
        
        Returns:
            Dictionary with validation results
        """
        validated_levels = []
        
        for c1_level in component_1_levels:
            c1_price = c1_level["price"]
            
            for c3_level in component_3_levels:
                c3_price = c3_level["price"]
                
                # Check proximity
                if abs(c1_price - c3_price) / c1_price < self.proximity_threshold:
                    # Both components agree on this level
                    validated_levels.append({
                        "price": (c1_price + c3_price) / 2,  # Average price
                        "component_1": c1_level,
                        "component_3": c3_level,
                        "validation_strength": self._calculate_validation_strength(c1_level, c3_level),
                        "type_agreement": c1_level.get("type") == c3_level.get("type")
                    })
        
        # Calculate validation metrics
        validation_rate = len(validated_levels) / max(
            len(component_1_levels), 
            len(component_3_levels)
        ) if component_1_levels or component_3_levels else 0
        
        type_agreement_rate = sum(
            1 for v in validated_levels if v["type_agreement"]
        ) / len(validated_levels) if validated_levels else 0
        
        return {
            "validated_levels": validated_levels,
            "validation_rate": validation_rate,
            "type_agreement_rate": type_agreement_rate,
            "total_validated": len(validated_levels)
        }
    
    def score_timeframe_agreement(
        self,
        levels: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Score agreement across multiple timeframes
        
        Returns:
            Dictionary with timeframe agreement scores
        """
        # Group levels by price (with tolerance)
        price_clusters = self._cluster_levels_by_price(levels)
        
        timeframe_scores = []
        
        for cluster_price, cluster_levels in price_clusters.items():
            # Extract unique timeframes in this cluster
            timeframes = set()
            for level in cluster_levels:
                if "timeframe" in level:
                    timeframes.add(level["timeframe"])
            
            # Calculate timeframe diversity score
            diversity_score = len(timeframes) / len(self.timeframe_weights)
            
            # Calculate weighted timeframe score
            weighted_score = sum(
                self.timeframe_weights.get(tf, 0.05) 
                for tf in timeframes
            )
            
            timeframe_scores.append({
                "price": cluster_price,
                "timeframes": list(timeframes),
                "diversity_score": diversity_score,
                "weighted_score": weighted_score,
                "level_count": len(cluster_levels)
            })
        
        # Sort by weighted score
        timeframe_scores.sort(key=lambda x: x["weighted_score"], reverse=True)
        
        # Calculate overall agreement metrics
        if timeframe_scores:
            avg_diversity = np.mean([s["diversity_score"] for s in timeframe_scores])
            max_diversity = max([s["diversity_score"] for s in timeframe_scores])
            avg_weighted = np.mean([s["weighted_score"] for s in timeframe_scores])
        else:
            avg_diversity = 0
            max_diversity = 0
            avg_weighted = 0
        
        return {
            "timeframe_scores": timeframe_scores[:10],  # Top 10 levels
            "average_diversity": avg_diversity,
            "max_diversity": max_diversity,
            "average_weighted_score": avg_weighted
        }
    
    def combine_weighted_levels(
        self,
        levels: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Combine levels with weighting based on confluence
        
        Returns:
            List of combined levels with confluence-based weights
        """
        # Cluster levels by price
        price_clusters = self._cluster_levels_by_price(levels)
        
        combined_levels = []
        
        for cluster_price, cluster_levels in price_clusters.items():
            # Calculate combined metrics
            combined_strength = 0
            combined_weight = 0
            sources = []
            methods = []
            types = defaultdict(int)
            
            for level in cluster_levels:
                # Get method weight
                method = level.get("method", "unknown")
                method_weight = self.source_weights.get(method, 0.05)
                
                # Add to combined metrics
                level_strength = level.get("strength", 1.0)
                combined_strength += level_strength * method_weight
                combined_weight += method_weight
                
                # Collect sources and methods
                if "source" in level:
                    sources.append(level["source"])
                if "method" in level:
                    methods.append(level["method"])
                
                # Count types
                level_type = level.get("type", "neutral")
                types[level_type] += 1
            
            # Normalize combined strength
            if combined_weight > 0:
                combined_strength /= combined_weight
            
            # Determine dominant type
            dominant_type = max(types, key=types.get) if types else "neutral"
            
            # Calculate confluence score
            confluence_score = self._calculate_cluster_confluence(
                cluster_levels, 
                len(set(sources)), 
                len(set(methods))
            )
            
            combined_levels.append({
                "price": cluster_price,
                "strength": combined_strength,
                "type": dominant_type,
                "confluence_score": confluence_score,
                "source_count": len(set(sources)),
                "method_count": len(set(methods)),
                "level_count": len(cluster_levels),
                "sources": list(set(sources))[:5],  # Top 5 sources
                "methods": list(set(methods))
            })
        
        # Sort by confluence score
        combined_levels.sort(key=lambda x: x["confluence_score"], reverse=True)
        
        return combined_levels
    
    def _cluster_levels_by_price(
        self,
        levels: List[Dict[str, Any]]
    ) -> Dict[float, List[Dict[str, Any]]]:
        """
        Cluster levels by price with proximity threshold
        
        Returns:
            Dictionary of price clusters
        """
        if not levels:
            return {}
        
        # Sort levels by price
        sorted_levels = sorted(levels, key=lambda x: x.get("price", 0))
        
        clusters = {}
        
        for level in sorted_levels:
            price = level.get("price", 0)
            
            # Find existing cluster or create new one
            matched = False
            for cluster_price in list(clusters.keys()):
                if abs(price - cluster_price) / cluster_price < self.proximity_threshold:
                    clusters[cluster_price].append(level)
                    matched = True
                    break
            
            if not matched:
                clusters[price] = [level]
        
        return clusters
    
    def _calculate_pair_confluence(
        self,
        level1: Dict[str, Any],
        level2: Dict[str, Any]
    ) -> float:
        """
        Calculate confluence score between two levels
        
        Returns:
            Confluence score (0-1)
        """
        score = 0.5  # Base score for proximity match
        
        # Type agreement bonus
        if level1.get("type") == level2.get("type"):
            score += 0.2
        
        # Strength combination
        strength1 = level1.get("strength", 1.0)
        strength2 = level2.get("strength", 1.0)
        score += (strength1 + strength2) / 4
        
        # Method diversity bonus
        if level1.get("method") != level2.get("method"):
            score += 0.1
        
        return min(1.0, score)
    
    def _calculate_validation_strength(
        self,
        c1_level: Dict[str, Any],
        c3_level: Dict[str, Any]
    ) -> float:
        """
        Calculate validation strength between Component 1 and 3 levels
        
        Returns:
            Validation strength (0-1)
        """
        # Base strength from component strengths
        c1_strength = c1_level.get("strength", 1.0)
        c3_strength = c3_level.get("strength", 1.0)
        base_strength = (c1_strength + c3_strength) / 2
        
        # Type agreement multiplier
        type_multiplier = 1.2 if c1_level.get("type") == c3_level.get("type") else 0.8
        
        # Calculate final strength
        validation_strength = base_strength * type_multiplier
        
        return min(1.0, validation_strength)
    
    def _calculate_cluster_confluence(
        self,
        cluster_levels: List[Dict[str, Any]],
        source_count: int,
        method_count: int
    ) -> float:
        """
        Calculate confluence score for a cluster of levels
        
        Returns:
            Confluence score (0-1)
        """
        # Base score from number of levels
        level_score = min(1.0, len(cluster_levels) / 5)  # Max at 5 levels
        
        # Source diversity score
        source_score = min(1.0, source_count / 3)  # Max at 3 sources
        
        # Method diversity score
        method_score = min(1.0, method_count / 4)  # Max at 4 methods
        
        # Average strength of levels
        avg_strength = np.mean([l.get("strength", 1.0) for l in cluster_levels])
        
        # Calculate weighted confluence
        confluence = (
            level_score * 0.3 +
            source_score * 0.25 +
            method_score * 0.25 +
            avg_strength * 0.2
        )
        
        return confluence
    
    def _calculate_confluence_statistics(
        self,
        levels: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate overall confluence statistics
        
        Returns:
            Dictionary with confluence statistics
        """
        if not levels:
            return {
                "total_levels": 0,
                "unique_prices": 0,
                "average_cluster_size": 0,
                "max_cluster_size": 0,
                "method_distribution": {},
                "type_distribution": {}
            }
        
        # Cluster levels
        price_clusters = self._cluster_levels_by_price(levels)
        
        # Calculate cluster sizes
        cluster_sizes = [len(cluster) for cluster in price_clusters.values()]
        
        # Method distribution
        method_counts = defaultdict(int)
        for level in levels:
            method = level.get("method", "unknown")
            method_counts[method] += 1
        
        # Type distribution
        type_counts = defaultdict(int)
        for level in levels:
            level_type = level.get("type", "neutral")
            type_counts[level_type] += 1
        
        return {
            "total_levels": len(levels),
            "unique_prices": len(price_clusters),
            "average_cluster_size": np.mean(cluster_sizes) if cluster_sizes else 0,
            "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
            "method_distribution": dict(method_counts),
            "type_distribution": dict(type_counts),
            "confluence_ratio": len(price_clusters) / len(levels) if levels else 0
        }
    
    def get_strongest_levels(
        self,
        combined_levels: List[Dict[str, Any]],
        max_levels: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get the strongest levels based on confluence
        
        Args:
            combined_levels: List of combined levels
            max_levels: Maximum number of levels to return
            
        Returns:
            List of strongest levels
        """
        # Already sorted by confluence score
        return combined_levels[:max_levels]