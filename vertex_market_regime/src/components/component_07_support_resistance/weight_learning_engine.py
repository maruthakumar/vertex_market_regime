"""
Dynamic Weight Learning Engine for Support/Resistance Detection
Performance-based weight optimization for detection methods
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SupportResistanceWeightLearner:
    """
    Learns and optimizes weights for different S&R detection methods
    based on historical performance and DTE-specific patterns
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize weight learner
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Learning parameters
        self.learning_rate = config.get("learning_rate", 0.1)
        self.min_samples = config.get("min_samples", 50)
        self.lookback_window = config.get("lookback_window", 252)  # Trading days
        self.update_frequency = config.get("update_frequency", 20)  # Update every N samples
        
        # DTE ranges for specific learning
        self.dte_ranges = {
            "weekly": (0, 7),
            "monthly": (8, 30),
            "far_month": (31, 90)
        }
        
        # Initialize method weights
        self.method_weights = {
            "component_1_straddle": 0.15,
            "component_1_ema": 0.10,
            "component_1_vwap": 0.10,
            "component_1_pivot": 0.08,
            "component_3_cumulative": 0.15,
            "component_3_rolling": 0.10,
            "daily_pivots": 0.08,
            "weekly_pivots": 0.06,
            "monthly_pivots": 0.05,
            "volume_profile": 0.08,
            "moving_averages": 0.05,
            "fibonacci": 0.05,
            "psychological": 0.05,
            "daily_gaps": 0.05
        }
        
        # DTE-specific weights
        self.dte_weights = {
            dte_range: self.method_weights.copy() 
            for dte_range in self.dte_ranges.keys()
        }
        
        # Individual DTE weights (0-90)
        self.specific_dte_weights = {
            dte: self.method_weights.copy() 
            for dte in range(91)
        }
        
        # Performance tracking
        self.performance_history = defaultdict(lambda: deque(maxlen=self.lookback_window))
        self.dte_performance = {
            dte_range: defaultdict(lambda: deque(maxlen=self.lookback_window))
            for dte_range in self.dte_ranges.keys()
        }
        self.specific_dte_performance = {
            dte: defaultdict(lambda: deque(maxlen=100))
            for dte in range(91)
        }
        
        # Update counters
        self.update_counter = 0
        self.samples_since_update = 0
        
        logger.info("Initialized SupportResistanceWeightLearner with adaptive learning")
    
    def track_performance(
        self,
        level: Dict[str, Any],
        outcome: bool,
        dte: Optional[int] = None,
        market_context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Track performance of a level prediction
        
        Args:
            level: The predicted level with method information
            outcome: Whether the level held (True) or broke (False)
            dte: Days to expiry for context-specific learning
            market_context: Optional market context for enhanced learning
        """
        method = level.get("method", "unknown")
        
        # Track overall performance
        self.performance_history[method].append(1.0 if outcome else 0.0)
        
        # Track DTE-specific performance
        if dte is not None:
            # Find DTE range
            for range_name, (min_dte, max_dte) in self.dte_ranges.items():
                if min_dte <= dte <= max_dte:
                    self.dte_performance[range_name][method].append(1.0 if outcome else 0.0)
                    break
            
            # Track specific DTE if within range
            if 0 <= dte <= 90:
                self.specific_dte_performance[dte][method].append(1.0 if outcome else 0.0)
        
        # Update counter
        self.samples_since_update += 1
        
        # Check if we should update weights
        if self.samples_since_update >= self.update_frequency:
            self.update_weights(dte)
            self.samples_since_update = 0
    
    def update_weights(self, current_dte: Optional[int] = None) -> None:
        """
        Update method weights based on recent performance
        
        Args:
            current_dte: Current DTE for context-specific updates
        """
        # Update overall weights
        self._update_method_weights(
            self.method_weights,
            self.performance_history
        )
        
        # Update DTE-specific weights if applicable
        if current_dte is not None:
            # Update DTE range weights
            for range_name, (min_dte, max_dte) in self.dte_ranges.items():
                if min_dte <= current_dte <= max_dte:
                    self._update_method_weights(
                        self.dte_weights[range_name],
                        self.dte_performance[range_name]
                    )
            
            # Update specific DTE weights
            if 0 <= current_dte <= 90:
                self._update_method_weights(
                    self.specific_dte_weights[current_dte],
                    self.specific_dte_performance[current_dte]
                )
        
        self.update_counter += 1
        logger.debug(f"Updated weights (iteration {self.update_counter})")
    
    def _update_method_weights(
        self,
        weights: Dict[str, float],
        performance: Dict[str, deque]
    ) -> None:
        """
        Update weights dictionary based on performance history
        
        Args:
            weights: Weights dictionary to update
            performance: Performance history for each method
        """
        # Calculate performance scores for each method
        performance_scores = {}
        
        for method in weights.keys():
            if method in performance and len(performance[method]) >= self.min_samples:
                # Calculate recent performance
                recent_performance = np.mean(list(performance[method][-self.min_samples:]))
                
                # Calculate performance trend
                if len(performance[method]) >= self.min_samples * 2:
                    first_half = np.mean(list(performance[method][-self.min_samples * 2:-self.min_samples]))
                    second_half = recent_performance
                    trend = second_half - first_half
                else:
                    trend = 0
                
                # Combined score with trend
                performance_scores[method] = recent_performance + trend * 0.2
            else:
                # No performance data, keep current weight
                performance_scores[method] = weights[method]
        
        # Update weights using exponential moving average
        for method in weights.keys():
            if method in performance_scores:
                old_weight = weights[method]
                performance_score = performance_scores[method]
                
                # EMA update
                new_weight = old_weight * (1 - self.learning_rate) + performance_score * self.learning_rate
                
                # Apply bounds
                new_weight = max(0.01, min(0.3, new_weight))  # Keep weights between 1% and 30%
                
                weights[method] = new_weight
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            for method in weights:
                weights[method] /= total_weight
    
    def get_adaptive_weights(
        self,
        dte: Optional[int] = None,
        market_regime: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Get adaptive weights for current context
        
        Args:
            dte: Days to expiry
            market_regime: Optional market regime for context
            
        Returns:
            Dictionary of method weights
        """
        # If no DTE specified, return overall weights
        if dte is None:
            return self.method_weights.copy()
        
        # Check for specific DTE weights with sufficient data
        if 0 <= dte <= 90:
            specific_weights = self.specific_dte_weights[dte]
            
            # Check if we have enough data for this specific DTE
            has_sufficient_data = any(
                len(self.specific_dte_performance[dte][method]) >= self.min_samples
                for method in specific_weights.keys()
            )
            
            if has_sufficient_data:
                return specific_weights.copy()
        
        # Fall back to DTE range weights
        for range_name, (min_dte, max_dte) in self.dte_ranges.items():
            if min_dte <= dte <= max_dte:
                range_weights = self.dte_weights[range_name]
                
                # Check if we have enough data for this range
                has_range_data = any(
                    len(self.dte_performance[range_name][method]) >= self.min_samples
                    for method in range_weights.keys()
                )
                
                if has_range_data:
                    return range_weights.copy()
                break
        
        # Fall back to overall weights
        return self.method_weights.copy()
    
    def get_performance_metrics(
        self,
        method: Optional[str] = None,
        dte: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get performance metrics for methods
        
        Args:
            method: Specific method to get metrics for (None for all)
            dte: Specific DTE to get metrics for
            
        Returns:
            Dictionary of performance metrics
        """
        if method:
            # Get metrics for specific method
            return self._calculate_method_metrics(method, dte)
        else:
            # Get metrics for all methods
            metrics = {}
            for m in self.method_weights.keys():
                metrics[m] = self._calculate_method_metrics(m, dte)
            return metrics
    
    def _calculate_method_metrics(
        self,
        method: str,
        dte: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Calculate metrics for a specific method
        
        Returns:
            Dictionary of metrics
        """
        # Get appropriate performance history
        if dte is not None and 0 <= dte <= 90:
            history = self.specific_dte_performance[dte][method]
        else:
            history = self.performance_history[method]
        
        if not history:
            return {
                "samples": 0,
                "accuracy": 0,
                "recent_accuracy": 0,
                "trend": 0,
                "weight": self.method_weights.get(method, 0)
            }
        
        history_list = list(history)
        
        # Calculate metrics
        accuracy = np.mean(history_list)
        recent_accuracy = np.mean(history_list[-20:]) if len(history_list) >= 20 else accuracy
        
        # Calculate trend
        if len(history_list) >= 40:
            first_half = np.mean(history_list[-40:-20])
            second_half = np.mean(history_list[-20:])
            trend = second_half - first_half
        else:
            trend = 0
        
        return {
            "samples": len(history_list),
            "accuracy": accuracy,
            "recent_accuracy": recent_accuracy,
            "trend": trend,
            "weight": self.get_adaptive_weights(dte).get(method, 0)
        }
    
    def optimize_weights_batch(
        self,
        performance_data: List[Dict[str, Any]]
    ) -> None:
        """
        Optimize weights using batch performance data
        
        Args:
            performance_data: List of performance records
        """
        # Group by method and DTE
        method_performance = defaultdict(list)
        dte_method_performance = defaultdict(lambda: defaultdict(list))
        
        for record in performance_data:
            method = record.get("method", "unknown")
            outcome = record.get("outcome", False)
            dte = record.get("dte")
            
            method_performance[method].append(1.0 if outcome else 0.0)
            
            if dte is not None:
                for range_name, (min_dte, max_dte) in self.dte_ranges.items():
                    if min_dte <= dte <= max_dte:
                        dte_method_performance[range_name][method].append(1.0 if outcome else 0.0)
                        break
        
        # Update weights based on batch data
        for method, outcomes in method_performance.items():
            if len(outcomes) >= self.min_samples:
                performance_score = np.mean(outcomes)
                
                # Update main weights
                if method in self.method_weights:
                    old_weight = self.method_weights[method]
                    new_weight = old_weight * 0.7 + performance_score * 0.3
                    self.method_weights[method] = max(0.01, min(0.3, new_weight))
        
        # Normalize weights
        total = sum(self.method_weights.values())
        if total > 0:
            for method in self.method_weights:
                self.method_weights[method] /= total
        
        logger.info(f"Batch optimization complete with {len(performance_data)} records")
    
    def save_weights(self, filepath: str) -> None:
        """
        Save current weights to file
        
        Args:
            filepath: Path to save weights
        """
        import json
        
        weights_data = {
            "method_weights": self.method_weights,
            "dte_weights": self.dte_weights,
            "specific_dte_weights": self.specific_dte_weights,
            "update_counter": self.update_counter,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(weights_data, f, indent=2)
        
        logger.info(f"Saved weights to {filepath}")
    
    def load_weights(self, filepath: str) -> None:
        """
        Load weights from file
        
        Args:
            filepath: Path to load weights from
        """
        import json
        
        with open(filepath, 'r') as f:
            weights_data = json.load(f)
        
        self.method_weights = weights_data.get("method_weights", self.method_weights)
        self.dte_weights = weights_data.get("dte_weights", self.dte_weights)
        self.specific_dte_weights = weights_data.get("specific_dte_weights", self.specific_dte_weights)
        self.update_counter = weights_data.get("update_counter", 0)
        
        logger.info(f"Loaded weights from {filepath}")