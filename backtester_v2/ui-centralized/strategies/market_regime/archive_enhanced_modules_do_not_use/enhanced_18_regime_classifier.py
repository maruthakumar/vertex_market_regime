"""
Enhanced 18-Regime Classification System for Backtester V2

This module implements the complete 18-regime classification system based on the
original enhanced-market-regime-optimizer specification with directional and
volatility components for accurate market regime detection.

Features:
1. Complete 18-regime classification (vs previous 12 regimes)
2. Directional component analysis (-1 to +1)
3. Volatility component analysis (0 to 1)
4. Regime confidence scoring
5. Transition detection capabilities
6. Production-ready performance optimization

Author: The Augster
Date: 2025-01-16
Version: 1.0.0
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import os

logger = logging.getLogger(__name__)

@dataclass
class RegimeClassificationResult:
    """Result structure for regime classification"""
    regime: str
    confidence: float
    directional_score: float
    volatility_score: float
    regime_probability: float
    alternative_regimes: List[Tuple[str, float]]
    classification_timestamp: datetime

class Enhanced18RegimeClassifier:
    """
    Enhanced 18-Regime Classification System
    
    Implements complete 18-regime classification based on directional and volatility
    components, providing sophisticated market regime detection capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Enhanced 18-Regime Classifier"""
        self.config = config or {}
        
        # Load 18-regime configuration
        self.regime_config = self._load_18_regime_config()
        
        # Classification parameters
        self.confidence_threshold = float(self.config.get('confidence_threshold', 0.6))
        self.regime_smoothing = self.config.get('regime_smoothing', True)
        self.smoothing_window = int(self.config.get('smoothing_window', 3))
        
        # Performance optimization
        self.enable_caching = self.config.get('enable_caching', True)
        self.cache_size = int(self.config.get('cache_size', 1000))
        
        # Regime history for smoothing
        self.regime_history = []
        self.classification_cache = {}
        
        # Regime boundaries for optimization
        self._precompute_regime_boundaries()
        
        logger.info("Enhanced 18-Regime Classifier initialized with complete regime system")
    
    def _load_18_regime_config(self) -> Dict[str, Any]:
        """Load 18-regime configuration from JSON file"""
        try:
            # Try to load from enhanced optimizer package first
            config_path = "/srv/samba/shared/enhanced-market-regime-optimizer-final-package-updated/data/market_regime/market_regime_18_types_config.json"
            
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded 18-regime configuration from {config_path}")
                return config
            else:
                # Fallback to embedded configuration
                return self._get_embedded_18_regime_config()
                
        except Exception as e:
            logger.error(f"Error loading 18-regime configuration: {e}")
            return self._get_embedded_18_regime_config()
    
    def _get_embedded_18_regime_config(self) -> Dict[str, Any]:
        """Get embedded 18-regime configuration as fallback"""
        return {
            "market_regimes": [
                {"name": "Strong_Bullish", "directional_min": 0.7, "directional_max": 1.0, "volatility_min": 0.0, "volatility_max": 0.3, "description": "Strong bullish trend with low volatility"},
                {"name": "Bullish", "directional_min": 0.5, "directional_max": 0.7, "volatility_min": 0.0, "volatility_max": 0.3, "description": "Bullish trend with low volatility"},
                {"name": "Moderately_Bullish", "directional_min": 0.3, "directional_max": 0.5, "volatility_min": 0.0, "volatility_max": 0.3, "description": "Moderately bullish trend with low volatility"},
                {"name": "Weakly_Bullish", "directional_min": 0.10001, "directional_max": 0.3, "volatility_min": 0.0, "volatility_max": 0.3, "description": "Weakly bullish trend with low volatility"},
                {"name": "Bullish_Consolidation", "directional_min": 0.10001, "directional_max": 0.5, "volatility_min": 0.3, "volatility_max": 0.69999, "description": "Bullish consolidation with medium volatility"},
                {"name": "Bullish_Volatile", "directional_min": 0.3, "directional_max": 1.0, "volatility_min": 0.70001, "volatility_max": 1.0, "description": "Bullish trend with high volatility"},
                {"name": "Neutral_Bullish_Bias", "directional_min": 0.05, "directional_max": 0.1, "volatility_min": 0.0, "volatility_max": 0.5, "description": "Neutral market with slight bullish bias"},
                {"name": "Neutral", "directional_min": -0.05, "directional_max": 0.05, "volatility_min": 0.0, "volatility_max": 0.5, "description": "Neutral market with no directional bias"},
                {"name": "Neutral_Bearish_Bias", "directional_min": -0.1, "directional_max": -0.05, "volatility_min": 0.0, "volatility_max": 0.5, "description": "Neutral market with slight bearish bias"},
                {"name": "Bearish_Volatile", "directional_min": -1.0, "directional_max": -0.3, "volatility_min": 0.70001, "volatility_max": 1.0, "description": "Bearish trend with high volatility"},
                {"name": "Bearish_Consolidation", "directional_min": -0.5, "directional_max": -0.10001, "volatility_min": 0.3, "volatility_max": 0.69999, "description": "Bearish consolidation with medium volatility"},
                {"name": "Weakly_Bearish", "directional_min": -0.3, "directional_max": -0.10001, "volatility_min": 0.0, "volatility_max": 0.3, "description": "Weakly bearish trend with low volatility"},
                {"name": "Moderately_Bearish", "directional_min": -0.5, "directional_max": -0.3, "volatility_min": 0.0, "volatility_max": 0.3, "description": "Moderately bearish trend with low volatility"},
                {"name": "Bearish", "directional_min": -0.7, "directional_max": -0.5, "volatility_min": 0.0, "volatility_max": 0.3, "description": "Bearish trend with low volatility"},
                {"name": "Strong_Bearish", "directional_min": -1.0, "directional_max": -0.7, "volatility_min": 0.0, "volatility_max": 0.3, "description": "Strong bearish trend with low volatility"},
                {"name": "Volatility_Expansion", "directional_min": -0.1, "directional_max": 0.1, "volatility_min": 0.80001, "volatility_max": 1.0, "description": "High volatility with low directional bias"},
                {"name": "Bullish_To_Bearish_Transition", "directional_min": -0.09999, "directional_max": 0.3, "volatility_min": 0.7, "volatility_max": 0.8, "description": "Transition from bullish to bearish trend"},
                {"name": "Bearish_To_Bullish_Transition", "directional_min": -0.3, "directional_max": 0.09999, "volatility_min": 0.7, "volatility_max": 0.8, "description": "Transition from bearish to bullish trend"}
            ]
        }
    
    def _precompute_regime_boundaries(self):
        """Precompute regime boundaries for performance optimization"""
        self.regime_boundaries = []
        
        for regime in self.regime_config['market_regimes']:
            self.regime_boundaries.append({
                'name': regime['name'],
                'dir_min': regime['directional_min'],
                'dir_max': regime['directional_max'],
                'vol_min': regime['volatility_min'],
                'vol_max': regime['volatility_max'],
                'dir_center': (regime['directional_min'] + regime['directional_max']) / 2,
                'vol_center': (regime['volatility_min'] + regime['volatility_max']) / 2,
                'dir_range': regime['directional_max'] - regime['directional_min'],
                'vol_range': regime['volatility_max'] - regime['volatility_min'],
                'description': regime.get('description', '')
            })
    
    def classify_market_regime(self, directional_component: float, 
                             volatility_component: float) -> RegimeClassificationResult:
        """
        Classify market regime based on directional and volatility components
        
        Args:
            directional_component: Directional score (-1 to +1)
            volatility_component: Volatility score (0 to 1)
            
        Returns:
            RegimeClassificationResult with complete classification details
        """
        try:
            # Validate inputs
            directional_component = np.clip(directional_component, -1.0, 1.0)
            volatility_component = np.clip(volatility_component, 0.0, 1.0)
            
            # Check cache if enabled
            cache_key = f"{directional_component:.4f}_{volatility_component:.4f}"
            if self.enable_caching and cache_key in self.classification_cache:
                cached_result = self.classification_cache[cache_key]
                cached_result.classification_timestamp = datetime.now()
                return cached_result
            
            # Find matching regimes
            matching_regimes = []
            
            for regime in self.regime_boundaries:
                if (regime['dir_min'] <= directional_component <= regime['dir_max'] and
                    regime['vol_min'] <= volatility_component <= regime['vol_max']):
                    
                    # Calculate confidence based on distance from regime center
                    confidence = self._calculate_regime_confidence(
                        directional_component, volatility_component, regime
                    )
                    
                    matching_regimes.append({
                        'name': regime['name'],
                        'confidence': confidence,
                        'description': regime['description']
                    })
            
            # Sort by confidence
            matching_regimes.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Select best regime
            if matching_regimes:
                best_regime = matching_regimes[0]
                regime_name = best_regime['name']
                confidence = best_regime['confidence']
                
                # Calculate regime probability
                regime_probability = self._calculate_regime_probability(
                    directional_component, volatility_component, regime_name
                )
                
                # Get alternative regimes
                alternative_regimes = [
                    (regime['name'], regime['confidence']) 
                    for regime in matching_regimes[1:3]  # Top 2 alternatives
                ]
            else:
                # Default to Neutral if no match
                regime_name = "Neutral"
                confidence = 0.5
                regime_probability = 0.5
                alternative_regimes = []
            
            # Apply regime smoothing if enabled
            if self.regime_smoothing:
                regime_name = self._apply_regime_smoothing(regime_name)
            
            # Create result
            result = RegimeClassificationResult(
                regime=regime_name,
                confidence=confidence,
                directional_score=directional_component,
                volatility_score=volatility_component,
                regime_probability=regime_probability,
                alternative_regimes=alternative_regimes,
                classification_timestamp=datetime.now()
            )
            
            # Cache result if enabled
            if self.enable_caching:
                self._update_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error classifying market regime: {e}")
            return RegimeClassificationResult(
                regime="Neutral",
                confidence=0.5,
                directional_score=directional_component,
                volatility_score=volatility_component,
                regime_probability=0.5,
                alternative_regimes=[],
                classification_timestamp=datetime.now()
            )
    
    def _calculate_regime_confidence(self, dir_comp: float, vol_comp: float, 
                                   regime: Dict[str, Any]) -> float:
        """Calculate confidence in regime classification"""
        try:
            # Distance from regime center (normalized)
            dir_distance = abs(dir_comp - regime['dir_center'])
            vol_distance = abs(vol_comp - regime['vol_center'])
            
            # Normalize distances by regime range
            dir_confidence = 1.0 - (dir_distance / (regime['dir_range'] / 2)) if regime['dir_range'] > 0 else 1.0
            vol_confidence = 1.0 - (vol_distance / (regime['vol_range'] / 2)) if regime['vol_range'] > 0 else 1.0
            
            # Combined confidence (weighted average)
            combined_confidence = (dir_confidence * 0.6 + vol_confidence * 0.4)
            
            return np.clip(combined_confidence, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating regime confidence: {e}")
            return 0.5
    
    def _calculate_regime_probability(self, dir_comp: float, vol_comp: float, 
                                    regime_name: str) -> float:
        """Calculate probability of regime classification"""
        try:
            # Find regime boundaries
            regime_info = None
            for regime in self.regime_boundaries:
                if regime['name'] == regime_name:
                    regime_info = regime
                    break
            
            if not regime_info:
                return 0.5
            
            # Calculate probability based on position within regime boundaries
            dir_position = (dir_comp - regime_info['dir_min']) / regime_info['dir_range'] if regime_info['dir_range'] > 0 else 0.5
            vol_position = (vol_comp - regime_info['vol_min']) / regime_info['vol_range'] if regime_info['vol_range'] > 0 else 0.5
            
            # Probability is higher near center of regime
            dir_prob = 1.0 - abs(dir_position - 0.5) * 2
            vol_prob = 1.0 - abs(vol_position - 0.5) * 2
            
            combined_probability = (dir_prob * 0.6 + vol_prob * 0.4)
            
            return np.clip(combined_probability, 0.1, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating regime probability: {e}")
            return 0.5
    
    def _apply_regime_smoothing(self, current_regime: str) -> str:
        """Apply regime smoothing to reduce noise"""
        try:
            # Add current regime to history
            self.regime_history.append(current_regime)
            
            # Keep only recent history
            if len(self.regime_history) > self.smoothing_window:
                self.regime_history = self.regime_history[-self.smoothing_window:]
            
            # If we don't have enough history, return current regime
            if len(self.regime_history) < self.smoothing_window:
                return current_regime
            
            # Find most common regime in recent history
            regime_counts = {}
            for regime in self.regime_history:
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            # Return most common regime
            most_common_regime = max(regime_counts, key=regime_counts.get)
            
            return most_common_regime
            
        except Exception as e:
            logger.error(f"Error applying regime smoothing: {e}")
            return current_regime
    
    def _update_cache(self, cache_key: str, result: RegimeClassificationResult):
        """Update classification cache"""
        try:
            # Add to cache
            self.classification_cache[cache_key] = result
            
            # Limit cache size
            if len(self.classification_cache) > self.cache_size:
                # Remove oldest entries (simple FIFO)
                oldest_keys = list(self.classification_cache.keys())[:-self.cache_size]
                for key in oldest_keys:
                    del self.classification_cache[key]
                    
        except Exception as e:
            logger.error(f"Error updating cache: {e}")
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get regime classification statistics"""
        try:
            total_regimes = len(self.regime_config['market_regimes'])
            
            # Count regime history
            regime_counts = {}
            for regime in self.regime_history:
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            # Calculate statistics
            total_classifications = len(self.regime_history)
            most_common_regime = max(regime_counts, key=regime_counts.get) if regime_counts else "None"
            
            return {
                'total_regimes_available': total_regimes,
                'total_classifications': total_classifications,
                'most_common_regime': most_common_regime,
                'regime_distribution': regime_counts,
                'cache_size': len(self.classification_cache),
                'smoothing_enabled': self.regime_smoothing,
                'smoothing_window': self.smoothing_window
            }
            
        except Exception as e:
            logger.error(f"Error getting regime statistics: {e}")
            return {}
    
    def reset_history(self):
        """Reset regime history and cache"""
        try:
            self.regime_history = []
            self.classification_cache = {}
            logger.info("Regime history and cache reset")
            
        except Exception as e:
            logger.error(f"Error resetting history: {e}")
