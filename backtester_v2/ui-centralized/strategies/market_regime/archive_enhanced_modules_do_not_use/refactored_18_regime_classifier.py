"""
Refactored 18-Regime Classification System using Base Class

This module implements the complete 18-regime classification system based on the
original enhanced-market-regime-optimizer specification, now inheriting from
RegimeDetectorBase to eliminate code duplication.

Features:
1. Complete 18-regime classification
2. Directional component analysis (-1 to +1)
3. Volatility component analysis (0 to 1)
4. Inherits caching, monitoring, and validation from base class
5. Seamless integration with existing systems

Author: Market Regime System Optimizer
Date: 2025-07-07
Version: 2.0.0 - Refactored with base class
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
import os
from pathlib import Path

# Import base class
import sys
sys.path.append(str(Path(__file__).parent.parent))
from base.regime_detector_base import RegimeDetectorBase, RegimeClassification

logger = logging.getLogger(__name__)


class Refactored18RegimeClassifier(RegimeDetectorBase):
    """
    Refactored 18-Regime Classification System
    
    Implements complete 18-regime classification based on directional and volatility
    components using the common base class functionality.
    """
    
    def _initialize_detector(self):
        """Initialize 18-regime specific configuration"""
        # Load 18-regime configuration
        self.regime_config = self._load_18_regime_config()
        
        # Regime boundaries optimization
        self._precompute_regime_boundaries()
        
        logger.info("18-Regime Classifier initialized with refactored architecture")
        
    def get_regime_count(self) -> int:
        """Get number of regimes supported by this detector"""
        return 18
        
    def get_regime_mapping(self) -> Dict[str, str]:
        """Get mapping of regime IDs to descriptions"""
        return {regime['name']: regime['description'] 
                for regime in self.regime_config['market_regimes']}
        
    def _load_18_regime_config(self) -> Dict[str, Any]:
        """Load 18-regime configuration from JSON file or embedded config"""
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
                {"name": "Strong_Bullish", "directional_min": 0.7, "directional_max": 1.0, 
                 "volatility_min": 0.0, "volatility_max": 0.3, 
                 "description": "Strong bullish trend with low volatility"},
                {"name": "Moderate_Bullish", "directional_min": 0.4, "directional_max": 0.7, 
                 "volatility_min": 0.0, "volatility_max": 0.3, 
                 "description": "Moderate bullish trend with low volatility"},
                {"name": "Weak_Bullish", "directional_min": 0.1, "directional_max": 0.4, 
                 "volatility_min": 0.0, "volatility_max": 0.3, 
                 "description": "Weak bullish trend with low volatility"},
                {"name": "Neutral_Bullish", "directional_min": 0.0, "directional_max": 0.1, 
                 "volatility_min": 0.0, "volatility_max": 0.3, 
                 "description": "Neutral market with slight bullish bias"},
                {"name": "Sideways_High_Vol", "directional_min": -0.1, "directional_max": 0.1, 
                 "volatility_min": 0.7, "volatility_max": 1.0, 
                 "description": "Sideways market with high volatility"},
                {"name": "Sideways_Low_Vol", "directional_min": -0.1, "directional_max": 0.1, 
                 "volatility_min": 0.0, "volatility_max": 0.3, 
                 "description": "Sideways market with low volatility"},
                {"name": "Neutral_Bearish", "directional_min": -0.1, "directional_max": 0.0, 
                 "volatility_min": 0.0, "volatility_max": 0.3, 
                 "description": "Neutral market with slight bearish bias"},
                {"name": "Weak_Bearish", "directional_min": -0.4, "directional_max": -0.1, 
                 "volatility_min": 0.0, "volatility_max": 0.3, 
                 "description": "Weak bearish trend with low volatility"},
                {"name": "Moderate_Bearish", "directional_min": -0.7, "directional_max": -0.4, 
                 "volatility_min": 0.0, "volatility_max": 0.3, 
                 "description": "Moderate bearish trend with low volatility"},
                {"name": "Strong_Bearish", "directional_min": -1.0, "directional_max": -0.7, 
                 "volatility_min": 0.0, "volatility_max": 0.3, 
                 "description": "Strong bearish trend with low volatility"},
                {"name": "Volatile_Bullish", "directional_min": 0.3, "directional_max": 0.7, 
                 "volatility_min": 0.7, "volatility_max": 1.0, 
                 "description": "Bullish trend with high volatility"},
                {"name": "Volatile_Bearish", "directional_min": -0.7, "directional_max": -0.3, 
                 "volatility_min": 0.7, "volatility_max": 1.0, 
                 "description": "Bearish trend with high volatility"},
                {"name": "Trending_Bullish", "directional_min": 0.5, "directional_max": 0.9, 
                 "volatility_min": 0.3, "volatility_max": 0.7, 
                 "description": "Strong bullish trending market"},
                {"name": "Trending_Bearish", "directional_min": -0.9, "directional_max": -0.5, 
                 "volatility_min": 0.3, "volatility_max": 0.7, 
                 "description": "Strong bearish trending market"},
                {"name": "Range_Bound_High", "directional_min": -0.3, "directional_max": 0.3, 
                 "volatility_min": 0.5, "volatility_max": 0.8, 
                 "description": "Range-bound market with elevated volatility"},
                {"name": "Range_Bound_Low", "directional_min": -0.2, "directional_max": 0.2, 
                 "volatility_min": 0.1, "volatility_max": 0.4, 
                 "description": "Range-bound market with low volatility"},
                {"name": "Breakout_Bullish", "directional_min": 0.6, "directional_max": 1.0, 
                 "volatility_min": 0.4, "volatility_max": 0.8, 
                 "description": "Bullish breakout with increasing volatility"},
                {"name": "Breakout_Bearish", "directional_min": -1.0, "directional_max": -0.6, 
                 "volatility_min": 0.4, "volatility_max": 0.8, 
                 "description": "Bearish breakout with increasing volatility"}
            ]
        }
        
    def _precompute_regime_boundaries(self):
        """Precompute regime boundaries for optimization"""
        self.regime_boundaries = []
        
        for regime in self.regime_config['market_regimes']:
            self.regime_boundaries.append({
                'name': regime['name'],
                'dir_min': regime['directional_min'],
                'dir_max': regime['directional_max'],
                'vol_min': regime['volatility_min'],
                'vol_max': regime['volatility_max'],
                'dir_center': (regime['directional_min'] + regime['directional_max']) / 2,
                'vol_center': (regime['volatility_min'] + regime['volatility_max']) / 2
            })
            
    def _calculate_regime_internal(self, market_data: Dict[str, Any]) -> RegimeClassification:
        """
        Internal regime calculation implementation
        
        Args:
            market_data: Validated market data
            
        Returns:
            RegimeClassification result
        """
        timestamp = market_data['timestamp']
        
        # Calculate directional and volatility scores
        directional_score = self._calculate_directional_score(market_data)
        volatility_score = self._calculate_volatility_score(market_data)
        
        # Classify regime based on scores
        regime_result = self._classify_regime(directional_score, volatility_score)
        
        # Calculate alternative regimes
        alternatives = self._calculate_alternative_regimes(
            directional_score, volatility_score, regime_result['regime']
        )
        
        # Build result
        result = RegimeClassification(
            regime_id=regime_result['regime'],
            regime_name=regime_result['description'],
            confidence=regime_result['confidence'],
            timestamp=timestamp,
            volatility_score=volatility_score,
            directional_score=directional_score,
            alternative_regimes=alternatives,
            metadata={
                'regime_probability': regime_result['probability'],
                'detector': '18-regime',
                'version': '2.0.0',
                'scores': {
                    'directional': directional_score,
                    'volatility': volatility_score
                }
            }
        )
        
        return result
        
    def _calculate_directional_score(self, market_data: Dict[str, Any]) -> float:
        """
        Calculate directional score (-1 to +1)
        
        Positive values indicate bullish direction
        Negative values indicate bearish direction
        """
        score_components = []
        weights = []
        
        # Price momentum component
        if 'price_momentum' in market_data:
            score_components.append(market_data['price_momentum'])
            weights.append(0.3)
            
        # Option flow component
        if 'option_chain' in market_data and not market_data['option_chain'].empty:
            option_flow_score = self._calculate_option_flow_direction(market_data['option_chain'])
            score_components.append(option_flow_score)
            weights.append(0.25)
            
        # Greeks component
        if 'net_delta' in market_data:
            # Normalize net delta to -1 to +1
            delta_score = np.tanh(market_data['net_delta'] / 100)
            score_components.append(delta_score)
            weights.append(0.25)
            
        # Technical indicators
        if 'indicators' in market_data:
            tech_score = self._calculate_technical_direction(market_data['indicators'])
            score_components.append(tech_score)
            weights.append(0.2)
            
        # Calculate weighted average
        if score_components:
            total_weight = sum(weights[:len(score_components)])
            weighted_score = sum(s * w for s, w in zip(score_components, weights)) / total_weight
            return np.clip(weighted_score, -1.0, 1.0)
        else:
            return 0.0  # Neutral if no data
            
    def _calculate_volatility_score(self, market_data: Dict[str, Any]) -> float:
        """
        Calculate volatility score (0 to 1)
        
        0 indicates low volatility
        1 indicates high volatility
        """
        score_components = []
        weights = []
        
        # IV component
        if 'option_chain' in market_data and not market_data['option_chain'].empty:
            chain = market_data['option_chain']
            if 'implied_volatility' in chain.columns:
                avg_iv = chain['implied_volatility'].mean()
                # Normalize IV (assuming 0-100% range)
                iv_score = min(1.0, avg_iv / 50)  # 50% IV = 1.0 score
                score_components.append(iv_score)
                weights.append(0.4)
                
        # Historical volatility
        if 'historical_volatility' in market_data:
            hv_score = min(1.0, market_data['historical_volatility'] / 50)
            score_components.append(hv_score)
            weights.append(0.3)
            
        # ATR component
        if 'atr' in market_data and 'underlying_price' in market_data:
            atr_pct = market_data['atr'] / market_data['underlying_price']
            atr_score = min(1.0, atr_pct / 0.03)  # 3% ATR = 1.0 score
            score_components.append(atr_score)
            weights.append(0.2)
            
        # Volume spike component
        if 'volume_ratio' in market_data:
            # Volume ratio > 2 indicates high activity
            volume_score = min(1.0, market_data['volume_ratio'] / 2)
            score_components.append(volume_score)
            weights.append(0.1)
            
        # Calculate weighted average
        if score_components:
            total_weight = sum(weights[:len(score_components)])
            weighted_score = sum(s * w for s, w in zip(score_components, weights)) / total_weight
            return np.clip(weighted_score, 0.0, 1.0)
        else:
            return 0.5  # Medium volatility if no data
            
    def _calculate_option_flow_direction(self, option_chain: pd.DataFrame) -> float:
        """Calculate directional bias from option flow"""
        try:
            # Calculate call vs put volume
            call_volume = option_chain[option_chain['option_type'] == 'CE']['volume'].sum()
            put_volume = option_chain[option_chain['option_type'] == 'PE']['volume'].sum()
            
            total_volume = call_volume + put_volume
            if total_volume > 0:
                # Convert to -1 to +1 scale
                call_ratio = call_volume / total_volume
                return 2 * call_ratio - 1
            return 0
            
        except Exception as e:
            logger.warning(f"Error calculating option flow direction: {e}")
            return 0
            
    def _calculate_technical_direction(self, indicators: Dict[str, Any]) -> float:
        """Calculate directional score from technical indicators"""
        scores = []
        
        # RSI component
        if 'rsi' in indicators:
            # Convert RSI to directional score
            rsi_score = (indicators['rsi'] - 50) / 50
            scores.append(np.clip(rsi_score, -1, 1))
            
        # MACD component
        if 'macd_histogram' in indicators:
            # Normalize MACD histogram
            macd_score = np.tanh(indicators['macd_histogram'])
            scores.append(macd_score)
            
        # Moving average component
        if 'price_vs_ma' in indicators:
            # Price relative to MA
            ma_score = np.tanh(indicators['price_vs_ma'])
            scores.append(ma_score)
            
        # Average all available scores
        if scores:
            return np.mean(scores)
        return 0
        
    def _classify_regime(self, directional_score: float, 
                        volatility_score: float) -> Dict[str, Any]:
        """Classify regime based on directional and volatility scores"""
        best_regime = None
        best_distance = float('inf')
        
        # Find closest regime using Euclidean distance
        for boundary in self.regime_boundaries:
            # Check if point is within regime boundaries
            if (boundary['dir_min'] <= directional_score <= boundary['dir_max'] and
                boundary['vol_min'] <= volatility_score <= boundary['vol_max']):
                
                # Calculate distance to regime center
                distance = np.sqrt(
                    (directional_score - boundary['dir_center']) ** 2 +
                    (volatility_score - boundary['vol_center']) ** 2
                )
                
                if distance < best_distance:
                    best_distance = distance
                    best_regime = boundary
                    
        # If no regime contains the point, find nearest
        if best_regime is None:
            for boundary in self.regime_boundaries:
                distance = np.sqrt(
                    (directional_score - boundary['dir_center']) ** 2 +
                    (volatility_score - boundary['vol_center']) ** 2
                )
                
                if distance < best_distance:
                    best_distance = distance
                    best_regime = boundary
                    
        # Calculate confidence based on distance
        max_distance = np.sqrt(2)  # Maximum possible distance in normalized space
        confidence = 1.0 - (best_distance / max_distance)
        confidence = max(self.confidence_threshold, confidence)
        
        # Get regime details
        regime_name = best_regime['name']
        regime_info = next(r for r in self.regime_config['market_regimes'] 
                          if r['name'] == regime_name)
        
        return {
            'regime': regime_name,
            'description': regime_info['description'],
            'confidence': confidence,
            'probability': confidence,  # For compatibility
            'distance': best_distance
        }
        
    def _calculate_alternative_regimes(self, directional_score: float, 
                                     volatility_score: float,
                                     primary_regime: str) -> List[Tuple[str, float]]:
        """Calculate alternative regime possibilities"""
        alternatives = []
        
        # Calculate distance to all regimes
        for boundary in self.regime_boundaries:
            if boundary['name'] == primary_regime:
                continue
                
            # Calculate distance to regime center
            distance = np.sqrt(
                (directional_score - boundary['dir_center']) ** 2 +
                (volatility_score - boundary['vol_center']) ** 2
            )
            
            # Convert distance to confidence (inverse relationship)
            max_distance = np.sqrt(2)
            confidence = max(0.0, 1.0 - (distance / max_distance))
            
            if confidence > 0.1:  # Only include if reasonably close
                alternatives.append((boundary['name'], confidence * 0.8))  # Cap at 0.8
                
        # Sort by confidence
        alternatives.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 3
        return alternatives[:3]