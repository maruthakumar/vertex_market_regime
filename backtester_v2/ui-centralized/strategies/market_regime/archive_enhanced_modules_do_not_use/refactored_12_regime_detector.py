"""
Refactored 12-Regime Classification System using Base Class

This module implements the 12-regime classification system based on the
Triple Rolling Straddle Market Regime implementation plan, now inheriting
from RegimeDetectorBase to eliminate code duplication.

Features:
1. Complete 12-regime classification system
2. Volatility component analysis (LOW/MODERATE/HIGH)
3. Trend component analysis (DIRECTIONAL/NONDIRECTIONAL)
4. Structure component analysis (TRENDING/RANGE)
5. Inherits caching, monitoring, and validation from base class

Author: Market Regime System Optimizer
Date: 2025-07-07
Version: 2.0.0 - Refactored with base class
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass
import json
from pathlib import Path

# Import base class
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from base.regime_detector_base import RegimeDetectorBase, RegimeClassification

logger = logging.getLogger(__name__)


@dataclass
class Regime12Components:
    """Components for 12-regime classification"""
    volatility_level: str  # LOW, MODERATE, HIGH
    trend_type: str  # DIRECTIONAL, NONDIRECTIONAL
    structure_type: str  # TRENDING, RANGE
    component_scores: Dict[str, float]


class Refactored12RegimeDetector(RegimeDetectorBase):
    """
    Refactored 12-Regime Classification System
    
    Implements Volatility (3) × Trend (2) × Structure (2) = 12 regimes
    architecture using the common base class functionality.
    """
    
    def _initialize_detector(self):
        """Initialize 12-regime specific configuration"""
        # 12-regime mapping
        self.regime_definitions = {
            'R1': {'volatility': 'LOW', 'trend': 'DIRECTIONAL', 'structure': 'TRENDING', 
                   'name': 'Low Vol Directional Trending'},
            'R2': {'volatility': 'LOW', 'trend': 'DIRECTIONAL', 'structure': 'RANGE', 
                   'name': 'Low Vol Directional Range'},
            'R3': {'volatility': 'LOW', 'trend': 'NONDIRECTIONAL', 'structure': 'TRENDING', 
                   'name': 'Low Vol Non-Directional Trending'},
            'R4': {'volatility': 'LOW', 'trend': 'NONDIRECTIONAL', 'structure': 'RANGE', 
                   'name': 'Low Vol Non-Directional Range'},
            'R5': {'volatility': 'MODERATE', 'trend': 'DIRECTIONAL', 'structure': 'TRENDING', 
                   'name': 'Moderate Vol Directional Trending'},
            'R6': {'volatility': 'MODERATE', 'trend': 'DIRECTIONAL', 'structure': 'RANGE', 
                   'name': 'Moderate Vol Directional Range'},
            'R7': {'volatility': 'MODERATE', 'trend': 'NONDIRECTIONAL', 'structure': 'TRENDING', 
                   'name': 'Moderate Vol Non-Directional Trending'},
            'R8': {'volatility': 'MODERATE', 'trend': 'NONDIRECTIONAL', 'structure': 'RANGE', 
                   'name': 'Moderate Vol Non-Directional Range'},
            'R9': {'volatility': 'HIGH', 'trend': 'DIRECTIONAL', 'structure': 'TRENDING', 
                   'name': 'High Vol Directional Trending'},
            'R10': {'volatility': 'HIGH', 'trend': 'DIRECTIONAL', 'structure': 'RANGE', 
                    'name': 'High Vol Directional Range'},
            'R11': {'volatility': 'HIGH', 'trend': 'NONDIRECTIONAL', 'structure': 'TRENDING', 
                    'name': 'High Vol Non-Directional Trending'},
            'R12': {'volatility': 'HIGH', 'trend': 'NONDIRECTIONAL', 'structure': 'RANGE', 
                    'name': 'High Vol Non-Directional Range'}
        }
        
        # Component thresholds
        self.volatility_thresholds = {
            'low_threshold': 0.15,
            'high_threshold': 0.35
        }
        
        self.trend_thresholds = {
            'directional_min': 0.6
        }
        
        self.structure_thresholds = {
            'trending_min': 0.5
        }
        
        # Load 18->12 regime mapping if needed
        self.regime_18_to_12_map = self._load_regime_mapping()
        
        logger.info("12-Regime detector initialized with refactored architecture")
        
    def get_regime_count(self) -> int:
        """Get number of regimes supported by this detector"""
        return 12
        
    def get_regime_mapping(self) -> Dict[str, str]:
        """Get mapping of regime IDs to descriptions"""
        return {rid: rdef['name'] for rid, rdef in self.regime_definitions.items()}
        
    def _load_regime_mapping(self) -> Dict[str, str]:
        """Load 18->12 regime mapping configuration"""
        # Default mapping from 18 regimes to 12 regimes
        return {
            'Strong_Bullish': 'R1',
            'Moderate_Bullish': 'R5',
            'Weak_Bullish': 'R3',
            'Neutral_Bullish': 'R4',
            'Sideways_High_Vol': 'R8',
            'Sideways_Low_Vol': 'R4',
            'Neutral_Bearish': 'R4',
            'Weak_Bearish': 'R3',
            'Moderate_Bearish': 'R6',
            'Strong_Bearish': 'R2',
            'Volatile_Bullish': 'R9',
            'Volatile_Bearish': 'R10',
            'Trending_Bullish': 'R1',
            'Trending_Bearish': 'R2',
            'Range_Bound_High': 'R10',
            'Range_Bound_Low': 'R4',
            'Breakout_Bullish': 'R5',
            'Breakout_Bearish': 'R6'
        }
        
    def _calculate_regime_internal(self, market_data: Dict[str, Any]) -> RegimeClassification:
        """
        Internal regime calculation implementation
        
        Args:
            market_data: Validated market data
            
        Returns:
            RegimeClassification result
        """
        timestamp = market_data['timestamp']
        
        # Calculate components
        components = self._calculate_regime_components(market_data)
        
        # Determine regime from components
        regime_id = self._determine_regime_from_components(components)
        regime_def = self.regime_definitions[regime_id]
        
        # Calculate confidence
        confidence = self._calculate_regime_confidence(components, regime_id)
        
        # Get alternative regimes
        alternatives = self._get_alternative_regimes(components, regime_id)
        
        # Build result
        result = RegimeClassification(
            regime_id=regime_id,
            regime_name=regime_def['name'],
            confidence=confidence,
            timestamp=timestamp,
            volatility_score=components.component_scores['volatility'],
            directional_score=components.component_scores['trend'],
            alternative_regimes=alternatives,
            metadata={
                'components': {
                    'volatility': components.volatility_level,
                    'trend': components.trend_type,
                    'structure': components.structure_type
                },
                'scores': components.component_scores,
                'detector': '12-regime',
                'version': '2.0.0'
            }
        )
        
        return result
        
    def _calculate_regime_components(self, market_data: Dict[str, Any]) -> Regime12Components:
        """Calculate the three components for regime classification"""
        # Extract features from market data
        features = self._extract_features(market_data)
        
        # Calculate volatility component
        volatility_score = self._calculate_volatility_score(features)
        volatility_level = self._classify_volatility(volatility_score)
        
        # Calculate trend component
        trend_score = self._calculate_trend_score(features)
        trend_type = self._classify_trend(trend_score)
        
        # Calculate structure component
        structure_score = self._calculate_structure_score(features)
        structure_type = self._classify_structure(structure_score)
        
        return Regime12Components(
            volatility_level=volatility_level,
            trend_type=trend_type,
            structure_type=structure_type,
            component_scores={
                'volatility': volatility_score,
                'trend': trend_score,
                'structure': structure_score
            }
        )
        
    def _extract_features(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant features from market data"""
        features = {}
        
        # Basic price features
        features['underlying_price'] = market_data['underlying_price']
        features['timestamp'] = market_data['timestamp']
        
        # Extract from option chain if available
        if 'option_chain' in market_data and not market_data['option_chain'].empty:
            chain = market_data['option_chain']
            
            # IV features
            if 'implied_volatility' in chain.columns:
                features['avg_iv'] = chain['implied_volatility'].mean()
                features['iv_skew'] = self._calculate_iv_skew(chain)
            
            # Volume features
            if 'volume' in chain.columns:
                features['total_volume'] = chain['volume'].sum()
                features['call_put_ratio'] = self._calculate_call_put_ratio(chain)
                
            # Greeks if available
            if 'delta' in chain.columns:
                features['net_delta'] = chain['delta'].sum()
                
        # Technical indicators if provided
        if 'indicators' in market_data:
            features.update(market_data['indicators'])
            
        return features
        
    def _calculate_volatility_score(self, features: Dict[str, Any]) -> float:
        """Calculate volatility score (0-1)"""
        score = 0.5  # Default neutral score
        
        # Use IV if available
        if 'avg_iv' in features:
            iv = features['avg_iv']
            # Normalize IV to 0-1 scale (assuming 0-100% range)
            score = min(1.0, iv / 100.0)
            
        # Adjust based on other volatility indicators
        if 'atr' in features:
            atr_score = min(1.0, features['atr'] / features.get('underlying_price', 1))
            score = 0.7 * score + 0.3 * atr_score
            
        return score
        
    def _calculate_trend_score(self, features: Dict[str, Any]) -> float:
        """Calculate trend score (0-1, where >0.5 is directional)"""
        score = 0.5  # Default neutral
        
        # Use various trend indicators
        if 'rsi' in features:
            # RSI deviation from 50
            rsi_score = abs(features['rsi'] - 50) / 50
            score = 0.3 * score + 0.3 * rsi_score
            
        if 'macd_signal' in features:
            # Normalized MACD signal
            macd_score = min(1.0, abs(features['macd_signal']) / features.get('underlying_price', 1))
            score = 0.5 * score + 0.2 * macd_score
            
        if 'net_delta' in features:
            # Net delta indicates directional bias
            delta_score = min(1.0, abs(features['net_delta']) / 100)
            score = 0.7 * score + 0.3 * delta_score
            
        return score
        
    def _calculate_structure_score(self, features: Dict[str, Any]) -> float:
        """Calculate structure score (0-1, where >0.5 is trending)"""
        score = 0.5  # Default neutral
        
        # Use pattern and structure indicators
        if 'adx' in features:
            # ADX directly indicates trending vs ranging
            score = min(1.0, features['adx'] / 50)
            
        if 'bollinger_width' in features:
            # Wider bands suggest trending
            bb_score = min(1.0, features['bollinger_width'] / features.get('underlying_price', 1))
            score = 0.7 * score + 0.3 * bb_score
            
        return score
        
    def _classify_volatility(self, score: float) -> str:
        """Classify volatility level based on score"""
        if score < self.volatility_thresholds['low_threshold']:
            return 'LOW'
        elif score > self.volatility_thresholds['high_threshold']:
            return 'HIGH'
        else:
            return 'MODERATE'
            
    def _classify_trend(self, score: float) -> str:
        """Classify trend type based on score"""
        if score >= self.trend_thresholds['directional_min']:
            return 'DIRECTIONAL'
        else:
            return 'NONDIRECTIONAL'
            
    def _classify_structure(self, score: float) -> str:
        """Classify structure type based on score"""
        if score >= self.structure_thresholds['trending_min']:
            return 'TRENDING'
        else:
            return 'RANGE'
            
    def _determine_regime_from_components(self, components: Regime12Components) -> str:
        """Determine regime ID from components"""
        # Find matching regime
        for regime_id, regime_def in self.regime_definitions.items():
            if (regime_def['volatility'] == components.volatility_level and
                regime_def['trend'] == components.trend_type and
                regime_def['structure'] == components.structure_type):
                return regime_id
                
        # Fallback to R8 (moderate vol, non-directional, range)
        logger.warning(f"No regime match for components: {components}")
        return 'R8'
        
    def _calculate_regime_confidence(self, components: Regime12Components, regime_id: str) -> float:
        """Calculate confidence in regime classification"""
        # Base confidence from component scores
        scores = components.component_scores
        
        # How clearly defined are the components?
        volatility_clarity = abs(scores['volatility'] - 0.5) * 2  # 0-1
        trend_clarity = abs(scores['trend'] - 0.5) * 2
        structure_clarity = abs(scores['structure'] - 0.5) * 2
        
        # Average clarity
        confidence = (volatility_clarity + trend_clarity + structure_clarity) / 3
        
        # Apply minimum threshold
        confidence = max(self.confidence_threshold, confidence)
        
        return min(1.0, confidence)
        
    def _get_alternative_regimes(self, components: Regime12Components, 
                                primary_regime: str) -> List[Tuple[str, float]]:
        """Get alternative regime possibilities"""
        alternatives = []
        
        # Consider adjacent regimes based on component scores
        scores = components.component_scores
        
        # Check regimes with one component different
        for regime_id, regime_def in self.regime_definitions.items():
            if regime_id == primary_regime:
                continue
                
            # Calculate similarity
            similarity = 0
            if regime_def['volatility'] == components.volatility_level:
                similarity += 0.33
            if regime_def['trend'] == components.trend_type:
                similarity += 0.33
            if regime_def['structure'] == components.structure_type:
                similarity += 0.34
                
            # Only include reasonably similar regimes
            if similarity >= 0.66:  # At least 2 components match
                # Adjust by component score proximity
                confidence = similarity * 0.8  # Max 0.8 for alternatives
                alternatives.append((regime_id, confidence))
                
        # Sort by confidence
        alternatives.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 3
        return alternatives[:3]
        
    def _calculate_iv_skew(self, option_chain: pd.DataFrame) -> float:
        """Calculate IV skew from option chain"""
        try:
            calls = option_chain[option_chain['option_type'] == 'CE']
            puts = option_chain[option_chain['option_type'] == 'PE']
            
            if len(calls) > 0 and len(puts) > 0:
                call_iv = calls['implied_volatility'].mean()
                put_iv = puts['implied_volatility'].mean()
                return (put_iv - call_iv) / ((put_iv + call_iv) / 2)
            return 0
        except:
            return 0
            
    def _calculate_call_put_ratio(self, option_chain: pd.DataFrame) -> float:
        """Calculate call/put volume ratio"""
        try:
            call_volume = option_chain[option_chain['option_type'] == 'CE']['volume'].sum()
            put_volume = option_chain[option_chain['option_type'] == 'PE']['volume'].sum()
            
            if put_volume > 0:
                return call_volume / put_volume
            return 1.0
        except:
            return 1.0
            
    def convert_from_18_regime(self, regime_18: str) -> str:
        """Convert 18-regime classification to 12-regime"""
        return self.regime_18_to_12_map.get(regime_18, 'R8')  # Default to R8