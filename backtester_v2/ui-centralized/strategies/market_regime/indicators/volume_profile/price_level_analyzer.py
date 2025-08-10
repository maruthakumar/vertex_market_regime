"""
Price Level Analyzer
===================

Analyzes price action at specific volume levels to determine
strength and potential regime transitions.

Author: Market Regime Refactoring Team
Date: 2025-07-08
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class PriceLevelAnalyzer:
    """
    Analyzes price behavior at key volume levels
    
    Complements VolumeProfileAnalyzer by examining how price
    interacts with high/low volume nodes.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the price level analyzer"""
        self.config = config or {}
        
        # Configuration
        self.reaction_threshold = self.config.get('reaction_threshold', 0.002)  # 0.2%
        self.breakout_threshold = self.config.get('breakout_threshold', 0.005)  # 0.5%
        self.rejection_candles = self.config.get('rejection_candles', 3)
        self.confirmation_candles = self.config.get('confirmation_candles', 2)
        
        # State tracking
        self._level_interactions = {}
        self._breakout_signals = []
        
        logger.info("PriceLevelAnalyzer initialized")
    
    def analyze_level_interaction(self,
                                price_data: pd.DataFrame,
                                level: float,
                                level_type: str = 'resistance') -> Dict[str, Any]:
        """
        Analyze how price interacts with a specific level
        
        Args:
            price_data: DataFrame with OHLCV data
            level: Price level to analyze
            level_type: 'support' or 'resistance'
            
        Returns:
            Dict with interaction analysis
        """
        try:
            results = {
                'level': level,
                'level_type': level_type,
                'interaction_type': 'none',
                'strength': 0.0,
                'breakout_probability': 0.0,
                'touches': 0,
                'rejections': 0,
                'penetrations': 0
            }
            
            # Count interactions
            for i in range(len(price_data)):
                high = price_data.iloc[i]['high']
                low = price_data.iloc[i]['low']
                close = price_data.iloc[i]['close']
                
                # Check if price touched the level
                if low <= level <= high:
                    results['touches'] += 1
                    
                    # Check for rejection
                    if level_type == 'resistance' and close < level:
                        if high >= level * (1 + self.reaction_threshold):
                            results['rejections'] += 1
                    elif level_type == 'support' and close > level:
                        if low <= level * (1 - self.reaction_threshold):
                            results['rejections'] += 1
                    
                    # Check for penetration
                    if level_type == 'resistance' and close > level:
                        results['penetrations'] += 1
                    elif level_type == 'support' and close < level:
                        results['penetrations'] += 1
            
            # Determine interaction type
            if results['touches'] == 0:
                results['interaction_type'] = 'no_test'
            elif results['rejections'] > results['penetrations']:
                results['interaction_type'] = 'respected'
                results['strength'] = results['rejections'] / results['touches']
            elif results['penetrations'] > results['rejections']:
                results['interaction_type'] = 'broken'
                results['breakout_probability'] = results['penetrations'] / results['touches']
            else:
                results['interaction_type'] = 'contested'
                results['strength'] = 0.5
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing level interaction: {e}")
            return {
                'level': level,
                'level_type': level_type,
                'interaction_type': 'error',
                'strength': 0.0
            }
    
    def detect_breakout_patterns(self,
                               price_data: pd.DataFrame,
                               volume_levels: Dict[str, List[float]]) -> List[Dict[str, Any]]:
        """
        Detect potential breakout patterns at volume levels
        
        Args:
            price_data: DataFrame with OHLCV data
            volume_levels: Dict with 'support' and 'resistance' levels
            
        Returns:
            List of breakout signals
        """
        breakout_signals = []
        
        try:
            current_price = price_data.iloc[-1]['close']
            recent_data = price_data.tail(20)  # Last 20 candles
            
            # Check resistance breakouts
            for resistance in volume_levels.get('resistance', []):
                if self._is_breakout_pattern(recent_data, resistance, 'resistance'):
                    signal = {
                        'type': 'resistance_breakout',
                        'level': resistance,
                        'current_price': current_price,
                        'distance_pct': (current_price - resistance) / resistance,
                        'strength': self._calculate_breakout_strength(recent_data, resistance),
                        'timestamp': datetime.now()
                    }
                    breakout_signals.append(signal)
            
            # Check support breakdowns
            for support in volume_levels.get('support', []):
                if self._is_breakout_pattern(recent_data, support, 'support'):
                    signal = {
                        'type': 'support_breakdown',
                        'level': support,
                        'current_price': current_price,
                        'distance_pct': (support - current_price) / support,
                        'strength': self._calculate_breakout_strength(recent_data, support),
                        'timestamp': datetime.now()
                    }
                    breakout_signals.append(signal)
            
            # Cache signals
            self._breakout_signals = breakout_signals
            
            return breakout_signals
            
        except Exception as e:
            logger.error(f"Error detecting breakout patterns: {e}")
            return []
    
    def calculate_level_strength_score(self,
                                     market_data: pd.DataFrame,
                                     volume_profile_data: Dict[str, Any]) -> float:
        """
        Calculate overall strength score for price levels
        
        Args:
            market_data: Market data DataFrame
            volume_profile_data: Volume profile analysis results
            
        Returns:
            float: Level strength score between -1 and 1
        """
        try:
            score_components = []
            
            # 1. Price position relative to value area
            current_price = market_data.iloc[-1]['close']
            va_high = volume_profile_data.get('value_area_high', 0)
            va_low = volume_profile_data.get('value_area_low', 0)
            
            if va_high > 0 and va_low > 0:
                if current_price > va_high:
                    # Above value area - bullish bias
                    position_score = min((current_price - va_high) / va_high * 10, 1.0)
                elif current_price < va_low:
                    # Below value area - bearish bias
                    position_score = max((va_low - current_price) / va_low * -10, -1.0)
                else:
                    # Inside value area - neutral
                    position_score = 0.0
                
                score_components.append(('position', position_score, 0.4))
            
            # 2. Recent breakout signals
            if self._breakout_signals:
                recent_signal = self._breakout_signals[-1]
                if recent_signal['type'] == 'resistance_breakout':
                    breakout_score = recent_signal['strength']
                else:
                    breakout_score = -recent_signal['strength']
                
                score_components.append(('breakout', breakout_score, 0.3))
            
            # 3. Level density (more levels = stronger structure)
            hvn_count = volume_profile_data.get('hvn_count', 0)
            density_score = np.tanh(hvn_count / 10)  # Normalize
            score_components.append(('density', density_score * 0.5, 0.3))
            
            # Combine scores
            total_score = 0.0
            total_weight = 0.0
            
            for name, score, weight in score_components:
                total_score += score * weight
                total_weight += weight
                logger.debug(f"Price level {name}: {score:.3f} (weight: {weight})")
            
            if total_weight > 0:
                final_score = total_score / total_weight
            else:
                final_score = 0.0
            
            return np.clip(final_score, -1.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating level strength score: {e}")
            return 0.0
    
    def _is_breakout_pattern(self,
                           price_data: pd.DataFrame,
                           level: float,
                           level_type: str) -> bool:
        """Check if price shows breakout pattern at level"""
        if len(price_data) < self.confirmation_candles:
            return False
        
        recent_closes = price_data.tail(self.confirmation_candles)['close'].values
        
        if level_type == 'resistance':
            # All recent closes above level
            return all(close > level * (1 + self.breakout_threshold) for close in recent_closes)
        else:
            # All recent closes below level
            return all(close < level * (1 - self.breakout_threshold) for close in recent_closes)
    
    def _calculate_breakout_strength(self,
                                   price_data: pd.DataFrame,
                                   level: float) -> float:
        """Calculate strength of breakout"""
        try:
            # Volume surge
            recent_volume = price_data.tail(5)['volume'].mean()
            avg_volume = price_data['volume'].mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Price momentum
            price_change = (price_data.iloc[-1]['close'] - price_data.iloc[-5]['close']) / price_data.iloc[-5]['close']
            momentum = abs(price_change) * 10
            
            # Combine factors
            strength = (volume_ratio * 0.6 + momentum * 0.4)
            
            return np.clip(strength, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating breakout strength: {e}")
            return 0.5
    
    def get_regime_bias_from_levels(self) -> str:
        """Get regime bias based on recent level interactions"""
        if not self._breakout_signals:
            return "neutral"
        
        # Check last few signals
        recent_signals = self._breakout_signals[-3:]
        
        bullish_count = sum(1 for s in recent_signals if s['type'] == 'resistance_breakout')
        bearish_count = sum(1 for s in recent_signals if s['type'] == 'support_breakdown')
        
        if bullish_count > bearish_count:
            return "bullish"
        elif bearish_count > bullish_count:
            return "bearish"
        else:
            return "neutral"