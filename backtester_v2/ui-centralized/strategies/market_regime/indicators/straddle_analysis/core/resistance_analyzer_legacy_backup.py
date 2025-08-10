"""
Support and Resistance Level Analyzer

Integrates support/resistance level detection with straddle analysis
to identify key price levels that affect option pricing and regime.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
from collections import deque
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class SupportResistanceLevel:
    """Represents a support or resistance level"""
    price: float
    level_type: str  # 'support' or 'resistance'
    strength: float  # 0-1, higher means stronger level
    touches: int     # Number of times price touched this level
    last_touch: pd.Timestamp
    formation_time: pd.Timestamp
    

@dataclass
class ResistanceAnalysisResult:
    """Result of resistance analysis"""
    timestamp: pd.Timestamp
    current_price: float
    
    # Identified levels
    support_levels: List[SupportResistanceLevel]
    resistance_levels: List[SupportResistanceLevel]
    
    # Key metrics
    nearest_support: Optional[float]
    nearest_resistance: Optional[float]
    support_distance: float
    resistance_distance: float
    
    # Level statistics
    level_density: float  # How many levels in current price area
    price_position: str   # 'near_support', 'near_resistance', 'neutral'
    
    # Regime indicators
    level_strength_score: float
    breakout_probability: float
    

class ResistanceAnalyzer:
    """
    Support and Resistance Level Analyzer
    
    Identifies key price levels using:
    - Historical price pivots
    - Volume-weighted levels
    - Option strike concentrations
    - Previous day high/low/close
    - Psychological levels (round numbers)
    """
    
    def __init__(self, window_sizes: List[int] = [3, 5, 10, 15]):
        """
        Initialize Resistance Analyzer
        
        Args:
            window_sizes: Rolling window sizes for analysis
        """
        self.window_sizes = window_sizes
        
        # Level detection parameters
        self.min_touches = 2          # Minimum touches to confirm level
        self.level_tolerance = 0.002  # 0.2% tolerance for level clustering
        self.level_decay_factor = 0.95  # Decay factor for old levels
        self.max_levels = 10          # Maximum levels to track
        
        # Price history for level detection
        self.price_history = deque(maxlen=500)  # Keep last 500 data points
        self.volume_history = deque(maxlen=500)
        
        # Identified levels
        self.support_levels: List[SupportResistanceLevel] = []
        self.resistance_levels: List[SupportResistanceLevel] = []
        
        # Performance tracking
        self.level_accuracy_history = deque(maxlen=100)
        
        self.logger = logging.getLogger(f"{__name__}.resistance_analyzer")
        self.logger.info("Resistance Analyzer initialized")
    
    def analyze(self, data: Dict[str, Any], timestamp: pd.Timestamp) -> Optional[ResistanceAnalysisResult]:
        """
        Perform support/resistance analysis
        
        Args:
            data: Market data dictionary
            timestamp: Current timestamp
            
        Returns:
            ResistanceAnalysisResult or None if insufficient data
        """
        try:
            current_price = data.get('underlying_price') or data.get('spot_price')
            if not current_price:
                return None
            
            # Update price history
            self._update_price_history(data, timestamp)
            
            # Identify new levels
            self._identify_pivot_levels()
            self._identify_volume_levels()
            self._identify_strike_levels(data)
            self._identify_psychological_levels(current_price)
            
            # Update existing levels
            self._update_level_touches(current_price, timestamp)
            self._decay_old_levels(timestamp)
            self._consolidate_nearby_levels()
            
            # Sort and limit levels
            self._sort_and_limit_levels()
            
            # Find nearest levels
            nearest_support, support_distance = self._find_nearest_support(current_price)
            nearest_resistance, resistance_distance = self._find_nearest_resistance(current_price)
            
            # Calculate metrics
            level_density = self._calculate_level_density(current_price)
            price_position = self._determine_price_position(current_price, support_distance, resistance_distance)
            level_strength_score = self._calculate_level_strength_score()
            breakout_probability = self._calculate_breakout_probability(
                current_price, nearest_support, nearest_resistance, data
            )
            
            # Create result
            result = ResistanceAnalysisResult(
                timestamp=timestamp,
                current_price=current_price,
                support_levels=self.support_levels.copy(),
                resistance_levels=self.resistance_levels.copy(),
                nearest_support=nearest_support,
                nearest_resistance=nearest_resistance,
                support_distance=support_distance,
                resistance_distance=resistance_distance,
                level_density=level_density,
                price_position=price_position,
                level_strength_score=level_strength_score,
                breakout_probability=breakout_probability
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in resistance analysis: {e}")
            return None
    
    def _update_price_history(self, data: Dict[str, Any], timestamp: pd.Timestamp):
        """Update price and volume history"""
        price_data = {
            'timestamp': timestamp,
            'open': data.get('open', data.get('underlying_price', 0)),
            'high': data.get('high', data.get('underlying_price', 0)),
            'low': data.get('low', data.get('underlying_price', 0)),
            'close': data.get('close', data.get('underlying_price', 0)),
            'volume': data.get('volume', 0)
        }
        
        self.price_history.append(price_data)
        self.volume_history.append(data.get('volume', 0))
    
    def _identify_pivot_levels(self):
        """Identify support/resistance from price pivots"""
        if len(self.price_history) < 20:
            return
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(list(self.price_history))
        
        # Find pivot highs and lows
        for i in range(10, len(df) - 10):
            # Pivot high
            if df.iloc[i]['high'] == df.iloc[i-10:i+11]['high'].max():
                self._add_resistance_level(
                    price=df.iloc[i]['high'],
                    timestamp=df.iloc[i]['timestamp'],
                    strength=0.7
                )
            
            # Pivot low
            if df.iloc[i]['low'] == df.iloc[i-10:i+11]['low'].min():
                self._add_support_level(
                    price=df.iloc[i]['low'],
                    timestamp=df.iloc[i]['timestamp'],
                    strength=0.7
                )
    
    def _identify_volume_levels(self):
        """Identify levels with high volume"""
        if len(self.price_history) < 20:
            return
        
        # Find high volume nodes
        df = pd.DataFrame(list(self.price_history))
        volume_mean = df['volume'].mean()
        volume_std = df['volume'].std()
        
        # High volume threshold
        high_volume_threshold = volume_mean + 1.5 * volume_std
        
        for idx, row in df.iterrows():
            if row['volume'] > high_volume_threshold:
                # Add as both support and resistance (high volume node)
                price = (row['high'] + row['low']) / 2
                self._add_support_level(price, row['timestamp'], strength=0.8)
                self._add_resistance_level(price, row['timestamp'], strength=0.8)
    
    def _identify_strike_levels(self, data: Dict[str, Any]):
        """Identify levels from option strike concentrations"""
        # Look for strike data
        strikes = data.get('strikes', [])
        if not strikes:
            return
        
        underlying_price = data.get('underlying_price', 0)
        if underlying_price <= 0:
            return
        
        # Add major strikes as levels
        for strike in strikes:
            distance_pct = abs(strike - underlying_price) / underlying_price
            
            # Only consider strikes within 5% of current price
            if distance_pct <= 0.05:
                if strike > underlying_price:
                    self._add_resistance_level(
                        price=strike,
                        timestamp=pd.Timestamp.now(),
                        strength=0.6
                    )
                else:
                    self._add_support_level(
                        price=strike,
                        timestamp=pd.Timestamp.now(),
                        strength=0.6
                    )
    
    def _identify_psychological_levels(self, current_price: float):
        """Identify psychological levels (round numbers)"""
        # Determine rounding interval based on price
        if current_price < 100:
            interval = 5
        elif current_price < 1000:
            interval = 25
        elif current_price < 10000:
            interval = 100
        else:
            interval = 250
        
        # Find nearby round levels
        lower_round = (current_price // interval) * interval
        upper_round = lower_round + interval
        
        # Add as weak levels
        self._add_support_level(
            price=lower_round,
            timestamp=pd.Timestamp.now(),
            strength=0.4
        )
        self._add_resistance_level(
            price=upper_round,
            timestamp=pd.Timestamp.now(),
            strength=0.4
        )
    
    def _add_support_level(self, price: float, timestamp: pd.Timestamp, strength: float):
        """Add or update support level"""
        # Check if level already exists
        for level in self.support_levels:
            if abs(level.price - price) / price < self.level_tolerance:
                # Update existing level
                level.touches += 1
                level.strength = min(1.0, level.strength + 0.1)
                level.last_touch = timestamp
                return
        
        # Add new level
        new_level = SupportResistanceLevel(
            price=price,
            level_type='support',
            strength=strength,
            touches=1,
            last_touch=timestamp,
            formation_time=timestamp
        )
        self.support_levels.append(new_level)
    
    def _add_resistance_level(self, price: float, timestamp: pd.Timestamp, strength: float):
        """Add or update resistance level"""
        # Check if level already exists
        for level in self.resistance_levels:
            if abs(level.price - price) / price < self.level_tolerance:
                # Update existing level
                level.touches += 1
                level.strength = min(1.0, level.strength + 0.1)
                level.last_touch = timestamp
                return
        
        # Add new level
        new_level = SupportResistanceLevel(
            price=price,
            level_type='resistance',
            strength=strength,
            touches=1,
            last_touch=timestamp,
            formation_time=timestamp
        )
        self.resistance_levels.append(new_level)
    
    def _update_level_touches(self, current_price: float, timestamp: pd.Timestamp):
        """Update touches for levels near current price"""
        touch_tolerance = self.level_tolerance * 2  # Wider tolerance for touches
        
        # Check support levels
        for level in self.support_levels:
            if abs(current_price - level.price) / level.price < touch_tolerance:
                if current_price >= level.price:  # Bounced off support
                    level.touches += 1
                    level.strength = min(1.0, level.strength + 0.05)
                    level.last_touch = timestamp
        
        # Check resistance levels
        for level in self.resistance_levels:
            if abs(current_price - level.price) / level.price < touch_tolerance:
                if current_price <= level.price:  # Rejected at resistance
                    level.touches += 1
                    level.strength = min(1.0, level.strength + 0.05)
                    level.last_touch = timestamp
    
    def _decay_old_levels(self, current_time: pd.Timestamp):
        """Decay strength of old levels"""
        decay_threshold = pd.Timedelta(hours=24)  # Start decay after 24 hours
        
        # Decay support levels
        self.support_levels = [
            level for level in self.support_levels
            if self._decay_level(level, current_time, decay_threshold) > 0.1
        ]
        
        # Decay resistance levels
        self.resistance_levels = [
            level for level in self.resistance_levels
            if self._decay_level(level, current_time, decay_threshold) > 0.1
        ]
    
    def _decay_level(self, level: SupportResistanceLevel, current_time: pd.Timestamp, 
                     decay_threshold: pd.Timedelta) -> float:
        """Decay a single level and return new strength"""
        age = current_time - level.last_touch
        
        if age > decay_threshold:
            decay_periods = (age - decay_threshold).total_seconds() / 3600  # Hours
            level.strength *= (self.level_decay_factor ** decay_periods)
        
        return level.strength
    
    def _consolidate_nearby_levels(self):
        """Consolidate nearby levels into stronger levels"""
        # Consolidate support levels
        self.support_levels = self._consolidate_level_list(self.support_levels)
        
        # Consolidate resistance levels
        self.resistance_levels = self._consolidate_level_list(self.resistance_levels)
    
    def _consolidate_level_list(self, levels: List[SupportResistanceLevel]) -> List[SupportResistanceLevel]:
        """Consolidate a list of levels"""
        if len(levels) <= 1:
            return levels
        
        # Sort by price
        sorted_levels = sorted(levels, key=lambda x: x.price)
        consolidated = []
        
        i = 0
        while i < len(sorted_levels):
            current_level = sorted_levels[i]
            cluster = [current_level]
            
            # Find all nearby levels
            j = i + 1
            while j < len(sorted_levels):
                if abs(sorted_levels[j].price - current_level.price) / current_level.price < self.level_tolerance:
                    cluster.append(sorted_levels[j])
                    j += 1
                else:
                    break
            
            # Consolidate cluster
            if len(cluster) > 1:
                # Weighted average price
                total_weight = sum(l.strength * l.touches for l in cluster)
                avg_price = sum(l.price * l.strength * l.touches for l in cluster) / total_weight
                
                # Combined strength
                max_strength = max(l.strength for l in cluster)
                total_touches = sum(l.touches for l in cluster)
                
                # Most recent touch
                last_touch = max(l.last_touch for l in cluster)
                formation_time = min(l.formation_time for l in cluster)
                
                consolidated_level = SupportResistanceLevel(
                    price=avg_price,
                    level_type=current_level.level_type,
                    strength=min(1.0, max_strength * 1.2),  # Boost for consolidation
                    touches=total_touches,
                    last_touch=last_touch,
                    formation_time=formation_time
                )
                consolidated.append(consolidated_level)
            else:
                consolidated.append(current_level)
            
            i = j
        
        return consolidated
    
    def _sort_and_limit_levels(self):
        """Sort levels by strength and limit count"""
        # Sort by strength (descending)
        self.support_levels.sort(key=lambda x: x.strength, reverse=True)
        self.resistance_levels.sort(key=lambda x: x.strength, reverse=True)
        
        # Limit to max levels
        self.support_levels = self.support_levels[:self.max_levels]
        self.resistance_levels = self.resistance_levels[:self.max_levels]
    
    def _find_nearest_support(self, current_price: float) -> Tuple[Optional[float], float]:
        """Find nearest support level below current price"""
        valid_supports = [l for l in self.support_levels if l.price < current_price]
        
        if not valid_supports:
            return None, float('inf')
        
        # Find closest
        nearest = max(valid_supports, key=lambda x: x.price)
        distance_pct = (current_price - nearest.price) / current_price * 100
        
        return nearest.price, distance_pct
    
    def _find_nearest_resistance(self, current_price: float) -> Tuple[Optional[float], float]:
        """Find nearest resistance level above current price"""
        valid_resistances = [l for l in self.resistance_levels if l.price > current_price]
        
        if not valid_resistances:
            return None, float('inf')
        
        # Find closest
        nearest = min(valid_resistances, key=lambda x: x.price)
        distance_pct = (nearest.price - current_price) / current_price * 100
        
        return nearest.price, distance_pct
    
    def _calculate_level_density(self, current_price: float) -> float:
        """Calculate density of levels around current price"""
        # Count levels within 2% of current price
        nearby_range = current_price * 0.02
        
        nearby_count = sum(
            1 for l in self.support_levels + self.resistance_levels
            if abs(l.price - current_price) <= nearby_range
        )
        
        # Normalize (0-1)
        return min(nearby_count / 5, 1.0)  # 5 levels = max density
    
    def _determine_price_position(self, current_price: float, 
                                 support_distance: float, 
                                 resistance_distance: float) -> str:
        """Determine price position relative to levels"""
        if support_distance < 0.5:  # Within 0.5% of support
            return 'near_support'
        elif resistance_distance < 0.5:  # Within 0.5% of resistance
            return 'near_resistance'
        elif support_distance < resistance_distance:
            return 'closer_to_support'
        elif resistance_distance < support_distance:
            return 'closer_to_resistance'
        else:
            return 'neutral'
    
    def _calculate_level_strength_score(self) -> float:
        """Calculate overall strength of identified levels"""
        all_levels = self.support_levels + self.resistance_levels
        
        if not all_levels:
            return 0.0
        
        # Average strength weighted by touches
        total_weight = sum(l.strength * l.touches for l in all_levels)
        total_touches = sum(l.touches for l in all_levels)
        
        if total_touches == 0:
            return 0.0
        
        return total_weight / total_touches
    
    def _calculate_breakout_probability(self, current_price: float,
                                       nearest_support: Optional[float],
                                       nearest_resistance: Optional[float],
                                       data: Dict[str, Any]) -> float:
        """Calculate probability of breaking key levels"""
        try:
            # Get current volatility
            volatility = data.get('volatility', 0.01)  # Default 1% if not available
            
            # Calculate distances
            support_distance = abs(current_price - nearest_support) / current_price if nearest_support else 0.1
            resistance_distance = abs(nearest_resistance - current_price) / current_price if nearest_resistance else 0.1
            
            # Probability based on volatility vs distance
            support_break_prob = min(volatility / support_distance, 1.0) if support_distance > 0 else 0
            resistance_break_prob = min(volatility / resistance_distance, 1.0) if resistance_distance > 0 else 0
            
            # Return higher probability (more likely breakout)
            return max(support_break_prob, resistance_break_prob)
            
        except Exception:
            return 0.5
    
    def get_regime_contribution(self, analysis_result: ResistanceAnalysisResult) -> Dict[str, float]:
        """Get regime contribution from support/resistance analysis"""
        regime_indicators = {}
        
        try:
            # Level strength indicates market structure
            if analysis_result.level_strength_score > 0.7:
                regime_indicators['level_structure'] = 1.0  # Strong levels = structured market
            elif analysis_result.level_strength_score < 0.3:
                regime_indicators['level_structure'] = -1.0  # Weak levels = unstructured
            else:
                regime_indicators['level_structure'] = 0.0
            
            # Price position indicates trend pressure
            if analysis_result.price_position == 'near_resistance':
                regime_indicators['level_pressure'] = -0.5  # Bearish pressure
            elif analysis_result.price_position == 'near_support':
                regime_indicators['level_pressure'] = 0.5   # Bullish pressure
            else:
                regime_indicators['level_pressure'] = 0.0
            
            # Breakout probability indicates volatility
            if analysis_result.breakout_probability > 0.6:
                regime_indicators['breakout_potential'] = 1.0  # High volatility expected
            elif analysis_result.breakout_probability < 0.3:
                regime_indicators['breakout_potential'] = -1.0  # Low volatility expected
            else:
                regime_indicators['breakout_potential'] = 0.0
            
            # Level density indicates congestion
            if analysis_result.level_density > 0.6:
                regime_indicators['market_congestion'] = 1.0  # Congested
            else:
                regime_indicators['market_congestion'] = -1.0  # Clear
            
        except Exception as e:
            self.logger.error(f"Error calculating regime contribution: {e}")
        
        return regime_indicators
    
    def get_straddle_adjustments(self, analysis_result: ResistanceAnalysisResult) -> Dict[str, float]:
        """Get straddle position adjustments based on levels"""
        adjustments = {}
        
        try:
            # Near support - reduce put weight, increase call weight
            if analysis_result.price_position == 'near_support':
                adjustments['call_weight_adjustment'] = 0.1
                adjustments['put_weight_adjustment'] = -0.1
                adjustments['position_bias'] = 'bullish'
            
            # Near resistance - reduce call weight, increase put weight
            elif analysis_result.price_position == 'near_resistance':
                adjustments['call_weight_adjustment'] = -0.1
                adjustments['put_weight_adjustment'] = 0.1
                adjustments['position_bias'] = 'bearish'
            
            else:
                adjustments['call_weight_adjustment'] = 0.0
                adjustments['put_weight_adjustment'] = 0.0
                adjustments['position_bias'] = 'neutral'
            
            # High breakout probability - increase straddle size
            if analysis_result.breakout_probability > 0.6:
                adjustments['size_adjustment'] = 0.2  # Increase position by 20%
                adjustments['volatility_play'] = True
            else:
                adjustments['size_adjustment'] = 0.0
                adjustments['volatility_play'] = False
            
            # Level density affects strategy
            if analysis_result.level_density > 0.7:
                adjustments['strategy_suggestion'] = 'IRON_CONDOR'  # Range-bound
            else:
                adjustments['strategy_suggestion'] = 'STRADDLE'  # Breakout potential
            
        except Exception as e:
            self.logger.error(f"Error calculating straddle adjustments: {e}")
        
        return adjustments
    
    def get_analyzer_status(self) -> Dict[str, Any]:
        """Get resistance analyzer status"""
        return {
            'support_levels_count': len(self.support_levels),
            'resistance_levels_count': len(self.resistance_levels),
            'price_history_length': len(self.price_history),
            'strongest_support': self.support_levels[0].price if self.support_levels else None,
            'strongest_resistance': self.resistance_levels[0].price if self.resistance_levels else None,
            'parameters': {
                'min_touches': self.min_touches,
                'level_tolerance': self.level_tolerance,
                'level_decay_factor': self.level_decay_factor,
                'max_levels': self.max_levels
            }
        }