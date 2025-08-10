"""
Session Weight Manager - Time-Based Analysis Weighting
======================================================

Manages session-based weighting for time-sensitive analysis including
market open, midday, and close patterns with dynamic weight adjustment.

Author: Market Regime Refactoring Team
Date: 2025-07-06
Version: 2.0.0 - Enhanced Session Analysis
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import logging
from datetime import datetime, time
from enum import Enum

logger = logging.getLogger(__name__)

class MarketSession(Enum):
    """Market session classifications"""
    PRE_OPEN = "Pre_Open"           # 9:00-9:15
    OPENING = "Opening"             # 9:15-10:00
    MORNING = "Morning"             # 10:00-11:30
    MIDDAY = "Midday"              # 11:30-14:00
    AFTERNOON = "Afternoon"         # 14:00-15:00
    CLOSING = "Closing"             # 15:00-15:30
    POST_CLOSE = "Post_Close"       # 15:30-16:00

class SessionWeightManager:
    """
    Advanced session-based weight management
    
    Features:
    - Dynamic session classification
    - Time-sensitive weight calculation
    - Historical session pattern analysis
    - Volatility-based weight adjustment
    - Multi-timeframe session coordination
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Session Weight Manager"""
        self.config = config or {}
        
        # Session time boundaries (24-hour format)
        self.session_boundaries = self.config.get('session_boundaries', {
            MarketSession.PRE_OPEN: (time(9, 0), time(9, 15)),
            MarketSession.OPENING: (time(9, 15), time(10, 0)),
            MarketSession.MORNING: (time(10, 0), time(11, 30)),
            MarketSession.MIDDAY: (time(11, 30), time(14, 0)),
            MarketSession.AFTERNOON: (time(14, 0), time(15, 0)),
            MarketSession.CLOSING: (time(15, 0), time(15, 30)),
            MarketSession.POST_CLOSE: (time(15, 30), time(16, 0))
        })
        
        # Base session weights
        self.base_session_weights = self.config.get('base_session_weights', {
            MarketSession.PRE_OPEN: 0.3,        # Low activity
            MarketSession.OPENING: 1.5,         # High volatility
            MarketSession.MORNING: 1.2,         # Active trading
            MarketSession.MIDDAY: 1.0,          # Standard activity
            MarketSession.AFTERNOON: 1.1,       # Moderate activity
            MarketSession.CLOSING: 1.8,         # High activity/volatility
            MarketSession.POST_CLOSE: 0.4       # Low activity
        })
        
        # Adaptive weighting parameters
        self.enable_adaptive_weighting = self.config.get('enable_adaptive_weighting', True)
        self.volatility_adjustment_factor = self.config.get('volatility_adjustment_factor', 0.5)
        self.volume_adjustment_factor = self.config.get('volume_adjustment_factor', 0.3)
        
        # Historical pattern analysis
        self.enable_historical_patterns = self.config.get('enable_historical_patterns', True)
        self.pattern_lookback_days = self.config.get('pattern_lookback_days', 20)
        
        # Weight smoothing
        self.enable_weight_smoothing = self.config.get('enable_weight_smoothing', True)
        self.smoothing_factor = self.config.get('smoothing_factor', 0.7)
        
        # Session history tracking
        self.session_history = []
        self.weight_history = {}
        
        logger.info("SessionWeightManager initialized with dynamic session analysis")
    
    def get_session_weight(self, 
                          current_time: datetime,
                          market_conditions: Optional[Dict[str, Any]] = None) -> float:
        """
        Get session weight for current time
        
        Args:
            current_time: Current timestamp
            market_conditions: Market conditions for adaptive weighting
            
        Returns:
            float: Session weight for current time
        """
        try:
            # Classify current session
            current_session = self._classify_session(current_time.time())
            
            # Get base weight
            base_weight = self.base_session_weights.get(current_session, 1.0)
            
            # Apply adaptive adjustments if enabled
            if self.enable_adaptive_weighting and market_conditions:
                adjusted_weight = self._apply_adaptive_adjustments(
                    base_weight, current_session, market_conditions
                )
            else:
                adjusted_weight = base_weight
            
            # Apply historical pattern adjustments if enabled
            if self.enable_historical_patterns:
                pattern_adjusted_weight = self._apply_historical_pattern_adjustment(
                    adjusted_weight, current_session, current_time
                )
            else:
                pattern_adjusted_weight = adjusted_weight
            
            # Apply weight smoothing if enabled
            if self.enable_weight_smoothing:
                final_weight = self._apply_weight_smoothing(
                    pattern_adjusted_weight, current_session
                )
            else:
                final_weight = pattern_adjusted_weight
            
            # Record weight calculation
            self._record_weight_calculation(current_session, final_weight, current_time)
            
            return final_weight
            
        except Exception as e:
            logger.error(f"Error getting session weight: {e}")
            return 1.0
    
    def _classify_session(self, current_time: time) -> MarketSession:
        """Classify current time into market session"""
        try:
            for session, (start_time, end_time) in self.session_boundaries.items():
                if start_time <= current_time < end_time:
                    return session
            
            # Default fallback
            return MarketSession.MIDDAY
            
        except Exception as e:
            logger.error(f"Error classifying session: {e}")
            return MarketSession.MIDDAY
    
    def _apply_adaptive_adjustments(self, 
                                  base_weight: float,
                                  session: MarketSession,
                                  market_conditions: Dict[str, Any]) -> float:
        """Apply adaptive adjustments based on market conditions"""
        try:
            adjusted_weight = base_weight
            
            # Volatility adjustment
            volatility = market_conditions.get('volatility', 0.2)
            if volatility > 0.3:  # High volatility
                volatility_multiplier = 1.0 + (volatility - 0.3) * self.volatility_adjustment_factor
            elif volatility < 0.1:  # Low volatility
                volatility_multiplier = 1.0 - (0.1 - volatility) * self.volatility_adjustment_factor
            else:
                volatility_multiplier = 1.0
            
            adjusted_weight *= volatility_multiplier
            
            # Volume adjustment
            volume_ratio = market_conditions.get('volume_ratio', 1.0)  # Relative to average
            if volume_ratio > 1.5:  # High volume
                volume_multiplier = 1.0 + (volume_ratio - 1.5) * self.volume_adjustment_factor
            elif volume_ratio < 0.5:  # Low volume
                volume_multiplier = 1.0 - (0.5 - volume_ratio) * self.volume_adjustment_factor
            else:
                volume_multiplier = 1.0
            
            adjusted_weight *= volume_multiplier
            
            # Session-specific adjustments
            adjusted_weight = self._apply_session_specific_adjustments(
                adjusted_weight, session, market_conditions
            )
            
            return max(0.1, min(adjusted_weight, 3.0))  # Clamp to reasonable range
            
        except Exception as e:
            logger.error(f"Error applying adaptive adjustments: {e}")
            return base_weight
    
    def _apply_session_specific_adjustments(self, 
                                          weight: float,
                                          session: MarketSession,
                                          market_conditions: Dict[str, Any]) -> float:
        """Apply session-specific adjustments"""
        try:
            # Opening session adjustments
            if session == MarketSession.OPENING:
                # Higher weight for gap analysis
                gap_factor = market_conditions.get('gap_factor', 1.0)
                weight *= (1.0 + gap_factor * 0.2)
            
            # Closing session adjustments
            elif session == MarketSession.CLOSING:
                # Higher weight for expiry effects
                dte = market_conditions.get('dte', 30)
                if dte <= 1:  # Expiry day
                    weight *= 1.4
                elif dte <= 7:  # Expiry week
                    weight *= 1.2
            
            # Midday session adjustments
            elif session == MarketSession.MIDDAY:
                # Adjust for lunch hour effects
                current_hour = market_conditions.get('hour', 12)
                if 12 <= current_hour <= 13:  # Lunch hour
                    weight *= 0.8
            
            return weight
            
        except Exception as e:
            logger.error(f"Error applying session-specific adjustments: {e}")
            return weight
    
    def _apply_historical_pattern_adjustment(self, 
                                           weight: float,
                                           session: MarketSession,
                                           current_time: datetime) -> float:
        """Apply historical pattern-based adjustments"""
        try:
            # Get historical weights for this session
            historical_weights = self._get_historical_session_weights(session)
            
            if not historical_weights:
                return weight
            
            # Calculate historical average for this session
            avg_historical_weight = np.mean(historical_weights)
            
            # Adjust current weight based on historical pattern
            if len(historical_weights) >= 5:
                pattern_adjustment = 0.7 * weight + 0.3 * avg_historical_weight
            else:
                pattern_adjustment = weight
            
            # Day-of-week adjustment
            weekday = current_time.weekday()  # 0=Monday, 6=Sunday
            if weekday == 0:  # Monday
                pattern_adjustment *= 1.1  # Higher activity on Mondays
            elif weekday == 4:  # Friday
                pattern_adjustment *= 1.05  # Slightly higher on Fridays
            
            return pattern_adjustment
            
        except Exception as e:
            logger.error(f"Error applying historical pattern adjustment: {e}")
            return weight
    
    def _apply_weight_smoothing(self, weight: float, session: MarketSession) -> float:
        """Apply weight smoothing to reduce volatility"""
        try:
            # Get recent weight for this session
            recent_weights = self.weight_history.get(session, [])
            
            if not recent_weights:
                return weight
            
            # Exponential smoothing
            last_weight = recent_weights[-1]
            smoothed_weight = (self.smoothing_factor * weight + 
                             (1 - self.smoothing_factor) * last_weight)
            
            return smoothed_weight
            
        except Exception as e:
            logger.error(f"Error applying weight smoothing: {e}")
            return weight
    
    def _get_historical_session_weights(self, session: MarketSession) -> List[float]:
        """Get historical weights for a specific session"""
        try:
            session_weights = self.weight_history.get(session, [])
            
            # Return recent weights within lookback period
            if len(session_weights) > self.pattern_lookback_days:
                return session_weights[-self.pattern_lookback_days:]
            
            return session_weights
            
        except Exception as e:
            logger.error(f"Error getting historical session weights: {e}")
            return []
    
    def _record_weight_calculation(self, 
                                 session: MarketSession,
                                 weight: float,
                                 timestamp: datetime):
        """Record weight calculation for historical analysis"""
        try:
            # Record in session history
            record = {
                'timestamp': timestamp,
                'session': session,
                'weight': weight,
                'hour': timestamp.hour,
                'minute': timestamp.minute,
                'weekday': timestamp.weekday()
            }
            
            self.session_history.append(record)
            
            # Record in weight history by session
            if session not in self.weight_history:
                self.weight_history[session] = []
            
            self.weight_history[session].append(weight)
            
            # Trim histories
            if len(self.session_history) > 1000:
                self.session_history = self.session_history[-1000:]
            
            for sess in self.weight_history:
                if len(self.weight_history[sess]) > 100:
                    self.weight_history[sess] = self.weight_history[sess][-100:]
                    
        except Exception as e:
            logger.error(f"Error recording weight calculation: {e}")
    
    def get_session_analysis(self, current_time: datetime) -> Dict[str, Any]:
        """Get comprehensive session analysis"""
        try:
            current_session = self._classify_session(current_time.time())
            
            analysis = {
                'current_session': current_session.value,
                'session_boundaries': {
                    session.value: {
                        'start': start_time.strftime('%H:%M'),
                        'end': end_time.strftime('%H:%M')
                    }
                    for session, (start_time, end_time) in self.session_boundaries.items()
                },
                'base_weights': {
                    session.value: weight 
                    for session, weight in self.base_session_weights.items()
                },
                'session_statistics': self._calculate_session_statistics()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error getting session analysis: {e}")
            return {}
    
    def _calculate_session_statistics(self) -> Dict[str, Any]:
        """Calculate session-based statistics"""
        try:
            if not self.session_history:
                return {}
            
            statistics = {}
            
            for session in MarketSession:
                session_records = [
                    record for record in self.session_history 
                    if record['session'] == session
                ]
                
                if session_records:
                    weights = [record['weight'] for record in session_records]
                    statistics[session.value] = {
                        'count': len(session_records),
                        'avg_weight': np.mean(weights),
                        'std_weight': np.std(weights),
                        'min_weight': np.min(weights),
                        'max_weight': np.max(weights)
                    }
            
            return statistics
            
        except Exception as e:
            logger.error(f"Error calculating session statistics: {e}")
            return {}
    
    def get_optimal_session_timing(self) -> Dict[str, Any]:
        """Get optimal timing analysis for different activities"""
        try:
            if not self.session_history:
                return {'status': 'insufficient_data'}
            
            # Calculate average weights by session
            session_weights = {}
            for session in MarketSession:
                session_records = [
                    record for record in self.session_history 
                    if record['session'] == session
                ]
                
                if session_records:
                    avg_weight = np.mean([record['weight'] for record in session_records])
                    session_weights[session.value] = avg_weight
            
            # Identify optimal sessions
            if session_weights:
                highest_weight_session = max(session_weights.keys(), key=lambda k: session_weights[k])
                lowest_weight_session = min(session_weights.keys(), key=lambda k: session_weights[k])
                
                recommendations = {
                    'highest_activity_session': highest_weight_session,
                    'lowest_activity_session': lowest_weight_session,
                    'session_weights': session_weights,
                    'optimal_for_trading': [s for s, w in session_weights.items() if w > 1.2],
                    'avoid_for_trading': [s for s, w in session_weights.items() if w < 0.7]
                }
                
                return recommendations
            
            return {'status': 'no_data'}
            
        except Exception as e:
            logger.error(f"Error getting optimal session timing: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def update_session_boundaries(self, new_boundaries: Dict[MarketSession, Tuple[time, time]]):
        """Update session time boundaries"""
        try:
            for session, (start_time, end_time) in new_boundaries.items():
                if session in self.session_boundaries:
                    self.session_boundaries[session] = (start_time, end_time)
                    logger.info(f"Updated {session.value} boundaries: {start_time}-{end_time}")
                    
        except Exception as e:
            logger.error(f"Error updating session boundaries: {e}")
    
    def update_session_weights(self, new_weights: Dict[MarketSession, float]):
        """Update base session weights"""
        try:
            for session, weight in new_weights.items():
                if session in self.base_session_weights:
                    old_weight = self.base_session_weights[session]
                    self.base_session_weights[session] = weight
                    logger.info(f"Updated {session.value} weight: {old_weight} -> {weight}")
                    
        except Exception as e:
            logger.error(f"Error updating session weights: {e}")
    
    def get_session_weight_summary(self) -> Dict[str, Any]:
        """Get summary of session weight management"""
        try:
            summary = {
                'total_weight_calculations': len(self.session_history),
                'configuration': {
                    'adaptive_weighting': self.enable_adaptive_weighting,
                    'historical_patterns': self.enable_historical_patterns,
                    'weight_smoothing': self.enable_weight_smoothing,
                    'pattern_lookback_days': self.pattern_lookback_days
                },
                'session_coverage': {
                    session.value: len(self.weight_history.get(session, []))
                    for session in MarketSession
                }
            }
            
            if self.session_history:
                recent_analysis = self._calculate_session_statistics()
                summary['recent_session_statistics'] = recent_analysis
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating session weight summary: {e}")
            return {'status': 'error', 'error': str(e)}