"""
Adaptive Shift Manager for Enhanced OI System

This module implements intelligent strike shifting with:
- Configurable shift delays
- Historical performance-based threshold adjustments
- DTE-based sensitivity
- Market regime awareness
- Emergency override conditions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, field
from collections import deque
import json

logger = logging.getLogger(__name__)

@dataclass
class ShiftConfig:
    """Configuration for adaptive shifting."""
    # Base shift parameters
    shift_delay_minutes: int = 3
    base_oi_threshold: float = 0.20
    base_weight_threshold: float = 0.15
    
    # Emergency override thresholds
    emergency_oi_change: float = 0.50
    emergency_vega_change: float = 0.30
    emergency_delta_change: float = 0.30
    
    # DTE-based adjustments
    dte_high_threshold: int = 7  # DTE > 7 days
    dte_low_threshold: int = 2   # DTE <= 2 days
    
    # Performance tracking
    performance_window: int = 10  # Number of trades to track
    performance_threshold_good: float = 0.70
    performance_threshold_poor: float = 0.40
    
    # Market regime sensitivity
    enable_regime_sensitivity: bool = True
    trending_adjustment: float = 0.75  # Reduce thresholds by 25%
    sideways_adjustment: float = 1.30  # Increase thresholds by 30%
    high_vol_adjustment: float = 0.80  # Reduce emergency thresholds by 20%

@dataclass
class ShiftSignal:
    """Represents a potential shift signal."""
    timestamp: datetime
    current_strike: float
    new_strike: float
    current_oi: float
    new_oi: float
    oi_improvement: float
    weight_improvement: float
    shift_reason: str
    emergency_override: bool = False
    dte: int = 0
    market_regime: str = "normal"

@dataclass
class ShiftExecution:
    """Records an executed shift."""
    timestamp: datetime
    from_strike: float
    to_strike: float
    oi_improvement: float
    weight_improvement: float
    shift_reason: str
    dte: int
    market_regime: str
    performance_before: float = 0.0
    performance_after: float = 0.0

class HistoricalPerformanceTracker:
    """Tracks historical performance for threshold adjustments."""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.trade_history = deque(maxlen=window_size)
        self.shift_history = deque(maxlen=window_size * 2)
        
    def add_trade_result(self, trade_pnl: float, shift_count: int, dte: int, regime: str):
        """Add a trade result for performance tracking."""
        trade_record = {
            'timestamp': datetime.now(),
            'pnl': trade_pnl,
            'shift_count': shift_count,
            'dte': dte,
            'regime': regime,
            'performance_score': self._calculate_performance_score(trade_pnl, shift_count)
        }
        self.trade_history.append(trade_record)
        
    def add_shift_result(self, shift_execution: ShiftExecution):
        """Add a shift execution result."""
        self.shift_history.append(shift_execution)
        
    def _calculate_performance_score(self, pnl: float, shift_count: int) -> float:
        """Calculate performance score considering PnL and shift efficiency."""
        # Base score from PnL (normalized)
        base_score = max(0, min(1, (pnl + 1000) / 2000))  # Normalize around ±1000
        
        # Penalty for excessive shifts
        shift_penalty = max(0, (shift_count - 2) * 0.1)  # Penalty after 2 shifts
        
        return max(0, base_score - shift_penalty)
    
    def get_recent_performance(self) -> float:
        """Get recent performance score."""
        if not self.trade_history:
            return 0.5  # Neutral
            
        recent_scores = [trade['performance_score'] for trade in self.trade_history]
        return np.mean(recent_scores)
    
    def get_dte_performance(self, dte: int) -> float:
        """Get performance for specific DTE range."""
        if not self.trade_history:
            return 0.5
            
        dte_trades = [trade for trade in self.trade_history 
                     if abs(trade['dte'] - dte) <= 1]
        
        if not dte_trades:
            return 0.5
            
        return np.mean([trade['performance_score'] for trade in dte_trades])
    
    def get_regime_performance(self, regime: str) -> float:
        """Get performance for specific market regime."""
        if not self.trade_history:
            return 0.5
            
        regime_trades = [trade for trade in self.trade_history 
                        if trade['regime'] == regime]
        
        if not regime_trades:
            return 0.5
            
        return np.mean([trade['performance_score'] for trade in regime_trades])

class AdaptiveShiftManager:
    """Main class for adaptive strike shifting with historical learning."""
    
    def __init__(self, config: ShiftConfig = None):
        """Initialize the adaptive shift manager."""
        self.config = config or ShiftConfig()
        self.performance_tracker = HistoricalPerformanceTracker(self.config.performance_window)
        
        # Current state
        self.current_strikes = {}  # {leg_id: strike}
        self.pending_shifts = {}   # {leg_id: ShiftSignal}
        self.shift_timers = {}     # {leg_id: datetime}
        self.executed_shifts = []
        
        # Performance tracking
        self.current_trade_shifts = 0
        self.current_trade_start = None
        
        logger.info("Adaptive Shift Manager initialized")
    
    def evaluate_shift_signal(self, leg_id: str, current_strike: float, 
                            current_oi: float, new_strike: float, new_oi: float,
                            current_weights: Dict[str, float], new_weights: Dict[str, float],
                            dte: int, market_regime: str = "normal") -> Optional[ShiftSignal]:
        """Evaluate if a shift signal should be generated."""
        
        # Calculate improvements
        oi_improvement = (new_oi - current_oi) / current_oi if current_oi > 0 else 0
        weight_improvement = self._calculate_weight_improvement(current_weights, new_weights)
        
        # Get adaptive thresholds
        oi_threshold, weight_threshold, emergency_thresholds = self._get_adaptive_thresholds(
            dte, market_regime
        )
        
        # Check emergency conditions
        emergency_override = self._check_emergency_conditions(
            oi_improvement, current_weights, new_weights, emergency_thresholds
        )
        
        # Determine if shift is warranted
        shift_warranted = (
            emergency_override or
            (oi_improvement >= oi_threshold) or
            (weight_improvement >= weight_threshold)
        )
        
        if shift_warranted:
            shift_reason = self._determine_shift_reason(
                oi_improvement, weight_improvement, emergency_override, 
                oi_threshold, weight_threshold
            )
            
            return ShiftSignal(
                timestamp=datetime.now(),
                current_strike=current_strike,
                new_strike=new_strike,
                current_oi=current_oi,
                new_oi=new_oi,
                oi_improvement=oi_improvement,
                weight_improvement=weight_improvement,
                shift_reason=shift_reason,
                emergency_override=emergency_override,
                dte=dte,
                market_regime=market_regime
            )
        
        return None
    
    def _calculate_weight_improvement(self, current_weights: Dict[str, float], 
                                    new_weights: Dict[str, float]) -> float:
        """Calculate overall weight improvement."""
        if not current_weights or not new_weights:
            return 0.0
        
        # Calculate weighted improvement across all factors
        improvements = []
        for factor in current_weights:
            if factor in new_weights:
                current_val = current_weights[factor]
                new_val = new_weights[factor]
                if current_val > 0:
                    improvement = (new_val - current_val) / current_val
                    improvements.append(improvement)
        
        return np.mean(improvements) if improvements else 0.0
    
    def _get_adaptive_thresholds(self, dte: int, market_regime: str) -> Tuple[float, float, Dict[str, float]]:
        """Get adaptive thresholds based on DTE, performance, and market regime."""
        
        # Start with base thresholds
        oi_threshold = self.config.base_oi_threshold
        weight_threshold = self.config.base_weight_threshold
        emergency_thresholds = {
            'oi_change': self.config.emergency_oi_change,
            'vega_change': self.config.emergency_vega_change,
            'delta_change': self.config.emergency_delta_change
        }
        
        # DTE-based adjustments
        if dte > self.config.dte_high_threshold:
            # More conservative for longer DTE
            oi_threshold *= 1.25
            weight_threshold *= 1.20
        elif dte <= self.config.dte_low_threshold:
            # More aggressive for shorter DTE
            oi_threshold *= 0.75
            weight_threshold *= 0.80
        
        # Historical performance adjustments
        recent_performance = self.performance_tracker.get_recent_performance()
        dte_performance = self.performance_tracker.get_dte_performance(dte)
        regime_performance = self.performance_tracker.get_regime_performance(market_regime)
        
        # Average performance across dimensions
        avg_performance = np.mean([recent_performance, dte_performance, regime_performance])
        
        if avg_performance > self.config.performance_threshold_good:
            # Good performance - be more aggressive
            oi_threshold *= 0.80
            weight_threshold *= 0.80
        elif avg_performance < self.config.performance_threshold_poor:
            # Poor performance - be more conservative
            oi_threshold *= 1.30
            weight_threshold *= 1.30
        
        # Market regime adjustments
        if self.config.enable_regime_sensitivity:
            if market_regime == "trending":
                oi_threshold *= self.config.trending_adjustment
                weight_threshold *= self.config.trending_adjustment
            elif market_regime == "sideways":
                oi_threshold *= self.config.sideways_adjustment
                weight_threshold *= self.config.sideways_adjustment
            elif market_regime == "high_volatility":
                for key in emergency_thresholds:
                    emergency_thresholds[key] *= self.config.high_vol_adjustment
        
        return oi_threshold, weight_threshold, emergency_thresholds
    
    def _check_emergency_conditions(self, oi_improvement: float, 
                                  current_weights: Dict[str, float],
                                  new_weights: Dict[str, float],
                                  emergency_thresholds: Dict[str, float]) -> bool:
        """Check if emergency override conditions are met."""
        
        # Check OI change
        if abs(oi_improvement) >= emergency_thresholds['oi_change']:
            return True
        
        # Check Vega change
        vega_change = self._calculate_greek_change(current_weights, new_weights, 'vega')
        if abs(vega_change) >= emergency_thresholds['vega_change']:
            return True
        
        # Check Delta change
        delta_change = self._calculate_greek_change(current_weights, new_weights, 'delta')
        if abs(delta_change) >= emergency_thresholds['delta_change']:
            return True
        
        return False
    
    def _calculate_greek_change(self, current_weights: Dict[str, float],
                              new_weights: Dict[str, float], greek_type: str) -> float:
        """Calculate change in Greek factor weights."""
        greek_factors = [f for f in current_weights.keys() if greek_type in f.lower()]
        
        if not greek_factors:
            return 0.0
        
        current_greek = sum(current_weights.get(f, 0) for f in greek_factors)
        new_greek = sum(new_weights.get(f, 0) for f in greek_factors)
        
        if current_greek > 0:
            return (new_greek - current_greek) / current_greek
        
        return 0.0

    def _determine_shift_reason(self, oi_improvement: float, weight_improvement: float,
                              emergency_override: bool, oi_threshold: float,
                              weight_threshold: float) -> str:
        """Determine the reason for the shift."""
        if emergency_override:
            return "EMERGENCY_OVERRIDE"
        elif oi_improvement >= oi_threshold and weight_improvement >= weight_threshold:
            return "OI_AND_WEIGHT_IMPROVEMENT"
        elif oi_improvement >= oi_threshold:
            return "OI_IMPROVEMENT"
        elif weight_improvement >= weight_threshold:
            return "WEIGHT_IMPROVEMENT"
        else:
            return "UNKNOWN"

    def should_execute_shift(self, leg_id: str, shift_signal: ShiftSignal) -> bool:
        """Determine if a shift should be executed based on delay logic."""

        # Emergency overrides execute immediately
        if shift_signal.emergency_override:
            return True

        # Check if we have a pending shift for this leg
        if leg_id in self.pending_shifts:
            # Check if delay period has passed
            if leg_id in self.shift_timers:
                time_elapsed = (datetime.now() - self.shift_timers[leg_id]).total_seconds() / 60
                if time_elapsed >= self.config.shift_delay_minutes:
                    return True
            else:
                # Start the timer
                self.shift_timers[leg_id] = datetime.now()
                return False
        else:
            # New shift signal - start delay timer
            self.pending_shifts[leg_id] = shift_signal
            self.shift_timers[leg_id] = datetime.now()
            return False

        return False

    def execute_shift(self, leg_id: str, shift_signal: ShiftSignal) -> ShiftExecution:
        """Execute the shift and record it."""

        shift_execution = ShiftExecution(
            timestamp=datetime.now(),
            from_strike=shift_signal.current_strike,
            to_strike=shift_signal.new_strike,
            oi_improvement=shift_signal.oi_improvement,
            weight_improvement=shift_signal.weight_improvement,
            shift_reason=shift_signal.shift_reason,
            dte=shift_signal.dte,
            market_regime=shift_signal.market_regime
        )

        # Update current strikes
        self.current_strikes[leg_id] = shift_signal.new_strike

        # Clear pending shift and timer
        if leg_id in self.pending_shifts:
            del self.pending_shifts[leg_id]
        if leg_id in self.shift_timers:
            del self.shift_timers[leg_id]

        # Record the shift
        self.executed_shifts.append(shift_execution)
        self.current_trade_shifts += 1

        # Add to performance tracker
        self.performance_tracker.add_shift_result(shift_execution)

        logger.info(f"Shift executed for {leg_id}: {shift_execution.from_strike} → {shift_execution.to_strike} "
                   f"(Reason: {shift_execution.shift_reason})")

        return shift_execution

    def start_new_trade(self):
        """Start tracking a new trade."""
        self.current_trade_start = datetime.now()
        self.current_trade_shifts = 0

    def end_trade(self, trade_pnl: float, dte: int, market_regime: str):
        """End the current trade and record performance."""
        if self.current_trade_start:
            self.performance_tracker.add_trade_result(
                trade_pnl, self.current_trade_shifts, dte, market_regime
            )

            logger.info(f"Trade ended: PnL={trade_pnl:.2f}, Shifts={self.current_trade_shifts}, "
                       f"DTE={dte}, Regime={market_regime}")

        self.current_trade_start = None
        self.current_trade_shifts = 0

    def get_shift_statistics(self) -> Dict[str, Any]:
        """Get comprehensive shift statistics."""
        if not self.executed_shifts:
            return {"total_shifts": 0}

        shifts_df = pd.DataFrame([
            {
                'timestamp': shift.timestamp,
                'oi_improvement': shift.oi_improvement,
                'weight_improvement': shift.weight_improvement,
                'shift_reason': shift.shift_reason,
                'dte': shift.dte,
                'market_regime': shift.market_regime
            }
            for shift in self.executed_shifts
        ])

        stats = {
            'total_shifts': len(self.executed_shifts),
            'avg_oi_improvement': shifts_df['oi_improvement'].mean(),
            'avg_weight_improvement': shifts_df['weight_improvement'].mean(),
            'shift_reasons': shifts_df['shift_reason'].value_counts().to_dict(),
            'shifts_by_dte': shifts_df.groupby('dte').size().to_dict(),
            'shifts_by_regime': shifts_df.groupby('market_regime').size().to_dict(),
            'recent_performance': self.performance_tracker.get_recent_performance()
        }

        return stats

    def get_current_thresholds(self, dte: int, market_regime: str) -> Dict[str, float]:
        """Get current adaptive thresholds for monitoring."""
        oi_threshold, weight_threshold, emergency_thresholds = self._get_adaptive_thresholds(
            dte, market_regime
        )

        return {
            'oi_threshold': oi_threshold,
            'weight_threshold': weight_threshold,
            'emergency_oi_change': emergency_thresholds['oi_change'],
            'emergency_vega_change': emergency_thresholds['vega_change'],
            'emergency_delta_change': emergency_thresholds['delta_change']
        }

    def reset(self):
        """Reset the shift manager state."""
        self.current_strikes.clear()
        self.pending_shifts.clear()
        self.shift_timers.clear()
        self.executed_shifts.clear()
        self.current_trade_shifts = 0
        self.current_trade_start = None

        logger.info("Adaptive Shift Manager reset")
