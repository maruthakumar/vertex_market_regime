#!/usr/bin/env python3
"""
ORB Models - Data models for Opening Range Breakout strategy
"""

from typing import List, Optional, Union
from datetime import date, time
from dataclasses import dataclass, field
from enum import Enum


class ORBBreakoutType(Enum):
    """Breakout types for ORB strategy"""
    LOWBREAKOUT = -1  # Price breaks below range low
    HIGHBREAKOUT = 1  # Price breaks above range high
    NONE = 0          # No breakout yet


class ORBSignalDirection(Enum):
    """Signal direction based on breakout type"""
    BULLISH = 1   # High breakout -> bullish trades
    BEARISH = -1  # Low breakout -> bearish trades
    NEUTRAL = 0   # No signal


@dataclass
class ORBLegModel:
    """Model for ORB leg parameters"""
    leg_id: str
    instrument: str  # CE/PE/FUT
    transaction: str  # BUY/SELL
    expiry: str  # CW/NW/CM/NM
    lots: int
    
    # Strike selection
    strike_method: str
    strike_value: float = 0
    match_premium: Optional[str] = None
    strike_premium_condition: str = '='
    
    # Risk management
    wait_type: Optional[str] = None
    wait_value: float = 0
    sl_type: str = 'percentage'
    sl_value: float = 500  # Default 500% for SELL
    tgt_type: str = 'percentage'
    tgt_value: float = 100
    
    # Trailing SL
    trail_sl_type: Optional[str] = None
    sl_trail_at: Optional[float] = None
    sl_trail_by: Optional[float] = None
    
    # Re-entry
    sl_reentry_type: Optional[str] = None
    sl_reentry_no: int = 0
    tgt_reentry_type: Optional[str] = None
    tgt_reentry_no: int = 0
    
    # Hedge parameters
    open_hedge: bool = False
    hedge_strike_method: Optional[str] = None
    hedge_strike_value: Optional[float] = None
    hedge_strike_premium_condition: Optional[str] = None
    
    def should_execute_on_breakout(self, breakout_type: ORBBreakoutType) -> bool:
        """
        Determine if this leg should execute based on breakout type
        
        For ORB:
        - High breakout: Execute bullish legs (buy calls, sell puts)
        - Low breakout: Execute bearish legs (buy puts, sell calls)
        - Some strategies execute all legs on any breakout (straddle/strangle)
        """
        if breakout_type == ORBBreakoutType.NONE:
            return False
        
        # Check if this is a straddle/strangle strategy (has both CE and PE)
        # In this case, execute all legs on any breakout
        # This is indicated by having PREMIUM-based strike selection
        if self.strike_method == 'PREMIUM':
            return True
        
        if breakout_type == ORBBreakoutType.HIGHBREAKOUT:
            # Bullish breakout - execute bullish positions
            if self.instrument == 'CE' and self.transaction == 'BUY':
                return True
            if self.instrument == 'PE' and self.transaction == 'SELL':
                return True
            if self.instrument == 'FUT' and self.transaction == 'BUY':
                return True
        
        elif breakout_type == ORBBreakoutType.LOWBREAKOUT:
            # Bearish breakout - execute bearish positions
            if self.instrument == 'PE' and self.transaction == 'BUY':
                return True
            if self.instrument == 'CE' and self.transaction == 'SELL':
                return True
            if self.instrument == 'FUT' and self.transaction == 'SELL':
                return True
        
        return False


@dataclass
class ORBSettingModel:
    """Model for ORB general parameters"""
    strategy_name: str
    underlying: str = 'SPOT'
    index: str = 'NIFTY'
    weekdays: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    dte: int = 0
    
    # ORB-specific time parameters
    orb_range_start: time = time(9, 15, 0)
    orb_range_end: time = time(9, 20, 0)
    last_entry_time: time = time(15, 0, 0)
    end_time: time = time(15, 20, 0)
    
    # Risk management
    strategy_profit: Optional[float] = None
    strategy_loss: Optional[float] = None
    strategy_profit_reexecute_no: int = 0
    strategy_loss_reexecute_no: int = 0
    
    # Trailing parameters
    strategy_trailing_type: str = ''
    profit_reaches: Optional[float] = None
    lock_min_profit_at: Optional[float] = None
    increase_in_profit: Optional[float] = None
    trail_min_profit_by: Optional[float] = None
    
    # Tracking parameters
    tgt_tracking_from: str = 'close'
    tgt_register_price_from: str = 'tick'
    sl_tracking_from: str = 'close'
    sl_register_price_from: str = 'tick'
    pnl_calculation_from: str = 'close'
    
    # Boolean flags
    consider_hedge_pnl: bool = False
    on_expiry_day_trade_next: bool = False
    
    # Intervals (in seconds)
    stoploss_checking_interval: int = 60
    target_checking_interval: int = 60
    reentry_checking_interval: int = 60
    
    # Legs
    legs: List[ORBLegModel] = field(default_factory=list)
    
    def is_trading_day(self, weekday: int) -> bool:
        """Check if given weekday is a trading day"""
        return weekday in self.weekdays
    
    def get_entry_legs(self, breakout_type: ORBBreakoutType) -> List[ORBLegModel]:
        """Get legs that should be executed for given breakout type"""
        return [leg for leg in self.legs if leg.should_execute_on_breakout(breakout_type)]


@dataclass
class ORBRange:
    """Model for opening range data"""
    trade_date: date
    range_start: time
    range_end: time
    high: float
    low: float
    range_size: float
    
    @property
    def midpoint(self) -> float:
        """Get midpoint of the range"""
        return (self.high + self.low) / 2
    
    def check_breakout(self, price: float) -> ORBBreakoutType:
        """Check if price breaks the range"""
        if price > self.high:
            return ORBBreakoutType.HIGHBREAKOUT
        elif price < self.low:
            return ORBBreakoutType.LOWBREAKOUT
        else:
            return ORBBreakoutType.NONE


@dataclass
class ORBSignal:
    """Model for ORB breakout signal"""
    trade_date: date
    breakout_time: time
    breakout_type: ORBBreakoutType
    breakout_price: float
    range_high: float
    range_low: float
    signal_direction: ORBSignalDirection
    
    @classmethod
    def from_breakout(cls, trade_date: date, breakout_time: time, 
                      breakout_type: ORBBreakoutType, breakout_price: float,
                      orb_range: ORBRange) -> 'ORBSignal':
        """Create signal from breakout detection"""
        signal_direction = ORBSignalDirection.NEUTRAL
        
        if breakout_type == ORBBreakoutType.HIGHBREAKOUT:
            signal_direction = ORBSignalDirection.BULLISH
        elif breakout_type == ORBBreakoutType.LOWBREAKOUT:
            signal_direction = ORBSignalDirection.BEARISH
        
        return cls(
            trade_date=trade_date,
            breakout_time=breakout_time,
            breakout_type=breakout_type,
            breakout_price=breakout_price,
            range_high=orb_range.high,
            range_low=orb_range.low,
            signal_direction=signal_direction
        )


@dataclass
class ProcessedORBSignal:
    """Model for processed ORB signal ready for execution"""
    entrydate: date
    entrytime: time
    exitdate: date
    exittime: time
    lots: int
    signal_direction: ORBSignalDirection
    breakout_type: ORBBreakoutType
    range_high: float
    range_low: float
    breakout_price: float
    legs_to_execute: List[ORBLegModel]
    
    # Re-entry tracking
    is_reentry: bool = False
    reentry_count: int = 0
    original_signal_id: Optional[str] = None