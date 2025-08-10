#!/usr/bin/env python3
"""
OI Models - Data models for Open Interest strategy
Based on archive system and column mapping from input_sheets/oi/
"""

from typing import List, Optional, Union
from datetime import date, time
from dataclasses import dataclass, field
from enum import Enum


class OIMethod(Enum):
    """OI-based strike selection methods"""
    MAXOI_1 = "MAXOI_1"     # 1st highest OI
    MAXOI_2 = "MAXOI_2"     # 2nd highest OI  
    MAXOI_3 = "MAXOI_3"     # 3rd highest OI
    MAXCOI_1 = "MAXCOI_1"   # 1st highest Change in OI
    MAXCOI_2 = "MAXCOI_2"   # 2nd highest Change in OI
    MAXCOI_3 = "MAXCOI_3"   # 3rd highest Change in OI
    ATM = "ATM"             # ATM-based selection
    ITM1 = "ITM1"           # 1 strike ITM
    ITM2 = "ITM2"           # 2 strikes ITM
    OTM1 = "OTM1"           # 1 strike OTM
    OTM2 = "OTM2"           # 2 strikes OTM
    FIXED = "FIXED"         # Fixed strike value


class COIBasedOn(Enum):
    """COI calculation base - matching archive system"""
    YESTERDAY_CLOSE = "YESTERDAY_CLOSE"
    PREVIOUS_TIMESTAMP = "PREVIOUS_TIMESTAMP"


@dataclass
class OILegModel:
    """Model for OI leg parameters"""
    leg_id: str
    instrument: str  # CE/PE/FUT
    transaction: str  # BUY/SELL
    expiry: str      # CW/NW/CM/NM
    lots: int
    
    # OI-specific parameters
    oi_threshold: int = 800000  # Minimum OI required
    strike_method: str = "MAXOI_1"
    strike_value: float = 0
    match_premium: Optional[str] = None
    strike_premium_condition: str = '='
    
    # Risk parameters (consistent with TBS/ORB/TV)
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
    
    def should_execute_for_oi_rank(self, oi_rank: int, instrument_type: str) -> bool:
        """
        Determine if this leg should execute based on OI ranking
        
        Args:
            oi_rank: 1-based ranking (1 = highest OI)
            instrument_type: 'CE' or 'PE'
        """
        if self.instrument != instrument_type:
            return False
        
        # Extract rank from strike method
        if self.strike_method.startswith('MAXOI_'):
            target_rank = int(self.strike_method.split('_')[1])
            return oi_rank == target_rank
        elif self.strike_method.startswith('MAXCOI_'):
            target_rank = int(self.strike_method.split('_')[1])
            return oi_rank == target_rank
        else:
            # For non-OI methods (ATM, ITM, OTM, FIXED), always execute
            return True


@dataclass
class OISettingModel:
    """Model for OI strategy settings"""
    strategy_name: str
    timeframe: int = 3  # Minutes (must be multiple of 3 for OI)
    max_open_positions: int = 1
    underlying: str = 'SPOT'
    index: str = 'NIFTY'
    weekdays: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    dte: int = 0
    
    # Time parameters
    strike_selection_time: time = time(9, 16, 0)
    start_time: time = time(9, 16, 0)
    last_entry_time: time = time(15, 0, 0)
    end_time: time = time(15, 20, 0)
    
    # OI-specific parameters
    oi_method: OIMethod = OIMethod.MAXOI_1
    coi_based_on: COIBasedOn = COIBasedOn.YESTERDAY_CLOSE
    strike_count: int = 5  # Strikes each side of ATM to analyze (noofstrikeeachside from archive)
    
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
    
    # Partial exits (OI-specific feature)
    pnl_cal_time: Optional[time] = None
    lock_percent: Optional[float] = None
    trail_percent: Optional[float] = None
    sq_off_1_time: Optional[time] = None
    sq_off_1_percent: Optional[float] = None
    sq_off_2_time: Optional[time] = None
    sq_off_2_percent: Optional[float] = None
    
    # Legs
    legs: List[OILegModel] = field(default_factory=list)
    
    def is_trading_day(self, weekday: int) -> bool:
        """Check if given weekday is a trading day"""
        return weekday in self.weekdays
    
    def get_oi_legs(self) -> List[OILegModel]:
        """Get legs that use OI-based strike selection"""
        return [leg for leg in self.legs if leg.strike_method.startswith(('MAXOI_', 'MAXCOI_'))]
    
    def get_non_oi_legs(self) -> List[OILegModel]:
        """Get legs that use non-OI strike selection (ATM, ITM, OTM, FIXED)"""
        return [leg for leg in self.legs if not leg.strike_method.startswith(('MAXOI_', 'MAXCOI_'))]


@dataclass
class OIRanking:
    """Model for OI ranking data"""
    trade_date: date
    analysis_time: time
    ce_rankings: List[tuple]  # List of (strike, oi_value) tuples sorted by OI
    pe_rankings: List[tuple]  # List of (strike, oi_value) tuples sorted by OI
    coi_rankings: Optional[List[tuple]] = None  # COI rankings if applicable
    
    def get_strike_for_rank(self, rank: int, instrument_type: str) -> Optional[int]:
        """
        Get strike for given rank and instrument type
        
        Args:
            rank: 1-based ranking (1 = highest OI)
            instrument_type: 'CE' or 'PE'
            
        Returns:
            Strike price or None if rank not available
        """
        rankings = self.ce_rankings if instrument_type == 'CE' else self.pe_rankings
        
        if 1 <= rank <= len(rankings):
            return rankings[rank - 1][0]  # Return strike (first element of tuple)
        
        return None
    
    def get_oi_for_rank(self, rank: int, instrument_type: str) -> Optional[float]:
        """Get OI value for given rank and instrument type"""
        rankings = self.ce_rankings if instrument_type == 'CE' else self.pe_rankings
        
        if 1 <= rank <= len(rankings):
            return rankings[rank - 1][1]  # Return OI value (second element of tuple)
        
        return None


@dataclass
class OISignal:
    """Model for OI-based entry signal"""
    trade_date: date
    signal_time: time
    leg_id: str
    instrument: str  # CE/PE
    strike: int
    oi_rank: int
    oi_value: float
    underlying_price: float
    selection_method: str  # MAXOI_1, MAXCOI_1, etc.
    
    @property
    def signal_key(self) -> str:
        """Unique key for this signal"""
        return f"{self.trade_date}_{self.leg_id}_{self.strike}_{self.signal_time}"


@dataclass  
class ProcessedOISignal:
    """Model for processed OI signal ready for execution"""
    entrydate: date
    entrytime: time
    exitdate: date
    exittime: time
    lots: int
    leg_id: str
    instrument: str
    strike: int
    oi_rank: int
    oi_value: float
    selection_method: str
    underlying_at_entry: float
    
    # Re-entry tracking
    is_reentry: bool = False
    reentry_count: int = 0
    original_signal_id: Optional[str] = None