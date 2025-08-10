#!/usr/bin/env python3
"""
ORB Signal Generator - Generates entry/exit signals based on opening range breakouts
"""

from datetime import date, time, datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class BreakoutType(Enum):
    """Breakout direction types"""
    HIGHBREAKOUT = "HIGHBREAKOUT"  # Price breaks above range high
    LOWBREAKOUT = "LOWBREAKOUT"    # Price breaks below range low
    NONE = "NONE"                  # No breakout


class SignalGenerator:
    """Generates trading signals based on ORB breakouts"""
    
    def __init__(self):
        self._breakout_cache = {}  # Cache detected breakouts
    
    def detect_first_breakout(
        self,
        db_connection: Any,
        index: str,
        trade_date: date,
        range_high: float,
        range_low: float,
        range_end_time: time,
        last_entry_time: time,
        underlying_type: str = 'SPOT'
    ) -> Optional[Dict[str, Any]]:
        """
        Detect the first breakout of the opening range
        
        Args:
            db_connection: Database connection
            index: Index name
            trade_date: Trading date
            range_high: High of the opening range
            range_low: Low of the opening range
            range_end_time: End time of opening range
            last_entry_time: Last time to enter a trade
            underlying_type: SPOT or FUT
            
        Returns:
            Dictionary with breakout details or None
        """
        
        # Check cache
        cache_key = f"{index}_{trade_date}_{range_high}_{range_low}"
        if cache_key in self._breakout_cache:
            return self._breakout_cache[cache_key]
        
        table_name = self._get_table_name(index)
        price_column = self._get_price_column(underlying_type)
        
        sql = f"""
        WITH breakout_candidates AS (
            SELECT
                trade_time,
                {price_column} AS underlying_price,
                atm_strike,
                CASE
                    WHEN {price_column} > {range_high} THEN 'HIGHBREAKOUT'
                    WHEN {price_column} < {range_low} THEN 'LOWBREAKOUT'
                    ELSE 'NONE'
                END AS breakout_type,
                CASE
                    WHEN {price_column} > {range_high} THEN 
                        ({price_column} - {range_high}) / NULLIF({range_high} - {range_low}, 0) * 100
                    WHEN {price_column} < {range_low} THEN 
                        ({range_low} - {price_column}) / NULLIF({range_high} - {range_low}, 0) * 100
                    ELSE 0
                END AS breakout_strength_pct,
                ROW_NUMBER() OVER (
                    PARTITION BY CASE
                        WHEN {price_column} > {range_high} THEN 'HIGHBREAKOUT'
                        WHEN {price_column} < {range_low} THEN 'LOWBREAKOUT'
                        ELSE 'NONE'
                    END
                    ORDER BY trade_time
                ) AS breakout_rank
            FROM {table_name}
            WHERE trade_date = DATE '{trade_date}'
                AND trade_time > TIME '{range_end_time}'
                AND trade_time <= TIME '{last_entry_time}'
                AND ({price_column} > {range_high} OR {price_column} < {range_low})
        )
        SELECT *
        FROM breakout_candidates
        WHERE breakout_rank = 1 AND breakout_type != 'NONE'
        ORDER BY trade_time
        LIMIT 1
        """
        
        try:
            result = db_connection.execute(sql).fetchone()
            
            if result:
                breakout_data = {
                    'breakout_time': result[0],
                    'breakout_price': float(result[1]),
                    'atm_strike': int(result[2]) if result[2] else None,
                    'breakout_type': BreakoutType[result[3]],
                    'breakout_strength_pct': float(result[4]),
                    'range_high': range_high,
                    'range_low': range_low
                }
                
                # Cache the result
                self._breakout_cache[cache_key] = breakout_data
                
                logger.info(f"Detected {breakout_data['breakout_type'].value} at {breakout_data['breakout_time']} "
                          f"with price {breakout_data['breakout_price']} "
                          f"(strength: {breakout_data['breakout_strength_pct']:.2f}%)")
                
                return breakout_data
            else:
                logger.info(f"No breakout detected for {index} on {trade_date}")
                return None
                
        except Exception as e:
            logger.error(f"Error detecting breakout: {e}")
            return None
    
    def generate_entry_signal(
        self,
        breakout_data: Dict[str, Any],
        strategy_params: Dict[str, Any],
        leg_params: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate entry signals based on breakout and leg parameters
        
        Args:
            breakout_data: Breakout information
            strategy_params: Strategy-level parameters
            leg_params: List of leg configurations
            
        Returns:
            List of entry signals
        """
        if not breakout_data:
            return []
        
        signals = []
        breakout_type = breakout_data['breakout_type']
        
        for leg in leg_params:
            # Determine if this leg should be traded based on breakout direction
            if self._should_trade_leg(leg, breakout_type):
                signal = {
                    'leg_id': leg['leg_id'],
                    'signal_type': 'ENTRY',
                    'signal_time': breakout_data['breakout_time'],
                    'underlying_price': breakout_data['breakout_price'],
                    'atm_strike': breakout_data['atm_strike'],
                    'instrument': leg['instrument'],
                    'transaction': leg['transaction'],
                    'strike_method': leg['strike_method'],
                    'strike_value': leg['strike_value'],
                    'expiry': leg['expiry'],
                    'lots': leg['lots'],
                    'breakout_type': breakout_type.value,
                    'breakout_strength': breakout_data['breakout_strength_pct'],
                    
                    # Risk parameters
                    'sl_type': leg['sl_type'],
                    'sl_value': leg['sl_value'],
                    'tgt_type': leg['tgt_type'],
                    'tgt_value': leg['tgt_value'],
                    
                    # Range-based levels
                    'range_high': breakout_data['range_high'],
                    'range_low': breakout_data['range_low'],
                    
                    # Default SL/TP based on range
                    'default_sl': self._calculate_default_sl(
                        breakout_type, 
                        breakout_data['range_high'],
                        breakout_data['range_low']
                    ),
                    'default_target': self._calculate_default_target(
                        breakout_type,
                        breakout_data['breakout_price'],
                        breakout_data['range_high'],
                        breakout_data['range_low']
                    )
                }
                
                signals.append(signal)
        
        return signals
    
    def generate_exit_signal(
        self,
        current_time: time,
        end_time: time,
        active_positions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate exit signals based on time or other conditions
        
        Args:
            current_time: Current market time
            end_time: Strategy end time
            active_positions: List of currently active positions
            
        Returns:
            List of exit signals
        """
        signals = []
        
        # Time-based exit
        if current_time >= end_time:
            for position in active_positions:
                signals.append({
                    'leg_id': position['leg_id'],
                    'signal_type': 'EXIT',
                    'signal_time': end_time,
                    'exit_reason': 'TIME_EXIT',
                    'position_id': position.get('position_id')
                })
        
        return signals
    
    def _should_trade_leg(self, leg: Dict[str, Any], breakout_type: BreakoutType) -> bool:
        """
        Determine if a leg should be traded based on breakout direction
        
        Default logic:
        - HIGHBREAKOUT: Trade calls (buy call or sell put typically)
        - LOWBREAKOUT: Trade puts (buy put or sell call typically)
        """
        if breakout_type == BreakoutType.HIGHBREAKOUT:
            # Bullish breakout - typically trade calls or sell puts
            if leg['instrument'] == 'CE' and leg['transaction'] == 'BUY':
                return True
            if leg['instrument'] == 'PE' and leg['transaction'] == 'SELL':
                return True
        elif breakout_type == BreakoutType.LOWBREAKOUT:
            # Bearish breakout - typically trade puts or sell calls
            if leg['instrument'] == 'PE' and leg['transaction'] == 'BUY':
                return True
            if leg['instrument'] == 'CE' and leg['transaction'] == 'SELL':
                return True
        
        return False
    
    def _calculate_default_sl(
        self,
        breakout_type: BreakoutType,
        range_high: float,
        range_low: float
    ) -> float:
        """
        Calculate default stop loss based on breakout type
        
        - For HIGHBREAKOUT: SL at range low
        - For LOWBREAKOUT: SL at range high
        """
        if breakout_type == BreakoutType.HIGHBREAKOUT:
            return range_low
        elif breakout_type == BreakoutType.LOWBREAKOUT:
            return range_high
        return 0
    
    def _calculate_default_target(
        self,
        breakout_type: BreakoutType,
        breakout_price: float,
        range_high: float,
        range_low: float,
        target_multiplier: float = 2.0
    ) -> float:
        """
        Calculate default target based on range size
        
        Default: 2x the range size from breakout point
        """
        range_size = range_high - range_low
        
        if breakout_type == BreakoutType.HIGHBREAKOUT:
            return breakout_price + (range_size * target_multiplier)
        elif breakout_type == BreakoutType.LOWBREAKOUT:
            return breakout_price - (range_size * target_multiplier)
        return 0
    
    def check_reentry_conditions(
        self,
        db_connection: Any,
        index: str,
        trade_date: date,
        last_exit_time: time,
        last_exit_reason: str,
        reentry_params: Dict[str, Any],
        range_data: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """
        Check if re-entry conditions are met after a previous exit
        
        Args:
            db_connection: Database connection
            index: Index name
            trade_date: Trading date
            last_exit_time: Time of last exit
            last_exit_reason: Reason for last exit (SL_HIT, TGT_HIT)
            reentry_params: Re-entry configuration
            range_data: Original opening range data
            
        Returns:
            Re-entry signal if conditions are met
        """
        # Check if re-entry is allowed
        if last_exit_reason == 'SL_HIT' and reentry_params.get('sl_reentry_no', 0) <= 0:
            return None
        if last_exit_reason == 'TGT_HIT' and reentry_params.get('tgt_reentry_no', 0) <= 0:
            return None
        
        # Wait for reentry interval
        reentry_interval = reentry_params.get('reentry_checking_interval', 60)
        min_reentry_time = (datetime.combine(trade_date, last_exit_time) + 
                           timedelta(seconds=reentry_interval)).time()
        
        # Check for new breakout or continuation
        # Implementation depends on re-entry type (cost, original, instant new strike, etc.)
        # This is a simplified version
        
        return None  # Placeholder
    
    def _get_table_name(self, index: str) -> str:
        """Get table name for index"""
        table_map = {
            'NIFTY': 'nifty_option_chain',
            'BANKNIFTY': 'banknifty_option_chain',
            'FINNIFTY': 'finnifty_option_chain',
            'MIDCPNIFTY': 'midcpnifty_option_chain',
            'SENSEX': 'sensex_option_chain',
            'BANKEX': 'bankex_option_chain'
        }
        return table_map.get(index.upper(), 'nifty_option_chain')
    
    def _get_price_column(self, underlying_type: str) -> str:
        """Get price column based on underlying type"""
        if underlying_type.upper() == 'FUT':
            return 'fut_close'
        return 'spot'
    
    def clear_cache(self):
        """Clear the breakout cache"""
        self._breakout_cache.clear()
        logger.info("Breakout cache cleared")