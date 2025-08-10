#!/usr/bin/env python3
"""
ORB Range Calculator - Calculates opening range (high/low) for ORB strategies
"""

from datetime import date, time, datetime
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class RangeCalculator:
    """Calculates opening range for ORB strategies"""
    
    def __init__(self):
        self._range_cache = {}  # Cache calculated ranges
    
    def calculate_opening_range(
        self, 
        db_connection: Any,
        index: str,
        trade_date: date,
        range_start: time,
        range_end: time,
        underlying_type: str = 'SPOT'
    ) -> Optional[Dict[str, float]]:
        """
        Calculate the opening range (high/low) for a given date and time window
        
        Args:
            db_connection: Database connection
            index: Index name (NIFTY, BANKNIFTY, etc.)
            trade_date: Date to calculate range for
            range_start: Start time of opening range
            range_end: End time of opening range
            underlying_type: SPOT or FUT
            
        Returns:
            Dictionary with range_high, range_low, range_size
        """
        
        # Check cache first
        cache_key = f"{index}_{trade_date}_{range_start}_{range_end}_{underlying_type}"
        if cache_key in self._range_cache:
            logger.debug(f"Using cached range for {cache_key}")
            return self._range_cache[cache_key]
        
        # Build SQL query
        table_name = self._get_table_name(index)
        price_column = self._get_price_column(underlying_type)
        
        sql = f"""
        SELECT 
            MAX({price_column}) AS range_high,
            MIN({price_column}) AS range_low,
            MAX({price_column}) - MIN({price_column}) AS range_size,
            COUNT(*) AS tick_count,
            MIN(trade_time) AS actual_start_time,
            MAX(trade_time) AS actual_end_time
        FROM {table_name}
        WHERE trade_date = DATE '{trade_date}'
            AND trade_time BETWEEN TIME '{range_start}' AND TIME '{range_end}'
            AND {price_column} IS NOT NULL
            AND {price_column} > 0
        """
        
        try:
            result = db_connection.execute(sql).fetchone()
            
            if result and result[0] is not None:  # range_high exists
                range_data = {
                    'range_high': float(result[0]),
                    'range_low': float(result[1]),
                    'range_size': float(result[2]),
                    'tick_count': int(result[3]),
                    'actual_start_time': result[4],
                    'actual_end_time': result[5]
                }
                
                # Validate range
                if range_data['range_size'] <= 0:
                    logger.warning(f"Invalid range size: {range_data['range_size']}")
                    return None
                
                # Cache the result
                self._range_cache[cache_key] = range_data
                
                logger.info(f"Calculated range for {index} on {trade_date}: "
                          f"High={range_data['range_high']}, Low={range_data['range_low']}, "
                          f"Size={range_data['range_size']}")
                
                return range_data
            else:
                logger.warning(f"No data found for range calculation: {index} on {trade_date}")
                return None
                
        except Exception as e:
            logger.error(f"Error calculating opening range: {e}")
            return None
    
    def get_range_breakout_levels(
        self,
        range_data: Dict[str, float],
        buffer_points: float = 0
    ) -> Dict[str, float]:
        """
        Calculate breakout levels based on opening range
        
        Args:
            range_data: Dictionary with range_high, range_low
            buffer_points: Optional buffer to add/subtract from levels
            
        Returns:
            Dictionary with breakout levels
        """
        if not range_data:
            return {}
        
        return {
            'bullish_breakout_level': range_data['range_high'] + buffer_points,
            'bearish_breakout_level': range_data['range_low'] - buffer_points,
            'bullish_target_1': range_data['range_high'] + range_data['range_size'],
            'bullish_target_2': range_data['range_high'] + (2 * range_data['range_size']),
            'bearish_target_1': range_data['range_low'] - range_data['range_size'],
            'bearish_target_2': range_data['range_low'] - (2 * range_data['range_size']),
            'bullish_stoploss': range_data['range_low'] - buffer_points,
            'bearish_stoploss': range_data['range_high'] + buffer_points
        }
    
    def is_valid_range(
        self,
        range_data: Dict[str, float],
        min_range_size: Optional[float] = None,
        max_range_size: Optional[float] = None,
        min_tick_count: int = 10
    ) -> bool:
        """
        Validate if the calculated range is tradeable
        
        Args:
            range_data: Calculated range data
            min_range_size: Minimum acceptable range size
            max_range_size: Maximum acceptable range size  
            min_tick_count: Minimum number of ticks required
            
        Returns:
            True if range is valid for trading
        """
        if not range_data:
            return False
        
        # Check tick count
        if range_data.get('tick_count', 0) < min_tick_count:
            logger.warning(f"Insufficient ticks in range: {range_data.get('tick_count')}")
            return False
        
        # Check range size
        range_size = range_data.get('range_size', 0)
        
        if min_range_size and range_size < min_range_size:
            logger.warning(f"Range too small: {range_size} < {min_range_size}")
            return False
        
        if max_range_size and range_size > max_range_size:
            logger.warning(f"Range too large: {range_size} > {max_range_size}")
            return False
        
        return True
    
    def get_range_percentile(
        self,
        db_connection: Any,
        index: str,
        current_range_size: float,
        lookback_days: int = 20
    ) -> float:
        """
        Calculate what percentile the current range size falls into
        compared to historical ranges
        
        Args:
            db_connection: Database connection
            index: Index name
            current_range_size: Today's range size
            lookback_days: Number of days to look back
            
        Returns:
            Percentile (0-100)
        """
        table_name = self._get_table_name(index)
        
        sql = f"""
        WITH historical_ranges AS (
            SELECT 
                trade_date,
                MAX(spot) - MIN(spot) AS daily_range
            FROM {table_name}
            WHERE trade_date >= CURRENT_DATE - INTERVAL '{lookback_days}' DAY
                AND trade_date < CURRENT_DATE
                AND trade_time BETWEEN TIME '09:15:00' AND TIME '09:20:00'
            GROUP BY trade_date
        )
        SELECT 
            COUNT(CASE WHEN daily_range <= {current_range_size} THEN 1 END) * 100.0 / COUNT(*) AS percentile
        FROM historical_ranges
        """
        
        try:
            result = db_connection.execute(sql).fetchone()
            if result:
                return float(result[0])
            return 50.0  # Default to median
        except Exception as e:
            logger.error(f"Error calculating range percentile: {e}")
            return 50.0
    
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
        """Clear the range cache"""
        self._range_cache.clear()
        logger.info("Range cache cleared")