#!/usr/bin/env python3
"""
ORB Query Builder - Generates SQL queries for ORB strategy execution
"""

from datetime import date, time
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ORBQueryBuilder:
    """Builds SQL queries for ORB strategy execution"""
    
    def build_orb_query(
        self,
        strategy: Dict[str, Any],
        signals: List[Dict[str, Any]],
        trade_date: date
    ) -> str:
        """
        Build complete ORB query combining range calculation, breakout detection,
        and option selection
        
        Args:
            strategy: Strategy parameters
            signals: List of entry signals
            trade_date: Trading date
            
        Returns:
            Combined SQL query
        """
        if not signals:
            return "-- No signals to process"
        
        # Get strategy parameters
        index = strategy['index']
        table_name = self._get_table_name(index)
        
        # Build CTEs
        query_parts = [
            self._build_opening_range_cte(strategy, trade_date, table_name),
            self._build_breakout_cte(strategy, trade_date, table_name),
            self._build_option_selection_cte(signals, table_name)
        ]
        
        # Add main query
        query_parts.append(self._build_main_query(len(signals)))
        
        return "\n".join(query_parts)
    
    def _build_header_comment(self, strategy: Dict[str, Any], trade_date: date) -> str:
        """Build header comment for the query"""
        return f"""
-- ORB Strategy Query
-- Strategy: {strategy['strategy_name']}
-- Date: {trade_date}
-- Range: {strategy['orb_range_start']} to {strategy['orb_range_end']}
-- Index: {strategy['index']}
-- DTE: {strategy['dte']}
"""
    
    def _build_opening_range_cte(
        self,
        strategy: Dict[str, Any],
        trade_date: date,
        table_name: str
    ) -> str:
        """Build opening range calculation CTE"""
        
        price_column = 'spot' if strategy['underlying'] == 'SPOT' else 'fut_close'
        range_start = strategy['orb_range_start']
        range_end = strategy['orb_range_end']
        
        return f"""
WITH opening_range AS (
    SELECT 
        MAX({price_column}) AS range_high,
        MIN({price_column}) AS range_low,
        MAX({price_column}) - MIN({price_column}) AS range_size,
        COUNT(*) AS tick_count,
        AVG({price_column}) AS range_avg
    FROM {table_name}
    WHERE trade_date = DATE '{trade_date}'
        AND trade_time BETWEEN TIME '{range_start}' AND TIME '{range_end}'
        AND {price_column} IS NOT NULL
        AND {price_column} > 0
)"""
    
    def _build_breakout_cte(
        self,
        strategy: Dict[str, Any],
        trade_date: date,
        table_name: str
    ) -> str:
        """Build breakout detection CTE"""
        
        price_column = 'spot' if strategy['underlying'] == 'SPOT' else 'fut_close'
        range_end = strategy['orb_range_end']
        last_entry = strategy['last_entry_time']
        
        return f""",
breakout_detection AS (
    SELECT
        oc.trade_time AS breakout_time,
        oc.{price_column} AS breakout_price,
        oc.atm_strike,
        CASE
            WHEN oc.{price_column} > r.range_high THEN 'HIGHBREAKOUT'
            WHEN oc.{price_column} < r.range_low THEN 'LOWBREAKOUT'
            ELSE NULL
        END AS breakout_type,
        CASE
            WHEN oc.{price_column} > r.range_high THEN 
                (oc.{price_column} - r.range_high) / NULLIF(r.range_size, 0) * 100
            WHEN oc.{price_column} < r.range_low THEN 
                (r.range_low - oc.{price_column}) / NULLIF(r.range_size, 0) * 100
            ELSE 0
        END AS breakout_strength_pct,
        r.range_high,
        r.range_low,
        r.range_size
    FROM {table_name} oc
    CROSS JOIN opening_range r
    WHERE oc.trade_date = DATE '{trade_date}'
        AND oc.trade_time > TIME '{range_end}'
        AND oc.trade_time <= TIME '{last_entry}'
        AND (oc.{price_column} > r.range_high OR oc.{price_column} < r.range_low)
    ORDER BY oc.trade_time
    LIMIT 1
)"""
    
    def _build_option_selection_cte(
        self,
        signals: List[Dict[str, Any]],
        table_name: str
    ) -> str:
        """Build option selection CTEs for each leg"""
        
        if not signals:
            return ""
        
        ctes = []
        
        for i, signal in enumerate(signals):
            leg_cte = self._build_leg_cte(signal, i + 1, table_name)
            ctes.append(leg_cte)
        
        return ",\n".join(ctes)
    
    def _build_leg_cte(
        self,
        signal: Dict[str, Any],
        leg_num: int,
        table_name: str
    ) -> str:
        """Build CTE for a single leg option selection"""
        
        leg_id = signal['leg_id']
        instrument = signal['instrument']
        expiry_rule = signal['expiry']
        strike_method = signal['strike_method']
        strike_value = signal.get('strike_value', 0)
        
        # Build strike selection logic
        strike_logic = self._build_strike_selection(
            strike_method, 
            strike_value,
            instrument
        )
        
        # Build expiry selection logic
        expiry_logic = self._build_expiry_selection(expiry_rule)
        
        # Determine option columns
        if instrument == 'CE':
            symbol_col = 'ce_symbol'
            price_col = 'ce_close'
            oi_col = 'ce_oi'
        else:
            symbol_col = 'pe_symbol'
            price_col = 'pe_close'
            oi_col = 'pe_oi'
        
        return f""",
leg_{leg_num}_selection AS (
    SELECT
        '{leg_id}' AS leg_id,
        bd.breakout_time AS entry_time,
        bd.breakout_price AS underlying_at_entry,
        bd.breakout_type,
        bd.breakout_strength_pct,
        oc.strike,
        oc.{symbol_col} AS symbol,
        oc.{price_col} AS entry_price,
        oc.{oi_col} AS open_interest,
        oc.expiry_date,
        '{instrument}' AS instrument,
        '{signal['transaction']}' AS transaction_type,
        {signal['lots']} AS lots
    FROM {table_name} oc
    JOIN breakout_detection bd ON 1=1
    WHERE oc.trade_date = DATE '{signal['signal_time'].date()}'
        AND oc.trade_time = bd.breakout_time
        AND {expiry_logic}
        AND {strike_logic}
        AND oc.{symbol_col} IS NOT NULL
    ORDER BY oc.{price_col} DESC
    LIMIT 1
)"""
    
    def _build_strike_selection(
        self,
        strike_method: str,
        strike_value: float,
        instrument: str
    ) -> str:
        """Build strike selection logic"""
        
        if strike_method == 'ATM':
            if strike_value == 0:
                return "oc.strike = oc.atm_strike"
            else:
                # ATM with offset
                if instrument == 'CE':
                    if strike_value > 0:
                        return f"oc.strike = oc.atm_strike + ({strike_value} * oc.strike_step)"
                    else:
                        return f"oc.strike = oc.atm_strike - ({abs(strike_value)} * oc.strike_step)"
                else:  # PE
                    if strike_value > 0:
                        return f"oc.strike = oc.atm_strike - ({strike_value} * oc.strike_step)"
                    else:
                        return f"oc.strike = oc.atm_strike + ({abs(strike_value)} * oc.strike_step)"
        
        elif strike_method == 'FIXED':
            return f"oc.strike = {strike_value}"
        
        elif strike_method.startswith('ITM'):
            steps = int(strike_method[3:]) if len(strike_method) > 3 else 1
            if instrument == 'CE':
                return f"oc.strike = oc.atm_strike - ({steps} * oc.strike_step)"
            else:
                return f"oc.strike = oc.atm_strike + ({steps} * oc.strike_step)"
        
        elif strike_method.startswith('OTM'):
            steps = int(strike_method[3:]) if len(strike_method) > 3 else 1
            if instrument == 'CE':
                return f"oc.strike = oc.atm_strike + ({steps} * oc.strike_step)"
            else:
                return f"oc.strike = oc.atm_strike - ({steps} * oc.strike_step)"
        
        else:
            # Default to ATM
            return "oc.strike = oc.atm_strike"
    
    def _build_expiry_selection(self, expiry_rule: str) -> str:
        """Build expiry selection logic"""
        
        expiry_mapping = {
            'CW': "oc.expiry_bucket = 'CW'",
            'NW': "oc.expiry_bucket = 'NW'",
            'CM': "oc.expiry_bucket = 'CM'",
            'NM': "oc.expiry_bucket = 'NM'"
        }
        
        return expiry_mapping.get(expiry_rule, "oc.expiry_bucket = 'CW'")
    
    def _build_main_query(self, num_legs: int = 1) -> str:
        """Build main query to combine all CTEs"""
        
        if num_legs == 1:
            return "SELECT * FROM leg_1_selection"
        
        parts = [f"SELECT * FROM leg_{i}_selection" for i in range(1, num_legs + 1)]
        return "\nUNION ALL\n".join(parts)
    
    def build_exit_query(
        self,
        strategy: Dict[str, Any],
        positions: List[Dict[str, Any]],
        exit_time: time,
        trade_date: date
    ) -> str:
        """Build query for exit prices"""
        
        if not positions:
            return "-- No positions to exit"
        
        table_name = self._get_table_name(strategy['index'])
        
        # Build UNION of all position exits
        queries = []
        for pos in positions:
            instrument = pos['instrument']
            strike = pos['strike']
            expiry = pos['expiry_date']
            
            if instrument == 'CE':
                price_col = 'ce_close'
            else:
                price_col = 'pe_close'
            
            query = f"""
            SELECT
                '{pos['position_id']}' AS position_id,
                '{pos['leg_id']}' AS leg_id,
                {price_col} AS exit_price,
                spot AS underlying_at_exit
            FROM {table_name}
            WHERE trade_date = DATE '{trade_date}'
                AND trade_time = TIME '{exit_time}'
                AND strike = {strike}
                AND expiry_date = DATE '{expiry}'
            """
            queries.append(query)
        
        return "\nUNION ALL\n".join(queries)
    
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