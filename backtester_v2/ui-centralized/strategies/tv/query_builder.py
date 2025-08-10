#!/usr/bin/env python3
"""
TV Query Builder - Generates SQL queries for TV signal execution
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import date, time

logger = logging.getLogger(__name__)


class TVQueryBuilder:
    """Builds SQL queries for TV signal execution"""
    
    def build_query(
        self,
        signal: Dict[str, Any],
        tv_settings: Dict[str, Any]
    ) -> str:
        """
        Build SQL query for a TV signal (simplified for testing)
        
        Args:
            signal: Processed TV signal
            tv_settings: TV settings
            
        Returns:
            SQL query string
        """
        return f"""
        -- TV Signal Query for testing
        -- Trade: {signal.get('trade_no', 'N/A')}
        -- Direction: {signal.get('signal_direction', 'N/A')}
        SELECT 
            '{signal.get('trade_no', 'N/A')}' as trade_no,
            '{signal.get('signal_direction', 'N/A')}' as direction,
            1 as test_result
        """
    
    def build_signal_query(
        self,
        signal: Dict[str, Any],
        portfolio: Dict[str, Any],
        strategy: Dict[str, Any],
        tv_settings: Dict[str, Any]
    ) -> str:
        """
        Build SQL query for a single TV signal
        
        Args:
            signal: Processed TV signal
            portfolio: Portfolio settings
            strategy: TBS strategy with legs
            tv_settings: TV settings
            
        Returns:
            SQL query string
        """
        
        # Extract signal parameters
        entry_date = signal['entry_date']
        entry_time = signal['entry_time']
        exit_date = signal['exit_date']
        exit_time = signal['exit_time']
        lots = signal['lots']
        
        # Build query parts
        query_parts = []
        
        # Add header comment
        query_parts.append(f"""
        -- TV Signal Query
        -- Trade: {signal['trade_no']}
        -- Direction: {signal['signal_direction']}
        -- Entry: {entry_date} {entry_time}
        -- Exit: {exit_date} {exit_time}
        """)
        
        # Build leg queries
        for leg in strategy.get('legs', []):
            leg_query = self._build_leg_query(
                leg, signal, portfolio, strategy, tv_settings
            )
            if leg_query:
                query_parts.append(leg_query)
        
        # Combine all parts
        if len(query_parts) > 1:
            return '\nUNION ALL\n'.join(query_parts[1:])  # Skip header comment
        else:
            return "-- No valid legs for query generation"
    
    def _build_leg_query(
        self,
        leg: Dict[str, Any],
        signal: Dict[str, Any],
        portfolio: Dict[str, Any],
        strategy: Dict[str, Any],
        tv_settings: Dict[str, Any]
    ) -> Optional[str]:
        """Build query for a single leg"""
        
        # Extract leg parameters
        instrument = leg.get('option_type', 'CE')
        transaction_type = leg.get('transaction_type', 'BUY')
        strike_selection = leg.get('strike_selection', 'ATM')
        strike_value = leg.get('strike_value', 0)
        expiry_rule = leg.get('expiry_rule', 'CW')
        leg_lots = leg.get('quantity', 1) * signal['lots']
        
        # Build strike selection logic
        strike_logic = self._build_strike_logic(
            strike_selection, strike_value, instrument
        )
        
        # Build expiry selection logic
        expiry_logic = self._build_expiry_logic(expiry_rule)
        
        # Build the main query
        query = f"""
        WITH signal_params AS (
            SELECT
                DATE '{signal['entry_date']}' as entry_date,
                TIME '{signal['entry_time']}' as entry_time,
                DATE '{signal['exit_date']}' as exit_date,
                TIME '{signal['exit_time']}' as exit_time,
                {leg_lots} as lots,
                '{instrument}' as instrument,
                '{transaction_type}' as transaction_type
        ),
        {strike_logic},
        {expiry_logic},
        trades AS (
            SELECT
                sp.entry_date,
                sp.entry_time,
                sp.exit_date,
                sp.exit_time,
                sp.lots,
                sp.instrument,
                sp.transaction_type,
                ss.selected_strike as strike,
                es.selected_expiry as expiry_date,
                -- Entry price
                CASE 
                    WHEN sp.instrument = 'CE' THEN ce_close
                    WHEN sp.instrument = 'PE' THEN pe_close
                    ELSE 0
                END as entry_price,
                -- Calculate P&L based on transaction type
                CASE
                    WHEN sp.transaction_type = 'BUY' THEN
                        CASE 
                            WHEN sp.instrument = 'CE' THEN 
                                (ce_close_exit - ce_close) * sp.lots * {portfolio.get('lot_size', 50)}
                            WHEN sp.instrument = 'PE' THEN 
                                (pe_close_exit - pe_close) * sp.lots * {portfolio.get('lot_size', 50)}
                        END
                    WHEN sp.transaction_type = 'SELL' THEN
                        CASE 
                            WHEN sp.instrument = 'CE' THEN 
                                (ce_close - ce_close_exit) * sp.lots * {portfolio.get('lot_size', 50)}
                            WHEN sp.instrument = 'PE' THEN 
                                (pe_close - pe_close_exit) * sp.lots * {portfolio.get('lot_size', 50)}
                        END
                END as pnl
            FROM signal_params sp
            CROSS JOIN strike_selection ss
            CROSS JOIN expiry_selection es
            JOIN nifty_option_chain noc ON
                noc.trade_date = sp.entry_date
                AND noc.trade_time = sp.entry_time
                AND noc.strike = ss.selected_strike
                AND noc.expiry_date = es.selected_expiry
            LEFT JOIN LATERAL (
                SELECT 
                    ce_close as ce_close_exit,
                    pe_close as pe_close_exit
                FROM nifty_option_chain
                WHERE trade_date = sp.exit_date
                    AND trade_time = sp.exit_time
                    AND strike = ss.selected_strike
                    AND expiry_date = es.selected_expiry
                LIMIT 1
            ) exit_prices ON true
        )
        SELECT 
            '{signal['trade_no']}' as trade_no,
            '{leg.get('leg_no', 1)}' as leg_no,
            *
        FROM trades
        """
        
        return query
    
    def _build_strike_logic(
        self, 
        strike_selection: str, 
        strike_value: float,
        instrument: str
    ) -> str:
        """Build strike selection CTE"""
        
        if strike_selection == 'ATM':
            # ATM with offset
            if strike_value == 0:
                # Pure ATM
                return """
                strike_selection AS (
                    SELECT 
                        strike as selected_strike
                    FROM nifty_option_chain
                    WHERE trade_date = (SELECT entry_date FROM signal_params)
                        AND trade_time = (SELECT entry_time FROM signal_params)
                        AND spot IS NOT NULL
                    ORDER BY ABS(strike - spot)
                    LIMIT 1
                )
                """
            else:
                # ATM with offset
                offset_direction = 'ASC' if strike_value > 0 else 'DESC'
                offset_count = abs(int(strike_value))
                
                return f"""
                strike_selection AS (
                    WITH atm AS (
                        SELECT strike
                        FROM nifty_option_chain
                        WHERE trade_date = (SELECT entry_date FROM signal_params)
                            AND trade_time = (SELECT entry_time FROM signal_params)
                            AND spot IS NOT NULL
                        ORDER BY ABS(strike - spot)
                        LIMIT 1
                    ),
                    strikes_ordered AS (
                        SELECT DISTINCT strike
                        FROM nifty_option_chain
                        WHERE trade_date = (SELECT entry_date FROM signal_params)
                        ORDER BY strike {offset_direction}
                    )
                    SELECT strike as selected_strike
                    FROM strikes_ordered
                    WHERE strike {'>' if offset_direction == 'ASC' else '<'} (SELECT strike FROM atm)
                    LIMIT 1 OFFSET {offset_count - 1}
                )
                """
        
        elif strike_selection == 'FIXED':
            # Fixed strike
            return f"""
            strike_selection AS (
                SELECT {strike_value} as selected_strike
            )
            """
        
        else:
            # Default to ATM
            return """
            strike_selection AS (
                SELECT 
                    strike as selected_strike
                FROM nifty_option_chain
                WHERE trade_date = (SELECT entry_date FROM signal_params)
                    AND trade_time = (SELECT entry_time FROM signal_params)
                    AND spot IS NOT NULL
                ORDER BY ABS(strike - spot)
                LIMIT 1
            )
            """
    
    def _build_expiry_logic(self, expiry_rule: str) -> str:
        """Build expiry selection CTE"""
        
        if expiry_rule == 'CW':
            # Current week expiry
            return """
            expiry_selection AS (
                SELECT MIN(expiry_date) as selected_expiry
                FROM nifty_option_chain
                WHERE trade_date = (SELECT entry_date FROM signal_params)
                    AND expiry_date >= (SELECT entry_date FROM signal_params)
            )
            """
        elif expiry_rule == 'NW':
            # Next week expiry
            return """
            expiry_selection AS (
                WITH expiries AS (
                    SELECT DISTINCT expiry_date
                    FROM nifty_option_chain
                    WHERE trade_date = (SELECT entry_date FROM signal_params)
                        AND expiry_date >= (SELECT entry_date FROM signal_params)
                    ORDER BY expiry_date
                    LIMIT 2
                )
                SELECT MAX(expiry_date) as selected_expiry
                FROM expiries
            )
            """
        elif expiry_rule == 'CM':
            # Current month expiry
            return """
            expiry_selection AS (
                SELECT MIN(expiry_date) as selected_expiry
                FROM nifty_option_chain
                WHERE trade_date = (SELECT entry_date FROM signal_params)
                    AND expiry_date >= (SELECT entry_date FROM signal_params)
                    AND EXTRACT(DAY FROM expiry_date) > 20  -- Monthly expiry heuristic
            )
            """
        else:
            # Default to current week
            return """
            expiry_selection AS (
                SELECT MIN(expiry_date) as selected_expiry
                FROM nifty_option_chain
                WHERE trade_date = (SELECT entry_date FROM signal_params)
                    AND expiry_date >= (SELECT entry_date FROM signal_params)
            )
            """