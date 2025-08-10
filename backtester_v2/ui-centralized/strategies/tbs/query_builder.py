#!/usr/bin/env python3
"""
TBS Query Builder - Generates SQL queries for TBS strategy execution
"""

import logging
from heavyai import connect as heavyai_connect
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, date, time
import pandas as pd

logger = logging.getLogger(__name__)


class TBSQueryBuilder:
    """Builds SQL queries for TBS strategy execution with HeavyDB integration"""

    def __init__(self, table_name: str = 'nifty_option_chain'):
        self.table_name = table_name
        self.query_templates = {
            'atm_calculation': self._get_atm_template(),
            'option_selection': self._get_option_template(),
            'entry_exit': self._get_entry_exit_template()
        }

        # HeavyDB connection parameters
        self.heavydb_params = {
            'host': 'localhost',
            'port': 6274,
            'user': 'admin',
            'password': 'HyperInteractive',
            'dbname': 'heavyai'
        }
    
    def build_queries(self, strategy_params: Dict[str, Any]) -> List[str]:
        """
        Build all queries needed for a TBS strategy
        
        Args:
            strategy_params: Dictionary containing strategy parameters and legs
            
        Returns:
            List of SQL queries to execute
        """
        queries = []
        
        # Extract parameters
        portfolio_params = strategy_params.get('portfolio', {})
        legs = strategy_params.get('legs', [])
        
        start_date = portfolio_params.get('start_date')
        end_date = portfolio_params.get('end_date')
        index = portfolio_params.get('index', 'NIFTY')
        
        # Build queries for each leg
        for leg in legs:
            leg_queries = self._build_leg_queries(
                leg=leg,
                start_date=start_date,
                end_date=end_date,
                index=index,
                portfolio_params=portfolio_params
            )
            queries.extend(leg_queries)
        
        return queries
    
    def _build_leg_queries(self, leg: Dict[str, Any], start_date: date, 
                          end_date: date, index: str, 
                          portfolio_params: Dict[str, Any]) -> List[str]:
        """Build queries for a single leg"""
        queries = []
        
        # Get leg parameters
        option_type = leg.get('option_type', 'CE')
        strike_selection = leg.get('strike_selection', 'ATM')
        strike_value = leg.get('strike_value', 0)
        expiry_rule = leg.get('expiry_rule', 'CW')
        expiry_value = leg.get('expiry_value', 0)
        entry_time = leg.get('entry_time', time(9, 20))
        exit_time = leg.get('exit_time', time(15, 15))
        transaction_type = leg.get('transaction_type', 'SELL')
        quantity = leg.get('quantity', 1)
        
        # Build ATM calculation query if needed
        if strike_selection in ['ATM', 'ITM1', 'ITM2', 'OTM1', 'OTM2']:
            atm_query = self._build_atm_query(
                start_date=start_date,
                end_date=end_date,
                entry_time=entry_time,
                index=index
            )
            queries.append(atm_query)
        
        # Build option selection query
        option_query = self._build_option_query(
            option_type=option_type,
            strike_selection=strike_selection,
            strike_value=strike_value,
            expiry_rule=expiry_rule,
            expiry_value=expiry_value,
            start_date=start_date,
            end_date=end_date,
            entry_time=entry_time,
            exit_time=exit_time,
            index=index
        )
        queries.append(option_query)
        
        # Build entry/exit tracking query
        entry_exit_query = self._build_entry_exit_query(
            leg=leg,
            start_date=start_date,
            end_date=end_date,
            index=index,
            portfolio_params=portfolio_params
        )
        queries.append(entry_exit_query)
        
        return queries
    
    def _build_atm_query(self, start_date: date, end_date: date, 
                        entry_time: time, index: str) -> str:
        """Build query to calculate ATM strikes"""
        query = f"""
        WITH atm_calc AS (
            SELECT 
                trade_date,
                expiry_date,
                MIN(ABS(strike + ce_close - pe_close - strike)) as min_diff,
                FIRST_VALUE(strike) OVER (
                    PARTITION BY trade_date, expiry_date 
                    ORDER BY ABS(strike + ce_close - pe_close - strike)
                ) as atm_strike
            FROM {self.table_name}
            WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'
                AND symbol = '{index}'
                AND time = '{entry_time}'
                AND ce_close > 0 AND pe_close > 0
            GROUP BY trade_date, expiry_date, strike, ce_close, pe_close
        )
        SELECT DISTINCT trade_date, expiry_date, atm_strike
        FROM atm_calc
        ORDER BY trade_date, expiry_date
        """
        return query
    
    def _build_option_query(self, option_type: str, strike_selection: str,
                           strike_value: int, expiry_rule: str, expiry_value: int,
                           start_date: date, end_date: date, entry_time: time,
                           exit_time: time, index: str) -> str:
        """Build query to select options based on criteria"""
        
        # Handle strike selection
        strike_condition = self._get_strike_condition(strike_selection, strike_value)
        
        # Handle expiry selection
        expiry_condition = self._get_expiry_condition(expiry_rule, expiry_value)
        
        # Build price columns based on option type
        if option_type == 'CE':
            price_cols = "ce_open as entry_price, ce_close as exit_price"
        elif option_type == 'PE':
            price_cols = "pe_open as entry_price, pe_close as exit_price"
        else:  # FUT
            price_cols = "fut_open as entry_price, fut_close as exit_price"
        
        query = f"""
        WITH ranked_expiries AS (
            SELECT DISTINCT 
                trade_date,
                expiry_date,
                ROW_NUMBER() OVER (PARTITION BY trade_date ORDER BY expiry_date) as expiry_rank
            FROM {self.table_name}
            WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'
                AND symbol = '{index}'
                AND expiry_date > trade_date
        ),
        selected_options AS (
            SELECT 
                t.trade_date,
                t.expiry_date,
                t.strike,
                t.time,
                {price_cols},
                re.expiry_rank
            FROM {self.table_name} t
            JOIN ranked_expiries re 
                ON t.trade_date = re.trade_date 
                AND t.expiry_date = re.expiry_date
            WHERE t.trade_date BETWEEN '{start_date}' AND '{end_date}'
                AND t.symbol = '{index}'
                AND t.time IN ('{entry_time}', '{exit_time}')
                {strike_condition}
                {expiry_condition}
        )
        SELECT * FROM selected_options
        ORDER BY trade_date, time
        """
        return query
    
    def _build_entry_exit_query(self, leg: Dict[str, Any], start_date: date,
                               end_date: date, index: str,
                               portfolio_params: Dict[str, Any]) -> str:
        """Build query to track entries and exits with P&L calculation"""
        
        option_type = leg.get('option_type', 'CE')
        entry_time = leg.get('entry_time', time(9, 20))
        exit_time = leg.get('exit_time', time(15, 15))
        quantity = leg.get('quantity', 1)
        transaction_type = leg.get('transaction_type', 'SELL')
        sl_percent = leg.get('sl_percent')
        target_percent = leg.get('target_percent')
        
        # Price columns based on option type
        if option_type == 'CE':
            price_col = 'ce_close'
            high_col = 'ce_high'
            low_col = 'ce_low'
        elif option_type == 'PE':
            price_col = 'pe_close'
            high_col = 'pe_high'
            low_col = 'pe_low'
        else:
            price_col = 'fut_close'
            high_col = 'fut_high'
            low_col = 'fut_low'
        
        # Build SL/Target conditions
        sl_condition = ""
        target_condition = ""
        
        if sl_percent:
            if transaction_type == 'SELL':
                sl_condition = f"OR ({high_col} >= entry_price * (1 + {sl_percent}/100))"
            else:
                sl_condition = f"OR ({low_col} <= entry_price * (1 - {sl_percent}/100))"
        
        if target_percent:
            if transaction_type == 'SELL':
                target_condition = f"OR ({low_col} <= entry_price * (1 - {target_percent}/100))"
            else:
                target_condition = f"OR ({high_col} >= entry_price * (1 + {target_percent}/100))"
        
        query = f"""
        WITH entries AS (
            SELECT 
                trade_date,
                expiry_date,
                strike,
                {price_col} as entry_price,
                '{entry_time}' as entry_time
            FROM {self.table_name}
            WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'
                AND symbol = '{index}'
                AND time = '{entry_time}'
        ),
        exits AS (
            SELECT 
                e.trade_date,
                e.expiry_date,
                e.strike,
                e.entry_price,
                e.entry_time,
                t.time as exit_time,
                t.{price_col} as exit_price,
                CASE 
                    WHEN t.time = '{exit_time}' THEN 'TIME_EXIT'
                    {sl_condition.replace('OR', 'WHEN')} THEN 'SL_HIT'
                    {target_condition.replace('OR', 'WHEN')} THEN 'TARGET_HIT'
                    ELSE 'TIME_EXIT'
                END as exit_reason,
                CASE
                    WHEN '{transaction_type}' = 'SELL' THEN 
                        (e.entry_price - t.{price_col}) * {quantity}
                    ELSE 
                        (t.{price_col} - e.entry_price) * {quantity}
                END as pnl
            FROM entries e
            JOIN {self.table_name} t
                ON e.trade_date = t.trade_date
                AND e.expiry_date = t.expiry_date
                AND e.strike = t.strike
            WHERE t.time >= e.entry_time
                AND t.time <= '{exit_time}'
                AND (
                    t.time = '{exit_time}'
                    {sl_condition}
                    {target_condition}
                )
        )
        SELECT * FROM exits
        ORDER BY trade_date, exit_time
        """
        return query
    
    def _get_strike_condition(self, strike_selection: str, strike_value: int) -> str:
        """Get SQL condition for strike selection"""
        if strike_selection == 'FIXED':
            return f"AND t.strike = {strike_value}"
        elif strike_selection in ['ATM', 'ITM1', 'ITM2', 'OTM1', 'OTM2']:
            # This will be joined with ATM calculation results
            return ""  # Handle in join logic
        else:
            return ""
    
    def _get_expiry_condition(self, expiry_rule: str, expiry_value: int) -> str:
        """Get SQL condition for expiry selection"""
        if expiry_rule == 'CW':  # Current Week
            return "AND re.expiry_rank = 1"
        elif expiry_rule == 'NW':  # Next Week
            return "AND re.expiry_rank = 2"
        elif expiry_rule == 'CM':  # Current Month
            return "AND EXTRACT(MONTH FROM t.expiry_date) = EXTRACT(MONTH FROM t.trade_date)"
        elif expiry_rule == 'NM':  # Next Month
            return "AND EXTRACT(MONTH FROM t.expiry_date) = EXTRACT(MONTH FROM t.trade_date) + 1"
        elif expiry_rule == 'FIXED':
            return f"AND DATEDIFF('day', t.trade_date, t.expiry_date) = {expiry_value}"
        else:
            return ""
    
    def _get_atm_template(self) -> str:
        """Get template for ATM calculation query"""
        return """
        WITH synthetic_future AS (
            SELECT 
                trade_date,
                expiry_date,
                strike,
                strike + ce_close - pe_close as synthetic_price,
                ABS(strike - (strike + ce_close - pe_close)) as diff_from_strike
            FROM {table_name}
            WHERE {conditions}
        ),
        atm_strikes AS (
            SELECT 
                trade_date,
                expiry_date,
                strike as atm_strike,
                ROW_NUMBER() OVER (
                    PARTITION BY trade_date, expiry_date 
                    ORDER BY diff_from_strike
                ) as rn
            FROM synthetic_future
        )
        SELECT trade_date, expiry_date, atm_strike
        FROM atm_strikes
        WHERE rn = 1
        """
    
    def _get_option_template(self) -> str:
        """Get template for option selection query"""
        return """
        SELECT 
            trade_date,
            expiry_date,
            strike,
            time,
            {price_columns}
        FROM {table_name}
        WHERE {conditions}
        ORDER BY trade_date, time
        """
    
    def _get_entry_exit_template(self) -> str:
        """Get template for entry/exit tracking query"""
        return """
        WITH trades AS (
            SELECT 
                {select_columns}
            FROM {table_name}
            WHERE {conditions}
        )
        SELECT 
            {output_columns},
            {pnl_calculation} as pnl
        FROM trades
        """

    def connect_to_heavydb(self):
        """Establish connection to HeavyDB"""
        try:
            connection = heavyai_connect(
                host=self.heavydb_params['host'],
                port=self.heavydb_params['port'],
                user=self.heavydb_params['user'],
                password=self.heavydb_params['password'],
                dbname=self.heavydb_params['dbname']
            )
            logger.info("Successfully connected to HeavyDB")
            return connection
        except Exception as e:
            logger.error(f"Failed to connect to HeavyDB: {e}")
            raise

    def execute_query(self, query: str, connection=None):
        """Execute query on HeavyDB"""
        if connection is None:
            connection = self.connect_to_heavydb()

        try:
            cursor = connection.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            logger.info(f"Query executed successfully, returned {len(result)} rows")
            return result
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
        finally:
            if cursor:
                cursor.close()

    def build_optimized_query(self, base_query: str) -> str:
        """Build optimized query for HeavyDB GPU execution"""
        # Add GPU optimization hints
        optimized_query = f"/*+ cpu_mode=false, watchdog_max_size=0 */ {base_query}"
        return optimized_query