#!/usr/bin/env python3
"""
OI Query Builder - Generates SQL queries for OI strategy execution
"""

from datetime import date, time
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class OIQueryBuilder:
    """Builds SQL queries for OI strategy execution"""
    
    def build_oi_ranking_query(
        self,
        table_name: str,
        trade_date: date,
        analysis_time: time,
        strike_count: int = 5,
        underlying_type: str = 'SPOT'
    ) -> str:
        """
        Build query to calculate OI rankings for strike selection
        
        Args:
            table_name: Option chain table name
            trade_date: Trading date
            analysis_time: Time to analyze OI
            strike_count: Number of strikes each side of ATM to analyze
            underlying_type: 'SPOT' or 'FUTURE'
            
        Returns:
            SQL query for OI ranking
        """
        return f"""
        WITH atm_data AS (
            SELECT 
                atm_strike,
                spot,
                future_price,
                strike_step
            FROM {table_name}
            WHERE trade_date = DATE '{trade_date}'
                AND trade_time = TIME '{analysis_time}'
            LIMIT 1
        ),
        oi_analysis AS (
            SELECT 
                oc.strike,
                oc.ce_oi,
                oc.pe_oi,
                oc.ce_symbol,
                oc.pe_symbol,
                atm.atm_strike,
                ABS(oc.strike - atm.atm_strike) as distance_from_atm,
                atm.strike_step
            FROM {table_name} oc
            CROSS JOIN atm_data atm
            WHERE oc.trade_date = DATE '{trade_date}'
                AND oc.trade_time = TIME '{analysis_time}'
                AND ABS(oc.strike - atm.atm_strike) <= {strike_count} * atm.strike_step
                AND oc.ce_oi > 0 
                AND oc.pe_oi > 0
        )
        SELECT 
            strike,
            ce_oi,
            pe_oi,
            ce_symbol,
            pe_symbol,
            distance_from_atm
        FROM oi_analysis
        ORDER BY distance_from_atm, strike
        """
    
    def build_entry_signal_query(
        self,
        table_name: str,
        trade_date: date,
        strike: int,
        instrument: str,
        start_time: time,
        end_time: time
    ) -> str:
        """
        Build query to get entry signals for a specific strike
        
        Args:
            table_name: Option chain table name
            trade_date: Trading date
            strike: Strike price
            instrument: 'CE' or 'PE'
            start_time: Start time for entry
            end_time: End time for entry
            
        Returns:
            SQL query for entry signals
        """
        price_col = 'ce_close' if instrument == 'CE' else 'pe_close'
        symbol_col = 'ce_symbol' if instrument == 'CE' else 'pe_symbol'
        
        return f"""
        SELECT 
            trade_time,
            {price_col} as option_price,
            {symbol_col} as symbol,
            spot,
            future_price,
            expiry_date
        FROM {table_name}
        WHERE trade_date = DATE '{trade_date}'
            AND strike = {strike}
            AND trade_time >= TIME '{start_time}'
            AND trade_time <= TIME '{end_time}'
            AND {price_col} > 0
        ORDER BY trade_time
        LIMIT 1
        """
    
    def build_oi_entry_query(
        self,
        strategy: Dict[str, Any],
        signals: List[Dict[str, Any]],
        trade_date: date
    ) -> str:
        """
        Build OI entry query for strike selection and option data retrieval
        
        Args:
            strategy: Strategy parameters
            signals: List of OI entry signals
            trade_date: Trading date
            
        Returns:
            SQL query for option selection
        """
        if not signals:
            return "-- No OI signals to process"
        
        index = strategy['index']
        table_name = self._get_table_name(index)
        
        # Build CTEs for each signal
        query_parts = []
        
        for i, signal in enumerate(signals):
            leg_cte = self._build_oi_leg_cte(signal, i + 1, table_name, trade_date)
            query_parts.append(leg_cte)
        
        # Combine all leg selections
        main_query = self._build_main_union_query(len(signals))
        query_parts.append(main_query)
        
        return "\n".join(query_parts)
    
    def _build_oi_leg_cte(
        self,
        signal: Dict[str, Any],
        leg_num: int,
        table_name: str,
        trade_date: date
    ) -> str:
        """Build CTE for a single OI leg selection"""
        
        leg_id = signal['leg_id']
        instrument = signal['instrument']
        strike = signal['strike']
        signal_time = signal['signal_time']
        expiry_rule = signal.get('expiry', 'CW')
        
        # Build expiry selection logic
        expiry_logic = self._build_expiry_selection(expiry_rule)
        
        # Determine option columns based on instrument
        if instrument == 'CE':
            symbol_col = 'ce_symbol'
            price_col = 'ce_close'
            oi_col = 'ce_oi'
        else:  # PE
            symbol_col = 'pe_symbol'
            price_col = 'pe_close'
            oi_col = 'pe_oi'
        
        return f"""
oi_leg_{leg_num}_selection AS (
    SELECT
        '{leg_id}' AS leg_id,
        '{signal_time}' AS entry_time,
        oc.spot AS underlying_at_entry,
        {strike} AS strike,
        oc.{symbol_col} AS symbol,
        oc.{price_col} AS entry_price,
        oc.{oi_col} AS open_interest,
        oc.expiry_date,
        '{instrument}' AS instrument,
        '{signal['transaction']}' AS transaction_type,
        {signal['lots']} AS lots,
        '{signal['selection_method']}' AS selection_method,
        {signal['oi_rank']} AS oi_rank,
        {signal['oi_value']} AS oi_value
    FROM {table_name} oc
    WHERE oc.trade_date = DATE '{trade_date}'
        AND oc.trade_time = TIME '{signal_time}'
        AND oc.strike = {strike}
        AND {expiry_logic}
        AND oc.{symbol_col} IS NOT NULL
        AND oc.{price_col} > 0
    ORDER BY oc.{price_col} DESC
    LIMIT 1
)"""
    
    def _build_expiry_selection(self, expiry_rule: str) -> str:
        """Build expiry selection logic"""
        
        expiry_mapping = {
            'CW': "oc.expiry_bucket = 'CW'",
            'NW': "oc.expiry_bucket = 'NW'",
            'CM': "oc.expiry_bucket = 'CM'",
            'NM': "oc.expiry_bucket = 'NM'"
        }
        
        return expiry_mapping.get(expiry_rule, "oc.expiry_bucket = 'CW'")
    
    def _build_main_union_query(self, num_legs: int) -> str:
        """Build main query to combine all leg selections"""
        
        if num_legs == 1:
            return "SELECT * FROM oi_leg_1_selection"
        
        parts = [f"SELECT * FROM oi_leg_{i}_selection" for i in range(1, num_legs + 1)]
        return "\nUNION ALL\n".join(parts)
    
    def build_oi_monitoring_query(
        self,
        strategy: Dict[str, Any],
        trade_date: date,
        monitoring_time: time,
        active_strikes: List[int]
    ) -> str:
        """
        Build query to monitor OI changes throughout the day
        
        Args:
            strategy: Strategy parameters
            trade_date: Trading date
            monitoring_time: Current monitoring time
            active_strikes: List of strikes currently being monitored
            
        Returns:
            SQL query for OI monitoring
        """
        if not active_strikes:
            return "-- No strikes to monitor"
        
        table_name = self._get_table_name(strategy['index'])
        strike_list = ','.join(map(str, active_strikes))
        
        query = f"""
        WITH current_oi AS (
            SELECT 
                strike,
                ce_oi,
                pe_oi,
                spot as underlying_price,
                trade_time
            FROM {table_name}
            WHERE trade_date = DATE '{trade_date}'
                AND trade_time = TIME '{monitoring_time}'
                AND strike IN ({strike_list})
                AND ce_oi > 0
                AND pe_oi > 0
        ),
        oi_rankings AS (
            SELECT 
                strike,
                ce_oi,
                pe_oi,
                underlying_price,
                RANK() OVER (ORDER BY ce_oi DESC) as ce_rank,
                RANK() OVER (ORDER BY pe_oi DESC) as pe_rank
            FROM current_oi
        )
        SELECT 
            strike,
            ce_oi,
            pe_oi,
            underlying_price,
            ce_rank,
            pe_rank
        FROM oi_rankings
        ORDER BY ce_rank, pe_rank
        """
        
        return query
    
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
                oi_col = 'ce_oi'
            else:
                price_col = 'pe_close'
                oi_col = 'pe_oi'
            
            query = f"""
            SELECT
                '{pos['position_id']}' AS position_id,
                '{pos['leg_id']}' AS leg_id,
                {price_col} AS exit_price,
                {oi_col} AS exit_oi,
                spot AS underlying_at_exit
            FROM {table_name}
            WHERE trade_date = DATE '{trade_date}'
                AND trade_time = TIME '{exit_time}'
                AND strike = {strike}
                AND expiry_date = DATE '{expiry}'
            """
            queries.append(query)
        
        return "\nUNION ALL\n".join(queries)
    
    def build_oi_rank_change_query(
        self,
        strategy: Dict[str, Any],
        trade_date: date,
        start_time: time,
        end_time: time,
        target_strikes: List[int]
    ) -> str:
        """
        Build query to detect OI rank changes throughout the day
        This is used for dynamic strike switching based on OI changes
        
        Args:
            strategy: Strategy parameters
            trade_date: Trading date
            start_time: Start monitoring time
            end_time: End monitoring time
            target_strikes: Strikes to monitor for rank changes
            
        Returns:
            SQL query for OI rank change detection
        """
        if not target_strikes:
            return "-- No strikes to monitor for rank changes"
        
        table_name = self._get_table_name(strategy['index'])
        strike_list = ','.join(map(str, target_strikes))
        
        query = f"""
        WITH time_series_oi AS (
            SELECT 
                trade_time,
                strike,
                ce_oi,
                pe_oi,
                RANK() OVER (PARTITION BY trade_time ORDER BY ce_oi DESC) as ce_rank,
                RANK() OVER (PARTITION BY trade_time ORDER BY pe_oi DESC) as pe_rank
            FROM {table_name}
            WHERE trade_date = DATE '{trade_date}'
                AND trade_time BETWEEN TIME '{start_time}' AND TIME '{end_time}'
                AND strike IN ({strike_list})
                AND ce_oi > 0
                AND pe_oi > 0
        ),
        rank_changes AS (
            SELECT 
                strike,
                trade_time,
                ce_rank,
                pe_rank,
                LAG(ce_rank) OVER (PARTITION BY strike ORDER BY trade_time) as prev_ce_rank,
                LAG(pe_rank) OVER (PARTITION BY strike ORDER BY trade_time) as prev_pe_rank
            FROM time_series_oi
        )
        SELECT 
            strike,
            trade_time,
            ce_rank,
            pe_rank,
            prev_ce_rank,
            prev_pe_rank,
            CASE 
                WHEN ce_rank != prev_ce_rank THEN 'CE_RANK_CHANGE'
                WHEN pe_rank != prev_pe_rank THEN 'PE_RANK_CHANGE'
                ELSE 'NO_CHANGE'
            END as change_type
        FROM rank_changes
        WHERE prev_ce_rank IS NOT NULL
            AND (ce_rank != prev_ce_rank OR pe_rank != prev_pe_rank)
        ORDER BY trade_time, strike
        """
        
        return query
    
    def build_oi_threshold_check_query(
        self,
        strategy: Dict[str, Any],
        trade_date: date,
        check_time: time,
        strikes_and_thresholds: List[Dict[str, Any]]
    ) -> str:
        """
        Build query to check if strikes meet OI thresholds
        
        Args:
            strategy: Strategy parameters
            trade_date: Trading date
            check_time: Time to check
            strikes_and_thresholds: List of {strike, threshold, instrument} dicts
            
        Returns:
            SQL query for threshold checking
        """
        if not strikes_and_thresholds:
            return "-- No thresholds to check"
        
        table_name = self._get_table_name(strategy['index'])
        
        # Build individual checks for each strike/threshold
        checks = []
        for item in strikes_and_thresholds:
            strike = item['strike']
            threshold = item['threshold']
            instrument = item['instrument']
            
            oi_col = 'ce_oi' if instrument == 'CE' else 'pe_oi'
            
            check = f"""
            SELECT 
                {strike} as strike,
                '{instrument}' as instrument,
                {oi_col} as current_oi,
                {threshold} as threshold,
                CASE WHEN {oi_col} >= {threshold} THEN 'PASS' ELSE 'FAIL' END as threshold_check
            FROM {table_name}
            WHERE trade_date = DATE '{trade_date}'
                AND trade_time = TIME '{check_time}'
                AND strike = {strike}
            """
            checks.append(check)
        
        return "\nUNION ALL\n".join(checks)
    
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