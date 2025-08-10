#!/usr/bin/env python3
"""
OI Analyzer - Analyzes open interest patterns for strike selection
Core logic for MAXOI and MAXCOI calculations
"""

from typing import Dict, List, Tuple, Optional
from datetime import date, time, datetime, timedelta
import logging

from .models import OIRanking, COIBasedOn

logger = logging.getLogger(__name__)


class OIAnalyzer:
    """Analyzes open interest patterns for strike selection"""
    
    def __init__(self, db_connection):
        self.conn = db_connection
        self._ranking_cache: Dict[str, OIRanking] = {}
    
    def calculate_oi_rankings(
        self,
        index: str,
        trade_date: date,
        analysis_time: time,
        strike_count: int = 5,
        underlying_type: str = 'SPOT'
    ) -> Optional[OIRanking]:
        """
        Calculate OI rankings for both CE and PE options
        
        Args:
            index: Index name (NIFTY, BANKNIFTY, etc.)
            trade_date: Date to analyze
            analysis_time: Time to analyze OI
            strike_count: Number of strikes each side of ATM to analyze
            underlying_type: SPOT or FUT
            
        Returns:
            OIRanking object with sorted rankings
        """
        # Check cache first
        cache_key = f"{index}_{trade_date}_{analysis_time}_{strike_count}_{underlying_type}"
        if cache_key in self._ranking_cache:
            logger.debug(f"Using cached OI ranking for {cache_key}")
            return self._ranking_cache[cache_key]
        
        table_name = self._get_table_name(index)
        
        # Build query to get OI data around ATM
        query = f"""
        WITH atm_data AS (
            SELECT atm_strike, strike_step
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
                atm.atm_strike,
                atm.strike_step,
                ABS(oc.strike - atm.atm_strike) / atm.strike_step as strike_distance
            FROM {table_name} oc
            CROSS JOIN atm_data atm
            WHERE oc.trade_date = DATE '{trade_date}'
                AND oc.trade_time = TIME '{analysis_time}'
                AND ABS(oc.strike - atm.atm_strike) <= {strike_count} * atm.strike_step
                AND oc.ce_oi > 0 
                AND oc.pe_oi > 0
                AND oc.ce_symbol IS NOT NULL
                AND oc.pe_symbol IS NOT NULL
        )
        SELECT 
            strike,
            ce_oi,
            pe_oi,
            atm_strike,
            strike_distance
        FROM oi_analysis
        ORDER BY strike_distance, strike
        """
        
        try:
            cursor = self.conn.execute(query)
            results = cursor.fetchall()
            
            if not results:
                logger.warning(f"No OI data found for {index} on {trade_date} at {analysis_time}")
                return None
            
            ce_rankings = []
            pe_rankings = []
            
            for row in results:
                strike, ce_oi, pe_oi, atm_strike, strike_distance = row
                ce_rankings.append((int(strike), float(ce_oi)))
                pe_rankings.append((int(strike), float(pe_oi)))
            
            # Sort by OI value (highest first)
            ce_rankings.sort(key=lambda x: x[1], reverse=True)
            pe_rankings.sort(key=lambda x: x[1], reverse=True)
            
            ranking = OIRanking(
                trade_date=trade_date,
                analysis_time=analysis_time,
                ce_rankings=ce_rankings,
                pe_rankings=pe_rankings
            )
            
            # Cache the result
            self._ranking_cache[cache_key] = ranking
            
            logger.info(f"OI Rankings calculated for {index} on {trade_date} at {analysis_time}")
            logger.info(f"Top 3 CE strikes by OI: {ce_rankings[:3]}")
            logger.info(f"Top 3 PE strikes by OI: {pe_rankings[:3]}")
            
            return ranking
            
        except Exception as e:
            logger.error(f"Error calculating OI rankings: {e}")
            return None
    
    def calculate_coi_rankings(
        self,
        index: str,
        trade_date: date,
        analysis_time: time,
        coi_based_on: COIBasedOn = COIBasedOn.YESTERDAY_CLOSE,
        strike_count: int = 5,
        underlying_type: str = 'SPOT'
    ) -> Optional[OIRanking]:
        """
        Calculate Change in OI (COI) rankings
        
        Args:
            index: Index name
            trade_date: Date to analyze
            analysis_time: Time to analyze COI
            coi_based_on: Base for COI calculation
            strike_count: Number of strikes to analyze
            underlying_type: SPOT or FUT
            
        Returns:
            OIRanking object with COI-based rankings
        """
        table_name = self._get_table_name(index)
        
        if coi_based_on == COIBasedOn.YESTERDAY_CLOSE:
            # Compare with previous trading day's close
            reference_query = f"""
            WITH atm_data AS (
                SELECT atm_strike, strike_step
                FROM {table_name}
                WHERE trade_date = DATE '{trade_date}'
                    AND trade_time = TIME '{analysis_time}'
                LIMIT 1
            ),
            yesterday_data AS (
                SELECT 
                    oc.strike,
                    oc.ce_oi as prev_ce_oi,
                    oc.pe_oi as prev_pe_oi
                FROM {table_name} oc
                CROSS JOIN atm_data atm
                WHERE oc.trade_date = DATE '{trade_date}' - INTERVAL '1' DAY
                    AND oc.trade_time = (
                        SELECT MAX(trade_time) 
                        FROM {table_name} 
                        WHERE trade_date = DATE '{trade_date}' - INTERVAL '1' DAY
                    )
                    AND ABS(oc.strike - atm.atm_strike) <= {strike_count} * atm.strike_step
                    AND oc.ce_oi > 0 
                    AND oc.pe_oi > 0
            ),
            current_data AS (
                SELECT 
                    oc.strike,
                    oc.ce_oi as curr_ce_oi,
                    oc.pe_oi as curr_pe_oi
                FROM {table_name} oc
                CROSS JOIN atm_data atm
                WHERE oc.trade_date = DATE '{trade_date}'
                    AND oc.trade_time = TIME '{analysis_time}'
                    AND ABS(oc.strike - atm.atm_strike) <= {strike_count} * atm.strike_step
                    AND oc.ce_oi > 0 
                    AND oc.pe_oi > 0
            )
            SELECT 
                cd.strike,
                cd.curr_ce_oi - COALESCE(yd.prev_ce_oi, 0) as ce_coi,
                cd.curr_pe_oi - COALESCE(yd.prev_pe_oi, 0) as pe_coi,
                cd.curr_ce_oi,
                cd.curr_pe_oi
            FROM current_data cd
            LEFT JOIN yesterday_data yd ON cd.strike = yd.strike
            ORDER BY cd.strike
            """
        else:
            # Compare with previous timestamp (PREVIOUS_TIMESTAMP)
            reference_query = f"""
            WITH atm_data AS (
                SELECT atm_strike, strike_step
                FROM {table_name}
                WHERE trade_date = DATE '{trade_date}'
                    AND trade_time = TIME '{analysis_time}'
                LIMIT 1
            ),
            prev_timestamp AS (
                SELECT MAX(trade_time) as prev_time
                FROM {table_name}
                WHERE trade_date = DATE '{trade_date}'
                    AND trade_time < TIME '{analysis_time}'
            ),
            previous_data AS (
                SELECT 
                    oc.strike,
                    oc.ce_oi as prev_ce_oi,
                    oc.pe_oi as prev_pe_oi
                FROM {table_name} oc
                CROSS JOIN atm_data atm
                CROSS JOIN prev_timestamp pt
                WHERE oc.trade_date = DATE '{trade_date}'
                    AND oc.trade_time = pt.prev_time
                    AND ABS(oc.strike - atm.atm_strike) <= {strike_count} * atm.strike_step
                    AND oc.ce_oi > 0 
                    AND oc.pe_oi > 0
            ),
            current_data AS (
                SELECT 
                    oc.strike,
                    oc.ce_oi as curr_ce_oi,
                    oc.pe_oi as curr_pe_oi
                FROM {table_name} oc
                CROSS JOIN atm_data atm
                WHERE oc.trade_date = DATE '{trade_date}'
                    AND oc.trade_time = TIME '{analysis_time}'
                    AND ABS(oc.strike - atm.atm_strike) <= {strike_count} * atm.strike_step
                    AND oc.ce_oi > 0 
                    AND oc.pe_oi > 0
            )
            SELECT 
                cd.strike,
                cd.curr_ce_oi - COALESCE(pd.prev_ce_oi, cd.curr_ce_oi) as ce_coi,
                cd.curr_pe_oi - COALESCE(pd.prev_pe_oi, cd.curr_pe_oi) as pe_coi,
                cd.curr_ce_oi,
                cd.curr_pe_oi
            FROM current_data cd
            LEFT JOIN previous_data pd ON cd.strike = pd.strike
            ORDER BY cd.strike
            """
        
        try:
            cursor = self.conn.execute(reference_query)
            results = cursor.fetchall()
            
            if not results:
                logger.warning(f"No COI data found for {index} on {trade_date} at {analysis_time}")
                return None
            
            ce_coi_rankings = []
            pe_coi_rankings = []
            
            for row in results:
                strike, ce_coi, pe_coi, curr_ce_oi, curr_pe_oi = row
                # Use absolute COI values for ranking (biggest changes)
                ce_coi_rankings.append((int(strike), float(abs(ce_coi))))
                pe_coi_rankings.append((int(strike), float(abs(pe_coi))))
            
            # Sort by COI value (highest change first)
            ce_coi_rankings.sort(key=lambda x: x[1], reverse=True)
            pe_coi_rankings.sort(key=lambda x: x[1], reverse=True)
            
            ranking = OIRanking(
                trade_date=trade_date,
                analysis_time=analysis_time,
                ce_rankings=ce_coi_rankings,
                pe_rankings=pe_coi_rankings,
                coi_rankings=True
            )
            
            logger.info(f"COI Rankings calculated for {index} on {trade_date} at {analysis_time}")
            logger.info(f"Top 3 CE strikes by COI: {ce_coi_rankings[:3]}")
            logger.info(f"Top 3 PE strikes by COI: {pe_coi_rankings[:3]}")
            
            return ranking
            
        except Exception as e:
            logger.error(f"Error calculating COI rankings: {e}")
            return None
    
    def get_oi_value_for_strike(
        self,
        index: str,
        trade_date: date,
        analysis_time: time,
        strike: int,
        instrument_type: str,
        underlying_type: str = 'SPOT'
    ) -> Optional[float]:
        """
        Get OI value for a specific strike and instrument type
        
        Args:
            index: Index name
            trade_date: Date to check
            analysis_time: Time to check
            strike: Strike price
            instrument_type: 'CE' or 'PE'
            underlying_type: SPOT or FUT
            
        Returns:
            OI value or None if not found
        """
        table_name = self._get_table_name(index)
        oi_column = 'ce_oi' if instrument_type == 'CE' else 'pe_oi'
        
        query = f"""
        SELECT {oi_column}
        FROM {table_name}
        WHERE trade_date = DATE '{trade_date}'
            AND trade_time = TIME '{analysis_time}'
            AND strike = {strike}
            AND {oi_column} > 0
        LIMIT 1
        """
        
        try:
            cursor = self.conn.execute(query)
            result = cursor.fetchone()
            
            if result and result[0] is not None:
                return float(result[0])
            
        except Exception as e:
            logger.error(f"Error getting OI value: {e}")
        
        return None
    
    def validate_oi_threshold(
        self,
        ranking: OIRanking,
        rank: int,
        instrument_type: str,
        threshold: int
    ) -> bool:
        """
        Validate if OI value at given rank meets threshold
        
        Args:
            ranking: OIRanking object
            rank: 1-based rank to check
            instrument_type: 'CE' or 'PE'
            threshold: Minimum OI threshold
            
        Returns:
            True if threshold is met
        """
        oi_value = ranking.get_oi_for_rank(rank, instrument_type)
        
        if oi_value is None:
            return False
        
        meets_threshold = oi_value >= threshold
        
        if not meets_threshold:
            logger.warning(f"OI threshold not met: {oi_value} < {threshold} for rank {rank} {instrument_type}")
        
        return meets_threshold
    
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
    
    def clear_cache(self):
        """Clear the ranking cache"""
        self._ranking_cache.clear()
        logger.info("OI ranking cache cleared")