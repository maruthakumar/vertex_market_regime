#!/usr/bin/env python3
"""
HeavyDB Column Mapper
====================

Maps indicator calculations to correct HeavyDB column names
to fix column not found errors in queries.

Author: Claude Code
Date: 2025-07-02
"""

import logging

logger = logging.getLogger(__name__)

class HeavyDBColumnMapper:
    """Maps logical column names to actual HeavyDB schema"""
    
    def __init__(self):
        # Mapping from logical names to actual HeavyDB columns
        self.column_mapping = {
            # Option type classification
            'option_type': 'CASE WHEN strike <= spot THEN \'ITM\' ELSE \'OTM\' END',
            'ce_option_type': '\'CE\'',
            'pe_option_type': '\'PE\'',
            
            # Price columns
            'ce_ltp': 'ce_close',
            'pe_ltp': 'pe_close',
            'ce_price': 'ce_close',
            'pe_price': 'pe_close',
            
            # OI columns (already correct)
            'ce_oi': 'ce_oi',
            'pe_oi': 'pe_oi',
            
            # Greeks columns (already correct)
            'ce_delta': 'ce_delta',
            'pe_delta': 'pe_delta',
            'ce_gamma': 'ce_gamma',
            'pe_gamma': 'pe_gamma',
            'ce_theta': 'ce_theta',
            'pe_theta': 'pe_theta',
            'ce_vega': 'ce_vega',
            'pe_vega': 'pe_vega',
            
            # IV columns
            'ce_iv': 'ce_iv',
            'pe_iv': 'pe_iv',
            
            # Time columns
            'timestamp': 'CONCAT(trade_date, \' \', trade_time)',
            'trade_date': 'trade_date',
            'trade_time': 'trade_time',
            
            # Other columns
            'strike': 'strike',
            'spot': 'spot',
            'dte': 'dte'
        }
    
    def map_column(self, logical_name: str) -> str:
        """Map logical column name to actual HeavyDB column"""
        return self.column_mapping.get(logical_name, logical_name)
    
    def map_query(self, query: str) -> str:
        """Map all column references in a query"""
        mapped_query = query
        
        for logical, actual in self.column_mapping.items():
            # Replace column references (be careful with word boundaries)
            mapped_query = mapped_query.replace(f' {logical} ', f' {actual} ')
            mapped_query = mapped_query.replace(f' {logical},', f' {actual},')
            mapped_query = mapped_query.replace(f'({logical})', f'({actual})')
            mapped_query = mapped_query.replace(f' {logical}\n', f' {actual}\n')
        
        return mapped_query
    
    def get_straddle_query(self, atm_strike: float, trade_date: str = None) -> str:
        """Generate corrected straddle query"""
        date_filter = ""
        if trade_date:
            date_filter = f"AND trade_date = '{trade_date}'"
        else:
            date_filter = "AND trade_date = (SELECT MAX(trade_date) FROM nifty_option_chain)"
        
        return f"""
        SELECT 
            strike,
            'CE' as option_type,
            ce_close as price,
            ce_oi as oi,
            ce_iv as iv,
            ce_delta,
            ce_gamma,
            ce_theta,
            ce_vega
        FROM nifty_option_chain
        WHERE strike = {atm_strike}
        {date_filter}
        
        UNION ALL
        
        SELECT 
            strike,
            'PE' as option_type,
            pe_close as price,
            pe_oi as oi,
            pe_iv as iv,
            pe_delta,
            pe_gamma,
            pe_theta,
            pe_vega
        FROM nifty_option_chain
        WHERE strike = {atm_strike}
        {date_filter}
        """
    
    def get_oi_pa_query(self, lookback_periods: int = 20) -> str:
        """Generate corrected OI/PA trending query"""
        return f"""
        SELECT 
            trade_time,
            ce_oi,
            pe_oi,
            ce_close,
            pe_close,
            LAG(ce_oi, 1) OVER (ORDER BY trade_time) as prev_ce_oi,
            LAG(pe_oi, 1) OVER (ORDER BY trade_time) as prev_pe_oi,
            LAG(ce_close, 1) OVER (ORDER BY trade_time) as prev_ce_close,
            LAG(pe_close, 1) OVER (ORDER BY trade_time) as prev_pe_close
        FROM nifty_option_chain
        WHERE trade_date = (SELECT MAX(trade_date) FROM nifty_option_chain)
        AND ce_oi IS NOT NULL AND pe_oi IS NOT NULL
        ORDER BY trade_time DESC
        LIMIT {lookback_periods}
        """
    
    def get_greeks_query(self) -> str:
        """Generate corrected Greeks query"""
        return """
        SELECT 
            AVG(ce_delta) as avg_ce_delta,
            AVG(pe_delta) as avg_pe_delta,
            AVG(ce_gamma) as avg_ce_gamma,
            AVG(pe_gamma) as avg_pe_gamma,
            AVG(ce_theta) as avg_ce_theta,
            AVG(pe_theta) as avg_pe_theta,
            AVG(ce_vega) as avg_ce_vega,
            AVG(pe_vega) as avg_pe_vega
        FROM nifty_option_chain
        WHERE trade_date = (SELECT MAX(trade_date) FROM nifty_option_chain)
        AND ce_delta IS NOT NULL AND pe_delta IS NOT NULL
        """
    
    def get_correlation_query(self) -> str:
        """Generate corrected correlation query"""
        return """
        SELECT 
            ce_close,
            pe_close,
            ce_oi,
            pe_oi,
            ce_iv,
            pe_iv
        FROM nifty_option_chain
        WHERE trade_date = (SELECT MAX(trade_date) FROM nifty_option_chain)
        AND ce_close IS NOT NULL AND pe_close IS NOT NULL
        LIMIT 100
        """

# Global instance
column_mapper = HeavyDBColumnMapper()

def fix_query_columns(query: str) -> str:
    """Convenience function to fix column names in queries"""
    return column_mapper.map_query(query)