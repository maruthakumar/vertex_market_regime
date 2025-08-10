"""
Simplified Query Builder for POS Strategy
Builds HeavyDB queries for multi-leg option strategies
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import date, time

from .models_simple import SimplePOSStrategy, SimpleLegModel, SimplePortfolioModel

logger = logging.getLogger(__name__)


class SimplePOSQueryBuilder:
    """Simplified query builder for POS strategies"""
    
    def __init__(self, table_name: str = "nifty_option_chain"):
        self.table_name = table_name
    
    def build_position_query(self, strategy: SimplePOSStrategy) -> str:
        """Build query for multi-leg positions"""
        
        # Build expiry date mapping
        expiry_mapping = {
            'CURRENT_WEEK': 'cw',
            'NEXT_WEEK': 'nw', 
            'CURRENT_MONTH': 'cm',
            'NEXT_MONTH': 'nm'
        }
        
        # Build the main query with all legs
        leg_selections = []
        leg_conditions = []
        
        for leg in strategy.legs:
            expiry_col = expiry_mapping.get(leg.expiry_type, 'cw')
            
            # Build strike selection
            if leg.strike_selection == 'STRIKE_PRICE' and leg.strike_price:
                strike_condition = f"= {leg.strike_price}"
            elif leg.strike_selection == 'ATM':
                strike_condition = f"= ROUND(spot_price / 50) * 50 + {leg.strike_offset}"
            elif leg.strike_selection == 'OTM':
                if leg.option_type == 'CE':
                    strike_condition = f"= ROUND(spot_price / 50) * 50 + {abs(leg.strike_offset)}"
                else:  # PE
                    strike_condition = f"= ROUND(spot_price / 50) * 50 - {abs(leg.strike_offset)}"
            elif leg.strike_selection == 'ITM':
                if leg.option_type == 'CE':
                    strike_condition = f"= ROUND(spot_price / 50) * 50 - {abs(leg.strike_offset)}"
                else:  # PE
                    strike_condition = f"= ROUND(spot_price / 50) * 50 + {abs(leg.strike_offset)}"
            else:
                strike_condition = f"= ROUND(spot_price / 50) * 50"
            
            # Add selections for this leg
            leg_selections.extend([
                f"MAX(CASE WHEN option_type = '{leg.option_type}' AND expiry_date = {expiry_col}_expiry AND strike_price {strike_condition} THEN strike_price END) as {leg.leg_name}_strike",
                f"MAX(CASE WHEN option_type = '{leg.option_type}' AND expiry_date = {expiry_col}_expiry AND strike_price {strike_condition} THEN close_price END) as {leg.leg_name}_price",
                f"MAX(CASE WHEN option_type = '{leg.option_type}' AND expiry_date = {expiry_col}_expiry AND strike_price {strike_condition} THEN delta END) as {leg.leg_name}_delta",
                f"MAX(CASE WHEN option_type = '{leg.option_type}' AND expiry_date = {expiry_col}_expiry AND strike_price {strike_condition} THEN gamma END) as {leg.leg_name}_gamma",
                f"MAX(CASE WHEN option_type = '{leg.option_type}' AND expiry_date = {expiry_col}_expiry AND strike_price {strike_condition} THEN theta END) as {leg.leg_name}_theta",
                f"MAX(CASE WHEN option_type = '{leg.option_type}' AND expiry_date = {expiry_col}_expiry AND strike_price {strike_condition} THEN vega END) as {leg.leg_name}_vega",
                f"MAX(CASE WHEN option_type = '{leg.option_type}' AND expiry_date = {expiry_col}_expiry AND strike_price {strike_condition} THEN volume END) as {leg.leg_name}_volume",
                f"MAX(CASE WHEN option_type = '{leg.option_type}' AND expiry_date = {expiry_col}_expiry AND strike_price {strike_condition} THEN open_interest END) as {leg.leg_name}_oi"
            ])
        
        # Build the complete query
        query = f"""
        WITH daily_data AS (
            SELECT 
                trade_date,
                trade_time,
                -- Get spot price (from XX records or underlying_value)
                MAX(CASE WHEN option_type = 'XX' THEN close_price 
                    ELSE underlying_value END) as spot_price,
                -- Get expiry dates
                MAX(CASE WHEN option_type = 'CE' AND expiry_type = 'cw' THEN expiry_date END) as cw_expiry,
                MAX(CASE WHEN option_type = 'CE' AND expiry_type = 'nw' THEN expiry_date END) as nw_expiry,
                MAX(CASE WHEN option_type = 'CE' AND expiry_type = 'cm' THEN expiry_date END) as cm_expiry,
                MAX(CASE WHEN option_type = 'CE' AND expiry_type = 'nm' THEN expiry_date END) as nm_expiry,
                -- Leg data
                {','.join(leg_selections)}
            FROM {self.table_name}
            WHERE trade_date BETWEEN '{strategy.portfolio.start_date}' AND '{strategy.portfolio.end_date}'
            AND trade_time = '09:20:00'
            GROUP BY trade_date, trade_time
        )
        SELECT 
            trade_date,
            trade_time,
            spot_price,
            cw_expiry,
            nw_expiry,
            cm_expiry,
            nm_expiry,
            {','.join([f"{leg.leg_name}_strike, {leg.leg_name}_price, {leg.leg_name}_delta, {leg.leg_name}_gamma, {leg.leg_name}_theta, {leg.leg_name}_vega, {leg.leg_name}_volume, {leg.leg_name}_oi" for leg in strategy.legs])}
        FROM daily_data
        WHERE spot_price IS NOT NULL
        """
        
        # Add leg filters to ensure all legs have data
        leg_filters = []
        for leg in strategy.legs:
            leg_filters.append(f"AND {leg.leg_name}_price IS NOT NULL")
        
        query += '\n'.join(leg_filters)
        query += "\nORDER BY trade_date, trade_time"
        
        return self._clean_query(query)
    
    def build_simple_test_query(self, strategy: SimplePOSStrategy) -> str:
        """Build a simple test query to verify data availability"""
        
        query = f"""
        SELECT 
            trade_date,
            option_type,
            expiry_type,
            COUNT(DISTINCT strike_price) as strike_count,
            MIN(strike_price) as min_strike,
            MAX(strike_price) as max_strike,
            MAX(underlying_value) as spot_price
        FROM {self.table_name}
        WHERE trade_date BETWEEN '{strategy.portfolio.start_date}' AND '{strategy.portfolio.end_date}'
        AND trade_time = '09:20:00'
        GROUP BY trade_date, option_type, expiry_type
        ORDER BY trade_date, option_type, expiry_type
        LIMIT 100
        """
        
        return self._clean_query(query)
    
    def build_strike_availability_query(self, date: str) -> str:
        """Check available strikes for a specific date"""
        
        query = f"""
        SELECT 
            trade_date,
            option_type,
            expiry_type,
            expiry_date,
            strike_price,
            close_price,
            volume,
            open_interest,
            delta,
            underlying_value
        FROM {self.table_name}
        WHERE trade_date = '{date}'
        AND trade_time = '09:20:00'
        AND option_type IN ('CE', 'PE')
        ORDER BY option_type, expiry_type, strike_price
        """
        
        return self._clean_query(query)
    
    def _clean_query(self, query: str) -> str:
        """Clean up query formatting"""
        # Remove extra whitespace
        lines = [line.strip() for line in query.strip().split('\n') if line.strip()]
        return '\n'.join(lines)
    
    def build_single_leg_query(self, leg: SimpleLegModel, portfolio: SimplePortfolioModel) -> str:
        """Build query for a single leg (for testing)"""
        
        # Map expiry type
        expiry_mapping = {
            'CURRENT_WEEK': 'cw',
            'NEXT_WEEK': 'nw',
            'CURRENT_MONTH': 'cm',
            'NEXT_MONTH': 'nm'
        }
        expiry_type = expiry_mapping.get(leg.expiry_type, 'cw')
        
        query = f"""
        SELECT 
            trade_date,
            trade_time,
            underlying_value as spot_price,
            expiry_date,
            strike_price,
            option_type,
            close_price,
            volume,
            open_interest,
            delta,
            gamma,
            theta,
            vega
        FROM {self.table_name}
        WHERE trade_date BETWEEN '{portfolio.start_date}' AND '{portfolio.end_date}'
        AND trade_time = '09:20:00'
        AND option_type = '{leg.option_type}'
        AND expiry_type = '{expiry_type}'
        """
        
        # Add strike filter
        if leg.strike_selection == 'STRIKE_PRICE' and leg.strike_price:
            query += f"\nAND strike_price = {leg.strike_price}"
        
        query += "\nORDER BY trade_date, trade_time"
        
        return self._clean_query(query)