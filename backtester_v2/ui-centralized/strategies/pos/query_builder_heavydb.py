"""
Simplified Query Builder for HeavyDB - Optimized for GPU performance
Uses proven Iron Condor pattern to avoid complex nested queries
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import date, time


class SimplifiedPOSQueryBuilder:
    """Simplified query builder using HeavyDB-friendly patterns"""
    
    def __init__(self, table_name: str = "nifty_option_chain", 
                 enable_query_caching: bool = True, 
                 enable_optimizations: bool = True):
        self.table_name = table_name
        self.max_query_length = 3000  # Safe limit for HeavyDB
        self.lot_size = 50  # Standard lot size for NIFTY
        self.enable_query_caching = enable_query_caching
        self.enable_optimizations = enable_optimizations
        
    def build_position_query(self, strategy: Any) -> str:
        """Build position query using simplified pattern"""
        
        # Get active legs
        active_legs = self._get_active_legs(strategy)
        if not active_legs:
            raise ValueError("No active legs in strategy")
            
        # Calculate required strikes
        strike_conditions = self._build_strike_conditions(active_legs, strategy)
        
        # Build the query using Iron Condor pattern
        query = self._build_iron_condor_pattern_query(
            active_legs, 
            strike_conditions,
            strategy
        )
        
        # Validate query length
        if len(query) > self.max_query_length:
            # Fall back to modular approach
            return self._build_modular_query(active_legs, strategy)
            
        return query
    
    def _get_active_legs(self, strategy: Any) -> List[Any]:
        """Get active legs from strategy"""
        try:
            return strategy.get_active_legs()
        except AttributeError:
            # Fallback for simple models
            return getattr(strategy, 'legs', [])
    
    def _build_strike_conditions(self, legs: List[Any], strategy: Any) -> Dict[int, str]:
        """Build strike conditions for each leg"""
        conditions = {}
        
        for i, leg in enumerate(legs):
            # Get strike based on method (handle both attribute names)
            strike_method = getattr(leg, 'strike_method', None) or getattr(leg, 'strike_selection', 'ATM')
            strike_value = getattr(leg, 'strike_value', 0) or getattr(leg, 'strike_offset', 0)
            instrument = getattr(leg, 'instrument', None) or getattr(leg, 'option_type', 'CALL')
            
            if strike_method in ['ATM', 'ATM+0', 'ATM-0']:
                strike = "atm_strike"
            elif strike_method == 'OTM':
                if instrument in ['CALL', 'CE']:
                    strike = f"atm_strike + {strike_value}"
                else:
                    strike = f"atm_strike - {strike_value}"
            elif strike_method == 'ITM':
                if instrument in ['CALL', 'CE']:
                    strike = f"atm_strike - {strike_value}"
                else:
                    strike = f"atm_strike + {strike_value}"
            else:
                strike = "atm_strike"  # Default to ATM
                
            conditions[i] = strike
            
        return conditions
    
    def _build_iron_condor_pattern_query(self, legs: List[Any], 
                                        strike_conditions: Dict[int, str],
                                        strategy: Any) -> str:
        """Build query using Iron Condor pattern - simple and HeavyDB friendly"""
        
        # Build WHERE clause for strikes
        unique_strikes = list(set(strike_conditions.values()))
        strike_filter = f"strike IN ({', '.join(unique_strikes)})"
        
        # Get date range from strategy
        start_date = getattr(strategy.portfolio, 'start_date', '2024-01-01')
        end_date = getattr(strategy.portfolio, 'end_date', '2024-01-31')
        
        # Get expiry type (default to CM for more data)
        expiry_type = self._get_expiry_type(strategy)
        
        # Build leg type cases
        leg_type_cases = []
        for i, leg in enumerate(legs):
            strike = strike_conditions[i]
            leg_name = getattr(leg, 'leg_name', f'leg_{i}')
            leg_type_cases.append(f"WHEN strike = {strike} THEN '{leg_name}'")
        
        # Build the query
        query = f"""
        WITH leg_data AS (
            SELECT 
                trade_date,
                trade_time,
                spot,
                atm_strike,
                strike,
                expiry_bucket,
                ce_close,
                pe_close,
                ce_delta,
                pe_delta,
                ce_gamma,
                pe_gamma,
                ce_theta,
                pe_theta,
                ce_vega,
                pe_vega,
                ce_iv,
                pe_iv,
                ce_oi,
                pe_oi,
                ce_volume,
                pe_volume,
                -- Identify leg types
                CASE {' '.join(leg_type_cases)} ELSE NULL END as leg_type
            FROM {self.table_name}
            WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'
            AND expiry_bucket = '{expiry_type}'
            AND {strike_filter}
        ),
        position_summary AS (
            SELECT 
                trade_date,
                trade_time,
                spot,
                atm_strike,
                expiry_bucket,
                {self._build_leg_aggregations(legs)},
                {self._build_greek_aggregations(legs)},
                {self._build_total_calculations(legs)}
            FROM leg_data
            WHERE leg_type IS NOT NULL
            GROUP BY trade_date, trade_time, spot, atm_strike, expiry_bucket
        )
        SELECT *,
            {self._build_final_calculations(strategy)}
        FROM position_summary
        WHERE total_premium IS NOT NULL
        ORDER BY trade_date, trade_time
        """
        
        return query
    
    def _build_leg_aggregations(self, legs: List[Any]) -> str:
        """Build leg premium aggregations"""
        aggregations = []
        
        for i, leg in enumerate(legs):
            leg_name = getattr(leg, 'leg_name', f'leg_{i}')
            instrument = getattr(leg, 'instrument', 'CALL') or getattr(leg, 'option_type', 'CALL')
            transaction = getattr(leg, 'transaction', 'BUY') or getattr(leg, 'position_type', 'BUY')
            lots = getattr(leg, 'lots', 1)
            
            # Determine price column
            if instrument in ['CALL', 'CE']:
                price_col = 'ce_close'
            else:
                price_col = 'pe_close'
                
            # Determine position sign
            sign = '-' if transaction in ['SELL', 'SHORT'] else ''
            
            # Build aggregation
            agg = f"""
                SUM(CASE WHEN leg_type = '{leg_name}' 
                    THEN {sign}{price_col} * {lots} * {self.lot_size} 
                    ELSE 0 END) as {leg_name}_premium"""
            
            aggregations.append(agg)
            
        return ',\n                '.join(aggregations)
    
    def _build_greek_aggregations(self, legs: List[Any]) -> str:
        """Build Greek aggregations for all legs"""
        greeks = ['delta', 'gamma', 'theta', 'vega']
        aggregations = []
        
        for greek in greeks:
            greek_parts = []
            
            for i, leg in enumerate(legs):
                leg_name = getattr(leg, 'leg_name', f'leg_{i}')
                instrument = getattr(leg, 'instrument', 'CALL') or getattr(leg, 'option_type', 'CALL')
                transaction = getattr(leg, 'transaction', 'BUY') or getattr(leg, 'position_type', 'BUY')
                lots = getattr(leg, 'lots', 1)
                
                # Determine Greek column
                if instrument in ['CALL', 'CE']:
                    greek_col = f'ce_{greek}'
                else:
                    greek_col = f'pe_{greek}'
                    
                # Determine position sign
                sign = '-' if transaction in ['SELL', 'SHORT'] else ''
                
                # Build Greek part
                part = f"""CASE WHEN leg_type = '{leg_name}' 
                    THEN {sign}{greek_col} * {lots} * {self.lot_size} 
                    ELSE 0 END"""
                
                greek_parts.append(part)
            
            # Combine all parts for this Greek
            agg = f"SUM({' + '.join(greek_parts)}) as net_{greek}"
            aggregations.append(agg)
            
        return ',\n                '.join(aggregations)
    
    def _build_total_calculations(self, legs: List[Any]) -> str:
        """Build total premium calculation directly from source data"""
        premium_parts = []
        
        for i, leg in enumerate(legs):
            leg_name = getattr(leg, 'leg_name', f'leg_{i}')
            instrument = getattr(leg, 'instrument', 'CALL') or getattr(leg, 'option_type', 'CALL')
            transaction = getattr(leg, 'transaction', 'BUY') or getattr(leg, 'position_type', 'BUY')
            lots = getattr(leg, 'lots', 1)
            
            # Determine price column
            if instrument in ['CALL', 'CE']:
                price_col = 'ce_close'
            else:
                price_col = 'pe_close'
                
            # Determine position sign
            sign = '-' if transaction in ['SELL', 'SHORT'] else ''
            
            # Build premium calculation part
            part = f"""SUM(CASE WHEN leg_type = '{leg_name}' 
                THEN {sign}{price_col} * {lots} * {self.lot_size} 
                ELSE 0 END)"""
            
            premium_parts.append(part)
            
        return f"({' + '.join(premium_parts)}) as total_premium"
    
    def _build_final_calculations(self, strategy: Any) -> str:
        """Build final calculations like position type and breakevens"""
        calculations = []
        
        # Position type
        calculations.append("""
            CASE 
                WHEN total_premium > 0 THEN 'CREDIT'
                ELSE 'DEBIT'
            END as position_type""")
        
        # Breakevens
        calculations.append(f"""
            spot + (total_premium / {self.lot_size}) as upper_breakeven,
            spot - (total_premium / {self.lot_size}) as lower_breakeven""")
        
        # Delta status
        delta_threshold = 100  # Default threshold
        if hasattr(strategy, 'greek_limits') and strategy.greek_limits:
            delta_threshold = getattr(strategy.greek_limits, 'delta_neutral_threshold', 100)
            
        calculations.append(f"""
            CASE 
                WHEN ABS(net_delta) < {delta_threshold} THEN 'NEUTRAL'
                ELSE 'EXPOSED'
            END as delta_status""")
        
        return ',\n            '.join(calculations)
    
    def _get_expiry_type(self, strategy: Any) -> str:
        """Get expiry type from strategy"""
        expiry_mapping = {
            'CURRENT_WEEK': 'CW',
            'NEXT_WEEK': 'NW',
            'CURRENT_MONTH': 'CM',
            'NEXT_MONTH': 'NM'
        }
        
        # Try to get preferred expiry from strategy
        preferred_expiry = 'CURRENT_MONTH'  # Default
        
        if hasattr(strategy, 'strategy') and hasattr(strategy.strategy, 'preferred_expiry'):
            preferred_expiry = strategy.strategy.preferred_expiry
        elif hasattr(strategy, 'preferred_expiry'):
            preferred_expiry = strategy.preferred_expiry
            
        return expiry_mapping.get(str(preferred_expiry), 'CM')
    
    def _build_modular_query(self, legs: List[Any], strategy: Any) -> str:
        """Build modular query for complex strategies (fallback)"""
        # For now, just try to build a simpler version
        # In production, this would execute multiple smaller queries
        
        # Limit to first 4 legs to keep query simple
        limited_legs = legs[:4]
        strike_conditions = self._build_strike_conditions(limited_legs, strategy)
        
        return self._build_iron_condor_pattern_query(
            limited_legs,
            strike_conditions,
            strategy
        )
    
    def validate_query_length(self, query: str) -> bool:
        """Validate query is within HeavyDB limits"""
        return len(query) <= self.max_query_length
    
    def optimize_for_heavydb(self, query: str) -> str:
        """Apply HeavyDB-specific optimizations"""
        # Remove extra whitespace
        lines = [line.strip() for line in query.split('\n') if line.strip()]
        optimized = ' '.join(lines)
        
        # Remove comments
        optimized = ' '.join([part for part in optimized.split() if not part.startswith('--')])
        
        return optimized
    
    def build_complete_query(self, strategy: Any, query_type: str = "POSITION") -> str:
        """Build complete query based on query type (compatibility method)"""
        if query_type == "POSITION":
            return self.build_position_query(strategy)
        else:
            # For now, all query types use position query
            return self.build_position_query(strategy)