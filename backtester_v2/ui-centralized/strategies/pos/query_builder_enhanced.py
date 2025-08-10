"""
Enhanced Query Builder for POS Strategy with all advanced features
Now uses simplified HeavyDB-friendly query patterns for better performance
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import date, time, datetime, timedelta
import json
import sys
from pathlib import Path

# Add path to import simplified query builder
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

try:
    from .query_builder_heavydb import SimplifiedPOSQueryBuilder
    USE_SIMPLIFIED_BUILDER = True
except ImportError:
    try:
        from query_builder_heavydb import SimplifiedPOSQueryBuilder
        USE_SIMPLIFIED_BUILDER = True
    except ImportError:
        USE_SIMPLIFIED_BUILDER = False

try:
    from .models_enhanced import (
        CompletePOSStrategy, EnhancedLegModel, AdjustmentRule,
        StrikeMethod, InstrumentType, TransactionType,
        ExpiryType, MarketRegime, AdjustmentTrigger
    )
except ImportError:
    # Fallback to simple models if enhanced models have dependency issues
    from .models_simple import (
        SimplePOSStrategy as CompletePOSStrategy,
        SimpleLegModel as EnhancedLegModel
    )
    # Define minimal enums for compatibility
    class StrikeMethod:
        ATM = "ATM"
        OTM = "OTM"
        ITM = "ITM"
        FIXED = "FIXED"
        DELTA = "DELTA"
        PREMIUM = "PREMIUM"
        BE_OPTIMIZED = "BE_OPTIMIZED"
    
    class InstrumentType:
        CALL = "CE"
        PUT = "PE"
    
    class TransactionType:
        BUY = "BUY"
        SELL = "SELL"
    
    class ExpiryType:
        CURRENT_WEEK = "CURRENT_WEEK"
        NEXT_WEEK = "NEXT_WEEK"
        CURRENT_MONTH = "CURRENT_MONTH"
        NEXT_MONTH = "NEXT_MONTH"


class EnhancedPOSQueryBuilder:
    """Build complex HeavyDB queries for POS strategy with all features"""
    
    def __init__(self, enable_query_caching: bool = True, enable_optimizations: bool = True):
        self.table_name = "nifty_option_chain"
        self.enable_query_caching = enable_query_caching
        self.enable_optimizations = enable_optimizations
        self.query_cache = {}
        
        # Map expiry types to expiry_bucket values
        self.expiry_mapping = {
            ExpiryType.CURRENT_WEEK: 'CW',
            ExpiryType.NEXT_WEEK: 'NW',
            ExpiryType.CURRENT_MONTH: 'CM',
            ExpiryType.NEXT_MONTH: 'NM',
            'CURRENT_WEEK': 'CW',
            'NEXT_WEEK': 'NW',
            'CURRENT_MONTH': 'CM',
            'NEXT_MONTH': 'NM'
        }
        
        # Performance optimization settings
        self.optimization_config = {
            'use_indexed_columns': True,
            'limit_strike_range': True,
            'batch_processing': True,
            'parallel_leg_processing': False,  # Disabled for HeavyDB compatibility
            'memory_efficient_queries': True
        }
    
    def get_cached_query(self, cache_key: str) -> Optional[str]:
        """Get cached query if available"""
        if self.enable_query_caching:
            return self.query_cache.get(cache_key)
        return None
    
    def cache_query(self, cache_key: str, query: str) -> None:
        """Cache query for reuse"""
        if self.enable_query_caching:
            self.query_cache[cache_key] = query
    
    def optimize_query_structure(self, query: str) -> str:
        """Apply HeavyDB-specific optimizations"""
        if not self.enable_optimizations:
            return query
        
        # Remove unnecessary whitespace and comments
        lines = [line.strip() for line in query.split('\n') if line.strip() and not line.strip().startswith('--')]
        optimized = ' '.join(lines)
        
        # HeavyDB optimizations (no SQL hints as they're not supported)
        # Ensure date filters come first in WHERE clause for better performance
        # HeavyDB automatically optimizes based on column usage patterns
        
        return optimized
    
    def build_complete_query(self, strategy: CompletePOSStrategy, 
                           query_type: str = "POSITION") -> str:
        """Build complete query based on query type"""
        
        # Generate cache key
        cache_key = f"{query_type}_{hash(str(strategy))}"
        
        # Check cache first
        cached_query = self.get_cached_query(cache_key)
        if cached_query:
            return cached_query
        
        # Build query based on type
        if query_type == "POSITION":
            query = self.build_position_query(strategy)
        elif query_type == "MARKET_STRUCTURE":
            query = self.build_market_structure_query(strategy)
        elif query_type == "VOLATILITY_ANALYSIS":
            query = self.build_volatility_analysis_query(strategy)
        elif query_type == "GREEK_AGGREGATION":
            query = self.build_greek_aggregation_query(strategy)
        elif query_type == "ADJUSTMENT_SCAN":
            query = self.build_adjustment_scan_query(strategy)
        elif query_type == "BREAKEVEN_ANALYSIS":
            query = self.build_breakeven_analysis_query(strategy)
        else:
            query = self.build_position_query(strategy)
        
        # Apply optimizations
        optimized_query = self.optimize_query_structure(query)
        
        # Cache the optimized query
        self.cache_query(cache_key, optimized_query)
        
        return optimized_query
    
    def build_position_query(self, strategy: CompletePOSStrategy) -> str:
        """Build main position query with all enhancements"""
        
        # Use simplified builder if available and for position queries
        if USE_SIMPLIFIED_BUILDER:
            try:
                print("DEBUG: Using SimplifiedPOSQueryBuilder")
                simplified_builder = SimplifiedPOSQueryBuilder(table_name=self.table_name)
                query = simplified_builder.build_position_query(strategy)
                print(f"DEBUG: Simplified query length: {len(query)}")
                return query
            except Exception as e:
                # Fall back to complex builder if simplified fails
                print(f"Simplified builder failed, using complex builder: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"DEBUG: USE_SIMPLIFIED_BUILDER is {USE_SIMPLIFIED_BUILDER}")
        
        # Original complex query builder logic (as fallback)
        # Get legs with fallback for simple models
        try:
            active_legs = strategy.get_active_legs()
        except AttributeError:
            # Fallback for simple models
            active_legs = getattr(strategy, 'legs', [])
        
        if not active_legs:
            raise ValueError("No active legs in strategy")
        
        # Build complex multi-leg query
        query_parts = []
        
        # Base query with all required fields
        base_fields = """
            trade_date,
            trade_time,
            spot,
            atm_strike,
            future_close,
            expiry_date,
            expiry_bucket
        """
        
        # Add leg-specific calculations
        for i, leg in enumerate(active_legs):
            leg_alias = f"leg{i}"
            
            # Strike selection based on method
            strike_condition = self._build_strike_condition(leg, strategy)
            
            # Option type column selection (compatible with both models)
            instrument_type = getattr(leg, 'instrument', None) or getattr(leg, 'option_type', None)
            if instrument_type in [InstrumentType.CALL, 'CE', 'CALL']:
                price_col = "ce_close"
                delta_col = "ce_delta"
                gamma_col = "ce_gamma"
                theta_col = "ce_theta"
                vega_col = "ce_vega"
                iv_col = "ce_iv"
                oi_col = "ce_oi"
                volume_col = "ce_volume"
            else:
                price_col = "pe_close"
                delta_col = "pe_delta"
                gamma_col = "pe_gamma"
                theta_col = "pe_theta"
                vega_col = "pe_vega"
                iv_col = "pe_iv"
                oi_col = "pe_oi"
                volume_col = "pe_volume"
            
            # Position sign based on transaction type (compatible with both models)
            transaction_type = getattr(leg, 'transaction', None) or getattr(leg, 'position_type', None)
            position_sign = -1 if transaction_type in [TransactionType.SELL, 'SELL'] else 1
            
            # Get lots (compatible with both models)
            lots = getattr(leg, 'lots', 1)
            
            # Leg fields with dynamic sizing
            leg_fields = f"""
                MAX(CASE WHEN {strike_condition} THEN strike END) as {leg_alias}_strike,
                MAX(CASE WHEN {strike_condition} THEN {price_col} END) as {leg_alias}_price,
                MAX(CASE WHEN {strike_condition} THEN {price_col} * {position_sign} * {lots} * 50 END) as {leg_alias}_premium,
                MAX(CASE WHEN {strike_condition} THEN {delta_col} * {position_sign} * {lots} * 50 END) as {leg_alias}_delta,
                MAX(CASE WHEN {strike_condition} THEN {gamma_col} * {position_sign} * {lots} * 50 END) as {leg_alias}_gamma,
                MAX(CASE WHEN {strike_condition} THEN {theta_col} * {position_sign} * {lots} * 50 END) as {leg_alias}_theta,
                MAX(CASE WHEN {strike_condition} THEN {vega_col} * {position_sign} * {lots} * 50 END) as {leg_alias}_vega,
                MAX(CASE WHEN {strike_condition} THEN {iv_col} END) as {leg_alias}_iv,
                MAX(CASE WHEN {strike_condition} THEN {oi_col} END) as {leg_alias}_oi,
                MAX(CASE WHEN {strike_condition} THEN {volume_col} END) as {leg_alias}_volume
            """
            
            query_parts.append(leg_fields)
        
        # Add aggregate calculations
        aggregate_fields = self._build_aggregate_fields(active_legs)
        query_parts.append(aggregate_fields)
        
        # Add market condition fields if needed
        if strategy.market_structure and strategy.market_structure.enabled:
            market_fields = self._build_market_condition_fields()
            query_parts.append(market_fields)
        
        # Build WHERE clause with all filters
        where_conditions = self._build_where_conditions(strategy)
        
        # Construct final query
        query = f"""
        WITH position_data AS (
            SELECT 
                {base_fields},
                {','.join(query_parts)}
            FROM {self.table_name}
            WHERE {where_conditions}
            GROUP BY trade_date, trade_time, spot, atm_strike, future_close, expiry_date, expiry_bucket
        )
        SELECT *,
            -- Additional calculations
            total_premium as net_credit,
            CASE 
                WHEN total_premium > 0 THEN 'CREDIT'
                ELSE 'DEBIT'
            END as position_type,
            -- Breakeven calculations
            {self._build_breakeven_calculations(strategy)}
        FROM position_data
        WHERE total_premium IS NOT NULL
        ORDER BY trade_date, trade_time
        """
        
        return query
    
    def build_market_structure_query(self, strategy: CompletePOSStrategy) -> str:
        """Build query for market structure analysis"""
        
        if not strategy.market_structure or not strategy.market_structure.enabled:
            return ""
        
        config = strategy.market_structure
        
        query = f"""
        WITH market_data AS (
            SELECT 
                trade_date,
                trade_time,
                spot,
                futures_price,
                atm_strike,
                -- Volume analysis
                SUM(ce_volume + pe_volume) as total_volume,
                SUM(ce_oi + pe_oi) as total_oi,
                -- Put-Call ratio
                SUM(pe_volume) / NULLIF(SUM(ce_volume), 0) as pcr_volume,
                SUM(pe_oi) / NULLIF(SUM(ce_oi), 0) as pcr_oi,
                -- IV analysis
                AVG(ce_iv) as avg_call_iv,
                AVG(pe_iv) as avg_put_iv,
                AVG(pe_iv) - AVG(ce_iv) as put_call_skew,
                -- Market breadth (strikes above/below spot)
                SUM(CASE WHEN strike > spot AND ce_volume > 0 THEN 1 ELSE 0 END) as calls_above_spot,
                SUM(CASE WHEN strike < spot AND pe_volume > 0 THEN 1 ELSE 0 END) as puts_below_spot
            FROM {self.table_name}
            WHERE trade_date BETWEEN '{strategy.portfolio.start_date}' AND '{strategy.portfolio.end_date}'
            AND expiry_bucket = 'CW'  -- Current week for analysis
            GROUP BY trade_date, trade_time, spot, futures_price, atm_strike
        ),
        trend_analysis AS (
            SELECT 
                *,
                -- Moving averages
                AVG(spot) OVER (ORDER BY trade_date, trade_time ROWS BETWEEN {config.ma_periods[0]} PRECEDING AND CURRENT ROW) as ma_short,
                AVG(spot) OVER (ORDER BY trade_date, trade_time ROWS BETWEEN {config.ma_periods[1]} PRECEDING AND CURRENT ROW) as ma_medium,
                AVG(spot) OVER (ORDER BY trade_date, trade_time ROWS BETWEEN {config.ma_periods[2]} PRECEDING AND CURRENT ROW) as ma_long,
                -- Volume moving average
                AVG(total_volume) OVER (ORDER BY trade_date, trade_time ROWS BETWEEN {config.volume_ma_period} PRECEDING AND CURRENT ROW) as volume_ma,
                -- Volatility
                STDDEV(spot) OVER (ORDER BY trade_date, trade_time ROWS BETWEEN {config.volatility_lookback} PRECEDING AND CURRENT ROW) as realized_vol
            FROM market_data
        )
        SELECT *,
            -- Trend strength
            CASE 
                WHEN spot > ma_short AND ma_short > ma_medium AND ma_medium > ma_long THEN 'STRONG_UP'
                WHEN spot < ma_short AND ma_short < ma_medium AND ma_medium < ma_long THEN 'STRONG_DOWN'
                WHEN spot > ma_medium THEN 'WEAK_UP'
                WHEN spot < ma_medium THEN 'WEAK_DOWN'
                ELSE 'NEUTRAL'
            END as trend_state,
            -- Volume analysis
            CASE 
                WHEN total_volume > volume_ma * {config.unusual_volume_threshold} THEN 'HIGH_VOLUME'
                WHEN total_volume < volume_ma * 0.5 THEN 'LOW_VOLUME'
                ELSE 'NORMAL_VOLUME'
            END as volume_state,
            -- Market regime
            CASE
                WHEN realized_vol > {config.regime_change_threshold} THEN 'HIGH_VOL'
                ELSE 'NORMAL_VOL'
            END as volatility_regime
        FROM trend_analysis
        ORDER BY trade_date, trade_time
        """
        
        return query
    
    def build_volatility_analysis_query(self, strategy: CompletePOSStrategy) -> str:
        """Build query for volatility analysis"""
        
        vol_filter = strategy.strategy.volatility_filter
        
        query = f"""
        WITH vol_data AS (
            SELECT 
                trade_date,
                trade_time,
                spot,
                atm_strike,
                -- ATM IV as proxy for overall IV
                AVG(CASE WHEN strike = atm_strike THEN (ce_iv + pe_iv) / 2 END) as atm_iv,
                -- IV smile
                AVG(CASE WHEN strike < atm_strike * 0.95 THEN pe_iv END) as otm_put_iv,
                AVG(CASE WHEN strike > atm_strike * 1.05 THEN ce_iv END) as otm_call_iv,
                -- Term structure (if multiple expiries)
                AVG(CASE WHEN expiry_bucket = 'CW' THEN (ce_iv + pe_iv) / 2 END) as cw_iv,
                AVG(CASE WHEN expiry_bucket = 'NW' THEN (ce_iv + pe_iv) / 2 END) as nw_iv,
                AVG(CASE WHEN expiry_bucket = 'CM' THEN (ce_iv + pe_iv) / 2 END) as cm_iv
            FROM {self.table_name}
            WHERE trade_date BETWEEN '{strategy.portfolio.start_date}' AND '{strategy.portfolio.end_date}'
            GROUP BY trade_date, trade_time, spot, atm_strike
        ),
        iv_percentiles AS (
            SELECT 
                trade_date,
                atm_iv,
                -- Calculate IV percentile over lookback period
                PERCENT_RANK() OVER (
                    ORDER BY atm_iv 
                    ROWS BETWEEN {vol_filter.ivp_lookback} PRECEDING AND CURRENT ROW
                ) as iv_percentile,
                -- Calculate IV rank
                (atm_iv - MIN(atm_iv) OVER (ROWS BETWEEN {vol_filter.ivr_lookback} PRECEDING AND CURRENT ROW)) /
                NULLIF(MAX(atm_iv) OVER (ROWS BETWEEN {vol_filter.ivr_lookback} PRECEDING AND CURRENT ROW) - 
                       MIN(atm_iv) OVER (ROWS BETWEEN {vol_filter.ivr_lookback} PRECEDING AND CURRENT ROW), 0) as iv_rank
            FROM vol_data
        ),
        realized_vol AS (
            SELECT 
                trade_date,
                -- Historical volatility calculation
                STDDEV(LN(spot / LAG(spot) OVER (ORDER BY trade_date, trade_time))) 
                    OVER (ORDER BY trade_date ROWS BETWEEN 30 PRECEDING AND CURRENT ROW) * SQRT(252) as hv_30,
                -- ATR calculation
                AVG(ABS(spot - LAG(spot) OVER (ORDER BY trade_date, trade_time))) 
                    OVER (ORDER BY trade_date ROWS BETWEEN {vol_filter.atr_period} PRECEDING AND CURRENT ROW) as atr
            FROM (
                SELECT DISTINCT trade_date, trade_time, spot 
                FROM {self.table_name}
                WHERE trade_date BETWEEN '{strategy.portfolio.start_date}' AND '{strategy.portfolio.end_date}'
            )
        )
        SELECT 
            v.*,
            p.iv_percentile,
            p.iv_rank,
            r.hv_30,
            r.atr,
            -- IV premium
            v.atm_iv - r.hv_30 as iv_premium,
            -- Volatility regime
            CASE 
                WHEN p.iv_percentile < 0.2 THEN 'VERY_LOW'
                WHEN p.iv_percentile < 0.4 THEN 'LOW'
                WHEN p.iv_percentile < 0.6 THEN 'MEDIUM'
                WHEN p.iv_percentile < 0.8 THEN 'HIGH'
                ELSE 'VERY_HIGH'
            END as vol_regime
        FROM vol_data v
        JOIN iv_percentiles p ON v.trade_date = p.trade_date
        JOIN realized_vol r ON v.trade_date = r.trade_date
        ORDER BY v.trade_date, v.trade_time
        """
        
        return query
    
    def build_greek_aggregation_query(self, strategy: CompletePOSStrategy) -> str:
        """Build query for Greek aggregation and limits checking"""
        
        # Get current positions subquery
        position_query = self.build_position_query(strategy)
        
        query = f"""
        WITH positions AS (
            {position_query}
        ),
        greek_aggregates AS (
            SELECT 
                trade_date,
                trade_time,
                -- Portfolio level Greeks
                SUM(net_delta) as portfolio_delta,
                SUM(net_gamma) as portfolio_gamma,
                SUM(net_theta) as portfolio_theta,
                SUM(net_vega) as portfolio_vega,
                -- Position level Greeks (assuming single position for now)
                net_delta as position_delta,
                net_gamma as position_gamma,
                net_theta as position_theta,
                net_vega as position_vega,
                -- Greeks by leg type
                SUM(CASE WHEN position_type = 'CREDIT' THEN net_delta ELSE 0 END) as credit_delta,
                SUM(CASE WHEN position_type = 'DEBIT' THEN net_delta ELSE 0 END) as debit_delta
            FROM positions
            GROUP BY trade_date, trade_time, net_delta, net_gamma, net_theta, net_vega
        )
        SELECT *,
            -- Greek limit checks
            {self._build_greek_limit_checks(strategy)},
            -- Delta neutrality check
            ABS(portfolio_delta) as delta_exposure,
            CASE 
                WHEN ABS(portfolio_delta) < {strategy.greek_limits.delta_neutral_threshold if strategy.greek_limits else 100} 
                THEN 'NEUTRAL' 
                ELSE 'EXPOSED' 
            END as delta_status,
            -- Gamma scalping opportunity
            CASE 
                WHEN ABS(portfolio_gamma) > {strategy.greek_limits.gamma_scalp_threshold if strategy.greek_limits else 50}
                THEN 'SCALP_OPPORTUNITY'
                ELSE 'NO_ACTION'
            END as gamma_scalp_signal
        FROM greek_aggregates
        ORDER BY trade_date, trade_time
        """
        
        return query
    
    def build_adjustment_scan_query(self, strategy: CompletePOSStrategy) -> str:
        """Build query to scan for adjustment triggers"""
        
        if not strategy.adjustment_rules:
            return ""
        
        # Get base position data
        position_query = self.build_position_query(strategy)
        
        # Build conditions for each rule
        rule_conditions = []
        for rule in strategy.adjustment_rules:
            if rule.enabled:
                condition = self._build_adjustment_condition(rule)
                rule_conditions.append(f"""
                    CASE WHEN {condition} THEN '{rule.rule_id}' ELSE NULL END as rule_{rule.rule_id}_triggered
                """)
        
        query = f"""
        WITH position_data AS (
            {position_query}
        ),
        market_conditions AS (
            SELECT 
                p.*,
                -- Calculate time in position
                EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - MIN(trade_date || ' ' || trade_time) 
                    OVER (PARTITION BY 1))) / 60 as minutes_in_position,
                -- Calculate underlying move
                (spot - FIRST_VALUE(spot) OVER (PARTITION BY 1 ORDER BY trade_date, trade_time)) / 
                    FIRST_VALUE(spot) OVER (PARTITION BY 1 ORDER BY trade_date, trade_time) * 100 as underlying_move_pct,
                -- Get current P&L
                total_premium as current_pnl
            FROM position_data p
        )
        SELECT *,
            {','.join(rule_conditions)},
            -- Aggregate trigger status
            CASE 
                WHEN {' OR '.join([f'rule_{r.rule_id}_triggered IS NOT NULL' for r in strategy.adjustment_rules if r.enabled])}
                THEN 'ADJUSTMENT_NEEDED'
                ELSE 'NO_ADJUSTMENT'
            END as adjustment_status
        FROM market_conditions
        WHERE trade_date = (SELECT MAX(trade_date) FROM market_conditions)
        ORDER BY trade_time DESC
        LIMIT 1
        """
        
        return query
    
    def build_breakeven_analysis_query(self, strategy: CompletePOSStrategy) -> str:
        """Build query for breakeven analysis"""
        
        be_config = strategy.strategy.breakeven_config
        if not be_config.enabled:
            return ""
        
        # Get position data
        position_query = self.build_position_query(strategy)
        
        query = f"""
        WITH position_data AS (
            {position_query}
        ),
        breakeven_calc AS (
            SELECT *,
                -- Calculate theoretical breakevens
                spot + (total_premium / 50) as upper_breakeven_theoretical,
                spot - (total_premium / 50) as lower_breakeven_theoretical,
                -- Add buffer
                spot + (total_premium / 50) + {be_config.buffer} as upper_breakeven_buffered,
                spot - (total_premium / 50) - {be_config.buffer} as lower_breakeven_buffered,
                -- Include transaction costs if configured
                {self._build_transaction_cost_adjustment(strategy)}
            FROM position_data
        ),
        be_analysis AS (
            SELECT *,
                -- Distance to breakeven
                ABS(spot - upper_breakeven_buffered) as distance_to_upper_be,
                ABS(spot - lower_breakeven_buffered) as distance_to_lower_be,
                -- BE approach status
                CASE 
                    WHEN spot > upper_breakeven_buffered * (1 - {be_config.spot_price_threshold}) 
                        OR spot < lower_breakeven_buffered * (1 + {be_config.spot_price_threshold})
                    THEN 'APPROACHING_BE'
                    ELSE 'SAFE'
                END as be_status,
                -- Time decay impact
                {self._build_time_decay_impact(strategy)}
            FROM breakeven_calc
        )
        SELECT *,
            -- Action recommendations
            CASE 
                WHEN be_status = 'APPROACHING_BE' THEN '{be_config.approach_action}'
                WHEN spot > upper_breakeven_buffered OR spot < lower_breakeven_buffered THEN '{be_config.breach_action}'
                ELSE 'HOLD'
            END as recommended_action
        FROM be_analysis
        ORDER BY trade_date, trade_time
        """
        
        return query
    
    # Helper methods for query building
    
    def _build_strike_condition(self, leg: EnhancedLegModel, strategy: CompletePOSStrategy) -> str:
        """Build strike selection condition for a leg"""
        
        # Get expiry condition
        expiry = self._get_leg_expiry(leg, strategy)
        expiry_condition = f"expiry_bucket = '{self.expiry_mapping.get(expiry, 'CW')}'"
        
        # Build strike selection based on method
        if leg.strike_method == StrikeMethod.ATM:
            return f"strike = atm_strike AND {expiry_condition}"
            
        elif leg.strike_method == StrikeMethod.OTM:
            offset = leg.strike_value or 100
            if leg.instrument == InstrumentType.CALL:
                return f"strike = atm_strike + {offset} AND {expiry_condition}"
            else:
                return f"strike = atm_strike - {offset} AND {expiry_condition}"
                
        elif leg.strike_method == StrikeMethod.ITM:
            offset = leg.strike_value or 100
            if leg.instrument == InstrumentType.CALL:
                return f"strike = atm_strike - {offset} AND {expiry_condition}"
            else:
                return f"strike = atm_strike + {offset} AND {expiry_condition}"
                
        elif leg.strike_method == StrikeMethod.FIXED:
            return f"strike = {leg.strike_value} AND {expiry_condition}"
            
        elif leg.strike_method == StrikeMethod.DELTA:
            # Delta-based selection (approximate)
            target_delta = leg.strike_delta or 0.5
            if leg.instrument == InstrumentType.CALL:
                return f"ABS(ce_delta - {target_delta}) < 0.05 AND {expiry_condition}"
            else:
                return f"ABS(pe_delta + {target_delta}) < 0.05 AND {expiry_condition}"
                
        elif leg.strike_method == StrikeMethod.PREMIUM:
            # Premium-based selection
            target_premium = leg.strike_premium or 100
            if leg.instrument == InstrumentType.CALL:
                return f"ABS(ce_close - {target_premium}) < 10 AND {expiry_condition}"
            else:
                return f"ABS(pe_close - {target_premium}) < 10 AND {expiry_condition}"
                
        elif leg.strike_method == StrikeMethod.BE_OPTIMIZED:
            # Breakeven optimized selection
            target_distance = leg.target_be_distance or 150
            return f"ABS(strike - (spot + {target_distance})) < 50 AND {expiry_condition}"
            
        else:
            # Handle numbered variants (OTM1, ITM2, etc.)
            method_str = str(leg.strike_method.value)
            if method_str.startswith('OTM'):
                offset = int(method_str[3:]) * 100 if len(method_str) > 3 else 100
                if leg.instrument == InstrumentType.CALL:
                    return f"strike = atm_strike + {offset} AND {expiry_condition}"
                else:
                    return f"strike = atm_strike - {offset} AND {expiry_condition}"
            elif method_str.startswith('ITM'):
                offset = int(method_str[3:]) * 100 if len(method_str) > 3 else 100
                if leg.instrument == InstrumentType.CALL:
                    return f"strike = atm_strike - {offset} AND {expiry_condition}"
                else:
                    return f"strike = atm_strike + {offset} AND {expiry_condition}"
            else:
                return f"strike = atm_strike AND {expiry_condition}"
    
    def _get_leg_expiry(self, leg: EnhancedLegModel, strategy: CompletePOSStrategy) -> str:
        """Determine expiry for a leg"""
        if leg.is_weekly_leg:
            return 'CURRENT_WEEK'
        elif strategy.strategy.position_type == PositionType.WEEKLY:
            return 'CURRENT_WEEK'
        elif strategy.strategy.position_type == PositionType.MONTHLY:
            return 'CURRENT_MONTH'
        else:
            return strategy.strategy.preferred_expiry.value
    
    def _build_aggregate_fields(self, legs: List[EnhancedLegModel]) -> str:
        """Build aggregate calculation fields"""
        
        # Sum fields across all legs
        premium_sum = " + ".join([f"leg{i}_premium" for i in range(len(legs))])
        delta_sum = " + ".join([f"leg{i}_delta" for i in range(len(legs))])
        gamma_sum = " + ".join([f"leg{i}_gamma" for i in range(len(legs))])
        theta_sum = " + ".join([f"leg{i}_theta" for i in range(len(legs))])
        vega_sum = " + ".join([f"leg{i}_vega" for i in range(len(legs))])
        
        return f"""
            ({premium_sum}) as total_premium,
            ({delta_sum}) as net_delta,
            ({gamma_sum}) as net_gamma,
            ({theta_sum}) as net_theta,
            ({vega_sum}) as net_vega,
            -- Risk metrics
            SQRT(POWER({delta_sum}, 2) + POWER({gamma_sum}, 2)) as risk_score,
            -- Premium per unit of risk
            ({premium_sum}) / NULLIF(ABS({delta_sum}), 0) as premium_per_delta
        """
    
    def _build_where_conditions(self, strategy: CompletePOSStrategy) -> str:
        """Build WHERE clause conditions"""
        conditions = []
        
        # Date range
        conditions.append(f"trade_date BETWEEN '{strategy.portfolio.start_date}' AND '{strategy.portfolio.end_date}'")
        
        # Time filters
        if strategy.strategy.entry_config.time_start:
            conditions.append(f"trade_time >= '{strategy.strategy.entry_config.time_start}'")
        if strategy.strategy.entry_config.time_end:
            conditions.append(f"trade_time <= '{strategy.strategy.entry_config.time_end}'")
        
        # Entry days filter
        if strategy.strategy.entry_config.days:
            day_numbers = []
            day_map = {'monday': 1, 'tuesday': 2, 'wednesday': 3, 'thursday': 4, 'friday': 5}
            for day in strategy.strategy.entry_config.days:
                if day.lower() in day_map:
                    day_numbers.append(str(day_map[day.lower()]))
            if day_numbers:
                conditions.append(f"EXTRACT(DOW FROM trade_date) IN ({','.join(day_numbers)})")
        
        # Volume/OI filters
        if strategy.strategy.entry_config.min_volume:
            conditions.append(f"(ce_volume + pe_volume) >= {strategy.strategy.entry_config.min_volume}")
        if strategy.strategy.entry_config.min_oi:
            conditions.append(f"(ce_oi + pe_oi) >= {strategy.strategy.entry_config.min_oi}")
        
        # Strike range filters (to limit data)
        conditions.append("strike BETWEEN atm_strike - 1000 AND atm_strike + 1000")
        
        return " AND ".join(conditions)
    
    def _build_market_condition_fields(self) -> str:
        """Build market condition analysis fields"""
        return """
            -- Market breadth
            SUM(CASE WHEN ce_volume > pe_volume THEN 1 ELSE 0 END) / 
                NULLIF(COUNT(*), 0) as call_bias,
            -- Put-Call Ratio
            SUM(pe_oi) / NULLIF(SUM(ce_oi), 0) as pcr_oi,
            -- Max pain approximation
            MIN(ABS(strike - spot)) as distance_to_max_pain
        """
    
    def _build_breakeven_calculations(self, strategy: CompletePOSStrategy) -> str:
        """Build breakeven calculation fields"""
        
        if not strategy.strategy.breakeven_config.enabled:
            return "NULL as upper_be, NULL as lower_be"
        
        return """
            spot + (total_premium / 50) as upper_be,
            spot - (total_premium / 50) as lower_be,
            -- BE probability (simplified)
            CASE 
                WHEN total_premium > 0 THEN 0.68  -- Credit spread default
                ELSE 0.32  -- Debit spread default
            END as be_probability
        """
    
    def _build_greek_limit_checks(self, strategy: CompletePOSStrategy) -> str:
        """Build Greek limit checking conditions"""
        
        if not strategy.greek_limits or not strategy.greek_limits.enabled:
            return "'NO_LIMITS' as greek_status"
        
        limits = strategy.greek_limits
        checks = []
        
        if limits.portfolio_max_delta:
            checks.append(f"ABS(portfolio_delta) > {limits.portfolio_max_delta}")
        if limits.portfolio_max_gamma:
            checks.append(f"ABS(portfolio_gamma) > {limits.portfolio_max_gamma}")
        if limits.portfolio_max_theta:
            checks.append(f"portfolio_theta < -{limits.portfolio_max_theta}")
        if limits.portfolio_max_vega:
            checks.append(f"ABS(portfolio_vega) > {limits.portfolio_max_vega}")
        
        if checks:
            return f"""
                CASE 
                    WHEN {' OR '.join(checks)} THEN 'LIMIT_BREACH'
                    ELSE 'WITHIN_LIMITS'
                END as greek_status
            """
        else:
            return "'NO_LIMITS' as greek_status"
    
    def _build_adjustment_condition(self, rule: AdjustmentRule) -> str:
        """Build condition for adjustment rule trigger"""
        
        conditions = []
        
        # Time-based triggers
        if rule.trigger_type == AdjustmentTrigger.TIME_BASED:
            if rule.check_time and rule.min_time_in_position:
                conditions.append(f"minutes_in_position >= {rule.min_time_in_position}")
            if rule.max_time_in_position:
                conditions.append(f"minutes_in_position <= {rule.max_time_in_position}")
        
        # Price-based triggers
        elif rule.trigger_type == AdjustmentTrigger.PRICE_BASED:
            if rule.trigger_comparison == "GREATER":
                conditions.append(f"spot > {rule.trigger_value}")
            elif rule.trigger_comparison == "LESS":
                conditions.append(f"spot < {rule.trigger_value}")
            elif rule.trigger_comparison == "BETWEEN" and rule.trigger_value2:
                conditions.append(f"spot BETWEEN {rule.trigger_value} AND {rule.trigger_value2}")
        
        # PnL-based triggers
        elif rule.trigger_type == AdjustmentTrigger.PNL_BASED:
            if rule.check_pnl:
                if rule.min_pnl:
                    conditions.append(f"current_pnl >= {rule.min_pnl}")
                if rule.max_pnl:
                    conditions.append(f"current_pnl <= {rule.max_pnl}")
        
        # Greek-based triggers
        elif rule.trigger_type == AdjustmentTrigger.GREEK_BASED:
            if rule.delta_trigger:
                conditions.append(f"ABS(net_delta) > {rule.delta_trigger}")
            if rule.gamma_trigger:
                conditions.append(f"ABS(net_gamma) > {rule.gamma_trigger}")
            if rule.theta_trigger:
                conditions.append(f"net_theta < -{rule.theta_trigger}")
            if rule.vega_trigger:
                conditions.append(f"ABS(net_vega) > {rule.vega_trigger}")
        
        # BE-based triggers
        elif rule.trigger_type == AdjustmentTrigger.BE_BASED:
            conditions.append("be_status = 'APPROACHING_BE'")
        
        # Underlying move check
        if rule.check_underlying_move and rule.underlying_move_percent:
            conditions.append(f"ABS(underlying_move_pct) > {rule.underlying_move_percent}")
        
        return " AND ".join(conditions) if conditions else "FALSE"
    
    def _build_transaction_cost_adjustment(self, strategy: CompletePOSStrategy) -> str:
        """Build transaction cost adjustments for BE"""
        
        if not strategy.strategy.breakeven_config.include_commissions:
            return "0 as commission_adjustment"
        
        # Calculate total lots
        total_lots = sum(leg.lots for leg in strategy.get_active_legs())
        
        # Assume Rs 20 per lot for simplicity
        commission_per_lot = 20
        total_commission = total_lots * commission_per_lot * 2  # Entry and exit
        
        return f"{total_commission} / 50 as commission_adjustment"
    
    def _build_time_decay_impact(self, strategy: CompletePOSStrategy) -> str:
        """Build time decay impact on breakeven"""
        
        if not strategy.strategy.breakeven_config.time_decay_factor:
            return "0 as theta_impact"
        
        return """
            -- Estimate theta impact on BE
            net_theta * GREATEST(1, EXTRACT(EPOCH FROM (expiry_date - CURRENT_DATE)) / 86400) as theta_impact
        """