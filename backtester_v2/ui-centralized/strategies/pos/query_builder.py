"""
Query Builder for POS (Positional) Strategy
Generates optimized SQL queries for HeavyDB with multi-leg support
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import date, time, datetime, timedelta
import pandas as pd

from .models import (
    POSStrategyModel,
    POSLegModel,
    POSPortfolioModel,
    StrikeSelection,
    OptionType,
    PositionType,
    AdjustmentTrigger,
    AdjustmentRule
)
from .constants import (
    QUERY_TEMPLATES,
    DB_COLUMN_MAPPINGS,
    ERROR_MESSAGES
)

logger = logging.getLogger(__name__)


class POSQueryBuilder:
    """Build optimized SQL queries for positional strategies"""
    
    def __init__(self, table_name: str = "nifty_option_chain"):
        self.table_name = table_name
        self.query_cache = {}
        
    def build_queries(self, strategy_model: POSStrategyModel) -> List[str]:
        """
        Build all queries needed for the strategy
        
        Args:
            strategy_model: Complete strategy configuration
            
        Returns:
            List of SQL queries to execute
        """
        queries = []
        
        try:
            # Generate main multi-leg query
            main_query = self._build_multi_leg_query(
                strategy_model.legs,
                strategy_model.portfolio
            )
            queries.append(main_query)
            
            # Generate Greeks calculation query if enabled
            if strategy_model.portfolio.calculate_greeks:
                greeks_query = self._build_greeks_query(
                    strategy_model.legs,
                    strategy_model.portfolio
                )
                queries.append(greeks_query)
                
            # Generate adjustment monitoring queries
            if strategy_model.portfolio.enable_adjustments:
                adjustment_queries = self._build_adjustment_queries(
                    strategy_model.legs,
                    strategy_model.portfolio
                )
                queries.extend(adjustment_queries)
                
            # Generate market regime filter query if configured
            if strategy_model.portfolio.market_regime_filter:
                regime_query = self._build_market_regime_query(
                    strategy_model.portfolio
                )
                queries.append(regime_query)
                
            return queries
            
        except Exception as e:
            logger.error(f"Error building queries: {str(e)}")
            raise ValueError(f"Query building failed: {str(e)}")
    
    def _build_multi_leg_query(self, 
                              legs: List[POSLegModel],
                              portfolio: POSPortfolioModel) -> str:
        """Build query for multi-leg options strategy"""
        
        # Build CTE for each leg
        leg_ctes = []
        leg_selections = []
        
        for leg in legs:
            leg_cte = self._build_leg_cte(leg, portfolio)
            leg_ctes.append(leg_cte)
            leg_selections.append(self._build_leg_selection(leg))
            
        # Build strike selection CTE
        strike_cte = self._build_strike_selection_cte(legs, portfolio)
        
        # Build main query
        query = f"""
        WITH trading_dates AS (
            SELECT DISTINCT trade_date
            FROM {self.table_name}
            WHERE trade_date BETWEEN '{portfolio.start_date}' AND '{portfolio.end_date}'
            AND index_name = '{portfolio.index_name}'
            ORDER BY trade_date
        ),
        {strike_cte},
        {','.join(leg_ctes)},
        combined_legs AS (
            SELECT 
                td.trade_date,
                {','.join(leg_selections)}
            FROM trading_dates td
            {self._build_leg_joins(legs)}
        ),
        portfolio_positions AS (
            SELECT 
                *,
                -- Calculate portfolio metrics
                {self._build_portfolio_calculations(legs)}
            FROM combined_legs
        )
        SELECT * FROM portfolio_positions
        WHERE trade_date BETWEEN '{portfolio.start_date}' AND '{portfolio.end_date}'
        ORDER BY trade_date, leg_1_trade_time
        """
        
        return self._optimize_query(query)
    
    def _build_leg_cte(self, leg: POSLegModel, portfolio: POSPortfolioModel) -> str:
        """Build CTE for individual leg"""
        
        # Build strike filter based on selection method
        strike_filter = self._build_strike_filter(leg, portfolio)
        
        # Build time filter
        time_filter = self._build_time_filter(leg)
        
        query = f"""
        leg_{leg.leg_id}_data AS (
            SELECT 
                l.trade_date,
                l.trade_time,
                l.expiry_date,
                l.strike_price,
                l.option_type,
                l.open_price,
                l.high_price,
                l.low_price,
                l.close_price,
                l.volume,
                l.open_interest,
                l.implied_volatility,
                l.underlying_value,
                l.delta,
                l.gamma,
                l.theta,
                l.vega,
                l.rho,
                -- Position calculations
                {leg.lots * leg.lot_size} as position_size,
                '{leg.position_type}' as position_type,
                CASE 
                    WHEN '{leg.position_type}' = 'BUY' THEN -1 * l.close_price * {leg.lots * leg.lot_size}
                    ELSE l.close_price * {leg.lots * leg.lot_size}
                END as premium,
                -- Entry/Exit flags
                CASE 
                    WHEN l.trade_time >= '{leg.entry_time}' 
                    AND l.trade_time < '{leg.entry_time}'::time + INTERVAL '{leg.entry_buffer_time} minutes'
                    THEN 1 ELSE 0 
                END as entry_flag,
                CASE 
                    WHEN l.trade_time >= '{leg.exit_time}'::time - INTERVAL '{leg.exit_buffer_time} minutes'
                    AND l.trade_time <= '{leg.exit_time}'
                    THEN 1 ELSE 0 
                END as exit_flag
            FROM {self.table_name} l
            INNER JOIN strike_selection_{leg.leg_id} ss
                ON l.trade_date = ss.trade_date
                AND l.strike_price = ss.selected_strike
                AND l.expiry_date = ss.expiry_date
            WHERE l.option_type = '{leg.option_type}'
            AND l.index_name = '{portfolio.index_name}'
            {time_filter}
            {self._build_additional_filters(leg)}
        )"""
        
        return query
    
    def _build_strike_selection_cte(self, 
                                   legs: List[POSLegModel],
                                   portfolio: POSPortfolioModel) -> str:
        """Build CTEs for strike selection logic"""
        
        strike_ctes = []
        
        for leg in legs:
            if leg.strike_selection == StrikeSelection.ATM:
                cte = self._build_atm_strike_cte(leg, portfolio)
            elif leg.strike_selection == StrikeSelection.OTM:
                cte = self._build_otm_strike_cte(leg, portfolio)
            elif leg.strike_selection == StrikeSelection.ITM:
                cte = self._build_itm_strike_cte(leg, portfolio)
            elif leg.strike_selection == StrikeSelection.STRIKE_PRICE:
                cte = self._build_fixed_strike_cte(leg, portfolio)
            elif leg.strike_selection == StrikeSelection.DELTA_BASED:
                cte = self._build_delta_strike_cte(leg, portfolio)
            elif leg.strike_selection == StrikeSelection.PERCENTAGE_BASED:
                cte = self._build_percentage_strike_cte(leg, portfolio)
            else:
                raise ValueError(f"Unknown strike selection: {leg.strike_selection}")
                
            strike_ctes.append(cte)
            
        return ',\n'.join(strike_ctes)
    
    def _build_atm_strike_cte(self, leg: POSLegModel, portfolio: POSPortfolioModel) -> str:
        """Build CTE for ATM strike selection"""
        
        offset = leg.strike_offset or 0
        
        return f"""
        strike_selection_{leg.leg_id} AS (
            SELECT 
                trade_date,
                underlying_value,
                -- Find nearest expiry
                (SELECT MIN(expiry_date) 
                 FROM {self.table_name} 
                 WHERE trade_date = t.trade_date 
                 AND expiry_date > t.trade_date
                 AND index_name = '{portfolio.index_name}'
                ) as expiry_date,
                -- Find ATM strike with offset
                (SELECT strike_price
                 FROM {self.table_name}
                 WHERE trade_date = t.trade_date
                 AND index_name = '{portfolio.index_name}'
                 AND option_type = '{leg.option_type}'
                 AND expiry_date = (
                     SELECT MIN(expiry_date) 
                     FROM {self.table_name} 
                     WHERE trade_date = t.trade_date 
                     AND expiry_date > t.trade_date
                     AND index_name = '{portfolio.index_name}'
                 )
                 ORDER BY ABS(strike_price - (underlying_value + {offset}))
                 LIMIT 1
                ) as selected_strike
            FROM (
                SELECT DISTINCT trade_date, MAX(underlying_value) as underlying_value
                FROM {self.table_name}
                WHERE trade_date BETWEEN '{portfolio.start_date}' AND '{portfolio.end_date}'
                AND index_name = '{portfolio.index_name}'
                GROUP BY trade_date
            ) t
        )"""
    
    def _build_otm_strike_cte(self, leg: POSLegModel, portfolio: POSPortfolioModel) -> str:
        """Build CTE for OTM strike selection"""
        
        offset = leg.strike_offset or 100
        
        if leg.option_type == OptionType.CALL:
            strike_condition = f"strike_price >= underlying_value + {offset}"
            order_by = "strike_price ASC"
        else:  # PUT
            strike_condition = f"strike_price <= underlying_value - {offset}"
            order_by = "strike_price DESC"
            
        return f"""
        strike_selection_{leg.leg_id} AS (
            SELECT 
                trade_date,
                underlying_value,
                expiry_date,
                (SELECT strike_price
                 FROM {self.table_name}
                 WHERE trade_date = t.trade_date
                 AND index_name = '{portfolio.index_name}'
                 AND option_type = '{leg.option_type}'
                 AND expiry_date = t.expiry_date
                 AND {strike_condition}
                 ORDER BY {order_by}
                 LIMIT 1
                ) as selected_strike
            FROM (
                SELECT DISTINCT 
                    trade_date, 
                    MAX(underlying_value) as underlying_value,
                    MIN(expiry_date) as expiry_date
                FROM {self.table_name}
                WHERE trade_date BETWEEN '{portfolio.start_date}' AND '{portfolio.end_date}'
                AND index_name = '{portfolio.index_name}'
                AND expiry_date > trade_date
                GROUP BY trade_date
            ) t
        )"""
    
    def _build_delta_strike_cte(self, leg: POSLegModel, portfolio: POSPortfolioModel) -> str:
        """Build CTE for delta-based strike selection"""
        
        target_delta = leg.delta_target
        if not target_delta:
            raise ValueError(f"Delta target required for leg {leg.leg_id}")
            
        return f"""
        strike_selection_{leg.leg_id} AS (
            SELECT 
                trade_date,
                underlying_value,
                expiry_date,
                (SELECT strike_price
                 FROM {self.table_name}
                 WHERE trade_date = t.trade_date
                 AND index_name = '{portfolio.index_name}'
                 AND option_type = '{leg.option_type}'
                 AND expiry_date = t.expiry_date
                 AND trade_time = '09:20:00'
                 ORDER BY ABS(delta - {target_delta})
                 LIMIT 1
                ) as selected_strike
            FROM (
                SELECT DISTINCT 
                    trade_date, 
                    MAX(underlying_value) as underlying_value,
                    MIN(expiry_date) as expiry_date
                FROM {self.table_name}
                WHERE trade_date BETWEEN '{portfolio.start_date}' AND '{portfolio.end_date}'
                AND index_name = '{portfolio.index_name}'
                AND expiry_date > trade_date
                GROUP BY trade_date
            ) t
        )"""
    
    def _build_greeks_query(self, 
                          legs: List[POSLegModel],
                          portfolio: POSPortfolioModel) -> str:
        """Build query for Greek calculations"""
        
        leg_greek_calcs = []
        for leg in legs:
            position_multiplier = 1 if leg.position_type == PositionType.BUY else -1
            leg_greek_calcs.append(f"""
                leg_{leg.leg_id}_delta * {position_multiplier} * {leg.lots * leg.lot_size} as leg_{leg.leg_id}_portfolio_delta,
                leg_{leg.leg_id}_gamma * {position_multiplier} * {leg.lots * leg.lot_size} as leg_{leg.leg_id}_portfolio_gamma,
                leg_{leg.leg_id}_theta * {position_multiplier} * {leg.lots * leg.lot_size} as leg_{leg.leg_id}_portfolio_theta,
                leg_{leg.leg_id}_vega * {position_multiplier} * {leg.lots * leg.lot_size} as leg_{leg.leg_id}_portfolio_vega
            """)
            
        # Sum all leg Greeks
        total_greek_calcs = []
        for greek in ['delta', 'gamma', 'theta', 'vega']:
            leg_sum = ' + '.join([f"leg_{leg.leg_id}_portfolio_{greek}" for leg in legs])
            total_greek_calcs.append(f"{leg_sum} as total_portfolio_{greek}")
            
        query = f"""
        WITH portfolio_greeks AS (
            SELECT 
                trade_date,
                trade_time,
                {','.join(leg_greek_calcs)},
                {','.join(total_greek_calcs)}
            FROM portfolio_positions
        )
        SELECT 
            pg.*,
            -- Greek limit checks
            CASE WHEN total_portfolio_delta > {portfolio.greek_limits.max_delta or 999999} THEN 1 ELSE 0 END as delta_limit_breach,
            CASE WHEN total_portfolio_gamma > {portfolio.greek_limits.max_gamma or 999999} THEN 1 ELSE 0 END as gamma_limit_breach,
            CASE WHEN total_portfolio_theta < {portfolio.greek_limits.min_theta or -999999} THEN 1 ELSE 0 END as theta_limit_breach,
            CASE WHEN total_portfolio_vega > {portfolio.greek_limits.max_vega or 999999} THEN 1 ELSE 0 END as vega_limit_breach
        FROM portfolio_greeks pg
        """
        
        return query
    
    def _build_adjustment_queries(self,
                                legs: List[POSLegModel],
                                portfolio: POSPortfolioModel) -> List[str]:
        """Build queries for adjustment monitoring"""
        
        queries = []
        
        # Collect all adjustment rules
        all_rules = []
        for leg in legs:
            for rule in leg.adjustment_rules:
                all_rules.append((leg, rule))
                
        # Group rules by trigger type for efficiency
        rules_by_trigger = {}
        for leg, rule in all_rules:
            trigger_type = rule.trigger_type
            if trigger_type not in rules_by_trigger:
                rules_by_trigger[trigger_type] = []
            rules_by_trigger[trigger_type].append((leg, rule))
            
        # Build queries for each trigger type
        for trigger_type, rules in rules_by_trigger.items():
            if trigger_type == AdjustmentTrigger.PRICE_BASED:
                query = self._build_price_adjustment_query(rules, portfolio)
            elif trigger_type == AdjustmentTrigger.GREEK_BASED:
                query = self._build_greek_adjustment_query(rules, portfolio)
            elif trigger_type == AdjustmentTrigger.TIME_BASED:
                query = self._build_time_adjustment_query(rules, portfolio)
            elif trigger_type == AdjustmentTrigger.PNL_BASED:
                query = self._build_pnl_adjustment_query(rules, portfolio)
            else:
                continue
                
            queries.append(query)
            
        return queries
    
    def _build_price_adjustment_query(self,
                                    rules: List[Tuple[POSLegModel, AdjustmentRule]],
                                    portfolio: POSPortfolioModel) -> str:
        """Build query for price-based adjustments"""
        
        conditions = []
        for leg, rule in rules:
            condition = f"""
                CASE 
                    WHEN {rule.trigger_condition} THEN '{rule.rule_id}'
                    ELSE NULL
                END as {rule.rule_id}_trigger
            """
            conditions.append(condition)
            
        query = f"""
        SELECT 
            trade_date,
            trade_time,
            underlying_value,
            {','.join(conditions)},
            -- Combine all triggers
            ARRAY_REMOVE(ARRAY[
                {','.join([f"{rule.rule_id}_trigger" for _, rule in rules])}
            ], NULL) as triggered_rules
        FROM portfolio_positions
        WHERE trade_date BETWEEN '{portfolio.start_date}' AND '{portfolio.end_date}'
        """
        
        return query
    
    def _build_leg_selection(self, leg: POSLegModel) -> str:
        """Build selection fields for a leg"""
        
        fields = [
            f"leg_{leg.leg_id}_strike_price",
            f"leg_{leg.leg_id}_trade_time",
            f"leg_{leg.leg_id}_close_price",
            f"leg_{leg.leg_id}_volume",
            f"leg_{leg.leg_id}_open_interest",
            f"leg_{leg.leg_id}_implied_volatility",
            f"leg_{leg.leg_id}_delta",
            f"leg_{leg.leg_id}_gamma",
            f"leg_{leg.leg_id}_theta",
            f"leg_{leg.leg_id}_vega",
            f"leg_{leg.leg_id}_premium",
            f"leg_{leg.leg_id}_position_size"
        ]
        
        return ',\n'.join(fields)
    
    def _build_leg_joins(self, legs: List[POSLegModel]) -> str:
        """Build joins for combining legs"""
        
        joins = []
        for i, leg in enumerate(legs):
            if i == 0:
                join = f"""
                LEFT JOIN leg_{leg.leg_id}_data leg_{leg.leg_id}
                    ON td.trade_date = leg_{leg.leg_id}.trade_date
                """
            else:
                # Join on time alignment
                join = f"""
                LEFT JOIN leg_{leg.leg_id}_data leg_{leg.leg_id}
                    ON td.trade_date = leg_{leg.leg_id}.trade_date
                    AND leg_1.trade_time = leg_{leg.leg_id}.trade_time
                """
            joins.append(join)
            
        return '\n'.join(joins)
    
    def _build_portfolio_calculations(self, legs: List[POSLegModel]) -> str:
        """Build portfolio-level calculations"""
        
        # Premium calculation
        premium_sum = ' + '.join([f"leg_{leg.leg_id}_premium" for leg in legs])
        
        # Position count
        position_count = len(legs)
        
        calculations = f"""
            {premium_sum} as total_premium,
            {position_count} as total_legs,
            -- Max loss calculation (simplified)
            CASE 
                WHEN {premium_sum} < 0 THEN {premium_sum}
                ELSE -999999
            END as max_loss,
            -- Breakeven calculations would go here
            0 as upper_breakeven,
            0 as lower_breakeven
        """
        
        return calculations
    
    def _build_strike_filter(self, leg: POSLegModel, portfolio: POSPortfolioModel) -> str:
        """Build strike filter based on selection method"""
        
        if leg.strike_selection == StrikeSelection.STRIKE_PRICE:
            return f"AND strike_price = {leg.strike_price}"
        else:
            # Dynamic selection handled in CTEs
            return ""
    
    def _build_time_filter(self, leg: POSLegModel) -> str:
        """Build time filter for leg"""
        
        return f"""
            AND trade_time >= '{leg.entry_time}'
            AND trade_time <= '{leg.exit_time}'
        """
    
    def _build_additional_filters(self, leg: POSLegModel) -> str:
        """Build any additional filters for the leg"""
        
        filters = []
        
        if leg.skip_expiry_day:
            filters.append("AND trade_date != expiry_date")
            
        return '\n'.join(filters)
    
    def _build_fixed_strike_cte(self, leg: POSLegModel, portfolio: POSPortfolioModel) -> str:
        """Build CTE for fixed strike selection"""
        
        return f"""
        strike_selection_{leg.leg_id} AS (
            SELECT 
                trade_date,
                MAX(underlying_value) as underlying_value,
                MIN(expiry_date) as expiry_date,
                {leg.strike_price} as selected_strike
            FROM {self.table_name}
            WHERE trade_date BETWEEN '{portfolio.start_date}' AND '{portfolio.end_date}'
            AND index_name = '{portfolio.index_name}'
            AND expiry_date > trade_date
            GROUP BY trade_date
        )"""
    
    def _build_itm_strike_cte(self, leg: POSLegModel, portfolio: POSPortfolioModel) -> str:
        """Build CTE for ITM strike selection"""
        
        offset = leg.strike_offset or 100
        
        if leg.option_type == OptionType.CALL:
            strike_condition = f"strike_price <= underlying_value - {offset}"
            order_by = "strike_price DESC"
        else:  # PUT
            strike_condition = f"strike_price >= underlying_value + {offset}"
            order_by = "strike_price ASC"
            
        return f"""
        strike_selection_{leg.leg_id} AS (
            SELECT 
                trade_date,
                underlying_value,
                expiry_date,
                (SELECT strike_price
                 FROM {self.table_name}
                 WHERE trade_date = t.trade_date
                 AND index_name = '{portfolio.index_name}'
                 AND option_type = '{leg.option_type}'
                 AND expiry_date = t.expiry_date
                 AND {strike_condition}
                 ORDER BY {order_by}
                 LIMIT 1
                ) as selected_strike
            FROM (
                SELECT DISTINCT 
                    trade_date, 
                    MAX(underlying_value) as underlying_value,
                    MIN(expiry_date) as expiry_date
                FROM {self.table_name}
                WHERE trade_date BETWEEN '{portfolio.start_date}' AND '{portfolio.end_date}'
                AND index_name = '{portfolio.index_name}'
                AND expiry_date > trade_date
                GROUP BY trade_date
            ) t
        )"""
    
    def _build_percentage_strike_cte(self, leg: POSLegModel, portfolio: POSPortfolioModel) -> str:
        """Build CTE for percentage-based strike selection"""
        
        percentage = leg.percentage_offset or 0.02  # Default 2%
        
        return f"""
        strike_selection_{leg.leg_id} AS (
            SELECT 
                trade_date,
                underlying_value,
                expiry_date,
                (SELECT strike_price
                 FROM {self.table_name}
                 WHERE trade_date = t.trade_date
                 AND index_name = '{portfolio.index_name}'
                 AND option_type = '{leg.option_type}'
                 AND expiry_date = t.expiry_date
                 ORDER BY ABS(strike_price - (underlying_value * (1 + {percentage})))
                 LIMIT 1
                ) as selected_strike
            FROM (
                SELECT DISTINCT 
                    trade_date, 
                    MAX(underlying_value) as underlying_value,
                    MIN(expiry_date) as expiry_date
                FROM {self.table_name}
                WHERE trade_date BETWEEN '{portfolio.start_date}' AND '{portfolio.end_date}'
                AND index_name = '{portfolio.index_name}'
                AND expiry_date > trade_date
                GROUP BY trade_date
            ) t
        )"""
    
    def _build_market_regime_query(self, portfolio: POSPortfolioModel) -> str:
        """Build query for market regime filtering"""
        
        filters = []
        
        if portfolio.market_regime_filter.vix_filter:
            vix = portfolio.market_regime_filter.vix_filter
            if vix.min_vix:
                filters.append(f"vix_value >= {vix.min_vix}")
            if vix.max_vix:
                filters.append(f"vix_value <= {vix.max_vix}")
                
        # Add other regime filters here
        
        where_clause = " AND ".join(filters) if filters else "1=1"
        
        query = f"""
        SELECT 
            trade_date,
            MAX(vix_value) as vix_value,
            -- Market regime classification
            CASE 
                WHEN vix_value < 15 THEN 'LOW_VOL'
                WHEN vix_value < 25 THEN 'NORMAL'
                WHEN vix_value < 35 THEN 'HIGH_VOL'
                ELSE 'EXTREME'
            END as volatility_regime
        FROM market_data
        WHERE trade_date BETWEEN '{portfolio.start_date}' AND '{portfolio.end_date}'
        AND {where_clause}
        GROUP BY trade_date
        """
        
        return query
    
    def _optimize_query(self, query: str) -> str:
        """Optimize query for GPU execution"""
        
        # Remove extra whitespace
        query = ' '.join(query.split())
        
        # Add GPU hints (specific to HeavyDB)
        query = f"/*+ cpu_mode=false, watchdog_max_size=0 */ {query}"
        
        return query