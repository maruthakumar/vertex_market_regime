#!/usr/bin/env python3
"""
TBS Query Builder - Optimized version for high-performance SQL generation
Performance target: <2s query generation, >200K rows/sec processing
"""

import logging
import pymapd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, date, time
import pandas as pd
import functools
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json

logger = logging.getLogger(__name__)

class OptimizedTBSQueryBuilder:
    """High-performance SQL query builder for TBS strategy execution"""

    def __init__(self, table_name: str = 'nifty_option_chain'):
        self.table_name = table_name
        self.query_cache = {}  # Cache for compiled queries
        
        # Pre-compiled query templates for maximum performance
        self.optimized_templates = {
            'atm_vectorized': self._get_atm_vectorized_template(),
            'option_batch': self._get_option_batch_template(),
            'entry_exit_streaming': self._get_entry_exit_streaming_template(),
            'bulk_pnl': self._get_bulk_pnl_template()
        }
        
        # HeavyDB optimization parameters
        self.heavydb_params = {
            'host': 'localhost',
            'port': 6274,
            'user': 'admin',
            'password': 'HyperInteractive',
            'dbname': 'heavyai'
        }
        
        # GPU optimization settings
        self.gpu_optimization = {
            'cpu_mode': False,
            'watchdog_max_size': 0,
            'parallel_top_min': 1000,
            'group_by_buffer_size': 2000000000,  # 2GB
            'enable_columnar_output': True,
            'enable_lazy_fetch': True
        }
        
        # Performance tracking
        self.query_metrics = {
            'queries_generated': 0,
            'cache_hits': 0,
            'total_execution_time': 0.0,
            'avg_rows_per_second': 0.0
        }
        
    def build_queries_optimized(self, strategy_params: Dict[str, Any]) -> List[str]:
        """
        Build optimized queries for TBS strategy execution
        Target: <2s generation time, batched operations
        """
        import time
        start_time = time.perf_counter()
        
        # Extract parameters
        portfolio_params = strategy_params.get('portfolio_settings', {})
        strategies = strategy_params.get('strategies', [])
        
        if not strategies:
            logger.warning("No strategies provided for query building")
            return []
            
        # Generate cache key for entire strategy set
        cache_key = self._generate_cache_key(strategy_params)
        
        # Check cache first
        if cache_key in self.query_cache:
            self.query_metrics['cache_hits'] += 1
            logger.info(f"Cache hit for strategy queries: {cache_key[:8]}...")
            return self.query_cache[cache_key]
        
        # Build queries in parallel for multiple strategies
        all_queries = []
        
        # Use thread pool for parallel query generation
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_strategy = {
                executor.submit(
                    self._build_strategy_queries_optimized, 
                    strategy, 
                    portfolio_params
                ): strategy for strategy in strategies
            }
            
            for future in as_completed(future_to_strategy):
                strategy = future_to_strategy[future]
                try:
                    strategy_queries = future.result()
                    all_queries.extend(strategy_queries)
                except Exception as e:
                    logger.error(f"Failed to build queries for strategy {strategy.get('strategy_name', 'unknown')}: {e}")
        
        # Apply GPU optimizations to all queries
        optimized_queries = [self._apply_gpu_optimizations(q) for q in all_queries]
        
        # Cache the result
        self.query_cache[cache_key] = optimized_queries
        
        # Update metrics
        self.query_metrics['queries_generated'] += len(optimized_queries)
        generation_time = time.perf_counter() - start_time
        self.query_metrics['total_execution_time'] += generation_time
        
        logger.info(f"Generated {len(optimized_queries)} optimized queries in {generation_time:.3f}s")
        
        return optimized_queries
        
    def _build_strategy_queries_optimized(self, strategy: Dict[str, Any], 
                                        portfolio_params: Dict[str, Any]) -> List[str]:
        """Build optimized queries for a single strategy"""
        queries = []
        legs = strategy.get('legs', [])
        
        if not legs:
            return []
            
        # Extract common parameters
        start_date = portfolio_params.get('start_date')
        end_date = portfolio_params.get('end_date')
        index = portfolio_params.get('index', 'NIFTY')
        
        # Group legs by similar characteristics for batch processing
        leg_groups = self._group_legs_for_batching(legs)
        
        for group_type, grouped_legs in leg_groups.items():
            if group_type == 'atm_based':
                # Generate single ATM calculation for all ATM-based legs
                atm_query = self._build_atm_vectorized_query(
                    grouped_legs, start_date, end_date, index
                )
                queries.append(atm_query)
                
            elif group_type == 'fixed_strike':
                # Batch fixed strike legs together
                batch_query = self._build_fixed_strike_batch_query(
                    grouped_legs, start_date, end_date, index, portfolio_params
                )
                queries.append(batch_query)
                
            elif group_type == 'premium_based':
                # Handle premium-based selections
                premium_query = self._build_premium_batch_query(
                    grouped_legs, start_date, end_date, index, portfolio_params
                )
                queries.append(premium_query)
        
        # Generate single comprehensive P&L calculation query
        pnl_query = self._build_comprehensive_pnl_query(
            legs, start_date, end_date, index, portfolio_params
        )
        queries.append(pnl_query)
        
        return queries
        
    def _group_legs_for_batching(self, legs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group legs by similar characteristics for efficient batch processing"""
        groups = {
            'atm_based': [],
            'fixed_strike': [],
            'premium_based': [],
            'delta_based': []
        }
        
        for leg in legs:
            strike_selection = leg.get('strike_selection', 'ATM')
            
            if strike_selection in ['ATM', 'ITM1', 'ITM2', 'OTM1', 'OTM2']:
                groups['atm_based'].append(leg)
            elif strike_selection == 'FIXED':
                groups['fixed_strike'].append(leg)
            elif strike_selection == 'PREMIUM':
                groups['premium_based'].append(leg)  
            elif strike_selection == 'DELTA':
                groups['delta_based'].append(leg)
                
        # Remove empty groups
        return {k: v for k, v in groups.items() if v}
        
    def _build_atm_vectorized_query(self, legs: List[Dict[str, Any]], 
                                   start_date: date, end_date: date, index: str) -> str:
        """Build vectorized ATM calculation for multiple legs"""
        
        # Extract unique expiry rules from legs
        expiry_rules = list(set(leg.get('expiry_rule', 'CW') for leg in legs))
        entry_times = list(set(leg.get('entry_time', time(9, 20)) for leg in legs))
        
        # Build comprehensive ATM query handling all variations
        query = f"""
        /*+ cpu_mode=false, watchdog_max_size=0, parallel_top_min=1000 */
        WITH spot_prices AS (
            SELECT DISTINCT
                trade_date,
                time,
                fut_close as spot_price
            FROM {self.table_name}
            WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'
                AND symbol = '{index}'
                AND time IN ({','.join([f"'{t}'" for t in entry_times])})
                AND fut_close > 0
        ),
        expiry_ranking AS (
            SELECT DISTINCT 
                trade_date,
                expiry_date,
                ROW_NUMBER() OVER (PARTITION BY trade_date ORDER BY expiry_date) as expiry_rank,
                EXTRACT(MONTH FROM expiry_date) as expiry_month,
                EXTRACT(MONTH FROM trade_date) as trade_month
            FROM {self.table_name}
            WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'
                AND symbol = '{index}'
                AND expiry_date > trade_date
        ),
        synthetic_future AS (
            SELECT 
                t.trade_date,
                t.time,
                t.expiry_date,
                t.strike,
                er.expiry_rank,
                er.expiry_month,
                er.trade_month,
                sp.spot_price,
                t.strike + t.ce_close - t.pe_close as synthetic_price,
                ABS(t.strike - sp.spot_price) as strike_spot_diff,
                t.ce_close,
                t.pe_close
            FROM {self.table_name} t
            JOIN expiry_ranking er ON t.trade_date = er.trade_date AND t.expiry_date = er.expiry_date
            JOIN spot_prices sp ON t.trade_date = sp.trade_date AND t.time = sp.time
            WHERE t.trade_date BETWEEN '{start_date}' AND '{end_date}'
                AND t.symbol = '{index}'
                AND t.time IN ({','.join([f"'{t}'" for t in entry_times])})
                AND t.ce_close > 0 AND t.pe_close > 0
        ),
        atm_strikes AS (
            SELECT 
                trade_date,
                time,
                expiry_date,
                expiry_rank,
                expiry_month,
                trade_month,
                spot_price,
                strike as atm_strike,
                synthetic_price,
                ROW_NUMBER() OVER (
                    PARTITION BY trade_date, time, expiry_date 
                    ORDER BY strike_spot_diff
                ) as atm_rank
            FROM synthetic_future
        ),
        strike_offsets AS (
            SELECT 
                trade_date,
                time,
                expiry_date,
                expiry_rank,
                expiry_month,
                trade_month,
                spot_price,
                atm_strike,
                synthetic_price,
                atm_strike as atm_strike_0,
                LAG(atm_strike, 1) OVER (PARTITION BY trade_date, time, expiry_date ORDER BY atm_strike) as itm_strike_1,
                LAG(atm_strike, 2) OVER (PARTITION BY trade_date, time, expiry_date ORDER BY atm_strike) as itm_strike_2,
                LEAD(atm_strike, 1) OVER (PARTITION BY trade_date, time, expiry_date ORDER BY atm_strike) as otm_strike_1,
                LEAD(atm_strike, 2) OVER (PARTITION BY trade_date, time, expiry_date ORDER BY atm_strike) as otm_strike_2
            FROM atm_strikes
            WHERE atm_rank = 1
        )
        SELECT 
            trade_date,
            time,
            expiry_date,
            expiry_rank,
            expiry_month,
            trade_month,
            spot_price,
            atm_strike,
            synthetic_price,
            atm_strike_0,
            COALESCE(itm_strike_1, atm_strike) as itm_strike_1,
            COALESCE(itm_strike_2, atm_strike) as itm_strike_2,
            COALESCE(otm_strike_1, atm_strike) as otm_strike_1,
            COALESCE(otm_strike_2, atm_strike) as otm_strike_2,
            -- Add expiry rule filters
            CASE WHEN expiry_rank = 1 THEN 'CW' ELSE NULL END as current_week,
            CASE WHEN expiry_rank = 2 THEN 'NW' ELSE NULL END as next_week,
            CASE WHEN expiry_month = trade_month THEN 'CM' ELSE NULL END as current_monthly,
            CASE WHEN expiry_month = trade_month + 1 THEN 'NM' ELSE NULL END as next_monthly
        FROM strike_offsets
        ORDER BY trade_date, time, expiry_date
        """
        
        return query
        
    def _build_fixed_strike_batch_query(self, legs: List[Dict[str, Any]], 
                                      start_date: date, end_date: date, index: str,
                                      portfolio_params: Dict[str, Any]) -> str:
        """Build batched query for fixed strike legs"""
        
        # Extract all unique strikes and option types
        strike_option_pairs = []
        for leg in legs:
            strike = leg.get('strike_value', 0)
            option_type = leg.get('option_type', 'CE')
            entry_time = leg.get('entry_time', time(9, 20))
            exit_time = leg.get('exit_time', time(15, 15))
            strike_option_pairs.append((strike, option_type, entry_time, exit_time))
        
        # Remove duplicates
        unique_pairs = list(set(strike_option_pairs))
        
        # Build conditions for all strikes
        strike_conditions = []
        for strike, option_type, entry_time, exit_time in unique_pairs:
            strike_conditions.append(f"(strike = {strike} AND time IN ('{entry_time}', '{exit_time}'))")
        
        condition_sql = " OR ".join(strike_conditions)
        
        query = f"""
        /*+ cpu_mode=false, watchdog_max_size=0 */
        WITH fixed_strikes AS (
            SELECT 
                trade_date,
                expiry_date,
                strike,
                time,
                ce_open, ce_high, ce_low, ce_close,
                pe_open, pe_high, pe_low, pe_close,
                fut_open, fut_high, fut_low, fut_close,
                volume,
                open_interest
            FROM {self.table_name}
            WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'
                AND symbol = '{index}'
                AND ({condition_sql})
        )
        SELECT * FROM fixed_strikes
        ORDER BY trade_date, strike, time
        """
        
        return query
        
    def _build_premium_batch_query(self, legs: List[Dict[str, Any]], 
                                 start_date: date, end_date: date, index: str,
                                 portfolio_params: Dict[str, Any]) -> str:
        """Build batched query for premium-based selections"""
        
        # Extract premium ranges and option types
        premium_conditions = []
        for leg in legs:
            option_type = leg.get('option_type', 'CE')
            strike_value = leg.get('strike_value', 100)  # Premium range
            entry_time = leg.get('entry_time', time(9, 20))
            
            if option_type == 'CE':
                premium_conditions.append(f"(ce_close BETWEEN {strike_value-10} AND {strike_value+10} AND time = '{entry_time}')")
            elif option_type == 'PE':
                premium_conditions.append(f"(pe_close BETWEEN {strike_value-10} AND {strike_value+10} AND time = '{entry_time}')")
        
        condition_sql = " OR ".join(premium_conditions)
        
        query = f"""
        /*+ cpu_mode=false, watchdog_max_size=0 */
        WITH premium_based AS (
            SELECT 
                trade_date,
                expiry_date,
                strike,
                time,
                ce_close,
                pe_close,
                ce_open, ce_high, ce_low, 
                pe_open, pe_high, pe_low,
                ROW_NUMBER() OVER (
                    PARTITION BY trade_date, expiry_date, time 
                    ORDER BY ABS(ce_close - pe_close)
                ) as premium_rank
            FROM {self.table_name}
            WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'
                AND symbol = '{index}'
                AND ({condition_sql})
        )
        SELECT * FROM premium_based
        WHERE premium_rank <= 3  -- Top 3 closest premium matches per day
        ORDER BY trade_date, expiry_date, premium_rank
        """
        
        return query
        
    def _build_comprehensive_pnl_query(self, legs: List[Dict[str, Any]], 
                                     start_date: date, end_date: date, index: str,
                                     portfolio_params: Dict[str, Any]) -> str:
        """Build comprehensive P&L calculation query for all legs"""
        
        # Generate leg-specific P&L calculations
        leg_pnl_selects = []
        
        for i, leg in enumerate(legs):
            option_type = leg.get('option_type', 'CE')
            transaction_type = leg.get('transaction_type', 'SELL')
            quantity = leg.get('quantity', 1)
            entry_time = leg.get('entry_time', time(9, 20))
            exit_time = leg.get('exit_time', time(15, 15))
            
            # Price columns based on option type
            if option_type == 'CE':
                entry_price_col = 'ce_open'
                exit_price_col = 'ce_close'
                high_col = 'ce_high'
                low_col = 'ce_low'
            elif option_type == 'PE':
                entry_price_col = 'pe_open'
                exit_price_col = 'pe_close'
                high_col = 'pe_high'
                low_col = 'pe_low'
            else:  # FUT
                entry_price_col = 'fut_open'
                exit_price_col = 'fut_close'
                high_col = 'fut_high'
                low_col = 'fut_low'
            
            # P&L calculation based on transaction type
            if transaction_type == 'SELL':
                pnl_calc = f"(entry.{entry_price_col} - exit.{exit_price_col}) * {quantity}"
            else:
                pnl_calc = f"(exit.{exit_price_col} - entry.{entry_price_col}) * {quantity}"
            
            leg_pnl_selects.append(f"""
            SELECT 
                '{leg.get('strategy_name', 'Unknown')}' as strategy_name,
                {i+1} as leg_no,
                '{option_type}' as option_type,
                '{transaction_type}' as transaction_type,
                entry.trade_date,
                entry.expiry_date,
                entry.strike,
                entry.{entry_price_col} as entry_price,
                exit.{exit_price_col} as exit_price,
                exit.time as exit_time,
                {pnl_calc} as leg_pnl,
                -- SL/Target tracking
                CASE 
                    WHEN exit.{high_col} >= entry.{entry_price_col} * 1.5 THEN 'SL_HIT'
                    WHEN exit.{low_col} <= entry.{entry_price_col} * 0.5 THEN 'TARGET_HIT'
                    ELSE 'TIME_EXIT'
                END as exit_reason
            FROM {self.table_name} entry
            JOIN {self.table_name} exit
                ON entry.trade_date = exit.trade_date
                AND entry.expiry_date = exit.expiry_date  
                AND entry.strike = exit.strike
            WHERE entry.trade_date BETWEEN '{start_date}' AND '{end_date}'
                AND entry.symbol = '{index}'
                AND entry.time = '{entry_time}'
                AND exit.time = '{exit_time}'
                AND entry.{entry_price_col} > 0
                AND exit.{exit_price_col} > 0
            """)
        
        # Combine all leg P&L calculations
        combined_pnl_query = f"""
        /*+ cpu_mode=false, watchdog_max_size=0, group_by_buffer_size=2000000000 */
        WITH all_leg_pnls AS (
            {' UNION ALL '.join(leg_pnl_selects)}
        ),
        strategy_summary AS (
            SELECT 
                strategy_name,
                trade_date,
                COUNT(*) as total_legs,
                SUM(leg_pnl) as total_pnl,
                AVG(leg_pnl) as avg_leg_pnl,
                MIN(leg_pnl) as min_leg_pnl,
                MAX(leg_pnl) as max_leg_pnl,
                STDDEV(leg_pnl) as pnl_stddev,
                -- Entry/Exit timing
                MIN(CASE WHEN entry_price > 0 THEN entry_price END) as first_entry_price,
                MAX(CASE WHEN exit_price > 0 THEN exit_price END) as last_exit_price,
                -- Exit reason summary
                COUNT(CASE WHEN exit_reason = 'SL_HIT' THEN 1 END) as sl_exits,
                COUNT(CASE WHEN exit_reason = 'TARGET_HIT' THEN 1 END) as target_exits,
                COUNT(CASE WHEN exit_reason = 'TIME_EXIT' THEN 1 END) as time_exits
            FROM all_leg_pnls
            GROUP BY strategy_name, trade_date
        )
        SELECT 
            alp.*,
            ss.total_pnl as strategy_total_pnl,
            ss.avg_leg_pnl as strategy_avg_pnl,
            ss.pnl_stddev as strategy_pnl_stddev,
            ss.sl_exits,
            ss.target_exits,
            ss.time_exits,
            -- Performance metrics  
            CASE 
                WHEN ss.total_pnl > 0 THEN 'PROFIT'
                WHEN ss.total_pnl < 0 THEN 'LOSS'
                ELSE 'BREAKEVEN'
            END as trade_outcome,
            -- Risk metrics
            ABS(ss.min_leg_pnl) as max_leg_loss,
            ss.max_leg_pnl as max_leg_profit,
            CASE 
                WHEN ss.pnl_stddev > 0 THEN ss.total_pnl / ss.pnl_stddev 
                ELSE 0 
            END as risk_adjusted_return
        FROM all_leg_pnls alp
        LEFT JOIN strategy_summary ss 
            ON alp.strategy_name = ss.strategy_name 
            AND alp.trade_date = ss.trade_date
        ORDER BY strategy_name, trade_date, leg_no
        """
        
        return combined_pnl_query
        
    def _apply_gpu_optimizations(self, query: str) -> str:
        """Apply GPU-specific optimizations to query"""
        
        # Add GPU optimization hints
        gpu_hints = [
            "/*+ cpu_mode=false */",
            "/*+ watchdog_max_size=0 */", 
            "/*+ parallel_top_min=1000 */",
            "/*+ group_by_buffer_size=2000000000 */",
            "/*+ enable_columnar_output=true */",
            "/*+ enable_lazy_fetch=true */"
        ]
        
        # Check if query already has hints
        if "/*+" not in query:
            hint_string = " ".join(gpu_hints)
            query = f"{hint_string}\n{query}"
        
        # Add table-specific optimizations
        if self.table_name in query:
            # Add optimal column order for GPU processing
            query = query.replace(
                f"FROM {self.table_name}",
                f"FROM {self.table_name} /*+ use_hash_join=true */"
            )
        
        return query
        
    def _generate_cache_key(self, strategy_params: Dict[str, Any]) -> str:
        """Generate cache key for strategy parameters"""
        # Create deterministic hash of strategy parameters
        key_data = json.dumps(strategy_params, sort_keys=True, default=str)
        return hashlib.md5(key_data.encode()).hexdigest()
        
    def _get_atm_vectorized_template(self) -> str:
        """Vectorized ATM calculation template"""
        return """
        WITH synthetic_future AS (
            SELECT 
                trade_date, time, expiry_date, strike,
                strike + ce_close - pe_close as synthetic_price,
                ABS(strike - {spot_price}) as strike_diff
            FROM {table_name}
            WHERE {conditions}
        ),
        atm_strikes AS (
            SELECT 
                trade_date, time, expiry_date,
                strike as atm_strike,
                ROW_NUMBER() OVER (
                    PARTITION BY trade_date, time, expiry_date 
                    ORDER BY strike_diff
                ) as rn
            FROM synthetic_future
        )
        SELECT * FROM atm_strikes WHERE rn = 1
        """
        
    def _get_option_batch_template(self) -> str:
        """Batch option selection template"""
        return """
        SELECT 
            trade_date, expiry_date, strike, time,
            ce_open, ce_high, ce_low, ce_close,
            pe_open, pe_high, pe_low, pe_close,
            fut_open, fut_high, fut_low, fut_close,
            volume, open_interest
        FROM {table_name}
        WHERE {batch_conditions}
        ORDER BY trade_date, strike, time
        """
        
    def _get_entry_exit_streaming_template(self) -> str:
        """Streaming entry/exit processing template"""
        return """
        WITH entries AS (
            SELECT * FROM {table_name} 
            WHERE {entry_conditions}
        ),
        exits AS (
            SELECT * FROM {table_name}
            WHERE {exit_conditions}
        )
        SELECT 
            e.*, x.exit_price, x.exit_time,
            {pnl_calculation} as pnl
        FROM entries e
        JOIN exits x ON {join_conditions}
        """
        
    def _get_bulk_pnl_template(self) -> str:
        """Bulk P&L calculation template"""
        return """
        WITH trade_legs AS (
            {leg_unions}
        )
        SELECT 
            strategy_name, trade_date,
            SUM(leg_pnl) as total_pnl,
            COUNT(*) as total_legs,
            AVG(leg_pnl) as avg_pnl,
            STDDEV(leg_pnl) as pnl_stddev
        FROM trade_legs
        GROUP BY strategy_name, trade_date
        ORDER BY trade_date
        """
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get query builder performance metrics"""
        return {
            'queries_generated': self.query_metrics['queries_generated'],
            'cache_hits': self.query_metrics['cache_hits'],
            'cache_hit_rate': (
                self.query_metrics['cache_hits'] / max(1, self.query_metrics['queries_generated'])
            ) * 100,
            'total_execution_time': self.query_metrics['total_execution_time'],
            'avg_query_generation_time': (
                self.query_metrics['total_execution_time'] / 
                max(1, self.query_metrics['queries_generated'])
            ),
            'cache_size': len(self.query_cache)
        }
        
    def clear_cache(self):
        """Clear query cache"""
        self.query_cache.clear()
        logger.info("Query cache cleared")
        
    def optimize_for_dataset_size(self, estimated_rows: int):
        """Optimize query parameters based on dataset size"""
        if estimated_rows > 10_000_000:  # >10M rows
            self.gpu_optimization['group_by_buffer_size'] = 4_000_000_000  # 4GB
            self.gpu_optimization['parallel_top_min'] = 10000
        elif estimated_rows > 1_000_000:  # >1M rows  
            self.gpu_optimization['group_by_buffer_size'] = 2_000_000_000  # 2GB
            self.gpu_optimization['parallel_top_min'] = 5000
        else:
            self.gpu_optimization['group_by_buffer_size'] = 1_000_000_000  # 1GB
            self.gpu_optimization['parallel_top_min'] = 1000
            
        logger.info(f"Query optimization adjusted for {estimated_rows:,} rows")

# Backward compatibility
class TBSQueryBuilderOptimized(OptimizedTBSQueryBuilder):
    """Backward compatibility wrapper"""
    
    def build_queries(self, strategy_params: Dict[str, Any]) -> List[str]:
        """Wrapper for optimized query building"""
        return self.build_queries_optimized(strategy_params)