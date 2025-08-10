"""
Query Builder for ML Indicator Strategy
Generates optimized SQL queries for HeavyDB with indicator calculations
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import date, time, datetime, timedelta
import pandas as pd

from .models import (
    MLIndicatorStrategyModel,
    MLIndicatorPortfolioModel,
    MLLegModel,
    IndicatorConfig,
    SMCConfig,
    SignalCondition,
    IndicatorType,
    TALibIndicator,
    ComparisonOperator,
    SignalLogic,
    Timeframe,
    MLModelConfig,
    MLFeatureConfig
)
from .constants import (
    QUERY_TEMPLATES,
    DB_COLUMN_MAPPINGS,
    TALIB_PARAMS,
    TIMEFRAME_MINUTES,
    ERROR_MESSAGES
)

logger = logging.getLogger(__name__)


class MLIndicatorQueryBuilder:
    """Build optimized SQL queries for ML indicator strategies with HeavyDB integration"""

    def __init__(self, table_name: str = "nifty_option_chain"):
        self.table_name = table_name
        self.query_cache = {}
        self.connection_params = {
            'host': 'localhost',
            'port': 6274,
            'user': 'admin',
            'password': 'HyperInteractive',
            'dbname': 'heavyai'
        }
        
    def build_queries(self, strategy_model: MLIndicatorStrategyModel) -> List[str]:
        """
        Build all queries needed for the ML indicator strategy
        
        Args:
            strategy_model: Complete strategy configuration
            
        Returns:
            List of SQL queries to execute
        """
        queries = []
        
        try:
            # Generate base price data query
            base_query = self._build_base_price_query(strategy_model.portfolio)
            
            # Generate indicator calculation queries
            indicator_queries = self._build_indicator_queries(
                strategy_model.indicators,
                strategy_model.portfolio
            )
            
            # Generate SMC queries if configured
            if strategy_model.smc_config:
                smc_queries = self._build_smc_queries(
                    strategy_model.smc_config,
                    strategy_model.portfolio
                )
                indicator_queries.extend(smc_queries)
            
            # Generate signal evaluation query
            signal_query = self._build_signal_query(
                strategy_model.entry_signals,
                strategy_model.exit_signals,
                strategy_model.signal_logic,
                strategy_model.portfolio
            )
            
            # Generate multi-leg queries if configured
            if strategy_model.legs:
                leg_queries = self._build_multi_leg_queries(
                    strategy_model.legs,
                    strategy_model.portfolio
                )
                queries.extend(leg_queries)
            else:
                # Single instrument query
                main_query = self._combine_queries(
                    base_query,
                    indicator_queries,
                    signal_query
                )
                queries.append(main_query)
            
            # Generate ML feature query if ML model is configured
            if strategy_model.ml_config:
                ml_query = self._build_ml_feature_query(
                    strategy_model.ml_config,
                    strategy_model.ml_feature_config,
                    strategy_model.portfolio
                )
                queries.append(ml_query)
                
            return queries
            
        except Exception as e:
            logger.error(f"Error building queries: {str(e)}")
            raise ValueError(f"Query building failed: {str(e)}")
    
    def _build_base_price_query(self, portfolio: MLIndicatorPortfolioModel) -> str:
        """Build base query for price data"""
        
        query = f"""
        price_data AS (
            SELECT 
                trade_date,
                trade_time,
                CAST(trade_date AS TIMESTAMP) + CAST(trade_time AS TIME) as datetime,
                index_name,
                expiry_date,
                strike_price,
                option_type,
                open_price,
                high_price,
                low_price,
                close_price,
                volume,
                open_interest,
                underlying_value,
                -- Calculate returns
                (close_price - LAG(close_price, 1) OVER (ORDER BY trade_date, trade_time)) / 
                NULLIF(LAG(close_price, 1) OVER (ORDER BY trade_date, trade_time), 0) as returns,
                -- Calculate typical price
                (high_price + low_price + close_price) / 3 as typical_price,
                -- Calculate weighted close
                (high_price + low_price + 2 * close_price) / 4 as weighted_close,
                -- Row number for time-based calculations
                ROW_NUMBER() OVER (ORDER BY trade_date, trade_time) as row_num
            FROM {self.table_name}
            WHERE trade_date BETWEEN '{portfolio.start_date}' AND '{portfolio.end_date}'
            AND index_name = '{portfolio.index_name}'
            AND option_type = 'CE'  -- Default to calls for spot tracking
            AND strike_price = (
                -- Select ATM strike for underlying tracking
                SELECT strike_price
                FROM {self.table_name} t2
                WHERE t2.trade_date = {self.table_name}.trade_date
                AND t2.trade_time = {self.table_name}.trade_time
                AND t2.index_name = {self.table_name}.index_name
                AND t2.option_type = 'CE'
                ORDER BY ABS(t2.strike_price - t2.underlying_value)
                LIMIT 1
            )
            ORDER BY trade_date, trade_time
        )"""
        
        return query
    
    def _build_indicator_queries(self, 
                               indicators: List[IndicatorConfig],
                               portfolio: MLIndicatorPortfolioModel) -> List[str]:
        """Build queries for indicator calculations"""
        
        queries = []
        
        # Group indicators by timeframe for efficiency
        indicators_by_timeframe = {}
        for indicator in indicators:
            tf = indicator.timeframe
            if tf not in indicators_by_timeframe:
                indicators_by_timeframe[tf] = []
            indicators_by_timeframe[tf].append(indicator)
        
        # Build queries for each timeframe
        for timeframe, tf_indicators in indicators_by_timeframe.items():
            if timeframe == Timeframe.M1:
                # Use raw data
                tf_query = self._build_indicator_calculations(tf_indicators, "price_data")
            else:
                # Build resampled data
                resample_query = self._build_timeframe_resample(timeframe, portfolio)
                tf_query = self._build_indicator_calculations(
                    tf_indicators, 
                    f"tf_{timeframe.value}_data"
                )
                queries.append(resample_query)
                
            queries.append(tf_query)
            
        return queries
    
    def _build_timeframe_resample(self, 
                                 timeframe: Timeframe,
                                 portfolio: MLIndicatorPortfolioModel) -> str:
        """Build query to resample data to specific timeframe"""
        
        minutes = TIMEFRAME_MINUTES[timeframe.value]
        
        query = f"""
        tf_{timeframe.value}_data AS (
            SELECT 
                DATE_TRUNC('day', datetime) + 
                INTERVAL '{minutes} minutes' * (EXTRACT(EPOCH FROM datetime - DATE_TRUNC('day', datetime)) / {minutes * 60})::INT as bar_time,
                FIRST_VALUE(open_price) OVER w as open_price,
                MAX(high_price) OVER w as high_price,
                MIN(low_price) OVER w as low_price,
                LAST_VALUE(close_price) OVER w as close_price,
                SUM(volume) OVER w as volume,
                LAST_VALUE(underlying_value) OVER w as underlying_value,
                COUNT(*) OVER w as bar_count
            FROM price_data
            WINDOW w AS (
                PARTITION BY DATE_TRUNC('day', datetime) + 
                INTERVAL '{minutes} minutes' * (EXTRACT(EPOCH FROM datetime - DATE_TRUNC('day', datetime)) / {minutes * 60})::INT
                ORDER BY datetime
                ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
            )
        )"""
        
        return query
    
    def _build_indicator_calculations(self,
                                    indicators: List[IndicatorConfig],
                                    source_table: str) -> str:
        """Build indicator calculation queries"""
        
        calculations = []
        
        for indicator in indicators:
            if indicator.indicator_type == IndicatorType.TALIB:
                calc = self._build_talib_indicator(indicator, source_table)
            elif indicator.indicator_type == IndicatorType.VOLUME:
                calc = self._build_volume_indicator(indicator, source_table)
            elif indicator.indicator_type == IndicatorType.CUSTOM:
                calc = self._build_custom_indicator(indicator, source_table)
            else:
                continue
                
            calculations.append(calc)
            
        query = f"""
        indicators_{source_table} AS (
            SELECT 
                *,
                {','.join(calculations)}
            FROM {source_table}
        )"""
        
        return query
    
    def _build_talib_indicator(self, 
                             indicator: IndicatorConfig,
                             source_table: str) -> str:
        """Build SQL for TA-Lib indicator calculation"""
        
        indicator_name = indicator.indicator_name.upper()
        params = indicator.parameters
        
        # Map TA-Lib indicators to SQL window functions
        if indicator_name == "SMA":
            period = params.get('timeperiod', 20)
            calc = f"""
                AVG(close_price) OVER (
                    ORDER BY row_num 
                    ROWS BETWEEN {period - 1} PRECEDING AND CURRENT ROW
                ) as {indicator_name}_{period}"""
                
        elif indicator_name == "EMA":
            period = params.get('timeperiod', 20)
            alpha = 2.0 / (period + 1)
            # Simplified EMA using recursive calculation
            calc = f"""
                CASE 
                    WHEN row_num < {period} THEN NULL
                    WHEN row_num = {period} THEN AVG(close_price) OVER (
                        ORDER BY row_num ROWS BETWEEN {period - 1} PRECEDING AND CURRENT ROW
                    )
                    ELSE {alpha} * close_price + (1 - {alpha}) * LAG({indicator_name}_{period}, 1) OVER (ORDER BY row_num)
                END as {indicator_name}_{period}"""
                
        elif indicator_name == "RSI":
            period = params.get('timeperiod', 14)
            calc = f"""
                100 - (100 / (1 + (
                    AVG(CASE WHEN returns > 0 THEN returns ELSE 0 END) OVER (
                        ORDER BY row_num ROWS BETWEEN {period - 1} PRECEDING AND CURRENT ROW
                    ) / NULLIF(
                    AVG(CASE WHEN returns < 0 THEN ABS(returns) ELSE 0 END) OVER (
                        ORDER BY row_num ROWS BETWEEN {period - 1} PRECEDING AND CURRENT ROW
                    ), 0)
                ))) as {indicator_name}_{period}"""
                
        elif indicator_name == "BBANDS":
            period = params.get('timeperiod', 20)
            nbdev = params.get('nbdevup', 2)
            calc = f"""
                AVG(close_price) OVER (
                    ORDER BY row_num ROWS BETWEEN {period - 1} PRECEDING AND CURRENT ROW
                ) as BB_MIDDLE_{period},
                AVG(close_price) OVER (
                    ORDER BY row_num ROWS BETWEEN {period - 1} PRECEDING AND CURRENT ROW
                ) + {nbdev} * STDDEV(close_price) OVER (
                    ORDER BY row_num ROWS BETWEEN {period - 1} PRECEDING AND CURRENT ROW
                ) as BB_UPPER_{period},
                AVG(close_price) OVER (
                    ORDER BY row_num ROWS BETWEEN {period - 1} PRECEDING AND CURRENT ROW
                ) - {nbdev} * STDDEV(close_price) OVER (
                    ORDER BY row_num ROWS BETWEEN {period - 1} PRECEDING AND CURRENT ROW
                ) as BB_LOWER_{period}"""
                
        elif indicator_name == "MACD":
            fast = params.get('fastperiod', 12)
            slow = params.get('slowperiod', 26)
            signal = params.get('signalperiod', 9)
            # Simplified MACD
            calc = f"""
                EMA_{fast} - EMA_{slow} as MACD,
                AVG(EMA_{fast} - EMA_{slow}) OVER (
                    ORDER BY row_num ROWS BETWEEN {signal - 1} PRECEDING AND CURRENT ROW
                ) as MACD_SIGNAL,
                (EMA_{fast} - EMA_{slow}) - AVG(EMA_{fast} - EMA_{slow}) OVER (
                    ORDER BY row_num ROWS BETWEEN {signal - 1} PRECEDING AND CURRENT ROW
                ) as MACD_HIST"""
                
        elif indicator_name == "ATR":
            period = params.get('timeperiod', 14)
            calc = f"""
                AVG(GREATEST(
                    high_price - low_price,
                    ABS(high_price - LAG(close_price, 1) OVER (ORDER BY row_num)),
                    ABS(low_price - LAG(close_price, 1) OVER (ORDER BY row_num))
                )) OVER (
                    ORDER BY row_num ROWS BETWEEN {period - 1} PRECEDING AND CURRENT ROW
                ) as {indicator_name}_{period}"""
                
        elif indicator_name == "ADX":
            period = params.get('timeperiod', 14)
            # Simplified ADX
            calc = f"""
                -- This is a simplified version
                AVG(ABS(
                    (high_price - LAG(high_price, 1) OVER (ORDER BY row_num)) - 
                    (LAG(low_price, 1) OVER (ORDER BY row_num) - low_price)
                ) / NULLIF(
                    (high_price - LAG(high_price, 1) OVER (ORDER BY row_num)) + 
                    (LAG(low_price, 1) OVER (ORDER BY row_num) - low_price), 0
                )) OVER (
                    ORDER BY row_num ROWS BETWEEN {period - 1} PRECEDING AND CURRENT ROW
                ) * 100 as {indicator_name}_{period}"""
                
        elif indicator_name == "CCI":
            period = params.get('timeperiod', 20)
            calc = f"""
                (typical_price - AVG(typical_price) OVER (
                    ORDER BY row_num ROWS BETWEEN {period - 1} PRECEDING AND CURRENT ROW
                )) / (0.015 * AVG(ABS(typical_price - AVG(typical_price) OVER (
                    ORDER BY row_num ROWS BETWEEN {period - 1} PRECEDING AND CURRENT ROW
                ))) OVER (
                    ORDER BY row_num ROWS BETWEEN {period - 1} PRECEDING AND CURRENT ROW
                )) as {indicator_name}_{period}"""
                
        elif indicator_name == "MFI":
            period = params.get('timeperiod', 14)
            calc = f"""
                100 - (100 / (1 + 
                    SUM(CASE WHEN typical_price > LAG(typical_price, 1) OVER (ORDER BY row_num) 
                        THEN typical_price * volume ELSE 0 END) OVER (
                        ORDER BY row_num ROWS BETWEEN {period - 1} PRECEDING AND CURRENT ROW
                    ) / NULLIF(
                    SUM(CASE WHEN typical_price < LAG(typical_price, 1) OVER (ORDER BY row_num) 
                        THEN typical_price * volume ELSE 0 END) OVER (
                        ORDER BY row_num ROWS BETWEEN {period - 1} PRECEDING AND CURRENT ROW
                    ), 0)
                )) as {indicator_name}_{period}"""
                
        elif indicator_name == "STOCH":
            fastk = params.get('fastk_period', 5)
            slowk = params.get('slowk_period', 3)
            slowd = params.get('slowd_period', 3)
            calc = f"""
                ((close_price - MIN(low_price) OVER (
                    ORDER BY row_num ROWS BETWEEN {fastk - 1} PRECEDING AND CURRENT ROW
                )) / NULLIF(MAX(high_price) OVER (
                    ORDER BY row_num ROWS BETWEEN {fastk - 1} PRECEDING AND CURRENT ROW
                ) - MIN(low_price) OVER (
                    ORDER BY row_num ROWS BETWEEN {fastk - 1} PRECEDING AND CURRENT ROW
                ), 0)) * 100 as STOCH_K,
                AVG(((close_price - MIN(low_price) OVER (
                    ORDER BY row_num ROWS BETWEEN {fastk - 1} PRECEDING AND CURRENT ROW
                )) / NULLIF(MAX(high_price) OVER (
                    ORDER BY row_num ROWS BETWEEN {fastk - 1} PRECEDING AND CURRENT ROW
                ) - MIN(low_price) OVER (
                    ORDER BY row_num ROWS BETWEEN {fastk - 1} PRECEDING AND CURRENT ROW
                ), 0)) * 100) OVER (
                    ORDER BY row_num ROWS BETWEEN {slowd - 1} PRECEDING AND CURRENT ROW
                ) as STOCH_D"""
        else:
            # Default to SMA if indicator not implemented
            calc = f"""
                AVG(close_price) OVER (
                    ORDER BY row_num ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                ) as {indicator_name}_DEFAULT"""
                
        return calc
    
    def _build_volume_indicator(self,
                              indicator: IndicatorConfig,
                              source_table: str) -> str:
        """Build SQL for volume-based indicators"""
        
        indicator_name = indicator.indicator_name.upper()
        params = indicator.parameters
        
        if indicator_name == "OBV":
            calc = f"""
                SUM(CASE 
                    WHEN close_price > LAG(close_price, 1) OVER (ORDER BY row_num) THEN volume
                    WHEN close_price < LAG(close_price, 1) OVER (ORDER BY row_num) THEN -volume
                    ELSE 0
                END) OVER (ORDER BY row_num) as OBV"""
                
        elif indicator_name == "VOLUME_PROFILE":
            bins = params.get('bins', 24)
            calc = f"""
                -- Volume at price level
                SUM(volume) OVER (
                    PARTITION BY ROUND(close_price / 10) * 10
                    ORDER BY row_num
                ) as VOLUME_AT_PRICE"""
                
        elif indicator_name == "CVD":
            # Cumulative Volume Delta
            calc = f"""
                SUM(CASE 
                    WHEN close_price > (high_price + low_price) / 2 THEN volume
                    ELSE -volume
                END) OVER (ORDER BY row_num) as CVD"""
                
        else:
            calc = f"volume as {indicator_name}"
            
        return calc
    
    def _build_custom_indicator(self,
                              indicator: IndicatorConfig,
                              source_table: str) -> str:
        """Build SQL for custom indicators"""
        
        # Placeholder for custom indicator logic
        return f"0 as {indicator.indicator_name}_CUSTOM"
    
    def _build_smc_queries(self,
                         smc_config: SMCConfig,
                         portfolio: MLIndicatorPortfolioModel) -> List[str]:
        """Build queries for Smart Money Concepts"""
        
        queries = []
        
        if smc_config.detect_bos:
            queries.append(self._build_bos_query(smc_config, portfolio))
            
        if smc_config.detect_order_blocks:
            queries.append(self._build_order_block_query(smc_config, portfolio))
            
        if smc_config.detect_fvg:
            queries.append(self._build_fvg_query(smc_config, portfolio))
            
        if smc_config.detect_liquidity:
            queries.append(self._build_liquidity_query(smc_config, portfolio))
            
        return queries
    
    def _build_bos_query(self, smc_config: SMCConfig, portfolio: MLIndicatorPortfolioModel) -> str:
        """Build query for Break of Structure detection"""
        
        lookback = smc_config.structure_lookback
        
        query = f"""
        bos_detection AS (
            SELECT 
                *,
                -- Detect swing highs
                CASE 
                    WHEN high_price = MAX(high_price) OVER (
                        ORDER BY row_num ROWS BETWEEN {lookback} PRECEDING AND {lookback} FOLLOWING
                    ) THEN 1 ELSE 0 
                END as swing_high,
                -- Detect swing lows
                CASE 
                    WHEN low_price = MIN(low_price) OVER (
                        ORDER BY row_num ROWS BETWEEN {lookback} PRECEDING AND {lookback} FOLLOWING
                    ) THEN 1 ELSE 0 
                END as swing_low,
                -- Previous swing high
                MAX(CASE WHEN swing_high = 1 THEN high_price END) OVER (
                    ORDER BY row_num ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) as prev_swing_high,
                -- Previous swing low
                MIN(CASE WHEN swing_low = 1 THEN low_price END) OVER (
                    ORDER BY row_num ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) as prev_swing_low,
                -- BOS detection
                CASE 
                    WHEN close_price > prev_swing_high AND LAG(close_price, 1) OVER (ORDER BY row_num) <= prev_swing_high THEN 1
                    WHEN close_price < prev_swing_low AND LAG(close_price, 1) OVER (ORDER BY row_num) >= prev_swing_low THEN -1
                    ELSE 0
                END as bos_signal
            FROM indicators_price_data
        )"""
        
        return query
    
    def _build_order_block_query(self, smc_config: SMCConfig, portfolio: MLIndicatorPortfolioModel) -> str:
        """Build query for Order Block detection"""
        
        lookback = smc_config.order_block_lookback
        volume_mult = smc_config.parameters.get('min_volume_multiplier', 1.5)
        
        query = f"""
        order_blocks AS (
            SELECT 
                *,
                -- Detect potential order blocks (high volume + reversal)
                CASE 
                    WHEN volume > AVG(volume) OVER (
                        ORDER BY row_num ROWS BETWEEN {lookback} PRECEDING AND CURRENT ROW
                    ) * {volume_mult}
                    AND (
                        -- Bullish order block
                        (close_price > open_price AND 
                         LEAD(low_price, 1) OVER (ORDER BY row_num) > high_price)
                        OR
                        -- Bearish order block
                        (close_price < open_price AND 
                         LEAD(high_price, 1) OVER (ORDER BY row_num) < low_price)
                    )
                    THEN 1 ELSE 0
                END as is_order_block,
                -- Order block type
                CASE 
                    WHEN is_order_block = 1 AND close_price > open_price THEN 'BULLISH'
                    WHEN is_order_block = 1 AND close_price < open_price THEN 'BEARISH'
                    ELSE NULL
                END as order_block_type,
                -- Order block level (mid-point)
                (high_price + low_price) / 2 as order_block_level
            FROM indicators_price_data
        )"""
        
        return query
    
    def _build_fvg_query(self, smc_config: SMCConfig, portfolio: MLIndicatorPortfolioModel) -> str:
        """Build query for Fair Value Gap detection"""
        
        min_gap_size = smc_config.fvg_min_size
        
        query = f"""
        fvg_detection AS (
            SELECT 
                *,
                -- Detect Fair Value Gaps
                CASE 
                    -- Bullish FVG (BISI)
                    WHEN LAG(high_price, 2) OVER (ORDER BY row_num) < low_price 
                    AND (low_price - LAG(high_price, 2) OVER (ORDER BY row_num)) / low_price > {min_gap_size}
                    THEN 1
                    -- Bearish FVG (SIBI)
                    WHEN LAG(low_price, 2) OVER (ORDER BY row_num) > high_price
                    AND (LAG(low_price, 2) OVER (ORDER BY row_num) - high_price) / high_price > {min_gap_size}
                    THEN -1
                    ELSE 0
                END as fvg_signal,
                -- FVG boundaries
                CASE 
                    WHEN fvg_signal = 1 THEN LAG(high_price, 2) OVER (ORDER BY row_num)
                    WHEN fvg_signal = -1 THEN LAG(low_price, 2) OVER (ORDER BY row_num)
                    ELSE NULL
                END as fvg_start,
                CASE 
                    WHEN fvg_signal = 1 THEN low_price
                    WHEN fvg_signal = -1 THEN high_price
                    ELSE NULL
                END as fvg_end
            FROM indicators_price_data
        )"""
        
        return query
    
    def _build_liquidity_query(self, smc_config: SMCConfig, portfolio: MLIndicatorPortfolioModel) -> str:
        """Build query for liquidity pool detection"""
        
        lookback = smc_config.structure_lookback
        threshold = smc_config.liquidity_threshold
        
        query = f"""
        liquidity_pools AS (
            SELECT 
                *,
                -- Detect liquidity pools (multiple touches at same level)
                COUNT(*) OVER (
                    PARTITION BY ROUND(high_price / 10) * 10
                    ORDER BY row_num
                    ROWS BETWEEN {lookback} PRECEDING AND CURRENT ROW
                ) as high_touches,
                COUNT(*) OVER (
                    PARTITION BY ROUND(low_price / 10) * 10
                    ORDER BY row_num
                    ROWS BETWEEN {lookback} PRECEDING AND CURRENT ROW
                ) as low_touches,
                -- Liquidity grab detection
                CASE 
                    WHEN high_price > MAX(high_price) OVER (
                        ORDER BY row_num ROWS BETWEEN {lookback} PRECEDING AND 1 PRECEDING
                    ) AND close_price < open_price
                    THEN 1  -- Sell-side liquidity grab
                    WHEN low_price < MIN(low_price) OVER (
                        ORDER BY row_num ROWS BETWEEN {lookback} PRECEDING AND 1 PRECEDING
                    ) AND close_price > open_price
                    THEN -1  -- Buy-side liquidity grab
                    ELSE 0
                END as liquidity_grab
            FROM indicators_price_data
        )"""
        
        return query
    
    def _build_signal_query(self,
                          entry_signals: List[SignalCondition],
                          exit_signals: List[SignalCondition],
                          signal_logic: SignalLogic,
                          portfolio: MLIndicatorPortfolioModel) -> str:
        """Build query for signal evaluation"""
        
        # Build entry signal conditions
        entry_conditions = []
        for signal in entry_signals:
            condition = self._build_signal_condition(signal)
            entry_conditions.append(condition)
            
        # Build exit signal conditions
        exit_conditions = []
        for signal in exit_signals:
            condition = self._build_signal_condition(signal)
            exit_conditions.append(condition)
            
        # Combine based on logic
        if signal_logic == SignalLogic.AND:
            entry_logic = " AND ".join(entry_conditions) if entry_conditions else "FALSE"
            exit_logic = " AND ".join(exit_conditions) if exit_conditions else "FALSE"
        elif signal_logic == SignalLogic.OR:
            entry_logic = " OR ".join(entry_conditions) if entry_conditions else "FALSE"
            exit_logic = " OR ".join(exit_conditions) if exit_conditions else "FALSE"
        else:
            # Weighted or ML-based logic
            entry_logic = self._build_weighted_signal(entry_signals)
            exit_logic = self._build_weighted_signal(exit_signals)
            
        query = f"""
        signals AS (
            SELECT 
                *,
                CASE WHEN {entry_logic} THEN 1 ELSE 0 END as entry_signal,
                CASE WHEN {exit_logic} THEN 1 ELSE 0 END as exit_signal,
                -- Signal strength for weighted logic
                {self._build_signal_strength(entry_signals)} as entry_strength,
                {self._build_signal_strength(exit_signals)} as exit_strength
            FROM indicators_price_data
        )"""
        
        return query
    
    def _build_signal_condition(self, signal: SignalCondition) -> str:
        """Build SQL condition for a signal"""
        
        indicator = signal.indicator_name
        operator = signal.condition_type
        
        if operator == ComparisonOperator.GREATER:
            return f"{indicator} > {signal.threshold_value}"
        elif operator == ComparisonOperator.LESS:
            return f"{indicator} < {signal.threshold_value}"
        elif operator == ComparisonOperator.EQUAL:
            return f"{indicator} = {signal.threshold_value}"
        elif operator == ComparisonOperator.CROSSES_ABOVE:
            return f"""({indicator} > {signal.threshold_value} AND 
                       LAG({indicator}, 1) OVER (ORDER BY row_num) <= {signal.threshold_value})"""
        elif operator == ComparisonOperator.CROSSES_BELOW:
            return f"""({indicator} < {signal.threshold_value} AND 
                       LAG({indicator}, 1) OVER (ORDER BY row_num) >= {signal.threshold_value})"""
        elif operator == ComparisonOperator.BETWEEN:
            return f"{indicator} BETWEEN {signal.threshold_value} AND {signal.secondary_value}"
        elif operator == ComparisonOperator.OUTSIDE:
            return f"{indicator} NOT BETWEEN {signal.threshold_value} AND {signal.secondary_value}"
        else:
            return "FALSE"
    
    def _build_weighted_signal(self, signals: List[SignalCondition]) -> str:
        """Build weighted signal combination"""
        
        weighted_conditions = []
        total_weight = sum(s.weight for s in signals)
        
        for signal in signals:
            condition = self._build_signal_condition(signal)
            weighted = f"(CASE WHEN {condition} THEN {signal.weight} ELSE 0 END)"
            weighted_conditions.append(weighted)
            
        if weighted_conditions:
            return f"({' + '.join(weighted_conditions)}) / {total_weight} > 0.5"
        else:
            return "FALSE"
    
    def _build_signal_strength(self, signals: List[SignalCondition]) -> str:
        """Calculate signal strength"""
        
        if not signals:
            return "0"
            
        strength_calcs = []
        for signal in signals:
            condition = self._build_signal_condition(signal)
            strength_calcs.append(f"CASE WHEN {condition} THEN {signal.weight} ELSE 0 END")
            
        return f"({' + '.join(strength_calcs)}) / {sum(s.weight for s in signals)}"
    
    def _build_multi_leg_queries(self,
                               legs: List[MLLegModel],
                               portfolio: MLIndicatorPortfolioModel) -> List[str]:
        """Build queries for multi-leg ML strategies"""
        
        queries = []
        
        # Build individual leg queries
        for leg in legs:
            leg_query = self._build_ml_leg_query(leg, portfolio)
            queries.append(leg_query)
            
        # Build portfolio combination query
        portfolio_query = self._build_ml_portfolio_query(legs, portfolio)
        queries.append(portfolio_query)
        
        return queries
    
    def _build_ml_leg_query(self, leg: MLLegModel, portfolio: MLIndicatorPortfolioModel) -> str:
        """Build query for individual ML leg"""
        
        query = f"""
        SELECT 
            s.trade_date,
            s.trade_time,
            s.entry_signal,
            s.exit_signal,
            s.entry_strength,
            l.strike_price,
            l.option_type,
            l.close_price,
            l.volume,
            l.open_interest,
            l.delta,
            l.gamma,
            l.theta,
            l.vega,
            {leg.lots * leg.lot_size} as position_size,
            '{leg.position_type}' as position_type
        FROM signals s
        INNER JOIN {self.table_name} l
            ON s.trade_date = l.trade_date
            AND s.trade_time = l.trade_time
        WHERE l.option_type = '{leg.option_type}'
        AND l.strike_price = (
            -- Strike selection logic
            {self._build_ml_strike_selection(leg)}
        )
        AND s.trade_time >= '{leg.entry_time_start}'
        AND s.trade_time <= '{leg.entry_time_end}'
        AND s.entry_strength >= {leg.min_signal_confidence}
        """
        
        return query
    
    def _build_ml_strike_selection(self, leg: MLLegModel) -> str:
        """Build strike selection for ML leg"""
        
        if leg.strike_selection == "ATM":
            offset = leg.strike_offset or 0
            return f"""
                SELECT strike_price
                FROM {self.table_name} t2
                WHERE t2.trade_date = l.trade_date
                AND t2.trade_time = l.trade_time
                AND t2.option_type = l.option_type
                ORDER BY ABS(t2.strike_price - (t2.underlying_value + {offset}))
                LIMIT 1
            """
        else:
            # Add other strike selection methods
            return "l.strike_price"
    
    def _build_ml_portfolio_query(self, legs: List[MLLegModel], portfolio: MLIndicatorPortfolioModel) -> str:
        """Build portfolio combination query for ML strategy"""
        
        leg_joins = []
        leg_fields = []
        
        for i, leg in enumerate(legs):
            if i == 0:
                # First leg is the base
                continue
            else:
                leg_joins.append(f"""
                    LEFT JOIN ml_leg_{leg.leg_id} l{leg.leg_id}
                        ON l1.trade_date = l{leg.leg_id}.trade_date
                        AND l1.trade_time = l{leg.leg_id}.trade_time
                """)
                
            leg_fields.extend([
                f"l{leg.leg_id}.strike_price as leg_{leg.leg_id}_strike",
                f"l{leg.leg_id}.close_price as leg_{leg.leg_id}_price",
                f"l{leg.leg_id}.position_size as leg_{leg.leg_id}_size"
            ])
            
        query = f"""
        SELECT 
            l1.trade_date,
            l1.trade_time,
            l1.entry_signal,
            l1.exit_signal,
            l1.strike_price as leg_1_strike,
            l1.close_price as leg_1_price,
            l1.position_size as leg_1_size,
            {','.join(leg_fields) if leg_fields else '1 as dummy'}
        FROM ml_leg_1 l1
        {' '.join(leg_joins)}
        WHERE l1.entry_signal = 1 OR l1.exit_signal = 1
        """
        
        return query
    
    def _build_ml_feature_query(self,
                              ml_config: MLModelConfig,
                              feature_config: MLFeatureConfig,
                              portfolio: MLIndicatorPortfolioModel) -> str:
        """Build query for ML feature extraction"""
        
        # This would be used for model training/prediction
        feature_calcs = []
        
        if feature_config.price_features:
            for period in feature_config.return_periods:
                feature_calcs.append(f"""
                    (close_price - LAG(close_price, {period}) OVER (ORDER BY row_num)) / 
                    NULLIF(LAG(close_price, {period}) OVER (ORDER BY row_num), 0) as return_{period}
                """)
                
        query = f"""
        ml_features AS (
            SELECT 
                trade_date,
                trade_time,
                {','.join(feature_calcs)},
                -- Target variable (future return)
                LEAD((close_price - LAG(close_price, 1) OVER (ORDER BY row_num)) / 
                     NULLIF(LAG(close_price, 1) OVER (ORDER BY row_num), 0), 
                     {ml_config.prediction_horizon}) OVER (ORDER BY row_num) as target
            FROM indicators_price_data
        )
        SELECT * FROM ml_features
        WHERE target IS NOT NULL
        """
        
        return query
    
    def _combine_queries(self, base_query: str, indicator_queries: List[str], signal_query: str) -> str:
        """Combine all sub-queries into final query"""
        
        # Build WITH clause
        with_clauses = [base_query]
        with_clauses.extend(indicator_queries)
        with_clauses.append(signal_query)
        
        # Remove "WITH" from subsequent queries
        combined_with = "WITH " + ",\n".join(
            clause.replace("WITH ", "") for clause in with_clauses
        )
        
        # Build final SELECT
        final_query = f"""
        {combined_with}
        SELECT * FROM signals
        ORDER BY trade_date, trade_time
        """
        
        return self._optimize_query(final_query)
    
    def _optimize_query(self, query: str) -> str:
        """Optimize query for GPU execution"""

        # Remove extra whitespace
        query = ' '.join(query.split())

        # Add GPU hints
        query = f"/*+ cpu_mode=false, watchdog_max_size=0 */ {query}"

        return query

    def build_market_data_query(self, strategy_model: MLIndicatorStrategyModel,
                               start_date: datetime, end_date: datetime) -> str:
        """Build market data query for ML analysis"""
        symbols = [leg.symbol for leg in strategy_model.legs]
        symbol_filter = "', '".join(symbols)

        query = f"""
        SELECT
            timestamp,
            symbol,
            open_price,
            high_price,
            low_price,
            close_price,
            volume,
            option_type,
            strike_price,
            expiry_date
        FROM {self.table_name}
        WHERE symbol IN ('{symbol_filter}')
        AND timestamp BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY timestamp, symbol
        """

        return self._optimize_query(query)

    def build_option_chain_query(self, symbol: str, expiry_date: datetime,
                                strike_range: Optional[tuple] = None) -> str:
        """Build option chain query"""
        query = f"""
        SELECT
            timestamp,
            symbol,
            option_type,
            strike_price,
            expiry_date,
            open_price,
            high_price,
            low_price,
            close_price,
            volume,
            open_interest
        FROM {self.table_name}
        WHERE symbol = '{symbol}'
        AND expiry_date = '{expiry_date.date()}'
        """

        if strike_range:
            query += f" AND strike_price BETWEEN {strike_range[0]} AND {strike_range[1]}"

        query += " ORDER BY strike_price, option_type"

        return self._optimize_query(query)

    def build_indicators_query(self, strategy_model: MLIndicatorStrategyModel,
                              start_date: datetime, end_date: datetime) -> str:
        """Build indicators query for ML training"""
        symbols = [leg.symbol for leg in strategy_model.legs]
        symbol_filter = "', '".join(symbols)

        query = f"""
        SELECT
            timestamp,
            symbol,
            close_price,
            volume,
            LAG(close_price, 1) OVER (PARTITION BY symbol ORDER BY timestamp) as prev_close,
            LAG(close_price, 14) OVER (PARTITION BY symbol ORDER BY timestamp) as close_14d_ago,
            AVG(close_price) OVER (PARTITION BY symbol ORDER BY timestamp ROWS 13 PRECEDING) as sma_14,
            AVG(close_price) OVER (PARTITION BY symbol ORDER BY timestamp ROWS 49 PRECEDING) as sma_50,
            AVG(volume) OVER (PARTITION BY symbol ORDER BY timestamp ROWS 19 PRECEDING) as vol_avg_20
        FROM {self.table_name}
        WHERE symbol IN ('{symbol_filter}')
        AND timestamp BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY timestamp, symbol
        """

        return self._optimize_query(query)

    def build_optimized_market_data_query(self, strategy_model: MLIndicatorStrategyModel) -> str:
        """Build optimized market data query with performance enhancements"""
        symbols = [leg.symbol for leg in strategy_model.legs]
        symbol_filter = "', '".join(symbols)

        query = f"""
        SELECT /*+ USE_COLUMNAR_SCAN */
            timestamp,
            symbol,
            close_price,
            volume
        FROM {self.table_name}
        WHERE symbol IN ('{symbol_filter}')
        AND timestamp >= CURRENT_DATE - INTERVAL '30' DAY
        ORDER BY timestamp DESC
        LIMIT 10000
        """

        return self._optimize_query(query)

    def build_optimized_indicators_query(self, strategy_model: MLIndicatorStrategyModel) -> str:
        """Build optimized indicators query"""
        symbols = [leg.symbol for leg in strategy_model.legs]
        symbol_filter = "', '".join(symbols)

        query = f"""
        SELECT /*+ USE_COLUMNAR_SCAN */
            symbol,
            timestamp,
            close_price,
            AVG(close_price) OVER w14 as sma_14,
            AVG(close_price) OVER w50 as sma_50,
            STDDEV(close_price) OVER w20 as volatility_20
        FROM {self.table_name}
        WHERE symbol IN ('{symbol_filter}')
        AND timestamp >= CURRENT_DATE - INTERVAL '60' DAY
        WINDOW
            w14 AS (PARTITION BY symbol ORDER BY timestamp ROWS 13 PRECEDING),
            w50 AS (PARTITION BY symbol ORDER BY timestamp ROWS 49 PRECEDING),
            w20 AS (PARTITION BY symbol ORDER BY timestamp ROWS 19 PRECEDING)
        ORDER BY timestamp DESC
        """

        return self._optimize_query(query)

    def build_optimized_option_chain_query(self, strategy_model: MLIndicatorStrategyModel) -> str:
        """Build optimized option chain query"""
        symbols = [leg.symbol for leg in strategy_model.legs]
        symbol_filter = "', '".join(symbols)

        query = f"""
        SELECT /*+ USE_COLUMNAR_SCAN */
            symbol,
            option_type,
            strike_price,
            close_price,
            volume,
            open_interest
        FROM {self.table_name}
        WHERE symbol IN ('{symbol_filter}')
        AND expiry_date = (
            SELECT MIN(expiry_date)
            FROM {self.table_name}
            WHERE expiry_date > CURRENT_DATE
        )
        ORDER BY symbol, strike_price
        """

        return self._optimize_query(query)

    def build_signals_insert_query(self, signals) -> str:
        """Build query to insert ML signals"""
        if not signals:
            return ""

        values = []
        for signal in signals:
            values.append(f"('{signal.signal_id}', '{signal.timestamp}', '{signal.symbol}', '{signal.action}', {signal.quantity}, {signal.confidence})")

        query = f"""
        INSERT INTO ml_signals (signal_id, timestamp, symbol, action, quantity, confidence)
        VALUES {', '.join(values)}
        """

        return query

    def build_trades_insert_query(self, trades) -> str:
        """Build query to insert ML trades"""
        if not trades:
            return ""

        values = []
        for trade in trades:
            values.append(f"('{trade.trade_id}', '{trade.timestamp}', '{trade.symbol}', '{trade.action}', {trade.quantity}, {trade.price})")

        query = f"""
        INSERT INTO ml_trades (trade_id, timestamp, symbol, action, quantity, price)
        VALUES {', '.join(values)}
        """

        return query