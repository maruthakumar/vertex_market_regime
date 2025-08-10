"""
Market Regime Strategy Query Builder
Comprehensive HeavyDB query builder for market regime analysis and 18-regime classification

This module provides optimized SQL query generation for:
- Market regime detection and classification
- 18-regime system (Volatility × Trend × Structure)
- Correlation matrix analysis
- Option chain data processing
- Real-time regime monitoring

Author: The Augster
Date: 2025-01-19
Framework: SuperClaude v3 Enhanced HeavyDB Integration
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RegimeQueryConfig:
    """Configuration for regime analysis queries"""
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    timeframes: List[str] = None
    volatility_window: int = 20
    trend_window: int = 50
    correlation_window: int = 30
    
    def __post_init__(self):
        if self.timeframes is None:
            self.timeframes = ['1m', '5m', '15m', '1h', '1d']


class MarketRegimeQueryBuilder:
    """
    Comprehensive HeavyDB query builder for Market Regime Strategy
    
    Provides optimized SQL queries for 18-regime classification system
    and comprehensive market regime analysis with GPU acceleration
    """
    
    def __init__(self, table_name: str = "nifty_option_chain"):
        self.table_name = table_name
        self.connection_params = {
            'host': 'localhost',
            'port': 6274,
            'user': 'admin',
            'password': 'HyperInteractive',
            'dbname': 'heavyai'
        }
        self.query_cache = {}
    
    def build_regime_detection_query(self, config: RegimeQueryConfig) -> str:
        """
        Build comprehensive regime detection query for 18-regime classification
        
        Args:
            config: Regime query configuration
            
        Returns:
            Optimized SQL query for regime detection
        """
        symbols_filter = "', '".join(config.symbols)
        
        query = f"""
        WITH market_data AS (
            SELECT /*+ USE_COLUMNAR_SCAN */
                timestamp,
                symbol,
                close_price,
                volume,
                open_interest,
                LAG(close_price, 1) OVER (PARTITION BY symbol ORDER BY timestamp) as prev_close,
                LAG(close_price, {config.volatility_window}) OVER (PARTITION BY symbol ORDER BY timestamp) as close_vol_window,
                LAG(close_price, {config.trend_window}) OVER (PARTITION BY symbol ORDER BY timestamp) as close_trend_window
            FROM {self.table_name}
            WHERE symbol IN ('{symbols_filter}')
            AND timestamp BETWEEN '{config.start_date}' AND '{config.end_date}'
            ORDER BY timestamp, symbol
        ),
        volatility_analysis AS (
            SELECT 
                timestamp,
                symbol,
                close_price,
                volume,
                open_interest,
                prev_close,
                -- Volatility calculation (20-period rolling)
                STDDEV(close_price) OVER (
                    PARTITION BY symbol 
                    ORDER BY timestamp 
                    ROWS {config.volatility_window - 1} PRECEDING
                ) as volatility,
                -- Average True Range for volatility
                AVG(ABS(close_price - prev_close)) OVER (
                    PARTITION BY symbol 
                    ORDER BY timestamp 
                    ROWS {config.volatility_window - 1} PRECEDING
                ) as atr,
                close_trend_window
            FROM market_data
            WHERE prev_close IS NOT NULL
        ),
        trend_analysis AS (
            SELECT 
                timestamp,
                symbol,
                close_price,
                volume,
                open_interest,
                volatility,
                atr,
                -- Trend calculation (50-period SMA)
                AVG(close_price) OVER (
                    PARTITION BY symbol 
                    ORDER BY timestamp 
                    ROWS {config.trend_window - 1} PRECEDING
                ) as sma_trend,
                -- Price momentum
                CASE 
                    WHEN close_trend_window IS NOT NULL 
                    THEN (close_price - close_trend_window) / close_trend_window * 100
                    ELSE 0
                END as momentum_pct
            FROM volatility_analysis
        ),
        regime_classification AS (
            SELECT 
                timestamp,
                symbol,
                close_price,
                volume,
                open_interest,
                volatility,
                atr,
                sma_trend,
                momentum_pct,
                -- Volatility regime (LOW, MEDIUM, HIGH)
                CASE 
                    WHEN volatility <= PERCENTILE_CONT(0.33) WITHIN GROUP (ORDER BY volatility) OVER (PARTITION BY symbol) 
                    THEN 'LOW'
                    WHEN volatility <= PERCENTILE_CONT(0.67) WITHIN GROUP (ORDER BY volatility) OVER (PARTITION BY symbol)
                    THEN 'MEDIUM'
                    ELSE 'HIGH'
                END as volatility_regime,
                -- Trend regime (BEARISH, NEUTRAL, BULLISH)
                CASE 
                    WHEN momentum_pct < -2.0 THEN 'BEARISH'
                    WHEN momentum_pct > 2.0 THEN 'BULLISH'
                    ELSE 'NEUTRAL'
                END as trend_regime,
                -- Structure regime (BREAKDOWN, BREAKOUT)
                CASE 
                    WHEN close_price > sma_trend THEN 'BREAKOUT'
                    ELSE 'BREAKDOWN'
                END as structure_regime
            FROM trend_analysis
            WHERE sma_trend IS NOT NULL
        )
        SELECT 
            timestamp,
            symbol,
            close_price,
            volume,
            open_interest,
            volatility,
            atr,
            momentum_pct,
            volatility_regime,
            trend_regime,
            structure_regime,
            -- Combined 18-regime classification
            CONCAT(volatility_regime, '_', trend_regime, '_', structure_regime) as regime_classification,
            -- Regime confidence score
            CASE 
                WHEN ABS(momentum_pct) > 5.0 AND volatility > atr * 1.5 THEN 'HIGH'
                WHEN ABS(momentum_pct) > 2.0 OR volatility > atr THEN 'MEDIUM'
                ELSE 'LOW'
            END as regime_confidence
        FROM regime_classification
        ORDER BY timestamp DESC, symbol
        """
        
        return self._optimize_query(query)
    
    def build_correlation_matrix_query(self, config: RegimeQueryConfig) -> str:
        """
        Build correlation matrix query for cross-asset regime analysis
        
        Args:
            config: Regime query configuration
            
        Returns:
            Optimized SQL query for correlation matrix calculation
        """
        symbols_filter = "', '".join(config.symbols)
        
        query = f"""
        WITH price_returns AS (
            SELECT /*+ USE_COLUMNAR_SCAN */
                timestamp,
                symbol,
                close_price,
                LAG(close_price, 1) OVER (PARTITION BY symbol ORDER BY timestamp) as prev_close,
                CASE 
                    WHEN LAG(close_price, 1) OVER (PARTITION BY symbol ORDER BY timestamp) IS NOT NULL
                    THEN (close_price - LAG(close_price, 1) OVER (PARTITION BY symbol ORDER BY timestamp)) / 
                         LAG(close_price, 1) OVER (PARTITION BY symbol ORDER BY timestamp)
                    ELSE 0
                END as return_pct
            FROM {self.table_name}
            WHERE symbol IN ('{symbols_filter}')
            AND timestamp BETWEEN '{config.start_date}' AND '{config.end_date}'
        ),
        correlation_data AS (
            SELECT 
                timestamp,
                symbol,
                return_pct,
                -- Rolling correlation calculation
                AVG(return_pct) OVER (
                    PARTITION BY symbol 
                    ORDER BY timestamp 
                    ROWS {config.correlation_window - 1} PRECEDING
                ) as avg_return,
                STDDEV(return_pct) OVER (
                    PARTITION BY symbol 
                    ORDER BY timestamp 
                    ROWS {config.correlation_window - 1} PRECEDING
                ) as return_volatility
            FROM price_returns
            WHERE prev_close IS NOT NULL
        )
        SELECT 
            timestamp,
            symbol,
            return_pct,
            avg_return,
            return_volatility,
            -- Normalized returns for correlation
            CASE 
                WHEN return_volatility > 0 
                THEN (return_pct - avg_return) / return_volatility
                ELSE 0
            END as normalized_return
        FROM correlation_data
        WHERE return_volatility IS NOT NULL
        ORDER BY timestamp DESC, symbol
        """
        
        return self._optimize_query(query)
    
    def build_option_chain_regime_query(self, symbol: str, expiry_date: datetime) -> str:
        """
        Build option chain query for regime-specific analysis
        
        Args:
            symbol: Trading symbol
            expiry_date: Option expiry date
            
        Returns:
            Optimized SQL query for option chain regime analysis
        """
        query = f"""
        WITH option_data AS (
            SELECT /*+ USE_COLUMNAR_SCAN */
                timestamp,
                symbol,
                option_type,
                strike_price,
                expiry_date,
                close_price,
                volume,
                open_interest,
                -- Greeks calculation (simplified)
                CASE 
                    WHEN option_type = 'CE' 
                    THEN close_price / (strike_price * 0.01)  -- Simplified delta approximation
                    ELSE -close_price / (strike_price * 0.01)
                END as delta_approx,
                -- Implied volatility proxy
                close_price / strike_price as iv_proxy
            FROM {self.table_name}
            WHERE symbol = '{symbol}'
            AND expiry_date = '{expiry_date.date()}'
            AND timestamp >= CURRENT_DATE - INTERVAL '7' DAY
        ),
        regime_indicators AS (
            SELECT 
                timestamp,
                symbol,
                option_type,
                strike_price,
                close_price,
                volume,
                open_interest,
                delta_approx,
                iv_proxy,
                -- Put-Call ratio
                SUM(CASE WHEN option_type = 'PE' THEN volume ELSE 0 END) OVER (
                    PARTITION BY timestamp, symbol 
                ) / NULLIF(SUM(CASE WHEN option_type = 'CE' THEN volume ELSE 0 END) OVER (
                    PARTITION BY timestamp, symbol 
                ), 0) as pcr_volume,
                -- OI Put-Call ratio
                SUM(CASE WHEN option_type = 'PE' THEN open_interest ELSE 0 END) OVER (
                    PARTITION BY timestamp, symbol 
                ) / NULLIF(SUM(CASE WHEN option_type = 'CE' THEN open_interest ELSE 0 END) OVER (
                    PARTITION BY timestamp, symbol 
                ), 0) as pcr_oi
            FROM option_data
        )
        SELECT 
            timestamp,
            symbol,
            option_type,
            strike_price,
            close_price,
            volume,
            open_interest,
            delta_approx,
            iv_proxy,
            pcr_volume,
            pcr_oi,
            -- Regime classification based on option metrics
            CASE 
                WHEN pcr_volume > 1.2 AND pcr_oi > 1.1 THEN 'BEARISH_HIGH_VOL'
                WHEN pcr_volume < 0.8 AND pcr_oi < 0.9 THEN 'BULLISH_LOW_VOL'
                WHEN pcr_volume > 1.0 THEN 'BEARISH_NEUTRAL'
                WHEN pcr_volume < 1.0 THEN 'BULLISH_NEUTRAL'
                ELSE 'NEUTRAL'
            END as option_regime
        FROM regime_indicators
        ORDER BY timestamp DESC, strike_price
        """
        
        return self._optimize_query(query)
    
    def build_real_time_regime_query(self, symbols: List[str], lookback_minutes: int = 60) -> str:
        """
        Build real-time regime monitoring query
        
        Args:
            symbols: List of symbols to monitor
            lookback_minutes: Lookback period in minutes
            
        Returns:
            Optimized SQL query for real-time regime monitoring
        """
        symbols_filter = "', '".join(symbols)
        
        query = f"""
        WITH recent_data AS (
            SELECT /*+ USE_COLUMNAR_SCAN */
                timestamp,
                symbol,
                close_price,
                volume,
                LAG(close_price, 1) OVER (PARTITION BY symbol ORDER BY timestamp) as prev_close
            FROM {self.table_name}
            WHERE symbol IN ('{symbols_filter}')
            AND timestamp >= NOW() - INTERVAL '{lookback_minutes}' MINUTE
            ORDER BY timestamp DESC
        ),
        regime_signals AS (
            SELECT 
                timestamp,
                symbol,
                close_price,
                volume,
                prev_close,
                -- Price change
                CASE 
                    WHEN prev_close IS NOT NULL 
                    THEN (close_price - prev_close) / prev_close * 100
                    ELSE 0
                END as price_change_pct,
                -- Volume surge detection
                volume / AVG(volume) OVER (
                    PARTITION BY symbol 
                    ORDER BY timestamp 
                    ROWS 19 PRECEDING
                ) as volume_ratio,
                -- Volatility spike
                STDDEV(close_price) OVER (
                    PARTITION BY symbol 
                    ORDER BY timestamp 
                    ROWS 9 PRECEDING
                ) as short_vol
            FROM recent_data
        )
        SELECT 
            timestamp,
            symbol,
            close_price,
            volume,
            price_change_pct,
            volume_ratio,
            short_vol,
            -- Real-time regime classification
            CASE 
                WHEN ABS(price_change_pct) > 2.0 AND volume_ratio > 2.0 THEN 'BREAKOUT_HIGH_VOL'
                WHEN ABS(price_change_pct) > 1.0 AND volume_ratio > 1.5 THEN 'TRENDING_MEDIUM_VOL'
                WHEN ABS(price_change_pct) < 0.5 AND volume_ratio < 0.8 THEN 'CONSOLIDATION_LOW_VOL'
                WHEN price_change_pct > 1.0 THEN 'BULLISH_MOMENTUM'
                WHEN price_change_pct < -1.0 THEN 'BEARISH_MOMENTUM'
                ELSE 'NEUTRAL_RANGE'
            END as current_regime,
            -- Alert level
            CASE 
                WHEN ABS(price_change_pct) > 3.0 OR volume_ratio > 3.0 THEN 'HIGH'
                WHEN ABS(price_change_pct) > 1.5 OR volume_ratio > 2.0 THEN 'MEDIUM'
                ELSE 'LOW'
            END as alert_level
        FROM regime_signals
        WHERE prev_close IS NOT NULL
        ORDER BY timestamp DESC, symbol
        LIMIT 1000
        """
        
        return self._optimize_query(query)
    
    def build_regime_transition_query(self, config: RegimeQueryConfig) -> str:
        """
        Build regime transition analysis query
        
        Args:
            config: Regime query configuration
            
        Returns:
            Optimized SQL query for regime transition analysis
        """
        symbols_filter = "', '".join(config.symbols)
        
        query = f"""
        WITH regime_history AS (
            SELECT 
                timestamp,
                symbol,
                close_price,
                -- Simplified regime classification
                CASE 
                    WHEN close_price > AVG(close_price) OVER (
                        PARTITION BY symbol 
                        ORDER BY timestamp 
                        ROWS 49 PRECEDING
                    ) THEN 'BULLISH'
                    ELSE 'BEARISH'
                END as regime,
                LAG(CASE 
                    WHEN close_price > AVG(close_price) OVER (
                        PARTITION BY symbol 
                        ORDER BY timestamp 
                        ROWS 49 PRECEDING
                    ) THEN 'BULLISH'
                    ELSE 'BEARISH'
                END, 1) OVER (PARTITION BY symbol ORDER BY timestamp) as prev_regime
            FROM {self.table_name}
            WHERE symbol IN ('{symbols_filter}')
            AND timestamp BETWEEN '{config.start_date}' AND '{config.end_date}'
        )
        SELECT 
            timestamp,
            symbol,
            close_price,
            regime,
            prev_regime,
            -- Regime transition detection
            CASE 
                WHEN regime != prev_regime AND prev_regime IS NOT NULL 
                THEN CONCAT(prev_regime, '_TO_', regime)
                ELSE 'NO_TRANSITION'
            END as regime_transition,
            -- Transition strength
            CASE 
                WHEN regime != prev_regime AND prev_regime IS NOT NULL 
                THEN ABS(close_price - LAG(close_price, 1) OVER (PARTITION BY symbol ORDER BY timestamp)) / 
                     LAG(close_price, 1) OVER (PARTITION BY symbol ORDER BY timestamp) * 100
                ELSE 0
            END as transition_strength
        FROM regime_history
        WHERE prev_regime IS NOT NULL
        ORDER BY timestamp DESC, symbol
        """
        
        return self._optimize_query(query)
    
    def _optimize_query(self, query: str) -> str:
        """
        Optimize query for GPU execution with HeavyDB hints
        
        Args:
            query: Raw SQL query
            
        Returns:
            Optimized SQL query with GPU hints
        """
        # Remove extra whitespace
        query = ' '.join(query.split())
        
        # Add GPU optimization hints
        query = f"/*+ cpu_mode=false, watchdog_max_size=0, enable_columnar_output=true */ {query}"
        
        return query
    
    def get_regime_summary_query(self, symbols: List[str], days_back: int = 30) -> str:
        """
        Get regime summary statistics query
        
        Args:
            symbols: List of symbols
            days_back: Number of days to look back
            
        Returns:
            SQL query for regime summary statistics
        """
        symbols_filter = "', '".join(symbols)
        
        query = f"""
        SELECT /*+ USE_COLUMNAR_SCAN */
            symbol,
            COUNT(*) as total_observations,
            COUNT(DISTINCT DATE(timestamp)) as trading_days,
            AVG(close_price) as avg_price,
            STDDEV(close_price) as price_volatility,
            MIN(close_price) as min_price,
            MAX(close_price) as max_price,
            SUM(volume) as total_volume,
            AVG(volume) as avg_volume
        FROM {self.table_name}
        WHERE symbol IN ('{symbols_filter}')
        AND timestamp >= CURRENT_DATE - INTERVAL '{days_back}' DAY
        GROUP BY symbol
        ORDER BY symbol
        """
        
        return self._optimize_query(query)
