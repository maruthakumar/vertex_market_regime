"""
Market Regime Strategy

This module implements the market regime detection as a strategy
following the backtester_v2 architecture patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, date, timedelta
import logging

# Import BaseStrategy from core
try:
    from ..core.base_strategy import BaseStrategy
except ImportError:
    # Fallback base strategy class
    class BaseStrategy:
        def __init__(self, config):
            self.config = config
            self.db_connection = None
from .models import RegimeConfig, RegimeClassification, RegimeType, PerformanceMetrics
from .calculator import RegimeCalculator
from .classifier import RegimeClassifier
from .performance import PerformanceTracker

logger = logging.getLogger(__name__)

class MarketRegimeStrategy(BaseStrategy):
    """
    Market regime detection and classification strategy
    
    This strategy analyzes market conditions and classifies them into
    different regime types using multiple indicators and timeframes.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the market regime strategy
        
        Args:
            config (Dict[str, Any]): Strategy configuration
        """
        super().__init__(config)
        
        # Parse regime-specific configuration
        self.regime_config = self._parse_regime_config(config)
        
        # Initialize components
        self.regime_calculator = RegimeCalculator(self.regime_config)
        self.regime_classifier = RegimeClassifier(self.regime_config)
        self.performance_tracker = PerformanceTracker(self.regime_config)
        
        # State tracking
        self.current_regime = None
        self.regime_history = []
        self.last_update = None
        
        logger.info(f"MarketRegimeStrategy initialized for {self.regime_config.symbol}")
    
    def _parse_regime_config(self, config: Dict[str, Any]) -> RegimeConfig:
        """Parse configuration into RegimeConfig object"""
        try:
            # If already a RegimeConfig object
            if isinstance(config.get('regime_config'), RegimeConfig):
                return config['regime_config']
            
            # Create from dictionary
            regime_dict = config.get('regime_config', {})
            
            # Set defaults if not provided
            defaults = {
                'strategy_name': config.get('strategy_name', 'MarketRegime'),
                'symbol': config.get('symbol', 'NIFTY'),
                'indicators': [],
                'lookback_days': 252,
                'update_frequency': 'MINUTE',
                'performance_window': 100,
                'learning_rate': 0.01,
                'confidence_threshold': 0.6,
                'regime_smoothing': 3,
                'enable_gpu': True,
                'enable_caching': True
            }
            
            # Merge with defaults
            for key, value in defaults.items():
                if key not in regime_dict:
                    regime_dict[key] = value
            
            return RegimeConfig(**regime_dict)
            
        except Exception as e:
            logger.error(f"Error parsing regime config: {e}")
            # Return minimal config
            return RegimeConfig(
                strategy_name='MarketRegime',
                symbol='NIFTY',
                indicators=[]
            )
    
    def execute(self, start_date: str, end_date: str, **kwargs) -> pd.DataFrame:
        """
        Execute regime detection for date range
        
        Args:
            start_date (str): Start date for analysis
            end_date (str): End date for analysis
            **kwargs: Additional parameters
            
        Returns:
            pd.DataFrame: Regime classifications and analysis results
        """
        try:
            logger.info(f"Executing regime analysis from {start_date} to {end_date}")
            
            # Fetch market data
            market_data = self._fetch_market_data(start_date, end_date, **kwargs)
            
            if market_data.empty:
                logger.warning("No market data available for regime analysis")
                return pd.DataFrame()
            
            # Calculate regimes
            regime_classifications = self.regime_calculator.calculate_regime(market_data, **kwargs)
            
            if not regime_classifications:
                logger.warning("No regime classifications generated")
                return pd.DataFrame()
            
            # Convert to DataFrame
            results_df = self._classifications_to_dataframe(regime_classifications)
            
            # Apply smoothing if enabled
            if self.regime_config.regime_smoothing > 1:
                results_df = self._apply_regime_smoothing(results_df)
            
            # Update performance tracking
            if self.regime_config.performance_window > 0:
                self._update_performance_tracking(results_df, market_data)
            
            # Store results
            self.regime_history.extend(regime_classifications)
            self.last_update = datetime.now()
            
            # Update current regime
            if not results_df.empty:
                latest_regime = results_df.iloc[-1]
                self.current_regime = {
                    'regime_type': latest_regime['regime_type'],
                    'confidence': latest_regime['confidence'],
                    'timestamp': results_df.index[-1]  # Use the index (ts) instead of column
                }
            
            logger.info(f"Generated {len(results_df)} regime classifications")
            return results_df
            
        except Exception as e:
            logger.error(f"Error executing regime strategy: {e}")
            return pd.DataFrame()
    
    def _fetch_market_data(self, start_date: str, end_date: str, **kwargs) -> pd.DataFrame:
        """
        Fetch required market data using query builder

        CRITICAL FIX: Replaced sample data usage with real HeavyDB integration
        using properly quoted identifiers to avoid reserved keyword conflicts
        """
        try:
            # CRITICAL FIX: Use real HeavyDB query with proper quoted identifiers
            query = self._build_market_data_query(start_date, end_date)

            # Try to get HeavyDB connection
            if not hasattr(self, 'db_connection') or not self.db_connection:
                self.db_connection = self._get_heavydb_connection()

            if self.db_connection:
                logger.info(f"Fetching real market data from HeavyDB for {start_date} to {end_date}")
                market_data = pd.read_sql(query, self.db_connection)

                # Ensure proper datetime index
                if 'ts' in market_data.columns:
                    market_data['ts'] = pd.to_datetime(market_data['ts'])
                    market_data.set_index('ts', inplace=True)
                elif 'trade_date' in market_data.columns:
                    market_data['trade_date'] = pd.to_datetime(market_data['trade_date'])
                    market_data.set_index('trade_date', inplace=True)

                logger.info(f"Successfully fetched {len(market_data)} rows of real market data")
                return market_data
            else:
                logger.warning("No database connection available, falling back to sample data")
                return self._generate_sample_data(start_date, end_date)

        except Exception as e:
            logger.error(f"Error fetching market data from HeavyDB: {e}")
            logger.warning("Falling back to sample data due to database error")
            return self._generate_sample_data(start_date, end_date)

    def _get_heavydb_connection(self):
        """Get HeavyDB connection with proper error handling"""
        try:
            # Import HeavyDB connection utilities
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'refactored_core'))

            from heavydb_connection import get_connection
            return get_connection()

        except ImportError as e:
            logger.error(f"HeavyDB connection module not available: {e}")
            return None
        except Exception as e:
            logger.error(f"Error establishing HeavyDB connection: {e}")
            return None
    
    def _build_market_data_query(self, start_date: str, end_date: str) -> str:
        """
        Build comprehensive market data query with proper quoted identifiers

        CRITICAL FIX: Added proper quoting for reserved keywords (open, high, low, close, volume, symbol)
        to resolve HeavyDB reserved keyword conflicts
        """
        # Get symbol from config, default to NIFTY
        symbol = getattr(self.regime_config, 'symbol', 'NIFTY')

        query = f"""
        WITH price_data AS (
            SELECT /*+ gpu_enable */
                trade_date as ts,
                index_name as "symbol",
                spot as underlying_price,
                -- Use quoted identifiers for reserved keywords
                ce_open as "open",
                ce_high as "high",
                ce_low as "low",
                ce_close as "close",
                ce_volume as "volume"
            FROM nifty_option_chain
            WHERE index_name = '{symbol}'
                AND trade_date BETWEEN '{start_date}' AND '{end_date}'
                AND call_strike_type = 'ATM'  -- Get ATM data for underlying price
            GROUP BY trade_date, index_name, spot, ce_open, ce_high, ce_low, ce_close, ce_volume
        ),
        option_data AS (
            SELECT /*+ gpu_enable */
                o.trade_date as ts,
                o.strike,
                -- Map CE/PE to standard option_type format
                CASE
                    WHEN o.ce_close > 0 THEN 'CE'
                    WHEN o.pe_close > 0 THEN 'PE'
                    ELSE 'UNKNOWN'
                END as option_type,
                COALESCE(o.ce_close, o.pe_close, 0) as premium,
                COALESCE(o.ce_oi, o.pe_oi, 0) as open_interest,
                COALESCE(o.ce_volume, o.pe_volume, 0) as option_volume,
                -- Calculate moneyness based on spot price
                CASE
                    WHEN ABS(o.strike - o.spot) <= 50 THEN 'ATM'
                    WHEN o.strike < o.spot - 50 THEN 'ITM'
                    WHEN o.strike > o.spot + 50 THEN 'OTM'
                END as moneyness
            FROM nifty_option_chain o
            WHERE o.index_name = '{symbol}'
                AND o.trade_date BETWEEN '{start_date}' AND '{end_date}'
                AND (o.ce_close > 0 OR o.pe_close > 0)  -- Ensure we have valid option data
        ),
        aggregated_options AS (
            SELECT
                ts,
                moneyness,
                SUM(CASE WHEN option_type = 'CE' THEN premium ELSE 0 END) as ce_premium,
                SUM(CASE WHEN option_type = 'PE' THEN premium ELSE 0 END) as pe_premium,
                SUM(premium) as total_premium,
                SUM(open_interest) as total_oi,
                SUM(option_volume) as total_option_volume
            FROM option_data
            GROUP BY ts, moneyness
        )
        SELECT /*+ gpu_enable */
            p.ts,
            p."symbol",
            p.underlying_price,
            p."open",
            p."high",
            p."low",
            p."close",
            p."volume",
            -- ATM premiums
            MAX(CASE WHEN a.moneyness = 'ATM' THEN a.ce_premium END) as atm_ce_premium,
            MAX(CASE WHEN a.moneyness = 'ATM' THEN a.pe_premium END) as atm_pe_premium,
            MAX(CASE WHEN a.moneyness = 'ATM' THEN a.total_premium END) as atm_straddle_premium,
            MAX(CASE WHEN a.moneyness = 'ATM' THEN a.total_oi END) as atm_total_oi,
            -- ITM premiums
            MAX(CASE WHEN a.moneyness = 'ITM' THEN a.total_premium END) as itm_straddle_premium,
            -- OTM premiums
            MAX(CASE WHEN a.moneyness = 'OTM' THEN a.total_premium END) as otm_straddle_premium
        FROM price_data p
        LEFT JOIN aggregated_options a ON p.ts = a.ts
        GROUP BY p.ts, p."symbol", p.underlying_price, p."open", p."high", p."low", p."close", p."volume"
        ORDER BY p.ts
        """

        return query
    
    def _generate_sample_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """PRODUCTION MODE: NO SYNTHETIC DATA GENERATION - Use real HeavyDB data only"""
        try:
            logger.error("PRODUCTION MODE: Synthetic data generation is disabled. Use real HeavyDB data only.")
            logger.error("Cannot generate sample data - system should connect to HeavyDB instead.")
            
            # Return empty DataFrame to force system to use real data
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error in sample data generation (disabled): {e}")
            return pd.DataFrame()
    
    def _classifications_to_dataframe(self, classifications: List[RegimeClassification]) -> pd.DataFrame:
        """Convert regime classifications to DataFrame"""
        try:
            if not classifications:
                return pd.DataFrame()

            data = []
            for classification in classifications:
                row = {
                    'ts': classification.timestamp,  # Use 'ts' instead of 'timestamp'
                    'symbol': classification.symbol,
                    'regime_type': classification.regime_type.value,
                    'regime_score': classification.regime_score,
                    'confidence': classification.confidence,
                    **classification.component_scores,
                    **classification.timeframe_scores,
                    **classification.metadata
                }
                data.append(row)

            df = pd.DataFrame(data)
            df.set_index('ts', inplace=True)

            return df
            
        except Exception as e:
            logger.error(f"Error converting classifications to DataFrame: {e}")
            return pd.DataFrame()
    
    def _apply_regime_smoothing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply smoothing to regime classifications"""
        try:
            if 'regime_type' not in df.columns or len(df) < self.regime_config.regime_smoothing:
                return df
            
            # Apply rolling mode for regime smoothing
            smoothing_window = self.regime_config.regime_smoothing
            
            # Convert regime types to numeric for smoothing
            regime_numeric = df['regime_type'].map({
                'STRONG_BEARISH': -2,
                'MODERATE_BEARISH': -1,
                'WEAK_BEARISH': -0.5,
                'NEUTRAL': 0,
                'SIDEWAYS': 0,
                'WEAK_BULLISH': 0.5,
                'MODERATE_BULLISH': 1,
                'STRONG_BULLISH': 2
            })
            
            # Apply rolling mean and convert back
            smoothed_numeric = regime_numeric.rolling(window=smoothing_window, center=True).mean()
            
            # Convert back to regime types
            def numeric_to_regime(score):
                if pd.isna(score):
                    return 'NEUTRAL'
                elif score >= 1.5:
                    return 'STRONG_BULLISH'
                elif score >= 0.75:
                    return 'MODERATE_BULLISH'
                elif score >= 0.25:
                    return 'WEAK_BULLISH'
                elif score >= -0.25:
                    return 'NEUTRAL'
                elif score >= -0.75:
                    return 'WEAK_BEARISH'
                elif score >= -1.5:
                    return 'MODERATE_BEARISH'
                else:
                    return 'STRONG_BEARISH'
            
            df['regime_type_smoothed'] = smoothed_numeric.apply(numeric_to_regime)
            
            return df
            
        except Exception as e:
            logger.error(f"Error applying regime smoothing: {e}")
            return df
    
    def _update_performance_tracking(self, results_df: pd.DataFrame, market_data: pd.DataFrame):
        """Update performance tracking with latest results"""
        try:
            if self.performance_tracker:
                self.performance_tracker.update_performance(results_df, market_data)
        except Exception as e:
            logger.error(f"Error updating performance tracking: {e}")
    
    def get_current_regime(self) -> Optional[Dict[str, Any]]:
        """Get current market regime"""
        return self.current_regime
    
    def get_regime_history(self, lookback_hours: int = 24) -> List[RegimeClassification]:
        """Get recent regime history"""
        if not self.regime_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        return [r for r in self.regime_history if r.timestamp >= cutoff_time]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all indicators"""
        try:
            if self.performance_tracker:
                return self.performance_tracker.get_performance_summary()
            return {}
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}

    # Implement abstract methods from BaseStrategy
    def parse_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse input data for regime analysis"""
        return input_data

    def generate_query(self, params: Dict[str, Any]) -> List[str]:
        """Generate queries for regime analysis"""
        return []

    def process_results(self, results: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process regime analysis results"""
        return {'results': results}

    def validate_input(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate input data"""
        return True, []
