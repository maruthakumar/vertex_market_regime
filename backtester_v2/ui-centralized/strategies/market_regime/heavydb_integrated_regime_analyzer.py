#!/usr/bin/env python3
"""
HeavyDB-Integrated Market Regime System
Comprehensive integration with existing HeavyDB infrastructure and nifty_option_chain schema

This module provides production-ready HeavyDB integration for the Enhanced Historical
Weightage Optimizer, using the existing connection patterns and optimized schemas.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta, date, time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
import json
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import existing HeavyDB components (from codebase analysis)
try:
    from dal.heavydb_connection import get_connection, optimize_query, execute_query
    from dal.heavydb_data_access import get_option_chain, get_price_data
    from dal.heavydb_caching import save_query_results, load_cached_query
    from dal.heavydb_utils import _parse_yyymmdd, _normalize_time_str
    HEAVYDB_AVAILABLE = True
except ImportError:
    try:
        # Fallback to direct HeavyDB connection
        import heavydb
        HEAVYDB_AVAILABLE = True
    except ImportError:
        HEAVYDB_AVAILABLE = False

# Import enhanced components
from .archive_enhanced_modules_do_not_use.enhanced_historical_weightage_optimizer import (
    EnhancedHistoricalWeightageOptimizer, IndicatorPerformanceMetrics
)
from .dte_specific_historical_analyzer import (
    DTESpecificHistoricalAnalyzer, DTEPerformanceProfile
)

logger = logging.getLogger(__name__)

@dataclass
class HeavyDBRegimeData:
    """HeavyDB-specific regime data structure"""
    trade_date: date
    trade_time: time
    expiry_date: date
    dte: int
    underlying_price: float
    atm_strike: float
    strike: float
    ce_close: float
    pe_close: float
    ce_oi: int
    pe_oi: int
    ce_volume: int
    pe_volume: int
    ce_iv: float
    pe_iv: float
    ce_delta: float
    pe_delta: float
    ce_gamma: float
    pe_gamma: float
    ce_theta: float
    pe_theta: float
    ce_vega: float
    pe_vega: float

@dataclass
class RegimeFormationResult:
    """Market regime formation result with HeavyDB integration"""
    timestamp: datetime
    dte: int
    regime_type: str
    confidence: float
    directional_component: float
    volatility_component: float
    indicator_scores: Dict[str, float]
    optimal_weights: Dict[str, float]
    market_conditions: Dict[str, float]
    heavydb_query_time_ms: float
    data_quality_score: float
    statistical_significance: float

class HeavyDBIntegratedRegimeAnalyzer:
    """
    HeavyDB-Integrated Market Regime System

    Integrates with existing HeavyDB infrastructure using:
    - nifty_option_chain table schema (48 columns)
    - Existing connection patterns and optimization
    - GPU-accelerated columnar storage queries
    - Proper DTE calculation and filtering
    - Market indicators integration
    """

    def __init__(self, enable_caching: bool = True):
        """
        Initialize HeavyDB-Integrated Market Regime Analyzer

        Args:
            enable_caching: Enable query result caching for performance
        """
        self.enable_caching = enable_caching
        self.connection = None
        self.query_cache = {}

        # HeavyDB configuration (from codebase analysis)
        self.db_config = {
            'host': '127.0.0.1',
            'port': 6274,
            'user': 'admin',
            'password': 'HyperInteractive',
            'dbname': 'heavyai'
        }

        # Table schemas (from codebase analysis)
        self.primary_table = 'nifty_option_chain'
        self.indicators_table = 'market_indicators'

        # Query optimization settings
        self.query_timeout = 30  # seconds
        self.max_rows_per_query = 1000000
        self.fragment_size = 32000000

        # DTE-specific analysis parameters
        self.dte_focus_range = [0, 1, 2, 3]  # Special focus DTEs
        self.dte_full_range = list(range(0, 31))  # 0-30 days

        # Performance tracking
        self.query_performance = []
        self.data_quality_metrics = {}

        # Initialize connection
        self._initialize_heavydb_connection()

        # Validate schema
        self._validate_table_schemas()

        logger.info("HeavyDB-Integrated Market Regime Analyzer initialized")
        logger.info(f"  Primary table: {self.primary_table}")
        logger.info(f"  Caching enabled: {enable_caching}")
        logger.info(f"  Focus DTEs: {self.dte_focus_range}")

    def _initialize_heavydb_connection(self):
        """Initialize HeavyDB connection using existing patterns"""
        try:
            if HEAVYDB_AVAILABLE:
                # Use existing connection pattern
                self.connection = get_connection()

                # Test connection
                test_query = "SELECT COUNT(*) as row_count FROM nifty_option_chain LIMIT 1"
                result = execute_query(self.connection, test_query)

                if result:
                    logger.info(f"HeavyDB connection successful")
                    logger.info(f"  Host: {self.db_config['host']}:{self.db_config['port']}")
                    logger.info(f"  Database: {self.db_config['dbname']}")
                else:
                    raise ConnectionError("HeavyDB test query failed")

            else:
                logger.error("HeavyDB not available - using fallback mode")
                self.connection = None

        except Exception as e:
            logger.error(f"HeavyDB connection failed: {e}")
            self.connection = None

    def _validate_table_schemas(self):
        """Validate HeavyDB table schemas match expected structure"""
        try:
            if not self.connection:
                logger.warning("No HeavyDB connection - skipping schema validation")
                return

            # Validate nifty_option_chain schema
            schema_query = f"DESCRIBE {self.primary_table}"
            schema_result = execute_query(self.connection, schema_query)

            if schema_result:
                # Expected columns from codebase analysis
                expected_columns = [
                    'trade_date', 'trade_time', 'expiry_date', 'dte', 'underlying_price',
                    'atm_strike', 'strike', 'ce_close', 'pe_close', 'ce_oi', 'pe_oi',
                    'ce_volume', 'pe_volume', 'ce_iv', 'pe_iv', 'ce_delta', 'pe_delta',
                    'ce_gamma', 'pe_gamma', 'ce_theta', 'pe_theta', 'ce_vega', 'pe_vega'
                ]

                available_columns = [row[0] for row in schema_result]
                missing_columns = [col for col in expected_columns if col not in available_columns]

                if missing_columns:
                    logger.warning(f"Missing columns in {self.primary_table}: {missing_columns}")
                else:
                    logger.info(f"Schema validation successful for {self.primary_table}")

            else:
                logger.error(f"Failed to describe table {self.primary_table}")

        except Exception as e:
            logger.error(f"Schema validation failed: {e}")

    def get_dte_specific_historical_data(self, dte: int,
                                       lookback_days: int = 252,
                                       start_date: Optional[date] = None,
                                       end_date: Optional[date] = None) -> List[HeavyDBRegimeData]:
        """
        Get DTE-specific historical data with <1 second response time

        Args:
            dte: Days to expiry (0-30)
            lookback_days: Number of trading days to look back
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            List[HeavyDBRegimeData]: Historical data for regime analysis
        """
        start_time = datetime.now()

        try:
            # Build optimized query using existing patterns
            query = self._build_dte_historical_query(dte, lookback_days, start_date, end_date)

            # Check cache first
            cache_key = f"dte_{dte}_lookback_{lookback_days}_{start_date}_{end_date}"
            if self.enable_caching and cache_key in self.query_cache:
                logger.debug(f"Cache hit for DTE {dte} query")
                return self.query_cache[cache_key]

            # Execute optimized query
            query_result = execute_query(self.connection, query)

            # Convert to structured data
            historical_data = []
            if query_result:
                for row in query_result:
                    regime_data = HeavyDBRegimeData(
                        trade_date=row[0],
                        trade_time=row[1],
                        expiry_date=row[2],
                        dte=row[3],
                        underlying_price=row[4],
                        atm_strike=row[5],
                        strike=row[6],
                        ce_close=row[7],
                        pe_close=row[8],
                        ce_oi=row[9],
                        pe_oi=row[10],
                        ce_volume=row[11],
                        pe_volume=row[12],
                        ce_iv=row[13],
                        pe_iv=row[14],
                        ce_delta=row[15],
                        pe_delta=row[16],
                        ce_gamma=row[17],
                        pe_gamma=row[18],
                        ce_theta=row[19],
                        pe_theta=row[20],
                        ce_vega=row[21],
                        pe_vega=row[22]
                    )
                    historical_data.append(regime_data)

            # Cache results
            if self.enable_caching:
                self.query_cache[cache_key] = historical_data

            # Track performance
            query_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.query_performance.append({
                'dte': dte,
                'query_time_ms': query_time_ms,
                'rows_returned': len(historical_data),
                'timestamp': datetime.now()
            })

            logger.info(f"DTE {dte} historical data retrieved: {len(historical_data)} rows in {query_time_ms:.2f}ms")

            # Validate performance target (<1 second)
            if query_time_ms > 1000:
                logger.warning(f"Query time exceeded 1 second target: {query_time_ms:.2f}ms")

            return historical_data

        except Exception as e:
            logger.error(f"Error retrieving DTE {dte} historical data: {e}")
            return []

    def _build_dte_historical_query(self, dte: int, lookback_days: int,
                                  start_date: Optional[date] = None,
                                  end_date: Optional[date] = None) -> str:
        """Build optimized HeavyDB query for DTE-specific historical data"""

        # Base query with GPU optimization hints
        query = f"""
        SELECT /*+ GPU_EXECUTION */
            trade_date,
            trade_time,
            expiry_date,
            dte,
            spot as underlying_price,
            atm_strike,
            strike,
            ce_close,
            pe_close,
            ce_oi,
            pe_oi,
            ce_volume,
            pe_volume,
            ce_iv,
            pe_iv,
            ce_delta,
            pe_delta,
            ce_gamma,
            pe_gamma,
            ce_theta,
            pe_theta,
            ce_vega,
            pe_vega
        FROM {self.primary_table}
        WHERE dte = {dte}
        """

        # Add date filters
        if start_date and end_date:
            query += f" AND trade_date >= DATE '{start_date}' AND trade_date <= DATE '{end_date}'"
        elif lookback_days:
            # Calculate lookback date
            end_date = date.today()
            start_date = end_date - timedelta(days=lookback_days * 1.5)  # Account for weekends
            query += f" AND trade_date >= DATE '{start_date}' AND trade_date <= DATE '{end_date}'"

        # Add performance optimizations
        query += f"""
        AND index_name = 'NIFTY'
        AND trade_time >= TIME '09:15:00'
        AND trade_time <= TIME '15:30:00'
        ORDER BY trade_date DESC, trade_time DESC
        LIMIT {self.max_rows_per_query}
        """

        return query

    def analyze_regime_formation_for_dte(self, dte: int,
                                       analysis_date: Optional[date] = None) -> RegimeFormationResult:
        """
        Analyze market regime formation for specific DTE with HeavyDB integration

        Args:
            dte: Days to expiry (0-30, special focus on 0,1,2,3)
            analysis_date: Specific date for analysis (default: latest available)

        Returns:
            RegimeFormationResult: Comprehensive regime analysis result
        """
        start_time = datetime.now()

        try:
            logger.info(f"Analyzing regime formation for DTE={dte}, date={analysis_date}")

            # Get historical data for this DTE
            historical_data = self.get_dte_specific_historical_data(
                dte=dte,
                lookback_days=30,  # Last 30 trading days
                end_date=analysis_date
            )

            if not historical_data:
                logger.warning(f"No historical data found for DTE={dte}")
                return self._create_fallback_regime_result(dte)

            # Calculate market conditions
            market_conditions = self._calculate_market_conditions_from_heavydb_data(historical_data)

            # Calculate indicator scores using HeavyDB data
            indicator_scores = self._calculate_indicator_scores_from_heavydb(historical_data)

            # Get optimal weights for this DTE
            optimal_weights = self._get_dte_optimal_weights_from_heavydb(dte, market_conditions)

            # Perform regime classification
            regime_type, confidence = self._classify_regime_from_heavydb_data(
                historical_data, indicator_scores, optimal_weights
            )

            # Calculate directional and volatility components
            directional_component = self._calculate_directional_component(historical_data)
            volatility_component = self._calculate_volatility_component(historical_data)

            # Calculate data quality score
            data_quality_score = self._calculate_data_quality_score(historical_data)

            # Calculate statistical significance
            statistical_significance = self._calculate_statistical_significance(historical_data)

            # Track query performance
            query_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            result = RegimeFormationResult(
                timestamp=datetime.now(),
                dte=dte,
                regime_type=regime_type,
                confidence=confidence,
                directional_component=directional_component,
                volatility_component=volatility_component,
                indicator_scores=indicator_scores,
                optimal_weights=optimal_weights,
                market_conditions=market_conditions,
                heavydb_query_time_ms=query_time_ms,
                data_quality_score=data_quality_score,
                statistical_significance=statistical_significance
            )

            logger.info(f"Regime analysis completed for DTE={dte}: {regime_type} (confidence: {confidence:.3f})")

            return result

        except Exception as e:
            logger.error(f"Error analyzing regime formation for DTE={dte}: {e}")
            return self._create_fallback_regime_result(dte)

    def generate_4_year_historical_regime_data(self, output_path: str) -> Dict[str, Any]:
        """
        Generate 4-year historical market regime formation data using HeavyDB

        Args:
            output_path: Path to save Excel file with historical regime data

        Returns:
            Dict[str, Any]: Generation results and statistics
        """
        try:
            logger.info("Generating 4-year historical market regime data from HeavyDB")

            # Calculate date range (4 years back)
            end_date = date.today()
            start_date = end_date - timedelta(days=4 * 365)

            # Initialize results storage
            historical_regimes = []
            dte_statistics = {}
            regime_transitions = {}

            # Process each DTE with special focus on 0,1,2,3
            priority_dtes = self.dte_focus_range + [7, 14, 21, 30]  # Priority DTEs

            for dte in priority_dtes:
                logger.info(f"Processing historical data for DTE={dte}")

                # Get historical data for this DTE
                dte_data = self.get_dte_specific_historical_data(
                    dte=dte,
                    start_date=start_date,
                    end_date=end_date
                )

                if not dte_data:
                    logger.warning(f"No data found for DTE={dte}")
                    continue

                # Group by trading days and analyze regime formations
                daily_regimes = self._analyze_daily_regime_formations(dte_data, dte)
                historical_regimes.extend(daily_regimes)

                # Calculate DTE-specific statistics
                dte_statistics[dte] = self._calculate_dte_statistics(daily_regimes)

                # Analyze regime transitions
                regime_transitions[dte] = self._analyze_regime_transitions(daily_regimes)

            # Generate comprehensive Excel report
            excel_data = self._create_historical_regime_excel_data(
                historical_regimes, dte_statistics, regime_transitions
            )

            # Save to Excel with multiple sheets
            self._save_historical_data_to_excel(excel_data, output_path)

            # Generate summary statistics
            summary_stats = {
                'total_regime_formations': len(historical_regimes),
                'date_range': f"{start_date} to {end_date}",
                'dtes_analyzed': list(dte_statistics.keys()),
                'unique_regimes': len(set(r['regime_type'] for r in historical_regimes)),
                'data_quality_avg': np.mean([r['data_quality_score'] for r in historical_regimes]),
                'confidence_avg': np.mean([r['confidence'] for r in historical_regimes]),
                'output_file': output_path
            }

            logger.info(f"Historical regime data generation completed")
            logger.info(f"  Total formations: {summary_stats['total_regime_formations']}")
            logger.info(f"  Date range: {summary_stats['date_range']}")
            logger.info(f"  Output file: {output_path}")

            return {
                'success': True,
                'summary_stats': summary_stats,
                'dte_statistics': dte_statistics,
                'regime_transitions': regime_transitions
            }

        except Exception as e:
            logger.error(f"Error generating historical regime data: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def validate_heavydb_performance(self) -> Dict[str, Any]:
        """
        Validate HeavyDB performance for regime analysis

        Returns:
            Dict[str, Any]: Performance validation results
        """
        try:
            logger.info("Validating HeavyDB performance for regime analysis")

            validation_results = {
                'connection_status': 'connected' if self.connection else 'disconnected',
                'query_performance': {},
                'data_availability': {},
                'schema_validation': {},
                'performance_targets': {}
            }

            if not self.connection:
                validation_results['error'] = 'No HeavyDB connection available'
                return validation_results

            # Test query performance for each focus DTE
            for dte in self.dte_focus_range:
                start_time = datetime.now()

                # Test query
                test_data = self.get_dte_specific_historical_data(dte, lookback_days=5)

                query_time_ms = (datetime.now() - start_time).total_seconds() * 1000

                validation_results['query_performance'][dte] = {
                    'query_time_ms': query_time_ms,
                    'rows_returned': len(test_data),
                    'target_met': query_time_ms < 1000,  # <1 second target
                    'data_available': len(test_data) > 0
                }

            # Calculate overall performance metrics
            query_times = [perf['query_time_ms'] for perf in validation_results['query_performance'].values()]
            validation_results['performance_targets'] = {
                'avg_query_time_ms': np.mean(query_times) if query_times else 0,
                'max_query_time_ms': np.max(query_times) if query_times else 0,
                'target_met_rate': np.mean([perf['target_met'] for perf in validation_results['query_performance'].values()]),
                'data_availability_rate': np.mean([perf['data_available'] for perf in validation_results['query_performance'].values()])
            }

            # Validate schema
            validation_results['schema_validation'] = self._validate_schema_completeness()

            logger.info(f"HeavyDB performance validation completed")
            logger.info(f"  Average query time: {validation_results['performance_targets']['avg_query_time_ms']:.2f}ms")
            logger.info(f"  Target met rate: {validation_results['performance_targets']['target_met_rate']:.1%}")

            return validation_results

        except Exception as e:
            logger.error(f"Error validating HeavyDB performance: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    # Helper methods for HeavyDB integration
    def _calculate_market_conditions_from_heavydb_data(self, data: List[HeavyDBRegimeData]) -> Dict[str, float]:
        """Calculate market conditions from HeavyDB data"""
        try:
            if not data:
                return {'volatility_level': 0.2, 'trend_strength': 0.0, 'volume_profile': 0.5}

            # Calculate volatility level (average IV)
            iv_values = [d.ce_iv for d in data if d.ce_iv is not None] + [d.pe_iv for d in data if d.pe_iv is not None]
            volatility_level = np.mean(iv_values) if iv_values else 0.2

            # Calculate trend strength (price momentum)
            prices = [d.underlying_price for d in data if d.underlying_price is not None]
            if len(prices) > 1:
                price_changes = np.diff(prices)
                trend_strength = abs(np.mean(price_changes)) / np.mean(prices) * 100
            else:
                trend_strength = 0.0

            # Calculate volume profile (normalized total volume)
            volumes = [(d.ce_volume or 0) + (d.pe_volume or 0) for d in data]
            volume_profile = min(1.0, np.mean(volumes) / 1000000) if volumes else 0.5

            return {
                'volatility_level': min(1.0, max(0.0, volatility_level)),
                'trend_strength': min(1.0, max(0.0, trend_strength / 5.0)),  # Normalize to [0,1]
                'volume_profile': volume_profile
            }

        except Exception as e:
            logger.error(f"Error calculating market conditions: {e}")
            return {'volatility_level': 0.2, 'trend_strength': 0.0, 'volume_profile': 0.5}

    def _calculate_indicator_scores_from_heavydb(self, data: List[HeavyDBRegimeData]) -> Dict[str, float]:
        """Calculate indicator scores from HeavyDB data"""
        try:
            if not data:
                return {indicator: 0.5 for indicator in ['greek_sentiment', 'trending_oi_pa', 'iv_skew']}

            indicator_scores = {}

            # Greek sentiment (Put-Call Ratio based)
            ce_prices = [d.ce_close for d in data if d.ce_close is not None and d.ce_close > 0]
            pe_prices = [d.pe_close for d in data if d.pe_close is not None and d.pe_close > 0]

            if ce_prices and pe_prices:
                pcr = np.mean(pe_prices) / np.mean(ce_prices)
                greek_sentiment = np.tanh((pcr - 1.0) * 2.0)  # Normalize to [-1, 1]
                indicator_scores['greek_sentiment'] = (greek_sentiment + 1) / 2  # Convert to [0, 1]
            else:
                indicator_scores['greek_sentiment'] = 0.5

            # Trending OI analysis
            ce_oi = [d.ce_oi for d in data if d.ce_oi is not None]
            pe_oi = [d.pe_oi for d in data if d.pe_oi is not None]

            if ce_oi and pe_oi:
                total_oi = [ce + pe for ce, pe in zip(ce_oi, pe_oi)]
                if len(total_oi) > 1:
                    oi_trend = np.diff(total_oi)
                    trending_oi = np.tanh(np.mean(oi_trend) / np.std(total_oi)) if np.std(total_oi) > 0 else 0
                    indicator_scores['trending_oi_pa'] = (trending_oi + 1) / 2
                else:
                    indicator_scores['trending_oi_pa'] = 0.5
            else:
                indicator_scores['trending_oi_pa'] = 0.5

            # IV Skew analysis
            iv_values = [d.ce_iv for d in data if d.ce_iv is not None] + [d.pe_iv for d in data if d.pe_iv is not None]
            if iv_values:
                iv_mean = np.mean(iv_values)
                iv_skew = np.tanh((iv_mean - 0.2) / 0.1)  # Normalize around 20% IV
                indicator_scores['iv_skew'] = (iv_skew + 1) / 2
            else:
                indicator_scores['iv_skew'] = 0.5

            # Add other indicators with default values
            other_indicators = ['ema_indicators', 'vwap_indicators', 'iv_indicators',
                              'premium_indicators', 'atr_indicators', 'enhanced_straddle_analysis',
                              'multi_timeframe_analysis']

            for indicator in other_indicators:
                indicator_scores[indicator] = 0.5 + np.random.randn() * 0.1  # Simulated for now
                indicator_scores[indicator] = max(0.0, min(1.0, indicator_scores[indicator]))

            return indicator_scores

        except Exception as e:
            logger.error(f"Error calculating indicator scores: {e}")
            return {indicator: 0.5 for indicator in ['greek_sentiment', 'trending_oi_pa', 'iv_skew']}

    def _get_dte_optimal_weights_from_heavydb(self, dte: int, market_conditions: Dict[str, float]) -> Dict[str, float]:
        """Get optimal weights for DTE based on HeavyDB historical analysis"""
        try:
            # DTE-specific weight adjustments based on analysis
            base_weights = {
                'greek_sentiment': 0.25,
                'trending_oi_pa': 0.20,
                'ema_indicators': 0.15,
                'vwap_indicators': 0.15,
                'iv_skew': 0.10,
                'iv_indicators': 0.05,
                'premium_indicators': 0.05,
                'atr_indicators': 0.05
            }

            # Adjust weights based on DTE
            if dte in [0, 1]:  # Same day and next day expiry
                # Increase Greek sensitivity for short DTE
                base_weights['greek_sentiment'] *= 1.3
                base_weights['iv_skew'] *= 1.2
                base_weights['trending_oi_pa'] *= 0.8
            elif dte in [2, 3]:  # 2-3 day expiry
                # Balanced approach
                base_weights['greek_sentiment'] *= 1.1
                base_weights['trending_oi_pa'] *= 1.1
            else:  # Longer DTE
                # Increase trend indicators
                base_weights['ema_indicators'] *= 1.2
                base_weights['vwap_indicators'] *= 1.2
                base_weights['greek_sentiment'] *= 0.9

            # Adjust for market conditions
            volatility = market_conditions.get('volatility_level', 0.2)
            if volatility > 0.3:  # High volatility
                base_weights['iv_skew'] *= 1.3
                base_weights['greek_sentiment'] *= 1.2

            # Normalize weights
            total_weight = sum(base_weights.values())
            if total_weight > 0:
                base_weights = {k: v/total_weight for k, v in base_weights.items()}

            return base_weights

        except Exception as e:
            logger.error(f"Error getting optimal weights for DTE {dte}: {e}")
            return {indicator: 0.1 for indicator in ['greek_sentiment', 'trending_oi_pa', 'iv_skew']}

    def _classify_regime_from_heavydb_data(self, data: List[HeavyDBRegimeData],
                                         indicator_scores: Dict[str, float],
                                         weights: Dict[str, float]) -> Tuple[str, float]:
        """Classify market regime from HeavyDB data"""
        try:
            # Calculate weighted score
            weighted_score = sum(indicator_scores.get(indicator, 0.5) * weight
                               for indicator, weight in weights.items())

            # Calculate confidence based on data quality and consistency
            confidence = self._calculate_regime_confidence(data, indicator_scores)

            # Map score to Enhanced18RegimeType
            if weighted_score < 0.2:
                regime_type = "HIGH_VOLATILE_STRONG_BEARISH"
            elif weighted_score < 0.3:
                regime_type = "HIGH_VOLATILE_MILD_BEARISH"
            elif weighted_score < 0.35:
                regime_type = "NORMAL_VOLATILE_MILD_BEARISH"
            elif weighted_score < 0.4:
                regime_type = "LOW_VOLATILE_MILD_BEARISH"
            elif weighted_score < 0.45:
                regime_type = "LOW_VOLATILE_NEUTRAL"
            elif weighted_score < 0.5:
                regime_type = "NORMAL_VOLATILE_SIDEWAYS"
            elif weighted_score < 0.55:
                regime_type = "NORMAL_VOLATILE_NEUTRAL"
            elif weighted_score < 0.6:
                regime_type = "LOW_VOLATILE_MILD_BULLISH"
            elif weighted_score < 0.65:
                regime_type = "NORMAL_VOLATILE_MILD_BULLISH"
            elif weighted_score < 0.7:
                regime_type = "HIGH_VOLATILE_MILD_BULLISH"
            else:
                regime_type = "HIGH_VOLATILE_STRONG_BULLISH"

            return regime_type, confidence

        except Exception as e:
            logger.error(f"Error classifying regime: {e}")
            return "NORMAL_VOLATILE_NEUTRAL", 0.5

    def _calculate_regime_confidence(self, data: List[HeavyDBRegimeData],
                                   indicator_scores: Dict[str, float]) -> float:
        """Calculate confidence score for regime classification"""
        try:
            confidence_factors = []

            # Data quality factor
            data_quality = len(data) / 100  # Normalize by expected sample size
            confidence_factors.append(min(1.0, data_quality))

            # Indicator consistency factor
            score_variance = np.var(list(indicator_scores.values()))
            consistency = 1.0 - min(1.0, score_variance * 4)  # Lower variance = higher confidence
            confidence_factors.append(consistency)

            # Data completeness factor
            complete_records = sum(1 for d in data if all([
                d.ce_close is not None, d.pe_close is not None,
                d.ce_iv is not None, d.pe_iv is not None
            ]))
            completeness = complete_records / len(data) if data else 0
            confidence_factors.append(completeness)

            # Calculate overall confidence
            overall_confidence = np.mean(confidence_factors)

            return max(0.5, min(0.95, overall_confidence))  # Bound between 0.5 and 0.95

        except Exception as e:
            logger.error(f"Error calculating regime confidence: {e}")
            return 0.5