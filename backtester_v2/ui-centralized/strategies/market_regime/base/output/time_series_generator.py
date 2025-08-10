#!/usr/bin/env python3
"""
Time Series Generator for Market Regime Analysis
===============================================

Specialized component for generating 1-minute time series CSV output
with comprehensive market regime data and Excel parameter integration.

Features:
- 1-minute time series CSV generation
- Multi-timeframe data aggregation
- Regime transition tracking
- Parameter injection from Excel configuration
- Data quality validation and reporting
- Performance optimization for large datasets

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json

from ..common_utils import TimeUtils, MathUtils, ErrorHandler, DataValidator

logger = logging.getLogger(__name__)


class TimeSeriesGenerator:
    """
    Specialized generator for 1-minute time series CSV output with regime data
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Time Series Generator"""
        self.config = config
        self.target_timeframe = config.get('target_timeframe', '1min')
        self.include_transitions = config.get('include_transitions', True)
        self.include_confidence_bands = config.get('include_confidence_bands', True)
        self.data_quality_threshold = config.get('data_quality_threshold', 0.95)
        
        # Initialize utilities
        self.time_utils = TimeUtils()
        self.math_utils = MathUtils()
        self.error_handler = ErrorHandler()
        self.data_validator = DataValidator()
        
        logger.info(f"Time Series Generator initialized for {self.target_timeframe} output")
    
    def generate_time_series(
        self,
        regime_data: pd.DataFrame,
        market_data: pd.DataFrame,
        parameters: Dict[str, Any],
        symbol: str = 'NIFTY'
    ) -> pd.DataFrame:
        """
        Generate comprehensive 1-minute time series with regime data
        
        Args:
            regime_data: Market regime analysis results
            market_data: Raw market data (OHLCV)
            parameters: Excel configuration parameters
            symbol: Trading symbol
            
        Returns:
            DataFrame with complete time series data
        """
        try:
            # Validate input data
            if not self._validate_input_data(regime_data, market_data):
                raise TimeSeriesValidationError("Input data validation failed")
            
            # Ensure both datasets have consistent timestamps
            aligned_data = self._align_timestamps(regime_data, market_data)
            
            # Generate base time series structure
            base_timeseries = self._create_base_timeseries(
                aligned_data['regime'], aligned_data['market'], parameters
            )
            
            # Add regime-specific columns
            enriched_timeseries = self._add_regime_columns(
                base_timeseries, aligned_data['regime'], parameters
            )
            
            # Add market data columns
            complete_timeseries = self._add_market_columns(
                enriched_timeseries, aligned_data['market']
            )
            
            # Add derived indicators
            final_timeseries = self._add_derived_indicators(
                complete_timeseries, parameters
            )
            
            # Add regime transition tracking
            if self.include_transitions:
                final_timeseries = self._add_transition_tracking(final_timeseries)
            
            # Add confidence bands
            if self.include_confidence_bands:
                final_timeseries = self._add_confidence_bands(final_timeseries)
            
            # Add parameter columns for full traceability
            parameter_enriched = self._inject_parameters(
                final_timeseries, parameters, symbol
            )
            
            # Final validation and quality check
            validated_timeseries = self._validate_output_quality(parameter_enriched)
            
            logger.info(f"Generated time series with {len(validated_timeseries)} records for {symbol}")
            return validated_timeseries
            
        except Exception as e:
            error_msg = f"Error generating time series: {e}"
            self.error_handler.handle_error(error_msg, e)
            return pd.DataFrame()
    
    def _validate_input_data(self, regime_data: pd.DataFrame, market_data: pd.DataFrame) -> bool:
        """
        Validate input data quality and structure
        """
        try:
            # Check regime data
            required_regime_cols = ['timestamp', 'regime_name', 'confidence_score']
            for col in required_regime_cols:
                if col not in regime_data.columns:
                    logger.error(f"Missing required regime column: {col}")
                    return False
            
            # Check market data
            required_market_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in required_market_cols:
                if col not in market_data.columns:
                    logger.error(f"Missing required market column: {col}")
                    return False
            
            # Check data completeness
            regime_completeness = 1 - (regime_data.isnull().sum().sum() / (len(regime_data) * len(regime_data.columns)))
            market_completeness = 1 - (market_data.isnull().sum().sum() / (len(market_data) * len(market_data.columns)))
            
            if regime_completeness < self.data_quality_threshold:
                logger.warning(f"Regime data completeness {regime_completeness:.2%} below threshold {self.data_quality_threshold:.2%}")
            
            if market_completeness < self.data_quality_threshold:
                logger.warning(f"Market data completeness {market_completeness:.2%} below threshold {self.data_quality_threshold:.2%}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating input data: {e}")
            return False
    
    def _align_timestamps(self, regime_data: pd.DataFrame, market_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Align timestamps between regime and market data
        """
        try:
            # Ensure timestamp columns are datetime
            regime_data = regime_data.copy()
            market_data = market_data.copy()
            
            regime_data['timestamp'] = pd.to_datetime(regime_data['timestamp'])
            market_data['timestamp'] = pd.to_datetime(market_data['timestamp'])
            
            # Sort by timestamp
            regime_data = regime_data.sort_values('timestamp')
            market_data = market_data.sort_values('timestamp')
            
            # Find common timestamp range
            start_time = max(regime_data['timestamp'].min(), market_data['timestamp'].min())
            end_time = min(regime_data['timestamp'].max(), market_data['timestamp'].max())
            
            # Filter to common range
            regime_aligned = regime_data[
                (regime_data['timestamp'] >= start_time) & 
                (regime_data['timestamp'] <= end_time)
            ].copy()
            
            market_aligned = market_data[
                (market_data['timestamp'] >= start_time) & 
                (market_data['timestamp'] <= end_time)
            ].copy()
            
            logger.info(f"Aligned data from {start_time} to {end_time}")
            logger.info(f"Regime records: {len(regime_aligned)}, Market records: {len(market_aligned)}")
            
            return {'regime': regime_aligned, 'market': market_aligned}
            
        except Exception as e:
            logger.error(f"Error aligning timestamps: {e}")
            return {'regime': regime_data, 'market': market_data}
    
    def _create_base_timeseries(self, regime_data: pd.DataFrame, market_data: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
        """
        Create base time series structure
        """
        try:
            # Use market data as the base since it typically has more frequent timestamps
            base_df = market_data[['timestamp']].copy()
            
            # Create a complete 1-minute time series if gaps exist
            if self.target_timeframe == '1min':
                start_time = base_df['timestamp'].min()
                end_time = base_df['timestamp'].max()
                
                # Generate complete 1-minute range
                complete_range = pd.date_range(
                    start=start_time,
                    end=end_time,
                    freq='1min'
                )
                
                # Create complete DataFrame
                complete_df = pd.DataFrame({'timestamp': complete_range})
                
                # Merge with existing data
                base_df = complete_df.merge(base_df, on='timestamp', how='left')
            
            # Add basic identification columns
            base_df['symbol'] = parameters.get('symbol', 'NIFTY')
            base_df['timeframe'] = self.target_timeframe
            base_df['analysis_version'] = '2.0.0'
            base_df['generation_timestamp'] = datetime.now()
            
            return base_df
            
        except Exception as e:
            logger.error(f"Error creating base time series: {e}")
            return pd.DataFrame()
    
    def _add_regime_columns(self, base_df: pd.DataFrame, regime_data: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
        """
        Add regime analysis columns to time series
        """
        try:
            # Merge regime data with base time series
            merged_df = base_df.merge(
                regime_data, 
                on='timestamp', 
                how='left'
            )
            
            # Forward fill regime data for missing timestamps
            regime_columns = ['regime_name', 'confidence_score', 'final_score']
            for col in regime_columns:
                if col in merged_df.columns:
                    merged_df[col] = merged_df[col].fillna(method='ffill')
            
            # Add regime categorization
            if 'regime_name' in merged_df.columns:
                merged_df['regime_category'] = merged_df['regime_name'].apply(
                    self._categorize_regime
                )
                merged_df['directional_bias'] = merged_df['regime_name'].apply(
                    self._extract_directional_bias
                )
                merged_df['volatility_level'] = merged_df['regime_name'].apply(
                    self._extract_volatility_level
                )
            
            # Add confidence categories
            if 'confidence_score' in merged_df.columns:
                merged_df['confidence_category'] = merged_df['confidence_score'].apply(
                    self._categorize_confidence
                )
            
            # Add trading mode information
            trading_mode = parameters.get('trading_mode', 'hybrid')
            merged_df['trading_mode'] = trading_mode
            
            return merged_df
            
        except Exception as e:
            logger.error(f"Error adding regime columns: {e}")
            return base_df
    
    def _add_market_columns(self, df: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add market data columns (OHLCV)
        """
        try:
            # Merge market data
            market_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            available_cols = [col for col in market_cols if col in market_data.columns]
            
            merged_df = df.merge(
                market_data[available_cols],
                on='timestamp',
                how='left'
            )
            
            # Forward fill OHLCV data for missing timestamps
            ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in ohlcv_cols:
                if col in merged_df.columns:
                    merged_df[col] = merged_df[col].fillna(method='ffill')
            
            # Add basic technical indicators
            if all(col in merged_df.columns for col in ['high', 'low', 'close']):
                merged_df['typical_price'] = (merged_df['high'] + merged_df['low'] + merged_df['close']) / 3
                merged_df['price_range'] = merged_df['high'] - merged_df['low']
                merged_df['price_change'] = merged_df['close'].pct_change()
            
            return merged_df
            
        except Exception as e:
            logger.error(f"Error adding market columns: {e}")
            return df
    
    def _add_derived_indicators(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
        """
        Add derived technical indicators
        """
        try:
            if 'close' not in df.columns:
                return df
            
            # Simple moving averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            # Exponential moving averages
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # RSI
            if len(df) > 14:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding derived indicators: {e}")
            return df
    
    def _add_transition_tracking(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add regime transition tracking
        """
        try:
            if 'regime_name' not in df.columns:
                return df
            
            # Track regime changes
            df['regime_changed'] = df['regime_name'] != df['regime_name'].shift(1)
            df['regime_duration'] = df.groupby((df['regime_name'] != df['regime_name'].shift(1)).cumsum()).cumcount() + 1
            
            # Previous regime tracking
            df['previous_regime'] = df['regime_name'].shift(1)
            
            # Transition type classification
            df['transition_type'] = df.apply(self._classify_transition, axis=1)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding transition tracking: {e}")
            return df
    
    def _add_confidence_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add confidence bands and quality metrics
        """
        try:
            if 'confidence_score' not in df.columns:
                return df
            
            # Rolling confidence statistics
            df['confidence_ma_10'] = df['confidence_score'].rolling(window=10).mean()
            df['confidence_std_10'] = df['confidence_score'].rolling(window=10).std()
            df['confidence_upper_band'] = df['confidence_ma_10'] + (df['confidence_std_10'] * 1.5)
            df['confidence_lower_band'] = df['confidence_ma_10'] - (df['confidence_std_10'] * 1.5)
            
            # Confidence trend
            df['confidence_trend'] = df['confidence_score'].rolling(window=5).apply(
                lambda x: 1 if x.iloc[-1] > x.iloc[0] else (-1 if x.iloc[-1] < x.iloc[0] else 0)
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding confidence bands: {e}")
            return df
    
    def _inject_parameters(self, df: pd.DataFrame, parameters: Dict[str, Any], symbol: str) -> pd.DataFrame:
        """
        Inject Excel parameters into the time series for full traceability
        """
        try:
            # Add key parameters as columns
            key_params = [
                'trading_mode', 'timeframe_weights', 'indicator_weights',
                'volatility_thresholds', 'directional_thresholds'
            ]
            
            for param in key_params:
                if param in parameters:
                    value = parameters[param]
                    if isinstance(value, (dict, list)):
                        df[f'param_{param}'] = json.dumps(value)
                    else:
                        df[f'param_{param}'] = value
            
            # Add generation metadata
            df['param_config_source'] = parameters.get('config_file', 'unknown')
            df['param_analysis_symbol'] = symbol
            df['param_generation_timestamp'] = datetime.now().isoformat()
            
            return df
            
        except Exception as e:
            logger.error(f"Error injecting parameters: {e}")
            return df
    
    def _validate_output_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate output data quality
        """
        try:
            # Check for critical missing data
            critical_cols = ['timestamp', 'symbol', 'timeframe']
            for col in critical_cols:
                if col not in df.columns or df[col].isnull().all():
                    raise TimeSeriesValidationError(f"Critical column {col} is missing or empty")
            
            # Calculate data quality score
            total_cells = len(df) * len(df.columns)
            missing_cells = df.isnull().sum().sum()
            quality_score = 1 - (missing_cells / total_cells)
            
            # Add quality metadata
            df['data_quality_score'] = quality_score
            df['total_records'] = len(df)
            df['missing_data_percentage'] = (missing_cells / total_cells) * 100
            
            logger.info(f"Output quality score: {quality_score:.3f} ({100*quality_score:.1f}%)")
            
            if quality_score < self.data_quality_threshold:
                logger.warning(f"Data quality {quality_score:.3f} below threshold {self.data_quality_threshold}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error validating output quality: {e}")
            return df
    
    # Helper methods for categorization
    def _categorize_regime(self, regime_name: str) -> str:
        """Categorize regime into broad categories"""
        if pd.isna(regime_name):
            return 'Unknown'
        
        regime_lower = regime_name.lower()
        if 'bullish' in regime_lower:
            return 'Bullish'
        elif 'bearish' in regime_lower:
            return 'Bearish'
        elif 'neutral' in regime_lower:
            return 'Neutral'
        else:
            return 'Other'
    
    def _extract_directional_bias(self, regime_name: str) -> str:
        """Extract directional bias"""
        if pd.isna(regime_name):
            return 'Unknown'
        
        regime_lower = regime_name.lower()
        if 'strong_bullish' in regime_lower:
            return 'Strong Bullish'
        elif 'mild_bullish' in regime_lower:
            return 'Mild Bullish'
        elif 'neutral' in regime_lower:
            return 'Neutral'
        elif 'mild_bearish' in regime_lower:
            return 'Mild Bearish'
        elif 'strong_bearish' in regime_lower:
            return 'Strong Bearish'
        else:
            return 'Unknown'
    
    def _extract_volatility_level(self, regime_name: str) -> str:
        """Extract volatility level"""
        if pd.isna(regime_name):
            return 'Unknown'
        
        regime_lower = regime_name.lower()
        if 'high_vol' in regime_lower:
            return 'High'
        elif 'normal_vol' in regime_lower:
            return 'Normal'
        elif 'low_vol' in regime_lower:
            return 'Low'
        else:
            return 'Unknown'
    
    def _categorize_confidence(self, confidence: float) -> str:
        """Categorize confidence score"""
        if pd.isna(confidence):
            return 'Unknown'
        
        if confidence >= 0.85:
            return 'Very High'
        elif confidence >= 0.75:
            return 'High'
        elif confidence >= 0.60:
            return 'Medium'
        elif confidence >= 0.45:
            return 'Low'
        else:
            return 'Very Low'
    
    def _classify_transition(self, row) -> str:
        """Classify regime transition type"""
        if pd.isna(row.get('regime_name')) or pd.isna(row.get('previous_regime')):
            return 'Unknown'
        
        if not row.get('regime_changed', False):
            return 'No Change'
        
        current = row['regime_name'].lower()
        previous = row['previous_regime'].lower()
        
        # Directional transitions
        if 'bullish' in previous and 'bearish' in current:
            return 'Bull to Bear'
        elif 'bearish' in previous and 'bullish' in current:
            return 'Bear to Bull'
        elif 'neutral' in previous and 'bullish' in current:
            return 'Neutral to Bull'
        elif 'neutral' in previous and 'bearish' in current:
            return 'Neutral to Bear'
        elif 'bullish' in previous and 'neutral' in current:
            return 'Bull to Neutral'
        elif 'bearish' in previous and 'neutral' in current:
            return 'Bear to Neutral'
        else:
            return 'Other Transition'


class TimeSeriesValidationError(Exception):
    """Custom exception for time series validation errors"""
    pass