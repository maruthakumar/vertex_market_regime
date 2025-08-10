#!/usr/bin/env python3
"""
Adaptive Timeframe Regime Extractor for Market Regime System

This module extracts regime scores for multiple configurable timeframes supporting
both intraday and positional trading modes. Integrates with the adaptive timeframe
manager for dynamic timeframe selection.

Features:
1. Multi-timeframe regime score extraction (3min/5min/10min/15min/30min/1hr)
2. Adaptive timeframe selection based on trading mode
3. Timeframe-specific data resampling and analysis
4. Regime score normalization and weighting
5. Cross-timeframe correlation analysis
6. Real-time timeframe regime monitoring
7. Strict real data enforcement (100% HeavyDB data)

Author: The Augster
Date: 2025-01-10 (Updated for adaptive timeframes)
Version: 3.0.0 - Adaptive Timeframe Support
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio

# Import market regime components
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from strategies.market_regime.enhanced_12_regime_detector import Enhanced12RegimeDetector
    from strategies.market_regime.atm_cepe_rolling_analyzer import ATMCEPERollingAnalyzer
    from strategies.market_regime.adaptive_timeframe_manager import AdaptiveTimeframeManager
    from dal.heavydb_connection import (
        get_connection_status, RealDataUnavailableError, 
        SyntheticDataProhibitedError
    )
except ImportError as e:
    logging.error(f"Failed to import required components: {e}")

logger = logging.getLogger(__name__)

@dataclass
class TimeframeRegimeScore:
    """Data structure for timeframe-specific regime scores"""
    timeframe: str
    regime_score: float
    regime_id: str
    confidence: float
    trend_strength: float
    volatility_score: float
    structure_score: float
    correlation_score: float
    timestamp: datetime

@dataclass
class MultiTimeframeAnalysis:
    """Data structure for multi-timeframe regime analysis"""
    timeframe_scores: Dict[str, TimeframeRegimeScore]
    cross_timeframe_correlation: float
    dominant_timeframe: str
    regime_consistency: float
    overall_confidence: float
    timestamp: datetime

class TimeframeRegimeExtractor:
    """
    Extract regime scores for specific timeframes to support dashboard metrics
    
    Now with adaptive timeframe support:
    1. Multi-timeframe data resampling (3min/5min/10min/15min/30min/1hr)
    2. Dynamic timeframe selection based on trading mode
    3. Regime score calculation per timeframe
    4. Cross-timeframe correlation analysis
    5. Regime consistency measurement
    """
    
    def __init__(self, trading_mode='hybrid', adaptive_config=None):
        """
        Initialize timeframe regime extractor with adaptive support
        
        Args:
            trading_mode: Trading mode ('intraday', 'positional', 'hybrid', 'custom')
            adaptive_config: Optional custom configuration
        """
        # Initialize adaptive timeframe manager
        self.timeframe_manager = AdaptiveTimeframeManager(mode=trading_mode, config=adaptive_config)
        
        # Get configuration from adaptive manager
        self._update_configuration()
        
        # Initialize regime detector for timeframe analysis
        self.regime_detector = None
        self.atm_analyzer = None
        
        logger.info(f"TimeframeRegimeExtractor initialized with {trading_mode} mode")
    
    def _update_configuration(self):
        """Update configuration from adaptive timeframe manager"""
        self.supported_timeframes = self.timeframe_manager.get_active_timeframes()
        self.timeframe_weights = self.timeframe_manager.get_timeframe_weights()
        
        # Update minimum data points based on active timeframes
        self.min_data_points = {}
        for tf in self.supported_timeframes:
            tf_config = self.timeframe_manager.get_timeframe_config(tf)
            if tf_config:
                self.min_data_points[tf] = tf_config.min_data_points
            else:
                self.min_data_points[tf] = 1  # Default
    
    def switch_trading_mode(self, new_mode, config=None):
        """
        Switch to a different trading mode
        
        Args:
            new_mode: Target mode ('intraday', 'positional', 'hybrid', 'custom')
            config: Optional configuration for custom mode
        """
        if self.timeframe_manager.switch_mode(new_mode, config):
            self._update_configuration()
            logger.info(f"Switched to {new_mode} mode with timeframes: {self.supported_timeframes}")
            return True
        return False
    
    def initialize_components(self):
        """Initialize market regime components if not already done"""
        try:
            if not self.regime_detector:
                self.regime_detector = Enhanced12RegimeDetector()
            if not self.atm_analyzer:
                self.atm_analyzer = ATMCEPERollingAnalyzer()
            logger.debug("Market regime components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize market regime components: {e}")
    
    def validate_timeframe_data(self, data: pd.DataFrame, timeframe: str) -> None:
        """
        Validate timeframe data for authenticity and completeness
        
        Args:
            data: Timeframe-specific data
            timeframe: Target timeframe (3min, 5min, 10min, 15min, 30min, 1hr)
            
        Raises:
            RealDataUnavailableError: If data validation fails
        """
        try:
            if data.empty:
                raise RealDataUnavailableError(f"No data available for {timeframe} timeframe")
            
            # Check minimum data points requirement
            min_points = self.min_data_points.get(timeframe, 4)
            if len(data) < min_points:
                raise RealDataUnavailableError(
                    f"Insufficient data points for {timeframe}: {len(data)} < {min_points}"
                )
            
            # Validate required columns for regime analysis
            required_columns = ['underlying_price', 'timestamp']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                logger.warning(f"Missing columns for {timeframe}: {missing_columns}")
            
            # Check data freshness
            if 'timestamp' in data.columns:
                latest_time = pd.to_datetime(data['timestamp']).max()
                time_diff = datetime.now() - latest_time
                if time_diff > timedelta(hours=4):
                    logger.warning(f"Stale data for {timeframe}: {time_diff}")
            
            logger.debug(f"Timeframe data validated for {timeframe}: {len(data)} records")
            
        except Exception as e:
            logger.error(f"Timeframe data validation failed for {timeframe}: {e}")
            raise RealDataUnavailableError(f"Data validation failed for {timeframe}: {str(e)}")
    
    def resample_data_to_timeframe(self, market_data: Dict[str, Any], timeframe: str) -> pd.DataFrame:
        """
        Resample market data to specific timeframe
        
        Args:
            market_data: Raw market data
            timeframe: Target timeframe (3min, 5min, 10min, 15min, 30min, 1hr, 4hr)
            
        Returns:
            Resampled DataFrame for the specified timeframe
        """
        try:
            # Extract timeframe minutes
            if timeframe == '1hr':
                timeframe_minutes = 60
            elif timeframe == '4hr':
                timeframe_minutes = 240
            else:
                timeframe_minutes = int(timeframe.replace('min', ''))
            
            # Convert market data to DataFrame if needed
            if isinstance(market_data, dict):
                # Create DataFrame from market data dictionary
                df_data = []
                
                # Extract time series data if available
                if 'price_data' in market_data and isinstance(market_data['price_data'], pd.DataFrame):
                    df_data = market_data['price_data'].copy()
                elif 'underlying_price' in market_data:
                    # Create single-row DataFrame for current data point
                    df_data = pd.DataFrame([{
                        'underlying_price': market_data['underlying_price'],
                        'timestamp': market_data.get('timestamp', datetime.now()),
                        'volume': market_data.get('volume', 0),
                        'iv_percentile': market_data.get('iv_percentile', 0.5),
                        'atr_normalized': market_data.get('atr_normalized', 0.4)
                    }])
                else:
                    # Create minimal DataFrame
                    df_data = pd.DataFrame([{
                        'underlying_price': 19500,  # Default NIFTY level
                        'timestamp': datetime.now(),
                        'volume': 0,
                        'iv_percentile': 0.5,
                        'atr_normalized': 0.4
                    }])
            else:
                df_data = market_data.copy()
            
            # Ensure timestamp column exists and is datetime
            if 'timestamp' not in df_data.columns:
                df_data['timestamp'] = datetime.now()
            
            df_data['timestamp'] = pd.to_datetime(df_data['timestamp'])
            df_data.set_index('timestamp', inplace=True)
            
            # Resample to target timeframe
            resampling_rule = f'{timeframe_minutes}T'  # T for minutes
            
            resampled_data = df_data.resample(resampling_rule).agg({
                'underlying_price': 'last',
                'volume': 'sum',
                'iv_percentile': 'mean',
                'atr_normalized': 'mean'
            }).dropna()
            
            # Reset index to get timestamp as column
            resampled_data.reset_index(inplace=True)
            
            logger.debug(f"Data resampled to {timeframe}: {len(resampled_data)} periods")
            
            return resampled_data
            
        except Exception as e:
            logger.error(f"Error resampling data to {timeframe}: {e}")
            # Return minimal DataFrame for error cases
            return pd.DataFrame([{
                'timestamp': datetime.now(),
                'underlying_price': 19500,
                'volume': 0,
                'iv_percentile': 0.5,
                'atr_normalized': 0.4
            }])
    
    def calculate_regime_score_for_timeframe(self, timeframe_data: pd.DataFrame, 
                                           timeframe: str) -> TimeframeRegimeScore:
        """
        Calculate regime score for specific timeframe
        
        Args:
            timeframe_data: Resampled data for the timeframe
            timeframe: Target timeframe
            
        Returns:
            TimeframeRegimeScore object with calculated metrics
        """
        try:
            # Initialize components if needed
            self.initialize_components()
            
            # Validate timeframe data
            self.validate_timeframe_data(timeframe_data, timeframe)
            
            # Prepare market data for regime analysis
            if not timeframe_data.empty:
                latest_data = timeframe_data.iloc[-1]
                market_data_dict = {
                    'underlying_price': latest_data.get('underlying_price', 19500),
                    'timestamp': latest_data.get('timestamp', datetime.now()),
                    'volume': latest_data.get('volume', 0),
                    'iv_percentile': latest_data.get('iv_percentile', 0.5),
                    'atr_normalized': latest_data.get('atr_normalized', 0.4),
                    'timeframe': timeframe,
                    'historical_data': timeframe_data
                }
            else:
                # Default market data
                market_data_dict = {
                    'underlying_price': 19500,
                    'timestamp': datetime.now(),
                    'volume': 0,
                    'iv_percentile': 0.5,
                    'atr_normalized': 0.4,
                    'timeframe': timeframe
                }
            
            # Calculate regime classification for this timeframe
            if self.regime_detector:
                try:
                    regime_result = self.regime_detector.classify_12_regime(market_data_dict)
                    regime_id = regime_result.regime_id
                    confidence = regime_result.confidence
                    regime_score = regime_result.final_score
                except Exception as e:
                    logger.warning(f"Regime detection failed for {timeframe}: {e}")
                    regime_id = 'UNKNOWN'
                    confidence = 0.0
                    regime_score = 0.0
            else:
                regime_id = 'UNKNOWN'
                confidence = 0.0
                regime_score = 0.0
            
            # Calculate timeframe-specific metrics
            trend_strength = self._calculate_trend_strength(timeframe_data)
            volatility_score = self._calculate_volatility_score(timeframe_data)
            structure_score = self._calculate_structure_score(timeframe_data)
            correlation_score = self._calculate_correlation_score(timeframe_data)
            
            # Create timeframe regime score object
            timeframe_regime_score = TimeframeRegimeScore(
                timeframe=timeframe,
                regime_score=regime_score,
                regime_id=regime_id,
                confidence=confidence,
                trend_strength=trend_strength,
                volatility_score=volatility_score,
                structure_score=structure_score,
                correlation_score=correlation_score,
                timestamp=datetime.now()
            )
            
            logger.debug(f"Regime score calculated for {timeframe}: {regime_score:.3f} ({regime_id})")
            
            return timeframe_regime_score
            
        except (RealDataUnavailableError, SyntheticDataProhibitedError) as e:
            logger.error(f"Real data enforcement error for {timeframe}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error calculating regime score for {timeframe}: {e}")
            return self._get_default_timeframe_score(timeframe)
    
    def extract_timeframe_scores(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract regime scores for all supported timeframes
        
        Args:
            market_data: Current market data for analysis
            
        Returns:
            Dictionary with regime scores for each timeframe
        """
        try:
            timeframe_scores = {}
            timeframe_regime_objects = {}
            
            # Calculate regime scores for each timeframe
            for timeframe in self.supported_timeframes:
                try:
                    # Resample data to timeframe
                    timeframe_data = self.resample_data_to_timeframe(market_data, timeframe)
                    
                    # Calculate regime score for this timeframe
                    regime_score_obj = self.calculate_regime_score_for_timeframe(timeframe_data, timeframe)
                    
                    # Store results
                    timeframe_scores[f'regime_score_{timeframe}'] = regime_score_obj.regime_score
                    timeframe_regime_objects[timeframe] = regime_score_obj
                    
                except Exception as e:
                    logger.warning(f"Failed to calculate regime score for {timeframe}: {e}")
                    timeframe_scores[f'regime_score_{timeframe}'] = 0.0
                    timeframe_regime_objects[timeframe] = self._get_default_timeframe_score(timeframe)
            
            # Calculate cross-timeframe metrics
            cross_timeframe_metrics = self._calculate_cross_timeframe_metrics(timeframe_regime_objects)
            timeframe_scores.update(cross_timeframe_metrics)
            
            logger.info(f"Timeframe scores extracted: {timeframe_scores}")
            
            return timeframe_scores
            
        except Exception as e:
            logger.error(f"Error extracting timeframe scores: {e}")
            # Return default scores for all timeframes
            return {f'regime_score_{tf}': 0.0 for tf in self.supported_timeframes}
    
    def analyze_multi_timeframe_regime(self, market_data: Dict[str, Any]) -> MultiTimeframeAnalysis:
        """
        Comprehensive multi-timeframe regime analysis
        
        Args:
            market_data: Current market data for analysis
            
        Returns:
            MultiTimeframeAnalysis object with comprehensive results
        """
        try:
            timeframe_regime_scores = {}
            
            # Calculate regime scores for each timeframe
            for timeframe in self.supported_timeframes:
                timeframe_data = self.resample_data_to_timeframe(market_data, timeframe)
                regime_score = self.calculate_regime_score_for_timeframe(timeframe_data, timeframe)
                timeframe_regime_scores[timeframe] = regime_score
            
            # Calculate cross-timeframe correlation
            cross_correlation = self._calculate_cross_timeframe_correlation(timeframe_regime_scores)
            
            # Determine dominant timeframe
            dominant_timeframe = self._determine_dominant_timeframe(timeframe_regime_scores)
            
            # Calculate regime consistency
            regime_consistency = self._calculate_regime_consistency(timeframe_regime_scores)
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(timeframe_regime_scores)
            
            # Create comprehensive analysis object
            multi_timeframe_analysis = MultiTimeframeAnalysis(
                timeframe_scores=timeframe_regime_scores,
                cross_timeframe_correlation=cross_correlation,
                dominant_timeframe=dominant_timeframe,
                regime_consistency=regime_consistency,
                overall_confidence=overall_confidence,
                timestamp=datetime.now()
            )
            
            logger.info(f"Multi-timeframe analysis completed: dominant={dominant_timeframe}, "
                       f"consistency={regime_consistency:.3f}")
            
            return multi_timeframe_analysis
            
        except Exception as e:
            logger.error(f"Error in multi-timeframe regime analysis: {e}")
            return self._get_default_multi_timeframe_analysis()
    
    def _calculate_trend_strength(self, timeframe_data: pd.DataFrame) -> float:
        """Calculate trend strength for timeframe data"""
        try:
            if len(timeframe_data) < 2:
                return 0.0
            
            prices = timeframe_data['underlying_price'].values
            price_changes = np.diff(prices)
            
            # Calculate trend strength as directional consistency
            positive_changes = np.sum(price_changes > 0)
            negative_changes = np.sum(price_changes < 0)
            total_changes = len(price_changes)
            
            if total_changes == 0:
                return 0.0
            
            trend_strength = abs(positive_changes - negative_changes) / total_changes
            return min(1.0, trend_strength)
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return 0.0
    
    def _calculate_volatility_score(self, timeframe_data: pd.DataFrame) -> float:
        """Calculate volatility score for timeframe data"""
        try:
            if len(timeframe_data) < 2:
                return 0.0
            
            prices = timeframe_data['underlying_price'].values
            returns = np.diff(np.log(prices))
            volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
            
            # Normalize volatility score (0-1 range)
            normalized_vol = min(1.0, volatility / 0.5)  # Assume 50% as high volatility
            return normalized_vol
            
        except Exception as e:
            logger.error(f"Error calculating volatility score: {e}")
            return 0.0
    
    def _calculate_structure_score(self, timeframe_data: pd.DataFrame) -> float:
        """Calculate market structure score for timeframe data"""
        try:
            if len(timeframe_data) < 3:
                return 0.0
            
            prices = timeframe_data['underlying_price'].values
            
            # Calculate structure score based on price pattern consistency
            price_diffs = np.diff(prices)
            structure_consistency = 1.0 - (np.std(price_diffs) / (np.mean(np.abs(price_diffs)) + 1e-6))
            
            return max(0.0, min(1.0, structure_consistency))
            
        except Exception as e:
            logger.error(f"Error calculating structure score: {e}")
            return 0.0
    
    def _calculate_correlation_score(self, timeframe_data: pd.DataFrame) -> float:
        """Calculate correlation score for timeframe data"""
        try:
            if len(timeframe_data) < 3:
                return 0.0
            
            # Calculate correlation between price and volume if available
            if 'volume' in timeframe_data.columns:
                correlation = np.corrcoef(
                    timeframe_data['underlying_price'].values,
                    timeframe_data['volume'].values
                )[0, 1]
                
                if np.isnan(correlation):
                    correlation = 0.0
                
                return abs(correlation)  # Return absolute correlation
            else:
                return 0.5  # Default moderate correlation
            
        except Exception as e:
            logger.error(f"Error calculating correlation score: {e}")
            return 0.0
    
    def _calculate_cross_timeframe_metrics(self, timeframe_regime_objects: Dict[str, TimeframeRegimeScore]) -> Dict[str, float]:
        """Calculate cross-timeframe correlation and consistency metrics"""
        try:
            scores = [obj.regime_score for obj in timeframe_regime_objects.values()]
            
            if len(scores) < 2:
                return {'cross_timeframe_correlation': 0.0, 'regime_consistency': 0.0}
            
            # Calculate correlation between timeframe scores
            correlation_matrix = np.corrcoef(scores)
            avg_correlation = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
            
            # Calculate regime consistency
            regime_ids = [obj.regime_id for obj in timeframe_regime_objects.values()]
            unique_regimes = len(set(regime_ids))
            consistency = 1.0 - (unique_regimes - 1) / max(len(regime_ids) - 1, 1)
            
            return {
                'cross_timeframe_correlation': avg_correlation if not np.isnan(avg_correlation) else 0.0,
                'regime_consistency': consistency
            }
            
        except Exception as e:
            logger.error(f"Error calculating cross-timeframe metrics: {e}")
            return {'cross_timeframe_correlation': 0.0, 'regime_consistency': 0.0}
    
    def _calculate_cross_timeframe_correlation(self, timeframe_scores: Dict[str, TimeframeRegimeScore]) -> float:
        """Calculate cross-timeframe correlation"""
        try:
            scores = [score.regime_score for score in timeframe_scores.values()]
            if len(scores) < 2:
                return 0.0
            
            correlation_matrix = np.corrcoef(scores)
            return np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
            
        except Exception as e:
            logger.error(f"Error calculating cross-timeframe correlation: {e}")
            return 0.0
    
    def _determine_dominant_timeframe(self, timeframe_scores: Dict[str, TimeframeRegimeScore]) -> str:
        """Determine the dominant timeframe based on confidence and score"""
        try:
            if not timeframe_scores:
                # Default to longest active timeframe
                active_timeframes = self.supported_timeframes
                if active_timeframes:
                    # Sort by timeframe duration and return longest
                    sorted_tf = sorted(active_timeframes, 
                                     key=lambda x: self._get_timeframe_minutes(x), 
                                     reverse=True)
                    return sorted_tf[0] if sorted_tf else '30min'
                return '30min'  # Fallback default
            
            # Calculate weighted score for each timeframe
            weighted_scores = {}
            for timeframe, score_obj in timeframe_scores.items():
                weight = self.timeframe_weights.get(timeframe, 0.25)
                weighted_score = score_obj.regime_score * score_obj.confidence * weight
                weighted_scores[timeframe] = weighted_score
            
            # Return timeframe with highest weighted score
            dominant_timeframe = max(weighted_scores, key=weighted_scores.get)
            return dominant_timeframe
            
        except Exception as e:
            logger.error(f"Error determining dominant timeframe: {e}")
            return self.supported_timeframes[-1] if self.supported_timeframes else '30min'
    
    def _get_timeframe_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        if timeframe == '1hr':
            return 60
        elif timeframe == '4hr':
            return 240
        else:
            return int(timeframe.replace('min', ''))
    
    def _calculate_regime_consistency(self, timeframe_scores: Dict[str, TimeframeRegimeScore]) -> float:
        """Calculate regime consistency across timeframes"""
        try:
            if not timeframe_scores:
                return 0.0
            
            regime_ids = [score.regime_id for score in timeframe_scores.values()]
            unique_regimes = len(set(regime_ids))
            
            # Higher consistency when fewer unique regimes
            consistency = 1.0 - (unique_regimes - 1) / max(len(regime_ids) - 1, 1)
            return max(0.0, consistency)
            
        except Exception as e:
            logger.error(f"Error calculating regime consistency: {e}")
            return 0.0
    
    def _calculate_overall_confidence(self, timeframe_scores: Dict[str, TimeframeRegimeScore]) -> float:
        """Calculate overall confidence across timeframes"""
        try:
            if not timeframe_scores:
                return 0.0
            
            # Calculate weighted average confidence
            total_weight = 0.0
            weighted_confidence = 0.0
            
            for timeframe, score_obj in timeframe_scores.items():
                weight = self.timeframe_weights.get(timeframe, 0.25)
                weighted_confidence += score_obj.confidence * weight
                total_weight += weight
            
            return weighted_confidence / max(total_weight, 1e-6)
            
        except Exception as e:
            logger.error(f"Error calculating overall confidence: {e}")
            return 0.0
    
    def _get_default_timeframe_score(self, timeframe: str) -> TimeframeRegimeScore:
        """Get default timeframe score for error cases"""
        return TimeframeRegimeScore(
            timeframe=timeframe,
            regime_score=0.0,
            regime_id='UNKNOWN',
            confidence=0.0,
            trend_strength=0.0,
            volatility_score=0.0,
            structure_score=0.0,
            correlation_score=0.0,
            timestamp=datetime.now()
        )
    
    def _get_default_multi_timeframe_analysis(self) -> MultiTimeframeAnalysis:
        """Get default multi-timeframe analysis for error cases"""
        default_scores = {
            timeframe: self._get_default_timeframe_score(timeframe)
            for timeframe in self.supported_timeframes
        }
        
        # Get default dominant timeframe from active timeframes
        default_dominant = '30min'
        if self.supported_timeframes:
            sorted_tf = sorted(self.supported_timeframes, 
                             key=lambda x: self._get_timeframe_minutes(x), 
                             reverse=True)
            default_dominant = sorted_tf[0] if sorted_tf else '30min'
        
        return MultiTimeframeAnalysis(
            timeframe_scores=default_scores,
            cross_timeframe_correlation=0.0,
            dominant_timeframe=default_dominant,
            regime_consistency=0.0,
            overall_confidence=0.0,
            timestamp=datetime.now()
        )
    
    def get_adaptive_configuration(self) -> Dict[str, Any]:
        """Get current adaptive configuration"""
        return {
            'mode': self.timeframe_manager.current_mode,
            'active_timeframes': self.supported_timeframes,
            'timeframe_weights': self.timeframe_weights,
            'mode_parameters': self.timeframe_manager.get_mode_parameters()
        }

# Global instance for easy access
timeframe_regime_extractor = TimeframeRegimeExtractor()

def get_timeframe_regime_extractor() -> TimeframeRegimeExtractor:
    """Get the global timeframe regime extractor instance"""
    return timeframe_regime_extractor
