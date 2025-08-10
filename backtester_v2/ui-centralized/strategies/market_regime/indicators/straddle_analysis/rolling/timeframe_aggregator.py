"""
Timeframe Aggregator for Rolling Analysis

Aggregates data across different timeframes for comprehensive analysis.
Supports [3, 5, 10, 15] minute windows with efficient data management.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class TimeframeData:
    """Data for a specific timeframe"""
    timeframe: int  # minutes
    data_points: deque
    aggregated_stats: Dict[str, float]
    last_update: pd.Timestamp


class TimeframeAggregator:
    """
    Aggregates market data across multiple timeframes for rolling analysis
    """
    
    def __init__(self, timeframes: List[int] = None):
        """
        Initialize timeframe aggregator
        
        Args:
            timeframes: List of timeframes in minutes (default: [3, 5, 10, 15])
        """
        self.timeframes = timeframes or [3, 5, 10, 15]
        self.max_timeframe = max(self.timeframes)
        
        # Storage for each timeframe
        self.timeframe_data = {}
        for tf in self.timeframes:
            self.timeframe_data[tf] = TimeframeData(
                timeframe=tf,
                data_points=deque(maxlen=100),  # Keep last 100 data points per timeframe
                aggregated_stats={},
                last_update=pd.Timestamp.now()
            )
        
        # Raw data storage
        self.raw_data = deque(maxlen=1000)  # Keep raw tick data
        
        logger.info(f"TimeframeAggregator initialized with timeframes: {self.timeframes}")
    
    def add_data_point(self, timestamp: pd.Timestamp, data: Dict[str, Any]):
        """
        Add a new data point and update all timeframes
        
        Args:
            timestamp: Timestamp of the data point
            data: Market data dictionary
        """
        try:
            # Store raw data
            data_point = {
                'timestamp': timestamp,
                **data
            }
            self.raw_data.append(data_point)
            
            # Update each timeframe
            for tf in self.timeframes:
                self._update_timeframe(tf, timestamp, data)
            
        except Exception as e:
            logger.error(f"Error adding data point: {e}")
    
    def _update_timeframe(self, timeframe: int, timestamp: pd.Timestamp, data: Dict[str, Any]):
        """Update a specific timeframe with new data"""
        tf_data = self.timeframe_data[timeframe]
        
        # Check if we need to create a new timeframe bar
        if self._should_create_new_bar(tf_data, timestamp, timeframe):
            # Aggregate data since last update
            aggregated = self._aggregate_data_for_timeframe(timeframe, timestamp)
            
            if aggregated:
                tf_data.data_points.append(aggregated)
                tf_data.last_update = timestamp
                
                # Update rolling statistics
                self._update_rolling_stats(tf_data)
    
    def _should_create_new_bar(
        self, 
        tf_data: TimeframeData, 
        current_timestamp: pd.Timestamp, 
        timeframe: int
    ) -> bool:
        """Check if we should create a new timeframe bar"""
        
        if not tf_data.data_points:
            return True  # First data point
        
        # Calculate time difference
        time_diff = (current_timestamp - tf_data.last_update).total_seconds() / 60
        
        return time_diff >= timeframe
    
    def _aggregate_data_for_timeframe(
        self, 
        timeframe: int, 
        end_timestamp: pd.Timestamp
    ) -> Optional[Dict[str, Any]]:
        """Aggregate raw data for a specific timeframe"""
        
        try:
            # Find data points within the timeframe
            start_timestamp = end_timestamp - pd.Timedelta(minutes=timeframe)
            
            relevant_data = [
                dp for dp in self.raw_data
                if start_timestamp <= dp['timestamp'] <= end_timestamp
            ]
            
            if not relevant_data:
                return None
            
            # Aggregate the data
            aggregated = {
                'timestamp': end_timestamp,
                'timeframe': timeframe,
                'data_count': len(relevant_data)
            }
            
            # Price aggregation
            price_fields = [
                'underlying_price', 'ATM_CE', 'ATM_PE', 'ITM1_CE', 
                'ITM1_PE', 'OTM1_CE', 'OTM1_PE'
            ]
            
            for field in price_fields:
                values = [dp.get(field, 0) for dp in relevant_data if dp.get(field) is not None]
                if values:
                    aggregated[f'{field}_open'] = values[0]
                    aggregated[f'{field}_high'] = max(values)
                    aggregated[f'{field}_low'] = min(values)
                    aggregated[f'{field}_close'] = values[-1]
                    aggregated[f'{field}_avg'] = np.mean(values)
            
            # Volume aggregation
            volume_fields = ['volume', 'open_interest']
            for field in volume_fields:
                values = [dp.get(field, 0) for dp in relevant_data if dp.get(field) is not None]
                if values:
                    aggregated[f'{field}_total'] = sum(values)
                    aggregated[f'{field}_avg'] = np.mean(values)
            
            # Greeks aggregation (use last available values)
            greek_fields = [
                'atm_ce_delta', 'atm_pe_delta', 'atm_ce_gamma', 'atm_pe_gamma',
                'atm_ce_theta', 'atm_pe_theta', 'atm_ce_vega', 'atm_pe_vega'
            ]
            
            for field in greek_fields:
                values = [dp.get(field) for dp in relevant_data if dp.get(field) is not None]
                if values:
                    aggregated[field] = values[-1]  # Use most recent
            
            # Calculate straddle prices
            self._calculate_straddle_prices(aggregated)
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Error aggregating data for timeframe {timeframe}: {e}")
            return None
    
    def _calculate_straddle_prices(self, aggregated: Dict[str, Any]):
        """Calculate straddle combination prices from aggregated data"""
        
        try:
            # ATM Straddle
            atm_ce = aggregated.get('ATM_CE_close', 0)
            atm_pe = aggregated.get('ATM_PE_close', 0)
            if atm_ce and atm_pe:
                aggregated['atm_straddle_price'] = atm_ce + atm_pe
            
            # ITM1 Straddle
            itm1_ce = aggregated.get('ITM1_CE_close', 0)
            itm1_pe = aggregated.get('ITM1_PE_close', 0)
            if itm1_ce and itm1_pe:
                aggregated['itm1_straddle_price'] = itm1_ce + itm1_pe
            
            # OTM1 Straddle
            otm1_ce = aggregated.get('OTM1_CE_close', 0)
            otm1_pe = aggregated.get('OTM1_PE_close', 0)
            if otm1_ce and otm1_pe:
                aggregated['otm1_straddle_price'] = otm1_ce + otm1_pe
            
            # Combined straddle (equal weight for now)
            straddle_prices = []
            for straddle in ['atm_straddle_price', 'itm1_straddle_price', 'otm1_straddle_price']:
                if aggregated.get(straddle):
                    straddle_prices.append(aggregated[straddle])
            
            if straddle_prices:
                aggregated['combined_straddle_price'] = np.mean(straddle_prices)
            
        except Exception as e:
            logger.error(f"Error calculating straddle prices: {e}")
    
    def _update_rolling_stats(self, tf_data: TimeframeData):
        """Update rolling statistics for a timeframe"""
        
        try:
            if len(tf_data.data_points) < 2:
                return
            
            # Get recent data points
            recent_data = list(tf_data.data_points)[-20:]  # Last 20 bars
            
            # Calculate moving averages
            price_fields = ['underlying_price_close', 'atm_straddle_price', 'combined_straddle_price']
            
            for field in price_fields:
                values = [dp.get(field) for dp in recent_data if dp.get(field) is not None]
                if len(values) >= 2:
                    tf_data.aggregated_stats[f'{field}_ma5'] = np.mean(values[-5:]) if len(values) >= 5 else np.mean(values)
                    tf_data.aggregated_stats[f'{field}_ma10'] = np.mean(values[-10:]) if len(values) >= 10 else np.mean(values)
                    tf_data.aggregated_stats[f'{field}_volatility'] = np.std(values) if len(values) > 1 else 0
            
            # Calculate trend indicators
            self._calculate_trend_indicators(tf_data, recent_data)
            
        except Exception as e:
            logger.error(f"Error updating rolling stats: {e}")
    
    def _calculate_trend_indicators(self, tf_data: TimeframeData, recent_data: List[Dict[str, Any]]):
        """Calculate trend indicators for the timeframe"""
        
        try:
            if len(recent_data) < 5:
                return
            
            # Price momentum
            prices = [dp.get('underlying_price_close', 0) for dp in recent_data[-10:]]
            if len(prices) >= 2:
                price_change = (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0
                tf_data.aggregated_stats['price_momentum'] = price_change
            
            # Straddle momentum
            straddle_prices = [dp.get('combined_straddle_price', 0) for dp in recent_data[-5:]]
            if len(straddle_prices) >= 2:
                straddle_change = (straddle_prices[-1] - straddle_prices[0]) / straddle_prices[0] if straddle_prices[0] != 0 else 0
                tf_data.aggregated_stats['straddle_momentum'] = straddle_change
            
            # Volume trend
            volumes = [dp.get('volume_avg', 0) for dp in recent_data[-5:]]
            if len(volumes) >= 2:
                volume_trend = np.polyfit(range(len(volumes)), volumes, 1)[0]  # Linear slope
                tf_data.aggregated_stats['volume_trend'] = volume_trend
            
        except Exception as e:
            logger.error(f"Error calculating trend indicators: {e}")
    
    def get_timeframe_data(self, timeframe: int) -> Optional[TimeframeData]:
        """Get data for a specific timeframe"""
        return self.timeframe_data.get(timeframe)
    
    def get_latest_aggregated_data(self, timeframe: int) -> Optional[Dict[str, Any]]:
        """Get the latest aggregated data for a timeframe"""
        tf_data = self.timeframe_data.get(timeframe)
        if tf_data and tf_data.data_points:
            return tf_data.data_points[-1]
        return None
    
    def get_rolling_statistics(self, timeframe: int) -> Dict[str, float]:
        """Get rolling statistics for a timeframe"""
        tf_data = self.timeframe_data.get(timeframe)
        if tf_data:
            return tf_data.aggregated_stats.copy()
        return {}
    
    def get_cross_timeframe_analysis(self) -> Dict[str, Any]:
        """Get analysis across all timeframes"""
        
        analysis = {
            'timeframes_analyzed': self.timeframes,
            'data_availability': {},
            'momentum_alignment': {},
            'volatility_structure': {}
        }
        
        try:
            # Check data availability
            for tf in self.timeframes:
                tf_data = self.timeframe_data.get(tf)
                analysis['data_availability'][tf] = len(tf_data.data_points) if tf_data else 0
            
            # Momentum alignment across timeframes
            momentums = {}
            for tf in self.timeframes:
                stats = self.get_rolling_statistics(tf)
                momentum = stats.get('price_momentum', 0)
                momentums[tf] = momentum
            
            analysis['momentum_alignment'] = momentums
            
            # Check if momentum is aligned (same direction)
            positive_momentum = sum(1 for m in momentums.values() if m > 0.001)
            negative_momentum = sum(1 for m in momentums.values() if m < -0.001)
            
            if positive_momentum >= len(self.timeframes) * 0.75:
                analysis['momentum_consensus'] = 'bullish'
            elif negative_momentum >= len(self.timeframes) * 0.75:
                analysis['momentum_consensus'] = 'bearish'
            else:
                analysis['momentum_consensus'] = 'mixed'
            
            # Volatility structure
            volatilities = {}
            for tf in self.timeframes:
                stats = self.get_rolling_statistics(tf)
                vol = stats.get('underlying_price_close_volatility', 0)
                volatilities[tf] = vol
            
            analysis['volatility_structure'] = volatilities
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in cross-timeframe analysis: {e}")
            return analysis
    
    def cleanup_old_data(self, cutoff_hours: int = 24):
        """Remove data older than specified hours"""
        
        try:
            cutoff_time = pd.Timestamp.now() - pd.Timedelta(hours=cutoff_hours)
            
            # Clean raw data
            while self.raw_data and self.raw_data[0]['timestamp'] < cutoff_time:
                self.raw_data.popleft()
            
            # Clean timeframe data
            for tf_data in self.timeframe_data.values():
                while (tf_data.data_points and 
                       tf_data.data_points[0].get('timestamp', pd.Timestamp.now()) < cutoff_time):
                    tf_data.data_points.popleft()
            
            logger.info(f"Cleaned data older than {cutoff_hours} hours")
            
        except Exception as e:
            logger.error(f"Error cleaning old data: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the aggregator"""
        
        status = {
            'timeframes': self.timeframes,
            'raw_data_points': len(self.raw_data),
            'timeframe_status': {}
        }
        
        for tf in self.timeframes:
            tf_data = self.timeframe_data[tf]
            status['timeframe_status'][tf] = {
                'data_points': len(tf_data.data_points),
                'last_update': tf_data.last_update.isoformat() if tf_data.last_update else None,
                'stats_available': len(tf_data.aggregated_stats)
            }
        
        return status