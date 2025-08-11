"""
Rolling Straddle Structure Definition for Component 1

Time-series rolling straddle calculation with minute-by-minute ATM/ITM1/OTM1 
dynamic selection as spot price moves. Revolutionary approach applying technical 
indicators to rolling straddle prices instead of underlying prices.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd

# GPU acceleration
try:
    import cudf
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


@dataclass
class RollingStraddlePoint:
    """Single rolling straddle data point"""
    timestamp: str
    atm_straddle: float
    itm1_straddle: float
    otm1_straddle: float
    atm_ce: float
    atm_pe: float
    itm1_ce: float
    itm1_pe: float
    otm1_ce: float
    otm1_pe: float
    combined_volume: float
    spot: float
    dte: int
    expiry_date: str


@dataclass
class RollingStraddleTimeSeries:
    """Time series of rolling straddle data"""
    timestamps: List[str]
    atm_straddle_series: np.ndarray
    itm1_straddle_series: np.ndarray
    otm1_straddle_series: np.ndarray
    volume_series: np.ndarray
    spot_series: np.ndarray
    processing_time_ms: float
    data_points: int
    missing_data_count: int
    metadata: Dict[str, Any]


class RollingStraddleEngine:
    """
    Revolutionary Rolling Straddle Engine
    
    Implements time-series rolling straddle calculation where strikes dynamically
    adjust minute-by-minute as spot price moves. Technical indicators (EMA, VWAP,
    Pivots) are applied to these rolling straddle prices, not underlying prices.
    
    Key Features:
    - Time-series rolling behavior (strikes "roll" each minute)
    - ATM/ITM1/OTM1 dynamic strike selection
    - Symmetric straddle calculation (ce_close + pe_close)
    - Volume combination for VWAP (ce_volume + pe_volume)
    - GPU-accelerated processing
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Rolling Straddle Engine
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Performance configuration
        self.processing_budget_ms = config.get('processing_budget_ms', 50)  # Component budget
        self.use_gpu = config.get('use_gpu', GPU_AVAILABLE)
        
        # Rolling straddle configuration
        self.target_strikes = ['ATM', 'ITM1', 'OTM1']
        self.handle_missing_data = config.get('handle_missing_data', True)
        self.forward_fill_limit = config.get('forward_fill_limit', 5)  # Max minutes to forward-fill
        
        # Validation thresholds
        self.min_straddle_value = config.get('min_straddle_value', 0.01)
        self.max_straddle_value = config.get('max_straddle_value', 10000.0)
        self.min_volume = config.get('min_volume', 1)
        
        self.logger.info(f"RollingStraddleEngine initialized with GPU={self.use_gpu}")
    
    async def calculate_rolling_straddle_series(self, 
                                              filtered_data: Union[pd.DataFrame, 'cudf.DataFrame']) -> RollingStraddleTimeSeries:
        """
        Calculate complete rolling straddle time series
        
        Args:
            filtered_data: Filtered DataFrame with straddle columns
            
        Returns:
            RollingStraddleTimeSeries with complete time series data
        """
        start_time = time.time()
        
        try:
            # Group by timestamp for minute-by-minute processing
            unique_timestamps = sorted(filtered_data['trade_time'].unique())
            
            # Initialize arrays for time series
            timestamps = []
            atm_straddles = []
            itm1_straddles = []
            otm1_straddles = []
            volumes = []
            spots = []
            missing_count = 0
            
            self.logger.info(f"Processing {len(unique_timestamps)} timestamps for rolling straddle calculation")
            
            # Process each timestamp
            for timestamp in unique_timestamps:
                # Get data for this minute
                timestamp_data = filtered_data[filtered_data['trade_time'] == timestamp]
                
                # Calculate rolling straddle point
                straddle_point = await self._calculate_straddle_point(timestamp_data, timestamp)
                
                if straddle_point:
                    timestamps.append(straddle_point.timestamp)
                    atm_straddles.append(straddle_point.atm_straddle)
                    itm1_straddles.append(straddle_point.itm1_straddle)
                    otm1_straddles.append(straddle_point.otm1_straddle)
                    volumes.append(straddle_point.combined_volume)
                    spots.append(straddle_point.spot)
                else:
                    missing_count += 1
            
            # Convert to numpy arrays
            atm_series = np.array(atm_straddles)
            itm1_series = np.array(itm1_straddles)
            otm1_series = np.array(otm1_straddles)
            volume_series = np.array(volumes)
            spot_series = np.array(spots)
            
            # Handle missing data with forward fill
            if self.handle_missing_data and missing_count > 0:
                atm_series = self._forward_fill_array(atm_series)
                itm1_series = self._forward_fill_array(itm1_series)
                otm1_series = self._forward_fill_array(otm1_series)
                volume_series = self._forward_fill_array(volume_series)
                
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Performance validation
            if processing_time > self.processing_budget_ms:
                self.logger.warning(f"Rolling straddle processing {processing_time:.2f}ms exceeded budget {self.processing_budget_ms}ms")
            
            return RollingStraddleTimeSeries(
                timestamps=timestamps,
                atm_straddle_series=atm_series,
                itm1_straddle_series=itm1_series,
                otm1_straddle_series=otm1_series,
                volume_series=volume_series,
                spot_series=spot_series,
                processing_time_ms=processing_time,
                data_points=len(timestamps),
                missing_data_count=missing_count,
                metadata={
                    'target_strikes': self.target_strikes,
                    'min_straddle': float(np.min([atm_series.min(), itm1_series.min(), otm1_series.min()])),
                    'max_straddle': float(np.max([atm_series.max(), itm1_series.max(), otm1_series.max()])),
                    'total_volume': float(np.sum(volume_series)),
                    'spot_range': (float(np.min(spot_series)), float(np.max(spot_series)))
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to calculate rolling straddle series: {e}")
            raise
    
    async def _calculate_straddle_point(self, 
                                      timestamp_data: Union[pd.DataFrame, 'cudf.DataFrame'], 
                                      timestamp: str) -> Optional[RollingStraddlePoint]:
        """
        Calculate rolling straddle point for a single timestamp
        
        Args:
            timestamp_data: Data for single timestamp
            timestamp: Current timestamp string
            
        Returns:
            RollingStraddlePoint or None if calculation fails
        """
        try:
            # Use nearest expiry only (first expiry in data)
            nearest_expiry = timestamp_data['expiry_date'].iloc[0]
            expiry_data = timestamp_data[timestamp_data['expiry_date'] == nearest_expiry]
            
            # Extract rolling straddles using optimized approach
            atm_straddle = self._extract_straddle_price(expiry_data, 'ATM', 'ATM')
            itm1_straddle = self._extract_straddle_price(expiry_data, 'ITM1', 'OTM1')  # Bullish bias
            otm1_straddle = self._extract_straddle_price(expiry_data, 'OTM1', 'ITM1')  # Bearish bias
            
            # Extract individual CE/PE components
            atm_ce, atm_pe = self._extract_individual_prices(expiry_data, 'ATM', 'ATM')
            itm1_ce, itm1_pe = self._extract_individual_prices(expiry_data, 'ITM1', 'OTM1')
            otm1_ce, otm1_pe = self._extract_individual_prices(expiry_data, 'OTM1', 'ITM1')
            
            # Calculate combined volume (ce_volume + pe_volume)
            combined_volume = self._calculate_combined_volume(expiry_data)
            
            # Get spot and metadata
            spot = float(expiry_data['spot'].iloc[0])
            dte = int(expiry_data['dte'].iloc[0])
            
            # Validate straddle values
            if not self._validate_straddle_values([atm_straddle, itm1_straddle, otm1_straddle]):
                self.logger.warning(f"Invalid straddle values at {timestamp}")
                return None
            
            return RollingStraddlePoint(
                timestamp=timestamp,
                atm_straddle=atm_straddle,
                itm1_straddle=itm1_straddle,
                otm1_straddle=otm1_straddle,
                atm_ce=atm_ce,
                atm_pe=atm_pe,
                itm1_ce=itm1_ce,
                itm1_pe=itm1_pe,
                otm1_ce=otm1_ce,
                otm1_pe=otm1_pe,
                combined_volume=combined_volume,
                spot=spot,
                dte=dte,
                expiry_date=str(nearest_expiry)
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate straddle point for {timestamp}: {e}")
            return None
    
    def _extract_straddle_price(self, 
                               data: Union[pd.DataFrame, 'cudf.DataFrame'], 
                               call_strike_type: str, 
                               put_strike_type: str) -> float:
        """
        Extract straddle price for specific strike combination
        
        Args:
            data: Expiry-filtered data
            call_strike_type: Call strike type (ATM, ITM1, OTM1)
            put_strike_type: Put strike type (ATM, ITM1, OTM1)
            
        Returns:
            Straddle price (ce_close + pe_close)
        """
        try:
            # Filter to specific strike combination
            straddle_data = data[
                (data['call_strike_type'] == call_strike_type) & 
                (data['put_strike_type'] == put_strike_type)
            ]
            
            if len(straddle_data) == 0:
                # Fallback: try to find any matching strikes
                straddle_data = data[
                    (data['call_strike_type'] == call_strike_type) | 
                    (data['put_strike_type'] == put_strike_type)
                ]
                
                if len(straddle_data) == 0:
                    return 0.0
            
            # Calculate straddle price (ce_close + pe_close)
            ce_close = float(straddle_data['ce_close'].iloc[0])
            pe_close = float(straddle_data['pe_close'].iloc[0])
            
            return ce_close + pe_close
            
        except Exception as e:
            self.logger.debug(f"Failed to extract straddle for {call_strike_type}/{put_strike_type}: {e}")
            return 0.0
    
    def _extract_individual_prices(self, 
                                  data: Union[pd.DataFrame, 'cudf.DataFrame'], 
                                  call_strike_type: str, 
                                  put_strike_type: str) -> Tuple[float, float]:
        """
        Extract individual CE and PE prices
        
        Args:
            data: Expiry-filtered data
            call_strike_type: Call strike type
            put_strike_type: Put strike type
            
        Returns:
            Tuple of (ce_close, pe_close)
        """
        try:
            straddle_data = data[
                (data['call_strike_type'] == call_strike_type) & 
                (data['put_strike_type'] == put_strike_type)
            ]
            
            if len(straddle_data) == 0:
                return 0.0, 0.0
            
            ce_close = float(straddle_data['ce_close'].iloc[0])
            pe_close = float(straddle_data['pe_close'].iloc[0])
            
            return ce_close, pe_close
            
        except:
            return 0.0, 0.0
    
    def _calculate_combined_volume(self, data: Union[pd.DataFrame, 'cudf.DataFrame']) -> float:
        """
        Calculate combined volume (ce_volume + pe_volume) for VWAP calculation
        
        Args:
            data: Expiry-filtered data
            
        Returns:
            Combined volume
        """
        try:
            # Sum all CE and PE volumes for this timestamp
            ce_volumes = data['ce_volume'].fillna(0)
            pe_volumes = data['pe_volume'].fillna(0)
            
            total_ce_volume = float(ce_volumes.sum())
            total_pe_volume = float(pe_volumes.sum())
            
            return total_ce_volume + total_pe_volume
            
        except:
            return 0.0
    
    def _validate_straddle_values(self, straddle_values: List[float]) -> bool:
        """
        Validate straddle values for sanity
        
        Args:
            straddle_values: List of straddle prices
            
        Returns:
            True if values are valid
        """
        for value in straddle_values:
            if np.isnan(value) or np.isinf(value):
                return False
            if value < self.min_straddle_value or value > self.max_straddle_value:
                return False
        return True
    
    def _forward_fill_array(self, array: np.ndarray, limit: Optional[int] = None) -> np.ndarray:
        """
        Forward fill missing values in numpy array
        
        Args:
            array: Input array with potential NaN values
            limit: Maximum number of consecutive fills
            
        Returns:
            Forward-filled array
        """
        if limit is None:
            limit = self.forward_fill_limit
        
        # Convert to pandas for forward fill, then back to numpy
        series = pd.Series(array)
        filled_series = series.fillna(method='ffill', limit=limit)
        return filled_series.values
    
    async def get_straddle_summary(self, time_series: RollingStraddleTimeSeries) -> Dict[str, Any]:
        """
        Get summary statistics for rolling straddle time series
        
        Args:
            time_series: RollingStraddleTimeSeries data
            
        Returns:
            Summary statistics dictionary
        """
        return {
            'data_points': time_series.data_points,
            'missing_data_count': time_series.missing_data_count,
            'processing_time_ms': time_series.processing_time_ms,
            'atm_straddle_stats': {
                'mean': float(np.mean(time_series.atm_straddle_series)),
                'std': float(np.std(time_series.atm_straddle_series)),
                'min': float(np.min(time_series.atm_straddle_series)),
                'max': float(np.max(time_series.atm_straddle_series))
            },
            'itm1_straddle_stats': {
                'mean': float(np.mean(time_series.itm1_straddle_series)),
                'std': float(np.std(time_series.itm1_straddle_series)),
                'min': float(np.min(time_series.itm1_straddle_series)),
                'max': float(np.max(time_series.itm1_straddle_series))
            },
            'otm1_straddle_stats': {
                'mean': float(np.mean(time_series.otm1_straddle_series)),
                'std': float(np.std(time_series.otm1_straddle_series)),
                'min': float(np.min(time_series.otm1_straddle_series)),
                'max': float(np.max(time_series.otm1_straddle_series))
            },
            'volume_stats': {
                'total': float(np.sum(time_series.volume_series)),
                'mean': float(np.mean(time_series.volume_series)),
                'std': float(np.std(time_series.volume_series))
            },
            'spot_range': {
                'min': float(np.min(time_series.spot_series)),
                'max': float(np.max(time_series.spot_series)),
                'range': float(np.max(time_series.spot_series) - np.min(time_series.spot_series))
            }
        }
    
    async def validate_time_series_continuity(self, time_series: RollingStraddleTimeSeries) -> Dict[str, Any]:
        """
        Validate time series continuity and detect gaps
        
        Args:
            time_series: RollingStraddleTimeSeries data
            
        Returns:
            Continuity validation results
        """
        timestamps = pd.to_datetime(time_series.timestamps)
        
        # Detect time gaps
        time_diffs = timestamps.diff()[1:]  # Skip first NaT
        expected_interval = pd.Timedelta(minutes=1)
        
        gaps = time_diffs[time_diffs > expected_interval * 1.5]  # Allow 50% tolerance
        
        return {
            'total_timestamps': len(timestamps),
            'time_gaps_detected': len(gaps),
            'largest_gap_minutes': float(gaps.max().total_seconds() / 60) if len(gaps) > 0 else 0,
            'continuity_percentage': float((len(timestamps) - len(gaps)) / len(timestamps) * 100),
            'gap_locations': [str(ts) for ts in gaps.index] if len(gaps) > 0 else []
        }


# Factory function
def create_rolling_straddle_engine(config: Dict[str, Any]) -> RollingStraddleEngine:
    """Create and configure RollingStraddleEngine instance"""
    return RollingStraddleEngine(config)