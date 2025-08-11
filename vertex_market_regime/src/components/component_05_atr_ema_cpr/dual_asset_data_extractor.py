"""
Component 5: Dual-Asset Data Extraction System

Comprehensive data extraction system for ATR-EMA-CPR analysis supporting both
straddle price construction and underlying price analysis across multiple
timeframes with complete production schema integration.

Features:
- Rolling straddle price construction from ce_open/ce_close, pe_open/pe_close
- Multi-timeframe underlying price extraction (daily/weekly/monthly)
- Volume/OI integration for cross-validation
- Zone-based analysis across 4 production zones
- Complete 48-column production schema alignment
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import asyncio
import time
import warnings

warnings.filterwarnings('ignore')


@dataclass
class StraddlePriceData:
    """Straddle price data structure for ATR-EMA-CPR analysis"""
    straddle_open: np.ndarray
    straddle_high: np.ndarray
    straddle_low: np.ndarray
    straddle_close: np.ndarray
    straddle_volume: np.ndarray
    straddle_oi: np.ndarray
    timestamps: np.ndarray
    dte_values: np.ndarray
    zone_names: List[str]
    metadata: Dict[str, Any]


@dataclass
class UnderlyingPriceData:
    """Underlying price data structure for multi-timeframe ATR-EMA-CPR analysis"""
    spot_prices: np.ndarray
    future_open: np.ndarray
    future_high: np.ndarray
    future_low: np.ndarray
    future_close: np.ndarray
    future_volume: np.ndarray
    future_oi: np.ndarray
    timestamps: np.ndarray
    timeframes: Dict[str, Dict[str, np.ndarray]]  # daily/weekly/monthly data
    metadata: Dict[str, Any]


@dataclass
class DualAssetExtractionResult:
    """Complete dual-asset data extraction result"""
    straddle_data: StraddlePriceData
    underlying_data: UnderlyingPriceData
    extraction_time_ms: float
    data_quality_score: float
    zone_coverage: Dict[str, int]
    dte_coverage: Dict[int, int]
    metadata: Dict[str, Any]


class DualAssetDataExtractor:
    """
    Dual-Asset Data Extraction System for Component 5
    
    Extracts and processes both straddle prices and underlying prices
    for comprehensive ATR-EMA-CPR analysis with production alignment.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize dual-asset data extractor
        
        Args:
            config: Extractor configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # Extraction parameters
        self.required_zones = ['OPEN', 'MID_MORN', 'LUNCH', 'AFTERNOON', 'CLOSE']
        self.target_dte_range = (0, 90)
        self.min_data_quality_threshold = 0.85
        
        # Straddle construction parameters
        self.straddle_calculation_method = config.get('straddle_method', 'atm_rolling')
        self.rolling_window = config.get('rolling_window', 5)
        
        # Multi-timeframe parameters
        self.timeframes = {
            'daily': {'atr_periods': [14, 21, 50], 'ema_periods': [20, 50, 100, 200]},
            'weekly': {'atr_periods': [14, 21, 50], 'ema_periods': [10, 20, 50]},
            'monthly': {'atr_periods': [14, 21], 'ema_periods': [6, 12, 24]}
        }
        
        self.logger.info(f"Initialized dual-asset data extractor for Component 5")

    async def extract_dual_asset_data(self, parquet_data: pd.DataFrame) -> DualAssetExtractionResult:
        """
        Extract dual-asset data for ATR-EMA-CPR analysis
        
        Args:
            parquet_data: Production parquet data (48-column schema)
            
        Returns:
            DualAssetExtractionResult with both straddle and underlying data
        """
        start_time = time.time()
        
        try:
            # Extract straddle price data
            straddle_data = await self._extract_straddle_prices(parquet_data)
            
            # Extract underlying price data with multi-timeframe analysis
            underlying_data = await self._extract_underlying_prices(parquet_data)
            
            # Validate data quality
            data_quality_score = self._validate_data_quality(straddle_data, underlying_data)
            
            # Calculate coverage metrics
            zone_coverage = self._calculate_zone_coverage(parquet_data)
            dte_coverage = self._calculate_dte_coverage(parquet_data)
            
            extraction_time_ms = (time.time() - start_time) * 1000
            
            return DualAssetExtractionResult(
                straddle_data=straddle_data,
                underlying_data=underlying_data,
                extraction_time_ms=extraction_time_ms,
                data_quality_score=data_quality_score,
                zone_coverage=zone_coverage,
                dte_coverage=dte_coverage,
                metadata={
                    'extraction_method': 'dual_asset_comprehensive',
                    'records_processed': len(parquet_data),
                    'straddle_method': self.straddle_calculation_method,
                    'timeframes_analyzed': list(self.timeframes.keys()),
                    'timestamp': datetime.utcnow()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Dual-asset extraction failed: {str(e)}")
            raise

    async def _extract_straddle_prices(self, data: pd.DataFrame) -> StraddlePriceData:
        """Extract rolling straddle prices from CE/PE data"""
        
        # Group by timestamp and DTE for straddle construction
        straddle_prices = []
        straddle_volumes = []
        straddle_ois = []
        timestamps = []
        dte_values = []
        zone_names = []
        
        # Process each timestamp group
        for (timestamp, dte), group in data.groupby(['timestamp', 'dte']):
            if len(group) < 2:  # Need both CE and PE data
                continue
                
            # Extract CE and PE data
            ce_data = group[group['strike_type'] == 'CE']
            pe_data = group[group['strike_type'] == 'PE']
            
            if len(ce_data) == 0 or len(pe_data) == 0:
                continue
            
            # Find ATM or closest strikes for straddle construction
            spot_price = group['spot'].iloc[0]
            atm_strike = self._find_atm_strike(ce_data, pe_data, spot_price)
            
            # Get ATM CE and PE data
            atm_ce = ce_data[ce_data['strike'] == atm_strike].iloc[0] if len(ce_data[ce_data['strike'] == atm_strike]) > 0 else ce_data.iloc[0]
            atm_pe = pe_data[pe_data['strike'] == atm_strike].iloc[0] if len(pe_data[pe_data['strike'] == atm_strike]) > 0 else pe_data.iloc[0]
            
            # Calculate straddle prices (CE + PE)
            straddle_open = atm_ce['ce_open'] + atm_pe['pe_open']
            straddle_high = max(atm_ce['ce_high'], atm_pe['pe_high']) + max(atm_ce['ce_high'], atm_pe['pe_high'])  # Conservative estimate
            straddle_low = min(atm_ce['ce_low'], atm_pe['pe_low']) + min(atm_ce['ce_low'], atm_pe['pe_low'])  # Conservative estimate  
            straddle_close = atm_ce['ce_close'] + atm_pe['pe_close']
            
            # Calculate combined volume and OI
            straddle_volume = atm_ce['ce_volume'] + atm_pe['pe_volume']
            straddle_oi = atm_ce['ce_oi'] + atm_pe['pe_oi']
            
            straddle_prices.append([straddle_open, straddle_high, straddle_low, straddle_close])
            straddle_volumes.append(straddle_volume)
            straddle_ois.append(straddle_oi)
            timestamps.append(timestamp)
            dte_values.append(dte)
            zone_names.append(group['zone_name'].iloc[0])
        
        # Convert to numpy arrays
        straddle_array = np.array(straddle_prices)
        
        return StraddlePriceData(
            straddle_open=straddle_array[:, 0],
            straddle_high=straddle_array[:, 1], 
            straddle_low=straddle_array[:, 2],
            straddle_close=straddle_array[:, 3],
            straddle_volume=np.array(straddle_volumes),
            straddle_oi=np.array(straddle_ois),
            timestamps=np.array(timestamps),
            dte_values=np.array(dte_values),
            zone_names=zone_names,
            metadata={
                'construction_method': self.straddle_calculation_method,
                'data_points': len(straddle_prices),
                'unique_dtes': len(set(dte_values)),
                'unique_zones': len(set(zone_names))
            }
        )

    async def _extract_underlying_prices(self, data: pd.DataFrame) -> UnderlyingPriceData:
        """Extract underlying prices with multi-timeframe analysis"""
        
        # Extract unique timestamp data (avoiding duplicates from multiple strikes)
        underlying_data = data.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        
        # Base underlying price arrays
        spot_prices = underlying_data['spot'].values
        future_open = underlying_data['future_open'].values
        future_high = underlying_data['future_high'].values
        future_low = underlying_data['future_low'].values
        future_close = underlying_data['future_close'].values
        future_volume = underlying_data['future_volume'].values
        future_oi = underlying_data['future_oi'].values
        timestamps = underlying_data['timestamp'].values
        
        # Create multi-timeframe data
        timeframes_data = {}
        for timeframe, params in self.timeframes.items():
            tf_data = await self._create_timeframe_data(
                underlying_data, timeframe, params
            )
            timeframes_data[timeframe] = tf_data
        
        return UnderlyingPriceData(
            spot_prices=spot_prices,
            future_open=future_open,
            future_high=future_high,
            future_low=future_low,
            future_close=future_close,
            future_volume=future_volume,
            future_oi=future_oi,
            timestamps=timestamps,
            timeframes=timeframes_data,
            metadata={
                'data_points': len(underlying_data),
                'timeframes': list(self.timeframes.keys()),
                'date_range': (timestamps[0], timestamps[-1])
            }
        )

    async def _create_timeframe_data(self, data: pd.DataFrame, timeframe: str, params: Dict[str, List[int]]) -> Dict[str, np.ndarray]:
        """Create timeframe-specific data for multi-timeframe analysis"""
        
        # Convert timestamp to datetime if needed
        if not isinstance(data['timestamp'].iloc[0], pd.Timestamp):
            data = data.copy()
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Resample data based on timeframe
        if timeframe == 'daily':
            resampled = data.set_index('timestamp').resample('D').agg({
                'spot': 'last',
                'future_open': 'first',
                'future_high': 'max',
                'future_low': 'min',
                'future_close': 'last',
                'future_volume': 'sum',
                'future_oi': 'last'
            }).dropna()
        elif timeframe == 'weekly':
            resampled = data.set_index('timestamp').resample('W').agg({
                'spot': 'last',
                'future_open': 'first',
                'future_high': 'max',
                'future_low': 'min',
                'future_close': 'last',
                'future_volume': 'sum',
                'future_oi': 'last'
            }).dropna()
        elif timeframe == 'monthly':
            resampled = data.set_index('timestamp').resample('M').agg({
                'spot': 'last',
                'future_open': 'first',
                'future_high': 'max',
                'future_low': 'min',
                'future_close': 'last',
                'future_volume': 'sum',
                'future_oi': 'last'
            }).dropna()
        else:
            resampled = data.set_index('timestamp')
        
        return {
            'spot': resampled['spot'].values,
            'open': resampled['future_open'].values,
            'high': resampled['future_high'].values,
            'low': resampled['future_low'].values,
            'close': resampled['future_close'].values,
            'volume': resampled['future_volume'].values,
            'oi': resampled['future_oi'].values,
            'timestamps': resampled.index.values,
            'atr_periods': np.array(params['atr_periods']),
            'ema_periods': np.array(params['ema_periods'])
        }

    def _find_atm_strike(self, ce_data: pd.DataFrame, pe_data: pd.DataFrame, spot_price: float) -> float:
        """Find ATM or closest available strike for straddle construction"""
        
        # Get available strikes from both CE and PE
        ce_strikes = set(ce_data['strike'].values)
        pe_strikes = set(pe_data['strike'].values)
        common_strikes = ce_strikes.intersection(pe_strikes)
        
        if not common_strikes:
            # Use closest available strikes if no common strikes
            all_strikes = list(ce_strikes.union(pe_strikes))
        else:
            all_strikes = list(common_strikes)
        
        # Find closest to spot price
        closest_strike = min(all_strikes, key=lambda x: abs(x - spot_price))
        return closest_strike

    def _validate_data_quality(self, straddle_data: StraddlePriceData, underlying_data: UnderlyingPriceData) -> float:
        """Validate dual-asset data quality"""
        
        quality_checks = []
        
        # Check straddle data quality
        straddle_quality = 1.0
        if len(straddle_data.straddle_close) > 0:
            # Check for reasonable price ranges
            if np.any(straddle_data.straddle_close <= 0):
                straddle_quality *= 0.8
            
            # Check for extreme values
            straddle_std = np.std(straddle_data.straddle_close)
            straddle_mean = np.mean(straddle_data.straddle_close)
            if straddle_std > straddle_mean * 2:  # High volatility check
                straddle_quality *= 0.9
        
        quality_checks.append(straddle_quality)
        
        # Check underlying data quality
        underlying_quality = 1.0
        if len(underlying_data.future_close) > 0:
            # Check for reasonable price ranges
            if np.any(underlying_data.future_close <= 0):
                underlying_quality *= 0.8
            
            # Check volume consistency
            if np.any(underlying_data.future_volume < 0):
                underlying_quality *= 0.9
        
        quality_checks.append(underlying_quality)
        
        # Check timeframe data quality
        tf_quality = 1.0
        for tf_name, tf_data in underlying_data.timeframes.items():
            if len(tf_data['close']) == 0:
                tf_quality *= 0.7
        
        quality_checks.append(tf_quality)
        
        return np.mean(quality_checks)

    def _calculate_zone_coverage(self, data: pd.DataFrame) -> Dict[str, int]:
        """Calculate coverage across production zones"""
        return data['zone_name'].value_counts().to_dict()

    def _calculate_dte_coverage(self, data: pd.DataFrame) -> Dict[int, int]:
        """Calculate coverage across DTE values"""
        return data['dte'].value_counts().to_dict()