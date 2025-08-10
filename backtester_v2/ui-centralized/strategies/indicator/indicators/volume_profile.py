"""
Volume Profile Analysis for IND Strategy Agent
Advanced volume analysis including POC, VAH, VAL, and volume distribution
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class VolumeProfile:
    """
    Volume Profile Analysis Implementation
    
    Provides:
    - Point of Control (POC) - price with highest volume
    - Value Area High (VAH) and Low (VAL) - 70% of volume range
    - Volume Distribution analysis
    - Volume Node identification
    """
    
    def __init__(self):
        self.profile_cache = {}
        
    async def calculate(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate volume profile indicators"""
        try:
            logger.info("🔊 Calculating Volume Profile indicators")
            
            # Make a copy to avoid modifying original data
            result_df = df.copy()
            
            # Extract parameters
            lookback_period = params.get('lookback_period', 50)
            price_buckets = params.get('price_buckets', 20)
            value_area_percent = params.get('value_area_percent', 0.70)
            
            # Initialize volume profile columns
            result_df['vp_poc'] = np.nan
            result_df['vp_vah'] = np.nan
            result_df['vp_val'] = np.nan
            result_df['vp_volume_at_price'] = 0
            result_df['vp_profile_strength'] = 0.0
            result_df['vp_volume_imbalance'] = 0.0
            
            # Calculate volume profile for each period
            for i in range(lookback_period, len(result_df)):
                try:
                    # Get lookback data
                    lookback_data = result_df.iloc[i-lookback_period:i]
                    
                    # Calculate volume profile
                    poc, vah, val, profile_strength, volume_imbalance = self._calculate_volume_profile_window(
                        lookback_data, price_buckets, value_area_percent
                    )
                    
                    # Update result data
                    result_df.iloc[i, result_df.columns.get_loc('vp_poc')] = poc
                    result_df.iloc[i, result_df.columns.get_loc('vp_vah')] = vah
                    result_df.iloc[i, result_df.columns.get_loc('vp_val')] = val
                    result_df.iloc[i, result_df.columns.get_loc('vp_profile_strength')] = profile_strength
                    result_df.iloc[i, result_df.columns.get_loc('vp_volume_imbalance')] = volume_imbalance
                    
                    # Volume at current price level
                    current_price = result_df.iloc[i]['close_price']
                    volume_at_price = self._get_volume_at_price(lookback_data, current_price, price_buckets)
                    result_df.iloc[i, result_df.columns.get_loc('vp_volume_at_price')] = volume_at_price
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error calculating volume profile for row {i}: {e}")
                    continue
            
            logger.info("✅ Volume Profile indicators calculated successfully")
            return result_df
            
        except Exception as e:
            logger.error(f"❌ Error calculating Volume Profile: {e}")
            return df
    
    def _calculate_volume_profile_window(
        self, 
        data: pd.DataFrame, 
        price_buckets: int, 
        value_area_percent: float
    ) -> Tuple[float, float, float, float, float]:
        """Calculate volume profile for a specific window"""
        try:
            if len(data) == 0:
                return np.nan, np.nan, np.nan, 0.0, 0.0
            
            # Get price range
            min_price = data['low_price'].min()
            max_price = data['high_price'].max()
            price_range = max_price - min_price
            
            if price_range == 0:
                return data['close_price'].iloc[-1], np.nan, np.nan, 0.0, 0.0
            
            # Create price buckets
            bucket_size = price_range / price_buckets
            price_levels = np.linspace(min_price, max_price, price_buckets + 1)
            
            # Calculate volume at each price level
            volume_at_level = np.zeros(price_buckets)
            
            for i, row in data.iterrows():
                # Approximate volume distribution across OHLC range
                low, high, volume = row['low_price'], row['high_price'], row['volume']
                
                # Find bucket indices for this candle's range
                start_bucket = max(0, int((low - min_price) / bucket_size))
                end_bucket = min(price_buckets - 1, int((high - min_price) / bucket_size))
                
                # Distribute volume across buckets in the range
                buckets_in_range = max(1, end_bucket - start_bucket + 1)
                volume_per_bucket = volume / buckets_in_range
                
                for bucket_idx in range(start_bucket, end_bucket + 1):
                    if 0 <= bucket_idx < price_buckets:
                        volume_at_level[bucket_idx] += volume_per_bucket
            
            # Find Point of Control (POC) - highest volume price level
            poc_idx = np.argmax(volume_at_level)
            poc = price_levels[poc_idx] + bucket_size / 2  # Middle of bucket
            
            # Calculate Value Area (VAH/VAL)
            total_volume = np.sum(volume_at_level)
            target_volume = total_volume * value_area_percent
            
            # Start from POC and expand outward until we reach target volume
            vah_idx = val_idx = poc_idx
            current_volume = volume_at_level[poc_idx]
            
            while current_volume < target_volume and (vah_idx < price_buckets - 1 or val_idx > 0):
                # Check which direction has more volume
                upper_volume = volume_at_level[vah_idx + 1] if vah_idx < price_buckets - 1 else 0
                lower_volume = volume_at_level[val_idx - 1] if val_idx > 0 else 0
                
                if upper_volume >= lower_volume and vah_idx < price_buckets - 1:
                    vah_idx += 1
                    current_volume += volume_at_level[vah_idx]
                elif val_idx > 0:
                    val_idx -= 1
                    current_volume += volume_at_level[val_idx]
                else:
                    break
            
            vah = price_levels[vah_idx + 1]  # Upper bound of bucket
            val = price_levels[val_idx]      # Lower bound of bucket
            
            # Calculate profile strength (concentration of volume)
            max_volume = np.max(volume_at_level)
            avg_volume = np.mean(volume_at_level)
            profile_strength = max_volume / avg_volume if avg_volume > 0 else 0
            
            # Calculate volume imbalance (skewness of distribution)
            upper_half_volume = np.sum(volume_at_level[poc_idx:])
            lower_half_volume = np.sum(volume_at_level[:poc_idx + 1])
            total_half_volume = upper_half_volume + lower_half_volume
            
            if total_half_volume > 0:
                volume_imbalance = (upper_half_volume - lower_half_volume) / total_half_volume
            else:
                volume_imbalance = 0.0
            
            return poc, vah, val, profile_strength, volume_imbalance
            
        except Exception as e:
            logger.error(f"❌ Error calculating volume profile window: {e}")
            return np.nan, np.nan, np.nan, 0.0, 0.0
    
    def _get_volume_at_price(
        self, 
        data: pd.DataFrame, 
        target_price: float, 
        price_buckets: int
    ) -> float:
        """Get approximate volume at a specific price level"""
        try:
            if len(data) == 0:
                return 0.0
            
            # Get price range
            min_price = data['low_price'].min()
            max_price = data['high_price'].max()
            price_range = max_price - min_price
            
            if price_range == 0:
                return data['volume'].sum()
            
            bucket_size = price_range / price_buckets
            target_bucket = int((target_price - min_price) / bucket_size)
            target_bucket = max(0, min(price_buckets - 1, target_bucket))
            
            # Calculate volume in target bucket
            bucket_min = min_price + target_bucket * bucket_size
            bucket_max = bucket_min + bucket_size
            
            total_volume = 0.0
            
            for i, row in data.iterrows():
                low, high, volume = row['low_price'], row['high_price'], row['volume']
                
                # Check if candle range overlaps with target bucket
                overlap_start = max(low, bucket_min)
                overlap_end = min(high, bucket_max)
                
                if overlap_start < overlap_end:
                    # Calculate proportion of candle's range in target bucket
                    candle_range = high - low
                    overlap_range = overlap_end - overlap_start
                    
                    if candle_range > 0:
                        proportion = overlap_range / candle_range
                        total_volume += volume * proportion
                    else:
                        # Single price point - assign full volume if in bucket
                        if bucket_min <= low <= bucket_max:
                            total_volume += volume
            
            return total_volume
            
        except Exception as e:
            logger.error(f"❌ Error getting volume at price: {e}")
            return 0.0
    
    def get_volume_profile_signals(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate trading signals based on volume profile analysis"""
        try:
            signals = []
            
            for idx, row in df.iterrows():
                if pd.isna(row.get('vp_poc')):
                    continue
                
                signal_data = {
                    'timestamp': idx,
                    'price': row.get('close_price', 0),
                    'signals': []
                }
                
                # POC support/resistance signals
                current_price = row.get('close_price', 0)
                poc = row.get('vp_poc', 0)
                vah = row.get('vp_vah', 0)
                val = row.get('vp_val', 0)
                
                # Price near POC
                poc_tolerance = abs(current_price * 0.005)  # 0.5% tolerance
                if abs(current_price - poc) < poc_tolerance:
                    signal_data['signals'].append({
                        'type': 'POC_INTERACTION',
                        'strength': row.get('vp_profile_strength', 0),
                        'description': f'Price interacting with POC at {poc:.2f}'
                    })
                
                # Value Area signals
                if not pd.isna(vah) and not pd.isna(val):
                    if current_price > vah:
                        signal_data['signals'].append({
                            'type': 'ABOVE_VALUE_AREA',
                            'strength': (current_price - vah) / vah,
                            'description': f'Price above Value Area High ({vah:.2f})'
                        })
                    elif current_price < val:
                        signal_data['signals'].append({
                            'type': 'BELOW_VALUE_AREA',
                            'strength': (val - current_price) / val,
                            'description': f'Price below Value Area Low ({val:.2f})'
                        })
                    else:
                        signal_data['signals'].append({
                            'type': 'INSIDE_VALUE_AREA',
                            'strength': 0.5,
                            'description': f'Price inside Value Area ({val:.2f} - {vah:.2f})'
                        })
                
                # Volume imbalance signals
                volume_imbalance = row.get('vp_volume_imbalance', 0)
                if abs(volume_imbalance) > 0.3:  # Significant imbalance
                    signal_data['signals'].append({
                        'type': 'VOLUME_IMBALANCE',
                        'direction': 'BULLISH' if volume_imbalance > 0 else 'BEARISH',
                        'strength': abs(volume_imbalance),
                        'description': f'Volume imbalance: {volume_imbalance:.2f}'
                    })
                
                # Add signal if any conditions met
                if signal_data['signals']:
                    signals.append(signal_data)
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Error generating volume profile signals: {e}")
            return []