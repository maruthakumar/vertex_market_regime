"""
Order Flow Analysis for IND Strategy Agent
Advanced order flow analysis including bid/ask imbalance, delta analysis, and flow patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class OrderFlow:
    """
    Order Flow Analysis Implementation
    
    Provides:
    - Delta analysis (buying vs selling pressure)
    - Cumulative Delta tracking
    - Volume profile integration
    - Absorption and exhaustion detection
    - Institutional footprint analysis
    """
    
    def __init__(self):
        self.flow_cache = {}
        self.delta_history = []
        
    async def analyze(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Analyze order flow patterns
        
        Args:
            df: DataFrame with OHLCV data
            params: Order flow analysis parameters
            
        Returns:
            DataFrame with order flow indicators
        """
        try:
            logger.info("=� Analyzing Order Flow patterns")
            
            # Make a copy to avoid modifying original data
            result_df = df.copy()
            
            # Extract parameters
            delta_period = params.get('delta_period', 14)
            flow_threshold = params.get('flow_threshold', 0.6)
            volume_factor = params.get('volume_factor', 1.5)
            
            # Calculate basic delta (simplified without tick data)
            result_df = self._calculate_delta_approximation(result_df, params)
            
            # Calculate cumulative delta
            result_df['of_cumulative_delta'] = result_df['of_delta'].fillna(0).cumsum()
            
            # Delta divergence analysis
            result_df = self._calculate_delta_divergence(result_df, params)
            
            # Volume flow analysis
            result_df = self._calculate_volume_flow(result_df, params)
            
            # Absorption patterns
            result_df = self._detect_absorption_patterns(result_df, params)
            
            # Institutional footprint
            result_df = self._calculate_institutional_footprint(result_df, params)
            
            logger.info(" Order Flow analysis completed successfully")
            return result_df
            
        except Exception as e:
            logger.error(f"L Error analyzing Order Flow: {e}")
            return df
    
    def _calculate_delta_approximation(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculate delta approximation using OHLCV data
        Note: True delta requires tick-by-tick bid/ask data
        """
        try:
            # Approximate delta using price action and volume
            # Green candles (close > open) assume more buying
            # Red candles (close < open) assume more selling
            
            # Basic delta approximation
            price_change = df['close_price'] - df['open_price']
            range_factor = abs(price_change) / (df['high_price'] - df['low_price']).replace(0, 1)
            
            # Estimate buying vs selling volume
            buying_volume = np.where(
                price_change > 0,
                df['volume'] * (0.5 + range_factor * 0.3),  # More buying if strong green candle
                df['volume'] * (0.5 - range_factor * 0.3)   # Less buying if strong red candle
            )
            
            selling_volume = df['volume'] - buying_volume
            
            # Delta = Buying Volume - Selling Volume
            df['of_delta'] = buying_volume - selling_volume
            df['of_buying_volume'] = buying_volume
            df['of_selling_volume'] = selling_volume
            
            # Delta percentage
            total_volume = df['volume'].replace(0, 1)
            df['of_delta_pct'] = df['of_delta'] / total_volume
            
            return df
            
        except Exception as e:
            logger.error(f"L Error calculating delta approximation: {e}")
            return df
    
    def _calculate_delta_divergence(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate delta divergence with price"""
        try:
            divergence_period = params.get('divergence_period', 14)
            
            # Price momentum
            price_momentum = df['close_price'].pct_change(periods=divergence_period)
            
            # Delta momentum
            delta_momentum = df['of_delta'].pct_change(periods=divergence_period)
            
            # Bullish divergence: price down, delta up (hidden buying)
            bullish_div = (price_momentum < -0.02) & (delta_momentum > 0.1)
            
            # Bearish divergence: price up, delta down (hidden selling)
            bearish_div = (price_momentum > 0.02) & (delta_momentum < -0.1)
            
            df['of_bullish_divergence'] = bullish_div
            df['of_bearish_divergence'] = bearish_div
            
            # Divergence strength
            df['of_divergence_strength'] = np.where(
                bullish_div | bearish_div,
                abs(price_momentum) + abs(delta_momentum),
                0
            )
            
            return df
            
        except Exception as e:
            logger.error(f"L Error calculating delta divergence: {e}")
            return df
    
    def _calculate_volume_flow(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate volume flow patterns"""
        try:
            flow_period = params.get('flow_period', 10)
            
            # Volume momentum
            volume_ma = df['volume'].rolling(window=flow_period, min_periods=1).mean()
            df['of_volume_ratio'] = df['volume'] / volume_ma
            
            # Flow strength based on volume and price movement
            price_change_pct = df['close_price'].pct_change()
            volume_change_pct = df['volume'].pct_change()
            
            # Strong flow: high volume with directional price movement
            df['of_flow_strength'] = abs(price_change_pct) * df['of_volume_ratio']
            
            # Flow direction
            df['of_flow_direction'] = np.where(
                df['of_delta'] > 0, 'BUYING', 'SELLING'
            )
            
            return df
            
        except Exception as e:
            logger.error(f"L Error calculating volume flow: {e}")
            return df
    
    def _detect_absorption_patterns(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Detect volume absorption patterns"""
        try:
            absorption_threshold = params.get('absorption_threshold', 2.0)
            
            # High volume with small price movement indicates absorption
            volume_spike = df['of_volume_ratio'] > absorption_threshold
            price_change_pct = abs(df['close_price'].pct_change())
            small_price_move = price_change_pct < df['close_price'].pct_change().rolling(20).std()
            
            absorption = volume_spike & small_price_move
            
            df['of_absorption'] = absorption
            df['of_absorption_strength'] = np.where(
                absorption,
                df['of_volume_ratio'] / (price_change_pct + 0.001),  # Higher ratio = stronger absorption
                0
            )
            
            return df
            
        except Exception as e:
            logger.error(f"L Error detecting absorption patterns: {e}")
            return df
    
    def _calculate_institutional_footprint(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate institutional footprint indicators"""
        try:
            # Large volume threshold for institutional activity
            institutional_threshold = params.get('institutional_threshold', 3.0)
            
            # Identify potential institutional activity
            large_volume = df['of_volume_ratio'] > institutional_threshold
            
            # Iceberg pattern: repeated large volumes at similar price levels
            price_tolerance = params.get('price_tolerance', 0.01)  # 1% tolerance
            
            institutional_activity = []
            
            for i in range(len(df)):
                if not large_volume.iloc[i]:
                    institutional_activity.append(False)
                    continue
                
                current_price = df['close_price'].iloc[i]
                
                # Look for similar price levels with large volume in recent history
                lookback = min(20, i)
                recent_data = df.iloc[max(0, i-lookback):i]
                
                similar_price_mask = abs(recent_data['close_price'] - current_price) / current_price < price_tolerance
                similar_large_volumes = (recent_data.loc[similar_price_mask, 'of_volume_ratio'] > institutional_threshold).sum()
                
                # If multiple large volumes at similar prices, likely institutional
                institutional_activity.append(similar_large_volumes >= 2)
            
            df['of_institutional_activity'] = institutional_activity
            
            # Institutional pressure direction
            df['of_institutional_pressure'] = np.where(
                df['of_institutional_activity'],
                np.where(df['of_delta'] > 0, 'BUYING', 'SELLING'),
                'NONE'
            )
            
            return df
            
        except Exception as e:
            logger.error(f"L Error calculating institutional footprint: {e}")
            return df
    
    def get_orderflow_signals(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate trading signals based on order flow analysis"""
        try:
            signals = []
            
            for idx, row in df.iterrows():
                signal_data = {
                    'timestamp': idx,
                    'price': row.get('close_price', 0),
                    'signals': []
                }
                
                # Delta divergence signals
                if row.get('of_bullish_divergence', False):
                    signal_data['signals'].append({
                        'type': 'BULLISH_DELTA_DIVERGENCE',
                        'strength': row.get('of_divergence_strength', 0),
                        'description': 'Price declining but delta increasing (hidden buying)'
                    })
                
                if row.get('of_bearish_divergence', False):
                    signal_data['signals'].append({
                        'type': 'BEARISH_DELTA_DIVERGENCE',
                        'strength': row.get('of_divergence_strength', 0),
                        'description': 'Price rising but delta decreasing (hidden selling)'
                    })
                
                # Absorption signals
                if row.get('of_absorption', False):
                    signal_data['signals'].append({
                        'type': 'VOLUME_ABSORPTION',
                        'strength': row.get('of_absorption_strength', 0),
                        'description': 'High volume with minimal price movement'
                    })
                
                # Institutional activity signals
                if row.get('of_institutional_activity', False):
                    pressure = row.get('of_institutional_pressure', 'NONE')
                    if pressure != 'NONE':
                        signal_data['signals'].append({
                            'type': 'INSTITUTIONAL_ACTIVITY',
                            'direction': pressure,
                            'strength': row.get('of_volume_ratio', 1),
                            'description': f'Institutional {pressure.lower()} detected'
                        })
                
                # Add signal if any conditions met
                if signal_data['signals']:
                    signals.append(signal_data)
            
            return signals
            
        except Exception as e:
            logger.error(f"L Error generating order flow signals: {e}")
            return []