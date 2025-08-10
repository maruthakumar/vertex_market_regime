"""
Multi-Timeframe Analyzer
========================

Analyzes market regime across multiple timeframes and fuses the signals
for comprehensive regime detection.

Author: Market Regime Refactoring Team
Date: 2025-07-08
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


class MultiTimeframeAnalyzer:
    """
    Analyzes market regime across multiple timeframes
    
    This is one of the 9 active components in the enhanced market regime system.
    Base weight: 0.15 (15% of total regime signal)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the multi-timeframe analyzer"""
        self.config = config or {}
        
        # Timeframe configuration from Excel MultiTimeframeConfig
        self.timeframes = {
            '1min': {'weight': 0.30, 'window': 30, 'bars': 20},
            '5min': {'weight': 0.35, 'window': 12, 'bars': 10},
            '15min': {'weight': 0.20, 'window': 8, 'bars': 8},
            '30min': {'weight': 0.10, 'window': 6, 'bars': 6},
            '60min': {'weight': 0.05, 'window': 4, 'bars': 4}
        }
        
        # Override with config if provided
        if 'timeframes' in self.config:
            self.timeframes.update(self.config['timeframes'])
        
        # Consensus requirements
        self.min_consensus = self.config.get('min_consensus', 0.6)
        self.strong_consensus = self.config.get('strong_consensus', 0.8)
        
        # Data storage for each timeframe
        self.timeframe_data = {tf: deque(maxlen=info['window'] * 60) 
                              for tf, info in self.timeframes.items()}
        
        # Signal history
        self.signal_history = {tf: deque(maxlen=100) for tf in self.timeframes}
        self.fusion_history = deque(maxlen=100)
        
        logger.info(f"MultiTimeframeAnalyzer initialized with {len(self.timeframes)} timeframes")
    
    def analyze(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Main analysis method for multi-timeframe regime detection
        
        Args:
            market_data: DataFrame with 1-minute data
                - datetime_
                - close or underlying_close
                - high, low, open
                - volume
                
        Returns:
            Dict with multi-timeframe analysis results
        """
        try:
            # Prepare data for each timeframe
            timeframe_signals = {}
            
            for timeframe, tf_config in self.timeframes.items():
                # Resample data to timeframe
                tf_data = self._resample_to_timeframe(market_data, timeframe)
                
                if len(tf_data) >= tf_config['bars']:
                    # Calculate regime signal for this timeframe
                    signal = self._calculate_timeframe_signal(tf_data, timeframe)
                    timeframe_signals[timeframe] = signal
                    
                    # Store in history
                    self.signal_history[timeframe].append({
                        'timestamp': datetime.now(),
                        'signal': signal
                    })
            
            # Fuse signals across timeframes
            fusion_result = self._fuse_timeframe_signals(timeframe_signals)
            
            # Prepare results
            results = {
                'multi_timeframe_score': fusion_result['combined_signal'],
                'timeframe_signals': timeframe_signals,
                'consensus_level': fusion_result['consensus'],
                'dominant_timeframe': fusion_result['dominant_tf'],
                'alignment_score': fusion_result['alignment'],
                'trend_consistency': self._calculate_trend_consistency(timeframe_signals),
                'timeframe_divergence': self._calculate_timeframe_divergence(timeframe_signals),
                'regime_stability': self._calculate_regime_stability(),
                'tf_1min': timeframe_signals.get('1min', 0.0),
                'tf_5min': timeframe_signals.get('5min', 0.0),
                'tf_15min': timeframe_signals.get('15min', 0.0),
                'timestamp': datetime.now()
            }
            
            # Store fusion result
            self.fusion_history.append(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in multi-timeframe analysis: {e}")
            return self._get_default_results()
    
    def _resample_to_timeframe(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resample 1-minute data to specified timeframe
        
        Args:
            data: 1-minute market data
            timeframe: Target timeframe ('5min', '15min', etc.)
            
        Returns:
            Resampled DataFrame
        """
        # Extract timeframe value
        tf_minutes = int(timeframe.replace('min', ''))
        
        if tf_minutes == 1:
            return data
        
        # Set datetime as index for resampling
        df = data.copy()
        if 'datetime_' in df.columns:
            df.set_index('datetime_', inplace=True)
        
        # Resample rules
        ohlc_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Add underlying_close if present
        if 'underlying_close' in df.columns:
            ohlc_dict['underlying_close'] = 'last'
        
        # Perform resampling
        resampled = df.resample(f'{tf_minutes}T').agg(ohlc_dict)
        
        return resampled.dropna()
    
    def _calculate_timeframe_signal(self, data: pd.DataFrame, timeframe: str) -> float:
        """
        Calculate regime signal for a specific timeframe
        
        Args:
            data: Resampled data for the timeframe
            timeframe: Timeframe identifier
            
        Returns:
            float: Signal between -1 and 1
        """
        try:
            # Price trend component
            if 'close' in data.columns:
                prices = data['close'].values
            else:
                prices = data['underlying_close'].values
            
            if len(prices) < 2:
                return 0.0
            
            # Simple trend calculation
            returns = np.diff(prices) / prices[:-1]
            avg_return = np.mean(returns)
            trend_signal = np.tanh(avg_return * 100)  # Scale and bound
            
            # Volatility component
            volatility = np.std(returns) * np.sqrt(252 * 375 / int(timeframe.replace('min', '')))
            vol_signal = (volatility - 0.15) / 0.15  # Normalize around 15% vol
            vol_signal = np.clip(vol_signal, -1, 1)
            
            # Volume trend (if available)
            if 'volume' in data.columns and len(data) > 1:
                volume_trend = (data['volume'].iloc[-1] - data['volume'].mean()) / data['volume'].std()
                volume_signal = np.tanh(volume_trend)
            else:
                volume_signal = 0.0
            
            # Combine signals
            combined_signal = (
                trend_signal * 0.5 +
                vol_signal * 0.3 +
                volume_signal * 0.2
            )
            
            return np.clip(combined_signal, -1.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating {timeframe} signal: {e}")
            return 0.0
    
    def _fuse_timeframe_signals(self, signals: Dict[str, float]) -> Dict[str, Any]:
        """
        Fuse signals from multiple timeframes
        
        Args:
            signals: Dict of timeframe -> signal
            
        Returns:
            Dict with fusion results
        """
        if not signals:
            return {
                'combined_signal': 0.0,
                'consensus': 0.0,
                'dominant_tf': None,
                'alignment': 0.0
            }
        
        # Calculate weighted average
        weighted_sum = 0.0
        total_weight = 0.0
        
        for tf, signal in signals.items():
            weight = self.timeframes[tf]['weight']
            weighted_sum += signal * weight
            total_weight += weight
        
        combined_signal = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Calculate consensus (how many timeframes agree on direction)
        directions = [1 if s > 0 else -1 if s < 0 else 0 for s in signals.values()]
        if directions:
            most_common_direction = max(set(directions), key=directions.count)
            consensus = directions.count(most_common_direction) / len(directions)
        else:
            consensus = 0.0
        
        # Find dominant timeframe (strongest signal)
        if signals:
            dominant_tf = max(signals.items(), key=lambda x: abs(x[1]))[0]
        else:
            dominant_tf = None
        
        # Calculate alignment (standard deviation of signals)
        if len(signals) > 1:
            signal_values = list(signals.values())
            alignment = 1.0 - np.std(signal_values)  # Higher alignment = lower std
            alignment = np.clip(alignment, 0.0, 1.0)
        else:
            alignment = 1.0
        
        return {
            'combined_signal': combined_signal,
            'consensus': consensus,
            'dominant_tf': dominant_tf,
            'alignment': alignment
        }
    
    def _calculate_trend_consistency(self, signals: Dict[str, float]) -> float:
        """
        Calculate how consistent the trend is across timeframes
        
        Args:
            signals: Timeframe signals
            
        Returns:
            float: Consistency score (0-1)
        """
        if len(signals) < 2:
            return 1.0
        
        # Check if all signals have the same sign
        signs = [np.sign(s) for s in signals.values() if s != 0]
        
        if not signs:
            return 0.5
        
        # All same sign = high consistency
        if all(s == signs[0] for s in signs):
            # Also consider magnitude consistency
            magnitudes = [abs(s) for s in signals.values()]
            mag_consistency = 1.0 - (np.std(magnitudes) / np.mean(magnitudes) if np.mean(magnitudes) > 0 else 0)
            return 0.7 + 0.3 * mag_consistency
        else:
            # Mixed signs = lower consistency
            same_sign_ratio = max(signs.count(1), signs.count(-1)) / len(signs)
            return same_sign_ratio * 0.5
    
    def _calculate_timeframe_divergence(self, signals: Dict[str, float]) -> float:
        """
        Calculate divergence between timeframes
        
        Args:
            signals: Timeframe signals
            
        Returns:
            float: Divergence score (0-1, higher = more divergence)
        """
        if len(signals) < 2:
            return 0.0
        
        # Calculate pairwise differences
        signal_list = list(signals.values())
        divergences = []
        
        for i in range(len(signal_list)):
            for j in range(i + 1, len(signal_list)):
                divergences.append(abs(signal_list[i] - signal_list[j]))
        
        # Average divergence
        avg_divergence = np.mean(divergences) if divergences else 0.0
        
        # Normalize (max divergence is 2: from -1 to +1)
        return avg_divergence / 2.0
    
    def _calculate_regime_stability(self) -> float:
        """
        Calculate regime stability based on historical signals
        
        Returns:
            float: Stability score (0-1)
        """
        if len(self.fusion_history) < 10:
            return 0.5
        
        # Get recent combined signals
        recent_signals = [h['multi_timeframe_score'] for h in list(self.fusion_history)[-20:]]
        
        # Calculate stability as inverse of volatility
        signal_volatility = np.std(recent_signals)
        
        # Normalize (typical volatility range 0-0.5)
        stability = 1.0 - min(signal_volatility * 2, 1.0)
        
        return stability
    
    def get_timeframe_alignment(self) -> Dict[str, Any]:
        """Get current alignment status across timeframes"""
        if not self.signal_history['1min']:
            return {'aligned': False, 'score': 0.0}
        
        current_signals = {}
        for tf in self.timeframes:
            if self.signal_history[tf]:
                current_signals[tf] = self.signal_history[tf][-1]['signal']
        
        if len(current_signals) < 2:
            return {'aligned': False, 'score': 0.0}
        
        # Check alignment
        signs = [np.sign(s) for s in current_signals.values()]
        aligned = all(s == signs[0] for s in signs)
        
        # Calculate alignment score
        if aligned:
            # Check magnitude alignment
            magnitudes = [abs(s) for s in current_signals.values()]
            score = 1.0 - (np.std(magnitudes) / (np.mean(magnitudes) + 0.001))
        else:
            score = max(signs.count(1), signs.count(-1)) / len(signs) * 0.5
        
        return {
            'aligned': aligned,
            'score': score,
            'direction': 'bullish' if signs[0] > 0 else 'bearish' if signs[0] < 0 else 'neutral'
        }
    
    def _get_default_results(self) -> Dict[str, Any]:
        """Get default results when analysis fails"""
        return {
            'multi_timeframe_score': 0.0,
            'timeframe_signals': {},
            'consensus_level': 0.0,
            'dominant_timeframe': None,
            'alignment_score': 0.0,
            'trend_consistency': 0.5,
            'timeframe_divergence': 0.0,
            'regime_stability': 0.5,
            'tf_1min': 0.0,
            'tf_5min': 0.0,
            'tf_15min': 0.0,
            'timestamp': datetime.now()
        }