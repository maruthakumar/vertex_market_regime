"""
Market Regime Detector - Detection Logic

This module contains the logic for detecting market regime components
based on market data and indicators.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Detector for market regime components
    
    This detector analyzes market data and indicators to detect:
    - Volatility levels
    - Trend direction
    - Market structure
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the regime detector
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Detection parameters
        self.lookback_periods = config.get('lookback_periods', {
            'short': 20,
            'medium': 50,
            'long': 100
        })
        
        self.smoothing_factor = config.get('smoothing_factor', 0.2)
        
    def detect(self, 
              market_data: pd.DataFrame,
              indicator_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect regime components from market data and indicators
        
        Args:
            market_data: DataFrame with market data
            indicator_results: Results from indicator calculations
            
        Returns:
            Dictionary with detection results
        """
        try:
            detection_result = {
                'volatility': self._detect_volatility(market_data, indicator_results),
                'trend': self._detect_trend(market_data, indicator_results),
                'structure': self._detect_structure(market_data, indicator_results),
                'indicator_agreement': self._calculate_indicator_agreement(indicator_results),
                'timestamp': datetime.now()
            }
            
            return detection_result
            
        except Exception as e:
            self.logger.error(f"Error in regime detection: {e}")
            raise
    
    def _detect_volatility(self, 
                          data: pd.DataFrame,
                          indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Detect volatility characteristics"""
        volatility = {
            'value': 0.0,
            'confidence': 0.0,
            'probabilities': {}
        }
        
        # Use ATR indicator if available
        if indicators.get('atr'):
            atr_value = indicators['atr'].get('value', 0)
            volatility['value'] = atr_value / data['close'].iloc[-1]  # Normalize by price
        else:
            # Fallback to standard deviation
            returns = data['close'].pct_change().dropna()
            volatility['value'] = returns.std()
        
        # Calculate probabilities for each level
        vol_value = volatility['value']
        
        # Simple probability calculation - can be enhanced
        if vol_value < 0.01:
            volatility['probabilities'] = {'LOW': 0.8, 'MEDIUM': 0.2, 'HIGH': 0.0}
        elif vol_value < 0.02:
            volatility['probabilities'] = {'LOW': 0.2, 'MEDIUM': 0.6, 'HIGH': 0.2}
        else:
            volatility['probabilities'] = {'LOW': 0.0, 'MEDIUM': 0.3, 'HIGH': 0.7}
        
        # Calculate confidence based on indicator availability
        volatility['confidence'] = 0.8 if indicators.get('atr') else 0.6
        
        return volatility
    
    def _detect_trend(self,
                     data: pd.DataFrame,
                     indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Detect trend characteristics"""
        trend = {
            'value': 0.0,
            'confidence': 0.0,
            'probabilities': {}
        }
        
        # Calculate trend using multiple timeframes
        short_ma = data['close'].rolling(self.lookback_periods['short']).mean()
        long_ma = data['close'].rolling(self.lookback_periods['long']).mean()
        
        # Trend strength
        if len(data) >= self.lookback_periods['long']:
            trend_value = (short_ma.iloc[-1] - long_ma.iloc[-1]) / long_ma.iloc[-1]
            trend['value'] = trend_value
        
        # Use OI/PA indicator if available
        if indicators.get('oi_pa'):
            oi_trend = indicators['oi_pa'].get('trend', 0)
            # Combine price trend with OI trend
            trend['value'] = 0.7 * trend['value'] + 0.3 * oi_trend
        
        # Calculate probabilities
        if trend['value'] < -0.005:
            trend['probabilities'] = {'BEARISH': 0.8, 'NEUTRAL': 0.2, 'BULLISH': 0.0}
        elif trend['value'] > 0.005:
            trend['probabilities'] = {'BEARISH': 0.0, 'NEUTRAL': 0.2, 'BULLISH': 0.8}
        else:
            trend['probabilities'] = {'BEARISH': 0.2, 'NEUTRAL': 0.6, 'BULLISH': 0.2}
        
        # Confidence based on trend consistency
        trend['confidence'] = self._calculate_trend_confidence(data)
        
        return trend
    
    def _detect_structure(self,
                         data: pd.DataFrame,
                         indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Detect market structure"""
        structure = {
            'score': 0.5,  # 0 = trending, 1 = ranging
            'confidence': 0.0,
            'probabilities': {}
        }
        
        # Calculate ADX for trend strength
        adx = self._calculate_adx(data)
        
        # Use straddle indicator if available
        if indicators.get('straddle'):
            straddle_signal = indicators['straddle'].get('structure', 0.5)
            structure['score'] = 0.6 * (1 - adx/100) + 0.4 * straddle_signal
        else:
            structure['score'] = 1 - adx/100  # Higher ADX = more trending
        
        # Calculate probabilities
        if structure['score'] > 0.7:
            structure['probabilities'] = {'RANGING': 0.8, 'TRENDING': 0.2}
        elif structure['score'] < 0.3:
            structure['probabilities'] = {'RANGING': 0.2, 'TRENDING': 0.8}
        else:
            structure['probabilities'] = {'RANGING': 0.5, 'TRENDING': 0.5}
        
        structure['confidence'] = 0.7
        
        return structure
    
    def _calculate_indicator_agreement(self, indicators: Dict[str, Any]) -> float:
        """Calculate agreement level among indicators"""
        if not indicators:
            return 0.5
        
        signals = []
        
        # Collect signals from each indicator
        for name, indicator in indicators.items():
            if indicator and isinstance(indicator, dict):
                signal = indicator.get('signal', 0)
                if signal != 0:
                    signals.append(signal)
        
        if not signals:
            return 0.5
        
        # Calculate agreement as inverse of standard deviation
        signal_std = np.std(signals)
        agreement = 1.0 - min(signal_std, 1.0)
        
        return agreement
    
    def _calculate_trend_confidence(self, data: pd.DataFrame) -> float:
        """Calculate confidence in trend detection"""
        if len(data) < self.lookback_periods['long']:
            return 0.5
        
        # Check trend consistency
        returns = data['close'].pct_change().dropna()
        positive_days = (returns > 0).sum()
        total_days = len(returns)
        
        # More consistent direction = higher confidence
        consistency = abs(positive_days / total_days - 0.5) * 2
        
        return min(0.5 + consistency * 0.5, 0.9)
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average Directional Index (simplified)"""
        if len(data) < period * 2:
            return 25.0  # Default neutral value
        
        # Simplified ADX calculation
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate +DM and -DM
        up_move = high.diff()
        down_move = -low.diff()
        
        pos_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        neg_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        
        # Calculate ATR
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Calculate +DI and -DI
        pos_di = 100 * (pos_dm.rolling(window=period).mean() / atr)
        neg_di = 100 * (neg_dm.rolling(window=period).mean() / atr)
        
        # Calculate DX and ADX
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
        adx = dx.rolling(window=period).mean()
        
        return adx.iloc[-1] if not np.isnan(adx.iloc[-1]) else 25.0