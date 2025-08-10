"""
Market Regime Analyzer - Core Analysis Logic

This module contains the core analysis logic for processing market data
and generating regime insights.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class MarketRegimeAnalyzer:
    """
    Core analyzer for market regime analysis
    
    Responsibilities:
    1. Analyze market structure
    2. Calculate regime metrics
    3. Aggregate signals from multiple sources
    4. Provide regime insights
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the analyzer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize analysis parameters
        self.volatility_thresholds = config.get('volatility_thresholds', {
            'low': 0.01,
            'medium': 0.02,
            'high': 0.03
        })
        
        self.trend_thresholds = config.get('trend_thresholds', {
            'bearish': -0.005,
            'neutral': 0.005,
            'bullish': 0.01
        })
        
        self.timeframes = config.get('timeframes', ['3min', '5min', '15min'])
        
    def analyze_market_structure(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market structure including volatility, trend, and momentum
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Dictionary with structure analysis
        """
        try:
            structure = {
                'volatility': self._analyze_volatility(data),
                'trend': self._analyze_trend(data),
                'momentum': self._analyze_momentum(data),
                'volume': self._analyze_volume(data),
                'market_phase': self._determine_market_phase(data)
            }
            
            return structure
            
        except Exception as e:
            self.logger.error(f"Error analyzing market structure: {e}")
            raise
    
    def _analyze_volatility(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze volatility metrics"""
        volatility = {}
        
        # Calculate returns
        returns = data['close'].pct_change().dropna()
        
        # Standard deviation
        volatility['std'] = returns.std()
        
        # ATR-based volatility
        if 'high' in data.columns and 'low' in data.columns:
            volatility['atr'] = self._calculate_atr(data)
        
        # Volatility classification
        if volatility['std'] < self.volatility_thresholds['low']:
            volatility['level'] = 'LOW'
        elif volatility['std'] < self.volatility_thresholds['medium']:
            volatility['level'] = 'MEDIUM'
        else:
            volatility['level'] = 'HIGH'
        
        # Volatility percentile
        volatility['percentile'] = self._calculate_volatility_percentile(volatility['std'])
        
        return volatility
    
    def _analyze_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trend characteristics"""
        trend = {}
        
        # Calculate moving averages
        ma_short = data['close'].rolling(window=20).mean()
        ma_long = data['close'].rolling(window=50).mean()
        
        # Trend direction
        current_price = data['close'].iloc[-1]
        trend_strength = (current_price - ma_long.iloc[-1]) / ma_long.iloc[-1]
        
        if trend_strength < self.trend_thresholds['bearish']:
            trend['direction'] = 'BEARISH'
        elif trend_strength > self.trend_thresholds['bullish']:
            trend['direction'] = 'BULLISH'
        else:
            trend['direction'] = 'NEUTRAL'
        
        trend['strength'] = abs(trend_strength)
        
        # Trend consistency
        trend['consistency'] = self._calculate_trend_consistency(data)
        
        # MA crossover signals
        trend['ma_signal'] = 'BULLISH' if ma_short.iloc[-1] > ma_long.iloc[-1] else 'BEARISH'
        
        return trend
    
    def _analyze_momentum(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze momentum indicators"""
        momentum = {}
        
        # RSI
        momentum['rsi'] = self._calculate_rsi(data['close'])
        
        # MACD
        macd_result = self._calculate_macd(data['close'])
        momentum['macd'] = macd_result['macd']
        momentum['macd_signal'] = macd_result['signal']
        momentum['macd_histogram'] = macd_result['histogram']
        
        # Momentum classification
        if momentum['rsi'] > 70:
            momentum['state'] = 'OVERBOUGHT'
        elif momentum['rsi'] < 30:
            momentum['state'] = 'OVERSOLD'
        else:
            momentum['state'] = 'NEUTRAL'
        
        return momentum
    
    def _analyze_volume(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume patterns"""
        if 'volume' not in data.columns:
            return {'available': False}
        
        volume = {'available': True}
        
        # Volume moving average
        volume['ma'] = data['volume'].rolling(window=20).mean().iloc[-1]
        
        # Current vs average
        current_volume = data['volume'].iloc[-1]
        volume['ratio'] = current_volume / volume['ma'] if volume['ma'] > 0 else 1.0
        
        # Volume trend
        volume['trend'] = 'INCREASING' if volume['ratio'] > 1.2 else 'NORMAL'
        
        return volume
    
    def _determine_market_phase(self, data: pd.DataFrame) -> str:
        """Determine overall market phase"""
        # Simplified implementation
        volatility = self._analyze_volatility(data)
        trend = self._analyze_trend(data)
        
        if volatility['level'] == 'HIGH' and trend['direction'] == 'BEARISH':
            return 'PANIC'
        elif volatility['level'] == 'HIGH' and trend['direction'] == 'BULLISH':
            return 'EUPHORIA'
        elif volatility['level'] == 'LOW' and trend['direction'] == 'NEUTRAL':
            return 'CONSOLIDATION'
        else:
            return 'NORMAL'
    
    def aggregate_signals(self, 
                         signals: Dict[str, Any], 
                         regime: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate signals from multiple indicators based on regime
        
        Args:
            signals: Raw signals from indicators
            regime: Current regime classification
            
        Returns:
            Aggregated signal dictionary
        """
        aggregated = {
            'action': 'HOLD',
            'confidence': 0.0,
            'reasons': [],
            'risk_level': 'MEDIUM'
        }
        
        # Count bullish/bearish signals
        bullish_count = 0
        bearish_count = 0
        total_confidence = 0.0
        
        for indicator, signal in signals.get('indicators', {}).items():
            if signal and isinstance(signal, dict):
                if signal.get('action') == 'BUY':
                    bullish_count += 1
                    total_confidence += signal.get('confidence', 0.5)
                elif signal.get('action') == 'SELL':
                    bearish_count += 1
                    total_confidence += signal.get('confidence', 0.5)
        
        # Determine action based on consensus
        total_signals = bullish_count + bearish_count
        if total_signals > 0:
            if bullish_count > bearish_count * 1.5:
                aggregated['action'] = 'BUY'
                aggregated['confidence'] = total_confidence / total_signals
                aggregated['reasons'].append(f"{bullish_count} bullish signals")
            elif bearish_count > bullish_count * 1.5:
                aggregated['action'] = 'SELL'
                aggregated['confidence'] = total_confidence / total_signals
                aggregated['reasons'].append(f"{bearish_count} bearish signals")
        
        # Adjust based on regime
        aggregated = self._adjust_for_regime(aggregated, regime)
        
        return aggregated
    
    def _adjust_for_regime(self, 
                          signals: Dict[str, Any], 
                          regime: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust signals based on current regime"""
        regime_type = regime.get('regime', '')
        
        # High volatility regimes require higher confidence
        if 'HIGH_VOLATILITY' in regime_type:
            signals['confidence'] *= 0.8
            signals['risk_level'] = 'HIGH'
            signals['reasons'].append("High volatility regime - reduced confidence")
        
        # Trending regimes reinforce directional signals
        if 'BULLISH_TRENDING' in regime_type and signals['action'] == 'BUY':
            signals['confidence'] *= 1.2
            signals['reasons'].append("Bullish trending regime supports buy signal")
        elif 'BEARISH_TRENDING' in regime_type and signals['action'] == 'SELL':
            signals['confidence'] *= 1.2
            signals['reasons'].append("Bearish trending regime supports sell signal")
        
        # Ranging regimes favor mean reversion
        if 'RANGING' in regime_type:
            signals['risk_level'] = 'LOW'
            signals['reasons'].append("Ranging regime - mean reversion favorable")
        
        # Cap confidence at 1.0
        signals['confidence'] = min(signals['confidence'], 1.0)
        
        return signals
    
    # Helper methods
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        return atr
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1]
    
    def _calculate_macd(self, prices: pd.Series, 
                       fast: int = 12, 
                       slow: int = 26, 
                       signal: int = 9) -> Dict[str, float]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line.iloc[-1],
            'signal': signal_line.iloc[-1],
            'histogram': histogram.iloc[-1]
        }
    
    def _calculate_trend_consistency(self, data: pd.DataFrame) -> float:
        """Calculate trend consistency score"""
        # Simple implementation - can be enhanced
        returns = data['close'].pct_change().dropna()
        positive_returns = (returns > 0).sum()
        total_returns = len(returns)
        
        return positive_returns / total_returns if total_returns > 0 else 0.5
    
    def _calculate_volatility_percentile(self, current_vol: float) -> float:
        """Calculate volatility percentile (placeholder)"""
        # This would typically use historical data
        return np.clip(current_vol * 100, 0, 100)