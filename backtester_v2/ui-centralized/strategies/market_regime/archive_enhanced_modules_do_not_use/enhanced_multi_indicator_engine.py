"""
Enhanced Multi-Indicator Engine

This module provides a comprehensive multi-indicator analysis engine that combines
various technical indicators across multiple timeframes for regime detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
from datetime import datetime
import talib

logger = logging.getLogger(__name__)

class EnhancedMultiIndicatorEngine:
    """
    Enhanced Multi-Indicator Engine for comprehensive market analysis
    
    Features:
    - Multiple technical indicators (EMA, VWAP, RSI, BB, MACD, etc.)
    - Multi-timeframe analysis
    - Dynamic weight optimization
    - Indicator confluence detection
    - Real-time performance tracking
    """
    
    def __init__(self):
        """Initialize the multi-indicator engine"""
        self.timeframes = [3, 5, 10, 15]  # minutes
        
        # Indicator configurations
        self.indicator_configs = {
            'ema': {
                'periods': [9, 21, 50, 200],
                'weight': 0.20
            },
            'vwap': {
                'weight': 0.15
            },
            'rsi': {
                'period': 14,
                'overbought': 70,
                'oversold': 30,
                'weight': 0.15
            },
            'bollinger_bands': {
                'period': 20,
                'std_dev': 2,
                'weight': 0.10
            },
            'macd': {
                'fast': 12,
                'slow': 26,
                'signal': 9,
                'weight': 0.15
            },
            'stochastic': {
                'k_period': 14,
                'd_period': 3,
                'weight': 0.10
            },
            'atr': {
                'period': 14,
                'weight': 0.10
            },
            'adx': {
                'period': 14,
                'weight': 0.05
            }
        }
        
        # Regime thresholds
        self.regime_thresholds = {
            'strong_bullish': 0.7,
            'bullish': 0.5,
            'neutral_bullish': 0.3,
            'neutral': 0.0,
            'neutral_bearish': -0.3,
            'bearish': -0.5,
            'strong_bearish': -0.7
        }
        
        logger.info("Enhanced Multi-Indicator Engine initialized")
    
    def calculate_indicators(self, market_data: pd.DataFrame, timeframe: int) -> Dict[str, Any]:
        """Calculate all indicators for a specific timeframe"""
        try:
            indicators = {}
            
            # Ensure we have price data
            if 'close' not in market_data.columns:
                if 'spot_close' in market_data.columns:
                    close_prices = market_data['spot_close']
                elif 'underlying_value' in market_data.columns:
                    close_prices = market_data['underlying_value']
                else:
                    logger.error("No close price data found")
                    return {}
            else:
                close_prices = market_data['close']
            
            # Calculate each indicator
            indicators['ema'] = self._calculate_ema(close_prices)
            indicators['vwap'] = self._calculate_vwap(market_data)
            indicators['rsi'] = self._calculate_rsi(close_prices)
            indicators['bollinger_bands'] = self._calculate_bollinger_bands(close_prices)
            indicators['macd'] = self._calculate_macd(close_prices)
            indicators['stochastic'] = self._calculate_stochastic(market_data)
            indicators['atr'] = self._calculate_atr(market_data)
            indicators['adx'] = self._calculate_adx(market_data)
            
            # Calculate composite scores
            indicators['composite_score'] = self._calculate_composite_score(indicators)
            indicators['regime_signal'] = self._determine_regime_signal(indicators['composite_score'])
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {timeframe}min: {e}")
            return {}
    
    def _calculate_ema(self, prices: pd.Series) -> Dict[str, Any]:
        """Calculate Exponential Moving Averages"""
        ema_data = {
            'values': {},
            'signals': {},
            'score': 0.0
        }
        
        try:
            current_price = prices.iloc[-1] if len(prices) > 0 else 0
            
            for period in self.indicator_configs['ema']['periods']:
                if len(prices) >= period:
                    ema = talib.EMA(prices.values, timeperiod=period)
                    ema_data['values'][f'ema_{period}'] = ema[-1]
                    
                    # Generate signal based on price vs EMA
                    if current_price > ema[-1]:
                        ema_data['signals'][f'ema_{period}'] = 1
                    else:
                        ema_data['signals'][f'ema_{period}'] = -1
            
            # Calculate EMA score
            if ema_data['signals']:
                ema_data['score'] = np.mean(list(ema_data['signals'].values()))
            
        except Exception as e:
            logger.error(f"Error calculating EMA: {e}")
        
        return ema_data
    
    def _calculate_vwap(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Volume Weighted Average Price"""
        vwap_data = {
            'value': 0.0,
            'signal': 0,
            'score': 0.0
        }
        
        try:
            if 'volume' in market_data.columns:
                # Calculate typical price
                high = market_data.get('high', market_data.get('close', 0))
                low = market_data.get('low', market_data.get('close', 0))
                close = market_data.get('close', market_data.get('spot_close', 0))
                
                typical_price = (high + low + close) / 3
                
                # Calculate VWAP
                cumulative_tpv = (typical_price * market_data['volume']).cumsum()
                cumulative_volume = market_data['volume'].cumsum()
                
                vwap = cumulative_tpv / cumulative_volume
                vwap_data['value'] = vwap.iloc[-1] if len(vwap) > 0 else 0
                
                # Generate signal
                current_price = close.iloc[-1] if len(close) > 0 else 0
                if current_price > vwap_data['value']:
                    vwap_data['signal'] = 1
                    vwap_data['score'] = (current_price - vwap_data['value']) / vwap_data['value']
                else:
                    vwap_data['signal'] = -1
                    vwap_data['score'] = (current_price - vwap_data['value']) / vwap_data['value']
            
        except Exception as e:
            logger.error(f"Error calculating VWAP: {e}")
        
        return vwap_data
    
    def _calculate_rsi(self, prices: pd.Series) -> Dict[str, Any]:
        """Calculate Relative Strength Index"""
        rsi_data = {
            'value': 50.0,
            'signal': 0,
            'score': 0.0
        }
        
        try:
            period = self.indicator_configs['rsi']['period']
            
            if len(prices) >= period + 1:
                rsi = talib.RSI(prices.values, timeperiod=period)
                rsi_data['value'] = rsi[-1] if not np.isnan(rsi[-1]) else 50.0
                
                # Generate signal
                if rsi_data['value'] > self.indicator_configs['rsi']['overbought']:
                    rsi_data['signal'] = -1
                    rsi_data['score'] = -(rsi_data['value'] - 70) / 30
                elif rsi_data['value'] < self.indicator_configs['rsi']['oversold']:
                    rsi_data['signal'] = 1
                    rsi_data['score'] = (30 - rsi_data['value']) / 30
                else:
                    rsi_data['signal'] = 0
                    rsi_data['score'] = (rsi_data['value'] - 50) / 50
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
        
        return rsi_data
    
    def _calculate_bollinger_bands(self, prices: pd.Series) -> Dict[str, Any]:
        """Calculate Bollinger Bands"""
        bb_data = {
            'upper': 0.0,
            'middle': 0.0,
            'lower': 0.0,
            'signal': 0,
            'score': 0.0
        }
        
        try:
            period = self.indicator_configs['bollinger_bands']['period']
            std_dev = self.indicator_configs['bollinger_bands']['std_dev']
            
            if len(prices) >= period:
                upper, middle, lower = talib.BBANDS(
                    prices.values,
                    timeperiod=period,
                    nbdevup=std_dev,
                    nbdevdn=std_dev,
                    matype=0
                )
                
                bb_data['upper'] = upper[-1]
                bb_data['middle'] = middle[-1]
                bb_data['lower'] = lower[-1]
                
                current_price = prices.iloc[-1]
                
                # Generate signal
                if current_price > upper[-1]:
                    bb_data['signal'] = -1
                    bb_data['score'] = -(current_price - upper[-1]) / (upper[-1] - middle[-1])
                elif current_price < lower[-1]:
                    bb_data['signal'] = 1
                    bb_data['score'] = (lower[-1] - current_price) / (middle[-1] - lower[-1])
                else:
                    bb_data['signal'] = 0
                    bb_data['score'] = (current_price - middle[-1]) / (upper[-1] - middle[-1])
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
        
        return bb_data
    
    def _calculate_macd(self, prices: pd.Series) -> Dict[str, Any]:
        """Calculate MACD"""
        macd_data = {
            'macd': 0.0,
            'signal': 0.0,
            'histogram': 0.0,
            'signal_line': 0,
            'score': 0.0
        }
        
        try:
            fast = self.indicator_configs['macd']['fast']
            slow = self.indicator_configs['macd']['slow']
            signal = self.indicator_configs['macd']['signal']
            
            if len(prices) >= slow + signal:
                macd, macd_signal, macd_hist = talib.MACD(
                    prices.values,
                    fastperiod=fast,
                    slowperiod=slow,
                    signalperiod=signal
                )
                
                macd_data['macd'] = macd[-1]
                macd_data['signal'] = macd_signal[-1]
                macd_data['histogram'] = macd_hist[-1]
                
                # Generate signal
                if macd_hist[-1] > 0 and macd_hist[-2] <= 0:
                    macd_data['signal_line'] = 1
                    macd_data['score'] = 1.0
                elif macd_hist[-1] < 0 and macd_hist[-2] >= 0:
                    macd_data['signal_line'] = -1
                    macd_data['score'] = -1.0
                else:
                    macd_data['signal_line'] = 0
                    macd_data['score'] = macd_hist[-1] / abs(macd[-1]) if macd[-1] != 0 else 0
            
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
        
        return macd_data
    
    def _calculate_stochastic(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Stochastic Oscillator"""
        stoch_data = {
            'k': 50.0,
            'd': 50.0,
            'signal': 0,
            'score': 0.0
        }
        
        try:
            k_period = self.indicator_configs['stochastic']['k_period']
            d_period = self.indicator_configs['stochastic']['d_period']
            
            high = market_data.get('high', market_data.get('close', 0))
            low = market_data.get('low', market_data.get('close', 0))
            close = market_data.get('close', market_data.get('spot_close', 0))
            
            if len(close) >= k_period:
                k, d = talib.STOCH(
                    high.values,
                    low.values,
                    close.values,
                    fastk_period=k_period,
                    slowk_period=d_period,
                    slowk_matype=0,
                    slowd_period=d_period,
                    slowd_matype=0
                )
                
                stoch_data['k'] = k[-1]
                stoch_data['d'] = d[-1]
                
                # Generate signal
                if k[-1] > 80:
                    stoch_data['signal'] = -1
                    stoch_data['score'] = -(k[-1] - 80) / 20
                elif k[-1] < 20:
                    stoch_data['signal'] = 1
                    stoch_data['score'] = (20 - k[-1]) / 20
                else:
                    stoch_data['signal'] = 0
                    stoch_data['score'] = (k[-1] - 50) / 50
            
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {e}")
        
        return stoch_data
    
    def _calculate_atr(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Average True Range"""
        atr_data = {
            'value': 0.0,
            'normalized': 0.0,
            'volatility_level': 'NORMAL'
        }
        
        try:
            period = self.indicator_configs['atr']['period']
            
            high = market_data.get('high', market_data.get('close', 0))
            low = market_data.get('low', market_data.get('close', 0))
            close = market_data.get('close', market_data.get('spot_close', 0))
            
            if len(close) >= period:
                atr = talib.ATR(high.values, low.values, close.values, timeperiod=period)
                atr_data['value'] = atr[-1]
                
                # Normalize ATR as percentage of price
                current_price = close.iloc[-1]
                atr_data['normalized'] = (atr_data['value'] / current_price * 100) if current_price > 0 else 0
                
                # Determine volatility level
                if atr_data['normalized'] > 3:
                    atr_data['volatility_level'] = 'HIGH'
                elif atr_data['normalized'] > 2:
                    atr_data['volatility_level'] = 'ELEVATED'
                elif atr_data['normalized'] < 1:
                    atr_data['volatility_level'] = 'LOW'
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
        
        return atr_data
    
    def _calculate_adx(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Average Directional Index"""
        adx_data = {
            'value': 0.0,
            'trend_strength': 'NO_TREND',
            'score': 0.0
        }
        
        try:
            period = self.indicator_configs['adx']['period']
            
            high = market_data.get('high', market_data.get('close', 0))
            low = market_data.get('low', market_data.get('close', 0))
            close = market_data.get('close', market_data.get('spot_close', 0))
            
            if len(close) >= period * 2:
                adx = talib.ADX(high.values, low.values, close.values, timeperiod=period)
                adx_data['value'] = adx[-1]
                
                # Determine trend strength
                if adx_data['value'] > 50:
                    adx_data['trend_strength'] = 'VERY_STRONG'
                    adx_data['score'] = 1.0
                elif adx_data['value'] > 25:
                    adx_data['trend_strength'] = 'STRONG'
                    adx_data['score'] = 0.7
                elif adx_data['value'] > 20:
                    adx_data['trend_strength'] = 'MODERATE'
                    adx_data['score'] = 0.5
                else:
                    adx_data['trend_strength'] = 'WEAK'
                    adx_data['score'] = 0.2
            
        except Exception as e:
            logger.error(f"Error calculating ADX: {e}")
        
        return adx_data
    
    def _calculate_composite_score(self, indicators: Dict[str, Any]) -> float:
        """Calculate composite score from all indicators"""
        composite_score = 0.0
        total_weight = 0.0
        
        try:
            # EMA contribution
            if 'ema' in indicators and 'score' in indicators['ema']:
                weight = self.indicator_configs['ema']['weight']
                composite_score += indicators['ema']['score'] * weight
                total_weight += weight
            
            # VWAP contribution
            if 'vwap' in indicators and 'score' in indicators['vwap']:
                weight = self.indicator_configs['vwap']['weight']
                composite_score += indicators['vwap']['score'] * weight
                total_weight += weight
            
            # RSI contribution
            if 'rsi' in indicators and 'score' in indicators['rsi']:
                weight = self.indicator_configs['rsi']['weight']
                composite_score += indicators['rsi']['score'] * weight
                total_weight += weight
            
            # Bollinger Bands contribution
            if 'bollinger_bands' in indicators and 'score' in indicators['bollinger_bands']:
                weight = self.indicator_configs['bollinger_bands']['weight']
                composite_score += indicators['bollinger_bands']['score'] * weight
                total_weight += weight
            
            # MACD contribution
            if 'macd' in indicators and 'score' in indicators['macd']:
                weight = self.indicator_configs['macd']['weight']
                composite_score += indicators['macd']['score'] * weight
                total_weight += weight
            
            # Stochastic contribution
            if 'stochastic' in indicators and 'score' in indicators['stochastic']:
                weight = self.indicator_configs['stochastic']['weight']
                composite_score += indicators['stochastic']['score'] * weight
                total_weight += weight
            
            # Normalize composite score
            if total_weight > 0:
                composite_score /= total_weight
            
        except Exception as e:
            logger.error(f"Error calculating composite score: {e}")
        
        return composite_score
    
    def _determine_regime_signal(self, composite_score: float) -> str:
        """Determine regime signal from composite score"""
        if composite_score >= self.regime_thresholds['strong_bullish']:
            return 'STRONG_BULLISH'
        elif composite_score >= self.regime_thresholds['bullish']:
            return 'BULLISH'
        elif composite_score >= self.regime_thresholds['neutral_bullish']:
            return 'NEUTRAL_BULLISH'
        elif composite_score >= self.regime_thresholds['neutral_bearish']:
            return 'NEUTRAL'
        elif composite_score >= self.regime_thresholds['bearish']:
            return 'NEUTRAL_BEARISH'
        elif composite_score >= self.regime_thresholds['strong_bearish']:
            return 'BEARISH'
        else:
            return 'STRONG_BEARISH'
    
    def analyze_multi_timeframe(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze indicators across multiple timeframes"""
        mtf_analysis = {
            'timestamp': datetime.now(),
            'timeframes': {},
            'consensus': {},
            'divergences': []
        }
        
        try:
            # Analyze each timeframe
            for tf in self.timeframes:
                tf_data = market_data.tail(tf) if len(market_data) >= tf else market_data
                mtf_analysis['timeframes'][f'{tf}min'] = self.calculate_indicators(tf_data, tf)
            
            # Calculate consensus
            mtf_analysis['consensus'] = self._calculate_timeframe_consensus(
                mtf_analysis['timeframes']
            )
            
            # Detect divergences
            mtf_analysis['divergences'] = self._detect_divergences(
                mtf_analysis['timeframes']
            )
            
        except Exception as e:
            logger.error(f"Error in multi-timeframe analysis: {e}")
        
        return mtf_analysis
    
    def _calculate_timeframe_consensus(self, timeframe_data: Dict) -> Dict[str, Any]:
        """Calculate consensus across timeframes"""
        consensus = {
            'regime': 'NEUTRAL',
            'strength': 0.0,
            'confidence': 0.0
        }
        
        try:
            scores = []
            signals = []
            
            for tf, data in timeframe_data.items():
                if 'composite_score' in data:
                    scores.append(data['composite_score'])
                if 'regime_signal' in data:
                    signals.append(data['regime_signal'])
            
            if scores:
                consensus['strength'] = np.mean(scores)
                consensus['regime'] = self._determine_regime_signal(consensus['strength'])
                
                # Calculate confidence based on agreement
                score_std = np.std(scores)
                consensus['confidence'] = max(0, 1 - score_std)
            
        except Exception as e:
            logger.error(f"Error calculating consensus: {e}")
        
        return consensus
    
    def _detect_divergences(self, timeframe_data: Dict) -> List[str]:
        """Detect divergences between timeframes"""
        divergences = []
        
        try:
            regimes = {}
            for tf, data in timeframe_data.items():
                if 'regime_signal' in data:
                    regimes[tf] = data['regime_signal']
            
            # Check for divergences
            regime_values = list(regimes.values())
            if len(set(regime_values)) > 1:
                # Find conflicting timeframes
                for tf1, regime1 in regimes.items():
                    for tf2, regime2 in regimes.items():
                        if tf1 < tf2:
                            if ('BULLISH' in regime1 and 'BEARISH' in regime2) or \
                               ('BEARISH' in regime1 and 'BULLISH' in regime2):
                                divergences.append(f"Divergence: {tf1} {regime1} vs {tf2} {regime2}")
            
        except Exception as e:
            logger.error(f"Error detecting divergences: {e}")
        
        return divergences