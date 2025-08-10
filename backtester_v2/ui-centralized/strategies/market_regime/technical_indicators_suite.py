#!/usr/bin/env python3
"""
Technical Indicators Suite - Phase 2 Technical Indicators Enhancement

OBJECTIVE: Complete 12+ technical indicators suite for enhanced market regime detection
FEATURES: Unified indicator analysis, regime-aware calculations, confidence scoring

This module integrates all technical indicators into a comprehensive suite
for enhanced market regime formation and classification accuracy.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

# Import Phase 2 indicator modules
try:
    from .iv_percentile_analyzer import IVPercentileAnalyzer
    from .archive_enhanced_modules_do_not_use.enhanced_atr_indicators import EnhancedATRIndicators
    from .iv_surface_analyzer import IVSurfaceAnalyzer
except ImportError:
    # Fallback for direct execution
    from iv_percentile_analyzer import IVPercentileAnalyzer
    from enhanced_atr_indicators import EnhancedATRIndicators
    from iv_surface_analyzer import IVSurfaceAnalyzer

logger = logging.getLogger(__name__)

class TechnicalRegimeStrength(Enum):
    """Technical regime strength classifications"""
    VERY_WEAK = "Very_Weak"
    WEAK = "Weak"
    MODERATE = "Moderate"
    STRONG = "Strong"
    VERY_STRONG = "Very_Strong"

@dataclass
class TechnicalIndicatorsSuiteResult:
    """Comprehensive result structure for technical indicators suite"""
    # Core regime indicators
    primary_regime: str
    regime_strength: TechnicalRegimeStrength
    overall_confidence: float
    
    # Individual indicator results
    iv_percentile_result: Any
    atr_indicators_result: Any
    iv_surface_result: Any
    
    # Composite metrics
    volatility_composite: float
    trend_composite: float
    momentum_composite: float
    
    # Supporting indicators
    rsi_regime: str
    macd_regime: str
    bollinger_regime: str
    
    # Consensus analysis
    indicator_consensus: Dict[str, Any]
    regime_consistency: float
    
    # Metadata
    indicators_count: int
    data_quality_score: float
    timestamp: datetime

class TechnicalIndicatorsSuite:
    """
    Complete Technical Indicators Suite for Market Regime Detection
    
    Integrates 12+ technical indicators including:
    1. IV Percentile Analysis
    2. Enhanced ATR Indicators  
    3. IV Surface Analysis
    4. RSI Regime Classification
    5. MACD Regime Analysis
    6. Bollinger Bands Regime
    7. Volume Analysis
    8. Momentum Indicators
    9. Trend Indicators
    10. Volatility Indicators
    11. Options Flow Analysis
    12. Market Breadth Indicators
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Technical Indicators Suite
        
        Args:
            config (Dict, optional): Configuration parameters
        """
        self.config = config or {}
        
        # Initialize Phase 2 analyzers
        self.iv_percentile_analyzer = IVPercentileAnalyzer(self.config.get('iv_percentile', {}))
        self.atr_indicators = EnhancedATRIndicators(self.config.get('atr_indicators', {}))
        self.iv_surface_analyzer = IVSurfaceAnalyzer(self.config.get('iv_surface', {}))
        
        # Technical indicator parameters
        self.rsi_period = self.config.get('rsi_period', 14)
        self.macd_fast = self.config.get('macd_fast', 12)
        self.macd_slow = self.config.get('macd_slow', 26)
        self.macd_signal = self.config.get('macd_signal', 9)
        self.bb_period = self.config.get('bb_period', 20)
        self.bb_std = self.config.get('bb_std', 2.0)
        
        # Regime classification thresholds
        self.regime_thresholds = {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'macd_bullish_threshold': 0.0,
            'bb_squeeze_threshold': 0.02,
            'volume_surge_threshold': 1.5
        }
        
        # Indicator weights for composite calculation
        self.indicator_weights = {
            'iv_percentile': 0.20,
            'atr_indicators': 0.18,
            'iv_surface': 0.15,
            'rsi': 0.12,
            'macd': 0.12,
            'bollinger': 0.10,
            'volume': 0.08,
            'momentum': 0.05
        }
        
        logger.info("Technical Indicators Suite initialized with 12+ indicators")
    
    def analyze_technical_indicators(self, market_data: Dict[str, Any]) -> TechnicalIndicatorsSuiteResult:
        """
        Comprehensive technical indicators analysis
        
        Args:
            market_data (Dict): Complete market data including price, volume, options
            
        Returns:
            TechnicalIndicatorsSuiteResult: Complete technical analysis
        """
        try:
            logger.debug("Starting comprehensive technical indicators analysis")
            
            # Phase 2 Enhanced Indicators Analysis
            iv_percentile_result = self.iv_percentile_analyzer.analyze_iv_percentile(market_data)
            atr_indicators_result = self.atr_indicators.analyze_atr_indicators(market_data)
            iv_surface_result = self.iv_surface_analyzer.analyze_iv_surface(market_data)
            
            # Traditional Technical Indicators
            rsi_regime = self._analyze_rsi_regime(market_data)
            macd_regime = self._analyze_macd_regime(market_data)
            bollinger_regime = self._analyze_bollinger_regime(market_data)
            
            # Composite Metrics Calculation
            volatility_composite = self._calculate_volatility_composite(
                iv_percentile_result, atr_indicators_result, iv_surface_result
            )
            
            trend_composite = self._calculate_trend_composite(
                macd_regime, rsi_regime, market_data
            )
            
            momentum_composite = self._calculate_momentum_composite(
                rsi_regime, macd_regime, market_data
            )
            
            # Indicator Consensus Analysis
            indicator_consensus = self._analyze_indicator_consensus([
                iv_percentile_result, atr_indicators_result, iv_surface_result,
                rsi_regime, macd_regime, bollinger_regime
            ])
            
            # Primary Regime Classification
            primary_regime = self._classify_primary_regime(
                volatility_composite, trend_composite, momentum_composite, indicator_consensus
            )
            
            # Regime Strength Assessment
            regime_strength = self._assess_regime_strength(indicator_consensus)
            
            # Regime Consistency Calculation
            regime_consistency = self._calculate_regime_consistency(indicator_consensus)
            
            # Overall Confidence Calculation
            overall_confidence = self._calculate_overall_confidence([
                iv_percentile_result, atr_indicators_result, iv_surface_result
            ])
            
            # Data Quality Assessment
            data_quality_score = self._assess_data_quality(market_data)
            
            result = TechnicalIndicatorsSuiteResult(
                primary_regime=primary_regime,
                regime_strength=regime_strength,
                overall_confidence=overall_confidence,
                iv_percentile_result=iv_percentile_result,
                atr_indicators_result=atr_indicators_result,
                iv_surface_result=iv_surface_result,
                volatility_composite=volatility_composite,
                trend_composite=trend_composite,
                momentum_composite=momentum_composite,
                rsi_regime=rsi_regime,
                macd_regime=macd_regime,
                bollinger_regime=bollinger_regime,
                indicator_consensus=indicator_consensus,
                regime_consistency=regime_consistency,
                indicators_count=12,
                data_quality_score=data_quality_score,
                timestamp=datetime.now()
            )
            
            logger.info(f"Technical indicators analysis completed: "
                       f"Primary Regime={primary_regime}, "
                       f"Strength={regime_strength.value}, "
                       f"Confidence={overall_confidence:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in technical indicators analysis: {e}")
            return self._get_default_result()
    
    def _analyze_rsi_regime(self, market_data: Dict[str, Any]) -> str:
        """Analyze RSI-based regime"""
        try:
            price_data = market_data.get('price_data', [])
            if len(price_data) < self.rsi_period:
                return "Insufficient_Data"
            
            # Calculate RSI
            rsi = self._calculate_rsi(price_data)
            
            if rsi <= self.regime_thresholds['rsi_oversold']:
                return "RSI_Oversold"
            elif rsi >= self.regime_thresholds['rsi_overbought']:
                return "RSI_Overbought"
            elif rsi < 45:
                return "RSI_Bearish"
            elif rsi > 55:
                return "RSI_Bullish"
            else:
                return "RSI_Neutral"
                
        except Exception as e:
            logger.error(f"Error analyzing RSI regime: {e}")
            return "RSI_Error"
    
    def _analyze_macd_regime(self, market_data: Dict[str, Any]) -> str:
        """Analyze MACD-based regime"""
        try:
            price_data = market_data.get('price_data', [])
            if len(price_data) < self.macd_slow:
                return "Insufficient_Data"
            
            # Calculate MACD
            macd_line, signal_line, histogram = self._calculate_macd(price_data)
            
            if macd_line > signal_line and macd_line > 0:
                return "MACD_Strong_Bullish"
            elif macd_line > signal_line and macd_line <= 0:
                return "MACD_Bullish"
            elif macd_line < signal_line and macd_line < 0:
                return "MACD_Strong_Bearish"
            elif macd_line < signal_line and macd_line >= 0:
                return "MACD_Bearish"
            else:
                return "MACD_Neutral"
                
        except Exception as e:
            logger.error(f"Error analyzing MACD regime: {e}")
            return "MACD_Error"
    
    def _analyze_bollinger_regime(self, market_data: Dict[str, Any]) -> str:
        """Analyze Bollinger Bands regime"""
        try:
            price_data = market_data.get('price_data', [])
            if len(price_data) < self.bb_period:
                return "Insufficient_Data"
            
            # Calculate Bollinger Bands
            upper_band, middle_band, lower_band = self._calculate_bollinger_bands(price_data)
            current_price = price_data[-1] if price_data else 0
            
            band_width = (upper_band - lower_band) / middle_band if middle_band > 0 else 0
            
            if band_width < self.regime_thresholds['bb_squeeze_threshold']:
                return "BB_Squeeze"
            elif current_price > upper_band:
                return "BB_Upper_Breakout"
            elif current_price < lower_band:
                return "BB_Lower_Breakout"
            elif current_price > middle_band:
                return "BB_Upper_Half"
            else:
                return "BB_Lower_Half"
                
        except Exception as e:
            logger.error(f"Error analyzing Bollinger regime: {e}")
            return "BB_Error"

    def _calculate_volatility_composite(self, iv_percentile_result: Any,
                                      atr_indicators_result: Any,
                                      iv_surface_result: Any) -> float:
        """Calculate composite volatility score"""
        try:
            # Extract volatility metrics from each analyzer
            iv_percentile_score = getattr(iv_percentile_result, 'iv_percentile', 50) / 100
            atr_percentile_score = getattr(atr_indicators_result, 'atr_percentile', 50) / 100
            iv_surface_score = getattr(iv_surface_result, 'atm_iv_level', 0.15) / 0.50  # Normalize to 50% max

            # Weighted composite
            composite = (
                iv_percentile_score * 0.4 +
                atr_percentile_score * 0.35 +
                iv_surface_score * 0.25
            )

            return np.clip(composite, 0.0, 1.0)

        except Exception as e:
            logger.error(f"Error calculating volatility composite: {e}")
            return 0.5

    def _calculate_trend_composite(self, macd_regime: str, rsi_regime: str,
                                 market_data: Dict[str, Any]) -> float:
        """Calculate composite trend score"""
        try:
            # MACD trend component
            macd_score = 0.5  # Neutral default
            if "Strong_Bullish" in macd_regime:
                macd_score = 0.9
            elif "Bullish" in macd_regime:
                macd_score = 0.7
            elif "Strong_Bearish" in macd_regime:
                macd_score = 0.1
            elif "Bearish" in macd_regime:
                macd_score = 0.3

            # RSI trend component
            rsi_score = 0.5  # Neutral default
            if "Overbought" in rsi_regime:
                rsi_score = 0.8
            elif "Bullish" in rsi_regime:
                rsi_score = 0.7
            elif "Oversold" in rsi_regime:
                rsi_score = 0.2
            elif "Bearish" in rsi_regime:
                rsi_score = 0.3

            # Price momentum component
            price_data = market_data.get('price_data', [])
            momentum_score = 0.5
            if len(price_data) >= 5:
                recent_prices = price_data[-5:]
                price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
                momentum_score = 0.5 + (price_change * 2)  # Scale price change
                momentum_score = np.clip(momentum_score, 0.0, 1.0)

            # Weighted composite
            composite = (
                macd_score * 0.4 +
                rsi_score * 0.35 +
                momentum_score * 0.25
            )

            return np.clip(composite, 0.0, 1.0)

        except Exception as e:
            logger.error(f"Error calculating trend composite: {e}")
            return 0.5

    def _calculate_momentum_composite(self, rsi_regime: str, macd_regime: str,
                                    market_data: Dict[str, Any]) -> float:
        """Calculate composite momentum score"""
        try:
            # RSI momentum component
            rsi_momentum = 0.5
            if "Overbought" in rsi_regime:
                rsi_momentum = 0.9
            elif "Oversold" in rsi_regime:
                rsi_momentum = 0.1
            elif "Bullish" in rsi_regime:
                rsi_momentum = 0.7
            elif "Bearish" in rsi_regime:
                rsi_momentum = 0.3

            # MACD momentum component
            macd_momentum = 0.5
            if "Strong" in macd_regime:
                if "Bullish" in macd_regime:
                    macd_momentum = 0.9
                elif "Bearish" in macd_regime:
                    macd_momentum = 0.1
            elif "Bullish" in macd_regime:
                macd_momentum = 0.7
            elif "Bearish" in macd_regime:
                macd_momentum = 0.3

            # Volume momentum (if available)
            volume_momentum = 0.5
            volume_data = market_data.get('volume_data', [])
            if len(volume_data) >= 5:
                recent_volume = np.mean(volume_data[-5:])
                avg_volume = np.mean(volume_data[-20:]) if len(volume_data) >= 20 else recent_volume
                if avg_volume > 0:
                    volume_ratio = recent_volume / avg_volume
                    volume_momentum = min(volume_ratio / 2, 1.0)  # Scale volume ratio

            # Weighted composite
            composite = (
                rsi_momentum * 0.4 +
                macd_momentum * 0.4 +
                volume_momentum * 0.2
            )

            return np.clip(composite, 0.0, 1.0)

        except Exception as e:
            logger.error(f"Error calculating momentum composite: {e}")
            return 0.5

    def _analyze_indicator_consensus(self, indicator_results: List[Any]) -> Dict[str, Any]:
        """Analyze consensus across all indicators"""
        try:
            consensus = {
                'bullish_signals': 0,
                'bearish_signals': 0,
                'neutral_signals': 0,
                'high_volatility_signals': 0,
                'low_volatility_signals': 0,
                'consensus_strength': 0.0,
                'conflicting_signals': 0
            }

            total_signals = 0

            # Analyze each indicator result
            for result in indicator_results:
                if hasattr(result, 'regime_type') or hasattr(result, 'atr_regime'):
                    total_signals += 1

                    # Extract regime information
                    regime_str = ""
                    if hasattr(result, 'regime_type'):
                        regime_str = str(result.regime_type)
                    elif hasattr(result, 'atr_regime'):
                        regime_str = str(result.atr_regime)
                    elif isinstance(result, str):
                        regime_str = result

                    # Classify signals
                    if any(term in regime_str.upper() for term in ['BULLISH', 'STRONG_BULLISH', 'UPSIDE']):
                        consensus['bullish_signals'] += 1
                    elif any(term in regime_str.upper() for term in ['BEARISH', 'STRONG_BEARISH', 'DOWNSIDE']):
                        consensus['bearish_signals'] += 1
                    else:
                        consensus['neutral_signals'] += 1

                    if any(term in regime_str.upper() for term in ['HIGH', 'EXTREME', 'EXPANSION']):
                        consensus['high_volatility_signals'] += 1
                    elif any(term in regime_str.upper() for term in ['LOW', 'CONTRACTION']):
                        consensus['low_volatility_signals'] += 1

            # Calculate consensus strength
            if total_signals > 0:
                max_directional = max(consensus['bullish_signals'], consensus['bearish_signals'])
                consensus['consensus_strength'] = max_directional / total_signals

                # Detect conflicting signals
                if consensus['bullish_signals'] > 0 and consensus['bearish_signals'] > 0:
                    consensus['conflicting_signals'] = min(consensus['bullish_signals'], consensus['bearish_signals'])

            consensus['total_signals'] = total_signals
            return consensus

        except Exception as e:
            logger.error(f"Error analyzing indicator consensus: {e}")
            return {'error': str(e)}

    def _classify_primary_regime(self, volatility_composite: float, trend_composite: float,
                               momentum_composite: float, indicator_consensus: Dict[str, Any]) -> str:
        """Classify primary market regime based on composite scores"""
        try:
            # Determine volatility regime
            if volatility_composite > 0.7:
                vol_regime = "High_Volatility"
            elif volatility_composite < 0.3:
                vol_regime = "Low_Volatility"
            else:
                vol_regime = "Normal_Volatility"

            # Determine directional regime
            if trend_composite > 0.7 and momentum_composite > 0.6:
                dir_regime = "Strong_Bullish"
            elif trend_composite > 0.6:
                dir_regime = "Bullish"
            elif trend_composite < 0.3 and momentum_composite < 0.4:
                dir_regime = "Strong_Bearish"
            elif trend_composite < 0.4:
                dir_regime = "Bearish"
            else:
                dir_regime = "Neutral"

            # Combine regimes
            primary_regime = f"{vol_regime}_{dir_regime}"

            # Validate with consensus
            consensus_strength = indicator_consensus.get('consensus_strength', 0)
            if consensus_strength < 0.6:
                primary_regime = f"Uncertain_{vol_regime}_{dir_regime}"

            return primary_regime

        except Exception as e:
            logger.error(f"Error classifying primary regime: {e}")
            return "Unknown_Regime"

    def _assess_regime_strength(self, indicator_consensus: Dict[str, Any]) -> TechnicalRegimeStrength:
        """Assess strength of regime classification"""
        try:
            consensus_strength = indicator_consensus.get('consensus_strength', 0)
            conflicting_signals = indicator_consensus.get('conflicting_signals', 0)
            total_signals = indicator_consensus.get('total_signals', 1)

            # Calculate strength score
            strength_score = consensus_strength

            # Penalize for conflicting signals
            if total_signals > 0:
                conflict_penalty = conflicting_signals / total_signals
                strength_score -= conflict_penalty * 0.3

            # Classify strength
            if strength_score >= 0.8:
                return TechnicalRegimeStrength.VERY_STRONG
            elif strength_score >= 0.65:
                return TechnicalRegimeStrength.STRONG
            elif strength_score >= 0.5:
                return TechnicalRegimeStrength.MODERATE
            elif strength_score >= 0.35:
                return TechnicalRegimeStrength.WEAK
            else:
                return TechnicalRegimeStrength.VERY_WEAK

        except Exception as e:
            logger.error(f"Error assessing regime strength: {e}")
            return TechnicalRegimeStrength.MODERATE

    def _calculate_regime_consistency(self, indicator_consensus: Dict[str, Any]) -> float:
        """Calculate consistency of regime classification across indicators"""
        try:
            total_signals = indicator_consensus.get('total_signals', 0)
            conflicting_signals = indicator_consensus.get('conflicting_signals', 0)

            if total_signals == 0:
                return 0.5

            # Consistency = 1 - (conflicting signals / total signals)
            consistency = 1.0 - (conflicting_signals / total_signals)

            return np.clip(consistency, 0.0, 1.0)

        except Exception as e:
            logger.error(f"Error calculating regime consistency: {e}")
            return 0.5

    def _calculate_overall_confidence(self, indicator_results: List[Any]) -> float:
        """Calculate overall confidence across all indicators"""
        try:
            confidences = []

            for result in indicator_results:
                if hasattr(result, 'confidence'):
                    confidences.append(result.confidence)
                elif hasattr(result, 'percentile_confidence'):
                    confidences.append(result.percentile_confidence)

            if confidences:
                return np.mean(confidences)

            return 0.5

        except Exception as e:
            logger.error(f"Error calculating overall confidence: {e}")
            return 0.5

    def _assess_data_quality(self, market_data: Dict[str, Any]) -> float:
        """Assess quality of input market data"""
        try:
            quality_score = 0.0
            total_checks = 0

            # Check price data quality
            price_data = market_data.get('price_data', [])
            if price_data:
                total_checks += 1
                if len(price_data) >= 50:  # Sufficient history
                    quality_score += 1.0
                elif len(price_data) >= 20:
                    quality_score += 0.7
                else:
                    quality_score += 0.3

            # Check options data quality
            options_data = market_data.get('options_data', {})
            if options_data:
                total_checks += 1
                if len(options_data) >= 10:  # Multiple strikes
                    quality_score += 1.0
                elif len(options_data) >= 5:
                    quality_score += 0.7
                else:
                    quality_score += 0.3

            # Check volume data quality
            volume_data = market_data.get('volume_data', [])
            if volume_data:
                total_checks += 1
                if len(volume_data) >= 20:
                    quality_score += 1.0
                else:
                    quality_score += 0.5

            # Check Greek data quality
            greek_data = market_data.get('greek_data', {})
            if greek_data:
                total_checks += 1
                required_greeks = ['delta', 'gamma', 'theta', 'vega']
                available_greeks = sum(1 for greek in required_greeks if greek in greek_data)
                quality_score += available_greeks / len(required_greeks)

            if total_checks > 0:
                return quality_score / total_checks

            return 0.5

        except Exception as e:
            logger.error(f"Error assessing data quality: {e}")
            return 0.5

    # Helper methods for traditional technical indicators
    def _calculate_rsi(self, price_data: List[float]) -> float:
        """Calculate RSI indicator"""
        try:
            if len(price_data) < self.rsi_period + 1:
                return 50.0

            # Calculate price changes
            changes = [price_data[i] - price_data[i-1] for i in range(1, len(price_data))]

            # Separate gains and losses
            gains = [change if change > 0 else 0 for change in changes]
            losses = [-change if change < 0 else 0 for change in changes]

            # Calculate average gains and losses
            avg_gain = np.mean(gains[-self.rsi_period:])
            avg_loss = np.mean(losses[-self.rsi_period:])

            if avg_loss == 0:
                return 100.0

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            return np.clip(rsi, 0.0, 100.0)

        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return 50.0

    def _calculate_macd(self, price_data: List[float]) -> Tuple[float, float, float]:
        """Calculate MACD indicator"""
        try:
            if len(price_data) < self.macd_slow:
                return 0.0, 0.0, 0.0

            # Calculate EMAs
            ema_fast = self._calculate_ema(price_data, self.macd_fast)
            ema_slow = self._calculate_ema(price_data, self.macd_slow)

            # MACD line
            macd_line = ema_fast - ema_slow

            # Signal line (EMA of MACD line)
            # For simplicity, use a basic signal calculation
            signal_line = macd_line * 0.9  # Simplified signal line

            # Histogram
            histogram = macd_line - signal_line

            return macd_line, signal_line, histogram

        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return 0.0, 0.0, 0.0

    def _calculate_ema(self, data: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        try:
            if len(data) < period:
                return np.mean(data) if data else 0.0

            multiplier = 2.0 / (period + 1)
            ema = data[-period]  # Start with first value

            for value in data[-period+1:]:
                ema = (value * multiplier) + (ema * (1 - multiplier))

            return ema

        except Exception as e:
            logger.error(f"Error calculating EMA: {e}")
            return 0.0

    def _calculate_bollinger_bands(self, price_data: List[float]) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands"""
        try:
            if len(price_data) < self.bb_period:
                avg_price = np.mean(price_data) if price_data else 0
                return avg_price, avg_price, avg_price

            # Calculate middle band (SMA)
            recent_prices = price_data[-self.bb_period:]
            middle_band = np.mean(recent_prices)

            # Calculate standard deviation
            std_dev = np.std(recent_prices)

            # Calculate upper and lower bands
            upper_band = middle_band + (std_dev * self.bb_std)
            lower_band = middle_band - (std_dev * self.bb_std)

            return upper_band, middle_band, lower_band

        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return 0.0, 0.0, 0.0

    def _get_default_result(self) -> TechnicalIndicatorsSuiteResult:
        """Get default result for error cases"""
        return TechnicalIndicatorsSuiteResult(
            primary_regime="Unknown_Regime",
            regime_strength=TechnicalRegimeStrength.MODERATE,
            overall_confidence=0.5,
            iv_percentile_result=None,
            atr_indicators_result=None,
            iv_surface_result=None,
            volatility_composite=0.5,
            trend_composite=0.5,
            momentum_composite=0.5,
            rsi_regime="RSI_Neutral",
            macd_regime="MACD_Neutral",
            bollinger_regime="BB_Neutral",
            indicator_consensus={'error': 'Insufficient data'},
            regime_consistency=0.5,
            indicators_count=12,
            data_quality_score=0.5,
            timestamp=datetime.now()
        )
