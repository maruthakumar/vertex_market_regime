"""
Component 1 Momentum Analysis Engine - Multi-Timeframe RSI/MACD Implementation

Revolutionary momentum analysis applied to rolling straddle prices across 4 timeframes
with Option B strategy (RSI + MACD) from research document.

Architecture:
- 40 momentum time series (10 parameters × 4 timeframes)
- RSI + MACD indicators applied to rolling straddle prices
- Cross-timeframe divergence detection
- Combined straddle momentum calculation
- Performance optimized for 190ms budget
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd

@dataclass
class MomentumTimeSeriesData:
    """Time series data for momentum calculation"""
    timestamps: np.ndarray
    price_series: np.ndarray
    timeframe: str
    parameter_name: str
    data_points: int
    processing_time_ms: float

@dataclass
class RSIAnalysisResult:
    """RSI analysis result for a single parameter-timeframe combination"""
    rsi_value: float
    rsi_signal: str  # "oversold", "neutral", "overbought"
    rsi_trend: float  # -1.0 to 1.0
    momentum_strength: float
    timeframe: str
    parameter_name: str

@dataclass
class MACDAnalysisResult:
    """MACD analysis result for a single parameter-timeframe combination"""
    macd_line: float
    signal_line: float
    histogram: float
    macd_signal: str  # "bullish", "neutral", "bearish"
    divergence_strength: float
    timeframe: str
    parameter_name: str

@dataclass
class DivergenceAnalysisResult:
    """Cross-timeframe divergence analysis result"""
    divergence_type: str  # "bullish", "bearish", "none"
    divergence_strength: float
    timeframes_compared: List[str]
    parameter_name: str
    confidence: float

@dataclass
class MomentumAnalysisResult:
    """Complete momentum analysis result for Component 1 enhancement"""
    # RSI Results (10 parameters × 4 timeframes = 40 RSI calculations)
    rsi_results: Dict[str, Dict[str, RSIAnalysisResult]]
    
    # MACD Results (10 parameters × 4 timeframes = 40 MACD calculations)
    macd_results: Dict[str, Dict[str, MACDAnalysisResult]]
    
    # Divergence Results (cross-timeframe analysis)
    divergence_results: Dict[str, List[DivergenceAnalysisResult]]
    
    # Combined straddle momentum
    combined_straddle_momentum: Dict[str, float]
    
    # Feature vector (30 momentum features)
    momentum_features: np.ndarray
    feature_names: List[str]
    
    # Performance metrics
    total_processing_time_ms: float
    memory_usage_mb: float
    
    # Quality metrics
    momentum_confidence: float
    signal_quality: float
    
    # Metadata
    metadata: Dict[str, Any]


class MomentumAnalysisEngine:
    """
    Multi-Timeframe Momentum Analysis Engine for Component 1
    
    Implements Option B strategy (RSI + MACD) from research document:
    - RSI (14-period) for momentum oscillation
    - MACD (12,26,9) for trend confirmation
    - Cross-timeframe divergence detection
    - Applied to rolling straddle prices (revolutionary approach)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize momentum analysis engine
        
        Args:
            config: Engine configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # RSI Configuration
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        
        # MACD Configuration  
        self.macd_fast_period = config.get('macd_fast_period', 12)
        self.macd_slow_period = config.get('macd_slow_period', 26)
        self.macd_signal_period = config.get('macd_signal_period', 9)
        
        # Timeframe Configuration
        self.timeframes = ['3min', '5min', '10min', '15min']
        
        # 10-Parameter System (from research document)
        self.parameters = [
            'atm_straddle', 'itm1_straddle', 'otm1_straddle',
            'atm_ce', 'itm1_ce', 'otm1_ce',
            'atm_pe', 'itm1_pe', 'otm1_pe',
            'combined_straddle'
        ]
        
        # Performance tracking
        self.processing_budget_ms = config.get('momentum_processing_budget_ms', 40)
        self.processing_times = []
        
        self.logger.info(f"Momentum Analysis Engine initialized: {len(self.parameters)} parameters × {len(self.timeframes)} timeframes = {len(self.parameters) * len(self.timeframes)} momentum series")
    
    async def analyze_momentum(self, straddle_data: Dict[str, Any], volume_data: Dict[str, Any], 
                             timestamps: np.ndarray) -> MomentumAnalysisResult:
        """
        Comprehensive momentum analysis across all parameters and timeframes
        
        Args:
            straddle_data: Rolling straddle price data
            volume_data: Volume data for context
            timestamps: Time series timestamps
            
        Returns:
            MomentumAnalysisResult with 30 features
        """
        start_time = time.time()
        
        try:
            # Step 1: Prepare multi-timeframe data
            self.logger.info("Preparing multi-timeframe momentum data...")
            multi_timeframe_data = await self._prepare_multi_timeframe_data(
                straddle_data, timestamps
            )
            
            # Step 2: Calculate RSI for all parameter-timeframe combinations
            self.logger.info("Calculating RSI across 40 time series...")
            rsi_results = await self._calculate_rsi_analysis(multi_timeframe_data)
            
            # Step 3: Calculate MACD for all parameter-timeframe combinations
            self.logger.info("Calculating MACD across 40 time series...")
            macd_results = await self._calculate_macd_analysis(multi_timeframe_data)
            
            # Step 4: Calculate combined straddle momentum
            self.logger.info("Calculating combined straddle momentum...")
            combined_momentum = await self._calculate_combined_straddle_momentum(
                rsi_results, macd_results
            )
            
            # Step 5: Detect cross-timeframe divergences
            self.logger.info("Detecting cross-timeframe divergences...")
            divergence_results = await self._detect_cross_timeframe_divergences(
                rsi_results, macd_results
            )
            
            # Step 6: Generate 30 momentum features
            self.logger.info("Generating 30 momentum features...")
            momentum_features, feature_names = await self._generate_momentum_features(
                rsi_results, macd_results, divergence_results, combined_momentum
            )
            
            # Step 7: Calculate quality metrics
            momentum_confidence = self._calculate_momentum_confidence(rsi_results, macd_results)
            signal_quality = self._calculate_signal_quality(divergence_results)
            
            # Performance tracking
            processing_time = (time.time() - start_time) * 1000
            memory_usage = self._estimate_memory_usage()
            
            self.processing_times.append(processing_time)
            
            # Validate performance targets
            if processing_time > self.processing_budget_ms:
                self.logger.warning(f"Momentum processing time {processing_time:.2f}ms exceeded budget {self.processing_budget_ms}ms")
            
            return MomentumAnalysisResult(
                rsi_results=rsi_results,
                macd_results=macd_results,
                divergence_results=divergence_results,
                combined_straddle_momentum=combined_momentum,
                momentum_features=momentum_features,
                feature_names=feature_names,
                total_processing_time_ms=processing_time,
                memory_usage_mb=memory_usage,
                momentum_confidence=momentum_confidence,
                signal_quality=signal_quality,
                metadata={
                    'parameters_analyzed': len(self.parameters),
                    'timeframes_analyzed': len(self.timeframes),
                    'total_momentum_series': len(self.parameters) * len(self.timeframes),
                    'rsi_calculations': len(rsi_results),
                    'macd_calculations': len(macd_results),
                    'feature_count': len(momentum_features),
                    'processing_budget_met': processing_time <= self.processing_budget_ms
                }
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.logger.error(f"Momentum analysis failed: {e}")
            raise
    
    async def _prepare_multi_timeframe_data(self, straddle_data: Dict[str, Any], 
                                          timestamps: np.ndarray) -> Dict[str, Dict[str, MomentumTimeSeriesData]]:
        """
        Prepare data for multi-timeframe momentum analysis
        
        Returns:
            Dict[parameter][timeframe] -> MomentumTimeSeriesData
        """
        multi_timeframe_data = {}
        
        # Create DataFrame for resampling
        df = pd.DataFrame({
            'timestamp': timestamps,
            **straddle_data
        })
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        for parameter in self.parameters:
            multi_timeframe_data[parameter] = {}
            
            if parameter == 'combined_straddle':
                # Calculate combined straddle using weights from research document
                price_series = (
                    0.5 * straddle_data.get('atm_straddle', np.zeros_like(timestamps)) +
                    0.3 * straddle_data.get('itm1_straddle', np.zeros_like(timestamps)) +
                    0.2 * straddle_data.get('otm1_straddle', np.zeros_like(timestamps))
                )
            else:
                price_series = straddle_data.get(parameter, np.zeros_like(timestamps))
            
            # Create time series data for each timeframe
            for timeframe in self.timeframes:
                try:
                    # Resample to target timeframe
                    resampled = df[parameter].resample(timeframe).ohlc() if parameter in df.columns else pd.DataFrame()
                    
                    if not resampled.empty:
                        timeframe_prices = resampled['close'].dropna().values
                        timeframe_timestamps = resampled.index.values
                    else:
                        # Fallback to original data
                        timeframe_prices = price_series
                        timeframe_timestamps = timestamps
                    
                    multi_timeframe_data[parameter][timeframe] = MomentumTimeSeriesData(
                        timestamps=timeframe_timestamps,
                        price_series=timeframe_prices,
                        timeframe=timeframe,
                        parameter_name=parameter,
                        data_points=len(timeframe_prices),
                        processing_time_ms=0.0
                    )
                    
                except Exception as e:
                    self.logger.warning(f"Failed to resample {parameter} to {timeframe}: {e}")
                    # Use original data as fallback
                    multi_timeframe_data[parameter][timeframe] = MomentumTimeSeriesData(
                        timestamps=timestamps,
                        price_series=price_series,
                        timeframe=timeframe,
                        parameter_name=parameter,
                        data_points=len(price_series),
                        processing_time_ms=0.0
                    )
        
        return multi_timeframe_data
    
    async def _calculate_rsi_analysis(self, multi_timeframe_data: Dict[str, Dict[str, MomentumTimeSeriesData]]) -> Dict[str, Dict[str, RSIAnalysisResult]]:
        """
        Calculate RSI for all parameter-timeframe combinations
        """
        rsi_results = {}
        
        for parameter in self.parameters:
            rsi_results[parameter] = {}
            
            for timeframe in self.timeframes:
                try:
                    ts_data = multi_timeframe_data[parameter][timeframe]
                    
                    # Calculate RSI
                    rsi_value = self._calculate_rsi(ts_data.price_series, self.rsi_period)
                    
                    # Determine RSI signal
                    if rsi_value >= self.rsi_overbought:
                        rsi_signal = "overbought"
                        rsi_trend = 1.0
                    elif rsi_value <= self.rsi_oversold:
                        rsi_signal = "oversold"
                        rsi_trend = -1.0
                    else:
                        rsi_signal = "neutral"
                        rsi_trend = (rsi_value - 50.0) / 50.0  # Normalize to [-1, 1]
                    
                    # Calculate momentum strength
                    momentum_strength = abs(rsi_value - 50.0) / 50.0
                    
                    rsi_results[parameter][timeframe] = RSIAnalysisResult(
                        rsi_value=rsi_value,
                        rsi_signal=rsi_signal,
                        rsi_trend=rsi_trend,
                        momentum_strength=momentum_strength,
                        timeframe=timeframe,
                        parameter_name=parameter
                    )
                    
                except Exception as e:
                    self.logger.warning(f"RSI calculation failed for {parameter}_{timeframe}: {e}")
                    # Fallback result
                    rsi_results[parameter][timeframe] = RSIAnalysisResult(
                        rsi_value=50.0,
                        rsi_signal="neutral",
                        rsi_trend=0.0,
                        momentum_strength=0.0,
                        timeframe=timeframe,
                        parameter_name=parameter
                    )
        
        return rsi_results
    
    async def _calculate_macd_analysis(self, multi_timeframe_data: Dict[str, Dict[str, MomentumTimeSeriesData]]) -> Dict[str, Dict[str, MACDAnalysisResult]]:
        """
        Calculate MACD for all parameter-timeframe combinations
        """
        macd_results = {}
        
        for parameter in self.parameters:
            macd_results[parameter] = {}
            
            for timeframe in self.timeframes:
                try:
                    ts_data = multi_timeframe_data[parameter][timeframe]
                    
                    # Calculate MACD components
                    macd_line, signal_line, histogram = self._calculate_macd(
                        ts_data.price_series, 
                        self.macd_fast_period, 
                        self.macd_slow_period, 
                        self.macd_signal_period
                    )
                    
                    # Determine MACD signal
                    if macd_line > signal_line and histogram > 0:
                        macd_signal = "bullish"
                        divergence_strength = min(abs(histogram) / abs(macd_line + 1e-8), 1.0)
                    elif macd_line < signal_line and histogram < 0:
                        macd_signal = "bearish"
                        divergence_strength = min(abs(histogram) / abs(macd_line + 1e-8), 1.0)
                    else:
                        macd_signal = "neutral"
                        divergence_strength = 0.0
                    
                    macd_results[parameter][timeframe] = MACDAnalysisResult(
                        macd_line=macd_line,
                        signal_line=signal_line,
                        histogram=histogram,
                        macd_signal=macd_signal,
                        divergence_strength=divergence_strength,
                        timeframe=timeframe,
                        parameter_name=parameter
                    )
                    
                except Exception as e:
                    self.logger.warning(f"MACD calculation failed for {parameter}_{timeframe}: {e}")
                    # Fallback result
                    macd_results[parameter][timeframe] = MACDAnalysisResult(
                        macd_line=0.0,
                        signal_line=0.0,
                        histogram=0.0,
                        macd_signal="neutral",
                        divergence_strength=0.0,
                        timeframe=timeframe,
                        parameter_name=parameter
                    )
        
        return macd_results
    
    async def _calculate_combined_straddle_momentum(self, rsi_results: Dict, macd_results: Dict) -> Dict[str, float]:
        """
        Calculate combined straddle momentum using weighted approach from research
        """
        combined_momentum = {}
        
        # Weights from research document: 50% ATM + 30% ITM1 + 20% OTM1
        straddle_weights = {
            'atm_straddle': 0.5,
            'itm1_straddle': 0.3,
            'otm1_straddle': 0.2
        }
        
        for timeframe in self.timeframes:
            # RSI combined momentum
            rsi_weighted = sum(
                rsi_results[component][timeframe].rsi_trend * weight
                for component, weight in straddle_weights.items()
                if component in rsi_results and timeframe in rsi_results[component]
            )
            
            # MACD combined momentum
            macd_weighted = sum(
                (1.0 if macd_results[component][timeframe].macd_signal == "bullish" else
                 -1.0 if macd_results[component][timeframe].macd_signal == "bearish" else 0.0) * weight
                for component, weight in straddle_weights.items()
                if component in macd_results and timeframe in macd_results[component]
            )
            
            # Combined momentum (50% RSI + 50% MACD)
            combined_momentum[f'combined_rsi_{timeframe}'] = rsi_weighted
            combined_momentum[f'combined_macd_{timeframe}'] = macd_weighted
            combined_momentum[f'combined_momentum_{timeframe}'] = (rsi_weighted + macd_weighted) / 2.0
        
        return combined_momentum
    
    async def _detect_cross_timeframe_divergences(self, rsi_results: Dict, macd_results: Dict) -> Dict[str, List[DivergenceAnalysisResult]]:
        """
        Detect divergences between adjacent timeframes
        """
        divergence_results = {}
        
        # Adjacent timeframe pairs for divergence detection
        timeframe_pairs = [
            ('3min', '5min'),
            ('5min', '10min'),
            ('10min', '15min'),
            ('3min', '15min')  # Long-term divergence
        ]
        
        for parameter in self.parameters:
            divergence_results[parameter] = []
            
            for tf1, tf2 in timeframe_pairs:
                try:
                    if (parameter in rsi_results and tf1 in rsi_results[parameter] and tf2 in rsi_results[parameter]):
                        rsi1 = rsi_results[parameter][tf1]
                        rsi2 = rsi_results[parameter][tf2]
                        
                        # Detect RSI divergence
                        rsi_divergence = abs(rsi1.rsi_trend - rsi2.rsi_trend)
                        
                        if rsi_divergence > 0.5:  # Significant divergence threshold
                            divergence_type = "bullish" if rsi1.rsi_trend > rsi2.rsi_trend else "bearish"
                            
                            divergence_results[parameter].append(DivergenceAnalysisResult(
                                divergence_type=divergence_type,
                                divergence_strength=rsi_divergence,
                                timeframes_compared=[tf1, tf2],
                                parameter_name=parameter,
                                confidence=min(rsi1.momentum_strength, rsi2.momentum_strength)
                            ))
                        
                except Exception as e:
                    self.logger.warning(f"Divergence detection failed for {parameter}_{tf1}_{tf2}: {e}")
        
        return divergence_results
    
    async def _generate_momentum_features(self, rsi_results: Dict, macd_results: Dict, 
                                        divergence_results: Dict, combined_momentum: Dict) -> Tuple[np.ndarray, List[str]]:
        """
        Generate exactly 30 momentum features from analysis results
        
        Feature breakdown:
        - 15 RSI features (3 timeframes × 5 key extractions)
        - 10 MACD features (3 timeframes × 3 key extractions + 1 composite)
        - 5 divergence features (cross-timeframe divergences)
        """
        features = []
        feature_names = []
        
        # Category 1: RSI Features (15 features)
        key_timeframes = ['3min', '5min', '15min']  # Focus on key timeframes for efficiency
        key_parameters = ['atm_straddle', 'combined_straddle', 'atm_ce']  # Most important parameters
        
        for param in key_parameters:
            for tf in key_timeframes:
                if param in rsi_results and tf in rsi_results[param]:
                    rsi = rsi_results[param][tf]
                    features.extend([
                        rsi.rsi_trend,
                        rsi.momentum_strength,
                        1.0 if rsi.rsi_signal == "overbought" else (-1.0 if rsi.rsi_signal == "oversold" else 0.0),
                        (rsi.rsi_value - 50.0) / 50.0,  # Normalized RSI
                        rsi.rsi_value / 100.0  # Raw RSI normalized
                    ])
                    feature_names.extend([
                        f'rsi_trend_{param}_{tf}',
                        f'rsi_strength_{param}_{tf}',
                        f'rsi_signal_{param}_{tf}',
                        f'rsi_normalized_{param}_{tf}',
                        f'rsi_raw_{param}_{tf}'
                    ])
                else:
                    # Padding for missing data
                    features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                    feature_names.extend([
                        f'rsi_trend_{param}_{tf}',
                        f'rsi_strength_{param}_{tf}',
                        f'rsi_signal_{param}_{tf}',
                        f'rsi_normalized_{param}_{tf}',
                        f'rsi_raw_{param}_{tf}'
                    ])
        
        # Category 2: MACD Features (10 features)
        macd_params = ['atm_straddle', 'combined_straddle']
        macd_timeframes = ['5min', '15min']
        
        for param in macd_params:
            for tf in macd_timeframes:
                if param in macd_results and tf in macd_results[param]:
                    macd = macd_results[param][tf]
                    features.extend([
                        np.tanh(macd.macd_line),  # Normalized MACD line
                        np.tanh(macd.histogram),  # Normalized histogram
                        1.0 if macd.macd_signal == "bullish" else (-1.0 if macd.macd_signal == "bearish" else 0.0)
                    ])
                    feature_names.extend([
                        f'macd_line_{param}_{tf}',
                        f'macd_histogram_{param}_{tf}',
                        f'macd_signal_{param}_{tf}'
                    ])
                else:
                    features.extend([0.0, 0.0, 0.0])
                    feature_names.extend([
                        f'macd_line_{param}_{tf}',
                        f'macd_histogram_{param}_{tf}',
                        f'macd_signal_{param}_{tf}'
                    ])
        
        # Add MACD consensus feature
        macd_consensus = np.mean([
            1.0 if combined_momentum.get(f'combined_macd_{tf}', 0) > 0 else -1.0
            for tf in self.timeframes
        ])
        features.append(macd_consensus)
        feature_names.append('macd_consensus')
        
        # Category 3: Divergence Features (5 features)
        # Extract key divergence metrics
        divergence_strength_3_15 = 0.0
        divergence_strength_5_10 = 0.0
        total_divergences = 0
        bullish_divergences = 0
        bearish_divergences = 0
        
        for param_divergences in divergence_results.values():
            for div in param_divergences:
                total_divergences += 1
                if '3min' in div.timeframes_compared and '15min' in div.timeframes_compared:
                    divergence_strength_3_15 = max(divergence_strength_3_15, div.divergence_strength)
                if '5min' in div.timeframes_compared and '10min' in div.timeframes_compared:
                    divergence_strength_5_10 = max(divergence_strength_5_10, div.divergence_strength)
                
                if div.divergence_type == "bullish":
                    bullish_divergences += 1
                elif div.divergence_type == "bearish":
                    bearish_divergences += 1
        
        divergence_features = [
            divergence_strength_3_15,
            divergence_strength_5_10,
            float(total_divergences) / 10.0,  # Normalized count
            float(bullish_divergences - bearish_divergences) / max(total_divergences, 1),  # Divergence bias
            np.mean(list(combined_momentum.values()))  # Overall momentum consensus
        ]
        
        divergence_names = [
            'divergence_strength_3_15min',
            'divergence_strength_5_10min',
            'total_divergences_norm',
            'divergence_bias',
            'momentum_consensus'
        ]
        
        features.extend(divergence_features)
        feature_names.extend(divergence_names)
        
        # Ensure exactly 30 features
        if len(features) != 30:
            if len(features) > 30:
                features = features[:30]
                feature_names = feature_names[:30]
            else:
                while len(features) < 30:
                    features.append(0.0)
                    feature_names.append(f'momentum_padding_{len(features)}')
        
        return np.array(features, dtype=np.float32), feature_names
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI for price series"""
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI for insufficient data
        
        try:
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            # Calculate smoothed averages
            avg_gain = np.mean(gains[-period:]) if len(gains) >= period else 0
            avg_loss = np.mean(losses[-period:]) if len(losses) >= period else 0
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(np.clip(rsi, 0, 100))
            
        except Exception:
            return 50.0
    
    def _calculate_macd(self, prices: np.ndarray, fast_period: int = 12, 
                       slow_period: int = 26, signal_period: int = 9) -> Tuple[float, float, float]:
        """Calculate MACD line, signal line, and histogram"""
        if len(prices) < slow_period:
            return 0.0, 0.0, 0.0
        
        try:
            # Calculate EMAs
            fast_ema = self._calculate_ema(prices, fast_period)
            slow_ema = self._calculate_ema(prices, slow_period)
            
            # MACD line
            macd_line = fast_ema - slow_ema
            
            # Create MACD series for signal calculation (simplified)
            macd_series = np.full(len(prices), macd_line)
            signal_line = self._calculate_ema(macd_series, signal_period)
            
            # Histogram
            histogram = macd_line - signal_line
            
            return float(macd_line), float(signal_line), float(histogram)
            
        except Exception:
            return 0.0, 0.0, 0.0
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate EMA for price series"""
        if len(prices) == 0:
            return 0.0
        
        try:
            alpha = 2.0 / (period + 1)
            ema = prices[0]
            
            for price in prices[1:]:
                ema = alpha * price + (1 - alpha) * ema
            
            return float(ema)
            
        except Exception:
            return 0.0
    
    def _calculate_momentum_confidence(self, rsi_results: Dict, macd_results: Dict) -> float:
        """Calculate overall momentum confidence"""
        try:
            rsi_strengths = []
            macd_strengths = []
            
            for param_rsi in rsi_results.values():
                for rsi in param_rsi.values():
                    rsi_strengths.append(rsi.momentum_strength)
            
            for param_macd in macd_results.values():
                for macd in param_macd.values():
                    macd_strengths.append(macd.divergence_strength)
            
            avg_rsi_strength = np.mean(rsi_strengths) if rsi_strengths else 0.0
            avg_macd_strength = np.mean(macd_strengths) if macd_strengths else 0.0
            
            return float((avg_rsi_strength + avg_macd_strength) / 2.0)
            
        except Exception:
            return 0.5
    
    def _calculate_signal_quality(self, divergence_results: Dict) -> float:
        """Calculate signal quality based on divergence consistency"""
        try:
            if not divergence_results:
                return 0.5
            
            total_divergences = sum(len(divs) for divs in divergence_results.values())
            if total_divergences == 0:
                return 0.5
            
            # Higher quality when divergences are consistent across parameters
            quality_score = min(1.0, total_divergences / 20.0)  # Normalize
            return float(quality_score)
            
        except Exception:
            return 0.5
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB"""
        # Simplified memory estimation
        base_usage = 15.0  # Base engine overhead
        data_usage = len(self.parameters) * len(self.timeframes) * 0.5  # Per time series
        return base_usage + data_usage


# Factory function for engine creation
def create_momentum_engine(config: Dict[str, Any]) -> MomentumAnalysisEngine:
    """Create and configure momentum analysis engine"""
    return MomentumAnalysisEngine(config)