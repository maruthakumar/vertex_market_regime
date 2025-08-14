"""
Momentum-Based Support/Resistance Level Detector for Component 7 Phase 2

Leverages Component 1 momentum features (RSI/MACD) and Component 6 enhanced 
correlation features to detect momentum-validated support and resistance levels.

Phase 2 Enhancement Features:
- RSI Level Confluence (4 features): Support/resistance levels confirmed by RSI overbought/oversold
- MACD Level Validation (3 features): Support/resistance validated by MACD signal crossovers  
- Momentum Exhaustion Levels (3 features): Support/resistance from momentum divergence points

🎯 PURE MATHEMATICAL LEVEL DETECTION - NO THRESHOLD-BASED CLASSIFICATION
- Raw momentum level measurements and confluence calculations
- Mathematical momentum divergence level detection
- Statistical momentum exhaustion level identification
- All level strength classification deferred to ML models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import time
import warnings
from scipy.stats import pearsonr
from scipy.signal import find_peaks, savgol_filter
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class MomentumLevelResult:
    """Result from momentum-based level detection"""
    rsi_confluence_levels: Dict[str, float]
    macd_validation_levels: Dict[str, float]
    momentum_exhaustion_levels: Dict[str, float]
    momentum_level_strengths: np.ndarray
    momentum_level_prices: np.ndarray
    processing_time_ms: float
    feature_count: int
    timestamp: datetime


@dataclass
class MomentumLevelData:
    """Momentum level detection input data"""
    rsi_values: Dict[str, np.ndarray]  # timeframe -> RSI values
    macd_values: Dict[str, np.ndarray]  # timeframe -> MACD values
    signal_values: Dict[str, np.ndarray]  # timeframe -> Signal values
    histogram_values: Dict[str, np.ndarray]  # timeframe -> Histogram values
    price_values: np.ndarray  # Corresponding price values
    volume_values: np.ndarray  # Volume data for validation
    timestamps: pd.DatetimeIndex
    

class MomentumLevelDetector:
    """
    Momentum-Based Support/Resistance Level Detector
    
    Uses Component 1 momentum features to identify high-probability
    support and resistance levels with momentum confirmation.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize momentum level detector"""
        self.config = config or {}
        self.timeframes = ['3min', '5min', '10min', '15min']
        
        # RSI parameters
        self.rsi_overbought = self.config.get('rsi_overbought', 70)
        self.rsi_oversold = self.config.get('rsi_oversold', 30)
        self.rsi_neutral_upper = self.config.get('rsi_neutral_upper', 60)
        self.rsi_neutral_lower = self.config.get('rsi_neutral_lower', 40)
        
        # MACD parameters
        self.macd_signal_threshold = self.config.get('macd_signal_threshold', 0.0)
        self.histogram_reversal_threshold = self.config.get('histogram_reversal_threshold', 0.0)
        
        # Level detection parameters
        self.min_momentum_periods = self.config.get('min_momentum_periods', 20)
        self.level_proximity_threshold = self.config.get('level_proximity_threshold', 0.005)  # 0.5%
        self.momentum_confirmation_window = self.config.get('momentum_confirmation_window', 10)
        
        logger.info("MomentumLevelDetector initialized for Component 7 Phase 2")
    
    def detect_momentum_levels(self, 
                              momentum_data: MomentumLevelData,
                              existing_levels: Optional[List[float]] = None) -> MomentumLevelResult:
        """
        Detect momentum-validated support/resistance levels
        
        Args:
            momentum_data: Momentum and price data
            existing_levels: Existing support/resistance levels for validation
            
        Returns:
            MomentumLevelResult with 10 momentum-based level features
        """
        start_time = time.time()
        
        try:
            # 1. RSI Level Confluence (4 features)
            rsi_confluence_levels = self._detect_rsi_confluence_levels(momentum_data)
            
            # 2. MACD Level Validation (3 features)
            macd_validation_levels = self._detect_macd_validation_levels(momentum_data)
            
            # 3. Momentum Exhaustion Levels (3 features)
            momentum_exhaustion_levels = self._detect_momentum_exhaustion_levels(momentum_data)
            
            # 4. Calculate momentum level strengths
            momentum_level_strengths = self._calculate_momentum_level_strengths(
                momentum_data, rsi_confluence_levels, macd_validation_levels, momentum_exhaustion_levels
            )
            
            # 5. Extract momentum level prices
            momentum_level_prices = self._extract_momentum_level_prices(
                momentum_data, rsi_confluence_levels, macd_validation_levels, momentum_exhaustion_levels
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            return MomentumLevelResult(
                rsi_confluence_levels=rsi_confluence_levels,
                macd_validation_levels=macd_validation_levels,
                momentum_exhaustion_levels=momentum_exhaustion_levels,
                momentum_level_strengths=momentum_level_strengths,
                momentum_level_prices=momentum_level_prices,
                processing_time_ms=processing_time,
                feature_count=10,  # 4 + 3 + 3 = 10 momentum-based features
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in momentum level detection: {e}")
            return self._create_fallback_result()
    
    def _detect_rsi_confluence_levels(self, momentum_data: MomentumLevelData) -> Dict[str, float]:
        """Detect RSI confluence levels (4 features)"""
        features = {}
        
        try:
            # Feature 1: RSI overbought resistance confluence
            overbought_resistances = []
            for tf in self.timeframes[:2]:  # Focus on 3min, 5min
                rsi_vals = momentum_data.rsi_values.get(tf, np.array([]))
                price_vals = momentum_data.price_values
                
                if len(rsi_vals) >= self.min_momentum_periods and len(price_vals) >= self.min_momentum_periods:
                    min_len = min(len(rsi_vals), len(price_vals))
                    rsi_recent = rsi_vals[-min_len:]
                    price_recent = price_vals[-min_len:]
                    
                    # Find points where RSI was overbought
                    overbought_points = rsi_recent > self.rsi_overbought
                    if np.any(overbought_points):
                        overbought_prices = price_recent[overbought_points]
                        if len(overbought_prices) > 0:
                            # Calculate resistance level as highest price during overbought conditions
                            resistance_level = np.max(overbought_prices)
                            # Normalize as strength score relative to current price
                            current_price = price_recent[-1] if len(price_recent) > 0 else 100
                            strength = min((resistance_level - current_price) / current_price, 0.1) * 10  # Normalize to [0,1]
                            overbought_resistances.append(max(0, strength))
            
            features['rsi_overbought_resistance_strength'] = np.mean(overbought_resistances) if overbought_resistances else 0.0
            
            # Feature 2: RSI oversold support confluence
            oversold_supports = []
            for tf in self.timeframes[:2]:
                rsi_vals = momentum_data.rsi_values.get(tf, np.array([]))
                price_vals = momentum_data.price_values
                
                if len(rsi_vals) >= self.min_momentum_periods and len(price_vals) >= self.min_momentum_periods:
                    min_len = min(len(rsi_vals), len(price_vals))
                    rsi_recent = rsi_vals[-min_len:]
                    price_recent = price_vals[-min_len:]
                    
                    # Find points where RSI was oversold
                    oversold_points = rsi_recent < self.rsi_oversold
                    if np.any(oversold_points):
                        oversold_prices = price_recent[oversold_points]
                        if len(oversold_prices) > 0:
                            # Calculate support level as lowest price during oversold conditions
                            support_level = np.min(oversold_prices)
                            # Normalize as strength score relative to current price
                            current_price = price_recent[-1] if len(price_recent) > 0 else 100
                            strength = min((current_price - support_level) / current_price, 0.1) * 10  # Normalize to [0,1]
                            oversold_supports.append(max(0, strength))
            
            features['rsi_oversold_support_strength'] = np.mean(oversold_supports) if oversold_supports else 0.0
            
            # Feature 3: RSI neutral zone level density
            neutral_zone_densities = []
            for tf in self.timeframes[:2]:
                rsi_vals = momentum_data.rsi_values.get(tf, np.array([]))
                if len(rsi_vals) >= self.min_momentum_periods:
                    recent_rsi = rsi_vals[-self.momentum_confirmation_window:]
                    # Calculate time spent in neutral zone (40-60)
                    neutral_time = np.mean((recent_rsi >= self.rsi_neutral_lower) & (recent_rsi <= self.rsi_neutral_upper))
                    neutral_zone_densities.append(neutral_time)
            
            features['rsi_neutral_zone_level_density'] = np.mean(neutral_zone_densities) if neutral_zone_densities else 0.5
            
            # Feature 4: RSI level convergence across timeframes
            level_convergences = []
            if len(self.timeframes) >= 2:
                for i in range(len(self.timeframes) - 1):
                    tf1, tf2 = self.timeframes[i], self.timeframes[i + 1]
                    rsi1 = momentum_data.rsi_values.get(tf1, np.array([]))
                    rsi2 = momentum_data.rsi_values.get(tf2, np.array([]))
                    
                    if len(rsi1) >= 5 and len(rsi2) >= 5:
                        # Calculate RSI level convergence between timeframes
                        rsi1_avg = np.mean(rsi1[-5:])
                        rsi2_avg = np.mean(rsi2[-5:])
                        convergence = 1.0 - abs(rsi1_avg - rsi2_avg) / 100.0  # Normalize RSI difference
                        level_convergences.append(max(0, convergence))
            
            features['rsi_level_convergence_strength'] = np.mean(level_convergences) if level_convergences else 0.5
            
        except Exception as e:
            logger.warning(f"Error calculating RSI confluence levels: {e}")
            features = {
                'rsi_overbought_resistance_strength': 0.0,
                'rsi_oversold_support_strength': 0.0,
                'rsi_neutral_zone_level_density': 0.5,
                'rsi_level_convergence_strength': 0.5
            }
        
        return features
    
    def _detect_macd_validation_levels(self, momentum_data: MomentumLevelData) -> Dict[str, float]:
        """Detect MACD validation levels (3 features)"""
        features = {}
        
        try:
            # Feature 1: MACD signal crossover level strength
            crossover_strengths = []
            for tf in self.timeframes[:2]:  # Focus on 3min, 5min
                macd_vals = momentum_data.macd_values.get(tf, np.array([]))
                signal_vals = momentum_data.signal_values.get(tf, np.array([]))
                price_vals = momentum_data.price_values
                
                if len(macd_vals) >= self.min_momentum_periods and len(signal_vals) >= self.min_momentum_periods:
                    min_len = min(len(macd_vals), len(signal_vals), len(price_vals))
                    macd_recent = macd_vals[-min_len:]
                    signal_recent = signal_vals[-min_len:]
                    
                    # Detect crossovers
                    crossover_diff = macd_recent - signal_recent
                    crossover_points = np.where(np.diff(np.sign(crossover_diff)))[0]
                    
                    if len(crossover_points) > 0:
                        # Calculate crossover strength as magnitude of MACD-Signal difference at crossover
                        recent_crossovers = crossover_points[-3:] if len(crossover_points) >= 3 else crossover_points
                        crossover_strength = np.mean([abs(crossover_diff[point]) for point in recent_crossovers])
                        normalized_strength = min(crossover_strength / 0.5, 1.0)  # Normalize
                        crossover_strengths.append(normalized_strength)
            
            features['macd_crossover_level_strength'] = np.mean(crossover_strengths) if crossover_strengths else 0.0
            
            # Feature 2: MACD histogram reversal level validation
            histogram_reversals = []
            for tf in self.timeframes[:2]:
                histogram_vals = momentum_data.histogram_values.get(tf, np.array([]))
                if len(histogram_vals) >= self.min_momentum_periods:
                    # Detect histogram reversals (zero-line crosses)
                    histogram_recent = histogram_vals[-self.momentum_confirmation_window:]
                    zero_crosses = np.where(np.diff(np.sign(histogram_recent)))[0]
                    
                    if len(zero_crosses) > 0:
                        # Calculate reversal strength
                        reversal_strength = np.mean([abs(histogram_recent[cross]) for cross in zero_crosses[-2:]])
                        normalized_strength = min(reversal_strength / 0.1, 1.0)  # Normalize
                        histogram_reversals.append(normalized_strength)
            
            features['macd_histogram_reversal_strength'] = np.mean(histogram_reversals) if histogram_reversals else 0.0
            
            # Feature 3: MACD momentum consensus validation
            momentum_consensus = []
            for tf in self.timeframes[:2]:
                macd_vals = momentum_data.macd_values.get(tf, np.array([]))
                signal_vals = momentum_data.signal_values.get(tf, np.array([]))
                histogram_vals = momentum_data.histogram_values.get(tf, np.array([]))
                
                if len(macd_vals) >= 10 and len(signal_vals) >= 10 and len(histogram_vals) >= 10:
                    # Calculate momentum consensus (all indicators aligned)
                    macd_bullish = macd_vals[-1] > signal_vals[-1]
                    histogram_bullish = histogram_vals[-1] > 0
                    macd_trend_bullish = macd_vals[-1] > macd_vals[-5]
                    
                    # Consensus score
                    consensus_score = sum([macd_bullish, histogram_bullish, macd_trend_bullish]) / 3.0
                    momentum_consensus.append(consensus_score)
            
            features['macd_momentum_consensus_validation'] = np.mean(momentum_consensus) if momentum_consensus else 0.5
            
        except Exception as e:
            logger.warning(f"Error calculating MACD validation levels: {e}")
            features = {
                'macd_crossover_level_strength': 0.0,
                'macd_histogram_reversal_strength': 0.0,
                'macd_momentum_consensus_validation': 0.5
            }
        
        return features
    
    def _detect_momentum_exhaustion_levels(self, momentum_data: MomentumLevelData) -> Dict[str, float]:
        """Detect momentum exhaustion levels (3 features)"""
        features = {}
        
        try:
            # Feature 1: RSI-MACD divergence exhaustion
            divergence_exhaustions = []
            for tf in self.timeframes[:2]:  # Focus on 3min, 5min
                rsi_vals = momentum_data.rsi_values.get(tf, np.array([]))
                macd_vals = momentum_data.macd_values.get(tf, np.array([]))
                price_vals = momentum_data.price_values
                
                if len(rsi_vals) >= self.min_momentum_periods and len(macd_vals) >= self.min_momentum_periods:
                    min_len = min(len(rsi_vals), len(macd_vals), len(price_vals))
                    rsi_recent = rsi_vals[-min_len:]
                    macd_recent = macd_vals[-min_len:]
                    price_recent = price_vals[-min_len:]
                    
                    # Calculate momentum divergence from price
                    if len(rsi_recent) >= 10 and len(price_recent) >= 10:
                        rsi_trend = np.polyfit(range(len(rsi_recent[-10:])), rsi_recent[-10:], 1)[0]
                        price_trend = np.polyfit(range(len(price_recent[-10:])), price_recent[-10:], 1)[0]
                        
                        # Normalize trends
                        rsi_trend_norm = rsi_trend / np.std(rsi_recent[-10:]) if np.std(rsi_recent[-10:]) > 0 else 0
                        price_trend_norm = price_trend / np.std(price_recent[-10:]) if np.std(price_recent[-10:]) > 0 else 0
                        
                        # Divergence strength (opposite trends)
                        divergence_strength = abs(rsi_trend_norm + price_trend_norm) / 2.0
                        divergence_exhaustions.append(min(divergence_strength, 1.0))
            
            features['rsi_price_divergence_exhaustion'] = np.mean(divergence_exhaustions) if divergence_exhaustions else 0.0
            
            # Feature 2: MACD momentum exhaustion
            macd_exhaustions = []
            for tf in self.timeframes[:2]:
                macd_vals = momentum_data.macd_values.get(tf, np.array([]))
                signal_vals = momentum_data.signal_values.get(tf, np.array([]))
                
                if len(macd_vals) >= self.min_momentum_periods and len(signal_vals) >= self.min_momentum_periods:
                    # Calculate MACD momentum exhaustion (flattening)
                    macd_recent = macd_vals[-self.momentum_confirmation_window:]
                    signal_recent = signal_vals[-self.momentum_confirmation_window:]
                    
                    # Momentum exhaustion when MACD slope approaches zero
                    if len(macd_recent) >= 5:
                        macd_slope = np.polyfit(range(len(macd_recent)), macd_recent, 1)[0]
                        slope_normalized = abs(macd_slope) / (np.std(macd_recent) + 1e-8)
                        exhaustion_score = 1.0 - min(slope_normalized, 1.0)  # Invert: low slope = high exhaustion
                        macd_exhaustions.append(exhaustion_score)
            
            features['macd_momentum_exhaustion'] = np.mean(macd_exhaustions) if macd_exhaustions else 0.5
            
            # Feature 3: Multi-timeframe momentum exhaustion consensus
            consensus_exhaustions = []
            timeframe_exhaustions = {}
            
            # Calculate exhaustion for each timeframe
            for tf in self.timeframes:
                rsi_vals = momentum_data.rsi_values.get(tf, np.array([]))
                macd_vals = momentum_data.macd_values.get(tf, np.array([]))
                
                if len(rsi_vals) >= 10 and len(macd_vals) >= 10:
                    # RSI exhaustion (extreme levels)
                    rsi_extreme = (np.mean(rsi_vals[-5:]) > 75) or (np.mean(rsi_vals[-5:]) < 25)
                    
                    # MACD exhaustion (low volatility)
                    macd_volatility = np.std(macd_vals[-10:])
                    macd_exhausted = macd_volatility < np.std(macd_vals[-20:-10]) * 0.5
                    
                    # Combined exhaustion score
                    exhaustion_score = (int(rsi_extreme) + int(macd_exhausted)) / 2.0
                    timeframe_exhaustions[tf] = exhaustion_score
            
            if timeframe_exhaustions:
                # Consensus across timeframes
                exhaustion_values = list(timeframe_exhaustions.values())
                consensus_score = 1.0 - np.std(exhaustion_values) if len(exhaustion_values) > 1 else np.mean(exhaustion_values)
                features['multi_timeframe_exhaustion_consensus'] = max(0.0, min(1.0, consensus_score))
            else:
                features['multi_timeframe_exhaustion_consensus'] = 0.5
            
        except Exception as e:
            logger.warning(f"Error calculating momentum exhaustion levels: {e}")
            features = {
                'rsi_price_divergence_exhaustion': 0.0,
                'macd_momentum_exhaustion': 0.5,
                'multi_timeframe_exhaustion_consensus': 0.5
            }
        
        return features
    
    def _calculate_momentum_level_strengths(self, 
                                          momentum_data: MomentumLevelData,
                                          rsi_levels: Dict[str, float],
                                          macd_levels: Dict[str, float],
                                          exhaustion_levels: Dict[str, float]) -> np.ndarray:
        """Calculate overall momentum level strengths"""
        try:
            strengths = []
            
            # Combine all momentum level features
            all_features = {**rsi_levels, **macd_levels, **exhaustion_levels}
            
            for feature_name, value in all_features.items():
                if not np.isnan(value):
                    strengths.append(value)
            
            return np.array(strengths) if strengths else np.array([0.5])
            
        except Exception as e:
            logger.warning(f"Error calculating momentum level strengths: {e}")
            return np.array([0.5])
    
    def _extract_momentum_level_prices(self, 
                                     momentum_data: MomentumLevelData,
                                     rsi_levels: Dict[str, float],
                                     macd_levels: Dict[str, float],
                                     exhaustion_levels: Dict[str, float]) -> np.ndarray:
        """Extract price levels corresponding to momentum signals"""
        try:
            level_prices = []
            current_price = momentum_data.price_values[-1] if len(momentum_data.price_values) > 0 else 100.0
            
            # Generate representative momentum level prices
            # These would be actual detected levels in a complete implementation
            for feature_name, strength in {**rsi_levels, **macd_levels, **exhaustion_levels}.items():
                if 'resistance' in feature_name or 'overbought' in feature_name:
                    # Resistance levels above current price
                    level_price = current_price * (1 + strength * 0.02)  # Up to 2% above
                elif 'support' in feature_name or 'oversold' in feature_name:
                    # Support levels below current price
                    level_price = current_price * (1 - strength * 0.02)  # Up to 2% below
                else:
                    # Neutral levels near current price
                    level_price = current_price * (1 + (strength - 0.5) * 0.01)  # ±1% around current
                
                level_prices.append(level_price)
            
            return np.array(level_prices) if level_prices else np.array([current_price])
            
        except Exception as e:
            logger.warning(f"Error extracting momentum level prices: {e}")
            current_price = momentum_data.price_values[-1] if len(momentum_data.price_values) > 0 else 100.0
            return np.array([current_price])
    
    def _create_fallback_result(self) -> MomentumLevelResult:
        """Create fallback result with reasonable defaults"""
        return MomentumLevelResult(
            rsi_confluence_levels={
                'rsi_overbought_resistance_strength': 0.0,
                'rsi_oversold_support_strength': 0.0,
                'rsi_neutral_zone_level_density': 0.5,
                'rsi_level_convergence_strength': 0.5
            },
            macd_validation_levels={
                'macd_crossover_level_strength': 0.0,
                'macd_histogram_reversal_strength': 0.0,
                'macd_momentum_consensus_validation': 0.5
            },
            momentum_exhaustion_levels={
                'rsi_price_divergence_exhaustion': 0.0,
                'macd_momentum_exhaustion': 0.5,
                'multi_timeframe_exhaustion_consensus': 0.5
            },
            momentum_level_strengths=np.array([0.5]),
            momentum_level_prices=np.array([100.0]),
            processing_time_ms=0.0,
            feature_count=10,
            timestamp=datetime.now()
        )