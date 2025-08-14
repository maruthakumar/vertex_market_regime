"""
Multi-Timeframe OI Rollup Module

Performs multi-timeframe analysis of OI patterns with weighted rollups:
- 3min, 5min, 10min, 15min timeframe analysis
- Adaptive timeframe selection based on market conditions
- Timeframe synthesis and agreement scoring
- Primary signal extraction from 5min and 15min windows
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TimeframeSignal(Enum):
    """Types of timeframe signals"""
    STRONG_BULLISH = "strong_bullish"
    MILD_BULLISH = "mild_bullish"
    NEUTRAL = "neutral"
    MILD_BEARISH = "mild_bearish"
    STRONG_BEARISH = "strong_bearish"
    DIVERGENT = "divergent"


@dataclass
class TimeframeMetrics:
    """Metrics for a specific timeframe"""
    timeframe: str
    oi_momentum: float
    volume_momentum: float
    price_correlation: float
    signal_strength: float
    signal_type: TimeframeSignal
    trend_consistency: float
    reversal_probability: float
    timeframe_confidence: float


@dataclass
class MultiTimeframeAnalysis:
    """Combined multi-timeframe analysis results"""
    timeframe_metrics: Dict[str, TimeframeMetrics]
    primary_5min_signal: float
    primary_15min_signal: float
    timeframe_agreement: float
    dominant_timeframe: str
    synthesis_signal: float
    divergence_score: float
    adaptive_weight_distribution: Dict[str, float]
    signal_clarity: float
    regime_stability: float


class MultiTimeframeOIRollup:
    """
    Performs multi-timeframe OI analysis with adaptive weighting
    """
    
    def __init__(self, timeframe_config: Dict[str, Dict] = None):
        """
        Initialize Multi-Timeframe OI Rollup
        
        Args:
            timeframe_config: Configuration for each timeframe
        """
        self.timeframe_config = timeframe_config or {
            '3min': {'weight': 0.15, 'periods': [3, 5, 8], 'window': 3},
            '5min': {'weight': 0.35, 'periods': [3, 5, 8, 10], 'window': 5},
            '10min': {'weight': 0.30, 'periods': [5, 10, 15], 'window': 10},
            '15min': {'weight': 0.20, 'periods': [3, 5, 10, 15], 'window': 15}
        }
        self.signal_history = {tf: [] for tf in self.timeframe_config.keys()}
        logger.info(f"Initialized MultiTimeframeOIRollup with timeframes: {list(self.timeframe_config.keys())}")
    
    def analyze_timeframes(self, df: pd.DataFrame) -> MultiTimeframeAnalysis:
        """
        Perform multi-timeframe analysis on OI data
        
        Args:
            df: Production data with timestamp index
            
        Returns:
            Multi-timeframe analysis results
        """
        timeframe_metrics = {}
        
        # Analyze each timeframe
        for timeframe, config in self.timeframe_config.items():
            metrics = self._analyze_single_timeframe(df, timeframe, config)
            timeframe_metrics[timeframe] = metrics
        
        # Calculate primary signals (5min and 15min focus)
        primary_5min = self._calculate_primary_signal(timeframe_metrics.get('5min'))
        primary_15min = self._calculate_primary_signal(timeframe_metrics.get('15min'))
        
        # Calculate timeframe agreement
        agreement = self._calculate_timeframe_agreement(timeframe_metrics)
        
        # Determine dominant timeframe
        dominant = self._determine_dominant_timeframe(timeframe_metrics)
        
        # Synthesize signals
        synthesis = self._synthesize_signals(timeframe_metrics, primary_5min, primary_15min)
        
        # Calculate divergence
        divergence = self._calculate_timeframe_divergence(timeframe_metrics)
        
        # Adaptive weight distribution
        adaptive_weights = self._calculate_adaptive_weights(timeframe_metrics, agreement)
        
        # Signal clarity
        clarity = self._calculate_signal_clarity(timeframe_metrics)
        
        # Regime stability
        stability = self._calculate_regime_stability(timeframe_metrics)
        
        return MultiTimeframeAnalysis(
            timeframe_metrics=timeframe_metrics,
            primary_5min_signal=primary_5min,
            primary_15min_signal=primary_15min,
            timeframe_agreement=agreement,
            dominant_timeframe=dominant,
            synthesis_signal=synthesis,
            divergence_score=divergence,
            adaptive_weight_distribution=adaptive_weights,
            signal_clarity=clarity,
            regime_stability=stability
        )
    
    def _analyze_single_timeframe(self, df: pd.DataFrame, timeframe: str, 
                                 config: Dict) -> TimeframeMetrics:
        """Analyze a single timeframe"""
        
        # Resample data to timeframe
        resampled_df = self._resample_to_timeframe(df, config['window'])
        
        # Calculate OI momentum
        oi_momentum = self._calculate_oi_momentum(resampled_df, config['periods'])
        
        # Calculate volume momentum
        volume_momentum = self._calculate_volume_momentum(resampled_df, config['periods'])
        
        # Price correlation
        price_correlation = self._calculate_price_correlation(resampled_df)
        
        # Signal strength
        signal_strength = self._calculate_signal_strength(oi_momentum, volume_momentum, price_correlation)
        
        # Classify signal type
        signal_type = self._classify_signal_type(oi_momentum, volume_momentum, price_correlation)
        
        # Trend consistency
        trend_consistency = self._calculate_trend_consistency(resampled_df, config['periods'])
        
        # Reversal probability
        reversal_prob = self._calculate_reversal_probability(oi_momentum, trend_consistency)
        
        # Timeframe confidence
        confidence = self._calculate_timeframe_confidence(signal_strength, trend_consistency)
        
        # Store in history
        self.signal_history[timeframe].append(signal_strength)
        if len(self.signal_history[timeframe]) > 100:
            self.signal_history[timeframe] = self.signal_history[timeframe][-100:]
        
        return TimeframeMetrics(
            timeframe=timeframe,
            oi_momentum=oi_momentum,
            volume_momentum=volume_momentum,
            price_correlation=price_correlation,
            signal_strength=signal_strength,
            signal_type=signal_type,
            trend_consistency=trend_consistency,
            reversal_probability=reversal_prob,
            timeframe_confidence=confidence
        )
    
    def _resample_to_timeframe(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Resample data to specified timeframe"""
        
        # Create a copy to avoid modifying original
        resampled = df.copy()
        
        # If we have a datetime index, use it for resampling
        if 'trade_time' in df.columns:
            try:
                resampled['trade_time'] = pd.to_datetime(resampled['trade_time'])
                resampled = resampled.set_index('trade_time')
                
                # Resample to specified minutes
                rule = f'{window}T'  # T for minutes
                resampled = resampled.resample(rule).agg({
                    'ce_oi': 'last',
                    'pe_oi': 'last',
                    'ce_volume': 'sum',
                    'pe_volume': 'sum',
                    'spot': 'last',
                    'ce_close': 'last',
                    'pe_close': 'last'
                }).dropna()
            except Exception as e:
                logger.warning(f"Datetime resampling failed: {e}, using rolling window")
                # Fallback to rolling window
                return self._rolling_resample(df, window)
        else:
            # Use rolling window approach
            return self._rolling_resample(df, window)
        
        return resampled
    
    def _rolling_resample(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Fallback rolling window resampling"""
        
        resampled_data = {}
        
        # Define aggregation rules
        agg_rules = {
            'ce_oi': 'last',
            'pe_oi': 'last',
            'ce_volume': 'sum',
            'pe_volume': 'sum',
            'spot': 'last',
            'ce_close': 'last',
            'pe_close': 'last'
        }
        
        for col, rule in agg_rules.items():
            if col in df.columns:
                if rule == 'last':
                    resampled_data[col] = df[col].rolling(window).apply(lambda x: x.iloc[-1] if len(x) > 0 else np.nan)
                elif rule == 'sum':
                    resampled_data[col] = df[col].rolling(window).sum()
                elif rule == 'mean':
                    resampled_data[col] = df[col].rolling(window).mean()
        
        return pd.DataFrame(resampled_data).dropna()
    
    def _calculate_oi_momentum(self, df: pd.DataFrame, periods: List[int]) -> float:
        """Calculate OI momentum for the timeframe"""
        
        if 'ce_oi' not in df.columns or 'pe_oi' not in df.columns:
            return 0.0
        
        total_oi = df['ce_oi'] + df['pe_oi']
        
        momentum_scores = []
        for period in periods:
            if len(total_oi) > period:
                momentum = total_oi.diff(period).mean() / (total_oi.mean() + 1)
                momentum_scores.append(momentum)
        
        if momentum_scores:
            # Weighted average (shorter periods get higher weight)
            weights = [1/i for i in range(1, len(momentum_scores) + 1)]
            weights = np.array(weights) / sum(weights)
            return np.average(momentum_scores, weights=weights)
        
        return 0.0
    
    def _calculate_volume_momentum(self, df: pd.DataFrame, periods: List[int]) -> float:
        """Calculate volume momentum for the timeframe"""
        
        if 'ce_volume' not in df.columns or 'pe_volume' not in df.columns:
            return 0.0
        
        total_volume = df['ce_volume'] + df['pe_volume']
        
        momentum_scores = []
        for period in periods:
            if len(total_volume) > period:
                momentum = total_volume.diff(period).mean() / (total_volume.mean() + 1)
                momentum_scores.append(momentum)
        
        if momentum_scores:
            weights = [1/i for i in range(1, len(momentum_scores) + 1)]
            weights = np.array(weights) / sum(weights)
            return np.average(momentum_scores, weights=weights)
        
        return 0.0
    
    def _calculate_price_correlation(self, df: pd.DataFrame) -> float:
        """Calculate OI-price correlation for the timeframe"""
        
        if 'ce_oi' not in df.columns or 'spot' not in df.columns:
            return 0.0
        
        # Calculate correlation between OI changes and price changes
        oi_changes = (df['ce_oi'] + df.get('pe_oi', 0)).pct_change()
        price_changes = df['spot'].pct_change()
        
        # Remove NaN values
        valid_mask = ~(oi_changes.isna() | price_changes.isna())
        
        if valid_mask.sum() > 2:
            correlation = oi_changes[valid_mask].corr(price_changes[valid_mask])
            return correlation if not np.isnan(correlation) else 0.0
        
        return 0.0
    
    def _calculate_signal_strength(self, oi_mom: float, vol_mom: float, 
                                  price_corr: float) -> float:
        """Calculate overall signal strength"""
        
        # Combine momentum and correlation
        momentum_strength = (abs(oi_mom) + abs(vol_mom)) / 2
        
        # Correlation adds to strength if aligned
        if np.sign(oi_mom) == np.sign(price_corr):
            correlation_boost = abs(price_corr) * 0.3
        else:
            correlation_boost = -abs(price_corr) * 0.2
        
        signal_strength = momentum_strength + correlation_boost
        
        # Normalize to [-1, 1]
        return np.clip(signal_strength * np.sign(oi_mom), -1.0, 1.0)
    
    def _classify_signal_type(self, oi_mom: float, vol_mom: float, 
                             price_corr: float) -> TimeframeSignal:
        """Classify the signal type"""
        
        # Calculate composite score
        composite = (oi_mom * 0.5 + vol_mom * 0.3 + price_corr * 0.2)
        
        # Check for divergence
        if np.sign(oi_mom) != np.sign(vol_mom) and abs(oi_mom - vol_mom) > 0.3:
            return TimeframeSignal.DIVERGENT
        
        # Classify based on strength
        if composite > 0.5:
            return TimeframeSignal.STRONG_BULLISH
        elif composite > 0.2:
            return TimeframeSignal.MILD_BULLISH
        elif composite < -0.5:
            return TimeframeSignal.STRONG_BEARISH
        elif composite < -0.2:
            return TimeframeSignal.MILD_BEARISH
        else:
            return TimeframeSignal.NEUTRAL
    
    def _calculate_trend_consistency(self, df: pd.DataFrame, periods: List[int]) -> float:
        """Calculate trend consistency across periods"""
        
        if 'ce_oi' not in df.columns:
            return 0.0
        
        total_oi = df['ce_oi'] + df.get('pe_oi', 0)
        
        # Check trend direction across different periods
        trends = []
        for period in periods:
            if len(total_oi) > period:
                trend = np.sign(total_oi.diff(period).mean())
                trends.append(trend)
        
        if trends:
            # Consistency is high when all trends align
            consistency = abs(np.mean(trends))
            return consistency
        
        return 0.0
    
    def _calculate_reversal_probability(self, momentum: float, consistency: float) -> float:
        """Calculate probability of trend reversal"""
        
        reversal_prob = 0.0
        
        # High momentum with low consistency suggests reversal
        if abs(momentum) > 0.5 and consistency < 0.3:
            reversal_prob += 0.4
        
        # Extreme momentum often reverses
        if abs(momentum) > 0.8:
            reversal_prob += 0.3
        
        # Very high consistency reduces reversal probability
        if consistency > 0.8:
            reversal_prob *= 0.5
        
        return np.clip(reversal_prob, 0.0, 1.0)
    
    def _calculate_timeframe_confidence(self, signal_strength: float, 
                                       consistency: float) -> float:
        """Calculate confidence in timeframe signal"""
        
        # Strong signals with high consistency = high confidence
        confidence = abs(signal_strength) * 0.6 + consistency * 0.4
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _calculate_primary_signal(self, metrics: Optional[TimeframeMetrics]) -> float:
        """Calculate primary signal from timeframe metrics"""
        
        if not metrics:
            return 0.0
        
        # Primary signal combines momentum and strength
        primary = metrics.signal_strength * 0.6 + metrics.oi_momentum * 0.4
        
        # Adjust for confidence
        primary *= metrics.timeframe_confidence
        
        return primary
    
    def _calculate_timeframe_agreement(self, metrics: Dict[str, TimeframeMetrics]) -> float:
        """Calculate agreement across timeframes"""
        
        if not metrics:
            return 0.0
        
        # Extract signal types
        signals = [m.signal_type for m in metrics.values()]
        
        # Count bullish, bearish, neutral
        bullish_count = sum(1 for s in signals if 'bullish' in s.value.lower())
        bearish_count = sum(1 for s in signals if 'bearish' in s.value.lower())
        neutral_count = sum(1 for s in signals if s == TimeframeSignal.NEUTRAL)
        divergent_count = sum(1 for s in signals if s == TimeframeSignal.DIVERGENT)
        
        total = len(signals)
        
        if total == 0:
            return 0.0
        
        # High agreement when most signals align
        max_count = max(bullish_count, bearish_count, neutral_count)
        agreement = max_count / total
        
        # Penalize divergence
        agreement -= divergent_count / total * 0.5
        
        return np.clip(agreement, 0.0, 1.0)
    
    def _determine_dominant_timeframe(self, metrics: Dict[str, TimeframeMetrics]) -> str:
        """Determine which timeframe has the strongest signal"""
        
        if not metrics:
            return "none"
        
        # Find timeframe with highest confidence-weighted signal strength
        max_score = 0.0
        dominant = "none"
        
        for tf, m in metrics.items():
            score = abs(m.signal_strength) * m.timeframe_confidence
            if score > max_score:
                max_score = score
                dominant = tf
        
        return dominant
    
    def _synthesize_signals(self, metrics: Dict[str, TimeframeMetrics],
                           primary_5min: float, primary_15min: float) -> float:
        """Synthesize signals across all timeframes"""
        
        if not metrics:
            return 0.0
        
        # Weighted combination with emphasis on 5min and 15min
        synthesis = 0.0
        
        # Primary signals (5min and 15min)
        synthesis += primary_5min * 0.35
        synthesis += primary_15min * 0.20
        
        # Add other timeframes with their configured weights
        for tf, config in self.timeframe_config.items():
            if tf in metrics and tf not in ['5min', '15min']:
                synthesis += metrics[tf].signal_strength * config['weight']
        
        return np.clip(synthesis, -1.0, 1.0)
    
    def _calculate_timeframe_divergence(self, metrics: Dict[str, TimeframeMetrics]) -> float:
        """Calculate divergence between timeframes"""
        
        if not metrics or len(metrics) < 2:
            return 0.0
        
        # Calculate pairwise divergences
        signals = [m.signal_strength for m in metrics.values()]
        
        # Standard deviation of signals indicates divergence
        divergence = np.std(signals)
        
        # Normalize to [0, 1]
        return min(divergence * 2, 1.0)
    
    def _calculate_adaptive_weights(self, metrics: Dict[str, TimeframeMetrics],
                                   agreement: float) -> Dict[str, float]:
        """Calculate adaptive weights based on performance"""
        
        weights = {}
        
        # Base weights from config
        for tf, config in self.timeframe_config.items():
            weights[tf] = config['weight']
        
        # Adjust based on confidence and agreement
        if agreement > 0.7:
            # High agreement: maintain base weights
            pass
        else:
            # Low agreement: weight towards most confident timeframe
            for tf in weights:
                if tf in metrics:
                    # Increase weight for high confidence timeframes
                    confidence_factor = metrics[tf].timeframe_confidence
                    weights[tf] *= (0.5 + 0.5 * confidence_factor)
        
        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def _calculate_signal_clarity(self, metrics: Dict[str, TimeframeMetrics]) -> float:
        """Calculate clarity of the overall signal"""
        
        if not metrics:
            return 0.0
        
        # Clarity is high when signals are strong and aligned
        strengths = [abs(m.signal_strength) for m in metrics.values()]
        avg_strength = np.mean(strengths)
        
        # Check alignment
        signs = [np.sign(m.signal_strength) for m in metrics.values()]
        alignment = abs(np.mean(signs))
        
        clarity = avg_strength * 0.6 + alignment * 0.4
        
        return np.clip(clarity, 0.0, 1.0)
    
    def _calculate_regime_stability(self, metrics: Dict[str, TimeframeMetrics]) -> float:
        """Calculate regime stability across timeframes"""
        
        stability_score = 0.0
        
        # Check consistency within each timeframe
        for tf, m in metrics.items():
            stability_score += m.trend_consistency * (1 - m.reversal_probability)
        
        if metrics:
            stability_score /= len(metrics)
        
        return np.clip(stability_score, 0.0, 1.0)
    
    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract multi-timeframe features for Component 3
        
        Args:
            df: Production data
            
        Returns:
            Array of timeframe features
        """
        analysis = self.analyze_timeframes(df)
        
        features = [
            analysis.primary_5min_signal,
            analysis.primary_15min_signal,
            analysis.timeframe_agreement,
            analysis.synthesis_signal,
            analysis.divergence_score,
            analysis.signal_clarity,
            analysis.regime_stability
        ]
        
        # Add dominant timeframe encoding
        dominant_encoding = {
            '3min': 0.25,
            '5min': 0.50,
            '10min': 0.75,
            '15min': 1.00,
            'none': 0.0
        }
        features.append(dominant_encoding.get(analysis.dominant_timeframe, 0.0))
        
        # Add adaptive weights
        for tf in ['3min', '5min', '10min', '15min']:
            features.append(analysis.adaptive_weight_distribution.get(tf, 0.0))
        
        # Add individual timeframe signals
        for tf in ['3min', '5min', '10min', '15min']:
            if tf in analysis.timeframe_metrics:
                features.append(analysis.timeframe_metrics[tf].signal_strength)
            else:
                features.append(0.0)
        
        return np.array(features)