"""
Pattern Detector for Strategy Inversion

Detects specific patterns in strategy performance that indicate
optimal inversion opportunities using advanced signal processing.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import signal, stats
from scipy.ndimage import uniform_filter1d
import warnings

logger = logging.getLogger(__name__)

@dataclass
class DetectedPattern:
    """Detected pattern in strategy data"""
    pattern_name: str
    pattern_type: str
    start_index: int
    end_index: int
    strength: float
    confidence: float
    suggested_action: str
    pattern_metrics: Dict[str, float]

class PatternDetector:
    """
    Advanced pattern detection for strategy inversion opportunities
    
    Uses signal processing and statistical techniques to identify
    patterns that suggest profitable inversion strategies.
    """
    
    def __init__(self,
                 min_pattern_length: int = 10,
                 detection_sensitivity: float = 0.1,
                 confidence_threshold: float = 0.6):
        """
        Initialize pattern detector
        
        Args:
            min_pattern_length: Minimum length for pattern detection
            detection_sensitivity: Sensitivity for pattern detection
            confidence_threshold: Minimum confidence for pattern reporting
        """
        self.min_pattern_length = min_pattern_length
        self.detection_sensitivity = detection_sensitivity
        self.confidence_threshold = confidence_threshold
        
        # Pattern detection parameters
        self.smoothing_window = 5
        self.trend_window = 20
        self.volatility_window = 15
        
        logger.info("PatternDetector initialized")
    
    def detect_all_patterns(self, returns: pd.Series) -> List[DetectedPattern]:
        """
        Detect all patterns in strategy returns
        
        Args:
            returns: Strategy returns series
            
        Returns:
            List of detected patterns
        """
        if len(returns) < self.min_pattern_length * 2:
            logger.warning("Insufficient data for pattern detection")
            return []
        
        patterns = []
        
        # Clean and prepare data
        cleaned_returns = self._clean_returns(returns)
        
        # Detect different pattern types
        patterns.extend(self._detect_persistent_drawdown_patterns(cleaned_returns))
        patterns.extend(self._detect_volatility_spike_patterns(cleaned_returns))
        patterns.extend(self._detect_trend_reversal_patterns(cleaned_returns))
        patterns.extend(self._detect_mean_reversion_patterns(cleaned_returns))
        patterns.extend(self._detect_momentum_patterns(cleaned_returns))
        patterns.extend(self._detect_cyclical_patterns(cleaned_returns))
        patterns.extend(self._detect_regime_shift_patterns(cleaned_returns))
        
        # Filter by confidence threshold
        high_confidence_patterns = [
            p for p in patterns if p.confidence >= self.confidence_threshold
        ]
        
        # Sort by strength
        high_confidence_patterns.sort(key=lambda x: x.strength, reverse=True)
        
        logger.info(f"Detected {len(high_confidence_patterns)} high-confidence patterns")
        
        return high_confidence_patterns
    
    def _detect_persistent_drawdown_patterns(self, returns: pd.Series) -> List[DetectedPattern]:
        """Detect persistent drawdown patterns that suggest inversion"""
        patterns = []
        
        # Calculate cumulative returns and drawdowns
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        # Smooth drawdown series
        smoothed_dd = pd.Series(
            uniform_filter1d(drawdown.values, size=self.smoothing_window),
            index=drawdown.index
        )
        
        # Find periods of persistent drawdown
        persistent_dd_threshold = -0.05  # 5% drawdown threshold
        in_drawdown = smoothed_dd < persistent_dd_threshold
        
        # Group consecutive drawdown periods
        drawdown_periods = self._group_consecutive_periods(in_drawdown)
        
        for start_idx, end_idx in drawdown_periods:
            if end_idx - start_idx >= self.min_pattern_length:
                # Calculate pattern metrics
                period_returns = returns.iloc[start_idx:end_idx+1]
                period_dd = smoothed_dd.iloc[start_idx:end_idx+1]
                
                # Pattern strength based on duration and severity
                duration_score = min(1.0, (end_idx - start_idx) / 60)  # Normalize by ~3 months
                severity_score = min(1.0, abs(period_dd.min()) / 0.3)  # Normalize by 30% max
                strength = 0.6 * severity_score + 0.4 * duration_score
                
                # Confidence based on consistency
                dd_consistency = 1.0 - (period_dd.std() / abs(period_dd.mean() + 1e-8))
                confidence = min(1.0, max(0.0, dd_consistency))
                
                pattern = DetectedPattern(
                    pattern_name=f"persistent_drawdown_{start_idx}_{end_idx}",
                    pattern_type="persistent_drawdown",
                    start_index=start_idx,
                    end_index=end_idx,
                    strength=strength,
                    confidence=confidence,
                    suggested_action="invert_during_period",
                    pattern_metrics={
                        'max_drawdown': abs(period_dd.min()),
                        'avg_drawdown': abs(period_dd.mean()),
                        'duration_days': end_idx - start_idx + 1,
                        'recovery_probability': self._estimate_recovery_probability(period_returns)
                    }
                )
                
                patterns.append(pattern)
        
        return patterns
    
    def _detect_volatility_spike_patterns(self, returns: pd.Series) -> List[DetectedPattern]:
        """Detect volatility spike patterns"""
        patterns = []
        
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=self.volatility_window).std()
        
        # Detect volatility spikes
        vol_threshold = rolling_vol.quantile(0.8)  # Top 20% volatility
        vol_spikes = rolling_vol > vol_threshold
        
        # Group consecutive spike periods
        spike_periods = self._group_consecutive_periods(vol_spikes)
        
        for start_idx, end_idx in spike_periods:
            if end_idx - start_idx >= self.min_pattern_length // 2:  # Shorter minimum for vol spikes
                period_returns = returns.iloc[start_idx:end_idx+1]
                period_vol = rolling_vol.iloc[start_idx:end_idx+1]
                
                # Check if high volatility coincides with negative returns
                negative_return_ratio = (period_returns < 0).mean()
                
                if negative_return_ratio > 0.6:  # 60% of high-vol period is negative
                    # Pattern strength
                    vol_intensity = period_vol.mean() / (rolling_vol.mean() + 1e-8)
                    negative_intensity = abs(period_returns[period_returns < 0].mean())
                    strength = min(1.0, 0.5 * vol_intensity + 0.5 * negative_intensity * 10)
                    
                    # Confidence based on consistency of pattern
                    confidence = min(1.0, negative_return_ratio)
                    
                    pattern = DetectedPattern(
                        pattern_name=f"volatility_spike_{start_idx}_{end_idx}",
                        pattern_type="volatility_spike",
                        start_index=start_idx,
                        end_index=end_idx,
                        strength=strength,
                        confidence=confidence,
                        suggested_action="conditional_invert",
                        pattern_metrics={
                            'avg_volatility': period_vol.mean(),
                            'vol_spike_ratio': vol_intensity,
                            'negative_return_ratio': negative_return_ratio,
                            'avg_negative_return': period_returns[period_returns < 0].mean()
                        }
                    )
                    
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_trend_reversal_patterns(self, returns: pd.Series) -> List[DetectedPattern]:
        """Detect trend reversal patterns using moving averages"""
        patterns = []
        
        # Calculate multiple moving averages
        short_ma = returns.rolling(window=5).mean()
        medium_ma = returns.rolling(window=15).mean()
        long_ma = returns.rolling(window=self.trend_window).mean()
        
        # Detect trend changes
        short_vs_medium = short_ma - medium_ma
        medium_vs_long = medium_ma - long_ma
        
        # Find significant trend reversals
        short_trend_change = short_vs_medium.diff()
        medium_trend_change = medium_vs_long.diff()
        
        # Combine signals
        combined_signal = short_trend_change + medium_trend_change
        
        # Detect significant reversals
        reversal_threshold = combined_signal.std() * 1.5
        significant_reversals = abs(combined_signal) > reversal_threshold
        
        reversal_indices = np.where(significant_reversals)[0]
        
        for reversal_idx in reversal_indices:
            # Analyze period around reversal
            lookback = self.min_pattern_length // 2
            start_idx = max(0, reversal_idx - lookback)
            end_idx = min(len(returns) - 1, reversal_idx + lookback)
            
            if end_idx - start_idx >= self.min_pattern_length:
                period_returns = returns.iloc[start_idx:end_idx+1]
                
                # Check if reversal leads to poor performance
                post_reversal_returns = returns.iloc[reversal_idx:end_idx+1]
                if len(post_reversal_returns) > 5 and post_reversal_returns.mean() < -0.001:
                    
                    # Pattern strength based on reversal magnitude and subsequent performance
                    reversal_magnitude = abs(combined_signal.iloc[reversal_idx])
                    subsequent_loss = abs(post_reversal_returns.sum())
                    strength = min(1.0, 0.4 * reversal_magnitude / reversal_threshold + 0.6 * subsequent_loss * 20)
                    
                    # Confidence based on clarity of reversal
                    pre_trend = medium_vs_long.iloc[start_idx:reversal_idx].mean()
                    post_trend = medium_vs_long.iloc[reversal_idx:end_idx+1].mean()
                    trend_change_clarity = abs(post_trend - pre_trend) / (abs(pre_trend) + 1e-8)
                    confidence = min(1.0, trend_change_clarity)
                    
                    pattern = DetectedPattern(
                        pattern_name=f"trend_reversal_{reversal_idx}",
                        pattern_type="trend_reversal",
                        start_index=start_idx,
                        end_index=end_idx,
                        strength=strength,
                        confidence=confidence,
                        suggested_action="smart_invert",
                        pattern_metrics={
                            'reversal_magnitude': reversal_magnitude,
                            'post_reversal_performance': post_reversal_returns.sum(),
                            'trend_change_clarity': trend_change_clarity,
                            'reversal_date_index': reversal_idx
                        }
                    )
                    
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_mean_reversion_patterns(self, returns: pd.Series) -> List[DetectedPattern]:
        """Detect mean reversion patterns that might benefit from inversion"""
        patterns = []
        
        # Calculate z-score of cumulative returns
        cumulative_returns = returns.cumsum()
        rolling_mean = cumulative_returns.rolling(window=self.trend_window * 2).mean()
        rolling_std = cumulative_returns.rolling(window=self.trend_window * 2).std()
        z_score = (cumulative_returns - rolling_mean) / (rolling_std + 1e-8)
        
        # Detect periods of extreme deviation
        extreme_threshold = 2.0  # 2 standard deviations
        extreme_negative = z_score < -extreme_threshold
        
        # Group consecutive extreme periods
        extreme_periods = self._group_consecutive_periods(extreme_negative)
        
        for start_idx, end_idx in extreme_periods:
            if end_idx - start_idx >= self.min_pattern_length:
                period_returns = returns.iloc[start_idx:end_idx+1]
                period_z_score = z_score.iloc[start_idx:end_idx+1]
                
                # Pattern strength based on how extreme the deviation is
                max_deviation = abs(period_z_score.min())
                strength = min(1.0, max_deviation / 4.0)  # Normalize by 4 sigma
                
                # Confidence based on consistency of extreme readings
                extreme_consistency = (period_z_score < -extreme_threshold).mean()
                confidence = extreme_consistency
                
                # Check if this is likely to mean revert (historically)
                mean_reversion_score = self._calculate_mean_reversion_score(
                    cumulative_returns, start_idx, end_idx
                )
                
                if mean_reversion_score > 0.6:  # High probability of mean reversion
                    pattern = DetectedPattern(
                        pattern_name=f"mean_reversion_{start_idx}_{end_idx}",
                        pattern_type="mean_reversion",
                        start_index=start_idx,
                        end_index=end_idx,
                        strength=strength,
                        confidence=confidence,
                        suggested_action="temporary_invert",
                        pattern_metrics={
                            'max_z_score_deviation': max_deviation,
                            'mean_reversion_probability': mean_reversion_score,
                            'extreme_period_consistency': extreme_consistency,
                            'cumulative_loss': period_returns.sum()
                        }
                    )
                    
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_momentum_patterns(self, returns: pd.Series) -> List[DetectedPattern]:
        """Detect negative momentum patterns"""
        patterns = []
        
        # Calculate momentum indicators
        momentum_short = returns.rolling(window=5).sum()
        momentum_medium = returns.rolling(window=15).sum()
        momentum_long = returns.rolling(window=30).sum()
        
        # Detect consistent negative momentum
        negative_momentum = (
            (momentum_short < 0) & 
            (momentum_medium < 0) & 
            (momentum_long < 0)
        )
        
        # Group consecutive negative momentum periods
        momentum_periods = self._group_consecutive_periods(negative_momentum)
        
        for start_idx, end_idx in momentum_periods:
            if end_idx - start_idx >= self.min_pattern_length:
                period_returns = returns.iloc[start_idx:end_idx+1]
                
                # Calculate momentum strength
                period_momentum_short = momentum_short.iloc[start_idx:end_idx+1]
                period_momentum_medium = momentum_medium.iloc[start_idx:end_idx+1]
                period_momentum_long = momentum_long.iloc[start_idx:end_idx+1]
                
                # Average momentum across timeframes
                avg_momentum = (
                    period_momentum_short.mean() + 
                    period_momentum_medium.mean() + 
                    period_momentum_long.mean()
                ) / 3
                
                # Pattern strength based on momentum magnitude
                strength = min(1.0, abs(avg_momentum) * 10)
                
                # Confidence based on consistency across timeframes
                momentum_signals = [
                    (period_momentum_short < 0).mean(),
                    (period_momentum_medium < 0).mean(),
                    (period_momentum_long < 0).mean()
                ]
                confidence = min(momentum_signals)
                
                pattern = DetectedPattern(
                    pattern_name=f"negative_momentum_{start_idx}_{end_idx}",
                    pattern_type="negative_momentum",
                    start_index=start_idx,
                    end_index=end_idx,
                    strength=strength,
                    confidence=confidence,
                    suggested_action="momentum_invert",
                    pattern_metrics={
                        'avg_momentum': avg_momentum,
                        'short_momentum_consistency': momentum_signals[0],
                        'medium_momentum_consistency': momentum_signals[1],
                        'long_momentum_consistency': momentum_signals[2],
                        'period_total_return': period_returns.sum()
                    }
                )
                
                patterns.append(pattern)
        
        return patterns
    
    def _detect_cyclical_patterns(self, returns: pd.Series) -> List[DetectedPattern]:
        """Detect cyclical underperformance patterns"""
        patterns = []
        
        if len(returns) < 100:  # Need sufficient data for cyclical analysis
            return patterns
        
        try:
            # Use FFT to detect cyclical patterns
            fft_values = np.fft.fft(returns.values)
            frequencies = np.fft.fftfreq(len(returns))
            
            # Find dominant frequencies (excluding DC component)
            power_spectrum = np.abs(fft_values[1:len(fft_values)//2])
            freq_spectrum = frequencies[1:len(frequencies)//2]
            
            # Find peaks in power spectrum
            peak_indices, _ = signal.find_peaks(power_spectrum, height=np.std(power_spectrum))
            
            if len(peak_indices) > 0:
                # Analyze the strongest cyclical pattern
                strongest_peak_idx = peak_indices[np.argmax(power_spectrum[peak_indices])]
                dominant_frequency = freq_spectrum[strongest_peak_idx]
                cycle_length = int(1 / abs(dominant_frequency)) if dominant_frequency != 0 else len(returns)
                
                # Only consider reasonable cycle lengths
                if 10 <= cycle_length <= len(returns) // 3:
                    # Analyze performance within cycles
                    cycle_performance = []
                    
                    for start in range(0, len(returns) - cycle_length, cycle_length):
                        cycle_returns = returns.iloc[start:start + cycle_length]
                        cycle_performance.append(cycle_returns.sum())
                    
                    if cycle_performance:
                        # Check if cycles are consistently underperforming
                        negative_cycles = sum(1 for perf in cycle_performance if perf < 0)
                        negative_ratio = negative_cycles / len(cycle_performance)
                        
                        if negative_ratio > 0.7:  # 70% of cycles are negative
                            avg_cycle_performance = np.mean(cycle_performance)
                            
                            pattern = DetectedPattern(
                                pattern_name=f"cyclical_underperformance_{cycle_length}",
                                pattern_type="cyclical_pattern",
                                start_index=0,
                                end_index=len(returns) - 1,
                                strength=min(1.0, abs(avg_cycle_performance) * 50),
                                confidence=negative_ratio,
                                suggested_action="cyclical_invert",
                                pattern_metrics={
                                    'cycle_length': cycle_length,
                                    'negative_cycle_ratio': negative_ratio,
                                    'avg_cycle_performance': avg_cycle_performance,
                                    'total_cycles_analyzed': len(cycle_performance),
                                    'dominant_frequency': dominant_frequency
                                }
                            )
                            
                            patterns.append(pattern)
        
        except Exception as e:
            logger.warning(f"Error in cyclical pattern detection: {e}")
        
        return patterns
    
    def _detect_regime_shift_patterns(self, returns: pd.Series) -> List[DetectedPattern]:
        """Detect regime shift patterns using change point detection"""
        patterns = []
        
        if len(returns) < 50:
            return patterns
        
        try:
            # Simple change point detection using rolling statistics
            window = 20
            rolling_mean = returns.rolling(window=window).mean()
            rolling_std = returns.rolling(window=window).std()
            
            # Detect significant changes in mean and volatility
            mean_changes = rolling_mean.diff().abs()
            std_changes = rolling_std.diff().abs()
            
            # Combined change signal
            combined_changes = (
                mean_changes / (rolling_mean.std() + 1e-8) +
                std_changes / (rolling_std.std() + 1e-8)
            )
            
            # Find significant change points
            change_threshold = combined_changes.quantile(0.9)  # Top 10% of changes
            significant_changes = combined_changes > change_threshold
            
            change_indices = np.where(significant_changes)[0]
            
            # Analyze periods after change points
            for change_idx in change_indices:
                analysis_start = change_idx
                analysis_end = min(len(returns) - 1, change_idx + self.min_pattern_length * 2)
                
                if analysis_end - analysis_start >= self.min_pattern_length:
                    # Compare pre and post change performance
                    pre_period = returns.iloc[max(0, change_idx - window):change_idx]
                    post_period = returns.iloc[analysis_start:analysis_end+1]
                    
                    if len(pre_period) > 5 and len(post_period) > 5:
                        pre_mean = pre_period.mean()
                        post_mean = post_period.mean()
                        
                        # Check if regime shift led to underperformance
                        performance_deterioration = pre_mean - post_mean
                        
                        if performance_deterioration > 0.005:  # 0.5% deterioration threshold
                            # Pattern strength based on magnitude of change
                            change_magnitude = combined_changes.iloc[change_idx]
                            strength = min(1.0, change_magnitude / change_threshold)
                            
                            # Confidence based on persistence of new regime
                            regime_persistence = (post_period < pre_mean).mean()
                            confidence = regime_persistence
                            
                            pattern = DetectedPattern(
                                pattern_name=f"regime_shift_{change_idx}",
                                pattern_type="regime_shift",
                                start_index=analysis_start,
                                end_index=analysis_end,
                                strength=strength,
                                confidence=confidence,
                                suggested_action="post_shift_invert",
                                pattern_metrics={
                                    'change_point_index': change_idx,
                                    'pre_regime_mean': pre_mean,
                                    'post_regime_mean': post_mean,
                                    'performance_deterioration': performance_deterioration,
                                    'change_magnitude': change_magnitude,
                                    'regime_persistence': regime_persistence
                                }
                            )
                            
                            patterns.append(pattern)
        
        except Exception as e:
            logger.warning(f"Error in regime shift detection: {e}")
        
        return patterns
    
    # Utility methods
    
    def _clean_returns(self, returns: pd.Series) -> pd.Series:
        """Clean returns data"""
        cleaned = returns.replace([np.inf, -np.inf], np.nan).fillna(0)
        return cleaned
    
    def _group_consecutive_periods(self, mask: pd.Series) -> List[Tuple[int, int]]:
        """Group consecutive True values in boolean mask"""
        periods = []
        start_idx = None
        
        for i, value in enumerate(mask):
            if value and start_idx is None:
                start_idx = i
            elif not value and start_idx is not None:
                periods.append((start_idx, i - 1))
                start_idx = None
        
        # Handle case where period extends to end
        if start_idx is not None:
            periods.append((start_idx, len(mask) - 1))
        
        return periods
    
    def _estimate_recovery_probability(self, returns: pd.Series) -> float:
        """Estimate probability of recovery from drawdown"""
        # Simple heuristic based on historical patterns
        cumulative = returns.cumsum()
        
        if len(returns) < 10:
            return 0.5
        
        # Check how often negative periods are followed by recovery
        negative_periods = cumulative < 0
        recoveries = 0
        total_negative_periods = 0
        
        in_negative = False
        for i, is_negative in enumerate(negative_periods):
            if is_negative and not in_negative:
                in_negative = True
                total_negative_periods += 1
            elif not is_negative and in_negative:
                in_negative = False
                recoveries += 1
        
        if total_negative_periods == 0:
            return 0.5
        
        return recoveries / total_negative_periods
    
    def _calculate_mean_reversion_score(self,
                                      cumulative_returns: pd.Series,
                                      start_idx: int,
                                      end_idx: int) -> float:
        """Calculate mean reversion probability score"""
        
        # Look at historical behavior after similar extreme periods
        threshold = cumulative_returns.iloc[start_idx:end_idx+1].min()
        
        # Find other periods with similar extremes
        extreme_periods = cumulative_returns < threshold * 0.9  # 90% of current extreme
        
        reversions = 0
        total_extremes = 0
        
        # Analyze what happened after extreme periods
        for i, is_extreme in enumerate(extreme_periods):
            if is_extreme and i < len(cumulative_returns) - 20:  # Need lookahead
                total_extremes += 1
                # Check if it reverted within 20 periods
                future_values = cumulative_returns.iloc[i:i+20]
                if future_values.max() > cumulative_returns.iloc[i] * 1.1:  # 10% recovery
                    reversions += 1
        
        if total_extremes == 0:
            return 0.5
        
        return reversions / total_extremes