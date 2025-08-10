"""
Adaptive Noise Filter

This module implements advanced noise filtering with multiple levels,
microstructure noise handling, and false positive prevention.

Key Features:
- Multi-level noise filtering architecture
- Microstructure noise detection and handling
- Adaptive filter parameters based on market conditions
- False positive prevention with learning mechanisms
- Real-time noise assessment and filtering
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from collections import deque, defaultdict
import logging
from dataclasses import dataclass, field
from enum import Enum
import warnings
from scipy import signal as scipy_signal
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)


class NoiseType(Enum):
    """Types of detected noise"""
    MICROSTRUCTURE = "microstructure"
    VOLUME_ANOMALY = "volume_anomaly"
    VOLATILITY_BURST = "volatility_burst"
    SIGNAL_INTERFERENCE = "signal_interference"
    DATA_QUALITY = "data_quality"
    SYSTEMATIC = "systematic"


class FilterLevel(Enum):
    """Filter processing levels"""
    LEVEL_1_BASIC = "level_1_basic"
    LEVEL_2_INTERMEDIATE = "level_2_intermediate"
    LEVEL_3_ADVANCED = "level_3_advanced"


@dataclass
class NoiseDetectionResult:
    """Result of noise detection analysis"""
    has_noise: bool
    noise_types: List[NoiseType]
    noise_score: float
    confidence: float
    filtered_signal: Any
    metadata: Dict[str, Any]


@dataclass
class FilterConfig:
    """Configuration for noise filtering"""
    # Basic filtering
    enable_basic_filter: bool = True
    basic_threshold: float = 0.1
    
    # Microstructure filtering
    enable_microstructure_filter: bool = True
    microstructure_window: int = 5
    microstructure_threshold: float = 0.05
    
    # Volume filtering
    enable_volume_filter: bool = True
    volume_zscore_threshold: float = 2.5
    volume_window: int = 20
    
    # Volatility filtering
    enable_volatility_filter: bool = True
    volatility_threshold: float = 3.0
    volatility_window: int = 10
    
    # Adaptive parameters
    enable_adaptive_tuning: bool = True
    adaptation_rate: float = 0.1
    market_regime_adjustment: bool = True
    
    # Learning parameters
    enable_learning: bool = True
    false_positive_penalty: float = 0.2
    true_positive_reward: float = 0.1


class AdaptiveNoiseFilter:
    """
    Advanced noise filtering system with adaptive parameters
    """
    
    def __init__(self, config: Optional[FilterConfig] = None):
        """
        Initialize adaptive noise filter
        
        Args:
            config: Filter configuration
        """
        self.config = config or FilterConfig()
        
        # Filter state
        self.filter_history = deque(maxlen=1000)
        self.noise_statistics = defaultdict(list)
        self.filter_parameters = self._initialize_filter_parameters()
        
        # Market data buffers
        self.price_buffer = deque(maxlen=100)
        self.volume_buffer = deque(maxlen=100)
        self.volatility_buffer = deque(maxlen=100)
        self.regime_scores_buffer = deque(maxlen=100)
        
        # Adaptive learning
        self.filter_performance = deque(maxlen=500)
        self.false_positive_count = 0
        self.true_positive_count = 0
        self.total_filtered = 0
        
        # Microstructure noise detection
        self.microstructure_patterns = {}
        self.pattern_detector = DBSCAN(eps=0.1, min_samples=3)
        
        # Signal processing components
        self.scaler = StandardScaler()
        self.noise_estimator_trained = False
        
        logger.info("AdaptiveNoiseFilter initialized")
    
    def _initialize_filter_parameters(self) -> Dict[str, float]:
        """Initialize adaptive filter parameters"""
        
        return {
            'microstructure_sensitivity': self.config.microstructure_threshold,
            'volume_sensitivity': self.config.volume_zscore_threshold,
            'volatility_sensitivity': self.config.volatility_threshold,
            'basic_sensitivity': self.config.basic_threshold,
            'adaptation_factor': 1.0
        }
    
    def filter_regime_signal(self, regime_scores: Dict[int, float],
                           market_data: Dict[str, Any],
                           current_regime: int) -> NoiseDetectionResult:
        """
        Apply comprehensive noise filtering to regime signals
        
        Args:
            regime_scores: Current regime scores
            market_data: Market data context
            current_regime: Current active regime
            
        Returns:
            Noise detection and filtering result
        """
        start_time = datetime.now()
        
        # Store data in buffers
        self._update_buffers(regime_scores, market_data)
        
        # Multi-level filtering
        level_1_result = self._apply_level_1_filtering(regime_scores, market_data)
        level_2_result = self._apply_level_2_filtering(
            level_1_result.filtered_signal, market_data, current_regime
        )
        level_3_result = self._apply_level_3_filtering(
            level_2_result.filtered_signal, market_data, current_regime
        )
        
        # Combine results
        final_result = self._combine_filter_results([
            level_1_result, level_2_result, level_3_result
        ])
        
        # Adaptive parameter updates
        if self.config.enable_adaptive_tuning:
            self._update_adaptive_parameters(final_result, market_data)
        
        # Record filtering statistics
        self._record_filter_statistics(final_result, start_time)
        
        return final_result
    
    def _update_buffers(self, regime_scores: Dict[int, float],
                       market_data: Dict[str, Any]):
        """Update internal data buffers"""
        
        # Store regime scores
        self.regime_scores_buffer.append({
            'timestamp': datetime.now(),
            'scores': regime_scores.copy()
        })
        
        # Store market data
        if 'spot_price' in market_data:
            self.price_buffer.append(market_data['spot_price'])
        
        if 'volume' in market_data:
            self.volume_buffer.append(market_data['volume'])
        
        if 'volatility' in market_data:
            self.volatility_buffer.append(market_data['volatility'])
    
    def _apply_level_1_filtering(self, regime_scores: Dict[int, float],
                               market_data: Dict[str, Any]) -> NoiseDetectionResult:
        """Apply Level 1 (Basic) filtering"""
        
        noise_types = []
        noise_score = 0.0
        filtered_scores = regime_scores.copy()
        
        # Basic threshold filtering
        if self.config.enable_basic_filter:
            basic_noise = self._detect_basic_noise(regime_scores)
            if basic_noise['has_noise']:
                noise_types.append(NoiseType.SIGNAL_INTERFERENCE)
                noise_score += basic_noise['score']
                filtered_scores = basic_noise['filtered_scores']
        
        # Data quality checks
        quality_noise = self._detect_data_quality_issues(regime_scores, market_data)
        if quality_noise['has_noise']:
            noise_types.append(NoiseType.DATA_QUALITY)
            noise_score += quality_noise['score']
            filtered_scores = quality_noise['filtered_scores']
        
        return NoiseDetectionResult(
            has_noise=len(noise_types) > 0,
            noise_types=noise_types,
            noise_score=noise_score,
            confidence=0.8,
            filtered_signal=filtered_scores,
            metadata={'level': FilterLevel.LEVEL_1_BASIC}
        )
    
    def _apply_level_2_filtering(self, regime_scores: Dict[int, float],
                               market_data: Dict[str, Any],
                               current_regime: int) -> NoiseDetectionResult:
        """Apply Level 2 (Intermediate) filtering"""
        
        noise_types = []
        noise_score = 0.0
        filtered_scores = regime_scores.copy()
        
        # Volume-based noise detection
        if self.config.enable_volume_filter:
            volume_noise = self._detect_volume_noise(market_data)
            if volume_noise['has_noise']:
                noise_types.append(NoiseType.VOLUME_ANOMALY)
                noise_score += volume_noise['score']
                # Apply volume-based score adjustment
                filtered_scores = self._apply_volume_adjustment(
                    filtered_scores, volume_noise['adjustment_factor']
                )
        
        # Volatility burst detection
        if self.config.enable_volatility_filter:
            volatility_noise = self._detect_volatility_burst(market_data)
            if volatility_noise['has_noise']:
                noise_types.append(NoiseType.VOLATILITY_BURST)
                noise_score += volatility_noise['score']
                # Apply volatility-based filtering
                filtered_scores = self._apply_volatility_filtering(
                    filtered_scores, volatility_noise['intensity']
                )
        
        return NoiseDetectionResult(
            has_noise=len(noise_types) > 0,
            noise_types=noise_types,
            noise_score=noise_score,
            confidence=0.85,
            filtered_signal=filtered_scores,
            metadata={'level': FilterLevel.LEVEL_2_INTERMEDIATE}
        )
    
    def _apply_level_3_filtering(self, regime_scores: Dict[int, float],
                               market_data: Dict[str, Any],
                               current_regime: int) -> NoiseDetectionResult:
        """Apply Level 3 (Advanced) filtering"""
        
        noise_types = []
        noise_score = 0.0
        filtered_scores = regime_scores.copy()
        
        # Microstructure noise detection
        if self.config.enable_microstructure_filter:
            microstructure_noise = self._detect_microstructure_noise(
                regime_scores, market_data
            )
            if microstructure_noise['has_noise']:
                noise_types.append(NoiseType.MICROSTRUCTURE)
                noise_score += microstructure_noise['score']
                filtered_scores = microstructure_noise['filtered_scores']
        
        # Systematic noise detection
        systematic_noise = self._detect_systematic_noise(regime_scores, current_regime)
        if systematic_noise['has_noise']:
            noise_types.append(NoiseType.SYSTEMATIC)
            noise_score += systematic_noise['score']
            filtered_scores = systematic_noise['filtered_scores']
        
        return NoiseDetectionResult(
            has_noise=len(noise_types) > 0,
            noise_types=noise_types,
            noise_score=noise_score,
            confidence=0.9,
            filtered_signal=filtered_scores,
            metadata={'level': FilterLevel.LEVEL_3_ADVANCED}
        )
    
    def _detect_basic_noise(self, regime_scores: Dict[int, float]) -> Dict[str, Any]:
        """Detect basic signal noise"""
        
        threshold = self.filter_parameters['basic_sensitivity']
        
        # Check for unrealistic score distributions
        scores = list(regime_scores.values())
        score_variance = np.var(scores)
        
        if score_variance < 0.001:  # All scores too similar
            # Apply small random perturbation
            filtered_scores = {}
            for regime_id, score in regime_scores.items():
                perturbation = np.random.normal(0, 0.01)
                filtered_scores[regime_id] = max(0, min(1, score + perturbation))
            
            # Renormalize
            total = sum(filtered_scores.values())
            if total > 0:
                filtered_scores = {k: v/total for k, v in filtered_scores.items()}
            
            return {
                'has_noise': True,
                'score': 0.3,
                'filtered_scores': filtered_scores
            }
        
        # Check for extreme values
        extreme_count = sum(1 for score in scores if score > 0.95 or score < 0.05)
        if extreme_count > len(scores) * 0.5:
            # Smooth extreme values
            filtered_scores = {}
            for regime_id, score in regime_scores.items():
                if score > 0.95:
                    filtered_scores[regime_id] = 0.95
                elif score < 0.05:
                    filtered_scores[regime_id] = 0.05
                else:
                    filtered_scores[regime_id] = score
            
            return {
                'has_noise': True,
                'score': 0.2,
                'filtered_scores': filtered_scores
            }
        
        return {
            'has_noise': False,
            'score': 0.0,
            'filtered_scores': regime_scores
        }
    
    def _detect_data_quality_issues(self, regime_scores: Dict[int, float],
                                  market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect data quality issues"""
        
        issues = []
        
        # Check for NaN or infinite values
        for regime_id, score in regime_scores.items():
            if not np.isfinite(score):
                issues.append('invalid_regime_scores')
                break
        
        # Check market data quality
        for key, value in market_data.items():
            if isinstance(value, (int, float)) and not np.isfinite(value):
                issues.append('invalid_market_data')
                break
        
        if issues:
            # Clean scores
            filtered_scores = {}
            for regime_id, score in regime_scores.items():
                if np.isfinite(score):
                    filtered_scores[regime_id] = float(score)
                else:
                    filtered_scores[regime_id] = 1.0 / len(regime_scores)
            
            # Renormalize
            total = sum(filtered_scores.values())
            if total > 0:
                filtered_scores = {k: v/total for k, v in filtered_scores.items()}
            
            return {
                'has_noise': True,
                'score': 0.5,
                'filtered_scores': filtered_scores
            }
        
        return {
            'has_noise': False,
            'score': 0.0,
            'filtered_scores': regime_scores
        }
    
    def _detect_volume_noise(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect volume-based noise"""
        
        current_volume = market_data.get('volume', 0)
        
        if len(self.volume_buffer) < self.config.volume_window:
            return {'has_noise': False, 'score': 0.0}
        
        # Calculate volume z-score
        recent_volumes = list(self.volume_buffer)[-self.config.volume_window:]
        volume_mean = np.mean(recent_volumes)
        volume_std = np.std(recent_volumes)
        
        if volume_std > 0:
            volume_zscore = abs(current_volume - volume_mean) / volume_std
            
            threshold = self.filter_parameters['volume_sensitivity']
            
            if volume_zscore > threshold:
                # Volume anomaly detected
                adjustment_factor = 1.0 / (1.0 + volume_zscore / threshold)
                
                return {
                    'has_noise': True,
                    'score': min(volume_zscore / threshold, 1.0),
                    'adjustment_factor': adjustment_factor
                }
        
        return {'has_noise': False, 'score': 0.0, 'adjustment_factor': 1.0}
    
    def _detect_volatility_burst(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect volatility burst noise"""
        
        current_volatility = market_data.get('volatility', 0.0)
        
        if len(self.volatility_buffer) < self.config.volatility_window:
            return {'has_noise': False, 'score': 0.0}
        
        # Calculate volatility percentile
        recent_volatilities = list(self.volatility_buffer)[-self.config.volatility_window:]
        volatility_percentile = np.percentile(recent_volatilities, 95)
        
        threshold = self.filter_parameters['volatility_sensitivity']
        
        if current_volatility > volatility_percentile * threshold:
            # Volatility burst detected
            intensity = current_volatility / volatility_percentile
            
            return {
                'has_noise': True,
                'score': min(intensity / threshold, 1.0),
                'intensity': intensity
            }
        
        return {'has_noise': False, 'score': 0.0, 'intensity': 1.0}
    
    def _detect_microstructure_noise(self, regime_scores: Dict[int, float],
                                   market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect microstructure noise patterns"""
        
        if len(self.regime_scores_buffer) < self.config.microstructure_window:
            return {'has_noise': False, 'score': 0.0, 'filtered_scores': regime_scores}
        
        # Analyze recent score patterns
        recent_scores = list(self.regime_scores_buffer)[-self.config.microstructure_window:]
        
        # Extract score sequences for each regime
        regime_sequences = defaultdict(list)
        for score_data in recent_scores:
            for regime_id, score in score_data['scores'].items():
                regime_sequences[regime_id].append(score)
        
        # Detect oscillation patterns
        oscillation_detected = False
        max_oscillation = 0.0
        
        for regime_id, sequence in regime_sequences.items():
            if len(sequence) >= 3:
                # Check for rapid oscillations
                oscillations = 0
                for i in range(1, len(sequence) - 1):
                    if ((sequence[i-1] < sequence[i] > sequence[i+1]) or
                        (sequence[i-1] > sequence[i] < sequence[i+1])):
                        oscillations += 1
                
                oscillation_ratio = oscillations / (len(sequence) - 2)
                max_oscillation = max(max_oscillation, oscillation_ratio)
                
                if oscillation_ratio > 0.5:  # More than 50% oscillations
                    oscillation_detected = True
        
        if oscillation_detected:
            # Apply smoothing filter
            filtered_scores = {}
            
            for regime_id in regime_scores:
                if regime_id in regime_sequences and len(regime_sequences[regime_id]) >= 3:
                    # Apply simple moving average
                    recent_values = regime_sequences[regime_id][-3:]
                    smoothed_score = np.mean(recent_values)
                    filtered_scores[regime_id] = smoothed_score
                else:
                    filtered_scores[regime_id] = regime_scores[regime_id]
            
            # Renormalize
            total = sum(filtered_scores.values())
            if total > 0:
                filtered_scores = {k: v/total for k, v in filtered_scores.items()}
            
            return {
                'has_noise': True,
                'score': max_oscillation,
                'filtered_scores': filtered_scores
            }
        
        return {'has_noise': False, 'score': 0.0, 'filtered_scores': regime_scores}
    
    def _detect_systematic_noise(self, regime_scores: Dict[int, float],
                               current_regime: int) -> Dict[str, Any]:
        """Detect systematic bias or noise"""
        
        if len(self.regime_scores_buffer) < 20:
            return {'has_noise': False, 'score': 0.0, 'filtered_scores': regime_scores}
        
        # Check for systematic bias towards certain regimes
        recent_scores = list(self.regime_scores_buffer)[-20:]
        
        # Calculate average scores for each regime
        regime_averages = defaultdict(list)
        for score_data in recent_scores:
            for regime_id, score in score_data['scores'].items():
                regime_averages[regime_id].append(score)
        
        avg_scores = {
            regime_id: np.mean(scores)
            for regime_id, scores in regime_averages.items()
        }
        
        # Check for unrealistic dominance
        max_avg = max(avg_scores.values()) if avg_scores else 0
        min_avg = min(avg_scores.values()) if avg_scores else 0
        
        if max_avg - min_avg > 0.7:  # One regime dominates too much
            # Apply bias correction
            correction_factor = 0.1  # Reduce extreme differences
            
            filtered_scores = {}
            for regime_id, score in regime_scores.items():
                avg_score = avg_scores.get(regime_id, 0.5)
                
                # Move score towards regime average
                corrected_score = score * (1 - correction_factor) + avg_score * correction_factor
                filtered_scores[regime_id] = corrected_score
            
            # Renormalize
            total = sum(filtered_scores.values())
            if total > 0:
                filtered_scores = {k: v/total for k, v in filtered_scores.items()}
            
            return {
                'has_noise': True,
                'score': (max_avg - min_avg) - 0.5,  # Excess above threshold
                'filtered_scores': filtered_scores
            }
        
        return {'has_noise': False, 'score': 0.0, 'filtered_scores': regime_scores}
    
    def _apply_volume_adjustment(self, regime_scores: Dict[int, float],
                               adjustment_factor: float) -> Dict[int, float]:
        """Apply volume-based score adjustment"""
        
        # Reduce confidence in all scores during volume anomalies
        adjusted_scores = {}
        
        for regime_id, score in regime_scores.items():
            # Move scores towards uniform distribution
            uniform_score = 1.0 / len(regime_scores)
            adjusted_score = score * adjustment_factor + uniform_score * (1 - adjustment_factor)
            adjusted_scores[regime_id] = adjusted_score
        
        # Renormalize
        total = sum(adjusted_scores.values())
        if total > 0:
            adjusted_scores = {k: v/total for k, v in adjusted_scores.items()}
        
        return adjusted_scores
    
    def _apply_volatility_filtering(self, regime_scores: Dict[int, float],
                                  intensity: float) -> Dict[int, float]:
        """Apply volatility-based filtering"""
        
        # During high volatility, reduce extreme scores
        smoothing_factor = min(0.3, (intensity - 1.0) * 0.1)
        
        if smoothing_factor <= 0:
            return regime_scores
        
        # Calculate target (smoothed) distribution
        mean_score = np.mean(list(regime_scores.values()))
        
        filtered_scores = {}
        for regime_id, score in regime_scores.items():
            # Move extreme scores towards mean
            if abs(score - mean_score) > 0.3:
                smoothed_score = score * (1 - smoothing_factor) + mean_score * smoothing_factor
                filtered_scores[regime_id] = smoothed_score
            else:
                filtered_scores[regime_id] = score
        
        # Renormalize
        total = sum(filtered_scores.values())
        if total > 0:
            filtered_scores = {k: v/total for k, v in filtered_scores.items()}
        
        return filtered_scores
    
    def _combine_filter_results(self, results: List[NoiseDetectionResult]) -> NoiseDetectionResult:
        """Combine results from multiple filter levels"""
        
        # Aggregate noise information
        all_noise_types = []
        total_noise_score = 0.0
        has_any_noise = False
        
        for result in results:
            if result.has_noise:
                has_any_noise = True
                all_noise_types.extend(result.noise_types)
                total_noise_score += result.noise_score
        
        # Use the most processed signal (from Level 3)
        final_signal = results[-1].filtered_signal
        
        # Calculate combined confidence
        combined_confidence = np.mean([result.confidence for result in results])
        
        return NoiseDetectionResult(
            has_noise=has_any_noise,
            noise_types=list(set(all_noise_types)),  # Remove duplicates
            noise_score=total_noise_score / len(results),
            confidence=combined_confidence,
            filtered_signal=final_signal,
            metadata={
                'levels_processed': len(results),
                'individual_results': [r.metadata for r in results]
            }
        )
    
    def _update_adaptive_parameters(self, result: NoiseDetectionResult,
                                  market_data: Dict[str, Any]):
        """Update adaptive filter parameters based on results"""
        
        if not self.config.enable_adaptive_tuning:
            return
        
        adaptation_rate = self.config.adaptation_rate
        
        # Adjust based on noise detection results
        if result.has_noise:
            # Increase sensitivity if noise was detected
            for param in self.filter_parameters:
                if 'sensitivity' in param:
                    current_value = self.filter_parameters[param]
                    # Slightly increase sensitivity
                    new_value = current_value * (1 + adaptation_rate * 0.1)
                    self.filter_parameters[param] = min(new_value, current_value * 1.5)
        
        # Market regime-based adjustments
        if self.config.market_regime_adjustment:
            volatility = market_data.get('volatility', 0.2)
            
            # Higher volatility = more sensitive filtering
            volatility_factor = 1.0 + (volatility - 0.2) * 0.5
            self.filter_parameters['adaptation_factor'] = volatility_factor
    
    def _record_filter_statistics(self, result: NoiseDetectionResult,
                                start_time: datetime):
        """Record filtering statistics for analysis"""
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        filter_record = {
            'timestamp': start_time,
            'processing_time_ms': processing_time * 1000,
            'noise_detected': result.has_noise,
            'noise_types': [nt.value for nt in result.noise_types],
            'noise_score': result.noise_score,
            'confidence': result.confidence,
            'filter_parameters': self.filter_parameters.copy()
        }
        
        self.filter_history.append(filter_record)
        
        # Update counters
        if result.has_noise:
            self.total_filtered += 1
    
    def update_filter_feedback(self, was_false_positive: bool):
        """Update filter based on feedback about filtering decision"""
        
        if was_false_positive:
            self.false_positive_count += 1
            
            # Reduce filter sensitivity
            penalty = self.config.false_positive_penalty
            for param in self.filter_parameters:
                if 'sensitivity' in param:
                    current_value = self.filter_parameters[param]
                    self.filter_parameters[param] = current_value * (1 + penalty)
        else:
            self.true_positive_count += 1
            
            # Slightly increase filter sensitivity
            reward = self.config.true_positive_reward
            for param in self.filter_parameters:
                if 'sensitivity' in param:
                    current_value = self.filter_parameters[param]
                    self.filter_parameters[param] = current_value * (1 - reward)
        
        # Record performance
        total_decisions = self.false_positive_count + self.true_positive_count
        if total_decisions > 0:
            accuracy = self.true_positive_count / total_decisions
            self.filter_performance.append(accuracy)
    
    def get_filter_statistics(self) -> Dict[str, Any]:
        """Get comprehensive filter statistics"""
        
        recent_history = list(self.filter_history)[-100:]
        
        # Noise type frequency
        noise_type_counts = defaultdict(int)
        for record in recent_history:
            for noise_type in record['noise_types']:
                noise_type_counts[noise_type] += 1
        
        # Performance metrics
        total_decisions = self.false_positive_count + self.true_positive_count
        accuracy = self.true_positive_count / total_decisions if total_decisions > 0 else 0.0
        
        # Processing performance
        processing_times = [r['processing_time_ms'] for r in recent_history]
        
        return {
            'total_filtered': self.total_filtered,
            'filter_accuracy': accuracy,
            'false_positive_rate': self.false_positive_count / max(total_decisions, 1),
            'noise_type_distribution': dict(noise_type_counts),
            'current_parameters': self.filter_parameters.copy(),
            'processing_stats': {
                'avg_processing_time_ms': np.mean(processing_times) if processing_times else 0,
                'max_processing_time_ms': np.max(processing_times) if processing_times else 0,
                'total_processed': len(self.filter_history)
            },
            'recent_performance': list(self.filter_performance)[-20:] if self.filter_performance else []
        }
    
    def export_filter_analysis(self, filepath: str):
        """Export detailed filter analysis"""
        
        analysis = {
            'configuration': self.config.__dict__,
            'statistics': self.get_filter_statistics(),
            'parameter_evolution': [
                {
                    'timestamp': record['timestamp'].isoformat(),
                    'parameters': record['filter_parameters']
                }
                for record in list(self.filter_history)[-50:]  # Last 50 records
            ],
            'noise_detection_history': [
                {
                    'timestamp': record['timestamp'].isoformat(),
                    'noise_detected': record['noise_detected'],
                    'noise_types': record['noise_types'],
                    'noise_score': record['noise_score'],
                    'confidence': record['confidence']
                }
                for record in list(self.filter_history)[-100:]  # Last 100 records
            ]
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Filter analysis exported to {filepath}")


# Example usage
if __name__ == "__main__":
    # Create adaptive noise filter
    config = FilterConfig(
        enable_microstructure_filter=True,
        enable_volume_filter=True,
        enable_volatility_filter=True,
        enable_adaptive_tuning=True,
        enable_learning=True
    )
    
    noise_filter = AdaptiveNoiseFilter(config)
    
    # Simulate filtering
    for i in range(100):
        # Generate regime scores with noise
        regime_scores = {j: np.random.beta(2, 2) for j in range(8)}
        
        # Add artificial noise occasionally
        if i % 20 == 0:
            # Add oscillation noise
            for j in range(8):
                regime_scores[j] += 0.1 * np.sin(i * 0.5)
        
        # Normalize scores
        total = sum(regime_scores.values())
        regime_scores = {k: v/total for k, v in regime_scores.items()}
        
        # Generate market data
        market_data = {
            'spot_price': 25000 + 1000 * np.random.randn(),
            'volume': max(1000, 5000 + 2000 * np.random.randn()),
            'volatility': max(0.05, 0.2 + 0.1 * np.random.randn())
        }
        
        # Add volume spikes occasionally
        if i % 30 == 0:
            market_data['volume'] *= 5
        
        # Apply filtering
        current_regime = max(regime_scores.items(), key=lambda x: x[1])[0]
        result = noise_filter.filter_regime_signal(
            regime_scores, market_data, current_regime
        )
        
        # Simulate feedback (90% accuracy)
        was_false_positive = np.random.random() < 0.1
        if result.has_noise:
            noise_filter.update_filter_feedback(was_false_positive)
    
    # Get statistics
    stats = noise_filter.get_filter_statistics()
    
    print("Noise Filter Statistics:")
    print(f"Total filtered: {stats['total_filtered']}")
    print(f"Filter accuracy: {stats['filter_accuracy']:.2%}")
    print(f"False positive rate: {stats['false_positive_rate']:.2%}")
    print(f"Noise types detected: {list(stats['noise_type_distribution'].keys())}")
    print(f"Average processing time: {stats['processing_stats']['avg_processing_time_ms']:.2f}ms")