"""
Market Regime Classifier

This module handles the classification of market regimes based on
aggregated indicator signals and confidence scores.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

from .models import RegimeConfig, RegimeType, RegimeClassification

logger = logging.getLogger(__name__)

class RegimeClassifier:
    """
    Classify market regimes based on aggregated scores
    
    This class takes aggregated indicator signals and classifies them
    into specific regime types with confidence scores.
    """
    
    def __init__(self, config: RegimeConfig):
        """
        Initialize the regime classifier
        
        Args:
            config (RegimeConfig): Configuration for classification
        """
        self.config = config
        self.thresholds = self._initialize_thresholds()
        self.regime_history = []
        
        logger.info("RegimeClassifier initialized")
    
    def _initialize_thresholds(self) -> Dict[str, float]:
        """Initialize classification thresholds"""
        return {
            'strong_bullish': 1.5,
            'moderate_bullish': 0.75,
            'weak_bullish': 0.25,
            'neutral_upper': 0.25,
            'neutral_lower': -0.25,
            'weak_bearish': -0.75,
            'moderate_bearish': -1.5,
            'strong_bearish': -2.0,
            'high_volatility': 1.8,
            'low_volatility': 0.3,
            'transition_threshold': 0.1
        }
    
    def classify(self, aggregated_signals: pd.DataFrame) -> List[RegimeClassification]:
        """
        Classify regimes based on aggregated signals
        
        Args:
            aggregated_signals (pd.DataFrame): Aggregated indicator signals
            
        Returns:
            List[RegimeClassification]: Classified regimes
        """
        try:
            classifications = []
            
            for timestamp, row in aggregated_signals.iterrows():
                classification = self._classify_single_point(timestamp, row)
                if classification:
                    classifications.append(classification)
            
            # Apply post-processing
            classifications = self._post_process_classifications(classifications)
            
            logger.info(f"Classified {len(classifications)} regime points")
            return classifications
            
        except Exception as e:
            logger.error(f"Error in regime classification: {e}")
            return []
    
    def _classify_single_point(self, timestamp: datetime, signals: pd.Series) -> Optional[RegimeClassification]:
        """Classify a single data point"""
        try:
            regime_score = signals.get('weighted_signal', 0.0)
            confidence = signals.get('confidence', 0.5)
            
            # Skip if confidence is too low
            if confidence < self.config.confidence_threshold:
                return None
            
            # Determine regime type
            regime_type = self._score_to_regime_type(regime_score, signals)
            
            # Calculate final confidence
            final_confidence = self._calculate_final_confidence(regime_score, confidence, signals)
            
            # Extract component scores
            component_scores = self._extract_component_scores(signals)
            
            # Extract timeframe scores
            timeframe_scores = self._extract_timeframe_scores(signals)
            
            # Create classification
            classification = RegimeClassification(
                timestamp=timestamp,
                symbol=self.config.symbol,
                regime_type=regime_type,
                regime_score=regime_score,
                confidence=final_confidence,
                component_scores=component_scores,
                timeframe_scores=timeframe_scores,
                metadata={
                    'total_weight': signals.get('total_weight', 0.0),
                    'num_indicators': len(component_scores),
                    'classification_method': 'threshold_based'
                }
            )
            
            return classification
            
        except Exception as e:
            logger.error(f"Error classifying single point at {timestamp}: {e}")
            return None
    
    def _score_to_regime_type(self, score: float, signals: pd.Series) -> RegimeType:
        """Convert regime score to regime type with additional logic"""
        
        # Check for special conditions first
        volatility_indicators = [col for col in signals.index if 'volatility' in col.lower() or 'atr' in col.lower()]
        if volatility_indicators:
            avg_volatility = np.mean([signals.get(col, 0) for col in volatility_indicators])
            if abs(avg_volatility) > self.thresholds['high_volatility']:
                return RegimeType.HIGH_VOLATILITY
            elif abs(avg_volatility) < self.thresholds['low_volatility']:
                return RegimeType.LOW_VOLATILITY
        
        # Check for sideways market
        if self._is_sideways_market(signals):
            return RegimeType.SIDEWAYS
        
        # Standard threshold-based classification
        if score >= self.thresholds['strong_bullish']:
            return RegimeType.STRONG_BULLISH
        elif score >= self.thresholds['moderate_bullish']:
            return RegimeType.MODERATE_BULLISH
        elif score >= self.thresholds['weak_bullish']:
            return RegimeType.WEAK_BULLISH
        elif score >= self.thresholds['neutral_lower'] and score <= self.thresholds['neutral_upper']:
            return RegimeType.NEUTRAL
        elif score >= self.thresholds['weak_bearish']:
            return RegimeType.WEAK_BEARISH
        elif score >= self.thresholds['moderate_bearish']:
            return RegimeType.MODERATE_BEARISH
        else:
            return RegimeType.STRONG_BEARISH
    
    def _is_sideways_market(self, signals: pd.Series) -> bool:
        """Detect sideways market conditions"""
        try:
            # Look for trend indicators
            trend_indicators = [col for col in signals.index if any(x in col.lower() for x in ['ema', 'trend', 'momentum'])]
            
            if trend_indicators:
                trend_signals = [signals.get(col, 0) for col in trend_indicators]
                avg_trend = np.mean(trend_signals)
                trend_std = np.std(trend_signals)
                
                # Sideways if low average trend and low standard deviation
                if abs(avg_trend) < 0.2 and trend_std < 0.3:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting sideways market: {e}")
            return False
    
    def _calculate_final_confidence(self, regime_score: float, base_confidence: float, signals: pd.Series) -> float:
        """Calculate final confidence score with adjustments"""
        try:
            final_confidence = base_confidence
            
            # Adjust based on signal strength
            signal_strength = abs(regime_score)
            if signal_strength > 1.5:
                final_confidence *= 1.2  # Boost confidence for strong signals
            elif signal_strength < 0.3:
                final_confidence *= 0.8  # Reduce confidence for weak signals
            
            # Adjust based on indicator agreement
            component_signals = [signals.get(col, 0) for col in signals.index if col.endswith('_signal')]
            if len(component_signals) > 1:
                signal_agreement = 1.0 - np.std(component_signals) / (np.mean(np.abs(component_signals)) + 1e-6)
                final_confidence *= (0.5 + 0.5 * signal_agreement)
            
            # Adjust based on historical consistency
            if len(self.regime_history) > 0:
                recent_regimes = self.regime_history[-5:]  # Last 5 classifications
                if len(set(r.regime_type for r in recent_regimes)) == 1:
                    final_confidence *= 1.1  # Boost for consistency
            
            return np.clip(final_confidence, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating final confidence: {e}")
            return base_confidence
    
    def _extract_component_scores(self, signals: pd.Series) -> Dict[str, float]:
        """Extract individual indicator scores"""
        component_scores = {}
        
        for col in signals.index:
            if col.endswith('_signal'):
                indicator_id = col.replace('_signal', '')
                component_scores[indicator_id] = signals[col]
        
        return component_scores
    
    def _extract_timeframe_scores(self, signals: pd.Series) -> Dict[int, float]:
        """Extract timeframe-specific scores"""
        timeframe_scores = {}
        
        # Look for timeframe-specific columns
        for col in signals.index:
            if 'timeframe' in col.lower() and col.endswith('_signal'):
                try:
                    # Extract timeframe from column name
                    parts = col.split('_')
                    for part in parts:
                        if part.isdigit():
                            timeframe = int(part)
                            timeframe_scores[timeframe] = signals[col]
                            break
                except:
                    continue
        
        return timeframe_scores
    
    def _post_process_classifications(self, classifications: List[RegimeClassification]) -> List[RegimeClassification]:
        """Apply post-processing to classifications"""
        try:
            if len(classifications) < 2:
                return classifications
            
            # Apply transition detection
            classifications = self._detect_transitions(classifications)
            
            # Apply smoothing if enabled
            if self.config.regime_smoothing > 1:
                classifications = self._apply_smoothing(classifications)
            
            # Update history
            self.regime_history.extend(classifications)
            
            # Keep only recent history
            max_history = 1000
            if len(self.regime_history) > max_history:
                self.regime_history = self.regime_history[-max_history:]
            
            return classifications
            
        except Exception as e:
            logger.error(f"Error in post-processing: {e}")
            return classifications
    
    def _detect_transitions(self, classifications: List[RegimeClassification]) -> List[RegimeClassification]:
        """Detect and mark regime transitions"""
        try:
            for i in range(1, len(classifications)):
                current = classifications[i]
                previous = classifications[i-1]
                
                # Check for regime change
                if current.regime_type != previous.regime_type:
                    # Calculate transition confidence
                    score_diff = abs(current.regime_score - previous.regime_score)
                    
                    if score_diff < self.thresholds['transition_threshold']:
                        # Mark as transition if change is small
                        current.regime_type = RegimeType.TRANSITION
                        current.metadata['transition_from'] = previous.regime_type.value
                        current.metadata['transition_to'] = current.regime_type.value
            
            return classifications
            
        except Exception as e:
            logger.error(f"Error detecting transitions: {e}")
            return classifications
    
    def _apply_smoothing(self, classifications: List[RegimeClassification]) -> List[RegimeClassification]:
        """Apply smoothing to reduce noise in classifications"""
        try:
            if len(classifications) < self.config.regime_smoothing:
                return classifications
            
            smoothed = []
            window = self.config.regime_smoothing
            
            for i in range(len(classifications)):
                start_idx = max(0, i - window // 2)
                end_idx = min(len(classifications), i + window // 2 + 1)
                
                window_classifications = classifications[start_idx:end_idx]
                
                # Find most common regime in window
                regime_counts = {}
                for c in window_classifications:
                    regime = c.regime_type
                    regime_counts[regime] = regime_counts.get(regime, 0) + 1
                
                most_common_regime = max(regime_counts, key=regime_counts.get)
                
                # Create smoothed classification
                smoothed_classification = RegimeClassification(
                    timestamp=classifications[i].timestamp,
                    symbol=classifications[i].symbol,
                    regime_type=most_common_regime,
                    regime_score=classifications[i].regime_score,
                    confidence=classifications[i].confidence,
                    component_scores=classifications[i].component_scores,
                    timeframe_scores=classifications[i].timeframe_scores,
                    metadata={
                        **classifications[i].metadata,
                        'smoothed': True,
                        'original_regime': classifications[i].regime_type.value
                    }
                )
                
                smoothed.append(smoothed_classification)
            
            return smoothed
            
        except Exception as e:
            logger.error(f"Error applying smoothing: {e}")
            return classifications
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get statistics about regime classifications"""
        try:
            if not self.regime_history:
                return {}
            
            # Count regime types
            regime_counts = {}
            for classification in self.regime_history:
                regime = classification.regime_type.value
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            # Calculate average confidence
            avg_confidence = np.mean([c.confidence for c in self.regime_history])
            
            # Calculate regime durations
            regime_durations = self._calculate_regime_durations()
            
            return {
                'total_classifications': len(self.regime_history),
                'regime_distribution': regime_counts,
                'average_confidence': avg_confidence,
                'regime_durations': regime_durations,
                'last_update': self.regime_history[-1].timestamp.isoformat() if self.regime_history else None
            }
            
        except Exception as e:
            logger.error(f"Error getting regime statistics: {e}")
            return {}
    
    def _calculate_regime_durations(self) -> Dict[str, float]:
        """Calculate average duration for each regime type"""
        try:
            durations = {}
            current_regime = None
            current_start = None
            
            for classification in self.regime_history:
                if current_regime != classification.regime_type:
                    # End of previous regime
                    if current_regime and current_start:
                        duration = (classification.timestamp - current_start).total_seconds() / 60  # minutes
                        regime_name = current_regime.value
                        if regime_name not in durations:
                            durations[regime_name] = []
                        durations[regime_name].append(duration)
                    
                    # Start of new regime
                    current_regime = classification.regime_type
                    current_start = classification.timestamp
            
            # Calculate averages
            avg_durations = {}
            for regime, duration_list in durations.items():
                avg_durations[regime] = np.mean(duration_list)
            
            return avg_durations
            
        except Exception as e:
            logger.error(f"Error calculating regime durations: {e}")
            return {}
