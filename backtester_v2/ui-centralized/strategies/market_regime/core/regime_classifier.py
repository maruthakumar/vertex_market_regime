"""
Market Regime Classifier - 18-Regime Classification System

This module implements the 18-regime classification system based on:
- Volatility: Low, Medium, High (3 levels)
- Trend: Bearish, Neutral, Bullish (3 levels)
- Structure: Ranging, Trending (2 levels)

Total: 3 × 3 × 2 = 18 regimes
"""

import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class VolatilityLevel(Enum):
    """Volatility levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class TrendDirection(Enum):
    """Trend directions"""
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    BULLISH = "BULLISH"


class MarketStructure(Enum):
    """Market structure types"""
    RANGING = "RANGING"
    TRENDING = "TRENDING"


class RegimeType(Enum):
    """18 Market Regime Types"""
    # Low Volatility Regimes
    LOW_VOLATILITY_BEARISH_RANGING = "LOW_VOLATILITY_BEARISH_RANGING"
    LOW_VOLATILITY_BEARISH_TRENDING = "LOW_VOLATILITY_BEARISH_TRENDING"
    LOW_VOLATILITY_NEUTRAL_RANGING = "LOW_VOLATILITY_NEUTRAL_RANGING"
    LOW_VOLATILITY_NEUTRAL_TRENDING = "LOW_VOLATILITY_NEUTRAL_TRENDING"
    LOW_VOLATILITY_BULLISH_RANGING = "LOW_VOLATILITY_BULLISH_RANGING"
    LOW_VOLATILITY_BULLISH_TRENDING = "LOW_VOLATILITY_BULLISH_TRENDING"
    
    # Medium Volatility Regimes
    MEDIUM_VOLATILITY_BEARISH_RANGING = "MEDIUM_VOLATILITY_BEARISH_RANGING"
    MEDIUM_VOLATILITY_BEARISH_TRENDING = "MEDIUM_VOLATILITY_BEARISH_TRENDING"
    MEDIUM_VOLATILITY_NEUTRAL_RANGING = "MEDIUM_VOLATILITY_NEUTRAL_RANGING"
    MEDIUM_VOLATILITY_NEUTRAL_TRENDING = "MEDIUM_VOLATILITY_NEUTRAL_TRENDING"
    MEDIUM_VOLATILITY_BULLISH_RANGING = "MEDIUM_VOLATILITY_BULLISH_RANGING"
    MEDIUM_VOLATILITY_BULLISH_TRENDING = "MEDIUM_VOLATILITY_BULLISH_TRENDING"
    
    # High Volatility Regimes
    HIGH_VOLATILITY_BEARISH_RANGING = "HIGH_VOLATILITY_BEARISH_RANGING"
    HIGH_VOLATILITY_BEARISH_TRENDING = "HIGH_VOLATILITY_BEARISH_TRENDING"
    HIGH_VOLATILITY_NEUTRAL_RANGING = "HIGH_VOLATILITY_NEUTRAL_RANGING"
    HIGH_VOLATILITY_NEUTRAL_TRENDING = "HIGH_VOLATILITY_NEUTRAL_TRENDING"
    HIGH_VOLATILITY_BULLISH_RANGING = "HIGH_VOLATILITY_BULLISH_RANGING"
    HIGH_VOLATILITY_BULLISH_TRENDING = "HIGH_VOLATILITY_BULLISH_TRENDING"


class RegimeClassifier:
    """
    Classifier for 18-regime market classification system
    
    This classifier takes detection results and classifies them into
    one of the 18 predefined market regimes.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the regime classifier
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize thresholds
        self._initialize_thresholds()
        
        # Initialize regime mappings
        self._initialize_regime_mappings()
        
        # Regime stability tracking
        self.regime_history = []
        self.max_history = config.get('regime_history_size', 10)
        
    def _initialize_thresholds(self):
        """Initialize classification thresholds"""
        # Volatility thresholds
        self.volatility_thresholds = self.config.get('volatility_thresholds', {
            'low': 0.01,
            'medium': 0.02,
            'high': 0.03
        })
        
        # Trend thresholds
        self.trend_thresholds = self.config.get('trend_thresholds', {
            'bearish': -0.005,
            'neutral_low': -0.002,
            'neutral_high': 0.002,
            'bullish': 0.005
        })
        
        # Structure thresholds
        self.structure_thresholds = self.config.get('structure_thresholds', {
            'ranging_threshold': 0.7,
            'trending_threshold': 0.3
        })
        
    def _initialize_regime_mappings(self):
        """Initialize regime characteristic mappings"""
        self.regime_characteristics = {}
        
        for regime in RegimeType:
            # Parse regime name to extract characteristics
            parts = regime.value.split('_')
            volatility = parts[0]
            trend = parts[2]
            structure = parts[3]
            
            self.regime_characteristics[regime] = {
                'volatility': volatility,
                'trend': trend,
                'structure': structure,
                'risk_level': self._determine_risk_level(volatility, trend, structure),
                'suitable_strategies': self._determine_suitable_strategies(regime)
            }
    
    def classify(self, detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify market regime based on detection results
        
        Args:
            detection_result: Results from regime detector
            
        Returns:
            Dictionary with regime classification and metadata
        """
        try:
            # Extract components
            volatility = self._classify_volatility(detection_result)
            trend = self._classify_trend(detection_result)
            structure = self._classify_structure(detection_result)
            
            # Determine regime
            regime = self._determine_regime(volatility, trend, structure)
            
            # Calculate confidence
            confidence = self._calculate_confidence(detection_result, regime)
            
            # Check regime stability
            stability = self._check_regime_stability(regime)
            
            # Get alternative regimes
            alternatives = self._get_alternative_regimes(detection_result, regime)
            
            result = {
                'regime': regime.value,
                'components': {
                    'volatility': volatility.value,
                    'trend': trend.value,
                    'structure': structure.value
                },
                'confidence': confidence,
                'stability': stability,
                'alternatives': alternatives,
                'characteristics': self.regime_characteristics[regime],
                'timestamp': detection_result.get('timestamp')
            }
            
            # Update history
            self._update_regime_history(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error classifying regime: {e}")
            raise
    
    def _classify_volatility(self, data: Dict[str, Any]) -> VolatilityLevel:
        """Classify volatility level"""
        volatility_value = data.get('volatility', {}).get('value', 0.015)
        
        if volatility_value < self.volatility_thresholds['low']:
            return VolatilityLevel.LOW
        elif volatility_value < self.volatility_thresholds['medium']:
            return VolatilityLevel.MEDIUM
        else:
            return VolatilityLevel.HIGH
    
    def _classify_trend(self, data: Dict[str, Any]) -> TrendDirection:
        """Classify trend direction"""
        trend_value = data.get('trend', {}).get('value', 0.0)
        
        if trend_value < self.trend_thresholds['bearish']:
            return TrendDirection.BEARISH
        elif trend_value > self.trend_thresholds['bullish']:
            return TrendDirection.BULLISH
        else:
            return TrendDirection.NEUTRAL
    
    def _classify_structure(self, data: Dict[str, Any]) -> MarketStructure:
        """Classify market structure"""
        structure_score = data.get('structure', {}).get('score', 0.5)
        
        if structure_score > self.structure_thresholds['ranging_threshold']:
            return MarketStructure.RANGING
        else:
            return MarketStructure.TRENDING
    
    def _determine_regime(self,
                         volatility: VolatilityLevel,
                         trend: TrendDirection,
                         structure: MarketStructure) -> RegimeType:
        """Determine regime type from components"""
        regime_name = f"{volatility.value}_VOLATILITY_{trend.value}_{structure.value}"
        
        try:
            return RegimeType(regime_name)
        except ValueError:
            self.logger.error(f"Invalid regime combination: {regime_name}")
            # Return a default regime
            return RegimeType.MEDIUM_VOLATILITY_NEUTRAL_RANGING
    
    def _calculate_confidence(self, 
                            detection_result: Dict[str, Any],
                            regime: RegimeType) -> float:
        """Calculate confidence in regime classification"""
        confidence_components = []
        
        # Volatility confidence
        vol_conf = detection_result.get('volatility', {}).get('confidence', 0.5)
        confidence_components.append(vol_conf)
        
        # Trend confidence
        trend_conf = detection_result.get('trend', {}).get('confidence', 0.5)
        confidence_components.append(trend_conf)
        
        # Structure confidence
        struct_conf = detection_result.get('structure', {}).get('confidence', 0.5)
        confidence_components.append(struct_conf)
        
        # Indicator agreement
        indicator_agreement = detection_result.get('indicator_agreement', 0.5)
        confidence_components.append(indicator_agreement)
        
        # Calculate weighted average
        base_confidence = np.mean(confidence_components)
        
        # Adjust for regime stability
        if len(self.regime_history) > 0:
            last_regime = self.regime_history[-1].get('regime')
            if last_regime == regime.value:
                base_confidence *= 1.1  # Boost confidence for stable regime
            else:
                base_confidence *= 0.9  # Reduce confidence for regime change
        
        return min(base_confidence, 1.0)
    
    def _check_regime_stability(self, current_regime: RegimeType) -> Dict[str, Any]:
        """Check regime stability"""
        if len(self.regime_history) < 2:
            return {
                'is_stable': False,
                'duration': 0,
                'changes': 0
            }
        
        # Count consecutive same regimes
        consecutive_count = 0
        for i in range(len(self.regime_history) - 1, -1, -1):
            if self.regime_history[i].get('regime') == current_regime.value:
                consecutive_count += 1
            else:
                break
        
        # Count regime changes in history
        changes = 0
        for i in range(1, len(self.regime_history)):
            if self.regime_history[i].get('regime') != self.regime_history[i-1].get('regime'):
                changes += 1
        
        return {
            'is_stable': consecutive_count >= 3,
            'duration': consecutive_count,
            'changes': changes,
            'change_rate': changes / len(self.regime_history)
        }
    
    def _get_alternative_regimes(self,
                               detection_result: Dict[str, Any],
                               primary_regime: RegimeType) -> List[Dict[str, Any]]:
        """Get alternative regime possibilities"""
        alternatives = []
        
        # Get component probabilities
        vol_probs = detection_result.get('volatility', {}).get('probabilities', {})
        trend_probs = detection_result.get('trend', {}).get('probabilities', {})
        struct_probs = detection_result.get('structure', {}).get('probabilities', {})
        
        # Calculate alternatives based on probabilities
        for vol in VolatilityLevel:
            for trend in TrendDirection:
                for struct in MarketStructure:
                    regime = self._determine_regime(vol, trend, struct)
                    if regime != primary_regime:
                        prob = (vol_probs.get(vol.value, 0.1) * 
                               trend_probs.get(trend.value, 0.1) * 
                               struct_probs.get(struct.value, 0.5))
                        
                        if prob > 0.1:  # Only include significant alternatives
                            alternatives.append({
                                'regime': regime.value,
                                'probability': prob
                            })
        
        # Sort by probability and return top 3
        alternatives.sort(key=lambda x: x['probability'], reverse=True)
        return alternatives[:3]
    
    def _determine_risk_level(self, 
                            volatility: str, 
                            trend: str, 
                            structure: str) -> str:
        """Determine risk level for regime"""
        if volatility == "HIGH":
            return "HIGH"
        elif volatility == "MEDIUM" and structure == "TRENDING":
            return "MEDIUM-HIGH"
        elif volatility == "MEDIUM":
            return "MEDIUM"
        else:
            return "LOW"
    
    def _determine_suitable_strategies(self, regime: RegimeType) -> List[str]:
        """Determine suitable strategies for regime"""
        suitable = []
        
        regime_name = regime.value
        
        # High volatility strategies
        if "HIGH_VOLATILITY" in regime_name:
            suitable.extend(["STRADDLE", "STRANGLE", "IRON_CONDOR"])
        
        # Trending strategies
        if "TRENDING" in regime_name:
            if "BULLISH" in regime_name:
                suitable.extend(["BULL_SPREAD", "CALL_RATIO"])
            elif "BEARISH" in regime_name:
                suitable.extend(["BEAR_SPREAD", "PUT_RATIO"])
        
        # Ranging strategies
        if "RANGING" in regime_name:
            suitable.extend(["IRON_BUTTERFLY", "SHORT_STRADDLE"])
        
        # Low volatility strategies
        if "LOW_VOLATILITY" in regime_name:
            suitable.extend(["CALENDAR_SPREAD", "DIAGONAL_SPREAD"])
        
        return list(set(suitable))  # Remove duplicates
    
    def _update_regime_history(self, result: Dict[str, Any]):
        """Update regime history"""
        self.regime_history.append(result)
        
        # Keep only recent history
        if len(self.regime_history) > self.max_history:
            self.regime_history.pop(0)
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get statistics about regime history"""
        if not self.regime_history:
            return {'available': False}
        
        regime_counts = {}
        for entry in self.regime_history:
            regime = entry.get('regime')
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        return {
            'available': True,
            'total_observations': len(self.regime_history),
            'regime_distribution': regime_counts,
            'most_common': max(regime_counts, key=regime_counts.get),
            'stability_score': self._calculate_stability_score()
        }
    
    def _calculate_stability_score(self) -> float:
        """Calculate overall stability score"""
        if len(self.regime_history) < 2:
            return 0.0
        
        changes = 0
        for i in range(1, len(self.regime_history)):
            if self.regime_history[i].get('regime') != self.regime_history[i-1].get('regime'):
                changes += 1
        
        return 1.0 - (changes / (len(self.regime_history) - 1))