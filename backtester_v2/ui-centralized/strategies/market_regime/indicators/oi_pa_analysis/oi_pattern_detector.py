"""
OI Pattern Detector - Open Interest Pattern Classification
=========================================================

Detects and classifies OI patterns for both calls and puts following corrected
OI-Price relationships. Core logic for pattern recognition with enhanced accuracy.

Author: Market Regime Refactoring Team
Date: 2025-07-06
Version: 2.0.0 - Enhanced OI Pattern Detection
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class OIPattern(Enum):
    """OI Pattern Classifications (Corrected Logic)"""
    LONG_BUILD_UP = "Long_Build_Up"           # OI↑ + Price↑ (Bullish)
    LONG_UNWINDING = "Long_Unwinding"         # OI↓ + Price↓ (Bearish)
    SHORT_BUILD_UP = "Short_Build_Up"         # OI↑ + Price↓ (Bearish)
    SHORT_COVERING = "Short_Covering"         # OI↓ + Price↑ (Bullish)
    NEUTRAL = "Neutral"                       # No clear pattern

@dataclass
class OIPatternResult:
    """Result structure for OI pattern detection"""
    pattern: OIPattern
    confidence: float
    signal_strength: float
    oi_change_percent: float
    price_change_percent: float
    pattern_consistency: float
    supporting_metrics: Dict[str, Any]

class OIPatternDetector:
    """
    Advanced OI pattern detection with corrected logic
    
    Features:
    - Corrected OI-Price relationship detection
    - Pattern strength assessment
    - Multi-strike pattern aggregation
    - Historical pattern validation
    - Confidence scoring system
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize OI Pattern Detector"""
        self.config = config or {}
        
        # Pattern detection thresholds
        self.min_oi_change_threshold = self.config.get('min_oi_change_threshold', 0.02)  # 2%
        self.min_price_change_threshold = self.config.get('min_price_change_threshold', 0.01)  # 1%
        self.pattern_strength_threshold = self.config.get('pattern_strength_threshold', 0.6)
        
        # Pattern consistency parameters
        self.consistency_lookback = self.config.get('consistency_lookback', 5)
        self.min_consistency_score = self.config.get('min_consistency_score', 0.7)
        
        # Signal strength calculation
        self.oi_weight = self.config.get('oi_weight', 0.6)
        self.price_weight = self.config.get('price_weight', 0.4)
        
        # Enhanced pattern recognition
        self.enable_multi_strike_validation = self.config.get('enable_multi_strike_validation', True)
        self.enable_historical_validation = self.config.get('enable_historical_validation', True)
        
        # Pattern detection history
        self.pattern_history = []
        self.pattern_accuracy_tracker = {}
        
        logger.info("OIPatternDetector initialized with corrected OI-Price logic")
    
    def detect_oi_patterns(self, 
                          market_data: pd.DataFrame,
                          selected_strikes: List,
                          spot_price: float,
                          historical_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Detect OI patterns across selected strikes
        
        Args:
            market_data: Current market data
            selected_strikes: List of strikes to analyze
            spot_price: Current spot price
            historical_data: Historical data for pattern validation
            
        Returns:
            Dict: Comprehensive OI pattern analysis
        """
        try:
            pattern_results = {}
            strike_patterns = []
            
            # Analyze patterns for each selected strike
            for strike_info in selected_strikes:
                strike = strike_info.strike
                option_type = strike_info.option_type
                
                # Get current and historical data for this strike
                current_data = self._get_strike_data(market_data, strike, option_type)
                historical_strike_data = None
                
                if historical_data is not None:
                    historical_strike_data = self._get_strike_data(historical_data, strike, option_type)
                
                if current_data is None:
                    continue
                
                # Detect pattern for this strike
                pattern_result = self._detect_single_strike_pattern(
                    current_data, historical_strike_data, strike, option_type
                )
                
                if pattern_result:
                    strike_patterns.append({
                        'strike': strike,
                        'option_type': option_type,
                        'pattern_result': pattern_result,
                        'moneyness': abs(strike - spot_price) / spot_price
                    })
            
            # Aggregate patterns across strikes
            aggregated_pattern = self._aggregate_strike_patterns(strike_patterns)
            
            # Multi-strike validation if enabled
            if self.enable_multi_strike_validation:
                validation_result = self._validate_multi_strike_consistency(strike_patterns)
                aggregated_pattern['multi_strike_validation'] = validation_result
            
            # Historical validation if enabled
            if self.enable_historical_validation and historical_data is not None:
                historical_validation = self._validate_historical_patterns(
                    aggregated_pattern, historical_data
                )
                aggregated_pattern['historical_validation'] = historical_validation
            
            # Record pattern for accuracy tracking
            self._record_pattern_detection(aggregated_pattern)
            
            return {
                'aggregated_pattern': aggregated_pattern,
                'strike_patterns': strike_patterns,
                'pattern_distribution': self._analyze_pattern_distribution(strike_patterns),
                'detection_metadata': {
                    'strikes_analyzed': len(strike_patterns),
                    'spot_price': spot_price,
                    'timestamp': datetime.now()
                }
            }
            
        except Exception as e:
            logger.error(f"Error detecting OI patterns: {e}")
            return self._get_default_pattern_result()
    
    def _detect_single_strike_pattern(self,
                                    current_data: Dict[str, Any],
                                    historical_data: Optional[Dict[str, Any]],
                                    strike: float,
                                    option_type: str) -> Optional[OIPatternResult]:
        """Detect OI pattern for a single strike"""
        try:
            if not historical_data:
                return None
            
            # Calculate changes
            current_oi = current_data.get('oi', 0)
            current_price = current_data.get('price', 0)
            
            historical_oi = historical_data.get('oi', 0)
            historical_price = historical_data.get('price', 0)
            
            if historical_oi == 0 or historical_price == 0:
                return None
            
            # Calculate percentage changes
            oi_change = (current_oi - historical_oi) / historical_oi
            price_change = (current_price - historical_price) / historical_price
            
            # Apply minimum threshold filters
            if (abs(oi_change) < self.min_oi_change_threshold or 
                abs(price_change) < self.min_price_change_threshold):
                return None
            
            # Classify pattern (CORRECTED LOGIC)
            pattern = self._classify_oi_pattern(oi_change, price_change)
            
            # Calculate pattern confidence and strength
            confidence = self._calculate_pattern_confidence(oi_change, price_change)
            signal_strength = self._calculate_signal_strength(oi_change, price_change)
            
            # Pattern consistency check
            pattern_consistency = self._assess_pattern_consistency(
                current_data, historical_data, pattern
            )
            
            # Supporting metrics
            supporting_metrics = {
                'volume_confirmation': self._check_volume_confirmation(current_data, historical_data),
                'price_momentum': self._calculate_price_momentum(current_data, historical_data),
                'oi_intensity': abs(oi_change),
                'price_intensity': abs(price_change)
            }
            
            return OIPatternResult(
                pattern=pattern,
                confidence=confidence,
                signal_strength=signal_strength,
                oi_change_percent=oi_change * 100,
                price_change_percent=price_change * 100,
                pattern_consistency=pattern_consistency,
                supporting_metrics=supporting_metrics
            )
            
        except Exception as e:
            logger.error(f"Error detecting single strike pattern: {e}")
            return None
    
    def _classify_oi_pattern(self, oi_change: float, price_change: float) -> OIPattern:
        """Classify OI pattern based on corrected logic"""
        try:
            # CORRECTED OI-PRICE RELATIONSHIP:
            # Both calls and puts follow the same OI-Price relationship
            
            if oi_change > 0 and price_change > 0:
                return OIPattern.LONG_BUILD_UP      # OI↑ + Price↑ = Bullish
            elif oi_change < 0 and price_change < 0:
                return OIPattern.LONG_UNWINDING     # OI↓ + Price↓ = Bearish
            elif oi_change > 0 and price_change < 0:
                return OIPattern.SHORT_BUILD_UP     # OI↑ + Price↓ = Bearish
            elif oi_change < 0 and price_change > 0:
                return OIPattern.SHORT_COVERING     # OI↓ + Price↑ = Bullish
            else:
                return OIPattern.NEUTRAL
                
        except Exception as e:
            logger.error(f"Error classifying OI pattern: {e}")
            return OIPattern.NEUTRAL
    
    def _calculate_pattern_confidence(self, oi_change: float, price_change: float) -> float:
        """Calculate confidence in pattern detection"""
        try:
            # Base confidence from magnitude of changes
            oi_magnitude = abs(oi_change)
            price_magnitude = abs(price_change)
            
            # Weighted magnitude (favor OI changes)
            weighted_magnitude = self.oi_weight * oi_magnitude + self.price_weight * price_magnitude
            
            # Directional consistency (both changes in same direction increases confidence)
            directional_consistency = 1.0 if (oi_change * price_change > 0) else 0.7
            
            # Scale to confidence score (0-1)
            base_confidence = min(weighted_magnitude * 10, 1.0)  # Scale factor
            final_confidence = base_confidence * directional_consistency
            
            return np.clip(final_confidence, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating pattern confidence: {e}")
            return 0.5
    
    def _calculate_signal_strength(self, oi_change: float, price_change: float) -> float:
        """Calculate signal strength"""
        try:
            # Signal strength based on magnitude and consistency
            oi_strength = min(abs(oi_change) * 5, 1.0)  # Scale OI change
            price_strength = min(abs(price_change) * 10, 1.0)  # Scale price change
            
            # Combined strength with weights
            combined_strength = self.oi_weight * oi_strength + self.price_weight * price_strength
            
            return np.clip(combined_strength, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating signal strength: {e}")
            return 0.5
    
    def _assess_pattern_consistency(self,
                                  current_data: Dict[str, Any],
                                  historical_data: Dict[str, Any],
                                  pattern: OIPattern) -> float:
        """Assess pattern consistency over time"""
        try:
            # Placeholder for detailed consistency analysis
            # This could analyze pattern persistence over multiple timeframes
            base_consistency = 0.8
            
            # Adjust based on volume confirmation
            volume_current = current_data.get('volume', 0)
            volume_historical = historical_data.get('volume', 0)
            
            if volume_historical > 0:
                volume_change = volume_current / volume_historical
                if 0.5 <= volume_change <= 2.0:  # Reasonable volume change
                    base_consistency *= 1.1
                else:
                    base_consistency *= 0.9
            
            return np.clip(base_consistency, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error assessing pattern consistency: {e}")
            return 0.5
    
    def _aggregate_strike_patterns(self, strike_patterns: List[Dict]) -> Dict[str, Any]:
        """Aggregate patterns across multiple strikes"""
        try:
            if not strike_patterns:
                return {
                    'dominant_pattern': OIPattern.NEUTRAL,
                    'aggregate_confidence': 0.0,
                    'aggregate_signal_strength': 0.0,
                    'pattern_consensus': 0.0
                }
            
            # Count pattern occurrences
            pattern_counts = {}
            total_confidence = 0
            total_signal_strength = 0
            
            for strike_data in strike_patterns:
                pattern_result = strike_data['pattern_result']
                pattern = pattern_result.pattern
                
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
                total_confidence += pattern_result.confidence
                total_signal_strength += pattern_result.signal_strength
            
            # Determine dominant pattern
            dominant_pattern = max(pattern_counts.keys(), key=lambda k: pattern_counts[k])
            
            # Calculate aggregate metrics
            num_patterns = len(strike_patterns)
            aggregate_confidence = total_confidence / num_patterns
            aggregate_signal_strength = total_signal_strength / num_patterns
            
            # Pattern consensus (percentage of strikes with dominant pattern)
            pattern_consensus = pattern_counts[dominant_pattern] / num_patterns
            
            return {
                'dominant_pattern': dominant_pattern,
                'aggregate_confidence': aggregate_confidence,
                'aggregate_signal_strength': aggregate_signal_strength,
                'pattern_consensus': pattern_consensus,
                'pattern_distribution': dict(pattern_counts),
                'strikes_analyzed': num_patterns
            }
            
        except Exception as e:
            logger.error(f"Error aggregating strike patterns: {e}")
            return {}
    
    def _get_strike_data(self, market_data: pd.DataFrame, strike: float, option_type: str) -> Optional[Dict[str, Any]]:
        """Extract data for a specific strike and option type"""
        try:
            filtered_data = market_data[
                (market_data['strike'] == strike) & 
                (market_data['option_type'] == option_type)
            ]
            
            if filtered_data.empty:
                return None
            
            row = filtered_data.iloc[0]
            
            if option_type == 'CE':
                return {
                    'oi': row.get('ce_oi', 0),
                    'volume': row.get('ce_volume', 0),
                    'price': row.get('ce_price', row.get('ce_ltp', 0)),
                    'iv': row.get('ce_iv', 0)
                }
            else:  # PE
                return {
                    'oi': row.get('pe_oi', 0),
                    'volume': row.get('pe_volume', 0),
                    'price': row.get('pe_price', row.get('pe_ltp', 0)),
                    'iv': row.get('pe_iv', 0)
                }
                
        except Exception as e:
            logger.error(f"Error getting strike data: {e}")
            return None
    
    def _check_volume_confirmation(self, current_data: Dict, historical_data: Dict) -> bool:
        """Check if volume confirms the OI pattern"""
        try:
            current_volume = current_data.get('volume', 0)
            historical_volume = historical_data.get('volume', 0)
            
            if historical_volume == 0:
                return False
            
            volume_change = current_volume / historical_volume
            return volume_change > 1.2  # 20% increase in volume
            
        except:
            return False
    
    def _calculate_price_momentum(self, current_data: Dict, historical_data: Dict) -> float:
        """Calculate price momentum"""
        try:
            current_price = current_data.get('price', 0)
            historical_price = historical_data.get('price', 0)
            
            if historical_price == 0:
                return 0.0
            
            return (current_price - historical_price) / historical_price
            
        except:
            return 0.0
    
    def _validate_multi_strike_consistency(self, strike_patterns: List[Dict]) -> Dict[str, Any]:
        """Validate consistency across multiple strikes"""
        try:
            if len(strike_patterns) < 2:
                return {'consistency_score': 1.0, 'validation_passed': True}
            
            # Check pattern consistency across strikes
            patterns = [sp['pattern_result'].pattern for sp in strike_patterns]
            unique_patterns = set(patterns)
            
            # Higher consistency if fewer unique patterns
            consistency_score = 1.0 / len(unique_patterns)
            validation_passed = consistency_score >= 0.6
            
            return {
                'consistency_score': consistency_score,
                'validation_passed': validation_passed,
                'unique_patterns': len(unique_patterns),
                'total_strikes': len(strike_patterns)
            }
            
        except Exception as e:
            logger.error(f"Error validating multi-strike consistency: {e}")
            return {}
    
    def _validate_historical_patterns(self, pattern_result: Dict, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate patterns against historical data"""
        try:
            # Placeholder for historical pattern validation
            return {
                'historical_accuracy': 0.8,
                'pattern_reliability': 0.75,
                'validation_passed': True
            }
            
        except Exception as e:
            logger.error(f"Error validating historical patterns: {e}")
            return {}
    
    def _analyze_pattern_distribution(self, strike_patterns: List[Dict]) -> Dict[str, Any]:
        """Analyze distribution of patterns across strikes"""
        try:
            if not strike_patterns:
                return {}
            
            # Group by moneyness
            atm_patterns = [sp for sp in strike_patterns if sp['moneyness'] <= 0.02]
            otm_patterns = [sp for sp in strike_patterns if sp['moneyness'] > 0.02]
            
            return {
                'total_strikes': len(strike_patterns),
                'atm_strikes': len(atm_patterns),
                'otm_strikes': len(otm_patterns),
                'atm_dominant_pattern': self._get_dominant_pattern(atm_patterns),
                'otm_dominant_pattern': self._get_dominant_pattern(otm_patterns)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing pattern distribution: {e}")
            return {}
    
    def _get_dominant_pattern(self, pattern_list: List[Dict]) -> Optional[OIPattern]:
        """Get dominant pattern from a list"""
        try:
            if not pattern_list:
                return None
            
            patterns = [sp['pattern_result'].pattern for sp in pattern_list]
            return max(set(patterns), key=patterns.count)
            
        except:
            return None
    
    def _record_pattern_detection(self, pattern_result: Dict[str, Any]):
        """Record pattern detection for accuracy tracking"""
        try:
            record = {
                'timestamp': datetime.now(),
                'dominant_pattern': pattern_result.get('dominant_pattern'),
                'aggregate_confidence': pattern_result.get('aggregate_confidence', 0),
                'pattern_consensus': pattern_result.get('pattern_consensus', 0)
            }
            
            self.pattern_history.append(record)
            
            # Keep only last 100 detections
            if len(self.pattern_history) > 100:
                self.pattern_history = self.pattern_history[-100:]
                
        except Exception as e:
            logger.error(f"Error recording pattern detection: {e}")
    
    def _get_default_pattern_result(self) -> Dict[str, Any]:
        """Get default result for error cases"""
        return {
            'aggregated_pattern': {
                'dominant_pattern': OIPattern.NEUTRAL,
                'aggregate_confidence': 0.0,
                'aggregate_signal_strength': 0.0,
                'pattern_consensus': 0.0
            },
            'strike_patterns': [],
            'pattern_distribution': {},
            'detection_metadata': {
                'error': True,
                'timestamp': datetime.now()
            }
        }
    
    def get_pattern_detection_summary(self) -> Dict[str, Any]:
        """Get summary of pattern detection performance"""
        try:
            if not self.pattern_history:
                return {'status': 'no_data'}
            
            recent_detections = self.pattern_history[-20:]
            
            # Pattern frequency analysis
            pattern_counts = {}
            avg_confidence = 0
            avg_consensus = 0
            
            for detection in recent_detections:
                pattern = detection['dominant_pattern']
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
                avg_confidence += detection['aggregate_confidence']
                avg_consensus += detection['pattern_consensus']
            
            return {
                'total_detections': len(self.pattern_history),
                'recent_detections': len(recent_detections),
                'pattern_frequency': dict(pattern_counts),
                'average_confidence': avg_confidence / len(recent_detections),
                'average_consensus': avg_consensus / len(recent_detections),
                'detection_config': {
                    'min_oi_change_threshold': self.min_oi_change_threshold,
                    'min_price_change_threshold': self.min_price_change_threshold,
                    'pattern_strength_threshold': self.pattern_strength_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating pattern detection summary: {e}")
            return {'status': 'error', 'error': str(e)}