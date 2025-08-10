"""
Divergence Detector - Multi-Type Divergence Detection System
===========================================================

Detects 5 types of divergences in options market data to identify
potential regime changes and institutional activity patterns.

Author: Market Regime Refactoring Team
Date: 2025-07-06
Version: 2.0.0 - Enhanced Divergence Detection
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)

class DivergenceType(Enum):
    """Types of divergence detection"""
    PATTERN_DIVERGENCE = "Pattern_Divergence"                    # Conflicting OI patterns
    OI_PRICE_DIVERGENCE = "OI_Price_Divergence"                 # OI vs Price direction conflict
    CALL_PUT_DIVERGENCE = "Call_Put_Divergence"                 # CE vs PE flow divergence
    INSTITUTIONAL_RETAIL_DIVERGENCE = "Institutional_Retail_Divergence"  # Different flow patterns
    CROSS_STRIKE_DIVERGENCE = "Cross_Strike_Divergence"         # ATM vs OTM divergence

@dataclass
class DivergenceResult:
    """Result structure for divergence detection"""
    divergence_type: DivergenceType
    strength: float
    confidence: float
    supporting_evidence: Dict[str, Any]
    time_detected: datetime
    persistence_score: float

class DivergenceDetector:
    """
    Comprehensive divergence detection system
    
    Features:
    - 5 distinct divergence types
    - Strength and confidence scoring
    - Persistence tracking
    - Multi-timeframe validation
    - Historical pattern comparison
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Divergence Detector"""
        self.config = config or {}
        
        # Divergence detection thresholds
        self.min_divergence_strength = self.config.get('min_divergence_strength', 0.3)
        self.min_confidence_threshold = self.config.get('min_confidence_threshold', 0.6)
        
        # Pattern divergence parameters
        self.pattern_similarity_threshold = self.config.get('pattern_similarity_threshold', 0.7)
        
        # OI-Price divergence parameters
        self.correlation_threshold = self.config.get('correlation_threshold', -0.3)  # Negative correlation
        
        # Call-Put divergence parameters
        self.flow_divergence_threshold = self.config.get('flow_divergence_threshold', 0.4)
        
        # Institutional-Retail divergence parameters
        self.size_threshold = self.config.get('institutional_size_threshold', 10000)
        self.flow_difference_threshold = self.config.get('flow_difference_threshold', 0.5)
        
        # Cross-strike divergence parameters
        self.moneyness_threshold = self.config.get('moneyness_threshold', 0.02)  # 2% for ATM
        
        # Persistence tracking
        self.persistence_memory = self.config.get('persistence_memory', 5)  # Track last 5 periods
        
        # Divergence history
        self.divergence_history = []
        self.persistence_tracker = {div_type: [] for div_type in DivergenceType}
        
        logger.info("DivergenceDetector initialized with 5 divergence types")
    
    def detect_all_divergences(self, 
                             market_data: pd.DataFrame,
                             pattern_results: Dict[str, Any],
                             flow_analysis: Dict[str, Any],
                             spot_price: float,
                             historical_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Detect all types of divergences
        
        Args:
            market_data: Current market data
            pattern_results: OI pattern detection results
            flow_analysis: Volume flow analysis results
            spot_price: Current spot price
            historical_data: Historical data for comparison
            
        Returns:
            Dict: Comprehensive divergence analysis
        """
        try:
            divergences_detected = []
            divergence_scores = {}
            
            # 1. Pattern Divergence Detection
            pattern_divergence = self._detect_pattern_divergence(pattern_results)
            if pattern_divergence:
                divergences_detected.append(pattern_divergence)
                divergence_scores[DivergenceType.PATTERN_DIVERGENCE] = pattern_divergence.strength
            
            # 2. OI-Price Divergence Detection
            oi_price_divergence = self._detect_oi_price_divergence(market_data, historical_data)
            if oi_price_divergence:
                divergences_detected.append(oi_price_divergence)
                divergence_scores[DivergenceType.OI_PRICE_DIVERGENCE] = oi_price_divergence.strength
            
            # 3. Call-Put Divergence Detection
            call_put_divergence = self._detect_call_put_divergence(market_data, flow_analysis)
            if call_put_divergence:
                divergences_detected.append(call_put_divergence)
                divergence_scores[DivergenceType.CALL_PUT_DIVERGENCE] = call_put_divergence.strength
            
            # 4. Institutional-Retail Divergence Detection
            inst_retail_divergence = self._detect_institutional_retail_divergence(flow_analysis)
            if inst_retail_divergence:
                divergences_detected.append(inst_retail_divergence)
                divergence_scores[DivergenceType.INSTITUTIONAL_RETAIL_DIVERGENCE] = inst_retail_divergence.strength
            
            # 5. Cross-Strike Divergence Detection
            cross_strike_divergence = self._detect_cross_strike_divergence(market_data, spot_price)
            if cross_strike_divergence:
                divergences_detected.append(cross_strike_divergence)
                divergence_scores[DivergenceType.CROSS_STRIKE_DIVERGENCE] = cross_strike_divergence.strength
            
            # Update persistence tracking
            self._update_persistence_tracking(divergences_detected)
            
            # Calculate overall divergence metrics
            overall_metrics = self._calculate_overall_divergence_metrics(
                divergences_detected, divergence_scores
            )
            
            return {
                'divergences_detected': divergences_detected,
                'divergence_scores': divergence_scores,
                'overall_metrics': overall_metrics,
                'persistence_analysis': self._analyze_divergence_persistence(),
                'detection_metadata': {
                    'total_divergences': len(divergences_detected),
                    'strongest_divergence': self._get_strongest_divergence(divergences_detected),
                    'timestamp': datetime.now()
                }
            }
            
        except Exception as e:
            logger.error(f"Error detecting divergences: {e}")
            return self._get_default_divergence_result()
    
    def _detect_pattern_divergence(self, pattern_results: Dict[str, Any]) -> Optional[DivergenceResult]:
        """Detect pattern divergence across different analysis methods"""
        try:
            strike_patterns = pattern_results.get('strike_patterns', [])
            if len(strike_patterns) < 2:
                return None
            
            # Analyze pattern consistency
            patterns = [sp['pattern_result'].pattern for sp in strike_patterns]
            unique_patterns = set(patterns)
            
            # Divergence exists if patterns are inconsistent
            if len(unique_patterns) > 2:  # More than 2 different patterns
                pattern_consistency = len(unique_patterns) / len(patterns)
                divergence_strength = min(pattern_consistency * 2, 1.0)
                
                if divergence_strength >= self.min_divergence_strength:
                    confidence = self._calculate_pattern_divergence_confidence(strike_patterns)
                    
                    return DivergenceResult(
                        divergence_type=DivergenceType.PATTERN_DIVERGENCE,
                        strength=divergence_strength,
                        confidence=confidence,
                        supporting_evidence={
                            'unique_patterns': len(unique_patterns),
                            'total_patterns': len(patterns),
                            'pattern_distribution': dict(zip(*np.unique(patterns, return_counts=True)))
                        },
                        time_detected=datetime.now(),
                        persistence_score=self._get_persistence_score(DivergenceType.PATTERN_DIVERGENCE)
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting pattern divergence: {e}")
            return None
    
    def _detect_oi_price_divergence(self, 
                                  market_data: pd.DataFrame,
                                  historical_data: Optional[pd.DataFrame]) -> Optional[DivergenceResult]:
        """Detect OI-Price divergence"""
        try:
            if historical_data is None:
                return None
            
            # Calculate OI and price changes
            current_total_oi = market_data['ce_oi'].sum() + market_data['pe_oi'].sum()
            historical_total_oi = historical_data['ce_oi'].sum() + historical_data['pe_oi'].sum()
            
            current_avg_price = (market_data['ce_ltp'].mean() + market_data['pe_ltp'].mean()) / 2
            historical_avg_price = (historical_data['ce_ltp'].mean() + historical_data['pe_ltp'].mean()) / 2
            
            if historical_total_oi == 0 or historical_avg_price == 0:
                return None
            
            oi_change = (current_total_oi - historical_total_oi) / historical_total_oi
            price_change = (current_avg_price - historical_avg_price) / historical_avg_price
            
            # Check for divergence (negative correlation)
            if oi_change * price_change < 0 and abs(oi_change) > 0.05 and abs(price_change) > 0.02:
                # Calculate correlation over time if more data available
                correlation = oi_change * price_change  # Simplified correlation
                
                if correlation < self.correlation_threshold:
                    divergence_strength = min(abs(correlation) * 2, 1.0)
                    
                    if divergence_strength >= self.min_divergence_strength:
                        confidence = min(abs(oi_change) + abs(price_change), 1.0)
                        
                        return DivergenceResult(
                            divergence_type=DivergenceType.OI_PRICE_DIVERGENCE,
                            strength=divergence_strength,
                            confidence=confidence,
                            supporting_evidence={
                                'oi_change_percent': oi_change * 100,
                                'price_change_percent': price_change * 100,
                                'correlation': correlation,
                                'divergence_direction': 'oi_up_price_down' if oi_change > 0 else 'oi_down_price_up'
                            },
                            time_detected=datetime.now(),
                            persistence_score=self._get_persistence_score(DivergenceType.OI_PRICE_DIVERGENCE)
                        )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting OI-Price divergence: {e}")
            return None
    
    def _detect_call_put_divergence(self, 
                                  market_data: pd.DataFrame,
                                  flow_analysis: Dict[str, Any]) -> Optional[DivergenceResult]:
        """Detect Call-Put flow divergence"""
        try:
            # Calculate call and put flows
            call_flow = market_data['ce_oi'].sum() + market_data['ce_volume'].sum()
            put_flow = market_data['pe_oi'].sum() + market_data['pe_volume'].sum()
            
            total_flow = call_flow + put_flow
            if total_flow == 0:
                return None
            
            call_ratio = call_flow / total_flow
            put_ratio = put_flow / total_flow
            
            # Check for significant divergence from 50-50 split
            flow_imbalance = abs(call_ratio - 0.5)
            
            if flow_imbalance >= self.flow_divergence_threshold:
                divergence_strength = min(flow_imbalance * 2, 1.0)
                
                if divergence_strength >= self.min_divergence_strength:
                    # Calculate confidence based on flow magnitudes
                    confidence = min(total_flow / 100000, 1.0)  # Scale by flow size
                    
                    return DivergenceResult(
                        divergence_type=DivergenceType.CALL_PUT_DIVERGENCE,
                        strength=divergence_strength,
                        confidence=confidence,
                        supporting_evidence={
                            'call_flow_ratio': call_ratio,
                            'put_flow_ratio': put_ratio,
                            'flow_imbalance': flow_imbalance,
                            'dominant_side': 'calls' if call_ratio > 0.5 else 'puts',
                            'total_flow': total_flow
                        },
                        time_detected=datetime.now(),
                        persistence_score=self._get_persistence_score(DivergenceType.CALL_PUT_DIVERGENCE)
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting Call-Put divergence: {e}")
            return None
    
    def _detect_institutional_retail_divergence(self, flow_analysis: Dict[str, Any]) -> Optional[DivergenceResult]:
        """Detect Institutional-Retail flow divergence"""
        try:
            institutional_flow = flow_analysis.get('institutional_flow', {})
            retail_flow = flow_analysis.get('retail_flow', {})
            
            if not institutional_flow or not retail_flow:
                return None
            
            # Calculate sentiment divergence
            inst_total = institutional_flow.get('total', 0)
            retail_total = retail_flow.get('total', 0)
            
            if inst_total == 0 or retail_total == 0:
                return None
            
            # Calculate sentiment for each group
            inst_call_ratio = institutional_flow.get('calls', 0) / inst_total
            retail_call_ratio = retail_flow.get('calls', 0) / retail_total
            
            sentiment_divergence = abs(inst_call_ratio - retail_call_ratio)
            
            if sentiment_divergence >= self.flow_difference_threshold:
                divergence_strength = min(sentiment_divergence * 2, 1.0)
                
                if divergence_strength >= self.min_divergence_strength:
                    # Calculate confidence based on flow sizes
                    confidence = min((inst_total + retail_total) / 200000, 1.0)
                    
                    return DivergenceResult(
                        divergence_type=DivergenceType.INSTITUTIONAL_RETAIL_DIVERGENCE,
                        strength=divergence_strength,
                        confidence=confidence,
                        supporting_evidence={
                            'institutional_call_ratio': inst_call_ratio,
                            'retail_call_ratio': retail_call_ratio,
                            'sentiment_divergence': sentiment_divergence,
                            'institutional_flow': inst_total,
                            'retail_flow': retail_total
                        },
                        time_detected=datetime.now(),
                        persistence_score=self._get_persistence_score(DivergenceType.INSTITUTIONAL_RETAIL_DIVERGENCE)
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting Institutional-Retail divergence: {e}")
            return None
    
    def _detect_cross_strike_divergence(self, 
                                      market_data: pd.DataFrame,
                                      spot_price: float) -> Optional[DivergenceResult]:
        """Detect Cross-Strike divergence (ATM vs OTM patterns)"""
        try:
            # Classify strikes by moneyness
            market_data = market_data.copy()
            market_data['moneyness'] = abs(market_data['strike'] - spot_price) / spot_price
            
            atm_data = market_data[market_data['moneyness'] <= self.moneyness_threshold]
            otm_data = market_data[market_data['moneyness'] > self.moneyness_threshold]
            
            if atm_data.empty or otm_data.empty:
                return None
            
            # Calculate flow patterns for ATM and OTM
            atm_call_flow = atm_data['ce_oi'].sum() + atm_data['ce_volume'].sum()
            atm_put_flow = atm_data['pe_oi'].sum() + atm_data['pe_volume'].sum()
            atm_total = atm_call_flow + atm_put_flow
            
            otm_call_flow = otm_data['ce_oi'].sum() + otm_data['ce_volume'].sum()
            otm_put_flow = otm_data['pe_oi'].sum() + otm_data['pe_volume'].sum()
            otm_total = otm_call_flow + otm_put_flow
            
            if atm_total == 0 or otm_total == 0:
                return None
            
            # Calculate sentiment for each group
            atm_sentiment = (atm_call_flow - atm_put_flow) / atm_total
            otm_sentiment = (otm_call_flow - otm_put_flow) / otm_total
            
            sentiment_divergence = abs(atm_sentiment - otm_sentiment)
            
            if sentiment_divergence >= 0.3:  # Threshold for cross-strike divergence
                divergence_strength = min(sentiment_divergence * 1.5, 1.0)
                
                if divergence_strength >= self.min_divergence_strength:
                    confidence = min((atm_total + otm_total) / 100000, 1.0)
                    
                    return DivergenceResult(
                        divergence_type=DivergenceType.CROSS_STRIKE_DIVERGENCE,
                        strength=divergence_strength,
                        confidence=confidence,
                        supporting_evidence={
                            'atm_sentiment': atm_sentiment,
                            'otm_sentiment': otm_sentiment,
                            'sentiment_divergence': sentiment_divergence,
                            'atm_strikes': len(atm_data),
                            'otm_strikes': len(otm_data),
                            'atm_flow': atm_total,
                            'otm_flow': otm_total
                        },
                        time_detected=datetime.now(),
                        persistence_score=self._get_persistence_score(DivergenceType.CROSS_STRIKE_DIVERGENCE)
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting Cross-Strike divergence: {e}")
            return None
    
    def _calculate_pattern_divergence_confidence(self, strike_patterns: List[Dict]) -> float:
        """Calculate confidence for pattern divergence"""
        try:
            if not strike_patterns:
                return 0.0
            
            # Base confidence on number of strikes and their individual confidences
            avg_confidence = np.mean([
                sp['pattern_result'].confidence for sp in strike_patterns
            ])
            
            # Adjust for number of strikes (more strikes = higher confidence)
            strike_factor = min(len(strike_patterns) / 10, 1.0)
            
            return avg_confidence * strike_factor
            
        except:
            return 0.5
    
    def _update_persistence_tracking(self, divergences: List[DivergenceResult]):
        """Update persistence tracking for all divergence types"""
        try:
            current_time = datetime.now()
            
            # Record current divergences
            for divergence in divergences:
                div_type = divergence.divergence_type
                self.persistence_tracker[div_type].append({
                    'timestamp': current_time,
                    'strength': divergence.strength,
                    'confidence': divergence.confidence
                })
            
            # Record absence for types not detected
            detected_types = {d.divergence_type for d in divergences}
            for div_type in DivergenceType:
                if div_type not in detected_types:
                    self.persistence_tracker[div_type].append({
                        'timestamp': current_time,
                        'strength': 0.0,
                        'confidence': 0.0
                    })
            
            # Trim history to persistence memory limit
            for div_type in DivergenceType:
                if len(self.persistence_tracker[div_type]) > self.persistence_memory:
                    self.persistence_tracker[div_type] = self.persistence_tracker[div_type][-self.persistence_memory:]
                    
        except Exception as e:
            logger.error(f"Error updating persistence tracking: {e}")
    
    def _get_persistence_score(self, divergence_type: DivergenceType) -> float:
        """Get persistence score for a divergence type"""
        try:
            history = self.persistence_tracker.get(divergence_type, [])
            if not history:
                return 0.0
            
            # Calculate persistence as average strength over recent history
            recent_strengths = [h['strength'] for h in history]
            return np.mean(recent_strengths)
            
        except:
            return 0.0
    
    def _calculate_overall_divergence_metrics(self, 
                                            divergences: List[DivergenceResult],
                                            divergence_scores: Dict[DivergenceType, float]) -> Dict[str, Any]:
        """Calculate overall divergence metrics"""
        try:
            if not divergences:
                return {
                    'overall_divergence_strength': 0.0,
                    'overall_confidence': 0.0,
                    'divergence_concentration': 0.0,
                    'risk_level': 'low'
                }
            
            # Calculate weighted metrics
            total_strength = sum(d.strength for d in divergences)
            avg_strength = total_strength / len(divergences)
            
            avg_confidence = np.mean([d.confidence for d in divergences])
            
            # Divergence concentration (how many types detected)
            concentration = len(divergences) / len(DivergenceType)
            
            # Risk level based on strength and concentration
            if avg_strength > 0.7 and concentration > 0.6:
                risk_level = 'high'
            elif avg_strength > 0.5 or concentration > 0.4:
                risk_level = 'medium'
            else:
                risk_level = 'low'
            
            return {
                'overall_divergence_strength': avg_strength,
                'overall_confidence': avg_confidence,
                'divergence_concentration': concentration,
                'risk_level': risk_level,
                'total_divergences': len(divergences),
                'strongest_type': max(divergence_scores.keys(), key=lambda k: divergence_scores[k]) if divergence_scores else None
            }
            
        except Exception as e:
            logger.error(f"Error calculating overall divergence metrics: {e}")
            return {}
    
    def _analyze_divergence_persistence(self) -> Dict[str, Any]:
        """Analyze persistence patterns across all divergence types"""
        try:
            persistence_analysis = {}
            
            for div_type in DivergenceType:
                history = self.persistence_tracker.get(div_type, [])
                if history:
                    strengths = [h['strength'] for h in history]
                    persistence_analysis[div_type.value] = {
                        'average_strength': np.mean(strengths),
                        'max_strength': np.max(strengths),
                        'persistence_ratio': len([s for s in strengths if s > 0]) / len(strengths),
                        'trend': 'increasing' if len(strengths) > 1 and strengths[-1] > strengths[0] else 'decreasing'
                    }
            
            return persistence_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing divergence persistence: {e}")
            return {}
    
    def _get_strongest_divergence(self, divergences: List[DivergenceResult]) -> Optional[str]:
        """Get the strongest divergence type"""
        try:
            if not divergences:
                return None
            
            strongest = max(divergences, key=lambda d: d.strength)
            return strongest.divergence_type.value
            
        except:
            return None
    
    def _get_default_divergence_result(self) -> Dict[str, Any]:
        """Get default result for error cases"""
        return {
            'divergences_detected': [],
            'divergence_scores': {},
            'overall_metrics': {
                'overall_divergence_strength': 0.0,
                'overall_confidence': 0.0,
                'divergence_concentration': 0.0,
                'risk_level': 'low'
            },
            'persistence_analysis': {},
            'detection_metadata': {
                'error': True,
                'timestamp': datetime.now()
            }
        }
    
    def get_divergence_detection_summary(self) -> Dict[str, Any]:
        """Get summary of divergence detection performance"""
        try:
            summary = {
                'detection_config': {
                    'min_divergence_strength': self.min_divergence_strength,
                    'min_confidence_threshold': self.min_confidence_threshold,
                    'persistence_memory': self.persistence_memory
                },
                'persistence_tracking': {}
            }
            
            # Summarize persistence tracking
            for div_type in DivergenceType:
                history = self.persistence_tracker.get(div_type, [])
                if history:
                    strengths = [h['strength'] for h in history]
                    summary['persistence_tracking'][div_type.value] = {
                        'total_detections': len([s for s in strengths if s > 0]),
                        'average_strength': np.mean(strengths),
                        'recent_activity': strengths[-3:] if len(strengths) >= 3 else strengths
                    }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating divergence detection summary: {e}")
            return {'status': 'error', 'error': str(e)}