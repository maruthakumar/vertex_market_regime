"""
Intelligent Transition Manager

This module implements intelligent transition management with noise filtering,
immature transition detection, and hysteresis-based stability controls.

Key Features:
- Multi-level noise filtering
- Immature transition detection and prevention
- Adaptive hysteresis based on market conditions
- Confidence-based transition validation
- False positive prevention
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

warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)


class TransitionSignal(Enum):
    """Transition signal types"""
    CONFIRMED = "confirmed"
    TENTATIVE = "tentative"
    REJECTED = "rejected"
    PENDING = "pending"


@dataclass
class TransitionCandidate:
    """Represents a potential regime transition"""
    from_regime: int
    to_regime: int
    signal_strength: float
    confidence: float
    timestamp: datetime
    market_data: Dict[str, Any]
    supporting_indicators: List[str]
    noise_score: float
    maturity_score: float
    hysteresis_check: bool
    final_decision: TransitionSignal = TransitionSignal.PENDING


@dataclass
class TransitionDecision:
    """Final decision on regime transition"""
    approved: bool
    new_regime: int
    confidence: float
    decision_time: datetime
    reasons: List[str]
    filtered_signals: int
    hysteresis_applied: bool
    noise_level: float


@dataclass
class NoiseFilterConfig:
    """Configuration for noise filtering"""
    enable_microstructure_filter: bool = True
    enable_volume_filter: bool = True
    enable_persistence_filter: bool = True
    microstructure_threshold: float = 0.05
    volume_threshold_multiplier: float = 0.5
    persistence_window: int = 3
    noise_score_threshold: float = 0.3


@dataclass
class HysteresisConfig:
    """Configuration for hysteresis management"""
    base_hysteresis: float = 0.05
    adaptive_factor: float = 2.0
    volatility_adjustment: bool = True
    trend_adjustment: bool = True
    max_hysteresis: float = 0.15
    min_hysteresis: float = 0.01


class IntelligentTransitionManager:
    """
    Manages regime transitions with intelligent filtering and validation
    """
    
    def __init__(self, regime_count: int = 12,
                 noise_config: Optional[NoiseFilterConfig] = None,
                 hysteresis_config: Optional[HysteresisConfig] = None):
        """
        Initialize intelligent transition manager
        
        Args:
            regime_count: Number of regimes in the system
            noise_config: Noise filtering configuration
            hysteresis_config: Hysteresis configuration
        """
        self.regime_count = regime_count
        self.noise_config = noise_config or NoiseFilterConfig()
        self.hysteresis_config = hysteresis_config or HysteresisConfig()
        
        # Current state
        self.current_regime = 0
        self.regime_start_time = datetime.now()
        self.regime_confidence = 1.0
        
        # Transition tracking
        self.transition_candidates = deque(maxlen=100)
        self.transition_history = deque(maxlen=1000)
        self.rejected_transitions = deque(maxlen=500)
        
        # Noise filtering state
        self.market_data_buffer = deque(maxlen=50)
        self.regime_buffer = deque(maxlen=20)
        self.signal_persistence = defaultdict(int)
        
        # Hysteresis tracking
        self.regime_hysteresis: Dict[int, float] = {
            i: self.hysteresis_config.base_hysteresis 
            for i in range(regime_count)
        }
        self.last_transition_time = datetime.now()
        self.false_transition_count = defaultdict(int)
        
        # Performance metrics
        self.total_transitions = 0
        self.filtered_transitions = 0
        self.false_positives_prevented = 0
        self.immature_transitions_caught = 0
        
        logger.info(f"IntelligentTransitionManager initialized for {regime_count} regimes")
    
    def evaluate_transition(self, proposed_regime: int, 
                          regime_scores: Dict[int, float],
                          market_data: Dict[str, Any],
                          external_confidence: float = 1.0) -> TransitionDecision:
        """
        Evaluate and potentially approve a regime transition
        
        Args:
            proposed_regime: Candidate regime
            regime_scores: All regime scores
            market_data: Current market data
            external_confidence: Confidence from external systems
            
        Returns:
            Transition decision
        """
        start_time = datetime.now()
        
        # Store market data
        self.market_data_buffer.append({
            'timestamp': start_time,
            'data': market_data.copy(),
            'regime_scores': regime_scores.copy()
        })
        
        # Check if transition is needed
        if proposed_regime == self.current_regime:
            return TransitionDecision(
                approved=False,
                new_regime=self.current_regime,
                confidence=self.regime_confidence,
                decision_time=start_time,
                reasons=['no_change_needed'],
                filtered_signals=0,
                hysteresis_applied=False,
                noise_level=0.0
            )
        
        # Create transition candidate
        candidate = self._create_transition_candidate(
            proposed_regime, regime_scores, market_data, external_confidence
        )
        
        # Apply filters and checks
        decision_reasons = []
        filters_applied = 0
        
        # 1. Noise filtering
        noise_result = self._apply_noise_filters(candidate, market_data)
        if not noise_result['passed']:
            decision_reasons.extend(noise_result['reasons'])
            filters_applied += len(noise_result['reasons'])
        
        # 2. Maturity check
        maturity_result = self._check_transition_maturity(candidate)
        if not maturity_result['passed']:
            decision_reasons.extend(maturity_result['reasons'])
            filters_applied += 1
        
        # 3. Hysteresis check
        hysteresis_result = self._apply_hysteresis(candidate, market_data)
        if not hysteresis_result['passed']:
            decision_reasons.extend(hysteresis_result['reasons'])
            filters_applied += 1
        
        # 4. Confidence validation
        confidence_result = self._validate_confidence(candidate, regime_scores)
        if not confidence_result['passed']:
            decision_reasons.extend(confidence_result['reasons'])
            filters_applied += 1
        
        # Make final decision
        all_checks_passed = (
            noise_result['passed'] and 
            maturity_result['passed'] and 
            hysteresis_result['passed'] and 
            confidence_result['passed']
        )
        
        if all_checks_passed:
            # Approve transition
            candidate.final_decision = TransitionSignal.CONFIRMED
            self._execute_transition(proposed_regime, candidate.confidence, start_time)
            decision_reasons.append('transition_approved')
            
            decision = TransitionDecision(
                approved=True,
                new_regime=proposed_regime,
                confidence=candidate.confidence,
                decision_time=start_time,
                reasons=decision_reasons,
                filtered_signals=filters_applied,
                hysteresis_applied=hysteresis_result.get('applied', False),
                noise_level=candidate.noise_score
            )
        else:
            # Reject transition
            candidate.final_decision = TransitionSignal.REJECTED
            self.rejected_transitions.append(candidate)
            self.filtered_transitions += 1
            
            decision = TransitionDecision(
                approved=False,
                new_regime=self.current_regime,
                confidence=self.regime_confidence,
                decision_time=start_time,
                reasons=decision_reasons,
                filtered_signals=filters_applied,
                hysteresis_applied=hysteresis_result.get('applied', False),
                noise_level=candidate.noise_score
            )
        
        # Store candidate for analysis
        self.transition_candidates.append(candidate)
        
        return decision
    
    def _create_transition_candidate(self, proposed_regime: int,
                                   regime_scores: Dict[int, float],
                                   market_data: Dict[str, Any],
                                   external_confidence: float) -> TransitionCandidate:
        """Create a transition candidate for evaluation"""
        
        # Calculate signal strength
        current_score = regime_scores.get(self.current_regime, 0.0)
        proposed_score = regime_scores.get(proposed_regime, 0.0)
        signal_strength = proposed_score - current_score
        
        # Identify supporting indicators
        supporting_indicators = self._identify_supporting_indicators(
            proposed_regime, market_data
        )
        
        # Calculate base confidence
        confidence = min(proposed_score * external_confidence, 1.0)
        
        candidate = TransitionCandidate(
            from_regime=self.current_regime,
            to_regime=proposed_regime,
            signal_strength=signal_strength,
            confidence=confidence,
            timestamp=datetime.now(),
            market_data=market_data.copy(),
            supporting_indicators=supporting_indicators,
            noise_score=0.0,  # Will be calculated by noise filters
            maturity_score=0.0,  # Will be calculated by maturity check
            hysteresis_check=False  # Will be set by hysteresis check
        )
        
        return candidate
    
    def _identify_supporting_indicators(self, proposed_regime: int,
                                      market_data: Dict[str, Any]) -> List[str]:
        """Identify market indicators supporting the transition"""
        indicators = []
        
        # Volatility indicators
        volatility = market_data.get('volatility', 0.0)
        if volatility > 0.3 and proposed_regime >= self.regime_count * 2 // 3:
            indicators.append('high_volatility_regime')
        elif volatility < 0.1 and proposed_regime < self.regime_count // 3:
            indicators.append('low_volatility_regime')
        
        # Trend indicators
        trend = market_data.get('trend', 0.0)
        if trend > 0.01:
            indicators.append('bullish_trend')
        elif trend < -0.01:
            indicators.append('bearish_trend')
        
        # Volume indicators
        volume_ratio = market_data.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:
            indicators.append('high_volume')
        elif volume_ratio < 0.5:
            indicators.append('low_volume')
        
        # Greeks indicators
        if 'total_delta' in market_data:
            delta = market_data['total_delta']
            if abs(delta) > 1000:
                indicators.append('significant_delta')
        
        if 'total_gamma' in market_data:
            gamma = market_data['total_gamma']
            if abs(gamma) > 500:
                indicators.append('high_gamma')
        
        return indicators
    
    def _apply_noise_filters(self, candidate: TransitionCandidate,
                           market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply noise filtering to transition candidate"""
        
        noise_score = 0.0
        failed_filters = []
        
        # 1. Microstructure noise filter
        if self.noise_config.enable_microstructure_filter:
            microstructure_noise = self._detect_microstructure_noise(market_data)
            if microstructure_noise > self.noise_config.microstructure_threshold:
                noise_score += microstructure_noise
                failed_filters.append('microstructure_noise')
        
        # 2. Volume-based noise filter
        if self.noise_config.enable_volume_filter:
            volume_noise = self._detect_volume_noise(market_data)
            if volume_noise > 0.5:
                noise_score += volume_noise * 0.3
                failed_filters.append('volume_noise')
        
        # 3. Signal persistence filter
        if self.noise_config.enable_persistence_filter:
            persistence_score = self._check_signal_persistence(candidate)
            if persistence_score < 0.5:
                noise_score += (1.0 - persistence_score) * 0.4
                failed_filters.append('insufficient_persistence')
        
        # Update candidate noise score
        candidate.noise_score = noise_score
        
        # Check if noise threshold exceeded
        passed = noise_score < self.noise_config.noise_score_threshold
        
        return {
            'passed': passed,
            'reasons': failed_filters,
            'noise_score': noise_score
        }
    
    def _detect_microstructure_noise(self, market_data: Dict[str, Any]) -> float:
        """Detect microstructure noise in market data"""
        
        if len(self.market_data_buffer) < 5:
            return 0.0
        
        # Get recent price changes
        recent_data = list(self.market_data_buffer)[-5:]
        price_changes = []
        
        for i in range(1, len(recent_data)):
            prev_price = recent_data[i-1]['data'].get('spot_price', 0)
            curr_price = recent_data[i]['data'].get('spot_price', 0)
            if prev_price > 0 and curr_price > 0:
                change = abs(curr_price - prev_price) / prev_price
                price_changes.append(change)
        
        if not price_changes:
            return 0.0
        
        # Check for excessive small fluctuations
        small_changes = [c for c in price_changes if c < 0.001]
        if len(small_changes) >= 3:
            return 0.8  # High microstructure noise
        
        # Check for price reversals (noise characteristic)
        reversals = 0
        for i in range(1, len(price_changes)):
            if i < len(price_changes) - 1:
                if ((price_changes[i-1] > 0 and price_changes[i+1] > 0 and price_changes[i] < 0) or
                    (price_changes[i-1] < 0 and price_changes[i+1] < 0 and price_changes[i] > 0)):
                    reversals += 1
        
        return min(reversals / 3.0, 1.0)
    
    def _detect_volume_noise(self, market_data: Dict[str, Any]) -> float:
        """Detect volume-based noise"""
        
        volume_ratio = market_data.get('volume_ratio', 1.0)
        volume = market_data.get('volume', 0)
        
        # Check if volume is unusually low
        if volume_ratio < self.noise_config.volume_threshold_multiplier:
            return 0.7
        
        # Check recent volume pattern
        if len(self.market_data_buffer) >= 3:
            recent_volumes = [
                data['data'].get('volume', 0) 
                for data in list(self.market_data_buffer)[-3:]
            ]
            
            # Detect volume spikes (potential noise)
            current_volume = market_data.get('volume', 0)
            avg_recent = np.mean(recent_volumes) if recent_volumes else 1
            
            if avg_recent > 0 and current_volume > avg_recent * 3:
                return 0.6  # Volume spike
        
        return 0.0
    
    def _check_signal_persistence(self, candidate: TransitionCandidate) -> float:
        """Check if transition signal is persistent"""
        
        to_regime = candidate.to_regime
        
        # Count recent signals for this regime
        recent_signals = 0
        total_recent = 0
        
        for data in list(self.market_data_buffer)[-self.noise_config.persistence_window:]:
            regime_scores = data.get('regime_scores', {})
            if regime_scores:
                total_recent += 1
                best_regime = max(regime_scores.items(), key=lambda x: x[1])[0]
                if best_regime == to_regime:
                    recent_signals += 1
        
        if total_recent == 0:
            return 0.5  # Neutral when no data
        
        persistence_ratio = recent_signals / total_recent
        
        # Update persistence tracking
        self.signal_persistence[to_regime] = persistence_ratio
        
        return persistence_ratio
    
    def _check_transition_maturity(self, candidate: TransitionCandidate) -> Dict[str, Any]:
        """Check if transition has sufficient maturity"""
        
        # Check minimum time in current regime
        time_in_regime = datetime.now() - self.regime_start_time
        min_regime_duration = timedelta(minutes=15)  # Configurable
        
        if time_in_regime < min_regime_duration:
            return {
                'passed': False,
                'reasons': ['insufficient_regime_duration'],
                'time_in_regime': time_in_regime.total_seconds()
            }
        
        # Check signal strength growth
        signal_growth = self._analyze_signal_growth(candidate.to_regime)
        
        if signal_growth < 0.1:  # Signal not growing
            return {
                'passed': False,
                'reasons': ['weak_signal_growth'],
                'signal_growth': signal_growth
            }
        
        # Calculate maturity score
        maturity_score = min(
            time_in_regime.total_seconds() / min_regime_duration.total_seconds(),
            1.0
        ) * 0.7 + signal_growth * 0.3
        
        candidate.maturity_score = maturity_score
        
        return {
            'passed': maturity_score > 0.6,
            'reasons': [] if maturity_score > 0.6 else ['low_maturity_score'],
            'maturity_score': maturity_score
        }
    
    def _analyze_signal_growth(self, regime: int) -> float:
        """Analyze how the signal for a regime is growing"""
        
        if len(self.market_data_buffer) < 3:
            return 0.5
        
        # Get recent scores for this regime
        recent_scores = []
        for data in list(self.market_data_buffer)[-5:]:
            regime_scores = data.get('regime_scores', {})
            if regime in regime_scores:
                recent_scores.append(regime_scores[regime])
        
        if len(recent_scores) < 2:
            return 0.5
        
        # Calculate growth trend
        score_growth = 0.0
        for i in range(1, len(recent_scores)):
            growth = recent_scores[i] - recent_scores[i-1]
            score_growth += growth
        
        # Normalize growth
        return max(0.0, min(score_growth / len(recent_scores), 1.0))
    
    def _apply_hysteresis(self, candidate: TransitionCandidate,
                         market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply hysteresis to prevent oscillations"""
        
        from_regime = candidate.from_regime
        to_regime = candidate.to_regime
        
        # Get current hysteresis for source regime
        current_hysteresis = self.regime_hysteresis[from_regime]
        
        # Adjust hysteresis based on market conditions
        adjusted_hysteresis = self._calculate_adaptive_hysteresis(
            current_hysteresis, market_data
        )
        
        # Check if signal strength exceeds hysteresis threshold
        required_strength = current_hysteresis + adjusted_hysteresis
        actual_strength = candidate.signal_strength
        
        if actual_strength < required_strength:
            candidate.hysteresis_check = False
            return {
                'passed': False,
                'reasons': ['hysteresis_threshold_not_met'],
                'required_strength': required_strength,
                'actual_strength': actual_strength,
                'applied': True
            }
        
        candidate.hysteresis_check = True
        
        return {
            'passed': True,
            'reasons': [],
            'required_strength': required_strength,
            'actual_strength': actual_strength,
            'applied': True
        }
    
    def _calculate_adaptive_hysteresis(self, base_hysteresis: float,
                                     market_data: Dict[str, Any]) -> float:
        """Calculate adaptive hysteresis based on market conditions"""
        
        adjustment = 0.0
        
        # Volatility adjustment
        if self.hysteresis_config.volatility_adjustment:
            volatility = market_data.get('volatility', 0.2)
            # Higher volatility = higher hysteresis
            vol_adjustment = (volatility - 0.2) * self.hysteresis_config.adaptive_factor
            adjustment += vol_adjustment
        
        # Trend strength adjustment
        if self.hysteresis_config.trend_adjustment:
            trend = abs(market_data.get('trend', 0.0))
            # Strong trends = lower hysteresis
            trend_adjustment = -trend * self.hysteresis_config.adaptive_factor * 0.5
            adjustment += trend_adjustment
        
        # Apply bounds
        final_hysteresis = base_hysteresis + adjustment
        final_hysteresis = max(
            self.hysteresis_config.min_hysteresis,
            min(final_hysteresis, self.hysteresis_config.max_hysteresis)
        )
        
        return final_hysteresis - base_hysteresis
    
    def _validate_confidence(self, candidate: TransitionCandidate,
                           regime_scores: Dict[int, float]) -> Dict[str, Any]:
        """Validate transition confidence"""
        
        min_confidence = 0.65  # Configurable
        
        if candidate.confidence < min_confidence:
            return {
                'passed': False,
                'reasons': ['insufficient_confidence'],
                'confidence': candidate.confidence,
                'required': min_confidence
            }
        
        # Check relative confidence vs other regimes
        sorted_scores = sorted(regime_scores.items(), key=lambda x: x[1], reverse=True)
        
        if len(sorted_scores) >= 2:
            best_score = sorted_scores[0][1]
            second_best = sorted_scores[1][1]
            
            # Ensure sufficient margin
            margin = best_score - second_best
            min_margin = 0.1
            
            if margin < min_margin:
                return {
                    'passed': False,
                    'reasons': ['insufficient_margin'],
                    'margin': margin,
                    'required': min_margin
                }
        
        return {
            'passed': True,
            'reasons': [],
            'confidence': candidate.confidence
        }
    
    def _execute_transition(self, new_regime: int, confidence: float,
                          transition_time: datetime):
        """Execute approved regime transition"""
        
        old_regime = self.current_regime
        
        # Update state
        self.current_regime = new_regime
        self.regime_confidence = confidence
        self.regime_start_time = transition_time
        self.last_transition_time = transition_time
        
        # Record transition
        self.transition_history.append({
            'from_regime': old_regime,
            'to_regime': new_regime,
            'timestamp': transition_time,
            'confidence': confidence
        })
        
        # Update statistics
        self.total_transitions += 1
        
        # Update hysteresis based on transition success
        self._update_hysteresis_on_transition(old_regime, new_regime)
        
        logger.info(f"Regime transition executed: {old_regime} → {new_regime} "
                   f"(confidence: {confidence:.3f})")
    
    def _update_hysteresis_on_transition(self, from_regime: int, to_regime: int):
        """Update hysteresis values based on transition outcome"""
        
        # Reduce hysteresis for successful transitions
        current_hysteresis = self.regime_hysteresis[from_regime]
        reduction_factor = 0.95
        
        self.regime_hysteresis[from_regime] = max(
            current_hysteresis * reduction_factor,
            self.hysteresis_config.min_hysteresis
        )
        
        # Track false transitions (quick reversals)
        # This would be updated by external feedback
    
    def update_false_transition_feedback(self, was_false_positive: bool):
        """Update based on feedback about transition quality"""
        
        if was_false_positive:
            # Increase hysteresis for current regime
            current_hysteresis = self.regime_hysteresis[self.current_regime]
            increase_factor = 1.1
            
            self.regime_hysteresis[self.current_regime] = min(
                current_hysteresis * increase_factor,
                self.hysteresis_config.max_hysteresis
            )
            
            self.false_positives_prevented += 1
            self.false_transition_count[self.current_regime] += 1
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current transition manager state"""
        
        return {
            'current_regime': self.current_regime,
            'regime_confidence': self.regime_confidence,
            'time_in_regime': (datetime.now() - self.regime_start_time).total_seconds(),
            'regime_hysteresis': self.regime_hysteresis.copy(),
            'total_transitions': self.total_transitions,
            'filtered_transitions': self.filtered_transitions,
            'filter_rate': self.filtered_transitions / max(self.total_transitions + self.filtered_transitions, 1),
            'false_positives_prevented': self.false_positives_prevented
        }
    
    def get_filter_statistics(self) -> Dict[str, Any]:
        """Get detailed filtering statistics"""
        
        # Analyze recent rejections
        recent_rejections = list(self.rejected_transitions)[-50:]
        
        rejection_reasons = defaultdict(int)
        for candidate in recent_rejections:
            if hasattr(candidate, 'final_decision'):
                # This would contain rejection reasons
                rejection_reasons['noise_filter'] += 1
        
        # Calculate noise statistics
        noise_scores = [c.noise_score for c in recent_rejections if c.noise_score > 0]
        avg_noise = np.mean(noise_scores) if noise_scores else 0.0
        
        return {
            'total_candidates': len(self.transition_candidates),
            'total_rejections': len(self.rejected_transitions),
            'rejection_rate': len(self.rejected_transitions) / max(len(self.transition_candidates), 1),
            'rejection_reasons': dict(rejection_reasons),
            'average_noise_score': avg_noise,
            'hysteresis_distribution': self.regime_hysteresis.copy(),
            'false_transition_counts': dict(self.false_transition_count)
        }
    
    def export_transition_log(self, filepath: str):
        """Export transition history and statistics"""
        
        log_data = {
            'configuration': {
                'regime_count': self.regime_count,
                'noise_config': self.noise_config.__dict__,
                'hysteresis_config': self.hysteresis_config.__dict__
            },
            'current_state': self.get_current_state(),
            'statistics': self.get_filter_statistics(),
            'transition_history': [
                {
                    'from_regime': t['from_regime'],
                    'to_regime': t['to_regime'],
                    'timestamp': t['timestamp'].isoformat(),
                    'confidence': t['confidence']
                }
                for t in list(self.transition_history)
            ],
            'recent_candidates': [
                {
                    'from_regime': c.from_regime,
                    'to_regime': c.to_regime,
                    'signal_strength': c.signal_strength,
                    'confidence': c.confidence,
                    'timestamp': c.timestamp.isoformat(),
                    'noise_score': c.noise_score,
                    'maturity_score': c.maturity_score,
                    'final_decision': c.final_decision.value
                }
                for c in list(self.transition_candidates)[-20:]
            ]
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        logger.info(f"Transition log exported to {filepath}")


# Example usage
if __name__ == "__main__":
    # Create intelligent transition manager
    noise_config = NoiseFilterConfig(
        enable_microstructure_filter=True,
        enable_volume_filter=True,
        enable_persistence_filter=True,
        noise_score_threshold=0.3
    )
    
    hysteresis_config = HysteresisConfig(
        base_hysteresis=0.05,
        adaptive_factor=2.0,
        volatility_adjustment=True,
        trend_adjustment=True
    )
    
    manager = IntelligentTransitionManager(
        regime_count=8,
        noise_config=noise_config,
        hysteresis_config=hysteresis_config
    )
    
    # Simulate regime transitions
    regime_scores = {i: np.random.random() for i in range(8)}
    regime_scores[3] = 0.85  # Strong signal for regime 3
    
    market_data = {
        'volatility': 0.15,
        'trend': 0.02,
        'volume_ratio': 1.2,
        'spot_price': 25000,
        'volume': 50000,
        'total_delta': 1500,
        'total_gamma': -300
    }
    
    # Evaluate transition
    decision = manager.evaluate_transition(
        proposed_regime=3,
        regime_scores=regime_scores,
        market_data=market_data,
        external_confidence=0.9
    )
    
    print("\nTransition Decision:")
    print(f"Approved: {decision.approved}")
    print(f"New regime: {decision.new_regime}")
    print(f"Confidence: {decision.confidence:.3f}")
    print(f"Reasons: {decision.reasons}")
    print(f"Filtered signals: {decision.filtered_signals}")
    print(f"Noise level: {decision.noise_level:.3f}")
    
    # Get current state
    state = manager.get_current_state()
    print(f"\nCurrent state:")
    print(f"Current regime: {state['current_regime']}")
    print(f"Filter rate: {state['filter_rate']:.2%}")
    print(f"Total transitions: {state['total_transitions']}")