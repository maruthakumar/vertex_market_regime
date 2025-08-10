"""
Weight Optimizer for Triple Straddle Analysis

Dynamically calculates optimal weights for combining ATM, ITM1, and OTM1 straddles
based on market conditions, regime detection, and efficiency metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class WeightOptimizationResult:
    """Result of weight optimization"""
    atm_weight: float
    itm1_weight: float
    otm1_weight: float
    confidence: float
    rationale: str
    optimization_metrics: Dict[str, float]


class WeightOptimizer:
    """
    Optimizes weights for combining multiple straddle strategies based on:
    - Market regime conditions
    - Historical efficiency
    - Risk-return profiles
    - Correlation patterns
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize weight optimizer"""
        self.config = config or self._get_default_config()
        
        # Weight constraints
        self.min_weight = self.config.get('min_weight', 0.05)
        self.max_weight = self.config.get('max_weight', 0.70)
        self.weight_sum_tolerance = self.config.get('weight_sum_tolerance', 0.01)
        
        # Optimization parameters
        self.efficiency_weight = self.config.get('efficiency_weight', 0.4)
        self.risk_weight = self.config.get('risk_weight', 0.3)
        self.regime_weight = self.config.get('regime_weight', 0.3)
        
        logger.info("WeightOptimizer initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'min_weight': 0.05,
            'max_weight': 0.70,
            'weight_sum_tolerance': 0.01,
            'efficiency_weight': 0.4,
            'risk_weight': 0.3,
            'regime_weight': 0.3,
            'volatility_adjustment': True,
            'correlation_adjustment': True
        }
    
    def calculate_optimal_weights(
        self,
        atm_metrics: Dict[str, float],
        itm1_metrics: Dict[str, float],
        otm1_metrics: Dict[str, float],
        market_context: Dict[str, Any]
    ) -> WeightOptimizationResult:
        """
        Calculate optimal weights for the three straddle strategies
        
        Args:
            atm_metrics: ATM straddle performance metrics
            itm1_metrics: ITM1 straddle performance metrics  
            otm1_metrics: OTM1 straddle performance metrics
            market_context: Current market conditions and regime info
            
        Returns:
            WeightOptimizationResult with optimal weights and rationale
        """
        try:
            # Calculate efficiency scores
            efficiency_scores = self._calculate_efficiency_scores(
                atm_metrics, itm1_metrics, otm1_metrics
            )
            
            # Calculate risk scores
            risk_scores = self._calculate_risk_scores(
                atm_metrics, itm1_metrics, otm1_metrics
            )
            
            # Calculate regime suitability scores
            regime_scores = self._calculate_regime_scores(
                atm_metrics, itm1_metrics, otm1_metrics, market_context
            )
            
            # Combine scores with weights
            combined_scores = {
                'atm': (
                    efficiency_scores['atm'] * self.efficiency_weight +
                    risk_scores['atm'] * self.risk_weight +
                    regime_scores['atm'] * self.regime_weight
                ),
                'itm1': (
                    efficiency_scores['itm1'] * self.efficiency_weight +
                    risk_scores['itm1'] * self.risk_weight +
                    regime_scores['itm1'] * self.regime_weight
                ),
                'otm1': (
                    efficiency_scores['otm1'] * self.efficiency_weight +
                    risk_scores['otm1'] * self.risk_weight +
                    regime_scores['otm1'] * self.regime_weight
                )
            }
            
            # Normalize scores to weights
            raw_weights = self._normalize_scores_to_weights(combined_scores)
            
            # Apply constraints and adjustments
            final_weights = self._apply_weight_constraints(raw_weights, market_context)
            
            # Calculate confidence and rationale
            confidence = self._calculate_confidence(
                efficiency_scores, risk_scores, regime_scores, market_context
            )
            rationale = self._generate_rationale(
                final_weights, efficiency_scores, risk_scores, regime_scores
            )
            
            optimization_metrics = {
                'efficiency_scores': efficiency_scores,
                'risk_scores': risk_scores,
                'regime_scores': regime_scores,
                'combined_scores': combined_scores
            }
            
            return WeightOptimizationResult(
                atm_weight=final_weights['atm'],
                itm1_weight=final_weights['itm1'],
                otm1_weight=final_weights['otm1'],
                confidence=confidence,
                rationale=rationale,
                optimization_metrics=optimization_metrics
            )
            
        except Exception as e:
            logger.error(f"Weight optimization failed: {e}")
            # Return equal weights as fallback
            return self._get_equal_weight_fallback()
    
    def _calculate_efficiency_scores(
        self,
        atm_metrics: Dict[str, float],
        itm1_metrics: Dict[str, float],
        otm1_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate efficiency scores based on price-to-risk ratios"""
        
        def get_efficiency(metrics: Dict[str, float]) -> float:
            premium = metrics.get('premium', 0)
            implied_vol = metrics.get('implied_volatility', 0.15)
            theta_decay = abs(metrics.get('theta', -20))
            
            if theta_decay == 0:
                return 0.5  # Neutral score
            
            # Higher premium relative to theta decay = better efficiency
            efficiency = premium / theta_decay
            return min(max(efficiency / 10, 0), 1)  # Normalize to 0-1
        
        return {
            'atm': get_efficiency(atm_metrics),
            'itm1': get_efficiency(itm1_metrics),
            'otm1': get_efficiency(otm1_metrics)
        }
    
    def _calculate_risk_scores(
        self,
        atm_metrics: Dict[str, float],
        itm1_metrics: Dict[str, float],
        otm1_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate risk scores (lower risk = higher score)"""
        
        def get_risk_score(metrics: Dict[str, float]) -> float:
            gamma = metrics.get('gamma', 0.02)
            vega = metrics.get('vega', 50)
            
            # Higher gamma/vega = higher risk = lower score
            gamma_risk = gamma / 0.05  # Normalize around typical ATM gamma
            vega_risk = vega / 100     # Normalize around typical ATM vega
            
            risk_factor = (gamma_risk + vega_risk) / 2
            return max(0, 1 - risk_factor)  # Invert so lower risk = higher score
        
        return {
            'atm': get_risk_score(atm_metrics),
            'itm1': get_risk_score(itm1_metrics),
            'otm1': get_risk_score(otm1_metrics)
        }
    
    def _calculate_regime_scores(
        self,
        atm_metrics: Dict[str, float],
        itm1_metrics: Dict[str, float],
        otm1_metrics: Dict[str, float],
        market_context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate regime suitability scores"""
        
        current_regime = market_context.get('market_regime', 'neutral')
        volatility = market_context.get('implied_volatility', 0.15)
        trend_strength = market_context.get('trend_strength', 0.5)
        
        # Base scores
        scores = {'atm': 0.5, 'itm1': 0.5, 'otm1': 0.5}
        
        # Regime-based adjustments
        if 'high_vol' in current_regime.lower():
            scores['atm'] += 0.2  # ATM benefits from high volatility
            scores['otm1'] += 0.1  # OTM gets some benefit
            scores['itm1'] -= 0.1  # ITM less attractive in high vol
        elif 'low_vol' in current_regime.lower():
            scores['itm1'] += 0.2  # ITM better in low vol (less theta decay)
            scores['atm'] -= 0.1
            scores['otm1'] -= 0.1
        
        if 'trending' in current_regime.lower():
            scores['itm1'] += 0.15  # ITM benefits from directional moves
            scores['otm1'] += 0.1   # OTM can benefit from momentum
            scores['atm'] -= 0.1    # ATM neutral to direction
        elif 'sideways' in current_regime.lower():
            scores['atm'] += 0.2    # ATM ideal for range-bound markets
            scores['itm1'] -= 0.05
            scores['otm1'] -= 0.1
        
        # Volatility adjustments
        if volatility > 0.25:  # High volatility
            scores['atm'] += 0.1
        elif volatility < 0.10:  # Low volatility
            scores['itm1'] += 0.1
            scores['otm1'] -= 0.1
        
        # Ensure scores are in valid range
        for key in scores:
            scores[key] = max(0, min(1, scores[key]))
        
        return scores
    
    def _normalize_scores_to_weights(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize scores to sum to 1.0"""
        total_score = sum(scores.values())
        
        if total_score == 0:
            # Equal weights if all scores are zero
            return {'atm': 1/3, 'itm1': 1/3, 'otm1': 1/3}
        
        return {
            'atm': scores['atm'] / total_score,
            'itm1': scores['itm1'] / total_score,
            'otm1': scores['otm1'] / total_score
        }
    
    def _apply_weight_constraints(
        self, 
        weights: Dict[str, float], 
        market_context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Apply minimum/maximum weight constraints"""
        
        constrained_weights = weights.copy()
        
        # Apply minimum weight constraint
        for key in constrained_weights:
            if constrained_weights[key] < self.min_weight:
                constrained_weights[key] = self.min_weight
        
        # Apply maximum weight constraint
        for key in constrained_weights:
            if constrained_weights[key] > self.max_weight:
                constrained_weights[key] = self.max_weight
        
        # Renormalize to ensure sum = 1.0
        total_weight = sum(constrained_weights.values())
        if abs(total_weight - 1.0) > self.weight_sum_tolerance:
            for key in constrained_weights:
                constrained_weights[key] /= total_weight
        
        return constrained_weights
    
    def _calculate_confidence(
        self,
        efficiency_scores: Dict[str, float],
        risk_scores: Dict[str, float],
        regime_scores: Dict[str, float],
        market_context: Dict[str, Any]
    ) -> float:
        """Calculate confidence in the weight optimization"""
        
        # Score variance (lower variance = higher confidence)
        all_scores = []
        for strategy in ['atm', 'itm1', 'otm1']:
            all_scores.extend([
                efficiency_scores[strategy],
                risk_scores[strategy], 
                regime_scores[strategy]
            ])
        
        score_variance = np.var(all_scores)
        variance_confidence = max(0, 1 - score_variance * 4)  # Scale variance
        
        # Market context confidence
        regime_clarity = market_context.get('regime_confidence', 0.5)
        data_quality = market_context.get('data_quality', 0.8)
        
        # Combine confidence factors
        overall_confidence = (
            variance_confidence * 0.4 +
            regime_clarity * 0.3 +
            data_quality * 0.3
        )
        
        return max(0.3, min(0.95, overall_confidence))  # Bounded between 30-95%
    
    def _generate_rationale(
        self,
        weights: Dict[str, float],
        efficiency_scores: Dict[str, float],
        risk_scores: Dict[str, float],
        regime_scores: Dict[str, float]
    ) -> str:
        """Generate human-readable rationale for weight allocation"""
        
        # Find dominant strategy
        max_weight_strategy = max(weights.keys(), key=lambda k: weights[k])
        max_weight = weights[max_weight_strategy]
        
        if max_weight > 0.5:
            dominance = "heavily weighted towards"
        elif max_weight > 0.4:
            dominance = "favors"
        else:
            dominance = "balanced across strategies with slight preference for"
        
        # Find top scoring dimensions
        top_efficiency = max(efficiency_scores.keys(), key=lambda k: efficiency_scores[k])
        top_risk = max(risk_scores.keys(), key=lambda k: risk_scores[k])
        top_regime = max(regime_scores.keys(), key=lambda k: regime_scores[k])
        
        rationale = f"Weight allocation {dominance} {max_weight_strategy.upper()} "
        rationale += f"({weights[max_weight_strategy]:.1%}). "
        
        # Add reasoning
        reasons = []
        if top_efficiency == max_weight_strategy:
            reasons.append("best efficiency ratio")
        if top_risk == max_weight_strategy:
            reasons.append("favorable risk profile")
        if top_regime == max_weight_strategy:
            reasons.append("regime suitability")
        
        if reasons:
            rationale += f"Primary factors: {', '.join(reasons)}."
        
        return rationale
    
    def _get_equal_weight_fallback(self) -> WeightOptimizationResult:
        """Return equal weights as fallback when optimization fails"""
        return WeightOptimizationResult(
            atm_weight=1/3,
            itm1_weight=1/3,
            otm1_weight=1/3,
            confidence=0.5,
            rationale="Equal weights applied due to optimization failure",
            optimization_metrics={}
        )
    
    def update_weights_based_on_performance(
        self,
        current_weights: Dict[str, float],
        performance_data: Dict[str, Dict[str, float]],
        lookback_periods: int = 10
    ) -> Dict[str, float]:
        """
        Update weights based on recent performance data
        
        Args:
            current_weights: Current weight allocation
            performance_data: Historical performance for each strategy
            lookback_periods: Number of periods to consider
            
        Returns:
            Updated weights
        """
        try:
            # Calculate performance-based adjustments
            adjustments = {}
            
            for strategy in ['atm', 'itm1', 'otm1']:
                perf_data = performance_data.get(strategy, {})
                
                # Calculate recent performance metrics
                returns = perf_data.get('returns', [])
                if len(returns) >= lookback_periods:
                    recent_returns = returns[-lookback_periods:]
                    avg_return = np.mean(recent_returns)
                    return_stability = 1 / (1 + np.std(recent_returns))
                    
                    # Performance score (return adjusted for stability)
                    perf_score = avg_return * return_stability
                    adjustments[strategy] = perf_score
                else:
                    adjustments[strategy] = 0  # No adjustment if insufficient data
            
            # Apply gradual adjustments (max 10% shift per update)
            max_adjustment = 0.1
            updated_weights = current_weights.copy()
            
            if any(adjustments.values()):
                # Normalize adjustments
                total_adj = sum(abs(adj) for adj in adjustments.values())
                if total_adj > 0:
                    for strategy in adjustments:
                        adj_factor = adjustments[strategy] / total_adj
                        weight_change = adj_factor * max_adjustment
                        updated_weights[strategy] += weight_change
                
                # Renormalize
                total_weight = sum(updated_weights.values())
                for strategy in updated_weights:
                    updated_weights[strategy] /= total_weight
            
            return updated_weights
            
        except Exception as e:
            logger.error(f"Performance-based weight update failed: {e}")
            return current_weights