"""
Combined Straddle Analyzer

Combines all three straddles (ATM, ITM1, OTM1) with configurable weights
to create a comprehensive triple straddle analysis. Provides unified regime
contribution and optimal weighting based on market conditions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
from .atm_straddle_analyzer import ATMStraddleAnalyzer, StraddleAnalysisResult
from .itm1_straddle_analyzer import ITM1StraddleAnalyzer
from .otm1_straddle_analyzer import OTM1StraddleAnalyzer
from ..core.calculation_engine import CalculationEngine
from ..rolling.window_manager import RollingWindowManager
from ..config.excel_reader import StraddleConfig
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class CombinedStraddleResult:
    """Result of combined triple straddle analysis"""
    timestamp: pd.Timestamp
    
    # Individual straddle results
    atm_result: Optional[StraddleAnalysisResult]
    itm1_result: Optional[StraddleAnalysisResult]
    otm1_result: Optional[StraddleAnalysisResult]
    
    # Combined metrics
    combined_price: float
    combined_greeks: Dict[str, float]
    combined_metrics: Dict[str, float]
    
    # Weighted regime indicators
    regime_indicators: Dict[str, float]
    
    # Strategy recommendations
    strategy_signals: Dict[str, float]
    optimal_weights: Dict[str, float]
    
    # Market regime classification
    market_regime: str
    regime_confidence: float


class CombinedStraddleAnalyzer:
    """
    Combined Triple Straddle Analyzer
    
    Integrates ATM, ITM1, and OTM1 straddles with dynamic weighting
    to provide comprehensive market analysis and regime detection.
    
    Key features:
    - Dynamic weight optimization based on market conditions
    - Unified regime classification across all straddles
    - Risk-adjusted position recommendations
    - Correlation-based diversification benefits
    """
    
    def __init__(self,
                 config: StraddleConfig,
                 calculation_engine: CalculationEngine,
                 window_manager: RollingWindowManager):
        """
        Initialize Combined Straddle analyzer
        
        Args:
            config: Straddle configuration
            calculation_engine: Shared calculation engine
            window_manager: Rolling window manager
        """
        self.config = config
        self.calculation_engine = calculation_engine
        self.window_manager = window_manager
        
        # Initialize individual straddle analyzers
        self.atm_analyzer = ATMStraddleAnalyzer(config, calculation_engine, window_manager)
        self.itm1_analyzer = ITM1StraddleAnalyzer(config, calculation_engine, window_manager)
        self.otm1_analyzer = OTM1StraddleAnalyzer(config, calculation_engine, window_manager)
        
        # Default straddle weights from config
        self.default_weights = {
            'atm': config.straddle_weights.get('atm', 0.5),
            'itm1': config.straddle_weights.get('itm1', 0.3),
            'otm1': config.straddle_weights.get('otm1', 0.2)
        }
        
        # Normalize weights
        weight_sum = sum(self.default_weights.values())
        if weight_sum > 0:
            for key in self.default_weights:
                self.default_weights[key] /= weight_sum
        
        # Regime detection thresholds
        self.regime_thresholds = {
            'volatility': {'low': -0.3, 'high': 0.3},
            'trend': {'low': -0.3, 'high': 0.3},
            'structure': {'low': -0.3, 'high': 0.3}
        }
        
        # Performance tracking
        self.performance_history = []
        self.max_history_length = 100
        
        self.logger = logging.getLogger(f"{__name__}.combined_straddle")
        self.logger.info(f"Combined Straddle analyzer initialized with weights: {self.default_weights}")
    
    def analyze(self, data: Dict[str, Any], timestamp: pd.Timestamp) -> Optional[CombinedStraddleResult]:
        """
        Perform comprehensive combined straddle analysis
        
        Args:
            data: Market data dictionary
            timestamp: Current timestamp
            
        Returns:
            CombinedStraddleResult or None if insufficient data
        """
        try:
            # Analyze individual straddles
            atm_result = self.atm_analyzer.analyze(data, timestamp)
            itm1_result = self.itm1_analyzer.analyze(data, timestamp)
            otm1_result = self.otm1_analyzer.analyze(data, timestamp)
            
            # Check if we have at least one valid result
            valid_results = [r for r in [atm_result, itm1_result, otm1_result] if r is not None]
            if not valid_results:
                self.logger.warning("No valid straddle results available")
                return None
            
            # Calculate optimal weights based on market conditions
            optimal_weights = self._calculate_optimal_weights(
                atm_result, itm1_result, otm1_result
            )
            
            # Calculate combined price
            combined_price = self._calculate_combined_price(
                atm_result, itm1_result, otm1_result, optimal_weights
            )
            
            # Calculate combined Greeks
            combined_greeks = self._calculate_combined_greeks(
                atm_result, itm1_result, otm1_result, optimal_weights
            )
            
            # Calculate combined metrics
            combined_metrics = self._calculate_combined_metrics(
                atm_result, itm1_result, otm1_result, 
                combined_price, combined_greeks, optimal_weights
            )
            
            # Calculate regime indicators
            regime_indicators = self._calculate_regime_indicators(
                atm_result, itm1_result, otm1_result, optimal_weights
            )
            
            # Determine market regime
            market_regime, regime_confidence = self._determine_market_regime(regime_indicators)
            
            # Calculate strategy signals
            strategy_signals = self._calculate_strategy_signals(
                atm_result, itm1_result, otm1_result,
                combined_metrics, regime_indicators, optimal_weights
            )
            
            # Create result
            result = CombinedStraddleResult(
                timestamp=timestamp,
                atm_result=atm_result,
                itm1_result=itm1_result,
                otm1_result=otm1_result,
                combined_price=combined_price,
                combined_greeks=combined_greeks,
                combined_metrics=combined_metrics,
                regime_indicators=regime_indicators,
                strategy_signals=strategy_signals,
                optimal_weights=optimal_weights,
                market_regime=market_regime,
                regime_confidence=regime_confidence
            )
            
            # Update performance history
            self._update_performance_history(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in combined straddle analysis: {e}")
            return None
    
    def _calculate_optimal_weights(self, atm_result: Optional[StraddleAnalysisResult],
                                  itm1_result: Optional[StraddleAnalysisResult],
                                  otm1_result: Optional[StraddleAnalysisResult]) -> Dict[str, float]:
        """Calculate optimal weights based on market conditions"""
        weights = self.default_weights.copy()
        
        try:
            # Collect efficiency scores
            efficiency_scores = {}
            if atm_result:
                efficiency_scores['atm'] = atm_result.straddle_metrics.get('straddle_efficiency', 0.5)
            if itm1_result:
                efficiency_scores['itm1'] = itm1_result.straddle_metrics.get('itm_efficiency', 0.5)
            if otm1_result:
                efficiency_scores['otm1'] = otm1_result.straddle_metrics.get('otm_efficiency', 0.5)
            
            # Collect regime suitability scores
            regime_scores = self._calculate_regime_suitability_scores(
                atm_result, itm1_result, otm1_result
            )
            
            # Combine efficiency and regime scores
            combined_scores = {}
            for straddle in ['atm', 'itm1', 'otm1']:
                if straddle in efficiency_scores and straddle in regime_scores:
                    combined_scores[straddle] = (
                        0.6 * efficiency_scores[straddle] + 
                        0.4 * regime_scores[straddle]
                    )
            
            # Calculate risk-adjusted weights
            risk_adjusted_weights = self._calculate_risk_adjusted_weights(
                atm_result, itm1_result, otm1_result, combined_scores
            )
            
            # Blend with default weights (momentum)
            momentum_factor = 0.7  # 70% new weights, 30% old weights
            for straddle in weights:
                if straddle in risk_adjusted_weights:
                    weights[straddle] = (
                        momentum_factor * risk_adjusted_weights[straddle] + 
                        (1 - momentum_factor) * self.default_weights[straddle]
                    )
            
            # Normalize weights
            weight_sum = sum(weights.values())
            if weight_sum > 0:
                for key in weights:
                    weights[key] /= weight_sum
            
        except Exception as e:
            self.logger.error(f"Error calculating optimal weights: {e}")
            weights = self.default_weights.copy()
        
        return weights
    
    def _calculate_regime_suitability_scores(self, atm_result: Optional[StraddleAnalysisResult],
                                           itm1_result: Optional[StraddleAnalysisResult],
                                           otm1_result: Optional[StraddleAnalysisResult]) -> Dict[str, float]:
        """Calculate how suitable each straddle is for current regime"""
        scores = {}
        
        try:
            # Aggregate regime indicators
            volatility_score = 0
            trend_score = 0
            structure_score = 0
            count = 0
            
            if atm_result:
                volatility_score += atm_result.regime_indicators.get('iv_regime', 0)
                trend_score += atm_result.regime_indicators.get('trend_uncertainty', 0)
                structure_score += atm_result.regime_indicators.get('market_structure_quality', 0)
                count += 1
            
            if itm1_result:
                volatility_score += itm1_result.regime_indicators.get('itm_volatility_consensus', 0)
                trend_score += abs(itm1_result.regime_indicators.get('itm_trend_bias', 0))
                structure_score += itm1_result.regime_indicators.get('itm_market_efficiency', 0)
                count += 1
            
            if otm1_result:
                volatility_score += otm1_result.regime_indicators.get('volatility_regime', 0)
                trend_score += otm1_result.regime_indicators.get('explosive_move_potential', 0)
                structure_score += otm1_result.regime_indicators.get('otm_market_quality', 0)
                count += 1
            
            if count > 0:
                volatility_score /= count
                trend_score /= count
                structure_score /= count
            
            # Score each straddle based on regime
            # ATM: Best for moderate volatility, neutral markets
            if atm_result:
                atm_vol_score = 1.0 - abs(volatility_score)  # Prefers neutral volatility
                atm_trend_score = 1.0 - abs(trend_score)     # Prefers neutral trend
                atm_structure_score = structure_score        # Benefits from good structure
                scores['atm'] = (atm_vol_score + atm_trend_score + atm_structure_score) / 3
            
            # ITM1: Best for trending markets with protection
            if itm1_result:
                itm_vol_score = 0.5 + volatility_score * 0.3  # Moderate volatility preference
                itm_trend_score = 0.5 + trend_score * 0.5     # Benefits from trends
                itm_structure_score = structure_score          # Benefits from good structure
                scores['itm1'] = (itm_vol_score + itm_trend_score + itm_structure_score) / 3
            
            # OTM1: Best for high volatility, explosive moves
            if otm1_result:
                otm_vol_score = 0.5 + volatility_score * 0.5  # High volatility preference
                otm_trend_score = trend_score                  # Needs explosive moves
                otm_structure_score = 0.5 + structure_score * 0.3  # Less structure dependent
                scores['otm1'] = (otm_vol_score + otm_trend_score + otm_structure_score) / 3
            
        except Exception as e:
            self.logger.error(f"Error calculating regime suitability: {e}")
        
        return scores
    
    def _calculate_risk_adjusted_weights(self, atm_result: Optional[StraddleAnalysisResult],
                                       itm1_result: Optional[StraddleAnalysisResult],
                                       otm1_result: Optional[StraddleAnalysisResult],
                                       combined_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate risk-adjusted weights considering correlations"""
        weights = {}
        
        try:
            # Get risk metrics
            risk_metrics = {}
            
            if atm_result:
                risk_metrics['atm'] = {
                    'decay': atm_result.straddle_metrics.get('decay_as_pct', 0),
                    'efficiency': atm_result.straddle_metrics.get('straddle_efficiency', 0.5),
                    'score': combined_scores.get('atm', 0.5)
                }
            
            if itm1_result:
                risk_metrics['itm1'] = {
                    'decay': itm1_result.straddle_metrics.get('time_value_ratio', 0) * 
                            abs(itm1_result.combined_greeks.get('net_theta', 0)),
                    'efficiency': itm1_result.straddle_metrics.get('itm_efficiency', 0.5),
                    'score': combined_scores.get('itm1', 0.5)
                }
            
            if otm1_result:
                risk_metrics['otm1'] = {
                    'decay': otm1_result.straddle_metrics.get('decay_as_pct', 0),
                    'efficiency': otm1_result.straddle_metrics.get('otm_efficiency', 0.5),
                    'score': combined_scores.get('otm1', 0.5)
                }
            
            # Calculate raw weights based on score / (1 + decay_risk)
            for straddle, metrics in risk_metrics.items():
                decay_factor = 1 + metrics['decay'] / 10  # Penalize high decay
                raw_weight = metrics['score'] / decay_factor
                weights[straddle] = max(raw_weight, 0.1)  # Minimum 10% weight
            
            # Apply diversification bonus
            # If all three straddles are available, boost weights slightly
            if len(weights) == 3:
                for straddle in weights:
                    weights[straddle] *= 1.1
            
        except Exception as e:
            self.logger.error(f"Error calculating risk-adjusted weights: {e}")
        
        return weights
    
    def _calculate_combined_price(self, atm_result: Optional[StraddleAnalysisResult],
                                 itm1_result: Optional[StraddleAnalysisResult],
                                 otm1_result: Optional[StraddleAnalysisResult],
                                 weights: Dict[str, float]) -> float:
        """Calculate weighted combined price"""
        combined_price = 0.0
        
        if atm_result and 'atm' in weights:
            combined_price += atm_result.combined_price * weights['atm']
        
        if itm1_result and 'itm1' in weights:
            combined_price += itm1_result.combined_price * weights['itm1']
        
        if otm1_result and 'otm1' in weights:
            combined_price += otm1_result.combined_price * weights['otm1']
        
        return combined_price
    
    def _calculate_combined_greeks(self, atm_result: Optional[StraddleAnalysisResult],
                                  itm1_result: Optional[StraddleAnalysisResult],
                                  otm1_result: Optional[StraddleAnalysisResult],
                                  weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate weighted combined Greeks"""
        combined_greeks = {
            'net_delta': 0.0,
            'net_gamma': 0.0,
            'net_theta': 0.0,
            'net_vega': 0.0,
            'net_rho': 0.0
        }
        
        # Aggregate Greeks with weights
        if atm_result and 'atm' in weights:
            for greek in combined_greeks:
                combined_greeks[greek] += atm_result.combined_greeks.get(greek, 0) * weights['atm']
        
        if itm1_result and 'itm1' in weights:
            for greek in combined_greeks:
                combined_greeks[greek] += itm1_result.combined_greeks.get(greek, 0) * weights['itm1']
        
        if otm1_result and 'otm1' in weights:
            for greek in combined_greeks:
                combined_greeks[greek] += otm1_result.combined_greeks.get(greek, 0) * weights['otm1']
        
        # Add aggregate metrics
        combined_greeks['total_gamma_exposure'] = abs(combined_greeks['net_gamma'])
        combined_greeks['total_vega_exposure'] = abs(combined_greeks['net_vega'])
        combined_greeks['theta_burn_rate'] = abs(combined_greeks['net_theta'])
        combined_greeks['delta_neutrality'] = 1.0 - min(abs(combined_greeks['net_delta']) / 0.3, 1.0)
        
        return combined_greeks
    
    def _calculate_combined_metrics(self, atm_result: Optional[StraddleAnalysisResult],
                                   itm1_result: Optional[StraddleAnalysisResult],
                                   otm1_result: Optional[StraddleAnalysisResult],
                                   combined_price: float,
                                   combined_greeks: Dict[str, float],
                                   weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate combined triple straddle metrics"""
        metrics = {}
        
        try:
            # Price composition
            metrics['total_premium'] = combined_price
            
            if atm_result:
                metrics['atm_contribution'] = atm_result.combined_price * weights.get('atm', 0)
            if itm1_result:
                metrics['itm1_contribution'] = itm1_result.combined_price * weights.get('itm1', 0)
            if otm1_result:
                metrics['otm1_contribution'] = otm1_result.combined_price * weights.get('otm1', 0)
            
            # Risk metrics
            metrics['total_theta_risk'] = combined_greeks['theta_burn_rate']
            metrics['daily_decay_pct'] = (
                metrics['total_theta_risk'] / combined_price * 100 
                if combined_price > 0 else 0
            )
            
            # Efficiency metrics
            efficiency_sum = 0
            efficiency_count = 0
            
            if atm_result:
                efficiency_sum += atm_result.straddle_metrics.get('straddle_efficiency', 0.5) * weights.get('atm', 0)
                efficiency_count += weights.get('atm', 0)
            if itm1_result:
                efficiency_sum += itm1_result.straddle_metrics.get('itm_efficiency', 0.5) * weights.get('itm1', 0)
                efficiency_count += weights.get('itm1', 0)
            if otm1_result:
                efficiency_sum += otm1_result.straddle_metrics.get('otm_efficiency', 0.5) * weights.get('otm1', 0)
                efficiency_count += weights.get('otm1', 0)
            
            metrics['weighted_efficiency'] = efficiency_sum / efficiency_count if efficiency_count > 0 else 0.5
            
            # Breakeven analysis (weighted average)
            upper_breakeven = 0
            lower_breakeven = 0
            weight_sum = 0
            
            if atm_result and 'atm' in weights:
                upper_breakeven += atm_result.straddle_metrics.get('upper_breakeven', 0) * weights['atm']
                lower_breakeven += atm_result.straddle_metrics.get('lower_breakeven', 0) * weights['atm']
                weight_sum += weights['atm']
            
            if itm1_result and 'itm1' in weights:
                upper_breakeven += itm1_result.straddle_metrics.get('upper_breakeven', 0) * weights['itm1']
                lower_breakeven += itm1_result.straddle_metrics.get('lower_breakeven', 0) * weights['itm1']
                weight_sum += weights['itm1']
            
            if otm1_result and 'otm1' in weights:
                upper_breakeven += otm1_result.straddle_metrics.get('upper_breakeven', 0) * weights['otm1']
                lower_breakeven += otm1_result.straddle_metrics.get('lower_breakeven', 0) * weights['otm1']
                weight_sum += weights['otm1']
            
            if weight_sum > 0:
                metrics['combined_upper_breakeven'] = upper_breakeven / weight_sum
                metrics['combined_lower_breakeven'] = lower_breakeven / weight_sum
                metrics['combined_breakeven_width'] = metrics['combined_upper_breakeven'] - metrics['combined_lower_breakeven']
            
            # Diversification benefit
            metrics['diversification_score'] = self._calculate_diversification_score(
                atm_result, itm1_result, otm1_result, weights
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating combined metrics: {e}")
        
        return metrics
    
    def _calculate_diversification_score(self, atm_result: Optional[StraddleAnalysisResult],
                                       itm1_result: Optional[StraddleAnalysisResult],
                                       otm1_result: Optional[StraddleAnalysisResult],
                                       weights: Dict[str, float]) -> float:
        """Calculate diversification benefit from combining straddles"""
        try:
            # Count active straddles
            active_count = sum(1 for r in [atm_result, itm1_result, otm1_result] if r is not None)
            
            if active_count <= 1:
                return 0.0  # No diversification with single straddle
            
            # Calculate weight concentration (lower is better)
            weight_values = list(weights.values())
            weight_variance = np.var(weight_values)
            concentration_score = 1.0 - min(weight_variance * 10, 1.0)
            
            # Different characteristics provide diversification
            characteristic_diversity = active_count / 3.0  # Normalize by maximum
            
            # Combine scores
            diversification_score = (concentration_score + characteristic_diversity) / 2
            
            return diversification_score
            
        except Exception:
            return 0.5
    
    def _calculate_regime_indicators(self, atm_result: Optional[StraddleAnalysisResult],
                                   itm1_result: Optional[StraddleAnalysisResult],
                                   otm1_result: Optional[StraddleAnalysisResult],
                                   weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate weighted regime indicators"""
        regime_indicators = {}
        
        try:
            # Collect all regime indicators
            all_indicators = {}
            
            if atm_result:
                for key, value in atm_result.regime_indicators.items():
                    all_indicators[f'atm_{key}'] = value * weights.get('atm', 0)
            
            if itm1_result:
                for key, value in itm1_result.regime_indicators.items():
                    all_indicators[f'itm1_{key}'] = value * weights.get('itm1', 0)
            
            if otm1_result:
                for key, value in otm1_result.regime_indicators.items():
                    all_indicators[f'otm1_{key}'] = value * weights.get('otm1', 0)
            
            # Aggregate key regime dimensions
            regime_indicators['volatility_consensus'] = self._aggregate_volatility_indicators(all_indicators)
            regime_indicators['trend_consensus'] = self._aggregate_trend_indicators(all_indicators)
            regime_indicators['structure_consensus'] = self._aggregate_structure_indicators(all_indicators)
            
            # Combined regime scores
            regime_indicators['regime_clarity'] = self._calculate_regime_clarity(all_indicators)
            regime_indicators['regime_stability'] = self._calculate_regime_stability()
            
            # Triple straddle specific indicators
            regime_indicators['straddle_harmony'] = self._calculate_straddle_harmony(
                atm_result, itm1_result, otm1_result
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating regime indicators: {e}")
        
        return regime_indicators
    
    def _aggregate_volatility_indicators(self, indicators: Dict[str, float]) -> float:
        """Aggregate volatility regime indicators"""
        vol_keys = [k for k in indicators if 'volatility' in k.lower() or 'vega' in k.lower()]
        if not vol_keys:
            return 0.0
        
        return np.mean([indicators[k] for k in vol_keys])
    
    def _aggregate_trend_indicators(self, indicators: Dict[str, float]) -> float:
        """Aggregate trend regime indicators"""
        trend_keys = [k for k in indicators if 'trend' in k.lower() or 'momentum' in k.lower()]
        if not trend_keys:
            return 0.0
        
        return np.mean([indicators[k] for k in trend_keys])
    
    def _aggregate_structure_indicators(self, indicators: Dict[str, float]) -> float:
        """Aggregate market structure indicators"""
        structure_keys = [k for k in indicators if 'structure' in k.lower() or 'efficiency' in k.lower()]
        if not structure_keys:
            return 0.0
        
        return np.mean([indicators[k] for k in structure_keys])
    
    def _calculate_regime_clarity(self, indicators: Dict[str, float]) -> float:
        """Calculate how clear/strong the regime signals are"""
        # Higher absolute values indicate clearer regime
        abs_values = [abs(v) for v in indicators.values()]
        return np.mean(abs_values) if abs_values else 0.5
    
    def _calculate_regime_stability(self) -> float:
        """Calculate regime stability from historical data"""
        if len(self.performance_history) < 3:
            return 0.5  # Neutral if insufficient history
        
        # Check regime consistency over recent history
        recent_regimes = [h.market_regime for h in self.performance_history[-5:]]
        if len(set(recent_regimes)) == 1:
            return 1.0  # Very stable
        elif len(set(recent_regimes)) == 2:
            return 0.7  # Moderately stable
        else:
            return 0.3  # Unstable
    
    def _calculate_straddle_harmony(self, atm_result: Optional[StraddleAnalysisResult],
                                   itm1_result: Optional[StraddleAnalysisResult],
                                   otm1_result: Optional[StraddleAnalysisResult]) -> float:
        """Calculate how well the straddles work together"""
        harmony_score = 0.5
        
        try:
            # Check if all straddles point in same direction
            signals = []
            
            if atm_result:
                signals.append(atm_result.strategy_signals.get('entry_signal', 0.5))
            if itm1_result:
                signals.append(itm1_result.strategy_signals.get('entry_signal', 0.5))
            if otm1_result:
                signals.append(otm1_result.strategy_signals.get('entry_signal', 0.5))
            
            if len(signals) > 1:
                # Low variance = high harmony
                signal_variance = np.var(signals)
                harmony_score = 1.0 - min(signal_variance * 4, 1.0)
            
        except Exception:
            pass
        
        return harmony_score
    
    def _determine_market_regime(self, regime_indicators: Dict[str, float]) -> Tuple[str, float]:
        """Determine overall market regime from indicators"""
        volatility = regime_indicators.get('volatility_consensus', 0)
        trend = regime_indicators.get('trend_consensus', 0)
        structure = regime_indicators.get('structure_consensus', 0)
        
        # Classify volatility
        if volatility > self.regime_thresholds['volatility']['high']:
            vol_regime = 'HIGH_VOL'
        elif volatility < self.regime_thresholds['volatility']['low']:
            vol_regime = 'LOW_VOL'
        else:
            vol_regime = 'MEDIUM_VOL'
        
        # Classify trend
        if trend > self.regime_thresholds['trend']['high']:
            trend_regime = 'TRENDING_UP'
        elif trend < self.regime_thresholds['trend']['low']:
            trend_regime = 'TRENDING_DOWN'
        else:
            trend_regime = 'RANGING'
        
        # Classify structure
        if structure > self.regime_thresholds['structure']['high']:
            structure_regime = 'STRUCTURED'
        elif structure < self.regime_thresholds['structure']['low']:
            structure_regime = 'CHOPPY'
        else:
            structure_regime = 'MIXED'
        
        # Combine into overall regime
        market_regime = f"{vol_regime}_{trend_regime}_{structure_regime}"
        
        # Calculate confidence
        clarity = regime_indicators.get('regime_clarity', 0.5)
        stability = regime_indicators.get('regime_stability', 0.5)
        harmony = regime_indicators.get('straddle_harmony', 0.5)
        
        regime_confidence = (clarity + stability + harmony) / 3
        
        return market_regime, regime_confidence
    
    def _calculate_strategy_signals(self, atm_result: Optional[StraddleAnalysisResult],
                                   itm1_result: Optional[StraddleAnalysisResult],
                                   otm1_result: Optional[StraddleAnalysisResult],
                                   combined_metrics: Dict[str, float],
                                   regime_indicators: Dict[str, float],
                                   weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate unified strategy signals"""
        signals = {}
        
        try:
            # Aggregate entry signals
            entry_sum = 0
            entry_weight = 0
            
            if atm_result and 'atm' in weights:
                entry_sum += atm_result.strategy_signals.get('entry_signal', 0.5) * weights['atm']
                entry_weight += weights['atm']
            
            if itm1_result and 'itm1' in weights:
                entry_sum += itm1_result.strategy_signals.get('entry_signal', 0.5) * weights['itm1']
                entry_weight += weights['itm1']
            
            if otm1_result and 'otm1' in weights:
                entry_sum += otm1_result.strategy_signals.get('entry_signal', 0.5) * weights['otm1']
                entry_weight += weights['otm1']
            
            signals['combined_entry_signal'] = entry_sum / entry_weight if entry_weight > 0 else 0.5
            
            # Position sizing
            signals['position_size_recommendation'] = self._calculate_position_size(
                combined_metrics, regime_indicators
            )
            
            # Straddle allocation signals
            signals['atm_allocation'] = weights.get('atm', 0)
            signals['itm1_allocation'] = weights.get('itm1', 0)
            signals['otm1_allocation'] = weights.get('otm1', 0)
            
            # Risk management signals
            decay_risk = combined_metrics.get('daily_decay_pct', 0)
            if decay_risk > 5:
                signals['decay_alert'] = 1.0
            elif decay_risk > 3:
                signals['decay_alert'] = 0.5
            else:
                signals['decay_alert'] = 0.0
            
            # Regime-based adjustments
            vol_regime = regime_indicators.get('volatility_consensus', 0)
            if vol_regime > 0.5:
                signals['volatility_opportunity'] = 1.0
                signals['suggested_adjustment'] = 'INCREASE_OTM'
            elif vol_regime < -0.5:
                signals['volatility_opportunity'] = -1.0
                signals['suggested_adjustment'] = 'INCREASE_ITM'
            else:
                signals['volatility_opportunity'] = 0.0
                signals['suggested_adjustment'] = 'MAINTAIN_BALANCE'
            
            # Overall recommendation
            if signals['combined_entry_signal'] > 0.7:
                signals['recommendation'] = 'STRONG_BUY'
            elif signals['combined_entry_signal'] > 0.5:
                signals['recommendation'] = 'BUY'
            elif signals['combined_entry_signal'] < 0.3:
                signals['recommendation'] = 'AVOID'
            else:
                signals['recommendation'] = 'NEUTRAL'
            
        except Exception as e:
            self.logger.error(f"Error calculating strategy signals: {e}")
        
        return signals
    
    def _calculate_position_size(self, combined_metrics: Dict[str, float],
                                regime_indicators: Dict[str, float]) -> float:
        """Calculate recommended position size"""
        try:
            size_factors = []
            
            # Efficiency factor
            efficiency = combined_metrics.get('weighted_efficiency', 0.5)
            size_factors.append(efficiency)
            
            # Decay risk factor
            decay_pct = combined_metrics.get('daily_decay_pct', 0)
            if decay_pct < 2:
                size_factors.append(1.0)
            elif decay_pct < 5:
                size_factors.append(0.7)
            else:
                size_factors.append(0.4)
            
            # Regime clarity factor
            clarity = regime_indicators.get('regime_clarity', 0.5)
            size_factors.append(clarity)
            
            # Diversification benefit
            diversification = combined_metrics.get('diversification_score', 0.5)
            size_factors.append(0.5 + diversification * 0.5)  # Boost for diversification
            
            return np.mean(size_factors) if size_factors else 0.5
            
        except Exception:
            return 0.5
    
    def _update_performance_history(self, result: CombinedStraddleResult):
        """Update performance history"""
        self.performance_history.append(result)
        
        # Maintain history size
        if len(self.performance_history) > self.max_history_length:
            self.performance_history.pop(0)
    
    def get_analyzer_status(self) -> Dict[str, Any]:
        """Get combined analyzer status"""
        return {
            'default_weights': self.default_weights,
            'atm_status': self.atm_analyzer.get_straddle_status(),
            'itm1_status': self.itm1_analyzer.get_straddle_status(),
            'otm1_status': self.otm1_analyzer.get_straddle_status(),
            'regime_thresholds': self.regime_thresholds,
            'history_length': len(self.performance_history),
            'latest_regime': self.performance_history[-1].market_regime if self.performance_history else None
        }
    
    def get_performance_summary(self, periods: int = 20) -> Dict[str, Any]:
        """Get performance summary over recent periods"""
        if not self.performance_history:
            return {}
        
        recent_history = self.performance_history[-periods:]
        
        return {
            'average_efficiency': np.mean([h.combined_metrics.get('weighted_efficiency', 0.5) 
                                          for h in recent_history]),
            'regime_distribution': self._get_regime_distribution(recent_history),
            'average_weights': self._get_average_weights(recent_history),
            'performance_stability': self._calculate_performance_stability(recent_history)
        }
    
    def _get_regime_distribution(self, history: List[CombinedStraddleResult]) -> Dict[str, float]:
        """Get distribution of market regimes"""
        regime_counts = {}
        
        for result in history:
            regime = result.market_regime
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        total = len(history)
        return {regime: count/total for regime, count in regime_counts.items()}
    
    def _get_average_weights(self, history: List[CombinedStraddleResult]) -> Dict[str, float]:
        """Get average straddle weights"""
        weight_sums = {'atm': 0, 'itm1': 0, 'otm1': 0}
        
        for result in history:
            for straddle in weight_sums:
                weight_sums[straddle] += result.optimal_weights.get(straddle, 0)
        
        count = len(history)
        return {straddle: total/count for straddle, total in weight_sums.items()}
    
    def _calculate_performance_stability(self, history: List[CombinedStraddleResult]) -> float:
        """Calculate performance stability metric"""
        if len(history) < 2:
            return 0.5
        
        efficiencies = [h.combined_metrics.get('weighted_efficiency', 0.5) for h in history]
        return 1.0 - min(np.std(efficiencies) * 2, 1.0)  # Lower std = higher stability