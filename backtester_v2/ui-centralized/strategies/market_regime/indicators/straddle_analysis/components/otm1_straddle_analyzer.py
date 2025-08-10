"""
OTM1 Straddle Combination Analyzer

Combines OTM1 Call (CE) and OTM1 Put (PE) components to analyze the complete
OTM1 straddle strategy. Focuses on high leverage, lottery ticket characteristics,
and tail risk protection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
from .otm1_ce_analyzer import OTM1CallAnalyzer
from .otm1_pe_analyzer import OTM1PutAnalyzer
from .atm_straddle_analyzer import StraddleAnalysisResult
from ..core.calculation_engine import CalculationEngine
from ..rolling.window_manager import RollingWindowManager
from ..config.excel_reader import StraddleConfig
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class OTM1StraddleAnalyzer:
    """
    OTM1 Straddle Analyzer
    
    Combines OTM1 Call and Put analysis for OTM straddle strategy.
    Key characteristics:
    - Pure time value (no intrinsic value)
    - High leverage and gamma
    - Lottery ticket upside with tail risk protection
    - Highest theta decay rate
    - Maximum sensitivity to volatility changes
    """
    
    def __init__(self,
                 config: StraddleConfig,
                 calculation_engine: CalculationEngine,
                 window_manager: RollingWindowManager):
        """
        Initialize OTM1 Straddle analyzer
        
        Args:
            config: Straddle configuration
            calculation_engine: Shared calculation engine
            window_manager: Rolling window manager
        """
        self.config = config
        self.calculation_engine = calculation_engine
        self.window_manager = window_manager
        
        # Initialize component analyzers
        self.call_analyzer = OTM1CallAnalyzer(config, calculation_engine, window_manager)
        self.put_analyzer = OTM1PutAnalyzer(config, calculation_engine, window_manager)
        
        # Straddle configuration
        self.straddle_name = 'otm1_straddle'
        self.call_weight = config.component_weights.get('otm1_ce', 0.5)
        self.put_weight = config.component_weights.get('otm1_pe', 0.5)
        
        # Ensure weights sum to 1 for the straddle
        weight_sum = self.call_weight + self.put_weight
        if weight_sum > 0:
            self.call_weight /= weight_sum
            self.put_weight /= weight_sum
        
        # OTM-specific thresholds
        self.max_time_value_ratio = 1.0    # Should be 100% time value
        self.min_leverage_ratio = 10.0     # Minimum leverage expected
        self.decay_warning_threshold = 5.0  # Daily decay % warning
        self.probability_threshold = 0.3    # Minimum probability for consideration
        
        self.logger = logging.getLogger(f"{__name__}.{self.straddle_name}")
        self.logger.info(f"OTM1 Straddle analyzer initialized (CE: {self.call_weight:.2f}, PE: {self.put_weight:.2f})")
    
    def analyze(self, data: Dict[str, Any], timestamp: pd.Timestamp) -> Optional[StraddleAnalysisResult]:
        """
        Perform comprehensive OTM1 straddle analysis
        
        Args:
            data: Market data dictionary
            timestamp: Current timestamp
            
        Returns:
            StraddleAnalysisResult or None if insufficient data
        """
        try:
            # Analyze individual components
            call_result = self.call_analyzer.analyze(data, timestamp)
            put_result = self.put_analyzer.analyze(data, timestamp)
            
            # Both components must be available
            if not call_result or not put_result:
                self.logger.warning("Insufficient data for OTM1 straddle analysis")
                return None
            
            # Calculate combined price
            combined_price = (
                call_result.current_price * self.call_weight + 
                put_result.current_price * self.put_weight
            )
            
            # Calculate combined Greeks
            combined_greeks = self._calculate_combined_greeks(call_result, put_result)
            
            # Calculate straddle-specific metrics
            straddle_metrics = self._calculate_straddle_metrics(
                call_result, put_result, combined_price, combined_greeks, data
            )
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                call_result, put_result, combined_price, straddle_metrics
            )
            
            # Calculate strategy signals
            strategy_signals = self._calculate_strategy_signals(
                call_result, put_result, straddle_metrics, combined_greeks
            )
            
            # Create result object
            result = StraddleAnalysisResult(
                straddle_name=self.straddle_name,
                timestamp=timestamp,
                call_result=call_result,
                put_result=put_result,
                combined_price=combined_price,
                combined_greeks=combined_greeks,
                straddle_metrics=straddle_metrics,
                regime_indicators={},  # Will be filled below
                performance_metrics=performance_metrics,
                strategy_signals=strategy_signals
            )
            
            # Calculate regime contribution
            result.regime_indicators = self._calculate_regime_contribution(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in OTM1 straddle analysis: {e}")
            return None
    
    def _calculate_combined_greeks(self, call_result: Any, put_result: Any) -> Dict[str, float]:
        """Calculate combined Greeks for OTM1 straddle"""
        combined_greeks = {}
        
        # Extract Greeks from both legs
        call_greeks = call_result.component_metrics
        put_greeks = put_result.component_metrics
        
        # Delta: OTM straddle has low absolute deltas
        call_delta = call_greeks.get('delta', 0.3)    # Lower for OTM call
        put_delta = put_greeks.get('delta', -0.3)     # Less negative for OTM put
        combined_greeks['net_delta'] = (
            call_delta * self.call_weight + 
            put_delta * self.put_weight
        )
        combined_greeks['gross_delta'] = abs(call_delta) + abs(put_delta)
        combined_greeks['delta_ratio'] = abs(call_delta / put_delta) if put_delta != 0 else 1.0
        
        # Gamma: Higher for OTM near expiry
        call_gamma = call_greeks.get('gamma', 0)
        put_gamma = put_greeks.get('gamma', 0)
        combined_greeks['net_gamma'] = (
            call_gamma * self.call_weight + 
            put_gamma * self.put_weight
        )
        combined_greeks['gamma_intensity'] = combined_greeks['net_gamma'] * 100  # Per 1% move
        
        # Theta: Highest decay for OTM
        call_theta = call_greeks.get('theta', 0)
        put_theta = put_greeks.get('theta', 0)
        combined_greeks['net_theta'] = (
            call_theta * self.call_weight + 
            put_theta * self.put_weight
        )
        
        # Vega: High sensitivity to volatility
        call_vega = call_greeks.get('vega', 0)
        put_vega = put_greeks.get('vega', 0)
        combined_greeks['net_vega'] = (
            call_vega * self.call_weight + 
            put_vega * self.put_weight
        )
        
        # OTM-specific Greek metrics
        combined_greeks['theta_to_premium_ratio'] = (
            abs(combined_greeks['net_theta']) / call_result.current_price 
            if call_result.current_price > 0 else 0
        )
        combined_greeks['vega_to_premium_ratio'] = (
            combined_greeks['net_vega'] / call_result.current_price 
            if call_result.current_price > 0 else 0
        )
        
        return combined_greeks
    
    def _calculate_straddle_metrics(self, call_result: Any, put_result: Any,
                                   combined_price: float, combined_greeks: Dict[str, float],
                                   data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate OTM1 straddle-specific metrics"""
        metrics = {}
        
        try:
            # Price metrics
            metrics['straddle_premium'] = combined_price
            metrics['call_premium'] = call_result.current_price
            metrics['put_premium'] = put_result.current_price
            metrics['premium_balance'] = (
                call_result.current_price / put_result.current_price 
                if put_result.current_price > 0 else 1.0
            )
            
            # Time value confirmation (should be 100% for OTM)
            metrics['time_value'] = combined_price  # All premium is time value
            metrics['time_value_ratio'] = 1.0       # Always 100% for OTM
            
            # Get underlying price and strikes
            underlying_price = data.get('underlying_price') or data.get('spot_price', 0)
            call_distance = call_result.component_metrics.get('distance_to_strike', 0)
            put_distance = put_result.component_metrics.get('distance_to_strike', 0)
            
            # Breakeven analysis
            if underlying_price > 0:
                # OTM strikes
                otm_call_strike = underlying_price + call_distance
                otm_put_strike = underlying_price - put_distance
                
                # Breakevens are further out for OTM
                metrics['upper_breakeven'] = otm_call_strike + combined_price
                metrics['lower_breakeven'] = otm_put_strike - combined_price
                metrics['breakeven_width'] = metrics['upper_breakeven'] - metrics['lower_breakeven']
                metrics['breakeven_width_pct'] = (metrics['breakeven_width'] / underlying_price) * 100
                
                # Distance to profit
                metrics['upper_profit_distance'] = (metrics['upper_breakeven'] - underlying_price) / underlying_price * 100
                metrics['lower_profit_distance'] = (underlying_price - metrics['lower_breakeven']) / underlying_price * 100
                metrics['min_profit_distance'] = min(metrics['upper_profit_distance'], metrics['lower_profit_distance'])
            
            # Leverage analysis
            call_leverage = call_result.component_metrics.get('leverage_ratio', 10)
            put_hedge_efficiency = put_result.component_metrics.get('hedge_efficiency', 0.5)
            
            metrics['upside_leverage'] = call_leverage
            metrics['downside_protection_efficiency'] = put_hedge_efficiency
            metrics['combined_leverage'] = (call_leverage + 1/put_hedge_efficiency) / 2 if put_hedge_efficiency > 0 else call_leverage
            
            # Probability analysis
            call_prob = call_result.component_metrics.get('probability_itm', 0.2)
            put_prob = put_result.component_metrics.get('probability_itm', 0.2)  # Note: This would be probability of finishing ITM
            
            metrics['call_probability'] = call_prob
            metrics['put_probability'] = put_prob
            metrics['any_side_probability'] = call_prob + put_prob - (call_prob * put_prob)  # P(A or B)
            metrics['both_sides_probability'] = call_prob * put_prob  # P(A and B) - very low
            
            # Lottery value
            call_lottery = call_result.component_metrics.get('lottery_ticket_value', 1.0)
            put_insurance = put_result.component_metrics.get('insurance_value', 10.0)
            
            metrics['lottery_value'] = call_lottery
            metrics['insurance_value'] = put_insurance
            metrics['risk_reward_score'] = (call_lottery + put_insurance) / 2
            
            # Time decay analysis
            metrics['daily_decay'] = abs(combined_greeks['net_theta'])
            metrics['decay_as_pct'] = (metrics['daily_decay'] / combined_price * 100) if combined_price > 0 else 0
            metrics['days_to_50pct_decay'] = 50 / metrics['decay_as_pct'] if metrics['decay_as_pct'] > 0 else float('inf')
            
            # Skew analysis
            metrics['put_call_skew'] = put_result.component_metrics.get('put_call_skew', 0)
            metrics['fear_gauge'] = put_result.component_metrics.get('fear_gauge', 0.5)
            
            # OTM efficiency
            metrics['otm_efficiency'] = self._calculate_otm_efficiency(
                call_result.component_metrics, put_result.component_metrics, metrics
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating OTM1 straddle metrics: {e}")
        
        return metrics
    
    def _calculate_otm_efficiency(self, call_metrics: Dict[str, float],
                                 put_metrics: Dict[str, float],
                                 straddle_metrics: Dict[str, float]) -> float:
        """Calculate OTM straddle efficiency"""
        try:
            efficiency_factors = []
            
            # Probability efficiency
            any_prob = straddle_metrics.get('any_side_probability', 0)
            if any_prob > 0.4:  # >40% chance of profit
                efficiency_factors.append(1.0)
            elif any_prob < 0.2:  # <20% chance
                efficiency_factors.append(0.3)
            else:
                efficiency_factors.append((any_prob - 0.2) / 0.2)
            
            # Liquidity efficiency
            call_liquidity = call_metrics.get('liquidity_score', 0.3)
            put_liquidity = put_metrics.get('liquidity_score', 0.4)
            avg_liquidity = (call_liquidity + put_liquidity) / 2
            efficiency_factors.append(avg_liquidity)
            
            # Risk-reward efficiency
            risk_reward = straddle_metrics.get('risk_reward_score', 1.0)
            if risk_reward > 5:
                efficiency_factors.append(1.0)
            elif risk_reward < 1:
                efficiency_factors.append(0.2)
            else:
                efficiency_factors.append((risk_reward - 1) / 4)
            
            # Decay efficiency (lower is better for OTM)
            decay_pct = straddle_metrics.get('decay_as_pct', 0)
            if decay_pct < 3:
                efficiency_factors.append(1.0)
            elif decay_pct > 10:
                efficiency_factors.append(0.2)
            else:
                efficiency_factors.append(1.0 - (decay_pct - 3) / 7)
            
            return np.mean(efficiency_factors) if efficiency_factors else 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_performance_metrics(self, call_result: Any, put_result: Any,
                                     combined_price: float,
                                     straddle_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate OTM1 straddle performance metrics"""
        metrics = {}
        
        try:
            # Price performance
            call_change_pct = call_result.price_change_percent
            put_change_pct = put_result.price_change_percent
            
            # OTM options have amplified moves
            metrics['straddle_return_pct'] = (
                call_change_pct * self.call_weight + 
                put_change_pct * self.put_weight
            )
            
            # Leverage-adjusted returns
            call_leverage = straddle_metrics.get('upside_leverage', 10)
            metrics['leveraged_call_return'] = call_change_pct * min(call_leverage / 10, 3)  # Cap at 3x
            metrics['leveraged_put_return'] = put_change_pct * 2  # Puts have protection value
            
            # Winner analysis
            if abs(call_change_pct) > abs(put_change_pct):
                metrics['winning_side'] = 'CALL'
                metrics['winning_return'] = call_change_pct
                metrics['losing_return'] = put_change_pct
            else:
                metrics['winning_side'] = 'PUT'
                metrics['winning_return'] = put_change_pct
                metrics['losing_return'] = call_change_pct
            
            metrics['return_spread'] = abs(call_change_pct - put_change_pct)
            
            # Risk metrics
            metrics['max_loss'] = combined_price  # 100% loss possible
            metrics['max_loss_pct'] = 100.0
            metrics['current_risk'] = combined_price * (1 - metrics['straddle_return_pct'] / 100)
            
            # Time decay impact
            daily_decay = straddle_metrics.get('daily_decay', 0)
            metrics['theta_impact'] = daily_decay / combined_price * 100 if combined_price > 0 else 0
            metrics['breakeven_days'] = straddle_metrics.get('days_to_50pct_decay', 30)
            
            # Volatility performance
            for window in ['3min', '5min', '10min', '15min']:
                if window in call_result.rolling_metrics:
                    # OTM straddles need high realized volatility
                    call_vol = call_result.rolling_metrics[window].get(f'volatility_{window}', 0)
                    put_vol = put_result.rolling_metrics[window].get(f'volatility_{window}', 0)
                    avg_vol = (call_vol + put_vol) / 2
                    
                    # Compare to breakeven requirement
                    breakeven_vol = straddle_metrics.get('min_profit_distance', 5) / 100
                    metrics[f'vol_to_breakeven_ratio_{window}'] = avg_vol / breakeven_vol if breakeven_vol > 0 else 0
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
        
        return metrics
    
    def _calculate_strategy_signals(self, call_result: Any, put_result: Any,
                                  straddle_metrics: Dict[str, float],
                                  combined_greeks: Dict[str, float]) -> Dict[str, float]:
        """Calculate trading signals for OTM1 straddle strategy"""
        signals = {}
        
        try:
            # Efficiency signal
            efficiency = straddle_metrics.get('otm_efficiency', 0.5)
            if efficiency > 0.7:
                signals['efficiency_signal'] = 1.0
            elif efficiency < 0.3:
                signals['efficiency_signal'] = -1.0
            else:
                signals['efficiency_signal'] = (efficiency - 0.3) / 0.4
            
            # Probability signal
            any_prob = straddle_metrics.get('any_side_probability', 0)
            if any_prob > self.probability_threshold:
                signals['probability_signal'] = 1.0
            else:
                signals['probability_signal'] = any_prob / self.probability_threshold
            
            # Decay warning (critical for OTM)
            decay_pct = straddle_metrics.get('decay_as_pct', 0)
            if decay_pct > self.decay_warning_threshold:
                signals['decay_critical'] = 1.0
            elif decay_pct > 3:
                signals['decay_critical'] = (decay_pct - 3) / (self.decay_warning_threshold - 3)
            else:
                signals['decay_critical'] = 0.0
            
            # Leverage opportunity
            combined_leverage = straddle_metrics.get('combined_leverage', 10)
            if combined_leverage > 20:
                signals['leverage_opportunity'] = 1.0
            elif combined_leverage < self.min_leverage_ratio:
                signals['leverage_opportunity'] = -1.0
            else:
                signals['leverage_opportunity'] = (combined_leverage - 10) / 10
            
            # Skew signal (put-call skew)
            skew = straddle_metrics.get('put_call_skew', 0)
            fear = straddle_metrics.get('fear_gauge', 0.5)
            
            if fear > 0.7:  # High fear
                signals['market_sentiment'] = -1.0  # Bearish
                signals['protection_value'] = 1.0   # Puts valuable
            elif fear < 0.3:  # Low fear
                signals['market_sentiment'] = 1.0   # Bullish
                signals['protection_value'] = 0.0   # Puts cheap
            else:
                signals['market_sentiment'] = 0.0
                signals['protection_value'] = fear
            
            # Volatility requirement
            vega = combined_greeks.get('net_vega', 0)
            signals['volatility_need'] = min(vega / 100, 1.0)  # High vega = needs volatility
            
            # Entry/exit signals
            signals['entry_signal'] = self._calculate_entry_signal(
                straddle_metrics, combined_greeks, signals
            )
            signals['exit_signal'] = self._calculate_exit_signal(
                straddle_metrics, combined_greeks, signals
            )
            
            # Position sizing (smaller for OTM due to high risk)
            signals['position_size_factor'] = self._calculate_position_size_factor(
                straddle_metrics, signals
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating strategy signals: {e}")
        
        return signals
    
    def _calculate_entry_signal(self, metrics: Dict[str, float],
                               greeks: Dict[str, float],
                               signals: Dict[str, float]) -> float:
        """Calculate OTM straddle entry signal"""
        try:
            entry_factors = []
            
            # Efficiency check
            if signals.get('efficiency_signal', 0) > 0:
                entry_factors.append(1.0)
            elif signals.get('efficiency_signal', 0) < 0:
                entry_factors.append(0.0)
            else:
                entry_factors.append(0.5)
            
            # Probability check
            prob_signal = signals.get('probability_signal', 0)
            entry_factors.append(prob_signal)
            
            # Decay not critical
            if signals.get('decay_critical', 0) < 0.5:
                entry_factors.append(1.0)
            else:
                entry_factors.append(0.2)
            
            # Good leverage opportunity
            if signals.get('leverage_opportunity', 0) > 0:
                entry_factors.append(0.8)
            else:
                entry_factors.append(0.3)
            
            # Volatility expectation
            if signals.get('volatility_need', 0) > 0.5:
                # High vega means we need volatility event
                entry_factors.append(0.7)
            
            return np.mean(entry_factors) if entry_factors else 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_exit_signal(self, metrics: Dict[str, float],
                              greeks: Dict[str, float],
                              signals: Dict[str, float]) -> float:
        """Calculate OTM straddle exit signal"""
        try:
            exit_factors = []
            
            # Critical decay
            if signals.get('decay_critical', 0) > 0.8:
                exit_factors.append(1.0)
            
            # Lost efficiency
            if signals.get('efficiency_signal', 0) < -0.5:
                exit_factors.append(0.8)
            
            # Probability too low
            if signals.get('probability_signal', 0) < 0.5:
                exit_factors.append(0.7)
            
            # Hit profit target or stop loss
            return_pct = metrics.get('straddle_return_pct', 0)
            if return_pct > 50:  # 50% profit on OTM
                exit_factors.append(1.0)
            elif return_pct < -70:  # 70% loss
                exit_factors.append(1.0)
            
            # Days to significant decay
            days_to_decay = metrics.get('days_to_50pct_decay', 30)
            if days_to_decay < 5:
                exit_factors.append(0.9)
            
            return np.mean(exit_factors) if exit_factors else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_position_size_factor(self, metrics: Dict[str, float],
                                       signals: Dict[str, float]) -> float:
        """Calculate position sizing factor for OTM straddle"""
        try:
            size_factors = []
            
            # Base size on efficiency
            efficiency = metrics.get('otm_efficiency', 0.5)
            size_factors.append(efficiency * 0.5)  # Max 50% for OTM
            
            # Reduce for high decay
            if signals.get('decay_critical', 0) > 0.5:
                size_factors.append(0.3)
            else:
                size_factors.append(0.6)
            
            # Probability adjustment
            prob = metrics.get('any_side_probability', 0.2)
            size_factors.append(min(prob * 2, 1.0))  # Scale up low probabilities
            
            # Risk-reward adjustment
            risk_reward = metrics.get('risk_reward_score', 1.0)
            if risk_reward > 3:
                size_factors.append(0.7)
            else:
                size_factors.append(0.4)
            
            return np.mean(size_factors) if size_factors else 0.3
            
        except Exception:
            return 0.3
    
    def _calculate_regime_contribution(self, result: StraddleAnalysisResult) -> Dict[str, float]:
        """Calculate OTM1 straddle contribution to regime formation"""
        regime_indicators = {}
        
        try:
            # Component regime indicators
            call_regime = result.call_result.regime_indicators
            put_regime = result.put_result.regime_indicators
            
            # Volatility regime (OTM is pure volatility play)
            volatility_indicators = self._aggregate_volatility_regime(
                call_regime, put_regime, result.straddle_metrics, result.combined_greeks
            )
            regime_indicators.update(volatility_indicators)
            
            # Trend regime (OTM benefits from large moves)
            trend_indicators = self._aggregate_trend_regime(
                call_regime, put_regime, result.straddle_metrics
            )
            regime_indicators.update(trend_indicators)
            
            # Structure regime
            structure_indicators = self._aggregate_structure_regime(
                call_regime, put_regime, result.straddle_metrics
            )
            regime_indicators.update(structure_indicators)
            
            # OTM-specific regime
            otm_indicators = self._calculate_otm_regime(
                result.straddle_metrics, result.combined_greeks, result.strategy_signals
            )
            regime_indicators.update(otm_indicators)
            
            # Overall assessment
            regime_indicators['regime_type'] = self._determine_regime_type(regime_indicators)
            regime_indicators['regime_strength'] = self._calculate_regime_strength(regime_indicators)
            
        except Exception as e:
            self.logger.error(f"Error calculating regime contribution: {e}")
        
        return regime_indicators
    
    def _aggregate_volatility_regime(self, call_regime: Dict[str, float],
                                    put_regime: Dict[str, float],
                                    straddle_metrics: Dict[str, float],
                                    combined_greeks: Dict[str, float]) -> Dict[str, float]:
        """Aggregate volatility regime for OTM straddle"""
        indicators = {}
        
        # OTM straddles are most sensitive to volatility
        vega = combined_greeks.get('net_vega', 0)
        vega_ratio = combined_greeks.get('vega_to_premium_ratio', 0)
        
        if vega_ratio > 1.0:  # Vega > Premium
            indicators['otm_volatility_sensitivity'] = 1.0
        elif vega_ratio < 0.3:
            indicators['otm_volatility_sensitivity'] = -1.0
        else:
            indicators['otm_volatility_sensitivity'] = (vega_ratio - 0.3) / 0.7
        
        # Volatility opportunity from components
        call_vol_signal = call_regime.get('otm_volatility_signal', 0)
        put_vol_signal = put_regime.get('put_volatility_signal', 0)
        indicators['volatility_opportunity'] = (call_vol_signal + put_vol_signal) / 2
        
        # Fear/greed from put skew
        fear_gauge = straddle_metrics.get('fear_gauge', 0.5)
        if fear_gauge > 0.7:
            indicators['volatility_regime'] = 1.0  # High volatility expected
        elif fear_gauge < 0.3:
            indicators['volatility_regime'] = -1.0  # Low volatility expected
        else:
            indicators['volatility_regime'] = (fear_gauge - 0.3) / 0.4
        
        # Tail risk premium
        tail_premium = put_regime.get('tail_risk_volatility', 0)
        indicators['tail_risk_regime'] = tail_premium
        
        return indicators
    
    def _aggregate_trend_regime(self, call_regime: Dict[str, float],
                               put_regime: Dict[str, float],
                               straddle_metrics: Dict[str, float]) -> Dict[str, float]:
        """Aggregate trend regime for OTM straddle"""
        indicators = {}
        
        # OTM straddles need explosive moves
        call_momentum = call_regime.get('otm_momentum_signal', 0)
        put_momentum = put_regime.get('put_momentum_signal', 0)
        
        # Best when both sides show potential
        indicators['explosive_move_potential'] = max(abs(call_momentum), abs(put_momentum))
        
        # Lottery value indicates trend opportunity
        lottery_value = straddle_metrics.get('lottery_value', 1.0)
        if lottery_value > 2:
            indicators['trend_opportunity'] = 1.0
        elif lottery_value < 0.5:
            indicators['trend_opportunity'] = -1.0
        else:
            indicators['trend_opportunity'] = (lottery_value - 0.5) / 1.5
        
        # Market sentiment
        sentiment = call_regime.get('market_sentiment', 0)
        indicators['directional_bias'] = sentiment
        
        # Breakeven requirement
        min_distance = straddle_metrics.get('min_profit_distance', 5)
        if min_distance > 7:  # Needs >7% move
            indicators['movement_requirement'] = 1.0  # High movement needed
        elif min_distance < 3:
            indicators['movement_requirement'] = -1.0  # Low movement needed
        else:
            indicators['movement_requirement'] = (min_distance - 3) / 4
        
        return indicators
    
    def _aggregate_structure_regime(self, call_regime: Dict[str, float],
                                   put_regime: Dict[str, float],
                                   straddle_metrics: Dict[str, float]) -> Dict[str, float]:
        """Aggregate structure regime for OTM straddle"""
        indicators = {}
        
        # Liquidity structure
        call_liquidity = call_regime.get('otm_liquidity_structure', 0)
        put_liquidity = put_regime.get('put_market_structure', 0)
        indicators['otm_market_quality'] = (call_liquidity + put_liquidity) / 2
        
        # Speculation level
        call_speculation = call_regime.get('speculation_regime', 0)
        indicators['speculation_intensity'] = call_speculation
        
        # Protection demand
        protection_efficiency = put_regime.get('protection_efficiency', 0)
        indicators['hedging_demand'] = protection_efficiency
        
        # Market accessibility
        call_access = call_regime.get('market_accessibility', 0.5)
        put_access = put_regime.get('hedging_accessibility', 0.5)
        indicators['otm_accessibility'] = (call_access + put_access) / 2
        
        return indicators
    
    def _calculate_otm_regime(self, straddle_metrics: Dict[str, float],
                             combined_greeks: Dict[str, float],
                             strategy_signals: Dict[str, float]) -> Dict[str, float]:
        """Calculate OTM-specific regime indicators"""
        indicators = {}
        
        # Decay regime
        decay_critical = strategy_signals.get('decay_critical', 0)
        if decay_critical > 0.7:
            indicators['decay_regime'] = -1.0  # Critical decay
        elif decay_critical < 0.3:
            indicators['decay_regime'] = 1.0   # Manageable decay
        else:
            indicators['decay_regime'] = 1.0 - (decay_critical * 2)
        
        # Leverage regime
        leverage_opp = strategy_signals.get('leverage_opportunity', 0)
        indicators['leverage_regime'] = leverage_opp
        
        # Probability regime
        prob_signal = strategy_signals.get('probability_signal', 0)
        indicators['probability_regime'] = prob_signal
        
        # Risk-reward regime
        risk_reward = straddle_metrics.get('risk_reward_score', 1.0)
        if risk_reward > 5:
            indicators['risk_reward_regime'] = 1.0
        elif risk_reward < 2:
            indicators['risk_reward_regime'] = -1.0
        else:
            indicators['risk_reward_regime'] = (risk_reward - 2) / 3
        
        # OTM efficiency regime
        efficiency = straddle_metrics.get('otm_efficiency', 0.5)
        indicators['otm_efficiency_regime'] = (efficiency - 0.5) * 2  # Center at 0.5
        
        return indicators
    
    def _determine_regime_type(self, indicators: Dict[str, float]) -> str:
        """Determine OTM straddle regime type"""
        volatility_score = indicators.get('volatility_regime', 0)
        movement_req = indicators.get('movement_requirement', 0)
        decay_regime = indicators.get('decay_regime', 0)
        speculation = indicators.get('speculation_intensity', 0)
        
        if volatility_score > 0.5:
            if movement_req > 0.5:
                return 'HIGH_VOL_EXPLOSIVE'  # Best for OTM
            else:
                return 'HIGH_VOL_CONTAINED'
        else:
            if decay_regime < -0.5:
                return 'LOW_VOL_DECAY'  # Worst for OTM
            elif speculation > 0.5:
                return 'LOW_VOL_SPECULATIVE'
            else:
                return 'LOW_VOL_DORMANT'
    
    def _calculate_regime_strength(self, indicators: Dict[str, float]) -> float:
        """Calculate regime strength for OTM straddle"""
        key_indicators = [
            'otm_volatility_sensitivity',
            'explosive_move_potential',
            'otm_efficiency_regime',
            'risk_reward_regime',
            'otm_market_quality'
        ]
        
        values = [abs(indicators.get(key, 0)) for key in key_indicators]
        return np.mean(values) if values else 0.5
    
    def get_straddle_status(self) -> Dict[str, Any]:
        """Get OTM1 straddle analyzer status"""
        return {
            'straddle_name': self.straddle_name,
            'call_weight': self.call_weight,
            'put_weight': self.put_weight,
            'call_status': self.call_analyzer.get_component_status(),
            'put_status': self.put_analyzer.get_component_status(),
            'max_time_value_ratio': self.max_time_value_ratio,
            'min_leverage_ratio': self.min_leverage_ratio,
            'decay_warning_threshold': self.decay_warning_threshold,
            'probability_threshold': self.probability_threshold
        }