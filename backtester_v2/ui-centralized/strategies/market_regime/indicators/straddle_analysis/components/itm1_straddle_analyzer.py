"""
ITM1 Straddle Combination Analyzer

Combines ITM1 Call (CE) and ITM1 Put (PE) components to analyze the complete
ITM1 straddle strategy. Provides unified analysis with focus on directional
bias and intrinsic value characteristics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
from .itm1_ce_analyzer import ITM1CallAnalyzer
from .itm1_pe_analyzer import ITM1PutAnalyzer
from .atm_straddle_analyzer import StraddleAnalysisResult
from ..core.calculation_engine import CalculationEngine
from ..rolling.window_manager import RollingWindowManager
from ..config.excel_reader import StraddleConfig
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ITM1StraddleAnalyzer:
    """
    ITM1 Straddle Analyzer
    
    Combines ITM1 Call and Put analysis for ITM straddle strategy.
    Key characteristics:
    - Higher intrinsic value component
    - Directional bias (calls ITM when bullish, puts ITM when bearish)
    - Lower time decay relative to premium
    - Different breakeven characteristics than ATM
    """
    
    def __init__(self,
                 config: StraddleConfig,
                 calculation_engine: CalculationEngine,
                 window_manager: RollingWindowManager):
        """
        Initialize ITM1 Straddle analyzer
        
        Args:
            config: Straddle configuration
            calculation_engine: Shared calculation engine
            window_manager: Rolling window manager
        """
        self.config = config
        self.calculation_engine = calculation_engine
        self.window_manager = window_manager
        
        # Initialize component analyzers
        self.call_analyzer = ITM1CallAnalyzer(config, calculation_engine, window_manager)
        self.put_analyzer = ITM1PutAnalyzer(config, calculation_engine, window_manager)
        
        # Straddle configuration
        self.straddle_name = 'itm1_straddle'
        self.call_weight = config.component_weights.get('itm1_ce', 0.5)
        self.put_weight = config.component_weights.get('itm1_pe', 0.5)
        
        # Ensure weights sum to 1 for the straddle
        weight_sum = self.call_weight + self.put_weight
        if weight_sum > 0:
            self.call_weight /= weight_sum
            self.put_weight /= weight_sum
        
        # ITM-specific thresholds
        self.min_intrinsic_ratio = 0.3     # Minimum intrinsic value ratio
        self.delta_bias_threshold = 0.2    # Expected delta bias for ITM
        self.efficiency_threshold = 0.7    # ITM efficiency threshold
        
        self.logger = logging.getLogger(f"{__name__}.{self.straddle_name}")
        self.logger.info(f"ITM1 Straddle analyzer initialized (CE: {self.call_weight:.2f}, PE: {self.put_weight:.2f})")
    
    def analyze(self, data: Dict[str, Any], timestamp: pd.Timestamp) -> Optional[StraddleAnalysisResult]:
        """
        Perform comprehensive ITM1 straddle analysis
        
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
                self.logger.warning("Insufficient data for ITM1 straddle analysis")
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
            self.logger.error(f"Error in ITM1 straddle analysis: {e}")
            return None
    
    def _calculate_combined_greeks(self, call_result: Any, put_result: Any) -> Dict[str, float]:
        """Calculate combined Greeks for ITM1 straddle"""
        combined_greeks = {}
        
        # Extract Greeks from both legs
        call_greeks = call_result.component_metrics
        put_greeks = put_result.component_metrics
        
        # Delta: ITM straddle has directional bias
        call_delta = call_greeks.get('delta', 0.7)    # Higher for ITM call
        put_delta = put_greeks.get('delta', -0.7)     # More negative for ITM put
        combined_greeks['net_delta'] = (
            call_delta * self.call_weight + 
            put_delta * self.put_weight
        )
        combined_greeks['gross_delta'] = abs(call_delta) + abs(put_delta)
        combined_greeks['delta_spread'] = abs(call_delta + put_delta)  # Asymmetry measure
        
        # Gamma: Lower than ATM but still significant
        call_gamma = call_greeks.get('gamma', 0)
        put_gamma = put_greeks.get('gamma', 0)
        combined_greeks['net_gamma'] = (
            call_gamma * self.call_weight + 
            put_gamma * self.put_weight
        )
        
        # Theta: Lower decay rate due to intrinsic value
        call_theta = call_greeks.get('theta', 0)
        put_theta = put_greeks.get('theta', 0)
        combined_greeks['net_theta'] = (
            call_theta * self.call_weight + 
            put_theta * self.put_weight
        )
        
        # Vega: Lower than ATM
        call_vega = call_greeks.get('vega', 0)
        put_vega = put_greeks.get('vega', 0)
        combined_greeks['net_vega'] = (
            call_vega * self.call_weight + 
            put_vega * self.put_weight
        )
        
        # Rho: Higher sensitivity to interest rates
        call_rho = call_greeks.get('rho', 0)
        put_rho = put_greeks.get('rho', 0)
        combined_greeks['net_rho'] = (
            call_rho * self.call_weight + 
            put_rho * self.put_weight
        )
        
        return combined_greeks
    
    def _calculate_straddle_metrics(self, call_result: Any, put_result: Any,
                                   combined_price: float, combined_greeks: Dict[str, float],
                                   data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate ITM1 straddle-specific metrics"""
        metrics = {}
        
        try:
            # Price metrics
            metrics['straddle_premium'] = combined_price
            metrics['call_premium'] = call_result.current_price
            metrics['put_premium'] = put_result.current_price
            
            # Intrinsic value analysis
            call_intrinsic = call_result.component_metrics.get('intrinsic_value', 0)
            put_intrinsic = put_result.component_metrics.get('intrinsic_value', 0)
            
            metrics['total_intrinsic_value'] = call_intrinsic + put_intrinsic
            metrics['total_time_value'] = combined_price - metrics['total_intrinsic_value']
            metrics['intrinsic_ratio'] = (
                metrics['total_intrinsic_value'] / combined_price 
                if combined_price > 0 else 0
            )
            metrics['time_value_ratio'] = 1 - metrics['intrinsic_ratio']
            
            # ITM characteristics
            metrics['call_itm_ratio'] = call_intrinsic / call_result.current_price if call_result.current_price > 0 else 0
            metrics['put_itm_ratio'] = put_intrinsic / put_result.current_price if put_result.current_price > 0 else 0
            metrics['dominant_leg'] = 'CALL' if call_intrinsic > put_intrinsic else 'PUT'
            
            # Get underlying price and strikes
            underlying_price = data.get('underlying_price') or data.get('spot_price', 0)
            itm1_call_strike = call_result.component_metrics.get('moneyness', 1.0) * underlying_price
            itm1_put_strike = underlying_price / put_result.component_metrics.get('moneyness', 1.0)
            
            # Breakeven analysis (adjusted for ITM)
            if underlying_price > 0:
                # Upper breakeven = call strike + net premium
                metrics['upper_breakeven'] = itm1_call_strike + combined_price
                # Lower breakeven = put strike - net premium
                metrics['lower_breakeven'] = itm1_put_strike - combined_price
                metrics['breakeven_width'] = metrics['upper_breakeven'] - metrics['lower_breakeven']
                metrics['breakeven_asymmetry'] = (
                    (metrics['upper_breakeven'] - underlying_price) - 
                    (underlying_price - metrics['lower_breakeven'])
                )
            
            # Delta characteristics
            metrics['delta_bias'] = combined_greeks['net_delta']
            metrics['delta_neutrality'] = 1.0 - min(abs(metrics['delta_bias']) / 0.5, 1.0)
            metrics['is_delta_biased'] = abs(metrics['delta_bias']) > self.delta_bias_threshold
            
            # Efficiency metrics
            metrics['itm_efficiency'] = self._calculate_itm_efficiency(
                call_result.component_metrics, put_result.component_metrics
            )
            
            # Leverage analysis
            call_leverage = call_result.component_metrics.get('delta_leverage', 0)
            put_leverage = put_result.component_metrics.get('delta_leverage', 0)
            metrics['combined_leverage'] = (
                call_leverage * self.call_weight + 
                put_leverage * self.put_weight
            )
            
            # Protection analysis
            metrics['downside_protection'] = put_intrinsic / underlying_price * 100 if underlying_price > 0 else 0
            metrics['upside_participation'] = call_intrinsic / underlying_price * 100 if underlying_price > 0 else 0
            
        except Exception as e:
            self.logger.error(f"Error calculating ITM1 straddle metrics: {e}")
        
        return metrics
    
    def _calculate_itm_efficiency(self, call_metrics: Dict[str, float],
                                 put_metrics: Dict[str, float]) -> float:
        """Calculate ITM straddle efficiency"""
        try:
            efficiency_factors = []
            
            # ITM efficiency from components
            call_efficiency = call_metrics.get('itm_efficiency', 0.5)
            put_efficiency = put_metrics.get('itm_efficiency', 0.5)
            efficiency_factors.append((call_efficiency + put_efficiency) / 2)
            
            # Intrinsic value balance
            call_intrinsic_ratio = call_metrics.get('intrinsic_ratio', 0)
            put_intrinsic_ratio = put_metrics.get('intrinsic_ratio', 0)
            
            # Both should have significant intrinsic value
            if call_intrinsic_ratio > self.min_intrinsic_ratio and \
               put_intrinsic_ratio > self.min_intrinsic_ratio:
                efficiency_factors.append(1.0)
            else:
                efficiency_factors.append(0.5)
            
            # Liquidity efficiency
            call_liquidity = call_metrics.get('liquidity_score', 0.5)
            put_liquidity = put_metrics.get('liquidity_score', 0.5)
            efficiency_factors.append((call_liquidity + put_liquidity) / 2)
            
            return np.mean(efficiency_factors) if efficiency_factors else 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_performance_metrics(self, call_result: Any, put_result: Any,
                                     combined_price: float,
                                     straddle_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate ITM1 straddle performance metrics"""
        metrics = {}
        
        try:
            # Price performance
            call_change_pct = call_result.price_change_percent
            put_change_pct = put_result.price_change_percent
            
            metrics['straddle_return_pct'] = (
                call_change_pct * self.call_weight + 
                put_change_pct * self.put_weight
            )
            
            # Component performance
            metrics['call_performance'] = call_change_pct
            metrics['put_performance'] = put_change_pct
            metrics['performance_spread'] = abs(call_change_pct - put_change_pct)
            
            # Intrinsic vs time value performance
            intrinsic_ratio = straddle_metrics.get('intrinsic_ratio', 0)
            metrics['intrinsic_protection'] = intrinsic_ratio  # Portion protected by intrinsic
            metrics['time_value_risk'] = 1 - intrinsic_ratio   # Portion at risk from decay
            
            # Directional performance
            delta_bias = straddle_metrics.get('delta_bias', 0)
            if delta_bias > 0:  # Net long delta
                metrics['directional_pnl'] = call_change_pct * abs(delta_bias)
            else:  # Net short delta
                metrics['directional_pnl'] = put_change_pct * abs(delta_bias)
            
            # Risk metrics
            metrics['max_loss'] = straddle_metrics.get('total_time_value', combined_price)
            metrics['protected_value'] = straddle_metrics.get('total_intrinsic_value', 0)
            metrics['risk_ratio'] = metrics['max_loss'] / combined_price if combined_price > 0 else 1.0
            
            # Rolling window performance
            for window in ['3min', '5min', '10min', '15min']:
                if window in call_result.rolling_metrics:
                    call_sharpe = call_result.rolling_metrics[window].get(f'sharpe_ratio_{window}', 0)
                    put_sharpe = put_result.rolling_metrics[window].get(f'sharpe_ratio_{window}', 0)
                    metrics[f'straddle_sharpe_{window}'] = (
                        call_sharpe * self.call_weight + 
                        put_sharpe * self.put_weight
                    )
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
        
        return metrics
    
    def _calculate_strategy_signals(self, call_result: Any, put_result: Any,
                                  straddle_metrics: Dict[str, float],
                                  combined_greeks: Dict[str, float]) -> Dict[str, float]:
        """Calculate trading signals for ITM1 straddle strategy"""
        signals = {}
        
        try:
            # Efficiency signal
            efficiency = straddle_metrics.get('itm_efficiency', 0.5)
            if efficiency > self.efficiency_threshold:
                signals['efficiency_signal'] = 1.0
            elif efficiency < 0.3:
                signals['efficiency_signal'] = -1.0
            else:
                signals['efficiency_signal'] = 0.0
            
            # Intrinsic value signal
            intrinsic_ratio = straddle_metrics.get('intrinsic_ratio', 0)
            if intrinsic_ratio < self.min_intrinsic_ratio:
                signals['intrinsic_warning'] = 1.0  # Too little intrinsic value
            else:
                signals['intrinsic_warning'] = 0.0
            
            # Delta bias signal
            delta_bias = straddle_metrics.get('delta_bias', 0)
            if abs(delta_bias) > 0.3:
                signals['delta_hedge_signal'] = 1.0  # Need to hedge delta
                signals['hedge_direction'] = 'SHORT' if delta_bias > 0 else 'LONG'
            else:
                signals['delta_hedge_signal'] = 0.0
            
            # Decay signal (less critical for ITM)
            time_value_ratio = straddle_metrics.get('time_value_ratio', 0)
            theta = combined_greeks.get('net_theta', 0)
            decay_impact = abs(theta) / straddle_metrics.get('total_time_value', 1) if time_value_ratio > 0 else 0
            
            if decay_impact > 0.05:  # >5% daily decay of time value
                signals['decay_warning'] = 1.0
            elif decay_impact > 0.02:
                signals['decay_warning'] = 0.5
            else:
                signals['decay_warning'] = 0.0
            
            # Volatility opportunity
            vega = combined_greeks.get('net_vega', 0)
            signals['volatility_opportunity'] = min(vega / 100, 1.0)  # Normalized vega signal
            
            # Entry/exit signals
            signals['entry_signal'] = self._calculate_entry_signal(
                straddle_metrics, combined_greeks, signals
            )
            signals['exit_signal'] = self._calculate_exit_signal(
                straddle_metrics, combined_greeks, signals
            )
            
            # Position adjustment signals
            signals['roll_signal'] = self._calculate_roll_signal(
                straddle_metrics, signals
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating strategy signals: {e}")
        
        return signals
    
    def _calculate_entry_signal(self, metrics: Dict[str, float],
                               greeks: Dict[str, float],
                               signals: Dict[str, float]) -> float:
        """Calculate ITM straddle entry signal"""
        try:
            entry_factors = []
            
            # High efficiency is good
            if signals.get('efficiency_signal', 0) > 0:
                entry_factors.append(1.0)
            elif signals.get('efficiency_signal', 0) < 0:
                entry_factors.append(0.0)
            else:
                entry_factors.append(0.5)
            
            # Adequate intrinsic value
            if signals.get('intrinsic_warning', 0) == 0:
                entry_factors.append(1.0)
            else:
                entry_factors.append(0.3)
            
            # Manageable delta bias
            if signals.get('delta_hedge_signal', 0) < 0.5:
                entry_factors.append(1.0)
            else:
                entry_factors.append(0.5)
            
            # Low decay warning
            if signals.get('decay_warning', 0) < 0.5:
                entry_factors.append(1.0)
            else:
                entry_factors.append(0.3)
            
            return np.mean(entry_factors) if entry_factors else 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_exit_signal(self, metrics: Dict[str, float],
                              greeks: Dict[str, float],
                              signals: Dict[str, float]) -> float:
        """Calculate ITM straddle exit signal"""
        try:
            exit_factors = []
            
            # Loss of efficiency
            if signals.get('efficiency_signal', 0) < 0:
                exit_factors.append(0.8)
            
            # Excessive delta bias
            if signals.get('delta_hedge_signal', 0) > 0.5:
                exit_factors.append(0.7)
            
            # High decay warning
            if signals.get('decay_warning', 0) > 0.7:
                exit_factors.append(0.8)
            
            # Profit/loss thresholds
            return_pct = metrics.get('straddle_return_pct', 0)
            if return_pct > 30:  # 30% profit
                exit_factors.append(1.0)
            elif return_pct < -40:  # 40% loss
                exit_factors.append(1.0)
            
            return np.mean(exit_factors) if exit_factors else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_roll_signal(self, metrics: Dict[str, float],
                              signals: Dict[str, float]) -> float:
        """Calculate signal to roll ITM straddle"""
        try:
            roll_factors = []
            
            # Low intrinsic value suggests rolling
            if signals.get('intrinsic_warning', 0) > 0.5:
                roll_factors.append(0.8)
            
            # High time value decay
            if signals.get('decay_warning', 0) > 0.7:
                roll_factors.append(0.7)
            
            # Loss of ITM characteristics
            intrinsic_ratio = metrics.get('intrinsic_ratio', 0)
            if intrinsic_ratio < 0.2:
                roll_factors.append(0.9)
            
            return np.mean(roll_factors) if roll_factors else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_regime_contribution(self, result: StraddleAnalysisResult) -> Dict[str, float]:
        """Calculate ITM1 straddle contribution to regime formation"""
        regime_indicators = {}
        
        try:
            # Component regime indicators
            call_regime = result.call_result.regime_indicators
            put_regime = result.put_result.regime_indicators
            
            # Volatility regime (ITM perspective)
            volatility_indicators = self._aggregate_volatility_regime(
                call_regime, put_regime, result.straddle_metrics, result.combined_greeks
            )
            regime_indicators.update(volatility_indicators)
            
            # Trend regime (ITM has directional bias)
            trend_indicators = self._aggregate_trend_regime(
                call_regime, put_regime, result.straddle_metrics, result.combined_greeks
            )
            regime_indicators.update(trend_indicators)
            
            # Structure regime
            structure_indicators = self._aggregate_structure_regime(
                call_regime, put_regime, result.straddle_metrics
            )
            regime_indicators.update(structure_indicators)
            
            # ITM-specific regime
            itm_indicators = self._calculate_itm_regime(
                result.straddle_metrics, result.combined_greeks, result.strategy_signals
            )
            regime_indicators.update(itm_indicators)
            
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
        """Aggregate volatility regime for ITM straddle"""
        indicators = {}
        
        # ITM straddles have lower vega but still benefit from volatility
        vega = combined_greeks.get('net_vega', 0)
        if vega > 75:
            indicators['itm_vega_regime'] = 1.0
        elif vega < 25:
            indicators['itm_vega_regime'] = -1.0
        else:
            indicators['itm_vega_regime'] = (vega - 25) / 50
        
        # Time value ratio indicates volatility dependence
        time_value_ratio = straddle_metrics.get('time_value_ratio', 0)
        indicators['volatility_dependence'] = time_value_ratio
        
        # Component consensus
        call_vol = call_regime.get('theta_volatility_signal', 0)
        put_vol = put_regime.get('put_volatility_signal', 0)
        indicators['itm_volatility_consensus'] = (call_vol + put_vol) / 2
        
        return indicators
    
    def _aggregate_trend_regime(self, call_regime: Dict[str, float],
                               put_regime: Dict[str, float],
                               straddle_metrics: Dict[str, float],
                               combined_greeks: Dict[str, float]) -> Dict[str, float]:
        """Aggregate trend regime for ITM straddle"""
        indicators = {}
        
        # Delta bias indicates trend exposure
        delta_bias = straddle_metrics.get('delta_bias', 0)
        if delta_bias > 0.2:
            indicators['itm_trend_bias'] = 1.0  # Bullish bias
        elif delta_bias < -0.2:
            indicators['itm_trend_bias'] = -1.0  # Bearish bias
        else:
            indicators['itm_trend_bias'] = delta_bias * 5  # Scale up small biases
        
        # Dominant leg performance
        dominant_leg = straddle_metrics.get('dominant_leg', '')
        if dominant_leg == 'CALL':
            indicators['momentum_direction'] = call_regime.get('delta_trend_signal', 0)
        else:
            indicators['momentum_direction'] = put_regime.get('put_delta_trend_signal', 0)
        
        # Leverage factor
        leverage = straddle_metrics.get('combined_leverage', 0)
        indicators['leveraged_trend_exposure'] = abs(delta_bias) * min(leverage / 20, 1.0)
        
        return indicators
    
    def _aggregate_structure_regime(self, call_regime: Dict[str, float],
                                   put_regime: Dict[str, float],
                                   straddle_metrics: Dict[str, float]) -> Dict[str, float]:
        """Aggregate structure regime for ITM straddle"""
        indicators = {}
        
        # ITM liquidity structure
        call_liquidity = call_regime.get('liquidity_structure', 0)
        put_liquidity = put_regime.get('put_liquidity_structure', 0)
        indicators['itm_liquidity_quality'] = (call_liquidity + put_liquidity) / 2
        
        # Efficiency as structure indicator
        efficiency = straddle_metrics.get('itm_efficiency', 0.5)
        if efficiency > 0.7:
            indicators['itm_market_efficiency'] = 1.0
        elif efficiency < 0.3:
            indicators['itm_market_efficiency'] = -1.0
        else:
            indicators['itm_market_efficiency'] = (efficiency - 0.3) / 0.4
        
        return indicators
    
    def _calculate_itm_regime(self, straddle_metrics: Dict[str, float],
                             combined_greeks: Dict[str, float],
                             strategy_signals: Dict[str, float]) -> Dict[str, float]:
        """Calculate ITM-specific regime indicators"""
        indicators = {}
        
        # Intrinsic value regime
        intrinsic_ratio = straddle_metrics.get('intrinsic_ratio', 0)
        if intrinsic_ratio > 0.6:
            indicators['intrinsic_regime'] = 1.0  # Deep ITM
        elif intrinsic_ratio < 0.3:
            indicators['intrinsic_regime'] = -1.0  # Barely ITM
        else:
            indicators['intrinsic_regime'] = (intrinsic_ratio - 0.3) / 0.3
        
        # Protection regime
        downside_protection = straddle_metrics.get('downside_protection', 0)
        upside_participation = straddle_metrics.get('upside_participation', 0)
        indicators['protection_balance'] = (downside_protection + upside_participation) / 2
        
        # Efficiency regime
        indicators['itm_efficiency_regime'] = strategy_signals.get('efficiency_signal', 0)
        
        return indicators
    
    def _determine_regime_type(self, indicators: Dict[str, float]) -> str:
        """Determine ITM straddle regime type"""
        volatility_score = indicators.get('itm_volatility_consensus', 0)
        trend_score = indicators.get('itm_trend_bias', 0)
        intrinsic_score = indicators.get('intrinsic_regime', 0)
        
        if intrinsic_score > 0.5:
            if abs(trend_score) > 0.5:
                return 'DEEP_ITM_DIRECTIONAL'
            else:
                return 'DEEP_ITM_BALANCED'
        else:
            if volatility_score > 0.5:
                return 'SHALLOW_ITM_VOLATILE'
            else:
                return 'SHALLOW_ITM_STABLE'
    
    def _calculate_regime_strength(self, indicators: Dict[str, float]) -> float:
        """Calculate regime strength for ITM straddle"""
        key_indicators = [
            'intrinsic_regime',
            'itm_efficiency_regime',
            'itm_liquidity_quality',
            'protection_balance'
        ]
        
        values = [abs(indicators.get(key, 0)) for key in key_indicators]
        return np.mean(values) if values else 0.5
    
    def get_straddle_status(self) -> Dict[str, Any]:
        """Get ITM1 straddle analyzer status"""
        return {
            'straddle_name': self.straddle_name,
            'call_weight': self.call_weight,
            'put_weight': self.put_weight,
            'call_status': self.call_analyzer.get_component_status(),
            'put_status': self.put_analyzer.get_component_status(),
            'min_intrinsic_ratio': self.min_intrinsic_ratio,
            'delta_bias_threshold': self.delta_bias_threshold,
            'efficiency_threshold': self.efficiency_threshold
        }