"""
ATM Straddle Combination Analyzer

Combines ATM Call (CE) and ATM Put (PE) components to analyze the complete
ATM straddle strategy. Provides unified analysis of both legs with rolling
windows [3,5,10,15] minutes and comprehensive regime contribution.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
from .atm_ce_analyzer import ATMCallAnalyzer
from .atm_pe_analyzer import ATMPutAnalyzer
from ..core.calculation_engine import CalculationEngine
from ..rolling.window_manager import RollingWindowManager
from ..config.excel_reader import StraddleConfig
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class StraddleAnalysisResult:
    """Result of straddle combination analysis"""
    straddle_name: str
    timestamp: pd.Timestamp
    
    # Individual component results
    call_result: Any  # ComponentAnalysisResult
    put_result: Any   # ComponentAnalysisResult
    
    # Combined metrics
    combined_price: float
    combined_greeks: Dict[str, float]
    straddle_metrics: Dict[str, float]
    
    # Regime indicators
    regime_indicators: Dict[str, float]
    
    # Performance metrics
    performance_metrics: Dict[str, float]
    
    # Strategy signals
    strategy_signals: Dict[str, float]


class ATMStraddleAnalyzer:
    """
    ATM Straddle Analyzer
    
    Combines ATM Call and Put analysis for complete straddle strategy.
    Provides:
    - Combined Greeks analysis
    - Net position metrics
    - Volatility exposure
    - Directional neutrality assessment
    - Straddle-specific regime indicators
    """
    
    def __init__(self,
                 config: StraddleConfig,
                 calculation_engine: CalculationEngine,
                 window_manager: RollingWindowManager):
        """
        Initialize ATM Straddle analyzer
        
        Args:
            config: Straddle configuration
            calculation_engine: Shared calculation engine
            window_manager: Rolling window manager
        """
        self.config = config
        self.calculation_engine = calculation_engine
        self.window_manager = window_manager
        
        # Initialize component analyzers
        self.call_analyzer = ATMCallAnalyzer(config, calculation_engine, window_manager)
        self.put_analyzer = ATMPutAnalyzer(config, calculation_engine, window_manager)
        
        # Straddle configuration
        self.straddle_name = 'atm_straddle'
        self.call_weight = config.component_weights.get('atm_ce', 0.5)
        self.put_weight = config.component_weights.get('atm_pe', 0.5)
        
        # Ensure weights sum to 1 for the straddle
        weight_sum = self.call_weight + self.put_weight
        if weight_sum > 0:
            self.call_weight /= weight_sum
            self.put_weight /= weight_sum
        
        # Straddle-specific thresholds
        self.delta_neutral_threshold = 0.1  # Max absolute delta for neutrality
        self.gamma_threshold = 0.1         # High gamma threshold
        self.vega_threshold = 100          # High vega threshold
        
        self.logger = logging.getLogger(f"{__name__}.{self.straddle_name}")
        self.logger.info(f"ATM Straddle analyzer initialized (CE: {self.call_weight:.2f}, PE: {self.put_weight:.2f})")
    
    def analyze(self, data: Dict[str, Any], timestamp: pd.Timestamp) -> Optional[StraddleAnalysisResult]:
        """
        Perform comprehensive ATM straddle analysis
        
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
                self.logger.warning("Insufficient data for straddle analysis")
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
                call_result, put_result, combined_price, combined_greeks
            )
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                call_result, put_result, combined_price
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
            self.logger.error(f"Error in straddle analysis: {e}")
            return None
    
    def _calculate_combined_greeks(self, call_result: Any, put_result: Any) -> Dict[str, float]:
        """Calculate combined Greeks for the straddle"""
        combined_greeks = {}
        
        # Extract Greeks from both legs
        call_greeks = call_result.component_metrics
        put_greeks = put_result.component_metrics
        
        # Delta: Net delta (should be near zero for ATM straddle)
        call_delta = call_greeks.get('delta', 0.5)
        put_delta = put_greeks.get('delta', -0.5)
        combined_greeks['net_delta'] = (
            call_delta * self.call_weight + 
            put_delta * self.put_weight
        )
        combined_greeks['gross_delta'] = abs(call_delta) + abs(put_delta)
        
        # Gamma: Sum of gammas (long gamma position)
        call_gamma = call_greeks.get('gamma', 0)
        put_gamma = put_greeks.get('gamma', 0)
        combined_greeks['net_gamma'] = (
            call_gamma * self.call_weight + 
            put_gamma * self.put_weight
        )
        
        # Theta: Sum of thetas (negative for long straddle)
        call_theta = call_greeks.get('theta', 0)
        put_theta = put_greeks.get('theta', 0)
        combined_greeks['net_theta'] = (
            call_theta * self.call_weight + 
            put_theta * self.put_weight
        )
        
        # Vega: Sum of vegas (long vega position)
        call_vega = call_greeks.get('vega', 0)
        put_vega = put_greeks.get('vega', 0)
        combined_greeks['net_vega'] = (
            call_vega * self.call_weight + 
            put_vega * self.put_weight
        )
        
        # Rho: Net rho
        call_rho = call_greeks.get('rho', 0)
        put_rho = put_greeks.get('rho', 0)
        combined_greeks['net_rho'] = (
            call_rho * self.call_weight + 
            put_rho * self.put_weight
        )
        
        return combined_greeks
    
    def _calculate_straddle_metrics(self, call_result: Any, put_result: Any, 
                                   combined_price: float, combined_greeks: Dict[str, float]) -> Dict[str, float]:
        """Calculate straddle-specific metrics"""
        metrics = {}
        
        try:
            # Price metrics
            metrics['straddle_premium'] = combined_price
            metrics['call_premium'] = call_result.current_price
            metrics['put_premium'] = put_result.current_price
            metrics['premium_ratio'] = (
                call_result.current_price / put_result.current_price 
                if put_result.current_price > 0 else 1.0
            )
            
            # Breakeven analysis
            underlying_price = call_result.component_metrics.get('moneyness', 1.0) * \
                             call_result.component_metrics.get('strike', 0)
            if underlying_price > 0:
                metrics['upper_breakeven'] = underlying_price + combined_price
                metrics['lower_breakeven'] = underlying_price - combined_price
                metrics['breakeven_width'] = 2 * combined_price
                metrics['breakeven_width_pct'] = (metrics['breakeven_width'] / underlying_price) * 100
            
            # Delta neutrality
            metrics['delta_neutrality'] = 1.0 - min(abs(combined_greeks['net_delta']) / self.delta_neutral_threshold, 1.0)
            metrics['is_delta_neutral'] = abs(combined_greeks['net_delta']) < self.delta_neutral_threshold
            
            # Gamma exposure
            metrics['gamma_exposure'] = combined_greeks['net_gamma']
            metrics['gamma_scalp_potential'] = combined_greeks['net_gamma'] * underlying_price * 0.01  # Per 1% move
            
            # Theta decay
            metrics['daily_decay'] = combined_greeks['net_theta']
            metrics['decay_as_pct'] = (
                abs(combined_greeks['net_theta']) / combined_price * 100 
                if combined_price > 0 else 0
            )
            
            # Vega exposure
            metrics['vega_exposure'] = combined_greeks['net_vega']
            metrics['volatility_sensitivity'] = combined_greeks['net_vega'] / combined_price if combined_price > 0 else 0
            
            # Put-call parity check
            if 'put_call_parity' in put_result.component_metrics:
                parity_data = put_result.component_metrics['put_call_parity']
                if isinstance(parity_data, dict):
                    metrics['parity_deviation'] = parity_data.get('parity_deviation', 0)
                    metrics['parity_efficiency'] = 1.0 - min(abs(metrics['parity_deviation']) / 10, 1.0)
            
            # Implied volatility metrics (if available)
            call_iv = call_result.component_metrics.get('implied_volatility', 0)
            put_iv = put_result.component_metrics.get('implied_volatility', 0)
            if call_iv > 0 and put_iv > 0:
                metrics['avg_implied_volatility'] = (call_iv + put_iv) / 2
                metrics['iv_skew'] = put_iv - call_iv
                metrics['iv_spread'] = abs(call_iv - put_iv)
            
            # Straddle efficiency
            metrics['straddle_efficiency'] = self._calculate_straddle_efficiency(
                metrics, combined_greeks
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating straddle metrics: {e}")
        
        return metrics
    
    def _calculate_straddle_efficiency(self, metrics: Dict[str, float], 
                                      combined_greeks: Dict[str, float]) -> float:
        """Calculate overall straddle efficiency"""
        try:
            efficiency_factors = []
            
            # Delta neutrality contributes to efficiency
            if 'delta_neutrality' in metrics:
                efficiency_factors.append(metrics['delta_neutrality'])
            
            # Parity efficiency
            if 'parity_efficiency' in metrics:
                efficiency_factors.append(metrics['parity_efficiency'])
            
            # Gamma/theta ratio (want high gamma relative to theta decay)
            if combined_greeks['net_gamma'] > 0 and combined_greeks['net_theta'] < 0:
                gamma_theta_ratio = combined_greeks['net_gamma'] / abs(combined_greeks['net_theta'])
                efficiency_factors.append(min(gamma_theta_ratio / 0.01, 1.0))  # Normalize
            
            # Low IV spread is efficient
            if 'iv_spread' in metrics:
                iv_efficiency = 1.0 - min(metrics['iv_spread'] / 5, 1.0)
                efficiency_factors.append(iv_efficiency)
            
            return np.mean(efficiency_factors) if efficiency_factors else 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_performance_metrics(self, call_result: Any, put_result: Any, 
                                     combined_price: float) -> Dict[str, float]:
        """Calculate straddle performance metrics"""
        metrics = {}
        
        try:
            # Combined price changes
            call_change_pct = call_result.price_change_percent
            put_change_pct = put_result.price_change_percent
            
            metrics['straddle_return_pct'] = (
                call_change_pct * self.call_weight + 
                put_change_pct * self.put_weight
            )
            
            # Leg performance
            metrics['call_contribution'] = call_change_pct * self.call_weight
            metrics['put_contribution'] = put_change_pct * self.put_weight
            metrics['winning_leg'] = 'CALL' if call_change_pct > put_change_pct else 'PUT'
            
            # Volatility capture
            # Straddle profits from large moves in either direction
            underlying_move = abs(call_change_pct - put_change_pct) / 2  # Approximate
            metrics['volatility_capture'] = max(call_change_pct, put_change_pct)
            metrics['directional_exposure'] = call_change_pct + put_change_pct  # Should be near 0
            
            # Risk metrics from rolling windows
            for window in ['3min', '5min', '10min', '15min']:
                if window in call_result.rolling_metrics:
                    call_vol = call_result.rolling_metrics[window].get(f'volatility_{window}', 0)
                    put_vol = put_result.rolling_metrics[window].get(f'volatility_{window}', 0)
                    metrics[f'straddle_volatility_{window}'] = np.sqrt(
                        (call_vol * self.call_weight) ** 2 + 
                        (put_vol * self.put_weight) ** 2
                    )
            
            # Maximum favorable/adverse excursion
            metrics['max_profit_potential'] = float('inf')  # Unlimited for straddle
            metrics['max_loss'] = combined_price  # Maximum loss is premium paid
            metrics['risk_reward_ratio'] = 0  # Undefined for unlimited upside
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
        
        return metrics
    
    def _calculate_strategy_signals(self, call_result: Any, put_result: Any,
                                  straddle_metrics: Dict[str, float],
                                  combined_greeks: Dict[str, float]) -> Dict[str, float]:
        """Calculate trading signals for straddle strategy"""
        signals = {}
        
        try:
            # Volatility signals
            if 'avg_implied_volatility' in straddle_metrics:
                iv = straddle_metrics['avg_implied_volatility']
                # Historical volatility comparison would go here
                if iv > 0.3:  # High IV
                    signals['volatility_signal'] = -1.0  # Sell volatility
                elif iv < 0.15:  # Low IV
                    signals['volatility_signal'] = 1.0   # Buy volatility
                else:
                    signals['volatility_signal'] = 0.0
            
            # Gamma scalping signal
            gamma = combined_greeks['net_gamma']
            if gamma > self.gamma_threshold:
                signals['gamma_scalp_signal'] = 1.0  # Good for gamma scalping
            else:
                signals['gamma_scalp_signal'] = 0.0
            
            # Decay warning
            decay_pct = straddle_metrics.get('decay_as_pct', 0)
            if decay_pct > 5:  # >5% daily decay
                signals['decay_warning'] = 1.0  # High decay warning
            elif decay_pct > 2:
                signals['decay_warning'] = 0.5  # Moderate decay
            else:
                signals['decay_warning'] = 0.0
            
            # Breakeven probability (simplified)
            breakeven_width_pct = straddle_metrics.get('breakeven_width_pct', 0)
            if breakeven_width_pct > 0:
                # Rough probability estimate based on breakeven width
                if breakeven_width_pct < 5:
                    signals['breakeven_probability'] = 0.8  # High probability
                elif breakeven_width_pct < 10:
                    signals['breakeven_probability'] = 0.5  # Medium probability
                else:
                    signals['breakeven_probability'] = 0.2  # Low probability
            
            # Entry/exit signals
            signals['entry_signal'] = self._calculate_entry_signal(
                straddle_metrics, combined_greeks, signals
            )
            signals['exit_signal'] = self._calculate_exit_signal(
                straddle_metrics, combined_greeks, signals
            )
            
            # Position sizing suggestion
            signals['position_size_factor'] = self._calculate_position_size_factor(
                straddle_metrics, combined_greeks
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating strategy signals: {e}")
        
        return signals
    
    def _calculate_entry_signal(self, metrics: Dict[str, float], 
                               greeks: Dict[str, float], 
                               signals: Dict[str, float]) -> float:
        """Calculate straddle entry signal"""
        try:
            entry_factors = []
            
            # Low IV is good for entry
            if signals.get('volatility_signal', 0) > 0:
                entry_factors.append(1.0)
            elif signals.get('volatility_signal', 0) < 0:
                entry_factors.append(0.0)
            
            # Delta neutral is preferred
            if metrics.get('is_delta_neutral', False):
                entry_factors.append(1.0)
            else:
                entry_factors.append(metrics.get('delta_neutrality', 0.5))
            
            # High efficiency is good
            efficiency = metrics.get('straddle_efficiency', 0.5)
            entry_factors.append(efficiency)
            
            # Low decay is preferred
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
        """Calculate straddle exit signal"""
        try:
            exit_factors = []
            
            # High decay suggests exit
            if signals.get('decay_warning', 0) > 0.5:
                exit_factors.append(1.0)
            
            # Loss of delta neutrality
            if not metrics.get('is_delta_neutral', True):
                exit_factors.append(0.7)
            
            # Profit target reached (would need P&L calculation)
            # This is simplified
            straddle_return = metrics.get('straddle_return_pct', 0)
            if straddle_return > 20:  # 20% profit
                exit_factors.append(1.0)
            elif straddle_return < -50:  # 50% loss
                exit_factors.append(1.0)
            
            return np.mean(exit_factors) if exit_factors else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_position_size_factor(self, metrics: Dict[str, float], 
                                       greeks: Dict[str, float]) -> float:
        """Calculate position sizing factor based on risk"""
        try:
            size_factors = []
            
            # Efficiency affects size
            efficiency = metrics.get('straddle_efficiency', 0.5)
            size_factors.append(efficiency)
            
            # Lower size for high decay
            decay_pct = metrics.get('decay_as_pct', 0)
            if decay_pct < 2:
                size_factors.append(1.0)
            elif decay_pct < 5:
                size_factors.append(0.7)
            else:
                size_factors.append(0.4)
            
            # Vega exposure (cap exposure for high vega)
            vega = abs(greeks.get('net_vega', 0))
            if vega < 50:
                size_factors.append(1.0)
            elif vega < 100:
                size_factors.append(0.7)
            else:
                size_factors.append(0.5)
            
            return np.mean(size_factors) if size_factors else 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_regime_contribution(self, result: StraddleAnalysisResult) -> Dict[str, float]:
        """Calculate ATM straddle contribution to regime formation"""
        regime_indicators = {}
        
        try:
            # Aggregate component regime indicators
            call_regime = result.call_result.regime_indicators
            put_regime = result.put_result.regime_indicators
            
            # Volatility regime (straddle is pure volatility play)
            volatility_indicators = self._aggregate_volatility_regime(
                call_regime, put_regime, result.straddle_metrics
            )
            regime_indicators.update(volatility_indicators)
            
            # Trend regime (straddle is trend-neutral)
            trend_indicators = self._aggregate_trend_regime(
                call_regime, put_regime, result.straddle_metrics
            )
            regime_indicators.update(trend_indicators)
            
            # Structure regime
            structure_indicators = self._aggregate_structure_regime(
                call_regime, put_regime, result.straddle_metrics
            )
            regime_indicators.update(structure_indicators)
            
            # Straddle-specific regime
            straddle_indicators = self._calculate_straddle_regime(
                result.straddle_metrics, result.combined_greeks, result.strategy_signals
            )
            regime_indicators.update(straddle_indicators)
            
            # Overall regime assessment
            regime_indicators['regime_type'] = self._determine_regime_type(regime_indicators)
            regime_indicators['regime_strength'] = self._calculate_regime_strength(regime_indicators)
            
        except Exception as e:
            self.logger.error(f"Error calculating regime contribution: {e}")
        
        return regime_indicators
    
    def _aggregate_volatility_regime(self, call_regime: Dict[str, float], 
                                    put_regime: Dict[str, float],
                                    straddle_metrics: Dict[str, float]) -> Dict[str, float]:
        """Aggregate volatility regime from components"""
        indicators = {}
        
        # Average component volatility signals
        call_vol = call_regime.get('volatility_regime', 0)
        put_vol = put_regime.get('volatility_regime', 0)
        indicators['component_volatility_consensus'] = (call_vol + put_vol) / 2
        
        # Vega exposure indicates volatility sensitivity
        vega = straddle_metrics.get('vega_exposure', 0)
        if vega > 100:
            indicators['vega_regime'] = 1.0  # High volatility sensitivity
        elif vega < 50:
            indicators['vega_regime'] = -1.0  # Low volatility sensitivity
        else:
            indicators['vega_regime'] = 0.0
        
        # IV level regime
        avg_iv = straddle_metrics.get('avg_implied_volatility', 0)
        if avg_iv > 0.25:
            indicators['iv_regime'] = 1.0  # High volatility regime
        elif avg_iv < 0.15:
            indicators['iv_regime'] = -1.0  # Low volatility regime
        else:
            indicators['iv_regime'] = 0.0
        
        # Breakeven width indicates expected volatility
        breakeven_width_pct = straddle_metrics.get('breakeven_width_pct', 0)
        if breakeven_width_pct > 8:
            indicators['expected_volatility'] = 1.0  # High expected volatility
        elif breakeven_width_pct < 4:
            indicators['expected_volatility'] = -1.0  # Low expected volatility
        else:
            indicators['expected_volatility'] = (breakeven_width_pct - 4) / 4
        
        return indicators
    
    def _aggregate_trend_regime(self, call_regime: Dict[str, float], 
                               put_regime: Dict[str, float],
                               straddle_metrics: Dict[str, float]) -> Dict[str, float]:
        """Aggregate trend regime from components"""
        indicators = {}
        
        # Straddle is directionally neutral
        # Look for conflicting signals between calls and puts
        call_trend = call_regime.get('trend_regime', 0)
        put_trend = put_regime.get('put_trend_signal', 0)
        
        # Trend disagreement is good for straddles
        trend_disagreement = abs(call_trend - put_trend)
        indicators['trend_uncertainty'] = min(trend_disagreement, 1.0)
        
        # Delta neutrality indicates no directional bias
        indicators['directional_neutrality'] = straddle_metrics.get('delta_neutrality', 0)
        
        # Premium ratio indicates market bias
        premium_ratio = straddle_metrics.get('premium_ratio', 1.0)
        if premium_ratio > 1.2:  # Calls expensive
            indicators['market_bias'] = 0.5  # Bullish bias
        elif premium_ratio < 0.8:  # Puts expensive
            indicators['market_bias'] = -0.5  # Bearish bias
        else:
            indicators['market_bias'] = 0.0  # Neutral
        
        return indicators
    
    def _aggregate_structure_regime(self, call_regime: Dict[str, float], 
                                   put_regime: Dict[str, float],
                                   straddle_metrics: Dict[str, float]) -> Dict[str, float]:
        """Aggregate structure regime from components"""
        indicators = {}
        
        # Average liquidity from components
        call_structure = call_regime.get('structure_regime', 0)
        put_structure = put_regime.get('structure_regime', 0)
        indicators['market_structure_quality'] = (call_structure + put_structure) / 2
        
        # Parity efficiency indicates market efficiency
        indicators['market_efficiency'] = straddle_metrics.get('parity_efficiency', 0.5)
        
        # IV skew indicates market structure
        iv_skew = straddle_metrics.get('iv_skew', 0)
        if abs(iv_skew) > 0.05:  # 5% skew
            indicators['skew_structure'] = -1.0  # Stressed structure
        elif abs(iv_skew) < 0.02:  # 2% skew
            indicators['skew_structure'] = 1.0   # Balanced structure
        else:
            indicators['skew_structure'] = 0.0
        
        return indicators
    
    def _calculate_straddle_regime(self, straddle_metrics: Dict[str, float],
                                  combined_greeks: Dict[str, float],
                                  strategy_signals: Dict[str, float]) -> Dict[str, float]:
        """Calculate straddle-specific regime indicators"""
        indicators = {}
        
        # Gamma regime
        gamma = combined_greeks.get('net_gamma', 0)
        if gamma > 0.1:
            indicators['gamma_regime'] = 1.0  # High gamma environment
        elif gamma < 0.02:
            indicators['gamma_regime'] = -1.0  # Low gamma environment
        else:
            indicators['gamma_regime'] = 0.0
        
        # Decay regime
        decay_warning = strategy_signals.get('decay_warning', 0)
        indicators['decay_regime'] = decay_warning
        
        # Straddle efficiency regime
        efficiency = straddle_metrics.get('straddle_efficiency', 0.5)
        if efficiency > 0.7:
            indicators['efficiency_regime'] = 1.0  # Efficient market for straddles
        elif efficiency < 0.3:
            indicators['efficiency_regime'] = -1.0  # Inefficient market
        else:
            indicators['efficiency_regime'] = 0.0
        
        # Entry attractiveness
        indicators['entry_attractiveness'] = strategy_signals.get('entry_signal', 0.5)
        
        return indicators
    
    def _determine_regime_type(self, indicators: Dict[str, float]) -> str:
        """Determine overall regime type"""
        # Simplified regime classification
        volatility_score = np.mean([
            indicators.get('component_volatility_consensus', 0),
            indicators.get('iv_regime', 0),
            indicators.get('expected_volatility', 0)
        ])
        
        structure_score = np.mean([
            indicators.get('market_structure_quality', 0),
            indicators.get('market_efficiency', 0),
            indicators.get('skew_structure', 0)
        ])
        
        if volatility_score > 0.5:
            if structure_score > 0:
                return 'HIGH_VOL_TRENDING'
            else:
                return 'HIGH_VOL_CHOPPY'
        else:
            if structure_score > 0:
                return 'LOW_VOL_TRENDING'
            else:
                return 'LOW_VOL_RANGING'
    
    def _calculate_regime_strength(self, indicators: Dict[str, float]) -> float:
        """Calculate regime strength/confidence"""
        # Average absolute values of key indicators
        key_indicators = [
            'component_volatility_consensus',
            'iv_regime',
            'gamma_regime',
            'efficiency_regime',
            'market_structure_quality'
        ]
        
        values = [abs(indicators.get(key, 0)) for key in key_indicators]
        return np.mean(values) if values else 0.5
    
    def get_straddle_status(self) -> Dict[str, Any]:
        """Get straddle analyzer status"""
        return {
            'straddle_name': self.straddle_name,
            'call_weight': self.call_weight,
            'put_weight': self.put_weight,
            'call_status': self.call_analyzer.get_component_status(),
            'put_status': self.put_analyzer.get_component_status(),
            'delta_neutral_threshold': self.delta_neutral_threshold,
            'gamma_threshold': self.gamma_threshold,
            'vega_threshold': self.vega_threshold
        }