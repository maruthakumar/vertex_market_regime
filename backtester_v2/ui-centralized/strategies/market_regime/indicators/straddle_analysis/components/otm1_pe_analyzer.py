"""
OTM1 Put (PE) Component Analyzer

Analyzes OTM1 (Out-of-The-Money 1 strike) Put options with rolling analysis 
across [3,5,10,15] minute windows. Provides comprehensive technical analysis, 
performance tracking, and regime contribution for the OTM1 Put component.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from .base_component_analyzer import BaseComponentAnalyzer, ComponentAnalysisResult
from ..core.calculation_engine import CalculationEngine
from ..rolling.window_manager import RollingWindowManager
from ..config.excel_reader import StraddleConfig
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class OTM1PutAnalyzer(BaseComponentAnalyzer):
    """
    OTM1 Put (PE) Component Analyzer
    
    Specialized analyzer for OTM1 Put options with:
    - Strike selection 1 strike below ATM
    - Lower negative delta sensitivity (typically -0.2 to -0.4)
    - No intrinsic value (pure time value)
    - OTM Put-specific regime indicators for downside protection
    """
    
    def __init__(self, 
                 config: StraddleConfig,
                 calculation_engine: CalculationEngine,
                 window_manager: RollingWindowManager):
        """
        Initialize OTM1 Put analyzer
        
        Args:
            config: Straddle configuration
            calculation_engine: Shared calculation engine
            window_manager: Rolling window manager
        """
        super().__init__('otm1_pe', config, calculation_engine, window_manager)
        
        # OTM1 Put-specific configuration
        self.strike_offset = 1  # Number of strikes out-of-the-money for puts
        self.delta_target = -0.3 # Target delta for OTM1 puts
        
        # Greeks thresholds for OTM1 puts
        self.delta_thresholds = {'low': -0.5, 'high': -0.1}
        self.gamma_thresholds = {'low': 0.005, 'high': 0.04}
        self.theta_thresholds = {'low': -60, 'high': -5}
        self.vega_thresholds = {'low': 10, 'high': 80}
        
        # OTM Put characteristics
        self.max_time_value_ratio = 1.0  # 100% time value for OTM
        self.protection_multiplier = 2.5  # Protection leverage
        
        self.logger.info("OTM1 Put analyzer initialized")
    
    def extract_component_price(self, data: Dict[str, Any]) -> Optional[float]:
        """
        Extract OTM1 Put price from market data
        
        Args:
            data: Market data dictionary containing option chain
            
        Returns:
            OTM1 Put price or None if not available
        """
        try:
            # Look for OTM1 Put price in different possible keys
            otm1_pe_keys = [
                'OTM1_PE', 'otm1_pe', 'OTM1_PUT', 'otm1_put',
                'otm1_pe_price', 'OTM1_PE_LTP', 'otm1_pe_ltp',
                'OTM_PE_1', 'otm_pe_1'
            ]
            
            for key in otm1_pe_keys:
                if key in data and data[key] is not None:
                    price = float(data[key])
                    if price > 0:  # Valid option price
                        return price
            
            # If direct OTM1 price not available, find from option chain
            underlying_price = data.get('underlying_price') or data.get('spot_price')
            if underlying_price and 'option_chain' in data:
                return self._find_otm1_put_from_chain(data['option_chain'], underlying_price)
            
            # Try alternative data structures
            if 'strikes' in data and 'pe_prices' in data:
                return self._find_otm1_from_strikes_data(data, underlying_price, 'PE')
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting OTM1 Put price: {e}")
            return None
    
    def _find_otm1_put_from_chain(self, option_chain: List[Dict], underlying_price: float) -> Optional[float]:
        """Find OTM1 Put from option chain data"""
        try:
            # Find PE options with strikes below underlying (OTM for puts)
            pe_options = [
                opt for opt in option_chain 
                if opt.get('option_type') == 'PE' and opt.get('strike', 0) < underlying_price
            ]
            
            if not pe_options:
                return None
            
            # Sort by strike descending (highest OTM strike first)
            pe_options.sort(key=lambda x: x.get('strike', 0), reverse=True)
            
            # Get the first OTM strike (1 strike OTM)
            if len(pe_options) >= self.strike_offset:
                otm1_option = pe_options[self.strike_offset - 1]
                return otm1_option.get('ltp') or otm1_option.get('price')
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding OTM1 Put from chain: {e}")
            return None
    
    def _find_otm1_from_strikes_data(self, data: Dict, underlying_price: float, option_type: str) -> Optional[float]:
        """Find OTM1 option from strikes and prices data"""
        try:
            strikes = data.get('strikes', [])
            prices_key = 'ce_prices' if option_type == 'CE' else 'pe_prices'
            prices = data.get(prices_key, [])
            
            if len(strikes) != len(prices):
                return None
            
            # Find OTM strikes for puts (strikes below underlying)
            otm_indices = [
                i for i, strike in enumerate(strikes) 
                if strike < underlying_price
            ]
            
            if not otm_indices:
                return None
            
            # Sort by strike descending
            otm_indices.sort(key=lambda i: strikes[i], reverse=True)
            
            # Get OTM1 strike
            if len(otm_indices) >= self.strike_offset:
                otm1_idx = otm_indices[self.strike_offset - 1]
                return prices[otm1_idx]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding OTM1 from strikes data: {e}")
            return None
    
    def calculate_component_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate OTM1 Put specific metrics
        
        Args:
            data: Market data dictionary
            
        Returns:
            Dictionary with OTM1 Put metrics
        """
        metrics = {}
        
        try:
            # Basic OTM1 Put metrics
            underlying_price = data.get('underlying_price') or data.get('spot_price', 0)
            otm1_pe_price = self.extract_component_price(data)
            
            if otm1_pe_price and underlying_price:
                # Get OTM1 strike
                strike = self._get_otm1_strike(data, underlying_price)
                
                # Intrinsic value (always 0 for OTM)
                metrics['intrinsic_value'] = 0.0
                
                # Time value (entire premium for OTM)
                metrics['time_value'] = otm1_pe_price
                metrics['time_value_ratio'] = 1.0  # 100% time value
                
                # Moneyness (inverted for puts)
                metrics['moneyness'] = strike / underlying_price if underlying_price > 0 else 1.0
                metrics['otm_percentage'] = (underlying_price - strike) / underlying_price * 100 if underlying_price > 0 else 0
                
                # OTM Put premium characteristics
                metrics['premium_to_spot_ratio'] = otm1_pe_price / underlying_price if underlying_price > 0 else 0
                metrics['distance_to_strike'] = underlying_price - strike
                metrics['distance_to_strike_pct'] = metrics['otm_percentage']
                
                # Protection calculations
                metrics['protection_cost'] = otm1_pe_price / underlying_price * 100  # As % of spot
                metrics['protection_level'] = strike  # Protection starts at strike
                metrics['max_loss_protection'] = (underlying_price - strike + otm1_pe_price) / underlying_price * 100
                
                # Hedge efficiency
                metrics['hedge_efficiency'] = self._calculate_hedge_efficiency(
                    otm1_pe_price, underlying_price, strike
                )
                
                # Put skew indicator
                otm1_ce_price = self._get_companion_call_price(data)
                if otm1_ce_price:
                    metrics['put_call_skew'] = (otm1_pe_price - otm1_ce_price) / otm1_ce_price * 100 if otm1_ce_price > 0 else 0
            
            # Greeks analysis (if available)
            greeks = self._extract_greeks(data)
            if greeks:
                metrics.update(greeks)
                
                # OTM Put-specific Greek analysis
                metrics['delta_regime'] = self._classify_delta_regime(greeks.get('delta', -0.3))
                metrics['gamma_regime'] = self._classify_gamma_regime(greeks.get('gamma', 0))
                metrics['theta_decay_rate'] = abs(greeks.get('theta', 0)) / otm1_pe_price if otm1_pe_price > 0 else 0
                
                # OTM Put sensitivity metrics
                metrics['delta_sensitivity'] = abs(greeks.get('delta', -0.3)) * self.protection_multiplier
                metrics['gamma_sensitivity'] = greeks.get('gamma', 0) * underlying_price / 100  # Per 1% move
                metrics['vega_impact'] = greeks.get('vega', 0) / otm1_pe_price if otm1_pe_price > 0 else 0
            
            # Volume and liquidity
            metrics['volume'] = data.get('volume', 0)
            metrics['open_interest'] = data.get('open_interest', 0)
            metrics['liquidity_score'] = self._calculate_liquidity_score(data)
            
            # OTM Put-specific indicators
            metrics['tail_risk_premium'] = self._calculate_tail_risk_premium(otm1_pe_price, metrics)
            metrics['insurance_value'] = self._calculate_insurance_value(data, otm1_pe_price, strike)
            metrics['fear_gauge'] = self._calculate_fear_gauge(metrics)
            
        except Exception as e:
            self.logger.error(f"Error calculating OTM1 Put metrics: {e}")
        
        return metrics
    
    def _get_otm1_strike(self, data: Dict[str, Any], underlying_price: float) -> float:
        """Get OTM1 strike price for puts"""
        try:
            # Direct OTM1 strike if available
            if 'otm1_strike' in data:
                return float(data['otm1_strike'])
            
            # Calculate from strikes array
            if 'strikes' in data:
                strikes = sorted([s for s in data['strikes'] if s < underlying_price], reverse=True)
                if len(strikes) >= self.strike_offset:
                    return strikes[self.strike_offset - 1]
            
            # Default: estimate based on typical strike intervals
            strike_interval = 50  # Common for NIFTY
            return underlying_price - (strike_interval * self.strike_offset)
            
        except Exception:
            return underlying_price - 50  # Fallback
    
    def _get_companion_call_price(self, data: Dict[str, Any]) -> Optional[float]:
        """Get corresponding OTM1 Call price for skew analysis"""
        otm1_ce_keys = [
            'OTM1_CE', 'otm1_ce', 'OTM1_CALL', 'otm1_call',
            'otm1_ce_price', 'OTM1_CE_LTP', 'otm1_ce_ltp'
        ]
        
        for key in otm1_ce_keys:
            if key in data and data[key] is not None:
                try:
                    return float(data[key])
                except (ValueError, TypeError):
                    continue
        return None
    
    def _extract_greeks(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract Greeks from market data"""
        greeks = {}
        
        # Common Greek keys for OTM puts
        greek_mappings = {
            'delta': ['delta', 'DELTA', 'Delta', 'otm1_pe_delta', 'OTM1_PE_DELTA'],
            'gamma': ['gamma', 'GAMMA', 'Gamma', 'otm1_pe_gamma', 'OTM1_PE_GAMMA'],
            'theta': ['theta', 'THETA', 'Theta', 'otm1_pe_theta', 'OTM1_PE_THETA'],
            'vega': ['vega', 'VEGA', 'Vega', 'otm1_pe_vega', 'OTM1_PE_VEGA'],
            'rho': ['rho', 'RHO', 'Rho', 'otm1_pe_rho', 'OTM1_PE_RHO']
        }
        
        for greek, keys in greek_mappings.items():
            for key in keys:
                if key in data and data[key] is not None:
                    try:
                        greeks[greek] = float(data[key])
                        break
                    except (ValueError, TypeError):
                        continue
        
        return greeks
    
    def _calculate_hedge_efficiency(self, option_price: float, spot_price: float, strike: float) -> float:
        """Calculate hedge efficiency for OTM put"""
        try:
            if option_price <= 0 or spot_price <= 0:
                return 0.0
            
            # Protection range = spot - strike
            protection_range = spot_price - strike
            
            # Cost as percentage of protection
            cost_per_protection = option_price / protection_range if protection_range > 0 else 0
            
            # Efficiency inversely related to cost
            efficiency = 1 / (1 + cost_per_protection * 10)
            
            return min(efficiency, 1.0)
            
        except Exception:
            return 0.5
    
    def _calculate_tail_risk_premium(self, option_price: float, metrics: Dict[str, Any]) -> float:
        """Calculate tail risk premium"""
        try:
            # Compare to theoretical value based on normal distribution
            distance_pct = metrics.get('distance_to_strike_pct', 0)
            
            # Simplified calculation - in practice use option model
            if distance_pct > 5:  # Far OTM
                theoretical_ratio = 0.001
            elif distance_pct > 3:
                theoretical_ratio = 0.005
            else:
                theoretical_ratio = 0.01
            
            actual_ratio = metrics.get('premium_to_spot_ratio', 0)
            
            # Tail risk premium = actual / theoretical
            if theoretical_ratio > 0:
                return min(actual_ratio / theoretical_ratio, 5.0)
            
            return 1.0
            
        except Exception:
            return 1.0
    
    def _calculate_insurance_value(self, data: Dict[str, Any], option_price: float, strike: float) -> float:
        """Calculate insurance value of OTM put"""
        try:
            underlying_price = data.get('underlying_price', 0)
            if underlying_price <= 0:
                return 0.0
            
            # Maximum payout = strike - 0 (worst case)
            max_payout = strike
            
            # Cost = option premium
            cost = option_price
            
            # Insurance value = max_payout / cost
            if cost > 0:
                return min(max_payout / cost, 100)
            
            return 0.0
            
        except Exception:
            return 10.0  # Default insurance multiplier
    
    def _calculate_fear_gauge(self, metrics: Dict[str, Any]) -> float:
        """Calculate market fear gauge based on OTM put metrics"""
        try:
            fear_factors = []
            
            # Put-call skew indicates fear
            if 'put_call_skew' in metrics:
                skew = metrics['put_call_skew']
                if skew > 20:
                    fear_factors.append(1.0)
                elif skew < -10:
                    fear_factors.append(0.0)
                else:
                    fear_factors.append((skew + 10) / 30)
            
            # High tail risk premium indicates fear
            if 'tail_risk_premium' in metrics:
                premium = metrics['tail_risk_premium']
                fear_factors.append(min(premium / 3, 1.0))
            
            # High protection cost indicates fear
            if 'protection_cost' in metrics:
                cost = metrics['protection_cost']
                if cost > 3:
                    fear_factors.append(1.0)
                elif cost < 1:
                    fear_factors.append(0.0)
                else:
                    fear_factors.append((cost - 1) / 2)
            
            return np.mean(fear_factors) if fear_factors else 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_liquidity_score(self, data: Dict[str, Any]) -> float:
        """Calculate liquidity score for OTM put option"""
        try:
            volume = data.get('volume', 0)
            open_interest = data.get('open_interest', 0)
            bid = data.get('bid', 0)
            ask = data.get('ask', 0)
            
            # OTM puts often have better liquidity than OTM calls (hedging demand)
            # Volume score
            volume_score = min(volume / 750, 1.0)
            
            # OI score
            oi_score = min(open_interest / 7500, 1.0)
            
            # Spread score
            if bid > 0 and ask > 0:
                spread_pct = (ask - bid) / ((ask + bid) / 2) * 100
                spread_score = max(0, 1 - spread_pct / 8)
            else:
                spread_score = 0.4
            
            # Weighted average (higher weight on OI for puts)
            return (volume_score * 0.3 + oi_score * 0.5 + spread_score * 0.2)
            
        except Exception:
            return 0.4
    
    def _classify_delta_regime(self, delta: float) -> str:
        """Classify delta regime for OTM1 Put"""
        if delta > self.delta_thresholds['high']:  # Less negative
            return 'DEEP_OTM'  # Very low probability
        elif delta < self.delta_thresholds['low']:  # More negative
            return 'NEAR_ATM'  # Moving toward ATM
        else:
            return 'NORMAL_OTM'  # Typical OTM1
    
    def _classify_gamma_regime(self, gamma: float) -> str:
        """Classify gamma regime for OTM1 Put"""
        abs_gamma = abs(gamma)
        if abs_gamma < self.gamma_thresholds['low']:
            return 'LOW_GAMMA'  # Deep OTM, low sensitivity
        elif abs_gamma > self.gamma_thresholds['high']:
            return 'HIGH_GAMMA'  # Significant gamma risk
        else:
            return 'NORMAL_GAMMA'
    
    def calculate_regime_contribution(self, analysis_result: ComponentAnalysisResult) -> Dict[str, float]:
        """
        Calculate OTM1 Put contribution to regime formation
        
        Args:
            analysis_result: Component analysis result
            
        Returns:
            Dictionary with regime indicators
        """
        regime_indicators = {}
        
        try:
            # Volatility contribution (OTM Put perspective)
            volatility_indicators = self._calculate_volatility_contribution(analysis_result)
            regime_indicators.update(volatility_indicators)
            
            # Trend contribution (OTM puts indicate downside risk)
            trend_indicators = self._calculate_trend_contribution(analysis_result)
            regime_indicators.update(trend_indicators)
            
            # Structure contribution
            structure_indicators = self._calculate_structure_contribution(analysis_result)
            regime_indicators.update(structure_indicators)
            
            # OTM Put-specific indicators
            otm_put_indicators = self._calculate_otm_put_specific_indicators(analysis_result)
            regime_indicators.update(otm_put_indicators)
            
            # Overall regime confidence
            regime_indicators['regime_confidence'] = self._calculate_regime_confidence(analysis_result)
            
        except Exception as e:
            self.logger.error(f"Error calculating regime contribution: {e}")
        
        return regime_indicators
    
    def _calculate_volatility_contribution(self, result: ComponentAnalysisResult) -> Dict[str, float]:
        """Calculate volatility regime contribution from OTM Put perspective"""
        indicators = {}
        
        try:
            # Fear gauge indicates volatility expectations
            if 'fear_gauge' in result.component_metrics:
                fear = result.component_metrics['fear_gauge']
                
                if fear > 0.7:
                    indicators['put_volatility_signal'] = 1.0  # High volatility expected
                elif fear < 0.3:
                    indicators['put_volatility_signal'] = -1.0  # Low volatility expected
                else:
                    indicators['put_volatility_signal'] = 0.0
                
                indicators['fear_intensity'] = fear
            
            # Vega impact for volatility sensitivity
            if 'vega_impact' in result.component_metrics:
                vega_impact = result.component_metrics['vega_impact']
                indicators['put_vega_sensitivity'] = min(vega_impact * 2, 1.0)
            
            # Tail risk premium as volatility indicator
            if 'tail_risk_premium' in result.component_metrics:
                tail_premium = result.component_metrics['tail_risk_premium']
                
                if tail_premium > 2:
                    indicators['tail_risk_volatility'] = 1.0
                elif tail_premium < 0.5:
                    indicators['tail_risk_volatility'] = -1.0
                else:
                    indicators['tail_risk_volatility'] = (tail_premium - 0.5) / 1.5
            
        except Exception as e:
            self.logger.warning(f"Error calculating volatility contribution: {e}")
        
        return indicators
    
    def _calculate_trend_contribution(self, result: ComponentAnalysisResult) -> Dict[str, float]:
        """Calculate trend regime contribution from OTM Put perspective"""
        indicators = {}
        
        try:
            # OTM puts indicate downside concerns
            if 'protection_cost' in result.component_metrics:
                protection_cost = result.component_metrics['protection_cost']
                
                # High protection cost suggests bearish sentiment
                if protection_cost > 2.5:
                    indicators['put_trend_signal'] = -1.0  # Bearish
                elif protection_cost < 1:
                    indicators['put_trend_signal'] = 1.0   # Bullish (low hedging)
                else:
                    indicators['put_trend_signal'] = 1 - (protection_cost / 2.5)
                
                indicators['hedging_demand'] = min(protection_cost / 3, 1.0)
            
            # Put-call skew for trend
            if 'put_call_skew' in result.component_metrics:
                skew = result.component_metrics['put_call_skew']
                
                # Positive skew (puts expensive) suggests bearish bias
                if skew > 15:
                    indicators['skew_trend_signal'] = -1.0
                elif skew < -15:
                    indicators['skew_trend_signal'] = 1.0
                else:
                    indicators['skew_trend_signal'] = -skew / 15
            
            # Price momentum (inverse for puts)
            price_change_pct = result.price_change_percent
            if abs(price_change_pct) > 0.1:
                # OTM puts gain on fear/downside
                if price_change_pct > 2:  # Put price increasing
                    indicators['put_momentum_signal'] = -1.0  # Bearish market
                elif price_change_pct < -2:  # Put price decreasing
                    indicators['put_momentum_signal'] = 1.0   # Bullish market
                else:
                    indicators['put_momentum_signal'] = -price_change_pct / 2
            
        except Exception as e:
            self.logger.warning(f"Error calculating trend contribution: {e}")
        
        return indicators
    
    def _calculate_structure_contribution(self, result: ComponentAnalysisResult) -> Dict[str, float]:
        """Calculate structure regime contribution"""
        indicators = {}
        
        try:
            # Liquidity structure for OTM puts
            if 'liquidity_score' in result.component_metrics:
                liquidity = result.component_metrics['liquidity_score']
                
                # Good put liquidity suggests mature market
                if liquidity > 0.6:
                    indicators['put_market_structure'] = 1.0  # Well-structured
                elif liquidity < 0.3:
                    indicators['put_market_structure'] = -1.0  # Poor structure
                else:
                    indicators['put_market_structure'] = 0.0
                
                indicators['hedging_accessibility'] = liquidity
            
            # Insurance value indicates market structure
            if 'insurance_value' in result.component_metrics:
                insurance_val = result.component_metrics['insurance_value']
                
                # High insurance value suggests efficient protection market
                if insurance_val > 20:
                    indicators['protection_efficiency'] = 1.0
                elif insurance_val < 5:
                    indicators['protection_efficiency'] = -1.0
                else:
                    indicators['protection_efficiency'] = (insurance_val - 5) / 15
            
        except Exception as e:
            self.logger.warning(f"Error calculating structure contribution: {e}")
        
        return indicators
    
    def _calculate_otm_put_specific_indicators(self, result: ComponentAnalysisResult) -> Dict[str, float]:
        """Calculate OTM Put-specific regime indicators"""
        indicators = {}
        
        try:
            # Hedge efficiency as regime indicator
            if 'hedge_efficiency' in result.component_metrics:
                efficiency = result.component_metrics['hedge_efficiency']
                
                if efficiency > 0.7:
                    indicators['hedge_regime'] = 1.0  # Efficient hedging
                elif efficiency < 0.3:
                    indicators['hedge_regime'] = -1.0  # Expensive hedging
                else:
                    indicators['hedge_regime'] = 0.0
                
                indicators['protection_value'] = efficiency
            
            # Maximum loss protection
            if 'max_loss_protection' in result.component_metrics:
                max_protection = result.component_metrics['max_loss_protection']
                
                # Higher protection percentage suggests defensive market
                if max_protection > 10:
                    indicators['defensive_regime'] = 1.0
                elif max_protection < 3:
                    indicators['defensive_regime'] = -1.0
                else:
                    indicators['defensive_regime'] = (max_protection - 3) / 7
                
                indicators['downside_coverage'] = min(max_protection / 15, 1.0)
            
            # Distance to strike for urgency
            if 'distance_to_strike_pct' in result.component_metrics:
                distance_pct = result.component_metrics['distance_to_strike_pct']
                
                # Closer distance suggests increased concern
                if distance_pct < 2:
                    indicators['strike_urgency'] = 1.0  # High urgency
                elif distance_pct > 5:
                    indicators['strike_urgency'] = -1.0  # Low urgency
                else:
                    indicators['strike_urgency'] = (5 - distance_pct) / 3
                
                indicators['protection_proximity'] = max(0, 1 - distance_pct / 10)
            
        except Exception as e:
            self.logger.warning(f"Error calculating OTM Put-specific indicators: {e}")
        
        return indicators
    
    def _calculate_regime_confidence(self, result: ComponentAnalysisResult) -> float:
        """Calculate overall regime confidence for OTM Put component"""
        try:
            confidence_factors = []
            
            # Data availability confidence
            available_windows = sum(1 for window in self.rolling_windows 
                                  if window in result.rolling_metrics)
            confidence_factors.append(available_windows / len(self.rolling_windows))
            
            # Volume confidence (adjusted for puts)
            if result.volume_ratio > 0.4:
                confidence_factors.append(min(result.volume_ratio * 1.2, 1.0))
            
            # Delta confidence (OTM Put should have negative low delta)
            if 'delta' in result.component_metrics:
                delta = result.component_metrics['delta']
                if -0.5 < delta < -0.1:  # Proper OTM put delta range
                    confidence_factors.append(0.8)
                else:
                    confidence_factors.append(0.3)  # Low confidence if delta out of range
            
            # Time value confidence (OTM should be 100% time value)
            if 'time_value_ratio' in result.component_metrics:
                time_ratio = result.component_metrics['time_value_ratio']
                confidence_factors.append(time_ratio)  # Should be 1.0 for OTM
            
            # Liquidity confidence (puts usually have better liquidity)
            if 'liquidity_score' in result.component_metrics:
                liquidity = result.component_metrics['liquidity_score']
                confidence_factors.append(min(liquidity * 1.2, 1.0))
            
            # Fear gauge confidence
            if 'fear_gauge' in result.component_metrics:
                fear = result.component_metrics['fear_gauge']
                # Moderate fear levels are more reliable
                if 0.3 < fear < 0.7:
                    confidence_factors.append(0.9)
                else:
                    confidence_factors.append(0.6)
            
            return np.mean(confidence_factors) if confidence_factors else 0.5
            
        except Exception:
            return 0.5