"""
OTM1 Call (CE) Component Analyzer

Analyzes OTM1 (Out-of-The-Money 1 strike) Call options with rolling analysis 
across [3,5,10,15] minute windows. Provides comprehensive technical analysis, 
performance tracking, and regime contribution for the OTM1 Call component.
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


class OTM1CallAnalyzer(BaseComponentAnalyzer):
    """
    OTM1 Call (CE) Component Analyzer
    
    Specialized analyzer for OTM1 Call options with:
    - Strike selection 1 strike above ATM
    - Lower delta sensitivity (typically 0.2-0.4)
    - No intrinsic value (pure time value)
    - OTM-specific regime indicators
    """
    
    def __init__(self, 
                 config: StraddleConfig,
                 calculation_engine: CalculationEngine,
                 window_manager: RollingWindowManager):
        """
        Initialize OTM1 Call analyzer
        
        Args:
            config: Straddle configuration
            calculation_engine: Shared calculation engine
            window_manager: Rolling window manager
        """
        super().__init__('otm1_ce', config, calculation_engine, window_manager)
        
        # OTM1-specific configuration
        self.strike_offset = 1  # Number of strikes out-of-the-money
        self.delta_target = 0.3 # Target delta for OTM1 calls
        
        # Greeks thresholds for OTM1 calls
        self.delta_thresholds = {'low': 0.1, 'high': 0.5}
        self.gamma_thresholds = {'low': 0.005, 'high': 0.04}
        self.theta_thresholds = {'low': -60, 'high': -5}
        self.vega_thresholds = {'low': 10, 'high': 80}
        
        # OTM characteristics
        self.max_time_value_ratio = 1.0  # 100% time value for OTM
        self.leverage_multiplier = 3.0    # Higher leverage for OTM
        
        self.logger.info("OTM1 Call analyzer initialized")
    
    def extract_component_price(self, data: Dict[str, Any]) -> Optional[float]:
        """
        Extract OTM1 Call price from market data
        
        Args:
            data: Market data dictionary containing option chain
            
        Returns:
            OTM1 Call price or None if not available
        """
        try:
            # Look for OTM1 Call price in different possible keys
            otm1_ce_keys = [
                'OTM1_CE', 'otm1_ce', 'OTM1_CALL', 'otm1_call',
                'otm1_ce_price', 'OTM1_CE_LTP', 'otm1_ce_ltp',
                'OTM_CE_1', 'otm_ce_1'
            ]
            
            for key in otm1_ce_keys:
                if key in data and data[key] is not None:
                    price = float(data[key])
                    if price > 0:  # Valid option price
                        return price
            
            # If direct OTM1 price not available, find from option chain
            underlying_price = data.get('underlying_price') or data.get('spot_price')
            if underlying_price and 'option_chain' in data:
                return self._find_otm1_call_from_chain(data['option_chain'], underlying_price)
            
            # Try alternative data structures
            if 'strikes' in data and 'ce_prices' in data:
                return self._find_otm1_from_strikes_data(data, underlying_price, 'CE')
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting OTM1 Call price: {e}")
            return None
    
    def _find_otm1_call_from_chain(self, option_chain: List[Dict], underlying_price: float) -> Optional[float]:
        """Find OTM1 Call from option chain data"""
        try:
            # Find CE options with strikes above underlying
            ce_options = [
                opt for opt in option_chain 
                if opt.get('option_type') == 'CE' and opt.get('strike', 0) > underlying_price
            ]
            
            if not ce_options:
                return None
            
            # Sort by strike ascending (lowest OTM strike first)
            ce_options.sort(key=lambda x: x.get('strike', 0))
            
            # Get the first OTM strike (1 strike OTM)
            if len(ce_options) >= self.strike_offset:
                otm1_option = ce_options[self.strike_offset - 1]
                return otm1_option.get('ltp') or otm1_option.get('price')
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding OTM1 Call from chain: {e}")
            return None
    
    def _find_otm1_from_strikes_data(self, data: Dict, underlying_price: float, option_type: str) -> Optional[float]:
        """Find OTM1 option from strikes and prices data"""
        try:
            strikes = data.get('strikes', [])
            prices_key = 'ce_prices' if option_type == 'CE' else 'pe_prices'
            prices = data.get(prices_key, [])
            
            if len(strikes) != len(prices):
                return None
            
            # Find OTM strikes for calls
            otm_indices = [
                i for i, strike in enumerate(strikes) 
                if strike > underlying_price
            ]
            
            if not otm_indices:
                return None
            
            # Sort by strike ascending
            otm_indices.sort(key=lambda i: strikes[i])
            
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
        Calculate OTM1 Call specific metrics
        
        Args:
            data: Market data dictionary
            
        Returns:
            Dictionary with OTM1 Call metrics
        """
        metrics = {}
        
        try:
            # Basic OTM1 Call metrics
            underlying_price = data.get('underlying_price') or data.get('spot_price', 0)
            otm1_ce_price = self.extract_component_price(data)
            
            if otm1_ce_price and underlying_price:
                # Get OTM1 strike
                strike = self._get_otm1_strike(data, underlying_price)
                
                # Intrinsic value (always 0 for OTM)
                metrics['intrinsic_value'] = 0.0
                
                # Time value (entire premium for OTM)
                metrics['time_value'] = otm1_ce_price
                metrics['time_value_ratio'] = 1.0  # 100% time value
                
                # Moneyness
                metrics['moneyness'] = underlying_price / strike if strike > 0 else 1.0
                metrics['otm_percentage'] = (strike - underlying_price) / underlying_price * 100 if underlying_price > 0 else 0
                
                # OTM premium characteristics
                metrics['premium_to_spot_ratio'] = otm1_ce_price / underlying_price if underlying_price > 0 else 0
                metrics['distance_to_strike'] = strike - underlying_price
                metrics['distance_to_strike_pct'] = metrics['otm_percentage']
                
                # Leverage calculation (OTM has high leverage)
                metrics['leverage_ratio'] = self._calculate_leverage_ratio(
                    otm1_ce_price, underlying_price, strike
                )
                
                # Probability estimates (simplified)
                metrics['probability_itm'] = self._estimate_probability_itm(
                    underlying_price, strike, data.get('implied_volatility', 0.2)
                )
            
            # Greeks analysis (if available)
            greeks = self._extract_greeks(data)
            if greeks:
                metrics.update(greeks)
                
                # OTM-specific Greek analysis
                metrics['delta_regime'] = self._classify_delta_regime(greeks.get('delta', 0.3))
                metrics['gamma_regime'] = self._classify_gamma_regime(greeks.get('gamma', 0))
                metrics['theta_decay_rate'] = abs(greeks.get('theta', 0)) / otm1_ce_price if otm1_ce_price > 0 else 0
                
                # OTM sensitivity metrics
                metrics['delta_sensitivity'] = greeks.get('delta', 0.3) * self.leverage_multiplier
                metrics['gamma_sensitivity'] = greeks.get('gamma', 0) * underlying_price / 100  # Per 1% move
                metrics['vega_impact'] = greeks.get('vega', 0) / otm1_ce_price if otm1_ce_price > 0 else 0
            
            # Volume and liquidity
            metrics['volume'] = data.get('volume', 0)
            metrics['open_interest'] = data.get('open_interest', 0)
            metrics['liquidity_score'] = self._calculate_liquidity_score(data)
            
            # OTM-specific indicators
            metrics['lottery_ticket_value'] = self._calculate_lottery_value(otm1_ce_price, metrics.get('probability_itm', 0))
            metrics['time_decay_intensity'] = self._calculate_time_decay_intensity(data, otm1_ce_price)
            
        except Exception as e:
            self.logger.error(f"Error calculating OTM1 Call metrics: {e}")
        
        return metrics
    
    def _get_otm1_strike(self, data: Dict[str, Any], underlying_price: float) -> float:
        """Get OTM1 strike price"""
        try:
            # Direct OTM1 strike if available
            if 'otm1_strike' in data:
                return float(data['otm1_strike'])
            
            # Calculate from strikes array
            if 'strikes' in data:
                strikes = sorted([s for s in data['strikes'] if s > underlying_price])
                if len(strikes) >= self.strike_offset:
                    return strikes[self.strike_offset - 1]
            
            # Default: estimate based on typical strike intervals
            strike_interval = 50  # Common for NIFTY
            return underlying_price + (strike_interval * self.strike_offset)
            
        except Exception:
            return underlying_price + 50  # Fallback
    
    def _extract_greeks(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract Greeks from market data"""
        greeks = {}
        
        # Common Greek keys for OTM calls
        greek_mappings = {
            'delta': ['delta', 'DELTA', 'Delta', 'otm1_ce_delta', 'OTM1_CE_DELTA'],
            'gamma': ['gamma', 'GAMMA', 'Gamma', 'otm1_ce_gamma', 'OTM1_CE_GAMMA'],
            'theta': ['theta', 'THETA', 'Theta', 'otm1_ce_theta', 'OTM1_CE_THETA'],
            'vega': ['vega', 'VEGA', 'Vega', 'otm1_ce_vega', 'OTM1_CE_VEGA'],
            'rho': ['rho', 'RHO', 'Rho', 'otm1_ce_rho', 'OTM1_CE_RHO']
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
    
    def _calculate_leverage_ratio(self, option_price: float, spot_price: float, strike: float) -> float:
        """Calculate leverage ratio for OTM option"""
        try:
            if option_price <= 0 or spot_price <= 0:
                return 0.0
            
            # Leverage = (Spot Price / Option Price) * Delta (approximated)
            # For OTM, we use simplified calculation
            basic_leverage = spot_price / option_price
            
            # Adjust for OTM distance
            distance_factor = 1 - ((strike - spot_price) / spot_price)
            adjusted_leverage = basic_leverage * max(distance_factor, 0.1)
            
            return min(adjusted_leverage, 100)  # Cap at 100x
            
        except Exception:
            return 10.0  # Default OTM leverage
    
    def _estimate_probability_itm(self, spot: float, strike: float, iv: float) -> float:
        """Estimate probability of finishing ITM (simplified)"""
        try:
            if iv <= 0:
                iv = 0.2  # Default 20% IV
            
            # Simplified probability calculation
            # In practice, use Black-Scholes N(d2)
            distance = (strike - spot) / spot
            vol_adjusted_distance = distance / (iv * np.sqrt(30/365))  # Assume 30 days
            
            # Approximate normal CDF
            if vol_adjusted_distance > 3:
                return 0.001
            elif vol_adjusted_distance > 2:
                return 0.023
            elif vol_adjusted_distance > 1:
                return 0.159
            elif vol_adjusted_distance > 0:
                return 0.5 - 0.341 * vol_adjusted_distance
            else:
                return 0.5
            
        except Exception:
            return 0.3  # Default OTM probability
    
    def _calculate_liquidity_score(self, data: Dict[str, Any]) -> float:
        """Calculate liquidity score for OTM option"""
        try:
            volume = data.get('volume', 0)
            open_interest = data.get('open_interest', 0)
            bid = data.get('bid', 0)
            ask = data.get('ask', 0)
            
            # OTM options often have lower liquidity
            # Volume score (lower threshold for OTM)
            volume_score = min(volume / 500, 1.0)
            
            # OI score (lower threshold for OTM)
            oi_score = min(open_interest / 5000, 1.0)
            
            # Spread score (wider spreads expected)
            if bid > 0 and ask > 0:
                spread_pct = (ask - bid) / ((ask + bid) / 2) * 100
                spread_score = max(0, 1 - spread_pct / 10)  # More tolerant
            else:
                spread_score = 0.3
            
            # Weighted average (adjusted for OTM)
            return (volume_score * 0.3 + oi_score * 0.4 + spread_score * 0.3)
            
        except Exception:
            return 0.3  # Lower default for OTM
    
    def _calculate_lottery_value(self, premium: float, probability: float) -> float:
        """Calculate lottery ticket value (expected payoff vs premium)"""
        try:
            if premium <= 0 or probability <= 0:
                return 0.0
            
            # Expected payoff = probability * potential gain
            # For OTM calls, potential gain is theoretically unlimited
            # Use conservative 10x premium as potential gain
            expected_payoff = probability * (premium * 10)
            
            # Lottery value = expected payoff / premium
            lottery_value = expected_payoff / premium
            
            return min(lottery_value, 5.0)  # Cap at 5x
            
        except Exception:
            return 1.0
    
    def _calculate_time_decay_intensity(self, data: Dict[str, Any], option_price: float) -> float:
        """Calculate time decay intensity for OTM option"""
        try:
            theta = data.get('theta', 0)
            days_to_expiry = data.get('days_to_expiry', 30)
            
            if option_price <= 0 or days_to_expiry <= 0:
                return 0.0
            
            # Daily decay as percentage of premium
            daily_decay_pct = abs(theta) / option_price * 100
            
            # Adjust for time to expiry (acceleration near expiry)
            if days_to_expiry < 7:
                intensity_multiplier = 3.0
            elif days_to_expiry < 15:
                intensity_multiplier = 2.0
            else:
                intensity_multiplier = 1.0
            
            return min(daily_decay_pct * intensity_multiplier, 10.0)  # Cap at 10% daily
            
        except Exception:
            return 2.0  # Default OTM decay intensity
    
    def _classify_delta_regime(self, delta: float) -> str:
        """Classify delta regime for OTM1 Call"""
        if delta < self.delta_thresholds['low']:
            return 'DEEP_OTM'  # Very low probability
        elif delta > self.delta_thresholds['high']:
            return 'NEAR_ATM'  # Moving toward ATM
        else:
            return 'NORMAL_OTM'  # Typical OTM1
    
    def _classify_gamma_regime(self, gamma: float) -> str:
        """Classify gamma regime for OTM1 Call"""
        abs_gamma = abs(gamma)
        if abs_gamma < self.gamma_thresholds['low']:
            return 'LOW_GAMMA'  # Deep OTM, low sensitivity
        elif abs_gamma > self.gamma_thresholds['high']:
            return 'HIGH_GAMMA'  # Significant gamma risk
        else:
            return 'NORMAL_GAMMA'
    
    def calculate_regime_contribution(self, analysis_result: ComponentAnalysisResult) -> Dict[str, float]:
        """
        Calculate OTM1 Call contribution to regime formation
        
        Args:
            analysis_result: Component analysis result
            
        Returns:
            Dictionary with regime indicators
        """
        regime_indicators = {}
        
        try:
            # Volatility contribution (OTM perspective)
            volatility_indicators = self._calculate_volatility_contribution(analysis_result)
            regime_indicators.update(volatility_indicators)
            
            # Trend contribution (OTM calls benefit from strong uptrends)
            trend_indicators = self._calculate_trend_contribution(analysis_result)
            regime_indicators.update(trend_indicators)
            
            # Structure contribution
            structure_indicators = self._calculate_structure_contribution(analysis_result)
            regime_indicators.update(structure_indicators)
            
            # OTM-specific indicators
            otm_indicators = self._calculate_otm_specific_indicators(analysis_result)
            regime_indicators.update(otm_indicators)
            
            # Overall regime confidence
            regime_indicators['regime_confidence'] = self._calculate_regime_confidence(analysis_result)
            
        except Exception as e:
            self.logger.error(f"Error calculating regime contribution: {e}")
        
        return regime_indicators
    
    def _calculate_volatility_contribution(self, result: ComponentAnalysisResult) -> Dict[str, float]:
        """Calculate volatility regime contribution from OTM perspective"""
        indicators = {}
        
        try:
            # OTM options are highly sensitive to volatility
            if 'vega_impact' in result.component_metrics:
                vega_impact = result.component_metrics['vega_impact']
                
                # High vega impact suggests volatility opportunity
                if vega_impact > 0.5:
                    indicators['otm_volatility_signal'] = 1.0
                elif vega_impact < 0.1:
                    indicators['otm_volatility_signal'] = -1.0
                else:
                    indicators['otm_volatility_signal'] = 0.0
                
                indicators['vega_sensitivity'] = min(vega_impact * 2, 1.0)
            
            # Time decay intensity indicates volatility need
            if 'time_decay_intensity' in result.component_metrics:
                decay_intensity = result.component_metrics['time_decay_intensity']
                
                # High decay needs high volatility to compensate
                if decay_intensity > 5:
                    indicators['decay_pressure'] = 1.0  # High volatility needed
                elif decay_intensity < 2:
                    indicators['decay_pressure'] = -1.0  # Low volatility acceptable
                else:
                    indicators['decay_pressure'] = 0.0
            
        except Exception as e:
            self.logger.warning(f"Error calculating volatility contribution: {e}")
        
        return indicators
    
    def _calculate_trend_contribution(self, result: ComponentAnalysisResult) -> Dict[str, float]:
        """Calculate trend regime contribution from OTM perspective"""
        indicators = {}
        
        try:
            # OTM calls need strong bullish trends
            if 'delta' in result.component_metrics:
                delta = result.component_metrics['delta']
                
                # Low delta OTM needs momentum
                if delta < 0.2:
                    indicators['otm_trend_requirement'] = 1.0  # Needs strong trend
                elif delta > 0.4:
                    indicators['otm_trend_requirement'] = 0.0  # Less trend dependent
                else:
                    indicators['otm_trend_requirement'] = 0.5
                
                indicators['momentum_sensitivity'] = 1 - delta  # Higher for lower delta
            
            # Price momentum analysis
            price_change_pct = result.price_change_percent
            if abs(price_change_pct) > 0.1:
                # OTM calls amplify upward moves
                if price_change_pct > 3:  # Strong gain
                    indicators['otm_momentum_signal'] = 1.0
                elif price_change_pct < -3:  # Strong loss
                    indicators['otm_momentum_signal'] = -1.0
                else:
                    indicators['otm_momentum_signal'] = price_change_pct / 5
                
                # Leverage effect
                leverage = result.component_metrics.get('leverage_ratio', 10)
                indicators['leveraged_momentum'] = min(abs(price_change_pct) * leverage / 100, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Error calculating trend contribution: {e}")
        
        return indicators
    
    def _calculate_structure_contribution(self, result: ComponentAnalysisResult) -> Dict[str, float]:
        """Calculate structure regime contribution"""
        indicators = {}
        
        try:
            # Liquidity structure for OTM
            if 'liquidity_score' in result.component_metrics:
                liquidity = result.component_metrics['liquidity_score']
                
                # OTM needs reasonable liquidity for entry/exit
                if liquidity > 0.5:
                    indicators['otm_liquidity_structure'] = 1.0  # Tradeable
                elif liquidity < 0.2:
                    indicators['otm_liquidity_structure'] = -1.0  # Illiquid
                else:
                    indicators['otm_liquidity_structure'] = 0.0
                
                indicators['market_accessibility'] = liquidity
            
            # Volume patterns
            volume_ratio = result.volume_ratio
            if volume_ratio > 2:  # High relative volume
                indicators['otm_interest_level'] = 1.0  # High speculation
            elif volume_ratio < 0.5:
                indicators['otm_interest_level'] = -1.0  # Low interest
            else:
                indicators['otm_interest_level'] = 0.0
            
        except Exception as e:
            self.logger.warning(f"Error calculating structure contribution: {e}")
        
        return indicators
    
    def _calculate_otm_specific_indicators(self, result: ComponentAnalysisResult) -> Dict[str, float]:
        """Calculate OTM-specific regime indicators"""
        indicators = {}
        
        try:
            # Lottery value assessment
            if 'lottery_ticket_value' in result.component_metrics:
                lottery_value = result.component_metrics['lottery_ticket_value']
                
                if lottery_value > 2:
                    indicators['speculation_regime'] = 1.0  # High speculation
                elif lottery_value < 0.5:
                    indicators['speculation_regime'] = -1.0  # Low speculation
                else:
                    indicators['speculation_regime'] = 0.0
                
                indicators['lottery_attractiveness'] = min(lottery_value / 3, 1.0)
            
            # Probability assessment
            if 'probability_itm' in result.component_metrics:
                prob_itm = result.component_metrics['probability_itm']
                
                # Very low probability suggests ranging market
                if prob_itm < 0.1:
                    indicators['probability_regime'] = -1.0  # Ranging
                elif prob_itm > 0.3:
                    indicators['probability_regime'] = 1.0  # Trending
                else:
                    indicators['probability_regime'] = 0.0
                
                indicators['success_likelihood'] = prob_itm
            
            # Distance to strike analysis
            if 'distance_to_strike_pct' in result.component_metrics:
                distance_pct = result.component_metrics['distance_to_strike_pct']
                
                # Closer distance suggests momentum
                if distance_pct < 2:
                    indicators['strike_proximity'] = 1.0  # Near strike
                elif distance_pct > 5:
                    indicators['strike_proximity'] = -1.0  # Far from strike
                else:
                    indicators['strike_proximity'] = 0.0
                
                indicators['reachability_factor'] = max(0, 1 - distance_pct / 10)
            
        except Exception as e:
            self.logger.warning(f"Error calculating OTM-specific indicators: {e}")
        
        return indicators
    
    def _calculate_regime_confidence(self, result: ComponentAnalysisResult) -> float:
        """Calculate overall regime confidence for OTM component"""
        try:
            confidence_factors = []
            
            # Data availability confidence
            available_windows = sum(1 for window in self.rolling_windows 
                                  if window in result.rolling_metrics)
            confidence_factors.append(available_windows / len(self.rolling_windows))
            
            # Volume confidence (lower threshold for OTM)
            if result.volume_ratio > 0.3:
                confidence_factors.append(min(result.volume_ratio * 1.5, 1.0))
            
            # Delta confidence (OTM should have low delta)
            if 'delta' in result.component_metrics:
                delta = result.component_metrics['delta']
                if 0.1 < delta < 0.5:  # Proper OTM delta range
                    confidence_factors.append(0.8)
                else:
                    confidence_factors.append(0.3)  # Low confidence if delta out of range
            
            # Time value confidence (OTM should be 100% time value)
            if 'time_value_ratio' in result.component_metrics:
                time_ratio = result.component_metrics['time_value_ratio']
                confidence_factors.append(time_ratio)  # Should be 1.0 for OTM
            
            # Liquidity confidence (adjusted for OTM)
            if 'liquidity_score' in result.component_metrics:
                liquidity = result.component_metrics['liquidity_score']
                confidence_factors.append(min(liquidity * 1.5, 1.0))  # Boost for OTM
            
            return np.mean(confidence_factors) if confidence_factors else 0.5
            
        except Exception:
            return 0.5