"""
ITM1 Put (PE) Component Analyzer

Analyzes ITM1 (In-The-Money 1 strike) Put options with rolling analysis 
across [3,5,10,15] minute windows. Provides comprehensive technical analysis, 
performance tracking, and regime contribution for the ITM1 Put component.
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


class ITM1PutAnalyzer(BaseComponentAnalyzer):
    """
    ITM1 Put (PE) Component Analyzer
    
    Specialized analyzer for ITM1 Put options with:
    - Strike selection 1 strike above ATM
    - Higher negative delta sensitivity (typically -0.6 to -0.8)
    - Intrinsic value tracking for puts
    - ITM Put-specific regime indicators
    """
    
    def __init__(self, 
                 config: StraddleConfig,
                 calculation_engine: CalculationEngine,
                 window_manager: RollingWindowManager):
        """
        Initialize ITM1 Put analyzer
        
        Args:
            config: Straddle configuration
            calculation_engine: Shared calculation engine
            window_manager: Rolling window manager
        """
        super().__init__('itm1_pe', config, calculation_engine, window_manager)
        
        # ITM1 Put-specific configuration
        self.strike_offset = 1  # Number of strikes in-the-money for puts
        self.delta_target = -0.7 # Target delta for ITM1 puts
        
        # Greeks thresholds for ITM1 puts
        self.delta_thresholds = {'low': -0.85, 'high': -0.5}
        self.gamma_thresholds = {'low': 0.005, 'high': 0.03}
        self.theta_thresholds = {'low': -30, 'high': -5}
        self.vega_thresholds = {'low': 5, 'high': 50}
        
        # ITM Put characteristics
        self.min_intrinsic_ratio = 0.2  # Minimum intrinsic value ratio for ITM
        
        self.logger.info("ITM1 Put analyzer initialized")
    
    def extract_component_price(self, data: Dict[str, Any]) -> Optional[float]:
        """
        Extract ITM1 Put price from market data
        
        Args:
            data: Market data dictionary containing option chain
            
        Returns:
            ITM1 Put price or None if not available
        """
        try:
            # Look for ITM1 Put price in different possible keys
            itm1_pe_keys = [
                'ITM1_PE', 'itm1_pe', 'ITM1_PUT', 'itm1_put',
                'itm1_pe_price', 'ITM1_PE_LTP', 'itm1_pe_ltp',
                'ITM_PE_1', 'itm_pe_1'
            ]
            
            for key in itm1_pe_keys:
                if key in data and data[key] is not None:
                    price = float(data[key])
                    if price > 0:  # Valid option price
                        return price
            
            # If direct ITM1 price not available, find from option chain
            underlying_price = data.get('underlying_price') or data.get('spot_price')
            if underlying_price and 'option_chain' in data:
                return self._find_itm1_put_from_chain(data['option_chain'], underlying_price)
            
            # Try alternative data structures
            if 'strikes' in data and 'pe_prices' in data:
                return self._find_itm1_from_strikes_data(data, underlying_price, 'PE')
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting ITM1 Put price: {e}")
            return None
    
    def _find_itm1_put_from_chain(self, option_chain: List[Dict], underlying_price: float) -> Optional[float]:
        """Find ITM1 Put from option chain data"""
        try:
            # Find PE options with strikes above underlying (ITM for puts)
            pe_options = [
                opt for opt in option_chain 
                if opt.get('option_type') == 'PE' and opt.get('strike', 0) > underlying_price
            ]
            
            if not pe_options:
                return None
            
            # Sort by strike ascending (lowest ITM strike first)
            pe_options.sort(key=lambda x: x.get('strike', 0))
            
            # Get the first ITM strike (1 strike ITM)
            if len(pe_options) >= self.strike_offset:
                itm1_option = pe_options[self.strike_offset - 1]
                return itm1_option.get('ltp') or itm1_option.get('price')
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding ITM1 Put from chain: {e}")
            return None
    
    def _find_itm1_from_strikes_data(self, data: Dict, underlying_price: float, option_type: str) -> Optional[float]:
        """Find ITM1 option from strikes and prices data"""
        try:
            strikes = data.get('strikes', [])
            prices_key = 'ce_prices' if option_type == 'CE' else 'pe_prices'
            prices = data.get(prices_key, [])
            
            if len(strikes) != len(prices):
                return None
            
            # Find ITM strikes for puts (strikes above underlying)
            itm_indices = [
                i for i, strike in enumerate(strikes) 
                if strike > underlying_price
            ]
            
            if not itm_indices:
                return None
            
            # Sort by strike ascending
            itm_indices.sort(key=lambda i: strikes[i])
            
            # Get ITM1 strike
            if len(itm_indices) >= self.strike_offset:
                itm1_idx = itm_indices[self.strike_offset - 1]
                return prices[itm1_idx]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding ITM1 from strikes data: {e}")
            return None
    
    def calculate_component_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate ITM1 Put specific metrics
        
        Args:
            data: Market data dictionary
            
        Returns:
            Dictionary with ITM1 Put metrics
        """
        metrics = {}
        
        try:
            # Basic ITM1 Put metrics
            underlying_price = data.get('underlying_price') or data.get('spot_price', 0)
            itm1_pe_price = self.extract_component_price(data)
            
            if itm1_pe_price and underlying_price:
                # Get ITM1 strike
                strike = self._get_itm1_strike(data, underlying_price)
                
                # Intrinsic value (significant for ITM puts)
                metrics['intrinsic_value'] = max(0, strike - underlying_price)
                
                # Time value
                metrics['time_value'] = itm1_pe_price - metrics['intrinsic_value']
                
                # ITM-specific ratios
                metrics['intrinsic_ratio'] = metrics['intrinsic_value'] / itm1_pe_price if itm1_pe_price > 0 else 0
                metrics['time_value_ratio'] = metrics['time_value'] / itm1_pe_price if itm1_pe_price > 0 else 0
                
                # Moneyness (inverted for puts)
                metrics['moneyness'] = strike / underlying_price if underlying_price > 0 else 1.0
                metrics['itm_percentage'] = (strike - underlying_price) / underlying_price * 100 if underlying_price > 0 else 0
                
                # ITM Put efficiency
                metrics['itm_efficiency'] = self._calculate_itm_put_efficiency(
                    itm1_pe_price, metrics['intrinsic_value'], metrics['time_value']
                )
                
                # Put-call relationship with ITM1 CE
                itm1_ce_price = self._get_companion_call_price(data)
                if itm1_ce_price:
                    metrics['put_call_relationship'] = self._calculate_put_call_relationship(
                        itm1_ce_price, itm1_pe_price, underlying_price, strike
                    )
            
            # Greeks analysis (if available)
            greeks = self._extract_greeks(data)
            if greeks:
                metrics.update(greeks)
                
                # ITM Put-specific Greek analysis
                metrics['delta_regime'] = self._classify_delta_regime(greeks.get('delta', -0.7))
                metrics['gamma_regime'] = self._classify_gamma_regime(greeks.get('gamma', 0))
                metrics['theta_decay_rate'] = abs(greeks.get('theta', 0)) / itm1_pe_price if itm1_pe_price > 0 else 0
                
                # Put delta leverage (negative)
                metrics['delta_leverage'] = abs(greeks.get('delta', -0.7)) * underlying_price / itm1_pe_price if itm1_pe_price > 0 else 0
            
            # Volume and liquidity
            metrics['volume'] = data.get('volume', 0)
            metrics['open_interest'] = data.get('open_interest', 0)
            metrics['liquidity_score'] = self._calculate_liquidity_score(data)
            
            # ITM Put premium analysis
            metrics['itm_put_premium'] = self._calculate_itm_put_premium(data, itm1_pe_price)
            
            # Downside protection level
            metrics['protection_level'] = self._calculate_protection_level(data, itm1_pe_price, strike)
            
        except Exception as e:
            self.logger.error(f"Error calculating ITM1 Put metrics: {e}")
        
        return metrics
    
    def _get_itm1_strike(self, data: Dict[str, Any], underlying_price: float) -> float:
        """Get ITM1 strike price for puts"""
        try:
            # Direct ITM1 strike if available
            if 'itm1_strike' in data:
                return float(data['itm1_strike'])
            
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
    
    def _get_companion_call_price(self, data: Dict[str, Any]) -> Optional[float]:
        """Get corresponding ITM1 Call price for put-call analysis"""
        itm1_ce_keys = [
            'ITM1_CE', 'itm1_ce', 'ITM1_CALL', 'itm1_call',
            'itm1_ce_price', 'ITM1_CE_LTP', 'itm1_ce_ltp'
        ]
        
        for key in itm1_ce_keys:
            if key in data and data[key] is not None:
                try:
                    return float(data[key])
                except (ValueError, TypeError):
                    continue
        return None
    
    def _extract_greeks(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract Greeks from market data"""
        greeks = {}
        
        # Common Greek keys for ITM puts
        greek_mappings = {
            'delta': ['delta', 'DELTA', 'Delta', 'itm1_pe_delta', 'ITM1_PE_DELTA'],
            'gamma': ['gamma', 'GAMMA', 'Gamma', 'itm1_pe_gamma', 'ITM1_PE_GAMMA'],
            'theta': ['theta', 'THETA', 'Theta', 'itm1_pe_theta', 'ITM1_PE_THETA'],
            'vega': ['vega', 'VEGA', 'Vega', 'itm1_pe_vega', 'ITM1_PE_VEGA'],
            'rho': ['rho', 'RHO', 'Rho', 'itm1_pe_rho', 'ITM1_PE_RHO']
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
    
    def _calculate_itm_put_efficiency(self, option_price: float, intrinsic: float, time_value: float) -> float:
        """Calculate ITM put option efficiency"""
        try:
            if option_price <= 0:
                return 0.0
            
            # ITM efficiency based on intrinsic/time value balance
            intrinsic_ratio = intrinsic / option_price
            
            # Ideal ITM put has high intrinsic ratio but some time value
            if intrinsic_ratio > 0.8:  # Deep ITM
                efficiency = 0.7
            elif intrinsic_ratio > 0.6:  # Good ITM
                efficiency = 0.9
            elif intrinsic_ratio > 0.4:  # Moderate ITM
                efficiency = 1.0
            else:  # Barely ITM
                efficiency = 0.8
            
            # Adjust for time value
            if time_value < 0:  # Negative time value (anomaly)
                efficiency *= 0.5
            elif time_value / option_price > 0.3:  # High time value
                efficiency *= 0.9
            
            return efficiency
            
        except Exception:
            return 0.5
    
    def _calculate_put_call_relationship(self, call_price: float, put_price: float, 
                                       spot_price: float, strike: float) -> Dict[str, float]:
        """Calculate put-call relationship for ITM options"""
        try:
            # For ITM options, the relationship is different
            # ITM Put intrinsic = strike - spot
            # ITM Call intrinsic = spot - strike (different strike)
            
            put_intrinsic = max(0, strike - spot_price)
            put_time_value = put_price - put_intrinsic
            
            # Analyze time value relationship
            return {
                'put_premium': put_price,
                'call_premium': call_price,
                'put_intrinsic': put_intrinsic,
                'put_time_value': put_time_value,
                'premium_ratio': put_price / call_price if call_price > 0 else 0,
                'time_value_spread': put_time_value
            }
        except Exception:
            return {'premium_ratio': 1.0}
    
    def _calculate_liquidity_score(self, data: Dict[str, Any]) -> float:
        """Calculate liquidity score for ITM put option"""
        try:
            volume = data.get('volume', 0)
            open_interest = data.get('open_interest', 0)
            bid = data.get('bid', 0)
            ask = data.get('ask', 0)
            
            # Volume score
            volume_score = min(volume / 1000, 1.0)
            
            # OI score
            oi_score = min(open_interest / 10000, 1.0)
            
            # Spread score
            if bid > 0 and ask > 0:
                spread_pct = (ask - bid) / ((ask + bid) / 2) * 100
                spread_score = max(0, 1 - spread_pct / 5)
            else:
                spread_score = 0.5
            
            # Weighted average
            return (volume_score * 0.4 + oi_score * 0.4 + spread_score * 0.2)
            
        except Exception:
            return 0.5
    
    def _calculate_itm_put_premium(self, data: Dict[str, Any], option_price: float) -> Dict[str, float]:
        """Calculate ITM put premium characteristics"""
        premium = {}
        
        try:
            underlying_price = data.get('underlying_price', 0)
            strike = self._get_itm1_strike(data, underlying_price)
            
            if underlying_price > 0 and strike > 0:
                # Premium over intrinsic
                intrinsic = max(0, strike - underlying_price)
                premium['over_intrinsic'] = option_price - intrinsic
                premium['over_intrinsic_pct'] = (premium['over_intrinsic'] / intrinsic * 100) if intrinsic > 0 else 0
                
                # Breakeven analysis for puts
                premium['breakeven'] = strike - option_price
                premium['breakeven_distance'] = underlying_price - premium['breakeven']
                premium['breakeven_pct'] = (premium['breakeven_distance'] / underlying_price * 100)
                
        except Exception:
            pass
        
        return premium
    
    def _calculate_protection_level(self, data: Dict[str, Any], option_price: float, strike: float) -> float:
        """Calculate downside protection level for ITM put"""
        try:
            underlying_price = data.get('underlying_price', 0)
            
            if underlying_price > 0:
                # Protection = strike - option_price
                protection_price = strike - option_price
                protection_pct = (underlying_price - protection_price) / underlying_price * 100
                return max(protection_pct, 0)
            
            return 0.0
        except Exception:
            return 0.0
    
    def _classify_delta_regime(self, delta: float) -> str:
        """Classify delta regime for ITM1 Put"""
        if delta > self.delta_thresholds['high']:  # Less negative
            return 'LOW_DELTA'  # Moving toward ATM
        elif delta < self.delta_thresholds['low']:  # More negative
            return 'HIGH_DELTA'  # Deep ITM
        else:
            return 'NORMAL_DELTA'  # Typical ITM1
    
    def _classify_gamma_regime(self, gamma: float) -> str:
        """Classify gamma regime for ITM1 Put"""
        abs_gamma = abs(gamma)
        if abs_gamma < self.gamma_thresholds['low']:
            return 'LOW_GAMMA'  # Deep ITM, low sensitivity
        elif abs_gamma > self.gamma_thresholds['high']:
            return 'HIGH_GAMMA'  # Near ATM, high sensitivity
        else:
            return 'NORMAL_GAMMA'
    
    def calculate_regime_contribution(self, analysis_result: ComponentAnalysisResult) -> Dict[str, float]:
        """
        Calculate ITM1 Put contribution to regime formation
        
        Args:
            analysis_result: Component analysis result
            
        Returns:
            Dictionary with regime indicators
        """
        regime_indicators = {}
        
        try:
            # Volatility contribution (ITM Put perspective)
            volatility_indicators = self._calculate_volatility_contribution(analysis_result)
            regime_indicators.update(volatility_indicators)
            
            # Trend contribution (ITM Put has strong directional bias)
            trend_indicators = self._calculate_trend_contribution(analysis_result)
            regime_indicators.update(trend_indicators)
            
            # Structure contribution
            structure_indicators = self._calculate_structure_contribution(analysis_result)
            regime_indicators.update(structure_indicators)
            
            # ITM Put-specific indicators
            itm_put_indicators = self._calculate_itm_put_specific_indicators(analysis_result)
            regime_indicators.update(itm_put_indicators)
            
            # Overall regime confidence
            regime_indicators['regime_confidence'] = self._calculate_regime_confidence(analysis_result)
            
        except Exception as e:
            self.logger.error(f"Error calculating regime contribution: {e}")
        
        return regime_indicators
    
    def _calculate_volatility_contribution(self, result: ComponentAnalysisResult) -> Dict[str, float]:
        """Calculate volatility regime contribution from ITM Put perspective"""
        indicators = {}
        
        try:
            # Time value decay rate indicates volatility
            if 'theta_decay_rate' in result.component_metrics:
                decay_rate = result.component_metrics['theta_decay_rate']
                
                # High decay rate suggests high volatility
                if decay_rate > 0.02:  # >2% daily decay
                    indicators['theta_volatility_signal'] = 1.0
                elif decay_rate < 0.005:  # <0.5% daily decay
                    indicators['theta_volatility_signal'] = -1.0
                else:
                    indicators['theta_volatility_signal'] = 0.0
                
                indicators['put_decay_intensity'] = min(decay_rate * 50, 1.0)
            
            # Vega for volatility sensitivity
            if 'vega' in result.component_metrics:
                vega = result.component_metrics['vega']
                indicators['put_vega_exposure'] = min(abs(vega) / 50, 1.0)  # Normalize for ITM
            
        except Exception as e:
            self.logger.warning(f"Error calculating volatility contribution: {e}")
        
        return indicators
    
    def _calculate_trend_contribution(self, result: ComponentAnalysisResult) -> Dict[str, float]:
        """Calculate trend regime contribution from ITM Put perspective"""
        indicators = {}
        
        try:
            # ITM puts have strong bearish bias
            if 'delta' in result.component_metrics:
                delta = result.component_metrics['delta']
                
                # More negative delta ITM suggests strong bearish trend
                if delta < -0.8:
                    indicators['put_delta_trend_signal'] = -1.0  # Strong bearish
                elif delta > -0.6:
                    indicators['put_delta_trend_signal'] = 0.5  # Weakening bearish
                else:
                    indicators['put_delta_trend_signal'] = -0.5  # Moderate bearish
                
                indicators['put_directional_strength'] = abs(delta)
            
            # Price momentum (puts gain when underlying falls)
            price_change_pct = result.price_change_percent
            if abs(price_change_pct) > 0.1:
                # ITM puts move inversely with underlying
                if price_change_pct > 1:  # Put price increasing
                    indicators['put_price_momentum_signal'] = -1.0  # Bearish market
                elif price_change_pct < -1:  # Put price decreasing
                    indicators['put_price_momentum_signal'] = 1.0  # Bullish market
                else:
                    indicators['put_price_momentum_signal'] = -price_change_pct / 2
                
                indicators['put_momentum_strength'] = min(abs(price_change_pct) / 5, 1.0)
            
            # Delta leverage indicates trend strength
            if 'delta_leverage' in result.component_metrics:
                leverage = result.component_metrics['delta_leverage']
                indicators['put_leverage_factor'] = min(leverage / 10, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Error calculating trend contribution: {e}")
        
        return indicators
    
    def _calculate_structure_contribution(self, result: ComponentAnalysisResult) -> Dict[str, float]:
        """Calculate structure regime contribution"""
        indicators = {}
        
        try:
            # Liquidity score indicates market structure
            if 'liquidity_score' in result.component_metrics:
                liquidity = result.component_metrics['liquidity_score']
                
                if liquidity > 0.7:
                    indicators['put_liquidity_structure'] = 1.0  # Well-structured
                elif liquidity < 0.3:
                    indicators['put_liquidity_structure'] = -1.0  # Poor structure
                else:
                    indicators['put_liquidity_structure'] = 0.0
                
                indicators['put_market_depth'] = liquidity
            
            # Intrinsic ratio indicates structure efficiency
            if 'intrinsic_ratio' in result.component_metrics:
                intrinsic_ratio = result.component_metrics['intrinsic_ratio']
                
                # Very high intrinsic ratio suggests trending market
                if intrinsic_ratio > 0.8:
                    indicators['put_efficiency_structure'] = 1.0
                elif intrinsic_ratio < 0.4:
                    indicators['put_efficiency_structure'] = -1.0
                else:
                    indicators['put_efficiency_structure'] = 0.0
            
        except Exception as e:
            self.logger.warning(f"Error calculating structure contribution: {e}")
        
        return indicators
    
    def _calculate_itm_put_specific_indicators(self, result: ComponentAnalysisResult) -> Dict[str, float]:
        """Calculate ITM Put-specific regime indicators"""
        indicators = {}
        
        try:
            # ITM Put efficiency as regime indicator
            if 'itm_efficiency' in result.component_metrics:
                efficiency = result.component_metrics['itm_efficiency']
                indicators['itm_put_quality'] = efficiency
                
                if efficiency > 0.85:
                    indicators['itm_put_regime'] = 1.0  # Efficient ITM put market
                elif efficiency < 0.6:
                    indicators['itm_put_regime'] = -1.0  # Inefficient ITM put market
                else:
                    indicators['itm_put_regime'] = 0.0
            
            # Protection level analysis
            if 'protection_level' in result.component_metrics:
                protection = result.component_metrics['protection_level']
                
                # Higher protection suggests defensive market
                if protection > 15:
                    indicators['protection_regime'] = 1.0  # High protection demand
                elif protection < 5:
                    indicators['protection_regime'] = -1.0  # Low protection demand
                else:
                    indicators['protection_regime'] = 0.0
                
                indicators['protection_intensity'] = min(protection / 20, 1.0)
            
            # Breakeven analysis
            if 'itm_put_premium' in result.component_metrics:
                premium_data = result.component_metrics['itm_put_premium']
                if isinstance(premium_data, dict):
                    breakeven_pct = premium_data.get('breakeven_pct', 0)
                    
                    # Closer breakeven suggests confident bearish market
                    if abs(breakeven_pct) < 3:
                        indicators['put_breakeven_confidence'] = 1.0
                    elif abs(breakeven_pct) > 7:
                        indicators['put_breakeven_confidence'] = -1.0
                    else:
                        indicators['put_breakeven_confidence'] = 0.0
                    
                    indicators['put_breakeven_distance'] = min(abs(breakeven_pct) / 10, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Error calculating ITM Put-specific indicators: {e}")
        
        return indicators
    
    def _calculate_regime_confidence(self, result: ComponentAnalysisResult) -> float:
        """Calculate overall regime confidence for ITM Put component"""
        try:
            confidence_factors = []
            
            # Data availability confidence
            available_windows = sum(1 for window in self.rolling_windows 
                                  if window in result.rolling_metrics)
            confidence_factors.append(available_windows / len(self.rolling_windows))
            
            # Volume confidence
            if result.volume_ratio > 0.5:
                confidence_factors.append(min(result.volume_ratio, 1.0))
            
            # Delta confidence (ITM Put should have negative delta)
            if 'delta' in result.component_metrics:
                delta = result.component_metrics['delta']
                if delta < -0.6:  # Proper ITM put delta
                    confidence_factors.append(abs(delta))
                else:
                    confidence_factors.append(0.3)  # Low confidence if delta not negative enough
            
            # Intrinsic value confidence
            if 'intrinsic_ratio' in result.component_metrics:
                intrinsic_ratio = result.component_metrics['intrinsic_ratio']
                if intrinsic_ratio > self.min_intrinsic_ratio:
                    confidence_factors.append(min(intrinsic_ratio, 1.0))
                else:
                    confidence_factors.append(0.2)  # Low confidence if not properly ITM
            
            # Liquidity confidence
            if 'liquidity_score' in result.component_metrics:
                confidence_factors.append(result.component_metrics['liquidity_score'])
            
            return np.mean(confidence_factors) if confidence_factors else 0.5
            
        except Exception:
            return 0.5