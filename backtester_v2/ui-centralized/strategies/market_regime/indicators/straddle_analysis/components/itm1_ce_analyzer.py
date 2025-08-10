"""
ITM1 Call (CE) Component Analyzer

Analyzes ITM1 (In-The-Money 1 strike) Call options with rolling analysis 
across [3,5,10,15] minute windows. Provides comprehensive technical analysis, 
performance tracking, and regime contribution for the ITM1 Call component.
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


class ITM1CallAnalyzer(BaseComponentAnalyzer):
    """
    ITM1 Call (CE) Component Analyzer
    
    Specialized analyzer for ITM1 Call options with:
    - Strike selection 1 strike below ATM
    - Higher delta sensitivity (typically 0.6-0.8)
    - Intrinsic value tracking
    - ITM-specific regime indicators
    """
    
    def __init__(self, 
                 config: StraddleConfig,
                 calculation_engine: CalculationEngine,
                 window_manager: RollingWindowManager):
        """
        Initialize ITM1 Call analyzer
        
        Args:
            config: Straddle configuration
            calculation_engine: Shared calculation engine
            window_manager: Rolling window manager
        """
        super().__init__('itm1_ce', config, calculation_engine, window_manager)
        
        # ITM1-specific configuration
        self.strike_offset = 1  # Number of strikes in-the-money
        self.delta_target = 0.7 # Target delta for ITM1 calls
        
        # Greeks thresholds for ITM1 calls
        self.delta_thresholds = {'low': 0.5, 'high': 0.85}
        self.gamma_thresholds = {'low': 0.005, 'high': 0.03}
        self.theta_thresholds = {'low': -30, 'high': -5}
        self.vega_thresholds = {'low': 5, 'high': 50}
        
        # ITM characteristics
        self.min_intrinsic_ratio = 0.2  # Minimum intrinsic value ratio for ITM
        
        self.logger.info("ITM1 Call analyzer initialized")
    
    def extract_component_price(self, data: Dict[str, Any]) -> Optional[float]:
        """
        Extract ITM1 Call price from market data
        
        Args:
            data: Market data dictionary containing option chain
            
        Returns:
            ITM1 Call price or None if not available
        """
        try:
            # Look for ITM1 Call price in different possible keys
            itm1_ce_keys = [
                'ITM1_CE', 'itm1_ce', 'ITM1_CALL', 'itm1_call',
                'itm1_ce_price', 'ITM1_CE_LTP', 'itm1_ce_ltp',
                'ITM_CE_1', 'itm_ce_1'
            ]
            
            for key in itm1_ce_keys:
                if key in data and data[key] is not None:
                    price = float(data[key])
                    if price > 0:  # Valid option price
                        return price
            
            # If direct ITM1 price not available, find from option chain
            underlying_price = data.get('underlying_price') or data.get('spot_price')
            if underlying_price and 'option_chain' in data:
                return self._find_itm1_call_from_chain(data['option_chain'], underlying_price)
            
            # Try alternative data structures
            if 'strikes' in data and 'ce_prices' in data:
                return self._find_itm1_from_strikes_data(data, underlying_price, 'CE')
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting ITM1 Call price: {e}")
            return None
    
    def _find_itm1_call_from_chain(self, option_chain: List[Dict], underlying_price: float) -> Optional[float]:
        """Find ITM1 Call from option chain data"""
        try:
            # Find CE options with strikes below underlying
            ce_options = [
                opt for opt in option_chain 
                if opt.get('option_type') == 'CE' and opt.get('strike', 0) < underlying_price
            ]
            
            if not ce_options:
                return None
            
            # Sort by strike descending (highest ITM strike first)
            ce_options.sort(key=lambda x: x.get('strike', 0), reverse=True)
            
            # Get the first ITM strike (1 strike ITM)
            if len(ce_options) >= self.strike_offset:
                itm1_option = ce_options[self.strike_offset - 1]
                return itm1_option.get('ltp') or itm1_option.get('price')
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding ITM1 Call from chain: {e}")
            return None
    
    def _find_itm1_from_strikes_data(self, data: Dict, underlying_price: float, option_type: str) -> Optional[float]:
        """Find ITM1 option from strikes and prices data"""
        try:
            strikes = data.get('strikes', [])
            prices_key = 'ce_prices' if option_type == 'CE' else 'pe_prices'
            prices = data.get(prices_key, [])
            
            if len(strikes) != len(prices):
                return None
            
            # Find ITM strikes for calls
            itm_indices = [
                i for i, strike in enumerate(strikes) 
                if strike < underlying_price
            ]
            
            if not itm_indices:
                return None
            
            # Sort by strike descending
            itm_indices.sort(key=lambda i: strikes[i], reverse=True)
            
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
        Calculate ITM1 Call specific metrics
        
        Args:
            data: Market data dictionary
            
        Returns:
            Dictionary with ITM1 Call metrics
        """
        metrics = {}
        
        try:
            # Basic ITM1 Call metrics
            underlying_price = data.get('underlying_price') or data.get('spot_price', 0)
            itm1_ce_price = self.extract_component_price(data)
            
            if itm1_ce_price and underlying_price:
                # Get ITM1 strike
                strike = self._get_itm1_strike(data, underlying_price)
                
                # Intrinsic value (significant for ITM)
                metrics['intrinsic_value'] = max(0, underlying_price - strike)
                
                # Time value
                metrics['time_value'] = itm1_ce_price - metrics['intrinsic_value']
                
                # ITM-specific ratios
                metrics['intrinsic_ratio'] = metrics['intrinsic_value'] / itm1_ce_price if itm1_ce_price > 0 else 0
                metrics['time_value_ratio'] = metrics['time_value'] / itm1_ce_price if itm1_ce_price > 0 else 0
                
                # Moneyness
                metrics['moneyness'] = underlying_price / strike if strike > 0 else 1.0
                metrics['itm_percentage'] = (underlying_price - strike) / strike * 100 if strike > 0 else 0
                
                # ITM efficiency
                metrics['itm_efficiency'] = self._calculate_itm_efficiency(
                    itm1_ce_price, metrics['intrinsic_value'], metrics['time_value']
                )
            
            # Greeks analysis (if available)
            greeks = self._extract_greeks(data)
            if greeks:
                metrics.update(greeks)
                
                # ITM-specific Greek analysis
                metrics['delta_regime'] = self._classify_delta_regime(greeks.get('delta', 0.7))
                metrics['gamma_regime'] = self._classify_gamma_regime(greeks.get('gamma', 0))
                metrics['theta_decay_rate'] = abs(greeks.get('theta', 0)) / itm1_ce_price if itm1_ce_price > 0 else 0
                
                # Delta leverage
                metrics['delta_leverage'] = greeks.get('delta', 0.7) * underlying_price / itm1_ce_price if itm1_ce_price > 0 else 0
            
            # Volume and liquidity
            metrics['volume'] = data.get('volume', 0)
            metrics['open_interest'] = data.get('open_interest', 0)
            metrics['liquidity_score'] = self._calculate_liquidity_score(data)
            
            # ITM premium analysis
            metrics['itm_premium'] = self._calculate_itm_premium(data, itm1_ce_price)
            
        except Exception as e:
            self.logger.error(f"Error calculating ITM1 Call metrics: {e}")
        
        return metrics
    
    def _get_itm1_strike(self, data: Dict[str, Any], underlying_price: float) -> float:
        """Get ITM1 strike price"""
        try:
            # Direct ITM1 strike if available
            if 'itm1_strike' in data:
                return float(data['itm1_strike'])
            
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
    
    def _extract_greeks(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract Greeks from market data"""
        greeks = {}
        
        # Common Greek keys for ITM calls
        greek_mappings = {
            'delta': ['delta', 'DELTA', 'Delta', 'itm1_ce_delta', 'ITM1_CE_DELTA'],
            'gamma': ['gamma', 'GAMMA', 'Gamma', 'itm1_ce_gamma', 'ITM1_CE_GAMMA'],
            'theta': ['theta', 'THETA', 'Theta', 'itm1_ce_theta', 'ITM1_CE_THETA'],
            'vega': ['vega', 'VEGA', 'Vega', 'itm1_ce_vega', 'ITM1_CE_VEGA'],
            'rho': ['rho', 'RHO', 'Rho', 'itm1_ce_rho', 'ITM1_CE_RHO']
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
    
    def _calculate_itm_efficiency(self, option_price: float, intrinsic: float, time_value: float) -> float:
        """Calculate ITM option efficiency"""
        try:
            if option_price <= 0:
                return 0.0
            
            # ITM efficiency based on intrinsic/time value balance
            intrinsic_ratio = intrinsic / option_price
            
            # Ideal ITM has high intrinsic ratio but some time value
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
    
    def _calculate_liquidity_score(self, data: Dict[str, Any]) -> float:
        """Calculate liquidity score for ITM option"""
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
    
    def _calculate_itm_premium(self, data: Dict[str, Any], option_price: float) -> Dict[str, float]:
        """Calculate ITM premium characteristics"""
        premium = {}
        
        try:
            underlying_price = data.get('underlying_price', 0)
            strike = self._get_itm1_strike(data, underlying_price)
            
            if underlying_price > 0 and strike > 0:
                # Premium over intrinsic
                intrinsic = max(0, underlying_price - strike)
                premium['over_intrinsic'] = option_price - intrinsic
                premium['over_intrinsic_pct'] = (premium['over_intrinsic'] / intrinsic * 100) if intrinsic > 0 else 0
                
                # Breakeven analysis
                premium['breakeven'] = strike + option_price
                premium['breakeven_distance'] = premium['breakeven'] - underlying_price
                premium['breakeven_pct'] = (premium['breakeven_distance'] / underlying_price * 100)
                
        except Exception:
            pass
        
        return premium
    
    def _classify_delta_regime(self, delta: float) -> str:
        """Classify delta regime for ITM1 Call"""
        if delta < self.delta_thresholds['low']:
            return 'LOW_DELTA'  # Moving toward ATM
        elif delta > self.delta_thresholds['high']:
            return 'HIGH_DELTA'  # Deep ITM
        else:
            return 'NORMAL_DELTA'  # Typical ITM1
    
    def _classify_gamma_regime(self, gamma: float) -> str:
        """Classify gamma regime for ITM1 Call"""
        abs_gamma = abs(gamma)
        if abs_gamma < self.gamma_thresholds['low']:
            return 'LOW_GAMMA'  # Deep ITM, low sensitivity
        elif abs_gamma > self.gamma_thresholds['high']:
            return 'HIGH_GAMMA'  # Near ATM, high sensitivity
        else:
            return 'NORMAL_GAMMA'
    
    def calculate_regime_contribution(self, analysis_result: ComponentAnalysisResult) -> Dict[str, float]:
        """
        Calculate ITM1 Call contribution to regime formation
        
        Args:
            analysis_result: Component analysis result
            
        Returns:
            Dictionary with regime indicators
        """
        regime_indicators = {}
        
        try:
            # Volatility contribution (ITM perspective)
            volatility_indicators = self._calculate_volatility_contribution(analysis_result)
            regime_indicators.update(volatility_indicators)
            
            # Trend contribution (ITM has strong directional bias)
            trend_indicators = self._calculate_trend_contribution(analysis_result)
            regime_indicators.update(trend_indicators)
            
            # Structure contribution
            structure_indicators = self._calculate_structure_contribution(analysis_result)
            regime_indicators.update(structure_indicators)
            
            # ITM-specific indicators
            itm_indicators = self._calculate_itm_specific_indicators(analysis_result)
            regime_indicators.update(itm_indicators)
            
            # Overall regime confidence
            regime_indicators['regime_confidence'] = self._calculate_regime_confidence(analysis_result)
            
        except Exception as e:
            self.logger.error(f"Error calculating regime contribution: {e}")
        
        return regime_indicators
    
    def _calculate_volatility_contribution(self, result: ComponentAnalysisResult) -> Dict[str, float]:
        """Calculate volatility regime contribution from ITM perspective"""
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
                
                indicators['decay_intensity'] = min(decay_rate * 50, 1.0)
            
            # Vega for volatility sensitivity
            if 'vega' in result.component_metrics:
                vega = result.component_metrics['vega']
                indicators['vega_exposure'] = min(abs(vega) / 50, 1.0)  # Normalize for ITM
            
        except Exception as e:
            self.logger.warning(f"Error calculating volatility contribution: {e}")
        
        return indicators
    
    def _calculate_trend_contribution(self, result: ComponentAnalysisResult) -> Dict[str, float]:
        """Calculate trend regime contribution from ITM perspective"""
        indicators = {}
        
        try:
            # ITM calls have strong bullish bias
            if 'delta' in result.component_metrics:
                delta = result.component_metrics['delta']
                
                # High delta ITM suggests strong bullish trend
                if delta > 0.8:
                    indicators['delta_trend_signal'] = 1.0  # Strong bullish
                elif delta < 0.6:
                    indicators['delta_trend_signal'] = -0.5  # Weakening bullish
                else:
                    indicators['delta_trend_signal'] = 0.5  # Moderate bullish
                
                indicators['directional_strength'] = delta
            
            # Price momentum
            price_change_pct = result.price_change_percent
            if abs(price_change_pct) > 0.1:
                # ITM calls move strongly with underlying
                if price_change_pct > 1:
                    indicators['price_momentum_signal'] = 1.0
                elif price_change_pct < -1:
                    indicators['price_momentum_signal'] = -1.0
                else:
                    indicators['price_momentum_signal'] = price_change_pct / 2
                
                indicators['momentum_strength'] = min(abs(price_change_pct) / 5, 1.0)
            
            # Delta leverage indicates trend strength
            if 'delta_leverage' in result.component_metrics:
                leverage = result.component_metrics['delta_leverage']
                indicators['leverage_factor'] = min(leverage / 10, 1.0)
            
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
                    indicators['liquidity_structure'] = 1.0  # Well-structured
                elif liquidity < 0.3:
                    indicators['liquidity_structure'] = -1.0  # Poor structure
                else:
                    indicators['liquidity_structure'] = 0.0
                
                indicators['market_depth'] = liquidity
            
            # Intrinsic ratio indicates structure efficiency
            if 'intrinsic_ratio' in result.component_metrics:
                intrinsic_ratio = result.component_metrics['intrinsic_ratio']
                
                # Very high intrinsic ratio suggests trending market
                if intrinsic_ratio > 0.8:
                    indicators['efficiency_structure'] = 1.0
                elif intrinsic_ratio < 0.4:
                    indicators['efficiency_structure'] = -1.0
                else:
                    indicators['efficiency_structure'] = 0.0
            
        except Exception as e:
            self.logger.warning(f"Error calculating structure contribution: {e}")
        
        return indicators
    
    def _calculate_itm_specific_indicators(self, result: ComponentAnalysisResult) -> Dict[str, float]:
        """Calculate ITM-specific regime indicators"""
        indicators = {}
        
        try:
            # ITM efficiency as regime indicator
            if 'itm_efficiency' in result.component_metrics:
                efficiency = result.component_metrics['itm_efficiency']
                indicators['itm_quality'] = efficiency
                
                if efficiency > 0.85:
                    indicators['itm_regime'] = 1.0  # Efficient ITM market
                elif efficiency < 0.6:
                    indicators['itm_regime'] = -1.0  # Inefficient ITM market
                else:
                    indicators['itm_regime'] = 0.0
            
            # Breakeven analysis
            if 'itm_premium' in result.component_metrics:
                premium_data = result.component_metrics['itm_premium']
                if isinstance(premium_data, dict):
                    breakeven_pct = premium_data.get('breakeven_pct', 0)
                    
                    # Lower breakeven distance suggests confident market
                    if abs(breakeven_pct) < 2:
                        indicators['breakeven_confidence'] = 1.0
                    elif abs(breakeven_pct) > 5:
                        indicators['breakeven_confidence'] = -1.0
                    else:
                        indicators['breakeven_confidence'] = 0.0
                    
                    indicators['breakeven_distance'] = min(abs(breakeven_pct) / 10, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Error calculating ITM-specific indicators: {e}")
        
        return indicators
    
    def _calculate_regime_confidence(self, result: ComponentAnalysisResult) -> float:
        """Calculate overall regime confidence for ITM component"""
        try:
            confidence_factors = []
            
            # Data availability confidence
            available_windows = sum(1 for window in self.rolling_windows 
                                  if window in result.rolling_metrics)
            confidence_factors.append(available_windows / len(self.rolling_windows))
            
            # Volume confidence
            if result.volume_ratio > 0.5:
                confidence_factors.append(min(result.volume_ratio, 1.0))
            
            # Delta confidence (ITM should have high delta)
            if 'delta' in result.component_metrics:
                delta = result.component_metrics['delta']
                if delta > 0.6:  # Proper ITM delta
                    confidence_factors.append(delta)
                else:
                    confidence_factors.append(0.3)  # Low confidence if delta too low
            
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