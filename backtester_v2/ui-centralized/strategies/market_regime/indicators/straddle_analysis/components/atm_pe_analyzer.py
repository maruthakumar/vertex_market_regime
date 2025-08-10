"""
ATM Put (PE) Component Analyzer

Analyzes ATM Put options with rolling analysis across [3,5,10,15] minute windows.
Provides comprehensive technical analysis, performance tracking, and regime contribution
for the ATM Put component of the triple straddle system.
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


class ATMPutAnalyzer(BaseComponentAnalyzer):
    """
    ATM Put (PE) Component Analyzer
    
    Specialized analyzer for ATM Put options with:
    - Strike selection based on underlying price
    - Put-call parity analysis
    - Put-specific Greeks analysis
    - ATM Put regime indicators
    """
    
    def __init__(self, 
                 config: StraddleConfig,
                 calculation_engine: CalculationEngine,
                 window_manager: RollingWindowManager):
        """
        Initialize ATM Put analyzer
        
        Args:
            config: Straddle configuration
            calculation_engine: Shared calculation engine
            window_manager: Rolling window manager
        """
        super().__init__('atm_pe', config, calculation_engine, window_manager)
        
        # ATM-specific configuration
        self.atm_tolerance = 0.02  # 2% tolerance for ATM strike selection
        self.delta_target = -0.5   # Target delta for ATM puts
        
        # Greeks thresholds for ATM puts
        self.delta_thresholds = {'low': -0.7, 'high': -0.3}
        self.gamma_thresholds = {'low': 0.01, 'high': 0.05}
        self.theta_thresholds = {'low': -50, 'high': -10}
        self.vega_thresholds = {'low': 10, 'high': 100}
        
        self.logger.info("ATM Put analyzer initialized")
    
    def extract_component_price(self, data: Dict[str, Any]) -> Optional[float]:
        """
        Extract ATM Put price from market data
        
        Args:
            data: Market data dictionary containing option chain
            
        Returns:
            ATM Put price or None if not available
        """
        try:
            # Get underlying price for ATM strike selection
            underlying_price = data.get('underlying_price') or data.get('spot_price')
            if underlying_price is None:
                return None
            
            # Look for ATM Put price in different possible keys
            atm_pe_keys = [
                'ATM_PE', 'atm_pe', 'ATM_PUT', 'atm_put',
                'atm_pe_price', 'ATM_PE_LTP', 'atm_pe_ltp'
            ]
            
            for key in atm_pe_keys:
                if key in data and data[key] is not None:
                    price = float(data[key])
                    if price > 0:  # Valid option price
                        return price
            
            # If direct ATM price not available, find closest strike
            if 'option_chain' in data:
                return self._find_atm_put_from_chain(data['option_chain'], underlying_price)
            
            # Try alternative data structures
            if 'strikes' in data and 'pe_prices' in data:
                return self._find_atm_from_strikes_data(data, underlying_price, 'PE')
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting ATM Put price: {e}")
            return None
    
    def _find_atm_put_from_chain(self, option_chain: List[Dict], underlying_price: float) -> Optional[float]:
        """Find ATM Put from option chain data"""
        try:
            # Find strike closest to underlying price
            pe_options = [opt for opt in option_chain if opt.get('option_type') == 'PE']
            
            if not pe_options:
                return None
            
            # Find closest strike
            closest_option = min(
                pe_options,
                key=lambda x: abs(x.get('strike', 0) - underlying_price)
            )
            
            # Check if within ATM tolerance
            strike_diff = abs(closest_option.get('strike', 0) - underlying_price)
            if strike_diff / underlying_price <= self.atm_tolerance:
                return closest_option.get('ltp') or closest_option.get('price')
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding ATM Put from chain: {e}")
            return None
    
    def _find_atm_from_strikes_data(self, data: Dict, underlying_price: float, option_type: str) -> Optional[float]:
        """Find ATM option from strikes and prices data"""
        try:
            strikes = data.get('strikes', [])
            prices_key = 'ce_prices' if option_type == 'CE' else 'pe_prices'
            prices = data.get(prices_key, [])
            
            if len(strikes) != len(prices):
                return None
            
            # Find closest strike index
            strike_diffs = [abs(strike - underlying_price) for strike in strikes]
            min_diff_idx = strike_diffs.index(min(strike_diffs))
            
            # Check ATM tolerance
            if strike_diffs[min_diff_idx] / underlying_price <= self.atm_tolerance:
                return prices[min_diff_idx]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding ATM from strikes data: {e}")
            return None
    
    def calculate_component_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate ATM Put specific metrics
        
        Args:
            data: Market data dictionary
            
        Returns:
            Dictionary with ATM Put metrics
        """
        metrics = {}
        
        try:
            # Basic ATM Put metrics
            underlying_price = data.get('underlying_price') or data.get('spot_price', 0)
            atm_pe_price = self.extract_component_price(data)
            
            if atm_pe_price and underlying_price:
                # Intrinsic value
                strike = data.get('strike', underlying_price)
                metrics['intrinsic_value'] = max(0, strike - underlying_price)
                
                # Time value
                metrics['time_value'] = atm_pe_price - metrics['intrinsic_value']
                
                # Moneyness
                metrics['moneyness'] = underlying_price / strike if strike > 0 else 1.0
                metrics['atm_deviation'] = abs(metrics['moneyness'] - 1.0)
                
                # Put-call parity analysis
                atm_ce_price = self._get_companion_call_price(data)
                if atm_ce_price:
                    metrics['put_call_parity'] = self._calculate_put_call_parity(
                        atm_ce_price, atm_pe_price, underlying_price, strike
                    )
            
            # Greeks analysis (if available)
            greeks = self._extract_greeks(data)
            if greeks:
                metrics.update(greeks)
                
                # Greeks-based indicators (Put-specific)
                metrics['delta_regime'] = self._classify_delta_regime(greeks.get('delta', -0.5))
                metrics['gamma_regime'] = self._classify_gamma_regime(greeks.get('gamma', 0))
                metrics['theta_exposure'] = abs(greeks.get('theta', 0))
                metrics['vega_sensitivity'] = greeks.get('vega', 0)
                
                # Put-specific delta analysis
                metrics['put_delta_strength'] = abs(greeks.get('delta', -0.5))
            
            # Volume and open interest
            metrics['volume'] = data.get('volume', 0)
            metrics['open_interest'] = data.get('open_interest', 0)
            metrics['volume_oi_ratio'] = (
                metrics['volume'] / metrics['open_interest'] 
                if metrics['open_interest'] > 0 else 0
            )
            
            # Bid-ask spread analysis
            bid = data.get('bid', 0)
            ask = data.get('ask', 0)
            if bid > 0 and ask > 0:
                metrics['bid_ask_spread'] = ask - bid
                metrics['bid_ask_spread_pct'] = (ask - bid) / ((ask + bid) / 2) * 100
                metrics['mid_price'] = (ask + bid) / 2
                metrics['price_vs_mid'] = (atm_pe_price - metrics['mid_price']) / metrics['mid_price'] * 100
            
            # Put-specific indicators
            metrics['put_premium_ratio'] = self._calculate_put_premium_ratio(atm_pe_price, underlying_price)
            metrics['downside_protection'] = self._calculate_downside_protection(data, atm_pe_price)
            
        except Exception as e:
            self.logger.error(f"Error calculating ATM Put metrics: {e}")
        
        return metrics
    
    def _get_companion_call_price(self, data: Dict[str, Any]) -> Optional[float]:
        """Get corresponding ATM Call price for put-call parity"""
        atm_ce_keys = [
            'ATM_CE', 'atm_ce', 'ATM_CALL', 'atm_call',
            'atm_ce_price', 'ATM_CE_LTP', 'atm_ce_ltp'
        ]
        
        for key in atm_ce_keys:
            if key in data and data[key] is not None:
                try:
                    return float(data[key])
                except (ValueError, TypeError):
                    continue
        return None
    
    def _calculate_put_call_parity(self, call_price: float, put_price: float, 
                                  spot_price: float, strike: float) -> Dict[str, float]:
        """Calculate put-call parity analysis"""
        try:
            # Simplified put-call parity (assuming zero interest rate and no dividends)
            # C - P = S - K
            theoretical_diff = spot_price - strike
            actual_diff = call_price - put_price
            parity_deviation = actual_diff - theoretical_diff
            
            return {
                'theoretical_diff': theoretical_diff,
                'actual_diff': actual_diff,
                'parity_deviation': parity_deviation,
                'parity_ratio': actual_diff / theoretical_diff if theoretical_diff != 0 else 1.0
            }
        except Exception:
            return {'parity_deviation': 0.0, 'parity_ratio': 1.0}
    
    def _extract_greeks(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract Greeks from market data"""
        greeks = {}
        
        # Common Greek keys for puts
        greek_mappings = {
            'delta': ['delta', 'DELTA', 'Delta', 'atm_pe_delta'],
            'gamma': ['gamma', 'GAMMA', 'Gamma', 'atm_pe_gamma'],
            'theta': ['theta', 'THETA', 'Theta', 'atm_pe_theta'],
            'vega': ['vega', 'VEGA', 'Vega', 'atm_pe_vega'],
            'rho': ['rho', 'RHO', 'Rho', 'atm_pe_rho']
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
    
    def _classify_delta_regime(self, delta: float) -> str:
        """Classify delta regime for ATM Put (negative delta)"""
        if delta > self.delta_thresholds['high']:  # Less negative
            return 'LOW_DELTA'
        elif delta < self.delta_thresholds['low']:  # More negative
            return 'HIGH_DELTA'
        else:
            return 'NORMAL_DELTA'
    
    def _classify_gamma_regime(self, gamma: float) -> str:
        """Classify gamma regime for ATM Put"""
        abs_gamma = abs(gamma)
        if abs_gamma < self.gamma_thresholds['low']:
            return 'LOW_GAMMA'
        elif abs_gamma > self.gamma_thresholds['high']:
            return 'HIGH_GAMMA'
        else:
            return 'NORMAL_GAMMA'
    
    def _calculate_put_premium_ratio(self, put_price: float, underlying_price: float) -> float:
        """Calculate put premium as ratio of underlying price"""
        try:
            if underlying_price > 0:
                return put_price / underlying_price
            return 0.0
        except Exception:
            return 0.0
    
    def _calculate_downside_protection(self, data: Dict[str, Any], put_price: float) -> float:
        """Calculate downside protection percentage"""
        try:
            underlying_price = data.get('underlying_price') or data.get('spot_price', 0)
            strike = data.get('strike', underlying_price)
            
            if underlying_price > 0:
                # Maximum protection = strike - current_price + put_premium
                max_protection = strike - underlying_price + put_price
                protection_percentage = max_protection / underlying_price * 100
                return max(protection_percentage, 0)
            
            return 0.0
        except Exception:
            return 0.0
    
    def calculate_regime_contribution(self, analysis_result: ComponentAnalysisResult) -> Dict[str, float]:
        """
        Calculate ATM Put contribution to regime formation
        
        Args:
            analysis_result: Component analysis result
            
        Returns:
            Dictionary with regime indicators
        """
        regime_indicators = {}
        
        try:
            # Volatility contribution
            volatility_indicators = self._calculate_volatility_contribution(analysis_result)
            regime_indicators.update(volatility_indicators)
            
            # Trend contribution (put-specific)
            trend_indicators = self._calculate_trend_contribution(analysis_result)
            regime_indicators.update(trend_indicators)
            
            # Structure contribution
            structure_indicators = self._calculate_structure_contribution(analysis_result)
            regime_indicators.update(structure_indicators)
            
            # Put-specific regime signals
            put_indicators = self._calculate_put_specific_indicators(analysis_result)
            regime_indicators.update(put_indicators)
            
            # Overall regime confidence
            regime_indicators['regime_confidence'] = self._calculate_regime_confidence(analysis_result)
            
        except Exception as e:
            self.logger.error(f"Error calculating regime contribution: {e}")
        
        return regime_indicators
    
    def _calculate_volatility_contribution(self, result: ComponentAnalysisResult) -> Dict[str, float]:
        """Calculate volatility regime contribution"""
        indicators = {}
        
        try:
            # Use price volatility from shortest window
            vol_key = f'volatility_3min'
            if vol_key in result.rolling_metrics.get('3min', {}):
                volatility = result.rolling_metrics['3min'][vol_key]
                
                # Classify volatility regime
                if volatility > 0.03:
                    indicators['volatility_regime'] = 1.0  # High volatility
                elif volatility < 0.01:
                    indicators['volatility_regime'] = -1.0  # Low volatility
                else:
                    indicators['volatility_regime'] = 0.0  # Medium volatility
                
                indicators['volatility_value'] = volatility
            
            # Vega-based volatility exposure
            if 'vega' in result.component_metrics:
                vega = result.component_metrics['vega']
                indicators['vega_exposure'] = min(abs(vega) / 100, 1.0)  # Normalize
            
        except Exception as e:
            self.logger.warning(f"Error calculating volatility contribution: {e}")
        
        return indicators
    
    def _calculate_trend_contribution(self, result: ComponentAnalysisResult) -> Dict[str, float]:
        """Calculate trend regime contribution (Put perspective)"""
        indicators = {}
        
        try:
            # Put delta analysis for trend
            if 'delta' in result.component_metrics:
                delta = result.component_metrics['delta']
                
                # More negative delta suggests stronger bearish trend
                if delta < -0.6:
                    indicators['put_trend_signal'] = -1.0  # Strong bearish
                elif delta > -0.4:
                    indicators['put_trend_signal'] = 1.0   # Weak bearish/bullish
                else:
                    indicators['put_trend_signal'] = 0.0   # Neutral
                
                indicators['delta_strength'] = abs(delta)
            
            # Price trend analysis
            price_change_pct = result.price_change_percent
            if abs(price_change_pct) > 0.1:
                # Put prices increase when underlying falls
                if price_change_pct > 2:  # Put price increasing
                    indicators['price_trend_signal'] = -1.0  # Bearish underlying
                elif price_change_pct < -2:  # Put price decreasing
                    indicators['price_trend_signal'] = 1.0   # Bullish underlying
                else:
                    indicators['price_trend_signal'] = 0.0
                
                indicators['price_momentum'] = abs(price_change_pct) / 10  # Normalize
            
        except Exception as e:
            self.logger.warning(f"Error calculating trend contribution: {e}")
        
        return indicators
    
    def _calculate_structure_contribution(self, result: ComponentAnalysisResult) -> Dict[str, float]:
        """Calculate structure regime contribution"""
        indicators = {}
        
        try:
            # Put-call parity deviation indicates market structure
            if 'put_call_parity' in result.component_metrics:
                parity_data = result.component_metrics['put_call_parity']
                if isinstance(parity_data, dict):
                    parity_deviation = parity_data.get('parity_deviation', 0)
                    
                    # Large deviations suggest market stress/ranging
                    if abs(parity_deviation) > 5:
                        indicators['structure_regime'] = -1.0  # Ranging/stressed
                    elif abs(parity_deviation) < 1:
                        indicators['structure_regime'] = 1.0   # Trending/efficient
                    else:
                        indicators['structure_regime'] = 0.0
                    
                    indicators['parity_efficiency'] = max(0, 1 - abs(parity_deviation) / 10)
            
            # Volume analysis
            volume_ratio = result.volume_ratio
            if volume_ratio > 1.5:
                indicators['volume_structure'] = 1.0  # High activity
            elif volume_ratio < 0.7:
                indicators['volume_structure'] = -1.0  # Low activity
            else:
                indicators['volume_structure'] = 0.0
            
        except Exception as e:
            self.logger.warning(f"Error calculating structure contribution: {e}")
        
        return indicators
    
    def _calculate_put_specific_indicators(self, result: ComponentAnalysisResult) -> Dict[str, float]:
        """Calculate put-specific regime indicators"""
        indicators = {}
        
        try:
            # Downside protection analysis
            if 'downside_protection' in result.component_metrics:
                protection = result.component_metrics['downside_protection']
                
                if protection > 10:
                    indicators['protection_regime'] = 1.0  # High protection
                elif protection < 2:
                    indicators['protection_regime'] = -1.0  # Low protection
                else:
                    indicators['protection_regime'] = 0.0
                
                indicators['protection_level'] = min(protection / 20, 1.0)  # Normalize
            
            # Put premium ratio analysis
            if 'put_premium_ratio' in result.component_metrics:
                premium_ratio = result.component_metrics['put_premium_ratio']
                
                # High premium ratios suggest fear/volatility
                if premium_ratio > 0.05:
                    indicators['fear_indicator'] = 1.0
                elif premium_ratio < 0.02:
                    indicators['fear_indicator'] = -1.0
                else:
                    indicators['fear_indicator'] = 0.0
                
                indicators['premium_level'] = min(premium_ratio * 50, 1.0)  # Normalize
            
        except Exception as e:
            self.logger.warning(f"Error calculating put-specific indicators: {e}")
        
        return indicators
    
    def _calculate_regime_confidence(self, result: ComponentAnalysisResult) -> float:
        """Calculate overall regime confidence"""
        try:
            confidence_factors = []
            
            # Data availability confidence
            available_windows = sum(1 for window in self.rolling_windows 
                                  if window in result.rolling_metrics)
            confidence_factors.append(available_windows / len(self.rolling_windows))
            
            # Volume confidence
            if result.volume_ratio > 0.5:
                confidence_factors.append(min(result.volume_ratio, 1.0))
            
            # Price movement confidence
            if abs(result.price_change_percent) > 0.1:
                confidence_factors.append(min(abs(result.price_change_percent) / 5, 1.0))
            
            # Greeks confidence (if available)
            if 'delta' in result.component_metrics:
                delta = result.component_metrics['delta']
                delta_confidence = 1 - abs(abs(delta) - 0.5) * 2  # Higher confidence closer to -0.5
                confidence_factors.append(max(delta_confidence, 0.1))
            
            # Put-call parity confidence
            if 'put_call_parity' in result.component_metrics:
                parity_data = result.component_metrics['put_call_parity']
                if isinstance(parity_data, dict):
                    parity_deviation = abs(parity_data.get('parity_deviation', 0))
                    parity_confidence = max(0.1, 1 - parity_deviation / 10)
                    confidence_factors.append(parity_confidence)
            
            return np.mean(confidence_factors) if confidence_factors else 0.5
            
        except Exception:
            return 0.5