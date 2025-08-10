"""
ATM Call (CE) Component Analyzer

Analyzes ATM Call options with rolling analysis across [3,5,10,15] minute windows.
Provides comprehensive technical analysis, performance tracking, and regime contribution
for the ATM Call component of the triple straddle system.
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


class ATMCallAnalyzer(BaseComponentAnalyzer):
    """
    ATM Call (CE) Component Analyzer
    
    Specialized analyzer for ATM Call options with:
    - Strike selection based on underlying price
    - Delta-based sensitivity analysis
    - Call-specific Greeks analysis
    - ATM-specific regime indicators
    """
    
    def __init__(self, 
                 config: StraddleConfig,
                 calculation_engine: CalculationEngine,
                 window_manager: RollingWindowManager):
        """
        Initialize ATM Call analyzer
        
        Args:
            config: Straddle configuration
            calculation_engine: Shared calculation engine
            window_manager: Rolling window manager
        """
        super().__init__('atm_ce', config, calculation_engine, window_manager)
        
        # ATM-specific configuration
        self.atm_tolerance = 0.02  # 2% tolerance for ATM strike selection
        self.delta_target = 0.5    # Target delta for ATM calls
        
        # Greeks thresholds for ATM calls
        self.delta_thresholds = {'low': 0.3, 'high': 0.7}
        self.gamma_thresholds = {'low': 0.01, 'high': 0.05}
        self.theta_thresholds = {'low': -50, 'high': -10}
        self.vega_thresholds = {'low': 10, 'high': 100}
        
        self.logger.info("ATM Call analyzer initialized")
    
    def extract_component_price(self, data: Dict[str, Any]) -> Optional[float]:
        """
        Extract ATM Call price from market data
        
        Args:
            data: Market data dictionary containing option chain
            
        Returns:
            ATM Call price or None if not available
        """
        try:
            # Get underlying price for ATM strike selection
            underlying_price = data.get('underlying_price') or data.get('spot_price')
            if underlying_price is None:
                return None
            
            # Look for ATM Call price in different possible keys
            atm_ce_keys = [
                'ATM_CE', 'atm_ce', 'ATM_CALL', 'atm_call',
                'atm_ce_price', 'ATM_CE_LTP', 'atm_ce_ltp'
            ]
            
            for key in atm_ce_keys:
                if key in data and data[key] is not None:
                    price = float(data[key])
                    if price > 0:  # Valid option price
                        return price
            
            # If direct ATM price not available, find closest strike
            if 'option_chain' in data:
                return self._find_atm_call_from_chain(data['option_chain'], underlying_price)
            
            # Try alternative data structures
            if 'strikes' in data and 'ce_prices' in data:
                return self._find_atm_from_strikes_data(data, underlying_price, 'CE')
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting ATM Call price: {e}")
            return None
    
    def _find_atm_call_from_chain(self, option_chain: List[Dict], underlying_price: float) -> Optional[float]:
        """Find ATM Call from option chain data"""
        try:
            # Find strike closest to underlying price
            ce_options = [opt for opt in option_chain if opt.get('option_type') == 'CE']
            
            if not ce_options:
                return None
            
            # Find closest strike
            closest_option = min(
                ce_options,
                key=lambda x: abs(x.get('strike', 0) - underlying_price)
            )
            
            # Check if within ATM tolerance
            strike_diff = abs(closest_option.get('strike', 0) - underlying_price)
            if strike_diff / underlying_price <= self.atm_tolerance:
                return closest_option.get('ltp') or closest_option.get('price')
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding ATM Call from chain: {e}")
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
        Calculate ATM Call specific metrics
        
        Args:
            data: Market data dictionary
            
        Returns:
            Dictionary with ATM Call metrics
        """
        metrics = {}
        
        try:
            # Basic ATM Call metrics
            underlying_price = data.get('underlying_price') or data.get('spot_price', 0)
            atm_ce_price = self.extract_component_price(data)
            
            if atm_ce_price and underlying_price:
                # Intrinsic value (always 0 for ATM at inception)
                metrics['intrinsic_value'] = max(0, underlying_price - data.get('strike', underlying_price))
                
                # Time value
                metrics['time_value'] = atm_ce_price - metrics['intrinsic_value']
                
                # Moneyness
                strike = data.get('strike', underlying_price)
                metrics['moneyness'] = underlying_price / strike if strike > 0 else 1.0
                metrics['atm_deviation'] = abs(metrics['moneyness'] - 1.0)
            
            # Greeks analysis (if available)
            greeks = self._extract_greeks(data)
            if greeks:
                metrics.update(greeks)
                
                # Greeks-based indicators
                metrics['delta_regime'] = self._classify_delta_regime(greeks.get('delta', 0.5))
                metrics['gamma_regime'] = self._classify_gamma_regime(greeks.get('gamma', 0))
                metrics['theta_exposure'] = abs(greeks.get('theta', 0))
                metrics['vega_sensitivity'] = greeks.get('vega', 0)
            
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
                metrics['price_vs_mid'] = (atm_ce_price - metrics['mid_price']) / metrics['mid_price'] * 100
            
            # ATM-specific regime indicators
            metrics['atm_efficiency'] = self._calculate_atm_efficiency(data, atm_ce_price)
            metrics['relative_value'] = self._calculate_relative_value(data, atm_ce_price)
            
        except Exception as e:
            self.logger.error(f"Error calculating ATM Call metrics: {e}")
        
        return metrics
    
    def _extract_greeks(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract Greeks from market data"""
        greeks = {}
        
        # Common Greek keys
        greek_mappings = {
            'delta': ['delta', 'DELTA', 'Delta', 'atm_ce_delta'],
            'gamma': ['gamma', 'GAMMA', 'Gamma', 'atm_ce_gamma'],
            'theta': ['theta', 'THETA', 'Theta', 'atm_ce_theta'],
            'vega': ['vega', 'VEGA', 'Vega', 'atm_ce_vega'],
            'rho': ['rho', 'RHO', 'Rho', 'atm_ce_rho']
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
        """Classify delta regime for ATM Call"""
        if delta < self.delta_thresholds['low']:
            return 'LOW_DELTA'
        elif delta > self.delta_thresholds['high']:
            return 'HIGH_DELTA'
        else:
            return 'NORMAL_DELTA'
    
    def _classify_gamma_regime(self, gamma: float) -> str:
        """Classify gamma regime for ATM Call"""
        abs_gamma = abs(gamma)
        if abs_gamma < self.gamma_thresholds['low']:
            return 'LOW_GAMMA'
        elif abs_gamma > self.gamma_thresholds['high']:
            return 'HIGH_GAMMA'
        else:
            return 'NORMAL_GAMMA'
    
    def _calculate_atm_efficiency(self, data: Dict[str, Any], atm_price: float) -> float:
        """Calculate ATM pricing efficiency"""
        try:
            # Simple efficiency based on bid-ask spread and volume
            bid = data.get('bid', 0)
            ask = data.get('ask', 0)
            volume = data.get('volume', 0)
            
            if bid > 0 and ask > 0 and atm_price > 0:
                spread_efficiency = 1 - ((ask - bid) / atm_price)
                volume_efficiency = min(volume / 1000, 1.0)  # Normalize volume
                return (spread_efficiency + volume_efficiency) / 2
            
            return 0.5  # Neutral efficiency
            
        except Exception:
            return 0.5
    
    def _calculate_relative_value(self, data: Dict[str, Any], atm_price: float) -> float:
        """Calculate relative value compared to theoretical value"""
        try:
            # Compare to simple Black-Scholes estimate if IV available
            iv = data.get('implied_volatility') or data.get('iv')
            if iv and atm_price:
                # Simplified relative value calculation
                # In practice, this would use a proper option pricing model
                theo_premium = data.get('theoretical_value', atm_price)
                if theo_premium > 0:
                    return atm_price / theo_premium
            
            return 1.0  # Neutral relative value
            
        except Exception:
            return 1.0
    
    def calculate_regime_contribution(self, analysis_result: ComponentAnalysisResult) -> Dict[str, float]:
        """
        Calculate ATM Call contribution to regime formation
        
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
            
            # Trend contribution
            trend_indicators = self._calculate_trend_contribution(analysis_result)
            regime_indicators.update(trend_indicators)
            
            # Structure contribution
            structure_indicators = self._calculate_structure_contribution(analysis_result)
            regime_indicators.update(structure_indicators)
            
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
            
            # Greeks-based volatility
            if 'vega' in result.component_metrics:
                vega = result.component_metrics['vega']
                indicators['vega_exposure'] = min(abs(vega) / 100, 1.0)  # Normalize
            
        except Exception as e:
            self.logger.warning(f"Error calculating volatility contribution: {e}")
        
        return indicators
    
    def _calculate_trend_contribution(self, result: ComponentAnalysisResult) -> Dict[str, float]:
        """Calculate trend regime contribution"""
        indicators = {}
        
        try:
            # Use EMA trend from multiple windows
            ema_signals = []
            
            for window in ['3min', '5min', '10min']:
                if window in result.rolling_metrics:
                    window_metrics = result.rolling_metrics[window]
                    
                    # Compare EMA 20 vs current price
                    ema_20_key = f'ema_20_{window.replace("min", "")}min'
                    if ema_20_key in result.technical_indicators:
                        ema_20 = result.technical_indicators[ema_20_key]
                        price_vs_ema = (result.current_price - ema_20) / ema_20
                        ema_signals.append(price_vs_ema)
            
            if ema_signals:
                avg_ema_signal = np.mean(ema_signals)
                
                if avg_ema_signal > 0.02:
                    indicators['trend_regime'] = 1.0  # Bullish
                elif avg_ema_signal < -0.02:
                    indicators['trend_regime'] = -1.0  # Bearish
                else:
                    indicators['trend_regime'] = 0.0  # Neutral
                
                indicators['trend_strength'] = abs(avg_ema_signal)
            
        except Exception as e:
            self.logger.warning(f"Error calculating trend contribution: {e}")
        
        return indicators
    
    def _calculate_structure_contribution(self, result: ComponentAnalysisResult) -> Dict[str, float]:
        """Calculate structure regime contribution"""
        indicators = {}
        
        try:
            # Use volume and bid-ask metrics for structure
            volume_ratio = result.volume_ratio
            
            # High volume suggests trending, low volume suggests ranging
            if volume_ratio > 1.5:
                indicators['structure_regime'] = 1.0  # Trending
            elif volume_ratio < 0.7:
                indicators['structure_regime'] = -1.0  # Ranging
            else:
                indicators['structure_regime'] = 0.0  # Neutral
            
            indicators['volume_signal'] = min(volume_ratio / 2, 1.0)  # Normalize
            
            # Add bid-ask spread contribution if available
            if 'bid_ask_spread_pct' in result.component_metrics:
                spread_pct = result.component_metrics['bid_ask_spread_pct']
                # Wide spreads suggest ranging market
                if spread_pct > 5:
                    indicators['structure_regime'] -= 0.3
                elif spread_pct < 1:
                    indicators['structure_regime'] += 0.3
                
                indicators['liquidity_signal'] = max(0, 1 - spread_pct / 10)
            
        except Exception as e:
            self.logger.warning(f"Error calculating structure contribution: {e}")
        
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
                delta_confidence = 1 - abs(delta - 0.5) * 2  # Higher confidence closer to 0.5
                confidence_factors.append(max(delta_confidence, 0.1))
            
            return np.mean(confidence_factors) if confidence_factors else 0.5
            
        except Exception:
            return 0.5