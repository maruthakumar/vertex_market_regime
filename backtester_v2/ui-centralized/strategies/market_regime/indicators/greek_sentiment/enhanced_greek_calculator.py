"""
Enhanced Greek Calculator with Second-Order Greeks
=================================================

Extends the base Greek calculator to include second-order Greeks:
- Vanna (∂²V/∂S∂σ) - Rate of change of delta with respect to volatility
- Volga/Vomma (∂²V/∂σ²) - Rate of change of vega with respect to volatility
- Charm (∂²V/∂S∂t) - Rate of change of delta with respect to time
- Color (∂³V/∂S²∂t) - Rate of change of gamma with respect to time
- Speed (∂³V/∂S³) - Rate of change of gamma with respect to spot
- Ultima (∂³V/∂σ³) - Third derivative with respect to volatility

Author: Market Regime Refactoring Team
Date: 2025-07-08
Version: 1.0.0
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from scipy.stats import norm

# Import base Greek calculator
from .greek_calculator import GreekCalculator

logger = logging.getLogger(__name__)

class EnhancedGreekCalculator(GreekCalculator):
    """
    Enhanced Greek calculator that includes second-order Greeks
    when enabled in configuration (enable_vanna: TRUE)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize enhanced Greek calculator"""
        super().__init__(config)
        
        # Second-order Greek configuration
        self.enable_vanna = self.config.get('enable_vanna', False)
        self.enable_volga = self.config.get('enable_volga', False)
        self.enable_charm = self.config.get('enable_charm', False)
        self.enable_color = self.config.get('enable_color', False)
        self.enable_speed = self.config.get('enable_speed', False)
        self.enable_ultima = self.config.get('enable_ultima', False)
        
        # Add normalization factors for second-order Greeks
        if self.enable_vanna:
            self.normalization_factors['vanna'] = {
                'method': 'divide',
                'factor': self.config.get('vanna_factor', 100.0),
                'description': 'Vanna normalization for NIFTY options'
            }
            self.validation_thresholds['vanna'] = {'min': -2.0, 'max': 2.0}
            self.normalization_stats['vanna'] = []
        
        if self.enable_volga:
            self.normalization_factors['volga'] = {
                'method': 'divide',
                'factor': self.config.get('volga_factor', 1000.0),
                'description': 'Volga normalization for NIFTY options'
            }
            self.validation_thresholds['volga'] = {'min': 0.0, 'max': 5.0}
            self.normalization_stats['volga'] = []
        
        logger.info(f"Enhanced Greek Calculator initialized - Vanna: {self.enable_vanna}")
    
    def calculate_second_order_greeks(self,
                                    spot: float,
                                    strike: float,
                                    time_to_expiry: float,
                                    volatility: float,
                                    risk_free_rate: float,
                                    dividend_yield: float = 0.0,
                                    option_type: str = 'call') -> Dict[str, float]:
        """
        Calculate second-order Greeks using Black-Scholes formulas
        
        Args:
            spot: Underlying spot price
            strike: Option strike price
            time_to_expiry: Time to expiry in years
            volatility: Implied volatility (annualized)
            risk_free_rate: Risk-free rate
            dividend_yield: Dividend yield
            option_type: 'call' or 'put'
            
        Returns:
            Dict containing second-order Greek values
        """
        try:
            second_order_greeks = {}
            
            # Basic calculations
            if time_to_expiry <= 0:
                # At expiry, all second-order Greeks are zero
                return {
                    'vanna': 0.0, 'volga': 0.0, 'charm': 0.0,
                    'color': 0.0, 'speed': 0.0, 'ultima': 0.0
                }
            
            # Black-Scholes d1 and d2
            sqrt_t = np.sqrt(time_to_expiry)
            d1 = (np.log(spot / strike) + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_expiry) / (volatility * sqrt_t)
            d2 = d1 - volatility * sqrt_t
            
            # Standard normal PDF and CDF
            n_d1 = norm.pdf(d1)
            n_d2 = norm.pdf(d2)
            N_d1 = norm.cdf(d1)
            N_d2 = norm.cdf(d2)
            
            # Discount factor
            discount = np.exp(-risk_free_rate * time_to_expiry)
            div_discount = np.exp(-dividend_yield * time_to_expiry)
            
            # Calculate Vanna (∂²V/∂S∂σ)
            if self.enable_vanna:
                # Vanna = -e^(-q*T) * n(d1) * d2 / σ
                vanna = -div_discount * n_d1 * d2 / volatility
                second_order_greeks['vanna'] = vanna
                logger.debug(f"Calculated Vanna: {vanna:.6f}")
            
            # Calculate Volga/Vomma (∂²V/∂σ²)
            if self.enable_volga:
                # Volga = vega * d1 * d2 / σ
                vega = spot * div_discount * n_d1 * sqrt_t
                volga = vega * d1 * d2 / volatility
                second_order_greeks['volga'] = volga
                logger.debug(f"Calculated Volga: {volga:.6f}")
            
            # Calculate Charm (∂²V/∂S∂t)
            if self.enable_charm:
                if option_type.lower() == 'call':
                    # Charm for call = -e^(-q*T) * n(d1) * (2*(r-q)*T - d2*σ*sqrt(T)) / (2*T*σ*sqrt(T))
                    charm = -div_discount * n_d1 * (2*(risk_free_rate - dividend_yield)*time_to_expiry - d2*volatility*sqrt_t) / (2*time_to_expiry*volatility*sqrt_t)
                else:
                    # Charm for put
                    charm = -div_discount * n_d1 * (2*(risk_free_rate - dividend_yield)*time_to_expiry - d2*volatility*sqrt_t) / (2*time_to_expiry*volatility*sqrt_t)
                    charm += risk_free_rate * strike * discount * N_d2
                
                second_order_greeks['charm'] = charm
                logger.debug(f"Calculated Charm: {charm:.6f}")
            
            # Calculate Color (∂³V/∂S²∂t)
            if self.enable_color:
                # Color = -e^(-q*T) * n(d1) / (2*S*T*σ*sqrt(T)) * 
                #         (2*q*T + 1 + d1*(2*(r-q)*T - d2*σ*sqrt(T))/(σ*sqrt(T)))
                term1 = 2 * dividend_yield * time_to_expiry + 1
                term2 = d1 * (2*(risk_free_rate - dividend_yield)*time_to_expiry - d2*volatility*sqrt_t) / (volatility*sqrt_t)
                color = -div_discount * n_d1 / (2*spot*time_to_expiry*volatility*sqrt_t) * (term1 + term2)
                second_order_greeks['color'] = color
                logger.debug(f"Calculated Color: {color:.6f}")
            
            # Calculate Speed (∂³V/∂S³)
            if self.enable_speed:
                # Speed = -gamma/S * (d1/(σ*sqrt(T)) + 1)
                gamma = div_discount * n_d1 / (spot * volatility * sqrt_t)
                speed = -gamma / spot * (d1 / (volatility * sqrt_t) + 1)
                second_order_greeks['speed'] = speed
                logger.debug(f"Calculated Speed: {speed:.6f}")
            
            # Calculate Ultima (∂³V/∂σ³)
            if self.enable_ultima:
                # Ultima = vega/σ² * (d1*d2*(1-d1*d2) + d1² + d2²)
                vega = spot * div_discount * n_d1 * sqrt_t
                ultima = vega / (volatility**2) * (d1*d2*(1 - d1*d2) + d1**2 + d2**2)
                second_order_greeks['ultima'] = ultima
                logger.debug(f"Calculated Ultima: {ultima:.6f}")
            
            return second_order_greeks
            
        except Exception as e:
            logger.error(f"Error calculating second-order Greeks: {e}")
            return {
                'vanna': 0.0, 'volga': 0.0, 'charm': 0.0,
                'color': 0.0, 'speed': 0.0, 'ultima': 0.0
            }
    
    def calculate_greek_contributions(self, 
                                    dte_adjusted_greeks: Dict[str, float],
                                    enable_validation: bool = True) -> Dict[str, float]:
        """
        Calculate Greek contributions including second-order Greeks
        
        Extends parent method to include second-order Greeks when enabled
        """
        # Get first-order Greek contributions
        contributions = super().calculate_greek_contributions(
            dte_adjusted_greeks, enable_validation
        )
        
        # Add second-order Greeks if enabled and present
        if self.enable_vanna and 'vanna' in dte_adjusted_greeks:
            vanna_contribution = self._normalize_greek(
                'vanna', dte_adjusted_greeks['vanna'], enable_validation
            )
            contributions['vanna'] = vanna_contribution
            logger.debug(f"Vanna contribution: {vanna_contribution:.4f}")
        
        if self.enable_volga and 'volga' in dte_adjusted_greeks:
            volga_contribution = self._normalize_greek(
                'volga', dte_adjusted_greeks['volga'], enable_validation
            )
            contributions['volga'] = volga_contribution
        
        # Add other second-order Greeks similarly
        for greek in ['charm', 'color', 'speed', 'ultima']:
            if self.config.get(f'enable_{greek}', False) and greek in dte_adjusted_greeks:
                contributions[greek] = self._normalize_greek(
                    greek, dte_adjusted_greeks[greek], enable_validation
                )
        
        return contributions
    
    def _normalize_greek(self, greek: str, value: float, 
                        enable_validation: bool = True) -> float:
        """
        Normalize a Greek value using configured factors
        
        Args:
            greek: Greek name
            value: Raw Greek value
            enable_validation: Whether to validate the value
            
        Returns:
            Normalized Greek value
        """
        if greek in self.normalization_factors:
            norm_config = self.normalization_factors[greek]
            
            # Apply normalization based on method
            if norm_config['method'] == 'direct':
                normalized_value = np.clip(value, -1.0, 1.0)
            elif norm_config['method'] == 'scale':
                normalized_value = np.clip(value * norm_config['factor'], -1.0, 1.0)
            elif norm_config['method'] == 'divide':
                normalized_value = np.clip(value / norm_config['factor'], -1.0, 1.0)
            else:
                normalized_value = np.clip(value, -1.0, 1.0)
            
            # Validation if enabled
            if enable_validation:
                normalized_value = self._validate_greek_value(greek, normalized_value)
            
            return normalized_value
        else:
            # Unknown Greek - direct clipping
            return np.clip(value, -1.0, 1.0)
    
    def calculate_enhanced_sentiment(self,
                                   market_data: pd.DataFrame,
                                   config: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Calculate enhanced Greek sentiment including second-order Greeks
        
        Args:
            market_data: DataFrame with options data
            config: Optional configuration overrides
            
        Returns:
            Dict with sentiment scores including second-order Greeks
        """
        try:
            results = {
                'sentiment_score': 0.0,
                'first_order_score': 0.0,
                'second_order_score': 0.0,
                'components': {}
            }
            
            # Calculate first-order Greeks sentiment
            first_order_greeks = ['delta', 'gamma', 'theta', 'vega']
            first_order_total = 0.0
            first_order_weights = 0.0
            
            for greek in first_order_greeks:
                if greek in market_data.columns:
                    weight = self.config.get(f'{greek}_weight', 0.25)
                    value = market_data[greek].mean()
                    normalized = self._normalize_greek(greek, value)
                    
                    first_order_total += normalized * weight
                    first_order_weights += weight
                    results['components'][greek] = normalized
            
            if first_order_weights > 0:
                results['first_order_score'] = first_order_total / first_order_weights
            
            # Calculate second-order Greeks sentiment if enabled
            if self.enable_vanna or self.enable_volga:
                second_order_total = 0.0
                second_order_weights = 0.0
                
                # Calculate Vanna contribution
                if self.enable_vanna:
                    # Simulate vanna calculation (in production, use actual formulas)
                    spot = market_data['underlying_close'].iloc[-1]
                    for _, row in market_data.iterrows():
                        vanna = self.calculate_second_order_greeks(
                            spot=spot,
                            strike=row['strike'],
                            time_to_expiry=row.get('dte', 1) / 365,
                            volatility=row.get('iv', 0.15),
                            risk_free_rate=0.065,
                            option_type=row['option_type']
                        )['vanna']
                        
                        # Aggregate vanna values
                        # This is simplified - in production, weight by OI, volume, etc.
                    
                    vanna_weight = self.config.get('vanna_weight', 0.1)
                    vanna_value = 0.0  # Placeholder - calculate from aggregated values
                    normalized_vanna = self._normalize_greek('vanna', vanna_value)
                    
                    second_order_total += normalized_vanna * vanna_weight
                    second_order_weights += vanna_weight
                    results['components']['vanna'] = normalized_vanna
                
                if second_order_weights > 0:
                    results['second_order_score'] = second_order_total / second_order_weights
            
            # Combine first and second order scores
            if self.enable_vanna or self.enable_volga:
                # Weight first-order more heavily
                first_weight = 0.8
                second_weight = 0.2
                results['sentiment_score'] = (
                    results['first_order_score'] * first_weight +
                    results['second_order_score'] * second_weight
                )
            else:
                results['sentiment_score'] = results['first_order_score']
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating enhanced sentiment: {e}")
            return {
                'sentiment_score': 0.0,
                'first_order_score': 0.0,
                'second_order_score': 0.0,
                'components': {}
            }
    
    def get_enhanced_calculation_summary(self) -> Dict[str, Any]:
        """Get summary including second-order Greeks statistics"""
        summary = super().get_calculation_summary()
        
        # Add second-order Greek statistics
        second_order_stats = {}
        
        for greek in ['vanna', 'volga', 'charm', 'color', 'speed', 'ultima']:
            if greek in self.normalization_stats and self.normalization_stats[greek]:
                recent_stats = self.normalization_stats[greek][-20:]
                values = [stat['normalized_value'] for stat in recent_stats]
                
                second_order_stats[greek] = {
                    'enabled': self.config.get(f'enable_{greek}', False),
                    'avg_value': np.mean(values) if values else 0.0,
                    'std_value': np.std(values) if values else 0.0,
                    'calculations': len(values)
                }
        
        summary['second_order_greeks'] = second_order_stats
        summary['vanna_enabled'] = self.enable_vanna
        
        return summary