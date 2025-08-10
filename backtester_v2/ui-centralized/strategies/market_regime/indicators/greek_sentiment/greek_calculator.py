"""
Greek Calculator - Core Greek Calculation Engine
===============================================

Handles core Greek calculations with market-calibrated normalization factors
and enhanced computational methods for Indian options market.

Author: Market Regime Refactoring Team
Date: 2025-07-06
Version: 2.0.0 - Enhanced Greek Calculations
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class GreekCalculator:
    """
    Core Greek calculation engine with enhanced features
    
    Features:
    - Market-calibrated normalization factors
    - Risk-adjusted Greek calculations
    - Multi-strike Greek aggregation
    - Outlier detection and filtering
    - Greeks validation and quality assessment
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Greek Calculator"""
        self.config = config or {}
        
        # Market-calibrated normalization factors (preserved from original)
        self.normalization_factors = {
            'delta': {
                'method': 'direct',
                'factor': 1.0,
                'description': 'Delta naturally bounded, no scaling needed'
            },
            'gamma': {
                'method': 'scale',
                'factor': self.config.get('gamma_factor', 50.0),  # Calibrated for NIFTY
                'description': 'Gamma scaling for NIFTY options (typical range 0.001-0.02)'
            },
            'theta': {
                'method': 'scale',
                'factor': self.config.get('theta_factor', 5.0),   # Calibrated for daily theta
                'description': 'Theta scaling for daily decay (typical range -0.1 to -0.4)'
            },
            'vega': {
                'method': 'divide',
                'factor': self.config.get('vega_factor', 20.0),  # Calibrated for NIFTY vega
                'description': 'Vega normalization for NIFTY options (typical range 5-40)'
            }
        }
        
        # Enhanced calculation parameters
        self.enable_outlier_detection = self.config.get('enable_outlier_detection', True)
        self.outlier_threshold = self.config.get('outlier_threshold', 3.0)  # Standard deviations
        self.enable_risk_adjustment = self.config.get('enable_risk_adjustment', True)
        self.quality_assessment = self.config.get('quality_assessment', True)
        
        # Validation thresholds
        self.validation_thresholds = {
            'delta': {'min': -1.0, 'max': 1.0},
            'gamma': {'min': 0.0, 'max': 0.1},
            'theta': {'min': -2.0, 'max': 0.1},
            'vega': {'min': 0.0, 'max': 100.0}
        }
        
        # Greeks calculation history
        self.calculation_history = []
        self.normalization_stats = {greek: [] for greek in ['delta', 'gamma', 'theta', 'vega']}
        
        logger.info("GreekCalculator initialized with market-calibrated normalization")
    
    def calculate_greek_contributions(self, 
                                    dte_adjusted_greeks: Dict[str, float],
                                    enable_validation: bool = True) -> Dict[str, float]:
        """
        Calculate individual Greek contributions with market-calibrated normalization
        
        PRESERVED LOGIC: This maintains the exact normalization logic from the original
        enhanced Greek sentiment analysis while adding quality assessment.
        
        Args:
            dte_adjusted_greeks: DTE-adjusted Greek values
            enable_validation: Whether to enable validation checks
            
        Returns:
            Dict[str, float]: Normalized Greek contributions
        """
        try:
            contributions = {}
            calculation_metadata = {
                'timestamp': datetime.now(),
                'input_greeks': dte_adjusted_greeks.copy(),
                'normalization_applied': {},
                'quality_scores': {},
                'outliers_detected': []
            }
            
            # Apply market-calibrated normalization (PRESERVED LOGIC)
            for greek, raw_value in dte_adjusted_greeks.items():
                if greek in self.normalization_factors:
                    norm_config = self.normalization_factors[greek]
                    
                    # Apply normalization based on method
                    if norm_config['method'] == 'direct':
                        # Delta: already in proper range
                        normalized_value = np.clip(raw_value, -1.0, 1.0)
                        
                    elif norm_config['method'] == 'scale':
                        # Gamma, Theta: multiply by calibrated factor
                        normalized_value = np.clip(raw_value * norm_config['factor'], -1.0, 1.0)
                        
                    elif norm_config['method'] == 'divide':
                        # Vega: divide by calibrated factor
                        normalized_value = np.clip(raw_value / norm_config['factor'], -1.0, 1.0)
                        
                    else:
                        # Fallback: direct clipping
                        normalized_value = np.clip(raw_value, -1.0, 1.0)
                    
                    # Record normalization details
                    calculation_metadata['normalization_applied'][greek] = {
                        'method': norm_config['method'],
                        'factor': norm_config['factor'],
                        'raw_value': raw_value,
                        'normalized_value': normalized_value
                    }
                    
                    logger.debug(f"Greek {greek}: raw={raw_value:.6f}, normalized={normalized_value:.6f} "
                               f"(factor={norm_config['factor']}, method={norm_config['method']})")
                    
                else:
                    # Unknown Greek: direct clipping
                    normalized_value = np.clip(raw_value, -1.0, 1.0)
                    logger.debug(f"Unknown Greek {greek}: raw={raw_value:.6f}, normalized={normalized_value:.6f}")
                
                # Outlier detection if enabled
                if self.enable_outlier_detection:
                    is_outlier = self._detect_outlier(greek, normalized_value)
                    if is_outlier:
                        calculation_metadata['outliers_detected'].append(greek)
                        # Apply outlier adjustment
                        normalized_value = self._adjust_outlier(greek, normalized_value)
                
                # Quality assessment if enabled
                if self.quality_assessment:
                    quality_score = self._assess_greek_quality(greek, raw_value, normalized_value)
                    calculation_metadata['quality_scores'][greek] = quality_score
                
                # Validation if enabled
                if enable_validation:
                    normalized_value = self._validate_greek_value(greek, normalized_value)
                
                contributions[greek] = normalized_value
                
                # Update normalization statistics
                self.normalization_stats[greek].append({
                    'timestamp': datetime.now(),
                    'raw_value': raw_value,
                    'normalized_value': normalized_value,
                    'factor_used': norm_config.get('factor', 1.0) if greek in self.normalization_factors else 1.0
                })
            
            # Record calculation in history
            calculation_metadata['final_contributions'] = contributions.copy()
            self._record_calculation(calculation_metadata)
            
            # Log normalization summary (PRESERVED LOGIC)
            logger.info(f"Greek normalization applied: "
                       f"delta={contributions.get('delta', 0):.4f}, "
                       f"gamma={contributions.get('gamma', 0):.4f}, "
                       f"theta={contributions.get('theta', 0):.4f}, "
                       f"vega={contributions.get('vega', 0):.4f}")
            
            return contributions
            
        except Exception as e:
            logger.error(f"Error calculating Greek contributions: {e}")
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
    
    def aggregate_multi_strike_greeks(self, 
                                    strike_greeks: List[Dict[str, Any]],
                                    aggregation_method: str = 'weighted_average') -> Dict[str, float]:
        """
        Aggregate Greeks across multiple strikes
        
        Args:
            strike_greeks: List of Greek dictionaries with weights
            aggregation_method: Method for aggregation
            
        Returns:
            Dict[str, float]: Aggregated Greeks
        """
        try:
            if not strike_greeks:
                return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
            
            aggregated_greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
            
            if aggregation_method == 'weighted_average':
                total_weight = sum(sg.get('weight', 1.0) for sg in strike_greeks)
                
                if total_weight > 0:
                    for strike_greek in strike_greeks:
                        weight = strike_greek.get('weight', 1.0) / total_weight
                        
                        for greek in aggregated_greeks:
                            greek_value = strike_greek.get(greek, 0)
                            aggregated_greeks[greek] += greek_value * weight
            
            elif aggregation_method == 'simple_average':
                for greek in aggregated_greeks:
                    values = [sg.get(greek, 0) for sg in strike_greeks]
                    aggregated_greeks[greek] = np.mean(values)
            
            elif aggregation_method == 'risk_weighted':
                # Risk-weighted aggregation based on position size and moneyness
                aggregated_greeks = self._risk_weighted_aggregation(strike_greeks)
            
            return aggregated_greeks
            
        except Exception as e:
            logger.error(f"Error aggregating multi-strike Greeks: {e}")
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
    
    def calculate_risk_adjusted_greeks(self, 
                                     greeks: Dict[str, float],
                                     market_conditions: Dict[str, Any]) -> Dict[str, float]:
        """Calculate risk-adjusted Greeks based on market conditions"""
        try:
            if not self.enable_risk_adjustment:
                return greeks
            
            risk_adjusted = greeks.copy()
            
            # Volatility risk adjustment
            volatility = market_conditions.get('volatility', 0.2)
            if volatility > 0.4:  # High volatility
                risk_adjusted['gamma'] *= 1.2  # Higher gamma risk
                risk_adjusted['vega'] *= 0.9   # Reduced vega sensitivity
            elif volatility < 0.1:  # Low volatility
                risk_adjusted['gamma'] *= 0.8  # Lower gamma risk
                risk_adjusted['vega'] *= 1.1   # Enhanced vega sensitivity
            
            # Time risk adjustment
            dte = market_conditions.get('dte', 30)
            if dte <= 3:  # Very near expiry
                risk_adjusted['theta'] *= 1.5  # Enhanced time decay risk
                risk_adjusted['gamma'] *= 1.3  # Higher gamma risk
            
            # Liquidity risk adjustment
            avg_volume = market_conditions.get('avg_volume', 1000)
            if avg_volume < 500:  # Low liquidity
                # Reduce sensitivity due to liquidity risk
                for greek in risk_adjusted:
                    risk_adjusted[greek] *= 0.9
            
            return risk_adjusted
            
        except Exception as e:
            logger.error(f"Error calculating risk-adjusted Greeks: {e}")
            return greeks
    
    def _detect_outlier(self, greek: str, value: float) -> bool:
        """Detect if Greek value is an outlier"""
        try:
            if greek not in self.normalization_stats or len(self.normalization_stats[greek]) < 10:
                return False
            
            # Get recent values for this Greek
            recent_values = [
                stat['normalized_value'] 
                for stat in self.normalization_stats[greek][-20:]
            ]
            
            if not recent_values:
                return False
            
            # Calculate Z-score
            mean_value = np.mean(recent_values)
            std_value = np.std(recent_values)
            
            if std_value > 0:
                z_score = abs(value - mean_value) / std_value
                return z_score > self.outlier_threshold
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting outlier for {greek}: {e}")
            return False
    
    def _adjust_outlier(self, greek: str, value: float) -> float:
        """Adjust outlier value to reduce impact"""
        try:
            if greek not in self.normalization_stats:
                return value
            
            # Get recent values
            recent_values = [
                stat['normalized_value'] 
                for stat in self.normalization_stats[greek][-20:]
            ]
            
            if not recent_values:
                return value
            
            # Use median as robust central tendency
            median_value = np.median(recent_values)
            
            # Cap the value at reasonable distance from median
            max_distance = 0.5  # 50% max deviation from median
            
            if value > median_value + max_distance:
                adjusted_value = median_value + max_distance
            elif value < median_value - max_distance:
                adjusted_value = median_value - max_distance
            else:
                adjusted_value = value
            
            if adjusted_value != value:
                logger.debug(f"Outlier adjusted for {greek}: {value:.4f} -> {adjusted_value:.4f}")
            
            return adjusted_value
            
        except Exception as e:
            logger.error(f"Error adjusting outlier for {greek}: {e}")
            return value
    
    def _assess_greek_quality(self, greek: str, raw_value: float, normalized_value: float) -> float:
        """Assess quality of Greek calculation"""
        try:
            quality_score = 1.0
            
            # Check if raw value is reasonable
            thresholds = self.validation_thresholds.get(greek, {})
            raw_min = thresholds.get('min', float('-inf'))
            raw_max = thresholds.get('max', float('inf'))
            
            if not (raw_min <= abs(raw_value) <= raw_max):
                quality_score *= 0.7
            
            # Check normalization effectiveness
            if abs(normalized_value) > 0.95:  # Very close to bounds
                quality_score *= 0.8
            
            # Check for extreme values
            if abs(raw_value) < 1e-10:  # Extremely small
                quality_score *= 0.5
            
            return np.clip(quality_score, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error assessing Greek quality for {greek}: {e}")
            return 0.5
    
    def _validate_greek_value(self, greek: str, value: float) -> float:
        """Validate Greek value against thresholds"""
        try:
            thresholds = self.validation_thresholds.get(greek)
            if not thresholds:
                return value
            
            min_val = thresholds.get('min', -1.0)
            max_val = thresholds.get('max', 1.0)
            
            validated_value = np.clip(value, min_val, max_val)
            
            if validated_value != value:
                logger.debug(f"Greek {greek} validated: {value:.4f} -> {validated_value:.4f}")
            
            return validated_value
            
        except Exception as e:
            logger.error(f"Error validating Greek {greek}: {e}")
            return value
    
    def _risk_weighted_aggregation(self, strike_greeks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Perform risk-weighted aggregation of Greeks"""
        try:
            aggregated = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
            
            # Calculate risk weights based on position size and moneyness
            total_risk_weight = 0
            
            for strike_greek in strike_greeks:
                position_size = strike_greek.get('position_size', 1.0)
                moneyness = strike_greek.get('moneyness', 0.0)
                
                # Risk weight decreases with distance from ATM
                distance_factor = np.exp(-abs(moneyness) * 5)  # Exponential decay
                risk_weight = position_size * distance_factor
                
                for greek in aggregated:
                    greek_value = strike_greek.get(greek, 0)
                    aggregated[greek] += greek_value * risk_weight
                
                total_risk_weight += risk_weight
            
            # Normalize by total risk weight
            if total_risk_weight > 0:
                for greek in aggregated:
                    aggregated[greek] /= total_risk_weight
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Error in risk-weighted aggregation: {e}")
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
    
    def _record_calculation(self, calculation_metadata: Dict[str, Any]):
        """Record calculation details for analysis"""
        try:
            self.calculation_history.append(calculation_metadata)
            
            # Keep only last 100 calculations
            if len(self.calculation_history) > 100:
                self.calculation_history = self.calculation_history[-100:]
            
            # Trim normalization stats
            for greek in self.normalization_stats:
                if len(self.normalization_stats[greek]) > 100:
                    self.normalization_stats[greek] = self.normalization_stats[greek][-100:]
                    
        except Exception as e:
            logger.error(f"Error recording calculation: {e}")
    
    def get_calculation_summary(self) -> Dict[str, Any]:
        """Get summary of Greek calculations"""
        try:
            if not self.calculation_history:
                return {'status': 'no_data'}
            
            recent_calculations = self.calculation_history[-20:]
            
            # Quality statistics
            quality_stats = {}
            for greek in ['delta', 'gamma', 'theta', 'vega']:
                qualities = [
                    calc['quality_scores'].get(greek, 0.5) 
                    for calc in recent_calculations 
                    if 'quality_scores' in calc and greek in calc['quality_scores']
                ]
                
                if qualities:
                    quality_stats[greek] = {
                        'avg_quality': np.mean(qualities),
                        'min_quality': np.min(qualities),
                        'max_quality': np.max(qualities)
                    }
            
            # Outlier statistics
            outlier_counts = {}
            for greek in ['delta', 'gamma', 'theta', 'vega']:
                outlier_count = sum(
                    1 for calc in recent_calculations 
                    if greek in calc.get('outliers_detected', [])
                )
                outlier_counts[greek] = outlier_count
            
            # Normalization factor effectiveness
            normalization_effectiveness = {}
            for greek, norm_config in self.normalization_factors.items():
                if greek in self.normalization_stats and self.normalization_stats[greek]:
                    recent_stats = self.normalization_stats[greek][-20:]
                    raw_values = [stat['raw_value'] for stat in recent_stats]
                    normalized_values = [stat['normalized_value'] for stat in recent_stats]
                    
                    normalization_effectiveness[greek] = {
                        'raw_std': np.std(raw_values),
                        'normalized_std': np.std(normalized_values),
                        'factor': norm_config['factor'],
                        'method': norm_config['method']
                    }
            
            return {
                'total_calculations': len(self.calculation_history),
                'recent_calculations': len(recent_calculations),
                'quality_statistics': quality_stats,
                'outlier_statistics': outlier_counts,
                'normalization_effectiveness': normalization_effectiveness,
                'configuration': {
                    'outlier_detection': self.enable_outlier_detection,
                    'risk_adjustment': self.enable_risk_adjustment,
                    'quality_assessment': self.quality_assessment,
                    'outlier_threshold': self.outlier_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating calculation summary: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def update_normalization_factors(self, 
                                   greek: str,
                                   new_factor: float,
                                   new_method: Optional[str] = None):
        """Update normalization factors"""
        try:
            if greek in self.normalization_factors:
                old_factor = self.normalization_factors[greek]['factor']
                self.normalization_factors[greek]['factor'] = new_factor
                
                if new_method:
                    old_method = self.normalization_factors[greek]['method']
                    self.normalization_factors[greek]['method'] = new_method
                    logger.info(f"Updated {greek} normalization: factor {old_factor} -> {new_factor}, "
                               f"method {old_method} -> {new_method}")
                else:
                    logger.info(f"Updated {greek} normalization factor: {old_factor} -> {new_factor}")
            else:
                logger.warning(f"Unknown Greek for normalization update: {greek}")
                
        except Exception as e:
            logger.error(f"Error updating normalization factors: {e}")
    
    def get_current_normalization_config(self) -> Dict[str, Any]:
        """Get current normalization configuration"""
        return {
            'normalization_factors': self.normalization_factors.copy(),
            'validation_thresholds': self.validation_thresholds.copy(),
            'outlier_detection': {
                'enabled': self.enable_outlier_detection,
                'threshold': self.outlier_threshold
            },
            'risk_adjustment': self.enable_risk_adjustment,
            'quality_assessment': self.quality_assessment
        }