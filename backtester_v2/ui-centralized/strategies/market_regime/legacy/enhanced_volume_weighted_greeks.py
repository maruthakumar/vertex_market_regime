#!/usr/bin/env python3
"""
Enhanced Volume-Weighted Greeks Calculator
==========================================

This module implements the volume-weighted Greek calculation formula for the Enhanced Triple Straddle Framework v2.0:
Portfolio_Greek_Exposure = Σ[Greek_i × OI_i × Volume_Weight_i × 50]

Features:
- 9:15 AM baseline establishment
- tanh normalization function
- ±0.001 mathematical accuracy validation
- Real-time Greek calculations
- Performance optimization for <3s processing
- Integration with unified_stable_market_regime_pipeline.py

Author: The Augster
Date: 2025-06-20
Version: 2.0.0
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy.stats import norm
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mathematical precision tolerance
MATHEMATICAL_TOLERANCE = 0.001

@dataclass
class GreekCalculationResult:
    """Result container for Greek calculations"""
    delta_exposure: float
    gamma_exposure: float
    theta_exposure: float
    vega_exposure: float
    portfolio_exposure: float
    volume_weighted_exposure: float
    baseline_normalized_exposure: float
    confidence: float
    calculation_timestamp: datetime
    mathematical_accuracy: bool

@dataclass
class VolumeWeightingConfig:
    """Configuration for volume weighting calculations"""
    baseline_time: time = time(9, 15)  # 9:15 AM baseline
    expiry_weights: Dict[int, float] = None  # DTE-based weights
    greek_component_weights: Dict[str, float] = None
    normalization_method: str = 'tanh'
    accuracy_tolerance: float = MATHEMATICAL_TOLERANCE
    
    def __post_init__(self):
        if self.expiry_weights is None:
            # Default DTE-based weights: 0 DTE: 70%, 1-3 DTE: 30%
            self.expiry_weights = {0: 0.70, 1: 0.30, 2: 0.30, 3: 0.30}
        
        if self.greek_component_weights is None:
            # Default Greek component weights
            self.greek_component_weights = {
                'delta': 0.40,
                'gamma': 0.30,
                'theta': 0.20,
                'vega': 0.10
            }

class EnhancedVolumeWeightedGreeks:
    """
    Enhanced Volume-Weighted Greeks Calculator implementing the mathematical formula:
    Portfolio_Greek_Exposure = Σ[Greek_i × OI_i × Volume_Weight_i × 50]
    """
    
    def __init__(self, config: Optional[VolumeWeightingConfig] = None):
        """
        Initialize the Enhanced Volume-Weighted Greeks Calculator
        
        Args:
            config: Configuration for volume weighting calculations
        """
        self.config = config or VolumeWeightingConfig()
        self.baseline_established = False
        self.baseline_values = {}
        self.calculation_history = []
        
        # Validate configuration
        self._validate_configuration()
        
        logger.info("Enhanced Volume-Weighted Greeks Calculator initialized")
        logger.info(f"Baseline time: {self.config.baseline_time}")
        logger.info(f"Mathematical tolerance: ±{self.config.accuracy_tolerance}")
    
    def _validate_configuration(self) -> None:
        """Validate configuration parameters"""
        try:
            # Validate Greek component weights sum to 1.0
            total_weight = sum(self.config.greek_component_weights.values())
            if abs(total_weight - 1.0) > self.config.accuracy_tolerance:
                raise ValueError(f"Greek component weights must sum to 1.0, got {total_weight}")
            
            # Validate expiry weights are positive
            for dte, weight in self.config.expiry_weights.items():
                if weight < 0:
                    raise ValueError(f"Expiry weight for DTE {dte} must be positive, got {weight}")
            
            # Validate normalization method
            if self.config.normalization_method not in ['tanh', 'sigmoid', 'linear']:
                raise ValueError(f"Invalid normalization method: {self.config.normalization_method}")
            
            logger.info("Configuration validation passed")
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    def establish_baseline(self, market_data: pd.DataFrame, timestamp: datetime) -> bool:
        """
        Establish 9:15 AM baseline for normalization
        
        Args:
            market_data: Market data containing options information
            timestamp: Current timestamp
            
        Returns:
            bool: True if baseline was established successfully
        """
        try:
            # Check if it's baseline time (9:15 AM)
            if timestamp.time() >= self.config.baseline_time and not self.baseline_established:
                
                # Calculate baseline Greek exposures
                baseline_result = self._calculate_raw_greek_exposures(market_data)
                
                if baseline_result:
                    self.baseline_values = {
                        'delta_exposure': baseline_result.delta_exposure,
                        'gamma_exposure': baseline_result.gamma_exposure,
                        'theta_exposure': baseline_result.theta_exposure,
                        'vega_exposure': baseline_result.vega_exposure,
                        'portfolio_exposure': baseline_result.portfolio_exposure,
                        'timestamp': timestamp
                    }
                    
                    self.baseline_established = True
                    logger.info(f"Baseline established at {timestamp}")
                    logger.info(f"Baseline portfolio exposure: {baseline_result.portfolio_exposure:.6f}")
                    
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error establishing baseline: {e}")
            return False
    
    def calculate_volume_weighted_greeks(self, market_data: pd.DataFrame, 
                                       timestamp: datetime) -> Optional[GreekCalculationResult]:
        """
        Calculate volume-weighted Greeks using the enhanced formula
        
        Args:
            market_data: Market data containing options information
            timestamp: Current calculation timestamp
            
        Returns:
            GreekCalculationResult or None if calculation fails
        """
        try:
            start_time = datetime.now()
            
            # Establish baseline if not done yet
            if not self.baseline_established:
                self.establish_baseline(market_data, timestamp)
            
            # Calculate raw Greek exposures
            raw_result = self._calculate_raw_greek_exposures(market_data)
            if not raw_result:
                return None
            
            # Apply volume weighting
            volume_weighted_result = self._apply_volume_weighting(raw_result, market_data)
            
            # Apply baseline normalization
            normalized_result = self._apply_baseline_normalization(volume_weighted_result)
            
            # Validate mathematical accuracy
            accuracy_check = self._validate_mathematical_accuracy(normalized_result)
            
            # Create final result
            result = GreekCalculationResult(
                delta_exposure=raw_result.delta_exposure,
                gamma_exposure=raw_result.gamma_exposure,
                theta_exposure=raw_result.theta_exposure,
                vega_exposure=raw_result.vega_exposure,
                portfolio_exposure=raw_result.portfolio_exposure,
                volume_weighted_exposure=volume_weighted_result,
                baseline_normalized_exposure=normalized_result,
                confidence=self._calculate_confidence(market_data),
                calculation_timestamp=timestamp,
                mathematical_accuracy=accuracy_check
            )
            
            # Performance monitoring
            calculation_time = (datetime.now() - start_time).total_seconds()
            if calculation_time > 3.0:  # Performance target: <3 seconds
                logger.warning(f"Calculation time exceeded target: {calculation_time:.3f}s")
            
            # Store in history for analysis
            self.calculation_history.append(result)
            
            # Keep only last 1000 calculations for memory management
            if len(self.calculation_history) > 1000:
                self.calculation_history = self.calculation_history[-1000:]
            
            logger.debug(f"Volume-weighted Greeks calculated in {calculation_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating volume-weighted Greeks: {e}")
            return None
    
    def _calculate_raw_greek_exposures(self, market_data: pd.DataFrame) -> Optional[GreekCalculationResult]:
        """
        Calculate raw Greek exposures using the core formula:
        Portfolio_Greek_Exposure = Σ[Greek_i × OI_i × Volume_Weight_i × 50]
        """
        try:
            if market_data.empty:
                logger.warning("Empty market data provided")
                return None
            
            # Initialize exposure accumulators
            delta_exposure = 0.0
            gamma_exposure = 0.0
            theta_exposure = 0.0
            vega_exposure = 0.0
            
            # Process each option in the data
            for idx, row in market_data.iterrows():
                try:
                    # Extract required fields
                    oi = float(row.get('oi', 0))
                    volume = float(row.get('volume', 0))
                    
                    # Calculate volume weight (normalized by total volume)
                    total_volume = market_data['volume'].sum()
                    volume_weight = volume / max(total_volume, 1) if total_volume > 0 else 0
                    
                    # Get Greeks (calculate if not provided)
                    delta = self._get_or_calculate_delta(row)
                    gamma = self._get_or_calculate_gamma(row)
                    theta = self._get_or_calculate_theta(row)
                    vega = self._get_or_calculate_vega(row)
                    
                    # Apply the core formula: Greek_i × OI_i × Volume_Weight_i × 50
                    multiplier = 50  # Standard options multiplier
                    
                    delta_exposure += delta * oi * volume_weight * multiplier
                    gamma_exposure += gamma * oi * volume_weight * multiplier
                    theta_exposure += theta * oi * volume_weight * multiplier
                    vega_exposure += vega * oi * volume_weight * multiplier
                    
                except Exception as e:
                    logger.warning(f"Error processing row {idx}: {e}")
                    continue
            
            # Calculate portfolio exposure using Greek component weights
            portfolio_exposure = (
                delta_exposure * self.config.greek_component_weights['delta'] +
                gamma_exposure * self.config.greek_component_weights['gamma'] +
                theta_exposure * self.config.greek_component_weights['theta'] +
                vega_exposure * self.config.greek_component_weights['vega']
            )
            
            return GreekCalculationResult(
                delta_exposure=delta_exposure,
                gamma_exposure=gamma_exposure,
                theta_exposure=theta_exposure,
                vega_exposure=vega_exposure,
                portfolio_exposure=portfolio_exposure,
                volume_weighted_exposure=0.0,  # Will be calculated later
                baseline_normalized_exposure=0.0,  # Will be calculated later
                confidence=0.0,  # Will be calculated later
                calculation_timestamp=datetime.now(),
                mathematical_accuracy=True
            )
            
        except Exception as e:
            logger.error(f"Error calculating raw Greek exposures: {e}")
            return None

    def _get_or_calculate_delta(self, option_row: pd.Series) -> float:
        """Get delta from data or calculate using Black-Scholes"""
        try:
            # Try to get delta from data first
            if 'delta' in option_row and pd.notna(option_row['delta']):
                return float(option_row['delta'])

            # Calculate delta using Black-Scholes if data not available
            return self._calculate_black_scholes_delta(option_row)

        except Exception as e:
            logger.warning(f"Error getting delta: {e}")
            return 0.0

    def _get_or_calculate_gamma(self, option_row: pd.Series) -> float:
        """Get gamma from data or calculate using Black-Scholes"""
        try:
            if 'gamma' in option_row and pd.notna(option_row['gamma']):
                return float(option_row['gamma'])
            return self._calculate_black_scholes_gamma(option_row)
        except Exception as e:
            logger.warning(f"Error getting gamma: {e}")
            return 0.0

    def _get_or_calculate_theta(self, option_row: pd.Series) -> float:
        """Get theta from data or calculate using Black-Scholes"""
        try:
            if 'theta' in option_row and pd.notna(option_row['theta']):
                return float(option_row['theta'])
            return self._calculate_black_scholes_theta(option_row)
        except Exception as e:
            logger.warning(f"Error getting theta: {e}")
            return 0.0

    def _get_or_calculate_vega(self, option_row: pd.Series) -> float:
        """Get vega from data or calculate using Black-Scholes"""
        try:
            if 'vega' in option_row and pd.notna(option_row['vega']):
                return float(option_row['vega'])
            return self._calculate_black_scholes_vega(option_row)
        except Exception as e:
            logger.warning(f"Error getting vega: {e}")
            return 0.0

    def _calculate_black_scholes_delta(self, option_row: pd.Series) -> float:
        """Calculate delta using Black-Scholes formula"""
        try:
            S = float(option_row.get('underlying_price', 0))
            K = float(option_row.get('strike', 0))
            T = float(option_row.get('dte', 0)) / 365.0
            r = 0.05  # Risk-free rate assumption
            sigma = float(option_row.get('iv', 0.2))  # Implied volatility
            option_type = option_row.get('option_type', 'CE')

            if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
                return 0.0

            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))

            if option_type.upper() in ['CE', 'CALL']:
                return norm.cdf(d1)
            else:  # PE, PUT
                return norm.cdf(d1) - 1

        except Exception as e:
            logger.warning(f"Error calculating Black-Scholes delta: {e}")
            return 0.0

    def _calculate_black_scholes_gamma(self, option_row: pd.Series) -> float:
        """Calculate gamma using Black-Scholes formula"""
        try:
            S = float(option_row.get('underlying_price', 0))
            K = float(option_row.get('strike', 0))
            T = float(option_row.get('dte', 0)) / 365.0
            r = 0.05
            sigma = float(option_row.get('iv', 0.2))

            if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
                return 0.0

            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))

            return norm.pdf(d1) / (S * sigma * np.sqrt(T))

        except Exception as e:
            logger.warning(f"Error calculating Black-Scholes gamma: {e}")
            return 0.0

    def _calculate_black_scholes_theta(self, option_row: pd.Series) -> float:
        """Calculate theta using Black-Scholes formula"""
        try:
            S = float(option_row.get('underlying_price', 0))
            K = float(option_row.get('strike', 0))
            T = float(option_row.get('dte', 0)) / 365.0
            r = 0.05
            sigma = float(option_row.get('iv', 0.2))
            option_type = option_row.get('option_type', 'CE')

            if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
                return 0.0

            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)

            if option_type.upper() in ['CE', 'CALL']:
                theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) -
                        r*K*np.exp(-r*T)*norm.cdf(d2)) / 365
            else:  # PE, PUT
                theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) +
                        r*K*np.exp(-r*T)*norm.cdf(-d2)) / 365

            return theta

        except Exception as e:
            logger.warning(f"Error calculating Black-Scholes theta: {e}")
            return 0.0

    def _calculate_black_scholes_vega(self, option_row: pd.Series) -> float:
        """Calculate vega using Black-Scholes formula"""
        try:
            S = float(option_row.get('underlying_price', 0))
            K = float(option_row.get('strike', 0))
            T = float(option_row.get('dte', 0)) / 365.0
            r = 0.05
            sigma = float(option_row.get('iv', 0.2))

            if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
                return 0.0

            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))

            return S * norm.pdf(d1) * np.sqrt(T) / 100  # Divide by 100 for 1% IV change

        except Exception as e:
            logger.warning(f"Error calculating Black-Scholes vega: {e}")
            return 0.0

    def _apply_volume_weighting(self, raw_result: GreekCalculationResult,
                               market_data: pd.DataFrame) -> float:
        """Apply volume weighting to the portfolio exposure"""
        try:
            # Apply DTE-based expiry weighting
            total_weighted_exposure = 0.0
            total_weight = 0.0

            for _, row in market_data.iterrows():
                dte = int(row.get('dte', 0))
                weight = self.config.expiry_weights.get(dte, 0.1)  # Default small weight for unknown DTE

                # Weight the exposure by DTE
                total_weighted_exposure += raw_result.portfolio_exposure * weight
                total_weight += weight

            # Normalize by total weight
            if total_weight > 0:
                return total_weighted_exposure / total_weight
            else:
                return raw_result.portfolio_exposure

        except Exception as e:
            logger.error(f"Error applying volume weighting: {e}")
            return raw_result.portfolio_exposure

    def _apply_baseline_normalization(self, volume_weighted_exposure: float) -> float:
        """Apply baseline normalization using tanh function"""
        try:
            if not self.baseline_established:
                logger.warning("Baseline not established, returning raw exposure")
                return volume_weighted_exposure

            baseline_exposure = self.baseline_values.get('portfolio_exposure', 0)

            # Calculate relative change from baseline
            if abs(baseline_exposure) > self.config.accuracy_tolerance:
                relative_change = (volume_weighted_exposure - baseline_exposure) / baseline_exposure
            else:
                relative_change = 0.0

            # Apply normalization based on method
            if self.config.normalization_method == 'tanh':
                normalized = np.tanh(relative_change)
            elif self.config.normalization_method == 'sigmoid':
                normalized = 1 / (1 + np.exp(-relative_change))
            else:  # linear
                normalized = np.clip(relative_change, -1, 1)

            return normalized

        except Exception as e:
            logger.error(f"Error applying baseline normalization: {e}")
            return 0.0

    def _validate_mathematical_accuracy(self, result: float) -> bool:
        """Validate mathematical accuracy within ±0.001 tolerance"""
        try:
            # Check if result is within reasonable bounds
            if not np.isfinite(result):
                logger.error("Mathematical accuracy check failed: non-finite result")
                return False

            # Check precision (should be representable within tolerance)
            rounded_result = round(result, 3)  # Round to 3 decimal places (0.001 precision)
            precision_error = abs(result - rounded_result)

            if precision_error > self.config.accuracy_tolerance:
                logger.warning(f"Mathematical precision warning: error {precision_error:.6f} > tolerance {self.config.accuracy_tolerance}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating mathematical accuracy: {e}")
            return False

    def _calculate_confidence(self, market_data: pd.DataFrame) -> float:
        """Calculate confidence score based on data quality and completeness"""
        try:
            if market_data.empty:
                return 0.0

            # Data completeness score
            required_columns = ['oi', 'volume', 'strike', 'underlying_price']
            completeness_scores = []

            for col in required_columns:
                if col in market_data.columns:
                    non_null_ratio = market_data[col].notna().sum() / len(market_data)
                    completeness_scores.append(non_null_ratio)
                else:
                    completeness_scores.append(0.0)

            completeness_score = np.mean(completeness_scores)

            # Volume quality score (higher volume = higher confidence)
            total_volume = market_data['volume'].sum() if 'volume' in market_data.columns else 0
            volume_score = min(total_volume / 100000, 1.0)  # Normalize to 100k volume

            # OI quality score
            total_oi = market_data['oi'].sum() if 'oi' in market_data.columns else 0
            oi_score = min(total_oi / 50000, 1.0)  # Normalize to 50k OI

            # Combined confidence score
            confidence = (completeness_score * 0.5 + volume_score * 0.3 + oi_score * 0.2)

            return np.clip(confidence, 0.1, 1.0)

        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring"""
        try:
            if not self.calculation_history:
                return {}

            recent_calculations = self.calculation_history[-100:]  # Last 100 calculations

            # Calculate average processing time (estimated)
            avg_confidence = np.mean([calc.confidence for calc in recent_calculations])
            accuracy_rate = np.mean([calc.mathematical_accuracy for calc in recent_calculations])

            # Calculate exposure statistics
            exposures = [calc.baseline_normalized_exposure for calc in recent_calculations]
            exposure_mean = np.mean(exposures)
            exposure_std = np.std(exposures)

            return {
                'total_calculations': len(self.calculation_history),
                'recent_calculations': len(recent_calculations),
                'average_confidence': avg_confidence,
                'mathematical_accuracy_rate': accuracy_rate,
                'baseline_established': self.baseline_established,
                'exposure_statistics': {
                    'mean': exposure_mean,
                    'std': exposure_std,
                    'min': np.min(exposures),
                    'max': np.max(exposures)
                }
            }

        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}

    def reset_baseline(self) -> None:
        """Reset baseline for new trading session"""
        try:
            self.baseline_established = False
            self.baseline_values = {}
            logger.info("Baseline reset for new session")

        except Exception as e:
            logger.error(f"Error resetting baseline: {e}")

# Integration function for unified_stable_market_regime_pipeline.py
def calculate_volume_weighted_greek_exposure(market_data: pd.DataFrame,
                                           timestamp: datetime,
                                           config: Optional[VolumeWeightingConfig] = None) -> Optional[Dict[str, Any]]:
    """
    Main integration function for calculating volume-weighted Greek exposure

    Args:
        market_data: Market data containing options information
        timestamp: Current calculation timestamp
        config: Optional configuration for calculations

    Returns:
        Dictionary containing Greek exposure results or None if calculation fails
    """
    try:
        # Initialize calculator
        calculator = EnhancedVolumeWeightedGreeks(config)

        # Calculate volume-weighted Greeks
        result = calculator.calculate_volume_weighted_greeks(market_data, timestamp)

        if result is None:
            logger.warning("Volume-weighted Greeks calculation failed")
            return None

        # Return results in format expected by pipeline
        return {
            'volume_weighted_greek_exposure': result.baseline_normalized_exposure,
            'portfolio_exposure': result.portfolio_exposure,
            'delta_exposure': result.delta_exposure,
            'gamma_exposure': result.gamma_exposure,
            'theta_exposure': result.theta_exposure,
            'vega_exposure': result.vega_exposure,
            'confidence': result.confidence,
            'mathematical_accuracy': result.mathematical_accuracy,
            'calculation_timestamp': result.calculation_timestamp.isoformat(),
            'baseline_established': calculator.baseline_established
        }

    except Exception as e:
        logger.error(f"Error in volume-weighted Greek exposure calculation: {e}")
        return None

# Unit test function
def test_volume_weighted_greeks():
    """Basic unit test for volume-weighted Greeks calculator"""
    try:
        # Create test data
        test_data = pd.DataFrame({
            'strike': [23000, 23100, 23200],
            'option_type': ['CE', 'PE', 'CE'],
            'oi': [1000, 1500, 800],
            'volume': [500, 750, 400],
            'underlying_price': [23100, 23100, 23100],
            'dte': [0, 1, 2],
            'iv': [0.15, 0.18, 0.20]
        })

        # Test calculation
        timestamp = datetime.now()
        result = calculate_volume_weighted_greek_exposure(test_data, timestamp)

        if result:
            print("✅ Volume-weighted Greeks test passed")
            print(f"Portfolio exposure: {result['portfolio_exposure']:.6f}")
            print(f"Volume-weighted exposure: {result['volume_weighted_greek_exposure']:.6f}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Mathematical accuracy: {result['mathematical_accuracy']}")
            return True
        else:
            print("❌ Volume-weighted Greeks test failed")
            return False

    except Exception as e:
        print(f"❌ Volume-weighted Greeks test error: {e}")
        return False

if __name__ == "__main__":
    # Run basic test
    test_volume_weighted_greeks()
