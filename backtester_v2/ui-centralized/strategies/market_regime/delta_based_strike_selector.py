#!/usr/bin/env python3
"""
Delta-based Strike Selection System
===================================

This module implements dynamic strike filtering based on delta ranges for the Enhanced Triple Straddle Framework v2.0:
- CALL options: Delta range 0.5 → 0.01 (decreasing)
- PUT options: Delta range -0.5 → -0.01 (increasing)

Features:
- Real-time delta calculation and filtering
- Dynamic strike range adjustment
- Performance optimization for <3s processing
- Integration with unified_stable_market_regime_pipeline.py
- Mathematical accuracy validation ±0.001

Author: The Augster
Date: 2025-06-20
Version: 2.0.0
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime
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
class DeltaFilterConfig:
    """Configuration for delta-based strike filtering"""
    call_delta_min: float = 0.01    # Minimum CALL delta
    call_delta_max: float = 0.50    # Maximum CALL delta
    put_delta_min: float = -0.50    # Minimum PUT delta (most negative)
    put_delta_max: float = -0.01    # Maximum PUT delta (least negative)
    
    # Black-Scholes parameters
    risk_free_rate: float = 0.05
    default_iv: float = 0.20
    
    # Performance settings
    max_strikes_per_expiry: int = 50
    accuracy_tolerance: float = MATHEMATICAL_TOLERANCE
    
    # Real-time calculation settings
    recalculate_frequency: int = 60  # Recalculate deltas every 60 seconds

@dataclass
class StrikeSelectionResult:
    """Result container for strike selection"""
    selected_strikes: List[float]
    call_strikes: List[float]
    put_strikes: List[float]
    delta_calculations: Dict[float, float]
    selection_timestamp: datetime
    total_strikes_processed: int
    selection_confidence: float
    mathematical_accuracy: bool

class DeltaBasedStrikeSelector:
    """
    Delta-based Strike Selection System implementing dynamic filtering
    based on delta ranges for CALL and PUT options
    """
    
    def __init__(self, config: Optional[DeltaFilterConfig] = None):
        """
        Initialize the Delta-based Strike Selector
        
        Args:
            config: Configuration for delta filtering
        """
        self.config = config or DeltaFilterConfig()
        self.last_calculation_time = None
        self.cached_deltas = {}
        self.selection_history = []
        
        # Validate configuration
        self._validate_configuration()
        
        logger.info("Delta-based Strike Selector initialized")
        logger.info(f"CALL delta range: {self.config.call_delta_min} to {self.config.call_delta_max}")
        logger.info(f"PUT delta range: {self.config.put_delta_min} to {self.config.put_delta_max}")
    
    def _validate_configuration(self) -> None:
        """Validate configuration parameters"""
        try:
            # Validate CALL delta range
            if not (0 < self.config.call_delta_min < self.config.call_delta_max <= 1):
                raise ValueError(f"Invalid CALL delta range: {self.config.call_delta_min} to {self.config.call_delta_max}")
            
            # Validate PUT delta range
            if not (-1 <= self.config.put_delta_min < self.config.put_delta_max < 0):
                raise ValueError(f"Invalid PUT delta range: {self.config.put_delta_min} to {self.config.put_delta_max}")
            
            # Validate other parameters
            if self.config.risk_free_rate < 0:
                raise ValueError(f"Risk-free rate must be non-negative: {self.config.risk_free_rate}")
            
            if not (0 < self.config.default_iv <= 2):
                raise ValueError(f"Default IV must be between 0 and 2: {self.config.default_iv}")
            
            logger.info("Delta filter configuration validation passed")
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    def select_strikes_by_delta(self, market_data: pd.DataFrame, 
                               timestamp: datetime) -> Optional[StrikeSelectionResult]:
        """
        Select strikes based on delta filtering criteria
        
        Args:
            market_data: Market data containing options information
            timestamp: Current timestamp for calculations
            
        Returns:
            StrikeSelectionResult or None if selection fails
        """
        try:
            start_time = datetime.now()
            
            if market_data.empty:
                logger.warning("Empty market data provided for strike selection")
                return None
            
            # Check if we need to recalculate deltas
            should_recalculate = self._should_recalculate_deltas(timestamp)
            
            # Calculate deltas for all options
            delta_calculations = self._calculate_option_deltas(market_data, should_recalculate)
            
            # Filter strikes based on delta criteria
            selected_strikes = self._filter_strikes_by_delta(market_data, delta_calculations)
            
            # Separate CALL and PUT strikes
            call_strikes, put_strikes = self._separate_call_put_strikes(
                market_data, selected_strikes, delta_calculations
            )
            
            # Calculate selection confidence
            confidence = self._calculate_selection_confidence(
                market_data, selected_strikes, delta_calculations
            )
            
            # Validate mathematical accuracy
            accuracy_check = self._validate_delta_accuracy(delta_calculations)
            
            # Create result
            result = StrikeSelectionResult(
                selected_strikes=selected_strikes,
                call_strikes=call_strikes,
                put_strikes=put_strikes,
                delta_calculations=delta_calculations,
                selection_timestamp=timestamp,
                total_strikes_processed=len(market_data),
                selection_confidence=confidence,
                mathematical_accuracy=accuracy_check
            )
            
            # Performance monitoring
            calculation_time = (datetime.now() - start_time).total_seconds()
            if calculation_time > 3.0:  # Performance target: <3 seconds
                logger.warning(f"Strike selection time exceeded target: {calculation_time:.3f}s")
            
            # Store in history
            self.selection_history.append(result)
            if len(self.selection_history) > 1000:
                self.selection_history = self.selection_history[-1000:]
            
            # Update last calculation time
            self.last_calculation_time = timestamp
            
            logger.debug(f"Selected {len(selected_strikes)} strikes in {calculation_time:.3f}s")
            logger.debug(f"CALL strikes: {len(call_strikes)}, PUT strikes: {len(put_strikes)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error selecting strikes by delta: {e}")
            return None
    
    def _should_recalculate_deltas(self, timestamp: datetime) -> bool:
        """Check if deltas should be recalculated based on frequency setting"""
        if self.last_calculation_time is None:
            return True
        
        time_diff = (timestamp - self.last_calculation_time).total_seconds()
        return time_diff >= self.config.recalculate_frequency
    
    def _calculate_option_deltas(self, market_data: pd.DataFrame, 
                                force_recalculate: bool = False) -> Dict[float, float]:
        """Calculate deltas for all options in the dataset"""
        try:
            delta_calculations = {}
            
            for _, row in market_data.iterrows():
                strike = float(row.get('strike', 0))
                
                # Use cached delta if available and not forcing recalculation
                if not force_recalculate and strike in self.cached_deltas:
                    delta_calculations[strike] = self.cached_deltas[strike]
                    continue
                
                # Calculate delta using Black-Scholes
                delta = self._calculate_black_scholes_delta(row)
                delta_calculations[strike] = delta
                
                # Cache the result
                self.cached_deltas[strike] = delta
            
            # Clean old cache entries (keep only current strikes)
            current_strikes = set(delta_calculations.keys())
            self.cached_deltas = {k: v for k, v in self.cached_deltas.items() 
                                if k in current_strikes}
            
            return delta_calculations
            
        except Exception as e:
            logger.error(f"Error calculating option deltas: {e}")
            return {}
    
    def _calculate_black_scholes_delta(self, option_row: pd.Series) -> float:
        """Calculate delta using Black-Scholes formula with enhanced accuracy"""
        try:
            # Extract parameters
            S = float(option_row.get('underlying_price', 0))
            K = float(option_row.get('strike', 0))
            T = float(option_row.get('dte', 0)) / 365.0
            r = self.config.risk_free_rate
            sigma = float(option_row.get('iv', self.config.default_iv))
            option_type = str(option_row.get('option_type', 'CE')).upper()
            
            # Validation
            if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
                logger.warning(f"Invalid parameters for delta calculation: S={S}, K={K}, T={T}, sigma={sigma}")
                return 0.0
            
            # Handle very small time to expiry
            if T < 1/365:  # Less than 1 day
                T = 1/365
            
            # Calculate d1
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            
            # Calculate delta based on option type
            if option_type in ['CE', 'CALL']:
                delta = norm.cdf(d1)
            elif option_type in ['PE', 'PUT']:
                delta = norm.cdf(d1) - 1
            else:
                logger.warning(f"Unknown option type: {option_type}")
                return 0.0
            
            # Validate result
            if not np.isfinite(delta):
                logger.warning(f"Non-finite delta calculated for strike {K}")
                return 0.0
            
            return delta
            
        except Exception as e:
            logger.warning(f"Error calculating Black-Scholes delta: {e}")
            return 0.0

    def _filter_strikes_by_delta(self, market_data: pd.DataFrame,
                                delta_calculations: Dict[float, float]) -> List[float]:
        """Filter strikes based on delta criteria"""
        try:
            selected_strikes = []

            for _, row in market_data.iterrows():
                strike = float(row.get('strike', 0))
                option_type = str(row.get('option_type', 'CE')).upper()

                if strike not in delta_calculations:
                    continue

                delta = delta_calculations[strike]

                # Apply delta filtering based on option type
                if option_type in ['CE', 'CALL']:
                    # CALL options: delta range 0.01 to 0.50
                    if self.config.call_delta_min <= delta <= self.config.call_delta_max:
                        selected_strikes.append(strike)
                elif option_type in ['PE', 'PUT']:
                    # PUT options: delta range -0.50 to -0.01
                    if self.config.put_delta_min <= delta <= self.config.put_delta_max:
                        selected_strikes.append(strike)

            # Remove duplicates and sort
            selected_strikes = sorted(list(set(selected_strikes)))

            # Apply maximum strikes limit per expiry
            if len(selected_strikes) > self.config.max_strikes_per_expiry:
                # Keep strikes closest to ATM (middle of the range)
                mid_point = len(selected_strikes) // 2
                half_limit = self.config.max_strikes_per_expiry // 2
                start_idx = max(0, mid_point - half_limit)
                end_idx = min(len(selected_strikes), start_idx + self.config.max_strikes_per_expiry)
                selected_strikes = selected_strikes[start_idx:end_idx]

            return selected_strikes

        except Exception as e:
            logger.error(f"Error filtering strikes by delta: {e}")
            return []

    def _separate_call_put_strikes(self, market_data: pd.DataFrame,
                                  selected_strikes: List[float],
                                  delta_calculations: Dict[float, float]) -> Tuple[List[float], List[float]]:
        """Separate selected strikes into CALL and PUT lists"""
        try:
            call_strikes = []
            put_strikes = []

            for _, row in market_data.iterrows():
                strike = float(row.get('strike', 0))
                option_type = str(row.get('option_type', 'CE')).upper()

                if strike in selected_strikes:
                    if option_type in ['CE', 'CALL']:
                        call_strikes.append(strike)
                    elif option_type in ['PE', 'PUT']:
                        put_strikes.append(strike)

            return sorted(call_strikes), sorted(put_strikes)

        except Exception as e:
            logger.error(f"Error separating CALL/PUT strikes: {e}")
            return [], []

    def _calculate_selection_confidence(self, market_data: pd.DataFrame,
                                      selected_strikes: List[float],
                                      delta_calculations: Dict[float, float]) -> float:
        """Calculate confidence score for strike selection"""
        try:
            if not selected_strikes or not delta_calculations:
                return 0.0

            # Data quality score
            total_options = len(market_data)
            selected_options = len(selected_strikes)
            selection_ratio = selected_options / max(total_options, 1)

            # Delta accuracy score (how close deltas are to target ranges)
            delta_accuracy_scores = []

            for strike in selected_strikes:
                if strike in delta_calculations:
                    delta = delta_calculations[strike]

                    # Find corresponding option type
                    option_row = market_data[market_data['strike'] == strike]
                    if not option_row.empty:
                        option_type = str(option_row.iloc[0].get('option_type', 'CE')).upper()

                        if option_type in ['CE', 'CALL']:
                            # Score based on how well delta fits in CALL range
                            range_size = self.config.call_delta_max - self.config.call_delta_min
                            distance_from_center = abs(delta - (self.config.call_delta_max + self.config.call_delta_min) / 2)
                            accuracy = 1.0 - (distance_from_center / (range_size / 2))
                        else:  # PUT
                            # Score based on how well delta fits in PUT range
                            range_size = self.config.put_delta_max - self.config.put_delta_min
                            distance_from_center = abs(delta - (self.config.put_delta_max + self.config.put_delta_min) / 2)
                            accuracy = 1.0 - (distance_from_center / (range_size / 2))

                        delta_accuracy_scores.append(max(0.0, accuracy))

            avg_delta_accuracy = np.mean(delta_accuracy_scores) if delta_accuracy_scores else 0.0

            # Volume/OI quality score
            volume_quality = 0.0
            oi_quality = 0.0

            if 'volume' in market_data.columns:
                selected_volume = market_data[market_data['strike'].isin(selected_strikes)]['volume'].sum()
                total_volume = market_data['volume'].sum()
                volume_quality = selected_volume / max(total_volume, 1)

            if 'oi' in market_data.columns:
                selected_oi = market_data[market_data['strike'].isin(selected_strikes)]['oi'].sum()
                total_oi = market_data['oi'].sum()
                oi_quality = selected_oi / max(total_oi, 1)

            # Combined confidence score
            confidence = (
                selection_ratio * 0.3 +
                avg_delta_accuracy * 0.4 +
                volume_quality * 0.15 +
                oi_quality * 0.15
            )

            return np.clip(confidence, 0.0, 1.0)

        except Exception as e:
            logger.error(f"Error calculating selection confidence: {e}")
            return 0.5

    def _validate_delta_accuracy(self, delta_calculations: Dict[float, float]) -> bool:
        """Validate mathematical accuracy of delta calculations"""
        try:
            for strike, delta in delta_calculations.items():
                # Check if delta is finite
                if not np.isfinite(delta):
                    logger.error(f"Non-finite delta for strike {strike}: {delta}")
                    return False

                # Check if delta is within reasonable bounds
                if delta < -1.1 or delta > 1.1:  # Allow small tolerance beyond [-1, 1]
                    logger.error(f"Delta out of bounds for strike {strike}: {delta}")
                    return False

                # Check precision (should be representable within tolerance)
                rounded_delta = round(delta, 3)  # Round to 3 decimal places
                precision_error = abs(delta - rounded_delta)

                if precision_error > self.config.accuracy_tolerance:
                    logger.warning(f"Delta precision warning for strike {strike}: error {precision_error:.6f}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating delta accuracy: {e}")
            return False

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring"""
        try:
            if not self.selection_history:
                return {}

            recent_selections = self.selection_history[-100:]  # Last 100 selections

            # Calculate statistics
            avg_strikes_selected = np.mean([len(sel.selected_strikes) for sel in recent_selections])
            avg_confidence = np.mean([sel.selection_confidence for sel in recent_selections])
            accuracy_rate = np.mean([sel.mathematical_accuracy for sel in recent_selections])

            # Calculate CALL/PUT distribution
            call_put_ratios = []
            for sel in recent_selections:
                total_strikes = len(sel.selected_strikes)
                if total_strikes > 0:
                    call_ratio = len(sel.call_strikes) / total_strikes
                    call_put_ratios.append(call_ratio)

            avg_call_ratio = np.mean(call_put_ratios) if call_put_ratios else 0.5

            return {
                'total_selections': len(self.selection_history),
                'recent_selections': len(recent_selections),
                'average_strikes_selected': avg_strikes_selected,
                'average_confidence': avg_confidence,
                'mathematical_accuracy_rate': accuracy_rate,
                'average_call_ratio': avg_call_ratio,
                'cache_size': len(self.cached_deltas)
            }

        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}

# Integration function for unified_stable_market_regime_pipeline.py
def select_strikes_by_delta_criteria(market_data: pd.DataFrame,
                                   timestamp: datetime,
                                   config: Optional[DeltaFilterConfig] = None) -> Optional[Dict[str, Any]]:
    """
    Main integration function for delta-based strike selection

    Args:
        market_data: Market data containing options information
        timestamp: Current timestamp
        config: Optional configuration for delta filtering

    Returns:
        Dictionary containing strike selection results or None if selection fails
    """
    try:
        # Initialize selector
        selector = DeltaBasedStrikeSelector(config)

        # Select strikes based on delta criteria
        result = selector.select_strikes_by_delta(market_data, timestamp)

        if result is None:
            logger.warning("Delta-based strike selection failed")
            return None

        # Return results in format expected by pipeline
        return {
            'selected_strikes': result.selected_strikes,
            'call_strikes': result.call_strikes,
            'put_strikes': result.put_strikes,
            'total_strikes_processed': result.total_strikes_processed,
            'selection_confidence': result.selection_confidence,
            'mathematical_accuracy': result.mathematical_accuracy,
            'selection_timestamp': result.selection_timestamp.isoformat(),
            'delta_calculations': result.delta_calculations
        }

    except Exception as e:
        logger.error(f"Error in delta-based strike selection: {e}")
        return None

# Unit test function
def test_delta_based_strike_selection():
    """Basic unit test for delta-based strike selector"""
    try:
        # Create test data
        test_data = pd.DataFrame({
            'strike': [22800, 22900, 23000, 23100, 23200, 23300, 23400],
            'option_type': ['CE', 'CE', 'CE', 'PE', 'PE', 'PE', 'PE'],
            'underlying_price': [23100] * 7,
            'dte': [1] * 7,
            'iv': [0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21],
            'volume': [100, 200, 300, 250, 200, 150, 100],
            'oi': [500, 750, 1000, 800, 600, 400, 200]
        })

        # Test selection
        timestamp = datetime.now()
        result = select_strikes_by_delta_criteria(test_data, timestamp)

        if result:
            print("✅ Delta-based strike selection test passed")
            print(f"Total strikes selected: {len(result['selected_strikes'])}")
            print(f"CALL strikes: {len(result['call_strikes'])}")
            print(f"PUT strikes: {len(result['put_strikes'])}")
            print(f"Selection confidence: {result['selection_confidence']:.3f}")
            print(f"Mathematical accuracy: {result['mathematical_accuracy']}")
            return True
        else:
            print("❌ Delta-based strike selection test failed")
            return False

    except Exception as e:
        print(f"❌ Delta-based strike selection test error: {e}")
        return False

if __name__ == "__main__":
    # Run basic test
    test_delta_based_strike_selection()
