#!/usr/bin/env python3
"""
IV Surface Analyzer - Phase 2 Technical Indicators Enhancement

OBJECTIVE: Deploy 3D volatility surface analysis for comprehensive market regime detection
FEATURES: Multi-dimensional IV analysis, surface curvature detection, regime classification

This analyzer creates a 3D implied volatility surface across strikes and expiries
to provide comprehensive volatility structure analysis for market regime formation.
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class IVSurfaceRegime(Enum):
    """IV Surface regime classifications"""
    FLAT_SURFACE = "Flat_Surface"                    # Low volatility, minimal skew
    STEEP_SKEW = "Steep_Skew"                       # High put skew
    REVERSE_SKEW = "Reverse_Skew"                   # Call skew higher than put skew
    VOLATILITY_SMILE = "Volatility_Smile"           # U-shaped volatility curve
    TERM_STRUCTURE_NORMAL = "Term_Structure_Normal" # Normal forward curve
    TERM_STRUCTURE_INVERTED = "Term_Structure_Inverted" # Inverted curve
    HIGH_SURFACE_VOLATILITY = "High_Surface_Volatility" # Elevated across all strikes

@dataclass
class IVSurfaceResult:
    """Result structure for IV surface analysis"""
    surface_regime: IVSurfaceRegime
    atm_iv_level: float
    skew_measure: float
    term_structure_slope: float
    surface_curvature: float
    volatility_spread: float
    confidence: float
    strike_analysis: Dict[str, float]
    expiry_analysis: Dict[str, float]
    surface_metrics: Dict[str, Any]
    timestamp: datetime

class IVSurfaceAnalyzer:
    """
    IV Surface Analyzer for 3D volatility structure analysis
    
    Analyzes the implied volatility surface across strikes and expiries
    to identify complex volatility patterns and market regime characteristics.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize IV Surface Analyzer
        
        Args:
            config (Dict, optional): Configuration parameters
        """
        self.config = config or {}
        
        # Strike analysis parameters
        self.strike_ranges = {
            'deep_itm': (-300, -100),    # Deep ITM options
            'itm': (-100, -25),          # ITM options
            'atm': (-25, 25),            # ATM options
            'otm': (25, 100),            # OTM options
            'deep_otm': (100, 300)       # Deep OTM options
        }
        
        # Expiry analysis buckets
        self.expiry_buckets = {
            'weekly': (0, 7),            # 0-7 DTE
            'monthly': (8, 35),          # 8-35 DTE
            'quarterly': (36, 90),       # 36-90 DTE
            'long_term': (91, 365)       # 91+ DTE
        }
        
        # Surface regime thresholds
        self.regime_thresholds = {
            'skew_threshold': 0.05,      # 5% skew threshold
            'curvature_threshold': 0.02, # 2% curvature threshold
            'term_slope_threshold': 0.03 # 3% term structure slope
        }
        
        # Minimum data requirements
        self.min_strikes = 5
        self.min_expiries = 2
        
        logger.info("IV Surface Analyzer initialized")
    
    def analyze_iv_surface(self, market_data: Dict[str, Any]) -> IVSurfaceResult:
        """
        Analyze IV surface structure and classify regime

        Args:
            market_data (Dict): Market data including options chain with IV data

        Returns:
            IVSurfaceResult: Comprehensive IV surface analysis
        """
        try:
            # Handle both old format (options_data) and new format (direct list)
            if isinstance(market_data, list):
                # Direct list of market data records from HeavyDB
                surface_data = self._parse_heavydb_surface_data(market_data)
                underlying_price = market_data[0].get('underlying_price', 0) if market_data else 0
            else:
                # Legacy format
                options_data = market_data.get('options_data', {})
                underlying_price = market_data.get('underlying_price', 0)

                if not options_data or underlying_price == 0:
                    logger.warning("Insufficient options data for IV surface analysis")
                    return self._get_default_result()

                # Parse and structure IV surface data
                surface_data = self._parse_surface_data(options_data, underlying_price)

            if len(surface_data) < self.min_strikes:
                logger.warning(f"Insufficient strikes for IV surface analysis: {len(surface_data)} < {self.min_strikes}")
                return self._get_default_result()

            # Calculate ATM IV level
            atm_iv_level = self._calculate_atm_iv_level(surface_data, underlying_price)

            # Calculate skew measure
            skew_measure = self._calculate_skew_measure(surface_data, underlying_price)

            # Calculate term structure slope
            term_structure_slope = self._calculate_term_structure_slope(surface_data)

            # Calculate surface curvature
            surface_curvature = self._calculate_surface_curvature(surface_data)

            # Calculate volatility spread
            volatility_spread = self._calculate_volatility_spread(surface_data)

            # Perform strike analysis
            strike_analysis = self._perform_strike_analysis(surface_data, underlying_price)

            # Perform expiry analysis
            expiry_analysis = self._perform_expiry_analysis(surface_data)

            # Classify surface regime
            surface_regime = self._classify_surface_regime(
                skew_measure, term_structure_slope, surface_curvature, atm_iv_level
            )

            # Calculate confidence
            confidence = self._calculate_confidence(surface_data, {})

            # Prepare surface metrics
            surface_metrics = self._prepare_surface_metrics(
                surface_data, underlying_price
            )

            result = IVSurfaceResult(
                surface_regime=surface_regime,
                atm_iv_level=atm_iv_level,
                skew_measure=skew_measure,
                term_structure_slope=term_structure_slope,
                surface_curvature=surface_curvature,
                volatility_spread=volatility_spread,
                confidence=confidence,
                strike_analysis=strike_analysis,
                expiry_analysis=expiry_analysis,
                surface_metrics=surface_metrics,
                timestamp=datetime.now()
            )

            logger.debug(f"IV surface analysis completed: Regime={surface_regime.value}, "
                        f"ATM IV={atm_iv_level:.3f}, Skew={skew_measure:.3f}, "
                        f"Term Slope={term_structure_slope:.3f}")

            return result

        except Exception as e:
            logger.error(f"Error in IV surface analysis: {e}")
            return self._get_default_result()
    
    def _parse_surface_data(self, options_data: Dict[str, Any], 
                          underlying_price: float) -> List[Dict[str, Any]]:
        """Parse options data into surface structure"""
        try:
            surface_points = []
            
            for strike_key, strike_data in options_data.items():
                try:
                    strike = float(strike_key)
                    moneyness = (strike - underlying_price) / underlying_price
                    
                    # Process calls
                    if 'CE' in strike_data:
                        ce_data = strike_data['CE']
                        ce_iv = ce_data.get('iv', 0)
                        ce_dte = ce_data.get('dte', 30)
                        
                        if ce_iv > 0:
                            surface_points.append({
                                'strike': strike,
                                'moneyness': moneyness,
                                'dte': ce_dte,
                                'iv': ce_iv,
                                'option_type': 'CE'
                            })
                    
                    # Process puts
                    if 'PE' in strike_data:
                        pe_data = strike_data['PE']
                        pe_iv = pe_data.get('iv', 0)
                        pe_dte = pe_data.get('dte', 30)
                        
                        if pe_iv > 0:
                            surface_points.append({
                                'strike': strike,
                                'moneyness': moneyness,
                                'dte': pe_dte,
                                'iv': pe_iv,
                                'option_type': 'PE'
                            })
                            
                except (ValueError, TypeError) as e:
                    logger.debug(f"Skipping invalid strike data: {strike_key}, {e}")
                    continue
            
            return surface_points
            
        except Exception as e:
            logger.error(f"Error parsing surface data: {e}")
            return []

    def _parse_heavydb_surface_data(self, market_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse HeavyDB market data into surface structure"""
        try:
            surface_points = []

            for record in market_data:
                try:
                    # Extract data from HeavyDB record format
                    underlying_price = record.get('underlying_price', 0)
                    strike = record.get('strike', 0)
                    dte = record.get('dte', 30)

                    # Get IV values with enhanced validation and normalization
                    ce_iv = record.get('ce_iv', 0)
                    pe_iv = record.get('pe_iv', 0)

                    # Enhanced IV validation and normalization for extreme values
                    if ce_iv and pe_iv and underlying_price and strike:
                        # Normalize extreme CE IV values (like 0.01)
                        if 0.001 <= ce_iv <= 0.05:
                            ce_iv = max(0.05, ce_iv * 10)  # Scale up extremely low values

                        # Normalize extreme PE IV values (like 60+)
                        if pe_iv > 2.0:
                            pe_iv = min(2.0, pe_iv / 10)  # Scale down extremely high values

                        # Final validation after normalization
                        if 0.05 <= ce_iv <= 2.0 and 0.05 <= pe_iv <= 2.0:
                            moneyness = (strike - underlying_price) / underlying_price

                            # Add CE data point
                            surface_points.append({
                                'strike': strike,
                                'moneyness': moneyness,
                                'dte': dte,
                                'iv': ce_iv,
                                'option_type': 'CE'
                            })

                            # Add PE data point
                            surface_points.append({
                                'strike': strike,
                                'moneyness': moneyness,
                                'dte': dte,
                                'iv': pe_iv,
                                'option_type': 'PE'
                            })

                except (ValueError, TypeError, KeyError) as e:
                    logger.debug(f"Skipping invalid HeavyDB record: {e}")
                    continue

            logger.debug(f"Parsed {len(surface_points)} surface points from {len(market_data)} HeavyDB records")
            return surface_points

        except Exception as e:
            logger.error(f"Error parsing HeavyDB surface data: {e}")
            return []

    def _calculate_atm_iv_level(self, surface_data: List[Dict[str, Any]],
                              underlying_price: float) -> float:
        """Calculate ATM IV level"""
        try:
            atm_ivs = []
            
            for point in surface_data:
                # Consider ATM if within ±2% of underlying
                if abs(point['moneyness']) <= 0.02:
                    atm_ivs.append(point['iv'])
            
            if atm_ivs:
                return np.mean(atm_ivs)
            
            # Fallback: find closest to ATM
            closest_point = min(surface_data, key=lambda x: abs(x['moneyness']))
            return closest_point['iv']
            
        except Exception as e:
            logger.error(f"Error calculating ATM IV level: {e}")
            return 0.15  # Default 15% IV
    
    def _calculate_skew_measure(self, surface_data: List[Dict[str, Any]], 
                              underlying_price: float) -> float:
        """Calculate volatility skew measure"""
        try:
            # Get OTM put and call IVs
            otm_put_ivs = []
            otm_call_ivs = []
            
            for point in surface_data:
                moneyness = point['moneyness']
                
                # OTM puts (strikes below underlying)
                if point['option_type'] == 'PE' and moneyness < -0.05:
                    otm_put_ivs.append(point['iv'])
                
                # OTM calls (strikes above underlying)
                elif point['option_type'] == 'CE' and moneyness > 0.05:
                    otm_call_ivs.append(point['iv'])
            
            if otm_put_ivs and otm_call_ivs:
                avg_put_iv = np.mean(otm_put_ivs)
                avg_call_iv = np.mean(otm_call_ivs)
                
                # Skew = Put IV - Call IV (positive = put skew)
                return avg_put_iv - avg_call_iv
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating skew measure: {e}")
            return 0.0
    
    def _calculate_term_structure_slope(self, surface_data: List[Dict[str, Any]]) -> float:
        """Calculate term structure slope"""
        try:
            # Group by DTE and calculate average IV
            dte_groups = {}
            for point in surface_data:
                dte = point['dte']
                if dte not in dte_groups:
                    dte_groups[dte] = []
                dte_groups[dte].append(point['iv'])
            
            # Calculate average IV for each DTE
            dte_avg_ivs = []
            for dte, ivs in dte_groups.items():
                dte_avg_ivs.append((dte, np.mean(ivs)))
            
            if len(dte_avg_ivs) < 2:
                return 0.0
            
            # Sort by DTE
            dte_avg_ivs.sort(key=lambda x: x[0])
            
            # Calculate slope (change in IV per day)
            short_term = dte_avg_ivs[0]
            long_term = dte_avg_ivs[-1]
            
            dte_diff = long_term[0] - short_term[0]
            iv_diff = long_term[1] - short_term[1]
            
            if dte_diff > 0:
                return iv_diff / dte_diff
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating term structure slope: {e}")
            return 0.0
    
    def _calculate_surface_curvature(self, surface_data: List[Dict[str, Any]]) -> float:
        """Calculate surface curvature (convexity)"""
        try:
            if len(surface_data) < 3:
                return 0.0
            
            # Sort by moneyness
            sorted_data = sorted(surface_data, key=lambda x: x['moneyness'])
            
            # Calculate second derivative approximation
            curvatures = []
            for i in range(1, len(sorted_data) - 1):
                prev_point = sorted_data[i-1]
                curr_point = sorted_data[i]
                next_point = sorted_data[i+1]
                
                # Second derivative approximation
                d2_iv = (next_point['iv'] - 2*curr_point['iv'] + prev_point['iv'])
                dm2 = (next_point['moneyness'] - prev_point['moneyness']) ** 2
                
                if dm2 > 0:
                    curvature = d2_iv / dm2
                    curvatures.append(curvature)
            
            if curvatures:
                return np.mean(curvatures)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating surface curvature: {e}")
            return 0.0
    
    def _calculate_volatility_spread(self, surface_data: List[Dict[str, Any]]) -> float:
        """Calculate volatility spread across the surface"""
        try:
            if not surface_data:
                return 0.0

            ivs = [point['iv'] for point in surface_data]
            return max(ivs) - min(ivs)

        except Exception as e:
            logger.error(f"Error calculating volatility spread: {e}")
            return 0.0

    def _perform_strike_analysis(self, surface_data: List[Dict[str, Any]],
                               underlying_price: float) -> Dict[str, float]:
        """Perform analysis across different strike ranges"""
        try:
            strike_analysis = {}

            for range_name, (min_offset, max_offset) in self.strike_ranges.items():
                range_ivs = []

                for point in surface_data:
                    strike_offset = point['strike'] - underlying_price

                    if min_offset <= strike_offset <= max_offset:
                        range_ivs.append(point['iv'])

                if range_ivs:
                    strike_analysis[range_name] = {
                        'avg_iv': np.mean(range_ivs),
                        'max_iv': np.max(range_ivs),
                        'min_iv': np.min(range_ivs),
                        'iv_std': np.std(range_ivs),
                        'data_points': len(range_ivs)
                    }
                else:
                    strike_analysis[range_name] = {
                        'avg_iv': 0.0,
                        'max_iv': 0.0,
                        'min_iv': 0.0,
                        'iv_std': 0.0,
                        'data_points': 0
                    }

            return strike_analysis

        except Exception as e:
            logger.error(f"Error in strike analysis: {e}")
            return {}

    def _perform_expiry_analysis(self, surface_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Perform analysis across different expiry buckets"""
        try:
            expiry_analysis = {}

            for bucket_name, (min_dte, max_dte) in self.expiry_buckets.items():
                bucket_ivs = []

                for point in surface_data:
                    dte = point['dte']

                    if min_dte <= dte <= max_dte:
                        bucket_ivs.append(point['iv'])

                if bucket_ivs:
                    expiry_analysis[bucket_name] = {
                        'avg_iv': np.mean(bucket_ivs),
                        'max_iv': np.max(bucket_ivs),
                        'min_iv': np.min(bucket_ivs),
                        'iv_std': np.std(bucket_ivs),
                        'data_points': len(bucket_ivs)
                    }
                else:
                    expiry_analysis[bucket_name] = {
                        'avg_iv': 0.0,
                        'max_iv': 0.0,
                        'min_iv': 0.0,
                        'iv_std': 0.0,
                        'data_points': 0
                    }

            return expiry_analysis

        except Exception as e:
            logger.error(f"Error in expiry analysis: {e}")
            return {}

    def _classify_surface_regime(self, skew_measure: float, term_structure_slope: float,
                               surface_curvature: float, atm_iv_level: float) -> IVSurfaceRegime:
        """Classify IV surface regime based on calculated metrics"""
        try:
            # High volatility regime
            if atm_iv_level > 0.30:  # 30% IV threshold
                return IVSurfaceRegime.HIGH_SURFACE_VOLATILITY

            # Term structure analysis
            if abs(term_structure_slope) > self.regime_thresholds['term_slope_threshold']:
                if term_structure_slope < 0:
                    return IVSurfaceRegime.TERM_STRUCTURE_INVERTED
                else:
                    return IVSurfaceRegime.TERM_STRUCTURE_NORMAL

            # Skew analysis
            if abs(skew_measure) > self.regime_thresholds['skew_threshold']:
                if skew_measure > 0:
                    return IVSurfaceRegime.STEEP_SKEW
                else:
                    return IVSurfaceRegime.REVERSE_SKEW

            # Curvature analysis (smile detection)
            if surface_curvature > self.regime_thresholds['curvature_threshold']:
                return IVSurfaceRegime.VOLATILITY_SMILE

            # Default to flat surface
            return IVSurfaceRegime.FLAT_SURFACE

        except Exception as e:
            logger.error(f"Error classifying surface regime: {e}")
            return IVSurfaceRegime.FLAT_SURFACE

    def _calculate_confidence(self, surface_data: List[Dict[str, Any]],
                            options_data: Dict[str, Any]) -> float:
        """Calculate confidence in surface analysis"""
        try:
            # Data coverage confidence
            unique_strikes = len(set(point['strike'] for point in surface_data))
            unique_expiries = len(set(point['dte'] for point in surface_data))

            strike_coverage = min(unique_strikes / 10, 1.0)  # Target 10 strikes
            expiry_coverage = min(unique_expiries / 4, 1.0)  # Target 4 expiries

            # Data quality confidence
            total_points = len(surface_data)
            valid_iv_points = sum(1 for point in surface_data if 0.01 <= point['iv'] <= 1.0)
            data_quality = valid_iv_points / total_points if total_points > 0 else 0.0

            # Surface consistency confidence
            ivs = [point['iv'] for point in surface_data]
            if len(ivs) > 1:
                iv_std = np.std(ivs)
                iv_mean = np.mean(ivs)
                consistency = max(0.1, 1.0 - (iv_std / iv_mean)) if iv_mean > 0 else 0.5
            else:
                consistency = 0.5

            # Combined confidence
            combined_confidence = (
                strike_coverage * 0.3 +
                expiry_coverage * 0.3 +
                data_quality * 0.25 +
                consistency * 0.15
            )

            return np.clip(combined_confidence, 0.1, 1.0)

        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5

    def _prepare_surface_metrics(self, surface_data: List[Dict[str, Any]],
                               underlying_price: float) -> Dict[str, Any]:
        """Prepare comprehensive surface metrics"""
        try:
            metrics = {
                'total_surface_points': len(surface_data),
                'unique_strikes': len(set(point['strike'] for point in surface_data)),
                'unique_expiries': len(set(point['dte'] for point in surface_data)),
                'underlying_price': underlying_price,
                'analysis_timestamp': datetime.now()
            }

            if surface_data:
                ivs = [point['iv'] for point in surface_data]
                strikes = [point['strike'] for point in surface_data]
                dtes = [point['dte'] for point in surface_data]

                metrics.update({
                    'iv_statistics': {
                        'mean': np.mean(ivs),
                        'std': np.std(ivs),
                        'min': np.min(ivs),
                        'max': np.max(ivs),
                        'median': np.median(ivs)
                    },
                    'strike_range': {
                        'min_strike': np.min(strikes),
                        'max_strike': np.max(strikes),
                        'strike_span': np.max(strikes) - np.min(strikes)
                    },
                    'expiry_range': {
                        'min_dte': np.min(dtes),
                        'max_dte': np.max(dtes),
                        'dte_span': np.max(dtes) - np.min(dtes)
                    }
                })

            return metrics

        except Exception as e:
            logger.error(f"Error preparing surface metrics: {e}")
            return {'error': str(e)}

    def _get_default_result(self) -> IVSurfaceResult:
        """Get default result for error cases"""
        return IVSurfaceResult(
            surface_regime=IVSurfaceRegime.FLAT_SURFACE,
            atm_iv_level=0.15,
            skew_measure=0.0,
            term_structure_slope=0.0,
            surface_curvature=0.0,
            volatility_spread=0.0,
            confidence=0.1,
            strike_analysis={},
            expiry_analysis={},
            surface_metrics={'error': 'Insufficient data'},
            timestamp=datetime.now()
        )
