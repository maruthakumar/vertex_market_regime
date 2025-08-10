#!/usr/bin/env python3
"""
Advanced Analytics Integration Module for Market Regime Triple Straddle Engine
Phase 3 Implementation: Advanced Analytics Integration

This module implements the advanced analytics enhancements specified in the 
Market Regime Gaps Implementation V1.0 document:

1. Real-Time IV Surface Integration
2. Cross-Greek Correlation Analysis
3. Stress Testing Framework for Extreme Volatility Scenarios

Performance Targets:
- Real-time IV surface analysis with <500ms latency
- Cross-Greek correlation tracking with 95% accuracy
- Stress testing for 99.9% VaR scenarios

Author: Senior Quantitative Trading Expert
Date: June 2025
Version: 1.0 - Phase 3 Advanced Analytics Integration
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
import time
from scipy.interpolate import griddata, RBFInterpolator
from scipy.stats import norm
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class IVSurfacePoint:
    """Individual point on the IV surface"""
    strike: float
    dte: int
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    timestamp: datetime

@dataclass
class GreekCorrelation:
    """Cross-Greek correlation data"""
    greek1: str
    greek2: str
    correlation: float
    confidence_interval: Tuple[float, float]
    sample_size: int
    timestamp: datetime

@dataclass
class StressTestResult:
    """Stress test scenario result"""
    scenario_name: str
    stress_factor: float
    portfolio_pnl: float
    max_drawdown: float
    var_99: float
    expected_shortfall: float
    component_impacts: Dict[str, float]
    timestamp: datetime

class RealTimeIVSurfaceAnalyzer:
    """Real-time implied volatility surface analysis and integration"""
    
    def __init__(self, max_surface_points: int = 1000):
        self.max_surface_points = max_surface_points
        self.surface_points = deque(maxlen=max_surface_points)
        self.surface_cache = {}
        self.interpolation_cache = {}
        
        # Surface analysis parameters
        self.strike_range = (0.7, 1.3)  # 70% to 130% moneyness
        self.dte_range = (0, 30)        # 0 to 30 DTE
        self.interpolation_method = 'cubic'
        
        # Performance metrics
        self.analysis_metrics = {
            'total_surfaces_analyzed': 0,
            'average_analysis_time': 0.0,
            'cache_hit_rate': 0.0,
            'interpolation_accuracy': 0.0
        }
    
    def update_surface_point(self, strike: float, dte: int, market_data: Dict[str, Any]):
        """Update a single point on the IV surface"""
        try:
            # Calculate implied volatility using Black-Scholes approximation
            underlying_price = market_data.get('underlying_price', 100.0)
            option_price = market_data.get('option_price', 5.0)
            risk_free_rate = market_data.get('risk_free_rate', 0.05)
            
            # Simple IV calculation (in practice, use more sophisticated methods)
            moneyness = strike / underlying_price
            time_to_expiry = dte / 365.0
            
            if time_to_expiry > 0:
                implied_vol = self._calculate_implied_volatility(
                    option_price, underlying_price, strike, time_to_expiry, risk_free_rate
                )
            else:
                implied_vol = 0.0
            
            # Calculate Greeks
            greeks = self._calculate_greeks(
                underlying_price, strike, time_to_expiry, implied_vol, risk_free_rate
            )
            
            # Create surface point
            surface_point = IVSurfacePoint(
                strike=strike,
                dte=dte,
                implied_volatility=implied_vol,
                delta=greeks['delta'],
                gamma=greeks['gamma'],
                theta=greeks['theta'],
                vega=greeks['vega'],
                timestamp=datetime.now()
            )
            
            self.surface_points.append(surface_point)
            
            # Invalidate relevant caches
            self._invalidate_surface_cache(strike, dte)
            
            return surface_point
            
        except Exception as e:
            logger.error(f"Error updating surface point: {e}")
            return None
    
    def analyze_surface_regime(self, component_strikes: Dict[str, float], 
                             current_dte: int) -> Dict[str, Any]:
        """Analyze IV surface regime for component strikes"""
        start_time = time.time()
        
        try:
            # Generate cache key
            cache_key = self._generate_surface_cache_key(component_strikes, current_dte)
            
            # Check cache first
            if cache_key in self.surface_cache:
                self.analysis_metrics['cache_hit_rate'] += 1
                return self.surface_cache[cache_key]
            
            # Analyze surface characteristics
            surface_analysis = {
                'timestamp': datetime.now().isoformat(),
                'dte': current_dte,
                'surface_characteristics': self._analyze_surface_characteristics(),
                'component_analysis': {},
                'regime_classification': {},
                'volatility_smile': self._analyze_volatility_smile(current_dte),
                'term_structure': self._analyze_term_structure(component_strikes)
            }
            
            # Analyze each component
            for component_name, strike in component_strikes.items():
                component_analysis = self._analyze_component_surface(strike, current_dte)
                surface_analysis['component_analysis'][component_name] = component_analysis
            
            # Classify surface regime
            surface_analysis['regime_classification'] = self._classify_surface_regime(
                surface_analysis['surface_characteristics']
            )
            
            # Cache results
            self.surface_cache[cache_key] = surface_analysis
            
            # Update performance metrics
            analysis_time = time.time() - start_time
            self._update_analysis_metrics(analysis_time)
            
            return surface_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing surface regime: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _calculate_implied_volatility(self, option_price: float, underlying_price: float,
                                    strike: float, time_to_expiry: float, 
                                    risk_free_rate: float) -> float:
        """Calculate implied volatility using Newton-Raphson method"""
        try:
            # Initial guess
            vol = 0.2
            
            for _ in range(10):  # Maximum 10 iterations
                d1 = (np.log(underlying_price / strike) + 
                      (risk_free_rate + 0.5 * vol**2) * time_to_expiry) / (vol * np.sqrt(time_to_expiry))
                d2 = d1 - vol * np.sqrt(time_to_expiry)
                
                # Black-Scholes call price
                bs_price = (underlying_price * norm.cdf(d1) - 
                           strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2))
                
                # Vega for Newton-Raphson
                vega = underlying_price * norm.pdf(d1) * np.sqrt(time_to_expiry)
                
                if abs(vega) < 1e-6:
                    break
                
                # Newton-Raphson update
                vol_new = vol - (bs_price - option_price) / vega
                
                if abs(vol_new - vol) < 1e-6:
                    break
                
                vol = max(0.01, min(5.0, vol_new))  # Constrain volatility
            
            return vol
            
        except Exception as e:
            logger.error(f"Error calculating implied volatility: {e}")
            return 0.2  # Default volatility
    
    def _calculate_greeks(self, underlying_price: float, strike: float,
                         time_to_expiry: float, volatility: float,
                         risk_free_rate: float) -> Dict[str, float]:
        """Calculate option Greeks"""
        try:
            if time_to_expiry <= 0:
                return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
            
            d1 = (np.log(underlying_price / strike) + 
                  (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
            d2 = d1 - volatility * np.sqrt(time_to_expiry)
            
            # Calculate Greeks
            delta = norm.cdf(d1)
            gamma = norm.pdf(d1) / (underlying_price * volatility * np.sqrt(time_to_expiry))
            theta = (-(underlying_price * norm.pdf(d1) * volatility) / (2 * np.sqrt(time_to_expiry)) -
                    risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2))
            vega = underlying_price * norm.pdf(d1) * np.sqrt(time_to_expiry)
            
            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta / 365,  # Per day
                'vega': vega / 100     # Per 1% vol change
            }
            
        except Exception as e:
            logger.error(f"Error calculating Greeks: {e}")
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
    
    def _analyze_surface_characteristics(self) -> Dict[str, float]:
        """Analyze overall surface characteristics"""
        if not self.surface_points:
            return {}
        
        recent_points = [p for p in self.surface_points 
                        if (datetime.now() - p.timestamp).seconds < 300]  # Last 5 minutes
        
        if not recent_points:
            return {}
        
        ivs = [p.implied_volatility for p in recent_points]
        
        return {
            'average_iv': np.mean(ivs),
            'iv_std': np.std(ivs),
            'iv_skew': self._calculate_skew(recent_points),
            'iv_kurtosis': self._calculate_kurtosis(recent_points),
            'surface_convexity': self._calculate_surface_convexity(recent_points),
            'points_count': len(recent_points)
        }
    
    def _analyze_volatility_smile(self, dte: int) -> Dict[str, float]:
        """Analyze volatility smile for specific DTE"""
        dte_points = [p for p in self.surface_points if p.dte == dte]
        
        if len(dte_points) < 3:
            return {}
        
        # Sort by strike
        dte_points.sort(key=lambda x: x.strike)
        
        strikes = [p.strike for p in dte_points]
        ivs = [p.implied_volatility for p in dte_points]
        
        # Calculate smile characteristics
        atm_iv = ivs[len(ivs)//2] if ivs else 0
        min_iv = min(ivs) if ivs else 0
        max_iv = max(ivs) if ivs else 0
        
        return {
            'atm_iv': atm_iv,
            'smile_asymmetry': (max_iv - min_iv) / max(atm_iv, 0.01),
            'smile_convexity': self._calculate_smile_convexity(strikes, ivs),
            'smile_width': max_iv - min_iv
        }
    
    def _analyze_term_structure(self, component_strikes: Dict[str, float]) -> Dict[str, Any]:
        """Analyze volatility term structure"""
        term_structure = {}
        
        for component_name, strike in component_strikes.items():
            # Get points for this strike across different DTEs
            strike_points = [p for p in self.surface_points 
                           if abs(p.strike - strike) < 0.01]  # Small tolerance
            
            if len(strike_points) < 2:
                continue
            
            # Sort by DTE
            strike_points.sort(key=lambda x: x.dte)
            
            dtes = [p.dte for p in strike_points]
            ivs = [p.implied_volatility for p in strike_points]
            
            if len(dtes) >= 2:
                term_structure[component_name] = {
                    'term_slope': (ivs[-1] - ivs[0]) / max(dtes[-1] - dtes[0], 1),
                    'term_convexity': self._calculate_term_convexity(dtes, ivs),
                    'front_month_iv': ivs[0] if ivs else 0,
                    'back_month_iv': ivs[-1] if ivs else 0
                }
        
        return term_structure
    
    def _classify_surface_regime(self, characteristics: Dict[str, float]) -> Dict[str, str]:
        """Classify the current IV surface regime"""
        if not characteristics:
            return {'regime': 'unknown', 'confidence': 'low'}
        
        avg_iv = characteristics.get('average_iv', 0.2)
        iv_skew = characteristics.get('iv_skew', 0)
        surface_convexity = characteristics.get('surface_convexity', 0)
        
        # Simple regime classification
        if avg_iv < 0.15:
            regime = 'low_volatility'
        elif avg_iv > 0.4:
            regime = 'high_volatility'
        else:
            regime = 'normal_volatility'
        
        # Refine based on skew and convexity
        if abs(iv_skew) > 0.1:
            regime += '_skewed'
        
        if surface_convexity > 0.05:
            regime += '_convex'
        
        confidence = 'high' if characteristics.get('points_count', 0) > 50 else 'medium'
        
        return {
            'regime': regime,
            'confidence': confidence,
            'avg_iv': avg_iv,
            'skew': iv_skew,
            'convexity': surface_convexity
        }
    
    def _calculate_skew(self, points: List[IVSurfacePoint]) -> float:
        """Calculate IV skew"""
        if len(points) < 3:
            return 0
        
        ivs = [p.implied_volatility for p in points]
        return float(pd.Series(ivs).skew())
    
    def _calculate_kurtosis(self, points: List[IVSurfacePoint]) -> float:
        """Calculate IV kurtosis"""
        if len(points) < 4:
            return 0
        
        ivs = [p.implied_volatility for p in points]
        return float(pd.Series(ivs).kurtosis())
    
    def _calculate_surface_convexity(self, points: List[IVSurfacePoint]) -> float:
        """Calculate overall surface convexity"""
        if len(points) < 5:
            return 0
        
        # Simple convexity measure based on second derivative
        points_sorted = sorted(points, key=lambda x: (x.dte, x.strike))
        ivs = [p.implied_volatility for p in points_sorted]
        
        if len(ivs) < 3:
            return 0
        
        # Calculate second differences
        second_diffs = []
        for i in range(1, len(ivs) - 1):
            second_diff = ivs[i+1] - 2*ivs[i] + ivs[i-1]
            second_diffs.append(second_diff)
        
        return np.mean(second_diffs) if second_diffs else 0
    
    def _calculate_smile_convexity(self, strikes: List[float], ivs: List[float]) -> float:
        """Calculate smile convexity"""
        if len(strikes) < 3:
            return 0
        
        # Fit quadratic and return coefficient
        try:
            coeffs = np.polyfit(strikes, ivs, 2)
            return coeffs[0]  # Quadratic coefficient
        except:
            return 0
    
    def _calculate_term_convexity(self, dtes: List[int], ivs: List[float]) -> float:
        """Calculate term structure convexity"""
        if len(dtes) < 3:
            return 0
        
        try:
            coeffs = np.polyfit(dtes, ivs, 2)
            return coeffs[0]  # Quadratic coefficient
        except:
            return 0
    
    def _generate_surface_cache_key(self, component_strikes: Dict[str, float], 
                                  dte: int) -> str:
        """Generate cache key for surface analysis"""
        strikes_str = "_".join([f"{k}:{v:.2f}" for k, v in sorted(component_strikes.items())])
        return f"surface_{strikes_str}_dte{dte}"
    
    def _invalidate_surface_cache(self, strike: float, dte: int):
        """Invalidate relevant cache entries"""
        keys_to_remove = []
        for key in self.surface_cache.keys():
            if f"dte{dte}" in key or f":{strike:.2f}" in key:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.surface_cache[key]
    
    def _analyze_component_surface(self, strike: float, dte: int) -> Dict[str, float]:
        """Analyze surface characteristics for specific component"""
        # Find nearby points
        nearby_points = [
            p for p in self.surface_points
            if abs(p.strike - strike) < 5.0 and abs(p.dte - dte) <= 2
        ]
        
        if not nearby_points:
            return {}
        
        # Calculate component-specific metrics
        ivs = [p.implied_volatility for p in nearby_points]
        deltas = [p.delta for p in nearby_points]
        gammas = [p.gamma for p in nearby_points]
        vegas = [p.vega for p in nearby_points]
        
        return {
            'local_iv': np.mean(ivs),
            'iv_volatility': np.std(ivs),
            'average_delta': np.mean(deltas),
            'average_gamma': np.mean(gammas),
            'average_vega': np.mean(vegas),
            'points_used': len(nearby_points)
        }
    
    def _update_analysis_metrics(self, analysis_time: float):
        """Update performance metrics"""
        self.analysis_metrics['total_surfaces_analyzed'] += 1
        
        # Update average analysis time
        total_analyses = self.analysis_metrics['total_surfaces_analyzed']
        current_avg = self.analysis_metrics['average_analysis_time']
        
        self.analysis_metrics['average_analysis_time'] = (
            (current_avg * (total_analyses - 1) + analysis_time) / total_analyses
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get IV surface analysis performance metrics"""
        return {
            'analysis_metrics': self.analysis_metrics,
            'surface_points_count': len(self.surface_points),
            'cache_size': len(self.surface_cache),
            'recent_surface_activity': self._get_recent_activity_summary()
        }
    
    def _get_recent_activity_summary(self) -> Dict[str, int]:
        """Get summary of recent surface activity"""
        now = datetime.now()
        recent_points = [
            p for p in self.surface_points
            if (now - p.timestamp).seconds < 300  # Last 5 minutes
        ]
        
        return {
            'points_last_5min': len(recent_points),
            'unique_strikes_last_5min': len(set(p.strike for p in recent_points)),
            'unique_dtes_last_5min': len(set(p.dte for p in recent_points))
        }

class CrossGreekCorrelationAnalyzer:
    """Cross-Greek correlation analysis for advanced risk management"""

    def __init__(self, correlation_window: int = 100):
        self.correlation_window = correlation_window
        self.greek_history = deque(maxlen=correlation_window * 10)
        self.correlation_cache = {}
        self.correlation_thresholds = {
            'high': 0.8,
            'medium': 0.5,
            'low': 0.3
        }

        # Greek pairs to analyze
        self.greek_pairs = [
            ('delta', 'gamma'), ('delta', 'theta'), ('delta', 'vega'),
            ('gamma', 'theta'), ('gamma', 'vega'), ('theta', 'vega')
        ]

        # Performance metrics
        self.analysis_metrics = {
            'total_correlations_calculated': 0,
            'high_correlation_alerts': 0,
            'average_calculation_time': 0.0
        }

    def update_greek_data(self, component_name: str, greeks: Dict[str, float]):
        """Update Greek data for correlation analysis"""
        try:
            greek_entry = {
                'timestamp': datetime.now(),
                'component': component_name,
                'delta': greeks.get('delta', 0),
                'gamma': greeks.get('gamma', 0),
                'theta': greeks.get('theta', 0),
                'vega': greeks.get('vega', 0)
            }

            self.greek_history.append(greek_entry)

            # Invalidate relevant correlation cache
            self._invalidate_correlation_cache(component_name)

        except Exception as e:
            logger.error(f"Error updating Greek data: {e}")

    def analyze_cross_greek_correlations(self, components: List[str]) -> Dict[str, Any]:
        """Analyze cross-Greek correlations across components"""
        start_time = time.time()

        try:
            # Generate cache key
            cache_key = f"correlations_{'_'.join(sorted(components))}"

            # Check cache
            if cache_key in self.correlation_cache:
                return self.correlation_cache[cache_key]

            # Get recent Greek data
            recent_data = self._get_recent_greek_data(components)

            if len(recent_data) < 10:  # Minimum data requirement
                return {'error': 'Insufficient data for correlation analysis'}

            # Calculate correlations
            correlation_results = {
                'timestamp': datetime.now().isoformat(),
                'components_analyzed': components,
                'data_points_used': len(recent_data),
                'greek_correlations': {},
                'cross_component_correlations': {},
                'correlation_alerts': [],
                'correlation_summary': {}
            }

            # Intra-Greek correlations (within same component)
            for component in components:
                component_data = [entry for entry in recent_data if entry['component'] == component]
                if len(component_data) >= 10:
                    intra_correlations = self._calculate_intra_greek_correlations(component_data)
                    correlation_results['greek_correlations'][component] = intra_correlations

            # Cross-component correlations (same Greek across components)
            cross_correlations = self._calculate_cross_component_correlations(recent_data, components)
            correlation_results['cross_component_correlations'] = cross_correlations

            # Generate correlation alerts
            alerts = self._generate_correlation_alerts(
                correlation_results['greek_correlations'],
                correlation_results['cross_component_correlations']
            )
            correlation_results['correlation_alerts'] = alerts

            # Summary statistics
            correlation_results['correlation_summary'] = self._generate_correlation_summary(
                correlation_results
            )

            # Cache results
            self.correlation_cache[cache_key] = correlation_results

            # Update performance metrics
            calculation_time = time.time() - start_time
            self._update_correlation_metrics(calculation_time, len(alerts))

            return correlation_results

        except Exception as e:
            logger.error(f"Error analyzing cross-Greek correlations: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

    def _get_recent_greek_data(self, components: List[str]) -> List[Dict[str, Any]]:
        """Get recent Greek data for specified components"""
        cutoff_time = datetime.now() - timedelta(minutes=30)  # Last 30 minutes

        return [
            entry for entry in self.greek_history
            if entry['timestamp'] > cutoff_time and entry['component'] in components
        ]

    def _calculate_intra_greek_correlations(self, component_data: List[Dict[str, Any]]) -> Dict[str, GreekCorrelation]:
        """Calculate correlations between Greeks within same component"""
        correlations = {}

        # Convert to DataFrame for easier correlation calculation
        df = pd.DataFrame(component_data)

        for greek1, greek2 in self.greek_pairs:
            if greek1 in df.columns and greek2 in df.columns:
                correlation_coeff = df[greek1].corr(df[greek2])

                # Calculate confidence interval (simplified)
                n = len(df)
                confidence_interval = self._calculate_correlation_confidence_interval(
                    correlation_coeff, n
                )

                correlations[f"{greek1}_{greek2}"] = GreekCorrelation(
                    greek1=greek1,
                    greek2=greek2,
                    correlation=correlation_coeff,
                    confidence_interval=confidence_interval,
                    sample_size=n,
                    timestamp=datetime.now()
                )

        return correlations

    def _calculate_cross_component_correlations(self, data: List[Dict[str, Any]],
                                              components: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate correlations of same Greek across different components"""
        cross_correlations = {}

        # Create DataFrame
        df = pd.DataFrame(data)

        greeks = ['delta', 'gamma', 'theta', 'vega']

        for greek in greeks:
            greek_correlations = {}

            # Pivot data to have components as columns
            pivot_data = df.pivot_table(
                index='timestamp',
                columns='component',
                values=greek,
                aggfunc='mean'
            ).fillna(method='ffill').fillna(0)

            # Calculate correlations between components for this Greek
            for i, comp1 in enumerate(components):
                for j, comp2 in enumerate(components):
                    if i < j and comp1 in pivot_data.columns and comp2 in pivot_data.columns:
                        correlation = pivot_data[comp1].corr(pivot_data[comp2])
                        greek_correlations[f"{comp1}_{comp2}"] = correlation

            cross_correlations[greek] = greek_correlations

        return cross_correlations

    def _calculate_correlation_confidence_interval(self, correlation: float,
                                                 sample_size: int) -> Tuple[float, float]:
        """Calculate confidence interval for correlation coefficient"""
        if sample_size < 3:
            return (correlation, correlation)

        # Fisher transformation
        z = 0.5 * np.log((1 + correlation) / (1 - correlation))
        se = 1 / np.sqrt(sample_size - 3)

        # 95% confidence interval
        z_lower = z - 1.96 * se
        z_upper = z + 1.96 * se

        # Transform back
        r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)

        return (r_lower, r_upper)

    def _generate_correlation_alerts(self, greek_correlations: Dict[str, Dict],
                                   cross_correlations: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """Generate alerts for high correlations"""
        alerts = []

        # Check intra-Greek correlations
        for component, correlations in greek_correlations.items():
            for pair_name, correlation_obj in correlations.items():
                if abs(correlation_obj.correlation) > self.correlation_thresholds['high']:
                    alerts.append({
                        'type': 'high_intra_greek_correlation',
                        'component': component,
                        'greek_pair': pair_name,
                        'correlation': correlation_obj.correlation,
                        'severity': 'high' if abs(correlation_obj.correlation) > 0.9 else 'medium',
                        'timestamp': datetime.now().isoformat()
                    })

        # Check cross-component correlations
        for greek, correlations in cross_correlations.items():
            for pair_name, correlation in correlations.items():
                if abs(correlation) > self.correlation_thresholds['high']:
                    alerts.append({
                        'type': 'high_cross_component_correlation',
                        'greek': greek,
                        'component_pair': pair_name,
                        'correlation': correlation,
                        'severity': 'high' if abs(correlation) > 0.9 else 'medium',
                        'timestamp': datetime.now().isoformat()
                    })

        return alerts

    def _generate_correlation_summary(self, correlation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics for correlations"""
        all_correlations = []

        # Collect all correlation values
        for component_correlations in correlation_results['greek_correlations'].values():
            for correlation_obj in component_correlations.values():
                all_correlations.append(abs(correlation_obj.correlation))

        for greek_correlations in correlation_results['cross_component_correlations'].values():
            for correlation in greek_correlations.values():
                all_correlations.append(abs(correlation))

        if not all_correlations:
            return {}

        return {
            'total_correlations_calculated': len(all_correlations),
            'average_absolute_correlation': np.mean(all_correlations),
            'max_absolute_correlation': np.max(all_correlations),
            'high_correlation_count': sum(1 for c in all_correlations if c > self.correlation_thresholds['high']),
            'correlation_distribution': {
                'high': sum(1 for c in all_correlations if c > self.correlation_thresholds['high']),
                'medium': sum(1 for c in all_correlations if self.correlation_thresholds['medium'] < c <= self.correlation_thresholds['high']),
                'low': sum(1 for c in all_correlations if c <= self.correlation_thresholds['medium'])
            }
        }

    def _invalidate_correlation_cache(self, component_name: str):
        """Invalidate correlation cache entries for component"""
        keys_to_remove = [key for key in self.correlation_cache.keys() if component_name in key]
        for key in keys_to_remove:
            del self.correlation_cache[key]

    def _update_correlation_metrics(self, calculation_time: float, alert_count: int):
        """Update correlation analysis metrics"""
        self.analysis_metrics['total_correlations_calculated'] += 1
        self.analysis_metrics['high_correlation_alerts'] += alert_count

        # Update average calculation time
        total_calcs = self.analysis_metrics['total_correlations_calculated']
        current_avg = self.analysis_metrics['average_calculation_time']

        self.analysis_metrics['average_calculation_time'] = (
            (current_avg * (total_calcs - 1) + calculation_time) / total_calcs
        )

    def get_correlation_performance_metrics(self) -> Dict[str, Any]:
        """Get correlation analysis performance metrics"""
        return {
            'analysis_metrics': self.analysis_metrics,
            'greek_data_points': len(self.greek_history),
            'cache_size': len(self.correlation_cache),
            'correlation_window': self.correlation_window,
            'recent_activity': self._get_recent_correlation_activity()
        }

    def _get_recent_correlation_activity(self) -> Dict[str, int]:
        """Get recent correlation analysis activity"""
        cutoff_time = datetime.now() - timedelta(minutes=10)
        recent_entries = [
            entry for entry in self.greek_history
            if entry['timestamp'] > cutoff_time
        ]

        return {
            'greek_updates_last_10min': len(recent_entries),
            'unique_components_last_10min': len(set(entry['component'] for entry in recent_entries))
        }

class StressTestingFramework:
    """Stress testing framework for extreme volatility scenarios"""

    def __init__(self):
        self.stress_scenarios = self._initialize_stress_scenarios()
        self.stress_test_history = deque(maxlen=1000)
        self.portfolio_components = [
            'atm_straddle', 'itm1_straddle', 'otm1_straddle',
            'combined_straddle', 'atm_ce', 'atm_pe'
        ]

        # Risk metrics configuration
        self.confidence_levels = [0.95, 0.99, 0.999]  # 95%, 99%, 99.9%
        self.time_horizons = [1, 5, 10, 22]  # 1 day, 1 week, 2 weeks, 1 month

        # Performance metrics
        self.stress_test_metrics = {
            'total_stress_tests': 0,
            'extreme_scenarios_detected': 0,
            'average_test_time': 0.0,
            'var_breaches': 0
        }

    def _initialize_stress_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """Initialize predefined stress test scenarios"""
        return {
            'market_crash': {
                'name': 'Market Crash',
                'underlying_shock': -0.20,  # 20% drop
                'volatility_shock': 2.0,     # 100% vol increase
                'correlation_shock': 0.9,    # High correlation
                'description': 'Severe market crash with volatility spike'
            },
            'volatility_explosion': {
                'name': 'Volatility Explosion',
                'underlying_shock': 0.0,     # No price change
                'volatility_shock': 3.0,     # 200% vol increase
                'correlation_shock': 0.8,    # High correlation
                'description': 'Extreme volatility increase without directional move'
            },
            'flash_crash': {
                'name': 'Flash Crash',
                'underlying_shock': -0.10,   # 10% drop
                'volatility_shock': 1.5,     # 50% vol increase
                'correlation_shock': 0.95,   # Very high correlation
                'description': 'Rapid market decline with correlation breakdown'
            },
            'volatility_collapse': {
                'name': 'Volatility Collapse',
                'underlying_shock': 0.0,     # No price change
                'volatility_shock': 0.3,     # 70% vol decrease
                'correlation_shock': 0.2,    # Low correlation
                'description': 'Extreme volatility compression'
            },
            'black_swan': {
                'name': 'Black Swan Event',
                'underlying_shock': -0.30,   # 30% drop
                'volatility_shock': 4.0,     # 300% vol increase
                'correlation_shock': 0.98,   # Near perfect correlation
                'description': 'Extreme tail event with massive volatility spike'
            },
            'regime_shift': {
                'name': 'Regime Shift',
                'underlying_shock': 0.05,    # 5% move
                'volatility_shock': 1.8,     # 80% vol increase
                'correlation_shock': 0.7,    # Moderate correlation
                'description': 'Sudden market regime change'
            }
        }

    def run_comprehensive_stress_test(self, current_portfolio: Dict[str, Any],
                                    market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive stress test across all scenarios"""
        start_time = time.time()

        try:
            stress_test_results = {
                'timestamp': datetime.now().isoformat(),
                'portfolio_snapshot': current_portfolio.copy(),
                'base_market_data': market_data.copy(),
                'scenario_results': {},
                'risk_metrics': {},
                'stress_test_summary': {},
                'recommendations': []
            }

            # Run each stress scenario
            for scenario_name, scenario_config in self.stress_scenarios.items():
                scenario_result = self._run_single_stress_scenario(
                    scenario_name, scenario_config, current_portfolio, market_data
                )
                stress_test_results['scenario_results'][scenario_name] = scenario_result

            # Calculate comprehensive risk metrics
            stress_test_results['risk_metrics'] = self._calculate_comprehensive_risk_metrics(
                stress_test_results['scenario_results']
            )

            # Generate stress test summary
            stress_test_results['stress_test_summary'] = self._generate_stress_test_summary(
                stress_test_results
            )

            # Generate recommendations
            stress_test_results['recommendations'] = self._generate_stress_test_recommendations(
                stress_test_results
            )

            # Store results
            self.stress_test_history.append(stress_test_results)

            # Update performance metrics
            test_time = time.time() - start_time
            self._update_stress_test_metrics(test_time, stress_test_results)

            return stress_test_results

        except Exception as e:
            logger.error(f"Error running comprehensive stress test: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

    def _run_single_stress_scenario(self, scenario_name: str, scenario_config: Dict[str, Any],
                                   portfolio: Dict[str, Any], market_data: Dict[str, Any]) -> StressTestResult:
        """Run single stress test scenario"""
        try:
            # Apply stress shocks to market data
            stressed_market_data = self._apply_stress_shocks(market_data, scenario_config)

            # Calculate portfolio P&L under stress
            portfolio_pnl = self._calculate_stressed_portfolio_pnl(
                portfolio, market_data, stressed_market_data
            )

            # Calculate component impacts
            component_impacts = self._calculate_component_impacts(
                portfolio, market_data, stressed_market_data
            )

            # Calculate risk metrics
            max_drawdown = self._calculate_max_drawdown(portfolio_pnl)
            var_99 = self._calculate_var(portfolio_pnl, 0.99)
            expected_shortfall = self._calculate_expected_shortfall(portfolio_pnl, 0.99)

            return StressTestResult(
                scenario_name=scenario_name,
                stress_factor=scenario_config.get('volatility_shock', 1.0),
                portfolio_pnl=portfolio_pnl,
                max_drawdown=max_drawdown,
                var_99=var_99,
                expected_shortfall=expected_shortfall,
                component_impacts=component_impacts,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Error running stress scenario {scenario_name}: {e}")
            return StressTestResult(
                scenario_name=scenario_name,
                stress_factor=0.0,
                portfolio_pnl=0.0,
                max_drawdown=0.0,
                var_99=0.0,
                expected_shortfall=0.0,
                component_impacts={},
                timestamp=datetime.now()
            )

    def _apply_stress_shocks(self, market_data: Dict[str, Any],
                           scenario_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply stress shocks to market data"""
        stressed_data = market_data.copy()

        # Apply underlying price shock
        underlying_shock = scenario_config.get('underlying_shock', 0.0)
        current_price = market_data.get('underlying_price', 100.0)
        stressed_data['underlying_price'] = current_price * (1 + underlying_shock)

        # Apply volatility shock
        vol_shock = scenario_config.get('volatility_shock', 1.0)
        current_vol = market_data.get('implied_volatility', 0.2)
        stressed_data['implied_volatility'] = current_vol * vol_shock

        # Apply VIX shock (proxy for market volatility)
        current_vix = market_data.get('vix', 20.0)
        stressed_data['vix'] = current_vix * vol_shock

        # Apply correlation shock
        correlation_shock = scenario_config.get('correlation_shock', 0.5)
        stressed_data['correlation_adjustment'] = correlation_shock

        return stressed_data

    def _calculate_stressed_portfolio_pnl(self, portfolio: Dict[str, Any],
                                        base_market_data: Dict[str, Any],
                                        stressed_market_data: Dict[str, Any]) -> float:
        """Calculate portfolio P&L under stress scenario"""
        try:
            total_pnl = 0.0

            # Calculate P&L for each component
            for component in self.portfolio_components:
                component_weight = portfolio.get('weights', {}).get(component, 0.0)
                component_position = portfolio.get('positions', {}).get(component, 0.0)

                if component_weight > 0 and component_position != 0:
                    # Calculate component P&L
                    base_value = self._calculate_component_value(
                        component, base_market_data, component_position
                    )
                    stressed_value = self._calculate_component_value(
                        component, stressed_market_data, component_position
                    )

                    component_pnl = (stressed_value - base_value) * component_weight
                    total_pnl += component_pnl

            return total_pnl

        except Exception as e:
            logger.error(f"Error calculating stressed portfolio P&L: {e}")
            return 0.0

    def _calculate_component_value(self, component: str, market_data: Dict[str, Any],
                                 position: float) -> float:
        """Calculate component value given market data"""
        # Simplified component valuation
        underlying_price = market_data.get('underlying_price', 100.0)
        implied_vol = market_data.get('implied_volatility', 0.2)

        # Component-specific valuation logic
        if 'straddle' in component:
            # Straddle value approximation
            base_value = underlying_price * 0.1 * implied_vol
        elif 'ce' in component:
            # Call option value approximation
            base_value = max(0, underlying_price - 100) + underlying_price * 0.05 * implied_vol
        elif 'pe' in component:
            # Put option value approximation
            base_value = max(0, 100 - underlying_price) + underlying_price * 0.05 * implied_vol
        else:
            base_value = underlying_price * 0.05

        return base_value * position

    def _calculate_component_impacts(self, portfolio: Dict[str, Any],
                                   base_market_data: Dict[str, Any],
                                   stressed_market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate individual component impacts under stress"""
        component_impacts = {}

        for component in self.portfolio_components:
            component_weight = portfolio.get('weights', {}).get(component, 0.0)
            component_position = portfolio.get('positions', {}).get(component, 0.0)

            if component_weight > 0:
                base_value = self._calculate_component_value(
                    component, base_market_data, component_position
                )
                stressed_value = self._calculate_component_value(
                    component, stressed_market_data, component_position
                )

                impact = (stressed_value - base_value) * component_weight
                component_impacts[component] = impact

        return component_impacts

    def _calculate_max_drawdown(self, pnl: float) -> float:
        """Calculate maximum drawdown (simplified for single scenario)"""
        return min(0, pnl)  # Simplified - in practice would use time series

    def _calculate_var(self, pnl: float, confidence_level: float) -> float:
        """Calculate Value at Risk (simplified for single scenario)"""
        # In practice, this would use historical simulation or Monte Carlo
        return pnl if pnl < 0 else 0

    def _calculate_expected_shortfall(self, pnl: float, confidence_level: float) -> float:
        """Calculate Expected Shortfall (simplified for single scenario)"""
        # In practice, this would be the average of losses beyond VaR
        return pnl if pnl < 0 else 0

    def _calculate_comprehensive_risk_metrics(self, scenario_results: Dict[str, StressTestResult]) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics across all scenarios"""
        all_pnls = [result.portfolio_pnl for result in scenario_results.values()]
        all_drawdowns = [result.max_drawdown for result in scenario_results.values()]
        all_vars = [result.var_99 for result in scenario_results.values()]

        return {
            'worst_case_pnl': min(all_pnls) if all_pnls else 0,
            'best_case_pnl': max(all_pnls) if all_pnls else 0,
            'average_pnl': np.mean(all_pnls) if all_pnls else 0,
            'pnl_volatility': np.std(all_pnls) if all_pnls else 0,
            'worst_drawdown': min(all_drawdowns) if all_drawdowns else 0,
            'average_var_99': np.mean(all_vars) if all_vars else 0,
            'scenarios_with_losses': sum(1 for pnl in all_pnls if pnl < 0),
            'extreme_loss_scenarios': sum(1 for pnl in all_pnls if pnl < -1000)  # Threshold-based
        }

    def _generate_stress_test_summary(self, stress_test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate stress test summary"""
        scenario_results = stress_test_results['scenario_results']
        risk_metrics = stress_test_results['risk_metrics']

        # Find worst and best scenarios
        worst_scenario = min(scenario_results.items(), key=lambda x: x[1].portfolio_pnl)
        best_scenario = max(scenario_results.items(), key=lambda x: x[1].portfolio_pnl)

        return {
            'total_scenarios_tested': len(scenario_results),
            'worst_scenario': {
                'name': worst_scenario[0],
                'pnl': worst_scenario[1].portfolio_pnl,
                'description': self.stress_scenarios[worst_scenario[0]]['description']
            },
            'best_scenario': {
                'name': best_scenario[0],
                'pnl': best_scenario[1].portfolio_pnl,
                'description': self.stress_scenarios[best_scenario[0]]['description']
            },
            'risk_assessment': self._assess_portfolio_risk(risk_metrics),
            'stress_test_score': self._calculate_stress_test_score(risk_metrics)
        }

    def _generate_stress_test_recommendations(self, stress_test_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on stress test results"""
        recommendations = []
        risk_metrics = stress_test_results['risk_metrics']

        # Check for extreme losses
        if risk_metrics['worst_case_pnl'] < -5000:
            recommendations.append("Portfolio shows extreme vulnerability to stress scenarios - consider reducing position sizes")

        # Check for high volatility
        if risk_metrics['pnl_volatility'] > 2000:
            recommendations.append("High P&L volatility across scenarios - implement volatility hedging")

        # Check for concentration risk
        scenario_results = stress_test_results['scenario_results']
        correlation_sensitive_scenarios = ['market_crash', 'flash_crash', 'black_swan']
        correlation_losses = sum(scenario_results[scenario].portfolio_pnl
                               for scenario in correlation_sensitive_scenarios
                               if scenario in scenario_results)

        if correlation_losses < -3000:
            recommendations.append("High correlation risk detected - diversify component exposures")

        # Check for volatility risk
        vol_sensitive_scenarios = ['volatility_explosion', 'volatility_collapse']
        vol_losses = sum(abs(scenario_results[scenario].portfolio_pnl)
                        for scenario in vol_sensitive_scenarios
                        if scenario in scenario_results)

        if vol_losses > 2000:
            recommendations.append("High volatility sensitivity - consider vega hedging")

        if not recommendations:
            recommendations.append("Portfolio shows reasonable resilience to stress scenarios")

        return recommendations

    def _assess_portfolio_risk(self, risk_metrics: Dict[str, Any]) -> str:
        """Assess overall portfolio risk level"""
        worst_case = abs(risk_metrics['worst_case_pnl'])
        extreme_scenarios = risk_metrics['extreme_loss_scenarios']

        if worst_case > 10000 or extreme_scenarios > 2:
            return 'HIGH'
        elif worst_case > 5000 or extreme_scenarios > 1:
            return 'MEDIUM'
        else:
            return 'LOW'

    def _calculate_stress_test_score(self, risk_metrics: Dict[str, Any]) -> float:
        """Calculate overall stress test score (0-100)"""
        # Normalize metrics to 0-100 scale
        worst_case_score = max(0, 100 - abs(risk_metrics['worst_case_pnl']) / 100)
        volatility_score = max(0, 100 - risk_metrics['pnl_volatility'] / 50)
        scenario_score = max(0, 100 - risk_metrics['scenarios_with_losses'] * 10)

        # Weighted average
        overall_score = (worst_case_score * 0.5 + volatility_score * 0.3 + scenario_score * 0.2)

        return min(100, max(0, overall_score))

    def _update_stress_test_metrics(self, test_time: float, stress_test_results: Dict[str, Any]):
        """Update stress testing performance metrics"""
        self.stress_test_metrics['total_stress_tests'] += 1

        # Count extreme scenarios
        risk_metrics = stress_test_results.get('risk_metrics', {})
        if risk_metrics.get('extreme_loss_scenarios', 0) > 0:
            self.stress_test_metrics['extreme_scenarios_detected'] += 1

        # Update average test time
        total_tests = self.stress_test_metrics['total_stress_tests']
        current_avg = self.stress_test_metrics['average_test_time']

        self.stress_test_metrics['average_test_time'] = (
            (current_avg * (total_tests - 1) + test_time) / total_tests
        )

    def get_stress_testing_metrics(self) -> Dict[str, Any]:
        """Get stress testing performance metrics"""
        return {
            'stress_test_metrics': self.stress_test_metrics,
            'available_scenarios': list(self.stress_scenarios.keys()),
            'test_history_size': len(self.stress_test_history),
            'recent_stress_tests': self._get_recent_stress_test_summary()
        }

    def _get_recent_stress_test_summary(self) -> Dict[str, Any]:
        """Get summary of recent stress tests"""
        if not self.stress_test_history:
            return {}

        recent_tests = list(self.stress_test_history)[-10:]  # Last 10 tests

        avg_worst_case = np.mean([
            test['risk_metrics']['worst_case_pnl']
            for test in recent_tests
            if 'risk_metrics' in test
        ])

        return {
            'recent_tests_count': len(recent_tests),
            'average_worst_case_pnl': avg_worst_case,
            'last_test_timestamp': recent_tests[-1].get('timestamp', 'N/A') if recent_tests else 'N/A'
        }

class IntegratedAdvancedAnalyticsSystem:
    """Integrated advanced analytics system combining all Phase 3 components"""

    def __init__(self):
        self.iv_surface_analyzer = RealTimeIVSurfaceAnalyzer()
        self.greek_correlation_analyzer = CrossGreekCorrelationAnalyzer()
        self.stress_testing_framework = StressTestingFramework()

        # Integration configuration
        self.analysis_weights = {
            'iv_surface': 0.4,      # 40% weight to IV surface analysis
            'greek_correlation': 0.35,  # 35% weight to Greek correlation analysis
            'stress_testing': 0.25   # 25% weight to stress testing
        }

        self.integration_history = deque(maxlen=500)
        self.performance_metrics = {
            'total_integrated_analyses': 0,
            'average_analysis_time': 0.0,
            'alert_generation_rate': 0.0
        }

    def run_comprehensive_advanced_analysis(self, market_data: Dict[str, Any],
                                          portfolio_data: Dict[str, Any],
                                          component_strikes: Dict[str, float],
                                          current_dte: int) -> Dict[str, Any]:
        """Run comprehensive advanced analytics analysis"""
        start_time = time.time()

        try:
            # Step 1: IV Surface Analysis
            iv_analysis = self.iv_surface_analyzer.analyze_surface_regime(
                component_strikes, current_dte
            )

            # Step 2: Greek Correlation Analysis
            components = list(component_strikes.keys())
            correlation_analysis = self.greek_correlation_analyzer.analyze_cross_greek_correlations(
                components
            )

            # Step 3: Stress Testing
            stress_test_results = self.stress_testing_framework.run_comprehensive_stress_test(
                portfolio_data, market_data
            )

            # Step 4: Integrate Results
            integrated_results = {
                'timestamp': datetime.now().isoformat(),
                'iv_surface_analysis': iv_analysis,
                'greek_correlation_analysis': correlation_analysis,
                'stress_test_results': stress_test_results,
                'integrated_assessment': self._generate_integrated_assessment(
                    iv_analysis, correlation_analysis, stress_test_results
                ),
                'comprehensive_alerts': self._generate_comprehensive_alerts(
                    iv_analysis, correlation_analysis, stress_test_results
                ),
                'performance_metrics': self._get_integrated_performance_metrics()
            }

            # Store results
            self.integration_history.append(integrated_results)

            # Update performance metrics
            analysis_time = time.time() - start_time
            self._update_integration_metrics(analysis_time, integrated_results)

            return integrated_results

        except Exception as e:
            logger.error(f"Error in comprehensive advanced analysis: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

    def _generate_integrated_assessment(self, iv_analysis: Dict[str, Any],
                                      correlation_analysis: Dict[str, Any],
                                      stress_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate integrated assessment combining all analyses"""
        assessment = {
            'overall_risk_level': 'MEDIUM',
            'confidence_score': 0.0,
            'key_findings': [],
            'risk_factors': [],
            'opportunities': []
        }

        # Assess IV surface regime
        if 'regime_classification' in iv_analysis:
            regime = iv_analysis['regime_classification']
            if 'high_volatility' in regime.get('regime', ''):
                assessment['risk_factors'].append('High volatility regime detected')
            elif 'low_volatility' in regime.get('regime', ''):
                assessment['opportunities'].append('Low volatility environment for premium selling')

        # Assess correlation risks
        if 'correlation_alerts' in correlation_analysis:
            high_corr_alerts = [
                alert for alert in correlation_analysis['correlation_alerts']
                if alert.get('severity') == 'high'
            ]
            if len(high_corr_alerts) > 2:
                assessment['risk_factors'].append('Multiple high correlation alerts detected')

        # Assess stress test results
        if 'risk_metrics' in stress_results:
            risk_assessment = stress_results.get('stress_test_summary', {}).get('risk_assessment', 'MEDIUM')
            if risk_assessment == 'HIGH':
                assessment['risk_factors'].append('High stress test vulnerability')
                assessment['overall_risk_level'] = 'HIGH'
            elif risk_assessment == 'LOW':
                assessment['opportunities'].append('Portfolio shows stress resilience')

        # Calculate confidence score
        data_quality_scores = []
        if iv_analysis.get('surface_characteristics', {}).get('points_count', 0) > 50:
            data_quality_scores.append(0.9)
        else:
            data_quality_scores.append(0.6)

        if correlation_analysis.get('data_points_used', 0) > 20:
            data_quality_scores.append(0.9)
        else:
            data_quality_scores.append(0.6)

        if len(stress_results.get('scenario_results', {})) >= 5:
            data_quality_scores.append(0.9)
        else:
            data_quality_scores.append(0.7)

        assessment['confidence_score'] = np.mean(data_quality_scores)

        # Generate key findings
        assessment['key_findings'] = [
            f"IV surface regime: {iv_analysis.get('regime_classification', {}).get('regime', 'unknown')}",
            f"Correlation alerts: {len(correlation_analysis.get('correlation_alerts', []))}",
            f"Stress test score: {stress_results.get('stress_test_summary', {}).get('stress_test_score', 0):.1f}/100"
        ]

        return assessment

    def _generate_comprehensive_alerts(self, iv_analysis: Dict[str, Any],
                                     correlation_analysis: Dict[str, Any],
                                     stress_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate comprehensive alerts from all analyses"""
        alerts = []

        # IV Surface alerts
        if 'regime_classification' in iv_analysis:
            regime = iv_analysis['regime_classification']
            if regime.get('confidence') == 'high' and 'extreme' in regime.get('regime', ''):
                alerts.append({
                    'type': 'iv_surface_extreme_regime',
                    'severity': 'high',
                    'message': f"Extreme IV regime detected: {regime.get('regime')}",
                    'source': 'iv_surface_analyzer'
                })

        # Greek correlation alerts
        correlation_alerts = correlation_analysis.get('correlation_alerts', [])
        for alert in correlation_alerts:
            if alert.get('severity') == 'high':
                alerts.append({
                    'type': 'high_greek_correlation',
                    'severity': 'high',
                    'message': f"High correlation detected: {alert.get('type')}",
                    'source': 'greek_correlation_analyzer'
                })

        # Stress testing alerts
        if 'risk_metrics' in stress_results:
            risk_metrics = stress_results['risk_metrics']
            if risk_metrics.get('extreme_loss_scenarios', 0) > 1:
                alerts.append({
                    'type': 'extreme_stress_vulnerability',
                    'severity': 'high',
                    'message': f"Portfolio vulnerable to {risk_metrics['extreme_loss_scenarios']} extreme scenarios",
                    'source': 'stress_testing_framework'
                })

        return alerts

    def _get_integrated_performance_metrics(self) -> Dict[str, Any]:
        """Get integrated performance metrics from all components"""
        return {
            'iv_surface_metrics': self.iv_surface_analyzer.get_performance_metrics(),
            'correlation_metrics': self.greek_correlation_analyzer.get_correlation_performance_metrics(),
            'stress_testing_metrics': self.stress_testing_framework.get_stress_testing_metrics(),
            'integration_metrics': self.performance_metrics
        }

    def _update_integration_metrics(self, analysis_time: float, results: Dict[str, Any]):
        """Update integration performance metrics"""
        self.performance_metrics['total_integrated_analyses'] += 1

        # Update average analysis time
        total_analyses = self.performance_metrics['total_integrated_analyses']
        current_avg = self.performance_metrics['average_analysis_time']

        self.performance_metrics['average_analysis_time'] = (
            (current_avg * (total_analyses - 1) + analysis_time) / total_analyses
        )

        # Update alert generation rate
        alert_count = len(results.get('comprehensive_alerts', []))
        current_rate = self.performance_metrics['alert_generation_rate']

        self.performance_metrics['alert_generation_rate'] = (
            (current_rate * (total_analyses - 1) + alert_count) / total_analyses
        )

# Factory function for easy instantiation
def create_advanced_analytics_system() -> IntegratedAdvancedAnalyticsSystem:
    """Factory function to create integrated advanced analytics system"""
    return IntegratedAdvancedAnalyticsSystem()
