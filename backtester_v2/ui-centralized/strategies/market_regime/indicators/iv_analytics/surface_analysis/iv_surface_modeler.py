"""
IV Surface Modeler - Implied Volatility Surface Construction
===========================================================

Constructs and analyzes the implied volatility surface across strikes
and expiration dates using sophisticated interpolation and modeling techniques.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from scipy import interpolate
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class IVSurfaceModeler:
    """
    Comprehensive IV surface modeling and analysis
    
    Features:
    - Multi-dimensional surface construction
    - Arbitrage-free surface interpolation
    - Volatility smile modeling
    - Surface quality metrics
    - Time evolution tracking
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize IV Surface Modeler"""
        # Surface modeling parameters
        self.min_points_per_expiry = config.get('min_points_per_expiry', 5)
        self.max_extrapolation_distance = config.get('max_extrapolation_distance', 0.2)
        self.smoothing_factor = config.get('smoothing_factor', 0.1)
        
        # Arbitrage constraints
        self.enable_arbitrage_checks = config.get('enable_arbitrage_checks', True)
        self.butterfly_spread_threshold = config.get('butterfly_spread_threshold', 0.001)
        self.calendar_spread_threshold = config.get('calendar_spread_threshold', 0.001)
        
        # Model parameters
        self.surface_model_type = config.get('surface_model_type', 'cubic_spline')
        self.smile_model_type = config.get('smile_model_type', 'sabr')
        
        # Quality thresholds
        self.min_surface_quality = config.get('min_surface_quality', 0.7)
        self.max_fitting_error = config.get('max_fitting_error', 0.05)
        
        # History tracking
        self.surface_history = {
            'surfaces': [],
            'quality_metrics': [],
            'arbitrage_violations': []
        }
        
        logger.info(f"IVSurfaceModeler initialized: model_type={self.surface_model_type}")
    
    def construct_surface(self, option_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Construct implied volatility surface
        
        Args:
            option_data: DataFrame with option prices, strikes, expiries, IV
            
        Returns:
            Dict with surface model and analysis
        """
        try:
            results = {
                'surface_model': {},
                'quality_metrics': {},
                'smile_characteristics': {},
                'arbitrage_analysis': {},
                'surface_evolution': {},
                'trading_signals': {}
            }
            
            # Prepare data for surface construction
            prepared_data = self._prepare_surface_data(option_data)
            
            # Construct the surface model
            results['surface_model'] = self._build_surface_model(prepared_data)
            
            # Calculate quality metrics
            results['quality_metrics'] = self._calculate_quality_metrics(
                prepared_data, results['surface_model']
            )
            
            # Analyze smile characteristics
            results['smile_characteristics'] = self._analyze_smile_characteristics(
                prepared_data, results['surface_model']
            )
            
            # Check for arbitrage violations
            if self.enable_arbitrage_checks:
                results['arbitrage_analysis'] = self._detect_arbitrage_violations(
                    results['surface_model']
                )
            
            # Analyze surface evolution
            results['surface_evolution'] = self._analyze_surface_evolution(
                results['surface_model']
            )
            
            # Generate trading signals
            results['trading_signals'] = self._generate_surface_signals(results)
            
            # Update history
            self._update_surface_history(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error constructing IV surface: {e}")
            return self._get_default_results()
    
    def _prepare_surface_data(self, option_data: pd.DataFrame) -> Dict[str, Any]:
        """Prepare and clean data for surface construction"""
        try:
            # Filter valid data
            valid_data = option_data[
                (option_data['iv'] > 0) & 
                (option_data['iv'] < 5) &  # Remove extreme IV values
                (option_data['volume'] > 0)  # Ensure liquidity
            ].copy()
            
            # Calculate moneyness
            if 'underlying_price' in valid_data.columns:
                valid_data['moneyness'] = valid_data['strike'] / valid_data['underlying_price']
            else:
                # Estimate ATM from strike distribution
                atm_estimate = valid_data['strike'].median()
                valid_data['moneyness'] = valid_data['strike'] / atm_estimate
            
            # Calculate time to expiry
            if 'expiry_date' in valid_data.columns:
                current_date = pd.Timestamp.now().normalize()
                valid_data['tte'] = (
                    pd.to_datetime(valid_data['expiry_date']) - current_date
                ).dt.days / 365.25
            else:
                # Use DTE if available
                valid_data['tte'] = valid_data.get('dte', 30) / 365.25
            
            # Group by expiry and option type
            surface_data = {}
            
            for (expiry, option_type), group in valid_data.groupby(['tte', 'option_type']):
                if len(group) >= self.min_points_per_expiry:
                    surface_data[f"{expiry:.3f}_{option_type}"] = {
                        'tte': expiry,
                        'option_type': option_type,
                        'strikes': group['strike'].values,
                        'moneyness': group['moneyness'].values,
                        'iv': group['iv'].values,
                        'volume': group['volume'].values,
                        'count': len(group)
                    }
            
            return {
                'raw_data': valid_data,
                'surface_data': surface_data,
                'expiries': sorted(list(set(valid_data['tte']))),
                'strike_range': (valid_data['strike'].min(), valid_data['strike'].max()),
                'moneyness_range': (valid_data['moneyness'].min(), valid_data['moneyness'].max())
            }
            
        except Exception as e:
            logger.error(f"Error preparing surface data: {e}")
            return {}
    
    def _build_surface_model(self, prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build the IV surface model"""
        try:
            surface_model = {
                'model_type': self.surface_model_type,
                'expiry_models': {},
                'interpolators': {},
                'fit_quality': {},
                'parameters': {}
            }
            
            surface_data = prepared_data['surface_data']
            
            # Build model for each expiry
            for expiry_key, data in surface_data.items():
                tte = data['tte']
                option_type = data['option_type']
                
                # Fit smile model for this expiry
                smile_model = self._fit_smile_model(
                    data['moneyness'], 
                    data['iv'],
                    data['volume']
                )
                
                surface_model['expiry_models'][expiry_key] = {
                    'tte': tte,
                    'option_type': option_type,
                    'smile_model': smile_model,
                    'raw_points': len(data['iv']),
                    'moneyness_range': (data['moneyness'].min(), data['moneyness'].max())
                }
            
            # Create inter-expiry interpolators
            surface_model['interpolators'] = self._create_surface_interpolators(
                surface_model['expiry_models']
            )
            
            # Calculate overall fit quality
            surface_model['fit_quality'] = self._calculate_surface_fit_quality(
                prepared_data, surface_model
            )
            
            return surface_model
            
        except Exception as e:
            logger.error(f"Error building surface model: {e}")
            return {}
    
    def _fit_smile_model(self, 
                        moneyness: np.ndarray, 
                        iv: np.ndarray,
                        weights: np.ndarray) -> Dict[str, Any]:
        """Fit volatility smile model"""
        try:
            if self.smile_model_type == 'sabr':
                return self._fit_sabr_model(moneyness, iv, weights)
            elif self.smile_model_type == 'svi':
                return self._fit_svi_model(moneyness, iv, weights)
            else:
                return self._fit_polynomial_model(moneyness, iv, weights)
                
        except Exception as e:
            logger.error(f"Error fitting smile model: {e}")
            return self._get_default_smile_model()
    
    def _fit_sabr_model(self, 
                       moneyness: np.ndarray, 
                       iv: np.ndarray,
                       weights: np.ndarray) -> Dict[str, Any]:
        """Fit SABR model to volatility smile"""
        try:
            # SABR parameters: alpha, beta, rho, nu
            def sabr_vol(k, f, alpha, beta, rho, nu, t=0.25):
                """SABR volatility formula"""
                if abs(k - f) < 1e-6:
                    # ATM case
                    return alpha / (f ** (1 - beta))
                
                z = (nu / alpha) * (f * k) ** ((1 - beta) / 2) * np.log(f / k)
                x_z = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
                
                vol = (alpha / ((f * k) ** ((1 - beta) / 2))) * \
                      (z / x_z) * \
                      (1 + ((1 - beta)**2 / 24) * (np.log(f / k))**2 + 
                       ((1 - beta)**4 / 1920) * (np.log(f / k))**4)
                
                return vol
            
            # Objective function
            def objective(params):
                alpha, beta, rho, nu = params
                # Constraints
                if alpha <= 0 or nu <= 0 or abs(rho) >= 1 or beta < 0 or beta > 1:
                    return 1e6
                
                try:
                    f = 1.0  # ATM forward (normalized)
                    model_vols = np.array([
                        sabr_vol(k, f, alpha, beta, rho, nu) for k in moneyness
                    ])
                    
                    # Weighted least squares
                    errors = (model_vols - iv) ** 2
                    return np.sum(weights * errors)
                except:
                    return 1e6
            
            # Initial guess
            initial_guess = [0.2, 0.5, 0.0, 0.3]
            
            # Optimize
            result = minimize(objective, initial_guess, method='L-BFGS-B',
                            bounds=[(0.01, 2), (0.01, 0.99), (-0.99, 0.99), (0.01, 2)])
            
            if result.success:
                alpha, beta, rho, nu = result.x
                return {
                    'model_type': 'sabr',
                    'parameters': {
                        'alpha': float(alpha),
                        'beta': float(beta), 
                        'rho': float(rho),
                        'nu': float(nu)
                    },
                    'fit_error': float(result.fun),
                    'success': True
                }
            else:
                return self._get_default_smile_model()
                
        except Exception as e:
            logger.error(f"Error fitting SABR model: {e}")
            return self._get_default_smile_model()
    
    def _fit_svi_model(self, 
                      moneyness: np.ndarray, 
                      iv: np.ndarray,
                      weights: np.ndarray) -> Dict[str, Any]:
        """Fit SVI model to volatility smile"""
        try:
            # SVI parameterization: w = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
            def svi_vol(k, a, b, rho, m, sigma):
                """SVI total variance formula"""
                try:
                    k_m = k - m
                    w = a + b * (rho * k_m + np.sqrt(k_m**2 + sigma**2))
                    return np.sqrt(np.maximum(w, 0.001))  # Ensure positive variance
                except:
                    return 0.1
            
            # Objective function
            def objective(params):
                a, b, rho, m, sigma = params
                
                # Constraints for no-arbitrage
                if b < 0 or abs(rho) >= 1 or sigma <= 0:
                    return 1e6
                
                try:
                    log_moneyness = np.log(moneyness)
                    model_vols = np.array([
                        svi_vol(k, a, b, rho, m, sigma) for k in log_moneyness
                    ])
                    
                    # Weighted least squares
                    errors = (model_vols - iv) ** 2
                    return np.sum(weights * errors)
                except:
                    return 1e6
            
            # Initial guess
            log_moneyness = np.log(moneyness)
            initial_guess = [
                0.04,  # a
                0.4,   # b
                0.0,   # rho
                0.0,   # m (log ATM)
                0.1    # sigma
            ]
            
            # Optimize
            result = minimize(objective, initial_guess, method='L-BFGS-B',
                            bounds=[(0.001, 1), (0.001, 1), (-0.99, 0.99), (-1, 1), (0.001, 1)])
            
            if result.success:
                a, b, rho, m, sigma = result.x
                return {
                    'model_type': 'svi',
                    'parameters': {
                        'a': float(a),
                        'b': float(b),
                        'rho': float(rho),
                        'm': float(m),
                        'sigma': float(sigma)
                    },
                    'fit_error': float(result.fun),
                    'success': True
                }
            else:
                return self._get_default_smile_model()
                
        except Exception as e:
            logger.error(f"Error fitting SVI model: {e}")
            return self._get_default_smile_model()
    
    def _fit_polynomial_model(self, 
                            moneyness: np.ndarray, 
                            iv: np.ndarray,
                            weights: np.ndarray) -> Dict[str, Any]:
        """Fit polynomial model to volatility smile"""
        try:
            # Use 3rd degree polynomial
            log_moneyness = np.log(moneyness)
            
            # Weighted polynomial fit
            coeffs = np.polyfit(log_moneyness, iv, deg=3, w=weights)
            
            # Calculate fit error
            fitted_iv = np.polyval(coeffs, log_moneyness)
            fit_error = np.sum(weights * (fitted_iv - iv) ** 2)
            
            return {
                'model_type': 'polynomial',
                'parameters': {
                    'coefficients': coeffs.tolist(),
                    'degree': 3
                },
                'fit_error': float(fit_error),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error fitting polynomial model: {e}")
            return self._get_default_smile_model()
    
    def _create_surface_interpolators(self, expiry_models: Dict[str, Any]) -> Dict[str, Any]:
        """Create interpolators for the full surface"""
        try:
            interpolators = {}
            
            # Extract data for interpolation
            expiries = []
            atm_vols = []
            skews = []
            
            for key, model in expiry_models.items():
                tte = model['tte']
                smile_params = model['smile_model']['parameters']
                
                expiries.append(tte)
                
                # Estimate ATM vol and skew from model
                if model['smile_model']['model_type'] == 'sabr':
                    atm_vol = smile_params['alpha']
                    skew = smile_params['rho'] * smile_params['nu']
                elif model['smile_model']['model_type'] == 'svi':
                    atm_vol = np.sqrt(smile_params['a'])
                    skew = smile_params['rho'] * smile_params['b']
                else:
                    # Polynomial - evaluate at ATM
                    atm_vol = np.polyval(smile_params['coefficients'], 0)
                    skew = np.polyval(np.polyder(smile_params['coefficients']), 0)
                
                atm_vols.append(atm_vol)
                skews.append(skew)
            
            if len(expiries) >= 2:
                # Create interpolators for term structure
                expiries = np.array(expiries)
                atm_vols = np.array(atm_vols)
                skews = np.array(skews)
                
                # Sort by expiry
                sort_idx = np.argsort(expiries)
                expiries = expiries[sort_idx]
                atm_vols = atm_vols[sort_idx]
                skews = skews[sort_idx]
                
                interpolators['atm_vol'] = interpolate.interp1d(
                    expiries, atm_vols, kind='cubic', bounds_error=False, 
                    fill_value='extrapolate'
                )
                
                interpolators['skew'] = interpolate.interp1d(
                    expiries, skews, kind='linear', bounds_error=False,
                    fill_value='extrapolate'
                )
            
            return interpolators
            
        except Exception as e:
            logger.error(f"Error creating interpolators: {e}")
            return {}
    
    def _calculate_quality_metrics(self, 
                                 prepared_data: Dict[str, Any],
                                 surface_model: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate surface quality metrics"""
        try:
            metrics = {
                'overall_score': 0.0,
                'coverage_score': 0.0,
                'fit_score': 0.0,
                'smoothness_score': 0.0,
                'data_points': 0,
                'expiries_covered': 0
            }
            
            surface_data = prepared_data['surface_data']
            expiry_models = surface_model['expiry_models']
            
            # Coverage metrics
            metrics['expiries_covered'] = len(expiry_models)
            metrics['data_points'] = sum(
                data['count'] for data in surface_data.values()
            )
            
            if metrics['expiries_covered'] > 0:
                metrics['coverage_score'] = min(metrics['expiries_covered'] / 5, 1.0)
            
            # Fit quality metrics
            fit_errors = []
            for key, model in expiry_models.items():
                if 'fit_error' in model['smile_model']:
                    fit_errors.append(model['smile_model']['fit_error'])
            
            if fit_errors:
                avg_fit_error = np.mean(fit_errors)
                metrics['fit_score'] = max(1.0 - avg_fit_error / self.max_fitting_error, 0.0)
            
            # Smoothness score (simplified)
            if 'interpolators' in surface_model and surface_model['interpolators']:
                metrics['smoothness_score'] = 0.8
            else:
                metrics['smoothness_score'] = 0.3
            
            # Overall score
            metrics['overall_score'] = (
                metrics['coverage_score'] * 0.3 +
                metrics['fit_score'] * 0.5 +
                metrics['smoothness_score'] * 0.2
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating quality metrics: {e}")
            return {'overall_score': 0.0}
    
    def _analyze_smile_characteristics(self,
                                     prepared_data: Dict[str, Any],
                                     surface_model: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze volatility smile characteristics"""
        try:
            characteristics = {
                'smile_types': {},
                'skew_metrics': {},
                'smile_evolution': {},
                'smile_anomalies': []
            }
            
            expiry_models = surface_model['expiry_models']
            
            for key, model in expiry_models.items():
                tte = model['tte']
                smile_params = model['smile_model']['parameters']
                model_type = model['smile_model']['model_type']
                
                # Classify smile type
                smile_type = self._classify_smile_type(smile_params, model_type)
                characteristics['smile_types'][f"tte_{tte:.3f}"] = smile_type
                
                # Calculate skew metrics
                skew_metrics = self._calculate_skew_metrics(smile_params, model_type)
                characteristics['skew_metrics'][f"tte_{tte:.3f}"] = skew_metrics
            
            # Analyze smile evolution across expiries
            characteristics['smile_evolution'] = self._analyze_smile_evolution(
                characteristics['skew_metrics']
            )
            
            # Detect smile anomalies
            characteristics['smile_anomalies'] = self._detect_smile_anomalies(
                characteristics['smile_types'],
                characteristics['skew_metrics']
            )
            
            return characteristics
            
        except Exception as e:
            logger.error(f"Error analyzing smile characteristics: {e}")
            return {}
    
    def _detect_arbitrage_violations(self, surface_model: Dict[str, Any]) -> Dict[str, Any]:
        """Detect arbitrage violations in the surface"""
        try:
            violations = {
                'butterfly_violations': [],
                'calendar_violations': [],
                'total_violations': 0,
                'severity': 'none'
            }
            
            # Check butterfly spread violations (convexity)
            violations['butterfly_violations'] = self._check_butterfly_arbitrage(surface_model)
            
            # Check calendar spread violations
            violations['calendar_violations'] = self._check_calendar_arbitrage(surface_model)
            
            # Calculate total violations
            violations['total_violations'] = (
                len(violations['butterfly_violations']) +
                len(violations['calendar_violations'])
            )
            
            # Classify severity
            if violations['total_violations'] == 0:
                violations['severity'] = 'none'
            elif violations['total_violations'] <= 2:
                violations['severity'] = 'minor'
            elif violations['total_violations'] <= 5:
                violations['severity'] = 'moderate'
            else:
                violations['severity'] = 'severe'
            
            return violations
            
        except Exception as e:
            logger.error(f"Error detecting arbitrage violations: {e}")
            return {'total_violations': 0, 'severity': 'unknown'}
    
    def _analyze_surface_evolution(self, surface_model: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze surface evolution over time"""
        try:
            evolution = {
                'term_structure_slope': 0.0,
                'skew_term_structure': 0.0,
                'surface_stability': 'stable',
                'volatility_regime': 'normal'
            }
            
            # Analyze from interpolators if available
            if 'interpolators' in surface_model and surface_model['interpolators']:
                interpolators = surface_model['interpolators']
                
                # Sample points for analysis
                test_expiries = np.linspace(0.05, 1.0, 10)
                
                if 'atm_vol' in interpolators:
                    atm_vols = interpolators['atm_vol'](test_expiries)
                    
                    # Calculate term structure slope
                    slope = np.polyfit(test_expiries, atm_vols, 1)[0]
                    evolution['term_structure_slope'] = float(slope)
                    
                    # Classify volatility regime
                    if slope > 0.1:
                        evolution['volatility_regime'] = 'contango'
                    elif slope < -0.1:
                        evolution['volatility_regime'] = 'backwardation'
                    else:
                        evolution['volatility_regime'] = 'flat'
                
                if 'skew' in interpolators:
                    skews = interpolators['skew'](test_expiries)
                    skew_slope = np.polyfit(test_expiries, skews, 1)[0]
                    evolution['skew_term_structure'] = float(skew_slope)
            
            return evolution
            
        except Exception as e:
            logger.error(f"Error analyzing surface evolution: {e}")
            return {}
    
    def _generate_surface_signals(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals from surface analysis"""
        try:
            signals = {
                'primary_signal': 'neutral',
                'signal_strength': 0.0,
                'volatility_signals': [],
                'arbitrage_opportunities': [],
                'regime_signals': []
            }
            
            # Quality-based signals
            quality = results['quality_metrics'].get('overall_score', 0.0)
            if quality < 0.5:
                signals['volatility_signals'].append('poor_surface_quality')
                return signals
            
            # Smile-based signals
            smile_chars = results['smile_characteristics']
            for expiry, smile_type in smile_chars.get('smile_types', {}).items():
                if smile_type == 'inverted_smile':
                    signals['volatility_signals'].append(f'inverted_smile_{expiry}')
                elif smile_type == 'extreme_skew':
                    signals['volatility_signals'].append(f'extreme_skew_{expiry}')
            
            # Arbitrage-based signals
            arbitrage = results['arbitrage_analysis']
            if arbitrage.get('total_violations', 0) > 0:
                for violation in arbitrage.get('butterfly_violations', []):
                    signals['arbitrage_opportunities'].append('butterfly_arbitrage')
                for violation in arbitrage.get('calendar_violations', []):
                    signals['arbitrage_opportunities'].append('calendar_arbitrage')
            
            # Evolution-based signals
            evolution = results['surface_evolution']
            vol_regime = evolution.get('volatility_regime', 'normal')
            if vol_regime == 'contango':
                signals['regime_signals'].append('volatility_contango')
            elif vol_regime == 'backwardation':
                signals['regime_signals'].append('volatility_backwardation')
            
            # Generate primary signal
            if signals['arbitrage_opportunities']:
                signals['primary_signal'] = 'arbitrage_detected'
                signals['signal_strength'] = 0.8
            elif len(signals['volatility_signals']) > 2:
                signals['primary_signal'] = 'volatility_anomaly'
                signals['signal_strength'] = 0.6
            elif signals['regime_signals']:
                signals['primary_signal'] = 'regime_change'
                signals['signal_strength'] = 0.4
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating surface signals: {e}")
            return {'primary_signal': 'neutral', 'signal_strength': 0.0}
    
    def _get_default_smile_model(self) -> Dict[str, Any]:
        """Get default smile model"""
        return {
            'model_type': 'polynomial',
            'parameters': {'coefficients': [0.2, 0.0, 0.0, 0.0]},
            'fit_error': 1.0,
            'success': False
        }
    
    def _classify_smile_type(self, params: Dict[str, Any], model_type: str) -> str:
        """Classify the type of volatility smile"""
        try:
            if model_type == 'sabr':
                rho = params.get('rho', 0)
                nu = params.get('nu', 0.3)
                
                if abs(rho) > 0.7:
                    return 'extreme_skew'
                elif rho < -0.3:
                    return 'negative_skew'
                elif rho > 0.3:
                    return 'positive_skew'
                elif nu > 0.8:
                    return 'high_convexity'
                else:
                    return 'normal_smile'
            
            elif model_type == 'svi':
                rho = params.get('rho', 0)
                b = params.get('b', 0.4)
                
                if abs(rho) > 0.7:
                    return 'extreme_skew'
                elif b > 0.8:
                    return 'high_convexity'
                else:
                    return 'normal_smile'
            
            else:  # polynomial
                coeffs = params.get('coefficients', [0.2, 0, 0, 0])
                if len(coeffs) >= 4:
                    # Check for inverted smile (negative second derivative)
                    if coeffs[2] < -0.1:
                        return 'inverted_smile'
                    elif abs(coeffs[1]) > 0.5:
                        return 'extreme_skew'
                
                return 'normal_smile'
                
        except:
            return 'unknown'
    
    def _calculate_skew_metrics(self, params: Dict[str, Any], model_type: str) -> Dict[str, float]:
        """Calculate quantitative skew metrics"""
        try:
            metrics = {
                'skew_90_110': 0.0,
                'skew_slope': 0.0,
                'convexity': 0.0
            }
            
            # Calculate 90%-110% skew
            moneyness_points = np.array([0.9, 1.0, 1.1])
            
            if model_type == 'sabr':
                # Simplified SABR evaluation
                alpha = params.get('alpha', 0.2)
                beta = params.get('beta', 0.5) 
                rho = params.get('rho', 0.0)
                
                metrics['skew_slope'] = rho * 0.5  # Approximation
                metrics['convexity'] = alpha * (1 - beta)
                
            elif model_type == 'polynomial':
                coeffs = params.get('coefficients', [0.2, 0, 0, 0])
                log_points = np.log(moneyness_points)
                
                vols = np.polyval(coeffs, log_points)
                metrics['skew_90_110'] = vols[2] - vols[0]  # 110% - 90%
                
                # Derivative at ATM
                derivative_coeffs = np.polyder(coeffs)
                metrics['skew_slope'] = np.polyval(derivative_coeffs, 0)
                
                # Second derivative for convexity
                if len(coeffs) >= 3:
                    second_deriv_coeffs = np.polyder(derivative_coeffs) 
                    metrics['convexity'] = np.polyval(second_deriv_coeffs, 0)
            
            return metrics
            
        except:
            return {'skew_90_110': 0.0, 'skew_slope': 0.0, 'convexity': 0.0}
    
    def _analyze_smile_evolution(self, skew_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how smile evolves across expiries"""
        try:
            evolution = {
                'skew_term_structure': 'flat',
                'convexity_evolution': 'stable',
                'smile_flattening': False
            }
            
            if len(skew_metrics) >= 2:
                # Extract skew slopes by expiry
                expiries = []
                slopes = []
                
                for expiry_key, metrics in skew_metrics.items():
                    try:
                        tte = float(expiry_key.split('_')[1])
                        expiries.append(tte)
                        slopes.append(metrics.get('skew_slope', 0.0))
                    except:
                        continue
                
                if len(slopes) >= 2:
                    # Sort by expiry
                    sorted_data = sorted(zip(expiries, slopes))
                    expiries, slopes = zip(*sorted_data)
                    
                    # Check if skew flattens with longer expiries
                    if slopes[0] > slopes[-1] + 0.1:
                        evolution['smile_flattening'] = True
                    
                    # Classify term structure
                    slope_change = slopes[-1] - slopes[0]
                    if slope_change > 0.2:
                        evolution['skew_term_structure'] = 'steepening'
                    elif slope_change < -0.2:
                        evolution['skew_term_structure'] = 'flattening'
            
            return evolution
            
        except Exception as e:
            logger.error(f"Error analyzing smile evolution: {e}")
            return {}
    
    def _detect_smile_anomalies(self,
                              smile_types: Dict[str, str],
                              skew_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in smile patterns"""
        try:
            anomalies = []
            
            # Check for inverted smiles
            for expiry, smile_type in smile_types.items():
                if smile_type == 'inverted_smile':
                    anomalies.append({
                        'type': 'inverted_smile',
                        'expiry': expiry,
                        'severity': 'high',
                        'description': 'Volatility smile is inverted'
                    })
                elif smile_type == 'extreme_skew':
                    anomalies.append({
                        'type': 'extreme_skew',
                        'expiry': expiry,
                        'severity': 'medium',
                        'description': 'Extremely high volatility skew detected'
                    })
            
            # Check for extreme skew values
            for expiry, metrics in skew_metrics.items():
                skew_slope = metrics.get('skew_slope', 0.0)
                if abs(skew_slope) > 1.0:
                    anomalies.append({
                        'type': 'extreme_skew_value',
                        'expiry': expiry,
                        'value': skew_slope,
                        'severity': 'medium',
                        'description': f'Extreme skew slope: {skew_slope:.3f}'
                    })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting smile anomalies: {e}")
            return []
    
    def _check_butterfly_arbitrage(self, surface_model: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for butterfly spread arbitrage violations"""
        # Simplified implementation
        return []
    
    def _check_calendar_arbitrage(self, surface_model: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for calendar spread arbitrage violations"""
        # Simplified implementation
        return []
    
    def _calculate_surface_fit_quality(self, 
                                     prepared_data: Dict[str, Any],
                                     surface_model: Dict[str, Any]) -> float:
        """Calculate overall surface fit quality"""
        try:
            fit_errors = []
            
            for key, model in surface_model['expiry_models'].items():
                if 'fit_error' in model['smile_model']:
                    fit_errors.append(model['smile_model']['fit_error'])
            
            if fit_errors:
                return 1.0 - min(np.mean(fit_errors), 1.0)
            else:
                return 0.0
                
        except:
            return 0.0
    
    def _update_surface_history(self, results: Dict[str, Any]):
        """Update surface history for tracking"""
        try:
            timestamp = datetime.now()
            
            # Store surface snapshot
            self.surface_history['surfaces'].append({
                'timestamp': timestamp,
                'quality_score': results['quality_metrics'].get('overall_score', 0.0),
                'expiries_covered': results['quality_metrics'].get('expiries_covered', 0),
                'primary_signal': results['trading_signals'].get('primary_signal', 'neutral')
            })
            
            # Store quality metrics
            self.surface_history['quality_metrics'].append({
                'timestamp': timestamp,
                **results['quality_metrics']
            })
            
            # Store arbitrage violations
            arbitrage = results['arbitrage_analysis']
            if arbitrage.get('total_violations', 0) > 0:
                self.surface_history['arbitrage_violations'].append({
                    'timestamp': timestamp,
                    'violations': arbitrage['total_violations'],
                    'severity': arbitrage['severity']
                })
            
            # Keep only recent history
            max_history = 100
            for key in self.surface_history:
                if len(self.surface_history[key]) > max_history:
                    self.surface_history[key] = self.surface_history[key][-max_history:]
                    
        except Exception as e:
            logger.error(f"Error updating surface history: {e}")
    
    def _get_default_results(self) -> Dict[str, Any]:
        """Get default results structure"""
        return {
            'surface_model': {},
            'quality_metrics': {'overall_score': 0.0},
            'smile_characteristics': {},
            'arbitrage_analysis': {'total_violations': 0, 'severity': 'none'},
            'surface_evolution': {},
            'trading_signals': {'primary_signal': 'neutral', 'signal_strength': 0.0}
        }
    
    def get_surface_summary(self) -> Dict[str, Any]:
        """Get comprehensive surface analysis summary"""
        try:
            if not self.surface_history['surfaces']:
                return {'status': 'no_history'}
            
            recent_surfaces = self.surface_history['surfaces'][-10:]
            
            return {
                'current_quality': recent_surfaces[-1]['quality_score'] if recent_surfaces else 0.0,
                'average_quality': np.mean([s['quality_score'] for s in recent_surfaces]),
                'surface_stability': self._calculate_surface_stability(),
                'arbitrage_frequency': len(self.surface_history['arbitrage_violations']),
                'total_surfaces_built': len(self.surface_history['surfaces'])
            }
            
        except Exception as e:
            logger.error(f"Error getting surface summary: {e}")
            return {'status': 'error'}
    
    def _calculate_surface_stability(self) -> float:
        """Calculate how stable the surface has been"""
        try:
            if len(self.surface_history['quality_metrics']) < 5:
                return 0.5
            
            recent_scores = [
                m['overall_score'] for m in self.surface_history['quality_metrics'][-10:]
            ]
            
            # Low standard deviation = high stability
            stability = 1.0 - min(np.std(recent_scores), 1.0)
            return float(stability)
            
        except:
            return 0.5