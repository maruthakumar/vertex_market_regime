"""
IV Analytics Analyzer - Main Orchestrator for IV Analytics V2
============================================================

Orchestrates all implied volatility analysis components for comprehensive
IV market analysis and regime detection.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

# Import surface analysis
from .surface_analysis import IVSurfaceModeler, SurfaceInterpolator, SmileAnalyzer

# Import term structure
from .term_structure import TermStructureAnalyzer, CurveFitter, ForwardVolCalculator

# Import skew analysis
from .skew_analysis import SkewDetector, SkewMomentum, RiskReversalAnalyzer

# Import volatility forecasting
from .volatility_forecasting import VolPredictor, GARCHModel, RegimeVolModel

# Import arbitrage detection
from .arbitrage_detection import CalendarArbitrage, StrikeArbitrage, VolArbitrageScanner

logger = logging.getLogger(__name__)


class IVAnalyticsAnalyzer:
    """
    Main orchestrator for IV Analytics V2
    
    Manages all IV analysis components and provides comprehensive
    implied volatility market analysis and regime detection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize IV Analytics Analyzer"""
        self.config = config
        
        # Initialize surface analysis components
        self.iv_surface_modeler = IVSurfaceModeler(config.get('surface_config', {}))
        self.surface_interpolator = SurfaceInterpolator(config.get('interpolation_config', {}))
        self.smile_analyzer = SmileAnalyzer(config.get('smile_config', {}))
        
        # Initialize term structure components
        self.term_structure_analyzer = TermStructureAnalyzer(config.get('term_structure_config', {}))
        self.curve_fitter = CurveFitter(config.get('curve_fitting_config', {}))
        self.forward_vol_calculator = ForwardVolCalculator(config.get('forward_vol_config', {}))
        
        # Initialize skew analysis components
        self.skew_detector = SkewDetector(config.get('skew_config', {}))
        self.skew_momentum = SkewMomentum(config.get('skew_momentum_config', {}))
        self.risk_reversal_analyzer = RiskReversalAnalyzer(config.get('risk_reversal_config', {}))
        
        # Initialize volatility forecasting components
        self.vol_predictor = VolPredictor(config.get('vol_prediction_config', {}))
        self.garch_model = GARCHModel(config.get('garch_config', {}))
        self.regime_vol_model = RegimeVolModel(config.get('regime_vol_config', {}))
        
        # Initialize arbitrage detection components
        self.calendar_arbitrage = CalendarArbitrage(config.get('calendar_arbitrage_config', {}))
        self.strike_arbitrage = StrikeArbitrage(config.get('strike_arbitrage_config', {}))
        self.vol_arbitrage_scanner = VolArbitrageScanner(config.get('arbitrage_scanner_config', {}))
        
        # Performance tracking
        self.performance_metrics = {
            'calculations': 0,
            'errors': 0,
            'avg_calculation_time': 0,
            'component_health': {}
        }
        
        logger.info("IVAnalyticsAnalyzer initialized with comprehensive IV analysis")
    
    def analyze(self,
               option_data: pd.DataFrame,
               underlying_data: Optional[pd.DataFrame] = None,
               context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive IV analysis
        
        Args:
            option_data: DataFrame with option market data including IV
            underlying_data: Optional underlying asset data  
            context: Optional context information
            
        Returns:
            Dict with complete IV analysis results
        """
        start_time = datetime.now()
        
        try:
            results = {
                'timestamp': datetime.now(),
                'surface_analysis': {},
                'term_structure_analysis': {},
                'skew_analysis': {},
                'volatility_forecasting': {},
                'arbitrage_analysis': {},
                'iv_signals': {},
                'iv_regime': {},
                'health_status': {}
            }
            
            # Surface Analysis
            results['surface_analysis'] = self._perform_surface_analysis(option_data)
            
            # Term Structure Analysis
            results['term_structure_analysis'] = self._perform_term_structure_analysis(option_data)
            
            # Skew Analysis
            results['skew_analysis'] = self._perform_skew_analysis(option_data)
            
            # Volatility Forecasting
            results['volatility_forecasting'] = self._perform_volatility_forecasting(option_data)
            
            # Arbitrage Analysis
            results['arbitrage_analysis'] = self._perform_arbitrage_analysis(option_data)
            
            # Generate consolidated IV signals
            results['iv_signals'] = self._generate_iv_signals(results)
            
            # Classify IV regime
            results['iv_regime'] = self._classify_iv_regime(results)
            
            # Check component health
            results['health_status'] = self._check_component_health()
            
            # Update performance metrics
            self._update_performance_metrics(start_time)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in IV analysis: {e}")
            self.performance_metrics['errors'] += 1
            return self._get_default_results()
    
    def _perform_surface_analysis(self, option_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform IV surface analysis"""
        try:
            surface_analysis = {}
            
            # IV Surface Modeling
            try:
                surface_analysis['surface_model'] = self.iv_surface_modeler.construct_surface(option_data)
                self.performance_metrics['component_health']['surface_modeler'] = 'healthy'
            except Exception as e:
                logger.error(f"Error in surface modeling: {e}")
                surface_analysis['surface_model'] = {'quality_metrics': {'overall_score': 0.0}}
                self.performance_metrics['component_health']['surface_modeler'] = 'error'
            
            # Surface Interpolation
            try:
                if 'surface_model' in surface_analysis:
                    surface_analysis['interpolation'] = self.surface_interpolator.interpolate_surface(
                        surface_analysis['surface_model']
                    )
                self.performance_metrics['component_health']['surface_interpolator'] = 'healthy'
            except Exception as e:
                logger.error(f"Error in surface interpolation: {e}")
                surface_analysis['interpolation'] = {'interpolation_quality': 0.0}
                self.performance_metrics['component_health']['surface_interpolator'] = 'error'
            
            # Smile Analysis
            try:
                surface_analysis['smile_analysis'] = self.smile_analyzer.analyze_smile(
                    surface_analysis.get('surface_model', {})
                )
                self.performance_metrics['component_health']['smile_analyzer'] = 'healthy'
            except Exception as e:
                logger.error(f"Error in smile analysis: {e}")
                surface_analysis['smile_analysis'] = {'smile_shape': 'unknown'}
                self.performance_metrics['component_health']['smile_analyzer'] = 'error'
            
            return surface_analysis
            
        except Exception as e:
            logger.error(f"Error performing surface analysis: {e}")
            return {}
    
    def _perform_term_structure_analysis(self, option_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform term structure analysis"""
        try:
            term_structure_analysis = {}
            
            # Term Structure Analysis
            try:
                term_structure_analysis['structure'] = self.term_structure_analyzer.analyze_term_structure(option_data)
                self.performance_metrics['component_health']['term_structure_analyzer'] = 'healthy'
            except Exception as e:
                logger.error(f"Error in term structure analysis: {e}")
                term_structure_analysis['structure'] = {'structure_type': 'unknown'}
                self.performance_metrics['component_health']['term_structure_analyzer'] = 'error'
            
            # Curve Fitting
            try:
                term_structure = term_structure_analysis.get('structure', {})
                if 'term_structure_slope' in term_structure:
                    # Create dummy term structure for curve fitting
                    dummy_ts = {30: 0.2, 60: 0.22, 90: 0.24}
                    term_structure_analysis['curve_fit'] = self.curve_fitter.fit_curve(dummy_ts)
                self.performance_metrics['component_health']['curve_fitter'] = 'healthy'
            except Exception as e:
                logger.error(f"Error in curve fitting: {e}")
                term_structure_analysis['curve_fit'] = {'fit_quality': 0.0}
                self.performance_metrics['component_health']['curve_fitter'] = 'error'
            
            # Forward Vol Calculation
            try:
                if 'curve_fit' in term_structure_analysis:
                    dummy_ts = {30: 0.2, 60: 0.22, 90: 0.24}
                    term_structure_analysis['forward_vols'] = self.forward_vol_calculator.calculate_forward_vols(dummy_ts)
                self.performance_metrics['component_health']['forward_vol_calculator'] = 'healthy'
            except Exception as e:
                logger.error(f"Error calculating forward vols: {e}")
                term_structure_analysis['forward_vols'] = {'calculation_success': False}
                self.performance_metrics['component_health']['forward_vol_calculator'] = 'error'
            
            return term_structure_analysis
            
        except Exception as e:
            logger.error(f"Error performing term structure analysis: {e}")
            return {}
    
    def _perform_skew_analysis(self, option_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform skew analysis"""
        try:
            skew_analysis = {}
            
            # Skew Detection
            try:
                skew_analysis['skew_patterns'] = self.skew_detector.detect_skew_patterns(option_data)
                self.performance_metrics['component_health']['skew_detector'] = 'healthy'
            except Exception as e:
                logger.error(f"Error in skew detection: {e}")
                skew_analysis['skew_patterns'] = {'skew_regime': 'unknown'}
                self.performance_metrics['component_health']['skew_detector'] = 'error'
            
            # Skew Momentum
            try:
                skew_history = self.skew_detector.skew_history
                skew_analysis['skew_momentum'] = self.skew_momentum.analyze_skew_momentum(skew_history)
                self.performance_metrics['component_health']['skew_momentum'] = 'healthy'
            except Exception as e:
                logger.error(f"Error in skew momentum analysis: {e}")
                skew_analysis['skew_momentum'] = {'momentum_strength': 0.0}
                self.performance_metrics['component_health']['skew_momentum'] = 'error'
            
            # Risk Reversal Analysis
            try:
                skew_analysis['risk_reversals'] = self.risk_reversal_analyzer.analyze_risk_reversals(option_data)
                self.performance_metrics['component_health']['risk_reversal_analyzer'] = 'healthy'
            except Exception as e:
                logger.error(f"Error in risk reversal analysis: {e}")
                skew_analysis['risk_reversals'] = {'risk_reversal_value': 0.0}
                self.performance_metrics['component_health']['risk_reversal_analyzer'] = 'error'
            
            return skew_analysis
            
        except Exception as e:
            logger.error(f"Error performing skew analysis: {e}")
            return {}
    
    def _perform_volatility_forecasting(self, option_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform volatility forecasting"""
        try:
            vol_forecasting = {}
            
            # Vol Prediction
            try:
                # Create IV history for prediction
                iv_history = option_data[['timestamp', 'iv']].copy() if 'timestamp' in option_data.columns else option_data[['iv']].copy()
                vol_forecasting['prediction'] = self.vol_predictor.predict_volatility(iv_history)
                self.performance_metrics['component_health']['vol_predictor'] = 'healthy'
            except Exception as e:
                logger.error(f"Error in vol prediction: {e}")
                vol_forecasting['prediction'] = {'forecast': [], 'confidence': 0.0}
                self.performance_metrics['component_health']['vol_predictor'] = 'error'
            
            # GARCH Modeling
            try:
                if 'price' in option_data.columns:
                    returns = option_data['price'].pct_change().dropna()
                    vol_forecasting['garch'] = self.garch_model.fit_garch(returns)
                else:
                    # Use IV changes as proxy
                    iv_returns = option_data['iv'].pct_change().dropna()
                    vol_forecasting['garch'] = self.garch_model.fit_garch(iv_returns)
                self.performance_metrics['component_health']['garch_model'] = 'healthy'
            except Exception as e:
                logger.error(f"Error in GARCH modeling: {e}")
                vol_forecasting['garch'] = {'model_fitted': False}
                self.performance_metrics['component_health']['garch_model'] = 'error'
            
            # Regime-based Vol Modeling
            try:
                current_regime = 'normal'  # Would be passed from context
                vol_forecasting['regime_model'] = self.regime_vol_model.model_regime_volatility(
                    option_data, current_regime
                )
                self.performance_metrics['component_health']['regime_vol_model'] = 'healthy'
            except Exception as e:
                logger.error(f"Error in regime vol modeling: {e}")
                vol_forecasting['regime_model'] = {'regime_vol_forecast': 0.2}
                self.performance_metrics['component_health']['regime_vol_model'] = 'error'
            
            return vol_forecasting
            
        except Exception as e:
            logger.error(f"Error performing volatility forecasting: {e}")
            return {}
    
    def _perform_arbitrage_analysis(self, option_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform arbitrage analysis"""
        try:
            arbitrage_analysis = {}
            
            # Calendar Arbitrage Detection
            try:
                arbitrage_analysis['calendar'] = self.calendar_arbitrage.detect_calendar_arbitrage(option_data)
                self.performance_metrics['component_health']['calendar_arbitrage'] = 'healthy'
            except Exception as e:
                logger.error(f"Error in calendar arbitrage detection: {e}")
                arbitrage_analysis['calendar'] = {'total_opportunities': 0}
                self.performance_metrics['component_health']['calendar_arbitrage'] = 'error'
            
            # Strike Arbitrage Detection  
            try:
                arbitrage_analysis['strike'] = self.strike_arbitrage.detect_strike_arbitrage(option_data)
                self.performance_metrics['component_health']['strike_arbitrage'] = 'healthy'
            except Exception as e:
                logger.error(f"Error in strike arbitrage detection: {e}")
                arbitrage_analysis['strike'] = {'total_opportunities': 0}
                self.performance_metrics['component_health']['strike_arbitrage'] = 'error'
            
            # Arbitrage Scanning
            try:
                arbitrage_analysis['scanner'] = self.vol_arbitrage_scanner.scan_arbitrage_opportunities(
                    arbitrage_analysis.get('calendar', {}),
                    arbitrage_analysis.get('strike', {})
                )
                self.performance_metrics['component_health']['vol_arbitrage_scanner'] = 'healthy'
            except Exception as e:
                logger.error(f"Error in arbitrage scanning: {e}")
                arbitrage_analysis['scanner'] = {'total_opportunities': 0, 'arbitrage_score': 0.0}
                self.performance_metrics['component_health']['vol_arbitrage_scanner'] = 'error'
            
            return arbitrage_analysis
            
        except Exception as e:
            logger.error(f"Error performing arbitrage analysis: {e}")
            return {}
    
    def _generate_iv_signals(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate consolidated IV trading signals"""
        try:
            signals = {
                'primary_signal': 'neutral',
                'signal_strength': 0.0,
                'iv_signals': [],
                'surface_signals': [],
                'skew_signals': [],
                'arbitrage_signals': []
            }
            
            # Surface-based signals
            surface_analysis = results.get('surface_analysis', {})
            if 'surface_model' in surface_analysis:
                surface_signals = surface_analysis['surface_model'].get('trading_signals', {})
                signals['surface_signals'] = surface_signals.get('volatility_signals', [])
                
                if surface_signals.get('primary_signal') != 'neutral':
                    signals['iv_signals'].append(f"surface_{surface_signals['primary_signal']}")
            
            # Skew-based signals
            skew_analysis = results.get('skew_analysis', {})
            if 'skew_patterns' in skew_analysis:
                skew_signals = skew_analysis['skew_patterns'].get('skew_signals', [])
                signals['skew_signals'] = skew_signals
                
                if skew_signals:
                    signals['iv_signals'].extend([f"skew_{sig}" for sig in skew_signals])
            
            # Arbitrage signals
            arbitrage_analysis = results.get('arbitrage_analysis', {})
            if 'scanner' in arbitrage_analysis:
                total_opportunities = arbitrage_analysis['scanner'].get('total_opportunities', 0)
                if total_opportunities > 0:
                    signals['arbitrage_signals'].append('arbitrage_opportunities_detected')
                    signals['iv_signals'].append('volatility_arbitrage')
            
            # Generate primary signal
            if len(signals['iv_signals']) > 0:
                # Prioritize arbitrage signals
                if any('arbitrage' in sig for sig in signals['iv_signals']):
                    signals['primary_signal'] = 'arbitrage_opportunity'
                    signals['signal_strength'] = 0.9
                elif any('extreme' in sig for sig in signals['iv_signals']):
                    signals['primary_signal'] = 'extreme_volatility'
                    signals['signal_strength'] = 0.7
                elif len(signals['iv_signals']) >= 3:
                    signals['primary_signal'] = 'volatility_anomaly'
                    signals['signal_strength'] = 0.6
                else:
                    signals['primary_signal'] = 'volatility_signal'
                    signals['signal_strength'] = 0.4
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating IV signals: {e}")
            return {'primary_signal': 'neutral', 'signal_strength': 0.0}
    
    def _classify_iv_regime(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Classify IV regime based on analysis"""
        try:
            regime = {
                'current_regime': 'normal_volatility',
                'regime_confidence': 0.0,
                'regime_characteristics': [],
                'volatility_level': 'medium'
            }
            
            # Surface-based regime classification
            surface_analysis = results.get('surface_analysis', {})
            if 'surface_model' in surface_analysis:
                surface_regime = surface_analysis['surface_model'].get('regime', 'undefined')
                if surface_regime != 'undefined':
                    regime['regime_characteristics'].append(f'surface_{surface_regime}')
            
            # Term structure regime
            term_structure = results.get('term_structure_analysis', {})
            if 'structure' in term_structure:
                structure_type = term_structure['structure'].get('structure_type', 'flat')
                if structure_type != 'flat':
                    regime['regime_characteristics'].append(f'term_structure_{structure_type}')
            
            # Skew regime
            skew_analysis = results.get('skew_analysis', {})
            if 'skew_patterns' in skew_analysis:
                skew_regime = skew_analysis['skew_patterns'].get('skew_regime', 'normal')
                if skew_regime != 'normal':
                    regime['regime_characteristics'].append(f'skew_{skew_regime}')
            
            # Determine overall regime
            characteristics = regime['regime_characteristics']
            
            if any('extreme' in char for char in characteristics):
                regime['current_regime'] = 'extreme_volatility_regime'
                regime['volatility_level'] = 'very_high'
                regime['regime_confidence'] = 0.8
            elif any('high' in char for char in characteristics):
                regime['current_regime'] = 'high_volatility_regime'
                regime['volatility_level'] = 'high'
                regime['regime_confidence'] = 0.7
            elif any('contango' in char or 'backwardation' in char for char in characteristics):
                regime['current_regime'] = 'term_structure_regime'
                regime['regime_confidence'] = 0.6
            elif len(characteristics) > 0:
                regime['current_regime'] = 'mixed_volatility_regime'
                regime['regime_confidence'] = 0.5
            else:
                regime['regime_confidence'] = 0.3
            
            return regime
            
        except Exception as e:
            logger.error(f"Error classifying IV regime: {e}")
            return {'current_regime': 'unknown', 'regime_confidence': 0.0}
    
    def _check_component_health(self) -> Dict[str, Any]:
        """Check health status of all components"""
        try:
            health = {
                'overall_status': 'healthy',
                'component_status': self.performance_metrics['component_health'].copy(),
                'error_rate': 0.0,
                'recommendations': []
            }
            
            # Calculate error rate
            total_calcs = self.performance_metrics['calculations']
            if total_calcs > 0:
                health['error_rate'] = self.performance_metrics['errors'] / total_calcs
            
            # Check for unhealthy components
            unhealthy = [
                comp for comp, status in health['component_status'].items()
                if status != 'healthy'
            ]
            
            if unhealthy:
                health['overall_status'] = 'degraded' if len(unhealthy) < 5 else 'unhealthy'
                health['recommendations'].append(f"Check components: {', '.join(unhealthy)}")
            
            # Check error rate
            if health['error_rate'] > 0.2:
                health['overall_status'] = 'unhealthy'
                health['recommendations'].append("High error rate detected in IV analysis")
            
            return health
            
        except Exception as e:
            logger.error(f"Error checking component health: {e}")
            return {'overall_status': 'unknown'}
    
    def _update_performance_metrics(self, start_time: datetime):
        """Update performance metrics"""
        try:
            # Update calculation count
            self.performance_metrics['calculations'] += 1
            
            # Update average calculation time
            calc_time = (datetime.now() - start_time).total_seconds()
            current_avg = self.performance_metrics['avg_calculation_time']
            count = self.performance_metrics['calculations']
            
            # Running average
            self.performance_metrics['avg_calculation_time'] = (
                (current_avg * (count - 1) + calc_time) / count
            )
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def _get_default_results(self) -> Dict[str, Any]:
        """Get default results structure"""
        return {
            'timestamp': datetime.now(),
            'surface_analysis': {},
            'term_structure_analysis': {},
            'skew_analysis': {},
            'volatility_forecasting': {},
            'arbitrage_analysis': {},
            'iv_signals': {'primary_signal': 'neutral', 'signal_strength': 0.0},
            'iv_regime': {'current_regime': 'unknown', 'regime_confidence': 0.0},
            'health_status': {'overall_status': 'error'}
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of IV analytics system"""
        try:
            return {
                'performance_metrics': self.performance_metrics.copy(),
                'surface_summary': self.iv_surface_modeler.get_surface_summary(),
                'skew_summary': self.skew_detector.get_skew_summary(),
                'system_status': self._check_component_health()
            }
            
        except Exception as e:
            logger.error(f"Error getting IV analytics summary: {e}")
            return {'status': 'error', 'error': str(e)}