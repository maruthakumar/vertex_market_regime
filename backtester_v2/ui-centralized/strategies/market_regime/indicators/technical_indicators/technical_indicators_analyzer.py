"""
Technical Indicators Analyzer - Main Orchestrator
================================================

Orchestrates all technical indicators for both option and underlying data,
performs indicator fusion, and provides comprehensive technical analysis.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

# Import option-based indicators
from .option_based import OptionRSI, OptionMACD, OptionBollinger, OptionVolumeFlow

# Import underlying-based indicators
from .underlying_based import PriceRSI, PriceMACD, PriceBollinger, TrendStrength

# Import composite analysis
from .composite import IndicatorFusion, RegimeClassifier

logger = logging.getLogger(__name__)


class TechnicalIndicatorsAnalyzer:
    """
    Main orchestrator for Technical Indicators V2
    
    Manages all technical indicator calculations, fusion,
    and regime classification for comprehensive market analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Technical Indicators Analyzer"""
        self.config = config
        
        # Initialize option-based indicators
        self.option_rsi = OptionRSI(config.get('option_rsi_config', {}))
        self.option_macd = OptionMACD(config.get('option_macd_config', {}))
        self.option_bollinger = OptionBollinger(config.get('option_bollinger_config', {}))
        self.option_volume_flow = OptionVolumeFlow(config.get('option_volume_flow_config', {}))
        
        # Initialize underlying-based indicators
        self.price_rsi = PriceRSI(config.get('price_rsi_config', {}))
        self.price_macd = PriceMACD(config.get('price_macd_config', {}))
        self.price_bollinger = PriceBollinger(config.get('price_bollinger_config', {}))
        self.trend_strength = TrendStrength(config.get('trend_strength_config', {}))
        
        # Initialize composite analysis
        self.indicator_fusion = IndicatorFusion(config.get('fusion_config', {}))
        self.regime_classifier = RegimeClassifier(config.get('regime_config', {}))
        
        # Performance tracking
        self.performance_metrics = {
            'calculations': 0,
            'errors': 0,
            'avg_calculation_time': 0,
            'component_health': {}
        }
        
        logger.info("TechnicalIndicatorsAnalyzer initialized with modular architecture")
    
    def analyze(self,
               option_data: pd.DataFrame,
               underlying_data: pd.DataFrame,
               context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive technical analysis
        
        Args:
            option_data: DataFrame with option market data
            underlying_data: DataFrame with underlying asset data
            context: Optional context information
            
        Returns:
            Dict with complete technical analysis results
        """
        start_time = datetime.now()
        
        try:
            results = {
                'timestamp': datetime.now(),
                'option_indicators': {},
                'underlying_indicators': {},
                'fusion_analysis': {},
                'regime_classification': {},
                'signals': {},
                'recommendations': {},
                'health_status': {}
            }
            
            # Calculate option-based indicators
            results['option_indicators'] = self._calculate_option_indicators(option_data)
            
            # Calculate underlying-based indicators
            results['underlying_indicators'] = self._calculate_underlying_indicators(underlying_data)
            
            # Perform indicator fusion
            results['fusion_analysis'] = self.indicator_fusion.fuse_indicators(
                results['option_indicators'],
                results['underlying_indicators']
            )
            
            # Classify market regime
            results['regime_classification'] = self.regime_classifier.classify_regime(
                {
                    'option': results['option_indicators'],
                    'underlying': results['underlying_indicators']
                },
                results['fusion_analysis']
            )
            
            # Generate consolidated signals
            results['signals'] = self._generate_consolidated_signals(results)
            
            # Generate recommendations
            results['recommendations'] = self._generate_recommendations(results)
            
            # Check component health
            results['health_status'] = self._check_component_health()
            
            # Update performance metrics
            self._update_performance_metrics(start_time)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in technical analysis: {e}")
            self.performance_metrics['errors'] += 1
            return self._get_default_results()
    
    def _calculate_option_indicators(self, option_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all option-based indicators"""
        try:
            indicators = {}
            
            # RSI
            try:
                indicators['rsi'] = self.option_rsi.calculate_option_rsi(option_data)
                self.performance_metrics['component_health']['option_rsi'] = 'healthy'
            except Exception as e:
                logger.error(f"Error calculating option RSI: {e}")
                indicators['rsi'] = {'status': 'error', 'error': str(e)}
                self.performance_metrics['component_health']['option_rsi'] = 'error'
            
            # MACD
            try:
                indicators['macd'] = self.option_macd.calculate_option_macd(option_data)
                self.performance_metrics['component_health']['option_macd'] = 'healthy'
            except Exception as e:
                logger.error(f"Error calculating option MACD: {e}")
                indicators['macd'] = {'status': 'error', 'error': str(e)}
                self.performance_metrics['component_health']['option_macd'] = 'error'
            
            # Bollinger Bands
            try:
                indicators['bollinger'] = self.option_bollinger.calculate_option_bollinger(option_data)
                self.performance_metrics['component_health']['option_bollinger'] = 'healthy'
            except Exception as e:
                logger.error(f"Error calculating option Bollinger: {e}")
                indicators['bollinger'] = {'status': 'error', 'error': str(e)}
                self.performance_metrics['component_health']['option_bollinger'] = 'error'
            
            # Volume Flow
            try:
                indicators['volume_flow'] = self.option_volume_flow.analyze_volume_flow(option_data)
                self.performance_metrics['component_health']['option_volume_flow'] = 'healthy'
            except Exception as e:
                logger.error(f"Error calculating volume flow: {e}")
                indicators['volume_flow'] = {'status': 'error', 'error': str(e)}
                self.performance_metrics['component_health']['option_volume_flow'] = 'error'
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating option indicators: {e}")
            return {}
    
    def _calculate_underlying_indicators(self, underlying_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all underlying-based indicators"""
        try:
            indicators = {}
            
            # RSI
            try:
                indicators['rsi'] = self.price_rsi.calculate_price_rsi(underlying_data)
                self.performance_metrics['component_health']['price_rsi'] = 'healthy'
            except Exception as e:
                logger.error(f"Error calculating price RSI: {e}")
                indicators['rsi'] = {'status': 'error', 'error': str(e)}
                self.performance_metrics['component_health']['price_rsi'] = 'error'
            
            # MACD
            try:
                indicators['macd'] = self.price_macd.calculate_price_macd(underlying_data)
                self.performance_metrics['component_health']['price_macd'] = 'healthy'
            except Exception as e:
                logger.error(f"Error calculating price MACD: {e}")
                indicators['macd'] = {'status': 'error', 'error': str(e)}
                self.performance_metrics['component_health']['price_macd'] = 'error'
            
            # Bollinger Bands
            try:
                indicators['bollinger'] = self.price_bollinger.calculate_price_bollinger(underlying_data)
                self.performance_metrics['component_health']['price_bollinger'] = 'healthy'
            except Exception as e:
                logger.error(f"Error calculating price Bollinger: {e}")
                indicators['bollinger'] = {'status': 'error', 'error': str(e)}
                self.performance_metrics['component_health']['price_bollinger'] = 'error'
            
            # Trend Strength
            try:
                indicators['trend_strength'] = self.trend_strength.analyze_trend_strength(underlying_data)
                self.performance_metrics['component_health']['trend_strength'] = 'healthy'
            except Exception as e:
                logger.error(f"Error calculating trend strength: {e}")
                indicators['trend_strength'] = {'status': 'error', 'error': str(e)}
                self.performance_metrics['component_health']['trend_strength'] = 'error'
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating underlying indicators: {e}")
            return {}
    
    def _generate_consolidated_signals(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate consolidated signals from all analysis"""
        try:
            signals = {
                'primary_signal': 'neutral',
                'signal_strength': 0.0,
                'signal_confidence': 0.0,
                'supporting_signals': [],
                'opposing_signals': [],
                'key_levels': {}
            }
            
            # Get fusion signal
            fusion_signal = results['fusion_analysis'].get('fused_signals', {})
            signals['primary_signal'] = fusion_signal.get('composite_signal', 'neutral')
            signals['signal_strength'] = fusion_signal.get('composite_strength', 0.0)
            
            # Get confidence from fusion
            signals['signal_confidence'] = results['fusion_analysis'].get(
                'confidence_scores', {}
            ).get('overall', 0.0)
            
            # Collect supporting signals
            if results['regime_classification'].get('supporting_indicators'):
                signals['supporting_signals'] = results['regime_classification']['supporting_indicators']
            
            # Identify opposing signals (divergences)
            for divergence in results['fusion_analysis'].get('divergences', []):
                signals['opposing_signals'].append(divergence['type'])
            
            # Extract key levels
            signals['key_levels'] = self._extract_key_levels(results)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating consolidated signals: {e}")
            return {'primary_signal': 'neutral', 'signal_strength': 0.0}
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable recommendations"""
        try:
            # Start with fusion recommendations
            recommendations = results['fusion_analysis'].get('recommendations', {}).copy()
            
            # Add regime-based adjustments
            regime = results['regime_classification'].get('current_regime', 'undefined')
            regime_confidence = results['regime_classification'].get('regime_confidence', 0.0)
            
            # Adjust based on regime
            if 'trending' in regime and regime_confidence > 0.7:
                recommendations['strategy'] = 'trend_following'
                recommendations['preferred_indicators'] = ['macd', 'trend_strength']
            elif 'ranging' in regime and regime_confidence > 0.7:
                recommendations['strategy'] = 'mean_reversion'
                recommendations['preferred_indicators'] = ['rsi', 'bollinger']
            elif 'volatile' in regime:
                recommendations['strategy'] = 'volatility_trading'
                recommendations['risk_adjustment'] = 'reduce_position_size'
            
            # Add specific entry/exit recommendations
            recommendations['entry_exit'] = self._generate_entry_exit_recommendations(results)
            
            # Add risk management
            recommendations['risk_management'] = self._generate_risk_recommendations(results)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return {}
    
    def _extract_key_levels(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key price levels from indicators"""
        try:
            key_levels = {
                'resistance': [],
                'support': [],
                'pivot': None
            }
            
            # From Bollinger Bands
            if 'bollinger' in results['underlying_indicators']:
                bollinger = results['underlying_indicators']['bollinger']
                if 'bands' in bollinger:
                    key_levels['resistance'].append({
                        'level': bollinger['bands']['upper'],
                        'source': 'bollinger_upper'
                    })
                    key_levels['support'].append({
                        'level': bollinger['bands']['lower'],
                        'source': 'bollinger_lower'
                    })
                    key_levels['pivot'] = bollinger['bands']['middle']
            
            # From Trend Strength (S/R levels)
            if 'trend_strength' in results['underlying_indicators']:
                sr = results['underlying_indicators']['trend_strength'].get('support_resistance', {})
                if sr.get('nearest_resistance'):
                    key_levels['resistance'].append({
                        'level': sr['nearest_resistance'],
                        'source': 'price_action'
                    })
                if sr.get('nearest_support'):
                    key_levels['support'].append({
                        'level': sr['nearest_support'],
                        'source': 'price_action'
                    })
            
            # Sort levels
            key_levels['resistance'] = sorted(key_levels['resistance'], key=lambda x: x['level'])
            key_levels['support'] = sorted(key_levels['support'], key=lambda x: x['level'], reverse=True)
            
            return key_levels
            
        except Exception as e:
            logger.error(f"Error extracting key levels: {e}")
            return {}
    
    def _generate_entry_exit_recommendations(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate specific entry/exit recommendations"""
        try:
            entry_exit = {
                'entry_conditions': [],
                'exit_conditions': [],
                'stop_loss': None,
                'take_profit': None
            }
            
            signal = results['signals']['primary_signal']
            strength = results['signals']['signal_strength']
            key_levels = results['signals']['key_levels']
            
            # Entry conditions
            if signal in ['bullish', 'strong_buy']:
                entry_exit['entry_conditions'].append('wait_for_pullback_to_support')
                if key_levels.get('support'):
                    entry_exit['entry_conditions'].append(
                        f"enter_near_{key_levels['support'][0]['level']:.2f}"
                    )
            elif signal in ['bearish', 'strong_sell']:
                entry_exit['entry_conditions'].append('wait_for_rally_to_resistance')
                if key_levels.get('resistance'):
                    entry_exit['entry_conditions'].append(
                        f"enter_near_{key_levels['resistance'][0]['level']:.2f}"
                    )
            
            # Exit conditions based on indicators
            if 'rsi' in results['option_indicators']:
                option_rsi = results['option_indicators']['rsi']
                if option_rsi.get('regime') == 'extreme_overbought_regime':
                    entry_exit['exit_conditions'].append('rsi_extreme_overbought')
                elif option_rsi.get('regime') == 'extreme_oversold_regime':
                    entry_exit['exit_conditions'].append('rsi_extreme_oversold')
            
            # Stop loss and take profit
            if key_levels.get('support') and signal in ['bullish', 'strong_buy']:
                entry_exit['stop_loss'] = key_levels['support'][0]['level'] * 0.98
            
            if key_levels.get('resistance') and signal in ['bearish', 'strong_sell']:
                entry_exit['stop_loss'] = key_levels['resistance'][0]['level'] * 1.02
            
            return entry_exit
            
        except Exception as e:
            logger.error(f"Error generating entry/exit recommendations: {e}")
            return {}
    
    def _generate_risk_recommendations(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate risk management recommendations"""
        try:
            risk_mgmt = {
                'position_size': 1.0,
                'risk_level': 'medium',
                'warnings': [],
                'hedging_suggestion': None
            }
            
            # Adjust based on confidence
            confidence = results['signals']['signal_confidence']
            if confidence < 0.5:
                risk_mgmt['position_size'] = 0.5
                risk_mgmt['warnings'].append('low_signal_confidence')
            elif confidence > 0.8:
                risk_mgmt['position_size'] = 1.0
            else:
                risk_mgmt['position_size'] = 0.75
            
            # Check for high volatility
            if 'bollinger' in results['option_indicators']:
                vol_state = results['option_indicators']['bollinger'].get('volatility_state')
                if vol_state in ['extreme_expansion', 'high_volatility']:
                    risk_mgmt['risk_level'] = 'high'
                    risk_mgmt['position_size'] *= 0.7
                    risk_mgmt['warnings'].append('high_volatility')
            
            # Check for divergences
            divergences = results['fusion_analysis'].get('divergences', [])
            if len(divergences) > 1:
                risk_mgmt['risk_level'] = 'high'
                risk_mgmt['warnings'].append('multiple_divergences')
                risk_mgmt['hedging_suggestion'] = 'consider_protective_options'
            
            # Volume flow warnings
            if 'volume_flow' in results['option_indicators']:
                smart_money = results['option_indicators']['volume_flow'].get('smart_money', {})
                if any(sm.get('detected') for sm in smart_money.values()):
                    risk_mgmt['warnings'].append('smart_money_activity')
            
            return risk_mgmt
            
        except Exception as e:
            logger.error(f"Error generating risk recommendations: {e}")
            return {'position_size': 0.5, 'risk_level': 'medium'}
    
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
                health['overall_status'] = 'degraded' if len(unhealthy) < 3 else 'unhealthy'
                health['recommendations'].append(f"Check components: {', '.join(unhealthy)}")
            
            # Check error rate
            if health['error_rate'] > 0.1:
                health['overall_status'] = 'unhealthy'
                health['recommendations'].append("High error rate detected")
            
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
            'option_indicators': {},
            'underlying_indicators': {},
            'fusion_analysis': self.indicator_fusion._get_default_results(),
            'regime_classification': self.regime_classifier._get_default_results(),
            'signals': {'primary_signal': 'neutral', 'signal_strength': 0.0},
            'recommendations': {'action': 'hold'},
            'health_status': {'overall_status': 'error'}
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of technical analysis system"""
        try:
            return {
                'performance_metrics': self.performance_metrics.copy(),
                'option_rsi_summary': self.option_rsi.get_rsi_summary(),
                'option_macd_summary': self.option_macd.get_macd_summary(),
                'option_bollinger_summary': self.option_bollinger.get_bollinger_summary(),
                'option_flow_summary': self.option_volume_flow.get_flow_summary(),
                'price_rsi_summary': self.price_rsi.get_rsi_analysis(),
                'price_macd_summary': self.price_macd.get_macd_analysis(),
                'price_bollinger_summary': self.price_bollinger.get_bollinger_analysis(),
                'trend_strength_summary': self.trend_strength.get_trend_analysis(),
                'fusion_summary': self.indicator_fusion.get_fusion_analysis(),
                'regime_summary': self.regime_classifier.get_regime_analysis()
            }
            
        except Exception as e:
            logger.error(f"Error getting summary: {e}")
            return {'status': 'error', 'error': str(e)}