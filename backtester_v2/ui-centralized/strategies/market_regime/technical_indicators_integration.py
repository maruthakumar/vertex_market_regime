"""
Technical Indicators Integration Manager

This module provides integration between the enhanced technical indicators
and the Excel configuration system, enabling hot-reloading and dynamic
parameter management for the Market Regime Detection System.

Features:
1. Dynamic initialization of technical indicators from Excel config
2. Hot-reloading of indicator parameters
3. Real-time configuration updates
4. Performance monitoring and optimization
5. WebSocket integration for live parameter updates
6. Progressive disclosure based on user skill level

Author: The Augster
Date: 2025-01-16
Version: 1.0.0
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from pathlib import Path
import asyncio
import json

from .excel_config_manager import MarketRegimeExcelManager
from .iv_percentile_analyzer import IVPercentileAnalyzer
from .iv_skew_analyzer import IVSkewAnalyzer
from .archive_enhanced_modules_do_not_use.enhanced_atr_indicators import EnhancedATRIndicators

logger = logging.getLogger(__name__)

class TechnicalIndicatorsIntegration:
    """
    Integration manager for technical indicators with Excel configuration
    
    Provides seamless integration between Excel-driven configuration and
    technical indicator analyzers with hot-reloading capabilities.
    """
    
    def __init__(self, excel_config_path: Optional[str] = None):
        """Initialize Technical Indicators Integration"""
        self.excel_manager = MarketRegimeExcelManager(excel_config_path)
        
        # Initialize technical indicator analyzers
        self.analyzers = {}
        self.config_cache = {}
        self.last_config_update = datetime.now()
        
        # WebSocket callbacks for real-time updates
        self.update_callbacks: List[Callable] = []
        
        # User skill level for progressive disclosure
        self.user_skill_level = "expert"  # novice, intermediate, expert
        
        # Performance monitoring
        self.performance_metrics = {
            'config_reload_count': 0,
            'parameter_updates': 0,
            'analyzer_initializations': 0,
            'last_reload_time': None
        }
        
        logger.info("Technical Indicators Integration initialized")
    
    def initialize_analyzers(self) -> bool:
        """Initialize all technical indicator analyzers with Excel configuration"""
        try:
            # Get technical indicators configuration
            tech_config = self.excel_manager.get_technical_indicators_config()
            
            # Initialize IV Percentile Analyzer
            if tech_config.get('IVPercentile', {}).get('EnableAnalysis', True):
                iv_percentile_config = tech_config['IVPercentile']
                self.analyzers['iv_percentile'] = IVPercentileAnalyzer(iv_percentile_config)
                logger.info("✅ IV Percentile Analyzer initialized")
            
            # Initialize IV Skew Analyzer
            if tech_config.get('IVSkew', {}).get('EnableAnalysis', True):
                iv_skew_config = tech_config['IVSkew']
                self.analyzers['iv_skew'] = IVSkewAnalyzer(iv_skew_config)
                logger.info("✅ IV Skew Analyzer initialized")
            
            # Initialize Enhanced ATR Indicators
            if tech_config.get('EnhancedATR', {}).get('EnableAnalysis', True):
                atr_config = tech_config['EnhancedATR']
                self.analyzers['enhanced_atr'] = EnhancedATRIndicators(atr_config)
                logger.info("✅ Enhanced ATR Indicators initialized")
            
            # Cache configuration for hot-reloading
            self.config_cache = tech_config.copy()
            self.performance_metrics['analyzer_initializations'] += 1
            
            logger.info(f"🚀 Initialized {len(self.analyzers)} technical indicator analyzers")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error initializing analyzers: {e}")
            return False
    
    def hot_reload_configuration(self) -> bool:
        """Hot-reload configuration and update analyzers"""
        try:
            # Reload Excel configuration
            if not self.excel_manager.hot_reload_configuration():
                logger.error("Failed to hot-reload Excel configuration")
                return False
            
            # Get updated configuration
            new_tech_config = self.excel_manager.get_technical_indicators_config()
            
            # Check if configuration has changed
            if new_tech_config == self.config_cache:
                logger.debug("No configuration changes detected")
                return True
            
            # Update analyzers with new configuration
            updated_analyzers = []
            
            # Update IV Percentile Analyzer
            if 'IVPercentile' in new_tech_config:
                iv_config = new_tech_config['IVPercentile']
                if 'iv_percentile' in self.analyzers:
                    self._update_analyzer_config(self.analyzers['iv_percentile'], iv_config)
                    updated_analyzers.append('IV Percentile')
                elif iv_config.get('EnableAnalysis', True):
                    # Re-initialize if newly enabled
                    self.analyzers['iv_percentile'] = IVPercentileAnalyzer(iv_config)
                    updated_analyzers.append('IV Percentile (new)')
            
            # Update IV Skew Analyzer
            if 'IVSkew' in new_tech_config:
                skew_config = new_tech_config['IVSkew']
                if 'iv_skew' in self.analyzers:
                    self._update_analyzer_config(self.analyzers['iv_skew'], skew_config)
                    updated_analyzers.append('IV Skew')
                elif skew_config.get('EnableAnalysis', True):
                    self.analyzers['iv_skew'] = IVSkewAnalyzer(skew_config)
                    updated_analyzers.append('IV Skew (new)')
            
            # Update Enhanced ATR Indicators
            if 'EnhancedATR' in new_tech_config:
                atr_config = new_tech_config['EnhancedATR']
                if 'enhanced_atr' in self.analyzers:
                    self._update_analyzer_config(self.analyzers['enhanced_atr'], atr_config)
                    updated_analyzers.append('Enhanced ATR')
                elif atr_config.get('EnableAnalysis', True):
                    self.analyzers['enhanced_atr'] = EnhancedATRIndicators(atr_config)
                    updated_analyzers.append('Enhanced ATR (new)')
            
            # Update cache and metrics
            self.config_cache = new_tech_config.copy()
            self.last_config_update = datetime.now()
            self.performance_metrics['config_reload_count'] += 1
            self.performance_metrics['last_reload_time'] = self.last_config_update
            
            # Notify WebSocket callbacks
            self._notify_config_update(updated_analyzers)
            
            logger.info(f"🔄 Hot-reloaded configuration for: {', '.join(updated_analyzers)}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error during hot-reload: {e}")
            return False
    
    def _update_analyzer_config(self, analyzer: Any, new_config: Dict[str, Any]):
        """Update analyzer configuration dynamically"""
        try:
            # Update configuration attributes
            for param, value in new_config.items():
                if hasattr(analyzer, 'config'):
                    analyzer.config[param] = value
                
                # Update specific analyzer attributes
                if hasattr(analyzer, param.lower()):
                    setattr(analyzer, param.lower(), value)
            
            # Reset history if configuration significantly changed
            if hasattr(analyzer, 'reset_history'):
                analyzer.reset_history()
            
            self.performance_metrics['parameter_updates'] += 1
            
        except Exception as e:
            logger.error(f"Error updating analyzer config: {e}")
    
    def get_analyzer_status(self) -> Dict[str, Any]:
        """Get status of all technical indicator analyzers"""
        try:
            status = {
                'analyzers': {},
                'performance_metrics': self.performance_metrics.copy(),
                'last_config_update': self.last_config_update.isoformat(),
                'user_skill_level': self.user_skill_level
            }
            
            for analyzer_name, analyzer in self.analyzers.items():
                analyzer_status = {
                    'enabled': True,
                    'initialized': analyzer is not None,
                    'config_loaded': analyzer_name in self.config_cache
                }
                
                # Get analyzer-specific statistics if available
                if hasattr(analyzer, 'get_current_statistics'):
                    analyzer_status['statistics'] = analyzer.get_current_statistics()
                
                status['analyzers'][analyzer_name] = analyzer_status
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting analyzer status: {e}")
            return {'error': str(e)}
    
    def analyze_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market data using all enabled technical indicators"""
        try:
            results = {
                'timestamp': datetime.now().isoformat(),
                'indicators': {},
                'regime_components': {},
                'overall_confidence': 0.0
            }
            
            total_confidence = 0.0
            active_analyzers = 0
            
            # Run IV Percentile analysis
            if 'iv_percentile' in self.analyzers:
                iv_result = self.analyzers['iv_percentile'].analyze_iv_percentile(market_data)
                results['indicators']['iv_percentile'] = {
                    'current_iv': iv_result.current_iv,
                    'iv_percentile': iv_result.iv_percentile,
                    'regime': iv_result.iv_regime.value,
                    'confidence': iv_result.confidence,
                    'regime_strength': iv_result.regime_strength
                }
                results['regime_components']['iv_percentile'] = self.analyzers['iv_percentile'].get_regime_component(market_data)
                total_confidence += iv_result.confidence
                active_analyzers += 1
            
            # Run IV Skew analysis
            if 'iv_skew' in self.analyzers:
                skew_result = self.analyzers['iv_skew'].analyze_iv_skew(market_data)
                results['indicators']['iv_skew'] = {
                    'put_call_skew': skew_result.put_call_skew,
                    'sentiment': skew_result.skew_sentiment.value,
                    'confidence': skew_result.confidence,
                    'skew_strength': skew_result.skew_strength
                }
                results['regime_components']['iv_skew'] = self.analyzers['iv_skew'].get_regime_component(market_data)
                total_confidence += skew_result.confidence
                active_analyzers += 1
            
            # Run Enhanced ATR analysis
            if 'enhanced_atr' in self.analyzers:
                atr_result = self.analyzers['enhanced_atr'].analyze_atr_indicators(market_data)
                results['indicators']['enhanced_atr'] = {
                    'atr_14': atr_result.atr_14,
                    'atr_percentile': atr_result.atr_percentile,
                    'regime': atr_result.atr_regime.value,
                    'confidence': atr_result.confidence,
                    'breakout_signals': atr_result.breakout_signals
                }
                results['regime_components']['enhanced_atr'] = self.analyzers['enhanced_atr'].get_regime_component(market_data)
                total_confidence += atr_result.confidence
                active_analyzers += 1
            
            # Calculate overall confidence
            if active_analyzers > 0:
                results['overall_confidence'] = total_confidence / active_analyzers
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing market data: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def register_update_callback(self, callback: Callable):
        """Register callback for configuration updates"""
        self.update_callbacks.append(callback)
    
    def _notify_config_update(self, updated_analyzers: List[str]):
        """Notify all registered callbacks of configuration updates"""
        try:
            update_data = {
                'timestamp': datetime.now().isoformat(),
                'updated_analyzers': updated_analyzers,
                'performance_metrics': self.performance_metrics
            }
            
            for callback in self.update_callbacks:
                try:
                    callback(update_data)
                except Exception as e:
                    logger.error(f"Error in update callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error notifying config update: {e}")
    
    def set_user_skill_level(self, skill_level: str):
        """Set user skill level for progressive disclosure"""
        if skill_level in ['novice', 'intermediate', 'expert']:
            self.user_skill_level = skill_level
            logger.info(f"User skill level set to: {skill_level}")
        else:
            logger.warning(f"Invalid skill level: {skill_level}")
    
    def get_filtered_parameters(self) -> Dict[str, Any]:
        """Get parameters filtered by user skill level"""
        try:
            all_config = self.excel_manager.get_technical_indicators_config()
            
            if self.user_skill_level == 'novice':
                # Show only basic enable/disable parameters
                filtered_config = {}
                for indicator_type, config in all_config.items():
                    filtered_config[indicator_type] = {
                        'EnableAnalysis': config.get('EnableAnalysis', True)
                    }
                return filtered_config
            
            elif self.user_skill_level == 'intermediate':
                # Show core parameters, hide advanced tuning
                filtered_config = {}
                intermediate_params = {
                    'IVPercentile': ['EnableAnalysis', 'MinDataPoints', 'LowThreshold', 'HighThreshold'],
                    'IVSkew': ['EnableAnalysis', 'MinStrikes', 'NeutralLowerThreshold', 'NeutralUpperThreshold'],
                    'EnhancedATR': ['EnableAnalysis', 'ShortPeriod', 'MediumPeriod', 'LongPeriod']
                }
                
                for indicator_type, config in all_config.items():
                    if indicator_type in intermediate_params:
                        filtered_config[indicator_type] = {
                            param: config.get(param) for param in intermediate_params[indicator_type]
                            if param in config
                        }
                return filtered_config
            
            else:  # expert
                # Show all parameters
                return all_config
                
        except Exception as e:
            logger.error(f"Error filtering parameters: {e}")
            return {}
