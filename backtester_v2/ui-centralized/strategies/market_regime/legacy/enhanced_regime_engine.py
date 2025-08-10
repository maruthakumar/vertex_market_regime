"""
Enhanced Market Regime Engine
============================

This module provides the core market regime detection engine that integrates the enhanced
market regime optimizer package with the backtester framework, supporting Excel-based
configuration and comprehensive indicator analysis.

Features:
- Integration with enhanced market regime optimizer package
- Excel configuration support via MarketRegimeExcelParser
- Comprehensive indicator engine with 13+ indicators
- Dynamic weight adjustment based on performance
- Multi-timeframe analysis (3min, 5min, 10min, 15min)
- Strike analysis (ATM, ITM1, OTM1, Combined)
- DTE-specific adaptations
- Real-time regime detection with confidence scoring
- GPU acceleration support
- WebSocket streaming compatibility

Author: Market Regime Integration Team
Date: 2025-06-15
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import asyncio
import json

# Import configuration manager
try:
    from ..config_manager import get_config_manager
    config_manager = get_config_manager()
except ImportError:
    # Fallback for standalone testing
    sys.path.append(str(Path(__file__).parent.parent))
    from config_manager import get_config_manager
    config_manager = get_config_manager()

# Add enhanced package to path
enhanced_package_path = "/srv/samba/shared/enhanced-market-regime-optimizer-final-package-updated"
if enhanced_package_path not in sys.path:
    sys.path.insert(0, enhanced_package_path)

# Import from enhanced package
try:
    from src.market_regime_processor import MarketRegimeProcessor
    from src.stable_market_regime_classifier import StableMarketRegimeClassifier
    from utils.feature_engineering.ema_indicators.ema_indicators import EMAIndicators
    from utils.feature_engineering.vwap_indicators.vwap_indicators import VWAPIndicators
    from utils.feature_engineering.greek_sentiment.greek_sentiment import GreekSentiment
    from utils.feature_engineering.trending_oi_pa.trending_oi_pa_analysis import TrendingOIWithPAAnalysis
    from utils.feature_engineering.iv_skew.iv_skew import IVSkewAnalysis
    from utils.feature_engineering.atr_indicators.atr_indicators import ATRIndicators
    from utils.feature_engineering.premium_indicators.premium_indicators import PremiumIndicators
    ENHANCED_PACKAGE_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ Enhanced market regime package imported successfully")
except ImportError as e:
    ENHANCED_PACKAGE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Enhanced package not available: {e}")

# Import local components
try:
    from .excel_config_parser import MarketRegimeExcelParser, MarketRegimeConfig
    from .archive_enhanced_modules_do_not_use.enhanced_indicator_parameters import EnhancedIndicatorParameters, UserLevel
    from .models import RegimeType, RegimeClassification
except ImportError:
    # Handle relative import when running as script
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    from excel_config_parser import MarketRegimeExcelParser, MarketRegimeConfig
    from enhanced_indicator_parameters import EnhancedIndicatorParameters, UserLevel
    try:
        from models import RegimeType, RegimeClassification
    except ImportError:
        # Create minimal fallback classes
        class RegimeType:
            STRONG_BULLISH = "Strong_Bullish"
            MILD_BULLISH = "Mild_Bullish"
            NEUTRAL = "Neutral"
            MILD_BEARISH = "Mild_Bearish"
            STRONG_BEARISH = "Strong_Bearish"

        class RegimeClassification:
            def __init__(self, regime_type, confidence, timestamp):
                self.regime_type = regime_type
                self.confidence = confidence
                self.timestamp = timestamp

class EnhancedMarketRegimeEngine:
    """
    Enhanced Market Regime Detection Engine with Excel configuration support
    """
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize Enhanced Market Regime Engine
        
        Args:
            config_path (str, optional): Path to Excel configuration file
            config_dict (Dict[str, Any], optional): Configuration dictionary
        """
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Configuration
        self.config = None
        self.excel_parser = MarketRegimeExcelParser()

        # Enhanced parameter system
        self.enhanced_params = EnhancedIndicatorParameters()
        self.current_parameters = {}  # Current parameter values for each indicator
        self.parameter_validation_enabled = True

        # Enhanced package components
        self.enhanced_available = ENHANCED_PACKAGE_AVAILABLE
        self.regime_processor = None
        self.stable_classifier = None

        # Indicator engines
        self.indicator_engines = {}
        self.indicator_weights = {}
        self.dynamic_weights_enabled = True
        
        # Performance tracking
        self.performance_history = {}
        self.weight_adjustment_history = {}
        
        # State tracking
        self.current_regime = None
        self.regime_history = []
        self.confidence_history = []
        self.last_update = None
        
        # Initialize configuration
        if config_path:
            self._load_excel_config(config_path)
        elif config_dict:
            self._load_dict_config(config_dict)
        else:
            self._load_default_config()
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info(f"✅ EnhancedMarketRegimeEngine initialized")
        self.logger.info(f"   📊 Regime Mode: {self.config.regime_mode}")
        self.logger.info(f"   🔧 Indicators: {len(self.config.indicators)}")
        self.logger.info(f"   ⚡ Enhanced Package: {'Available' if self.enhanced_available else 'Not Available'}")
    
    def _load_excel_config(self, config_path: str) -> None:
        """Load configuration from Excel file"""
        try:
            self.logger.info(f"📁 Loading Excel configuration: {config_path}")
            
            # Validate and parse Excel file
            is_valid, error_msg, regime_mode = self.excel_parser.validate_excel_file(config_path)
            if not is_valid:
                raise ValueError(f"Excel validation failed: {error_msg}")
            
            # Parse configuration
            self.config = self.excel_parser.parse_excel_config(config_path)

            # Load enhanced parameters if available
            self._load_enhanced_parameters()

            self.logger.info(f"✅ Excel configuration loaded successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load Excel configuration: {e}")
            raise
    
    def _load_dict_config(self, config_dict: Dict[str, Any]) -> None:
        """Load configuration from dictionary"""
        try:
            self.logger.info("📋 Loading dictionary configuration")
            
            # Convert dictionary to MarketRegimeConfig
            # This is a simplified implementation - in production, you'd want more robust conversion
            self.config = MarketRegimeConfig(
                regime_mode=config_dict.get('regime_mode', '18_REGIME')
            )
            
            # Set basic indicator weights if provided
            if 'indicator_weights' in config_dict:
                for name, weight in config_dict['indicator_weights'].items():
                    self.indicator_weights[name] = weight
            
            self.logger.info("✅ Dictionary configuration loaded")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load dictionary configuration: {e}")
            raise
    
    def _load_default_config(self) -> None:
        """Load default configuration"""
        try:
            self.logger.info("🔧 Loading default configuration")
            
            # Create default configuration
            self.config = MarketRegimeConfig(regime_mode="18_REGIME")
            
            # Set default indicator weights
            self.indicator_weights = {
                'Greek_Sentiment': 1.0,
                'Trending_OI_PA': 0.9,
                'EMA_ATM': 0.8,
                'EMA_Combined': 0.9,
                'VWAP_ATM': 0.8,
                'VWAP_Combined': 0.9,
                'IV_Skew': 0.6,
                'ATR_Indicators': 0.5,
                'Premium_Indicators': 0.4
            }
            
            self.logger.info("✅ Default configuration loaded")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load default configuration: {e}")
            raise
    
    def _initialize_components(self) -> None:
        """Initialize all engine components"""
        try:
            self.logger.info("🚀 Initializing engine components...")
            
            # Initialize indicator engines
            self._initialize_indicator_engines()
            
            # Initialize enhanced package components if available
            if self.enhanced_available:
                self._initialize_enhanced_components()
            else:
                self._initialize_fallback_components()
            
            self.logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Component initialization failed: {e}")
            raise
    
    def _initialize_indicator_engines(self) -> None:
        """Initialize individual indicator engines"""
        try:
            if not self.enhanced_available:
                self.logger.warning("⚠️ Enhanced package not available, using fallback indicators")
                return
            
            # Initialize indicator engines based on configuration
            for indicator_name, indicator_config in self.config.indicators.items():
                if not indicator_config.enabled:
                    continue
                
                try:
                    if indicator_name.startswith('EMA_'):
                        self.indicator_engines[indicator_name] = EMAIndicators()
                    elif indicator_name.startswith('VWAP_'):
                        self.indicator_engines[indicator_name] = VWAPIndicators()
                    elif indicator_name == 'Greek_Sentiment':
                        self.indicator_engines[indicator_name] = GreekSentiment()
                    elif indicator_name == 'Trending_OI_PA':
                        self.indicator_engines[indicator_name] = TrendingOIWithPAAnalysis()
                    elif indicator_name == 'IV_Skew':
                        self.indicator_engines[indicator_name] = IVSkewAnalysis()
                    elif indicator_name == 'ATR_Indicators':
                        self.indicator_engines[indicator_name] = ATRIndicators()
                    elif indicator_name == 'Premium_Indicators':
                        self.indicator_engines[indicator_name] = PremiumIndicators()
                    
                    # Set initial weight
                    self.indicator_weights[indicator_name] = indicator_config.base_weight
                    
                    self.logger.info(f"   ✅ {indicator_name}: weight={indicator_config.base_weight}")
                    
                except Exception as e:
                    self.logger.warning(f"   ⚠️ Failed to initialize {indicator_name}: {e}")
            
            self.logger.info(f"📊 Initialized {len(self.indicator_engines)} indicator engines")
            
        except Exception as e:
            self.logger.error(f"❌ Indicator engine initialization failed: {e}")
            raise
    
    def _initialize_enhanced_components(self) -> None:
        """Initialize enhanced package components"""
        try:
            # Create configuration for enhanced components
            enhanced_config = {
                'regime_mode': self.config.regime_mode,
                'confidence_threshold': self.config.dynamic_weights.confidence_threshold,
                'smoothing_periods': self.config.dynamic_weights.regime_smoothing_periods
            }
            
            # Initialize market regime processor
            self.regime_processor = MarketRegimeProcessor(enhanced_config)
            
            # Initialize stable classifier for multi-timeframe analysis
            self.stable_classifier = StableMarketRegimeClassifier(enhanced_config)
            
            self.logger.info("✅ Enhanced package components initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced component initialization failed: {e}")
            raise
    
    def _initialize_fallback_components(self) -> None:
        """Initialize fallback components when enhanced package is not available"""
        try:
            self.logger.info("🔄 Initializing fallback components...")
            
            # Create simple fallback regime processor
            self.regime_processor = self._create_fallback_processor()
            
            self.logger.info("✅ Fallback components initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Fallback component initialization failed: {e}")
            raise
    
    def _create_fallback_processor(self):
        """Create a simple fallback regime processor"""
        class FallbackRegimeProcessor:
            def __init__(self, config):
                self.config = config
                self.logger = logging.getLogger("FallbackRegimeProcessor")
            
            def process_market_data(self, market_data: pd.DataFrame) -> Dict[str, Any]:
                """Simple fallback regime processing"""
                try:
                    if market_data.empty:
                        return {}
                    
                    # Simple regime detection based on price movement
                    latest_data = market_data.iloc[-1]
                    
                    # Calculate simple directional score
                    if 'close' in market_data.columns and len(market_data) > 1:
                        price_change = (latest_data['close'] - market_data.iloc[-2]['close']) / market_data.iloc[-2]['close']
                        
                        if price_change > 0.02:
                            regime_type = "Strong_Bullish"
                            confidence = 0.8
                        elif price_change > 0.01:
                            regime_type = "Mild_Bullish"
                            confidence = 0.7
                        elif price_change < -0.02:
                            regime_type = "Strong_Bearish"
                            confidence = 0.8
                        elif price_change < -0.01:
                            regime_type = "Mild_Bearish"
                            confidence = 0.7
                        else:
                            regime_type = "Neutral"
                            confidence = 0.6
                    else:
                        regime_type = "Neutral"
                        confidence = 0.5
                    
                    return {
                        'regime_type': regime_type,
                        'confidence': confidence,
                        'timestamp': latest_data.name if hasattr(latest_data, 'name') else datetime.now(),
                        'regime_score': price_change if 'price_change' in locals() else 0.0,
                        'component_scores': {'fallback': 1.0}
                    }
                    
                except Exception as e:
                    self.logger.error(f"Fallback processing error: {e}")
                    return {}
        
        return FallbackRegimeProcessor(self.config)

    def _load_enhanced_parameters(self) -> None:
        """Load enhanced parameters from configuration"""
        try:
            self.logger.info("📊 Loading enhanced parameters...")

            # Load detailed parameters if available in config
            if hasattr(self.config, 'detailed_parameters') and self.config.detailed_parameters:
                for indicator_name, params in self.config.detailed_parameters.items():
                    if indicator_name not in self.current_parameters:
                        self.current_parameters[indicator_name] = {}

                    for param_name, param_details in params.items():
                        # Use current_value if available, otherwise default_value
                        current_value = param_details.get('current_value', param_details.get('default_value'))
                        self.current_parameters[indicator_name][param_name] = current_value

                        self.logger.debug(f"   📋 {indicator_name}.{param_name}: {current_value}")

            # Load default parameters for indicators not in config
            for indicator_name in self.enhanced_params.get_all_indicators():
                if indicator_name not in self.current_parameters:
                    param_set = self.enhanced_params.get_indicator_parameters(indicator_name)
                    if param_set:
                        self.current_parameters[indicator_name] = {}
                        for param_name, param_def in param_set.parameters.items():
                            self.current_parameters[indicator_name][param_name] = param_def.default_value

            self.logger.info(f"✅ Loaded parameters for {len(self.current_parameters)} indicators")

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load enhanced parameters: {e}")

    def update_indicator_parameters(self, indicator_name: str, parameter_updates: Dict[str, Any]) -> bool:
        """
        Update parameters for a specific indicator with validation

        Args:
            indicator_name (str): Name of the indicator
            parameter_updates (Dict[str, Any]): Parameter updates to apply

        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            self.logger.info(f"🔄 Updating parameters for {indicator_name}")

            # Validate parameters if validation is enabled
            if self.parameter_validation_enabled:
                is_valid, errors = self.enhanced_params.validate_parameter_values(indicator_name, parameter_updates)
                if not is_valid:
                    self.logger.error(f"❌ Parameter validation failed for {indicator_name}: {errors}")
                    return False

            # Initialize indicator parameters if not exists
            if indicator_name not in self.current_parameters:
                self.current_parameters[indicator_name] = {}

            # Apply updates
            for param_name, param_value in parameter_updates.items():
                old_value = self.current_parameters[indicator_name].get(param_name, 'N/A')
                self.current_parameters[indicator_name][param_name] = param_value
                self.logger.debug(f"   📊 {param_name}: {old_value} → {param_value}")

            # Reinitialize indicator engine with new parameters if available
            if indicator_name in self.indicator_engines:
                self._reinitialize_indicator_engine(indicator_name)

            self.logger.info(f"✅ Parameters updated for {indicator_name}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to update parameters for {indicator_name}: {e}")
            return False

    def _reinitialize_indicator_engine(self, indicator_name: str) -> None:
        """Reinitialize indicator engine with new parameters"""
        try:
            if not self.enhanced_available:
                self.logger.warning(f"⚠️ Cannot reinitialize {indicator_name}: enhanced package not available")
                return

            # Get current parameters
            current_params = self.current_parameters.get(indicator_name, {})

            # Reinitialize based on indicator type
            if indicator_name.startswith('EMA_'):
                # Pass parameters to EMA engine
                self.indicator_engines[indicator_name] = EMAIndicators(**current_params)
            elif indicator_name.startswith('VWAP_'):
                # Pass parameters to VWAP engine
                self.indicator_engines[indicator_name] = VWAPIndicators(**current_params)
            elif indicator_name == 'Greek_Sentiment':
                # Pass parameters to Greek Sentiment engine
                self.indicator_engines[indicator_name] = GreekSentiment(**current_params)
            elif indicator_name == 'Trending_OI_PA':
                # Pass parameters to Trending OI engine
                self.indicator_engines[indicator_name] = TrendingOIWithPAAnalysis(**current_params)
            # Add other indicator types as needed

            self.logger.info(f"🔄 Reinitialized {indicator_name} with updated parameters")

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to reinitialize {indicator_name}: {e}")

    def get_indicator_parameters(self, indicator_name: str) -> Dict[str, Any]:
        """Get current parameters for an indicator"""
        return self.current_parameters.get(indicator_name, {}).copy()

    def get_all_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get all current parameters"""
        return {name: params.copy() for name, params in self.current_parameters.items()}

    def apply_parameter_preset(self, indicator_name: str, preset_name: str) -> bool:
        """
        Apply a parameter preset to an indicator

        Args:
            indicator_name (str): Name of the indicator
            preset_name (str): Name of the preset (conservative, balanced, aggressive)

        Returns:
            bool: True if preset applied successfully
        """
        try:
            self.logger.info(f"🎯 Applying {preset_name} preset to {indicator_name}")

            # Get preset configuration
            preset_config = self.enhanced_params.get_preset_configuration(indicator_name, preset_name)
            if not preset_config:
                self.logger.error(f"❌ Preset {preset_name} not found for {indicator_name}")
                return False

            # Apply preset parameters
            return self.update_indicator_parameters(indicator_name, preset_config)

        except Exception as e:
            self.logger.error(f"❌ Failed to apply preset {preset_name} to {indicator_name}: {e}")
            return False

    def get_parameter_performance_impact(self, indicator_name: str, parameter_changes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze potential performance impact of parameter changes

        Args:
            indicator_name (str): Name of the indicator
            parameter_changes (Dict[str, Any]): Proposed parameter changes

        Returns:
            Dict[str, Any]: Impact analysis
        """
        try:
            impact_analysis = {
                'indicator_name': indicator_name,
                'parameter_changes': parameter_changes,
                'impact_level': 'unknown',
                'risk_assessment': 'medium',
                'recommendations': []
            }

            # Get parameter definitions
            param_set = self.enhanced_params.get_indicator_parameters(indicator_name)
            if not param_set:
                impact_analysis['impact_level'] = 'unknown'
                impact_analysis['recommendations'].append("Parameter definitions not available")
                return impact_analysis

            # Analyze impact level based on parameter importance
            high_impact_count = 0
            critical_impact_count = 0

            for param_name, new_value in parameter_changes.items():
                if param_name in param_set.parameters:
                    param_def = param_set.parameters[param_name]
                    current_value = self.current_parameters.get(indicator_name, {}).get(param_name, param_def.default_value)

                    # Calculate change magnitude
                    if isinstance(new_value, (int, float)) and isinstance(current_value, (int, float)):
                        change_ratio = abs(new_value - current_value) / max(abs(current_value), 1e-6)

                        if param_def.impact_level == 'critical':
                            critical_impact_count += 1
                            if change_ratio > 0.2:  # 20% change in critical parameter
                                impact_analysis['recommendations'].append(f"Large change in critical parameter {param_name}: {change_ratio:.1%}")
                        elif param_def.impact_level == 'high':
                            high_impact_count += 1
                            if change_ratio > 0.5:  # 50% change in high impact parameter
                                impact_analysis['recommendations'].append(f"Significant change in {param_name}: {change_ratio:.1%}")

            # Determine overall impact level
            if critical_impact_count > 0:
                impact_analysis['impact_level'] = 'critical'
                impact_analysis['risk_assessment'] = 'high'
            elif high_impact_count > 0:
                impact_analysis['impact_level'] = 'high'
                impact_analysis['risk_assessment'] = 'medium'
            else:
                impact_analysis['impact_level'] = 'low'
                impact_analysis['risk_assessment'] = 'low'

            return impact_analysis

        except Exception as e:
            self.logger.error(f"❌ Failed to analyze parameter impact: {e}")
            return {'error': str(e)}

    def calculate_market_regime(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate market regime for given market data

        Args:
            market_data (pd.DataFrame): Market data with OHLCV and options data

        Returns:
            Dict[str, Any]: Regime analysis results
        """
        try:
            self.logger.debug(f"🔍 Calculating market regime for {len(market_data)} data points")

            if market_data.empty:
                return self._create_empty_result()

            # Process market data through regime processor
            if self.regime_processor:
                regime_result = self.regime_processor.process_market_data(market_data)
            else:
                regime_result = self._fallback_regime_calculation(market_data)

            # Update internal state
            if regime_result:
                self._update_regime_state(regime_result)

            # Add metadata
            regime_result['engine_type'] = 'enhanced' if self.enhanced_available else 'fallback'
            regime_result['config_mode'] = self.config.regime_mode
            regime_result['indicator_count'] = len(self.indicator_engines)

            return regime_result

        except Exception as e:
            self.logger.error(f"❌ Market regime calculation failed: {e}")
            return self._create_error_result(str(e))

    def calculate_regime_batch(self, market_data: pd.DataFrame, chunk_size: int = 1000) -> pd.DataFrame:
        """
        Calculate market regime for large datasets in batches

        Args:
            market_data (pd.DataFrame): Large market dataset
            chunk_size (int): Size of processing chunks

        Returns:
            pd.DataFrame: Regime results for all data points
        """
        try:
            self.logger.info(f"📊 Processing {len(market_data)} data points in batches of {chunk_size}")

            results = []

            for i in range(0, len(market_data), chunk_size):
                chunk_data = market_data.iloc[i:i+chunk_size]

                # Calculate regime for chunk
                regime_result = self.calculate_market_regime(chunk_data)

                if regime_result and 'regime_type' in regime_result:
                    # Create result row
                    result_row = {
                        'timestamp': chunk_data.index[-1] if not chunk_data.empty else pd.Timestamp.now(),
                        'regime_type': regime_result['regime_type'],
                        'confidence': regime_result.get('confidence', 0.0),
                        'regime_score': regime_result.get('regime_score', 0.0)
                    }
                    results.append(result_row)

                # Progress logging
                if (i // chunk_size + 1) % 10 == 0:
                    progress = min(100, (i + chunk_size) / len(market_data) * 100)
                    self.logger.info(f"   📈 Progress: {progress:.1f}%")

            # Convert to DataFrame
            results_df = pd.DataFrame(results)
            if not results_df.empty:
                results_df.set_index('timestamp', inplace=True)

            self.logger.info(f"✅ Batch processing completed: {len(results_df)} regime points")
            return results_df

        except Exception as e:
            self.logger.error(f"❌ Batch regime calculation failed: {e}")
            return pd.DataFrame()

    def get_current_regime(self) -> Optional[Dict[str, Any]]:
        """Get current regime state"""
        return self.current_regime

    def get_regime_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get regime history"""
        return self.regime_history[-limit:] if self.regime_history else []

    def update_indicator_weights(self, performance_data: Dict[str, float]) -> None:
        """
        Update indicator weights based on performance data

        Args:
            performance_data (Dict[str, float]): Performance metrics for each indicator
        """
        try:
            if not self.dynamic_weights_enabled:
                return

            self.logger.debug("🔄 Updating indicator weights based on performance")

            learning_rate = self.config.dynamic_weights.learning_rate

            for indicator_name, performance in performance_data.items():
                if indicator_name in self.indicator_weights:
                    current_weight = self.indicator_weights[indicator_name]

                    # Get indicator config for bounds
                    indicator_config = self.config.indicators.get(indicator_name)
                    if indicator_config:
                        min_weight = indicator_config.min_weight
                        max_weight = indicator_config.max_weight

                        # Calculate weight adjustment
                        weight_adjustment = learning_rate * (performance - 0.5)  # 0.5 is neutral performance
                        new_weight = current_weight + weight_adjustment

                        # Apply bounds
                        new_weight = max(min_weight, min(max_weight, new_weight))

                        # Update weight
                        self.indicator_weights[indicator_name] = new_weight

                        # Track adjustment
                        if indicator_name not in self.weight_adjustment_history:
                            self.weight_adjustment_history[indicator_name] = []

                        self.weight_adjustment_history[indicator_name].append({
                            'timestamp': datetime.now(),
                            'old_weight': current_weight,
                            'new_weight': new_weight,
                            'performance': performance,
                            'adjustment': weight_adjustment
                        })

                        self.logger.debug(f"   📊 {indicator_name}: {current_weight:.3f} → {new_weight:.3f} (perf: {performance:.3f})")

        except Exception as e:
            self.logger.error(f"❌ Weight update failed: {e}")

    def get_indicator_weights(self) -> Dict[str, float]:
        """Get current indicator weights"""
        return self.indicator_weights.copy()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics and statistics"""
        try:
            metrics = {
                'total_regime_calculations': len(self.regime_history),
                'current_regime': self.current_regime,
                'average_confidence': 0.0,
                'regime_distribution': {},
                'weight_adjustments': len(self.weight_adjustment_history),
                'last_update': self.last_update,
                'engine_type': 'enhanced' if self.enhanced_available else 'fallback'
            }

            # Calculate average confidence
            if self.confidence_history:
                metrics['average_confidence'] = np.mean(self.confidence_history)

            # Calculate regime distribution
            regime_types = [r.get('regime_type', 'Unknown') for r in self.regime_history]
            if regime_types:
                unique_regimes, counts = np.unique(regime_types, return_counts=True)
                metrics['regime_distribution'] = dict(zip(unique_regimes, counts.tolist()))

            return metrics

        except Exception as e:
            self.logger.error(f"❌ Performance metrics calculation failed: {e}")
            return {}

    def _update_regime_state(self, regime_result: Dict[str, Any]) -> None:
        """Update internal regime state"""
        try:
            # Update current regime
            self.current_regime = regime_result
            self.last_update = datetime.now()

            # Add to history
            regime_result['calculated_at'] = self.last_update
            self.regime_history.append(regime_result)

            # Maintain history size
            max_history = 1000
            if len(self.regime_history) > max_history:
                self.regime_history = self.regime_history[-max_history:]

            # Track confidence
            confidence = regime_result.get('confidence', 0.0)
            self.confidence_history.append(confidence)

            # Maintain confidence history size
            if len(self.confidence_history) > max_history:
                self.confidence_history = self.confidence_history[-max_history:]

        except Exception as e:
            self.logger.error(f"❌ State update failed: {e}")

    def _create_empty_result(self) -> Dict[str, Any]:
        """Create empty result for no data scenarios"""
        return {
            'regime_type': 'Unknown',
            'confidence': 0.0,
            'regime_score': 0.0,
            'timestamp': datetime.now(),
            'component_scores': {},
            'error': 'No market data provided'
        }

    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result"""
        return {
            'regime_type': 'Error',
            'confidence': 0.0,
            'regime_score': 0.0,
            'timestamp': datetime.now(),
            'component_scores': {},
            'error': error_message
        }

    def _fallback_regime_calculation(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Fallback regime calculation when enhanced package is not available"""
        try:
            if market_data.empty:
                return self._create_empty_result()

            # Simple price-based regime detection
            latest_data = market_data.iloc[-1]

            # Calculate price change if possible
            if len(market_data) > 1 and 'close' in market_data.columns:
                price_change = (latest_data['close'] - market_data.iloc[-2]['close']) / market_data.iloc[-2]['close']

                # Simple regime classification
                if price_change > 0.02:
                    regime_type = "Strong_Bullish"
                    confidence = 0.8
                elif price_change > 0.005:
                    regime_type = "Mild_Bullish"
                    confidence = 0.7
                elif price_change < -0.02:
                    regime_type = "Strong_Bearish"
                    confidence = 0.8
                elif price_change < -0.005:
                    regime_type = "Mild_Bearish"
                    confidence = 0.7
                else:
                    regime_type = "Neutral"
                    confidence = 0.6

                regime_score = price_change
            else:
                regime_type = "Neutral"
                confidence = 0.5
                regime_score = 0.0

            return {
                'regime_type': regime_type,
                'confidence': confidence,
                'regime_score': regime_score,
                'timestamp': latest_data.name if hasattr(latest_data, 'name') else datetime.now(),
                'component_scores': {'price_momentum': regime_score},
                'method': 'fallback'
            }

        except Exception as e:
            self.logger.error(f"❌ Fallback calculation failed: {e}")
            return self._create_error_result(str(e))

    def export_configuration(self, output_path: str) -> None:
        """Export current configuration to JSON"""
        try:
            config_export = {
                'regime_mode': self.config.regime_mode,
                'indicator_weights': self.indicator_weights,
                'dynamic_weights_enabled': self.dynamic_weights_enabled,
                'enhanced_available': self.enhanced_available,
                'performance_metrics': self.get_performance_metrics(),
                'export_timestamp': datetime.now().isoformat()
            }

            with open(output_path, 'w') as f:
                json.dump(config_export, f, indent=2, default=str)

            self.logger.info(f"✅ Configuration exported to: {output_path}")

        except Exception as e:
            self.logger.error(f"❌ Configuration export failed: {e}")
            raise


def main():
    """Test function for Enhanced Market Regime Engine"""
    try:
        print("🧪 Testing Enhanced Market Regime Engine")
        print("=" * 50)

        # Test with Excel configuration
        excel_config_path = os.path.join(config_manager.paths.get_input_sheets_path(), "market_regime_18_config.xlsx")

        if Path(excel_config_path).exists():
            print(f"📁 Testing with Excel config: {excel_config_path}")

            # Initialize engine
            engine = EnhancedMarketRegimeEngine(config_path=excel_config_path)

            # Create sample market data
            dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
            sample_data = pd.DataFrame({
                'timestamp': dates,
                'open': 22000 + np.random.randn(100) * 50,
                'high': 22050 + np.random.randn(100) * 50,
                'low': 21950 + np.random.randn(100) * 50,
                'close': 22000 + np.random.randn(100) * 50,
                'volume': 1000 + np.random.randint(0, 500, 100)
            })
            sample_data.set_index('timestamp', inplace=True)

            print(f"📊 Testing with {len(sample_data)} sample data points")

            # Calculate regime
            regime_result = engine.calculate_market_regime(sample_data)

            print(f"🎯 Regime Result:")
            print(f"   Type: {regime_result.get('regime_type', 'Unknown')}")
            print(f"   Confidence: {regime_result.get('confidence', 0.0):.3f}")
            print(f"   Score: {regime_result.get('regime_score', 0.0):.3f}")
            print(f"   Engine: {regime_result.get('engine_type', 'Unknown')}")

            # Get performance metrics
            metrics = engine.get_performance_metrics()
            print(f"📈 Performance Metrics:")
            print(f"   Calculations: {metrics.get('total_regime_calculations', 0)}")
            print(f"   Avg Confidence: {metrics.get('average_confidence', 0.0):.3f}")
            print(f"   Engine Type: {metrics.get('engine_type', 'Unknown')}")

            # Export configuration
            export_path = "/tmp/regime_engine_test_config.json"
            engine.export_configuration(export_path)
            print(f"💾 Configuration exported to: {export_path}")

        else:
            print(f"⚠️ Excel config not found: {excel_config_path}")
            print("   Testing with default configuration...")

            # Test with default configuration
            engine = EnhancedMarketRegimeEngine()
            print(f"🔧 Default engine initialized")
            print(f"   Enhanced Available: {engine.enhanced_available}")
            print(f"   Indicators: {len(engine.indicator_engines)}")

        print("\n✅ Enhanced Market Regime Engine test completed!")

    except Exception as e:
        print(f"❌ Engine test failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
