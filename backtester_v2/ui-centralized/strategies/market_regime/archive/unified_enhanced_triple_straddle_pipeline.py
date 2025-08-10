#!/usr/bin/env python3
"""
Unified Enhanced Triple Straddle Pipeline for Enhanced Triple Straddle Framework v2.0
====================================================================================

This module integrates all Phase 1 + Phase 2 components into a unified pipeline:

Phase 1 Components:
- Enhanced Volume-Weighted Greeks Calculator
- Delta-based Strike Selection System
- Enhanced Trending OI PA Analysis with Mathematical Correlation

Phase 2 Components:
- Hybrid Classification System (70%/30% weight distribution)
- Enhanced Performance Monitor
- Enhanced Excel Configuration Integration

Features:
- Feature flags for gradual rollout
- Backward compatibility switches
- Comprehensive error handling
- Integration test suite
- Performance targets: <3s processing, >85% accuracy, ±0.001 mathematical precision

Author: The Augster
Date: 2025-06-20
Version: 2.0.0
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Phase 1 Component Imports
try:
    from enhanced_volume_weighted_greeks import calculate_volume_weighted_greek_exposure, VolumeWeightingConfig
    from delta_based_strike_selector import select_strikes_by_delta_criteria, DeltaFilterConfig
    from enhanced_trending_oi_pa_analysis import EnhancedTrendingOIWithPAAnalysis
except ImportError:
    # Fallback for relative imports
    from .archive_enhanced_modules_do_not_use.enhanced_volume_weighted_greeks import calculate_volume_weighted_greek_exposure, VolumeWeightingConfig
    from .delta_based_strike_selector import select_strikes_by_delta_criteria, DeltaFilterConfig
    from .archive_enhanced_modules_do_not_use.enhanced_trending_oi_pa_analysis import EnhancedTrendingOIWithPAAnalysis

# Phase 2 Component Imports
try:
    from hybrid_classification_system import classify_hybrid_market_regime, HybridMarketRegimeClassifier
    from enhanced_performance_monitor import monitor_component_performance, EnhancedPerformanceMonitor
    from enhanced_excel_config_generator import generate_configuration_for_pipeline
except ImportError:
    # Fallback for relative imports
    from .hybrid_classification_system import classify_hybrid_market_regime, HybridMarketRegimeClassifier
    from .archive_enhanced_modules_do_not_use.enhanced_performance_monitor import monitor_component_performance, EnhancedPerformanceMonitor
    from .archive_enhanced_modules_do_not_use.enhanced_excel_config_generator import generate_configuration_for_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mathematical precision tolerance
MATHEMATICAL_TOLERANCE = 0.001

@dataclass
class PipelineConfig:
    """Configuration for the unified pipeline"""
    # Feature flags
    enable_phase1_components: bool = True
    enable_phase2_components: bool = True
    enable_volume_weighted_greeks: bool = True
    enable_delta_strike_selection: bool = True
    enable_enhanced_trending_oi: bool = True
    enable_hybrid_classification: bool = True
    enable_performance_monitoring: bool = True
    
    # Backward compatibility
    enable_legacy_fallback: bool = True
    preserve_existing_interfaces: bool = True
    
    # Performance settings
    max_processing_time: float = 3.0
    min_accuracy_threshold: float = 0.85
    mathematical_tolerance: float = MATHEMATICAL_TOLERANCE
    
    # Configuration profile
    configuration_profile: str = 'Balanced'  # Conservative, Balanced, Aggressive

@dataclass
class PipelineResult:
    """Result container for unified pipeline"""
    # Phase 1 Results
    volume_weighted_greeks: Optional[Dict[str, Any]] = None
    delta_strike_selection: Optional[Dict[str, Any]] = None
    enhanced_trending_oi: Optional[Dict[str, Any]] = None
    
    # Phase 2 Results
    hybrid_classification: Optional[Dict[str, Any]] = None
    performance_monitoring: Optional[Dict[str, Any]] = None
    
    # Pipeline Metadata
    pipeline_success: bool = False
    processing_time: float = 0.0
    mathematical_accuracy: bool = False
    error_messages: List[str] = None
    warning_messages: List[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.error_messages is None:
            self.error_messages = []
        if self.warning_messages is None:
            self.warning_messages = []
        if self.timestamp is None:
            self.timestamp = datetime.now()

class UnifiedEnhancedTripleStraddlePipeline:
    """
    Unified Pipeline for Enhanced Triple Straddle Framework v2.0
    Integrates all Phase 1 + Phase 2 components with feature flags and performance monitoring
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize Unified Enhanced Triple Straddle Pipeline
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        
        # Initialize component instances
        self.performance_monitor = None
        self.trending_oi_analyzer = None
        self.hybrid_classifier = None
        
        # Pipeline state
        self.pipeline_history = []
        self.component_configs = {}
        
        # Initialize components based on feature flags
        self._initialize_components()
        
        logger.info("Unified Enhanced Triple Straddle Pipeline initialized")
        logger.info(f"Phase 1 components enabled: {self.config.enable_phase1_components}")
        logger.info(f"Phase 2 components enabled: {self.config.enable_phase2_components}")
        logger.info(f"Configuration profile: {self.config.configuration_profile}")
    
    def _initialize_components(self) -> None:
        """Initialize pipeline components based on feature flags"""
        try:
            # Load configuration for the selected profile
            if self.config.enable_phase2_components:
                config_result = generate_configuration_for_pipeline(
                    self.config.configuration_profile
                )
                if config_result:
                    self.component_configs = config_result.get('config_data', {})
                    logger.info(f"Loaded {self.config.configuration_profile} configuration profile")
            
            # Initialize Performance Monitor (Phase 2)
            if self.config.enable_performance_monitoring and self.config.enable_phase2_components:
                self.performance_monitor = EnhancedPerformanceMonitor()
                self.performance_monitor.start_monitoring()
                logger.info("Performance monitoring initialized and started")
            
            # Initialize Enhanced Trending OI PA Analysis (Phase 1 Enhanced)
            if self.config.enable_enhanced_trending_oi and self.config.enable_phase1_components:
                trending_oi_config = self._get_trending_oi_config()
                self.trending_oi_analyzer = EnhancedTrendingOIWithPAAnalysis(trending_oi_config)
                logger.info("Enhanced Trending OI PA Analysis initialized")
            
            # Initialize Hybrid Classifier (Phase 2)
            if self.config.enable_hybrid_classification and self.config.enable_phase2_components:
                hybrid_config = self._get_hybrid_classification_config()
                self.hybrid_classifier = HybridMarketRegimeClassifier(hybrid_config)
                logger.info("Hybrid Classification System initialized")
            
        except Exception as e:
            logger.error(f"Error initializing pipeline components: {e}")
    
    def process_market_data(self, market_data: pd.DataFrame, 
                          timestamp: Optional[datetime] = None) -> PipelineResult:
        """
        Process market data through the unified pipeline
        
        Args:
            market_data: Market data containing options information
            timestamp: Processing timestamp
            
        Returns:
            PipelineResult containing all component results
        """
        try:
            start_time = datetime.now()
            if timestamp is None:
                timestamp = start_time
            
            logger.info(f"Processing market data through unified pipeline at {timestamp}")
            
            # Initialize result container
            result = PipelineResult(timestamp=timestamp)
            
            # Phase 1: Enhanced Volume-Weighted Greeks Calculator
            if self.config.enable_volume_weighted_greeks and self.config.enable_phase1_components:
                result.volume_weighted_greeks = self._process_volume_weighted_greeks(
                    market_data, timestamp, result
                )
            
            # Phase 1: Delta-based Strike Selection System
            if self.config.enable_delta_strike_selection and self.config.enable_phase1_components:
                result.delta_strike_selection = self._process_delta_strike_selection(
                    market_data, timestamp, result
                )
            
            # Phase 1: Enhanced Trending OI PA Analysis
            if self.config.enable_enhanced_trending_oi and self.config.enable_phase1_components:
                result.enhanced_trending_oi = self._process_enhanced_trending_oi(
                    market_data, timestamp, result
                )
            
            # Phase 2: Hybrid Classification System
            if self.config.enable_hybrid_classification and self.config.enable_phase2_components:
                result.hybrid_classification = self._process_hybrid_classification(
                    result, timestamp
                )
            
            # Calculate final pipeline metrics
            result.processing_time = (datetime.now() - start_time).total_seconds()
            result.mathematical_accuracy = self._validate_pipeline_mathematical_accuracy(result)
            result.pipeline_success = self._determine_pipeline_success(result)
            
            # Phase 2: Performance Monitoring
            if self.config.enable_performance_monitoring and self.config.enable_phase2_components:
                result.performance_monitoring = self._process_performance_monitoring(result)
            
            # Store in pipeline history
            self.pipeline_history.append(result)
            if len(self.pipeline_history) > 1000:
                self.pipeline_history = self.pipeline_history[-1000:]
            
            # Log pipeline completion
            logger.info(f"Pipeline processing completed in {result.processing_time:.3f}s")
            logger.info(f"Pipeline success: {result.pipeline_success}")
            logger.info(f"Mathematical accuracy: {result.mathematical_accuracy}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in unified pipeline processing: {e}")
            error_result = PipelineResult(timestamp=timestamp)
            error_result.error_messages.append(str(e))
            error_result.processing_time = (datetime.now() - start_time).total_seconds()
            return error_result
    
    def _process_volume_weighted_greeks(self, market_data: pd.DataFrame, 
                                      timestamp: datetime, result: PipelineResult) -> Optional[Dict[str, Any]]:
        """Process Volume-Weighted Greeks calculation"""
        try:
            logger.debug("Processing Volume-Weighted Greeks...")
            
            # Get configuration
            config = self._get_volume_weighted_greeks_config()
            
            # Calculate volume-weighted Greeks
            greeks_result = calculate_volume_weighted_greek_exposure(
                market_data, timestamp, config
            )
            
            if greeks_result:
                logger.debug(f"Volume-Weighted Greeks calculated: exposure={greeks_result.get('volume_weighted_greek_exposure', 0):.6f}")
                return greeks_result
            else:
                result.warning_messages.append("Volume-Weighted Greeks calculation failed")
                return None
                
        except Exception as e:
            logger.error(f"Error in Volume-Weighted Greeks processing: {e}")
            result.error_messages.append(f"Volume-Weighted Greeks error: {e}")
            return None
    
    def _process_delta_strike_selection(self, market_data: pd.DataFrame, 
                                      timestamp: datetime, result: PipelineResult) -> Optional[Dict[str, Any]]:
        """Process Delta-based Strike Selection"""
        try:
            logger.debug("Processing Delta-based Strike Selection...")
            
            # Get configuration
            config = self._get_delta_strike_selection_config()
            
            # Select strikes based on delta criteria
            selection_result = select_strikes_by_delta_criteria(
                market_data, timestamp, config
            )
            
            if selection_result:
                logger.debug(f"Delta-based Strike Selection completed: {len(selection_result.get('selected_strikes', []))} strikes selected")
                return selection_result
            else:
                result.warning_messages.append("Delta-based Strike Selection failed")
                return None
                
        except Exception as e:
            logger.error(f"Error in Delta-based Strike Selection processing: {e}")
            result.error_messages.append(f"Delta Strike Selection error: {e}")
            return None

    def _process_enhanced_trending_oi(self, market_data: pd.DataFrame,
                                    timestamp: datetime, result: PipelineResult) -> Optional[Dict[str, Any]]:
        """Process Enhanced Trending OI PA Analysis"""
        try:
            logger.debug("Processing Enhanced Trending OI PA Analysis...")

            if self.trending_oi_analyzer is None:
                result.warning_messages.append("Enhanced Trending OI analyzer not initialized")
                return None

            # Prepare market data for trending OI analysis
            market_data_dict = self._prepare_market_data_for_trending_oi(market_data, timestamp)

            # Analyze trending OI with PA
            oi_result = self.trending_oi_analyzer.analyze_trending_oi_pa(market_data_dict)

            if oi_result:
                logger.debug(f"Enhanced Trending OI PA Analysis completed: signal={oi_result.get('oi_signal', 0):.3f}")
                return oi_result
            else:
                result.warning_messages.append("Enhanced Trending OI PA Analysis failed")
                return None

        except Exception as e:
            logger.error(f"Error in Enhanced Trending OI PA Analysis processing: {e}")
            result.error_messages.append(f"Enhanced Trending OI error: {e}")
            return None

    def _process_hybrid_classification(self, pipeline_result: PipelineResult,
                                     timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Process Hybrid Classification System"""
        try:
            logger.debug("Processing Hybrid Classification System...")

            if self.hybrid_classifier is None:
                pipeline_result.warning_messages.append("Hybrid classifier not initialized")
                return None

            # Prepare enhanced system data from Phase 1 results
            enhanced_system_data = self._prepare_enhanced_system_data(pipeline_result)

            # Prepare timeframe hierarchy data (existing system)
            timeframe_hierarchy_data = self._prepare_timeframe_hierarchy_data(pipeline_result)

            # Classify market regime using hybrid system
            classification_result = classify_hybrid_market_regime(
                enhanced_system_data, timeframe_hierarchy_data
            )

            if classification_result:
                logger.debug(f"Hybrid Classification completed: regime={classification_result.get('hybrid_regime_classification', {}).get('regime_name', 'Unknown')}")
                return classification_result
            else:
                pipeline_result.warning_messages.append("Hybrid Classification failed")
                return None

        except Exception as e:
            logger.error(f"Error in Hybrid Classification processing: {e}")
            pipeline_result.error_messages.append(f"Hybrid Classification error: {e}")
            return None

    def _process_performance_monitoring(self, pipeline_result: PipelineResult) -> Optional[Dict[str, Any]]:
        """Process Performance Monitoring"""
        try:
            logger.debug("Processing Performance Monitoring...")

            if self.performance_monitor is None:
                pipeline_result.warning_messages.append("Performance monitor not initialized")
                return None

            # Calculate overall pipeline accuracy
            accuracy = self._calculate_pipeline_accuracy(pipeline_result)

            # Record performance metrics
            monitoring_result = monitor_component_performance(
                component_name="unified_pipeline",
                processing_time=pipeline_result.processing_time,
                accuracy=accuracy,
                mathematical_accuracy=pipeline_result.mathematical_accuracy,
                monitor_instance=self.performance_monitor
            )

            if monitoring_result:
                logger.debug(f"Performance monitoring completed: targets_met={monitoring_result.get('performance_targets_met', {})}")
                return monitoring_result
            else:
                pipeline_result.warning_messages.append("Performance monitoring failed")
                return None

        except Exception as e:
            logger.error(f"Error in Performance Monitoring processing: {e}")
            pipeline_result.error_messages.append(f"Performance Monitoring error: {e}")
            return None

    def _get_volume_weighted_greeks_config(self) -> Optional[VolumeWeightingConfig]:
        """Get Volume-Weighted Greeks configuration"""
        try:
            if 'VolumeWeightedGreeksConfig' in self.component_configs:
                config_data = self.component_configs['VolumeWeightedGreeksConfig']
                return VolumeWeightingConfig(
                    baseline_time=datetime.strptime(f"{config_data.get('baseline_time_hour', 9)}:{config_data.get('baseline_time_minute', 15)}", "%H:%M").time(),
                    expiry_weights={
                        0: config_data.get('dte_0_weight', 0.70),
                        1: config_data.get('dte_1_weight', 0.30),
                        2: config_data.get('dte_2_weight', 0.30),
                        3: config_data.get('dte_3_weight', 0.30)
                    },
                    greek_component_weights={
                        'delta': config_data.get('delta_component_weight', 0.40),
                        'gamma': config_data.get('gamma_component_weight', 0.30),
                        'theta': config_data.get('theta_component_weight', 0.20),
                        'vega': config_data.get('vega_component_weight', 0.10)
                    },
                    normalization_method=config_data.get('normalization_method', 'tanh'),
                    accuracy_tolerance=config_data.get('accuracy_tolerance', MATHEMATICAL_TOLERANCE)
                )
            else:
                return VolumeWeightingConfig()  # Use defaults

        except Exception as e:
            logger.error(f"Error getting Volume-Weighted Greeks config: {e}")
            return VolumeWeightingConfig()

    def _get_delta_strike_selection_config(self) -> Optional[DeltaFilterConfig]:
        """Get Delta-based Strike Selection configuration"""
        try:
            if 'DeltaStrikeSelectionConfig' in self.component_configs:
                config_data = self.component_configs['DeltaStrikeSelectionConfig']
                return DeltaFilterConfig(
                    call_delta_min=config_data.get('call_delta_min', 0.01),
                    call_delta_max=config_data.get('call_delta_max', 0.50),
                    put_delta_min=config_data.get('put_delta_min', -0.50),
                    put_delta_max=config_data.get('put_delta_max', -0.01),
                    max_strikes_per_expiry=config_data.get('max_strikes_per_expiry', 50),
                    recalculate_frequency=config_data.get('recalculate_frequency_seconds', 60),
                    accuracy_tolerance=config_data.get('mathematical_tolerance', MATHEMATICAL_TOLERANCE)
                )
            else:
                return DeltaFilterConfig()  # Use defaults

        except Exception as e:
            logger.error(f"Error getting Delta Strike Selection config: {e}")
            return DeltaFilterConfig()

    def _get_trending_oi_config(self) -> Dict[str, Any]:
        """Get Enhanced Trending OI PA Analysis configuration"""
        try:
            if 'MathematicalAccuracyConfig' in self.component_configs:
                config_data = self.component_configs['MathematicalAccuracyConfig']
                return {
                    'enable_pearson_correlation': True,
                    'enable_time_decay_weighting': True,
                    'enable_mathematical_validation': True,
                    'correlation_threshold': config_data.get('correlation_threshold', 0.80),
                    'lambda_decay': config_data.get('time_decay_lambda', 0.1),
                    'pattern_similarity_threshold': config_data.get('pattern_similarity_threshold', 0.75),
                    'enable_18_regime_classification': True,
                    'enable_volatility_component': True,
                    'enable_dynamic_thresholds': True
                }
            else:
                return {
                    'enable_pearson_correlation': True,
                    'enable_time_decay_weighting': True,
                    'enable_mathematical_validation': True,
                    'correlation_threshold': 0.80,
                    'lambda_decay': 0.1,
                    'enable_18_regime_classification': True,
                    'enable_volatility_component': True,
                    'enable_dynamic_thresholds': True
                }

        except Exception as e:
            logger.error(f"Error getting Trending OI config: {e}")
            return {}

    def _get_hybrid_classification_config(self) -> Optional[Dict[str, Any]]:
        """Get Hybrid Classification System configuration"""
        try:
            if 'HybridClassificationConfig' in self.component_configs:
                return self.component_configs['HybridClassificationConfig']
            else:
                return {
                    'enhanced_system_weight': 0.70,
                    'timeframe_hierarchy_weight': 0.30,
                    'agreement_threshold': 0.75,
                    'confidence_threshold': 0.60,
                    'max_processing_time_seconds': 3.0
                }

        except Exception as e:
            logger.error(f"Error getting Hybrid Classification config: {e}")
            return None

    def _prepare_market_data_for_trending_oi(self, market_data: pd.DataFrame, timestamp: datetime) -> Dict[str, Any]:
        """Prepare market data for trending OI analysis"""
        try:
            # Convert DataFrame to format expected by trending OI analyzer
            underlying_price = market_data['underlying_price'].iloc[0] if 'underlying_price' in market_data.columns else 23100

            # Group options by strike
            options_data = {}
            for _, row in market_data.iterrows():
                strike = float(row.get('strike', 0))
                option_type = row.get('option_type', 'CE')

                if strike not in options_data:
                    options_data[strike] = {}

                options_data[strike][option_type] = {
                    'oi': row.get('oi', 0),
                    'volume': row.get('volume', 0),
                    'close': row.get('close', row.get('ltp', 0)),
                    'previous_oi': row.get('previous_oi', row.get('oi', 0) * 0.95),
                    'previous_close': row.get('previous_close', row.get('close', row.get('ltp', 0)) * 0.98)
                }

            return {
                'underlying_price': underlying_price,
                'volatility': market_data.get('iv', pd.Series([0.15])).mean(),
                'strikes': list(options_data.keys()),
                'options_data': options_data,
                'timestamp': timestamp
            }

        except Exception as e:
            logger.error(f"Error preparing market data for trending OI: {e}")
            return {}

    def _prepare_enhanced_system_data(self, pipeline_result: PipelineResult) -> Dict[str, Any]:
        """Prepare enhanced system data from Phase 1 results"""
        try:
            enhanced_data = {}

            # Volume-weighted Greeks data
            if pipeline_result.volume_weighted_greeks:
                enhanced_data['volume_weighted_greek_exposure'] = pipeline_result.volume_weighted_greeks.get('volume_weighted_greek_exposure', 0.0)
                enhanced_data['portfolio_exposure'] = pipeline_result.volume_weighted_greeks.get('portfolio_exposure', 0.0)
                enhanced_data['confidence'] = pipeline_result.volume_weighted_greeks.get('confidence', 0.5)

            # Delta strike selection data
            if pipeline_result.delta_strike_selection:
                enhanced_data['selected_strikes_count'] = len(pipeline_result.delta_strike_selection.get('selected_strikes', []))
                enhanced_data['selection_confidence'] = pipeline_result.delta_strike_selection.get('selection_confidence', 0.5)

            # Enhanced trending OI data
            if pipeline_result.enhanced_trending_oi:
                enhanced_data['oi_signal'] = pipeline_result.enhanced_trending_oi.get('oi_signal', 0.0)
                enhanced_data['oi_confidence'] = pipeline_result.enhanced_trending_oi.get('confidence', 0.5)

                # Mathematical correlation data
                if 'correlation_analysis' in pipeline_result.enhanced_trending_oi:
                    corr_analysis = pipeline_result.enhanced_trending_oi['correlation_analysis']
                    enhanced_data['pearson_correlation'] = corr_analysis.get('pearson_correlation', 0.0)
                    enhanced_data['correlation_threshold_met'] = corr_analysis.get('correlation_threshold_met', False)

            # Calculate directional and volatility components for 18-regime classification
            enhanced_data['directional_component'] = self._calculate_directional_component(pipeline_result)
            enhanced_data['volatility_component'] = self._calculate_volatility_component(pipeline_result)

            return enhanced_data

        except Exception as e:
            logger.error(f"Error preparing enhanced system data: {e}")
            return {}

    def _prepare_timeframe_hierarchy_data(self, pipeline_result: PipelineResult) -> Dict[str, Any]:
        """Prepare timeframe hierarchy data (existing system)"""
        try:
            # Simulate existing timeframe hierarchy system data
            # In a real implementation, this would come from the existing system

            timeframe_data = {
                'primary_timeframe_signal': 0.0,
                'secondary_timeframe_signal': 0.0,
                'timeframe_agreement': 0.75,
                'timeframe_confidence': 0.65
            }

            # Use enhanced trending OI data if available
            if pipeline_result.enhanced_trending_oi:
                oi_signal = pipeline_result.enhanced_trending_oi.get('oi_signal', 0.0)
                oi_confidence = pipeline_result.enhanced_trending_oi.get('confidence', 0.5)

                # Map to timeframe signals
                timeframe_data['primary_timeframe_signal'] = oi_signal * 0.8  # 3-minute equivalent
                timeframe_data['secondary_timeframe_signal'] = oi_signal * 0.6  # 15-minute equivalent
                timeframe_data['timeframe_confidence'] = oi_confidence

            return timeframe_data

        except Exception as e:
            logger.error(f"Error preparing timeframe hierarchy data: {e}")
            return {}

    def _calculate_directional_component(self, pipeline_result: PipelineResult) -> float:
        """Calculate directional component for 18-regime classification"""
        try:
            directional_signals = []

            # Volume-weighted Greeks directional signal
            if pipeline_result.volume_weighted_greeks:
                greek_exposure = pipeline_result.volume_weighted_greeks.get('volume_weighted_greek_exposure', 0.0)
                directional_signals.append(greek_exposure)

            # Enhanced trending OI directional signal
            if pipeline_result.enhanced_trending_oi:
                oi_signal = pipeline_result.enhanced_trending_oi.get('oi_signal', 0.0)
                directional_signals.append(oi_signal)

            # Calculate weighted average
            if directional_signals:
                return np.mean(directional_signals)
            else:
                return 0.0

        except Exception as e:
            logger.error(f"Error calculating directional component: {e}")
            return 0.0

    def _calculate_volatility_component(self, pipeline_result: PipelineResult) -> float:
        """Calculate volatility component for 18-regime classification"""
        try:
            # Use a base volatility estimate
            base_volatility = 0.15

            # Adjust based on Greek exposure volatility
            if pipeline_result.volume_weighted_greeks:
                vega_exposure = pipeline_result.volume_weighted_greeks.get('vega_exposure', 0.0)
                # Higher vega exposure suggests higher volatility environment
                volatility_adjustment = abs(vega_exposure) * 0.1
                base_volatility += volatility_adjustment

            # Adjust based on OI signal volatility
            if pipeline_result.enhanced_trending_oi:
                oi_confidence = pipeline_result.enhanced_trending_oi.get('confidence', 0.5)
                # Lower confidence suggests higher volatility
                volatility_adjustment = (1.0 - oi_confidence) * 0.05
                base_volatility += volatility_adjustment

            return np.clip(base_volatility, 0.05, 0.50)  # Keep within reasonable bounds

        except Exception as e:
            logger.error(f"Error calculating volatility component: {e}")
            return 0.15

    def _calculate_pipeline_accuracy(self, pipeline_result: PipelineResult) -> float:
        """Calculate overall pipeline accuracy"""
        try:
            accuracy_scores = []

            # Volume-weighted Greeks accuracy
            if pipeline_result.volume_weighted_greeks:
                confidence = pipeline_result.volume_weighted_greeks.get('confidence', 0.5)
                math_accuracy = pipeline_result.volume_weighted_greeks.get('mathematical_accuracy', False)
                accuracy = confidence * (1.0 if math_accuracy else 0.8)
                accuracy_scores.append(accuracy)

            # Delta strike selection accuracy
            if pipeline_result.delta_strike_selection:
                confidence = pipeline_result.delta_strike_selection.get('selection_confidence', 0.5)
                math_accuracy = pipeline_result.delta_strike_selection.get('mathematical_accuracy', False)
                accuracy = confidence * (1.0 if math_accuracy else 0.8)
                accuracy_scores.append(accuracy)

            # Enhanced trending OI accuracy
            if pipeline_result.enhanced_trending_oi:
                confidence = pipeline_result.enhanced_trending_oi.get('confidence', 0.5)
                math_accuracy = pipeline_result.enhanced_trending_oi.get('mathematical_accuracy', False)
                accuracy = confidence * (1.0 if math_accuracy else 0.8)
                accuracy_scores.append(accuracy)

            # Hybrid classification accuracy
            if pipeline_result.hybrid_classification:
                hybrid_confidence = pipeline_result.hybrid_classification.get('hybrid_regime_classification', {}).get('confidence', 0.5)
                accuracy_scores.append(hybrid_confidence)

            # Calculate weighted average
            if accuracy_scores:
                return np.mean(accuracy_scores)
            else:
                return 0.5

        except Exception as e:
            logger.error(f"Error calculating pipeline accuracy: {e}")
            return 0.5

    def _validate_pipeline_mathematical_accuracy(self, pipeline_result: PipelineResult) -> bool:
        """Validate mathematical accuracy across all pipeline components"""
        try:
            accuracy_checks = []

            # Check Volume-Weighted Greeks mathematical accuracy
            if pipeline_result.volume_weighted_greeks:
                math_accuracy = pipeline_result.volume_weighted_greeks.get('mathematical_accuracy', False)
                accuracy_checks.append(math_accuracy)

            # Check Delta Strike Selection mathematical accuracy
            if pipeline_result.delta_strike_selection:
                math_accuracy = pipeline_result.delta_strike_selection.get('mathematical_accuracy', False)
                accuracy_checks.append(math_accuracy)

            # Check Enhanced Trending OI mathematical accuracy
            if pipeline_result.enhanced_trending_oi:
                math_accuracy = pipeline_result.enhanced_trending_oi.get('mathematical_accuracy', False)
                accuracy_checks.append(math_accuracy)

            # Check Hybrid Classification mathematical accuracy
            if pipeline_result.hybrid_classification:
                math_accuracy = pipeline_result.hybrid_classification.get('hybrid_regime_classification', {}).get('mathematical_accuracy', False)
                accuracy_checks.append(math_accuracy)

            # All components must pass mathematical accuracy validation
            return all(accuracy_checks) if accuracy_checks else False

        except Exception as e:
            logger.error(f"Error validating pipeline mathematical accuracy: {e}")
            return False

    def _determine_pipeline_success(self, pipeline_result: PipelineResult) -> bool:
        """Determine overall pipeline success"""
        try:
            # Check processing time
            if pipeline_result.processing_time > self.config.max_processing_time:
                pipeline_result.warning_messages.append(f"Processing time {pipeline_result.processing_time:.3f}s exceeds target {self.config.max_processing_time}s")

            # Check accuracy
            accuracy = self._calculate_pipeline_accuracy(pipeline_result)
            if accuracy < self.config.min_accuracy_threshold:
                pipeline_result.warning_messages.append(f"Pipeline accuracy {accuracy:.3f} below threshold {self.config.min_accuracy_threshold}")

            # Success criteria
            success_criteria = [
                len(pipeline_result.error_messages) == 0,  # No errors
                pipeline_result.processing_time <= self.config.max_processing_time * 1.2,  # Within 20% of time target
                accuracy >= self.config.min_accuracy_threshold * 0.9,  # Within 10% of accuracy target
                pipeline_result.mathematical_accuracy  # Mathematical accuracy validation passed
            ]

            return all(success_criteria)

        except Exception as e:
            logger.error(f"Error determining pipeline success: {e}")
            return False

    def get_pipeline_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive pipeline performance dashboard"""
        try:
            if not self.pipeline_history:
                return {'status': 'No pipeline data available'}

            recent_results = self.pipeline_history[-100:]  # Last 100 pipeline runs

            # Calculate performance metrics
            processing_times = [r.processing_time for r in recent_results]
            accuracies = [self._calculate_pipeline_accuracy(r) for r in recent_results]
            success_rates = [r.pipeline_success for r in recent_results]
            math_accuracy_rates = [r.mathematical_accuracy for r in recent_results]

            dashboard = {
                'pipeline_summary': {
                    'total_runs': len(self.pipeline_history),
                    'recent_runs': len(recent_results),
                    'avg_processing_time': np.mean(processing_times),
                    'max_processing_time': np.max(processing_times),
                    'avg_accuracy': np.mean(accuracies),
                    'success_rate': np.mean(success_rates),
                    'mathematical_accuracy_rate': np.mean(math_accuracy_rates)
                },
                'performance_targets': {
                    'processing_time_compliance': np.mean([t <= self.config.max_processing_time for t in processing_times]),
                    'accuracy_compliance': np.mean([a >= self.config.min_accuracy_threshold for a in accuracies]),
                    'mathematical_accuracy_compliance': np.mean(math_accuracy_rates),
                    'overall_compliance': np.mean(success_rates)
                },
                'component_status': {
                    'phase1_enabled': self.config.enable_phase1_components,
                    'phase2_enabled': self.config.enable_phase2_components,
                    'volume_weighted_greeks': self.config.enable_volume_weighted_greeks,
                    'delta_strike_selection': self.config.enable_delta_strike_selection,
                    'enhanced_trending_oi': self.config.enable_enhanced_trending_oi,
                    'hybrid_classification': self.config.enable_hybrid_classification,
                    'performance_monitoring': self.config.enable_performance_monitoring
                },
                'configuration': {
                    'profile': self.config.configuration_profile,
                    'max_processing_time': self.config.max_processing_time,
                    'min_accuracy_threshold': self.config.min_accuracy_threshold,
                    'mathematical_tolerance': self.config.mathematical_tolerance
                }
            }

            # Add performance monitor dashboard if available
            if self.performance_monitor:
                monitor_dashboard = self.performance_monitor.get_performance_dashboard()
                dashboard['detailed_performance'] = monitor_dashboard

            return dashboard

        except Exception as e:
            logger.error(f"Error getting pipeline performance dashboard: {e}")
            return {'error': str(e)}

    def cleanup(self) -> None:
        """Cleanup pipeline resources"""
        try:
            if self.performance_monitor:
                self.performance_monitor.stop_monitoring()

            logger.info("Pipeline cleanup completed")

        except Exception as e:
            logger.error(f"Error during pipeline cleanup: {e}")

# Integration function for external systems
def process_market_data_unified_pipeline(market_data: pd.DataFrame,
                                        config: Optional[PipelineConfig] = None,
                                        timestamp: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
    """
    Main integration function for processing market data through the unified pipeline

    Args:
        market_data: Market data containing options information
        config: Optional pipeline configuration
        timestamp: Optional processing timestamp

    Returns:
        Dictionary containing unified pipeline results or None if processing fails
    """
    try:
        # Initialize pipeline
        pipeline = UnifiedEnhancedTripleStraddlePipeline(config)

        # Process market data
        result = pipeline.process_market_data(market_data, timestamp)

        if result.pipeline_success:
            # Return results in format expected by external systems
            return {
                'pipeline_processing_successful': True,
                'processing_time': result.processing_time,
                'mathematical_accuracy': result.mathematical_accuracy,
                'pipeline_accuracy': pipeline._calculate_pipeline_accuracy(result),
                'timestamp': result.timestamp.isoformat(),

                # Phase 1 Results
                'volume_weighted_greeks': result.volume_weighted_greeks,
                'delta_strike_selection': result.delta_strike_selection,
                'enhanced_trending_oi': result.enhanced_trending_oi,

                # Phase 2 Results
                'hybrid_classification': result.hybrid_classification,
                'performance_monitoring': result.performance_monitoring,

                # Pipeline Metadata
                'error_messages': result.error_messages,
                'warning_messages': result.warning_messages,
                'configuration_profile': config.configuration_profile if config else 'Balanced'
            }
        else:
            logger.warning("Unified pipeline processing failed")
            return {
                'pipeline_processing_successful': False,
                'error_messages': result.error_messages,
                'warning_messages': result.warning_messages,
                'processing_time': result.processing_time
            }

    except Exception as e:
        logger.error(f"Error in unified pipeline processing: {e}")
        return None
    finally:
        # Cleanup
        if 'pipeline' in locals():
            pipeline.cleanup()

# Unit test function
def test_unified_enhanced_triple_straddle_pipeline():
    """Comprehensive unit test for unified pipeline"""
    try:
        logger.info("Testing Unified Enhanced Triple Straddle Pipeline...")

        # Create test market data
        test_data = pd.DataFrame({
            'strike': [22800, 22900, 23000, 23100, 23200, 23300, 23400],
            'option_type': ['CE', 'CE', 'CE', 'PE', 'PE', 'PE', 'PE'],
            'underlying_price': [23100] * 7,
            'dte': [1] * 7,
            'iv': [0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21],
            'volume': [100, 200, 300, 250, 200, 150, 100],
            'oi': [500, 750, 1000, 800, 600, 400, 200],
            'close': [150, 100, 60, 100, 150, 200, 250],
            'ltp': [150, 100, 60, 100, 150, 200, 250]
        })

        # Test with different configuration profiles
        profiles = ['Conservative', 'Balanced', 'Aggressive']

        for profile in profiles:
            config = PipelineConfig(configuration_profile=profile)
            result = process_market_data_unified_pipeline(test_data, config)

            if result and result['pipeline_processing_successful']:
                logger.info(f"✅ {profile} profile pipeline test PASSED")
                logger.info(f"   Processing time: {result['processing_time']:.3f}s")
                logger.info(f"   Pipeline accuracy: {result['pipeline_accuracy']:.3f}")
                logger.info(f"   Mathematical accuracy: {result['mathematical_accuracy']}")
                logger.info(f"   Components processed: {len([k for k in result.keys() if k.endswith('_greeks') or k.endswith('_selection') or k.endswith('_oi') or k.endswith('_classification')])}")
            else:
                logger.error(f"❌ {profile} profile pipeline test FAILED")
                if result:
                    logger.error(f"   Errors: {result.get('error_messages', [])}")
                return False

        logger.info("✅ Unified Enhanced Triple Straddle Pipeline test PASSED")
        return True

    except Exception as e:
        logger.error(f"❌ Unified Enhanced Triple Straddle Pipeline test ERROR: {e}")
        return False

if __name__ == "__main__":
    # Run comprehensive test
    test_unified_enhanced_triple_straddle_pipeline()
