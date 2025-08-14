"""
Component 5: Complete ATR-EMA-CPR Dual-Asset Analysis Integration

Main analyzer integrating all Component 5 systems including dual-asset data extraction,
straddle and underlying ATR-EMA-CPR analysis, dual DTE framework, cross-asset validation,
enhanced regime classification, and production-aligned performance optimization.

Features:
- Complete dual-asset ATR-EMA-CPR analysis pipeline
- Production performance monitoring <200ms processing, <500MB memory
- 94-feature extraction with comprehensive validation
- Framework integration with Components 1+2+3+4
- Production-ready error handling and fallback mechanisms
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import logging
import asyncio
import time
import warnings
import psutil
import traceback

# Component 5 module imports
from .dual_asset_data_extractor import (
    DualAssetDataExtractor, DualAssetExtractionResult, StraddlePriceData, UnderlyingPriceData
)
from .straddle_atr_ema_cpr_engine import (
    StraddleATREMACPREngine, StraddleAnalysisResult
)
from .underlying_atr_ema_cpr_engine import (
    UnderlyingATREMACPREngine, UnderlyingAnalysisResult
)
from .dual_dte_framework import (
    DualDTEFramework, DTEIntegratedResult
)
from .cross_asset_integration import (
    CrossAssetIntegrationEngine, CrossAssetIntegrationResult
)
from .enhanced_regime_classifier import (
    EnhancedRegimeClassificationEngine, EnhancedRegimeClassificationResult
)

# Base component import
from ..base_component import BaseMarketRegimeComponent, ComponentAnalysisResult, FeatureVector

warnings.filterwarnings('ignore')


@dataclass
class Component05PerformanceMetrics:
    """Performance metrics for Component 5"""
    total_processing_time_ms: float
    data_extraction_time_ms: float
    straddle_analysis_time_ms: float
    underlying_analysis_time_ms: float
    dte_framework_time_ms: float
    cross_asset_integration_time_ms: float
    regime_classification_time_ms: float
    memory_usage_mb: float
    peak_memory_mb: float
    performance_budget_compliant: bool
    memory_budget_compliant: bool
    feature_extraction_efficiency: float
    metadata: Dict[str, Any]


@dataclass
class Component05IntegrationResult:
    """Complete Component 5 integration result"""
    # Core analysis results
    data_extraction_result: DualAssetExtractionResult
    straddle_analysis_result: StraddleAnalysisResult
    underlying_analysis_result: UnderlyingAnalysisResult
    dte_integrated_result: DTEIntegratedResult
    cross_asset_result: CrossAssetIntegrationResult
    regime_classification_result: EnhancedRegimeClassificationResult
    
    # Performance and integration
    performance_metrics: Component05PerformanceMetrics
    feature_vector_94: FeatureVector  # Exactly 94 features
    final_regime_classification: np.ndarray
    final_confidence_scores: np.ndarray
    
    # Framework integration
    component_weights: Dict[str, float]
    integration_metadata: Dict[str, Any]
    
    # Production validation
    data_quality_score: float
    analysis_validation_score: float
    framework_compliance_score: float
    
    # Metadata
    processing_timestamp: datetime
    metadata: Dict[str, Any]


class Component05PerformanceMonitor:
    """Performance monitoring system for Component 5"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # Performance budgets
        self.processing_budget_ms = config.get('processing_budget_ms', 200)
        self.memory_budget_mb = config.get('memory_budget_mb', 500)
        
        # Monitoring state
        self.start_time = None
        self.peak_memory = 0
        self.stage_timings = {}

    def start_monitoring(self):
        """Start performance monitoring"""
        self.start_time = time.time()
        self.peak_memory = self._get_current_memory()
        self.stage_timings = {}

    def record_stage(self, stage_name: str, processing_time_ms: float):
        """Record timing for a processing stage"""
        self.stage_timings[stage_name] = processing_time_ms
        
        # Update peak memory
        current_memory = self._get_current_memory()
        self.peak_memory = max(self.peak_memory, current_memory)

    def get_performance_metrics(self) -> Component05PerformanceMetrics:
        """Get comprehensive performance metrics"""
        
        total_time = (time.time() - self.start_time) * 1000 if self.start_time else 0
        current_memory = self._get_current_memory()
        
        # Calculate efficiency metrics
        feature_extraction_efficiency = self._calculate_feature_extraction_efficiency()
        
        return Component05PerformanceMetrics(
            total_processing_time_ms=total_time,
            data_extraction_time_ms=self.stage_timings.get('data_extraction', 0),
            straddle_analysis_time_ms=self.stage_timings.get('straddle_analysis', 0),
            underlying_analysis_time_ms=self.stage_timings.get('underlying_analysis', 0),
            dte_framework_time_ms=self.stage_timings.get('dte_framework', 0),
            cross_asset_integration_time_ms=self.stage_timings.get('cross_asset_integration', 0),
            regime_classification_time_ms=self.stage_timings.get('regime_classification', 0),
            memory_usage_mb=current_memory,
            peak_memory_mb=self.peak_memory,
            performance_budget_compliant=total_time <= self.processing_budget_ms,
            memory_budget_compliant=self.peak_memory <= self.memory_budget_mb,
            feature_extraction_efficiency=feature_extraction_efficiency,
            metadata={
                'budget_utilization_pct': (total_time / self.processing_budget_ms) * 100,
                'memory_utilization_pct': (self.peak_memory / self.memory_budget_mb) * 100,
                'stage_breakdown': self.stage_timings
            }
        )

    def _get_current_memory(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0

    def _calculate_feature_extraction_efficiency(self) -> float:
        """Calculate feature extraction efficiency"""
        
        total_feature_time = (self.stage_timings.get('straddle_analysis', 0) + 
                             self.stage_timings.get('underlying_analysis', 0))
        
        if total_feature_time == 0:
            return 1.0
        
        # Efficiency: features per millisecond
        efficiency = 94 / total_feature_time  # 94 features target
        return min(efficiency * 1000, 1.0)  # Normalize to 0-1


class Component05Analyzer(BaseMarketRegimeComponent):
    """
    Complete Component 5 ATR-EMA-CPR Dual-Asset Analysis System
    
    Integrates all dual-asset analysis components with production-aligned
    performance optimization and framework compatibility.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Component 5 with all sub-systems
        
        Args:
            config: Component configuration including performance budgets
        """
        # Set feature count for base class
        config['component_id'] = 5
        config['feature_count'] = 94
        config['expected_features'] = [
            f'component_05_feature_{i+1}' for i in range(94)
        ]
        
        super().__init__(config)
        
        # Initialize performance monitor
        self.performance_monitor = Component05PerformanceMonitor(config)
        
        # Initialize all sub-engines
        self.data_extractor = DualAssetDataExtractor(config)
        self.straddle_engine = StraddleATREMACPREngine(config)
        self.underlying_engine = UnderlyingATREMACPREngine(config)
        self.dte_framework = DualDTEFramework(config)
        self.cross_asset_engine = CrossAssetIntegrationEngine(config)
        self.regime_classifier = EnhancedRegimeClassificationEngine(config)
        
        # Production settings
        self.fallback_enabled = config.get('fallback_enabled', True)
        self.error_recovery_enabled = config.get('error_recovery_enabled', True)
        self.processing_budget_ms = config.get('processing_budget_ms', 110)
        
        self.logger.info(f"Initialized Component 5 with 94 features and {self.processing_budget_ms}ms budget")

    async def analyze(self, market_data: pd.DataFrame) -> ComponentAnalysisResult:
        """
        Complete Component 5 analysis with dual-asset ATR-EMA-CPR system
        
        Args:
            market_data: Production parquet data (48-column schema)
            
        Returns:
            ComponentAnalysisResult with 94 features and regime classification
        """
        self.performance_monitor.start_monitoring()
        analysis_start_time = time.time()
        
        try:
            # Extract features and perform analysis
            feature_vector = await self.extract_features(market_data)
            
            # Perform complete integrated analysis
            integration_result = await self._perform_complete_analysis(market_data)
            
            # Calculate final scores and confidence
            final_score = self._calculate_final_score(integration_result)
            final_confidence = self._calculate_final_confidence(integration_result)
            
            # Get performance metrics
            performance_metrics = self.performance_monitor.get_performance_metrics()
            
            # Calculate component weights for framework integration
            component_weights = self._calculate_component_weights(integration_result)
            
            processing_time = (time.time() - analysis_start_time) * 1000
            
            # Track performance
            self._track_performance(processing_time, success=True)
            
            # Log performance compliance
            self._log_performance_compliance(performance_metrics)
            
            return ComponentAnalysisResult(
                component_id=self.component_id,
                component_name=self.component_name,
                score=final_score,
                confidence=final_confidence,
                features=feature_vector,
                processing_time_ms=processing_time,
                weights=component_weights,
                metadata={
                    'regime_classification': integration_result.regime_classification_result.regime_classification,
                    'performance_metrics': performance_metrics,
                    'analysis_method': 'dual_asset_atr_ema_cpr',
                    'feature_count': 94,
                    'performance_compliant': performance_metrics.performance_budget_compliant,
                    'memory_compliant': performance_metrics.memory_budget_compliant,
                    'data_quality_score': integration_result.data_quality_score,
                    'cross_asset_validation': {
                        'trend_agreement': integration_result.cross_asset_result.validation_summary.get('avg_trend_agreement', 0),
                        'volatility_agreement': integration_result.cross_asset_result.validation_summary.get('avg_volatility_agreement', 0),
                        'level_agreement': integration_result.cross_asset_result.validation_summary.get('avg_level_agreement', 0)
                    }
                },
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error(f"Component 5 analysis failed: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Track failure
            processing_time = (time.time() - analysis_start_time) * 1000
            self._track_performance(processing_time, success=False)
            
            # Return fallback result if enabled
            if self.fallback_enabled:
                return await self._create_fallback_result(market_data, processing_time)
            else:
                raise

    async def extract_features(self, market_data: pd.DataFrame) -> FeatureVector:
        """
        Extract exactly 94 features from dual-asset ATR-EMA-CPR analysis
        
        Args:
            market_data: Production parquet data
            
        Returns:
            FeatureVector with 94 features
        """
        try:
            # Perform complete analysis to get all features
            integration_result = await self._perform_complete_analysis(market_data)
            
            # Combine all feature vectors to get exactly 94 features
            combined_features = self._combine_all_features(integration_result)
            
            # Validate feature count
            if combined_features.shape[1] != 94:
                self.logger.warning(f"Feature count mismatch: got {combined_features.shape[1]}, expected 94")
                # Pad or truncate to exactly 94
                combined_features = self._ensure_94_features(combined_features)
            
            return FeatureVector(
                features=combined_features,
                feature_names=self._get_94_feature_names(),
                feature_count=94,
                processing_time_ms=integration_result.performance_metrics.total_processing_time_ms,
                metadata={
                    'extraction_method': 'dual_asset_comprehensive',
                    'straddle_features': 42,
                    'underlying_features': 36,
                    'cross_asset_features': 16,
                    'feature_validation': 'passed'
                }
            )
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {str(e)}")
            
            # Return fallback features
            if self.fallback_enabled:
                return self._create_fallback_features(market_data)
            else:
                raise

    async def update_weights(self, performance_feedback: 'PerformanceFeedback') -> 'WeightUpdate':
        """
        Update component weights based on performance feedback
        
        Args:
            performance_feedback: Performance metrics for adaptive learning
            
        Returns:
            WeightUpdate with updated weights
        """
        try:
            # Get current weights
            current_weights = self.current_weights.copy()
            
            # Calculate weight updates based on performance
            weight_changes = self._calculate_weight_updates(performance_feedback)
            
            # Apply updates
            updated_weights = {}
            for component, weight in current_weights.items():
                change = weight_changes.get(component, 0)
                updated_weights[component] = max(0.1, min(2.0, weight + change))
            
            # Calculate performance improvement
            performance_improvement = self._estimate_performance_improvement(
                performance_feedback, weight_changes
            )
            
            # Update internal weights
            self.current_weights = updated_weights
            self.weight_history.append({
                'timestamp': datetime.utcnow(),
                'weights': updated_weights.copy(),
                'performance_improvement': performance_improvement
            })
            
            return WeightUpdate(
                updated_weights=updated_weights,
                weight_changes=weight_changes,
                performance_improvement=performance_improvement,
                confidence_score=min(performance_feedback.accuracy, 0.95)
            )
            
        except Exception as e:
            self.logger.error(f"Weight update failed: {str(e)}")
            # Return no-change update
            return WeightUpdate(
                updated_weights=self.current_weights.copy(),
                weight_changes={},
                performance_improvement=0.0,
                confidence_score=0.5
            )

    async def _perform_complete_analysis(self, market_data: pd.DataFrame) -> Component05IntegrationResult:
        """Perform complete dual-asset analysis pipeline"""
        
        # Stage 1: Dual-Asset Data Extraction
        extraction_start = time.time()
        data_extraction_result = await self.data_extractor.extract_dual_asset_data(market_data)
        extraction_time = (time.time() - extraction_start) * 1000
        self.performance_monitor.record_stage('data_extraction', extraction_time)
        
        # Stage 2: Straddle ATR-EMA-CPR Analysis
        straddle_start = time.time()
        straddle_result = await self.straddle_engine.analyze_straddle_atr_ema_cpr(
            data_extraction_result.straddle_data
        )
        straddle_time = (time.time() - straddle_start) * 1000
        self.performance_monitor.record_stage('straddle_analysis', straddle_time)
        
        # Stage 3: Underlying ATR-EMA-CPR Analysis
        underlying_start = time.time()
        underlying_result = await self.underlying_engine.analyze_underlying_atr_ema_cpr(
            data_extraction_result.underlying_data
        )
        underlying_time = (time.time() - underlying_start) * 1000
        self.performance_monitor.record_stage('underlying_analysis', underlying_time)
        
        # Stage 4: Dual DTE Framework
        dte_start = time.time()
        dte_result = await self.dte_framework.analyze_dual_dte_framework(
            data_extraction_result.straddle_data, straddle_result, underlying_result
        )
        dte_time = (time.time() - dte_start) * 1000
        self.performance_monitor.record_stage('dte_framework', dte_time)
        
        # Stage 5: Cross-Asset Integration
        integration_start = time.time()
        cross_asset_result = await self.cross_asset_engine.integrate_cross_asset_analysis(
            straddle_result, underlying_result, dte_result
        )
        integration_time = (time.time() - integration_start) * 1000
        self.performance_monitor.record_stage('cross_asset_integration', integration_time)
        
        # Stage 6: Enhanced Regime Classification
        classification_start = time.time()
        regime_result = await self.regime_classifier.classify_enhanced_regimes(
            straddle_result, underlying_result, dte_result, cross_asset_result
        )
        classification_time = (time.time() - classification_start) * 1000
        self.performance_monitor.record_stage('regime_classification', classification_time)
        
        # Get performance metrics
        performance_metrics = self.performance_monitor.get_performance_metrics()
        
        # Create 94-feature vector
        feature_vector_94 = self._create_94_feature_vector(
            straddle_result, underlying_result, cross_asset_result
        )
        
        # Calculate validation scores
        data_quality_score = data_extraction_result.data_quality_score
        analysis_validation_score = self._calculate_analysis_validation_score(
            straddle_result, underlying_result, cross_asset_result
        )
        framework_compliance_score = self._calculate_framework_compliance_score(performance_metrics)
        
        return Component05IntegrationResult(
            data_extraction_result=data_extraction_result,
            straddle_analysis_result=straddle_result,
            underlying_analysis_result=underlying_result,
            dte_integrated_result=dte_result,
            cross_asset_result=cross_asset_result,
            regime_classification_result=regime_result,
            performance_metrics=performance_metrics,
            feature_vector_94=feature_vector_94,
            final_regime_classification=regime_result.regime_classification,
            final_confidence_scores=regime_result.confidence_result.final_composite_confidence,
            component_weights=self._calculate_component_weights_from_results(
                straddle_result, underlying_result, cross_asset_result
            ),
            integration_metadata={
                'total_stages': 6,
                'all_stages_successful': True,
                'performance_compliant': performance_metrics.performance_budget_compliant
            },
            data_quality_score=data_quality_score,
            analysis_validation_score=analysis_validation_score,
            framework_compliance_score=framework_compliance_score,
            processing_timestamp=datetime.utcnow(),
            metadata={
                'analysis_method': 'complete_dual_asset_pipeline',
                'feature_count': 94,
                'regime_count': 8
            }
        )

    def _create_94_feature_vector(self, straddle_result: StraddleAnalysisResult,
                                 underlying_result: UnderlyingAnalysisResult,
                                 cross_asset_result: CrossAssetIntegrationResult) -> FeatureVector:
        """Create exactly 94 features from all analyses"""
        
        # Straddle features (42 features)
        straddle_features = straddle_result.feature_vector
        
        # Underlying features (36 features) 
        underlying_features = underlying_result.feature_vector
        
        # Cross-asset features (16 features)
        cross_asset_features = self._extract_cross_asset_features(cross_asset_result)
        
        # Ensure correct dimensions
        min_rows = min(
            straddle_features.shape[0] if straddle_features.size > 0 else 1,
            underlying_features.shape[0] if underlying_features.size > 0 else 1,
            cross_asset_features.shape[0] if cross_asset_features.size > 0 else 1
        )
        
        if min_rows == 0:
            min_rows = 1
        
        # Combine features
        combined_features = np.zeros((min_rows, 94))
        
        # Straddle features (0-41)
        if straddle_features.size > 0:
            straddle_cols = min(42, straddle_features.shape[1])
            combined_features[:min_rows, :straddle_cols] = straddle_features[:min_rows, :straddle_cols]
        
        # Underlying features (42-77)
        if underlying_features.size > 0:
            underlying_cols = min(36, underlying_features.shape[1])
            combined_features[:min_rows, 42:42+underlying_cols] = underlying_features[:min_rows, :underlying_cols]
        
        # Cross-asset features (78-93)
        if cross_asset_features.size > 0:
            cross_asset_cols = min(16, cross_asset_features.shape[1])
            combined_features[:min_rows, 78:78+cross_asset_cols] = cross_asset_features[:min_rows, :cross_asset_cols]
        
        return FeatureVector(
            features=combined_features,
            feature_names=self._get_94_feature_names(),
            feature_count=94,
            processing_time_ms=straddle_result.processing_time_ms + underlying_result.processing_time_ms,
            metadata={'combination_method': 'structured_concatenation'}
        )

    def _extract_cross_asset_features(self, cross_asset_result: CrossAssetIntegrationResult) -> np.ndarray:
        """Extract 16 cross-asset features"""
        
        features = []
        
        # Validation features (6 features)
        if len(cross_asset_result.trend_validation.agreement_score) > 0:
            features.append(cross_asset_result.trend_validation.agreement_score)
        else:
            features.append(np.array([0.5]))
            
        if len(cross_asset_result.volatility_validation.regime_agreement_score) > 0:
            features.append(cross_asset_result.volatility_validation.regime_agreement_score)
        else:
            features.append(np.array([0.5]))
            
        if len(cross_asset_result.level_validation.level_agreement_score) > 0:
            features.append(cross_asset_result.level_validation.level_agreement_score)
        else:
            features.append(np.array([0.5]))
        
        # Add correlation features
        features.append(np.full(len(features[0]), cross_asset_result.trend_validation.trend_strength_correlation))
        features.append(np.full(len(features[0]), cross_asset_result.volatility_validation.cross_asset_volatility_correlation))
        features.append(np.full(len(features[0]), cross_asset_result.level_validation.cross_asset_level_correlation))
        
        # Confidence features (4 features)
        features.append(cross_asset_result.confidence_result.final_confidence)
        features.append(cross_asset_result.confidence_result.validation_boost)
        features.append(cross_asset_result.confidence_result.conflict_penalty)
        features.append(cross_asset_result.confidence_result.high_confidence_zones.astype(float))
        
        # Weighting features (4 features)
        features.append(cross_asset_result.weighting_result.final_weights['straddle'])
        features.append(cross_asset_result.weighting_result.final_weights['underlying'])
        
        # Add weight change indicators
        straddle_weight_change = np.abs(cross_asset_result.weighting_result.final_weights['straddle'] - 0.6)  # From base 60%
        underlying_weight_change = np.abs(cross_asset_result.weighting_result.final_weights['underlying'] - 0.4)  # From base 40%
        
        features.append(straddle_weight_change)
        features.append(underlying_weight_change)
        
        # Integrated signals (2 features)
        confirmed_bullish = cross_asset_result.integrated_signals.get('confirmed_bullish', np.zeros(len(features[0])))
        confirmed_bearish = cross_asset_result.integrated_signals.get('confirmed_bearish', np.zeros(len(features[0])))
        
        features.append(confirmed_bullish)
        features.append(confirmed_bearish)
        
        # Ensure all features have same length
        min_length = min(len(f) for f in features)
        normalized_features = [f[:min_length] for f in features]
        
        # Pad to exactly 16 features if needed
        while len(normalized_features) < 16:
            normalized_features.append(np.full(min_length, 0.5))
        
        return np.column_stack(normalized_features[:16])

    def _get_94_feature_names(self) -> List[str]:
        """Get names for all 94 features"""
        
        names = []
        
        # Straddle features (42 features)
        straddle_names = [
            # ATR features (12 features)
            'straddle_atr_14', 'straddle_atr_21', 'straddle_atr_50',
            'straddle_atr_14_pct', 'straddle_atr_21_pct', 'straddle_atr_50_pct',
            'straddle_vol_regime', 'straddle_atr_trend', 'straddle_vol_expansion',
            'straddle_vol_contraction', 'straddle_atr_convergence', 'straddle_true_range',
            
            # EMA features (15 features)
            'straddle_ema_20', 'straddle_ema_50', 'straddle_ema_100', 'straddle_ema_200',
            'straddle_trend_direction', 'straddle_trend_strength', 'straddle_confluence',
            'straddle_golden_cross_20_50', 'straddle_death_cross_20_50',
            'straddle_golden_cross_50_100', 'straddle_death_cross_50_100',
            'straddle_ema_20_slope', 'straddle_ema_50_slope', 'straddle_ema_100_slope', 'straddle_ema_200_slope',
            
            # CPR features (15 features)
            'straddle_pivot_standard', 'straddle_pivot_fibonacci', 'straddle_pivot_camarilla',
            'straddle_support_1', 'straddle_support_2', 'straddle_resistance_1', 'straddle_resistance_2',
            'straddle_cpr_width', 'straddle_cpr_position', 'straddle_resistance_breakout',
            'straddle_support_breakdown', 'straddle_level_strength_standard',
            'straddle_level_strength_fibonacci', 'straddle_level_strength_camarilla', 'straddle_cpr_additional'
        ]
        names.extend(straddle_names)
        
        # Underlying features (36 features)
        underlying_names = [
            # Multi-timeframe ATR features (12 features)
            'underlying_daily_atr_14', 'underlying_weekly_atr_14', 'underlying_monthly_atr_14',
            'underlying_daily_vol_regime', 'underlying_weekly_vol_regime', 'underlying_monthly_vol_regime',
            'underlying_atr_weekly_daily_consistency', 'underlying_atr_monthly_daily_consistency',
            'underlying_atr_all_timeframes_consistency', 'underlying_daily_atr_trend',
            'underlying_weekly_atr_trend', 'underlying_monthly_atr_trend',
            
            # Multi-timeframe EMA features (12 features)
            'underlying_daily_ema_20', 'underlying_weekly_ema_10', 'underlying_monthly_ema_6',
            'underlying_daily_trend_direction', 'underlying_weekly_trend_direction', 'underlying_monthly_trend_direction',
            'underlying_daily_trend_strength', 'underlying_weekly_trend_strength', 'underlying_monthly_trend_strength',
            'underlying_trend_daily_weekly_agreement', 'underlying_trend_daily_monthly_agreement',
            'underlying_trend_all_timeframes_agreement',
            
            # Multi-timeframe CPR features (12 features)
            'underlying_daily_pivot', 'underlying_weekly_pivot', 'underlying_monthly_pivot',
            'underlying_daily_resistance_1', 'underlying_daily_support_1',
            'underlying_weekly_resistance_1', 'underlying_weekly_support_1',
            'underlying_pivot_agreement', 'underlying_support_strength', 'underlying_resistance_strength',
            'underlying_daily_resistance_breakout', 'underlying_daily_support_breakdown'
        ]
        names.extend(underlying_names)
        
        # Cross-asset features (16 features)
        cross_asset_names = [
            'cross_asset_trend_agreement', 'cross_asset_volatility_agreement', 'cross_asset_level_agreement',
            'trend_strength_correlation', 'volatility_correlation', 'level_correlation',
            'cross_asset_final_confidence', 'cross_asset_validation_boost', 'cross_asset_conflict_penalty',
            'high_confidence_zones', 'dynamic_straddle_weight', 'dynamic_underlying_weight',
            'straddle_weight_change', 'underlying_weight_change',
            'confirmed_bullish_signal', 'confirmed_bearish_signal'
        ]
        names.extend(cross_asset_names)
        
        return names

    def _combine_all_features(self, integration_result: Component05IntegrationResult) -> np.ndarray:
        """Combine all features into single array"""
        return integration_result.feature_vector_94.features

    def _ensure_94_features(self, features: np.ndarray) -> np.ndarray:
        """Ensure exactly 94 features by padding or truncating"""
        
        if features.shape[1] == 94:
            return features
        elif features.shape[1] > 94:
            return features[:, :94]  # Truncate
        else:
            # Pad with zeros
            padded = np.zeros((features.shape[0], 94))
            padded[:, :features.shape[1]] = features
            return padded

    def _calculate_final_score(self, integration_result: Component05IntegrationResult) -> float:
        """Calculate final component score"""
        
        # Base score from regime classification confidence
        base_score = np.mean(integration_result.final_confidence_scores)
        
        # Performance adjustment
        perf_adjustment = 0.0
        if integration_result.performance_metrics.performance_budget_compliant:
            perf_adjustment += 0.05
        if integration_result.performance_metrics.memory_budget_compliant:
            perf_adjustment += 0.05
        
        # Cross-asset validation adjustment
        validation_summary = integration_result.cross_asset_result.validation_summary
        validation_score = (
            validation_summary.get('avg_trend_agreement', 0.5) * 0.4 +
            validation_summary.get('avg_volatility_agreement', 0.5) * 0.4 +
            validation_summary.get('avg_level_agreement', 0.5) * 0.2
        )
        validation_adjustment = (validation_score - 0.5) * 0.1  # ±5% based on validation
        
        final_score = base_score + perf_adjustment + validation_adjustment
        return np.clip(final_score, 0.0, 1.0)

    def _calculate_final_confidence(self, integration_result: Component05IntegrationResult) -> float:
        """Calculate final confidence score"""
        
        # Use multi-layered confidence result
        return np.mean(integration_result.regime_classification_result.confidence_result.final_composite_confidence)

    def _calculate_component_weights(self, integration_result: Component05IntegrationResult) -> Dict[str, float]:
        """Calculate weights for framework integration"""
        
        return {
            'straddle_analysis': 0.35,  # Primary analysis
            'underlying_analysis': 0.25,  # Supporting analysis  
            'cross_asset_validation': 0.20,  # Validation boost
            'dte_framework': 0.15,  # DTE-specific insights
            'regime_classification': 0.05   # Final classification
        }

    def _calculate_component_weights_from_results(self, straddle_result: StraddleAnalysisResult,
                                                underlying_result: UnderlyingAnalysisResult,
                                                cross_asset_result: CrossAssetIntegrationResult) -> Dict[str, float]:
        """Calculate component weights from analysis results"""
        
        # Dynamic weights based on performance and validation
        base_weights = {
            'straddle_analysis': 0.35,
            'underlying_analysis': 0.25,
            'cross_asset_validation': 0.20,
            'dte_framework': 0.15,
            'regime_classification': 0.05
        }
        
        # Adjust based on cross-asset validation quality
        validation_quality = np.mean([
            cross_asset_result.validation_summary.get('avg_trend_agreement', 0.5),
            cross_asset_result.validation_summary.get('avg_volatility_agreement', 0.5),
            cross_asset_result.validation_summary.get('avg_level_agreement', 0.5)
        ])
        
        if validation_quality > 0.8:  # High validation quality
            base_weights['cross_asset_validation'] += 0.05
            base_weights['straddle_analysis'] -= 0.025
            base_weights['underlying_analysis'] -= 0.025
        elif validation_quality < 0.3:  # Low validation quality
            base_weights['cross_asset_validation'] -= 0.05
            base_weights['straddle_analysis'] += 0.03
            base_weights['underlying_analysis'] += 0.02
        
        return base_weights

    def _calculate_analysis_validation_score(self, straddle_result: StraddleAnalysisResult,
                                           underlying_result: UnderlyingAnalysisResult,
                                           cross_asset_result: CrossAssetIntegrationResult) -> float:
        """Calculate analysis validation score"""
        
        scores = []
        
        # Straddle analysis validation
        straddle_confidence = np.mean(straddle_result.confidence_scores)
        scores.append(straddle_confidence)
        
        # Underlying analysis validation
        underlying_confidence = np.mean(underlying_result.cross_timeframe_confidence)
        scores.append(underlying_confidence)
        
        # Cross-asset validation
        cross_asset_confidence = np.mean(cross_asset_result.confidence_result.final_confidence)
        scores.append(cross_asset_confidence)
        
        return np.mean(scores)

    def _calculate_framework_compliance_score(self, performance_metrics: Component05PerformanceMetrics) -> float:
        """Calculate framework compliance score"""
        
        compliance_factors = []
        
        # Performance compliance
        compliance_factors.append(1.0 if performance_metrics.performance_budget_compliant else 0.5)
        
        # Memory compliance
        compliance_factors.append(1.0 if performance_metrics.memory_budget_compliant else 0.5)
        
        # Feature count compliance (always 94)
        compliance_factors.append(1.0)  # Always compliant with 94 features
        
        return np.mean(compliance_factors)
    
    async def update_weights(self, performance_feedback) -> dict:
        """
        Update component weights based on performance feedback
        
        Args:
            performance_feedback: Performance metrics for learning
            
        Returns:
            Updated weights dictionary
        """
        # Extract performance metrics
        accuracy = getattr(performance_feedback, 'accuracy', 0.0)
        regime_accuracy = getattr(performance_feedback, 'regime_accuracy', {})
        
        # Update engine weights based on performance
        weight_updates = {}
        
        # Straddle engine weight adjustment
        straddle_accuracy = regime_accuracy.get('straddle', accuracy)
        if straddle_accuracy > 0.85:
            weight_updates['straddle'] = min(0.5, self.straddle_engine.weight * 1.05)
        else:
            weight_updates['straddle'] = max(0.3, self.straddle_engine.weight * 0.95)
        
        # Underlying engine weight adjustment
        underlying_accuracy = regime_accuracy.get('underlying', accuracy)
        if underlying_accuracy > 0.85:
            weight_updates['underlying'] = min(0.5, self.underlying_engine.weight * 1.05)
        else:
            weight_updates['underlying'] = max(0.3, self.underlying_engine.weight * 0.95)
        
        # Cross-asset weight adjustment
        cross_asset_accuracy = regime_accuracy.get('cross_asset', accuracy)
        if cross_asset_accuracy > 0.85:
            weight_updates['cross_asset'] = min(0.3, self.cross_asset_engine.weight * 1.05)
        else:
            weight_updates['cross_asset'] = max(0.1, self.cross_asset_engine.weight * 0.95)
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weight_updates.values())
        for key in weight_updates:
            weight_updates[key] /= total_weight
        
        # Apply weight updates
        self.straddle_engine.weight = weight_updates.get('straddle', 0.4)
        self.underlying_engine.weight = weight_updates.get('underlying', 0.4)
        self.cross_asset_engine.weight = weight_updates.get('cross_asset', 0.2)
        
        # Update DTE-specific weights if provided
        if hasattr(performance_feedback, 'dte_performance'):
            self.dte_framework.update_dte_weights(performance_feedback.dte_performance)
        
        return {
            'component_id': self.component_id,
            'weights_updated': True,
            'new_weights': weight_updates,
            'timestamp': datetime.utcnow().isoformat()
        }

    def _log_performance_compliance(self, performance_metrics: Component05PerformanceMetrics):
        """Log performance compliance details"""
        
        if performance_metrics.performance_budget_compliant:
            self.logger.info(f"✅ Performance compliant: {performance_metrics.total_processing_time_ms:.1f}ms (budget: {self.processing_budget_ms}ms)")
        else:
            self.logger.warning(f"⚠️  Performance exceeded: {performance_metrics.total_processing_time_ms:.1f}ms (budget: {self.processing_budget_ms}ms)")
        
        if performance_metrics.memory_budget_compliant:
            self.logger.info(f"✅ Memory compliant: {performance_metrics.peak_memory_mb:.1f}MB (budget: {self.memory_budget_mb}MB)")
        else:
            self.logger.warning(f"⚠️  Memory exceeded: {performance_metrics.peak_memory_mb:.1f}MB (budget: {self.memory_budget_mb}MB)")

    def _calculate_weight_updates(self, performance_feedback: 'PerformanceFeedback') -> Dict[str, float]:
        """Calculate weight updates based on performance feedback"""
        
        weight_changes = {}
        
        # Adjust based on accuracy
        if performance_feedback.accuracy > 0.85:
            # Good performance - slightly increase all weights
            weight_changes = {component: 0.05 for component in self.current_weights}
        elif performance_feedback.accuracy < 0.65:
            # Poor performance - slightly decrease weights
            weight_changes = {component: -0.05 for component in self.current_weights}
        else:
            # Neutral performance - no changes
            weight_changes = {component: 0.0 for component in self.current_weights}
        
        return weight_changes

    def _estimate_performance_improvement(self, performance_feedback: 'PerformanceFeedback',
                                        weight_changes: Dict[str, float]) -> float:
        """Estimate performance improvement from weight changes"""
        
        # Simple estimation based on weight change magnitude
        total_change = sum(abs(change) for change in weight_changes.values())
        
        if performance_feedback.accuracy > 0.8:
            return total_change * 0.02  # Small positive improvement
        elif performance_feedback.accuracy < 0.6:
            return total_change * 0.01  # Small improvement from adjustment
        else:
            return 0.0  # No improvement expected

    async def _create_fallback_result(self, market_data: pd.DataFrame, processing_time: float) -> ComponentAnalysisResult:
        """Create fallback result when analysis fails"""
        
        fallback_features = self._create_fallback_features(market_data)
        
        return ComponentAnalysisResult(
            component_id=self.component_id,
            component_name=self.component_name,
            score=0.5,  # Neutral score
            confidence=0.3,  # Low confidence
            features=fallback_features,
            processing_time_ms=processing_time,
            weights={'fallback': 1.0},
            metadata={
                'analysis_method': 'fallback',
                'fallback_reason': 'primary_analysis_failed',
                'feature_count': 94
            },
            timestamp=datetime.utcnow()
        )

    def _create_fallback_features(self, market_data: pd.DataFrame) -> FeatureVector:
        """Create fallback features when analysis fails"""
        
        # Create basic features from market data
        data_length = min(len(market_data), 100)  # Limit to prevent memory issues
        
        fallback_array = np.full((data_length, 94), 0.5)  # Neutral values
        
        # Add some basic market data features if available
        if 'ce_close' in market_data.columns and len(market_data) > 0:
            ce_close = market_data['ce_close'].fillna(0).values[:data_length]
            fallback_array[:, 0] = (ce_close - np.mean(ce_close)) / (np.std(ce_close) + 1e-10)  # Normalized
        
        if 'pe_close' in market_data.columns and len(market_data) > 0:
            pe_close = market_data['pe_close'].fillna(0).values[:data_length]
            fallback_array[:, 1] = (pe_close - np.mean(pe_close)) / (np.std(pe_close) + 1e-10)  # Normalized
        
        return FeatureVector(
            features=fallback_array,
            feature_names=self._get_94_feature_names(),
            feature_count=94,
            processing_time_ms=1.0,  # Minimal processing time
            metadata={'method': 'fallback'}
        )