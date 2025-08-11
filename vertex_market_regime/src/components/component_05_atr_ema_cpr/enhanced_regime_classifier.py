"""
Component 5: Enhanced Regime Classification System

Advanced regime classification system integrating dual-asset ATR-EMA-CPR analysis,
DTE-adaptive framework, and cross-asset validation for comprehensive market regime
determination with institutional flow detection and multi-layered confidence scoring.

Features:
- Integrated regime classification using both straddle and underlying analysis
- 8-regime classification enhanced with dual-asset insights
- Regime transition detection with cross-asset validation
- Institutional flow detection through dual-asset analysis
- Multi-layered confidence scoring with cross-asset validation
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
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

from .dual_asset_data_extractor import StraddlePriceData, UnderlyingPriceData
from .straddle_atr_ema_cpr_engine import StraddleAnalysisResult
from .underlying_atr_ema_cpr_engine import UnderlyingAnalysisResult
from .dual_dte_framework import DTEIntegratedResult
from .cross_asset_integration import CrossAssetIntegrationResult

warnings.filterwarnings('ignore')


@dataclass
class RegimeFeatures:
    """Features used for regime classification"""
    volatility_features: np.ndarray
    trend_features: np.ndarray
    momentum_features: np.ndarray
    cross_asset_features: np.ndarray
    dte_features: np.ndarray
    institutional_flow_features: np.ndarray
    confidence_features: np.ndarray
    feature_names: List[str]
    metadata: Dict[str, Any]


@dataclass
class RegimeTransitionResult:
    """Result of regime transition analysis"""
    transition_probabilities: np.ndarray
    transition_triggers: Dict[str, np.ndarray]
    transition_confidence: np.ndarray
    transition_momentum: np.ndarray
    regime_stability_score: np.ndarray
    upcoming_transition_signals: np.ndarray
    metadata: Dict[str, Any]


@dataclass
class InstitutionalFlowResult:
    """Result of institutional flow detection"""
    institutional_pressure: np.ndarray
    flow_direction: np.ndarray  # -1: selling, 0: neutral, 1: buying
    flow_magnitude: np.ndarray
    smart_money_signals: np.ndarray
    large_order_detection: np.ndarray
    flow_divergence_alerts: np.ndarray
    flow_confirmation_score: np.ndarray
    metadata: Dict[str, Any]


@dataclass
class MultiLayeredConfidenceResult:
    """Multi-layered confidence scoring result"""
    base_confidence: np.ndarray
    cross_asset_confidence: np.ndarray
    dte_confidence: np.ndarray
    regime_stability_confidence: np.ndarray
    institutional_flow_confidence: np.ndarray
    final_composite_confidence: np.ndarray
    confidence_tier: np.ndarray  # 1: Low, 2: Medium, 3: High
    confidence_breakdown: Dict[str, np.ndarray]
    metadata: Dict[str, Any]


@dataclass
class EnhancedRegimeClassificationResult:
    """Complete enhanced regime classification result"""
    regime_classification: np.ndarray  # 8-regime classification
    regime_probabilities: np.ndarray  # Probabilities for each regime
    regime_features: RegimeFeatures
    transition_result: RegimeTransitionResult
    institutional_flow_result: InstitutionalFlowResult
    confidence_result: MultiLayeredConfidenceResult
    regime_characteristics: Dict[int, Dict[str, Any]]
    trading_signals: Dict[str, np.ndarray]
    risk_warnings: Dict[str, np.ndarray]
    processing_time_ms: float
    metadata: Dict[str, Any]


class RegimeFeatureExtractor:
    """Extracts comprehensive features for regime classification"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")

    async def extract_regime_features(self, straddle_result: StraddleAnalysisResult,
                                    underlying_result: UnderlyingAnalysisResult,
                                    dte_result: DTEIntegratedResult,
                                    cross_asset_result: CrossAssetIntegrationResult) -> RegimeFeatures:
        """Extract comprehensive features for regime classification"""
        
        # Extract volatility features
        volatility_features = self._extract_volatility_features(straddle_result, underlying_result)
        
        # Extract trend features
        trend_features = self._extract_trend_features(straddle_result, underlying_result)
        
        # Extract momentum features
        momentum_features = self._extract_momentum_features(straddle_result, underlying_result)
        
        # Extract cross-asset features
        cross_asset_features = self._extract_cross_asset_features(cross_asset_result)
        
        # Extract DTE features
        dte_features = self._extract_dte_features(dte_result)
        
        # Extract institutional flow features
        institutional_flow_features = self._extract_institutional_flow_features(straddle_result, underlying_result)
        
        # Extract confidence features
        confidence_features = self._extract_confidence_features(
            straddle_result, underlying_result, cross_asset_result
        )
        
        # Create feature names
        feature_names = self._create_feature_names()
        
        return RegimeFeatures(
            volatility_features=volatility_features,
            trend_features=trend_features,
            momentum_features=momentum_features,
            cross_asset_features=cross_asset_features,
            dte_features=dte_features,
            institutional_flow_features=institutional_flow_features,
            confidence_features=confidence_features,
            feature_names=feature_names,
            metadata={
                'feature_extraction_method': 'comprehensive_dual_asset',
                'total_features': len(feature_names),
                'feature_categories': 7
            }
        )

    def _extract_volatility_features(self, straddle_result: StraddleAnalysisResult,
                                   underlying_result: UnderlyingAnalysisResult) -> np.ndarray:
        """Extract volatility-related features"""
        
        features = []
        data_length = len(straddle_result.atr_result.atr_14)
        
        # Straddle volatility features
        features.append(straddle_result.atr_result.atr_14)
        features.append(straddle_result.atr_result.volatility_regime.astype(float))
        features.append(straddle_result.atr_result.atr_trend.astype(float))
        
        # Underlying volatility features (daily)
        underlying_atr = underlying_result.atr_result.daily_atr.get('atr_14', np.zeros(data_length))
        underlying_vol_regime = underlying_result.atr_result.volatility_regimes.get('daily', np.zeros(data_length))
        
        features.append(underlying_atr)
        features.append(underlying_vol_regime.astype(float))
        
        # Volatility ratio and divergence
        vol_ratio = np.divide(straddle_result.atr_result.atr_14, underlying_atr, 
                             out=np.ones_like(straddle_result.atr_result.atr_14), where=underlying_atr!=0)
        features.append(vol_ratio)
        
        # Cross-timeframe volatility consistency
        cross_tf_consistency = underlying_result.atr_result.cross_timeframe_consistency.get(
            'all_timeframes', np.full(data_length, 0.5)
        )
        features.append(cross_tf_consistency)
        
        return np.column_stack(features)

    def _extract_trend_features(self, straddle_result: StraddleAnalysisResult,
                              underlying_result: UnderlyingAnalysisResult) -> np.ndarray:
        """Extract trend-related features"""
        
        features = []
        data_length = len(straddle_result.ema_result.trend_direction)
        
        # Straddle trend features
        features.append(straddle_result.ema_result.trend_direction.astype(float))
        features.append(straddle_result.ema_result.trend_strength)
        features.append(straddle_result.ema_result.confluence_zones)
        
        # Underlying trend features (daily)
        underlying_trend = underlying_result.ema_result.trend_directions.get('daily', np.zeros(data_length))
        underlying_strength = underlying_result.ema_result.trend_strengths.get('daily', np.zeros(data_length))
        
        features.append(underlying_trend.astype(float))
        features.append(underlying_strength)
        
        # Cross-timeframe trend agreement
        trend_agreement = underlying_result.ema_result.cross_timeframe_agreement.get(
            'all_timeframes', np.full(data_length, 0.5)
        )
        features.append(trend_agreement)
        
        # EMA slopes
        straddle_ema_20_slope = straddle_result.ema_result.slope_analysis.get('ema_20_slope', np.zeros(data_length))
        features.append(straddle_ema_20_slope)
        
        return np.column_stack(features)

    def _extract_momentum_features(self, straddle_result: StraddleAnalysisResult,
                                 underlying_result: UnderlyingAnalysisResult) -> np.ndarray:
        """Extract momentum-related features"""
        
        features = []
        data_length = len(straddle_result.atr_result.atr_14)
        
        # Price momentum (rate of change)
        straddle_close = straddle_result.feature_vector[:, 3] if straddle_result.feature_vector.shape[1] > 3 else np.zeros(data_length)  # Straddle close
        straddle_momentum = np.zeros_like(straddle_close)
        
        for i in range(5, len(straddle_close)):
            if straddle_close[i-5] != 0:
                straddle_momentum[i] = (straddle_close[i] - straddle_close[i-5]) / straddle_close[i-5]
        
        features.append(straddle_momentum)
        
        # Volume momentum
        straddle_volume = straddle_result.feature_vector[:, -5] if straddle_result.feature_vector.shape[1] > 5 else np.zeros(data_length)
        volume_momentum = np.zeros_like(straddle_volume)
        
        for i in range(5, len(straddle_volume)):
            if np.mean(straddle_volume[i-5:i]) != 0:
                volume_momentum[i] = (straddle_volume[i] - np.mean(straddle_volume[i-5:i])) / np.mean(straddle_volume[i-5:i])
        
        features.append(volume_momentum)
        
        # Acceleration (second derivative of price)
        price_acceleration = np.zeros_like(straddle_momentum)
        for i in range(1, len(straddle_momentum)):
            price_acceleration[i] = straddle_momentum[i] - straddle_momentum[i-1]
        
        features.append(price_acceleration)
        
        return np.column_stack(features)

    def _extract_cross_asset_features(self, cross_asset_result: CrossAssetIntegrationResult) -> np.ndarray:
        """Extract cross-asset validation features"""
        
        features = []
        
        # Validation scores
        if len(cross_asset_result.trend_validation.agreement_score) > 0:
            features.append(cross_asset_result.trend_validation.agreement_score)
        
        if len(cross_asset_result.volatility_validation.regime_agreement_score) > 0:
            features.append(cross_asset_result.volatility_validation.regime_agreement_score)
        
        if len(cross_asset_result.level_validation.level_agreement_score) > 0:
            features.append(cross_asset_result.level_validation.level_agreement_score)
        
        # Confidence features
        features.append(cross_asset_result.confidence_result.final_confidence)
        features.append(cross_asset_result.confidence_result.validation_boost)
        features.append(cross_asset_result.confidence_result.conflict_penalty)
        
        # Dynamic weights
        features.append(cross_asset_result.weighting_result.final_weights['straddle'])
        features.append(cross_asset_result.weighting_result.final_weights['underlying'])
        
        return np.column_stack(features) if features else np.array([]).reshape(0, 0)

    def _extract_dte_features(self, dte_result: DTEIntegratedResult) -> np.ndarray:
        """Extract DTE-related features"""
        
        features = []
        
        # DTE confidence from dual framework
        if len(dte_result.integrated_confidence) > 0:
            features.append(dte_result.integrated_confidence)
        
        # Feature contributions
        if 'dte_specific_contribution' in dte_result.feature_contributions:
            features.append(dte_result.feature_contributions['dte_specific_contribution'])
        
        if 'dte_range_contribution' in dte_result.feature_contributions:
            features.append(dte_result.feature_contributions['dte_range_contribution'])
        
        # Unified regime classification from DTE analysis
        if len(dte_result.unified_regime_classification) > 0:
            features.append(dte_result.unified_regime_classification.astype(float))
        
        return np.column_stack(features) if features else np.array([]).reshape(0, 0)

    def _extract_institutional_flow_features(self, straddle_result: StraddleAnalysisResult,
                                           underlying_result: UnderlyingAnalysisResult) -> np.ndarray:
        """Extract institutional flow indicators"""
        
        features = []
        data_length = len(straddle_result.atr_result.atr_14)
        
        # Volume-based institutional flow detection
        straddle_volume = straddle_result.feature_vector[:, -5] if straddle_result.feature_vector.shape[1] > 5 else np.ones(data_length)
        
        # Volume percentile (institutional activity indicator)
        volume_percentile = np.zeros_like(straddle_volume)
        for i in range(20, len(straddle_volume)):
            if i > 20:
                hist_volume = straddle_volume[i-20:i]
                volume_percentile[i] = stats.percentileofscore(hist_volume, straddle_volume[i]) / 100
        
        features.append(volume_percentile)
        
        # Open Interest changes (proxy from volume patterns)
        oi_momentum = np.zeros_like(straddle_volume)
        for i in range(5, len(straddle_volume)):
            recent_volume = np.mean(straddle_volume[i-5:i])
            older_volume = np.mean(straddle_volume[i-10:i-5]) if i >= 10 else recent_volume
            if older_volume > 0:
                oi_momentum[i] = (recent_volume - older_volume) / older_volume
        
        features.append(oi_momentum)
        
        # Large order detection (volume spikes with low volatility)
        large_order_indicator = np.zeros(data_length)
        for i in range(len(volume_percentile)):
            if (volume_percentile[i] > 0.8 and  # High volume
                straddle_result.atr_result.volatility_regime[i] <= 0):  # Low/medium volatility
                large_order_indicator[i] = 1
        
        features.append(large_order_indicator)
        
        return np.column_stack(features)

    def _extract_confidence_features(self, straddle_result: StraddleAnalysisResult,
                                   underlying_result: UnderlyingAnalysisResult,
                                   cross_asset_result: CrossAssetIntegrationResult) -> np.ndarray:
        """Extract confidence-related features"""
        
        features = []
        
        # Individual analysis confidence
        features.append(straddle_result.confidence_scores)
        features.append(underlying_result.cross_timeframe_confidence)
        
        # Cross-asset confidence
        features.append(cross_asset_result.confidence_result.final_confidence)
        
        # Confidence stability (rolling std of confidence)
        conf_stability = np.zeros_like(straddle_result.confidence_scores)
        for i in range(10, len(conf_stability)):
            conf_window = straddle_result.confidence_scores[i-10:i]
            conf_stability[i] = 1 - np.std(conf_window)  # Higher stability = lower std
        
        features.append(conf_stability)
        
        return np.column_stack(features)

    def _create_feature_names(self) -> List[str]:
        """Create descriptive feature names"""
        
        names = []
        
        # Volatility features (7 features)
        names.extend([
            'straddle_atr_14', 'straddle_vol_regime', 'straddle_atr_trend',
            'underlying_atr_14', 'underlying_vol_regime', 'vol_ratio',
            'cross_tf_vol_consistency'
        ])
        
        # Trend features (7 features)
        names.extend([
            'straddle_trend_direction', 'straddle_trend_strength', 'straddle_confluence',
            'underlying_trend_direction', 'underlying_trend_strength', 'cross_tf_trend_agreement',
            'straddle_ema_20_slope'
        ])
        
        # Momentum features (3 features)
        names.extend([
            'straddle_price_momentum', 'volume_momentum', 'price_acceleration'
        ])
        
        # Cross-asset features (8 features)
        names.extend([
            'trend_agreement', 'vol_agreement', 'level_agreement',
            'cross_asset_confidence', 'validation_boost', 'conflict_penalty',
            'straddle_weight', 'underlying_weight'
        ])
        
        # DTE features (4 features)
        names.extend([
            'dte_integrated_confidence', 'dte_specific_contribution',
            'dte_range_contribution', 'dte_unified_regime'
        ])
        
        # Institutional flow features (3 features)
        names.extend([
            'volume_percentile', 'oi_momentum', 'large_order_indicator'
        ])
        
        # Confidence features (4 features)
        names.extend([
            'straddle_confidence', 'underlying_confidence',
            'cross_asset_final_confidence', 'confidence_stability'
        ])
        
        return names


class RegimeTransitionAnalyzer:
    """Analyzes regime transitions with cross-asset validation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # Transition detection parameters
        self.transition_threshold = config.get('transition_threshold', 0.7)
        self.stability_window = config.get('stability_window', 10)

    async def analyze_regime_transitions(self, regime_features: RegimeFeatures,
                                       current_regimes: np.ndarray,
                                       cross_asset_result: CrossAssetIntegrationResult) -> RegimeTransitionResult:
        """Analyze regime transition probabilities and triggers"""
        
        data_length = len(current_regimes)
        
        # Calculate transition probabilities
        transition_probabilities = self._calculate_transition_probabilities(
            regime_features, current_regimes
        )
        
        # Identify transition triggers
        transition_triggers = self._identify_transition_triggers(
            regime_features, cross_asset_result
        )
        
        # Calculate transition confidence
        transition_confidence = self._calculate_transition_confidence(
            transition_probabilities, regime_features, cross_asset_result
        )
        
        # Analyze transition momentum
        transition_momentum = self._analyze_transition_momentum(
            regime_features, current_regimes
        )
        
        # Calculate regime stability
        regime_stability = self._calculate_regime_stability(
            current_regimes, regime_features
        )
        
        # Generate upcoming transition signals
        upcoming_transitions = self._generate_upcoming_transition_signals(
            transition_probabilities, transition_confidence, regime_stability
        )
        
        return RegimeTransitionResult(
            transition_probabilities=transition_probabilities,
            transition_triggers=transition_triggers,
            transition_confidence=transition_confidence,
            transition_momentum=transition_momentum,
            regime_stability_score=regime_stability,
            upcoming_transition_signals=upcoming_transitions,
            metadata={
                'transition_method': 'cross_asset_validated',
                'stability_window': self.stability_window,
                'transition_threshold': self.transition_threshold
            }
        )

    def _calculate_transition_probabilities(self, regime_features: RegimeFeatures,
                                          current_regimes: np.ndarray) -> np.ndarray:
        """Calculate probabilities of regime transitions"""
        
        transition_probs = np.zeros(len(current_regimes))
        
        # Use volatility and trend features to predict transitions
        vol_features = regime_features.volatility_features
        trend_features = regime_features.trend_features
        
        for i in range(self.stability_window, len(current_regimes)):
            # Recent regime stability
            recent_regimes = current_regimes[i-self.stability_window:i]
            regime_changes = np.sum(np.diff(recent_regimes) != 0)
            
            # Feature instability
            if vol_features.shape[0] > i and trend_features.shape[0] > i:
                vol_instability = np.std(vol_features[i-5:i, 1]) if i >= 5 else 0  # Vol regime std
                trend_instability = np.std(trend_features[i-5:i, 0]) if i >= 5 else 0  # Trend direction std
                
                # Combined instability score
                instability_score = (vol_instability + trend_instability) / 2
                
                # Transition probability based on instability and recent changes
                transition_probs[i] = min(1.0, (instability_score + regime_changes / self.stability_window) / 2)
        
        return transition_probs

    def _identify_transition_triggers(self, regime_features: RegimeFeatures,
                                    cross_asset_result: CrossAssetIntegrationResult) -> Dict[str, np.ndarray]:
        """Identify specific triggers for regime transitions"""
        
        data_length = len(regime_features.volatility_features) if len(regime_features.volatility_features) > 0 else 100
        triggers = {}
        
        # Volatility expansion trigger
        if len(regime_features.volatility_features) > 0:
            vol_regime = regime_features.volatility_features[:, 1]  # Volatility regime
            vol_expansion_trigger = np.zeros(data_length)
            
            for i in range(1, len(vol_regime)):
                if vol_regime[i] > vol_regime[i-1] and vol_regime[i] >= 1:  # Moving to high volatility
                    vol_expansion_trigger[i] = 1
            
            triggers['volatility_expansion'] = vol_expansion_trigger
        
        # Trend reversal trigger
        if len(regime_features.trend_features) > 0:
            trend_direction = regime_features.trend_features[:, 0]  # Trend direction
            trend_reversal_trigger = np.zeros(data_length)
            
            for i in range(1, len(trend_direction)):
                if (trend_direction[i] * trend_direction[i-1] < 0 and  # Direction change
                    abs(trend_direction[i]) == 1 and abs(trend_direction[i-1]) == 1):  # Both non-neutral
                    trend_reversal_trigger[i] = 1
            
            triggers['trend_reversal'] = trend_reversal_trigger
        
        # Cross-asset disagreement trigger
        if len(cross_asset_result.trend_validation.conflicting_signals) > 0:
            triggers['cross_asset_conflict'] = cross_asset_result.trend_validation.conflicting_signals
        
        # Institutional flow trigger
        if len(regime_features.institutional_flow_features) > 0:
            large_orders = regime_features.institutional_flow_features[:, 2]  # Large order indicator
            triggers['institutional_activity'] = large_orders
        
        return triggers

    def _calculate_transition_confidence(self, transition_probs: np.ndarray,
                                       regime_features: RegimeFeatures,
                                       cross_asset_result: CrossAssetIntegrationResult) -> np.ndarray:
        """Calculate confidence in transition predictions"""
        
        # Base confidence from cross-asset validation
        base_confidence = cross_asset_result.confidence_result.final_confidence
        
        # Adjust based on feature confidence
        feature_confidence = regime_features.confidence_features[:, 2] if len(regime_features.confidence_features) > 0 else base_confidence  # Cross-asset confidence
        
        # Combined confidence
        transition_confidence = (base_confidence * 0.6 + feature_confidence * 0.4)
        
        # Penalize confidence during high transition probability periods
        high_transition_penalty = transition_probs * 0.2
        transition_confidence = np.clip(transition_confidence - high_transition_penalty, 0.1, 0.95)
        
        return transition_confidence

    def _analyze_transition_momentum(self, regime_features: RegimeFeatures,
                                   current_regimes: np.ndarray) -> np.ndarray:
        """Analyze momentum of regime transitions"""
        
        momentum = np.zeros(len(current_regimes))
        
        # Use momentum features if available
        if len(regime_features.momentum_features) > 0:
            price_momentum = regime_features.momentum_features[:, 0]  # Price momentum
            momentum = np.abs(price_momentum)  # Absolute momentum as transition indicator
        
        # Combine with regime change momentum
        for i in range(5, len(current_regimes)):
            regime_momentum = np.sum(np.diff(current_regimes[i-5:i]) != 0) / 4  # Normalized regime changes
            momentum[i] = (momentum[i] + regime_momentum) / 2 if len(regime_features.momentum_features) > 0 else regime_momentum
        
        return momentum

    def _calculate_regime_stability(self, current_regimes: np.ndarray,
                                  regime_features: RegimeFeatures) -> np.ndarray:
        """Calculate regime stability scores"""
        
        stability = np.ones(len(current_regimes))  # Default stable
        
        for i in range(self.stability_window, len(current_regimes)):
            # Recent regime consistency
            recent_regimes = current_regimes[i-self.stability_window:i]
            unique_regimes = len(np.unique(recent_regimes))
            regime_consistency = 1.0 - (unique_regimes - 1) / (self.stability_window - 1)
            
            # Feature stability
            feature_stability = 1.0
            if len(regime_features.confidence_features) > 0:
                confidence_stability = regime_features.confidence_features[i, 3] if i < len(regime_features.confidence_features) else 0.5  # Confidence stability
                feature_stability = confidence_stability
            
            # Combined stability
            stability[i] = (regime_consistency * 0.7 + feature_stability * 0.3)
        
        return stability

    def _generate_upcoming_transition_signals(self, transition_probs: np.ndarray,
                                            transition_confidence: np.ndarray,
                                            regime_stability: np.ndarray) -> np.ndarray:
        """Generate signals for upcoming regime transitions"""
        
        upcoming_signals = np.zeros(len(transition_probs))
        
        for i in range(len(transition_probs)):
            if (transition_probs[i] > self.transition_threshold and
                transition_confidence[i] > 0.6 and
                regime_stability[i] < 0.5):  # High transition prob, good confidence, low stability
                upcoming_signals[i] = 1
        
        return upcoming_signals


class InstitutionalFlowDetector:
    """Detects institutional flow patterns using dual-asset analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # Flow detection parameters
        self.large_order_threshold = config.get('large_order_threshold', 0.8)
        self.flow_confirmation_window = config.get('flow_confirmation_window', 5)

    async def detect_institutional_flow(self, regime_features: RegimeFeatures,
                                      straddle_result: StraddleAnalysisResult,
                                      underlying_result: UnderlyingAnalysisResult) -> InstitutionalFlowResult:
        """Detect institutional flow patterns"""
        
        data_length = len(regime_features.institutional_flow_features) if len(regime_features.institutional_flow_features) > 0 else 100
        
        # Calculate institutional pressure
        institutional_pressure = self._calculate_institutional_pressure(regime_features)
        
        # Determine flow direction
        flow_direction = self._determine_flow_direction(
            regime_features, straddle_result, underlying_result
        )
        
        # Calculate flow magnitude
        flow_magnitude = self._calculate_flow_magnitude(regime_features)
        
        # Identify smart money signals
        smart_money_signals = self._identify_smart_money_signals(
            regime_features, flow_direction, institutional_pressure
        )
        
        # Detect large orders
        large_order_detection = self._detect_large_orders(regime_features)
        
        # Generate flow divergence alerts
        flow_divergence_alerts = self._generate_flow_divergence_alerts(
            flow_direction, straddle_result, underlying_result
        )
        
        # Calculate flow confirmation scores
        flow_confirmation = self._calculate_flow_confirmation(
            flow_direction, institutional_pressure, regime_features
        )
        
        return InstitutionalFlowResult(
            institutional_pressure=institutional_pressure,
            flow_direction=flow_direction,
            flow_magnitude=flow_magnitude,
            smart_money_signals=smart_money_signals,
            large_order_detection=large_order_detection,
            flow_divergence_alerts=flow_divergence_alerts,
            flow_confirmation_score=flow_confirmation,
            metadata={
                'detection_method': 'dual_asset_flow_analysis',
                'large_order_threshold': self.large_order_threshold,
                'confirmation_window': self.flow_confirmation_window
            }
        )

    def _calculate_institutional_pressure(self, regime_features: RegimeFeatures) -> np.ndarray:
        """Calculate institutional pressure indicator"""
        
        if len(regime_features.institutional_flow_features) == 0:
            return np.zeros(100)
        
        # Use volume percentile as primary institutional pressure indicator
        volume_percentile = regime_features.institutional_flow_features[:, 0]
        oi_momentum = regime_features.institutional_flow_features[:, 1]
        
        # Combined institutional pressure
        institutional_pressure = (volume_percentile * 0.7 + np.abs(oi_momentum) * 0.3)
        
        return institutional_pressure

    def _determine_flow_direction(self, regime_features: RegimeFeatures,
                                straddle_result: StraddleAnalysisResult,
                                underlying_result: UnderlyingAnalysisResult) -> np.ndarray:
        """Determine institutional flow direction"""
        
        data_length = len(regime_features.institutional_flow_features) if len(regime_features.institutional_flow_features) > 0 else 100
        flow_direction = np.zeros(data_length)
        
        # Use price momentum and trend agreement to infer direction
        if len(regime_features.momentum_features) > 0:
            price_momentum = regime_features.momentum_features[:, 0]  # Price momentum
            
            for i in range(len(flow_direction)):
                if i < len(price_momentum):
                    if price_momentum[i] > 0.02:  # Positive momentum > 2%
                        flow_direction[i] = 1  # Buying pressure
                    elif price_momentum[i] < -0.02:  # Negative momentum < -2%
                        flow_direction[i] = -1  # Selling pressure
                    else:
                        flow_direction[i] = 0  # Neutral
        
        # Confirm with trend agreement
        if len(regime_features.cross_asset_features) > 0:
            trend_agreement = regime_features.cross_asset_features[:, 0]  # Trend agreement
            
            for i in range(len(flow_direction)):
                if i < len(trend_agreement) and trend_agreement[i] < 0.3:  # Low agreement
                    flow_direction[i] = 0  # Neutralize on low agreement
        
        return flow_direction

    def _calculate_flow_magnitude(self, regime_features: RegimeFeatures) -> np.ndarray:
        """Calculate magnitude of institutional flow"""
        
        if len(regime_features.institutional_flow_features) == 0:
            return np.zeros(100)
        
        volume_percentile = regime_features.institutional_flow_features[:, 0]
        oi_momentum = regime_features.institutional_flow_features[:, 1]
        
        # Magnitude combines volume intensity and OI changes
        magnitude = np.sqrt(volume_percentile**2 + (np.abs(oi_momentum) * 2)**2) / 2
        
        return magnitude

    def _identify_smart_money_signals(self, regime_features: RegimeFeatures,
                                    flow_direction: np.ndarray,
                                    institutional_pressure: np.ndarray) -> np.ndarray:
        """Identify smart money activity signals"""
        
        smart_money = np.zeros(len(flow_direction))
        
        for i in range(len(smart_money)):
            # Smart money characteristics:
            # 1. High institutional pressure
            # 2. Clear directional flow
            # 3. Counter-trend or early trend identification
            
            if (i < len(institutional_pressure) and
                institutional_pressure[i] > 0.7 and  # High pressure
                abs(flow_direction[i]) == 1):  # Clear direction
                
                # Check if this is counter to recent trend (smart money often contrarian)
                if i >= 10:
                    recent_flow = flow_direction[i-10:i]
                    if len(recent_flow[recent_flow != 0]) > 0:
                        dominant_flow = 1 if np.sum(recent_flow > 0) > np.sum(recent_flow < 0) else -1
                        if flow_direction[i] != dominant_flow:  # Counter-trend
                            smart_money[i] = 1
        
        return smart_money

    def _detect_large_orders(self, regime_features: RegimeFeatures) -> np.ndarray:
        """Detect large institutional orders"""
        
        if len(regime_features.institutional_flow_features) == 0:
            return np.zeros(100)
        
        # Use the pre-calculated large order indicator
        return regime_features.institutional_flow_features[:, 2]

    def _generate_flow_divergence_alerts(self, flow_direction: np.ndarray,
                                       straddle_result: StraddleAnalysisResult,
                                       underlying_result: UnderlyingAnalysisResult) -> np.ndarray:
        """Generate alerts for flow divergence between assets"""
        
        divergence_alerts = np.zeros(len(flow_direction))
        
        # Compare flow direction with trend agreement
        straddle_trend = straddle_result.ema_result.trend_direction
        underlying_trend = underlying_result.ema_result.trend_directions.get('daily', np.array([]))
        
        min_length = min(len(flow_direction), len(straddle_trend), len(underlying_trend))
        
        for i in range(min_length):
            # Alert when flow direction disagrees with both trend analyses
            if (flow_direction[i] != 0 and
                straddle_trend[i] != 0 and underlying_trend[i] != 0 and
                flow_direction[i] != straddle_trend[i] and
                flow_direction[i] != underlying_trend[i]):
                divergence_alerts[i] = 1
        
        return divergence_alerts

    def _calculate_flow_confirmation(self, flow_direction: np.ndarray,
                                   institutional_pressure: np.ndarray,
                                   regime_features: RegimeFeatures) -> np.ndarray:
        """Calculate flow confirmation scores"""
        
        confirmation = np.zeros(len(flow_direction))
        
        for i in range(self.flow_confirmation_window, len(flow_direction)):
            # Recent flow consistency
            recent_flow = flow_direction[i-self.flow_confirmation_window:i]
            
            if len(recent_flow[recent_flow != 0]) > 0:
                # Consistency in direction
                positive_flow = np.sum(recent_flow == 1)
                negative_flow = np.sum(recent_flow == -1)
                total_directional = positive_flow + negative_flow
                
                if total_directional > 0:
                    directional_consistency = max(positive_flow, negative_flow) / total_directional
                    
                    # Combine with institutional pressure
                    avg_pressure = np.mean(institutional_pressure[i-self.flow_confirmation_window:i])
                    confirmation[i] = (directional_consistency * 0.7 + avg_pressure * 0.3)
        
        return confirmation


class MultiLayeredConfidenceScorer:
    """Multi-layered confidence scoring system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")

    async def calculate_multi_layered_confidence(self, regime_features: RegimeFeatures,
                                               cross_asset_result: CrossAssetIntegrationResult,
                                               dte_result: DTEIntegratedResult,
                                               transition_result: RegimeTransitionResult,
                                               institutional_result: InstitutionalFlowResult) -> MultiLayeredConfidenceResult:
        """Calculate comprehensive multi-layered confidence scores"""
        
        data_length = len(regime_features.confidence_features) if len(regime_features.confidence_features) > 0 else 100
        
        # Base confidence from individual analyses
        base_confidence = self._calculate_base_confidence(regime_features)
        
        # Cross-asset validation confidence
        cross_asset_confidence = cross_asset_result.confidence_result.final_confidence
        
        # DTE-specific confidence
        dte_confidence = self._calculate_dte_confidence(dte_result, data_length)
        
        # Regime stability confidence
        regime_stability_confidence = transition_result.regime_stability_score
        
        # Institutional flow confidence
        institutional_flow_confidence = institutional_result.flow_confirmation_score
        
        # Calculate final composite confidence
        final_confidence = self._calculate_composite_confidence(
            base_confidence, cross_asset_confidence, dte_confidence,
            regime_stability_confidence, institutional_flow_confidence
        )
        
        # Determine confidence tiers
        confidence_tiers = self._determine_confidence_tiers(final_confidence)
        
        # Create detailed breakdown
        confidence_breakdown = self._create_confidence_breakdown(
            base_confidence, cross_asset_confidence, dte_confidence,
            regime_stability_confidence, institutional_flow_confidence
        )
        
        return MultiLayeredConfidenceResult(
            base_confidence=base_confidence,
            cross_asset_confidence=cross_asset_confidence,
            dte_confidence=dte_confidence,
            regime_stability_confidence=regime_stability_confidence,
            institutional_flow_confidence=institutional_flow_confidence,
            final_composite_confidence=final_confidence,
            confidence_tier=confidence_tiers,
            confidence_breakdown=confidence_breakdown,
            metadata={
                'confidence_layers': 5,
                'scoring_method': 'weighted_composite',
                'tier_thresholds': [0.4, 0.7, 0.9]
            }
        )

    def _calculate_base_confidence(self, regime_features: RegimeFeatures) -> np.ndarray:
        """Calculate base confidence from feature quality"""
        
        if len(regime_features.confidence_features) == 0:
            return np.full(100, 0.5)
        
        # Use existing confidence features
        straddle_conf = regime_features.confidence_features[:, 0]  # Straddle confidence
        underlying_conf = regime_features.confidence_features[:, 1]  # Underlying confidence
        
        # Weighted average
        base_confidence = straddle_conf * 0.6 + underlying_conf * 0.4
        
        return base_confidence

    def _calculate_dte_confidence(self, dte_result: DTEIntegratedResult, data_length: int) -> np.ndarray:
        """Calculate DTE-specific confidence"""
        
        if len(dte_result.integrated_confidence) >= data_length:
            return dte_result.integrated_confidence[:data_length]
        else:
            return np.full(data_length, 0.5)  # Default medium confidence

    def _calculate_composite_confidence(self, base_confidence: np.ndarray,
                                      cross_asset_confidence: np.ndarray,
                                      dte_confidence: np.ndarray,
                                      regime_stability_confidence: np.ndarray,
                                      institutional_flow_confidence: np.ndarray) -> np.ndarray:
        """Calculate weighted composite confidence"""
        
        # Ensure all arrays have same length
        min_length = min(len(base_confidence), len(cross_asset_confidence), 
                        len(dte_confidence), len(regime_stability_confidence),
                        len(institutional_flow_confidence))
        
        if min_length == 0:
            return np.array([0.5])
        
        # Truncate all arrays to min_length
        base = base_confidence[:min_length]
        cross_asset = cross_asset_confidence[:min_length]
        dte = dte_confidence[:min_length]
        stability = regime_stability_confidence[:min_length]
        institutional = institutional_flow_confidence[:min_length]
        
        # Weighted combination
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # Base, Cross-asset, DTE, Stability, Institutional
        
        composite = (base * weights[0] + 
                    cross_asset * weights[1] +
                    dte * weights[2] +
                    stability * weights[3] +
                    institutional * weights[4])
        
        return np.clip(composite, 0.1, 0.95)

    def _determine_confidence_tiers(self, final_confidence: np.ndarray) -> np.ndarray:
        """Determine confidence tiers (1: Low, 2: Medium, 3: High)"""
        
        tiers = np.ones(len(final_confidence), dtype=int)  # Default low
        
        tiers[final_confidence >= 0.4] = 1  # Low confidence
        tiers[final_confidence >= 0.7] = 2  # Medium confidence
        tiers[final_confidence >= 0.9] = 3  # High confidence
        
        return tiers

    def _create_confidence_breakdown(self, base_confidence: np.ndarray,
                                   cross_asset_confidence: np.ndarray,
                                   dte_confidence: np.ndarray,
                                   regime_stability_confidence: np.ndarray,
                                   institutional_flow_confidence: np.ndarray) -> Dict[str, np.ndarray]:
        """Create detailed confidence breakdown"""
        
        min_length = min(len(base_confidence), len(cross_asset_confidence))
        
        return {
            'base_layer_contribution': base_confidence[:min_length] * 0.3,
            'cross_asset_layer_contribution': cross_asset_confidence[:min_length] * 0.25,
            'dte_layer_contribution': dte_confidence[:min_length] * 0.2,
            'stability_layer_contribution': regime_stability_confidence[:min_length] * 0.15,
            'institutional_layer_contribution': institutional_flow_confidence[:min_length] * 0.1,
            'layer_weights': np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        }


class EnhancedRegimeClassificationEngine:
    """Main engine for enhanced regime classification"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # Initialize sub-engines
        self.feature_extractor = RegimeFeatureExtractor(config)
        self.transition_analyzer = RegimeTransitionAnalyzer(config)
        self.flow_detector = InstitutionalFlowDetector(config)
        self.confidence_scorer = MultiLayeredConfidenceScorer(config)
        
        # Classification parameters
        self.regime_definitions = self._define_regime_characteristics()

    def _define_regime_characteristics(self) -> Dict[int, Dict[str, Any]]:
        """Define characteristics of each regime"""
        
        return {
            0: {'name': 'Low Vol Bullish', 'volatility': 'low', 'trend': 'bullish', 'risk': 'moderate'},
            1: {'name': 'Low Vol Bearish', 'volatility': 'low', 'trend': 'bearish', 'risk': 'moderate'},
            2: {'name': 'Medium Vol Bullish', 'volatility': 'medium', 'trend': 'bullish', 'risk': 'moderate'},
            3: {'name': 'Medium Vol Bearish', 'volatility': 'medium', 'trend': 'bearish', 'risk': 'moderate'},
            4: {'name': 'High Vol Bullish', 'volatility': 'high', 'trend': 'bullish', 'risk': 'high'},
            5: {'name': 'High Vol Bearish', 'volatility': 'high', 'trend': 'bearish', 'risk': 'high'},
            6: {'name': 'Extreme Volatility', 'volatility': 'extreme', 'trend': 'neutral', 'risk': 'very_high'},
            7: {'name': 'Neutral Transition', 'volatility': 'variable', 'trend': 'neutral', 'risk': 'high'}
        }

    async def classify_enhanced_regimes(self, straddle_result: StraddleAnalysisResult,
                                      underlying_result: UnderlyingAnalysisResult,
                                      dte_result: DTEIntegratedResult,
                                      cross_asset_result: CrossAssetIntegrationResult) -> EnhancedRegimeClassificationResult:
        """Perform enhanced regime classification with all features"""
        
        start_time = time.time()
        
        try:
            # Extract comprehensive features
            regime_features = await self.feature_extractor.extract_regime_features(
                straddle_result, underlying_result, dte_result, cross_asset_result
            )
            
            # Perform initial regime classification
            initial_regimes = self._perform_initial_classification(regime_features, cross_asset_result)
            
            # Analyze regime transitions
            transition_result = await self.transition_analyzer.analyze_regime_transitions(
                regime_features, initial_regimes, cross_asset_result
            )
            
            # Detect institutional flow
            institutional_result = await self.flow_detector.detect_institutional_flow(
                regime_features, straddle_result, underlying_result
            )
            
            # Calculate multi-layered confidence
            confidence_result = await self.confidence_scorer.calculate_multi_layered_confidence(
                regime_features, cross_asset_result, dte_result, 
                transition_result, institutional_result
            )
            
            # Refine regime classification with all information
            final_regimes, regime_probabilities = self._refine_regime_classification(
                initial_regimes, regime_features, transition_result, 
                institutional_result, confidence_result
            )
            
            # Generate trading signals
            trading_signals = self._generate_trading_signals(
                final_regimes, regime_features, transition_result, institutional_result
            )
            
            # Generate risk warnings
            risk_warnings = self._generate_risk_warnings(
                final_regimes, transition_result, confidence_result
            )
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            return EnhancedRegimeClassificationResult(
                regime_classification=final_regimes,
                regime_probabilities=regime_probabilities,
                regime_features=regime_features,
                transition_result=transition_result,
                institutional_flow_result=institutional_result,
                confidence_result=confidence_result,
                regime_characteristics=self.regime_definitions,
                trading_signals=trading_signals,
                risk_warnings=risk_warnings,
                processing_time_ms=processing_time_ms,
                metadata={
                    'classification_engine': 'enhanced_dual_asset',
                    'total_features': len(regime_features.feature_names),
                    'regimes': 8,
                    'confidence_layers': 5
                }
            )
            
        except Exception as e:
            self.logger.error(f"Enhanced regime classification failed: {str(e)}")
            raise

    def _perform_initial_classification(self, regime_features: RegimeFeatures,
                                      cross_asset_result: CrossAssetIntegrationResult) -> np.ndarray:
        """Perform initial regime classification using cross-asset analysis"""
        
        # Use cross-asset regime classification as starting point
        initial_regimes = cross_asset_result.cross_asset_regime_classification
        
        # Refine with volatility and trend features
        if len(regime_features.volatility_features) > 0 and len(regime_features.trend_features) > 0:
            vol_regime = regime_features.volatility_features[:, 1].astype(int)  # Straddle vol regime
            trend_direction = regime_features.trend_features[:, 0].astype(int)  # Straddle trend
            
            # Combine vol and trend for refined classification
            for i in range(len(initial_regimes)):
                if i < len(vol_regime) and i < len(trend_direction):
                    vol = vol_regime[i]
                    trend = trend_direction[i]
                    
                    # Map to 8-regime system
                    if vol <= -1 and trend >= 1:
                        initial_regimes[i] = 0  # Low vol bullish
                    elif vol <= -1 and trend <= -1:
                        initial_regimes[i] = 1  # Low vol bearish
                    elif vol == 0 and trend >= 1:
                        initial_regimes[i] = 2  # Medium vol bullish
                    elif vol == 0 and trend <= -1:
                        initial_regimes[i] = 3  # Medium vol bearish
                    elif vol >= 1 and trend >= 1:
                        initial_regimes[i] = 4  # High vol bullish
                    elif vol >= 1 and trend <= -1:
                        initial_regimes[i] = 5  # High vol bearish
                    elif vol >= 2:
                        initial_regimes[i] = 6  # Extreme volatility
                    else:
                        initial_regimes[i] = 7  # Neutral/transition
        
        return initial_regimes

    def _refine_regime_classification(self, initial_regimes: np.ndarray,
                                    regime_features: RegimeFeatures,
                                    transition_result: RegimeTransitionResult,
                                    institutional_result: InstitutionalFlowResult,
                                    confidence_result: MultiLayeredConfidenceResult) -> Tuple[np.ndarray, np.ndarray]:
        """Refine regime classification with all available information"""
        
        refined_regimes = initial_regimes.copy()
        regime_probabilities = np.zeros((len(refined_regimes), 8))  # 8 regimes
        
        for i in range(len(refined_regimes)):
            # Base probability distribution (one-hot for initial regime)
            base_probs = np.zeros(8)
            if 0 <= initial_regimes[i] <= 7:
                base_probs[int(initial_regimes[i])] = 0.7  # 70% confidence in initial
                base_probs += 0.3 / 7  # Distribute remaining 30% across others
            
            # Adjust based on transition probability
            if i < len(transition_result.transition_probabilities):
                transition_prob = transition_result.transition_probabilities[i]
                if transition_prob > 0.7:  # High transition probability
                    # Increase probability of transition regime
                    base_probs[7] += 0.2  # Increase neutral/transition
                    base_probs = base_probs / np.sum(base_probs)  # Renormalize
            
            # Adjust based on institutional flow
            if i < len(institutional_result.smart_money_signals):
                if institutional_result.smart_money_signals[i] == 1:
                    # Smart money activity - might indicate regime change
                    base_probs[7] += 0.1
                    base_probs = base_probs / np.sum(base_probs)
            
            # Adjust based on confidence
            if i < len(confidence_result.final_composite_confidence):
                confidence = confidence_result.final_composite_confidence[i]
                if confidence < 0.4:  # Low confidence
                    # Flatten distribution
                    base_probs = base_probs * 0.5 + np.ones(8) * 0.5 / 8
            
            regime_probabilities[i] = base_probs
            
            # Final regime is most probable
            refined_regimes[i] = np.argmax(base_probs)
        
        return refined_regimes, regime_probabilities

    def _generate_trading_signals(self, regimes: np.ndarray,
                                regime_features: RegimeFeatures,
                                transition_result: RegimeTransitionResult,
                                institutional_result: InstitutionalFlowResult) -> Dict[str, np.ndarray]:
        """Generate trading signals based on regime classification"""
        
        signals = {}
        
        # Regime-based signals
        signals['bullish_regime'] = np.isin(regimes, [0, 2, 4]).astype(int)  # Bullish regimes
        signals['bearish_regime'] = np.isin(regimes, [1, 3, 5]).astype(int)  # Bearish regimes
        signals['high_vol_regime'] = np.isin(regimes, [4, 5, 6]).astype(int)  # High volatility regimes
        
        # Transition signals
        signals['regime_transition_imminent'] = transition_result.upcoming_transition_signals
        
        # Institutional flow signals
        signals['smart_money_activity'] = institutional_result.smart_money_signals
        signals['large_institutional_orders'] = institutional_result.large_order_detection
        
        # Combined signals
        signals['high_conviction_bullish'] = np.logical_and(
            signals['bullish_regime'],
            institutional_result.flow_direction == 1
        ).astype(int)
        
        signals['high_conviction_bearish'] = np.logical_and(
            signals['bearish_regime'],
            institutional_result.flow_direction == -1
        ).astype(int)
        
        return signals

    def _generate_risk_warnings(self, regimes: np.ndarray,
                              transition_result: RegimeTransitionResult,
                              confidence_result: MultiLayeredConfidenceResult) -> Dict[str, np.ndarray]:
        """Generate risk warnings based on regime analysis"""
        
        warnings = {}
        
        # High volatility warning
        warnings['extreme_volatility_warning'] = (regimes == 6).astype(int)
        
        # Regime uncertainty warning
        warnings['regime_uncertainty_warning'] = (
            confidence_result.confidence_tier == 1
        ).astype(int)
        
        # Transition risk warning
        warnings['transition_risk_warning'] = (
            transition_result.transition_probabilities > 0.8
        ).astype(int)
        
        # Low stability warning
        warnings['low_stability_warning'] = (
            transition_result.regime_stability_score < 0.3
        ).astype(int)
        
        return warnings