"""
Master Integration Feature Engineering Engine

Generates 48 cross-component integration features for ML consumption.
No classification logic - pure feature engineering only.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class IntegrationFeatures:
    """Container for 48 master integration features"""
    
    # Component Agreement Features (12)
    pairwise_agreement_mean: float
    pairwise_agreement_std: float
    multi_component_consensus: float
    signal_divergence_score: float
    weighted_agreement_score: float
    component_alignment_ratio: float
    agreement_stability: float
    agreement_momentum: float
    agreement_acceleration: float
    agreement_entropy: float
    agreement_concentration: float
    agreement_dispersion: float
    
    # DTE-Adaptive Features (12)
    dte_specific_pattern: float
    dte_weekly_pattern: float
    dte_monthly_pattern: float
    dte_far_pattern: float
    dte_transition_score: float
    dte_evolution_rate: float
    dte_performance_mean: float
    dte_performance_std: float
    dte_reliability_score: float
    dte_consistency_score: float
    dte_adaptation_rate: float
    dte_efficiency_ratio: float
    
    # System Coherence Metrics (12)
    signal_stability_score: float
    transition_probability: float
    integration_quality: float
    coherence_index: float
    system_entropy: float
    signal_noise_ratio: float
    confidence_mean: float
    confidence_std: float
    health_score: float
    robustness_metric: float
    consistency_score: float
    reliability_index: float
    
    # Cross-Component Synergies (12)
    synergy_score_mean: float
    synergy_score_std: float
    interaction_effect_strength: float
    triad_synergy_score: float
    complementary_signal_ratio: float
    antagonistic_signal_ratio: float
    synergy_concentration: float
    synergy_dispersion: float
    synergy_momentum: float
    synergy_acceleration: float
    synergy_stability: float
    synergy_efficiency: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert features to dictionary"""
        return {
            # Component Agreement Features
            'agreement_mean': self.pairwise_agreement_mean,
            'agreement_std': self.pairwise_agreement_std,
            'consensus_score': self.multi_component_consensus,
            'divergence_score': self.signal_divergence_score,
            'weighted_agreement': self.weighted_agreement_score,
            'alignment_ratio': self.component_alignment_ratio,
            'agreement_stability': self.agreement_stability,
            'agreement_momentum': self.agreement_momentum,
            'agreement_acceleration': self.agreement_acceleration,
            'agreement_entropy': self.agreement_entropy,
            'agreement_concentration': self.agreement_concentration,
            'agreement_dispersion': self.agreement_dispersion,
            
            # DTE-Adaptive Features
            'dte_specific': self.dte_specific_pattern,
            'dte_weekly': self.dte_weekly_pattern,
            'dte_monthly': self.dte_monthly_pattern,
            'dte_far': self.dte_far_pattern,
            'dte_transition': self.dte_transition_score,
            'dte_evolution': self.dte_evolution_rate,
            'dte_perf_mean': self.dte_performance_mean,
            'dte_perf_std': self.dte_performance_std,
            'dte_reliability': self.dte_reliability_score,
            'dte_consistency': self.dte_consistency_score,
            'dte_adaptation': self.dte_adaptation_rate,
            'dte_efficiency': self.dte_efficiency_ratio,
            
            # System Coherence Metrics
            'stability_score': self.signal_stability_score,
            'transition_prob': self.transition_probability,
            'integration_quality': self.integration_quality,
            'coherence_index': self.coherence_index,
            'system_entropy': self.system_entropy,
            'signal_noise_ratio': self.signal_noise_ratio,
            'confidence_mean': self.confidence_mean,
            'confidence_std': self.confidence_std,
            'health_score': self.health_score,
            'robustness_metric': self.robustness_metric,
            'consistency_score': self.consistency_score,
            'reliability_index': self.reliability_index,
            
            # Cross-Component Synergies
            'synergy_mean': self.synergy_score_mean,
            'synergy_std': self.synergy_score_std,
            'interaction_strength': self.interaction_effect_strength,
            'triad_synergy': self.triad_synergy_score,
            'complementary_ratio': self.complementary_signal_ratio,
            'antagonistic_ratio': self.antagonistic_signal_ratio,
            'synergy_concentration': self.synergy_concentration,
            'synergy_dispersion': self.synergy_dispersion,
            'synergy_momentum': self.synergy_momentum,
            'synergy_acceleration': self.synergy_acceleration,
            'synergy_stability': self.synergy_stability,
            'synergy_efficiency': self.synergy_efficiency
        }


class MasterIntegrationFeatureEngine:
    """
    Master Integration Feature Engineering Engine
    
    Generates 48 cross-component integration features from Components 1-7 outputs.
    Pure feature engineering with no classification logic.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the feature engineering engine
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.processing_budget_ms = self.config.get('processing_budget_ms', 100)
        self.memory_budget_mb = self.config.get('memory_budget_mb', 150)
        
        # Component weights for weighted features
        self.component_weights = self.config.get('component_weights', {
            'component_01': 1.0,
            'component_02': 1.0,
            'component_03': 1.0,
            'component_04': 1.0,
            'component_05': 1.0,
            'component_06': 1.0,
            'component_07': 1.0
        })
        
        # DTE range definitions
        self.dte_ranges = {
            'weekly': (0, 7),
            'monthly': (8, 30),
            'far': (31, 90)
        }
        
        # Feature validation
        self.expected_feature_count = 48
        
        logger.info(f"Initialized MasterIntegrationFeatureEngine with budget: {self.processing_budget_ms}ms")
    
    def extract_features(
        self,
        component_outputs: Dict[str, Dict[str, Any]],
        dte: Optional[int] = None
    ) -> IntegrationFeatures:
        """
        Extract 48 integration features from component outputs
        
        Args:
            component_outputs: Dictionary of component outputs
            dte: Days to expiry (optional)
            
        Returns:
            IntegrationFeatures object containing all 48 features
        """
        start_time = time.time()
        
        # Normalize component signals
        normalized_signals = self._normalize_component_signals(component_outputs)
        
        # Extract feature groups
        agreement_features = self._extract_agreement_features(normalized_signals)
        dte_features = self._extract_dte_features(normalized_signals, dte)
        coherence_features = self._extract_coherence_features(normalized_signals)
        synergy_features = self._extract_synergy_features(normalized_signals)
        
        # Create feature object
        features = IntegrationFeatures(
            # Component Agreement Features
            pairwise_agreement_mean=agreement_features['mean'],
            pairwise_agreement_std=agreement_features['std'],
            multi_component_consensus=agreement_features['consensus'],
            signal_divergence_score=agreement_features['divergence'],
            weighted_agreement_score=agreement_features['weighted'],
            component_alignment_ratio=agreement_features['alignment'],
            agreement_stability=agreement_features['stability'],
            agreement_momentum=agreement_features['momentum'],
            agreement_acceleration=agreement_features['acceleration'],
            agreement_entropy=agreement_features['entropy'],
            agreement_concentration=agreement_features['concentration'],
            agreement_dispersion=agreement_features['dispersion'],
            
            # DTE-Adaptive Features
            dte_specific_pattern=dte_features['specific'],
            dte_weekly_pattern=dte_features['weekly'],
            dte_monthly_pattern=dte_features['monthly'],
            dte_far_pattern=dte_features['far'],
            dte_transition_score=dte_features['transition'],
            dte_evolution_rate=dte_features['evolution'],
            dte_performance_mean=dte_features['perf_mean'],
            dte_performance_std=dte_features['perf_std'],
            dte_reliability_score=dte_features['reliability'],
            dte_consistency_score=dte_features['consistency'],
            dte_adaptation_rate=dte_features['adaptation'],
            dte_efficiency_ratio=dte_features['efficiency'],
            
            # System Coherence Metrics
            signal_stability_score=coherence_features['stability'],
            transition_probability=coherence_features['transition'],
            integration_quality=coherence_features['quality'],
            coherence_index=coherence_features['index'],
            system_entropy=coherence_features['entropy'],
            signal_noise_ratio=coherence_features['snr'],
            confidence_mean=coherence_features['conf_mean'],
            confidence_std=coherence_features['conf_std'],
            health_score=coherence_features['health'],
            robustness_metric=coherence_features['robustness'],
            consistency_score=coherence_features['consistency'],
            reliability_index=coherence_features['reliability'],
            
            # Cross-Component Synergies
            synergy_score_mean=synergy_features['mean'],
            synergy_score_std=synergy_features['std'],
            interaction_effect_strength=synergy_features['interaction'],
            triad_synergy_score=synergy_features['triad'],
            complementary_signal_ratio=synergy_features['complementary'],
            antagonistic_signal_ratio=synergy_features['antagonistic'],
            synergy_concentration=synergy_features['concentration'],
            synergy_dispersion=synergy_features['dispersion'],
            synergy_momentum=synergy_features['momentum'],
            synergy_acceleration=synergy_features['acceleration'],
            synergy_stability=synergy_features['stability'],
            synergy_efficiency=synergy_features['efficiency']
        )
        
        # Performance monitoring
        processing_time = (time.time() - start_time) * 1000
        if processing_time > self.processing_budget_ms:
            logger.warning(f"Processing time {processing_time:.2f}ms exceeded budget {self.processing_budget_ms}ms")
        
        # Validate feature count
        feature_dict = features.to_dict()
        if len(feature_dict) != self.expected_feature_count:
            logger.error(f"Feature count mismatch: expected {self.expected_feature_count}, got {len(feature_dict)}")
        
        return features
    
    def _normalize_component_signals(
        self,
        component_outputs: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Normalize component signals to [-1, 1] range
        
        Args:
            component_outputs: Raw component outputs
            
        Returns:
            Normalized component signals
        """
        normalized = {}
        
        for comp_name, comp_data in component_outputs.items():
            normalized[comp_name] = {}
            
            # Extract and normalize key signals
            for key, value in comp_data.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    # Clip to [-1, 1] range
                    normalized[comp_name][key] = np.clip(value, -1.0, 1.0)
                elif isinstance(value, (int, float)) and np.isnan(value):
                    normalized[comp_name][key] = 0.0
        
        return normalized
    
    def _extract_agreement_features(
        self,
        normalized_signals: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Extract component agreement features (12 features)
        
        Args:
            normalized_signals: Normalized component signals
            
        Returns:
            Dictionary of agreement features
        """
        # Extract primary signals from each component
        primary_signals = []
        for comp_name, signals in normalized_signals.items():
            # Get the first available signal as primary
            if signals:
                primary_signals.append(list(signals.values())[0])
        
        if not primary_signals:
            return self._get_default_agreement_features()
        
        primary_signals = np.array(primary_signals)
        
        # Calculate pairwise agreements
        n_components = len(primary_signals)
        agreements = []
        
        for i in range(n_components):
            for j in range(i + 1, n_components):
                # Agreement as 1 - normalized distance
                agreement = 1.0 - abs(primary_signals[i] - primary_signals[j]) / 2.0
                agreements.append(agreement)
        
        agreements = np.array(agreements) if agreements else np.array([0.5])
        
        # Calculate features
        return {
            'mean': float(np.mean(agreements)),
            'std': float(np.std(agreements)),
            'consensus': float(np.mean(primary_signals > 0) if len(primary_signals) > 0 else 0.5),
            'divergence': float(np.std(primary_signals) if len(primary_signals) > 0 else 0.0),
            'weighted': float(np.average(primary_signals, weights=list(self.component_weights.values())[:len(primary_signals)])),
            'alignment': float(np.sum(np.sign(primary_signals) == np.sign(np.mean(primary_signals))) / len(primary_signals) if len(primary_signals) > 0 else 0.5),
            'stability': float(1.0 - np.std(agreements) if len(agreements) > 0 else 0.5),
            'momentum': float(np.mean(np.diff(agreements)) if len(agreements) > 1 else 0.0),
            'acceleration': float(np.mean(np.diff(np.diff(agreements))) if len(agreements) > 2 else 0.0),
            'entropy': float(-np.sum(agreements * np.log(agreements + 1e-10)) / len(agreements) if len(agreements) > 0 else 0.0),
            'concentration': float(np.max(agreements) if len(agreements) > 0 else 0.5),
            'dispersion': float(np.max(agreements) - np.min(agreements) if len(agreements) > 0 else 0.0)
        }
    
    def _extract_dte_features(
        self,
        normalized_signals: Dict[str, Dict[str, float]],
        dte: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Extract DTE-adaptive features (12 features)
        
        Args:
            normalized_signals: Normalized component signals
            dte: Days to expiry
            
        Returns:
            Dictionary of DTE features
        """
        if dte is None:
            dte = 30  # Default to monthly expiry
        
        # DTE-specific pattern
        dte_factor = np.exp(-dte / 30.0)  # Exponential decay factor
        
        # DTE range patterns
        weekly_pattern = 1.0 if 0 <= dte <= 7 else 0.0
        monthly_pattern = 1.0 if 8 <= dte <= 30 else 0.0
        far_pattern = 1.0 if dte > 30 else 0.0
        
        # Calculate DTE-based features
        signal_values = []
        for signals in normalized_signals.values():
            signal_values.extend(signals.values())
        
        signal_array = np.array(signal_values) if signal_values else np.array([0.0])
        
        return {
            'specific': float(dte_factor),
            'weekly': float(weekly_pattern),
            'monthly': float(monthly_pattern),
            'far': float(far_pattern),
            'transition': float(1.0 / (1.0 + np.exp(-0.1 * (dte - 15)))),  # Sigmoid transition
            'evolution': float(np.abs(dte - 30) / 30.0),  # Distance from monthly
            'perf_mean': float(np.mean(signal_array) * dte_factor),
            'perf_std': float(np.std(signal_array) * dte_factor),
            'reliability': float(1.0 - np.std(signal_array) * dte_factor),
            'consistency': float(1.0 - np.var(signal_array)),
            'adaptation': float(np.tanh(dte / 30.0)),
            'efficiency': float(np.mean(np.abs(signal_array)) / (np.std(signal_array) + 1e-10))
        }
    
    def _extract_coherence_features(
        self,
        normalized_signals: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Extract system coherence features (12 features)
        
        Args:
            normalized_signals: Normalized component signals
            
        Returns:
            Dictionary of coherence features
        """
        # Collect all signal values
        all_signals = []
        confidence_scores = []
        
        for comp_name, signals in normalized_signals.items():
            signal_values = list(signals.values())
            all_signals.extend(signal_values)
            
            # Assume confidence as absolute signal strength
            if signal_values:
                confidence_scores.append(np.mean(np.abs(signal_values)))
        
        all_signals = np.array(all_signals) if all_signals else np.array([0.0])
        confidence_scores = np.array(confidence_scores) if confidence_scores else np.array([0.5])
        
        # Calculate coherence metrics
        signal_mean = np.mean(all_signals)
        signal_std = np.std(all_signals)
        
        return {
            'stability': float(1.0 / (1.0 + signal_std)),
            'transition': float(np.mean(np.abs(np.diff(all_signals))) if len(all_signals) > 1 else 0.0),
            'quality': float(np.mean(confidence_scores)),
            'index': float(1.0 - signal_std),
            'entropy': float(-np.sum(np.abs(all_signals) * np.log(np.abs(all_signals) + 1e-10)) / len(all_signals)),
            'snr': float(np.abs(signal_mean) / (signal_std + 1e-10)),
            'conf_mean': float(np.mean(confidence_scores)),
            'conf_std': float(np.std(confidence_scores)),
            'health': float(np.mean(confidence_scores) * (1.0 - signal_std)),
            'robustness': float(1.0 - np.var(all_signals)),
            'consistency': float(1.0 - np.std(confidence_scores)),
            'reliability': float(np.min(confidence_scores) if len(confidence_scores) > 0 else 0.5)
        }
    
    def _extract_synergy_features(
        self,
        normalized_signals: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Extract cross-component synergy features (12 features)
        
        Args:
            normalized_signals: Normalized component signals
            
        Returns:
            Dictionary of synergy features
        """
        # Calculate synergies between components
        synergy_scores = []
        component_names = list(normalized_signals.keys())
        
        # Pairwise synergies
        for i, comp1 in enumerate(component_names):
            for j, comp2 in enumerate(component_names):
                if i < j:
                    signals1 = list(normalized_signals[comp1].values())
                    signals2 = list(normalized_signals[comp2].values())
                    
                    if signals1 and signals2:
                        # Synergy as product of aligned signals
                        synergy = np.mean([s1 * s2 for s1 in signals1[:1] for s2 in signals2[:1]])
                        synergy_scores.append(synergy)
        
        # Triad synergies
        triad_scores = []
        for i in range(len(component_names)):
            for j in range(i + 1, len(component_names)):
                for k in range(j + 1, len(component_names)):
                    signals = []
                    for comp in [component_names[i], component_names[j], component_names[k]]:
                        comp_signals = list(normalized_signals[comp].values())
                        if comp_signals:
                            signals.append(comp_signals[0])
                    
                    if len(signals) == 3:
                        triad_synergy = np.prod(signals)
                        triad_scores.append(triad_synergy)
        
        synergy_array = np.array(synergy_scores) if synergy_scores else np.array([0.0])
        triad_array = np.array(triad_scores) if triad_scores else np.array([0.0])
        
        # Calculate complementary and antagonistic ratios
        complementary = np.sum(synergy_array > 0) / (len(synergy_array) + 1e-10)
        antagonistic = np.sum(synergy_array < 0) / (len(synergy_array) + 1e-10)
        
        return {
            'mean': float(np.mean(synergy_array)),
            'std': float(np.std(synergy_array)),
            'interaction': float(np.max(np.abs(synergy_array)) if len(synergy_array) > 0 else 0.0),
            'triad': float(np.mean(triad_array)),
            'complementary': float(complementary),
            'antagonistic': float(antagonistic),
            'concentration': float(np.max(np.abs(synergy_array)) if len(synergy_array) > 0 else 0.0),
            'dispersion': float(np.ptp(synergy_array) if len(synergy_array) > 0 else 0.0),
            'momentum': float(np.mean(np.diff(synergy_array)) if len(synergy_array) > 1 else 0.0),
            'acceleration': float(np.mean(np.diff(np.diff(synergy_array))) if len(synergy_array) > 2 else 0.0),
            'stability': float(1.0 - np.std(synergy_array)),
            'efficiency': float(np.mean(np.abs(synergy_array)) / (np.std(synergy_array) + 1e-10))
        }
    
    def _get_default_agreement_features(self) -> Dict[str, float]:
        """Get default agreement features when no signals available"""
        return {
            'mean': 0.5, 'std': 0.0, 'consensus': 0.5, 'divergence': 0.0,
            'weighted': 0.0, 'alignment': 0.5, 'stability': 0.5, 'momentum': 0.0,
            'acceleration': 0.0, 'entropy': 0.0, 'concentration': 0.5, 'dispersion': 0.0
        }
    
    def validate_features(self, features: IntegrationFeatures) -> bool:
        """
        Validate that all features are properly generated
        
        Args:
            features: IntegrationFeatures object to validate
            
        Returns:
            True if valid, False otherwise
        """
        feature_dict = features.to_dict()
        
        # Check feature count
        if len(feature_dict) != self.expected_feature_count:
            logger.error(f"Invalid feature count: {len(feature_dict)} != {self.expected_feature_count}")
            return False
        
        # Check for NaN or infinite values
        for name, value in feature_dict.items():
            if np.isnan(value) or np.isinf(value):
                logger.error(f"Invalid feature value for {name}: {value}")
                return False
        
        return True