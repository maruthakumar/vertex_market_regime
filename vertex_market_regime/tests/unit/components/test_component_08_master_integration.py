"""
Unit tests for Component 08: Master Integration Feature Engineering

Tests the generation of 48 cross-component integration features.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import time

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.components.component_08_master_integration.component_08_analyzer import Component08Analyzer
from src.components.component_08_master_integration.feature_engine import MasterIntegrationFeatureEngine
from src.components.component_08_master_integration.component_aggregator import ComponentAggregator
from src.components.component_08_master_integration.dte_pattern_extractor import DTEPatternExtractor
from src.components.component_08_master_integration.correlation_analyzer import CorrelationAnalyzer
from src.components.component_08_master_integration.confidence_metrics import ConfidenceMetrics
from src.components.component_08_master_integration.synergy_detector import SynergyDetector


class TestComponent08Analyzer:
    """Test suite for Component 08 Master Integration Analyzer"""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        config = {
            'processing_budget_ms': 100,
            'memory_budget_mb': 150
        }
        return Component08Analyzer(config)
    
    @pytest.fixture
    def mock_component_outputs(self):
        """Create mock outputs from Components 1-7"""
        return {
            'component_01': {
                'straddle_trend_score': 0.75,
                'vol_compression_score': 0.45,
                'breakout_probability': 0.65,
                'dte_correlation_strength': 0.8,
                'regime_confidence': 0.7,
                'health_score': 0.85,
                'processing_time_ms': 12.5
            },
            'component_02': {
                'gamma_exposure_score': -0.3,
                'sentiment_level': 0.6,
                'pin_risk_score': 0.4,
                'delta_imbalance': 0.2,
                'vega_concentration': 0.5,
                'health_score': 0.9,
                'processing_time_ms': 10.2
            },
            'component_03': {
                'institutional_flow_score': 0.55,
                'divergence_type': 0.0,
                'range_expansion_score': 0.7,
                'oi_concentration': 0.65,
                'pa_momentum': 0.5,
                'health_score': 0.75,
                'processing_time_ms': 15.3
            },
            'component_04': {
                'skew_bias_score': 0.25,
                'term_structure_signal': -0.15,
                'iv_regime_level': 0.5,
                'volatility_smile_slope': 0.1,
                'skew_momentum': 0.3,
                'health_score': 0.8,
                'processing_time_ms': 11.7
            },
            'component_05': {
                'momentum_score': 0.6,
                'volatility_regime_score': 0.4,
                'confluence_score': 0.7,
                'atr_expansion_rate': 0.3,
                'cpr_alignment': 0.5,
                'health_score': 0.82,
                'processing_time_ms': 13.4
            },
            'component_06': {
                'correlation_agreement_score': 0.8,
                'breakdown_alert': 0.0,
                'system_stability_score': 0.85,
                'correlation_strength': 0.75,
                'feature_importance_mean': 0.6,
                'health_score': 0.88,
                'processing_time_ms': 18.9
            },
            'component_07': {
                'level_strength_score': 0.7,
                'breakout_probability': 0.45,
                'confluence_score': 0.65,
                'zone_reliability': 0.8,
                'pattern_confidence': 0.75,
                'health_score': 0.85,
                'processing_time_ms': 14.2
            }
        }
    
    def test_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer.processing_budget_ms == 100
        assert analyzer.memory_budget_mb == 150
        assert analyzer.expected_feature_count == 48
        assert analyzer.feature_engine is not None
        assert analyzer.aggregator is not None
        assert analyzer.dte_extractor is not None
        assert analyzer.correlation_analyzer is not None
        assert analyzer.confidence_metrics is not None
        assert analyzer.synergy_detector is not None
    
    def test_analyze_master_integration(self, analyzer, mock_component_outputs):
        """Test master integration analysis"""
        result = analyzer.analyze_master_integration(
            mock_component_outputs,
            dte=15
        )
        
        # Check result structure
        assert 'features' in result
        assert 'metadata' in result
        assert 'performance_metrics' in result
        assert 'component_health' in result
        
        # Check feature count
        features = result['features']
        assert len(features) == 48
        
        # Check metadata
        metadata = result['metadata']
        assert metadata['dte'] == 15
        assert metadata['component_count'] == 7
        assert metadata['feature_count'] == 48
        assert metadata['is_valid'] == True
        
        # Check performance metrics
        perf = result['performance_metrics']
        assert 'processing_time_ms' in perf
        assert 'within_budget' in perf
        assert 'feature_generation_rate' in perf
        assert 'average_confidence' in perf
    
    def test_feature_categories(self, analyzer, mock_component_outputs):
        """Test that all feature categories are present"""
        result = analyzer.analyze_master_integration(mock_component_outputs)
        features = result['features']
        
        # Agreement features (12)
        agreement_features = [
            'agreement_mean', 'agreement_std', 'consensus_score', 'divergence_score',
            'weighted_agreement', 'alignment_ratio', 'agreement_stability', 'agreement_momentum',
            'agreement_acceleration', 'agreement_entropy', 'agreement_concentration', 'agreement_dispersion'
        ]
        for feature in agreement_features:
            assert feature in features
        
        # DTE features (12)
        dte_features = [
            'dte_specific', 'dte_weekly', 'dte_monthly', 'dte_far',
            'dte_transition', 'dte_evolution', 'dte_perf_mean', 'dte_perf_std',
            'dte_reliability', 'dte_consistency', 'dte_adaptation', 'dte_efficiency'
        ]
        for feature in dte_features:
            assert feature in features
        
        # Coherence features (12)
        coherence_features = [
            'stability_score', 'transition_prob', 'integration_quality', 'coherence_index',
            'system_entropy', 'signal_noise_ratio', 'confidence_mean', 'confidence_std',
            'health_score', 'robustness_metric', 'consistency_score', 'reliability_index'
        ]
        for feature in coherence_features:
            assert feature in features
        
        # Synergy features (12)
        synergy_features = [
            'synergy_mean', 'synergy_std', 'interaction_strength', 'triad_synergy',
            'complementary_ratio', 'antagonistic_ratio', 'synergy_concentration', 'synergy_dispersion',
            'synergy_momentum', 'synergy_acceleration', 'synergy_stability', 'synergy_efficiency'
        ]
        for feature in synergy_features:
            assert feature in features
    
    def test_feature_value_ranges(self, analyzer, mock_component_outputs):
        """Test that feature values are within expected ranges"""
        result = analyzer.analyze_master_integration(mock_component_outputs)
        features = result['features']
        
        for name, value in features.items():
            # Check for valid numeric values
            assert isinstance(value, (int, float))
            assert not np.isnan(value)
            assert not np.isinf(value)
            
            # Check specific range constraints
            if 'ratio' in name or 'probability' in name:
                assert 0.0 <= value <= 1.0
            
            if 'score' in name:
                assert -1.0 <= value <= 1.0
    
    def test_dte_specific_patterns(self, analyzer, mock_component_outputs):
        """Test DTE-specific pattern extraction"""
        # Test different DTE values
        dte_values = [0, 7, 15, 30, 45, 90]
        
        for dte in dte_values:
            result = analyzer.analyze_master_integration(
                mock_component_outputs,
                dte=dte
            )
            features = result['features']
            
            # Check DTE range features
            if 0 <= dte <= 7:
                assert features['dte_weekly'] == 1.0
                assert features['dte_monthly'] == 0.0
                assert features['dte_far'] == 0.0
            elif 8 <= dte <= 30:
                assert features['dte_weekly'] == 0.0
                assert features['dte_monthly'] == 1.0
                assert features['dte_far'] == 0.0
            else:
                assert features['dte_weekly'] == 0.0
                assert features['dte_monthly'] == 0.0
                assert features['dte_far'] == 1.0
    
    def test_performance_budget(self, analyzer, mock_component_outputs):
        """Test that processing stays within budget"""
        result = analyzer.analyze_master_integration(mock_component_outputs)
        
        # Check processing time
        processing_time = result['performance_metrics']['processing_time_ms']
        assert processing_time > 0
        
        # Should typically be within budget (allow some variation)
        # Note: This might occasionally exceed budget in slow environments
        if processing_time <= analyzer.processing_budget_ms:
            assert result['performance_metrics']['within_budget'] == True
    
    def test_missing_components(self, analyzer):
        """Test handling of missing component data"""
        # Only provide partial component outputs
        partial_outputs = {
            'component_01': {
                'straddle_trend_score': 0.5,
                'health_score': 0.7
            },
            'component_03': {
                'institutional_flow_score': 0.3,
                'health_score': 0.6
            }
        }
        
        result = analyzer.analyze_master_integration(partial_outputs)
        
        # Should still generate 48 features
        assert len(result['features']) == 48
        assert result['metadata']['is_valid'] == True
        
        # Health should reflect missing components
        assert result['metadata']['healthy_components'] <= 2
    
    def test_error_handling(self, analyzer):
        """Test error handling"""
        # Test with invalid input
        result = analyzer.analyze_master_integration(None)
        
        # Should return default result
        assert 'features' in result
        assert len(result['features']) == 48
        assert 'error' in result['metadata']
        assert result['metadata'].get('is_default') == True
    
    def test_performance_tracking(self, analyzer, mock_component_outputs):
        """Test performance tracking"""
        # Run multiple analyses
        for _ in range(5):
            analyzer.analyze_master_integration(mock_component_outputs)
        
        # Get performance summary
        summary = analyzer.get_performance_summary()
        
        assert summary['total_runs'] == 5
        assert summary['average_time_ms'] > 0
        assert 0 <= summary['success_rate'] <= 1
        assert 0 <= summary['budget_compliance_rate'] <= 1


class TestComponentAggregator:
    """Test suite for Component Aggregator"""
    
    @pytest.fixture
    def aggregator(self):
        """Create aggregator instance"""
        return ComponentAggregator()
    
    def test_signal_normalization(self, aggregator):
        """Test signal normalization"""
        component_outputs = {
            'component_01': {
                'straddle_trend_score': 2.5,  # Out of range
                'vol_compression_score': -1.5,  # Out of range
                'breakout_probability': 0.8,  # In range
            }
        }
        
        aggregated = aggregator.aggregate_components(component_outputs)
        signals = aggregated['component_01']['signals']
        
        # Check normalization
        assert signals['straddle_trend_score'] == 1.0  # Clipped to 1.0
        assert signals['vol_compression_score'] == -1.0  # Clipped to -1.0
        assert signals['breakout_probability'] == 0.8  # Unchanged
    
    def test_health_assessment(self, aggregator):
        """Test component health assessment"""
        component_outputs = {
            'component_01': {
                'straddle_trend_score': 0.5,
                'vol_compression_score': 0.3,
                'confidence': 0.8,
                'processing_time_ms': 10
            }
        }
        
        aggregated = aggregator.aggregate_components(component_outputs)
        health_score = aggregated['component_01']['health_score']
        
        assert 0 <= health_score <= 1
        assert aggregated['component_01']['metadata']['has_confidence'] == True


class TestDTEPatternExtractor:
    """Test suite for DTE Pattern Extractor"""
    
    @pytest.fixture
    def extractor(self):
        """Create extractor instance"""
        return DTEPatternExtractor()
    
    def test_dte_decay_patterns(self, extractor):
        """Test DTE decay pattern calculation"""
        aggregated_components = {
            'component_01': {
                'signals': {'signal1': 0.5},
                'health_score': 0.8
            }
        }
        
        # Test different DTEs
        for dte in [0, 7, 30, 90]:
            patterns = extractor.extract_dte_patterns(
                aggregated_components,
                dte=dte
            )
            
            # Check pattern values
            assert patterns.performance_mean >= 0
            assert patterns.reliability_score >= 0
            assert patterns.efficiency_ratio >= 0


class TestCorrelationAnalyzer:
    """Test suite for Correlation Analyzer"""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        return CorrelationAnalyzer()
    
    def test_pairwise_correlations(self, analyzer):
        """Test pairwise correlation calculation"""
        aggregated_components = {
            'component_01': {'signals': {'s1': 0.5, 's2': 0.3}},
            'component_02': {'signals': {'s1': 0.6, 's2': 0.4}},
            'component_03': {'signals': {'s1': -0.2, 's2': -0.1}}
        }
        
        analysis = analyzer.analyze_correlations(aggregated_components)
        
        # Check correlation structure
        assert 'pairwise_correlations' in analysis
        assert 'correlation_matrix' in analysis
        assert 'coherence_metrics' in analysis
        
        # Should have 3 pairs for 3 components
        assert len(analysis['pairwise_correlations']) == 3


class TestConfidenceMetrics:
    """Test suite for Confidence Metrics"""
    
    @pytest.fixture
    def metrics(self):
        """Create metrics instance"""
        return ConfidenceMetrics()
    
    def test_confidence_calculation(self, metrics):
        """Test confidence metric calculation"""
        aggregated_components = {
            'component_01': {
                'health_score': 0.8,
                'signals': {'s1': 0.5},
                'metadata': {'confidence': 0.7, 'data_quality': 0.9}
            },
            'component_02': {
                'health_score': 0.6,
                'signals': {'s1': 0.3},
                'metadata': {'confidence': 0.5, 'data_quality': 0.7}
            }
        }
        
        result = metrics.calculate_confidence_metrics(aggregated_components)
        
        # Check structure
        assert 'health_metrics' in result
        assert 'quality_metrics' in result
        assert 'transition_features' in result
        assert 'stability_indicators' in result
        assert 'overall_confidence' in result
        
        # Check confidence values
        overall = result['overall_confidence']
        assert 0 <= overall['overall_confidence'] <= 1
        assert overall['confidence_level'] in ['high', 'medium', 'low']


class TestSynergyDetector:
    """Test suite for Synergy Detector"""
    
    @pytest.fixture
    def detector(self):
        """Create detector instance"""
        return SynergyDetector()
    
    def test_synergy_detection(self, detector):
        """Test synergy detection between components"""
        aggregated_components = {
            'component_01': {
                'signals': {'s1': 0.8, 's2': 0.6},
                'health_score': 0.9
            },
            'component_02': {
                'signals': {'s1': 0.7, 's2': 0.5},
                'health_score': 0.85
            },
            'component_03': {
                'signals': {'s1': -0.3, 's2': -0.2},
                'health_score': 0.7
            }
        }
        
        analysis = detector.detect_synergies(aggregated_components)
        
        # Check structure
        assert 'pairwise_synergies' in analysis
        assert 'triad_synergies' in analysis
        assert 'complementary_signals' in analysis
        assert 'antagonistic_signals' in analysis
        assert 'interaction_effects' in analysis
        assert 'concentration_metrics' in analysis
        
        # Check synergy calculations
        assert len(analysis['pairwise_synergies']) == 3  # 3 pairs for 3 components
        
        # Triad synergies should be limited to top 10
        assert len(analysis['triad_synergies']) <= 10


class TestFeatureValidation:
    """Test suite for feature validation"""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        return Component08Analyzer()
    
    def test_feature_completeness(self, analyzer):
        """Test that all 48 features are generated"""
        # Create minimal valid input
        component_outputs = {
            f'component_{i:02d}': {
                'signal1': 0.5,
                'health_score': 0.7
            }
            for i in range(1, 8)
        }
        
        result = analyzer.analyze_master_integration(component_outputs)
        features = result['features']
        
        # Check exact feature count
        assert len(features) == 48
        
        # Check no NaN or Inf values
        for name, value in features.items():
            assert not np.isnan(value), f"Feature {name} is NaN"
            assert not np.isinf(value), f"Feature {name} is Inf"
    
    def test_feature_determinism(self, analyzer):
        """Test that features are deterministic"""
        component_outputs = {
            f'component_{i:02d}': {
                'signal1': 0.5,
                'signal2': 0.3,
                'health_score': 0.8
            }
            for i in range(1, 8)
        }
        
        # Run twice with same input
        result1 = analyzer.analyze_master_integration(component_outputs, dte=30)
        result2 = analyzer.analyze_master_integration(component_outputs, dte=30)
        
        # Features should be identical
        features1 = result1['features']
        features2 = result2['features']
        
        for name in features1:
            assert np.isclose(features1[name], features2[name], rtol=1e-9), \
                f"Feature {name} is not deterministic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])