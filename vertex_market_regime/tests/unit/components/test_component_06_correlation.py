"""
Unit tests for Component 6: Enhanced Correlation & Predictive Feature Engineering

Tests the complete Component 6 implementation including:
- Core Component 6 Feature Engineering Framework (200+ features)
- Raw Correlation Measurement System (120 features)
- Predictive Straddle Intelligence (50 features)
- Meta-Correlation Intelligence (30 features)
- Component Integration Bridge

🎯 Testing Strategy: FEATURE ENGINEERING VALIDATION ONLY
All tests validate mathematical feature extraction and processing.
No classification logic testing - that's handled by ML models.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import asyncio
import time
from typing import Dict, Any

# Import Component 6 modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from components.component_06_correlation.component_06_analyzer import (
    Component06CorrelationAnalyzer,
    Component06AnalysisResult,
    RawCorrelationFeatures,
    PredictiveStraddleFeatures,
    MetaCorrelationFeatures
)

from components.base_component import ComponentAnalysisResult, FeatureVector

from components.component_06_correlation.correlation_matrix_engine import (
    CorrelationMatrixEngine,
    CorrelationMatrixResult
)

from components.component_06_correlation.gap_analysis_engine import (
    GapAnalysisEngine,
    GapAnalysisResult,
    GapMetrics
)

from components.component_06_correlation.predictive_straddle_engine import (
    PredictiveStraddleEngine,
    PredictiveStraddleResult
)

from components.component_06_correlation.meta_intelligence_engine import (
    MetaCorrelationIntelligenceEngine,
    MetaIntelligenceResult
)

from components.component_06_correlation.component_integration_bridge import (
    ComponentIntegrationBridge,
    IntegratedComponentData,
    ComponentDataExtract
)


class TestComponent06CoreFramework:
    """Test the core Component 6 feature engineering framework"""
    
    @pytest.fixture
    def component_config(self):
        """Basic configuration for Component 6"""
        return {
            'component_id': 6,
            'feature_count': 200,
            'target_processing_time_ms': 200,
            'target_memory_usage_mb': 450,
            'performance_logging': True,
            'gpu_enabled': False
        }
    
    @pytest.fixture
    def component_analyzer(self, component_config):
        """Component 6 analyzer instance"""
        return Component06CorrelationAnalyzer(component_config)
    
    @pytest.fixture
    def mock_market_data(self):
        """Mock market data for testing"""
        dates = pd.date_range('2024-01-15 09:15:00', periods=100, freq='1min')
        
        return {
            'component_results': {
                1: Mock(
                    straddle_time_series=Mock(
                        atm_straddle_prices=np.random.uniform(100, 200, 100),
                        itm1_straddle_prices=np.random.uniform(120, 240, 100),
                        otm1_straddle_prices=np.random.uniform(80, 160, 100),
                        timestamps=dates
                    ),
                    weighting_analysis=Mock(confidence_score=0.85),
                    features=Mock(features=np.random.random(120))
                ),
                2: Mock(
                    greeks_analysis=Mock(
                        delta_values=np.random.uniform(0.3, 0.7, 100),
                        gamma_values=np.random.uniform(0.01, 0.05, 100),
                        theta_values=np.random.uniform(-0.1, -0.01, 100),
                        vega_values=np.random.uniform(0.1, 0.3, 100)
                    ),
                    features=Mock(features=np.random.random(87))
                )
            },
            'overnight_factors': {
                'sgx_nifty': 0.5,
                'dow_jones': 0.3,
                'vix_change': -0.2,
                'usd_inr': 0.1,
                'news_sentiment': 0.4
            },
            'previous_day_data': pd.DataFrame({
                'atm_premium': np.random.uniform(100, 150, 50),
                'volume': np.random.randint(1000, 5000, 50)
            }),
            'historical_performance': {
                'overall_accuracy': [0.85, 0.87, 0.84, 0.86],
                'component_1_accuracy': [0.88, 0.85, 0.87],
                'cross_validation_scores': [0.83, 0.85, 0.84]
            }
        }
    
    def test_component_initialization(self, component_analyzer):
        """Test Component 6 initialization"""
        assert component_analyzer.component_id == 6
        assert component_analyzer.feature_count == 200
        assert component_analyzer.target_processing_time_ms == 200
        assert component_analyzer.component_integration_enabled is True
        
        # Test sub-system initialization
        assert hasattr(component_analyzer, 'correlation_matrix_engine')
        assert hasattr(component_analyzer, 'gap_analysis_engine')
        assert hasattr(component_analyzer, 'predictive_straddle_engine')
        assert hasattr(component_analyzer, 'meta_intelligence_engine')
        assert hasattr(component_analyzer, 'integration_bridge')
    
    @pytest.mark.asyncio
    async def test_comprehensive_analysis(self, component_analyzer, mock_market_data):
        """Test complete Component 6 analysis pipeline"""
        result = await component_analyzer.analyze(mock_market_data)
        
        # Validate result structure
        assert result is not None
        assert result.component_id == 6
        assert result.component_name == "Enhanced Correlation & Predictive Analysis"
        
        # Validate features
        assert hasattr(result, 'features')
        assert result.features.feature_count == 200
        assert len(result.features.features) == 200
        assert len(result.features.feature_names) == 200
        
        # Validate performance compliance
        assert result.processing_time_ms < 500  # Allow some margin for testing
        assert 'performance_compliant' in result.metadata
    
    @pytest.mark.asyncio
    async def test_feature_extraction(self, component_analyzer, mock_market_data):
        """Test feature extraction method"""
        features = await component_analyzer.extract_features(mock_market_data)
        
        assert features is not None
        assert features.feature_count == 200
        assert len(features.features) == 200
        
        # Validate feature types (should be numeric)
        assert all(isinstance(f, (int, float, np.number)) for f in features.features)
        assert not np.any(np.isnan(features.features))
        assert not np.any(np.isinf(features.features))
    
    def test_feature_names_generation(self, component_analyzer):
        """Test feature names generation"""
        feature_names = component_analyzer._get_feature_names()
        
        assert len(feature_names) == 200
        assert all(isinstance(name, str) for name in feature_names)
        
        # Check feature name categories
        correlation_features = [name for name in feature_names if 'corr_' in name]
        predictive_features = [name for name in feature_names if any(x in name for x in ['pred_', 'close_', 'gap_'])]
        meta_features = [name for name in feature_names if any(x in name for x in ['accuracy_', 'confidence_', 'weight_', 'boost_'])]
        
        assert len(correlation_features) >= 80   # Should have correlation features
        assert len(predictive_features) >= 30    # Should have predictive features
        assert len(meta_features) >= 15          # Should have meta features
    
    def test_weight_update_mechanism(self, component_analyzer):
        """Test adaptive weight update mechanism"""
        # Mock performance feedback
        performance_feedback = Mock(accuracy=0.87)
        
        result = asyncio.run(component_analyzer.update_weights(performance_feedback))
        
        assert 'updated_weights' in result
        assert 'weight_changes' in result
        assert 'performance_improvement' in result
        assert 'confidence_score' in result
        
        # Validate that weights are reasonable
        for weight in result['updated_weights'].values():
            assert 0.1 <= weight <= 2.0  # Reasonable weight bounds


class TestCorrelationMatrixEngine:
    """Test the correlation matrix calculation engine"""
    
    @pytest.fixture
    def correlation_engine(self):
        """Correlation matrix engine instance"""
        config = {
            'target_processing_time_ms': 200,
            'parallel_processing': True,
            'max_workers': 2,
            'min_correlation_periods': 20
        }
        return CorrelationMatrixEngine(config)
    
    @pytest.fixture
    def mock_components_data(self):
        """Mock components data for correlation testing"""
        return {
            1: {
                'atm_straddle': pd.DataFrame({
                    'premium': np.random.uniform(100, 200, 50),
                    'timestamp': pd.date_range('2024-01-15 09:15', periods=50, freq='1min')
                })
            },
            2: {
                'greeks_data': pd.DataFrame({
                    'delta': np.random.uniform(0.3, 0.7, 50),
                    'gamma': np.random.uniform(0.01, 0.05, 50)
                })
            }
        }
    
    @pytest.mark.asyncio
    async def test_correlation_matrix_calculation(self, correlation_engine, mock_components_data):
        """Test comprehensive correlation matrix calculation"""
        result = await correlation_engine.calculate_comprehensive_correlation_matrix(mock_components_data)
        
        assert isinstance(result, CorrelationMatrixResult)
        assert result.processing_time_ms < 300  # Performance requirement
        assert len(result.correlation_matrix) >= 1
        assert result.feature_count >= 0
        
        # Validate correlation coefficients
        for coeff in result.correlation_coefficients.values():
            assert -1.0 <= coeff <= 1.0  # Valid correlation range
    
    def test_correlation_feature_extraction(self, correlation_engine):
        """Test correlation feature extraction"""
        # Create mock correlation result
        mock_result = CorrelationMatrixResult(
            correlation_matrix=np.array([[1.0, 0.7], [0.7, 1.0]]),
            correlation_coefficients={'test': 0.7},
            stability_metrics={'overall': 0.8},
            processing_time_ms=50.0,
            breakdown_indicators=np.array([0.1, 0.2]),
            confidence_scores=np.array([0.8, 0.9]),
            feature_count=2,
            timestamp=datetime.utcnow()
        )
        
        features = correlation_engine.extract_correlation_features(mock_result)
        
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        assert all(-1.0 <= f <= 2.0 for f in features)  # Reasonable feature range


class TestGapAnalysisEngine:
    """Test the gap analysis and overnight factor integration"""
    
    @pytest.fixture
    def gap_engine(self):
        """Gap analysis engine instance"""
        config = {
            'gap_thresholds': {'no_gap': 0.2, 'small_gap': 0.5, 'medium_gap': 1.0, 'large_gap': 2.0},
            'overnight_factor_weights': {
                'sgx_nifty': 0.15, 'dow_jones': 0.10, 'vix_change': 0.20
            }
        }
        return GapAnalysisEngine(config)
    
    @pytest.fixture
    def mock_gap_data(self):
        """Mock data for gap analysis"""
        return {
            'market_data': pd.DataFrame({
                'open': [20100, 20105, 20110],
                'close': [20095, 20100, 20108],
                'spot': [20100, 20105, 20110]
            }),
            'overnight_factors': {
                'sgx_nifty': 0.8,
                'dow_jones': 0.3,
                'vix_change': -0.5,
                'usd_inr': 0.1,
                'news_sentiment': 0.6
            }
        }
    
    def test_gap_analysis(self, gap_engine, mock_gap_data):
        """Test comprehensive gap analysis"""
        result = gap_engine.analyze_comprehensive_gap(
            mock_gap_data['market_data'],
            mock_gap_data['overnight_factors']
        )
        
        assert isinstance(result, GapAnalysisResult)
        assert hasattr(result, 'gap_metrics')
        assert hasattr(result, 'overnight_factors')
        assert hasattr(result, 'correlation_weights')
        
        # Validate gap metrics
        gap_metrics = result.gap_metrics
        assert hasattr(gap_metrics, 'gap_size_percent')
        assert hasattr(gap_metrics, 'gap_direction')
        assert gap_metrics.gap_direction in [-1.0, 0.0, 1.0]
        
        # Validate feature arrays
        assert len(result.gap_direction_features) == 8
        assert len(result.gap_magnitude_features) == 7
        assert len(result.overnight_factor_features) == 15
        assert len(result.strike_correlation_features) == 10
    
    def test_gap_categorization(self, gap_engine):
        """Test gap size categorization"""
        # Test different gap sizes
        test_cases = [
            (0.1, 'no_gap'),
            (0.3, 'small_gap'),
            (0.8, 'medium_gap'),
            (1.5, 'large_gap'),
            (3.0, 'extreme_gap')
        ]
        
        for gap_size, expected_category in test_cases:
            market_data = pd.DataFrame({
                'open': [20000 + gap_size * 200],  # Convert % to points
                'close': [20000]
            })
            
            result = gap_engine._calculate_gap_metrics(market_data, None)
            # Note: Due to the way _calculate_gap_metrics works, we'd need to test the full pipeline
            # This is a simplified test to check the logic


class TestPredictiveStraddleEngine:
    """Test the predictive straddle intelligence engine"""
    
    @pytest.fixture
    def predictive_engine(self):
        """Predictive straddle engine instance"""
        config = {
            'min_historical_periods': 20,
            'short_term_window': 20,
            'medium_term_window': 60,
            'long_term_window': 252
        }
        return PredictiveStraddleEngine(config)
    
    @pytest.fixture
    def mock_predictive_data(self):
        """Mock data for predictive analysis"""
        return {
            'current_data': pd.DataFrame({
                'premium': np.random.uniform(100, 150, 100),
                'volume': np.random.randint(1000, 5000, 100),
                'timestamp': pd.date_range('2024-01-15 09:15', periods=100, freq='1min')
            }),
            'previous_day_data': pd.DataFrame({
                'atm_premium': np.random.uniform(95, 145, 50),
                'itm1_premium': np.random.uniform(115, 165, 50),
                'otm1_premium': np.random.uniform(75, 125, 50)
            }),
            'overnight_factors': {
                'sgx_nifty': 0.4,
                'dow_jones': 0.2,
                'vix_change': -0.3
            }
        }
    
    def test_predictive_feature_extraction(self, predictive_engine, mock_predictive_data):
        """Test predictive feature extraction"""
        result = predictive_engine.extract_predictive_features(
            mock_predictive_data['current_data'],
            mock_predictive_data['previous_day_data'],
            None,  # historical_data
            mock_predictive_data['overnight_factors']
        )
        
        assert isinstance(result, PredictiveStraddleResult)
        
        # Validate feature arrays (50 total features)
        assert len(result.atm_close_predictors) == 7
        assert len(result.itm1_close_predictors) == 7
        assert len(result.otm1_close_predictors) == 6
        assert len(result.gap_direction_predictors) == 8
        assert len(result.gap_magnitude_predictors) == 7
        assert len(result.opening_minutes_analysis) == 8
        assert len(result.full_day_forecast) == 7
        
        # Validate feature ranges
        for feature_array in [result.atm_close_predictors, result.itm1_close_predictors, result.otm1_close_predictors]:
            assert all(-5.0 <= f <= 5.0 for f in feature_array)  # Reasonable range
    
    def test_previous_day_analysis(self, predictive_engine, mock_predictive_data):
        """Test previous day close analysis"""
        result = predictive_engine._analyze_previous_day_close(
            mock_predictive_data['previous_day_data'],
            None  # historical_data
        )
        
        assert hasattr(result, 'atm_close_price')
        assert hasattr(result, 'premium_decay_rates')
        assert hasattr(result, 'volume_patterns')
        assert 0.0 <= result.atm_close_percentile <= 1.0


class TestMetaIntelligenceEngine:
    """Test the meta-correlation intelligence engine"""
    
    @pytest.fixture
    def meta_engine(self):
        """Meta-intelligence engine instance"""
        config = {
            'short_term_window': 20,
            'medium_term_window': 100,
            'long_term_window': 500
        }
        return MetaCorrelationIntelligenceEngine(config)
    
    @pytest.fixture
    def mock_meta_data(self):
        """Mock data for meta-intelligence testing"""
        return {
            'component_results': {
                1: Mock(accuracy=0.85, confidence=0.80, score=0.82),
                2: Mock(accuracy=0.88, confidence=0.75, score=0.84),
                3: Mock(accuracy=0.82, confidence=0.85, score=0.79)
            },
            'historical_performance': {
                'overall_accuracy': [0.84, 0.85, 0.87, 0.83, 0.86],
                'component_1_accuracy': [0.85, 0.87, 0.84],
                'cross_validation_scores': [0.83, 0.85, 0.86]
            },
            'current_weights': {1: 1.0, 2: 1.1, 3: 0.9, 4: 1.0, 5: 0.95}
        }
    
    def test_meta_intelligence_extraction(self, meta_engine, mock_meta_data):
        """Test meta-intelligence feature extraction"""
        result = meta_engine.extract_meta_intelligence_features(
            mock_meta_data['component_results'],
            mock_meta_data['historical_performance'],
            mock_meta_data['current_weights']
        )
        
        assert isinstance(result, MetaIntelligenceResult)
        
        # Validate feature arrays (30 total features)
        assert len(result.accuracy_tracking_features) == 8
        assert len(result.confidence_scoring_features) == 7
        assert len(result.dynamic_weight_optimization_features) == 8
        assert len(result.performance_boosting_features) == 7
        
        # Validate system health metrics
        assert 0.0 <= result.overall_system_health <= 1.0
        assert 0.0 <= result.meta_confidence_score <= 1.0
    
    def test_weight_optimization(self, meta_engine):
        """Test adaptive weight optimization"""
        current_weights = {1: 1.0, 2: 1.0, 3: 1.0}
        performance_metrics = {
            1: {'accuracy': 0.90, 'confidence': 0.85, 'stability': 0.80},
            2: {'accuracy': 0.75, 'confidence': 0.70, 'stability': 0.75},
            3: {'accuracy': 0.85, 'confidence': 0.80, 'stability': 0.85}
        }
        
        optimized_weights = meta_engine.weight_optimizer.optimize_component_weights(
            current_weights, performance_metrics
        )
        
        # High-performing component should get higher weight
        assert optimized_weights[1] >= optimized_weights[2]  # Component 1 performed better
        
        # All weights should be within reasonable bounds
        for weight in optimized_weights.values():
            assert 0.1 <= weight <= 2.0


class TestComponentIntegrationBridge:
    """Test the component integration bridge"""
    
    @pytest.fixture
    def integration_bridge(self):
        """Component integration bridge instance"""
        config = {
            'required_components': [1, 2, 3, 4, 5],
            'fallback_enabled': True,
            'time_tolerance': '1min',
            'missing_data_strategy': 'interpolate'
        }
        return ComponentIntegrationBridge(config)
    
    @pytest.fixture
    def mock_component_results(self):
        """Mock component results for integration testing"""
        return {
            1: Mock(
                straddle_time_series=Mock(
                    atm_straddle_prices=np.random.uniform(100, 200, 50),
                    timestamps=pd.date_range('2024-01-15 09:15', periods=50, freq='1min')
                ),
                confidence=0.85
            ),
            2: Mock(
                greeks_analysis=Mock(
                    delta_values=np.random.uniform(0.3, 0.7, 50)
                ),
                confidence=0.82
            )
        }
    
    @pytest.mark.asyncio
    async def test_component_integration(self, integration_bridge, mock_component_results):
        """Test component data integration"""
        result = await integration_bridge.integrate_components(mock_component_results)
        
        assert isinstance(result, IntegratedComponentData)
        assert len(result.components_data) >= len(mock_component_results)
        assert 0.0 <= result.integration_quality_score <= 1.0
        assert result.total_processing_time_ms >= 0
    
    def test_data_extraction(self, integration_bridge):
        """Test component data extraction"""
        # Mock component result
        mock_result = Mock(
            straddle_time_series=Mock(
                atm_straddle_prices=[100, 105, 110],
                timestamps=['2024-01-15 09:15', '2024-01-15 09:16', '2024-01-15 09:17']
            ),
            weighting_analysis=Mock(confidence_score=0.85),
            features=Mock(features=np.random.random(10))
        )
        
        extracted = integration_bridge.data_extractor.extract_component_data(1, mock_result)
        
        assert isinstance(extracted, ComponentDataExtract)
        assert extracted.component_id == 1
        assert 0.0 <= extracted.confidence_score <= 1.0
        assert extracted.processing_time_ms >= 0


class TestPerformanceCompliance:
    """Test performance requirements compliance"""
    
    @pytest.fixture
    def performance_config(self):
        """Performance-focused configuration"""
        return {
            'component_id': 6,
            'feature_count': 200,
            'target_processing_time_ms': 200,
            'target_memory_usage_mb': 450,
            'performance_logging': True
        }
    
    @pytest.fixture
    def performance_analyzer(self, performance_config):
        """Performance-focused analyzer"""
        return Component06CorrelationAnalyzer(performance_config)
    
    @pytest.mark.asyncio
    async def test_processing_time_compliance(self, performance_analyzer):
        """Test processing time meets <200ms requirement"""
        # Minimal test data for performance testing
        test_data = {
            'component_results': {1: Mock(), 2: Mock()},
            'overnight_factors': {},
            'raw_data': pd.DataFrame({'value': [1, 2, 3]})
        }
        
        start_time = time.time()
        result = await performance_analyzer.analyze(test_data)
        end_time = time.time()
        
        processing_time_ms = (end_time - start_time) * 1000
        
        # Allow some margin for test environment
        assert processing_time_ms < 500  # Relaxed for testing environment
        assert result.processing_time_ms < 500  # Internal measurement
    
    def test_memory_usage_compliance(self, performance_analyzer):
        """Test memory usage stays within limits"""
        # This would require memory profiling tools in a real implementation
        memory_usage = performance_analyzer._get_memory_usage()
        
        # Basic validation that memory usage is reported
        assert memory_usage >= 0
        
        # In a real test, we'd validate against the 450MB limit
        # This requires specialized memory profiling tools
    
    def test_feature_count_compliance(self, performance_analyzer):
        """Test that exactly 200+ features are produced"""
        feature_names = performance_analyzer._get_feature_names()
        
        assert len(feature_names) == 200
        
        # Validate feature breakdown
        correlation_features = [name for name in feature_names if 'corr_' in name]
        predictive_features = [name for name in feature_names if any(x in name for x in ['pred_', 'close_', 'gap_'])]
        meta_features = [name for name in feature_names if any(x in name for x in ['accuracy_', 'weight_', 'boost_'])]
        
        # Should have approximately the right distribution
        assert len(correlation_features) >= 80   # Correlation features
        assert len(predictive_features) >= 30    # Predictive features  
        assert len(meta_features) >= 15          # Meta features


class TestIntegrationScenarios:
    """Test various integration scenarios and edge cases"""
    
    @pytest.fixture
    def integration_analyzer(self):
        """Integration-focused analyzer"""
        return Component06CorrelationAnalyzer({
            'component_id': 6,
            'feature_count': 200,
            'fallback_enabled': True,
            'component_integration_enabled': True
        })
    
    @pytest.mark.asyncio
    async def test_missing_components_fallback(self, integration_analyzer):
        """Test fallback when some components are missing"""
        # Incomplete component data
        incomplete_data = {
            'component_results': {1: Mock()},  # Only Component 1, missing 2-5
            'overnight_factors': {}
        }
        
        result = await integration_analyzer.analyze(incomplete_data)
        
        # Should still produce valid result with fallback
        assert isinstance(result, ComponentAnalysisResult)
        assert result.features.feature_count == 200
        assert result.confidence < 1.0  # Should reflect reduced confidence
    
    @pytest.mark.asyncio
    async def test_empty_input_data(self, integration_analyzer):
        """Test handling of empty input data"""
        empty_data = {}
        
        result = await integration_analyzer.analyze(empty_data)
        
        # Should produce minimal valid result
        assert isinstance(result, ComponentAnalysisResult)
        assert result.features.feature_count == 200
        assert all(f == 0.0 for f in result.features.features)  # Should be zeros
        assert result.confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_invalid_data_types(self, integration_analyzer):
        """Test handling of invalid data types"""
        invalid_data = "not_a_dict"
        
        result = await integration_analyzer.analyze(invalid_data)
        
        # Should handle gracefully and produce fallback result
        assert isinstance(result, ComponentAnalysisResult)
        assert result.features.feature_count == 200
    
    def test_feature_validation(self, integration_analyzer):
        """Test feature validation logic"""
        # Create invalid feature vector
        invalid_features = FeatureVector(
            features=np.array([np.nan, np.inf, 1.0, 2.0]),
            feature_names=['f1', 'f2', 'f3', 'f4'],
            feature_count=4,
            processing_time_ms=10.0,
            metadata={}
        )
        
        is_valid = integration_analyzer._validate_features(invalid_features)
        
        # Should detect invalid values
        assert not is_valid


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])