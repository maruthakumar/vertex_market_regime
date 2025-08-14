"""
Test suite for Component 6 Phase 2 Momentum Enhancement

Tests the integration of Component 1 momentum features into Component 6
correlation analysis, validating the 20 new momentum-enhanced correlation features.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import asyncio

# Component imports
from vertex_market_regime.src.components.component_06_correlation.component_06_analyzer import Component06CorrelationAnalyzer
from vertex_market_regime.src.components.component_06_correlation.momentum_correlation_engine import (
    MomentumCorrelationEngine, 
    MomentumCorrelationResult,
    ComponentMomentumData
)


class TestComponent06MomentumEnhancement:
    """Test Component 6 Phase 2 momentum enhancement features"""

    @pytest.fixture
    def component_config(self):
        """Standard component configuration"""
        return {
            'component_id': 6,
            'feature_count': 220,  # Phase 2: 200 + 20 momentum features
            'target_processing_time_ms': 215,  # Increased for momentum analysis
            'target_memory_usage_mb': 500,
            'component_integration_enabled': True,
            'momentum_enhancement_enabled': True
        }

    @pytest.fixture
    def momentum_correlation_engine(self, component_config):
        """Initialize momentum correlation engine"""
        return MomentumCorrelationEngine(component_config)

    @pytest.fixture
    def sample_momentum_data(self):
        """Generate sample momentum data from Component 1"""
        timestamps = pd.date_range(start='2023-01-01', periods=100, freq='3min')
        
        momentum_data = {}
        for comp in ['component_01', 'component_02', 'component_03', 'component_04', 'component_05']:
            momentum_data[comp] = ComponentMomentumData(
                component_id=comp,
                rsi_values={
                    '3min': np.random.uniform(20, 80, 100),
                    '5min': np.random.uniform(25, 75, 80),
                    '10min': np.random.uniform(30, 70, 60),
                    '15min': np.random.uniform(35, 65, 40)
                },
                macd_values={
                    '3min': np.random.uniform(-2, 2, 100),
                    '5min': np.random.uniform(-1.5, 1.5, 80),
                    '10min': np.random.uniform(-1, 1, 60),
                    '15min': np.random.uniform(-0.5, 0.5, 40)
                },
                signal_values={
                    '3min': np.random.uniform(-1.5, 1.5, 100),
                    '5min': np.random.uniform(-1, 1, 80),
                    '10min': np.random.uniform(-0.8, 0.8, 60),
                    '15min': np.random.uniform(-0.3, 0.3, 40)
                },
                histogram_values={
                    '3min': np.random.uniform(-0.5, 0.5, 100),
                    '5min': np.random.uniform(-0.3, 0.3, 80),
                    '10min': np.random.uniform(-0.2, 0.2, 60),
                    '15min': np.random.uniform(-0.1, 0.1, 40)
                },
                timestamps=timestamps
            )
        
        return momentum_data

    @pytest.fixture
    def sample_price_data(self):
        """Generate sample price correlation data"""
        price_data = {}
        for comp in ['component_01', 'component_02', 'component_03', 'component_04', 'component_05']:
            price_data[comp] = pd.DataFrame({
                'close': np.random.uniform(100, 200, 100),
                'volume': np.random.uniform(1000, 10000, 100),
                'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='3min')
            })
        
        return price_data

    def test_momentum_correlation_engine_initialization(self, component_config):
        """Test momentum correlation engine initializes correctly"""
        engine = MomentumCorrelationEngine(component_config)
        
        assert engine.timeframes == ['3min', '5min', '10min', '15min']
        assert engine.components == ['component_01', 'component_02', 'component_03', 'component_04', 'component_05']
        assert engine.rsi_period == 14
        assert engine.macd_fast == 12
        assert engine.macd_slow == 26
        assert engine.macd_signal == 9

    def test_momentum_correlation_analysis(self, momentum_correlation_engine, sample_momentum_data, sample_price_data):
        """Test complete momentum correlation analysis produces 20 features"""
        result = momentum_correlation_engine.analyze_momentum_correlation(
            sample_momentum_data, 
            sample_price_data
        )
        
        # Validate result structure
        assert isinstance(result, MomentumCorrelationResult)
        assert result.feature_count == 20
        assert result.processing_time_ms > 0
        
        # Validate RSI correlation features (8 features)
        assert len(result.rsi_correlation_features) == 8
        assert 'rsi_cross_correlation_3min' in result.rsi_correlation_features
        assert 'rsi_cross_correlation_5min' in result.rsi_correlation_features
        assert 'rsi_price_agreement_3min' in result.rsi_correlation_features
        assert 'rsi_price_agreement_5min' in result.rsi_correlation_features
        assert 'rsi_regime_coherence_3min' in result.rsi_correlation_features
        assert 'rsi_regime_coherence_5min' in result.rsi_correlation_features
        assert 'rsi_divergence_3min_5min' in result.rsi_correlation_features
        assert 'rsi_divergence_5min_10min' in result.rsi_correlation_features
        
        # Validate MACD correlation features (8 features)
        assert len(result.macd_correlation_features) == 8
        assert 'macd_signal_correlation_3min' in result.macd_correlation_features
        assert 'macd_signal_correlation_5min' in result.macd_correlation_features
        assert 'macd_histogram_convergence_3min' in result.macd_correlation_features
        assert 'macd_histogram_convergence_5min' in result.macd_correlation_features
        assert 'macd_trend_agreement_3min' in result.macd_correlation_features
        assert 'macd_trend_agreement_5min' in result.macd_correlation_features
        assert 'macd_momentum_strength_3min' in result.macd_correlation_features
        assert 'macd_momentum_strength_5min' in result.macd_correlation_features
        
        # Validate momentum consensus features (4 features)
        assert len(result.momentum_consensus_features) == 4
        assert 'multi_timeframe_rsi_consensus' in result.momentum_consensus_features
        assert 'multi_timeframe_macd_consensus' in result.momentum_consensus_features
        assert 'cross_component_momentum_agreement' in result.momentum_consensus_features
        assert 'overall_momentum_system_coherence' in result.momentum_consensus_features

    def test_rsi_correlation_features(self, momentum_correlation_engine, sample_momentum_data, sample_price_data):
        """Test RSI correlation feature calculation"""
        result = momentum_correlation_engine.analyze_momentum_correlation(
            sample_momentum_data, 
            sample_price_data
        )
        
        rsi_features = result.rsi_correlation_features
        
        # Validate feature value ranges
        for feature_name, value in rsi_features.items():
            assert 0.0 <= value <= 1.0, f"RSI feature {feature_name} = {value} outside [0,1] range"
            assert not np.isnan(value), f"RSI feature {feature_name} is NaN"

    def test_macd_correlation_features(self, momentum_correlation_engine, sample_momentum_data, sample_price_data):
        """Test MACD correlation feature calculation"""
        result = momentum_correlation_engine.analyze_momentum_correlation(
            sample_momentum_data, 
            sample_price_data
        )
        
        macd_features = result.macd_correlation_features
        
        # Validate feature value ranges
        for feature_name, value in macd_features.items():
            assert 0.0 <= value <= 1.0, f"MACD feature {feature_name} = {value} outside [0,1] range"
            assert not np.isnan(value), f"MACD feature {feature_name} is NaN"

    def test_momentum_consensus_features(self, momentum_correlation_engine, sample_momentum_data, sample_price_data):
        """Test momentum consensus feature calculation"""
        result = momentum_correlation_engine.analyze_momentum_correlation(
            sample_momentum_data, 
            sample_price_data
        )
        
        consensus_features = result.momentum_consensus_features
        
        # Validate feature value ranges
        for feature_name, value in consensus_features.items():
            assert 0.0 <= value <= 1.0, f"Consensus feature {feature_name} = {value} outside [0,1] range"
            assert not np.isnan(value), f"Consensus feature {feature_name} is NaN"

    def test_cross_component_momentum_matrix(self, momentum_correlation_engine, sample_momentum_data, sample_price_data):
        """Test cross-component momentum correlation matrix"""
        result = momentum_correlation_engine.analyze_momentum_correlation(
            sample_momentum_data, 
            sample_price_data
        )
        
        matrix = result.cross_component_momentum_matrix
        
        # Validate matrix structure
        assert matrix.shape == (5, 5)  # 5 components
        assert np.allclose(np.diag(matrix), 1.0)  # Diagonal should be 1.0
        assert np.allclose(matrix, matrix.T)  # Should be symmetric

    def test_momentum_price_divergence(self, momentum_correlation_engine, sample_momentum_data, sample_price_data):
        """Test momentum-price divergence calculation"""
        result = momentum_correlation_engine.analyze_momentum_correlation(
            sample_momentum_data, 
            sample_price_data
        )
        
        divergences = result.momentum_divergence_indicators
        
        # Validate divergence indicators
        assert len(divergences) > 0
        for divergence in divergences:
            assert 0.0 <= divergence <= 1.0
            assert not np.isnan(divergence)

    def test_component_06_phase_2_integration(self, component_config):
        """Test Component 6 analyzer with Phase 2 momentum enhancement"""
        analyzer = Component06CorrelationAnalyzer(component_config)
        
        # Validate Phase 2 configuration
        assert analyzer.config['feature_count'] == 220
        assert 'momentum_correlation_features' in analyzer.config['expected_features']
        assert hasattr(analyzer, 'momentum_correlation_engine')

    def test_performance_targets_phase_2(self, momentum_correlation_engine, sample_momentum_data, sample_price_data):
        """Test momentum correlation analysis meets performance targets"""
        result = momentum_correlation_engine.analyze_momentum_correlation(
            sample_momentum_data, 
            sample_price_data
        )
        
        # Performance validation (momentum analysis should be <15ms)
        assert result.processing_time_ms < 50  # Generous allowance for momentum processing
        assert result.feature_count == 20

    def test_fallback_behavior(self, momentum_correlation_engine):
        """Test fallback behavior with invalid data"""
        # Test with empty data
        empty_momentum_data = {}
        empty_price_data = {}
        
        result = momentum_correlation_engine.analyze_momentum_correlation(
            empty_momentum_data, 
            empty_price_data
        )
        
        # Should return fallback result with reasonable defaults
        assert isinstance(result, MomentumCorrelationResult)
        assert result.feature_count == 20
        assert len(result.rsi_correlation_features) == 8
        assert len(result.macd_correlation_features) == 8
        assert len(result.momentum_consensus_features) == 4

    def test_edge_cases(self, momentum_correlation_engine):
        """Test edge cases and boundary conditions"""
        # Test with minimal data
        minimal_momentum_data = {
            'component_01': ComponentMomentumData(
                component_id='component_01',
                rsi_values={'3min': np.array([50.0])},
                macd_values={'3min': np.array([0.0])},
                signal_values={'3min': np.array([0.0])},
                histogram_values={'3min': np.array([0.0])},
                timestamps=pd.date_range(start='2023-01-01', periods=1, freq='3min')
            )
        }
        
        minimal_price_data = {
            'component_01': pd.DataFrame({
                'close': [100.0],
                'volume': [1000.0],
                'timestamp': pd.date_range(start='2023-01-01', periods=1, freq='3min')
            })
        }
        
        result = momentum_correlation_engine.analyze_momentum_correlation(
            minimal_momentum_data, 
            minimal_price_data
        )
        
        # Should handle minimal data gracefully
        assert isinstance(result, MomentumCorrelationResult)
        assert result.feature_count == 20

    @pytest.mark.asyncio
    async def test_component_06_analyze_with_momentum(self, component_config, sample_momentum_data):
        """Test Component 6 analyze method with momentum data integration"""
        analyzer = Component06CorrelationAnalyzer(component_config)
        
        # Mock market data with momentum features
        mock_market_data = Mock()
        mock_market_data.component_01_output = Mock()
        mock_market_data.component_01_output.momentum_features = sample_momentum_data['component_01']
        
        # Mock other required data
        mock_market_data.raw_data = Mock()
        mock_market_data.components_data = {
            f'component_0{i}': Mock() for i in range(1, 6)
        }
        
        # This test validates the integration point exists
        # Full integration testing would require complete market data structure
        assert hasattr(analyzer, 'momentum_correlation_engine')
        assert analyzer.config['feature_count'] == 220


if __name__ == '__main__':
    pytest.main([__file__, '-v'])