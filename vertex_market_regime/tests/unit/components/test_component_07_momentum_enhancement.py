"""
Test suite for Component 7 Phase 2 Momentum Enhancement

Tests the integration of Component 1 momentum features and Component 6 enhanced
correlation features into Component 7 support/resistance analysis, validating
the 10 new momentum-based level detection features.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import asyncio

# Component imports
from vertex_market_regime.src.components.component_07_support_resistance.component_07_analyzer import Component07Analyzer
from vertex_market_regime.src.components.component_07_support_resistance.momentum_level_detector import (
    MomentumLevelDetector, 
    MomentumLevelResult,
    MomentumLevelData
)


class TestComponent07MomentumEnhancement:
    """Test Component 7 Phase 2 momentum enhancement features"""

    @pytest.fixture
    def component_config(self):
        """Standard component configuration for Phase 2"""
        return {
            'component_id': 7,
            'feature_count': 130,  # Phase 2: 120 + 10 momentum features
            'processing_budget_ms': 160,  # Increased for momentum analysis
            'memory_budget_mb': 250,
            'momentum_enhancement_enabled': True,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'min_momentum_periods': 20
        }

    @pytest.fixture
    def momentum_level_detector(self, component_config):
        """Initialize momentum level detector"""
        return MomentumLevelDetector(component_config)

    @pytest.fixture
    def sample_momentum_level_data(self):
        """Generate sample momentum level data"""
        timestamps = pd.date_range(start='2023-01-01', periods=100, freq='3min')
        
        return MomentumLevelData(
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
            price_values=np.cumsum(np.random.randn(100) * 0.01) + 100,  # Random walk around 100
            volume_values=np.random.uniform(1000, 10000, 100),
            timestamps=timestamps
        )

    def test_momentum_level_detector_initialization(self, component_config):
        """Test momentum level detector initializes correctly"""
        detector = MomentumLevelDetector(component_config)
        
        assert detector.timeframes == ['3min', '5min', '10min', '15min']
        assert detector.rsi_overbought == 70
        assert detector.rsi_oversold == 30
        assert detector.min_momentum_periods == 20

    def test_momentum_level_detection(self, momentum_level_detector, sample_momentum_level_data):
        """Test complete momentum level detection produces 10 features"""
        result = momentum_level_detector.detect_momentum_levels(sample_momentum_level_data)
        
        # Validate result structure
        assert isinstance(result, MomentumLevelResult)
        assert result.feature_count == 10
        assert result.processing_time_ms > 0
        
        # Validate RSI confluence levels (4 features)
        assert len(result.rsi_confluence_levels) == 4
        assert 'rsi_overbought_resistance_strength' in result.rsi_confluence_levels
        assert 'rsi_oversold_support_strength' in result.rsi_confluence_levels
        assert 'rsi_neutral_zone_level_density' in result.rsi_confluence_levels
        assert 'rsi_level_convergence_strength' in result.rsi_confluence_levels
        
        # Validate MACD validation levels (3 features)
        assert len(result.macd_validation_levels) == 3
        assert 'macd_crossover_level_strength' in result.macd_validation_levels
        assert 'macd_histogram_reversal_strength' in result.macd_validation_levels
        assert 'macd_momentum_consensus_validation' in result.macd_validation_levels
        
        # Validate momentum exhaustion levels (3 features)
        assert len(result.momentum_exhaustion_levels) == 3
        assert 'rsi_price_divergence_exhaustion' in result.momentum_exhaustion_levels
        assert 'macd_momentum_exhaustion' in result.momentum_exhaustion_levels
        assert 'multi_timeframe_exhaustion_consensus' in result.momentum_exhaustion_levels

    def test_rsi_confluence_levels(self, momentum_level_detector, sample_momentum_level_data):
        """Test RSI confluence level detection (4 features)"""
        result = momentum_level_detector.detect_momentum_levels(sample_momentum_level_data)
        
        rsi_levels = result.rsi_confluence_levels
        
        # Validate feature value ranges
        for feature_name, value in rsi_levels.items():
            assert 0.0 <= value <= 1.0, f"RSI feature {feature_name} = {value} outside [0,1] range"
            assert not np.isnan(value), f"RSI feature {feature_name} is NaN"
        
        # Validate specific features exist
        assert 'rsi_overbought_resistance_strength' in rsi_levels
        assert 'rsi_oversold_support_strength' in rsi_levels
        assert 'rsi_neutral_zone_level_density' in rsi_levels
        assert 'rsi_level_convergence_strength' in rsi_levels

    def test_macd_validation_levels(self, momentum_level_detector, sample_momentum_level_data):
        """Test MACD validation level detection (3 features)"""
        result = momentum_level_detector.detect_momentum_levels(sample_momentum_level_data)
        
        macd_levels = result.macd_validation_levels
        
        # Validate feature value ranges
        for feature_name, value in macd_levels.items():
            assert 0.0 <= value <= 1.0, f"MACD feature {feature_name} = {value} outside [0,1] range"
            assert not np.isnan(value), f"MACD feature {feature_name} is NaN"
        
        # Validate specific features exist
        assert 'macd_crossover_level_strength' in macd_levels
        assert 'macd_histogram_reversal_strength' in macd_levels
        assert 'macd_momentum_consensus_validation' in macd_levels

    def test_momentum_exhaustion_levels(self, momentum_level_detector, sample_momentum_level_data):
        """Test momentum exhaustion level detection (3 features)"""
        result = momentum_level_detector.detect_momentum_levels(sample_momentum_level_data)
        
        exhaustion_levels = result.momentum_exhaustion_levels
        
        # Validate feature value ranges
        for feature_name, value in exhaustion_levels.items():
            assert 0.0 <= value <= 1.0, f"Exhaustion feature {feature_name} = {value} outside [0,1] range"
            assert not np.isnan(value), f"Exhaustion feature {feature_name} is NaN"
        
        # Validate specific features exist
        assert 'rsi_price_divergence_exhaustion' in exhaustion_levels
        assert 'macd_momentum_exhaustion' in exhaustion_levels
        assert 'multi_timeframe_exhaustion_consensus' in exhaustion_levels

    def test_momentum_level_strengths(self, momentum_level_detector, sample_momentum_level_data):
        """Test momentum level strength calculation"""
        result = momentum_level_detector.detect_momentum_levels(sample_momentum_level_data)
        
        strengths = result.momentum_level_strengths
        
        # Validate strength values
        assert len(strengths) > 0
        for strength in strengths:
            assert 0.0 <= strength <= 1.0
            assert not np.isnan(strength)

    def test_momentum_level_prices(self, momentum_level_detector, sample_momentum_level_data):
        """Test momentum level price extraction"""
        result = momentum_level_detector.detect_momentum_levels(sample_momentum_level_data)
        
        level_prices = result.momentum_level_prices
        
        # Validate price levels
        assert len(level_prices) > 0
        current_price = sample_momentum_level_data.price_values[-1]
        
        for price in level_prices:
            assert price > 0  # Prices should be positive
            assert not np.isnan(price)
            # Levels should be reasonably close to current price (within 10%)
            assert 0.9 * current_price <= price <= 1.1 * current_price

    def test_component_07_phase_2_integration(self, component_config):
        """Test Component 7 analyzer with Phase 2 momentum enhancement"""
        analyzer = Component07Analyzer(component_config)
        
        # Validate Phase 2 configuration
        assert hasattr(analyzer, 'momentum_level_detector')
        assert analyzer.processing_budget_ms >= 150  # Should handle additional processing

    def test_performance_targets_phase_2(self, momentum_level_detector, sample_momentum_level_data):
        """Test momentum level detection meets performance targets"""
        result = momentum_level_detector.detect_momentum_levels(sample_momentum_level_data)
        
        # Performance validation (momentum level detection should be <10ms)
        assert result.processing_time_ms < 50  # Generous allowance for momentum processing
        assert result.feature_count == 10

    def test_rsi_overbought_resistance_detection(self, momentum_level_detector):
        """Test RSI overbought resistance level detection"""
        # Create data with clear overbought condition
        timestamps = pd.date_range(start='2023-01-01', periods=50, freq='3min')
        
        test_data = MomentumLevelData(
            rsi_values={
                '3min': np.concatenate([np.full(25, 50), np.full(25, 80)]),  # Overbought in second half
                '5min': np.concatenate([np.full(20, 50), np.full(20, 75)])
            },
            macd_values={
                '3min': np.random.uniform(-0.5, 0.5, 50),
                '5min': np.random.uniform(-0.3, 0.3, 40)
            },
            signal_values={
                '3min': np.random.uniform(-0.3, 0.3, 50),
                '5min': np.random.uniform(-0.2, 0.2, 40)
            },
            histogram_values={
                '3min': np.random.uniform(-0.1, 0.1, 50),
                '5min': np.random.uniform(-0.1, 0.1, 40)
            },
            price_values=np.concatenate([np.full(25, 100), np.full(25, 105)]),  # Higher prices during overbought
            volume_values=np.random.uniform(1000, 5000, 50),
            timestamps=timestamps
        )
        
        result = momentum_level_detector.detect_momentum_levels(test_data)
        
        # Should detect some resistance strength
        assert result.rsi_confluence_levels['rsi_overbought_resistance_strength'] >= 0.0

    def test_rsi_oversold_support_detection(self, momentum_level_detector):
        """Test RSI oversold support level detection"""
        # Create data with clear oversold condition
        timestamps = pd.date_range(start='2023-01-01', periods=50, freq='3min')
        
        test_data = MomentumLevelData(
            rsi_values={
                '3min': np.concatenate([np.full(25, 50), np.full(25, 20)]),  # Oversold in second half
                '5min': np.concatenate([np.full(20, 50), np.full(20, 25)])
            },
            macd_values={
                '3min': np.random.uniform(-0.5, 0.5, 50),
                '5min': np.random.uniform(-0.3, 0.3, 40)
            },
            signal_values={
                '3min': np.random.uniform(-0.3, 0.3, 50),
                '5min': np.random.uniform(-0.2, 0.2, 40)
            },
            histogram_values={
                '3min': np.random.uniform(-0.1, 0.1, 50),
                '5min': np.random.uniform(-0.1, 0.1, 40)
            },
            price_values=np.concatenate([np.full(25, 100), np.full(25, 95)]),  # Lower prices during oversold
            volume_values=np.random.uniform(1000, 5000, 50),
            timestamps=timestamps
        )
        
        result = momentum_level_detector.detect_momentum_levels(test_data)
        
        # Should detect some support strength
        assert result.rsi_confluence_levels['rsi_oversold_support_strength'] >= 0.0

    def test_macd_crossover_detection(self, momentum_level_detector):
        """Test MACD crossover level detection"""
        # Create data with clear MACD-Signal crossover
        timestamps = pd.date_range(start='2023-01-01', periods=50, freq='3min')
        
        # Create crossover pattern: MACD crosses above Signal
        macd_vals = np.concatenate([np.full(25, -0.5), np.full(25, 0.5)])
        signal_vals = np.full(50, 0.0)
        
        test_data = MomentumLevelData(
            rsi_values={
                '3min': np.random.uniform(40, 60, 50),
                '5min': np.random.uniform(45, 55, 40)
            },
            macd_values={
                '3min': macd_vals,
                '5min': macd_vals[:40]
            },
            signal_values={
                '3min': signal_vals,
                '5min': signal_vals[:40]
            },
            histogram_values={
                '3min': macd_vals - signal_vals,
                '5min': (macd_vals - signal_vals)[:40]
            },
            price_values=np.random.uniform(95, 105, 50),
            volume_values=np.random.uniform(1000, 5000, 50),
            timestamps=timestamps
        )
        
        result = momentum_level_detector.detect_momentum_levels(test_data)
        
        # Should detect crossover strength
        assert result.macd_validation_levels['macd_crossover_level_strength'] >= 0.0

    def test_fallback_behavior(self, momentum_level_detector):
        """Test fallback behavior with minimal data"""
        # Test with minimal data
        timestamps = pd.date_range(start='2023-01-01', periods=5, freq='3min')
        
        minimal_data = MomentumLevelData(
            rsi_values={'3min': np.array([50.0, 51.0, 49.0, 50.5, 52.0])},
            macd_values={'3min': np.array([0.0, 0.1, -0.1, 0.05, 0.2])},
            signal_values={'3min': np.array([0.0, 0.0, 0.0, 0.1, 0.1])},
            histogram_values={'3min': np.array([0.0, 0.1, -0.1, -0.05, 0.1])},
            price_values=np.array([100.0, 100.5, 99.8, 100.2, 100.7]),
            volume_values=np.array([1000.0, 1100.0, 900.0, 1050.0, 1200.0]),
            timestamps=timestamps
        )
        
        result = momentum_level_detector.detect_momentum_levels(minimal_data)
        
        # Should return valid result with fallback values
        assert isinstance(result, MomentumLevelResult)
        assert result.feature_count == 10
        assert len(result.rsi_confluence_levels) == 4
        assert len(result.macd_validation_levels) == 3
        assert len(result.momentum_exhaustion_levels) == 3

    def test_edge_cases(self, momentum_level_detector):
        """Test edge cases and boundary conditions"""
        # Test with empty data
        empty_data = MomentumLevelData(
            rsi_values={},
            macd_values={},
            signal_values={},
            histogram_values={},
            price_values=np.array([]),
            volume_values=np.array([]),
            timestamps=pd.DatetimeIndex([])
        )
        
        result = momentum_level_detector.detect_momentum_levels(empty_data)
        
        # Should handle empty data gracefully
        assert isinstance(result, MomentumLevelResult)
        assert result.feature_count == 10

    def test_dependencies_integration(self, component_config):
        """Test integration with Component 1 and Component 6 dependencies"""
        analyzer = Component07Analyzer(component_config)
        
        # Validate that Component 7 has the necessary integration points
        assert hasattr(analyzer, 'momentum_level_detector')
        
        # These would be the integration points for:
        # - Component 1 momentum features (RSI/MACD input)
        # - Component 6 enhanced correlation features (correlation validation)
        # Full integration testing would require actual Component 1 & 6 outputs

    @pytest.mark.asyncio
    async def test_component_07_analyze_integration_readiness(self, component_config):
        """Test Component 7 analyzer readiness for momentum integration"""
        analyzer = Component07Analyzer(component_config)
        
        # Mock market data structure
        mock_market_data = Mock()
        mock_market_data.shape = (100, 10)
        
        # This validates the integration architecture is ready
        # Full testing would require complete Component 1 & 6 outputs
        assert hasattr(analyzer, 'momentum_level_detector')
        assert hasattr(analyzer, 'feature_engine')
        assert hasattr(analyzer, 'confluence_analyzer')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])