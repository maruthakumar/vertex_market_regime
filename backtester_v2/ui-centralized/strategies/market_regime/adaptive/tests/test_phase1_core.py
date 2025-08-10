"""
Test Suite for Phase 1 Core Infrastructure

Tests for:
- Adaptive Regime Config Manager
- Historical Regime Analyzer  
- Regime Definition Builder
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
from datetime import datetime, timedelta

# Import modules to test
import sys
sys.path.append('/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/market_regime')

from adaptive.config.adaptive_regime_config_manager import (
    AdaptiveRegimeConfigManager, AdaptiveRegimeConfig, RegimeCount
)
from adaptive.analysis.historical_regime_analyzer import (
    HistoricalRegimeAnalyzer, RegimePattern, TransitionDynamics
)
from adaptive.core.regime_definition_builder import (
    RegimeDefinitionBuilder, RegimeDefinition, RegimeBoundary
)


class TestAdaptiveRegimeConfigManager:
    """Test suite for configuration management"""
    
    @pytest.fixture
    def temp_config_file(self):
        """Create temporary config file"""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            manager = AdaptiveRegimeConfigManager()
            manager.generate_template(tmp.name)
            yield tmp.name
        os.unlink(tmp.name)
    
    def test_default_configuration(self):
        """Test loading default configuration"""
        manager = AdaptiveRegimeConfigManager()
        assert manager.config is not None
        assert manager.config.regime_count == 12
        assert manager.config.historical_lookback_days == 90
        assert manager.config.intraday_window == '5min'
    
    def test_configuration_validation(self):
        """Test parameter validation"""
        manager = AdaptiveRegimeConfigManager()
        
        # Valid parameters
        valid_params = {
            'regime_count': 12,
            'historical_lookback_days': 90,
            'intraday_window': '5min',
            'transition_sensitivity': 0.7,
            'adaptive_learning_rate': 0.05,
            'min_regime_duration': 15,
            'noise_filter_window': 5,
            'enable_asl': True,
            'enable_hysteresis': True,
            'confidence_threshold': 0.65
        }
        assert manager.validate_parameters(valid_params) == True
        
        # Invalid regime count
        invalid_params = valid_params.copy()
        invalid_params['regime_count'] = 10
        assert manager.validate_parameters(invalid_params) == False
        
        # Invalid learning rate
        invalid_params = valid_params.copy()
        invalid_params['adaptive_learning_rate'] = 0.5
        assert manager.validate_parameters(invalid_params) == False
    
    def test_template_generation(self):
        """Test configuration template generation"""
        manager = AdaptiveRegimeConfigManager()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            template_path = Path(tmpdir) / "test_template.xlsx"
            result_path = manager.generate_template(str(template_path))
            
            assert result_path.exists()
            
            # Verify sheets exist
            df_main = pd.read_excel(result_path, sheet_name='Adaptive Regime Formation')
            assert len(df_main) > 0
            assert 'Parameter' in df_main.columns
            assert 'Value' in df_main.columns
            
            df_profiles = pd.read_excel(result_path, sheet_name='Profile Examples')
            assert len(df_profiles) > 0
    
    def test_regime_specific_config(self):
        """Test regime-specific configuration adjustments"""
        manager = AdaptiveRegimeConfigManager()
        
        config_8 = manager.get_regime_specific_config(8)
        assert config_8['min_regime_duration'] == 20
        assert config_8['transition_sensitivity'] == 0.75
        
        config_18 = manager.get_regime_specific_config(18)
        assert config_18['min_regime_duration'] == 10
        assert config_18['transition_sensitivity'] == 0.65
        
        with pytest.raises(ValueError):
            manager.get_regime_specific_config(15)
    
    def test_profile_presets(self):
        """Test configuration profiles"""
        manager = AdaptiveRegimeConfigManager()
        
        # Conservative profile
        conservative = manager.PROFILE_PRESETS['conservative']
        assert conservative['transition_sensitivity'] == 0.8
        assert conservative['adaptive_learning_rate'] == 0.02
        
        # Aggressive profile
        aggressive = manager.PROFILE_PRESETS['aggressive']
        assert aggressive['transition_sensitivity'] == 0.6
        assert aggressive['adaptive_learning_rate'] == 0.1
    
    def test_config_export_summary(self):
        """Test configuration summary export"""
        manager = AdaptiveRegimeConfigManager()
        summary = manager.export_config_summary()
        
        assert 'regime_count' in summary
        assert 'profile' in summary
        assert 'last_updated' in summary
        assert summary['regime_count'] == 12
        assert summary['asl_enabled'] == True


class TestHistoricalRegimeAnalyzer:
    """Test suite for historical analysis"""
    
    @pytest.fixture
    def sample_market_data(self):
        """Generate sample market data"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=5000, freq='5min')
        
        # Generate realistic market data
        price = 100
        prices = []
        volumes = []
        
        for i in range(5000):
            # Add trend and noise
            trend = 0.0001 * i
            noise = np.random.randn() * 0.5
            price = price * (1 + trend + noise/100)
            prices.append(price)
            
            # Volume with some patterns
            base_volume = 5000
            volume = base_volume + np.random.randint(-2000, 2000)
            volumes.append(volume)
        
        return pd.DataFrame({
            'timestamp': dates,
            'price': prices,
            'volume': volumes
        })
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization"""
        config = {
            'historical_lookback_days': 90,
            'regime_count': 12,
            'intraday_window': '5min'
        }
        
        analyzer = HistoricalRegimeAnalyzer(config)
        assert analyzer.lookback_days == 90
        assert analyzer.regime_count == 12
        assert analyzer.intraday_window == '5min'
    
    def test_data_preparation(self, sample_market_data):
        """Test data preparation and feature engineering"""
        analyzer = HistoricalRegimeAnalyzer({'regime_count': 12})
        prepared_data = analyzer._prepare_data(sample_market_data)
        
        # Check required columns exist
        required_features = ['returns', 'log_returns', 'volume_ratio', 'volatility_20']
        for feature in required_features:
            assert feature in prepared_data.columns
        
        # Check no NaN values after preparation
        assert prepared_data.isna().sum().sum() == 0
        
        # Check data length (should be less due to rolling calculations)
        assert len(prepared_data) < len(sample_market_data)
    
    def test_feature_extraction(self, sample_market_data):
        """Test feature extraction for clustering"""
        analyzer = HistoricalRegimeAnalyzer({'regime_count': 12})
        prepared_data = analyzer._prepare_data(sample_market_data)
        features = analyzer._extract_features(prepared_data)
        
        # Check feature matrix shape
        assert features.shape[0] == len(prepared_data)
        assert features.shape[1] > 0
        
        # Check features are standardized
        assert np.abs(features.mean()) < 0.1  # Close to 0
        assert 0.5 < features.std() < 1.5  # Close to 1
    
    def test_clustering(self, sample_market_data):
        """Test regime clustering"""
        analyzer = HistoricalRegimeAnalyzer({
            'regime_count': 8,
            'clustering_algorithm': 'kmeans'
        })
        
        prepared_data = analyzer._prepare_data(sample_market_data)
        features = analyzer._extract_features(prepared_data)
        labels = analyzer._perform_clustering(features)
        
        # Check labels
        assert len(labels) == len(features)
        assert set(labels) == set(range(8))  # All regimes present
        
        # Check minimum duration enforcement
        regime_changes = np.sum(np.diff(labels) != 0)
        assert regime_changes < len(labels) * 0.5  # Not too many changes
    
    def test_regime_characteristics(self, sample_market_data):
        """Test regime characteristic analysis"""
        analyzer = HistoricalRegimeAnalyzer({'regime_count': 8})
        
        prepared_data = analyzer._prepare_data(sample_market_data)
        features = analyzer._extract_features(prepared_data)
        labels = analyzer._perform_clustering(features)
        
        patterns = analyzer._analyze_regime_characteristics(prepared_data, labels)
        
        # Check all regimes analyzed
        assert len(patterns) <= 8
        
        # Check pattern structure
        for regime_id, pattern in patterns.items():
            assert isinstance(pattern.volatility_range, tuple)
            assert isinstance(pattern.trend_range, tuple)
            assert 'mean' in pattern.volume_profile
            assert pattern.average_duration > 0
            assert 0 <= pattern.stability_score <= 1
    
    def test_transition_matrix(self, sample_market_data):
        """Test transition matrix calculation"""
        analyzer = HistoricalRegimeAnalyzer({'regime_count': 8})
        
        # Simple test with known transitions
        labels = np.array([0, 0, 1, 1, 2, 2, 0, 0, 1, 1])
        transition_matrix = analyzer._build_transition_matrix(labels)
        
        # Check matrix properties
        assert transition_matrix.shape == (8, 8)
        
        # Check rows sum to 1 (probability matrix)
        for i in range(8):
            row_sum = transition_matrix.loc[i].sum()
            if row_sum > 0:  # Only for regimes with transitions
                assert abs(row_sum - 1.0) < 0.01
    
    def test_full_analysis(self, sample_market_data):
        """Test complete analysis pipeline"""
        analyzer = HistoricalRegimeAnalyzer({
            'historical_lookback_days': 30,
            'regime_count': 8,
            'intraday_window': '5min'
        })
        
        results = analyzer.analyze_historical_patterns(sample_market_data)
        
        # Check all expected outputs
        assert 'regime_patterns' in results
        assert 'transition_matrix' in results
        assert 'transition_dynamics' in results
        assert 'stability_metrics' in results
        assert 'feature_importance' in results
        assert 'cluster_quality' in results
        
        # Check stability metrics
        stability = results['stability_metrics']
        assert 'total_transitions' in stability
        assert 'average_regime_duration' in stability
        assert stability['regime_persistence'] >= 0


class TestRegimeDefinitionBuilder:
    """Test suite for regime definition builder"""
    
    @pytest.fixture
    def mock_analysis_results(self):
        """Create mock analysis results"""
        patterns = {}
        for i in range(12):
            patterns[i] = RegimePattern(
                regime_id=i,
                volatility_range=(0.1 + i*0.05, 0.15 + i*0.05),
                trend_range=(-0.01 + i*0.002, 0.01 + i*0.002),
                volume_profile={'mean': 5000, 'std': 1000, 'skew': 0.1, 'relative': 1.0},
                average_duration=30 + i*5,
                transition_probabilities={},
                characteristic_features=['test_feature'],
                stability_score=0.7
            )
        
        return {
            'regime_patterns': patterns,
            'transition_matrix': pd.DataFrame(np.eye(12) * 0.8 + 0.02),
            'stability_metrics': {'average_regime_duration': 45}
        }
    
    def test_builder_initialization(self):
        """Test builder initialization"""
        builder = RegimeDefinitionBuilder(regime_count=12)
        assert builder.regime_count == 12
        
        # Test invalid counts
        with pytest.raises(ValueError):
            RegimeDefinitionBuilder(regime_count=10)
    
    def test_single_definition_building(self, mock_analysis_results):
        """Test building single regime definition"""
        builder = RegimeDefinitionBuilder(regime_count=12)
        
        pattern = mock_analysis_results['regime_patterns'][0]
        definition = builder._build_single_definition(
            0, pattern, pd.DataFrame(), {}
        )
        
        assert definition.regime_id == 0
        assert definition.name == "Low Vol Bullish Trending"
        assert isinstance(definition.boundaries, RegimeBoundary)
        assert len(definition.strategy_preferences) > 0
        assert 'position_size_multiplier' in definition.risk_parameters
    
    def test_boundary_expansion(self):
        """Test boundary expansion logic"""
        builder = RegimeDefinitionBuilder(regime_count=12)
        
        original = (0.1, 0.2)
        expanded = builder._expand_bounds(original, 0.1)
        
        assert expanded[0] < original[0]
        assert expanded[1] > original[1]
        assert expanded[1] - expanded[0] > original[1] - original[0]
    
    def test_strategy_preferences(self):
        """Test strategy preference determination"""
        builder = RegimeDefinitionBuilder(regime_count=12)
        
        # Low volatility features
        prefs = builder._determine_strategy_preferences(['low_volatility'])
        assert 'TBS' in prefs
        assert 'ML_INDICATOR' in prefs
        
        # High volatility features
        prefs = builder._determine_strategy_preferences(['high_volatility'])
        assert 'MARKET_REGIME' in prefs
        assert 'POS' in prefs
        
        # Directional features
        prefs = builder._determine_strategy_preferences(['bullish_bias'])
        assert 'TBS' in prefs or 'TV' in prefs
    
    def test_risk_parameters(self):
        """Test risk parameter calculation"""
        builder = RegimeDefinitionBuilder(regime_count=12)
        
        # Low volatility pattern
        low_vol_pattern = type('Pattern', (), {
            'volatility_range': (0.05, 0.10),
            'stability_score': 0.8
        })()
        
        risk_params = builder._calculate_risk_parameters(low_vol_pattern)
        assert risk_params['position_size_multiplier'] > 1.0
        assert risk_params['stop_loss_multiplier'] < 1.0
        
        # High volatility pattern
        high_vol_pattern = type('Pattern', (), {
            'volatility_range': (0.4, 0.5),
            'stability_score': 0.3
        })()
        
        risk_params = builder._calculate_risk_parameters(high_vol_pattern)
        assert risk_params['position_size_multiplier'] < 1.0
        assert risk_params['stop_loss_multiplier'] > 1.0
    
    def test_full_definition_building(self, mock_analysis_results):
        """Test building complete set of definitions"""
        builder = RegimeDefinitionBuilder(regime_count=12)
        definitions = builder.build_regime_definitions(mock_analysis_results)
        
        assert len(definitions) == 12
        
        for regime_id, definition in definitions.items():
            assert definition.regime_id == regime_id
            assert definition.name in builder.REGIME_NAMES[12]
            assert definition.boundaries is not None
            assert len(definition.strategy_preferences) > 0
    
    def test_boundary_optimization(self, mock_analysis_results):
        """Test boundary optimization"""
        builder = RegimeDefinitionBuilder(regime_count=8)
        definitions = builder.build_regime_definitions(mock_analysis_results)
        
        # Check no overlapping volatility bounds for 8-regime
        sorted_defs = sorted(
            definitions.values(),
            key=lambda x: x.boundaries.volatility_bounds[0]
        )
        
        for i in range(1, len(sorted_defs)):
            prev_upper = sorted_defs[i-1].boundaries.volatility_bounds[1]
            curr_lower = sorted_defs[i].boundaries.volatility_bounds[0]
            assert prev_upper <= curr_lower  # No overlap
    
    def test_regime_lookup(self, mock_analysis_results):
        """Test regime lookup by conditions"""
        builder = RegimeDefinitionBuilder(regime_count=12)
        definitions = builder.build_regime_definitions(mock_analysis_results)
        
        # Test finding regime by conditions
        test_volatility = 0.15
        test_trend = 0.005
        test_volume = 1.0
        
        regime = builder.get_regime_by_conditions(
            test_volatility, test_trend, test_volume
        )
        
        # Should find a regime or None
        if regime:
            assert isinstance(regime, RegimeDefinition)
            assert regime.boundaries.contains(test_volatility, test_trend, test_volume)
    
    def test_export_definitions(self, mock_analysis_results):
        """Test exporting definitions to JSON"""
        builder = RegimeDefinitionBuilder(regime_count=12)
        definitions = builder.build_regime_definitions(mock_analysis_results)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "definitions.json"
            json_str = builder.export_definitions(str(output_path))
            
            assert output_path.exists()
            assert len(json_str) > 0
            
            # Verify JSON structure
            import json
            loaded = json.loads(json_str)
            assert len(loaded) == 12
            assert '0' in loaded  # String keys in JSON
            assert 'boundaries' in loaded['0']


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance tests for Phase 1 modules"""
    
    @pytest.mark.benchmark
    def test_config_loading_performance(self, benchmark):
        """Benchmark configuration loading"""
        manager = AdaptiveRegimeConfigManager()
        
        def load_config():
            manager._load_defaults()
            return manager.config
        
        result = benchmark(load_config)
        assert result is not None
    
    @pytest.mark.benchmark
    def test_feature_extraction_performance(self, benchmark):
        """Benchmark feature extraction"""
        analyzer = HistoricalRegimeAnalyzer({'regime_count': 12})
        
        # Generate larger dataset
        dates = pd.date_range('2024-01-01', periods=10000, freq='5min')
        data = pd.DataFrame({
            'timestamp': dates,
            'price': 100 + np.random.randn(10000).cumsum() * 0.5,
            'volume': np.random.randint(1000, 10000, 10000)
        })
        
        prepared_data = analyzer._prepare_data(data)
        
        def extract_features():
            return analyzer._extract_features(prepared_data)
        
        features = benchmark(extract_features)
        assert features.shape[0] > 0
    
    @pytest.mark.benchmark  
    def test_clustering_performance(self, benchmark):
        """Benchmark clustering performance"""
        analyzer = HistoricalRegimeAnalyzer({
            'regime_count': 12,
            'clustering_algorithm': 'kmeans'
        })
        
        # Generate feature matrix
        features = np.random.randn(5000, 10)
        
        def perform_clustering():
            return analyzer._perform_clustering(features)
        
        labels = benchmark(perform_clustering)
        assert len(labels) == 5000


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])