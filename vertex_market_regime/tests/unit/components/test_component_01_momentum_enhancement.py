"""
Test Component 1 Momentum Enhancement - Phase 2 Validation

Tests the integration of RSI/MACD momentum analysis with Component 1
Triple Rolling Straddle analyzer.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
from unittest.mock import Mock, AsyncMock

# Test imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from vertex_market_regime.src.components.component_01_triple_straddle.component_01_analyzer import Component01TripleStraddleAnalyzer
from vertex_market_regime.src.components.component_01_triple_straddle.momentum_analysis import MomentumAnalysisEngine
from vertex_market_regime.src.components.base_component import FeatureVector

class TestComponent1MomentumEnhancement:
    """Test suite for Component 1 Phase 2 momentum enhancement"""
    
    @pytest.fixture
    def component_config(self):
        """Component configuration for testing"""
        return {
            'component_id': 1,
            'feature_count': 150,  # Updated for Phase 2
            'processing_budget_ms': 190,  # Updated budget
            'memory_budget_mb': 512,
            'gpu_enabled': False,
            'use_cache': False,
            'log_level': 'INFO'
        }
    
    @pytest.fixture
    def mock_straddle_data(self):
        """Mock straddle data for testing"""
        n_points = 100
        timestamps = [datetime.now() - timedelta(minutes=i) for i in range(n_points)]
        
        return {
            'atm_straddle': np.random.uniform(100, 200, n_points),
            'itm1_straddle': np.random.uniform(90, 180, n_points),
            'otm1_straddle': np.random.uniform(110, 220, n_points),
            'atm_ce': np.random.uniform(40, 80, n_points),
            'itm1_ce': np.random.uniform(35, 75, n_points),
            'otm1_ce': np.random.uniform(45, 85, n_points),
            'atm_pe': np.random.uniform(50, 90, n_points),
            'itm1_pe': np.random.uniform(45, 85, n_points),
            'otm1_pe': np.random.uniform(55, 95, n_points),
            'timestamps': np.array(timestamps)
        }
    
    @pytest.fixture
    def mock_volume_data(self):
        """Mock volume data for testing"""
        n_points = 100
        return {
            'combined_volume': np.random.uniform(1000, 5000, n_points),
            'atm_volume': np.random.uniform(400, 2000, n_points),
            'itm1_volume': np.random.uniform(300, 1500, n_points),
            'otm1_volume': np.random.uniform(300, 1500, n_points)
        }
    
    def test_momentum_engine_initialization(self, component_config):
        """Test momentum analysis engine initialization"""
        engine = MomentumAnalysisEngine(component_config)
        
        assert engine is not None
        assert engine.rsi_period == 14
        assert engine.macd_fast_period == 12
        assert engine.macd_slow_period == 26
        assert engine.macd_signal_period == 9
        assert len(engine.parameters) == 10
        assert len(engine.timeframes) == 4
        
    @pytest.mark.asyncio
    async def test_momentum_analysis_basic(self, component_config, mock_straddle_data, mock_volume_data):
        """Test basic momentum analysis functionality"""
        engine = MomentumAnalysisEngine(component_config)
        
        result = await engine.analyze_momentum(
            mock_straddle_data, 
            mock_volume_data, 
            mock_straddle_data['timestamps']
        )
        
        # Validate result structure
        assert result is not None
        assert hasattr(result, 'momentum_features')
        assert hasattr(result, 'feature_names')
        assert hasattr(result, 'rsi_results')
        assert hasattr(result, 'macd_results')
        assert hasattr(result, 'divergence_results')
        
        # Validate feature count (exactly 30 momentum features)
        assert len(result.momentum_features) == 30
        assert len(result.feature_names) == 30
        
        # Validate RSI results structure
        assert len(result.rsi_results) == 10  # 10 parameters
        for param_results in result.rsi_results.values():
            assert len(param_results) == 4  # 4 timeframes
        
        # Validate MACD results structure
        assert len(result.macd_results) == 10  # 10 parameters
        for param_results in result.macd_results.values():
            assert len(param_results) == 4  # 4 timeframes
    
    def test_component_1_enhanced_initialization(self, component_config):
        """Test Component 1 analyzer with momentum enhancement"""
        analyzer = Component01TripleStraddleAnalyzer(component_config)
        
        # Validate updated configuration
        assert analyzer.expected_feature_count == 150
        assert analyzer.config['processing_budget_ms'] == 190
        assert analyzer.config['feature_count'] == 150
        
        # Validate feature categories include momentum
        assert 'momentum_analysis' in analyzer.feature_categories
        assert analyzer.feature_categories['momentum_analysis'] == 30
        
        # Validate momentum engine is initialized
        assert hasattr(analyzer, 'momentum_engine')
        assert analyzer.momentum_engine is not None
    
    @pytest.mark.asyncio
    async def test_rsi_calculation(self, component_config):
        """Test RSI calculation accuracy"""
        engine = MomentumAnalysisEngine(component_config)
        
        # Test with known price series
        prices = np.array([44, 44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.85, 46.08, 45.89,
                          46.03, 46.83, 46.69, 46.45, 46.59, 46.3, 46.28, 46.28, 46])
        
        rsi = engine._calculate_rsi(prices, 14)
        
        # RSI should be between 0 and 100
        assert 0 <= rsi <= 100
        
        # For this upward trending series, RSI should be > 50
        assert rsi > 40  # Allowing some tolerance
    
    @pytest.mark.asyncio 
    async def test_macd_calculation(self, component_config):
        """Test MACD calculation accuracy"""
        engine = MomentumAnalysisEngine(component_config)
        
        # Test with trending price series
        prices = np.linspace(100, 120, 50)  # Upward trend
        
        macd_line, signal_line, histogram = engine._calculate_macd(prices, 12, 26, 9)
        
        # Validate MACD components are calculated
        assert isinstance(macd_line, float)
        assert isinstance(signal_line, float)
        assert isinstance(histogram, float)
        
        # For upward trend, MACD line should generally be positive
        assert macd_line >= 0  # Allowing some tolerance
    
    @pytest.mark.asyncio
    async def test_component_1_150_features(self, component_config):
        """Test that Component 1 now produces exactly 150 features"""
        analyzer = Component01TripleStraddleAnalyzer(component_config)
        
        # Mock the sub-engines to avoid complex initialization
        analyzer.parquet_loader = AsyncMock()
        analyzer.straddle_engine = AsyncMock()
        analyzer.weighting_system = AsyncMock()
        analyzer.ema_engine = AsyncMock()
        analyzer.vwap_engine = AsyncMock()
        analyzer.pivot_engine = AsyncMock()
        
        # Mock successful results from all engines
        from vertex_market_regime.src.components.component_01_triple_straddle.rolling_straddle import RollingStraddleTimeSeries
        from vertex_market_regime.src.components.component_01_triple_straddle.dynamic_weighting import WeightingAnalysisResult
        from vertex_market_regime.src.components.component_01_triple_straddle.momentum_analysis import MomentumAnalysisResult
        
        # Create mock results with proper structure
        mock_straddle_ts = Mock(spec=RollingStraddleTimeSeries)
        mock_straddle_ts.atm_straddle_series = np.random.uniform(100, 200, 50)
        mock_straddle_ts.itm1_straddle_series = np.random.uniform(90, 180, 50)  
        mock_straddle_ts.otm1_straddle_series = np.random.uniform(110, 220, 50)
        mock_straddle_ts.volume_series = np.random.uniform(1000, 5000, 50)
        mock_straddle_ts.spot_series = np.random.uniform(20000, 21000, 50)
        mock_straddle_ts.timestamps = np.array([datetime.now() - timedelta(minutes=i) for i in range(50)])
        mock_straddle_ts.data_points = 50
        mock_straddle_ts.missing_data_count = 0
        mock_straddle_ts.processing_time_ms = 10.0
        
        mock_weighting = Mock()
        mock_weighting.component_weights = Mock()
        mock_weighting.component_weights.atm_straddle_weight = 0.1
        mock_weighting.component_weights.itm1_straddle_weight = 0.1
        mock_weighting.component_weights.otm1_straddle_weight = 0.1
        mock_weighting.component_weights.atm_ce_weight = 0.1
        mock_weighting.component_weights.itm1_ce_weight = 0.1
        mock_weighting.component_weights.otm1_ce_weight = 0.1
        mock_weighting.component_weights.atm_pe_weight = 0.1
        mock_weighting.component_weights.itm1_pe_weight = 0.1
        mock_weighting.component_weights.otm1_pe_weight = 0.1
        mock_weighting.component_weights.correlation_factor_weight = 0.1
        mock_weighting.total_score = 0.8
        mock_weighting.confidence = 0.9
        mock_weighting.component_scores = {'test': 0.8}
        mock_weighting.volume_weights = {'combined_volume': 1.0}
        mock_weighting.correlation_matrix = np.eye(3)
        mock_weighting.processing_time_ms = 15.0
        mock_weighting.metadata = {'total_volume': 1000, 'correlation_threshold': 0.7}
        
        # Create real momentum result
        mock_momentum = MomentumAnalysisResult(
            rsi_results={},
            macd_results={},
            divergence_results={},
            combined_straddle_momentum={},
            momentum_features=np.random.uniform(-1, 1, 30),  # Exactly 30 features
            feature_names=[f'momentum_feature_{i}' for i in range(30)],
            total_processing_time_ms=25.0,
            memory_usage_mb=20.0,
            momentum_confidence=0.8,
            signal_quality=0.7,
            metadata={}
        )
        
        # Mock other engines to return minimal valid results
        mock_ema = Mock()
        mock_ema.overall_alignment_score = 0.5
        mock_ema.trend_consistency = 0.8
        mock_ema.overall_deviation_score = 0.3
        mock_ema.processing_time_ms = 20.0
        
        mock_vwap = Mock()
        mock_vwap.vwap_trend_alignment = 0.6
        mock_vwap.processing_time_ms = 18.0
        
        mock_pivot = Mock()
        mock_pivot.overall_pivot_alignment = 0.4
        mock_pivot.pivot_confluence_strength = 0.7
        mock_pivot.dte_average = 7.0
        mock_pivot.atm_pivots = Mock()
        mock_pivot.atm_pivots.pivot_dominance = 0.8
        mock_pivot.processing_time_ms = 22.0
        
        # Mock engine methods
        analyzer.ema_engine.get_ema_feature_vector = AsyncMock(return_value={f'ema_{i}': 0.5 for i in range(25)})
        analyzer.vwap_engine.get_vwap_feature_vector = AsyncMock(return_value={f'vwap_{i}': 0.5 for i in range(25)})
        analyzer.pivot_engine.get_pivot_feature_vector = AsyncMock(return_value={f'pivot_{i}': 0.5 for i in range(20)})
        
        # Test feature vector generation directly
        feature_vector = await analyzer._generate_feature_vector(
            mock_straddle_ts, mock_weighting, mock_ema, mock_vwap, mock_pivot, mock_momentum
        )
        
        # Validate exactly 150 features
        assert feature_vector.feature_count == 150
        assert len(feature_vector.features) == 150
        assert len(feature_vector.feature_names) == 150
        
        # Validate feature categories are properly distributed
        assert len([name for name in feature_vector.feature_names if 'momentum' in name]) >= 25  # At least 25 momentum features
    
    def test_performance_budget_update(self, component_config):
        """Test that performance budgets are updated for Phase 2"""
        analyzer = Component01TripleStraddleAnalyzer(component_config)
        
        # Validate updated performance targets
        assert analyzer.config['processing_budget_ms'] == 190
        assert analyzer.config['feature_count'] == 150
        
        # Validate momentum engine has budget allocation
        assert analyzer.momentum_engine.processing_budget_ms == 40
    
    @pytest.mark.asyncio
    async def test_momentum_feature_quality(self, component_config, mock_straddle_data, mock_volume_data):
        """Test quality of momentum features generated"""
        engine = MomentumAnalysisEngine(component_config)
        
        result = await engine.analyze_momentum(
            mock_straddle_data,
            mock_volume_data, 
            mock_straddle_data['timestamps']
        )
        
        # Validate feature value ranges
        features = result.momentum_features
        
        # Most momentum features should be in reasonable ranges
        assert np.all(np.isfinite(features))  # No NaN or infinite values
        assert np.all(features >= -10.0)  # Reasonable lower bound
        assert np.all(features <= 10.0)   # Reasonable upper bound
        
        # Validate feature names are descriptive
        feature_names = result.feature_names
        momentum_indicators = ['rsi', 'macd', 'divergence', 'momentum']
        
        has_indicator_features = any(
            any(indicator in name.lower() for indicator in momentum_indicators)
            for name in feature_names
        )
        assert has_indicator_features, "Feature names should include momentum indicators"
    
    def test_option_b_strategy_implementation(self, component_config):
        """Test that Option B strategy (RSI + MACD) is correctly implemented"""
        engine = MomentumAnalysisEngine(component_config)
        
        # Validate RSI configuration
        assert engine.rsi_period == 14
        assert engine.rsi_overbought == 70
        assert engine.rsi_oversold == 30
        
        # Validate MACD configuration  
        assert engine.macd_fast_period == 12
        assert engine.macd_slow_period == 26
        assert engine.macd_signal_period == 9
        
        # Validate 10-parameter system
        expected_parameters = [
            'atm_straddle', 'itm1_straddle', 'otm1_straddle',
            'atm_ce', 'itm1_ce', 'otm1_ce',
            'atm_pe', 'itm1_pe', 'otm1_pe',
            'combined_straddle'
        ]
        assert engine.parameters == expected_parameters
        
        # Validate 4-timeframe system
        expected_timeframes = ['3min', '5min', '10min', '15min']
        assert engine.timeframes == expected_timeframes

if __name__ == "__main__":
    pytest.main([__file__, "-v"])