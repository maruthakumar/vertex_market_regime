"""
Comprehensive Test Suite for Component 5: ATR-EMA-CPR Dual-Asset Analysis

Tests all aspects of Component 5 including dual-asset data extraction,
straddle and underlying analysis engines, DTE framework, cross-asset integration,
enhanced regime classification, and production performance compliance.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
import time
from unittest.mock import Mock, patch

# Component 5 imports
from vertex_market_regime.src.components.component_05_atr_ema_cpr import (
    Component05Analyzer,
    DualAssetDataExtractor,
    StraddleATREMACPREngine,
    UnderlyingATREMACPREngine,
    DualDTEFramework,
    CrossAssetIntegrationEngine,
    EnhancedRegimeClassificationEngine,
    Component05PerformanceMonitor,
    COMPONENT_INFO
)

# Base component imports
from vertex_market_regime.src.components.base_component import (
    ComponentAnalysisResult,
    FeatureVector,
    PerformanceFeedback,
    ComponentStatus
)


class TestComponentData:
    """Test data generator for Component 5 tests"""
    
    @staticmethod
    def create_production_sample_data(num_rows: int = 1000) -> pd.DataFrame:
        """Create sample production data matching 48-column schema"""
        
        np.random.seed(42)  # For reproducible tests
        
        # Generate timestamps
        start_date = datetime.now() - timedelta(days=30)
        timestamps = [start_date + timedelta(minutes=5*i) for i in range(num_rows)]
        
        # Generate DTE values (0-90)
        dte_values = np.random.choice(range(0, 91), size=num_rows)
        
        # Generate zone names
        zones = ['OPEN', 'MID_MORN', 'LUNCH', 'AFTERNOON', 'CLOSE']
        zone_names = np.random.choice(zones, size=num_rows)
        
        # Generate strike types
        strike_types = np.random.choice(['CE', 'PE'], size=num_rows)
        
        # Generate realistic option prices
        base_spot = 18000
        spot_prices = base_spot + np.random.normal(0, 200, num_rows).cumsum()
        
        # Generate strikes around spot
        strike_offsets = np.random.choice([-500, -250, 0, 250, 500], size=num_rows)
        strikes = spot_prices + strike_offsets
        
        # Generate option prices with realistic relationships
        ce_open = np.maximum(np.random.normal(200, 50, num_rows), 5)
        ce_high = ce_open * (1 + np.abs(np.random.normal(0, 0.02, num_rows)))
        ce_low = ce_open * (1 - np.abs(np.random.normal(0, 0.02, num_rows)))
        ce_close = ce_open + np.random.normal(0, 20, num_rows)
        
        pe_open = np.maximum(np.random.normal(180, 45, num_rows), 5)
        pe_high = pe_open * (1 + np.abs(np.random.normal(0, 0.02, num_rows)))
        pe_low = pe_open * (1 - np.abs(np.random.normal(0, 0.02, num_rows)))
        pe_close = pe_open + np.random.normal(0, 18, num_rows)
        
        # Generate volume and OI
        ce_volume = np.random.exponential(1000, num_rows)
        pe_volume = np.random.exponential(900, num_rows)
        ce_oi = np.random.exponential(5000, num_rows)
        pe_oi = np.random.exponential(4500, num_rows)
        
        # Generate future prices
        future_open = spot_prices + np.random.normal(0, 5, num_rows)
        future_high = future_open + np.abs(np.random.normal(0, 10, num_rows))
        future_low = future_open - np.abs(np.random.normal(0, 10, num_rows))
        future_close = future_open + np.random.normal(0, 8, num_rows)
        future_volume = np.random.exponential(50000, num_rows)
        future_oi = np.random.exponential(100000, num_rows)
        
        # Create DataFrame with 48-column production schema
        data = pd.DataFrame({
            'timestamp': timestamps,
            'dte': dte_values,
            'zone_name': zone_names,
            'strike_type': strike_types,
            'strike': strikes,
            'spot': spot_prices,
            
            # CE data
            'ce_open': ce_open,
            'ce_high': ce_high, 
            'ce_low': ce_low,
            'ce_close': ce_close,
            'ce_volume': ce_volume,
            'ce_oi': ce_oi,
            
            # PE data
            'pe_open': pe_open,
            'pe_high': pe_high,
            'pe_low': pe_low,
            'pe_close': pe_close,
            'pe_volume': pe_volume,
            'pe_oi': pe_oi,
            
            # Future data
            'future_open': future_open,
            'future_high': future_high,
            'future_low': future_low,
            'future_close': future_close,
            'future_volume': future_volume,
            'future_oi': future_oi,
            
            # Additional columns to match 48-column schema
            'iv': np.random.normal(0.25, 0.05, num_rows),
            'delta': np.random.normal(0.5, 0.3, num_rows),
            'gamma': np.random.exponential(0.001, num_rows),
            'theta': -np.random.exponential(5, num_rows),
            'vega': np.random.exponential(20, num_rows),
            'rho': np.random.normal(0, 10, num_rows),
        })
        
        # Add remaining columns to reach 48 columns
        for i in range(len(data.columns), 48):
            data[f'additional_col_{i}'] = np.random.normal(0, 1, num_rows)
        
        return data


class TestComponent05Analyzer:
    """Test suite for main Component 5 analyzer"""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for Component 5"""
        return {
            'component_id': 5,
            'feature_count': 94,
            'processing_budget_ms': 200,
            'memory_budget_mb': 500,
            'gpu_enabled': False,
            'learning_enabled': True,
            'fallback_enabled': True,
            'error_recovery_enabled': True,
            'project_id': 'test-project',
            'region': 'us-central1'
        }
    
    @pytest.fixture
    def sample_data(self):
        """Sample production data"""
        return TestComponentData.create_production_sample_data(500)
    
    @pytest.fixture
    def component_analyzer(self, sample_config):
        """Component 5 analyzer instance"""
        return Component05Analyzer(sample_config)
    
    def test_component_initialization(self, component_analyzer):
        """Test component initialization"""
        assert component_analyzer.component_id == 5
        assert component_analyzer.feature_count == 94
        assert component_analyzer.component_name == 'Component05Analyzer'
        assert hasattr(component_analyzer, 'data_extractor')
        assert hasattr(component_analyzer, 'straddle_engine')
        assert hasattr(component_analyzer, 'underlying_engine')
        assert hasattr(component_analyzer, 'dte_framework')
        assert hasattr(component_analyzer, 'cross_asset_engine')
        assert hasattr(component_analyzer, 'regime_classifier')
    
    @pytest.mark.asyncio
    async def test_analyze_complete_pipeline(self, component_analyzer, sample_data):
        """Test complete analysis pipeline"""
        result = await component_analyzer.analyze(sample_data)
        
        # Basic result validation
        assert isinstance(result, ComponentAnalysisResult)
        assert result.component_id == 5
        assert result.component_name == 'Component05Analyzer'
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        
        # Feature validation
        assert isinstance(result.features, FeatureVector)
        assert result.features.feature_count == 94
        assert len(result.features.feature_names) == 94
        assert result.features.features.shape[1] == 94
        
        # Performance validation
        assert result.processing_time_ms > 0
        assert 'performance_metrics' in result.metadata
        assert 'regime_classification' in result.metadata
        
    @pytest.mark.asyncio
    async def test_extract_features(self, component_analyzer, sample_data):
        """Test feature extraction"""
        features = await component_analyzer.extract_features(sample_data)
        
        assert isinstance(features, FeatureVector)
        assert features.feature_count == 94
        assert len(features.feature_names) == 94
        assert features.features.shape[1] == 94
        assert features.processing_time_ms > 0
        
        # Validate feature names structure
        feature_names = features.feature_names
        
        # Check straddle features (first 42)
        straddle_features = feature_names[:42]
        assert any('straddle_atr' in name for name in straddle_features)
        assert any('straddle_ema' in name for name in straddle_features)
        assert any('straddle_pivot' in name for name in straddle_features)
        
        # Check underlying features (next 36)
        underlying_features = feature_names[42:78]
        assert any('underlying' in name for name in underlying_features)
        
        # Check cross-asset features (last 16)
        cross_asset_features = feature_names[78:94]
        assert any('cross_asset' in name for name in cross_asset_features)
    
    @pytest.mark.asyncio
    async def test_performance_compliance(self, component_analyzer, sample_data):
        """Test performance budget compliance"""
        start_time = time.time()
        result = await component_analyzer.analyze(sample_data)
        processing_time = (time.time() - start_time) * 1000
        
        # Check processing time compliance (<200ms target)
        # Allow some tolerance for test environment
        assert processing_time < 1000  # 1 second tolerance for tests
        
        # Check performance metadata
        perf_metrics = result.metadata.get('performance_metrics')
        assert perf_metrics is not None
        assert hasattr(perf_metrics, 'performance_budget_compliant')
        assert hasattr(perf_metrics, 'memory_budget_compliant')
    
    @pytest.mark.asyncio 
    async def test_fallback_functionality(self, sample_config):
        """Test fallback functionality when analysis fails"""
        # Create analyzer with fallback enabled
        analyzer = Component05Analyzer(sample_config)
        
        # Create invalid data to trigger fallback
        invalid_data = pd.DataFrame({'invalid_column': [1, 2, 3]})
        
        result = await analyzer.analyze(invalid_data)
        
        # Should return fallback result, not raise exception
        assert isinstance(result, ComponentAnalysisResult)
        assert result.confidence < 0.5  # Low confidence for fallback
        assert 'fallback' in result.metadata.get('analysis_method', '')
    
    @pytest.mark.asyncio
    async def test_update_weights(self, component_analyzer):
        """Test weight update functionality"""
        performance_feedback = PerformanceFeedback(
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1_score=0.85,
            regime_specific_performance={'regime_0': 0.9, 'regime_1': 0.8},
            timestamp=datetime.utcnow()
        )
        
        weight_update = await component_analyzer.update_weights(performance_feedback)
        
        assert hasattr(weight_update, 'updated_weights')
        assert hasattr(weight_update, 'weight_changes')
        assert hasattr(weight_update, 'performance_improvement')
        assert 0.0 <= weight_update.confidence_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_health_check(self, component_analyzer):
        """Test component health check"""
        health_status = await component_analyzer.health_check()
        
        assert hasattr(health_status, 'component')
        assert hasattr(health_status, 'status')
        assert health_status.component == 'Component05Analyzer'
        assert health_status.feature_count == 94
        assert health_status.memory_usage_mb >= 0


class TestDualAssetDataExtractor:
    """Test suite for dual-asset data extractor"""
    
    @pytest.fixture
    def extractor_config(self):
        return {
            'straddle_method': 'atm_rolling',
            'rolling_window': 5
        }
    
    @pytest.fixture
    def data_extractor(self, extractor_config):
        return DualAssetDataExtractor(extractor_config)
    
    @pytest.fixture
    def sample_data(self):
        return TestComponentData.create_production_sample_data(200)
    
    @pytest.mark.asyncio
    async def test_dual_asset_extraction(self, data_extractor, sample_data):
        """Test dual-asset data extraction"""
        result = await data_extractor.extract_dual_asset_data(sample_data)
        
        # Validate extraction result
        assert hasattr(result, 'straddle_data')
        assert hasattr(result, 'underlying_data')
        assert result.extraction_time_ms > 0
        assert 0.0 <= result.data_quality_score <= 1.0
        
        # Validate straddle data
        straddle_data = result.straddle_data
        assert len(straddle_data.straddle_open) > 0
        assert len(straddle_data.straddle_close) == len(straddle_data.straddle_open)
        assert len(straddle_data.timestamps) == len(straddle_data.straddle_open)
        
        # Validate underlying data
        underlying_data = result.underlying_data
        assert len(underlying_data.spot_prices) > 0
        assert len(underlying_data.future_close) == len(underlying_data.spot_prices)
        assert 'daily' in underlying_data.timeframes
        assert 'weekly' in underlying_data.timeframes
        assert 'monthly' in underlying_data.timeframes
    
    def test_zone_coverage_calculation(self, data_extractor, sample_data):
        """Test zone coverage calculation"""
        zone_coverage = data_extractor._calculate_zone_coverage(sample_data)
        
        assert isinstance(zone_coverage, dict)
        assert len(zone_coverage) > 0
        # Should have coverage across multiple zones
        assert sum(zone_coverage.values()) == len(sample_data)
    
    def test_dte_coverage_calculation(self, data_extractor, sample_data):
        """Test DTE coverage calculation"""
        dte_coverage = data_extractor._calculate_dte_coverage(sample_data)
        
        assert isinstance(dte_coverage, dict)
        assert len(dte_coverage) > 0
        assert all(0 <= dte <= 90 for dte in dte_coverage.keys())


class TestStraddleEngine:
    """Test suite for straddle ATR-EMA-CPR engine"""
    
    @pytest.fixture
    def engine_config(self):
        return {'component_id': 5}
    
    @pytest.fixture
    def straddle_engine(self, engine_config):
        return StraddleATREMACPREngine(engine_config)
    
    @pytest.fixture
    def sample_straddle_data(self):
        """Create sample straddle price data"""
        from vertex_market_regime.src.components.component_05_atr_ema_cpr.dual_asset_data_extractor import StraddlePriceData
        
        n_points = 100
        return StraddlePriceData(
            straddle_open=np.random.normal(400, 50, n_points),
            straddle_high=np.random.normal(420, 55, n_points),
            straddle_low=np.random.normal(380, 45, n_points),
            straddle_close=np.random.normal(410, 52, n_points),
            straddle_volume=np.random.exponential(2000, n_points),
            straddle_oi=np.random.exponential(10000, n_points),
            timestamps=np.arange(n_points),
            dte_values=np.random.choice(range(0, 91), n_points),
            zone_names=['MID_MORN'] * n_points,
            metadata={'test_data': True}
        )
    
    @pytest.mark.asyncio
    async def test_straddle_analysis(self, straddle_engine, sample_straddle_data):
        """Test complete straddle analysis"""
        result = await straddle_engine.analyze_straddle_atr_ema_cpr(sample_straddle_data)
        
        # Validate result structure
        assert hasattr(result, 'atr_result')
        assert hasattr(result, 'ema_result')
        assert hasattr(result, 'cpr_result')
        assert hasattr(result, 'regime_classification')
        assert hasattr(result, 'confidence_scores')
        assert hasattr(result, 'feature_vector')
        
        # Validate processing time
        assert result.processing_time_ms > 0
        
        # Validate feature vector (should have 42 straddle features)
        assert result.feature_vector.shape[1] == 42
        
        # Validate regime classification
        assert len(result.regime_classification) == len(sample_straddle_data.straddle_close)
        assert all(0 <= regime <= 7 for regime in result.regime_classification)


class TestUnderlyingEngine:
    """Test suite for underlying ATR-EMA-CPR engine"""
    
    @pytest.fixture
    def engine_config(self):
        return {'component_id': 5}
    
    @pytest.fixture
    def underlying_engine(self, engine_config):
        return UnderlyingATREMACPREngine(engine_config)
    
    @pytest.fixture
    def sample_underlying_data(self):
        """Create sample underlying price data"""
        from vertex_market_regime.src.components.component_05_atr_ema_cpr.dual_asset_data_extractor import UnderlyingPriceData
        
        n_points = 100
        base_data = {
            'spot': np.random.normal(18000, 200, n_points).cumsum() + 18000,
            'open': np.random.normal(18000, 200, n_points),
            'high': np.random.normal(18100, 200, n_points), 
            'low': np.random.normal(17900, 200, n_points),
            'close': np.random.normal(18000, 200, n_points),
            'volume': np.random.exponential(50000, n_points),
            'oi': np.random.exponential(100000, n_points),
            'timestamps': np.arange(n_points),
            'atr_periods': np.array([14, 21, 50]),
            'ema_periods': np.array([20, 50, 100, 200])
        }
        
        timeframes = {
            'daily': base_data.copy(),
            'weekly': base_data.copy(),
            'monthly': base_data.copy()
        }
        
        return UnderlyingPriceData(
            spot_prices=base_data['spot'],
            future_open=base_data['open'],
            future_high=base_data['high'],
            future_low=base_data['low'],
            future_close=base_data['close'],
            future_volume=base_data['volume'],
            future_oi=base_data['oi'],
            timestamps=base_data['timestamps'],
            timeframes=timeframes,
            metadata={'test_data': True}
        )
    
    @pytest.mark.asyncio
    async def test_underlying_analysis(self, underlying_engine, sample_underlying_data):
        """Test multi-timeframe underlying analysis"""
        result = await underlying_engine.analyze_underlying_atr_ema_cpr(sample_underlying_data)
        
        # Validate result structure
        assert hasattr(result, 'atr_result')
        assert hasattr(result, 'ema_result') 
        assert hasattr(result, 'cpr_result')
        assert hasattr(result, 'combined_regime_classification')
        assert hasattr(result, 'cross_timeframe_confidence')
        
        # Validate multi-timeframe results
        assert 'daily' in result.atr_result.daily_atr
        assert 'weekly' in result.atr_result.weekly_atr
        assert 'monthly' in result.atr_result.monthly_atr
        
        # Validate feature vector (should have 36 underlying features)
        assert result.feature_vector.shape[1] == 36
        
        # Validate processing time
        assert result.processing_time_ms > 0


class TestCrossAssetIntegration:
    """Test suite for cross-asset integration"""
    
    @pytest.fixture
    def integration_config(self):
        return {
            'trend_agreement_threshold': 0.7,
            'max_validation_boost': 0.3,
            'max_conflict_penalty': 0.4
        }
    
    @pytest.fixture
    def integration_engine(self, integration_config):
        return CrossAssetIntegrationEngine(integration_config)
    
    @pytest.mark.asyncio
    async def test_cross_asset_integration(self, integration_engine):
        """Test cross-asset integration with mock results"""
        # Create mock analysis results
        from unittest.mock import MagicMock
        
        mock_straddle_result = MagicMock()
        mock_straddle_result.ema_result.trend_direction = np.array([1, -1, 0, 1])
        mock_straddle_result.ema_result.trend_strength = np.array([0.8, 0.7, 0.3, 0.9])
        mock_straddle_result.atr_result.volatility_regime = np.array([0, 1, 0, 2])
        mock_straddle_result.atr_result.atr_14 = np.array([0.02, 0.03, 0.025, 0.035])
        mock_straddle_result.cpr_result.pivot_points = {'standard': np.array([400, 410, 405, 415])}
        mock_straddle_result.confidence_scores = np.array([0.8, 0.7, 0.6, 0.9])
        mock_straddle_result.processing_time_ms = 50
        
        mock_underlying_result = MagicMock()
        mock_underlying_result.ema_result.trend_directions = {'daily': np.array([1, -1, 1, 1])}
        mock_underlying_result.ema_result.trend_strengths = {'daily': np.array([0.7, 0.8, 0.4, 0.8])}
        mock_underlying_result.atr_result.volatility_regimes = {'daily': np.array([0, 1, 1, 2])}
        mock_underlying_result.atr_result.daily_atr = {'atr_14': np.array([0.018, 0.028, 0.022, 0.032])}
        mock_underlying_result.cpr_result.daily_cpr = {'standard': {'pivot': np.array([18000, 18100, 18050, 18150])}}
        mock_underlying_result.cross_timeframe_confidence = np.array([0.75, 0.8, 0.65, 0.85])
        mock_underlying_result.processing_time_ms = 45
        
        mock_dte_result = MagicMock()
        mock_dte_result.integrated_confidence = np.array([0.8, 0.7, 0.6, 0.9])
        
        result = await integration_engine.integrate_cross_asset_analysis(
            mock_straddle_result, mock_underlying_result, mock_dte_result
        )
        
        # Validate integration result
        assert hasattr(result, 'trend_validation')
        assert hasattr(result, 'volatility_validation')
        assert hasattr(result, 'level_validation')
        assert hasattr(result, 'confidence_result')
        assert hasattr(result, 'weighting_result')
        assert hasattr(result, 'integrated_signals')
        
        # Validate processing time
        assert result.processing_time_ms > 0


class TestPerformanceMonitor:
    """Test suite for performance monitoring"""
    
    @pytest.fixture
    def monitor_config(self):
        return {
            'processing_budget_ms': 200,
            'memory_budget_mb': 500
        }
    
    @pytest.fixture
    def performance_monitor(self, monitor_config):
        return Component05PerformanceMonitor(monitor_config)
    
    def test_performance_monitoring(self, performance_monitor):
        """Test performance monitoring functionality"""
        performance_monitor.start_monitoring()
        
        # Record some stages
        performance_monitor.record_stage('data_extraction', 30)
        performance_monitor.record_stage('straddle_analysis', 45)
        performance_monitor.record_stage('underlying_analysis', 40)
        
        # Get metrics
        metrics = performance_monitor.get_performance_metrics()
        
        assert hasattr(metrics, 'total_processing_time_ms')
        assert hasattr(metrics, 'performance_budget_compliant')
        assert hasattr(metrics, 'memory_budget_compliant')
        assert hasattr(metrics, 'feature_extraction_efficiency')
        
        # Validate stage timings
        assert metrics.data_extraction_time_ms == 30
        assert metrics.straddle_analysis_time_ms == 45
        assert metrics.underlying_analysis_time_ms == 40


class TestComponentInfo:
    """Test suite for component metadata"""
    
    def test_component_info_structure(self):
        """Test component info metadata"""
        assert COMPONENT_INFO['id'] == 5
        assert COMPONENT_INFO['features'] == 94
        assert COMPONENT_INFO['performance_budget_ms'] == 200
        assert COMPONENT_INFO['memory_budget_mb'] == 500
        assert COMPONENT_INFO['regimes'] == 8
        
        # Validate capability list
        assert 'Dual-asset analysis (straddle + underlying)' in COMPONENT_INFO['key_capabilities']
        assert 'Multi-timeframe underlying analysis (daily/weekly/monthly)' in COMPONENT_INFO['key_capabilities']
        assert 'Production performance optimization' in COMPONENT_INFO['key_capabilities']


class TestIntegrationScenarios:
    """Test suite for integration scenarios"""
    
    @pytest.fixture
    def full_config(self):
        return {
            'component_id': 5,
            'feature_count': 94,
            'processing_budget_ms': 1000,  # More lenient for tests
            'memory_budget_mb': 1000,
            'gpu_enabled': False,
            'learning_enabled': True,
            'fallback_enabled': True
        }
    
    @pytest.fixture
    def full_analyzer(self, full_config):
        return Component05Analyzer(full_config)
    
    @pytest.mark.asyncio
    async def test_end_to_end_integration(self, full_analyzer):
        """Test complete end-to-end integration"""
        # Create comprehensive test data
        test_data = TestComponentData.create_production_sample_data(300)
        
        # Run complete analysis
        result = await full_analyzer.analyze(test_data)
        
        # Comprehensive validation
        assert isinstance(result, ComponentAnalysisResult)
        assert result.component_id == 5
        assert result.features.feature_count == 94
        assert len(result.features.feature_names) == 94
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        
        # Validate metadata completeness
        metadata = result.metadata
        assert 'regime_classification' in metadata
        assert 'performance_metrics' in metadata
        assert 'data_quality_score' in metadata
        assert 'cross_asset_validation' in metadata
        
        # Validate cross-asset validation metrics
        cross_asset = metadata['cross_asset_validation']
        assert 'trend_agreement' in cross_asset
        assert 'volatility_agreement' in cross_asset
        assert 'level_agreement' in cross_asset
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, full_analyzer):
        """Test error handling and recovery mechanisms"""
        # Test with malformed data
        malformed_data = pd.DataFrame({
            'bad_column': [1, 2, 3],
            'another_bad': ['a', 'b', 'c']
        })
        
        # Should not raise exception due to fallback
        result = await full_analyzer.analyze(malformed_data)
        
        assert isinstance(result, ComponentAnalysisResult)
        assert result.confidence < 0.5  # Should have low confidence
        assert 'fallback' in str(result.metadata.get('analysis_method', ''))
    
    def test_memory_usage_tracking(self, full_analyzer):
        """Test memory usage tracking"""
        monitor = full_analyzer.performance_monitor
        
        # Memory should be trackable
        initial_memory = monitor._get_current_memory()
        assert initial_memory >= 0
        
        # Performance monitoring should work
        monitor.start_monitoring()
        monitor.record_stage('test_stage', 10)
        metrics = monitor.get_performance_metrics()
        
        assert metrics.memory_usage_mb >= 0
        assert metrics.peak_memory_mb >= initial_memory


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])