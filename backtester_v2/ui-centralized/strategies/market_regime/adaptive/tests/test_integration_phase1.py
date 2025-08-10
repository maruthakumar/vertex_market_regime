"""
Integration Tests for Phase 1 Core Infrastructure

End-to-end testing of:
- Configuration loading to analysis
- Analysis to regime definition
- Complete pipeline validation
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json
from datetime import datetime, timedelta

# Import modules
import sys
sys.path.append('/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/market_regime')

from adaptive.config.adaptive_regime_config_manager import AdaptiveRegimeConfigManager
from adaptive.analysis.historical_regime_analyzer import HistoricalRegimeAnalyzer
from adaptive.core.regime_definition_builder import RegimeDefinitionBuilder


class TestPhase1Integration:
    """Integration tests for Phase 1 modules"""
    
    @pytest.fixture
    def realistic_market_data(self):
        """Generate realistic market data for testing"""
        np.random.seed(42)
        
        # Generate 30 days of 5-minute data
        dates = pd.date_range('2024-01-01', periods=8640, freq='5min')  # 30 days
        
        # Create different market regimes
        data_points = []
        
        # Define regime periods
        regime_periods = [
            (0, 2000, 'low_vol_bullish'),      # Days 1-7
            (2000, 4000, 'high_vol_neutral'),   # Days 7-14
            (4000, 6000, 'med_vol_bearish'),    # Days 14-21
            (6000, 8640, 'low_vol_range')       # Days 21-30
        ]
        
        price = 100.0
        
        for start, end, regime_type in regime_periods:
            for i in range(start, end):
                # Regime-specific characteristics
                if regime_type == 'low_vol_bullish':
                    trend = 0.0002
                    volatility = 0.001
                    volume_mult = 1.0
                elif regime_type == 'high_vol_neutral':
                    trend = 0.0
                    volatility = 0.005
                    volume_mult = 1.5
                elif regime_type == 'med_vol_bearish':
                    trend = -0.0001
                    volatility = 0.003
                    volume_mult = 1.2
                else:  # low_vol_range
                    trend = 0.0
                    volatility = 0.0015
                    volume_mult = 0.8
                
                # Generate price with trend and volatility
                price_change = trend + np.random.randn() * volatility
                price = price * (1 + price_change)
                
                # Generate volume
                base_volume = 5000
                volume = int(base_volume * volume_mult * (1 + np.random.randn() * 0.2))
                
                data_points.append({
                    'timestamp': dates[i],
                    'price': price,
                    'volume': volume
                })
        
        return pd.DataFrame(data_points)
    
    def test_config_to_analysis_pipeline(self, realistic_market_data):
        """Test configuration driving analysis"""
        # Step 1: Create and load configuration
        config_manager = AdaptiveRegimeConfigManager()
        
        # Test with different regime counts
        for regime_count in [8, 12, 18]:
            config_dict = config_manager.DEFAULT_CONFIG.copy()
            config_dict['regime_count'] = regime_count
            config_dict['historical_lookback_days'] = 30
            
            # Step 2: Initialize analyzer with config
            analyzer = HistoricalRegimeAnalyzer(config_dict)
            
            # Step 3: Run analysis
            results = analyzer.analyze_historical_patterns(realistic_market_data)
            
            # Verify results respect configuration
            assert 'regime_patterns' in results
            patterns = results['regime_patterns']
            assert len(patterns) <= regime_count
            
            # Check transition matrix size
            transition_matrix = results['transition_matrix']
            assert transition_matrix.shape == (regime_count, regime_count)
    
    def test_analysis_to_definition_pipeline(self, realistic_market_data):
        """Test analysis results to regime definitions"""
        # Step 1: Run analysis
        config = {
            'regime_count': 12,
            'historical_lookback_days': 30,
            'intraday_window': '5min'
        }
        
        analyzer = HistoricalRegimeAnalyzer(config)
        analysis_results = analyzer.analyze_historical_patterns(realistic_market_data)
        
        # Step 2: Build definitions from analysis
        builder = RegimeDefinitionBuilder(regime_count=12)
        definitions = builder.build_regime_definitions(analysis_results)
        
        # Verify definitions match analysis
        assert len(definitions) == 12
        
        # Check each definition has required components
        for regime_id, definition in definitions.items():
            assert definition.regime_id == regime_id
            assert definition.boundaries is not None
            assert len(definition.characteristic_features) >= 0
            assert len(definition.strategy_preferences) > 0
            assert definition.risk_parameters['position_size_multiplier'] > 0
    
    def test_complete_pipeline_with_profiles(self, realistic_market_data):
        """Test complete pipeline with different profiles"""
        profiles = ['conservative', 'balanced', 'aggressive']
        
        for profile in profiles:
            # Step 1: Configuration with profile
            config_manager = AdaptiveRegimeConfigManager()
            config = config_manager.DEFAULT_CONFIG.copy()
            config.update(config_manager.PROFILE_PRESETS[profile])
            config['profile_name'] = profile
            
            # Step 2: Analysis
            analyzer = HistoricalRegimeAnalyzer(config)
            analysis_results = analyzer.analyze_historical_patterns(realistic_market_data)
            
            # Step 3: Definition building
            builder = RegimeDefinitionBuilder(config['regime_count'])
            definitions = builder.build_regime_definitions(analysis_results)
            
            # Verify profile affects results
            if profile == 'conservative':
                # Conservative should have higher confidence thresholds
                avg_confidence = np.mean([
                    d.boundaries.confidence_threshold 
                    for d in definitions.values()
                ])
                assert avg_confidence >= 0.6
            
            elif profile == 'aggressive':
                # Aggressive should allow more regimes
                stability = analysis_results['stability_metrics']
                assert stability['transition_rate'] <= 0.2  # Still reasonable
    
    def test_regime_detection_accuracy(self, realistic_market_data):
        """Test regime detection on known data patterns"""
        # Configure for 8 regimes (simpler for testing)
        config = {
            'regime_count': 8,
            'historical_lookback_days': 30,
            'clustering_algorithm': 'kmeans'
        }
        
        # Run full pipeline
        analyzer = HistoricalRegimeAnalyzer(config)
        analysis_results = analyzer.analyze_historical_patterns(realistic_market_data)
        
        # Check clustering quality
        quality = analysis_results['cluster_quality']
        assert 'silhouette_score' in quality
        assert quality['silhouette_score'] > 0.2  # Reasonable separation
        
        # Check regime stability
        stability = analysis_results['stability_metrics']
        assert stability['average_regime_duration'] > 100  # At least 100 periods
        assert stability['regime_persistence'] > 0.8  # 80% persistence
    
    def test_configuration_reload_and_update(self):
        """Test configuration reload functionality"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.xlsx"
            
            # Create initial configuration
            manager = AdaptiveRegimeConfigManager()
            manager.generate_template(str(config_path))
            
            # Load and modify
            manager2 = AdaptiveRegimeConfigManager(str(config_path))
            original_config = manager2.load_configuration()
            
            # Modify and save
            modified_config = manager2.config
            modified_config.regime_count = 18
            modified_config.transition_sensitivity = 0.8
            manager2.save_configuration(modified_config, str(config_path))
            
            # Reload and verify
            manager3 = AdaptiveRegimeConfigManager(str(config_path))
            reloaded_config = manager3.load_configuration()
            
            assert reloaded_config.regime_count == 18
            assert reloaded_config.transition_sensitivity == 0.8
    
    def test_error_handling_and_recovery(self):
        """Test error handling in pipeline"""
        # Test with invalid data
        invalid_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='5min'),
            'price': [np.nan] * 10,  # All NaN
            'volume': [1000] * 10
        })
        
        config = {'regime_count': 12}
        analyzer = HistoricalRegimeAnalyzer(config)
        
        # Should handle gracefully
        with pytest.raises(Exception):
            analyzer.analyze_historical_patterns(invalid_data)
        
        # Test with insufficient data
        small_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=50, freq='5min'),
            'price': 100 + np.random.randn(50).cumsum() * 0.5,
            'volume': np.random.randint(1000, 2000, 50)
        })
        
        # Should still work but with warnings
        results = analyzer.analyze_historical_patterns(small_data)
        assert 'regime_patterns' in results
    
    def test_export_import_definitions(self, realistic_market_data):
        """Test exporting and importing regime definitions"""
        # Run full pipeline
        config = {'regime_count': 12}
        analyzer = HistoricalRegimeAnalyzer(config)
        analysis_results = analyzer.analyze_historical_patterns(realistic_market_data)
        
        builder = RegimeDefinitionBuilder(regime_count=12)
        definitions = builder.build_regime_definitions(analysis_results)
        
        # Export to JSON
        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "definitions.json"
            json_str = builder.export_definitions(str(export_path))
            
            # Verify JSON structure
            with open(export_path, 'r') as f:
                loaded_defs = json.load(f)
            
            assert len(loaded_defs) == 12
            
            # Check each definition
            for regime_id_str, def_data in loaded_defs.items():
                regime_id = int(regime_id_str)
                assert 'name' in def_data
                assert 'boundaries' in def_data
                assert 'strategy_preferences' in def_data
                assert 'risk_parameters' in def_data
    
    def test_performance_metrics(self, realistic_market_data):
        """Test performance meets requirements"""
        import time
        
        config = {
            'regime_count': 12,
            'historical_lookback_days': 30
        }
        
        # Measure analysis time
        analyzer = HistoricalRegimeAnalyzer(config)
        
        start_time = time.time()
        results = analyzer.analyze_historical_patterns(realistic_market_data)
        analysis_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert analysis_time < 10.0  # 10 seconds for 30 days of data
        
        # Measure definition building time
        builder = RegimeDefinitionBuilder(regime_count=12)
        
        start_time = time.time()
        definitions = builder.build_regime_definitions(results)
        build_time = time.time() - start_time
        
        assert build_time < 1.0  # Should be fast
        
        print(f"\nPerformance Metrics:")
        print(f"Analysis time: {analysis_time:.3f}s")
        print(f"Definition build time: {build_time:.3f}s")
        print(f"Data points processed: {len(realistic_market_data)}")
        print(f"Processing rate: {len(realistic_market_data)/analysis_time:.0f} points/sec")
    
    def test_regime_summary_generation(self, realistic_market_data):
        """Test summary generation for reporting"""
        # Run pipeline
        config = {'regime_count': 8}
        analyzer = HistoricalRegimeAnalyzer(config)
        analysis_results = analyzer.analyze_historical_patterns(realistic_market_data)
        
        builder = RegimeDefinitionBuilder(regime_count=8)
        definitions = builder.build_regime_definitions(analysis_results)
        
        # Get summaries
        analysis_summary = analyzer.get_regime_summary()
        definition_summary = builder.get_regime_summary()
        
        # Verify summary contents
        assert len(analysis_summary) <= 8
        assert len(definition_summary) == 8
        
        # Check summary columns
        assert 'stability_score' in analysis_summary.columns
        assert 'preferred_strategies' in definition_summary.columns
        
        print("\nRegime Analysis Summary:")
        print(analysis_summary)
        print("\nRegime Definition Summary:")
        print(definition_summary)


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_minimum_data_requirements(self):
        """Test with minimum viable data"""
        # Create minimal dataset
        min_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='5min'),
            'price': 100 + np.random.randn(100).cumsum() * 0.5,
            'volume': np.random.randint(1000, 2000, 100)
        })
        
        config = {'regime_count': 8}  # Fewer regimes for small data
        analyzer = HistoricalRegimeAnalyzer(config)
        
        # Should work but with limited regimes
        results = analyzer.analyze_historical_patterns(min_data)
        patterns = results['regime_patterns']
        
        # May have fewer regimes than requested
        assert len(patterns) <= 8
    
    def test_extreme_market_conditions(self):
        """Test with extreme market scenarios"""
        # Create extreme volatility data
        dates = pd.date_range('2024-01-01', periods=1000, freq='5min')
        
        # Extreme volatility period
        prices = [100]
        for i in range(1, 1000):
            # Large random moves
            change = np.random.randn() * 0.05  # 5% moves
            prices.append(prices[-1] * (1 + change))
        
        extreme_data = pd.DataFrame({
            'timestamp': dates,
            'price': prices,
            'volume': np.random.randint(100, 10000, 1000)  # High volume variance
        })
        
        config = {'regime_count': 12}
        analyzer = HistoricalRegimeAnalyzer(config)
        
        # Should handle without crashing
        results = analyzer.analyze_historical_patterns(extreme_data)
        
        # Check identifies high volatility
        patterns = results['regime_patterns']
        high_vol_regimes = [
            p for p in patterns.values() 
            if 'high_volatility' in p.characteristic_features
        ]
        assert len(high_vol_regimes) > 0
    
    def test_missing_features(self):
        """Test handling of missing optional features"""
        # Create data with only required columns
        basic_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=500, freq='5min'),
            'price': 100 + np.random.randn(500).cumsum() * 0.5,
            'volume': np.random.randint(1000, 2000, 500)
        })
        
        config = {'regime_count': 8}
        analyzer = HistoricalRegimeAnalyzer(config)
        
        # Should create features from basic data
        results = analyzer.analyze_historical_patterns(basic_data)
        assert 'regime_patterns' in results
    
    def test_regime_count_edge_cases(self):
        """Test extreme regime counts"""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=2000, freq='5min'),
            'price': 100 + np.random.randn(2000).cumsum() * 0.5,
            'volume': np.random.randint(1000, 5000, 2000)
        })
        
        # Test each supported regime count
        for regime_count in [8, 12, 18]:
            config = {'regime_count': regime_count}
            analyzer = HistoricalRegimeAnalyzer(config)
            results = analyzer.analyze_historical_patterns(data)
            
            builder = RegimeDefinitionBuilder(regime_count)
            definitions = builder.build_regime_definitions(results)
            
            # Should create exactly the requested number
            assert len(definitions) == regime_count


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])