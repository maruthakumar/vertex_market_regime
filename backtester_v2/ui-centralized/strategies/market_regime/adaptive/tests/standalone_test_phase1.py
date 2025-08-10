"""
Standalone Test Script for Phase 1 Modules

Run this script directly to test Phase 1 implementation
without dependency issues.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json
from datetime import datetime, timedelta

# Add parent directories to path
current_dir = Path(__file__).parent
adaptive_dir = current_dir.parent
sys.path.insert(0, str(adaptive_dir))

# Import modules
from config.adaptive_regime_config_manager import AdaptiveRegimeConfigManager
from analysis.historical_regime_analyzer import HistoricalRegimeAnalyzer
from core.regime_definition_builder import RegimeDefinitionBuilder


def test_config_manager():
    """Test configuration manager"""
    print("\n=== Testing Configuration Manager ===")
    
    # Test 1: Default configuration
    print("\n1. Testing default configuration...")
    manager = AdaptiveRegimeConfigManager()
    assert manager.config is not None
    assert manager.config.regime_count == 12
    print("✓ Default configuration loaded successfully")
    
    # Test 2: Template generation
    print("\n2. Testing template generation...")
    with tempfile.TemporaryDirectory() as tmpdir:
        template_path = Path(tmpdir) / "test_template.xlsx"
        result_path = manager.generate_template(str(template_path))
        assert result_path.exists()
        print(f"✓ Template generated at {result_path}")
    
    # Test 3: Parameter validation
    print("\n3. Testing parameter validation...")
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
    print("✓ Valid parameters accepted")
    
    # Test invalid parameters
    invalid_params = valid_params.copy()
    invalid_params['regime_count'] = 10  # Invalid
    assert manager.validate_parameters(invalid_params) == False
    print("✓ Invalid parameters rejected")
    
    # Test 4: Regime-specific configuration
    print("\n4. Testing regime-specific configurations...")
    config_8 = manager.get_regime_specific_config(8)
    assert config_8['min_regime_duration'] == 20
    print(f"✓ 8-regime config: min_duration={config_8['min_regime_duration']}")
    
    config_18 = manager.get_regime_specific_config(18)
    assert config_18['min_regime_duration'] == 10
    print(f"✓ 18-regime config: min_duration={config_18['min_regime_duration']}")
    
    # Test 5: Configuration summary
    print("\n5. Testing configuration summary export...")
    summary = manager.export_config_summary()
    assert 'regime_count' in summary
    assert summary['regime_count'] == 12
    print("✓ Configuration summary exported successfully")
    
    return True


def test_historical_analyzer():
    """Test historical regime analyzer"""
    print("\n=== Testing Historical Regime Analyzer ===")
    
    # Generate sample data
    print("\n1. Generating sample market data...")
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=2000, freq='5min')
    
    prices = [100]
    for i in range(1, 2000):
        change = np.random.randn() * 0.002 + 0.0001  # Small positive drift
        prices.append(prices[-1] * (1 + change))
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'price': prices,
        'volume': np.random.randint(3000, 7000, 2000)
    })
    print(f"✓ Generated {len(sample_data)} data points")
    
    # Test 2: Initialize analyzer
    print("\n2. Initializing analyzer...")
    config = {
        'historical_lookback_days': 30,
        'regime_count': 8,
        'intraday_window': '5min',
        'clustering_algorithm': 'kmeans'
    }
    analyzer = HistoricalRegimeAnalyzer(config)
    print("✓ Analyzer initialized")
    
    # Test 3: Data preparation
    print("\n3. Testing data preparation...")
    prepared_data = analyzer._prepare_data(sample_data)
    assert 'returns' in prepared_data.columns
    assert 'volatility_20' in prepared_data.columns
    assert prepared_data.isna().sum().sum() == 0
    print(f"✓ Data prepared with {len(prepared_data.columns)} features")
    
    # Test 4: Feature extraction
    print("\n4. Testing feature extraction...")
    features = analyzer._extract_features(prepared_data)
    assert features.shape[0] == len(prepared_data)
    assert features.shape[1] > 0
    print(f"✓ Extracted {features.shape[1]} features")
    
    # Test 5: Clustering
    print("\n5. Testing regime clustering...")
    labels = analyzer._perform_clustering(features)
    assert len(labels) == len(features)
    unique_regimes = len(np.unique(labels))
    print(f"✓ Clustered into {unique_regimes} regimes")
    
    # Test 6: Full analysis
    print("\n6. Running full analysis pipeline...")
    results = analyzer.analyze_historical_patterns(sample_data)
    
    assert 'regime_patterns' in results
    assert 'transition_matrix' in results
    assert 'stability_metrics' in results
    
    stability = results['stability_metrics']
    print(f"✓ Analysis complete:")
    print(f"  - Total transitions: {stability['total_transitions']}")
    print(f"  - Average regime duration: {stability['average_regime_duration']:.1f}")
    print(f"  - Regime persistence: {stability['regime_persistence']:.3f}")
    
    return True


def test_regime_builder():
    """Test regime definition builder"""
    print("\n=== Testing Regime Definition Builder ===")
    
    # Generate mock analysis results
    print("\n1. Creating mock analysis results...")
    mock_patterns = {}
    for i in range(12):
        mock_patterns[i] = type('Pattern', (), {
            'volatility_range': (0.05 + i*0.02, 0.10 + i*0.02),
            'trend_range': (-0.005 + i*0.001, 0.005 + i*0.001),
            'volume_profile': {'mean': 5000, 'std': 1000, 'skew': 0.1, 'relative': 1.0},
            'average_duration': 30 + i*5,
            'stability_score': 0.7 - i*0.02,
            'characteristic_features': ['test_feature'],
            'transition_probabilities': {}
        })()
    
    mock_results = {
        'regime_patterns': mock_patterns,
        'transition_matrix': pd.DataFrame(np.eye(12) * 0.8 + 0.02),
        'stability_metrics': {'average_regime_duration': 45}
    }
    print("✓ Mock data created")
    
    # Test 2: Initialize builder
    print("\n2. Initializing regime definition builder...")
    builder = RegimeDefinitionBuilder(regime_count=12)
    print("✓ Builder initialized for 12 regimes")
    
    # Test 3: Build definitions
    print("\n3. Building regime definitions...")
    definitions = builder.build_regime_definitions(mock_results)
    assert len(definitions) == 12
    print(f"✓ Built {len(definitions)} regime definitions")
    
    # Test 4: Verify definition structure
    print("\n4. Verifying definition structure...")
    for regime_id, definition in definitions.items():
        assert definition.regime_id == regime_id
        assert definition.boundaries is not None
        assert len(definition.strategy_preferences) > 0
        assert definition.risk_parameters['position_size_multiplier'] > 0
    print("✓ All definitions have required components")
    
    # Test 5: Export definitions
    print("\n5. Testing definition export...")
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "definitions.json"
        json_str = builder.export_definitions(str(export_path))
        
        assert export_path.exists()
        with open(export_path, 'r') as f:
            loaded = json.load(f)
        assert len(loaded) == 12
        print("✓ Definitions exported to JSON successfully")
    
    # Test 6: Get regime summary
    print("\n6. Getting regime summary...")
    summary = builder.get_regime_summary()
    print(f"✓ Summary generated with {len(summary)} regimes")
    print("\nSample regime definitions:")
    print(summary[['regime_id', 'name', 'confidence_threshold', 'position_multiplier']].head())
    
    return True


def test_end_to_end_integration():
    """Test end-to-end integration"""
    print("\n=== Testing End-to-End Integration ===")
    
    # Step 1: Configuration
    print("\n1. Setting up configuration...")
    config_manager = AdaptiveRegimeConfigManager()
    config = {
        'regime_count': 8,
        'historical_lookback_days': 30,
        'intraday_window': '5min',
        'clustering_algorithm': 'kmeans'
    }
    print("✓ Configuration set")
    
    # Step 2: Generate realistic data
    print("\n2. Generating realistic market data...")
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=5000, freq='5min')
    
    # Create regime-like patterns
    data_points = []
    price = 100.0
    
    for i in range(5000):
        # Simulate different regimes
        if i < 1000:  # Low vol bullish
            trend = 0.0002
            volatility = 0.001
        elif i < 2500:  # High vol neutral
            trend = 0.0
            volatility = 0.004
        elif i < 4000:  # Med vol bearish
            trend = -0.0001
            volatility = 0.002
        else:  # Low vol range
            trend = 0.0
            volatility = 0.0015
        
        price_change = trend + np.random.randn() * volatility
        price = price * (1 + price_change)
        
        data_points.append({
            'timestamp': dates[i],
            'price': price,
            'volume': np.random.randint(3000, 7000)
        })
    
    market_data = pd.DataFrame(data_points)
    print(f"✓ Generated {len(market_data)} data points with regime patterns")
    
    # Step 3: Run analysis
    print("\n3. Running historical analysis...")
    analyzer = HistoricalRegimeAnalyzer(config)
    analysis_results = analyzer.analyze_historical_patterns(market_data)
    
    patterns = analysis_results['regime_patterns']
    print(f"✓ Identified {len(patterns)} regime patterns")
    
    # Step 4: Build definitions
    print("\n4. Building regime definitions...")
    builder = RegimeDefinitionBuilder(regime_count=8)
    definitions = builder.build_regime_definitions(analysis_results)
    print(f"✓ Built {len(definitions)} regime definitions")
    
    # Step 5: Test regime detection
    print("\n5. Testing regime detection...")
    test_conditions = [
        (0.01, 0.002, 1.0),   # Low vol bullish
        (0.04, 0.0, 1.2),     # High vol neutral
        (0.02, -0.001, 1.1)   # Med vol bearish
    ]
    
    for vol, trend, volume in test_conditions:
        regime = builder.get_regime_by_conditions(vol, trend, volume)
        if regime:
            print(f"✓ Detected regime: {regime.name}")
        else:
            print(f"  No regime found for conditions: vol={vol}, trend={trend}")
    
    # Step 6: Performance summary
    print("\n6. Integration test summary:")
    print(f"  - Configuration: {config['regime_count']} regimes")
    print(f"  - Data points: {len(market_data)}")
    print(f"  - Patterns found: {len(patterns)}")
    print(f"  - Definitions created: {len(definitions)}")
    print(f"  - Stability score: {analysis_results['stability_metrics']['regime_persistence']:.3f}")
    
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("ADAPTIVE MARKET REGIME FORMATION SYSTEM - PHASE 1 TESTS")
    print("=" * 60)
    
    all_passed = True
    
    try:
        # Test each module
        if not test_config_manager():
            all_passed = False
            print("❌ Configuration Manager tests failed")
        
        if not test_historical_analyzer():
            all_passed = False
            print("❌ Historical Analyzer tests failed")
        
        if not test_regime_builder():
            all_passed = False
            print("❌ Regime Definition Builder tests failed")
        
        if not test_end_to_end_integration():
            all_passed = False
            print("❌ Integration tests failed")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL PHASE 1 TESTS PASSED!")
    else:
        print("❌ Some tests failed")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)