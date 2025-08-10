"""
Standalone Test Script for Phase 2 Modules

Run this script directly to test Phase 2 implementation
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

# Import Phase 2 modules
from core.adaptive_scoring_layer import AdaptiveScoringLayer, ASLConfiguration, ComponentScore
from analysis.transition_matrix_analyzer import TransitionMatrixAnalyzer, TransitionPattern, MarkovChainAnalysis
from core.dynamic_boundary_optimizer import DynamicBoundaryOptimizer, RegimeBoundary, OptimizationResult


def test_adaptive_scoring_layer():
    """Test Adaptive Scoring Layer"""
    print("\n=== Testing Adaptive Scoring Layer ===")
    
    # Test 1: Initialization
    print("\n1. Testing ASL initialization...")
    config = ASLConfiguration(
        learning_rate=0.05,
        decay_factor=0.95,
        performance_window=50
    )
    asl = AdaptiveScoringLayer(config)
    
    assert len(asl.components) == 5
    assert all(comp in asl.weights for comp in asl.components)
    print("✓ ASL initialized with 5 components")
    
    # Test 2: Component score calculation
    print("\n2. Testing component score calculation...")
    market_data = {
        'regime_count': 8,
        'triple_straddle_value': 120.5,
        'total_delta': 1500,
        'total_gamma': -300,
        'total_vega': 800,
        'call_open_interest': 50000,
        'put_open_interest': 45000,
        'oi_change_percent': 5.2,
        'rsi': 58,
        'macd_signal': 0.5,
        'bb_position': 0.7,
        'ml_regime_prediction': 0.75,
        'ml_confidence': 0.82,
        'volatility': 0.25
    }
    
    scores = asl.calculate_regime_scores(market_data)
    assert len(scores) == 8
    assert all(0 <= score <= 1 for score in scores.values())
    print(f"✓ Calculated scores for {len(scores)} regimes")
    
    # Test 3: Weight adaptation
    print("\n3. Testing weight adaptation...")
    initial_weights = asl.weights.copy()
    
    # Simulate multiple predictions and updates
    for i in range(20):
        scores = asl.calculate_regime_scores(market_data)
        actual_regime = max(scores, key=scores.get)  # Best predicted regime
        asl.update_weights_based_on_performance(scores, actual_regime)
    
    # Check if weights changed
    weights_changed = any(
        asl.weights[comp] != initial_weights[comp] 
        for comp in asl.components
    )
    print(f"✓ Weights adapted: {weights_changed}")
    
    # Test 4: Performance metrics
    print("\n4. Testing performance metrics...")
    metrics = asl.get_performance_metrics()
    assert 'total_predictions' in metrics
    assert 'current_weights' in metrics
    assert metrics['total_predictions'] >= 20
    print(f"✓ Total predictions: {metrics['total_predictions']}")
    print(f"  Current learning rate: {metrics['current_learning_rate']:.6f}")
    
    # Test 5: State persistence
    print("\n5. Testing state save/load...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        filepath = f.name
    
    asl.save_state(filepath)
    
    # Create new instance and load state
    asl2 = AdaptiveScoringLayer(config)
    asl2.load_state(filepath)
    
    # Verify weights loaded correctly
    weights_match = all(
        asl.weights[comp] == asl2.weights[comp]
        for comp in asl.components
    )
    print(f"✓ State persistence working: {weights_match}")
    
    # Cleanup
    Path(filepath).unlink(missing_ok=True)
    
    return True


def test_transition_matrix_analyzer():
    """Test Transition Matrix Analyzer"""
    print("\n=== Testing Transition Matrix Analyzer ===")
    
    # Test 1: Initialization
    print("\n1. Testing analyzer initialization...")
    analyzer = TransitionMatrixAnalyzer(regime_count=8)
    assert analyzer.regime_count == 8
    print("✓ Analyzer initialized for 8 regimes")
    
    # Test 2: Generate realistic regime sequence
    print("\n2. Generating regime sequence with patterns...")
    np.random.seed(42)
    regime_sequence = []
    current_regime = 0
    
    # Create realistic transitions
    for i in range(1000):
        if np.random.random() < 0.1:  # 10% chance of transition
            # Prefer nearby regimes
            if current_regime == 0:
                current_regime = np.random.choice([0, 1])
            elif current_regime == 7:
                current_regime = np.random.choice([6, 7])
            else:
                current_regime = np.random.choice([
                    current_regime - 1, current_regime, current_regime + 1
                ])
        regime_sequence.append(current_regime)
    
    print(f"✓ Generated sequence with {len(regime_sequence)} data points")
    
    # Test 3: Analyze transitions
    print("\n3. Analyzing transition patterns...")
    timestamps = [datetime.now() - timedelta(minutes=5*i) for i in range(1000)]
    timestamps.reverse()
    
    # Generate features for transitions
    features = []
    for i in range(1000):
        features.append({
            'volatility': 0.1 + 0.3 * (regime_sequence[i] / 7),
            'trend': -0.02 + 0.04 * (regime_sequence[i] / 7),
            'volume': 0.8 + 0.4 * np.random.random()
        })
    
    results = analyzer.analyze_transitions(regime_sequence, timestamps, features)
    
    assert 'transition_matrix' in results
    assert 'transition_patterns' in results
    assert 'markov_analysis' in results
    assert 'stability_metrics' in results
    
    print("✓ Transition analysis complete")
    
    # Check available keys
    if 'stability_metrics' in results and results['stability_metrics']:
        metrics = results['stability_metrics']
        print(f"  Total transitions: {metrics.get('total_transitions', 'N/A')}")
        if 'transition_rate' in metrics:
            print(f"  Transition rate: {metrics['transition_rate']:.3f}")
        if 'average_regime_duration' in metrics:
            print(f"  Average regime duration: {metrics['average_regime_duration']:.1f}")
    else:
        print("  Stability metrics not available")
    
    # Test 4: Transition matrix properties
    print("\n4. Testing transition matrix properties...")
    matrix = results['transition_matrix']
    
    # Check matrix is valid probability matrix
    for i in range(8):
        row_sum = matrix.loc[i].sum()
        assert abs(row_sum - 1.0) < 0.01, f"Row {i} doesn't sum to 1: {row_sum}"
    
    print("✓ Transition matrix is valid stochastic matrix")
    
    # Test 5: Markov analysis
    print("\n5. Testing Markov chain analysis...")
    markov = results['markov_analysis']
    if markov:
        assert hasattr(markov, 'stationary_distribution')
        assert hasattr(markov, 'ergodic')
        assert hasattr(markov, 'irreducible')
        print(f"✓ Markov analysis: ergodic={markov.ergodic}, irreducible={markov.irreducible}")
        
        # Check stationary distribution
        if markov.stationary_distribution is not None:
            stat_sum = markov.stationary_distribution.sum()
            assert abs(stat_sum - 1.0) < 0.01, f"Stationary distribution doesn't sum to 1: {stat_sum}"
    
    # Test 6: Pattern identification
    print("\n6. Testing pattern identification...")
    patterns = results['transition_patterns']
    print(f"✓ Identified {len(patterns)} transition patterns")
    
    # Show top patterns
    if patterns:
        sorted_patterns = sorted(
            patterns.items(), 
            key=lambda x: x[1].occurrence_count, 
            reverse=True
        )[:3]
        
        print("  Top transition patterns:")
        for (from_r, to_r), pattern in sorted_patterns:
            print(f"    {from_r} → {to_r}: count={pattern.occurrence_count}, "
                  f"prob={pattern.probability:.3f}, stability={pattern.stability_score:.3f}")
    
    # Test 7: Next regime prediction
    print("\n7. Testing regime prediction...")
    current_regime = 3
    next_probs = analyzer.predict_next_regime(current_regime, features[0])
    
    assert len(next_probs) == 8
    assert abs(sum(next_probs.values()) - 1.0) < 0.01
    
    best_next = max(next_probs.items(), key=lambda x: x[1])
    print(f"✓ From regime {current_regime}, most likely next: "
          f"regime {best_next[0]} (prob={best_next[1]:.3f})")
    
    # Test 8: Export functionality
    print("\n8. Testing export...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        filepath = f.name
    
    analyzer.export_analysis(filepath)
    
    # Verify file created
    assert Path(filepath).exists()
    with open(filepath, 'r') as f:
        exported = json.load(f)
    assert 'regime_count' in exported
    assert exported['regime_count'] == 8
    print("✓ Analysis exported successfully")
    
    # Cleanup
    Path(filepath).unlink(missing_ok=True)
    
    return True


def test_dynamic_boundary_optimizer():
    """Test Dynamic Boundary Optimizer"""
    print("\n=== Testing Dynamic Boundary Optimizer ===")
    
    # Test 1: Initialization
    print("\n1. Testing optimizer initialization...")
    optimizer = DynamicBoundaryOptimizer(
        regime_count=8,
        optimization_window=50,
        update_frequency=10
    )
    
    assert optimizer.regime_count == 8
    assert len(optimizer.current_boundaries) == 8
    print("✓ Optimizer initialized with 8 regime boundaries")
    
    # Test 2: Boundary structure
    print("\n2. Testing boundary structure...")
    for i in range(8):
        boundary = optimizer.current_boundaries[i]
        assert boundary.regime_id == i
        assert len(boundary.volatility_bounds) == 2
        assert len(boundary.trend_bounds) == 2
        assert len(boundary.volume_bounds) == 2
        assert boundary.volatility_bounds[0] < boundary.volatility_bounds[1]
    print("✓ All boundaries properly structured")
    
    # Test 3: Performance data update
    print("\n3. Testing performance tracking...")
    performance_data = []
    
    # Generate realistic performance data
    for i in range(100):
        # Create pattern where some regimes are better predicted
        if i % 8 in [0, 1, 2]:  # Good prediction for first 3 regimes
            predicted = i % 8
            actual = i % 8
        else:  # Poorer prediction for others
            predicted = i % 8
            actual = (i % 8 + np.random.randint(-1, 2)) % 8
        
        performance_data.append({
            'predicted_regime': predicted,
            'actual_regime': actual,
            'timestamp': datetime.now() - timedelta(minutes=i*5)
        })
    
    optimizer._update_performance_metrics(performance_data)
    
    # Check accuracy updated
    high_accuracy_regimes = sum(1 for acc in optimizer.regime_accuracy.values() if acc > 0.7)
    print(f"✓ Performance metrics updated: {high_accuracy_regimes} high-accuracy regimes")
    
    # Test 4: Boundary optimization
    print("\n4. Testing boundary optimization...")
    market_conditions = {
        'volatility': 0.15,
        'trend': 0.002,
        'volume_ratio': 1.1
    }
    
    result = optimizer.optimize_boundaries(performance_data, market_conditions)
    
    assert isinstance(result, OptimizationResult)
    assert result.iterations > 0
    print(f"✓ Optimization completed in {result.iterations} iterations")
    print(f"  Converged: {result.convergence_status}")
    print(f"  Improvement: {result.improvement:.2%}")
    print(f"  Time: {result.optimization_time:.2f}s")
    
    # Test 5: Regime transition with hysteresis
    print("\n5. Testing regime transition logic...")
    current_regime = 3
    
    # Test transition check
    new_regime, confidence = optimizer.check_regime_transition(
        current_regime, market_conditions
    )
    
    assert 0 <= new_regime < 8
    assert 0 <= confidence <= 1
    print(f"✓ Transition check: current={current_regime}, suggested={new_regime}, "
          f"confidence={confidence:.3f}")
    
    # Test 6: Adaptive hysteresis
    print("\n6. Testing adaptive hysteresis...")
    initial_hysteresis = optimizer.current_boundaries[0].hysteresis_factor
    
    # Simulate false transitions
    transition_data = [
        {'from_regime': 0, 'to_regime': 1, 'duration': 3, 'reversed_quickly': True},
        {'from_regime': 0, 'to_regime': 1, 'duration': 2, 'reversed_quickly': True},
        {'from_regime': 0, 'to_regime': 1, 'duration': 4, 'reversed_quickly': True}
    ]
    
    optimizer.update_hysteresis(transition_data)
    
    new_hysteresis = optimizer.current_boundaries[0].hysteresis_factor
    assert new_hysteresis > initial_hysteresis
    print(f"✓ Hysteresis adapted: {initial_hysteresis:.3f} → {new_hysteresis:.3f}")
    
    # Test 7: Optimization metrics
    print("\n7. Testing optimization metrics...")
    metrics = optimizer.get_optimization_metrics()
    
    assert 'total_optimizations' in metrics
    assert 'success_rate' in metrics
    assert 'regime_accuracy' in metrics
    assert metrics['total_optimizations'] >= 1
    
    print(f"✓ Optimization metrics:")
    print(f"  Total optimizations: {metrics['total_optimizations']}")
    print(f"  Success rate: {metrics['success_rate']:.1%}")
    print(f"  Average improvement: {metrics['average_improvement']:.2%}")
    
    # Test 8: Boundary export
    print("\n8. Testing boundary export...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        filepath = f.name
    
    optimizer.export_boundaries(filepath)
    
    # Verify export
    with open(filepath, 'r') as f:
        exported = json.load(f)
    
    assert exported['regime_count'] == 8
    assert len(exported['boundaries']) == 8
    print("✓ Boundaries exported successfully")
    
    # Cleanup
    Path(filepath).unlink(missing_ok=True)
    
    return True


def test_phase2_integration():
    """Test integration of Phase 2 modules"""
    print("\n=== Testing Phase 2 Integration ===")
    
    # Initialize all modules
    print("\n1. Initializing all Phase 2 modules...")
    asl = AdaptiveScoringLayer(ASLConfiguration())
    transition_analyzer = TransitionMatrixAnalyzer(regime_count=8)
    boundary_optimizer = DynamicBoundaryOptimizer(regime_count=8)
    print("✓ All modules initialized")
    
    # Generate integrated test data
    print("\n2. Generating integrated test scenario...")
    np.random.seed(42)
    
    # Simulate market data over time
    market_history = []
    regime_sequence = []
    current_regime = 0
    
    for i in range(500):
        # Market conditions evolve
        volatility = 0.1 + 0.05 * np.sin(i / 50) + 0.02 * np.random.randn()
        trend = 0.001 * np.sin(i / 100) + 0.0005 * np.random.randn()
        volume_ratio = 1.0 + 0.2 * np.sin(i / 30) + 0.1 * np.random.randn()
        
        market_data = {
            'regime_count': 8,
            'volatility': abs(volatility),
            'trend': trend,
            'volume_ratio': abs(volume_ratio),
            'triple_straddle_value': 100 + 20 * volatility,
            'total_delta': 1000 * trend,
            'total_gamma': -500 * volatility,
            'total_vega': 1000 * volatility,
            'call_open_interest': 50000 * (1 + trend),
            'put_open_interest': 50000 * (1 - trend),
            'oi_change_percent': 5 * np.random.randn(),
            'rsi': 50 + 30 * trend,
            'macd_signal': trend,
            'bb_position': 0.5 + 0.3 * trend,
            'ml_regime_prediction': 0.5 + 0.3 * np.random.randn(),
            'ml_confidence': 0.7 + 0.2 * np.random.randn()
        }
        
        market_history.append(market_data)
        
        # Use ASL to score regimes
        regime_scores = asl.calculate_regime_scores(market_data)
        predicted_regime = max(regime_scores.items(), key=lambda x: x[1])[0]
        
        # Check for regime transition using boundary optimizer
        new_regime, confidence = boundary_optimizer.check_regime_transition(
            current_regime, market_data
        )
        
        if confidence > 0.7:  # High confidence transition
            current_regime = new_regime
        
        regime_sequence.append(current_regime)
        
        # Update ASL with feedback every 10 steps
        if i > 0 and i % 10 == 0:
            asl.update_weights_based_on_performance(regime_scores, current_regime)
    
    print(f"✓ Generated {len(market_history)} integrated data points")
    
    # Test 3: Analyze transitions with the sequence
    print("\n3. Analyzing regime transitions...")
    features = [
        {'volatility': m['volatility'], 'trend': m['trend'], 'volume': m['volume_ratio']}
        for m in market_history
    ]
    
    transition_results = transition_analyzer.analyze_transitions(
        regime_sequence, features=features
    )
    
    print(f"✓ Transition analysis complete")
    print(f"  Unique regimes visited: {len(set(regime_sequence))}")
    
    if 'stability_metrics' in transition_results and transition_results['stability_metrics']:
        metrics = transition_results['stability_metrics']
        if 'total_transitions' in metrics:
            print(f"  Total transitions: {metrics['total_transitions']}")
        if 'average_regime_duration' in metrics:
            print(f"  Average duration: {metrics['average_regime_duration']:.1f}")
    
    # Test 4: Optimize boundaries based on performance
    print("\n4. Optimizing boundaries based on integrated performance...")
    
    # Create performance data from the simulation
    performance_data = []
    for i in range(100, len(regime_sequence)):
        performance_data.append({
            'predicted_regime': predicted_regime,
            'actual_regime': regime_sequence[i],
            'timestamp': datetime.now() - timedelta(minutes=(len(regime_sequence)-i)*5)
        })
    
    optimization_result = boundary_optimizer.optimize_boundaries(
        performance_data, 
        market_history[-1]  # Current market conditions
    )
    
    print(f"✓ Boundary optimization complete")
    print(f"  Improvement: {optimization_result.improvement:.2%}")
    
    # Test 5: End-to-end prediction
    print("\n5. Testing end-to-end prediction flow...")
    
    # Get current market data
    current_market = market_history[-1]
    
    # ASL scores regimes
    regime_scores = asl.calculate_regime_scores(current_market)
    
    # Get transition probabilities
    current_regime = regime_sequence[-1]
    next_regime_probs = transition_analyzer.predict_next_regime(
        current_regime, features[-1]
    )
    
    # Combine scores with transition probabilities
    combined_scores = {}
    for regime_id in range(8):
        asl_score = regime_scores.get(regime_id, 0.0)
        transition_prob = next_regime_probs.get(regime_id, 0.0)
        combined_scores[regime_id] = asl_score * 0.6 + transition_prob * 0.4
    
    # Final prediction
    final_prediction = max(combined_scores.items(), key=lambda x: x[1])
    
    print(f"✓ End-to-end prediction:")
    print(f"  Current regime: {current_regime}")
    print(f"  Predicted next regime: {final_prediction[0]}")
    print(f"  Confidence: {final_prediction[1]:.3f}")
    
    # Test 6: Module metrics
    print("\n6. Collecting module metrics...")
    
    asl_metrics = asl.get_performance_metrics()
    optimizer_metrics = boundary_optimizer.get_optimization_metrics()
    
    print(f"✓ Module metrics collected:")
    print(f"  ASL predictions: {asl_metrics['total_predictions']}")
    print(f"  ASL weight variance: {asl_metrics['weight_variance']:.4f}")
    print(f"  Boundary optimizations: {optimizer_metrics['total_optimizations']}")
    print(f"  Optimization success rate: {optimizer_metrics['success_rate']:.1%}")
    
    return True


def main():
    """Run all Phase 2 tests"""
    print("=" * 60)
    print("ADAPTIVE MARKET REGIME FORMATION SYSTEM - PHASE 2 TESTS")
    print("=" * 60)
    
    all_passed = True
    
    try:
        # Test each module
        if not test_adaptive_scoring_layer():
            all_passed = False
            print("❌ Adaptive Scoring Layer tests failed")
        
        if not test_transition_matrix_analyzer():
            all_passed = False
            print("❌ Transition Matrix Analyzer tests failed")
        
        if not test_dynamic_boundary_optimizer():
            all_passed = False
            print("❌ Dynamic Boundary Optimizer tests failed")
        
        if not test_phase2_integration():
            all_passed = False
            print("❌ Phase 2 integration tests failed")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL PHASE 2 TESTS PASSED!")
        print("\nPhase 2 Modules Completed:")
        print("  1. Adaptive Scoring Layer (ASL) ✓")
        print("  2. Transition Matrix Analyzer ✓")
        print("  3. Dynamic Boundary Optimizer ✓")
        print("\nReady to proceed with Phase 3: Intelligence Layer")
    else:
        print("❌ Some tests failed")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)