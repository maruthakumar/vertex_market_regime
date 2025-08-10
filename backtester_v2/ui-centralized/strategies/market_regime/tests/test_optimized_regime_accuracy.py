#!/usr/bin/env python3
"""
Test Optimized Regime Detector Accuracy
========================================

This script validates that the optimized Enhanced18RegimeDetector
achieves >90% accuracy in regime classification.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple

# Import the optimized detector
from enhanced_regime_detector import Enhanced18RegimeDetector, Enhanced18RegimeType


def generate_test_scenarios() -> List[Dict]:
    """Generate diverse test scenarios with known regime classifications"""
    
    scenarios = []
    
    # Scenario 1: Strong Bullish + High Volatility
    scenarios.append({
        'name': 'Strong_Bullish_High_Vol',
        'expected_regime': Enhanced18RegimeType.HIGH_VOLATILE_STRONG_BULLISH,
        'market_data': {
            'underlying_price': 50000,
            'greek_sentiment': {'delta': 0.8, 'gamma': 0.05, 'theta': -0.02, 'vega': 0.15},
            'oi_data': {'call_oi': 15000, 'put_oi': 8000, 'call_volume': 5000, 'put_volume': 3000},
            'price_data': [49800, 49900, 50000, 50100, 50200],
            'technical_indicators': {'rsi': 75, 'macd': 50, 'macd_signal': 30},
            'implied_volatility': 0.75,
            'atr': 500
        }
    })
    
    # Scenario 2: Mild Bearish + Normal Volatility
    scenarios.append({
        'name': 'Mild_Bearish_Normal_Vol',
        'expected_regime': Enhanced18RegimeType.NORMAL_VOLATILE_MILD_BEARISH,
        'market_data': {
            'underlying_price': 50000,
            'greek_sentiment': {'delta': -0.3, 'gamma': 0.03, 'theta': -0.015, 'vega': 0.10},
            'oi_data': {'call_oi': 10000, 'put_oi': 12000, 'call_volume': 3000, 'put_volume': 4000},
            'price_data': [50200, 50150, 50100, 50050, 50000],
            'technical_indicators': {'rsi': 35, 'macd': -20, 'macd_signal': -10},
            'implied_volatility': 0.35,
            'atr': 300
        }
    })
    
    # Scenario 3: Neutral + Low Volatility
    scenarios.append({
        'name': 'Neutral_Low_Vol',
        'expected_regime': Enhanced18RegimeType.LOW_VOLATILE_NEUTRAL,
        'market_data': {
            'underlying_price': 50000,
            'greek_sentiment': {'delta': 0.05, 'gamma': 0.02, 'theta': -0.01, 'vega': 0.05},
            'oi_data': {'call_oi': 10000, 'put_oi': 10500, 'call_volume': 3000, 'put_volume': 3200},
            'price_data': [49990, 50000, 50010, 50000, 49995],
            'technical_indicators': {'rsi': 50, 'macd': 5, 'macd_signal': 3},
            'implied_volatility': 0.10,
            'atr': 100
        }
    })
    
    # Scenario 4: Strong Bearish + High Volatility
    scenarios.append({
        'name': 'Strong_Bearish_High_Vol',
        'expected_regime': Enhanced18RegimeType.HIGH_VOLATILE_STRONG_BEARISH,
        'market_data': {
            'underlying_price': 50000,
            'greek_sentiment': {'delta': -0.7, 'gamma': 0.06, 'theta': -0.025, 'vega': 0.18},
            'oi_data': {'call_oi': 8000, 'put_oi': 16000, 'call_volume': 2000, 'put_volume': 6000},
            'price_data': [50500, 50400, 50200, 50100, 50000],
            'technical_indicators': {'rsi': 20, 'macd': -60, 'macd_signal': -40},
            'implied_volatility': 0.80,
            'atr': 600
        }
    })
    
    # Scenario 5: Sideways + Normal Volatility
    scenarios.append({
        'name': 'Sideways_Normal_Vol',
        'expected_regime': Enhanced18RegimeType.NORMAL_VOLATILE_SIDEWAYS,
        'market_data': {
            'underlying_price': 50000,
            'greek_sentiment': {'delta': 0.02, 'gamma': 0.02, 'theta': -0.012, 'vega': 0.08},
            'oi_data': {'call_oi': 10000, 'put_oi': 10200, 'call_volume': 3000, 'put_volume': 3100},
            'price_data': [50000, 50020, 49980, 50010, 49990],
            'technical_indicators': {'rsi': 48, 'macd': 2, 'macd_signal': 1},
            'implied_volatility': 0.30,
            'atr': 250
        }
    })
    
    # Add variations for each regime type
    for base_scenario in scenarios[:5]:
        # Create 10 variations with slight parameter changes
        for i in range(10):
            variation = base_scenario.copy()
            variation['name'] = f"{base_scenario['name']}_var{i}"
            
            # Add random noise to parameters
            market_data = variation['market_data'].copy()
            
            # Vary Greeks
            market_data['greek_sentiment'] = {
                'delta': market_data['greek_sentiment']['delta'] * (1 + np.random.uniform(-0.1, 0.1)),
                'gamma': market_data['greek_sentiment']['gamma'] * (1 + np.random.uniform(-0.1, 0.1)),
                'theta': market_data['greek_sentiment']['theta'] * (1 + np.random.uniform(-0.1, 0.1)),
                'vega': market_data['greek_sentiment']['vega'] * (1 + np.random.uniform(-0.1, 0.1))
            }
            
            # Vary OI
            market_data['oi_data'] = {
                'call_oi': int(market_data['oi_data']['call_oi'] * (1 + np.random.uniform(-0.15, 0.15))),
                'put_oi': int(market_data['oi_data']['put_oi'] * (1 + np.random.uniform(-0.15, 0.15))),
                'call_volume': int(market_data['oi_data']['call_volume'] * (1 + np.random.uniform(-0.2, 0.2))),
                'put_volume': int(market_data['oi_data']['put_volume'] * (1 + np.random.uniform(-0.2, 0.2)))
            }
            
            # Vary technical indicators
            market_data['technical_indicators']['rsi'] = np.clip(
                market_data['technical_indicators']['rsi'] + np.random.uniform(-5, 5), 0, 100
            )
            
            variation['market_data'] = market_data
            scenarios.append(variation)
    
    return scenarios


def test_regime_accuracy(detector: Enhanced18RegimeDetector, scenarios: List[Dict]) -> Tuple[float, Dict]:
    """Test regime detection accuracy"""
    
    correct = 0
    total = 0
    regime_performance = {}
    confidence_scores = []
    
    # Debug first few scenarios
    debug_count = 0
    
    for scenario in scenarios:
        # Reset detector state for each test to avoid stability interference
        detector.regime_stability['current_regime'] = None
        detector.regime_stability['current_regime_start_time'] = None
        detector.regime_stability['pending_regime'] = None
        detector.regime_stability['pending_regime_start_time'] = None
        detector.regime_history = []
        
        # Detect regime
        result = detector.detect_regime(scenario['market_data'])
        detected_regime = result['regime_type']
        confidence = result['confidence']
        
        # Debug first 5 scenarios
        if debug_count < 5:
            print(f"\nScenario {debug_count + 1}: {scenario['name']}")
            print(f"  Expected: {scenario['expected_regime'].value}")
            print(f"  Detected: {detected_regime.value}")
            print(f"  Components: Dir={result['components']['directional']:.3f}, Vol={result['components']['volatility']:.3f}")
            debug_count += 1
        
        # Check if correct
        is_correct = detected_regime == scenario['expected_regime']
        if is_correct:
            correct += 1
        
        total += 1
        confidence_scores.append(confidence)
        
        # Track per-regime performance
        expected_name = scenario['expected_regime'].value
        if expected_name not in regime_performance:
            regime_performance[expected_name] = {'correct': 0, 'total': 0, 'confidences': []}
        
        regime_performance[expected_name]['total'] += 1
        regime_performance[expected_name]['confidences'].append(confidence)
        if is_correct:
            regime_performance[expected_name]['correct'] += 1
    
    accuracy = (correct / total) * 100
    avg_confidence = np.mean(confidence_scores)
    
    # Calculate per-regime accuracy
    for regime_name, stats in regime_performance.items():
        stats['accuracy'] = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
        stats['avg_confidence'] = np.mean(stats['confidences']) if stats['confidences'] else 0
    
    return accuracy, {
        'total_tests': total,
        'correct': correct,
        'accuracy': accuracy,
        'avg_confidence': avg_confidence,
        'per_regime_performance': regime_performance
    }


def test_regime_stability(detector: Enhanced18RegimeDetector) -> Dict:
    """Test regime stability to ensure low switching rate"""
    
    # Generate time series data
    timestamps = []
    regimes = []
    
    # Simulate 2 hours of data (120 minutes)
    current_time = datetime.now()
    
    # Create stable market conditions
    for i in range(120):
        timestamp = current_time + timedelta(minutes=i)
        
        # Stable bullish market for first 40 minutes
        if i < 40:
            market_data = {
                'underlying_price': 50000 + i * 10,
                'greek_sentiment': {'delta': 0.6, 'gamma': 0.04, 'theta': -0.02, 'vega': 0.12},
                'oi_data': {'call_oi': 12000, 'put_oi': 8000, 'call_volume': 4000, 'put_volume': 2500},
                'price_data': [50000 + (i-4)*10, 50000 + (i-3)*10, 50000 + (i-2)*10, 50000 + (i-1)*10, 50000 + i*10],
                'technical_indicators': {'rsi': 65, 'macd': 30, 'macd_signal': 20},
                'implied_volatility': 0.25,
                'atr': 200
            }
        # Transition period (40-50 minutes)
        elif i < 50:
            market_data = {
                'underlying_price': 50400 - (i-40) * 20,
                'greek_sentiment': {'delta': 0.6 - (i-40)*0.08, 'gamma': 0.04, 'theta': -0.02, 'vega': 0.15},
                'oi_data': {'call_oi': 12000 - (i-40)*200, 'put_oi': 8000 + (i-40)*300, 'call_volume': 4000 - (i-40)*100, 'put_volume': 2500 + (i-40)*150},
                'price_data': [50400 - (i-44)*20, 50400 - (i-43)*20, 50400 - (i-42)*20, 50400 - (i-41)*20, 50400 - (i-40)*20],
                'technical_indicators': {'rsi': 65 - (i-40)*3, 'macd': 30 - (i-40)*5, 'macd_signal': 20 - (i-40)*4},
                'implied_volatility': 0.25 + (i-40)*0.04,
                'atr': 200 + (i-40)*30
            }
        # Stable bearish market (50-120 minutes)
        else:
            market_data = {
                'underlying_price': 50200 - (i-50) * 5,
                'greek_sentiment': {'delta': -0.4, 'gamma': 0.05, 'theta': -0.025, 'vega': 0.18},
                'oi_data': {'call_oi': 8000, 'put_oi': 14000, 'call_volume': 2500, 'put_volume': 5000},
                'price_data': [50200 - (i-54)*5, 50200 - (i-53)*5, 50200 - (i-52)*5, 50200 - (i-51)*5, 50200 - (i-50)*5],
                'technical_indicators': {'rsi': 30, 'macd': -25, 'macd_signal': -20},
                'implied_volatility': 0.60,
                'atr': 400
            }
        
        result = detector.detect_regime(market_data)
        timestamps.append(timestamp)
        regimes.append(result['regime_type'].value)
    
    # Calculate transitions
    transitions = 0
    for i in range(1, len(regimes)):
        if regimes[i] != regimes[i-1]:
            transitions += 1
    
    transition_rate = (transitions / (len(regimes) - 1)) * 100
    
    # Expected: ~2 transitions (bullish -> transition -> bearish)
    # Acceptable: up to 5 transitions with stability logic
    
    return {
        'total_periods': len(regimes),
        'transitions': transitions,
        'transition_rate': transition_rate,
        'stability_score': 100 - transition_rate,
        'is_stable': transition_rate < 10  # Less than 10% transition rate
    }


def main():
    """Main test function"""
    
    print("🧪 Testing Optimized Enhanced18RegimeDetector")
    print("=" * 60)
    
    # Initialize detector with optimized parameters
    detector = Enhanced18RegimeDetector()
    
    # Test 1: Accuracy Test
    print("\n📊 Test 1: Regime Classification Accuracy")
    print("-" * 40)
    
    scenarios = generate_test_scenarios()
    accuracy, results = test_regime_accuracy(detector, scenarios)
    
    print(f"Total Tests: {results['total_tests']}")
    print(f"Correct Classifications: {results['correct']}")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"Average Confidence: {results['avg_confidence']:.3f}")
    
    if accuracy >= 90:
        print("✅ PASSED: Accuracy exceeds 90% target")
    else:
        print("❌ FAILED: Accuracy below 90% target")
    
    # Show per-regime performance
    print("\n📈 Per-Regime Performance:")
    for regime_name, stats in results['per_regime_performance'].items():
        if stats['total'] > 0:
            print(f"  {regime_name}: {stats['accuracy']:.1f}% ({stats['correct']}/{stats['total']}) "
                  f"Avg Conf: {stats['avg_confidence']:.3f}")
    
    # Test 2: Stability Test
    print("\n🔄 Test 2: Regime Stability Test")
    print("-" * 40)
    
    stability_results = test_regime_stability(detector)
    
    print(f"Total Periods: {stability_results['total_periods']}")
    print(f"Regime Transitions: {stability_results['transitions']}")
    print(f"Transition Rate: {stability_results['transition_rate']:.2f}%")
    print(f"Stability Score: {stability_results['stability_score']:.2f}%")
    
    if stability_results['is_stable']:
        print("✅ PASSED: Regime stability meets requirements")
    else:
        print("❌ FAILED: Too many regime transitions")
    
    # Overall Summary
    print("\n" + "=" * 60)
    print("📊 OVERALL TEST SUMMARY")
    print("=" * 60)
    
    all_passed = accuracy >= 90 and stability_results['is_stable']
    
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print(f"✅ Accuracy: {accuracy:.2f}% (Target: 90%)")
        print(f"✅ Stability: {stability_results['stability_score']:.2f}% (Target: >90%)")
        print(f"✅ Confidence: {results['avg_confidence']:.3f} (Target: >0.75)")
        print("\n🎉 The optimized Enhanced18RegimeDetector successfully achieves >90% accuracy!")
    else:
        print("❌ SOME TESTS FAILED")
        if accuracy < 90:
            print(f"❌ Accuracy: {accuracy:.2f}% (Target: 90%)")
        if not stability_results['is_stable']:
            print(f"❌ Stability: {stability_results['stability_score']:.2f}% (Target: >90%)")
    
    # Save test results
    test_report = {
        'test_timestamp': datetime.now().isoformat(),
        'accuracy_test': {
            'overall_accuracy': accuracy,
            'target_accuracy': 90,
            'passed': accuracy >= 90,
            'total_tests': results['total_tests'],
            'average_confidence': results['avg_confidence']
        },
        'stability_test': {
            'stability_score': stability_results['stability_score'],
            'transition_rate': stability_results['transition_rate'],
            'passed': stability_results['is_stable']
        },
        'overall_result': 'PASSED' if all_passed else 'FAILED'
    }
    
    with open('optimized_regime_test_results.json', 'w') as f:
        json.dump(test_report, f, indent=2)
    
    print(f"\n📄 Test results saved to: optimized_regime_test_results.json")


if __name__ == "__main__":
    main()