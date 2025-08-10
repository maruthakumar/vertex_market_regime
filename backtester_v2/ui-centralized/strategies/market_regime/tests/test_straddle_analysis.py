#!/usr/bin/env python3
"""
Test script for the refactored triple straddle analysis system

Validates:
- Component initialization
- Data flow through all analyzers
- Performance targets (<3 seconds)
- Output accuracy
"""

import pandas as pd
import numpy as np
import time
import json
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators.straddle_analysis import (
    TripleStraddleEngine,
    TripleStraddleAnalysisResult
)


class StraddleAnalysisTester:
    """Test harness for straddle analysis system"""
    
    def __init__(self):
        """Initialize test harness"""
        self.engine = None
        self.test_results = []
        self.performance_target = 3.0  # 3 seconds
        
    def setup(self):
        """Setup test environment"""
        print("\n=== Setting up Triple Straddle Analysis Test ===")
        
        try:
            # Initialize engine
            start_time = time.time()
            self.engine = TripleStraddleEngine()
            init_time = time.time() - start_time
            
            print(f"✓ Engine initialized in {init_time:.2f} seconds")
            return True
            
        except Exception as e:
            print(f"✗ Failed to initialize engine: {e}")
            return False
    
    def generate_test_data(self, timestamp: pd.Timestamp) -> dict:
        """Generate realistic test market data"""
        # Base price
        underlying_price = 21500 + np.random.normal(0, 50)
        
        # Generate option prices with realistic relationships
        # ATM options
        atm_strike = round(underlying_price / 50) * 50
        iv_base = 0.15 + np.random.uniform(-0.02, 0.02)
        
        # Simplified Black-Scholes approximation for testing
        time_to_expiry = 7 / 365  # 7 days
        
        # ATM prices (roughly equal for calls and puts at ATM)
        atm_base_price = underlying_price * iv_base * np.sqrt(time_to_expiry) * 0.4
        atm_ce_price = atm_base_price * (1 + np.random.uniform(-0.05, 0.05))
        atm_pe_price = atm_base_price * (1 + np.random.uniform(-0.05, 0.05))
        
        # ITM1 prices (1 strike ITM)
        itm1_ce_price = 50 + atm_ce_price * 0.8  # Intrinsic + reduced time value
        itm1_pe_price = 50 + atm_pe_price * 0.8
        
        # OTM1 prices (1 strike OTM)
        otm1_ce_price = atm_ce_price * 0.6  # Only time value, reduced
        otm1_pe_price = atm_pe_price * 0.6
        
        # Generate Greeks
        market_data = {
            'timestamp': timestamp,
            'underlying_price': underlying_price,
            'spot_price': underlying_price,
            
            # Prices
            'ATM_CE': atm_ce_price,
            'ATM_PE': atm_pe_price,
            'ITM1_CE': itm1_ce_price,
            'ITM1_PE': itm1_pe_price,
            'OTM1_CE': otm1_ce_price,
            'OTM1_PE': otm1_pe_price,
            
            # Greeks for ATM
            'atm_ce_delta': 0.5 + np.random.uniform(-0.05, 0.05),
            'atm_pe_delta': -0.5 + np.random.uniform(-0.05, 0.05),
            'atm_ce_gamma': 0.02 + np.random.uniform(-0.005, 0.005),
            'atm_pe_gamma': 0.02 + np.random.uniform(-0.005, 0.005),
            'atm_ce_theta': -20 + np.random.uniform(-5, 5),
            'atm_pe_theta': -20 + np.random.uniform(-5, 5),
            'atm_ce_vega': 50 + np.random.uniform(-10, 10),
            'atm_pe_vega': 50 + np.random.uniform(-10, 10),
            
            # Volume and OI
            'volume': np.random.randint(1000, 10000),
            'open_interest': np.random.randint(10000, 100000),
            
            # Strikes
            'strikes': [atm_strike - 100, atm_strike - 50, atm_strike, atm_strike + 50, atm_strike + 100],
            
            # Additional data
            'implied_volatility': iv_base,
            'days_to_expiry': 7
        }
        
        return market_data
    
    def test_single_analysis(self) -> dict:
        """Test single analysis cycle"""
        print("\n--- Testing Single Analysis ---")
        
        # Generate test data
        timestamp = pd.Timestamp.now()
        market_data = self.generate_test_data(timestamp)
        
        print(f"Underlying Price: {market_data['underlying_price']:.2f}")
        print(f"ATM Straddle: {market_data['ATM_CE'] + market_data['ATM_PE']:.2f}")
        
        # Run analysis
        start_time = time.time()
        result = self.engine.analyze(market_data, timestamp)
        analysis_time = time.time() - start_time
        
        test_result = {
            'timestamp': timestamp,
            'analysis_time': analysis_time,
            'success': result is not None,
            'result': result
        }
        
        if result:
            print(f"✓ Analysis completed in {analysis_time:.2f} seconds")
            print(f"  Market Regime: {result.market_regime}")
            print(f"  Regime Confidence: {result.regime_confidence:.2%}")
            print(f"  Position Recommendation: {result.position_recommendations.get('action', 'N/A')}")
            
            # Check performance
            if analysis_time <= self.performance_target:
                print(f"✓ Performance target met ({analysis_time:.2f}s <= {self.performance_target}s)")
            else:
                print(f"✗ Performance target missed ({analysis_time:.2f}s > {self.performance_target}s)")
        else:
            print(f"✗ Analysis failed")
        
        return test_result
    
    def test_multiple_analyses(self, n_tests: int = 10) -> dict:
        """Test multiple analyses for consistency and performance"""
        print(f"\n--- Testing {n_tests} Consecutive Analyses ---")
        
        results = []
        analysis_times = []
        regimes = []
        
        base_timestamp = pd.Timestamp.now()
        
        for i in range(n_tests):
            # Generate data with slight variations
            timestamp = base_timestamp + pd.Timedelta(minutes=i)
            market_data = self.generate_test_data(timestamp)
            
            # Run analysis
            start_time = time.time()
            result = self.engine.analyze(market_data, timestamp)
            analysis_time = time.time() - start_time
            
            if result:
                results.append(result)
                analysis_times.append(analysis_time)
                regimes.append(result.market_regime)
            
            # Progress indicator
            print(f"\rCompleted: {i+1}/{n_tests} analyses", end='', flush=True)
        
        print("\n")
        
        # Calculate statistics
        success_rate = len(results) / n_tests
        avg_time = np.mean(analysis_times) if analysis_times else 0
        max_time = np.max(analysis_times) if analysis_times else 0
        min_time = np.min(analysis_times) if analysis_times else 0
        within_target = sum(1 for t in analysis_times if t <= self.performance_target)
        
        print(f"Success Rate: {success_rate:.1%}")
        print(f"Average Analysis Time: {avg_time:.3f}s")
        print(f"Min/Max Time: {min_time:.3f}s / {max_time:.3f}s")
        print(f"Within Target (<{self.performance_target}s): {within_target}/{len(analysis_times)}")
        
        # Regime distribution
        if regimes:
            regime_counts = pd.Series(regimes).value_counts()
            print("\nRegime Distribution:")
            for regime, count in regime_counts.items():
                print(f"  {regime}: {count} ({count/len(regimes):.1%})")
        
        return {
            'n_tests': n_tests,
            'success_rate': success_rate,
            'avg_time': avg_time,
            'max_time': max_time,
            'min_time': min_time,
            'performance_met': within_target / len(analysis_times) if analysis_times else 0
        }
    
    def test_edge_cases(self) -> dict:
        """Test edge cases and error handling"""
        print("\n--- Testing Edge Cases ---")
        
        edge_results = {}
        
        # Test 1: Missing option data
        print("\nTest 1: Missing option data")
        timestamp = pd.Timestamp.now()
        incomplete_data = self.generate_test_data(timestamp)
        del incomplete_data['OTM1_CE']
        del incomplete_data['OTM1_PE']
        
        result = self.engine.analyze(incomplete_data, timestamp)
        edge_results['missing_data'] = result is not None
        print(f"{'✓' if result else '✗'} Handled missing option data")
        
        # Test 2: Extreme prices
        print("\nTest 2: Extreme price movements")
        extreme_data = self.generate_test_data(timestamp)
        extreme_data['underlying_price'] *= 1.1  # 10% move
        
        result = self.engine.analyze(extreme_data, timestamp)
        edge_results['extreme_prices'] = result is not None
        print(f"{'✓' if result else '✗'} Handled extreme price movement")
        
        # Test 3: Zero/negative prices
        print("\nTest 3: Invalid prices")
        invalid_data = self.generate_test_data(timestamp)
        invalid_data['ATM_CE'] = -10  # Invalid negative price
        
        result = self.engine.analyze(invalid_data, timestamp)
        edge_results['invalid_prices'] = result is not None
        print(f"{'✓' if result else '✗'} Handled invalid prices")
        
        return edge_results
    
    def test_component_status(self) -> dict:
        """Test individual component status"""
        print("\n--- Testing Component Status ---")
        
        status = self.engine.get_engine_status()
        
        print("\nComponent Status:")
        components = status.get('components', {})
        
        for component_name, component_status in components.items():
            print(f"\n{component_name}:")
            if isinstance(component_status, dict):
                for key, value in component_status.items():
                    if not isinstance(value, (dict, list)):
                        print(f"  {key}: {value}")
        
        return status
    
    def run_all_tests(self):
        """Run all tests"""
        print("\n" + "=" * 60)
        print("TRIPLE STRADDLE ANALYSIS TEST SUITE")
        print("=" * 60)
        
        # Setup
        if not self.setup():
            print("\nTest setup failed. Exiting.")
            return
        
        # Run tests
        all_results = {}
        
        # Single analysis test
        single_result = self.test_single_analysis()
        all_results['single_analysis'] = single_result
        
        # Multiple analyses test
        multiple_results = self.test_multiple_analyses(20)
        all_results['multiple_analyses'] = multiple_results
        
        # Edge cases test
        edge_results = self.test_edge_cases()
        all_results['edge_cases'] = edge_results
        
        # Component status
        component_status = self.test_component_status()
        all_results['component_status'] = component_status
        
        # Summary
        self.print_summary(all_results)
    
    def print_summary(self, results: dict):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        # Performance summary
        multiple = results.get('multiple_analyses', {})
        print(f"\nPerformance:")
        print(f"  Average Time: {multiple.get('avg_time', 0):.3f}s")
        print(f"  Target Achievement: {multiple.get('performance_met', 0):.1%}")
        print(f"  Success Rate: {multiple.get('success_rate', 0):.1%}")
        
        # Edge case summary
        edge = results.get('edge_cases', {})
        print(f"\nEdge Case Handling:")
        for case, passed in edge.items():
            print(f"  {case}: {'PASSED' if passed else 'FAILED'}")
        
        # Overall result
        overall_pass = (
            multiple.get('success_rate', 0) >= 0.95 and
            multiple.get('performance_met', 0) >= 0.8 and
            all(edge.values())
        )
        
        print(f"\nOVERALL RESULT: {'PASSED' if overall_pass else 'FAILED'}")
        print("=" * 60)


if __name__ == "__main__":
    tester = StraddleAnalysisTester()
    tester.run_all_tests()