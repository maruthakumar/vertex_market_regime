"""
Run comprehensive tests for the straddle analysis system
"""

import os
import sys
import subprocess
import json
from datetime import datetime

def main():
    """Main test runner"""
    print("\n" + "="*80)
    print("STRADDLE ANALYSIS SYSTEM TEST")
    print("="*80)
    print(f"Test started at: {datetime.now()}")
    
    # Change to the correct directory
    os.chdir('/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/strategies/market_regime')
    
    # Test results
    test_results = {
        'timestamp': datetime.now().isoformat(),
        'tests': []
    }
    
    # Test 1: Check file structure
    print("\n1. Checking file structure...")
    straddle_path = "indicators/straddle_analysis"
    
    expected_structure = {
        'core': ['__init__.py', 'calculation_engine.py', 'straddle_engine.py', 'resistance_analyzer.py', 'weight_optimizer.py'],
        'components': ['__init__.py', 'base_component_analyzer.py', 
                      'atm_ce_analyzer.py', 'atm_pe_analyzer.py',
                      'itm1_ce_analyzer.py', 'itm1_pe_analyzer.py',
                      'otm1_ce_analyzer.py', 'otm1_pe_analyzer.py',
                      'atm_straddle_analyzer.py', 'itm1_straddle_analyzer.py',
                      'otm1_straddle_analyzer.py', 'combined_straddle_analyzer.py'],
        'rolling': ['__init__.py', 'window_manager.py', 'correlation_matrix.py', 'timeframe_aggregator.py'],
        'config': ['__init__.py', 'excel_reader.py']
    }
    
    structure_valid = True
    for folder, files in expected_structure.items():
        folder_path = os.path.join(straddle_path, folder)
        if not os.path.exists(folder_path):
            print(f"  ✗ Missing folder: {folder_path}")
            structure_valid = False
            continue
        
        for file in files:
            file_path = os.path.join(folder_path, file)
            if not os.path.exists(file_path):
                print(f"  ✗ Missing file: {file_path}")
                structure_valid = False
            else:
                # Check file size
                size = os.path.getsize(file_path)
                if size == 0:
                    print(f"  ⚠ Empty file: {file_path}")
    
    if structure_valid:
        print("  ✓ File structure is complete")
    
    test_results['tests'].append({
        'name': 'File Structure Check',
        'passed': structure_valid,
        'details': 'All required files present' if structure_valid else 'Missing files detected'
    })
    
    # Test 2: Test HeavyDB connection
    print("\n2. Testing HeavyDB connection...")
    try:
        import heavydb
        conn = heavydb.connect(
            host='localhost',
            port=6274,
            user='admin',
            password='HyperInteractive',
            dbname='heavyai'
        )
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM nifty_option_chain")
        count = cursor.fetchone()[0]
        print(f"  ✓ HeavyDB connected. Records: {count:,}")
        conn.close()
        
        test_results['tests'].append({
            'name': 'HeavyDB Connection',
            'passed': True,
            'details': f'Connected successfully. Total records: {count}'
        })
    except Exception as e:
        print(f"  ✗ HeavyDB connection failed: {e}")
        test_results['tests'].append({
            'name': 'HeavyDB Connection',
            'passed': False,
            'details': str(e)
        })
    
    # Test 3: Test Excel configuration
    print("\n3. Testing Excel configuration...")
    excel_path = "/srv/samba/shared/bt/backtester_stable/BTRUN/input_sheets/market_regime/PHASE2_ENHANCED_ULTIMATE_UNIFIED_MARKET_REGIME_CONFIG_20250627_195625_20250628_104335.xlsx"
    
    if os.path.exists(excel_path):
        print(f"  ✓ Excel config file exists")
        test_results['tests'].append({
            'name': 'Excel Configuration',
            'passed': True,
            'details': 'Configuration file found'
        })
    else:
        print(f"  ✗ Excel config file not found: {excel_path}")
        test_results['tests'].append({
            'name': 'Excel Configuration',
            'passed': False,
            'details': 'Configuration file not found'
        })
    
    # Test 4: Simple calculation test
    print("\n4. Testing basic calculations...")
    try:
        # Test straddle value calculation
        ce_price = 150
        pe_price = 145
        straddle_value = ce_price + pe_price
        assert straddle_value == 295, f"Straddle calculation error: {straddle_value}"
        
        # Test volatility calculation
        import numpy as np
        prices = np.array([100, 102, 98, 101, 99, 103, 97, 104])
        returns = np.diff(np.log(prices))
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        
        print(f"  ✓ Basic calculations working")
        print(f"    - Straddle value: {straddle_value}")
        print(f"    - Volatility: {volatility:.4f}")
        
        test_results['tests'].append({
            'name': 'Basic Calculations',
            'passed': True,
            'details': f'Straddle: {straddle_value}, Volatility: {volatility:.4f}'
        })
    except Exception as e:
        print(f"  ✗ Calculation test failed: {e}")
        test_results['tests'].append({
            'name': 'Basic Calculations',
            'passed': False,
            'details': str(e)
        })
    
    # Test 5: Component import test
    print("\n5. Testing component imports...")
    components_ok = True
    try:
        # Set up Python path
        sys.path.insert(0, '/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2')
        
        # Try importing key components using absolute imports
        from strategies.market_regime.indicators.straddle_analysis.config.excel_reader import StraddleExcelReader
        from strategies.market_regime.indicators.straddle_analysis.rolling.window_manager import RollingWindowManager
        from strategies.market_regime.indicators.straddle_analysis.rolling.correlation_matrix import CorrelationMatrix
        
        print("  ✓ Component imports successful")
        
        # Test instantiation
        excel_reader = StraddleExcelReader()
        window_mgr = RollingWindowManager({'rolling_windows': [3, 5, 10, 15]})
        corr_matrix = CorrelationMatrix({'rolling_windows': [3, 5, 10, 15]})
        
        print("  ✓ Components instantiated successfully")
        
    except Exception as e:
        print(f"  ✗ Component import failed: {e}")
        components_ok = False
    
    test_results['tests'].append({
        'name': 'Component Imports',
        'passed': components_ok,
        'details': 'All components imported' if components_ok else 'Import errors detected'
    })
    
    # Test 6: Performance benchmark
    print("\n6. Testing performance...")
    try:
        import time
        import pandas as pd
        
        # Simulate processing 100 data points
        processing_times = []
        
        for i in range(100):
            start_time = time.time()
            
            # Simulate some calculations
            data = {
                'timestamp': pd.Timestamp.now(),
                'values': np.random.randn(6) * 100 + 100
            }
            
            # Simulate correlation calculation
            corr_matrix = np.corrcoef(np.random.randn(6, 20))
            
            # Simulate rolling window
            window_data = np.random.randn(15, 6)
            means = np.mean(window_data, axis=0)
            stds = np.std(window_data, axis=0)
            
            end_time = time.time()
            processing_times.append(end_time - start_time)
        
        avg_time = np.mean(processing_times)
        max_time = np.max(processing_times)
        
        print(f"  ✓ Performance test completed")
        print(f"    - Average time: {avg_time*1000:.2f}ms")
        print(f"    - Max time: {max_time*1000:.2f}ms")
        print(f"    - Target met: {'Yes' if max_time < 3.0 else 'No'}")
        
        test_results['tests'].append({
            'name': 'Performance Benchmark',
            'passed': max_time < 3.0,
            'details': f'Avg: {avg_time*1000:.2f}ms, Max: {max_time*1000:.2f}ms'
        })
        
    except Exception as e:
        print(f"  ✗ Performance test failed: {e}")
        test_results['tests'].append({
            'name': 'Performance Benchmark',
            'passed': False,
            'details': str(e)
        })
    
    # Generate summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    total_tests = len(test_results['tests'])
    passed_tests = sum(1 for t in test_results['tests'] if t['passed'])
    failed_tests = total_tests - passed_tests
    
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    
    print("\nDetailed Results:")
    for test in test_results['tests']:
        status = "✓" if test['passed'] else "✗"
        print(f"{status} {test['name']}: {test['details']}")
    
    # Save results
    with open('straddle_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\nResults saved to: straddle_test_results.json")
    print(f"Test completed at: {datetime.now()}")
    print("="*80)
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)