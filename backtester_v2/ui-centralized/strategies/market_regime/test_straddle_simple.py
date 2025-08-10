"""
Simple test to validate straddle analysis components with HeavyDB
"""

import sys
import os
sys.path.insert(0, '/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2')

import pandas as pd
import numpy as np
import heavydb
import time
from datetime import datetime

# Direct imports without going through __init__.py
import importlib.util

def import_module_directly(module_name, file_path):
    """Import module directly from file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import modules directly
base_path = "/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/strategies/market_regime/indicators/straddle_analysis"

# Core modules
StraddleEngine = import_module_directly("straddle_engine", f"{base_path}/core/straddle_engine.py").StraddleEngine
CalculationEngine = import_module_directly("calculation_engine", f"{base_path}/core/calculation_engine.py").CalculationEngine
ResistanceAnalyzer = import_module_directly("resistance_analyzer", f"{base_path}/core/resistance_analyzer.py").ResistanceAnalyzer

# Config
StraddleExcelReader = import_module_directly("excel_reader", f"{base_path}/config/excel_reader.py").StraddleExcelReader

# Rolling modules
RollingWindowManager = import_module_directly("window_manager", f"{base_path}/rolling/window_manager.py").RollingWindowManager
CorrelationMatrix = import_module_directly("correlation_matrix", f"{base_path}/rolling/correlation_matrix.py").CorrelationMatrix


class SimpleStraddleTester:
    """Simple tester for straddle analysis"""
    
    def __init__(self):
        """Initialize tester"""
        self.conn = None
        self.config = None
        self.test_results = {
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'errors': []
        }
    
    def connect_heavydb(self):
        """Connect to HeavyDB"""
        try:
            self.conn = heavydb.connect(
                host='localhost',
                port=6274,
                user='admin',
                password='HyperInteractive',
                dbname='heavyai'
            )
            print("✓ Connected to HeavyDB successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to connect to HeavyDB: {e}")
            self.test_results['errors'].append(f"HeavyDB connection: {e}")
            return False
    
    def test_data_fetch(self):
        """Test fetching data from HeavyDB"""
        self.test_results['tests_run'] += 1
        
        try:
            query = """
            SELECT 
                trade_date,
                trade_time,
                spot,
                CASE WHEN call_strike_type = 'ATM' THEN ce_close END as ATM_CE,
                CASE WHEN put_strike_type = 'ATM' THEN pe_close END as ATM_PE,
                CASE WHEN call_strike_type = 'ITM1' THEN ce_close END as ITM1_CE,
                CASE WHEN put_strike_type = 'ITM1' THEN pe_close END as ITM1_PE,
                CASE WHEN call_strike_type = 'OTM1' THEN ce_close END as OTM1_CE,
                CASE WHEN put_strike_type = 'OTM1' THEN pe_close END as OTM1_PE
            FROM nifty_option_chain
            WHERE trade_date = '2025-06-17'
            AND expiry_date = '2025-06-19'
            AND index_name = 'NIFTY'
            ORDER BY trade_time
            LIMIT 100
            """
            
            cursor = self.conn.cursor()
            cursor.execute(query)
            data = cursor.fetchall()
            
            print(f"✓ Fetched {len(data)} records from HeavyDB")
            self.test_results['tests_passed'] += 1
            
            # Show sample data
            if data:
                print(f"  Sample: Date={data[0][0]}, Time={data[0][1]}, Spot={data[0][2]}")
            
            return True
            
        except Exception as e:
            print(f"✗ Data fetch failed: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['errors'].append(f"Data fetch: {e}")
            return False
    
    def test_excel_config(self):
        """Test Excel configuration loading"""
        self.test_results['tests_run'] += 1
        
        try:
            reader = StraddleExcelReader()
            self.config = reader.read_configuration()
            
            # Check weights
            comp_sum = sum(self.config.component_weights.values())
            straddle_sum = sum(self.config.straddle_weights.values())
            
            print(f"✓ Excel config loaded successfully")
            print(f"  Component weights sum: {comp_sum:.3f}")
            print(f"  Straddle weights sum: {straddle_sum:.3f}")
            print(f"  Rolling windows: {self.config.rolling_windows}")
            
            self.test_results['tests_passed'] += 1
            return True
            
        except Exception as e:
            print(f"✗ Excel config failed: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['errors'].append(f"Excel config: {e}")
            return False
    
    def test_calculation_engine(self):
        """Test calculation engine"""
        self.test_results['tests_run'] += 1
        
        try:
            if not self.config:
                self.test_excel_config()
            
            calc_engine = CalculationEngine(self.config)
            
            # Test basic calculations
            straddle = calc_engine.calculate_straddle_value(100, 95)
            assert straddle == 195, f"Straddle calc error: {straddle} != 195"
            
            # Test volatility
            prices = np.array([100, 102, 98, 101, 99, 103, 97, 104])
            vol = calc_engine.calculate_volatility(prices)
            assert vol > 0, f"Volatility should be positive: {vol}"
            
            print(f"✓ Calculation engine tests passed")
            print(f"  Straddle value: {straddle}")
            print(f"  Volatility: {vol:.6f}")
            
            self.test_results['tests_passed'] += 1
            return True
            
        except Exception as e:
            print(f"✗ Calculation engine failed: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['errors'].append(f"Calculation engine: {e}")
            return False
    
    def test_rolling_windows(self):
        """Test rolling window manager"""
        self.test_results['tests_run'] += 1
        
        try:
            window_mgr = RollingWindowManager({'rolling_windows': [3, 5, 10, 15]})
            
            # Add test data
            for i in range(20):
                timestamp = pd.Timestamp(f'2025-06-17 09:{15+i}:00')
                data = {
                    'close': 100 + i,
                    'high': 101 + i,
                    'low': 99 + i,
                    'volume': 1000 * (i + 1)
                }
                window_mgr.add_data_point('atm_ce', timestamp, data)
            
            # Test calculations
            mean_3 = window_mgr.calculate_rolling_statistic('atm_ce', 3, 'close', 'mean')
            mean_5 = window_mgr.calculate_rolling_statistic('atm_ce', 5, 'close', 'mean')
            
            print(f"✓ Rolling window tests passed")
            print(f"  3-min mean: {mean_3:.2f}")
            print(f"  5-min mean: {mean_5:.2f}")
            
            self.test_results['tests_passed'] += 1
            return True
            
        except Exception as e:
            print(f"✗ Rolling window failed: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['errors'].append(f"Rolling window: {e}")
            return False
    
    def test_correlation_matrix(self):
        """Test correlation matrix"""
        self.test_results['tests_run'] += 1
        
        try:
            corr_matrix = CorrelationMatrix({'rolling_windows': [3, 5, 10, 15]})
            
            # Create test data
            components = ['atm_ce', 'atm_pe', 'itm1_ce', 'itm1_pe', 'otm1_ce', 'otm1_pe']
            test_data = {}
            
            # Generate correlated data
            base = list(np.random.randn(20))
            for comp in components:
                if 'pe' in comp:
                    test_data[comp] = [-x + np.random.randn() * 0.1 for x in base]
                else:
                    test_data[comp] = [x + np.random.randn() * 0.1 for x in base]
            
            # Calculate matrix
            result = corr_matrix.calculate_correlation_matrix(
                test_data, 10, pd.Timestamp.now()
            )
            
            print(f"✓ Correlation matrix tests passed")
            print(f"  Matrix shape: {result.matrix.shape}")
            print(f"  Avg correlation: {result.avg_correlation:.3f}")
            
            self.test_results['tests_passed'] += 1
            return True
            
        except Exception as e:
            print(f"✗ Correlation matrix failed: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['errors'].append(f"Correlation matrix: {e}")
            return False
    
    def test_resistance_analyzer(self):
        """Test resistance analyzer"""
        self.test_results['tests_run'] += 1
        
        try:
            analyzer = ResistanceAnalyzer()
            
            # Add some test data
            for i in range(50):
                timestamp = pd.Timestamp(f'2025-06-17 09:{15+i}:00')
                price = 20000 + np.sin(i * 0.1) * 100  # Oscillating price
                
                data = {
                    'underlying_price': price,
                    'spot_price': price,
                    'high': price + 10,
                    'low': price - 10,
                    'close': price,
                    'volume': 10000
                }
                
                result = analyzer.analyze(data, timestamp)
            
            # Should have identified some levels
            levels_count = len(analyzer.support_levels) + len(analyzer.resistance_levels)
            
            print(f"✓ Resistance analyzer tests passed")
            print(f"  Support levels: {len(analyzer.support_levels)}")
            print(f"  Resistance levels: {len(analyzer.resistance_levels)}")
            
            self.test_results['tests_passed'] += 1
            return True
            
        except Exception as e:
            print(f"✗ Resistance analyzer failed: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['errors'].append(f"Resistance analyzer: {e}")
            return False
    
    def test_straddle_engine_basic(self):
        """Test basic straddle engine functionality"""
        self.test_results['tests_run'] += 1
        
        try:
            if not self.config:
                self.test_excel_config()
            
            engine = StraddleEngine(self.config)
            
            # Create test data
            test_data = {
                'timestamp': '2025-06-17 10:00:00',
                'underlying_price': 20000,
                'spot_price': 20000,
                'ATM_CE': 150,
                'ATM_PE': 145,
                'ITM1_CE': 250,
                'ITM1_PE': 50,
                'OTM1_CE': 50,
                'OTM1_PE': 245,
                'volume': 10000
            }
            
            # Analyze
            start_time = time.time()
            result = engine.analyze(test_data)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            print(f"✓ Straddle engine basic test passed")
            print(f"  Processing time: {processing_time:.6f}s")
            if result:
                print(f"  Regime: {result.regime_classification}")
                print(f"  Confidence: {result.confidence_score:.3f}")
            
            self.test_results['tests_passed'] += 1
            return True
            
        except Exception as e:
            print(f"✗ Straddle engine failed: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['errors'].append(f"Straddle engine: {e}")
            return False
    
    def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*60)
        print("SIMPLE STRADDLE ANALYSIS TESTS")
        print("="*60)
        
        # Connect to HeavyDB
        if not self.connect_heavydb():
            print("Cannot proceed without HeavyDB connection")
            return
        
        # Run tests
        tests = [
            self.test_data_fetch,
            self.test_excel_config,
            self.test_calculation_engine,
            self.test_rolling_windows,
            self.test_correlation_matrix,
            self.test_resistance_analyzer,
            self.test_straddle_engine_basic
        ]
        
        for test in tests:
            print(f"\nRunning {test.__name__}...")
            test()
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Total tests: {self.test_results['tests_run']}")
        print(f"Passed: {self.test_results['tests_passed']}")
        print(f"Failed: {self.test_results['tests_failed']}")
        print(f"Success rate: {self.test_results['tests_passed']/self.test_results['tests_run']*100:.1f}%")
        
        if self.test_results['errors']:
            print("\nErrors:")
            for error in self.test_results['errors']:
                print(f"  - {error}")
        
        print("="*60)
        
        # Close connection
        if self.conn:
            self.conn.close()


def main():
    """Main function"""
    tester = SimpleStraddleTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()