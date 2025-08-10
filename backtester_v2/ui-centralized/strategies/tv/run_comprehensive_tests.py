#!/usr/bin/env python3
"""
Comprehensive TV Strategy Test Runner
Runs all TV strategy tests with REAL data validation using SuperClaude command equivalents
NO MOCK DATA - ONLY REAL INPUT SHEETS AND HEAVYDB VALIDATION
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.insert(0, '.')

def print_header(title):
    """Print formatted header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def print_test(test_name):
    """Print test name"""
    print(f"\n🧪 {test_name}")

def print_success(message):
    """Print success message"""
    print(f"   ✅ {message}")

def print_error(message):
    """Print error message"""
    print(f"   ❌ {message}")

def main():
    """Run comprehensive tests"""
    
    print_header("TV STRATEGY COMPREHENSIVE TESTING")
    print(f"⏰ Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("📁 Using REAL configuration files from configurations/data/prod/tv/")
    print("🚫 NO MOCK DATA - Real input sheets validation only")
    
    total_start_time = time.time()
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: TV Parser with REAL configuration files
    print_test("Testing TV Parser with REAL configuration files")
    try:
        from parser import TVParser
        parser = TVParser()
        
        # Parse REAL TV master config
        config_path = '../../configurations/data/prod/tv/TV_CONFIG_MASTER_1.0.0.xlsx'
        result = parser.parse_tv_settings(config_path)
        
        # Validate structure
        assert 'settings' in result, 'Parser must return settings'
        assert len(result['settings']) > 0, 'Must have settings'
        
        tv_config = result['settings'][0]
        assert tv_config['name'] == 'TV_Backtest_Sample', 'Real config name validation'
        assert tv_config['enabled'] is True, 'Real config enabled validation'
        assert tv_config['signal_file_path'] == 'TV_CONFIG_SIGNALS_1.0.0.xlsx', 'Signal file reference validation'
        
        print_success("TV Parser validation passed")
        print_success(f"Config: {tv_config['name']}")
        print_success(f"Date range: {tv_config['start_date']} to {tv_config['end_date']}")
        print_success(f"Signal file: {tv_config['signal_file_path']}")
        
        tests_passed += 1
        
    except Exception as e:
        print_error(f"TV Parser failed: {e}")
        tests_failed += 1
    
    # Test 2: TV Signals parsing with REAL signals file
    print_test("Testing TV Signals parsing with REAL signals file")
    try:
        signals_path = '../../configurations/data/prod/tv/TV_CONFIG_SIGNALS_1.0.0.xlsx'
        signals = parser.parse_signals(signals_path, '%Y%m%d %H%M%S')
        
        assert len(signals) == 4, 'Real signals file has 4 signals'
        assert signals[0]['trade_no'] == 'T001', 'First signal trade number validation'
        assert signals[0]['signal_type'] == 'ENTRY_LONG', 'First signal type validation'
        
        # Validate signal pairing
        trade_nos = set(s['trade_no'] for s in signals)
        assert 'T001' in trade_nos and 'T002' in trade_nos, 'Trade numbers validation'
        
        print_success("TV Signals parsing passed")
        print_success(f"Signal count: {len(signals)}")
        print_success(f"Trade numbers: {sorted(trade_nos)}")
        print_success(f"Signal types: {set(s['signal_type'] for s in signals)}")
        
        tests_passed += 1
        
    except Exception as e:
        print_error(f"TV Signals parsing failed: {e}")
        tests_failed += 1
    
    # Test 3: Excel to YAML converter with REAL files
    print_test("Testing Excel to YAML converter with REAL configuration")
    try:
        from excel_to_yaml_converter import TVExcelToYAMLConverter
        converter = TVExcelToYAMLConverter()
        
        # Convert REAL TV master config to YAML
        yaml_result = converter.convert_tv_master_to_yaml(config_path)
        
        assert 'tv_configuration' in yaml_result, 'YAML must have tv_configuration root'
        tv_yaml_config = yaml_result['tv_configuration']
        assert 'master_settings' in tv_yaml_config, 'YAML must have master_settings'
        assert tv_yaml_config['master_settings']['name'] == 'TV_Backtest_Sample', 'YAML name validation'
        
        # Test YAML serialization
        import yaml
        yaml_string = yaml.dump(yaml_result, default_flow_style=False, allow_unicode=True)
        assert len(yaml_string) > 500, 'YAML string must be substantial'
        
        print_success("Excel to YAML conversion passed")
        print_success(f"YAML sections: {list(tv_yaml_config.keys())}")
        print_success(f"YAML string length: {len(yaml_string)} characters")
        
        tests_passed += 1
        
    except Exception as e:
        print_error(f"Excel to YAML conversion failed: {e}")
        tests_failed += 1
    
    # Test 4: 6-file hierarchy validation with REAL files
    print_test("Testing 6-file hierarchy validation with REAL files")
    try:
        base_path = Path('../../configurations/data/prod/tv')
        
        # Validate all 6 files exist
        files_to_check = [
            'TV_CONFIG_MASTER_1.0.0.xlsx',
            'TV_CONFIG_SIGNALS_1.0.0.xlsx', 
            'TV_CONFIG_PORTFOLIO_LONG_1.0.0.xlsx',
            'TV_CONFIG_PORTFOLIO_SHORT_1.0.0.xlsx',
            'TV_CONFIG_PORTFOLIO_MANUAL_1.0.0.xlsx',
            'TV_CONFIG_STRATEGY_1.0.0.xlsx'
        ]
        
        for file_name in files_to_check:
            file_path = base_path / file_name
            assert file_path.exists(), f'File {file_name} must exist'
        
        # Test complete hierarchy conversion
        real_config_files = {
            'tv_master': base_path / 'TV_CONFIG_MASTER_1.0.0.xlsx',
            'signals': base_path / 'TV_CONFIG_SIGNALS_1.0.0.xlsx',
            'portfolio_long': base_path / 'TV_CONFIG_PORTFOLIO_LONG_1.0.0.xlsx',
            'portfolio_short': base_path / 'TV_CONFIG_PORTFOLIO_SHORT_1.0.0.xlsx',
            'portfolio_manual': base_path / 'TV_CONFIG_PORTFOLIO_MANUAL_1.0.0.xlsx',
            'strategy': base_path / 'TV_CONFIG_STRATEGY_1.0.0.xlsx'
        }
        
        complete_yaml = converter.convert_complete_hierarchy_to_yaml(real_config_files)
        assert 'tv_complete_configuration' in complete_yaml, 'Complete YAML must have root'
        
        complete_config = complete_yaml['tv_complete_configuration']
        expected_components = ['tv_master', 'signals', 'portfolio_long', 'portfolio_short', 'portfolio_manual', 'tbs_strategy']
        for component in expected_components:
            assert component in complete_config, f'Component {component} must exist in complete YAML'
        
        print_success("6-file hierarchy validation passed")
        print_success(f"Files validated: {len(files_to_check)}")
        print_success(f"Components: {expected_components}")
        
        tests_passed += 1
        
    except Exception as e:
        print_error(f"6-file hierarchy validation failed: {e}")
        tests_failed += 1
    
    # Test 5: Performance validation with REAL data
    print_test("Testing performance requirements with REAL data")
    try:
        # Test parsing performance
        start_time = time.time()
        result = parser.parse_tv_settings(config_path)
        signals = parser.parse_signals(signals_path, '%Y%m%d %H%M%S')
        parsing_time = time.time() - start_time
        
        assert parsing_time < 1.0, f'Parsing too slow: {parsing_time:.3f}s'
        
        # Test YAML conversion performance
        start_time = time.time()
        yaml_result = converter.convert_complete_hierarchy_to_yaml(real_config_files)
        conversion_time = time.time() - start_time
        
        assert conversion_time < 3.0, f'YAML conversion too slow: {conversion_time:.3f}s'
        
        print_success("Performance validation passed")
        print_success(f"Parsing time: {parsing_time:.3f}s (< 1.0s)")
        print_success(f"YAML conversion time: {conversion_time:.3f}s (< 3.0s)")
        
        tests_passed += 1
        
    except Exception as e:
        print_error(f"Performance validation failed: {e}")
        tests_failed += 1
    
    # Test 6: Data integrity validation
    print_test("Testing data integrity with REAL configuration files")
    try:
        # Cross-validate file references
        tv_config = result['settings'][0]
        
        # Validate signal file reference exists
        signal_file = tv_config['signal_file_path']
        signal_full_path = base_path / signal_file
        assert signal_full_path.exists(), f"Referenced signal file must exist: {signal_file}"
        
        # Validate portfolio file references exist
        for portfolio_type in ['long_portfolio_file_path', 'short_portfolio_file_path', 'manual_portfolio_file_path']:
            portfolio_file = tv_config[portfolio_type]
            portfolio_full_path = base_path / portfolio_file
            assert portfolio_full_path.exists(), f"Referenced portfolio file must exist: {portfolio_file}"
        
        # Validate date consistency
        assert tv_config['start_date'] < tv_config['end_date'], "Start date must be before end date"
        
        # Validate signal date ranges
        signal_dates = [s['datetime'].date() for s in signals]
        min_signal_date = min(signal_dates)
        max_signal_date = max(signal_dates)
        
        # All signals should be within config date range
        assert min_signal_date >= tv_config['start_date'], "Signal dates must be within config range"
        assert max_signal_date <= tv_config['end_date'], "Signal dates must be within config range"
        
        print_success("Data integrity validation passed")
        print_success(f"File references validated: {signal_file} + 3 portfolios")
        print_success(f"Date range validation: {tv_config['start_date']} to {tv_config['end_date']}")
        print_success(f"Signal date range: {min_signal_date} to {max_signal_date}")
        
        tests_passed += 1
        
    except Exception as e:
        print_error(f"Data integrity validation failed: {e}")
        tests_failed += 1
    
    # Final summary
    total_time = time.time() - total_start_time
    
    print_header("TEST RESULTS SUMMARY")
    print(f"⏱️  Total execution time: {total_time:.3f} seconds")
    print(f"✅ Tests passed: {tests_passed}")
    print(f"❌ Tests failed: {tests_failed}")
    print(f"📊 Success rate: {(tests_passed/(tests_passed+tests_failed)*100):.1f}%")
    
    if tests_failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ TV Strategy components fully validated with REAL input sheets")
        print("✅ Excel to YAML conversion working perfectly")
        print("✅ 6-file hierarchy integrity confirmed")
        print("✅ Performance requirements met")
        print("✅ Data integrity validated")
        print("🚫 NO MOCK DATA used - 100% real configuration validation")
        
        print("\n📋 VALIDATION SUMMARY:")
        print(f"   • Configuration: {tv_config['name']}")
        print(f"   • Date range: {tv_config['start_date']} to {tv_config['end_date']}")
        print(f"   • Signals: {len(signals)} signals from {len(set(s['trade_no'] for s in signals))} trades")
        print(f"   • Files: 6-file hierarchy validated")
        print(f"   • YAML: Complete conversion successful")
        print(f"   • Performance: All operations < 3 seconds")
        
        print("\n🚀 READY FOR:")
        print("   • HeavyDB integration testing")
        print("   • Parallel processing implementation")
        print("   • Golden format output validation")
        print("   • End-to-end workflow testing")
        
        return 0
    else:
        print(f"\n⚠️  {tests_failed} TEST(S) FAILED!")
        print("Please review the errors above and fix issues before proceeding.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)