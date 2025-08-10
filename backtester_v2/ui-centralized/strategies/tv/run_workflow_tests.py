#!/usr/bin/env python3
"""
TV Strategy Workflow Test Runner
Tests the complete 6-file workflow without requiring HeavyDB connection
Focuses on configuration parsing, signal processing, and workflow validation
"""

import sys
import time
from pathlib import Path
from datetime import datetime
import os
import pandas as pd

# Add current directory to path
sys.path.insert(0, '.')

def print_header(title):
    """Print formatted header"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print('='*80)

def print_test(test_name):
    """Print test name"""
    print(f"\n🧪 {test_name}")

def print_success(message):
    """Print success message"""
    print(f"   ✅ {message}")

def print_error(message):
    """Print error message"""
    print(f"   ❌ {message}")

def print_info(message):
    """Print info message"""
    print(f"   ℹ️  {message}")

def main():
    """Run workflow tests for TV strategy"""
    
    print_header("TV STRATEGY 6-FILE WORKFLOW TESTING")
    print(f"⏰ Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🔄 Testing complete 6-file workflow")
    print("📁 Using REAL configuration files")
    print("✅ Focus on workflow validation without HeavyDB dependency")
    
    total_start_time = time.time()
    tests_passed = 0
    tests_failed = 0
    
    # Import modules
    try:
        from parser import TVParser
        from signal_processor import SignalProcessor
        from query_builder import TVQueryBuilder
        from processor import TVProcessor
        from excel_to_yaml_converter import TVExcelToYAMLConverter
        from parallel_processor import TVParallelProcessor, TVBatchProcessor
        
        parser = TVParser()
        signal_processor = SignalProcessor()
        query_builder = TVQueryBuilder()
        processor = TVProcessor()
        yaml_converter = TVExcelToYAMLConverter()
        
    except ImportError as e:
        print_error(f"Module import failed: {e}")
        return 1
    
    # Define real configuration files
    base_path = Path('../../configurations/data/prod/tv')
    real_config_files = {
        'tv_master': base_path / 'TV_CONFIG_MASTER_1.0.0.xlsx',
        'signals': base_path / 'TV_CONFIG_SIGNALS_1.0.0.xlsx',
        'portfolio_long': base_path / 'TV_CONFIG_PORTFOLIO_LONG_1.0.0.xlsx',
        'portfolio_short': base_path / 'TV_CONFIG_PORTFOLIO_SHORT_1.0.0.xlsx',
        'portfolio_manual': base_path / 'TV_CONFIG_PORTFOLIO_MANUAL_1.0.0.xlsx',
        'strategy': base_path / 'TV_CONFIG_STRATEGY_1.0.0.xlsx'
    }
    
    # Verify all files exist
    print_test("Verifying 6-file hierarchy exists")
    try:
        for file_key, file_path in real_config_files.items():
            assert file_path.exists(), f"Missing {file_key}: {file_path}"
            print_success(f"Found {file_key}: {file_path.name}")
        
        tests_passed += 1
        
    except Exception as e:
        print_error(f"File verification failed: {e}")
        tests_failed += 1
        return 1
    
    # Test 1: Complete 6-File Parsing
    print_test("Testing complete 6-file configuration parsing")
    try:
        # Parse TV Master
        tv_config_result = parser.parse_tv_settings(str(real_config_files['tv_master']))
        tv_config = tv_config_result['settings'][0]
        
        assert tv_config['name'] == 'TV_Backtest_Sample'
        assert tv_config['enabled'] is True
        assert tv_config['start_date'].year == 2024
        assert tv_config['end_date'].year == 2024
        
        print_success("TV Master configuration parsed successfully")
        print_info(f"Config name: {tv_config['name']}")
        print_info(f"Date range: {tv_config['start_date']} to {tv_config['end_date']}")
        
        # Parse Signals
        signals = parser.parse_signals(str(real_config_files['signals']), tv_config['signal_date_format'])
        assert len(signals) == 4
        assert signals[0]['trade_no'] == 'T001'
        assert signals[1]['trade_no'] == 'T001'
        assert signals[2]['trade_no'] == 'T002'
        assert signals[3]['trade_no'] == 'T002'
        
        print_success(f"Signals parsed: {len(signals)}")
        print_info(f"Unique trades: {len(set(s['trade_no'] for s in signals))}")
        
        # Process Signals
        processed_signals = signal_processor.process_signals(signals, tv_config)
        assert len(processed_signals) > 0
        assert all('entry_date' in sig for sig in processed_signals)
        assert all('exit_date' in sig for sig in processed_signals)
        assert all('signal_direction' in sig for sig in processed_signals)
        
        print_success(f"Processed signals: {len(processed_signals)}")
        
        tests_passed += 1
        
    except Exception as e:
        print_error(f"6-file parsing failed: {e}")
        tests_failed += 1
    
    # Test 2: Portfolio Configuration Validation
    print_test("Testing portfolio configuration parsing")
    try:
        portfolio_configs = {}
        
        # Parse all portfolio files
        for portfolio_type in ['long', 'short', 'manual']:
            portfolio_key = f'portfolio_{portfolio_type}'
            portfolio_path = real_config_files[portfolio_key]
            
            # Parse PortfolioSetting
            portfolio_df = pd.read_excel(str(portfolio_path), sheet_name='PortfolioSetting', engine='openpyxl')
            
            portfolio_config = {
                'capital': int(portfolio_df.iloc[0]['Capital']),
                'max_risk': int(portfolio_df.iloc[0]['MaxRisk']),
                'max_positions': int(portfolio_df.iloc[0]['MaxPositions']),
                'risk_per_trade': int(portfolio_df.iloc[0]['RiskPerTrade']),
                'use_kelly': portfolio_df.iloc[0]['UseKellyCriterion'],
                'rebalance_freq': portfolio_df.iloc[0]['RebalanceFrequency']
            }
            
            portfolio_configs[portfolio_type] = portfolio_config
            
            # Validate values
            assert portfolio_config['capital'] == 1000000
            assert portfolio_config['max_risk'] == 5
            assert portfolio_config['max_positions'] == 5
            assert portfolio_config['risk_per_trade'] == 2
            
            print_success(f"{portfolio_type.capitalize()} portfolio validated")
        
        print_info("All portfolio configurations validated")
        print_info(f"Capital: ₹{portfolio_configs['long']['capital']:,}")
        print_info(f"Max positions: {portfolio_configs['long']['max_positions']}")
        print_info(f"Risk per trade: {portfolio_configs['long']['risk_per_trade']}%")
        
        tests_passed += 1
        
    except Exception as e:
        print_error(f"Portfolio parsing failed: {e}")
        tests_failed += 1
    
    # Test 3: Strategy Configuration Validation
    print_test("Testing TBS strategy configuration")
    try:
        strategy_path = real_config_files['strategy']
        
        # Parse GeneralParameter
        general_df = pd.read_excel(str(strategy_path), sheet_name='GeneralParameter', engine='openpyxl')
        general_params = general_df.iloc[0]
        
        assert general_params['StrategyName'] == 'TV_Strategy_Sample'
        assert general_params['Underlying'] == 'SPOT'
        assert general_params['Index'] == 'NIFTY'
        assert general_params['DTE'] == 0
        
        # Parse LegParameter
        legs_df = pd.read_excel(str(strategy_path), sheet_name='LegParameter', engine='openpyxl')
        
        assert len(legs_df) > 0
        first_leg = legs_df.iloc[0]
        assert first_leg['LegID'] == 'leg1'
        assert first_leg['Instrument'] == 'call'
        assert first_leg['Transaction'] == 'sell'
        assert first_leg['StrikeMethod'] == 'ATM'
        
        print_success("Strategy configuration validated")
        print_info(f"Strategy: {general_params['StrategyName']}")
        print_info(f"Index: {general_params['Index']}")
        print_info(f"Legs configured: {len(legs_df)}")
        
        tests_passed += 1
        
    except Exception as e:
        print_error(f"Strategy parsing failed: {e}")
        tests_failed += 1
    
    # Test 4: Signal Processing Workflow
    print_test("Testing signal processing workflow")
    try:
        # Test signal pairing
        long_signals = [s for s in signals if 'Long' in s['signal_type']]
        short_signals = [s for s in signals if 'Short' in s['signal_type']]
        
        assert len(long_signals) == 2  # Entry and Exit
        assert len(short_signals) == 2  # Entry and Exit
        
        # Test processed signal structure
        for signal in processed_signals:
            assert 'trade_no' in signal
            assert 'signal_direction' in signal
            assert 'entry_date' in signal
            assert 'entry_time' in signal
            assert 'exit_date' in signal
            assert 'exit_time' in signal
            assert 'lots' in signal
            assert 'portfolio_file' in signal
        
        # Validate portfolio file assignments
        long_processed = [s for s in processed_signals if s['signal_direction'] == 'LONG']
        short_processed = [s for s in processed_signals if s['signal_direction'] == 'SHORT']
        
        for sig in long_processed:
            assert sig['portfolio_file'] == tv_config['long_portfolio_file_path']
        
        for sig in short_processed:
            assert sig['portfolio_file'] == tv_config['short_portfolio_file_path']
        
        print_success("Signal processing workflow validated")
        print_info(f"Long trades: {len(long_processed)}")
        print_info(f"Short trades: {len(short_processed)}")
        
        tests_passed += 1
        
    except Exception as e:
        print_error(f"Signal processing failed: {e}")
        tests_failed += 1
    
    # Test 5: Excel to YAML Conversion
    print_test("Testing Excel to YAML conversion for 6-file hierarchy")
    try:
        # Convert complete hierarchy
        yaml_result = yaml_converter.convert_complete_hierarchy_to_yaml(real_config_files)
        
        assert 'tv_complete_configuration' in yaml_result
        config = yaml_result['tv_complete_configuration']
        
        # Validate all components present
        required_components = ['tv_master', 'signals', 'portfolio_long', 
                             'portfolio_short', 'portfolio_manual', 'tbs_strategy']
        
        for component in required_components:
            assert component in config, f"Missing {component} in YAML"
        
        # Validate YAML structure
        assert 'metadata' in config
        assert 'conversion_timestamp' in config['metadata']
        assert config['tv_master']['master_settings']['name'] == 'TV_Backtest_Sample'
        assert len(config['signals']['signals']) == 4
        
        # Test YAML serialization
        import yaml
        yaml_string = yaml.dump(yaml_result, default_flow_style=False, allow_unicode=True)
        assert len(yaml_string) > 1000
        
        print_success("Excel to YAML conversion successful")
        print_info(f"Components converted: {', '.join(required_components)}")
        print_info(f"YAML size: {len(yaml_string):,} characters")
        
        tests_passed += 1
        
    except Exception as e:
        print_error(f"YAML conversion failed: {e}")
        tests_failed += 1
    
    # Test 6: Query Builder Validation
    print_test("Testing query builder for signals")
    try:
        # Test query generation for each processed signal
        queries_generated = 0
        
        for signal in processed_signals[:2]:  # Test first 2 signals
            # Build simple query
            query = query_builder.build_query(signal, tv_config)
            
            assert isinstance(query, str)
            assert len(query) > 0
            assert signal['trade_no'] in query
            assert signal['signal_direction'] in query
            
            queries_generated += 1
        
        print_success(f"Query builder validated")
        print_info(f"Queries generated: {queries_generated}")
        
        tests_passed += 1
        
    except Exception as e:
        print_error(f"Query builder test failed: {e}")
        tests_failed += 1
    
    # Test 7: Parallel Processing Integration
    print_test("Testing parallel processing integration")
    try:
        parallel_processor = TVParallelProcessor(max_workers=2)
        
        # Create a simple job
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        job = parallel_processor.create_job(
            job_id="workflow_test_job",
            tv_config_path=str(real_config_files['tv_master']),
            signal_files=[str(real_config_files['signals'])],
            output_directory=temp_dir,
            priority=1
        )
        
        assert job.job_id == "workflow_test_job"
        assert job.status == 'pending'
        assert len(job.signal_files) == 1
        
        # Process job
        results = parallel_processor.process_job(job)
        
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].output_file is not None
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        
        print_success("Parallel processing integration validated")
        print_info(f"Job processed successfully")
        print_info(f"Output generated: {Path(results[0].output_file).name}")
        
        tests_passed += 1
        
    except Exception as e:
        print_error(f"Parallel processing test failed: {e}")
        tests_failed += 1
    
    # Test 8: Performance Validation
    print_test("Testing workflow performance")
    try:
        start_time = time.time()
        
        # Parse all files
        parser.parse_tv_settings(str(real_config_files['tv_master']))
        parser.parse_signals(str(real_config_files['signals']), '%Y%m%d %H%M%S')
        
        for portfolio_key in ['portfolio_long', 'portfolio_short', 'portfolio_manual']:
            pd.read_excel(str(real_config_files[portfolio_key]), sheet_name='PortfolioSetting', engine='openpyxl')
        
        pd.read_excel(str(real_config_files['strategy']), sheet_name='GeneralParameter', engine='openpyxl')
        pd.read_excel(str(real_config_files['strategy']), sheet_name='LegParameter', engine='openpyxl')
        
        parsing_time = time.time() - start_time
        
        print_success(f"Performance validation passed")
        print_info(f"6-file parsing time: {parsing_time:.3f}s")
        print_info(f"Performance requirement: < 3.0s")
        
        assert parsing_time < 3.0, f"Parsing too slow: {parsing_time:.3f}s"
        
        tests_passed += 1
        
    except Exception as e:
        print_error(f"Performance test failed: {e}")
        tests_failed += 1
    
    # Final summary
    total_time = time.time() - total_start_time
    
    print_header("WORKFLOW TEST RESULTS SUMMARY")
    print(f"⏱️  Total execution time: {total_time:.3f} seconds")
    print(f"✅ Tests passed: {tests_passed}")
    print(f"❌ Tests failed: {tests_failed}")
    print(f"📊 Success rate: {(tests_passed/(tests_passed+tests_failed)*100):.1f}%")
    
    if tests_failed == 0:
        print("\n🎉 ALL WORKFLOW TESTS PASSED!")
        print("✅ Complete 6-file hierarchy validated")
        print("✅ Configuration parsing working perfectly")
        print("✅ Signal processing workflow confirmed")
        print("✅ Portfolio allocation validated")
        print("✅ Excel to YAML conversion functional")
        print("✅ Query generation validated")
        print("✅ Parallel processing integrated")
        print("✅ Performance requirements met")
        
        print("\n📋 WORKFLOW VALIDATION SUMMARY:")
        print(f"   • Configuration Files: 6-file hierarchy")
        print(f"   • Signals: {len(signals)} parsed and processed")
        print(f"   • Portfolios: Long, Short, Manual validated")
        print(f"   • Strategy: TBS configuration validated")
        print(f"   • Performance: All operations < 3 seconds")
        
        print("\n🚀 PHASE 2 COMPLETED:")
        print("   • 6-file workflow fully tested")
        print("   • Ready for HeavyDB integration")
        print("   • Ready for golden format validation")
        
        return 0
    else:
        print(f"\n⚠️  {tests_failed} TEST(S) FAILED!")
        print("Please review the errors above and fix issues before proceeding.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)