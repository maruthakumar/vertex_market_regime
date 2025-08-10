"""
Simple test script for Excel Configuration System
Tests core functionality without complex imports
"""

import os
import sys
import time
import pandas as pd
import yaml
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_excel_file_discovery():
    """Test discovery of Excel files"""
    logger.info("🔍 Testing Excel file discovery")
    
    base_path = Path("/srv/samba/shared/bt/backtester_stable/worktrees/ui-refactor/ui-centralized/backtester_v2/configurations")
    prod_dir = base_path / "data" / "prod"
    
    if not prod_dir.exists():
        logger.error(f"Production directory not found: {prod_dir}")
        return False
    
    # Count files by strategy
    strategy_counts = {}
    total_files = 0
    
    for strategy_dir in prod_dir.iterdir():
        if strategy_dir.is_dir():
            strategy_type = strategy_dir.name
            excel_files = []
            
            for pattern in ['*.xlsx', '*.xls', '*.xlsm']:
                excel_files.extend(strategy_dir.glob(pattern))
            
            strategy_counts[strategy_type] = len(excel_files)
            total_files += len(excel_files)
            
            logger.info(f"  📁 {strategy_type}: {len(excel_files)} files")
    
    logger.info(f"📊 Total: {total_files} Excel files across {len(strategy_counts)} strategies")
    return total_files > 0

def test_excel_to_yaml_conversion():
    """Test basic Excel to YAML conversion"""
    logger.info("📄 Testing Excel to YAML conversion")
    
    base_path = Path("/srv/samba/shared/bt/backtester_stable/worktrees/ui-refactor/ui-centralized/backtester_v2/configurations")
    
    # Find a test file
    tbs_dir = base_path / "data" / "prod" / "tbs"
    
    if not tbs_dir.exists():
        logger.error(f"TBS directory not found: {tbs_dir}")
        return False
    
    excel_files = list(tbs_dir.glob("*.xlsx"))
    if not excel_files:
        logger.error("No Excel files found in TBS directory")
        return False
    
    test_file = excel_files[0]
    logger.info(f"  📄 Testing with: {test_file.name}")
    
    start_time = time.time()
    
    try:
        # Read Excel file
        excel_file = pd.ExcelFile(test_file)
        sheets = excel_file.sheet_names
        
        logger.info(f"  📋 Found {len(sheets)} sheets: {sheets}")
        
        # Convert each sheet
        yaml_data = {}
        
        for sheet_name in sheets:
            if sheet_name.startswith('_'):
                continue
            
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            
            if df.empty:
                continue
            
            # Convert to dictionary
            if len(df.columns) == 2:
                # Key-value format
                sheet_data = {}
                for _, row in df.iterrows():
                    key = str(row.iloc[0]).strip()
                    value = row.iloc[1]
                    
                    if key and not pd.isna(row.iloc[0]):
                        # Convert value
                        if pd.isna(value):
                            value = None
                        elif isinstance(value, (int, float)):
                            value = float(value) if value != int(value) else int(value)
                        else:
                            value = str(value)
                        
                        sheet_data[key.lower().replace(' ', '_')] = value
                
                yaml_data[sheet_name.lower().replace(' ', '_')] = sheet_data
            else:
                # Table format
                sheet_data = []
                for _, row in df.iterrows():
                    row_data = {}
                    for col in df.columns:
                        value = row[col]
                        if not pd.isna(value):
                            row_data[str(col).lower().replace(' ', '_')] = value
                    if row_data:
                        sheet_data.append(row_data)
                
                yaml_data[sheet_name.lower().replace(' ', '_')] = sheet_data
        
        conversion_time = time.time() - start_time
        
        # Save YAML
        yaml_path = test_file.with_suffix('.yml')
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False)
        
        logger.info(f"  ✅ Conversion completed in {conversion_time:.3f}s")
        logger.info(f"  📝 YAML saved to: {yaml_path}")
        logger.info(f"  📊 Converted {len(yaml_data)} sheets")
        
        # Performance check
        if conversion_time < 0.1:
            logger.info("  ⚡ Performance: EXCELLENT (<100ms)")
        elif conversion_time < 0.2:
            logger.info("  ⚡ Performance: GOOD (<200ms)")
        else:
            logger.info("  ⚠️ Performance: NEEDS IMPROVEMENT (>200ms)")
        
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Conversion failed: {e}")
        return False

def test_pandas_validation():
    """Test pandas validation"""
    logger.info("🔍 Testing pandas validation")
    
    base_path = Path("/srv/samba/shared/bt/backtester_stable/worktrees/ui-refactor/ui-centralized/backtester_v2/configurations")
    
    # Find a test file
    tbs_dir = base_path / "data" / "prod" / "tbs"
    
    if not tbs_dir.exists():
        logger.error(f"TBS directory not found: {tbs_dir}")
        return False
    
    excel_files = list(tbs_dir.glob("*.xlsx"))
    if not excel_files:
        logger.error("No Excel files found in TBS directory")
        return False
    
    test_file = excel_files[0]
    logger.info(f"  📄 Validating: {test_file.name}")
    
    try:
        # Read Excel file
        excel_file = pd.ExcelFile(test_file)
        
        validation_results = {}
        
        for sheet_name in excel_file.sheet_names:
            if sheet_name.startswith('_'):
                continue
            
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            
            # Basic validation
            sheet_validation = {
                'sheet_name': sheet_name,
                'row_count': len(df),
                'column_count': len(df.columns),
                'empty_rows': df.isnull().all(axis=1).sum(),
                'empty_columns': df.isnull().all(axis=0).sum(),
                'has_data': not df.empty,
                'data_types': df.dtypes.to_dict() if not df.empty else {}
            }
            
            validation_results[sheet_name] = sheet_validation
        
        # Print validation results
        for sheet_name, validation in validation_results.items():
            logger.info(f"  📋 {sheet_name}:")
            logger.info(f"    Rows: {validation['row_count']}, Columns: {validation['column_count']}")
            logger.info(f"    Empty rows: {validation['empty_rows']}, Empty columns: {validation['empty_columns']}")
            logger.info(f"    Has data: {'✅' if validation['has_data'] else '❌'}")
        
        logger.info(f"  ✅ Validation completed for {len(validation_results)} sheets")
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Validation failed: {e}")
        return False

def test_performance_benchmark():
    """Test performance across multiple files"""
    logger.info("⚡ Testing performance benchmark")
    
    base_path = Path("/srv/samba/shared/bt/backtester_stable/worktrees/ui-refactor/ui-centralized/backtester_v2/configurations")
    prod_dir = base_path / "data" / "prod"
    
    if not prod_dir.exists():
        logger.error(f"Production directory not found: {prod_dir}")
        return False
    
    # Find test files
    test_files = []
    for strategy_dir in prod_dir.iterdir():
        if strategy_dir.is_dir():
            excel_files = list(strategy_dir.glob("*.xlsx"))
            if excel_files:
                test_files.append(excel_files[0])  # Test first file from each strategy
    
    if not test_files:
        logger.error("No test files found")
        return False
    
    logger.info(f"  📊 Testing {len(test_files)} files")
    
    performance_results = []
    
    for test_file in test_files:
        start_time = time.time()
        
        try:
            # Simple read test
            excel_file = pd.ExcelFile(test_file)
            sheets = excel_file.sheet_names
            
            # Count data points
            total_cells = 0
            for sheet_name in sheets:
                if not sheet_name.startswith('_'):
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                    total_cells += df.size
            
            duration = time.time() - start_time
            
            performance_results.append({
                'file': test_file.name,
                'strategy': test_file.parent.name,
                'duration': duration,
                'sheets': len(sheets),
                'cells': total_cells,
                'meets_target': duration < 0.1
            })
            
            logger.info(f"    📄 {test_file.name}: {duration:.3f}s ({sheets} sheets, {total_cells} cells)")
            
        except Exception as e:
            logger.error(f"    ❌ {test_file.name}: {e}")
            performance_results.append({
                'file': test_file.name,
                'strategy': test_file.parent.name,
                'duration': 0,
                'error': str(e)
            })
    
    # Analyze results
    successful_tests = [r for r in performance_results if 'error' not in r]
    
    if successful_tests:
        avg_duration = sum(r['duration'] for r in successful_tests) / len(successful_tests)
        max_duration = max(r['duration'] for r in successful_tests)
        min_duration = min(r['duration'] for r in successful_tests)
        target_met_count = sum(1 for r in successful_tests if r['meets_target'])
        target_met_rate = target_met_count / len(successful_tests)
        
        logger.info(f"  📈 Performance Summary:")
        logger.info(f"    Average: {avg_duration:.3f}s")
        logger.info(f"    Range: {min_duration:.3f}s - {max_duration:.3f}s")
        logger.info(f"    Target (<100ms): {target_met_count}/{len(successful_tests)} ({target_met_rate:.1%})")
        
        # Performance rating
        if target_met_rate > 0.8:
            logger.info("  ⚡ Performance: EXCELLENT")
        elif target_met_rate > 0.5:
            logger.info("  ⚡ Performance: GOOD")
        else:
            logger.info("  ⚠️ Performance: NEEDS IMPROVEMENT")
        
        return target_met_rate > 0.5
    
    return False

def main():
    """Main test function"""
    logger.info("🚀 Excel Configuration System - Simple Test Suite")
    logger.info("=" * 60)
    
    tests = [
        ("Excel File Discovery", test_excel_file_discovery),
        ("Excel to YAML Conversion", test_excel_to_yaml_conversion),
        ("Pandas Validation", test_pandas_validation),
        ("Performance Benchmark", test_performance_benchmark)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running: {test_name}")
        try:
            start_time = time.time()
            success = test_func()
            duration = time.time() - start_time
            
            results.append({
                'test': test_name,
                'success': success,
                'duration': duration
            })
            
            logger.info(f"{'✅' if success else '❌'} {test_name}: {'PASSED' if success else 'FAILED'} ({duration:.3f}s)")
            
        except Exception as e:
            logger.error(f"❌ {test_name}: FAILED - {e}")
            results.append({
                'test': test_name,
                'success': False,
                'duration': 0,
                'error': str(e)
            })
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📋 TEST SUMMARY")
    logger.info("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r['success'])
    failed_tests = total_tests - passed_tests
    total_duration = sum(r['duration'] for r in results)
    
    logger.info(f"Tests run: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {failed_tests}")
    logger.info(f"Success rate: {passed_tests/total_tests:.1%}")
    logger.info(f"Total duration: {total_duration:.3f}s")
    
    if failed_tests > 0:
        logger.info("\n❌ Failed tests:")
        for result in results:
            if not result['success']:
                error_msg = result.get('error', 'Unknown error')
                logger.info(f"  - {result['test']}: {error_msg}")
    
    logger.info("=" * 60)
    
    return 0 if failed_tests == 0 else 1

if __name__ == "__main__":
    sys.exit(main())