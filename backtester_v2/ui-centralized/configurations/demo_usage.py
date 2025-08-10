"""
Demo Usage of Excel Configuration System
Shows practical usage patterns for the unified system
"""

import sys
import os
import time
import logging
from pathlib import Path

# Add the configurations directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demo_basic_conversion():
    """Demo: Basic Excel to YAML conversion"""
    logger.info("📄 Demo: Basic Excel to YAML conversion")
    
    from converters.excel_to_yaml import convert_excel_to_yaml
    
    # Find a test file
    base_path = Path("/srv/samba/shared/bt/backtester_stable/worktrees/ui-refactor/ui-centralized/backtester_v2/configurations")
    test_file = base_path / "data" / "prod" / "tbs" / "TBS_CONFIG_STRATEGY_1.0.0.xlsx"
    
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return
    
    try:
        # Convert Excel to YAML
        yaml_path = convert_excel_to_yaml(str(test_file))
        
        logger.info(f"✅ Converted {test_file.name} to {yaml_path}")
        
        # Show YAML content snippet
        with open(yaml_path, 'r') as f:
            yaml_content = f.read()
            logger.info(f"📝 YAML preview (first 500 chars):\n{yaml_content[:500]}...")
    
    except Exception as e:
        logger.error(f"❌ Conversion failed: {e}")

def demo_batch_conversion():
    """Demo: Batch conversion of multiple files"""
    logger.info("📁 Demo: Batch conversion of multiple files")
    
    from converters.excel_to_yaml import batch_convert_excel_to_yaml
    
    # Find test files
    base_path = Path("/srv/samba/shared/bt/backtester_stable/worktrees/ui-refactor/ui-centralized/backtester_v2/configurations")
    tbs_dir = base_path / "data" / "prod" / "tbs"
    
    if not tbs_dir.exists():
        logger.error(f"TBS directory not found: {tbs_dir}")
        return
    
    excel_files = list(tbs_dir.glob("*.xlsx"))
    if not excel_files:
        logger.error("No Excel files found")
        return
    
    try:
        # Batch convert
        output_dir = tbs_dir / "yaml_output"
        output_dir.mkdir(exist_ok=True)
        
        file_paths = [str(f) for f in excel_files]
        output_mapping = batch_convert_excel_to_yaml(file_paths, str(output_dir))
        
        logger.info(f"✅ Batch converted {len(output_mapping)} files")
        for input_path, output_path in output_mapping.items():
            logger.info(f"  📄 {Path(input_path).name} → {Path(output_path).name}")
    
    except Exception as e:
        logger.error(f"❌ Batch conversion failed: {e}")

def demo_performance_monitoring():
    """Demo: Performance monitoring and metrics"""
    logger.info("⚡ Demo: Performance monitoring")
    
    from converters.excel_to_yaml import ExcelToYAMLConverter
    
    # Create converter
    converter = ExcelToYAMLConverter()
    
    # Find test files
    base_path = Path("/srv/samba/shared/bt/backtester_stable/worktrees/ui-refactor/ui-centralized/backtester_v2/configurations")
    test_files = []
    
    for strategy_dir in (base_path / "data" / "prod").iterdir():
        if strategy_dir.is_dir():
            excel_files = list(strategy_dir.glob("*.xlsx"))
            if excel_files:
                test_files.append(excel_files[0])  # Take first file from each strategy
    
    if not test_files:
        logger.error("No test files found")
        return
    
    # Convert files and collect metrics
    logger.info(f"📊 Testing {len(test_files)} files...")
    
    for test_file in test_files:
        try:
            yaml_data, metrics = converter.convert_single_file(str(test_file))
            
            logger.info(f"  📄 {test_file.name}:")
            logger.info(f"    Success: {'✅' if metrics.success else '❌'}")
            logger.info(f"    Duration: {metrics.total_time:.3f}s")
            logger.info(f"    File size: {metrics.file_size:,} bytes")
            logger.info(f"    Sheets: {metrics.sheet_count}")
            logger.info(f"    Target met: {'✅' if metrics.total_time < 0.1 else '❌'}")
            
        except Exception as e:
            logger.error(f"    ❌ {test_file.name}: {e}")
    
    # Get overall performance stats
    stats = converter.get_performance_stats()
    logger.info(f"\n📈 Overall Performance:")
    logger.info(f"  Total conversions: {stats.get('total_conversions', 0)}")
    logger.info(f"  Success rate: {stats.get('success_rate', 0):.1%}")
    logger.info(f"  Average time: {stats.get('avg_total_time', 0):.3f}s")
    logger.info(f"  Target met rate: {stats.get('target_met_rate', 0):.1%}")

def demo_validation_system():
    """Demo: Validation system with pandas"""
    logger.info("🔍 Demo: Validation system")
    
    from converters.excel_to_yaml import ExcelToYAMLConverter
    
    # Create converter
    converter = ExcelToYAMLConverter()
    
    # Find test file
    base_path = Path("/srv/samba/shared/bt/backtester_stable/worktrees/ui-refactor/ui-centralized/backtester_v2/configurations")
    test_file = base_path / "data" / "prod" / "tbs" / "TBS_CONFIG_STRATEGY_1.0.0.xlsx"
    
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return
    
    try:
        # Convert with validation
        yaml_data, metrics = converter.convert_single_file(str(test_file), strategy_type='tbs')
        
        if metrics.success:
            logger.info(f"✅ Conversion successful: {test_file.name}")
            logger.info(f"  Processing time: {metrics.processing_time:.3f}s")
            logger.info(f"  Validation time: {metrics.validation_time:.3f}s")
            logger.info(f"  Total time: {metrics.total_time:.3f}s")
            
            # Show validation details
            validation_result = converter._validate_with_pandas(yaml_data, 'tbs', str(test_file))
            logger.info(f"  Validation result: {'✅ Valid' if validation_result['valid'] else '❌ Invalid'}")
            
            if validation_result['errors']:
                logger.info(f"  Errors: {len(validation_result['errors'])}")
                for error in validation_result['errors'][:3]:  # Show first 3 errors
                    logger.info(f"    - {error}")
            
            if validation_result['warnings']:
                logger.info(f"  Warnings: {len(validation_result['warnings'])}")
                for warning in validation_result['warnings'][:3]:  # Show first 3 warnings
                    logger.info(f"    - {warning}")
            
        else:
            logger.error(f"❌ Conversion failed: {metrics.error_message}")
    
    except Exception as e:
        logger.error(f"❌ Validation demo failed: {e}")

def demo_yaml_structure():
    """Demo: Show YAML structure for different strategy types"""
    logger.info("📋 Demo: YAML structure analysis")
    
    from converters.excel_to_yaml import ExcelToYAMLConverter
    
    # Create converter
    converter = ExcelToYAMLConverter()
    
    # Test different strategies
    base_path = Path("/srv/samba/shared/bt/backtester_stable/worktrees/ui-refactor/ui-centralized/backtester_v2/configurations")
    
    strategies_to_test = ['tbs', 'orb', 'oi', 'ml', 'mr']
    
    for strategy in strategies_to_test:
        strategy_dir = base_path / "data" / "prod" / strategy
        
        if not strategy_dir.exists():
            logger.warning(f"Strategy directory not found: {strategy}")
            continue
        
        excel_files = list(strategy_dir.glob("*.xlsx"))
        if not excel_files:
            logger.warning(f"No Excel files found for strategy: {strategy}")
            continue
        
        test_file = excel_files[0]
        
        try:
            yaml_data, metrics = converter.convert_single_file(str(test_file), strategy_type=strategy)
            
            if metrics.success:
                logger.info(f"📊 {strategy.upper()} Strategy ({test_file.name}):")
                logger.info(f"  Sheets: {metrics.sheet_count}")
                logger.info(f"  YAML structure:")
                
                for sheet_name, sheet_data in yaml_data.items():
                    if sheet_name.startswith('_'):
                        continue
                    
                    if isinstance(sheet_data, dict):
                        logger.info(f"    {sheet_name}: {len(sheet_data)} parameters")
                        # Show sample parameters
                        sample_params = list(sheet_data.keys())[:3]
                        logger.info(f"      Sample: {sample_params}")
                    elif isinstance(sheet_data, list):
                        logger.info(f"    {sheet_name}: {len(sheet_data)} rows")
                        if sheet_data:
                            sample_cols = list(sheet_data[0].keys())[:3] if isinstance(sheet_data[0], dict) else []
                            logger.info(f"      Sample columns: {sample_cols}")
                    else:
                        logger.info(f"    {sheet_name}: {type(sheet_data).__name__}")
                
                logger.info(f"  Performance: {metrics.total_time:.3f}s")
                logger.info("")
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze {strategy}: {e}")

def demo_caching_performance():
    """Demo: Caching performance improvement"""
    logger.info("🚀 Demo: Caching performance")
    
    from converters.excel_to_yaml import ExcelToYAMLConverter
    
    # Create converter
    converter = ExcelToYAMLConverter()
    
    # Find test file
    base_path = Path("/srv/samba/shared/bt/backtester_stable/worktrees/ui-refactor/ui-centralized/backtester_v2/configurations")
    test_file = base_path / "data" / "prod" / "tbs" / "TBS_CONFIG_STRATEGY_1.0.0.xlsx"
    
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return
    
    try:
        # First conversion (no cache)
        start_time = time.time()
        yaml_data1, metrics1 = converter.convert_single_file(str(test_file))
        first_time = time.time() - start_time
        
        # Second conversion (with cache)
        start_time = time.time()
        yaml_data2, metrics2 = converter.convert_single_file(str(test_file))
        second_time = time.time() - start_time
        
        logger.info(f"📊 Caching Performance Test:")
        logger.info(f"  First conversion: {first_time:.3f}s")
        logger.info(f"  Second conversion: {second_time:.3f}s")
        logger.info(f"  Speed improvement: {first_time/second_time:.1f}x faster")
        logger.info(f"  Cache hit: {'✅' if second_time < first_time/2 else '❌'}")
        
    except Exception as e:
        logger.error(f"❌ Caching demo failed: {e}")

def main():
    """Main demo function"""
    logger.info("🚀 Excel Configuration System - Demo Usage")
    logger.info("=" * 60)
    
    demos = [
        ("Basic Conversion", demo_basic_conversion),
        ("Batch Conversion", demo_batch_conversion),
        ("Performance Monitoring", demo_performance_monitoring),
        ("Validation System", demo_validation_system),
        ("YAML Structure Analysis", demo_yaml_structure),
        ("Caching Performance", demo_caching_performance)
    ]
    
    for demo_name, demo_func in demos:
        logger.info(f"\n🎯 {demo_name}")
        logger.info("-" * 40)
        
        try:
            demo_func()
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Demo completed successfully!")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()