#!/usr/bin/env python3
"""
Validation script for Vertex Market Regime modular structure

Verifies that all necessary files and components are in place
and validates the configuration migration.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

def check_file_exists(filepath: Path) -> bool:
    """Check if file exists and is not empty"""
    return filepath.exists() and filepath.stat().st_size > 0

def validate_directory_structure() -> Dict[str, bool]:
    """Validate the complete directory structure"""
    
    required_structure = [
        # Root files
        "README.md",
        "requirements.txt", 
        "setup.py",
        
        # Configuration directories
        "configs/excel/MR_CONFIG_REGIME_1.0.0.xlsx",
        "configs/excel/MR_CONFIG_STRATEGY_1.0.0.xlsx",
        "configs/excel/MR_CONFIG_OPTIMIZATION_1.0.0.xlsx",
        "configs/excel/MR_CONFIG_PORTFOLIO_1.0.0.xlsx",
        "configs/excel/excel_parser.py",
        
        # Source code structure
        "src/components/base_component.py",
        "src/components/component_02_greeks_sentiment/greeks_analyzer.py",
        
        # Scripts
        "scripts/setup_environment.sh",
    ]
    
    results = {}
    for path_str in required_structure:
        filepath = PROJECT_ROOT / path_str
        results[path_str] = check_file_exists(filepath)
    
    return results

def validate_component_structure() -> Dict[str, bool]:
    """Validate component directory structure"""
    
    component_dirs = [
        "component_01_triple_straddle",
        "component_02_greeks_sentiment", 
        "component_03_oi_pa_trending",
        "component_04_iv_skew",
        "component_05_atr_ema_cpr",
        "component_06_correlation",
        "component_07_support_resistance",
        "component_08_master_integration"
    ]
    
    results = {}
    components_base = PROJECT_ROOT / "src" / "components"
    
    for comp_dir in component_dirs:
        comp_path = components_base / comp_dir
        results[f"component_dir_{comp_dir}"] = comp_path.exists()
        
        # Check for __init__.py
        init_file = comp_path / "__init__.py"
        results[f"component_init_{comp_dir}"] = init_file.exists()
    
    return results

def validate_configuration_files() -> Dict[str, bool]:
    """Validate Excel configuration files"""
    
    excel_dir = PROJECT_ROOT / "configs" / "excel"
    excel_files = [
        "MR_CONFIG_REGIME_1.0.0.xlsx",
        "MR_CONFIG_STRATEGY_1.0.0.xlsx",
        "MR_CONFIG_OPTIMIZATION_1.0.0.xlsx", 
        "MR_CONFIG_PORTFOLIO_1.0.0.xlsx"
    ]
    
    results = {}
    for filename in excel_files:
        filepath = excel_dir / filename
        results[f"excel_{filename}"] = check_file_exists(filepath)
    
    return results

def test_configuration_migration() -> Dict[str, bool]:
    """Test configuration migration functionality"""
    
    results = {}
    
    try:
        # Import the configuration bridge
        from configs.excel.excel_parser import ExcelConfigurationBridge
        results["import_excel_parser"] = True
        
        # Initialize parser
        excel_dir = str(PROJECT_ROOT / "configs" / "excel")
        parser = ExcelConfigurationBridge(excel_dir)
        results["initialize_parser"] = True
        
        # Test configuration migration
        master_config = parser.migrate_all_configurations()
        results["migrate_configurations"] = True
        
        # Validate migrated config
        validation = parser.validate_configuration(master_config)
        results["validate_config"] = validation.get('overall', False)
        
        # Check specific fixes
        results["gamma_weight_fixed"] = validation.get('gamma_weight_fixed', False)
        results["component_count_correct"] = validation.get('component_count', False)
        results["feature_count_correct"] = validation.get('feature_count', False)
        
        # Check component details
        comp_2 = next((c for c in master_config.components if c.component_id == 2), None)
        if comp_2:
            results["component_2_exists"] = True
            results["component_2_gamma_corrected"] = comp_2.parameters.get('gamma_weight_corrected') == 1.5
        else:
            results["component_2_exists"] = False
            results["component_2_gamma_corrected"] = False
        
    except Exception as e:
        print(f"❌ Configuration migration test failed: {e}")
        results["migration_error"] = str(e)
        return {"test_failed": False, "error": str(e)}
    
    return results

def test_component_loading() -> Dict[str, bool]:
    """Test component loading"""
    
    results = {}
    
    try:
        # Test base component import
        from src.components.base_component import BaseMarketRegimeComponent, ComponentFactory
        results["import_base_component"] = True
        
        # Test Greeks component import
        from src.components.component_02_greeks_sentiment.greeks_analyzer import GreeksAnalyzer
        results["import_greeks_analyzer"] = True
        
        # Test component initialization
        config = {
            'component_id': 2,
            'feature_count': 98,
            'gamma_weight_corrected': True
        }
        
        greeks_analyzer = GreeksAnalyzer(config)
        results["initialize_greeks_analyzer"] = True
        
        # Verify gamma weight fix
        gamma_weight = greeks_analyzer.greek_weights.get('gamma', 0.0)
        results["gamma_weight_is_1_5"] = gamma_weight == 1.5
        
        # Test feature count
        results["correct_feature_count"] = greeks_analyzer.feature_count == 98
        
    except Exception as e:
        print(f"❌ Component loading test failed: {e}")
        results["component_loading_error"] = str(e)
        return {"test_failed": False, "error": str(e)}
    
    return results

def generate_validation_report(results: Dict[str, Dict[str, bool]]) -> str:
    """Generate formatted validation report"""
    
    report = f"""
# Vertex Market Regime Structure Validation Report

**Generated**: {Path(__file__).name}
**Project Root**: {PROJECT_ROOT}

## Directory Structure Validation

"""
    
    for category, tests in results.items():
        if isinstance(tests, dict):
            passed = sum(1 for result in tests.values() if result is True)
            total = len([r for r in tests.values() if isinstance(r, bool)])
            
            status = "✅ PASSED" if passed == total else f"⚠️  {passed}/{total} PASSED"
            report += f"### {category.replace('_', ' ').title()}: {status}\n\n"
            
            for test_name, result in tests.items():
                if isinstance(result, bool):
                    status_icon = "✅" if result else "❌"
                    report += f"- **{test_name}**: {status_icon}\n"
                elif isinstance(result, str):
                    report += f"- **{test_name}**: {result}\n"
            
            report += "\n"
    
    # Summary
    all_tests = []
    for tests in results.values():
        if isinstance(tests, dict):
            all_tests.extend([r for r in tests.values() if isinstance(r, bool)])
    
    total_passed = sum(1 for result in all_tests if result)
    total_tests = len(all_tests)
    overall_status = "✅ PASSED" if total_passed == total_tests else "⚠️  NEEDS ATTENTION"
    
    report += f"""
## Summary

- **Overall Status**: {overall_status}
- **Tests Passed**: {total_passed}/{total_tests}
- **Success Rate**: {(total_passed/total_tests)*100:.1f}%

## Critical Fixes Verified

"""
    
    # Check critical fixes
    config_results = results.get('configuration_migration', {})
    if config_results.get('gamma_weight_fixed'):
        report += "- ✅ **Gamma Weight Fix**: Component 2 gamma weight corrected from 0.0 to 1.5\n"
    else:
        report += "- ❌ **Gamma Weight Fix**: Not verified\n"
    
    if config_results.get('component_count_correct'):
        report += "- ✅ **Component Count**: All 8 components present\n"
    else:
        report += "- ❌ **Component Count**: Incorrect component count\n"
    
    if config_results.get('feature_count_correct'):
        report += "- ✅ **Feature Count**: Total 774 features configured\n"
    else:
        report += "- ❌ **Feature Count**: Incorrect total feature count\n"
    
    report += "\n## Next Steps\n\n"
    
    if total_passed == total_tests:
        report += "✅ **All validations passed!** Ready to proceed with component development.\n\n"
        report += "**Recommended next steps:**\n"
        report += "1. Run setup script: `./scripts/setup_environment.sh`\n"
        report += "2. Test configuration migration: `python configs/excel/excel_parser.py`\n"
        report += "3. Begin component implementation starting with Component 1\n"
    else:
        report += "⚠️  **Some validations failed.** Please address the issues above before proceeding.\n"
    
    return report

def main():
    """Main validation function"""
    
    print("🔍 Validating Vertex Market Regime structure...")
    print(f"📁 Project root: {PROJECT_ROOT}")
    
    # Run all validations
    validation_results = {
        'directory_structure': validate_directory_structure(),
        'component_structure': validate_component_structure(),
        'configuration_files': validate_configuration_files(),
        'configuration_migration': test_configuration_migration(),
        'component_loading': test_component_loading()
    }
    
    # Generate report
    report = generate_validation_report(validation_results)
    
    # Save report
    report_path = PROJECT_ROOT / "validation_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Print summary
    print("\n" + "="*60)
    print(report)
    print("="*60)
    print(f"📄 Full report saved to: {report_path}")
    
    # Calculate overall success
    all_tests = []
    for tests in validation_results.values():
        if isinstance(tests, dict):
            all_tests.extend([r for r in tests.values() if isinstance(r, bool)])
    
    success_rate = sum(1 for result in all_tests if result) / len(all_tests)
    
    if success_rate == 1.0:
        print("🎉 ALL VALIDATIONS PASSED!")
        return 0
    elif success_rate >= 0.8:
        print(f"⚠️  MOSTLY SUCCESSFUL ({success_rate*100:.1f}%)")
        return 1
    else:
        print(f"❌ VALIDATION FAILED ({success_rate*100:.1f}%)")
        return 2

if __name__ == "__main__":
    sys.exit(main())