#!/usr/bin/env python3
"""
Phase 2 Day 5: Excel Configuration Integration Test
Complete testing and validation of Excel configuration system

Author: The Augster
Date: 2025-06-20
Version: 5.0.0 (Phase 2 Day 5 Complete Test)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Phase2Day5ExcelIntegrationTest:
    """
    Complete test suite for Phase 2 Day 5 Excel Configuration Integration
    
    Tests all components:
    1. Excel Configuration Templates
    2. Hot-reloading Configuration System
    3. Progressive Disclosure UI Integration
    4. Production Integration
    """
    
    def __init__(self):
        """Initialize test suite"""
        
        self.test_results = {}
        self.config_file = Path("excel_config_templates/DTE_ENHANCED_CONFIGURATION_TEMPLATE.xlsx")
        self.json_file = Path("excel_config_templates/DTE_ENHANCED_CONFIGURATION_TEMPLATE.json")
        
        logger.info("🧪 Phase 2 Day 5 Excel Integration Test Suite initialized")
    
    def run_complete_test_suite(self) -> Dict[str, Any]:
        """Run complete test suite for Phase 2 Day 5"""
        
        logger.info("\n" + "="*80)
        logger.info("PHASE 2 DAY 5: EXCEL CONFIGURATION INTEGRATION TEST SUITE")
        logger.info("="*80)
        
        start_time = time.time()
        
        # Test 1: Excel Configuration Templates
        self.test_results['excel_templates'] = self._test_excel_configuration_templates()
        
        # Test 2: Configuration Structure Validation
        self.test_results['structure_validation'] = self._test_configuration_structure()
        
        # Test 3: Progressive Disclosure Logic
        self.test_results['progressive_disclosure'] = self._test_progressive_disclosure_logic()
        
        # Test 4: Hot-reload Simulation
        self.test_results['hot_reload'] = self._test_hot_reload_simulation()
        
        # Test 5: Strategy Type Configuration
        self.test_results['strategy_config'] = self._test_strategy_type_configuration()
        
        # Test 6: Performance Configuration
        self.test_results['performance_config'] = self._test_performance_configuration()
        
        # Test 7: Production Integration Compatibility
        self.test_results['production_integration'] = self._test_production_integration_compatibility()
        
        # Test 8: Excel Parser Compatibility
        self.test_results['excel_parser_compatibility'] = self._test_excel_parser_compatibility()
        
        total_time = time.time() - start_time
        
        # Generate comprehensive test report
        self._generate_test_report(total_time)
        
        return self.test_results
    
    def _test_excel_configuration_templates(self) -> Dict[str, Any]:
        """Test Excel configuration templates creation and structure"""
        
        logger.info("\n📊 Testing Excel Configuration Templates...")
        
        test_result = {
            'test_name': 'Excel Configuration Templates',
            'status': 'PASS',
            'details': {},
            'issues': []
        }
        
        try:
            # Check if Excel file exists
            if not self.config_file.exists():
                test_result['status'] = 'FAIL'
                test_result['issues'].append('Excel configuration file not found')
                return test_result
            
            # Check if JSON file exists
            if not self.json_file.exists():
                test_result['status'] = 'FAIL'
                test_result['issues'].append('JSON configuration file not found')
                return test_result
            
            # Read Excel file and validate sheets
            excel_file = pd.ExcelFile(self.config_file)
            expected_sheets = [
                'DTE_Learning_Config',
                'ML_Model_Config',
                'Strategy_Config',
                'Performance_Config',
                'UI_Config',
                'Validation_Config',
                'Rolling_Config',
                'Regime_Config'
            ]
            
            missing_sheets = [sheet for sheet in expected_sheets if sheet not in excel_file.sheet_names]
            if missing_sheets:
                test_result['status'] = 'FAIL'
                test_result['issues'].append(f'Missing sheets: {missing_sheets}')
            
            test_result['details']['total_sheets'] = len(excel_file.sheet_names)
            test_result['details']['expected_sheets'] = len(expected_sheets)
            test_result['details']['sheets_present'] = excel_file.sheet_names
            
            # Validate JSON structure
            with open(self.json_file, 'r') as f:
                json_config = json.load(f)
            
            test_result['details']['json_sheets'] = len(json_config)
            test_result['details']['json_structure_valid'] = isinstance(json_config, dict)
            
            logger.info(f"   ✅ Excel file: {len(excel_file.sheet_names)} sheets")
            logger.info(f"   ✅ JSON file: {len(json_config)} sheet configurations")
            
        except Exception as e:
            test_result['status'] = 'FAIL'
            test_result['issues'].append(f'Exception: {str(e)}')
            logger.error(f"   ❌ Error testing Excel templates: {e}")
        
        return test_result
    
    def _test_configuration_structure(self) -> Dict[str, Any]:
        """Test configuration structure and parameter validation"""
        
        logger.info("\n⚙️ Testing Configuration Structure...")
        
        test_result = {
            'test_name': 'Configuration Structure',
            'status': 'PASS',
            'details': {},
            'issues': []
        }
        
        try:
            # Test DTE Learning Config structure
            dte_df = pd.read_excel(self.config_file, sheet_name='DTE_Learning_Config')
            
            required_dte_params = [
                'DTE_LEARNING_ENABLED',
                'DTE_RANGE_MIN',
                'DTE_RANGE_MAX',
                'DTE_FOCUS_RANGE_MIN',
                'DTE_FOCUS_RANGE_MAX',
                'ATM_BASE_WEIGHT',
                'ITM1_BASE_WEIGHT',
                'OTM1_BASE_WEIGHT'
            ]
            
            missing_dte_params = [p for p in required_dte_params if p not in dte_df['Parameter'].values]
            if missing_dte_params:
                test_result['issues'].append(f'Missing DTE parameters: {missing_dte_params}')
            
            test_result['details']['dte_parameters'] = len(dte_df)
            test_result['details']['required_dte_parameters'] = len(required_dte_params)
            
            # Test Strategy Config structure
            strategy_df = pd.read_excel(self.config_file, sheet_name='Strategy_Config')
            strategy_types = strategy_df['Strategy_Type'].unique()
            expected_strategies = ['TBS', 'TV', 'ORB', 'OI', 'Indicator', 'POS']
            
            missing_strategies = [s for s in expected_strategies if s not in strategy_types]
            if missing_strategies:
                test_result['issues'].append(f'Missing strategy types: {missing_strategies}')
            
            test_result['details']['strategy_types'] = len(strategy_types)
            test_result['details']['expected_strategies'] = len(expected_strategies)
            test_result['details']['strategy_configurations'] = len(strategy_df)
            
            # Test Progressive Disclosure structure
            ui_df = pd.read_excel(self.config_file, sheet_name='UI_Config')
            skill_levels = ui_df['Skill_Level'].unique()
            expected_skill_levels = ['Novice', 'Intermediate', 'Expert']
            
            missing_skill_levels = [s for s in expected_skill_levels if s not in skill_levels]
            if missing_skill_levels:
                test_result['issues'].append(f'Missing skill levels: {missing_skill_levels}')
            
            test_result['details']['skill_levels'] = len(skill_levels)
            test_result['details']['ui_configurations'] = len(ui_df)
            
            if test_result['issues']:
                test_result['status'] = 'FAIL'
            
            logger.info(f"   ✅ DTE parameters: {len(dte_df)}")
            logger.info(f"   ✅ Strategy types: {len(strategy_types)}")
            logger.info(f"   ✅ Skill levels: {len(skill_levels)}")
            
        except Exception as e:
            test_result['status'] = 'FAIL'
            test_result['issues'].append(f'Exception: {str(e)}')
            logger.error(f"   ❌ Error testing configuration structure: {e}")
        
        return test_result
    
    def _test_progressive_disclosure_logic(self) -> Dict[str, Any]:
        """Test progressive disclosure logic"""
        
        logger.info("\n🎯 Testing Progressive Disclosure Logic...")
        
        test_result = {
            'test_name': 'Progressive Disclosure Logic',
            'status': 'PASS',
            'details': {},
            'issues': []
        }
        
        try:
            # Simulate progressive disclosure for different skill levels
            skill_levels = ['Novice', 'Intermediate', 'Expert']
            
            for skill_level in skill_levels:
                # Read DTE Learning Config
                dte_df = pd.read_excel(self.config_file, sheet_name='DTE_Learning_Config')
                
                # Filter parameters by skill level
                skill_params = dte_df[dte_df['Skill_Level'] == skill_level]
                
                test_result['details'][f'{skill_level.lower()}_parameters'] = len(skill_params)
                
                # Validate skill level hierarchy
                if skill_level == 'Novice':
                    expected_min_params = 5  # Basic parameters
                elif skill_level == 'Intermediate':
                    expected_min_params = 10  # More parameters
                else:  # Expert
                    expected_min_params = 15  # Most parameters
                
                if len(skill_params) < expected_min_params:
                    test_result['issues'].append(f'{skill_level} has too few parameters: {len(skill_params)}')
            
            # Test parameter visibility hierarchy
            novice_count = test_result['details']['novice_parameters']
            intermediate_count = test_result['details']['intermediate_parameters']
            expert_count = test_result['details']['expert_parameters']
            
            # Validate hierarchy: Novice <= Intermediate <= Expert (in terms of complexity)
            if not (novice_count <= intermediate_count <= expert_count):
                test_result['issues'].append('Progressive disclosure hierarchy not maintained')
            
            test_result['details']['hierarchy_valid'] = novice_count <= intermediate_count <= expert_count
            
            if test_result['issues']:
                test_result['status'] = 'FAIL'
            
            logger.info(f"   ✅ Novice parameters: {novice_count}")
            logger.info(f"   ✅ Intermediate parameters: {intermediate_count}")
            logger.info(f"   ✅ Expert parameters: {expert_count}")
            
        except Exception as e:
            test_result['status'] = 'FAIL'
            test_result['issues'].append(f'Exception: {str(e)}')
            logger.error(f"   ❌ Error testing progressive disclosure: {e}")
        
        return test_result
    
    def _test_hot_reload_simulation(self) -> Dict[str, Any]:
        """Test hot-reload simulation"""
        
        logger.info("\n🔄 Testing Hot-reload Simulation...")
        
        test_result = {
            'test_name': 'Hot-reload Simulation',
            'status': 'PASS',
            'details': {},
            'issues': []
        }
        
        try:
            # Read current configuration
            dte_df = pd.read_excel(self.config_file, sheet_name='DTE_Learning_Config')
            
            # Count hot-reloadable parameters
            hot_reload_params = dte_df[dte_df.get('Hot_Reload', True) == True]
            non_hot_reload_params = dte_df[dte_df.get('Hot_Reload', True) == False]
            
            test_result['details']['hot_reloadable_parameters'] = len(hot_reload_params)
            test_result['details']['non_hot_reloadable_parameters'] = len(non_hot_reload_params)
            test_result['details']['total_parameters'] = len(dte_df)
            
            # Validate that critical parameters are hot-reloadable
            critical_hot_reload_params = [
                'DTE_LEARNING_ENABLED',
                'ATM_BASE_WEIGHT',
                'ITM1_BASE_WEIGHT',
                'OTM1_BASE_WEIGHT',
                'CONFIDENCE_THRESHOLD'
            ]
            
            for param in critical_hot_reload_params:
                param_row = dte_df[dte_df['Parameter'] == param]
                if not param_row.empty:
                    is_hot_reloadable = param_row.iloc[0].get('Hot_Reload', True)
                    if not is_hot_reloadable:
                        test_result['issues'].append(f'Critical parameter {param} not hot-reloadable')
            
            # Simulate configuration change validation
            test_changes = [
                {'parameter': 'ATM_BASE_WEIGHT', 'old_value': 0.5, 'new_value': 0.6},
                {'parameter': 'DTE_LEARNING_ENABLED', 'old_value': True, 'new_value': False},
                {'parameter': 'TARGET_PROCESSING_TIME', 'old_value': 3.0, 'new_value': 2.5}
            ]
            
            valid_changes = 0
            for change in test_changes:
                # Simple validation simulation
                if self._validate_parameter_change(change):
                    valid_changes += 1
            
            test_result['details']['test_changes'] = len(test_changes)
            test_result['details']['valid_changes'] = valid_changes
            
            if test_result['issues']:
                test_result['status'] = 'FAIL'
            
            logger.info(f"   ✅ Hot-reloadable parameters: {len(hot_reload_params)}")
            logger.info(f"   ✅ Valid test changes: {valid_changes}/{len(test_changes)}")
            
        except Exception as e:
            test_result['status'] = 'FAIL'
            test_result['issues'].append(f'Exception: {str(e)}')
            logger.error(f"   ❌ Error testing hot-reload: {e}")
        
        return test_result
    
    def _validate_parameter_change(self, change: Dict[str, Any]) -> bool:
        """Simulate parameter change validation"""
        
        try:
            parameter = change['parameter']
            new_value = change['new_value']
            
            # Simple validation rules
            if 'WEIGHT' in parameter:
                return 0.05 <= new_value <= 0.80
            elif parameter == 'DTE_LEARNING_ENABLED':
                return isinstance(new_value, bool)
            elif parameter == 'TARGET_PROCESSING_TIME':
                return 0.1 <= new_value <= 60.0
            
            return True
            
        except Exception:
            return False
    
    def _test_strategy_type_configuration(self) -> Dict[str, Any]:
        """Test strategy type configuration"""
        
        logger.info("\n⚙️ Testing Strategy Type Configuration...")
        
        test_result = {
            'test_name': 'Strategy Type Configuration',
            'status': 'PASS',
            'details': {},
            'issues': []
        }
        
        try:
            strategy_df = pd.read_excel(self.config_file, sheet_name='Strategy_Config')
            
            # Test all 6 strategy types
            expected_strategies = ['TBS', 'TV', 'ORB', 'OI', 'Indicator', 'POS']
            
            for strategy in expected_strategies:
                strategy_rows = strategy_df[strategy_df['Strategy_Type'] == strategy]
                
                if strategy_rows.empty:
                    test_result['issues'].append(f'No configuration for strategy {strategy}')
                    continue
                
                # Check required parameters for each strategy
                required_params = ['dte_learning_enabled', 'default_dte_focus', 'weight_optimization']
                strategy_params = strategy_rows['Parameter'].values
                
                missing_params = [p for p in required_params if p not in strategy_params]
                if missing_params:
                    test_result['issues'].append(f'Strategy {strategy} missing parameters: {missing_params}')
                
                test_result['details'][f'{strategy}_parameters'] = len(strategy_rows)
            
            test_result['details']['total_strategies'] = len(expected_strategies)
            test_result['details']['configured_strategies'] = len(strategy_df['Strategy_Type'].unique())
            
            if test_result['issues']:
                test_result['status'] = 'FAIL'
            
            logger.info(f"   ✅ Configured strategies: {len(strategy_df['Strategy_Type'].unique())}")
            logger.info(f"   ✅ Total strategy configurations: {len(strategy_df)}")
            
        except Exception as e:
            test_result['status'] = 'FAIL'
            test_result['issues'].append(f'Exception: {str(e)}')
            logger.error(f"   ❌ Error testing strategy configuration: {e}")
        
        return test_result
    
    def _test_performance_configuration(self) -> Dict[str, Any]:
        """Test performance configuration"""
        
        logger.info("\n⚡ Testing Performance Configuration...")
        
        test_result = {
            'test_name': 'Performance Configuration',
            'status': 'PASS',
            'details': {},
            'issues': []
        }
        
        try:
            perf_df = pd.read_excel(self.config_file, sheet_name='Performance_Config')
            
            # Check critical performance parameters
            critical_perf_params = [
                'TARGET_PROCESSING_TIME',
                'PARALLEL_PROCESSING_ENABLED',
                'MAX_WORKERS',
                'ENABLE_CACHING',
                'ENABLE_VECTORIZATION'
            ]
            
            missing_perf_params = [p for p in critical_perf_params if p not in perf_df['Parameter'].values]
            if missing_perf_params:
                test_result['issues'].append(f'Missing performance parameters: {missing_perf_params}')
            
            # Validate performance target
            target_time_row = perf_df[perf_df['Parameter'] == 'TARGET_PROCESSING_TIME']
            if not target_time_row.empty:
                target_time = target_time_row.iloc[0]['Value']
                if target_time > 5.0:  # Should be <= 5 seconds for good performance
                    test_result['issues'].append(f'Performance target too high: {target_time}s')
                test_result['details']['target_processing_time'] = target_time
            
            test_result['details']['performance_parameters'] = len(perf_df)
            test_result['details']['critical_parameters_present'] = len(critical_perf_params) - len(missing_perf_params)
            
            if test_result['issues']:
                test_result['status'] = 'FAIL'
            
            logger.info(f"   ✅ Performance parameters: {len(perf_df)}")
            logger.info(f"   ✅ Target processing time: {test_result['details'].get('target_processing_time', 'N/A')}s")
            
        except Exception as e:
            test_result['status'] = 'FAIL'
            test_result['issues'].append(f'Exception: {str(e)}')
            logger.error(f"   ❌ Error testing performance configuration: {e}")
        
        return test_result
    
    def _test_production_integration_compatibility(self) -> Dict[str, Any]:
        """Test production integration compatibility"""
        
        logger.info("\n🏭 Testing Production Integration Compatibility...")
        
        test_result = {
            'test_name': 'Production Integration Compatibility',
            'status': 'PASS',
            'details': {},
            'issues': []
        }
        
        try:
            # Test HeavyDB integration parameters
            validation_df = pd.read_excel(self.config_file, sheet_name='Validation_Config')
            
            heavydb_params = [
                'REAL_DATA_ENFORCEMENT',
                'SYNTHETIC_DATA_ALLOWED',
                'DATA_QUALITY_CHECKS'
            ]
            
            missing_heavydb_params = [p for p in heavydb_params if p not in validation_df['Parameter'].values]
            if missing_heavydb_params:
                test_result['issues'].append(f'Missing HeavyDB parameters: {missing_heavydb_params}')
            
            # Check real data enforcement
            real_data_row = validation_df[validation_df['Parameter'] == 'REAL_DATA_ENFORCEMENT']
            if not real_data_row.empty:
                real_data_enforced = real_data_row.iloc[0]['Value']
                test_result['details']['real_data_enforcement'] = real_data_enforced
                if not real_data_enforced:
                    test_result['issues'].append('Real data enforcement not enabled')
            
            # Check synthetic data policy
            synthetic_data_row = validation_df[validation_df['Parameter'] == 'SYNTHETIC_DATA_ALLOWED']
            if not synthetic_data_row.empty:
                synthetic_allowed = synthetic_data_row.iloc[0]['Value']
                test_result['details']['synthetic_data_allowed'] = synthetic_allowed
                if synthetic_allowed:
                    test_result['issues'].append('Synthetic data fallbacks allowed (should be disabled for production)')
            
            test_result['details']['validation_parameters'] = len(validation_df)
            test_result['details']['heavydb_compatibility'] = len(heavydb_params) - len(missing_heavydb_params)
            
            if test_result['issues']:
                test_result['status'] = 'FAIL'
            
            logger.info(f"   ✅ Validation parameters: {len(validation_df)}")
            logger.info(f"   ✅ Real data enforcement: {test_result['details'].get('real_data_enforcement', 'N/A')}")
            
        except Exception as e:
            test_result['status'] = 'FAIL'
            test_result['issues'].append(f'Exception: {str(e)}')
            logger.error(f"   ❌ Error testing production integration: {e}")
        
        return test_result
    
    def _test_excel_parser_compatibility(self) -> Dict[str, Any]:
        """Test Excel parser compatibility"""
        
        logger.info("\n📊 Testing Excel Parser Compatibility...")
        
        test_result = {
            'test_name': 'Excel Parser Compatibility',
            'status': 'PASS',
            'details': {},
            'issues': []
        }
        
        try:
            # Test that all sheets can be read successfully
            excel_file = pd.ExcelFile(self.config_file)
            readable_sheets = 0
            
            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(self.config_file, sheet_name=sheet_name)
                    
                    # Check required columns
                    if 'Parameter' in df.columns and 'Value' in df.columns:
                        readable_sheets += 1
                    else:
                        test_result['issues'].append(f'Sheet {sheet_name} missing required columns')
                        
                except Exception as e:
                    test_result['issues'].append(f'Cannot read sheet {sheet_name}: {str(e)}')
            
            test_result['details']['total_sheets'] = len(excel_file.sheet_names)
            test_result['details']['readable_sheets'] = readable_sheets
            test_result['details']['parser_compatibility'] = readable_sheets / len(excel_file.sheet_names)
            
            # Test JSON compatibility
            try:
                with open(self.json_file, 'r') as f:
                    json_config = json.load(f)
                test_result['details']['json_parseable'] = True
            except Exception as e:
                test_result['issues'].append(f'JSON parsing error: {str(e)}')
                test_result['details']['json_parseable'] = False
            
            if test_result['issues']:
                test_result['status'] = 'FAIL'
            
            logger.info(f"   ✅ Readable sheets: {readable_sheets}/{len(excel_file.sheet_names)}")
            logger.info(f"   ✅ JSON parseable: {test_result['details']['json_parseable']}")
            
        except Exception as e:
            test_result['status'] = 'FAIL'
            test_result['issues'].append(f'Exception: {str(e)}')
            logger.error(f"   ❌ Error testing Excel parser compatibility: {e}")
        
        return test_result
    
    def _generate_test_report(self, total_time: float):
        """Generate comprehensive test report"""
        
        logger.info("\n" + "="*80)
        logger.info("PHASE 2 DAY 5 EXCEL INTEGRATION TEST RESULTS")
        logger.info("="*80)
        
        # Count results
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'PASS')
        failed_tests = total_tests - passed_tests
        
        logger.info(f"⏱️ Total test time: {total_time:.3f}s")
        logger.info(f"📊 Total tests: {total_tests}")
        logger.info(f"✅ Passed: {passed_tests}")
        logger.info(f"❌ Failed: {failed_tests}")
        logger.info(f"📈 Success rate: {(passed_tests/total_tests)*100:.1f}%")
        
        # Detailed results
        logger.info(f"\n📋 Detailed Test Results:")
        for test_name, result in self.test_results.items():
            status_icon = "✅" if result['status'] == 'PASS' else "❌"
            logger.info(f"   {status_icon} {result['test_name']}: {result['status']}")
            
            if result['issues']:
                for issue in result['issues']:
                    logger.info(f"      ⚠️ {issue}")
        
        # Overall assessment
        logger.info(f"\n🎯 Overall Assessment:")
        if failed_tests == 0:
            logger.info("🎉 ALL TESTS PASSED - Excel Configuration Integration SUCCESSFUL!")
            logger.info("✅ Ready for production deployment")
        elif failed_tests <= 2:
            logger.info("⚠️ MOSTLY SUCCESSFUL - Minor issues detected")
            logger.info("🔧 Address issues before production deployment")
        else:
            logger.info("❌ SIGNIFICANT ISSUES - Major problems detected")
            logger.info("🛠️ Requires fixes before proceeding")
        
        # Save test report
        self._save_test_report(total_time)
    
    def _save_test_report(self, total_time: float):
        """Save test report to file"""
        
        try:
            report_data = {
                'test_execution': {
                    'timestamp': datetime.now().isoformat(),
                    'total_time': total_time,
                    'phase': 'Phase_2_Day_5_Excel_Integration'
                },
                'test_summary': {
                    'total_tests': len(self.test_results),
                    'passed_tests': sum(1 for r in self.test_results.values() if r['status'] == 'PASS'),
                    'failed_tests': sum(1 for r in self.test_results.values() if r['status'] == 'FAIL'),
                    'success_rate': (sum(1 for r in self.test_results.values() if r['status'] == 'PASS') / len(self.test_results)) * 100
                },
                'test_results': self.test_results
            }
            
            report_file = Path(f"phase2_day5_excel_integration_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            logger.info(f"📄 Test report saved: {report_file}")
            
        except Exception as e:
            logger.error(f"❌ Error saving test report: {e}")

def main():
    """Main execution function"""
    
    # Initialize and run test suite
    test_suite = Phase2Day5ExcelIntegrationTest()
    results = test_suite.run_complete_test_suite()
    
    return results

if __name__ == "__main__":
    main()
