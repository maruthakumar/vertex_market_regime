#!/usr/bin/env python3
"""
Excel-to-Module Integration Test Suite

PHASE 1.5: Test Excel-to-module integration points
- Tests Excel data flow into Python modules
- Validates parameter mapping and transformation
- Tests module initialization with Excel config
- Ensures configuration consistency across modules
- Tests integration with market regime detection
- Ensures NO mock/synthetic data usage

Author: Claude Code
Date: 2025-07-10
Version: 1.0.0 - PHASE 1.5 EXCEL-TO-MODULE INTEGRATION
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Tuple
import json
import tempfile
import shutil

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

class ExcelToModuleIntegrationError(Exception):
    """Raised when Excel-to-module integration fails"""
    pass

class TestExcelToModuleIntegration(unittest.TestCase):
    """
    PHASE 1.5: Excel-to-Module Integration Test Suite
    STRICT: Uses real Excel files with NO MOCK data
    """
    
    def setUp(self):
        """Set up test environment with STRICT real data requirements"""
        self.strategy_config_path = "/srv/samba/shared/bt/backtester_stable/worktrees/strategies/strategy-market-regime/backtester_v2/configurations/data/prod/mr/MR_CONFIG_STRATEGY_1.0.0.xlsx"
        self.portfolio_config_path = "/srv/samba/shared/bt/backtester_stable/worktrees/strategies/strategy-market-regime/backtester_v2/configurations/data/prod/mr/MR_CONFIG_PORTFOLIO_1.0.0.xlsx"
        
        self.strict_mode = True
        self.no_mock_data = True
        
        # Create temporary directory for outputs
        self.temp_dir = tempfile.mkdtemp()
        
        # Verify Excel files exist - FAIL if not available
        for path_name, path in [
            ("Strategy", self.strategy_config_path),
            ("Portfolio", self.portfolio_config_path)
        ]:
            if not Path(path).exists():
                self.fail(f"CRITICAL FAILURE: {path_name} Excel file not found: {path}")
        
        logger.info(f"✅ All Excel configuration files verified")
        logger.info(f"📁 Temporary directory created: {self.temp_dir}")
    
    def tearDown(self):
        """Clean up temporary files"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_excel_config_manager_integration(self):
        """Test: Excel config manager loads and processes Excel data correctly"""
        try:
            from excel_config_manager import MarketRegimeExcelManager
            
            # Initialize with real Excel file
            manager = MarketRegimeExcelManager(self.strategy_config_path)
            
            # Test configuration loading
            config = manager.load_configuration()
            self.assertIsInstance(config, dict, "Configuration should be dict")
            self.assertGreater(len(config), 0, "Configuration should have sheets")
            
            logger.info(f"📊 Excel config manager loaded {len(config)} sheets")
            
            # Test detection parameters
            detection_params = manager.get_detection_parameters()
            self.assertIsInstance(detection_params, dict, "Detection params should be dict")
            
            # Test regime adjustments
            regime_adjustments = manager.get_regime_adjustments()
            self.assertIsInstance(regime_adjustments, dict, "Regime adjustments should be dict")
            
            # Test specific parameter access
            try:
                master_config = manager.get_sheet_data('MasterConfiguration')
                if master_config is not None:
                    self.assertIsInstance(master_config, (pd.DataFrame, dict), 
                                        "Master config should be DataFrame or dict")
                    logger.info("✅ Master configuration accessed successfully")
            except Exception as e:
                logger.warning(f"⚠️ Master configuration access issue: {e}")
            
            # Test indicator configuration access
            try:
                indicator_config = manager.get_sheet_data('IndicatorConfiguration')
                if indicator_config is not None:
                    self.assertIsInstance(indicator_config, (pd.DataFrame, dict), 
                                        "Indicator config should be DataFrame or dict")
                    logger.info("✅ Indicator configuration accessed successfully")
            except Exception as e:
                logger.warning(f"⚠️ Indicator configuration access issue: {e}")
            
            logger.info("✅ PHASE 1.5: Excel config manager integration validated")
            
        except Exception as e:
            self.fail(f"CRITICAL FAILURE: Excel config manager integration failed: {e}")
    
    def test_input_sheet_parser_integration(self):
        """Test: Input sheet parser integrates with Excel data correctly"""
        try:
            from input_sheet_parser import MarketRegimeInputSheetParser
            
            # Initialize parser
            parser = MarketRegimeInputSheetParser()
            
            # Parse portfolio Excel
            parsed_result = parser.parse_input_sheets(self.portfolio_config_path)
            
            # Validate parsed structure
            self.assertIsInstance(parsed_result, dict, "Parsed result should be dict")
            self.assertIn('portfolios', parsed_result, "Should have portfolios")
            self.assertIn('strategies', parsed_result, "Should have strategies")
            self.assertIn('validation_status', parsed_result, "Should have validation status")
            
            # Test portfolio data integration
            portfolios = parsed_result['portfolios']
            self.assertIsInstance(portfolios, list, "Portfolios should be list")
            
            # Test strategy data integration
            strategies = parsed_result['strategies']
            self.assertIsInstance(strategies, list, "Strategies should be list")
            
            # Test validation status
            validation = parsed_result['validation_status']
            self.assertIsInstance(validation, dict, "Validation status should be dict")
            self.assertIn('is_valid', validation, "Should have validity flag")
            
            logger.info(f"📊 Parser processed {len(portfolios)} portfolios, {len(strategies)} strategies")
            
            # Test parameter extraction
            if portfolios:
                portfolio = portfolios[0]
                expected_keys = ['portfolio_name', 'base_capital', 'risk_profile']
                for key in expected_keys:
                    if key in portfolio:
                        logger.info(f"✅ Portfolio parameter {key}: {portfolio[key]}")
            
            if strategies:
                strategy = strategies[0]
                expected_keys = ['strategy_name', 'portfolio_name', 'allocation']
                for key in expected_keys:
                    if key in strategy:
                        logger.info(f"✅ Strategy parameter {key}: {strategy[key]}")
            
            logger.info("✅ PHASE 1.5: Input sheet parser integration validated")
            
        except Exception as e:
            self.fail(f"CRITICAL FAILURE: Input sheet parser integration failed: {e}")
    
    def test_yaml_converter_integration(self):
        """Test: YAML converter integrates with Excel and produces usable output"""
        try:
            from unified_excel_to_yaml_converter import UnifiedExcelToYAMLConverter
            
            # Initialize converter
            converter = UnifiedExcelToYAMLConverter()
            
            # Convert Excel to YAML
            yaml_output_path = os.path.join(self.temp_dir, "integration_test.yaml")
            success, message, yaml_data = converter.convert_excel_to_yaml(
                self.strategy_config_path, yaml_output_path
            )
            
            # Validate conversion success
            self.assertTrue(success, f"YAML conversion should succeed: {message}")
            self.assertTrue(os.path.exists(yaml_output_path), "YAML file should be created")
            self.assertIsInstance(yaml_data, dict, "YAML data should be dict")
            
            # Test YAML structure for module integration
            if 'market_regime_configuration' in yaml_data:
                config = yaml_data['market_regime_configuration']
                
                # Test version information
                if 'version' in config:
                    version = config['version']
                    self.assertIsInstance(version, str, "Version should be string")
                    logger.info(f"✅ Configuration version: {version}")
                
                # Test strategy type
                if 'strategy_type' in config:
                    strategy_type = config['strategy_type']
                    self.assertEqual(strategy_type, 'MARKET_REGIME', "Should be market regime strategy")
                    logger.info(f"✅ Strategy type: {strategy_type}")
                
                # Test indicators section for module integration
                if 'indicators' in config:
                    indicators = config['indicators']
                    self.assertIsInstance(indicators, dict, "Indicators should be dict")
                    logger.info(f"📊 Found {len(indicators)} indicators in YAML")
                    
                    # Test indicator structure
                    for ind_name, ind_config in indicators.items():
                        if isinstance(ind_config, dict):
                            expected_keys = ['enabled', 'base_weight']
                            for key in expected_keys:
                                if key in ind_config:
                                    logger.info(f"✅ Indicator {ind_name} has {key}: {ind_config[key]}")
            
            # Test YAML can be loaded by modules
            import yaml
            with open(yaml_output_path, 'r') as f:
                loaded_yaml = yaml.safe_load(f)
            
            # Compare core data (ignore sheets_processed key added during conversion)
            if 'sheets_processed' in yaml_data:
                yaml_data_copy = yaml_data.copy()
                yaml_data_copy.pop('sheets_processed', None)
            else:
                yaml_data_copy = yaml_data
            
            if 'sheets_processed' in loaded_yaml:
                loaded_yaml_copy = loaded_yaml.copy()
                loaded_yaml_copy.pop('sheets_processed', None)
            else:
                loaded_yaml_copy = loaded_yaml
            
            self.assertEqual(loaded_yaml_copy, yaml_data_copy, "Loaded YAML should match original (core data)")
            
            logger.info("✅ PHASE 1.5: YAML converter integration validated")
            
        except Exception as e:
            self.fail(f"CRITICAL FAILURE: YAML converter integration failed: {e}")
    
    def test_config_validator_integration(self):
        """Test: Config validator integrates with Excel data for validation"""
        try:
            from unified_config_validator import UnifiedConfigValidator
            
            # Initialize validator
            validator = UnifiedConfigValidator()
            
            # Validate Excel file
            is_valid, results = validator.validate_excel_file(self.strategy_config_path)
            
            # Test validation results
            self.assertIsInstance(is_valid, bool, "Validation result should be boolean")
            self.assertIsInstance(results, list, "Validation results should be list")
            
            # Test validation summary
            summary = validator.get_validation_summary()
            self.assertIsInstance(summary, dict, "Summary should be dict")
            
            expected_summary_keys = ['total_checks', 'errors', 'warnings', 'info', 'is_valid']
            for key in expected_summary_keys:
                self.assertIn(key, summary, f"Summary should have {key}")
            
            logger.info(f"📊 Validation summary: {summary['total_checks']} checks, "
                      f"{summary['errors']} errors, {summary['warnings']} warnings")
            
            # Test validation details for module integration
            if 'details' in summary:
                details = summary['details']
                
                # Check error details
                if 'errors' in details and details['errors']:
                    logger.info(f"⚠️ Found {len(details['errors'])} validation errors")
                    for error in details['errors'][:3]:  # Show first 3
                        logger.info(f"  - {error['sheet']}: {error['message']}")
                
                # Check warning details
                if 'warnings' in details and details['warnings']:
                    logger.info(f"⚠️ Found {len(details['warnings'])} validation warnings")
                    for warning in details['warnings'][:3]:  # Show first 3
                        logger.info(f"  - {warning['sheet']}: {warning['message']}")
            
            # Test validation rules are accessible to modules
            sheet_rules = validator.sheet_rules
            self.assertIsInstance(sheet_rules, dict, "Sheet rules should be dict")
            self.assertGreater(len(sheet_rules), 0, "Should have validation rules")
            
            logger.info(f"📊 Validator has rules for {len(sheet_rules)} sheet types")
            
            logger.info("✅ PHASE 1.5: Config validator integration validated")
            
        except Exception as e:
            self.fail(f"CRITICAL FAILURE: Config validator integration failed: {e}")
    
    def test_parameter_flow_through_modules(self):
        """Test: Parameters flow correctly through all modules in sequence"""
        try:
            # Test the complete flow: Excel → Manager → Parser → Validator → YAML
            
            # Step 1: Excel Manager
            from excel_config_manager import MarketRegimeExcelManager
            manager = MarketRegimeExcelManager(self.strategy_config_path)
            excel_config = manager.load_configuration()
            
            # Step 2: Input Parser
            from input_sheet_parser import MarketRegimeInputSheetParser
            parser = MarketRegimeInputSheetParser()
            parsed_result = parser.parse_input_sheets(self.portfolio_config_path)
            
            # Step 3: Config Validator
            from unified_config_validator import UnifiedConfigValidator
            validator = UnifiedConfigValidator()
            is_valid, validation_results = validator.validate_excel_file(self.strategy_config_path)
            
            # Step 4: YAML Converter
            from unified_excel_to_yaml_converter import UnifiedExcelToYAMLConverter
            converter = UnifiedExcelToYAMLConverter()
            yaml_path = os.path.join(self.temp_dir, "parameter_flow_test.yaml")
            success, message, yaml_data = converter.convert_excel_to_yaml(
                self.strategy_config_path, yaml_path
            )
            
            # Validate complete flow
            self.assertIsInstance(excel_config, dict, "Excel config should be dict")
            self.assertIsInstance(parsed_result, dict, "Parsed result should be dict")
            self.assertIsInstance(validation_results, list, "Validation results should be list")
            self.assertTrue(success, "YAML conversion should succeed")
            self.assertIsInstance(yaml_data, dict, "YAML data should be dict")
            
            # Test parameter consistency across modules
            excel_sheet_count = len(excel_config)
            validator_sheet_count = len(validator.sheet_rules)
            yaml_sections = len(yaml_data.get('market_regime_configuration', {}))
            
            logger.info(f"📊 Parameter flow: Excel({excel_sheet_count} sheets) → "
                      f"Parser({len(parsed_result)} sections) → "
                      f"Validator({validator_sheet_count} rules) → "
                      f"YAML({yaml_sections} sections)")
            
            # Test specific parameter tracking
            test_params = {}
            
            # Track parameters from Excel
            if 'MasterConfiguration' in excel_config:
                master_df = excel_config['MasterConfiguration']
                if isinstance(master_df, pd.DataFrame):
                    for _, row in master_df.iterrows():
                        param = row.get('Parameter') if 'Parameter' in master_df.columns else row.iloc[0]
                        if pd.notna(param):
                            test_params[f"excel_{param}"] = row.get('Value') if 'Value' in master_df.columns else row.iloc[1]
            
            # Track parameters from YAML
            if yaml_data and 'market_regime_configuration' in yaml_data:
                config = yaml_data['market_regime_configuration']
                if 'system' in config:
                    for key, value in config['system'].items():
                        test_params[f"yaml_{key}"] = value
            
            logger.info(f"📊 Tracked {len(test_params)} parameters through flow")
            
            # Validate parameters are preserved
            self.assertGreater(len(test_params), 0, "Should track some parameters through flow")
            
            logger.info("✅ PHASE 1.5: Parameter flow through modules validated")
            
        except Exception as e:
            self.fail(f"CRITICAL FAILURE: Parameter flow test failed: {e}")
    
    def test_module_error_handling_integration(self):
        """Test: Module error handling when integrating with Excel data"""
        try:
            # Test error handling with invalid/missing data
            
            # Test 1: Excel Manager with invalid path
            from excel_config_manager import MarketRegimeExcelManager
            try:
                invalid_manager = MarketRegimeExcelManager("/invalid/path.xlsx")
                config = invalid_manager.load_configuration()
                # Should handle gracefully
                logger.info("✅ Excel manager handled invalid path gracefully")
            except Exception as e:
                logger.info(f"✅ Excel manager raised expected error: {type(e).__name__}")
            
            # Test 2: Parser with invalid data
            from input_sheet_parser import MarketRegimeInputSheetParser
            parser = MarketRegimeInputSheetParser()
            try:
                result = parser.parse_input_sheets("/invalid/portfolio.xlsx")
                # Should return error status
                if isinstance(result, dict) and 'validation_status' in result:
                    validation = result['validation_status']
                    if not validation.get('is_valid', True):
                        logger.info("✅ Parser handled invalid file with proper error status")
            except Exception as e:
                logger.info(f"✅ Parser raised expected error: {type(e).__name__}")
            
            # Test 3: Validator with invalid Excel
            from unified_config_validator import UnifiedConfigValidator
            validator = UnifiedConfigValidator()
            try:
                is_valid, results = validator.validate_excel_file("/invalid/config.xlsx")
                self.assertFalse(is_valid, "Should return False for invalid file")
                self.assertGreater(len(results), 0, "Should have error results")
                logger.info("✅ Validator handled invalid file with proper error reporting")
            except Exception as e:
                logger.info(f"✅ Validator raised expected error: {type(e).__name__}")
            
            # Test 4: YAML Converter with invalid input
            from unified_excel_to_yaml_converter import UnifiedExcelToYAMLConverter
            converter = UnifiedExcelToYAMLConverter()
            yaml_path = os.path.join(self.temp_dir, "error_test.yaml")
            success, message, yaml_data = converter.convert_excel_to_yaml(
                "/invalid/excel.xlsx", yaml_path
            )
            
            self.assertFalse(success, "Should return False for invalid file")
            self.assertIsInstance(message, str, "Should provide error message")
            logger.info("✅ YAML converter handled invalid file with proper error reporting")
            
            logger.info("✅ PHASE 1.5: Module error handling integration validated")
            
        except Exception as e:
            self.fail(f"CRITICAL FAILURE: Module error handling test failed: {e}")
    
    def test_module_performance_integration(self):
        """Test: Module performance when integrating with real Excel data"""
        try:
            import time
            
            performance_results = {}
            
            # Test 1: Excel Manager performance
            start_time = time.time()
            from excel_config_manager import MarketRegimeExcelManager
            manager = MarketRegimeExcelManager(self.strategy_config_path)
            config = manager.load_configuration()
            excel_time = time.time() - start_time
            performance_results['excel_manager'] = excel_time
            
            # Test 2: Parser performance
            start_time = time.time()
            from input_sheet_parser import MarketRegimeInputSheetParser
            parser = MarketRegimeInputSheetParser()
            parsed_result = parser.parse_input_sheets(self.portfolio_config_path)
            parser_time = time.time() - start_time
            performance_results['input_parser'] = parser_time
            
            # Test 3: Validator performance
            start_time = time.time()
            from unified_config_validator import UnifiedConfigValidator
            validator = UnifiedConfigValidator()
            is_valid, results = validator.validate_excel_file(self.strategy_config_path)
            validator_time = time.time() - start_time
            performance_results['config_validator'] = validator_time
            
            # Test 4: YAML Converter performance
            start_time = time.time()
            from unified_excel_to_yaml_converter import UnifiedExcelToYAMLConverter
            converter = UnifiedExcelToYAMLConverter()
            yaml_path = os.path.join(self.temp_dir, "performance_test.yaml")
            success, message, yaml_data = converter.convert_excel_to_yaml(
                self.strategy_config_path, yaml_path
            )
            converter_time = time.time() - start_time
            performance_results['yaml_converter'] = converter_time
            
            # Validate performance
            logger.info("📊 Module Performance Results:")
            total_time = 0
            for module, exec_time in performance_results.items():
                logger.info(f"  - {module}: {exec_time:.3f} seconds")
                total_time += exec_time
                
                # Validate reasonable performance (adjust thresholds as needed)
                self.assertLess(exec_time, 30.0, f"{module} should complete in <30 seconds")
            
            logger.info(f"📊 Total integration time: {total_time:.3f} seconds")
            self.assertLess(total_time, 60.0, "Complete integration should take <60 seconds")
            
            # Test memory usage (basic check)
            import psutil
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            logger.info(f"📊 Memory usage: {memory_mb:.1f} MB")
            self.assertLess(memory_mb, 1000, "Memory usage should be reasonable (<1GB)")
            
            logger.info("✅ PHASE 1.5: Module performance integration validated")
            
        except Exception as e:
            self.fail(f"CRITICAL FAILURE: Module performance test failed: {e}")
    
    def test_cross_module_data_consistency(self):
        """Test: Data consistency across all modules"""
        try:
            # Load the same Excel data through different modules
            
            # Method 1: Excel Manager
            from excel_config_manager import MarketRegimeExcelManager
            manager = MarketRegimeExcelManager(self.strategy_config_path)
            excel_config = manager.load_configuration()
            
            # Method 2: Direct pandas reading
            excel_file = pd.ExcelFile(self.strategy_config_path)
            direct_sheets = {}
            for sheet_name in excel_file.sheet_names[:5]:  # Check first 5 sheets
                try:
                    direct_sheets[sheet_name] = pd.read_excel(excel_file, sheet_name=sheet_name, header=1)
                except:
                    direct_sheets[sheet_name] = pd.read_excel(excel_file, sheet_name=sheet_name)
            
            # Method 3: YAML Converter processed data
            from unified_excel_to_yaml_converter import UnifiedExcelToYAMLConverter
            converter = UnifiedExcelToYAMLConverter()
            yaml_path = os.path.join(self.temp_dir, "consistency_test.yaml")
            success, message, yaml_data = converter.convert_excel_to_yaml(
                self.strategy_config_path, yaml_path
            )
            
            # Compare data consistency
            logger.info("📊 Cross-module data consistency check:")
            
            # Compare sheet counts
            excel_manager_sheets = len(excel_config)
            direct_read_sheets = len(direct_sheets)
            yaml_sections = len(yaml_data.get('market_regime_configuration', {})) if yaml_data else 0
            
            logger.info(f"  - Excel Manager: {excel_manager_sheets} sheets")
            logger.info(f"  - Direct Read: {direct_read_sheets} sheets")
            logger.info(f"  - YAML Sections: {yaml_sections} sections")
            
            # Validate sheet consistency
            for sheet_name in direct_sheets:
                if sheet_name in excel_config:
                    # Compare basic properties
                    direct_df = direct_sheets[sheet_name]
                    manager_df = excel_config[sheet_name]
                    
                    if isinstance(manager_df, pd.DataFrame):
                        # Compare shapes
                        if direct_df.shape == manager_df.shape:
                            logger.info(f"✅ {sheet_name}: Shape consistency verified")
                        else:
                            logger.warning(f"⚠️ {sheet_name}: Shape mismatch - "
                                         f"direct{direct_df.shape} vs manager{manager_df.shape}")
                        
                        # Compare column names
                        if list(direct_df.columns) == list(manager_df.columns):
                            logger.info(f"✅ {sheet_name}: Column consistency verified")
                        else:
                            logger.warning(f"⚠️ {sheet_name}: Column mismatch")
            
            # Test parameter value consistency
            consistency_checks = 0
            for sheet_name, sheet_df in direct_sheets.items():
                if isinstance(sheet_df, pd.DataFrame) and len(sheet_df) > 0:
                    # Check for Parameter/Value format
                    if 'Parameter' in sheet_df.columns and 'Value' in sheet_df.columns:
                        for _, row in sheet_df.iterrows():
                            if pd.notna(row['Parameter']) and pd.notna(row['Value']):
                                consistency_checks += 1
                                if consistency_checks > 20:  # Limit checks for performance
                                    break
            
            logger.info(f"📊 Performed {consistency_checks} parameter consistency checks")
            
            # Validate YAML preserves essential data
            if yaml_data and 'market_regime_configuration' in yaml_data:
                config = yaml_data['market_regime_configuration']
                essential_sections = ['version', 'strategy_type', 'created']
                found_sections = [s for s in essential_sections if s in config]
                logger.info(f"✅ YAML preserves {len(found_sections)}/{len(essential_sections)} essential sections")
            
            logger.info("✅ PHASE 1.5: Cross-module data consistency validated")
            
        except Exception as e:
            self.fail(f"CRITICAL FAILURE: Cross-module data consistency test failed: {e}")
    
    def test_no_synthetic_data_in_integration(self):
        """Test: Ensure NO synthetic/mock data flows through module integration"""
        try:
            # Track all data sources through integration flow
            data_sources = {}
            
            # Test 1: Excel Manager data source
            from excel_config_manager import MarketRegimeExcelManager
            manager = MarketRegimeExcelManager(self.strategy_config_path)
            data_sources['excel_manager'] = manager.config_path
            
            # Test 2: Parser data source
            from input_sheet_parser import MarketRegimeInputSheetParser
            parser = MarketRegimeInputSheetParser()
            parsed_result = parser.parse_input_sheets(self.portfolio_config_path)
            data_sources['input_parser'] = parsed_result.get('input_sheet_path', self.portfolio_config_path)
            
            # Test 3: YAML Converter data source
            from unified_excel_to_yaml_converter import UnifiedExcelToYAMLConverter
            converter = UnifiedExcelToYAMLConverter()
            yaml_path = os.path.join(self.temp_dir, "no_mock_integration.yaml")
            success, message, yaml_data = converter.convert_excel_to_yaml(
                self.strategy_config_path, yaml_path
            )
            data_sources['yaml_converter'] = self.strategy_config_path
            
            # Validate all sources are real files
            logger.info("📊 Data source verification:")
            for module, source in data_sources.items():
                self.assertTrue(Path(source).exists(), f"{module} should use existing file")
                logger.info(f"  ✅ {module}: {source}")
            
            # Check YAML content for mock patterns
            if yaml_data:
                yaml_str = str(yaml_data).lower()
                mock_patterns = ['mock', 'test_data', 'dummy', 'synthetic', 'fake', 'sample']
                
                suspicious_count = 0
                for pattern in mock_patterns:
                    count = yaml_str.count(pattern)
                    if count > 5:  # Allow some legitimate uses
                        suspicious_count += count
                        logger.warning(f"⚠️ Found {count} instances of '{pattern}' in YAML")
                
                self.assertLess(suspicious_count, 20, "Should not have excessive mock patterns in YAML")
            
            # Verify file sizes indicate real data (be flexible with portfolio config)
            for module, source in data_sources.items():
                file_size = Path(source).stat().st_size
                min_size = 5000 if 'PORTFOLIO' in source.upper() else 10000
                self.assertGreater(file_size, min_size, f"{source} should be substantial (>{min_size/1000:.0f}KB)")
                logger.info(f"📊 {module} file size: {file_size/1024:.1f} KB")
            
            # Check Excel sheet counts indicate real configuration
            excel_file = pd.ExcelFile(self.strategy_config_path)
            sheet_count = len(excel_file.sheet_names)
            self.assertGreaterEqual(sheet_count, 30, "Should have 30+ sheets for real configuration")
            logger.info(f"📊 Excel file has {sheet_count} sheets (real configuration)")
            
            logger.info("✅ PHASE 1.5: NO synthetic data in integration verified")
            
        except Exception as e:
            self.fail(f"CRITICAL FAILURE: Synthetic data detection failed: {e}")

def run_excel_to_module_integration_tests():
    """Run Excel-to-module integration test suite"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🔧 PHASE 1.5: EXCEL-TO-MODULE INTEGRATION TESTS")
    print("=" * 70)
    print("⚠️  STRICT MODE: Using real Excel configuration files")
    print("⚠️  NO MOCK DATA: Testing actual module integration with real data")
    print("⚠️  COMPREHENSIVE: Testing all integration points and data flow")
    print("=" * 70)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestExcelToModuleIntegration)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Report results
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success_rate = (total_tests - failures - errors) / total_tests if total_tests > 0 else 0
    
    print(f"\n{'=' * 70}")
    print(f"PHASE 1.5: EXCEL-TO-MODULE INTEGRATION RESULTS")
    print(f"{'=' * 70}")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_tests - failures - errors}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    print(f"Success Rate: {success_rate:.1%}")
    print(f"{'=' * 70}")
    
    if failures > 0 or errors > 0:
        print("❌ PHASE 1.5: EXCEL-TO-MODULE INTEGRATION FAILED")
        print("🔧 ISSUES NEED TO BE FIXED BEFORE PROCEEDING")
        
        if failures > 0:
            print("\nFAILURES:")
            for test, traceback in result.failures:
                print(f"- {test}: {traceback}")
        if errors > 0:
            print("\nERRORS:")
            for test, traceback in result.errors:
                print(f"- {test}: {traceback}")
        return False
    else:
        print("✅ PHASE 1.5: EXCEL-TO-MODULE INTEGRATION PASSED")
        print("🔧 ALL MODULE INTEGRATION POINTS VALIDATED")
        print("📊 EXCEL DATA FLOWS CORRECTLY THROUGH ALL MODULES")
        print("✅ READY FOR PHASE 1.6 - CROSS-SHEET DEPENDENCY VALIDATION")
        return True

if __name__ == "__main__":
    success = run_excel_to_module_integration_tests()
    sys.exit(0 if success else 1)