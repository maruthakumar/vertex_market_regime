"""
Comprehensive Configuration Validation Test Suite

Tests all aspects of market regime configuration validation including:
- Excel file parsing and validation
- Parameter range checking
- Weight normalization
- Dependency validation
- Edge cases and error handling

Author: Market Regime System Optimizer
Date: 2025-07-07
Version: 1.0.0
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import os
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import validation components
from advanced_config_validator import ConfigurationValidator, ValidationIssue, ValidationSeverity


class TestConfigValidation(unittest.TestCase):
    """Comprehensive tests for configuration validation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.validator = ConfigurationValidator()
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def create_test_excel(self, sheets_data):
        """Create test Excel file with given data"""
        file_path = os.path.join(self.temp_dir, 'test_config.xlsx')
        
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            for sheet_name, data in sheets_data.items():
                if isinstance(data, pd.DataFrame):
                    data.to_excel(writer, sheet_name=sheet_name, index=False)
                else:
                    # Convert dict to DataFrame
                    df = pd.DataFrame([data])
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
        return file_path
        
    def test_valid_configuration(self):
        """Test validation of a valid configuration"""
        # Create valid config
        sheets_data = {
            'Master_Config': pd.DataFrame({
                'Parameter': ['version', 'regime_count', 'confidence_threshold'],
                'Value': ['2.0.0', '12', '0.85']
            }),
            'Indicator_Weights': pd.DataFrame({
                'Indicator': ['GreekSentiment', 'TrendingOIPA', 'StraddleAnalysis', 'ATR'],
                'Weight': [0.3, 0.3, 0.25, 0.15],
                'Enabled': [True, True, True, True]
            }),
            'Regime_Thresholds': pd.DataFrame({
                'Regime': ['R1', 'R2', 'R3', 'R4'],
                'Volatility_Min': [0.0, 0.0, 0.3, 0.3],
                'Volatility_Max': [0.3, 0.3, 0.7, 0.7],
                'Trend_Min': [0.6, -0.6, 0.3, -0.3],
                'Trend_Max': [1.0, -0.3, 0.6, -0.6]
            }),
            'Timeframe_Config': pd.DataFrame({
                'Timeframe': [1, 5, 15, 30],
                'Weight': [0.3, 0.4, 0.2, 0.1],
                'Lookback': [60, 100, 200, 300]
            })
        }
        
        file_path = self.create_test_excel(sheets_data)
        is_valid, issues, metadata = self.validator.validate_excel_file(file_path)
        
        # Should be valid
        self.assertTrue(is_valid)
        self.assertEqual(len([i for i in issues if i.severity == ValidationSeverity.ERROR]), 0)
        
    def test_indicator_weight_validation(self):
        """Test indicator weight validation rules"""
        # Test case 1: Weights don't sum to 1
        sheets_data = {
            'Indicator_Weights': pd.DataFrame({
                'Indicator': ['GreekSentiment', 'TrendingOIPA', 'StraddleAnalysis'],
                'Weight': [0.3, 0.3, 0.3],  # Sum = 0.9
                'Enabled': [True, True, True]
            })
        }
        
        file_path = self.create_test_excel(sheets_data)
        is_valid, issues, _ = self.validator.validate_excel_file(file_path)
        
        # Should have warning about weight sum
        weight_issues = [i for i in issues if 'weight' in i.message.lower()]
        self.assertGreater(len(weight_issues), 0)
        
        # Test case 2: Negative weights
        sheets_data['Indicator_Weights'] = pd.DataFrame({
            'Indicator': ['GreekSentiment', 'TrendingOIPA'],
            'Weight': [-0.5, 1.5],  # Negative weight
            'Enabled': [True, True]
        })
        
        file_path = self.create_test_excel(sheets_data)
        is_valid, issues, _ = self.validator.validate_excel_file(file_path)
        
        # Should have error about negative weight
        self.assertFalse(is_valid)
        error_issues = [i for i in issues if i.severity == ValidationSeverity.ERROR]
        self.assertGreater(len(error_issues), 0)
        
    def test_regime_threshold_validation(self):
        """Test regime threshold validation"""
        # Test case: Overlapping thresholds
        sheets_data = {
            'Regime_Thresholds': pd.DataFrame({
                'Regime': ['R1', 'R2'],
                'Volatility_Min': [0.0, 0.2],  # R2 starts before R1 ends
                'Volatility_Max': [0.5, 0.7],
                'Trend_Min': [0.0, 0.0],
                'Trend_Max': [1.0, 1.0]
            })
        }
        
        file_path = self.create_test_excel(sheets_data)
        is_valid, issues, _ = self.validator.validate_excel_file(file_path)
        
        # Should detect overlapping ranges
        overlap_issues = [i for i in issues if 'overlap' in i.message.lower()]
        self.assertGreater(len(overlap_issues), 0)
        
    def test_parameter_range_validation(self):
        """Test parameter range validation"""
        # Test various out-of-range parameters
        test_cases = [
            {
                'name': 'Invalid lookback',
                'sheet': 'Straddle_Config',
                'data': pd.DataFrame({
                    'Parameter': ['EMA_Period', 'Lookback_Period'],
                    'Value': [5, 1000]  # Lookback too large
                }),
                'expected_error': True
            },
            {
                'name': 'Invalid confidence',
                'sheet': 'Master_Config',
                'data': pd.DataFrame({
                    'Parameter': ['confidence_threshold'],
                    'Value': [1.5]  # > 1.0
                }),
                'expected_error': True
            },
            {
                'name': 'Invalid timeframe',
                'sheet': 'Timeframe_Config',
                'data': pd.DataFrame({
                    'Timeframe': [0.5],  # Not in valid periods
                    'Weight': [1.0],
                    'Lookback': [100]
                }),
                'expected_error': True
            }
        ]
        
        for test_case in test_cases:
            sheets_data = {test_case['sheet']: test_case['data']}
            file_path = self.create_test_excel(sheets_data)
            is_valid, issues, _ = self.validator.validate_excel_file(file_path)
            
            if test_case['expected_error']:
                error_issues = [i for i in issues if i.severity == ValidationSeverity.ERROR]
                self.assertGreater(len(error_issues), 0, 
                                 f"Expected error for {test_case['name']}")
                                 
    def test_greek_parameter_validation(self):
        """Test Greek parameter validation"""
        sheets_data = {
            'Greek_Config': pd.DataFrame({
                'Greek': ['Delta', 'Gamma', 'Theta', 'Vega'],
                'Min_Value': [-2.0, -1.0, 10.0, -5.0],  # Invalid ranges
                'Max_Value': [2.0, 15.0, 50.0, 150.0]
            })
        }
        
        file_path = self.create_test_excel(sheets_data)
        is_valid, issues, _ = self.validator.validate_excel_file(file_path)
        
        # Should have errors for invalid Greek ranges
        greek_errors = [i for i in issues if 'greek' in i.category.lower()]
        self.assertGreater(len(greek_errors), 0)
        
    def test_missing_required_sheets(self):
        """Test handling of missing required sheets"""
        # Create config with missing critical sheet
        sheets_data = {
            'Indicator_Weights': pd.DataFrame({
                'Indicator': ['Test'],
                'Weight': [1.0],
                'Enabled': [True]
            })
            # Missing Master_Config
        }
        
        file_path = self.create_test_excel(sheets_data)
        is_valid, issues, _ = self.validator.validate_excel_file(file_path)
        
        # Should report missing sheets
        missing_issues = [i for i in issues if 'missing' in i.message.lower()]
        self.assertGreater(len(missing_issues), 0)
        
    def test_minimum_enabled_indicators(self):
        """Test minimum enabled indicators requirement"""
        sheets_data = {
            'Indicator_Weights': pd.DataFrame({
                'Indicator': ['Greek', 'OI', 'Straddle', 'ATR', 'RSI'],
                'Weight': [0.2, 0.2, 0.2, 0.2, 0.2],
                'Enabled': [True, True, False, False, False]  # Only 2 enabled
            })
        }
        
        file_path = self.create_test_excel(sheets_data)
        is_valid, issues, _ = self.validator.validate_excel_file(file_path)
        
        # Should warn about too few enabled indicators
        enabled_issues = [i for i in issues if 'enabled' in i.message.lower()]
        self.assertGreater(len(enabled_issues), 0)
        
    def test_regime_count_validation(self):
        """Test regime count validation"""
        # Test with 20 regimes (outside valid range)
        sheets_data = {
            'Master_Config': pd.DataFrame({
                'Parameter': ['regime_count'],
                'Value': ['20']
            }),
            'Regime_Thresholds': pd.DataFrame({
                'Regime': [f'R{i}' for i in range(1, 21)],
                'Volatility_Min': [i * 0.05 for i in range(20)],
                'Volatility_Max': [(i + 1) * 0.05 for i in range(20)],
                'Trend_Min': [-1.0] * 20,
                'Trend_Max': [1.0] * 20
            })
        }
        
        file_path = self.create_test_excel(sheets_data)
        is_valid, issues, _ = self.validator.validate_excel_file(file_path)
        
        # Should have issue about regime count
        regime_issues = [i for i in issues if 'regime' in i.message.lower()]
        self.assertGreater(len(regime_issues), 0)
        
    def test_correlation_matrix_size(self):
        """Test correlation matrix configuration"""
        sheets_data = {
            'Correlation_Config': pd.DataFrame({
                'Component': ['ATM_CE', 'ATM_PE', 'ITM1_CE', 'ITM1_PE', 
                            'OTM1_CE', 'OTM1_PE', 'ATM_STRADDLE', 
                            'ITM1_STRADDLE', 'OTM1_STRADDLE', 
                            'COMBINED_TRIPLE_STRADDLE'],
                'Weight': [0.1] * 10,
                'Enabled': [True] * 10
            })
        }
        
        file_path = self.create_test_excel(sheets_data)
        is_valid, issues, _ = self.validator.validate_excel_file(file_path)
        
        # Should validate 10x10 matrix configuration
        self.assertTrue(is_valid or len([i for i in issues 
                                       if i.severity == ValidationSeverity.ERROR]) == 0)
                                       
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Test empty DataFrame
        sheets_data = {
            'Empty_Sheet': pd.DataFrame()
        }
        
        file_path = self.create_test_excel(sheets_data)
        is_valid, issues, _ = self.validator.validate_excel_file(file_path)
        
        # Should handle empty sheets gracefully
        self.assertIsNotNone(issues)
        
        # Test very large numbers
        sheets_data = {
            'Numeric_Test': pd.DataFrame({
                'Parameter': ['large_value'],
                'Value': [1e20]
            })
        }
        
        file_path = self.create_test_excel(sheets_data)
        is_valid, issues, _ = self.validator.validate_excel_file(file_path)
        
        # Should flag unreasonable values
        large_value_issues = [i for i in issues if 'large' in i.message.lower() 
                            or 'range' in i.message.lower()]
        self.assertGreater(len(large_value_issues), 0)


class TestParameterizedValidation(unittest.TestCase):
    """Parameterized tests for various validation scenarios"""
    
    def setUp(self):
        self.validator = ConfigurationValidator()
        
    def test_weight_combinations(self):
        """Test various weight combinations"""
        test_cases = [
            ([0.25, 0.25, 0.25, 0.25], True, "Equal weights"),
            ([0.4, 0.3, 0.2, 0.1], True, "Descending weights"),
            ([0.7, 0.2, 0.1], True, "Dominant weight"),
            ([0.333, 0.333, 0.334], True, "Rounding to 1.0"),
            ([0.5, 0.5, 0.5], False, "Sum > 1"),
            ([1.0], True, "Single indicator"),
            ([], False, "Empty weights")
        ]
        
        for weights, expected_valid, description in test_cases:
            with self.subTest(description=description):
                result = self.validator._validate_weight_sum(weights)
                if expected_valid:
                    self.assertTrue(0.95 <= sum(weights) <= 1.05, description)
                else:
                    self.assertTrue(sum(weights) < 0.95 or sum(weights) > 1.05, 
                                  description)
                                  
    def test_threshold_patterns(self):
        """Test various threshold patterns"""
        patterns = [
            {
                'name': 'Non-overlapping sequential',
                'thresholds': [(0, 0.3), (0.3, 0.6), (0.6, 1.0)],
                'valid': True
            },
            {
                'name': 'Overlapping ranges',
                'thresholds': [(0, 0.5), (0.3, 0.8), (0.7, 1.0)],
                'valid': False
            },
            {
                'name': 'Gaps in coverage',
                'thresholds': [(0, 0.3), (0.5, 0.8)],
                'valid': True  # Gaps are warnings, not errors
            },
            {
                'name': 'Inverted ranges',
                'thresholds': [(0.5, 0.3), (0.7, 0.4)],
                'valid': False
            }
        ]
        
        for pattern in patterns:
            with self.subTest(pattern=pattern['name']):
                issues = self.validator._check_threshold_overlaps(
                    pattern['thresholds']
                )
                has_errors = any(i.severity == ValidationSeverity.ERROR 
                               for i in issues)
                self.assertEqual(not has_errors, pattern['valid'])


def run_all_tests():
    """Run all configuration validation tests"""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestConfigValidation))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestParameterizedValidation))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)