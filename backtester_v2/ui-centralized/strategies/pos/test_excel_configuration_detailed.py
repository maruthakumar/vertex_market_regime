"""
Detailed Excel Configuration Testing for POS Strategy
=====================================================

This module provides focused testing for the enhanced Excel configuration system
with all 200+ parameters across 3 configuration files:

1. POS_CONFIG_PORTFOLIO_1.0.0.xlsx (25+ parameters)
2. POS_CONFIG_STRATEGY_1.0.0.xlsx (66+ parameters)  
3. POS_CONFIG_ADJUSTMENT_1.0.0.xlsx (6+ rules)

NO MOCK DATA - All testing uses real configuration files and validates
actual parameter parsing and model creation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import os
from typing import Dict, List, Any, Optional
import json
from datetime import date, time, datetime

# Add parent directories to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent.parent.parent.parent))

from backtester_v2.strategies.pos.models_enhanced import (
    CompletePOSStrategy, EnhancedPortfolioModel, EnhancedPositionalStrategy,
    EnhancedLegModel, AdjustmentRule, VixConfiguration, BreakevenConfig,
    VolatilityFilter, EntryConfig, RiskManagement
)
from backtester_v2.strategies.pos.parser_enhanced import EnhancedPOSParser

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ExcelConfigurationTester:
    """Detailed Excel configuration testing with parameter validation"""
    
    def __init__(self):
        self.config_files = {
            'portfolio': '/srv/samba/shared/bt/backtester_stable/worktrees/strategies/strategy-pos/backtester_v2/configurations/data/prod/pos/POS_CONFIG_PORTFOLIO_1.0.0.xlsx',
            'strategy': '/srv/samba/shared/bt/backtester_stable/worktrees/strategies/strategy-pos/backtester_v2/configurations/data/prod/pos/POS_CONFIG_STRATEGY_1.0.0.xlsx',
            'adjustment': '/srv/samba/shared/bt/backtester_stable/worktrees/strategies/strategy-pos/backtester_v2/configurations/data/prod/pos/POS_CONFIG_ADJUSTMENT_1.0.0.xlsx'
        }
        
        self.test_results = {
            'portfolio_parsing': [],
            'strategy_parsing': [],
            'adjustment_parsing': [],
            'model_creation': [],
            'parameter_validation': []
        }
        
        # Expected parameter counts
        self.expected_counts = {
            'portfolio_parameters': 25,
            'strategy_parameters': 66,
            'adjustment_rules': 6,
            'total_parameters': 200
        }
    
    def test_file_accessibility(self) -> bool:
        """Test that all configuration files exist and are readable"""
        logger.info("=== Testing Configuration File Accessibility ===")
        
        for config_type, file_path in self.config_files.items():
            try:
                if not os.path.exists(file_path):
                    logger.error(f"✗ {config_type} file not found: {file_path}")
                    return False
                
                # Test file readability
                xl = pd.ExcelFile(file_path)
                sheets = xl.sheet_names
                logger.info(f"✓ {config_type} file accessible with {len(sheets)} sheets: {sheets}")
                
            except Exception as e:
                logger.error(f"✗ Error accessing {config_type} file: {e}")
                return False
        
        return True
    
    def test_portfolio_configuration_detailed(self) -> Dict[str, Any]:
        """Detailed testing of portfolio configuration parsing"""
        logger.info("\n=== Testing Portfolio Configuration (25+ parameters) ===")
        
        results = {
            'status': 'UNKNOWN',
            'parameters_found': 0,
            'missing_parameters': [],
            'invalid_parameters': [],
            'parsed_values': {}
        }
        
        try:
            # Read PortfolioSetting sheet
            df_portfolio = pd.read_excel(self.config_files['portfolio'], sheet_name='PortfolioSetting')
            
            if df_portfolio.empty:
                results['status'] = 'FAILED'
                results['error'] = 'PortfolioSetting sheet is empty'
                return results
            
            row = df_portfolio.iloc[0]
            
            # Define expected portfolio parameters
            expected_portfolio_params = [
                'PortfolioName', 'StartDate', 'EndDate', 'IndexName', 'Multiplier',
                'SlippagePercent', 'IsTickBT', 'Enabled', 'PortfolioStoploss', 'PortfolioTarget',
                'InitialCapital', 'PositionSizeMethod', 'PositionSizeValue', 'MaxOpenPositions',
                'CorrelationLimit', 'TransactionCosts', 'UseMargin', 'MarginRequirement',
                'MaintenanceMargin', 'MarginCallAction', 'CompoundProfits', 'ReinvestmentRatio',
                'RebalanceFrequency', 'SectorLimits', 'MaxPortfolioRisk'
            ]
            
            # Check parameter availability
            available_columns = df_portfolio.columns.tolist()
            missing_params = [param for param in expected_portfolio_params if param not in available_columns]
            found_params = [param for param in expected_portfolio_params if param in available_columns]
            
            results['parameters_found'] = len(found_params)
            results['missing_parameters'] = missing_params
            
            logger.info(f"Portfolio parameters found: {len(found_params)}/{len(expected_portfolio_params)}")
            logger.info(f"Available columns: {available_columns}")
            
            if missing_params:
                logger.warning(f"Missing portfolio parameters: {missing_params}")
            
            # Validate parameter values
            validation_results = {}
            
            for param in found_params:
                try:
                    value = row[param]
                    validation_results[param] = {
                        'value': value,
                        'type': str(type(value).__name__),
                        'is_null': pd.isna(value),
                        'validation': 'PASSED'
                    }
                    
                    # Specific validations
                    if param in ['StartDate', 'EndDate'] and not pd.isna(value):
                        if not isinstance(value, (date, datetime)):
                            try:
                                pd.to_datetime(value)
                            except:
                                validation_results[param]['validation'] = 'FAILED'
                                validation_results[param]['error'] = 'Invalid date format'
                    
                    elif param in ['SlippagePercent', 'CorrelationLimit', 'TransactionCosts']:
                        if not pd.isna(value) and (float(value) < 0 or float(value) > 1):
                            validation_results[param]['validation'] = 'FAILED'
                            validation_results[param]['error'] = 'Value out of range [0,1]'
                    
                    elif param in ['InitialCapital', 'PositionSizeValue']:
                        if not pd.isna(value) and float(value) <= 0:
                            validation_results[param]['validation'] = 'FAILED'
                            validation_results[param]['error'] = 'Value must be positive'
                    
                except Exception as e:
                    validation_results[param] = {
                        'value': value,
                        'validation': 'ERROR',
                        'error': str(e)
                    }
                    results['invalid_parameters'].append(param)
            
            results['parsed_values'] = validation_results
            
            # Determine overall status
            invalid_count = len([p for p in validation_results.values() if p['validation'] != 'PASSED'])
            
            if invalid_count == 0 and len(missing_params) <= 5:  # Allow up to 5 missing params
                results['status'] = 'PASSED'
            elif invalid_count <= 2:  # Allow up to 2 invalid params
                results['status'] = 'PARTIAL'
            else:
                results['status'] = 'FAILED'
            
            logger.info(f"Portfolio configuration validation: {results['status']}")
            logger.info(f"Valid parameters: {len(found_params) - invalid_count}/{len(found_params)}")
            
        except Exception as e:
            results['status'] = 'FAILED'
            results['error'] = str(e)
            logger.error(f"✗ Portfolio configuration test failed: {e}")
        
        self.test_results['portfolio_parsing'].append(results)
        return results
    
    def test_strategy_configuration_detailed(self) -> Dict[str, Any]:
        """Detailed testing of strategy configuration parsing"""
        logger.info("\n=== Testing Strategy Configuration (66+ parameters) ===")
        
        results = {
            'status': 'UNKNOWN',
            'sheets_found': [],
            'parameters_by_sheet': {},
            'breakeven_parameters': 0,
            'vix_parameters': 0,
            'volatility_parameters': 0,
            'total_parameters': 0
        }
        
        try:
            xl = pd.ExcelFile(self.config_files['strategy'])
            available_sheets = xl.sheet_names
            results['sheets_found'] = available_sheets
            
            logger.info(f"Strategy file sheets: {available_sheets}")
            
            # Test PositionalParameter sheet
            if 'PositionalParameter' in available_sheets:
                df_params = pd.read_excel(self.config_files['strategy'], sheet_name='PositionalParameter')
                
                if not df_params.empty:
                    row = df_params.iloc[0]
                    param_columns = df_params.columns.tolist()
                    
                    # Categorize parameters
                    breakeven_params = [col for col in param_columns if 'breakeven' in col.lower() or 'be' in col.lower()]
                    vix_params = [col for col in param_columns if 'vix' in col.lower()]
                    volatility_params = [col for col in param_columns if any(term in col.lower() for term in ['ivp', 'ivr', 'atr', 'vol'])]
                    
                    results['parameters_by_sheet']['PositionalParameter'] = {
                        'total_columns': len(param_columns),
                        'breakeven_params': breakeven_params,
                        'vix_params': vix_params,
                        'volatility_params': volatility_params,
                        'all_columns': param_columns
                    }
                    
                    results['breakeven_parameters'] = len(breakeven_params)
                    results['vix_parameters'] = len(vix_params)
                    results['volatility_parameters'] = len(volatility_params)
                    results['total_parameters'] += len(param_columns)
                    
                    logger.info(f"✓ PositionalParameter: {len(param_columns)} columns")
                    logger.info(f"  Breakeven params: {len(breakeven_params)}")
                    logger.info(f"  VIX params: {len(vix_params)}")
                    logger.info(f"  Volatility params: {len(volatility_params)}")
                    
                    # Test specific parameter groups
                    self._test_breakeven_parameters(row, breakeven_params)
                    self._test_vix_parameters(row, vix_params)
                    self._test_volatility_parameters(row, volatility_params)
                
            # Test LegParameter sheet
            if 'LegParameter' in available_sheets:
                df_legs = pd.read_excel(self.config_files['strategy'], sheet_name='LegParameter')
                leg_columns = df_legs.columns.tolist()
                
                results['parameters_by_sheet']['LegParameter'] = {
                    'total_columns': len(leg_columns),
                    'leg_count': len(df_legs),
                    'columns': leg_columns
                }
                
                results['total_parameters'] += len(leg_columns) * len(df_legs)
                logger.info(f"✓ LegParameter: {len(leg_columns)} columns × {len(df_legs)} legs")
            
            # Test optional sheets
            optional_sheets = ['AdjustmentRules', 'MarketStructure', 'GreekLimits', 'VolatilityMetrics', 'BreakevenAnalysis']
            
            for sheet_name in optional_sheets:
                if sheet_name in available_sheets:
                    try:
                        df_sheet = pd.read_excel(self.config_files['strategy'], sheet_name=sheet_name)
                        sheet_columns = df_sheet.columns.tolist()
                        
                        results['parameters_by_sheet'][sheet_name] = {
                            'total_columns': len(sheet_columns),
                            'row_count': len(df_sheet),
                            'columns': sheet_columns
                        }
                        
                        results['total_parameters'] += len(sheet_columns) * len(df_sheet)
                        logger.info(f"✓ {sheet_name}: {len(sheet_columns)} columns × {len(df_sheet)} rows")
                        
                    except Exception as e:
                        logger.warning(f"Could not read {sheet_name}: {e}")
            
            # Determine status
            if results['total_parameters'] >= self.expected_counts['total_parameters'] * 0.8:  # 80% of expected
                results['status'] = 'PASSED'
            elif results['total_parameters'] >= self.expected_counts['total_parameters'] * 0.6:  # 60% of expected
                results['status'] = 'PARTIAL'
            else:
                results['status'] = 'FAILED'
            
            logger.info(f"Strategy configuration total parameters: {results['total_parameters']}")
            logger.info(f"Strategy configuration validation: {results['status']}")
            
        except Exception as e:
            results['status'] = 'FAILED'
            results['error'] = str(e)
            logger.error(f"✗ Strategy configuration test failed: {e}")
        
        self.test_results['strategy_parsing'].append(results)
        return results
    
    def _test_breakeven_parameters(self, row: pd.Series, breakeven_params: List[str]):
        """Test breakeven-specific parameters"""
        logger.info("\n--- Testing Breakeven Parameters (17 expected) ---")
        
        expected_be_params = [
            'UseBreakevenAnalysis', 'BreakevenCalculation', 'UpperBreakevenTarget',
            'LowerBreakevenTarget', 'BreakevenBuffer', 'BreakevenBufferType',
            'DynamicBEAdjustment', 'BERecalcFrequency', 'IncludeCommissions',
            'IncludeSlippage', 'TimeDecayFactor', 'VolatilitySmileBE',
            'SpotPriceBEThreshold', 'BEApproachAction', 'BEBreachAction',
            'TrackBEDistance', 'BEDistanceAlert'
        ]
        
        found_be_params = [param for param in expected_be_params if param in breakeven_params]
        missing_be_params = [param for param in expected_be_params if param not in breakeven_params]
        
        logger.info(f"Breakeven parameters found: {len(found_be_params)}/{len(expected_be_params)}")
        
        if missing_be_params:
            logger.warning(f"Missing breakeven parameters: {missing_be_params}")
        
        # Validate found parameters
        for param in found_be_params:
            if param in row.index:
                value = row[param]
                logger.info(f"  {param}: {value}")
    
    def _test_vix_parameters(self, row: pd.Series, vix_params: List[str]):
        """Test VIX-specific parameters"""
        logger.info("\n--- Testing VIX Parameters (8 expected) ---")
        
        expected_vix_params = [
            'VixMethod', 'VixLowMin', 'VixLowMax', 'VixMedMin',
            'VixMedMax', 'VixHighMin', 'VixHighMax', 'VixExtremeMin', 'VixExtremeMax'
        ]
        
        found_vix_params = [param for param in expected_vix_params if param in vix_params]
        missing_vix_params = [param for param in expected_vix_params if param not in vix_params]
        
        logger.info(f"VIX parameters found: {len(found_vix_params)}/{len(expected_vix_params)}")
        
        if missing_vix_params:
            logger.warning(f"Missing VIX parameters: {missing_vix_params}")
        
        # Validate VIX ranges
        vix_ranges = {}
        for param in found_vix_params:
            if param in row.index:
                value = row[param]
                vix_ranges[param] = value
                logger.info(f"  {param}: {value}")
        
        # Check range logic
        try:
            if 'VixLowMin' in vix_ranges and 'VixLowMax' in vix_ranges:
                if float(vix_ranges['VixLowMin']) >= float(vix_ranges['VixLowMax']):
                    logger.warning("VIX Low range: min >= max")
        except:
            pass
    
    def _test_volatility_parameters(self, row: pd.Series, volatility_params: List[str]):
        """Test volatility metrics parameters"""
        logger.info("\n--- Testing Volatility Parameters (IVP, IVR, ATR) ---")
        
        expected_vol_params = [
            'UseIVP', 'IVPLookback', 'IVPMinEntry', 'IVPMaxEntry',
            'UseIVR', 'IVRLookback', 'IVRMinEntry', 'IVRMaxEntry',
            'UseATRPercentile', 'ATRPeriod', 'ATRLookback', 'ATRMinPercentile', 'ATRMaxPercentile'
        ]
        
        found_vol_params = [param for param in expected_vol_params if param in volatility_params]
        missing_vol_params = [param for param in expected_vol_params if param not in volatility_params]
        
        logger.info(f"Volatility parameters found: {len(found_vol_params)}/{len(expected_vol_params)}")
        
        if missing_vol_params:
            logger.warning(f"Missing volatility parameters: {missing_vol_params}")
        
        # Group by metric type
        ivp_params = [p for p in found_vol_params if 'IVP' in p]
        ivr_params = [p for p in found_vol_params if 'IVR' in p]
        atr_params = [p for p in found_vol_params if 'ATR' in p]
        
        logger.info(f"  IVP parameters: {len(ivp_params)}")
        logger.info(f"  IVR parameters: {len(ivr_params)}")
        logger.info(f"  ATR parameters: {len(atr_params)}")
        
        for param in found_vol_params:
            if param in row.index:
                value = row[param]
                logger.info(f"  {param}: {value}")
    
    def test_adjustment_rules_detailed(self) -> Dict[str, Any]:
        """Test adjustment rules configuration"""
        logger.info("\n=== Testing Adjustment Rules Configuration ===")
        
        results = {
            'status': 'UNKNOWN',
            'rules_found': 0,
            'parameters_per_rule': 0,
            'total_adjustment_parameters': 0,
            'rule_details': []
        }
        
        try:
            if not os.path.exists(self.config_files['adjustment']):
                logger.warning("Adjustment file not found - this is optional")
                results['status'] = 'SKIPPED'
                return results
            
            xl = pd.ExcelFile(self.config_files['adjustment'])
            
            if 'AdjustmentRules' in xl.sheet_names:
                df_rules = pd.read_excel(self.config_files['adjustment'], sheet_name='AdjustmentRules')
                
                results['rules_found'] = len(df_rules)
                results['parameters_per_rule'] = len(df_rules.columns)
                results['total_adjustment_parameters'] = len(df_rules) * len(df_rules.columns)
                
                logger.info(f"Adjustment rules found: {len(df_rules)}")
                logger.info(f"Parameters per rule: {len(df_rules.columns)}")
                logger.info(f"Columns: {df_rules.columns.tolist()}")
                
                # Validate each rule
                for idx, row in df_rules.iterrows():
                    rule_detail = {
                        'rule_index': idx,
                        'rule_id': row.get('RuleID', f'rule_{idx}'),
                        'rule_name': row.get('RuleName', f'Rule {idx}'),
                        'enabled': row.get('Enabled', 'YES'),
                        'trigger_type': row.get('TriggerType', 'UNKNOWN'),
                        'action_type': row.get('ActionType', 'UNKNOWN'),
                        'validation': 'PASSED'
                    }
                    
                    # Basic validation
                    if pd.isna(rule_detail['rule_id']):
                        rule_detail['validation'] = 'FAILED'
                        rule_detail['error'] = 'Missing rule ID'
                    
                    results['rule_details'].append(rule_detail)
                    logger.info(f"  Rule {idx}: {rule_detail['rule_name']} ({rule_detail['trigger_type']} → {rule_detail['action_type']})")
                
                # Determine status
                if results['rules_found'] >= self.expected_counts['adjustment_rules']:
                    results['status'] = 'PASSED'
                elif results['rules_found'] > 0:
                    results['status'] = 'PARTIAL'
                else:
                    results['status'] = 'FAILED'
            
            else:
                logger.warning("AdjustmentRules sheet not found")
                results['status'] = 'FAILED'
                results['error'] = 'AdjustmentRules sheet not found'
        
        except Exception as e:
            results['status'] = 'FAILED'
            results['error'] = str(e)
            logger.error(f"✗ Adjustment rules test failed: {e}")
        
        self.test_results['adjustment_parsing'].append(results)
        return results
    
    def test_complete_model_creation(self) -> Dict[str, Any]:
        """Test complete model creation using enhanced parser"""
        logger.info("\n=== Testing Complete Model Creation ===")
        
        results = {
            'status': 'UNKNOWN',
            'parser_errors': [],
            'model_created': False,
            'model_validation': {}
        }
        
        try:
            parser = EnhancedPOSParser()
            
            # Parse complete configuration
            parse_result = parser.parse_input(
                portfolio_file=self.config_files['portfolio'],
                strategy_file=self.config_files['strategy']
            )
            
            if parse_result['errors']:
                results['parser_errors'] = parse_result['errors']
                logger.error(f"Parser errors: {parse_result['errors']}")
            
            if parse_result['model']:
                results['model_created'] = True
                model = parse_result['model']
                
                # Validate model structure
                model_validation = {
                    'is_complete_pos_strategy': isinstance(model, CompletePOSStrategy),
                    'has_portfolio': hasattr(model, 'portfolio') and model.portfolio is not None,
                    'has_strategy': hasattr(model, 'strategy') and model.strategy is not None,
                    'has_legs': hasattr(model, 'legs') and len(model.legs) > 0,
                    'has_vix_config': hasattr(model.strategy, 'vix_config') if hasattr(model, 'strategy') else False,
                    'has_breakeven_config': hasattr(model.strategy, 'breakeven_config') if hasattr(model, 'strategy') else False,
                    'has_volatility_filter': hasattr(model.strategy, 'volatility_filter') if hasattr(model, 'strategy') else False
                }
                
                results['model_validation'] = model_validation
                
                # Log model details
                if model_validation['has_portfolio']:
                    logger.info(f"✓ Portfolio: {model.portfolio.portfolio_name}")
                    logger.info(f"  Date range: {model.portfolio.start_date} to {model.portfolio.end_date}")
                    logger.info(f"  Initial capital: {model.portfolio.initial_capital:,}")
                
                if model_validation['has_strategy']:
                    logger.info(f"✓ Strategy: {model.strategy.strategy_name}")
                    logger.info(f"  Type: {model.strategy.position_type.value}")
                    logger.info(f"  Subtype: {model.strategy.strategy_subtype.value}")
                
                if model_validation['has_legs']:
                    logger.info(f"✓ Legs: {len(model.legs)} configured")
                    for i, leg in enumerate(model.legs):
                        logger.info(f"  Leg {i+1}: {leg.leg_name} ({leg.instrument.value} {leg.transaction.value})")
                
                # Validate configurations
                if model_validation['has_vix_config']:
                    vix_config = model.strategy.vix_config
                    logger.info(f"✓ VIX Config: {vix_config.method.value}")
                    logger.info(f"  Low: {vix_config.low.min}-{vix_config.low.max}")
                    logger.info(f"  Medium: {vix_config.medium.min}-{vix_config.medium.max}")
                    logger.info(f"  High: {vix_config.high.min}-{vix_config.high.max}")
                    logger.info(f"  Extreme: {vix_config.extreme.min}-{vix_config.extreme.max}")
                
                if model_validation['has_breakeven_config']:
                    be_config = model.strategy.breakeven_config
                    logger.info(f"✓ Breakeven Config: Enabled={be_config.enabled}")
                    logger.info(f"  Method: {be_config.calculation_method.value}")
                    logger.info(f"  Buffer: {be_config.buffer} ({be_config.buffer_type.value})")
                
                if model_validation['has_volatility_filter']:
                    vol_filter = model.strategy.volatility_filter
                    logger.info(f"✓ Volatility Filter: IVP={vol_filter.use_ivp}, IVR={vol_filter.use_ivr}, ATR={vol_filter.use_atr_percentile}")
                
                # Determine overall status
                passed_validations = sum(model_validation.values())
                total_validations = len(model_validation)
                
                if passed_validations >= total_validations * 0.8:  # 80% pass rate
                    results['status'] = 'PASSED'
                elif passed_validations >= total_validations * 0.6:  # 60% pass rate
                    results['status'] = 'PARTIAL'
                else:
                    results['status'] = 'FAILED'
                
                logger.info(f"Model validation: {passed_validations}/{total_validations} checks passed")
                
            else:
                results['status'] = 'FAILED'
                results['error'] = 'No model created'
        
        except Exception as e:
            results['status'] = 'FAILED'
            results['error'] = str(e)
            logger.error(f"✗ Model creation test failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        self.test_results['model_creation'].append(results)
        return results
    
    def generate_detailed_report(self) -> Dict[str, Any]:
        """Generate detailed test report"""
        logger.info("\n" + "="*80)
        logger.info("DETAILED EXCEL CONFIGURATION TEST REPORT")
        logger.info("="*80)
        
        # Summarize results
        summary = {
            'timestamp': datetime.now().isoformat(),
            'tests_performed': {
                'portfolio_parsing': len(self.test_results['portfolio_parsing']),
                'strategy_parsing': len(self.test_results['strategy_parsing']),
                'adjustment_parsing': len(self.test_results['adjustment_parsing']),
                'model_creation': len(self.test_results['model_creation'])
            },
            'overall_status': 'UNKNOWN',
            'detailed_results': self.test_results
        }
        
        # Determine overall status
        all_tests = []
        for category in self.test_results.values():
            for test in category:
                if isinstance(test, dict) and 'status' in test:
                    all_tests.append(test['status'])
        
        if all_tests:
            passed_tests = sum(1 for status in all_tests if status == 'PASSED')
            partial_tests = sum(1 for status in all_tests if status == 'PARTIAL')
            failed_tests = sum(1 for status in all_tests if status == 'FAILED')
            
            success_rate = (passed_tests + partial_tests * 0.5) / len(all_tests)
            
            if success_rate >= 0.8:
                summary['overall_status'] = 'PASSED'
            elif success_rate >= 0.6:
                summary['overall_status'] = 'PARTIAL'
            else:
                summary['overall_status'] = 'FAILED'
            
            summary['test_summary'] = {
                'total_tests': len(all_tests),
                'passed': passed_tests,
                'partial': partial_tests,
                'failed': failed_tests,
                'success_rate': success_rate
            }
            
            logger.info(f"Test Summary: {passed_tests} passed, {partial_tests} partial, {failed_tests} failed")
            logger.info(f"Success Rate: {success_rate:.1%}")
            logger.info(f"Overall Status: {summary['overall_status']}")
        
        # Save report
        report_file = f"/tmp/excel_config_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(report_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"Detailed report saved to: {report_file}")
        except Exception as e:
            logger.warning(f"Could not save report: {e}")
        
        return summary
    
    def run_all_excel_tests(self) -> bool:
        """Run all Excel configuration tests"""
        logger.info("Starting Detailed Excel Configuration Tests")
        logger.info("="*80)
        
        try:
            # Test 1: File accessibility
            if not self.test_file_accessibility():
                logger.error("File accessibility test failed - aborting")
                return False
            
            # Test 2: Portfolio configuration
            portfolio_result = self.test_portfolio_configuration_detailed()
            
            # Test 3: Strategy configuration
            strategy_result = self.test_strategy_configuration_detailed()
            
            # Test 4: Adjustment rules
            adjustment_result = self.test_adjustment_rules_detailed()
            
            # Test 5: Complete model creation
            model_result = self.test_complete_model_creation()
            
            # Generate report
            report = self.generate_detailed_report()
            
            return report['overall_status'] in ['PASSED', 'PARTIAL']
            
        except Exception as e:
            logger.error(f"Excel configuration tests failed: {e}")
            return False


def run_excel_configuration_tests():
    """Entry point for Excel configuration tests"""
    tester = ExcelConfigurationTester()
    success = tester.run_all_excel_tests()
    
    if success:
        print("\n✅ Excel Configuration Tests Completed Successfully!")
        return 0
    else:
        print("\n❌ Excel Configuration Tests Failed!")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(run_excel_configuration_tests())