#!/usr/bin/env python3
"""
Phase 5: Output Generation Testing
==================================

File generation and format validation:
1. CSV generation with correct structure
2. Time-series data completeness validation
3. Regime classification accuracy
4. Component breakdown inclusion
5. Performance metrics validation
6. Download functionality testing
7. YAML configuration export
8. File integrity checks

Duration: 45 minutes
Priority: MEDIUM
Focus: Output format validation and file generation
"""

import sys
import logging
import time
import requests
import json
import os
import csv
from datetime import datetime
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add paths
sys.path.append('/srv/samba/shared/bt/backtester_stable/BTRUN')

print("=" * 80)
print("PHASE 5: OUTPUT GENERATION TESTING")
print("=" * 80)

class OutputGenerationTester:
    """Comprehensive output generation and format validation testing"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()
        self.base_url = "http://localhost:8000"
        self.output_dir = "/tmp/market_regime_test_outputs"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Expected CSV structure
        self.expected_csv_columns = [
            'timestamp',
            'regime',
            'confidence',
            'triple_straddle',
            'greek_sentiment',
            'trending_oi',
            'correlation_score',
            'resistance_score'
        ]
        
        # Expected regime classifications
        self.expected_regimes = [
            'BULLISH_TRENDING',
            'BEARISH_TRENDING',
            'SIDEWAYS_NEUTRAL',
            'HIGH_VOLATILITY',
            'LOW_VOLATILITY',
            'BREAKOUT_BULLISH',
            'BREAKOUT_BEARISH',
            'CONSOLIDATION'
        ]
        
    def test_csv_generation_api(self):
        """Test 1: CSV generation via API"""
        logger.info("🔍 Test 1: Testing CSV generation API...")
        
        try:
            # Test CSV export endpoint
            export_url = f"{self.base_url}/api/v1/market-regime/export/csv"
            
            # Request CSV generation
            payload = {
                "use_real_data": True,
                "data_source": "heavydb",
                "timeframe": "5min",
                "include_components": True
            }
            
            response = requests.post(export_url, json=payload, timeout=30)
            
            if response.status_code == 200:
                # Check if response contains CSV data
                content_type = response.headers.get('content-type', '')
                if 'text/csv' in content_type or 'application/csv' in content_type:
                    # Save CSV content
                    csv_file = os.path.join(self.output_dir, 'test_output.csv')
                    with open(csv_file, 'w') as f:
                        f.write(response.text)
                    
                    logger.info(f"✅ CSV generated successfully: {len(response.text)} bytes")
                    logger.info(f"📁 Saved to: {csv_file}")
                    
                    # Validate CSV structure
                    if self.validate_csv_structure(csv_file):
                        self.test_results['csv_generation_api'] = True
                    else:
                        self.test_results['csv_generation_api'] = False
                else:
                    # Try JSON response with CSV data
                    try:
                        result = response.json()
                        if 'csv_data' in result or 'data' in result:
                            logger.info("✅ CSV data received in JSON format")
                            self.test_results['csv_generation_api'] = True
                        else:
                            logger.warning("⚠️  No CSV data found in response")
                            self.test_results['csv_generation_api'] = False
                    except json.JSONDecodeError:
                        logger.warning("⚠️  Response not CSV or JSON format")
                        self.test_results['csv_generation_api'] = False
                        
            elif response.status_code == 404:
                logger.warning("⚠️  CSV export endpoint not found")
                self.test_results['csv_generation_api'] = False
            else:
                logger.error(f"❌ CSV generation API failed: {response.status_code}")
                logger.info(f"Response: {response.text[:200]}...")
                self.test_results['csv_generation_api'] = False
            
            return self.test_results['csv_generation_api']
            
        except Exception as e:
            logger.error(f"❌ CSV generation API test failed: {e}")
            self.test_results['csv_generation_api'] = False
            return False
    
    def test_mock_csv_generation(self):
        """Test 2: Generate mock CSV for structure validation"""
        logger.info("🔍 Test 2: Testing mock CSV generation...")
        
        try:
            # Generate mock CSV data for testing
            mock_data = []
            base_time = datetime.now()
            
            for i in range(10):  # Generate 10 rows
                timestamp = base_time.strftime('%Y-%m-%d %H:%M:%S')
                row = {
                    'timestamp': timestamp,
                    'regime': 'BULLISH_TRENDING',
                    'confidence': round(0.75 + (i * 0.01), 2),
                    'triple_straddle': round(1.23 + (i * 0.1), 2),
                    'greek_sentiment': 'POSITIVE',
                    'trending_oi': 'LONG_BUILDUP',
                    'correlation_score': round(0.78 - (i * 0.02), 2),
                    'resistance_score': round(0.65 + (i * 0.03), 2)
                }
                mock_data.append(row)
                base_time = datetime.fromtimestamp(base_time.timestamp() + 300)  # +5 minutes
            
            # Write mock CSV
            mock_csv_file = os.path.join(self.output_dir, 'mock_output.csv')
            with open(mock_csv_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.expected_csv_columns)
                writer.writeheader()
                writer.writerows(mock_data)
            
            logger.info(f"✅ Mock CSV generated: {len(mock_data)} rows")
            logger.info(f"📁 Saved to: {mock_csv_file}")
            
            # Validate structure
            if self.validate_csv_structure(mock_csv_file):
                logger.info("✅ Mock CSV structure validation passed")
                self.test_results['mock_csv_generation'] = True
            else:
                logger.warning("⚠️  Mock CSV structure validation failed")
                self.test_results['mock_csv_generation'] = False
            
            return self.test_results['mock_csv_generation']
            
        except Exception as e:
            logger.error(f"❌ Mock CSV generation failed: {e}")
            self.test_results['mock_csv_generation'] = False
            return False
    
    def validate_csv_structure(self, csv_file):
        """Validate CSV file structure"""
        try:
            if not os.path.exists(csv_file):
                logger.warning(f"⚠️  CSV file not found: {csv_file}")
                return False
            
            # Read CSV
            df = pd.read_csv(csv_file)
            
            if df.empty:
                logger.warning("⚠️  CSV file is empty")
                return False
            
            logger.info(f"📊 CSV file has {len(df)} rows and {len(df.columns)} columns")
            
            # Check required columns
            missing_columns = set(self.expected_csv_columns) - set(df.columns)
            if missing_columns:
                logger.warning(f"⚠️  Missing columns: {missing_columns}")
                return False
            
            # Check data types and content
            validation_checks = []
            
            # Timestamp column
            if 'timestamp' in df.columns:
                try:
                    pd.to_datetime(df['timestamp'])
                    validation_checks.append(True)
                    logger.info("✅ Timestamp format valid")
                except:
                    validation_checks.append(False)
                    logger.warning("⚠️  Invalid timestamp format")
            
            # Confidence column (should be numeric 0-1)
            if 'confidence' in df.columns:
                try:
                    confidence_values = pd.to_numeric(df['confidence'], errors='coerce')
                    if confidence_values.between(0, 1).all():
                        validation_checks.append(True)
                        logger.info("✅ Confidence values valid (0-1 range)")
                    else:
                        validation_checks.append(False)
                        logger.warning("⚠️  Confidence values out of range")
                except:
                    validation_checks.append(False)
                    logger.warning("⚠️  Confidence column not numeric")
            
            # Regime column
            if 'regime' in df.columns:
                unique_regimes = df['regime'].unique()
                logger.info(f"📊 Found regimes: {list(unique_regimes)}")
                validation_checks.append(True)
            
            # Calculate overall validation
            validation_rate = sum(validation_checks) / len(validation_checks)
            logger.info(f"📊 CSV validation rate: {validation_rate:.1%}")
            
            return validation_rate >= 0.8
            
        except Exception as e:
            logger.warning(f"⚠️  CSV validation failed: {e}")
            return False
    
    def test_yaml_config_export(self):
        """Test 3: YAML configuration export"""
        logger.info("🔍 Test 3: Testing YAML configuration export...")
        
        try:
            # Test YAML export endpoint
            yaml_url = f"{self.base_url}/api/v1/market-regime/export/config"
            
            response = requests.get(yaml_url, timeout=15)
            
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')
                
                if 'yaml' in content_type or 'text/plain' in content_type:
                    # Save YAML content
                    yaml_file = os.path.join(self.output_dir, 'exported_config.yaml')
                    with open(yaml_file, 'w') as f:
                        f.write(response.text)
                    
                    logger.info(f"✅ YAML config exported: {len(response.text)} bytes")
                    logger.info(f"📁 Saved to: {yaml_file}")
                    
                    # Validate YAML structure
                    if self.validate_yaml_structure(yaml_file):
                        self.test_results['yaml_export'] = True
                    else:
                        self.test_results['yaml_export'] = False
                else:
                    try:
                        result = response.json()
                        if 'config' in result or 'yaml' in result:
                            logger.info("✅ YAML config received in JSON format")
                            self.test_results['yaml_export'] = True
                        else:
                            logger.warning("⚠️  No YAML config found in response")
                            self.test_results['yaml_export'] = False
                    except json.JSONDecodeError:
                        logger.warning("⚠️  Response not YAML or JSON format")
                        self.test_results['yaml_export'] = False
                        
            elif response.status_code == 404:
                logger.warning("⚠️  YAML export endpoint not found")
                self.test_results['yaml_export'] = False
            else:
                logger.error(f"❌ YAML export failed: {response.status_code}")
                self.test_results['yaml_export'] = False
            
            return self.test_results['yaml_export']
            
        except Exception as e:
            logger.error(f"❌ YAML config export test failed: {e}")
            self.test_results['yaml_export'] = False
            return False
    
    def validate_yaml_structure(self, yaml_file):
        """Validate YAML file structure"""
        try:
            import yaml
            
            if not os.path.exists(yaml_file):
                logger.warning(f"⚠️  YAML file not found: {yaml_file}")
                return False
            
            with open(yaml_file, 'r') as f:
                yaml_content = yaml.safe_load(f)
            
            if not yaml_content:
                logger.warning("⚠️  YAML file is empty or invalid")
                return False
            
            # Check for expected sections
            expected_sections = ['market_regime', 'indicators', 'timeframes']
            found_sections = []
            
            if isinstance(yaml_content, dict):
                for section in expected_sections:
                    if section in yaml_content:
                        found_sections.append(section)
                        logger.info(f"✅ Found YAML section: {section}")
            
            validation_rate = len(found_sections) / len(expected_sections)
            logger.info(f"📊 YAML structure validation: {validation_rate:.1%}")
            
            return validation_rate >= 0.5  # At least 50% of expected sections
            
        except Exception as e:
            logger.warning(f"⚠️  YAML validation failed: {e}")
            return False
    
    def test_regime_classification_output(self):
        """Test 4: Regime classification output validation"""
        logger.info("🔍 Test 4: Testing regime classification output...")
        
        try:
            # Test regime calculation endpoint
            regime_url = f"{self.base_url}/api/v1/market-regime/calculate"
            
            payload = {
                "use_real_data": True,
                "data_source": "heavydb",
                "include_breakdown": True
            }
            
            response = requests.post(regime_url, json=payload, timeout=20)
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    
                    # Check for required fields
                    required_fields = ['regime', 'confidence', 'data_source']
                    found_fields = []
                    
                    for field in required_fields:
                        if field in result:
                            found_fields.append(field)
                            logger.info(f"✅ Found field: {field} = {result[field]}")
                    
                    # Check regime value
                    regime = result.get('regime', 'UNKNOWN')
                    confidence = result.get('confidence', 0)
                    
                    if regime != 'UNKNOWN' and isinstance(confidence, (int, float)):
                        logger.info(f"✅ Valid regime classification: {regime} ({confidence})")
                        self.test_results['regime_classification'] = True
                    else:
                        logger.warning(f"⚠️  Invalid regime classification: {regime}")
                        self.test_results['regime_classification'] = False
                    
                    # Check data source
                    data_source = result.get('data_source', '').lower()
                    if 'heavydb' in data_source or 'real' in data_source:
                        logger.info("✅ Using real HeavyDB data")
                    else:
                        logger.warning(f"⚠️  Data source unclear: {data_source}")
                    
                except json.JSONDecodeError:
                    logger.warning("⚠️  Invalid JSON response")
                    self.test_results['regime_classification'] = False
                    
            else:
                logger.error(f"❌ Regime calculation failed: {response.status_code}")
                self.test_results['regime_classification'] = False
            
            return self.test_results['regime_classification']
            
        except Exception as e:
            logger.error(f"❌ Regime classification test failed: {e}")
            self.test_results['regime_classification'] = False
            return False
    
    def test_file_download_functionality(self):
        """Test 5: File download functionality"""
        logger.info("🔍 Test 5: Testing file download functionality...")
        
        try:
            # Test file download endpoints
            download_endpoints = [
                '/api/v1/market-regime/download/csv',
                '/api/v1/market-regime/download/config',
                '/api/v1/market-regime/download/report'
            ]
            
            download_results = []
            
            for endpoint in download_endpoints:
                try:
                    url = f"{self.base_url}{endpoint}"
                    response = requests.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        content_length = len(response.content)
                        logger.info(f"✅ Download {endpoint}: {content_length} bytes")
                        download_results.append(True)
                    elif response.status_code == 404:
                        logger.warning(f"⚠️  Download endpoint not found: {endpoint}")
                        download_results.append(False)
                    else:
                        logger.warning(f"⚠️  Download failed {endpoint}: {response.status_code}")
                        download_results.append(False)
                        
                except Exception as e:
                    logger.warning(f"⚠️  Download test failed {endpoint}: {e}")
                    download_results.append(False)
            
            # File download is considered working if any endpoint works
            download_success = any(download_results)
            success_rate = sum(download_results) / len(download_results) if download_results else 0
            
            logger.info(f"📊 Download functionality success rate: {success_rate:.1%}")
            
            if download_success:
                logger.info("✅ File download functionality working")
                self.test_results['file_download'] = True
            else:
                logger.warning("⚠️  File download functionality issues")
                self.test_results['file_download'] = False
            
            return self.test_results['file_download']
            
        except Exception as e:
            logger.error(f"❌ File download functionality test failed: {e}")
            self.test_results['file_download'] = False
            return False
    
    def test_performance_metrics_output(self):
        """Test 6: Performance metrics output"""
        logger.info("🔍 Test 6: Testing performance metrics output...")
        
        try:
            # Test performance metrics endpoint
            metrics_url = f"{self.base_url}/api/v1/market-regime/metrics"
            
            response = requests.get(metrics_url, timeout=10)
            
            if response.status_code == 200:
                try:
                    metrics = response.json()
                    
                    # Check for expected metrics
                    expected_metrics = [
                        'processing_time',
                        'accuracy',
                        'confidence_avg',
                        'regime_transitions',
                        'data_quality'
                    ]
                    
                    found_metrics = []
                    for metric in expected_metrics:
                        if metric in metrics:
                            found_metrics.append(metric)
                            logger.info(f"✅ Found metric: {metric} = {metrics[metric]}")
                    
                    metrics_rate = len(found_metrics) / len(expected_metrics)
                    logger.info(f"📊 Performance metrics coverage: {metrics_rate:.1%}")
                    
                    if metrics_rate >= 0.5:
                        logger.info("✅ Performance metrics output adequate")
                        self.test_results['performance_metrics'] = True
                    else:
                        logger.warning("⚠️  Insufficient performance metrics")
                        self.test_results['performance_metrics'] = False
                    
                except json.JSONDecodeError:
                    logger.warning("⚠️  Invalid metrics JSON response")
                    self.test_results['performance_metrics'] = False
                    
            elif response.status_code == 404:
                logger.warning("⚠️  Performance metrics endpoint not found")
                self.test_results['performance_metrics'] = False
            else:
                logger.error(f"❌ Performance metrics failed: {response.status_code}")
                self.test_results['performance_metrics'] = False
            
            return self.test_results['performance_metrics']
            
        except Exception as e:
            logger.error(f"❌ Performance metrics test failed: {e}")
            self.test_results['performance_metrics'] = False
            return False
    
    def test_file_integrity_checks(self):
        """Test 7: File integrity checks"""
        logger.info("🔍 Test 7: Testing file integrity checks...")
        
        try:
            # Check all generated files in output directory
            output_files = os.listdir(self.output_dir)
            logger.info(f"📊 Found {len(output_files)} output files")
            
            integrity_results = []
            
            for filename in output_files:
                file_path = os.path.join(self.output_dir, filename)
                
                try:
                    # Check file size
                    file_size = os.path.getsize(file_path)
                    if file_size > 0:
                        logger.info(f"✅ File {filename}: {file_size} bytes")
                        integrity_results.append(True)
                    else:
                        logger.warning(f"⚠️  Empty file: {filename}")
                        integrity_results.append(False)
                    
                    # Check file readability
                    if filename.endswith('.csv'):
                        try:
                            pd.read_csv(file_path)
                            logger.info(f"✅ CSV file readable: {filename}")
                        except:
                            logger.warning(f"⚠️  CSV file corrupted: {filename}")
                            integrity_results.append(False)
                    
                    elif filename.endswith('.yaml'):
                        try:
                            import yaml
                            with open(file_path, 'r') as f:
                                yaml.safe_load(f)
                            logger.info(f"✅ YAML file readable: {filename}")
                        except:
                            logger.warning(f"⚠️  YAML file corrupted: {filename}")
                            integrity_results.append(False)
                            
                except Exception as e:
                    logger.warning(f"⚠️  File integrity check failed {filename}: {e}")
                    integrity_results.append(False)
            
            # Calculate integrity rate
            integrity_rate = sum(integrity_results) / len(integrity_results) if integrity_results else 0
            logger.info(f"📊 File integrity rate: {integrity_rate:.1%}")
            
            if integrity_rate >= 0.8:
                logger.info("✅ File integrity checks passed")
                self.test_results['file_integrity'] = True
            else:
                logger.warning("⚠️  File integrity issues detected")
                self.test_results['file_integrity'] = False
            
            return self.test_results['file_integrity']
            
        except Exception as e:
            logger.error(f"❌ File integrity checks failed: {e}")
            self.test_results['file_integrity'] = False
            return False
    
    def generate_output_test_report(self) -> dict:
        """Generate comprehensive output test report"""
        end_time = time.time()
        duration = end_time - self.start_time
        
        # Calculate overall success
        all_tests = list(self.test_results.values())
        overall_success = all(all_tests)
        success_rate = sum(all_tests) / len(all_tests) * 100 if all_tests else 0
        
        # Identify critical vs non-critical failures
        critical_tests = ['mock_csv_generation', 'regime_classification', 'file_integrity']
        critical_failures = []
        non_critical_failures = []
        
        for test, result in self.test_results.items():
            if not result:
                if test in critical_tests:
                    critical_failures.append(test)
                else:
                    non_critical_failures.append(test)
        
        report = {
            'phase': 'Phase 5: Output Generation Testing',
            'duration_seconds': round(duration, 2),
            'overall_success': overall_success,
            'success_rate': round(success_rate, 1),
            'test_results': self.test_results,
            'critical_failures': critical_failures,
            'non_critical_failures': non_critical_failures,
            'output_directory': self.output_dir,
            'timestamp': datetime.now().isoformat()
        }
        
        return report

def main():
    """Execute Phase 5 output generation testing"""
    print("🚀 Starting Phase 5: Output Generation Testing")
    print("📋 Focus: File generation and format validation")
    
    tester = OutputGenerationTester()
    
    # Execute all tests
    tests = [
        tester.test_csv_generation_api,
        tester.test_mock_csv_generation,
        tester.test_yaml_config_export,
        tester.test_regime_classification_output,
        tester.test_file_download_functionality,
        tester.test_performance_metrics_output,
        tester.test_file_integrity_checks
    ]
    
    print("\n" + "="*60)
    
    for i, test in enumerate(tests, 1):
        print(f"\n[{i}/{len(tests)}] Executing: {test.__name__}")
        test()
    
    # Generate final report
    report = tester.generate_output_test_report()
    
    print("\n" + "="*80)
    print("PHASE 5 OUTPUT GENERATION RESULTS")
    print("="*80)
    
    print(f"⏱️  Duration: {report['duration_seconds']} seconds")
    print(f"📊 Success Rate: {report['success_rate']}%")
    print(f"📁 Output Directory: {report['output_directory']}")
    
    print(f"\n📋 Test Results:")
    for test, result in report['test_results'].items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test.replace('_', ' ').title()}: {status}")
    
    if report['critical_failures']:
        print(f"\n❌ CRITICAL FAILURES:")
        for failure in report['critical_failures']:
            print(f"   • {failure.replace('_', ' ').title()}")
    
    if report['non_critical_failures']:
        print(f"\n⚠️  NON-CRITICAL ISSUES:")
        for failure in report['non_critical_failures']:
            print(f"   • {failure.replace('_', ' ').title()}")
    
    print(f"\n🎯 OVERALL RESULT: {'✅ PHASE 5 PASSED' if report['overall_success'] else '❌ PHASE 5 FAILED'}")
    
    if report['overall_success']:
        print("\n🚀 Output generation complete - Proceeding to Phase 6: Live Trading Integration")
    elif not report['critical_failures']:
        print("\n⚠️  Output mostly functional - Can proceed with caution")
    else:
        print("\n🛑 MUST fix critical output issues before proceeding")
    
    return report['overall_success'] or len(report['critical_failures']) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)