"""
Phase 5 Tests with REAL HeavyDB Data

CRITICAL: This test uses REAL market data from HeavyDB
NO MOCK DATA ALLOWED as per enterprise requirements

Tests all Phase 5 Validation & Integration modules:
- System Validator
- Integration Orchestrator  
- Production Deployer
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
import time
import threading
import json
import tempfile
import os
import shutil
try:
    from pyheavydb import connect
except ImportError:
    try:
        from heavyai import connect
    except ImportError:
        print("Error: No HeavyDB connector available")
        sys.exit(1)

# Add parent directories to path
current_dir = Path(__file__).parent
adaptive_dir = current_dir.parent
backtester_dir = adaptive_dir.parent.parent
sys.path.insert(0, str(adaptive_dir))
sys.path.insert(0, str(backtester_dir))

# Import Phase 5 modules
from validation.system_validator import (
    SystemValidator, ValidationLevel, ValidationStatus, ComponentType,
    ValidationTest, ValidationResult, ValidationReport
)
from validation.integration_orchestrator import (
    IntegrationOrchestrator, ComponentState, PipelineStage, OrchestrationMode,
    ComponentInfo, PipelineTask, DataFlowPath
)
from validation.production_deployer import (
    ProductionDeployer, DeploymentStage, DeploymentStatus, EnvironmentType,
    DeploymentConfig, DeploymentStep, DeploymentResult
)

# Import all previous phase modules for integration testing
from infrastructure.base_regime_detector import BaseRegimeDetector
from infrastructure.configuration_manager import ConfigurationManager
from infrastructure.historical_analyzer import HistoricalAnalyzer
from core.adaptive_scoring_layer import AdaptiveScoringLayer
from analysis.transition_matrix_analyzer import TransitionMatrixAnalyzer
from core.dynamic_boundary_optimizer import DynamicBoundaryOptimizer
from intelligence.intelligent_transition_manager import IntelligentTransitionManager
from intelligence.regime_stability_monitor import RegimeStabilityMonitor
from intelligence.adaptive_noise_filter import AdaptiveNoiseFilter
from optimization.performance_feedback_system import PerformanceFeedbackSystem
from optimization.continuous_learning_engine import ContinuousLearningEngine
from optimization.regime_optimization_scheduler import RegimeOptimizationScheduler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# HeavyDB connection parameters (from CLAUDE.md)
HEAVYDB_CONFIG = {
    'host': 'localhost',
    'port': 6274,
    'user': 'admin',
    'password': 'HyperInteractive',
    'dbname': 'heavyai'
}

# Test deployment directory
TEST_DEPLOYMENT_DIR = "/tmp/adaptive_regime_test_deployment"


class TestPhase5WithHeavyDB:
    """Test Phase 5 modules with REAL HeavyDB data"""
    
    def __init__(self):
        self.conn = None
        self.market_data = None
        
        # Phase 5 modules
        self.system_validator = None
        self.integration_orchestrator = None
        self.production_deployer = None
        
        # All system components
        self.components = {}
        
    def connect_to_heavydb(self):
        """Connect to HeavyDB"""
        try:
            self.conn = connect(
                host=HEAVYDB_CONFIG['host'],
                port=HEAVYDB_CONFIG['port'],
                user=HEAVYDB_CONFIG['user'],
                password=HEAVYDB_CONFIG['password'],
                dbname=HEAVYDB_CONFIG['dbname']
            )
            logger.info("✅ Connected to HeavyDB")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to HeavyDB: {e}")
            return False
    
    def load_market_data(self):
        """Load REAL market data from HeavyDB"""
        try:
            # Load NIFTY option chain data
            query = """
            SELECT 
                time_stamp,
                strike_price,
                spot_price,
                call_bid_price,
                call_ask_price,
                put_bid_price,
                put_ask_price,
                call_oi,
                put_oi,
                total_call_oi,
                total_put_oi,
                pcr,
                option_type,
                call_ltp,
                put_ltp,
                strike_distance
            FROM nifty_option_chain
            WHERE time_stamp >= CAST('2024-01-01' AS DATE)
            ORDER BY time_stamp
            LIMIT 10000
            """
            
            df = pd.read_sql(query, self.conn)
            self.market_data = df
            logger.info(f"✅ Loaded {len(df)} rows of REAL market data from HeavyDB")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load market data: {e}")
            return False
    
    def initialize_all_components(self):
        """Initialize all system components for validation"""
        try:
            # Phase 1 - Core Infrastructure
            logger.info("Initializing Phase 1 components...")
            self.components['base_regime_detector'] = BaseRegimeDetector()
            self.components['configuration_manager'] = ConfigurationManager()
            self.components['historical_analyzer'] = HistoricalAnalyzer()
            
            # Phase 2 - Adaptive Components
            logger.info("Initializing Phase 2 components...")
            self.components['adaptive_scoring_layer'] = AdaptiveScoringLayer()
            self.components['transition_matrix_analyzer'] = TransitionMatrixAnalyzer()
            self.components['dynamic_boundary_optimizer'] = DynamicBoundaryOptimizer()
            
            # Phase 3 - Intelligence Layer
            logger.info("Initializing Phase 3 components...")
            self.components['intelligent_transition_manager'] = IntelligentTransitionManager()
            self.components['regime_stability_monitor'] = RegimeStabilityMonitor()
            self.components['adaptive_noise_filter'] = AdaptiveNoiseFilter()
            
            # Phase 4 - Optimization & Feedback
            logger.info("Initializing Phase 4 components...")
            self.components['performance_feedback_system'] = PerformanceFeedbackSystem()
            self.components['continuous_learning_engine'] = ContinuousLearningEngine()
            self.components['regime_optimization_scheduler'] = RegimeOptimizationScheduler()
            
            logger.info("✅ All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            return False
    
    def test_system_validator(self):
        """Test System Validator with comprehensive validation"""
        logger.info("\n" + "="*80)
        logger.info("TESTING SYSTEM VALIDATOR")
        logger.info("="*80)
        
        try:
            # Initialize System Validator
            self.system_validator = SystemValidator()
            
            # Register all components
            for comp_name, comp_instance in self.components.items():
                self.system_validator.register_component(
                    component_type=comp_name,
                    component_instance=comp_instance
                )
            
            # Test 1: Basic validation
            logger.info("\n1. Running basic validation...")
            basic_report = self.system_validator.validate_system(
                validation_level=ValidationLevel.BASIC
            )
            
            logger.info(f"Basic validation completed:")
            logger.info(f"  - Total tests: {basic_report.total_tests}")
            logger.info(f"  - Passed: {basic_report.passed_tests}")
            logger.info(f"  - Failed: {basic_report.failed_tests}")
            logger.info(f"  - Overall status: {basic_report.overall_status}")
            
            # Test 2: Standard validation with market data
            logger.info("\n2. Running standard validation with real market data...")
            
            # Prepare test data
            test_data = self.prepare_validation_data()
            
            standard_report = self.system_validator.validate_system(
                validation_level=ValidationLevel.STANDARD,
                test_data=test_data
            )
            
            logger.info(f"Standard validation completed:")
            logger.info(f"  - Total tests: {standard_report.total_tests}")
            logger.info(f"  - Passed: {standard_report.passed_tests}")
            logger.info(f"  - Failed: {standard_report.failed_tests}")
            
            # Test 3: Performance benchmarking
            logger.info("\n3. Running performance benchmarking...")
            performance_report = self.system_validator.benchmark_system_performance(
                test_data=test_data
            )
            
            logger.info("Performance benchmarks:")
            for component, metrics in performance_report.items():
                logger.info(f"  {component}:")
                for metric, value in metrics.items():
                    logger.info(f"    - {metric}: {value:.4f}")
            
            # Test 4: Component-specific validation
            logger.info("\n4. Testing component-specific validation...")
            
            # Validate ASL component
            asl_result = self.system_validator.validate_component(
                component_type='adaptive_scoring_layer',
                test_data=test_data
            )
            
            logger.info(f"ASL validation result: {asl_result.status}")
            if asl_result.warnings:
                logger.info(f"  Warnings: {asl_result.warnings}")
            
            # Test 5: Generate comprehensive report
            logger.info("\n5. Generating comprehensive validation report...")
            
            # Create temporary report directory
            report_dir = tempfile.mkdtemp()
            report_path = os.path.join(report_dir, "validation_report.json")
            
            # Generate report
            self.system_validator.generate_validation_report(
                output_path=report_path,
                include_performance=True,
                include_recommendations=True
            )
            
            logger.info(f"✅ Validation report generated at: {report_path}")
            
            # Clean up
            shutil.rmtree(report_dir)
            
            logger.info("\n✅ System Validator tests completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ System Validator test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_integration_orchestrator(self):
        """Test Integration Orchestrator with full pipeline"""
        logger.info("\n" + "="*80)
        logger.info("TESTING INTEGRATION ORCHESTRATOR")
        logger.info("="*80)
        
        try:
            # Initialize Integration Orchestrator
            self.integration_orchestrator = IntegrationOrchestrator()
            
            # Test 1: Component registration
            logger.info("\n1. Registering all components...")
            
            for comp_name, comp_instance in self.components.items():
                success = self.integration_orchestrator.register_component(
                    component_id=comp_name,
                    component_type=comp_name,
                    module_reference=comp_instance,
                    dependencies=self.get_component_dependencies(comp_name)
                )
                
                if success:
                    logger.info(f"  ✅ Registered: {comp_name}")
                else:
                    logger.error(f"  ❌ Failed to register: {comp_name}")
            
            # Test 2: Initialize orchestrator
            logger.info("\n2. Initializing orchestrator...")
            success = self.integration_orchestrator.initialize()
            logger.info(f"Orchestrator initialization: {'✅ Success' if success else '❌ Failed'}")
            
            # Test 3: Start orchestrator
            logger.info("\n3. Starting orchestrator...")
            self.integration_orchestrator.start()
            
            # Give time for initialization
            time.sleep(2)
            
            # Test 4: Process market data through pipeline
            logger.info("\n4. Processing market data through complete pipeline...")
            
            # Process batches of market data
            batch_size = 100
            num_batches = min(5, len(self.market_data) // batch_size)
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size
                batch_data = self.market_data.iloc[start_idx:end_idx]
                
                # Submit batch to orchestrator
                task_id = self.integration_orchestrator.submit_data_batch(
                    data=batch_data,
                    priority=5
                )
                
                logger.info(f"  Submitted batch {i+1}/{num_batches} (task_id: {task_id})")
                
                # Wait for processing
                time.sleep(0.5)
            
            # Test 5: Get orchestrator status
            logger.info("\n5. Checking orchestrator status...")
            status = self.integration_orchestrator.get_orchestrator_status()
            
            logger.info(f"Orchestrator Status:")
            logger.info(f"  - State: {status['state']}")
            logger.info(f"  - Tasks processed: {status['tasks_processed']}")
            logger.info(f"  - Tasks pending: {status['tasks_pending']}")
            logger.info(f"  - Active components: {status['active_components']}")
            
            # Test 6: Component health check
            logger.info("\n6. Checking component health...")
            health_status = self.integration_orchestrator.check_component_health()
            
            for comp_id, health in health_status.items():
                logger.info(f"  {comp_id}: {health['state']} (errors: {health['error_count']})")
            
            # Test 7: Test different orchestration modes
            logger.info("\n7. Testing orchestration modes...")
            
            # Switch to real-time mode
            self.integration_orchestrator.set_orchestration_mode(
                OrchestrationMode.REAL_TIME
            )
            logger.info("  Switched to REAL_TIME mode")
            
            # Process single data point
            single_data = self.market_data.iloc[0:1]
            task_id = self.integration_orchestrator.submit_data_batch(
                data=single_data,
                priority=10
            )
            
            time.sleep(1)
            
            # Test 8: Performance metrics
            logger.info("\n8. Collecting performance metrics...")
            metrics = self.integration_orchestrator.get_performance_metrics()
            
            logger.info("Pipeline Performance:")
            for stage, perf in metrics.items():
                logger.info(f"  {stage}:")
                logger.info(f"    - Avg latency: {perf.get('avg_latency', 0):.2f}ms")
                logger.info(f"    - Throughput: {perf.get('throughput', 0):.2f} items/sec")
            
            # Test 9: Stop orchestrator
            logger.info("\n9. Stopping orchestrator...")
            self.integration_orchestrator.stop()
            
            logger.info("\n✅ Integration Orchestrator tests completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Integration Orchestrator test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_production_deployer(self):
        """Test Production Deployer with test deployment"""
        logger.info("\n" + "="*80)
        logger.info("TESTING PRODUCTION DEPLOYER")
        logger.info("="*80)
        
        try:
            # Create test deployment configuration
            test_config = DeploymentConfig(
                environment=EnvironmentType.TESTING,
                deployment_path=TEST_DEPLOYMENT_DIR,
                config_path=os.path.join(TEST_DEPLOYMENT_DIR, "config"),
                log_path=os.path.join(TEST_DEPLOYMENT_DIR, "logs"),
                data_path=os.path.join(TEST_DEPLOYMENT_DIR, "data"),
                heavydb_host=HEAVYDB_CONFIG['host'],
                heavydb_port=HEAVYDB_CONFIG['port'],
                heavydb_user=HEAVYDB_CONFIG['user'],
                heavydb_password=HEAVYDB_CONFIG['password'],
                heavydb_database=HEAVYDB_CONFIG['dbname']
            )
            
            # Initialize Production Deployer
            self.production_deployer = ProductionDeployer(test_config)
            
            # Test 1: Validate environment
            logger.info("\n1. Validating deployment environment...")
            env_valid = self.production_deployer.validate_environment()
            logger.info(f"Environment validation: {'✅ Passed' if env_valid else '❌ Failed'}")
            
            # Test 2: Check dependencies
            logger.info("\n2. Checking system dependencies...")
            deps_valid = self.production_deployer.check_dependencies()
            logger.info(f"Dependencies check: {'✅ All present' if deps_valid else '❌ Missing dependencies'}")
            
            # Test 3: Validate database connection
            logger.info("\n3. Validating HeavyDB connection...")
            db_valid = self.production_deployer.validate_database_connection()
            logger.info(f"Database connection: {'✅ Connected' if db_valid else '❌ Failed'}")
            
            # Test 4: Plan deployment
            logger.info("\n4. Planning deployment...")
            deployment_plan = self.production_deployer.plan_deployment()
            
            logger.info(f"Deployment plan created with {len(deployment_plan.steps)} steps:")
            for step in deployment_plan.steps[:5]:  # Show first 5 steps
                logger.info(f"  - {step.stage.value}: {step.description}")
            
            # Test 5: Execute test deployment
            logger.info("\n5. Executing test deployment...")
            
            # Create deployment with dry_run first
            logger.info("  Running dry-run deployment...")
            dry_run_result = self.production_deployer.deploy(dry_run=True)
            
            if dry_run_result.status == DeploymentStatus.COMPLETED:
                logger.info("  ✅ Dry-run successful")
                
                # Execute actual test deployment
                logger.info("  Executing actual deployment...")
                deployment_result = self.production_deployer.deploy(dry_run=False)
                
                logger.info(f"\nDeployment Result:")
                logger.info(f"  - Status: {deployment_result.status}")
                logger.info(f"  - Duration: {deployment_result.duration:.2f}s")
                logger.info(f"  - Steps completed: {len(deployment_result.completed_steps)}")
                
                if deployment_result.failed_steps:
                    logger.warning(f"  - Failed steps: {len(deployment_result.failed_steps)}")
                    for step in deployment_result.failed_steps:
                        logger.warning(f"    - {step}: {deployment_result.errors.get(step, 'Unknown error')}")
            else:
                logger.warning("  ❌ Dry-run failed, skipping actual deployment")
            
            # Test 6: Verify deployment
            logger.info("\n6. Verifying deployment...")
            verification = self.production_deployer.verify_deployment()
            
            logger.info("Deployment verification:")
            for check, result in verification.items():
                logger.info(f"  - {check}: {'✅ Passed' if result else '❌ Failed'}")
            
            # Test 7: Get deployment status
            logger.info("\n7. Getting deployment status...")
            status = self.production_deployer.get_deployment_status()
            
            logger.info(f"Current deployment status:")
            logger.info(f"  - Environment: {status['environment']}")
            logger.info(f"  - Version: {status['version']}")
            logger.info(f"  - Status: {status['status']}")
            logger.info(f"  - Components: {len(status['components'])}")
            
            # Test 8: Test rollback capability (if deployment succeeded)
            if deployment_result.status == DeploymentStatus.COMPLETED:
                logger.info("\n8. Testing rollback capability...")
                
                # Create backup before rollback
                backup_id = self.production_deployer.create_backup()
                logger.info(f"  Created backup: {backup_id}")
                
                # Perform rollback
                rollback_result = self.production_deployer.rollback()
                logger.info(f"  Rollback result: {rollback_result.status}")
            
            # Clean up test deployment
            logger.info("\n9. Cleaning up test deployment...")
            if os.path.exists(TEST_DEPLOYMENT_DIR):
                shutil.rmtree(TEST_DEPLOYMENT_DIR)
                logger.info("  ✅ Test deployment cleaned up")
            
            logger.info("\n✅ Production Deployer tests completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Production Deployer test failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Clean up on failure
            if os.path.exists(TEST_DEPLOYMENT_DIR):
                shutil.rmtree(TEST_DEPLOYMENT_DIR)
            
            return False
    
    def test_integrated_validation_flow(self):
        """Test complete integrated validation and deployment flow"""
        logger.info("\n" + "="*80)
        logger.info("TESTING INTEGRATED VALIDATION FLOW")
        logger.info("="*80)
        
        try:
            # Step 1: System validation
            logger.info("\n1. Running full system validation...")
            
            validation_report = self.system_validator.validate_system(
                validation_level=ValidationLevel.PRODUCTION
            )
            
            if validation_report.overall_status != ValidationStatus.PASSED:
                logger.warning("System validation found issues:")
                for issue in validation_report.issues:
                    logger.warning(f"  - {issue}")
                    
            # Step 2: Integration orchestration test
            logger.info("\n2. Testing end-to-end integration...")
            
            # Process sample data through complete pipeline
            sample_data = self.market_data.head(500)
            
            # Submit to orchestrator
            self.integration_orchestrator.set_orchestration_mode(
                OrchestrationMode.ADAPTIVE
            )
            
            task_id = self.integration_orchestrator.submit_data_batch(
                data=sample_data,
                priority=10
            )
            
            # Wait for processing
            time.sleep(5)
            
            # Check results
            pipeline_metrics = self.integration_orchestrator.get_performance_metrics()
            logger.info("Pipeline processing metrics:")
            for stage, metrics in pipeline_metrics.items():
                if metrics.get('items_processed', 0) > 0:
                    logger.info(f"  {stage}: {metrics['items_processed']} items processed")
            
            # Step 3: Production readiness check
            logger.info("\n3. Checking production readiness...")
            
            readiness_checks = {
                'system_validation': validation_report.overall_status == ValidationStatus.PASSED,
                'integration_test': self.integration_orchestrator.get_orchestrator_status()['state'] == 'running',
                'performance_baseline': len(pipeline_metrics) > 0,
                'database_connectivity': self.production_deployer.validate_database_connection()
            }
            
            all_ready = all(readiness_checks.values())
            
            logger.info("Production readiness:")
            for check, result in readiness_checks.items():
                logger.info(f"  - {check}: {'✅ Ready' if result else '❌ Not ready'}")
            
            logger.info(f"\nOverall readiness: {'✅ READY FOR PRODUCTION' if all_ready else '❌ NOT READY'}")
            
            # Step 4: Generate deployment report
            logger.info("\n4. Generating deployment readiness report...")
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'validation_summary': {
                    'total_tests': validation_report.total_tests,
                    'passed_tests': validation_report.passed_tests,
                    'failed_tests': validation_report.failed_tests,
                    'overall_status': validation_report.overall_status.value
                },
                'integration_status': self.integration_orchestrator.get_orchestrator_status(),
                'readiness_checks': readiness_checks,
                'recommendations': validation_report.recommendations if hasattr(validation_report, 'recommendations') else []
            }
            
            # Save report
            report_path = os.path.join(current_dir, 'phase5_test_report.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"✅ Deployment readiness report saved to: {report_path}")
            
            logger.info("\n✅ Integrated validation flow completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Integrated validation flow failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def prepare_validation_data(self):
        """Prepare test data for validation"""
        # Calculate features from market data
        features = []
        
        for _, row in self.market_data.iterrows():
            feature_vector = np.array([
                row['pcr'],
                row['total_call_oi'] / (row['total_put_oi'] + 1),
                row['strike_distance'],
                (row['call_bid_price'] + row['call_ask_price']) / 2,
                (row['put_bid_price'] + row['put_ask_price']) / 2,
                row['call_oi'],
                row['put_oi']
            ])
            features.append(feature_vector)
        
        return {
            'features': np.array(features),
            'timestamps': self.market_data['time_stamp'].values,
            'market_data': self.market_data
        }
    
    def get_component_dependencies(self, component_name):
        """Get component dependencies"""
        dependencies = {
            'base_regime_detector': [],
            'configuration_manager': [],
            'historical_analyzer': ['base_regime_detector'],
            'adaptive_scoring_layer': ['base_regime_detector'],
            'transition_matrix_analyzer': ['base_regime_detector', 'historical_analyzer'],
            'dynamic_boundary_optimizer': ['adaptive_scoring_layer'],
            'intelligent_transition_manager': ['transition_matrix_analyzer', 'adaptive_scoring_layer'],
            'regime_stability_monitor': ['intelligent_transition_manager'],
            'adaptive_noise_filter': ['regime_stability_monitor'],
            'performance_feedback_system': ['adaptive_scoring_layer', 'intelligent_transition_manager'],
            'continuous_learning_engine': ['performance_feedback_system'],
            'regime_optimization_scheduler': ['performance_feedback_system', 'continuous_learning_engine']
        }
        return dependencies.get(component_name, [])
    
    def run_all_tests(self):
        """Run all Phase 5 tests"""
        logger.info("\n" + "="*80)
        logger.info("PHASE 5 COMPREHENSIVE TEST SUITE")
        logger.info("Testing with REAL HeavyDB Market Data")
        logger.info("="*80)
        
        # Connect to HeavyDB
        if not self.connect_to_heavydb():
            logger.error("Cannot proceed without HeavyDB connection")
            return False
        
        # Load market data
        if not self.load_market_data():
            logger.error("Cannot proceed without market data")
            return False
        
        # Initialize all components
        if not self.initialize_all_components():
            logger.error("Cannot proceed without components")
            return False
        
        # Run individual module tests
        test_results = {
            'system_validator': self.test_system_validator(),
            'integration_orchestrator': self.test_integration_orchestrator(),
            'production_deployer': self.test_production_deployer(),
            'integrated_flow': self.test_integrated_validation_flow()
        }
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("PHASE 5 TEST SUMMARY")
        logger.info("="*80)
        
        for test_name, result in test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
        
        all_passed = all(test_results.values())
        
        logger.info("\n" + "="*80)
        if all_passed:
            logger.info("🎉 ALL PHASE 5 TESTS PASSED! 🎉")
            logger.info("System is validated and ready for production deployment")
        else:
            logger.info("❌ Some tests failed. Please review and fix issues.")
        logger.info("="*80)
        
        # Close HeavyDB connection
        if self.conn:
            self.conn.close()
        
        return all_passed


def main():
    """Main test runner"""
    tester = TestPhase5WithHeavyDB()
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()