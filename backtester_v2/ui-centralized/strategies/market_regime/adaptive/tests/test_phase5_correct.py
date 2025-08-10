"""
Phase 5 Test Suite with Correct Method Signatures

Tests Phase 5 modules using their actual API
"""

import sys
import os
import json
import tempfile
import shutil
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import logging

# Add parent directories to path
current_dir = Path(__file__).parent
adaptive_dir = current_dir.parent
sys.path.insert(0, str(adaptive_dir))

# Import Phase 5 modules
from validation.system_validator import (
    SystemValidator, ValidationLevel, ValidationStatus, ComponentType,
    ValidationTest, ValidationResult, ValidationReport
)
from validation.integration_orchestrator import (
    IntegrationOrchestrator, ComponentState, OrchestrationMode,
    OrchestrationConfig
)
from validation.production_deployer import (
    ProductionDeployer, DeploymentConfig, EnvironmentType,
    DeploymentResult, ServiceHealth
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockComponent:
    """Mock component for testing"""
    def __init__(self, name: str):
        self.name = name
        self.initialized = False
        self.error_count = 0
        
    def initialize(self) -> bool:
        self.initialized = True
        return True
        
    def process(self, data: Any) -> Any:
        return {'processed': True, 'component': self.name}
        
    def get_status(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'initialized': self.initialized,
            'error_count': self.error_count
        }
    
    def calculate_regime(self, features: np.ndarray) -> int:
        """Mock regime calculation"""
        return np.random.randint(0, 5)
    
    def get_configuration(self) -> Dict[str, Any]:
        """Mock configuration"""
        return {'name': self.name, 'version': '1.0'}


def test_system_validator():
    """Test System Validator with correct API"""
    logger.info("\n" + "="*60)
    logger.info("TESTING SYSTEM VALIDATOR")
    logger.info("="*60)
    
    try:
        # Initialize validator
        validator = SystemValidator()
        
        # Create mock modules dictionary
        modules = {
            ComponentType.BASE_REGIME_DETECTOR: MockComponent('base_regime_detector'),
            ComponentType.CONFIGURATION_MANAGER: MockComponent('configuration_manager'),
            ComponentType.HISTORICAL_ANALYZER: MockComponent('historical_analyzer'),
            ComponentType.ADAPTIVE_SCORING_LAYER: MockComponent('adaptive_scoring_layer'),
            ComponentType.TRANSITION_MATRIX_ANALYZER: MockComponent('transition_matrix_analyzer'),
            ComponentType.DYNAMIC_BOUNDARY_OPTIMIZER: MockComponent('dynamic_boundary_optimizer')
        }
        
        # Set modules
        logger.info("\n1. Setting modules...")
        validator.set_modules(modules)
        logger.info("  ✅ Modules set")
        
        # Create mock market data
        logger.info("\n2. Setting market data...")
        dates = pd.date_range('2024-01-01', periods=100, freq='5min')
        market_data = pd.DataFrame({
            'time_stamp': dates,
            'spot_price': np.random.uniform(19000, 20000, 100),
            'pcr': np.random.uniform(0.5, 1.5, 100),
            'total_call_oi': np.random.uniform(1000000, 2000000, 100),
            'total_put_oi': np.random.uniform(1000000, 2000000, 100)
        })
        
        validator.set_market_data(market_data)
        logger.info("  ✅ Market data set")
        
        # Run validation
        logger.info("\n3. Running system validation...")
        report = validator.validate_system()
        
        logger.info(f"\nValidation Report:")
        logger.info(f"  - Level: {report.validation_level}")
        logger.info(f"  - Total tests: {report.total_tests}")
        logger.info(f"  - Passed: {report.passed_tests}")
        logger.info(f"  - Failed: {report.failed_tests}")
        logger.info(f"  - Warnings: {report.warnings}")
        logger.info(f"  - Overall status: {report.overall_status}")
        
        # Export report
        logger.info("\n4. Exporting validation report...")
        report_path = os.path.join(tempfile.gettempdir(), 'validation_report.json')
        validator.export_validation_report(report, report_path)
        logger.info(f"  ✅ Report exported to: {report_path}")
        
        # Clean up
        if os.path.exists(report_path):
            os.remove(report_path)
        
        logger.info("\n✅ System Validator test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ System Validator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_orchestrator():
    """Test Integration Orchestrator with correct API"""
    logger.info("\n" + "="*60)
    logger.info("TESTING INTEGRATION ORCHESTRATOR")
    logger.info("="*60)
    
    try:
        # Create orchestrator configuration
        config = OrchestrationConfig(
            mode=OrchestrationMode.ADAPTIVE,
            max_workers=2,
            task_timeout=30.0,
            retry_attempts=2
        )
        
        # Initialize orchestrator
        orchestrator = IntegrationOrchestrator(config)
        
        # Register mock components
        components = {
            'data_ingestion': MockComponent('data_ingestion'),
            'feature_engineering': MockComponent('feature_engineering'),
            'regime_scoring': MockComponent('regime_scoring'),
            'regime_decision': MockComponent('regime_decision')
        }
        
        logger.info("\n1. Registering components...")
        for comp_id, comp in components.items():
            orchestrator.register_component(
                component_id=comp_id,
                component_type=comp_id,
                module_reference=comp,
                dependencies=[],
                initialization_params={}
            )
            logger.info(f"  ✅ Registered: {comp_id}")
        
        # Initialize system
        logger.info("\n2. Initializing system...")
        success = orchestrator.initialize_system()
        logger.info(f"  System initialization: {'✅ Success' if success else '❌ Failed'}")
        
        # Start orchestration
        logger.info("\n3. Starting orchestration...")
        orchestrator.start_orchestration()
        logger.info("  ✅ Orchestration started")
        
        # Give time for startup
        time.sleep(1)
        
        # Process market data
        logger.info("\n4. Processing market data...")
        test_data = {
            'timestamp': datetime.now(),
            'spot_price': 19500,
            'pcr': 0.85,
            'features': np.array([0.85, 1.2, 0.3, -0.5])
        }
        
        result = orchestrator.process_market_data(test_data)
        logger.info(f"  Processing result: {result}")
        
        # Get system status
        logger.info("\n5. Getting system status...")
        status = orchestrator.get_system_status()
        logger.info(f"  Status: {status['status']}")
        logger.info(f"  Active components: {len(status['components'])}")
        logger.info(f"  Tasks processed: {status['performance']['tasks_processed']}")
        
        # Export report
        logger.info("\n6. Exporting orchestration report...")
        report_path = os.path.join(tempfile.gettempdir(), 'orchestration_report.json')
        orchestrator.export_orchestration_report(report_path)
        logger.info(f"  ✅ Report exported to: {report_path}")
        
        # Stop orchestration
        logger.info("\n7. Stopping orchestration...")
        orchestrator.stop_orchestration()
        logger.info("  ✅ Orchestration stopped")
        
        # Clean up
        if os.path.exists(report_path):
            os.remove(report_path)
        
        logger.info("\n✅ Integration Orchestrator test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration Orchestrator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_production_deployer():
    """Test Production Deployer with correct API"""
    logger.info("\n" + "="*60)
    logger.info("TESTING PRODUCTION DEPLOYER")
    logger.info("="*60)
    
    # Test deployment directory
    test_dir = "/tmp/test_deployment"
    
    try:
        # Create test configuration
        config = DeploymentConfig(
            environment=EnvironmentType.TESTING,
            deployment_path=test_dir,
            config_path=os.path.join(test_dir, "config"),
            log_path=os.path.join(test_dir, "logs"),
            data_path=os.path.join(test_dir, "data"),
            service_name="test_regime_system",
            service_port=8080
        )
        
        # Initialize deployer
        deployer = ProductionDeployer(config)
        
        # Run deployment
        logger.info("\n1. Running test deployment...")
        result = deployer.deploy()
        
        logger.info(f"\nDeployment Result:")
        logger.info(f"  - Status: {result.status}")
        logger.info(f"  - Duration: {result.duration:.2f}s")
        logger.info(f"  - Steps completed: {len(result.completed_steps)}")
        
        if result.failed_steps:
            logger.warning(f"  - Failed steps: {len(result.failed_steps)}")
            for step in result.failed_steps[:3]:  # Show first 3
                logger.warning(f"    - {step}")
        
        # Check service health
        logger.info("\n2. Checking service health...")
        health = deployer.check_service_health()
        logger.info(f"  Service health: {health.status}")
        logger.info(f"  - CPU usage: {health.metrics.get('cpu_usage', 0):.1f}%")
        logger.info(f"  - Memory usage: {health.metrics.get('memory_usage', 0):.1f}%")
        
        # Generate deployment report
        logger.info("\n3. Generating deployment report...")
        report_path = os.path.join(tempfile.gettempdir(), 'deployment_report.json')
        deployer.generate_deployment_report(result, report_path)
        logger.info(f"  ✅ Report generated at: {report_path}")
        
        # Clean up
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        if os.path.exists(report_path):
            os.remove(report_path)
        
        logger.info("\n✅ Production Deployer test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Production Deployer test failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up on failure
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        
        return False


def main():
    """Run all tests"""
    logger.info("\n" + "="*60)
    logger.info("PHASE 5 TEST SUITE")
    logger.info("Testing with Correct API")
    logger.info("="*60)
    
    # Run tests
    results = {
        'System Validator': test_system_validator(),
        'Integration Orchestrator': test_integration_orchestrator(),
        'Production Deployer': test_production_deployer()
    }
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    logger.info("\n" + "="*60)
    if all_passed:
        logger.info("✅ ALL TESTS PASSED")
        logger.info("Phase 5 modules are working correctly!")
    else:
        logger.info("❌ SOME TESTS FAILED")
        logger.info("Please check the error messages above")
    logger.info("="*60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())