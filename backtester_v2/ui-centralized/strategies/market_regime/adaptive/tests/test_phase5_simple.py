"""
Simple Phase 5 Test Suite

Tests Phase 5 modules in isolation with simulated components
"""

import sys
import os
import json
import tempfile
import shutil
import time
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
    ValidationTest, ValidationResult
)
from validation.integration_orchestrator import (
    IntegrationOrchestrator, ComponentState, OrchestrationMode
)
from validation.production_deployer import (
    ProductionDeployer, DeploymentConfig, EnvironmentType
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


def test_system_validator():
    """Test System Validator"""
    logger.info("\n" + "="*60)
    logger.info("TESTING SYSTEM VALIDATOR")
    logger.info("="*60)
    
    try:
        # Initialize validator
        validator = SystemValidator()
        
        # Register mock components
        components = {
            'component1': MockComponent('component1'),
            'component2': MockComponent('component2'),
            'component3': MockComponent('component3')
        }
        
        for name, comp in components.items():
            validator.register_component(name, comp)
        
        # Test basic validation
        logger.info("\n1. Running basic validation...")
        report = validator.validate_system(ValidationLevel.BASIC)
        
        logger.info(f"Basic validation completed:")
        logger.info(f"  - Total tests: {report.total_tests}")
        logger.info(f"  - Passed: {report.passed_tests}")
        logger.info(f"  - Status: {report.overall_status}")
        
        # Test component validation
        logger.info("\n2. Testing component validation...")
        result = validator.validate_component('component1', {})
        logger.info(f"Component validation: {result.status}")
        
        # Test data quality validation
        logger.info("\n3. Testing data quality validation...")
        test_data = {
            'features': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            'timestamps': ['2024-01-01', '2024-01-02', '2024-01-03']
        }
        
        quality_result = validator.validate_data_quality(test_data)
        logger.info(f"Data quality validation: {quality_result.get('status', 'Unknown')}")
        
        logger.info("\n✅ System Validator test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ System Validator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_orchestrator():
    """Test Integration Orchestrator"""
    logger.info("\n" + "="*60)
    logger.info("TESTING INTEGRATION ORCHESTRATOR")
    logger.info("="*60)
    
    try:
        # Initialize orchestrator
        orchestrator = IntegrationOrchestrator()
        
        # Register mock components
        components = {
            'data_ingestion': MockComponent('data_ingestion'),
            'processing': MockComponent('processing'),
            'output': MockComponent('output')
        }
        
        for comp_id, comp in components.items():
            success = orchestrator.register_component(
                component_id=comp_id,
                component_type=comp_id,
                module_reference=comp,
                dependencies=[]
            )
            logger.info(f"  Registered {comp_id}: {'✅' if success else '❌'}")
        
        # Initialize and start
        logger.info("\n1. Initializing orchestrator...")
        orchestrator.initialize()
        
        logger.info("\n2. Starting orchestrator...")
        orchestrator.start()
        
        # Give time for initialization
        time.sleep(1)
        
        # Submit test data
        logger.info("\n3. Submitting test data...")
        test_data = {'test': 'data', 'value': 42}
        task_id = orchestrator.submit_data_batch(test_data, priority=5)
        logger.info(f"  Submitted task: {task_id}")
        
        # Wait for processing
        time.sleep(1)
        
        # Check status
        logger.info("\n4. Checking orchestrator status...")
        status = orchestrator.get_orchestrator_status()
        logger.info(f"  State: {status['state']}")
        logger.info(f"  Tasks processed: {status['tasks_processed']}")
        
        # Stop orchestrator
        logger.info("\n5. Stopping orchestrator...")
        orchestrator.stop()
        
        logger.info("\n✅ Integration Orchestrator test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration Orchestrator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_production_deployer():
    """Test Production Deployer"""
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
            data_path=os.path.join(test_dir, "data")
        )
        
        # Initialize deployer
        deployer = ProductionDeployer(config)
        
        # Test environment validation
        logger.info("\n1. Validating environment...")
        env_valid = deployer.validate_environment()
        logger.info(f"  Environment: {'✅ Valid' if env_valid else '❌ Invalid'}")
        
        # Test dependency check
        logger.info("\n2. Checking dependencies...")
        deps_valid = deployer.check_dependencies()
        logger.info(f"  Dependencies: {'✅ OK' if deps_valid else '❌ Missing'}")
        
        # Plan deployment
        logger.info("\n3. Planning deployment...")
        plan = deployer.plan_deployment()
        logger.info(f"  Created plan with {len(plan.steps)} steps")
        
        # Test dry run
        logger.info("\n4. Running deployment (dry run)...")
        result = deployer.deploy(dry_run=True)
        logger.info(f"  Dry run status: {result.status}")
        
        # Get status
        logger.info("\n5. Getting deployment status...")
        status = deployer.get_deployment_status()
        logger.info(f"  Status: {status['status']}")
        logger.info(f"  Environment: {status['environment']}")
        
        # Clean up
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        
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
    logger.info("PHASE 5 SIMPLE TEST SUITE")
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
    else:
        logger.info("❌ SOME TESTS FAILED")
    logger.info("="*60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())