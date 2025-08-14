#!/usr/bin/env python3
"""
Story 2.5 Deployment Validation Script
Validates the complete implementation of IAM, Artifact Registry, Budgets/Monitoring
"""

import sys
import logging
import yaml
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime

from google.cloud import iam
from google.cloud import artifactregistry_v1
from google.cloud import monitoring_v3
from google.cloud import logging_v2
from google.api_core import exceptions as gcp_exceptions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Story25Validator:
    """Validates Story 2.5 implementation completeness."""
    
    def __init__(self, config_dir: Path):
        """Initialize validator with configuration directory."""
        self.config_dir = config_dir
        self.project_id = None
        self.region = None
        self.validation_results = []
        
        # Load configurations
        self._load_configurations()
        
        # Initialize clients
        self.iam_client = iam.IAMClient()
        self.ar_client = artifactregistry_v1.ArtifactRegistryClient()
        self.monitoring_client = monitoring_v3.MetricServiceClient()
        self.logging_client = logging_v2.Client()

    def _load_configurations(self):
        """Load all configuration files."""
        try:
            # Load IAM config
            iam_config_path = self.config_dir / "iam_config.yaml"
            with open(iam_config_path, 'r') as f:
                self.iam_config = yaml.safe_load(f)
            
            self.project_id = self.iam_config['project_id']
            self.region = self.iam_config['region']
            
            # Load other configs
            config_files = [
                'artifact_registry_config.yaml',
                'budget_config.yaml',
                'security_config.yaml',
                'monitoring_config.yaml'
            ]
            
            self.configs = {}
            for config_file in config_files:
                config_path = self.config_dir / config_file
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config_name = config_file.replace('_config.yaml', '')
                        self.configs[config_name] = yaml.safe_load(f)
                        
            logger.info(f"Loaded configurations for project: {self.project_id}")
            
        except Exception as e:
            logger.error(f"Failed to load configurations: {e}")
            raise

    def validate_iam_configuration(self) -> Tuple[bool, List[str]]:
        """Validate IAM service accounts and role bindings."""
        logger.info("Validating IAM configuration...")
        
        issues = []
        success = True
        
        try:
            # Validate service accounts
            service_accounts = self.iam_config['service_accounts']
            
            for sa_name, sa_config in service_accounts.items():
                sa_email = f"{sa_config['name']}@{self.project_id}.iam.gserviceaccount.com"
                
                try:
                    # Check service account exists
                    sa_resource = f"projects/{self.project_id}/serviceAccounts/{sa_email}"
                    sa = self.iam_client.get_service_account(name=sa_resource)
                    
                    logger.info(f"✅ Service account exists: {sa_email}")
                    
                    # Validate display name
                    if sa.display_name != sa_config['display_name']:
                        issues.append(f"Display name mismatch for {sa_email}")
                        success = False
                        
                except gcp_exceptions.NotFound:
                    issues.append(f"Service account not found: {sa_email}")
                    success = False
                    
            # Validate IAM policy
            project_resource = f"projects/{self.project_id}"
            policy = self.iam_client.get_iam_policy(resource=project_resource)
            
            # Extract member roles
            member_roles = {}
            for binding in policy.bindings:
                for member in binding.members:
                    if member not in member_roles:
                        member_roles[member] = []
                    member_roles[member].append(binding.role)
            
            # Check role assignments
            for sa_name, sa_config in service_accounts.items():
                sa_email = f"{sa_config['name']}@{self.project_id}.iam.gserviceaccount.com"
                member_key = f"serviceAccount:{sa_email}"
                
                if member_key in member_roles:
                    actual_roles = set(member_roles[member_key])
                    expected_roles = set(sa_config['roles'])
                    
                    missing_roles = expected_roles - actual_roles
                    if missing_roles:
                        issues.append(f"Missing roles for {sa_email}: {missing_roles}")
                        success = False
                    else:
                        logger.info(f"✅ All roles assigned for: {sa_email}")
                        
                else:
                    issues.append(f"No roles found for service account: {sa_email}")
                    success = False
                    
        except Exception as e:
            issues.append(f"IAM validation error: {e}")
            success = False
            
        return success, issues

    def validate_artifact_registry(self) -> Tuple[bool, List[str]]:
        """Validate Artifact Registry setup."""
        logger.info("Validating Artifact Registry...")
        
        issues = []
        success = True
        
        try:
            # Check repository exists
            repository_name = f"projects/{self.project_id}/locations/{self.region}/repositories/mr-ml"
            repository = self.ar_client.get_repository(name=repository_name)
            
            logger.info(f"✅ Repository exists: {repository.name}")
            
            # Validate repository configuration
            if repository.format_ != artifactregistry_v1.Repository.Format.DOCKER:
                issues.append("Repository format is not DOCKER")
                success = False
                
            # Check labels
            expected_labels = ['environment', 'project', 'purpose', 'team']
            for label in expected_labels:
                if label not in repository.labels:
                    issues.append(f"Missing label: {label}")
                    success = False
                    
            # Test Docker authentication
            try:
                auth_result = subprocess.run([
                    "gcloud", "auth", "configure-docker", 
                    f"{self.region}-docker.pkg.dev",
                    "--quiet"
                ], capture_output=True, text=True, timeout=30)
                
                if auth_result.returncode == 0:
                    logger.info("✅ Docker authentication configured")
                else:
                    issues.append(f"Docker auth failed: {auth_result.stderr}")
                    success = False
                    
            except Exception as e:
                issues.append(f"Docker auth test error: {e}")
                success = False
                
        except gcp_exceptions.NotFound:
            issues.append("Artifact Registry repository not found")
            success = False
        except Exception as e:
            issues.append(f"Artifact Registry validation error: {e}")
            success = False
            
        return success, issues

    def validate_security_monitoring(self) -> Tuple[bool, List[str]]:
        """Validate security monitoring setup."""
        logger.info("Validating security monitoring...")
        
        issues = []
        success = True
        
        try:
            # Check audit logging configuration
            parent = f"projects/{self.project_id}"
            sinks = self.logging_client.list_sinks(parent=parent)
            
            security_sink_found = False
            for sink in sinks:
                if "security-audit-logs" in sink.name:
                    security_sink_found = True
                    logger.info(f"✅ Security audit log sink found: {sink.name}")
                    
                    # Validate sink filter
                    required_services = [
                        "aiplatform.googleapis.com",
                        "artifactregistry.googleapis.com",
                        "iam.googleapis.com"
                    ]
                    
                    for service in required_services:
                        if service not in sink.filter:
                            issues.append(f"Service missing from audit filter: {service}")
                            success = False
                            
            if not security_sink_found:
                issues.append("Security audit log sink not found")
                success = False
                
            # Check alert policies
            project_name = f"projects/{self.project_id}"
            policies = self.monitoring_client.list_alert_policies(name=project_name)
            
            expected_alerts = [
                "Privileged Role Assignment",
                "Service Account Key Creation",
                "Vertex AI Operation Failures"
            ]
            
            found_alerts = []
            for policy in policies:
                if any(expected in policy.display_name for expected in expected_alerts):
                    found_alerts.append(policy.display_name)
                    logger.info(f"✅ Alert policy found: {policy.display_name}")
            
            if len(found_alerts) < len(expected_alerts):
                missing = len(expected_alerts) - len(found_alerts)
                issues.append(f"Missing {missing} expected alert policies")
                success = False
                
        except Exception as e:
            issues.append(f"Security monitoring validation error: {e}")
            success = False
            
        return success, issues

    def validate_monitoring_infrastructure(self) -> Tuple[bool, List[str]]:
        """Validate monitoring and alerting infrastructure."""
        logger.info("Validating monitoring infrastructure...")
        
        issues = []
        success = True
        
        try:
            # Check notification channels
            project_name = f"projects/{self.project_id}"
            channels = self.monitoring_client.list_notification_channels(name=project_name)
            
            email_channels = [ch for ch in channels if ch.type == "email"]
            if not email_channels:
                issues.append("No email notification channels found")
                success = False
            else:
                logger.info(f"✅ Found {len(email_channels)} email notification channels")
                
            # Check dashboards
            dashboards = self.monitoring_client.list_dashboards(parent=project_name)
            
            expected_dashboards = ["ML Operations", "Security"]
            dashboard_names = [d.display_name for d in dashboards]
            
            for expected in expected_dashboards:
                if not any(expected in name for name in dashboard_names):
                    issues.append(f"Missing dashboard containing: {expected}")
                    success = False
                else:
                    logger.info(f"✅ Dashboard found containing: {expected}")
                    
            # Check uptime checks
            uptime_checks = self.monitoring_client.list_uptime_check_configs(parent=project_name)
            uptime_count = len(list(uptime_checks))
            
            if uptime_count == 0:
                issues.append("No uptime checks configured")
                success = False
            else:
                logger.info(f"✅ Found {uptime_count} uptime checks")
                
        except Exception as e:
            issues.append(f"Monitoring infrastructure validation error: {e}")
            success = False
            
        return success, issues

    def validate_cost_monitoring(self) -> Tuple[bool, List[str]]:
        """Validate cost monitoring and budget setup."""
        logger.info("Validating cost monitoring...")
        
        issues = []
        success = True
        
        try:
            # Check if budget configuration exists
            if 'budget' in self.configs:
                budget_config = self.configs['budget']
                
                # Validate budget structure
                required_sections = ['budgets', 'cost_allocation', 'alerting']
                for section in required_sections:
                    if section not in budget_config:
                        issues.append(f"Missing budget config section: {section}")
                        success = False
                        
                # Check budget types
                expected_budgets = ['vertex_ai', 'storage', 'compute']
                budgets = budget_config.get('budgets', {})
                
                for budget_type in expected_budgets:
                    if budget_type not in budgets:
                        issues.append(f"Missing budget configuration: {budget_type}")
                        success = False
                    else:
                        logger.info(f"✅ Budget configuration found: {budget_type}")
                        
                logger.info("✅ Cost monitoring configuration validated")
                
            else:
                issues.append("Budget configuration file not found")
                success = False
                
        except Exception as e:
            issues.append(f"Cost monitoring validation error: {e}")
            success = False
            
        return success, issues

    def validate_terraform_configuration(self) -> Tuple[bool, List[str]]:
        """Validate Terraform configuration files."""
        logger.info("Validating Terraform configuration...")
        
        issues = []
        success = True
        
        terraform_dir = Path(__file__).parent.parent / "terraform"
        
        required_files = [
            "iam_service_accounts.tf",
            "artifact_registry.tf",
            "audit_logging.tf",
            "monitoring.tf"
        ]
        
        for tf_file in required_files:
            tf_path = terraform_dir / tf_file
            if tf_path.exists():
                logger.info(f"✅ Terraform file exists: {tf_file}")
                
                # Basic syntax validation
                try:
                    with open(tf_path, 'r') as f:
                        content = f.read()
                        
                    # Check for required resources
                    if "resource \"google_" not in content:
                        issues.append(f"No Google resources found in: {tf_file}")
                        success = False
                        
                except Exception as e:
                    issues.append(f"Error reading Terraform file {tf_file}: {e}")
                    success = False
                    
            else:
                issues.append(f"Missing Terraform file: {tf_file}")
                success = False
                
        return success, issues

    def test_end_to_end_operations(self) -> Tuple[bool, List[str]]:
        """Test end-to-end operations."""
        logger.info("Testing end-to-end operations...")
        
        issues = []
        success = True
        
        try:
            # Test Vertex AI API access
            result = subprocess.run([
                "gcloud", "ai", "locations", "list",
                "--project", self.project_id,
                "--format", "json"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info("✅ Vertex AI API access successful")
            else:
                issues.append(f"Vertex AI API access failed: {result.stderr}")
                success = False
                
            # Test Artifact Registry access
            result = subprocess.run([
                "gcloud", "artifacts", "repositories", "list",
                "--project", self.project_id,
                "--location", self.region,
                "--format", "json"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info("✅ Artifact Registry access successful")
            else:
                issues.append(f"Artifact Registry access failed: {result.stderr}")
                success = False
                
        except Exception as e:
            issues.append(f"End-to-end test error: {e}")
            success = False
            
        return success, issues

    def generate_validation_report(self) -> str:
        """Generate comprehensive validation report."""
        report = []
        report.append("# Story 2.5 Validation Report")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append(f"Project: {self.project_id}")
        report.append(f"Region: {self.region}")
        report.append("")
        
        # Overall summary
        total_tests = len(self.validation_results)
        passed_tests = sum(1 for result in self.validation_results if result['success'])
        
        report.append("## Summary")
        report.append(f"- Total validations: {total_tests}")
        report.append(f"- Passed: {passed_tests}")
        report.append(f"- Failed: {total_tests - passed_tests}")
        report.append(f"- Success rate: {(passed_tests/total_tests)*100:.1f}%")
        report.append("")
        
        # Detailed results
        report.append("## Detailed Results")
        for result in self.validation_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            report.append(f"### {result['test_name']} - {status}")
            
            if result['issues']:
                report.append("Issues found:")
                for issue in result['issues']:
                    report.append(f"- {issue}")
            else:
                report.append("No issues found.")
            report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        if passed_tests == total_tests:
            report.append("✅ All validations passed. Story 2.5 implementation is complete.")
        else:
            report.append("❌ Some validations failed. Review the issues above and:")
            report.append("1. Fix any missing configurations")
            report.append("2. Re-run Terraform apply if needed")
            report.append("3. Verify service account permissions")
            report.append("4. Check monitoring setup")
        
        return "\n".join(report)

    def run_all_validations(self) -> bool:
        """Run all validation tests."""
        logger.info("Starting Story 2.5 validation...")
        
        validation_tests = [
            ("IAM Configuration", self.validate_iam_configuration),
            ("Artifact Registry", self.validate_artifact_registry),
            ("Security Monitoring", self.validate_security_monitoring),
            ("Monitoring Infrastructure", self.validate_monitoring_infrastructure),
            ("Cost Monitoring", self.validate_cost_monitoring),
            ("Terraform Configuration", self.validate_terraform_configuration),
            ("End-to-End Operations", self.test_end_to_end_operations)
        ]
        
        all_passed = True
        
        for test_name, test_func in validation_tests:
            logger.info(f"Running validation: {test_name}")
            
            try:
                success, issues = test_func()
                
                self.validation_results.append({
                    'test_name': test_name,
                    'success': success,
                    'issues': issues
                })
                
                if not success:
                    all_passed = False
                    logger.error(f"❌ {test_name} validation failed")
                    for issue in issues:
                        logger.error(f"  - {issue}")
                else:
                    logger.info(f"✅ {test_name} validation passed")
                    
            except Exception as e:
                logger.error(f"❌ {test_name} validation error: {e}")
                self.validation_results.append({
                    'test_name': test_name,
                    'success': False,
                    'issues': [str(e)]
                })
                all_passed = False
        
        # Generate and save report
        report = self.generate_validation_report()
        
        report_path = Path(__file__).parent.parent / "docs" / "story_2_5_validation_report.md"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Validation report saved: {report_path}")
        
        if all_passed:
            logger.info("🎉 All validations passed! Story 2.5 implementation is complete.")
        else:
            logger.error("❌ Some validations failed. Check the report for details.")
        
        return all_passed

def main():
    """Main entry point."""
    config_dir = Path(__file__).parent.parent / "configs"
    
    if not config_dir.exists():
        print(f"Error: Config directory not found: {config_dir}")
        sys.exit(1)
    
    try:
        validator = Story25Validator(config_dir)
        success = validator.run_all_validations()
        sys.exit(0 if success else 1)
        
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()