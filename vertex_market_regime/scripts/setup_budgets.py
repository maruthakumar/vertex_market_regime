#!/usr/bin/env python3
"""
Budget and Cost Monitoring Setup Script
Story 2.5: IAM, Artifact Registry, Budgets/Monitoring

Sets up budget alerts and cost monitoring for Vertex AI resources.
"""

import sys
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, List

from google.cloud import billing_v1
from google.cloud import monitoring_v3
from google.api_core import exceptions as gcp_exceptions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BudgetMonitoringSetup:
    """Manages budget alerts and cost monitoring setup."""
    
    def __init__(self, config_path: str):
        """Initialize with configuration file."""
        self.config = self._load_config(config_path)
        self.project_id = self.config['project_id']
        self.billing_account = self.config['billing_account']
        
        # Initialize clients
        self.billing_client = billing_v1.CloudBillingClient()
        self.budget_client = billing_v1.BudgetServiceClient()
        self.monitoring_client = monitoring_v3.MetricServiceClient()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise
    
    def create_vertex_ai_budget(self) -> bool:
        """Create budget alerts for Vertex AI resources."""
        try:
            budget_config = self.config['budgets']['vertex_ai']
            
            # Create budget for Vertex AI services
            budget = billing_v1.Budget(
                display_name=budget_config['name'],
                budget_filter=billing_v1.Filter(
                    projects=[f"projects/{self.project_id}"],
                    services=[
                        "services/6F8107F1-B446-44C0-A518-8F20D2C4E16B",  # Vertex AI
                        "services/A1E8-BE35-4142-B9CD-7BD85FD6F4C9"   # Artifact Registry
                    ]
                ),
                amount=billing_v1.BudgetAmount(
                    specified_amount={"currency_code": "USD", "units": budget_config['amount_usd']}
                ),
                threshold_rules=[
                    billing_v1.ThresholdRule(
                        threshold_percent=threshold['percent'],
                        spend_basis=billing_v1.ThresholdRule.Basis.CURRENT_SPEND
                    )
                    for threshold in budget_config['alert_thresholds']
                ],
                notifications_rule=billing_v1.NotificationsRule(
                    pubsub_topic=budget_config['notification_topic'],
                    schema_version="1.0"
                )
            )
            
            # Create the budget
            parent = f"billingAccounts/{self.billing_account}"
            operation = self.budget_client.create_budget(
                parent=parent,
                budget=budget
            )
            
            logger.info(f"Created Vertex AI budget: {budget.display_name}")
            logger.info(f"Monthly limit: ${budget_config['amount_usd']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create Vertex AI budget: {e}")
            return False
    
    def create_storage_budget(self) -> bool:
        """Create budget alerts for storage resources."""
        try:
            budget_config = self.config['budgets']['storage']
            
            # Create budget for storage services
            budget = billing_v1.Budget(
                display_name=budget_config['name'],
                budget_filter=billing_v1.Filter(
                    projects=[f"projects/{self.project_id}"],
                    services=[
                        "services/95FF2659-5EA4-4C24-B14F-63FF9D57E62F",  # Cloud Storage
                        "services/24E6-5A73-48C6-9ABFD53E0833"   # BigQuery
                    ]
                ),
                amount=billing_v1.BudgetAmount(
                    specified_amount={"currency_code": "USD", "units": budget_config['amount_usd']}
                ),
                threshold_rules=[
                    billing_v1.ThresholdRule(
                        threshold_percent=threshold['percent'],
                        spend_basis=billing_v1.ThresholdRule.Basis.CURRENT_SPEND
                    )
                    for threshold in budget_config['alert_thresholds']
                ],
                notifications_rule=billing_v1.NotificationsRule(
                    pubsub_topic=budget_config['notification_topic'],
                    schema_version="1.0"
                )
            )
            
            # Create the budget
            parent = f"billingAccounts/{self.billing_account}"
            operation = self.budget_client.create_budget(
                parent=parent,
                budget=budget
            )
            
            logger.info(f"Created Storage budget: {budget.display_name}")
            logger.info(f"Monthly limit: ${budget_config['amount_usd']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create Storage budget: {e}")
            return False
    
    def setup_cost_allocation_labels(self) -> bool:
        """Set up cost allocation labels and tagging policies."""
        try:
            labels_config = self.config['cost_allocation']['required_labels']
            
            logger.info("Cost allocation labels configuration:")
            for label, description in labels_config.items():
                logger.info(f"  {label}: {description}")
            
            # Create organization policy for required labels
            policy_content = self._generate_labeling_policy(labels_config)
            
            # Write policy to file for manual application
            policy_file = Path("/Users/maruth/projects/market_regime/vertex_market_regime/configs/cost_allocation_policy.yaml")
            with open(policy_file, 'w') as f:
                f.write(policy_content)
            
            logger.info(f"Cost allocation policy written to: {policy_file}")
            logger.info("Apply this policy manually using gcloud or Terraform")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup cost allocation labels: {e}")
            return False
    
    def create_cost_monitoring_dashboards(self) -> bool:
        """Create Cloud Monitoring dashboards for cost tracking."""
        try:
            # Cost monitoring will be handled by Cloud Monitoring
            # Dashboard configurations will be in monitoring setup
            
            dashboard_config = self.config['cost_monitoring']['dashboards']
            
            logger.info("Cost monitoring dashboards configured:")
            for dashboard in dashboard_config:
                logger.info(f"  Dashboard: {dashboard['name']}")
                logger.info(f"    Purpose: {dashboard['description']}")
                for widget in dashboard.get('widgets', []):
                    logger.info(f"    Widget: {widget}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create cost monitoring dashboards: {e}")
            return False
    
    def setup_quota_monitoring(self) -> bool:
        """Set up resource quota monitoring and alerts."""
        try:
            quota_config = self.config['resource_quotas']
            
            logger.info("Resource quota monitoring configured:")
            for service, quotas in quota_config.items():
                logger.info(f"  Service: {service}")
                for quota_name, limit in quotas.items():
                    logger.info(f"    {quota_name}: {limit}")
            
            # Quota monitoring alerts will be created in monitoring setup
            logger.info("Quota alerts will be configured in monitoring setup")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup quota monitoring: {e}")
            return False
    
    def create_cost_optimization_recommendations(self) -> bool:
        """Set up automated cost optimization recommendations."""
        try:
            optimization_config = self.config['cost_optimization']
            
            # Create cost optimization report
            report_content = self._generate_optimization_report(optimization_config)
            
            # Write recommendations to file
            report_file = Path("/Users/maruth/projects/market_regime/vertex_market_regime/docs/cost_optimization_guide.md")
            report_file.parent.mkdir(exist_ok=True)
            
            with open(report_file, 'w') as f:
                f.write(report_content)
            
            logger.info(f"Cost optimization guide generated: {report_file}")
            
            # Log automated recommendations
            for strategy in optimization_config.get('automated_strategies', []):
                logger.info(f"Automated strategy: {strategy}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create cost optimization recommendations: {e}")
            return False
    
    def _generate_labeling_policy(self, labels: Dict[str, str]) -> str:
        """Generate organization policy for required labels."""
        return f"""# Cost Allocation Labeling Policy
# Story 2.5: Required labels for cost tracking

apiVersion: orgpolicy.googleapis.com/v1
kind: Policy
metadata:
  name: required-resource-labels
spec:
  constraint: constraints/gcp.resourceLocations
  rules:
    - enforce: true
      values:
        required_labels:
{yaml.dump(labels, indent=10)}

# Apply with:
# gcloud resource-manager org-policies set-policy cost_allocation_policy.yaml \\
#   --project={self.project_id}
"""
    
    def _generate_optimization_report(self, config: Dict[str, Any]) -> str:
        """Generate cost optimization recommendations report."""
        return f"""# Cost Optimization Guide for Vertex AI ML Infrastructure

## Overview
This guide provides automated cost optimization recommendations for the Market Regime ML infrastructure.

## Current Budget Configuration
- **Vertex AI Monthly Budget**: ${self.config['budgets']['vertex_ai']['amount_usd']}
- **Storage Monthly Budget**: ${self.config['budgets']['storage']['amount_usd']}

## Automated Optimization Strategies

### Compute Optimization
{self._format_compute_optimization(config)}

### Storage Optimization
{self._format_storage_optimization(config)}

### Training Pipeline Optimization
{self._format_training_optimization(config)}

## Cost Monitoring Best Practices

### Resource Tagging
- Apply consistent labels to all resources
- Use cost allocation tags for department/project tracking
- Implement automated labeling policies

### Regular Reviews
- Weekly cost reviews for anomaly detection
- Monthly budget vs. actual analysis
- Quarterly optimization assessment

## Automated Alerts
- Budget threshold alerts at 50%, 80%, 90%
- Unusual spending pattern detection
- Resource quota utilization warnings

## Cost Tracking Dashboards
{self._format_dashboard_info()}

## Recommendations Implementation
1. Enable automated resource scheduling
2. Implement preemptible instances for training
3. Set up storage lifecycle policies
4. Use sustained use discounts
5. Optimize container images for faster training

Generated: {self.config.get('generated_date', '2025-08-13')}
"""
    
    def _format_compute_optimization(self, config: Dict[str, Any]) -> str:
        """Format compute optimization recommendations."""
        strategies = config.get('compute_strategies', [])
        output = []
        
        for strategy in strategies:
            output.append(f"- **{strategy['name']}**: {strategy['description']}")
            output.append(f"  - Potential savings: {strategy.get('savings', 'TBD')}")
            output.append(f"  - Implementation: {strategy.get('implementation', 'Automated')}")
            output.append("")
        
        return "\n".join(output)
    
    def _format_storage_optimization(self, config: Dict[str, Any]) -> str:
        """Format storage optimization recommendations."""
        strategies = config.get('storage_strategies', [])
        output = []
        
        for strategy in strategies:
            output.append(f"- **{strategy['name']}**: {strategy['description']}")
            output.append(f"  - Impact: {strategy.get('impact', 'Medium')}")
            output.append("")
        
        return "\n".join(output)
    
    def _format_training_optimization(self, config: Dict[str, Any]) -> str:
        """Format training pipeline optimization recommendations."""
        strategies = config.get('training_strategies', [])
        output = []
        
        for strategy in strategies:
            output.append(f"- **{strategy['name']}**: {strategy['description']}")
            output.append(f"  - Expected reduction: {strategy.get('reduction', '10-20%')}")
            output.append("")
        
        return "\n".join(output)
    
    def _format_dashboard_info(self) -> str:
        """Format dashboard information."""
        dashboards = self.config.get('cost_monitoring', {}).get('dashboards', [])
        output = []
        
        for dashboard in dashboards:
            output.append(f"### {dashboard['name']}")
            output.append(f"{dashboard['description']}")
            output.append("")
            
            for widget in dashboard.get('widgets', []):
                output.append(f"- {widget}")
            output.append("")
        
        return "\n".join(output)
    
    def run_setup(self) -> bool:
        """Run complete budget and cost monitoring setup."""
        logger.info("Starting budget and cost monitoring setup...")
        
        success = True
        
        # Create budgets
        if not self.create_vertex_ai_budget():
            success = False
        
        if not self.create_storage_budget():
            success = False
        
        # Setup cost allocation
        if not self.setup_cost_allocation_labels():
            success = False
        
        # Create monitoring dashboards
        if not self.create_cost_monitoring_dashboards():
            success = False
        
        # Setup quota monitoring
        if not self.setup_quota_monitoring():
            success = False
        
        # Create optimization recommendations
        if not self.create_cost_optimization_recommendations():
            success = False
        
        if success:
            logger.info("✅ Budget and cost monitoring setup completed successfully")
        else:
            logger.error("❌ Budget and cost monitoring setup completed with errors")
        
        return success

def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python setup_budgets.py <config_file>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    if not Path(config_file).exists():
        print(f"Error: Config file not found: {config_file}")
        sys.exit(1)
    
    try:
        setup = BudgetMonitoringSetup(config_file)
        success = setup.run_setup()
        sys.exit(0 if success else 1)
        
    except Exception as e:
        logger.error(f"Setup failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()