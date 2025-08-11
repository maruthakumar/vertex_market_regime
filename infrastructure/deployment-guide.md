# Market Regime Master Framework - Infrastructure Deployment Guide
**Version:** 1.0  
**Date:** 2025-08-10  
**Infrastructure as Code:** Terraform v1.0+

## Overview

This guide provides step-by-step instructions for deploying the complete Google Cloud Platform infrastructure required for the Market Regime Master Framework enhancement. The infrastructure supports the 8-component adaptive learning system with Vertex AI integration.

## Prerequisites

### Required Tools
- **Terraform:** >= 1.0 ([Install Guide](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli))
- **Google Cloud SDK:** Latest version ([Install Guide](https://cloud.google.com/sdk/docs/install))
- **Git:** For version control

### Required Accounts & Access
- **GCP Project:** With billing enabled
- **GCP User Account:** With Project Owner or Editor permissions
- **Billing Account ID:** For cost management and budgets

### Local Environment Setup
```bash
# Verify installations
terraform --version  # Should be >= 1.0
gcloud --version     # Should be latest
git --version        # Any recent version

# Authenticate with Google Cloud
gcloud auth login
gcloud auth application-default login
```

## Pre-Deployment Checklist

### 1. GCP Project Preparation
```bash
# Set your project ID
export PROJECT_ID="your-market-regime-project"

# Set the project as default
gcloud config set project $PROJECT_ID

# Verify billing is enabled
gcloud beta billing projects describe $PROJECT_ID
```

### 2. Required GCP APIs
```bash
# Enable required APIs (automated in Terraform, but good to verify)
gcloud services enable aiplatform.googleapis.com
gcloud services enable bigquery.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable compute.googleapis.com
gcloud services enable monitoring.googleapis.com
gcloud services enable logging.googleapis.com
```

### 3. Cost Management Setup
- **Budget Alerts:** Ensure billing account has budget alerts configured
- **Resource Quotas:** Verify GPU quotas in your selected region
- **Cost Monitoring:** Set up billing exports to BigQuery (optional)

## Deployment Steps

### Step 1: Infrastructure Repository Setup
```bash
# Navigate to your project directory
cd /Users/maruth/projects/market_regime/

# Verify Terraform files are present
ls infrastructure/terraform/
# Should show: main.tf, variables.tf, terraform.tfvars.example

# Navigate to Terraform directory
cd infrastructure/terraform/
```

### Step 2: Configuration Setup
```bash
# Create your configuration from example
cp terraform.tfvars.example terraform.tfvars

# Edit configuration with your values
nano terraform.tfvars
```

**Critical Configuration Values:**
```hcl
# REQUIRED - Update these values
project_id         = "your-actual-project-id"
billing_account_id = "123456-789012-345678"  # Your billing account

# RECOMMENDED - Customize for your environment
environment = "dev"  # or "staging", "prod"
cost_budget_amount = 500  # Monthly budget in USD

# SECURITY - Restrict IP ranges in production
security_config = {
  allowed_ip_ranges = ["YOUR.OFFICE.IP.RANGE/24"]  # Replace with actual ranges
  enable_audit_logs = true
}
```

### Step 3: Terraform Initialization
```bash
# Initialize Terraform
terraform init

# Expected output:
# - Downloads required providers (Google, Google Beta)
# - Initializes backend
# - Ready for planning
```

### Step 4: Deployment Planning
```bash
# Generate deployment plan
terraform plan

# Review the plan carefully:
# - ~25-30 resources to be created
# - Estimated monthly cost
# - Security configurations
# - Resource locations
```

**Plan Verification Checklist:**
- [ ] All resources in correct region (us-central1)
- [ ] Service accounts with least-privilege access
- [ ] Budget alerts configured correctly
- [ ] Storage buckets with lifecycle policies
- [ ] BigQuery tables with partitioning
- [ ] VPN gateway for secure connectivity

### Step 5: Infrastructure Deployment
```bash
# Deploy infrastructure
terraform apply

# Review the plan one more time
# Type 'yes' to confirm deployment

# Deployment typically takes 10-15 minutes
```

### Step 6: Deployment Verification
```bash
# Verify key resources
terraform output

# Expected outputs:
# - Service account emails
# - Storage bucket names
# - BigQuery dataset ID
# - ML workbench URL
# - VPN gateway information
```

## Post-Deployment Configuration

### 1. Service Account Keys (CRITICAL)
```bash
# Generate service account keys for local development
gcloud iam service-accounts keys create vertex-ai-key.json \
  --iam-account=$(terraform output -raw vertex_ai_service_account_email)

gcloud iam service-accounts keys create data-pipeline-key.json \
  --iam-account=$(terraform output -raw data_pipeline_service_account_email)

# Store keys securely
chmod 600 *.json
# Add to .gitignore
echo "*.json" >> .gitignore
```

### 2. Local Environment Configuration
```bash
# Set environment variables
export GOOGLE_APPLICATION_CREDENTIALS="./vertex-ai-key.json"
export PROJECT_ID=$(terraform output -raw project_id)
export BIGQUERY_DATASET=$(terraform output -raw bigquery_dataset_id)

# Test Vertex AI connectivity
python3 -c "
from google.cloud import aiplatform
aiplatform.init(project='$PROJECT_ID', location='us-central1')
print('Vertex AI connection successful!')
"
```

### 3. Data Pipeline Integration
```bash
# Test BigQuery connectivity
bq ls $BIGQUERY_DATASET

# Expected output: List of tables
# - component_analysis_results
# - master_regime_analysis

# Test Cloud Storage access
gsutil ls gs://$(terraform output -raw model_artifacts_bucket)
gsutil ls gs://$(terraform output -raw data_lake_bucket)
```

### 4. Monitoring Setup Verification
```bash
# Check budget alerts
gcloud beta billing budgets list --billing-account=$(grep billing_account_id terraform.tfvars | cut -d'"' -f2)

# Verify monitoring channels
gcloud alpha monitoring channels list
```

## Security Hardening (Production)

### 1. Network Security
```bash
# Verify firewall rules
gcloud compute firewall-rules list --filter="network:market-regime-framework-network"

# Update allowed IP ranges (if needed)
gcloud compute firewall-rules update market-regime-framework-allow-ssh \
  --source-ranges="YOUR.OFFICE.IP.RANGE/24"
```

### 2. IAM Security Review
```bash
# Audit service account permissions
gcloud projects get-iam-policy $PROJECT_ID \
  --flatten="bindings[].members" \
  --format="table(bindings.role)" \
  --filter="bindings.members:*@market-regime-framework*"
```

### 3. Audit Logging (Production)
```bash
# Verify audit logging is enabled
gcloud logging sinks list

# Check for any unusual activity
gcloud logging read "protoPayload.authenticationInfo.principalEmail:*@market-regime-framework*" \
  --limit=10 --format=json
```

## Cost Optimization

### 1. Resource Right-Sizing
```bash
# Check Vertex AI usage
gcloud ai platform models list

# Monitor BigQuery usage
bq show --format=prettyjson $BIGQUERY_DATASET

# Review storage usage
gsutil du -sh gs://$(terraform output -raw model_artifacts_bucket)
```

### 2. Budget Monitoring
```bash
# Set up billing alerts (if not automated)
gcloud beta billing budgets create \
  --billing-account=$BILLING_ACCOUNT_ID \
  --display-name="Market Regime Framework Budget" \
  --budget-amount=500USD

# Monitor current usage
gcloud billing projects describe $PROJECT_ID
```

## Troubleshooting

### Common Issues

#### 1. Quota Exceeded (GPU)
```bash
# Check current quotas
gcloud compute project-info describe --project=$PROJECT_ID

# Request quota increase
# Go to: https://console.cloud.google.com/iam-admin/quotas
# Filter: "Compute Engine API"
# Find: "GPUs (all regions)" and "NVIDIA T4 GPUs"
```

#### 2. Permission Denied
```bash
# Verify authentication
gcloud auth list

# Verify project permissions
gcloud projects get-iam-policy $PROJECT_ID \
  --flatten="bindings[].members" \
  --filter="bindings.members:$(gcloud config get-value account)"
```

#### 3. Billing Issues
```bash
# Check billing status
gcloud beta billing projects describe $PROJECT_ID

# Verify billing account
gcloud beta billing accounts list
```

#### 4. Network Connectivity
```bash
# Test VPN connectivity
gcloud compute vpn-gateways list

# Check firewall rules
gcloud compute firewall-rules list --filter="ALLOW"
```

### Recovery Procedures

#### Partial Deployment Failure
```bash
# Check Terraform state
terraform state list

# Import existing resources if needed
terraform import google_project_service.apis[\"aiplatform.googleapis.com\"] $PROJECT_ID/aiplatform.googleapis.com

# Retry deployment
terraform apply
```

#### Complete Infrastructure Reset
```bash
# Destroy all resources (CAUTION: This deletes everything)
terraform destroy

# Wait for complete cleanup (5-10 minutes)
# Redeploy from scratch
terraform apply
```

## Maintenance & Updates

### Regular Maintenance Tasks

#### Weekly
- [ ] Review cost reports and budget alerts
- [ ] Check resource utilization metrics
- [ ] Verify backup and snapshot policies

#### Monthly
- [ ] Update Terraform providers
- [ ] Review and rotate service account keys
- [ ] Audit security configurations
- [ ] Optimize resource allocations based on usage

#### Quarterly
- [ ] Review and update budget allocations
- [ ] Conduct security audit
- [ ] Update disaster recovery procedures
- [ ] Review and optimize data retention policies

### Infrastructure Updates
```bash
# Update Terraform providers
terraform init -upgrade

# Plan updates
terraform plan

# Apply updates during maintenance window
terraform apply
```

## Integration with Development Workflow

### 1. Environment Promotion
```bash
# Development → Staging
cd environments/staging/
terraform workspace select staging
terraform apply -var-file="staging.tfvars"

# Staging → Production  
cd environments/production/
terraform workspace select production
terraform apply -var-file="production.tfvars"
```

### 2. CI/CD Integration
```yaml
# .github/workflows/infrastructure.yml
name: Infrastructure Deployment
on:
  push:
    branches: [main]
    paths: ['infrastructure/**']

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: hashicorp/setup-terraform@v2
      - name: Terraform Apply
        run: |
          cd infrastructure/terraform
          terraform init
          terraform apply -auto-approve
```

## Support & Documentation

### Key Resources
- **Terraform Docs:** https://developer.hashicorp.com/terraform/docs
- **GCP Documentation:** https://cloud.google.com/docs
- **Vertex AI Docs:** https://cloud.google.com/vertex-ai/docs

### Getting Help
1. **Internal Documentation:** Check project-specific docs in `/docs/`
2. **GCP Support:** Use Google Cloud Console support
3. **Terraform Issues:** HashiCorp documentation and community forums

### Emergency Contacts
- **Infrastructure Team:** infrastructure@company.com
- **GCP Support:** [Your GCP Support Plan]
- **On-Call:** [Your incident response system]

---

**⚠️ IMPORTANT NOTES:**
- Always test in development environment first
- Maintain Terraform state backups
- Document any manual changes outside of Terraform
- Regular cost monitoring is essential for cloud resources
- Security review required for production deployments

This infrastructure deployment provides the foundation for the Market Regime Master Framework's 8-component adaptive learning system with comprehensive monitoring, cost controls, and security measures.