#!/usr/bin/env python3
"""
BigQuery Deployment Script
Creates BigQuery dataset and all component tables for offline feature storage
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional
from google.cloud import bigquery
from google.cloud.exceptions import GoogleCloudError, NotFound, Conflict


class BigQueryDeployer:
    """Deploys BigQuery dataset and tables for the Market Regime system"""
    
    def __init__(self, project_id: str = "arched-bot-269016", environment: str = "dev"):
        """
        Initialize BigQuery deployer
        
        Args:
            project_id: GCP project ID
            environment: Environment (dev/staging/prod)
        """
        self.project_id = project_id
        self.environment = environment
        self.dataset_id = f"market_regime_{environment}"
        self.client = bigquery.Client(project=project_id)
        self.ddl_dir = Path(__file__).parent / "ddl"
        
    def create_dataset(self) -> bool:
        """
        Create BigQuery dataset with proper configuration
        
        Returns:
            True if successful, False otherwise
        """
        print(f"Creating dataset: {self.dataset_id}")
        
        try:
            # Check if dataset already exists
            try:
                dataset = self.client.get_dataset(self.dataset_id)
                print(f"  ℹ Dataset {self.dataset_id} already exists")
                return True
            except NotFound:
                pass  # Dataset doesn't exist, continue with creation
            
            # Configure dataset
            dataset = bigquery.Dataset(f"{self.project_id}.{self.dataset_id}")
            dataset.location = "US"
            dataset.description = f"Market Regime Feature Tables - {self.environment.upper()} environment"
            
            # Set labels
            dataset.labels = {
                "environment": self.environment,
                "project": "market_regime",
                "component": "feature_store",
                "version": "1.0"
            }
            
            # Set default table expiration based on environment
            if self.environment == "dev":
                dataset.default_table_expiration_ms = 30 * 24 * 60 * 60 * 1000  # 30 days
            elif self.environment == "staging":
                dataset.default_table_expiration_ms = 90 * 24 * 60 * 60 * 1000  # 90 days
            # Production: no expiration
            
            # Create dataset
            dataset = self.client.create_dataset(dataset, timeout=30)
            print(f"  ✓ Created dataset {dataset.dataset_id}")
            return True
            
        except GoogleCloudError as e:
            print(f"  ✗ Failed to create dataset: {str(e)}")
            return False
        except Exception as e:
            print(f"  ✗ Unexpected error creating dataset: {str(e)}")
            return False
    
    def deploy_all_tables(self) -> Dict[str, bool]:
        """
        Deploy all component tables from DDL files
        
        Returns:
            Dictionary with deployment status for each table
        """
        ddl_files = sorted(self.ddl_dir.glob("*.sql"))
        results = {}
        
        print(f"\nDeploying {len(ddl_files)} tables...")
        print("-" * 50)
        
        for ddl_file in ddl_files:
            table_name = ddl_file.stem
            print(f"\nDeploying table: {table_name}")
            
            success = self.deploy_single_table(ddl_file)
            results[table_name] = success
            
            if success:
                print(f"  ✓ Successfully deployed {table_name}")
            else:
                print(f"  ✗ Failed to deploy {table_name}")
                
            # Small delay between deployments
            time.sleep(1)
        
        return results
    
    def deploy_single_table(self, ddl_file: Path) -> bool:
        """
        Deploy a single table from DDL file
        
        Args:
            ddl_file: Path to DDL file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read DDL content
            with open(ddl_file, 'r') as f:
                ddl_content = f.read()
            
            # Replace environment placeholder
            ddl_content = ddl_content.replace("{env}", self.environment)
            
            # Execute DDL
            query_job = self.client.query(ddl_content)
            query_job.result()  # Wait for completion
            
            return True
            
        except GoogleCloudError as e:
            print(f"    Error: {str(e)}")
            return False
        except Exception as e:
            print(f"    Unexpected error: {str(e)}")
            return False
    
    def verify_deployment(self) -> Dict[str, Dict]:
        """
        Verify that all tables were created successfully
        
        Returns:
            Verification results for each table
        """
        expected_tables = [
            "c1_features", "c2_features", "c3_features", "c4_features",
            "c5_features", "c6_features", "c7_features", "c8_features",
            "training_dataset", "mr_load_audit"
        ]
        
        results = {}
        print("\nVerifying deployment...")
        print("-" * 50)
        
        try:
            # List all tables in dataset
            tables = list(self.client.list_tables(self.dataset_id))
            table_names = [table.table_id for table in tables]
            
            for expected_table in expected_tables:
                if expected_table in table_names:
                    # Get table info
                    table_ref = f"{self.project_id}.{self.dataset_id}.{expected_table}"
                    table = self.client.get_table(table_ref)
                    
                    results[expected_table] = {
                        "exists": True,
                        "num_rows": table.num_rows,
                        "partitioning": str(table.time_partitioning) if table.time_partitioning else None,
                        "clustering": table.clustering_fields,
                        "created": table.created.isoformat() if table.created else None
                    }
                    
                    print(f"  ✓ {expected_table} - {table.num_rows} rows")
                    if table.time_partitioning:
                        print(f"    Partitioned by: {table.time_partitioning.field}")
                    if table.clustering_fields:
                        print(f"    Clustered by: {', '.join(table.clustering_fields)}")
                        
                else:
                    results[expected_table] = {"exists": False}
                    print(f"  ✗ {expected_table} - Missing")
            
        except Exception as e:
            print(f"  ✗ Verification failed: {str(e)}")
            return {}
        
        return results
    
    def get_deployment_summary(self, deployment_results: Dict[str, bool], 
                              verification_results: Dict[str, Dict]) -> Dict:
        """
        Generate deployment summary
        
        Args:
            deployment_results: Results from table deployment
            verification_results: Results from verification
            
        Returns:
            Deployment summary
        """
        successful_deployments = sum(1 for success in deployment_results.values() if success)
        total_deployments = len(deployment_results)
        
        verified_tables = sum(1 for result in verification_results.values() 
                            if result.get("exists", False))
        expected_tables = len(verification_results)
        
        return {
            "environment": self.environment,
            "dataset_id": self.dataset_id,
            "deployment_success_rate": f"{successful_deployments}/{total_deployments}",
            "verification_success_rate": f"{verified_tables}/{expected_tables}",
            "deployment_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "successful": successful_deployments == total_deployments and verified_tables == expected_tables
        }


def main():
    """Main deployment function"""
    print("=" * 60)
    print("BigQuery Market Regime Feature Store Deployment")
    print("=" * 60)
    
    # Check for environment argument
    environment = "dev"
    if len(sys.argv) > 1:
        environment = sys.argv[1].lower()
        if environment not in ["dev", "staging", "prod"]:
            print(f"Invalid environment: {environment}")
            print("Valid environments: dev, staging, prod")
            sys.exit(1)
    
    print(f"Environment: {environment.upper()}")
    print(f"Project ID: arched-bot-269016")
    
    try:
        # Initialize deployer
        deployer = BigQueryDeployer(environment=environment)
        
        # Step 1: Create dataset
        print("\nStep 1: Creating BigQuery Dataset")
        print("-" * 30)
        dataset_success = deployer.create_dataset()
        
        if not dataset_success:
            print("✗ Dataset creation failed. Aborting deployment.")
            sys.exit(1)
        
        # Step 2: Deploy all tables
        print("\nStep 2: Deploying Feature Tables")
        print("-" * 30)
        deployment_results = deployer.deploy_all_tables()
        
        # Step 3: Verify deployment
        print("\nStep 3: Verifying Deployment")
        print("-" * 30)
        verification_results = deployer.verify_deployment()
        
        # Step 4: Generate summary
        print("\nStep 4: Deployment Summary")
        print("-" * 30)
        summary = deployer.get_deployment_summary(deployment_results, verification_results)
        
        print(f"Environment: {summary['environment']}")
        print(f"Dataset: {summary['dataset_id']}")
        print(f"Deployment Success: {summary['deployment_success_rate']}")
        print(f"Verification Success: {summary['verification_success_rate']}")
        print(f"Timestamp: {summary['deployment_timestamp']}")
        print(f"Overall Status: {'✓ SUCCESS' if summary['successful'] else '✗ FAILED'}")
        
        # Print next steps
        print("\n" + "=" * 60)
        print("Next Steps:")
        print("-" * 60)
        if summary['successful']:
            print("✓ BigQuery feature store is ready!")
            print("✓ You can now run the data population pipeline")
            print("✓ Execute query performance validation")
            print("\nUseful commands:")
            print(f"  bq ls {summary['dataset_id']}")
            print(f"  bq show {summary['dataset_id']}")
        else:
            print("✗ Deployment had issues. Please review the errors above.")
            print("✗ Fix any issues and re-run deployment")
        
        return summary['successful']
        
    except Exception as e:
        print(f"\n✗ Deployment failed with error: {str(e)}")
        print("\nNote: This script requires Google Cloud credentials.")
        print("Please ensure you have authenticated with: gcloud auth application-default login")
        return False


if __name__ == "__main__":
    # Example usage:
    # python deploy_bigquery.py dev
    # python deploy_bigquery.py staging
    # python deploy_bigquery.py prod
    
    success = main()
    sys.exit(0 if success else 1)