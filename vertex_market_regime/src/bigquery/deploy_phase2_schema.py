#!/usr/bin/env python3
"""
BigQuery Phase 2 Schema Deployment and Validation Script

Deploys all Phase 2 enhanced schemas with momentum features and validates
the complete Epic 1 Phase 2 implementation (932 total features).

Component Enhancements:
- Component 1: 120 → 150 features (+30 momentum)
- Component 6: 200 → 220 features (+20 momentum-enhanced correlation)  
- Component 7: 120 → 130 features (+10 momentum-based level detection)
- Total System: 872 → 932 features (+60 enhancements)
"""

import os
import sys
import logging
import time
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from google.cloud import bigquery
from google.cloud.exceptions import GoogleCloudError, NotFound

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Phase2SchemaDeployer:
    """Deploy and validate Phase 2 enhanced BigQuery schemas"""
    
    def __init__(self, project_id: str = "arched-bot-269016", environment: str = "dev"):
        """Initialize deployer with project and environment"""
        self.project_id = project_id
        self.environment = environment
        self.client = bigquery.Client(project=project_id)
        self.dataset_id = f"market_regime_{environment}"
        self.ddl_dir = Path(__file__).parent / "ddl"
        
        # Phase 2 enhanced feature counts
        self.expected_features = {
            "c1_features": 150,  # Phase 2: 120 + 30 momentum
            "c2_features": 98,   # Unchanged
            "c3_features": 105,  # Unchanged
            "c4_features": 87,   # Unchanged  
            "c5_features": 94,   # Unchanged
            "c6_features": 220,  # Phase 2: 200 + 20 momentum-enhanced correlation
            "c7_features": 130,  # Phase 2: 120 + 10 momentum-based levels
            "c8_features": 48,   # Unchanged
            "training_dataset": 932  # Phase 2 total
        }
        
        logger.info(f"Initialized Phase 2 deployer for {project_id}.{self.dataset_id}")
    
    def deploy_all_schemas(self) -> Dict[str, bool]:
        """
        Deploy all Phase 2 enhanced schemas
        
        Returns:
            Deployment results for each table
        """
        results = {}
        
        logger.info("🚀 Starting Epic 1 Phase 2 Schema Deployment")
        logger.info("=" * 70)
        logger.info("Enhanced Components:")
        logger.info("  • Component 1: +30 momentum features (150 total)")
        logger.info("  • Component 6: +20 momentum-enhanced correlation features (220 total)")
        logger.info("  • Component 7: +10 momentum-based level features (130 total)")
        logger.info("  • Total System: 932 features (+60 enhancements)")
        logger.info("=" * 70)
        
        # Deploy component tables in dependency order
        deployment_order = [
            "c1_features.sql",  # Foundation: momentum features
            "c2_features.sql", 
            "c3_features.sql",
            "c4_features.sql",
            "c5_features.sql",
            "c6_features.sql",  # Uses Component 1 momentum
            "c7_features.sql",  # Uses Components 1 + 6
            "c8_features.sql",
            "training_dataset.sql"  # Final: all enhanced features
        ]
        
        for ddl_file in deployment_order:
            table_name = ddl_file.replace('.sql', '')
            logger.info(f"\n📋 Deploying {table_name}...")
            
            try:
                success = self.deploy_single_schema(ddl_file)
                results[table_name] = success
                
                if success:
                    logger.info(f"✅ {table_name} deployed successfully")
                    # Validate feature count for enhanced tables
                    if table_name in ["c1_features", "c6_features", "c7_features"]:
                        self.validate_enhanced_features(table_name)
                else:
                    logger.error(f"❌ {table_name} deployment failed")
                    
            except Exception as e:
                logger.error(f"❌ {table_name} deployment error: {e}")
                results[table_name] = False
        
        return results
    
    def deploy_single_schema(self, ddl_file: str) -> bool:
        """
        Deploy a single DDL file
        
        Args:
            ddl_file: DDL filename
            
        Returns:
            True if successful, False otherwise
        """
        ddl_path = self.ddl_dir / ddl_file
        
        if not ddl_path.exists():
            logger.error(f"DDL file not found: {ddl_path}")
            return False
        
        try:
            # Read DDL content
            with open(ddl_path, 'r') as f:
                ddl_content = f.read()
            
            # Replace environment placeholder
            ddl_content = ddl_content.replace('{env}', self.environment)
            
            # Execute DDL
            job = self.client.query(ddl_content)
            job.result()  # Wait for completion
            
            logger.info(f"  📊 Schema executed successfully")
            return True
            
        except GoogleCloudError as e:
            logger.error(f"  ❌ BigQuery error: {e}")
            return False
        except Exception as e:
            logger.error(f"  ❌ Unexpected error: {e}")
            return False
    
    def validate_enhanced_features(self, table_name: str) -> bool:
        """
        Validate Phase 2 enhanced features for specific tables
        
        Args:
            table_name: Table to validate
            
        Returns:
            True if validation passes
        """
        try:
            table_ref = f"{self.project_id}.{self.dataset_id}.{table_name}"
            table = self.client.get_table(table_ref)
            
            # Count features (exclude common columns and metadata)
            excluded_columns = {
                'symbol', 'ts_minute', 'date', 'dte', 'zone_name', 
                'created_at', 'updated_at'
            }
            
            feature_columns = [
                field.name for field in table.schema 
                if field.name not in excluded_columns
            ]
            
            actual_count = len(feature_columns)
            expected_count = self.expected_features.get(table_name, 0)
            
            logger.info(f"  🔍 Feature validation: {actual_count}/{expected_count} features")
            
            if actual_count == expected_count:
                logger.info(f"  ✅ Feature count validation passed")
                
                # Log Phase 2 momentum features for enhanced components
                if table_name == "c1_features":
                    momentum_features = [col for col in feature_columns if 'rsi_' in col or 'macd_' in col or 'momentum_' in col]
                    logger.info(f"  📈 Component 1 momentum features: {len(momentum_features)}/30")
                elif table_name == "c6_features":
                    correlation_features = [col for col in feature_columns if 'rsi_cross' in col or 'macd_signal' in col or 'momentum_consensus' in col]
                    logger.info(f"  🔗 Component 6 momentum-correlation features: {len(correlation_features)}/20")
                elif table_name == "c7_features":
                    level_features = [col for col in feature_columns if 'rsi_overbought' in col or 'macd_crossover' in col or 'exhaustion' in col]
                    logger.info(f"  📊 Component 7 momentum-level features: {len(level_features)}/10")
                
                return True
            else:
                logger.warning(f"  ⚠️ Feature count mismatch: expected {expected_count}, got {actual_count}")
                return False
                
        except NotFound:
            logger.error(f"  ❌ Table {table_name} not found")
            return False
        except Exception as e:
            logger.error(f"  ❌ Validation error: {e}")
            return False
    
    def validate_training_dataset(self) -> bool:
        """
        Validate the complete training dataset with all 932 features
        
        Returns:
            True if validation passes
        """
        logger.info("\n🎯 Validating Complete Training Dataset (Phase 2)")
        
        try:
            query = f"""
            SELECT COUNT(*) as column_count
            FROM `{self.project_id}.{self.dataset_id}.INFORMATION_SCHEMA.COLUMNS`
            WHERE table_name = 'training_dataset'
            AND column_name NOT IN ('symbol', 'ts_minute', 'date', 'dte', 'zone_name')
            """
            
            job = self.client.query(query)
            result = list(job.result())[0]
            actual_features = result.column_count
            expected_features = 932
            
            logger.info(f"  📊 Training dataset features: {actual_features}/{expected_features}")
            
            if actual_features == expected_features:
                logger.info("  ✅ Training dataset validation passed")
                return True
            else:
                logger.warning(f"  ⚠️ Training dataset feature mismatch")
                return False
                
        except Exception as e:
            logger.error(f"  ❌ Training dataset validation error: {e}")
            return False
    
    def generate_deployment_report(self, results: Dict[str, bool]) -> None:
        """
        Generate comprehensive deployment report
        
        Args:
            results: Deployment results by table
        """
        logger.info("\n" + "=" * 70)
        logger.info("📋 EPIC 1 PHASE 2 DEPLOYMENT REPORT")
        logger.info("=" * 70)
        
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        logger.info(f"📊 Deployment Status: {successful}/{total} tables deployed successfully")
        logger.info("")
        
        # Enhanced components summary
        enhanced_components = {
            "c1_features": ("Component 1 Triple Straddle", "120 → 150", "+30 momentum"),
            "c6_features": ("Component 6 Correlation", "200 → 220", "+20 momentum-correlation"),
            "c7_features": ("Component 7 Support/Resistance", "120 → 130", "+10 momentum-levels")
        }
        
        logger.info("🚀 Phase 2 Enhanced Components:")
        for table, (name, change, desc) in enhanced_components.items():
            status = "✅" if results.get(table, False) else "❌"
            logger.info(f"  {status} {name}: {change} features ({desc})")
        
        logger.info("")
        logger.info("📈 Epic 1 Phase 2 Summary:")
        logger.info(f"  • Total Features: 872 → 932 (+60 momentum enhancements)")
        logger.info(f"  • Coordinated Implementation: Component 1 → 6 → 7 dependency chain")
        logger.info(f"  • Processing Budget: Component 1 expanded to 190ms (approved)")
        logger.info(f"  • Feature Synergy: Momentum → Correlation → Support/Resistance")
        
        if successful == total:
            logger.info("\n🎉 Epic 1 Phase 2 deployment completed successfully!")
            logger.info("   Ready for production ML model training with enhanced features.")
        else:
            logger.warning(f"\n⚠️ Deployment incomplete: {total - successful} tables failed")
            logger.warning("   Review error logs and retry failed deployments.")
        
        logger.info("=" * 70)
    
    def run_complete_deployment(self) -> bool:
        """
        Run complete Phase 2 deployment and validation
        
        Returns:
            True if all deployments successful
        """
        start_time = time.time()
        
        # Deploy all schemas
        results = self.deploy_all_schemas()
        
        # Validate training dataset
        training_valid = self.validate_training_dataset()
        results["training_validation"] = training_valid
        
        # Generate report
        self.generate_deployment_report(results)
        
        # Performance summary
        duration = time.time() - start_time
        logger.info(f"\n⏱️ Total deployment time: {duration:.2f} seconds")
        
        # Return overall success
        all_successful = all(results.values())
        return all_successful


def main():
    """Main deployment entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy BigQuery Phase 2 Enhanced Schemas")
    parser.add_argument("--env", default="dev", choices=["dev", "staging", "prod"],
                       help="Target environment")
    parser.add_argument("--project", default="arched-bot-269016",
                       help="GCP Project ID")
    
    args = parser.parse_args()
    
    # Initialize deployer
    deployer = Phase2SchemaDeployer(
        project_id=args.project,
        environment=args.env
    )
    
    # Run deployment
    success = deployer.run_complete_deployment()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()