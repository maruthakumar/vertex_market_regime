"""
Vertex AI Feature Store Integration Module
Handles batch ingestion from BigQuery and online serving configuration
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from google.cloud import aiplatform
from google.cloud import bigquery


@dataclass
class FeatureStoreConfig:
    """Configuration for Vertex AI Feature Store"""
    project_id: str = "arched-bot-269016"
    location: str = "us-central1"
    featurestore_id: str = "market_regime_featurestore"
    entity_type_id: str = "instrument_minute"
    online_serving_ttl_hours: int = 48
    batch_serving_ttl_days: int = 90


class FeatureStoreBigQueryIntegration:
    """Manages integration between BigQuery and Vertex AI Feature Store"""
    
    def __init__(self, config: Optional[FeatureStoreConfig] = None):
        """Initialize integration with configuration"""
        self.config = config or FeatureStoreConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize clients
        aiplatform.init(
            project=self.config.project_id,
            location=self.config.location
        )
        self.bq_client = bigquery.Client(project=self.config.project_id)
    
    def create_batch_ingestion_pipeline(self, environment: str = "dev") -> Dict[str, Any]:
        """
        Design batch ingestion pipeline from BigQuery to Feature Store
        
        Args:
            environment: Environment (dev/staging/prod)
            
        Returns:
            Pipeline configuration dictionary
        """
        pipeline_config = {
            "name": f"market_regime_batch_ingestion_{environment}",
            "source": {
                "type": "bigquery",
                "dataset": f"market_regime_{environment}",
                "tables": [
                    "c1_features", "c2_features", "c3_features", "c4_features",
                    "c5_features", "c6_features", "c7_features", "c8_features"
                ]
            },
            "destination": {
                "type": "vertex_ai_feature_store",
                "featurestore_id": self.config.featurestore_id,
                "entity_type_id": self.config.entity_type_id
            },
            "features": {
                "online_features": [
                    # Component 1
                    "c1_momentum_score", "c1_vol_compression", "c1_breakout_probability",
                    # Component 2
                    "c2_gamma_exposure", "c2_sentiment_level", "c2_pin_risk_score",
                    # Component 3
                    "c3_institutional_flow_score", "c3_divergence_type", "c3_range_expansion_score",
                    # Component 4
                    "c4_skew_bias_score", "c4_term_structure_signal", "c4_iv_regime_level",
                    # Component 5
                    "c5_momentum_score", "c5_volatility_regime_score", "c5_confluence_score",
                    # Component 6
                    "c6_correlation_agreement_score", "c6_breakdown_alert", "c6_system_stability_score",
                    # Component 7
                    "c7_level_strength_score", "c7_breakout_probability",
                    # Component 8
                    "c8_component_agreement_score", "c8_integration_confidence", "c8_transition_probability_hint",
                    # Context
                    "zone_name", "ts_minute", "symbol", "dte"
                ],
                "total_online_count": 32
            },
            "ingestion": {
                "mode": "batch",
                "frequency": "every_1_minute",
                "batch_size": 10000,
                "parallelism": 4
            },
            "data_validation": {
                "null_checks": True,
                "schema_validation": True,
                "range_checks": True,
                "anomaly_detection": True
            },
            "monitoring": {
                "metrics": [
                    "ingestion_latency_ms",
                    "records_processed",
                    "validation_failures",
                    "feature_freshness"
                ],
                "alerting": {
                    "latency_threshold_ms": 5000,
                    "failure_rate_threshold": 0.01
                }
            }
        }
        
        return pipeline_config
    
    def configure_online_serving(self) -> Dict[str, Any]:
        """
        Configure online serving specifications for <50ms latency
        
        Returns:
            Online serving configuration
        """
        serving_config = {
            "endpoint_configuration": {
                "name": "market_regime_online_endpoint",
                "machine_type": "n1-standard-4",
                "min_replica_count": 2,
                "max_replica_count": 10,
                "auto_scaling": {
                    "target_cpu_utilization": 70,
                    "target_throughput_utilization": 70
                }
            },
            "performance_targets": {
                "latency_p50_ms": 25,
                "latency_p95_ms": 40,
                "latency_p99_ms": 50,
                "throughput_rps": 1000
            },
            "caching": {
                "enabled": True,
                "ttl_seconds": 60,
                "cache_size_mb": 1024
            },
            "optimization": {
                "feature_selection": "online_only",
                "batch_prediction": True,
                "connection_pooling": True,
                "compression": "snappy"
            },
            "monitoring": {
                "latency_tracking": True,
                "error_rate_tracking": True,
                "feature_drift_detection": True
            }
        }
        
        return serving_config
    
    def define_feature_versioning_strategy(self) -> Dict[str, Any]:
        """
        Define feature versioning strategy for Feature Store
        
        Returns:
            Versioning strategy configuration
        """
        versioning_strategy = {
            "versioning_scheme": "semantic",
            "version_format": "v{major}.{minor}.{patch}",
            "current_version": "v1.0.0",
            "version_metadata": {
                "created_by": "system",
                "created_at": datetime.utcnow().isoformat(),
                "description": "Initial feature set from 8-component system"
            },
            "compatibility_rules": {
                "backward_compatible": True,
                "deprecation_period_days": 30,
                "migration_support": True
            },
            "version_tracking": {
                "storage": "bigquery",
                "table": "feature_version_history",
                "columns": [
                    "version_id",
                    "feature_name",
                    "feature_type",
                    "created_at",
                    "deprecated_at",
                    "migration_path"
                ]
            },
            "rollback_capability": {
                "enabled": True,
                "max_rollback_versions": 3,
                "rollback_window_hours": 24
            }
        }
        
        return versioning_strategy
    
    def create_point_in_time_retrieval_procedure(self) -> str:
        """
        Create SQL procedure for point-in-time correct feature retrieval
        
        Returns:
            SQL procedure as string
        """
        procedure_sql = """
        CREATE OR REPLACE PROCEDURE `arched-bot-269016.market_regime_{env}.get_point_in_time_features`(
            IN request_timestamp TIMESTAMP,
            IN symbol_filter STRING,
            IN dte_filter INT64,
            IN lookback_minutes INT64
        )
        BEGIN
            -- Point-in-time correct feature retrieval
            -- Ensures no data leakage by only using features available at request_timestamp
            
            DECLARE cutoff_timestamp TIMESTAMP;
            SET cutoff_timestamp = TIMESTAMP_SUB(request_timestamp, INTERVAL lookback_minutes MINUTE);
            
            -- Create temporary table with point-in-time features
            CREATE TEMP TABLE pit_features AS
            SELECT 
                t.symbol,
                t.ts_minute,
                t.date,
                t.dte,
                t.zone_name,
                -- Online features only for real-time serving
                t.c1_momentum_score,
                t.c1_vol_compression,
                t.c1_breakout_probability,
                t.c2_gamma_exposure,
                t.c2_sentiment_level,
                t.c2_pin_risk_score,
                t.c3_institutional_flow_score,
                t.c3_divergence_type,
                t.c3_range_expansion_score,
                t.c4_skew_bias_score,
                t.c4_term_structure_signal,
                t.c4_iv_regime_level,
                t.c5_momentum_score,
                t.c5_volatility_regime_score,
                t.c5_confluence_score,
                t.c6_correlation_agreement_score,
                t.c6_breakdown_alert,
                t.c6_system_stability_score,
                t.c7_level_strength_score,
                t.c7_breakout_probability,
                t.c8_component_agreement_score,
                t.c8_integration_confidence,
                t.c8_transition_probability_hint,
                -- Metadata
                request_timestamp as query_timestamp,
                TIMESTAMP_DIFF(request_timestamp, t.ts_minute, SECOND) as feature_age_seconds
            FROM `arched-bot-269016.market_regime_{env}.training_dataset` t
            WHERE t.ts_minute <= request_timestamp
                AND t.ts_minute >= cutoff_timestamp
                AND t.symbol = symbol_filter
                AND t.dte = dte_filter
            ORDER BY t.ts_minute DESC
            LIMIT 1;
            
            -- Return the point-in-time features
            SELECT * FROM pit_features;
            
            -- Log the retrieval for audit
            INSERT INTO `arched-bot-269016.market_regime_{env}.feature_retrieval_audit`
            (retrieval_timestamp, request_timestamp, symbol, dte, feature_age_seconds, success)
            SELECT 
                CURRENT_TIMESTAMP(),
                request_timestamp,
                symbol,
                dte,
                feature_age_seconds,
                TRUE
            FROM pit_features;
            
        END;
        """
        
        return procedure_sql
    
    def create_feature_lineage_tracking(self) -> Dict[str, Any]:
        """
        Create feature lineage tracking specifications
        
        Returns:
            Lineage tracking configuration
        """
        lineage_config = {
            "tracking_enabled": True,
            "lineage_storage": {
                "type": "bigquery",
                "dataset": "market_regime_metadata",
                "tables": {
                    "feature_lineage": {
                        "columns": [
                            "feature_id",
                            "feature_name",
                            "source_component",
                            "source_table",
                            "transformation_pipeline",
                            "created_timestamp",
                            "last_updated"
                        ]
                    },
                    "feature_dependencies": {
                        "columns": [
                            "feature_id",
                            "depends_on_feature_id",
                            "dependency_type",
                            "transformation_function"
                        ]
                    },
                    "feature_usage": {
                        "columns": [
                            "feature_id",
                            "model_id",
                            "usage_type",
                            "importance_score",
                            "last_accessed"
                        ]
                    }
                }
            },
            "lineage_graph": {
                "visualization": "enabled",
                "format": "dag",
                "update_frequency": "daily"
            },
            "data_quality_tracking": {
                "null_rate": True,
                "distribution_drift": True,
                "schema_changes": True,
                "outlier_detection": True
            },
            "compliance": {
                "gdpr_compliant": True,
                "data_retention_days": 90,
                "pii_detection": True,
                "audit_logging": True
            }
        }
        
        return lineage_config
    
    def estimate_costs(self, environment: str = "dev") -> Dict[str, float]:
        """
        Estimate costs for Feature Store and BigQuery operations
        
        Args:
            environment: Environment (dev/staging/prod)
            
        Returns:
            Cost estimates in USD
        """
        # Assumptions for cost calculation
        daily_entities = 1000000  # 1M entities per day
        online_features = 32
        storage_gb = 45  # From spec
        queries_per_day = 100000
        
        costs = {
            "bigquery": {
                "storage_monthly": storage_gb * 0.02,  # $0.02 per GB
                "queries_monthly": (queries_per_day * 30 * 0.005) / 1000,  # $5 per TB
                "streaming_inserts_monthly": (daily_entities * 30 * 0.01) / 1000000  # $0.01 per 1000 rows
            },
            "feature_store": {
                "online_serving_monthly": (daily_entities * online_features * 30 * 0.0005) / 1000000,  # $0.50 per million
                "batch_export_monthly": (storage_gb * 0.12),  # Export costs
                "entity_storage_monthly": (daily_entities * 30 * 0.0001) / 1000000  # Storage costs
            },
            "total_estimated_monthly": 0
        }
        
        # Calculate totals
        costs["bigquery"]["total"] = sum(costs["bigquery"].values())
        costs["feature_store"]["total"] = sum(costs["feature_store"].values())
        costs["total_estimated_monthly"] = costs["bigquery"]["total"] + costs["feature_store"]["total"]
        
        # Adjust for environment
        if environment == "dev":
            costs = {k: v * 0.1 if isinstance(v, (int, float)) else v for k, v in costs.items()}
        elif environment == "staging":
            costs = {k: v * 0.3 if isinstance(v, (int, float)) else v for k, v in costs.items()}
        
        return costs
    
    def generate_integration_script(self, environment: str = "dev") -> str:
        """
        Generate executable integration script
        
        Args:
            environment: Environment (dev/staging/prod)
            
        Returns:
            Python script as string
        """
        script = f"""#!/usr/bin/env python3
'''
Feature Store BigQuery Integration Script
Environment: {environment}
Generated: {datetime.utcnow().isoformat()}
'''

from google.cloud import aiplatform
from google.cloud import bigquery
import time
import logging

# Configuration
PROJECT_ID = "arched-bot-269016"
LOCATION = "us-central1"
FEATURESTORE_ID = "market_regime_featurestore"
ENTITY_TYPE_ID = "instrument_minute"
DATASET_ID = "market_regime_{environment}"

# Initialize
aiplatform.init(project=PROJECT_ID, location=LOCATION)
bq_client = bigquery.Client(project=PROJECT_ID)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ingest_features_to_feature_store():
    '''Ingest features from BigQuery to Feature Store'''
    
    # Get Feature Store
    featurestore = aiplatform.Featurestore(featurestore_id=FEATURESTORE_ID)
    entity_type = featurestore.get_entity_type(entity_type_id=ENTITY_TYPE_ID)
    
    # Define ingestion query
    query = f'''
    SELECT 
        CONCAT(symbol, '_', FORMAT_TIMESTAMP('%Y%m%d%H%M', ts_minute), '_', CAST(dte AS STRING)) as entity_id,
        ts_minute,
        -- Online features
        c1_momentum_score, c1_vol_compression, c1_breakout_probability,
        c2_gamma_exposure, c2_sentiment_level, c2_pin_risk_score,
        c3_institutional_flow_score, c3_divergence_type, c3_range_expansion_score,
        c4_skew_bias_score, c4_term_structure_signal, c4_iv_regime_level,
        c5_momentum_score, c5_volatility_regime_score, c5_confluence_score,
        c6_correlation_agreement_score, c6_breakdown_alert, c6_system_stability_score,
        c7_level_strength_score, c7_breakout_probability,
        c8_component_agreement_score, c8_integration_confidence, c8_transition_probability_hint,
        zone_name, symbol, dte
    FROM `{{PROJECT_ID}}.{{DATASET_ID}}.training_dataset`
    WHERE ts_minute >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
    '''
    
    # Start ingestion job
    logger.info("Starting batch ingestion job...")
    ingestion_job = entity_type.batch_create_features(
        feature_configs={{
            "c1_momentum_score": {{"value_type": "DOUBLE"}},
            # ... (other feature configs)
        }},
        bigquery_source_uri=f"bq://{{PROJECT_ID}}.{{DATASET_ID}}.temp_ingestion_table",
        entity_id_field="entity_id",
        feature_time_field="ts_minute",
        disable_online_serving=False
    )
    
    # Monitor job
    logger.info(f"Ingestion job started: {{ingestion_job.name}}")
    ingestion_job.wait()
    logger.info("Ingestion completed successfully")
    
    return ingestion_job

if __name__ == "__main__":
    start_time = time.time()
    try:
        job = ingest_features_to_feature_store()
        logger.info(f"Integration completed in {{time.time() - start_time:.2f}} seconds")
    except Exception as e:
        logger.error(f"Integration failed: {{str(e)}}")
        raise
"""
        
        return script


if __name__ == "__main__":
    # Initialize integration
    integration = FeatureStoreBigQueryIntegration()
    
    # Generate configurations
    print("Batch Ingestion Pipeline Configuration:")
    print("-" * 50)
    pipeline = integration.create_batch_ingestion_pipeline("dev")
    print(f"Pipeline Name: {pipeline['name']}")
    print(f"Online Features: {pipeline['features']['total_online_count']}")
    print(f"Ingestion Frequency: {pipeline['ingestion']['frequency']}")
    
    print("\nOnline Serving Configuration:")
    print("-" * 50)
    serving = integration.configure_online_serving()
    print(f"Target Latency P99: {serving['performance_targets']['latency_p99_ms']}ms")
    print(f"Target Throughput: {serving['performance_targets']['throughput_rps']} RPS")
    
    print("\nFeature Versioning Strategy:")
    print("-" * 50)
    versioning = integration.define_feature_versioning_strategy()
    print(f"Current Version: {versioning['current_version']}")
    print(f"Versioning Scheme: {versioning['versioning_scheme']}")
    
    print("\nCost Estimates (Monthly USD):")
    print("-" * 50)
    costs = integration.estimate_costs("dev")
    print(f"BigQuery: ${costs['bigquery']['total']:.2f}")
    print(f"Feature Store: ${costs['feature_store']['total']:.2f}")
    print(f"Total: ${costs['total_estimated_monthly']:.2f}")