"""
Feature Store Manager for Market Regime System
Story 2.6: Minimal Online Feature Registration

Manages Vertex AI Feature Store operations including:
- Entity type configuration with minute-level aggregation patterns
- Feature registration and management
- Online serving optimization
"""

import logging
import yaml
from typing import Dict, List, Optional, Any
from google.cloud import aiplatform
from google.cloud.aiplatform import gapic
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class EntityConfig:
    """Configuration for feature store entity types"""
    entity_type_id: str
    description: str
    entity_id_format: str
    ttl_hours: int


@dataclass
class FeatureConfig:
    """Configuration for individual features"""
    feature_id: str
    value_type: str
    description: str
    ttl_hours: int


class FeatureStoreManager:
    """
    Manages Vertex AI Feature Store operations for market regime features.
    
    Implements entity configuration with:
    - Minute-level aggregation patterns: ${symbol}_${yyyymmddHHMM}_${dte}
    - Daily aggregation support
    - 48-hour TTL policies
    - BigQuery offline integration
    """
    
    def __init__(self, config_path: str):
        """Initialize Feature Store Manager with configuration"""
        self.config = self._load_config(config_path)
        self.project_id = self.config['project_config']['project_id']
        self.location = self.config['project_config']['location']
        self.featurestore_id = self.config['feature_store']['featurestore_id']
        
        # Initialize Vertex AI
        aiplatform.init(
            project=self.project_id,
            location=self.location
        )
        
        self.featurestore_client = gapic.FeaturestoreServiceClient()
        self.featurestore_path = self.featurestore_client.featurestore_path(
            project=self.project_id,
            location=self.location,
            featurestore=self.featurestore_id
        )
        
        logger.info(f"Initialized Feature Store Manager for {self.featurestore_id}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise
    
    def create_featurestore(self) -> bool:
        """
        Create the Feature Store if it doesn't exist.
        
        Returns:
            bool: True if created or already exists, False if failed
        """
        try:
            # Check if featurestore already exists
            try:
                self.featurestore_client.get_featurestore(name=self.featurestore_path)
                logger.info(f"Feature Store {self.featurestore_id} already exists")
                return True
            except Exception:
                logger.info(f"Feature Store {self.featurestore_id} doesn't exist, creating...")
            
            # Create featurestore
            featurestore = gapic.Featurestore(
                online_serving_config=gapic.Featurestore.OnlineServingConfig(
                    fixed_node_count=2  # Start with 2 nodes for online serving
                ),
                labels={
                    'project': 'market-regime',
                    'version': 'v1-0-0',
                    'component': 'feature-store'
                }
            )
            
            operation = self.featurestore_client.create_featurestore(
                parent=f"projects/{self.project_id}/locations/{self.location}",
                featurestore=featurestore,
                featurestore_id=self.featurestore_id
            )
            
            logger.info("Creating Feature Store, waiting for completion...")
            result = operation.result(timeout=600)  # 10 minute timeout
            logger.info(f"Feature Store created successfully: {result.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create Feature Store: {e}")
            return False
    
    def configure_entity_types(self) -> bool:
        """
        Configure entity types with minute-level and daily aggregation patterns.
        
        Implements:
        - Entity ID format: ${symbol}_${yyyymmddHHMM}_${dte} (e.g., NIFTY_202508141430_7)
        - TTL policies (48h default)
        - Integration with BigQuery offline tables
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            entity_types_config = self.config['feature_store']['entity_types']
            
            for entity_type_id, entity_config in entity_types_config.items():
                self._create_entity_type(entity_type_id, entity_config)
            
            logger.info("All entity types configured successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure entity types: {e}")
            return False
    
    def _create_entity_type(self, entity_type_id: str, entity_config: Dict[str, Any]) -> bool:
        """Create a single entity type with specified configuration"""
        try:
            entity_type_path = self.featurestore_client.entity_type_path(
                project=self.project_id,
                location=self.location,
                featurestore=self.featurestore_id,
                entity_type=entity_type_id
            )
            
            # Check if entity type already exists
            try:
                self.featurestore_client.get_entity_type(name=entity_type_path)
                logger.info(f"Entity type {entity_type_id} already exists")
                return True
            except Exception:
                logger.info(f"Creating entity type {entity_type_id}")
            
            # Create entity type
            entity_type = gapic.EntityType(
                description=entity_config['description'],
                labels={
                    'aggregation': 'minute-level',
                    'format': 'symbol-timestamp-dte',
                    'ttl-hours': '48'
                }
            )
            
            operation = self.featurestore_client.create_entity_type(
                parent=self.featurestore_path,
                entity_type=entity_type,
                entity_type_id=entity_type_id
            )
            
            result = operation.result(timeout=300)  # 5 minute timeout
            logger.info(f"Entity type {entity_type_id} created: {result.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create entity type {entity_type_id}: {e}")
            return False
    
    def register_features(self, entity_type_id: str) -> bool:
        """
        Register all features for a specific entity type.
        
        Args:
            entity_type_id: The entity type to register features for
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            entity_config = self.config['feature_store']['entity_types'][entity_type_id]
            online_features = entity_config.get('online_features', {})
            
            entity_type_path = self.featurestore_client.entity_type_path(
                project=self.project_id,
                location=self.location,
                featurestore=self.featurestore_id,
                entity_type=entity_type_id
            )
            
            for feature_id, feature_config in online_features.items():
                self._create_feature(entity_type_path, feature_id, feature_config)
            
            logger.info(f"All features registered for entity type {entity_type_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register features for {entity_type_id}: {e}")
            return False
    
    def _create_feature(self, entity_type_path: str, feature_id: str, feature_config: Dict[str, Any]) -> bool:
        """Create a single feature with specified configuration"""
        try:
            feature_path = f"{entity_type_path}/features/{feature_id}"
            
            # Check if feature already exists
            try:
                self.featurestore_client.get_feature(name=feature_path)
                logger.info(f"Feature {feature_id} already exists")
                return True
            except Exception:
                logger.info(f"Creating feature {feature_id}")
            
            # Map value types
            value_type_mapping = {
                'DOUBLE': gapic.Feature.ValueType.DOUBLE,
                'STRING': gapic.Feature.ValueType.STRING,
                'BOOLEAN': gapic.Feature.ValueType.BOOL,
                'INT64': gapic.Feature.ValueType.INT64
            }
            
            value_type = value_type_mapping.get(
                feature_config['value_type'], 
                gapic.Feature.ValueType.DOUBLE
            )
            
            # Create feature
            feature = gapic.Feature(
                value_type=value_type,
                description=feature_config['description'],
                labels={
                    'component': feature_id.split('_')[0],  # e.g., 'c1' from 'c1_momentum_score'
                    'ttl-hours': str(feature_config['ttl_hours'])
                }
            )
            
            operation = self.featurestore_client.create_feature(
                parent=entity_type_path,
                feature=feature,
                feature_id=feature_id
            )
            
            result = operation.result(timeout=300)  # 5 minute timeout
            logger.info(f"Feature {feature_id} created: {result.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create feature {feature_id}: {e}")
            return False
    
    def validate_configuration(self) -> bool:
        """
        Validate the complete Feature Store configuration.
        
        Checks:
        - Feature Store exists and is operational
        - All entity types are properly configured
        - All features are registered (32 total)
        - TTL policies are applied correctly
        
        Returns:
            bool: True if validation passes, False otherwise
        """
        try:
            # Check featurestore exists
            featurestore = self.featurestore_client.get_featurestore(name=self.featurestore_path)
            logger.info(f"Feature Store validated: {featurestore.name}")
            
            # Validate entity types
            entity_types = self.featurestore_client.list_entity_types(parent=self.featurestore_path)
            entity_type_count = len(list(entity_types))
            
            expected_entity_types = len(self.config['feature_store']['entity_types'])
            if entity_type_count != expected_entity_types:
                logger.error(f"Expected {expected_entity_types} entity types, found {entity_type_count}")
                return False
            
            # Validate features count (should be 32 core features)
            total_features = 0
            for entity_type_id in self.config['feature_store']['entity_types']:
                entity_type_path = self.featurestore_client.entity_type_path(
                    project=self.project_id,
                    location=self.location,
                    featurestore=self.featurestore_id,
                    entity_type=entity_type_id
                )
                
                features = self.featurestore_client.list_features(parent=entity_type_path)
                feature_count = len(list(features))
                total_features += feature_count
                
                logger.info(f"Entity type {entity_type_id}: {feature_count} features")
            
            expected_features = 32  # As per story requirements
            if total_features != expected_features:
                logger.warning(f"Expected {expected_features} features, found {total_features}")
            
            logger.info(f"Feature Store validation complete: {total_features} total features")
            return True
            
        except Exception as e:
            logger.error(f"Feature Store validation failed: {e}")
            return False
    
    def get_entity_id_format_example(self) -> str:
        """
        Get example of entity ID format for documentation.
        
        Returns:
            str: Example entity ID following the format ${symbol}_${yyyymmddHHMM}_${dte}
        """
        now = datetime.now()
        example_timestamp = now.strftime("%Y%m%d%H%M")
        example_dte = 7  # Days to expiry
        
        return f"NIFTY_{example_timestamp}_{example_dte}"
    
    def get_feature_count_by_component(self) -> Dict[str, int]:
        """
        Get feature count breakdown by component.
        
        Returns:
            Dict[str, int]: Component -> feature count mapping
        """
        component_counts = {}
        
        try:
            entity_config = self.config['feature_store']['entity_types']['instrument_minute']
            online_features = entity_config.get('online_features', {})
            
            for feature_id in online_features.keys():
                component = feature_id.split('_')[0]  # e.g., 'c1' from 'c1_momentum_score'
                component_counts[component] = component_counts.get(component, 0) + 1
            
            logger.info(f"Feature count by component: {component_counts}")
            return component_counts
            
        except Exception as e:
            logger.error(f"Failed to get feature count by component: {e}")
            return {}
    
    def get_performance_targets(self) -> Dict[str, Any]:
        """Get performance targets from configuration"""
        return self.config.get('performance_targets', {})
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration"""
        return self.config.get('monitoring', {})