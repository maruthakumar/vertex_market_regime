"""
BigQuery Integration Validator for Feature Store
Story 2.6: Minimal Online Feature Registration - Subtask 1.4

Validates integration between BigQuery offline feature tables and Vertex AI Feature Store:
- Schema consistency validation
- Feature mapping verification
- Data type compatibility checks
- Ingestion pipeline validation
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
from google.cloud import bigquery
from google.cloud import aiplatform
import yaml
import re

logger = logging.getLogger(__name__)


class BigQueryIntegrationValidator:
    """
    Validates BigQuery offline feature tables integration with Feature Store.
    
    Validates:
    - Schema consistency between BigQuery and Feature Store
    - Feature name mapping and compatibility
    - Data type conversions
    - Required columns presence
    - Entity ID generation compatibility
    """
    
    def __init__(self, config_path: str, project_id: str, location: str):
        """Initialize BigQuery Integration Validator"""
        self.config = self._load_config(config_path)
        self.project_id = project_id
        self.location = location
        
        # Initialize clients
        self.bq_client = bigquery.Client(project=project_id)
        
        # Feature Store configuration
        self.featurestore_config = self.config['feature_store']
        self.ingestion_config = self.config['ingestion']
        
        logger.info("BigQuery Integration Validator initialized")
    
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
    
    def validate_offline_tables_exist(self, environment: str = "dev") -> Dict[str, Any]:
        """
        Validate that all required BigQuery offline feature tables exist.
        
        Args:
            environment: Environment name (dev, staging, prod)
            
        Returns:
            Dict[str, Any]: Validation results
        """
        validation_results = {
            'tables_found': [],
            'tables_missing': [],
            'total_expected': 0,
            'total_found': 0,
            'validation_passed': True
        }
        
        try:
            # Get expected tables from config
            expected_tables = self.ingestion_config['data_sources']['bigquery']['tables']
            validation_results['total_expected'] = len(expected_tables)
            
            dataset_pattern = self.ingestion_config['data_sources']['bigquery']['dataset_pattern']
            dataset_name = dataset_pattern.format(env=environment)
            
            for table_name in expected_tables:
                table_id = f"{self.project_id}.{dataset_name}.{table_name}"
                
                try:
                    table = self.bq_client.get_table(table_id)
                    validation_results['tables_found'].append({
                        'table_name': table_name,
                        'table_id': table_id,
                        'num_rows': table.num_rows,
                        'created': table.created.isoformat() if table.created else None,
                        'modified': table.modified.isoformat() if table.modified else None
                    })
                    validation_results['total_found'] += 1
                    
                except Exception as e:
                    validation_results['tables_missing'].append({
                        'table_name': table_name,
                        'table_id': table_id,
                        'error': str(e)
                    })
                    validation_results['validation_passed'] = False
            
            logger.info(f"Table validation: {validation_results['total_found']}/{validation_results['total_expected']} tables found")
            return validation_results
            
        except Exception as e:
            logger.error(f"Failed to validate offline tables: {e}")
            validation_results['validation_passed'] = False
            validation_results['error'] = str(e)
            return validation_results
    
    def validate_schema_compatibility(self, environment: str = "dev") -> Dict[str, Any]:
        """
        Validate schema compatibility between BigQuery tables and Feature Store.
        
        Args:
            environment: Environment name
            
        Returns:
            Dict[str, Any]: Schema validation results
        """
        validation_results = {
            'compatible_features': [],
            'incompatible_features': [],
            'missing_features': [],
            'extra_features': [],
            'validation_passed': True,
            'summary': {}
        }
        
        try:
            # Get Feature Store online features
            online_features = self.featurestore_config['entity_types']['instrument_minute']['online_features']
            
            # Validate each component table
            component_tables = ['c1_features', 'c2_features', 'c3_features', 'c4_features',
                              'c5_features', 'c6_features', 'c7_features', 'c8_features']
            
            for table_name in component_tables:
                component_num = table_name.split('_')[0]  # e.g., 'c1' from 'c1_features'
                table_validation = self._validate_table_schema(
                    table_name, component_num, online_features, environment
                )
                
                validation_results['summary'][table_name] = table_validation
                
                # Aggregate results
                validation_results['compatible_features'].extend(table_validation.get('compatible_features', []))
                validation_results['incompatible_features'].extend(table_validation.get('incompatible_features', []))
                validation_results['missing_features'].extend(table_validation.get('missing_features', []))
                
                if not table_validation.get('validation_passed', False):
                    validation_results['validation_passed'] = False
            
            logger.info(f"Schema validation complete: {len(validation_results['compatible_features'])} compatible features")
            return validation_results
            
        except Exception as e:
            logger.error(f"Failed to validate schema compatibility: {e}")
            validation_results['validation_passed'] = False
            validation_results['error'] = str(e)
            return validation_results
    
    def _validate_table_schema(
        self, 
        table_name: str, 
        component_num: str, 
        online_features: Dict[str, Any],
        environment: str
    ) -> Dict[str, Any]:
        """Validate schema for a specific component table"""
        table_validation = {
            'table_name': table_name,
            'compatible_features': [],
            'incompatible_features': [],
            'missing_features': [],
            'validation_passed': True
        }
        
        try:
            # Get table schema
            dataset_pattern = self.ingestion_config['data_sources']['bigquery']['dataset_pattern']
            dataset_name = dataset_pattern.format(env=environment)
            table_id = f"{self.project_id}.{dataset_name}.{table_name}"
            
            table = self.bq_client.get_table(table_id)
            
            # Create field mapping
            bq_fields = {field.name: field for field in table.schema}
            
            # Check component features
            component_features = {
                feature_id: feature_config 
                for feature_id, feature_config in online_features.items()
                if feature_id.startswith(component_num + '_')
            }
            
            for feature_id, feature_config in component_features.items():
                if feature_id in bq_fields:
                    bq_field = bq_fields[feature_id]
                    compatibility = self._check_field_compatibility(feature_config, bq_field)
                    
                    if compatibility['compatible']:
                        table_validation['compatible_features'].append({
                            'feature_id': feature_id,
                            'bq_type': bq_field.field_type,
                            'fs_type': feature_config['value_type'],
                            'description': feature_config.get('description', '')
                        })
                    else:
                        table_validation['incompatible_features'].append({
                            'feature_id': feature_id,
                            'bq_type': bq_field.field_type,
                            'fs_type': feature_config['value_type'],
                            'reason': compatibility['reason']
                        })
                        table_validation['validation_passed'] = False
                else:
                    table_validation['missing_features'].append({
                        'feature_id': feature_id,
                        'expected_type': feature_config['value_type'],
                        'description': feature_config.get('description', '')
                    })
                    table_validation['validation_passed'] = False
            
            return table_validation
            
        except Exception as e:
            logger.error(f"Failed to validate table schema for {table_name}: {e}")
            table_validation['validation_passed'] = False
            table_validation['error'] = str(e)
            return table_validation
    
    def _check_field_compatibility(self, feature_config: Dict[str, Any], bq_field: bigquery.SchemaField) -> Dict[str, Any]:
        """Check compatibility between Feature Store and BigQuery field types"""
        fs_type = feature_config['value_type']
        bq_type = bq_field.field_type
        
        # Type compatibility mapping
        compatible_mappings = {
            'DOUBLE': ['FLOAT', 'FLOAT64', 'NUMERIC', 'BIGNUMERIC'],
            'STRING': ['STRING'],
            'BOOLEAN': ['BOOLEAN', 'BOOL'],
            'INT64': ['INTEGER', 'INT64']
        }
        
        compatible_bq_types = compatible_mappings.get(fs_type, [])
        
        if bq_type in compatible_bq_types:
            return {'compatible': True, 'reason': 'Direct type match'}
        
        # Check for acceptable conversions
        acceptable_conversions = {
            ('DOUBLE', 'INTEGER'): 'Integer can be converted to double',
            ('DOUBLE', 'INT64'): 'Integer can be converted to double',
            ('STRING', 'FLOAT'): 'Float can be converted to string (warning)',
            ('STRING', 'INTEGER'): 'Integer can be converted to string (warning)'
        }
        
        conversion_key = (fs_type, bq_type)
        if conversion_key in acceptable_conversions:
            return {
                'compatible': True, 
                'reason': acceptable_conversions[conversion_key],
                'conversion_required': True
            }
        
        return {
            'compatible': False,
            'reason': f'Incompatible types: Feature Store {fs_type} vs BigQuery {bq_type}'
        }
    
    def validate_entity_id_generation(self, environment: str = "dev") -> Dict[str, Any]:
        """
        Validate that BigQuery tables contain required columns for entity ID generation.
        
        Entity ID format: ${symbol}_${yyyymmddHHMM}_${dte}
        Required columns: symbol, ts_minute, dte
        
        Args:
            environment: Environment name
            
        Returns:
            Dict[str, Any]: Entity ID validation results
        """
        validation_results = {
            'tables_validated': [],
            'validation_passed': True,
            'missing_columns': [],
            'entity_id_examples': []
        }
        
        required_columns = ['symbol', 'ts_minute', 'dte']
        
        try:
            component_tables = ['c1_features', 'c2_features', 'c3_features', 'c4_features',
                              'c5_features', 'c6_features', 'c7_features', 'c8_features']
            
            dataset_pattern = self.ingestion_config['data_sources']['bigquery']['dataset_pattern']
            dataset_name = dataset_pattern.format(env=environment)
            
            for table_name in component_tables:
                table_id = f"{self.project_id}.{dataset_name}.{table_name}"
                
                try:
                    table = self.bq_client.get_table(table_id)
                    field_names = [field.name for field in table.schema]
                    
                    missing_cols = [col for col in required_columns if col not in field_names]
                    
                    table_result = {
                        'table_name': table_name,
                        'has_required_columns': len(missing_cols) == 0,
                        'missing_columns': missing_cols,
                        'available_columns': field_names
                    }
                    
                    validation_results['tables_validated'].append(table_result)
                    
                    if missing_cols:
                        validation_results['validation_passed'] = False
                        validation_results['missing_columns'].extend([
                            {'table': table_name, 'column': col} for col in missing_cols
                        ])
                    
                    # Generate sample entity ID if possible
                    if not missing_cols:
                        sample_query = f"""
                        SELECT symbol, ts_minute, dte
                        FROM `{table_id}`
                        WHERE symbol IS NOT NULL 
                        AND ts_minute IS NOT NULL 
                        AND dte IS NOT NULL
                        LIMIT 1
                        """
                        
                        try:
                            query_job = self.bq_client.query(sample_query)
                            for row in query_job:
                                entity_id = self._generate_entity_id_from_row(row)
                                validation_results['entity_id_examples'].append({
                                    'table': table_name,
                                    'entity_id': entity_id,
                                    'symbol': row.symbol,
                                    'timestamp': row.ts_minute.isoformat(),
                                    'dte': row.dte
                                })
                                break
                        except Exception as e:
                            logger.warning(f"Could not generate sample entity ID for {table_name}: {e}")
                
                except Exception as e:
                    logger.error(f"Failed to validate entity ID generation for {table_name}: {e}")
                    validation_results['validation_passed'] = False
            
            logger.info(f"Entity ID validation: {len(validation_results['entity_id_examples'])} examples generated")
            return validation_results
            
        except Exception as e:
            logger.error(f"Failed to validate entity ID generation: {e}")
            validation_results['validation_passed'] = False
            validation_results['error'] = str(e)
            return validation_results
    
    def _generate_entity_id_from_row(self, row) -> str:
        """Generate entity ID from BigQuery row data"""
        symbol = row.symbol
        timestamp = row.ts_minute
        dte = row.dte
        
        # Format timestamp as yyyymmddHHMM
        timestamp_str = timestamp.strftime("%Y%m%d%H%M")
        
        return f"{symbol}_{timestamp_str}_{dte}"
    
    def run_comprehensive_validation(self, environment: str = "dev") -> Dict[str, Any]:
        """
        Run comprehensive validation of BigQuery integration.
        
        Args:
            environment: Environment name
            
        Returns:
            Dict[str, Any]: Complete validation results
        """
        comprehensive_results = {
            'overall_passed': True,
            'validation_timestamp': logger.info("Starting comprehensive BigQuery integration validation"),
            'results': {}
        }
        
        try:
            # 1. Validate tables exist
            logger.info("Validating offline tables existence...")
            table_validation = self.validate_offline_tables_exist(environment)
            comprehensive_results['results']['table_existence'] = table_validation
            
            if not table_validation['validation_passed']:
                comprehensive_results['overall_passed'] = False
            
            # 2. Validate schema compatibility
            logger.info("Validating schema compatibility...")
            schema_validation = self.validate_schema_compatibility(environment)
            comprehensive_results['results']['schema_compatibility'] = schema_validation
            
            if not schema_validation['validation_passed']:
                comprehensive_results['overall_passed'] = False
            
            # 3. Validate entity ID generation
            logger.info("Validating entity ID generation...")
            entity_id_validation = self.validate_entity_id_generation(environment)
            comprehensive_results['results']['entity_id_generation'] = entity_id_validation
            
            if not entity_id_validation['validation_passed']:
                comprehensive_results['overall_passed'] = False
            
            # Summary statistics
            comprehensive_results['summary'] = {
                'total_tables_expected': table_validation.get('total_expected', 0),
                'total_tables_found': table_validation.get('total_found', 0),
                'compatible_features_count': len(schema_validation.get('compatible_features', [])),
                'incompatible_features_count': len(schema_validation.get('incompatible_features', [])),
                'entity_id_examples_count': len(entity_id_validation.get('entity_id_examples', []))
            }
            
            logger.info(f"Comprehensive validation complete. Overall passed: {comprehensive_results['overall_passed']}")
            return comprehensive_results
            
        except Exception as e:
            logger.error(f"Comprehensive validation failed: {e}")
            comprehensive_results['overall_passed'] = False
            comprehensive_results['error'] = str(e)
            return comprehensive_results
    
    def get_integration_recommendations(self) -> List[str]:
        """Get recommendations for improving BigQuery integration"""
        recommendations = [
            "Ensure all component tables (c1-c8) have consistent schema structure",
            "Verify that symbol, ts_minute, and dte columns exist in all tables",
            "Check data type compatibility between BigQuery and Feature Store",
            "Monitor ingestion pipeline performance and latency",
            "Implement data quality checks for null values and outliers",
            "Set up monitoring for schema drift detection",
            "Consider partitioning BigQuery tables by date for performance",
            "Implement backup and recovery procedures for feature data",
            "Test feature freshness and TTL behavior",
            "Validate feature serving latency meets <50ms requirement"
        ]
        
        return recommendations