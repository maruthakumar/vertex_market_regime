"""
Entity Relationship Documentation for Market Regime Feature Store
Story 2.6: Minimal Online Feature Registration - Subtask 1.5

Documents:
- Entity relationship mapping between BigQuery and Feature Store
- Access patterns for online feature serving
- Data lineage and feature dependencies
- Performance optimization patterns
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class EntityRelationshipDocumenter:
    """
    Documents entity relationships and access patterns for Feature Store.
    
    Provides comprehensive documentation for:
    - Entity type relationships
    - Feature dependencies across components
    - Access patterns for different use cases
    - Performance optimization strategies
    """
    
    def __init__(self):
        """Initialize Entity Relationship Documenter"""
        logger.info("Entity Relationship Documenter initialized")
    
    def get_entity_relationship_mapping(self) -> Dict[str, Any]:
        """
        Get comprehensive entity relationship mapping.
        
        Returns:
            Dict[str, Any]: Complete entity relationship documentation
        """
        return {
            'entity_types': self._get_entity_type_documentation(),
            'feature_relationships': self._get_feature_relationship_documentation(),
            'access_patterns': self._get_access_pattern_documentation(),
            'data_lineage': self._get_data_lineage_documentation(),
            'performance_patterns': self._get_performance_pattern_documentation(),
            'generated_at': datetime.now().isoformat()
        }
    
    def _get_entity_type_documentation(self) -> Dict[str, Any]:
        """Document entity type structure and relationships"""
        return {
            'primary_entity_type': {
                'name': 'instrument_minute',
                'description': 'Market regime features at minute-level granularity',
                'entity_id_format': '${symbol}_${yyyymmddHHMM}_${dte}',
                'entity_id_examples': [
                    'NIFTY_202508141430_7',
                    'BANKNIFTY_202508141500_14', 
                    'FINNIFTY_202508141515_21'
                ],
                'entity_id_components': {
                    'symbol': {
                        'description': 'Market instrument symbol',
                        'valid_values': ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'SENSEX'],
                        'format': 'Uppercase alphabetic string'
                    },
                    'timestamp': {
                        'description': 'Minute-level timestamp in IST',
                        'format': 'yyyymmddHHMM',
                        'timezone': 'Asia/Kolkata',
                        'trading_hours': '09:15 - 15:30'
                    },
                    'dte': {
                        'description': 'Days to expiry for options',
                        'range': '0-45 days',
                        'common_values': [0, 3, 7, 14, 21, 30]
                    }
                }
            },
            'entity_relationships': {
                'temporal_relationship': {
                    'description': 'Time-series relationship across minutes',
                    'granularity': 'minute-level',
                    'aggregation_support': 'daily, hourly, custom intervals'
                },
                'symbol_relationship': {
                    'description': 'Cross-symbol correlation analysis',
                    'supported_correlations': 'NIFTY-BANKNIFTY, NIFTY-FINNIFTY'
                },
                'dte_relationship': {
                    'description': 'Cross-DTE feature dependencies',
                    'dependencies': 'momentum transfer, volatility surface'
                }
            }
        }
    
    def _get_feature_relationship_documentation(self) -> Dict[str, Any]:
        """Document feature dependencies and relationships"""
        return {
            'component_features': {
                'c1_triple_straddle': {
                    'primary_features': [
                        'c1_momentum_score',
                        'c1_vol_compression', 
                        'c1_breakout_probability',
                        'c1_transition_probability'
                    ],
                    'dependencies': ['underlying_price', 'straddle_prices', 'volume_data'],
                    'relationships': {
                        'internal': 'momentum_score influences transition_probability',
                        'external': 'correlates with c5_momentum_score, c8_component_agreement_score'
                    }
                },
                'c2_greeks_sentiment': {
                    'primary_features': [
                        'c2_gamma_exposure',
                        'c2_sentiment_level',
                        'c2_pin_risk_score',
                        'c2_max_pain_level'
                    ],
                    'dependencies': ['option_prices', 'greeks_calculations', 'volume_analysis'],
                    'relationships': {
                        'internal': 'gamma_exposure drives pin_risk_score',
                        'external': 'correlates with c4_iv_regime_level, c8_integration_confidence'
                    }
                },
                'c3_oi_pa_trending': {
                    'primary_features': [
                        'c3_institutional_flow_score',
                        'c3_divergence_type',
                        'c3_range_expansion_score',
                        'c3_volume_profile'
                    ],
                    'dependencies': ['open_interest', 'volume_data', 'price_action'],
                    'relationships': {
                        'internal': 'institutional_flow influences range_expansion',
                        'external': 'correlates with c1_breakout_probability, c7_breakout_probability'
                    }
                },
                'c4_iv_skew': {
                    'primary_features': [
                        'c4_skew_bias_score',
                        'c4_term_structure_signal',
                        'c4_iv_regime_level',
                        'c4_volatility_rank'
                    ],
                    'dependencies': ['implied_volatility', 'historical_volatility', 'skew_data'],
                    'relationships': {
                        'internal': 'iv_regime_level affects skew_bias_score',
                        'external': 'correlates with c2_sentiment_level, c5_volatility_regime_score'
                    }
                },
                'c5_atr_ema_cpr': {
                    'primary_features': [
                        'c5_momentum_score',
                        'c5_volatility_regime_score',
                        'c5_confluence_score',
                        'c5_trend_strength'
                    ],
                    'dependencies': ['atr_values', 'ema_calculations', 'cpr_levels'],
                    'relationships': {
                        'internal': 'trend_strength influences momentum_score',
                        'external': 'correlates with c1_momentum_score, c7_level_strength_score'
                    }
                },
                'c6_correlation': {
                    'primary_features': [
                        'c6_correlation_agreement_score',
                        'c6_breakdown_alert',
                        'c6_system_stability_score',
                        'c6_prediction_confidence'
                    ],
                    'dependencies': ['all_component_outputs', 'correlation_matrices'],
                    'relationships': {
                        'internal': 'correlation_agreement drives system_stability',
                        'external': 'meta-level analysis of all components'
                    }
                },
                'c7_support_resistance': {
                    'primary_features': [
                        'c7_level_strength_score',
                        'c7_breakout_probability',
                        'c7_support_confluence',
                        'c7_resistance_confluence'
                    ],
                    'dependencies': ['price_levels', 'volume_confirmation', 'historical_levels'],
                    'relationships': {
                        'internal': 'confluence_scores influence breakout_probability',
                        'external': 'correlates with c3_range_expansion_score, c1_breakout_probability'
                    }
                },
                'c8_master_integration': {
                    'primary_features': [
                        'c8_component_agreement_score',
                        'c8_integration_confidence',
                        'c8_transition_probability_hint',
                        'c8_regime_classification'
                    ],
                    'dependencies': ['all_component_scores', 'confidence_metrics'],
                    'relationships': {
                        'internal': 'component_agreement drives integration_confidence',
                        'external': 'final output combining all component insights'
                    }
                }
            },
            'cross_component_relationships': {
                'momentum_correlation': {
                    'components': ['c1', 'c5'],
                    'features': ['c1_momentum_score', 'c5_momentum_score'],
                    'relationship_type': 'reinforcing'
                },
                'breakout_consensus': {
                    'components': ['c1', 'c3', 'c7'],
                    'features': ['c1_breakout_probability', 'c3_range_expansion_score', 'c7_breakout_probability'],
                    'relationship_type': 'consensus_building'
                },
                'volatility_regime': {
                    'components': ['c2', 'c4', 'c5'],
                    'features': ['c2_gamma_exposure', 'c4_iv_regime_level', 'c5_volatility_regime_score'],
                    'relationship_type': 'regime_definition'
                },
                'system_confidence': {
                    'components': ['c6', 'c8'],
                    'features': ['c6_system_stability_score', 'c8_integration_confidence'],
                    'relationship_type': 'meta_analysis'
                }
            }
        }
    
    def _get_access_pattern_documentation(self) -> Dict[str, Any]:
        """Document feature access patterns for different use cases"""
        return {
            'real_time_serving': {
                'description': 'Low-latency feature serving for real-time inference',
                'latency_target': '<50ms (p99)',
                'access_patterns': [
                    {
                        'name': 'single_entity_lookup',
                        'description': 'Get all features for specific entity (symbol+timestamp+dte)',
                        'example_entity_id': 'NIFTY_202508141430_7',
                        'expected_features': 32,
                        'latency_target': '<25ms'
                    },
                    {
                        'name': 'multi_entity_batch',
                        'description': 'Get features for multiple entities (different DTEs)',
                        'example_entity_ids': [
                            'NIFTY_202508141430_7',
                            'NIFTY_202508141430_14',
                            'NIFTY_202508141430_21'
                        ],
                        'expected_features': 96,
                        'latency_target': '<40ms'
                    },
                    {
                        'name': 'component_specific',
                        'description': 'Get features for specific component only',
                        'example_features': ['c1_momentum_score', 'c1_breakout_probability'],
                        'latency_target': '<15ms'
                    }
                ]
            },
            'batch_inference': {
                'description': 'Batch processing for historical analysis',
                'throughput_target': '>1000 RPS',
                'access_patterns': [
                    {
                        'name': 'time_range_query',
                        'description': 'Get features for symbol across time range',
                        'example_query': 'NIFTY features from 09:15 to 15:30',
                        'expected_entities': 375,  # 6.25 hours * 60 minutes
                        'throughput_target': '>500 entities/second'
                    },
                    {
                        'name': 'cross_symbol_analysis',
                        'description': 'Compare features across symbols at same time',
                        'example_symbols': ['NIFTY', 'BANKNIFTY', 'FINNIFTY'],
                        'throughput_target': '>1000 entities/second'
                    }
                ]
            },
            'model_training': {
                'description': 'Feature serving for model training pipelines',
                'consistency_requirement': 'offline-online feature consistency',
                'access_patterns': [
                    {
                        'name': 'training_dataset_generation',
                        'description': 'Generate training datasets with point-in-time correctness',
                        'time_range': 'Historical data with proper temporal alignment',
                        'feature_freshness': 'Must match offline BigQuery features'
                    },
                    {
                        'name': 'feature_validation',
                        'description': 'Validate feature quality and distribution',
                        'checks': ['null_values', 'outliers', 'distribution_drift'],
                        'frequency': 'Daily validation runs'
                    }
                ]
            }
        }
    
    def _get_data_lineage_documentation(self) -> Dict[str, Any]:
        """Document data lineage and feature dependencies"""
        return {
            'data_sources': {
                'primary_source': {
                    'type': 'BigQuery offline tables',
                    'tables': ['c1_features', 'c2_features', 'c3_features', 'c4_features',
                              'c5_features', 'c6_features', 'c7_features', 'c8_features'],
                    'update_frequency': 'minute-level streaming + daily batch'
                },
                'source_dependencies': {
                    'market_data': 'Real-time market data feeds',
                    'options_data': 'Options pricing and greeks data',
                    'calculated_features': 'Component-specific feature calculations'
                }
            },
            'transformation_pipeline': {
                'steps': [
                    {
                        'step': 1,
                        'name': 'data_ingestion',
                        'description': 'Ingest raw market data into BigQuery',
                        'frequency': 'Real-time streaming'
                    },
                    {
                        'step': 2,
                        'name': 'feature_calculation',
                        'description': 'Calculate component features using Epic 1 algorithms',
                        'frequency': 'Minute-level batch processing'
                    },
                    {
                        'step': 3,
                        'name': 'feature_validation',
                        'description': 'Validate feature quality and consistency',
                        'frequency': 'Per batch'
                    },
                    {
                        'step': 4,
                        'name': 'online_ingestion',
                        'description': 'Ingest validated features into Feature Store',
                        'frequency': 'Near real-time (within 30 seconds)'
                    }
                ]
            },
            'feature_lineage': {
                'upstream_dependencies': {
                    'market_regime_components': 'Epic 1 component implementations',
                    'feature_calculations': 'Component-specific analyzers',
                    'data_quality_checks': 'Validation and cleansing rules'
                },
                'downstream_consumers': {
                    'real_time_inference': 'Epic 3 serving endpoints',
                    'model_training': 'Epic 4 training pipelines',
                    'monitoring_systems': 'Feature drift and quality monitoring'
                }
            }
        }
    
    def _get_performance_pattern_documentation(self) -> Dict[str, Any]:
        """Document performance optimization patterns"""
        return {
            'caching_strategies': {
                'feature_vector_caching': {
                    'description': 'Cache complete feature vectors for frequently accessed entities',
                    'ttl': '60 seconds',
                    'cache_size': '1GB',
                    'hit_ratio_target': '>80%'
                },
                'component_level_caching': {
                    'description': 'Cache individual component features separately',
                    'ttl': '30 seconds',
                    'use_case': 'Component-specific queries'
                }
            },
            'optimization_patterns': {
                'batch_serving': {
                    'description': 'Batch multiple feature requests for efficiency',
                    'batch_size': '100 entities',
                    'latency_improvement': '40% reduction vs individual requests'
                },
                'connection_pooling': {
                    'description': 'Maintain persistent connections to Feature Store',
                    'pool_size': '10 connections',
                    'connection_reuse': 'Reduces connection overhead by 60%'
                },
                'feature_compression': {
                    'description': 'Compress feature vectors for network efficiency',
                    'compression_type': 'snappy',
                    'size_reduction': '30% average'
                }
            },
            'monitoring_patterns': {
                'latency_monitoring': {
                    'metrics': ['p50', 'p95', 'p99'],
                    'alerting_thresholds': {
                        'p99': '55ms',
                        'p95': '45ms',
                        'p50': '30ms'
                    }
                },
                'throughput_monitoring': {
                    'metrics': ['requests_per_second', 'features_per_second'],
                    'targets': {
                        'rps': 1000,
                        'features_per_second': 32000
                    }
                },
                'quality_monitoring': {
                    'metrics': ['null_ratio', 'outlier_detection', 'distribution_drift'],
                    'frequency': 'Continuous monitoring with hourly reports'
                }
            }
        }
    
    def export_documentation(self, output_path: str) -> bool:
        """
        Export entity relationship documentation to file.
        
        Args:
            output_path: Path to save documentation
            
        Returns:
            bool: True if successful
        """
        try:
            documentation = self.get_entity_relationship_mapping()
            
            with open(output_path, 'w') as f:
                json.dump(documentation, f, indent=2, default=str)
            
            logger.info(f"Entity relationship documentation exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export documentation: {e}")
            return False
    
    def get_access_pattern_examples(self) -> Dict[str, str]:
        """Get code examples for common access patterns"""
        return {
            'single_entity_lookup': '''
# Single entity feature lookup
from vertex_market_regime.features.feature_store_client import FeatureStoreClient

client = FeatureStoreClient()
entity_id = "NIFTY_202508141430_7"
features = client.get_online_features(entity_id)
''',
            'batch_entity_lookup': '''
# Batch entity feature lookup
entity_ids = [
    "NIFTY_202508141430_7",
    "NIFTY_202508141430_14", 
    "NIFTY_202508141430_21"
]
features_batch = client.get_online_features_batch(entity_ids)
''',
            'component_specific_lookup': '''
# Component-specific feature lookup
component_features = client.get_component_features(
    entity_id="NIFTY_202508141430_7",
    component="c1"
)
''',
            'time_range_query': '''
# Time range feature query
from datetime import datetime
features_range = client.get_features_for_time_range(
    symbol="NIFTY",
    start_time=datetime(2025, 8, 14, 9, 15),
    end_time=datetime(2025, 8, 14, 15, 30),
    dte=7
)
'''
        }
    
    def get_entity_relationship_summary(self) -> Dict[str, Any]:
        """Get summary of entity relationships for quick reference"""
        return {
            'entity_format': '${symbol}_${yyyymmddHHMM}_${dte}',
            'total_features': 32,
            'components': 8,
            'features_per_component': 4,
            'ttl_hours': 48,
            'latency_target': '<50ms',
            'throughput_target': '>1000 RPS',
            'update_frequency': 'minute-level',
            'supported_symbols': ['NIFTY', 'BANKNIFTY', 'FINNIFTY'],
            'dte_range': '0-45 days',
            'trading_hours': '09:15-15:30 IST'
        }