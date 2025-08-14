"""
Component Signal Aggregator

Aggregates and normalizes signals from Components 1-7 for integration processing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class ComponentAggregator:
    """
    Aggregates and normalizes component signals for master integration
    """
    
    # Expected signals from each component
    COMPONENT_SIGNALS = {
        'component_01': [
            'straddle_trend_score', 'vol_compression_score', 'breakout_probability',
            'dte_correlation_strength', 'regime_confidence'
        ],
        'component_02': [
            'gamma_exposure_score', 'sentiment_level', 'pin_risk_score',
            'delta_imbalance', 'vega_concentration'
        ],
        'component_03': [
            'institutional_flow_score', 'divergence_type', 'range_expansion_score',
            'oi_concentration', 'pa_momentum'
        ],
        'component_04': [
            'skew_bias_score', 'term_structure_signal', 'iv_regime_level',
            'volatility_smile_slope', 'skew_momentum'
        ],
        'component_05': [
            'momentum_score', 'volatility_regime_score', 'confluence_score',
            'atr_expansion_rate', 'cpr_alignment'
        ],
        'component_06': [
            'correlation_agreement_score', 'breakdown_alert', 'system_stability_score',
            'correlation_strength', 'feature_importance_mean'
        ],
        'component_07': [
            'level_strength_score', 'breakout_probability', 'confluence_score',
            'zone_reliability', 'pattern_confidence'
        ]
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the component aggregator
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Signal normalization ranges
        self.normalization_ranges = self.config.get('normalization_ranges', {
            'default': (-1.0, 1.0),
            'probability': (0.0, 1.0),
            'score': (-1.0, 1.0)
        })
        
        # Component health thresholds
        self.health_thresholds = self.config.get('health_thresholds', {
            'min_signals': 3,
            'max_nan_ratio': 0.3
        })
        
        logger.info("Initialized ComponentAggregator")
    
    def aggregate_components(
        self,
        component_outputs: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate and normalize component outputs
        
        Args:
            component_outputs: Raw outputs from Components 1-7
            
        Returns:
            Aggregated and normalized component signals with metadata
        """
        aggregated = {}
        
        for comp_name, expected_signals in self.COMPONENT_SIGNALS.items():
            if comp_name in component_outputs:
                comp_data = component_outputs[comp_name]
                
                # Extract and normalize signals
                normalized_signals = self._normalize_component(
                    comp_data,
                    expected_signals,
                    comp_name
                )
                
                # Add metadata
                metadata = self._extract_metadata(comp_data, comp_name)
                
                # Assess component health
                health_score = self._assess_component_health(
                    normalized_signals,
                    metadata
                )
                
                aggregated[comp_name] = {
                    'signals': normalized_signals,
                    'metadata': metadata,
                    'health_score': health_score,
                    'timestamp': comp_data.get('timestamp'),
                    'processing_time': comp_data.get('processing_time_ms', 0)
                }
            else:
                logger.warning(f"Component {comp_name} not found in outputs")
                aggregated[comp_name] = self._get_default_component_data(comp_name)
        
        return aggregated
    
    def _normalize_component(
        self,
        comp_data: Dict[str, Any],
        expected_signals: List[str],
        comp_name: str
    ) -> Dict[str, float]:
        """
        Normalize component signals to standard range
        
        Args:
            comp_data: Component output data
            expected_signals: List of expected signal names
            comp_name: Component name for logging
            
        Returns:
            Dictionary of normalized signals
        """
        normalized = {}
        
        for signal_name in expected_signals:
            if signal_name in comp_data:
                value = comp_data[signal_name]
                
                # Handle different value types
                if isinstance(value, (int, float)) and not np.isnan(value):
                    # Determine normalization range based on signal type
                    if 'probability' in signal_name:
                        norm_range = self.normalization_ranges['probability']
                    elif 'score' in signal_name:
                        norm_range = self.normalization_ranges['score']
                    else:
                        norm_range = self.normalization_ranges['default']
                    
                    # Normalize to range
                    normalized_value = self._normalize_value(value, norm_range)
                    normalized[signal_name] = normalized_value
                    
                elif isinstance(value, (int, float)) and np.isnan(value):
                    # Handle NaN values
                    normalized[signal_name] = 0.0
                    logger.debug(f"NaN value for {comp_name}.{signal_name}, using 0.0")
                    
                else:
                    # Handle non-numeric values
                    if isinstance(value, str):
                        # Convert categorical to numeric if possible
                        normalized[signal_name] = self._encode_categorical(value)
                    else:
                        normalized[signal_name] = 0.0
            else:
                # Signal not found, use default
                normalized[signal_name] = 0.0
                logger.debug(f"Signal {signal_name} not found in {comp_name}, using 0.0")
        
        return normalized
    
    def _normalize_value(
        self,
        value: float,
        norm_range: Tuple[float, float]
    ) -> float:
        """
        Normalize a value to specified range
        
        Args:
            value: Value to normalize
            norm_range: Target normalization range (min, max)
            
        Returns:
            Normalized value
        """
        min_val, max_val = norm_range
        
        # Clip to range
        return np.clip(value, min_val, max_val)
    
    def _encode_categorical(self, value: str) -> float:
        """
        Encode categorical values to numeric
        
        Args:
            value: Categorical string value
            
        Returns:
            Encoded numeric value
        """
        # Common categorical mappings
        categorical_map = {
            'bullish': 1.0,
            'bearish': -1.0,
            'neutral': 0.0,
            'high': 1.0,
            'medium': 0.0,
            'low': -1.0,
            'strong': 1.0,
            'weak': -1.0,
            'none': 0.0
        }
        
        return categorical_map.get(value.lower(), 0.0)
    
    def _extract_metadata(
        self,
        comp_data: Dict[str, Any],
        comp_name: str
    ) -> Dict[str, Any]:
        """
        Extract metadata from component output
        
        Args:
            comp_data: Component output data
            comp_name: Component name
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            'component_name': comp_name,
            'signal_count': len([k for k in comp_data.keys() if not k.startswith('_')]),
            'has_confidence': 'confidence' in comp_data or 'confidence_score' in comp_data,
            'has_processing_time': 'processing_time_ms' in comp_data,
            'data_quality': self._assess_data_quality(comp_data)
        }
        
        # Extract confidence if available
        if 'confidence' in comp_data:
            metadata['confidence'] = comp_data['confidence']
        elif 'confidence_score' in comp_data:
            metadata['confidence'] = comp_data['confidence_score']
        else:
            metadata['confidence'] = 0.5  # Default medium confidence
        
        return metadata
    
    def _assess_data_quality(self, comp_data: Dict[str, Any]) -> float:
        """
        Assess quality of component data
        
        Args:
            comp_data: Component output data
            
        Returns:
            Data quality score (0-1)
        """
        numeric_values = []
        
        for key, value in comp_data.items():
            if not key.startswith('_') and isinstance(value, (int, float)):
                numeric_values.append(value)
        
        if not numeric_values:
            return 0.0
        
        # Calculate quality metrics
        nan_ratio = sum(1 for v in numeric_values if np.isnan(v)) / len(numeric_values)
        inf_ratio = sum(1 for v in numeric_values if np.isinf(v)) / len(numeric_values)
        
        # Quality score based on data validity
        quality = 1.0 - (nan_ratio + inf_ratio)
        
        return max(0.0, min(1.0, quality))
    
    def _assess_component_health(
        self,
        signals: Dict[str, float],
        metadata: Dict[str, Any]
    ) -> float:
        """
        Assess health of component based on signals and metadata
        
        Args:
            signals: Normalized component signals
            metadata: Component metadata
            
        Returns:
            Health score (0-1)
        """
        health_factors = []
        
        # Factor 1: Signal completeness
        signal_completeness = len(signals) / max(1, len(self.COMPONENT_SIGNALS.get(
            metadata['component_name'], [])))
        health_factors.append(signal_completeness)
        
        # Factor 2: Data quality
        health_factors.append(metadata['data_quality'])
        
        # Factor 3: Confidence score
        health_factors.append(metadata.get('confidence', 0.5))
        
        # Factor 4: Signal validity (no extreme values)
        valid_signals = sum(1 for v in signals.values() if -1.0 <= v <= 1.0)
        signal_validity = valid_signals / max(1, len(signals))
        health_factors.append(signal_validity)
        
        # Overall health score
        health_score = np.mean(health_factors)
        
        return float(health_score)
    
    def _get_default_component_data(self, comp_name: str) -> Dict[str, Any]:
        """
        Get default component data when component is missing
        
        Args:
            comp_name: Component name
            
        Returns:
            Default component data structure
        """
        default_signals = {
            signal: 0.0 for signal in self.COMPONENT_SIGNALS.get(comp_name, [])
        }
        
        return {
            'signals': default_signals,
            'metadata': {
                'component_name': comp_name,
                'signal_count': 0,
                'has_confidence': False,
                'has_processing_time': False,
                'data_quality': 0.0,
                'confidence': 0.5
            },
            'health_score': 0.0,
            'timestamp': None,
            'processing_time': 0
        }
    
    def get_aggregation_summary(
        self,
        aggregated_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Get summary statistics of aggregated components
        
        Args:
            aggregated_data: Aggregated component data
            
        Returns:
            Summary statistics
        """
        summary = {
            'total_components': len(aggregated_data),
            'healthy_components': sum(
                1 for comp in aggregated_data.values()
                if comp['health_score'] > 0.7
            ),
            'average_health': np.mean([
                comp['health_score'] for comp in aggregated_data.values()
            ]),
            'total_signals': sum(
                len(comp['signals']) for comp in aggregated_data.values()
            ),
            'average_confidence': np.mean([
                comp['metadata'].get('confidence', 0.5)
                for comp in aggregated_data.values()
            ]),
            'total_processing_time': sum(
                comp.get('processing_time', 0)
                for comp in aggregated_data.values()
            )
        }
        
        return summary