"""
Regime Definition Builder

This module converts clustering results and analysis into concrete regime
definitions with boundaries, characteristics, and operational parameters.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
from dataclasses import dataclass, field
import json
from scipy import stats
from sklearn.mixture import GaussianMixture

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class RegimeBoundary:
    """Defines boundaries for a regime"""
    volatility_bounds: Tuple[float, float]
    trend_bounds: Tuple[float, float]
    volume_bounds: Tuple[float, float]
    confidence_threshold: float
    
    def contains(self, volatility: float, trend: float, volume: float) -> bool:
        """Check if given values fall within regime boundaries"""
        return (self.volatility_bounds[0] <= volatility <= self.volatility_bounds[1] and
                self.trend_bounds[0] <= trend <= self.trend_bounds[1] and
                self.volume_bounds[0] <= volume <= self.volume_bounds[1])


@dataclass
class RegimeDefinition:
    """Complete definition of a market regime"""
    regime_id: int
    name: str
    description: str
    boundaries: RegimeBoundary
    characteristic_features: List[str]
    entry_conditions: Dict[str, Any]
    exit_conditions: Dict[str, Any]
    strategy_preferences: List[str]
    risk_parameters: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class RegimeDefinitionBuilder:
    """
    Builds comprehensive regime definitions from analysis results
    """
    
    # Regime naming templates based on count
    REGIME_NAMES = {
        8: [
            "Strong Bullish Trend",
            "Moderate Bullish",
            "Weak Bullish",
            "Neutral Range",
            "Volatile Neutral",
            "Weak Bearish",
            "Moderate Bearish",
            "Strong Bearish Trend"
        ],
        12: [
            "Low Vol Bullish Trending",
            "Low Vol Bullish Range",
            "Low Vol Neutral Trending",
            "Low Vol Neutral Range",
            "Med Vol Bullish Trending",
            "Med Vol Bullish Range",
            "Med Vol Neutral Trending",
            "Med Vol Neutral Range",
            "High Vol Bullish Trending",
            "High Vol Bullish Range",
            "High Vol Bearish Trending",
            "High Vol Bearish Range"
        ],
        18: [
            "Low Vol Strong Bullish",
            "Low Vol Mild Bullish",
            "Low Vol Neutral",
            "Low Vol Sideways",
            "Low Vol Mild Bearish",
            "Low Vol Strong Bearish",
            "Normal Vol Strong Bullish",
            "Normal Vol Mild Bullish",
            "Normal Vol Neutral",
            "Normal Vol Sideways",
            "Normal Vol Mild Bearish",
            "Normal Vol Strong Bearish",
            "High Vol Strong Bullish",
            "High Vol Mild Bullish",
            "High Vol Neutral",
            "High Vol Sideways",
            "High Vol Mild Bearish",
            "High Vol Strong Bearish"
        ]
    }
    
    def __init__(self, regime_count: int):
        """
        Initialize regime definition builder
        
        Args:
            regime_count: Number of regimes (8, 12, or 18)
        """
        if regime_count not in [8, 12, 18]:
            raise ValueError(f"Invalid regime count: {regime_count}. Must be 8, 12, or 18")
        
        self.regime_count = regime_count
        self.regime_definitions: Dict[int, RegimeDefinition] = {}
        
        logger.info(f"RegimeDefinitionBuilder initialized for {regime_count} regimes")
    
    def build_regime_definitions(self, analysis_results: Dict[str, Any]) -> Dict[int, RegimeDefinition]:
        """
        Convert analysis results to regime definitions
        
        Args:
            analysis_results: Results from historical analysis
            
        Returns:
            Dictionary of regime definitions
        """
        logger.info("Building regime definitions from analysis results")
        
        try:
            regime_patterns = analysis_results.get('regime_patterns', {})
            transition_matrix = analysis_results.get('transition_matrix', pd.DataFrame())
            stability_metrics = analysis_results.get('stability_metrics', {})
            
            # Build definition for each regime
            for regime_id in range(self.regime_count):
                if regime_id in regime_patterns:
                    pattern = regime_patterns[regime_id]
                    definition = self._build_single_definition(
                        regime_id, pattern, transition_matrix, stability_metrics
                    )
                else:
                    # Create default definition if pattern missing
                    definition = self._create_default_definition(regime_id)
                
                self.regime_definitions[regime_id] = definition
            
            # Optimize boundaries to minimize overlap
            self._optimize_boundaries()
            
            # Validate definitions
            self._validate_definitions()
            
            logger.info(f"Built {len(self.regime_definitions)} regime definitions")
            
            return self.regime_definitions
            
        except Exception as e:
            logger.error(f"Error building regime definitions: {e}")
            raise
    
    def _build_single_definition(self, regime_id: int, pattern: Any,
                               transition_matrix: pd.DataFrame,
                               stability_metrics: Dict[str, float]) -> RegimeDefinition:
        """
        Build definition for a single regime
        
        Args:
            regime_id: Regime identifier
            pattern: Regime pattern from analysis
            transition_matrix: Transition probabilities
            stability_metrics: Overall stability metrics
            
        Returns:
            Complete regime definition
        """
        # Get regime name
        regime_names = self.REGIME_NAMES[self.regime_count]
        name = regime_names[regime_id] if regime_id < len(regime_names) else f"Regime {regime_id}"
        
        # Create boundaries with safety margins
        boundaries = RegimeBoundary(
            volatility_bounds=self._expand_bounds(pattern.volatility_range, 0.1),
            trend_bounds=self._expand_bounds(pattern.trend_range, 0.1),
            volume_bounds=(
                pattern.volume_profile['mean'] - 2 * pattern.volume_profile['std'],
                pattern.volume_profile['mean'] + 2 * pattern.volume_profile['std']
            ),
            confidence_threshold=0.6 + pattern.stability_score * 0.2
        )
        
        # Define entry conditions
        entry_conditions = self._define_entry_conditions(
            regime_id, pattern, transition_matrix
        )
        
        # Define exit conditions
        exit_conditions = self._define_exit_conditions(
            regime_id, pattern, transition_matrix
        )
        
        # Determine strategy preferences based on characteristics
        strategy_preferences = self._determine_strategy_preferences(
            pattern.characteristic_features
        )
        
        # Set risk parameters
        risk_parameters = self._calculate_risk_parameters(pattern)
        
        # Generate description
        description = self._generate_description(name, pattern)
        
        # Create metadata
        metadata = {
            'average_duration': pattern.average_duration,
            'stability_score': pattern.stability_score,
            'transition_probabilities': pattern.transition_probabilities,
            'creation_timestamp': datetime.now().isoformat()
        }
        
        return RegimeDefinition(
            regime_id=regime_id,
            name=name,
            description=description,
            boundaries=boundaries,
            characteristic_features=pattern.characteristic_features,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            strategy_preferences=strategy_preferences,
            risk_parameters=risk_parameters,
            metadata=metadata
        )
    
    def _expand_bounds(self, bounds: Tuple[float, float], margin: float) -> Tuple[float, float]:
        """
        Expand bounds by given margin
        
        Args:
            bounds: Original bounds
            margin: Expansion margin (0-1)
            
        Returns:
            Expanded bounds
        """
        range_size = bounds[1] - bounds[0]
        expansion = range_size * margin
        return (bounds[0] - expansion, bounds[1] + expansion)
    
    def _define_entry_conditions(self, regime_id: int, pattern: Any,
                               transition_matrix: pd.DataFrame) -> Dict[str, Any]:
        """
        Define conditions for entering a regime
        
        Args:
            regime_id: Regime identifier
            pattern: Regime pattern
            transition_matrix: Transition probabilities
            
        Returns:
            Entry conditions dictionary
        """
        entry_conditions = {
            'min_confidence': 0.6,
            'min_duration': 5,  # minutes
            'confirmation_required': True
        }
        
        # Add conditions based on most likely previous regimes
        if not transition_matrix.empty and regime_id in transition_matrix.columns:
            # Find regimes that commonly transition to this one
            transition_probs = transition_matrix[regime_id]
            likely_sources = transition_probs[transition_probs > 0.1].index.tolist()
            
            entry_conditions['likely_from_regimes'] = likely_sources
            entry_conditions['transition_threshold'] = 0.1
        
        # Add feature-based conditions
        if 'high_volatility' in pattern.characteristic_features:
            entry_conditions['volatility_increase'] = 0.2  # 20% increase required
        
        if 'bullish_bias' in pattern.characteristic_features:
            entry_conditions['trend_positive'] = True
            entry_conditions['momentum_threshold'] = 0.0
        elif 'bearish_bias' in pattern.characteristic_features:
            entry_conditions['trend_negative'] = True
            entry_conditions['momentum_threshold'] = 0.0
        
        return entry_conditions
    
    def _define_exit_conditions(self, regime_id: int, pattern: Any,
                              transition_matrix: pd.DataFrame) -> Dict[str, Any]:
        """
        Define conditions for exiting a regime
        
        Args:
            regime_id: Regime identifier
            pattern: Regime pattern
            transition_matrix: Transition probabilities
            
        Returns:
            Exit conditions dictionary
        """
        exit_conditions = {
            'boundary_breach': True,  # Exit if boundaries breached
            'min_duration_met': True,  # Must meet minimum duration first
            'confirmation_required': True
        }
        
        # Add conditions based on likely next regimes
        if not transition_matrix.empty and regime_id in transition_matrix.index:
            # Find regimes this commonly transitions to
            transition_probs = transition_matrix.loc[regime_id]
            likely_targets = transition_probs[transition_probs > 0.1].index.tolist()
            
            exit_conditions['likely_to_regimes'] = likely_targets
            exit_conditions['transition_threshold'] = 0.1
        
        # Add stability-based conditions
        if pattern.stability_score < 0.5:
            exit_conditions['quick_exit_allowed'] = True
            exit_conditions['min_duration_override'] = 3  # Can exit after 3 minutes if unstable
        
        return exit_conditions
    
    def _determine_strategy_preferences(self, features: List[str]) -> List[str]:
        """
        Determine preferred strategies based on regime features
        
        Args:
            features: Characteristic features of regime
            
        Returns:
            List of preferred strategy types
        """
        preferences = []
        
        # Map features to strategies
        if 'low_volatility' in features:
            preferences.extend(['TBS', 'ML_INDICATOR'])
        elif 'high_volatility' in features:
            preferences.extend(['MARKET_REGIME', 'POS'])
        
        if 'bullish_bias' in features or 'bearish_bias' in features:
            preferences.extend(['TBS', 'TV'])
        
        if 'near_support' in features or 'near_resistance' in features:
            preferences.extend(['ORB', 'OI'])
        
        if 'high_volume' in features:
            preferences.append('OI')
        
        # Default if no specific preferences
        if not preferences:
            preferences = ['MARKET_REGIME', 'ML_INDICATOR']
        
        # Remove duplicates while preserving order
        seen = set()
        preferences = [x for x in preferences if not (x in seen or seen.add(x))]
        
        return preferences
    
    def _calculate_risk_parameters(self, pattern: Any) -> Dict[str, float]:
        """
        Calculate risk parameters for regime
        
        Args:
            pattern: Regime pattern
            
        Returns:
            Risk parameters dictionary
        """
        # Base risk parameters
        risk_params = {
            'position_size_multiplier': 1.0,
            'stop_loss_multiplier': 1.0,
            'take_profit_multiplier': 1.0,
            'max_positions': 3
        }
        
        # Adjust based on volatility
        avg_volatility = np.mean(pattern.volatility_range)
        if avg_volatility < 0.15:
            # Low volatility - can be more aggressive
            risk_params['position_size_multiplier'] = 1.5
            risk_params['stop_loss_multiplier'] = 0.8
            risk_params['max_positions'] = 5
        elif avg_volatility > 0.35:
            # High volatility - be conservative
            risk_params['position_size_multiplier'] = 0.5
            risk_params['stop_loss_multiplier'] = 1.5
            risk_params['max_positions'] = 2
        
        # Adjust based on stability
        if pattern.stability_score > 0.7:
            risk_params['position_size_multiplier'] *= 1.2
        elif pattern.stability_score < 0.3:
            risk_params['position_size_multiplier'] *= 0.8
        
        return risk_params
    
    def _generate_description(self, name: str, pattern: Any) -> str:
        """
        Generate human-readable description
        
        Args:
            name: Regime name
            pattern: Regime pattern
            
        Returns:
            Description string
        """
        features = pattern.characteristic_features
        
        desc_parts = [f"{name} regime"]
        
        if 'low_volatility' in features:
            desc_parts.append("with stable price action")
        elif 'high_volatility' in features:
            desc_parts.append("with volatile price swings")
        
        if 'bullish_bias' in features:
            desc_parts.append("and upward momentum")
        elif 'bearish_bias' in features:
            desc_parts.append("and downward pressure")
        
        if pattern.average_duration > 60:
            desc_parts.append("that typically persists for extended periods")
        elif pattern.average_duration < 20:
            desc_parts.append("that tends to be short-lived")
        
        return " ".join(desc_parts)
    
    def _create_default_definition(self, regime_id: int) -> RegimeDefinition:
        """
        Create default definition when pattern is missing
        
        Args:
            regime_id: Regime identifier
            
        Returns:
            Default regime definition
        """
        regime_names = self.REGIME_NAMES[self.regime_count]
        name = regime_names[regime_id] if regime_id < len(regime_names) else f"Regime {regime_id}"
        
        # Default boundaries based on regime position
        volatility_position = (regime_id % 3) / 2  # 0, 0.5, or 1
        trend_position = (regime_id // 3 - 1.5) / 1.5  # -1 to 1
        
        boundaries = RegimeBoundary(
            volatility_bounds=(volatility_position * 0.3, (volatility_position + 0.5) * 0.3),
            trend_bounds=(trend_position * 0.01 - 0.005, trend_position * 0.01 + 0.005),
            volume_bounds=(0.5, 2.0),  # Relative volume
            confidence_threshold=0.6
        )
        
        return RegimeDefinition(
            regime_id=regime_id,
            name=name,
            description=f"Default definition for {name}",
            boundaries=boundaries,
            characteristic_features=[],
            entry_conditions={'min_confidence': 0.6},
            exit_conditions={'boundary_breach': True},
            strategy_preferences=['MARKET_REGIME'],
            risk_parameters={'position_size_multiplier': 1.0},
            metadata={'is_default': True}
        )
    
    def _optimize_boundaries(self):
        """
        Optimize regime boundaries to minimize overlap
        """
        if self.regime_count <= 8:
            # For smaller regime counts, ensure clear separation
            self._enforce_boundary_separation()
        else:
            # For larger counts, use GMM for smoother boundaries
            self._smooth_boundaries_gmm()
    
    def _enforce_boundary_separation(self):
        """
        Enforce minimum separation between regime boundaries
        """
        # Sort regimes by volatility center
        sorted_regimes = sorted(
            self.regime_definitions.items(),
            key=lambda x: np.mean(x[1].boundaries.volatility_bounds)
        )
        
        # Ensure no overlap in volatility dimension
        for i in range(1, len(sorted_regimes)):
            prev_id, prev_def = sorted_regimes[i-1]
            curr_id, curr_def = sorted_regimes[i]
            
            if prev_def.boundaries.volatility_bounds[1] > curr_def.boundaries.volatility_bounds[0]:
                # Overlap detected, adjust boundaries
                midpoint = (prev_def.boundaries.volatility_bounds[1] + 
                           curr_def.boundaries.volatility_bounds[0]) / 2
                
                prev_def.boundaries.volatility_bounds = (
                    prev_def.boundaries.volatility_bounds[0],
                    midpoint - 0.01
                )
                curr_def.boundaries.volatility_bounds = (
                    midpoint + 0.01,
                    curr_def.boundaries.volatility_bounds[1]
                )
    
    def _smooth_boundaries_gmm(self):
        """
        Use Gaussian Mixture Model for smoother boundaries
        """
        try:
            # Prepare feature data for GMM
            features = []
            
            # Collect volatility and trend features from regime definitions
            for regime_id, regime_def in self.regime_definitions.items():
                vol_bounds = regime_def.boundaries.volatility_bounds
                trend_bounds = regime_def.boundaries.trend_bounds
                
                # Create feature vector [vol_low, vol_high, trend_low, trend_high]
                feature = [
                    vol_bounds[0],  # volatility lower bound
                    vol_bounds[1],  # volatility upper bound
                    trend_bounds[0],  # trend lower bound
                    trend_bounds[1]   # trend upper bound
                ]
                features.append(feature)
            
            if len(features) < 3:
                logger.warning("Insufficient regime definitions for GMM smoothing")
                return
            
            features = np.array(features)
            
            # Fit GMM with optimal number of components
            n_components = min(len(features) // 3, 6)  # At least 3 samples per component
            
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type='full',
                random_state=42,
                n_init=5,
                init_params='k-means++'
            )
            
            # Fit the model
            gmm.fit(features)
            
            # Generate smoothed boundaries using GMM predictions
            # Sample from the fitted distribution
            smoothed_samples, _ = gmm.sample(n_samples=len(features) * 2)
            
            # Calculate smoothed boundaries as weighted average
            for i, (regime_id, regime_def) in enumerate(self.regime_definitions.items()):
                if i < len(features):
                    # Get GMM prediction for this regime
                    probs = gmm.predict_proba(features[i].reshape(1, -1))[0]
                    
                    # Weight smoothed samples by their probability
                    smoothed_boundary = np.zeros(4)
                    for j in range(n_components):
                        component_mean = gmm.means_[j]
                        smoothed_boundary += probs[j] * component_mean
                    
                    # Apply smoothing with decay factor
                    alpha = 0.3  # Smoothing factor
                    
                    # Update volatility boundaries
                    current_vol = regime_def.boundaries.volatility_bounds
                    regime_def.boundaries.volatility_bounds = (
                        (1 - alpha) * current_vol[0] + alpha * smoothed_boundary[0],
                        (1 - alpha) * current_vol[1] + alpha * smoothed_boundary[1]
                    )
                    
                    # Update trend boundaries
                    current_trend = regime_def.boundaries.trend_bounds
                    regime_def.boundaries.trend_bounds = (
                        (1 - alpha) * current_trend[0] + alpha * smoothed_boundary[2],
                        (1 - alpha) * current_trend[1] + alpha * smoothed_boundary[3]
                    )
                    
                    # Ensure boundaries are valid
                    regime_def.boundaries.volatility_bounds = (
                        max(0, regime_def.boundaries.volatility_bounds[0]),
                        min(1, regime_def.boundaries.volatility_bounds[1])
                    )
            
            # Recalculate transition zones with smoothed boundaries
            self._calculate_transition_zones()
            
            logger.info(f"GMM boundary smoothing applied with {n_components} components")
            logger.debug(f"GMM converged: {gmm.converged_}, n_iter: {gmm.n_iter_}")
            
        except Exception as e:
            logger.warning(f"GMM boundary smoothing failed: {e}, keeping original boundaries")
    
    def _validate_definitions(self):
        """
        Validate regime definitions for consistency
        """
        issues = []
        
        # Check for complete coverage
        volatility_coverage = set()
        for regime_def in self.regime_definitions.values():
            vol_range = regime_def.boundaries.volatility_bounds
            volatility_coverage.add((vol_range[0], vol_range[1]))
        
        # Check for reasonable boundaries
        for regime_id, regime_def in self.regime_definitions.items():
            bounds = regime_def.boundaries
            
            # Volatility bounds check
            if bounds.volatility_bounds[1] - bounds.volatility_bounds[0] < 0.05:
                issues.append(f"Regime {regime_id}: Volatility range too narrow")
            
            # Trend bounds check
            if bounds.trend_bounds[1] - bounds.trend_bounds[0] < 0.001:
                issues.append(f"Regime {regime_id}: Trend range too narrow")
            
            # Risk parameters check
            if regime_def.risk_parameters['position_size_multiplier'] < 0.1:
                issues.append(f"Regime {regime_id}: Position size multiplier too low")
        
        if issues:
            logger.warning(f"Validation issues found: {issues}")
        else:
            logger.info("All regime definitions validated successfully")
    
    def optimize_boundaries(self, historical_data: pd.DataFrame) -> Dict[int, RegimeDefinition]:
        """
        Optimize regime boundaries based on historical data
        
        Args:
            historical_data: Historical market data with features
            
        Returns:
            Updated regime definitions
        """
        logger.info("Optimizing regime boundaries with historical data")
        
        # This would implement sophisticated boundary optimization
        # For now, return current definitions
        return self.regime_definitions
    
    def export_definitions(self, output_path: Optional[str] = None) -> str:
        """
        Export regime definitions to JSON
        
        Args:
            output_path: Output file path
            
        Returns:
            JSON string of definitions
        """
        export_data = {}
        
        for regime_id, definition in self.regime_definitions.items():
            export_data[regime_id] = {
                'name': definition.name,
                'description': definition.description,
                'boundaries': {
                    'volatility': definition.boundaries.volatility_bounds,
                    'trend': definition.boundaries.trend_bounds,
                    'volume': definition.boundaries.volume_bounds,
                    'confidence_threshold': definition.boundaries.confidence_threshold
                },
                'features': definition.characteristic_features,
                'entry_conditions': definition.entry_conditions,
                'exit_conditions': definition.exit_conditions,
                'strategy_preferences': definition.strategy_preferences,
                'risk_parameters': definition.risk_parameters,
                'metadata': definition.metadata
            }
        
        json_str = json.dumps(export_data, indent=2, default=str)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(json_str)
            logger.info(f"Regime definitions exported to {output_path}")
        
        return json_str
    
    def get_regime_by_conditions(self, volatility: float, trend: float, 
                                volume: float) -> Optional[RegimeDefinition]:
        """
        Get regime that matches given market conditions
        
        Args:
            volatility: Current volatility
            trend: Current trend
            volume: Current volume ratio
            
        Returns:
            Matching regime definition or None
        """
        candidates = []
        
        for regime_id, definition in self.regime_definitions.items():
            if definition.boundaries.contains(volatility, trend, volume):
                candidates.append(definition)
        
        if not candidates:
            return None
        
        # If multiple candidates, return the one with highest confidence threshold
        return max(candidates, key=lambda x: x.boundaries.confidence_threshold)
    
    def get_regime_summary(self) -> pd.DataFrame:
        """
        Get summary DataFrame of all regime definitions
        
        Returns:
            Summary DataFrame
        """
        summary_data = []
        
        for regime_id, definition in self.regime_definitions.items():
            summary_data.append({
                'regime_id': regime_id,
                'name': definition.name,
                'volatility_low': definition.boundaries.volatility_bounds[0],
                'volatility_high': definition.boundaries.volatility_bounds[1],
                'trend_low': definition.boundaries.trend_bounds[0],
                'trend_high': definition.boundaries.trend_bounds[1],
                'confidence_threshold': definition.boundaries.confidence_threshold,
                'preferred_strategies': ', '.join(definition.strategy_preferences),
                'position_multiplier': definition.risk_parameters['position_size_multiplier']
            })
        
        return pd.DataFrame(summary_data)


# Example usage
if __name__ == "__main__":
    # Create builder for 12 regimes
    builder = RegimeDefinitionBuilder(regime_count=12)
    
    # Mock analysis results
    mock_results = {
        'regime_patterns': {
            0: type('Pattern', (), {
                'volatility_range': (0.05, 0.15),
                'trend_range': (0.0, 0.01),
                'volume_profile': {'mean': 5000, 'std': 1000},
                'average_duration': 45,
                'stability_score': 0.8,
                'characteristic_features': ['low_volatility', 'bullish_bias'],
                'transition_probabilities': {0: 0.7, 1: 0.2, 2: 0.1}
            })()
        },
        'transition_matrix': pd.DataFrame(),
        'stability_metrics': {}
    }
    
    # Build definitions
    definitions = builder.build_regime_definitions(mock_results)
    
    print("\nRegime Definitions Summary:")
    print(builder.get_regime_summary())
    
    # Export to JSON
    json_output = builder.export_definitions()
    print("\nExported JSON (first 500 chars):")
    print(json_output[:500] + "...")