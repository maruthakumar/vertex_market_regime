"""
Cross-Component Synergy Detector

Detects and measures synergistic effects between components.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
from itertools import combinations

logger = logging.getLogger(__name__)


class SynergyDetector:
    """
    Detects synergistic and antagonistic effects between components
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the synergy detector
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Synergy detection parameters
        self.synergy_params = self.config.get('synergy_params', {
            'synergy_threshold': 0.3,
            'antagonism_threshold': -0.3,
            'interaction_strength_min': 0.1
        })
        
        # Component interaction weights
        self.interaction_weights = self.config.get('interaction_weights', {
            ('component_01', 'component_02'): 1.5,  # Straddle-Greeks strong interaction
            ('component_01', 'component_04'): 1.3,  # Straddle-IV skew interaction
            ('component_02', 'component_03'): 1.2,  # Greeks-OI interaction
            ('component_04', 'component_05'): 1.1,  # IV-Technical interaction
            ('component_06', 'component_07'): 1.2,  # Correlation-SR interaction
        })
        
        logger.info("Initialized SynergyDetector")
    
    def detect_synergies(
        self,
        aggregated_components: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Detect synergies between all component combinations
        
        Args:
            aggregated_components: Aggregated component data
            
        Returns:
            Dictionary containing synergy analysis results
        """
        # Calculate pairwise synergies
        pairwise_synergies = self._calculate_pairwise_synergies(
            aggregated_components
        )
        
        # Calculate triad synergies
        triad_synergies = self._calculate_triad_synergies(
            aggregated_components
        )
        
        # Identify complementary signals
        complementary_signals = self._identify_complementary_signals(
            pairwise_synergies
        )
        
        # Identify antagonistic signals
        antagonistic_signals = self._identify_antagonistic_signals(
            pairwise_synergies
        )
        
        # Calculate interaction effects
        interaction_effects = self._calculate_interaction_effects(
            aggregated_components,
            pairwise_synergies
        )
        
        # Synergy concentration analysis
        concentration_metrics = self._analyze_synergy_concentration(
            pairwise_synergies,
            triad_synergies
        )
        
        return {
            'pairwise_synergies': pairwise_synergies,
            'triad_synergies': triad_synergies,
            'complementary_signals': complementary_signals,
            'antagonistic_signals': antagonistic_signals,
            'interaction_effects': interaction_effects,
            'concentration_metrics': concentration_metrics
        }
    
    def _calculate_pairwise_synergies(
        self,
        aggregated_components: Dict[str, Dict[str, Any]]
    ) -> Dict[Tuple[str, str], float]:
        """
        Calculate synergy scores for all component pairs
        
        Args:
            aggregated_components: Aggregated component data
            
        Returns:
            Dictionary of pairwise synergy scores
        """
        synergies = {}
        component_names = list(aggregated_components.keys())
        
        for comp1, comp2 in combinations(component_names, 2):
            if comp1 in aggregated_components and comp2 in aggregated_components:
                # Get component signals
                signals1 = aggregated_components[comp1].get('signals', {})
                signals2 = aggregated_components[comp2].get('signals', {})
                
                # Get component health scores
                health1 = aggregated_components[comp1].get('health_score', 0.5)
                health2 = aggregated_components[comp2].get('health_score', 0.5)
                
                # Calculate base synergy
                synergy = self._calculate_base_synergy(signals1, signals2)
                
                # Apply health adjustment
                synergy *= (health1 * health2)
                
                # Apply interaction weight if defined
                pair_key = (comp1, comp2)
                if pair_key in self.interaction_weights:
                    synergy *= self.interaction_weights[pair_key]
                elif (comp2, comp1) in self.interaction_weights:
                    synergy *= self.interaction_weights[(comp2, comp1)]
                
                synergies[pair_key] = float(synergy)
        
        return synergies
    
    def _calculate_base_synergy(
        self,
        signals1: Dict[str, float],
        signals2: Dict[str, float]
    ) -> float:
        """
        Calculate base synergy between two signal sets
        
        Args:
            signals1: First component signals
            signals2: Second component signals
            
        Returns:
            Base synergy score
        """
        if not signals1 or not signals2:
            return 0.0
        
        # Get primary signals
        primary1 = list(signals1.values())[:3]
        primary2 = list(signals2.values())[:3]
        
        if not primary1 or not primary2:
            return 0.0
        
        # Calculate interaction as product of aligned signals
        synergy_components = []
        
        for s1 in primary1[:min(len(primary1), len(primary2))]:
            for s2 in primary2[:min(len(primary1), len(primary2))]:
                # Synergy is high when signals align (both positive or both negative)
                # and low when they oppose
                interaction = s1 * s2
                synergy_components.append(interaction)
        
        if synergy_components:
            base_synergy = np.mean(synergy_components)
        else:
            base_synergy = 0.0
        
        return base_synergy
    
    def _calculate_triad_synergies(
        self,
        aggregated_components: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Calculate synergies for three-component combinations
        
        Args:
            aggregated_components: Aggregated component data
            
        Returns:
            List of triad synergy results
        """
        triad_synergies = []
        component_names = list(aggregated_components.keys())
        
        # Calculate for all 3-component combinations
        for comp1, comp2, comp3 in combinations(component_names, 3):
            if all(c in aggregated_components for c in [comp1, comp2, comp3]):
                # Get signals
                signals1 = aggregated_components[comp1].get('signals', {})
                signals2 = aggregated_components[comp2].get('signals', {})
                signals3 = aggregated_components[comp3].get('signals', {})
                
                # Get health scores
                health1 = aggregated_components[comp1].get('health_score', 0.5)
                health2 = aggregated_components[comp2].get('health_score', 0.5)
                health3 = aggregated_components[comp3].get('health_score', 0.5)
                
                # Calculate triad synergy
                if signals1 and signals2 and signals3:
                    # Get primary signals
                    s1 = list(signals1.values())[0] if signals1 else 0.0
                    s2 = list(signals2.values())[0] if signals2 else 0.0
                    s3 = list(signals3.values())[0] if signals3 else 0.0
                    
                    # Three-way interaction
                    triad_interaction = s1 * s2 * s3
                    
                    # Apply health adjustment
                    triad_interaction *= (health1 * health2 * health3)
                    
                    triad_synergies.append({
                        'components': (comp1, comp2, comp3),
                        'synergy': float(triad_interaction),
                        'type': 'synergistic' if triad_interaction > 0 else 'antagonistic'
                    })
        
        # Sort by absolute synergy strength
        triad_synergies.sort(key=lambda x: abs(x['synergy']), reverse=True)
        
        # Return top 10 strongest triads
        return triad_synergies[:10]
    
    def _identify_complementary_signals(
        self,
        pairwise_synergies: Dict[Tuple[str, str], float]
    ) -> List[Dict[str, Any]]:
        """
        Identify complementary signal pairs
        
        Args:
            pairwise_synergies: Pairwise synergy scores
            
        Returns:
            List of complementary signal pairs
        """
        complementary = []
        
        for pair, synergy in pairwise_synergies.items():
            if synergy > self.synergy_params['synergy_threshold']:
                complementary.append({
                    'pair': pair,
                    'synergy': synergy,
                    'strength': 'strong' if synergy > 0.6 else 'moderate'
                })
        
        # Sort by synergy strength
        complementary.sort(key=lambda x: x['synergy'], reverse=True)
        
        return complementary
    
    def _identify_antagonistic_signals(
        self,
        pairwise_synergies: Dict[Tuple[str, str], float]
    ) -> List[Dict[str, Any]]:
        """
        Identify antagonistic signal pairs
        
        Args:
            pairwise_synergies: Pairwise synergy scores
            
        Returns:
            List of antagonistic signal pairs
        """
        antagonistic = []
        
        for pair, synergy in pairwise_synergies.items():
            if synergy < self.synergy_params['antagonism_threshold']:
                antagonistic.append({
                    'pair': pair,
                    'synergy': synergy,
                    'strength': 'strong' if synergy < -0.6 else 'moderate'
                })
        
        # Sort by antagonism strength (most negative first)
        antagonistic.sort(key=lambda x: x['synergy'])
        
        return antagonistic
    
    def _calculate_interaction_effects(
        self,
        aggregated_components: Dict[str, Dict[str, Any]],
        pairwise_synergies: Dict[Tuple[str, str], float]
    ) -> Dict[str, float]:
        """
        Calculate interaction effect strengths
        
        Args:
            aggregated_components: Aggregated component data
            pairwise_synergies: Pairwise synergy scores
            
        Returns:
            Dictionary of interaction effect metrics
        """
        if not pairwise_synergies:
            return {
                'mean_interaction': 0.0,
                'max_interaction': 0.0,
                'interaction_variance': 0.0,
                'dominant_interaction': 0.0
            }
        
        synergy_values = list(pairwise_synergies.values())
        
        # Calculate interaction metrics
        mean_interaction = np.mean(np.abs(synergy_values))
        max_interaction = np.max(np.abs(synergy_values))
        interaction_variance = np.var(synergy_values)
        
        # Find dominant interaction
        max_pair = max(pairwise_synergies.items(), key=lambda x: abs(x[1]))
        dominant_interaction = max_pair[1]
        
        # Calculate interaction complexity
        positive_interactions = sum(1 for s in synergy_values if s > 0)
        negative_interactions = sum(1 for s in synergy_values if s < 0)
        interaction_ratio = positive_interactions / (negative_interactions + 1e-10)
        
        return {
            'mean_interaction': float(mean_interaction),
            'max_interaction': float(max_interaction),
            'interaction_variance': float(interaction_variance),
            'dominant_interaction': float(dominant_interaction),
            'interaction_ratio': float(interaction_ratio),
            'positive_count': int(positive_interactions),
            'negative_count': int(negative_interactions)
        }
    
    def _analyze_synergy_concentration(
        self,
        pairwise_synergies: Dict[Tuple[str, str], float],
        triad_synergies: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Analyze concentration and dispersion of synergies
        
        Args:
            pairwise_synergies: Pairwise synergy scores
            triad_synergies: Triad synergy results
            
        Returns:
            Dictionary of concentration metrics
        """
        if not pairwise_synergies:
            return {
                'concentration': 0.0,
                'dispersion': 0.0,
                'momentum': 0.0,
                'acceleration': 0.0,
                'stability': 0.5,
                'efficiency': 0.0
            }
        
        synergy_values = np.array(list(pairwise_synergies.values()))
        
        # Concentration: how concentrated synergies are
        abs_synergies = np.abs(synergy_values)
        if np.sum(abs_synergies) > 0:
            concentration = np.max(abs_synergies) / np.sum(abs_synergies)
        else:
            concentration = 0.0
        
        # Dispersion: spread of synergy values
        dispersion = np.ptp(synergy_values)  # Peak-to-peak range
        
        # Momentum and acceleration (using recent history if available)
        # For now, using simple proxies
        momentum = np.mean(synergy_values)
        acceleration = np.var(synergy_values)
        
        # Stability: inverse of variance
        stability = 1.0 / (1.0 + np.var(synergy_values))
        
        # Efficiency: mean absolute synergy per standard deviation
        if np.std(synergy_values) > 0:
            efficiency = np.mean(abs_synergies) / np.std(synergy_values)
        else:
            efficiency = np.mean(abs_synergies)
        
        return {
            'concentration': float(concentration),
            'dispersion': float(dispersion),
            'momentum': float(momentum),
            'acceleration': float(acceleration),
            'stability': float(stability),
            'efficiency': float(efficiency)
        }
    
    def get_synergy_features(
        self,
        synergy_analysis: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Extract features from synergy analysis for ML consumption
        
        Args:
            synergy_analysis: Full synergy analysis results
            
        Returns:
            Dictionary of synergy features
        """
        features = {}
        
        # Pairwise synergy features
        pairwise = synergy_analysis['pairwise_synergies']
        if pairwise:
            synergy_values = list(pairwise.values())
            features['synergy_mean'] = float(np.mean(synergy_values))
            features['synergy_std'] = float(np.std(synergy_values))
            features['synergy_max'] = float(np.max(np.abs(synergy_values)))
        else:
            features['synergy_mean'] = 0.0
            features['synergy_std'] = 0.0
            features['synergy_max'] = 0.0
        
        # Triad synergy features
        triads = synergy_analysis['triad_synergies']
        if triads:
            triad_values = [t['synergy'] for t in triads]
            features['triad_synergy_mean'] = float(np.mean(triad_values))
            features['triad_synergy_max'] = float(np.max(np.abs(triad_values)))
        else:
            features['triad_synergy_mean'] = 0.0
            features['triad_synergy_max'] = 0.0
        
        # Complementary and antagonistic ratios
        complementary = synergy_analysis['complementary_signals']
        antagonistic = synergy_analysis['antagonistic_signals']
        total_pairs = len(pairwise) if pairwise else 1
        
        features['complementary_ratio'] = float(len(complementary) / total_pairs)
        features['antagonistic_ratio'] = float(len(antagonistic) / total_pairs)
        
        # Interaction effects
        interaction = synergy_analysis['interaction_effects']
        features['interaction_strength'] = interaction['mean_interaction']
        features['interaction_variance'] = interaction['interaction_variance']
        
        # Concentration metrics
        concentration = synergy_analysis['concentration_metrics']
        features['synergy_concentration'] = concentration['concentration']
        features['synergy_dispersion'] = concentration['dispersion']
        features['synergy_stability'] = concentration['stability']
        features['synergy_efficiency'] = concentration['efficiency']
        
        return features