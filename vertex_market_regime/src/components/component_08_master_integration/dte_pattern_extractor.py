"""
DTE-Adaptive Pattern Extractor

Extracts DTE-specific patterns and temporal features from component signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DTEPatterns:
    """Container for DTE-specific patterns"""
    
    # Specific DTE patterns (0-90)
    dte_specific_patterns: Dict[int, float]
    
    # DTE range patterns
    weekly_pattern: float  # 0-7 days
    monthly_pattern: float  # 8-30 days
    far_pattern: float  # 31+ days
    
    # Temporal evolution metrics
    evolution_rate: float
    transition_score: float
    adaptation_rate: float
    
    # Performance metrics by DTE
    performance_mean: float
    performance_std: float
    reliability_score: float
    consistency_score: float
    efficiency_ratio: float


class DTEPatternExtractor:
    """
    Extracts DTE-adaptive patterns from component signals
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the DTE pattern extractor
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # DTE range definitions
        self.dte_ranges = {
            'weekly': (0, 7),
            'monthly': (8, 30),
            'far': (31, 90)
        }
        
        # DTE decay parameters
        self.decay_params = self.config.get('decay_params', {
            'theta': 0.05,  # Time decay rate
            'gamma_adjustment': 1.5,  # Gamma effect multiplier
            'vega_sensitivity': 0.8  # Vega sensitivity factor
        })
        
        # Historical performance tracking
        self.historical_performance = {}
        
        logger.info("Initialized DTEPatternExtractor")
    
    def extract_dte_patterns(
        self,
        aggregated_components: Dict[str, Dict[str, Any]],
        dte: Optional[int] = None,
        historical_data: Optional[pd.DataFrame] = None
    ) -> DTEPatterns:
        """
        Extract DTE-specific patterns from aggregated components
        
        Args:
            aggregated_components: Aggregated component data
            dte: Current days to expiry
            historical_data: Optional historical performance data
            
        Returns:
            DTEPatterns object containing all DTE-related features
        """
        if dte is None:
            dte = 30  # Default to monthly expiry
        
        # Extract specific DTE patterns
        dte_specific = self._extract_dte_specific_patterns(
            aggregated_components, dte
        )
        
        # Extract range-based patterns
        range_patterns = self._extract_range_patterns(
            aggregated_components, dte
        )
        
        # Calculate temporal evolution metrics
        evolution_metrics = self._calculate_evolution_metrics(
            aggregated_components, dte
        )
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(
            aggregated_components, dte, historical_data
        )
        
        return DTEPatterns(
            dte_specific_patterns=dte_specific,
            weekly_pattern=range_patterns['weekly'],
            monthly_pattern=range_patterns['monthly'],
            far_pattern=range_patterns['far'],
            evolution_rate=evolution_metrics['evolution_rate'],
            transition_score=evolution_metrics['transition_score'],
            adaptation_rate=evolution_metrics['adaptation_rate'],
            performance_mean=performance_metrics['mean'],
            performance_std=performance_metrics['std'],
            reliability_score=performance_metrics['reliability'],
            consistency_score=performance_metrics['consistency'],
            efficiency_ratio=performance_metrics['efficiency']
        )
    
    def _extract_dte_specific_patterns(
        self,
        aggregated_components: Dict[str, Dict[str, Any]],
        current_dte: int
    ) -> Dict[int, float]:
        """
        Extract patterns for specific DTE values
        
        Args:
            aggregated_components: Aggregated component data
            current_dte: Current DTE value
            
        Returns:
            Dictionary of DTE-specific pattern values
        """
        dte_patterns = {}
        
        # Calculate patterns for key DTE points
        key_dtes = [0, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 45, 60, 90]
        
        for dte in key_dtes:
            if dte <= 90:
                # Calculate DTE-specific pattern based on theta decay
                theta_factor = np.exp(-self.decay_params['theta'] * dte)
                
                # Gamma effect increases as DTE decreases
                gamma_factor = 1.0
                if dte <= 7:
                    gamma_factor = self.decay_params['gamma_adjustment'] * (1.0 - dte / 7.0)
                
                # Vega effect decreases as DTE decreases
                vega_factor = self.decay_params['vega_sensitivity'] * np.sqrt(dte / 30.0)
                
                # Combine factors
                pattern_value = theta_factor * (1.0 + gamma_factor) * (1.0 + vega_factor)
                
                # Adjust based on component signals
                component_adjustment = self._get_component_dte_adjustment(
                    aggregated_components, dte
                )
                
                dte_patterns[dte] = float(pattern_value * component_adjustment)
        
        # Add current DTE if not in key DTEs
        if current_dte not in dte_patterns and 0 <= current_dte <= 90:
            theta_factor = np.exp(-self.decay_params['theta'] * current_dte)
            gamma_factor = 1.0
            if current_dte <= 7:
                gamma_factor = self.decay_params['gamma_adjustment'] * (1.0 - current_dte / 7.0)
            vega_factor = self.decay_params['vega_sensitivity'] * np.sqrt(current_dte / 30.0)
            
            pattern_value = theta_factor * (1.0 + gamma_factor) * (1.0 + vega_factor)
            component_adjustment = self._get_component_dte_adjustment(
                aggregated_components, current_dte
            )
            
            dte_patterns[current_dte] = float(pattern_value * component_adjustment)
        
        return dte_patterns
    
    def _get_component_dte_adjustment(
        self,
        aggregated_components: Dict[str, Dict[str, Any]],
        dte: int
    ) -> float:
        """
        Get DTE-based adjustment from component signals
        
        Args:
            aggregated_components: Aggregated component data
            dte: DTE value
            
        Returns:
            Adjustment factor based on component signals
        """
        adjustments = []
        
        # Component-specific DTE sensitivities
        dte_sensitivities = {
            'component_01': 1.2 if dte <= 7 else 0.8,  # Straddle more sensitive near expiry
            'component_02': 1.5 if dte <= 3 else 0.9,  # Greeks very sensitive near expiry
            'component_03': 1.0,  # OI/PA relatively stable
            'component_04': 1.1 if dte <= 15 else 0.9,  # IV skew matters more near expiry
            'component_05': 0.9,  # Technical indicators less DTE-sensitive
            'component_06': 0.8,  # Correlation less DTE-sensitive
            'component_07': 0.85  # Support/resistance less DTE-sensitive
        }
        
        for comp_name, comp_data in aggregated_components.items():
            if comp_name in dte_sensitivities:
                signals = comp_data.get('signals', {})
                health = comp_data.get('health_score', 0.5)
                
                # Get primary signal strength
                signal_strength = np.mean(list(signals.values())[:3]) if signals else 0.0
                
                # Apply DTE sensitivity
                sensitivity = dte_sensitivities[comp_name]
                adjustment = signal_strength * sensitivity * health
                
                adjustments.append(adjustment)
        
        return np.mean(adjustments) if adjustments else 1.0
    
    def _extract_range_patterns(
        self,
        aggregated_components: Dict[str, Dict[str, Any]],
        current_dte: int
    ) -> Dict[str, float]:
        """
        Extract patterns for DTE ranges (weekly, monthly, far)
        
        Args:
            aggregated_components: Aggregated component data
            current_dte: Current DTE value
            
        Returns:
            Dictionary of range-based patterns
        """
        range_patterns = {}
        
        for range_name, (min_dte, max_dte) in self.dte_ranges.items():
            if min_dte <= current_dte <= max_dte:
                # Current DTE is in this range
                range_activity = 1.0
                
                # Calculate position within range
                range_position = (current_dte - min_dte) / max(1, max_dte - min_dte)
                
                # Apply range-specific patterns
                if range_name == 'weekly':
                    # Weekly patterns: high gamma, high theta
                    pattern = self._calculate_weekly_pattern(
                        aggregated_components, range_position
                    )
                elif range_name == 'monthly':
                    # Monthly patterns: balanced Greeks
                    pattern = self._calculate_monthly_pattern(
                        aggregated_components, range_position
                    )
                else:  # far
                    # Far month patterns: high vega, low gamma
                    pattern = self._calculate_far_pattern(
                        aggregated_components, range_position
                    )
                
                range_patterns[range_name] = float(range_activity * pattern)
            else:
                # Current DTE not in this range
                range_patterns[range_name] = 0.0
        
        return range_patterns
    
    def _calculate_weekly_pattern(
        self,
        aggregated_components: Dict[str, Dict[str, Any]],
        position: float
    ) -> float:
        """Calculate weekly expiry pattern"""
        # Extract gamma-sensitive components
        gamma_components = ['component_02']  # Greeks component
        
        gamma_signal = 0.0
        for comp_name in gamma_components:
            if comp_name in aggregated_components:
                signals = aggregated_components[comp_name].get('signals', {})
                gamma_signal = signals.get('gamma_exposure_score', 0.0)
        
        # Weekly pattern intensifies as we approach expiry
        intensity = 1.0 - position  # Higher intensity closer to expiry
        pattern = intensity * (1.0 + abs(gamma_signal))
        
        return min(2.0, pattern)  # Cap at 2.0
    
    def _calculate_monthly_pattern(
        self,
        aggregated_components: Dict[str, Dict[str, Any]],
        position: float
    ) -> float:
        """Calculate monthly expiry pattern"""
        # Balanced pattern across all components
        all_signals = []
        
        for comp_data in aggregated_components.values():
            signals = comp_data.get('signals', {})
            if signals:
                all_signals.extend(list(signals.values())[:2])
        
        if all_signals:
            # Monthly pattern is more stable
            pattern = 0.5 + 0.5 * np.tanh(np.mean(all_signals))
        else:
            pattern = 0.5
        
        return pattern
    
    def _calculate_far_pattern(
        self,
        aggregated_components: Dict[str, Dict[str, Any]],
        position: float
    ) -> float:
        """Calculate far month pattern"""
        # Vega-sensitive components
        vega_components = ['component_04']  # IV skew component
        
        vega_signal = 0.0
        for comp_name in vega_components:
            if comp_name in aggregated_components:
                signals = aggregated_components[comp_name].get('signals', {})
                vega_signal = signals.get('term_structure_signal', 0.0)
        
        # Far month pattern is vega-dominated
        pattern = 0.3 + 0.7 * abs(vega_signal)
        
        return pattern
    
    def _calculate_evolution_metrics(
        self,
        aggregated_components: Dict[str, Dict[str, Any]],
        current_dte: int
    ) -> Dict[str, float]:
        """
        Calculate temporal evolution metrics
        
        Args:
            aggregated_components: Aggregated component data
            current_dte: Current DTE value
            
        Returns:
            Dictionary of evolution metrics
        """
        # Evolution rate: how fast patterns change with DTE
        if current_dte > 0:
            evolution_rate = 1.0 / np.sqrt(current_dte)
        else:
            evolution_rate = 2.0  # Maximum at expiry
        
        # Transition score: likelihood of regime change
        transition_score = self._calculate_transition_probability(
            aggregated_components, current_dte
        )
        
        # Adaptation rate: how quickly system adapts to DTE changes
        if current_dte <= 7:
            adaptation_rate = 1.5  # Fast adaptation near expiry
        elif current_dte <= 30:
            adaptation_rate = 1.0  # Normal adaptation
        else:
            adaptation_rate = 0.7  # Slow adaptation for far months
        
        return {
            'evolution_rate': float(evolution_rate),
            'transition_score': float(transition_score),
            'adaptation_rate': float(adaptation_rate)
        }
    
    def _calculate_transition_probability(
        self,
        aggregated_components: Dict[str, Dict[str, Any]],
        current_dte: int
    ) -> float:
        """
        Calculate probability of pattern transition
        
        Args:
            aggregated_components: Aggregated component data
            current_dte: Current DTE value
            
        Returns:
            Transition probability score
        """
        # Key transition points
        transition_points = [0, 1, 3, 7, 15, 30, 45]
        
        # Find distance to nearest transition point
        distances = [abs(current_dte - tp) for tp in transition_points]
        min_distance = min(distances)
        
        # Higher probability near transition points
        if min_distance == 0:
            base_probability = 0.8
        elif min_distance == 1:
            base_probability = 0.6
        elif min_distance <= 3:
            base_probability = 0.4
        else:
            base_probability = 0.2
        
        # Adjust based on component signals
        signal_volatility = self._calculate_signal_volatility(aggregated_components)
        
        transition_probability = base_probability * (1.0 + signal_volatility)
        
        return min(1.0, transition_probability)
    
    def _calculate_signal_volatility(
        self,
        aggregated_components: Dict[str, Dict[str, Any]]
    ) -> float:
        """Calculate volatility of component signals"""
        all_signals = []
        
        for comp_data in aggregated_components.values():
            signals = comp_data.get('signals', {})
            if signals:
                all_signals.extend(list(signals.values()))
        
        if len(all_signals) > 1:
            return float(np.std(all_signals))
        else:
            return 0.0
    
    def _calculate_performance_metrics(
        self,
        aggregated_components: Dict[str, Dict[str, Any]],
        current_dte: int,
        historical_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Calculate DTE-based performance metrics
        
        Args:
            aggregated_components: Aggregated component data
            current_dte: Current DTE value
            historical_data: Optional historical performance data
            
        Returns:
            Dictionary of performance metrics
        """
        # Calculate current performance
        current_performance = self._assess_current_performance(aggregated_components)
        
        # Update historical tracking
        if current_dte not in self.historical_performance:
            self.historical_performance[current_dte] = []
        self.historical_performance[current_dte].append(current_performance)
        
        # Calculate metrics
        if historical_data is not None and not historical_data.empty:
            # Use provided historical data
            perf_mean = float(historical_data['performance'].mean())
            perf_std = float(historical_data['performance'].std())
        else:
            # Use tracked performance
            all_performances = []
            for dte_perfs in self.historical_performance.values():
                all_performances.extend(dte_perfs)
            
            if all_performances:
                perf_mean = float(np.mean(all_performances))
                perf_std = float(np.std(all_performances))
            else:
                perf_mean = current_performance
                perf_std = 0.0
        
        # Reliability: inverse of performance variance
        reliability = 1.0 / (1.0 + perf_std) if perf_std >= 0 else 1.0
        
        # Consistency: how stable performance is
        consistency = 1.0 - min(1.0, perf_std / (abs(perf_mean) + 1e-10))
        
        # Efficiency: performance per unit of risk
        efficiency = abs(perf_mean) / (perf_std + 1e-10)
        
        return {
            'mean': perf_mean,
            'std': perf_std,
            'reliability': float(reliability),
            'consistency': float(consistency),
            'efficiency': float(min(10.0, efficiency))  # Cap efficiency at 10
        }
    
    def _assess_current_performance(
        self,
        aggregated_components: Dict[str, Dict[str, Any]]
    ) -> float:
        """
        Assess current performance based on component signals
        
        Args:
            aggregated_components: Aggregated component data
            
        Returns:
            Current performance score
        """
        performance_scores = []
        
        for comp_data in aggregated_components.values():
            health = comp_data.get('health_score', 0.5)
            confidence = comp_data.get('metadata', {}).get('confidence', 0.5)
            
            # Performance based on health and confidence
            perf = health * confidence
            performance_scores.append(perf)
        
        if performance_scores:
            return float(np.mean(performance_scores))
        else:
            return 0.5