"""
10-Component Dynamic Weighting System for Component 1

Implements the sophisticated weighting system with:
- Primary Rolling Straddle Components (30%): ATM, ITM1, OTM1 straddles (10% each)
- Individual CE Components (30%): ATM CE, ITM1 CE, OTM1 CE (10% each)  
- Individual PE Components (30%): ATM PE, ITM1 PE, OTM1 PE (10% each)
- Cross-Component Analysis (10%): correlation_factor (10%)

Features adaptive learning, volume-weighted component scoring, and performance-based
weight optimization with anti-overfitting constraints.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Optional sklearn import
try:
    from sklearn.preprocessing import MinMaxScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# GPU acceleration
try:
    import cudf
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


@dataclass
class ComponentWeights:
    """Individual component weights"""
    # Primary Rolling Straddle Components (30% total)
    atm_straddle_weight: float = 0.10
    itm1_straddle_weight: float = 0.10
    otm1_straddle_weight: float = 0.10
    
    # Individual CE Components (30% total)
    atm_ce_weight: float = 0.10
    itm1_ce_weight: float = 0.10
    otm1_ce_weight: float = 0.10
    
    # Individual PE Components (30% total)
    atm_pe_weight: float = 0.10
    itm1_pe_weight: float = 0.10
    otm1_pe_weight: float = 0.10
    
    # Cross-Component Analysis (10% total)
    correlation_factor_weight: float = 0.10


@dataclass
class WeightingAnalysisResult:
    """Result from dynamic weighting analysis"""
    component_weights: ComponentWeights
    component_scores: Dict[str, float]
    volume_weights: Dict[str, float]
    correlation_matrix: np.ndarray
    total_score: float
    confidence: float
    processing_time_ms: float
    metadata: Dict[str, Any]


@dataclass
class VolumeWeightingResult:
    """Volume-weighted component scoring result"""
    volume_scores: Dict[str, float]
    normalized_scores: Dict[str, float]
    volume_bias_factor: float
    total_volume: float


class DynamicWeightingSystem:
    """
    10-Component Dynamic Weighting System
    
    Revolutionary approach combining rolling straddle components with individual
    CE/PE analysis and cross-component correlation factors. Implements adaptive
    learning with volume-weighted scoring and performance-based optimization.
    
    Component Breakdown:
    1-3. Primary Straddles: ATM, ITM1, OTM1 rolling straddle prices
    4-6. CE Components: Individual call option prices  
    7-9. PE Components: Individual put option prices
    10. Correlation Factor: Cross-component correlation analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Dynamic Weighting System
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Performance configuration
        self.processing_budget_ms = config.get('processing_budget_ms', 30)
        self.use_gpu = config.get('use_gpu', GPU_AVAILABLE)
        
        # Initial equal weighting (10% per component)
        self.current_weights = ComponentWeights()
        
        # Adaptive learning configuration
        self.learning_enabled = config.get('learning_enabled', True)
        self.learning_rate = config.get('learning_rate', 0.01)
        self.weight_bounds = config.get('weight_bounds', (0.02, 0.25))  # Min 2%, Max 25%
        self.anti_overfitting_strength = config.get('anti_overfitting_strength', 0.1)
        
        # Volume weighting configuration
        self.volume_weight_enabled = config.get('volume_weight_enabled', True)
        self.volume_decay_factor = config.get('volume_decay_factor', 0.9)
        self.min_volume_threshold = config.get('min_volume_threshold', 100)
        
        # Cross-component correlation configuration
        self.correlation_lookback = config.get('correlation_lookback', 50)  # 50 minutes
        self.correlation_threshold = config.get('correlation_threshold', 0.7)
        
        # Performance tracking
        self.weight_history = []
        self.performance_history = []
        
        self.logger.info("DynamicWeightingSystem initialized with 10-component structure")
    
    async def calculate_dynamic_weights(self, 
                                      straddle_data: Dict[str, np.ndarray],
                                      volume_data: Dict[str, np.ndarray],
                                      performance_feedback: Optional[Dict[str, float]] = None) -> WeightingAnalysisResult:
        """
        Calculate dynamic weights for all 10 components
        
        Args:
            straddle_data: Dictionary with straddle time series data
            volume_data: Dictionary with volume data
            performance_feedback: Optional performance metrics for adaptive learning
            
        Returns:
            WeightingAnalysisResult with calculated weights and analysis
        """
        start_time = time.time()
        
        try:
            # Step 1: Calculate volume-weighted component scores
            volume_weights = await self._calculate_volume_weights(volume_data)
            
            # Step 2: Calculate cross-component correlation matrix
            correlation_matrix = await self._calculate_correlation_matrix(straddle_data)
            
            # Step 3: Calculate individual component scores
            component_scores = await self._calculate_component_scores(
                straddle_data, volume_weights, correlation_matrix
            )
            
            # Step 4: Apply adaptive learning if enabled and feedback provided
            if self.learning_enabled and performance_feedback:
                updated_weights = await self._apply_adaptive_learning(
                    component_scores, performance_feedback
                )
            else:
                updated_weights = self.current_weights
            
            # Step 5: Normalize weights to sum to 1.0
            normalized_weights = self._normalize_weights(updated_weights)
            
            # Step 6: Calculate total score and confidence
            total_score = self._calculate_total_score(component_scores, normalized_weights)
            confidence = self._calculate_confidence(component_scores, correlation_matrix)
            
            # Update current weights
            self.current_weights = normalized_weights
            
            processing_time = (time.time() - start_time) * 1000
            
            # Performance validation
            if processing_time > self.processing_budget_ms:
                self.logger.warning(f"Weighting processing {processing_time:.2f}ms exceeded budget {self.processing_budget_ms}ms")
            
            return WeightingAnalysisResult(
                component_weights=normalized_weights,
                component_scores=component_scores,
                volume_weights=volume_weights.volume_scores,
                correlation_matrix=correlation_matrix,
                total_score=total_score,
                confidence=confidence,
                processing_time_ms=processing_time,
                metadata={
                    'learning_enabled': self.learning_enabled,
                    'volume_weighting_enabled': self.volume_weight_enabled,
                    'total_volume': volume_weights.total_volume,
                    'correlation_threshold': self.correlation_threshold,
                    'weight_bounds': self.weight_bounds
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to calculate dynamic weights: {e}")
            raise
    
    async def _calculate_volume_weights(self, volume_data: Dict[str, np.ndarray]) -> VolumeWeightingResult:
        """
        Calculate volume-weighted component scoring
        
        Args:
            volume_data: Dictionary with volume time series
            
        Returns:
            VolumeWeightingResult with volume-based weights
        """
        try:
            volume_scores = {}
            total_volume = 0.0
            
            # Expected volume components
            volume_components = [
                'atm_volume', 'itm1_volume', 'otm1_volume',
                'atm_ce_volume', 'itm1_ce_volume', 'otm1_ce_volume',
                'atm_pe_volume', 'itm1_pe_volume', 'otm1_pe_volume',
                'combined_volume'  # For correlation factor
            ]
            
            # Calculate volume scores for each component
            for component in volume_components:
                if component in volume_data:
                    # Use recent volume with decay factor
                    volume_series = volume_data[component]
                    if len(volume_series) > 0:
                        # Apply exponential decay to recent volumes
                        weights = np.power(self.volume_decay_factor, np.arange(len(volume_series))[::-1])
                        weighted_volume = np.average(volume_series, weights=weights)
                        volume_scores[component] = float(max(weighted_volume, self.min_volume_threshold))
                        total_volume += weighted_volume
                    else:
                        volume_scores[component] = self.min_volume_threshold
                else:
                    # Fallback for missing volume data
                    volume_scores[component] = self.min_volume_threshold
            
            # Normalize volume scores
            if total_volume > 0:
                normalized_scores = {k: v / total_volume for k, v in volume_scores.items()}
            else:
                # Equal weighting fallback
                normalized_scores = {k: 1.0 / len(volume_scores) for k in volume_scores.keys()}
            
            # Calculate volume bias factor
            volume_std = np.std(list(volume_scores.values()))
            volume_mean = np.mean(list(volume_scores.values()))
            volume_bias_factor = volume_std / max(volume_mean, 1.0)
            
            return VolumeWeightingResult(
                volume_scores=volume_scores,
                normalized_scores=normalized_scores,
                volume_bias_factor=volume_bias_factor,
                total_volume=total_volume
            )
            
        except Exception as e:
            self.logger.error(f"Failed to calculate volume weights: {e}")
            # Return equal weights fallback
            equal_weight = 1.0 / 10
            return VolumeWeightingResult(
                volume_scores={f'component_{i}': equal_weight for i in range(10)},
                normalized_scores={f'component_{i}': equal_weight for i in range(10)},
                volume_bias_factor=0.0,
                total_volume=0.0
            )
    
    async def _calculate_correlation_matrix(self, straddle_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate cross-component correlation matrix
        
        Args:
            straddle_data: Dictionary with straddle time series
            
        Returns:
            Correlation matrix (10x10)
        """
        try:
            # Prepare data matrix for correlation calculation
            data_components = [
                'atm_straddle', 'itm1_straddle', 'otm1_straddle',
                'atm_ce', 'itm1_ce', 'otm1_ce',
                'atm_pe', 'itm1_pe', 'otm1_pe'
            ]
            
            # Build data matrix
            data_matrix = []
            for component in data_components:
                if component in straddle_data:
                    series = straddle_data[component]
                    # Use recent data for correlation
                    recent_data = series[-self.correlation_lookback:] if len(series) > self.correlation_lookback else series
                    data_matrix.append(recent_data)
                else:
                    # Fallback: create synthetic data
                    synthetic_data = np.random.randn(self.correlation_lookback) * 0.1
                    data_matrix.append(synthetic_data)
            
            # Ensure all series have same length
            min_length = min(len(series) for series in data_matrix)
            if min_length > 0:
                data_matrix = [series[-min_length:] for series in data_matrix]
                
                # Calculate correlation matrix
                if min_length > 5:  # Minimum data points for reliable correlation
                    data_array = np.array(data_matrix)
                    correlation_matrix = np.corrcoef(data_array)
                    
                    # Handle NaN values
                    correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
                    
                    # Add correlation factor as 10th component (average correlation)
                    expanded_matrix = np.zeros((10, 10))
                    expanded_matrix[:9, :9] = correlation_matrix
                    
                    # Calculate correlation factor (average off-diagonal correlation)
                    off_diagonal = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
                    avg_correlation = np.mean(np.abs(off_diagonal))
                    expanded_matrix[9, :9] = avg_correlation
                    expanded_matrix[:9, 9] = avg_correlation
                    expanded_matrix[9, 9] = 1.0
                    
                    return expanded_matrix
            
            # Fallback: identity matrix (no correlation)
            return np.eye(10)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate correlation matrix: {e}")
            return np.eye(10)
    
    async def _calculate_component_scores(self, 
                                        straddle_data: Dict[str, np.ndarray],
                                        volume_weights: VolumeWeightingResult,
                                        correlation_matrix: np.ndarray) -> Dict[str, float]:
        """
        Calculate individual component scores
        
        Args:
            straddle_data: Straddle time series data
            volume_weights: Volume weighting results
            correlation_matrix: Component correlation matrix
            
        Returns:
            Dictionary of component scores
        """
        try:
            component_scores = {}
            
            # Component mapping
            components = [
                'atm_straddle', 'itm1_straddle', 'otm1_straddle',
                'atm_ce', 'itm1_ce', 'otm1_ce', 
                'atm_pe', 'itm1_pe', 'otm1_pe',
                'correlation_factor'
            ]
            
            # Calculate scores for each component
            for i, component in enumerate(components):
                if component == 'correlation_factor':
                    # Correlation factor score based on matrix analysis
                    correlation_strength = np.mean(np.abs(correlation_matrix[i, :i]))
                    component_scores[component] = float(correlation_strength)
                    
                elif component in straddle_data:
                    # Price-based component score
                    series = straddle_data[component]
                    if len(series) > 1:
                        # Calculate volatility-adjusted score
                        price_change = np.abs(np.diff(series))
                        volatility = np.std(price_change) if len(price_change) > 0 else 0.0
                        recent_value = float(series[-1])
                        
                        # Combine price level with volatility
                        base_score = recent_value / (1.0 + volatility)
                        
                        # Apply volume weighting if available
                        volume_key = f"{component.replace('_straddle', '')}_volume"
                        if volume_key in volume_weights.normalized_scores:
                            volume_weight = volume_weights.normalized_scores[volume_key]
                            score = base_score * (1.0 + volume_weight)
                        else:
                            score = base_score
                        
                        component_scores[component] = float(score)
                    else:
                        component_scores[component] = 0.0
                else:
                    # Missing component
                    component_scores[component] = 0.0
            
            # Normalize scores to [0, 1] range
            if component_scores:
                max_score = max(component_scores.values())
                if max_score > 0:
                    component_scores = {k: v / max_score for k, v in component_scores.items()}
            
            return component_scores
            
        except Exception as e:
            self.logger.error(f"Failed to calculate component scores: {e}")
            # Equal scores fallback
            return {f'component_{i}': 0.1 for i in range(10)}
    
    async def _apply_adaptive_learning(self, 
                                     component_scores: Dict[str, float],
                                     performance_feedback: Dict[str, float]) -> ComponentWeights:
        """
        Apply adaptive learning to update component weights
        
        Args:
            component_scores: Current component scores
            performance_feedback: Performance metrics for learning
            
        Returns:
            Updated ComponentWeights
        """
        try:
            # Get performance metrics
            accuracy = performance_feedback.get('accuracy', 0.5)
            precision = performance_feedback.get('precision', 0.5)
            recall = performance_feedback.get('recall', 0.5)
            
            # Calculate performance adjustment factor
            performance_score = (accuracy + precision + recall) / 3.0
            adjustment_factor = (performance_score - 0.5) * self.learning_rate
            
            # Update weights based on component scores and performance
            current = self.current_weights
            
            # Apply learning with anti-overfitting constraints
            new_weights = ComponentWeights(
                atm_straddle_weight=self._adjust_weight(
                    current.atm_straddle_weight, 
                    component_scores.get('atm_straddle', 0.0),
                    adjustment_factor
                ),
                itm1_straddle_weight=self._adjust_weight(
                    current.itm1_straddle_weight,
                    component_scores.get('itm1_straddle', 0.0),
                    adjustment_factor
                ),
                otm1_straddle_weight=self._adjust_weight(
                    current.otm1_straddle_weight,
                    component_scores.get('otm1_straddle', 0.0), 
                    adjustment_factor
                ),
                atm_ce_weight=self._adjust_weight(
                    current.atm_ce_weight,
                    component_scores.get('atm_ce', 0.0),
                    adjustment_factor
                ),
                itm1_ce_weight=self._adjust_weight(
                    current.itm1_ce_weight,
                    component_scores.get('itm1_ce', 0.0),
                    adjustment_factor
                ),
                otm1_ce_weight=self._adjust_weight(
                    current.otm1_ce_weight,
                    component_scores.get('otm1_ce', 0.0),
                    adjustment_factor
                ),
                atm_pe_weight=self._adjust_weight(
                    current.atm_pe_weight,
                    component_scores.get('atm_pe', 0.0),
                    adjustment_factor
                ),
                itm1_pe_weight=self._adjust_weight(
                    current.itm1_pe_weight,
                    component_scores.get('itm1_pe', 0.0),
                    adjustment_factor
                ),
                otm1_pe_weight=self._adjust_weight(
                    current.otm1_pe_weight,
                    component_scores.get('otm1_pe', 0.0),
                    adjustment_factor
                ),
                correlation_factor_weight=self._adjust_weight(
                    current.correlation_factor_weight,
                    component_scores.get('correlation_factor', 0.0),
                    adjustment_factor
                )
            )
            
            return new_weights
            
        except Exception as e:
            self.logger.error(f"Failed to apply adaptive learning: {e}")
            return self.current_weights
    
    def _adjust_weight(self, current_weight: float, component_score: float, adjustment_factor: float) -> float:
        """
        Adjust individual weight with bounds checking and anti-overfitting
        
        Args:
            current_weight: Current weight value
            component_score: Component performance score
            adjustment_factor: Learning adjustment factor
            
        Returns:
            Adjusted weight value
        """
        # Calculate raw adjustment
        score_adjustment = component_score * adjustment_factor
        
        # Apply anti-overfitting constraints
        regularization = self.anti_overfitting_strength * (current_weight - 0.10)  # Pull towards equal weight
        
        # Calculate new weight
        new_weight = current_weight + score_adjustment - regularization
        
        # Apply bounds
        min_weight, max_weight = self.weight_bounds
        new_weight = np.clip(new_weight, min_weight, max_weight)
        
        return float(new_weight)
    
    def _normalize_weights(self, weights: ComponentWeights) -> ComponentWeights:
        """
        Normalize weights to sum to 1.0
        
        Args:
            weights: ComponentWeights to normalize
            
        Returns:
            Normalized ComponentWeights
        """
        # Get all weight values
        weight_values = [
            weights.atm_straddle_weight, weights.itm1_straddle_weight, weights.otm1_straddle_weight,
            weights.atm_ce_weight, weights.itm1_ce_weight, weights.otm1_ce_weight,
            weights.atm_pe_weight, weights.itm1_pe_weight, weights.otm1_pe_weight,
            weights.correlation_factor_weight
        ]
        
        # Normalize
        total_weight = sum(weight_values)
        if total_weight > 0:
            normalization_factor = 1.0 / total_weight
            
            return ComponentWeights(
                atm_straddle_weight=weights.atm_straddle_weight * normalization_factor,
                itm1_straddle_weight=weights.itm1_straddle_weight * normalization_factor,
                otm1_straddle_weight=weights.otm1_straddle_weight * normalization_factor,
                atm_ce_weight=weights.atm_ce_weight * normalization_factor,
                itm1_ce_weight=weights.itm1_ce_weight * normalization_factor,
                otm1_ce_weight=weights.otm1_ce_weight * normalization_factor,
                atm_pe_weight=weights.atm_pe_weight * normalization_factor,
                itm1_pe_weight=weights.itm1_pe_weight * normalization_factor,
                otm1_pe_weight=weights.otm1_pe_weight * normalization_factor,
                correlation_factor_weight=weights.correlation_factor_weight * normalization_factor
            )
        else:
            # Return equal weights
            return ComponentWeights()
    
    def _calculate_total_score(self, component_scores: Dict[str, float], weights: ComponentWeights) -> float:
        """Calculate weighted total score"""
        return float(
            component_scores.get('atm_straddle', 0.0) * weights.atm_straddle_weight +
            component_scores.get('itm1_straddle', 0.0) * weights.itm1_straddle_weight +
            component_scores.get('otm1_straddle', 0.0) * weights.otm1_straddle_weight +
            component_scores.get('atm_ce', 0.0) * weights.atm_ce_weight +
            component_scores.get('itm1_ce', 0.0) * weights.itm1_ce_weight +
            component_scores.get('otm1_ce', 0.0) * weights.otm1_ce_weight +
            component_scores.get('atm_pe', 0.0) * weights.atm_pe_weight +
            component_scores.get('itm1_pe', 0.0) * weights.itm1_pe_weight +
            component_scores.get('otm1_pe', 0.0) * weights.otm1_pe_weight +
            component_scores.get('correlation_factor', 0.0) * weights.correlation_factor_weight
        )
    
    def _calculate_confidence(self, component_scores: Dict[str, float], correlation_matrix: np.ndarray) -> float:
        """Calculate confidence based on component consistency"""
        try:
            # Calculate confidence from score consistency
            scores = list(component_scores.values())
            score_std = np.std(scores)
            score_mean = np.mean(scores)
            
            # Coefficient of variation
            if score_mean > 0:
                consistency = 1.0 - (score_std / score_mean)
            else:
                consistency = 0.5
            
            # Factor in correlation strength
            avg_correlation = np.mean(np.abs(correlation_matrix))
            correlation_confidence = min(avg_correlation * 2, 1.0)
            
            # Combined confidence
            confidence = (consistency * 0.7 + correlation_confidence * 0.3)
            return float(np.clip(confidence, 0.0, 1.0))
            
        except:
            return 0.5


# Factory function
def create_dynamic_weighting_system(config: Dict[str, Any]) -> DynamicWeightingSystem:
    """Create and configure DynamicWeightingSystem instance"""
    return DynamicWeightingSystem(config)