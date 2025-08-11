"""
Component 5: Dual DTE Analysis Framework

Advanced DTE analysis framework supporting both specific DTE analysis (dte=0, dte=1, ..., dte=90)
and DTE range analysis (dte_0_to_7, dte_8_to_30, dte_31_plus) with adaptive learning parameters
and comprehensive ATR-EMA-CPR integration.

Features:
- Specific DTE percentile system with individual DTE tracking
- DTE range percentile system with categorical analysis
- DTE-adaptive parameters with different configurations per DTE/range
- Historical parameter optimization for both specific and range analysis
- Unified analysis combining specific and range-based insights
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import logging
import asyncio
import time
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

from .dual_asset_data_extractor import StraddlePriceData, UnderlyingPriceData
from .straddle_atr_ema_cpr_engine import StraddleAnalysisResult
from .underlying_atr_ema_cpr_engine import UnderlyingAnalysisResult

warnings.filterwarnings('ignore')


@dataclass
class DTESpecificAnalysis:
    """Analysis results for specific DTE values"""
    dte_value: int
    data_count: int
    atr_percentiles: Dict[str, float]
    ema_trend_percentiles: Dict[str, float]
    cpr_level_percentiles: Dict[str, float]
    volatility_regime_distribution: Dict[int, float]
    trend_direction_distribution: Dict[int, float]
    performance_metrics: Dict[str, float]
    confidence_score: float
    learned_parameters: Dict[str, Any]


@dataclass
class DTERangeAnalysis:
    """Analysis results for DTE ranges"""
    range_name: str
    dte_range: Tuple[int, int]
    combined_data_count: int
    aggregated_percentiles: Dict[str, Dict[str, float]]
    regime_stability: Dict[str, float]
    cross_dte_consistency: Dict[str, float]
    range_specific_parameters: Dict[str, Any]
    range_performance: Dict[str, float]
    confidence_score: float


@dataclass
class DTEIntegratedResult:
    """Combined result from specific DTE and range analysis"""
    specific_dte_results: Dict[int, DTESpecificAnalysis]
    range_results: Dict[str, DTERangeAnalysis]
    dte_transition_patterns: Dict[str, np.ndarray]
    cross_dte_correlations: Dict[str, Dict[str, float]]
    unified_regime_classification: np.ndarray
    adaptive_weights: Dict[str, Dict[str, float]]
    integrated_confidence: np.ndarray
    feature_contributions: Dict[str, np.ndarray]
    metadata: Dict[str, Any]


class DTESpecificAnalyzer:
    """Analyzer for individual DTE analysis with historical percentile tracking"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # DTE-specific parameters
        self.dte_range = range(0, 91)  # DTE 0 to 90
        self.min_data_points_per_dte = config.get('min_data_points_per_dte', 10)
        
        # Historical database for percentile tracking
        self.historical_database = defaultdict(list)
        
        # Adaptive parameters per DTE
        self.dte_specific_params = self._initialize_dte_parameters()

    def _initialize_dte_parameters(self) -> Dict[int, Dict[str, Any]]:
        """Initialize DTE-specific parameters"""
        
        params = {}
        
        for dte in self.dte_range:
            # Parameters vary based on DTE characteristics
            if dte <= 7:  # Very short-term
                params[dte] = {
                    'atr_weight': 1.5,  # Higher volatility weight
                    'ema_weight': 0.8,  # Lower trend weight
                    'cpr_weight': 1.2,  # Higher pivot weight
                    'vol_sensitivity': 0.9,
                    'trend_smoothing': 0.7
                }
            elif dte <= 30:  # Short to medium-term
                params[dte] = {
                    'atr_weight': 1.2,
                    'ema_weight': 1.0,
                    'cpr_weight': 1.0,
                    'vol_sensitivity': 0.8,
                    'trend_smoothing': 0.8
                }
            elif dte <= 60:  # Medium to long-term
                params[dte] = {
                    'atr_weight': 1.0,
                    'ema_weight': 1.2,
                    'cpr_weight': 0.9,
                    'vol_sensitivity': 0.7,
                    'trend_smoothing': 0.9
                }
            else:  # Long-term
                params[dte] = {
                    'atr_weight': 0.8,
                    'ema_weight': 1.4,  # Higher trend weight
                    'cpr_weight': 0.8,
                    'vol_sensitivity': 0.6,
                    'trend_smoothing': 1.0
                }
        
        return params

    async def analyze_specific_dte(self, dte_value: int, 
                                 straddle_result: StraddleAnalysisResult,
                                 underlying_result: UnderlyingAnalysisResult,
                                 dte_mask: np.ndarray) -> DTESpecificAnalysis:
        """Analyze specific DTE value with historical percentile tracking"""
        
        if dte_value not in self.dte_range:
            raise ValueError(f"DTE value {dte_value} outside supported range")
        
        # Extract data for specific DTE
        dte_indices = np.where(dte_mask)[0]
        data_count = len(dte_indices)
        
        if data_count < self.min_data_points_per_dte:
            self.logger.warning(f"Insufficient data points for DTE {dte_value}: {data_count}")
            return self._create_empty_dte_analysis(dte_value)
        
        # Calculate ATR percentiles for this DTE
        atr_percentiles = self._calculate_dte_atr_percentiles(
            straddle_result, underlying_result, dte_indices, dte_value
        )
        
        # Calculate EMA trend percentiles
        ema_trend_percentiles = self._calculate_dte_ema_percentiles(
            straddle_result, underlying_result, dte_indices, dte_value
        )
        
        # Calculate CPR level percentiles
        cpr_level_percentiles = self._calculate_dte_cpr_percentiles(
            straddle_result, underlying_result, dte_indices, dte_value
        )
        
        # Analyze regime distributions for this DTE
        vol_regime_dist = self._analyze_volatility_regime_distribution(
            straddle_result, underlying_result, dte_indices
        )
        
        trend_direction_dist = self._analyze_trend_direction_distribution(
            straddle_result, underlying_result, dte_indices
        )
        
        # Calculate performance metrics
        performance_metrics = self._calculate_dte_performance(
            straddle_result, underlying_result, dte_indices, dte_value
        )
        
        # Calculate confidence score
        confidence_score = self._calculate_dte_confidence(
            data_count, performance_metrics, dte_value
        )
        
        # Get learned parameters for this DTE
        learned_parameters = self._get_learned_dte_parameters(dte_value, performance_metrics)
        
        return DTESpecificAnalysis(
            dte_value=dte_value,
            data_count=data_count,
            atr_percentiles=atr_percentiles,
            ema_trend_percentiles=ema_trend_percentiles,
            cpr_level_percentiles=cpr_level_percentiles,
            volatility_regime_distribution=vol_regime_dist,
            trend_direction_distribution=trend_direction_dist,
            performance_metrics=performance_metrics,
            confidence_score=confidence_score,
            learned_parameters=learned_parameters
        )

    def _calculate_dte_atr_percentiles(self, straddle_result: StraddleAnalysisResult,
                                     underlying_result: UnderlyingAnalysisResult,
                                     dte_indices: np.ndarray, dte_value: int) -> Dict[str, float]:
        """Calculate ATR percentiles for specific DTE"""
        
        percentiles = {}
        
        # Straddle ATR percentiles
        if hasattr(straddle_result.atr_result, 'atr_14') and len(straddle_result.atr_result.atr_14) > 0:
            dte_atr_values = straddle_result.atr_result.atr_14[dte_indices]
            valid_values = dte_atr_values[~np.isnan(dte_atr_values)]
            
            if len(valid_values) > 0:
                percentiles['straddle_atr_14_mean'] = np.mean(valid_values)
                percentiles['straddle_atr_14_std'] = np.std(valid_values)
                percentiles['straddle_atr_14_p50'] = np.percentile(valid_values, 50)
                percentiles['straddle_atr_14_p75'] = np.percentile(valid_values, 75)
                percentiles['straddle_atr_14_p90'] = np.percentile(valid_values, 90)
        
        # Underlying ATR percentiles (daily)
        if hasattr(underlying_result.atr_result, 'daily_atr') and 'atr_14' in underlying_result.atr_result.daily_atr:
            underlying_atr = underlying_result.atr_result.daily_atr['atr_14']
            if len(underlying_atr) > max(dte_indices) if len(dte_indices) > 0 else False:
                dte_underlying_atr = underlying_atr[dte_indices]
                valid_underlying = dte_underlying_atr[~np.isnan(dte_underlying_atr)]
                
                if len(valid_underlying) > 0:
                    percentiles['underlying_atr_14_mean'] = np.mean(valid_underlying)
                    percentiles['underlying_atr_14_std'] = np.std(valid_underlying)
                    percentiles['underlying_atr_14_p50'] = np.percentile(valid_underlying, 50)
                    percentiles['underlying_atr_14_p75'] = np.percentile(valid_underlying, 75)
                    percentiles['underlying_atr_14_p90'] = np.percentile(valid_underlying, 90)
        
        # Store in historical database for percentile tracking
        self.historical_database[f'dte_{dte_value}_atr'].extend(percentiles.values())
        
        return percentiles

    def _calculate_dte_ema_percentiles(self, straddle_result: StraddleAnalysisResult,
                                     underlying_result: UnderlyingAnalysisResult,
                                     dte_indices: np.ndarray, dte_value: int) -> Dict[str, float]:
        """Calculate EMA trend percentiles for specific DTE"""
        
        percentiles = {}
        
        # Straddle EMA trend strength
        if hasattr(straddle_result.ema_result, 'trend_strength') and len(straddle_result.ema_result.trend_strength) > 0:
            dte_trend_strength = straddle_result.ema_result.trend_strength[dte_indices]
            valid_strength = dte_trend_strength[~np.isnan(dte_trend_strength)]
            
            if len(valid_strength) > 0:
                percentiles['straddle_trend_strength_mean'] = np.mean(valid_strength)
                percentiles['straddle_trend_strength_p75'] = np.percentile(valid_strength, 75)
                percentiles['straddle_trend_strength_p90'] = np.percentile(valid_strength, 90)
        
        # Underlying trend strength (daily)
        if hasattr(underlying_result.ema_result, 'trend_strengths') and 'daily' in underlying_result.ema_result.trend_strengths:
            underlying_strength = underlying_result.ema_result.trend_strengths['daily']
            if len(underlying_strength) > max(dte_indices) if len(dte_indices) > 0 else False:
                dte_underlying_strength = underlying_strength[dte_indices]
                valid_underlying_strength = dte_underlying_strength[~np.isnan(dte_underlying_strength)]
                
                if len(valid_underlying_strength) > 0:
                    percentiles['underlying_trend_strength_mean'] = np.mean(valid_underlying_strength)
                    percentiles['underlying_trend_strength_p75'] = np.percentile(valid_underlying_strength, 75)
                    percentiles['underlying_trend_strength_p90'] = np.percentile(valid_underlying_strength, 90)
        
        return percentiles

    def _calculate_dte_cpr_percentiles(self, straddle_result: StraddleAnalysisResult,
                                     underlying_result: UnderlyingAnalysisResult,
                                     dte_indices: np.ndarray, dte_value: int) -> Dict[str, float]:
        """Calculate CPR level percentiles for specific DTE"""
        
        percentiles = {}
        
        # Straddle CPR width analysis
        if hasattr(straddle_result.cpr_result, 'cpr_width') and len(straddle_result.cpr_result.cpr_width) > 0:
            dte_cpr_width = straddle_result.cpr_result.cpr_width[dte_indices]
            valid_width = dte_cpr_width[~np.isnan(dte_cpr_width)]
            
            if len(valid_width) > 0:
                percentiles['straddle_cpr_width_mean'] = np.mean(valid_width)
                percentiles['straddle_cpr_width_p50'] = np.percentile(valid_width, 50)
                percentiles['straddle_cpr_width_p75'] = np.percentile(valid_width, 75)
        
        # Add underlying CPR analysis if available
        # (Implementation would depend on underlying CPR result structure)
        
        return percentiles

    def _analyze_volatility_regime_distribution(self, straddle_result: StraddleAnalysisResult,
                                              underlying_result: UnderlyingAnalysisResult,
                                              dte_indices: np.ndarray) -> Dict[int, float]:
        """Analyze volatility regime distribution for DTE"""
        
        distribution = {}
        
        # Get straddle volatility regimes for this DTE
        if hasattr(straddle_result, 'regime_classification') and len(straddle_result.regime_classification) > 0:
            dte_regimes = straddle_result.regime_classification[dte_indices]
            
            # Calculate distribution
            unique_regimes, counts = np.unique(dte_regimes, return_counts=True)
            total_count = len(dte_regimes)
            
            for regime, count in zip(unique_regimes, counts):
                distribution[int(regime)] = count / total_count
        
        return distribution

    def _analyze_trend_direction_distribution(self, straddle_result: StraddleAnalysisResult,
                                            underlying_result: UnderlyingAnalysisResult,
                                            dte_indices: np.ndarray) -> Dict[int, float]:
        """Analyze trend direction distribution for DTE"""
        
        distribution = {}
        
        # Get straddle trend directions for this DTE  
        if hasattr(straddle_result.ema_result, 'trend_direction') and len(straddle_result.ema_result.trend_direction) > 0:
            dte_trends = straddle_result.ema_result.trend_direction[dte_indices]
            
            # Calculate distribution
            unique_trends, counts = np.unique(dte_trends, return_counts=True)
            total_count = len(dte_trends)
            
            for trend, count in zip(unique_trends, counts):
                distribution[int(trend)] = count / total_count
        
        return distribution

    def _calculate_dte_performance(self, straddle_result: StraddleAnalysisResult,
                                 underlying_result: UnderlyingAnalysisResult,
                                 dte_indices: np.ndarray, dte_value: int) -> Dict[str, float]:
        """Calculate performance metrics for specific DTE"""
        
        metrics = {}
        
        # Processing time per data point
        if hasattr(straddle_result, 'processing_time_ms'):
            metrics['avg_processing_time_per_point'] = straddle_result.processing_time_ms / max(len(dte_indices), 1)
        
        # Confidence scores
        if hasattr(straddle_result, 'confidence_scores') and len(straddle_result.confidence_scores) > 0:
            dte_confidence = straddle_result.confidence_scores[dte_indices]
            metrics['avg_confidence'] = np.mean(dte_confidence[~np.isnan(dte_confidence)])
            metrics['confidence_stability'] = 1 - np.std(dte_confidence[~np.isnan(dte_confidence)])
        
        # DTE-specific accuracy metrics (would be calculated from backtesting results)
        metrics['historical_accuracy'] = self._get_historical_accuracy(dte_value)
        
        return metrics

    def _get_historical_accuracy(self, dte_value: int) -> float:
        """Get historical accuracy for specific DTE (placeholder)"""
        # This would be calculated from historical backtesting results
        # For now, return a baseline based on DTE characteristics
        
        if dte_value <= 7:
            return 0.72  # Lower accuracy for very short-term
        elif dte_value <= 30:
            return 0.78  # Medium accuracy for short-term
        elif dte_value <= 60:
            return 0.82  # Higher accuracy for medium-term
        else:
            return 0.75  # Good accuracy for long-term

    def _calculate_dte_confidence(self, data_count: int, performance_metrics: Dict[str, float], dte_value: int) -> float:
        """Calculate confidence score for DTE analysis"""
        
        # Base confidence from data availability
        data_confidence = min(data_count / 50, 1.0)  # Normalize to 50+ data points
        
        # Performance-based confidence
        perf_confidence = performance_metrics.get('avg_confidence', 0.5)
        
        # Historical accuracy confidence
        hist_confidence = performance_metrics.get('historical_accuracy', 0.5)
        
        # Combined confidence
        combined_confidence = (data_confidence * 0.3 + perf_confidence * 0.4 + hist_confidence * 0.3)
        
        return np.clip(combined_confidence, 0.1, 0.95)

    def _get_learned_dte_parameters(self, dte_value: int, performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Get learned parameters for specific DTE"""
        
        base_params = self.dte_specific_params[dte_value].copy()
        
        # Adjust parameters based on performance
        historical_accuracy = performance_metrics.get('historical_accuracy', 0.5)
        
        if historical_accuracy > 0.8:
            # Good performance - slightly increase weights
            base_params['atr_weight'] *= 1.1
            base_params['ema_weight'] *= 1.1
        elif historical_accuracy < 0.6:
            # Poor performance - reduce weights
            base_params['atr_weight'] *= 0.9
            base_params['ema_weight'] *= 0.9
        
        base_params['last_updated'] = datetime.utcnow()
        base_params['performance_based_adjustment'] = historical_accuracy
        
        return base_params

    def _create_empty_dte_analysis(self, dte_value: int) -> DTESpecificAnalysis:
        """Create empty analysis for DTEs with insufficient data"""
        
        return DTESpecificAnalysis(
            dte_value=dte_value,
            data_count=0,
            atr_percentiles={},
            ema_trend_percentiles={},
            cpr_level_percentiles={},
            volatility_regime_distribution={},
            trend_direction_distribution={},
            performance_metrics={'historical_accuracy': 0.5},
            confidence_score=0.1,
            learned_parameters=self.dte_specific_params.get(dte_value, {})
        )


class DTERangeAnalyzer:
    """Analyzer for DTE range analysis with categorical learning"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # DTE range definitions
        self.dte_ranges = {
            'dte_0_to_7': (0, 7),
            'dte_8_to_30': (8, 30),
            'dte_31_plus': (31, 90)
        }
        
        # Range-specific parameters
        self.range_parameters = self._initialize_range_parameters()

    def _initialize_range_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Initialize parameters for each DTE range"""
        
        return {
            'dte_0_to_7': {
                'volatility_sensitivity': 0.95,
                'trend_lag_adjustment': 0.7,
                'pivot_importance': 1.3,
                'regime_stability_threshold': 0.6,
                'learning_rate': 0.15
            },
            'dte_8_to_30': {
                'volatility_sensitivity': 0.85,
                'trend_lag_adjustment': 0.85,
                'pivot_importance': 1.0,
                'regime_stability_threshold': 0.7,
                'learning_rate': 0.1
            },
            'dte_31_plus': {
                'volatility_sensitivity': 0.75,
                'trend_lag_adjustment': 1.0,
                'pivot_importance': 0.8,
                'regime_stability_threshold': 0.8,
                'learning_rate': 0.05
            }
        }

    async def analyze_dte_range(self, range_name: str,
                              specific_dte_results: Dict[int, DTESpecificAnalysis]) -> DTERangeAnalysis:
        """Analyze DTE range using specific DTE results"""
        
        if range_name not in self.dte_ranges:
            raise ValueError(f"Unknown DTE range: {range_name}")
        
        dte_range = self.dte_ranges[range_name]
        range_start, range_end = dte_range
        
        # Get specific DTE results within range
        range_dte_results = {
            dte: result for dte, result in specific_dte_results.items()
            if range_start <= dte <= range_end
        }
        
        if not range_dte_results:
            return self._create_empty_range_analysis(range_name, dte_range)
        
        # Aggregate data from range
        combined_data_count = sum(result.data_count for result in range_dte_results.values())
        
        # Aggregate percentiles across range
        aggregated_percentiles = self._aggregate_range_percentiles(range_dte_results)
        
        # Calculate regime stability across range
        regime_stability = self._calculate_regime_stability(range_dte_results)
        
        # Calculate cross-DTE consistency within range
        cross_dte_consistency = self._calculate_cross_dte_consistency(range_dte_results)
        
        # Get range-specific parameters
        range_specific_parameters = self._get_range_specific_parameters(range_name, range_dte_results)
        
        # Calculate range performance
        range_performance = self._calculate_range_performance(range_dte_results)
        
        # Calculate confidence score
        confidence_score = self._calculate_range_confidence(
            combined_data_count, range_performance, len(range_dte_results)
        )
        
        return DTERangeAnalysis(
            range_name=range_name,
            dte_range=dte_range,
            combined_data_count=combined_data_count,
            aggregated_percentiles=aggregated_percentiles,
            regime_stability=regime_stability,
            cross_dte_consistency=cross_dte_consistency,
            range_specific_parameters=range_specific_parameters,
            range_performance=range_performance,
            confidence_score=confidence_score
        )

    def _aggregate_range_percentiles(self, range_dte_results: Dict[int, DTESpecificAnalysis]) -> Dict[str, Dict[str, float]]:
        """Aggregate percentiles across DTE range"""
        
        aggregated = {
            'atr_percentiles': {},
            'ema_percentiles': {},
            'cpr_percentiles': {}
        }
        
        # Collect all percentile values
        atr_values = defaultdict(list)
        ema_values = defaultdict(list)
        cpr_values = defaultdict(list)
        
        for dte_result in range_dte_results.values():
            # ATR percentiles
            for key, value in dte_result.atr_percentiles.items():
                atr_values[key].append(value)
            
            # EMA percentiles
            for key, value in dte_result.ema_trend_percentiles.items():
                ema_values[key].append(value)
            
            # CPR percentiles
            for key, value in dte_result.cpr_level_percentiles.items():
                cpr_values[key].append(value)
        
        # Calculate aggregated statistics
        for key, values in atr_values.items():
            if values:
                aggregated['atr_percentiles'][key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        for key, values in ema_values.items():
            if values:
                aggregated['ema_percentiles'][key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        for key, values in cpr_values.items():
            if values:
                aggregated['cpr_percentiles'][key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return aggregated

    def _calculate_regime_stability(self, range_dte_results: Dict[int, DTESpecificAnalysis]) -> Dict[str, float]:
        """Calculate regime stability across DTE range"""
        
        stability = {}
        
        # Collect volatility regime distributions
        vol_regimes = []
        trend_regimes = []
        
        for dte_result in range_dte_results.values():
            if dte_result.volatility_regime_distribution:
                vol_regimes.append(dte_result.volatility_regime_distribution)
            if dte_result.trend_direction_distribution:
                trend_regimes.append(dte_result.trend_direction_distribution)
        
        # Calculate volatility regime stability
        if vol_regimes:
            stability['volatility_regime_stability'] = self._calculate_distribution_stability(vol_regimes)
        
        # Calculate trend regime stability
        if trend_regimes:
            stability['trend_regime_stability'] = self._calculate_distribution_stability(trend_regimes)
        
        return stability

    def _calculate_distribution_stability(self, distributions: List[Dict[int, float]]) -> float:
        """Calculate stability of regime distributions"""
        
        if len(distributions) < 2:
            return 1.0  # Perfect stability if only one distribution
        
        # Get all unique regimes
        all_regimes = set()
        for dist in distributions:
            all_regimes.update(dist.keys())
        
        # Calculate coefficient of variation for each regime
        regime_cvs = []
        
        for regime in all_regimes:
            regime_probs = [dist.get(regime, 0.0) for dist in distributions]
            
            if np.mean(regime_probs) > 0:
                cv = np.std(regime_probs) / np.mean(regime_probs)
                regime_cvs.append(cv)
        
        # Stability is inverse of average CV
        if regime_cvs:
            avg_cv = np.mean(regime_cvs)
            stability = max(0, 1 - avg_cv)
        else:
            stability = 0.5
        
        return stability

    def _calculate_cross_dte_consistency(self, range_dte_results: Dict[int, DTESpecificAnalysis]) -> Dict[str, float]:
        """Calculate consistency across DTEs within range"""
        
        consistency = {}
        
        # Performance consistency
        performances = [result.performance_metrics.get('historical_accuracy', 0.5) 
                       for result in range_dte_results.values()]
        
        if performances:
            consistency['performance_consistency'] = 1 - (np.std(performances) / max(np.mean(performances), 0.1))
        
        # Confidence consistency
        confidences = [result.confidence_score for result in range_dte_results.values()]
        
        if confidences:
            consistency['confidence_consistency'] = 1 - (np.std(confidences) / max(np.mean(confidences), 0.1))
        
        return consistency

    def _get_range_specific_parameters(self, range_name: str, 
                                     range_dte_results: Dict[int, DTESpecificAnalysis]) -> Dict[str, Any]:
        """Get parameters specific to DTE range"""
        
        base_params = self.range_parameters[range_name].copy()
        
        # Adjust based on range performance
        avg_performance = np.mean([
            result.performance_metrics.get('historical_accuracy', 0.5)
            for result in range_dte_results.values()
        ])
        
        if avg_performance > 0.8:
            base_params['volatility_sensitivity'] *= 1.05
            base_params['learning_rate'] *= 0.95  # Slower learning when performing well
        elif avg_performance < 0.6:
            base_params['volatility_sensitivity'] *= 0.95
            base_params['learning_rate'] *= 1.1   # Faster learning when underperforming
        
        base_params['avg_range_performance'] = avg_performance
        base_params['dte_count_in_range'] = len(range_dte_results)
        
        return base_params

    def _calculate_range_performance(self, range_dte_results: Dict[int, DTESpecificAnalysis]) -> Dict[str, float]:
        """Calculate performance metrics for DTE range"""
        
        performance = {}
        
        # Average performance across range
        performances = [result.performance_metrics.get('historical_accuracy', 0.5) 
                       for result in range_dte_results.values()]
        
        performance['avg_accuracy'] = np.mean(performances)
        performance['min_accuracy'] = np.min(performances)
        performance['max_accuracy'] = np.max(performances)
        performance['accuracy_range'] = np.max(performances) - np.min(performances)
        
        # Average confidence
        confidences = [result.confidence_score for result in range_dte_results.values()]
        performance['avg_confidence'] = np.mean(confidences)
        
        # Data coverage
        total_data = sum(result.data_count for result in range_dte_results.values())
        performance['total_data_points'] = total_data
        performance['avg_data_per_dte'] = total_data / len(range_dte_results)
        
        return performance

    def _calculate_range_confidence(self, combined_data_count: int, 
                                  range_performance: Dict[str, float], 
                                  dte_count: int) -> float:
        """Calculate confidence score for range analysis"""
        
        # Data availability confidence
        data_confidence = min(combined_data_count / 200, 1.0)
        
        # Performance confidence
        perf_confidence = range_performance.get('avg_accuracy', 0.5)
        
        # Coverage confidence (how many DTEs in range have data)
        expected_dte_count = {
            'dte_0_to_7': 8,
            'dte_8_to_30': 23,
            'dte_31_plus': 60
        }
        
        coverage_confidence = min(dte_count / expected_dte_count.get('dte_0_to_7', 8), 1.0)
        
        # Combined confidence
        combined_confidence = (data_confidence * 0.4 + perf_confidence * 0.4 + coverage_confidence * 0.2)
        
        return np.clip(combined_confidence, 0.1, 0.95)

    def _create_empty_range_analysis(self, range_name: str, dte_range: Tuple[int, int]) -> DTERangeAnalysis:
        """Create empty analysis for ranges with no data"""
        
        return DTERangeAnalysis(
            range_name=range_name,
            dte_range=dte_range,
            combined_data_count=0,
            aggregated_percentiles={},
            regime_stability={},
            cross_dte_consistency={},
            range_specific_parameters=self.range_parameters.get(range_name, {}),
            range_performance={'avg_accuracy': 0.5},
            confidence_score=0.1
        )


class DualDTEFramework:
    """Main framework combining specific DTE and range analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # Initialize analyzers
        self.specific_analyzer = DTESpecificAnalyzer(config)
        self.range_analyzer = DTERangeAnalyzer(config)
        
        # Integration parameters
        self.integration_weights = {
            'specific_dte_weight': config.get('specific_dte_weight', 0.6),
            'range_analysis_weight': config.get('range_analysis_weight', 0.4)
        }

    async def analyze_dual_dte_framework(self, straddle_data: StraddlePriceData,
                                       straddle_result: StraddleAnalysisResult,
                                       underlying_result: UnderlyingAnalysisResult) -> DTEIntegratedResult:
        """Complete dual DTE analysis combining specific and range analysis"""
        
        start_time = time.time()
        
        try:
            # Extract DTE values from straddle data
            dte_values = straddle_data.dte_values
            unique_dtes = np.unique(dte_values[~np.isnan(dte_values)])
            
            # Analyze each specific DTE
            specific_dte_results = {}
            
            for dte in unique_dtes:
                if 0 <= dte <= 90:  # Within supported range
                    dte_mask = (dte_values == dte)
                    dte_analysis = await self.specific_analyzer.analyze_specific_dte(
                        int(dte), straddle_result, underlying_result, dte_mask
                    )
                    specific_dte_results[int(dte)] = dte_analysis
            
            # Analyze DTE ranges
            range_results = {}
            for range_name in self.range_analyzer.dte_ranges.keys():
                range_analysis = await self.range_analyzer.analyze_dte_range(
                    range_name, specific_dte_results
                )
                range_results[range_name] = range_analysis
            
            # Calculate DTE transition patterns
            dte_transition_patterns = self._calculate_dte_transitions(specific_dte_results)
            
            # Calculate cross-DTE correlations
            cross_dte_correlations = self._calculate_cross_dte_correlations(specific_dte_results)
            
            # Generate unified regime classification
            unified_regime = self._generate_unified_regime_classification(
                straddle_data, specific_dte_results, range_results
            )
            
            # Calculate adaptive weights
            adaptive_weights = self._calculate_adaptive_weights(specific_dte_results, range_results)
            
            # Calculate integrated confidence
            integrated_confidence = self._calculate_integrated_confidence(
                straddle_data, specific_dte_results, range_results
            )
            
            # Calculate feature contributions
            feature_contributions = self._calculate_feature_contributions(
                straddle_data, specific_dte_results, range_results
            )
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            return DTEIntegratedResult(
                specific_dte_results=specific_dte_results,
                range_results=range_results,
                dte_transition_patterns=dte_transition_patterns,
                cross_dte_correlations=cross_dte_correlations,
                unified_regime_classification=unified_regime,
                adaptive_weights=adaptive_weights,
                integrated_confidence=integrated_confidence,
                feature_contributions=feature_contributions,
                metadata={
                    'framework': 'dual_dte_comprehensive',
                    'specific_dtes_analyzed': len(specific_dte_results),
                    'ranges_analyzed': len(range_results),
                    'processing_time_ms': processing_time_ms,
                    'integration_method': 'weighted_ensemble'
                }
            )
            
        except Exception as e:
            self.logger.error(f"Dual DTE framework analysis failed: {str(e)}")
            raise

    def _calculate_dte_transitions(self, specific_dte_results: Dict[int, DTESpecificAnalysis]) -> Dict[str, np.ndarray]:
        """Calculate DTE transition patterns"""
        
        transitions = {}
        
        # Sort DTEs for transition analysis
        sorted_dtes = sorted(specific_dte_results.keys())
        
        if len(sorted_dtes) < 2:
            return transitions
        
        # Calculate volatility regime transitions
        vol_transitions = []
        for i in range(len(sorted_dtes) - 1):
            current_dte = sorted_dtes[i]
            next_dte = sorted_dtes[i + 1]
            
            current_regimes = specific_dte_results[current_dte].volatility_regime_distribution
            next_regimes = specific_dte_results[next_dte].volatility_regime_distribution
            
            # Calculate regime change probability
            transition_prob = self._calculate_regime_transition_probability(current_regimes, next_regimes)
            vol_transitions.append(transition_prob)
        
        transitions['volatility_regime_transitions'] = np.array(vol_transitions)
        
        return transitions

    def _calculate_regime_transition_probability(self, current_regimes: Dict[int, float], 
                                               next_regimes: Dict[int, float]) -> float:
        """Calculate probability of regime transition between DTEs"""
        
        if not current_regimes or not next_regimes:
            return 0.5
        
        # Find most probable regimes
        current_regime = max(current_regimes, key=current_regimes.get)
        next_regime = max(next_regimes, key=next_regimes.get)
        
        # Transition probability (0 = no change, 1 = complete change)
        if current_regime == next_regime:
            return 0.0  # No transition
        else:
            # Weight by regime probabilities
            current_prob = current_regimes[current_regime]
            next_prob = next_regimes[next_regime]
            return 1 - (current_prob * next_prob)

    def _calculate_cross_dte_correlations(self, specific_dte_results: Dict[int, DTESpecificAnalysis]) -> Dict[str, Dict[str, float]]:
        """Calculate correlations between different DTEs"""
        
        correlations = {
            'performance_correlations': {},
            'regime_correlations': {}
        }
        
        dtes = list(specific_dte_results.keys())
        
        # Performance correlations
        performances = [specific_dte_results[dte].performance_metrics.get('historical_accuracy', 0.5) for dte in dtes]
        
        for i, dte1 in enumerate(dtes):
            for j, dte2 in enumerate(dtes):
                if i < j:  # Avoid duplicate pairs
                    correlation_key = f"dte_{dte1}_vs_dte_{dte2}"
                    
                    # Simple correlation based on performance difference
                    perf_diff = abs(performances[i] - performances[j])
                    correlation = max(0, 1 - perf_diff * 2)  # Higher correlation for similar performance
                    
                    correlations['performance_correlations'][correlation_key] = correlation
        
        return correlations

    def _generate_unified_regime_classification(self, straddle_data: StraddlePriceData,
                                              specific_dte_results: Dict[int, DTESpecificAnalysis],
                                              range_results: Dict[str, DTERangeAnalysis]) -> np.ndarray:
        """Generate unified regime classification using both specific and range analysis"""
        
        dte_values = straddle_data.dte_values
        unified_regime = np.full(len(dte_values), 0)  # Default neutral regime
        
        for i, dte in enumerate(dte_values):
            if np.isnan(dte):
                continue
                
            dte_int = int(dte)
            
            # Get specific DTE analysis if available
            specific_weight = 0.0
            specific_regime = 0
            
            if dte_int in specific_dte_results:
                specific_analysis = specific_dte_results[dte_int]
                specific_weight = specific_analysis.confidence_score
                
                # Get most probable regime from specific analysis
                if specific_analysis.volatility_regime_distribution:
                    specific_regime = max(specific_analysis.volatility_regime_distribution, 
                                        key=specific_analysis.volatility_regime_distribution.get)
            
            # Get range analysis
            range_weight = 0.0
            range_regime = 0
            
            for range_name, range_analysis in range_results.items():
                range_start, range_end = range_analysis.dte_range
                if range_start <= dte_int <= range_end:
                    range_weight = range_analysis.confidence_score
                    # Use range-specific regime logic (simplified)
                    range_regime = self._infer_range_regime(range_analysis)
                    break
            
            # Combine specific and range analysis
            total_weight = specific_weight + range_weight
            
            if total_weight > 0:
                specific_contribution = (specific_weight / total_weight) * specific_regime
                range_contribution = (range_weight / total_weight) * range_regime
                unified_regime[i] = int(round(specific_contribution + range_contribution))
            else:
                unified_regime[i] = 0  # Neutral if no analysis available
        
        return unified_regime

    def _infer_range_regime(self, range_analysis: DTERangeAnalysis) -> int:
        """Infer regime from range analysis"""
        
        # Simplified logic based on range performance
        avg_accuracy = range_analysis.range_performance.get('avg_accuracy', 0.5)
        
        if avg_accuracy > 0.8:
            return 2  # High confidence regime
        elif avg_accuracy > 0.6:
            return 1  # Medium confidence regime
        else:
            return 0  # Low confidence/neutral regime

    def _calculate_adaptive_weights(self, specific_dte_results: Dict[int, DTESpecificAnalysis],
                                  range_results: Dict[str, DTERangeAnalysis]) -> Dict[str, Dict[str, float]]:
        """Calculate adaptive weights for different analysis components"""
        
        weights = {
            'specific_dte_weights': {},
            'range_weights': {}
        }
        
        # Calculate weights for specific DTEs based on confidence and performance
        total_specific_score = 0
        specific_scores = {}
        
        for dte, result in specific_dte_results.items():
            score = result.confidence_score * result.performance_metrics.get('historical_accuracy', 0.5)
            specific_scores[dte] = score
            total_specific_score += score
        
        # Normalize specific DTE weights
        for dte, score in specific_scores.items():
            weights['specific_dte_weights'][dte] = score / max(total_specific_score, 1e-10)
        
        # Calculate weights for ranges
        total_range_score = 0
        range_scores = {}
        
        for range_name, result in range_results.items():
            score = result.confidence_score * result.range_performance.get('avg_accuracy', 0.5)
            range_scores[range_name] = score
            total_range_score += score
        
        # Normalize range weights
        for range_name, score in range_scores.items():
            weights['range_weights'][range_name] = score / max(total_range_score, 1e-10)
        
        return weights

    def _calculate_integrated_confidence(self, straddle_data: StraddlePriceData,
                                       specific_dte_results: Dict[int, DTESpecificAnalysis],
                                       range_results: Dict[str, DTERangeAnalysis]) -> np.ndarray:
        """Calculate integrated confidence scores"""
        
        dte_values = straddle_data.dte_values
        confidence = np.full(len(dte_values), 0.5)  # Default medium confidence
        
        for i, dte in enumerate(dte_values):
            if np.isnan(dte):
                continue
                
            dte_int = int(dte)
            
            # Get specific DTE confidence
            specific_confidence = 0.0
            if dte_int in specific_dte_results:
                specific_confidence = specific_dte_results[dte_int].confidence_score
            
            # Get range confidence
            range_confidence = 0.0
            for range_name, range_analysis in range_results.items():
                range_start, range_end = range_analysis.dte_range
                if range_start <= dte_int <= range_end:
                    range_confidence = range_analysis.confidence_score
                    break
            
            # Combine confidences
            if specific_confidence > 0 and range_confidence > 0:
                # Both available - weighted average
                confidence[i] = (specific_confidence * self.integration_weights['specific_dte_weight'] + 
                               range_confidence * self.integration_weights['range_analysis_weight'])
            elif specific_confidence > 0:
                # Only specific available
                confidence[i] = specific_confidence * 0.8  # Slight penalty for missing range
            elif range_confidence > 0:
                # Only range available
                confidence[i] = range_confidence * 0.7  # Penalty for missing specific
            else:
                # Neither available
                confidence[i] = 0.3  # Low confidence
        
        return confidence

    def _calculate_feature_contributions(self, straddle_data: StraddlePriceData,
                                       specific_dte_results: Dict[int, DTESpecificAnalysis],
                                       range_results: Dict[str, DTERangeAnalysis]) -> Dict[str, np.ndarray]:
        """Calculate feature contributions from dual DTE analysis"""
        
        dte_values = straddle_data.dte_values
        data_length = len(dte_values)
        
        contributions = {
            'dte_specific_contribution': np.zeros(data_length),
            'dte_range_contribution': np.zeros(data_length),
            'dte_transition_contribution': np.zeros(data_length)
        }
        
        for i, dte in enumerate(dte_values):
            if np.isnan(dte):
                continue
                
            dte_int = int(dte)
            
            # Specific DTE contribution
            if dte_int in specific_dte_results:
                result = specific_dte_results[dte_int]
                contributions['dte_specific_contribution'][i] = result.confidence_score
            
            # Range contribution
            for range_name, range_analysis in range_results.items():
                range_start, range_end = range_analysis.dte_range
                if range_start <= dte_int <= range_end:
                    contributions['dte_range_contribution'][i] = range_analysis.confidence_score
                    break
            
            # Transition contribution (based on DTE position)
            if 7 <= dte_int <= 30:  # Transition zone between short and medium term
                contributions['dte_transition_contribution'][i] = 0.8
            elif 30 <= dte_int <= 35:  # Transition zone between medium and long term
                contributions['dte_transition_contribution'][i] = 0.6
            else:
                contributions['dte_transition_contribution'][i] = 0.2
        
        return contributions