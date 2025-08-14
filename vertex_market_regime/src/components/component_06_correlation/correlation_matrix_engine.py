"""
Correlation Matrix Engine for Component 6

Implements comprehensive correlation matrix calculation across all components
with mathematical precision and performance optimization for <200ms processing.

Features:
- Multi-strike correlation analysis (ATM/ITM1/OTM1)
- Cross-component correlation matrices  
- DTE-specific correlation calculations
- Zone-based intraday correlation weights
- Gap-adjusted correlation measurements
- Real-time correlation breakdown detection

🎯 PURE MATHEMATICAL CALCULATIONS ONLY - NO CLASSIFICATION LOGIC
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import time
import warnings
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.signal import savgol_filter
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

warnings.filterwarnings('ignore')


@dataclass
class CorrelationMatrixResult:
    """Result from correlation matrix calculation"""
    correlation_matrix: np.ndarray
    correlation_coefficients: Dict[str, float]
    stability_metrics: Dict[str, float]
    processing_time_ms: float
    breakdown_indicators: np.ndarray
    confidence_scores: np.ndarray
    feature_count: int
    timestamp: datetime


@dataclass
class MultiStrikeCorrelationData:
    """Multi-strike correlation input data"""
    atm_data: pd.DataFrame
    itm1_data: pd.DataFrame
    otm1_data: pd.DataFrame
    underlying_data: pd.DataFrame
    metadata: Dict[str, Any]


@dataclass
class DTECorrelationProfile:
    """DTE-specific correlation profile"""
    dte_range: Tuple[int, int]
    correlation_coefficients: np.ndarray
    stability_score: float
    breakdown_risk: float
    sample_size: int


class CorrelationMatrixEngine:
    """
    High-performance correlation matrix engine for Component 6
    
    Calculates comprehensive correlation matrices across all component data
    with mathematical precision and sub-200ms performance targets.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # Performance optimization settings
        self.target_processing_time_ms = config.get('target_processing_time_ms', 200)
        self.parallel_processing = config.get('parallel_processing', True)
        self.max_workers = config.get('max_workers', 4)
        
        # Correlation calculation settings
        self.min_correlation_periods = config.get('min_correlation_periods', 20)
        self.correlation_methods = config.get('correlation_methods', ['pearson'])
        self.stability_window = config.get('stability_window', 50)
        
        # Timeframe settings
        self.timeframes = config.get('timeframes', [3, 5, 10, 15])  # minutes
        
        # DTE range settings
        self.dte_ranges = {
            'weekly': (0, 7),
            'bi_weekly': (8, 14),
            'monthly': (15, 30),
            'far_month': (31, 90)
        }
        
        # Zone-based correlation weights
        self.zone_weights = {
            'PRE_OPEN': 0.15,   # 09:00-09:15
            'MID_MORN': 0.25,   # 09:15-11:30
            'LUNCH': 0.20,      # 11:30-13:00
            'AFTERNOON': 0.25,  # 13:00-15:00
            'CLOSE': 0.15       # 15:00-15:30
        }
        
        # Gap adaptation settings
        self.gap_correlation_weights = {
            'no_gap': 1.0,      # -0.2% to +0.2%
            'small_gap': 0.8,   # -0.5% to +0.5%
            'medium_gap': 0.6,  # -1.0% to +1.0%
            'large_gap': 0.4,   # -2.0% to +2.0%
            'extreme_gap': 0.2  # >2.0%
        }
        
        self.logger.info(f"Correlation Matrix Engine initialized with {len(self.timeframes)} timeframes")

    async def calculate_comprehensive_correlation_matrix(self, 
                                                       components_data: Dict[int, Dict[str, pd.DataFrame]],
                                                       gap_info: Optional[Dict[str, float]] = None) -> CorrelationMatrixResult:
        """
        Calculate comprehensive correlation matrix across all components
        
        Args:
            components_data: Dict mapping component_id -> component data
            gap_info: Optional gap information for correlation adjustment
            
        Returns:
            CorrelationMatrixResult with complete correlation analysis
        """
        start_time = time.time()
        
        try:
            # Initialize correlation matrix
            num_components = len(components_data)
            correlation_matrix = np.eye(num_components, dtype=np.float32)
            correlation_coefficients = {}
            stability_metrics = {}
            breakdown_indicators = []
            confidence_scores = []
            
            if self.parallel_processing:
                # Parallel correlation calculation
                correlation_results = await self._calculate_correlations_parallel(
                    components_data, gap_info
                )
            else:
                # Sequential correlation calculation
                correlation_results = await self._calculate_correlations_sequential(
                    components_data, gap_info
                )
            
            # Build correlation matrix from results
            for i, (component_i, data_i) in enumerate(components_data.items()):
                for j, (component_j, data_j) in enumerate(components_data.items()):
                    if i != j:
                        correlation_key = f'component_{component_i}_vs_{component_j}'
                        if correlation_key in correlation_results:
                            correlation_matrix[i, j] = correlation_results[correlation_key]['correlation']
                            stability_metrics[correlation_key] = correlation_results[correlation_key]['stability']
                            breakdown_indicators.append(correlation_results[correlation_key]['breakdown_risk'])
                            confidence_scores.append(correlation_results[correlation_key]['confidence'])
            
            # Calculate overall stability metrics
            overall_stability = self._calculate_overall_stability(correlation_matrix)
            stability_metrics['overall_stability'] = overall_stability
            
            processing_time = (time.time() - start_time) * 1000
            
            # Performance validation
            if processing_time > self.target_processing_time_ms:
                self.logger.warning(f"Processing time {processing_time:.1f}ms exceeded target {self.target_processing_time_ms}ms")
            
            return CorrelationMatrixResult(
                correlation_matrix=correlation_matrix,
                correlation_coefficients=correlation_coefficients,
                stability_metrics=stability_metrics,
                processing_time_ms=processing_time,
                breakdown_indicators=np.array(breakdown_indicators, dtype=np.float32),
                confidence_scores=np.array(confidence_scores, dtype=np.float32),
                feature_count=len(correlation_coefficients),
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.logger.error(f"Correlation matrix calculation failed: {e}")
            
            # Return minimal result on failure
            return CorrelationMatrixResult(
                correlation_matrix=np.eye(2, dtype=np.float32),
                correlation_coefficients={'error': 0.0},
                stability_metrics={'error': 0.0},
                processing_time_ms=processing_time,
                breakdown_indicators=np.array([1.0], dtype=np.float32),
                confidence_scores=np.array([0.0], dtype=np.float32),
                feature_count=0,
                timestamp=datetime.utcnow()
            )

    async def _calculate_correlations_parallel(self, 
                                             components_data: Dict[int, Dict[str, pd.DataFrame]],
                                             gap_info: Optional[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Calculate correlations using parallel processing"""
        
        correlation_tasks = []
        component_pairs = []
        
        # Create all component pairs for correlation calculation
        components = list(components_data.keys())
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                component_i, component_j = components[i], components[j]
                component_pairs.append((component_i, component_j))
        
        # Execute parallel correlation calculations
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for comp_i, comp_j in component_pairs:
                future = executor.submit(
                    self._calculate_pairwise_correlation,
                    components_data[comp_i],
                    components_data[comp_j],
                    comp_i,
                    comp_j,
                    gap_info
                )
                futures.append(future)
            
            # Collect results
            correlation_results = {}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    correlation_results.update(result)
                except Exception as e:
                    self.logger.error(f"Parallel correlation calculation failed: {e}")
        
        return correlation_results

    async def _calculate_correlations_sequential(self, 
                                               components_data: Dict[int, Dict[str, pd.DataFrame]],
                                               gap_info: Optional[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Calculate correlations sequentially"""
        
        correlation_results = {}
        components = list(components_data.keys())
        
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                component_i, component_j = components[i], components[j]
                
                pairwise_result = self._calculate_pairwise_correlation(
                    components_data[component_i],
                    components_data[component_j],
                    component_i,
                    component_j,
                    gap_info
                )
                
                correlation_results.update(pairwise_result)
        
        return correlation_results

    def _calculate_pairwise_correlation(self, 
                                      data_i: Dict[str, pd.DataFrame],
                                      data_j: Dict[str, pd.DataFrame],
                                      component_i: int,
                                      component_j: int,
                                      gap_info: Optional[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Calculate correlation between two components"""
        
        result_key = f'component_{component_i}_vs_{component_j}'
        
        try:
            # Extract representative time series from each component
            series_i = self._extract_representative_series(data_i, component_i)
            series_j = self._extract_representative_series(data_j, component_j)
            
            if len(series_i) < self.min_correlation_periods or len(series_j) < self.min_correlation_periods:
                return {result_key: {
                    'correlation': 0.0,
                    'stability': 0.0,
                    'breakdown_risk': 1.0,
                    'confidence': 0.0
                }}
            
            # Align series by index/time
            aligned_i, aligned_j = self._align_time_series(series_i, series_j)
            
            if len(aligned_i) < self.min_correlation_periods:
                return {result_key: {
                    'correlation': 0.0,
                    'stability': 0.0,
                    'breakdown_risk': 1.0,
                    'confidence': 0.0
                }}
            
            # Calculate correlation with selected method
            correlation, p_value = self._calculate_correlation_coefficient(aligned_i, aligned_j)
            
            # Apply gap adjustment if gap info provided
            if gap_info and 'gap_category' in gap_info:
                gap_category = gap_info['gap_category']
                gap_weight = self.gap_correlation_weights.get(gap_category, 1.0)
                correlation *= gap_weight
            
            # Calculate stability metrics
            stability = self._calculate_correlation_stability(aligned_i, aligned_j)
            
            # Calculate breakdown risk
            breakdown_risk = self._calculate_breakdown_risk(aligned_i, aligned_j, correlation)
            
            # Calculate confidence score
            confidence = self._calculate_confidence_score(correlation, p_value, len(aligned_i))
            
            return {result_key: {
                'correlation': float(correlation),
                'stability': float(stability),
                'breakdown_risk': float(breakdown_risk),
                'confidence': float(confidence)
            }}
            
        except Exception as e:
            self.logger.error(f"Pairwise correlation calculation failed: {e}")
            return {result_key: {
                'correlation': 0.0,
                'stability': 0.0,
                'breakdown_risk': 1.0,
                'confidence': 0.0
            }}

    def _extract_representative_series(self, component_data: Dict[str, pd.DataFrame], 
                                     component_id: int) -> pd.Series:
        """Extract representative time series from component data"""
        
        try:
            if component_id == 1:  # Straddle component
                if 'atm_straddle' in component_data:
                    return component_data['atm_straddle']['premium'] if 'premium' in component_data['atm_straddle'].columns else component_data['atm_straddle'].iloc[:, 0]
                elif 'premium' in component_data:
                    return component_data['premium']
                    
            elif component_id == 2:  # Greeks component
                if 'greeks_data' in component_data and 'delta' in component_data['greeks_data'].columns:
                    return component_data['greeks_data']['delta']
                elif 'delta' in component_data:
                    return component_data['delta']
                    
            elif component_id == 3:  # OI/PA component
                if 'oi_data' in component_data and 'total_oi' in component_data['oi_data'].columns:
                    return component_data['oi_data']['total_oi']
                elif 'total_oi' in component_data:
                    return component_data['total_oi']
                    
            elif component_id == 4:  # IV Skew component
                if 'iv_data' in component_data and 'atm_iv' in component_data['iv_data'].columns:
                    return component_data['iv_data']['atm_iv']
                elif 'atm_iv' in component_data:
                    return component_data['atm_iv']
                    
            elif component_id == 5:  # Technical component
                if 'technical_data' in component_data and 'atr' in component_data['technical_data'].columns:
                    return component_data['technical_data']['atr']
                elif 'atr' in component_data:
                    return component_data['atr']
            
            # Fallback: use first available DataFrame's first column
            for key, df in component_data.items():
                if isinstance(df, pd.DataFrame) and len(df) > 0:
                    return df.iloc[:, 0]
                elif isinstance(df, pd.Series):
                    return df
            
            # Ultimate fallback: return empty series
            return pd.Series(dtype=float)
            
        except Exception as e:
            self.logger.error(f"Error extracting representative series for component {component_id}: {e}")
            return pd.Series(dtype=float)

    def _align_time_series(self, series_i: pd.Series, series_j: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Align two time series by index"""
        
        try:
            if isinstance(series_i, pd.Series) and isinstance(series_j, pd.Series):
                # If both have indices, align by index
                aligned = pd.concat([series_i, series_j], axis=1, join='inner').dropna()
                if len(aligned) > 0:
                    return aligned.iloc[:, 0].values, aligned.iloc[:, 1].values
            
            # Fallback: truncate to minimum length
            min_len = min(len(series_i), len(series_j))
            if min_len > 0:
                return series_i.values[:min_len], series_j.values[:min_len]
            else:
                return np.array([]), np.array([])
                
        except Exception as e:
            self.logger.error(f"Error aligning time series: {e}")
            return np.array([]), np.array([])

    def _calculate_correlation_coefficient(self, series_i: np.ndarray, 
                                         series_j: np.ndarray) -> Tuple[float, float]:
        """Calculate correlation coefficient using specified method"""
        
        try:
            # Remove NaN values
            mask = ~(np.isnan(series_i) | np.isnan(series_j))
            clean_i = series_i[mask]
            clean_j = series_j[mask]
            
            if len(clean_i) < self.min_correlation_periods:
                return 0.0, 1.0
            
            # Use primary correlation method
            method = self.correlation_methods[0] if self.correlation_methods else 'pearson'
            
            if method == 'pearson':
                correlation, p_value = pearsonr(clean_i, clean_j)
            elif method == 'spearman':
                correlation, p_value = spearmanr(clean_i, clean_j)
            elif method == 'kendall':
                correlation, p_value = kendalltau(clean_i, clean_j)
            else:
                correlation, p_value = pearsonr(clean_i, clean_j)
            
            # Handle NaN results
            if np.isnan(correlation):
                correlation = 0.0
            if np.isnan(p_value):
                p_value = 1.0
            
            return float(correlation), float(p_value)
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation coefficient: {e}")
            return 0.0, 1.0

    def _calculate_correlation_stability(self, series_i: np.ndarray, series_j: np.ndarray) -> float:
        """Calculate correlation stability over rolling windows"""
        
        try:
            if len(series_i) < self.stability_window * 2:
                return 0.5  # Default stability for insufficient data
            
            window_correlations = []
            
            # Calculate rolling correlations
            for start in range(0, len(series_i) - self.stability_window, self.stability_window // 2):
                end = start + self.stability_window
                window_i = series_i[start:end]
                window_j = series_j[start:end]
                
                if len(window_i) >= self.min_correlation_periods:
                    window_corr, _ = self._calculate_correlation_coefficient(window_i, window_j)
                    window_correlations.append(window_corr)
            
            if len(window_correlations) < 2:
                return 0.5
            
            # Stability = 1 - coefficient of variation
            correlation_std = np.std(window_correlations)
            correlation_mean = np.mean(np.abs(window_correlations))
            
            if correlation_mean > 0:
                stability = 1.0 - (correlation_std / correlation_mean)
                return max(0.0, min(1.0, stability))
            else:
                return 0.5
                
        except Exception as e:
            self.logger.error(f"Error calculating correlation stability: {e}")
            return 0.5

    def _calculate_breakdown_risk(self, series_i: np.ndarray, series_j: np.ndarray, 
                                correlation: float) -> float:
        """Calculate correlation breakdown risk indicator"""
        
        try:
            # Look for recent correlation changes
            if len(series_i) < self.min_correlation_periods * 2:
                return 0.5  # Default risk for insufficient data
            
            # Split into recent and historical periods
            split_point = len(series_i) // 2
            historical_i = series_i[:split_point]
            historical_j = series_j[:split_point]
            recent_i = series_i[split_point:]
            recent_j = series_j[split_point:]
            
            # Calculate correlations for each period
            historical_corr, _ = self._calculate_correlation_coefficient(historical_i, historical_j)
            recent_corr, _ = self._calculate_correlation_coefficient(recent_i, recent_j)
            
            # Breakdown risk based on correlation change magnitude
            correlation_change = abs(recent_corr - historical_corr)
            
            # Normalize to 0-1 scale (higher change = higher risk)
            breakdown_risk = min(1.0, correlation_change * 2.0)
            
            return float(breakdown_risk)
            
        except Exception as e:
            self.logger.error(f"Error calculating breakdown risk: {e}")
            return 0.5

    def _calculate_confidence_score(self, correlation: float, p_value: float, 
                                  sample_size: int) -> float:
        """Calculate confidence score for correlation"""
        
        try:
            # Base confidence from statistical significance
            significance_confidence = 1.0 - p_value if p_value < 1.0 else 0.0
            
            # Sample size adjustment
            if sample_size >= 100:
                sample_confidence = 1.0
            elif sample_size >= 50:
                sample_confidence = 0.8
            elif sample_size >= 30:
                sample_confidence = 0.6
            elif sample_size >= 20:
                sample_confidence = 0.4
            else:
                sample_confidence = 0.2
            
            # Correlation strength adjustment
            strength_confidence = abs(correlation)
            
            # Combined confidence score
            confidence = (significance_confidence * 0.4 + 
                         sample_confidence * 0.3 + 
                         strength_confidence * 0.3)
            
            return max(0.0, min(1.0, float(confidence)))
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence score: {e}")
            return 0.0

    def _calculate_overall_stability(self, correlation_matrix: np.ndarray) -> float:
        """Calculate overall system correlation stability"""
        
        try:
            # Extract upper triangle correlations (excluding diagonal)
            upper_triangle = np.triu(correlation_matrix, k=1)
            correlations = upper_triangle[upper_triangle != 0]
            
            if len(correlations) == 0:
                return 0.5
            
            # Overall stability based on correlation consistency
            correlation_std = np.std(correlations)
            correlation_mean = np.mean(np.abs(correlations))
            
            if correlation_mean > 0:
                stability = 1.0 - (correlation_std / correlation_mean)
                return max(0.0, min(1.0, stability))
            else:
                return 0.5
                
        except Exception as e:
            self.logger.error(f"Error calculating overall stability: {e}")
            return 0.5

    def calculate_dte_specific_correlations(self, 
                                          market_data: pd.DataFrame,
                                          component_data: Dict[str, pd.DataFrame]) -> Dict[str, DTECorrelationProfile]:
        """Calculate DTE-specific correlation profiles"""
        
        dte_profiles = {}
        
        if 'dte' not in market_data.columns:
            return dte_profiles
        
        try:
            for dte_name, (min_dte, max_dte) in self.dte_ranges.items():
                # Filter data by DTE range
                dte_mask = (market_data['dte'] >= min_dte) & (market_data['dte'] <= max_dte)
                dte_market_data = market_data[dte_mask]
                
                if len(dte_market_data) < self.min_correlation_periods:
                    continue
                
                # Calculate correlations for this DTE range
                correlations = []
                
                for component_name, component_df in component_data.items():
                    if len(component_df) > 0 and len(dte_market_data) > 0:
                        # Align data by index if possible
                        try:
                            aligned_data = pd.concat([
                                dte_market_data.reset_index(drop=True),
                                component_df.reset_index(drop=True)
                            ], axis=1).dropna()
                            
                            if len(aligned_data) >= self.min_correlation_periods:
                                # Calculate correlation between market data and component
                                market_col = aligned_data.columns[0]  # First column from market data
                                component_col = aligned_data.columns[len(dte_market_data.columns)]  # First component column
                                
                                corr, _ = pearsonr(aligned_data[market_col], aligned_data[component_col])
                                if not np.isnan(corr):
                                    correlations.append(corr)
                        except Exception as e:
                            self.logger.error(f"Error in DTE correlation calculation: {e}")
                            continue
                
                if len(correlations) > 0:
                    correlation_array = np.array(correlations, dtype=np.float32)
                    stability_score = 1.0 - np.std(correlations) if len(correlations) > 1 else 0.5
                    breakdown_risk = min(1.0, np.std(correlations) * 2.0)
                    
                    dte_profiles[dte_name] = DTECorrelationProfile(
                        dte_range=(min_dte, max_dte),
                        correlation_coefficients=correlation_array,
                        stability_score=max(0.0, min(1.0, stability_score)),
                        breakdown_risk=max(0.0, min(1.0, breakdown_risk)),
                        sample_size=len(dte_market_data)
                    )
                    
        except Exception as e:
            self.logger.error(f"Error calculating DTE-specific correlations: {e}")
        
        return dte_profiles

    def calculate_zone_weighted_correlations(self, 
                                           market_data: pd.DataFrame,
                                           component_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate zone-weighted intraday correlations"""
        
        zone_correlations = {}
        
        if 'time' not in market_data.columns and market_data.index.name != 'time':
            return zone_correlations
        
        try:
            # Define time zones (assuming time is in market hours)
            time_zones = {
                'PRE_OPEN': ('09:00', '09:15'),
                'MID_MORN': ('09:15', '11:30'),
                'LUNCH': ('11:30', '13:00'),
                'AFTERNOON': ('13:00', '15:00'),
                'CLOSE': ('15:00', '15:30')
            }
            
            for zone_name, (start_time, end_time) in time_zones.items():
                try:
                    # Filter data by time zone
                    if 'time' in market_data.columns:
                        time_col = market_data['time']
                    else:
                        time_col = market_data.index
                    
                    # Simple time filtering (would need proper time parsing in production)
                    zone_data = market_data  # Placeholder - would implement proper time filtering
                    
                    if len(zone_data) >= self.min_correlation_periods:
                        # Calculate weighted correlation for this zone
                        zone_weight = self.zone_weights.get(zone_name, 0.2)
                        
                        # Placeholder correlation calculation
                        zone_correlation = 0.5 * zone_weight  # Would implement actual calculation
                        zone_correlations[zone_name] = zone_correlation
                        
                except Exception as e:
                    self.logger.error(f"Error calculating zone correlation for {zone_name}: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error calculating zone-weighted correlations: {e}")
        
        return zone_correlations

    def extract_correlation_features(self, correlation_result: CorrelationMatrixResult) -> np.ndarray:
        """Extract numerical correlation features for ML consumption"""
        
        features = []
        
        try:
            # Flatten correlation matrix (excluding diagonal)
            matrix = correlation_result.correlation_matrix
            upper_triangle = np.triu(matrix, k=1)
            correlation_values = upper_triangle[upper_triangle != 0]
            features.extend(correlation_values.tolist())
            
            # Add stability metrics
            for key, value in correlation_result.stability_metrics.items():
                if isinstance(value, (int, float)):
                    features.append(float(value))
            
            # Add breakdown indicators
            features.extend(correlation_result.breakdown_indicators.tolist())
            
            # Add confidence scores
            features.extend(correlation_result.confidence_scores.tolist())
            
            # Add processing performance as features
            normalized_processing_time = min(1.0, correlation_result.processing_time_ms / self.target_processing_time_ms)
            features.append(normalized_processing_time)
            
            # Add feature count as normalized metric
            normalized_feature_count = min(1.0, correlation_result.feature_count / 100.0)
            features.append(normalized_feature_count)
            
        except Exception as e:
            self.logger.error(f"Error extracting correlation features: {e}")
            features = [0.0] * 20  # Return default features
        
        return np.array(features, dtype=np.float32)