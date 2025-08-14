"""
OI Data Quality Handler Module

Handles data quality issues in production OI data:
- Validates OI coverage against 99.98% requirement
- Interpolates missing OI values using smart strategies
- Generates data quality reports and metrics
- Applies fallback strategies for edge cases
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import interpolate
import warnings

logger = logging.getLogger(__name__)


class InterpolationMethod(Enum):
    """Interpolation methods for missing data"""
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    MEAN = "mean"
    ZERO = "zero"


@dataclass
class DataQualityMetrics:
    """Data quality assessment metrics"""
    total_rows: int
    missing_ce_oi: int
    missing_pe_oi: int
    missing_volume: int
    coverage_ce_oi: float
    coverage_pe_oi: float
    coverage_volume: float
    interpolated_values: int
    outliers_detected: int
    schema_compliance: float
    data_completeness_score: float
    interpolation_confidence: float
    outlier_detection_rate: float
    processing_reliability_metric: float


class OIDataQualityHandler:
    """
    Handles data quality issues in OI data with smart interpolation and fallback strategies
    """
    
    def __init__(self, missing_threshold: float = 0.0002):
        """
        Initialize OI Data Quality Handler
        
        Args:
            missing_threshold: Maximum acceptable missing data ratio (0.02% = 0.0002)
        """
        self.missing_threshold = missing_threshold
        self.quality_history = []
        self.interpolation_methods = {
            'ce_oi': InterpolationMethod.LINEAR,
            'pe_oi': InterpolationMethod.LINEAR,
            'ce_volume': InterpolationMethod.FORWARD_FILL,
            'pe_volume': InterpolationMethod.FORWARD_FILL
        }
        logger.info(f"Initialized OIDataQualityHandler with {missing_threshold*100:.2f}% missing threshold")
    
    def validate_oi_coverage(self, df: pd.DataFrame) -> DataQualityMetrics:
        """
        Validate OI data coverage against 99.98% requirement
        
        Args:
            df: Production data to validate
            
        Returns:
            Data quality metrics
        """
        metrics = self._calculate_quality_metrics(df)
        
        # Check against threshold
        if metrics.coverage_ce_oi < (1 - self.missing_threshold):
            logger.warning(f"CE OI coverage {metrics.coverage_ce_oi:.2%} below required {(1-self.missing_threshold):.2%}")
        
        if metrics.coverage_pe_oi < (1 - self.missing_threshold):
            logger.warning(f"PE OI coverage {metrics.coverage_pe_oi:.2%} below required {(1-self.missing_threshold):.2%}")
        
        # Store for historical tracking
        self.quality_history.append(metrics)
        
        return metrics
    
    def _calculate_quality_metrics(self, df: pd.DataFrame) -> DataQualityMetrics:
        """Calculate comprehensive data quality metrics"""
        total_rows = len(df)
        
        # Count missing values
        missing_ce_oi = df['ce_oi'].isna().sum() if 'ce_oi' in df.columns else total_rows
        missing_pe_oi = df['pe_oi'].isna().sum() if 'pe_oi' in df.columns else total_rows
        missing_ce_vol = df['ce_volume'].isna().sum() if 'ce_volume' in df.columns else total_rows
        missing_pe_vol = df['pe_volume'].isna().sum() if 'pe_volume' in df.columns else total_rows
        
        # Calculate coverage
        coverage_ce_oi = 1 - (missing_ce_oi / total_rows) if total_rows > 0 else 0
        coverage_pe_oi = 1 - (missing_pe_oi / total_rows) if total_rows > 0 else 0
        coverage_volume = 1 - ((missing_ce_vol + missing_pe_vol) / (2 * total_rows)) if total_rows > 0 else 0
        
        # Detect outliers
        outliers = self._detect_outliers(df)
        
        # Calculate composite scores
        data_completeness = (coverage_ce_oi + coverage_pe_oi + coverage_volume) / 3
        schema_compliance = self._calculate_schema_compliance(df)
        interpolation_confidence = self._calculate_interpolation_confidence(missing_ce_oi + missing_pe_oi, total_rows)
        outlier_rate = len(outliers) / total_rows if total_rows > 0 else 0
        
        # Processing reliability (composite metric)
        reliability = (data_completeness * 0.4 + 
                      schema_compliance * 0.3 + 
                      interpolation_confidence * 0.2 + 
                      (1 - outlier_rate) * 0.1)
        
        return DataQualityMetrics(
            total_rows=total_rows,
            missing_ce_oi=missing_ce_oi,
            missing_pe_oi=missing_pe_oi,
            missing_volume=missing_ce_vol + missing_pe_vol,
            coverage_ce_oi=coverage_ce_oi,
            coverage_pe_oi=coverage_pe_oi,
            coverage_volume=coverage_volume,
            interpolated_values=0,  # Will be updated during interpolation
            outliers_detected=len(outliers),
            schema_compliance=schema_compliance,
            data_completeness_score=data_completeness,
            interpolation_confidence=interpolation_confidence,
            outlier_detection_rate=outlier_rate,
            processing_reliability_metric=reliability
        )
    
    def _detect_outliers(self, df: pd.DataFrame) -> List[int]:
        """Detect outliers in OI data using IQR method"""
        outlier_indices = []
        
        for column in ['ce_oi', 'pe_oi', 'ce_volume', 'pe_volume']:
            if column in df.columns:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index.tolist()
                outlier_indices.extend(outliers)
        
        return list(set(outlier_indices))
    
    def _calculate_schema_compliance(self, df: pd.DataFrame) -> float:
        """Calculate schema compliance score"""
        required_columns = ['ce_oi', 'pe_oi', 'ce_volume', 'pe_volume', 
                           'call_strike_type', 'put_strike_type', 'dte', 'trade_time']
        
        present_columns = sum(1 for col in required_columns if col in df.columns)
        return present_columns / len(required_columns)
    
    def _calculate_interpolation_confidence(self, missing_count: int, total_count: int) -> float:
        """Calculate confidence in interpolation based on missing data ratio"""
        if total_count == 0:
            return 0.0
        
        missing_ratio = missing_count / total_count
        
        if missing_ratio == 0:
            return 1.0  # No interpolation needed
        elif missing_ratio < 0.001:  # Less than 0.1%
            return 0.95
        elif missing_ratio < 0.01:  # Less than 1%
            return 0.85
        elif missing_ratio < 0.05:  # Less than 5%
            return 0.70
        else:
            return 0.50  # Low confidence for high missing ratio
    
    def interpolate_missing_values(self, df: pd.DataFrame, 
                                  method: str = 'linear') -> Tuple[pd.DataFrame, int]:
        """
        Interpolate missing OI values using specified method
        
        Args:
            df: Data with missing values
            method: Interpolation method to use
            
        Returns:
            Tuple of (interpolated dataframe, number of interpolated values)
        """
        df_interpolated = df.copy()
        total_interpolated = 0
        
        for column in ['ce_oi', 'pe_oi', 'ce_volume', 'pe_volume']:
            if column not in df_interpolated.columns:
                continue
            
            missing_mask = df_interpolated[column].isna()
            missing_count = missing_mask.sum()
            
            if missing_count == 0:
                continue
            
            # Choose interpolation method
            if method == 'linear':
                interpolated = self._linear_interpolation(df_interpolated[column])
            elif method == 'polynomial':
                interpolated = self._polynomial_interpolation(df_interpolated[column])
            elif method == 'forward_fill':
                interpolated = df_interpolated[column].fillna(method='ffill')
            elif method == 'backward_fill':
                interpolated = df_interpolated[column].fillna(method='bfill')
            elif method == 'mean':
                interpolated = df_interpolated[column].fillna(df_interpolated[column].mean())
            else:  # zero fill
                interpolated = df_interpolated[column].fillna(0)
            
            # Apply interpolation
            df_interpolated[column] = interpolated
            total_interpolated += missing_count
            
            if missing_count > 0:
                logger.info(f"Interpolated {missing_count} missing values in {column} using {method}")
        
        return df_interpolated, total_interpolated
    
    def _linear_interpolation(self, series: pd.Series) -> pd.Series:
        """Apply linear interpolation to series"""
        # First try pandas interpolation
        interpolated = series.interpolate(method='linear', limit_direction='both')
        
        # If still has NaN at edges, use forward/backward fill
        interpolated = interpolated.fillna(method='ffill').fillna(method='bfill')
        
        # If still has NaN (all values missing), fill with 0
        interpolated = interpolated.fillna(0)
        
        return interpolated
    
    def _polynomial_interpolation(self, series: pd.Series, order: int = 2) -> pd.Series:
        """Apply polynomial interpolation to series"""
        if series.notna().sum() < order + 1:
            # Not enough points for polynomial interpolation
            return self._linear_interpolation(series)
        
        try:
            # Get valid data points
            valid_mask = series.notna()
            x_valid = np.where(valid_mask)[0]
            y_valid = series[valid_mask].values
            
            # Create polynomial interpolator
            poly = np.poly1d(np.polyfit(x_valid, y_valid, order))
            
            # Interpolate missing values
            interpolated = series.copy()
            missing_mask = series.isna()
            x_missing = np.where(missing_mask)[0]
            
            if len(x_missing) > 0:
                interpolated.iloc[x_missing] = poly(x_missing)
            
            return interpolated
        except Exception as e:
            logger.warning(f"Polynomial interpolation failed: {e}, falling back to linear")
            return self._linear_interpolation(series)
    
    def generate_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report
        
        Args:
            df: Data to analyze
            
        Returns:
            Quality report dictionary
        """
        metrics = self.validate_oi_coverage(df)
        
        report = {
            'summary': {
                'total_rows': metrics.total_rows,
                'data_completeness': f"{metrics.data_completeness_score:.2%}",
                'processing_reliability': f"{metrics.processing_reliability_metric:.2%}",
                'schema_compliance': f"{metrics.schema_compliance:.2%}"
            },
            'coverage': {
                'ce_oi': f"{metrics.coverage_ce_oi:.4%}",
                'pe_oi': f"{metrics.coverage_pe_oi:.4%}",
                'volume': f"{metrics.coverage_volume:.4%}"
            },
            'missing_data': {
                'ce_oi_missing': metrics.missing_ce_oi,
                'pe_oi_missing': metrics.missing_pe_oi,
                'volume_missing': metrics.missing_volume,
                'total_missing_ratio': f"{(metrics.missing_ce_oi + metrics.missing_pe_oi) / (2 * metrics.total_rows):.4%}"
            },
            'quality_scores': {
                'interpolation_confidence': f"{metrics.interpolation_confidence:.2%}",
                'outlier_rate': f"{metrics.outlier_detection_rate:.2%}",
                'outliers_detected': metrics.outliers_detected
            },
            'recommendations': self._generate_recommendations(metrics)
        }
        
        return report
    
    def _generate_recommendations(self, metrics: DataQualityMetrics) -> List[str]:
        """Generate recommendations based on quality metrics"""
        recommendations = []
        
        if metrics.coverage_ce_oi < 0.99:
            recommendations.append("CE OI coverage below 99%, consider data source validation")
        
        if metrics.coverage_pe_oi < 0.99:
            recommendations.append("PE OI coverage below 99%, consider data source validation")
        
        if metrics.outlier_detection_rate > 0.05:
            recommendations.append("High outlier rate detected, review data cleaning procedures")
        
        if metrics.interpolation_confidence < 0.8:
            recommendations.append("Low interpolation confidence, consider alternative data sources")
        
        if metrics.schema_compliance < 1.0:
            recommendations.append("Schema non-compliance detected, verify data pipeline")
        
        if not recommendations:
            recommendations.append("Data quality meets all requirements")
        
        return recommendations
    
    def apply_fallback_strategies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fallback strategies for edge cases
        
        Args:
            df: Data requiring fallback handling
            
        Returns:
            Data with fallback strategies applied
        """
        df_processed = df.copy()
        
        # Strategy 1: Handle completely missing columns
        required_columns = {
            'ce_oi': 0,
            'pe_oi': 0,
            'ce_volume': 0,
            'pe_volume': 0
        }
        
        for col, default_value in required_columns.items():
            if col not in df_processed.columns:
                logger.warning(f"Column {col} missing, adding with default value {default_value}")
                df_processed[col] = default_value
        
        # Strategy 2: Handle extreme outliers
        df_processed = self._handle_extreme_outliers(df_processed)
        
        # Strategy 3: Ensure non-negative values
        for col in ['ce_oi', 'pe_oi', 'ce_volume', 'pe_volume']:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].clip(lower=0)
        
        # Strategy 4: Handle strike type missing values
        if 'call_strike_type' in df_processed.columns:
            df_processed['call_strike_type'] = df_processed['call_strike_type'].fillna('UNKNOWN')
        
        if 'put_strike_type' in df_processed.columns:
            df_processed['put_strike_type'] = df_processed['put_strike_type'].fillna('UNKNOWN')
        
        # Strategy 5: Time series continuity
        df_processed = self._ensure_time_series_continuity(df_processed)
        
        return df_processed
    
    def _handle_extreme_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle extreme outliers by capping or replacing"""
        df_processed = df.copy()
        
        for column in ['ce_oi', 'pe_oi', 'ce_volume', 'pe_volume']:
            if column not in df_processed.columns:
                continue
            
            # Calculate bounds
            Q1 = df_processed[column].quantile(0.01)
            Q99 = df_processed[column].quantile(0.99)
            
            # Cap extreme values
            df_processed[column] = df_processed[column].clip(lower=Q1/10, upper=Q99*10)
            
            # Replace infinite values
            df_processed[column] = df_processed[column].replace([np.inf, -np.inf], np.nan)
            
            # Fill remaining NaN
            df_processed[column] = df_processed[column].fillna(df_processed[column].median())
        
        return df_processed
    
    def _ensure_time_series_continuity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure time series continuity in data"""
        if 'trade_time' not in df.columns:
            return df
        
        df_processed = df.copy()
        
        # Sort by time
        df_processed = df_processed.sort_values('trade_time')
        
        # Check for time gaps
        time_diff = pd.to_datetime(df_processed['trade_time']).diff()
        
        # Identify large gaps (> 5 minutes)
        large_gaps = time_diff > pd.Timedelta(minutes=5)
        
        if large_gaps.any():
            logger.warning(f"Found {large_gaps.sum()} time gaps > 5 minutes in data")
            
            # Mark discontinuities for special handling
            df_processed['time_gap'] = large_gaps
        
        return df_processed
    
    def clean_and_validate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, DataQualityMetrics]:
        """
        Complete cleaning and validation pipeline
        
        Args:
            df: Raw production data
            
        Returns:
            Tuple of (cleaned data, quality metrics)
        """
        # Step 1: Validate coverage
        initial_metrics = self.validate_oi_coverage(df)
        
        # Step 2: Apply fallback strategies
        df_fallback = self.apply_fallback_strategies(df)
        
        # Step 3: Interpolate missing values
        df_interpolated, num_interpolated = self.interpolate_missing_values(df_fallback)
        
        # Step 4: Final validation
        final_metrics = self.validate_oi_coverage(df_interpolated)
        final_metrics.interpolated_values = num_interpolated
        
        # Step 5: Generate quality report
        report = self.generate_quality_report(df_interpolated)
        
        # Log summary
        logger.info(f"Data quality pipeline complete: {final_metrics.data_completeness_score:.2%} completeness, "
                   f"{num_interpolated} values interpolated, {final_metrics.outliers_detected} outliers handled")
        
        return df_interpolated, final_metrics
    
    def extract_quality_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract data quality features for Component 3
        
        Args:
            df: Production data
            
        Returns:
            Array of quality features
        """
        metrics = self.validate_oi_coverage(df)
        
        features = [
            metrics.data_completeness_score,
            metrics.interpolation_confidence,
            metrics.outlier_detection_rate,
            metrics.schema_compliance,
            metrics.processing_reliability_metric,
            metrics.coverage_ce_oi,
            metrics.coverage_pe_oi,
            metrics.coverage_volume,
            min(metrics.interpolated_values / 100, 1.0),  # Normalized interpolation count
            min(metrics.outliers_detected / 100, 1.0)  # Normalized outlier count
        ]
        
        return np.array(features)