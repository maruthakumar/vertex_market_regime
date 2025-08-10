"""
Data Pipeline - Centralized Data Processing Pipeline
=================================================

Centralized data processing and validation pipeline for market regime analysis.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod

# Import base utilities
from ..base.common_utils import DataValidator, MathUtils, TimeUtils, OptionUtils, ErrorHandler

logger = logging.getLogger(__name__)


class DataProcessor(ABC):
    """Base class for data processors"""
    
    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process data"""
        pass
    
    @abstractmethod
    def validate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data"""
        pass


class OptionDataProcessor(DataProcessor):
    """Option data processor"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.option_utils = OptionUtils()
        self.data_validator = DataValidator()
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process option data"""
        try:
            processed_data = data.copy()
            
            # Filter liquid options
            processed_data = self.option_utils.filter_liquid_options(
                processed_data,
                min_volume=self.config.get('min_volume', 10),
                min_oi=self.config.get('min_oi', 50)
            )
            
            # Calculate derived fields
            if 'spot' in processed_data.columns:
                processed_data['moneyness'] = processed_data.apply(
                    lambda row: self.option_utils.calculate_moneyness(row['strike'], row['spot']),
                    axis=1
                )
                
                processed_data['option_position'] = processed_data.apply(
                    lambda row: self.option_utils.classify_option_position(
                        row['strike'], row['spot'], row['option_type']
                    ),
                    axis=1
                )
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing option data: {e}")
            return data
    
    def validate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate option data"""
        return self.data_validator.validate_option_data(data)


class UnderlyingDataProcessor(DataProcessor):
    """Underlying asset data processor"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.math_utils = MathUtils()
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process underlying data"""
        try:
            processed_data = data.copy()
            
            # Calculate returns
            if 'close' in processed_data.columns:
                processed_data['returns'] = processed_data['close'].pct_change()
                processed_data['log_returns'] = np.log(processed_data['close'] / processed_data['close'].shift(1))
            
            # Calculate volatility
            if 'returns' in processed_data.columns:
                volatility_window = self.config.get('volatility_window', 20)
                processed_data['volatility'] = processed_data['returns'].rolling(
                    window=volatility_window
                ).std() * np.sqrt(252)
            
            # Calculate volume ratios
            if 'volume' in processed_data.columns:
                volume_window = self.config.get('volume_window', 20)
                processed_data['volume_ma'] = processed_data['volume'].rolling(window=volume_window).mean()
                processed_data['volume_ratio'] = processed_data['volume'] / processed_data['volume_ma']
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing underlying data: {e}")
            return data
    
    def validate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate underlying data"""
        try:
            validation = {
                'is_valid': True,
                'errors': [],
                'warnings': [],
                'data_quality_score': 1.0
            }
            
            required_columns = ['close']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                validation['is_valid'] = False
                validation['errors'].append(f"Missing required columns: {missing_columns}")
                validation['data_quality_score'] -= 0.5
            
            if data.empty:
                validation['is_valid'] = False
                validation['errors'].append("DataFrame is empty")
                validation['data_quality_score'] = 0.0
            
            return validation
            
        except Exception as e:
            logger.error(f"Error validating underlying data: {e}")
            return {'is_valid': False, 'errors': [str(e)]}


class DataPipeline:
    """Centralized data processing pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Data Pipeline"""
        self.config = config
        self.processors = {}
        self.cache_enabled = config.get('cache_enabled', True)
        self.cache_ttl = config.get('cache_ttl', 300)  # 5 minutes
        
        # Data quality thresholds
        self.quality_thresholds = config.get('quality_thresholds', {
            'min_data_points': 10,
            'max_missing_ratio': 0.2,
            'min_quality_score': 0.6
        })
        
        # Initialize processors
        self._initialize_processors()
        
        # Processing metrics
        self.processing_metrics = {
            'total_processes': 0,
            'successful_processes': 0,
            'failed_processes': 0,
            'avg_processing_time': 0.0,
            'data_quality_scores': []
        }
        
        # Data cache
        self.data_cache = {}
        self.cache_timestamps = {}
        
        # Utilities
        self.data_validator = DataValidator()
        self.math_utils = MathUtils()
        self.time_utils = TimeUtils()
        self.error_handler = ErrorHandler()
        
        logger.info("DataPipeline initialized with comprehensive data processing")
    
    def _initialize_processors(self):
        """Initialize data processors"""
        try:
            self.processors['option'] = OptionDataProcessor(self.config.get('option_processor', {}))
            self.processors['underlying'] = UnderlyingDataProcessor(self.config.get('underlying_processor', {}))
            
        except Exception as e:
            logger.error(f"Error initializing processors: {e}")
    
    def process_data(self,
                    data: Dict[str, pd.DataFrame],
                    processing_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process input data through the pipeline
        
        Args:
            data: Dictionary of DataFrames (option_data, underlying_data, etc.)
            processing_options: Optional processing configuration
            
        Returns:
            Dict with processed data and metadata
        """
        start_time = datetime.now()
        
        try:
            processing_options = processing_options or {}
            
            # Generate cache key
            cache_key = self._generate_cache_key(data, processing_options)
            
            # Check cache
            if self.cache_enabled and self._is_cache_valid(cache_key):
                cached_result = self.data_cache[cache_key]
                logger.info("Returning cached processed data")
                return cached_result
            
            result = {
                'processed_data': {},
                'validation_results': {},
                'processing_metadata': {},
                'data_quality': {},
                'status': 'success'
            }
            
            # Process each data type
            for data_type, data_frame in data.items():
                try:
                    # Validate input data
                    validation_result = self._validate_input_data(data_frame, data_type)
                    result['validation_results'][data_type] = validation_result
                    
                    if not validation_result['is_valid']:
                        logger.warning(f"Invalid {data_type} data: {validation_result['errors']}")
                        continue
                    
                    # Process data
                    processed_data = self._process_data_type(data_frame, data_type, processing_options)
                    result['processed_data'][data_type] = processed_data
                    
                    # Calculate data quality metrics
                    quality_metrics = self._calculate_data_quality(processed_data, data_type)
                    result['data_quality'][data_type] = quality_metrics
                    
                    # Generate processing metadata
                    metadata = self._generate_processing_metadata(data_frame, processed_data, data_type)
                    result['processing_metadata'][data_type] = metadata
                    
                except Exception as e:
                    logger.error(f"Error processing {data_type} data: {e}")
                    result['validation_results'][data_type] = {'is_valid': False, 'error': str(e)}
                    result['status'] = 'partial_success'
            
            # Overall data quality assessment
            result['overall_quality'] = self._assess_overall_quality(result['data_quality'])
            
            # Cache result
            if self.cache_enabled:
                self._cache_result(cache_key, result)
            
            # Update processing metrics
            self._update_processing_metrics(start_time, True, result['overall_quality'])
            
            return result
            
        except Exception as e:
            logger.error(f"Error in data pipeline processing: {e}")
            self._update_processing_metrics(start_time, False, 0.0)
            return self._get_default_processing_result()
    
    def _validate_input_data(self, data: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        """Validate input data"""
        try:
            if data_type in self.processors:
                return self.processors[data_type].validate(data)
            else:
                # Generic validation
                validation = {
                    'is_valid': True,
                    'errors': [],
                    'warnings': [],
                    'data_quality_score': 1.0
                }
                
                if data.empty:
                    validation['is_valid'] = False
                    validation['errors'].append("DataFrame is empty")
                    validation['data_quality_score'] = 0.0
                
                # Check data size
                if len(data) < self.quality_thresholds['min_data_points']:
                    validation['warnings'].append(f"Low data point count: {len(data)}")
                    validation['data_quality_score'] -= 0.2
                
                # Check for missing values
                missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
                if missing_ratio > self.quality_thresholds['max_missing_ratio']:
                    validation['warnings'].append(f"High missing value ratio: {missing_ratio:.2%}")
                    validation['data_quality_score'] -= 0.3
                
                return validation
                
        except Exception as e:
            logger.error(f"Error validating {data_type} data: {e}")
            return {'is_valid': False, 'errors': [str(e)]}
    
    def _process_data_type(self,
                          data: pd.DataFrame,
                          data_type: str,
                          processing_options: Dict[str, Any]) -> pd.DataFrame:
        """Process specific data type"""
        try:
            if data_type in self.processors:
                return self.processors[data_type].process(data)
            else:
                # Generic processing
                processed_data = data.copy()
                
                # Basic cleaning
                processed_data = self._clean_data(processed_data)
                
                # Handle missing values
                processed_data = self._handle_missing_values(processed_data, processing_options)
                
                # Apply any generic transformations
                processed_data = self._apply_generic_transformations(processed_data, processing_options)
                
                return processed_data
                
        except Exception as e:
            logger.error(f"Error processing {data_type} data: {e}")
            return data
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply basic data cleaning"""
        try:
            cleaned_data = data.copy()
            
            # Remove duplicate rows
            cleaned_data = cleaned_data.drop_duplicates()
            
            # Remove rows with all NaN values
            cleaned_data = cleaned_data.dropna(how='all')
            
            # Clean numeric columns
            numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                # Remove infinite values
                cleaned_data[col] = cleaned_data[col].replace([np.inf, -np.inf], np.nan)
                
                # Remove extreme outliers (beyond 5 standard deviations)
                if cleaned_data[col].std() > 0:
                    mean_val = cleaned_data[col].mean()
                    std_val = cleaned_data[col].std()
                    outlier_mask = abs(cleaned_data[col] - mean_val) > 5 * std_val
                    cleaned_data.loc[outlier_mask, col] = np.nan
            
            return cleaned_data
            
        except Exception as e:
            logger.error(f"Error cleaning data: {e}")
            return data
    
    def _handle_missing_values(self,
                              data: pd.DataFrame,
                              processing_options: Dict[str, Any]) -> pd.DataFrame:
        """Handle missing values"""
        try:
            missing_strategy = processing_options.get('missing_strategy', 'forward_fill')
            
            if missing_strategy == 'forward_fill':
                return data.fillna(method='ffill')
            elif missing_strategy == 'backward_fill':
                return data.fillna(method='bfill')
            elif missing_strategy == 'interpolate':
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                for col in numeric_columns:
                    data[col] = data[col].interpolate()
                return data
            elif missing_strategy == 'mean_fill':
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                for col in numeric_columns:
                    data[col] = data[col].fillna(data[col].mean())
                return data
            elif missing_strategy == 'drop':
                return data.dropna()
            else:
                return data
                
        except Exception as e:
            logger.error(f"Error handling missing values: {e}")
            return data
    
    def _apply_generic_transformations(self,
                                     data: pd.DataFrame,
                                     processing_options: Dict[str, Any]) -> pd.DataFrame:
        """Apply generic data transformations"""
        try:
            transformations = processing_options.get('transformations', [])
            
            for transformation in transformations:
                if transformation == 'normalize':
                    numeric_columns = data.select_dtypes(include=[np.number]).columns
                    for col in numeric_columns:
                        if data[col].std() > 0:
                            data[col] = (data[col] - data[col].mean()) / data[col].std()
                
                elif transformation == 'log_transform':
                    numeric_columns = data.select_dtypes(include=[np.number]).columns
                    for col in numeric_columns:
                        if (data[col] > 0).all():
                            data[col] = np.log(data[col])
                
                elif transformation == 'winsorize':
                    numeric_columns = data.select_dtypes(include=[np.number]).columns
                    for col in numeric_columns:
                        lower_percentile = data[col].quantile(0.01)
                        upper_percentile = data[col].quantile(0.99)
                        data[col] = np.clip(data[col], lower_percentile, upper_percentile)
            
            return data
            
        except Exception as e:
            logger.error(f"Error applying transformations: {e}")
            return data
    
    def _calculate_data_quality(self, data: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        """Calculate data quality metrics"""
        try:
            quality = {
                'overall_score': 1.0,
                'completeness': 0.0,
                'consistency': 0.0,
                'accuracy': 0.0,
                'timeliness': 0.0,
                'issues': []
            }
            
            # Completeness
            total_cells = data.shape[0] * data.shape[1]
            non_null_cells = total_cells - data.isnull().sum().sum()
            quality['completeness'] = non_null_cells / total_cells if total_cells > 0 else 0
            
            # Consistency
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            consistency_scores = []
            
            for col in numeric_columns:
                if data[col].std() > 0:
                    # Check for reasonable value ranges
                    outlier_ratio = (abs(data[col] - data[col].mean()) > 3 * data[col].std()).sum() / len(data)
                    consistency_score = max(0, 1 - outlier_ratio)
                    consistency_scores.append(consistency_score)
            
            quality['consistency'] = np.mean(consistency_scores) if consistency_scores else 1.0
            
            # Accuracy (basic checks)
            accuracy_score = 1.0
            
            # Check for negative values where they shouldn't be
            if data_type == 'option':
                if 'volume' in data.columns and (data['volume'] < 0).any():
                    accuracy_score -= 0.2
                    quality['issues'].append("Negative volume values found")
                
                if 'oi' in data.columns and (data['oi'] < 0).any():
                    accuracy_score -= 0.2
                    quality['issues'].append("Negative open interest values found")
            
            quality['accuracy'] = max(0, accuracy_score)
            
            # Timeliness (if timestamp columns exist)
            timeliness_score = 1.0
            timestamp_columns = data.select_dtypes(include=['datetime64']).columns
            
            if len(timestamp_columns) > 0:
                for col in timestamp_columns:
                    latest_timestamp = data[col].max()
                    if pd.notna(latest_timestamp):
                        age_hours = (datetime.now() - latest_timestamp).total_seconds() / 3600
                        if age_hours > 24:  # Data older than 24 hours
                            timeliness_score -= 0.3
                            quality['issues'].append(f"Data age: {age_hours:.1f} hours")
            
            quality['timeliness'] = max(0, timeliness_score)
            
            # Overall score
            quality['overall_score'] = np.mean([
                quality['completeness'],
                quality['consistency'], 
                quality['accuracy'],
                quality['timeliness']
            ])
            
            return quality
            
        except Exception as e:
            logger.error(f"Error calculating data quality: {e}")
            return {'overall_score': 0.0, 'issues': [str(e)]}
    
    def _generate_processing_metadata(self,
                                    original_data: pd.DataFrame,
                                    processed_data: pd.DataFrame,
                                    data_type: str) -> Dict[str, Any]:
        """Generate processing metadata"""
        try:
            return {
                'data_type': data_type,
                'original_shape': original_data.shape,
                'processed_shape': processed_data.shape,
                'rows_removed': original_data.shape[0] - processed_data.shape[0],
                'columns_added': processed_data.shape[1] - original_data.shape[1],
                'processing_timestamp': datetime.now(),
                'memory_usage_mb': processed_data.memory_usage(deep=True).sum() / 1024 / 1024,
                'column_types': processed_data.dtypes.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error generating processing metadata: {e}")
            return {'data_type': data_type, 'error': str(e)}
    
    def _assess_overall_quality(self, data_quality: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Assess overall data quality across all data types"""
        try:
            if not data_quality:
                return {'overall_score': 0.0, 'quality_grade': 'F'}
            
            # Calculate weighted average
            quality_scores = [quality.get('overall_score', 0.0) for quality in data_quality.values()]
            overall_score = np.mean(quality_scores)
            
            # Determine quality grade
            if overall_score >= 0.9:
                quality_grade = 'A'
            elif overall_score >= 0.8:
                quality_grade = 'B'
            elif overall_score >= 0.7:
                quality_grade = 'C'
            elif overall_score >= 0.6:
                quality_grade = 'D'
            else:
                quality_grade = 'F'
            
            # Collect all issues
            all_issues = []
            for quality in data_quality.values():
                all_issues.extend(quality.get('issues', []))
            
            return {
                'overall_score': float(overall_score),
                'quality_grade': quality_grade,
                'data_types_processed': len(data_quality),
                'issues_count': len(all_issues),
                'all_issues': all_issues
            }
            
        except Exception as e:
            logger.error(f"Error assessing overall quality: {e}")
            return {'overall_score': 0.0, 'quality_grade': 'F'}
    
    def _generate_cache_key(self, data: Dict[str, pd.DataFrame], processing_options: Dict[str, Any]) -> str:
        """Generate cache key for data processing"""
        try:
            # Create hash from data characteristics and options
            key_components = []
            
            for data_type, data_frame in data.items():
                data_hash = hash(str(data_frame.shape) + str(data_frame.columns.tolist()))
                key_components.append(f"{data_type}:{data_hash}")
            
            options_hash = hash(str(sorted(processing_options.items())))
            key_components.append(f"options:{options_hash}")
            
            return f"data_pipeline_{'_'.join(key_components)}"
            
        except Exception as e:
            logger.error(f"Error generating cache key: {e}")
            return "default_cache_key"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached result is still valid"""
        try:
            if cache_key not in self.data_cache:
                return False
            
            cache_timestamp = self.cache_timestamps.get(cache_key)
            if not cache_timestamp:
                return False
            
            age_seconds = (datetime.now() - cache_timestamp).total_seconds()
            return age_seconds < self.cache_ttl
            
        except Exception as e:
            logger.error(f"Error checking cache validity: {e}")
            return False
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache processing result"""
        try:
            self.data_cache[cache_key] = result
            self.cache_timestamps[cache_key] = datetime.now()
            
            # Clean old cache entries
            self._clean_cache()
            
        except Exception as e:
            logger.error(f"Error caching result: {e}")
    
    def _clean_cache(self):
        """Clean expired cache entries"""
        try:
            current_time = datetime.now()
            expired_keys = []
            
            for cache_key, timestamp in self.cache_timestamps.items():
                if (current_time - timestamp).total_seconds() > self.cache_ttl:
                    expired_keys.append(cache_key)
            
            for key in expired_keys:
                if key in self.data_cache:
                    del self.data_cache[key]
                if key in self.cache_timestamps:
                    del self.cache_timestamps[key]
                    
        except Exception as e:
            logger.error(f"Error cleaning cache: {e}")
    
    def _update_processing_metrics(self, start_time: datetime, success: bool, quality_score: float):
        """Update processing metrics"""
        try:
            self.processing_metrics['total_processes'] += 1
            
            if success:
                self.processing_metrics['successful_processes'] += 1
            else:
                self.processing_metrics['failed_processes'] += 1
            
            # Update execution time
            processing_time = (datetime.now() - start_time).total_seconds()
            current_avg = self.processing_metrics['avg_processing_time']
            total_processes = self.processing_metrics['total_processes']
            
            # Running average
            self.processing_metrics['avg_processing_time'] = (
                (current_avg * (total_processes - 1) + processing_time) / total_processes
            )
            
            # Track quality scores
            self.processing_metrics['data_quality_scores'].append(quality_score)
            
            # Keep only recent quality scores
            if len(self.processing_metrics['data_quality_scores']) > 100:
                self.processing_metrics['data_quality_scores'] = self.processing_metrics['data_quality_scores'][-100:]
                
        except Exception as e:
            logger.error(f"Error updating processing metrics: {e}")
    
    def _get_default_processing_result(self) -> Dict[str, Any]:
        """Get default processing result when processing fails"""
        return {
            'processed_data': {},
            'validation_results': {},
            'processing_metadata': {},
            'data_quality': {},
            'overall_quality': {'overall_score': 0.0, 'quality_grade': 'F'},
            'status': 'failed'
        }
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status"""
        try:
            return {
                'pipeline_status': 'operational',
                'processors_loaded': len(self.processors),
                'cache_enabled': self.cache_enabled,
                'cache_entries': len(self.data_cache),
                'processing_metrics': self.processing_metrics.copy(),
                'average_quality_score': float(np.mean(self.processing_metrics['data_quality_scores'])) if self.processing_metrics['data_quality_scores'] else 0.0,
                'success_rate': (self.processing_metrics['successful_processes'] / self.processing_metrics['total_processes']) if self.processing_metrics['total_processes'] > 0 else 0.0,
                'configuration': {
                    'cache_ttl': self.cache_ttl,
                    'quality_thresholds': self.quality_thresholds
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting pipeline status: {e}")
            return {'pipeline_status': 'error', 'error': str(e)}