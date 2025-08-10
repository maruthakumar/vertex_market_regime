#!/usr/bin/env python3
"""
CSV Output Manager for Market Regime Analysis
============================================

Integrates the existing CSV handler into the modular structure with enhanced
functionality for market regime analysis output generation.

Features:
- Integrates existing TimeSeriesCSVHandler
- Enhanced parameter injection from Excel configuration
- Multiple output format support
- Metadata generation and validation
- Performance optimization for large datasets

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import logging
import json

# Import existing CSV handler
from ...csv_handlers.time_series_csv_handler import TimeSeriesCSVHandler
from ..common_utils import ErrorHandler, TimeUtils, MathUtils, CacheUtils

logger = logging.getLogger(__name__)


class CSVOutputManager:
    """
    Enhanced CSV output manager integrating existing handler with modular structure
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize CSV Output Manager"""
        self.config = config
        self.output_dir = Path(config.get('output_directory', 'output/csv'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize existing CSV handler
        self.csv_handler = TimeSeriesCSVHandler(str(self.output_dir))
        
        # Initialize utilities
        self.error_handler = ErrorHandler()
        self.cache_utils = CacheUtils()
        
        # Output configuration
        self.include_metadata = config.get('include_metadata', True)
        self.compress_output = config.get('compress_output', False)
        self.precision = config.get('numeric_precision', 6)
        
        logger.info(f"CSV Output Manager initialized with output directory: {self.output_dir}")
    
    def generate_regime_csv(
        self,
        regime_data: pd.DataFrame,
        parameters: Dict[str, Any],
        symbol: str = 'NIFTY',
        timeframe: str = '1min'
    ) -> Dict[str, str]:
        """
        Generate comprehensive regime analysis CSV output
        
        Args:
            regime_data: DataFrame with regime analysis results
            parameters: Excel configuration parameters
            symbol: Trading symbol
            timeframe: Data timeframe
            
        Returns:
            Dict with output file paths
        """
        try:
            # Enhance data with additional columns for CSV output
            enhanced_data = self._enhance_regime_data(
                regime_data, parameters, symbol, timeframe
            )
            
            # Generate filename with comprehensive naming
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"market_regime_{symbol}_{timeframe}_{timestamp}"
            
            # Generate main CSV using existing handler
            csv_path = self.csv_handler.generate_csv(
                enhanced_data,
                parameters,
                base_filename
            )
            
            output_files = {'main_csv': csv_path}
            
            # Generate additional output files if configured
            if self.config.get('generate_summary', True):
                summary_path = self._generate_summary_csv(
                    enhanced_data, base_filename + "_summary"
                )
                output_files['summary_csv'] = summary_path
            
            if self.config.get('generate_statistics', True):
                stats_path = self._generate_statistics_csv(
                    enhanced_data, parameters, base_filename + "_stats"
                )
                output_files['statistics_csv'] = stats_path
            
            # Generate enhanced metadata
            if self.include_metadata:
                metadata_path = self._generate_enhanced_metadata(
                    enhanced_data, parameters, base_filename + "_metadata.json"
                )
                output_files['metadata'] = metadata_path
            
            logger.info(f"Generated {len(output_files)} output files for {symbol} {timeframe}")
            return output_files
            
        except Exception as e:
            error_msg = f"Error generating regime CSV: {e}"
            self.error_handler.handle_error(error_msg, e)
            return {}
    
    def _enhance_regime_data(
        self,
        data: pd.DataFrame,
        parameters: Dict[str, Any],
        symbol: str,
        timeframe: str
    ) -> pd.DataFrame:
        """
        Enhance regime data with additional columns for comprehensive output
        """
        enhanced_df = data.copy()
        
        # Add basic identification columns
        enhanced_df['symbol'] = symbol
        enhanced_df['timeframe'] = timeframe
        enhanced_df['analysis_timestamp'] = datetime.now()
        
        # Add trading mode information from parameters
        trading_mode = parameters.get('trading_mode', 'hybrid')
        enhanced_df['trading_mode'] = trading_mode
        
        # Add timeframe weights if available
        timeframe_weights = parameters.get('timeframe_weights', {})
        if timeframe_weights:
            enhanced_df['timeframe_weight'] = timeframe_weights.get(timeframe, 1.0)
        
        # Add regime confidence intervals
        if 'confidence_score' in enhanced_df.columns:
            enhanced_df['confidence_category'] = enhanced_df['confidence_score'].apply(
                self._categorize_confidence
            )
        
        # Add directional and volatility components separately
        if 'regime_name' in enhanced_df.columns:
            enhanced_df['directional_component'] = enhanced_df['regime_name'].apply(
                self._extract_directional_component
            )
            enhanced_df['volatility_component'] = enhanced_df['regime_name'].apply(
                self._extract_volatility_component
            )
        
        # Round numeric columns to specified precision
        numeric_cols = enhanced_df.select_dtypes(include=[np.number]).columns
        enhanced_df[numeric_cols] = enhanced_df[numeric_cols].round(self.precision)
        
        return enhanced_df
    
    def _categorize_confidence(self, confidence: float) -> str:
        """
        Categorize confidence score into human-readable categories
        """
        if confidence >= 0.85:
            return 'Very High'
        elif confidence >= 0.75:
            return 'High'
        elif confidence >= 0.60:
            return 'Medium'
        elif confidence >= 0.45:
            return 'Low'
        else:
            return 'Very Low'
    
    def _extract_directional_component(self, regime_name: str) -> str:
        """
        Extract directional component from regime name
        """
        regime_lower = regime_name.lower()
        if 'strong_bullish' in regime_lower:
            return 'Strong Bullish'
        elif 'mild_bullish' in regime_lower:
            return 'Mild Bullish'
        elif 'neutral' in regime_lower:
            return 'Neutral'
        elif 'mild_bearish' in regime_lower:
            return 'Mild Bearish'
        elif 'strong_bearish' in regime_lower:
            return 'Strong Bearish'
        else:
            return 'Unknown'
    
    def _extract_volatility_component(self, regime_name: str) -> str:
        """
        Extract volatility component from regime name
        """
        regime_lower = regime_name.lower()
        if 'high_vol' in regime_lower:
            return 'High Volatility'
        elif 'normal_vol' in regime_lower:
            return 'Normal Volatility'
        elif 'low_vol' in regime_lower:
            return 'Low Volatility'
        else:
            return 'Unknown'
    
    def _generate_summary_csv(self, data: pd.DataFrame, filename: str) -> str:
        """
        Generate summary CSV with regime statistics
        """
        try:
            # Calculate regime distribution
            regime_summary = data.groupby('regime_name').agg({
                'confidence_score': ['count', 'mean', 'std'],
                'final_score': ['mean', 'min', 'max'],
                'timestamp': ['min', 'max']
            })
            
            # Flatten column names
            regime_summary.columns = ['_'.join(col).strip() for col in regime_summary.columns.values]
            
            # Add percentage distribution
            total_records = len(data)
            regime_summary['percentage'] = (regime_summary['confidence_score_count'] / total_records * 100).round(2)
            
            # Save summary
            summary_path = self.output_dir / f"{filename}.csv"
            regime_summary.to_csv(summary_path)
            
            return str(summary_path)
            
        except Exception as e:
            logger.error(f"Error generating summary CSV: {e}")
            return ""
    
    def _generate_statistics_csv(self, data: pd.DataFrame, parameters: Dict[str, Any], filename: str) -> str:
        """
        Generate detailed statistics CSV
        """
        try:
            stats_data = []
            
            # Basic statistics
            stats_data.append({
                'metric': 'Total Records',
                'value': len(data),
                'description': 'Total number of data points analyzed'
            })
            
            # Confidence statistics
            if 'confidence_score' in data.columns:
                stats_data.extend([
                    {
                        'metric': 'Average Confidence',
                        'value': round(data['confidence_score'].mean(), 4),
                        'description': 'Mean confidence score across all regimes'
                    },
                    {
                        'metric': 'Min Confidence',
                        'value': round(data['confidence_score'].min(), 4),
                        'description': 'Minimum confidence score'
                    },
                    {
                        'metric': 'Max Confidence',
                        'value': round(data['confidence_score'].max(), 4),
                        'description': 'Maximum confidence score'
                    }
                ])
            
            # Parameter statistics
            for key, value in parameters.items():
                if isinstance(value, (int, float)):
                    stats_data.append({
                        'metric': f'Parameter: {key}',
                        'value': value,
                        'description': f'Configuration parameter: {key}'
                    })
            
            # Create DataFrame and save
            stats_df = pd.DataFrame(stats_data)
            stats_path = self.output_dir / f"{filename}.csv"
            stats_df.to_csv(stats_path, index=False)
            
            return str(stats_path)
            
        except Exception as e:
            logger.error(f"Error generating statistics CSV: {e}")
            return ""
    
    def _generate_enhanced_metadata(self, data: pd.DataFrame, parameters: Dict[str, Any], filename: str) -> str:
        """
        Generate enhanced metadata file
        """
        try:
            metadata = {
                'generation_info': {
                    'timestamp': datetime.now().isoformat(),
                    'version': '2.0.0',
                    'generator': 'Market Regime CSV Output Manager'
                },
                'data_info': {
                    'total_records': len(data),
                    'columns': list(data.columns),
                    'date_range': {
                        'start': data['timestamp'].min() if 'timestamp' in data.columns else None,
                        'end': data['timestamp'].max() if 'timestamp' in data.columns else None
                    },
                    'unique_regimes': data['regime_name'].nunique() if 'regime_name' in data.columns else 0
                },
                'configuration': {
                    'parameters': parameters,
                    'output_settings': {
                        'precision': self.precision,
                        'include_metadata': self.include_metadata,
                        'compress_output': self.compress_output
                    }
                },
                'quality_metrics': {
                    'avg_confidence': float(data['confidence_score'].mean()) if 'confidence_score' in data.columns else None,
                    'regime_distribution': data['regime_name'].value_counts().to_dict() if 'regime_name' in data.columns else {},
                    'data_completeness': float((1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100)
                }
            }
            
            metadata_path = self.output_dir / filename
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            return str(metadata_path)
            
        except Exception as e:
            logger.error(f"Error generating enhanced metadata: {e}")
            return ""
    
    def get_output_summary(self) -> Dict[str, Any]:
        """
        Get summary of generated output files
        """
        output_files = list(self.output_dir.glob('*.csv'))
        metadata_files = list(self.output_dir.glob('*.json'))
        
        return {
            'output_directory': str(self.output_dir),
            'csv_files_count': len(output_files),
            'metadata_files_count': len(metadata_files),
            'total_files': len(output_files) + len(metadata_files),
            'latest_files': {
                'csv': str(max(output_files, key=lambda p: p.stat().st_mtime)) if output_files else None,
                'metadata': str(max(metadata_files, key=lambda p: p.stat().st_mtime)) if metadata_files else None
            }
        }


class CSVValidationError(Exception):
    """Custom exception for CSV validation errors"""
    pass