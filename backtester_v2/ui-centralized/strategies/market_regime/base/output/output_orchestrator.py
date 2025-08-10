#!/usr/bin/env python3
"""
Output Orchestrator for Market Regime Analysis
==============================================

Central orchestrator for all output generation in the market regime system.
Coordinates CSV generation, time series output, and parameter injection.

Features:
- Centralized output coordination
- Multiple output format support
- Parallel output generation
- Quality validation and reporting
- Performance optimization
- Output lifecycle management

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from .csv_output_manager import CSVOutputManager
from .time_series_generator import TimeSeriesGenerator
from .parameter_injector import ParameterInjector
from ..common_utils import ErrorHandler, PerformanceTimer

logger = logging.getLogger(__name__)


class OutputOrchestrator:
    """
    Central orchestrator for all market regime analysis output generation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Output Orchestrator"""
        self.config = config
        self.output_dir = Path(config.get('output_directory', 'output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize component managers
        self.csv_manager = CSVOutputManager(config)
        self.time_series_generator = TimeSeriesGenerator(config)
        self.parameter_injector = ParameterInjector(config)
        
        # Initialize utilities
        self.error_handler = ErrorHandler()
        self.performance_timer = PerformanceTimer()
        
        # Output configuration
        self.parallel_generation = config.get('parallel_output_generation', True)
        self.max_workers = config.get('output_max_workers', 2)
        self.output_formats = config.get('output_formats', ['csv', 'json', 'summary'])
        self.quality_validation = config.get('enable_quality_validation', True)
        
        # Output tracking
        self.generated_outputs = []
        self.generation_history = []
        
        logger.info(f"Output Orchestrator initialized with formats: {self.output_formats}")
    
    def generate_complete_output(
        self,
        regime_data: pd.DataFrame,
        market_data: pd.DataFrame,
        parameters: Dict[str, Any],
        symbol: str = 'NIFTY',
        timeframe: str = '1min'
    ) -> Dict[str, Any]:
        """
        Generate complete output suite for market regime analysis
        
        Args:
            regime_data: Market regime analysis results
            market_data: Raw market data (OHLCV)
            parameters: Excel configuration parameters
            symbol: Trading symbol
            timeframe: Data timeframe
            
        Returns:
            Dictionary with all generated output information
        """
        with self.performance_timer.measure('complete_output_generation'):
            try:
                logger.info(f"Starting complete output generation for {symbol} {timeframe}")
                
                # Validate input data
                if not self._validate_input_data(regime_data, market_data, parameters):
                    return self._create_error_result("Input data validation failed")
                
                # Generate time series data with parameters
                enriched_timeseries = self.time_series_generator.generate_time_series(
                    regime_data, market_data, parameters, symbol
                )
                
                if enriched_timeseries.empty:
                    return self._create_error_result("Time series generation failed")
                
                # Inject parameters for full traceability
                parameter_injected_timeseries = self.parameter_injector.inject_parameters(
                    enriched_timeseries, parameters, 'comprehensive'
                )
                
                # Generate outputs based on configuration
                output_results = {}
                
                if self.parallel_generation and len(self.output_formats) > 1:
                    # Parallel output generation
                    output_results = self._generate_outputs_parallel(
                        parameter_injected_timeseries, parameters, symbol, timeframe
                    )
                else:
                    # Sequential output generation
                    output_results = self._generate_outputs_sequential(
                        parameter_injected_timeseries, parameters, symbol, timeframe
                    )
                
                # Validate output quality
                if self.quality_validation:
                    quality_report = self._validate_output_quality(output_results)
                    output_results['quality_report'] = quality_report
                
                # Create generation summary
                generation_summary = self._create_generation_summary(
                    output_results, symbol, timeframe, len(parameter_injected_timeseries)
                )
                
                # Track generated outputs
                self._track_generated_outputs(output_results, generation_summary)
                
                logger.info(f"Complete output generation finished: {generation_summary['summary']}")
                return {
                    'success': True,
                    'outputs': output_results,
                    'summary': generation_summary,
                    'performance': self.performance_timer.get_stats()
                }
                
            except Exception as e:
                error_msg = f"Error in complete output generation: {e}"
                self.error_handler.handle_error(error_msg, e)
                return self._create_error_result(error_msg)
    
    def generate_csv_only(
        self,
        regime_data: pd.DataFrame,
        market_data: pd.DataFrame,
        parameters: Dict[str, Any],
        symbol: str = 'NIFTY',
        timeframe: str = '1min'
    ) -> Dict[str, Any]:
        """
        Generate only CSV output (faster option)
        """
        try:
            # Generate time series
            timeseries = self.time_series_generator.generate_time_series(
                regime_data, market_data, parameters, symbol
            )
            
            if timeseries.empty:
                return self._create_error_result("Time series generation failed")
            
            # Inject parameters
            injected_timeseries = self.parameter_injector.inject_parameters(
                timeseries, parameters, 'essential'
            )
            
            # Generate CSV
            csv_files = self.csv_manager.generate_regime_csv(
                injected_timeseries, parameters, symbol, timeframe
            )
            
            return {
                'success': True,
                'outputs': {'csv_files': csv_files},
                'summary': {
                    'format': 'csv_only',
                    'records': len(injected_timeseries),
                    'files_generated': len(csv_files)
                }
            }
            
        except Exception as e:
            error_msg = f"Error in CSV-only generation: {e}"
            self.error_handler.handle_error(error_msg, e)
            return self._create_error_result(error_msg)
    
    def _validate_input_data(self, regime_data: pd.DataFrame, market_data: pd.DataFrame, parameters: Dict[str, Any]) -> bool:
        """
        Validate input data for output generation
        """
        try:
            # Check DataFrames are not empty
            if regime_data.empty:
                logger.error("Regime data is empty")
                return False
            
            if market_data.empty:
                logger.error("Market data is empty")
                return False
            
            # Check required columns
            required_regime_cols = ['timestamp', 'regime_name']
            for col in required_regime_cols:
                if col not in regime_data.columns:
                    logger.error(f"Missing required regime column: {col}")
                    return False
            
            required_market_cols = ['timestamp', 'close']
            for col in required_market_cols:
                if col not in market_data.columns:
                    logger.error(f"Missing required market column: {col}")
                    return False
            
            # Check parameters
            if not isinstance(parameters, dict):
                logger.error("Parameters must be a dictionary")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating input data: {e}")
            return False
    
    def _generate_outputs_parallel(self, data: pd.DataFrame, parameters: Dict[str, Any], symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Generate outputs in parallel for better performance
        """
        output_results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit output generation tasks
            future_to_format = {}
            
            if 'csv' in self.output_formats:
                future = executor.submit(
                    self.csv_manager.generate_regime_csv,
                    data, parameters, symbol, timeframe
                )
                future_to_format[future] = 'csv'
            
            if 'json' in self.output_formats:
                future = executor.submit(
                    self._generate_json_output,
                    data, parameters, symbol, timeframe
                )
                future_to_format[future] = 'json'
            
            if 'summary' in self.output_formats:
                future = executor.submit(
                    self._generate_summary_output,
                    data, parameters, symbol, timeframe
                )
                future_to_format[future] = 'summary'
            
            # Collect results
            for future in as_completed(future_to_format):
                output_format = future_to_format[future]
                try:
                    result = future.result()
                    output_results[output_format] = result
                    logger.info(f"Generated {output_format} output")
                except Exception as e:
                    logger.error(f"Error generating {output_format} output: {e}")
                    output_results[output_format] = {'error': str(e)}
        
        return output_results
    
    def _generate_outputs_sequential(self, data: pd.DataFrame, parameters: Dict[str, Any], symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Generate outputs sequentially
        """
        output_results = {}
        
        try:
            if 'csv' in self.output_formats:
                csv_result = self.csv_manager.generate_regime_csv(
                    data, parameters, symbol, timeframe
                )
                output_results['csv'] = csv_result
                logger.info("Generated CSV output")
        except Exception as e:
            logger.error(f"Error generating CSV output: {e}")
            output_results['csv'] = {'error': str(e)}
        
        try:
            if 'json' in self.output_formats:
                json_result = self._generate_json_output(
                    data, parameters, symbol, timeframe
                )
                output_results['json'] = json_result
                logger.info("Generated JSON output")
        except Exception as e:
            logger.error(f"Error generating JSON output: {e}")
            output_results['json'] = {'error': str(e)}
        
        try:
            if 'summary' in self.output_formats:
                summary_result = self._generate_summary_output(
                    data, parameters, symbol, timeframe
                )
                output_results['summary'] = summary_result
                logger.info("Generated summary output")
        except Exception as e:
            logger.error(f"Error generating summary output: {e}")
            output_results['summary'] = {'error': str(e)}
        
        return output_results
    
    def _generate_json_output(self, data: pd.DataFrame, parameters: Dict[str, Any], symbol: str, timeframe: str) -> Dict[str, str]:
        """
        Generate JSON output format
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"market_regime_{symbol}_{timeframe}_{timestamp}.json"
            filepath = self.output_dir / filename
            
            # Convert DataFrame to JSON-serializable format
            json_data = {
                'metadata': {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'generation_timestamp': datetime.now().isoformat(),
                    'total_records': len(data),
                    'parameters': parameters
                },
                'data': data.to_dict('records')
            }
            
            # Write JSON file
            with open(filepath, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            
            return {'json_file': str(filepath)}
            
        except Exception as e:
            logger.error(f"Error generating JSON output: {e}")
            return {'error': str(e)}
    
    def _generate_summary_output(self, data: pd.DataFrame, parameters: Dict[str, Any], symbol: str, timeframe: str) -> Dict[str, str]:
        """
        Generate summary output format
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"market_regime_summary_{symbol}_{timeframe}_{timestamp}.json"
            filepath = self.output_dir / filename
            
            # Create summary statistics
            summary_data = {
                'generation_info': {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'generation_timestamp': datetime.now().isoformat(),
                    'total_records': len(data)
                },
                'data_summary': {
                    'columns': list(data.columns),
                    'numeric_columns': list(data.select_dtypes(include=[np.number]).columns),
                    'date_range': {
                        'start': str(data['timestamp'].min()) if 'timestamp' in data.columns else None,
                        'end': str(data['timestamp'].max()) if 'timestamp' in data.columns else None
                    }
                },
                'regime_analysis': {},
                'parameters_summary': {
                    'total_parameters': len(parameters),
                    'key_parameters': {k: v for k, v in parameters.items() if k in ['trading_mode', 'symbol', 'confidence_threshold']}
                }
            }
            
            # Add regime-specific summary if available
            if 'regime_name' in data.columns:
                regime_counts = data['regime_name'].value_counts().to_dict()
                summary_data['regime_analysis'] = {
                    'unique_regimes': len(regime_counts),
                    'regime_distribution': regime_counts,
                    'most_common_regime': max(regime_counts, key=regime_counts.get) if regime_counts else None
                }
            
            # Write summary file
            with open(filepath, 'w') as f:
                json.dump(summary_data, f, indent=2, default=str)
            
            return {'summary_file': str(filepath)}
            
        except Exception as e:
            logger.error(f"Error generating summary output: {e}")
            return {'error': str(e)}
    
    def _validate_output_quality(self, output_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate quality of generated outputs
        """
        quality_report = {
            'validation_timestamp': datetime.now().isoformat(),
            'overall_quality': 'unknown',
            'format_validations': {},
            'issues': []
        }
        
        try:
            successful_formats = 0
            total_formats = len(output_results)
            
            for output_format, result in output_results.items():
                if isinstance(result, dict) and 'error' not in result:
                    # Check if files were actually created
                    if output_format == 'csv' and 'main_csv' in result:
                        csv_path = Path(result['main_csv'])
                        if csv_path.exists() and csv_path.stat().st_size > 0:
                            quality_report['format_validations'][output_format] = 'success'
                            successful_formats += 1
                        else:
                            quality_report['format_validations'][output_format] = 'file_error'
                            quality_report['issues'].append(f"CSV file not created or empty: {result['main_csv']}")
                    
                    elif output_format in ['json', 'summary'] and f'{output_format}_file' in result:
                        file_path = Path(result[f'{output_format}_file'])
                        if file_path.exists() and file_path.stat().st_size > 0:
                            quality_report['format_validations'][output_format] = 'success'
                            successful_formats += 1
                        else:
                            quality_report['format_validations'][output_format] = 'file_error'
                            quality_report['issues'].append(f"{output_format.upper()} file not created or empty")
                    else:
                        quality_report['format_validations'][output_format] = 'success'
                        successful_formats += 1
                else:
                    quality_report['format_validations'][output_format] = 'error'
                    if isinstance(result, dict) and 'error' in result:
                        quality_report['issues'].append(f"{output_format}: {result['error']}")
            
            # Calculate overall quality
            success_ratio = successful_formats / total_formats if total_formats > 0 else 0
            if success_ratio == 1.0:
                quality_report['overall_quality'] = 'excellent'
            elif success_ratio >= 0.8:
                quality_report['overall_quality'] = 'good'
            elif success_ratio >= 0.5:
                quality_report['overall_quality'] = 'fair'
            else:
                quality_report['overall_quality'] = 'poor'
            
            quality_report['success_ratio'] = success_ratio
            
        except Exception as e:
            quality_report['overall_quality'] = 'error'
            quality_report['validation_error'] = str(e)
        
        return quality_report
    
    def _create_generation_summary(self, output_results: Dict[str, Any], symbol: str, timeframe: str, record_count: int) -> Dict[str, Any]:
        """
        Create comprehensive generation summary
        """
        return {
            'generation_timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'timeframe': timeframe,
            'record_count': record_count,
            'formats_generated': list(output_results.keys()),
            'successful_formats': [fmt for fmt, result in output_results.items() if isinstance(result, dict) and 'error' not in result],
            'failed_formats': [fmt for fmt, result in output_results.items() if isinstance(result, dict) and 'error' in result],
            'summary': f"Generated {len(output_results)} output formats for {symbol} {timeframe} with {record_count} records"
        }
    
    def _track_generated_outputs(self, output_results: Dict[str, Any], generation_summary: Dict[str, Any]):
        """
        Track generated outputs for management
        """
        self.generated_outputs.append({
            'timestamp': datetime.now(),
            'outputs': output_results,
            'summary': generation_summary
        })
        
        self.generation_history.append(generation_summary)
        
        # Keep only recent history (last 100 generations)
        if len(self.generation_history) > 100:
            self.generation_history = self.generation_history[-100:]
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """
        Create standardized error result
        """
        return {
            'success': False,
            'error': error_message,
            'timestamp': datetime.now().isoformat(),
            'outputs': {},
            'summary': {'error': error_message}
        }
    
    def get_generation_history(self) -> List[Dict[str, Any]]:
        """
        Get history of output generations
        """
        return self.generation_history.copy()
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """
        Get current status of the output orchestrator
        """
        return {
            'output_directory': str(self.output_dir),
            'supported_formats': self.output_formats,
            'parallel_generation': self.parallel_generation,
            'max_workers': self.max_workers,
            'quality_validation': self.quality_validation,
            'total_generations': len(self.generation_history),
            'performance_stats': self.performance_timer.get_stats()
        }


class OutputOrchestrationError(Exception):
    """Custom exception for output orchestration errors"""
    pass