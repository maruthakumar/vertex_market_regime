"""
Market Regime Executor

This module provides the main execution engine for market regime detection,
following the same patterns as other backtester_v2 systems (TBS, OI, ORB, etc.).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path

from .archive_enhanced_modules_do_not_use.enhanced_regime_detector import Enhanced18RegimeDetector, Enhanced18RegimeType
from .excel_config_manager import MarketRegimeExcelManager
from .parser import MarketRegimeParser
from .models import RegimeDetectionResult, RegimeConfiguration
from ..live_streaming.streaming_manager import LiveStreamingManager
from ..integration.database_manager import DatabaseManager

logger = logging.getLogger(__name__)

class MarketRegimeExecutor:
    """
    Market Regime Executor - Main execution engine for regime detection
    
    This class follows the same pattern as other backtester_v2 executors,
    providing parsing, query generation, execution, and results processing
    for market regime detection and analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Market Regime Executor
        
        Args:
            config (Dict, optional): Configuration parameters
        """
        self.config = config or {}
        
        # Core components
        self.regime_detector = None
        self.excel_manager = None
        self.parser = MarketRegimeParser()
        self.streaming_manager = None
        self.db_manager = None
        
        # Execution state
        self.current_regime = None
        self.regime_history = []
        self.execution_results = []
        
        # Configuration
        self.regime_config = None
        self.live_mode = False
        
        logger.info("MarketRegimeExecutor initialized")
    
    def parse_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse input configuration for market regime detection
        
        Args:
            input_data (Dict): Input configuration data
            
        Returns:
            Dict: Parsed parameters for regime detection
        """
        try:
            logger.info("Parsing market regime input configuration")
            
            # Parse Excel configuration if provided
            excel_path = input_data.get('excel_config_path')
            if excel_path and Path(excel_path).exists():
                self.excel_manager = MarketRegimeExcelManager(excel_path)
                
                # Validate configuration
                is_valid, errors = self.excel_manager.validate_configuration()
                if not is_valid:
                    raise ValueError(f"Invalid Excel configuration: {errors}")
                
                # Get configuration parameters
                detection_params = self.excel_manager.get_detection_parameters()
                regime_adjustments = self.excel_manager.get_regime_adjustments()
                strategy_mappings = self.excel_manager.get_strategy_mappings()
                live_config = self.excel_manager.get_live_trading_config()
                
                parsed_params = {
                    'detection_parameters': detection_params,
                    'regime_adjustments': regime_adjustments,
                    'strategy_mappings': strategy_mappings,
                    'live_trading_config': live_config,
                    'excel_config_loaded': True
                }
                
                logger.info(f"Excel configuration loaded: {len(detection_params)} parameters")
                
            else:
                # Use default configuration
                parsed_params = self.parser.parse_input(input_data)
                parsed_params['excel_config_loaded'] = False
                
                logger.info("Using default configuration parameters")
            
            # Parse execution parameters
            parsed_params.update({
                'start_date': input_data.get('start_date'),
                'end_date': input_data.get('end_date'),
                'symbols': input_data.get('symbols', ['NIFTY']),
                'timeframes': input_data.get('timeframes', ['1min', '5min', '15min']),
                'live_mode': input_data.get('live_mode', False),
                'backtest_mode': input_data.get('backtest_mode', True),
                'output_format': input_data.get('output_format', 'excel'),
                'save_results': input_data.get('save_results', True)
            })
            
            # Store configuration
            self.regime_config = parsed_params
            self.live_mode = parsed_params['live_mode']
            
            return parsed_params
            
        except Exception as e:
            logger.error(f"Error parsing input: {e}")
            raise
    
    def generate_query(self, params: Dict[str, Any]) -> List[str]:
        """
        Generate database queries for market regime analysis
        
        Args:
            params (Dict): Parsed parameters
            
        Returns:
            List[str]: List of SQL queries for data retrieval
        """
        try:
            logger.info("Generating queries for market regime analysis")
            
            queries = []
            
            # Base market data query
            symbols = params.get('symbols', ['NIFTY'])
            start_date = params.get('start_date')
            end_date = params.get('end_date')
            timeframes = params.get('timeframes', ['1min'])
            
            for symbol in symbols:
                for timeframe in timeframes:
                    # OHLC data query
                    ohlc_query = f"""
                    SELECT 
                        timestamp,
                        open_price as open,
                        high_price as high,
                        low_price as low,
                        close_price as close,
                        volume,
                        '{symbol}' as symbol,
                        '{timeframe}' as timeframe
                    FROM market_data_{timeframe}
                    WHERE symbol = '{symbol}'
                    AND timestamp BETWEEN '{start_date}' AND '{end_date}'
                    ORDER BY timestamp
                    """
                    queries.append(ohlc_query)
                    
                    # Options data query for OI analysis
                    oi_query = f"""
                    SELECT 
                        timestamp,
                        strike_price,
                        option_type,
                        open_interest,
                        volume,
                        implied_volatility,
                        delta,
                        gamma,
                        theta,
                        vega
                    FROM options_data
                    WHERE underlying = '{symbol}'
                    AND timestamp BETWEEN '{start_date}' AND '{end_date}'
                    ORDER BY timestamp, strike_price
                    """
                    queries.append(oi_query)
            
            # Technical indicators query
            tech_query = f"""
            SELECT 
                timestamp,
                symbol,
                rsi_14,
                macd,
                macd_signal,
                bb_upper,
                bb_lower,
                atr_14,
                volume_sma_20
            FROM technical_indicators
            WHERE symbol IN ({','.join([f"'{s}'" for s in symbols])})
            AND timestamp BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY timestamp
            """
            queries.append(tech_query)
            
            logger.info(f"Generated {len(queries)} queries for regime analysis")
            return queries
            
        except Exception as e:
            logger.error(f"Error generating queries: {e}")
            return []
    
    def execute_backtest(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute market regime detection backtest
        
        Args:
            params (Dict): Execution parameters
            
        Returns:
            Dict: Backtest results
        """
        try:
            logger.info("Starting market regime backtest execution")
            
            # Initialize regime detector with configuration
            detection_params = params.get('detection_parameters', {})
            self.regime_detector = Enhanced18RegimeDetector(config=detection_params)
            
            # Generate and execute queries
            queries = self.generate_query(params)
            
            # Initialize database manager
            self.db_manager = DatabaseManager()
            
            # Execute queries and collect data
            market_data = {}
            for query in queries:
                try:
                    result = self.db_manager.execute_query(query)
                    if 'market_data_' in query:
                        timeframe = self._extract_timeframe_from_query(query)
                        symbol = self._extract_symbol_from_query(query)
                        key = f"{symbol}_{timeframe}"
                        market_data[key] = result
                    elif 'options_data' in query:
                        market_data['options'] = result
                    elif 'technical_indicators' in query:
                        market_data['technical'] = result
                except Exception as e:
                    logger.error(f"Error executing query: {e}")
                    continue
            
            # Process market data for regime detection
            regime_results = self._process_market_data_for_regimes(market_data, params)
            
            # Generate execution results
            execution_results = {
                'regime_detection_results': regime_results,
                'configuration': params,
                'execution_summary': self._generate_execution_summary(regime_results),
                'performance_metrics': self._calculate_performance_metrics(regime_results),
                'regime_statistics': self._calculate_regime_statistics(regime_results)
            }
            
            self.execution_results.append(execution_results)
            
            logger.info(f"Backtest execution completed: {len(regime_results)} regime detections")
            return execution_results
            
        except Exception as e:
            logger.error(f"Error in backtest execution: {e}")
            return {'error': str(e)}
    
    def execute_live_trading(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute live market regime detection
        
        Args:
            params (Dict): Execution parameters
            
        Returns:
            Dict: Live execution status
        """
        try:
            logger.info("Starting live market regime detection")
            
            # Initialize regime detector
            detection_params = params.get('detection_parameters', {})
            self.regime_detector = Enhanced18RegimeDetector(config=detection_params)
            
            # Initialize streaming manager
            live_config = params.get('live_trading_config', {})
            streaming_config = {
                'aggregator': {
                    '1min_max_candles': 1440,
                    '5min_max_candles': 288,
                    '15min_max_candles': 96
                },
                'max_reconnection_attempts': 5,
                'enable_performance_logging': True
            }
            
            self.streaming_manager = LiveStreamingManager(config=streaming_config)
            
            # Add Kite streamer
            kite_config = {
                'type': 'kite',
                'regime_config': detection_params,
                'kite_config': {
                    'stream_interval_ms': live_config.get('StreamingIntervalMs', 100),
                    'regime_update_freq_sec': live_config.get('RegimeUpdateFreqSec', 60)
                }
            }
            
            self.streaming_manager.add_streamer('main_kite', kite_config)
            
            # Register callbacks
            self.streaming_manager.add_regime_callback(self._on_live_regime_update)
            self.streaming_manager.add_data_callback(self._on_live_data_update)
            
            # Start streaming
            success = self.streaming_manager.start_streaming()
            
            if success:
                logger.info("Live market regime detection started successfully")
                return {
                    'status': 'running',
                    'start_time': datetime.now().isoformat(),
                    'configuration': params,
                    'streaming_status': self.streaming_manager.get_streaming_status()
                }
            else:
                logger.error("Failed to start live streaming")
                return {
                    'status': 'failed',
                    'error': 'Failed to start streaming'
                }
                
        except Exception as e:
            logger.error(f"Error in live execution: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _process_market_data_for_regimes(self, market_data: Dict[str, Any], 
                                       params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process market data for regime detection"""
        try:
            regime_results = []
            
            # Get primary timeframe data
            primary_timeframe = params.get('timeframes', ['1min'])[0]
            symbols = params.get('symbols', ['NIFTY'])
            
            for symbol in symbols:
                key = f"{symbol}_{primary_timeframe}"
                if key not in market_data:
                    continue
                
                ohlc_data = market_data[key]
                options_data = market_data.get('options', pd.DataFrame())
                technical_data = market_data.get('technical', pd.DataFrame())
                
                # Process each time period
                for index, row in ohlc_data.iterrows():
                    timestamp = row['timestamp']
                    
                    # Prepare market data for regime detection
                    regime_input = self._prepare_regime_input(
                        row, options_data, technical_data, timestamp
                    )
                    
                    # Detect regime
                    regime_result = self.regime_detector.detect_regime(regime_input)
                    
                    # Store result
                    regime_results.append({
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'timeframe': primary_timeframe,
                        'regime_type': regime_result['regime_type'].value,
                        'confidence': regime_result['confidence'],
                        'components': regime_result['components'],
                        'market_data': regime_input
                    })
            
            return regime_results
            
        except Exception as e:
            logger.error(f"Error processing market data for regimes: {e}")
            return []
    
    def _prepare_regime_input(self, ohlc_row: pd.Series, options_data: pd.DataFrame,
                            technical_data: pd.DataFrame, timestamp: datetime) -> Dict[str, Any]:
        """Prepare input data for regime detection"""
        try:
            # Price data
            price_data = [ohlc_row['close']]  # Simplified for single point
            
            # Options data for the timestamp
            timestamp_options = options_data[options_data['timestamp'] == timestamp]
            
            if not timestamp_options.empty:
                # Calculate OI metrics
                call_oi = timestamp_options[timestamp_options['option_type'] == 'CE']['open_interest'].sum()
                put_oi = timestamp_options[timestamp_options['option_type'] == 'PE']['open_interest'].sum()
                call_volume = timestamp_options[timestamp_options['option_type'] == 'CE']['volume'].sum()
                put_volume = timestamp_options[timestamp_options['option_type'] == 'PE']['volume'].sum()
                
                # Calculate Greeks
                avg_delta = timestamp_options['delta'].mean()
                avg_gamma = timestamp_options['gamma'].mean()
                avg_theta = timestamp_options['theta'].mean()
                avg_vega = timestamp_options['vega'].mean()
                avg_iv = timestamp_options['implied_volatility'].mean()
            else:
                # Default values
                call_oi = put_oi = 1000000
                call_volume = put_volume = 50000
                avg_delta = avg_gamma = avg_theta = avg_vega = avg_iv = 0.0
            
            # Technical indicators
            tech_row = technical_data[technical_data['timestamp'] == timestamp]
            if not tech_row.empty:
                tech_row = tech_row.iloc[0]
                rsi = tech_row.get('rsi_14', 50)
                macd = tech_row.get('macd', 0)
                macd_signal = tech_row.get('macd_signal', 0)
                atr = tech_row.get('atr_14', 0)
            else:
                rsi = 50
                macd = macd_signal = atr = 0
            
            # Prepare regime input
            regime_input = {
                'price_data': price_data,
                'oi_data': {
                    'call_oi': call_oi,
                    'put_oi': put_oi,
                    'call_volume': call_volume,
                    'put_volume': put_volume
                },
                'greek_sentiment': {
                    'delta': avg_delta,
                    'gamma': avg_gamma,
                    'theta': avg_theta,
                    'vega': avg_vega
                },
                'technical_indicators': {
                    'rsi': rsi,
                    'macd': macd,
                    'macd_signal': macd_signal,
                    'ma_signal': (macd - macd_signal) if macd_signal != 0 else 0
                },
                'atr': atr,
                'implied_volatility': avg_iv,
                'timestamp': timestamp
            }
            
            return regime_input
            
        except Exception as e:
            logger.error(f"Error preparing regime input: {e}")
            return {}
    
    def _on_live_regime_update(self, regime_data: Dict[str, Any]):
        """Handle live regime updates"""
        try:
            self.current_regime = regime_data
            self.regime_history.append(regime_data)
            
            # Keep only recent history
            if len(self.regime_history) > 1000:
                self.regime_history = self.regime_history[-1000:]
            
            logger.info(f"Live regime update: {regime_data.get('regime_result', {}).get('regime_type', 'Unknown')}")
            
        except Exception as e:
            logger.error(f"Error handling live regime update: {e}")
    
    def _on_live_data_update(self, data: Dict[str, Any]):
        """Handle live data updates"""
        try:
            # Process live data for regime detection
            # This is handled by the streaming manager and regime detector
            pass
            
        except Exception as e:
            logger.error(f"Error handling live data update: {e}")
    
    def _generate_execution_summary(self, regime_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate execution summary"""
        try:
            if not regime_results:
                return {'message': 'No regime results to summarize'}
            
            total_detections = len(regime_results)
            unique_regimes = len(set(r['regime_type'] for r in regime_results))
            avg_confidence = np.mean([r['confidence'] for r in regime_results])
            
            # Regime distribution
            regime_counts = {}
            for result in regime_results:
                regime_type = result['regime_type']
                regime_counts[regime_type] = regime_counts.get(regime_type, 0) + 1
            
            return {
                'total_detections': total_detections,
                'unique_regimes_detected': unique_regimes,
                'average_confidence': avg_confidence,
                'regime_distribution': regime_counts,
                'execution_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating execution summary: {e}")
            return {'error': str(e)}
    
    def _calculate_performance_metrics(self, regime_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance metrics"""
        try:
            if not regime_results:
                return {}
            
            confidences = [r['confidence'] for r in regime_results]
            
            return {
                'confidence_statistics': {
                    'mean': np.mean(confidences),
                    'std': np.std(confidences),
                    'min': np.min(confidences),
                    'max': np.max(confidences),
                    'median': np.median(confidences)
                },
                'detection_rate': len(regime_results) / len(regime_results),  # Always 1 for successful detections
                'regime_stability': self._calculate_regime_stability(regime_results)
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {'error': str(e)}
    
    def _calculate_regime_statistics(self, regime_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate regime-specific statistics"""
        try:
            regime_stats = {}
            
            for result in regime_results:
                regime_type = result['regime_type']
                
                if regime_type not in regime_stats:
                    regime_stats[regime_type] = {
                        'count': 0,
                        'total_confidence': 0,
                        'avg_confidence': 0
                    }
                
                regime_stats[regime_type]['count'] += 1
                regime_stats[regime_type]['total_confidence'] += result['confidence']
                regime_stats[regime_type]['avg_confidence'] = (
                    regime_stats[regime_type]['total_confidence'] / 
                    regime_stats[regime_type]['count']
                )
            
            return regime_stats
            
        except Exception as e:
            logger.error(f"Error calculating regime statistics: {e}")
            return {'error': str(e)}
    
    def _calculate_regime_stability(self, regime_results: List[Dict[str, Any]]) -> float:
        """Calculate regime stability (how often regime changes)"""
        try:
            if len(regime_results) < 2:
                return 1.0
            
            changes = 0
            for i in range(1, len(regime_results)):
                if regime_results[i]['regime_type'] != regime_results[i-1]['regime_type']:
                    changes += 1
            
            stability = 1.0 - (changes / (len(regime_results) - 1))
            return stability
            
        except Exception as e:
            logger.error(f"Error calculating regime stability: {e}")
            return 0.0
    
    def _extract_timeframe_from_query(self, query: str) -> str:
        """Extract timeframe from SQL query"""
        if '1min' in query:
            return '1min'
        elif '5min' in query:
            return '5min'
        elif '15min' in query:
            return '15min'
        else:
            return '1min'
    
    def _extract_symbol_from_query(self, query: str) -> str:
        """Extract symbol from SQL query"""
        # Simple extraction - could be improved
        if 'NIFTY' in query:
            return 'NIFTY'
        else:
            return 'UNKNOWN'
    
    def get_current_regime(self) -> Optional[Dict[str, Any]]:
        """Get current regime (for live mode)"""
        return self.current_regime
    
    def get_regime_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get regime history"""
        return self.regime_history[-limit:] if self.regime_history else []
    
    def stop_live_execution(self) -> bool:
        """Stop live execution"""
        try:
            if self.streaming_manager:
                return self.streaming_manager.stop_streaming()
            return True
            
        except Exception as e:
            logger.error(f"Error stopping live execution: {e}")
            return False
    
    def get_execution_status(self) -> Dict[str, Any]:
        """Get current execution status"""
        try:
            status = {
                'live_mode': self.live_mode,
                'regime_detector_initialized': self.regime_detector is not None,
                'excel_config_loaded': self.excel_manager is not None,
                'current_regime': self.current_regime,
                'regime_history_count': len(self.regime_history),
                'execution_results_count': len(self.execution_results)
            }
            
            if self.streaming_manager:
                status['streaming_status'] = self.streaming_manager.get_streaming_status()
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting execution status: {e}")
            return {'error': str(e)}
