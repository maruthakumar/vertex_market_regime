"""
Market Regime Processor

This module handles the processing and integration of market regime strategies
with the backtester V2 system.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import json

# Add paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKTESTER_V2_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))

if BACKTESTER_V2_DIR not in sys.path:
    sys.path.append(BACKTESTER_V2_DIR)

try:
    from .market_regime_strategy import MarketRegimeStrategy
except ImportError:
    from strategies.market_regime_strategy import MarketRegimeStrategy

class MarketRegimeProcessor:
    """
    Market Regime Processor for Backtester V2 Integration
    
    This class handles the processing of market regime backtests,
    including file parsing, strategy execution, and result generation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Market Regime Processor"""
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Configuration
        self.regime_mode = config.get('regime_mode', '18_REGIME')
        self.dte_adaptation = config.get('dte_adaptation', True)
        self.dynamic_weights = config.get('dynamic_weights', True)
        self.start_date = config.get('start_date')
        self.end_date = config.get('end_date')
        self.index_name = config.get('index_name', 'NIFTY')
        
        # Files
        self.files = config.get('files', {})
        self.regime_config_file = self.files.get('regime_config')
        
        # Strategy instance
        self.strategy = None
        
        # Results
        self.backtest_results = {}
        self.performance_metrics = {}
        self.regime_analysis = {}
        
        self.logger.info(f"Market Regime Processor initialized for {self.index_name}")
    
    def validate_inputs(self) -> Tuple[bool, str]:
        """Validate input files and configuration"""
        try:
            # Check regime config file
            if not self.regime_config_file:
                return False, "Regime configuration file not provided"
            
            if not os.path.exists(self.regime_config_file):
                return False, f"Regime configuration file not found: {self.regime_config_file}"
            
            # Validate file format
            if not self.regime_config_file.endswith('.xlsx'):
                return False, "Regime configuration must be an Excel file (.xlsx)"
            
            # Validate regime mode
            if self.regime_mode not in ['8_REGIME', '18_REGIME']:
                return False, f"Invalid regime mode: {self.regime_mode}. Must be '8_REGIME' or '18_REGIME'"
            
            self.logger.info("✅ Input validation passed")
            return True, "Validation successful"
            
        except Exception as e:
            error_msg = f"Validation error: {e}"
            self.logger.error(error_msg)
            return False, error_msg
    
    def load_market_data(self) -> pd.DataFrame:
        """PRODUCTION MODE: Load real market data from HeavyDB - NO SYNTHETIC DATA"""
        try:
            # PRODUCTION MODE: Must use real HeavyDB data
            self.logger.error("PRODUCTION MODE: Synthetic market data generation is disabled.")
            self.logger.error("System must load actual market data from HeavyDB.")
            
            # Return empty DataFrame to force real data loading
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error loading market data: {e}")
            raise
    
    def run_backtest(self) -> Dict[str, Any]:
        """Run the market regime backtest"""
        try:
            self.logger.info("🚀 Starting Market Regime Backtest")
            
            # Validate inputs
            is_valid, validation_msg = self.validate_inputs()
            if not is_valid:
                raise ValueError(validation_msg)
            
            # Load market data
            market_data = self.load_market_data()
            
            # Initialize strategy
            strategy_config = {
                'regime_mode': self.regime_mode,
                'dte_adaptation': self.dte_adaptation,
                'dynamic_weights': self.dynamic_weights,
                'regime_config': self.regime_config_file
            }
            
            self.strategy = MarketRegimeStrategy(strategy_config)
            
            # Initialize strategy
            if not self.strategy.initialize():
                raise RuntimeError("Failed to initialize market regime strategy")
            
            # Run backtest
            self.logger.info("📊 Processing market data...")
            
            trades = []
            regime_history = []
            
            # Process data in chunks for better performance
            chunk_size = 100
            total_chunks = len(market_data) // chunk_size + 1
            
            for i in range(0, len(market_data), chunk_size):
                chunk_data = market_data.iloc[i:i+chunk_size]
                
                if len(chunk_data) == 0:
                    continue
                
                # Process market data
                regime_result = self.strategy.process_market_data(chunk_data)
                
                if regime_result:
                    # Generate signals
                    signals = regime_result.get('signals', {})
                    
                    # Execute trade if signals are strong enough
                    if signals.get('action') != 'HOLD':
                        trade_result = self.strategy.execute_trade(signals, chunk_data)
                        if trade_result.get('executed'):
                            trades.append(trade_result)
                    
                    # Store regime information
                    current_regime = regime_result.get('current_regime')
                    if current_regime:
                        regime_history.append(current_regime)
                
                # Progress logging
                if (i // chunk_size) % 10 == 0:
                    progress = min(100, (i / len(market_data)) * 100)
                    self.logger.info(f"Progress: {progress:.1f}% ({i}/{len(market_data)} points)")
            
            # Generate results
            self.backtest_results = {
                'strategy_type': 'MARKET_REGIME',
                'regime_mode': self.regime_mode,
                'dte_adaptation': self.dte_adaptation,
                'dynamic_weights': self.dynamic_weights,
                'total_data_points': len(market_data),
                'total_trades': len(trades),
                'regime_changes': len(regime_history),
                'trades': trades,
                'regime_history': regime_history,
                'data_period': {
                    'start': str(market_data.index[0]),
                    'end': str(market_data.index[-1])
                }
            }
            
            # Get strategy performance
            self.performance_metrics = self.strategy.get_strategy_performance()
            
            # Generate regime analysis
            self.regime_analysis = self._analyze_regime_performance(regime_history, trades)
            
            # Cleanup strategy
            self.strategy.cleanup()
            
            self.logger.info("✅ Market Regime Backtest completed successfully")
            
            return {
                'status': 'completed',
                'results': self.backtest_results,
                'performance': self.performance_metrics,
                'regime_analysis': self.regime_analysis
            }
            
        except Exception as e:
            self.logger.error(f"❌ Backtest failed: {e}")
            raise
    
    def _analyze_regime_performance(self, regime_history: List[Dict], trades: List[Dict]) -> Dict[str, Any]:
        """Analyze regime performance"""
        try:
            analysis = {
                'regime_distribution': {},
                'regime_stability': {},
                'trade_performance_by_regime': {},
                'regime_transition_analysis': {}
            }
            
            if not regime_history:
                return analysis
            
            # Regime distribution
            regime_types = [r['regime_type'] for r in regime_history]
            unique_regimes = list(set(regime_types))
            
            for regime in unique_regimes:
                count = regime_types.count(regime)
                analysis['regime_distribution'][regime] = {
                    'count': count,
                    'percentage': count / len(regime_types) * 100
                }
            
            # Regime stability (average duration)
            regime_durations = []
            current_regime = None
            duration = 0
            
            for regime_info in regime_history:
                if current_regime == regime_info['regime_type']:
                    duration += 1
                else:
                    if current_regime is not None:
                        regime_durations.append(duration)
                    current_regime = regime_info['regime_type']
                    duration = 1
            
            if duration > 0:
                regime_durations.append(duration)
            
            analysis['regime_stability'] = {
                'avg_duration': np.mean(regime_durations) if regime_durations else 0,
                'total_transitions': len(regime_durations),
                'stability_score': np.mean(regime_durations) / len(regime_history) if regime_history else 0
            }
            
            # Trade performance by regime
            for trade in trades:
                regime_type = trade.get('regime_type', 'UNKNOWN')
                if regime_type not in analysis['trade_performance_by_regime']:
                    analysis['trade_performance_by_regime'][regime_type] = {
                        'trades': 0,
                        'total_quantity': 0,
                        'avg_confidence': 0
                    }
                
                analysis['trade_performance_by_regime'][regime_type]['trades'] += 1
                analysis['trade_performance_by_regime'][regime_type]['total_quantity'] += trade.get('quantity', 0)
                analysis['trade_performance_by_regime'][regime_type]['avg_confidence'] += trade.get('confidence', 0)
            
            # Calculate averages
            for regime_type in analysis['trade_performance_by_regime']:
                regime_data = analysis['trade_performance_by_regime'][regime_type]
                if regime_data['trades'] > 0:
                    regime_data['avg_confidence'] /= regime_data['trades']
                    regime_data['avg_quantity'] = regime_data['total_quantity'] / regime_data['trades']
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing regime performance: {e}")
            return {}
    
    def generate_output_file(self, output_path: str) -> str:
        """Generate output Excel file with results"""
        try:
            self.logger.info(f"Generating output file: {output_path}")
            
            # Create Excel writer
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                
                # Summary sheet
                summary_data = {
                    'Metric': [
                        'Strategy Type', 'Regime Mode', 'DTE Adaptation', 'Dynamic Weights',
                        'Total Data Points', 'Total Trades', 'Regime Changes',
                        'Data Start', 'Data End', 'Success Rate'
                    ],
                    'Value': [
                        self.backtest_results.get('strategy_type', 'N/A'),
                        self.backtest_results.get('regime_mode', 'N/A'),
                        self.backtest_results.get('dte_adaptation', 'N/A'),
                        self.backtest_results.get('dynamic_weights', 'N/A'),
                        self.backtest_results.get('total_data_points', 0),
                        self.backtest_results.get('total_trades', 0),
                        self.backtest_results.get('regime_changes', 0),
                        self.backtest_results.get('data_period', {}).get('start', 'N/A'),
                        self.backtest_results.get('data_period', {}).get('end', 'N/A'),
                        f"{self.performance_metrics.get('success_rate', 0):.2%}"
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Trades sheet
                if self.backtest_results.get('trades'):
                    trades_df = pd.DataFrame(self.backtest_results['trades'])
                    trades_df.to_excel(writer, sheet_name='Trades', index=False)
                
                # Regime analysis sheet
                if self.regime_analysis:
                    regime_dist = self.regime_analysis.get('regime_distribution', {})
                    if regime_dist:
                        regime_df = pd.DataFrame([
                            {'Regime': regime, 'Count': data['count'], 'Percentage': data['percentage']}
                            for regime, data in regime_dist.items()
                        ])
                        regime_df.to_excel(writer, sheet_name='Regime_Distribution', index=False)
            
            self.logger.info(f"✅ Output file generated: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating output file: {e}")
            raise
