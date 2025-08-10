"""
Excel-Based Market Regime Engine

This is the main engine that provides Excel-based configuration for the actual existing system
at /srv/samba/shared/enhanced-market-regime-optimizer-final-package-updated/
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
import json

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from actual_system_excel_manager import ActualSystemExcelManager
from actual_system_integrator import ActualSystemIntegrator

logger = logging.getLogger(__name__)

class ExcelBasedRegimeEngine:
    """
    Main Excel-Based Market Regime Engine
    
    This engine provides:
    1. Excel-based configuration management
    2. Integration with actual existing system
    3. ATM straddle analysis with EMA/VWAP across 3-5-10-15 min timeframes
    4. Dynamic weightage with historical performance tracking
    5. Unified interface for backtester_v2 integration
    """
    
    def __init__(self, excel_config_path: str = None, auto_generate_template: bool = True):
        """
        Initialize Excel-Based Market Regime Engine
        
        Args:
            excel_config_path (str, optional): Path to Excel configuration file
            auto_generate_template (bool): Auto-generate template if config not found
        """
        self.excel_config_path = excel_config_path
        self.auto_generate_template = auto_generate_template
        
        # Initialize components
        self.excel_manager = None
        self.integrator = None
        self.is_initialized = False
        
        # Configuration cache
        self.config_cache = {}
        self.last_config_load = None
        
        # Performance tracking
        self.performance_history = {}
        self.regime_history = []
        
        # Initialize the engine
        self._initialize_engine()
        
        logger.info("ExcelBasedRegimeEngine initialized")
    
    def _initialize_engine(self):
        """Initialize the engine components"""
        try:
            # Initialize Excel manager
            self.excel_manager = ActualSystemExcelManager()
            
            # Handle configuration
            if self.excel_config_path and Path(self.excel_config_path).exists():
                # Load existing configuration
                logger.info(f"Loading existing configuration: {self.excel_config_path}")
                self.excel_manager.load_configuration(self.excel_config_path)
            elif self.auto_generate_template:
                # Generate template
                template_path = self.excel_config_path or "market_regime_config.xlsx"
                logger.info(f"Generating Excel template: {template_path}")
                self.excel_manager.generate_excel_template(template_path)
                self.excel_config_path = template_path
                self.excel_manager.load_configuration(template_path)
            else:
                logger.warning("No configuration provided and auto-generation disabled")
                return
            
            # Initialize integrator
            self.integrator = ActualSystemIntegrator(self.excel_config_path)
            
            # Cache configuration
            self._cache_configuration()
            
            self.is_initialized = True
            logger.info("✅ Engine initialization completed")
            
        except Exception as e:
            logger.error(f"❌ Engine initialization failed: {e}")
            self.is_initialized = False
    
    def _cache_configuration(self):
        """Cache configuration for faster access"""
        try:
            if not self.excel_manager:
                return
            
            self.config_cache = {
                'indicators': self.excel_manager.get_indicator_configuration(),
                'straddles': self.excel_manager.get_straddle_configuration(),
                'dynamic_weights': self.excel_manager.get_dynamic_weightage_configuration(),
                'timeframes': self.excel_manager.get_timeframe_configuration(),
                'greek_sentiment': self.excel_manager.get_greek_sentiment_configuration(),
                'regime_formation': self.excel_manager.get_regime_formation_configuration()
            }
            
            self.last_config_load = datetime.now()
            logger.info("Configuration cached successfully")
            
        except Exception as e:
            logger.error(f"Error caching configuration: {e}")
    
    def calculate_market_regime(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate market regime using Excel configuration and actual system
        
        Args:
            data (pd.DataFrame): Market data with required columns
            **kwargs: Additional parameters
            
        Returns:
            pd.DataFrame: Market regime results with comprehensive analysis
        """
        try:
            if not self.is_initialized:
                logger.error("Engine not initialized")
                return pd.DataFrame()
            
            logger.info("Calculating market regime with Excel-based configuration")
            
            # Validate input data
            required_columns = ['underlying_price', 'volume', 'atm_straddle_price']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                logger.warning(f"Missing required columns: {missing_columns}")
                # Try to create missing columns or use aliases
                data = self._handle_missing_columns(data, missing_columns)
            
            # Add timeframe-specific analysis
            enhanced_data = self._add_timeframe_analysis(data)
            
            # Calculate regime using integrator
            market_regime = self.integrator.calculate_market_regime(enhanced_data, **kwargs)
            
            if market_regime.empty:
                logger.warning("No market regime results generated")
                return pd.DataFrame()
            
            # Enhance results with Excel-specific features
            enhanced_regime = self._enhance_regime_results(market_regime, enhanced_data)
            
            # Update performance tracking
            self._update_performance_tracking(enhanced_regime)
            
            # Store in history
            self.regime_history.append({
                'timestamp': datetime.now(),
                'regime_count': len(enhanced_regime),
                'primary_regime': enhanced_regime['Market_Regime_Label'].mode().iloc[0] if not enhanced_regime.empty else 'Unknown'
            })
            
            logger.info(f"✅ Market regime calculated: {len(enhanced_regime)} points")
            return enhanced_regime
            
        except Exception as e:
            logger.error(f"Error calculating market regime: {e}")
            return pd.DataFrame()
    
    def _handle_missing_columns(self, data: pd.DataFrame, missing_columns: List[str]) -> pd.DataFrame:
        """Handle missing columns by creating aliases or synthetic data"""
        enhanced_data = data.copy()
        
        for col in missing_columns:
            if col == 'underlying_price':
                if 'price' in data.columns:
                    enhanced_data['underlying_price'] = data['price']
                elif 'close' in data.columns:
                    enhanced_data['underlying_price'] = data['close']
                else:
                    logger.warning("Cannot create underlying_price column")
            
            elif col == 'atm_straddle_price':
                if 'ATM_STRADDLE' in data.columns:
                    enhanced_data['atm_straddle_price'] = data['ATM_STRADDLE']
                elif 'atm_ce_price' in data.columns and 'atm_pe_price' in data.columns:
                    enhanced_data['atm_straddle_price'] = data['atm_ce_price'] + data['atm_pe_price']
                else:
                    logger.warning("Cannot create atm_straddle_price column")
            
            elif col == 'volume':
                if 'Volume' in data.columns:
                    enhanced_data['volume'] = data['Volume']
                else:
                    # Create synthetic volume
                    enhanced_data['volume'] = 10000
                    logger.warning("Created synthetic volume data")
        
        return enhanced_data
    
    def _add_timeframe_analysis(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add multi-timeframe analysis (3-5-10-15 min)"""
        try:
            enhanced_data = data.copy()
            
            # Get timeframe configuration
            timeframe_config = self.config_cache.get('timeframes')
            if timeframe_config is None:
                logger.warning("No timeframe configuration available")
                return enhanced_data
            
            enabled_timeframes = timeframe_config[timeframe_config['Enabled'] == True]
            
            for _, tf_row in enabled_timeframes.iterrows():
                timeframe = tf_row['Timeframe']
                
                try:
                    # Convert timeframe to pandas frequency
                    if timeframe.endswith('min'):
                        freq = timeframe.replace('min', 'T')
                    else:
                        freq = timeframe
                    
                    # Resample data for this timeframe
                    if 'datetime' in enhanced_data.columns:
                        tf_data = enhanced_data.set_index('datetime')
                    else:
                        tf_data = enhanced_data
                    
                    # Calculate OHLCV for this timeframe
                    ohlcv = tf_data.resample(freq).agg({
                        'underlying_price': ['first', 'max', 'min', 'last'],
                        'volume': 'sum',
                        'atm_straddle_price': ['first', 'max', 'min', 'last']
                    }).dropna()
                    
                    # Flatten column names
                    ohlcv.columns = ['_'.join(col).strip() for col in ohlcv.columns]
                    
                    # Calculate EMA for this timeframe
                    if f'underlying_price_last' in ohlcv.columns:
                        ohlcv[f'ema20_{timeframe}'] = ohlcv['underlying_price_last'].ewm(span=20).mean()
                        ohlcv[f'ema50_{timeframe}'] = ohlcv['underlying_price_last'].ewm(span=50).mean()
                    
                    if f'atm_straddle_price_last' in ohlcv.columns:
                        ohlcv[f'straddle_ema20_{timeframe}'] = ohlcv['atm_straddle_price_last'].ewm(span=20).mean()
                        ohlcv[f'straddle_ema50_{timeframe}'] = ohlcv['atm_straddle_price_last'].ewm(span=50).mean()
                    
                    # Calculate VWAP for this timeframe
                    if f'underlying_price_last' in ohlcv.columns and 'volume_sum' in ohlcv.columns:
                        typical_price = (ohlcv['underlying_price_max'] + ohlcv['underlying_price_min'] + ohlcv['underlying_price_last']) / 3
                        vwap_num = (typical_price * ohlcv['volume_sum']).cumsum()
                        vwap_den = ohlcv['volume_sum'].cumsum()
                        ohlcv[f'vwap_{timeframe}'] = vwap_num / vwap_den
                    
                    # Add previous day VWAP (simplified)
                    if f'vwap_{timeframe}' in ohlcv.columns:
                        ohlcv[f'prev_day_vwap_{timeframe}'] = ohlcv[f'vwap_{timeframe}'].shift(1)
                    
                    # Merge back to original data (forward fill)
                    for col in ohlcv.columns:
                        if col not in enhanced_data.columns:
                            # Reindex to match original data
                            if 'datetime' in enhanced_data.columns:
                                merged_series = ohlcv[col].reindex(enhanced_data['datetime'], method='ffill')
                                enhanced_data[col] = merged_series.values
                            else:
                                enhanced_data[col] = ohlcv[col].reindex(enhanced_data.index, method='ffill')
                    
                    logger.info(f"Added {timeframe} analysis")
                    
                except Exception as e:
                    logger.warning(f"Error adding {timeframe} analysis: {e}")
            
            return enhanced_data
            
        except Exception as e:
            logger.error(f"Error in timeframe analysis: {e}")
            return data
    
    def _enhance_regime_results(self, regime_results: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
        """Enhance regime results with Excel-specific features"""
        try:
            enhanced_results = regime_results.copy()
            
            # Add straddle analysis summary
            straddle_config = self.config_cache.get('straddles')
            if straddle_config is not None:
                enabled_straddles = straddle_config[straddle_config['Enabled'] == True]
                
                # Calculate weighted straddle score
                straddle_score = 0.0
                total_weight = 0.0
                
                for _, straddle_row in enabled_straddles.iterrows():
                    straddle_type = straddle_row['StraddleType']
                    weight = straddle_row['Weight']
                    
                    # Look for corresponding signal in results
                    signal_col = f'{straddle_type.lower()}_signal'
                    if signal_col in enhanced_results.columns:
                        straddle_score += enhanced_results[signal_col] * weight
                        total_weight += weight
                
                if total_weight > 0:
                    enhanced_results['Straddle_Composite_Score'] = straddle_score / total_weight
                    enhanced_results['Straddle_Confidence'] = total_weight / len(enabled_straddles)
            
            # Add timeframe consensus
            timeframe_config = self.config_cache.get('timeframes')
            if timeframe_config is not None:
                enabled_timeframes = timeframe_config[timeframe_config['Enabled'] == True]
                
                # Calculate timeframe consensus
                consensus_score = 0.0
                consensus_weight = 0.0
                
                for _, tf_row in enabled_timeframes.iterrows():
                    timeframe = tf_row['Timeframe']
                    weight = tf_row['Weight']
                    
                    # Look for timeframe-specific signals
                    tf_cols = [col for col in enhanced_results.columns if timeframe in col and 'signal' in col]
                    if tf_cols:
                        tf_signal = enhanced_results[tf_cols[0]]
                        consensus_score += tf_signal * weight
                        consensus_weight += weight
                
                if consensus_weight > 0:
                    enhanced_results['Timeframe_Consensus'] = consensus_score / consensus_weight
                    enhanced_results['Timeframe_Agreement'] = consensus_weight / len(enabled_timeframes)
            
            # Add Excel configuration metadata
            enhanced_results['Excel_Config_Applied'] = True
            enhanced_results['Config_Load_Time'] = self.last_config_load
            enhanced_results['Engine_Version'] = '1.0.0'
            
            # Add performance metrics
            if self.performance_history:
                avg_performance = np.mean(list(self.performance_history.values()))
                enhanced_results['Historical_Performance'] = avg_performance
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Error enhancing regime results: {e}")
            return regime_results
    
    def _update_performance_tracking(self, regime_results: pd.DataFrame):
        """Update performance tracking based on regime results"""
        try:
            if regime_results.empty:
                return
            
            # Simple performance metric based on regime consistency
            if 'Market_Regime_Label' in regime_results.columns:
                regime_changes = (regime_results['Market_Regime_Label'] != regime_results['Market_Regime_Label'].shift(1)).sum()
                stability_score = 1.0 - (regime_changes / len(regime_results))
                
                # Update performance history
                timestamp = datetime.now().strftime('%Y-%m-%d')
                self.performance_history[timestamp] = stability_score
                
                # Keep only last 30 days
                if len(self.performance_history) > 30:
                    oldest_key = min(self.performance_history.keys())
                    del self.performance_history[oldest_key]
            
        except Exception as e:
            logger.error(f"Error updating performance tracking: {e}")
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status"""
        status = {
            'is_initialized': self.is_initialized,
            'excel_config_path': self.excel_config_path,
            'last_config_load': self.last_config_load,
            'config_cache_size': len(self.config_cache),
            'performance_history_size': len(self.performance_history),
            'regime_history_size': len(self.regime_history)
        }
        
        # Add integrator status if available
        if self.integrator:
            integrator_status = self.integrator.get_system_status()
            status.update({f'integrator_{k}': v for k, v in integrator_status.items()})
        
        # Add configuration summary
        if self.config_cache:
            for config_type, config_df in self.config_cache.items():
                if isinstance(config_df, pd.DataFrame):
                    status[f'{config_type}_count'] = len(config_df)
                    if 'Enabled' in config_df.columns:
                        status[f'{config_type}_enabled'] = config_df['Enabled'].sum()
        
        return status
    
    def reload_configuration(self) -> bool:
        """Reload configuration from Excel file"""
        try:
            if not self.excel_config_path or not Path(self.excel_config_path).exists():
                logger.error("No valid configuration file to reload")
                return False
            
            logger.info("Reloading configuration...")
            
            # Reload Excel manager
            self.excel_manager.load_configuration(self.excel_config_path)
            
            # Reinitialize integrator
            self.integrator = ActualSystemIntegrator(self.excel_config_path)
            
            # Update cache
            self._cache_configuration()
            
            logger.info("✅ Configuration reloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error reloading configuration: {e}")
            return False
    
    def update_weights_from_performance(self, performance_data: Dict[str, float]) -> bool:
        """Update dynamic weights based on performance data"""
        try:
            if not self.integrator:
                logger.error("Integrator not available")
                return False
            
            # Update weights
            success = self.integrator.update_weights_from_performance(performance_data)
            
            if success:
                # Update cache
                self._cache_configuration()
                
                # Save updated configuration
                if self.excel_config_path:
                    self.integrator.save_updated_configuration()
                
                logger.info("✅ Weights updated from performance data")
            
            return success
            
        except Exception as e:
            logger.error(f"Error updating weights: {e}")
            return False
    
    def generate_configuration_template(self, output_path: str = None) -> str:
        """Generate a new Excel configuration template"""
        try:
            output_path = output_path or "new_market_regime_config.xlsx"
            
            if not self.excel_manager:
                self.excel_manager = ActualSystemExcelManager()
            
            template_path = self.excel_manager.generate_excel_template(output_path)
            logger.info(f"✅ Configuration template generated: {template_path}")
            
            return template_path
            
        except Exception as e:
            logger.error(f"Error generating template: {e}")
            return None
    
    def export_performance_report(self, output_path: str = None) -> str:
        """Export performance report to JSON"""
        try:
            output_path = output_path or f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            report = {
                'engine_status': self.get_engine_status(),
                'performance_history': self.performance_history,
                'regime_history': self.regime_history,
                'configuration_summary': {}
            }
            
            # Add configuration summary
            if self.config_cache:
                for config_type, config_df in self.config_cache.items():
                    if isinstance(config_df, pd.DataFrame):
                        report['configuration_summary'][config_type] = {
                            'total_count': len(config_df),
                            'enabled_count': config_df['Enabled'].sum() if 'Enabled' in config_df.columns else 'N/A',
                            'columns': list(config_df.columns)
                        }
            
            # Save report
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"✅ Performance report exported: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting performance report: {e}")
            return None
