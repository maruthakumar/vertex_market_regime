"""
Market Regime Strategy Implementation

This module implements the Market Regime strategy for the backtester V2 system,
integrating with the comprehensive market regime detection system.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging

# Add market regime module to path
MARKET_REGIME_PATH = '/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/market_regime'
if MARKET_REGIME_PATH not in sys.path:
    sys.path.append(MARKET_REGIME_PATH)

# Add core module to path
CORE_PATH = '/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/core'
if CORE_PATH not in sys.path:
    sys.path.append(CORE_PATH)

try:
    from base_strategy import BaseStrategy
except ImportError:
    # Fallback base strategy
    class BaseStrategy:
        def __init__(self, config: Dict[str, Any]):
            self.config = config
            self.logger = logging.getLogger(self.__class__.__name__)

try:
    from actual_system_excel_manager import ActualSystemExcelManager
    from actual_system_integrator import ActualSystemIntegrator
    from excel_based_regime_engine import ExcelBasedRegimeEngine
    # Import the new ML integration module
    from strategies.market_regime_ml_integration import MarketRegimeMLIntegration
    MARKET_REGIME_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Market regime modules not available: {e}")
    MARKET_REGIME_AVAILABLE = False

class MarketRegimeStrategy(BaseStrategy):
    """
    Market Regime Strategy Implementation
    
    This strategy uses advanced market regime detection to optimize trading decisions
    based on current market conditions, volatility levels, and DTE considerations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Market Regime Strategy"""
        super().__init__(config)
        
        self.strategy_name = "MARKET_REGIME"
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Strategy configuration
        self.regime_mode = config.get('regime_mode', '18_REGIME')
        self.dte_adaptation = config.get('dte_adaptation', True)
        self.dynamic_weights = config.get('dynamic_weights', True)
        self.config_file = config.get('regime_config')
        
        # Initialize regime engine
        self.regime_engine = None
        self.excel_manager = None
        self.ml_integration = None  # New ML integration
        self.current_regime = None
        self.regime_history = []
        self.weight_history = {}
        self.use_ml_integration = config.get('use_ml_integration', True)
        
        # Performance tracking
        self.regime_performance = {}
        self.total_trades = 0
        self.successful_trades = 0
        
        self.logger.info(f"Market Regime Strategy initialized with mode: {self.regime_mode}")
    
    def initialize(self) -> bool:
        """Initialize the market regime strategy"""
        try:
            if not MARKET_REGIME_AVAILABLE:
                self.logger.error("Market regime modules not available")
                return False
            
            if not self.config_file or not os.path.exists(self.config_file):
                self.logger.error(f"Configuration file not found: {self.config_file}")
                return False
            
            # Initialize Excel manager
            self.excel_manager = ActualSystemExcelManager()
            self.excel_manager.load_configuration(self.config_file)
            
            # Initialize regime engine
            self.regime_engine = ExcelBasedRegimeEngine(self.config_file)
            
            # Initialize ML integration if enabled
            if self.use_ml_integration:
                self.logger.info("Initializing ML integration for enhanced regime detection")
                self.ml_integration = MarketRegimeMLIntegration(self.config_file)
                if self.ml_integration.is_initialized:
                    self.logger.info("✅ ML integration initialized successfully")
                else:
                    self.logger.warning("⚠️ ML integration failed to initialize, falling back to standard engine")
                    self.use_ml_integration = False
            
            # Validate configuration
            regime_config = self.excel_manager.get_regime_formation_configuration()
            complexity_config = self.excel_manager.get_regime_complexity_configuration()
            
            self.logger.info(f"Loaded regime configuration: {len(regime_config)} regime types")
            self.logger.info(f"Regime mode: {self.regime_mode}")
            self.logger.info(f"DTE adaptation: {self.dte_adaptation}")
            self.logger.info(f"Dynamic weights: {self.dynamic_weights}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize market regime strategy: {e}")
            return False
    
    def process_market_data(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Process market data and detect regime"""
        try:
            if self.regime_engine is None:
                self.logger.error("Regime engine not initialized")
                return {}
            
            # Calculate market regime
            if self.use_ml_integration and self.ml_integration and self.ml_integration.is_initialized:
                # Use ML-enhanced detection
                self.logger.debug("Using ML-enhanced regime detection")
                
                # Calculate features
                features = self.ml_integration.calculate_features(market_data)
                
                # Get ensemble regime detection
                import asyncio
                ml_result = asyncio.run(self.ml_integration.ensemble_regime_detection(market_data))
                
                # Convert ML result to regime_results format
                regime_results = self._convert_ml_to_regime_results(ml_result, market_data)
                
                # Log ML statistics
                ml_stats = self.ml_integration.get_ml_statistics()
                if ml_stats:
                    self.logger.debug(f"ML Statistics: {ml_stats}")
            else:
                # Use standard regime engine
                self.logger.debug("Using standard regime detection")
                regime_results = self.regime_engine.calculate_market_regime(market_data)
            
            if regime_results.empty:
                self.logger.warning("No regime results generated")
                return {}
            
            # Get current regime
            latest_regime = regime_results.iloc[-1]
            self.current_regime = {
                'regime_type': latest_regime.get('Market_Regime_Label', 'UNKNOWN'),
                'regime_score': latest_regime.get('Market_Regime_Score', 0.0),
                'regime_confidence': latest_regime.get('Market_Regime_Confidence', 0.0),
                'timestamp': regime_results.index[-1]
            }
            
            # Update regime history
            self.regime_history.append(self.current_regime)
            
            # Keep only last 100 regime records
            if len(self.regime_history) > 100:
                self.regime_history = self.regime_history[-100:]
            
            # Calculate regime-based signals
            signals = self._generate_regime_signals(market_data, regime_results)
            
            return {
                'current_regime': self.current_regime,
                'regime_results': regime_results,
                'signals': signals,
                'regime_history': self.regime_history[-10:]  # Last 10 regimes
            }
            
        except Exception as e:
            self.logger.error(f"Error processing market data: {e}")
            return {}
    
    def _generate_regime_signals(self, market_data: pd.DataFrame, regime_results: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signals based on regime analysis"""
        try:
            signals = {
                'action': 'HOLD',
                'confidence': 0.0,
                'regime_based_adjustments': {},
                'risk_adjustments': {},
                'position_sizing': 1.0
            }
            
            if self.current_regime is None:
                return signals
            
            regime_type = self.current_regime['regime_type']
            regime_confidence = self.current_regime['regime_confidence']
            
            # Regime-based signal generation
            if 'STRONG_BULLISH' in regime_type:
                signals['action'] = 'BUY'
                signals['confidence'] = min(0.9, regime_confidence + 0.2)
                signals['position_sizing'] = 1.2  # Increase position size
                
            elif 'MILD_BULLISH' in regime_type:
                signals['action'] = 'BUY'
                signals['confidence'] = min(0.7, regime_confidence + 0.1)
                signals['position_sizing'] = 1.0
                
            elif 'STRONG_BEARISH' in regime_type:
                signals['action'] = 'SELL'
                signals['confidence'] = min(0.9, regime_confidence + 0.2)
                signals['position_sizing'] = 1.2
                
            elif 'MILD_BEARISH' in regime_type:
                signals['action'] = 'SELL'
                signals['confidence'] = min(0.7, regime_confidence + 0.1)
                signals['position_sizing'] = 1.0
                
            elif 'HIGH_VOLATILE' in regime_type:
                signals['action'] = 'HEDGE'
                signals['confidence'] = regime_confidence
                signals['position_sizing'] = 0.7  # Reduce position size
                signals['risk_adjustments']['volatility_hedge'] = True
                
            elif 'SIDEWAYS' in regime_type or 'NEUTRAL' in regime_type:
                signals['action'] = 'HOLD'
                signals['confidence'] = regime_confidence * 0.5
                signals['position_sizing'] = 0.8
                
            # DTE-based adjustments
            if self.dte_adaptation and 'dte' in market_data.columns:
                current_dte = market_data['dte'].iloc[-1] if len(market_data) > 0 else 5
                
                if current_dte <= 1:  # Expiry day or T-1
                    signals['position_sizing'] *= 0.6  # Reduce exposure
                    signals['risk_adjustments']['expiry_risk'] = True
                elif current_dte <= 3:  # High volatility period
                    signals['position_sizing'] *= 0.8
                    signals['risk_adjustments']['high_volatility_period'] = True
            
            # Confidence-based adjustments
            if regime_confidence < 0.6:
                signals['position_sizing'] *= 0.7  # Reduce position for low confidence
                signals['risk_adjustments']['low_confidence'] = True
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating regime signals: {e}")
            return {'action': 'HOLD', 'confidence': 0.0}
    
    def execute_trade(self, signals: Dict[str, Any], market_data: pd.DataFrame) -> Dict[str, Any]:
        """Execute trade based on regime signals"""
        try:
            trade_result = {
                'executed': False,
                'action': signals.get('action', 'HOLD'),
                'quantity': 0,
                'price': 0.0,
                'regime_type': self.current_regime['regime_type'] if self.current_regime else 'UNKNOWN',
                'confidence': signals.get('confidence', 0.0),
                'timestamp': datetime.now()
            }
            
            action = signals.get('action', 'HOLD')
            confidence = signals.get('confidence', 0.0)
            position_sizing = signals.get('position_sizing', 1.0)
            
            # Execute only if confidence is above threshold
            confidence_threshold = 0.6
            if confidence < confidence_threshold:
                trade_result['reason'] = f"Confidence {confidence:.2f} below threshold {confidence_threshold}"
                return trade_result
            
            # Calculate position size
            base_quantity = 100  # Base quantity
            adjusted_quantity = int(base_quantity * position_sizing)
            
            if action in ['BUY', 'SELL']:
                # Get current price
                current_price = market_data['underlying_price'].iloc[-1] if len(market_data) > 0 else 0
                
                trade_result.update({
                    'executed': True,
                    'quantity': adjusted_quantity,
                    'price': current_price,
                    'position_sizing': position_sizing,
                    'regime_adjustments': signals.get('regime_based_adjustments', {}),
                    'risk_adjustments': signals.get('risk_adjustments', {})
                })
                
                # Update performance tracking
                self.total_trades += 1
                
                # Track regime performance
                regime_type = self.current_regime['regime_type']
                if regime_type not in self.regime_performance:
                    self.regime_performance[regime_type] = {'trades': 0, 'success': 0}
                
                self.regime_performance[regime_type]['trades'] += 1
                
                self.logger.info(f"Executed {action} trade: {adjusted_quantity} @ {current_price:.2f} (Regime: {regime_type})")
            
            return trade_result
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return {'executed': False, 'error': str(e)}
    
    def get_strategy_performance(self) -> Dict[str, Any]:
        """Get strategy performance metrics"""
        try:
            performance = {
                'total_trades': self.total_trades,
                'successful_trades': self.successful_trades,
                'success_rate': self.successful_trades / self.total_trades if self.total_trades > 0 else 0,
                'regime_performance': self.regime_performance,
                'current_regime': self.current_regime,
                'regime_history_count': len(self.regime_history),
                'strategy_config': {
                    'regime_mode': self.regime_mode,
                    'dte_adaptation': self.dte_adaptation,
                    'dynamic_weights': self.dynamic_weights
                }
            }
            
            # Calculate regime distribution
            if self.regime_history:
                regime_types = [r['regime_type'] for r in self.regime_history]
                regime_distribution = {}
                for regime in set(regime_types):
                    regime_distribution[regime] = regime_types.count(regime) / len(regime_types)
                
                performance['regime_distribution'] = regime_distribution
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Error getting strategy performance: {e}")
            return {}
    
    def _convert_ml_to_regime_results(self, ml_result: Dict[str, Any], market_data: pd.DataFrame) -> pd.DataFrame:
        """Convert ML result to regime_results DataFrame format"""
        try:
            # Create DataFrame with proper columns
            regime_results = pd.DataFrame(index=market_data.index[-1:])
            
            # Map ML result to expected columns
            regime_results['Market_Regime_Label'] = ml_result.get('regime_name', 'UNKNOWN')
            regime_results['Market_Regime_Score'] = ml_result.get('confidence_score', 0.0)
            regime_results['Market_Regime_Confidence'] = ml_result.get('ml_confidence', 0.0)
            
            # Add component scores
            feature_scores = ml_result.get('feature_scores', {})
            regime_results['Greek_Sentiment_Score'] = feature_scores.get('greek_sentiment', 0.0)
            regime_results['Trending_OI_PA_Score'] = feature_scores.get('trending_oi_pa', 0.0)
            regime_results['IV_Skew_Score'] = feature_scores.get('iv_skew', 0.0)
            regime_results['Straddle_Score'] = feature_scores.get('straddle', 0.0)
            
            # Add ML metadata
            regime_results['ML_Enhanced'] = True
            regime_results['Ensemble_Agreement'] = ml_result.get('ensemble_agreement', 0.0)
            
            return regime_results
            
        except Exception as e:
            self.logger.error(f"Error converting ML result: {e}")
            return pd.DataFrame()
    
    def cleanup(self):
        """Cleanup strategy resources"""
        try:
            self.logger.info("Cleaning up Market Regime Strategy")
            
            # Save final performance report
            performance = self.get_strategy_performance()
            
            # Log final statistics
            self.logger.info(f"Final Performance Summary:")
            self.logger.info(f"  Total Trades: {performance.get('total_trades', 0)}")
            self.logger.info(f"  Success Rate: {performance.get('success_rate', 0):.2%}")
            self.logger.info(f"  Regime Distribution: {performance.get('regime_distribution', {})}")
            self.logger.info(f"  ML Integration Used: {self.use_ml_integration}")
            
            # Get ML statistics if available
            if self.ml_integration:
                ml_stats = self.ml_integration.get_ml_statistics()
                if ml_stats:
                    self.logger.info(f"  ML Engine Stats: {ml_stats}")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def __str__(self):
        return f"MarketRegimeStrategy(mode={self.regime_mode}, dte_adaptation={self.dte_adaptation})"
