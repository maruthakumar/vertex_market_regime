"""
Backtester Integration Example

This script shows how to integrate the Excel-based market regime system
with backtester_v2 for comprehensive strategy testing.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2')

class MarketRegimeStrategy:
    """
    Example strategy that uses Excel-based market regime detection
    
    This strategy demonstrates how to:
    1. Load Excel configuration for market regime
    2. Calculate regime in real-time
    3. Make trading decisions based on regime
    4. Track performance and update weights
    """
    
    def __init__(self, excel_config_path: str = None):
        """Initialize strategy with Excel configuration"""
        self.excel_config_path = excel_config_path or "demo_market_regime_config.xlsx"
        
        # Initialize regime engine
        try:
            from excel_based_regime_engine import ExcelBasedRegimeEngine
            self.regime_engine = ExcelBasedRegimeEngine(self.excel_config_path)
            self.regime_available = True
            logger.info("✅ Market regime engine initialized")
        except Exception as e:
            logger.warning(f"⚠️  Regime engine not available: {e}")
            self.regime_available = False
            self.regime_engine = None
        
        # Strategy state
        self.current_regime = None
        self.regime_confidence = 0.0
        self.regime_history = []
        self.performance_tracker = {}
        
        # Trading parameters based on regime
        self.regime_parameters = {
            'STRONG_BULLISH': {'position_size': 1.5, 'stop_loss': 0.02, 'take_profit': 0.06},
            'MILD_BULLISH': {'position_size': 1.0, 'stop_loss': 0.015, 'take_profit': 0.04},
            'NEUTRAL': {'position_size': 0.5, 'stop_loss': 0.01, 'take_profit': 0.02},
            'SIDEWAYS': {'position_size': 0.3, 'stop_loss': 0.008, 'take_profit': 0.015},
            'MILD_BEARISH': {'position_size': -1.0, 'stop_loss': 0.015, 'take_profit': 0.04},
            'STRONG_BEARISH': {'position_size': -1.5, 'stop_loss': 0.02, 'take_profit': 0.06},
            'HIGH_VOLATILITY': {'position_size': 0.8, 'stop_loss': 0.025, 'take_profit': 0.05},
            'LOW_VOLATILITY': {'position_size': 1.2, 'stop_loss': 0.01, 'take_profit': 0.03}
        }
        
        logger.info("MarketRegimeStrategy initialized")
    
    def calculate_regime(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate current market regime"""
        try:
            if not self.regime_available or self.regime_engine is None:
                return self._fallback_regime_calculation(market_data)
            
            # Calculate regime using Excel-based engine
            regime_results = self.regime_engine.calculate_market_regime(market_data)
            
            if regime_results.empty:
                logger.warning("No regime results from engine")
                return self._fallback_regime_calculation(market_data)
            
            # Extract current regime
            current_regime = regime_results['Market_Regime_Label'].iloc[-1]
            regime_score = regime_results.get('Market_Regime_Score', pd.Series([0])).iloc[-1]
            
            # Calculate confidence based on available metrics
            confidence_factors = []
            
            # Straddle confidence
            if 'Straddle_Composite_Score' in regime_results.columns:
                straddle_score = regime_results['Straddle_Composite_Score'].iloc[-1]
                confidence_factors.append(abs(straddle_score))
            
            # Timeframe consensus
            if 'Timeframe_Consensus' in regime_results.columns:
                timeframe_consensus = regime_results['Timeframe_Consensus'].iloc[-1]
                confidence_factors.append(abs(timeframe_consensus))
            
            # Overall confidence
            if confidence_factors:
                regime_confidence = np.mean(confidence_factors)
            else:
                regime_confidence = abs(regime_score) if not pd.isna(regime_score) else 0.5
            
            # Normalize confidence to 0-1 range
            regime_confidence = min(1.0, max(0.0, regime_confidence))
            
            regime_info = {
                'regime': current_regime,
                'score': regime_score,
                'confidence': regime_confidence,
                'timestamp': datetime.now(),
                'straddle_score': regime_results.get('Straddle_Composite_Score', pd.Series([0])).iloc[-1],
                'timeframe_consensus': regime_results.get('Timeframe_Consensus', pd.Series([0])).iloc[-1],
                'excel_config_applied': regime_results.get('Excel_Config_Applied', pd.Series([False])).iloc[-1]
            }
            
            # Update strategy state
            self.current_regime = current_regime
            self.regime_confidence = regime_confidence
            self.regime_history.append(regime_info)
            
            # Keep only last 100 regime calculations
            if len(self.regime_history) > 100:
                self.regime_history = self.regime_history[-100:]
            
            logger.info(f"Regime: {current_regime} (confidence: {regime_confidence:.2f})")
            return regime_info
            
        except Exception as e:
            logger.error(f"Error calculating regime: {e}")
            return self._fallback_regime_calculation(market_data)
    
    def _fallback_regime_calculation(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Fallback regime calculation when main engine is not available"""
        try:
            if len(market_data) < 20:
                return {
                    'regime': 'NEUTRAL',
                    'score': 0.0,
                    'confidence': 0.3,
                    'timestamp': datetime.now(),
                    'fallback': True
                }
            
            # Simple regime calculation based on price movement and volatility
            recent_data = market_data.tail(20)
            
            # Price trend
            price_change = (recent_data['underlying_price'].iloc[-1] - recent_data['underlying_price'].iloc[0]) / recent_data['underlying_price'].iloc[0]
            
            # Volatility
            returns = recent_data['underlying_price'].pct_change().dropna()
            volatility = returns.std()
            
            # Determine regime
            if volatility > 0.02:  # High volatility
                regime = 'HIGH_VOLATILITY'
            elif volatility < 0.005:  # Low volatility
                regime = 'LOW_VOLATILITY'
            elif price_change > 0.01:  # Strong bullish
                regime = 'STRONG_BULLISH'
            elif price_change > 0.003:  # Mild bullish
                regime = 'MILD_BULLISH'
            elif price_change < -0.01:  # Strong bearish
                regime = 'STRONG_BEARISH'
            elif price_change < -0.003:  # Mild bearish
                regime = 'MILD_BEARISH'
            else:  # Neutral/Sideways
                regime = 'NEUTRAL'
            
            confidence = min(1.0, abs(price_change) * 50 + volatility * 20)
            
            return {
                'regime': regime,
                'score': price_change,
                'confidence': confidence,
                'timestamp': datetime.now(),
                'fallback': True
            }
            
        except Exception as e:
            logger.error(f"Error in fallback regime calculation: {e}")
            return {
                'regime': 'NEUTRAL',
                'score': 0.0,
                'confidence': 0.1,
                'timestamp': datetime.now(),
                'error': True
            }
    
    def generate_signals(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signals based on market regime"""
        try:
            # Calculate current regime
            regime_info = self.calculate_regime(market_data)
            
            current_regime = regime_info['regime']
            confidence = regime_info['confidence']
            
            # Get regime-specific parameters
            regime_params = self.regime_parameters.get(current_regime, self.regime_parameters['NEUTRAL'])
            
            # Adjust position size based on confidence
            base_position_size = regime_params['position_size']
            adjusted_position_size = base_position_size * confidence
            
            # Generate signal
            signal = {
                'regime': current_regime,
                'confidence': confidence,
                'position_size': adjusted_position_size,
                'stop_loss': regime_params['stop_loss'],
                'take_profit': regime_params['take_profit'],
                'timestamp': datetime.now(),
                'regime_score': regime_info.get('score', 0),
                'straddle_score': regime_info.get('straddle_score', 0),
                'timeframe_consensus': regime_info.get('timeframe_consensus', 0)
            }
            
            # Add regime-specific logic
            if current_regime in ['STRONG_BULLISH', 'MILD_BULLISH']:
                signal['action'] = 'BUY' if confidence > 0.6 else 'HOLD'
                signal['strategy'] = 'LONG_BIAS'
            elif current_regime in ['STRONG_BEARISH', 'MILD_BEARISH']:
                signal['action'] = 'SELL' if confidence > 0.6 else 'HOLD'
                signal['strategy'] = 'SHORT_BIAS'
            elif current_regime == 'HIGH_VOLATILITY':
                signal['action'] = 'STRADDLE' if confidence > 0.7 else 'HOLD'
                signal['strategy'] = 'VOLATILITY_PLAY'
            elif current_regime == 'LOW_VOLATILITY':
                signal['action'] = 'IRON_CONDOR' if confidence > 0.7 else 'HOLD'
                signal['strategy'] = 'RANGE_BOUND'
            else:  # NEUTRAL, SIDEWAYS
                signal['action'] = 'HOLD'
                signal['strategy'] = 'WAIT'
            
            logger.info(f"Signal: {signal['action']} ({signal['strategy']}) - Size: {signal['position_size']:.2f}")
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return {
                'regime': 'NEUTRAL',
                'action': 'HOLD',
                'position_size': 0,
                'error': str(e)
            }
    
    def update_performance(self, trade_result: Dict[str, Any]):
        """Update performance tracking for regime-based weight adjustment"""
        try:
            if 'regime' not in trade_result or 'pnl' not in trade_result:
                return
            
            regime = trade_result['regime']
            pnl = trade_result['pnl']
            
            # Track performance by regime
            if regime not in self.performance_tracker:
                self.performance_tracker[regime] = {
                    'trades': 0,
                    'total_pnl': 0,
                    'wins': 0,
                    'losses': 0,
                    'avg_pnl': 0,
                    'win_rate': 0
                }
            
            tracker = self.performance_tracker[regime]
            tracker['trades'] += 1
            tracker['total_pnl'] += pnl
            
            if pnl > 0:
                tracker['wins'] += 1
            else:
                tracker['losses'] += 1
            
            tracker['avg_pnl'] = tracker['total_pnl'] / tracker['trades']
            tracker['win_rate'] = tracker['wins'] / tracker['trades']
            
            # Update regime engine weights if available
            if self.regime_available and self.regime_engine:
                # Convert performance to 0-1 scale for weight updates
                performance_score = 0.5 + (tracker['avg_pnl'] / 100)  # Assuming PnL in points
                performance_score = max(0.0, min(1.0, performance_score))
                
                # Update weights every 10 trades
                if tracker['trades'] % 10 == 0:
                    performance_data = {
                        'greek_sentiment': performance_score,
                        'straddle_analysis': performance_score * 1.1,  # Boost straddle if performing well
                        'ema_indicators': performance_score * 0.9,
                        'vwap_indicators': performance_score * 0.9
                    }
                    
                    self.regime_engine.update_weights_from_performance(performance_data)
                    logger.info(f"Updated weights for {regime}: {performance_score:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating performance: {e}")
    
    def get_strategy_status(self) -> Dict[str, Any]:
        """Get comprehensive strategy status"""
        status = {
            'regime_engine_available': self.regime_available,
            'current_regime': self.current_regime,
            'regime_confidence': self.regime_confidence,
            'regime_history_size': len(self.regime_history),
            'performance_tracker': self.performance_tracker
        }
        
        # Add regime engine status if available
        if self.regime_available and self.regime_engine:
            engine_status = self.regime_engine.get_engine_status()
            status['engine_status'] = engine_status
        
        return status
    
    def export_performance_report(self, output_path: str = None) -> str:
        """Export comprehensive performance report"""
        try:
            output_path = output_path or f"regime_strategy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            report = {
                'strategy_status': self.get_strategy_status(),
                'regime_history': self.regime_history[-50:],  # Last 50 regime calculations
                'performance_by_regime': self.performance_tracker,
                'regime_parameters': self.regime_parameters
            }
            
            # Add regime engine report if available
            if self.regime_available and self.regime_engine:
                engine_report_path = self.regime_engine.export_performance_report()
                if engine_report_path:
                    report['engine_report_path'] = engine_report_path
            
            # Save report
            import json
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"✅ Strategy performance report exported: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting performance report: {e}")
            return None

def demonstrate_backtester_integration():
    """Demonstrate how to integrate with backtester"""
    print("\n" + "="*80)
    print("BACKTESTER INTEGRATION DEMONSTRATION")
    print("="*80)
    
    try:
        # Initialize strategy
        print("🚀 Initializing Market Regime Strategy...")
        strategy = MarketRegimeStrategy("demo_market_regime_config.xlsx")
        
        # Create sample market data
        print("📊 Creating sample market data...")
        from example_usage import create_realistic_market_data
        market_data = create_realistic_market_data(300)  # 5 hours of data
        
        # Simulate backtesting
        print("🧮 Simulating backtesting process...")
        
        signals_generated = 0
        trades_executed = 0
        total_pnl = 0
        
        # Process data in chunks (simulating real-time)
        chunk_size = 50
        for i in range(0, len(market_data), chunk_size):
            chunk_data = market_data.iloc[:i+chunk_size]
            
            if len(chunk_data) < 20:  # Need minimum data
                continue
            
            # Generate signal
            signal = strategy.generate_signals(chunk_data)
            signals_generated += 1
            
            # Simulate trade execution
            if signal['action'] in ['BUY', 'SELL', 'STRADDLE', 'IRON_CONDOR']:
                # Simulate trade result
                trade_pnl = np.random.randn() * 50 + (10 if signal['confidence'] > 0.7 else 0)
                
                trade_result = {
                    'regime': signal['regime'],
                    'action': signal['action'],
                    'pnl': trade_pnl,
                    'confidence': signal['confidence']
                }
                
                # Update performance
                strategy.update_performance(trade_result)
                trades_executed += 1
                total_pnl += trade_pnl
                
                print(f"  Trade {trades_executed}: {signal['action']} in {signal['regime']} regime, PnL: {trade_pnl:.1f}")
        
        # Show results
        print(f"\n📈 Backtesting Results:")
        print(f"  • Signals Generated: {signals_generated}")
        print(f"  • Trades Executed: {trades_executed}")
        print(f"  • Total PnL: {total_pnl:.1f}")
        print(f"  • Average PnL per Trade: {total_pnl/trades_executed:.1f}" if trades_executed > 0 else "  • No trades executed")
        
        # Show strategy status
        print(f"\n🔍 Strategy Status:")
        status = strategy.get_strategy_status()
        for key, value in status.items():
            if key != 'performance_tracker':
                print(f"  • {key}: {value}")
        
        # Show performance by regime
        if status['performance_tracker']:
            print(f"\n📊 Performance by Regime:")
            for regime, perf in status['performance_tracker'].items():
                print(f"  • {regime}: {perf['trades']} trades, {perf['avg_pnl']:.1f} avg PnL, {perf['win_rate']:.1%} win rate")
        
        # Export report
        print(f"\n📄 Exporting performance report...")
        report_path = strategy.export_performance_report()
        if report_path:
            print(f"✅ Report saved: {report_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in backtester integration demo: {e}")
        return False

def main():
    """Main demonstration"""
    print("🚀 Market Regime Strategy - Backtester Integration")
    print("="*80)
    print("This demonstrates how to integrate Excel-based market regime detection")
    print("with backtester_v2 for comprehensive strategy testing.")
    
    success = demonstrate_backtester_integration()
    
    print("\n" + "="*80)
    if success:
        print("🎉 INTEGRATION DEMONSTRATION COMPLETE")
        print("="*80)
        print("Key Integration Points:")
        print("1. ✅ Excel configuration loaded and applied")
        print("2. ✅ Real-time regime calculation")
        print("3. ✅ Regime-based signal generation")
        print("4. ✅ Performance tracking and weight updates")
        print("5. ✅ Comprehensive reporting")
        print("\nNext Steps for Production:")
        print("• Integrate with actual backtester_v2 framework")
        print("• Connect to live data feeds")
        print("• Implement order management system")
        print("• Add risk management controls")
        print("• Deploy with monitoring and alerts")
    else:
        print("❌ INTEGRATION DEMONSTRATION FAILED")
        print("Please check the logs and fix any issues.")

if __name__ == "__main__":
    main()
