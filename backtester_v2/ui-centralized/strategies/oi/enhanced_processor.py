"""Enhanced OI processor with dynamic weightage support."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date, timedelta
import logging

from .enhanced_models import (
    EnhancedOIConfig, EnhancedLegConfig, DynamicWeightConfig,
    FactorConfig, PerformanceMetrics
)
from .enhanced_parser import EnhancedOIParser
from .dynamic_weight_engine import DynamicWeightEngine
from .adaptive_shift_manager import AdaptiveShiftManager, ShiftConfig
from .processor import OIProcessor  # Import existing processor
from .oi_analyzer import OIAnalyzer
from .query_builder import OIQueryBuilder

logger = logging.getLogger(__name__)

class EnhancedOIProcessor(OIProcessor):
    """Enhanced OI processor with dynamic weightage support."""
    
    def __init__(self, db_connection, enhanced_config: EnhancedOIConfig = None,
                 leg_configs: List[EnhancedLegConfig] = None,
                 weight_config: DynamicWeightConfig = None,
                 factor_configs: List[FactorConfig] = None,
                 shift_config: ShiftConfig = None):
        """Initialize enhanced OI processor."""
        super().__init__(db_connection)

        self.enhanced_config = enhanced_config
        self.leg_configs = leg_configs or []
        self.weight_config = weight_config
        self.factor_configs = factor_configs or []
        self.shift_config = shift_config or ShiftConfig()

        # Initialize dynamic weight engine if enabled
        self.weight_engine = None
        if (enhanced_config and enhanced_config.enable_dynamic_weights and
            weight_config and factor_configs):
            self.weight_engine = DynamicWeightEngine(weight_config, factor_configs)
            logger.info("Dynamic weight engine enabled")

        # Initialize adaptive shift manager
        self.shift_manager = AdaptiveShiftManager(self.shift_config)
        logger.info("Adaptive shift manager enabled")

        # Enhanced components
        self.parser = EnhancedOIParser()
        self.performance_tracker = {}
        self.market_condition_tracker = {}
        self.factor_data_cache = {}

        # Golden file format compatibility
        self.golden_file_columns = self._initialize_golden_file_format()
        
    def _initialize_golden_file_format(self) -> Dict[str, List[str]]:
        """Initialize golden file format structure for output compatibility."""
        return {
            'PortfolioParameter': ['Head', 'Value'],
            'GeneralParameter': [
                'StrategyName', 'MoveSlToCost', 'Underlying', 'Index', 'Weekdays', 'DTE',
                'StrikeSelectionTime', 'StartTime', 'LastEntryTime', 'EndTime', 'StrategyProfit',
                'StrategyLoss', 'StrategyProfitReExecuteNo', 'StrategyLossReExecuteNo',
                'StrategyTrailingType', 'PnLCalTime', 'LockPercent', 'TrailPercent',
                'SqOff1Time', 'SqOff1Percent', 'SqOff2Time', 'SqOff2Percent',
                'ProfitReaches', 'LockMinProfitAt', 'IncreaseInProfit', 'TrailMinProfitBy',
                'TgtTrackingFrom', 'TgtRegisterPriceFrom', 'SlTrackingFrom', 'SlRegisterPriceFrom',
                'PnLCalculationFrom', 'ConsiderHedgePnLForStgyPnL', 'StoplossCheckingInterval',
                'TargetCheckingInterval', 'ReEntryCheckingInterval', 'OnExpiryDayTradeNextExpiry'
            ],
            'LegParameter': [
                'StrategyName', 'IsIdle', 'LegID', 'Instrument', 'Transaction', 'Expiry',
                'W&Type', 'W&TValue', 'TrailW&T', 'StrikeMethod', 'MatchPremium', 'StrikeValue',
                'StrikePremiumCondition', 'SLType', 'SLValue', 'TGTType', 'TGTValue',
                'TrailSLType', 'SL_TrailAt', 'SL_TrailBy', 'Lots', 'ReEntryType',
                'ReEnteriesCount', 'OnEntry_OpenTradeOn', 'OnEntry_SqOffTradeOff',
                'OnEntry_SqOffAllLegs', 'OnEntry_OpenTradeDelay', 'OnEntry_SqOffDelay',
                'OnExit_OpenTradeOn', 'OnExit_SqOffTradeOff', 'OnExit_SqOffAllLegs',
                'OnExit_OpenAllLegs', 'OnExit_OpenTradeDelay', 'OnExit_SqOffDelay',
                'OpenHedge', 'HedgeStrikeMethod', 'HedgeStrikeValue', 'HedgeStrikePremiumCondition'
            ],
            'Metrics': ['Particulars', 'Combined', 'Strategy_Name'],
            'PORTFOLIO Trans': [
                'Portfolio Name', 'Strategy Name', 'ID', 'Entry Date', 'Enter On', 'Entry Day',
                'Exit Date', 'Exit at', 'Exit Day', 'Index', 'Expiry', 'Strike', 'CE/PE',
                'Trade', 'Qty', 'Entry at', 'Exit at.1', 'Points', 'Points After Slippage',
                'PNL', 'AfterSlippage', 'Taxes', 'Net PNL', 'Re-entry No', 'SL Re-entry No',
                'TGT Re-entry No', 'Reason', 'Strategy Entry No', 'Index At Entry',
                'Index At Exit', 'MaxProfit', 'MaxLoss'
            ]
        }
    
    def process_enhanced_strategy(self, start_date: date, end_date: date) -> Dict[str, Any]:
        """Process enhanced strategy with dynamic weights."""
        try:
            logger.info(f"Processing enhanced OI strategy from {start_date} to {end_date}")
            
            # Initialize results structure
            results = {
                'strategy_name': self.enhanced_config.strategy_name,
                'start_date': start_date,
                'end_date': end_date,
                'trades': [],
                'performance_metrics': {},
                'weight_adjustments': [],
                'golden_file_data': {},
                'processing_summary': {}
            }
            
            # Process each trading day
            current_date = start_date
            daily_results = []
            
            while current_date <= end_date:
                if self._is_trading_day(current_date):
                    daily_result = self._process_trading_day(current_date)
                    if daily_result:
                        daily_results.append(daily_result)
                        results['trades'].extend(daily_result.get('trades', []))
                
                current_date += timedelta(days=1)
            
            # Calculate final performance metrics
            results['performance_metrics'] = self._calculate_performance_metrics(results['trades'])
            
            # Generate golden file format output
            results['golden_file_data'] = self._generate_golden_file_output(results)
            
            # Add processing summary
            results['processing_summary'] = {
                'total_trading_days': len(daily_results),
                'total_trades': len(results['trades']),
                'dynamic_weights_enabled': self.weight_engine is not None,
                'weight_adjustments': len(results['weight_adjustments']),
                'processing_time': datetime.now()
            }
            
            logger.info(f"Enhanced strategy processing completed. Total trades: {len(results['trades'])}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing enhanced strategy: {e}")
            raise
    
    def _process_trading_day(self, trade_date: date) -> Dict[str, Any]:
        """Process a single trading day with enhanced logic."""
        try:
            # Calculate market conditions for the day
            market_conditions = self._calculate_market_conditions(trade_date)
            
            # Update weights if needed and enabled
            if self.weight_engine and self.weight_engine.should_rebalance():
                performance_data = self._calculate_factor_performance(trade_date)
                factor_data = self._get_factor_data(trade_date)
                
                updated_weights = self.weight_engine.update_weights(
                    performance_data, market_conditions, factor_data
                )
                
                # Store weight adjustment
                weight_adjustment = {
                    'date': trade_date,
                    'weights': updated_weights.copy(),
                    'market_conditions': market_conditions.copy(),
                    'performance_data': performance_data.copy()
                }
                
                logger.info(f"Updated weights for {trade_date}: {updated_weights}")
            
            # Get current weights for trading decisions
            current_weights = self.weight_engine.get_current_weights() if self.weight_engine else {}
            
            # Process trades for the day using enhanced logic
            daily_trades = self._execute_enhanced_trades(trade_date, current_weights, market_conditions)
            
            # Update performance tracking
            self._update_performance_tracking(daily_trades, trade_date)
            
            return {
                'date': trade_date,
                'trades': daily_trades,
                'weights': current_weights.copy(),
                'market_conditions': market_conditions.copy()
            }
            
        except Exception as e:
            logger.error(f"Error processing trading day {trade_date}: {e}")
            return None
    
    def _calculate_market_conditions(self, trade_date: date) -> Dict[str, float]:
        """Calculate market conditions for weight adjustment."""
        try:
            # This would typically query HeavyDB for market data
            # For now, implement basic market condition calculation
            
            # Query market data (simplified)
            market_data = self._get_market_data(trade_date)
            
            if market_data.empty:
                return {
                    'volatility': 0.5,
                    'trend_strength': 0.0,
                    'liquidity': 0.5,
                    'regime': 'normal'
                }
            
            # Calculate volatility (simplified)
            if 'close' in market_data.columns:
                returns = market_data['close'].pct_change().dropna()
                volatility = returns.std() if len(returns) > 1 else 0.5
            else:
                volatility = 0.5
            
            # Calculate trend strength (simplified)
            if 'close' in market_data.columns and len(market_data) > 20:
                short_ma = market_data['close'].rolling(5).mean().iloc[-1]
                long_ma = market_data['close'].rolling(20).mean().iloc[-1]
                trend_strength = (short_ma - long_ma) / long_ma if long_ma > 0 else 0.0
            else:
                trend_strength = 0.0
            
            # Calculate liquidity (simplified using volume if available)
            if 'volume' in market_data.columns:
                avg_volume = market_data['volume'].mean()
                recent_volume = market_data['volume'].iloc[-1] if not market_data.empty else avg_volume
                liquidity = min(1.0, recent_volume / avg_volume) if avg_volume > 0 else 0.5
            else:
                liquidity = 0.5
            
            # Determine regime
            if volatility > 0.8:
                regime = 'high_volatility'
            elif abs(trend_strength) > 0.02:
                regime = 'trending'
            else:
                regime = 'sideways'
            
            return {
                'volatility': min(1.0, volatility),
                'trend_strength': max(-1.0, min(1.0, trend_strength)),
                'liquidity': max(0.0, min(1.0, liquidity)),
                'regime': regime
            }
            
        except Exception as e:
            logger.warning(f"Error calculating market conditions: {e}")
            return {
                'volatility': 0.5,
                'trend_strength': 0.0,
                'liquidity': 0.5,
                'regime': 'normal'
            }
    
    def _calculate_factor_performance(self, trade_date: date) -> Dict[str, float]:
        """Calculate performance of each factor."""
        try:
            # This would calculate actual factor performance based on historical data
            # For now, return simulated performance data
            
            performance_data = {}
            
            # Get recent performance data
            recent_trades = self._get_recent_trades(trade_date, days=30)
            
            if recent_trades:
                # Calculate factor-specific performance
                performance_data['oi_factor'] = self._calculate_oi_factor_performance(recent_trades)
                performance_data['coi_factor'] = self._calculate_coi_factor_performance(recent_trades)
                performance_data['greek_factor'] = self._calculate_greek_factor_performance(recent_trades)
                performance_data['market_factor'] = self._calculate_market_factor_performance(recent_trades)
                performance_data['performance_factor'] = 0.5  # Neutral
            else:
                # Default performance values
                performance_data = {
                    'oi_factor': 0.5,
                    'coi_factor': 0.5,
                    'greek_factor': 0.5,
                    'market_factor': 0.5,
                    'performance_factor': 0.5
                }
            
            return performance_data
            
        except Exception as e:
            logger.warning(f"Error calculating factor performance: {e}")
            return {
                'oi_factor': 0.5,
                'coi_factor': 0.5,
                'greek_factor': 0.5,
                'market_factor': 0.5,
                'performance_factor': 0.5
            }
    
    def _get_factor_data(self, trade_date: date) -> Dict[str, pd.Series]:
        """Get factor data for correlation analysis."""
        try:
            # This would get actual factor data from the database
            # For now, return simulated factor data
            
            dates = pd.date_range(end=trade_date, periods=30, freq='D')
            
            factor_data = {
                'oi_factor': pd.Series(np.random.randn(30), index=dates),
                'coi_factor': pd.Series(np.random.randn(30), index=dates),
                'greek_factor': pd.Series(np.random.randn(30), index=dates),
                'market_factor': pd.Series(np.random.randn(30), index=dates),
                'performance_factor': pd.Series(np.random.randn(30), index=dates)
            }
            
            return factor_data
            
        except Exception as e:
            logger.warning(f"Error getting factor data: {e}")
            return {}
    
    def _execute_enhanced_trades(self, trade_date: date, weights: Dict[str, float], 
                                market_conditions: Dict[str, float]) -> List[Dict[str, Any]]:
        """Execute trades using enhanced logic with dynamic weights."""
        try:
            trades = []
            
            # Get OI rankings using enhanced logic
            rankings = self._get_enhanced_oi_rankings(trade_date, weights)
            
            if not rankings:
                return trades
            
            # Process each leg configuration
            for leg_config in self.leg_configs:
                trade = self._execute_leg_trade(leg_config, rankings, trade_date, weights, market_conditions)
                if trade:
                    trades.append(trade)
            
            return trades
            
        except Exception as e:
            logger.error(f"Error executing enhanced trades: {e}")
            return []
    
    def _get_enhanced_oi_rankings(self, trade_date: date, weights: Dict[str, float]) -> Optional[Any]:
        """Get OI rankings using enhanced logic with dynamic weights."""
        try:
            # For testing purposes, return simulated rankings
            # In production, this would use the actual OI analyzer
            rankings = {
                'CE': [
                    {'strike': 23000, 'oi': 1500000, 'weight_score': 0.85},
                    {'strike': 23050, 'oi': 1200000, 'weight_score': 0.75},
                    {'strike': 22950, 'oi': 1100000, 'weight_score': 0.70}
                ],
                'PE': [
                    {'strike': 23000, 'oi': 1400000, 'weight_score': 0.80},
                    {'strike': 22950, 'oi': 1300000, 'weight_score': 0.78},
                    {'strike': 23050, 'oi': 1000000, 'weight_score': 0.65}
                ]
            }

            # Apply dynamic weights to rankings if available
            if rankings and weights:
                rankings = self._apply_weights_to_rankings(rankings, weights)

            return rankings

        except Exception as e:
            logger.error(f"Error getting enhanced OI rankings: {e}")
            return None
    
    def _apply_weights_to_rankings(self, rankings: Any, weights: Dict[str, float]) -> Any:
        """Apply dynamic weights to OI rankings."""
        # This would implement the actual weight application logic
        # For now, return rankings as-is
        return rankings
    
    def _execute_leg_trade(self, leg_config: EnhancedLegConfig, rankings: Any, 
                          trade_date: date, weights: Dict[str, float], 
                          market_conditions: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Execute trade for a specific leg configuration."""
        try:
            # This would implement the actual trade execution logic
            # For now, return a simulated trade
            
            trade = {
                'strategy_name': leg_config.strategy_name,
                'leg_id': leg_config.leg_id,
                'trade_date': trade_date,
                'instrument': leg_config.instrument,
                'transaction': leg_config.transaction,
                'strike': 23000,  # Simulated
                'entry_price': 50.0,  # Simulated
                'exit_price': 45.0,  # Simulated
                'quantity': leg_config.lots * 75,  # Assuming 75 per lot
                'pnl': 375.0,  # Simulated
                'weights_used': weights.copy(),
                'market_conditions': market_conditions.copy()
            }
            
            return trade
            
        except Exception as e:
            logger.error(f"Error executing leg trade: {e}")
            return None

    def _calculate_performance_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        if not trades:
            return {}

        trades_df = pd.DataFrame(trades)

        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            strategy_name=self.enhanced_config.strategy_name
        )

        metrics.calculate_metrics(trades_df)

        return {
            'total_trades': metrics.total_trades,
            'winning_trades': metrics.winning_trades,
            'losing_trades': metrics.losing_trades,
            'total_pnl': metrics.total_pnl,
            'max_profit': metrics.max_profit,
            'max_loss': metrics.max_loss,
            'sharpe_ratio': metrics.sharpe_ratio,
            'max_drawdown': metrics.max_drawdown,
            'hit_rate': metrics.hit_rate,
            'avg_profit': metrics.avg_profit,
            'avg_loss': metrics.avg_loss,
            'profit_factor': metrics.profit_factor
        }

    def _generate_golden_file_output(self, results: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Generate output in golden file format."""
        try:
            golden_data = {}

            # PortfolioParameter sheet
            portfolio_params = [
                ['Portfolio Name', self.enhanced_config.strategy_name],
                ['Start Date', results['start_date'].strftime('%d_%m_%Y')],
                ['End Date', results['end_date'].strftime('%d_%m_%Y')],
                ['Total Trades', len(results['trades'])],
                ['Total PnL', results['performance_metrics'].get('total_pnl', 0)]
            ]
            golden_data['PortfolioParameter'] = pd.DataFrame(portfolio_params, columns=['Head', 'Value'])

            # GeneralParameter sheet
            general_params = []
            config = self.enhanced_config
            general_params.append([
                config.strategy_name, 'NO', config.underlying, config.index, config.weekdays,
                config.dte, config.strike_selection_time, config.start_time, config.last_entry_time,
                config.end_time, config.strategy_profit, config.strategy_loss, 0, 0,
                'Lock Minimum Profit', '151500', 0, 0, '000000', 0, '000000', 0,
                0, 0, 0, 0, 'LTP', 'LTP', 'LTP', 'LTP', 'LTP', 'NO', 10, 10, 10, 'NO'
            ])
            golden_data['GeneralParameter'] = pd.DataFrame(general_params, columns=self.golden_file_columns['GeneralParameter'])

            # LegParameter sheet
            leg_params = []
            for leg_config in self.leg_configs:
                leg_params.append([
                    leg_config.strategy_name, 'NO', leg_config.leg_id, leg_config.instrument,
                    leg_config.transaction, leg_config.expiry, 'NO', 0, 'NO', leg_config.strike_method,
                    'NO', leg_config.strike_value, 'NO', leg_config.sl_type, leg_config.sl_value,
                    leg_config.tgt_type, leg_config.tgt_value, leg_config.trail_sl_type,
                    leg_config.sl_trail_at, leg_config.sl_trail_by, leg_config.lots, 'NO', 0,
                    'NO', 'NO', 'NO', 0, 0, 'NO', 'NO', 'NO', 'NO', 0, 0,
                    'NO', 'ATM', 0, 'NO'
                ])
            golden_data['LegParameter'] = pd.DataFrame(leg_params, columns=self.golden_file_columns['LegParameter'])

            # Metrics sheet
            metrics = results['performance_metrics']
            metrics_data = [
                ['Total Trades', metrics.get('total_trades', 0), config.strategy_name],
                ['Winning Trades', metrics.get('winning_trades', 0), config.strategy_name],
                ['Losing Trades', metrics.get('losing_trades', 0), config.strategy_name],
                ['Total PnL', metrics.get('total_pnl', 0), config.strategy_name],
                ['Hit Rate %', metrics.get('hit_rate', 0) * 100, config.strategy_name],
                ['Sharpe Ratio', metrics.get('sharpe_ratio', 0), config.strategy_name],
                ['Max Drawdown', metrics.get('max_drawdown', 0), config.strategy_name]
            ]
            golden_data['Metrics'] = pd.DataFrame(metrics_data, columns=['Particulars', 'Combined', 'Strategy_Name'])

            # PORTFOLIO Trans sheet
            if results['trades']:
                trades_data = []
                for i, trade in enumerate(results['trades']):
                    trades_data.append([
                        config.strategy_name, trade['strategy_name'], i+1, trade['trade_date'],
                        config.start_time, trade['trade_date'].strftime('%A'), trade['trade_date'],
                        config.end_time, trade['trade_date'].strftime('%A'), config.index,
                        'current', trade['strike'], trade['instrument'], trade['transaction'],
                        trade['quantity'], trade['entry_price'], trade['exit_price'],
                        trade['exit_price'] - trade['entry_price'],
                        trade['exit_price'] - trade['entry_price'],  # After slippage
                        trade['pnl'], trade['pnl'], 0, trade['pnl'], 0, 0, 0, 'Normal Exit', 1,
                        23000, 23050, trade['pnl'] if trade['pnl'] > 0 else 0,
                        trade['pnl'] if trade['pnl'] < 0 else 0
                    ])
                golden_data['PORTFOLIO Trans'] = pd.DataFrame(trades_data, columns=self.golden_file_columns['PORTFOLIO Trans'])

            return golden_data

        except Exception as e:
            logger.error(f"Error generating golden file output: {e}")
            return {}

    def _is_trading_day(self, trade_date: date) -> bool:
        """Check if the given date is a trading day."""
        weekday = trade_date.weekday() + 1  # Monday = 1
        trading_days = [int(d) for d in self.enhanced_config.weekdays.split(',')]
        return weekday in trading_days

    def _get_market_data(self, trade_date: date) -> pd.DataFrame:
        """Get market data for the given date."""
        # This would query actual market data from HeavyDB
        # For now, return simulated data
        return pd.DataFrame({
            'date': [trade_date],
            'close': [23000],
            'volume': [1000000]
        })

    def _get_recent_trades(self, trade_date: date, days: int = 30) -> List[Dict[str, Any]]:
        """Get recent trades for performance calculation."""
        # This would query actual trade history
        # For now, return empty list
        return []

    def _calculate_oi_factor_performance(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate OI factor performance."""
        # Simplified performance calculation
        if not trades:
            return 0.5

        total_pnl = sum(trade.get('pnl', 0) for trade in trades)
        return 0.6 if total_pnl > 0 else 0.4

    def _calculate_coi_factor_performance(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate COI factor performance."""
        if not trades:
            return 0.5

        winning_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
        hit_rate = winning_trades / len(trades)
        return hit_rate

    def _calculate_greek_factor_performance(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate Greek factor performance."""
        if not trades:
            return 0.5

        # Simplified calculation based on trade consistency
        pnls = [trade.get('pnl', 0) for trade in trades]
        if len(pnls) > 1:
            std_dev = np.std(pnls)
            mean_pnl = np.mean(pnls)
            return 0.6 if std_dev < abs(mean_pnl) else 0.4

        return 0.5

    def _calculate_market_factor_performance(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate market factor performance."""
        if not trades:
            return 0.5

        # Simplified calculation based on recent performance
        recent_trades = trades[-10:] if len(trades) > 10 else trades
        recent_pnl = sum(trade.get('pnl', 0) for trade in recent_trades)
        return 0.6 if recent_pnl > 0 else 0.4

    def _update_performance_tracking(self, trades: List[Dict[str, Any]], trade_date: date):
        """Update performance tracking data."""
        if not trades:
            return

        daily_pnl = sum(trade.get('pnl', 0) for trade in trades)

        if trade_date not in self.performance_tracker:
            self.performance_tracker[trade_date] = {
                'trades': 0,
                'pnl': 0,
                'winning_trades': 0
            }

        self.performance_tracker[trade_date]['trades'] += len(trades)
        self.performance_tracker[trade_date]['pnl'] += daily_pnl
        self.performance_tracker[trade_date]['winning_trades'] += sum(1 for trade in trades if trade.get('pnl', 0) > 0)
