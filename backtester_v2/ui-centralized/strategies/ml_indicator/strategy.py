"""
ML Indicator Strategy Implementation

This module implements the main strategy logic for ML-based trading signals.
It integrates various technical indicators, market structure analysis, and
machine learning models to generate trading signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging

from ...core.base_strategy import BaseStrategy
from .models import MLSignal, MLTrade
from .indicators import (
    TALibWrapper,
    CustomIndicators,
    MarketStructure,
    VolumeProfile,
    OrderFlow,
    SMCIndicators,
    CandlestickPatterns
)
from .ml import (
    FeatureEngineering,
    SignalGeneration,
    ModelTraining,
    ModelEvaluation
)
from .constants import (
    DEFAULT_LOOKBACK_PERIOD,
    DEFAULT_PREDICTION_HORIZON,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_RISK_PERCENT,
    DEFAULT_POSITION_SIZE,
    INDICATOR_GROUPS,
    ML_MODELS
)

logger = logging.getLogger(__name__)


class MLIndicatorStrategy(BaseStrategy):
    """
    Machine Learning based trading strategy that combines multiple
    technical indicators and market analysis techniques.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ML Indicator strategy.
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self.config = config or {}
        
        # Initialize indicator modules
        self.talib = TALibWrapper()
        self.custom = CustomIndicators()
        self.market_structure = MarketStructure()
        self.volume_profile = VolumeProfile()
        self.orderflow = OrderFlow()
        self.smc = SMCIndicators()
        self.candlestick = CandlestickPatterns()
        
        # Initialize ML modules
        self.feature_eng = FeatureEngineering()
        self.signal_gen = SignalGeneration()
        self.model_training = ModelTraining()
        self.model_eval = ModelEvaluation()
        
        # Trading state
        self.positions = {}
        self.signals = []
        self.trades = []
        self.features_cache = {}
        self.model_cache = {}
        
        # Performance tracking
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0
        }
        
        logger.info(f"Initialized MLIndicatorStrategy with config: {config}")
    
    def parse_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse input data for ML Indicator strategy
        
        Args:
            input_data: Dictionary containing strategy configuration
            
        Returns:
            Parsed parameters dictionary
        """
        try:
            # For now, return a simple parsed structure
            # In production, this would parse Excel files or configuration
            parsed = {
                'portfolio': input_data.get('portfolio', {}),
                'indicators': input_data.get('indicators', []),
                'entry_conditions': input_data.get('entry_conditions', []),
                'exit_conditions': input_data.get('exit_conditions', []),
                'ml_config': input_data.get('ml_config', {})
            }
            
            return parsed
            
        except Exception as e:
            logger.error(f"Error parsing ML input: {str(e)}")
            raise ValueError(f"Failed to parse input: {str(e)}")
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators based on configuration.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with calculated indicators
        """
        logger.debug("Calculating technical indicators")
        
        # Make a copy to avoid modifying original data
        df = data.copy()
        
        # Standard TA-Lib indicators
        if self.config.use_talib_indicators:
            df = self.talib.calculate_all_indicators(
                df,
                self.config.talib_indicators
            )
        
        # Custom indicators
        if self.config.use_custom_indicators:
            df = self.custom.calculate_all_indicators(
                df,
                self.config.custom_indicators
            )
        
        # Market structure analysis
        if self.config.use_market_structure:
            df = self.market_structure.analyze(
                df,
                self.config.market_structure_params
            )
        
        # Volume profile
        if self.config.use_volume_profile:
            df = self.volume_profile.calculate(
                df,
                self.config.volume_profile_params
            )
        
        # Order flow analysis
        if self.config.use_orderflow:
            df = self.orderflow.analyze(
                df,
                self.config.orderflow_params
            )
        
        # Smart Money Concepts
        if self.config.use_smc:
            df = self.smc.analyze(
                df,
                self.config.smc_params
            )
        
        # Candlestick patterns
        if self.config.use_candlestick_patterns:
            df = self.candlestick.detect_patterns(
                df,
                self.config.candlestick_patterns
            )
        
        return df
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for ML model.
        
        Args:
            data: DataFrame with calculated indicators
            
        Returns:
            DataFrame with engineered features
        """
        logger.debug("Preparing features for ML model")
        
        # Engineer features
        features = self.feature_eng.create_features(
            data,
            lookback_period=self.config.lookback_period,
            feature_groups=self.config.feature_groups
        )
        
        # Handle missing values
        features = self.feature_eng.handle_missing_values(
            features,
            method=self.config.missing_value_method
        )
        
        # Scale features if required
        if self.config.scale_features:
            features = self.feature_eng.scale_features(
                features,
                method=self.config.scaling_method
            )
        
        # Feature selection
        if self.config.feature_selection:
            features = self.feature_eng.select_features(
                features,
                method=self.config.feature_selection_method,
                n_features=self.config.n_features
            )
        
        return features
    
    def generate_signals(
        self,
        features: pd.DataFrame,
        model: Optional[Any] = None
    ) -> List[MLSignal]:
        """
        Generate trading signals using ML model.
        
        Args:
            features: DataFrame with prepared features
            model: Pre-trained model (optional)
            
        Returns:
            List of ML trading signals
        """
        logger.debug("Generating ML signals")
        
        # Load or train model
        if model is None:
            if self.config.model_path:
                model = self.signal_gen.load_model(self.config.model_path)
            else:
                logger.warning("No model provided, using rule-based signals")
                return self._generate_rule_based_signals(features)
        
        # Generate predictions
        predictions = self.signal_gen.predict(
            model,
            features,
            prediction_type=self.config.prediction_type
        )
        
        # Convert predictions to signals
        signals = []
        for idx, row in predictions.iterrows():
            if row['confidence'] >= self.config.confidence_threshold:
                signal = MLSignal(
                    timestamp=idx,
                    symbol=self.config.symbol,
                    direction=row['direction'],
                    confidence=row['confidence'],
                    predicted_move=row.get('predicted_move', 0),
                    features=row.get('important_features', {}),
                    model_name=self.config.model_name
                )
                signals.append(signal)
        
        return signals
    
    def _generate_rule_based_signals(
        self,
        features: pd.DataFrame
    ) -> List[MLSignal]:
        """
        Generate signals based on indicator rules when no ML model is available.
        
        Args:
            features: DataFrame with indicators
            
        Returns:
            List of rule-based signals
        """
        signals = []
        
        # Example rule-based logic
        for idx, row in features.iterrows():
            # Long signal conditions
            long_conditions = []
            short_conditions = []
            
            # RSI conditions
            if 'rsi' in row:
                long_conditions.append(row['rsi'] < 30)
                short_conditions.append(row['rsi'] > 70)
            
            # MACD conditions
            if 'macd' in row and 'macd_signal' in row:
                long_conditions.append(row['macd'] > row['macd_signal'])
                short_conditions.append(row['macd'] < row['macd_signal'])
            
            # Volume conditions
            if 'volume_ratio' in row:
                long_conditions.append(row['volume_ratio'] > 1.5)
                short_conditions.append(row['volume_ratio'] > 1.5)
            
            # Generate signals
            if all(long_conditions) and len(long_conditions) > 0:
                signal = MLSignal(
                    timestamp=idx,
                    symbol=self.config.symbol,
                    direction='LONG',
                    confidence=0.7,
                    predicted_move=0,
                    features={'rule_based': True},
                    model_name='rule_based'
                )
                signals.append(signal)
            elif all(short_conditions) and len(short_conditions) > 0:
                signal = MLSignal(
                    timestamp=idx,
                    symbol=self.config.symbol,
                    direction='SHORT',
                    confidence=0.7,
                    predicted_move=0,
                    features={'rule_based': True},
                    model_name='rule_based'
                )
                signals.append(signal)
        
        return signals
    
    def execute_trades(
        self,
        signals: List[MLSignal],
        current_data: pd.DataFrame
    ) -> List[MLTrade]:
        """
        Execute trades based on signals and risk management rules.
        
        Args:
            signals: List of ML signals
            current_data: Current market data
            
        Returns:
            List of executed trades
        """
        logger.debug(f"Executing trades for {len(signals)} signals")
        
        trades = []
        
        for signal in signals:
            # Check if we already have a position
            if signal.symbol in self.positions:
                # Check for exit conditions
                if self._should_exit_position(signal, current_data):
                    trade = self._close_position(signal, current_data)
                    if trade:
                        trades.append(trade)
            else:
                # Check entry conditions
                if self._should_enter_position(signal, current_data):
                    trade = self._open_position(signal, current_data)
                    if trade:
                        trades.append(trade)
        
        # Check stop loss and take profit for existing positions
        self._check_exit_conditions(current_data, trades)
        
        return trades
    
    def _should_enter_position(
        self,
        signal: MLSignal,
        current_data: pd.DataFrame
    ) -> bool:
        """
        Check if position entry conditions are met.
        
        Args:
            signal: ML signal
            current_data: Current market data
            
        Returns:
            True if should enter position
        """
        # Check signal confidence
        if signal.confidence < self.config.confidence_threshold:
            return False
        
        # Check market conditions
        if self.config.check_market_conditions:
            market_favorable = self._check_market_conditions(
                signal.direction,
                current_data
            )
            if not market_favorable:
                return False
        
        # Check risk limits
        if len(self.positions) >= self.config.max_positions:
            return False
        
        # Check correlation with existing positions
        if self.config.check_correlation:
            if self._position_correlated(signal.symbol):
                return False
        
        return True
    
    def _should_exit_position(
        self,
        signal: MLSignal,
        current_data: pd.DataFrame
    ) -> bool:
        """
        Check if position exit conditions are met.
        
        Args:
            signal: ML signal
            current_data: Current market data
            
        Returns:
            True if should exit position
        """
        position = self.positions.get(signal.symbol)
        if not position:
            return False
        
        # Check for opposite signal
        if signal.direction != position.direction:
            return True
        
        # Check time-based exit
        if self.config.max_holding_period:
            holding_time = datetime.now() - position.entry_time
            if holding_time > timedelta(hours=self.config.max_holding_period):
                return True
        
        return False
    
    def _open_position(
        self,
        signal: MLSignal,
        current_data: pd.DataFrame
    ) -> Optional[MLTrade]:
        """
        Open a new position based on signal.
        
        Args:
            signal: ML signal
            current_data: Current market data
            
        Returns:
            MLTrade object if position opened
        """
        try:
            # Get current price
            current_price = current_data.iloc[-1]['close']
            
            # Calculate position size
            position_size = self._calculate_position_size(
                signal,
                current_price
            )
            
            # Calculate stop loss and take profit
            stop_loss, take_profit = self._calculate_sl_tp(
                signal,
                current_price
            )
            
            # Create trade
            trade = MLTrade(
                signal=signal,
                entry_price=current_price,
                position_size=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                entry_time=datetime.now()
            )
            
            # Update positions
            self.positions[signal.symbol] = trade
            
            logger.info(f"Opened position: {trade}")
            return trade
            
        except Exception as e:
            logger.error(f"Error opening position: {e}")
            return None
    
    def _close_position(
        self,
        signal: MLSignal,
        current_data: pd.DataFrame
    ) -> Optional[MLTrade]:
        """
        Close an existing position.
        
        Args:
            signal: ML signal (can be exit signal)
            current_data: Current market data
            
        Returns:
            MLTrade object with exit details
        """
        try:
            position = self.positions.get(signal.symbol)
            if not position:
                return None
            
            # Get current price
            current_price = current_data.iloc[-1]['close']
            
            # Update trade with exit details
            position.exit_price = current_price
            position.exit_time = datetime.now()
            position.pnl = self._calculate_pnl(position)
            position.exit_reason = signal.direction if signal else 'stop_loss'
            
            # Update performance metrics
            self._update_performance_metrics(position)
            
            # Remove from positions
            del self.positions[signal.symbol]
            
            logger.info(f"Closed position: {position}")
            return position
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return None
    
    def _calculate_position_size(
        self,
        signal: MLSignal,
        current_price: float
    ) -> float:
        """
        Calculate position size based on risk management rules.
        
        Args:
            signal: ML signal
            current_price: Current market price
            
        Returns:
            Position size
        """
        # Fixed position size
        if self.config.position_sizing == 'fixed':
            return self.config.fixed_position_size
        
        # Risk-based position sizing
        elif self.config.position_sizing == 'risk_based':
            account_size = self.config.account_size
            risk_amount = account_size * (self.config.risk_percent / 100)
            
            # Calculate stop distance
            stop_distance = current_price * (self.config.default_stop_percent / 100)
            
            # Position size = Risk Amount / Stop Distance
            position_size = risk_amount / stop_distance
            
            return min(position_size, self.config.max_position_size)
        
        # Kelly criterion
        elif self.config.position_sizing == 'kelly':
            win_rate = self.performance_metrics['win_rate']
            avg_win = self._calculate_average_win()
            avg_loss = self._calculate_average_loss()
            
            if avg_loss > 0:
                win_loss_ratio = avg_win / avg_loss
                kelly_percent = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
                kelly_percent = max(0, min(kelly_percent, 0.25))  # Cap at 25%
                
                return self.config.account_size * kelly_percent
        
        return self.config.default_position_size
    
    def _calculate_sl_tp(
        self,
        signal: MLSignal,
        entry_price: float
    ) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit levels.
        
        Args:
            signal: ML signal
            entry_price: Entry price
            
        Returns:
            Tuple of (stop_loss, take_profit)
        """
        # Use predicted move if available
        if signal.predicted_move > 0:
            if signal.direction == 'LONG':
                take_profit = entry_price * (1 + signal.predicted_move)
                stop_loss = entry_price * (1 - signal.predicted_move * self.config.risk_reward_ratio)
            else:
                take_profit = entry_price * (1 - signal.predicted_move)
                stop_loss = entry_price * (1 + signal.predicted_move * self.config.risk_reward_ratio)
        else:
            # Use default percentages
            if signal.direction == 'LONG':
                stop_loss = entry_price * (1 - self.config.default_stop_percent / 100)
                take_profit = entry_price * (1 + self.config.default_tp_percent / 100)
            else:
                stop_loss = entry_price * (1 + self.config.default_stop_percent / 100)
                take_profit = entry_price * (1 - self.config.default_tp_percent / 100)
        
        return stop_loss, take_profit
    
    def _check_exit_conditions(
        self,
        current_data: pd.DataFrame,
        trades: List[MLTrade]
    ) -> None:
        """
        Check stop loss and take profit conditions for all positions.
        
        Args:
            current_data: Current market data
            trades: List to append exit trades
        """
        current_price = current_data.iloc[-1]['close']
        
        for symbol, position in list(self.positions.items()):
            exit_trade = False
            
            # Check stop loss
            if position.direction == 'LONG':
                if current_price <= position.stop_loss:
                    exit_trade = True
                    position.exit_reason = 'stop_loss'
                elif current_price >= position.take_profit:
                    exit_trade = True
                    position.exit_reason = 'take_profit'
            else:  # SHORT
                if current_price >= position.stop_loss:
                    exit_trade = True
                    position.exit_reason = 'stop_loss'
                elif current_price <= position.take_profit:
                    exit_trade = True
                    position.exit_reason = 'take_profit'
            
            # Execute exit if conditions met
            if exit_trade:
                exit_signal = MLSignal(
                    timestamp=current_data.index[-1],
                    symbol=symbol,
                    direction='EXIT',
                    confidence=1.0,
                    predicted_move=0,
                    features={'exit_reason': position.exit_reason},
                    model_name='exit_manager'
                )
                
                closed_trade = self._close_position(exit_signal, current_data)
                if closed_trade:
                    trades.append(closed_trade)
    
    def _calculate_pnl(self, trade: MLTrade) -> float:
        """
        Calculate profit/loss for a trade.
        
        Args:
            trade: MLTrade object
            
        Returns:
            PnL amount
        """
        if trade.direction == 'LONG':
            pnl = (trade.exit_price - trade.entry_price) * trade.position_size
        else:  # SHORT
            pnl = (trade.entry_price - trade.exit_price) * trade.position_size
        
        # Subtract transaction costs
        pnl -= self.config.transaction_cost * trade.position_size * 2
        
        return pnl
    
    def _update_performance_metrics(self, trade: MLTrade) -> None:
        """
        Update performance metrics after trade completion.
        
        Args:
            trade: Completed MLTrade object
        """
        self.performance_metrics['total_trades'] += 1
        
        if trade.pnl > 0:
            self.performance_metrics['winning_trades'] += 1
        else:
            self.performance_metrics['losing_trades'] += 1
        
        self.performance_metrics['total_pnl'] += trade.pnl
        
        # Update win rate
        if self.performance_metrics['total_trades'] > 0:
            self.performance_metrics['win_rate'] = (
                self.performance_metrics['winning_trades'] /
                self.performance_metrics['total_trades']
            )
    
    def _check_market_conditions(
        self,
        direction: str,
        current_data: pd.DataFrame
    ) -> bool:
        """
        Check if market conditions are favorable for trading.
        
        Args:
            direction: Trade direction (LONG/SHORT)
            current_data: Current market data
            
        Returns:
            True if conditions are favorable
        """
        # Check volatility
        if 'atr' in current_data.columns:
            current_volatility = current_data.iloc[-1]['atr']
            avg_volatility = current_data['atr'].rolling(20).mean().iloc[-1]
            
            if current_volatility > avg_volatility * self.config.max_volatility_filter:
                return False
        
        # Check trend alignment
        if self.config.trend_filter and 'ema_200' in current_data.columns:
            current_price = current_data.iloc[-1]['close']
            ema_200 = current_data.iloc[-1]['ema_200']
            
            if direction == 'LONG' and current_price < ema_200:
                return False
            elif direction == 'SHORT' and current_price > ema_200:
                return False
        
        # Check volume
        if 'volume' in current_data.columns:
            current_volume = current_data.iloc[-1]['volume']
            avg_volume = current_data['volume'].rolling(20).mean().iloc[-1]
            
            if current_volume < avg_volume * self.config.min_volume_filter:
                return False
        
        return True
    
    def _position_correlated(self, symbol: str) -> bool:
        """
        Check if new position would be too correlated with existing positions.
        
        Args:
            symbol: Symbol to check
            
        Returns:
            True if position is too correlated
        """
        # Simple implementation - can be enhanced with actual correlation calculation
        # For now, just check if we already have positions in similar instruments
        
        if self.config.max_correlated_positions <= 0:
            return False
        
        # Count positions in same sector/type
        correlated_count = 0
        for pos_symbol in self.positions:
            if self._are_symbols_correlated(symbol, pos_symbol):
                correlated_count += 1
        
        return correlated_count >= self.config.max_correlated_positions
    
    def _are_symbols_correlated(self, symbol1: str, symbol2: str) -> bool:
        """
        Check if two symbols are correlated.
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            
        Returns:
            True if symbols are correlated
        """
        # Simple implementation - can be enhanced
        # For now, just check if they're the same base symbol
        base1 = symbol1.split('_')[0]
        base2 = symbol2.split('_')[0]
        
        return base1 == base2
    
    def _calculate_average_win(self) -> float:
        """Calculate average winning trade amount."""
        winning_trades = [t for t in self.trades if t.pnl > 0]
        if not winning_trades:
            return 0
        return sum(t.pnl for t in winning_trades) / len(winning_trades)
    
    def _calculate_average_loss(self) -> float:
        """Calculate average losing trade amount."""
        losing_trades = [t for t in self.trades if t.pnl < 0]
        if not losing_trades:
            return 0
        return abs(sum(t.pnl for t in losing_trades) / len(losing_trades))
    
    def run(
        self,
        data: pd.DataFrame,
        model: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Run the ML indicator strategy on provided data.
        
        Args:
            data: Market data DataFrame
            model: Pre-trained ML model (optional)
            
        Returns:
            Dictionary containing trades and performance metrics
        """
        logger.info(f"Running ML Indicator Strategy on {len(data)} data points")
        
        try:
            # Calculate indicators
            data_with_indicators = self.calculate_indicators(data)
            
            # Prepare features
            features = self.prepare_features(data_with_indicators)
            
            # Generate signals
            signals = self.generate_signals(features, model)
            
            # Execute trades
            trades = self.execute_trades(signals, data_with_indicators)
            
            # Store results
            self.signals.extend(signals)
            self.trades.extend(trades)
            
            # Return results
            return {
                'trades': trades,
                'signals': signals,
                'performance_metrics': self.performance_metrics,
                'final_positions': self.positions.copy()
            }
            
        except Exception as e:
            logger.error(f"Error running ML Indicator Strategy: {e}")
            raise
    
    def generate_query(self, params: Dict[str, Any]) -> List[str]:
        """
        Generate SQL queries for ML Indicator strategy
        
        Args:
            params: Parameters from parsed input
            
        Returns:
            List of SQL query strings
        """
        # For now, return a simple query
        # In production, this would build complex queries with indicators
        query = f"""
        SELECT 
            trade_date,
            trade_time,
            strike,
            expiry_date,
            spot,
            ce_open as open_price,
            ce_high as high_price,
            ce_low as low_price,
            ce_close as close_price,
            ce_volume as volume,
            ce_oi as open_interest,
            ce_iv as implied_volatility,
            ce_delta as delta,
            ce_gamma as gamma,
            ce_theta as theta,
            ce_vega as vega
        FROM nifty_option_chain
        WHERE trade_date >= '2024-04-01'
          AND trade_date <= '2024-04-01'
        ORDER BY trade_date, trade_time
        LIMIT 1000
        """
        return [query]
    
    def process_results(self, results: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw query results
        
        Args:
            results: Query results DataFrame
            params: Original parameters
            
        Returns:
            Processed results dictionary
        """
        # For now, return basic processing
        return {
            'trades': self.trades,
            'signals': self.signals,
            'performance_metrics': self.performance_metrics,
            'raw_data': results
        }
    
    def validate_input(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate input data
        
        Args:
            data: Input data to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Basic validation
        if not data:
            errors.append("Input data is empty")
            
        # Check for required fields based on input type
        if 'portfolio_file' in data or 'ml_config_file' in data:
            # File-based input validation
            if 'portfolio_file' in data and not data['portfolio_file']:
                errors.append("portfolio_file is required")
            if 'ml_config_file' in data and not data['ml_config_file']:
                errors.append("ml_config_file is required")
        else:
            # Direct config validation
            if 'portfolio' not in data:
                errors.append("portfolio configuration is required")
                
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary.
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            'total_trades': self.performance_metrics['total_trades'],
            'winning_trades': self.performance_metrics['winning_trades'],
            'losing_trades': self.performance_metrics['losing_trades'],
            'win_rate': self.performance_metrics['win_rate'],
            'total_pnl': self.performance_metrics['total_pnl'],
            'average_win': self._calculate_average_win(),
            'average_loss': self._calculate_average_loss(),
            'profit_factor': self._calculate_profit_factor(),
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'max_drawdown': self._calculate_max_drawdown(),
            'trades': len(self.trades),
            'open_positions': len(self.positions)
        }
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0
        
        return gross_profit / gross_loss
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio of returns."""
        if not self.trades:
            return 0
        
        returns = [t.pnl for t in self.trades]
        if len(returns) < 2:
            return 0
        
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
        
        # Annualized Sharpe ratio (assuming daily returns)
        return (avg_return / std_return) * np.sqrt(252)
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if not self.trades:
            return 0
        
        cumulative_pnl = []
        running_pnl = 0
        
        for trade in self.trades:
            running_pnl += trade.pnl
            cumulative_pnl.append(running_pnl)
        
        peak = cumulative_pnl[0]
        max_dd = 0
        
        for pnl in cumulative_pnl:
            if pnl > peak:
                peak = pnl
            
            drawdown = (peak - pnl) / peak if peak > 0 else 0
            max_dd = max(max_dd, drawdown)
        
        return max_dd