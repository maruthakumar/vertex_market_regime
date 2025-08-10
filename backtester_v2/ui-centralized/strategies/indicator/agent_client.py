#!/usr/bin/env python3
"""
BMAD IND (Indicator) Strategy Agent Client
Specialized gRPC client for Indicator strategy with 200+ TA-Lib indicators
SuperClaude v3 Implementation - Phase 2: Strategy Agent Implementation
"""

import asyncio
import grpc
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# BMAD agent base imports
import sys
from pathlib import Path
current_dir = Path(__file__).parent
bmad_dir = current_dir.parent.parent / "bmad_agent_communication"
if str(bmad_dir) not in sys.path:
    sys.path.insert(0, str(bmad_dir))

from grpc_base import BMadAgentBase, AgentConfig, create_agent_config
from generated import bmad_agents_pb2, bmad_agents_pb2_grpc

# Strategy imports
from .strategy import IndicatorStrategy
from .indicators import (
    TALibWrapper,
    CustomIndicators,
    MarketStructure,
    VolumeProfile,
    OrderFlow,
    SMCIndicators,
    CandlestickPatterns
)
from .models import IndicatorConfig, IndicatorSignal, IndicatorTrade

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class IndicatorAgentConfig:
    """Configuration for IND strategy agent"""
    talib_indicators: List[str]
    custom_indicators: List[str] 
    smc_indicators: List[str]
    volume_analysis: bool = True
    candlestick_patterns: bool = True
    market_structure_analysis: bool = True
    signal_threshold: float = 0.7
    max_positions: int = 5
    risk_percent: float = 2.0
    lookback_period: int = 50

class BMadIndStrategyAgent(BMadAgentBase):
    """BMAD Indicator Strategy Agent with 200+ TA-Lib indicators"""
    
    def __init__(self, config: AgentConfig, strategy_config: IndicatorAgentConfig):
        super().__init__(config)
        self.strategy_config = strategy_config
        
        # Initialize strategy components
        self.strategy = IndicatorStrategy()
        self.talib_wrapper = TALibWrapper()
        self.custom_indicators = CustomIndicators()
        self.market_structure = MarketStructure()
        self.volume_profile = VolumeProfile()
        self.orderflow = OrderFlow()
        self.smc_indicators = SMCIndicators()
        self.candlestick_patterns = CandlestickPatterns()
        
        # Trading state
        self.active_positions: Dict[str, IndicatorTrade] = {}
        self.signal_history: List[IndicatorSignal] = []
        self.performance_metrics: Dict[str, float] = {}
        
        # Cache for computed indicators
        self.indicator_cache: Dict[str, pd.DataFrame] = {}
        self.last_data_update: Optional[datetime] = None
        
        logger.info(f"✅ IND Strategy Agent initialized: {config.agent_id}")
    
    async def _register_services(self):
        """Register gRPC services for IND agent"""
        try:
            # Add strategy agent service
            bmad_agents_pb2_grpc.add_StrategyAgentServiceServicer_to_server(
                self.IndicatorAgentServicer(self), 
                self.server
            )
            logger.info(f"✅ Strategy services registered for {self.config.agent_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to register services for {self.config.agent_id}: {e}")
            raise
    
    class IndicatorAgentServicer(bmad_agents_pb2_grpc.StrategyAgentServiceServicer):
        """gRPC service implementation for IND agent"""
        
        def __init__(self, agent):
            self.agent = agent
        
        async def CalculateIndicators(self, request, context):
            """Calculate TA-Lib and custom indicators"""
            try:
                logger.info(f"📊 Calculating indicators request from {request.requester_id}")
                
                # Parse request data
                market_data = pd.DataFrame(json.loads(request.market_data))
                indicator_configs = json.loads(request.indicator_configs)
                
                # Calculate all requested indicators
                results = await self.agent._calculate_all_indicators(
                    market_data, 
                    indicator_configs
                )
                
                # Return response
                return bmad_agents_pb2.CalculateIndicatorsResponse(
                    success=True,
                    indicator_data=json.dumps(results.to_dict('records')),
                    metadata=json.dumps({
                        "indicators_calculated": len(indicator_configs),
                        "data_points": len(results),
                        "calculation_time": datetime.now(timezone.utc).isoformat()
                    })
                )
                
            except Exception as e:
                logger.error(f"❌ Error calculating indicators: {e}")
                return bmad_agents_pb2.CalculateIndicatorsResponse(
                    success=False,
                    error_message=str(e)
                )
        
        async def GenerateSignals(self, request, context):
            """Generate trading signals based on indicators"""
            try:
                logger.info(f"🎯 Generating signals request from {request.requester_id}")
                
                # Parse request data
                indicator_data = pd.DataFrame(json.loads(request.indicator_data))
                signal_config = json.loads(request.signal_config)
                
                # Generate signals
                signals = await self.agent._generate_trading_signals(
                    indicator_data, 
                    signal_config
                )
                
                # Convert signals to protobuf format
                signal_messages = []
                for signal in signals:
                    signal_msg = bmad_agents_pb2.Signal(
                        timestamp=signal.timestamp.isoformat(),
                        symbol=signal.symbol,
                        direction=signal.direction,
                        confidence=signal.confidence,
                        predicted_move=signal.predicted_move,
                        indicators=json.dumps(signal.indicators),
                        signal_source=signal.signal_source
                    )
                    signal_messages.append(signal_msg)
                
                return bmad_agents_pb2.GenerateSignalsResponse(
                    success=True,
                    signals=signal_messages,
                    metadata=json.dumps({
                        "signals_generated": len(signals),
                        "avg_confidence": np.mean([s.confidence for s in signals]) if signals else 0,
                        "generation_time": datetime.now(timezone.utc).isoformat()
                    })
                )
                
            except Exception as e:
                logger.error(f"❌ Error generating signals: {e}")
                return bmad_agents_pb2.GenerateSignalsResponse(
                    success=False,
                    error_message=str(e)
                )
        
        async def ValidateParameters(self, request, context):
            """Validate indicator parameters"""
            try:
                logger.info(f"✅ Parameter validation request from {request.requester_id}")
                
                # Parse parameter data
                parameters = json.loads(request.parameters)
                
                # Validate all parameters
                validation_results = await self.agent._validate_indicator_parameters(parameters)
                
                return bmad_agents_pb2.ValidateParametersResponse(
                    success=True,
                    validation_results=json.dumps(validation_results),
                    is_valid=all(r["is_valid"] for r in validation_results.values()),
                    error_count=sum(1 for r in validation_results.values() if not r["is_valid"])
                )
                
            except Exception as e:
                logger.error(f"❌ Error validating parameters: {e}")
                return bmad_agents_pb2.ValidateParametersResponse(
                    success=False,
                    error_message=str(e)
                )
        
        async def GetPerformanceMetrics(self, request, context):
            """Get agent performance metrics"""
            try:
                metrics = await self.agent._collect_agent_specific_metrics()
                
                return bmad_agents_pb2.GetPerformanceMetricsResponse(
                    success=True,
                    metrics=json.dumps(metrics),
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
                
            except Exception as e:
                logger.error(f"❌ Error getting performance metrics: {e}")
                return bmad_agents_pb2.GetPerformanceMetricsResponse(
                    success=False,
                    error_message=str(e)
                )
    
    async def _calculate_all_indicators(
        self, 
        market_data: pd.DataFrame, 
        indicator_configs: Dict[str, Any]
    ) -> pd.DataFrame:
        """Calculate all requested indicators"""
        try:
            logger.info(f"📊 Calculating {len(indicator_configs)} indicators")
            
            # Start with market data copy
            result_data = market_data.copy()
            
            # Calculate TA-Lib indicators
            if "talib_indicators" in indicator_configs:
                for indicator_config in indicator_configs["talib_indicators"]:
                    try:
                        indicator_name = indicator_config["name"]
                        indicator_params = indicator_config.get("params", {})
                        
                        # Get SQL implementation from TA-Lib wrapper
                        sql_query = self.talib_wrapper.get_indicator_sql(
                            indicator_name,
                            indicator_params,
                            "market_data",
                            indicator_config.get("price_column", "close_price")
                        )
                        
                        # For now, calculate using pandas (production would use HeavyDB)
                        indicator_values = self._calculate_talib_indicator_pandas(
                            result_data,
                            indicator_name,
                            indicator_params
                        )
                        
                        # Add to result data
                        column_name = f"{indicator_name}_{indicator_params.get('timeperiod', '')}"
                        result_data[column_name] = indicator_values
                        
                        logger.debug(f"✅ Calculated {indicator_name}")
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to calculate {indicator_config.get('name', 'unknown')}: {e}")
            
            # Calculate custom indicators
            if "custom_indicators" in indicator_configs:
                custom_results = await self.custom_indicators.calculate_all_indicators(
                    result_data,
                    indicator_configs["custom_indicators"]
                )
                result_data = custom_results
            
            # Calculate market structure indicators
            if "market_structure" in indicator_configs:
                structure_results = await self.market_structure.analyze(
                    result_data,
                    indicator_configs["market_structure"]
                )
                result_data = structure_results
            
            # Calculate volume profile
            if "volume_profile" in indicator_configs:
                volume_results = await self.volume_profile.calculate(
                    result_data,
                    indicator_configs["volume_profile"]
                )
                result_data = volume_results
            
            # Calculate order flow
            if "orderflow" in indicator_configs:
                flow_results = await self.orderflow.analyze(
                    result_data,
                    indicator_configs["orderflow"]
                )
                result_data = flow_results
            
            # Calculate Smart Money Concepts (Enhanced Implementation)
            if "smc_indicators" in indicator_configs:
                smc_results = await self._calculate_smc_indicators(
                    result_data,
                    indicator_configs["smc_indicators"]
                )
                result_data = smc_results
            
            # Detect candlestick patterns
            if "candlestick_patterns" in indicator_configs:
                pattern_results = await self.candlestick_patterns.detect_patterns(
                    result_data,
                    indicator_configs["candlestick_patterns"]
                )
                result_data = pattern_results
            
            # Cache results
            cache_key = f"indicators_{len(indicator_configs)}_{datetime.now().strftime('%H%M')}"
            self.indicator_cache[cache_key] = result_data
            
            logger.info(f"✅ Calculated all indicators: {len(result_data.columns)} total columns")
            return result_data
            
        except Exception as e:
            logger.error(f"❌ Error calculating indicators: {e}")
            raise
    
    def _calculate_talib_indicator_pandas(
        self,
        data: pd.DataFrame,
        indicator_name: str,
        params: Dict[str, Any]
    ) -> pd.Series:
        """Calculate TA-Lib indicator using pandas (fallback)"""
        try:
            import talib
            
            # Extract OHLCV data
            open_prices = data['open_price'].values if 'open_price' in data.columns else data['close_price'].values
            high_prices = data['high_price'].values if 'high_price' in data.columns else data['close_price'].values
            low_prices = data['low_price'].values if 'low_price' in data.columns else data['close_price'].values
            close_prices = data['close_price'].values
            volume = data['volume'].values if 'volume' in data.columns else np.ones(len(close_prices))
            
            # Calculate based on indicator type
            if indicator_name == "SMA":
                return pd.Series(talib.SMA(close_prices, timeperiod=params.get('timeperiod', 20)))
            elif indicator_name == "EMA":
                return pd.Series(talib.EMA(close_prices, timeperiod=params.get('timeperiod', 20)))
            elif indicator_name == "RSI":
                return pd.Series(talib.RSI(close_prices, timeperiod=params.get('timeperiod', 14)))
            elif indicator_name == "MACD":
                macd, signal, hist = talib.MACD(
                    close_prices,
                    fastperiod=params.get('fastperiod', 12),
                    slowperiod=params.get('slowperiod', 26),
                    signalperiod=params.get('signalperiod', 9)
                )
                return pd.Series(macd)
            elif indicator_name == "BBANDS":
                upper, middle, lower = talib.BBANDS(
                    close_prices,
                    timeperiod=params.get('timeperiod', 20),
                    nbdevup=params.get('nbdevup', 2),
                    nbdevdn=params.get('nbdevdn', 2)
                )
                return pd.Series(middle)
            elif indicator_name == "STOCH":
                slowk, slowd = talib.STOCH(
                    high_prices, low_prices, close_prices,
                    fastk_period=params.get('fastk_period', 14),
                    slowk_period=params.get('slowk_period', 3),
                    slowd_period=params.get('slowd_period', 3)
                )
                return pd.Series(slowk)
            elif indicator_name == "ATR":
                return pd.Series(talib.ATR(high_prices, low_prices, close_prices, timeperiod=params.get('timeperiod', 14)))
            elif indicator_name == "ADX":
                return pd.Series(talib.ADX(high_prices, low_prices, close_prices, timeperiod=params.get('timeperiod', 14)))
            elif indicator_name == "CCI":
                return pd.Series(talib.CCI(high_prices, low_prices, close_prices, timeperiod=params.get('timeperiod', 14)))
            elif indicator_name == "WILLR":
                return pd.Series(talib.WILLR(high_prices, low_prices, close_prices, timeperiod=params.get('timeperiod', 14)))
            elif indicator_name == "MFI":
                return pd.Series(talib.MFI(high_prices, low_prices, close_prices, volume, timeperiod=params.get('timeperiod', 14)))
            elif indicator_name == "OBV":
                return pd.Series(talib.OBV(close_prices, volume))
            else:
                logger.warning(f"⚠️ Unsupported TA-Lib indicator: {indicator_name}")
                return pd.Series(np.zeros(len(close_prices)))
                
        except ImportError:
            logger.warning("⚠️ TA-Lib not available, returning zeros")
            return pd.Series(np.zeros(len(data)))
        except Exception as e:
            logger.error(f"❌ Error calculating {indicator_name}: {e}")
            return pd.Series(np.zeros(len(data)))
    
    async def _calculate_smc_indicators(
        self,
        data: pd.DataFrame,
        smc_config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Calculate Smart Money Concepts indicators with enhanced analysis"""
        try:
            logger.info(f"🧠 Calculating Smart Money Concepts indicators")
            
            # Start with data copy
            result_data = data.copy()
            
            # Get requested SMC indicators
            smc_indicators = smc_config.get("indicators", [])
            if not smc_indicators:
                smc_indicators = ["BOS", "CHOCH", "ORDER_BLOCKS", "FVG", "LIQUIDITY", "MARKET_STRUCTURE", "PREMIUM_DISCOUNT"]
            
            # Calculate each SMC indicator
            for smc_indicator in smc_indicators:
                try:
                    if smc_indicator == "BOS":
                        result_data = await self._calculate_bos(result_data, smc_config)
                    elif smc_indicator == "CHOCH":
                        result_data = await self._calculate_choch(result_data, smc_config)
                    elif smc_indicator == "ORDER_BLOCKS":
                        result_data = await self._calculate_order_blocks(result_data, smc_config)
                    elif smc_indicator == "FVG":
                        result_data = await self._calculate_fvg(result_data, smc_config)
                    elif smc_indicator == "LIQUIDITY":
                        result_data = await self._calculate_liquidity(result_data, smc_config)
                    elif smc_indicator == "MARKET_STRUCTURE":
                        result_data = await self._calculate_market_structure_smc(result_data, smc_config)
                    elif smc_indicator == "PREMIUM_DISCOUNT":
                        result_data = await self._calculate_premium_discount(result_data, smc_config)
                        
                    logger.debug(f"✅ Calculated SMC indicator: {smc_indicator}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to calculate SMC indicator {smc_indicator}: {e}")
            
            # Calculate combined SMC signal
            result_data = await self._calculate_combined_smc_signal(result_data, smc_config)
            
            logger.info(f"✅ Completed SMC indicators calculation")
            return result_data
            
        except Exception as e:
            logger.error(f"❌ Error calculating SMC indicators: {e}")
            return data
    
    async def _calculate_bos(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Break of Structure (BOS) signals"""
        try:
            lookback = config.get("bos_lookback", 20)
            
            # Calculate recent highs and lows
            data['recent_high'] = data['high_price'].rolling(window=lookback, min_periods=1).max()
            data['recent_low'] = data['low_price'].rolling(window=lookback, min_periods=1).min()
            
            # Shift for comparison
            data['prev_recent_high'] = data['recent_high'].shift(1)
            data['prev_recent_low'] = data['recent_low'].shift(1)
            data['prev_close'] = data['close_price'].shift(1)
            
            # BOS detection
            bullish_bos = (data['close_price'] > data['prev_recent_high']) & (data['prev_close'] <= data['prev_recent_high'])
            bearish_bos = (data['close_price'] < data['prev_recent_low']) & (data['prev_close'] >= data['prev_recent_low'])
            
            data['bos_signal'] = 0
            data.loc[bullish_bos, 'bos_signal'] = 1
            data.loc[bearish_bos, 'bos_signal'] = -1
            
            # BOS strength (how significant the break is)
            data['bos_strength'] = 0.0
            data.loc[bullish_bos, 'bos_strength'] = (data['close_price'] - data['prev_recent_high']) / data['prev_recent_high']
            data.loc[bearish_bos, 'bos_strength'] = (data['prev_recent_low'] - data['close_price']) / data['prev_recent_low']
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error calculating BOS: {e}")
            return data
    
    async def _calculate_choch(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Change of Character (CHoCH) signals"""
        try:
            # Calculate trend based on higher highs/lows
            data['higher_high'] = (data['high_price'] > data['high_price'].shift(1)) & (data['low_price'] > data['low_price'].shift(1))
            data['lower_low'] = (data['high_price'] < data['high_price'].shift(1)) & (data['low_price'] < data['low_price'].shift(1))
            
            # Assign trend values
            data['current_trend'] = 0
            data.loc[data['higher_high'], 'current_trend'] = 1  # Uptrend
            data.loc[data['lower_low'], 'current_trend'] = -1  # Downtrend
            
            # Forward fill trends
            data['current_trend'] = data['current_trend'].replace(0, np.nan).fillna(method='ffill').fillna(0)
            data['prev_trend'] = data['current_trend'].shift(1)
            
            # CHoCH detection
            bullish_choch = (data['current_trend'] == 1) & (data['prev_trend'] == -1)
            bearish_choch = (data['current_trend'] == -1) & (data['prev_trend'] == 1)
            
            data['choch_signal'] = 0
            data.loc[bullish_choch, 'choch_signal'] = 1
            data.loc[bearish_choch, 'choch_signal'] = -1
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error calculating CHoCH: {e}")
            return data
    
    async def _calculate_order_blocks(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Order Block detection"""
        try:
            lookback = config.get("ob_lookback", 20)
            multiplier = config.get("ob_multiplier", 1.5)
            
            # Calculate candle properties
            data['candle_body'] = abs(data['close_price'] - data['open_price'])
            data['candle_type'] = np.where(data['close_price'] > data['open_price'], 'BULLISH', 
                                 np.where(data['close_price'] < data['open_price'], 'BEARISH', 'DOJI'))
            
            # Average body size
            data['avg_body_size'] = data['candle_body'].rolling(window=lookback, min_periods=1).mean()
            
            # Large candle detection
            data['large_candle'] = data['candle_body'] > (data['avg_body_size'] * multiplier)
            
            # Next candle properties
            data['next_candle_type'] = data['candle_type'].shift(-1)
            data['next_high'] = data['high_price'].shift(-1)
            data['next_low'] = data['low_price'].shift(-1)
            
            # Order block detection
            bullish_ob = (data['candle_type'] == 'BEARISH') & data['large_candle'] & \
                        (data['next_candle_type'] == 'BULLISH') & (data['next_high'] > data['high_price'])
            
            bearish_ob = (data['candle_type'] == 'BULLISH') & data['large_candle'] & \
                        (data['next_candle_type'] == 'BEARISH') & (data['next_low'] < data['low_price'])
            
            data['order_block_type'] = 'NONE'
            data.loc[bullish_ob, 'order_block_type'] = 'BULLISH_OB'
            data.loc[bearish_ob, 'order_block_type'] = 'BEARISH_OB'
            
            # Order block levels
            data['order_block_level'] = np.nan
            data.loc[bullish_ob, 'order_block_level'] = data['high_price']
            data.loc[bearish_ob, 'order_block_level'] = data['low_price']
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error calculating Order Blocks: {e}")
            return data
    
    async def _calculate_fvg(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Fair Value Gap (FVG) detection"""
        try:
            # Previous and next candle high/low
            data['prev_high'] = data['high_price'].shift(1)
            data['prev_low'] = data['low_price'].shift(1)
            data['next_high'] = data['high_price'].shift(-1)
            data['next_low'] = data['low_price'].shift(-1)
            
            # FVG detection
            bullish_fvg = (data['prev_high'] < data['next_low']) & \
                         (data['high_price'] > data['prev_high']) & \
                         (data['low_price'] < data['next_low'])
            
            bearish_fvg = (data['prev_low'] > data['next_high']) & \
                         (data['low_price'] < data['prev_low']) & \
                         (data['high_price'] > data['next_high'])
            
            data['fvg_signal'] = 0
            data.loc[bullish_fvg, 'fvg_signal'] = 1
            data.loc[bearish_fvg, 'fvg_signal'] = -1
            
            # FVG boundaries
            data['fvg_level_1'] = np.nan
            data['fvg_level_2'] = np.nan
            
            data.loc[bullish_fvg, 'fvg_level_1'] = data['prev_high']  # Bottom
            data.loc[bullish_fvg, 'fvg_level_2'] = data['next_low']   # Top
            
            data.loc[bearish_fvg, 'fvg_level_1'] = data['prev_low']   # Top
            data.loc[bearish_fvg, 'fvg_level_2'] = data['next_high']  # Bottom
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error calculating FVG: {e}")
            return data
    
    async def _calculate_liquidity(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Liquidity levels and sweeps"""
        try:
            lookback = config.get("liquidity_lookback", 50)
            tolerance = config.get("liquidity_tolerance", 0.001)
            min_touches = config.get("min_touches", 2)
            
            # Recent highs and lows (potential liquidity)
            data['recent_high'] = data['high_price'].rolling(window=lookback, min_periods=1).max()
            data['recent_low'] = data['low_price'].rolling(window=lookback, min_periods=1).min()
            
            # Count touches at these levels
            high_touches = []
            low_touches = []
            
            for i in range(len(data)):
                if i < lookback:
                    high_touches.append(0)
                    low_touches.append(0)
                    continue
                
                recent_data = data.iloc[max(0, i-lookback):i+1]
                recent_high = recent_data['high_price'].max()
                recent_low = recent_data['low_price'].min()
                
                # Count high touches
                high_touch_count = sum(1 for h in recent_data['high_price'] 
                                     if abs(h - recent_high) < (recent_high * tolerance))
                
                # Count low touches
                low_touch_count = sum(1 for l in recent_data['low_price'] 
                                    if abs(l - recent_low) < (recent_low * tolerance))
                
                high_touches.append(high_touch_count)
                low_touches.append(low_touch_count)
            
            data['high_touches'] = high_touches
            data['low_touches'] = low_touches
            
            # Liquidity grab detection
            buy_side_grab = (data['high_price'] > data['recent_high']) & \
                           (data['close_price'] < data['recent_high']) & \
                           (data['high_touches'] >= min_touches)
            
            sell_side_grab = (data['low_price'] < data['recent_low']) & \
                            (data['close_price'] > data['recent_low']) & \
                            (data['low_touches'] >= min_touches)
            
            data['liquidity_grab'] = 0
            data.loc[buy_side_grab, 'liquidity_grab'] = -1  # Bearish after liquidity grab
            data.loc[sell_side_grab, 'liquidity_grab'] = 1   # Bullish after liquidity grab
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error calculating Liquidity: {e}")
            return data
    
    async def _calculate_market_structure_smc(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Calculate SMC Market Structure analysis"""
        try:
            # Swing high and low detection
            data['swing_high'] = (data['high_price'] > data['high_price'].shift(1)) & \
                                (data['high_price'] > data['high_price'].shift(-1))
            data['swing_low'] = (data['low_price'] < data['low_price'].shift(1)) & \
                               (data['low_price'] < data['low_price'].shift(-1))
            
            # Get actual swing levels
            data['swing_high_level'] = np.where(data['swing_high'], data['high_price'], np.nan)
            data['swing_low_level'] = np.where(data['swing_low'], data['low_price'], np.nan)
            
            # Forward fill swing levels
            data['prev_swing_high'] = data['swing_high_level'].fillna(method='ffill')
            data['prev_swing_low'] = data['swing_low_level'].fillna(method='ffill')
            
            # Shift for comparison
            data['prev_prev_swing_high'] = data['prev_swing_high'].shift(1)
            data['prev_prev_swing_low'] = data['prev_swing_low'].shift(1)
            
            # Market structure determination
            bullish_structure = (data['prev_swing_high'] > data['prev_prev_swing_high']) & \
                               (data['prev_swing_low'] > data['prev_prev_swing_low'])
            
            bearish_structure = (data['prev_swing_high'] < data['prev_prev_swing_high']) & \
                               (data['prev_swing_low'] < data['prev_prev_swing_low'])
            
            data['market_structure'] = 'RANGING'
            data.loc[bullish_structure, 'market_structure'] = 'BULLISH'
            data.loc[bearish_structure, 'market_structure'] = 'BEARISH'
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error calculating SMC Market Structure: {e}")
            return data
    
    async def _calculate_premium_discount(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Premium/Discount zones"""
        try:
            range_period = config.get("range_period", 50)
            premium_threshold = config.get("premium_threshold", 0.7)
            discount_threshold = config.get("discount_threshold", 0.7)
            
            # Calculate range
            data['range_high'] = data['high_price'].rolling(window=range_period, min_periods=1).max()
            data['range_low'] = data['low_price'].rolling(window=range_period, min_periods=1).min()
            data['range_midpoint'] = (data['range_high'] + data['range_low']) / 2
            
            # Premium/Discount zones
            premium_level = data['range_midpoint'] + (data['range_high'] - data['range_midpoint']) * premium_threshold
            discount_level = data['range_midpoint'] - (data['range_midpoint'] - data['range_low']) * discount_threshold
            
            data['price_zone'] = 'EQUILIBRIUM'
            data.loc[data['close_price'] > premium_level, 'price_zone'] = 'PREMIUM'
            data.loc[data['close_price'] < discount_level, 'price_zone'] = 'DISCOUNT'
            
            # Zone percentage
            range_size = data['range_high'] - data['range_low']
            data['zone_percentage'] = np.where(range_size > 0, 
                                             (data['close_price'] - data['range_low']) / range_size * 100, 
                                             50)
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error calculating Premium/Discount: {e}")
            return data
    
    async def _calculate_combined_smc_signal(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Calculate combined SMC signal from all indicators"""
        try:
            # Default weights for SMC signals
            weights = config.get("signal_weights", {
                "bos_signal": 0.25,
                "choch_signal": 0.20,
                "order_block_type": 0.20,
                "fvg_signal": 0.15,
                "liquidity_grab": 0.20
            })
            
            # Initialize combined signal
            data['smc_combined_signal'] = 0.0
            
            for idx, row in data.iterrows():
                signals = {}
                
                # Collect available signals
                if 'bos_signal' in data.columns:
                    signals['bos_signal'] = row.get('bos_signal', 0)
                if 'choch_signal' in data.columns:
                    signals['choch_signal'] = row.get('choch_signal', 0)
                if 'order_block_type' in data.columns:
                    ob_type = row.get('order_block_type', 'NONE')
                    if ob_type == 'BULLISH_OB':
                        signals['order_block_type'] = 1
                    elif ob_type == 'BEARISH_OB':
                        signals['order_block_type'] = -1
                    else:
                        signals['order_block_type'] = 0
                if 'fvg_signal' in data.columns:
                    signals['fvg_signal'] = row.get('fvg_signal', 0)
                if 'liquidity_grab' in data.columns:
                    signals['liquidity_grab'] = row.get('liquidity_grab', 0)
                
                # Calculate weighted signal
                total_score = 0
                total_weight = 0
                
                for signal_name, signal_value in signals.items():
                    if signal_name in weights and signal_value is not None:
                        total_score += signal_value * weights[signal_name]
                        total_weight += weights[signal_name]
                
                if total_weight > 0:
                    data.loc[idx, 'smc_combined_signal'] = total_score / total_weight
            
            # Add signal strength
            data['smc_signal_strength'] = abs(data['smc_combined_signal'])
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error calculating combined SMC signal: {e}")
            return data
    
    async def _generate_trading_signals(
        self,
        indicator_data: pd.DataFrame,
        signal_config: Dict[str, Any]
    ) -> List[IndicatorSignal]:
        """Generate trading signals based on indicator data"""
        try:
            logger.info(f"🎯 Generating signals with {len(indicator_data)} data points")
            
            signals = []
            
            # Iterate through data points
            for idx, row in indicator_data.iterrows():
                # Long signal conditions
                long_conditions = []
                short_conditions = []
                signal_sources = []
                
                # RSI oversold/overbought
                if 'RSI_14' in row:
                    rsi_value = row['RSI_14']
                    if not pd.isna(rsi_value):
                        long_conditions.append(rsi_value < 30)
                        short_conditions.append(rsi_value > 70)
                        signal_sources.append('RSI')
                
                # MACD crossover
                if 'MACD_' in str(row.index):
                    macd_cols = [col for col in row.index if 'MACD' in col]
                    if macd_cols:
                        macd_value = row[macd_cols[0]]
                        if not pd.isna(macd_value):
                            long_conditions.append(macd_value > 0)
                            short_conditions.append(macd_value < 0)
                            signal_sources.append('MACD')
                
                # Bollinger Bands
                if 'BBANDS_LOWER_20' in row and 'BBANDS_UPPER_20' in row:
                    price = row.get('close_price', row.get('close', 0))
                    bb_lower = row['BBANDS_LOWER_20']
                    bb_upper = row['BBANDS_UPPER_20']
                    
                    if not pd.isna(bb_lower) and not pd.isna(bb_upper):
                        long_conditions.append(price <= bb_lower)
                        short_conditions.append(price >= bb_upper)
                        signal_sources.append('BBANDS')
                
                # Smart Money Concepts signals
                if 'smc_combined_signal' in row:
                    smc_signal = row['smc_combined_signal']
                    smc_strength = row.get('smc_signal_strength', 0)
                    
                    if not pd.isna(smc_signal) and smc_strength > 0.3:  # Minimum strength threshold
                        long_conditions.append(smc_signal > 0.5)
                        short_conditions.append(smc_signal < -0.5)
                        signal_sources.append('SMC')
                
                # Individual SMC signals for enhanced analysis
                if 'bos_signal' in row:
                    bos = row['bos_signal']
                    if not pd.isna(bos) and bos != 0:
                        long_conditions.append(bos > 0)
                        short_conditions.append(bos < 0)
                        signal_sources.append('BOS')
                
                if 'choch_signal' in row:
                    choch = row['choch_signal']
                    if not pd.isna(choch) and choch != 0:
                        long_conditions.append(choch > 0)
                        short_conditions.append(choch < 0)
                        signal_sources.append('CHoCH')
                
                if 'order_block_type' in row:
                    ob_type = row['order_block_type']
                    if ob_type == 'BULLISH_OB':
                        long_conditions.append(True)
                        signal_sources.append('ORDER_BLOCK')
                    elif ob_type == 'BEARISH_OB':
                        short_conditions.append(True)
                        signal_sources.append('ORDER_BLOCK')
                
                if 'fvg_signal' in row:
                    fvg = row['fvg_signal']
                    if not pd.isna(fvg) and fvg != 0:
                        long_conditions.append(fvg > 0)
                        short_conditions.append(fvg < 0)
                        signal_sources.append('FVG')
                
                if 'liquidity_grab' in row:
                    liquidity = row['liquidity_grab']
                    if not pd.isna(liquidity) and liquidity != 0:
                        long_conditions.append(liquidity > 0)
                        short_conditions.append(liquidity < 0)
                        signal_sources.append('LIQUIDITY')
                
                # Premium/Discount zone consideration
                if 'price_zone' in row:
                    zone = row['price_zone']
                    if zone == 'DISCOUNT':
                        long_conditions.append(True)  # Favor longs in discount zones
                        signal_sources.append('DISCOUNT_ZONE')
                    elif zone == 'PREMIUM':
                        short_conditions.append(True)  # Favor shorts in premium zones
                        signal_sources.append('PREMIUM_ZONE')
                
                # Volume Profile signals
                if 'vp_poc' in row and not pd.isna(row['vp_poc']):
                    current_price = row.get('close_price', 0)
                    poc = row['vp_poc']
                    vah = row.get('vp_vah')
                    val = row.get('vp_val')
                    
                    # POC support/resistance
                    poc_tolerance = abs(current_price * 0.005)  # 0.5% tolerance
                    if abs(current_price - poc) < poc_tolerance:
                        # Price at POC can bounce either direction
                        long_conditions.append(True)
                        short_conditions.append(True)
                        signal_sources.append('POC_INTERACTION')
                    
                    # Value Area signals
                    if not pd.isna(vah) and not pd.isna(val):
                        if current_price < val:
                            long_conditions.append(True)  # Below value area = oversold
                            signal_sources.append('BELOW_VALUE_AREA')
                        elif current_price > vah:
                            short_conditions.append(True)  # Above value area = overbought
                            signal_sources.append('ABOVE_VALUE_AREA')
                    
                    # Volume imbalance
                    volume_imbalance = row.get('vp_volume_imbalance', 0)
                    if abs(volume_imbalance) > 0.3:
                        if volume_imbalance > 0:
                            long_conditions.append(True)  # Bullish volume imbalance
                            signal_sources.append('VOLUME_IMBALANCE_BULL')
                        else:
                            short_conditions.append(True)  # Bearish volume imbalance
                            signal_sources.append('VOLUME_IMBALANCE_BEAR')
                
                # Order Flow signals
                if 'of_delta' in row:
                    # Delta divergence signals
                    if row.get('of_bullish_divergence', False):
                        long_conditions.append(True)
                        signal_sources.append('DELTA_DIVERGENCE_BULL')
                    
                    if row.get('of_bearish_divergence', False):
                        short_conditions.append(True)
                        signal_sources.append('DELTA_DIVERGENCE_BEAR')
                    
                    # Absorption signals (contrarian)
                    if row.get('of_absorption', False):
                        # High volume with small price move suggests accumulation/distribution
                        long_conditions.append(True)
                        short_conditions.append(True)
                        signal_sources.append('ABSORPTION')
                    
                    # Institutional activity
                    if row.get('of_institutional_activity', False):
                        pressure = row.get('of_institutional_pressure', 'NONE')
                        if pressure == 'BUYING':
                            long_conditions.append(True)
                            signal_sources.append('INSTITUTIONAL_BUYING')
                        elif pressure == 'SELLING':
                            short_conditions.append(True)
                            signal_sources.append('INSTITUTIONAL_SELLING')
                
                # Volume confirmation
                if 'volume' in row:
                    volume = row['volume']
                    # Simple volume filter (could be enhanced)
                    volume_confirmed = volume > indicator_data['volume'].mean() if not pd.isna(volume) else True
                    if volume_confirmed:
                        long_conditions.append(True)
                        short_conditions.append(True)
                        signal_sources.append('VOLUME')
                
                # Generate signals if conditions met
                min_conditions = signal_config.get('min_conditions', 2)
                confidence_threshold = signal_config.get('confidence_threshold', 0.6)
                
                if len([c for c in long_conditions if c]) >= min_conditions:
                    confidence = len([c for c in long_conditions if c]) / len(long_conditions) if long_conditions else 0
                    if confidence >= confidence_threshold:
                        # Enhanced indicator data for signal
                        indicator_data_dict = {
                            'rsi': row.get('RSI_14'),
                            'macd': row.get('MACD_'),
                            'bb_position': 'lower' if 'BBANDS_LOWER_20' in row else None,
                            'volume_confirmed': volume_confirmed if 'volume' in row else None,
                            # SMC indicators
                            'smc_combined_signal': row.get('smc_combined_signal'),
                            'smc_signal_strength': row.get('smc_signal_strength'),
                            'bos_signal': row.get('bos_signal'),
                            'choch_signal': row.get('choch_signal'),
                            'order_block_type': row.get('order_block_type'),
                            'fvg_signal': row.get('fvg_signal'),
                            'liquidity_grab': row.get('liquidity_grab'),
                            'price_zone': row.get('price_zone'),
                            'zone_percentage': row.get('zone_percentage'),
                            'market_structure': row.get('market_structure'),
                            # Volume Profile indicators
                            'vp_poc': row.get('vp_poc'),
                            'vp_vah': row.get('vp_vah'),
                            'vp_val': row.get('vp_val'),
                            'vp_volume_at_price': row.get('vp_volume_at_price'),
                            'vp_profile_strength': row.get('vp_profile_strength'),
                            'vp_volume_imbalance': row.get('vp_volume_imbalance'),
                            # Order Flow indicators
                            'of_delta': row.get('of_delta'),
                            'of_delta_pct': row.get('of_delta_pct'),
                            'of_cumulative_delta': row.get('of_cumulative_delta'),
                            'of_bullish_divergence': row.get('of_bullish_divergence'),
                            'of_bearish_divergence': row.get('of_bearish_divergence'),
                            'of_divergence_strength': row.get('of_divergence_strength'),
                            'of_absorption': row.get('of_absorption'),
                            'of_absorption_strength': row.get('of_absorption_strength'),
                            'of_institutional_activity': row.get('of_institutional_activity'),
                            'of_institutional_pressure': row.get('of_institutional_pressure'),
                            'of_flow_direction': row.get('of_flow_direction'),
                            'of_flow_strength': row.get('of_flow_strength'),
                            'signal_sources': signal_sources,
                            'conditions_met': len([c for c in long_conditions if c]),
                            'total_conditions': len(long_conditions)
                        }
                        
                        # Adjust predicted move based on SMC strength
                        predicted_move = 0.02  # Base 2% expected move
                        if 'smc_signal_strength' in row:
                            smc_strength = row.get('smc_signal_strength', 0)
                            predicted_move = max(0.015, min(0.05, 0.02 + (smc_strength * 0.03)))
                        
                        signal = IndicatorSignal(
                            timestamp=idx if isinstance(idx, datetime) else datetime.now(timezone.utc),
                            symbol=signal_config.get('symbol', 'NIFTY'),
                            direction='LONG',
                            confidence=confidence,
                            predicted_move=predicted_move,
                            indicators=indicator_data_dict,
                            signal_source='enhanced_technical_indicators'
                        )
                        signals.append(signal)
                
                elif len([c for c in short_conditions if c]) >= min_conditions:
                    confidence = len([c for c in short_conditions if c]) / len(short_conditions) if short_conditions else 0
                    if confidence >= confidence_threshold:
                        # Enhanced indicator data for short signal (matching long signal structure)
                        indicator_data_dict = {
                            'rsi': row.get('RSI_14'),
                            'macd': row.get('MACD_'),
                            'bb_position': 'upper' if 'BBANDS_UPPER_20' in row else None,
                            'volume_confirmed': volume_confirmed if 'volume' in row else None,
                            # SMC indicators
                            'smc_combined_signal': row.get('smc_combined_signal'),
                            'smc_signal_strength': row.get('smc_signal_strength'),
                            'bos_signal': row.get('bos_signal'),
                            'choch_signal': row.get('choch_signal'),
                            'order_block_type': row.get('order_block_type'),
                            'fvg_signal': row.get('fvg_signal'),
                            'liquidity_grab': row.get('liquidity_grab'),
                            'price_zone': row.get('price_zone'),
                            'zone_percentage': row.get('zone_percentage'),
                            'market_structure': row.get('market_structure'),
                            # Volume Profile indicators
                            'vp_poc': row.get('vp_poc'),
                            'vp_vah': row.get('vp_vah'),
                            'vp_val': row.get('vp_val'),
                            'vp_volume_at_price': row.get('vp_volume_at_price'),
                            'vp_profile_strength': row.get('vp_profile_strength'),
                            'vp_volume_imbalance': row.get('vp_volume_imbalance'),
                            # Order Flow indicators
                            'of_delta': row.get('of_delta'),
                            'of_delta_pct': row.get('of_delta_pct'),
                            'of_cumulative_delta': row.get('of_cumulative_delta'),
                            'of_bullish_divergence': row.get('of_bullish_divergence'),
                            'of_bearish_divergence': row.get('of_bearish_divergence'),
                            'of_divergence_strength': row.get('of_divergence_strength'),
                            'of_absorption': row.get('of_absorption'),
                            'of_absorption_strength': row.get('of_absorption_strength'),
                            'of_institutional_activity': row.get('of_institutional_activity'),
                            'of_institutional_pressure': row.get('of_institutional_pressure'),
                            'of_flow_direction': row.get('of_flow_direction'),
                            'of_flow_strength': row.get('of_flow_strength'),
                            'signal_sources': signal_sources,
                            'conditions_met': len([c for c in short_conditions if c]),
                            'total_conditions': len(short_conditions)
                        }
                        
                        # Adjust predicted move based on SMC strength (matching long signal logic)
                        predicted_move = 0.02  # Base 2% expected move
                        if 'smc_signal_strength' in row:
                            smc_strength = row.get('smc_signal_strength', 0)
                            predicted_move = max(0.015, min(0.05, 0.02 + (smc_strength * 0.03)))
                        
                        signal = IndicatorSignal(
                            timestamp=idx if isinstance(idx, datetime) else datetime.now(timezone.utc),
                            symbol=signal_config.get('symbol', 'NIFTY'),
                            direction='SHORT',
                            confidence=confidence,
                            predicted_move=predicted_move,
                            indicators=indicator_data_dict,
                            signal_source='enhanced_technical_indicators'
                        )
                        signals.append(signal)
            
            # Store signal history
            self.signal_history.extend(signals)
            
            logger.info(f"✅ Generated {len(signals)} trading signals")
            return signals
            
        except Exception as e:
            logger.error(f"❌ Error generating signals: {e}")
            return []
    
    async def _validate_indicator_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate indicator parameters"""
        try:
            logger.info(f"✅ Validating {len(parameters)} parameter sets")
            
            validation_results = {}
            
            for param_group, param_data in parameters.items():
                try:
                    if param_group == "talib_indicators":
                        validation_results[param_group] = self._validate_talib_parameters(param_data)
                    elif param_group == "custom_indicators":
                        validation_results[param_group] = self._validate_custom_parameters(param_data)
                    elif param_group == "smc_indicators":
                        validation_results[param_group] = self._validate_smc_parameters(param_data)
                    elif param_group == "signal_config":
                        validation_results[param_group] = self._validate_signal_parameters(param_data)
                    else:
                        validation_results[param_group] = {
                            "is_valid": False,
                            "errors": [f"Unknown parameter group: {param_group}"]
                        }
                        
                except Exception as e:
                    validation_results[param_group] = {
                        "is_valid": False,
                        "errors": [f"Validation error: {str(e)}"]
                    }
            
            return validation_results
            
        except Exception as e:
            logger.error(f"❌ Error validating parameters: {e}")
            return {"validation_error": {"is_valid": False, "errors": [str(e)]}}
    
    def _validate_talib_parameters(self, talib_params: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate TA-Lib indicator parameters"""
        errors = []
        warnings = []
        
        for param_set in talib_params:
            indicator_name = param_set.get("name")
            if not indicator_name:
                errors.append("Missing indicator name")
                continue
            
            if indicator_name not in self.talib_wrapper.get_supported_indicators():
                errors.append(f"Unsupported indicator: {indicator_name}")
                continue
            
            # Validate indicator-specific parameters
            is_valid, error_msg = self.talib_wrapper.validate_indicator_config(param_set)
            if not is_valid:
                errors.append(f"{indicator_name}: {error_msg}")
            
            # Check for reasonable parameter ranges
            params = param_set.get("params", {})
            if "timeperiod" in params:
                timeperiod = params["timeperiod"]
                if timeperiod < 1 or timeperiod > 200:
                    warnings.append(f"{indicator_name}: timeperiod {timeperiod} may be unreasonable")
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "validated_count": len(talib_params)
        }
    
    def _validate_custom_parameters(self, custom_params: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate custom indicator parameters"""
        errors = []
        warnings = []
        
        for param_set in custom_params:
            indicator_name = param_set.get("name")
            if not indicator_name:
                errors.append("Missing custom indicator name")
                continue
            
            # Basic validation for custom indicators
            params = param_set.get("params", {})
            if not isinstance(params, dict):
                errors.append(f"{indicator_name}: params must be a dictionary")
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "validated_count": len(custom_params)
        }
    
    def _validate_smc_parameters(self, smc_params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Smart Money Concepts parameters"""
        errors = []
        warnings = []
        
        # Validate SMC-specific parameters
        if "swing_length" in smc_params:
            swing_length = smc_params["swing_length"]
            if not isinstance(swing_length, int) or swing_length < 3:
                errors.append("swing_length must be integer >= 3")
        
        if "liquidity_range" in smc_params:
            liquidity_range = smc_params["liquidity_range"]
            if not isinstance(liquidity_range, (int, float)) or liquidity_range <= 0:
                errors.append("liquidity_range must be positive number")
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def _validate_signal_parameters(self, signal_params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate signal generation parameters"""
        errors = []
        warnings = []
        
        if "confidence_threshold" in signal_params:
            threshold = signal_params["confidence_threshold"]
            if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
                errors.append("confidence_threshold must be between 0 and 1")
        
        if "min_conditions" in signal_params:
            min_conditions = signal_params["min_conditions"]
            if not isinstance(min_conditions, int) or min_conditions < 1:
                errors.append("min_conditions must be positive integer")
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    async def _collect_agent_specific_metrics(self) -> Dict[str, float]:
        """Collect IND agent specific metrics"""
        try:
            metrics = {
                "active_positions": len(self.active_positions),
                "signals_generated_today": len([s for s in self.signal_history 
                                               if s.timestamp.date() == datetime.now().date()]),
                "avg_signal_confidence": np.mean([s.confidence for s in self.signal_history[-100:]]) if self.signal_history else 0,
                "indicator_cache_size": len(self.indicator_cache),
                "talib_indicators_available": len(self.talib_wrapper.get_supported_indicators()),
                "last_calculation_time": (datetime.now(timezone.utc) - self.last_data_update).total_seconds() 
                                       if self.last_data_update else 0
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error collecting agent metrics: {e}")
            return {}
    
    async def _check_agent_specific_health(self) -> bool:
        """Check IND agent specific health"""
        try:
            # Check if TA-Lib wrapper is functional
            supported_indicators = self.talib_wrapper.get_supported_indicators()
            if len(supported_indicators) == 0:
                logger.error("❌ No TA-Lib indicators available")
                return False
            
            # Check cache size (prevent memory leaks)
            if len(self.indicator_cache) > 100:
                logger.warning("⚠️ Indicator cache size exceeding limits")
                # Clean old cache entries
                cache_keys = list(self.indicator_cache.keys())
                for key in cache_keys[:-50]:  # Keep only latest 50
                    del self.indicator_cache[key]
            
            return True
            
        except Exception as e:
            logger.error(f"❌ IND agent health check failed: {e}")
            return False

# Factory functions
def create_ind_agent_config(
    agent_id: str = "bmad-ind-strategy-agent",
    port: int = 8008,
    talib_indicators: Optional[List[str]] = None,
    custom_indicators: Optional[List[str]] = None,
    smc_indicators: Optional[List[str]] = None
) -> Tuple[AgentConfig, IndicatorAgentConfig]:
    """Create IND agent configuration"""
    
    # Default TA-Lib indicators
    if talib_indicators is None:
        talib_indicators = [
            "SMA", "EMA", "RSI", "MACD", "BBANDS", "STOCH", "ATR", "ADX", 
            "CCI", "WILLR", "MFI", "OBV", "TRIX", "ROC", "STOCHRSI"
        ]
    
    # Default custom indicators
    if custom_indicators is None:
        custom_indicators = [
            "VWAP", "TWAP", "PIVOT_POINTS", "FIBONACCI_RETRACEMENTS"
        ]
    
    # Default SMC indicators
    if smc_indicators is None:
        smc_indicators = [
            "MARKET_STRUCTURE", "LIQUIDITY_SWEEPS", "ORDER_BLOCKS", "FAIR_VALUE_GAPS"
        ]
    
    # Base agent config
    agent_config = create_agent_config(
        agent_id=agent_id,
        agent_type="strategy",
        port=port,
        version="1.0.0"
    )
    
    # Strategy-specific config
    strategy_config = IndicatorAgentConfig(
        talib_indicators=talib_indicators,
        custom_indicators=custom_indicators,
        smc_indicators=smc_indicators,
        volume_analysis=True,
        candlestick_patterns=True,
        market_structure_analysis=True,
        signal_threshold=0.7,
        max_positions=5,
        risk_percent=2.0,
        lookback_period=50
    )
    
    return agent_config, strategy_config

def create_ind_agent(
    agent_config: Optional[AgentConfig] = None,
    strategy_config: Optional[IndicatorAgentConfig] = None
) -> BMadIndStrategyAgent:
    """Create IND strategy agent"""
    
    if agent_config is None or strategy_config is None:
        agent_config, strategy_config = create_ind_agent_config()
    
    return BMadIndStrategyAgent(agent_config, strategy_config)

if __name__ == "__main__":
    # Example usage and testing
    async def test_ind_agent():
        """Test IND strategy agent"""
        
        logger.info("🧪 Testing BMAD IND Strategy Agent")
        
        try:
            # Create agent
            agent = create_ind_agent()
            
            # Start agent
            await agent.start_server()
            
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
        finally:
            if 'agent' in locals():
                await agent.shutdown()
    
    # Run test
    asyncio.run(test_ind_agent())