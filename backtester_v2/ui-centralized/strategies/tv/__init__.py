"""
TV (TradingView) Strategy Module
Handles external signal processing from TradingView with real-time WebSocket integration
"""

from .strategy import TVStrategy
from .parser import TVParser
from .query_builder import TVQueryBuilder
from .processor import TVProcessor
from .signal_processor import SignalProcessor as TVSignalProcessor
from .websocket_integration import TVWebSocketIntegration, WebSocketConfig, ConnectionState, get_tv_websocket_integration
from .models import (
    TVConfigModel,
    TVSignal,
    TVPosition,
    TVExecutionResult,
    TVMarketData,
    TVRiskMetrics,
    WebSocketEvent,
    SignalSource,
    SignalType,
    SignalStatus,
    PositionType,
    OptionType,
    OrderType,
    WebSocketEventType
)

__version__ = "1.0.0"

__all__ = [
    'TVStrategy',
    'TVParser',
    'TVSignalProcessor',
    'TVQueryBuilder',
    'TVProcessor',
    'TVConfigModel',
    'TVSignal',
    'TVPosition',
    'TVExecutionResult',
    'TVMarketData',
    'TVRiskMetrics',
    'WebSocketEvent',
    'SignalSource',
    'SignalType',
    'SignalStatus',
    'PositionType',
    'OptionType',
    'OrderType',
    'WebSocketEventType',
    'TVWebSocketIntegration',
    'WebSocketConfig',
    'ConnectionState',
    'get_tv_websocket_integration'
]