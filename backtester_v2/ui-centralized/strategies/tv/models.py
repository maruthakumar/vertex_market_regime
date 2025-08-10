"""
TV (TradingView) Strategy Data Models
Supports external signal processing from TradingView with real-time integration
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from datetime import date, time, datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
import pandas as pd


class SignalSource(str, Enum):
    """Sources of trading signals"""
    TRADINGVIEW = "TRADINGVIEW"
    WEBHOOK = "WEBHOOK"
    API = "API"
    MANUAL = "MANUAL"
    AUTOMATED = "AUTOMATED"


class SignalType(str, Enum):
    """Types of trading signals"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"
    ENTRY = "ENTRY"
    EXIT = "EXIT"


class SignalStatus(str, Enum):
    """Status of trading signals"""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    EXECUTED = "EXECUTED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class PositionType(str, Enum):
    """Position types for options"""
    BUY = "BUY"
    SELL = "SELL"


class OptionType(str, Enum):
    """Option types"""
    CALL = "CE"
    PUT = "PE"


class OrderType(str, Enum):
    """Order types for execution"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LIMIT = "STOP_LIMIT"


class WebSocketEventType(str, Enum):
    """WebSocket event types for real-time updates"""
    SIGNAL_RECEIVED = "SIGNAL_RECEIVED"
    SIGNAL_PROCESSED = "SIGNAL_PROCESSED"
    ORDER_PLACED = "ORDER_PLACED"
    ORDER_FILLED = "ORDER_FILLED"
    POSITION_UPDATED = "POSITION_UPDATED"
    ERROR_OCCURRED = "ERROR_OCCURRED"


@dataclass
class TVSignal:
    """TradingView signal data structure"""
    signal_id: str
    timestamp: datetime
    source: SignalSource
    signal_type: SignalType
    symbol: str
    action: str  # BUY, SELL, HOLD
    quantity: int
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float = 1.0
    status: SignalStatus = SignalStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    webhook_data: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate signal parameters"""
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")


@dataclass
class TVPosition:
    """Individual position in TV strategy"""
    symbol: str
    option_type: OptionType
    strike_price: float
    expiry_date: date
    position_type: PositionType
    quantity: int
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    entry_signal_id: Optional[str] = None
    exit_signal_id: Optional[str] = None
    order_type: OrderType = OrderType.MARKET
    current_pnl: float = 0.0
    
    def __post_init__(self):
        """Validate position parameters"""
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")
        if self.strike_price <= 0:
            raise ValueError("Strike price must be positive")


@dataclass
class WebSocketEvent:
    """WebSocket event for real-time updates"""
    event_id: str
    event_type: WebSocketEventType
    timestamp: datetime
    data: Dict[str, Any]
    source: str = "TV_STRATEGY"
    processed: bool = False
    
    def mark_processed(self):
        """Mark event as processed"""
        self.processed = True


class TVConfigModel(BaseModel):
    """Pydantic model for TV strategy configuration"""
    
    # Signal processing parameters
    signal_sources: List[SignalSource] = Field(default_factory=lambda: [SignalSource.TRADINGVIEW], description="Signal sources")
    webhook_url: Optional[str] = Field(None, description="Webhook URL for signal reception")
    api_key: Optional[str] = Field(None, description="API key for authentication")
    signal_timeout_minutes: int = Field(30, ge=1, le=1440, description="Signal timeout in minutes")
    
    # Position parameters
    max_positions: int = Field(5, ge=1, le=20, description="Maximum number of positions")
    position_size: int = Field(50, ge=1, description="Default position size")
    max_trades_per_day: int = Field(20, ge=1, description="Maximum trades per day")
    
    # Risk management
    max_loss_per_trade: float = Field(1000.0, ge=0, description="Maximum loss per trade")
    max_daily_loss: float = Field(5000.0, ge=0, description="Maximum daily loss")
    profit_target: Optional[float] = Field(None, ge=0, description="Profit target per trade")
    
    # WebSocket configuration
    enable_websocket: bool = Field(True, description="Enable WebSocket for real-time updates")
    websocket_url: Optional[str] = Field(None, description="WebSocket server URL")
    websocket_timeout: int = Field(30, ge=1, description="WebSocket timeout in seconds")
    
    # Signal validation
    require_confirmation: bool = Field(False, description="Require manual confirmation for signals")
    min_confidence: float = Field(0.5, ge=0, le=1, description="Minimum signal confidence")
    
    # Strategy-specific parameters
    symbols: List[str] = Field(default_factory=list, description="Trading symbols")
    signal_filters: Dict[str, Any] = Field(default_factory=dict, description="Signal filtering criteria")
    
    @validator('symbols')
    def validate_symbols(cls, v):
        """Validate trading symbols"""
        if not v:
            raise ValueError('At least one symbol must be specified')
        return v
    
    @validator('min_confidence')
    def validate_confidence(cls, v):
        """Validate confidence range"""
        if not 0 <= v <= 1:
            raise ValueError('Confidence must be between 0 and 1')
        return v


@dataclass
class TVExecutionResult:
    """Result of TV strategy execution"""
    strategy_name: str = "TV"
    execution_timestamp: datetime = field(default_factory=datetime.now)
    signals_received: List[TVSignal] = field(default_factory=list)
    signals_processed: List[TVSignal] = field(default_factory=list)
    positions_opened: List[TVPosition] = field(default_factory=list)
    positions_closed: List[TVPosition] = field(default_factory=list)
    websocket_events: List[WebSocketEvent] = field(default_factory=list)
    total_pnl: float = 0.0
    execution_time_ms: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    
    def add_signal(self, signal: TVSignal, processed: bool = False):
        """Add a signal to the appropriate list"""
        self.signals_received.append(signal)
        if processed:
            self.signals_processed.append(signal)
    
    def add_position(self, position: TVPosition, action: str):
        """Add a position to the appropriate list"""
        if action.upper() == "OPEN":
            self.positions_opened.append(position)
        elif action.upper() == "CLOSE":
            self.positions_closed.append(position)
    
    def add_websocket_event(self, event: WebSocketEvent):
        """Add a WebSocket event"""
        self.websocket_events.append(event)
    
    def calculate_pnl(self):
        """Calculate total P&L from closed positions"""
        total_pnl = 0.0
        for position in self.positions_closed:
            if position.entry_price and position.exit_price:
                if position.position_type == PositionType.BUY:
                    pnl = (position.exit_price - position.entry_price) * position.quantity
                else:  # SELL
                    pnl = (position.entry_price - position.exit_price) * position.quantity
                total_pnl += pnl
        
        self.total_pnl = total_pnl
        return total_pnl


@dataclass
class TVMarketData:
    """Market data structure for TV strategy"""
    symbol: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    option_chain: Optional[Dict[str, Any]] = None
    real_time_data: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate market data"""
        prices = [self.open_price, self.high_price, self.low_price, self.close_price]
        if any(price <= 0 for price in prices):
            raise ValueError("All prices must be positive")
        if self.volume < 0:
            raise ValueError("Volume cannot be negative")


@dataclass
class TVRiskMetrics:
    """Risk metrics for TV strategy"""
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    var_95: float = 0.0  # Value at Risk 95%
    expected_shortfall: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    signal_accuracy: float = 0.0
    avg_signal_latency_ms: float = 0.0
    websocket_uptime: float = 0.0
    
    def update_signal_metrics(self, signals: List[TVSignal]):
        """Update signal-related metrics"""
        if not signals:
            return
        
        # Calculate signal accuracy
        executed_signals = [s for s in signals if s.status == SignalStatus.EXECUTED]
        if executed_signals:
            self.signal_accuracy = len(executed_signals) / len(signals)
        
        # Calculate average confidence
        avg_confidence = sum(s.confidence for s in signals) / len(signals)
        
        # Update WebSocket uptime (simplified calculation)
        self.websocket_uptime = 0.95  # Placeholder - would be calculated from actual uptime data


# Type aliases for convenience
TVConfig = Union[TVConfigModel, Dict[str, Any]]
TVData = Union[TVMarketData, pd.DataFrame]
TVResult = TVExecutionResult
