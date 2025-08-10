"""
TradingView Strategy WebSocket Integration
Real-time signal processing and event handling for TradingView strategy

This module provides comprehensive WebSocket integration for:
- Real-time signal reception from TradingView
- Webhook processing and validation
- Event-driven signal processing
- Real-time market data streaming
- Connection management and failover

Author: The Augster
Date: 2025-01-19
Framework: SuperClaude v3 Enhanced WebSocket Integration
"""

import asyncio
import json
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import websockets
from websockets.exceptions import ConnectionClosed, InvalidURI
import aiohttp
from aiohttp import web
import ssl

from .models import TVSignal, TVConfigModel, WebSocketEvent, WebSocketEventType, SignalSource, SignalStatus
from .signal_processor import SignalProcessor

logger = logging.getLogger(__name__)


class ConnectionState(str, Enum):
    """WebSocket connection states"""
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    FAILED = "FAILED"


@dataclass
class WebSocketConfig:
    """WebSocket configuration"""
    server_host: str = "localhost"
    server_port: int = 8765
    webhook_port: int = 8766
    max_connections: int = 100
    heartbeat_interval: int = 30
    reconnect_delay: int = 5
    max_reconnect_attempts: int = 10
    ssl_enabled: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None


class TVWebSocketIntegration:
    """
    Comprehensive WebSocket integration for TradingView Strategy
    
    Provides real-time signal processing, event handling, and connection management
    """
    
    def __init__(self, config: TVConfigModel, ws_config: Optional[WebSocketConfig] = None):
        """Initialize WebSocket integration"""
        self.config = config
        self.ws_config = ws_config or WebSocketConfig()
        self.signal_processor = SignalProcessor()
        
        # Connection management
        self.connection_state = ConnectionState.DISCONNECTED
        self.websocket = None
        self.webhook_app = None
        self.webhook_runner = None
        
        # Event handling
        self.event_handlers = {}
        self.signal_queue = asyncio.Queue()
        self.event_queue = asyncio.Queue()
        
        # Performance metrics
        self.metrics = {
            'signals_received': 0,
            'signals_processed': 0,
            'events_handled': 0,
            'connection_uptime': 0.0,
            'last_signal_time': None,
            'average_latency_ms': 0.0,
            'reconnection_count': 0
        }
        
        # Connection monitoring
        self.connection_start_time = None
        self.last_heartbeat = None
        self.reconnect_attempts = 0
        
        # Setup event handlers
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """Setup default event handlers"""
        self.register_handler(WebSocketEventType.SIGNAL_RECEIVED, self._handle_signal_received)
        self.register_handler(WebSocketEventType.SIGNAL_PROCESSED, self._handle_signal_processed)
        self.register_handler(WebSocketEventType.ERROR_OCCURRED, self._handle_error)
    
    def register_handler(self, event_type: WebSocketEventType, handler: Callable):
        """Register event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for {event_type}")
    
    async def start_websocket_server(self):
        """Start WebSocket server for real-time connections"""
        try:
            # Setup SSL context if enabled
            ssl_context = None
            if self.ws_config.ssl_enabled:
                ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                ssl_context.load_cert_chain(self.ws_config.ssl_cert_path, self.ws_config.ssl_key_path)
            
            # Start WebSocket server
            server = await websockets.serve(
                self._handle_websocket_connection,
                self.ws_config.server_host,
                self.ws_config.server_port,
                ssl=ssl_context,
                max_size=1024*1024,  # 1MB max message size
                ping_interval=self.ws_config.heartbeat_interval,
                ping_timeout=self.ws_config.heartbeat_interval * 2
            )
            
            self.connection_state = ConnectionState.CONNECTED
            self.connection_start_time = time.time()
            
            logger.info(f"WebSocket server started on {self.ws_config.server_host}:{self.ws_config.server_port}")
            
            # Start background tasks
            await asyncio.gather(
                self._process_signal_queue(),
                self._process_event_queue(),
                self._monitor_connection_health()
            )
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            self.connection_state = ConnectionState.FAILED
            raise
    
    async def start_webhook_server(self):
        """Start webhook server for TradingView signals"""
        try:
            self.webhook_app = web.Application()
            
            # Setup webhook routes
            self.webhook_app.router.add_post('/webhook/tradingview', self._handle_tradingview_webhook)
            self.webhook_app.router.add_post('/webhook/signal', self._handle_generic_signal_webhook)
            self.webhook_app.router.add_get('/health', self._handle_health_check)
            self.webhook_app.router.add_get('/metrics', self._handle_metrics_request)
            
            # Setup CORS if needed
            self.webhook_app.middlewares.append(self._cors_middleware)
            
            # Start webhook server
            self.webhook_runner = web.AppRunner(self.webhook_app)
            await self.webhook_runner.setup()
            
            site = web.TCPSite(self.webhook_runner, self.ws_config.server_host, self.ws_config.webhook_port)
            await site.start()
            
            logger.info(f"Webhook server started on {self.ws_config.server_host}:{self.ws_config.webhook_port}")
            
        except Exception as e:
            logger.error(f"Failed to start webhook server: {e}")
            raise
    
    async def _handle_websocket_connection(self, websocket, path):
        """Handle incoming WebSocket connections"""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"New WebSocket connection from {client_id}")
        
        try:
            # Send welcome message
            welcome_msg = {
                'type': 'welcome',
                'timestamp': datetime.now().isoformat(),
                'server_info': {
                    'strategy': 'TV',
                    'version': '1.0.0',
                    'capabilities': ['signal_processing', 'real_time_updates']
                }
            }
            await websocket.send(json.dumps(welcome_msg))
            
            # Handle messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._process_websocket_message(websocket, data, client_id)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON from {client_id}: {message}")
                    await websocket.send(json.dumps({'error': 'Invalid JSON format'}))
                except Exception as e:
                    logger.error(f"Error processing message from {client_id}: {e}")
                    await websocket.send(json.dumps({'error': str(e)}))
                    
        except ConnectionClosed:
            logger.info(f"WebSocket connection closed for {client_id}")
        except Exception as e:
            logger.error(f"WebSocket error for {client_id}: {e}")
    
    async def _process_websocket_message(self, websocket, data: Dict[str, Any], client_id: str):
        """Process incoming WebSocket message"""
        message_type = data.get('type')
        
        if message_type == 'signal':
            # Process trading signal
            signal = self._create_signal_from_websocket(data, client_id)
            await self.signal_queue.put(signal)
            
            # Send acknowledgment
            ack_msg = {
                'type': 'signal_ack',
                'signal_id': signal.signal_id,
                'timestamp': datetime.now().isoformat(),
                'status': 'received'
            }
            await websocket.send(json.dumps(ack_msg))
            
        elif message_type == 'heartbeat':
            # Respond to heartbeat
            pong_msg = {
                'type': 'pong',
                'timestamp': datetime.now().isoformat()
            }
            await websocket.send(json.dumps(pong_msg))
            self.last_heartbeat = time.time()
            
        elif message_type == 'subscribe':
            # Handle subscription requests
            await self._handle_subscription(websocket, data, client_id)
            
        else:
            logger.warning(f"Unknown message type '{message_type}' from {client_id}")
    
    async def _handle_tradingview_webhook(self, request):
        """Handle TradingView webhook requests"""
        try:
            data = await request.json()
            
            # Validate webhook data
            if not self._validate_tradingview_webhook(data):
                return web.json_response({'error': 'Invalid webhook data'}, status=400)
            
            # Create signal from webhook
            signal = self._create_signal_from_webhook(data, SignalSource.TRADINGVIEW)
            await self.signal_queue.put(signal)
            
            self.metrics['signals_received'] += 1
            
            return web.json_response({
                'status': 'success',
                'signal_id': signal.signal_id,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error handling TradingView webhook: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def _handle_generic_signal_webhook(self, request):
        """Handle generic signal webhook requests"""
        try:
            data = await request.json()
            
            # Create signal from generic webhook
            signal = self._create_signal_from_webhook(data, SignalSource.WEBHOOK)
            await self.signal_queue.put(signal)
            
            self.metrics['signals_received'] += 1
            
            return web.json_response({
                'status': 'success',
                'signal_id': signal.signal_id,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error handling generic webhook: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def _handle_health_check(self, request):
        """Handle health check requests"""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'connection_state': self.connection_state.value,
            'uptime_seconds': time.time() - self.connection_start_time if self.connection_start_time else 0,
            'metrics': self.metrics
        }
        return web.json_response(health_status)
    
    async def _handle_metrics_request(self, request):
        """Handle metrics requests"""
        return web.json_response(self.metrics)
    
    async def _cors_middleware(self, request, handler):
        """CORS middleware for webhook server"""
        response = await handler(request)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
    
    def _create_signal_from_websocket(self, data: Dict[str, Any], client_id: str) -> TVSignal:
        """Create TVSignal from WebSocket data"""
        signal_id = f"WS_{client_id}_{int(time.time() * 1000)}"
        
        return TVSignal(
            signal_id=signal_id,
            timestamp=datetime.now(),
            source=SignalSource.API,
            signal_type=data.get('signal_type', 'BUY'),
            symbol=data.get('symbol', 'NIFTY'),
            action=data.get('action', 'BUY'),
            quantity=data.get('quantity', 50),
            price=data.get('price'),
            confidence=data.get('confidence', 1.0),
            metadata={'client_id': client_id, 'websocket_data': data}
        )
    
    def _create_signal_from_webhook(self, data: Dict[str, Any], source: SignalSource) -> TVSignal:
        """Create TVSignal from webhook data"""
        signal_id = f"WH_{source.value}_{int(time.time() * 1000)}"
        
        return TVSignal(
            signal_id=signal_id,
            timestamp=datetime.now(),
            source=source,
            signal_type=data.get('signal_type', 'BUY'),
            symbol=data.get('symbol', 'NIFTY'),
            action=data.get('action', 'BUY'),
            quantity=data.get('quantity', 50),
            price=data.get('price'),
            confidence=data.get('confidence', 1.0),
            webhook_data=data
        )
    
    def _validate_tradingview_webhook(self, data: Dict[str, Any]) -> bool:
        """Validate TradingView webhook data"""
        required_fields = ['symbol', 'action']
        return all(field in data for field in required_fields)
    
    async def _process_signal_queue(self):
        """Process signals from the queue"""
        while True:
            try:
                signal = await self.signal_queue.get()
                
                # Process signal
                start_time = time.time()
                processed_signal = await self.signal_processor.process_signal(signal)
                processing_time = (time.time() - start_time) * 1000
                
                # Update metrics
                self.metrics['signals_processed'] += 1
                self.metrics['last_signal_time'] = datetime.now().isoformat()
                self.metrics['average_latency_ms'] = (
                    (self.metrics['average_latency_ms'] * (self.metrics['signals_processed'] - 1) + processing_time) /
                    self.metrics['signals_processed']
                )
                
                # Create and queue event
                event = WebSocketEvent(
                    event_id=f"EVT_{int(time.time() * 1000)}",
                    event_type=WebSocketEventType.SIGNAL_PROCESSED,
                    timestamp=datetime.now(),
                    data={'signal': processed_signal.__dict__, 'processing_time_ms': processing_time}
                )
                await self.event_queue.put(event)
                
            except Exception as e:
                logger.error(f"Error processing signal: {e}")
                
                # Create error event
                error_event = WebSocketEvent(
                    event_id=f"ERR_{int(time.time() * 1000)}",
                    event_type=WebSocketEventType.ERROR_OCCURRED,
                    timestamp=datetime.now(),
                    data={'error': str(e), 'signal_id': getattr(signal, 'signal_id', 'unknown')}
                )
                await self.event_queue.put(error_event)
    
    async def _process_event_queue(self):
        """Process events from the queue"""
        while True:
            try:
                event = await self.event_queue.get()
                
                # Handle event
                await self._handle_event(event)
                
                self.metrics['events_handled'] += 1
                
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    async def _handle_event(self, event: WebSocketEvent):
        """Handle WebSocket event"""
        handlers = self.event_handlers.get(event.event_type, [])
        
        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Error in event handler for {event.event_type}: {e}")
    
    async def _handle_signal_received(self, event: WebSocketEvent):
        """Handle signal received event"""
        logger.info(f"Signal received: {event.data.get('signal_id', 'unknown')}")
    
    async def _handle_signal_processed(self, event: WebSocketEvent):
        """Handle signal processed event"""
        signal_data = event.data.get('signal', {})
        processing_time = event.data.get('processing_time_ms', 0)
        logger.info(f"Signal processed: {signal_data.get('signal_id', 'unknown')} in {processing_time:.2f}ms")
    
    async def _handle_error(self, event: WebSocketEvent):
        """Handle error event"""
        error = event.data.get('error', 'Unknown error')
        signal_id = event.data.get('signal_id', 'unknown')
        logger.error(f"Error processing signal {signal_id}: {error}")
    
    async def _monitor_connection_health(self):
        """Monitor connection health and handle reconnections"""
        while True:
            try:
                # Update uptime
                if self.connection_start_time:
                    self.metrics['connection_uptime'] = time.time() - self.connection_start_time
                
                # Check connection health
                if self.connection_state == ConnectionState.CONNECTED:
                    # Check if heartbeat is recent
                    if self.last_heartbeat and (time.time() - self.last_heartbeat) > self.ws_config.heartbeat_interval * 3:
                        logger.warning("Heartbeat timeout detected")
                        self.connection_state = ConnectionState.RECONNECTING
                
                await asyncio.sleep(self.ws_config.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in connection health monitoring: {e}")
                await asyncio.sleep(5)
    
    async def _handle_subscription(self, websocket, data: Dict[str, Any], client_id: str):
        """Handle subscription requests"""
        subscription_type = data.get('subscription_type')
        
        if subscription_type == 'signals':
            # Subscribe to signal updates
            response = {
                'type': 'subscription_ack',
                'subscription_type': 'signals',
                'status': 'subscribed',
                'timestamp': datetime.now().isoformat()
            }
            await websocket.send(json.dumps(response))
            logger.info(f"Client {client_id} subscribed to signals")
        
        elif subscription_type == 'metrics':
            # Subscribe to metrics updates
            response = {
                'type': 'subscription_ack',
                'subscription_type': 'metrics',
                'status': 'subscribed',
                'timestamp': datetime.now().isoformat()
            }
            await websocket.send(json.dumps(response))
            logger.info(f"Client {client_id} subscribed to metrics")
    
    async def start(self):
        """Start WebSocket integration"""
        logger.info("Starting TV WebSocket integration...")
        
        try:
            # Start both servers concurrently
            await asyncio.gather(
                self.start_websocket_server(),
                self.start_webhook_server()
            )
        except Exception as e:
            logger.error(f"Failed to start WebSocket integration: {e}")
            raise
    
    async def stop(self):
        """Stop WebSocket integration"""
        logger.info("Stopping TV WebSocket integration...")
        
        try:
            # Close webhook server
            if self.webhook_runner:
                await self.webhook_runner.cleanup()
            
            # Close WebSocket connections
            if self.websocket:
                await self.websocket.close()
            
            self.connection_state = ConnectionState.DISCONNECTED
            
        except Exception as e:
            logger.error(f"Error stopping WebSocket integration: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get WebSocket integration metrics"""
        return self.metrics.copy()
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get connection status information"""
        return {
            'state': self.connection_state.value,
            'uptime_seconds': self.metrics['connection_uptime'],
            'reconnect_attempts': self.reconnect_attempts,
            'last_heartbeat': self.last_heartbeat,
            'server_info': {
                'websocket_port': self.ws_config.server_port,
                'webhook_port': self.ws_config.webhook_port,
                'ssl_enabled': self.ws_config.ssl_enabled
            }
        }


# Singleton instance for global access
_tv_websocket_integration = None

def get_tv_websocket_integration(config: TVConfigModel, ws_config: Optional[WebSocketConfig] = None) -> TVWebSocketIntegration:
    """Get singleton instance of TV WebSocket integration"""
    global _tv_websocket_integration
    
    if _tv_websocket_integration is None:
        _tv_websocket_integration = TVWebSocketIntegration(config, ws_config)
    
    return _tv_websocket_integration
