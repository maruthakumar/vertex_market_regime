"""
Open Interest Strategy WebSocket Integration
Real-time OI data streaming and analysis for Open Interest strategy

This module provides comprehensive WebSocket integration for:
- Real-time Open Interest data streaming
- Dynamic OI change detection and alerts
- Real-time OI pattern analysis
- Event-driven OI processing
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
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import websockets
from websockets.exceptions import ConnectionClosed, InvalidURI
import aiohttp
from aiohttp import web

from .models import OIConfig, OISignal, OIAnalysisResult
from .oi_analyzer import OIAnalyzer

logger = logging.getLogger(__name__)


class OIEventType(str, Enum):
    """Open Interest event types"""
    OI_DATA_RECEIVED = "OI_DATA_RECEIVED"
    OI_SPIKE_DETECTED = "OI_SPIKE_DETECTED"
    OI_PATTERN_IDENTIFIED = "OI_PATTERN_IDENTIFIED"
    PCR_CHANGE_DETECTED = "PCR_CHANGE_DETECTED"
    VOLUME_SURGE_DETECTED = "VOLUME_SURGE_DETECTED"
    ERROR_OCCURRED = "ERROR_OCCURRED"


@dataclass
class OIWebSocketEvent:
    """Open Interest WebSocket event"""
    event_id: str
    event_type: OIEventType
    timestamp: datetime
    data: Dict[str, Any]
    source: str = "OI_STRATEGY"
    processed: bool = False
    
    def mark_processed(self):
        """Mark event as processed"""
        self.processed = True


@dataclass
class OIStreamConfig:
    """OI streaming configuration"""
    server_host: str = "localhost"
    server_port: int = 8767
    data_feed_port: int = 8768
    max_connections: int = 50
    heartbeat_interval: int = 30
    oi_update_interval: int = 5  # seconds
    spike_threshold: float = 0.2  # 20% change
    volume_surge_threshold: float = 2.0  # 2x average
    pcr_alert_threshold: float = 0.1  # 10% change


class OIWebSocketIntegration:
    """
    Comprehensive WebSocket integration for Open Interest Strategy
    
    Provides real-time OI data streaming, pattern detection, and event handling
    """
    
    def __init__(self, config: OIConfig, stream_config: Optional[OIStreamConfig] = None):
        """Initialize OI WebSocket integration"""
        self.config = config
        self.stream_config = stream_config or OIStreamConfig()
        self.oi_analyzer = OIAnalyzer()
        
        # Connection management
        self.connected_clients = set()
        self.data_feed_server = None
        self.websocket_server = None
        
        # Event handling
        self.event_handlers = {}
        self.oi_data_queue = asyncio.Queue()
        self.event_queue = asyncio.Queue()
        
        # OI data tracking
        self.current_oi_data = {}
        self.previous_oi_data = {}
        self.oi_history = {}
        
        # Performance metrics
        self.metrics = {
            'oi_updates_received': 0,
            'oi_spikes_detected': 0,
            'patterns_identified': 0,
            'events_processed': 0,
            'connected_clients': 0,
            'data_feed_uptime': 0.0,
            'last_oi_update': None,
            'average_processing_time_ms': 0.0
        }
        
        # Setup event handlers
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """Setup default event handlers"""
        self.register_handler(OIEventType.OI_DATA_RECEIVED, self._handle_oi_data_received)
        self.register_handler(OIEventType.OI_SPIKE_DETECTED, self._handle_oi_spike_detected)
        self.register_handler(OIEventType.OI_PATTERN_IDENTIFIED, self._handle_oi_pattern_identified)
        self.register_handler(OIEventType.PCR_CHANGE_DETECTED, self._handle_pcr_change_detected)
        self.register_handler(OIEventType.VOLUME_SURGE_DETECTED, self._handle_volume_surge_detected)
        self.register_handler(OIEventType.ERROR_OCCURRED, self._handle_error)
    
    def register_handler(self, event_type: OIEventType, handler: Callable):
        """Register event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for {event_type}")
    
    async def start_websocket_server(self):
        """Start WebSocket server for OI data streaming"""
        try:
            # Start WebSocket server
            self.websocket_server = await websockets.serve(
                self._handle_websocket_connection,
                self.stream_config.server_host,
                self.stream_config.server_port,
                max_size=1024*1024,  # 1MB max message size
                ping_interval=self.stream_config.heartbeat_interval,
                ping_timeout=self.stream_config.heartbeat_interval * 2
            )
            
            logger.info(f"OI WebSocket server started on {self.stream_config.server_host}:{self.stream_config.server_port}")
            
            # Start background tasks
            await asyncio.gather(
                self._process_oi_data_queue(),
                self._process_event_queue(),
                self._monitor_oi_changes(),
                self._broadcast_oi_updates()
            )
            
        except Exception as e:
            logger.error(f"Failed to start OI WebSocket server: {e}")
            raise
    
    async def start_data_feed_server(self):
        """Start data feed server for receiving OI data"""
        try:
            app = web.Application()
            
            # Setup data feed routes
            app.router.add_post('/oi/update', self._handle_oi_data_update)
            app.router.add_post('/oi/bulk', self._handle_bulk_oi_update)
            app.router.add_get('/oi/status', self._handle_oi_status)
            app.router.add_get('/oi/metrics', self._handle_oi_metrics)
            
            # Setup CORS
            app.middlewares.append(self._cors_middleware)
            
            # Start data feed server
            runner = web.AppRunner(app)
            await runner.setup()
            
            site = web.TCPSite(runner, self.stream_config.server_host, self.stream_config.data_feed_port)
            await site.start()
            
            logger.info(f"OI data feed server started on {self.stream_config.server_host}:{self.stream_config.data_feed_port}")
            
        except Exception as e:
            logger.error(f"Failed to start OI data feed server: {e}")
            raise
    
    async def _handle_websocket_connection(self, websocket, path):
        """Handle incoming WebSocket connections"""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        self.connected_clients.add(websocket)
        self.metrics['connected_clients'] = len(self.connected_clients)
        
        logger.info(f"New OI WebSocket connection from {client_id}")
        
        try:
            # Send welcome message with current OI data
            welcome_msg = {
                'type': 'welcome',
                'timestamp': datetime.now().isoformat(),
                'current_oi_data': self.current_oi_data,
                'server_info': {
                    'strategy': 'OI',
                    'version': '1.0.0',
                    'capabilities': ['oi_streaming', 'pattern_detection', 'real_time_alerts']
                }
            }
            await websocket.send(json.dumps(welcome_msg))
            
            # Handle messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._process_oi_websocket_message(websocket, data, client_id)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON from {client_id}: {message}")
                    await websocket.send(json.dumps({'error': 'Invalid JSON format'}))
                except Exception as e:
                    logger.error(f"Error processing message from {client_id}: {e}")
                    await websocket.send(json.dumps({'error': str(e)}))
                    
        except ConnectionClosed:
            logger.info(f"OI WebSocket connection closed for {client_id}")
        except Exception as e:
            logger.error(f"OI WebSocket error for {client_id}: {e}")
        finally:
            self.connected_clients.discard(websocket)
            self.metrics['connected_clients'] = len(self.connected_clients)
    
    async def _process_oi_websocket_message(self, websocket, data: Dict[str, Any], client_id: str):
        """Process incoming OI WebSocket message"""
        message_type = data.get('type')
        
        if message_type == 'subscribe_oi':
            # Subscribe to OI updates
            symbols = data.get('symbols', ['NIFTY'])
            await self._handle_oi_subscription(websocket, symbols, client_id)
            
        elif message_type == 'get_oi_summary':
            # Get OI summary
            summary = await self._get_oi_summary(data.get('symbols', ['NIFTY']))
            response = {
                'type': 'oi_summary',
                'data': summary,
                'timestamp': datetime.now().isoformat()
            }
            await websocket.send(json.dumps(response))
            
        elif message_type == 'set_alert_thresholds':
            # Set custom alert thresholds
            await self._set_alert_thresholds(data.get('thresholds', {}))
            response = {
                'type': 'thresholds_updated',
                'timestamp': datetime.now().isoformat()
            }
            await websocket.send(json.dumps(response))
            
        elif message_type == 'heartbeat':
            # Respond to heartbeat
            pong_msg = {
                'type': 'pong',
                'timestamp': datetime.now().isoformat()
            }
            await websocket.send(json.dumps(pong_msg))
            
        else:
            logger.warning(f"Unknown OI message type '{message_type}' from {client_id}")
    
    async def _handle_oi_data_update(self, request):
        """Handle OI data update requests"""
        try:
            data = await request.json()
            
            # Validate OI data
            if not self._validate_oi_data(data):
                return web.json_response({'error': 'Invalid OI data'}, status=400)
            
            # Queue OI data for processing
            await self.oi_data_queue.put(data)
            
            self.metrics['oi_updates_received'] += 1
            
            return web.json_response({
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error handling OI data update: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def _handle_bulk_oi_update(self, request):
        """Handle bulk OI data updates"""
        try:
            data = await request.json()
            
            # Process bulk OI data
            oi_records = data.get('oi_data', [])
            for oi_record in oi_records:
                if self._validate_oi_data(oi_record):
                    await self.oi_data_queue.put(oi_record)
            
            self.metrics['oi_updates_received'] += len(oi_records)
            
            return web.json_response({
                'status': 'success',
                'records_processed': len(oi_records),
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error handling bulk OI update: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def _handle_oi_status(self, request):
        """Handle OI status requests"""
        status = {
            'status': 'active',
            'timestamp': datetime.now().isoformat(),
            'connected_clients': self.metrics['connected_clients'],
            'current_oi_symbols': list(self.current_oi_data.keys()),
            'last_update': self.metrics['last_oi_update'],
            'metrics': self.metrics
        }
        return web.json_response(status)
    
    async def _handle_oi_metrics(self, request):
        """Handle OI metrics requests"""
        return web.json_response(self.metrics)
    
    async def _cors_middleware(self, request, handler):
        """CORS middleware for data feed server"""
        response = await handler(request)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
    
    def _validate_oi_data(self, data: Dict[str, Any]) -> bool:
        """Validate OI data"""
        required_fields = ['symbol', 'timestamp', 'option_type', 'strike_price', 'open_interest']
        return all(field in data for field in required_fields)
    
    async def _process_oi_data_queue(self):
        """Process OI data from the queue"""
        while True:
            try:
                oi_data = await self.oi_data_queue.get()
                
                # Process OI data
                start_time = time.time()
                await self._process_oi_data(oi_data)
                processing_time = (time.time() - start_time) * 1000
                
                # Update metrics
                self.metrics['last_oi_update'] = datetime.now().isoformat()
                self.metrics['average_processing_time_ms'] = (
                    (self.metrics['average_processing_time_ms'] * self.metrics['oi_updates_received'] + processing_time) /
                    (self.metrics['oi_updates_received'] + 1)
                )
                
            except Exception as e:
                logger.error(f"Error processing OI data: {e}")
    
    async def _process_oi_data(self, oi_data: Dict[str, Any]):
        """Process individual OI data record"""
        symbol = oi_data['symbol']
        strike_price = oi_data['strike_price']
        option_type = oi_data['option_type']
        open_interest = oi_data['open_interest']
        
        # Create key for tracking
        oi_key = f"{symbol}_{option_type}_{strike_price}"
        
        # Store previous data
        if oi_key in self.current_oi_data:
            self.previous_oi_data[oi_key] = self.current_oi_data[oi_key].copy()
        
        # Update current data
        self.current_oi_data[oi_key] = oi_data
        
        # Detect changes and patterns
        await self._detect_oi_changes(oi_key, oi_data)
        
        # Create event
        event = OIWebSocketEvent(
            event_id=f"OI_{int(time.time() * 1000)}",
            event_type=OIEventType.OI_DATA_RECEIVED,
            timestamp=datetime.now(),
            data={'oi_key': oi_key, 'oi_data': oi_data}
        )
        await self.event_queue.put(event)
    
    async def _detect_oi_changes(self, oi_key: str, current_data: Dict[str, Any]):
        """Detect significant OI changes and patterns"""
        if oi_key not in self.previous_oi_data:
            return
        
        previous_data = self.previous_oi_data[oi_key]
        current_oi = current_data['open_interest']
        previous_oi = previous_data['open_interest']
        
        # Calculate change percentage
        if previous_oi > 0:
            change_pct = (current_oi - previous_oi) / previous_oi
            
            # Detect OI spike
            if abs(change_pct) >= self.stream_config.spike_threshold:
                event = OIWebSocketEvent(
                    event_id=f"SPIKE_{int(time.time() * 1000)}",
                    event_type=OIEventType.OI_SPIKE_DETECTED,
                    timestamp=datetime.now(),
                    data={
                        'oi_key': oi_key,
                        'change_pct': change_pct,
                        'previous_oi': previous_oi,
                        'current_oi': current_oi,
                        'data': current_data
                    }
                )
                await self.event_queue.put(event)
                self.metrics['oi_spikes_detected'] += 1
        
        # Detect volume surge
        current_volume = current_data.get('volume', 0)
        previous_volume = previous_data.get('volume', 0)
        
        if previous_volume > 0:
            volume_ratio = current_volume / previous_volume
            if volume_ratio >= self.stream_config.volume_surge_threshold:
                event = OIWebSocketEvent(
                    event_id=f"VOL_{int(time.time() * 1000)}",
                    event_type=OIEventType.VOLUME_SURGE_DETECTED,
                    timestamp=datetime.now(),
                    data={
                        'oi_key': oi_key,
                        'volume_ratio': volume_ratio,
                        'previous_volume': previous_volume,
                        'current_volume': current_volume,
                        'data': current_data
                    }
                )
                await self.event_queue.put(event)
    
    async def _process_event_queue(self):
        """Process events from the queue"""
        while True:
            try:
                event = await self.event_queue.get()
                
                # Handle event
                await self._handle_event(event)
                
                self.metrics['events_processed'] += 1
                
            except Exception as e:
                logger.error(f"Error processing OI event: {e}")
    
    async def _handle_event(self, event: OIWebSocketEvent):
        """Handle OI WebSocket event"""
        handlers = self.event_handlers.get(event.event_type, [])
        
        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Error in OI event handler for {event.event_type}: {e}")
    
    async def _handle_oi_data_received(self, event: OIWebSocketEvent):
        """Handle OI data received event"""
        oi_key = event.data.get('oi_key', 'unknown')
        logger.debug(f"OI data received: {oi_key}")
    
    async def _handle_oi_spike_detected(self, event: OIWebSocketEvent):
        """Handle OI spike detected event"""
        oi_key = event.data.get('oi_key', 'unknown')
        change_pct = event.data.get('change_pct', 0)
        logger.info(f"OI spike detected: {oi_key} changed by {change_pct:.2%}")
        
        # Broadcast alert to connected clients
        alert_msg = {
            'type': 'oi_spike_alert',
            'oi_key': oi_key,
            'change_pct': change_pct,
            'timestamp': datetime.now().isoformat(),
            'data': event.data
        }
        await self._broadcast_to_clients(alert_msg)
    
    async def _handle_oi_pattern_identified(self, event: OIWebSocketEvent):
        """Handle OI pattern identified event"""
        pattern = event.data.get('pattern', 'unknown')
        logger.info(f"OI pattern identified: {pattern}")
    
    async def _handle_pcr_change_detected(self, event: OIWebSocketEvent):
        """Handle PCR change detected event"""
        pcr_change = event.data.get('pcr_change', 0)
        logger.info(f"PCR change detected: {pcr_change:.2%}")
    
    async def _handle_volume_surge_detected(self, event: OIWebSocketEvent):
        """Handle volume surge detected event"""
        oi_key = event.data.get('oi_key', 'unknown')
        volume_ratio = event.data.get('volume_ratio', 0)
        logger.info(f"Volume surge detected: {oi_key} volume increased by {volume_ratio:.1f}x")
        
        # Broadcast alert to connected clients
        alert_msg = {
            'type': 'volume_surge_alert',
            'oi_key': oi_key,
            'volume_ratio': volume_ratio,
            'timestamp': datetime.now().isoformat(),
            'data': event.data
        }
        await self._broadcast_to_clients(alert_msg)
    
    async def _handle_error(self, event: OIWebSocketEvent):
        """Handle error event"""
        error = event.data.get('error', 'Unknown error')
        logger.error(f"OI WebSocket error: {error}")
    
    async def _monitor_oi_changes(self):
        """Monitor OI changes and detect patterns"""
        while True:
            try:
                # Analyze current OI data for patterns
                if self.current_oi_data:
                    patterns = await self._analyze_oi_patterns()
                    
                    for pattern in patterns:
                        event = OIWebSocketEvent(
                            event_id=f"PATTERN_{int(time.time() * 1000)}",
                            event_type=OIEventType.OI_PATTERN_IDENTIFIED,
                            timestamp=datetime.now(),
                            data={'pattern': pattern}
                        )
                        await self.event_queue.put(event)
                        self.metrics['patterns_identified'] += 1
                
                await asyncio.sleep(self.stream_config.oi_update_interval)
                
            except Exception as e:
                logger.error(f"Error in OI monitoring: {e}")
                await asyncio.sleep(5)
    
    async def _broadcast_oi_updates(self):
        """Broadcast OI updates to connected clients"""
        while True:
            try:
                if self.connected_clients and self.current_oi_data:
                    # Create OI update message
                    update_msg = {
                        'type': 'oi_update',
                        'timestamp': datetime.now().isoformat(),
                        'oi_data': self.current_oi_data,
                        'summary': await self._get_oi_summary()
                    }
                    
                    await self._broadcast_to_clients(update_msg)
                
                await asyncio.sleep(self.stream_config.oi_update_interval)
                
            except Exception as e:
                logger.error(f"Error broadcasting OI updates: {e}")
                await asyncio.sleep(5)
    
    async def _broadcast_to_clients(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        if not self.connected_clients:
            return
        
        message_json = json.dumps(message)
        disconnected_clients = set()
        
        for client in self.connected_clients:
            try:
                await client.send(message_json)
            except ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                logger.error(f"Error sending message to client: {e}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.connected_clients -= disconnected_clients
        self.metrics['connected_clients'] = len(self.connected_clients)
    
    async def _analyze_oi_patterns(self) -> List[str]:
        """Analyze OI data for patterns"""
        patterns = []
        
        # Simple pattern detection (can be enhanced)
        if len(self.current_oi_data) > 10:
            # Check for unusual OI distribution
            oi_values = [data['open_interest'] for data in self.current_oi_data.values()]
            avg_oi = sum(oi_values) / len(oi_values)
            
            high_oi_count = sum(1 for oi in oi_values if oi > avg_oi * 2)
            if high_oi_count > len(oi_values) * 0.1:  # More than 10% have high OI
                patterns.append('HIGH_OI_CONCENTRATION')
        
        return patterns
    
    async def _get_oi_summary(self, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get OI summary for specified symbols"""
        if symbols is None:
            symbols = list(set(data['symbol'] for data in self.current_oi_data.values()))
        
        summary = {}
        
        for symbol in symbols:
            symbol_data = [data for key, data in self.current_oi_data.items() if data['symbol'] == symbol]
            
            if symbol_data:
                total_oi = sum(data['open_interest'] for data in symbol_data)
                call_oi = sum(data['open_interest'] for data in symbol_data if data['option_type'] == 'CE')
                put_oi = sum(data['open_interest'] for data in symbol_data if data['option_type'] == 'PE')
                
                summary[symbol] = {
                    'total_oi': total_oi,
                    'call_oi': call_oi,
                    'put_oi': put_oi,
                    'pcr_oi': put_oi / call_oi if call_oi > 0 else 0,
                    'last_update': max(data['timestamp'] for data in symbol_data)
                }
        
        return summary
    
    async def _handle_oi_subscription(self, websocket, symbols: List[str], client_id: str):
        """Handle OI subscription request"""
        response = {
            'type': 'oi_subscription_ack',
            'symbols': symbols,
            'status': 'subscribed',
            'timestamp': datetime.now().isoformat()
        }
        await websocket.send(json.dumps(response))
        logger.info(f"Client {client_id} subscribed to OI updates for {symbols}")
    
    async def _set_alert_thresholds(self, thresholds: Dict[str, float]):
        """Set custom alert thresholds"""
        if 'spike_threshold' in thresholds:
            self.stream_config.spike_threshold = thresholds['spike_threshold']
        if 'volume_surge_threshold' in thresholds:
            self.stream_config.volume_surge_threshold = thresholds['volume_surge_threshold']
        if 'pcr_alert_threshold' in thresholds:
            self.stream_config.pcr_alert_threshold = thresholds['pcr_alert_threshold']
        
        logger.info(f"Alert thresholds updated: {thresholds}")
    
    async def start(self):
        """Start OI WebSocket integration"""
        logger.info("Starting OI WebSocket integration...")
        
        try:
            # Start both servers concurrently
            await asyncio.gather(
                self.start_websocket_server(),
                self.start_data_feed_server()
            )
        except Exception as e:
            logger.error(f"Failed to start OI WebSocket integration: {e}")
            raise
    
    async def stop(self):
        """Stop OI WebSocket integration"""
        logger.info("Stopping OI WebSocket integration...")
        
        try:
            # Close all connections
            for client in self.connected_clients:
                await client.close()
            
            # Close servers
            if self.websocket_server:
                self.websocket_server.close()
                await self.websocket_server.wait_closed()
            
        except Exception as e:
            logger.error(f"Error stopping OI WebSocket integration: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get OI WebSocket integration metrics"""
        return self.metrics.copy()
    
    def get_current_oi_data(self) -> Dict[str, Any]:
        """Get current OI data"""
        return self.current_oi_data.copy()


# Singleton instance for global access
_oi_websocket_integration = None

def get_oi_websocket_integration(config: OIConfig, stream_config: Optional[OIStreamConfig] = None) -> OIWebSocketIntegration:
    """Get singleton instance of OI WebSocket integration"""
    global _oi_websocket_integration
    
    if _oi_websocket_integration is None:
        _oi_websocket_integration = OIWebSocketIntegration(config, stream_config)
    
    return _oi_websocket_integration
