"""
Real-time Monitoring Dashboard for Market Regime Detection System

This module provides comprehensive real-time monitoring with WebSocket integration
for the enhanced Market Regime Detection System, supporting 50+ concurrent users
with <50ms response times and <2 second update latency.

Features:
1. Real-time regime detection status monitoring
2. Technical indicator status tracking
3. Performance metrics visualization
4. WebSocket-based live updates
5. Multi-user support with connection management
6. Progressive disclosure based on user skill level
7. Alert system for regime changes
8. Historical performance tracking

Author: The Augster
Date: 2025-01-16
Version: 1.0.0
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import websockets
from websockets.server import WebSocketServerProtocol
import threading
from collections import deque

logger = logging.getLogger(__name__)

class UserSkillLevel(Enum):
    """User skill levels for progressive disclosure"""
    NOVICE = "novice"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"

class AlertType(Enum):
    """Types of alerts"""
    REGIME_CHANGE = "regime_change"
    INDICATOR_ANOMALY = "indicator_anomaly"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SYSTEM_ERROR = "system_error"

@dataclass
class ConnectionInfo:
    """Information about WebSocket connection"""
    websocket: WebSocketServerProtocol
    user_id: str
    skill_level: UserSkillLevel
    connected_at: datetime
    last_activity: datetime
    subscriptions: Set[str]

@dataclass
class MonitoringData:
    """Real-time monitoring data structure"""
    timestamp: datetime
    regime_status: Dict[str, Any]
    technical_indicators: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    system_health: Dict[str, Any]
    alerts: List[Dict[str, Any]]

class RealtimeMonitoringDashboard:
    """
    Real-time monitoring dashboard with WebSocket integration
    
    Provides comprehensive real-time monitoring for the Market Regime Detection System
    with support for 50+ concurrent users and <50ms response times.
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        """Initialize Real-time Monitoring Dashboard"""
        self.host = host
        self.port = port
        
        # Connection management
        self.connections: Dict[str, ConnectionInfo] = {}
        self.max_connections = 100
        self.connection_timeout = 300  # 5 minutes
        
        # Data storage
        self.current_data: Optional[MonitoringData] = None
        self.historical_data = deque(maxlen=1000)  # Keep last 1000 data points
        self.alerts_queue = deque(maxlen=100)  # Keep last 100 alerts
        
        # Performance tracking
        self.performance_stats = {
            'total_connections': 0,
            'active_connections': 0,
            'messages_sent': 0,
            'messages_received': 0,
            'average_response_time': 0.0,
            'last_update_time': None
        }
        
        # Update frequency control
        self.update_interval = 1.0  # 1 second updates
        self.last_broadcast = time.time()
        
        # WebSocket server
        self.server = None
        self.running = False
        
        # Background tasks
        self.cleanup_task = None
        self.monitoring_task = None
        
        logger.info(f"Real-time Monitoring Dashboard initialized on {host}:{port}")
    
    async def start_server(self):
        """Start the WebSocket server"""
        try:
            self.server = await websockets.serve(
                self.handle_connection,
                self.host,
                self.port,
                ping_interval=30,
                ping_timeout=10,
                max_size=1024*1024,  # 1MB max message size
                compression=None  # Disable compression for lower latency
            )
            
            self.running = True
            
            # Start background tasks
            self.cleanup_task = asyncio.create_task(self.cleanup_connections())
            self.monitoring_task = asyncio.create_task(self.monitoring_loop())
            
            logger.info(f"🚀 WebSocket server started on {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"❌ Failed to start WebSocket server: {e}")
            raise
    
    async def stop_server(self):
        """Stop the WebSocket server"""
        try:
            self.running = False
            
            # Cancel background tasks
            if self.cleanup_task:
                self.cleanup_task.cancel()
            if self.monitoring_task:
                self.monitoring_task.cancel()
            
            # Close all connections
            if self.connections:
                await asyncio.gather(
                    *[conn.websocket.close() for conn in self.connections.values()],
                    return_exceptions=True
                )
            
            # Stop server
            if self.server:
                self.server.close()
                await self.server.wait_closed()
            
            logger.info("✅ WebSocket server stopped")
            
        except Exception as e:
            logger.error(f"❌ Error stopping server: {e}")
    
    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new WebSocket connection"""
        connection_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}:{time.time()}"
        
        try:
            # Check connection limit
            if len(self.connections) >= self.max_connections:
                await websocket.close(code=1013, reason="Server overloaded")
                return
            
            # Wait for authentication message
            auth_message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            auth_data = json.loads(auth_message)
            
            # Validate authentication
            if not self._validate_auth(auth_data):
                await websocket.close(code=1008, reason="Authentication failed")
                return
            
            # Create connection info
            connection_info = ConnectionInfo(
                websocket=websocket,
                user_id=auth_data.get('user_id', 'anonymous'),
                skill_level=UserSkillLevel(auth_data.get('skill_level', 'expert')),
                connected_at=datetime.now(),
                last_activity=datetime.now(),
                subscriptions=set(auth_data.get('subscriptions', ['all']))
            )
            
            self.connections[connection_id] = connection_info
            self.performance_stats['total_connections'] += 1
            self.performance_stats['active_connections'] = len(self.connections)
            
            # Send welcome message with current data
            await self._send_welcome_message(websocket, connection_info)
            
            logger.info(f"✅ New connection: {connection_info.user_id} ({connection_info.skill_level.value})")
            
            # Handle messages from this connection
            async for message in websocket:
                await self._handle_message(connection_id, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.debug(f"Connection closed: {connection_id}")
        except asyncio.TimeoutError:
            logger.warning(f"Authentication timeout: {connection_id}")
        except Exception as e:
            logger.error(f"Error handling connection {connection_id}: {e}")
        finally:
            # Clean up connection
            if connection_id in self.connections:
                del self.connections[connection_id]
                self.performance_stats['active_connections'] = len(self.connections)
    
    def _validate_auth(self, auth_data: Dict[str, Any]) -> bool:
        """Validate authentication data"""
        try:
            # Basic validation - in production, implement proper authentication
            required_fields = ['user_id', 'skill_level']
            
            for field in required_fields:
                if field not in auth_data:
                    return False
            
            # Validate skill level
            try:
                UserSkillLevel(auth_data['skill_level'])
            except ValueError:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating auth: {e}")
            return False
    
    async def _send_welcome_message(self, websocket: WebSocketServerProtocol, connection_info: ConnectionInfo):
        """Send welcome message with current data"""
        try:
            welcome_data = {
                'type': 'welcome',
                'timestamp': datetime.now().isoformat(),
                'user_id': connection_info.user_id,
                'skill_level': connection_info.skill_level.value,
                'server_info': {
                    'version': '1.0.0',
                    'update_interval': self.update_interval,
                    'max_connections': self.max_connections
                }
            }
            
            # Include current data if available
            if self.current_data:
                welcome_data['current_data'] = self._filter_data_by_skill_level(
                    self.current_data, connection_info.skill_level
                )
            
            await websocket.send(json.dumps(welcome_data))
            
        except Exception as e:
            logger.error(f"Error sending welcome message: {e}")
    
    async def _handle_message(self, connection_id: str, message: str):
        """Handle incoming message from client"""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            connection_info = self.connections.get(connection_id)
            if not connection_info:
                return
            
            connection_info.last_activity = datetime.now()
            self.performance_stats['messages_received'] += 1
            
            if message_type == 'subscribe':
                # Update subscriptions
                new_subscriptions = set(data.get('subscriptions', []))
                connection_info.subscriptions = new_subscriptions
                
                # Send acknowledgment
                response = {
                    'type': 'subscription_updated',
                    'subscriptions': list(new_subscriptions),
                    'timestamp': datetime.now().isoformat()
                }
                await connection_info.websocket.send(json.dumps(response))
                
            elif message_type == 'ping':
                # Respond to ping
                pong = {
                    'type': 'pong',
                    'timestamp': datetime.now().isoformat()
                }
                await connection_info.websocket.send(json.dumps(pong))
                
            elif message_type == 'get_historical':
                # Send historical data
                await self._send_historical_data(connection_info, data.get('params', {}))
                
        except Exception as e:
            logger.error(f"Error handling message from {connection_id}: {e}")
    
    async def _send_historical_data(self, connection_info: ConnectionInfo, params: Dict[str, Any]):
        """Send historical data to client"""
        try:
            # Get requested time range
            hours_back = params.get('hours_back', 1)
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            # Filter historical data
            filtered_data = [
                self._filter_data_by_skill_level(data, connection_info.skill_level)
                for data in self.historical_data
                if data.timestamp >= cutoff_time
            ]
            
            response = {
                'type': 'historical_data',
                'data': [asdict(data) for data in filtered_data],
                'timestamp': datetime.now().isoformat()
            }
            
            await connection_info.websocket.send(json.dumps(response))
            
        except Exception as e:
            logger.error(f"Error sending historical data: {e}")
    
    def _filter_data_by_skill_level(self, data: MonitoringData, skill_level: UserSkillLevel) -> Dict[str, Any]:
        """Filter monitoring data based on user skill level"""
        try:
            filtered_data = {
                'timestamp': data.timestamp.isoformat(),
                'regime_status': data.regime_status.copy(),
                'alerts': data.alerts.copy()
            }
            
            if skill_level == UserSkillLevel.NOVICE:
                # Show only basic regime information
                filtered_data['technical_indicators'] = {
                    'summary': {
                        'active_indicators': len(data.technical_indicators),
                        'overall_confidence': data.technical_indicators.get('overall_confidence', 0.0)
                    }
                }
                filtered_data['performance_metrics'] = {
                    'system_status': 'healthy' if data.system_health.get('overall_health', 0) > 0.8 else 'warning'
                }
                
            elif skill_level == UserSkillLevel.INTERMEDIATE:
                # Show core technical indicators and basic performance metrics
                filtered_data['technical_indicators'] = {
                    key: value for key, value in data.technical_indicators.items()
                    if key in ['iv_percentile', 'iv_skew', 'overall_confidence']
                }
                filtered_data['performance_metrics'] = {
                    key: value for key, value in data.performance_metrics.items()
                    if key in ['accuracy', 'latency', 'throughput']
                }
                
            else:  # EXPERT
                # Show all data
                filtered_data['technical_indicators'] = data.technical_indicators.copy()
                filtered_data['performance_metrics'] = data.performance_metrics.copy()
                filtered_data['system_health'] = data.system_health.copy()
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"Error filtering data: {e}")
            return {'error': str(e)}
    
    async def cleanup_connections(self):
        """Clean up inactive connections"""
        while self.running:
            try:
                current_time = datetime.now()
                inactive_connections = []
                
                for conn_id, conn_info in self.connections.items():
                    if (current_time - conn_info.last_activity).total_seconds() > self.connection_timeout:
                        inactive_connections.append(conn_id)
                
                # Remove inactive connections
                for conn_id in inactive_connections:
                    try:
                        await self.connections[conn_id].websocket.close()
                        del self.connections[conn_id]
                        logger.debug(f"Cleaned up inactive connection: {conn_id}")
                    except Exception as e:
                        logger.error(f"Error cleaning up connection {conn_id}: {e}")
                
                if inactive_connections:
                    self.performance_stats['active_connections'] = len(self.connections)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(60)
    
    async def monitoring_loop(self):
        """Main monitoring loop for data updates"""
        while self.running:
            try:
                # This would be called by the main market regime system
                # For now, we'll simulate with a placeholder
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(1)
    
    async def update_monitoring_data(self, new_data: MonitoringData):
        """Update monitoring data and broadcast to clients"""
        try:
            start_time = time.time()
            
            # Update current data
            self.current_data = new_data
            self.historical_data.append(new_data)
            
            # Broadcast to all connected clients
            if self.connections:
                broadcast_tasks = []
                
                for connection_info in self.connections.values():
                    if 'all' in connection_info.subscriptions or 'monitoring' in connection_info.subscriptions:
                        filtered_data = self._filter_data_by_skill_level(new_data, connection_info.skill_level)
                        
                        message = {
                            'type': 'monitoring_update',
                            'data': filtered_data
                        }
                        
                        broadcast_tasks.append(
                            self._safe_send(connection_info.websocket, json.dumps(message))
                        )
                
                # Send all messages concurrently
                if broadcast_tasks:
                    await asyncio.gather(*broadcast_tasks, return_exceptions=True)
            
            # Update performance metrics
            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            self.performance_stats['average_response_time'] = (
                self.performance_stats['average_response_time'] * 0.9 + response_time * 0.1
            )
            self.performance_stats['messages_sent'] += len(self.connections)
            self.performance_stats['last_update_time'] = datetime.now().isoformat()
            
            self.last_broadcast = time.time()
            
        except Exception as e:
            logger.error(f"Error updating monitoring data: {e}")
    
    async def _safe_send(self, websocket: WebSocketServerProtocol, message: str):
        """Safely send message to WebSocket"""
        try:
            await websocket.send(message)
        except websockets.exceptions.ConnectionClosed:
            pass  # Connection already closed
        except Exception as e:
            logger.error(f"Error sending message: {e}")
    
    def add_alert(self, alert_type: AlertType, message: str, severity: str = "info", data: Optional[Dict] = None):
        """Add alert to the alerts queue"""
        try:
            alert = {
                'type': alert_type.value,
                'message': message,
                'severity': severity,
                'timestamp': datetime.now().isoformat(),
                'data': data or {}
            }
            
            self.alerts_queue.append(alert)
            
            # Broadcast alert to connected clients
            if self.running and self.connections:
                asyncio.create_task(self._broadcast_alert(alert))
            
        except Exception as e:
            logger.error(f"Error adding alert: {e}")
    
    async def _broadcast_alert(self, alert: Dict[str, Any]):
        """Broadcast alert to all connected clients"""
        try:
            message = {
                'type': 'alert',
                'alert': alert
            }
            
            broadcast_tasks = [
                self._safe_send(conn.websocket, json.dumps(message))
                for conn in self.connections.values()
                if 'all' in conn.subscriptions or 'alerts' in conn.subscriptions
            ]
            
            if broadcast_tasks:
                await asyncio.gather(*broadcast_tasks, return_exceptions=True)
                
        except Exception as e:
            logger.error(f"Error broadcasting alert: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        return self.performance_stats.copy()
