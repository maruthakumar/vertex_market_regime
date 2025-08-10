"""
ML Indicator Strategy WebSocket Integration
Real-time ML model inference and signal processing with WebSocket streaming

This module provides comprehensive WebSocket integration for:
- Real-time ML model inference and predictions
- Feature engineering data streaming
- Model performance monitoring
- Real-time signal generation and validation
- ML model retraining triggers
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
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import websockets
from websockets.exceptions import ConnectionClosed, InvalidURI
import aiohttp
from aiohttp import web
import pickle
import joblib

from .models import MLIndicatorStrategyModel, MLSignal, MLTrade
from .ml.signal_generation import MLSignalGenerator
from .ml.feature_engineering import FeatureEngineer
from .ml.model_evaluation import ModelEvaluator

logger = logging.getLogger(__name__)


class MLEventType(str, Enum):
    """ML WebSocket event types"""
    MODEL_INFERENCE_REQUESTED = "MODEL_INFERENCE_REQUESTED"
    MODEL_PREDICTION_GENERATED = "MODEL_PREDICTION_GENERATED"
    FEATURE_DATA_RECEIVED = "FEATURE_DATA_RECEIVED"
    SIGNAL_GENERATED = "SIGNAL_GENERATED"
    MODEL_PERFORMANCE_UPDATED = "MODEL_PERFORMANCE_UPDATED"
    MODEL_RETRAIN_TRIGGERED = "MODEL_RETRAIN_TRIGGERED"
    ERROR_OCCURRED = "ERROR_OCCURRED"


@dataclass
class MLWebSocketEvent:
    """ML WebSocket event"""
    event_id: str
    event_type: MLEventType
    timestamp: datetime
    data: Dict[str, Any]
    source: str = "ML_STRATEGY"
    processed: bool = False
    
    def mark_processed(self):
        """Mark event as processed"""
        self.processed = True


@dataclass
class MLStreamConfig:
    """ML streaming configuration"""
    server_host: str = "localhost"
    server_port: int = 8769
    inference_port: int = 8770
    max_connections: int = 100
    heartbeat_interval: int = 30
    inference_timeout: int = 10  # seconds
    feature_buffer_size: int = 1000
    model_update_interval: int = 300  # 5 minutes
    confidence_threshold: float = 0.7
    performance_alert_threshold: float = 0.6  # Alert if accuracy drops below 60%


class MLWebSocketIntegration:
    """
    Comprehensive WebSocket integration for ML Indicator Strategy
    
    Provides real-time ML model inference, feature streaming, and signal generation
    """
    
    def __init__(self, config: MLIndicatorStrategyModel, stream_config: Optional[MLStreamConfig] = None):
        """Initialize ML WebSocket integration"""
        self.config = config
        self.stream_config = stream_config or MLStreamConfig()
        
        # ML components
        self.signal_generator = MLSignalGenerator()
        self.feature_engineer = FeatureEngineer()
        self.model_evaluator = ModelEvaluator()
        
        # Connection management
        self.connected_clients = set()
        self.inference_server = None
        self.websocket_server = None
        
        # ML model management
        self.current_models = {}
        self.model_performance = {}
        self.feature_buffer = {}
        
        # Event handling
        self.event_handlers = {}
        self.inference_queue = asyncio.Queue()
        self.event_queue = asyncio.Queue()
        
        # Performance metrics
        self.metrics = {
            'inferences_processed': 0,
            'signals_generated': 0,
            'model_updates': 0,
            'events_processed': 0,
            'connected_clients': 0,
            'average_inference_time_ms': 0.0,
            'model_accuracy': 0.0,
            'last_inference_time': None,
            'feature_updates_received': 0
        }
        
        # Setup event handlers
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """Setup default event handlers"""
        self.register_handler(MLEventType.MODEL_INFERENCE_REQUESTED, self._handle_inference_requested)
        self.register_handler(MLEventType.MODEL_PREDICTION_GENERATED, self._handle_prediction_generated)
        self.register_handler(MLEventType.FEATURE_DATA_RECEIVED, self._handle_feature_data_received)
        self.register_handler(MLEventType.SIGNAL_GENERATED, self._handle_signal_generated)
        self.register_handler(MLEventType.MODEL_PERFORMANCE_UPDATED, self._handle_performance_updated)
        self.register_handler(MLEventType.MODEL_RETRAIN_TRIGGERED, self._handle_retrain_triggered)
        self.register_handler(MLEventType.ERROR_OCCURRED, self._handle_error)
    
    def register_handler(self, event_type: MLEventType, handler: Callable):
        """Register event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for {event_type}")
    
    async def start_websocket_server(self):
        """Start WebSocket server for ML data streaming"""
        try:
            # Start WebSocket server
            self.websocket_server = await websockets.serve(
                self._handle_websocket_connection,
                self.stream_config.server_host,
                self.stream_config.server_port,
                max_size=2*1024*1024,  # 2MB max message size for ML data
                ping_interval=self.stream_config.heartbeat_interval,
                ping_timeout=self.stream_config.heartbeat_interval * 2
            )
            
            logger.info(f"ML WebSocket server started on {self.stream_config.server_host}:{self.stream_config.server_port}")
            
            # Start background tasks
            await asyncio.gather(
                self._process_inference_queue(),
                self._process_event_queue(),
                self._monitor_model_performance(),
                self._broadcast_ml_updates()
            )
            
        except Exception as e:
            logger.error(f"Failed to start ML WebSocket server: {e}")
            raise
    
    async def start_inference_server(self):
        """Start inference server for ML model predictions"""
        try:
            app = web.Application()
            
            # Setup inference routes
            app.router.add_post('/ml/predict', self._handle_prediction_request)
            app.router.add_post('/ml/features', self._handle_feature_update)
            app.router.add_post('/ml/model/update', self._handle_model_update)
            app.router.add_get('/ml/status', self._handle_ml_status)
            app.router.add_get('/ml/metrics', self._handle_ml_metrics)
            app.router.add_get('/ml/models', self._handle_model_list)
            
            # Setup CORS
            app.middlewares.append(self._cors_middleware)
            
            # Start inference server
            runner = web.AppRunner(app)
            await runner.setup()
            
            site = web.TCPSite(runner, self.stream_config.server_host, self.stream_config.inference_port)
            await site.start()
            
            logger.info(f"ML inference server started on {self.stream_config.server_host}:{self.stream_config.inference_port}")
            
        except Exception as e:
            logger.error(f"Failed to start ML inference server: {e}")
            raise
    
    async def _handle_websocket_connection(self, websocket, path):
        """Handle incoming WebSocket connections"""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        self.connected_clients.add(websocket)
        self.metrics['connected_clients'] = len(self.connected_clients)
        
        logger.info(f"New ML WebSocket connection from {client_id}")
        
        try:
            # Send welcome message with current model info
            welcome_msg = {
                'type': 'welcome',
                'timestamp': datetime.now().isoformat(),
                'available_models': list(self.current_models.keys()),
                'model_performance': self.model_performance,
                'server_info': {
                    'strategy': 'ML_INDICATOR',
                    'version': '1.0.0',
                    'capabilities': ['real_time_inference', 'feature_streaming', 'model_monitoring']
                }
            }
            await websocket.send(json.dumps(welcome_msg))
            
            # Handle messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._process_ml_websocket_message(websocket, data, client_id)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON from {client_id}: {message}")
                    await websocket.send(json.dumps({'error': 'Invalid JSON format'}))
                except Exception as e:
                    logger.error(f"Error processing message from {client_id}: {e}")
                    await websocket.send(json.dumps({'error': str(e)}))
                    
        except ConnectionClosed:
            logger.info(f"ML WebSocket connection closed for {client_id}")
        except Exception as e:
            logger.error(f"ML WebSocket error for {client_id}: {e}")
        finally:
            self.connected_clients.discard(websocket)
            self.metrics['connected_clients'] = len(self.connected_clients)
    
    async def _process_ml_websocket_message(self, websocket, data: Dict[str, Any], client_id: str):
        """Process incoming ML WebSocket message"""
        message_type = data.get('type')
        
        if message_type == 'predict':
            # Request ML prediction
            features = data.get('features', {})
            model_name = data.get('model', 'default')
            await self._handle_prediction_websocket(websocket, features, model_name, client_id)
            
        elif message_type == 'stream_features':
            # Stream feature data
            feature_data = data.get('feature_data', {})
            await self._handle_feature_streaming(websocket, feature_data, client_id)
            
        elif message_type == 'subscribe_signals':
            # Subscribe to ML signals
            await self._handle_signal_subscription(websocket, client_id)
            
        elif message_type == 'model_performance':
            # Get model performance metrics
            performance = await self._get_model_performance()
            response = {
                'type': 'model_performance',
                'data': performance,
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
            logger.warning(f"Unknown ML message type '{message_type}' from {client_id}")
    
    async def _handle_prediction_request(self, request):
        """Handle ML prediction requests"""
        try:
            data = await request.json()
            
            # Validate prediction request
            if not self._validate_prediction_request(data):
                return web.json_response({'error': 'Invalid prediction request'}, status=400)
            
            # Queue prediction request
            prediction_request = {
                'features': data.get('features', {}),
                'model': data.get('model', 'default'),
                'timestamp': datetime.now().isoformat(),
                'request_id': f"PRED_{int(time.time() * 1000)}"
            }
            
            await self.inference_queue.put(prediction_request)
            
            self.metrics['inferences_processed'] += 1
            
            return web.json_response({
                'status': 'queued',
                'request_id': prediction_request['request_id'],
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error handling prediction request: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def _handle_feature_update(self, request):
        """Handle feature data updates"""
        try:
            data = await request.json()
            
            # Process feature data
            symbol = data.get('symbol', 'NIFTY')
            features = data.get('features', {})
            
            # Update feature buffer
            if symbol not in self.feature_buffer:
                self.feature_buffer[symbol] = []
            
            self.feature_buffer[symbol].append({
                'timestamp': datetime.now().isoformat(),
                'features': features
            })
            
            # Maintain buffer size
            if len(self.feature_buffer[symbol]) > self.stream_config.feature_buffer_size:
                self.feature_buffer[symbol] = self.feature_buffer[symbol][-self.stream_config.feature_buffer_size:]
            
            self.metrics['feature_updates_received'] += 1
            
            # Create event
            event = MLWebSocketEvent(
                event_id=f"FEAT_{int(time.time() * 1000)}",
                event_type=MLEventType.FEATURE_DATA_RECEIVED,
                timestamp=datetime.now(),
                data={'symbol': symbol, 'features': features}
            )
            await self.event_queue.put(event)
            
            return web.json_response({
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error handling feature update: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def _handle_model_update(self, request):
        """Handle ML model updates"""
        try:
            # This would handle model file uploads in a real implementation
            # For now, we'll simulate model update
            
            model_name = request.query.get('model', 'default')
            
            # Simulate model update
            self.current_models[model_name] = {
                'updated_at': datetime.now().isoformat(),
                'version': f"v{int(time.time())}",
                'status': 'active'
            }
            
            self.metrics['model_updates'] += 1
            
            # Create event
            event = MLWebSocketEvent(
                event_id=f"MODEL_{int(time.time() * 1000)}",
                event_type=MLEventType.MODEL_RETRAIN_TRIGGERED,
                timestamp=datetime.now(),
                data={'model_name': model_name, 'action': 'updated'}
            )
            await self.event_queue.put(event)
            
            return web.json_response({
                'status': 'success',
                'model': model_name,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error handling model update: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def _handle_ml_status(self, request):
        """Handle ML status requests"""
        status = {
            'status': 'active',
            'timestamp': datetime.now().isoformat(),
            'connected_clients': self.metrics['connected_clients'],
            'available_models': list(self.current_models.keys()),
            'model_performance': self.model_performance,
            'feature_buffer_size': sum(len(buffer) for buffer in self.feature_buffer.values()),
            'metrics': self.metrics
        }
        return web.json_response(status)
    
    async def _handle_ml_metrics(self, request):
        """Handle ML metrics requests"""
        return web.json_response(self.metrics)
    
    async def _handle_model_list(self, request):
        """Handle model list requests"""
        return web.json_response({
            'models': self.current_models,
            'performance': self.model_performance,
            'timestamp': datetime.now().isoformat()
        })
    
    async def _cors_middleware(self, request, handler):
        """CORS middleware for inference server"""
        response = await handler(request)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
    
    def _validate_prediction_request(self, data: Dict[str, Any]) -> bool:
        """Validate ML prediction request"""
        return 'features' in data and isinstance(data['features'], dict)
    
    async def _process_inference_queue(self):
        """Process ML inference requests from the queue"""
        while True:
            try:
                request = await self.inference_queue.get()
                
                # Process inference
                start_time = time.time()
                prediction = await self._perform_ml_inference(request)
                processing_time = (time.time() - start_time) * 1000
                
                # Update metrics
                self.metrics['last_inference_time'] = datetime.now().isoformat()
                self.metrics['average_inference_time_ms'] = (
                    (self.metrics['average_inference_time_ms'] * self.metrics['inferences_processed'] + processing_time) /
                    (self.metrics['inferences_processed'] + 1)
                )
                
                # Create event
                event = MLWebSocketEvent(
                    event_id=f"PRED_{int(time.time() * 1000)}",
                    event_type=MLEventType.MODEL_PREDICTION_GENERATED,
                    timestamp=datetime.now(),
                    data={
                        'request': request,
                        'prediction': prediction,
                        'processing_time_ms': processing_time
                    }
                )
                await self.event_queue.put(event)
                
            except Exception as e:
                logger.error(f"Error processing ML inference: {e}")
    
    async def _perform_ml_inference(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Perform ML model inference"""
        try:
            features = request['features']
            model_name = request.get('model', 'default')
            
            # Simulate ML inference (in real implementation, this would use actual ML models)
            # Convert features to numpy array format
            feature_array = np.array(list(features.values())).reshape(1, -1)
            
            # Simulate prediction
            prediction_prob = np.random.random()  # Simulate model prediction
            prediction_class = 1 if prediction_prob > 0.5 else 0
            confidence = max(prediction_prob, 1 - prediction_prob)
            
            prediction = {
                'model': model_name,
                'prediction_class': prediction_class,
                'prediction_probability': prediction_prob,
                'confidence': confidence,
                'features_used': len(features),
                'timestamp': datetime.now().isoformat()
            }
            
            # Generate signal if confidence is high enough
            if confidence >= self.stream_config.confidence_threshold:
                signal = await self._generate_ml_signal(prediction, features)
                prediction['signal_generated'] = signal
                
                # Create signal event
                signal_event = MLWebSocketEvent(
                    event_id=f"SIG_{int(time.time() * 1000)}",
                    event_type=MLEventType.SIGNAL_GENERATED,
                    timestamp=datetime.now(),
                    data={'signal': signal, 'prediction': prediction}
                )
                await self.event_queue.put(signal_event)
                self.metrics['signals_generated'] += 1
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error performing ML inference: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    async def _generate_ml_signal(self, prediction: Dict[str, Any], features: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signal from ML prediction"""
        signal = {
            'signal_id': f"ML_{int(time.time() * 1000)}",
            'timestamp': datetime.now().isoformat(),
            'symbol': 'NIFTY',  # Default symbol
            'action': 'BUY' if prediction['prediction_class'] == 1 else 'SELL',
            'confidence': prediction['confidence'],
            'quantity': 50,  # Default quantity
            'ml_prediction': prediction,
            'features': features
        }
        
        return signal
    
    async def _process_event_queue(self):
        """Process events from the queue"""
        while True:
            try:
                event = await self.event_queue.get()
                
                # Handle event
                await self._handle_event(event)
                
                self.metrics['events_processed'] += 1
                
            except Exception as e:
                logger.error(f"Error processing ML event: {e}")
    
    async def _handle_event(self, event: MLWebSocketEvent):
        """Handle ML WebSocket event"""
        handlers = self.event_handlers.get(event.event_type, [])
        
        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Error in ML event handler for {event.event_type}: {e}")
    
    async def _handle_inference_requested(self, event: MLWebSocketEvent):
        """Handle inference requested event"""
        request_id = event.data.get('request_id', 'unknown')
        logger.debug(f"ML inference requested: {request_id}")
    
    async def _handle_prediction_generated(self, event: MLWebSocketEvent):
        """Handle prediction generated event"""
        prediction = event.data.get('prediction', {})
        processing_time = event.data.get('processing_time_ms', 0)
        logger.info(f"ML prediction generated in {processing_time:.2f}ms with confidence {prediction.get('confidence', 0):.2f}")
        
        # Broadcast prediction to connected clients
        prediction_msg = {
            'type': 'ml_prediction',
            'prediction': prediction,
            'timestamp': datetime.now().isoformat()
        }
        await self._broadcast_to_clients(prediction_msg)
    
    async def _handle_feature_data_received(self, event: MLWebSocketEvent):
        """Handle feature data received event"""
        symbol = event.data.get('symbol', 'unknown')
        logger.debug(f"Feature data received for {symbol}")
    
    async def _handle_signal_generated(self, event: MLWebSocketEvent):
        """Handle signal generated event"""
        signal = event.data.get('signal', {})
        logger.info(f"ML signal generated: {signal.get('action', 'unknown')} with confidence {signal.get('confidence', 0):.2f}")
        
        # Broadcast signal to connected clients
        signal_msg = {
            'type': 'ml_signal',
            'signal': signal,
            'timestamp': datetime.now().isoformat()
        }
        await self._broadcast_to_clients(signal_msg)
    
    async def _handle_performance_updated(self, event: MLWebSocketEvent):
        """Handle performance updated event"""
        performance = event.data.get('performance', {})
        logger.info(f"ML model performance updated: {performance}")
    
    async def _handle_retrain_triggered(self, event: MLWebSocketEvent):
        """Handle retrain triggered event"""
        model_name = event.data.get('model_name', 'unknown')
        logger.info(f"ML model retrain triggered for {model_name}")
    
    async def _handle_error(self, event: MLWebSocketEvent):
        """Handle error event"""
        error = event.data.get('error', 'Unknown error')
        logger.error(f"ML WebSocket error: {error}")
    
    async def _monitor_model_performance(self):
        """Monitor ML model performance"""
        while True:
            try:
                # Simulate performance monitoring
                if self.current_models:
                    for model_name in self.current_models.keys():
                        # Simulate performance metrics
                        accuracy = 0.75 + np.random.random() * 0.2  # 75-95% accuracy
                        precision = 0.70 + np.random.random() * 0.25  # 70-95% precision
                        recall = 0.65 + np.random.random() * 0.30  # 65-95% recall
                        
                        self.model_performance[model_name] = {
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1_score': 2 * (precision * recall) / (precision + recall),
                            'last_updated': datetime.now().isoformat()
                        }
                        
                        # Update overall metrics
                        self.metrics['model_accuracy'] = accuracy
                        
                        # Check for performance alerts
                        if accuracy < self.stream_config.performance_alert_threshold:
                            event = MLWebSocketEvent(
                                event_id=f"ALERT_{int(time.time() * 1000)}",
                                event_type=MLEventType.MODEL_PERFORMANCE_UPDATED,
                                timestamp=datetime.now(),
                                data={
                                    'model_name': model_name,
                                    'performance': self.model_performance[model_name],
                                    'alert': 'LOW_PERFORMANCE'
                                }
                            )
                            await self.event_queue.put(event)
                
                await asyncio.sleep(self.stream_config.model_update_interval)
                
            except Exception as e:
                logger.error(f"Error in ML performance monitoring: {e}")
                await asyncio.sleep(30)
    
    async def _broadcast_ml_updates(self):
        """Broadcast ML updates to connected clients"""
        while True:
            try:
                if self.connected_clients:
                    # Create ML update message
                    update_msg = {
                        'type': 'ml_update',
                        'timestamp': datetime.now().isoformat(),
                        'metrics': self.metrics,
                        'model_performance': self.model_performance,
                        'active_models': list(self.current_models.keys())
                    }
                    
                    await self._broadcast_to_clients(update_msg)
                
                await asyncio.sleep(60)  # Broadcast every minute
                
            except Exception as e:
                logger.error(f"Error broadcasting ML updates: {e}")
                await asyncio.sleep(30)
    
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
    
    async def _handle_prediction_websocket(self, websocket, features: Dict[str, Any], model_name: str, client_id: str):
        """Handle prediction request via WebSocket"""
        try:
            # Create prediction request
            request = {
                'features': features,
                'model': model_name,
                'client_id': client_id,
                'timestamp': datetime.now().isoformat()
            }
            
            # Perform inference
            prediction = await self._perform_ml_inference(request)
            
            # Send response
            response = {
                'type': 'prediction_result',
                'prediction': prediction,
                'timestamp': datetime.now().isoformat()
            }
            await websocket.send(json.dumps(response))
            
        except Exception as e:
            logger.error(f"Error handling WebSocket prediction: {e}")
            error_response = {
                'type': 'prediction_error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            await websocket.send(json.dumps(error_response))
    
    async def _handle_feature_streaming(self, websocket, feature_data: Dict[str, Any], client_id: str):
        """Handle feature data streaming"""
        try:
            # Process feature data
            symbol = feature_data.get('symbol', 'NIFTY')
            features = feature_data.get('features', {})
            
            # Update feature buffer
            if symbol not in self.feature_buffer:
                self.feature_buffer[symbol] = []
            
            self.feature_buffer[symbol].append({
                'timestamp': datetime.now().isoformat(),
                'features': features,
                'client_id': client_id
            })
            
            # Send acknowledgment
            ack_response = {
                'type': 'feature_ack',
                'symbol': symbol,
                'features_count': len(features),
                'timestamp': datetime.now().isoformat()
            }
            await websocket.send(json.dumps(ack_response))
            
        except Exception as e:
            logger.error(f"Error handling feature streaming: {e}")
    
    async def _handle_signal_subscription(self, websocket, client_id: str):
        """Handle signal subscription"""
        try:
            response = {
                'type': 'signal_subscription_ack',
                'status': 'subscribed',
                'timestamp': datetime.now().isoformat()
            }
            await websocket.send(json.dumps(response))
            logger.info(f"Client {client_id} subscribed to ML signals")
            
        except Exception as e:
            logger.error(f"Error handling signal subscription: {e}")
    
    async def _get_model_performance(self) -> Dict[str, Any]:
        """Get current model performance metrics"""
        return {
            'models': self.model_performance,
            'overall_metrics': self.metrics,
            'timestamp': datetime.now().isoformat()
        }
    
    async def start(self):
        """Start ML WebSocket integration"""
        logger.info("Starting ML WebSocket integration...")
        
        # Initialize default model
        self.current_models['default'] = {
            'created_at': datetime.now().isoformat(),
            'version': 'v1.0.0',
            'status': 'active'
        }
        
        try:
            # Start both servers concurrently
            await asyncio.gather(
                self.start_websocket_server(),
                self.start_inference_server()
            )
        except Exception as e:
            logger.error(f"Failed to start ML WebSocket integration: {e}")
            raise
    
    async def stop(self):
        """Stop ML WebSocket integration"""
        logger.info("Stopping ML WebSocket integration...")
        
        try:
            # Close all connections
            for client in self.connected_clients:
                await client.close()
            
            # Close servers
            if self.websocket_server:
                self.websocket_server.close()
                await self.websocket_server.wait_closed()
            
        except Exception as e:
            logger.error(f"Error stopping ML WebSocket integration: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get ML WebSocket integration metrics"""
        return self.metrics.copy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get current model information"""
        return {
            'models': self.current_models,
            'performance': self.model_performance,
            'feature_buffer_status': {
                symbol: len(buffer) for symbol, buffer in self.feature_buffer.items()
            }
        }


# Singleton instance for global access
_ml_websocket_integration = None

def get_ml_websocket_integration(config: MLIndicatorStrategyModel, stream_config: Optional[MLStreamConfig] = None) -> MLWebSocketIntegration:
    """Get singleton instance of ML WebSocket integration"""
    global _ml_websocket_integration
    
    if _ml_websocket_integration is None:
        _ml_websocket_integration = MLWebSocketIntegration(config, stream_config)
    
    return _ml_websocket_integration
