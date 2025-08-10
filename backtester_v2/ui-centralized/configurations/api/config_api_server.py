"""
Configuration API Server

FastAPI server that exposes the configuration management system
through HTTP endpoints for Next.js integration.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path

# Import the config API bridge
from config_api_bridge import ConfigAPIBridge, execute_operation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Configuration Management API",
    description="API for managing strategy configurations",
    version="2.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize API bridge
bridge = ConfigAPIBridge()

# WebSocket manager for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.subscriptions: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        # Remove from subscriptions
        for key in list(self.subscriptions.keys()):
            if websocket in self.subscriptions[key]:
                self.subscriptions[key].remove(websocket)
                if not self.subscriptions[key]:
                    del self.subscriptions[key]
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        message_str = json.dumps(message)
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_str)
            except:
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

    async def broadcast_to_subscribers(self, key: str, message: dict):
        """Broadcast to specific subscribers"""
        if key in self.subscriptions:
            message_str = json.dumps(message)
            disconnected = []
            
            for connection in self.subscriptions[key]:
                try:
                    await connection.send_text(message_str)
                except:
                    disconnected.append(connection)
            
            # Clean up disconnected clients
            for conn in disconnected:
                if conn in self.subscriptions[key]:
                    self.subscriptions[key].remove(conn)

    def subscribe(self, websocket: WebSocket, strategy_type: str, config_name: str):
        """Subscribe to specific configuration updates"""
        key = f"{strategy_type}:{config_name}"
        if key not in self.subscriptions:
            self.subscriptions[key] = []
        if websocket not in self.subscriptions[key]:
            self.subscriptions[key].append(websocket)
        logger.info(f"WebSocket subscribed to {key}")

    def unsubscribe(self, websocket: WebSocket, strategy_type: str, config_name: str):
        """Unsubscribe from configuration updates"""
        key = f"{strategy_type}:{config_name}"
        if key in self.subscriptions and websocket in self.subscriptions[key]:
            self.subscriptions[key].remove(websocket)
            if not self.subscriptions[key]:
                del self.subscriptions[key]
        logger.info(f"WebSocket unsubscribed from {key}")

manager = ConnectionManager()

# Request/Response models
class ConfigOperation(BaseModel):
    action: str
    strategy_type: Optional[str] = None
    config_name: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    options: Optional[Dict[str, Any]] = None

class ConfigResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    validation_errors: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Configuration operations endpoint
@app.post("/api/configurations/execute", response_model=ConfigResponse)
async def execute_configuration_operation(operation: ConfigOperation):
    """Execute a configuration operation"""
    try:
        result = execute_operation(operation.dict())
        
        # Broadcast updates via WebSocket
        if operation.action in ['create', 'update', 'delete'] and result.get('success'):
            await manager.broadcast({
                'type': f'config_{operation.action}d',
                'strategy_type': operation.strategy_type,
                'config_name': operation.config_name,
                'data': result.get('data'),
                'timestamp': datetime.now().isoformat()
            })
            
            # Also broadcast to specific subscribers
            if operation.strategy_type and operation.config_name:
                key = f"{operation.strategy_type}:{operation.config_name}"
                await manager.broadcast_to_subscribers(key, {
                    'type': f'config_{operation.action}d',
                    'data': result.get('data'),
                    'timestamp': datetime.now().isoformat()
                })
        
        return ConfigResponse(**result)
    except Exception as e:
        logger.error(f"Operation execution error: {e}")
        return ConfigResponse(success=False, error=str(e))

# List configurations
@app.get("/api/configurations")
async def list_configurations(
    strategy_type: Optional[str] = None,
    include_metadata: bool = False
):
    """List all configurations"""
    operation = {
        'action': 'list',
        'strategy_type': strategy_type,
        'options': {'include_metadata': include_metadata}
    }
    result = execute_operation(operation)
    return result

# Get specific configuration
@app.get("/api/configurations/{strategy_type}/{config_name}")
async def get_configuration(strategy_type: str, config_name: str):
    """Get a specific configuration"""
    operation = {
        'action': 'get',
        'strategy_type': strategy_type,
        'config_name': config_name
    }
    result = execute_operation(operation)
    if not result.get('success'):
        raise HTTPException(status_code=404, detail=result.get('error'))
    return result

# Create configuration
@app.post("/api/configurations/{strategy_type}/{config_name}")
async def create_configuration(
    strategy_type: str,
    config_name: str,
    data: Dict[str, Any]
):
    """Create a new configuration"""
    operation = {
        'action': 'create',
        'strategy_type': strategy_type,
        'config_name': config_name,
        'data': data
    }
    result = execute_operation(operation)
    if not result.get('success'):
        raise HTTPException(status_code=400, detail=result.get('error'))
    return result

# Update configuration
@app.put("/api/configurations/{strategy_type}/{config_name}")
async def update_configuration(
    strategy_type: str,
    config_name: str,
    data: Dict[str, Any],
    merge_strategy: str = 'override'
):
    """Update an existing configuration"""
    operation = {
        'action': 'update',
        'strategy_type': strategy_type,
        'config_name': config_name,
        'data': data,
        'options': {'merge_strategy': merge_strategy}
    }
    result = execute_operation(operation)
    if not result.get('success'):
        raise HTTPException(status_code=400, detail=result.get('error'))
    return result

# Delete configuration
@app.delete("/api/configurations/{strategy_type}/{config_name}")
async def delete_configuration(
    strategy_type: str,
    config_name: str,
    force: bool = False
):
    """Delete a configuration"""
    operation = {
        'action': 'delete',
        'strategy_type': strategy_type,
        'config_name': config_name,
        'options': {'force': force}
    }
    result = execute_operation(operation)
    if not result.get('success'):
        raise HTTPException(status_code=400, detail=result.get('error'))
    return result

# Validate configuration
@app.post("/api/configurations/validate")
async def validate_configuration(
    strategy_type: str,
    data: Dict[str, Any],
    config_name: Optional[str] = None
):
    """Validate configuration data"""
    operation = {
        'action': 'validate',
        'strategy_type': strategy_type,
        'config_name': config_name,
        'data': data
    }
    result = execute_operation(operation)
    return result

# Clone configuration
@app.post("/api/configurations/{strategy_type}/{config_name}/clone")
async def clone_configuration(
    strategy_type: str,
    config_name: str,
    new_name: str
):
    """Clone a configuration"""
    operation = {
        'action': 'clone',
        'strategy_type': strategy_type,
        'config_name': config_name,
        'options': {'new_name': new_name}
    }
    result = execute_operation(operation)
    if not result.get('success'):
        raise HTTPException(status_code=400, detail=result.get('error'))
    return result

# Get configuration statistics
@app.get("/api/configurations/statistics")
async def get_statistics():
    """Get configuration statistics"""
    try:
        stats = bridge.manager.get_statistics()
        stats['websocket_connections'] = len(manager.active_connections)
        stats['websocket_subscriptions'] = len(manager.subscriptions)
        return stats
    except Exception as e:
        logger.error(f"Statistics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Export configuration to Excel
@app.get("/api/configurations/export/{strategy_type}/{config_name}")
async def export_configuration(strategy_type: str, config_name: str):
    """Export configuration to Excel"""
    try:
        # Get configuration
        config = bridge.manager.get_configuration(strategy_type, config_name)
        if not config:
            raise HTTPException(status_code=404, detail="Configuration not found")
        
        # Export to Excel (implementation needed in gateway)
        # For now, return JSON
        return JSONResponse(content=config.to_dict())
    except Exception as e:
        logger.error(f"Export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get configuration schema
@app.get("/api/configurations/schema/{strategy_type}")
async def get_schema(strategy_type: str):
    """Get configuration schema for a strategy type"""
    try:
        config_class = bridge.manager._registry.get_class(strategy_type)
        if not config_class:
            raise HTTPException(status_code=404, detail=f"No schema for {strategy_type}")
        
        # Get schema from configuration class
        schema = {}
        if hasattr(config_class, 'get_schema'):
            schema = config_class.get_schema()
        else:
            # Basic schema extraction
            instance = config_class('temp')
            schema = {
                'properties': instance.to_dict(),
                'required': getattr(instance, 'required_fields', []),
                'strategy_type': strategy_type
            }
        
        return schema
    except Exception as e:
        logger.error(f"Schema error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Search configurations
@app.post("/api/configurations/search")
async def search_configurations(
    query: str,
    strategy_type: Optional[str] = None,
    limit: int = 10
):
    """Search configurations"""
    try:
        results = []
        configs = bridge.manager.list_configurations(strategy_type)
        
        # Simple search implementation
        for config_info in configs:
            if query.lower() in config_info['name'].lower():
                config = bridge.manager.get_configuration(
                    config_info['strategy_type'],
                    config_info['name']
                )
                if config:
                    results.append({
                        'strategy_type': config_info['strategy_type'],
                        'name': config_info['name'],
                        'data': config.to_dict()
                    })
                    
                    if len(results) >= limit:
                        break
        
        return {
            'success': True,
            'results': results,
            'total': len(results)
        }
    except Exception as e:
        logger.error(f"Search error: {e}")
        return {
            'success': False,
            'error': str(e)
        }

# Batch operations
@app.post("/api/configurations/batch")
async def batch_operations(operations: List[ConfigOperation]):
    """Execute multiple operations in batch"""
    results = []
    
    for operation in operations:
        try:
            result = execute_operation(operation.dict())
            results.append(result)
        except Exception as e:
            results.append({
                'success': False,
                'error': str(e)
            })
    
    return {'results': results}

# WebSocket endpoint for real-time updates
@app.websocket("/ws/configurations")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            # Handle different message types
            if data.get('type') == 'ping':
                await websocket.send_json({'type': 'pong'})
            
            elif data.get('type') == 'subscribe':
                strategy_type = data.get('strategy_type')
                config_name = data.get('config_name')
                if strategy_type and config_name:
                    manager.subscribe(websocket, strategy_type, config_name)
            
            elif data.get('type') == 'unsubscribe':
                strategy_type = data.get('strategy_type')
                config_name = data.get('config_name')
                if strategy_type and config_name:
                    manager.unsubscribe(websocket, strategy_type, config_name)
            
            elif data.get('type') == 'subscribe_all':
                # Subscribe to all updates (add to active connections already done)
                await websocket.send_json({
                    'type': 'subscribed',
                    'message': 'Subscribed to all configuration updates'
                })
            
            elif data.get('type') == 'broadcast':
                # Allow broadcasting from clients (with authentication in production)
                await manager.broadcast(data)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# File watcher integration for hot reload
async def start_file_watcher():
    """Start file watcher for configuration changes"""
    # This would integrate with the Python file watcher
    # For now, it's a placeholder
    logger.info("File watcher integration ready")

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Configuration API Server starting...")
    await start_file_watcher()
    logger.info("Configuration API Server started")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Configuration API Server shutting down...")
    # Close all WebSocket connections
    for connection in manager.active_connections:
        await connection.close()
    logger.info("Configuration API Server stopped")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info",
        reload=True
    )