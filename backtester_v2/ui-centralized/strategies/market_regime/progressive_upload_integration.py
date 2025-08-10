#!/usr/bin/env python3
"""
Progressive Upload Integration for Market Regime
===============================================

This module integrates the progressive upload system with the existing
market regime configuration API.
"""

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends
from progressive_upload_system import ProgressiveUploadAPI, create_progressive_upload_system
from excel_config_manager import MarketRegimeExcelParser
from api_integration import MarketRegimeAPI

logger = logging.getLogger(__name__)


class EnhancedMarketRegimeAPI(MarketRegimeAPI):
    """Enhanced Market Regime API with progressive upload support"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize progressive upload system
        self.progressive_upload = create_progressive_upload_system(
            excel_parser=self.excel_parser
        )
        
        # Add progressive upload routes
        self._setup_progressive_routes()
    
    def _setup_progressive_routes(self):
        """Setup progressive upload routes"""
        
        # Create a sub-router for progressive upload
        progressive_router = APIRouter(prefix="/progressive")
        
        # Add routes from progressive upload system
        self.progressive_upload.create_routes(progressive_router)
        
        # Add custom integration routes
        self._add_integration_routes(progressive_router)
        
        # Include the progressive router in main router
        self.router.include_router(progressive_router)
    
    def _add_integration_routes(self, router: APIRouter):
        """Add custom integration routes"""
        
        @router.post("/apply/{session_id}")
        async def apply_uploaded_config(session_id: str):
            """Apply configuration from completed upload session"""
            try:
                # Get upload session
                if session_id not in self.progressive_upload.tracker.sessions:
                    raise HTTPException(status_code=404, detail="Session not found")
                
                session = self.progressive_upload.tracker.sessions[session_id]
                
                # Check if upload is complete
                if session.status != UploadStatus.COMPLETED:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Upload not completed. Current status: {session.status}"
                    )
                
                # Apply configuration
                if session.final_path and session.final_path.exists():
                    config = self.excel_parser.parse_excel_config(str(session.final_path))
                    
                    # Store in active configurations
                    config_id = f"progressive_{session_id}"
                    self.active_configs[config_id] = config
                    
                    return JSONResponse(content={
                        "status": "success",
                        "message": "Configuration applied successfully",
                        "config_id": config_id,
                        "config_summary": {
                            "regime_mode": config.regime_mode,
                            "indicators_count": len(config.indicators),
                            "enabled_indicators": [
                                name for name, ind in config.indicators.items() 
                                if ind.enabled
                            ]
                        }
                    })
                else:
                    raise HTTPException(
                        status_code=404, 
                        detail="Configuration file not found"
                    )
                    
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to apply uploaded config: {e}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to apply configuration: {str(e)}"
                )
        
        @router.get("/sessions")
        async def list_upload_sessions():
            """List all upload sessions with their status"""
            try:
                sessions_list = []
                
                for session_id, session in self.progressive_upload.tracker.sessions.items():
                    sessions_list.append({
                        "session_id": session_id,
                        "filename": session.filename,
                        "status": session.status.value,
                        "progress": len(session.uploaded_chunks) / session.total_chunks * 100,
                        "start_time": session.start_time.isoformat(),
                        "last_activity": session.last_activity.isoformat(),
                        "validation_complete": bool(session.validation_results)
                    })
                
                return JSONResponse(content={
                    "status": "success",
                    "total_sessions": len(sessions_list),
                    "sessions": sessions_list
                })
                
            except Exception as e:
                logger.error(f"Failed to list sessions: {e}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to list sessions: {str(e)}"
                )
        
        @router.delete("/session/{session_id}")
        async def delete_upload_session(session_id: str):
            """Delete an upload session and its files"""
            try:
                if session_id not in self.progressive_upload.tracker.sessions:
                    raise HTTPException(status_code=404, detail="Session not found")
                
                session = self.progressive_upload.tracker.sessions[session_id]
                
                # Clean up files
                if session.temp_path and session.temp_path.exists():
                    session.temp_path.unlink()
                
                if session.final_path and session.final_path.exists():
                    session.final_path.unlink()
                
                # Remove session
                del self.progressive_upload.tracker.sessions[session_id]
                
                # Close WebSocket if exists
                if session_id in self.progressive_upload.tracker.websocket_connections:
                    await self.progressive_upload.tracker.websocket_connections[session_id].close()
                    del self.progressive_upload.tracker.websocket_connections[session_id]
                
                return JSONResponse(content={
                    "status": "success",
                    "message": "Session deleted successfully"
                })
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to delete session: {e}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to delete session: {str(e)}"
                )


def create_enhanced_market_regime_api(
    regime_engine=None,
    upload_dir: Optional[str] = None
) -> EnhancedMarketRegimeAPI:
    """Create enhanced market regime API with progressive upload"""
    
    return EnhancedMarketRegimeAPI(
        regime_engine=regime_engine,
        upload_dir=upload_dir or "/srv/samba/shared/bt/backtester_stable/uploads"
    )


# Example usage
if __name__ == "__main__":
    from fastapi import FastAPI
    import uvicorn
    
    app = FastAPI(title="Enhanced Market Regime API")
    
    # Create enhanced API
    api = create_enhanced_market_regime_api()
    
    # Include routes
    app.include_router(api.router, prefix="/api/v2/market_regime")
    
    # Add static files for demo
    from fastapi.staticfiles import StaticFiles
    app.mount("/static", StaticFiles(directory="static"), name="static")
    
    @app.get("/")
    async def root():
        """Redirect to demo page"""
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/static/progressive_upload_demo.html")
    
    # Run server
    print("🚀 Enhanced Market Regime API with Progressive Upload")
    print("📍 Demo: http://localhost:8000/")
    print("📍 API: http://localhost:8000/api/v2/market_regime/progressive/")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)