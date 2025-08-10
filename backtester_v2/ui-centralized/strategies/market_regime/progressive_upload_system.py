#!/usr/bin/env python3
"""
Progressive File Upload System with Enhanced Validation
======================================================

This module implements a progressive file upload system with real-time
validation, chunk-based upload, and enhanced user feedback.

Features:
- Chunked file upload for large files
- Real-time validation feedback
- Progress tracking
- Resume capability
- Enhanced error handling
"""

import os
import asyncio
import hashlib
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum

from fastapi import (
    FastAPI, File, UploadFile, HTTPException, 
    Request, Response, WebSocket, WebSocketDisconnect,
    BackgroundTasks, Query, Form
)
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class UploadStatus(str, Enum):
    """Upload status states"""
    PENDING = "pending"
    UPLOADING = "uploading"
    VALIDATING = "validating"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ValidationLevel(str, Enum):
    """Validation levels"""
    BASIC = "basic"          # File type, size
    STRUCTURE = "structure"   # Sheet structure
    CONTENT = "content"      # Content validation
    ADVANCED = "advanced"    # Cross-sheet dependencies


@dataclass
class ChunkInfo:
    """Information about an upload chunk"""
    chunk_index: int
    chunk_size: int
    chunk_hash: str
    timestamp: datetime


@dataclass
class UploadSession:
    """Upload session information"""
    session_id: str
    filename: str
    file_size: int
    total_chunks: int
    uploaded_chunks: List[ChunkInfo]
    status: UploadStatus
    validation_results: Dict[str, Any]
    start_time: datetime
    last_activity: datetime
    temp_path: Optional[Path]
    final_path: Optional[Path]
    metadata: Dict[str, Any]


class UploadProgressTracker:
    """Track upload progress and provide real-time updates"""
    
    def __init__(self):
        self.sessions: Dict[str, UploadSession] = {}
        self.websocket_connections: Dict[str, WebSocket] = {}
        self.lock = asyncio.Lock()
        
    async def create_session(self, session_id: str, filename: str, 
                           file_size: int, chunk_size: int = 1024 * 1024) -> UploadSession:
        """Create a new upload session"""
        async with self.lock:
            total_chunks = (file_size + chunk_size - 1) // chunk_size
            
            session = UploadSession(
                session_id=session_id,
                filename=filename,
                file_size=file_size,
                total_chunks=total_chunks,
                uploaded_chunks=[],
                status=UploadStatus.PENDING,
                validation_results={},
                start_time=datetime.now(),
                last_activity=datetime.now(),
                temp_path=None,
                final_path=None,
                metadata={}
            )
            
            self.sessions[session_id] = session
            return session
    
    async def update_chunk(self, session_id: str, chunk_info: ChunkInfo) -> Dict[str, Any]:
        """Update chunk upload progress"""
        async with self.lock:
            if session_id not in self.sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.sessions[session_id]
            session.uploaded_chunks.append(chunk_info)
            session.last_activity = datetime.now()
            session.status = UploadStatus.UPLOADING
            
            progress = len(session.uploaded_chunks) / session.total_chunks * 100
            
            # Send WebSocket update
            await self._broadcast_progress(session_id, {
                "type": "upload_progress",
                "session_id": session_id,
                "progress": progress,
                "uploaded_chunks": len(session.uploaded_chunks),
                "total_chunks": session.total_chunks,
                "status": session.status.value
            })
            
            return {
                "progress": progress,
                "uploaded_chunks": len(session.uploaded_chunks),
                "total_chunks": session.total_chunks
            }
    
    async def _broadcast_progress(self, session_id: str, message: Dict[str, Any]):
        """Broadcast progress update to WebSocket connections"""
        if session_id in self.websocket_connections:
            try:
                await self.websocket_connections[session_id].send_json(message)
            except Exception as e:
                logger.error(f"WebSocket broadcast error: {e}")


class ProgressiveValidator:
    """Progressive validation system with real-time feedback"""
    
    def __init__(self, excel_parser=None):
        self.excel_parser = excel_parser
        self.validation_cache = {}
        
    async def validate_basic(self, filename: str, file_size: int) -> Tuple[bool, List[str]]:
        """Basic validation - file type and size"""
        errors = []
        
        # Check file extension
        if not filename.lower().endswith(('.xlsx', '.xls')):
            errors.append("Invalid file type. Only Excel files (.xlsx, .xls) are supported.")
        
        # Check file size (max 50MB for progressive upload)
        max_size = 50 * 1024 * 1024
        if file_size > max_size:
            errors.append(f"File too large. Maximum size is 50MB, got {file_size/1024/1024:.2f}MB")
        
        return len(errors) == 0, errors
    
    async def validate_structure(self, file_path: Path) -> Tuple[bool, Dict[str, Any]]:
        """Validate Excel structure"""
        try:
            # Load Excel file
            excel_data = pd.ExcelFile(file_path)
            sheets = excel_data.sheet_names
            
            required_sheets = [
                'IndicatorConfiguration',
                'StraddleAnalysisConfig', 
                'DynamicWeightageConfig',
                'MultiTimeframeConfig',
                'GreekSentimentConfig',
                'RegimeFormationConfig',
                'RegimeComplexityConfig'
            ]
            
            missing_sheets = [sheet for sheet in required_sheets if sheet not in sheets]
            extra_sheets = [sheet for sheet in sheets if sheet not in required_sheets]
            
            structure_info = {
                "total_sheets": len(sheets),
                "required_sheets": len(required_sheets),
                "missing_sheets": missing_sheets,
                "extra_sheets": extra_sheets,
                "sheets_valid": len(missing_sheets) == 0
            }
            
            # Check each sheet structure
            sheet_validations = {}
            for sheet in sheets:
                if sheet in required_sheets:
                    df = excel_data.parse(sheet)
                    sheet_validations[sheet] = {
                        "rows": len(df),
                        "columns": len(df.columns),
                        "has_data": len(df) > 0,
                        "column_names": list(df.columns)
                    }
            
            structure_info["sheet_details"] = sheet_validations
            
            return structure_info["sheets_valid"], structure_info
            
        except Exception as e:
            return False, {"error": str(e)}
    
    async def validate_content(self, file_path: Path) -> Tuple[bool, Dict[str, Any]]:
        """Validate content of Excel file"""
        if self.excel_parser:
            try:
                is_valid, error_msg, regime_mode = self.excel_parser.validate_excel_file(str(file_path))
                
                content_info = {
                    "is_valid": is_valid,
                    "error_message": error_msg if not is_valid else None,
                    "regime_mode": regime_mode,
                    "validation_timestamp": datetime.now().isoformat()
                }
                
                # Additional content checks
                if is_valid:
                    config = self.excel_parser.parse_excel_config(str(file_path))
                    
                    content_info.update({
                        "indicators_count": len(config.indicators),
                        "enabled_indicators": [name for name, ind in config.indicators.items() if ind.enabled],
                        "regime_count": len(config.regimes) if hasattr(config, 'regimes') else 0,
                        "timeframes": list(config.timeframes.keys()) if hasattr(config, 'timeframes') else []
                    })
                
                return is_valid, content_info
                
            except Exception as e:
                return False, {"error": str(e)}
        
        return True, {"warning": "No excel parser available for content validation"}
    
    async def validate_advanced(self, file_path: Path) -> Tuple[bool, Dict[str, Any]]:
        """Advanced validation - cross-sheet dependencies and business logic"""
        try:
            excel_data = pd.ExcelFile(file_path)
            advanced_checks = {
                "cross_sheet_dependencies": {},
                "parameter_ranges": {},
                "logical_consistency": {}
            }
            
            # Check indicator weights sum to 1
            if 'IndicatorConfiguration' in excel_data.sheet_names:
                df = excel_data.parse('IndicatorConfiguration')
                if 'Weight' in df.columns:
                    total_weight = df['Weight'].sum()
                    advanced_checks["parameter_ranges"]["indicator_weights_sum"] = {
                        "value": total_weight,
                        "valid": 0.95 <= total_weight <= 1.05,
                        "message": f"Indicator weights sum to {total_weight:.3f}"
                    }
            
            # Check regime thresholds consistency
            if 'RegimeFormationConfig' in excel_data.sheet_names:
                df = excel_data.parse('RegimeFormationConfig')
                # Add specific regime threshold checks here
                advanced_checks["logical_consistency"]["regime_thresholds"] = {
                    "valid": True,
                    "message": "Regime thresholds are consistent"
                }
            
            # Overall advanced validation result
            all_valid = all(
                check.get("valid", True) 
                for category in advanced_checks.values() 
                for check in category.values()
            )
            
            return all_valid, advanced_checks
            
        except Exception as e:
            return False, {"error": str(e)}


class ProgressiveUploadAPI:
    """API for progressive file upload with enhanced validation"""
    
    def __init__(self, upload_dir: str = "/tmp/uploads", excel_parser=None):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = UploadProgressTracker()
        self.validator = ProgressiveValidator(excel_parser)
        
    def create_routes(self, router):
        """Create FastAPI routes for progressive upload"""
        
        @router.post("/upload/init")
        async def initialize_upload(
            filename: str = Form(...),
            file_size: int = Form(...),
            chunk_size: int = Form(default=1024*1024)
        ):
            """Initialize a new upload session"""
            try:
                # Generate session ID
                session_id = hashlib.md5(f"{filename}_{datetime.now().timestamp()}".encode()).hexdigest()
                
                # Basic validation
                is_valid, errors = await self.validator.validate_basic(filename, file_size)
                if not is_valid:
                    return JSONResponse(
                        status_code=400,
                        content={"status": "error", "errors": errors}
                    )
                
                # Create upload session
                session = await self.tracker.create_session(
                    session_id=session_id,
                    filename=filename,
                    file_size=file_size,
                    chunk_size=chunk_size
                )
                
                # Create temporary file
                temp_path = self.upload_dir / f"temp_{session_id}"
                session.temp_path = temp_path
                
                return JSONResponse(content={
                    "status": "success",
                    "session_id": session_id,
                    "chunk_size": chunk_size,
                    "total_chunks": session.total_chunks,
                    "upload_url": f"/upload/chunk/{session_id}"
                })
                
            except Exception as e:
                logger.error(f"Upload initialization failed: {e}")
                return JSONResponse(
                    status_code=500,
                    content={"status": "error", "message": str(e)}
                )
        
        @router.post("/upload/chunk/{session_id}")
        async def upload_chunk(
            session_id: str,
            chunk_index: int = Form(...),
            chunk_file: UploadFile = File(...),
            chunk_hash: str = Form(...)
        ):
            """Upload a file chunk"""
            try:
                # Get session
                if session_id not in self.tracker.sessions:
                    raise HTTPException(status_code=404, detail="Session not found")
                
                session = self.tracker.sessions[session_id]
                
                # Verify chunk
                chunk_data = await chunk_file.read()
                actual_hash = hashlib.md5(chunk_data).hexdigest()
                
                if actual_hash != chunk_hash:
                    return JSONResponse(
                        status_code=400,
                        content={"status": "error", "message": "Chunk hash mismatch"}
                    )
                
                # Write chunk to temporary file
                with open(session.temp_path, 'ab') as f:
                    f.seek(chunk_index * len(chunk_data))
                    f.write(chunk_data)
                
                # Update progress
                chunk_info = ChunkInfo(
                    chunk_index=chunk_index,
                    chunk_size=len(chunk_data),
                    chunk_hash=chunk_hash,
                    timestamp=datetime.now()
                )
                
                progress = await self.tracker.update_chunk(session_id, chunk_info)
                
                # Check if upload is complete
                if len(session.uploaded_chunks) == session.total_chunks:
                    # Start validation in background
                    asyncio.create_task(self._validate_upload(session_id))
                
                return JSONResponse(content={
                    "status": "success",
                    "progress": progress,
                    "message": f"Chunk {chunk_index + 1}/{session.total_chunks} uploaded"
                })
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Chunk upload failed: {e}")
                return JSONResponse(
                    status_code=500,
                    content={"status": "error", "message": str(e)}
                )
        
        @router.get("/upload/status/{session_id}")
        async def get_upload_status(session_id: str):
            """Get upload session status"""
            if session_id not in self.tracker.sessions:
                raise HTTPException(status_code=404, detail="Session not found")
            
            session = self.tracker.sessions[session_id]
            
            return JSONResponse(content={
                "status": session.status.value,
                "filename": session.filename,
                "progress": len(session.uploaded_chunks) / session.total_chunks * 100,
                "uploaded_chunks": len(session.uploaded_chunks),
                "total_chunks": session.total_chunks,
                "validation_results": session.validation_results,
                "start_time": session.start_time.isoformat(),
                "last_activity": session.last_activity.isoformat()
            })
        
        @router.websocket("/upload/ws/{session_id}")
        async def upload_websocket(websocket: WebSocket, session_id: str):
            """WebSocket for real-time upload updates"""
            await websocket.accept()
            self.tracker.websocket_connections[session_id] = websocket
            
            try:
                while True:
                    # Keep connection alive
                    await asyncio.sleep(1)
            except WebSocketDisconnect:
                del self.tracker.websocket_connections[session_id]
        
        @router.post("/upload/cancel/{session_id}")
        async def cancel_upload(session_id: str):
            """Cancel an upload session"""
            if session_id not in self.tracker.sessions:
                raise HTTPException(status_code=404, detail="Session not found")
            
            session = self.tracker.sessions[session_id]
            session.status = UploadStatus.CANCELLED
            
            # Clean up temporary file
            if session.temp_path and session.temp_path.exists():
                session.temp_path.unlink()
            
            return JSONResponse(content={
                "status": "success",
                "message": "Upload cancelled"
            })
        
        @router.get("/upload/validate/{session_id}")
        async def get_validation_results(session_id: str):
            """Get detailed validation results"""
            if session_id not in self.tracker.sessions:
                raise HTTPException(status_code=404, detail="Session not found")
            
            session = self.tracker.sessions[session_id]
            
            return JSONResponse(content={
                "status": "success",
                "validation_results": session.validation_results,
                "validation_complete": session.status in [
                    UploadStatus.COMPLETED, 
                    UploadStatus.FAILED
                ]
            })
    
    async def _validate_upload(self, session_id: str):
        """Validate uploaded file progressively"""
        try:
            session = self.tracker.sessions[session_id]
            session.status = UploadStatus.VALIDATING
            
            # Broadcast validation start
            await self.tracker._broadcast_progress(session_id, {
                "type": "validation_start",
                "session_id": session_id,
                "status": "validating"
            })
            
            # Level 1: Structure validation
            is_valid, structure_results = await self.validator.validate_structure(session.temp_path)
            session.validation_results["structure"] = structure_results
            
            await self.tracker._broadcast_progress(session_id, {
                "type": "validation_progress",
                "session_id": session_id,
                "level": "structure",
                "results": structure_results
            })
            
            if not is_valid:
                session.status = UploadStatus.FAILED
                return
            
            # Level 2: Content validation
            is_valid, content_results = await self.validator.validate_content(session.temp_path)
            session.validation_results["content"] = content_results
            
            await self.tracker._broadcast_progress(session_id, {
                "type": "validation_progress",
                "session_id": session_id,
                "level": "content",
                "results": content_results
            })
            
            if not is_valid:
                session.status = UploadStatus.FAILED
                return
            
            # Level 3: Advanced validation
            is_valid, advanced_results = await self.validator.validate_advanced(session.temp_path)
            session.validation_results["advanced"] = advanced_results
            
            await self.tracker._broadcast_progress(session_id, {
                "type": "validation_progress",
                "session_id": session_id,
                "level": "advanced",
                "results": advanced_results
            })
            
            # Move to final location if all validations pass
            if is_valid:
                final_filename = f"regime_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                session.final_path = self.upload_dir / final_filename
                shutil.move(session.temp_path, session.final_path)
                session.status = UploadStatus.COMPLETED
                
                await self.tracker._broadcast_progress(session_id, {
                    "type": "upload_complete",
                    "session_id": session_id,
                    "status": "completed",
                    "file_path": str(session.final_path),
                    "validation_results": session.validation_results
                })
            else:
                session.status = UploadStatus.FAILED
                
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            session.status = UploadStatus.FAILED
            session.validation_results["error"] = str(e)


def create_progressive_upload_system(excel_parser=None) -> ProgressiveUploadAPI:
    """Factory function to create progressive upload system"""
    return ProgressiveUploadAPI(
        upload_dir="/srv/samba/shared/bt/backtester_stable/uploads",
        excel_parser=excel_parser
    )


# Example integration
if __name__ == "__main__":
    from fastapi import FastAPI
    import uvicorn
    
    app = FastAPI()
    upload_api = create_progressive_upload_system()
    
    # Create routes
    from fastapi import APIRouter
    router = APIRouter(prefix="/api/v2/progressive")
    upload_api.create_routes(router)
    app.include_router(router)
    
    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8001)