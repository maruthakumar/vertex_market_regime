"""
Enhanced Upload API

REST API endpoints for uploading and managing configuration files
with advanced features like batch processing, progress tracking, and real-time updates.
"""

import asyncio
import logging
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import json

from fastapi import (
    APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks,
    WebSocket, WebSocketDisconnect, Query, Depends
)
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import aiofiles

from ..gateway import UnifiedConfigurationGateway, BatchProcessor
from ..gateway.strategy_detector import StrategyDetector
from ..core.exceptions import ConfigurationError, ParsingError, ValidationError

logger = logging.getLogger(__name__)

# Pydantic models for API responses
class UploadResponse(BaseModel):
    """Response for single file upload"""
    upload_id: str
    filename: str
    strategy_type: Optional[str] = None
    configuration_id: Optional[str] = None
    version_id: Optional[str] = None
    status: str
    message: str
    file_size: int
    processing_time: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

class BatchUploadResponse(BaseModel):
    """Response for batch upload initiation"""
    job_id: str
    total_files: int
    status: str
    message: str
    estimated_time: Optional[float] = None

class JobStatusResponse(BaseModel):
    """Response for job status queries"""
    job_id: str
    status: str
    progress: Dict[str, Any]
    results: Dict[str, Any]
    errors: List[str] = Field(default_factory=list)

class StrategyDetectionResponse(BaseModel):
    """Response for strategy detection"""
    filename: str
    detected_strategy: Optional[str] = None
    confidence: float = 0.0
    suggestions: List[Dict[str, Any]] = Field(default_factory=list)

class EnhancedUploadAPI:
    """
    Enhanced Upload API for configuration management
    
    Provides REST endpoints for:
    - Single file uploads with auto-detection
    - Batch folder processing
    - Progress tracking with WebSocket
    - Strategy detection and validation
    - File management and cleanup
    """
    
    def __init__(self, 
                 gateway: Optional[UnifiedConfigurationGateway] = None,
                 upload_dir: Optional[str] = None,
                 max_file_size: int = 50 * 1024 * 1024):  # 50MB
        """
        Initialize Enhanced Upload API
        
        Args:
            gateway: Unified configuration gateway
            upload_dir: Directory for temporary file storage
            max_file_size: Maximum file size in bytes
        """
        self.gateway = gateway or UnifiedConfigurationGateway()
        self.batch_processor = BatchProcessor(gateway=self.gateway)
        self.strategy_detector = StrategyDetector()
        
        self.upload_dir = Path(upload_dir or tempfile.gettempdir()) / "config_uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_file_size = max_file_size
        
        # WebSocket connections for progress tracking
        self.websocket_connections: Dict[str, WebSocket] = {}
        
        # Add progress callback to batch processor
        self.batch_processor.add_progress_callback(self._notify_websocket_clients)
        
        # Create router
        self.router = APIRouter(prefix="/api/v1/upload", tags=["Upload"])
        self._setup_routes()
        
        logger.info(f"EnhancedUploadAPI initialized with upload dir: {self.upload_dir}")
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.router.post("/single", response_model=UploadResponse)
        async def upload_single_file(
            background_tasks: BackgroundTasks,
            file: UploadFile = File(...),
            strategy_type: str = Form("auto"),
            author: str = Form("api_user"),
            commit_message: Optional[str] = Form(None),
            validate_only: bool = Form(False)
        ):
            """Upload and process a single configuration file"""
            return await self._upload_single_file(
                file, strategy_type, author, commit_message, validate_only, background_tasks
            )
        
        @self.router.post("/batch", response_model=BatchUploadResponse)
        async def upload_batch_files(
            background_tasks: BackgroundTasks,
            files: List[UploadFile] = File(...),
            author: str = Form("api_user"),
            strategy_filter: Optional[str] = Form(None),
            dry_run: bool = Form(False)
        ):
            """Upload and process multiple configuration files"""
            return await self._upload_batch_files(
                files, author, strategy_filter, dry_run, background_tasks
            )
        
        @self.router.post("/folder", response_model=BatchUploadResponse)
        async def process_folder(
            background_tasks: BackgroundTasks,
            folder_path: str = Form(...),
            pattern: str = Form("*.xlsx"),
            recursive: bool = Form(True),
            author: str = Form("api_user"),
            strategy_filter: Optional[str] = Form(None),
            dry_run: bool = Form(False)
        ):
            """Process all configuration files in a folder"""
            return await self._process_folder(
                folder_path, pattern, recursive, author, strategy_filter, dry_run, background_tasks
            )
        
        @self.router.get("/job/{job_id}/status", response_model=JobStatusResponse)
        async def get_job_status(job_id: str):
            """Get status of a batch processing job"""
            return await self._get_job_status(job_id)
        
        @self.router.get("/job/{job_id}/results")
        async def get_job_results(job_id: str):
            """Get detailed results of a completed job"""
            return await self._get_job_results(job_id)
        
        @self.router.delete("/job/{job_id}")
        async def cancel_job(job_id: str):
            """Cancel a running batch job"""
            return await self._cancel_job(job_id)
        
        @self.router.post("/detect", response_model=StrategyDetectionResponse)
        async def detect_strategy(file: UploadFile = File(...)):
            """Detect strategy type from uploaded file"""
            return await self._detect_strategy(file)
        
        @self.router.get("/jobs")
        async def list_jobs():
            """List all batch processing jobs"""
            return await self._list_jobs()
        
        @self.router.websocket("/progress/{job_id}")
        async def websocket_progress(websocket: WebSocket, job_id: str):
            """WebSocket endpoint for real-time progress updates"""
            await self._websocket_progress(websocket, job_id)
    
    async def _upload_single_file(self,
                                 file: UploadFile,
                                 strategy_type: str,
                                 author: str,
                                 commit_message: Optional[str],
                                 validate_only: bool,
                                 background_tasks: BackgroundTasks) -> UploadResponse:
        """Handle single file upload"""
        
        start_time = datetime.now()
        upload_id = str(uuid.uuid4())
        
        try:
            # Validate file
            if not file.filename:
                raise HTTPException(status_code=400, detail="No filename provided")
            
            if file.size and file.size > self.max_file_size:
                raise HTTPException(status_code=413, detail=f"File too large. Max size: {self.max_file_size} bytes")
            
            # Check file extension
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in {'.xlsx', '.xls', '.xlsm'}:
                raise HTTPException(status_code=400, detail="Invalid file type. Only Excel files are supported")
            
            # Save uploaded file temporarily
            temp_file_path = self.upload_dir / f"{upload_id}_{file.filename}"
            
            async with aiofiles.open(temp_file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            file_size = len(content)
            
            # Detect strategy type if auto
            if strategy_type == "auto":
                detected_strategy = self.strategy_detector.detect_strategy_type(str(temp_file_path))
                if not detected_strategy:
                    # Clean up temp file
                    temp_file_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=400, detail="Could not auto-detect strategy type")
                strategy_type = detected_strategy
            
            if validate_only:
                # Only validate, don't actually process
                # Clean up temp file
                temp_file_path.unlink(missing_ok=True)
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                return UploadResponse(
                    upload_id=upload_id,
                    filename=file.filename,
                    strategy_type=strategy_type,
                    status="validated",
                    message="File validation successful",
                    file_size=file_size,
                    processing_time=processing_time
                )
            
            # Process configuration
            config_name = Path(file.filename).stem
            commit_msg = commit_message or f"API upload: {file.filename}"
            
            config = self.gateway.load_configuration(
                strategy_type=strategy_type,
                file_path=str(temp_file_path),
                config_name=config_name,
                author=author,
                commit_message=commit_msg
            )
            
            # Extract metadata
            configuration_id = f"{strategy_type}_{config_name}"
            version_id = None
            
            if hasattr(config, '_metadata') and config._metadata:
                version_id = config._metadata.get('version_id')
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Schedule cleanup
            background_tasks.add_task(self._cleanup_temp_file, temp_file_path)
            
            return UploadResponse(
                upload_id=upload_id,
                filename=file.filename,
                strategy_type=strategy_type,
                configuration_id=configuration_id,
                version_id=version_id,
                status="success",
                message="File uploaded and processed successfully",
                file_size=file_size,
                processing_time=processing_time,
                metadata={
                    "config_name": config_name,
                    "author": author,
                    "commit_message": commit_msg
                }
            )
            
        except (ConfigurationError, ParsingError, ValidationError) as e:
            # Clean up temp file
            if 'temp_file_path' in locals():
                background_tasks.add_task(self._cleanup_temp_file, temp_file_path)
            
            raise HTTPException(status_code=400, detail=f"Configuration error: {str(e)}")
        
        except Exception as e:
            # Clean up temp file
            if 'temp_file_path' in locals():
                background_tasks.add_task(self._cleanup_temp_file, temp_file_path)
            
            logger.exception(f"Upload failed for {file.filename}")
            raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    
    async def _upload_batch_files(self,
                                 files: List[UploadFile],
                                 author: str,
                                 strategy_filter: Optional[str],
                                 dry_run: bool,
                                 background_tasks: BackgroundTasks) -> BatchUploadResponse:
        """Handle batch file upload"""
        
        try:
            if not files:
                raise HTTPException(status_code=400, detail="No files provided")
            
            if len(files) > 1000:  # Reasonable limit
                raise HTTPException(status_code=400, detail="Too many files. Maximum 1000 files per batch")
            
            # Parse strategy filter
            strategy_list = None
            if strategy_filter:
                strategy_list = [s.strip() for s in strategy_filter.split(',')]
            
            # Save all files temporarily
            temp_files = []
            total_size = 0
            
            for file in files:
                if not file.filename:
                    continue
                
                # Check file extension
                file_ext = Path(file.filename).suffix.lower()
                if file_ext not in {'.xlsx', '.xls', '.xlsm'}:
                    continue
                
                # Save file
                temp_file_path = self.upload_dir / f"batch_{uuid.uuid4()}_{file.filename}"
                
                async with aiofiles.open(temp_file_path, 'wb') as f:
                    content = await file.read()
                    await f.write(content)
                    total_size += len(content)
                
                temp_files.append(str(temp_file_path))
            
            if not temp_files:
                raise HTTPException(status_code=400, detail="No valid Excel files found")
            
            # Check total size
            if total_size > self.max_file_size * 10:  # 10x single file limit for batch
                raise HTTPException(status_code=413, detail="Batch too large")
            
            # Start batch processing
            job_id = self.batch_processor.process_files(
                file_paths=temp_files,
                author=author,
                strategy_filter=strategy_list,
                dry_run=dry_run
            )
            
            # Schedule cleanup of temp files
            for temp_file in temp_files:
                background_tasks.add_task(self._cleanup_temp_file, Path(temp_file))
            
            # Estimate processing time (rough estimate: 2 seconds per file)
            estimated_time = len(temp_files) * 2.0 / self.batch_processor.max_workers
            
            return BatchUploadResponse(
                job_id=job_id,
                total_files=len(temp_files),
                status="started",
                message=f"Batch processing started for {len(temp_files)} files",
                estimated_time=estimated_time
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Batch upload failed")
            raise HTTPException(status_code=500, detail=f"Batch upload failed: {str(e)}")
    
    async def _process_folder(self,
                             folder_path: str,
                             pattern: str,
                             recursive: bool,
                             author: str,
                             strategy_filter: Optional[str],
                             dry_run: bool,
                             background_tasks: BackgroundTasks) -> BatchUploadResponse:
        """Handle folder processing"""
        
        try:
            # Validate folder path
            folder = Path(folder_path)
            if not folder.exists():
                raise HTTPException(status_code=400, detail=f"Folder does not exist: {folder_path}")
            
            if not folder.is_dir():
                raise HTTPException(status_code=400, detail=f"Path is not a directory: {folder_path}")
            
            # Parse strategy filter
            strategy_list = None
            if strategy_filter:
                strategy_list = [s.strip() for s in strategy_filter.split(',')]
            
            # Start folder processing
            job_id = self.batch_processor.process_folder(
                folder_path=folder_path,
                pattern=pattern,
                recursive=recursive,
                author=author,
                strategy_filter=strategy_list,
                dry_run=dry_run
            )
            
            # Get job status for file count
            job_status = self.batch_processor.get_job_status(job_id)
            total_files = job_status.total_files if job_status else 0
            
            # Estimate processing time
            estimated_time = total_files * 2.0 / self.batch_processor.max_workers if total_files > 0 else 0
            
            return BatchUploadResponse(
                job_id=job_id,
                total_files=total_files,
                status="started",
                message=f"Folder processing started for {folder_path}",
                estimated_time=estimated_time
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Folder processing failed")
            raise HTTPException(status_code=500, detail=f"Folder processing failed: {str(e)}")
    
    async def _get_job_status(self, job_id: str) -> JobStatusResponse:
        """Get job status"""
        job_status = self.batch_processor.get_job_status(job_id)
        
        if not job_status:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return JobStatusResponse(
            job_id=job_id,
            status=job_status.status,
            progress={
                "total_files": job_status.total_files,
                "processed_files": job_status.processed_files,
                "progress_percentage": job_status.progress_percentage,
                "processing_time": job_status.processing_time,
                "files_per_second": job_status.files_per_second,
                "current_file": job_status.current_file
            },
            results={
                "successful_files": job_status.successful_files,
                "failed_files": job_status.failed_files,
                "duplicate_files": job_status.duplicate_files,
                "skipped_files": job_status.skipped_files
            },
            errors=job_status.error_summary
        )
    
    async def _get_job_results(self, job_id: str):
        """Get detailed job results"""
        summary = self.batch_processor.get_job_summary(job_id)
        
        if not summary:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return summary
    
    async def _cancel_job(self, job_id: str):
        """Cancel a job"""
        success = self.batch_processor.cancel_job(job_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="Could not cancel job (not found or already completed)")
        
        return {"message": f"Job {job_id} cancelled successfully"}
    
    async def _detect_strategy(self, file: UploadFile) -> StrategyDetectionResponse:
        """Detect strategy type from file"""
        
        try:
            if not file.filename:
                raise HTTPException(status_code=400, detail="No filename provided")
            
            # Save file temporarily
            temp_file_path = self.upload_dir / f"detect_{uuid.uuid4()}_{file.filename}"
            
            async with aiofiles.open(temp_file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            try:
                # Detect strategy
                detected_strategy = self.strategy_detector.detect_strategy_type(str(temp_file_path))
                confidence = 0.0
                suggestions = []
                
                if detected_strategy:
                    confidence = self.strategy_detector.get_detection_confidence(
                        str(temp_file_path), detected_strategy
                    )
                
                # Get suggestions
                suggestions = self.strategy_detector.suggest_strategy_types(str(temp_file_path))
                
                return StrategyDetectionResponse(
                    filename=file.filename,
                    detected_strategy=detected_strategy,
                    confidence=confidence,
                    suggestions=suggestions
                )
            
            finally:
                # Clean up temp file
                temp_file_path.unlink(missing_ok=True)
                
        except Exception as e:
            logger.exception("Strategy detection failed")
            raise HTTPException(status_code=500, detail=f"Strategy detection failed: {str(e)}")
    
    async def _list_jobs(self):
        """List all jobs"""
        jobs = self.batch_processor.list_jobs()
        
        return [
            {
                "job_id": job.job_id,
                "status": job.status,
                "total_files": job.total_files,
                "processed_files": job.processed_files,
                "progress_percentage": job.progress_percentage,
                "start_time": job.start_time.isoformat(),
                "end_time": job.end_time.isoformat() if job.end_time else None
            }
            for job in jobs
        ]
    
    async def _websocket_progress(self, websocket: WebSocket, job_id: str):
        """Handle WebSocket connection for progress updates"""
        await websocket.accept()
        
        # Store connection
        self.websocket_connections[job_id] = websocket
        
        try:
            # Send initial status
            job_status = self.batch_processor.get_job_status(job_id)
            if job_status:
                await websocket.send_json({
                    "type": "status",
                    "data": {
                        "job_id": job_id,
                        "status": job_status.status,
                        "progress": job_status.progress_percentage,
                        "processed_files": job_status.processed_files,
                        "total_files": job_status.total_files
                    }
                })
            
            # Keep connection alive
            while True:
                await asyncio.sleep(1)
                
                # Check if job is completed
                job_status = self.batch_processor.get_job_status(job_id)
                if job_status and job_status.status in ['completed', 'failed', 'cancelled']:
                    break
                    
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"WebSocket error for job {job_id}: {e}")
        finally:
            # Remove connection
            if job_id in self.websocket_connections:
                del self.websocket_connections[job_id]
    
    def _notify_websocket_clients(self, job_status):
        """Notify WebSocket clients of progress updates"""
        job_id = job_status.job_id
        
        if job_id in self.websocket_connections:
            websocket = self.websocket_connections[job_id]
            
            # Send update (non-blocking)
            asyncio.create_task(websocket.send_json({
                "type": "progress",
                "data": {
                    "job_id": job_id,
                    "status": job_status.status,
                    "progress": job_status.progress_percentage,
                    "processed_files": job_status.processed_files,
                    "total_files": job_status.total_files,
                    "current_file": job_status.current_file
                }
            }))
    
    def _cleanup_temp_file(self, file_path: Path):
        """Clean up temporary file"""
        try:
            file_path.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file {file_path}: {e}")
    
    def get_router(self) -> APIRouter:
        """Get the FastAPI router"""
        return self.router