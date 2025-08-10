"""
Batch Configuration Processor

Efficiently processes large batches of configuration files with parallel processing,
progress tracking, and comprehensive error handling.
"""

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
import threading
import json
import hashlib
from collections import defaultdict

from .unified_gateway import UnifiedConfigurationGateway
from .strategy_detector import StrategyDetector
from ..core.exceptions import ConfigurationError, ParsingError, ValidationError

logger = logging.getLogger(__name__)

@dataclass
class BatchJobResult:
    """Result of processing a single file in a batch"""
    file_path: str
    status: str  # 'success', 'failed', 'duplicate', 'skipped'
    strategy_type: Optional[str] = None
    configuration_id: Optional[str] = None
    version_id: Optional[str] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    file_size: int = 0
    duplicate_of: Optional[str] = None

@dataclass
class BatchJobStatus:
    """Status of a batch processing job"""
    job_id: str
    total_files: int
    processed_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    duplicate_files: int = 0
    skipped_files: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: str = 'running'  # 'running', 'completed', 'failed', 'cancelled'
    current_file: Optional[str] = None
    error_summary: List[str] = field(default_factory=list)
    
    @property
    def progress_percentage(self) -> float:
        return (self.processed_files / self.total_files * 100) if self.total_files > 0 else 0
    
    @property
    def processing_time(self) -> float:
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()
    
    @property
    def files_per_second(self) -> float:
        time_elapsed = self.processing_time
        return self.processed_files / time_elapsed if time_elapsed > 0 else 0

class BatchProcessor:
    """
    Batch processor for configuration files
    
    Handles large-scale processing of configuration files with:
    - Parallel processing for performance
    - Progress tracking and real-time updates
    - Error handling and recovery
    - Deduplication and validation
    - Resume capabilities for interrupted jobs
    """
    
    def __init__(self, 
                 gateway: Optional[UnifiedConfigurationGateway] = None,
                 max_workers: int = 10,
                 chunk_size: int = 100):
        """
        Initialize batch processor
        
        Args:
            gateway: Unified configuration gateway
            max_workers: Maximum number of parallel workers
            chunk_size: Number of files per processing chunk
        """
        self.gateway = gateway or UnifiedConfigurationGateway()
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        
        # Job tracking
        self.active_jobs: Dict[str, BatchJobStatus] = {}
        self.job_results: Dict[str, List[BatchJobResult]] = {}
        self._lock = threading.Lock()
        
        # Progress callbacks
        self.progress_callbacks: List[Callable[[BatchJobStatus], None]] = []
        
        # File filters
        self.supported_extensions = {'.xlsx', '.xls', '.xlsm'}
        
        logger.info(f"BatchProcessor initialized with {max_workers} workers")
    
    def add_progress_callback(self, callback: Callable[[BatchJobStatus], None]):
        """Add a progress callback function"""
        self.progress_callbacks.append(callback)
    
    def process_folder(self, 
                      folder_path: str,
                      pattern: str = "*.xlsx",
                      recursive: bool = True,
                      author: str = "batch_processor",
                      strategy_filter: Optional[List[str]] = None,
                      dry_run: bool = False) -> str:
        """
        Process all configuration files in a folder
        
        Args:
            folder_path: Path to folder containing Excel files
            pattern: File pattern to match (e.g., "*.xlsx", "tbs_*.xlsx")
            recursive: Process subdirectories recursively
            author: Author name for version control
            strategy_filter: List of strategy types to process (None = all)
            dry_run: Only analyze files without processing
            
        Returns:
            Job ID for tracking progress
        """
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise ValueError(f"Folder does not exist: {folder_path}")
        
        # Find all matching files
        if recursive:
            files = list(folder_path.rglob(pattern))
        else:
            files = list(folder_path.glob(pattern))
        
        # Filter by extension
        files = [f for f in files if f.suffix.lower() in self.supported_extensions]
        
        if not files:
            raise ValueError(f"No Excel files found in {folder_path}")
        
        return self.process_files(
            file_paths=[str(f) for f in files],
            author=author,
            strategy_filter=strategy_filter,
            dry_run=dry_run
        )
    
    def process_files(self,
                     file_paths: List[str],
                     author: str = "batch_processor", 
                     strategy_filter: Optional[List[str]] = None,
                     dry_run: bool = False) -> str:
        """
        Process a list of configuration files
        
        Args:
            file_paths: List of file paths to process
            author: Author name for version control
            strategy_filter: List of strategy types to process
            dry_run: Only analyze files without processing
            
        Returns:
            Job ID for tracking progress
        """
        # Generate job ID
        job_id = f"batch_{int(time.time())}_{len(file_paths)}"
        
        # Create job status
        job_status = BatchJobStatus(
            job_id=job_id,
            total_files=len(file_paths)
        )
        
        with self._lock:
            self.active_jobs[job_id] = job_status
            self.job_results[job_id] = []
        
        # Start processing in background thread
        def process_batch():
            try:
                self._process_batch_internal(
                    job_id, file_paths, author, strategy_filter, dry_run
                )
            except Exception as e:
                logger.error(f"Batch job {job_id} failed: {e}")
                with self._lock:
                    job_status.status = 'failed'
                    job_status.end_time = datetime.now()
                    job_status.error_summary.append(f"Job failed: {str(e)}")
        
        thread = threading.Thread(target=process_batch, daemon=True)
        thread.start()
        
        logger.info(f"Started batch job {job_id} with {len(file_paths)} files")
        return job_id
    
    def _process_batch_internal(self,
                               job_id: str,
                               file_paths: List[str],
                               author: str,
                               strategy_filter: Optional[List[str]],
                               dry_run: bool):
        """Internal batch processing implementation"""
        
        job_status = self.active_jobs[job_id]
        results = []
        
        try:
            # Pre-analyze files for deduplication
            file_hashes = {}
            duplicate_map = {}
            
            logger.info(f"Pre-analyzing {len(file_paths)} files for duplicates...")
            
            for file_path in file_paths:
                try:
                    file_hash = self._calculate_file_hash(file_path)
                    if file_hash in file_hashes:
                        duplicate_map[file_path] = file_hashes[file_hash]
                    else:
                        file_hashes[file_hash] = file_path
                except Exception as e:
                    logger.warning(f"Failed to hash {file_path}: {e}")
            
            logger.info(f"Found {len(duplicate_map)} duplicate files")
            
            # Process files in chunks with parallel workers
            unique_files = [f for f in file_paths if f not in duplicate_map]
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_file = {}
                
                for file_path in unique_files:
                    future = executor.submit(
                        self._process_single_file,
                        file_path, author, strategy_filter, dry_run
                    )
                    future_to_file[future] = file_path
                
                # Process completed tasks
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    
                    try:
                        result = future.result()
                        results.append(result)
                        
                        # Update job status
                        with self._lock:
                            job_status.processed_files += 1
                            job_status.current_file = file_path
                            
                            if result.status == 'success':
                                job_status.successful_files += 1
                            elif result.status == 'failed':
                                job_status.failed_files += 1
                                if result.error_message:
                                    job_status.error_summary.append(
                                        f"{Path(file_path).name}: {result.error_message}"
                                    )
                            elif result.status == 'skipped':
                                job_status.skipped_files += 1
                        
                        # Notify progress callbacks
                        self._notify_progress(job_status)
                        
                    except Exception as e:
                        logger.error(f"Failed to process {file_path}: {e}")
                        
                        # Create error result
                        error_result = BatchJobResult(
                            file_path=file_path,
                            status='failed',
                            error_message=str(e),
                            file_size=self._get_file_size(file_path)
                        )
                        results.append(error_result)
                        
                        with self._lock:
                            job_status.processed_files += 1
                            job_status.failed_files += 1
                            job_status.error_summary.append(f"{Path(file_path).name}: {str(e)}")
            
            # Handle duplicate files
            for duplicate_file, original_file in duplicate_map.items():
                duplicate_result = BatchJobResult(
                    file_path=duplicate_file,
                    status='duplicate',
                    duplicate_of=original_file,
                    file_size=self._get_file_size(duplicate_file)
                )
                results.append(duplicate_result)
                
                with self._lock:
                    job_status.processed_files += 1
                    job_status.duplicate_files += 1
            
            # Complete the job
            with self._lock:
                job_status.status = 'completed'
                job_status.end_time = datetime.now()
                self.job_results[job_id] = results
            
            logger.info(f"Batch job {job_id} completed: "
                       f"{job_status.successful_files} success, "
                       f"{job_status.failed_files} failed, "
                       f"{job_status.duplicate_files} duplicates")
            
            # Final progress notification
            self._notify_progress(job_status)
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            with self._lock:
                job_status.status = 'failed'
                job_status.end_time = datetime.now()
                job_status.error_summary.append(f"Batch processing failed: {str(e)}")
            
            self._notify_progress(job_status)
            raise
    
    def _process_single_file(self,
                           file_path: str,
                           author: str,
                           strategy_filter: Optional[List[str]],
                           dry_run: bool) -> BatchJobResult:
        """Process a single configuration file"""
        
        start_time = time.time()
        file_size = self._get_file_size(file_path)
        
        try:
            # Detect strategy type
            detector = StrategyDetector()
            strategy_type = detector.detect_strategy_type(file_path)
            
            if not strategy_type:
                return BatchJobResult(
                    file_path=file_path,
                    status='failed',
                    error_message="Could not detect strategy type",
                    processing_time=time.time() - start_time,
                    file_size=file_size
                )
            
            # Apply strategy filter
            if strategy_filter and strategy_type not in strategy_filter:
                return BatchJobResult(
                    file_path=file_path,
                    status='skipped',
                    strategy_type=strategy_type,
                    error_message=f"Strategy {strategy_type} not in filter",
                    processing_time=time.time() - start_time,
                    file_size=file_size
                )
            
            if dry_run:
                # Dry run - just validate detection
                return BatchJobResult(
                    file_path=file_path,
                    status='success',
                    strategy_type=strategy_type,
                    processing_time=time.time() - start_time,
                    file_size=file_size
                )
            
            # Load configuration
            config_name = Path(file_path).stem
            commit_message = f"Batch import: {Path(file_path).name}"
            
            config = self.gateway.load_configuration(
                strategy_type=strategy_type,
                file_path=file_path,
                config_name=config_name,
                author=author,
                commit_message=commit_message
            )
            
            # Extract version info if available
            version_id = None
            if hasattr(config, '_metadata') and config._metadata:
                version_id = config._metadata.get('version_id')
            
            configuration_id = f"{strategy_type}_{config_name}"
            
            return BatchJobResult(
                file_path=file_path,
                status='success',
                strategy_type=strategy_type,
                configuration_id=configuration_id,
                version_id=version_id,
                processing_time=time.time() - start_time,
                file_size=file_size
            )
            
        except (ConfigurationError, ParsingError, ValidationError) as e:
            return BatchJobResult(
                file_path=file_path,
                status='failed',
                error_message=str(e),
                processing_time=time.time() - start_time,
                file_size=file_size
            )
        except Exception as e:
            logger.exception(f"Unexpected error processing {file_path}")
            return BatchJobResult(
                file_path=file_path,
                status='failed', 
                error_message=f"Unexpected error: {str(e)}",
                processing_time=time.time() - start_time,
                file_size=file_size
            )
    
    def get_job_status(self, job_id: str) -> Optional[BatchJobStatus]:
        """Get status of a batch job"""
        with self._lock:
            return self.active_jobs.get(job_id)
    
    def get_job_results(self, job_id: str) -> Optional[List[BatchJobResult]]:
        """Get results of a completed batch job"""
        with self._lock:
            return self.job_results.get(job_id)
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running batch job"""
        with self._lock:
            if job_id in self.active_jobs:
                job_status = self.active_jobs[job_id]
                if job_status.status == 'running':
                    job_status.status = 'cancelled'
                    job_status.end_time = datetime.now()
                    return True
        return False
    
    def list_jobs(self) -> List[BatchJobStatus]:
        """List all batch jobs"""
        with self._lock:
            return list(self.active_jobs.values())
    
    def get_job_summary(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive job summary"""
        status = self.get_job_status(job_id)
        results = self.get_job_results(job_id)
        
        if not status:
            return None
        
        summary = {
            'job_id': job_id,
            'status': status.status,
            'progress': {
                'total_files': status.total_files,
                'processed_files': status.processed_files,
                'progress_percentage': status.progress_percentage,
                'processing_time': status.processing_time,
                'files_per_second': status.files_per_second
            },
            'results': {
                'successful_files': status.successful_files,
                'failed_files': status.failed_files,
                'duplicate_files': status.duplicate_files,
                'skipped_files': status.skipped_files
            }
        }
        
        if results:
            # Strategy breakdown
            strategy_counts = defaultdict(int)
            total_size = 0
            
            for result in results:
                if result.strategy_type:
                    strategy_counts[result.strategy_type] += 1
                total_size += result.file_size
            
            summary['analysis'] = {
                'strategy_breakdown': dict(strategy_counts),
                'total_file_size_mb': round(total_size / (1024 * 1024), 2),
                'average_processing_time': sum(r.processing_time for r in results) / len(results)
            }
            
            # Error analysis
            if status.error_summary:
                summary['errors'] = status.error_summary[:10]  # First 10 errors
        
        return summary
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _get_file_size(self, file_path: str) -> int:
        """Get file size in bytes"""
        try:
            return os.path.getsize(file_path)
        except:
            return 0
    
    def _notify_progress(self, job_status: BatchJobStatus):
        """Notify all progress callbacks"""
        for callback in self.progress_callbacks:
            try:
                callback(job_status)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
    
    def cleanup_completed_jobs(self, keep_recent: int = 10):
        """Clean up old completed job records"""
        with self._lock:
            # Keep only the most recent completed jobs
            completed_jobs = [
                (job_id, status) for job_id, status in self.active_jobs.items()
                if status.status in ['completed', 'failed', 'cancelled']
            ]
            
            # Sort by end time
            completed_jobs.sort(key=lambda x: x[1].end_time or datetime.min, reverse=True)
            
            # Remove old jobs
            for job_id, _ in completed_jobs[keep_recent:]:
                del self.active_jobs[job_id]
                if job_id in self.job_results:
                    del self.job_results[job_id]
            
            logger.info(f"Cleaned up {len(completed_jobs) - keep_recent} old job records")
    
    def export_job_results(self, job_id: str, output_path: str) -> bool:
        """Export job results to JSON file"""
        try:
            summary = self.get_job_summary(job_id)
            results = self.get_job_results(job_id)
            
            if not summary or not results:
                return False
            
            export_data = {
                'summary': summary,
                'detailed_results': [
                    {
                        'file_path': r.file_path,
                        'status': r.status,
                        'strategy_type': r.strategy_type,
                        'configuration_id': r.configuration_id,
                        'version_id': r.version_id,
                        'error_message': r.error_message,
                        'processing_time': r.processing_time,
                        'file_size': r.file_size,
                        'duplicate_of': r.duplicate_of
                    }
                    for r in results
                ],
                'exported_at': datetime.now().isoformat()
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported job {job_id} results to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export job results: {e}")
            return False