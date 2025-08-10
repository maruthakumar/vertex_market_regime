#!/usr/bin/env python3
"""
TV Parallel Processor
Handles parallel processing of multiple TV signal files for batch backtesting
"""

import os
import sys
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass
import json

from parser import TVParser
from signal_processor import SignalProcessor
from query_builder import TVQueryBuilder
from processor import TVProcessor

logger = logging.getLogger(__name__)


@dataclass
class TVProcessingJob:
    """Represents a single TV processing job"""
    job_id: str
    tv_config_path: str
    signal_files: List[str]
    output_directory: str
    priority: int = 1
    status: str = 'pending'
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    results: Optional[Dict[str, Any]] = None


@dataclass 
class TVProcessingResult:
    """Represents the result of a TV processing job"""
    job_id: str
    signal_file: str
    success: bool
    processing_time: float
    output_file: Optional[str] = None
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


class TVParallelProcessor:
    """Parallel processor for multiple TV signal files"""
    
    def __init__(self, max_workers: int = 4, heavydb_connection=None):
        """
        Initialize parallel processor
        
        Args:
            max_workers: Maximum number of parallel workers
            heavydb_connection: HeavyDB connection for query execution
        """
        self.max_workers = max_workers
        self.heavydb_connection = heavydb_connection
        
        # Initialize components
        self.parser = TVParser()
        self.signal_processor = SignalProcessor()
        self.query_builder = TVQueryBuilder()
        self.processor = TVProcessor()
        
        # Job management
        self.jobs: Dict[str, TVProcessingJob] = {}
        self.job_lock = threading.Lock()
        
        # Performance tracking
        self.total_jobs_processed = 0
        self.total_processing_time = 0.0
        self.failed_jobs = 0
        
        logger.info(f"TV Parallel Processor initialized with {max_workers} workers")
    
    def create_job(self, 
                   job_id: str, 
                   tv_config_path: str, 
                   signal_files: List[str], 
                   output_directory: str,
                   priority: int = 1) -> TVProcessingJob:
        """
        Create a new processing job
        
        Args:
            job_id: Unique identifier for the job
            tv_config_path: Path to TV configuration file
            signal_files: List of signal file paths to process
            output_directory: Directory for output files
            priority: Job priority (higher = more important)
            
        Returns:
            TVProcessingJob instance
        """
        job = TVProcessingJob(
            job_id=job_id,
            tv_config_path=tv_config_path,
            signal_files=signal_files,
            output_directory=output_directory,
            priority=priority
        )
        
        with self.job_lock:
            self.jobs[job_id] = job
        
        logger.info(f"Created job {job_id} with {len(signal_files)} signal files")
        return job
    
    def process_job(self, job: TVProcessingJob) -> List[TVProcessingResult]:
        """
        Process a single job with multiple signal files in parallel
        
        Args:
            job: TVProcessingJob to process
            
        Returns:
            List of TVProcessingResult for each signal file
        """
        logger.info(f"Starting job {job.job_id}")
        job.start_time = datetime.now()
        job.status = 'running'
        
        try:
            # Parse TV configuration once
            tv_config_result = self.parser.parse_tv_settings(job.tv_config_path)
            if not tv_config_result or 'settings' not in tv_config_result:
                raise ValueError(f"Failed to parse TV config: {job.tv_config_path}")
            
            tv_config = tv_config_result['settings'][0]  # Use first enabled setting
            
            # Create output directory
            output_dir = Path(job.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process signal files in parallel
            results = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all signal file processing tasks
                future_to_signal = {
                    executor.submit(
                        self._process_single_signal_file, 
                        job.job_id,
                        signal_file, 
                        tv_config, 
                        output_dir
                    ): signal_file 
                    for signal_file in job.signal_files
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_signal):
                    signal_file = future_to_signal[future]
                    try:
                        result = future.result()
                        results.append(result)
                        logger.info(f"Completed processing {signal_file} for job {job.job_id}")
                    except Exception as e:
                        error_result = TVProcessingResult(
                            job_id=job.job_id,
                            signal_file=signal_file,
                            success=False,
                            processing_time=0.0,
                            error_message=str(e)
                        )
                        results.append(error_result)
                        logger.error(f"Failed processing {signal_file} for job {job.job_id}: {e}")
            
            # Update job status
            job.end_time = datetime.now()
            job.status = 'completed'
            job.results = {
                'total_files': len(job.signal_files),
                'successful': sum(1 for r in results if r.success),
                'failed': sum(1 for r in results if not r.success),
                'total_time': (job.end_time - job.start_time).total_seconds(),
                'results': results
            }
            
            # Update statistics
            with self.job_lock:
                self.total_jobs_processed += 1
                self.total_processing_time += job.results['total_time']
                self.failed_jobs += job.results['failed']
            
            logger.info(f"Job {job.job_id} completed: {job.results['successful']}/{job.results['total_files']} successful")
            return results
            
        except Exception as e:
            job.end_time = datetime.now()
            job.status = 'failed'
            job.error_message = str(e)
            
            with self.job_lock:
                self.failed_jobs += 1
                
            logger.error(f"Job {job.job_id} failed: {e}")
            raise
    
    def _process_single_signal_file(self, 
                                    job_id: str,
                                    signal_file: str, 
                                    tv_config: Dict[str, Any], 
                                    output_dir: Path) -> TVProcessingResult:
        """
        Process a single signal file
        
        Args:
            job_id: Job identifier
            signal_file: Path to signal file
            tv_config: TV configuration dictionary
            output_dir: Output directory path
            
        Returns:
            TVProcessingResult
        """
        start_time = time.time()
        
        try:
            # Parse signals
            signals = self.parser.parse_signals(signal_file, tv_config['signal_date_format'])
            
            # Process signals
            processed_signals = self.signal_processor.process_signals(signals, tv_config)
            
            # Generate queries
            queries = []
            for signal in processed_signals:
                query = self.query_builder.build_query(signal, tv_config)
                queries.append((signal, query))
            
            # Execute queries if HeavyDB connection available
            results = []
            if self.heavydb_connection:
                for signal, query in queries:
                    try:
                        db_result = self.heavydb_connection.execute(query)
                        processed_result = self.processor.process_result(db_result, signal)
                        results.append(processed_result)
                    except Exception as e:
                        logger.warning(f"Query execution failed for signal {signal['trade_no']}: {e}")
                        # Continue with other signals
            
            # Generate output file
            signal_file_name = Path(signal_file).stem
            output_file = output_dir / f"{signal_file_name}_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Save results
            output_data = {
                'job_id': job_id,
                'signal_file': signal_file,
                'tv_config': tv_config['name'],
                'processing_timestamp': datetime.now().isoformat(),
                'signal_count': len(signals),
                'processed_signal_count': len(processed_signals),
                'query_count': len(queries),
                'results': results if results else [],
                'performance': {
                    'processing_time': time.time() - start_time,
                    'queries_per_second': len(queries) / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
                }
            }
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            
            processing_time = time.time() - start_time
            
            return TVProcessingResult(
                job_id=job_id,
                signal_file=signal_file,
                success=True,
                processing_time=processing_time,
                output_file=str(output_file),
                metrics={
                    'signal_count': len(signals),
                    'processed_signals': len(processed_signals),
                    'queries_generated': len(queries),
                    'results_count': len(results)
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return TVProcessingResult(
                job_id=job_id,
                signal_file=signal_file,
                success=False,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def process_multiple_jobs(self, jobs: List[TVProcessingJob]) -> Dict[str, List[TVProcessingResult]]:
        """
        Process multiple jobs in parallel
        
        Args:
            jobs: List of TVProcessingJob to process
            
        Returns:
            Dictionary mapping job_id to list of results
        """
        logger.info(f"Processing {len(jobs)} jobs in parallel")
        
        # Sort jobs by priority (higher priority first)
        sorted_jobs = sorted(jobs, key=lambda j: j.priority, reverse=True)
        
        all_results = {}
        
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(jobs))) as executor:
            # Submit all jobs
            future_to_job = {
                executor.submit(self.process_job, job): job 
                for job in sorted_jobs
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    results = future.result()
                    all_results[job.job_id] = results
                except Exception as e:
                    logger.error(f"Job {job.job_id} failed: {e}")
                    all_results[job.job_id] = []
        
        return all_results
    
    def get_job_status(self, job_id: str) -> Optional[TVProcessingJob]:
        """Get status of a specific job"""
        with self.job_lock:
            return self.jobs.get(job_id)
    
    def get_all_jobs(self) -> List[TVProcessingJob]:
        """Get all jobs"""
        with self.job_lock:
            return list(self.jobs.values())
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        with self.job_lock:
            return {
                'total_jobs_processed': self.total_jobs_processed,
                'total_processing_time': self.total_processing_time,
                'failed_jobs': self.failed_jobs,
                'success_rate': (self.total_jobs_processed - self.failed_jobs) / max(self.total_jobs_processed, 1) * 100,
                'average_job_time': self.total_processing_time / max(self.total_jobs_processed, 1),
                'active_workers': self.max_workers
            }
    
    def create_batch_job_from_directory(self, 
                                        job_id: str,
                                        tv_config_path: str,
                                        signal_directory: str,
                                        output_directory: str,
                                        file_pattern: str = "*.xlsx") -> TVProcessingJob:
        """
        Create a batch job from all signal files in a directory
        
        Args:
            job_id: Unique job identifier
            tv_config_path: Path to TV configuration file
            signal_directory: Directory containing signal files
            output_directory: Directory for output files
            file_pattern: Pattern to match signal files
            
        Returns:
            TVProcessingJob instance
        """
        signal_dir = Path(signal_directory)
        signal_files = list(signal_dir.glob(file_pattern))
        signal_file_paths = [str(f) for f in signal_files]
        
        logger.info(f"Found {len(signal_file_paths)} signal files in {signal_directory}")
        
        return self.create_job(
            job_id=job_id,
            tv_config_path=tv_config_path,
            signal_files=signal_file_paths,
            output_directory=output_directory
        )


class TVBatchProcessor:
    """High-level batch processor for TV strategies"""
    
    def __init__(self, max_workers: int = 4, heavydb_connection=None):
        """Initialize batch processor"""
        self.parallel_processor = TVParallelProcessor(max_workers, heavydb_connection)
        
    def process_signal_files_batch(self,
                                   tv_config_path: str,
                                   signal_files: List[str],
                                   output_directory: str) -> Dict[str, Any]:
        """
        Process multiple signal files as a batch
        
        Args:
            tv_config_path: Path to TV configuration file
            signal_files: List of signal file paths
            output_directory: Directory for output files
            
        Returns:
            Processing results summary
        """
        job_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create job
        job = self.parallel_processor.create_job(
            job_id=job_id,
            tv_config_path=tv_config_path,
            signal_files=signal_files,
            output_directory=output_directory
        )
        
        # Process job
        results = self.parallel_processor.process_job(job)
        
        # Generate summary
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        summary = {
            'job_id': job_id,
            'total_files': len(signal_files),
            'successful': len(successful_results),
            'failed': len(failed_results),
            'success_rate': len(successful_results) / len(signal_files) * 100 if signal_files else 0,
            'total_processing_time': sum(r.processing_time for r in results),
            'average_processing_time': sum(r.processing_time for r in results) / len(results) if results else 0,
            'output_files': [r.output_file for r in successful_results if r.output_file],
            'errors': [{'file': r.signal_file, 'error': r.error_message} for r in failed_results]
        }
        
        return summary
    
    def process_directory_batch(self,
                                tv_config_path: str,
                                signal_directory: str,
                                output_directory: str,
                                file_pattern: str = "*.xlsx") -> Dict[str, Any]:
        """
        Process all signal files in a directory
        
        Args:
            tv_config_path: Path to TV configuration file
            signal_directory: Directory containing signal files
            output_directory: Directory for output files
            file_pattern: Pattern to match signal files
            
        Returns:
            Processing results summary
        """
        # Find signal files
        signal_dir = Path(signal_directory)
        signal_files = [str(f) for f in signal_dir.glob(file_pattern)]
        
        if not signal_files:
            return {
                'error': f"No signal files found in {signal_directory} matching {file_pattern}",
                'total_files': 0,
                'successful': 0,
                'failed': 0
            }
        
        return self.process_signal_files_batch(
            tv_config_path=tv_config_path,
            signal_files=signal_files,
            output_directory=output_directory
        )