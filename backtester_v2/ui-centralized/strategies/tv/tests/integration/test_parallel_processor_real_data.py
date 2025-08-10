#!/usr/bin/env python3
"""
Integration Tests for TV Parallel Processor - REAL DATA VALIDATION
Tests parallel processing of multiple TV signal files with real HeavyDB validation
NO MOCK DATA - ONLY REAL INPUT SHEETS AND HEAVYDB CONNECTION
"""

import pytest
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import tempfile
import shutil

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parallel_processor import TVParallelProcessor, TVBatchProcessor, TVProcessingJob, TVProcessingResult
from parser import TVParser


class TestTVParallelProcessorRealData:
    """Test TV Parallel Processor with REAL data and HeavyDB"""
    
    def setup_method(self):
        """Setup for each test"""
        self.parser = TVParser()
        self.temp_dir = None
        
    def teardown_method(self):
        """Cleanup after each test"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @pytest.mark.integration
    @pytest.mark.heavydb
    def test_parallel_processor_initialization(self, heavydb_connection):
        """Test parallel processor initialization with HeavyDB connection"""
        # Test with default settings
        processor = TVParallelProcessor(max_workers=2, heavydb_connection=heavydb_connection)
        
        assert processor.max_workers == 2
        assert processor.heavydb_connection == heavydb_connection
        assert processor.total_jobs_processed == 0
        assert processor.total_processing_time == 0.0
        assert processor.failed_jobs == 0
        assert len(processor.jobs) == 0
        
        # Test components are initialized
        assert processor.parser is not None
        assert processor.signal_processor is not None
        assert processor.query_builder is not None
        assert processor.processor is not None
    
    @pytest.mark.integration
    @pytest.mark.heavydb
    def test_create_job_with_real_config(self, real_config_files, heavydb_connection):
        """Test job creation with real configuration files"""
        processor = TVParallelProcessor(max_workers=2, heavydb_connection=heavydb_connection)
        
        # Create temporary output directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create job with real files
        tv_master_path = str(real_config_files['tv_master'])
        signal_files = [str(real_config_files['signals'])]
        
        job = processor.create_job(
            job_id="test_job_001",
            tv_config_path=tv_master_path,
            signal_files=signal_files,
            output_directory=self.temp_dir,
            priority=1
        )
        
        # Validate job creation
        assert job.job_id == "test_job_001"
        assert job.tv_config_path == tv_master_path
        assert job.signal_files == signal_files
        assert job.output_directory == self.temp_dir
        assert job.priority == 1
        assert job.status == 'pending'
        assert job.start_time is None
        assert job.end_time is None
        
        # Validate job is stored in processor
        assert "test_job_001" in processor.jobs
        assert processor.jobs["test_job_001"] == job
    
    @pytest.mark.integration
    @pytest.mark.heavydb
    def test_process_single_job_with_real_data(self, real_config_files, heavydb_connection):
        """Test processing a single job with real configuration and signals"""
        processor = TVParallelProcessor(max_workers=2, heavydb_connection=heavydb_connection)
        
        # Create temporary output directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create job with real files
        tv_master_path = str(real_config_files['tv_master'])
        signal_files = [str(real_config_files['signals'])]
        
        job = processor.create_job(
            job_id="real_data_job",
            tv_config_path=tv_master_path,
            signal_files=signal_files,
            output_directory=self.temp_dir,
            priority=1
        )
        
        # Process the job
        start_time = time.time()
        results = processor.process_job(job)
        processing_time = time.time() - start_time
        
        # Validate processing results
        assert isinstance(results, list)
        assert len(results) == 1  # One signal file
        
        result = results[0]
        assert isinstance(result, TVProcessingResult)
        assert result.job_id == "real_data_job"
        assert result.signal_file == str(real_config_files['signals'])
        assert result.success is True
        assert result.processing_time > 0
        assert result.output_file is not None
        assert os.path.exists(result.output_file)
        
        # Validate job status updated
        assert job.status == 'completed'
        assert job.start_time is not None
        assert job.end_time is not None
        assert job.results is not None
        
        # Validate job results
        job_results = job.results
        assert job_results['total_files'] == 1
        assert job_results['successful'] == 1
        assert job_results['failed'] == 0
        assert job_results['total_time'] > 0
        
        # Validate performance requirement (<3 seconds)
        assert processing_time < 3.0, f"Processing took too long: {processing_time:.3f}s"
        
        # Validate output file content
        with open(result.output_file, 'r') as f:
            output_data = json.load(f)
        
        assert output_data['job_id'] == "real_data_job"
        assert output_data['signal_file'] == str(real_config_files['signals'])
        assert output_data['tv_config'] == 'TV_Backtest_Sample'
        assert output_data['signal_count'] == 4  # Real signals file has 4 signals
        assert 'processing_timestamp' in output_data
        assert 'performance' in output_data
    
    @pytest.mark.integration
    @pytest.mark.heavydb
    def test_parallel_processing_multiple_signal_files(self, real_config_files, heavydb_connection):
        """Test parallel processing with multiple copies of signal files"""
        processor = TVParallelProcessor(max_workers=3, heavydb_connection=heavydb_connection)
        
        # Create temporary directory and duplicate signal files
        self.temp_dir = tempfile.mkdtemp()
        signals_dir = Path(self.temp_dir) / "signals"
        signals_dir.mkdir()
        
        # Create multiple signal files by copying the real one
        original_signals = real_config_files['signals']
        signal_files = []
        
        for i in range(3):
            signal_copy = signals_dir / f"TV_SIGNALS_COPY_{i+1}.xlsx"
            shutil.copy2(original_signals, signal_copy)
            signal_files.append(str(signal_copy))
        
        # Create job with multiple signal files
        tv_master_path = str(real_config_files['tv_master'])
        output_dir = Path(self.temp_dir) / "output"
        output_dir.mkdir()
        
        job = processor.create_job(
            job_id="parallel_test_job",
            tv_config_path=tv_master_path,
            signal_files=signal_files,
            output_directory=str(output_dir),
            priority=1
        )
        
        # Process the job
        start_time = time.time()
        results = processor.process_job(job)
        processing_time = time.time() - start_time
        
        # Validate parallel processing results
        assert len(results) == 3  # All signal files processed
        
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        assert len(successful_results) == 3  # All should succeed
        assert len(failed_results) == 0
        
        # Validate each result
        for i, result in enumerate(successful_results):
            assert result.job_id == "parallel_test_job"
            assert result.signal_file in signal_files
            assert result.success is True
            assert result.processing_time > 0
            assert result.output_file is not None
            assert os.path.exists(result.output_file)
            
            # Validate metrics
            assert 'signal_count' in result.metrics
            assert result.metrics['signal_count'] == 4  # Each copy has 4 signals
            assert 'processed_signals' in result.metrics
            assert 'queries_generated' in result.metrics
        
        # Validate performance (parallel should be faster than sequential)
        # With 3 workers, should be significantly faster than 3x sequential time
        assert processing_time < 5.0, f"Parallel processing took too long: {processing_time:.3f}s"
        
        # Validate job statistics updated
        assert processor.total_jobs_processed == 1
        assert processor.total_processing_time > 0
        assert processor.failed_jobs == 0
    
    @pytest.mark.integration
    @pytest.mark.heavydb
    def test_batch_processor_with_real_data(self, real_config_files, heavydb_connection):
        """Test high-level batch processor with real data"""
        batch_processor = TVBatchProcessor(max_workers=2, heavydb_connection=heavydb_connection)
        
        # Create temporary directory with signal files
        self.temp_dir = tempfile.mkdtemp()
        signals_dir = Path(self.temp_dir) / "signals"
        signals_dir.mkdir()
        output_dir = Path(self.temp_dir) / "output"
        output_dir.mkdir()
        
        # Create multiple signal files
        original_signals = real_config_files['signals']
        signal_files = []
        
        for i in range(2):
            signal_copy = signals_dir / f"BATCH_SIGNALS_{i+1}.xlsx"
            shutil.copy2(original_signals, signal_copy)
            signal_files.append(str(signal_copy))
        
        # Process batch
        tv_master_path = str(real_config_files['tv_master'])
        
        start_time = time.time()
        summary = batch_processor.process_signal_files_batch(
            tv_config_path=tv_master_path,
            signal_files=signal_files,
            output_directory=str(output_dir)
        )
        processing_time = time.time() - start_time
        
        # Validate batch processing summary
        assert isinstance(summary, dict)
        assert summary['total_files'] == 2
        assert summary['successful'] == 2
        assert summary['failed'] == 0
        assert summary['success_rate'] == 100.0
        assert summary['total_processing_time'] > 0
        assert summary['average_processing_time'] > 0
        assert len(summary['output_files']) == 2
        assert len(summary['errors']) == 0
        
        # Validate all output files exist
        for output_file in summary['output_files']:
            assert os.path.exists(output_file)
            
            # Validate output file content
            with open(output_file, 'r') as f:
                output_data = json.load(f)
            
            assert 'job_id' in output_data
            assert output_data['signal_count'] == 4
            assert output_data['tv_config'] == 'TV_Backtest_Sample'
        
        # Validate performance
        assert processing_time < 4.0, f"Batch processing took too long: {processing_time:.3f}s"
    
    @pytest.mark.integration
    @pytest.mark.heavydb
    def test_batch_processor_from_directory(self, real_config_files, heavydb_connection):
        """Test batch processor with directory scanning"""
        batch_processor = TVBatchProcessor(max_workers=2, heavydb_connection=heavydb_connection)
        
        # Create temporary directory with signal files
        self.temp_dir = tempfile.mkdtemp()
        signals_dir = Path(self.temp_dir) / "signals"
        signals_dir.mkdir()
        output_dir = Path(self.temp_dir) / "output"
        output_dir.mkdir()
        
        # Create signal files in directory
        original_signals = real_config_files['signals']
        
        for i in range(3):
            signal_copy = signals_dir / f"DIR_SIGNALS_{i+1}.xlsx"
            shutil.copy2(original_signals, signal_copy)
        
        # Process directory batch
        tv_master_path = str(real_config_files['tv_master'])
        
        summary = batch_processor.process_directory_batch(
            tv_config_path=tv_master_path,
            signal_directory=str(signals_dir),
            output_directory=str(output_dir),
            file_pattern="*.xlsx"
        )
        
        # Validate directory batch processing
        assert summary['total_files'] == 3
        assert summary['successful'] == 3
        assert summary['failed'] == 0
        assert summary['success_rate'] == 100.0
        assert len(summary['output_files']) == 3
        assert len(summary['errors']) == 0
    
    @pytest.mark.integration
    @pytest.mark.heavydb
    def test_error_handling_invalid_signal_file(self, real_config_files, heavydb_connection):
        """Test error handling with invalid signal file"""
        processor = TVParallelProcessor(max_workers=2, heavydb_connection=heavydb_connection)
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create invalid signal file
        invalid_signal_file = Path(self.temp_dir) / "invalid_signals.xlsx"
        invalid_signal_file.write_text("This is not a valid Excel file")
        
        # Mix valid and invalid files
        tv_master_path = str(real_config_files['tv_master'])
        signal_files = [
            str(real_config_files['signals']),  # Valid
            str(invalid_signal_file)             # Invalid
        ]
        
        job = processor.create_job(
            job_id="error_test_job",
            tv_config_path=tv_master_path,
            signal_files=signal_files,
            output_directory=self.temp_dir,
            priority=1
        )
        
        # Process job (should handle errors gracefully)
        results = processor.process_job(job)
        
        # Validate error handling
        assert len(results) == 2
        
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        assert len(successful_results) == 1  # Valid file processed
        assert len(failed_results) == 1     # Invalid file failed
        
        # Validate failed result
        failed_result = failed_results[0]
        assert failed_result.signal_file == str(invalid_signal_file)
        assert failed_result.success is False
        assert failed_result.error_message is not None
        assert failed_result.output_file is None
        
        # Validate successful result
        successful_result = successful_results[0]
        assert successful_result.signal_file == str(real_config_files['signals'])
        assert successful_result.success is True
        assert successful_result.output_file is not None
    
    @pytest.mark.integration
    @pytest.mark.heavydb
    def test_performance_stats_tracking(self, real_config_files, heavydb_connection):
        """Test performance statistics tracking"""
        processor = TVParallelProcessor(max_workers=2, heavydb_connection=heavydb_connection)
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Process multiple jobs to track stats
        tv_master_path = str(real_config_files['tv_master'])
        signal_files = [str(real_config_files['signals'])]
        
        # Job 1
        job1 = processor.create_job("stats_job_1", tv_master_path, signal_files, self.temp_dir)
        results1 = processor.process_job(job1)
        
        # Job 2
        job2 = processor.create_job("stats_job_2", tv_master_path, signal_files, self.temp_dir)
        results2 = processor.process_job(job2)
        
        # Get performance stats
        stats = processor.get_performance_stats()
        
        # Validate stats
        assert stats['total_jobs_processed'] == 2
        assert stats['total_processing_time'] > 0
        assert stats['failed_jobs'] == 0
        assert stats['success_rate'] == 100.0
        assert stats['average_job_time'] > 0
        assert stats['active_workers'] == 2
        
        # Validate job status retrieval
        job1_status = processor.get_job_status("stats_job_1")
        assert job1_status is not None
        assert job1_status.status == 'completed'
        
        # Validate all jobs retrieval
        all_jobs = processor.get_all_jobs()
        assert len(all_jobs) == 2
        job_ids = {job.job_id for job in all_jobs}
        assert job_ids == {"stats_job_1", "stats_job_2"}


class TestTVParallelProcessorPerformance:
    """Performance tests for TV Parallel Processor"""
    
    def setup_method(self):
        """Setup for each test"""
        self.temp_dir = None
        
    def teardown_method(self):
        """Cleanup after each test"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @pytest.mark.performance
    @pytest.mark.heavydb
    def test_parallel_vs_sequential_performance(self, real_config_files, heavydb_connection):
        """Test parallel processing performance vs sequential"""
        # Create multiple signal files for testing
        self.temp_dir = tempfile.mkdtemp()
        signals_dir = Path(self.temp_dir) / "signals"
        signals_dir.mkdir()
        output_dir = Path(self.temp_dir) / "output"
        output_dir.mkdir()
        
        original_signals = real_config_files['signals']
        signal_files = []
        
        # Create 4 signal files for better parallel testing
        for i in range(4):
            signal_copy = signals_dir / f"PERF_SIGNALS_{i+1}.xlsx"
            shutil.copy2(original_signals, signal_copy)
            signal_files.append(str(signal_copy))
        
        tv_master_path = str(real_config_files['tv_master'])
        
        # Test sequential processing (1 worker)
        sequential_processor = TVParallelProcessor(max_workers=1, heavydb_connection=heavydb_connection)
        sequential_job = sequential_processor.create_job(
            "sequential_job", tv_master_path, signal_files, str(output_dir)
        )
        
        start_time = time.time()
        sequential_results = sequential_processor.process_job(sequential_job)
        sequential_time = time.time() - start_time
        
        # Clear output directory
        shutil.rmtree(output_dir)
        output_dir.mkdir()
        
        # Test parallel processing (4 workers)
        parallel_processor = TVParallelProcessor(max_workers=4, heavydb_connection=heavydb_connection)
        parallel_job = parallel_processor.create_job(
            "parallel_job", tv_master_path, signal_files, str(output_dir)
        )
        
        start_time = time.time()
        parallel_results = parallel_processor.process_job(parallel_job)
        parallel_time = time.time() - start_time
        
        # Validate both succeeded
        assert len(sequential_results) == 4
        assert len(parallel_results) == 4
        assert all(r.success for r in sequential_results)
        assert all(r.success for r in parallel_results)
        
        # Parallel should be faster (or at least not significantly slower)
        speedup_ratio = sequential_time / parallel_time
        print(f"Sequential time: {sequential_time:.3f}s")
        print(f"Parallel time: {parallel_time:.3f}s")
        print(f"Speedup ratio: {speedup_ratio:.2f}x")
        
        # With 4 workers and 4 files, we should see some speedup
        # Even if there's overhead, parallel shouldn't be more than 2x slower
        assert speedup_ratio > 0.5, f"Parallel processing is too slow: {speedup_ratio:.2f}x"
    
    @pytest.mark.performance
    @pytest.mark.heavydb
    def test_memory_usage_under_load(self, real_config_files, heavydb_connection):
        """Test memory usage during heavy parallel processing"""
        import psutil
        import gc
        
        # Monitor memory before test
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large number of signal files
        self.temp_dir = tempfile.mkdtemp()
        signals_dir = Path(self.temp_dir) / "signals"
        signals_dir.mkdir()
        output_dir = Path(self.temp_dir) / "output"
        output_dir.mkdir()
        
        original_signals = real_config_files['signals']
        signal_files = []
        
        # Create 10 signal files for memory testing
        for i in range(10):
            signal_copy = signals_dir / f"MEMORY_SIGNALS_{i+1}.xlsx"
            shutil.copy2(original_signals, signal_copy)
            signal_files.append(str(signal_copy))
        
        # Process with multiple workers
        processor = TVParallelProcessor(max_workers=4, heavydb_connection=heavydb_connection)
        tv_master_path = str(real_config_files['tv_master'])
        
        job = processor.create_job(
            "memory_test_job", tv_master_path, signal_files, str(output_dir)
        )
        
        # Process and monitor memory
        start_time = time.time()
        results = processor.process_job(job)
        processing_time = time.time() - start_time
        
        # Check memory after processing
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Force garbage collection
        gc.collect()
        after_gc_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Initial memory: {initial_memory:.1f} MB")
        print(f"Final memory: {final_memory:.1f} MB")
        print(f"After GC memory: {after_gc_memory:.1f} MB")
        print(f"Memory increase: {memory_increase:.1f} MB")
        print(f"Processing time: {processing_time:.3f}s")
        
        # Validate processing succeeded
        assert len(results) == 10
        assert all(r.success for r in results)
        
        # Memory increase should be reasonable (less than 500MB for this test)
        assert memory_increase < 500, f"Memory usage too high: {memory_increase:.1f} MB"
        
        # Processing should complete in reasonable time
        assert processing_time < 15.0, f"Processing took too long: {processing_time:.3f}s"