#!/usr/bin/env python3
"""
Enhanced Logging System for Market Regime 1-Year Analysis
========================================================
Comprehensive logging framework for tracking regime analysis,
performance metrics, and system health during long-duration tests.

Features:
- Structured logging with JSON format
- Real-time metrics collection
- Performance tracking
- Regime transition logging
- Error tracking and recovery
- Log aggregation and analysis

Author: Market Regime Testing Team
Date: 2025-06-27
"""

import logging
import json
import time
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
import queue
import psutil
import numpy as np
from enum import Enum

class LogLevel(Enum):
    """Log levels for regime analysis"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    REGIME_CHANGE = "REGIME_CHANGE"
    PERFORMANCE = "PERFORMANCE"
    ANALYSIS = "ANALYSIS"

@dataclass
class RegimeTransition:
    """Regime transition event"""
    timestamp: datetime
    from_regime: str
    to_regime: str
    confidence_before: float
    confidence_after: float
    transition_score: float
    component_contributions: Dict[str, float]
    market_conditions: Dict[str, Any]

@dataclass
class PerformanceMetrics:
    """Performance tracking metrics"""
    timestamp: datetime
    processing_time_ms: float
    records_processed: int
    memory_usage_mb: float
    cpu_usage_percent: float
    queue_size: int
    cache_hit_rate: float
    error_count: int

class EnhancedMarketRegimeLogger:
    """Enhanced logging system for Market Regime analysis"""
    
    def __init__(self, 
                 log_dir: str = "market_regime_logs",
                 buffer_size: int = 10000,
                 rotation_size_mb: int = 100,
                 enable_compression: bool = True):
        """
        Initialize enhanced logging system
        
        Args:
            log_dir: Directory for log files
            buffer_size: Size of log buffer
            rotation_size_mb: Max size before rotation
            enable_compression: Compress rotated logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.regime_logs_dir = self.log_dir / "regime_transitions"
        self.performance_logs_dir = self.log_dir / "performance"
        self.error_logs_dir = self.log_dir / "errors"
        self.analysis_logs_dir = self.log_dir / "analysis"
        
        for dir_path in [self.regime_logs_dir, self.performance_logs_dir, 
                        self.error_logs_dir, self.analysis_logs_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Initialize loggers
        self._setup_loggers()
        
        # Buffering and performance
        self.buffer_size = buffer_size
        self.log_buffer = deque(maxlen=buffer_size)
        self.metrics_buffer = deque(maxlen=1000)
        
        # Threading for async logging
        self.log_queue = queue.Queue()
        self.logging_thread = threading.Thread(target=self._logging_worker, daemon=True)
        self.logging_thread.start()
        
        # Performance tracking
        self.performance_tracker = {
            'start_time': datetime.now(),
            'total_records': 0,
            'regime_changes': 0,
            'errors': 0,
            'warnings': 0
        }
        
        # Regime analysis tracking
        self.regime_history = []
        self.regime_durations = defaultdict(list)
        self.transition_matrix = defaultdict(lambda: defaultdict(int))
        
        # Real-time metrics
        self.realtime_metrics = {
            'current_regime': None,
            'last_transition': None,
            'avg_confidence': 0,
            'processing_rate': 0
        }
        
        # Configuration
        self.rotation_size_mb = rotation_size_mb
        self.enable_compression = enable_compression
        
    def _setup_loggers(self):
        """Setup specialized loggers"""
        # Main logger
        self.logger = logging.getLogger('MarketRegime')
        self.logger.setLevel(logging.DEBUG)
        
        # Create formatters
        json_formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"module": "%(name)s", "message": %(message)s}'
        )
        
        # File handlers for different log types
        handlers = {
            'main': logging.FileHandler(
                self.log_dir / f"market_regime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            ),
            'regime': logging.FileHandler(
                self.regime_logs_dir / f"regime_transitions_{datetime.now().strftime('%Y%m%d')}.jsonl"
            ),
            'performance': logging.FileHandler(
                self.performance_logs_dir / f"performance_{datetime.now().strftime('%Y%m%d')}.jsonl"
            ),
            'error': logging.FileHandler(
                self.error_logs_dir / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
            )
        }
        
        for handler in handlers.values():
            handler.setFormatter(json_formatter)
            self.logger.addHandler(handler)
        
        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
    def log_regime_transition(self, transition: RegimeTransition):
        """Log regime transition event"""
        self.regime_history.append(transition)
        self.performance_tracker['regime_changes'] += 1
        
        # Update transition matrix
        self.transition_matrix[transition.from_regime][transition.to_regime] += 1
        
        # Log to file
        log_entry = {
            'type': 'REGIME_TRANSITION',
            'timestamp': transition.timestamp.isoformat(),
            'data': asdict(transition)
        }
        
        self.log_queue.put(('regime', json.dumps(log_entry)))
        
        # Update real-time metrics
        self.realtime_metrics['current_regime'] = transition.to_regime
        self.realtime_metrics['last_transition'] = transition.timestamp
        
    def log_performance_metrics(self, metrics: PerformanceMetrics):
        """Log performance metrics"""
        self.metrics_buffer.append(metrics)
        
        log_entry = {
            'type': 'PERFORMANCE_METRICS',
            'timestamp': metrics.timestamp.isoformat(),
            'data': asdict(metrics)
        }
        
        self.log_queue.put(('performance', json.dumps(log_entry)))
        
        # Update processing rate
        if len(self.metrics_buffer) > 10:
            recent_metrics = list(self.metrics_buffer)[-10:]
            total_records = sum(m.records_processed for m in recent_metrics)
            total_time = sum(m.processing_time_ms for m in recent_metrics) / 1000
            self.realtime_metrics['processing_rate'] = total_records / total_time if total_time > 0 else 0
    
    def log_analysis_checkpoint(self, 
                              checkpoint_name: str,
                              records_processed: int,
                              regime_distribution: Dict[str, int],
                              confidence_stats: Dict[str, float],
                              custom_data: Optional[Dict] = None):
        """Log analysis checkpoint for recovery and monitoring"""
        checkpoint = {
            'type': 'ANALYSIS_CHECKPOINT',
            'timestamp': datetime.now().isoformat(),
            'checkpoint_name': checkpoint_name,
            'records_processed': records_processed,
            'regime_distribution': regime_distribution,
            'confidence_stats': confidence_stats,
            'custom_data': custom_data or {}
        }
        
        self.log_queue.put(('analysis', json.dumps(checkpoint)))
        
    def log_error(self, error_type: str, error_message: str, 
                  stack_trace: Optional[str] = None,
                  context: Optional[Dict] = None):
        """Log error with context"""
        self.performance_tracker['errors'] += 1
        
        error_entry = {
            'type': 'ERROR',
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'message': error_message,
            'stack_trace': stack_trace,
            'context': context or {}
        }
        
        self.log_queue.put(('error', json.dumps(error_entry)))
        
    def log_warning(self, warning_type: str, message: str, 
                   context: Optional[Dict] = None):
        """Log warning"""
        self.performance_tracker['warnings'] += 1
        
        warning_entry = {
            'type': 'WARNING',
            'timestamp': datetime.now().isoformat(),
            'warning_type': warning_type,
            'message': message,
            'context': context or {}
        }
        
        self.log_queue.put(('main', json.dumps(warning_entry)))
    
    def _logging_worker(self):
        """Background worker for async logging"""
        handlers = {
            'main': self.log_dir / f"market_regime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            'regime': self.regime_logs_dir / f"regime_transitions_{datetime.now().strftime('%Y%m%d')}.jsonl",
            'performance': self.performance_logs_dir / f"performance_{datetime.now().strftime('%Y%m%d')}.jsonl",
            'error': self.error_logs_dir / f"errors_{datetime.now().strftime('%Y%m%d')}.log",
            'analysis': self.analysis_logs_dir / f"analysis_{datetime.now().strftime('%Y%m%d')}.jsonl"
        }
        
        file_handles = {}
        
        try:
            while True:
                try:
                    log_type, message = self.log_queue.get(timeout=1)
                    
                    # Get or create file handle
                    if log_type not in file_handles:
                        file_handles[log_type] = open(handlers[log_type], 'a')
                    
                    # Write log entry
                    file_handles[log_type].write(message + '\n')
                    file_handles[log_type].flush()
                    
                    # Check rotation
                    self._check_rotation(file_handles, handlers)
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Logging error: {e}")
                    
        finally:
            # Close all file handles
            for handle in file_handles.values():
                handle.close()
    
    def _check_rotation(self, file_handles: Dict, handlers: Dict):
        """Check and perform log rotation if needed"""
        for log_type, handle in file_handles.items():
            file_path = Path(handlers[log_type])
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                if size_mb > self.rotation_size_mb:
                    # Close current file
                    handle.close()
                    
                    # Rotate file
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    rotated_path = file_path.with_suffix(f'.{timestamp}.log')
                    file_path.rename(rotated_path)
                    
                    # Compress if enabled
                    if self.enable_compression:
                        import gzip
                        with open(rotated_path, 'rb') as f_in:
                            with gzip.open(f"{rotated_path}.gz", 'wb') as f_out:
                                f_out.writelines(f_in)
                        rotated_path.unlink()
                    
                    # Open new file
                    file_handles[log_type] = open(file_path, 'a')
    
    def get_regime_analysis_summary(self) -> Dict[str, Any]:
        """Get comprehensive regime analysis summary"""
        if not self.regime_history:
            return {'error': 'No regime history available'}
        
        # Calculate regime durations
        current_regime = None
        regime_start = None
        
        for transition in self.regime_history:
            if current_regime and regime_start:
                duration = (transition.timestamp - regime_start).total_seconds()
                self.regime_durations[current_regime].append(duration)
            
            current_regime = transition.to_regime
            regime_start = transition.timestamp
        
        # Calculate statistics
        regime_stats = {}
        for regime, durations in self.regime_durations.items():
            if durations:
                regime_stats[regime] = {
                    'count': len(durations),
                    'total_duration': sum(durations),
                    'avg_duration': np.mean(durations),
                    'max_duration': max(durations),
                    'min_duration': min(durations),
                    'std_duration': np.std(durations)
                }
        
        # Confidence analysis
        all_confidences = [t.confidence_after for t in self.regime_history]
        confidence_stats = {
            'mean': np.mean(all_confidences),
            'std': np.std(all_confidences),
            'min': min(all_confidences),
            'max': max(all_confidences),
            'percentiles': {
                '25': np.percentile(all_confidences, 25),
                '50': np.percentile(all_confidences, 50),
                '75': np.percentile(all_confidences, 75)
            }
        }
        
        return {
            'summary': {
                'total_transitions': len(self.regime_history),
                'unique_regimes': len(self.regime_durations),
                'analysis_duration': (datetime.now() - self.performance_tracker['start_time']).total_seconds(),
                'total_records': self.performance_tracker['total_records']
            },
            'regime_statistics': regime_stats,
            'confidence_analysis': confidence_stats,
            'transition_matrix': dict(self.transition_matrix),
            'performance': {
                'errors': self.performance_tracker['errors'],
                'warnings': self.performance_tracker['warnings'],
                'avg_processing_rate': self.realtime_metrics['processing_rate']
            }
        }
    
    def export_analysis_report(self, output_path: str):
        """Export comprehensive analysis report"""
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'analysis_duration': (datetime.now() - self.performance_tracker['start_time']).total_seconds(),
                'log_directory': str(self.log_dir)
            },
            'analysis_summary': self.get_regime_analysis_summary(),
            'regime_history': [asdict(t) for t in self.regime_history[-1000:]],  # Last 1000 transitions
            'performance_metrics': [asdict(m) for m in list(self.metrics_buffer)],
            'realtime_status': self.realtime_metrics
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Analysis report exported to: {output_path}")
    
    def close(self):
        """Close logging system and export final report"""
        # Export final report
        final_report_path = self.log_dir / f"final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.export_analysis_report(str(final_report_path))
        
        # Signal logging thread to stop
        self.log_queue.put(None)
        self.logging_thread.join(timeout=5)
        
        self.logger.info("Enhanced logging system closed")

# Singleton instance
_logger_instance = None

def get_logger() -> EnhancedMarketRegimeLogger:
    """Get singleton logger instance"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = EnhancedMarketRegimeLogger()
    return _logger_instance

# Example usage
if __name__ == "__main__":
    # Initialize logger
    logger = get_logger()
    
    # Log regime transition
    transition = RegimeTransition(
        timestamp=datetime.now(),
        from_regime="Neutral_Consolidation",
        to_regime="Bullish_Momentum",
        confidence_before=0.75,
        confidence_after=0.82,
        transition_score=0.91,
        component_contributions={
            'volatility': 0.3,
            'trend': 0.5,
            'structure': 0.2
        },
        market_conditions={
            'vix': 15.2,
            'spot_price': 19500,
            'volume': 125000
        }
    )
    logger.log_regime_transition(transition)
    
    # Log performance metrics
    metrics = PerformanceMetrics(
        timestamp=datetime.now(),
        processing_time_ms=125.5,
        records_processed=1000,
        memory_usage_mb=512.3,
        cpu_usage_percent=45.2,
        queue_size=250,
        cache_hit_rate=0.85,
        error_count=0
    )
    logger.log_performance_metrics(metrics)
    
    # Log analysis checkpoint
    logger.log_analysis_checkpoint(
        checkpoint_name="hourly_checkpoint",
        records_processed=50000,
        regime_distribution={
            'Bullish_Momentum': 15000,
            'Neutral_Consolidation': 20000,
            'Bearish_Momentum': 15000
        },
        confidence_stats={
            'mean': 0.78,
            'std': 0.12,
            'min': 0.45,
            'max': 0.95
        }
    )
    
    # Get summary
    summary = logger.get_regime_analysis_summary()
    print(json.dumps(summary, indent=2))
    
    # Close logger
    logger.close()