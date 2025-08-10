"""
Hot Reload Configuration System
Real-time configuration updates with file watching and change detection
Performance Target: <50ms change detection and reload
"""

import os
import time
import threading
import asyncio
from pathlib import Path
from typing import Dict, List, Callable, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import logging
from collections import defaultdict
import hashlib
import json

# File watching
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent

# Configuration system
from ..converters.excel_to_yaml import ExcelToYAMLConverter
from ..core.config_manager import ConfigurationManager
from ..core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

@dataclass
class ChangeEvent:
    """Represents a configuration change event"""
    file_path: str
    event_type: str  # 'modified', 'created', 'deleted'
    timestamp: datetime
    strategy_type: str
    config_name: str
    old_hash: Optional[str] = None
    new_hash: Optional[str] = None
    processing_time: float = 0.0
    success: bool = False
    error_message: Optional[str] = None

@dataclass
class ReloadStats:
    """Statistics for hot reload system"""
    total_events: int = 0
    successful_reloads: int = 0
    failed_reloads: int = 0
    avg_processing_time: float = 0.0
    max_processing_time: float = 0.0
    min_processing_time: float = float('inf')
    events_by_type: Dict[str, int] = field(default_factory=dict)
    events_by_strategy: Dict[str, int] = field(default_factory=dict)

class ConfigurationFileHandler(FileSystemEventHandler):
    """File system event handler for configuration files"""
    
    def __init__(self, hot_reload_system: 'HotReloadSystem'):
        self.hot_reload_system = hot_reload_system
        self.debounce_time = 0.1  # 100ms debounce
        self.pending_events = {}
        self.debounce_timer = None
        
    def on_modified(self, event):
        """Handle file modification events"""
        if not event.is_directory and self._is_config_file(event.src_path):
            self._debounce_event(event.src_path, 'modified')
    
    def on_created(self, event):
        """Handle file creation events"""
        if not event.is_directory and self._is_config_file(event.src_path):
            self._debounce_event(event.src_path, 'created')
    
    def on_deleted(self, event):
        """Handle file deletion events"""
        if not event.is_directory and self._is_config_file(event.src_path):
            self._debounce_event(event.src_path, 'deleted')
    
    def _is_config_file(self, file_path: str) -> bool:
        """Check if file is a configuration file"""
        path = Path(file_path)
        return path.suffix.lower() in ['.xlsx', '.xls', '.xlsm', '.yml', '.yaml', '.json']
    
    def _debounce_event(self, file_path: str, event_type: str):
        """Debounce file system events to avoid duplicate processing"""
        self.pending_events[file_path] = {
            'type': event_type,
            'timestamp': time.time()
        }
        
        # Cancel existing timer
        if self.debounce_timer:
            self.debounce_timer.cancel()
        
        # Start new timer
        self.debounce_timer = threading.Timer(
            self.debounce_time,
            self._process_pending_events
        )
        self.debounce_timer.start()
    
    def _process_pending_events(self):
        """Process all pending file system events"""
        events_to_process = dict(self.pending_events)
        self.pending_events.clear()
        
        for file_path, event_data in events_to_process.items():
            try:
                self.hot_reload_system._handle_file_change(
                    file_path,
                    event_data['type'],
                    event_data['timestamp']
                )
            except Exception as e:
                logger.error(f"Error processing file change event for {file_path}: {e}")

class HotReloadSystem:
    """
    Hot reload system for configuration files
    
    Features:
    - Real-time file monitoring with watchdog
    - Debounced change detection (<50ms)
    - Automatic YAML conversion for Excel files
    - Configuration validation and error handling
    - Performance monitoring and statistics
    - Callback system for change notifications
    - Thread-safe operations
    """
    
    def __init__(self, config_manager: ConfigurationManager, watch_paths: List[str]):
        """
        Initialize hot reload system
        
        Args:
            config_manager: Configuration manager instance
            watch_paths: List of directories to watch for changes
        """
        self.config_manager = config_manager
        self.watch_paths = [Path(p) for p in watch_paths]
        self.converter = ExcelToYAMLConverter()
        
        # File system monitoring
        self.observer = Observer()
        self.file_handler = ConfigurationFileHandler(self)
        self.watching = False
        
        # Change tracking
        self.file_hashes = {}
        self.change_history = []
        self.stats = ReloadStats()
        
        # Callbacks
        self.callbacks = defaultdict(list)  # {event_type: [callbacks]}
        self.global_callbacks = []
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.lock = threading.RLock()
        
        # Configuration
        self.max_history_size = 1000
        self.auto_convert_excel = True
        self.validate_on_reload = True
        
        logger.info(f"HotReloadSystem initialized with {len(watch_paths)} watch paths")
    
    def start_watching(self) -> None:
        """Start file system monitoring"""
        if self.watching:
            logger.warning("Hot reload system is already watching")
            return
        
        try:
            # Schedule observers for all watch paths
            for watch_path in self.watch_paths:
                if watch_path.exists():
                    self.observer.schedule(
                        self.file_handler,
                        str(watch_path),
                        recursive=True
                    )
                    logger.info(f"Watching directory: {watch_path}")
                else:
                    logger.warning(f"Watch path does not exist: {watch_path}")
            
            # Start observer
            self.observer.start()
            self.watching = True
            
            # Initialize file hashes
            self._initialize_file_hashes()
            
            logger.info("Hot reload system started")
            
        except Exception as e:
            logger.error(f"Failed to start hot reload system: {e}")
            raise
    
    def stop_watching(self) -> None:
        """Stop file system monitoring"""
        if not self.watching:
            return
        
        try:
            self.observer.stop()
            self.observer.join(timeout=5.0)
            self.watching = False
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            logger.info("Hot reload system stopped")
            
        except Exception as e:
            logger.error(f"Error stopping hot reload system: {e}")
    
    def add_callback(self, event_type: str, callback: Callable[[ChangeEvent], None]) -> str:
        """
        Add callback for specific event types
        
        Args:
            event_type: Event type ('modified', 'created', 'deleted', 'all')
            callback: Callback function
            
        Returns:
            Callback ID for removal
        """
        with self.lock:
            callback_id = f"{event_type}_{len(self.callbacks[event_type])}_{time.time()}"
            self.callbacks[event_type].append((callback_id, callback))
            
            logger.debug(f"Added callback for {event_type}: {callback_id}")
            return callback_id
    
    def remove_callback(self, callback_id: str) -> bool:
        """
        Remove callback by ID
        
        Args:
            callback_id: Callback ID returned by add_callback
            
        Returns:
            True if callback was found and removed
        """
        with self.lock:
            for event_type, callbacks in self.callbacks.items():
                for i, (cid, callback) in enumerate(callbacks):
                    if cid == callback_id:
                        del callbacks[i]
                        logger.debug(f"Removed callback: {callback_id}")
                        return True
            return False
    
    def add_global_callback(self, callback: Callable[[ChangeEvent], None]) -> None:
        """Add global callback that receives all events"""
        with self.lock:
            self.global_callbacks.append(callback)
    
    def remove_global_callback(self, callback: Callable[[ChangeEvent], None]) -> None:
        """Remove global callback"""
        with self.lock:
            if callback in self.global_callbacks:
                self.global_callbacks.remove(callback)
    
    def force_reload(self, file_path: str) -> ChangeEvent:
        """
        Force reload of a specific configuration file
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            Change event result
        """
        return self._handle_file_change(file_path, 'forced', time.time())
    
    def reload_strategy(self, strategy_type: str) -> List[ChangeEvent]:
        """
        Reload all configuration files for a strategy
        
        Args:
            strategy_type: Strategy type to reload
            
        Returns:
            List of change events
        """
        events = []
        
        for watch_path in self.watch_paths:
            strategy_dir = watch_path / "prod" / strategy_type
            if strategy_dir.exists():
                for file_path in strategy_dir.glob("*.xlsx"):
                    event = self.force_reload(str(file_path))
                    events.append(event)
        
        return events
    
    def get_file_status(self, file_path: str) -> Dict[str, Any]:
        """
        Get status information for a configuration file
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            Status information
        """
        path = Path(file_path)
        
        if not path.exists():
            return {
                'exists': False,
                'file_path': file_path,
                'error': 'File does not exist'
            }
        
        try:
            stat = path.stat()
            file_hash = self._get_file_hash(file_path)
            
            return {
                'exists': True,
                'file_path': file_path,
                'size': stat.st_size,
                'modified_time': datetime.fromtimestamp(stat.st_mtime),
                'hash': file_hash,
                'is_watched': any(path.is_relative_to(wp) for wp in self.watch_paths),
                'strategy_type': self._extract_strategy_type(file_path),
                'config_name': path.stem
            }
            
        except Exception as e:
            return {
                'exists': True,
                'file_path': file_path,
                'error': str(e)
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get hot reload system statistics"""
        with self.lock:
            return {
                'watching': self.watching,
                'watch_paths': [str(p) for p in self.watch_paths],
                'tracked_files': len(self.file_hashes),
                'change_history_size': len(self.change_history),
                'stats': {
                    'total_events': self.stats.total_events,
                    'successful_reloads': self.stats.successful_reloads,
                    'failed_reloads': self.stats.failed_reloads,
                    'success_rate': self.stats.successful_reloads / max(self.stats.total_events, 1),
                    'avg_processing_time': self.stats.avg_processing_time,
                    'max_processing_time': self.stats.max_processing_time,
                    'min_processing_time': self.stats.min_processing_time if self.stats.min_processing_time != float('inf') else 0,
                    'events_by_type': dict(self.stats.events_by_type),
                    'events_by_strategy': dict(self.stats.events_by_strategy)
                },
                'callbacks': {
                    'event_callbacks': sum(len(callbacks) for callbacks in self.callbacks.values()),
                    'global_callbacks': len(self.global_callbacks)
                }
            }
    
    def _handle_file_change(self, file_path: str, event_type: str, timestamp: float) -> ChangeEvent:
        """Handle file change event"""
        start_time = time.time()
        
        try:
            # Extract strategy info
            strategy_type = self._extract_strategy_type(file_path)
            config_name = Path(file_path).stem
            
            # Get file hashes
            old_hash = self.file_hashes.get(file_path)
            new_hash = self._get_file_hash(file_path) if Path(file_path).exists() else None
            
            # Create change event
            change_event = ChangeEvent(
                file_path=file_path,
                event_type=event_type,
                timestamp=datetime.fromtimestamp(timestamp),
                strategy_type=strategy_type,
                config_name=config_name,
                old_hash=old_hash,
                new_hash=new_hash
            )
            
            # Skip if file hasn't actually changed
            if event_type == 'modified' and old_hash == new_hash:
                change_event.success = True
                change_event.processing_time = time.time() - start_time
                return change_event
            
            # Process based on event type
            if event_type == 'deleted':
                self._handle_file_deletion(file_path, strategy_type, config_name)
            else:
                self._handle_file_update(file_path, strategy_type, config_name)
            
            # Update hash
            if new_hash:
                self.file_hashes[file_path] = new_hash
            elif file_path in self.file_hashes:
                del self.file_hashes[file_path]
            
            # Mark as successful
            change_event.success = True
            change_event.processing_time = time.time() - start_time
            
            logger.info(f"Successfully processed {event_type} event for {file_path} in {change_event.processing_time:.3f}s")
            
        except Exception as e:
            change_event.success = False
            change_event.error_message = str(e)
            change_event.processing_time = time.time() - start_time
            
            logger.error(f"Failed to process {event_type} event for {file_path}: {e}")
        
        # Update statistics
        self._update_statistics(change_event)
        
        # Add to history
        self._add_to_history(change_event)
        
        # Notify callbacks
        self._notify_callbacks(change_event)
        
        return change_event
    
    def _handle_file_update(self, file_path: str, strategy_type: str, config_name: str) -> None:
        """Handle file update (created or modified)"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Convert Excel to YAML if needed
        if self.auto_convert_excel and path.suffix.lower() in ['.xlsx', '.xls', '.xlsm']:
            yaml_data, metrics = self.converter.convert_single_file(file_path, strategy_type)
            
            if not metrics.success:
                raise ConfigurationError(f"Failed to convert Excel file: {metrics.error_message}")
            
            # Save YAML version
            yaml_path = path.with_suffix('.yml')
            self.converter.save_yaml(yaml_data, str(yaml_path))
            
            logger.debug(f"Converted {file_path} to YAML in {metrics.total_time:.3f}s")
        
        # Reload configuration
        try:
            config = self.config_manager.reload_configuration(strategy_type, config_name)
            
            if self.validate_on_reload:
                validation_result = self.config_manager.validate_configuration(config)
                if not validation_result['valid']:
                    logger.warning(f"Configuration validation failed for {file_path}: {validation_result['errors']}")
            
        except Exception as e:
            logger.error(f"Failed to reload configuration {strategy_type}/{config_name}: {e}")
            raise
    
    def _handle_file_deletion(self, file_path: str, strategy_type: str, config_name: str) -> None:
        """Handle file deletion"""
        try:
            # Remove from configuration manager
            self.config_manager.delete_configuration(strategy_type, config_name)
            
            # Remove YAML file if it exists
            yaml_path = Path(file_path).with_suffix('.yml')
            if yaml_path.exists():
                yaml_path.unlink()
                logger.debug(f"Removed YAML file: {yaml_path}")
            
        except Exception as e:
            logger.error(f"Failed to handle deletion of {file_path}: {e}")
            raise
    
    def _initialize_file_hashes(self) -> None:
        """Initialize file hashes for all watched files"""
        for watch_path in self.watch_paths:
            if watch_path.exists():
                for file_path in watch_path.rglob("*.xlsx"):
                    try:
                        self.file_hashes[str(file_path)] = self._get_file_hash(str(file_path))
                    except Exception as e:
                        logger.warning(f"Failed to initialize hash for {file_path}: {e}")
    
    def _get_file_hash(self, file_path: str) -> str:
        """Get hash of file for change detection"""
        try:
            stat = Path(file_path).stat()
            return hashlib.md5(f"{file_path}_{stat.st_mtime}_{stat.st_size}".encode()).hexdigest()
        except Exception:
            return ""
    
    def _extract_strategy_type(self, file_path: str) -> str:
        """Extract strategy type from file path"""
        path = Path(file_path)
        
        # Check if file is in a strategy directory
        for part in path.parts:
            if part in ['tbs', 'tv', 'orb', 'oi', 'ml', 'pos', 'mr']:
                return part
        
        # Check filename
        filename = path.stem.lower()
        for strategy_type in ['tbs', 'tv', 'orb', 'oi', 'ml', 'pos', 'mr']:
            if strategy_type in filename:
                return strategy_type
        
        return 'unknown'
    
    def _update_statistics(self, change_event: ChangeEvent) -> None:
        """Update system statistics"""
        with self.lock:
            self.stats.total_events += 1
            
            if change_event.success:
                self.stats.successful_reloads += 1
            else:
                self.stats.failed_reloads += 1
            
            # Update timing statistics
            processing_time = change_event.processing_time
            if processing_time > 0:
                # Update average
                total_successful = self.stats.successful_reloads
                if total_successful > 1:
                    self.stats.avg_processing_time = (
                        (self.stats.avg_processing_time * (total_successful - 1) + processing_time) / total_successful
                    )
                else:
                    self.stats.avg_processing_time = processing_time
                
                # Update min/max
                self.stats.max_processing_time = max(self.stats.max_processing_time, processing_time)
                self.stats.min_processing_time = min(self.stats.min_processing_time, processing_time)
            
            # Update event type statistics
            self.stats.events_by_type[change_event.event_type] = (
                self.stats.events_by_type.get(change_event.event_type, 0) + 1
            )
            
            # Update strategy statistics
            self.stats.events_by_strategy[change_event.strategy_type] = (
                self.stats.events_by_strategy.get(change_event.strategy_type, 0) + 1
            )
    
    def _add_to_history(self, change_event: ChangeEvent) -> None:
        """Add change event to history"""
        with self.lock:
            self.change_history.append(change_event)
            
            # Trim history if needed
            if len(self.change_history) > self.max_history_size:
                self.change_history = self.change_history[-self.max_history_size:]
    
    def _notify_callbacks(self, change_event: ChangeEvent) -> None:
        """Notify all relevant callbacks"""
        with self.lock:
            # Notify event-specific callbacks
            for callback_id, callback in self.callbacks.get(change_event.event_type, []):
                try:
                    callback(change_event)
                except Exception as e:
                    logger.error(f"Error in callback {callback_id}: {e}")
            
            # Notify 'all' event callbacks
            for callback_id, callback in self.callbacks.get('all', []):
                try:
                    callback(change_event)
                except Exception as e:
                    logger.error(f"Error in callback {callback_id}: {e}")
            
            # Notify global callbacks
            for callback in self.global_callbacks:
                try:
                    callback(change_event)
                except Exception as e:
                    logger.error(f"Error in global callback: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        self.start_watching()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_watching()

# Utility functions
def create_hot_reload_system(config_manager: ConfigurationManager, base_path: str) -> HotReloadSystem:
    """
    Create hot reload system with default configuration
    
    Args:
        config_manager: Configuration manager instance
        base_path: Base path for configuration files
        
    Returns:
        Configured hot reload system
    """
    watch_paths = [
        os.path.join(base_path, "data", "prod"),
        os.path.join(base_path, "data", "dev"),
    ]
    
    return HotReloadSystem(config_manager, watch_paths)

def setup_hot_reload_callbacks(hot_reload_system: HotReloadSystem) -> None:
    """
    Setup default callbacks for hot reload system
    
    Args:
        hot_reload_system: Hot reload system instance
    """
    def log_change_event(event: ChangeEvent):
        """Log change events"""
        if event.success:
            logger.info(f"Config reloaded: {event.strategy_type}/{event.config_name} ({event.event_type})")
        else:
            logger.error(f"Config reload failed: {event.strategy_type}/{event.config_name} - {event.error_message}")
    
    def performance_warning(event: ChangeEvent):
        """Warn about slow processing"""
        if event.processing_time > 0.05:  # 50ms threshold
            logger.warning(f"Slow config reload: {event.file_path} took {event.processing_time:.3f}s")
    
    hot_reload_system.add_global_callback(log_change_event)
    hot_reload_system.add_global_callback(performance_warning)