"""
Version Control System

Git-like version control for configuration files with change tracking,
rollback capabilities, and configuration comparison.
"""

from .version_manager import VersionManager, ConfigurationVersion
from .diff_engine import DiffEngine, ConfigurationDiff
from .history_tracker import HistoryTracker

__all__ = [
    'VersionManager',
    'ConfigurationVersion', 
    'DiffEngine',
    'ConfigurationDiff',
    'HistoryTracker'
]