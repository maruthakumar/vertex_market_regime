"""
Configuration Versioning Package
Version management and rollback capabilities
"""

from .version_manager import (
    ConfigurationVersionManager,
    ConfigVersion,
    VersionDiff,
    create_version_manager,
    auto_version_on_change
)

__all__ = [
    'ConfigurationVersionManager',
    'ConfigVersion',
    'VersionDiff',
    'create_version_manager',
    'auto_version_on_change'
]