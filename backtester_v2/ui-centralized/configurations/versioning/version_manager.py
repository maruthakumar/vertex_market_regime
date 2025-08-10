"""
Configuration Version Management System
Track changes and provide rollback capabilities for Excel configurations
"""

import os
import shutil
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
import logging
from contextlib import contextmanager
import threading

logger = logging.getLogger(__name__)

@dataclass
class ConfigVersion:
    """Represents a configuration version"""
    version_id: str
    strategy_type: str
    config_name: str
    file_path: str
    file_hash: str
    file_size: int
    timestamp: datetime
    user: str
    description: str
    tags: List[str]
    parent_version: Optional[str] = None
    is_backup: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConfigVersion':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

@dataclass
class VersionDiff:
    """Represents differences between two versions"""
    old_version: ConfigVersion
    new_version: ConfigVersion
    changes: Dict[str, Any]
    added_sheets: List[str]
    removed_sheets: List[str]
    modified_sheets: List[str]
    parameter_changes: Dict[str, Dict[str, Any]]

class ConfigurationVersionManager:
    """
    Configuration version management system
    
    Features:
    - Automatic versioning on configuration changes
    - Rollback capabilities
    - Version comparison and diff generation
    - Branching and tagging
    - Backup management
    - Performance optimization for large configurations
    """
    
    def __init__(self, base_path: str, max_versions: int = 100):
        """
        Initialize version manager
        
        Args:
            base_path: Base path for storing versions
            max_versions: Maximum versions to keep per configuration
        """
        self.base_path = Path(base_path)
        self.versions_dir = self.base_path / "versions"
        self.metadata_dir = self.base_path / "metadata"
        self.backups_dir = self.base_path / "backups"
        
        # Create directories
        for dir_path in [self.versions_dir, self.metadata_dir, self.backups_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.max_versions = max_versions
        self.lock = threading.RLock()
        
        # In-memory caches
        self.version_cache = {}
        self.metadata_cache = {}
        
        logger.info(f"ConfigurationVersionManager initialized at {base_path}")
    
    def create_version(self, file_path: str, strategy_type: str, config_name: str, 
                      user: str = "system", description: str = "", tags: List[str] = None) -> ConfigVersion:
        """
        Create a new version of a configuration
        
        Args:
            file_path: Path to configuration file
            strategy_type: Strategy type
            config_name: Configuration name
            user: User creating the version
            description: Version description
            tags: Version tags
            
        Returns:
            Created version
        """
        with self.lock:
            if tags is None:
                tags = []
            
            # Validate file exists
            source_path = Path(file_path)
            if not source_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {file_path}")
            
            # Generate version ID
            timestamp = datetime.now(timezone.utc)
            file_hash = self._get_file_hash(file_path)
            version_id = f"{strategy_type}_{config_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{file_hash[:8]}"
            
            # Get file metadata
            stat = source_path.stat()
            
            # Find parent version
            parent_version = self._get_latest_version_id(strategy_type, config_name)
            
            # Create version object
            version = ConfigVersion(
                version_id=version_id,
                strategy_type=strategy_type,
                config_name=config_name,
                file_path=file_path,
                file_hash=file_hash,
                file_size=stat.st_size,
                timestamp=timestamp,
                user=user,
                description=description,
                tags=tags,
                parent_version=parent_version,
                metadata=self._extract_file_metadata(file_path)
            )
            
            # Store version file
            version_file_path = self._get_version_file_path(version_id)
            shutil.copy2(file_path, version_file_path)
            
            # Store metadata
            self._save_version_metadata(version)
            
            # Update cache
            self._update_cache(version)
            
            # Cleanup old versions
            self._cleanup_old_versions(strategy_type, config_name)
            
            logger.info(f"Created version {version_id} for {strategy_type}/{config_name}")
            return version
    
    def get_version(self, version_id: str) -> Optional[ConfigVersion]:
        """
        Get version by ID
        
        Args:
            version_id: Version ID
            
        Returns:
            Version object or None if not found
        """
        with self.lock:
            # Check cache first
            if version_id in self.version_cache:
                return self.version_cache[version_id]
            
            # Load from metadata
            metadata_path = self.metadata_dir / f"{version_id}.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        data = json.load(f)
                    
                    version = ConfigVersion.from_dict(data)
                    self.version_cache[version_id] = version
                    return version
                    
                except Exception as e:
                    logger.error(f"Failed to load version {version_id}: {e}")
            
            return None
    
    def list_versions(self, strategy_type: str, config_name: str, 
                     limit: int = 50, offset: int = 0) -> List[ConfigVersion]:
        """
        List versions for a configuration
        
        Args:
            strategy_type: Strategy type
            config_name: Configuration name
            limit: Maximum number of versions to return
            offset: Number of versions to skip
            
        Returns:
            List of versions, sorted by timestamp (newest first)
        """
        with self.lock:
            versions = []
            
            # Find all version metadata files
            pattern = f"{strategy_type}_{config_name}_*"
            
            for metadata_file in self.metadata_dir.glob(f"{pattern}.json"):
                version_id = metadata_file.stem
                version = self.get_version(version_id)
                
                if version:
                    versions.append(version)
            
            # Sort by timestamp (newest first)
            versions.sort(key=lambda v: v.timestamp, reverse=True)
            
            # Apply pagination
            return versions[offset:offset + limit]
    
    def get_version_file(self, version_id: str) -> Optional[Path]:
        """
        Get path to version file
        
        Args:
            version_id: Version ID
            
        Returns:
            Path to version file or None if not found
        """
        version_file_path = self._get_version_file_path(version_id)
        return version_file_path if version_file_path.exists() else None
    
    def restore_version(self, version_id: str, target_path: str) -> bool:
        """
        Restore a version to a target path
        
        Args:
            version_id: Version ID to restore
            target_path: Target path for restoration
            
        Returns:
            True if successful
        """
        with self.lock:
            version = self.get_version(version_id)
            if not version:
                logger.error(f"Version not found: {version_id}")
                return False
            
            version_file_path = self._get_version_file_path(version_id)
            if not version_file_path.exists():
                logger.error(f"Version file not found: {version_file_path}")
                return False
            
            try:
                # Create backup of current file if it exists
                target_path = Path(target_path)
                if target_path.exists():
                    backup_path = self.backups_dir / f"{target_path.name}.backup.{int(time.time())}"
                    shutil.copy2(target_path, backup_path)
                    logger.info(f"Created backup at {backup_path}")
                
                # Restore version
                shutil.copy2(version_file_path, target_path)
                
                logger.info(f"Restored version {version_id} to {target_path}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to restore version {version_id}: {e}")
                return False
    
    def compare_versions(self, version_id1: str, version_id2: str) -> Optional[VersionDiff]:
        """
        Compare two versions
        
        Args:
            version_id1: First version ID
            version_id2: Second version ID
            
        Returns:
            Version diff object or None if comparison failed
        """
        version1 = self.get_version(version_id1)
        version2 = self.get_version(version_id2)
        
        if not version1 or not version2:
            return None
        
        # Load configuration data for comparison
        try:
            from ..converters.excel_to_yaml import ExcelToYAMLConverter
            
            converter = ExcelToYAMLConverter()
            
            # Get file paths
            file1_path = self._get_version_file_path(version_id1)
            file2_path = self._get_version_file_path(version_id2)
            
            # Convert to YAML for comparison
            data1, _ = converter.convert_single_file(str(file1_path))
            data2, _ = converter.convert_single_file(str(file2_path))
            
            # Compare data
            changes = self._compare_data(data1, data2)
            
            # Find sheet differences
            sheets1 = set(data1.keys())
            sheets2 = set(data2.keys())
            
            added_sheets = list(sheets2 - sheets1)
            removed_sheets = list(sheets1 - sheets2)
            modified_sheets = []
            
            for sheet in sheets1 & sheets2:
                if data1[sheet] != data2[sheet]:
                    modified_sheets.append(sheet)
            
            # Find parameter changes
            parameter_changes = {}
            for sheet in modified_sheets:
                parameter_changes[sheet] = self._compare_sheet_data(data1[sheet], data2[sheet])
            
            return VersionDiff(
                old_version=version1,
                new_version=version2,
                changes=changes,
                added_sheets=added_sheets,
                removed_sheets=removed_sheets,
                modified_sheets=modified_sheets,
                parameter_changes=parameter_changes
            )
            
        except Exception as e:
            logger.error(f"Failed to compare versions {version_id1} and {version_id2}: {e}")
            return None
    
    def tag_version(self, version_id: str, tags: List[str]) -> bool:
        """
        Add tags to a version
        
        Args:
            version_id: Version ID
            tags: Tags to add
            
        Returns:
            True if successful
        """
        with self.lock:
            version = self.get_version(version_id)
            if not version:
                return False
            
            # Update tags
            version.tags.extend(tags)
            version.tags = list(set(version.tags))  # Remove duplicates
            
            # Save metadata
            self._save_version_metadata(version)
            
            # Update cache
            self.version_cache[version_id] = version
            
            logger.info(f"Tagged version {version_id} with {tags}")
            return True
    
    def delete_version(self, version_id: str) -> bool:
        """
        Delete a version
        
        Args:
            version_id: Version ID to delete
            
        Returns:
            True if successful
        """
        with self.lock:
            version = self.get_version(version_id)
            if not version:
                return False
            
            try:
                # Delete version file
                version_file_path = self._get_version_file_path(version_id)
                if version_file_path.exists():
                    version_file_path.unlink()
                
                # Delete metadata
                metadata_path = self.metadata_dir / f"{version_id}.json"
                if metadata_path.exists():
                    metadata_path.unlink()
                
                # Remove from cache
                if version_id in self.version_cache:
                    del self.version_cache[version_id]
                
                logger.info(f"Deleted version {version_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to delete version {version_id}: {e}")
                return False
    
    def create_backup(self, strategy_type: str, config_name: str, 
                     description: str = "Manual backup") -> ConfigVersion:
        """
        Create a backup version
        
        Args:
            strategy_type: Strategy type
            config_name: Configuration name
            description: Backup description
            
        Returns:
            Created backup version
        """
        # Find current configuration file
        current_file = self._find_current_file(strategy_type, config_name)
        if not current_file:
            raise FileNotFoundError(f"Current configuration file not found for {strategy_type}/{config_name}")
        
        # Create version marked as backup
        version = self.create_version(
            file_path=current_file,
            strategy_type=strategy_type,
            config_name=config_name,
            user="system",
            description=description,
            tags=["backup"]
        )
        
        version.is_backup = True
        self._save_version_metadata(version)
        
        return version
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get version manager statistics"""
        with self.lock:
            total_versions = len(list(self.metadata_dir.glob("*.json")))
            
            # Count by strategy type
            strategy_counts = {}
            for metadata_file in self.metadata_dir.glob("*.json"):
                version_id = metadata_file.stem
                parts = version_id.split('_')
                if len(parts) >= 2:
                    strategy_type = parts[0]
                    strategy_counts[strategy_type] = strategy_counts.get(strategy_type, 0) + 1
            
            # Calculate storage usage
            storage_usage = 0
            for version_file in self.versions_dir.glob("*"):
                if version_file.is_file():
                    storage_usage += version_file.stat().st_size
            
            return {
                'total_versions': total_versions,
                'versions_by_strategy': strategy_counts,
                'storage_usage_bytes': storage_usage,
                'storage_usage_mb': storage_usage / (1024 * 1024),
                'cache_size': len(self.version_cache),
                'directories': {
                    'versions': str(self.versions_dir),
                    'metadata': str(self.metadata_dir),
                    'backups': str(self.backups_dir)
                }
            }
    
    def cleanup_old_versions(self, strategy_type: str, config_name: str, keep_count: int = None) -> int:
        """
        Clean up old versions
        
        Args:
            strategy_type: Strategy type
            config_name: Configuration name
            keep_count: Number of versions to keep (uses max_versions if None)
            
        Returns:
            Number of versions deleted
        """
        if keep_count is None:
            keep_count = self.max_versions
        
        return self._cleanup_old_versions(strategy_type, config_name, keep_count)
    
    def _get_version_file_path(self, version_id: str) -> Path:
        """Get path for version file"""
        return self.versions_dir / version_id
    
    def _save_version_metadata(self, version: ConfigVersion) -> None:
        """Save version metadata"""
        metadata_path = self.metadata_dir / f"{version.version_id}.json"
        
        with open(metadata_path, 'w') as f:
            json.dump(version.to_dict(), f, indent=2)
    
    def _update_cache(self, version: ConfigVersion) -> None:
        """Update version cache"""
        self.version_cache[version.version_id] = version
    
    def _get_latest_version_id(self, strategy_type: str, config_name: str) -> Optional[str]:
        """Get latest version ID for a configuration"""
        versions = self.list_versions(strategy_type, config_name, limit=1)
        return versions[0].version_id if versions else None
    
    def _get_file_hash(self, file_path: str) -> str:
        """Calculate file hash"""
        import hashlib
        
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _extract_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from file"""
        try:
            from ..converters.excel_to_yaml import ExcelToYAMLConverter
            
            converter = ExcelToYAMLConverter()
            path = Path(file_path)
            
            if path.suffix.lower() in ['.xlsx', '.xls', '.xlsm']:
                import pandas as pd
                excel_file = pd.ExcelFile(file_path)
                
                return {
                    'file_type': 'excel',
                    'file_size': path.stat().st_size,
                    'sheet_count': len(excel_file.sheet_names),
                    'sheet_names': excel_file.sheet_names,
                    'modification_time': path.stat().st_mtime
                }
            
            return {
                'file_type': 'other',
                'file_size': path.stat().st_size,
                'modification_time': path.stat().st_mtime
            }
            
        except Exception as e:
            logger.warning(f"Failed to extract metadata from {file_path}: {e}")
            return {}
    
    def _cleanup_old_versions(self, strategy_type: str, config_name: str, keep_count: int = None) -> int:
        """Clean up old versions"""
        if keep_count is None:
            keep_count = self.max_versions
        
        versions = self.list_versions(strategy_type, config_name, limit=1000)
        
        if len(versions) <= keep_count:
            return 0
        
        # Keep the most recent versions
        versions_to_delete = versions[keep_count:]
        deleted_count = 0
        
        for version in versions_to_delete:
            # Don't delete tagged versions
            if version.tags and any(tag in ['important', 'release', 'milestone'] for tag in version.tags):
                continue
            
            if self.delete_version(version.version_id):
                deleted_count += 1
        
        return deleted_count
    
    def _find_current_file(self, strategy_type: str, config_name: str) -> Optional[str]:
        """Find current configuration file"""
        # This would need to be implemented based on your file organization
        # For now, assume it's in the prod directory
        base_path = Path(self.base_path).parent / "data" / "prod" / strategy_type
        
        for pattern in [f"{config_name}.xlsx", f"{config_name}.xls", f"{config_name}.json"]:
            file_path = base_path / pattern
            if file_path.exists():
                return str(file_path)
        
        return None
    
    def _compare_data(self, data1: Dict[str, Any], data2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two data dictionaries"""
        changes = {
            'added': {},
            'removed': {},
            'modified': {}
        }
        
        # Find added and modified keys
        for key, value in data2.items():
            if key not in data1:
                changes['added'][key] = value
            elif data1[key] != value:
                changes['modified'][key] = {
                    'old': data1[key],
                    'new': value
                }
        
        # Find removed keys
        for key in data1:
            if key not in data2:
                changes['removed'][key] = data1[key]
        
        return changes
    
    def _compare_sheet_data(self, sheet1: Any, sheet2: Any) -> Dict[str, Any]:
        """Compare two sheet data structures"""
        if isinstance(sheet1, dict) and isinstance(sheet2, dict):
            return self._compare_data(sheet1, sheet2)
        elif isinstance(sheet1, list) and isinstance(sheet2, list):
            return {
                'length_change': len(sheet2) - len(sheet1),
                'old_length': len(sheet1),
                'new_length': len(sheet2)
            }
        else:
            return {
                'type_change': {
                    'old_type': type(sheet1).__name__,
                    'new_type': type(sheet2).__name__
                }
            }
    
    @contextmanager
    def version_transaction(self, strategy_type: str, config_name: str):
        """Context manager for version transactions"""
        # Create backup before changes
        backup_version = None
        try:
            backup_version = self.create_backup(strategy_type, config_name, "Transaction backup")
            yield backup_version
        except Exception as e:
            # Restore backup if something went wrong
            if backup_version:
                current_file = self._find_current_file(strategy_type, config_name)
                if current_file:
                    self.restore_version(backup_version.version_id, current_file)
            raise e

# Utility functions
def create_version_manager(base_path: str, max_versions: int = 100) -> ConfigurationVersionManager:
    """
    Create version manager with default settings
    
    Args:
        base_path: Base path for version storage
        max_versions: Maximum versions to keep
        
    Returns:
        Configured version manager
    """
    return ConfigurationVersionManager(base_path, max_versions)

def auto_version_on_change(version_manager: ConfigurationVersionManager, 
                          hot_reload_system) -> None:
    """
    Setup automatic versioning on configuration changes
    
    Args:
        version_manager: Version manager instance
        hot_reload_system: Hot reload system instance
    """
    def create_version_on_change(change_event):
        """Create version when configuration changes"""
        if change_event.success and change_event.event_type in ['modified', 'created']:
            try:
                version_manager.create_version(
                    file_path=change_event.file_path,
                    strategy_type=change_event.strategy_type,
                    config_name=change_event.config_name,
                    user="auto",
                    description=f"Auto-version on {change_event.event_type}",
                    tags=["auto-generated"]
                )
            except Exception as e:
                logger.error(f"Failed to create auto-version: {e}")
    
    hot_reload_system.add_global_callback(create_version_on_change)