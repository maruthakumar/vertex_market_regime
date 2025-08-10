"""
Version Manager

Manages configuration versions with Git-like functionality including
commits, branches, tags, and rollbacks.
"""

import hashlib
import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import uuid
import shutil

from ..core.base_config import BaseConfiguration

logger = logging.getLogger(__name__)

@dataclass
class ConfigurationVersion:
    """Represents a version of a configuration"""
    version_id: str
    configuration_id: str
    strategy_type: str
    version_number: str
    file_hash: str
    content_hash: str
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]
    commit_message: str
    author: str
    created_at: datetime
    parent_version_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "configuration_id": self.configuration_id,
            "strategy_type": self.strategy_type,
            "version_number": self.version_number,
            "file_hash": self.file_hash,
            "content_hash": self.content_hash,
            "parameters": self.parameters,
            "metadata": self.metadata,
            "commit_message": self.commit_message,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "parent_version_id": self.parent_version_id,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConfigurationVersion':
        return cls(
            version_id=data["version_id"],
            configuration_id=data["configuration_id"],
            strategy_type=data["strategy_type"],
            version_number=data["version_number"],
            file_hash=data["file_hash"],
            content_hash=data["content_hash"],
            parameters=data["parameters"],
            metadata=data["metadata"],
            commit_message=data["commit_message"],
            author=data["author"],
            created_at=datetime.fromisoformat(data["created_at"]),
            parent_version_id=data.get("parent_version_id"),
            tags=data.get("tags", [])
        )

class VersionManager:
    """
    Git-like version control for configurations
    
    Provides versioning, branching, tagging, and rollback capabilities
    for configuration files with efficient delta storage.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """Initialize version manager"""
        self.storage_path = Path(storage_path or 
            "/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/configurations/version_control/storage")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.storage_path / "versions.db"
        self.objects_path = self.storage_path / "objects"
        self.objects_path.mkdir(exist_ok=True)
        
        self._lock = threading.Lock()
        self._init_database()
        
    def _init_database(self):
        """Initialize version control database"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Versions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS versions (
                    version_id TEXT PRIMARY KEY,
                    configuration_id TEXT NOT NULL,
                    strategy_type TEXT NOT NULL,
                    version_number TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    commit_message TEXT NOT NULL,
                    author TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    parent_version_id TEXT,
                    FOREIGN KEY (parent_version_id) REFERENCES versions (version_id)
                )
            ''')
            
            # Tags table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tags (
                    tag_id TEXT PRIMARY KEY,
                    version_id TEXT NOT NULL,
                    tag_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (version_id) REFERENCES versions (version_id),
                    UNIQUE(tag_name)
                )
            ''')
            
            # Configuration metadata table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS configurations (
                    configuration_id TEXT PRIMARY KEY,
                    strategy_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    latest_version_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (latest_version_id) REFERENCES versions (version_id)
                )
            ''')
            
            # Branches table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS branches (
                    branch_id TEXT PRIMARY KEY,
                    configuration_id TEXT NOT NULL,
                    branch_name TEXT NOT NULL,
                    head_version_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (configuration_id) REFERENCES configurations (configuration_id),
                    FOREIGN KEY (head_version_id) REFERENCES versions (version_id),
                    UNIQUE(configuration_id, branch_name)
                )
            ''')
            
            # Create indices
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_versions_config ON versions(configuration_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_versions_hash ON versions(content_hash)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tags_version ON tags(version_id)')
            
            conn.commit()
    
    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def calculate_content_hash(self, content: Dict[str, Any]) -> str:
        """Calculate hash of configuration content"""
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    def commit_configuration(self, 
                           config: BaseConfiguration,
                           file_path: str,
                           author: str,
                           commit_message: str,
                           parent_version_id: Optional[str] = None) -> ConfigurationVersion:
        """
        Commit a configuration version
        
        Args:
            config: Configuration object
            file_path: Path to original Excel file
            author: Author of the change
            commit_message: Commit message
            parent_version_id: Parent version ID for branching
            
        Returns:
            Created configuration version
        """
        with self._lock:
            # Calculate hashes
            file_hash = self.calculate_file_hash(file_path)
            parameters = config.to_dict()
            content_hash = self.calculate_content_hash(parameters)
            
            # Check if this exact version already exists
            existing_version = self._get_version_by_content_hash(content_hash)
            if existing_version:
                logger.info(f"Configuration with content hash {content_hash} already exists")
                return existing_version
            
            # Create version
            version_id = str(uuid.uuid4())
            configuration_id = f"{config.strategy_type}_{config.strategy_name}"
            
            # Generate version number
            version_number = self._generate_version_number(configuration_id, parent_version_id)
            
            # Store file in objects directory
            object_path = self.objects_path / f"{file_hash}.xlsx"
            if not object_path.exists():
                shutil.copy2(file_path, object_path)
            
            # Create version object
            version = ConfigurationVersion(
                version_id=version_id,
                configuration_id=configuration_id,
                strategy_type=config.strategy_type,
                version_number=version_number,
                file_hash=file_hash,
                content_hash=content_hash,
                parameters=parameters,
                metadata={
                    "original_filename": Path(file_path).name,
                    "file_size": Path(file_path).stat().st_size,
                    "strategy_name": config.strategy_name
                },
                commit_message=commit_message,
                author=author,
                created_at=datetime.now(),
                parent_version_id=parent_version_id
            )
            
            # Store in database
            self._store_version(version)
            
            # Update configuration metadata
            self._update_configuration_metadata(configuration_id, version, config)
            
            logger.info(f"Committed version {version_number} for {configuration_id}")
            return version
    
    def _generate_version_number(self, configuration_id: str, parent_version_id: Optional[str]) -> str:
        """Generate semantic version number"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            if parent_version_id:
                # Branching - get parent version
                cursor.execute('''
                    SELECT version_number FROM versions WHERE version_id = ?
                ''', (parent_version_id,))
                result = cursor.fetchone()
                if result:
                    parent_version = result[0]
                    # Create branch version
                    major, minor, patch = parent_version.split('.')
                    return f"{major}.{minor}.{int(patch) + 1}"
            
            # Get latest version for this configuration
            cursor.execute('''
                SELECT version_number FROM versions 
                WHERE configuration_id = ? 
                ORDER BY created_at DESC LIMIT 1
            ''', (configuration_id,))
            
            result = cursor.fetchone()
            if result:
                # Increment minor version
                major, minor, patch = result[0].split('.')
                return f"{major}.{int(minor) + 1}.0"
            else:
                # First version
                return "1.0.0"
    
    def _store_version(self, version: ConfigurationVersion):
        """Store version in database"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO versions (
                    version_id, configuration_id, strategy_type, version_number,
                    file_hash, content_hash, parameters_json, metadata_json,
                    commit_message, author, created_at, parent_version_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                version.version_id,
                version.configuration_id,
                version.strategy_type,
                version.version_number,
                version.file_hash,
                version.content_hash,
                json.dumps(version.parameters),
                json.dumps(version.metadata),
                version.commit_message,
                version.author,
                version.created_at.isoformat(),
                version.parent_version_id
            ))
            
            conn.commit()
    
    def _update_configuration_metadata(self, configuration_id: str, 
                                     version: ConfigurationVersion,
                                     config: BaseConfiguration):
        """Update configuration metadata"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Check if configuration exists
            cursor.execute('''
                SELECT configuration_id FROM configurations WHERE configuration_id = ?
            ''', (configuration_id,))
            
            if cursor.fetchone():
                # Update existing
                cursor.execute('''
                    UPDATE configurations 
                    SET latest_version_id = ?, updated_at = ?
                    WHERE configuration_id = ?
                ''', (version.version_id, datetime.now().isoformat(), configuration_id))
            else:
                # Create new
                cursor.execute('''
                    INSERT INTO configurations (
                        configuration_id, strategy_type, name, description,
                        latest_version_id, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    configuration_id,
                    config.strategy_type,
                    config.strategy_name,
                    f"{config.strategy_type} configuration for {config.strategy_name}",
                    version.version_id,
                    datetime.now().isoformat(),
                    datetime.now().isoformat()
                ))
            
            conn.commit()
    
    def _get_version_by_content_hash(self, content_hash: str) -> Optional[ConfigurationVersion]:
        """Get version by content hash"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM versions WHERE content_hash = ?
            ''', (content_hash,))
            
            result = cursor.fetchone()
            if result:
                return self._row_to_version(result)
        
        return None
    
    def get_version(self, version_id: str) -> Optional[ConfigurationVersion]:
        """Get version by ID"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM versions WHERE version_id = ?
            ''', (version_id,))
            
            result = cursor.fetchone()
            if result:
                version = self._row_to_version(result)
                
                # Load tags
                cursor.execute('''
                    SELECT tag_name FROM tags WHERE version_id = ?
                ''', (version_id,))
                version.tags = [row[0] for row in cursor.fetchall()]
                
                return version
        
        return None
    
    def get_configuration_history(self, configuration_id: str) -> List[ConfigurationVersion]:
        """Get version history for a configuration"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM versions 
                WHERE configuration_id = ?
                ORDER BY created_at DESC
            ''', (configuration_id,))
            
            versions = []
            for row in cursor.fetchall():
                version = self._row_to_version(row)
                
                # Load tags for each version
                cursor.execute('''
                    SELECT tag_name FROM tags WHERE version_id = ?
                ''', (version.version_id,))
                version.tags = [tag_row[0] for tag_row in cursor.fetchall()]
                
                versions.append(version)
            
            return versions
    
    def get_latest_version(self, configuration_id: str) -> Optional[ConfigurationVersion]:
        """Get latest version of a configuration"""
        history = self.get_configuration_history(configuration_id)
        return history[0] if history else None
    
    def tag_version(self, version_id: str, tag_name: str) -> bool:
        """Tag a version"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO tags (tag_id, version_id, tag_name, created_at)
                    VALUES (?, ?, ?, ?)
                ''', (str(uuid.uuid4()), version_id, tag_name, datetime.now().isoformat()))
                
                conn.commit()
            
            return True
        except sqlite3.IntegrityError:
            logger.error(f"Tag {tag_name} already exists")
            return False
    
    def get_version_by_tag(self, tag_name: str) -> Optional[ConfigurationVersion]:
        """Get version by tag name"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT v.* FROM versions v
                JOIN tags t ON v.version_id = t.version_id
                WHERE t.tag_name = ?
            ''', (tag_name,))
            
            result = cursor.fetchone()
            if result:
                return self._row_to_version(result)
        
        return None
    
    def rollback_to_version(self, configuration_id: str, target_version_id: str) -> bool:
        """Rollback configuration to a specific version"""
        try:
            target_version = self.get_version(target_version_id)
            if not target_version or target_version.configuration_id != configuration_id:
                return False
            
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                # Update latest version pointer
                cursor.execute('''
                    UPDATE configurations 
                    SET latest_version_id = ?, updated_at = ?
                    WHERE configuration_id = ?
                ''', (target_version_id, datetime.now().isoformat(), configuration_id))
                
                conn.commit()
            
            logger.info(f"Rolled back {configuration_id} to version {target_version.version_number}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rollback {configuration_id}: {e}")
            return False
    
    def get_file_path(self, version: ConfigurationVersion) -> Optional[str]:
        """Get file path for a version"""
        file_path = self.objects_path / f"{version.file_hash}.xlsx"
        return str(file_path) if file_path.exists() else None
    
    def _row_to_version(self, row: tuple) -> ConfigurationVersion:
        """Convert database row to ConfigurationVersion"""
        return ConfigurationVersion(
            version_id=row[0],
            configuration_id=row[1],
            strategy_type=row[2],
            version_number=row[3],
            file_hash=row[4],
            content_hash=row[5],
            parameters=json.loads(row[6]),
            metadata=json.loads(row[7]),
            commit_message=row[8],
            author=row[9],
            created_at=datetime.fromisoformat(row[10]),
            parent_version_id=row[11]
        )
    
    def list_configurations(self) -> List[Dict[str, Any]]:
        """List all configurations"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT c.*, v.version_number, v.created_at as latest_version_date
                FROM configurations c
                LEFT JOIN versions v ON c.latest_version_id = v.version_id
                ORDER BY c.updated_at DESC
            ''')
            
            configurations = []
            for row in cursor.fetchall():
                configurations.append({
                    "configuration_id": row[0],
                    "strategy_type": row[1],
                    "name": row[2],
                    "description": row[3],
                    "latest_version_id": row[4],
                    "created_at": row[5],
                    "updated_at": row[6],
                    "latest_version_number": row[7],
                    "latest_version_date": row[8]
                })
            
            return configurations
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get version control statistics"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Total versions
            cursor.execute('SELECT COUNT(*) FROM versions')
            total_versions = cursor.fetchone()[0]
            
            # Total configurations
            cursor.execute('SELECT COUNT(*) FROM configurations')
            total_configurations = cursor.fetchone()[0]
            
            # Total tags
            cursor.execute('SELECT COUNT(*) FROM tags')
            total_tags = cursor.fetchone()[0]
            
            # Storage used
            storage_size = sum(f.stat().st_size for f in self.objects_path.iterdir() if f.is_file())
            
            # Versions per strategy
            cursor.execute('''
                SELECT strategy_type, COUNT(*) 
                FROM versions 
                GROUP BY strategy_type
            ''')
            strategy_breakdown = dict(cursor.fetchall())
            
            return {
                "total_versions": total_versions,
                "total_configurations": total_configurations,
                "total_tags": total_tags,
                "storage_size_bytes": storage_size,
                "storage_size_mb": round(storage_size / (1024 * 1024), 2),
                "strategy_breakdown": strategy_breakdown
            }