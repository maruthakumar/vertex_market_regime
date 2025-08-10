"""
Base configuration abstract class
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
import json
import hashlib
from pathlib import Path
import logging

from .exceptions import ValidationError, ConfigurationError

logger = logging.getLogger(__name__)

class BaseConfiguration(ABC):
    """
    Abstract base class for all strategy configurations
    
    This class defines the common interface and shared functionality
    for all configuration types (TBS, TV, ORB, OI, ML, POS, Market Regime)
    """
    
    def __init__(self, strategy_type: str, strategy_name: str):
        """
        Initialize base configuration
        
        Args:
            strategy_type: Type of strategy (tbs, tv, orb, oi, ml, pos, market_regime)
            strategy_name: Unique name for this configuration instance
        """
        self.strategy_type = strategy_type.lower()
        self.strategy_name = strategy_name
        self.version = "1.0.0"
        self.created_at = datetime.now()
        self.modified_at = datetime.now()
        self.metadata = {}
        self._data = {}
        self._is_validated = False
        self._checksum = None
        
    @abstractmethod
    def validate(self) -> bool:
        """
        Validate the configuration
        
        Returns:
            bool: True if valid, raises ValidationError if invalid
        """
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema for this configuration type
        
        Returns:
            Dict containing the JSON schema
        """
        pass
    
    @abstractmethod
    def get_default_values(self) -> Dict[str, Any]:
        """
        Get default values for all parameters
        
        Returns:
            Dict containing default values
        """
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary
        
        Returns:
            Dict representation of the configuration
        """
        return {
            "strategy_type": self.strategy_type,
            "strategy_name": self.strategy_name,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "metadata": self.metadata,
            "data": self._data,
            "checksum": self.calculate_checksum()
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        Load configuration from dictionary
        
        Args:
            data: Dictionary containing configuration data
        """
        self.strategy_type = data.get("strategy_type", self.strategy_type)
        self.strategy_name = data.get("strategy_name", self.strategy_name)
        self.version = data.get("version", self.version)
        
        if "created_at" in data:
            self.created_at = datetime.fromisoformat(data["created_at"])
        if "modified_at" in data:
            self.modified_at = datetime.fromisoformat(data["modified_at"])
            
        self.metadata = data.get("metadata", {})
        self._data = data.get("data", {})
        self._checksum = data.get("checksum")
        self._is_validated = False
    
    def to_json(self, indent: int = 2) -> str:
        """
        Convert configuration to JSON string
        
        Args:
            indent: Number of spaces for indentation
            
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def from_json(self, json_str: str) -> None:
        """
        Load configuration from JSON string
        
        Args:
            json_str: JSON string containing configuration
        """
        data = json.loads(json_str)
        self.from_dict(data)
    
    def save_to_file(self, file_path: Path) -> None:
        """
        Save configuration to file
        
        Args:
            file_path: Path to save the configuration
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            f.write(self.to_json())
        
        logger.info(f"Configuration saved to {file_path}")
    
    def load_from_file(self, file_path: Path) -> None:
        """
        Load configuration from file
        
        Args:
            file_path: Path to load the configuration from
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ConfigurationError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        self.from_dict(data)
        logger.info(f"Configuration loaded from {file_path}")
    
    def calculate_checksum(self) -> str:
        """
        Calculate checksum for the configuration data
        
        Returns:
            SHA256 hash of the configuration data
        """
        data_str = json.dumps(self._data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def verify_checksum(self) -> bool:
        """
        Verify the configuration checksum
        
        Returns:
            bool: True if checksum matches, False otherwise
        """
        if not self._checksum:
            return True  # No checksum to verify
        
        current_checksum = self.calculate_checksum()
        return current_checksum == self._checksum
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        data = self._data
        
        for k in keys[:-1]:
            if k not in data:
                data[k] = {}
            data = data[k]
        
        data[keys[-1]] = value
        self.modified_at = datetime.now()
        self._is_validated = False
        self._checksum = None
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update multiple configuration values
        
        Args:
            updates: Dictionary of updates to apply
        """
        for key, value in updates.items():
            self.set(key, value)
    
    def get_all_keys(self) -> List[str]:
        """
        Get all configuration keys
        
        Returns:
            List of all keys in dot notation
        """
        def extract_keys(data: dict, prefix: str = "") -> List[str]:
            keys = []
            for k, v in data.items():
                full_key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    keys.extend(extract_keys(v, full_key))
                else:
                    keys.append(full_key)
            return keys
        
        return extract_keys(self._data)
    
    def get_required_keys(self) -> Set[str]:
        """
        Get required configuration keys from schema
        
        Returns:
            Set of required keys
        """
        schema = self.get_schema()
        required = set()
        
        def extract_required(schema_part: dict, prefix: str = "") -> None:
            if "required" in schema_part:
                for key in schema_part["required"]:
                    full_key = f"{prefix}.{key}" if prefix else key
                    required.add(full_key)
            
            if "properties" in schema_part:
                for key, prop in schema_part["properties"].items():
                    if isinstance(prop, dict) and "properties" in prop:
                        extract_required(prop, f"{prefix}.{key}" if prefix else key)
        
        extract_required(schema)
        return required
    
    def is_valid(self) -> bool:
        """
        Check if configuration is valid without raising exception
        
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            self.validate()
            return True
        except ValidationError:
            return False
    
    def get_validation_errors(self) -> Dict[str, List[str]]:
        """
        Get validation errors without raising exception
        
        Returns:
            Dict mapping field names to error messages
        """
        try:
            self.validate()
            return {}
        except ValidationError as e:
            return e.errors
    
    def clone(self) -> 'BaseConfiguration':
        """
        Create a deep copy of the configuration
        
        Returns:
            New configuration instance with copied data
        """
        new_config = self.__class__(self.strategy_type, f"{self.strategy_name}_clone")
        new_config.from_dict(self.to_dict())
        new_config.created_at = datetime.now()
        new_config.modified_at = datetime.now()
        return new_config
    
    def diff(self, other: 'BaseConfiguration') -> Dict[str, Any]:
        """
        Compare with another configuration
        
        Args:
            other: Configuration to compare with
            
        Returns:
            Dict containing differences
        """
        if not isinstance(other, self.__class__):
            raise ConfigurationError("Cannot compare configurations of different types")
        
        differences = {
            "added": {},
            "removed": {},
            "modified": {}
        }
        
        self_keys = set(self.get_all_keys())
        other_keys = set(other.get_all_keys())
        
        # Find added keys
        for key in other_keys - self_keys:
            differences["added"][key] = other.get(key)
        
        # Find removed keys
        for key in self_keys - other_keys:
            differences["removed"][key] = self.get(key)
        
        # Find modified values
        for key in self_keys & other_keys:
            self_value = self.get(key)
            other_value = other.get(key)
            if self_value != other_value:
                differences["modified"][key] = {
                    "old": self_value,
                    "new": other_value
                }
        
        return differences
    
    def merge(self, other: 'BaseConfiguration', strategy: str = 'override') -> None:
        """
        Merge another configuration into this one
        
        Args:
            other: Configuration to merge
            strategy: Merge strategy ('override', 'keep', 'error')
        """
        if not isinstance(other, self.__class__):
            raise ConfigurationError("Cannot merge configurations of different types")
        
        for key in other.get_all_keys():
            other_value = other.get(key)
            
            if strategy == 'override':
                self.set(key, other_value)
            elif strategy == 'keep':
                if self.get(key) is None:
                    self.set(key, other_value)
            elif strategy == 'error':
                if self.get(key) is not None and self.get(key) != other_value:
                    raise ConfigurationError(f"Merge conflict at key: {key}")
                self.set(key, other_value)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.strategy_type}, name={self.strategy_name}, version={self.version})"
    
    def __str__(self) -> str:
        return f"{self.strategy_type.upper()} Configuration: {self.strategy_name} v{self.version}"