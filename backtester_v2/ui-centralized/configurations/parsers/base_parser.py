"""
Base parser abstract class
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

from ..core.exceptions import ParsingError

logger = logging.getLogger(__name__)

class BaseParser(ABC):
    """
    Abstract base class for configuration parsers
    
    This class defines the interface for parsing configuration files
    of different formats (Excel, JSON, YAML, etc.)
    """
    
    def __init__(self):
        """Initialize base parser"""
        self.supported_extensions = []
        self.validation_errors = []
        self.warnings = []
        
    @abstractmethod
    def parse(self, file_path: str) -> Dict[str, Any]:
        """
        Parse configuration file
        
        Args:
            file_path: Path to the configuration file
            
        Returns:
            Dictionary containing parsed configuration data
            
        Raises:
            ParsingError: If parsing fails
        """
        pass
    
    @abstractmethod
    def validate_structure(self, data: Dict[str, Any]) -> bool:
        """
        Validate the structure of parsed data
        
        Args:
            data: Parsed configuration data
            
        Returns:
            True if structure is valid, False otherwise
        """
        pass
    
    def can_parse(self, file_path: str) -> bool:
        """
        Check if this parser can handle the given file
        
        Args:
            file_path: Path to the configuration file
            
        Returns:
            True if parser can handle the file, False otherwise
        """
        path = Path(file_path)
        return path.suffix.lower() in self.supported_extensions
    
    def preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess parsed data
        
        This method can be overridden to perform common preprocessing steps
        like data type conversion, normalization, etc.
        
        Args:
            data: Raw parsed data
            
        Returns:
            Preprocessed data
        """
        # Remove empty values
        cleaned_data = self._remove_empty_values(data)
        
        # Convert string booleans
        cleaned_data = self._convert_string_booleans(cleaned_data)
        
        # Normalize keys
        cleaned_data = self._normalize_keys(cleaned_data)
        
        return cleaned_data
    
    def postprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Postprocess parsed data
        
        This method can be overridden to perform final processing steps
        like validation, enrichment, etc.
        
        Args:
            data: Preprocessed data
            
        Returns:
            Final processed data
        """
        # Add metadata
        data = self._add_metadata(data)
        
        # Validate required fields
        self._validate_required_fields(data)
        
        return data
    
    def parse_with_validation(self, file_path: str) -> Dict[str, Any]:
        """
        Parse file with full validation
        
        Args:
            file_path: Path to the configuration file
            
        Returns:
            Parsed and validated data
            
        Raises:
            ParsingError: If parsing or validation fails
        """
        # Check if file exists
        path = Path(file_path)
        if not path.exists():
            raise ParsingError(f"File not found: {file_path}", file_path=file_path)
        
        # Check if parser can handle the file
        if not self.can_parse(file_path):
            raise ParsingError(
                f"Unsupported file type: {path.suffix}",
                file_path=file_path
            )
        
        try:
            # Parse the file
            raw_data = self.parse(file_path)
            
            # Preprocess
            data = self.preprocess(raw_data)
            
            # Validate structure
            if not self.validate_structure(data):
                raise ParsingError(
                    "Invalid configuration structure",
                    file_path=file_path
                )
            
            # Postprocess
            data = self.postprocess(data)
            
            logger.info(f"Successfully parsed {file_path}")
            return data
            
        except ParsingError:
            raise
        except Exception as e:
            raise ParsingError(
                f"Failed to parse configuration: {str(e)}",
                file_path=file_path
            )
    
    def get_errors(self) -> List[str]:
        """
        Get parsing errors
        
        Returns:
            List of error messages
        """
        return self.validation_errors.copy()
    
    def get_warnings(self) -> List[str]:
        """
        Get parsing warnings
        
        Returns:
            List of warning messages
        """
        return self.warnings.copy()
    
    def clear_errors(self) -> None:
        """Clear all errors and warnings"""
        self.validation_errors.clear()
        self.warnings.clear()
    
    def _remove_empty_values(self, data: Any) -> Any:
        """Remove empty values from data recursively"""
        if isinstance(data, dict):
            return {
                k: self._remove_empty_values(v)
                for k, v in data.items()
                if v is not None and v != ""
            }
        elif isinstance(data, list):
            return [
                self._remove_empty_values(item)
                for item in data
                if item is not None and item != ""
            ]
        else:
            return data
    
    def _convert_string_booleans(self, data: Any) -> Any:
        """Convert string boolean values to actual booleans"""
        if isinstance(data, dict):
            return {
                k: self._convert_string_booleans(v)
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [self._convert_string_booleans(item) for item in data]
        elif isinstance(data, str):
            if data.lower() in ['true', 'yes', 'on', '1']:
                return True
            elif data.lower() in ['false', 'no', 'off', '0']:
                return False
        
        return data
    
    def _normalize_keys(self, data: Any) -> Any:
        """Normalize dictionary keys (e.g., convert to lowercase)"""
        if isinstance(data, dict):
            return {
                self._normalize_key(k): self._normalize_keys(v)
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [self._normalize_keys(item) for item in data]
        else:
            return data
    
    def _normalize_key(self, key: str) -> str:
        """Normalize a single key"""
        # Convert to lowercase and replace spaces with underscores
        return key.lower().replace(' ', '_').replace('-', '_')
    
    def _add_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Add parsing metadata"""
        if '_metadata' not in data:
            data['_metadata'] = {}
        
        data['_metadata']['parser'] = self.__class__.__name__
        data['_metadata']['parsed_at'] = str(Path.cwd())
        
        return data
    
    def _validate_required_fields(self, data: Dict[str, Any]) -> None:
        """Validate that required fields are present"""
        # This should be overridden in subclasses
        pass
    
    def add_error(self, error: str) -> None:
        """Add a validation error"""
        self.validation_errors.append(error)
        logger.error(f"Validation error: {error}")
    
    def add_warning(self, warning: str) -> None:
        """Add a validation warning"""
        self.warnings.append(warning)
        logger.warning(f"Validation warning: {warning}")
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(extensions={self.supported_extensions})"