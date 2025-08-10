"""
Configuration-specific exceptions
"""

class ConfigurationError(Exception):
    """Base exception for all configuration-related errors"""
    pass

class ValidationError(ConfigurationError):
    """Raised when configuration validation fails"""
    def __init__(self, message: str, errors: dict = None):
        super().__init__(message)
        self.errors = errors or {}

class ParsingError(ConfigurationError):
    """Raised when configuration parsing fails"""
    def __init__(self, message: str, file_path: str = None, line_number: int = None):
        super().__init__(message)
        self.file_path = file_path
        self.line_number = line_number

class StorageError(ConfigurationError):
    """Raised when configuration storage operations fail"""
    pass

class VersionError(ConfigurationError):
    """Raised when version control operations fail"""
    pass

class SchemaError(ConfigurationError):
    """Raised when schema validation fails"""
    def __init__(self, message: str, schema_errors: list = None):
        super().__init__(message)
        self.schema_errors = schema_errors or []

class DependencyError(ConfigurationError):
    """Raised when configuration dependencies are not met"""
    def __init__(self, message: str, missing_dependencies: list = None):
        super().__init__(message)
        self.missing_dependencies = missing_dependencies or []

class LockError(ConfigurationError):
    """Raised when configuration is locked and cannot be modified"""
    pass