"""
Configuration registry for strategy types
"""

from typing import Dict, Type, Optional
import logging

from .base_config import BaseConfiguration
from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)

class ConfigurationRegistry:
    """
    Registry for configuration classes
    
    This class maintains a mapping between strategy types and their
    corresponding configuration classes.
    """
    
    def __init__(self):
        """Initialize the registry"""
        self._registry: Dict[str, Type[BaseConfiguration]] = {}
        self._parsers = {}
        self._validators = {}
        self._schemas = {}
        
    def register(self, strategy_type: str, config_class: Type[BaseConfiguration]) -> None:
        """
        Register a configuration class for a strategy type
        
        Args:
            strategy_type: Type of strategy (tbs, tv, orb, etc.)
            config_class: Configuration class for the strategy
        """
        strategy_type = strategy_type.lower()
        
        if strategy_type in self._registry:
            logger.warning(f"Overriding existing registration for {strategy_type}")
        
        self._registry[strategy_type] = config_class
        logger.info(f"Registered {config_class.__name__} for {strategy_type}")
    
    def unregister(self, strategy_type: str) -> bool:
        """
        Unregister a configuration class
        
        Args:
            strategy_type: Type of strategy to unregister
            
        Returns:
            True if unregistered, False if not found
        """
        strategy_type = strategy_type.lower()
        
        if strategy_type in self._registry:
            del self._registry[strategy_type]
            logger.info(f"Unregistered configuration for {strategy_type}")
            return True
        
        return False
    
    def get_class(self, strategy_type: str) -> Optional[Type[BaseConfiguration]]:
        """
        Get configuration class for a strategy type
        
        Args:
            strategy_type: Type of strategy
            
        Returns:
            Configuration class or None if not found
        """
        strategy_type = strategy_type.lower()
        return self._registry.get(strategy_type)
    
    def get_instance(self, strategy_type: str, config_name: str) -> BaseConfiguration:
        """
        Create a configuration instance
        
        Args:
            strategy_type: Type of strategy
            config_name: Name for the configuration
            
        Returns:
            New configuration instance
        """
        config_class = self.get_class(strategy_type)
        
        if not config_class:
            raise ConfigurationError(f"No configuration class registered for {strategy_type}")
        
        return config_class(strategy_type, config_name)
    
    def list_registered(self) -> Dict[str, str]:
        """
        List all registered strategy types
        
        Returns:
            Dict mapping strategy types to class names
        """
        return {
            strategy_type: config_class.__name__
            for strategy_type, config_class in self._registry.items()
        }
    
    def is_registered(self, strategy_type: str) -> bool:
        """
        Check if a strategy type is registered
        
        Args:
            strategy_type: Type of strategy to check
            
        Returns:
            True if registered, False otherwise
        """
        strategy_type = strategy_type.lower()
        return strategy_type in self._registry
    
    def register_parser(self, strategy_type: str, parser_class: Type) -> None:
        """
        Register a parser for a strategy type
        
        Args:
            strategy_type: Type of strategy
            parser_class: Parser class for the strategy
        """
        strategy_type = strategy_type.lower()
        self._parsers[strategy_type] = parser_class
        logger.info(f"Registered parser for {strategy_type}")
    
    def get_parser(self, strategy_type: str) -> Optional[Type]:
        """
        Get parser class for a strategy type
        
        Args:
            strategy_type: Type of strategy
            
        Returns:
            Parser class or None if not found
        """
        strategy_type = strategy_type.lower()
        return self._parsers.get(strategy_type)
    
    def register_validator(self, strategy_type: str, validator_class: Type) -> None:
        """
        Register a validator for a strategy type
        
        Args:
            strategy_type: Type of strategy
            validator_class: Validator class for the strategy
        """
        strategy_type = strategy_type.lower()
        self._validators[strategy_type] = validator_class
        logger.info(f"Registered validator for {strategy_type}")
    
    def get_validator(self, strategy_type: str) -> Optional[Type]:
        """
        Get validator class for a strategy type
        
        Args:
            strategy_type: Type of strategy
            
        Returns:
            Validator class or None if not found
        """
        strategy_type = strategy_type.lower()
        return self._validators.get(strategy_type)
    
    def register_schema(self, strategy_type: str, schema: Dict) -> None:
        """
        Register a JSON schema for a strategy type
        
        Args:
            strategy_type: Type of strategy
            schema: JSON schema for the strategy
        """
        strategy_type = strategy_type.lower()
        self._schemas[strategy_type] = schema
        logger.info(f"Registered schema for {strategy_type}")
    
    def get_schema(self, strategy_type: str) -> Optional[Dict]:
        """
        Get JSON schema for a strategy type
        
        Args:
            strategy_type: Type of strategy
            
        Returns:
            JSON schema or None if not found
        """
        strategy_type = strategy_type.lower()
        return self._schemas.get(strategy_type)
    
    def get_all_schemas(self) -> Dict[str, Dict]:
        """
        Get all registered schemas
        
        Returns:
            Dict mapping strategy types to schemas
        """
        return self._schemas.copy()
    
    def clear(self) -> None:
        """Clear all registrations"""
        self._registry.clear()
        self._parsers.clear()
        self._validators.clear()
        self._schemas.clear()
        logger.info("Cleared all registrations")
    
    def __repr__(self) -> str:
        return f"ConfigurationRegistry(registered={len(self._registry)}, parsers={len(self._parsers)}, validators={len(self._validators)}, schemas={len(self._schemas)})"
    
    def __str__(self) -> str:
        registered = ', '.join(sorted(self._registry.keys()))
        return f"ConfigurationRegistry: {registered}" if registered else "ConfigurationRegistry: empty"