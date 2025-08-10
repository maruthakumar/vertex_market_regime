"""
Configuration API Bridge for Next.js Integration
 
This module provides a bridge between Next.js API routes and the Python
configuration management system, enabling seamless integration between
the frontend and backend configuration systems.
"""

import sys
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add the configurations directory to Python path
CONFIGS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(CONFIGS_DIR))

from core.config_manager import ConfigurationManager
from core.exceptions import ConfigurationError, ValidationError
from strategies.tbs_config import TBSConfiguration
from strategies.tv_config import TVConfiguration
from strategies.orb_config import ORBConfiguration
from strategies.oi_config import OIConfiguration
from strategies.ml_config import MLConfiguration
from strategies.pos_config import POSConfiguration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigAPIBridge:
    """
    Bridge class for handling API operations between Next.js and Python backend
    """
    
    def __init__(self):
        """Initialize the API bridge"""
        self.manager = ConfigurationManager()
        self._register_configuration_classes()
        logger.info("ConfigAPIBridge initialized")
    
    def _register_configuration_classes(self):
        """Register all strategy configuration classes"""
        strategy_classes = {
            'tbs': TBSConfiguration,
            'tv': TVConfiguration, 
            'orb': ORBConfiguration,
            'oi': OIConfiguration,
            'ml': MLConfiguration,
            'pos': POSConfiguration
        }
        
        for strategy_type, config_class in strategy_classes.items():
            try:
                self.manager.register_configuration_class(strategy_type, config_class)
                logger.info(f"Registered {strategy_type} configuration class")
            except Exception as e:
                logger.error(f"Failed to register {strategy_type} configuration class: {e}")
    
    def execute_operation(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a configuration operation
        
        Args:
            operation: Operation dictionary containing action and parameters
            
        Returns:
            Result dictionary with success status and data/error
        """
        try:
            action = operation.get('action')
            strategy_type = operation.get('strategy_type')
            config_name = operation.get('config_name')
            data = operation.get('data')
            options = operation.get('options', {})
            
            logger.info(f"Executing operation: {action} for {strategy_type}/{config_name}")
            
            # Route to appropriate handler
            if action == 'list':
                return self._handle_list(strategy_type, options)
            elif action == 'get':
                return self._handle_get(strategy_type, config_name, options)
            elif action == 'create':
                return self._handle_create(strategy_type, config_name, data, options)
            elif action == 'update':
                return self._handle_update(strategy_type, config_name, data, options)
            elif action == 'delete':
                return self._handle_delete(strategy_type, config_name, options)
            elif action == 'validate':
                return self._handle_validate(strategy_type, config_name, data, options)
            elif action == 'clone':
                return self._handle_clone(strategy_type, config_name, options)
            elif action == 'merge':
                return self._handle_merge(strategy_type, config_name, data, options)
            else:
                return {
                    'success': False,
                    'error': f'Unknown action: {action}'
                }
                
        except Exception as e:
            logger.error(f"Operation execution error: {e}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': f'Internal error: {str(e)}'
            }
    
    def _handle_list(self, strategy_type: Optional[str], options: Dict[str, Any]) -> Dict[str, Any]:
        """Handle list configurations operation"""
        try:
            configs = self.manager.list_configurations(strategy_type)
            
            result = {
                'success': True,
                'data': configs
            }
            
            if options.get('include_metadata'):
                stats = self.manager.get_statistics()
                result['metadata'] = {
                    'total_configs': stats['total_configurations'],
                    'strategy_types': list(stats['configurations_by_type'].keys()),
                    'last_modified': datetime.now().isoformat(),
                    'cache_size': stats['cache_size'],
                    'active_watchers': stats['active_watchers']
                }
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to list configurations: {str(e)}'
            }
    
    def _handle_get(self, strategy_type: str, config_name: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get configuration operation"""
        try:
            config = self.manager.get_configuration(strategy_type, config_name)
            
            if not config:
                return {
                    'success': False,
                    'error': f'Configuration {strategy_type}/{config_name} not found'
                }
            
            result = {
                'success': True,
                'data': config.to_dict()
            }
            
            # Add strategy-specific metadata
            if hasattr(config, 'get_strike_selection_config'):
                result['strike_selection_config'] = config.get_strike_selection_config()
            
            if hasattr(config, 'get_risk_limits'):
                result['risk_limits'] = config.get_risk_limits()
                
            if hasattr(config, 'is_multileg_strategy'):
                result['is_multileg'] = config.is_multileg_strategy()
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to get configuration: {str(e)}'
            }
    
    def _handle_create(self, strategy_type: str, config_name: str, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Handle create configuration operation"""
        try:
            # Get configuration class
            config_class = self.manager._registry.get_class(strategy_type)
            if not config_class:
                return {
                    'success': False,
                    'error': f'No configuration class registered for {strategy_type}'
                }
            
            # Create configuration instance
            config = config_class(config_name)
            
            # Set data
            if data:
                config.from_dict(data)
            
            # Validate
            try:
                config.validate()
                validation_result = {'valid': True, 'errors': {}}
            except ValidationError as e:
                validation_result = {'valid': False, 'errors': e.errors}
            
            if not validation_result['valid']:
                return {
                    'success': False,
                    'validation_errors': validation_result['errors']
                }
            
            # Save configuration
            file_path = self.manager.save_configuration(config)
            
            result = {
                'success': True,
                'data': config.to_dict(),
                'file_path': file_path
            }
            
            # Add strategy-specific metadata
            if hasattr(config, 'get_strike_selection_config'):
                result['strike_selection_config'] = config.get_strike_selection_config()
            
            if hasattr(config, 'get_risk_limits'):
                result['risk_limits'] = config.get_risk_limits()
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to create configuration: {str(e)}'
            }
    
    def _handle_update(self, strategy_type: str, config_name: str, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Handle update configuration operation"""
        try:
            # Get existing configuration
            config = self.manager.get_configuration(strategy_type, config_name)
            if not config:
                return {
                    'success': False,
                    'error': f'Configuration {strategy_type}/{config_name} not found'
                }
            
            # Update data
            merge_strategy = options.get('merge_strategy', 'override')
            
            if merge_strategy == 'override':
                config.from_dict(data)
            elif merge_strategy == 'merge':
                existing_data = config.to_dict()
                merged_data = {**existing_data, **data}
                config.from_dict(merged_data)
            elif merge_strategy == 'deep_merge':
                existing_data = config.to_dict()
                merged_data = self._deep_merge(existing_data, data)
                config.from_dict(merged_data)
            
            # Validate
            try:
                config.validate()
                validation_result = {'valid': True, 'errors': {}}
            except ValidationError as e:
                validation_result = {'valid': False, 'errors': e.errors}
            
            if not validation_result['valid']:
                return {
                    'success': False,
                    'validation_errors': validation_result['errors']
                }
            
            # Save updated configuration
            file_path = self.manager.save_configuration(config)
            
            result = {
                'success': True,
                'data': config.to_dict(),
                'file_path': file_path
            }
            
            # Add strategy-specific metadata
            if hasattr(config, 'get_strike_selection_config'):
                result['strike_selection_config'] = config.get_strike_selection_config()
            
            if hasattr(config, 'get_risk_limits'):
                result['risk_limits'] = config.get_risk_limits()
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to update configuration: {str(e)}'
            }
    
    def _handle_delete(self, strategy_type: str, config_name: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Handle delete configuration operation"""
        try:
            force = options.get('force', False)
            
            # Check if configuration exists
            config = self.manager.get_configuration(strategy_type, config_name)
            if not config:
                return {
                    'success': False,
                    'error': f'Configuration {strategy_type}/{config_name} not found'
                }
            
            # Delete configuration
            success = self.manager.delete_configuration(strategy_type, config_name)
            
            if success:
                return {
                    'success': True,
                    'message': f'Configuration {strategy_type}/{config_name} deleted successfully'
                }
            else:
                return {
                    'success': False,
                    'error': f'Failed to delete configuration {strategy_type}/{config_name}'
                }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to delete configuration: {str(e)}'
            }
    
    def _handle_validate(self, strategy_type: str, config_name: str, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Handle validate configuration operation"""
        try:
            # Get configuration class
            config_class = self.manager._registry.get_class(strategy_type)
            if not config_class:
                return {
                    'success': False,
                    'error': f'No configuration class registered for {strategy_type}'
                }
            
            # Create temporary configuration instance
            config = config_class(config_name or 'temp')
            
            # Set data
            if data:
                config.from_dict(data)
            
            # Validate
            try:
                config.validate()
                return {
                    'success': True,
                    'valid': True,
                    'errors': {},
                    'data': config.to_dict()
                }
            except ValidationError as e:
                return {
                    'success': True,
                    'valid': False,
                    'errors': e.errors,
                    'data': config.to_dict()
                }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to validate configuration: {str(e)}'
            }
    
    def _handle_clone(self, strategy_type: str, config_name: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Handle clone configuration operation"""
        try:
            new_name = options.get('new_name')
            if not new_name:
                return {
                    'success': False,
                    'error': 'new_name is required for clone operation'
                }
            
            cloned_config = self.manager.clone_configuration(strategy_type, config_name, new_name)
            
            return {
                'success': True,
                'data': cloned_config.to_dict(),
                'message': f'Configuration cloned as {new_name}'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to clone configuration: {str(e)}'
            }
    
    def _handle_merge(self, strategy_type: str, config_name: str, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Handle merge configurations operation"""
        try:
            config2_name = options.get('config2_name')
            merged_name = options.get('merged_name')
            merge_strategy = options.get('merge_strategy', 'override')
            
            if not config2_name or not merged_name:
                return {
                    'success': False,
                    'error': 'config2_name and merged_name are required for merge operation'
                }
            
            config1 = self.manager.get_configuration(strategy_type, config_name)
            config2 = self.manager.get_configuration(strategy_type, config2_name)
            
            if not config1 or not config2:
                return {
                    'success': False,
                    'error': 'One or both configurations not found'
                }
            
            merged_config = self.manager.merge_configurations(config1, config2, merged_name, merge_strategy)
            
            return {
                'success': True,
                'data': merged_config.to_dict(),
                'message': f'Configurations merged as {merged_name}'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to merge configurations: {str(e)}'
            }
    
    def _deep_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result

# Global bridge instance
_bridge_instance = None

def get_bridge() -> ConfigAPIBridge:
    """Get or create bridge instance"""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = ConfigAPIBridge()
    return _bridge_instance

def execute_operation(operation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for executing operations from Next.js
    
    Args:
        operation: Operation dictionary
        
    Returns:
        Result dictionary
    """
    bridge = get_bridge()
    return bridge.execute_operation(operation)

if __name__ == "__main__":
    # Handle command line execution
    if len(sys.argv) > 1:
        try:
            operation_json = sys.argv[1]
            operation = json.loads(operation_json)
            result = execute_operation(operation)
            print(json.dumps(result))
        except Exception as e:
            print(json.dumps({
                'success': False,
                'error': f'Command line execution error: {str(e)}'
            }))
    else:
        # Interactive mode for testing
        print("ConfigAPIBridge - Interactive Mode")
        print("Available operations: list, get, create, update, delete, validate, clone, merge")
        
        # Example operations
        examples = [
            {'action': 'list', 'strategy_type': 'tbs'},
            {'action': 'get', 'strategy_type': 'tbs', 'config_name': 'sample_tbs'},
            {'action': 'validate', 'strategy_type': 'tbs', 'config_name': 'test', 'data': {'portfolio_settings': {'capital': 100000}}}
        ]
        
        for example in examples:
            print(f"\nExample: {example}")
            result = execute_operation(example)
            print(f"Result: {result}")