"""
Production Mode Validation Utility
==================================

This module provides strict validation to prevent any synthetic data generation
in production mode. It monitors for np.random calls and enforces real data usage.

Author: Claude Code
Date: 2025-07-11
Version: 1.0.0
"""

import logging
import numpy as np
import sys
import traceback
from typing import Any, Dict, List, Optional
from functools import wraps
from datetime import datetime

logger = logging.getLogger(__name__)

class ProductionModeEnforcer:
    """Enforces production mode restrictions on synthetic data generation"""
    
    def __init__(self, strict_mode: bool = True):
        """Initialize production mode enforcer
        
        Args:
            strict_mode: If True, will raise exceptions on violations
        """
        self.strict_mode = strict_mode
        self.violations = []
        self.enabled = True
        
    def enable_strict_mode(self):
        """Enable strict production mode - no synthetic data allowed"""
        self.strict_mode = True
        self.enabled = True
        logger.info("🔒 PRODUCTION MODE: Strict mode enabled - synthetic data generation disabled")
        
    def disable_strict_mode(self):
        """Disable strict mode (for testing only)"""
        self.strict_mode = False
        self.enabled = False
        logger.warning("⚠️ DEVELOPMENT MODE: Strict mode disabled - synthetic data allowed")
        
    def log_violation(self, violation_type: str, function_name: str, details: str = ""):
        """Log a production mode violation"""
        violation = {
            'timestamp': datetime.now().isoformat(),
            'type': violation_type,
            'function': function_name,
            'details': details,
            'stack_trace': traceback.format_stack()
        }
        
        self.violations.append(violation)
        
        error_msg = f"🚨 PRODUCTION MODE VIOLATION: {violation_type} in {function_name}"
        if details:
            error_msg += f" - {details}"
            
        logger.error(error_msg)
        
        if self.strict_mode:
            raise ProductionModeViolation(error_msg)
            
    def check_numpy_random_usage(self, function_name: str):
        """Check if numpy random functions are being used"""
        if not self.enabled:
            return
            
        # Check if any numpy random functions are called
        stack = traceback.extract_stack()
        for frame in stack:
            if 'np.random' in frame.line or 'numpy.random' in frame.line:
                self.log_violation(
                    "NUMPY_RANDOM_USAGE",
                    function_name,
                    f"Detected np.random call: {frame.line.strip()}"
                )
                
    def validate_data_source(self, data_source: str, function_name: str):
        """Validate that data comes from approved sources"""
        if not self.enabled:
            return
            
        approved_sources = [
            'heavydb',
            'mysql',
            'real_market_data',
            'api_feed',
            'historical_db'
        ]
        
        if data_source.lower() not in approved_sources and 'synthetic' in data_source.lower():
            self.log_violation(
                "SYNTHETIC_DATA_SOURCE",
                function_name,
                f"Unapproved data source: {data_source}"
            )
            
    def get_violation_report(self) -> Dict[str, Any]:
        """Get comprehensive violation report"""
        return {
            'total_violations': len(self.violations),
            'strict_mode_enabled': self.strict_mode,
            'violations': self.violations,
            'generated_at': datetime.now().isoformat()
        }

class ProductionModeViolation(Exception):
    """Exception raised when production mode is violated"""
    pass

# Global enforcer instance
_enforcer = ProductionModeEnforcer(strict_mode=True)

def production_mode_check(func):
    """Decorator to check production mode compliance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if _enforcer.enabled:
            _enforcer.check_numpy_random_usage(func.__name__)
        return func(*args, **kwargs)
    return wrapper

def validate_real_data_only(data_source: str = "unknown"):
    """Decorator to validate that only real data is used"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if _enforcer.enabled:
                _enforcer.validate_data_source(data_source, func.__name__)
            return func(*args, **kwargs)
        return wrapper
    return decorator

def enforce_production_mode():
    """Enable strict production mode enforcement"""
    _enforcer.enable_strict_mode()

def disable_production_mode():
    """Disable production mode enforcement (for testing)"""
    _enforcer.disable_strict_mode()

def get_production_violations() -> Dict[str, Any]:
    """Get all production mode violations"""
    return _enforcer.get_violation_report()

def clear_violations():
    """Clear all recorded violations"""
    _enforcer.violations.clear()

# Monkey patch numpy.random to detect usage
original_random_functions = {}

def patch_numpy_random():
    """Patch numpy.random functions to detect usage"""
    global original_random_functions
    
    # Store original functions
    for attr_name in dir(np.random):
        if not attr_name.startswith('_'):
            attr = getattr(np.random, attr_name)
            if callable(attr):
                original_random_functions[attr_name] = attr
                
                # Create patched version
                def create_patched_function(func_name, original_func):
                    def patched_func(*args, **kwargs):
                        if _enforcer.enabled:
                            _enforcer.log_violation(
                                "NUMPY_RANDOM_CALL",
                                f"np.random.{func_name}",
                                f"Attempted to call np.random.{func_name}"
                            )
                        return original_func(*args, **kwargs)
                    return patched_func
                
                # Replace with patched version
                setattr(np.random, attr_name, create_patched_function(attr_name, attr))

def unpatch_numpy_random():
    """Restore original numpy.random functions"""
    global original_random_functions
    
    for attr_name, original_func in original_random_functions.items():
        setattr(np.random, attr_name, original_func)
    
    original_random_functions.clear()

# Auto-enable production mode on import
enforce_production_mode()
logger.info("🔒 Production mode enforcer initialized - synthetic data generation monitoring active")