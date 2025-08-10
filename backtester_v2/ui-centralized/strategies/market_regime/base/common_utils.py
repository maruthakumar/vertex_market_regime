"""
Common Utilities - Shared utilities for Market Regime Analysis
=============================================================

Common utility functions used across all market regime components.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
from datetime import datetime, timedelta
import json
import os

logger = logging.getLogger(__name__)


class DataValidator:
    """Data validation utilities"""
    
    @staticmethod
    def validate_option_data(data: pd.DataFrame) -> Dict[str, Any]:
        """Validate option data DataFrame"""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'data_quality_score': 1.0
        }
        
        required_columns = ['strike', 'option_type', 'dte', 'volume', 'oi']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Missing required columns: {missing_columns}")
            validation_result['data_quality_score'] -= 0.5
        
        if data.empty:
            validation_result['is_valid'] = False
            validation_result['errors'].append("DataFrame is empty")
            validation_result['data_quality_score'] = 0.0
            return validation_result
        
        # Check for data quality issues
        if 'volume' in data.columns:
            zero_volume_pct = (data['volume'] == 0).sum() / len(data)
            if zero_volume_pct > 0.5:
                validation_result['warnings'].append(f"High percentage of zero volume data: {zero_volume_pct:.2%}")
                validation_result['data_quality_score'] -= 0.2
        
        return validation_result
    
    @staticmethod
    def validate_numerical_data(data: Union[float, int, np.ndarray], 
                               min_val: Optional[float] = None,
                               max_val: Optional[float] = None,
                               allow_nan: bool = False) -> bool:
        """Validate numerical data"""
        try:
            if isinstance(data, (list, np.ndarray)):
                data = np.array(data)
                if not allow_nan and np.isnan(data).any():
                    return False
                if min_val is not None and np.any(data < min_val):
                    return False
                if max_val is not None and np.any(data > max_val):
                    return False
            else:
                if not allow_nan and np.isnan(data):
                    return False
                if min_val is not None and data < min_val:
                    return False
                if max_val is not None and data > max_val:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating numerical data: {e}")
            return False


class MathUtils:
    """Mathematical utility functions"""
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe division with default value for zero denominator"""
        try:
            if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
                return default
            result = numerator / denominator
            return result if not (np.isnan(result) or np.isinf(result)) else default
        except:
            return default
    
    @staticmethod
    def calculate_percentage_change(current: float, previous: float) -> float:
        """Calculate percentage change with safety checks"""
        if previous == 0 or np.isnan(previous) or np.isnan(current):
            return 0.0
        return ((current - previous) / previous) * 100
    
    @staticmethod
    def normalize_weights(weights: np.ndarray) -> np.ndarray:
        """Normalize weights to sum to 1"""
        try:
            weights = np.array(weights)
            total = np.sum(weights)
            if total == 0 or np.isnan(total):
                return np.ones(len(weights)) / len(weights)  # Equal weights
            return weights / total
        except:
            return np.ones(len(weights)) / len(weights)
    
    @staticmethod
    def calculate_exponential_decay(values: np.ndarray, decay_factor: float = 0.9) -> np.ndarray:
        """Calculate exponentially decaying weights"""
        try:
            n = len(values)
            weights = np.array([decay_factor ** i for i in range(n)])
            return weights / np.sum(weights)
        except:
            return np.ones(len(values)) / len(values)
    
    @staticmethod
    def rolling_statistics(data: np.ndarray, window: int) -> Dict[str, float]:
        """Calculate rolling statistics"""
        try:
            if len(data) < window:
                recent_data = data
            else:
                recent_data = data[-window:]
            
            return {
                'mean': float(np.mean(recent_data)),
                'std': float(np.std(recent_data)),
                'min': float(np.min(recent_data)),
                'max': float(np.max(recent_data)),
                'median': float(np.median(recent_data))
            }
        except:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0}


class TimeUtils:
    """Time and date utility functions"""
    
    @staticmethod
    def get_market_session(timestamp: datetime) -> str:
        """Determine market session from timestamp"""
        try:
            hour = timestamp.hour
            minute = timestamp.minute
            time_minutes = hour * 60 + minute
            
            # Market sessions (IST)
            if 570 <= time_minutes < 600:  # 9:30-10:00
                return 'opening'
            elif 600 <= time_minutes < 690:  # 10:00-11:30
                return 'early_session'
            elif 690 <= time_minutes < 810:  # 11:30-13:30
                return 'mid_session'
            elif 810 <= time_minutes < 870:  # 13:30-14:30
                return 'late_session'
            elif 870 <= time_minutes < 930:  # 14:30-15:30
                return 'closing'
            else:
                return 'after_hours'
                
        except:
            return 'unknown'
    
    @staticmethod
    def calculate_dte(expiry_date: datetime, current_date: Optional[datetime] = None) -> int:
        """Calculate days to expiry"""
        try:
            if current_date is None:
                current_date = datetime.now()
            
            delta = expiry_date - current_date
            return max(0, delta.days)
        except:
            return 0
    
    @staticmethod
    def get_business_days_between(start_date: datetime, end_date: datetime) -> int:
        """Get number of business days between two dates"""
        try:
            return pd.bdate_range(start_date, end_date).size
        except:
            return 0


class OptionUtils:
    """Option-specific utility functions"""
    
    @staticmethod
    def calculate_moneyness(strike: float, spot: float) -> float:
        """Calculate moneyness (strike/spot)"""
        try:
            return MathUtils.safe_divide(strike, spot, 1.0)
        except:
            return 1.0
    
    @staticmethod
    def classify_option_position(strike: float, spot: float, option_type: str) -> str:
        """Classify option as ITM/ATM/OTM"""
        try:
            moneyness = OptionUtils.calculate_moneyness(strike, spot)
            
            if option_type.upper() == 'CE':
                if moneyness < 0.98:
                    return 'ITM'
                elif moneyness <= 1.02:
                    return 'ATM'
                else:
                    return 'OTM'
            elif option_type.upper() == 'PE':
                if moneyness > 1.02:
                    return 'ITM'
                elif moneyness >= 0.98:
                    return 'ATM'
                else:
                    return 'OTM'
            else:
                return 'UNKNOWN'
        except:
            return 'UNKNOWN'
    
    @staticmethod
    def filter_liquid_options(data: pd.DataFrame, 
                             min_volume: int = 10, 
                             min_oi: int = 50) -> pd.DataFrame:
        """Filter for liquid options"""
        try:
            filtered = data.copy()
            
            if 'volume' in data.columns:
                filtered = filtered[filtered['volume'] >= min_volume]
            
            if 'oi' in data.columns:
                filtered = filtered[filtered['oi'] >= min_oi]
            
            return filtered
        except:
            return data
    
    @staticmethod
    def find_atm_strikes(data: pd.DataFrame, spot: float, option_type: Optional[str] = None) -> pd.DataFrame:
        """Find ATM strikes for given option type"""
        try:
            if option_type:
                data = data[data['option_type'] == option_type]
            
            # Calculate distance from ATM
            data = data.copy()
            data['atm_distance'] = abs(data['strike'] - spot)
            
            # Find closest strike for each option type
            if option_type:
                return data.loc[data['atm_distance'].idxmin():data['atm_distance'].idxmin()]
            else:
                atm_options = []
                for opt_type in data['option_type'].unique():
                    type_data = data[data['option_type'] == opt_type]
                    atm_idx = type_data['atm_distance'].idxmin()
                    atm_options.append(type_data.loc[atm_idx])
                
                return pd.DataFrame(atm_options)
        except:
            return pd.DataFrame()


class ConfigUtils:
    """Configuration utility functions"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            return {}
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str) -> bool:
        """Save configuration to JSON file"""
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error(f"Error saving config to {config_path}: {e}")
            return False
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configuration dictionaries"""
        try:
            merged = base_config.copy()
            
            for key, value in override_config.items():
                if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                    merged[key] = ConfigUtils.merge_configs(merged[key], value)
                else:
                    merged[key] = value
            
            return merged
        except:
            return base_config


class LoggingUtils:
    """Logging utility functions"""
    
    @staticmethod
    def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
        """Setup logger with consistent formatting"""
        logger = logging.getLogger(name)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(level)
        
        return logger
    
    @staticmethod
    def log_performance(func_name: str, execution_time: float, data_size: int = 0):
        """Log performance metrics"""
        logger = logging.getLogger('performance')
        logger.info(f"{func_name} executed in {execution_time:.4f}s (data_size: {data_size})")


class CacheUtils:
    """Caching utility functions"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set cached value with LRU eviction"""
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            # Remove least recently used
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        self.cache[key] = value
        self.access_order.append(key)
    
    def clear(self) -> None:
        """Clear cache"""
        self.cache.clear()
        self.access_order.clear()


class ErrorHandler:
    """Error handling utilities"""
    
    @staticmethod
    def safe_execute(func, *args, default=None, **kwargs):
        """Safely execute function with error handling"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error executing {func.__name__}: {e}")
            return default
    
    @staticmethod
    def retry_on_failure(func, max_retries: int = 3, delay: float = 1.0):
        """Retry function on failure"""
        import time
        
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                time.sleep(delay)


# Global utilities instances
cache_utils = CacheUtils()
math_utils = MathUtils()
time_utils = TimeUtils()
option_utils = OptionUtils()
config_utils = ConfigUtils()
data_validator = DataValidator()
logging_utils = LoggingUtils()
error_handler = ErrorHandler()


# Helper functions for backward compatibility
def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Backward compatible safe divide function"""
    return math_utils.safe_divide(numerator, denominator, default)


def validate_option_data(data: pd.DataFrame) -> bool:
    """Backward compatible data validation"""
    return data_validator.validate_option_data(data)['is_valid']


def calculate_moneyness(strike: float, spot: float) -> float:
    """Backward compatible moneyness calculation"""
    return option_utils.calculate_moneyness(strike, spot)