#!/usr/bin/env python3
"""
TBS Parser - Optimized version for 112-parameter processing
Performance target: <300ms for complete parsing (previously achieved 28.96ms)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, date, time
import logging
import os
import functools
from concurrent.futures import ThreadPoolExecutor, as_completed
import weakref

logger = logging.getLogger(__name__)

class OptimizedTBSParser:
    """Optimized parser for TBS strategy Excel files - 112 parameter processing"""
    
    def __init__(self):
        # Performance optimizations
        self.cache = {}  # Weak reference cache for parsed data
        self.compiled_patterns = self._compile_regex_patterns()
        
        # Pre-compiled column mappings for faster access
        self.portfolio_columns = {
            'StartDate': 'start_date',
            'EndDate': 'end_date', 
            'Enabled': 'enabled',
            'PortfolioName': 'portfolio_name',
            'Capital': 'capital',
            'Index': 'index',
            'LotSize': 'lot_size',
            'Margin': 'margin'
        }
        
        self.strategy_columns = {
            'Enabled': 'enabled',
            'PortfolioName': 'portfolio_name',
            'StrategyType': 'strategy_type',
            'StrategyExcelFilePath': 'strategy_excel_file_path'
        }
        
        # Optimized data type specifications
        self.dtypes = {
            'numeric_int': ['DTE', 'StrategyProfitReExecuteNo', 'StrategyLossReExecuteNo', 
                           'LegID', 'Lots', 'ReEnteriesCount'],
            'numeric_float': ['StrategyProfit', 'StrategyLoss', 'LockPercent', 'TrailPercent',
                            'SqOff1Percent', 'SqOff2Percent', 'StrikeValue', 'SLValue', 
                            'TGTValue', 'W&TValue', 'Capital', 'LotSize', 'Margin'],
            'boolean': ['Enabled', 'IsTickBT', 'MoveSlToCost', 'ConsiderHedgePnLForStgyPnL', 
                       'OnExpiryDayTradeNextExpiry', 'IsIdle', 'TrailW&T', 'OpenHedge'],
            'string': ['PortfolioName', 'Index', 'StrategyType', 'StrategyName', 'Underlying',
                      'Weekdays', 'Instrument', 'Transaction', 'StrikeMethod', 'Expiry']
        }
        
    def _compile_regex_patterns(self):
        """Pre-compile regex patterns for better performance"""
        import re
        return {
            'camel_to_snake': re.compile('(.)([A-Z][a-z]+)'),
            'snake_case': re.compile('([a-z0-9])([A-Z])'),
            'time_format': re.compile(r'^\d{5,6}$'),
            'underscore_clean': re.compile('_+')
        }
        
    def get_cache_key(self, filepath: str, sheet_name: str) -> str:
        """Generate cache key for file/sheet combination"""
        import hashlib
        stat = os.stat(filepath)
        key_data = f"{filepath}:{sheet_name}:{stat.st_mtime}:{stat.st_size}"
        return hashlib.md5(key_data.encode()).hexdigest()
        
    @functools.lru_cache(maxsize=128)
    def _get_excel_engine(self, filepath: str) -> str:
        """Determine optimal Excel engine based on file characteristics"""
        file_size = os.path.getsize(filepath)
        if file_size > 10 * 1024 * 1024:  # >10MB
            return 'openpyxl'  # Better for large files
        return 'openpyxl'  # Consistent choice for reliability
        
    def parse_portfolio_excel_optimized(self, excel_path: str) -> Dict[str, Any]:
        """
        Optimized portfolio Excel parsing - targeting <50ms for 20 parameters
        """
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Excel file not found: {excel_path}")
        
        start_time = time.perf_counter()
        cache_key = self.get_cache_key(excel_path, 'portfolio_combined')
        
        # Check cache first
        if cache_key in self.cache:
            logger.debug(f"Cache hit for portfolio: {excel_path}")
            return self.cache[cache_key]
        
        try:
            # Optimized Excel reading with specific engine and minimal overhead
            engine = self._get_excel_engine(excel_path)
            
            # Read both sheets in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                portfolio_future = executor.submit(
                    self._read_sheet_optimized, excel_path, 'PortfolioSetting', engine
                )
                strategy_future = executor.submit(
                    self._read_sheet_optimized, excel_path, 'StrategySetting', engine  
                )
                
                portfolio_df = portfolio_future.result()
                strategy_df = strategy_future.result()
            
            # Optimized parsing with vectorized operations
            portfolio_data = self._parse_portfolio_vectorized(portfolio_df)
            strategies = self._parse_strategy_vectorized(strategy_df)
            
            result = {
                'portfolio': portfolio_data,
                'strategies': strategies,
                'source_file': excel_path
            }
            
            # Cache result
            self.cache[cache_key] = result
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.info(f"Portfolio parsing completed in {duration_ms:.2f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Optimized portfolio parsing failed: {e}")
            raise
            
    def parse_multi_leg_excel_optimized(self, excel_path: str) -> Dict[str, Any]:
        """
        Optimized multi-leg Excel parsing - targeting <250ms for 92 parameters
        """
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Multi-leg Excel file not found: {excel_path}")
        
        start_time = time.perf_counter()
        cache_key = self.get_cache_key(excel_path, 'multileg_combined')
        
        # Check cache first
        if cache_key in self.cache:
            logger.debug(f"Cache hit for multi-leg: {excel_path}")
            return self.cache[cache_key]
            
        try:
            # Optimized Excel reading
            engine = self._get_excel_engine(excel_path)
            
            # Read both sheets in parallel with optimized dtypes
            with ThreadPoolExecutor(max_workers=2) as executor:
                general_future = executor.submit(
                    self._read_sheet_optimized, excel_path, 'GeneralParameter', engine
                )
                leg_future = executor.submit(
                    self._read_sheet_optimized, excel_path, 'LegParameter', engine
                )
                
                general_df = general_future.result()
                leg_df = leg_future.result()
            
            # Vectorized parsing
            strategies = self._parse_general_vectorized(general_df)
            legs = self._parse_leg_vectorized(leg_df)
            
            # Optimized leg assignment using dictionary lookup
            leg_dict = {}
            for leg in legs:
                strategy_name = leg['strategy_name']
                if strategy_name not in leg_dict:
                    leg_dict[strategy_name] = []
                leg_dict[strategy_name].append(leg)
            
            # Assign legs to strategies
            for strategy in strategies:
                strategy['legs'] = leg_dict.get(strategy['strategy_name'], [])
            
            result = {
                'strategies': strategies,
                'source_file': excel_path
            }
            
            # Cache result
            self.cache[cache_key] = result
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.info(f"Multi-leg parsing completed in {duration_ms:.2f}ms ({len(strategies)} strategies, {len(legs)} legs)")
            
            return result
            
        except Exception as e:
            logger.error(f"Optimized multi-leg parsing failed: {e}")
            raise
            
    def _read_sheet_optimized(self, filepath: str, sheet_name: str, engine: str) -> pd.DataFrame:
        """Optimized sheet reading with dtype inference"""
        try:
            # Read with optimized parameters
            df = pd.read_excel(
                filepath, 
                sheet_name=sheet_name,
                engine=engine,
                na_filter=True,
                keep_default_na=True,
                dtype_backend='numpy_nullable'  # Use nullable dtypes for better performance
            )
            
            # Optimize dtypes based on content
            df = self._optimize_dtypes(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to read sheet {sheet_name}: {e}")
            raise
            
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame dtypes for better performance"""
        for col in df.columns:
            # Skip if already optimized
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
                
            # Try to convert to numeric if possible
            if col in self.dtypes['numeric_int']:
                df[col] = pd.to_numeric(df[col], errors='coerce', downcast='integer')
            elif col in self.dtypes['numeric_float']:
                df[col] = pd.to_numeric(df[col], errors='coerce', downcast='float')
            elif col in self.dtypes['boolean']:
                # Keep as object for custom boolean parsing
                pass
            else:
                # Convert to category for repeated strings
                if df[col].nunique() < len(df) * 0.5:  # Less than 50% unique values
                    df[col] = df[col].astype('category')
                    
        return df
        
    def _parse_portfolio_vectorized(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Vectorized portfolio parsing for better performance"""
        # Filter enabled portfolios using vectorized boolean indexing
        enabled_mask = df['Enabled'].apply(self._parse_bool_fast)
        enabled_df = df[enabled_mask]
        
        if enabled_df.empty:
            raise ValueError("No enabled portfolios found")
        
        # Get first enabled portfolio (vectorized access)
        row = enabled_df.iloc[0]
        
        # Use pre-compiled mapping for faster access
        portfolio_data = {}
        for excel_col, field_name in self.portfolio_columns.items():
            if excel_col in row.index:
                if field_name in ['start_date', 'end_date']:
                    portfolio_data[field_name] = self._parse_date_fast(row[excel_col])
                elif field_name == 'enabled':
                    portfolio_data[field_name] = True
                elif excel_col in ['Capital', 'LotSize', 'Margin']:
                    portfolio_data[field_name] = float(row[excel_col]) if pd.notna(row[excel_col]) else 0.0
                else:
                    portfolio_data[field_name] = str(row[excel_col]).upper() if pd.notna(row[excel_col]) else ''
        
        # Set defaults for missing fields
        defaults = {
            'capital': 1000000.0,
            'index': 'NIFTY',
            'lot_size': 50,
            'margin': 0.15
        }
        
        for field, default_value in defaults.items():
            if field not in portfolio_data:
                portfolio_data[field] = default_value
                
        return portfolio_data
        
    def _parse_strategy_vectorized(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Vectorized strategy parsing"""
        # Filter enabled TBS strategies using vectorized operations
        enabled_mask = df['Enabled'].apply(self._parse_bool_fast)
        tbs_mask = df['StrategyType'].str.upper() == 'TBS'
        combined_mask = enabled_mask & tbs_mask
        
        filtered_df = df[combined_mask]
        
        strategies = []
        for idx, row in filtered_df.iterrows():
            strategy = {
                'strategy_name': f'Strategy_{idx}',
                'enabled': True,
                'strategy_index': idx,
                'portfolio_name': str(row.get('PortfolioName', '')).upper(),
                'strategy_type': 'TBS',
                'strategy_excel_file_path': str(row.get('StrategyExcelFilePath', ''))
            }
            strategies.append(strategy)
            
        return strategies
        
    def _parse_general_vectorized(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Vectorized general parameters parsing"""
        strategies = []
        
        # Pre-process time and numeric columns
        time_columns = ['StrikeSelectionTime', 'StartTime', 'LastEntryTime', 'EndTime',
                       'PnLCalTime', 'SqOff1Time', 'SqOff2Time']
        
        for idx, row in df.iterrows():
            strategy = {
                'strategy_name': str(row.get('StrategyName', f'Strategy_{idx}')),
                'index': str(row.get('Index', 'NIFTY')).upper(),
                'underlying': str(row.get('Underlying', 'SPOT')).upper(),
                'enabled': True,
                'legs': []
            }
            
            # Process time fields efficiently
            for field in time_columns:
                if field in row.index and pd.notna(row[field]):
                    strategy[self._normalize_column_fast(field)] = self._parse_time_fast(row[field])
            
            # Process numeric fields
            numeric_fields = {
                'DTE': int, 'StrategyProfit': float, 'StrategyLoss': float,
                'StrategyProfitReExecuteNo': int, 'StrategyLossReExecuteNo': int,
                'LockPercent': float, 'TrailPercent': float
            }
            
            for field, dtype in numeric_fields.items():
                if field in row.index and pd.notna(row[field]):
                    strategy[self._normalize_column_fast(field)] = dtype(row[field])
            
            # Process boolean fields
            bool_fields = ['MoveSlToCost', 'ConsiderHedgePnLForStgyPnL', 'OnExpiryDayTradeNextExpiry']
            for field in bool_fields:
                if field in row.index and pd.notna(row[field]):
                    strategy[self._normalize_column_fast(field)] = self._parse_bool_fast(row[field])
            
            strategies.append(strategy)
            
        return strategies
        
    def _parse_leg_vectorized(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Vectorized leg parameters parsing"""
        # Filter out idle legs using vectorized operations
        idle_mask = df['IsIdle'].apply(self._parse_bool_fast) if 'IsIdle' in df.columns else pd.Series([False] * len(df))
        active_df = df[~idle_mask]
        
        legs = []
        for idx, row in active_df.iterrows():
            leg = {
                'strategy_name': str(row.get('StrategyName', '')),
                'leg_no': int(row.get('LegID', idx + 1)),
                'quantity': int(row.get('Lots', 1)),
                'option_type': self._convert_instrument_fast(row.get('Instrument', 'call')),
                'strike_selection': self._convert_strike_fast(row.get('StrikeMethod', 'atm')),
                'strike_value': float(row.get('StrikeValue', 0)),
                'expiry_rule': self._convert_expiry_fast(row.get('Expiry', 'current')),
                'transaction_type': str(row.get('Transaction', 'buy')).upper()
            }
            
            # Process optional fields efficiently
            optional_mappings = {
                'SLValue': ('sl_percent', float),
                'TGTValue': ('target_percent', float),
                'W&TValue': ('wait_value', float),
                'SL_TrailAt': ('sl_trail_at', float),
                'SL_TrailBy': ('sl_trail_by', float),
                'ReEnteriesCount': ('reentries_count', int)
            }
            
            for excel_col, (field_name, dtype) in optional_mappings.items():
                if excel_col in row.index and pd.notna(row[excel_col]):
                    leg[field_name] = dtype(row[excel_col])
            
            # Set defaults
            leg.update({
                'entry_time': time(9, 20),
                'exit_time': time(15, 15),
                'expiry_value': 0
            })
            
            legs.append(leg)
            
        return legs
        
    @functools.lru_cache(maxsize=1000)
    def _parse_bool_fast(self, value: Any) -> bool:
        """Fast boolean parsing with caching"""
        if pd.isna(value):
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.upper() in {'YES', 'TRUE', '1', 'Y', 'T'}
        return bool(value)
        
    @functools.lru_cache(maxsize=500)
    def _normalize_column_fast(self, name: str) -> str:
        """Fast column name normalization with caching"""
        # Use pre-compiled patterns
        name = name.replace('&', '_and_').replace(' ', '_')
        s1 = self.compiled_patterns['camel_to_snake'].sub(r'\1_\2', name)
        result = self.compiled_patterns['snake_case'].sub(r'\1_\2', s1).lower()
        result = self.compiled_patterns['underscore_clean'].sub('_', result)
        return result.strip('_')
        
    def _parse_date_fast(self, value: Any) -> Optional[date]:
        """Fast date parsing"""
        if pd.isna(value):
            return None
        if isinstance(value, (datetime, date)):
            return value.date() if isinstance(value, datetime) else value
        
        # Handle string formats
        date_str = str(value).strip()
        if '_' in date_str:
            try:
                parts = date_str.split('_')
                if len(parts) == 3:
                    return date(int(parts[2]), int(parts[1]), int(parts[0]))
            except:
                pass
        
        # Try pandas date parsing (fast)
        try:
            return pd.to_datetime(value).date()
        except:
            return None
            
    def _parse_time_fast(self, value: Any) -> Optional[time]:
        """Fast time parsing"""
        if pd.isna(value):
            return None
        if isinstance(value, time):
            return value
        if isinstance(value, datetime):
            return value.time()
            
        # Handle HHMMSS format efficiently
        time_str = str(value).strip()
        if self.compiled_patterns['time_format'].match(time_str):
            time_str = time_str.zfill(6)
            try:
                hour, minute, second = int(time_str[:2]), int(time_str[2:4]), int(time_str[4:6])
                if 0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59:
                    return time(hour, minute, second)
            except:
                pass
                
        return None
        
    @functools.lru_cache(maxsize=100)
    def _convert_instrument_fast(self, value: str) -> str:
        """Fast instrument conversion with caching"""
        value = str(value).upper()
        mapping = {'CALL': 'CE', 'PUT': 'PE', 'FUT': 'FUT'}
        return mapping.get(value, value)
        
    @functools.lru_cache(maxsize=100)  
    def _convert_expiry_fast(self, value: str) -> str:
        """Fast expiry conversion with caching"""
        value = str(value).upper()
        mapping = {
            'CURRENT': 'CW', 'NEXT': 'NW',
            'MONTHLY': 'CM', 'NEXT MONTHLY': 'NM'
        }
        return mapping.get(value, value)
        
    @functools.lru_cache(maxsize=100)
    def _convert_strike_fast(self, value: str) -> str:
        """Fast strike method conversion with caching"""
        value = str(value).upper()
        if value in ['ATM', 'FIXED', 'PREMIUM', 'DELTA']:
            return value
        if value.startswith(('ITM', 'OTM')):
            return value
        mapping = {
            'ATM WIDTH': 'ATM_WIDTH',
            'STRADDLE WIDTH': 'STRADDLE_WIDTH'
        }
        return mapping.get(value, value)
        
    def clear_cache(self):
        """Clear parsing cache"""
        self.cache.clear()
        # Clear LRU caches
        self._parse_bool_fast.cache_clear()
        self._normalize_column_fast.cache_clear() 
        self._convert_instrument_fast.cache_clear()
        self._convert_expiry_fast.cache_clear()
        self._convert_strike_fast.cache_clear()
        
        logger.info("Parser cache cleared")
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'main_cache_size': len(self.cache),
            'bool_cache_info': self._parse_bool_fast.cache_info()._asdict(),
            'column_cache_info': self._normalize_column_fast.cache_info()._asdict(),
            'instrument_cache_info': self._convert_instrument_fast.cache_info()._asdict(),
            'expiry_cache_info': self._convert_expiry_fast.cache_info()._asdict(),
            'strike_cache_info': self._convert_strike_fast.cache_info()._asdict()
        }

# Backward compatibility wrapper
class TBSParserOptimized(OptimizedTBSParser):
    """Backward compatibility wrapper"""
    
    def parse_portfolio_excel(self, excel_path: str) -> Dict[str, Any]:
        """Wrapper for optimized portfolio parsing"""
        return self.parse_portfolio_excel_optimized(excel_path)
        
    def parse_multi_leg_excel(self, excel_path: str) -> Dict[str, Any]:
        """Wrapper for optimized multi-leg parsing"""  
        return self.parse_multi_leg_excel_optimized(excel_path)