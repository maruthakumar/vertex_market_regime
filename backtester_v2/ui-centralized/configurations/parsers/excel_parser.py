"""
Generic Excel configuration parser
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import logging
import json

from .base_parser import BaseParser
from ..core.exceptions import ParsingError

logger = logging.getLogger(__name__)

class ExcelParser(BaseParser):
    """
    Generic Excel parser for configuration files
    
    This parser can handle various Excel formats and provides
    intelligent parsing of different sheet structures.
    """
    
    def __init__(self):
        """Initialize Excel parser"""
        super().__init__()
        self.supported_extensions = ['.xlsx', '.xls', '.xlsm', '.xlsb']
        
    def parse(self, file_path: str) -> Dict[str, Any]:
        """
        Parse Excel configuration file
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            Dictionary containing parsed configuration data
        """
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            data = {}
            
            for sheet_name in excel_file.sheet_names:
                # Skip empty or metadata sheets
                if sheet_name.startswith('_') or sheet_name.lower() in ['metadata', 'readme', 'instructions']:
                    continue
                
                # Read sheet
                df = pd.read_excel(excel_file, sheet_name=sheet_name, header=0)
                
                # Skip empty sheets
                if df.empty:
                    continue
                
                # Parse sheet based on structure
                sheet_data = self._parse_sheet(df, sheet_name)
                
                if sheet_data:
                    data[self._normalize_sheet_name(sheet_name)] = sheet_data
            
            if not data:
                raise ParsingError("No valid data found in Excel file", file_path=file_path)
            
            return data
            
        except Exception as e:
            raise ParsingError(f"Failed to parse Excel file: {str(e)}", file_path=file_path)
    
    def validate_structure(self, data: Dict[str, Any]) -> bool:
        """
        Validate the structure of parsed Excel data
        
        Args:
            data: Parsed configuration data
            
        Returns:
            True if structure is valid
        """
        if not isinstance(data, dict):
            self.add_error("Parsed data must be a dictionary")
            return False
        
        if not data:
            self.add_error("No data found in configuration")
            return False
        
        # Check for at least one valid configuration section
        valid_sections = 0
        for section_name, section_data in data.items():
            if isinstance(section_data, (dict, list)) and section_data:
                valid_sections += 1
        
        if valid_sections == 0:
            self.add_error("No valid configuration sections found")
            return False
        
        return True
    
    def _parse_sheet(self, df: pd.DataFrame, sheet_name: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Parse a single Excel sheet
        
        Args:
            df: DataFrame containing sheet data
            sheet_name: Name of the sheet
            
        Returns:
            Parsed sheet data
        """
        # Clean column names
        df.columns = [self._clean_column_name(col) for col in df.columns]
        
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        if df.empty:
            return None
        
        # Detect sheet structure
        if self._is_key_value_sheet(df):
            return self._parse_key_value_sheet(df)
        elif self._is_table_sheet(df):
            return self._parse_table_sheet(df)
        elif self._is_matrix_sheet(df):
            return self._parse_matrix_sheet(df)
        else:
            # Default to table parsing
            return self._parse_table_sheet(df)
    
    def _is_key_value_sheet(self, df: pd.DataFrame) -> bool:
        """Check if sheet is in key-value format"""
        if len(df.columns) == 2:
            # Check if first column contains unique values (keys)
            first_col_values = df.iloc[:, 0].dropna()
            if len(first_col_values) == len(first_col_values.unique()):
                return True
        return False
    
    def _is_table_sheet(self, df: pd.DataFrame) -> bool:
        """Check if sheet is in table format"""
        # Table format has multiple columns with headers
        return len(df.columns) > 2 and len(df) > 1
    
    def _is_matrix_sheet(self, df: pd.DataFrame) -> bool:
        """Check if sheet is in matrix format"""
        # Matrix format has row and column headers
        if len(df.columns) > 3 and len(df) > 3:
            # Check if first column could be row headers
            first_col = df.iloc[:, 0]
            if first_col.notna().all() and len(first_col.unique()) == len(first_col):
                return True
        return False
    
    def _parse_key_value_sheet(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse key-value format sheet"""
        result = {}
        
        for _, row in df.iterrows():
            key = str(row.iloc[0]).strip()
            value = self._convert_value(row.iloc[1])
            
            if key and not pd.isna(row.iloc[0]):
                result[self._normalize_key(key)] = value
        
        return result
    
    def _parse_table_sheet(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Parse table format sheet"""
        result = []
        
        for _, row in df.iterrows():
            row_data = {}
            
            for col in df.columns:
                value = self._convert_value(row[col])
                if value is not None:
                    row_data[col] = value
            
            if row_data:  # Only add non-empty rows
                result.append(row_data)
        
        return result
    
    def _parse_matrix_sheet(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Parse matrix format sheet"""
        result = {}
        
        # First column as row headers
        row_headers = df.iloc[:, 0]
        
        for i, row_header in enumerate(row_headers):
            if pd.notna(row_header):
                row_key = self._normalize_key(str(row_header))
                result[row_key] = {}
                
                # Parse each column
                for j, col in enumerate(df.columns[1:], 1):
                    value = self._convert_value(df.iloc[i, j])
                    if value is not None:
                        result[row_key][col] = value
        
        return result
    
    def _convert_value(self, value: Any) -> Any:
        """Convert Excel value to appropriate Python type"""
        # Handle NaN/None
        if pd.isna(value):
            return None
        
        # Handle numpy types
        if isinstance(value, (np.integer, np.int64)):
            return int(value)
        elif isinstance(value, (np.floating, np.float64)):
            # Check if it's actually an integer
            if value == int(value):
                return int(value)
            return float(value)
        elif isinstance(value, np.bool_):
            return bool(value)
        
        # Handle strings
        if isinstance(value, str):
            value = value.strip()
            
            # Try to parse as JSON
            if value.startswith('[') or value.startswith('{'):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    pass
            
            # Try to parse as number
            try:
                if '.' in value:
                    return float(value)
                else:
                    return int(value)
            except ValueError:
                pass
            
            # Check for boolean strings
            if value.lower() in ['true', 'yes', 'on']:
                return True
            elif value.lower() in ['false', 'no', 'off']:
                return False
            
            # Return as string
            return value
        
        # Handle dates
        if isinstance(value, pd.Timestamp):
            return value.strftime('%Y-%m-%d %H:%M:%S')
        
        # Default
        return value
    
    def _clean_column_name(self, col: str) -> str:
        """Clean column name"""
        if not isinstance(col, str):
            col = str(col)
        
        # Remove extra whitespace
        col = ' '.join(col.split())
        
        # Replace special characters
        col = col.replace('(', '').replace(')', '')
        col = col.replace('[', '').replace(']', '')
        col = col.replace(',', '')
        col = col.replace('.', '')
        
        # Convert to snake_case
        col = col.lower().replace(' ', '_')
        
        return col
    
    def _normalize_sheet_name(self, sheet_name: str) -> str:
        """Normalize sheet name"""
        return sheet_name.lower().replace(' ', '_').replace('-', '_')
    
    def _normalize_key(self, key: str) -> str:
        """Normalize configuration key"""
        return key.lower().replace(' ', '_').replace('-', '_').replace('.', '_')
    
    def parse_with_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Parse Excel file and include metadata
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            Parsed data with metadata
        """
        data = self.parse_with_validation(file_path)
        
        # Add file metadata
        path = Path(file_path)
        data['_metadata'] = {
            'file_name': path.name,
            'file_size': path.stat().st_size,
            'modified_time': path.stat().st_mtime,
            'sheet_count': len(data) - (1 if '_metadata' in data else 0)
        }
        
        return data
    
    def extract_strategy_type(self, file_path: str) -> Optional[str]:
        """
        Try to extract strategy type from file name or content
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            Strategy type or None
        """
        path = Path(file_path)
        filename = path.stem.lower()
        
        # Check filename for strategy type
        strategy_types = ['tbs', 'tv', 'orb', 'oi', 'ml', 'pos', 'market_regime']
        
        for strategy_type in strategy_types:
            if strategy_type in filename:
                return strategy_type
        
        # Try to detect from content
        try:
            excel_file = pd.ExcelFile(file_path)
            
            # Check sheet names
            for sheet_name in excel_file.sheet_names:
                sheet_lower = sheet_name.lower()
                for strategy_type in strategy_types:
                    if strategy_type in sheet_lower:
                        return strategy_type
            
            # Check first sheet for strategy indicators
            if excel_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=excel_file.sheet_names[0], nrows=10)
                content = ' '.join(str(df.values.flatten()))
                
                for strategy_type in strategy_types:
                    if strategy_type in content.lower():
                        return strategy_type
        except:
            pass
        
        return None
    
    def __repr__(self) -> str:
        return f"ExcelParser(extensions={self.supported_extensions})"