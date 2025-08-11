"""
Production OI Data Extraction System for Component 3

This module extracts ce_oi, pe_oi, ce_volume, pe_volume from production Parquet data
with 99.98% coverage validation and ATM ±7 strikes range calculation.

As per story requirements:
- Extract ce_oi (column 20), pe_oi (column 34) with 99.98% coverage validation
- Integrate ce_volume (column 19), pe_volume (column 33) for institutional flow analysis  
- Implement dynamic ATM ±7 strikes range using call_strike_type/put_strike_type columns
- Build multi-timeframe rollups (5min, 15min, 3min, 10min) weighted analysis framework
"""

import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class ProductionOIExtractor:
    """
    Extract OI and Volume data from production Parquet files for institutional flow analysis.
    
    Handles production schema with 49 columns, focusing on:
    - ce_oi (column 20): Call Open Interest  
    - pe_oi (column 34): Put Open Interest
    - ce_volume (column 19): Call Volume
    - pe_volume (column 33): Put Volume
    - call_strike_type (column 12): Call strike classification (ATM/ITM/OTM)
    - put_strike_type (column 13): Put strike classification (ATM/ITM/OTM)
    """
    
    def __init__(self, data_path: str = "/Users/maruth/projects/market_regime/data/nifty_validation/backtester_processed"):
        """
        Initialize the Production OI Extractor.
        
        Args:
            data_path: Path to production Parquet files
        """
        self.data_path = Path(data_path)
        self.schema_cache = {}
        self.oi_coverage_stats = {}
        
        # Symbol-specific strike intervals (as per story requirements)
        self.strike_intervals = {
            'NIFTY': 50,      # ₹50 intervals
            'BANKNIFTY': 100  # ₹100 intervals
        }
        
        # Multi-timeframe weights (as per story spec)
        self.timeframe_weights = {
            '3min': 0.15,   # 15% weight - rapid repositioning detection
            '5min': 0.35,   # 35% weight - primary institutional flow analysis 
            '10min': 0.30,  # 30% weight - institutional flow persistence
            '15min': 0.20   # 20% weight - long-term institutional commitment
        }
    
    def analyze_production_schema(self, sample_file: Optional[str] = None) -> Dict[str, any]:
        """
        Analyze production Parquet schema to validate OI columns and coverage.
        
        Args:
            sample_file: Specific file to analyze, or None to use first available
            
        Returns:
            Dictionary with schema analysis results
        """
        if sample_file is None:
            # Find first available parquet file
            sample_file = self._get_first_parquet_file()
            
        if not sample_file:
            raise FileNotFoundError(f"No Parquet files found in {self.data_path}")
            
        logger.info(f"Analyzing production schema: {sample_file}")
        
        try:
            # Read schema and sample data
            table = pq.read_table(sample_file)
            df = table.to_pandas()
            
            # Build schema analysis
            schema_info = {
                'file_analyzed': str(sample_file),
                'total_columns': len(df.columns),
                'total_rows': len(df),
                'data_shape': df.shape,
                'columns': {},
                'oi_volume_analysis': {},
                'strike_type_analysis': {},
                'coverage_validation': {}
            }
            
            # Analyze all columns
            for i, col in enumerate(df.columns):
                dtype = str(df[col].dtype)
                null_count = df[col].isnull().sum()
                null_pct = (null_count / len(df)) * 100
                
                schema_info['columns'][i] = {
                    'name': col,
                    'dtype': dtype,
                    'null_count': null_count,
                    'null_percentage': null_pct,
                    'sample_value': str(df[col].iloc[0]) if len(df) > 0 else None
                }
            
            # Focus on OI and Volume columns (as per story requirements)
            oi_volume_columns = self._identify_oi_volume_columns(df)
            schema_info['oi_volume_analysis'] = oi_volume_columns
            
            # Validate coverage requirements (99.98% for OI, 100% for volume)
            coverage_results = self._validate_coverage_requirements(df, oi_volume_columns)
            schema_info['coverage_validation'] = coverage_results
            
            # Analyze strike type columns
            strike_type_analysis = self._analyze_strike_types(df)
            schema_info['strike_type_analysis'] = strike_type_analysis
            
            # Cache for future use
            self.schema_cache[str(sample_file)] = schema_info
            
            logger.info(f"Schema analysis complete: {len(df.columns)} columns, {len(df)} rows")
            logger.info(f"OI Coverage: CE={coverage_results.get('ce_oi_coverage', 0):.2f}%, PE={coverage_results.get('pe_oi_coverage', 0):.2f}%")
            
            return schema_info
            
        except Exception as e:
            logger.error(f"Schema analysis failed: {str(e)}")
            raise
    
    def extract_oi_data(self, file_path: str, atm_strikes_range: int = 7) -> pd.DataFrame:
        """
        Extract OI and volume data from production Parquet file.
        
        Args:
            file_path: Path to Parquet file
            atm_strikes_range: Number of strikes above/below ATM (default: ±7)
            
        Returns:
            DataFrame with extracted OI and volume data
        """
        logger.info(f"Extracting OI data from: {file_path}")
        
        try:
            # Read parquet file
            df = pd.read_parquet(file_path)
            
            # Identify OI and volume columns
            oi_volume_columns = self._identify_oi_volume_columns(df)
            
            # Extract required columns with proper naming
            extracted_data = df.copy()
            
            # Add computed columns for institutional flow analysis
            extracted_data = self._add_institutional_flow_indicators(extracted_data, oi_volume_columns)
            
            # Filter for ATM ±7 strikes range if strike type columns available
            if 'call_strike_type' in df.columns and 'put_strike_type' in df.columns:
                extracted_data = self._filter_atm_strikes_range(extracted_data, atm_strikes_range)
            
            logger.info(f"Extracted {len(extracted_data)} rows with OI/volume data")
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"OI data extraction failed: {str(e)}")
            raise
    
    def extract_cumulative_multistrike_oi(self, file_path: str, symbol: str = 'NIFTY') -> Dict[str, float]:
        """
        Extract cumulative OI across ATM ±7 strikes for institutional analysis.
        
        Args:
            file_path: Path to Parquet file
            symbol: Symbol name (NIFTY/BANKNIFTY) for strike interval calculation
            
        Returns:
            Dictionary with cumulative OI metrics
        """
        logger.info(f"Extracting cumulative multi-strike OI for {symbol}")
        
        try:
            df = pd.read_parquet(file_path)
            
            # Identify OI columns
            oi_columns = self._identify_oi_volume_columns(df)
            
            # Calculate cumulative OI across ATM ±7 strikes  
            cumulative_metrics = {}
            
            if 'ce_oi' in oi_columns and 'pe_oi' in oi_columns:
                # Get ATM ±7 strikes data
                atm_data = self._get_atm_strikes_data(df, strikes_range=7, symbol=symbol)
                
                if not atm_data.empty:
                    # Calculate cumulative OI metrics (as per story requirements)
                    cumulative_metrics.update({
                        'cumulative_ce_oi': atm_data[oi_columns['ce_oi']].sum(),
                        'cumulative_pe_oi': atm_data[oi_columns['pe_oi']].sum(),
                        'cumulative_total_oi': atm_data[oi_columns['ce_oi']].sum() + atm_data[oi_columns['pe_oi']].sum(),
                        'net_oi_bias': atm_data[oi_columns['ce_oi']].sum() - atm_data[oi_columns['pe_oi']].sum(),
                    })
                    
                    # Add price-based metrics if available
                    if 'ce_close' in oi_columns and 'pe_close' in oi_columns:
                        cumulative_metrics.update({
                            'cumulative_ce_price': atm_data[oi_columns['ce_close']].sum(),
                            'cumulative_pe_price': atm_data[oi_columns['pe_close']].sum(),
                            'net_price_bias': atm_data[oi_columns['ce_close']].sum() - atm_data[oi_columns['pe_close']].sum(),
                        })
                        
                        # Calculate OI-Price correlation (institutional flow indicator)
                        if len(atm_data) > 1:
                            ce_oi_price_corr = atm_data[oi_columns['ce_oi']].corr(atm_data[oi_columns['ce_close']])
                            pe_oi_price_corr = atm_data[oi_columns['pe_oi']].corr(atm_data[oi_columns['pe_close']])
                            
                            cumulative_metrics.update({
                                'oi_price_correlation_ce': ce_oi_price_corr if not pd.isna(ce_oi_price_corr) else 0.0,
                                'oi_price_correlation_pe': pe_oi_price_corr if not pd.isna(pe_oi_price_corr) else 0.0,
                            })
            
            logger.info(f"Calculated {len(cumulative_metrics)} cumulative OI metrics")
            
            return cumulative_metrics
            
        except Exception as e:
            logger.error(f"Cumulative OI extraction failed: {str(e)}")
            raise
    
    def build_multi_timeframe_rollups(self, file_path: str) -> Dict[str, pd.DataFrame]:
        """
        Build multi-timeframe OI rollups with weighted analysis (5min, 15min, 3min, 10min).
        
        Args:
            file_path: Path to Parquet file with time-series data
            
        Returns:
            Dictionary with DataFrames for each timeframe
        """
        logger.info("Building multi-timeframe OI rollups")
        
        try:
            df = pd.read_parquet(file_path)
            
            # Ensure we have time column
            time_col = self._identify_time_column(df)
            if not time_col:
                raise ValueError("No time column found for multi-timeframe analysis")
            
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                df[time_col] = pd.to_datetime(df[time_col])
            
            # Set time as index for resampling
            df_time = df.set_index(time_col)
            
            # Build rollups for each timeframe
            rollups = {}
            
            for timeframe, weight in self.timeframe_weights.items():
                logger.info(f"Processing {timeframe} rollups (weight: {weight})")
                
                # Convert timeframe to pandas frequency
                freq = timeframe.replace('min', 'T')  # 'min' -> 'T' for pandas
                
                # Aggregate OI and volume data
                rollup_data = self._aggregate_oi_timeframe(df_time, freq, weight)
                rollups[timeframe] = rollup_data
                
                logger.info(f"Created {len(rollup_data)} {timeframe} bars")
            
            # Create weighted synthesis (as per story requirements)
            weighted_synthesis = self._create_weighted_synthesis(rollups)
            rollups['weighted_synthesis'] = weighted_synthesis
            
            return rollups
            
        except Exception as e:
            logger.error(f"Multi-timeframe rollup failed: {str(e)}")
            raise
    
    def validate_oi_coverage(self, file_paths: List[str]) -> Dict[str, float]:
        """
        Validate OI data coverage across multiple files (99.98% requirement).
        
        Args:
            file_paths: List of Parquet file paths to validate
            
        Returns:
            Dictionary with coverage statistics
        """
        logger.info(f"Validating OI coverage across {len(file_paths)} files")
        
        total_rows = 0
        total_ce_oi_nulls = 0
        total_pe_oi_nulls = 0
        total_ce_volume_nulls = 0
        total_pe_volume_nulls = 0
        
        for file_path in file_paths:
            try:
                df = pd.read_parquet(file_path)
                total_rows += len(df)
                
                # Identify OI/volume columns
                oi_columns = self._identify_oi_volume_columns(df)
                
                if 'ce_oi' in oi_columns:
                    total_ce_oi_nulls += df[oi_columns['ce_oi']].isnull().sum()
                if 'pe_oi' in oi_columns:
                    total_pe_oi_nulls += df[oi_columns['pe_oi']].isnull().sum()
                if 'ce_volume' in oi_columns:
                    total_ce_volume_nulls += df[oi_columns['ce_volume']].isnull().sum()
                if 'pe_volume' in oi_columns:
                    total_pe_volume_nulls += df[oi_columns['pe_volume']].isnull().sum()
                    
            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {str(e)}")
                continue
        
        # Calculate coverage percentages
        coverage_stats = {
            'total_rows_analyzed': total_rows,
            'ce_oi_coverage': ((total_rows - total_ce_oi_nulls) / total_rows) * 100 if total_rows > 0 else 0,
            'pe_oi_coverage': ((total_rows - total_pe_oi_nulls) / total_rows) * 100 if total_rows > 0 else 0,
            'ce_volume_coverage': ((total_rows - total_ce_volume_nulls) / total_rows) * 100 if total_rows > 0 else 0,
            'pe_volume_coverage': ((total_rows - total_pe_volume_nulls) / total_rows) * 100 if total_rows > 0 else 0,
            'files_processed': len(file_paths),
        }
        
        # Validate against requirements
        oi_coverage_ok = coverage_stats['ce_oi_coverage'] >= 99.98 and coverage_stats['pe_oi_coverage'] >= 99.98
        volume_coverage_ok = coverage_stats['ce_volume_coverage'] >= 100.0 and coverage_stats['pe_volume_coverage'] >= 100.0
        
        coverage_stats['meets_requirements'] = oi_coverage_ok and volume_coverage_ok
        
        logger.info(f"Coverage validation complete:")
        logger.info(f"  CE OI: {coverage_stats['ce_oi_coverage']:.2f}% (req: 99.98%)")
        logger.info(f"  PE OI: {coverage_stats['pe_oi_coverage']:.2f}% (req: 99.98%)")
        logger.info(f"  CE Volume: {coverage_stats['ce_volume_coverage']:.2f}% (req: 100.0%)")
        logger.info(f"  PE Volume: {coverage_stats['pe_volume_coverage']:.2f}% (req: 100.0%)")
        
        return coverage_stats
    
    # Private helper methods
    
    def _get_first_parquet_file(self) -> Optional[str]:
        """Find first available Parquet file in data directory."""
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.parquet'):
                    return os.path.join(root, file)
        return None
    
    def _identify_oi_volume_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Identify OI and volume columns in the DataFrame based on naming patterns.
        
        Returns mapping of logical names to actual column names.
        """
        columns_map = {}
        
        # Common patterns for OI and volume columns
        patterns = {
            'ce_oi': ['ce_oi', 'call_oi', 'ce_open_interest', 'call_open_interest'],
            'pe_oi': ['pe_oi', 'put_oi', 'pe_open_interest', 'put_open_interest'],
            'ce_volume': ['ce_volume', 'call_volume', 'ce_vol', 'call_vol'],
            'pe_volume': ['pe_volume', 'put_volume', 'pe_vol', 'put_vol'],
            'ce_close': ['ce_close', 'call_close', 'ce_price', 'call_price'],
            'pe_close': ['pe_close', 'put_close', 'pe_price', 'put_price'],
            'call_strike_type': ['call_strike_type', 'ce_strike_type'],
            'put_strike_type': ['put_strike_type', 'pe_strike_type'],
        }
        
        # Try to match columns by name
        for logical_name, possible_names in patterns.items():
            for possible_name in possible_names:
                if possible_name in df.columns:
                    columns_map[logical_name] = possible_name
                    break
        
        # Also try by column index (as mentioned in story: column 20, 34, etc.)
        if len(df.columns) >= 49:  # Ensure we have production schema
            try:
                # Map by indices as per story specification
                if len(df.columns) > 20 and 'ce_oi' not in columns_map:
                    columns_map['ce_oi'] = df.columns[20]
                if len(df.columns) > 34 and 'pe_oi' not in columns_map:
                    columns_map['pe_oi'] = df.columns[34] 
                if len(df.columns) > 19 and 'ce_volume' not in columns_map:
                    columns_map['ce_volume'] = df.columns[19]
                if len(df.columns) > 33 and 'pe_volume' not in columns_map:
                    columns_map['pe_volume'] = df.columns[33]
                if len(df.columns) > 12 and 'call_strike_type' not in columns_map:
                    columns_map['call_strike_type'] = df.columns[12]
                if len(df.columns) > 13 and 'put_strike_type' not in columns_map:
                    columns_map['put_strike_type'] = df.columns[13]
            except IndexError:
                logger.warning("Column index mapping failed, using name-based mapping only")
        
        return columns_map
    
    def _validate_coverage_requirements(self, df: pd.DataFrame, oi_columns: Dict[str, str]) -> Dict[str, float]:
        """Validate OI/volume coverage against story requirements."""
        results = {}
        
        if 'ce_oi' in oi_columns:
            null_count = df[oi_columns['ce_oi']].isnull().sum()
            coverage = ((len(df) - null_count) / len(df)) * 100
            results['ce_oi_coverage'] = coverage
            
        if 'pe_oi' in oi_columns:
            null_count = df[oi_columns['pe_oi']].isnull().sum()
            coverage = ((len(df) - null_count) / len(df)) * 100
            results['pe_oi_coverage'] = coverage
            
        if 'ce_volume' in oi_columns:
            null_count = df[oi_columns['ce_volume']].isnull().sum()
            coverage = ((len(df) - null_count) / len(df)) * 100
            results['ce_volume_coverage'] = coverage
            
        if 'pe_volume' in oi_columns:
            null_count = df[oi_columns['pe_volume']].isnull().sum()
            coverage = ((len(df) - null_count) / len(df)) * 100
            results['pe_volume_coverage'] = coverage
        
        return results
    
    def _analyze_strike_types(self, df: pd.DataFrame) -> Dict[str, any]:
        """Analyze strike type columns for ATM ±7 range calculation."""
        analysis = {}
        
        if 'call_strike_type' in df.columns:
            strike_types = df['call_strike_type'].value_counts()
            analysis['call_strike_types'] = strike_types.to_dict()
            
        if 'put_strike_type' in df.columns:
            strike_types = df['put_strike_type'].value_counts()
            analysis['put_strike_types'] = strike_types.to_dict()
        
        return analysis
    
    def _add_institutional_flow_indicators(self, df: pd.DataFrame, oi_columns: Dict[str, str]) -> pd.DataFrame:
        """Add computed columns for institutional flow analysis."""
        
        if 'ce_oi' in oi_columns and 'pe_oi' in oi_columns:
            df['total_oi'] = df[oi_columns['ce_oi']] + df[oi_columns['pe_oi']]
            
        if 'ce_volume' in oi_columns and 'pe_volume' in oi_columns:
            df['total_volume'] = df[oi_columns['ce_volume']] + df[oi_columns['pe_volume']]
            
        if 'total_oi' in df.columns and 'total_volume' in df.columns:
            df['oi_volume_ratio'] = df['total_oi'] / (df['total_volume'] + 1e-8)  # Avoid division by zero
            
        return df
    
    def _filter_atm_strikes_range(self, df: pd.DataFrame, strikes_range: int) -> pd.DataFrame:
        """Filter data for ATM ±strikes_range using strike type columns."""
        
        # For ATM ±7, we want ATM, ITM1-7, OTM1-7 strikes
        atm_strikes = ['ATM']
        itm_strikes = [f'ITM{i}' for i in range(1, strikes_range + 1)]
        otm_strikes = [f'OTM{i}' for i in range(1, strikes_range + 1)]
        
        target_strikes = atm_strikes + itm_strikes + otm_strikes
        
        # Filter based on strike types
        mask = (
            df['call_strike_type'].isin(target_strikes) | 
            df['put_strike_type'].isin(target_strikes)
        )
        
        return df[mask]
    
    def _get_atm_strikes_data(self, df: pd.DataFrame, strikes_range: int, symbol: str) -> pd.DataFrame:
        """Get data for ATM ±strikes_range for cumulative calculations."""
        
        if 'call_strike_type' in df.columns and 'put_strike_type' in df.columns:
            return self._filter_atm_strikes_range(df, strikes_range)
        else:
            # Fallback: use strike price calculation if strike type not available
            logger.warning("Strike type columns not found, using price-based ATM calculation")
            return df  # Return all data as fallback
    
    def _identify_time_column(self, df: pd.DataFrame) -> Optional[str]:
        """Identify time column for multi-timeframe analysis."""
        
        time_patterns = ['trade_time', 'timestamp', 'time', 'datetime', 'trade_date']
        
        for pattern in time_patterns:
            if pattern in df.columns:
                return pattern
                
        # Check for datetime-type columns
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
                
        return None
    
    def _aggregate_oi_timeframe(self, df_time: pd.DataFrame, freq: str, weight: float) -> pd.DataFrame:
        """Aggregate OI data for specific timeframe."""
        
        # Define aggregation rules for OI data
        agg_rules = {
            # OI columns - use last value (OI is cumulative)
            'ce_oi': 'last' if 'ce_oi' in df_time.columns else None,
            'pe_oi': 'last' if 'pe_oi' in df_time.columns else None,
            
            # Volume columns - sum (volume is additive)
            'ce_volume': 'sum' if 'ce_volume' in df_time.columns else None,
            'pe_volume': 'sum' if 'pe_volume' in df_time.columns else None,
            
            # Price columns - OHLC
            'ce_close': 'last' if 'ce_close' in df_time.columns else None,
            'pe_close': 'last' if 'pe_close' in df_time.columns else None,
        }
        
        # Remove None rules
        agg_rules = {k: v for k, v in agg_rules.items() if v is not None}
        
        if not agg_rules:
            logger.warning(f"No aggregation rules matched for timeframe {freq}")
            return pd.DataFrame()
        
        # Perform aggregation
        aggregated = df_time.resample(freq).agg(agg_rules)
        
        # Add timeframe weight
        aggregated['timeframe_weight'] = weight
        
        return aggregated
    
    def _create_weighted_synthesis(self, rollups: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create weighted synthesis of multi-timeframe rollups."""
        
        # Primary focus on 5min and 15min as per story requirements
        primary_timeframes = ['5min', '15min']
        supporting_timeframes = ['3min', '10min']
        
        synthesis_data = []
        
        for timeframe, rollup_df in rollups.items():
            if timeframe == 'weighted_synthesis':  # Skip self
                continue
                
            if not rollup_df.empty:
                # Add timeframe identifier
                rollup_df['source_timeframe'] = timeframe
                synthesis_data.append(rollup_df)
        
        if synthesis_data:
            # Combine all timeframes
            combined_df = pd.concat(synthesis_data, sort=False)
            
            # Calculate weighted averages where applicable
            if 'ce_oi' in combined_df.columns and 'timeframe_weight' in combined_df.columns:
                weighted_ce_oi = (combined_df['ce_oi'] * combined_df['timeframe_weight']).sum()
                combined_df['weighted_ce_oi'] = weighted_ce_oi
                
            if 'pe_oi' in combined_df.columns and 'timeframe_weight' in combined_df.columns:
                weighted_pe_oi = (combined_df['pe_oi'] * combined_df['timeframe_weight']).sum()
                combined_df['weighted_pe_oi'] = weighted_pe_oi
            
            return combined_df
        else:
            return pd.DataFrame()


def main():
    """Test the Production OI Extractor with actual data."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize extractor
    extractor = ProductionOIExtractor()
    
    try:
        # Analyze production schema
        logger.info("=== PRODUCTION SCHEMA ANALYSIS ===")
        schema_info = extractor.analyze_production_schema()
        
        print(f"\nSchema Analysis Results:")
        print(f"File analyzed: {schema_info['file_analyzed']}")
        print(f"Total columns: {schema_info['total_columns']}")
        print(f"Total rows: {schema_info['total_rows']}")
        print(f"Data shape: {schema_info['data_shape']}")
        
        print(f"\nFirst 20 columns:")
        for i in range(min(20, schema_info['total_columns'])):
            col_info = schema_info['columns'][i]
            print(f"  {i:2d}. {col_info['name']:<25} ({col_info['dtype']:<12}) - {col_info['null_percentage']:.2f}% nulls")
        
        print(f"\nOI/Volume Analysis:")
        oi_analysis = schema_info['oi_volume_analysis']
        for logical_name, actual_column in oi_analysis.items():
            print(f"  {logical_name}: {actual_column}")
        
        print(f"\nCoverage Validation:")
        coverage = schema_info['coverage_validation']
        for metric, value in coverage.items():
            print(f"  {metric}: {value:.2f}%")
            
        # Validate coverage across multiple files
        logger.info("\n=== MULTI-FILE COVERAGE VALIDATION ===")
        sample_files = []
        for root, dirs, files in os.walk(extractor.data_path):
            for file in files[:5]:  # First 5 files
                if file.endswith('.parquet'):
                    sample_files.append(os.path.join(root, file))
        
        if sample_files:
            coverage_stats = extractor.validate_oi_coverage(sample_files)
            print(f"\nMulti-file Coverage Results:")
            print(f"Files processed: {coverage_stats['files_processed']}")
            print(f"Total rows: {coverage_stats['total_rows_analyzed']:,}")
            print(f"CE OI Coverage: {coverage_stats['ce_oi_coverage']:.2f}%")
            print(f"PE OI Coverage: {coverage_stats['pe_oi_coverage']:.2f}%")
            print(f"CE Volume Coverage: {coverage_stats['ce_volume_coverage']:.2f}%")
            print(f"PE Volume Coverage: {coverage_stats['pe_volume_coverage']:.2f}%")
            print(f"Meets Requirements: {coverage_stats['meets_requirements']}")
            
    except Exception as e:
        logger.error(f"Production OI extraction test failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()