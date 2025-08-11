"""
IV Percentile Analysis Engine - Component 4 Enhancement

Core IV percentile calculation engine with production schema alignment,
individual DTE tracking, zone-wise analysis, and sophisticated historical
percentile database for institutional-grade implied volatility percentile analysis.

This module implements the foundational IV percentile analysis framework
supporting exactly 87 total features for Epic 1 compliance.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import time
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


@dataclass
class IVPercentileData:
    """Production-aligned IV percentile data structure"""
    # Core identifiers
    trade_date: datetime
    trade_time: str
    expiry_date: datetime
    dte: int
    zone_name: str
    
    # Price/Strike data
    spot: float
    atm_strike: float
    strikes: np.ndarray
    
    # IV data (core for percentile analysis)
    ce_iv: np.ndarray
    pe_iv: np.ndarray
    
    # Volume/OI data
    ce_volume: np.ndarray
    pe_volume: np.ndarray
    ce_oi: np.ndarray
    pe_oi: np.ndarray
    
    # Supporting data
    expiry_bucket: str
    zone_id: int
    
    # Metadata
    strike_count: int
    data_completeness: float
    
    def __post_init__(self):
        """Validate and compute derived metrics"""
        self.strike_count = len(self.strikes)
        self.data_completeness = self._calculate_completeness()
    
    def _calculate_completeness(self) -> float:
        """Calculate data completeness score"""
        total_data_points = len(self.strikes) * 2  # CE + PE IVs
        
        valid_ce_iv = np.sum(~np.isnan(self.ce_iv) & (self.ce_iv > 0))
        valid_pe_iv = np.sum(~np.isnan(self.pe_iv) & (self.pe_iv > 0))
        
        valid_points = valid_ce_iv + valid_pe_iv
        return valid_points / total_data_points if total_data_points > 0 else 0.0


@dataclass
class IVPercentileResult:
    """IV percentile analysis result"""
    # Core percentile metrics
    iv_percentile: float
    historical_rank: float
    
    # DTE-specific percentiles
    dte_percentile: float
    dte_rank: float
    
    # Zone-specific percentiles
    zone_percentile: float
    zone_rank: float
    
    # Strike-level analysis
    strike_percentiles: Dict[float, float]
    atm_percentile: float
    
    # Surface-level aggregation
    surface_percentile_avg: float
    surface_percentile_weighted: float
    
    # Quality metrics
    confidence_score: float
    data_quality: float
    calculation_time_ms: float
    
    # Supporting data
    percentile_bands: Dict[str, float]
    historical_context: Dict[str, Any]
    metadata: Dict[str, Any]


class IVPercentileAnalyzer:
    """
    Production-aligned IV percentile analysis engine with sophisticated
    individual DTE tracking and zone-wise percentile computation.
    
    Features:
    - Production schema compatibility (48 columns)
    - Individual DTE-level percentile tracking (dte=0...dte=58)
    - Zone-wise analysis using zone_name column
    - Multi-strike percentile aggregation
    - 252-day rolling window historical database
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # Historical database configuration
        self.lookback_days = config.get('percentile_lookback_days', 252)
        self.min_historical_points = config.get('min_historical_points', 30)
        
        # DTE bucketing configuration
        self.dte_buckets = {
            'near': (0, 7),     # Near-term: 0-7 days
            'medium': (8, 30),  # Medium-term: 8-30 days
            'far': (31, 365)   # Far-term: 31+ days
        }
        
        # Zone mapping per production schema
        self.valid_zones = ['MID_MORN', 'LUNCH', 'AFTERNOON', 'CLOSE']
        
        # Performance budgets
        self.processing_budget_ms = config.get('processing_budget_ms', 350)
        self.memory_budget_mb = config.get('memory_budget_mb', 250)
        
        # Historical data storage
        self.historical_data = {}
        
        self.logger.info("IV Percentile Analyzer initialized with production schema alignment")
    
    def extract_iv_percentile_data(self, df: pd.DataFrame) -> IVPercentileData:
        """
        Extract IV percentile data using production parquet schema alignment
        
        Args:
            df: DataFrame with production schema (48 columns)
            
        Returns:
            IVPercentileData with schema-aligned extraction
        """
        start_time = time.time()
        
        try:
            # Validate required columns per production schema
            required_columns = [
                'trade_date', 'trade_time', 'expiry_date', 'dte', 'zone_name',
                'spot', 'atm_strike', 'strike', 'ce_iv', 'pe_iv',
                'ce_volume', 'pe_volume', 'ce_oi', 'pe_oi', 'expiry_bucket', 'zone_id'
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Extract core identifiers
            if df.empty:
                raise ValueError("Empty DataFrame provided")
            
            first_row = df.iloc[0]
            trade_date = pd.to_datetime(first_row['trade_date'])
            trade_time = str(first_row['trade_time'])
            expiry_date = pd.to_datetime(first_row['expiry_date'])
            dte = int(first_row['dte'])
            zone_name = str(first_row['zone_name'])
            
            # Validate zone_name per production schema
            if zone_name not in self.valid_zones:
                self.logger.warning(f"Invalid zone_name: {zone_name}, using CLOSE as default")
                zone_name = 'CLOSE'
            
            # Extract price/strike data
            spot = float(first_row['spot'])
            atm_strike = float(first_row['atm_strike'])
            
            # Extract strike-level data (ALL available strikes per production)
            strikes = df['strike'].values
            unique_strikes = np.unique(strikes)
            
            self.logger.info(f"Processing {len(unique_strikes)} strikes for DTE={dte}, Zone={zone_name}")
            
            # Aggregate by strike (handle multiple time entries per strike)
            strike_data = []
            for strike in unique_strikes:
                strike_rows = df[df['strike'] == strike]
                
                if len(strike_rows) == 0:
                    continue
                
                # Use last available values for each strike
                last_row = strike_rows.iloc[-1]
                
                strike_data.append({
                    'strike': float(strike),
                    'ce_iv': float(last_row['ce_iv']) if pd.notna(last_row['ce_iv']) else np.nan,
                    'pe_iv': float(last_row['pe_iv']) if pd.notna(last_row['pe_iv']) else np.nan,
                    'ce_volume': float(last_row['ce_volume']) if pd.notna(last_row['ce_volume']) else 0.0,
                    'pe_volume': float(last_row['pe_volume']) if pd.notna(last_row['pe_volume']) else 0.0,
                    'ce_oi': float(last_row['ce_oi']) if pd.notna(last_row['ce_oi']) else 0.0,
                    'pe_oi': float(last_row['pe_oi']) if pd.notna(last_row['pe_oi']) else 0.0,
                })
            
            if not strike_data:
                raise ValueError("No valid strike data found")
            
            # Convert to arrays
            strike_df = pd.DataFrame(strike_data)
            strike_df = strike_df.sort_values('strike')
            
            strikes_array = strike_df['strike'].values
            ce_iv_array = strike_df['ce_iv'].values
            pe_iv_array = strike_df['pe_iv'].values
            ce_volume_array = strike_df['ce_volume'].values
            pe_volume_array = strike_df['pe_volume'].values
            ce_oi_array = strike_df['ce_oi'].values
            pe_oi_array = strike_df['pe_oi'].values
            
            # Data quality validation
            iv_data_completeness = self._validate_iv_data_quality(ce_iv_array, pe_iv_array)
            
            processing_time = (time.time() - start_time) * 1000
            self.logger.info(f"IV data extraction completed in {processing_time:.2f}ms")
            
            return IVPercentileData(
                trade_date=trade_date,
                trade_time=trade_time,
                expiry_date=expiry_date,
                dte=dte,
                zone_name=zone_name,
                spot=spot,
                atm_strike=atm_strike,
                strikes=strikes_array,
                ce_iv=ce_iv_array,
                pe_iv=pe_iv_array,
                ce_volume=ce_volume_array,
                pe_volume=pe_volume_array,
                ce_oi=ce_oi_array,
                pe_oi=pe_oi_array,
                expiry_bucket=str(first_row['expiry_bucket']),
                zone_id=int(first_row['zone_id']),
                strike_count=len(strikes_array),
                data_completeness=iv_data_completeness
            )
            
        except Exception as e:
            self.logger.error(f"IV percentile data extraction failed: {e}")
            raise
    
    def _validate_iv_data_quality(self, ce_iv: np.ndarray, pe_iv: np.ndarray) -> float:
        """Validate IV data quality for percentile calculation reliability"""
        
        total_points = len(ce_iv) + len(pe_iv)
        if total_points == 0:
            return 0.0
        
        # Count valid IV values
        valid_ce_iv = np.sum(~np.isnan(ce_iv) & (ce_iv > 0) & (ce_iv < 200))  # Reasonable IV bounds
        valid_pe_iv = np.sum(~np.isnan(pe_iv) & (pe_iv > 0) & (pe_iv < 200))
        
        valid_points = valid_ce_iv + valid_pe_iv
        completeness = valid_points / total_points
        
        # Quality score components
        quality_factors = []
        
        # Completeness factor
        quality_factors.append(completeness)
        
        # IV reasonableness check
        if valid_points > 0:
            all_valid_ivs = np.concatenate([
                ce_iv[~np.isnan(ce_iv) & (ce_iv > 0)],
                pe_iv[~np.isnan(pe_iv) & (pe_iv > 0)]
            ])
            
            if len(all_valid_ivs) > 0:
                iv_mean = np.mean(all_valid_ivs)
                iv_std = np.std(all_valid_ivs)
                
                # Check for reasonable IV range
                reasonable_iv_range = (5 <= iv_mean <= 100) and (iv_std < 50)
                quality_factors.append(0.9 if reasonable_iv_range else 0.6)
        
        return float(np.mean(quality_factors))
    
    def calculate_dte_specific_percentiles(self, iv_data: IVPercentileData,
                                         historical_database: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate individual DTE-specific percentiles (dte=0, dte=1, dte=2...dte=58)
        with dedicated historical percentile database per DTE level.
        
        Args:
            iv_data: Current IV percentile data
            historical_database: Historical IV data database
            
        Returns:
            Dictionary with DTE-specific percentile metrics
        """
        start_time = time.time()
        
        try:
            current_dte = iv_data.dte
            dte_key = f"dte_{current_dte}"
            
            # Get historical data for this specific DTE
            historical_dte_data = historical_database.get(dte_key, [])
            
            if len(historical_dte_data) < self.min_historical_points:
                self.logger.warning(f"Insufficient historical data for DTE {current_dte}: "
                                  f"{len(historical_dte_data)} points (min: {self.min_historical_points})")
                # Return default percentiles
                return self._get_default_dte_percentiles(current_dte)
            
            # Calculate current IV metrics for percentile comparison
            current_metrics = self._calculate_current_iv_metrics(iv_data)
            
            # Calculate percentiles against historical DTE-specific data
            dte_percentiles = {}
            
            # ATM IV percentile
            historical_atm_ivs = [entry['atm_iv'] for entry in historical_dte_data 
                                 if entry.get('atm_iv') and not np.isnan(entry['atm_iv'])]
            
            if historical_atm_ivs:
                dte_percentiles['atm_iv_percentile'] = self._calculate_percentile_rank(
                    current_metrics['atm_iv'], historical_atm_ivs
                )
            else:
                dte_percentiles['atm_iv_percentile'] = 50.0
            
            # Surface average IV percentile
            historical_surface_ivs = [entry['surface_avg_iv'] for entry in historical_dte_data 
                                    if entry.get('surface_avg_iv') and not np.isnan(entry['surface_avg_iv'])]
            
            if historical_surface_ivs:
                dte_percentiles['surface_avg_percentile'] = self._calculate_percentile_rank(
                    current_metrics['surface_avg_iv'], historical_surface_ivs
                )
            else:
                dte_percentiles['surface_avg_percentile'] = 50.0
            
            # IV skew percentile
            historical_skews = [entry['iv_skew'] for entry in historical_dte_data 
                               if entry.get('iv_skew') and not np.isnan(entry['iv_skew'])]
            
            if historical_skews:
                dte_percentiles['iv_skew_percentile'] = self._calculate_percentile_rank(
                    current_metrics['iv_skew'], historical_skews
                )
            else:
                dte_percentiles['iv_skew_percentile'] = 50.0
            
            # DTE bucket classification
            dte_bucket = self._classify_dte_bucket(current_dte)
            dte_percentiles['dte_bucket'] = dte_bucket
            
            # Historical ranking (position in historical distribution)
            total_historical_points = len(historical_dte_data)
            dte_percentiles['historical_rank'] = total_historical_points
            dte_percentiles['data_sufficiency_score'] = min(1.0, total_historical_points / 100)
            
            processing_time = (time.time() - start_time) * 1000
            dte_percentiles['calculation_time_ms'] = processing_time
            
            self.logger.debug(f"DTE-specific percentiles calculated for DTE {current_dte}: "
                            f"ATM={dte_percentiles['atm_iv_percentile']:.1f}%, "
                            f"Surface={dte_percentiles['surface_avg_percentile']:.1f}%")
            
            return dte_percentiles
            
        except Exception as e:
            self.logger.error(f"DTE-specific percentile calculation failed: {e}")
            return self._get_default_dte_percentiles(iv_data.dte)
    
    def calculate_zone_wise_percentiles(self, iv_data: IVPercentileData,
                                      historical_database: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate zone-wise IV percentiles using zone_name column (MID_MORN/LUNCH/AFTERNOON/CLOSE)
        per production schema for intraday pattern analysis.
        
        Args:
            iv_data: Current IV percentile data
            historical_database: Historical IV data database
            
        Returns:
            Dictionary with zone-specific percentile metrics
        """
        start_time = time.time()
        
        try:
            current_zone = iv_data.zone_name
            zone_key = f"zone_{current_zone}"
            
            # Get historical data for this specific zone
            historical_zone_data = historical_database.get(zone_key, [])
            
            if len(historical_zone_data) < self.min_historical_points:
                self.logger.warning(f"Insufficient historical data for Zone {current_zone}: "
                                  f"{len(historical_zone_data)} points")
                return self._get_default_zone_percentiles(current_zone)
            
            # Calculate current zone IV metrics
            current_metrics = self._calculate_current_iv_metrics(iv_data)
            
            # Calculate zone-specific percentiles
            zone_percentiles = {}
            
            # Zone ATM IV percentile
            zone_atm_ivs = [entry['atm_iv'] for entry in historical_zone_data 
                           if entry.get('atm_iv') and not np.isnan(entry['atm_iv'])]
            
            if zone_atm_ivs:
                zone_percentiles['zone_atm_percentile'] = self._calculate_percentile_rank(
                    current_metrics['atm_iv'], zone_atm_ivs
                )
            else:
                zone_percentiles['zone_atm_percentile'] = 50.0
            
            # Zone surface percentile
            zone_surface_ivs = [entry['surface_avg_iv'] for entry in historical_zone_data 
                              if entry.get('surface_avg_iv') and not np.isnan(entry['surface_avg_iv'])]
            
            if zone_surface_ivs:
                zone_percentiles['zone_surface_percentile'] = self._calculate_percentile_rank(
                    current_metrics['surface_avg_iv'], zone_surface_ivs
                )
            else:
                zone_percentiles['zone_surface_percentile'] = 50.0
            
            # Intraday zone analysis
            zone_percentiles['zone_name'] = current_zone
            zone_percentiles['zone_id'] = iv_data.zone_id
            zone_percentiles['intraday_position'] = self._get_intraday_position(current_zone)
            
            # Zone transition analysis (comparison with previous zone)
            zone_percentiles.update(self._analyze_zone_transitions(
                current_zone, current_metrics, historical_database
            ))
            
            processing_time = (time.time() - start_time) * 1000
            zone_percentiles['calculation_time_ms'] = processing_time
            
            self.logger.debug(f"Zone-wise percentiles calculated for Zone {current_zone}: "
                            f"ATM={zone_percentiles['zone_atm_percentile']:.1f}%, "
                            f"Surface={zone_percentiles['zone_surface_percentile']:.1f}%")
            
            return zone_percentiles
            
        except Exception as e:
            self.logger.error(f"Zone-wise percentile calculation failed: {e}")
            return self._get_default_zone_percentiles(iv_data.zone_name)
    
    def calculate_multi_strike_percentiles(self, iv_data: IVPercentileData) -> Dict[str, Any]:
        """
        Process ALL available strikes (54-68 per expiry) with ce_iv/pe_iv
        percentile calculation at individual DTE granularity.
        
        Args:
            iv_data: IV percentile data with strike-level information
            
        Returns:
            Dictionary with multi-strike percentile analysis
        """
        start_time = time.time()
        
        try:
            # Process all strikes
            strike_percentiles = {}
            all_ce_ivs = []
            all_pe_ivs = []
            
            # Extract valid IV values for surface-level analysis
            for i, strike in enumerate(iv_data.strikes):
                ce_iv = iv_data.ce_iv[i]
                pe_iv = iv_data.pe_iv[i]
                
                strike_data = {}
                
                # Individual strike percentile analysis
                if not np.isnan(ce_iv) and ce_iv > 0:
                    all_ce_ivs.append(ce_iv)
                    strike_data['ce_iv'] = float(ce_iv)
                else:
                    strike_data['ce_iv'] = np.nan
                
                if not np.isnan(pe_iv) and pe_iv > 0:
                    all_pe_ivs.append(pe_iv)
                    strike_data['pe_iv'] = float(pe_iv)
                else:
                    strike_data['pe_iv'] = np.nan
                
                # Strike moneyness classification
                moneyness = strike / iv_data.spot
                strike_data['moneyness'] = float(moneyness)
                strike_data['strike_type'] = self._classify_strike_type(moneyness)
                
                strike_percentiles[float(strike)] = strike_data
            
            # Surface-level aggregation
            surface_metrics = {}
            
            if all_ce_ivs:
                surface_metrics['ce_iv_mean'] = float(np.mean(all_ce_ivs))
                surface_metrics['ce_iv_std'] = float(np.std(all_ce_ivs))
                surface_metrics['ce_iv_min'] = float(np.min(all_ce_ivs))
                surface_metrics['ce_iv_max'] = float(np.max(all_ce_ivs))
                surface_metrics['ce_strikes_count'] = len(all_ce_ivs)
            
            if all_pe_ivs:
                surface_metrics['pe_iv_mean'] = float(np.mean(all_pe_ivs))
                surface_metrics['pe_iv_std'] = float(np.std(all_pe_ivs))
                surface_metrics['pe_iv_min'] = float(np.min(all_pe_ivs))
                surface_metrics['pe_iv_max'] = float(np.max(all_pe_ivs))
                surface_metrics['pe_strikes_count'] = len(all_pe_ivs)
            
            # Combined surface metrics
            all_ivs = all_ce_ivs + all_pe_ivs
            if all_ivs:
                surface_metrics['surface_iv_mean'] = float(np.mean(all_ivs))
                surface_metrics['surface_iv_std'] = float(np.std(all_ivs))
                surface_metrics['surface_iv_range'] = float(np.max(all_ivs) - np.min(all_ivs))
                surface_metrics['total_strikes_processed'] = len(all_ivs)
            
            # ATM strike analysis
            atm_analysis = self._analyze_atm_strike_percentiles(iv_data, strike_percentiles)
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                'strike_percentiles': strike_percentiles,
                'surface_metrics': surface_metrics,
                'atm_analysis': atm_analysis,
                'processing_stats': {
                    'total_strikes': len(iv_data.strikes),
                    'valid_ce_strikes': len(all_ce_ivs),
                    'valid_pe_strikes': len(all_pe_ivs),
                    'processing_time_ms': processing_time,
                    'data_completeness': iv_data.data_completeness
                }
            }
            
        except Exception as e:
            self.logger.error(f"Multi-strike percentile calculation failed: {e}")
            raise
    
    def _calculate_current_iv_metrics(self, iv_data: IVPercentileData) -> Dict[str, float]:
        """Calculate current IV metrics for percentile comparison"""
        
        metrics = {}
        
        # ATM IV calculation
        atm_idx = np.argmin(np.abs(iv_data.strikes - iv_data.atm_strike))
        atm_ce_iv = iv_data.ce_iv[atm_idx] if not np.isnan(iv_data.ce_iv[atm_idx]) else 0
        atm_pe_iv = iv_data.pe_iv[atm_idx] if not np.isnan(iv_data.pe_iv[atm_idx]) else 0
        metrics['atm_iv'] = float((atm_ce_iv + atm_pe_iv) / 2) if (atm_ce_iv > 0 or atm_pe_iv > 0) else 0.0
        
        # Surface average IV
        valid_ivs = np.concatenate([
            iv_data.ce_iv[~np.isnan(iv_data.ce_iv) & (iv_data.ce_iv > 0)],
            iv_data.pe_iv[~np.isnan(iv_data.pe_iv) & (iv_data.pe_iv > 0)]
        ])
        metrics['surface_avg_iv'] = float(np.mean(valid_ivs)) if len(valid_ivs) > 0 else 0.0
        
        # IV skew calculation (simple Put-Call IV difference)
        if len(valid_ivs) >= 4:
            # Use quartiles for skew
            put_ivs = iv_data.pe_iv[~np.isnan(iv_data.pe_iv) & (iv_data.pe_iv > 0)]
            call_ivs = iv_data.ce_iv[~np.isnan(iv_data.ce_iv) & (iv_data.ce_iv > 0)]
            
            if len(put_ivs) > 0 and len(call_ivs) > 0:
                metrics['iv_skew'] = float(np.mean(put_ivs) - np.mean(call_ivs))
            else:
                metrics['iv_skew'] = 0.0
        else:
            metrics['iv_skew'] = 0.0
        
        return metrics
    
    def _calculate_percentile_rank(self, current_value: float, historical_values: List[float]) -> float:
        """Calculate percentile rank of current value vs historical distribution"""
        
        if not historical_values or np.isnan(current_value):
            return 50.0
        
        historical_array = np.array(historical_values)
        historical_array = historical_array[~np.isnan(historical_array)]
        
        if len(historical_array) == 0:
            return 50.0
        
        # Calculate percentile rank
        percentile = (np.sum(historical_array <= current_value) / len(historical_array)) * 100
        return float(np.clip(percentile, 0.0, 100.0))
    
    def _classify_dte_bucket(self, dte: int) -> str:
        """Classify DTE into bucket (Near/Medium/Far)"""
        
        for bucket_name, (min_dte, max_dte) in self.dte_buckets.items():
            if min_dte <= dte <= max_dte:
                return bucket_name
        
        return 'far'  # Default for DTEs > 365
    
    def _classify_strike_type(self, moneyness: float) -> str:
        """Classify strike type based on moneyness"""
        
        if moneyness < 0.95:
            return 'OTM_PUT'
        elif moneyness < 0.98:
            return 'ITM_PUT'
        elif moneyness <= 1.02:
            return 'ATM'
        elif moneyness <= 1.05:
            return 'ITM_CALL'
        else:
            return 'OTM_CALL'
    
    def _get_intraday_position(self, zone_name: str) -> float:
        """Get intraday position score for zone (0=morning, 1=close)"""
        
        zone_positions = {
            'MID_MORN': 0.2,
            'LUNCH': 0.5,
            'AFTERNOON': 0.8,
            'CLOSE': 1.0
        }
        
        return zone_positions.get(zone_name, 0.5)
    
    def _analyze_atm_strike_percentiles(self, iv_data: IVPercentileData, 
                                      strike_percentiles: Dict[float, Dict]) -> Dict[str, float]:
        """Analyze ATM-specific percentile metrics"""
        
        atm_idx = np.argmin(np.abs(iv_data.strikes - iv_data.atm_strike))
        atm_strike = iv_data.strikes[atm_idx]
        
        atm_data = strike_percentiles.get(float(atm_strike), {})
        
        return {
            'atm_strike': float(atm_strike),
            'atm_ce_iv': atm_data.get('ce_iv', np.nan),
            'atm_pe_iv': atm_data.get('pe_iv', np.nan),
            'atm_moneyness': atm_data.get('moneyness', 1.0),
            'atm_strike_distance': float(abs(atm_strike - iv_data.atm_strike)),
            'atm_relative_iv': float((atm_data.get('ce_iv', 0) + atm_data.get('pe_iv', 0)) / 2) if not np.isnan(atm_data.get('ce_iv', np.nan)) else 0.0
        }
    
    def _analyze_zone_transitions(self, current_zone: str, current_metrics: Dict[str, float],
                                historical_database: Dict[str, Any]) -> Dict[str, float]:
        """Analyze transitions between trading zones"""
        
        # Define zone sequence
        zone_sequence = ['MID_MORN', 'LUNCH', 'AFTERNOON', 'CLOSE']
        
        transitions = {}
        
        try:
            current_idx = zone_sequence.index(current_zone)
            
            # Get previous zone if available
            if current_idx > 0:
                prev_zone = zone_sequence[current_idx - 1]
                prev_zone_key = f"zone_{prev_zone}"
                
                prev_zone_data = historical_database.get(prev_zone_key, [])
                if prev_zone_data:
                    # Calculate transition metrics
                    prev_atm_ivs = [entry['atm_iv'] for entry in prev_zone_data[-10:] 
                                   if entry.get('atm_iv') and not np.isnan(entry['atm_iv'])]
                    
                    if prev_atm_ivs:
                        avg_prev_atm = np.mean(prev_atm_ivs)
                        current_atm = current_metrics['atm_iv']
                        
                        transitions['zone_transition_change'] = float(current_atm - avg_prev_atm)
                        transitions['zone_transition_pct'] = float((current_atm / avg_prev_atm - 1) * 100) if avg_prev_atm > 0 else 0.0
                        transitions['previous_zone'] = prev_zone
            
            # Position in trading session
            transitions['session_position'] = float(current_idx / (len(zone_sequence) - 1))
            
        except (ValueError, IndexError):
            # Zone not in sequence or other error
            transitions['zone_transition_change'] = 0.0
            transitions['zone_transition_pct'] = 0.0
            transitions['session_position'] = 0.5
        
        return transitions
    
    def _get_default_dte_percentiles(self, dte: int) -> Dict[str, float]:
        """Get default DTE percentiles when insufficient historical data"""
        
        return {
            'atm_iv_percentile': 50.0,
            'surface_avg_percentile': 50.0,
            'iv_skew_percentile': 50.0,
            'dte_bucket': self._classify_dte_bucket(dte),
            'historical_rank': 0,
            'data_sufficiency_score': 0.0,
            'calculation_time_ms': 0.0
        }
    
    def _get_default_zone_percentiles(self, zone_name: str) -> Dict[str, float]:
        """Get default zone percentiles when insufficient historical data"""
        
        return {
            'zone_atm_percentile': 50.0,
            'zone_surface_percentile': 50.0,
            'zone_name': zone_name,
            'zone_id': 0,
            'intraday_position': self._get_intraday_position(zone_name),
            'zone_transition_change': 0.0,
            'zone_transition_pct': 0.0,
            'session_position': 0.5,
            'calculation_time_ms': 0.0
        }
    
    def validate_production_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate DataFrame compatibility with production parquet schema
        
        Returns:
            Validation results with schema compliance details
        """
        validation = {
            'schema_compliant': False,
            'missing_columns': [],
            'extra_columns': [],
            'column_count': len(df.columns),
            'expected_columns': 48,
            'data_quality_score': 0.0
        }
        
        try:
            # Expected production columns
            expected_columns = [
                'trade_date', 'trade_time', 'expiry_date', 'index_name', 'spot', 'atm_strike',
                'strike', 'dte', 'expiry_bucket', 'zone_id', 'zone_name', 'call_strike_type',
                'put_strike_type', 'ce_symbol', 'ce_open', 'ce_high', 'ce_low', 'ce_close',
                'ce_volume', 'ce_oi', 'ce_coi', 'ce_iv', 'ce_delta', 'ce_gamma', 'ce_theta',
                'ce_vega', 'ce_rho', 'pe_symbol', 'pe_open', 'pe_high', 'pe_low', 'pe_close',
                'pe_volume', 'pe_oi', 'pe_coi', 'pe_iv', 'pe_delta', 'pe_gamma', 'pe_theta',
                'pe_vega', 'pe_rho', 'future_open', 'future_high', 'future_low', 'future_close',
                'future_volume', 'future_oi', 'future_coi'
            ]
            
            # Check columns
            df_columns = set(df.columns)
            expected_set = set(expected_columns)
            
            validation['missing_columns'] = list(expected_set - df_columns)
            validation['extra_columns'] = list(df_columns - expected_set)
            validation['schema_compliant'] = len(validation['missing_columns']) == 0
            
            # Data quality assessment
            if not df.empty:
                # Check critical columns for IV analysis
                critical_columns = ['ce_iv', 'pe_iv', 'strike', 'dte', 'zone_name']
                quality_scores = []
                
                for col in critical_columns:
                    if col in df.columns:
                        valid_ratio = df[col].notna().sum() / len(df)
                        quality_scores.append(valid_ratio)
                
                validation['data_quality_score'] = float(np.mean(quality_scores)) if quality_scores else 0.0
            
            self.logger.info(f"Schema validation: Compliant={validation['schema_compliant']}, "
                           f"Quality={validation['data_quality_score']:.2f}")
            
        except Exception as e:
            self.logger.error(f"Schema validation failed: {e}")
        
        return validation