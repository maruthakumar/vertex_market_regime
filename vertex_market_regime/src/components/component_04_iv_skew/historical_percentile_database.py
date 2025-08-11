"""
Historical IV Percentile Database Engine - Component 4 Enhancement

Advanced historical database system with 252-day rolling window for percentile
baseline establishment, DTE-specific storage, zone-wise historical tracking,
and efficient percentile interpolation for Component 4 IV analysis.

This module provides institutional-grade historical percentile database
with comprehensive DTE and zone granularity for superior percentile accuracy.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import pickle
import json
from pathlib import Path
import sqlite3
import threading
from collections import deque, defaultdict
import time

from .iv_percentile_analyzer import IVPercentileData


@dataclass
class HistoricalIVEntry:
    """Single historical IV entry for percentile database"""
    
    # Temporal identifiers
    trade_date: datetime
    trade_time: str
    
    # Context identifiers
    dte: int
    zone_name: str
    
    # IV metrics for percentile tracking
    atm_iv: float
    surface_avg_iv: float
    iv_skew: float
    
    # Strike-level IV distribution
    strike_iv_distribution: Dict[float, Tuple[float, float]]  # strike -> (ce_iv, pe_iv)
    
    # Quality metrics
    data_completeness: float
    strikes_count: int
    
    # Metadata
    expiry_date: datetime
    spot: float
    atm_strike: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'trade_date': self.trade_date.isoformat(),
            'trade_time': self.trade_time,
            'dte': self.dte,
            'zone_name': self.zone_name,
            'atm_iv': self.atm_iv,
            'surface_avg_iv': self.surface_avg_iv,
            'iv_skew': self.iv_skew,
            'strike_iv_distribution': self.strike_iv_distribution,
            'data_completeness': self.data_completeness,
            'strikes_count': self.strikes_count,
            'expiry_date': self.expiry_date.isoformat(),
            'spot': self.spot,
            'atm_strike': self.atm_strike
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HistoricalIVEntry':
        """Create from dictionary"""
        return cls(
            trade_date=pd.to_datetime(data['trade_date']),
            trade_time=data['trade_time'],
            dte=data['dte'],
            zone_name=data['zone_name'],
            atm_iv=data['atm_iv'],
            surface_avg_iv=data['surface_avg_iv'],
            iv_skew=data['iv_skew'],
            strike_iv_distribution=data['strike_iv_distribution'],
            data_completeness=data['data_completeness'],
            strikes_count=data['strikes_count'],
            expiry_date=pd.to_datetime(data['expiry_date']),
            spot=data['spot'],
            atm_strike=data['atm_strike']
        )


@dataclass
class PercentileDistribution:
    """Percentile distribution for efficient lookups"""
    
    # Distribution percentiles
    p1: float = 0.0
    p5: float = 0.0
    p10: float = 0.0
    p25: float = 0.0
    p50: float = 0.0
    p75: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    
    # Statistics
    mean: float = 0.0
    std: float = 0.0
    count: int = 0
    
    # Update timestamp
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def calculate_percentile_rank(self, value: float) -> float:
        """Calculate percentile rank for given value using distribution"""
        
        if self.count == 0:
            return 50.0
        
        # Use interpolation between known percentiles
        percentile_values = [
            (1, self.p1), (5, self.p5), (10, self.p10), (25, self.p25), (50, self.p50),
            (75, self.p75), (90, self.p90), (95, self.p95), (99, self.p99)
        ]
        
        # Handle edge cases
        if value <= self.p1:
            return 1.0
        if value >= self.p99:
            return 99.0
        
        # Find interpolation bounds
        for i in range(len(percentile_values) - 1):
            current_pct, current_val = percentile_values[i]
            next_pct, next_val = percentile_values[i + 1]
            
            if current_val <= value <= next_val:
                if next_val == current_val:
                    return float(current_pct)
                
                # Linear interpolation
                weight = (value - current_val) / (next_val - current_val)
                return float(current_pct + weight * (next_pct - current_pct))
        
        return 50.0  # Fallback
    
    def is_outdated(self, max_age_hours: int = 24) -> bool:
        """Check if distribution is outdated"""
        age = datetime.utcnow() - self.last_updated
        return age.total_seconds() > (max_age_hours * 3600)


class HistoricalPercentileDatabase:
    """
    Advanced historical IV percentile database with DTE-specific and zone-wise storage,
    252-day rolling window management, and efficient percentile interpolation.
    
    Features:
    - Individual DTE-level storage (dte=0, dte=1...dte=58)
    - Zone-wise historical tracking (MID_MORN/LUNCH/AFTERNOON/CLOSE)
    - Strike-level percentile storage and aggregation
    - 252-day rolling window with automatic cleanup
    - Efficient percentile distribution caching
    - Thread-safe concurrent operations
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # Database configuration
        self.lookback_days = config.get('percentile_lookback_days', 252)
        self.min_entries_for_percentile = config.get('min_entries_for_percentile', 30)
        self.max_dte_tracking = config.get('max_dte_tracking', 58)
        
        # Storage configuration
        self.storage_path = Path(config.get('storage_path', 'data/historical_percentiles'))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.use_sqlite = config.get('use_sqlite_storage', True)
        self.cache_size = config.get('cache_size', 1000)
        
        # In-memory storage structures
        self.dte_storage: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.lookback_days * 5))
        self.zone_storage: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.lookback_days * 20))
        self.combined_storage: deque = deque(maxlen=self.lookback_days * 100)
        
        # Percentile distribution cache
        self.percentile_cache: Dict[str, PercentileDistribution] = {}
        self.cache_lock = threading.RLock()
        
        # SQLite database for persistent storage
        if self.use_sqlite:
            self.db_path = self.storage_path / 'iv_percentiles.db'
            self._init_sqlite_database()
        
        self.logger.info(f"Historical Percentile Database initialized with {self.lookback_days}-day window")
    
    def _init_sqlite_database(self):
        """Initialize SQLite database for persistent storage"""
        
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                # Create main historical data table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS historical_iv_entries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trade_date TEXT NOT NULL,
                        trade_time TEXT NOT NULL,
                        dte INTEGER NOT NULL,
                        zone_name TEXT NOT NULL,
                        atm_iv REAL NOT NULL,
                        surface_avg_iv REAL NOT NULL,
                        iv_skew REAL NOT NULL,
                        data_completeness REAL NOT NULL,
                        strikes_count INTEGER NOT NULL,
                        expiry_date TEXT NOT NULL,
                        spot REAL NOT NULL,
                        atm_strike REAL NOT NULL,
                        strike_iv_data TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create percentile distributions cache table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS percentile_distributions (
                        cache_key TEXT PRIMARY KEY,
                        p1 REAL, p5 REAL, p10 REAL, p25 REAL, p50 REAL,
                        p75 REAL, p90 REAL, p95 REAL, p99 REAL,
                        mean_val REAL, std_val REAL, count_val INTEGER,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indices for efficient queries
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_trade_date ON historical_iv_entries(trade_date)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_dte ON historical_iv_entries(dte)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_zone_name ON historical_iv_entries(zone_name)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_dte_zone ON historical_iv_entries(dte, zone_name)')
                
                conn.commit()
                
            self.logger.info("SQLite database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"SQLite database initialization failed: {e}")
            self.use_sqlite = False
    
    def add_historical_entry(self, iv_data: IVPercentileData) -> bool:
        """
        Add new historical IV entry to database with DTE and zone-specific storage
        
        Args:
            iv_data: IV percentile data to store historically
            
        Returns:
            Success status
        """
        start_time = time.time()
        
        try:
            # Convert to historical entry
            historical_entry = self._convert_to_historical_entry(iv_data)
            
            # Add to in-memory storage
            self._add_to_memory_storage(historical_entry)
            
            # Add to persistent storage if enabled
            if self.use_sqlite:
                self._add_to_sqlite_storage(historical_entry)
            
            # Update percentile cache
            self._update_percentile_cache(historical_entry)
            
            # Cleanup old entries
            self._cleanup_old_entries()
            
            processing_time = (time.time() - start_time) * 1000
            self.logger.debug(f"Historical entry added in {processing_time:.2f}ms for "
                            f"DTE={historical_entry.dte}, Zone={historical_entry.zone_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add historical entry: {e}")
            return False
    
    def get_dte_percentile_distribution(self, dte: int) -> Optional[PercentileDistribution]:
        """
        Get percentile distribution for specific DTE level with caching
        
        Args:
            dte: Days to expiry
            
        Returns:
            PercentileDistribution for the DTE or None if insufficient data
        """
        cache_key = f"dte_{dte}"
        
        with self.cache_lock:
            # Check cache first
            if cache_key in self.percentile_cache:
                cached_dist = self.percentile_cache[cache_key]
                if not cached_dist.is_outdated():
                    return cached_dist
            
            # Get historical data for this DTE
            dte_entries = list(self.dte_storage[dte])
            
            if len(dte_entries) < self.min_entries_for_percentile:
                self.logger.debug(f"Insufficient data for DTE {dte}: {len(dte_entries)} entries")
                return None
            
            # Calculate percentile distribution
            atm_ivs = [entry.atm_iv for entry in dte_entries if not np.isnan(entry.atm_iv)]
            surface_ivs = [entry.surface_avg_iv for entry in dte_entries if not np.isnan(entry.surface_avg_iv)]
            
            if not atm_ivs:
                return None
            
            # Use ATM IVs as primary distribution (most reliable)
            distribution = self._calculate_percentile_distribution(atm_ivs)
            
            # Cache the result
            self.percentile_cache[cache_key] = distribution
            
            return distribution
    
    def get_zone_percentile_distribution(self, zone_name: str) -> Optional[PercentileDistribution]:
        """
        Get percentile distribution for specific zone with caching
        
        Args:
            zone_name: Zone name (MID_MORN/LUNCH/AFTERNOON/CLOSE)
            
        Returns:
            PercentileDistribution for the zone or None if insufficient data
        """
        cache_key = f"zone_{zone_name}"
        
        with self.cache_lock:
            # Check cache first
            if cache_key in self.percentile_cache:
                cached_dist = self.percentile_cache[cache_key]
                if not cached_dist.is_outdated():
                    return cached_dist
            
            # Get historical data for this zone
            zone_entries = list(self.zone_storage[zone_name])
            
            if len(zone_entries) < self.min_entries_for_percentile:
                self.logger.debug(f"Insufficient data for Zone {zone_name}: {len(zone_entries)} entries")
                return None
            
            # Calculate percentile distribution
            atm_ivs = [entry.atm_iv for entry in zone_entries if not np.isnan(entry.atm_iv)]
            
            if not atm_ivs:
                return None
            
            distribution = self._calculate_percentile_distribution(atm_ivs)
            
            # Cache the result
            self.percentile_cache[cache_key] = distribution
            
            return distribution
    
    def get_historical_database_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of historical database for diagnostics
        
        Returns:
            Database summary with coverage and quality metrics
        """
        summary = {
            'database_status': 'operational',
            'total_entries': len(self.combined_storage),
            'lookback_days': self.lookback_days,
            'dte_coverage': {},
            'zone_coverage': {},
            'cache_status': {},
            'data_quality': {}
        }
        
        try:
            # DTE coverage analysis
            for dte in range(self.max_dte_tracking + 1):
                dte_count = len(self.dte_storage[dte])
                summary['dte_coverage'][f'dte_{dte}'] = {
                    'entry_count': dte_count,
                    'sufficient_data': dte_count >= self.min_entries_for_percentile,
                    'coverage_ratio': min(1.0, dte_count / self.min_entries_for_percentile)
                }
            
            # Zone coverage analysis
            valid_zones = ['MID_MORN', 'LUNCH', 'AFTERNOON', 'CLOSE']
            for zone in valid_zones:
                zone_count = len(self.zone_storage[zone])
                summary['zone_coverage'][zone] = {
                    'entry_count': zone_count,
                    'sufficient_data': zone_count >= self.min_entries_for_percentile,
                    'coverage_ratio': min(1.0, zone_count / self.min_entries_for_percentile)
                }
            
            # Cache status
            summary['cache_status'] = {
                'cached_distributions': len(self.percentile_cache),
                'cache_hit_potential': len([k for k, v in self.percentile_cache.items() if not v.is_outdated()]),
                'memory_usage_estimate_mb': len(self.combined_storage) * 0.5 / 1024  # Rough estimate
            }
            
            # Data quality assessment
            if self.combined_storage:
                recent_entries = list(self.combined_storage)[-100:]  # Last 100 entries
                quality_scores = [entry.data_completeness for entry in recent_entries]
                
                summary['data_quality'] = {
                    'avg_completeness': float(np.mean(quality_scores)) if quality_scores else 0.0,
                    'min_completeness': float(np.min(quality_scores)) if quality_scores else 0.0,
                    'entries_above_80pct': sum(1 for score in quality_scores if score >= 0.8),
                    'total_recent_entries': len(quality_scores)
                }
            
        except Exception as e:
            summary['error'] = str(e)
            self.logger.error(f"Database summary generation failed: {e}")
        
        return summary
    
    def build_percentile_baseline(self, parquet_data_path: str) -> Dict[str, Any]:
        """
        Build comprehensive percentile baseline from historical parquet data
        
        Args:
            parquet_data_path: Path to historical parquet data directory
            
        Returns:
            Baseline building results and statistics
        """
        start_time = time.time()
        baseline_stats = {
            'files_processed': 0,
            'entries_added': 0,
            'dte_distributions_built': 0,
            'zone_distributions_built': 0,
            'processing_time_ms': 0.0,
            'errors': []
        }
        
        try:
            data_path = Path(parquet_data_path)
            
            if not data_path.exists():
                raise ValueError(f"Data path does not exist: {parquet_data_path}")
            
            # Process all parquet files
            parquet_files = list(data_path.rglob("*.parquet"))
            self.logger.info(f"Building baseline from {len(parquet_files)} parquet files")
            
            from .iv_percentile_analyzer import IVPercentileAnalyzer
            analyzer = IVPercentileAnalyzer(self.config)
            
            for file_path in parquet_files:
                try:
                    # Load parquet file
                    df = pd.read_parquet(file_path)
                    
                    # Validate schema
                    validation = analyzer.validate_production_schema(df)
                    if not validation['schema_compliant']:
                        self.logger.warning(f"Schema non-compliant file skipped: {file_path}")
                        continue
                    
                    # Extract IV data and add to historical database
                    iv_data = analyzer.extract_iv_percentile_data(df)
                    
                    if self.add_historical_entry(iv_data):
                        baseline_stats['entries_added'] += 1
                    
                    baseline_stats['files_processed'] += 1
                    
                    if baseline_stats['files_processed'] % 10 == 0:
                        self.logger.info(f"Processed {baseline_stats['files_processed']} files...")
                
                except Exception as e:
                    error_msg = f"Failed to process {file_path}: {e}"
                    baseline_stats['errors'].append(error_msg)
                    self.logger.error(error_msg)
            
            # Build final distributions
            baseline_stats['dte_distributions_built'] = self._build_dte_distributions()
            baseline_stats['zone_distributions_built'] = self._build_zone_distributions()
            
            processing_time = (time.time() - start_time) * 1000
            baseline_stats['processing_time_ms'] = processing_time
            
            self.logger.info(f"Percentile baseline built: {baseline_stats['entries_added']} entries, "
                           f"{baseline_stats['dte_distributions_built']} DTE distributions, "
                           f"{baseline_stats['zone_distributions_built']} zone distributions")
            
        except Exception as e:
            error_msg = f"Baseline building failed: {e}"
            baseline_stats['errors'].append(error_msg)
            self.logger.error(error_msg)
        
        return baseline_stats
    
    def _convert_to_historical_entry(self, iv_data: IVPercentileData) -> HistoricalIVEntry:
        """Convert IV percentile data to historical entry format"""
        
        # Calculate IV metrics
        atm_idx = np.argmin(np.abs(iv_data.strikes - iv_data.atm_strike))
        atm_ce_iv = iv_data.ce_iv[atm_idx] if not np.isnan(iv_data.ce_iv[atm_idx]) else 0.0
        atm_pe_iv = iv_data.pe_iv[atm_idx] if not np.isnan(iv_data.pe_iv[atm_idx]) else 0.0
        atm_iv = (atm_ce_iv + atm_pe_iv) / 2 if (atm_ce_iv > 0 or atm_pe_iv > 0) else 0.0
        
        # Surface average IV
        valid_ivs = np.concatenate([
            iv_data.ce_iv[~np.isnan(iv_data.ce_iv) & (iv_data.ce_iv > 0)],
            iv_data.pe_iv[~np.isnan(iv_data.pe_iv) & (iv_data.pe_iv > 0)]
        ])
        surface_avg_iv = float(np.mean(valid_ivs)) if len(valid_ivs) > 0 else 0.0
        
        # IV skew
        put_ivs = iv_data.pe_iv[~np.isnan(iv_data.pe_iv) & (iv_data.pe_iv > 0)]
        call_ivs = iv_data.ce_iv[~np.isnan(iv_data.ce_iv) & (iv_data.ce_iv > 0)]
        
        if len(put_ivs) > 0 and len(call_ivs) > 0:
            iv_skew = float(np.mean(put_ivs) - np.mean(call_ivs))
        else:
            iv_skew = 0.0
        
        # Strike-level distribution
        strike_distribution = {}
        for i, strike in enumerate(iv_data.strikes):
            ce_iv = float(iv_data.ce_iv[i]) if not np.isnan(iv_data.ce_iv[i]) else 0.0
            pe_iv = float(iv_data.pe_iv[i]) if not np.isnan(iv_data.pe_iv[i]) else 0.0
            strike_distribution[float(strike)] = (ce_iv, pe_iv)
        
        return HistoricalIVEntry(
            trade_date=iv_data.trade_date,
            trade_time=iv_data.trade_time,
            dte=iv_data.dte,
            zone_name=iv_data.zone_name,
            atm_iv=atm_iv,
            surface_avg_iv=surface_avg_iv,
            iv_skew=iv_skew,
            strike_iv_distribution=strike_distribution,
            data_completeness=iv_data.data_completeness,
            strikes_count=iv_data.strike_count,
            expiry_date=iv_data.expiry_date,
            spot=iv_data.spot,
            atm_strike=iv_data.atm_strike
        )
    
    def _add_to_memory_storage(self, entry: HistoricalIVEntry):
        """Add entry to in-memory storage structures"""
        
        # Add to DTE-specific storage
        self.dte_storage[entry.dte].append(entry)
        
        # Add to zone-specific storage
        self.zone_storage[entry.zone_name].append(entry)
        
        # Add to combined storage
        self.combined_storage.append(entry)
    
    def _add_to_sqlite_storage(self, entry: HistoricalIVEntry):
        """Add entry to SQLite persistent storage"""
        
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO historical_iv_entries (
                        trade_date, trade_time, dte, zone_name, atm_iv, surface_avg_iv,
                        iv_skew, data_completeness, strikes_count, expiry_date, spot,
                        atm_strike, strike_iv_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    entry.trade_date.isoformat(),
                    entry.trade_time,
                    entry.dte,
                    entry.zone_name,
                    entry.atm_iv,
                    entry.surface_avg_iv,
                    entry.iv_skew,
                    entry.data_completeness,
                    entry.strikes_count,
                    entry.expiry_date.isoformat(),
                    entry.spot,
                    entry.atm_strike,
                    json.dumps(entry.strike_iv_distribution)
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"SQLite storage failed: {e}")
    
    def _update_percentile_cache(self, entry: HistoricalIVEntry):
        """Update percentile cache with new entry"""
        
        with self.cache_lock:
            # Invalidate relevant cache entries
            dte_key = f"dte_{entry.dte}"
            zone_key = f"zone_{entry.zone_name}"
            
            # Mark as outdated to force recalculation
            if dte_key in self.percentile_cache:
                self.percentile_cache[dte_key].last_updated = datetime(2020, 1, 1)
            
            if zone_key in self.percentile_cache:
                self.percentile_cache[zone_key].last_updated = datetime(2020, 1, 1)
    
    def _cleanup_old_entries(self):
        """Clean up entries older than lookback window"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=self.lookback_days)
        
        # Cleanup is handled automatically by deque maxlen
        # But we can clean SQLite storage if needed
        if self.use_sqlite and len(self.combined_storage) % 100 == 0:  # Periodic cleanup
            try:
                with sqlite3.connect(str(self.db_path)) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        DELETE FROM historical_iv_entries 
                        WHERE trade_date < ?
                    ''', (cutoff_date.isoformat(),))
                    conn.commit()
            except Exception as e:
                self.logger.error(f"SQLite cleanup failed: {e}")
    
    def _calculate_percentile_distribution(self, values: List[float]) -> PercentileDistribution:
        """Calculate percentile distribution from values"""
        
        if not values:
            return PercentileDistribution()
        
        values_array = np.array(values)
        values_array = values_array[~np.isnan(values_array)]
        
        if len(values_array) == 0:
            return PercentileDistribution()
        
        # Calculate percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        pct_values = np.percentile(values_array, percentiles)
        
        return PercentileDistribution(
            p1=float(pct_values[0]),
            p5=float(pct_values[1]),
            p10=float(pct_values[2]),
            p25=float(pct_values[3]),
            p50=float(pct_values[4]),
            p75=float(pct_values[5]),
            p90=float(pct_values[6]),
            p95=float(pct_values[7]),
            p99=float(pct_values[8]),
            mean=float(np.mean(values_array)),
            std=float(np.std(values_array)),
            count=len(values_array),
            last_updated=datetime.utcnow()
        )
    
    def _build_dte_distributions(self) -> int:
        """Build percentile distributions for all DTEs with sufficient data"""
        
        built_count = 0
        
        for dte in range(self.max_dte_tracking + 1):
            if len(self.dte_storage[dte]) >= self.min_entries_for_percentile:
                distribution = self.get_dte_percentile_distribution(dte)
                if distribution:
                    built_count += 1
        
        return built_count
    
    def _build_zone_distributions(self) -> int:
        """Build percentile distributions for all zones with sufficient data"""
        
        built_count = 0
        valid_zones = ['MID_MORN', 'LUNCH', 'AFTERNOON', 'CLOSE']
        
        for zone in valid_zones:
            if len(self.zone_storage[zone]) >= self.min_entries_for_percentile:
                distribution = self.get_zone_percentile_distribution(zone)
                if distribution:
                    built_count += 1
        
        return built_count