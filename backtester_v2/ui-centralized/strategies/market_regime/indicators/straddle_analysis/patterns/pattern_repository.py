"""
Pattern Repository for Multi-Timeframe Pattern Recognition System

Manages storage, retrieval, and analysis of trading patterns across all 10 components
with comprehensive multi-timeframe analysis and validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class PatternSchema:
    """Comprehensive pattern schema for multi-timeframe analysis"""
    pattern_id: str
    pattern_type: str  # single_component, individual_straddle, combined_triple, cross_component
    
    # Timeframe analysis (3, 5, 10, 15 minutes)
    timeframe_analysis: Dict[str, Dict[str, Any]]
    
    # Component analysis for all 10 components
    components: Dict[str, Dict[str, Any]]
    
    # Cross-timeframe confluence
    cross_timeframe_confluence: Dict[str, float]
    
    # Market context
    market_context: Dict[str, Any]
    
    # Historical performance
    historical_performance: Dict[str, float]
    
    # Validation results (7-layer validation)
    validation_results: Dict[str, float]
    
    # Risk metrics
    risk_metrics: Dict[str, float]
    
    # Pattern metadata
    discovery_timestamp: datetime
    last_occurrence: datetime
    total_occurrences: int
    success_rate: float
    confidence_score: float
    
    # Statistical validation
    statistical_significance: Dict[str, float]
    
    # ML scoring
    ml_scores: Dict[str, float]


class PatternRepository:
    """
    Comprehensive Pattern Repository for 10-Component Analysis
    
    Manages patterns across:
    - Individual Components (6): ATM_CE, ATM_PE, ITM1_CE, ITM1_PE, OTM1_CE, OTM1_PE
    - Individual Straddles (3): ATM_STRADDLE, ITM1_STRADDLE, OTM1_STRADDLE  
    - Combined Triple Straddle (1): COMBINED_TRIPLE_STRADDLE
    - Cross-Component Patterns: Multiple component interactions
    
    Features:
    - Multi-timeframe pattern storage (3, 5, 10, 15 minutes)
    - 7-layer validation system for >90% success rate
    - Statistical significance testing
    - ML-based pattern scoring
    - Real-time pattern adaptation
    """
    
    def __init__(self, db_path: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize Pattern Repository
        
        Args:
            db_path: Path to SQLite database file
            config: Repository configuration
        """
        self.config = config or self._get_default_config()
        self.db_path = db_path or self.config.get('db_path', 'pattern_repository.db')
        
        # Component definitions (10 total)
        self.individual_components = [
            'ATM_CE', 'ATM_PE', 'ITM1_CE', 'ITM1_PE', 'OTM1_CE', 'OTM1_PE'
        ]
        self.individual_straddles = [
            'ATM_STRADDLE', 'ITM1_STRADDLE', 'OTM1_STRADDLE'
        ]
        self.combined_triple = ['COMBINED_TRIPLE_STRADDLE']
        
        self.all_components = (
            self.individual_components + 
            self.individual_straddles + 
            self.combined_triple
        )
        
        # Timeframes
        self.timeframes = [3, 5, 10, 15]  # minutes
        
        # DTE categories with pattern characteristics
        self.dte_categories = {
            "ultra_short": {
                "range": [0, 1],
                "characteristics": ["high_gamma", "fast_decay", "volatile_moves"],
                "pattern_weight": 0.9
            },
            "short": {
                "range": [2, 5], 
                "characteristics": ["moderate_gamma", "time_decay_active", "trend_following"],
                "pattern_weight": 0.8
            },
            "medium": {
                "range": [6, 15],
                "characteristics": ["balanced_greeks", "trend_patterns", "support_resistance"],
                "pattern_weight": 0.7
            },
            "long": {
                "range": [16, 30],
                "characteristics": ["low_gamma", "directional_bias", "fundamental_driven"],
                "pattern_weight": 0.6
            }
        }
        
        # Market zones
        self.market_zones = {
            "support_zone_1": {
                "price_range": [24700, 24750],
                "pattern_types": ["bounce_patterns", "reversal_patterns"],
                "success_rate": 0.68,
                "common_indicators": ["VWAP", "PIVOT_S1", "EMA_200"]
            },
            "resistance_zone_1": {
                "price_range": [24950, 25000],
                "pattern_types": ["rejection_patterns", "breakdown_patterns"], 
                "success_rate": 0.71,
                "common_indicators": ["PIVOT_R1", "EMA_100", "Previous_High"]
            },
            "neutral_zone": {
                "price_range": [24800, 24900],
                "pattern_types": ["consolidation_patterns", "breakout_patterns"],
                "success_rate": 0.45,
                "common_indicators": ["EMA_20", "VWAP", "Midpoint"]
            }
        }
        
        # Validation thresholds for >90% success rate
        self.validation_thresholds = {
            "timeframe_alignment": 0.91,
            "technical_confluence": 0.89,
            "volume_confirmation": 0.93,
            "statistical_significance": 0.87,
            "historical_consistency": 0.90,
            "risk_reward_ratio": 0.92,
            "market_context_fit": 0.88,
            "overall_success_threshold": 0.90
        }
        
        # Initialize database
        self._initialize_database()
        
        # Pattern caches for performance
        self._pattern_cache = {}
        self._cache_size_limit = 1000
        
        # Performance tracking
        self.pattern_discovery_count = 0
        self.validation_count = 0
        self.successful_patterns = 0
        
        self.logger = logging.getLogger(f"{__name__}.PatternRepository")
        self.logger.info(f"Pattern Repository initialized with {len(self.all_components)} components")
        self.logger.info(f"Tracking patterns across {len(self.timeframes)} timeframes")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default repository configuration"""
        return {
            'db_path': 'pattern_repository.db',
            'cache_size': 1000,
            'validation_enabled': True,
            'statistical_testing': True,
            'ml_scoring': True,
            'min_occurrences': 100,
            'min_success_rate': 0.90,
            'confidence_threshold': 0.85
        }
    
    def _initialize_database(self):
        """Initialize SQLite database for pattern storage"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create patterns table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS patterns (
                        pattern_id TEXT PRIMARY KEY,
                        pattern_type TEXT NOT NULL,
                        timeframe_analysis TEXT,
                        components TEXT,
                        cross_timeframe_confluence TEXT,
                        market_context TEXT,
                        historical_performance TEXT,
                        validation_results TEXT,
                        risk_metrics TEXT,
                        discovery_timestamp TIMESTAMP,
                        last_occurrence TIMESTAMP,
                        total_occurrences INTEGER,
                        success_rate REAL,
                        confidence_score REAL,
                        statistical_significance TEXT,
                        ml_scores TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create pattern occurrences table for detailed tracking
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS pattern_occurrences (
                        occurrence_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pattern_id TEXT,
                        timestamp TIMESTAMP,
                        market_data TEXT,
                        outcome REAL,
                        success BOOLEAN,
                        duration_minutes INTEGER,
                        max_profit REAL,
                        max_drawdown REAL,
                        FOREIGN KEY (pattern_id) REFERENCES patterns (pattern_id)
                    )
                ''')
                
                # Create performance tracking table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS pattern_performance (
                        performance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pattern_id TEXT,
                        date DATE,
                        success_rate REAL,
                        avg_return REAL,
                        max_drawdown REAL,
                        sharpe_ratio REAL,
                        total_trades INTEGER,
                        FOREIGN KEY (pattern_id) REFERENCES patterns (pattern_id)
                    )
                ''')
                
                # Create indices for performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_pattern_type ON patterns (pattern_type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_success_rate ON patterns (success_rate)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_confidence ON patterns (confidence_score)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON pattern_occurrences (timestamp)')
                
                conn.commit()
                self.logger.info("Pattern repository database initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            raise
    
    def store_pattern(self, pattern: PatternSchema) -> bool:
        """
        Store a validated pattern in the repository
        
        Args:
            pattern: PatternSchema object to store
            
        Returns:
            True if successfully stored, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Convert complex objects to JSON strings
                pattern_data = (
                    pattern.pattern_id,
                    pattern.pattern_type,
                    json.dumps(pattern.timeframe_analysis),
                    json.dumps(pattern.components),
                    json.dumps(pattern.cross_timeframe_confluence),
                    json.dumps(pattern.market_context),
                    json.dumps(pattern.historical_performance),
                    json.dumps(pattern.validation_results),
                    json.dumps(pattern.risk_metrics),
                    pattern.discovery_timestamp,
                    pattern.last_occurrence,
                    pattern.total_occurrences,
                    pattern.success_rate,
                    pattern.confidence_score,
                    json.dumps(pattern.statistical_significance),
                    json.dumps(pattern.ml_scores)
                )
                
                cursor.execute('''
                    INSERT OR REPLACE INTO patterns (
                        pattern_id, pattern_type, timeframe_analysis, components,
                        cross_timeframe_confluence, market_context, historical_performance,
                        validation_results, risk_metrics, discovery_timestamp,
                        last_occurrence, total_occurrences, success_rate,
                        confidence_score, statistical_significance, ml_scores
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', pattern_data)
                
                conn.commit()
                
                # Update cache
                self._pattern_cache[pattern.pattern_id] = pattern
                self._manage_cache_size()
                
                self.pattern_discovery_count += 1
                self.logger.debug(f"Pattern {pattern.pattern_id} stored successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Error storing pattern {pattern.pattern_id}: {e}")
            return False
    
    def get_pattern(self, pattern_id: str) -> Optional[PatternSchema]:
        """
        Retrieve a pattern by ID
        
        Args:
            pattern_id: Pattern identifier
            
        Returns:
            PatternSchema object or None if not found
        """
        # Check cache first
        if pattern_id in self._pattern_cache:
            return self._pattern_cache[pattern_id]
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM patterns WHERE pattern_id = ?', (pattern_id,))
                row = cursor.fetchone()
                
                if row:
                    pattern = self._row_to_pattern(row)
                    # Cache the pattern
                    self._pattern_cache[pattern_id] = pattern
                    return pattern
                    
        except Exception as e:
            self.logger.error(f"Error retrieving pattern {pattern_id}: {e}")
        
        return None
    
    def search_patterns(self, 
                       pattern_type: Optional[str] = None,
                       components: Optional[List[str]] = None,
                       timeframes: Optional[List[int]] = None,
                       min_success_rate: Optional[float] = None,
                       min_confidence: Optional[float] = None,
                       market_regime: Optional[str] = None,
                       dte_range: Optional[Tuple[int, int]] = None,
                       limit: int = 100) -> List[PatternSchema]:
        """
        Search patterns based on criteria
        
        Args:
            pattern_type: Type of pattern to search for
            components: List of components to include
            timeframes: List of timeframes to include
            min_success_rate: Minimum success rate threshold
            min_confidence: Minimum confidence score
            market_regime: Market regime filter
            dte_range: DTE range filter
            limit: Maximum number of results
            
        Returns:
            List of matching PatternSchema objects
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build dynamic query
                query = "SELECT * FROM patterns WHERE 1=1"
                params = []
                
                if pattern_type:
                    query += " AND pattern_type = ?"
                    params.append(pattern_type)
                
                if min_success_rate:
                    query += " AND success_rate >= ?"
                    params.append(min_success_rate)
                
                if min_confidence:
                    query += " AND confidence_score >= ?"
                    params.append(min_confidence)
                
                # Order by success rate and confidence
                query += " ORDER BY success_rate DESC, confidence_score DESC"
                
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                patterns = []
                for row in rows:
                    pattern = self._row_to_pattern(row)
                    
                    # Apply additional filters that require JSON parsing
                    if self._passes_advanced_filters(pattern, components, timeframes, 
                                                   market_regime, dte_range):
                        patterns.append(pattern)
                
                return patterns
                
        except Exception as e:
            self.logger.error(f"Error searching patterns: {e}")
            return []
    
    def _passes_advanced_filters(self, pattern: PatternSchema,
                               components: Optional[List[str]] = None,
                               timeframes: Optional[List[int]] = None,
                               market_regime: Optional[str] = None,
                               dte_range: Optional[Tuple[int, int]] = None) -> bool:
        """Check if pattern passes advanced filters"""
        try:
            # Component filter
            if components:
                pattern_components = list(pattern.components.keys())
                if not any(comp in pattern_components for comp in components):
                    return False
            
            # Timeframe filter
            if timeframes:
                pattern_timeframes = [int(tf.replace('min', '')) for tf in pattern.timeframe_analysis.keys()]
                if not any(tf in pattern_timeframes for tf in timeframes):
                    return False
            
            # Market regime filter
            if market_regime:
                if pattern.market_context.get('market_regime') != market_regime:
                    return False
            
            # DTE range filter
            if dte_range:
                pattern_dte = pattern.market_context.get('dte_range', [0, 0])
                if not (dte_range[0] <= pattern_dte[1] and dte_range[1] >= pattern_dte[0]):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Error in advanced filtering: {e}")
            return True  # Include pattern if filtering fails
    
    def _row_to_pattern(self, row) -> PatternSchema:
        """Convert database row to PatternSchema object"""
        return PatternSchema(
            pattern_id=row[0],
            pattern_type=row[1],
            timeframe_analysis=json.loads(row[2]),
            components=json.loads(row[3]),
            cross_timeframe_confluence=json.loads(row[4]),
            market_context=json.loads(row[5]),
            historical_performance=json.loads(row[6]),
            validation_results=json.loads(row[7]),
            risk_metrics=json.loads(row[8]),
            discovery_timestamp=datetime.fromisoformat(row[9]),
            last_occurrence=datetime.fromisoformat(row[10]),
            total_occurrences=row[11],
            success_rate=row[12],
            confidence_score=row[13],
            statistical_significance=json.loads(row[14]),
            ml_scores=json.loads(row[15])
        )
    
    def _manage_cache_size(self):
        """Manage cache size to prevent memory issues"""
        if len(self._pattern_cache) > self._cache_size_limit:
            # Remove oldest 20% of patterns
            remove_count = len(self._pattern_cache) // 5
            oldest_patterns = sorted(
                self._pattern_cache.items(),
                key=lambda x: x[1].last_occurrence
            )[:remove_count]
            
            for pattern_id, _ in oldest_patterns:
                del self._pattern_cache[pattern_id]
    
    def update_pattern_performance(self, pattern_id: str, 
                                 success: bool, outcome: float,
                                 duration_minutes: int,
                                 max_profit: float = 0.0,
                                 max_drawdown: float = 0.0) -> bool:
        """
        Update pattern performance with new occurrence data
        
        Args:
            pattern_id: Pattern identifier
            success: Whether the pattern was successful
            outcome: Actual return/outcome
            duration_minutes: Duration of the pattern
            max_profit: Maximum profit during the pattern
            max_drawdown: Maximum drawdown during the pattern
            
        Returns:
            True if successfully updated
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Record occurrence
                cursor.execute('''
                    INSERT INTO pattern_occurrences (
                        pattern_id, timestamp, outcome, success, 
                        duration_minutes, max_profit, max_drawdown
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (pattern_id, datetime.now(), outcome, success, 
                      duration_minutes, max_profit, max_drawdown))
                
                # Update pattern statistics
                cursor.execute('''
                    SELECT total_occurrences, success_rate FROM patterns 
                    WHERE pattern_id = ?
                ''', (pattern_id,))
                
                row = cursor.fetchone()
                if row:
                    total_occurrences, current_success_rate = row
                    
                    # Calculate new success rate
                    new_total = total_occurrences + 1
                    new_success_rate = (
                        (current_success_rate * total_occurrences + (1 if success else 0)) / new_total
                    )
                    
                    # Update pattern
                    cursor.execute('''
                        UPDATE patterns SET 
                            total_occurrences = ?,
                            success_rate = ?,
                            last_occurrence = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE pattern_id = ?
                    ''', (new_total, new_success_rate, datetime.now(), pattern_id))
                
                conn.commit()
                
                # Update cache if pattern is cached
                if pattern_id in self._pattern_cache:
                    pattern = self._pattern_cache[pattern_id]
                    pattern.total_occurrences += 1
                    pattern.success_rate = new_success_rate
                    pattern.last_occurrence = datetime.now()
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error updating pattern performance: {e}")
            return False
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get repository statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total patterns by type
                cursor.execute('''
                    SELECT pattern_type, COUNT(*), AVG(success_rate), AVG(confidence_score)
                    FROM patterns 
                    GROUP BY pattern_type
                ''')
                
                pattern_stats = {}
                for row in cursor.fetchall():
                    pattern_stats[row[0]] = {
                        'count': row[1],
                        'avg_success_rate': row[2],
                        'avg_confidence': row[3]
                    }
                
                # High performance patterns (>90% success rate)
                cursor.execute('SELECT COUNT(*) FROM patterns WHERE success_rate >= 0.90')
                high_performance_count = cursor.fetchone()[0]
                
                # Recent patterns (last 30 days)
                thirty_days_ago = datetime.now() - timedelta(days=30)
                cursor.execute('SELECT COUNT(*) FROM patterns WHERE discovery_timestamp >= ?', 
                             (thirty_days_ago,))
                recent_patterns = cursor.fetchone()[0]
                
                return {
                    'total_patterns': sum(stats['count'] for stats in pattern_stats.values()),
                    'pattern_types': pattern_stats,
                    'high_performance_patterns': high_performance_count,
                    'recent_patterns': recent_patterns,
                    'avg_success_rate': np.mean([stats['avg_success_rate'] for stats in pattern_stats.values()]),
                    'cache_size': len(self._pattern_cache),
                    'discovery_count': self.pattern_discovery_count
                }
                
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {}
    
    def cleanup_old_patterns(self, days_old: int = 365) -> int:
        """
        Clean up patterns older than specified days with low performance
        
        Args:
            days_old: Number of days threshold
            
        Returns:
            Number of patterns cleaned up
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete patterns that are old and have low success rate
                cursor.execute('''
                    DELETE FROM patterns 
                    WHERE discovery_timestamp < ? 
                    AND success_rate < 0.60 
                    AND total_occurrences < 50
                ''', (cutoff_date,))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                # Clear cache for deleted patterns
                self._pattern_cache.clear()
                
                self.logger.info(f"Cleaned up {deleted_count} old patterns")
                return deleted_count
                
        except Exception as e:
            self.logger.error(f"Error cleaning up patterns: {e}")
            return 0