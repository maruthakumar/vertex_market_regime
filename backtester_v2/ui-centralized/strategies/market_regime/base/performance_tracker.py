"""
Performance Tracker for Market Regime Indicators
===============================================

Tracks indicator performance with SQLite storage and statistical analysis.

Author: Market Regime Refactoring Team
Date: 2025-07-06
Version: 2.0.0 - Refactored Architecture
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import sqlite3
import pandas as pd
import numpy as np
from scipy import stats
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for an indicator"""
    accuracy: float = 0.5
    precision: float = 0.5
    recall: float = 0.5
    f1_score: float = 0.5
    sharpe_ratio: float = 0.0
    hit_rate: float = 0.5
    avg_confidence: float = 0.5
    error_rate: float = 0.0
    computation_time: float = 0.0
    data_quality: float = 1.0
    statistical_significance: float = 0.5  # p-value
    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    sample_size: int = 0
    last_updated: datetime = field(default_factory=datetime.now)

class PerformanceTracker:
    """
    Performance tracking system with SQLite persistence
    
    Tracks indicator performance over time with statistical significance testing
    """
    
    def __init__(self, db_path: str = "performance_tracker.db"):
        """Initialize performance tracker"""
        self.db_path = Path(db_path)
        self.connection: Optional[sqlite3.Connection] = None
        
        # Performance cache
        self.performance_cache: Dict[str, PerformanceMetrics] = {}
        self.cache_expiry = timedelta(minutes=5)
        self.last_cache_update = datetime.now()
        
        # Configuration
        self.min_sample_size = 10
        self.confidence_level = 0.95
        
        # Initialize database
        self._init_database()
        
        logger.info(f"PerformanceTracker initialized with database: {db_path}")
    
    def _init_database(self):
        """Initialize SQLite database"""
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            
            # Create tables
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS indicator_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    indicator_name TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    prediction REAL,
                    actual_outcome REAL,
                    confidence REAL,
                    computation_time REAL,
                    data_quality REAL,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    indicator_name TEXT NOT NULL,
                    metric_date DATE NOT NULL,
                    accuracy REAL,
                    precision_score REAL,
                    recall_score REAL,
                    f1_score REAL,
                    sharpe_ratio REAL,
                    hit_rate REAL,
                    avg_confidence REAL,
                    error_rate REAL,
                    avg_computation_time REAL,
                    avg_data_quality REAL,
                    statistical_significance REAL,
                    confidence_interval_lower REAL,
                    confidence_interval_upper REAL,
                    sample_size INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(indicator_name, metric_date)
                )
            """)
            
            # Create indexes
            self.connection.execute("CREATE INDEX IF NOT EXISTS idx_indicator_name ON indicator_performance(indicator_name)")
            self.connection.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON indicator_performance(timestamp)")
            self.connection.execute("CREATE INDEX IF NOT EXISTS idx_metrics_name_date ON performance_metrics(indicator_name, metric_date)")
            
            self.connection.commit()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def record_prediction(self, 
                         indicator_name: str,
                         prediction: float,
                         confidence: float,
                         actual_outcome: Optional[float] = None,
                         computation_time: float = 0.0,
                         data_quality: float = 1.0,
                         metadata: Optional[Dict[str, Any]] = None):
        """
        Record an indicator prediction
        
        Args:
            indicator_name: Name of the indicator
            prediction: Predicted value
            confidence: Confidence in prediction
            actual_outcome: Actual outcome (if known)
            computation_time: Time taken for computation
            data_quality: Quality of input data
            metadata: Additional metadata
        """
        try:
            metadata_json = json.dumps(metadata) if metadata else None
            
            self.connection.execute("""
                INSERT INTO indicator_performance 
                (indicator_name, timestamp, prediction, actual_outcome, confidence, 
                 computation_time, data_quality, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                indicator_name, datetime.now(), prediction, actual_outcome,
                confidence, computation_time, data_quality, metadata_json
            ))
            
            self.connection.commit()
            
            # Invalidate cache for this indicator
            if indicator_name in self.performance_cache:
                del self.performance_cache[indicator_name]
            
        except Exception as e:
            logger.error(f"Error recording prediction for {indicator_name}: {e}")
    
    def update_actual_outcome(self, 
                             indicator_name: str,
                             timestamp: datetime,
                             actual_outcome: float):
        """
        Update actual outcome for a prediction
        
        Args:
            indicator_name: Name of the indicator
            timestamp: Timestamp of the prediction
            actual_outcome: Actual outcome value
        """
        try:
            # Find the closest prediction within 1 minute
            time_window = timedelta(minutes=1)
            start_time = timestamp - time_window
            end_time = timestamp + time_window
            
            self.connection.execute("""
                UPDATE indicator_performance 
                SET actual_outcome = ?
                WHERE indicator_name = ? 
                AND timestamp BETWEEN ? AND ?
                AND actual_outcome IS NULL
                ORDER BY ABS(julianday(?) - julianday(timestamp)) 
                LIMIT 1
            """, (actual_outcome, indicator_name, start_time, end_time, timestamp))
            
            self.connection.commit()
            
            # Invalidate cache
            if indicator_name in self.performance_cache:
                del self.performance_cache[indicator_name]
                
        except Exception as e:
            logger.error(f"Error updating outcome for {indicator_name}: {e}")
    
    def calculate_performance_metrics(self, 
                                    indicator_name: str,
                                    lookback_days: int = 30) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics
        
        Args:
            indicator_name: Name of the indicator
            lookback_days: Days to look back for calculations
            
        Returns:
            PerformanceMetrics: Calculated metrics
        """
        try:
            # Check cache first
            if (indicator_name in self.performance_cache and 
                datetime.now() - self.last_cache_update < self.cache_expiry):
                return self.performance_cache[indicator_name]
            
            # Get data from database
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            
            query = """
                SELECT prediction, actual_outcome, confidence, computation_time, data_quality
                FROM indicator_performance 
                WHERE indicator_name = ? 
                AND timestamp >= ?
                AND actual_outcome IS NOT NULL
                ORDER BY timestamp
            """
            
            df = pd.read_sql_query(query, self.connection, params=(indicator_name, cutoff_date))
            
            if len(df) < self.min_sample_size:
                logger.warning(f"Insufficient data for {indicator_name}: {len(df)} samples")
                return PerformanceMetrics(sample_size=len(df))
            
            # Calculate metrics
            predictions = df['prediction'].values
            actuals = df['actual_outcome'].values
            confidences = df['confidence'].values
            
            # Binary classification metrics
            pred_binary = (predictions > 0.5).astype(int)
            actual_binary = (actuals > 0.5).astype(int)
            
            # Accuracy
            accuracy = np.mean(pred_binary == actual_binary)
            
            # Precision, Recall, F1
            tp = np.sum((pred_binary == 1) & (actual_binary == 1))
            fp = np.sum((pred_binary == 1) & (actual_binary == 0))
            fn = np.sum((pred_binary == 0) & (actual_binary == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Hit rate (accuracy weighted by confidence)
            hit_rate = np.average(pred_binary == actual_binary, weights=confidences)
            
            # Sharpe ratio (treating predictions as strategy returns)
            returns = np.where(pred_binary == actual_binary, confidences, -confidences)
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
            
            # Other metrics
            avg_confidence = np.mean(confidences)
            error_rate = 1.0 - accuracy
            avg_computation_time = np.mean(df['computation_time'])
            avg_data_quality = np.mean(df['data_quality'])
            
            # Statistical significance
            p_value, confidence_interval = self._calculate_statistical_significance(
                pred_binary, actual_binary
            )
            
            metrics = PerformanceMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                sharpe_ratio=sharpe_ratio,
                hit_rate=hit_rate,
                avg_confidence=avg_confidence,
                error_rate=error_rate,
                computation_time=avg_computation_time,
                data_quality=avg_data_quality,
                statistical_significance=p_value,
                confidence_interval=confidence_interval,
                sample_size=len(df),
                last_updated=datetime.now()
            )
            
            # Cache the result
            self.performance_cache[indicator_name] = metrics
            self.last_cache_update = datetime.now()
            
            # Store daily metrics
            self._store_daily_metrics(indicator_name, metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {indicator_name}: {e}")
            return PerformanceMetrics()
    
    def get_performance_trend(self, 
                            indicator_name: str,
                            days: int = 30) -> Dict[str, List[float]]:
        """
        Get performance trend over time
        
        Args:
            indicator_name: Name of the indicator
            days: Number of days to analyze
            
        Returns:
            Dict with trend data
        """
        try:
            query = """
                SELECT metric_date, accuracy, sharpe_ratio, avg_confidence, sample_size
                FROM performance_metrics 
                WHERE indicator_name = ?
                AND metric_date >= date('now', '-{} days')
                ORDER BY metric_date
            """.format(days)
            
            df = pd.read_sql_query(query, self.connection, params=(indicator_name,))
            
            return {
                'dates': df['metric_date'].tolist(),
                'accuracy': df['accuracy'].tolist(),
                'sharpe_ratio': df['sharpe_ratio'].tolist(),
                'confidence': df['avg_confidence'].tolist(),
                'sample_size': df['sample_size'].tolist()
            }
            
        except Exception as e:
            logger.error(f"Error getting trend for {indicator_name}: {e}")
            return {}
    
    def get_all_indicators_summary(self) -> Dict[str, PerformanceMetrics]:
        """Get performance summary for all indicators"""
        try:
            # Get list of all indicators
            query = "SELECT DISTINCT indicator_name FROM indicator_performance"
            result = self.connection.execute(query).fetchall()
            indicators = [row[0] for row in result]
            
            summary = {}
            for indicator in indicators:
                summary[indicator] = self.calculate_performance_metrics(indicator)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting all indicators summary: {e}")
            return {}
    
    def _calculate_statistical_significance(self, 
                                          predictions: np.ndarray,
                                          actuals: np.ndarray) -> Tuple[float, Tuple[float, float]]:
        """Calculate statistical significance of predictions"""
        try:
            accuracy = np.mean(predictions == actuals)
            n = len(predictions)
            
            # Binomial test against random chance (50%)
            successes = np.sum(predictions == actuals)
            p_value = 2 * min(
                stats.binom.cdf(successes, n, 0.5),
                1 - stats.binom.cdf(successes, n, 0.5)
            )
            
            # Confidence interval for accuracy
            z_score = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
            std_error = np.sqrt(accuracy * (1 - accuracy) / n)
            margin_error = z_score * std_error
            
            confidence_interval = (
                max(0.0, accuracy - margin_error),
                min(1.0, accuracy + margin_error)
            )
            
            return p_value, confidence_interval
            
        except Exception as e:
            logger.error(f"Error calculating statistical significance: {e}")
            return 0.5, (0.0, 1.0)
    
    def _store_daily_metrics(self, indicator_name: str, metrics: PerformanceMetrics):
        """Store daily aggregated metrics"""
        try:
            today = datetime.now().date()
            
            self.connection.execute("""
                INSERT OR REPLACE INTO performance_metrics
                (indicator_name, metric_date, accuracy, precision_score, recall_score, f1_score,
                 sharpe_ratio, hit_rate, avg_confidence, error_rate, avg_computation_time,
                 avg_data_quality, statistical_significance, confidence_interval_lower,
                 confidence_interval_upper, sample_size)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                indicator_name, today, metrics.accuracy, metrics.precision, metrics.recall,
                metrics.f1_score, metrics.sharpe_ratio, metrics.hit_rate, metrics.avg_confidence,
                metrics.error_rate, metrics.computation_time, metrics.data_quality,
                metrics.statistical_significance, metrics.confidence_interval[0],
                metrics.confidence_interval[1], metrics.sample_size
            ))
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Error storing daily metrics: {e}")
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old performance data"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Delete old performance records
            result = self.connection.execute("""
                DELETE FROM indicator_performance 
                WHERE timestamp < ?
            """, (cutoff_date,))
            
            deleted_count = result.rowcount
            self.connection.commit()
            
            logger.info(f"Cleaned up {deleted_count} old performance records")
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
    
    def export_performance_data(self, 
                               indicator_name: str,
                               output_path: str,
                               days: int = 30):
        """Export performance data to CSV"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            query = """
                SELECT * FROM indicator_performance 
                WHERE indicator_name = ? 
                AND timestamp >= ?
                ORDER BY timestamp
            """
            
            df = pd.read_sql_query(query, self.connection, params=(indicator_name, cutoff_date))
            df.to_csv(output_path, index=False)
            
            logger.info(f"Exported {len(df)} records to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")