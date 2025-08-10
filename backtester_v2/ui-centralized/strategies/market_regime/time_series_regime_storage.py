"""
Time-Series Market Regime Storage System

This module provides comprehensive time-series storage and retrieval for
market regime classifications, enabling historical analysis, performance
tracking, and regime-based strategy optimization.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import sqlite3
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class TimeSeriesRegimeStorage:
    """
    Time-series storage system for market regime classifications
    
    This class provides:
    - Historical regime storage with full metadata
    - Performance tracking and analysis
    - Regime transition analysis
    - User-specific regime configurations
    - Strategy consolidation support
    """
    
    def __init__(self, db_path: str = "market_regime_timeseries.db"):
        """
        Initialize Time-Series Regime Storage
        
        Args:
            db_path (str): Path to SQLite database file
        """
        self.db_path = db_path
        self.connection = None
        
        # Initialize database
        self._initialize_database()
        
        logger.info(f"TimeSeriesRegimeStorage initialized with database: {db_path}")
    
    def _initialize_database(self):
        """Initialize SQLite database with required tables"""
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.execute("PRAGMA foreign_keys = ON")
            
            # Create tables
            self._create_regime_history_table()
            self._create_indicator_values_table()
            self._create_regime_performance_table()
            self._create_user_configurations_table()
            self._create_regime_transitions_table()
            self._create_strategy_mappings_table()
            
            self.connection.commit()
            logger.info("✅ Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def _create_regime_history_table(self):
        """Create regime history table"""
        sql = """
        CREATE TABLE IF NOT EXISTS regime_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            regime_id VARCHAR(50) NOT NULL,
            regime_name VARCHAR(100) NOT NULL,
            regime_type VARCHAR(20) NOT NULL,
            confidence_score REAL NOT NULL,
            directional_component REAL,
            volatility_component REAL,
            indicator_agreement REAL,
            signal_strength REAL,
            market_condition VARCHAR(20),
            user_id VARCHAR(50),
            configuration_id VARCHAR(50),
            metadata TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(timestamp, symbol, timeframe, user_id, configuration_id)
        )
        """
        self.connection.execute(sql)
        
        # Create indexes
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_regime_timestamp ON regime_history(timestamp)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_regime_symbol ON regime_history(symbol)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_regime_user ON regime_history(user_id)")
    
    def _create_indicator_values_table(self):
        """Create indicator values table"""
        sql = """
        CREATE TABLE IF NOT EXISTS indicator_values (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            regime_history_id INTEGER NOT NULL,
            indicator_id VARCHAR(50) NOT NULL,
            indicator_name VARCHAR(100) NOT NULL,
            indicator_category VARCHAR(50) NOT NULL,
            raw_value REAL,
            normalized_value REAL,
            weight REAL NOT NULL,
            contribution REAL,
            performance_score REAL,
            metadata TEXT,
            FOREIGN KEY (regime_history_id) REFERENCES regime_history (id)
        )
        """
        self.connection.execute(sql)
        
        # Create indexes
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_indicator_regime ON indicator_values(regime_history_id)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_indicator_id ON indicator_values(indicator_id)")
    
    def _create_regime_performance_table(self):
        """Create regime performance tracking table"""
        sql = """
        CREATE TABLE IF NOT EXISTS regime_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            regime_id VARCHAR(50) NOT NULL,
            user_id VARCHAR(50),
            configuration_id VARCHAR(50),
            analysis_date DATE NOT NULL,
            total_occurrences INTEGER NOT NULL,
            average_confidence REAL NOT NULL,
            accuracy_score REAL,
            precision_score REAL,
            recall_score REAL,
            f1_score REAL,
            average_duration REAL,
            transition_accuracy REAL,
            false_positive_rate REAL,
            false_negative_rate REAL,
            performance_window_days INTEGER NOT NULL,
            benchmark_comparison REAL,
            metadata TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(regime_id, user_id, configuration_id, analysis_date)
        )
        """
        self.connection.execute(sql)
        
        # Create indexes
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_performance_regime ON regime_performance(regime_id)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_performance_date ON regime_performance(analysis_date)")
    
    def _create_user_configurations_table(self):
        """Create user configurations table"""
        sql = """
        CREATE TABLE IF NOT EXISTS user_configurations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id VARCHAR(50) NOT NULL,
            configuration_id VARCHAR(50) NOT NULL,
            configuration_name VARCHAR(100) NOT NULL,
            indicator_weights TEXT NOT NULL,
            regime_thresholds TEXT NOT NULL,
            confidence_settings TEXT NOT NULL,
            timeframe_settings TEXT NOT NULL,
            custom_parameters TEXT,
            performance_metrics TEXT,
            is_active BOOLEAN DEFAULT TRUE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, configuration_id)
        )
        """
        self.connection.execute(sql)
        
        # Create indexes
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_user_config ON user_configurations(user_id)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_config_active ON user_configurations(is_active)")
    
    def _create_regime_transitions_table(self):
        """Create regime transitions table"""
        sql = """
        CREATE TABLE IF NOT EXISTS regime_transitions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            from_regime_id VARCHAR(50) NOT NULL,
            to_regime_id VARCHAR(50) NOT NULL,
            transition_timestamp DATETIME NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            user_id VARCHAR(50),
            configuration_id VARCHAR(50),
            transition_confidence REAL NOT NULL,
            transition_duration REAL,
            trigger_indicators TEXT,
            market_conditions TEXT,
            was_predicted BOOLEAN DEFAULT FALSE,
            prediction_accuracy REAL,
            metadata TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
        self.connection.execute(sql)
        
        # Create indexes
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_transition_timestamp ON regime_transitions(transition_timestamp)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_transition_from ON regime_transitions(from_regime_id)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_transition_to ON regime_transitions(to_regime_id)")
    
    def _create_strategy_mappings_table(self):
        """Create strategy mappings table"""
        sql = """
        CREATE TABLE IF NOT EXISTS strategy_mappings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id VARCHAR(50) NOT NULL,
            configuration_id VARCHAR(50) NOT NULL,
            strategy_type VARCHAR(50) NOT NULL,
            regime_id VARCHAR(50) NOT NULL,
            enable_strategy BOOLEAN DEFAULT TRUE,
            weight_multiplier REAL DEFAULT 1.0,
            custom_parameters TEXT,
            performance_metrics TEXT,
            last_performance_update DATETIME,
            is_active BOOLEAN DEFAULT TRUE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, configuration_id, strategy_type, regime_id)
        )
        """
        self.connection.execute(sql)
        
        # Create indexes
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_strategy_user ON strategy_mappings(user_id)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_strategy_regime ON strategy_mappings(regime_id)")
    
    def store_regime_classification(self, regime_data: Dict[str, Any]) -> int:
        """
        Store regime classification with full metadata
        
        Args:
            regime_data (Dict): Complete regime classification data
            
        Returns:
            int: ID of stored regime record
        """
        try:
            # Insert regime history record
            regime_sql = """
            INSERT OR REPLACE INTO regime_history 
            (timestamp, symbol, timeframe, regime_id, regime_name, regime_type,
             confidence_score, directional_component, volatility_component,
             indicator_agreement, signal_strength, market_condition,
             user_id, configuration_id, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            regime_values = (
                regime_data['timestamp'],
                regime_data.get('symbol', 'NIFTY'),
                regime_data.get('timeframe', '5min'),
                regime_data['regime_id'],
                regime_data['regime_name'],
                regime_data['regime_type'],
                regime_data['confidence_score'],
                regime_data.get('directional_component'),
                regime_data.get('volatility_component'),
                regime_data.get('indicator_agreement'),
                regime_data.get('signal_strength'),
                regime_data.get('market_condition'),
                regime_data.get('user_id'),
                regime_data.get('configuration_id'),
                json.dumps(regime_data.get('metadata', {}))
            )
            
            cursor = self.connection.execute(regime_sql, regime_values)
            regime_history_id = cursor.lastrowid
            
            # Store indicator values if provided
            if 'indicator_values' in regime_data:
                self._store_indicator_values(regime_history_id, regime_data['indicator_values'])
            
            self.connection.commit()
            
            logger.debug(f"Stored regime classification: {regime_data['regime_id']} at {regime_data['timestamp']}")
            return regime_history_id
            
        except Exception as e:
            logger.error(f"Error storing regime classification: {e}")
            self.connection.rollback()
            raise
    
    def _store_indicator_values(self, regime_history_id: int, indicator_values: List[Dict[str, Any]]):
        """Store indicator values for a regime classification"""
        try:
            indicator_sql = """
            INSERT INTO indicator_values 
            (regime_history_id, indicator_id, indicator_name, indicator_category,
             raw_value, normalized_value, weight, contribution, performance_score, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            for indicator in indicator_values:
                values = (
                    regime_history_id,
                    indicator['indicator_id'],
                    indicator['indicator_name'],
                    indicator['indicator_category'],
                    indicator.get('raw_value'),
                    indicator.get('normalized_value'),
                    indicator['weight'],
                    indicator.get('contribution'),
                    indicator.get('performance_score'),
                    json.dumps(indicator.get('metadata', {}))
                )
                
                self.connection.execute(indicator_sql, values)
                
        except Exception as e:
            logger.error(f"Error storing indicator values: {e}")
            raise
    
    def get_regime_history(self, symbol: str = 'NIFTY', timeframe: str = '5min',
                          start_date: Optional[datetime] = None, end_date: Optional[datetime] = None,
                          user_id: Optional[str] = None, limit: int = 1000) -> pd.DataFrame:
        """
        Get regime history for analysis
        
        Args:
            symbol (str): Symbol to query
            timeframe (str): Timeframe to query
            start_date (datetime, optional): Start date filter
            end_date (datetime, optional): End date filter
            user_id (str, optional): User ID filter
            limit (int): Maximum records to return
            
        Returns:
            pd.DataFrame: Regime history data
        """
        try:
            sql = """
            SELECT * FROM regime_history 
            WHERE symbol = ? AND timeframe = ?
            """
            params = [symbol, timeframe]
            
            if start_date:
                sql += " AND timestamp >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                sql += " AND timestamp <= ?"
                params.append(end_date.isoformat())
            
            if user_id:
                sql += " AND user_id = ?"
                params.append(user_id)
            
            sql += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            df = pd.read_sql_query(sql, self.connection, params=params)
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['metadata'] = df['metadata'].apply(lambda x: json.loads(x) if x else {})
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting regime history: {e}")
            return pd.DataFrame()
    
    def get_regime_performance_analysis(self, regime_id: str, user_id: Optional[str] = None,
                                      days: int = 30) -> Dict[str, Any]:
        """
        Get comprehensive performance analysis for a regime
        
        Args:
            regime_id (str): Regime ID to analyze
            user_id (str, optional): User ID filter
            days (int): Analysis window in days
            
        Returns:
            Dict: Performance analysis results
        """
        try:
            # Get recent regime occurrences
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            sql = """
            SELECT * FROM regime_history 
            WHERE regime_id = ? AND timestamp >= ? AND timestamp <= ?
            """
            params = [regime_id, start_date.isoformat(), end_date.isoformat()]
            
            if user_id:
                sql += " AND user_id = ?"
                params.append(user_id)
            
            df = pd.read_sql_query(sql, self.connection, params=params)
            
            if df.empty:
                return {'error': 'No data found for analysis'}
            
            # Calculate performance metrics
            analysis = {
                'regime_id': regime_id,
                'analysis_period': f"{start_date.date()} to {end_date.date()}",
                'total_occurrences': len(df),
                'average_confidence': df['confidence_score'].mean(),
                'confidence_std': df['confidence_score'].std(),
                'min_confidence': df['confidence_score'].min(),
                'max_confidence': df['confidence_score'].max(),
                'regime_distribution': df['regime_name'].value_counts().to_dict(),
                'timeframe_distribution': df['timeframe'].value_counts().to_dict()
            }
            
            # Calculate regime stability (how long regimes last)
            df_sorted = df.sort_values('timestamp')
            if len(df_sorted) > 1:
                durations = []
                current_regime = None
                start_time = None
                
                for _, row in df_sorted.iterrows():
                    if current_regime != row['regime_id']:
                        if current_regime is not None and start_time is not None:
                            duration = (pd.to_datetime(row['timestamp']) - start_time).total_seconds() / 60
                            durations.append(duration)
                        current_regime = row['regime_id']
                        start_time = pd.to_datetime(row['timestamp'])
                
                if durations:
                    analysis['average_duration_minutes'] = np.mean(durations)
                    analysis['duration_std'] = np.std(durations)
                    analysis['min_duration'] = np.min(durations)
                    analysis['max_duration'] = np.max(durations)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error getting regime performance analysis: {e}")
            return {'error': str(e)}
    
    def store_user_configuration(self, user_id: str, configuration_id: str,
                               configuration_data: Dict[str, Any]) -> bool:
        """
        Store user-specific regime configuration
        
        Args:
            user_id (str): User identifier
            configuration_id (str): Configuration identifier
            configuration_data (Dict): Configuration data
            
        Returns:
            bool: Success status
        """
        try:
            sql = """
            INSERT OR REPLACE INTO user_configurations 
            (user_id, configuration_id, configuration_name, indicator_weights,
             regime_thresholds, confidence_settings, timeframe_settings,
             custom_parameters, performance_metrics, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            values = (
                user_id,
                configuration_id,
                configuration_data.get('configuration_name', f'Config_{configuration_id}'),
                json.dumps(configuration_data.get('indicator_weights', {})),
                json.dumps(configuration_data.get('regime_thresholds', {})),
                json.dumps(configuration_data.get('confidence_settings', {})),
                json.dumps(configuration_data.get('timeframe_settings', {})),
                json.dumps(configuration_data.get('custom_parameters', {})),
                json.dumps(configuration_data.get('performance_metrics', {})),
                datetime.now().isoformat()
            )
            
            self.connection.execute(sql, values)
            self.connection.commit()
            
            logger.info(f"Stored user configuration: {user_id}/{configuration_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing user configuration: {e}")
            return False
    
    def get_user_configurations(self, user_id: str) -> pd.DataFrame:
        """Get all configurations for a user"""
        try:
            sql = """
            SELECT * FROM user_configurations 
            WHERE user_id = ? AND is_active = TRUE
            ORDER BY updated_at DESC
            """
            
            df = pd.read_sql_query(sql, self.connection, params=[user_id])
            
            if not df.empty:
                # Parse JSON columns
                json_columns = ['indicator_weights', 'regime_thresholds', 'confidence_settings',
                              'timeframe_settings', 'custom_parameters', 'performance_metrics']
                
                for col in json_columns:
                    if col in df.columns:
                        df[col] = df[col].apply(lambda x: json.loads(x) if x else {})
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting user configurations: {e}")
            return pd.DataFrame()
    
    def close_connection(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
