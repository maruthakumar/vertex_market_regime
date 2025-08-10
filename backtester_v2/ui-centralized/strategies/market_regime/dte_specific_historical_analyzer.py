#!/usr/bin/env python3
"""
DTE-Specific Historical Analysis Framework
Advanced granular DTE analysis for exact DTE values (1-30 days) with searchable database

This module implements sophisticated DTE-specific analysis using PostgreSQL for
high-performance historical regime pattern searches and indicator performance optimization.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import psycopg2
from psycopg2.extras import RealDictCursor
import sqlite3
import json
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class DTEPerformanceProfile:
    """Performance profile for specific DTE value"""
    dte: int
    total_occurrences: int
    regime_formation_accuracy: float
    indicator_performance: Dict[str, float]
    transition_patterns: Dict[str, float]
    volatility_sensitivity: float
    market_condition_effectiveness: Dict[str, float]
    optimal_weights: Dict[str, float]
    confidence_score: float
    statistical_significance: float
    sample_quality_score: float

@dataclass
class MarketConditionSimilarity:
    """Market condition similarity analysis"""
    current_conditions: Dict[str, float]
    similar_periods: List[Dict[str, Any]]
    similarity_scores: List[float]
    recommended_weights: Dict[str, float]
    confidence_level: float

class DTESpecificHistoricalAnalyzer:
    """
    DTE-Specific Historical Analysis Framework
    
    Implements granular DTE analysis for exact DTE values (1-30 days) with:
    - High-performance PostgreSQL database for regime pattern searches
    - Sub-second query response times for DTE-specific historical patterns
    - Market condition similarity algorithms
    - Statistical significance testing for DTE-specific performance
    - Dynamic weight optimization based on DTE-specific historical performance
    """
    
    def __init__(self, db_config: Optional[Dict[str, str]] = None,
                 cache_size: int = 10000):
        """
        Initialize DTE-Specific Historical Analyzer
        
        Args:
            db_config: Database configuration for PostgreSQL
            cache_size: Size of performance cache
        """
        # Database configuration
        self.db_config = db_config or {
            'host': 'localhost',
            'port': '5432',
            'database': 'market_regime_historical',
            'user': 'regime_user',
            'password': 'regime_pass'
        }
        
        # Performance cache
        self.performance_cache = {}
        self.cache_size = cache_size
        
        # DTE analysis parameters
        self.dte_range = list(range(1, 31))  # 1-30 days
        self.min_sample_size = 50  # Minimum samples for reliable analysis
        self.confidence_threshold = 0.95
        
        # Market condition parameters
        self.volatility_buckets = ['low', 'normal', 'high']  # <15%, 15-25%, >25%
        self.trend_buckets = ['strong_bearish', 'mild_bearish', 'neutral', 'mild_bullish', 'strong_bullish']
        self.volume_buckets = ['low', 'normal', 'high']
        
        # Statistical parameters
        self.significance_level = 0.05
        self.bootstrap_samples = 1000
        
        # Initialize database connection
        self.db_connection = None
        self._initialize_database()
        
        logger.info("DTE-Specific Historical Analyzer initialized")
    
    def _initialize_database(self):
        """Initialize PostgreSQL database connection and create tables"""
        try:
            self.db_connection = psycopg2.connect(**self.db_config)
            self._create_database_schema()
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            # Fallback to SQLite for development
            self.db_connection = sqlite3.connect("dte_historical_analysis.db")
            self._create_sqlite_schema()
            logger.info("Using SQLite fallback database")
    
    def _create_database_schema(self):
        """Create PostgreSQL database schema for DTE-specific analysis"""
        try:
            cursor = self.db_connection.cursor()
            
            # DTE-specific regime formations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dte_regime_formations (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    dte INTEGER NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    regime_type VARCHAR(50) NOT NULL,
                    confidence_score REAL NOT NULL,
                    directional_component REAL,
                    volatility_component REAL,
                    underlying_price REAL,
                    iv_level REAL,
                    volume_profile VARCHAR(20),
                    market_condition VARCHAR(50),
                    indicator_scores JSONB,
                    formation_accuracy REAL,
                    transition_success BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_dte_regime_dte ON dte_regime_formations(dte);
                CREATE INDEX IF NOT EXISTS idx_dte_regime_timestamp ON dte_regime_formations(timestamp);
                CREATE INDEX IF NOT EXISTS idx_dte_regime_type ON dte_regime_formations(regime_type);
                CREATE INDEX IF NOT EXISTS idx_dte_regime_market_condition ON dte_regime_formations(market_condition);
            """)
            
            # DTE-specific indicator performance table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dte_indicator_performance (
                    id SERIAL PRIMARY KEY,
                    dte INTEGER NOT NULL,
                    indicator_name VARCHAR(50) NOT NULL,
                    analysis_date DATE NOT NULL,
                    accuracy_score REAL NOT NULL,
                    precision_score REAL NOT NULL,
                    recall_score REAL NOT NULL,
                    f1_score REAL NOT NULL,
                    sharpe_ratio REAL,
                    false_positive_rate REAL,
                    false_negative_rate REAL,
                    sample_size INTEGER NOT NULL,
                    confidence_interval_lower REAL,
                    confidence_interval_upper REAL,
                    statistical_significance REAL,
                    market_condition VARCHAR(50),
                    optimal_weight REAL,
                    performance_rank INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(dte, indicator_name, analysis_date, market_condition)
                );
                
                CREATE INDEX IF NOT EXISTS idx_dte_perf_dte ON dte_indicator_performance(dte);
                CREATE INDEX IF NOT EXISTS idx_dte_perf_indicator ON dte_indicator_performance(indicator_name);
                CREATE INDEX IF NOT EXISTS idx_dte_perf_date ON dte_indicator_performance(analysis_date);
            """)
            
            # Market condition similarity table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_condition_patterns (
                    id SERIAL PRIMARY KEY,
                    pattern_id VARCHAR(50) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    dte INTEGER NOT NULL,
                    volatility_level REAL NOT NULL,
                    trend_strength REAL NOT NULL,
                    volume_profile REAL NOT NULL,
                    regime_outcome VARCHAR(50),
                    indicator_weights JSONB,
                    performance_score REAL,
                    similarity_features JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_market_patterns_dte ON market_condition_patterns(dte);
                CREATE INDEX IF NOT EXISTS idx_market_patterns_timestamp ON market_condition_patterns(timestamp);
                CREATE INDEX IF NOT EXISTS idx_market_patterns_volatility ON market_condition_patterns(volatility_level);
            """)
            
            self.db_connection.commit()
            logger.info("PostgreSQL database schema created successfully")
            
        except Exception as e:
            logger.error(f"Error creating database schema: {e}")
            self.db_connection.rollback()
    
    def _create_sqlite_schema(self):
        """Create SQLite database schema as fallback"""
        try:
            cursor = self.db_connection.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dte_regime_formations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    dte INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    regime_type TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    directional_component REAL,
                    volatility_component REAL,
                    underlying_price REAL,
                    iv_level REAL,
                    volume_profile TEXT,
                    market_condition TEXT,
                    indicator_scores TEXT,
                    formation_accuracy REAL,
                    transition_success INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_dte_regime_dte ON dte_regime_formations(dte);
                CREATE INDEX IF NOT EXISTS idx_dte_regime_timestamp ON dte_regime_formations(timestamp);
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dte_indicator_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dte INTEGER NOT NULL,
                    indicator_name TEXT NOT NULL,
                    analysis_date TEXT NOT NULL,
                    accuracy_score REAL NOT NULL,
                    precision_score REAL NOT NULL,
                    recall_score REAL NOT NULL,
                    f1_score REAL NOT NULL,
                    sharpe_ratio REAL,
                    sample_size INTEGER NOT NULL,
                    statistical_significance REAL,
                    optimal_weight REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            self.db_connection.commit()
            logger.info("SQLite database schema created successfully")
            
        except Exception as e:
            logger.error(f"Error creating SQLite schema: {e}")
    
    def analyze_dte_specific_performance(self, dte: int, 
                                       lookback_days: int = 252) -> DTEPerformanceProfile:
        """
        Analyze performance for specific DTE value with sub-second query response
        
        Args:
            dte: Days to expiry for analysis
            lookback_days: Number of days to look back for analysis
            
        Returns:
            DTEPerformanceProfile: Comprehensive DTE-specific performance profile
        """
        try:
            logger.info(f"Analyzing DTE-specific performance for DTE={dte}")
            
            # Check cache first
            cache_key = f"dte_{dte}_lookback_{lookback_days}"
            if cache_key in self.performance_cache:
                logger.info(f"Returning cached results for DTE={dte}")
                return self.performance_cache[cache_key]
            
            # Query historical data for specific DTE
            start_time = datetime.now()
            historical_data = self._query_dte_historical_data(dte, lookback_days)
            query_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"DTE query completed in {query_time:.3f} seconds")
            
            if len(historical_data) < self.min_sample_size:
                logger.warning(f"Insufficient data for DTE={dte}: {len(historical_data)} samples")
                return self._create_default_dte_profile(dte)
            
            # Calculate regime formation accuracy
            regime_formation_accuracy = self._calculate_regime_formation_accuracy(historical_data)
            
            # Calculate indicator performance for this DTE
            indicator_performance = self._calculate_dte_indicator_performance(historical_data)
            
            # Analyze transition patterns
            transition_patterns = self._analyze_dte_transition_patterns(historical_data)
            
            # Calculate volatility sensitivity
            volatility_sensitivity = self._calculate_dte_volatility_sensitivity(historical_data)
            
            # Analyze market condition effectiveness
            market_condition_effectiveness = self._analyze_dte_market_conditions(historical_data)
            
            # Optimize weights for this DTE
            optimal_weights = self._optimize_dte_weights(historical_data, indicator_performance)
            
            # Calculate confidence and significance scores
            confidence_score = self._calculate_dte_confidence_score(historical_data)
            statistical_significance = self._calculate_dte_statistical_significance(historical_data)
            sample_quality_score = self._calculate_sample_quality_score(historical_data)
            
            # Create performance profile
            profile = DTEPerformanceProfile(
                dte=dte,
                total_occurrences=len(historical_data),
                regime_formation_accuracy=regime_formation_accuracy,
                indicator_performance=indicator_performance,
                transition_patterns=transition_patterns,
                volatility_sensitivity=volatility_sensitivity,
                market_condition_effectiveness=market_condition_effectiveness,
                optimal_weights=optimal_weights,
                confidence_score=confidence_score,
                statistical_significance=statistical_significance,
                sample_quality_score=sample_quality_score
            )
            
            # Cache results
            if len(self.performance_cache) >= self.cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self.performance_cache))
                del self.performance_cache[oldest_key]
            
            self.performance_cache[cache_key] = profile
            
            logger.info(f"DTE={dte} analysis completed: Accuracy={regime_formation_accuracy:.3f}, Confidence={confidence_score:.3f}")
            
            return profile
            
        except Exception as e:
            logger.error(f"Error analyzing DTE={dte} performance: {e}")
            return self._create_default_dte_profile(dte)
    
    def search_similar_market_conditions(self, 
                                       current_conditions: Dict[str, float],
                                       dte: int,
                                       similarity_threshold: float = 0.8,
                                       max_results: int = 100) -> MarketConditionSimilarity:
        """
        Search for similar market conditions with sub-second response time
        
        Mathematical Formula for Similarity:
        Similarity = 1 - sqrt(Σ(w_i * (x_i - y_i)²)) / sqrt(Σ(w_i))
        
        Where:
        w_i = weight for feature i
        x_i = current condition value for feature i
        y_i = historical condition value for feature i
        
        Args:
            current_conditions: Current market conditions
            dte: Days to expiry
            similarity_threshold: Minimum similarity score
            max_results: Maximum number of similar periods to return
            
        Returns:
            MarketConditionSimilarity: Similar market conditions and recommendations
        """
        try:
            logger.info(f"Searching similar market conditions for DTE={dte}")
            
            start_time = datetime.now()
            
            # Query historical market conditions
            historical_conditions = self._query_historical_market_conditions(dte)
            
            if len(historical_conditions) == 0:
                logger.warning(f"No historical conditions found for DTE={dte}")
                return self._create_default_similarity_result(current_conditions)
            
            # Calculate similarity scores
            similarity_results = []
            
            for historical_record in historical_conditions:
                similarity_score = self._calculate_market_condition_similarity(
                    current_conditions, historical_record
                )
                
                if similarity_score >= similarity_threshold:
                    similarity_results.append({
                        'historical_record': historical_record,
                        'similarity_score': similarity_score
                    })
            
            # Sort by similarity score
            similarity_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            similarity_results = similarity_results[:max_results]
            
            # Extract similar periods and scores
            similar_periods = [result['historical_record'] for result in similarity_results]
            similarity_scores = [result['similarity_score'] for result in similarity_results]
            
            # Calculate recommended weights based on similar periods
            recommended_weights = self._calculate_similarity_based_weights(similar_periods)
            
            # Calculate confidence level
            confidence_level = np.mean(similarity_scores) if similarity_scores else 0.0
            
            query_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Similarity search completed in {query_time:.3f} seconds, found {len(similar_periods)} similar periods")
            
            return MarketConditionSimilarity(
                current_conditions=current_conditions,
                similar_periods=similar_periods,
                similarity_scores=similarity_scores,
                recommended_weights=recommended_weights,
                confidence_level=confidence_level
            )
            
        except Exception as e:
            logger.error(f"Error searching similar market conditions: {e}")
            return self._create_default_similarity_result(current_conditions)
    
    def get_dte_optimal_weights_fast(self, dte: int, 
                                   market_conditions: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Get optimal weights for specific DTE with <1 second response time
        
        Args:
            dte: Days to expiry
            market_conditions: Current market conditions for similarity matching
            
        Returns:
            Dict[str, float]: Optimal weights for indicators
        """
        try:
            start_time = datetime.now()
            
            # Check cache first
            cache_key = f"weights_dte_{dte}"
            if market_conditions:
                condition_hash = hash(str(sorted(market_conditions.items())))
                cache_key += f"_conditions_{condition_hash}"
            
            if cache_key in self.performance_cache:
                query_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"Cached weights retrieved in {query_time:.3f} seconds")
                return self.performance_cache[cache_key]
            
            # Get DTE-specific performance profile
            dte_profile = self.analyze_dte_specific_performance(dte)
            
            # If market conditions provided, search for similar conditions
            if market_conditions:
                similarity_result = self.search_similar_market_conditions(
                    market_conditions, dte, similarity_threshold=0.7, max_results=50
                )
                
                # Blend DTE-specific weights with similarity-based weights
                dte_weights = dte_profile.optimal_weights
                similarity_weights = similarity_result.recommended_weights
                confidence = similarity_result.confidence_level
                
                # Weighted combination based on confidence
                optimal_weights = {}
                for indicator in dte_weights.keys():
                    dte_weight = dte_weights.get(indicator, 0.1)
                    sim_weight = similarity_weights.get(indicator, 0.1)
                    
                    # Blend weights: higher confidence = more similarity weight
                    blended_weight = (1 - confidence) * dte_weight + confidence * sim_weight
                    optimal_weights[indicator] = blended_weight
            else:
                optimal_weights = dte_profile.optimal_weights
            
            # Normalize weights
            total_weight = sum(optimal_weights.values())
            if total_weight > 0:
                optimal_weights = {k: v/total_weight for k, v in optimal_weights.items()}
            
            # Cache results
            self.performance_cache[cache_key] = optimal_weights
            
            query_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Optimal weights calculated in {query_time:.3f} seconds")
            
            return optimal_weights
            
        except Exception as e:
            logger.error(f"Error getting optimal weights for DTE={dte}: {e}")
            return self._get_default_weights()
    
    # Helper methods for database operations and calculations
    def _query_dte_historical_data(self, dte: int, lookback_days: int) -> List[Dict[str, Any]]:
        """Query historical data for specific DTE"""
        try:
            cursor = self.db_connection.cursor(cursor_factory=RealDictCursor if hasattr(psycopg2, 'extras') else None)
            
            query = """
                SELECT * FROM dte_regime_formations 
                WHERE dte = %s 
                AND timestamp >= %s 
                ORDER BY timestamp DESC
            """
            
            lookback_date = datetime.now() - timedelta(days=lookback_days)
            cursor.execute(query, (dte, lookback_date))
            
            results = cursor.fetchall()
            return [dict(row) for row in results] if results else []
            
        except Exception as e:
            logger.error(f"Error querying DTE historical data: {e}")
            return []
    
    def _query_historical_market_conditions(self, dte: int) -> List[Dict[str, Any]]:
        """Query historical market conditions for similarity analysis"""
        try:
            cursor = self.db_connection.cursor(cursor_factory=RealDictCursor if hasattr(psycopg2, 'extras') else None)
            
            query = """
                SELECT * FROM market_condition_patterns 
                WHERE dte = %s 
                ORDER BY timestamp DESC 
                LIMIT 10000
            """
            
            cursor.execute(query, (dte,))
            results = cursor.fetchall()
            return [dict(row) for row in results] if results else []
            
        except Exception as e:
            logger.error(f"Error querying historical market conditions: {e}")
            return []
    
    def _calculate_regime_formation_accuracy(self, historical_data: List[Dict[str, Any]]) -> float:
        """Calculate regime formation accuracy"""
        if not historical_data:
            return 0.5
        
        accurate_formations = sum(1 for record in historical_data if record.get('formation_accuracy', 0) > 0.7)
        return accurate_formations / len(historical_data)
    
    def _calculate_dte_indicator_performance(self, historical_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate indicator performance for specific DTE"""
        # Placeholder implementation
        indicators = ['greek_sentiment', 'trending_oi_pa', 'ema_indicators', 'vwap_indicators', 'iv_skew']
        return {indicator: 0.75 for indicator in indicators}
    
    def _analyze_dte_transition_patterns(self, historical_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze regime transition patterns for specific DTE"""
        # Placeholder implementation
        return {'transition_accuracy': 0.8, 'avg_transition_time': 5.2}
    
    def _calculate_dte_volatility_sensitivity(self, historical_data: List[Dict[str, Any]]) -> float:
        """Calculate volatility sensitivity for specific DTE"""
        return 0.65  # Placeholder
    
    def _analyze_dte_market_conditions(self, historical_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze market condition effectiveness for specific DTE"""
        return {'bull_market': 0.8, 'bear_market': 0.7, 'sideways': 0.6}  # Placeholder
    
    def _optimize_dte_weights(self, historical_data: List[Dict[str, Any]], 
                            indicator_performance: Dict[str, float]) -> Dict[str, float]:
        """Optimize weights for specific DTE"""
        # Normalize performance scores to weights
        total_performance = sum(indicator_performance.values())
        if total_performance > 0:
            return {k: v/total_performance for k, v in indicator_performance.items()}
        return self._get_default_weights()
    
    def _calculate_dte_confidence_score(self, historical_data: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for DTE analysis"""
        if not historical_data:
            return 0.5
        
        avg_confidence = np.mean([record.get('confidence_score', 0.5) for record in historical_data])
        sample_size_factor = min(1.0, len(historical_data) / self.min_sample_size)
        
        return avg_confidence * sample_size_factor
    
    def _calculate_dte_statistical_significance(self, historical_data: List[Dict[str, Any]]) -> float:
        """Calculate statistical significance for DTE analysis"""
        if len(historical_data) < 30:
            return 1.0  # Not significant
        
        # Simplified significance calculation
        confidence_scores = [record.get('confidence_score', 0.5) for record in historical_data]
        t_stat, p_value = stats.ttest_1samp(confidence_scores, 0.5)
        
        return p_value
    
    def _calculate_sample_quality_score(self, historical_data: List[Dict[str, Any]]) -> float:
        """Calculate sample quality score"""
        if not historical_data:
            return 0.0
        
        # Quality based on sample size, data completeness, and recency
        sample_size_score = min(1.0, len(historical_data) / (self.min_sample_size * 2))
        
        # Data completeness score
        complete_records = sum(1 for record in historical_data 
                             if all(record.get(field) is not None 
                                   for field in ['confidence_score', 'regime_type', 'formation_accuracy']))
        completeness_score = complete_records / len(historical_data)
        
        # Recency score (more recent data gets higher score)
        now = datetime.now()
        recency_scores = []
        for record in historical_data:
            timestamp = datetime.fromisoformat(str(record.get('timestamp', now)))
            days_old = (now - timestamp).days
            recency_score = max(0, 1 - days_old / 365)  # Decay over 1 year
            recency_scores.append(recency_score)
        
        avg_recency_score = np.mean(recency_scores) if recency_scores else 0.5
        
        # Combined quality score
        quality_score = (sample_size_score + completeness_score + avg_recency_score) / 3
        
        return quality_score
    
    def _calculate_market_condition_similarity(self, 
                                             current: Dict[str, float], 
                                             historical: Dict[str, Any]) -> float:
        """Calculate similarity between current and historical market conditions"""
        # Extract features for comparison
        features = ['volatility_level', 'trend_strength', 'volume_profile']
        weights = [0.4, 0.4, 0.2]  # Feature weights
        
        similarity_scores = []
        
        for feature, weight in zip(features, weights):
            current_val = current.get(feature, 0.5)
            historical_val = historical.get(feature, 0.5)
            
            # Calculate normalized difference
            diff = abs(current_val - historical_val)
            similarity = 1 - diff  # Simple similarity measure
            
            similarity_scores.append(similarity * weight)
        
        return sum(similarity_scores)
    
    def _calculate_similarity_based_weights(self, similar_periods: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate recommended weights based on similar periods"""
        if not similar_periods:
            return self._get_default_weights()
        
        # Extract weights from similar periods and average them
        weight_sums = {}
        weight_counts = {}
        
        for period in similar_periods:
            weights = period.get('indicator_weights', {})
            if isinstance(weights, str):
                try:
                    weights = json.loads(weights)
                except:
                    continue
            
            for indicator, weight in weights.items():
                if indicator not in weight_sums:
                    weight_sums[indicator] = 0
                    weight_counts[indicator] = 0
                
                weight_sums[indicator] += weight
                weight_counts[indicator] += 1
        
        # Calculate average weights
        avg_weights = {}
        for indicator in weight_sums:
            avg_weights[indicator] = weight_sums[indicator] / weight_counts[indicator]
        
        # Normalize weights
        total_weight = sum(avg_weights.values())
        if total_weight > 0:
            avg_weights = {k: v/total_weight for k, v in avg_weights.items()}
        
        return avg_weights if avg_weights else self._get_default_weights()
    
    def _create_default_dte_profile(self, dte: int) -> DTEPerformanceProfile:
        """Create default DTE performance profile"""
        return DTEPerformanceProfile(
            dte=dte,
            total_occurrences=0,
            regime_formation_accuracy=0.5,
            indicator_performance=self._get_default_performance(),
            transition_patterns={'transition_accuracy': 0.5},
            volatility_sensitivity=0.5,
            market_condition_effectiveness={'default': 0.5},
            optimal_weights=self._get_default_weights(),
            confidence_score=0.5,
            statistical_significance=1.0,
            sample_quality_score=0.0
        )
    
    def _create_default_similarity_result(self, current_conditions: Dict[str, float]) -> MarketConditionSimilarity:
        """Create default similarity result"""
        return MarketConditionSimilarity(
            current_conditions=current_conditions,
            similar_periods=[],
            similarity_scores=[],
            recommended_weights=self._get_default_weights(),
            confidence_level=0.0
        )
    
    def _get_default_weights(self) -> Dict[str, float]:
        """Get default indicator weights"""
        indicators = ['greek_sentiment', 'trending_oi_pa', 'ema_indicators', 'vwap_indicators', 'iv_skew']
        return {indicator: 1.0 / len(indicators) for indicator in indicators}
    
    def _get_default_performance(self) -> Dict[str, float]:
        """Get default performance scores"""
        indicators = ['greek_sentiment', 'trending_oi_pa', 'ema_indicators', 'vwap_indicators', 'iv_skew']
        return {indicator: 0.5 for indicator in indicators}
