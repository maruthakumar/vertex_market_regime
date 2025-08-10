"""
Correlation Analyzer - Mathematical Correlation Analysis
=======================================================

Performs mathematical correlation analysis including Pearson correlation,
time-decay weighting, and pattern similarity scoring for enhanced accuracy.

Author: Market Regime Refactoring Team
Date: 2025-07-06
Version: 2.0.0 - Enhanced Correlation Analysis
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CorrelationAnalysisResult:
    """Result structure for correlation analysis"""
    pearson_correlation: float
    correlation_confidence: float
    pattern_similarity_score: float
    time_decay_weight: float
    mathematical_accuracy: bool
    correlation_threshold_met: bool
    historical_pattern_match: Optional[Dict[str, Any]]

class CorrelationAnalyzer:
    """
    Advanced correlation analysis with mathematical precision
    
    Features:
    - Pearson correlation calculation with confidence intervals
    - Time-decay weighting using exponential decay
    - Pattern similarity scoring using cosine similarity
    - Mathematical accuracy validation (±0.001 tolerance)
    - Historical pattern matching
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Correlation Analyzer"""
        self.config = config or {}
        
        # Mathematical precision
        self.mathematical_tolerance = self.config.get('mathematical_tolerance', 0.001)
        
        # Correlation thresholds
        self.pearson_threshold = self.config.get('pearson_threshold', 0.80)  # >0.80 threshold
        self.min_correlation_confidence = self.config.get('min_correlation_confidence', 0.7)
        
        # Time-decay parameters
        self.decay_lambda = self.config.get('decay_lambda', 0.1)  # λ for exp(-λ × (T-t))
        self.max_time_periods = self.config.get('max_time_periods', 20)
        
        # Pattern similarity parameters
        self.similarity_threshold = self.config.get('similarity_threshold', 0.85)
        self.min_pattern_length = self.config.get('min_pattern_length', 5)
        
        # Historical analysis
        self.enable_historical_matching = self.config.get('enable_historical_matching', True)
        self.historical_lookback = self.config.get('historical_lookback', 100)
        
        # Analysis history
        self.correlation_history = []
        self.pattern_library = []
        
        logger.info("CorrelationAnalyzer initialized with mathematical precision")
    
    def analyze_correlations(self, 
                           current_data: pd.DataFrame,
                           historical_data: Optional[pd.DataFrame] = None,
                           reference_patterns: Optional[List[Dict]] = None) -> CorrelationAnalysisResult:
        """
        Perform comprehensive correlation analysis
        
        Args:
            current_data: Current market data
            historical_data: Historical data for comparison
            reference_patterns: Reference patterns for similarity analysis
            
        Returns:
            CorrelationAnalysisResult: Comprehensive correlation analysis
        """
        try:
            # Initialize result components
            pearson_correlation = 0.0
            correlation_confidence = 0.0
            pattern_similarity_score = 0.0
            time_decay_weight = 1.0
            mathematical_accuracy = True
            historical_pattern_match = None
            
            # 1. Pearson Correlation Analysis
            if historical_data is not None:
                pearson_result = self._calculate_pearson_correlation(current_data, historical_data)
                pearson_correlation = pearson_result['correlation']
                correlation_confidence = pearson_result['confidence']
                mathematical_accuracy = pearson_result['mathematical_accuracy']
            
            # 2. Time-Decay Weighting
            if historical_data is not None:
                time_decay_weight = self._calculate_time_decay_weight(historical_data)
            
            # 3. Pattern Similarity Analysis
            if reference_patterns:
                pattern_similarity_score = self._calculate_pattern_similarity(
                    current_data, reference_patterns
                )
            
            # 4. Historical Pattern Matching
            if self.enable_historical_matching and historical_data is not None:
                historical_pattern_match = self._find_historical_pattern_matches(
                    current_data, historical_data
                )
            
            # 5. Correlation Threshold Check
            correlation_threshold_met = (
                abs(pearson_correlation) >= self.pearson_threshold and
                correlation_confidence >= self.min_correlation_confidence
            )
            
            # Record analysis
            self._record_correlation_analysis(pearson_correlation, correlation_confidence)
            
            return CorrelationAnalysisResult(
                pearson_correlation=pearson_correlation,
                correlation_confidence=correlation_confidence,
                pattern_similarity_score=pattern_similarity_score,
                time_decay_weight=time_decay_weight,
                mathematical_accuracy=mathematical_accuracy,
                correlation_threshold_met=correlation_threshold_met,
                historical_pattern_match=historical_pattern_match
            )
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
            return self._get_default_correlation_result()
    
    def _calculate_pearson_correlation(self, 
                                     current_data: pd.DataFrame,
                                     historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Pearson correlation with confidence assessment"""
        try:
            # Extract price series for correlation
            current_prices = self._extract_price_series(current_data)
            historical_prices = self._extract_price_series(historical_data)
            
            if len(current_prices) < 3 or len(historical_prices) < 3:
                return {
                    'correlation': 0.0,
                    'confidence': 0.0,
                    'mathematical_accuracy': False,
                    'p_value': 1.0
                }
            
            # Align series lengths
            min_length = min(len(current_prices), len(historical_prices))
            current_aligned = current_prices[-min_length:]
            historical_aligned = historical_prices[-min_length:]
            
            # Calculate Pearson correlation
            correlation, p_value = pearsonr(current_aligned, historical_aligned)
            
            # Handle NaN values
            if np.isnan(correlation) or np.isnan(p_value):
                correlation = 0.0
                p_value = 1.0
            
            # Calculate confidence (inverse of p-value, adjusted)
            confidence = max(0.0, 1.0 - p_value) if p_value <= 1.0 else 0.0
            
            # Mathematical accuracy check
            mathematical_accuracy = self._validate_mathematical_accuracy(
                correlation, current_aligned, historical_aligned
            )
            
            return {
                'correlation': correlation,
                'confidence': confidence,
                'mathematical_accuracy': mathematical_accuracy,
                'p_value': p_value,
                'sample_size': min_length
            }
            
        except Exception as e:
            logger.error(f"Error calculating Pearson correlation: {e}")
            return {
                'correlation': 0.0,
                'confidence': 0.0,
                'mathematical_accuracy': False,
                'p_value': 1.0
            }
    
    def _extract_price_series(self, data: pd.DataFrame) -> np.ndarray:
        """Extract price series from market data"""
        try:
            # Try multiple price columns
            price_columns = ['ce_ltp', 'pe_ltp', 'close', 'price', 'underlying_price']
            
            for col in price_columns:
                if col in data.columns:
                    prices = data[col].dropna()
                    if len(prices) > 0:
                        return prices.values
            
            # Fallback: calculate average of CE and PE prices
            if 'ce_ltp' in data.columns and 'pe_ltp' in data.columns:
                ce_prices = data['ce_ltp'].fillna(0)
                pe_prices = data['pe_ltp'].fillna(0)
                avg_prices = (ce_prices + pe_prices) / 2
                return avg_prices.values
            
            # Last resort: use index as proxy
            return np.arange(len(data))
            
        except Exception as e:
            logger.error(f"Error extracting price series: {e}")
            return np.array([])
    
    def _calculate_time_decay_weight(self, historical_data: pd.DataFrame) -> float:
        """Calculate time-decay weight using exponential decay formula"""
        try:
            # exp(-λ × (T-t)) mathematical formulation
            if 'timestamp' in historical_data.columns:
                timestamps = pd.to_datetime(historical_data['timestamp'])
                current_time = timestamps.max()
                time_diffs = (current_time - timestamps).dt.total_seconds() / 3600  # Hours
                
                # Apply exponential decay
                weights = np.exp(-self.decay_lambda * time_diffs)
                
                # Return weighted average of weights (more recent = higher weight)
                return np.mean(weights)
            else:
                # Fallback: assume uniform time spacing
                n_periods = min(len(historical_data), self.max_time_periods)
                time_position = 1.0  # Current time
                return np.exp(-self.decay_lambda * (self.max_time_periods - time_position))
                
        except Exception as e:
            logger.error(f"Error calculating time decay weight: {e}")
            return 1.0
    
    def _calculate_pattern_similarity(self, 
                                    current_data: pd.DataFrame,
                                    reference_patterns: List[Dict]) -> float:
        """Calculate pattern similarity using cosine similarity"""
        try:
            if not reference_patterns:
                return 0.0
            
            # Extract current pattern
            current_pattern = self._extract_pattern_vector(current_data)
            
            if len(current_pattern) < self.min_pattern_length:
                return 0.0
            
            max_similarity = 0.0
            
            # Compare with each reference pattern
            for ref_pattern_dict in reference_patterns:
                ref_pattern = ref_pattern_dict.get('pattern_vector', [])
                
                if len(ref_pattern) < self.min_pattern_length:
                    continue
                
                # Align pattern lengths
                min_length = min(len(current_pattern), len(ref_pattern))
                current_aligned = current_pattern[-min_length:]
                ref_aligned = ref_pattern[-min_length:]
                
                # Calculate cosine similarity
                try:
                    cosine_dist = cosine(current_aligned, ref_aligned)
                    similarity = 1.0 - cosine_dist  # Convert distance to similarity
                    
                    if not np.isnan(similarity):
                        max_similarity = max(max_similarity, similarity)
                        
                except Exception:
                    continue
            
            return max_similarity
            
        except Exception as e:
            logger.error(f"Error calculating pattern similarity: {e}")
            return 0.0
    
    def _extract_pattern_vector(self, data: pd.DataFrame) -> np.ndarray:
        """Extract pattern vector from market data"""
        try:
            # Combine multiple features into pattern vector
            features = []
            
            # Price features
            if 'ce_ltp' in data.columns:
                features.extend(data['ce_ltp'].fillna(0).values)
            if 'pe_ltp' in data.columns:
                features.extend(data['pe_ltp'].fillna(0).values)
            
            # Volume features
            if 'ce_volume' in data.columns:
                features.extend(data['ce_volume'].fillna(0).values)
            if 'pe_volume' in data.columns:
                features.extend(data['pe_volume'].fillna(0).values)
            
            # OI features
            if 'ce_oi' in data.columns:
                features.extend(data['ce_oi'].fillna(0).values)
            if 'pe_oi' in data.columns:
                features.extend(data['pe_oi'].fillna(0).values)
            
            if not features:
                return np.array([])
            
            # Normalize the pattern vector
            pattern_vector = np.array(features)
            if np.std(pattern_vector) > 0:
                pattern_vector = (pattern_vector - np.mean(pattern_vector)) / np.std(pattern_vector)
            
            return pattern_vector
            
        except Exception as e:
            logger.error(f"Error extracting pattern vector: {e}")
            return np.array([])
    
    def _find_historical_pattern_matches(self, 
                                       current_data: pd.DataFrame,
                                       historical_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Find historical patterns that match current pattern"""
        try:
            current_pattern = self._extract_pattern_vector(current_data)
            
            if len(current_pattern) < self.min_pattern_length:
                return None
            
            # Sliding window through historical data
            best_match = None
            best_similarity = 0.0
            
            window_size = len(current_pattern)
            historical_length = len(historical_data)
            
            for i in range(0, historical_length - window_size + 1, window_size // 2):
                window_data = historical_data.iloc[i:i + window_size]
                historical_pattern = self._extract_pattern_vector(window_data)
                
                if len(historical_pattern) != len(current_pattern):
                    continue
                
                # Calculate similarity
                try:
                    cosine_dist = cosine(current_pattern, historical_pattern)
                    similarity = 1.0 - cosine_dist
                    
                    if not np.isnan(similarity) and similarity > best_similarity:
                        best_similarity = similarity
                        best_match = {
                            'similarity_score': similarity,
                            'historical_window_start': i,
                            'historical_window_end': i + window_size,
                            'pattern_length': len(current_pattern),
                            'match_quality': 'high' if similarity > 0.9 else 'medium' if similarity > 0.7 else 'low'
                        }
                        
                except Exception:
                    continue
            
            # Return best match if it meets threshold
            if best_match and best_similarity >= self.similarity_threshold:
                return best_match
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding historical pattern matches: {e}")
            return None
    
    def _validate_mathematical_accuracy(self, 
                                      correlation: float,
                                      series1: np.ndarray,
                                      series2: np.ndarray) -> bool:
        """Validate mathematical accuracy within tolerance"""
        try:
            # Recalculate correlation manually for validation
            if len(series1) != len(series2) or len(series1) < 2:
                return False
            
            # Manual Pearson correlation calculation
            mean1 = np.mean(series1)
            mean2 = np.mean(series2)
            
            numerator = np.sum((series1 - mean1) * (series2 - mean2))
            denominator = np.sqrt(np.sum((series1 - mean1)**2) * np.sum((series2 - mean2)**2))
            
            if denominator == 0:
                manual_correlation = 0.0
            else:
                manual_correlation = numerator / denominator
            
            # Check if within mathematical tolerance
            accuracy = abs(correlation - manual_correlation) <= self.mathematical_tolerance
            
            return accuracy
            
        except Exception as e:
            logger.error(f"Error validating mathematical accuracy: {e}")
            return False
    
    def _record_correlation_analysis(self, correlation: float, confidence: float):
        """Record correlation analysis for tracking"""
        try:
            record = {
                'timestamp': datetime.now(),
                'correlation': correlation,
                'confidence': confidence,
                'threshold_met': abs(correlation) >= self.pearson_threshold
            }
            
            self.correlation_history.append(record)
            
            # Keep only recent history
            if len(self.correlation_history) > 100:
                self.correlation_history = self.correlation_history[-100:]
                
        except Exception as e:
            logger.error(f"Error recording correlation analysis: {e}")
    
    def _get_default_correlation_result(self) -> CorrelationAnalysisResult:
        """Get default result for error cases"""
        return CorrelationAnalysisResult(
            pearson_correlation=0.0,
            correlation_confidence=0.0,
            pattern_similarity_score=0.0,
            time_decay_weight=1.0,
            mathematical_accuracy=False,
            correlation_threshold_met=False,
            historical_pattern_match=None
        )
    
    def analyze_correlation_trends(self) -> Dict[str, Any]:
        """Analyze trends in correlation analysis"""
        try:
            if len(self.correlation_history) < 5:
                return {'status': 'insufficient_data'}
            
            recent_history = self.correlation_history[-20:]
            
            # Correlation trend analysis
            correlations = [h['correlation'] for h in recent_history]
            confidences = [h['confidence'] for h in recent_history]
            
            avg_correlation = np.mean(correlations)
            correlation_volatility = np.std(correlations)
            avg_confidence = np.mean(confidences)
            
            # Threshold achievement rate
            threshold_met_count = len([h for h in recent_history if h['threshold_met']])
            threshold_rate = threshold_met_count / len(recent_history)
            
            # Trend direction
            if len(correlations) > 1:
                trend = 'increasing' if correlations[-1] > correlations[0] else 'decreasing'
            else:
                trend = 'stable'
            
            return {
                'average_correlation': avg_correlation,
                'correlation_volatility': correlation_volatility,
                'average_confidence': avg_confidence,
                'threshold_achievement_rate': threshold_rate,
                'correlation_trend': trend,
                'analysis_period': len(recent_history),
                'mathematical_precision': {
                    'tolerance': self.mathematical_tolerance,
                    'pearson_threshold': self.pearson_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing correlation trends: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_correlation_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of correlation analysis performance"""
        try:
            summary = {
                'total_analyses': len(self.correlation_history),
                'configuration': {
                    'mathematical_tolerance': self.mathematical_tolerance,
                    'pearson_threshold': self.pearson_threshold,
                    'decay_lambda': self.decay_lambda,
                    'similarity_threshold': self.similarity_threshold
                },
                'analysis_features': {
                    'time_decay_weighting': True,
                    'pattern_similarity': True,
                    'historical_matching': self.enable_historical_matching,
                    'mathematical_validation': True
                }
            }
            
            if self.correlation_history:
                recent_performance = self.analyze_correlation_trends()
                summary['recent_performance'] = recent_performance
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating correlation analysis summary: {e}")
            return {'status': 'error', 'error': str(e)}