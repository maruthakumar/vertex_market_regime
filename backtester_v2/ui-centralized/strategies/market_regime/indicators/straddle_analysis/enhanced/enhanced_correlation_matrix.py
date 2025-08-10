"""
Enhanced 10×10 Correlation Matrix for All Straddle Components

Manages rolling correlation analysis between all 10 option components:
- 6 Individual: ATM_CE, ATM_PE, ITM1_CE, ITM1_PE, OTM1_CE, OTM1_PE
- 3 Individual Straddles: ATM_STRADDLE, ITM1_STRADDLE, OTM1_STRADDLE
- 1 Combined Triple: COMBINED_TRIPLE_STRADDLE

Provides comprehensive correlation tracking across multiple timeframes (3,5,10,15 min)
for enhanced market regime formation and pattern recognition.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
from numba import jit, prange
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class EnhancedCorrelationResult:
    """Enhanced correlation analysis result for 10 components"""
    matrix: np.ndarray  # 10x10 correlation matrix
    component_names: List[str]  # All 10 component names
    timestamp: pd.Timestamp
    timeframe: int  # minutes
    
    # Correlation statistics
    avg_correlation: float
    max_correlation: float
    min_correlation: float
    correlation_variance: float
    
    # Multi-timeframe analysis
    timeframe_consistency: float
    correlation_trends: Dict[str, float]
    
    # Pattern-relevant correlations
    straddle_correlations: Dict[str, float]
    cross_component_correlations: Dict[str, float]
    regime_indicators: Dict[str, float]


@dataclass
class CorrelationPattern:
    """Detected correlation pattern"""
    pattern_id: str
    pattern_type: str  # convergence, divergence, stability, volatility
    components_involved: List[str]
    timeframes_affected: List[int]
    strength: float
    duration: int  # minutes
    confidence: float


class Enhanced10x10CorrelationMatrix:
    """
    Enhanced 10×10 Rolling Correlation Matrix Manager
    
    Tracks correlations between all 10 option components across multiple
    timeframes for comprehensive market regime analysis and pattern detection.
    
    Component Matrix (10×10):
    - Individual (6): ATM_CE, ATM_PE, ITM1_CE, ITM1_PE, OTM1_CE, OTM1_PE
    - Straddles (3): ATM_STRADDLE, ITM1_STRADDLE, OTM1_STRADDLE
    - Combined (1): COMBINED_TRIPLE_STRADDLE
    - 45 unique correlation pairs (10×10 symmetric matrix)
    - Multi-timeframe analysis across [3,5,10,15] minute windows
    """
    
    def __init__(self, config: Dict[str, Any], window_manager=None):
        """
        Initialize enhanced 10x10 correlation matrix manager
        
        Args:
            config: Configuration dictionary
            window_manager: Rolling window manager for data access
        """
        self.config = config
        self.window_manager = window_manager
        self.logger = logging.getLogger(f"{__name__}.Enhanced10x10CorrelationMatrix")
        
        # All 10 component definitions
        self.all_components = [
            'ATM_CE', 'ATM_PE', 'ITM1_CE', 'ITM1_PE', 'OTM1_CE', 'OTM1_PE',  # Individual (6)
            'ATM_STRADDLE', 'ITM1_STRADDLE', 'OTM1_STRADDLE',  # Individual Straddles (3)
            'COMBINED_TRIPLE_STRADDLE'  # Combined Triple (1)
        ]
        
        # Component categories for analysis
        self.individual_components = self.all_components[:6]
        self.straddle_components = self.all_components[6:9]
        self.combined_component = self.all_components[9:]
        
        self.n_components = len(self.all_components)  # 10 components
        
        # Multi-timeframe windows
        self.timeframes = config.get('rolling_windows', [3, 5, 10, 15])
        
        # Enhanced correlation thresholds for pattern detection
        self.correlation_thresholds = config.get('correlation_thresholds', {
            'very_high_correlation': 0.9,
            'high_correlation': 0.8,
            'medium_correlation': 0.5,
            'low_correlation': 0.2,
            'very_low_correlation': 0.1,
            'divergence_threshold': -0.3,
            'pattern_detection_threshold': 0.85
        })
        
        # Storage for correlation matrices by timeframe
        self.correlation_matrices = {}
        self.correlation_history = {}
        self.pattern_history = {}
        
        # Initialize storage for each timeframe
        for timeframe in self.timeframes:
            self.correlation_matrices[timeframe] = np.eye(self.n_components)
            self.correlation_history[timeframe] = []
            self.pattern_history[timeframe] = []
        
        # Component pair mapping for efficient access (45 unique pairs)
        self.component_pairs = []
        self.pair_indices = {}
        
        for i in range(self.n_components):
            for j in range(i + 1, self.n_components):
                pair_name = f"{self.all_components[i]}_{self.all_components[j]}"
                self.component_pairs.append((self.all_components[i], self.all_components[j]))
                self.pair_indices[pair_name] = (i, j)
        
        # Pattern detection settings
        self.pattern_detection_enabled = config.get('pattern_detection', True)
        self.min_pattern_duration = config.get('min_pattern_duration', 5)  # minutes
        self.pattern_confidence_threshold = config.get('pattern_confidence_threshold', 0.75)
        
        # Performance optimization
        self.calculation_cache = {}
        self.cache_duration = 60  # seconds
        
        self.logger.info(f"Enhanced 10×10 Correlation Matrix initialized")
        self.logger.info(f"Components: {self.all_components}")
        self.logger.info(f"Tracking {len(self.component_pairs)} unique correlation pairs")
        self.logger.info(f"Timeframes: {self.timeframes}")
    
    @staticmethod
    @jit(nopython=True)
    def _fast_correlation_matrix_10x10(data_matrix: np.ndarray) -> np.ndarray:
        """
        Ultra-fast 10x10 correlation matrix calculation using Numba
        
        Args:
            data_matrix: 2D array with shape (n_observations, 10_components)
            
        Returns:
            10x10 correlation matrix
        """
        n_obs, n_vars = data_matrix.shape
        corr_matrix = np.eye(n_vars)
        
        for i in prange(n_vars):
            for j in prange(i + 1, n_vars):
                # Calculate correlation between variables i and j
                x = data_matrix[:, i]
                y = data_matrix[:, j]
                
                # Remove NaN values
                valid_mask = ~(np.isnan(x) | np.isnan(y))
                if np.sum(valid_mask) < 3:
                    corr_matrix[i, j] = 0.0
                    corr_matrix[j, i] = 0.0
                    continue
                
                x_valid = x[valid_mask]
                y_valid = y[valid_mask]
                
                # Calculate means
                mean_x = np.mean(x_valid)
                mean_y = np.mean(y_valid)
                
                # Calculate correlation
                numerator = np.sum((x_valid - mean_x) * (y_valid - mean_y))
                
                sum_sq_x = np.sum((x_valid - mean_x) ** 2)
                sum_sq_y = np.sum((y_valid - mean_y) ** 2)
                
                if sum_sq_x == 0 or sum_sq_y == 0:
                    correlation = 0.0
                else:
                    correlation = numerator / np.sqrt(sum_sq_x * sum_sq_y)
                
                corr_matrix[i, j] = correlation
                corr_matrix[j, i] = correlation
        
        return corr_matrix
    
    def analyze_all_timeframes(self, timestamp: pd.Timestamp) -> Dict[int, EnhancedCorrelationResult]:
        """
        Analyze correlations across all timeframes
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            Dictionary mapping timeframe to correlation results
        """
        results = {}
        
        for timeframe in self.timeframes:
            try:
                result = self.analyze_timeframe(timeframe, timestamp)
                if result:
                    results[timeframe] = result
                    
                    # Store in history
                    self.correlation_history[timeframe].append(result)
                    
                    # Maintain history size (keep last 1000 results)
                    if len(self.correlation_history[timeframe]) > 1000:
                        self.correlation_history[timeframe] = self.correlation_history[timeframe][-1000:]
                    
            except Exception as e:
                self.logger.warning(f"Error analyzing timeframe {timeframe}: {e}")
                continue
        
        return results
    
    def analyze_timeframe(self, timeframe: int, timestamp: pd.Timestamp) -> Optional[EnhancedCorrelationResult]:
        """
        Analyze correlations for a specific timeframe
        
        Args:
            timeframe: Timeframe in minutes
            timestamp: Current timestamp
            
        Returns:
            EnhancedCorrelationResult or None if insufficient data
        """
        try:
            # Get data for all components
            component_data = self._get_component_data_matrix(timeframe)
            
            if component_data is None or component_data.shape[0] < 10:
                return None
            
            # Calculate 10x10 correlation matrix
            correlation_matrix = self._fast_correlation_matrix_10x10(component_data)
            
            # Calculate correlation statistics
            correlation_stats = self._calculate_correlation_statistics(correlation_matrix)
            
            # Calculate timeframe consistency
            timeframe_consistency = self._calculate_timeframe_consistency(timeframe, correlation_matrix)
            
            # Calculate correlation trends
            correlation_trends = self._calculate_correlation_trends(timeframe, correlation_matrix)
            
            # Calculate straddle-specific correlations
            straddle_correlations = self._calculate_straddle_correlations(correlation_matrix)
            
            # Calculate cross-component correlations
            cross_component_correlations = self._calculate_cross_component_correlations(correlation_matrix)
            
            # Calculate regime indicators
            regime_indicators = self._calculate_regime_indicators(correlation_matrix, timeframe)
            
            # Create result
            result = EnhancedCorrelationResult(
                matrix=correlation_matrix,
                component_names=self.all_components,
                timestamp=timestamp,
                timeframe=timeframe,
                avg_correlation=correlation_stats['avg'],
                max_correlation=correlation_stats['max'],
                min_correlation=correlation_stats['min'],
                correlation_variance=correlation_stats['variance'],
                timeframe_consistency=timeframe_consistency,
                correlation_trends=correlation_trends,
                straddle_correlations=straddle_correlations,
                cross_component_correlations=cross_component_correlations,
                regime_indicators=regime_indicators
            )
            
            # Update stored matrix
            self.correlation_matrices[timeframe] = correlation_matrix
            
            # Detect correlation patterns if enabled
            if self.pattern_detection_enabled:
                patterns = self._detect_correlation_patterns(result)
                if patterns:
                    self.pattern_history[timeframe].extend(patterns)
                    self.logger.debug(f"Detected {len(patterns)} correlation patterns in {timeframe}min timeframe")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing timeframe {timeframe}: {e}")
            return None
    
    def _get_component_data_matrix(self, timeframe: int) -> Optional[np.ndarray]:
        """Get data matrix for all 10 components for the specified timeframe"""
        if not self.window_manager:
            return None
        
        try:
            data_matrix = []
            
            # Get data for individual components (6)
            for component in self.individual_components:
                component_data, _ = self.window_manager.get_window_data(
                    component.lower(), timeframe
                )
                
                if component_data:
                    prices = [dp.get('close', dp.get('price', 0)) for dp in component_data]
                    data_matrix.append(prices)
                else:
                    # Fill with zeros if no data
                    data_matrix.append([0.0] * 50)  # Default length
            
            # Calculate straddle data (3)
            if len(data_matrix) >= 6:
                # ATM Straddle = ATM_CE + ATM_PE
                atm_straddle = [data_matrix[0][i] + data_matrix[1][i] for i in range(len(data_matrix[0]))]
                data_matrix.append(atm_straddle)
                
                # ITM1 Straddle = ITM1_CE + ITM1_PE
                itm1_straddle = [data_matrix[2][i] + data_matrix[3][i] for i in range(len(data_matrix[0]))]
                data_matrix.append(itm1_straddle)
                
                # OTM1 Straddle = OTM1_CE + OTM1_PE
                otm1_straddle = [data_matrix[4][i] + data_matrix[5][i] for i in range(len(data_matrix[0]))]
                data_matrix.append(otm1_straddle)
                
                # Combined Triple Straddle = ATM + ITM1 + OTM1 Straddles
                combined_triple = [
                    atm_straddle[i] + itm1_straddle[i] + otm1_straddle[i] 
                    for i in range(len(atm_straddle))
                ]
                data_matrix.append(combined_triple)
            
            # Ensure we have 10 components
            while len(data_matrix) < 10:
                data_matrix.append([0.0] * (len(data_matrix[0]) if data_matrix else 50))
            
            # Convert to numpy array and transpose (observations x components)
            data_array = np.array(data_matrix).T
            
            # Remove rows with all zeros or NaN
            valid_rows = ~np.all((data_array == 0) | np.isnan(data_array), axis=1)
            data_array = data_array[valid_rows]
            
            return data_array if data_array.shape[0] > 0 else None
            
        except Exception as e:
            self.logger.warning(f"Error creating data matrix for {timeframe}min: {e}")
            return None
    
    def _calculate_correlation_statistics(self, correlation_matrix: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive correlation statistics"""
        # Extract upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
        correlations = correlation_matrix[mask]
        
        # Remove NaN values
        valid_correlations = correlations[~np.isnan(correlations)]
        
        if len(valid_correlations) == 0:
            return {
                'avg': 0.0,
                'max': 0.0,
                'min': 0.0,
                'variance': 0.0,
                'std': 0.0,
                'skewness': 0.0
            }
        
        return {
            'avg': np.mean(valid_correlations),
            'max': np.max(valid_correlations),
            'min': np.min(valid_correlations),
            'variance': np.var(valid_correlations),
            'std': np.std(valid_correlations),
            'skewness': self._calculate_skewness(valid_correlations)
        }
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of correlation distribution"""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_timeframe_consistency(self, timeframe: int, correlation_matrix: np.ndarray) -> float:
        """Calculate consistency of correlations with other timeframes"""
        if timeframe not in self.correlation_matrices:
            return 0.0
        
        consistency_scores = []
        
        for other_timeframe in self.timeframes:
            if other_timeframe == timeframe:
                continue
            
            if other_timeframe in self.correlation_matrices:
                other_matrix = self.correlation_matrices[other_timeframe]
                
                # Calculate correlation between correlation matrices
                mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
                
                corr1 = correlation_matrix[mask]
                corr2 = other_matrix[mask]
                
                # Remove NaN values
                valid_mask = ~(np.isnan(corr1) | np.isnan(corr2))
                if np.sum(valid_mask) > 5:
                    consistency = np.corrcoef(corr1[valid_mask], corr2[valid_mask])[0, 1]
                    if not np.isnan(consistency):
                        consistency_scores.append(abs(consistency))
        
        return np.mean(consistency_scores) if consistency_scores else 0.0
    
    def _calculate_correlation_trends(self, timeframe: int, correlation_matrix: np.ndarray) -> Dict[str, float]:
        """Calculate trends in correlations over time"""
        trends = {}
        
        if len(self.correlation_history[timeframe]) < 5:
            return trends
        
        # Get recent correlation matrices
        recent_results = self.correlation_history[timeframe][-5:]
        
        # Calculate trends for key correlation pairs
        important_pairs = [
            ('ATM_CE', 'ATM_PE'),
            ('ITM1_CE', 'ITM1_PE'),
            ('OTM1_CE', 'OTM1_PE'),
            ('ATM_STRADDLE', 'ITM1_STRADDLE'),
            ('ATM_STRADDLE', 'COMBINED_TRIPLE_STRADDLE')
        ]
        
        for comp1, comp2 in important_pairs:
            try:
                i = self.all_components.index(comp1)
                j = self.all_components.index(comp2)
                
                correlations = [result.matrix[i, j] for result in recent_results]
                
                # Calculate trend (simple linear trend)
                if len(correlations) >= 3:
                    x = np.arange(len(correlations))
                    trend = np.polyfit(x, correlations, 1)[0]  # Slope
                    trends[f"{comp1}_{comp2}_trend"] = trend
                    
            except (ValueError, IndexError):
                continue
        
        return trends
    
    def _calculate_straddle_correlations(self, correlation_matrix: np.ndarray) -> Dict[str, float]:
        """Calculate straddle-specific correlations"""
        straddle_corr = {}
        
        try:
            # Individual component correlations within straddles
            straddle_corr['ATM_CE_PE'] = correlation_matrix[0, 1]  # ATM_CE vs ATM_PE
            straddle_corr['ITM1_CE_PE'] = correlation_matrix[2, 3]  # ITM1_CE vs ITM1_PE
            straddle_corr['OTM1_CE_PE'] = correlation_matrix[4, 5]  # OTM1_CE vs OTM1_PE
            
            # Straddle-to-straddle correlations
            straddle_corr['ATM_ITM1_Straddle'] = correlation_matrix[6, 7]  # ATM_STRADDLE vs ITM1_STRADDLE
            straddle_corr['ATM_OTM1_Straddle'] = correlation_matrix[6, 8]  # ATM_STRADDLE vs OTM1_STRADDLE
            straddle_corr['ITM1_OTM1_Straddle'] = correlation_matrix[7, 8]  # ITM1_STRADDLE vs OTM1_STRADDLE
            
            # Combined triple correlations
            straddle_corr['ATM_vs_Combined'] = correlation_matrix[6, 9]  # ATM_STRADDLE vs COMBINED_TRIPLE
            straddle_corr['ITM1_vs_Combined'] = correlation_matrix[7, 9]  # ITM1_STRADDLE vs COMBINED_TRIPLE
            straddle_corr['OTM1_vs_Combined'] = correlation_matrix[8, 9]  # OTM1_STRADDLE vs COMBINED_TRIPLE
            
        except IndexError as e:
            self.logger.warning(f"Error calculating straddle correlations: {e}")
        
        return straddle_corr
    
    def _calculate_cross_component_correlations(self, correlation_matrix: np.ndarray) -> Dict[str, float]:
        """Calculate cross-component correlations for pattern detection"""
        cross_corr = {}
        
        try:
            # Cross-strike correlations (CE vs CE, PE vs PE)
            cross_corr['ATM_CE_vs_ITM1_CE'] = correlation_matrix[0, 2]
            cross_corr['ATM_CE_vs_OTM1_CE'] = correlation_matrix[0, 4]
            cross_corr['ITM1_CE_vs_OTM1_CE'] = correlation_matrix[2, 4]
            
            cross_corr['ATM_PE_vs_ITM1_PE'] = correlation_matrix[1, 3]
            cross_corr['ATM_PE_vs_OTM1_PE'] = correlation_matrix[1, 5]
            cross_corr['ITM1_PE_vs_OTM1_PE'] = correlation_matrix[3, 5]
            
            # Cross-type correlations (CE vs PE across strikes)
            cross_corr['ATM_CE_vs_ITM1_PE'] = correlation_matrix[0, 3]
            cross_corr['ATM_CE_vs_OTM1_PE'] = correlation_matrix[0, 5]
            cross_corr['ITM1_CE_vs_ATM_PE'] = correlation_matrix[2, 1]
            cross_corr['ITM1_CE_vs_OTM1_PE'] = correlation_matrix[2, 5]
            cross_corr['OTM1_CE_vs_ATM_PE'] = correlation_matrix[4, 1]
            cross_corr['OTM1_CE_vs_ITM1_PE'] = correlation_matrix[4, 3]
            
        except IndexError as e:
            self.logger.warning(f"Error calculating cross-component correlations: {e}")
        
        return cross_corr
    
    def _calculate_regime_indicators(self, correlation_matrix: np.ndarray, timeframe: int) -> Dict[str, float]:
        """Calculate regime indicators based on correlation patterns"""
        regime_indicators = {}
        
        try:
            # Calculate average correlations by category
            straddle_correlations = self._calculate_straddle_correlations(correlation_matrix)
            cross_correlations = self._calculate_cross_component_correlations(correlation_matrix)
            
            # Regime indicators
            regime_indicators['straddle_coherence'] = np.mean([
                abs(straddle_correlations.get('ATM_CE_PE', 0)),
                abs(straddle_correlations.get('ITM1_CE_PE', 0)),
                abs(straddle_correlations.get('OTM1_CE_PE', 0))
            ])
            
            regime_indicators['cross_strike_consistency'] = np.mean([
                abs(cross_correlations.get('ATM_CE_vs_ITM1_CE', 0)),
                abs(cross_correlations.get('ATM_PE_vs_ITM1_PE', 0)),
                abs(cross_correlations.get('ITM1_CE_vs_OTM1_CE', 0)),
                abs(cross_correlations.get('ITM1_PE_vs_OTM1_PE', 0))
            ])
            
            regime_indicators['market_structure'] = (
                regime_indicators['straddle_coherence'] * 0.6 + 
                regime_indicators['cross_strike_consistency'] * 0.4
            )
            
            # Volatility regime (based on correlation variance)
            correlation_stats = self._calculate_correlation_statistics(correlation_matrix)
            regime_indicators['volatility_regime'] = min(1.0, correlation_stats['variance'] * 10)
            
            # Trend regime (based on straddle correlations)
            avg_straddle_corr = np.mean(list(straddle_correlations.values()))
            regime_indicators['trend_regime'] = abs(avg_straddle_corr)
            
            # Combined regime score
            regime_indicators['overall_regime_strength'] = np.mean([
                regime_indicators['market_structure'],
                regime_indicators['volatility_regime'],
                regime_indicators['trend_regime']
            ])
            
        except Exception as e:
            self.logger.warning(f"Error calculating regime indicators: {e}")
        
        return regime_indicators
    
    def _detect_correlation_patterns(self, result: EnhancedCorrelationResult) -> List[CorrelationPattern]:
        """Detect correlation patterns for trading opportunities"""
        patterns = []
        
        try:
            # Pattern 1: Correlation Convergence
            convergence_patterns = self._detect_convergence_patterns(result)
            patterns.extend(convergence_patterns)
            
            # Pattern 2: Correlation Divergence
            divergence_patterns = self._detect_divergence_patterns(result)
            patterns.extend(divergence_patterns)
            
            # Pattern 3: Correlation Stability
            stability_patterns = self._detect_stability_patterns(result)
            patterns.extend(stability_patterns)
            
            # Pattern 4: Correlation Volatility
            volatility_patterns = self._detect_volatility_patterns(result)
            patterns.extend(volatility_patterns)
            
        except Exception as e:
            self.logger.warning(f"Error detecting correlation patterns: {e}")
        
        return patterns
    
    def _detect_convergence_patterns(self, result: EnhancedCorrelationResult) -> List[CorrelationPattern]:
        """Detect correlation convergence patterns"""
        patterns = []
        
        # Check for high correlation emergence
        for i, comp1 in enumerate(self.all_components):
            for j, comp2 in enumerate(self.all_components[i+1:], i+1):
                correlation = result.matrix[i, j]
                
                if abs(correlation) > self.correlation_thresholds['pattern_detection_threshold']:
                    # Check if this is a new high correlation
                    historical_corr = self._get_historical_correlation(comp1, comp2, result.timeframe)
                    
                    if historical_corr and abs(correlation) > abs(historical_corr) + 0.1:
                        patterns.append(CorrelationPattern(
                            pattern_id=f"CONV_{comp1}_{comp2}_{result.timeframe}_{int(result.timestamp.timestamp())}",
                            pattern_type="convergence",
                            components_involved=[comp1, comp2],
                            timeframes_affected=[result.timeframe],
                            strength=abs(correlation),
                            duration=self._estimate_pattern_duration(comp1, comp2, result.timeframe),
                            confidence=min(0.9, abs(correlation) + 0.1)
                        ))
        
        return patterns
    
    def _detect_divergence_patterns(self, result: EnhancedCorrelationResult) -> List[CorrelationPattern]:
        """Detect correlation divergence patterns"""
        patterns = []
        
        # Check for correlation breakdown
        for i, comp1 in enumerate(self.all_components):
            for j, comp2 in enumerate(self.all_components[i+1:], i+1):
                correlation = result.matrix[i, j]
                
                if abs(correlation) < self.correlation_thresholds['low_correlation']:
                    # Check if this represents a divergence from historical high correlation
                    historical_corr = self._get_historical_correlation(comp1, comp2, result.timeframe)
                    
                    if historical_corr and abs(historical_corr) > 0.7 and abs(correlation) < 0.3:
                        patterns.append(CorrelationPattern(
                            pattern_id=f"DIV_{comp1}_{comp2}_{result.timeframe}_{int(result.timestamp.timestamp())}",
                            pattern_type="divergence",
                            components_involved=[comp1, comp2],
                            timeframes_affected=[result.timeframe],
                            strength=abs(historical_corr - correlation),
                            duration=self._estimate_pattern_duration(comp1, comp2, result.timeframe),
                            confidence=min(0.9, abs(historical_corr - correlation))
                        ))
        
        return patterns
    
    def _detect_stability_patterns(self, result: EnhancedCorrelationResult) -> List[CorrelationPattern]:
        """Detect correlation stability patterns"""
        patterns = []
        
        # Check for sustained high correlations
        if result.correlation_variance < 0.01:  # Low variance indicates stability
            stable_pairs = []
            
            for i, comp1 in enumerate(self.all_components):
                for j, comp2 in enumerate(self.all_components[i+1:], i+1):
                    correlation = result.matrix[i, j]
                    
                    if abs(correlation) > self.correlation_thresholds['high_correlation']:
                        stable_pairs.append((comp1, comp2))
            
            if len(stable_pairs) >= 3:  # Multiple stable correlations
                patterns.append(CorrelationPattern(
                    pattern_id=f"STAB_MULTI_{result.timeframe}_{int(result.timestamp.timestamp())}",
                    pattern_type="stability",
                    components_involved=[comp for pair in stable_pairs for comp in pair],
                    timeframes_affected=[result.timeframe],
                    strength=1.0 - result.correlation_variance,
                    duration=self.min_pattern_duration * 2,
                    confidence=0.8
                ))
        
        return patterns
    
    def _detect_volatility_patterns(self, result: EnhancedCorrelationResult) -> List[CorrelationPattern]:
        """Detect correlation volatility patterns"""
        patterns = []
        
        # Check for high correlation variance (unstable correlations)
        if result.correlation_variance > 0.1:
            volatile_components = []
            
            # Find components with most volatile correlations
            for i, comp in enumerate(self.all_components):
                comp_correlations = np.concatenate([
                    result.matrix[i, :i],
                    result.matrix[i, i+1:]
                ])
                comp_variance = np.var(comp_correlations)
                
                if comp_variance > 0.05:
                    volatile_components.append(comp)
            
            if len(volatile_components) >= 2:
                patterns.append(CorrelationPattern(
                    pattern_id=f"VOL_{result.timeframe}_{int(result.timestamp.timestamp())}",
                    pattern_type="volatility",
                    components_involved=volatile_components,
                    timeframes_affected=[result.timeframe],
                    strength=result.correlation_variance,
                    duration=self.min_pattern_duration,
                    confidence=min(0.9, result.correlation_variance * 5)
                ))
        
        return patterns
    
    def _get_historical_correlation(self, comp1: str, comp2: str, timeframe: int) -> Optional[float]:
        """Get historical correlation for comparison"""
        if timeframe not in self.correlation_history:
            return None
        
        if len(self.correlation_history[timeframe]) < 2:
            return None
        
        try:
            i = self.all_components.index(comp1)
            j = self.all_components.index(comp2)
            
            # Get average correlation from last 5 periods
            recent_results = self.correlation_history[timeframe][-5:-1]
            correlations = [result.matrix[i, j] for result in recent_results]
            
            return np.mean(correlations) if correlations else None
            
        except (ValueError, IndexError):
            return None
    
    def _estimate_pattern_duration(self, comp1: str, comp2: str, timeframe: int) -> int:
        """Estimate pattern duration based on historical data"""
        # Simple estimation - can be enhanced with ML
        base_duration = self.min_pattern_duration
        
        # Adjust based on component types
        if 'STRADDLE' in comp1 or 'STRADDLE' in comp2:
            base_duration *= 2  # Straddle patterns tend to last longer
        
        if 'COMBINED' in comp1 or 'COMBINED' in comp2:
            base_duration *= 3  # Combined patterns are more persistent
        
        # Adjust based on timeframe
        timeframe_multiplier = timeframe / 5.0  # Normalize to 5-minute base
        
        return int(base_duration * timeframe_multiplier)
    
    def get_correlation_summary(self, timeframes: Optional[List[int]] = None) -> Dict[str, Any]:
        """Get comprehensive correlation summary"""
        if timeframes is None:
            timeframes = self.timeframes
        
        summary = {
            'timeframe_summaries': {},
            'cross_timeframe_analysis': {},
            'pattern_summary': {},
            'regime_summary': {}
        }
        
        for timeframe in timeframes:
            if timeframe not in self.correlation_history:
                continue
            
            if not self.correlation_history[timeframe]:
                continue
            
            latest_result = self.correlation_history[timeframe][-1]
            
            summary['timeframe_summaries'][timeframe] = {
                'avg_correlation': latest_result.avg_correlation,
                'max_correlation': latest_result.max_correlation,
                'min_correlation': latest_result.min_correlation,
                'correlation_variance': latest_result.correlation_variance,
                'timeframe_consistency': latest_result.timeframe_consistency,
                'regime_strength': latest_result.regime_indicators.get('overall_regime_strength', 0)
            }
            
            # Pattern summary for this timeframe
            if timeframe in self.pattern_history:
                recent_patterns = [p for p in self.pattern_history[timeframe] 
                                 if p.confidence > self.pattern_confidence_threshold]
                
                summary['pattern_summary'][timeframe] = {
                    'total_patterns': len(self.pattern_history[timeframe]),
                    'high_confidence_patterns': len(recent_patterns),
                    'pattern_types': {}
                }
                
                # Count by pattern type
                for pattern in recent_patterns:
                    pattern_type = pattern.pattern_type
                    if pattern_type not in summary['pattern_summary'][timeframe]['pattern_types']:
                        summary['pattern_summary'][timeframe]['pattern_types'][pattern_type] = 0
                    summary['pattern_summary'][timeframe]['pattern_types'][pattern_type] += 1
        
        # Cross-timeframe consistency analysis
        if len(summary['timeframe_summaries']) > 1:
            timeframe_correlations = [
                data['avg_correlation'] for data in summary['timeframe_summaries'].values()
            ]
            
            summary['cross_timeframe_analysis'] = {
                'consistency_score': 1.0 - np.std(timeframe_correlations),
                'correlation_range': max(timeframe_correlations) - min(timeframe_correlations),
                'trending_correlation': np.mean(timeframe_correlations)
            }
        
        return summary
    
    def get_component_correlation_profile(self, component: str) -> Dict[str, Any]:
        """Get detailed correlation profile for a specific component"""
        if component not in self.all_components:
            return {}
        
        component_index = self.all_components.index(component)
        profile = {
            'component': component,
            'correlations_by_timeframe': {},
            'strongest_correlations': {},
            'weakest_correlations': {},
            'correlation_stability': {}
        }
        
        for timeframe in self.timeframes:
            if timeframe not in self.correlation_matrices:
                continue
            
            matrix = self.correlation_matrices[timeframe]
            component_correlations = {}
            
            for i, other_component in enumerate(self.all_components):
                if i != component_index:
                    correlation = matrix[component_index, i]
                    component_correlations[other_component] = correlation
            
            profile['correlations_by_timeframe'][timeframe] = component_correlations
            
            # Find strongest and weakest correlations
            sorted_correlations = sorted(
                component_correlations.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )
            
            profile['strongest_correlations'][timeframe] = sorted_correlations[:3]
            profile['weakest_correlations'][timeframe] = sorted_correlations[-3:]
        
        return profile
    
    def export_correlation_data(self, filepath: str, timeframes: Optional[List[int]] = None):
        """Export correlation data to file"""
        if timeframes is None:
            timeframes = self.timeframes
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'components': self.all_components,
            'timeframes': timeframes,
            'correlation_matrices': {},
            'correlation_history': {},
            'pattern_history': {}
        }
        
        for timeframe in timeframes:
            if timeframe in self.correlation_matrices:
                export_data['correlation_matrices'][timeframe] = self.correlation_matrices[timeframe].tolist()
            
            if timeframe in self.correlation_history:
                # Export last 100 results
                recent_history = self.correlation_history[timeframe][-100:]
                export_data['correlation_history'][timeframe] = [
                    {
                        'timestamp': result.timestamp.isoformat(),
                        'avg_correlation': result.avg_correlation,
                        'max_correlation': result.max_correlation,
                        'min_correlation': result.min_correlation,
                        'correlation_variance': result.correlation_variance,
                        'regime_indicators': result.regime_indicators
                    }
                    for result in recent_history
                ]
            
            if timeframe in self.pattern_history:
                export_data['pattern_history'][timeframe] = [
                    {
                        'pattern_id': pattern.pattern_id,
                        'pattern_type': pattern.pattern_type,
                        'components_involved': pattern.components_involved,
                        'strength': pattern.strength,
                        'confidence': pattern.confidence
                    }
                    for pattern in self.pattern_history[timeframe][-50:]  # Last 50 patterns
                ]
        
        # Save to file
        import json
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Correlation data exported to {filepath}")
    
    def clear_old_data(self, hours_old: int = 24):
        """Clear old correlation data to manage memory"""
        cutoff_time = pd.Timestamp.now() - pd.Timedelta(hours=hours_old)
        
        for timeframe in self.timeframes:
            if timeframe in self.correlation_history:
                # Keep only recent data
                self.correlation_history[timeframe] = [
                    result for result in self.correlation_history[timeframe]
                    if result.timestamp > cutoff_time
                ]
            
            if timeframe in self.pattern_history:
                # Keep only recent patterns
                recent_patterns = []
                for pattern in self.pattern_history[timeframe]:
                    # Patterns don't have timestamps, so use a different criteria
                    if len(recent_patterns) < 200:  # Keep last 200 patterns
                        recent_patterns.append(pattern)
                
                self.pattern_history[timeframe] = recent_patterns[-200:]
        
        self.logger.info(f"Cleared correlation data older than {hours_old} hours")