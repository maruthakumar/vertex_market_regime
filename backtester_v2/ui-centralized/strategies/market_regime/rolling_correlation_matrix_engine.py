#!/usr/bin/env python3
"""
Rolling Correlation Matrix Engine - 6×6 Component Correlation Analysis
Part of Comprehensive Triple Straddle Engine V2.0

This engine provides comprehensive 6×6 rolling correlation matrix analysis:
- Component-to-Component correlations (ATM/ITM1/OTM1/Combined/ATM_CE/ATM_PE)
- Indicator-to-Indicator correlations (EMA/VWAP/Pivot across components)
- Timeframe-to-Timeframe correlations (3min/5min/10min/15min)
- Dynamic correlation thresholds: High (>0.8), Medium (0.4-0.8), Low (0.1-0.4), Negative (<-0.1)
- Correlation-based regime strength determination
- Real-time correlation monitoring and alerts

Author: The Augster
Date: 2025-06-23
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import logging
from itertools import combinations

logger = logging.getLogger(__name__)

class RollingCorrelationMatrixEngine:
    """
    Rolling Correlation Matrix Engine for comprehensive 6×6 analysis
    
    Provides real-time rolling correlation analysis across all components,
    indicators, and timeframes with dynamic threshold classification
    and regime strength determination.
    """
    
    def __init__(self, correlation_windows: List[int] = [20, 50, 100], 
                 components: List[str] = None):
        """Initialize Rolling Correlation Matrix Engine"""
        self.correlation_windows = correlation_windows
        self.components = components or [
            'atm_straddle', 'itm1_straddle', 'otm1_straddle', 
            'combined_straddle', 'atm_ce', 'atm_pe'
        ]
        
        # Correlation thresholds as per documentation
        self.correlation_thresholds = {
            'high': 0.8,        # High correlation threshold
            'medium_high': 0.4, # Medium correlation threshold
            'low': 0.1,         # Low correlation threshold
            'negative': -0.1    # Negative correlation threshold
        }
        
        # Technical indicators to analyze
        self.technical_indicators = [
            'ema_20', 'ema_100', 'ema_200',
            'vwap_current', 'vwap_previous',
            'pivot_current', 'pivot_previous'
        ]
        
        # Timeframes for analysis
        self.timeframes = ['3min', '5min', '10min', '15min']
        
        logger.info("Rolling Correlation Matrix Engine initialized")
        logger.info(f"Components: {len(self.components)}")
        logger.info(f"Correlation windows: {self.correlation_windows}")
        logger.info(f"Thresholds - High: {self.correlation_thresholds['high']}, "
                   f"Medium: {self.correlation_thresholds['medium_high']}, "
                   f"Low: {self.correlation_thresholds['low']}")
    
    def calculate_real_time_correlations(self, technical_results: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Calculate comprehensive real-time rolling correlations
        
        Args:
            technical_results: Technical analysis results from all components
            
        Returns:
            Complete correlation analysis with 6×6 matrix and regime insights
        """
        try:
            logger.debug("Calculating real-time rolling correlations")
            
            # Extract price series for each component
            component_prices = self._extract_component_price_series(technical_results)
            
            # Calculate component-to-component correlations
            component_correlations = self._calculate_component_correlations(component_prices)
            
            # Calculate indicator-to-indicator correlations
            indicator_correlations = self._calculate_indicator_correlations(technical_results)
            
            # Calculate timeframe-to-timeframe correlations
            timeframe_correlations = self._calculate_timeframe_correlations(technical_results)
            
            # Build comprehensive 6×6 correlation matrix
            correlation_matrix = self._build_comprehensive_correlation_matrix(
                component_correlations, indicator_correlations, timeframe_correlations
            )
            
            # Analyze correlation patterns
            correlation_analysis = self._analyze_correlation_patterns(correlation_matrix)
            
            # Calculate regime strength from correlations
            regime_strength = self._calculate_regime_strength_from_correlations(
                correlation_matrix, correlation_analysis
            )
            
            # Generate correlation alerts
            correlation_alerts = self._generate_correlation_alerts(correlation_matrix)
            
            return {
                'correlation_matrix': correlation_matrix,
                'component_correlations': component_correlations,
                'indicator_correlations': indicator_correlations,
                'timeframe_correlations': timeframe_correlations,
                'correlation_analysis': correlation_analysis,
                'regime_strength': regime_strength,
                'correlation_alerts': correlation_alerts,
                'correlation_summary': self._generate_correlation_summary(correlation_analysis),
                'regime_confidence': regime_strength.get('overall_confidence', 0.5),
                'calculation_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating real-time correlations: {e}")
            return self._get_default_correlation_results()
    
    def _extract_component_price_series(self, technical_results: Dict[str, Dict]) -> Dict[str, pd.Series]:
        """Extract price series from technical results for each component"""
        try:
            component_prices = {}
            
            for component in self.components:
                if component in technical_results:
                    # Try to extract price data from different timeframes
                    for timeframe in self.timeframes:
                        if timeframe in technical_results[component]:
                            timeframe_data = technical_results[component][timeframe]
                            
                            # Look for EMA data as proxy for prices
                            if 'ema_indicators' in timeframe_data:
                                ema_data = timeframe_data['ema_indicators']
                                if 'ema_20' in ema_data:
                                    component_prices[f"{component}_{timeframe}"] = ema_data['ema_20']
                                    break
                            
                            # Look for VWAP data as alternative
                            elif 'vwap_indicators' in timeframe_data:
                                vwap_data = timeframe_data['vwap_indicators']
                                if 'vwap_current' in vwap_data:
                                    component_prices[f"{component}_{timeframe}"] = vwap_data['vwap_current']
                                    break
                
                # Fail gracefully if no data found - NO SYNTHETIC DATA GENERATION
                if not any(f"{component}_" in key for key in component_prices.keys()):
                    logger.warning(f"No price data found for component {component} - skipping component")
                    # Do not generate synthetic data - let the system handle missing data properly
            
            return component_prices
            
        except Exception as e:
            logger.error(f"Error extracting component price series: {e}")
            return {}
    
    def _calculate_component_correlations(self, component_prices: Dict[str, pd.Series]) -> Dict[str, Dict]:
        """Calculate component-to-component correlations"""
        try:
            component_correlations = {}
            
            # Get unique components
            components = list(set([key.split('_')[0] + '_' + key.split('_')[1] 
                                 for key in component_prices.keys()]))
            
            for window in self.correlation_windows:
                window_correlations = {}
                
                # Calculate correlations between all component pairs
                for comp1, comp2 in combinations(components, 2):
                    if comp1 in component_prices and comp2 in component_prices:
                        series1 = component_prices[comp1]
                        series2 = component_prices[comp2]
                        
                        # Calculate rolling correlation
                        rolling_corr = series1.rolling(window=window).corr(series2)
                        current_corr = rolling_corr.iloc[-1] if len(rolling_corr) > 0 else 0.0
                        
                        pair_key = f"{comp1}_vs_{comp2}"
                        window_correlations[pair_key] = {
                            'correlation': current_corr,
                            'classification': self._classify_correlation(current_corr),
                            'rolling_series': rolling_corr,
                            'stability': rolling_corr.std() if len(rolling_corr) > 1 else 0.0
                        }
                
                component_correlations[f'window_{window}'] = window_correlations
            
            return component_correlations
            
        except Exception as e:
            logger.error(f"Error calculating component correlations: {e}")
            return {}
    
    def _calculate_indicator_correlations(self, technical_results: Dict[str, Dict]) -> Dict[str, Dict]:
        """Calculate indicator-to-indicator correlations across components"""
        try:
            indicator_correlations = {}
            
            for window in self.correlation_windows:
                window_correlations = {}
                
                # Extract indicator data across components
                indicator_data = {}
                for component in self.components:
                    if component in technical_results:
                        for timeframe in self.timeframes:
                            if timeframe in technical_results[component]:
                                timeframe_data = technical_results[component][timeframe]
                                
                                # Extract EMA indicators
                                if 'ema_indicators' in timeframe_data:
                                    ema_data = timeframe_data['ema_indicators']
                                    for indicator in ['ema_20', 'ema_100', 'ema_200']:
                                        if indicator in ema_data:
                                            key = f"{component}_{timeframe}_{indicator}"
                                            indicator_data[key] = ema_data[indicator]
                                
                                # Extract VWAP indicators
                                if 'vwap_indicators' in timeframe_data:
                                    vwap_data = timeframe_data['vwap_indicators']
                                    for indicator in ['vwap_current', 'vwap_previous']:
                                        if indicator in vwap_data:
                                            key = f"{component}_{timeframe}_{indicator}"
                                            indicator_data[key] = vwap_data[indicator]
                
                # Calculate correlations between indicators
                indicator_keys = list(indicator_data.keys())
                for ind1, ind2 in combinations(indicator_keys, 2):
                    if ind1 in indicator_data and ind2 in indicator_data:
                        series1 = indicator_data[ind1]
                        series2 = indicator_data[ind2]
                        
                        # Calculate rolling correlation
                        rolling_corr = series1.rolling(window=window).corr(series2)
                        current_corr = rolling_corr.iloc[-1] if len(rolling_corr) > 0 else 0.0
                        
                        pair_key = f"{ind1}_vs_{ind2}"
                        window_correlations[pair_key] = {
                            'correlation': current_corr,
                            'classification': self._classify_correlation(current_corr),
                            'rolling_series': rolling_corr,
                            'stability': rolling_corr.std() if len(rolling_corr) > 1 else 0.0
                        }
                
                indicator_correlations[f'window_{window}'] = window_correlations
            
            return indicator_correlations
            
        except Exception as e:
            logger.error(f"Error calculating indicator correlations: {e}")
            return {}
    
    def _calculate_timeframe_correlations(self, technical_results: Dict[str, Dict]) -> Dict[str, Dict]:
        """Calculate timeframe-to-timeframe correlations"""
        try:
            timeframe_correlations = {}
            
            for window in self.correlation_windows:
                window_correlations = {}
                
                # Extract data by timeframe
                timeframe_data = {}
                for component in self.components:
                    if component in technical_results:
                        for timeframe in self.timeframes:
                            if timeframe in technical_results[component]:
                                tf_data = technical_results[component][timeframe]
                                
                                # Use EMA 20 as representative indicator
                                if 'ema_indicators' in tf_data and 'ema_20' in tf_data['ema_indicators']:
                                    key = f"{component}_{timeframe}"
                                    timeframe_data[key] = tf_data['ema_indicators']['ema_20']
                
                # Calculate correlations between timeframes for same component
                for component in self.components:
                    component_timeframes = [key for key in timeframe_data.keys() 
                                          if key.startswith(component)]
                    
                    for tf1, tf2 in combinations(component_timeframes, 2):
                        if tf1 in timeframe_data and tf2 in timeframe_data:
                            series1 = timeframe_data[tf1]
                            series2 = timeframe_data[tf2]
                            
                            # Calculate rolling correlation
                            rolling_corr = series1.rolling(window=window).corr(series2)
                            current_corr = rolling_corr.iloc[-1] if len(rolling_corr) > 0 else 0.0
                            
                            pair_key = f"{tf1}_vs_{tf2}"
                            window_correlations[pair_key] = {
                                'correlation': current_corr,
                                'classification': self._classify_correlation(current_corr),
                                'rolling_series': rolling_corr,
                                'stability': rolling_corr.std() if len(rolling_corr) > 1 else 0.0
                            }
                
                timeframe_correlations[f'window_{window}'] = window_correlations
            
            return timeframe_correlations
            
        except Exception as e:
            logger.error(f"Error calculating timeframe correlations: {e}")
            return {}
    
    def _classify_correlation(self, correlation: float) -> str:
        """Classify correlation based on thresholds"""
        try:
            if pd.isna(correlation):
                return 'undefined'
            elif correlation > self.correlation_thresholds['high']:
                return 'high_positive'
            elif correlation > self.correlation_thresholds['medium_high']:
                return 'medium_positive'
            elif correlation > self.correlation_thresholds['low']:
                return 'low_positive'
            elif correlation > self.correlation_thresholds['negative']:
                return 'neutral'
            else:
                return 'negative'
        except:
            return 'undefined'
    
    def _build_comprehensive_correlation_matrix(self, component_correlations: Dict,
                                              indicator_correlations: Dict,
                                              timeframe_correlations: Dict) -> Dict[str, Any]:
        """Build comprehensive 6×6 correlation matrix"""
        try:
            correlation_matrix = {}
            
            # Use the most recent window for the matrix
            primary_window = f'window_{self.correlation_windows[-1]}'
            
            # Component correlation matrix
            if primary_window in component_correlations:
                correlation_matrix['component_matrix'] = component_correlations[primary_window]
            
            # Indicator correlation matrix
            if primary_window in indicator_correlations:
                correlation_matrix['indicator_matrix'] = indicator_correlations[primary_window]
            
            # Timeframe correlation matrix
            if primary_window in timeframe_correlations:
                correlation_matrix['timeframe_matrix'] = timeframe_correlations[primary_window]
            
            # Create summary matrix
            correlation_matrix['summary_matrix'] = self._create_summary_matrix(
                component_correlations, indicator_correlations, timeframe_correlations
            )
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Error building comprehensive correlation matrix: {e}")
            return {}
    
    def _create_summary_matrix(self, component_correlations: Dict,
                             indicator_correlations: Dict,
                             timeframe_correlations: Dict) -> Dict[str, float]:
        """Create summary correlation matrix"""
        try:
            summary_matrix = {}
            
            # Average correlations across all windows
            all_correlations = []
            
            # Collect component correlations
            for window_data in component_correlations.values():
                for pair_data in window_data.values():
                    if 'correlation' in pair_data and not pd.isna(pair_data['correlation']):
                        all_correlations.append(pair_data['correlation'])
            
            # Collect indicator correlations
            for window_data in indicator_correlations.values():
                for pair_data in window_data.values():
                    if 'correlation' in pair_data and not pd.isna(pair_data['correlation']):
                        all_correlations.append(pair_data['correlation'])
            
            # Collect timeframe correlations
            for window_data in timeframe_correlations.values():
                for pair_data in window_data.values():
                    if 'correlation' in pair_data and not pd.isna(pair_data['correlation']):
                        all_correlations.append(pair_data['correlation'])
            
            if all_correlations:
                summary_matrix['average_correlation'] = np.mean(all_correlations)
                summary_matrix['correlation_std'] = np.std(all_correlations)
                summary_matrix['max_correlation'] = np.max(all_correlations)
                summary_matrix['min_correlation'] = np.min(all_correlations)
                summary_matrix['positive_correlations'] = sum(1 for c in all_correlations if c > 0)
                summary_matrix['negative_correlations'] = sum(1 for c in all_correlations if c < 0)
                summary_matrix['total_correlations'] = len(all_correlations)
            else:
                summary_matrix = {
                    'average_correlation': 0.0,
                    'correlation_std': 0.0,
                    'max_correlation': 0.0,
                    'min_correlation': 0.0,
                    'positive_correlations': 0,
                    'negative_correlations': 0,
                    'total_correlations': 0
                }
            
            return summary_matrix
            
        except Exception as e:
            logger.error(f"Error creating summary matrix: {e}")
            return {}
    
    def _analyze_correlation_patterns(self, correlation_matrix: Dict) -> Dict[str, Any]:
        """Analyze correlation patterns for regime insights"""
        try:
            analysis = {}
            
            # Analyze summary matrix
            summary = correlation_matrix.get('summary_matrix', {})
            
            analysis['correlation_strength'] = {
                'overall_strength': abs(summary.get('average_correlation', 0)),
                'consistency': 1 - summary.get('correlation_std', 1),
                'directional_bias': 1 if summary.get('average_correlation', 0) > 0 else -1,
                'extreme_correlations': (
                    summary.get('max_correlation', 0) - summary.get('min_correlation', 0)
                )
            }
            
            # Count correlation classifications
            classification_counts = {'high_positive': 0, 'medium_positive': 0, 'low_positive': 0,
                                   'neutral': 0, 'negative': 0, 'undefined': 0}
            
            for matrix_type in ['component_matrix', 'indicator_matrix', 'timeframe_matrix']:
                if matrix_type in correlation_matrix:
                    for pair_data in correlation_matrix[matrix_type].values():
                        if 'classification' in pair_data:
                            classification = pair_data['classification']
                            if classification in classification_counts:
                                classification_counts[classification] += 1
            
            analysis['classification_distribution'] = classification_counts
            
            # Calculate regime coherence
            total_pairs = sum(classification_counts.values())
            if total_pairs > 0:
                high_coherence = (classification_counts['high_positive'] + 
                                classification_counts['medium_positive']) / total_pairs
                analysis['regime_coherence'] = high_coherence
            else:
                analysis['regime_coherence'] = 0.0
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing correlation patterns: {e}")
            return {}
    
    def _calculate_regime_strength_from_correlations(self, correlation_matrix: Dict,
                                                   correlation_analysis: Dict) -> Dict[str, float]:
        """Calculate regime strength from correlation analysis"""
        try:
            regime_strength = {}
            
            # Base strength from correlation coherence
            coherence = correlation_analysis.get('regime_coherence', 0.0)
            regime_strength['coherence_strength'] = coherence
            
            # Strength from correlation consistency
            consistency = correlation_analysis.get('correlation_strength', {}).get('consistency', 0.0)
            regime_strength['consistency_strength'] = consistency
            
            # Strength from directional alignment
            directional_strength = abs(correlation_analysis.get('correlation_strength', {}).get('directional_bias', 0))
            regime_strength['directional_strength'] = directional_strength
            
            # Overall confidence calculation
            regime_strength['overall_confidence'] = (
                coherence * 0.4 +
                consistency * 0.3 +
                directional_strength * 0.3
            )
            
            # Regime classification
            confidence = regime_strength['overall_confidence']
            if confidence > 0.8:
                regime_strength['regime_classification'] = 'strong_regime'
            elif confidence > 0.6:
                regime_strength['regime_classification'] = 'moderate_regime'
            elif confidence > 0.4:
                regime_strength['regime_classification'] = 'weak_regime'
            else:
                regime_strength['regime_classification'] = 'no_clear_regime'
            
            return regime_strength
            
        except Exception as e:
            logger.error(f"Error calculating regime strength: {e}")
            return {'overall_confidence': 0.0, 'regime_classification': 'undefined'}
    
    def _generate_correlation_alerts(self, correlation_matrix: Dict) -> List[Dict[str, Any]]:
        """Generate correlation-based alerts"""
        try:
            alerts = []
            
            # Check for extreme correlations
            summary = correlation_matrix.get('summary_matrix', {})
            
            if summary.get('max_correlation', 0) > 0.95:
                alerts.append({
                    'type': 'extreme_positive_correlation',
                    'severity': 'high',
                    'message': f"Extremely high correlation detected: {summary['max_correlation']:.3f}",
                    'timestamp': datetime.now().isoformat()
                })
            
            if summary.get('min_correlation', 0) < -0.8:
                alerts.append({
                    'type': 'extreme_negative_correlation',
                    'severity': 'high',
                    'message': f"Extremely negative correlation detected: {summary['min_correlation']:.3f}",
                    'timestamp': datetime.now().isoformat()
                })
            
            # Check for correlation breakdown
            if summary.get('correlation_std', 0) > 0.5:
                alerts.append({
                    'type': 'correlation_instability',
                    'severity': 'medium',
                    'message': f"High correlation instability: std={summary['correlation_std']:.3f}",
                    'timestamp': datetime.now().isoformat()
                })
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating correlation alerts: {e}")
            return []
    
    def _generate_correlation_summary(self, correlation_analysis: Dict) -> Dict[str, Any]:
        """Generate correlation summary"""
        try:
            summary = {}
            
            # High-level metrics
            summary['regime_coherence'] = correlation_analysis.get('regime_coherence', 0.0)
            summary['correlation_strength'] = correlation_analysis.get('correlation_strength', {})
            summary['classification_distribution'] = correlation_analysis.get('classification_distribution', {})
            
            # Count high correlations
            classification_dist = correlation_analysis.get('classification_distribution', {})
            summary['high_correlations'] = (
                classification_dist.get('high_positive', 0) + 
                classification_dist.get('medium_positive', 0)
            )
            
            summary['total_pairs_analyzed'] = sum(classification_dist.values())
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating correlation summary: {e}")
            return {}
    
    def _get_default_correlation_results(self) -> Dict[str, Any]:
        """Get default correlation results when calculation fails"""
        return {
            'correlation_matrix': {},
            'component_correlations': {},
            'indicator_correlations': {},
            'timeframe_correlations': {},
            'correlation_analysis': {},
            'regime_strength': {'overall_confidence': 0.0, 'regime_classification': 'undefined'},
            'correlation_alerts': [],
            'correlation_summary': {'high_correlations': 0, 'total_pairs_analyzed': 0},
            'regime_confidence': 0.0,
            'calculation_timestamp': datetime.now().isoformat(),
            'error': 'Correlation calculation failed'
        }
