#!/usr/bin/env python3
"""
Dynamic Support & Resistance Engine - Cross-Component Confluence Analysis
Part of Comprehensive Triple Straddle Engine V2.0

This engine provides comprehensive support & resistance analysis:
- Cross-component S&R level identification
- Confluence zone detection with 0.5% tolerance
- Dynamic S&R strength scoring
- Multi-timeframe S&R validation
- Real-time S&R level monitoring
- S&R breakout/breakdown detection

Author: The Augster
Date: 2025-06-23
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class DynamicSupportResistanceEngine:
    """
    Dynamic Support & Resistance Engine for cross-component analysis
    
    Provides comprehensive S&R analysis across all 6 components with
    confluence zone detection and dynamic strength scoring.
    """
    
    def __init__(self, confluence_tolerance: float = 0.005):
        """Initialize Dynamic Support & Resistance Engine"""
        self.confluence_tolerance = confluence_tolerance  # 0.5% tolerance for confluence
        
        # S&R strength thresholds
        self.strength_thresholds = {
            'very_strong': 0.8,
            'strong': 0.6,
            'moderate': 0.4,
            'weak': 0.2
        }
        
        # Components to analyze
        self.components = [
            'atm_straddle', 'itm1_straddle', 'otm1_straddle',
            'combined_straddle', 'atm_ce', 'atm_pe'
        ]
        
        # Timeframes for analysis
        self.timeframes = ['3min', '5min', '10min', '15min']
        
        logger.info(f"Dynamic S&R Engine initialized - Confluence tolerance: {confluence_tolerance:.1%}")
    
    def calculate_comprehensive_sr_analysis(self, technical_results: Dict[str, Dict],
                                          component_prices: Dict[str, pd.Series],
                                          timeframes: List[str]) -> Dict[str, Any]:
        """
        Calculate comprehensive support & resistance analysis
        
        Args:
            technical_results: Technical analysis results from all components
            component_prices: Price series for all components
            timeframes: List of timeframes to analyze
            
        Returns:
            Complete S&R analysis with confluence zones and strength scores
        """
        try:
            logger.debug("Calculating comprehensive S&R analysis")
            
            # Extract S&R levels from technical analysis
            sr_levels = self._extract_sr_levels_from_technical_analysis(technical_results)
            
            # Calculate dynamic S&R levels from price action
            dynamic_sr_levels = self._calculate_dynamic_sr_levels(component_prices)
            
            # Identify confluence zones
            confluence_zones = self._identify_confluence_zones(sr_levels, dynamic_sr_levels)
            
            # Calculate S&R strength scores
            sr_strength_scores = self._calculate_sr_strength_scores(
                sr_levels, dynamic_sr_levels, confluence_zones
            )
            
            # Detect S&R breakouts/breakdowns
            sr_breakouts = self._detect_sr_breakouts(component_prices, confluence_zones)
            
            # Analyze cross-component S&R alignment
            cross_component_analysis = self._analyze_cross_component_sr_alignment(
                sr_levels, confluence_zones
            )
            
            # Generate S&R alerts
            sr_alerts = self._generate_sr_alerts(confluence_zones, sr_breakouts)
            
            return {
                'static_levels': sr_levels,
                'dynamic_levels': dynamic_sr_levels,
                'confluence_zones': confluence_zones,
                'sr_strength_scores': sr_strength_scores,
                'sr_breakouts': sr_breakouts,
                'cross_component_analysis': cross_component_analysis,
                'sr_alerts': sr_alerts,
                'sr_summary': self._generate_sr_summary(confluence_zones, sr_strength_scores),
                'calculation_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive S&R analysis: {e}")
            return self._get_default_sr_results()
    
    def _extract_sr_levels_from_technical_analysis(self, technical_results: Dict[str, Dict]) -> Dict[str, List]:
        """Extract S&R levels from technical analysis results"""
        try:
            sr_levels = defaultdict(list)
            
            for component in self.components:
                if component in technical_results:
                    for timeframe in self.timeframes:
                        if timeframe in technical_results[component]:
                            timeframe_data = technical_results[component][timeframe]
                            
                            # Extract pivot-based S&R levels
                            if 'pivot_indicators' in timeframe_data:
                                pivot_data = timeframe_data['pivot_indicators']
                                
                                # Support levels
                                if 'support_1' in pivot_data:
                                    sr_levels[f"{component}_{timeframe}_support"].extend(
                                        pivot_data['support_1'].dropna().tolist()
                                    )
                                if 'support_2' in pivot_data:
                                    sr_levels[f"{component}_{timeframe}_support"].extend(
                                        pivot_data['support_2'].dropna().tolist()
                                    )
                                
                                # Resistance levels
                                if 'resistance_1' in pivot_data:
                                    sr_levels[f"{component}_{timeframe}_resistance"].extend(
                                        pivot_data['resistance_1'].dropna().tolist()
                                    )
                                if 'resistance_2' in pivot_data:
                                    sr_levels[f"{component}_{timeframe}_resistance"].extend(
                                        pivot_data['resistance_2'].dropna().tolist()
                                    )
                                
                                # Pivot points
                                if 'pivot_current' in pivot_data:
                                    sr_levels[f"{component}_{timeframe}_pivot"].extend(
                                        pivot_data['pivot_current'].dropna().tolist()
                                    )
                            
                            # Extract VWAP-based S&R levels
                            if 'vwap_indicators' in timeframe_data:
                                vwap_data = timeframe_data['vwap_indicators']
                                
                                if 'vwap_current' in vwap_data:
                                    sr_levels[f"{component}_{timeframe}_vwap"].extend(
                                        vwap_data['vwap_current'].dropna().tolist()
                                    )
                                
                                if 'vwap_upper_band' in vwap_data:
                                    sr_levels[f"{component}_{timeframe}_resistance"].extend(
                                        vwap_data['vwap_upper_band'].dropna().tolist()
                                    )
                                
                                if 'vwap_lower_band' in vwap_data:
                                    sr_levels[f"{component}_{timeframe}_support"].extend(
                                        vwap_data['vwap_lower_band'].dropna().tolist()
                                    )
            
            # Clean and deduplicate levels
            cleaned_sr_levels = {}
            for key, levels in sr_levels.items():
                if levels:
                    # Remove outliers and duplicates
                    levels_array = np.array(levels)
                    levels_array = levels_array[~np.isnan(levels_array)]
                    levels_array = levels_array[levels_array > 0]  # Remove negative/zero values
                    
                    if len(levels_array) > 0:
                        # Remove outliers (beyond 3 standard deviations)
                        mean_level = np.mean(levels_array)
                        std_level = np.std(levels_array)
                        levels_array = levels_array[
                            abs(levels_array - mean_level) <= 3 * std_level
                        ]
                        
                        # Deduplicate similar levels
                        unique_levels = []
                        for level in sorted(levels_array):
                            if not unique_levels or abs(level - unique_levels[-1]) / unique_levels[-1] > 0.01:
                                unique_levels.append(level)
                        
                        cleaned_sr_levels[key] = unique_levels
            
            return cleaned_sr_levels
            
        except Exception as e:
            logger.error(f"Error extracting S&R levels: {e}")
            return {}
    
    def _calculate_dynamic_sr_levels(self, component_prices: Dict[str, pd.Series]) -> Dict[str, List]:
        """Calculate dynamic S&R levels from price action"""
        try:
            dynamic_levels = {}
            
            for component, prices in component_prices.items():
                if len(prices) < 20:  # Need minimum data
                    continue
                
                component_levels = {
                    'support': [],
                    'resistance': [],
                    'pivot': []
                }
                
                # Calculate rolling highs and lows
                rolling_high = prices.rolling(window=20).max()
                rolling_low = prices.rolling(window=20).min()
                
                # Identify local maxima and minima
                local_maxima = self._find_local_extrema(prices, 'max')
                local_minima = self._find_local_extrema(prices, 'min')
                
                # Add resistance levels from local maxima
                component_levels['resistance'].extend(local_maxima)
                
                # Add support levels from local minima
                component_levels['support'].extend(local_minima)
                
                # Add pivot levels (midpoints between support and resistance)
                if component_levels['support'] and component_levels['resistance']:
                    for support in component_levels['support']:
                        for resistance in component_levels['resistance']:
                            if resistance > support:
                                pivot = (support + resistance) / 2
                                component_levels['pivot'].append(pivot)
                
                # Add psychological levels (round numbers)
                current_price = prices.iloc[-1] if len(prices) > 0 else 100
                psychological_levels = self._calculate_psychological_levels(current_price)
                component_levels['pivot'].extend(psychological_levels)
                
                # Clean and sort levels
                for level_type in component_levels:
                    if component_levels[level_type]:
                        levels = sorted(list(set(component_levels[level_type])))
                        component_levels[level_type] = levels
                
                dynamic_levels[component] = component_levels
            
            return dynamic_levels
            
        except Exception as e:
            logger.error(f"Error calculating dynamic S&R levels: {e}")
            return {}
    
    def _find_local_extrema(self, prices: pd.Series, extrema_type: str, window: int = 5) -> List[float]:
        """Find local maxima or minima in price series"""
        try:
            extrema = []
            
            for i in range(window, len(prices) - window):
                current_price = prices.iloc[i]
                window_prices = prices.iloc[i-window:i+window+1]
                
                if extrema_type == 'max':
                    if current_price == window_prices.max():
                        extrema.append(current_price)
                elif extrema_type == 'min':
                    if current_price == window_prices.min():
                        extrema.append(current_price)
            
            return extrema
            
        except Exception as e:
            logger.error(f"Error finding local extrema: {e}")
            return []
    
    def _calculate_psychological_levels(self, current_price: float) -> List[float]:
        """Calculate psychological support/resistance levels"""
        try:
            psychological_levels = []
            
            # Round number levels
            price_magnitude = 10 ** (len(str(int(current_price))) - 1)
            
            # Add levels at round numbers
            for multiplier in [0.5, 1.0, 1.5, 2.0]:
                level = price_magnitude * multiplier
                if 0.5 * current_price <= level <= 2.0 * current_price:
                    psychological_levels.append(level)
            
            # Add levels at 50% and 100% intervals
            base_level = int(current_price / 50) * 50
            for offset in [-100, -50, 0, 50, 100]:
                level = base_level + offset
                if level > 0:
                    psychological_levels.append(level)
            
            return psychological_levels
            
        except Exception as e:
            logger.error(f"Error calculating psychological levels: {e}")
            return []
    
    def _identify_confluence_zones(self, sr_levels: Dict[str, List], 
                                 dynamic_levels: Dict[str, List]) -> List[Dict[str, Any]]:
        """Identify confluence zones where multiple S&R levels converge"""
        try:
            confluence_zones = []
            
            # Collect all levels
            all_levels = []
            
            # Add static levels
            for key, levels in sr_levels.items():
                for level in levels:
                    all_levels.append({
                        'level': level,
                        'source': key,
                        'type': 'static'
                    })
            
            # Add dynamic levels
            for component, level_types in dynamic_levels.items():
                for level_type, levels in level_types.items():
                    for level in levels:
                        all_levels.append({
                            'level': level,
                            'source': f"{component}_{level_type}",
                            'type': 'dynamic'
                        })
            
            # Sort levels by value
            all_levels.sort(key=lambda x: x['level'])
            
            # Find confluence zones
            i = 0
            while i < len(all_levels):
                confluence_group = [all_levels[i]]
                j = i + 1
                
                # Group nearby levels
                while j < len(all_levels):
                    level_diff = abs(all_levels[j]['level'] - all_levels[i]['level'])
                    relative_diff = level_diff / all_levels[i]['level']
                    
                    if relative_diff <= self.confluence_tolerance:
                        confluence_group.append(all_levels[j])
                        j += 1
                    else:
                        break
                
                # Create confluence zone if multiple levels converge
                if len(confluence_group) >= 2:
                    zone_levels = [item['level'] for item in confluence_group]
                    zone_center = np.mean(zone_levels)
                    zone_range = max(zone_levels) - min(zone_levels)
                    
                    confluence_zone = {
                        'center': zone_center,
                        'range': zone_range,
                        'levels': confluence_group,
                        'strength': len(confluence_group),
                        'components_involved': len(set(item['source'].split('_')[0] for item in confluence_group)),
                        'static_count': sum(1 for item in confluence_group if item['type'] == 'static'),
                        'dynamic_count': sum(1 for item in confluence_group if item['type'] == 'dynamic'),
                        'zone_type': self._classify_confluence_zone(confluence_group)
                    }
                    
                    confluence_zones.append(confluence_zone)
                
                i = j if j > i else i + 1
            
            # Sort by strength
            confluence_zones.sort(key=lambda x: x['strength'], reverse=True)
            
            return confluence_zones
            
        except Exception as e:
            logger.error(f"Error identifying confluence zones: {e}")
            return []
    
    def _classify_confluence_zone(self, confluence_group: List[Dict]) -> str:
        """Classify confluence zone type"""
        try:
            sources = [item['source'] for item in confluence_group]
            
            # Count different types
            support_count = sum(1 for source in sources if 'support' in source)
            resistance_count = sum(1 for source in sources if 'resistance' in source)
            pivot_count = sum(1 for source in sources if 'pivot' in source or 'vwap' in source)
            
            if support_count > resistance_count and support_count > pivot_count:
                return 'support_zone'
            elif resistance_count > support_count and resistance_count > pivot_count:
                return 'resistance_zone'
            else:
                return 'pivot_zone'
                
        except Exception as e:
            logger.error(f"Error classifying confluence zone: {e}")
            return 'unknown'
    
    def _calculate_sr_strength_scores(self, sr_levels: Dict[str, List],
                                    dynamic_levels: Dict[str, List],
                                    confluence_zones: List[Dict]) -> Dict[str, float]:
        """Calculate S&R strength scores"""
        try:
            strength_scores = {}
            
            # Score confluence zones
            for i, zone in enumerate(confluence_zones):
                zone_id = f"confluence_zone_{i}"
                
                # Base strength from number of converging levels
                base_strength = min(zone['strength'] / 10, 1.0)  # Normalize to max 1.0
                
                # Bonus for cross-component involvement
                component_bonus = min(zone['components_involved'] / 6, 1.0) * 0.2
                
                # Bonus for mix of static and dynamic levels
                mix_bonus = 0.1 if zone['static_count'] > 0 and zone['dynamic_count'] > 0 else 0
                
                # Calculate final strength
                final_strength = base_strength + component_bonus + mix_bonus
                strength_scores[zone_id] = min(final_strength, 1.0)
            
            # Score individual components
            for component in self.components:
                component_strength = 0.0
                level_count = 0
                
                # Count levels from this component
                for key, levels in sr_levels.items():
                    if component in key:
                        level_count += len(levels)
                
                if component in dynamic_levels:
                    for level_type, levels in dynamic_levels[component].items():
                        level_count += len(levels)
                
                # Calculate component strength
                if level_count > 0:
                    component_strength = min(level_count / 20, 1.0)  # Normalize
                
                strength_scores[f"{component}_strength"] = component_strength
            
            return strength_scores
            
        except Exception as e:
            logger.error(f"Error calculating S&R strength scores: {e}")
            return {}
    
    def _detect_sr_breakouts(self, component_prices: Dict[str, pd.Series],
                           confluence_zones: List[Dict]) -> Dict[str, List]:
        """Detect S&R breakouts and breakdowns"""
        try:
            breakouts = defaultdict(list)
            
            for component, prices in component_prices.items():
                if len(prices) < 2:
                    continue
                
                current_price = prices.iloc[-1]
                previous_price = prices.iloc[-2]
                
                for i, zone in enumerate(confluence_zones):
                    zone_center = zone['center']
                    zone_range = zone['range']
                    zone_upper = zone_center + zone_range / 2
                    zone_lower = zone_center - zone_range / 2
                    
                    # Check for breakout (upward)
                    if (previous_price <= zone_upper and current_price > zone_upper and
                        zone['zone_type'] in ['resistance_zone', 'pivot_zone']):
                        breakouts[component].append({
                            'type': 'breakout',
                            'zone_id': i,
                            'zone_center': zone_center,
                            'breakout_price': current_price,
                            'strength': zone['strength'],
                            'timestamp': datetime.now().isoformat()
                        })
                    
                    # Check for breakdown (downward)
                    if (previous_price >= zone_lower and current_price < zone_lower and
                        zone['zone_type'] in ['support_zone', 'pivot_zone']):
                        breakouts[component].append({
                            'type': 'breakdown',
                            'zone_id': i,
                            'zone_center': zone_center,
                            'breakdown_price': current_price,
                            'strength': zone['strength'],
                            'timestamp': datetime.now().isoformat()
                        })
            
            return dict(breakouts)
            
        except Exception as e:
            logger.error(f"Error detecting S&R breakouts: {e}")
            return {}
    
    def _analyze_cross_component_sr_alignment(self, sr_levels: Dict[str, List],
                                            confluence_zones: List[Dict]) -> Dict[str, Any]:
        """Analyze cross-component S&R alignment"""
        try:
            alignment_analysis = {}
            
            # Count components with S&R levels
            components_with_levels = set()
            for key in sr_levels.keys():
                component = key.split('_')[0] + '_' + key.split('_')[1]
                components_with_levels.add(component)
            
            alignment_analysis['components_with_sr'] = len(components_with_levels)
            alignment_analysis['total_components'] = len(self.components)
            alignment_analysis['coverage_ratio'] = len(components_with_levels) / len(self.components)
            
            # Analyze confluence zone distribution
            alignment_analysis['total_confluence_zones'] = len(confluence_zones)
            alignment_analysis['strong_zones'] = sum(1 for zone in confluence_zones if zone['strength'] >= 5)
            alignment_analysis['cross_component_zones'] = sum(1 for zone in confluence_zones if zone['components_involved'] >= 3)
            
            # Calculate alignment strength
            if confluence_zones:
                avg_zone_strength = np.mean([zone['strength'] for zone in confluence_zones])
                avg_component_involvement = np.mean([zone['components_involved'] for zone in confluence_zones])
                
                alignment_analysis['average_zone_strength'] = avg_zone_strength
                alignment_analysis['average_component_involvement'] = avg_component_involvement
                alignment_analysis['overall_alignment_score'] = (
                    alignment_analysis['coverage_ratio'] * 0.4 +
                    min(avg_zone_strength / 10, 1.0) * 0.3 +
                    min(avg_component_involvement / 6, 1.0) * 0.3
                )
            else:
                alignment_analysis['overall_alignment_score'] = 0.0
            
            return alignment_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing cross-component S&R alignment: {e}")
            return {}
    
    def _generate_sr_alerts(self, confluence_zones: List[Dict], 
                          sr_breakouts: Dict[str, List]) -> List[Dict[str, Any]]:
        """Generate S&R-based alerts"""
        try:
            alerts = []
            
            # Alert for strong confluence zones
            for i, zone in enumerate(confluence_zones):
                if zone['strength'] >= 7:
                    alerts.append({
                        'type': 'strong_confluence_zone',
                        'severity': 'high',
                        'message': f"Strong confluence zone at {zone['center']:.2f} with {zone['strength']} levels",
                        'zone_id': i,
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Alert for breakouts
            for component, breakout_list in sr_breakouts.items():
                for breakout in breakout_list:
                    if breakout['strength'] >= 5:
                        alerts.append({
                            'type': f"sr_{breakout['type']}",
                            'severity': 'medium',
                            'message': f"{component} {breakout['type']} at {breakout.get('breakout_price', breakout.get('breakdown_price', 0)):.2f}",
                            'component': component,
                            'timestamp': breakout['timestamp']
                        })
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating S&R alerts: {e}")
            return []
    
    def _generate_sr_summary(self, confluence_zones: List[Dict], 
                           sr_strength_scores: Dict[str, float]) -> Dict[str, Any]:
        """Generate S&R summary"""
        try:
            summary = {
                'total_confluence_zones': len(confluence_zones),
                'strong_zones': sum(1 for zone in confluence_zones if zone['strength'] >= 5),
                'average_zone_strength': np.mean([zone['strength'] for zone in confluence_zones]) if confluence_zones else 0,
                'max_zone_strength': max([zone['strength'] for zone in confluence_zones]) if confluence_zones else 0,
                'cross_component_zones': sum(1 for zone in confluence_zones if zone['components_involved'] >= 3),
                'overall_sr_strength': np.mean(list(sr_strength_scores.values())) if sr_strength_scores else 0
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating S&R summary: {e}")
            return {}
    
    def _get_default_sr_results(self) -> Dict[str, Any]:
        """Get default S&R results when calculation fails"""
        return {
            'static_levels': {},
            'dynamic_levels': {},
            'confluence_zones': [],
            'sr_strength_scores': {},
            'sr_breakouts': {},
            'cross_component_analysis': {},
            'sr_alerts': [],
            'sr_summary': {'total_confluence_zones': 0, 'overall_sr_strength': 0},
            'calculation_timestamp': datetime.now().isoformat(),
            'error': 'S&R calculation failed'
        }
