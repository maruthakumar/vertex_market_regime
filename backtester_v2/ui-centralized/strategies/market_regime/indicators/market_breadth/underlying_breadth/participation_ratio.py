"""
Participation Ratio - Market Participation Analysis for Breadth Assessment
=========================================================================

Analyzes market participation patterns for comprehensive breadth assessment.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ParticipationRatio:
    """Market participation ratio analyzer for breadth assessment"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Participation Ratio analyzer"""
        self.participation_window = config.get('participation_window', 20)
        self.volume_thresholds = config.get('volume_thresholds', [0.5, 1.0, 2.0])  # Relative to average
        self.price_change_thresholds = config.get('price_change_thresholds', [0.01, 0.02, 0.05])
        
        # Historical tracking
        self.participation_history = {
            'total_participation': [],
            'active_participation': [],
            'volume_participation': [],
            'price_participation': [],
            'sector_participation': [],
            'timestamps': []
        }
        
        # Participation quality tracking
        self.quality_metrics = {
            'high_quality_days': 0,
            'low_quality_days': 0,
            'participation_momentum': [],
            'breadth_quality_score': 0.0
        }
        
        logger.info("ParticipationRatio initialized")
    
    def analyze_participation_ratio(self, underlying_data: pd.DataFrame, sector_data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, Any]:
        """
        Analyze market participation ratios for breadth assessment
        
        Args:
            underlying_data: DataFrame with underlying market data
            sector_data: Optional sector-wise data breakdown
            
        Returns:
            Dict with participation analysis results
        """
        try:
            if underlying_data.empty:
                return self._get_default_participation_analysis()
            
            # Calculate basic participation metrics
            participation_metrics = self._calculate_participation_metrics(underlying_data)
            
            # Analyze volume participation
            volume_participation = self._analyze_volume_participation(underlying_data)
            
            # Analyze price participation
            price_participation = self._analyze_price_participation(underlying_data)
            
            # Analyze sector participation
            sector_participation = self._analyze_sector_participation(underlying_data, sector_data)
            
            # Calculate participation quality
            participation_quality = self._calculate_participation_quality(participation_metrics, volume_participation, price_participation)
            
            # Analyze participation trends
            participation_trends = self._analyze_participation_trends(participation_metrics)
            
            # Generate participation signals
            participation_signals = self._generate_participation_signals(participation_metrics, participation_quality, participation_trends)
            
            # Update historical tracking
            self._update_participation_history(participation_metrics, volume_participation, price_participation)
            
            return {
                'participation_metrics': participation_metrics,
                'volume_participation': volume_participation,
                'price_participation': price_participation,
                'sector_participation': sector_participation,
                'participation_quality': participation_quality,
                'participation_trends': participation_trends,
                'participation_signals': participation_signals,
                'breadth_score': self._calculate_participation_breadth_score(participation_metrics, participation_quality)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing participation ratio: {e}")
            return self._get_default_participation_analysis()
    
    def _calculate_participation_metrics(self, underlying_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic participation metrics"""
        try:
            metrics = {}
            
            total_issues = len(underlying_data)
            metrics['total_issues'] = total_issues
            
            if total_issues == 0:
                return {'total_issues': 0, 'overall_participation': 0.0}
            
            # Overall participation (issues with activity)
            if 'volume' in underlying_data.columns:
                active_issues = len(underlying_data[underlying_data['volume'] > 0])
                metrics['active_issues'] = active_issues
                metrics['overall_participation'] = float(active_issues / total_issues)
            else:
                metrics['active_issues'] = total_issues
                metrics['overall_participation'] = 1.0
            
            # Price movement participation
            if 'price_change' in underlying_data.columns:
                moving_issues = len(underlying_data[underlying_data['price_change'] != 0])
                metrics['moving_issues'] = moving_issues
                metrics['price_participation'] = float(moving_issues / total_issues)
                
                # Directional participation
                advancing_issues = len(underlying_data[underlying_data['price_change'] > 0])
                declining_issues = len(underlying_data[underlying_data['price_change'] < 0])
                unchanged_issues = total_issues - advancing_issues - declining_issues
                
                metrics['advancing_issues'] = advancing_issues
                metrics['declining_issues'] = declining_issues
                metrics['unchanged_issues'] = unchanged_issues
                
                metrics['advancing_ratio'] = float(advancing_issues / total_issues)
                metrics['declining_ratio'] = float(declining_issues / total_issues)
                metrics['unchanged_ratio'] = float(unchanged_issues / total_issues)
                
                # Net participation
                metrics['net_advancing'] = advancing_issues - declining_issues
                metrics['advance_decline_ratio'] = float(advancing_issues / declining_issues) if declining_issues > 0 else float('inf')
            
            # Volume-weighted participation
            if 'volume' in underlying_data.columns and underlying_data['volume'].sum() > 0:
                total_volume = underlying_data['volume'].sum()
                
                # High volume participation
                avg_volume = underlying_data['volume'].mean()
                high_volume_issues = len(underlying_data[underlying_data['volume'] > avg_volume * 2])
                metrics['high_volume_issues'] = high_volume_issues
                metrics['high_volume_participation'] = float(high_volume_issues / total_issues)
                
                # Volume concentration
                top_10_volume = underlying_data.nlargest(10, 'volume')['volume'].sum()
                metrics['top_10_volume_concentration'] = float(top_10_volume / total_volume)
                
                # Volume participation by quartiles
                volume_quartiles = underlying_data['volume'].quantile([0.25, 0.5, 0.75, 1.0])
                for i, (q, threshold) in enumerate(zip([25, 50, 75, 100], volume_quartiles)):
                    participating_issues = len(underlying_data[underlying_data['volume'] >= threshold])
                    metrics[f'volume_participation_q{q}'] = float(participating_issues / total_issues)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating participation metrics: {e}")
            return {'total_issues': 0, 'overall_participation': 0.0}
    
    def _analyze_volume_participation(self, underlying_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume-based participation patterns"""
        try:
            volume_participation = {}
            
            if 'volume' not in underlying_data.columns:
                return volume_participation
            
            total_issues = len(underlying_data)
            total_volume = underlying_data['volume'].sum()
            
            if total_issues == 0 or total_volume == 0:
                return volume_participation
            
            # Calculate average volume for thresholds
            avg_volume = underlying_data['volume'].mean()
            
            # Participation by volume thresholds
            for threshold in self.volume_thresholds:
                threshold_volume = avg_volume * threshold
                participating_issues = len(underlying_data[underlying_data['volume'] >= threshold_volume])
                participation_ratio = participating_issues / total_issues
                
                volume_participation[f'volume_threshold_{threshold}x'] = {
                    'participating_issues': participating_issues,
                    'participation_ratio': float(participation_ratio),
                    'threshold_volume': float(threshold_volume)
                }
            
            # Volume distribution analysis
            volume_data = underlying_data['volume'].sort_values(ascending=False)
            
            # Participation breadth by volume deciles
            decile_participation = {}
            for decile in range(1, 11):
                decile_threshold = np.percentile(underlying_data['volume'], 100 - decile * 10)
                participating_issues = len(underlying_data[underlying_data['volume'] >= decile_threshold])
                decile_participation[f'top_{decile}0_percent'] = float(participating_issues / total_issues)
            
            volume_participation['decile_participation'] = decile_participation
            
            # Volume momentum participation
            if 'volume_change' in underlying_data.columns:
                increasing_volume = len(underlying_data[underlying_data['volume_change'] > 0])
                decreasing_volume = len(underlying_data[underlying_data['volume_change'] < 0])
                
                volume_participation['increasing_volume_participation'] = float(increasing_volume / total_issues)
                volume_participation['decreasing_volume_participation'] = float(decreasing_volume / total_issues)
                volume_participation['volume_momentum_ratio'] = float(increasing_volume / decreasing_volume) if decreasing_volume > 0 else float('inf')
            
            # Unusual volume participation
            volume_std = underlying_data['volume'].std()
            unusual_volume_threshold = avg_volume + 2 * volume_std
            unusual_volume_issues = len(underlying_data[underlying_data['volume'] > unusual_volume_threshold])
            
            volume_participation['unusual_volume'] = {
                'issues': unusual_volume_issues,
                'participation_ratio': float(unusual_volume_issues / total_issues),
                'threshold': float(unusual_volume_threshold)
            }
            
            return volume_participation
            
        except Exception as e:
            logger.error(f"Error analyzing volume participation: {e}")
            return {}
    
    def _analyze_price_participation(self, underlying_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price movement-based participation patterns"""
        try:
            price_participation = {}
            
            if 'price_change' not in underlying_data.columns:
                return price_participation
            
            total_issues = len(underlying_data)
            
            if total_issues == 0:
                return price_participation
            
            # Participation by price change thresholds
            for threshold in self.price_change_thresholds:
                # Up moves above threshold
                strong_advances = len(underlying_data[underlying_data['price_change'] > threshold])
                # Down moves below threshold
                strong_declines = len(underlying_data[underlying_data['price_change'] < -threshold])
                # Any significant move
                significant_moves = strong_advances + strong_declines
                
                price_participation[f'price_threshold_{threshold:.3f}'] = {
                    'strong_advances': strong_advances,
                    'strong_declines': strong_declines,
                    'significant_moves': significant_moves,
                    'advance_participation': float(strong_advances / total_issues),
                    'decline_participation': float(strong_declines / total_issues),
                    'significant_participation': float(significant_moves / total_issues)
                }
            
            # Price range participation
            price_ranges = {
                'small': (0, 0.01),
                'medium': (0.01, 0.03),
                'large': (0.03, 0.05),
                'extreme': (0.05, float('inf'))
            }
            
            range_participation = {}
            for range_name, (min_change, max_change) in price_ranges.items():
                range_issues = len(underlying_data[
                    (abs(underlying_data['price_change']) > min_change) & 
                    (abs(underlying_data['price_change']) <= max_change)
                ])
                range_participation[range_name] = {
                    'issues': range_issues,
                    'participation_ratio': float(range_issues / total_issues)
                }
            
            price_participation['range_participation'] = range_participation
            
            # Volatility participation
            price_volatility = underlying_data['price_change'].std()
            high_volatility_threshold = price_volatility * 1.5
            high_volatility_issues = len(underlying_data[abs(underlying_data['price_change']) > high_volatility_threshold])
            
            price_participation['volatility_participation'] = {
                'high_volatility_issues': high_volatility_issues,
                'participation_ratio': float(high_volatility_issues / total_issues),
                'volatility_threshold': float(high_volatility_threshold)
            }
            
            # Directional consistency
            if total_issues > 0:
                advancing = len(underlying_data[underlying_data['price_change'] > 0])
                declining = len(underlying_data[underlying_data['price_change'] < 0])
                
                directional_dominance = max(advancing, declining) / total_issues
                price_participation['directional_consistency'] = {
                    'dominance_ratio': float(directional_dominance),
                    'consistency_type': 'bullish' if advancing > declining else 'bearish' if declining > advancing else 'neutral'
                }
            
            return price_participation
            
        except Exception as e:
            logger.error(f"Error analyzing price participation: {e}")
            return {}
    
    def _analyze_sector_participation(self, underlying_data: pd.DataFrame, sector_data: Optional[Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """Analyze sector-based participation patterns"""
        try:
            sector_participation = {}
            
            if sector_data is None or not sector_data:
                # Try to infer from underlying data
                if 'sector' in underlying_data.columns:
                    sectors = underlying_data['sector'].unique()
                    sector_participation['total_sectors'] = len(sectors)
                    
                    active_sectors = 0
                    sector_details = {}
                    
                    for sector in sectors:
                        sector_data_subset = underlying_data[underlying_data['sector'] == sector]
                        
                        if 'volume' in sector_data_subset.columns:
                            sector_volume = sector_data_subset['volume'].sum()
                            active_issues = len(sector_data_subset[sector_data_subset['volume'] > 0])
                        else:
                            sector_volume = 0
                            active_issues = len(sector_data_subset)
                        
                        if active_issues > 0:
                            active_sectors += 1
                        
                        sector_details[sector] = {
                            'total_issues': len(sector_data_subset),
                            'active_issues': active_issues,
                            'total_volume': float(sector_volume),
                            'participation_ratio': float(active_issues / len(sector_data_subset)) if len(sector_data_subset) > 0 else 0.0
                        }
                    
                    sector_participation['active_sectors'] = active_sectors
                    sector_participation['sector_participation_ratio'] = float(active_sectors / len(sectors)) if len(sectors) > 0 else 0.0
                    sector_participation['sector_details'] = sector_details
                
                return sector_participation
            
            # Analyze provided sector data
            total_sectors = len(sector_data)
            active_sectors = 0
            sector_details = {}
            total_volume_all_sectors = 0
            
            for sector_name, sector_df in sector_data.items():
                if sector_df.empty:
                    continue
                
                sector_issues = len(sector_df)
                sector_volume = sector_df['volume'].sum() if 'volume' in sector_df.columns else 0
                total_volume_all_sectors += sector_volume
                
                # Active issues in sector
                if 'volume' in sector_df.columns:
                    active_issues = len(sector_df[sector_df['volume'] > 0])
                else:
                    active_issues = sector_issues
                
                if active_issues > 0:
                    active_sectors += 1
                
                # Sector participation quality
                participation_ratio = active_issues / sector_issues if sector_issues > 0 else 0.0
                
                # Price movement in sector
                if 'price_change' in sector_df.columns:
                    advancing = len(sector_df[sector_df['price_change'] > 0])
                    declining = len(sector_df[sector_df['price_change'] < 0])
                    sector_direction = 'bullish' if advancing > declining else 'bearish' if declining > advancing else 'neutral'
                else:
                    advancing = 0
                    declining = 0
                    sector_direction = 'neutral'
                
                sector_details[sector_name] = {
                    'total_issues': sector_issues,
                    'active_issues': active_issues,
                    'total_volume': float(sector_volume),
                    'participation_ratio': float(participation_ratio),
                    'advancing_issues': advancing,
                    'declining_issues': declining,
                    'sector_direction': sector_direction
                }
            
            sector_participation['total_sectors'] = total_sectors
            sector_participation['active_sectors'] = active_sectors
            sector_participation['sector_participation_ratio'] = float(active_sectors / total_sectors) if total_sectors > 0 else 0.0
            sector_participation['sector_details'] = sector_details
            
            # Volume concentration by sector
            if total_volume_all_sectors > 0:
                sector_volume_concentration = {}
                for sector_name, details in sector_details.items():
                    sector_volume_concentration[sector_name] = details['total_volume'] / total_volume_all_sectors
                sector_participation['volume_concentration'] = sector_volume_concentration
            
            # Sector breadth quality
            participation_ratios = [details['participation_ratio'] for details in sector_details.values()]
            if participation_ratios:
                avg_participation = np.mean(participation_ratios)
                participation_consistency = 1.0 - np.std(participation_ratios)
                
                sector_participation['sector_breadth_quality'] = {
                    'average_participation': float(avg_participation),
                    'participation_consistency': float(max(participation_consistency, 0.0)),
                    'quality_score': float((avg_participation + max(participation_consistency, 0.0)) / 2)
                }
            
            return sector_participation
            
        except Exception as e:
            logger.error(f"Error analyzing sector participation: {e}")
            return {}
    
    def _calculate_participation_quality(self, participation_metrics: Dict[str, Any], volume_participation: Dict[str, Any], price_participation: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall participation quality"""
        try:
            quality = {
                'overall_quality_score': 0.0,
                'quality_components': {},
                'quality_classification': 'neutral'
            }
            
            quality_score = 0.0
            component_count = 0
            
            # Participation breadth component (30%)
            overall_participation = participation_metrics.get('overall_participation', 0.0)
            breadth_score = overall_participation
            quality['quality_components']['participation_breadth'] = float(breadth_score)
            quality_score += breadth_score * 0.3
            component_count += 1
            
            # Volume participation quality (25%)
            if volume_participation:
                # Average participation across volume thresholds
                volume_scores = []
                for threshold in self.volume_thresholds:
                    threshold_key = f'volume_threshold_{threshold}x'
                    if threshold_key in volume_participation:
                        volume_scores.append(volume_participation[threshold_key]['participation_ratio'])
                
                if volume_scores:
                    volume_quality = np.mean(volume_scores)
                    quality['quality_components']['volume_quality'] = float(volume_quality)
                    quality_score += volume_quality * 0.25
                    component_count += 1
            
            # Price participation quality (25%)
            if price_participation:
                # Significant price movement participation
                price_scores = []
                for threshold in self.price_change_thresholds:
                    threshold_key = f'price_threshold_{threshold:.3f}'
                    if threshold_key in price_participation:
                        price_scores.append(price_participation[threshold_key]['significant_participation'])
                
                if price_scores:
                    price_quality = np.mean(price_scores)
                    quality['quality_components']['price_quality'] = float(price_quality)
                    quality_score += price_quality * 0.25
                    component_count += 1
            
            # Directional consistency (20%)
            advancing_ratio = participation_metrics.get('advancing_ratio', 0.5)
            declining_ratio = participation_metrics.get('declining_ratio', 0.5)
            
            # Balanced participation gets higher score
            balance_score = 1.0 - abs(advancing_ratio - declining_ratio)
            # But strong directional bias also has value
            directional_strength = max(advancing_ratio, declining_ratio)
            
            consistency_score = (balance_score * 0.6 + directional_strength * 0.4)
            quality['quality_components']['directional_consistency'] = float(consistency_score)
            quality_score += consistency_score * 0.2
            component_count += 1
            
            # Calculate overall quality score
            if component_count > 0:
                quality['overall_quality_score'] = float(quality_score)
            
            # Classify quality
            if quality['overall_quality_score'] > 0.8:
                quality['quality_classification'] = 'excellent'
                self.quality_metrics['high_quality_days'] += 1
            elif quality['overall_quality_score'] > 0.6:
                quality['quality_classification'] = 'good'
            elif quality['overall_quality_score'] > 0.4:
                quality['quality_classification'] = 'moderate'
            elif quality['overall_quality_score'] > 0.2:
                quality['quality_classification'] = 'poor'
            else:
                quality['quality_classification'] = 'very_poor'
                self.quality_metrics['low_quality_days'] += 1
            
            # Update breadth quality score
            self.quality_metrics['breadth_quality_score'] = quality['overall_quality_score']
            
            return quality
            
        except Exception as e:
            logger.error(f"Error calculating participation quality: {e}")
            return {'overall_quality_score': 0.5, 'quality_classification': 'neutral'}
    
    def _analyze_participation_trends(self, participation_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trends in participation patterns"""
        try:
            trends = {}
            
            # Overall participation trend
            if len(self.participation_history['total_participation']) >= 5:
                recent_participation = self.participation_history['total_participation'][-5:] + [participation_metrics.get('overall_participation', 0.0)]
                participation_slope = self._calculate_slope(recent_participation)
                
                trends['participation_slope'] = float(participation_slope)
                trends['participation_trend'] = 'expanding' if participation_slope > 0.02 else 'contracting' if participation_slope < -0.02 else 'stable'
            
            # Advance/decline participation trend
            if len(self.participation_history['active_participation']) >= 3:
                recent_active = self.participation_history['active_participation'][-3:]
                current_active = participation_metrics.get('advancing_ratio', 0.5)
                recent_active.append(current_active)
                
                active_momentum = self._calculate_slope(recent_active)
                trends['advance_momentum'] = float(active_momentum)
                
                if active_momentum > 0.05:
                    trends['momentum_direction'] = 'increasingly_bullish'
                elif active_momentum < -0.05:
                    trends['momentum_direction'] = 'increasingly_bearish'
                else:
                    trends['momentum_direction'] = 'stable'
            
            # Participation momentum
            current_quality = self.quality_metrics['breadth_quality_score']
            self.quality_metrics['participation_momentum'].append(current_quality)
            
            if len(self.quality_metrics['participation_momentum']) > self.participation_window:
                self.quality_metrics['participation_momentum'].pop(0)
            
            if len(self.quality_metrics['participation_momentum']) >= 3:
                momentum_slope = self._calculate_slope(self.quality_metrics['participation_momentum'])
                trends['quality_momentum'] = float(momentum_slope)
                
                if momentum_slope > 0.02:
                    trends['quality_trend'] = 'improving'
                elif momentum_slope < -0.02:
                    trends['quality_trend'] = 'deteriorating'
                else:
                    trends['quality_trend'] = 'stable'
            
            # Historical comparison
            if len(self.participation_history['total_participation']) >= 10:
                recent_avg = np.mean(self.participation_history['total_participation'][-5:])
                historical_avg = np.mean(self.participation_history['total_participation'][:-5])
                
                if historical_avg > 0:
                    relative_performance = (recent_avg - historical_avg) / historical_avg
                    trends['historical_comparison'] = float(relative_performance)
                    
                    if relative_performance > 0.1:
                        trends['historical_assessment'] = 'above_average'
                    elif relative_performance < -0.1:
                        trends['historical_assessment'] = 'below_average'
                    else:
                        trends['historical_assessment'] = 'average'
            
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing participation trends: {e}")
            return {}
    
    def _generate_participation_signals(self, participation_metrics: Dict[str, Any], participation_quality: Dict[str, Any], participation_trends: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable participation signals"""
        try:
            signals = {
                'primary_signal': 'neutral',
                'signal_strength': 0.0,
                'participation_signals': [],
                'breadth_implications': []
            }
            
            # Quality-based signals
            quality_score = participation_quality.get('overall_quality_score', 0.5)
            quality_class = participation_quality.get('quality_classification', 'neutral')
            
            if quality_class in ['excellent', 'good']:
                signals['participation_signals'].append('high_quality_participation')
                signals['breadth_implications'].append('broad_healthy_market_engagement')
                signals['signal_strength'] += quality_score * 0.3
                
            elif quality_class in ['poor', 'very_poor']:
                signals['participation_signals'].append('low_quality_participation')
                signals['breadth_implications'].append('narrow_unhealthy_market_engagement')
                signals['signal_strength'] += (1.0 - quality_score) * 0.3
            
            # Participation level signals
            overall_participation = participation_metrics.get('overall_participation', 0.0)
            if overall_participation > 0.8:
                signals['participation_signals'].append('broad_market_participation')
                signals['breadth_implications'].append('widespread_market_activity')
                signals['signal_strength'] += 0.2
                
            elif overall_participation < 0.4:
                signals['participation_signals'].append('narrow_market_participation')
                signals['breadth_implications'].append('limited_market_activity')
                signals['signal_strength'] += 0.2
            
            # Trend-based signals
            participation_trend = participation_trends.get('participation_trend', 'stable')
            if participation_trend == 'expanding':
                signals['participation_signals'].append('expanding_participation')
                signals['breadth_implications'].append('increasing_market_breadth')
                signals['signal_strength'] += 0.2
                
            elif participation_trend == 'contracting':
                signals['participation_signals'].append('contracting_participation')
                signals['breadth_implications'].append('decreasing_market_breadth')
                signals['signal_strength'] += 0.2
            
            # Momentum signals
            momentum_direction = participation_trends.get('momentum_direction', 'stable')
            if momentum_direction == 'increasingly_bullish':
                signals['participation_signals'].append('bullish_participation_momentum')
                signals['breadth_implications'].append('strengthening_bullish_breadth')
                signals['signal_strength'] += 0.2
                
            elif momentum_direction == 'increasingly_bearish':
                signals['participation_signals'].append('bearish_participation_momentum')
                signals['breadth_implications'].append('strengthening_bearish_breadth')
                signals['signal_strength'] += 0.2
            
            # Quality trend signals
            quality_trend = participation_trends.get('quality_trend', 'stable')
            if quality_trend == 'improving':
                signals['participation_signals'].append('improving_participation_quality')
                signals['breadth_implications'].append('healthier_market_dynamics')
                signals['signal_strength'] += 0.1
                
            elif quality_trend == 'deteriorating':
                signals['participation_signals'].append('deteriorating_participation_quality')
                signals['breadth_implications'].append('weakening_market_dynamics')
                signals['signal_strength'] += 0.1
            
            # Determine primary signal
            if signals['signal_strength'] > 0.6:
                if any('high_quality' in sig or 'broad' in sig or 'expanding' in sig for sig in signals['participation_signals']):
                    signals['primary_signal'] = 'strong_positive_participation'
                elif any('low_quality' in sig or 'narrow' in sig or 'contracting' in sig for sig in signals['participation_signals']):
                    signals['primary_signal'] = 'strong_negative_participation'
                else:
                    signals['primary_signal'] = 'mixed_participation_signals'
            elif signals['signal_strength'] > 0.3:
                if any('bullish' in sig or 'improving' in sig for sig in signals['participation_signals']):
                    signals['primary_signal'] = 'positive_participation_bias'
                elif any('bearish' in sig or 'deteriorating' in sig for sig in signals['participation_signals']):
                    signals['primary_signal'] = 'negative_participation_bias'
                else:
                    signals['primary_signal'] = 'moderate_participation_change'
            elif signals['signal_strength'] > 0.1:
                signals['primary_signal'] = 'weak_participation_signal'
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating participation signals: {e}")
            return {'primary_signal': 'neutral', 'signal_strength': 0.0}
    
    def _calculate_participation_breadth_score(self, participation_metrics: Dict[str, Any], participation_quality: Dict[str, Any]) -> float:
        """Calculate overall breadth score from participation analysis"""
        try:
            score = 0.5  # Base neutral score
            
            # Overall participation contribution (40%)
            overall_participation = participation_metrics.get('overall_participation', 0.0)
            score += (overall_participation - 0.5) * 0.8  # Scale to ±0.4
            
            # Quality score contribution (30%)
            quality_score = participation_quality.get('overall_quality_score', 0.5)
            score += (quality_score - 0.5) * 0.6  # Scale to ±0.3
            
            # Directional balance contribution (20%)
            advancing_ratio = participation_metrics.get('advancing_ratio', 0.5)
            # Reward both balance and strong direction
            if 0.4 <= advancing_ratio <= 0.6:
                # Balanced market
                balance_contribution = 0.1
            elif advancing_ratio > 0.7 or advancing_ratio < 0.3:
                # Strong directional bias
                balance_contribution = 0.15 * (1 if advancing_ratio > 0.5 else -1)
            else:
                # Moderate bias
                balance_contribution = 0.05 * (1 if advancing_ratio > 0.5 else -1)
            
            score += balance_contribution
            
            # Volume participation contribution (10%)
            high_volume_participation = participation_metrics.get('high_volume_participation', 0.0)
            score += (high_volume_participation - 0.2) * 0.5  # Scale around 20% baseline
            
            return max(min(float(score), 1.0), 0.0)
            
        except Exception as e:
            logger.error(f"Error calculating participation breadth score: {e}")
            return 0.5
    
    def _calculate_slope(self, data: List[float]) -> float:
        """Calculate slope of data series"""
        try:
            if len(data) < 2:
                return 0.0
            
            x = np.arange(len(data))
            y = np.array(data)
            
            # Simple linear regression slope
            if np.std(x) > 0:
                slope = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x))
            else:
                slope = 0.0
            
            return float(slope)
            
        except Exception as e:
            logger.error(f"Error calculating slope: {e}")
            return 0.0
    
    def _update_participation_history(self, participation_metrics: Dict[str, Any], volume_participation: Dict[str, Any], price_participation: Dict[str, Any]):
        """Update historical participation tracking"""
        try:
            # Update basic metrics
            self.participation_history['total_participation'].append(participation_metrics.get('overall_participation', 0.0))
            self.participation_history['active_participation'].append(participation_metrics.get('advancing_ratio', 0.5))
            
            # Update volume participation
            if volume_participation and 'volume_threshold_1.0x' in volume_participation:
                self.participation_history['volume_participation'].append(
                    volume_participation['volume_threshold_1.0x']['participation_ratio']
                )
            
            # Update price participation
            if price_participation and 'price_threshold_0.010' in price_participation:
                self.participation_history['price_participation'].append(
                    price_participation['price_threshold_0.010']['significant_participation']
                )
            
            # Trim history to window size
            for key in ['total_participation', 'active_participation', 'volume_participation', 'price_participation']:
                if len(self.participation_history[key]) > self.participation_window * 2:
                    self.participation_history[key].pop(0)
            
        except Exception as e:
            logger.error(f"Error updating participation history: {e}")
    
    def _get_default_participation_analysis(self) -> Dict[str, Any]:
        """Get default participation analysis when data is insufficient"""
        return {
            'participation_metrics': {'total_issues': 0, 'overall_participation': 0.0},
            'volume_participation': {},
            'price_participation': {},
            'sector_participation': {},
            'participation_quality': {'overall_quality_score': 0.5, 'quality_classification': 'neutral'},
            'participation_trends': {},
            'participation_signals': {'primary_signal': 'neutral', 'signal_strength': 0.0},
            'breadth_score': 0.5
        }
    
    def get_participation_summary(self) -> Dict[str, Any]:
        """Get summary of participation analysis system"""
        try:
            return {
                'history_length': len(self.participation_history['total_participation']),
                'average_participation': np.mean(self.participation_history['total_participation']) if self.participation_history['total_participation'] else 0.0,
                'quality_metrics': self.quality_metrics.copy(),
                'current_quality_score': self.quality_metrics['breadth_quality_score'],
                'analysis_config': {
                    'participation_window': self.participation_window,
                    'volume_thresholds': self.volume_thresholds,
                    'price_change_thresholds': self.price_change_thresholds
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting participation summary: {e}")
            return {'status': 'error', 'error': str(e)}