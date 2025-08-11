"""
Institutional Flow Detection System for Component 3

This module implements institutional flow detection using volume-OI divergence analysis,
smart money positioning identification, liquidity absorption detection, and weighted
institutional flow scoring for sophisticated market regime analysis.

As per story requirements:
- Volume-OI divergence analysis: Detect divergence between volume flows and OI changes
- Smart money positioning: Identify institutional accumulation/distribution using OI+volume correlation patterns  
- Liquidity absorption detection: Analyze large OI changes with minimal price impact as institutional flow signals
- Institutional flow scoring: Create institutional_flow_score using weighted OI-volume analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import scipy.stats as stats

logger = logging.getLogger(__name__)


class InstitutionalFlowType(Enum):
    """Enum for institutional flow types."""
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    HEDGING = "hedging"
    ARBITRAGE = "arbitrage"
    NEUTRAL = "neutral"


class DivergenceType(Enum):
    """Enum for volume-OI divergence types."""
    BULLISH_DIVERGENCE = "bullish_divergence"
    BEARISH_DIVERGENCE = "bearish_divergence"
    NO_DIVERGENCE = "no_divergence"
    INSTITUTIONAL_HEDGING = "institutional_hedging"


@dataclass
class InstitutionalFlowMetrics:
    """Data class for institutional flow detection metrics."""
    
    # Volume-OI Divergence Analysis
    volume_oi_correlation: float = 0.0
    divergence_type: DivergenceType = DivergenceType.NO_DIVERGENCE
    divergence_strength: float = 0.0
    divergence_duration: int = 0
    
    # Smart Money Positioning  
    smart_money_positioning: InstitutionalFlowType = InstitutionalFlowType.NEUTRAL
    positioning_confidence: float = 0.0
    accumulation_strength: float = 0.0
    distribution_strength: float = 0.0
    
    # Liquidity Absorption Detection
    liquidity_absorption_detected: bool = False
    absorption_magnitude: float = 0.0
    price_impact_efficiency: float = 0.0
    large_block_activity: bool = False
    
    # Institutional Flow Score
    institutional_flow_score: float = 0.0
    flow_score_components: Dict[str, float] = None
    institutional_confidence: float = 0.0
    
    # Supporting Metrics
    volume_weighted_oi_change: float = 0.0
    oi_momentum: float = 0.0
    cross_strike_consistency: float = 0.0
    temporal_persistence: float = 0.0


class InstitutionalFlowDetector:
    """
    Institutional Flow Detection System for sophisticated institutional activity analysis.
    
    Detects institutional flows through:
    1. Volume-OI divergence patterns
    2. Smart money positioning identification 
    3. Liquidity absorption analysis
    4. Weighted institutional flow scoring
    """
    
    def __init__(self, divergence_threshold: float = 0.3, absorption_threshold: float = 2.0):
        """
        Initialize the Institutional Flow Detector.
        
        Args:
            divergence_threshold: Correlation threshold for volume-OI divergence detection
            absorption_threshold: Standard deviation threshold for liquidity absorption detection
        """
        self.divergence_threshold = divergence_threshold
        self.absorption_threshold = absorption_threshold
        
        # Historical data for persistence analysis
        self.historical_flows = []
        self.flow_patterns = {}
        
        # Weighted scoring components (as per story requirements)
        self.score_weights = {
            'volume_oi_divergence': 0.25,    # 25% weight on divergence analysis
            'smart_money_positioning': 0.30,  # 30% weight on positioning analysis
            'liquidity_absorption': 0.25,     # 25% weight on absorption analysis
            'cross_strike_consistency': 0.20  # 20% weight on consistency across strikes
        }
        
        logger.info(f"Initialized InstitutionalFlowDetector with divergence threshold: {divergence_threshold}, "
                   f"absorption threshold: {absorption_threshold}")
    
    def detect_institutional_flows(self, df: pd.DataFrame, 
                                 previous_period: Optional[pd.DataFrame] = None) -> InstitutionalFlowMetrics:
        """
        Detect institutional flows using comprehensive volume-OI analysis.
        
        Args:
            df: Current period DataFrame with OI, volume, and price data
            previous_period: Previous period DataFrame for temporal analysis (optional)
            
        Returns:
            InstitutionalFlowMetrics with comprehensive flow analysis
        """
        logger.info("Detecting institutional flows using volume-OI analysis")
        
        try:
            # Initialize metrics
            metrics = InstitutionalFlowMetrics()
            metrics.flow_score_components = {}
            
            # 1. Volume-OI Divergence Analysis
            logger.info("Analyzing volume-OI divergence patterns")
            divergence_metrics = self._analyze_volume_oi_divergence(df, previous_period)
            
            metrics.volume_oi_correlation = divergence_metrics['correlation']
            metrics.divergence_type = divergence_metrics['divergence_type']
            metrics.divergence_strength = divergence_metrics['strength']
            metrics.divergence_duration = divergence_metrics['duration']
            metrics.flow_score_components['divergence'] = divergence_metrics['score']
            
            # 2. Smart Money Positioning Analysis
            logger.info("Identifying smart money positioning patterns")
            positioning_metrics = self._identify_smart_money_positioning(df, previous_period)
            
            metrics.smart_money_positioning = positioning_metrics['positioning_type']
            metrics.positioning_confidence = positioning_metrics['confidence']
            metrics.accumulation_strength = positioning_metrics['accumulation_strength']
            metrics.distribution_strength = positioning_metrics['distribution_strength']
            metrics.flow_score_components['positioning'] = positioning_metrics['score']
            
            # 3. Liquidity Absorption Detection
            logger.info("Analyzing liquidity absorption patterns")
            absorption_metrics = self._detect_liquidity_absorption(df, previous_period)
            
            metrics.liquidity_absorption_detected = absorption_metrics['detected']
            metrics.absorption_magnitude = absorption_metrics['magnitude']
            metrics.price_impact_efficiency = absorption_metrics['price_efficiency']
            metrics.large_block_activity = absorption_metrics['large_blocks']
            metrics.flow_score_components['absorption'] = absorption_metrics['score']
            
            # 4. Cross-Strike Consistency Analysis
            logger.info("Analyzing cross-strike flow consistency")
            consistency_metrics = self._analyze_cross_strike_consistency(df)
            
            metrics.cross_strike_consistency = consistency_metrics['consistency']
            metrics.flow_score_components['consistency'] = consistency_metrics['score']
            
            # 5. Calculate Comprehensive Institutional Flow Score
            metrics.institutional_flow_score = self._calculate_institutional_flow_score(
                metrics.flow_score_components
            )
            
            # 6. Calculate Supporting Metrics
            supporting_metrics = self._calculate_supporting_metrics(df, previous_period)
            metrics.volume_weighted_oi_change = supporting_metrics['volume_weighted_oi_change']
            metrics.oi_momentum = supporting_metrics['oi_momentum']
            metrics.temporal_persistence = supporting_metrics['temporal_persistence']
            
            # 7. Calculate Overall Institutional Confidence
            metrics.institutional_confidence = self._calculate_institutional_confidence(metrics)
            
            # Store for historical analysis
            self._update_historical_flows(metrics)
            
            logger.info(f"Institutional flow detection complete: "
                       f"Score={metrics.institutional_flow_score:.3f}, "
                       f"Type={metrics.smart_money_positioning.value}, "
                       f"Confidence={metrics.institutional_confidence:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Institutional flow detection failed: {str(e)}")
            raise
    
    def analyze_volume_oi_divergence_patterns(self, df: pd.DataFrame, 
                                            lookback_periods: int = 10) -> Dict[str, any]:
        """
        Analyze volume-OI divergence patterns over multiple periods for trend identification.
        
        Args:
            df: DataFrame with time-series OI and volume data
            lookback_periods: Number of periods to analyze for patterns
            
        Returns:
            Dictionary with divergence pattern analysis
        """
        logger.info(f"Analyzing volume-OI divergence patterns over {lookback_periods} periods")
        
        try:
            analysis = {
                'pattern_type': 'unknown',
                'pattern_strength': 0.0,
                'pattern_persistence': 0.0,
                'institutional_implications': {},
                'divergence_periods': [],
                'correlation_trend': 'neutral'
            }
            
            if 'trade_time' not in df.columns:
                logger.warning("No time column found for divergence pattern analysis")
                return analysis
            
            # Sort by time and create periods
            df_sorted = df.sort_values('trade_time')
            
            # Calculate rolling correlations
            window_size = max(50, len(df_sorted) // lookback_periods)
            rolling_correlations = []
            
            for i in range(0, len(df_sorted) - window_size + 1, window_size // 2):
                window_data = df_sorted.iloc[i:i + window_size]
                
                if len(window_data) >= 10:  # Minimum data for correlation
                    total_volume = window_data['ce_volume'] + window_data['pe_volume']
                    total_oi = window_data['ce_oi'] + window_data['pe_oi']
                    
                    # Calculate volume and OI changes
                    volume_change = total_volume.diff().fillna(0)
                    oi_change = total_oi.diff().fillna(0)
                    
                    if volume_change.std() > 0 and oi_change.std() > 0:
                        correlation = volume_change.corr(oi_change)
                        rolling_correlations.append({
                            'period': i,
                            'correlation': correlation if not pd.isna(correlation) else 0.0,
                            'start_time': window_data['trade_time'].iloc[0],
                            'end_time': window_data['trade_time'].iloc[-1],
                            'avg_volume': total_volume.mean(),
                            'avg_oi': total_oi.mean()
                        })
            
            if not rolling_correlations:
                return analysis
            
            # Analyze correlation patterns
            correlations = [r['correlation'] for r in rolling_correlations]
            
            # Determine pattern type
            avg_correlation = np.mean(correlations)
            correlation_std = np.std(correlations)
            
            if avg_correlation < -self.divergence_threshold:
                analysis['pattern_type'] = 'persistent_divergence'
                analysis['institutional_implications'] = {
                    'interpretation': 'Sustained institutional activity with controlled volume',
                    'likely_strategy': 'Stealth accumulation or distribution',
                    'market_impact': 'Institutional positioning without market disruption'
                }
            elif avg_correlation > 0.5:
                analysis['pattern_type'] = 'aligned_flow'
                analysis['institutional_implications'] = {
                    'interpretation': 'Volume and OI moving together',
                    'likely_strategy': 'Retail-driven activity or momentum trading',
                    'market_impact': 'Transparent market activity'
                }
            else:
                analysis['pattern_type'] = 'mixed_patterns'
                analysis['institutional_implications'] = {
                    'interpretation': 'Variable institutional activity patterns',
                    'likely_strategy': 'Complex multi-strategy positioning',
                    'market_impact': 'Unclear directional intent'
                }
            
            # Calculate pattern strength and persistence
            analysis['pattern_strength'] = abs(avg_correlation)
            analysis['pattern_persistence'] = 1.0 - min(1.0, correlation_std / abs(avg_correlation + 1e-6))
            
            # Identify specific divergence periods
            analysis['divergence_periods'] = [
                r for r in rolling_correlations 
                if abs(r['correlation']) > self.divergence_threshold
            ]
            
            # Determine correlation trend
            if len(correlations) >= 3:
                recent_avg = np.mean(correlations[-3:])
                earlier_avg = np.mean(correlations[:3])
                
                if recent_avg - earlier_avg > 0.2:
                    analysis['correlation_trend'] = 'strengthening'
                elif recent_avg - earlier_avg < -0.2:
                    analysis['correlation_trend'] = 'weakening'
                else:
                    analysis['correlation_trend'] = 'stable'
            
            logger.info(f"Divergence pattern analysis complete: {analysis['pattern_type']} "
                       f"(strength: {analysis['pattern_strength']:.3f})")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Volume-OI divergence pattern analysis failed: {str(e)}")
            raise
    
    def detect_smart_money_accumulation(self, df: pd.DataFrame, 
                                      accumulation_threshold: float = 1.5) -> Dict[str, any]:
        """
        Detect smart money accumulation patterns using sustained OI building with controlled volume.
        
        Args:
            df: DataFrame with OI and volume data
            accumulation_threshold: Threshold for accumulation detection (in standard deviations)
            
        Returns:
            Dictionary with accumulation analysis
        """
        logger.info(f"Detecting smart money accumulation patterns")
        
        try:
            analysis = {
                'accumulation_detected': False,
                'accumulation_strength': 0.0,
                'accumulation_pattern': 'none',
                'sustained_periods': 0,
                'volume_efficiency': 0.0,
                'stealth_score': 0.0,
                'institutional_characteristics': {}
            }
            
            if len(df) < 5:
                logger.warning("Insufficient data for accumulation detection")
                return analysis
            
            # Calculate OI and volume metrics
            total_oi = df['ce_oi'] + df['pe_oi']
            total_volume = df['ce_volume'] + df['pe_volume']
            
            # Calculate OI changes over time
            oi_changes = total_oi.diff().fillna(0)
            volume_changes = total_volume.diff().fillna(0)
            
            # Calculate rolling metrics for sustained activity detection
            window = min(5, len(df) // 2)
            
            if window >= 2:
                rolling_oi_change = oi_changes.rolling(window=window, min_periods=1).sum()
                rolling_volume_change = volume_changes.rolling(window=window, min_periods=1).sum()
                
                # Calculate volume efficiency (OI change per unit volume)
                volume_efficiency = []
                for i in range(len(df)):
                    if rolling_volume_change.iloc[i] > 0:
                        efficiency = rolling_oi_change.iloc[i] / rolling_volume_change.iloc[i]
                        volume_efficiency.append(efficiency)
                    else:
                        volume_efficiency.append(0.0)
                
                volume_efficiency = np.array(volume_efficiency)
                
                # Detect accumulation characteristics
                if len(volume_efficiency) > 0:
                    avg_efficiency = np.mean(volume_efficiency)
                    efficiency_std = np.std(volume_efficiency)
                    
                    # High efficiency indicates institutional accumulation
                    if efficiency_std > 0:
                        efficiency_score = avg_efficiency / (efficiency_std + 1e-6)
                        
                        if efficiency_score > accumulation_threshold:
                            analysis['accumulation_detected'] = True
                            analysis['accumulation_strength'] = min(1.0, efficiency_score / accumulation_threshold)
                            
                            # Determine accumulation pattern
                            consistent_periods = sum(1 for eff in volume_efficiency if eff > avg_efficiency)
                            
                            if consistent_periods >= len(volume_efficiency) * 0.7:
                                analysis['accumulation_pattern'] = 'sustained_accumulation'
                            elif consistent_periods >= len(volume_efficiency) * 0.5:
                                analysis['accumulation_pattern'] = 'periodic_accumulation'
                            else:
                                analysis['accumulation_pattern'] = 'sporadic_accumulation'
                            
                            # Calculate stealth score (low volume relative to OI increase)
                            total_oi_increase = max(0, total_oi.iloc[-1] - total_oi.iloc[0])
                            total_volume_traded = total_volume.sum()
                            
                            if total_volume_traded > 0:
                                stealth_ratio = total_oi_increase / total_volume_traded
                                analysis['stealth_score'] = min(1.0, stealth_ratio * 100)  # Normalize
                            
                            # Identify institutional characteristics
                            analysis['institutional_characteristics'] = {
                                'high_volume_efficiency': avg_efficiency > np.median(volume_efficiency) * 1.5,
                                'low_market_impact': analysis['stealth_score'] > 0.3,
                                'sustained_activity': analysis['accumulation_pattern'] == 'sustained_accumulation',
                                'likely_institutional': (
                                    analysis['stealth_score'] > 0.3 and 
                                    analysis['accumulation_strength'] > 0.5
                                )
                            }
                    
                    analysis['volume_efficiency'] = avg_efficiency
                    analysis['sustained_periods'] = sum(1 for eff in volume_efficiency if eff > 0)
            
            logger.info(f"Smart money accumulation detection complete: "
                       f"Detected={analysis['accumulation_detected']}, "
                       f"Pattern={analysis['accumulation_pattern']}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Smart money accumulation detection failed: {str(e)}")
            raise
    
    def analyze_liquidity_absorption_events(self, df: pd.DataFrame) -> List[Dict[str, any]]:
        """
        Analyze liquidity absorption events (large OI changes with minimal price impact).
        
        Args:
            df: DataFrame with OI, volume, and price data
            
        Returns:
            List of liquidity absorption events
        """
        logger.info("Analyzing liquidity absorption events")
        
        try:
            events = []
            
            if len(df) < 3:
                logger.warning("Insufficient data for liquidity absorption analysis")
                return events
            
            # Calculate changes
            total_oi = df['ce_oi'] + df['pe_oi']
            total_volume = df['ce_volume'] + df['pe_volume']
            
            oi_changes = total_oi.diff().fillna(0)
            volume_changes = total_volume.diff().fillna(0)
            
            # Calculate price impact (if price data available)
            price_impact = []
            if 'ce_close' in df.columns and 'pe_close' in df.columns:
                straddle_price = df['ce_close'] + df['pe_close']
                price_changes = straddle_price.diff().fillna(0)
                
                # Calculate price impact per unit OI change
                for i in range(len(df)):
                    if abs(oi_changes.iloc[i]) > 0:
                        impact = abs(price_changes.iloc[i]) / abs(oi_changes.iloc[i])
                        price_impact.append(impact)
                    else:
                        price_impact.append(0.0)
            else:
                price_impact = [0.0] * len(df)
            
            # Statistical thresholds for absorption detection
            oi_change_threshold = np.std(oi_changes) * self.absorption_threshold
            volume_threshold = np.std(volume_changes) * self.absorption_threshold
            
            if len(price_impact) > 0:
                price_impact_threshold = np.percentile([p for p in price_impact if p > 0], 25)
            else:
                price_impact_threshold = 0.0
            
            # Detect absorption events
            for i in range(len(df)):
                oi_change = abs(oi_changes.iloc[i])
                volume_change = volume_changes.iloc[i]
                price_impact_val = price_impact[i]
                
                # Large OI change with minimal price impact
                if (oi_change > oi_change_threshold and 
                    price_impact_val < price_impact_threshold and
                    volume_change > 0):
                    
                    # Calculate absorption characteristics
                    absorption_efficiency = oi_change / (price_impact_val + 1e-6)
                    volume_ratio = volume_change / (oi_change + 1e-6)
                    
                    event = {
                        'timestamp': df.index[i] if hasattr(df.index[i], 'strftime') else i,
                        'oi_change': oi_changes.iloc[i],
                        'volume_change': volume_change,
                        'price_impact': price_impact_val,
                        'absorption_efficiency': absorption_efficiency,
                        'volume_ratio': volume_ratio,
                        'event_magnitude': oi_change / oi_change_threshold,
                        'institutional_probability': self._calculate_institutional_probability(
                            oi_change, volume_change, price_impact_val
                        )
                    }
                    
                    # Classify absorption type
                    if event['institutional_probability'] > 0.7:
                        event['absorption_type'] = 'institutional_block'
                    elif volume_ratio > 2.0:
                        event['absorption_type'] = 'high_volume_absorption'
                    else:
                        event['absorption_type'] = 'stealth_absorption'
                    
                    events.append(event)
            
            # Sort events by institutional probability
            events.sort(key=lambda x: x['institutional_probability'], reverse=True)
            
            logger.info(f"Liquidity absorption analysis complete: {len(events)} events detected")
            
            return events
            
        except Exception as e:
            logger.error(f"Liquidity absorption analysis failed: {str(e)}")
            raise
    
    # Private helper methods
    
    def _analyze_volume_oi_divergence(self, df: pd.DataFrame, 
                                    previous_period: Optional[pd.DataFrame]) -> Dict[str, any]:
        """Analyze volume-OI divergence patterns."""
        
        # Calculate total volumes and OI
        total_volume = df['ce_volume'] + df['pe_volume']
        total_oi = df['ce_oi'] + df['pe_oi']
        
        # Calculate changes
        if previous_period is not None:
            prev_volume = previous_period['ce_volume'] + previous_period['pe_volume']
            prev_oi = previous_period['ce_oi'] + previous_period['pe_oi']
            
            volume_change = total_volume.sum() - prev_volume.sum()
            oi_change = total_oi.sum() - prev_oi.sum()
        else:
            volume_change = total_volume.diff().fillna(0)
            oi_change = total_oi.diff().fillna(0)
            
            if hasattr(volume_change, 'sum'):
                volume_change = volume_change.sum()
            if hasattr(oi_change, 'sum'):
                oi_change = oi_change.sum()
        
        # Calculate correlation
        if len(df) > 1:
            vol_changes = total_volume.diff().fillna(0)
            oi_changes = total_oi.diff().fillna(0)
            
            if vol_changes.std() > 0 and oi_changes.std() > 0:
                correlation = vol_changes.corr(oi_changes)
                correlation = correlation if not pd.isna(correlation) else 0.0
            else:
                correlation = 0.0
        else:
            correlation = 0.0
        
        # Determine divergence type
        if correlation < -self.divergence_threshold:
            if oi_change > 0 and volume_change < oi_change * 0.5:
                divergence_type = DivergenceType.BULLISH_DIVERGENCE
            elif oi_change < 0 and volume_change > abs(oi_change) * 0.5:
                divergence_type = DivergenceType.BEARISH_DIVERGENCE
            else:
                divergence_type = DivergenceType.INSTITUTIONAL_HEDGING
        else:
            divergence_type = DivergenceType.NO_DIVERGENCE
        
        # Calculate strength and duration
        strength = abs(correlation) if abs(correlation) > self.divergence_threshold else 0.0
        duration = 1  # Default duration, would need historical data for actual calculation
        
        # Calculate score component
        if divergence_type != DivergenceType.NO_DIVERGENCE:
            score = strength * 0.8 + min(1.0, duration / 5.0) * 0.2
        else:
            score = 0.0
        
        return {
            'correlation': correlation,
            'divergence_type': divergence_type,
            'strength': strength,
            'duration': duration,
            'score': score
        }
    
    def _identify_smart_money_positioning(self, df: pd.DataFrame, 
                                        previous_period: Optional[pd.DataFrame]) -> Dict[str, any]:
        """Identify smart money positioning patterns."""
        
        # Calculate OI and volume metrics
        total_oi = df['ce_oi'] + df['pe_oi']
        total_volume = df['ce_volume'] + df['pe_volume']
        
        # Analyze positioning patterns
        accumulation_analysis = self.detect_smart_money_accumulation(df)
        
        # Determine positioning type
        if accumulation_analysis['accumulation_detected']:
            if accumulation_analysis['stealth_score'] > 0.5:
                positioning_type = InstitutionalFlowType.ACCUMULATION
            else:
                positioning_type = InstitutionalFlowType.HEDGING
        else:
            # Check for distribution patterns (opposite of accumulation)
            oi_decrease = total_oi.iloc[-1] < total_oi.iloc[0] if len(total_oi) > 1 else False
            high_volume = total_volume.sum() > total_volume.quantile(0.75)
            
            if oi_decrease and high_volume:
                positioning_type = InstitutionalFlowType.DISTRIBUTION
            else:
                positioning_type = InstitutionalFlowType.NEUTRAL
        
        # Calculate confidence based on pattern strength
        confidence = accumulation_analysis.get('accumulation_strength', 0.0)
        if positioning_type == InstitutionalFlowType.DISTRIBUTION:
            confidence = 0.7  # Default confidence for distribution detection
        elif positioning_type == InstitutionalFlowType.NEUTRAL:
            confidence = 0.1
        
        # Calculate score
        if positioning_type in [InstitutionalFlowType.ACCUMULATION, InstitutionalFlowType.DISTRIBUTION]:
            score = confidence * 0.9
        elif positioning_type == InstitutionalFlowType.HEDGING:
            score = confidence * 0.6
        else:
            score = 0.0
        
        return {
            'positioning_type': positioning_type,
            'confidence': confidence,
            'accumulation_strength': accumulation_analysis.get('accumulation_strength', 0.0),
            'distribution_strength': 0.7 if positioning_type == InstitutionalFlowType.DISTRIBUTION else 0.0,
            'score': score
        }
    
    def _detect_liquidity_absorption(self, df: pd.DataFrame, 
                                   previous_period: Optional[pd.DataFrame]) -> Dict[str, any]:
        """Detect liquidity absorption patterns."""
        
        # Analyze absorption events
        absorption_events = self.analyze_liquidity_absorption_events(df)
        
        # Calculate metrics
        detected = len(absorption_events) > 0
        
        if detected:
            # Calculate aggregate metrics
            magnitudes = [event['event_magnitude'] for event in absorption_events]
            efficiencies = [event['absorption_efficiency'] for event in absorption_events]
            institutional_probs = [event['institutional_probability'] for event in absorption_events]
            
            magnitude = np.mean(magnitudes)
            price_efficiency = np.mean(efficiencies)
            large_blocks = any(event['absorption_type'] == 'institutional_block' for event in absorption_events)
            
            # Calculate score based on institutional probability and magnitude
            score = np.mean(institutional_probs) * min(1.0, magnitude)
        else:
            magnitude = 0.0
            price_efficiency = 0.0
            large_blocks = False
            score = 0.0
        
        return {
            'detected': detected,
            'magnitude': magnitude,
            'price_efficiency': price_efficiency,
            'large_blocks': large_blocks,
            'score': score,
            'events': absorption_events
        }
    
    def _analyze_cross_strike_consistency(self, df: pd.DataFrame) -> Dict[str, any]:
        """Analyze consistency of institutional flows across different strikes."""
        
        if 'call_strike_type' not in df.columns or 'put_strike_type' not in df.columns:
            logger.warning("Strike type columns not available for consistency analysis")
            return {'consistency': 0.5, 'score': 0.5}
        
        # Group by strike types and calculate flow metrics
        strike_groups = {}
        
        for strike_type in ['ATM', 'ITM1', 'ITM2', 'OTM1', 'OTM2']:
            mask = (df['call_strike_type'] == strike_type) | (df['put_strike_type'] == strike_type)
            strike_data = df[mask]
            
            if not strike_data.empty:
                total_oi = strike_data['ce_oi'] + strike_data['pe_oi']
                total_volume = strike_data['ce_volume'] + strike_data['pe_volume']
                
                # Calculate flow ratio for this strike type
                if total_volume.sum() > 0:
                    flow_ratio = total_oi.sum() / total_volume.sum()
                    strike_groups[strike_type] = flow_ratio
        
        if len(strike_groups) < 2:
            return {'consistency': 0.5, 'score': 0.5}
        
        # Calculate consistency as inverse of coefficient of variation
        flow_ratios = list(strike_groups.values())
        consistency = 1.0 - (np.std(flow_ratios) / (np.mean(flow_ratios) + 1e-6))
        consistency = max(0.0, min(1.0, consistency))
        
        return {
            'consistency': consistency,
            'score': consistency,
            'strike_flow_ratios': strike_groups
        }
    
    def _calculate_institutional_flow_score(self, components: Dict[str, float]) -> float:
        """Calculate weighted institutional flow score."""
        
        total_score = 0.0
        total_weight = 0.0
        
        for component, weight in self.score_weights.items():
            if component in components:
                total_score += components[component] * weight
                total_weight += weight
        
        if total_weight > 0:
            return total_score / total_weight
        else:
            return 0.0
    
    def _calculate_supporting_metrics(self, df: pd.DataFrame, 
                                    previous_period: Optional[pd.DataFrame]) -> Dict[str, any]:
        """Calculate supporting metrics for institutional flow analysis."""
        
        total_oi = df['ce_oi'] + df['pe_oi']
        total_volume = df['ce_volume'] + df['pe_volume']
        
        # Volume-weighted OI change
        if previous_period is not None:
            prev_oi = previous_period['ce_oi'] + previous_period['pe_oi']
            oi_change = total_oi.sum() - prev_oi.sum()
            volume_weight = total_volume.sum() / (total_volume.sum() + 1)
            volume_weighted_oi_change = oi_change * volume_weight
        else:
            volume_weighted_oi_change = 0.0
        
        # OI momentum (rate of OI change)
        if len(total_oi) > 1:
            oi_momentum = (total_oi.iloc[-1] - total_oi.iloc[0]) / len(total_oi)
        else:
            oi_momentum = 0.0
        
        # Temporal persistence (would need more historical data)
        temporal_persistence = 0.5  # Default value
        
        return {
            'volume_weighted_oi_change': volume_weighted_oi_change,
            'oi_momentum': oi_momentum,
            'temporal_persistence': temporal_persistence
        }
    
    def _calculate_institutional_confidence(self, metrics: InstitutionalFlowMetrics) -> float:
        """Calculate overall institutional confidence score."""
        
        # Combine multiple confidence indicators
        positioning_confidence = metrics.positioning_confidence
        divergence_strength = metrics.divergence_strength
        absorption_magnitude = min(1.0, metrics.absorption_magnitude)
        consistency = metrics.cross_strike_consistency
        
        # Weight the confidence components
        weights = [0.3, 0.25, 0.25, 0.2]  # positioning, divergence, absorption, consistency
        values = [positioning_confidence, divergence_strength, absorption_magnitude, consistency]
        
        confidence = sum(w * v for w, v in zip(weights, values))
        
        return min(1.0, max(0.0, confidence))
    
    def _update_historical_flows(self, metrics: InstitutionalFlowMetrics):
        """Update historical flow data for pattern learning."""
        
        self.historical_flows.append({
            'timestamp': datetime.now(),
            'score': metrics.institutional_flow_score,
            'positioning': metrics.smart_money_positioning,
            'divergence_type': metrics.divergence_type,
            'confidence': metrics.institutional_confidence
        })
        
        # Keep only recent history
        if len(self.historical_flows) > 100:
            self.historical_flows = self.historical_flows[-100:]
    
    def _calculate_institutional_probability(self, oi_change: float, 
                                          volume_change: float, 
                                          price_impact: float) -> float:
        """Calculate probability that activity is institutional."""
        
        # High OI change with low price impact suggests institutional activity
        if price_impact == 0:
            price_impact = 1e-6  # Avoid division by zero
        
        efficiency_ratio = abs(oi_change) / price_impact
        volume_ratio = abs(oi_change) / (volume_change + 1e-6)
        
        # Institutional characteristics: high efficiency, controlled volume impact
        efficiency_score = min(1.0, efficiency_ratio / 1000.0)  # Normalize
        volume_score = min(1.0, volume_ratio / 10.0)  # Normalize
        
        # Combined probability
        probability = (efficiency_score * 0.6 + volume_score * 0.4)
        
        return min(1.0, max(0.0, probability))