"""
Cumulative Multi-Strike OI Analysis Engine for Component 3

This module implements cumulative OI summation across ATM ±7 strikes using actual
strike intervals (NIFTY: ₹50, BANKNIFTY: ₹100) with OI velocity and acceleration 
calculations for detecting momentum shifts in institutional positioning.

As per story requirements:
- Implement cumulative OI summation across ATM ±7 strikes using actual strike intervals
- Extract and sum ce_close prices across ATM ±7 strikes for institutional price impact analysis  
- Extract and sum pe_close prices across ATM ±7 strikes for comprehensive price correlation
- Create OI velocity calculations using time-series OI changes across strike ranges
- Add OI acceleration analysis for detecting momentum shifts in institutional positioning
- Implement symbol-specific OI behavior learning using actual NIFTY/BANKNIFTY OI distribution patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CumulativeOIMetrics:
    """Data class for cumulative OI metrics across ATM ±7 strikes."""
    
    # Cumulative OI metrics
    cumulative_ce_oi: float = 0.0
    cumulative_pe_oi: float = 0.0
    cumulative_total_oi: float = 0.0
    net_oi_bias: float = 0.0
    
    # Cumulative price metrics
    cumulative_ce_price: float = 0.0
    cumulative_pe_price: float = 0.0
    net_price_bias: float = 0.0
    
    # OI velocity and acceleration
    oi_velocity_ce: float = 0.0
    oi_velocity_pe: float = 0.0
    oi_acceleration_ce: float = 0.0
    oi_acceleration_pe: float = 0.0
    
    # Correlation metrics
    oi_price_correlation_ce: float = 0.0
    oi_price_correlation_pe: float = 0.0
    underlying_correlation: float = 0.0
    
    # Institutional flow indicators
    total_oi_concentration: float = 0.0
    institutional_flow_score: float = 0.0
    momentum_shift_detected: bool = False


class CumulativeMultiStrikeAnalyzer:
    """
    Cumulative Multi-Strike OI Analysis Engine for institutional positioning analysis.
    
    Analyzes OI patterns across ATM ±7 strikes using symbol-specific intervals:
    - NIFTY: ₹50 strike intervals  
    - BANKNIFTY: ₹100 strike intervals
    
    Calculates velocity, acceleration, and momentum shifts for institutional flow detection.
    """
    
    def __init__(self, symbol: str = 'NIFTY', strikes_range: int = 7):
        """
        Initialize the Cumulative Multi-Strike OI Analyzer.
        
        Args:
            symbol: Symbol name (NIFTY/BANKNIFTY) for strike interval calculation
            strikes_range: Number of strikes above/below ATM (default: ±7)
        """
        self.symbol = symbol.upper()
        self.strikes_range = strikes_range
        
        # Symbol-specific strike intervals (as per story requirements)
        self.strike_intervals = {
            'NIFTY': 50,      # ₹50 intervals
            'BANKNIFTY': 100  # ₹100 intervals
        }
        
        self.strike_interval = self.strike_intervals.get(self.symbol, 50)
        
        # Historical data for velocity/acceleration calculations
        self.historical_data = []
        self.velocity_window = 3  # Periods for velocity calculation
        self.acceleration_window = 2  # Periods for acceleration calculation
        
        # OI behavior patterns learning
        self.oi_distribution_patterns = {}
        self.learned_behaviors = {}
        
        logger.info(f"Initialized CumulativeMultiStrikeAnalyzer for {self.symbol} "
                   f"(±{strikes_range} strikes, ₹{self.strike_interval} intervals)")
    
    def analyze_cumulative_oi(self, df: pd.DataFrame, 
                             timestamp: Optional[datetime] = None) -> CumulativeOIMetrics:
        """
        Analyze cumulative OI across ATM ±7 strikes with velocity and acceleration.
        
        Args:
            df: DataFrame with OI and price data
            timestamp: Timestamp for time-series analysis (optional)
            
        Returns:
            CumulativeOIMetrics object with all calculated metrics
        """
        logger.info(f"Analyzing cumulative OI across ATM ±{self.strikes_range} strikes")
        
        try:
            # Filter for ATM ±7 strikes range
            atm_strikes_data = self._filter_atm_strikes_range(df)
            
            if atm_strikes_data.empty:
                logger.warning("No ATM strikes data found")
                return CumulativeOIMetrics()
            
            # Calculate cumulative OI metrics
            cumulative_oi = self._calculate_cumulative_oi_metrics(atm_strikes_data)
            
            # Calculate cumulative price metrics  
            cumulative_price = self._calculate_cumulative_price_metrics(atm_strikes_data)
            
            # Calculate OI-price correlations
            correlations = self._calculate_oi_price_correlations(atm_strikes_data)
            
            # Calculate velocity and acceleration if historical data available
            velocity_metrics = self._calculate_velocity_acceleration(cumulative_oi, timestamp)
            
            # Calculate institutional flow indicators
            institutional_metrics = self._calculate_institutional_flow_indicators(
                atm_strikes_data, cumulative_oi, cumulative_price
            )
            
            # Build comprehensive metrics object
            metrics = CumulativeOIMetrics(
                # Cumulative OI
                cumulative_ce_oi=cumulative_oi['cumulative_ce_oi'],
                cumulative_pe_oi=cumulative_oi['cumulative_pe_oi'],
                cumulative_total_oi=cumulative_oi['cumulative_total_oi'],
                net_oi_bias=cumulative_oi['net_oi_bias'],
                
                # Cumulative price  
                cumulative_ce_price=cumulative_price['cumulative_ce_price'],
                cumulative_pe_price=cumulative_price['cumulative_pe_price'],
                net_price_bias=cumulative_price['net_price_bias'],
                
                # Velocity and acceleration
                oi_velocity_ce=velocity_metrics['oi_velocity_ce'],
                oi_velocity_pe=velocity_metrics['oi_velocity_pe'],
                oi_acceleration_ce=velocity_metrics['oi_acceleration_ce'],
                oi_acceleration_pe=velocity_metrics['oi_acceleration_pe'],
                
                # Correlations
                oi_price_correlation_ce=correlations['oi_price_correlation_ce'],
                oi_price_correlation_pe=correlations['oi_price_correlation_pe'],
                underlying_correlation=correlations['underlying_correlation'],
                
                # Institutional flow
                total_oi_concentration=institutional_metrics['total_oi_concentration'],
                institutional_flow_score=institutional_metrics['institutional_flow_score'],
                momentum_shift_detected=institutional_metrics['momentum_shift_detected']
            )
            
            # Update historical data for future calculations
            self._update_historical_data(metrics, timestamp)
            
            # Learn OI behavior patterns
            self._learn_oi_behavior_patterns(atm_strikes_data, metrics)
            
            logger.info(f"Cumulative OI analysis complete: "
                       f"Total OI={metrics.cumulative_total_oi:,.0f}, "
                       f"Institutional Score={metrics.institutional_flow_score:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Cumulative OI analysis failed: {str(e)}")
            raise
    
    def analyze_symbol_specific_behavior(self, historical_data: List[pd.DataFrame]) -> Dict[str, any]:
        """
        Implement symbol-specific OI behavior learning using actual NIFTY/BANKNIFTY patterns.
        
        Args:
            historical_data: List of historical DataFrames for pattern learning
            
        Returns:
            Dictionary with learned behavior patterns
        """
        logger.info(f"Learning {self.symbol}-specific OI behavior patterns from {len(historical_data)} periods")
        
        try:
            behavior_patterns = {
                'symbol': self.symbol,
                'strike_interval': self.strike_interval,
                'total_periods_analyzed': len(historical_data),
                'oi_distribution_patterns': {},
                'velocity_patterns': {},
                'institutional_flow_patterns': {},
                'momentum_shift_patterns': {}
            }
            
            all_metrics = []
            
            # Analyze each historical period
            for i, df in enumerate(historical_data):
                try:
                    metrics = self.analyze_cumulative_oi(df)
                    all_metrics.append(metrics)
                except Exception as e:
                    logger.warning(f"Failed to analyze period {i}: {str(e)}")
                    continue
            
            if not all_metrics:
                logger.warning("No valid metrics calculated from historical data")
                return behavior_patterns
            
            # Learn OI distribution patterns
            behavior_patterns['oi_distribution_patterns'] = self._learn_oi_distribution_patterns(all_metrics)
            
            # Learn velocity patterns
            behavior_patterns['velocity_patterns'] = self._learn_velocity_patterns(all_metrics)
            
            # Learn institutional flow patterns
            behavior_patterns['institutional_flow_patterns'] = self._learn_institutional_patterns(all_metrics)
            
            # Learn momentum shift patterns
            behavior_patterns['momentum_shift_patterns'] = self._learn_momentum_shift_patterns(all_metrics)
            
            # Store learned behaviors for future use
            self.learned_behaviors = behavior_patterns
            
            logger.info(f"Symbol-specific behavior learning complete: "
                       f"Learned {len(behavior_patterns)} pattern types")
            
            return behavior_patterns
            
        except Exception as e:
            logger.error(f"Symbol-specific behavior learning failed: {str(e)}")
            raise
    
    def detect_momentum_shifts(self, current_metrics: CumulativeOIMetrics, 
                              threshold: float = 2.0) -> Dict[str, any]:
        """
        Detect momentum shifts in institutional positioning using OI acceleration analysis.
        
        Args:
            current_metrics: Current cumulative OI metrics
            threshold: Standard deviations threshold for momentum shift detection
            
        Returns:
            Dictionary with momentum shift analysis
        """
        logger.info("Detecting momentum shifts in institutional positioning")
        
        try:
            momentum_analysis = {
                'momentum_shift_detected': False,
                'ce_momentum_shift': False,
                'pe_momentum_shift': False,
                'shift_magnitude': 0.0,
                'shift_direction': 'neutral',
                'confidence_score': 0.0,
                'shift_type': 'none'
            }
            
            if len(self.historical_data) < self.acceleration_window:
                logger.info("Insufficient historical data for momentum shift detection")
                return momentum_analysis
            
            # Calculate historical acceleration statistics
            ce_accelerations = [m.oi_acceleration_ce for m in self.historical_data[-10:]]
            pe_accelerations = [m.oi_acceleration_pe for m in self.historical_data[-10:]]
            
            if not ce_accelerations or not pe_accelerations:
                return momentum_analysis
            
            # Calculate statistical thresholds
            ce_mean = np.mean(ce_accelerations)
            ce_std = np.std(ce_accelerations)
            pe_mean = np.mean(pe_accelerations)
            pe_std = np.std(pe_accelerations)
            
            # Detect CE momentum shift
            if ce_std > 0:
                ce_z_score = (current_metrics.oi_acceleration_ce - ce_mean) / ce_std
                if abs(ce_z_score) > threshold:
                    momentum_analysis['ce_momentum_shift'] = True
                    
            # Detect PE momentum shift
            if pe_std > 0:
                pe_z_score = (current_metrics.oi_acceleration_pe - pe_mean) / pe_std
                if abs(pe_z_score) > threshold:
                    momentum_analysis['pe_momentum_shift'] = True
            
            # Overall momentum shift detection
            momentum_analysis['momentum_shift_detected'] = (
                momentum_analysis['ce_momentum_shift'] or 
                momentum_analysis['pe_momentum_shift']
            )
            
            if momentum_analysis['momentum_shift_detected']:
                # Determine shift characteristics
                total_acceleration = current_metrics.oi_acceleration_ce + current_metrics.oi_acceleration_pe
                momentum_analysis['shift_magnitude'] = abs(total_acceleration)
                
                if total_acceleration > 0:
                    momentum_analysis['shift_direction'] = 'bullish'
                    momentum_analysis['shift_type'] = 'institutional_accumulation'
                else:
                    momentum_analysis['shift_direction'] = 'bearish'
                    momentum_analysis['shift_type'] = 'institutional_distribution'
                
                # Calculate confidence based on magnitude and consistency
                velocity_consistency = self._calculate_velocity_consistency(current_metrics)
                momentum_analysis['confidence_score'] = min(1.0, momentum_analysis['shift_magnitude'] * velocity_consistency)
                
                logger.info(f"Momentum shift detected: {momentum_analysis['shift_direction']} "
                           f"({momentum_analysis['shift_type']}, confidence: {momentum_analysis['confidence_score']:.3f})")
            
            return momentum_analysis
            
        except Exception as e:
            logger.error(f"Momentum shift detection failed: {str(e)}")
            return {'momentum_shift_detected': False, 'error': str(e)}
    
    def get_strike_range_analysis(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Get detailed analysis of the ATM ±7 strikes range for institutional positioning.
        
        Args:
            df: DataFrame with strike and OI data
            
        Returns:
            Dictionary with strike range analysis
        """
        logger.info(f"Analyzing ATM ±{self.strikes_range} strikes range")
        
        try:
            analysis = {
                'symbol': self.symbol,
                'strike_interval': self.strike_interval,
                'strikes_range': self.strikes_range,
                'atm_analysis': {},
                'itm_analysis': {},
                'otm_analysis': {},
                'range_distribution': {},
                'concentration_metrics': {}
            }
            
            # Filter ATM strikes data
            atm_data = self._filter_atm_strikes_range(df)
            
            if atm_data.empty:
                logger.warning("No ATM strikes data available for range analysis")
                return analysis
            
            # Analyze ATM strikes (where both call and put are ATM)
            atm_strikes = atm_data[
                (atm_data['call_strike_type'] == 'ATM') & 
                (atm_data['put_strike_type'] == 'ATM')
            ]
            
            if not atm_strikes.empty:
                analysis['atm_analysis'] = {
                    'count': len(atm_strikes),
                    'total_ce_oi': atm_strikes['ce_oi'].sum(),
                    'total_pe_oi': atm_strikes['pe_oi'].sum(),
                    'total_ce_volume': atm_strikes['ce_volume'].sum(),
                    'total_pe_volume': atm_strikes['pe_volume'].sum(),
                    'avg_ce_price': atm_strikes['ce_close'].mean(),
                    'avg_pe_price': atm_strikes['pe_close'].mean()
                }
            
            # Analyze ITM strikes (ITM1 through ITM7)
            itm_strikes = atm_data[
                atm_data['call_strike_type'].str.contains('ITM', na=False) |
                atm_data['put_strike_type'].str.contains('ITM', na=False)
            ]
            
            if not itm_strikes.empty:
                analysis['itm_analysis'] = {
                    'count': len(itm_strikes),
                    'total_ce_oi': itm_strikes['ce_oi'].sum(),
                    'total_pe_oi': itm_strikes['pe_oi'].sum(),
                    'oi_distribution': self._calculate_itm_oi_distribution(itm_strikes)
                }
            
            # Analyze OTM strikes (OTM1 through OTM7)
            otm_strikes = atm_data[
                atm_data['call_strike_type'].str.contains('OTM', na=False) |
                atm_data['put_strike_type'].str.contains('OTM', na=False)
            ]
            
            if not otm_strikes.empty:
                analysis['otm_analysis'] = {
                    'count': len(otm_strikes),
                    'total_ce_oi': otm_strikes['ce_oi'].sum(),
                    'total_pe_oi': otm_strikes['pe_oi'].sum(),
                    'oi_distribution': self._calculate_otm_oi_distribution(otm_strikes)
                }
            
            # Calculate range distribution metrics
            total_oi = atm_data['ce_oi'].sum() + atm_data['pe_oi'].sum()
            if total_oi > 0:
                analysis['range_distribution'] = {
                    'total_oi_in_range': total_oi,
                    'atm_oi_percentage': (analysis['atm_analysis'].get('total_ce_oi', 0) + 
                                        analysis['atm_analysis'].get('total_pe_oi', 0)) / total_oi * 100,
                    'itm_oi_percentage': (analysis['itm_analysis'].get('total_ce_oi', 0) + 
                                        analysis['itm_analysis'].get('total_pe_oi', 0)) / total_oi * 100,
                    'otm_oi_percentage': (analysis['otm_analysis'].get('total_ce_oi', 0) + 
                                        analysis['otm_analysis'].get('total_pe_oi', 0)) / total_oi * 100
                }
            
            # Calculate concentration metrics
            analysis['concentration_metrics'] = self._calculate_concentration_metrics(atm_data)
            
            logger.info(f"Strike range analysis complete: {len(atm_data)} strikes analyzed")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Strike range analysis failed: {str(e)}")
            raise
    
    # Private helper methods
    
    def _filter_atm_strikes_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame for ATM ±strikes_range using strike type columns."""
        
        # Build list of target strike types for ATM ±7
        atm_strikes = ['ATM']
        itm_strikes = [f'ITM{i}' for i in range(1, self.strikes_range + 1)]
        otm_strikes = [f'OTM{i}' for i in range(1, self.strikes_range + 1)]
        
        target_strikes = atm_strikes + itm_strikes + otm_strikes
        
        # Filter based on strike types
        mask = (
            df['call_strike_type'].isin(target_strikes) | 
            df['put_strike_type'].isin(target_strikes)
        )
        
        return df[mask].copy()
    
    def _calculate_cumulative_oi_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate cumulative OI metrics across ATM ±7 strikes."""
        
        cumulative_ce_oi = df['ce_oi'].sum()
        cumulative_pe_oi = df['pe_oi'].sum()
        
        return {
            'cumulative_ce_oi': float(cumulative_ce_oi),
            'cumulative_pe_oi': float(cumulative_pe_oi),
            'cumulative_total_oi': float(cumulative_ce_oi + cumulative_pe_oi),
            'net_oi_bias': float(cumulative_ce_oi - cumulative_pe_oi)
        }
    
    def _calculate_cumulative_price_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate cumulative price metrics across ATM ±7 strikes."""
        
        cumulative_ce_price = df['ce_close'].sum()
        cumulative_pe_price = df['pe_close'].sum()
        
        return {
            'cumulative_ce_price': float(cumulative_ce_price),
            'cumulative_pe_price': float(cumulative_pe_price),
            'net_price_bias': float(cumulative_ce_price - cumulative_pe_price)
        }
    
    def _calculate_oi_price_correlations(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate OI-price correlations for institutional flow analysis."""
        
        correlations = {
            'oi_price_correlation_ce': 0.0,
            'oi_price_correlation_pe': 0.0,
            'underlying_correlation': 0.0
        }
        
        if len(df) > 1:
            # CE OI-Price correlation
            ce_corr = df['ce_oi'].corr(df['ce_close'])
            correlations['oi_price_correlation_ce'] = ce_corr if not pd.isna(ce_corr) else 0.0
            
            # PE OI-Price correlation
            pe_corr = df['pe_oi'].corr(df['pe_close'])
            correlations['oi_price_correlation_pe'] = pe_corr if not pd.isna(pe_corr) else 0.0
            
            # Underlying correlation (if spot data available)
            if 'spot' in df.columns:
                total_oi = df['ce_oi'] + df['pe_oi']
                underlying_corr = total_oi.corr(df['spot'])
                correlations['underlying_correlation'] = underlying_corr if not pd.isna(underlying_corr) else 0.0
        
        return correlations
    
    def _calculate_velocity_acceleration(self, cumulative_oi: Dict[str, float], 
                                       timestamp: Optional[datetime]) -> Dict[str, float]:
        """Calculate OI velocity and acceleration from historical data."""
        
        velocity_metrics = {
            'oi_velocity_ce': 0.0,
            'oi_velocity_pe': 0.0,
            'oi_acceleration_ce': 0.0,
            'oi_acceleration_pe': 0.0
        }
        
        if len(self.historical_data) < 2:
            return velocity_metrics
        
        # Get recent historical data for calculations
        recent_data = self.historical_data[-self.velocity_window:]
        
        # Calculate CE OI velocity (rate of change)
        ce_oi_values = [data.cumulative_ce_oi for data in recent_data] + [cumulative_oi['cumulative_ce_oi']]
        if len(ce_oi_values) >= 2:
            velocity_metrics['oi_velocity_ce'] = float(ce_oi_values[-1] - ce_oi_values[-2])
        
        # Calculate PE OI velocity
        pe_oi_values = [data.cumulative_pe_oi for data in recent_data] + [cumulative_oi['cumulative_pe_oi']]
        if len(pe_oi_values) >= 2:
            velocity_metrics['oi_velocity_pe'] = float(pe_oi_values[-1] - pe_oi_values[-2])
        
        # Calculate acceleration (rate of change of velocity)
        if len(self.historical_data) >= self.acceleration_window:
            recent_velocities_ce = [data.oi_velocity_ce for data in recent_data]
            recent_velocities_pe = [data.oi_velocity_pe for data in recent_data]
            
            if len(recent_velocities_ce) >= 2:
                velocity_metrics['oi_acceleration_ce'] = float(
                    velocity_metrics['oi_velocity_ce'] - recent_velocities_ce[-1]
                )
            
            if len(recent_velocities_pe) >= 2:
                velocity_metrics['oi_acceleration_pe'] = float(
                    velocity_metrics['oi_velocity_pe'] - recent_velocities_pe[-1]
                )
        
        return velocity_metrics
    
    def _calculate_institutional_flow_indicators(self, df: pd.DataFrame, 
                                               cumulative_oi: Dict[str, float], 
                                               cumulative_price: Dict[str, float]) -> Dict[str, any]:
        """Calculate institutional flow indicators and concentration metrics."""
        
        # Calculate total market OI (assuming this is representative)
        market_total_oi = df['ce_oi'].sum() + df['pe_oi'].sum() if len(df) > 0 else 1
        
        # Calculate OI concentration in our ATM ±7 range
        total_oi_concentration = cumulative_oi['cumulative_total_oi'] / market_total_oi
        
        # Calculate institutional flow score based on multiple factors
        institutional_flow_score = self._calculate_institutional_flow_score(
            cumulative_oi, cumulative_price, total_oi_concentration
        )
        
        # Detect momentum shift
        momentum_shift_detected = False
        if len(self.historical_data) >= 2:
            recent_scores = [data.institutional_flow_score for data in self.historical_data[-3:]]
            if recent_scores and len(recent_scores) >= 2:
                score_change = institutional_flow_score - np.mean(recent_scores)
                momentum_shift_detected = abs(score_change) > 0.1  # Threshold for shift detection
        
        return {
            'total_oi_concentration': total_oi_concentration,
            'institutional_flow_score': institutional_flow_score,
            'momentum_shift_detected': momentum_shift_detected
        }
    
    def _calculate_institutional_flow_score(self, cumulative_oi: Dict[str, float], 
                                          cumulative_price: Dict[str, float], 
                                          concentration: float) -> float:
        """Calculate institutional flow score based on OI and price patterns."""
        
        # Base score from OI concentration
        concentration_score = min(1.0, concentration * 2.0)  # Higher concentration = higher score
        
        # OI bias component (balanced OI vs heavy bias)
        oi_bias = abs(cumulative_oi['net_oi_bias']) / max(1, cumulative_oi['cumulative_total_oi'])
        bias_score = 1.0 - min(1.0, oi_bias)  # Lower bias = higher score (institutional hedging)
        
        # Price-OI alignment (institutional efficiency indicator)
        if cumulative_oi['cumulative_total_oi'] > 0 and cumulative_price['cumulative_ce_price'] > 0:
            price_efficiency = cumulative_price['cumulative_ce_price'] / cumulative_oi['cumulative_total_oi']
            efficiency_score = min(1.0, price_efficiency / 10.0)  # Normalize price efficiency
        else:
            efficiency_score = 0.0
        
        # Weight the components
        institutional_flow_score = (
            concentration_score * 0.4 +    # 40% weight on concentration
            bias_score * 0.35 +            # 35% weight on OI balance
            efficiency_score * 0.25        # 25% weight on price efficiency
        )
        
        return min(1.0, max(0.0, institutional_flow_score))
    
    def _update_historical_data(self, metrics: CumulativeOIMetrics, timestamp: Optional[datetime]):
        """Update historical data for velocity/acceleration calculations."""
        
        # Add timestamp if provided
        if timestamp:
            metrics.timestamp = timestamp
        
        # Add to historical data
        self.historical_data.append(metrics)
        
        # Keep only recent history (last 50 data points for efficiency)
        if len(self.historical_data) > 50:
            self.historical_data = self.historical_data[-50:]
    
    def _learn_oi_behavior_patterns(self, df: pd.DataFrame, metrics: CumulativeOIMetrics):
        """Learn symbol-specific OI behavior patterns for future analysis."""
        
        pattern_key = f"{self.symbol}_{self.strikes_range}"
        
        if pattern_key not in self.oi_distribution_patterns:
            self.oi_distribution_patterns[pattern_key] = {
                'oi_concentrations': [],
                'velocity_patterns': [],
                'acceleration_patterns': [],
                'correlation_patterns': []
            }
        
        # Store current patterns
        patterns = self.oi_distribution_patterns[pattern_key]
        patterns['oi_concentrations'].append(metrics.total_oi_concentration)
        patterns['velocity_patterns'].append((metrics.oi_velocity_ce, metrics.oi_velocity_pe))
        patterns['acceleration_patterns'].append((metrics.oi_acceleration_ce, metrics.oi_acceleration_pe))
        patterns['correlation_patterns'].append((metrics.oi_price_correlation_ce, metrics.oi_price_correlation_pe))
        
        # Keep only recent patterns for learning
        max_patterns = 100
        for pattern_list in patterns.values():
            if len(pattern_list) > max_patterns:
                pattern_list[:] = pattern_list[-max_patterns:]
    
    def _learn_oi_distribution_patterns(self, metrics_list: List[CumulativeOIMetrics]) -> Dict[str, any]:
        """Learn OI distribution patterns from historical metrics."""
        
        if not metrics_list:
            return {}
        
        oi_concentrations = [m.total_oi_concentration for m in metrics_list]
        oi_ce_values = [m.cumulative_ce_oi for m in metrics_list]
        oi_pe_values = [m.cumulative_pe_oi for m in metrics_list]
        
        return {
            'avg_concentration': np.mean(oi_concentrations),
            'std_concentration': np.std(oi_concentrations),
            'avg_ce_oi': np.mean(oi_ce_values),
            'std_ce_oi': np.std(oi_ce_values),
            'avg_pe_oi': np.mean(oi_pe_values),
            'std_pe_oi': np.std(oi_pe_values),
            'ce_pe_correlation': np.corrcoef(oi_ce_values, oi_pe_values)[0, 1] if len(oi_ce_values) > 1 else 0.0
        }
    
    def _learn_velocity_patterns(self, metrics_list: List[CumulativeOIMetrics]) -> Dict[str, any]:
        """Learn velocity patterns from historical metrics."""
        
        if not metrics_list:
            return {}
        
        velocities_ce = [m.oi_velocity_ce for m in metrics_list]
        velocities_pe = [m.oi_velocity_pe for m in metrics_list]
        
        return {
            'avg_velocity_ce': np.mean(velocities_ce),
            'std_velocity_ce': np.std(velocities_ce),
            'avg_velocity_pe': np.mean(velocities_pe),
            'std_velocity_pe': np.std(velocities_pe),
            'velocity_correlation': np.corrcoef(velocities_ce, velocities_pe)[0, 1] if len(velocities_ce) > 1 else 0.0
        }
    
    def _learn_institutional_patterns(self, metrics_list: List[CumulativeOIMetrics]) -> Dict[str, any]:
        """Learn institutional flow patterns from historical metrics."""
        
        if not metrics_list:
            return {}
        
        institutional_scores = [m.institutional_flow_score for m in metrics_list]
        momentum_shifts = [m.momentum_shift_detected for m in metrics_list]
        
        return {
            'avg_institutional_score': np.mean(institutional_scores),
            'std_institutional_score': np.std(institutional_scores),
            'momentum_shift_frequency': sum(momentum_shifts) / len(momentum_shifts) if momentum_shifts else 0.0,
            'institutional_patterns_learned': len(metrics_list)
        }
    
    def _learn_momentum_shift_patterns(self, metrics_list: List[CumulativeOIMetrics]) -> Dict[str, any]:
        """Learn momentum shift patterns from historical metrics."""
        
        if not metrics_list:
            return {}
        
        accelerations_ce = [m.oi_acceleration_ce for m in metrics_list]
        accelerations_pe = [m.oi_acceleration_pe for m in metrics_list]
        momentum_shifts = [m.momentum_shift_detected for m in metrics_list]
        
        return {
            'avg_acceleration_ce': np.mean(accelerations_ce),
            'std_acceleration_ce': np.std(accelerations_ce),
            'avg_acceleration_pe': np.mean(accelerations_pe),
            'std_acceleration_pe': np.std(accelerations_pe),
            'shift_detection_threshold': np.std(accelerations_ce + accelerations_pe) * 2.0,
            'historical_shift_rate': sum(momentum_shifts) / len(momentum_shifts) if momentum_shifts else 0.0
        }
    
    def _calculate_velocity_consistency(self, metrics: CumulativeOIMetrics) -> float:
        """Calculate velocity consistency for momentum shift confidence scoring."""
        
        if len(self.historical_data) < 3:
            return 0.5  # Default moderate consistency
        
        recent_velocities_ce = [data.oi_velocity_ce for data in self.historical_data[-3:]]
        recent_velocities_pe = [data.oi_velocity_pe for data in self.historical_data[-3:]]
        
        # Calculate consistency as inverse of velocity standard deviation
        ce_consistency = 1.0 / (1.0 + np.std(recent_velocities_ce))
        pe_consistency = 1.0 / (1.0 + np.std(recent_velocities_pe))
        
        return (ce_consistency + pe_consistency) / 2.0
    
    def _calculate_itm_oi_distribution(self, itm_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate ITM OI distribution across ITM1-ITM7."""
        
        distribution = {}
        
        for i in range(1, self.strikes_range + 1):
            itm_level = f'ITM{i}'
            itm_rows = itm_data[
                (itm_data['call_strike_type'] == itm_level) |
                (itm_data['put_strike_type'] == itm_level)
            ]
            
            if not itm_rows.empty:
                distribution[itm_level] = {
                    'ce_oi': itm_rows['ce_oi'].sum(),
                    'pe_oi': itm_rows['pe_oi'].sum(),
                    'total_oi': itm_rows['ce_oi'].sum() + itm_rows['pe_oi'].sum()
                }
        
        return distribution
    
    def _calculate_otm_oi_distribution(self, otm_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate OTM OI distribution across OTM1-OTM7."""
        
        distribution = {}
        
        for i in range(1, self.strikes_range + 1):
            otm_level = f'OTM{i}'
            otm_rows = otm_data[
                (otm_data['call_strike_type'] == otm_level) |
                (otm_data['put_strike_type'] == otm_level)
            ]
            
            if not otm_rows.empty:
                distribution[otm_level] = {
                    'ce_oi': otm_rows['ce_oi'].sum(),
                    'pe_oi': otm_rows['pe_oi'].sum(),
                    'total_oi': otm_rows['ce_oi'].sum() + otm_rows['pe_oi'].sum()
                }
        
        return distribution
    
    def _calculate_concentration_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate OI concentration metrics across the strike range."""
        
        total_oi = df['ce_oi'].sum() + df['pe_oi'].sum()
        
        if total_oi == 0:
            return {'herfindahl_index': 0.0, 'gini_coefficient': 0.0}
        
        # Calculate Herfindahl-Hirschman Index for OI concentration
        strike_ois = df.groupby('strike')[['ce_oi', 'pe_oi']].sum().sum(axis=1)
        market_shares = strike_ois / total_oi
        hhi = (market_shares ** 2).sum()
        
        # Calculate Gini coefficient for OI distribution
        sorted_ois = np.sort(strike_ois.values)
        n = len(sorted_ois)
        cumulative_ois = np.cumsum(sorted_ois)
        gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_ois)) / (n * cumulative_ois[-1]) - (n + 1) / n
        
        return {
            'herfindahl_index': hhi,
            'gini_coefficient': gini,
            'total_strikes': len(strike_ois),
            'concentration_ratio_top5': strike_ois.nlargest(5).sum() / total_oi if len(strike_ois) >= 5 else 1.0
        }