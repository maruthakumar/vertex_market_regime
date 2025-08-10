"""
Option Momentum - Option-based Momentum Analysis for Market Breadth
==================================================================

Analyzes momentum patterns in option markets to assess breadth dynamics.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class OptionMomentum:
    """Option momentum analyzer for market breadth assessment"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Option Momentum analyzer"""
        self.momentum_window = config.get('momentum_window', 14)
        self.velocity_window = config.get('velocity_window', 5)
        self.momentum_threshold = config.get('momentum_threshold', 0.15)
        
        # Momentum tracking
        self.momentum_history = {
            'volume_momentum': [],
            'oi_momentum': [],
            'iv_momentum': [],
            'premium_momentum': [],
            'timestamps': []
        }
        
        # Velocity tracking
        self.velocity_metrics = {
            'acceleration': [],
            'deceleration': [],
            'direction_changes': 0
        }
        
        logger.info("OptionMomentum initialized")
    
    def analyze_option_momentum(self, option_data: pd.DataFrame, historical_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Analyze comprehensive option momentum for breadth assessment
        
        Args:
            option_data: Current option market data
            historical_data: Historical option data for momentum calculation
            
        Returns:
            Dict with momentum analysis results
        """
        try:
            if option_data.empty:
                return self._get_default_momentum_analysis()
            
            # Calculate basic momentum metrics
            basic_momentum = self._calculate_basic_momentum(option_data, historical_data)
            
            # Calculate advanced momentum metrics
            advanced_momentum = self._calculate_advanced_momentum(option_data, historical_data)
            
            # Analyze momentum divergences
            momentum_divergences = self._analyze_momentum_divergences(basic_momentum, advanced_momentum)
            
            # Calculate momentum velocity and acceleration
            velocity_analysis = self._analyze_momentum_velocity(basic_momentum)
            
            # Detect momentum regimes
            momentum_regimes = self._detect_momentum_regimes(basic_momentum, advanced_momentum)
            
            # Generate momentum signals
            momentum_signals = self._generate_momentum_signals(basic_momentum, advanced_momentum, momentum_regimes)
            
            # Update historical tracking
            self._update_momentum_history(basic_momentum, advanced_momentum)
            
            return {
                'basic_momentum': basic_momentum,
                'advanced_momentum': advanced_momentum,
                'momentum_divergences': momentum_divergences,
                'velocity_analysis': velocity_analysis,
                'momentum_regimes': momentum_regimes,
                'momentum_signals': momentum_signals,
                'breadth_score': self._calculate_momentum_breadth_score(basic_momentum, advanced_momentum)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing option momentum: {e}")
            return self._get_default_momentum_analysis()
    
    def _calculate_basic_momentum(self, current_data: pd.DataFrame, historical_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Calculate basic momentum metrics"""
        try:
            momentum = {}
            
            # Volume momentum
            current_volume = current_data['volume'].sum()
            momentum['current_volume'] = float(current_volume)
            
            if historical_data is not None and not historical_data.empty:
                historical_volume = historical_data['volume'].sum()
                momentum['volume_momentum'] = float((current_volume - historical_volume) / historical_volume) if historical_volume > 0 else 0.0
            else:
                momentum['volume_momentum'] = 0.0
            
            # Open Interest momentum
            if 'oi' in current_data.columns:
                current_oi = current_data['oi'].sum()
                momentum['current_oi'] = float(current_oi)
                
                if historical_data is not None and 'oi' in historical_data.columns:
                    historical_oi = historical_data['oi'].sum()
                    momentum['oi_momentum'] = float((current_oi - historical_oi) / historical_oi) if historical_oi > 0 else 0.0
                else:
                    momentum['oi_momentum'] = 0.0
            
            # IV momentum
            if 'iv' in current_data.columns:
                current_iv_avg = current_data['iv'].mean()
                momentum['current_iv'] = float(current_iv_avg)
                
                if historical_data is not None and 'iv' in historical_data.columns:
                    historical_iv_avg = historical_data['iv'].mean()
                    momentum['iv_momentum'] = float((current_iv_avg - historical_iv_avg) / historical_iv_avg) if historical_iv_avg > 0 else 0.0
                else:
                    momentum['iv_momentum'] = 0.0
            
            # Premium momentum
            if 'premium' in current_data.columns:
                current_premium = current_data['premium'].sum()
                momentum['current_premium'] = float(current_premium)
                
                if historical_data is not None and 'premium' in historical_data.columns:
                    historical_premium = historical_data['premium'].sum()
                    momentum['premium_momentum'] = float((current_premium - historical_premium) / historical_premium) if historical_premium > 0 else 0.0
                else:
                    momentum['premium_momentum'] = 0.0
            
            return momentum
            
        except Exception as e:
            logger.error(f"Error calculating basic momentum: {e}")
            return {'volume_momentum': 0.0, 'oi_momentum': 0.0}
    
    def _calculate_advanced_momentum(self, current_data: pd.DataFrame, historical_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Calculate advanced momentum metrics"""
        try:
            momentum = {}
            
            # Strike-level momentum analysis
            if 'strike' in current_data.columns:
                momentum['strike_momentum'] = self._analyze_strike_momentum(current_data, historical_data)
            
            # Moneyness momentum
            if 'moneyness' in current_data.columns:
                momentum['moneyness_momentum'] = self._analyze_moneyness_momentum(current_data, historical_data)
            
            # Term structure momentum
            if 'dte' in current_data.columns:
                momentum['term_momentum'] = self._analyze_term_momentum(current_data, historical_data)
            
            # Option type momentum
            if 'option_type' in current_data.columns:
                momentum['option_type_momentum'] = self._analyze_option_type_momentum(current_data, historical_data)
            
            # Sector momentum (if sector data available)
            if 'sector' in current_data.columns:
                momentum['sector_momentum'] = self._analyze_sector_momentum(current_data, historical_data)
            
            return momentum
            
        except Exception as e:
            logger.error(f"Error calculating advanced momentum: {e}")
            return {}
    
    def _analyze_strike_momentum(self, current_data: pd.DataFrame, historical_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Analyze momentum by strike levels"""
        try:
            strike_momentum = {}
            
            # Group by strike and calculate momentum
            current_strikes = current_data.groupby('strike')['volume'].sum()
            
            if historical_data is not None:
                historical_strikes = historical_data.groupby('strike')['volume'].sum()
                
                # Calculate momentum for overlapping strikes
                overlapping_strikes = set(current_strikes.index) & set(historical_strikes.index)
                
                if overlapping_strikes:
                    momentum_values = []
                    for strike in overlapping_strikes:
                        current_vol = current_strikes[strike]
                        historical_vol = historical_strikes[strike]
                        momentum = (current_vol - historical_vol) / historical_vol if historical_vol > 0 else 0.0
                        momentum_values.append(momentum)
                    
                    strike_momentum['average_momentum'] = float(np.mean(momentum_values))
                    strike_momentum['momentum_std'] = float(np.std(momentum_values))
                    strike_momentum['positive_momentum_count'] = int(sum(1 for m in momentum_values if m > 0))
                    strike_momentum['negative_momentum_count'] = int(sum(1 for m in momentum_values if m < 0))
                    
                    # Identify strongest momentum strikes
                    strike_momentum_dict = {strike: (current_strikes[strike] - historical_strikes[strike]) / historical_strikes[strike] 
                                          for strike in overlapping_strikes if historical_strikes[strike] > 0}
                    
                    if strike_momentum_dict:
                        top_momentum = sorted(strike_momentum_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                        strike_momentum['top_momentum_strikes'] = {str(k): float(v) for k, v in top_momentum}
            
            # Current strike concentration
            total_volume = current_strikes.sum()
            top_5_volume = current_strikes.nlargest(5).sum()
            strike_momentum['concentration'] = float(top_5_volume / total_volume) if total_volume > 0 else 0.0
            
            return strike_momentum
            
        except Exception as e:
            logger.error(f"Error analyzing strike momentum: {e}")
            return {}
    
    def _analyze_moneyness_momentum(self, current_data: pd.DataFrame, historical_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Analyze momentum by moneyness levels"""
        try:
            moneyness_momentum = {}
            
            if 'moneyness' not in current_data.columns:
                return moneyness_momentum
            
            # Define moneyness buckets
            buckets = {
                'deep_otm_put': (0.0, 0.85),
                'otm_put': (0.85, 0.95),
                'atm': (0.95, 1.05),
                'otm_call': (1.05, 1.15),
                'deep_otm_call': (1.15, 2.0)
            }
            
            current_buckets = {}
            for bucket_name, (low, high) in buckets.items():
                bucket_data = current_data[(current_data['moneyness'] >= low) & (current_data['moneyness'] < high)]
                current_buckets[bucket_name] = bucket_data['volume'].sum()
            
            if historical_data is not None and 'moneyness' in historical_data.columns:
                historical_buckets = {}
                for bucket_name, (low, high) in buckets.items():
                    bucket_data = historical_data[(historical_data['moneyness'] >= low) & (historical_data['moneyness'] < high)]
                    historical_buckets[bucket_name] = bucket_data['volume'].sum()
                
                # Calculate momentum for each bucket
                for bucket_name in buckets.keys():
                    current_vol = current_buckets.get(bucket_name, 0)
                    historical_vol = historical_buckets.get(bucket_name, 0)
                    
                    if historical_vol > 0:
                        momentum = (current_vol - historical_vol) / historical_vol
                        moneyness_momentum[f'{bucket_name}_momentum'] = float(momentum)
            
            # Current distribution
            total_volume = sum(current_buckets.values())
            if total_volume > 0:
                for bucket_name, volume in current_buckets.items():
                    moneyness_momentum[f'{bucket_name}_share'] = float(volume / total_volume)
            
            return moneyness_momentum
            
        except Exception as e:
            logger.error(f"Error analyzing moneyness momentum: {e}")
            return {}
    
    def _analyze_term_momentum(self, current_data: pd.DataFrame, historical_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Analyze momentum by time to expiration"""
        try:
            term_momentum = {}
            
            # Define term buckets
            buckets = {
                'weekly': (0, 7),
                'monthly': (7, 35),
                'quarterly': (35, 100),
                'long_term': (100, 365)
            }
            
            current_buckets = {}
            for bucket_name, (low, high) in buckets.items():
                bucket_data = current_data[(current_data['dte'] >= low) & (current_data['dte'] < high)]
                current_buckets[bucket_name] = bucket_data['volume'].sum()
            
            if historical_data is not None and 'dte' in historical_data.columns:
                historical_buckets = {}
                for bucket_name, (low, high) in buckets.items():
                    bucket_data = historical_data[(historical_data['dte'] >= low) & (historical_data['dte'] < high)]
                    historical_buckets[bucket_name] = bucket_data['volume'].sum()
                
                # Calculate momentum for each bucket
                for bucket_name in buckets.keys():
                    current_vol = current_buckets.get(bucket_name, 0)
                    historical_vol = historical_buckets.get(bucket_name, 0)
                    
                    if historical_vol > 0:
                        momentum = (current_vol - historical_vol) / historical_vol
                        term_momentum[f'{bucket_name}_momentum'] = float(momentum)
            
            # Term structure analysis
            total_volume = sum(current_buckets.values())
            if total_volume > 0:
                term_momentum['near_term_dominance'] = float((current_buckets.get('weekly', 0) + current_buckets.get('monthly', 0)) / total_volume)
            
            return term_momentum
            
        except Exception as e:
            logger.error(f"Error analyzing term momentum: {e}")
            return {}
    
    def _analyze_option_type_momentum(self, current_data: pd.DataFrame, historical_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Analyze momentum by option type (calls vs puts)"""
        try:
            type_momentum = {}
            
            # Current call/put volumes
            call_volume = current_data[current_data['option_type'] == 'CE']['volume'].sum()
            put_volume = current_data[current_data['option_type'] == 'PE']['volume'].sum()
            
            type_momentum['current_call_volume'] = float(call_volume)
            type_momentum['current_put_volume'] = float(put_volume)
            type_momentum['current_cp_ratio'] = float(call_volume / put_volume) if put_volume > 0 else 0.0
            
            if historical_data is not None and 'option_type' in historical_data.columns:
                hist_call_volume = historical_data[historical_data['option_type'] == 'CE']['volume'].sum()
                hist_put_volume = historical_data[historical_data['option_type'] == 'PE']['volume'].sum()
                
                # Calculate momentum
                if hist_call_volume > 0:
                    type_momentum['call_momentum'] = float((call_volume - hist_call_volume) / hist_call_volume)
                if hist_put_volume > 0:
                    type_momentum['put_momentum'] = float((put_volume - hist_put_volume) / hist_put_volume)
                
                # Ratio momentum
                if hist_put_volume > 0 and hist_call_volume > 0:
                    hist_cp_ratio = hist_call_volume / hist_put_volume
                    current_cp_ratio = call_volume / put_volume if put_volume > 0 else float('inf')
                    type_momentum['cp_ratio_momentum'] = float((current_cp_ratio - hist_cp_ratio) / hist_cp_ratio)
            
            return type_momentum
            
        except Exception as e:
            logger.error(f"Error analyzing option type momentum: {e}")
            return {}
    
    def _analyze_sector_momentum(self, current_data: pd.DataFrame, historical_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Analyze momentum by sector (if available)"""
        try:
            sector_momentum = {}
            
            # Group by sector
            current_sectors = current_data.groupby('sector')['volume'].sum()
            
            if historical_data is not None and 'sector' in historical_data.columns:
                historical_sectors = historical_data.groupby('sector')['volume'].sum()
                
                # Calculate momentum for each sector
                overlapping_sectors = set(current_sectors.index) & set(historical_sectors.index)
                
                sector_momentums = {}
                for sector in overlapping_sectors:
                    current_vol = current_sectors[sector]
                    historical_vol = historical_sectors[sector]
                    
                    if historical_vol > 0:
                        momentum = (current_vol - historical_vol) / historical_vol
                        sector_momentums[sector] = momentum
                
                if sector_momentums:
                    # Top momentum sectors
                    top_sectors = sorted(sector_momentums.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                    sector_momentum['top_momentum_sectors'] = {str(k): float(v) for k, v in top_sectors}
                    
                    # Sector breadth
                    positive_sectors = sum(1 for m in sector_momentums.values() if m > 0)
                    total_sectors = len(sector_momentums)
                    sector_momentum['positive_momentum_ratio'] = float(positive_sectors / total_sectors) if total_sectors > 0 else 0.0
            
            return sector_momentum
            
        except Exception as e:
            logger.error(f"Error analyzing sector momentum: {e}")
            return {}
    
    def _analyze_momentum_divergences(self, basic_momentum: Dict[str, Any], advanced_momentum: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze divergences in momentum patterns"""
        try:
            divergences = {
                'divergence_signals': [],
                'divergence_count': 0,
                'divergence_severity': 0.0
            }
            
            # Volume vs OI divergence
            vol_momentum = basic_momentum.get('volume_momentum', 0.0)
            oi_momentum = basic_momentum.get('oi_momentum', 0.0)
            
            if abs(vol_momentum - oi_momentum) > self.momentum_threshold:
                divergences['divergence_signals'].append('volume_oi_divergence')
                divergences['divergence_severity'] += min(abs(vol_momentum - oi_momentum), 1.0) * 0.3
            
            # Call vs Put momentum divergence
            type_momentum = advanced_momentum.get('option_type_momentum', {})
            call_momentum = type_momentum.get('call_momentum', 0.0)
            put_momentum = type_momentum.get('put_momentum', 0.0)
            
            if abs(call_momentum - put_momentum) > self.momentum_threshold * 1.5:
                divergences['divergence_signals'].append('call_put_momentum_divergence')
                divergences['divergence_severity'] += min(abs(call_momentum - put_momentum), 1.0) * 0.4
            
            # Term structure momentum divergence
            term_momentum = advanced_momentum.get('term_momentum', {})
            weekly_momentum = term_momentum.get('weekly_momentum', 0.0)
            monthly_momentum = term_momentum.get('monthly_momentum', 0.0)
            
            if abs(weekly_momentum - monthly_momentum) > self.momentum_threshold * 2:
                divergences['divergence_signals'].append('term_momentum_divergence')
                divergences['divergence_severity'] += min(abs(weekly_momentum - monthly_momentum), 1.0) * 0.3
            
            divergences['divergence_count'] = len(divergences['divergence_signals'])
            divergences['divergence_severity'] = min(divergences['divergence_severity'], 1.0)
            
            return divergences
            
        except Exception as e:
            logger.error(f"Error analyzing momentum divergences: {e}")
            return {'divergence_signals': [], 'divergence_count': 0, 'divergence_severity': 0.0}
    
    def _analyze_momentum_velocity(self, basic_momentum: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze momentum velocity and acceleration"""
        try:
            velocity = {'acceleration': 0.0, 'velocity_trend': 'stable'}
            
            # Track volume momentum acceleration
            volume_momentum = basic_momentum.get('volume_momentum', 0.0)
            self.velocity_metrics['acceleration'].append(volume_momentum)
            
            if len(self.velocity_metrics['acceleration']) > self.velocity_window:
                self.velocity_metrics['acceleration'].pop(0)
            
            if len(self.velocity_metrics['acceleration']) >= 3:
                recent_momentum = np.mean(self.velocity_metrics['acceleration'][-2:])
                previous_momentum = np.mean(self.velocity_metrics['acceleration'][:-2])
                
                acceleration = recent_momentum - previous_momentum
                velocity['acceleration'] = float(acceleration)
                
                if acceleration > self.momentum_threshold / 2:
                    velocity['velocity_trend'] = 'accelerating'
                elif acceleration < -self.momentum_threshold / 2:
                    velocity['velocity_trend'] = 'decelerating'
            
            return velocity
            
        except Exception as e:
            logger.error(f"Error analyzing momentum velocity: {e}")
            return {'acceleration': 0.0, 'velocity_trend': 'stable'}
    
    def _detect_momentum_regimes(self, basic_momentum: Dict[str, Any], advanced_momentum: Dict[str, Any]) -> Dict[str, Any]:
        """Detect momentum regimes for breadth classification"""
        try:
            regime = {
                'momentum_regime': 'normal',
                'regime_confidence': 0.0,
                'regime_characteristics': []
            }
            
            # Volume momentum regime
            vol_momentum = basic_momentum.get('volume_momentum', 0.0)
            if abs(vol_momentum) > self.momentum_threshold * 2:
                regime['regime_characteristics'].append('high_volume_momentum')
                regime['regime_confidence'] += 0.3
            
            # Breadth momentum analysis
            strike_momentum = advanced_momentum.get('strike_momentum', {})
            positive_strikes = strike_momentum.get('positive_momentum_count', 0)
            negative_strikes = strike_momentum.get('negative_momentum_count', 0)
            total_strikes = positive_strikes + negative_strikes
            
            if total_strikes > 0:
                breadth_ratio = positive_strikes / total_strikes
                if breadth_ratio > 0.7:
                    regime['regime_characteristics'].append('broad_positive_momentum')
                    regime['regime_confidence'] += 0.3
                elif breadth_ratio < 0.3:
                    regime['regime_characteristics'].append('broad_negative_momentum')
                    regime['regime_confidence'] += 0.3
                else:
                    regime['regime_characteristics'].append('mixed_momentum')
                    regime['regime_confidence'] += 0.1
            
            # Determine regime
            if regime['regime_confidence'] > 0.5:
                if any('high_volume' in char for char in regime['regime_characteristics']):
                    if any('broad_positive' in char for char in regime['regime_characteristics']):
                        regime['momentum_regime'] = 'strong_bullish_momentum'
                    elif any('broad_negative' in char for char in regime['regime_characteristics']):
                        regime['momentum_regime'] = 'strong_bearish_momentum'
                    else:
                        regime['momentum_regime'] = 'high_momentum_mixed'
                elif any('broad_positive' in char for char in regime['regime_characteristics']):
                    regime['momentum_regime'] = 'broad_positive_momentum'
                elif any('broad_negative' in char for char in regime['regime_characteristics']):
                    regime['momentum_regime'] = 'broad_negative_momentum'
            
            return regime
            
        except Exception as e:
            logger.error(f"Error detecting momentum regimes: {e}")
            return {'momentum_regime': 'normal', 'regime_confidence': 0.0}
    
    def _generate_momentum_signals(self, basic_momentum: Dict[str, Any], advanced_momentum: Dict[str, Any], regimes: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable momentum signals"""
        try:
            signals = {
                'primary_signal': 'neutral',
                'signal_strength': 0.0,
                'momentum_signals': [],
                'breadth_implications': []
            }
            
            # Volume momentum signals
            vol_momentum = basic_momentum.get('volume_momentum', 0.0)
            if vol_momentum > self.momentum_threshold:
                signals['momentum_signals'].append('positive_volume_momentum')
                signals['breadth_implications'].append('expanding_participation')
                signals['signal_strength'] += min(vol_momentum, 1.0) * 0.3
            elif vol_momentum < -self.momentum_threshold:
                signals['momentum_signals'].append('negative_volume_momentum')
                signals['breadth_implications'].append('contracting_participation')
                signals['signal_strength'] += min(abs(vol_momentum), 1.0) * 0.3
            
            # Regime-based signals
            momentum_regime = regimes.get('momentum_regime', 'normal')
            if 'strong' in momentum_regime:
                signals['momentum_signals'].append('strong_momentum_regime')
                signals['signal_strength'] += 0.4
                
                if 'bullish' in momentum_regime:
                    signals['breadth_implications'].append('broad_bullish_momentum')
                elif 'bearish' in momentum_regime:
                    signals['breadth_implications'].append('broad_bearish_momentum')
            
            # Strike breadth signals
            strike_momentum = advanced_momentum.get('strike_momentum', {})
            if 'positive_momentum_count' in strike_momentum and 'negative_momentum_count' in strike_momentum:
                positive = strike_momentum['positive_momentum_count']
                negative = strike_momentum['negative_momentum_count']
                total = positive + negative
                
                if total > 0:
                    breadth_ratio = positive / total
                    if breadth_ratio > 0.75:
                        signals['momentum_signals'].append('broad_positive_breadth')
                        signals['breadth_implications'].append('widespread_bullish_momentum')
                        signals['signal_strength'] += 0.3
                    elif breadth_ratio < 0.25:
                        signals['momentum_signals'].append('broad_negative_breadth')
                        signals['breadth_implications'].append('widespread_bearish_momentum')
                        signals['signal_strength'] += 0.3
            
            # Determine primary signal
            if signals['signal_strength'] > 0.6:
                if any('positive' in sig for sig in signals['momentum_signals']):
                    signals['primary_signal'] = 'bullish_momentum_expansion'
                elif any('negative' in sig for sig in signals['momentum_signals']):
                    signals['primary_signal'] = 'bearish_momentum_expansion'
                else:
                    signals['primary_signal'] = 'strong_momentum_shift'
            elif signals['signal_strength'] > 0.3:
                signals['primary_signal'] = 'moderate_momentum_change'
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating momentum signals: {e}")
            return {'primary_signal': 'neutral', 'signal_strength': 0.0}
    
    def _calculate_momentum_breadth_score(self, basic_momentum: Dict[str, Any], advanced_momentum: Dict[str, Any]) -> float:
        """Calculate breadth score from momentum analysis"""
        try:
            score = 0.5  # Base neutral score
            
            # Volume momentum contribution (30%)
            vol_momentum = basic_momentum.get('volume_momentum', 0.0)
            vol_score = min(abs(vol_momentum), 1.0)
            if vol_momentum > 0:
                score += vol_score * 0.3
            else:
                score -= vol_score * 0.3
            
            # Strike breadth contribution (40%)
            strike_momentum = advanced_momentum.get('strike_momentum', {})
            positive_strikes = strike_momentum.get('positive_momentum_count', 0)
            negative_strikes = strike_momentum.get('negative_momentum_count', 0)
            total_strikes = positive_strikes + negative_strikes
            
            if total_strikes > 0:
                breadth_ratio = positive_strikes / total_strikes
                # Convert to score: 0.75+ = +0.4, 0.25- = -0.4, 0.5 = 0
                breadth_contribution = (breadth_ratio - 0.5) * 0.8  # Scale to ±0.4
                score += breadth_contribution
            
            # OI momentum contribution (20%)
            oi_momentum = basic_momentum.get('oi_momentum', 0.0)
            oi_score = min(abs(oi_momentum), 1.0)
            if oi_momentum > 0:
                score += oi_score * 0.2
            else:
                score -= oi_score * 0.2
            
            # Term momentum balance contribution (10%)
            term_momentum = advanced_momentum.get('term_momentum', {})
            near_dominance = term_momentum.get('near_term_dominance', 0.5)
            # Balanced term structure (0.4-0.6) gets positive contribution
            if 0.4 <= near_dominance <= 0.6:
                score += 0.1
            else:
                score -= min(abs(near_dominance - 0.5), 0.5) * 0.2  # Penalty for imbalance
            
            return max(min(float(score), 1.0), 0.0)
            
        except Exception as e:
            logger.error(f"Error calculating momentum breadth score: {e}")
            return 0.5
    
    def _update_momentum_history(self, basic_momentum: Dict[str, Any], advanced_momentum: Dict[str, Any]):
        """Update historical momentum tracking"""
        try:
            # Update basic momentum history
            for key in ['volume_momentum', 'oi_momentum', 'iv_momentum', 'premium_momentum']:
                if key in basic_momentum:
                    momentum_key = key
                    if momentum_key not in self.momentum_history:
                        self.momentum_history[momentum_key] = []
                    
                    self.momentum_history[momentum_key].append(basic_momentum[key])
                    if len(self.momentum_history[momentum_key]) > self.momentum_window * 2:
                        self.momentum_history[momentum_key].pop(0)
            
        except Exception as e:
            logger.error(f"Error updating momentum history: {e}")
    
    def _get_default_momentum_analysis(self) -> Dict[str, Any]:
        """Get default momentum analysis when data is insufficient"""
        return {
            'basic_momentum': {'volume_momentum': 0.0, 'oi_momentum': 0.0},
            'advanced_momentum': {},
            'momentum_divergences': {'divergence_signals': [], 'divergence_count': 0},
            'velocity_analysis': {'acceleration': 0.0, 'velocity_trend': 'stable'},
            'momentum_regimes': {'momentum_regime': 'normal', 'regime_confidence': 0.0},
            'momentum_signals': {'primary_signal': 'neutral', 'signal_strength': 0.0},
            'breadth_score': 0.5
        }
    
    def get_momentum_summary(self) -> Dict[str, Any]:
        """Get summary of momentum analysis system"""
        try:
            return {
                'history_lengths': {key: len(values) for key, values in self.momentum_history.items()},
                'current_averages': {
                    key: np.mean(values) if values else 0.0 
                    for key, values in self.momentum_history.items()
                },
                'velocity_metrics': {
                    'recent_acceleration': self.velocity_metrics['acceleration'][-1] if self.velocity_metrics['acceleration'] else 0.0,
                    'direction_changes': self.velocity_metrics['direction_changes']
                },
                'analysis_config': {
                    'momentum_window': self.momentum_window,
                    'velocity_window': self.velocity_window,
                    'momentum_threshold': self.momentum_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting momentum summary: {e}")
            return {'status': 'error', 'error': str(e)}