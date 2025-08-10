"""
Option Ratio Analyzer - Option Ratio Analysis for Market Breadth
===============================================================

Analyzes various option ratios to assess market breadth and sentiment.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class OptionRatioAnalyzer:
    """Option ratio analyzer for market breadth assessment"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Option Ratio Analyzer"""
        self.ratio_window = config.get('ratio_window', 20)
        self.extreme_thresholds = config.get('extreme_thresholds', {
            'put_call_ratio': {'high': 1.5, 'low': 0.5},
            'call_put_volume': {'high': 3.0, 'low': 0.33},
            'skew_ratio': {'high': 1.2, 'low': 0.8}
        })
        
        # Historical tracking
        self.ratio_history = {
            'put_call_ratios': [],
            'volume_ratios': [],
            'oi_ratios': [],
            'timestamps': []
        }
        
        logger.info("OptionRatioAnalyzer initialized")
    
    def analyze_option_ratios(self, option_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze comprehensive option ratios for breadth assessment
        
        Args:
            option_data: DataFrame with option market data
            
        Returns:
            Dict with ratio analysis results
        """
        try:
            if option_data.empty:
                return self._get_default_ratio_analysis()
            
            # Calculate basic ratios
            basic_ratios = self._calculate_basic_ratios(option_data)
            
            # Calculate advanced ratios
            advanced_ratios = self._calculate_advanced_ratios(option_data)
            
            # Analyze ratio trends
            ratio_trends = self._analyze_ratio_trends(basic_ratios, advanced_ratios)
            
            # Detect ratio extremes
            ratio_extremes = self._detect_ratio_extremes(basic_ratios, advanced_ratios)
            
            # Generate breadth signals
            breadth_signals = self._generate_breadth_signals(basic_ratios, advanced_ratios, ratio_extremes)
            
            # Update historical tracking
            self._update_ratio_history(basic_ratios)
            
            return {
                'basic_ratios': basic_ratios,
                'advanced_ratios': advanced_ratios,
                'ratio_trends': ratio_trends,
                'ratio_extremes': ratio_extremes,
                'breadth_signals': breadth_signals,
                'breadth_score': self._calculate_ratio_breadth_score(basic_ratios, advanced_ratios)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing option ratios: {e}")
            return self._get_default_ratio_analysis()
    
    def _calculate_basic_ratios(self, option_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic option ratios"""
        try:
            ratios = {}
            
            # Put/Call ratios by volume and OI
            if 'option_type' in option_data.columns:
                calls = option_data[option_data['option_type'] == 'CE']
                puts = option_data[option_data['option_type'] == 'PE']
                
                # Volume ratios
                call_volume = calls['volume'].sum()
                put_volume = puts['volume'].sum()
                ratios['put_call_volume_ratio'] = float(put_volume / call_volume) if call_volume > 0 else 0.0
                ratios['call_put_volume_ratio'] = float(call_volume / put_volume) if put_volume > 0 else 0.0
                
                # Open Interest ratios
                if 'oi' in option_data.columns:
                    call_oi = calls['oi'].sum()
                    put_oi = puts['oi'].sum()
                    ratios['put_call_oi_ratio'] = float(put_oi / call_oi) if call_oi > 0 else 0.0
                    ratios['call_put_oi_ratio'] = float(call_oi / put_oi) if put_oi > 0 else 0.0
                
                # Premium ratios
                if 'premium' in option_data.columns:
                    call_premium = calls['premium'].sum()
                    put_premium = puts['premium'].sum()
                    ratios['put_call_premium_ratio'] = float(put_premium / call_premium) if call_premium > 0 else 0.0
            
            # Moneyness-based ratios
            if 'moneyness' in option_data.columns:
                itm_options = option_data[option_data['moneyness'] < 1.0]
                otm_options = option_data[option_data['moneyness'] > 1.0]
                
                itm_volume = itm_options['volume'].sum()
                otm_volume = otm_options['volume'].sum()
                ratios['itm_otm_volume_ratio'] = float(itm_volume / otm_volume) if otm_volume > 0 else 0.0
                
                if 'oi' in option_data.columns:
                    itm_oi = itm_options['oi'].sum()
                    otm_oi = otm_options['oi'].sum()
                    ratios['itm_otm_oi_ratio'] = float(itm_oi / otm_oi) if otm_oi > 0 else 0.0
            
            return ratios
            
        except Exception as e:
            logger.error(f"Error calculating basic ratios: {e}")
            return {}
    
    def _calculate_advanced_ratios(self, option_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate advanced option ratios"""
        try:
            ratios = {}
            
            # Term structure ratios
            if 'dte' in option_data.columns:
                near_term = option_data[option_data['dte'] <= 30]
                far_term = option_data[option_data['dte'] > 30]
                
                if not near_term.empty and not far_term.empty:
                    near_volume = near_term['volume'].sum()
                    far_volume = far_term['volume'].sum()
                    ratios['near_far_volume_ratio'] = float(near_volume / far_volume) if far_volume > 0 else 0.0
                    
                    if 'iv' in option_data.columns:
                        near_iv_avg = near_term['iv'].mean()
                        far_iv_avg = far_term['iv'].mean()
                        ratios['near_far_iv_ratio'] = float(near_iv_avg / far_iv_avg) if far_iv_avg > 0 else 1.0
            
            # Strike distribution ratios
            if 'strike' in option_data.columns and 'underlying_price' in option_data.columns:
                underlying_price = option_data['underlying_price'].iloc[0]
                
                # ATM band (±5%)
                atm_lower = underlying_price * 0.95
                atm_upper = underlying_price * 1.05
                atm_options = option_data[
                    (option_data['strike'] >= atm_lower) & 
                    (option_data['strike'] <= atm_upper)
                ]
                
                # OTM options
                otm_options = option_data[
                    (option_data['strike'] < atm_lower) | 
                    (option_data['strike'] > atm_upper)
                ]
                
                if not atm_options.empty and not otm_options.empty:
                    atm_volume = atm_options['volume'].sum()
                    otm_volume = otm_options['volume'].sum()
                    ratios['atm_otm_volume_ratio'] = float(atm_volume / otm_volume) if otm_volume > 0 else 0.0
            
            # Volatility skew ratios
            if 'iv' in option_data.columns and 'moneyness' in option_data.columns:
                # Separate into puts and calls for skew analysis
                if 'option_type' in option_data.columns:
                    puts = option_data[option_data['option_type'] == 'PE']
                    calls = option_data[option_data['option_type'] == 'CE']
                    
                    if not puts.empty and not calls.empty:
                        # OTM put IV (moneyness < 0.95)
                        otm_puts = puts[puts['moneyness'] < 0.95]
                        # OTM call IV (moneyness > 1.05)
                        otm_calls = calls[calls['moneyness'] > 1.05]
                        
                        if not otm_puts.empty and not otm_calls.empty:
                            put_iv_avg = otm_puts['iv'].mean()
                            call_iv_avg = otm_calls['iv'].mean()
                            ratios['put_call_skew_ratio'] = float(put_iv_avg / call_iv_avg) if call_iv_avg > 0 else 1.0
            
            # Participation ratios
            if 'volume' in option_data.columns:
                total_volume = option_data['volume'].sum()
                active_strikes = len(option_data[option_data['volume'] > 0])
                total_strikes = len(option_data)
                
                ratios['strike_participation_ratio'] = float(active_strikes / total_strikes) if total_strikes > 0 else 0.0
                
                # High volume participation (volume > 90th percentile)
                if total_volume > 0:
                    volume_90th = np.percentile(option_data['volume'], 90)
                    high_volume_strikes = len(option_data[option_data['volume'] > volume_90th])
                    ratios['high_volume_participation'] = float(high_volume_strikes / total_strikes) if total_strikes > 0 else 0.0
            
            return ratios
            
        except Exception as e:
            logger.error(f"Error calculating advanced ratios: {e}")
            return {}
    
    def _analyze_ratio_trends(self, basic_ratios: Dict[str, Any], advanced_ratios: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trends in option ratios"""
        try:
            trends = {}
            
            # Put/Call ratio trend
            if self.ratio_history['put_call_ratios'] and 'put_call_volume_ratio' in basic_ratios:
                current_pc = basic_ratios['put_call_volume_ratio']
                historical_avg = np.mean(self.ratio_history['put_call_ratios'][-self.ratio_window:])
                
                trends['put_call_trend'] = 'increasing' if current_pc > historical_avg * 1.1 else 'decreasing' if current_pc < historical_avg * 0.9 else 'stable'
                trends['put_call_deviation'] = float((current_pc - historical_avg) / historical_avg) if historical_avg > 0 else 0.0
            
            # Volume ratio trend
            if self.ratio_history['volume_ratios'] and 'call_put_volume_ratio' in basic_ratios:
                current_vol = basic_ratios['call_put_volume_ratio']
                historical_vol_avg = np.mean(self.ratio_history['volume_ratios'][-self.ratio_window:])
                
                trends['volume_ratio_trend'] = 'increasing' if current_vol > historical_vol_avg * 1.1 else 'decreasing' if current_vol < historical_vol_avg * 0.9 else 'stable'
                trends['volume_ratio_momentum'] = float((current_vol - historical_vol_avg) / historical_vol_avg) if historical_vol_avg > 0 else 0.0
            
            # Participation trend
            if 'strike_participation_ratio' in advanced_ratios:
                participation = advanced_ratios['strike_participation_ratio']
                if participation > 0.8:
                    trends['participation_trend'] = 'broad'
                elif participation > 0.5:
                    trends['participation_trend'] = 'moderate'
                else:
                    trends['participation_trend'] = 'narrow'
            
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing ratio trends: {e}")
            return {}
    
    def _detect_ratio_extremes(self, basic_ratios: Dict[str, Any], advanced_ratios: Dict[str, Any]) -> Dict[str, Any]:
        """Detect extreme ratio values that indicate breadth anomalies"""
        try:
            extremes = {
                'extreme_ratios': [],
                'extreme_count': 0,
                'severity_score': 0.0
            }
            
            # Check Put/Call ratio extremes
            pc_ratio = basic_ratios.get('put_call_volume_ratio', 1.0)
            pc_thresholds = self.extreme_thresholds['put_call_ratio']
            
            if pc_ratio > pc_thresholds['high']:
                extremes['extreme_ratios'].append({
                    'type': 'extreme_put_dominance',
                    'value': pc_ratio,
                    'threshold': pc_thresholds['high'],
                    'severity': min((pc_ratio - pc_thresholds['high']) / pc_thresholds['high'], 1.0)
                })
                extremes['severity_score'] += 0.4
            elif pc_ratio < pc_thresholds['low']:
                extremes['extreme_ratios'].append({
                    'type': 'extreme_call_dominance',
                    'value': pc_ratio,
                    'threshold': pc_thresholds['low'],
                    'severity': min((pc_thresholds['low'] - pc_ratio) / pc_thresholds['low'], 1.0)
                })
                extremes['severity_score'] += 0.4
            
            # Check Call/Put volume ratio extremes
            cp_vol_ratio = basic_ratios.get('call_put_volume_ratio', 1.0)
            cp_thresholds = self.extreme_thresholds['call_put_volume']
            
            if cp_vol_ratio > cp_thresholds['high']:
                extremes['extreme_ratios'].append({
                    'type': 'extreme_call_volume',
                    'value': cp_vol_ratio,
                    'threshold': cp_thresholds['high'],
                    'severity': min((cp_vol_ratio - cp_thresholds['high']) / cp_thresholds['high'], 1.0)
                })
                extremes['severity_score'] += 0.3
            elif cp_vol_ratio < cp_thresholds['low']:
                extremes['extreme_ratios'].append({
                    'type': 'extreme_put_volume', 
                    'value': cp_vol_ratio,
                    'threshold': cp_thresholds['low'],
                    'severity': min((cp_thresholds['low'] - cp_vol_ratio) / cp_thresholds['low'], 1.0)
                })
                extremes['severity_score'] += 0.3
            
            # Check participation extremes
            participation = advanced_ratios.get('strike_participation_ratio', 0.5)
            if participation < 0.2:
                extremes['extreme_ratios'].append({
                    'type': 'extreme_narrow_participation',
                    'value': participation,
                    'threshold': 0.2,
                    'severity': min((0.2 - participation) / 0.2, 1.0)
                })
                extremes['severity_score'] += 0.3
            
            extremes['extreme_count'] = len(extremes['extreme_ratios'])
            extremes['severity_score'] = min(extremes['severity_score'], 1.0)
            
            return extremes
            
        except Exception as e:
            logger.error(f"Error detecting ratio extremes: {e}")
            return {'extreme_ratios': [], 'extreme_count': 0, 'severity_score': 0.0}
    
    def _generate_breadth_signals(self, basic_ratios: Dict[str, Any], advanced_ratios: Dict[str, Any], extremes: Dict[str, Any]) -> Dict[str, Any]:
        """Generate breadth signals from ratio analysis"""
        try:
            signals = {
                'primary_signal': 'neutral',
                'signal_strength': 0.0,
                'ratio_signals': [],
                'breadth_implications': []
            }
            
            # Extreme ratio signals
            if extremes['extreme_count'] > 0:
                extreme_types = [ext['type'] for ext in extremes['extreme_ratios']]
                
                if any('narrow_participation' in ext for ext in extreme_types):
                    signals['ratio_signals'].append('narrow_breadth')
                    signals['breadth_implications'].append('concentrated_activity')
                    signals['signal_strength'] += 0.4
                
                if any('call_dominance' in ext or 'call_volume' in ext for ext in extreme_types):
                    signals['ratio_signals'].append('bullish_skew')
                    signals['breadth_implications'].append('call_dominated_breadth')
                    signals['signal_strength'] += 0.3
                
                if any('put_dominance' in ext or 'put_volume' in ext for ext in extreme_types):
                    signals['ratio_signals'].append('bearish_skew')
                    signals['breadth_implications'].append('put_dominated_breadth')
                    signals['signal_strength'] += 0.3
            
            # Participation signals
            participation = advanced_ratios.get('strike_participation_ratio', 0.5)
            if participation > 0.8:
                signals['ratio_signals'].append('broad_participation')
                signals['breadth_implications'].append('wide_market_engagement')
                signals['signal_strength'] += 0.2
            elif participation < 0.3:
                signals['ratio_signals'].append('narrow_participation')
                signals['breadth_implications'].append('limited_market_engagement')
                signals['signal_strength'] += 0.2
            
            # ITM/OTM flow signals
            itm_otm_ratio = basic_ratios.get('itm_otm_volume_ratio', 1.0)
            if itm_otm_ratio > 1.5:
                signals['ratio_signals'].append('itm_flow_dominance')
                signals['breadth_implications'].append('directional_conviction')
                signals['signal_strength'] += 0.2
            elif itm_otm_ratio < 0.67:
                signals['ratio_signals'].append('otm_flow_dominance')
                signals['breadth_implications'].append('speculative_activity')
                signals['signal_strength'] += 0.2
            
            # Determine primary signal
            if signals['signal_strength'] > 0.6:
                if any('narrow' in sig for sig in signals['ratio_signals']):
                    signals['primary_signal'] = 'contracting_breadth'
                elif any('broad' in sig for sig in signals['ratio_signals']):
                    signals['primary_signal'] = 'expanding_breadth'
                else:
                    signals['primary_signal'] = 'shifting_breadth'
            elif signals['signal_strength'] > 0.3:
                signals['primary_signal'] = 'moderate_breadth_change'
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating breadth signals: {e}")
            return {'primary_signal': 'neutral', 'signal_strength': 0.0}
    
    def _calculate_ratio_breadth_score(self, basic_ratios: Dict[str, Any], advanced_ratios: Dict[str, Any]) -> float:
        """Calculate breadth score from ratio analysis"""
        try:
            score = 0.5  # Base neutral score
            
            # Participation contribution (40%)
            participation = advanced_ratios.get('strike_participation_ratio', 0.5)
            score += (participation - 0.5) * 0.4
            
            # Put/Call balance contribution (30%)
            pc_ratio = basic_ratios.get('put_call_volume_ratio', 1.0)
            balance_score = 1.0 - abs(np.log(pc_ratio)) / 2.0 if pc_ratio > 0 else 0.0
            balance_score = max(min(balance_score, 1.0), 0.0)
            score += (balance_score - 0.5) * 0.3
            
            # ITM/OTM balance contribution (20%)
            itm_otm_ratio = basic_ratios.get('itm_otm_volume_ratio', 1.0)
            itm_balance = 1.0 - abs(np.log(itm_otm_ratio)) / 2.0 if itm_otm_ratio > 0 else 0.5
            itm_balance = max(min(itm_balance, 1.0), 0.0)
            score += (itm_balance - 0.5) * 0.2
            
            # High volume participation contribution (10%)
            high_vol_participation = advanced_ratios.get('high_volume_participation', 0.1)
            score += (high_vol_participation - 0.1) * 0.1 / 0.9  # Scale to contribute ±0.1
            
            return max(min(float(score), 1.0), 0.0)
            
        except Exception as e:
            logger.error(f"Error calculating ratio breadth score: {e}")
            return 0.5
    
    def _update_ratio_history(self, basic_ratios: Dict[str, Any]):
        """Update historical ratio tracking"""
        try:
            # Update Put/Call ratios
            if 'put_call_volume_ratio' in basic_ratios:
                self.ratio_history['put_call_ratios'].append(basic_ratios['put_call_volume_ratio'])
                if len(self.ratio_history['put_call_ratios']) > self.ratio_window * 2:
                    self.ratio_history['put_call_ratios'].pop(0)
            
            # Update volume ratios
            if 'call_put_volume_ratio' in basic_ratios:
                self.ratio_history['volume_ratios'].append(basic_ratios['call_put_volume_ratio'])
                if len(self.ratio_history['volume_ratios']) > self.ratio_window * 2:
                    self.ratio_history['volume_ratios'].pop(0)
            
            # Update OI ratios
            if 'put_call_oi_ratio' in basic_ratios:
                self.ratio_history['oi_ratios'].append(basic_ratios['put_call_oi_ratio'])
                if len(self.ratio_history['oi_ratios']) > self.ratio_window * 2:
                    self.ratio_history['oi_ratios'].pop(0)
            
        except Exception as e:
            logger.error(f"Error updating ratio history: {e}")
    
    def _get_default_ratio_analysis(self) -> Dict[str, Any]:
        """Get default ratio analysis when data is insufficient"""
        return {
            'basic_ratios': {},
            'advanced_ratios': {},
            'ratio_trends': {},
            'ratio_extremes': {'extreme_ratios': [], 'extreme_count': 0},
            'breadth_signals': {'primary_signal': 'neutral', 'signal_strength': 0.0},
            'breadth_score': 0.5
        }
    
    def get_ratio_summary(self) -> Dict[str, Any]:
        """Get summary of ratio analysis system"""
        try:
            return {
                'history_lengths': {
                    'put_call_ratios': len(self.ratio_history['put_call_ratios']),
                    'volume_ratios': len(self.ratio_history['volume_ratios']),
                    'oi_ratios': len(self.ratio_history['oi_ratios'])
                },
                'current_averages': {
                    'put_call_avg': np.mean(self.ratio_history['put_call_ratios']) if self.ratio_history['put_call_ratios'] else 1.0,
                    'volume_avg': np.mean(self.ratio_history['volume_ratios']) if self.ratio_history['volume_ratios'] else 1.0
                },
                'analysis_config': {
                    'ratio_window': self.ratio_window,
                    'extreme_thresholds': self.extreme_thresholds
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting ratio summary: {e}")
            return {'status': 'error', 'error': str(e)}