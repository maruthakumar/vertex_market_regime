"""
Option Volume Flow - Volume Flow Analysis for Options
====================================================

Analyzes option volume flow patterns, institutional vs retail activity,
and smart money movements in option markets.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, time
import logging

logger = logging.getLogger(__name__)


class OptionVolumeFlow:
    """
    Option-specific Volume Flow Analysis
    
    Features:
    - Volume profile analysis by strike
    - Institutional vs retail volume detection
    - Smart money flow indicators
    - Volume-weighted price analysis
    - Unusual volume detection
    - Put-Call volume ratios
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Option Volume Flow analyzer"""
        # Volume thresholds
        self.institutional_threshold = config.get('institutional_threshold', 100)
        self.retail_threshold = config.get('retail_threshold', 10)
        self.unusual_volume_multiplier = config.get('unusual_volume_multiplier', 2.0)
        
        # Analysis periods
        self.lookback_period = config.get('lookback_period', 20)
        self.volume_ma_period = config.get('volume_ma_period', 10)
        self.flow_smoothing_period = config.get('flow_smoothing_period', 5)
        
        # Smart money detection
        self.smart_money_strikes = config.get('smart_money_strikes', 5)
        self.smart_money_threshold = config.get('smart_money_threshold', 0.7)
        
        # Flow classification
        self.bullish_flow_threshold = config.get('bullish_flow_threshold', 0.6)
        self.bearish_flow_threshold = config.get('bearish_flow_threshold', 0.4)
        
        # Advanced features
        self.enable_time_analysis = config.get('enable_time_analysis', True)
        self.enable_strike_analysis = config.get('enable_strike_analysis', True)
        self.enable_flow_persistence = config.get('enable_flow_persistence', True)
        
        # Time windows for analysis
        self.time_windows = {
            'opening': (time(9, 15), time(9, 45)),
            'morning': (time(9, 45), time(12, 0)),
            'midday': (time(12, 0), time(14, 0)),
            'closing': (time(14, 0), time(15, 30))
        }
        
        # History tracking
        self.flow_history = {
            'institutional': [],
            'retail': [],
            'smart_money': [],
            'unusual_activity': [],
            'flow_regimes': []
        }
        
        logger.info(f"OptionVolumeFlow initialized: institutional_threshold={self.institutional_threshold}")
    
    def analyze_volume_flow(self,
                          option_data: pd.DataFrame,
                          option_type: str = 'both') -> Dict[str, Any]:
        """
        Analyze comprehensive option volume flow
        
        Args:
            option_data: DataFrame with option volume, OI, prices
            option_type: 'CE', 'PE', or 'both'
            
        Returns:
            Dict with volume flow analysis and signals
        """
        try:
            results = {
                'volume_profile': {},
                'institutional_flow': {},
                'retail_flow': {},
                'smart_money': {},
                'volume_analysis': {},
                'unusual_activity': [],
                'flow_signals': {},
                'regime': None
            }
            
            # Analyze for each option type
            if option_type in ['CE', 'both']:
                ce_data = option_data[option_data['option_type'] == 'CE']
                if not ce_data.empty:
                    results['volume_profile']['CE'] = self._analyze_volume_profile(ce_data)
                    results['institutional_flow']['CE'] = self._analyze_institutional_flow(ce_data)
                    results['retail_flow']['CE'] = self._analyze_retail_flow(ce_data)
                    results['smart_money']['CE'] = self._detect_smart_money(ce_data)
            
            if option_type in ['PE', 'both']:
                pe_data = option_data[option_data['option_type'] == 'PE']
                if not pe_data.empty:
                    results['volume_profile']['PE'] = self._analyze_volume_profile(pe_data)
                    results['institutional_flow']['PE'] = self._analyze_institutional_flow(pe_data)
                    results['retail_flow']['PE'] = self._analyze_retail_flow(pe_data)
                    results['smart_money']['PE'] = self._detect_smart_money(pe_data)
            
            # Comprehensive volume analysis
            results['volume_analysis'] = self._analyze_volume_characteristics(option_data)
            
            # Detect unusual activity
            results['unusual_activity'] = self._detect_unusual_activity(option_data, results)
            
            # Generate flow signals
            results['flow_signals'] = self._generate_flow_signals(results)
            
            # Classify flow regime
            results['regime'] = self._classify_flow_regime(results)
            
            # Update history
            self._update_flow_history(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing volume flow: {e}")
            return self._get_default_results()
    
    def _analyze_volume_profile(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume profile by strike"""
        try:
            # Group by strike and calculate volume metrics
            strike_volume = data.groupby('strike').agg({
                'volume': ['sum', 'mean', 'std'],
                'oi': 'last',
                'price': 'mean'
            })
            
            # Flatten column names
            strike_volume.columns = ['_'.join(col).strip() for col in strike_volume.columns]
            
            # Calculate volume-weighted average price (VWAP) by strike
            total_volume = strike_volume['volume_sum'].sum()
            if total_volume > 0:
                vwap = (strike_volume['volume_sum'] * strike_volume['price_mean']).sum() / total_volume
            else:
                vwap = 0
            
            # Identify high volume strikes
            volume_threshold = strike_volume['volume_sum'].quantile(0.8)
            high_volume_strikes = strike_volume[strike_volume['volume_sum'] > volume_threshold].index.tolist()
            
            # Calculate volume distribution
            volume_distribution = self._calculate_volume_distribution(strike_volume)
            
            # Time-based volume analysis
            time_analysis = {}
            if self.enable_time_analysis and 'timestamp' in data.columns:
                time_analysis = self._analyze_volume_by_time(data)
            
            return {
                'total_volume': int(total_volume),
                'vwap': float(vwap),
                'high_volume_strikes': high_volume_strikes,
                'volume_distribution': volume_distribution,
                'time_analysis': time_analysis,
                'concentration': self._calculate_volume_concentration(strike_volume),
                'status': 'calculated'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volume profile: {e}")
            return {'status': 'error'}
    
    def _analyze_institutional_flow(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze institutional volume flow"""
        try:
            # Filter for institutional-sized trades
            institutional_data = data[data['volume'] >= self.institutional_threshold]
            
            if institutional_data.empty:
                return {'status': 'no_institutional_activity'}
            
            # Calculate metrics
            total_inst_volume = institutional_data['volume'].sum()
            total_volume = data['volume'].sum()
            inst_percentage = total_inst_volume / total_volume if total_volume > 0 else 0
            
            # Analyze institutional preference by strike
            inst_by_strike = institutional_data.groupby('strike')['volume'].sum()
            preferred_strikes = inst_by_strike.nlargest(self.smart_money_strikes).index.tolist()
            
            # Calculate institutional flow direction
            inst_flow_direction = self._calculate_flow_direction(institutional_data)
            
            # Time-based institutional activity
            inst_time_pattern = {}
            if self.enable_time_analysis and 'timestamp' in institutional_data.columns:
                inst_time_pattern = self._analyze_institutional_timing(institutional_data)
            
            return {
                'total_volume': int(total_inst_volume),
                'percentage': float(inst_percentage),
                'preferred_strikes': preferred_strikes,
                'flow_direction': inst_flow_direction,
                'time_pattern': inst_time_pattern,
                'sentiment': self._classify_institutional_sentiment(inst_flow_direction, inst_percentage),
                'status': 'calculated'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing institutional flow: {e}")
            return {'status': 'error'}
    
    def _analyze_retail_flow(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze retail volume flow"""
        try:
            # Filter for retail-sized trades
            retail_data = data[data['volume'] <= self.retail_threshold]
            
            if retail_data.empty:
                return {'status': 'no_retail_activity'}
            
            # Calculate metrics
            total_retail_volume = retail_data['volume'].sum()
            total_volume = data['volume'].sum()
            retail_percentage = total_retail_volume / total_volume if total_volume > 0 else 0
            
            # Analyze retail preference by strike
            retail_by_strike = retail_data.groupby('strike')['volume'].sum()
            popular_strikes = retail_by_strike.nlargest(10).index.tolist()
            
            # Calculate retail flow direction
            retail_flow_direction = self._calculate_flow_direction(retail_data)
            
            # Retail activity pattern
            retail_pattern = self._analyze_retail_pattern(retail_data)
            
            return {
                'total_volume': int(total_retail_volume),
                'percentage': float(retail_percentage),
                'popular_strikes': popular_strikes,
                'flow_direction': retail_flow_direction,
                'activity_pattern': retail_pattern,
                'sentiment': self._classify_retail_sentiment(retail_flow_direction, retail_percentage),
                'status': 'calculated'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing retail flow: {e}")
            return {'status': 'error'}
    
    def _detect_smart_money(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect smart money flow patterns"""
        try:
            smart_money_indicators = {
                'detected': False,
                'confidence': 0.0,
                'strikes': [],
                'characteristics': [],
                'flow_type': 'neutral'
            }
            
            # Look for specific patterns
            patterns = []
            
            # Pattern 1: Large volume at specific strikes with low OI
            large_volume_low_oi = self._detect_large_volume_low_oi(data)
            if large_volume_low_oi['detected']:
                patterns.append('large_volume_low_oi')
                smart_money_indicators['strikes'].extend(large_volume_low_oi['strikes'])
            
            # Pattern 2: Concentrated institutional activity
            concentrated_inst = self._detect_concentrated_institutional(data)
            if concentrated_inst['detected']:
                patterns.append('concentrated_institutional')
                smart_money_indicators['strikes'].extend(concentrated_inst['strikes'])
            
            # Pattern 3: Strategic strike selection
            strategic_strikes = self._detect_strategic_strikes(data)
            if strategic_strikes['detected']:
                patterns.append('strategic_strikes')
                smart_money_indicators['strikes'].extend(strategic_strikes['strikes'])
            
            # Calculate confidence
            if patterns:
                smart_money_indicators['detected'] = True
                smart_money_indicators['confidence'] = len(patterns) / 3.0
                smart_money_indicators['characteristics'] = patterns
                smart_money_indicators['flow_type'] = self._classify_smart_money_flow(data, smart_money_indicators['strikes'])
            
            # Remove duplicates from strikes
            smart_money_indicators['strikes'] = list(set(smart_money_indicators['strikes']))
            
            return smart_money_indicators
            
        except Exception as e:
            logger.error(f"Error detecting smart money: {e}")
            return {'detected': False, 'confidence': 0.0}
    
    def _analyze_volume_characteristics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze overall volume characteristics"""
        try:
            analysis = {
                'put_call_volume_ratio': 0.0,
                'volume_trend': 'neutral',
                'volume_momentum': 0.0,
                'distribution_skew': 'neutral',
                'concentration_level': 'normal'
            }
            
            # Calculate put-call volume ratio
            ce_volume = data[data['option_type'] == 'CE']['volume'].sum()
            pe_volume = data[data['option_type'] == 'PE']['volume'].sum()
            
            if ce_volume > 0:
                analysis['put_call_volume_ratio'] = pe_volume / ce_volume
            
            # Analyze volume trend
            if 'timestamp' in data.columns:
                volume_by_time = data.groupby('timestamp')['volume'].sum()
                if len(volume_by_time) >= self.lookback_period:
                    analysis['volume_trend'] = self._calculate_volume_trend(volume_by_time)
                    analysis['volume_momentum'] = self._calculate_volume_momentum(volume_by_time)
            
            # Analyze distribution
            analysis['distribution_skew'] = self._analyze_distribution_skew(data)
            analysis['concentration_level'] = self._analyze_concentration_level(data)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing volume characteristics: {e}")
            return {}
    
    def _detect_unusual_activity(self,
                               option_data: pd.DataFrame,
                               flow_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect unusual volume activity"""
        try:
            unusual_activity = []
            
            # Check for volume spikes
            volume_spikes = self._detect_volume_spikes(option_data)
            unusual_activity.extend(volume_spikes)
            
            # Check for unusual strike activity
            unusual_strikes = self._detect_unusual_strike_activity(option_data)
            unusual_activity.extend(unusual_strikes)
            
            # Check for unusual time-based activity
            if self.enable_time_analysis:
                unusual_time = self._detect_unusual_time_activity(option_data)
                unusual_activity.extend(unusual_time)
            
            # Check for smart money alerts
            for option_type in ['CE', 'PE']:
                if option_type in flow_results['smart_money'] and flow_results['smart_money'][option_type]['detected']:
                    unusual_activity.append({
                        'type': f'{option_type}_smart_money_detected',
                        'confidence': flow_results['smart_money'][option_type]['confidence'],
                        'strikes': flow_results['smart_money'][option_type]['strikes']
                    })
            
            return unusual_activity
            
        except Exception as e:
            logger.error(f"Error detecting unusual activity: {e}")
            return []
    
    def _generate_flow_signals(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals from volume flow"""
        try:
            signals = {
                'primary_signal': 'neutral',
                'signal_strength': 0.0,
                'flow_direction': 'mixed',
                'institutional_bias': 'neutral',
                'smart_money_signal': 'none',
                'unusual_activity_count': len(results['unusual_activity'])
            }
            
            # Analyze institutional flow
            inst_scores = []
            for option_type in ['CE', 'PE']:
                if option_type in results['institutional_flow'] and results['institutional_flow'][option_type].get('status') == 'calculated':
                    sentiment = results['institutional_flow'][option_type]['sentiment']
                    if sentiment == 'bullish':
                        inst_scores.append(1 if option_type == 'CE' else -1)
                    elif sentiment == 'bearish':
                        inst_scores.append(-1 if option_type == 'CE' else 1)
            
            if inst_scores:
                avg_inst_score = np.mean(inst_scores)
                if avg_inst_score > 0.5:
                    signals['institutional_bias'] = 'bullish'
                elif avg_inst_score < -0.5:
                    signals['institutional_bias'] = 'bearish'
            
            # Analyze smart money
            smart_money_detected = False
            for option_type in ['CE', 'PE']:
                if option_type in results['smart_money'] and results['smart_money'][option_type]['detected']:
                    smart_money_detected = True
                    flow_type = results['smart_money'][option_type]['flow_type']
                    if flow_type in ['bullish', 'bearish']:
                        signals['smart_money_signal'] = flow_type
            
            # Analyze volume characteristics
            if 'volume_analysis' in results and results['volume_analysis']:
                pcr = results['volume_analysis'].get('put_call_volume_ratio', 1.0)
                if pcr > 1.5:
                    signals['flow_direction'] = 'bearish'
                elif pcr < 0.7:
                    signals['flow_direction'] = 'bullish'
                else:
                    signals['flow_direction'] = 'neutral'
            
            # Generate primary signal
            signal_components = []
            
            if signals['institutional_bias'] != 'neutral':
                signal_components.append(signals['institutional_bias'])
            
            if signals['smart_money_signal'] != 'none':
                signal_components.append(signals['smart_money_signal'])
            
            if signals['flow_direction'] != 'neutral':
                signal_components.append(signals['flow_direction'])
            
            if signal_components:
                # Count bullish vs bearish signals
                bullish_count = signal_components.count('bullish')
                bearish_count = signal_components.count('bearish')
                
                if bullish_count > bearish_count:
                    signals['primary_signal'] = 'bullish_flow'
                    signals['signal_strength'] = bullish_count / len(signal_components)
                elif bearish_count > bullish_count:
                    signals['primary_signal'] = 'bearish_flow'
                    signals['signal_strength'] = -bearish_count / len(signal_components)
                else:
                    signals['primary_signal'] = 'mixed_flow'
                    signals['signal_strength'] = 0.0
            
            # Adjust for unusual activity
            if signals['unusual_activity_count'] > 3:
                signals['primary_signal'] = 'high_activity_' + signals['primary_signal']
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating flow signals: {e}")
            return {
                'primary_signal': 'neutral',
                'signal_strength': 0.0
            }
    
    def _classify_flow_regime(self, results: Dict[str, Any]) -> str:
        """Classify volume flow regime"""
        try:
            # Check for smart money activity
            smart_money_count = sum(
                1 for opt in ['CE', 'PE']
                if opt in results['smart_money'] and results['smart_money'][opt]['detected']
            )
            
            if smart_money_count >= 2:
                return 'smart_money_accumulation_regime'
            elif smart_money_count == 1:
                return 'selective_accumulation_regime'
            
            # Check institutional vs retail balance
            inst_percentages = []
            retail_percentages = []
            
            for option_type in ['CE', 'PE']:
                if option_type in results['institutional_flow'] and results['institutional_flow'][option_type].get('status') == 'calculated':
                    inst_percentages.append(results['institutional_flow'][option_type]['percentage'])
                if option_type in results['retail_flow'] and results['retail_flow'][option_type].get('status') == 'calculated':
                    retail_percentages.append(results['retail_flow'][option_type]['percentage'])
            
            if inst_percentages:
                avg_inst = np.mean(inst_percentages)
                if avg_inst > 0.6:
                    return 'institutional_dominated_regime'
                elif avg_inst < 0.2:
                    return 'retail_dominated_regime'
            
            # Check volume characteristics
            if 'volume_analysis' in results and results['volume_analysis']:
                trend = results['volume_analysis'].get('volume_trend', 'neutral')
                if trend == 'increasing':
                    return 'expanding_activity_regime'
                elif trend == 'decreasing':
                    return 'contracting_activity_regime'
            
            return 'balanced_flow_regime'
            
        except Exception as e:
            logger.error(f"Error classifying flow regime: {e}")
            return 'undefined'
    
    def _calculate_volume_distribution(self, strike_volume: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume distribution metrics"""
        try:
            total_volume = strike_volume['volume_sum'].sum()
            if total_volume == 0:
                return {'top_5_concentration': 0.0, 'gini_coefficient': 0.0}
            
            # Top 5 strike concentration
            top_5_volume = strike_volume['volume_sum'].nlargest(5).sum()
            top_5_concentration = top_5_volume / total_volume
            
            # Gini coefficient for volume distribution
            volumes = strike_volume['volume_sum'].values
            gini = self._calculate_gini_coefficient(volumes)
            
            return {
                'top_5_concentration': float(top_5_concentration),
                'gini_coefficient': float(gini)
            }
            
        except:
            return {'top_5_concentration': 0.0, 'gini_coefficient': 0.0}
    
    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """Calculate Gini coefficient for distribution"""
        if len(values) == 0 or np.sum(values) == 0:
            return 0.0
        
        sorted_values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        
        return (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * cumsum[-1]) - (n + 1) / n
    
    def _analyze_volume_by_time(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume by time periods"""
        try:
            time_analysis = {}
            
            for period_name, (start_time, end_time) in self.time_windows.items():
                period_data = data[
                    (pd.to_datetime(data['timestamp']).dt.time >= start_time) &
                    (pd.to_datetime(data['timestamp']).dt.time < end_time)
                ]
                
                if not period_data.empty:
                    time_analysis[period_name] = {
                        'volume': int(period_data['volume'].sum()),
                        'percentage': float(period_data['volume'].sum() / data['volume'].sum())
                    }
            
            return time_analysis
            
        except:
            return {}
    
    def _calculate_volume_concentration(self, strike_volume: pd.DataFrame) -> str:
        """Calculate volume concentration level"""
        try:
            distribution = self._calculate_volume_distribution(strike_volume)
            
            if distribution['top_5_concentration'] > 0.8:
                return 'extreme_concentration'
            elif distribution['top_5_concentration'] > 0.6:
                return 'high_concentration'
            elif distribution['top_5_concentration'] < 0.3:
                return 'low_concentration'
            else:
                return 'normal_concentration'
                
        except:
            return 'unknown'
    
    def _calculate_flow_direction(self, data: pd.DataFrame) -> str:
        """Calculate flow direction from volume and price changes"""
        try:
            # Group by timestamp and calculate volume-weighted price
            time_groups = data.groupby('timestamp').agg({
                'volume': 'sum',
                'price': 'mean'
            })
            
            if len(time_groups) < 2:
                return 'neutral'
            
            # Calculate price change and volume
            price_change = time_groups['price'].iloc[-1] - time_groups['price'].iloc[0]
            total_volume = time_groups['volume'].sum()
            
            if price_change > 0 and total_volume > data['volume'].mean() * self.unusual_volume_multiplier:
                return 'strong_bullish'
            elif price_change > 0:
                return 'bullish'
            elif price_change < 0 and total_volume > data['volume'].mean() * self.unusual_volume_multiplier:
                return 'strong_bearish'
            elif price_change < 0:
                return 'bearish'
            else:
                return 'neutral'
                
        except:
            return 'neutral'
    
    def _classify_institutional_sentiment(self, flow_direction: str, percentage: float) -> str:
        """Classify institutional sentiment"""
        if percentage < 0.1:
            return 'absent'
        
        if flow_direction in ['strong_bullish', 'bullish'] and percentage > 0.5:
            return 'bullish'
        elif flow_direction in ['strong_bearish', 'bearish'] and percentage > 0.5:
            return 'bearish'
        elif percentage > 0.7:
            return 'dominant_neutral'
        else:
            return 'neutral'
    
    def _analyze_institutional_timing(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze timing of institutional activity"""
        try:
            timing_pattern = {}
            
            for period_name, (start_time, end_time) in self.time_windows.items():
                period_data = data[
                    (pd.to_datetime(data['timestamp']).dt.time >= start_time) &
                    (pd.to_datetime(data['timestamp']).dt.time < end_time)
                ]
                
                if not period_data.empty:
                    timing_pattern[period_name] = {
                        'volume': int(period_data['volume'].sum()),
                        'trades': len(period_data),
                        'avg_size': float(period_data['volume'].mean())
                    }
            
            return timing_pattern
            
        except:
            return {}
    
    def _analyze_retail_pattern(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze retail trading patterns"""
        try:
            pattern = {
                'avg_trade_size': float(data['volume'].mean()),
                'trade_frequency': len(data),
                'preferred_time': 'unknown'
            }
            
            if 'timestamp' in data.columns:
                # Find most active time period
                max_volume = 0
                for period_name, (start_time, end_time) in self.time_windows.items():
                    period_data = data[
                        (pd.to_datetime(data['timestamp']).dt.time >= start_time) &
                        (pd.to_datetime(data['timestamp']).dt.time < end_time)
                    ]
                    
                    period_volume = period_data['volume'].sum()
                    if period_volume > max_volume:
                        max_volume = period_volume
                        pattern['preferred_time'] = period_name
            
            return pattern
            
        except:
            return {}
    
    def _classify_retail_sentiment(self, flow_direction: str, percentage: float) -> str:
        """Classify retail sentiment"""
        if percentage < 0.1:
            return 'minimal'
        
        if flow_direction in ['strong_bullish', 'bullish'] and percentage > 0.3:
            return 'bullish'
        elif flow_direction in ['strong_bearish', 'bearish'] and percentage > 0.3:
            return 'bearish'
        elif percentage > 0.5:
            return 'active_neutral'
        else:
            return 'neutral'
    
    def _detect_large_volume_low_oi(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect large volume with low OI pattern"""
        try:
            # Calculate volume to OI ratio by strike
            strike_data = data.groupby('strike').agg({
                'volume': 'sum',
                'oi': 'last'
            })
            
            strike_data['vol_oi_ratio'] = strike_data['volume'] / (strike_data['oi'] + 1)
            
            # Find strikes with high ratio
            high_ratio_threshold = strike_data['vol_oi_ratio'].quantile(0.9)
            suspicious_strikes = strike_data[
                strike_data['vol_oi_ratio'] > high_ratio_threshold
            ].index.tolist()
            
            return {
                'detected': len(suspicious_strikes) > 0,
                'strikes': suspicious_strikes
            }
            
        except:
            return {'detected': False, 'strikes': []}
    
    def _detect_concentrated_institutional(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect concentrated institutional activity"""
        try:
            # Filter institutional trades
            inst_data = data[data['volume'] >= self.institutional_threshold]
            
            if inst_data.empty:
                return {'detected': False, 'strikes': []}
            
            # Check concentration
            strike_counts = inst_data['strike'].value_counts()
            
            # Look for strikes with multiple institutional trades
            concentrated_strikes = strike_counts[
                strike_counts >= 3
            ].index.tolist()
            
            return {
                'detected': len(concentrated_strikes) > 0,
                'strikes': concentrated_strikes
            }
            
        except:
            return {'detected': False, 'strikes': []}
    
    def _detect_strategic_strikes(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect strategic strike selection patterns"""
        try:
            # Identify ATM strike (simplified - using median)
            spot_price = data['underlying_price'].iloc[-1] if 'underlying_price' in data.columns else data['strike'].median()
            
            # Look for activity at key strikes
            key_strikes = []
            strike_data = data.groupby('strike')['volume'].sum()
            
            # Round number strikes near ATM
            for strike in strike_data.index:
                if abs(strike - spot_price) / spot_price < 0.05:  # Within 5% of ATM
                    if strike % 100 == 0:  # Round number
                        if strike_data[strike] > strike_data.mean():
                            key_strikes.append(strike)
            
            return {
                'detected': len(key_strikes) > 0,
                'strikes': key_strikes
            }
            
        except:
            return {'detected': False, 'strikes': []}
    
    def _classify_smart_money_flow(self, data: pd.DataFrame, strikes: List[float]) -> str:
        """Classify smart money flow type"""
        try:
            if not strikes:
                return 'neutral'
            
            # Analyze activity at smart money strikes
            smart_data = data[data['strike'].isin(strikes)]
            
            ce_volume = smart_data[smart_data['option_type'] == 'CE']['volume'].sum()
            pe_volume = smart_data[smart_data['option_type'] == 'PE']['volume'].sum()
            
            if ce_volume > pe_volume * 1.5:
                return 'bullish'
            elif pe_volume > ce_volume * 1.5:
                return 'bearish'
            else:
                return 'neutral'
                
        except:
            return 'neutral'
    
    def _calculate_volume_trend(self, volume_series: pd.Series) -> str:
        """Calculate volume trend"""
        try:
            if len(volume_series) < 5:
                return 'neutral'
            
            # Calculate moving average
            ma = volume_series.rolling(window=self.volume_ma_period).mean()
            
            # Check trend
            recent_ma = ma.iloc[-5:].values
            slope = np.polyfit(range(5), recent_ma, 1)[0]
            
            if slope > volume_series.mean() * 0.1:
                return 'increasing'
            elif slope < -volume_series.mean() * 0.1:
                return 'decreasing'
            else:
                return 'stable'
                
        except:
            return 'neutral'
    
    def _calculate_volume_momentum(self, volume_series: pd.Series) -> float:
        """Calculate volume momentum"""
        try:
            if len(volume_series) < self.lookback_period:
                return 0.0
            
            # Calculate rate of change
            current_volume = volume_series.iloc[-self.flow_smoothing_period:].mean()
            previous_volume = volume_series.iloc[-self.lookback_period:-self.flow_smoothing_period].mean()
            
            if previous_volume > 0:
                momentum = (current_volume - previous_volume) / previous_volume
                return float(np.clip(momentum, -1, 1))
            
            return 0.0
            
        except:
            return 0.0
    
    def _analyze_distribution_skew(self, data: pd.DataFrame) -> str:
        """Analyze volume distribution skew"""
        try:
            ce_volume = data[data['option_type'] == 'CE'].groupby('strike')['volume'].sum()
            pe_volume = data[data['option_type'] == 'PE'].groupby('strike')['volume'].sum()
            
            if ce_volume.sum() > pe_volume.sum() * 1.3:
                return 'call_skewed'
            elif pe_volume.sum() > ce_volume.sum() * 1.3:
                return 'put_skewed'
            else:
                return 'balanced'
                
        except:
            return 'neutral'
    
    def _analyze_concentration_level(self, data: pd.DataFrame) -> str:
        """Analyze overall concentration level"""
        try:
            strike_volume = data.groupby('strike')['volume'].sum()
            
            # Calculate concentration metrics
            total_strikes = len(strike_volume)
            volume_80_pct = strike_volume.sum() * 0.8
            
            # Count strikes needed for 80% of volume
            sorted_volumes = strike_volume.sort_values(ascending=False)
            cumsum = sorted_volumes.cumsum()
            strikes_for_80pct = len(cumsum[cumsum <= volume_80_pct]) + 1
            
            concentration_ratio = strikes_for_80pct / total_strikes
            
            if concentration_ratio < 0.2:
                return 'extreme_concentration'
            elif concentration_ratio < 0.4:
                return 'high_concentration'
            elif concentration_ratio > 0.7:
                return 'low_concentration'
            else:
                return 'normal_concentration'
                
        except:
            return 'normal'
    
    def _detect_volume_spikes(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect volume spikes"""
        spikes = []
        
        try:
            # Calculate average volume
            avg_volume = data.groupby('strike')['volume'].mean().mean()
            
            # Find strikes with unusual volume
            strike_volume = data.groupby('strike')['volume'].sum()
            spike_threshold = avg_volume * self.unusual_volume_multiplier
            
            for strike, volume in strike_volume.items():
                if volume > spike_threshold:
                    spikes.append({
                        'type': 'volume_spike',
                        'strike': strike,
                        'volume': int(volume),
                        'multiplier': float(volume / avg_volume)
                    })
                    
        except:
            pass
        
        return spikes
    
    def _detect_unusual_strike_activity(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect unusual activity at specific strikes"""
        unusual_strikes = []
        
        try:
            # Look for strikes with unusual OI changes
            if 'oi_change' in data.columns:
                strike_oi_change = data.groupby('strike')['oi_change'].sum()
                
                # Find extreme OI changes
                oi_std = strike_oi_change.std()
                oi_mean = strike_oi_change.mean()
                
                for strike, oi_change in strike_oi_change.items():
                    if abs(oi_change - oi_mean) > 2 * oi_std:
                        unusual_strikes.append({
                            'type': 'unusual_oi_change',
                            'strike': strike,
                            'oi_change': int(oi_change),
                            'std_deviations': float(abs(oi_change - oi_mean) / oi_std)
                        })
                        
        except:
            pass
        
        return unusual_strikes
    
    def _detect_unusual_time_activity(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect unusual time-based activity"""
        unusual_time = []
        
        try:
            # Analyze volume by minute
            minute_volume = data.groupby(pd.to_datetime(data['timestamp']).dt.floor('1min'))['volume'].sum()
            
            if len(minute_volume) > 10:
                # Find volume spikes by minute
                minute_avg = minute_volume.mean()
                minute_std = minute_volume.std()
                
                for timestamp, volume in minute_volume.items():
                    if volume > minute_avg + 2 * minute_std:
                        unusual_time.append({
                            'type': 'time_volume_spike',
                            'timestamp': timestamp,
                            'volume': int(volume),
                            'std_deviations': float((volume - minute_avg) / minute_std)
                        })
                        
        except:
            pass
        
        return unusual_time
    
    def _update_flow_history(self, results: Dict[str, Any]):
        """Update flow history for tracking"""
        try:
            timestamp = datetime.now()
            
            # Update institutional flow history
            for option_type in ['CE', 'PE']:
                if option_type in results['institutional_flow'] and results['institutional_flow'][option_type].get('status') == 'calculated':
                    self.flow_history['institutional'].append({
                        'timestamp': timestamp,
                        'option_type': option_type,
                        'percentage': results['institutional_flow'][option_type]['percentage'],
                        'sentiment': results['institutional_flow'][option_type]['sentiment']
                    })
            
            # Update smart money detections
            for option_type in ['CE', 'PE']:
                if option_type in results['smart_money'] and results['smart_money'][option_type]['detected']:
                    self.flow_history['smart_money'].append({
                        'timestamp': timestamp,
                        'option_type': option_type,
                        'confidence': results['smart_money'][option_type]['confidence'],
                        'flow_type': results['smart_money'][option_type]['flow_type']
                    })
            
            # Track unusual activity
            if results['unusual_activity']:
                self.flow_history['unusual_activity'].extend([
                    {**activity, 'timestamp': timestamp} for activity in results['unusual_activity']
                ])
            
            # Track regime
            self.flow_history['flow_regimes'].append({
                'timestamp': timestamp,
                'regime': results['regime']
            })
            
            # Keep only recent history
            max_history = 100
            for key in self.flow_history:
                if len(self.flow_history[key]) > max_history:
                    self.flow_history[key] = self.flow_history[key][-max_history:]
                    
        except Exception as e:
            logger.error(f"Error updating flow history: {e}")
    
    def _get_default_results(self) -> Dict[str, Any]:
        """Get default results structure"""
        return {
            'volume_profile': {},
            'institutional_flow': {},
            'retail_flow': {},
            'smart_money': {},
            'volume_analysis': {},
            'unusual_activity': [],
            'flow_signals': {
                'primary_signal': 'neutral',
                'signal_strength': 0.0
            },
            'regime': 'undefined'
        }
    
    def get_flow_summary(self) -> Dict[str, Any]:
        """Get summary of volume flow analysis"""
        try:
            summary = {
                'total_institutional_events': len(self.flow_history['institutional']),
                'total_smart_money_detections': len(self.flow_history['smart_money']),
                'recent_unusual_activity': 0,
                'dominant_flow': 'neutral',
                'regime_distribution': {}
            }
            
            # Count recent unusual activity
            recent_time = datetime.now() - pd.Timedelta(minutes=30)
            summary['recent_unusual_activity'] = sum(
                1 for activity in self.flow_history['unusual_activity']
                if activity.get('timestamp', datetime.min) > recent_time
            )
            
            # Determine dominant flow
            if self.flow_history['institutional']:
                recent_sentiments = [
                    h['sentiment'] for h in self.flow_history['institutional'][-10:]
                ]
                bullish_count = recent_sentiments.count('bullish')
                bearish_count = recent_sentiments.count('bearish')
                
                if bullish_count > bearish_count:
                    summary['dominant_flow'] = 'institutional_bullish'
                elif bearish_count > bullish_count:
                    summary['dominant_flow'] = 'institutional_bearish'
            
            # Calculate regime distribution
            if self.flow_history['flow_regimes']:
                regimes = [r['regime'] for r in self.flow_history['flow_regimes']]
                total = len(regimes)
                
                for regime in set(regimes):
                    count = regimes.count(regime)
                    summary['regime_distribution'][regime] = count / total
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting flow summary: {e}")
            return {'status': 'error'}