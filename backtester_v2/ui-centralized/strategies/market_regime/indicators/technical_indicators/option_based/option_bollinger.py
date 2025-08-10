"""
Option Bollinger Bands - Bollinger Bands for Options
===================================================

Calculates Bollinger Bands on option prices, implied volatility, and Greeks
to identify overbought/oversold conditions and volatility expansions.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class OptionBollinger:
    """
    Option-specific Bollinger Bands implementation
    
    Features:
    - Price-based Bollinger Bands for options
    - IV-based bands for volatility regime
    - Greek-based bands for sensitivity analysis
    - Band squeeze detection
    - Band breakout identification
    - Put-Call band divergence
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Option Bollinger Bands calculator"""
        # Bollinger parameters
        self.period = config.get('period', 20)
        self.num_std = config.get('num_std', 2.0)
        self.squeeze_threshold = config.get('squeeze_threshold', 0.1)
        
        # Component settings
        self.enable_price_bands = config.get('enable_price_bands', True)
        self.enable_iv_bands = config.get('enable_iv_bands', True)
        self.enable_greek_bands = config.get('enable_greek_bands', True)
        
        # Band width thresholds
        self.narrow_band_threshold = config.get('narrow_band_threshold', 0.05)
        self.wide_band_threshold = config.get('wide_band_threshold', 0.20)
        
        # Breakout settings
        self.breakout_confirmation_periods = config.get('breakout_confirmation_periods', 2)
        self.breakout_strength_threshold = config.get('breakout_strength_threshold', 0.5)
        
        # Advanced features
        self.enable_band_divergence = config.get('enable_band_divergence', True)
        self.enable_squeeze_detection = config.get('enable_squeeze_detection', True)
        self.enable_multi_timeframe = config.get('enable_multi_timeframe', True)
        
        # History tracking
        self.band_history = {
            'price': {'upper': [], 'middle': [], 'lower': [], 'width': []},
            'iv': {'upper': [], 'middle': [], 'lower': [], 'width': []},
            'greek': {'upper': [], 'middle': [], 'lower': [], 'width': []},
            'squeezes': [],
            'breakouts': []
        }
        
        logger.info(f"OptionBollinger initialized: period={self.period}, std={self.num_std}")
    
    def calculate_option_bollinger(self,
                                 option_data: pd.DataFrame,
                                 option_type: str = 'both') -> Dict[str, Any]:
        """
        Calculate comprehensive option Bollinger Bands
        
        Args:
            option_data: DataFrame with option prices, IV, Greeks
            option_type: 'CE', 'PE', or 'both'
            
        Returns:
            Dict with band values, signals, and analysis
        """
        try:
            results = {
                'price_bands': {},
                'iv_bands': {},
                'greek_bands': {},
                'band_analysis': {},
                'signals': {},
                'squeezes': [],
                'breakouts': [],
                'regime': None
            }
            
            # Calculate for each option type
            if option_type in ['CE', 'both']:
                ce_data = option_data[option_data['option_type'] == 'CE']
                if not ce_data.empty:
                    if self.enable_price_bands:
                        results['price_bands']['CE'] = self._calculate_price_bands(ce_data)
                    if self.enable_iv_bands:
                        results['iv_bands']['CE'] = self._calculate_iv_bands(ce_data)
                    if self.enable_greek_bands:
                        results['greek_bands']['CE'] = self._calculate_greek_bands(ce_data)
            
            if option_type in ['PE', 'both']:
                pe_data = option_data[option_data['option_type'] == 'PE']
                if not pe_data.empty:
                    if self.enable_price_bands:
                        results['price_bands']['PE'] = self._calculate_price_bands(pe_data)
                    if self.enable_iv_bands:
                        results['iv_bands']['PE'] = self._calculate_iv_bands(pe_data)
                    if self.enable_greek_bands:
                        results['greek_bands']['PE'] = self._calculate_greek_bands(pe_data)
            
            # Analyze bands
            results['band_analysis'] = self._analyze_bands(results)
            
            # Detect squeezes
            if self.enable_squeeze_detection:
                results['squeezes'] = self._detect_band_squeezes(results)
            
            # Detect breakouts
            results['breakouts'] = self._detect_breakouts(option_data, results)
            
            # Generate signals
            results['signals'] = self._generate_bollinger_signals(results)
            
            # Classify regime
            results['regime'] = self._classify_band_regime(results)
            
            # Update history
            self._update_band_history(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating option Bollinger Bands: {e}")
            return self._get_default_results()
    
    def _calculate_price_bands(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Bollinger Bands on option prices"""
        try:
            # Aggregate prices by timestamp
            price_series = data.groupby('timestamp')['price'].mean()
            
            if len(price_series) < self.period:
                return self._get_default_band_values()
            
            # Calculate moving average
            sma = price_series.rolling(window=self.period).mean()
            
            # Calculate standard deviation
            std = price_series.rolling(window=self.period).std()
            
            # Calculate bands
            upper_band = sma + (self.num_std * std)
            lower_band = sma - (self.num_std * std)
            
            # Get latest values
            latest_price = price_series.iloc[-1]
            latest_sma = sma.iloc[-1]
            latest_std = std.iloc[-1]
            latest_upper = upper_band.iloc[-1]
            latest_lower = lower_band.iloc[-1]
            
            # Calculate band width
            band_width = (latest_upper - latest_lower) / latest_sma if latest_sma > 0 else 0
            
            # Calculate position within bands
            position = self._calculate_band_position(latest_price, latest_upper, latest_lower)
            
            return {
                'upper': float(latest_upper),
                'middle': float(latest_sma),
                'lower': float(latest_lower),
                'current_price': float(latest_price),
                'band_width': float(band_width),
                'position': position,
                'trend': self._calculate_band_trend(sma),
                'volatility': self._classify_volatility(band_width),
                'status': 'calculated'
            }
            
        except Exception as e:
            logger.error(f"Error calculating price bands: {e}")
            return self._get_default_band_values()
    
    def _calculate_iv_bands(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Bollinger Bands on implied volatility"""
        try:
            # Aggregate IV by timestamp
            iv_series = data.groupby('timestamp')['iv'].mean()
            
            if len(iv_series) < self.period:
                return self._get_default_band_values()
            
            # Calculate moving average
            sma = iv_series.rolling(window=self.period).mean()
            
            # Calculate standard deviation
            std = iv_series.rolling(window=self.period).std()
            
            # Calculate bands
            upper_band = sma + (self.num_std * std)
            lower_band = sma - (self.num_std * std)
            
            # Get latest values
            latest_iv = iv_series.iloc[-1]
            latest_sma = sma.iloc[-1]
            latest_upper = upper_band.iloc[-1]
            latest_lower = lower_band.iloc[-1]
            
            # Calculate band width
            band_width = (latest_upper - latest_lower) / latest_sma if latest_sma > 0 else 0
            
            # Calculate position within bands
            position = self._calculate_band_position(latest_iv, latest_upper, latest_lower)
            
            return {
                'upper': float(latest_upper),
                'middle': float(latest_sma),
                'lower': float(latest_lower),
                'current_iv': float(latest_iv),
                'band_width': float(band_width),
                'position': position,
                'iv_regime': self._classify_iv_regime(position, band_width),
                'volatility_trend': self._calculate_band_trend(sma),
                'status': 'calculated'
            }
            
        except Exception as e:
            logger.error(f"Error calculating IV bands: {e}")
            return self._get_default_band_values()
    
    def _calculate_greek_bands(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Bollinger Bands on Greeks"""
        try:
            results = {}
            
            # Calculate bands for each Greek
            for greek in ['delta', 'gamma', 'vega', 'theta']:
                if greek in data.columns:
                    greek_series = data.groupby('timestamp')[greek].mean()
                    
                    if len(greek_series) >= self.period:
                        # Calculate moving average
                        sma = greek_series.rolling(window=self.period).mean()
                        
                        # Calculate standard deviation
                        std = greek_series.rolling(window=self.period).std()
                        
                        # Calculate bands
                        upper_band = sma + (self.num_std * std)
                        lower_band = sma - (self.num_std * std)
                        
                        # Get latest values
                        latest_value = greek_series.iloc[-1]
                        latest_upper = upper_band.iloc[-1]
                        latest_lower = lower_band.iloc[-1]
                        
                        # Calculate position
                        position = self._calculate_band_position(latest_value, latest_upper, latest_lower)
                        
                        results[greek] = {
                            'upper': float(latest_upper),
                            'middle': float(sma.iloc[-1]),
                            'lower': float(latest_lower),
                            'current': float(latest_value),
                            'position': position
                        }
            
            # Calculate composite Greek band analysis
            if results:
                positions = [r['position'] for r in results.values()]
                results['composite'] = {
                    'average_position': np.mean(positions),
                    'greek_alignment': self._check_greek_alignment(positions),
                    'extreme_greeks': self._identify_extreme_greeks(results)
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating Greek bands: {e}")
            return {}
    
    def _analyze_bands(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze band characteristics across all components"""
        try:
            analysis = {
                'band_convergence': {},
                'volatility_state': {},
                'trend_alignment': {},
                'overall_regime': None
            }
            
            # Analyze for each option type
            for option_type in ['CE', 'PE']:
                band_widths = []
                positions = []
                
                # Collect band metrics
                if option_type in results['price_bands'] and results['price_bands'][option_type]['status'] == 'calculated':
                    band_widths.append(results['price_bands'][option_type]['band_width'])
                    positions.append(results['price_bands'][option_type]['position'])
                
                if option_type in results['iv_bands'] and results['iv_bands'][option_type]['status'] == 'calculated':
                    band_widths.append(results['iv_bands'][option_type]['band_width'])
                    positions.append(results['iv_bands'][option_type]['position'])
                
                # Analyze convergence
                if band_widths:
                    avg_width = np.mean(band_widths)
                    analysis['band_convergence'][option_type] = self._classify_band_convergence(avg_width)
                
                # Analyze position
                if positions:
                    avg_position = np.mean(positions)
                    analysis['volatility_state'][option_type] = self._classify_volatility_state(avg_position)
            
            # Overall regime
            analysis['overall_regime'] = self._determine_overall_regime(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing bands: {e}")
            return {}
    
    def _detect_band_squeezes(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect Bollinger Band squeezes"""
        try:
            squeezes = []
            
            # Check for each option type
            for option_type in ['CE', 'PE']:
                # Price band squeeze
                if option_type in results['price_bands'] and results['price_bands'][option_type]['status'] == 'calculated':
                    band_width = results['price_bands'][option_type]['band_width']
                    
                    if band_width < self.squeeze_threshold:
                        squeezes.append({
                            'type': f'{option_type}_price_squeeze',
                            'band_width': band_width,
                            'severity': self._calculate_squeeze_severity(band_width),
                            'potential_breakout': self._estimate_breakout_potential(band_width)
                        })
                
                # IV band squeeze
                if option_type in results['iv_bands'] and results['iv_bands'][option_type]['status'] == 'calculated':
                    band_width = results['iv_bands'][option_type]['band_width']
                    
                    if band_width < self.squeeze_threshold:
                        squeezes.append({
                            'type': f'{option_type}_iv_squeeze',
                            'band_width': band_width,
                            'severity': self._calculate_squeeze_severity(band_width),
                            'volatility_compression': True
                        })
            
            return squeezes
            
        except Exception as e:
            logger.error(f"Error detecting band squeezes: {e}")
            return []
    
    def _detect_breakouts(self,
                        option_data: pd.DataFrame,
                        band_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect band breakouts"""
        try:
            breakouts = []
            
            # Check for each option type
            for option_type in ['CE', 'PE']:
                # Price band breakout
                if option_type in band_results['price_bands'] and band_results['price_bands'][option_type]['status'] == 'calculated':
                    position = band_results['price_bands'][option_type]['position']
                    
                    if position > 1.0:  # Above upper band
                        breakouts.append({
                            'type': f'{option_type}_price_breakout',
                            'direction': 'upper',
                            'strength': position - 1.0,
                            'confirmed': self._confirm_breakout(position)
                        })
                    elif position < 0.0:  # Below lower band
                        breakouts.append({
                            'type': f'{option_type}_price_breakout',
                            'direction': 'lower',
                            'strength': abs(position),
                            'confirmed': self._confirm_breakout(position)
                        })
                
                # IV band breakout
                if option_type in band_results['iv_bands'] and band_results['iv_bands'][option_type]['status'] == 'calculated':
                    position = band_results['iv_bands'][option_type]['position']
                    
                    if abs(position) > 1.0 or abs(position) < 0.0:
                        breakouts.append({
                            'type': f'{option_type}_iv_breakout',
                            'direction': 'upper' if position > 1.0 else 'lower',
                            'strength': abs(position - 0.5),
                            'volatility_event': True
                        })
            
            return breakouts
            
        except Exception as e:
            logger.error(f"Error detecting breakouts: {e}")
            return []
    
    def _generate_bollinger_signals(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals from Bollinger Bands"""
        try:
            signals = {
                'primary_signal': 'neutral',
                'signal_strength': 0.0,
                'band_signals': [],
                'squeeze_signals': [],
                'breakout_signals': []
            }
            
            # Analyze band positions
            positions = []
            for option_type in ['CE', 'PE']:
                if option_type in results['price_bands'] and results['price_bands'][option_type]['status'] == 'calculated':
                    positions.append(results['price_bands'][option_type]['position'])
            
            if positions:
                avg_position = np.mean(positions)
                
                # Generate primary signal based on position
                if avg_position > 0.8:
                    signals['primary_signal'] = 'overbought'
                    signals['signal_strength'] = -(avg_position - 0.5)
                elif avg_position < 0.2:
                    signals['primary_signal'] = 'oversold'
                    signals['signal_strength'] = (0.5 - avg_position)
                elif 0.4 <= avg_position <= 0.6:
                    signals['primary_signal'] = 'neutral'
                    signals['signal_strength'] = 0.0
                else:
                    signals['primary_signal'] = 'trending'
                    signals['signal_strength'] = (avg_position - 0.5) * 2
            
            # Add squeeze signals
            if results['squeezes']:
                signals['squeeze_signals'] = [s['type'] for s in results['squeezes']]
                if len(results['squeezes']) >= 2:
                    signals['primary_signal'] = 'consolidation'
            
            # Add breakout signals
            if results['breakouts']:
                confirmed_breakouts = [b for b in results['breakouts'] if b.get('confirmed', False)]
                if confirmed_breakouts:
                    signals['breakout_signals'] = [b['direction'] for b in confirmed_breakouts]
                    if any(b['direction'] == 'upper' for b in confirmed_breakouts):
                        signals['primary_signal'] = 'bullish_breakout'
                        signals['signal_strength'] = 0.8
                    elif any(b['direction'] == 'lower' for b in confirmed_breakouts):
                        signals['primary_signal'] = 'bearish_breakout'
                        signals['signal_strength'] = -0.8
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating Bollinger signals: {e}")
            return {
                'primary_signal': 'neutral',
                'signal_strength': 0.0
            }
    
    def _classify_band_regime(self, results: Dict[str, Any]) -> str:
        """Classify market regime based on bands"""
        try:
            # Check for squeezes
            if results['squeezes']:
                if len(results['squeezes']) >= 2:
                    return 'extreme_compression_regime'
                else:
                    return 'compression_regime'
            
            # Check for breakouts
            if results['breakouts']:
                confirmed_breakouts = [b for b in results['breakouts'] if b.get('confirmed', False)]
                if confirmed_breakouts:
                    upper_breakouts = sum(1 for b in confirmed_breakouts if b['direction'] == 'upper')
                    lower_breakouts = sum(1 for b in confirmed_breakouts if b['direction'] == 'lower')
                    
                    if upper_breakouts > lower_breakouts:
                        return 'bullish_expansion_regime'
                    elif lower_breakouts > upper_breakouts:
                        return 'bearish_expansion_regime'
                    else:
                        return 'volatile_regime'
            
            # Check band analysis
            if 'overall_regime' in results['band_analysis'] and results['band_analysis']['overall_regime']:
                return results['band_analysis']['overall_regime']
            
            return 'normal_volatility_regime'
            
        except Exception as e:
            logger.error(f"Error classifying band regime: {e}")
            return 'undefined'
    
    def _calculate_band_position(self, value: float, upper: float, lower: float) -> float:
        """Calculate position within bands (0 = lower, 1 = upper)"""
        if upper == lower:
            return 0.5
        return (value - lower) / (upper - lower)
    
    def _calculate_band_trend(self, sma_series: pd.Series) -> str:
        """Calculate band trend direction"""
        try:
            if len(sma_series) < 5:
                return 'neutral'
            
            recent_sma = sma_series.tail(5).values
            
            # Calculate slope
            slope = np.polyfit(range(5), recent_sma, 1)[0]
            
            if slope > 0.01:
                return 'rising'
            elif slope < -0.01:
                return 'falling'
            else:
                return 'neutral'
                
        except:
            return 'neutral'
    
    def _classify_volatility(self, band_width: float) -> str:
        """Classify volatility based on band width"""
        if band_width < self.narrow_band_threshold:
            return 'extremely_low_volatility'
        elif band_width < self.squeeze_threshold:
            return 'low_volatility'
        elif band_width > self.wide_band_threshold:
            return 'high_volatility'
        elif band_width > self.wide_band_threshold * 1.5:
            return 'extremely_high_volatility'
        else:
            return 'normal_volatility'
    
    def _classify_iv_regime(self, position: float, band_width: float) -> str:
        """Classify IV regime based on bands"""
        if band_width < self.narrow_band_threshold:
            return 'iv_compression'
        elif position > 0.8:
            return 'high_iv_regime'
        elif position < 0.2:
            return 'low_iv_regime'
        else:
            return 'normal_iv_regime'
    
    def _check_greek_alignment(self, positions: List[float]) -> str:
        """Check if Greeks are aligned in their band positions"""
        if not positions:
            return 'no_data'
        
        std_dev = np.std(positions)
        
        if std_dev < 0.1:
            return 'strongly_aligned'
        elif std_dev < 0.2:
            return 'aligned'
        elif std_dev < 0.3:
            return 'mixed'
        else:
            return 'divergent'
    
    def _identify_extreme_greeks(self, greek_results: Dict[str, Any]) -> List[str]:
        """Identify Greeks at extreme band positions"""
        extreme_greeks = []
        
        for greek, data in greek_results.items():
            if greek != 'composite' and isinstance(data, dict) and 'position' in data:
                if data['position'] > 0.9 or data['position'] < 0.1:
                    extreme_greeks.append(greek)
        
        return extreme_greeks
    
    def _classify_band_convergence(self, avg_width: float) -> str:
        """Classify band convergence state"""
        if avg_width < self.narrow_band_threshold:
            return 'extreme_convergence'
        elif avg_width < self.squeeze_threshold:
            return 'convergence'
        elif avg_width > self.wide_band_threshold:
            return 'divergence'
        else:
            return 'normal'
    
    def _classify_volatility_state(self, avg_position: float) -> str:
        """Classify volatility state based on position"""
        if avg_position > 0.8:
            return 'high_volatility_state'
        elif avg_position < 0.2:
            return 'low_volatility_state'
        else:
            return 'normal_volatility_state'
    
    def _determine_overall_regime(self, analysis: Dict[str, Any]) -> str:
        """Determine overall regime from analysis"""
        # Count convergence states
        convergence_states = list(analysis['band_convergence'].values())
        if convergence_states.count('extreme_convergence') >= 2:
            return 'pre_breakout_regime'
        elif convergence_states.count('convergence') >= 2:
            return 'consolidation_regime'
        elif convergence_states.count('divergence') >= 2:
            return 'expansion_regime'
        else:
            return 'mixed_regime'
    
    def _calculate_squeeze_severity(self, band_width: float) -> str:
        """Calculate squeeze severity"""
        if band_width < self.narrow_band_threshold:
            return 'extreme'
        elif band_width < self.squeeze_threshold * 0.7:
            return 'high'
        else:
            return 'moderate'
    
    def _estimate_breakout_potential(self, band_width: float) -> float:
        """Estimate breakout potential from squeeze"""
        # The tighter the squeeze, the higher the potential
        return 1.0 - (band_width / self.squeeze_threshold)
    
    def _confirm_breakout(self, position: float) -> bool:
        """Confirm if breakout is valid"""
        # Simplified - in production would check multiple periods
        return abs(position - 0.5) > self.breakout_strength_threshold
    
    def _update_band_history(self, results: Dict[str, Any]):
        """Update band history for tracking"""
        try:
            timestamp = datetime.now()
            
            # Update price band history
            for option_type in ['CE', 'PE']:
                if option_type in results['price_bands'] and results['price_bands'][option_type]['status'] == 'calculated':
                    self.band_history['price']['width'].append({
                        'timestamp': timestamp,
                        'option_type': option_type,
                        'value': results['price_bands'][option_type]['band_width']
                    })
            
            # Track squeezes and breakouts
            if results['squeezes']:
                self.band_history['squeezes'].extend(results['squeezes'])
            
            if results['breakouts']:
                self.band_history['breakouts'].extend(results['breakouts'])
            
            # Keep only recent history
            max_history = 100
            for component in ['price', 'iv', 'greek']:
                for metric in self.band_history[component]:
                    if len(self.band_history[component][metric]) > max_history:
                        self.band_history[component][metric] = self.band_history[component][metric][-max_history:]
                        
        except Exception as e:
            logger.error(f"Error updating band history: {e}")
    
    def _get_default_band_values(self) -> Dict[str, Any]:
        """Get default band values"""
        return {
            'upper': 0.0,
            'middle': 0.0,
            'lower': 0.0,
            'band_width': 0.0,
            'position': 0.5,
            'status': 'insufficient_data'
        }
    
    def _get_default_results(self) -> Dict[str, Any]:
        """Get default results structure"""
        return {
            'price_bands': {},
            'iv_bands': {},
            'greek_bands': {},
            'band_analysis': {},
            'signals': {
                'primary_signal': 'neutral',
                'signal_strength': 0.0
            },
            'squeezes': [],
            'breakouts': [],
            'regime': 'undefined'
        }
    
    def get_bollinger_summary(self) -> Dict[str, Any]:
        """Get summary of Bollinger Band analysis"""
        try:
            summary = {
                'total_squeezes': len(self.band_history['squeezes']),
                'total_breakouts': len(self.band_history['breakouts']),
                'recent_squeeze_count': 0,
                'recent_breakout_count': 0,
                'average_band_width': {},
                'volatility_trend': 'stable'
            }
            
            # Count recent events
            recent_time = datetime.now() - pd.Timedelta(minutes=30)
            summary['recent_squeeze_count'] = sum(
                1 for s in self.band_history['squeezes']
                if isinstance(s, dict) and s.get('timestamp', datetime.min) > recent_time
            )
            summary['recent_breakout_count'] = sum(
                1 for b in self.band_history['breakouts']
                if isinstance(b, dict) and b.get('timestamp', datetime.min) > recent_time
            )
            
            # Calculate average band widths
            if self.band_history['price']['width']:
                recent_widths = [w['value'] for w in self.band_history['price']['width'][-20:]]
                if recent_widths:
                    summary['average_band_width']['price'] = np.mean(recent_widths)
                    
                    # Determine volatility trend
                    if len(recent_widths) >= 5:
                        first_half = np.mean(recent_widths[:len(recent_widths)//2])
                        second_half = np.mean(recent_widths[len(recent_widths)//2:])
                        
                        if second_half > first_half * 1.2:
                            summary['volatility_trend'] = 'expanding'
                        elif second_half < first_half * 0.8:
                            summary['volatility_trend'] = 'contracting'
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting Bollinger summary: {e}")
            return {'status': 'error'}