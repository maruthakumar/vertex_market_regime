"""
Option MACD - Moving Average Convergence Divergence for Options
==============================================================

Calculates MACD on option prices, implied volatility, and Greeks
to identify momentum changes and trend reversals in option markets.

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


class OptionMACD:
    """
    Option-specific MACD implementation
    
    Features:
    - Price-based MACD for options
    - IV-based MACD for volatility momentum
    - Greek-based MACD for sensitivity momentum
    - Signal line crossovers and histogram analysis
    - Put-Call MACD divergence detection
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Option MACD calculator"""
        # MACD parameters
        self.fast_period = config.get('fast_period', 12)
        self.slow_period = config.get('slow_period', 26)
        self.signal_period = config.get('signal_period', 9)
        
        # Component weights
        self.price_weight = config.get('price_weight', 0.4)
        self.iv_weight = config.get('iv_weight', 0.3)
        self.greek_weight = config.get('greek_weight', 0.3)
        
        # Signal thresholds
        self.histogram_threshold = config.get('histogram_threshold', 0.0)
        self.divergence_threshold = config.get('divergence_threshold', 0.2)
        self.signal_strength_threshold = config.get('signal_strength_threshold', 0.5)
        
        # Advanced settings
        self.enable_multi_timeframe = config.get('enable_multi_timeframe', True)
        self.enable_divergence_detection = config.get('enable_divergence_detection', True)
        self.smoothing_factor = config.get('smoothing_factor', 3)
        
        # History tracking
        self.macd_history = {
            'price': {'macd': [], 'signal': [], 'histogram': []},
            'iv': {'macd': [], 'signal': [], 'histogram': []},
            'greek': {'macd': [], 'signal': [], 'histogram': []},
            'composite': {'macd': [], 'signal': [], 'histogram': []}
        }
        
        logger.info(f"OptionMACD initialized: fast={self.fast_period}, slow={self.slow_period}, signal={self.signal_period}")
    
    def calculate_option_macd(self, 
                            option_data: pd.DataFrame,
                            option_type: str = 'both') -> Dict[str, Any]:
        """
        Calculate comprehensive option MACD
        
        Args:
            option_data: DataFrame with option prices, IV, Greeks
            option_type: 'CE', 'PE', or 'both'
            
        Returns:
            Dict with MACD values, signals, and analysis
        """
        try:
            results = {
                'price_macd': {},
                'iv_macd': {},
                'greek_macd': {},
                'composite_macd': {},
                'signals': {},
                'crossovers': [],
                'divergences': [],
                'momentum': None
            }
            
            # Calculate for each option type
            if option_type in ['CE', 'both']:
                ce_data = option_data[option_data['option_type'] == 'CE']
                if not ce_data.empty:
                    results['price_macd']['CE'] = self._calculate_price_macd(ce_data)
                    results['iv_macd']['CE'] = self._calculate_iv_macd(ce_data)
                    results['greek_macd']['CE'] = self._calculate_greek_macd(ce_data)
            
            if option_type in ['PE', 'both']:
                pe_data = option_data[option_data['option_type'] == 'PE']
                if not pe_data.empty:
                    results['price_macd']['PE'] = self._calculate_price_macd(pe_data)
                    results['iv_macd']['PE'] = self._calculate_iv_macd(pe_data)
                    results['greek_macd']['PE'] = self._calculate_greek_macd(pe_data)
            
            # Calculate composite MACD
            results['composite_macd'] = self._calculate_composite_macd(results)
            
            # Detect crossovers
            results['crossovers'] = self._detect_crossovers(results)
            
            # Generate signals
            results['signals'] = self._generate_macd_signals(results)
            
            # Detect divergences
            if self.enable_divergence_detection:
                results['divergences'] = self._detect_macd_divergences(
                    option_data, results
                )
            
            # Classify momentum
            results['momentum'] = self._classify_momentum(results)
            
            # Update history
            self._update_macd_history(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating option MACD: {e}")
            return self._get_default_results()
    
    def _calculate_price_macd(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate MACD on option prices"""
        try:
            # Aggregate prices by timestamp
            price_series = data.groupby('timestamp')['price'].mean()
            
            if len(price_series) < self.slow_period:
                return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0, 'status': 'insufficient_data'}
            
            # Calculate EMAs
            ema_fast = price_series.ewm(span=self.fast_period, adjust=False).mean()
            ema_slow = price_series.ewm(span=self.slow_period, adjust=False).mean()
            
            # Calculate MACD line
            macd_line = ema_fast - ema_slow
            
            # Calculate signal line
            signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
            
            # Calculate histogram
            histogram = macd_line - signal_line
            
            # Apply smoothing if enabled
            if self.smoothing_factor > 1:
                macd_line = macd_line.rolling(window=self.smoothing_factor).mean()
                signal_line = signal_line.rolling(window=self.smoothing_factor).mean()
                histogram = histogram.rolling(window=self.smoothing_factor).mean()
            
            # Get latest values
            latest_macd = macd_line.iloc[-1]
            latest_signal = signal_line.iloc[-1]
            latest_histogram = histogram.iloc[-1]
            
            return {
                'macd': float(latest_macd),
                'signal': float(latest_signal),
                'histogram': float(latest_histogram),
                'trend': self._calculate_macd_trend(macd_line),
                'strength': self._calculate_macd_strength(latest_histogram),
                'status': 'calculated'
            }
            
        except Exception as e:
            logger.error(f"Error calculating price MACD: {e}")
            return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0, 'status': 'error'}
    
    def _calculate_iv_macd(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate MACD on implied volatility"""
        try:
            # Aggregate IV by timestamp
            iv_series = data.groupby('timestamp')['iv'].mean()
            
            if len(iv_series) < self.slow_period:
                return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0, 'status': 'insufficient_data'}
            
            # Calculate EMAs
            ema_fast = iv_series.ewm(span=self.fast_period, adjust=False).mean()
            ema_slow = iv_series.ewm(span=self.slow_period, adjust=False).mean()
            
            # Calculate MACD line
            macd_line = ema_fast - ema_slow
            
            # Calculate signal line
            signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
            
            # Calculate histogram
            histogram = macd_line - signal_line
            
            # Get latest values
            latest_macd = macd_line.iloc[-1]
            latest_signal = signal_line.iloc[-1]
            latest_histogram = histogram.iloc[-1]
            
            return {
                'macd': float(latest_macd),
                'signal': float(latest_signal),
                'histogram': float(latest_histogram),
                'volatility_momentum': self._classify_iv_momentum(latest_histogram),
                'iv_trend': self._calculate_macd_trend(macd_line),
                'status': 'calculated'
            }
            
        except Exception as e:
            logger.error(f"Error calculating IV MACD: {e}")
            return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0, 'status': 'error'}
    
    def _calculate_greek_macd(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate MACD on Greeks"""
        try:
            results = {}
            composite_histograms = []
            
            # Calculate MACD for each Greek
            for greek in ['delta', 'gamma', 'vega', 'theta']:
                if greek in data.columns:
                    greek_series = data.groupby('timestamp')[greek].mean()
                    
                    if len(greek_series) >= self.slow_period:
                        # Calculate EMAs
                        ema_fast = greek_series.ewm(span=self.fast_period, adjust=False).mean()
                        ema_slow = greek_series.ewm(span=self.slow_period, adjust=False).mean()
                        
                        # Calculate MACD line
                        macd_line = ema_fast - ema_slow
                        
                        # Calculate signal line
                        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
                        
                        # Calculate histogram
                        histogram = macd_line - signal_line
                        
                        results[greek] = {
                            'macd': float(macd_line.iloc[-1]),
                            'signal': float(signal_line.iloc[-1]),
                            'histogram': float(histogram.iloc[-1])
                        }
                        
                        composite_histograms.append(histogram.iloc[-1])
            
            # Calculate composite Greek MACD
            if composite_histograms:
                composite_histogram = np.mean(composite_histograms)
                results['composite'] = {
                    'histogram': composite_histogram,
                    'greek_momentum': self._classify_greek_momentum(composite_histogram)
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating Greek MACD: {e}")
            return {}
    
    def _calculate_composite_macd(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate composite MACD combining all components"""
        try:
            composite_values = {}
            
            # Calculate for each option type
            for option_type in ['CE', 'PE']:
                macd_values = []
                signal_values = []
                histogram_values = []
                weights = []
                
                # Price MACD (highest weight)
                if option_type in results['price_macd'] and results['price_macd'][option_type]['status'] == 'calculated':
                    macd_values.append(results['price_macd'][option_type]['macd'])
                    signal_values.append(results['price_macd'][option_type]['signal'])
                    histogram_values.append(results['price_macd'][option_type]['histogram'])
                    weights.append(self.price_weight)
                
                # IV MACD
                if option_type in results['iv_macd'] and results['iv_macd'][option_type]['status'] == 'calculated':
                    macd_values.append(results['iv_macd'][option_type]['macd'])
                    signal_values.append(results['iv_macd'][option_type]['signal'])
                    histogram_values.append(results['iv_macd'][option_type]['histogram'])
                    weights.append(self.iv_weight)
                
                # Greek MACD
                if option_type in results['greek_macd'] and 'composite' in results['greek_macd'][option_type]:
                    histogram_values.append(results['greek_macd'][option_type]['composite']['histogram'])
                    weights.append(self.greek_weight)
                
                # Calculate weighted composite
                if histogram_values:
                    composite_histogram = np.average(histogram_values[:len(weights)], weights=weights[:len(histogram_values)])
                    composite_values[option_type] = {
                        'histogram': composite_histogram,
                        'momentum': self._classify_histogram_momentum(composite_histogram)
                    }
                    
                    if macd_values and signal_values:
                        composite_values[option_type]['macd'] = np.average(macd_values[:len(weights)], weights=weights[:len(macd_values)])
                        composite_values[option_type]['signal'] = np.average(signal_values[:len(weights)], weights=weights[:len(signal_values)])
            
            # Calculate overall composite
            if composite_values:
                overall_histogram = np.mean([v['histogram'] for v in composite_values.values()])
                composite_values['overall'] = {
                    'histogram': overall_histogram,
                    'signal_strength': self._calculate_signal_strength(overall_histogram)
                }
            
            return composite_values
            
        except Exception as e:
            logger.error(f"Error calculating composite MACD: {e}")
            return {}
    
    def _detect_crossovers(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect MACD crossovers"""
        try:
            crossovers = []
            
            # Check for each option type
            for option_type in ['CE', 'PE']:
                # Price MACD crossover
                if option_type in results['price_macd'] and 'macd' in results['price_macd'][option_type]:
                    price_macd = results['price_macd'][option_type]
                    if self._check_crossover(price_macd['macd'], price_macd['signal']):
                        crossovers.append({
                            'type': f'{option_type}_price_crossover',
                            'direction': 'bullish' if price_macd['macd'] > price_macd['signal'] else 'bearish',
                            'strength': abs(price_macd['histogram'])
                        })
                
                # IV MACD crossover
                if option_type in results['iv_macd'] and 'macd' in results['iv_macd'][option_type]:
                    iv_macd = results['iv_macd'][option_type]
                    if self._check_crossover(iv_macd['macd'], iv_macd['signal']):
                        crossovers.append({
                            'type': f'{option_type}_iv_crossover',
                            'direction': 'bullish' if iv_macd['macd'] > iv_macd['signal'] else 'bearish',
                            'strength': abs(iv_macd['histogram'])
                        })
            
            return crossovers
            
        except Exception as e:
            logger.error(f"Error detecting crossovers: {e}")
            return []
    
    def _generate_macd_signals(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals from MACD values"""
        try:
            signals = {
                'primary_signal': 'neutral',
                'signal_strength': 0.0,
                'bullish_indicators': [],
                'bearish_indicators': [],
                'crossover_signals': []
            }
            
            # Check composite MACD
            if 'overall' in results['composite_macd']:
                histogram = results['composite_macd']['overall']['histogram']
                
                if histogram > self.signal_strength_threshold:
                    signals['primary_signal'] = 'strong_bullish'
                    signals['signal_strength'] = min(histogram / 2, 1.0)
                elif histogram > self.histogram_threshold:
                    signals['primary_signal'] = 'bullish'
                    signals['signal_strength'] = histogram
                elif histogram < -self.signal_strength_threshold:
                    signals['primary_signal'] = 'strong_bearish'
                    signals['signal_strength'] = max(histogram / 2, -1.0)
                elif histogram < -self.histogram_threshold:
                    signals['primary_signal'] = 'bearish'
                    signals['signal_strength'] = histogram
                else:
                    signals['primary_signal'] = 'neutral'
                    signals['signal_strength'] = histogram
            
            # Check individual option types
            for option_type in ['CE', 'PE']:
                if option_type in results['composite_macd']:
                    histogram = results['composite_macd'][option_type]['histogram']
                    
                    if histogram > self.histogram_threshold:
                        signals['bullish_indicators'].append(option_type)
                    elif histogram < -self.histogram_threshold:
                        signals['bearish_indicators'].append(option_type)
            
            # Add crossover signals
            if results['crossovers']:
                signals['crossover_signals'] = [
                    c['direction'] for c in results['crossovers']
                ]
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating MACD signals: {e}")
            return {
                'primary_signal': 'neutral',
                'signal_strength': 0.0
            }
    
    def _detect_macd_divergences(self,
                               option_data: pd.DataFrame,
                               macd_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect MACD divergences"""
        try:
            divergences = []
            
            # Price vs MACD divergence
            price_div = self._check_price_macd_divergence(option_data, macd_results)
            if price_div:
                divergences.append(price_div)
            
            # IV vs MACD divergence
            iv_div = self._check_iv_macd_divergence(option_data, macd_results)
            if iv_div:
                divergences.append(iv_div)
            
            # Put-Call MACD divergence
            pc_div = self._check_put_call_macd_divergence(macd_results)
            if pc_div:
                divergences.append(pc_div)
            
            return divergences
            
        except Exception as e:
            logger.error(f"Error detecting MACD divergences: {e}")
            return []
    
    def _check_crossover(self, macd: float, signal: float) -> bool:
        """Check if a crossover is occurring"""
        # Simplified implementation
        # In production, would check historical values
        return abs(macd - signal) < 0.05
    
    def _check_price_macd_divergence(self,
                                   option_data: pd.DataFrame,
                                   macd_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for price-MACD divergence"""
        # Simplified implementation
        return None
    
    def _check_iv_macd_divergence(self,
                                option_data: pd.DataFrame,
                                macd_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for IV-MACD divergence"""
        # Simplified implementation
        return None
    
    def _check_put_call_macd_divergence(self, macd_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for put-call MACD divergence"""
        try:
            if 'CE' in macd_results['composite_macd'] and 'PE' in macd_results['composite_macd']:
                ce_histogram = macd_results['composite_macd']['CE']['histogram']
                pe_histogram = macd_results['composite_macd']['PE']['histogram']
                
                # Check if histograms have opposite signs
                if (ce_histogram > 0 and pe_histogram < 0) or (ce_histogram < 0 and pe_histogram > 0):
                    divergence = abs(ce_histogram - pe_histogram)
                    
                    if divergence > self.divergence_threshold:
                        return {
                            'type': 'put_call_macd_divergence',
                            'ce_histogram': ce_histogram,
                            'pe_histogram': pe_histogram,
                            'divergence': divergence,
                            'signal': 'bullish' if ce_histogram > pe_histogram else 'bearish'
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking put-call MACD divergence: {e}")
            return None
    
    def _classify_momentum(self, results: Dict[str, Any]) -> str:
        """Classify overall momentum based on MACD"""
        try:
            if 'overall' not in results['composite_macd']:
                return 'undefined'
            
            histogram = results['composite_macd']['overall']['histogram']
            
            if histogram > self.signal_strength_threshold:
                return 'strong_bullish_momentum'
            elif histogram > self.histogram_threshold:
                return 'bullish_momentum'
            elif histogram < -self.signal_strength_threshold:
                return 'strong_bearish_momentum'
            elif histogram < -self.histogram_threshold:
                return 'bearish_momentum'
            elif abs(histogram) < 0.05:
                return 'neutral_momentum'
            else:
                return 'weak_momentum'
                
        except Exception as e:
            logger.error(f"Error classifying momentum: {e}")
            return 'undefined'
    
    def _calculate_macd_trend(self, macd_series: pd.Series) -> str:
        """Calculate MACD trend direction"""
        try:
            if len(macd_series) < 3:
                return 'neutral'
            
            recent_macd = macd_series.tail(3).values
            
            if recent_macd[-1] > recent_macd[-2] > recent_macd[-3]:
                return 'rising'
            elif recent_macd[-1] < recent_macd[-2] < recent_macd[-3]:
                return 'falling'
            else:
                return 'neutral'
                
        except:
            return 'neutral'
    
    def _calculate_macd_strength(self, histogram: float) -> str:
        """Calculate MACD signal strength"""
        abs_histogram = abs(histogram)
        
        if abs_histogram > self.signal_strength_threshold:
            return 'strong'
        elif abs_histogram > self.histogram_threshold:
            return 'moderate'
        else:
            return 'weak'
    
    def _classify_histogram_momentum(self, histogram: float) -> str:
        """Classify momentum based on histogram value"""
        if histogram > self.signal_strength_threshold:
            return 'accelerating_bullish'
        elif histogram > self.histogram_threshold:
            return 'bullish'
        elif histogram < -self.signal_strength_threshold:
            return 'accelerating_bearish'
        elif histogram < -self.histogram_threshold:
            return 'bearish'
        else:
            return 'neutral'
    
    def _classify_iv_momentum(self, histogram: float) -> str:
        """Classify IV momentum"""
        if histogram > 0.1:
            return 'iv_expansion'
        elif histogram < -0.1:
            return 'iv_contraction'
        else:
            return 'stable_iv'
    
    def _classify_greek_momentum(self, histogram: float) -> str:
        """Classify Greek momentum"""
        if histogram > 0.2:
            return 'strong_greek_bullish'
        elif histogram > 0:
            return 'greek_bullish'
        elif histogram < -0.2:
            return 'strong_greek_bearish'
        elif histogram < 0:
            return 'greek_bearish'
        else:
            return 'greek_neutral'
    
    def _calculate_signal_strength(self, histogram: float) -> float:
        """Calculate signal strength from histogram value"""
        # Normalize to [-1, 1] range
        return np.tanh(histogram * 2)
    
    def _update_macd_history(self, results: Dict[str, Any]):
        """Update MACD history for tracking"""
        try:
            timestamp = datetime.now()
            
            # Update composite history
            if 'overall' in results['composite_macd']:
                self.macd_history['composite']['histogram'].append({
                    'timestamp': timestamp,
                    'value': results['composite_macd']['overall']['histogram'],
                    'momentum': results['momentum']
                })
            
            # Keep only recent history
            max_history = 100
            for component in self.macd_history:
                for metric in self.macd_history[component]:
                    if len(self.macd_history[component][metric]) > max_history:
                        self.macd_history[component][metric] = self.macd_history[component][metric][-max_history:]
                        
        except Exception as e:
            logger.error(f"Error updating MACD history: {e}")
    
    def _get_default_results(self) -> Dict[str, Any]:
        """Get default results structure"""
        return {
            'price_macd': {},
            'iv_macd': {},
            'greek_macd': {},
            'composite_macd': {'overall': {'histogram': 0.0}},
            'signals': {
                'primary_signal': 'neutral',
                'signal_strength': 0.0
            },
            'crossovers': [],
            'divergences': [],
            'momentum': 'undefined'
        }
    
    def get_macd_summary(self) -> Dict[str, Any]:
        """Get summary of MACD analysis"""
        try:
            if not self.macd_history['composite']['histogram']:
                return {'status': 'no_history'}
            
            recent_histograms = [h['value'] for h in self.macd_history['composite']['histogram'][-20:]]
            
            return {
                'current_histogram': recent_histograms[-1] if recent_histograms else 0.0,
                'average_histogram': np.mean(recent_histograms),
                'histogram_trend': self._calculate_histogram_trend(recent_histograms),
                'momentum_distribution': self._calculate_momentum_distribution(),
                'total_calculations': len(self.macd_history['composite']['histogram'])
            }
            
        except Exception as e:
            logger.error(f"Error getting MACD summary: {e}")
            return {'status': 'error'}
    
    def _calculate_histogram_trend(self, histograms: List[float]) -> str:
        """Calculate histogram trend"""
        if len(histograms) < 3:
            return 'neutral'
        
        recent = histograms[-3:]
        if recent[-1] > recent[-2] > recent[-3]:
            return 'improving'
        elif recent[-1] < recent[-2] < recent[-3]:
            return 'deteriorating'
        else:
            return 'neutral'
    
    def _calculate_momentum_distribution(self) -> Dict[str, float]:
        """Calculate distribution of momentum states"""
        try:
            if not self.macd_history['composite']['histogram']:
                return {}
            
            momentum_states = [h['momentum'] for h in self.macd_history['composite']['histogram']]
            total = len(momentum_states)
            
            distribution = {}
            for state in set(momentum_states):
                count = momentum_states.count(state)
                distribution[state] = count / total
            
            return distribution
            
        except Exception as e:
            logger.error(f"Error calculating momentum distribution: {e}")
            return {}