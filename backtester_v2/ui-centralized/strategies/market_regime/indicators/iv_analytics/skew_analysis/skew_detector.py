"""
Skew Detector - Volatility Skew Detection and Analysis
=====================================================

Detects and analyzes volatility skew patterns for trading signals.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SkewDetector:
    """
    Volatility skew detection and analysis
    
    Features:
    - Put-call skew measurement
    - Skew slope calculation
    - Extreme skew detection
    - Skew trading signals
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Skew Detector"""
        self.extreme_skew_threshold = config.get('extreme_skew_threshold', 0.15)
        self.skew_change_threshold = config.get('skew_change_threshold', 0.05)
        self.lookback_periods = config.get('lookback_periods', 20)
        
        # History tracking
        self.skew_history = {
            'put_call_skew': [],
            'skew_slope': [],
            'extreme_events': []
        }
        
        logger.info(f"SkewDetector initialized: extreme_threshold={self.extreme_skew_threshold}")
    
    def detect_skew_patterns(self, option_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect volatility skew patterns"""
        try:
            results = {
                'put_call_skew': 0.0,
                'skew_slope': 0.0,
                'skew_regime': 'normal',
                'extreme_skew': False,
                'skew_signals': [],
                'skew_momentum': 0.0
            }
            
            # Calculate put-call skew
            results['put_call_skew'] = self._calculate_put_call_skew(option_data)
            
            # Calculate skew slope
            results['skew_slope'] = self._calculate_skew_slope(option_data)
            
            # Classify skew regime
            results['skew_regime'] = self._classify_skew_regime(
                results['put_call_skew'], results['skew_slope']
            )
            
            # Detect extreme skew
            results['extreme_skew'] = self._detect_extreme_skew(
                results['put_call_skew'], results['skew_slope']
            )
            
            # Calculate skew momentum
            results['skew_momentum'] = self._calculate_skew_momentum()
            
            # Generate skew signals
            results['skew_signals'] = self._generate_skew_signals(results)
            
            # Update history
            self._update_skew_history(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error detecting skew patterns: {e}")
            return self._get_default_results()
    
    def _calculate_put_call_skew(self, option_data: pd.DataFrame) -> float:
        """Calculate put-call skew"""
        try:
            # Filter for ATM options (within 5% of spot)
            if 'moneyness' in option_data.columns:
                atm_data = option_data[
                    (option_data['moneyness'] >= 0.95) & 
                    (option_data['moneyness'] <= 1.05)
                ]
            else:
                # Use all data if moneyness not available
                atm_data = option_data
            
            # Separate puts and calls
            puts = atm_data[atm_data['option_type'] == 'PE']
            calls = atm_data[atm_data['option_type'] == 'CE']
            
            if len(puts) > 0 and len(calls) > 0:
                put_iv = puts['iv'].mean()
                call_iv = calls['iv'].mean()
                
                # Put-call skew (put IV - call IV)
                skew = put_iv - call_iv
                return float(skew)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating put-call skew: {e}")
            return 0.0
    
    def _calculate_skew_slope(self, option_data: pd.DataFrame) -> float:
        """Calculate skew slope across strikes"""
        try:
            # Calculate skew slope for each expiry
            slopes = []
            
            for dte_group in option_data.groupby('dte'):
                dte, group = dte_group
                
                if len(group) < 3:
                    continue
                
                # Sort by strike
                group = group.sort_values('strike')
                
                # Calculate slope of IV vs log(strike/spot)
                if 'underlying_price' in group.columns:
                    spot = group['underlying_price'].iloc[0]
                    log_moneyness = np.log(group['strike'] / spot)
                else:
                    # Use normalized strikes
                    log_moneyness = np.log(group['strike'] / group['strike'].median())
                
                iv = group['iv'].values
                
                if len(log_moneyness) >= 3:
                    # Linear fit to get slope
                    slope = np.polyfit(log_moneyness, iv, 1)[0]
                    slopes.append(slope)
            
            if slopes:
                return float(np.mean(slopes))
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating skew slope: {e}")
            return 0.0
    
    def _classify_skew_regime(self, put_call_skew: float, skew_slope: float) -> str:
        """Classify the skew regime"""
        try:
            # Extreme skew conditions
            if abs(put_call_skew) > self.extreme_skew_threshold:
                if put_call_skew > 0:
                    return 'extreme_put_skew'
                else:
                    return 'extreme_call_skew'
            
            # Normal skew conditions
            if put_call_skew > 0.05:
                return 'put_skew'
            elif put_call_skew < -0.05:
                return 'call_skew'
            elif abs(skew_slope) > 0.5:
                return 'high_skew_slope'
            else:
                return 'normal_skew'
                
        except Exception as e:
            logger.error(f"Error classifying skew regime: {e}")
            return 'unknown'
    
    def _detect_extreme_skew(self, put_call_skew: float, skew_slope: float) -> bool:
        """Detect extreme skew conditions"""
        try:
            return (
                abs(put_call_skew) > self.extreme_skew_threshold or
                abs(skew_slope) > 1.0
            )
        except:
            return False
    
    def _calculate_skew_momentum(self) -> float:
        """Calculate skew momentum from history"""
        try:
            if len(self.skew_history['put_call_skew']) < 5:
                return 0.0
            
            recent_skews = [
                s['value'] for s in self.skew_history['put_call_skew'][-5:]
            ]
            
            # Calculate momentum as slope of recent skew values
            x = np.arange(len(recent_skews))
            momentum = np.polyfit(x, recent_skews, 1)[0]
            
            return float(momentum)
            
        except Exception as e:
            logger.error(f"Error calculating skew momentum: {e}")
            return 0.0
    
    def _generate_skew_signals(self, results: Dict[str, Any]) -> List[str]:
        """Generate trading signals from skew analysis"""
        try:
            signals = []
            
            skew_regime = results['skew_regime']
            extreme_skew = results['extreme_skew']
            skew_momentum = results['skew_momentum']
            
            # Extreme skew signals
            if extreme_skew:
                signals.append('extreme_skew_detected')
                
                if skew_regime == 'extreme_put_skew':
                    signals.append('extreme_put_skew')
                elif skew_regime == 'extreme_call_skew':
                    signals.append('extreme_call_skew')
            
            # Momentum signals
            if abs(skew_momentum) > 0.02:
                if skew_momentum > 0:
                    signals.append('skew_momentum_increasing')
                else:
                    signals.append('skew_momentum_decreasing')
            
            # Mean reversion signals
            if skew_regime in ['extreme_put_skew', 'extreme_call_skew']:
                if skew_momentum * results['put_call_skew'] < 0:  # Momentum opposite to skew
                    signals.append('skew_mean_reversion')
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating skew signals: {e}")
            return []
    
    def _update_skew_history(self, results: Dict[str, Any]):
        """Update skew history for tracking"""
        try:
            timestamp = datetime.now()
            
            # Update put-call skew history
            self.skew_history['put_call_skew'].append({
                'timestamp': timestamp,
                'value': results['put_call_skew'],
                'regime': results['skew_regime']
            })
            
            # Update skew slope history
            self.skew_history['skew_slope'].append({
                'timestamp': timestamp,
                'value': results['skew_slope']
            })
            
            # Record extreme events
            if results['extreme_skew']:
                self.skew_history['extreme_events'].append({
                    'timestamp': timestamp,
                    'type': results['skew_regime'],
                    'put_call_skew': results['put_call_skew'],
                    'skew_slope': results['skew_slope']
                })
            
            # Keep only recent history
            max_history = 100
            for key in self.skew_history:
                if len(self.skew_history[key]) > max_history:
                    self.skew_history[key] = self.skew_history[key][-max_history:]
                    
        except Exception as e:
            logger.error(f"Error updating skew history: {e}")
    
    def _get_default_results(self) -> Dict[str, Any]:
        """Get default results structure"""
        return {
            'put_call_skew': 0.0,
            'skew_slope': 0.0,
            'skew_regime': 'unknown',
            'extreme_skew': False,
            'skew_signals': [],
            'skew_momentum': 0.0
        }
    
    def get_skew_summary(self) -> Dict[str, Any]:
        """Get comprehensive skew analysis summary"""
        try:
            if not self.skew_history['put_call_skew']:
                return {'status': 'no_history'}
            
            recent_skews = [
                s['value'] for s in self.skew_history['put_call_skew'][-20:]
            ]
            
            return {
                'current_skew': recent_skews[-1] if recent_skews else 0.0,
                'average_skew': np.mean(recent_skews),
                'skew_volatility': np.std(recent_skews),
                'extreme_events_count': len(self.skew_history['extreme_events']),
                'skew_persistence': self._calculate_skew_persistence()
            }
            
        except Exception as e:
            logger.error(f"Error getting skew summary: {e}")
            return {'status': 'error'}
    
    def _calculate_skew_persistence(self) -> float:
        """Calculate how persistent skew regimes have been"""
        try:
            if len(self.skew_history['put_call_skew']) < 10:
                return 0.5
            
            recent_regimes = [
                s['regime'] for s in self.skew_history['put_call_skew'][-10:]
            ]
            
            # Count regime changes
            changes = 0
            for i in range(1, len(recent_regimes)):
                if recent_regimes[i] != recent_regimes[i-1]:
                    changes += 1
            
            # Fewer changes = more persistent
            persistence = 1.0 - (changes / (len(recent_regimes) - 1))
            return float(persistence)
            
        except:
            return 0.5