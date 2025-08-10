"""
Baseline Tracker - 9:15 AM Logic Implementation
==============================================

Handles session baseline tracking with 9:15 AM logic preservation.
This module maintains the original baseline tracking system while adding
enhancements for better performance and reliability.

Author: Market Regime Refactoring Team
Date: 2025-07-06
Version: 2.0.0 - Modular Architecture
"""

from typing import Dict, Any, Optional
from datetime import datetime, time
import logging
import numpy as np

logger = logging.getLogger(__name__)

class BaselineTracker:
    """
    Session baseline tracker with preserved 9:15 AM logic
    
    Maintains session baselines for Greek values with exponential smoothing
    and periodic updates. Preserves the original logic while adding
    enhanced tracking capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize baseline tracker"""
        self.config = config or {}
        
        # Configuration parameters (preserved from original)
        self.baseline_update_frequency = self.config.get('baseline_update_frequency', 30)  # minutes
        self.smoothing_alpha = self.config.get('smoothing_alpha', 0.1)
        self.market_open_time = time(9, 15)  # 9:15 AM IST
        self.baseline_establishment_window = self.config.get('establishment_window', 15)  # minutes
        
        # Session baseline storage
        self.session_baselines: Dict[str, Dict[str, Any]] = {}
        
        # Enhanced tracking
        self.baseline_quality_metrics = {}
        self.update_history = []
        
        logger.info("BaselineTracker initialized with 9:15 AM logic preservation")
    
    def update_baselines(self, weighted_greeks: Dict[str, float]) -> Dict[str, float]:
        """
        Update session baselines with preserved 9:15 AM logic
        
        Args:
            weighted_greeks: Current weighted Greek values
            
        Returns:
            Dict[str, float]: Current session baselines
        """
        try:
            current_time = datetime.now()
            session_date = current_time.date()
            
            # Check if we need to establish new baselines (new session)
            if session_date not in self.session_baselines:
                return self._establish_new_session_baselines(weighted_greeks, current_time, session_date)
            
            # Check if baselines need updating (periodic updates)
            if self._should_update_baselines(current_time, session_date):
                return self._update_existing_baselines(weighted_greeks, current_time, session_date)
            
            return self.session_baselines[session_date]
            
        except Exception as e:
            logger.error(f"Error updating baselines: {e}")
            return self._get_default_baselines()
    
    def _establish_new_session_baselines(self, 
                                       weighted_greeks: Dict[str, float],
                                       current_time: datetime,
                                       session_date) -> Dict[str, float]:
        """Establish new session baselines (9:15 AM logic)"""
        try:
            # Enhanced baseline establishment
            baseline_quality = self._assess_baseline_quality(weighted_greeks, current_time)
            
            self.session_baselines[session_date] = {
                'delta': weighted_greeks.get('delta', 0),
                'gamma': weighted_greeks.get('gamma', 0),
                'theta': weighted_greeks.get('theta', 0),
                'vega': weighted_greeks.get('vega', 0),
                'established_time': current_time,
                'last_update': current_time,
                'update_count': 1,
                'quality_score': baseline_quality,
                'establishment_method': self._get_establishment_method(current_time)
            }
            
            # Track quality metrics
            self.baseline_quality_metrics[session_date] = {
                'initial_quality': baseline_quality,
                'market_timing': self._assess_market_timing(current_time),
                'data_completeness': self._assess_data_completeness(weighted_greeks)
            }
            
            logger.info(f"Established new Greek baselines for session {session_date} (quality: {baseline_quality:.3f})")
            
            return self.session_baselines[session_date]
            
        except Exception as e:
            logger.error(f"Error establishing new baselines: {e}")
            return self._get_default_baselines()
    
    def _update_existing_baselines(self,
                                 weighted_greeks: Dict[str, float],
                                 current_time: datetime,
                                 session_date) -> Dict[str, float]:
        """Update existing baselines with exponential smoothing"""
        try:
            baselines = self.session_baselines[session_date]
            
            # Apply exponential smoothing (preserved logic)
            for greek in ['delta', 'gamma', 'theta', 'vega']:
                current_value = weighted_greeks.get(greek, 0)
                baseline_value = baselines.get(greek, 0)
                
                # Exponential smoothing update
                new_baseline = (
                    self.smoothing_alpha * current_value + 
                    (1 - self.smoothing_alpha) * baseline_value
                )
                baselines[greek] = new_baseline
            
            # Update metadata
            baselines['last_update'] = current_time
            baselines['update_count'] = baselines.get('update_count', 0) + 1
            
            # Update quality assessment
            current_quality = self._assess_baseline_quality(weighted_greeks, current_time)
            baselines['quality_score'] = (
                0.8 * baselines.get('quality_score', 0.5) + 0.2 * current_quality
            )
            
            # Record update history
            self.update_history.append({
                'timestamp': current_time,
                'session_date': session_date,
                'quality': current_quality,
                'update_method': 'exponential_smoothing'
            })
            
            # Keep only last 100 updates
            if len(self.update_history) > 100:
                self.update_history = self.update_history[-100:]
            
            logger.debug(f"Updated baselines for session {session_date} (quality: {current_quality:.3f})")
            
            return baselines
            
        except Exception as e:
            logger.error(f"Error updating existing baselines: {e}")
            return self.session_baselines[session_date]
    
    def _should_update_baselines(self, current_time: datetime, session_date) -> bool:
        """Check if baselines should be updated"""
        try:
            if session_date not in self.session_baselines:
                return False
            
            last_update = self.session_baselines[session_date]['last_update']
            time_since_update = (current_time - last_update).total_seconds()
            
            return time_since_update > (self.baseline_update_frequency * 60)
            
        except Exception as e:
            logger.error(f"Error checking update criteria: {e}")
            return False
    
    def calculate_baseline_changes(self,
                                 weighted_greeks: Dict[str, float],
                                 baselines: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate changes from session baselines (preserved logic)
        
        Args:
            weighted_greeks: Current Greek values
            baselines: Session baselines
            
        Returns:
            Dict[str, float]: Percentage changes from baselines
        """
        try:
            baseline_changes = {}
            
            for greek in ['delta', 'gamma', 'theta', 'vega']:
                current_value = weighted_greeks.get(greek, 0)
                baseline_value = baselines.get(greek, 0)
                
                # Calculate percentage change from baseline (preserved logic)
                if abs(baseline_value) > 1e-6:  # Avoid division by zero
                    change = (current_value - baseline_value) / abs(baseline_value)
                else:
                    change = current_value  # If baseline is zero, use absolute value
                
                baseline_changes[greek] = change
            
            return baseline_changes
            
        except Exception as e:
            logger.error(f"Error calculating baseline changes: {e}")
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
    
    def _assess_baseline_quality(self, weighted_greeks: Dict[str, float], current_time: datetime) -> float:
        """Assess quality of baseline establishment"""
        try:
            quality_score = 0.0
            
            # Market timing quality (9:15-9:30 AM is best)
            market_time = current_time.time()
            if time(9, 15) <= market_time <= time(9, 30):
                timing_quality = 1.0
            elif time(9, 0) <= market_time <= time(10, 0):
                timing_quality = 0.8
            elif time(9, 0) <= market_time <= time(15, 30):
                timing_quality = 0.6
            else:
                timing_quality = 0.3
            
            # Data completeness quality
            greek_values = [weighted_greeks.get(g, 0) for g in ['delta', 'gamma', 'theta', 'vega']]
            non_zero_greeks = sum(1 for v in greek_values if abs(v) > 1e-6)
            completeness_quality = non_zero_greeks / 4.0
            
            # Value reasonableness quality
            reasonableness_quality = 1.0
            if abs(weighted_greeks.get('delta', 0)) > 1.0:
                reasonableness_quality *= 0.8
            if abs(weighted_greeks.get('gamma', 0)) > 0.1:
                reasonableness_quality *= 0.9
            
            # Combined quality score
            quality_score = (
                timing_quality * 0.4 +
                completeness_quality * 0.4 +
                reasonableness_quality * 0.2
            )
            
            return np.clip(quality_score, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error assessing baseline quality: {e}")
            return 0.5
    
    def _get_establishment_method(self, current_time: datetime) -> str:
        """Get method used for baseline establishment"""
        market_time = current_time.time()
        
        if time(9, 15) <= market_time <= time(9, 30):
            return "market_open_optimal"
        elif time(9, 0) <= market_time <= time(10, 0):
            return "market_open_good"
        elif time(9, 0) <= market_time <= time(15, 30):
            return "intraday_standard"
        else:
            return "after_hours_fallback"
    
    def _assess_market_timing(self, current_time: datetime) -> str:
        """Assess market timing for baseline establishment"""
        market_time = current_time.time()
        
        if time(9, 15) <= market_time <= time(9, 30):
            return "optimal"
        elif time(9, 0) <= market_time <= time(10, 0):
            return "good"
        elif time(9, 0) <= market_time <= time(15, 30):
            return "acceptable"
        else:
            return "suboptimal"
    
    def _assess_data_completeness(self, weighted_greeks: Dict[str, float]) -> float:
        """Assess completeness of Greek data"""
        required_greeks = ['delta', 'gamma', 'theta', 'vega']
        available_greeks = sum(1 for greek in required_greeks 
                             if abs(weighted_greeks.get(greek, 0)) > 1e-6)
        return available_greeks / len(required_greeks)
    
    def _get_default_baselines(self) -> Dict[str, float]:
        """Get default baseline values"""
        return {
            'delta': 0,
            'gamma': 0,
            'theta': 0,
            'vega': 0,
            'established_time': datetime.now(),
            'last_update': datetime.now(),
            'update_count': 0,
            'quality_score': 0.0,
            'establishment_method': 'default_fallback'
        }
    
    def get_baseline_summary(self, session_date=None) -> Dict[str, Any]:
        """Get summary of baseline tracking"""
        if session_date is None:
            session_date = datetime.now().date()
        
        if session_date not in self.session_baselines:
            return {'status': 'no_baselines_established'}
        
        baselines = self.session_baselines[session_date]
        quality_metrics = self.baseline_quality_metrics.get(session_date, {})
        
        return {
            'baselines': {k: v for k, v in baselines.items() if k in ['delta', 'gamma', 'theta', 'vega']},
            'quality_score': baselines.get('quality_score', 0.0),
            'establishment_method': baselines.get('establishment_method', 'unknown'),
            'update_count': baselines.get('update_count', 0),
            'last_update': baselines.get('last_update'),
            'quality_metrics': quality_metrics,
            'status': 'active'
        }
    
    def reset_session_baselines(self, session_date=None):
        """Reset baselines for a session"""
        if session_date is None:
            session_date = datetime.now().date()
        
        if session_date in self.session_baselines:
            del self.session_baselines[session_date]
        
        if session_date in self.baseline_quality_metrics:
            del self.baseline_quality_metrics[session_date]
        
        logger.info(f"Reset baselines for session {session_date}")
    
    def get_baseline_health_status(self) -> Dict[str, Any]:
        """Get health status of baseline tracking system"""
        current_date = datetime.now().date()
        
        health_status = {
            'current_session_active': current_date in self.session_baselines,
            'total_sessions_tracked': len(self.session_baselines),
            'update_history_length': len(self.update_history),
            'average_quality_score': 0.0,
            'status': 'healthy'
        }
        
        if self.session_baselines:
            quality_scores = [
                baselines.get('quality_score', 0.0) 
                for baselines in self.session_baselines.values()
            ]
            health_status['average_quality_score'] = np.mean(quality_scores)
        
        if health_status['current_session_active']:
            current_baselines = self.session_baselines[current_date]
            health_status['current_quality'] = current_baselines.get('quality_score', 0.0)
            health_status['current_update_count'] = current_baselines.get('update_count', 0)
        
        return health_status