"""
OI Velocity Calculator Module

Calculates time-series OI velocity and acceleration metrics to detect:
- Momentum shifts in institutional positioning
- OI flow patterns and directional changes
- Acceleration/deceleration in OI building
- Cross-strike OI migration patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class VelocityPattern(Enum):
    """Types of OI velocity patterns"""
    ACCELERATING_BUILDUP = "accelerating_buildup"
    DECELERATING_BUILDUP = "decelerating_buildup"
    ACCELERATING_UNWINDING = "accelerating_unwinding"
    DECELERATING_UNWINDING = "decelerating_unwinding"
    STABLE = "stable"
    REVERSING = "reversing"


@dataclass
class OIVelocityMetrics:
    """OI velocity and acceleration metrics"""
    ce_velocity: float
    pe_velocity: float
    ce_acceleration: float
    pe_acceleration: float
    net_velocity: float
    net_acceleration: float
    velocity_pattern: VelocityPattern
    momentum_score: float
    velocity_divergence: float
    acceleration_divergence: float
    strike_migration_velocity: float
    cross_strike_flow_rate: float
    velocity_percentile: float
    acceleration_percentile: float
    momentum_exhaustion_score: float
    trend_acceleration_factor: float


class OIVelocityCalculator:
    """
    Calculates OI velocity and acceleration patterns for momentum analysis
    """
    
    def __init__(self, lookback_periods: List[int] = None):
        """
        Initialize OI Velocity Calculator
        
        Args:
            lookback_periods: Periods for velocity calculation
        """
        self.lookback_periods = lookback_periods or [5, 10, 20, 50]
        self.velocity_history = []
        self.acceleration_history = []
        logger.info(f"Initialized OIVelocityCalculator with periods: {self.lookback_periods}")
    
    def calculate_velocity_acceleration(self, df: pd.DataFrame) -> Dict[str, OIVelocityMetrics]:
        """
        Calculate velocity and acceleration for multiple timeframes
        
        Args:
            df: Production data with OI columns
            
        Returns:
            Velocity metrics by timeframe
        """
        metrics_by_period = {}
        
        for period in self.lookback_periods:
            metrics = self._calculate_period_metrics(df, period)
            metrics_by_period[f"period_{period}"] = metrics
        
        # Calculate composite metrics
        composite_metrics = self._calculate_composite_metrics(metrics_by_period)
        metrics_by_period["composite"] = composite_metrics
        
        return metrics_by_period
    
    def _calculate_period_metrics(self, df: pd.DataFrame, period: int) -> OIVelocityMetrics:
        """Calculate metrics for a specific period"""
        
        # Calculate CE velocity and acceleration
        ce_velocity = 0.0
        ce_acceleration = 0.0
        if 'ce_oi' in df.columns:
            ce_changes = df['ce_oi'].diff(period)
            ce_velocity = ce_changes.mean() / period if period > 0 else 0
            ce_acceleration = ce_changes.diff().mean() / (period ** 2) if period > 0 else 0
        
        # Calculate PE velocity and acceleration
        pe_velocity = 0.0
        pe_acceleration = 0.0
        if 'pe_oi' in df.columns:
            pe_changes = df['pe_oi'].diff(period)
            pe_velocity = pe_changes.mean() / period if period > 0 else 0
            pe_acceleration = pe_changes.diff().mean() / (period ** 2) if period > 0 else 0
        
        # Net metrics
        net_velocity = ce_velocity - pe_velocity
        net_acceleration = ce_acceleration - pe_acceleration
        
        # Velocity pattern classification
        velocity_pattern = self._classify_velocity_pattern(net_velocity, net_acceleration)
        
        # Momentum score
        momentum_score = self._calculate_momentum_score(ce_velocity, pe_velocity, ce_acceleration, pe_acceleration)
        
        # Divergence metrics
        velocity_divergence = abs(ce_velocity - pe_velocity) / (abs(ce_velocity) + abs(pe_velocity) + 1e-8)
        acceleration_divergence = abs(ce_acceleration - pe_acceleration) / (abs(ce_acceleration) + abs(pe_acceleration) + 1e-8)
        
        # Strike migration velocity
        strike_migration_velocity = self._calculate_strike_migration_velocity(df, period)
        
        # Cross-strike flow rate
        cross_strike_flow_rate = self._calculate_cross_strike_flow_rate(df, period)
        
        # Percentile calculations
        velocity_percentile = self._calculate_velocity_percentile(net_velocity)
        acceleration_percentile = self._calculate_acceleration_percentile(net_acceleration)
        
        # Momentum exhaustion
        momentum_exhaustion = self._calculate_momentum_exhaustion(ce_velocity, pe_velocity, ce_acceleration, pe_acceleration)
        
        # Trend acceleration factor
        trend_acceleration = self._calculate_trend_acceleration_factor(net_velocity, net_acceleration)
        
        return OIVelocityMetrics(
            ce_velocity=ce_velocity,
            pe_velocity=pe_velocity,
            ce_acceleration=ce_acceleration,
            pe_acceleration=pe_acceleration,
            net_velocity=net_velocity,
            net_acceleration=net_acceleration,
            velocity_pattern=velocity_pattern,
            momentum_score=momentum_score,
            velocity_divergence=velocity_divergence,
            acceleration_divergence=acceleration_divergence,
            strike_migration_velocity=strike_migration_velocity,
            cross_strike_flow_rate=cross_strike_flow_rate,
            velocity_percentile=velocity_percentile,
            acceleration_percentile=acceleration_percentile,
            momentum_exhaustion_score=momentum_exhaustion,
            trend_acceleration_factor=trend_acceleration
        )
    
    def _classify_velocity_pattern(self, velocity: float, acceleration: float) -> VelocityPattern:
        """Classify the velocity pattern"""
        
        if abs(velocity) < 100:  # Low velocity threshold
            return VelocityPattern.STABLE
        
        if velocity > 0:  # Building
            if acceleration > 0:
                return VelocityPattern.ACCELERATING_BUILDUP
            else:
                return VelocityPattern.DECELERATING_BUILDUP
        else:  # Unwinding
            if acceleration < 0:
                return VelocityPattern.ACCELERATING_UNWINDING
            else:
                return VelocityPattern.DECELERATING_UNWINDING
        
        # Check for reversal
        if np.sign(velocity) != np.sign(acceleration) and abs(acceleration) > abs(velocity) * 0.5:
            return VelocityPattern.REVERSING
        
        return VelocityPattern.STABLE
    
    def _calculate_momentum_score(self, ce_vel: float, pe_vel: float, 
                                 ce_acc: float, pe_acc: float) -> float:
        """Calculate composite momentum score"""
        
        # Velocity component
        vel_magnitude = np.sqrt(ce_vel**2 + pe_vel**2)
        vel_direction = np.sign(ce_vel - pe_vel)
        
        # Acceleration component
        acc_magnitude = np.sqrt(ce_acc**2 + pe_acc**2)
        acc_direction = np.sign(ce_acc - pe_acc)
        
        # Check alignment
        alignment = 1.0 if vel_direction == acc_direction else -0.5
        
        # Normalize and combine
        vel_score = np.tanh(vel_magnitude / 10000) * vel_direction
        acc_score = np.tanh(acc_magnitude / 1000) * acc_direction
        
        momentum_score = (vel_score * 0.6 + acc_score * 0.4) * alignment
        
        return np.clip(momentum_score, -1.0, 1.0)
    
    def _calculate_strike_migration_velocity(self, df: pd.DataFrame, period: int) -> float:
        """Calculate velocity of OI migration across strikes"""
        
        if 'call_strike_type' not in df.columns or 'put_strike_type' not in df.columns:
            return 0.0
        
        migration_velocity = 0.0
        
        # Track OI movement from OTM to ATM (or vice versa)
        strike_types = ['OTM2', 'OTM1', 'ATM', 'ITM1', 'ITM2']
        
        for i in range(len(strike_types) - 1):
            from_type = strike_types[i]
            to_type = strike_types[i + 1]
            
            # Calculate OI flow between strike types
            from_oi = df[df['call_strike_type'] == from_type]['ce_oi'].sum() if 'ce_oi' in df.columns else 0
            to_oi = df[df['call_strike_type'] == to_type]['ce_oi'].sum() if 'ce_oi' in df.columns else 0
            
            if from_oi > 0:
                flow_rate = (to_oi - from_oi) / from_oi
                migration_velocity += abs(flow_rate) / period
        
        return migration_velocity
    
    def _calculate_cross_strike_flow_rate(self, df: pd.DataFrame, period: int) -> float:
        """Calculate rate of OI flow across strikes"""
        
        if 'ce_oi' not in df.columns or 'pe_oi' not in df.columns:
            return 0.0
        
        # Group by strike if available
        if 'strike' in df.columns:
            strike_oi = df.groupby('strike')[['ce_oi', 'pe_oi']].sum()
            
            # Calculate flow between consecutive strikes
            ce_flow = strike_oi['ce_oi'].diff().abs().sum()
            pe_flow = strike_oi['pe_oi'].diff().abs().sum()
            
            total_oi = strike_oi.sum().sum()
            
            if total_oi > 0:
                flow_rate = (ce_flow + pe_flow) / (total_oi * period)
                return min(flow_rate, 1.0)
        
        return 0.0
    
    def _calculate_velocity_percentile(self, velocity: float) -> float:
        """Calculate historical percentile of velocity"""
        
        self.velocity_history.append(velocity)
        
        if len(self.velocity_history) < 2:
            return 0.5
        
        # Keep only recent history (last 100 values)
        if len(self.velocity_history) > 100:
            self.velocity_history = self.velocity_history[-100:]
        
        percentile = np.percentile(self.velocity_history, 
                                  [0, 25, 50, 75, 100])
        
        # Find which percentile bucket the current velocity falls into
        if velocity <= percentile[0]:
            return 0.0
        elif velocity <= percentile[1]:
            return 0.25
        elif velocity <= percentile[2]:
            return 0.50
        elif velocity <= percentile[3]:
            return 0.75
        else:
            return 1.0
    
    def _calculate_acceleration_percentile(self, acceleration: float) -> float:
        """Calculate historical percentile of acceleration"""
        
        self.acceleration_history.append(acceleration)
        
        if len(self.acceleration_history) < 2:
            return 0.5
        
        # Keep only recent history
        if len(self.acceleration_history) > 100:
            self.acceleration_history = self.acceleration_history[-100:]
        
        percentile = np.percentile(self.acceleration_history,
                                  [0, 25, 50, 75, 100])
        
        if acceleration <= percentile[0]:
            return 0.0
        elif acceleration <= percentile[1]:
            return 0.25
        elif acceleration <= percentile[2]:
            return 0.50
        elif acceleration <= percentile[3]:
            return 0.75
        else:
            return 1.0
    
    def _calculate_momentum_exhaustion(self, ce_vel: float, pe_vel: float,
                                      ce_acc: float, pe_acc: float) -> float:
        """Calculate momentum exhaustion score"""
        
        # Exhaustion occurs when velocity is high but acceleration is negative
        ce_exhaustion = 0.0
        if abs(ce_vel) > 1000:  # High velocity
            if np.sign(ce_vel) != np.sign(ce_acc):  # Opposite signs
                ce_exhaustion = abs(ce_acc) / (abs(ce_vel) + 1)
        
        pe_exhaustion = 0.0
        if abs(pe_vel) > 1000:
            if np.sign(pe_vel) != np.sign(pe_acc):
                pe_exhaustion = abs(pe_acc) / (abs(pe_vel) + 1)
        
        # Combined exhaustion score
        exhaustion = (ce_exhaustion + pe_exhaustion) / 2
        
        return min(exhaustion, 1.0)
    
    def _calculate_trend_acceleration_factor(self, velocity: float, acceleration: float) -> float:
        """Calculate trend acceleration factor"""
        
        if abs(velocity) < 100:  # Low velocity
            return 0.0
        
        # Acceleration factor is positive when acceleration aligns with velocity
        if np.sign(velocity) == np.sign(acceleration):
            # Trend is accelerating
            factor = abs(acceleration) / (abs(velocity) + 1)
            return min(factor, 1.0)
        else:
            # Trend is decelerating
            factor = -abs(acceleration) / (abs(velocity) + 1)
            return max(factor, -1.0)
    
    def _calculate_composite_metrics(self, period_metrics: Dict[str, OIVelocityMetrics]) -> OIVelocityMetrics:
        """Calculate composite metrics across all periods"""
        
        # Extract metrics from all periods (excluding composite itself)
        velocities = []
        accelerations = []
        momentum_scores = []
        
        for key, metrics in period_metrics.items():
            if key != "composite":
                velocities.append(metrics.net_velocity)
                accelerations.append(metrics.net_acceleration)
                momentum_scores.append(metrics.momentum_score)
        
        # Weighted average (shorter periods get higher weight for responsiveness)
        weights = [0.4, 0.3, 0.2, 0.1][:len(velocities)]
        weights = weights / np.sum(weights)
        
        avg_velocity = np.average(velocities, weights=weights)
        avg_acceleration = np.average(accelerations, weights=weights)
        avg_momentum = np.average(momentum_scores, weights=weights)
        
        # Create composite metrics
        return OIVelocityMetrics(
            ce_velocity=avg_velocity * 0.5,  # Approximate
            pe_velocity=-avg_velocity * 0.5,  # Approximate
            ce_acceleration=avg_acceleration * 0.5,
            pe_acceleration=-avg_acceleration * 0.5,
            net_velocity=avg_velocity,
            net_acceleration=avg_acceleration,
            velocity_pattern=self._classify_velocity_pattern(avg_velocity, avg_acceleration),
            momentum_score=avg_momentum,
            velocity_divergence=np.std(velocities) / (np.mean(np.abs(velocities)) + 1e-8),
            acceleration_divergence=np.std(accelerations) / (np.mean(np.abs(accelerations)) + 1e-8),
            strike_migration_velocity=np.mean([m.strike_migration_velocity for _, m in period_metrics.items() if _ != "composite"]),
            cross_strike_flow_rate=np.mean([m.cross_strike_flow_rate for _, m in period_metrics.items() if _ != "composite"]),
            velocity_percentile=self._calculate_velocity_percentile(avg_velocity),
            acceleration_percentile=self._calculate_acceleration_percentile(avg_acceleration),
            momentum_exhaustion_score=np.mean([m.momentum_exhaustion_score for _, m in period_metrics.items() if _ != "composite"]),
            trend_acceleration_factor=np.mean([m.trend_acceleration_factor for _, m in period_metrics.items() if _ != "composite"])
        )
    
    def detect_momentum_shifts(self, df: pd.DataFrame, threshold: float = 0.3) -> Dict[str, Any]:
        """
        Detect significant momentum shifts in OI patterns
        
        Args:
            df: Production data
            threshold: Significance threshold
            
        Returns:
            Detected momentum shifts
        """
        shifts = {
            'shift_detected': False,
            'shift_type': None,
            'shift_magnitude': 0.0,
            'shift_confidence': 0.0,
            'shift_details': []
        }
        
        # Calculate velocity metrics
        metrics = self.calculate_velocity_acceleration(df)
        composite = metrics.get("composite")
        
        if not composite:
            return shifts
        
        # Check for significant shifts
        if abs(composite.momentum_score) > threshold:
            shifts['shift_detected'] = True
            
            # Determine shift type
            if composite.velocity_pattern == VelocityPattern.ACCELERATING_BUILDUP:
                shifts['shift_type'] = "bullish_acceleration"
            elif composite.velocity_pattern == VelocityPattern.ACCELERATING_UNWINDING:
                shifts['shift_type'] = "bearish_acceleration"
            elif composite.velocity_pattern == VelocityPattern.REVERSING:
                shifts['shift_type'] = "momentum_reversal"
            else:
                shifts['shift_type'] = "momentum_shift"
            
            shifts['shift_magnitude'] = abs(composite.momentum_score)
            shifts['shift_confidence'] = self._calculate_shift_confidence(composite)
            
            # Add details
            shifts['shift_details'].append(f"Velocity: {composite.net_velocity:.2f}")
            shifts['shift_details'].append(f"Acceleration: {composite.net_acceleration:.4f}")
            shifts['shift_details'].append(f"Pattern: {composite.velocity_pattern.value}")
        
        return shifts
    
    def _calculate_shift_confidence(self, metrics: OIVelocityMetrics) -> float:
        """Calculate confidence in momentum shift"""
        
        confidence_factors = []
        
        # High percentile indicates unusual movement
        if metrics.velocity_percentile > 0.75 or metrics.velocity_percentile < 0.25:
            confidence_factors.append(0.3)
        
        # Acceleration alignment
        if metrics.trend_acceleration_factor > 0.5:
            confidence_factors.append(0.3)
        
        # Low exhaustion
        if metrics.momentum_exhaustion_score < 0.3:
            confidence_factors.append(0.2)
        
        # High momentum score
        if abs(metrics.momentum_score) > 0.5:
            confidence_factors.append(0.2)
        
        return min(sum(confidence_factors), 1.0)
    
    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract velocity features for Component 3
        
        Args:
            df: Production data
            
        Returns:
            Array of velocity features
        """
        features = []
        
        # Calculate metrics for all periods
        metrics = self.calculate_velocity_acceleration(df)
        
        # Extract features from composite metrics
        composite = metrics.get("composite")
        if composite:
            features.extend([
                composite.net_velocity / 10000,  # Normalized
                composite.net_acceleration / 1000,  # Normalized
                composite.momentum_score,
                composite.velocity_divergence,
                composite.acceleration_divergence,
                composite.strike_migration_velocity,
                composite.cross_strike_flow_rate,
                composite.velocity_percentile,
                composite.acceleration_percentile,
                composite.momentum_exhaustion_score,
                composite.trend_acceleration_factor
            ])
        
        # Detect momentum shifts
        shifts = self.detect_momentum_shifts(df)
        features.extend([
            float(shifts['shift_detected']),
            shifts['shift_magnitude'],
            shifts['shift_confidence']
        ])
        
        return np.array(features)