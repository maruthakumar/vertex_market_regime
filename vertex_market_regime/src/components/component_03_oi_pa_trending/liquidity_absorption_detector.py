"""
Liquidity Absorption Detector Module

Detects liquidity absorption patterns indicating institutional activity:
- Large OI changes with minimal price impact
- Block trade detection and analysis
- Stealth accumulation/distribution patterns
- Stop hunting and squeeze detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class AbsorptionType(Enum):
    """Types of liquidity absorption patterns"""
    INSTITUTIONAL_ACCUMULATION = "institutional_accumulation"
    INSTITUTIONAL_DISTRIBUTION = "institutional_distribution"
    STOP_HUNTING = "stop_hunting"
    SQUEEZE_SETUP = "squeeze_setup"
    STEALTH_POSITIONING = "stealth_positioning"
    BLOCK_ABSORPTION = "block_absorption"
    NO_ABSORPTION = "no_absorption"


@dataclass
class LiquidityAbsorptionMetrics:
    """Liquidity absorption detection metrics"""
    absorption_score: float
    absorption_type: AbsorptionType
    block_trade_probability: float
    stealth_accumulation_score: float
    distribution_intensity: float
    stop_hunting_probability: float
    squeeze_potential: float
    large_oi_small_price_events: int
    absorption_velocity: float
    absorption_persistence: float
    institutional_footprint: float
    market_maker_activity: float
    liquidity_vacuum_score: float
    price_suppression_score: float
    volume_absorption_ratio: float


class LiquidityAbsorptionDetector:
    """
    Detects liquidity absorption patterns in options markets
    """
    
    def __init__(self, sensitivity_threshold: float = 0.3):
        """
        Initialize Liquidity Absorption Detector
        
        Args:
            sensitivity_threshold: Threshold for absorption detection
        """
        self.sensitivity_threshold = sensitivity_threshold
        self.absorption_history = []
        self.event_log = []
        logger.info(f"Initialized LiquidityAbsorptionDetector with threshold: {sensitivity_threshold}")
    
    def detect_absorption_patterns(self, df: pd.DataFrame) -> LiquidityAbsorptionMetrics:
        """
        Detect liquidity absorption patterns in the data
        
        Args:
            df: Production data with OI, volume, and price columns
            
        Returns:
            Liquidity absorption metrics
        """
        # Calculate absorption score
        absorption_score = self._calculate_absorption_score(df)
        
        # Detect large OI changes with small price impact
        large_oi_events = self._detect_large_oi_small_price_events(df)
        
        # Block trade detection
        block_trade_prob = self._detect_block_trades(df)
        
        # Stealth accumulation analysis
        stealth_score = self._detect_stealth_accumulation(df)
        
        # Distribution intensity
        distribution_intensity = self._calculate_distribution_intensity(df)
        
        # Stop hunting detection
        stop_hunting_prob = self._detect_stop_hunting(df)
        
        # Squeeze potential
        squeeze_potential = self._calculate_squeeze_potential(df)
        
        # Classify absorption type
        absorption_type = self._classify_absorption_type(
            absorption_score, stealth_score, distribution_intensity, stop_hunting_prob
        )
        
        # Absorption velocity
        absorption_velocity = self._calculate_absorption_velocity(df)
        
        # Absorption persistence
        absorption_persistence = self._calculate_absorption_persistence(absorption_score)
        
        # Institutional footprint
        institutional_footprint = self._calculate_institutional_footprint(df)
        
        # Market maker activity
        mm_activity = self._detect_market_maker_activity(df)
        
        # Liquidity vacuum detection
        liquidity_vacuum = self._detect_liquidity_vacuum(df)
        
        # Price suppression analysis
        price_suppression = self._analyze_price_suppression(df)
        
        # Volume absorption ratio
        volume_absorption = self._calculate_volume_absorption_ratio(df)
        
        return LiquidityAbsorptionMetrics(
            absorption_score=absorption_score,
            absorption_type=absorption_type,
            block_trade_probability=block_trade_prob,
            stealth_accumulation_score=stealth_score,
            distribution_intensity=distribution_intensity,
            stop_hunting_probability=stop_hunting_prob,
            squeeze_potential=squeeze_potential,
            large_oi_small_price_events=len(large_oi_events),
            absorption_velocity=absorption_velocity,
            absorption_persistence=absorption_persistence,
            institutional_footprint=institutional_footprint,
            market_maker_activity=mm_activity,
            liquidity_vacuum_score=liquidity_vacuum,
            price_suppression_score=price_suppression,
            volume_absorption_ratio=volume_absorption
        )
    
    def _calculate_absorption_score(self, df: pd.DataFrame) -> float:
        """Calculate overall liquidity absorption score"""
        
        absorption_factors = []
        
        # OI change vs price change ratio
        if 'ce_oi' in df.columns and 'spot' in df.columns:
            oi_change = df['ce_oi'].pct_change().abs().mean()
            price_change = df['spot'].pct_change().abs().mean()
            
            if price_change > 0:
                # High OI change with low price change indicates absorption
                absorption_ratio = oi_change / price_change
                absorption_factors.append(min(absorption_ratio / 10, 1.0))
        
        # Volume concentration
        if 'ce_volume' in df.columns:
            volume_std = df['ce_volume'].std()
            volume_mean = df['ce_volume'].mean()
            
            if volume_mean > 0:
                # High concentration indicates absorption events
                concentration = volume_std / volume_mean
                absorption_factors.append(min(concentration, 1.0))
        
        # OI buildup rate
        if 'ce_oi' in df.columns and 'pe_oi' in df.columns:
            total_oi = df['ce_oi'] + df['pe_oi']
            oi_buildup_rate = total_oi.diff().mean() / (total_oi.mean() + 1)
            absorption_factors.append(abs(oi_buildup_rate) * 10)
        
        if absorption_factors:
            return np.clip(np.mean(absorption_factors), 0.0, 1.0)
        
        return 0.0
    
    def _detect_large_oi_small_price_events(self, df: pd.DataFrame) -> List[int]:
        """Detect events with large OI changes but small price impact"""
        
        events = []
        
        if 'ce_oi' not in df.columns or 'spot' not in df.columns:
            return events
        
        # Calculate rolling statistics
        oi_changes = df['ce_oi'].diff().abs()
        price_changes = df['spot'].pct_change().abs()
        
        # Define thresholds
        oi_threshold = oi_changes.quantile(0.8)  # Top 20% OI changes
        price_threshold = price_changes.quantile(0.3)  # Bottom 30% price changes
        
        # Find events
        for i in range(1, len(df)):
            if oi_changes.iloc[i] > oi_threshold and price_changes.iloc[i] < price_threshold:
                events.append(i)
                self.event_log.append({
                    'index': i,
                    'type': 'large_oi_small_price',
                    'oi_change': oi_changes.iloc[i],
                    'price_change': price_changes.iloc[i]
                })
        
        return events
    
    def _detect_block_trades(self, df: pd.DataFrame) -> float:
        """Detect probability of block trades"""
        
        if 'ce_volume' not in df.columns:
            return 0.0
        
        # Calculate volume distribution
        volume_mean = df['ce_volume'].mean()
        volume_std = df['ce_volume'].std()
        
        if volume_std == 0:
            return 0.0
        
        # Block trades are typically > 2 standard deviations
        block_threshold = volume_mean + 2 * volume_std
        block_trades = (df['ce_volume'] > block_threshold).sum()
        
        # Also check for volume spikes with OI changes
        block_probability = block_trades / len(df)
        
        # Adjust for OI impact
        if 'ce_oi' in df.columns:
            # Block trades should impact OI
            high_volume_mask = df['ce_volume'] > block_threshold
            if high_volume_mask.any():
                oi_impact = df.loc[high_volume_mask, 'ce_oi'].diff().abs().mean()
                avg_oi_change = df['ce_oi'].diff().abs().mean()
                
                if avg_oi_change > 0:
                    impact_ratio = oi_impact / avg_oi_change
                    block_probability *= min(impact_ratio, 2.0)
        
        return np.clip(block_probability * 5, 0.0, 1.0)  # Scale up for probability
    
    def _detect_stealth_accumulation(self, df: pd.DataFrame) -> float:
        """Detect stealth accumulation patterns"""
        
        stealth_score = 0.0
        
        if 'ce_oi' not in df.columns:
            return 0.0
        
        # Consistent small OI increases
        oi_changes = df['ce_oi'].diff()
        
        # Check for consistent positive changes
        positive_changes = (oi_changes > 0).sum()
        consistency_ratio = positive_changes / len(oi_changes)
        
        if consistency_ratio > 0.7:  # 70% positive changes
            stealth_score += 0.4
        
        # Low volatility in changes (systematic accumulation)
        if oi_changes.std() > 0:
            change_consistency = 1.0 - (oi_changes.std() / (abs(oi_changes.mean()) + 1))
            if change_consistency > 0.5:
                stealth_score += 0.3
        
        # Low volume during accumulation
        if 'ce_volume' in df.columns:
            volume_percentile = df['ce_volume'].quantile(0.4)
            low_volume_accumulation = ((df['ce_volume'] < volume_percentile) & (oi_changes > 0)).sum()
            
            if low_volume_accumulation > len(df) * 0.3:
                stealth_score += 0.3
        
        return min(stealth_score, 1.0)
    
    def _calculate_distribution_intensity(self, df: pd.DataFrame) -> float:
        """Calculate distribution intensity"""
        
        if 'ce_oi' not in df.columns or 'ce_volume' not in df.columns:
            return 0.0
        
        # Distribution: High volume with OI decrease
        oi_changes = df['ce_oi'].diff()
        
        # High volume events
        volume_threshold = df['ce_volume'].quantile(0.7)
        high_volume_mask = df['ce_volume'] > volume_threshold
        
        # OI decreases during high volume
        distribution_events = (high_volume_mask & (oi_changes < 0)).sum()
        
        intensity = distribution_events / len(df)
        
        # Check magnitude of distribution
        if distribution_events > 0:
            avg_distribution = abs(oi_changes[high_volume_mask & (oi_changes < 0)].mean())
            avg_oi_change = abs(oi_changes.mean())
            
            if avg_oi_change > 0:
                magnitude_factor = avg_distribution / avg_oi_change
                intensity *= min(magnitude_factor, 2.0)
        
        return np.clip(intensity * 5, 0.0, 1.0)
    
    def _detect_stop_hunting(self, df: pd.DataFrame) -> float:
        """Detect stop hunting patterns"""
        
        stop_hunting_score = 0.0
        
        if 'spot' not in df.columns:
            return 0.0
        
        # Look for price spikes followed by reversals
        price_changes = df['spot'].pct_change()
        
        for i in range(2, len(price_changes) - 1):
            # Spike detection
            if abs(price_changes.iloc[i]) > price_changes.std() * 2:
                # Check for reversal
                if np.sign(price_changes.iloc[i]) != np.sign(price_changes.iloc[i+1]):
                    reversal_magnitude = abs(price_changes.iloc[i+1])
                    
                    # Strong reversal indicates stop hunting
                    if reversal_magnitude > price_changes.std():
                        stop_hunting_score += 0.2
                        
                        # Check OI/volume patterns
                        if 'ce_volume' in df.columns:
                            # High volume during spike
                            if df['ce_volume'].iloc[i] > df['ce_volume'].quantile(0.8):
                                stop_hunting_score += 0.1
        
        return min(stop_hunting_score, 1.0)
    
    def _calculate_squeeze_potential(self, df: pd.DataFrame) -> float:
        """Calculate potential for short/long squeeze"""
        
        squeeze_score = 0.0
        
        if 'ce_oi' not in df.columns or 'pe_oi' not in df.columns:
            return 0.0
        
        # High OI concentration
        total_oi = df['ce_oi'] + df['pe_oi']
        oi_concentration = total_oi.std() / (total_oi.mean() + 1)
        
        if oi_concentration > 0.5:
            squeeze_score += 0.3
        
        # Imbalanced CE/PE ratio
        ce_total = df['ce_oi'].sum()
        pe_total = df['pe_oi'].sum()
        
        if pe_total > 0:
            ratio = ce_total / pe_total
            # Extreme ratios indicate squeeze potential
            if ratio > 2.0 or ratio < 0.5:
                squeeze_score += 0.3
        
        # Price pressure analysis
        if 'spot' in df.columns:
            # Consistent price movement with high OI
            price_trend = df['spot'].pct_change().mean()
            if abs(price_trend) > 0.001 and oi_concentration > 0.5:
                squeeze_score += 0.4
        
        return min(squeeze_score, 1.0)
    
    def _classify_absorption_type(self, absorption_score: float, stealth_score: float,
                                 distribution_intensity: float, 
                                 stop_hunting_prob: float) -> AbsorptionType:
        """Classify the type of liquidity absorption"""
        
        if absorption_score < self.sensitivity_threshold:
            return AbsorptionType.NO_ABSORPTION
        
        # Determine dominant pattern
        patterns = {
            'stealth': stealth_score,
            'distribution': distribution_intensity,
            'stop_hunting': stop_hunting_prob
        }
        
        dominant = max(patterns, key=patterns.get)
        
        if dominant == 'stealth' and stealth_score > 0.5:
            return AbsorptionType.STEALTH_POSITIONING
        elif dominant == 'distribution' and distribution_intensity > 0.5:
            return AbsorptionType.INSTITUTIONAL_DISTRIBUTION
        elif dominant == 'stop_hunting' and stop_hunting_prob > 0.5:
            return AbsorptionType.STOP_HUNTING
        elif absorption_score > 0.7:
            # High absorption without specific pattern
            if stealth_score > distribution_intensity:
                return AbsorptionType.INSTITUTIONAL_ACCUMULATION
            else:
                return AbsorptionType.BLOCK_ABSORPTION
        else:
            # Check for squeeze setup
            if self._calculate_squeeze_potential({'ce_oi': [1], 'pe_oi': [1]}) > 0.6:
                return AbsorptionType.SQUEEZE_SETUP
        
        return AbsorptionType.INSTITUTIONAL_ACCUMULATION
    
    def _calculate_absorption_velocity(self, df: pd.DataFrame) -> float:
        """Calculate velocity of liquidity absorption"""
        
        if 'ce_oi' not in df.columns:
            return 0.0
        
        # Rate of OI change relative to normal
        oi_changes = df['ce_oi'].diff().abs()
        
        # Recent vs historical velocity
        if len(oi_changes) > 20:
            recent_velocity = oi_changes.iloc[-10:].mean()
            historical_velocity = oi_changes.iloc[:-10].mean()
            
            if historical_velocity > 0:
                velocity_ratio = recent_velocity / historical_velocity
                return np.clip(velocity_ratio - 1.0, -1.0, 1.0)
        
        return 0.0
    
    def _calculate_absorption_persistence(self, current_score: float) -> float:
        """Calculate persistence of absorption patterns"""
        
        self.absorption_history.append(current_score)
        
        if len(self.absorption_history) < 3:
            return 0.0
        
        # Keep recent history
        if len(self.absorption_history) > 20:
            self.absorption_history = self.absorption_history[-20:]
        
        # Check consistency
        recent_scores = self.absorption_history[-5:]
        above_threshold = sum(1 for s in recent_scores if s > self.sensitivity_threshold)
        
        persistence = above_threshold / len(recent_scores)
        
        # Also check trend
        if len(self.absorption_history) > 5:
            trend = np.polyfit(range(5), recent_scores, 1)[0]
            if trend > 0:
                persistence = min(persistence + 0.2, 1.0)
        
        return persistence
    
    def _calculate_institutional_footprint(self, df: pd.DataFrame) -> float:
        """Calculate institutional footprint in the data"""
        
        footprint_score = 0.0
        
        # Large trades
        if 'ce_volume' in df.columns:
            volume_percentile_95 = df['ce_volume'].quantile(0.95)
            large_trades = (df['ce_volume'] > volume_percentile_95).sum()
            footprint_score += (large_trades / len(df)) * 2
        
        # Systematic patterns
        if 'ce_oi' in df.columns:
            oi_changes = df['ce_oi'].diff()
            # Low volatility in changes indicates systematic trading
            if oi_changes.std() > 0 and oi_changes.mean() != 0:
                systematic_score = 1.0 - (oi_changes.std() / abs(oi_changes.mean()))
                footprint_score += systematic_score * 0.3
        
        # Time-based patterns (institutional trading during specific times)
        if 'trade_time' in df.columns:
            # This would check for concentration during institutional hours
            # Simplified version here
            footprint_score += 0.2
        
        return np.clip(footprint_score, 0.0, 1.0)
    
    def _detect_market_maker_activity(self, df: pd.DataFrame) -> float:
        """Detect market maker activity patterns"""
        
        mm_score = 0.0
        
        # Balanced two-sided flow
        if 'ce_oi' in df.columns and 'pe_oi' in df.columns:
            ce_changes = df['ce_oi'].diff()
            pe_changes = df['pe_oi'].diff()
            
            # Market makers maintain balanced books
            correlation = ce_changes.corr(pe_changes)
            if correlation > 0.5:
                mm_score += 0.4
            
            # Consistent presence (low variance in activity)
            ce_variance = ce_changes.var()
            pe_variance = pe_changes.var()
            
            if ce_variance > 0 and pe_variance > 0:
                consistency = 1.0 / (1.0 + np.log(ce_variance + pe_variance))
                mm_score += consistency * 0.3
        
        # High frequency small trades
        if 'ce_volume' in df.columns:
            volume_median = df['ce_volume'].median()
            small_trades = (df['ce_volume'] < volume_median * 0.5).sum()
            
            if small_trades > len(df) * 0.3:
                mm_score += 0.3
        
        return min(mm_score, 1.0)
    
    def _detect_liquidity_vacuum(self, df: pd.DataFrame) -> float:
        """Detect liquidity vacuum conditions"""
        
        vacuum_score = 0.0
        
        # Low volume with high price volatility
        if 'ce_volume' in df.columns and 'spot' in df.columns:
            volume_percentile_20 = df['ce_volume'].quantile(0.2)
            low_volume_mask = df['ce_volume'] < volume_percentile_20
            
            if low_volume_mask.any():
                price_volatility = df.loc[low_volume_mask, 'spot'].pct_change().std()
                normal_volatility = df['spot'].pct_change().std()
                
                if normal_volatility > 0:
                    volatility_ratio = price_volatility / normal_volatility
                    if volatility_ratio > 1.5:
                        vacuum_score = min((volatility_ratio - 1.0) / 2, 1.0)
        
        return vacuum_score
    
    def _analyze_price_suppression(self, df: pd.DataFrame) -> float:
        """Analyze price suppression patterns"""
        
        suppression_score = 0.0
        
        if 'spot' not in df.columns or 'ce_oi' not in df.columns:
            return 0.0
        
        # Large OI increases with limited price movement
        oi_changes = df['ce_oi'].pct_change()
        price_changes = df['spot'].pct_change()
        
        # Find periods of OI buildup
        oi_buildup_mask = oi_changes > oi_changes.quantile(0.7)
        
        if oi_buildup_mask.any():
            # Check price movement during buildup
            price_movement = abs(price_changes[oi_buildup_mask]).mean()
            normal_movement = abs(price_changes).mean()
            
            if normal_movement > 0:
                suppression_ratio = 1.0 - (price_movement / normal_movement)
                suppression_score = max(suppression_ratio, 0.0)
        
        return suppression_score
    
    def _calculate_volume_absorption_ratio(self, df: pd.DataFrame) -> float:
        """Calculate ratio of volume absorbed vs traded"""
        
        if 'ce_volume' not in df.columns or 'ce_oi' not in df.columns:
            return 0.0
        
        # Volume that resulted in OI changes
        oi_changes = df['ce_oi'].diff().abs()
        volume = df['ce_volume']
        
        # Absorption ratio: OI change relative to volume
        total_oi_change = oi_changes.sum()
        total_volume = volume.sum()
        
        if total_volume > 0:
            absorption_ratio = total_oi_change / total_volume
            return min(absorption_ratio * 2, 1.0)  # Scale and cap
        
        return 0.0
    
    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract liquidity absorption features for Component 3
        
        Args:
            df: Production data
            
        Returns:
            Array of absorption features
        """
        metrics = self.detect_absorption_patterns(df)
        
        features = [
            metrics.absorption_score,
            metrics.block_trade_probability,
            metrics.stealth_accumulation_score,
            metrics.distribution_intensity,
            metrics.stop_hunting_probability,
            metrics.squeeze_potential,
            metrics.large_oi_small_price_events / 100,  # Normalized
            metrics.absorption_velocity,
            metrics.absorption_persistence,
            metrics.institutional_footprint,
            metrics.market_maker_activity,
            metrics.liquidity_vacuum_score,
            metrics.price_suppression_score,
            metrics.volume_absorption_ratio
        ]
        
        return np.array(features)