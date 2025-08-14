"""
Underlying Price Level Detection Engine
Traditional support/resistance detection from underlying asset prices
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class UnderlyingLevelDetector:
    """
    Detects traditional support/resistance levels from underlying price data
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize underlying level detector
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Detection parameters
        self.min_touches = config.get("min_touches", 2)
        self.touch_tolerance = config.get("touch_tolerance", 0.002)  # 0.2%
        self.lookback_periods = config.get("lookback_periods", 252)
        
        # Moving average periods
        self.ma_periods = [20, 50, 100, 200]
        
        # Fibonacci levels
        self.fib_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        
        # Volume profile parameters
        self.volume_bins = config.get("volume_bins", 50)
        self.volume_percentile = config.get("volume_percentile", 80)
        
        logger.info("Initialized UnderlyingLevelDetector")
    
    def detect_all_levels(
        self,
        market_data: pd.DataFrame,
        timeframe: str = "daily"
    ) -> List[Dict[str, Any]]:
        """
        Detect all traditional S&R levels from underlying price
        
        Args:
            market_data: DataFrame with OHLCV data
            timeframe: Timeframe for analysis
            
        Returns:
            List of detected levels
        """
        levels = []
        
        # Daily timeframe levels
        daily_levels = self.detect_daily_levels(market_data)
        levels.extend(daily_levels)
        
        # Weekly timeframe levels
        weekly_levels = self.detect_weekly_levels(market_data)
        levels.extend(weekly_levels)
        
        # Monthly timeframe levels
        monthly_levels = self.detect_monthly_levels(market_data)
        levels.extend(monthly_levels)
        
        # Psychological levels
        psych_levels = self.detect_psychological_levels(market_data)
        levels.extend(psych_levels)
        
        return levels
    
    def detect_daily_levels(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect daily timeframe levels (pivots, gaps, volume profile)
        
        Returns:
            List of daily levels
        """
        levels = []
        
        # Daily pivot points
        pivot_levels = self._calculate_daily_pivots(market_data)
        levels.extend(pivot_levels)
        
        # Gap levels
        gap_levels = self._detect_gaps(market_data)
        levels.extend(gap_levels)
        
        # Volume profile levels
        volume_levels = self._calculate_volume_profile(market_data, "daily")
        levels.extend(volume_levels)
        
        # Previous day high/low/close
        prev_levels = self._get_previous_day_levels(market_data)
        levels.extend(prev_levels)
        
        return levels
    
    def detect_weekly_levels(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect weekly timeframe levels (pivots, MAs)
        
        Returns:
            List of weekly levels
        """
        levels = []
        
        # Weekly pivot points
        weekly_pivots = self._calculate_weekly_pivots(market_data)
        levels.extend(weekly_pivots)
        
        # Moving averages
        ma_levels = self._calculate_moving_averages(market_data)
        levels.extend(ma_levels)
        
        # Weekly high/low
        weekly_extremes = self._get_weekly_extremes(market_data)
        levels.extend(weekly_extremes)
        
        return levels
    
    def detect_monthly_levels(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect monthly timeframe levels (pivots, Fibonacci)
        
        Returns:
            List of monthly levels
        """
        levels = []
        
        # Monthly pivot points
        monthly_pivots = self._calculate_monthly_pivots(market_data)
        levels.extend(monthly_pivots)
        
        # Fibonacci retracement levels
        fib_levels = self._calculate_fibonacci_levels(market_data)
        levels.extend(fib_levels)
        
        # Monthly high/low
        monthly_extremes = self._get_monthly_extremes(market_data)
        levels.extend(monthly_extremes)
        
        return levels
    
    def detect_psychological_levels(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect psychological and round number levels
        
        Returns:
            List of psychological levels
        """
        levels = []
        
        if len(market_data) > 0:
            current_price = market_data["close"].iloc[-1]
            
            # Determine round number interval based on price magnitude
            if current_price > 50000:
                intervals = [1000, 500, 250]
            elif current_price > 10000:
                intervals = [500, 250, 100]
            elif current_price > 1000:
                intervals = [100, 50, 25]
            else:
                intervals = [50, 25, 10]
            
            for interval in intervals:
                # Find nearby round numbers
                base = int(current_price / interval) * interval
                
                for i in range(-5, 6):
                    level_price = base + (i * interval)
                    
                    if level_price > 0:
                        distance = abs(level_price - current_price) / current_price
                        
                        # Only include levels within 10% of current price
                        if distance < 0.1:
                            level_type = "support" if level_price < current_price else "resistance"
                            
                            # Stronger levels at major round numbers
                            strength = 0.8 if level_price % (interval * 2) == 0 else 0.6
                            
                            levels.append({
                                "price": level_price,
                                "source": f"psychological_{interval}",
                                "type": level_type,
                                "strength": strength,
                                "method": "psychological",
                                "interval": interval
                            })
        
        return levels
    
    def _calculate_daily_pivots(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Calculate daily pivot points
        
        Returns:
            List of daily pivot levels
        """
        levels = []
        
        if len(market_data) > 0:
            # Get previous day's data
            high = market_data["high"].iloc[-1]
            low = market_data["low"].iloc[-1]
            close = market_data["close"].iloc[-1]
            
            # Standard pivot calculation
            pivot = (high + low + close) / 3
            
            # Resistance levels
            r1 = 2 * pivot - low
            r2 = pivot + (high - low)
            r3 = high + 2 * (pivot - low)
            
            # Support levels
            s1 = 2 * pivot - high
            s2 = pivot - (high - low)
            s3 = low - 2 * (high - pivot)
            
            # Camarilla pivots
            h4 = close + (high - low) * 1.1 / 2
            h3 = close + (high - low) * 1.1 / 4
            l3 = close - (high - low) * 1.1 / 4
            l4 = close - (high - low) * 1.1 / 2
            
            pivot_points = [
                (r3, "R3", "resistance", 0.6),
                (r2, "R2", "resistance", 0.8),
                (r1, "R1", "resistance", 1.0),
                (h4, "H4", "resistance", 0.7),
                (h3, "H3", "resistance", 0.9),
                (pivot, "PP", "neutral", 1.0),
                (l3, "L3", "support", 0.9),
                (l4, "L4", "support", 0.7),
                (s1, "S1", "support", 1.0),
                (s2, "S2", "support", 0.8),
                (s3, "S3", "support", 0.6)
            ]
            
            for price, label, level_type, strength in pivot_points:
                levels.append({
                    "price": price,
                    "source": f"daily_{label}",
                    "type": level_type,
                    "strength": strength,
                    "method": "daily_pivots",
                    "timeframe": "daily"
                })
        
        return levels
    
    def _detect_gaps(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect gap levels in price data
        
        Returns:
            List of gap levels
        """
        levels = []
        
        if len(market_data) > 1:
            # Calculate gaps
            gaps = market_data["open"].values[1:] - market_data["close"].values[:-1]
            gap_indices = np.where(np.abs(gaps) > market_data["close"].values[:-1] * 0.002)[0]
            
            for idx in gap_indices[-10:]:  # Last 10 gaps
                gap_size = gaps[idx]
                
                if gap_size > 0:  # Gap up
                    # Gap becomes support
                    levels.append({
                        "price": market_data["close"].iloc[idx],
                        "source": "gap_up",
                        "type": "support",
                        "strength": min(1.0, abs(gap_size) / market_data["close"].iloc[idx] * 100),
                        "method": "daily_gaps",
                        "timeframe": "daily"
                    })
                else:  # Gap down
                    # Gap becomes resistance
                    levels.append({
                        "price": market_data["open"].iloc[idx + 1],
                        "source": "gap_down",
                        "type": "resistance",
                        "strength": min(1.0, abs(gap_size) / market_data["close"].iloc[idx] * 100),
                        "method": "daily_gaps",
                        "timeframe": "daily"
                    })
        
        return levels
    
    def _calculate_volume_profile(
        self,
        market_data: pd.DataFrame,
        timeframe: str
    ) -> List[Dict[str, Any]]:
        """
        Calculate volume profile levels
        
        Returns:
            List of volume-based levels
        """
        levels = []
        
        if "volume" in market_data.columns and len(market_data) > 20:
            # Create price-volume histogram
            prices = market_data["close"].values
            volumes = market_data["volume"].values
            
            # Calculate price bins
            price_min, price_max = prices.min(), prices.max()
            bins = np.linspace(price_min, price_max, self.volume_bins)
            
            # Calculate volume at each price level
            volume_profile = np.zeros(len(bins) - 1)
            
            for i in range(len(bins) - 1):
                mask = (prices >= bins[i]) & (prices < bins[i + 1])
                volume_profile[i] = volumes[mask].sum()
            
            # Find high volume nodes (POC - Point of Control)
            if volume_profile.sum() > 0:
                volume_threshold = np.percentile(volume_profile, self.volume_percentile)
                high_volume_indices = np.where(volume_profile > volume_threshold)[0]
                
                for idx in high_volume_indices:
                    price_level = (bins[idx] + bins[idx + 1]) / 2
                    
                    # High volume nodes act as magnets
                    levels.append({
                        "price": price_level,
                        "source": f"volume_node_{timeframe}",
                        "type": "neutral",
                        "strength": volume_profile[idx] / volume_profile.max(),
                        "method": "volume_profile",
                        "timeframe": timeframe,
                        "volume": volume_profile[idx]
                    })
                
                # Value Area High/Low (70% of volume)
                cumsum = np.cumsum(volume_profile)
                total_volume = cumsum[-1]
                
                if total_volume > 0:
                    va_start = np.where(cumsum >= total_volume * 0.15)[0][0] if len(np.where(cumsum >= total_volume * 0.15)[0]) > 0 else 0
                    va_end = np.where(cumsum >= total_volume * 0.85)[0][0] if len(np.where(cumsum >= total_volume * 0.85)[0]) > 0 else len(bins) - 2
                    
                    # Value Area Low (Support)
                    levels.append({
                        "price": (bins[va_start] + bins[va_start + 1]) / 2,
                        "source": f"VAL_{timeframe}",
                        "type": "support",
                        "strength": 0.8,
                        "method": "volume_profile",
                        "timeframe": timeframe
                    })
                    
                    # Value Area High (Resistance)
                    levels.append({
                        "price": (bins[va_end] + bins[va_end + 1]) / 2,
                        "source": f"VAH_{timeframe}",
                        "type": "resistance",
                        "strength": 0.8,
                        "method": "volume_profile",
                        "timeframe": timeframe
                    })
        
        return levels
    
    def _get_previous_day_levels(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Get previous day's high, low, close levels
        
        Returns:
            List of previous day levels
        """
        levels = []
        
        if len(market_data) > 1:
            # Previous day high
            levels.append({
                "price": market_data["high"].iloc[-2],
                "source": "prev_day_high",
                "type": "resistance",
                "strength": 0.9,
                "method": "daily_pivots",
                "timeframe": "daily"
            })
            
            # Previous day low
            levels.append({
                "price": market_data["low"].iloc[-2],
                "source": "prev_day_low",
                "type": "support",
                "strength": 0.9,
                "method": "daily_pivots",
                "timeframe": "daily"
            })
            
            # Previous day close
            current_price = market_data["close"].iloc[-1]
            prev_close = market_data["close"].iloc[-2]
            
            levels.append({
                "price": prev_close,
                "source": "prev_day_close",
                "type": "support" if current_price > prev_close else "resistance",
                "strength": 0.8,
                "method": "daily_pivots",
                "timeframe": "daily"
            })
        
        return levels
    
    def _calculate_weekly_pivots(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Calculate weekly pivot points
        
        Returns:
            List of weekly pivot levels
        """
        levels = []
        
        if len(market_data) >= 5:  # At least one week of data
            # Get weekly high, low, close
            weekly_high = market_data["high"].iloc[-5:].max()
            weekly_low = market_data["low"].iloc[-5:].min()
            weekly_close = market_data["close"].iloc[-1]
            
            # Calculate weekly pivots
            pivot = (weekly_high + weekly_low + weekly_close) / 3
            
            # Resistance levels
            r1 = 2 * pivot - weekly_low
            r2 = pivot + (weekly_high - weekly_low)
            
            # Support levels
            s1 = 2 * pivot - weekly_high
            s2 = pivot - (weekly_high - weekly_low)
            
            weekly_pivots = [
                (r2, "W_R2", "resistance", 0.7),
                (r1, "W_R1", "resistance", 0.85),
                (pivot, "W_PP", "neutral", 0.9),
                (s1, "W_S1", "support", 0.85),
                (s2, "W_S2", "support", 0.7)
            ]
            
            for price, label, level_type, strength in weekly_pivots:
                levels.append({
                    "price": price,
                    "source": f"weekly_{label}",
                    "type": level_type,
                    "strength": strength,
                    "method": "weekly_pivots",
                    "timeframe": "weekly"
                })
        
        return levels
    
    def _calculate_moving_averages(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Calculate moving average levels
        
        Returns:
            List of MA-based levels
        """
        levels = []
        
        if len(market_data) > 0:
            current_price = market_data["close"].iloc[-1]
            
            for period in self.ma_periods:
                if len(market_data) >= period:
                    # Simple MA
                    sma = market_data["close"].rolling(period).mean().iloc[-1]
                    
                    # EMA
                    ema = market_data["close"].ewm(span=period, adjust=False).mean().iloc[-1]
                    
                    # SMA level
                    levels.append({
                        "price": sma,
                        "source": f"SMA_{period}",
                        "type": "support" if current_price > sma else "resistance",
                        "strength": 0.7 + (period / 1000),  # Longer MAs are stronger
                        "method": "moving_averages",
                        "timeframe": "dynamic"
                    })
                    
                    # EMA level
                    levels.append({
                        "price": ema,
                        "source": f"EMA_{period}",
                        "type": "support" if current_price > ema else "resistance",
                        "strength": 0.75 + (period / 1000),
                        "method": "moving_averages",
                        "timeframe": "dynamic"
                    })
        
        return levels
    
    def _get_weekly_extremes(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Get weekly high and low levels
        
        Returns:
            List of weekly extreme levels
        """
        levels = []
        
        if len(market_data) >= 5:
            weekly_high = market_data["high"].iloc[-5:].max()
            weekly_low = market_data["low"].iloc[-5:].min()
            
            levels.append({
                "price": weekly_high,
                "source": "weekly_high",
                "type": "resistance",
                "strength": 0.85,
                "method": "weekly_pivots",
                "timeframe": "weekly"
            })
            
            levels.append({
                "price": weekly_low,
                "source": "weekly_low",
                "type": "support",
                "strength": 0.85,
                "method": "weekly_pivots",
                "timeframe": "weekly"
            })
        
        return levels
    
    def _calculate_monthly_pivots(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Calculate monthly pivot points
        
        Returns:
            List of monthly pivot levels
        """
        levels = []
        
        if len(market_data) >= 20:  # At least one month of data
            # Get monthly high, low, close
            monthly_high = market_data["high"].iloc[-20:].max()
            monthly_low = market_data["low"].iloc[-20:].min()
            monthly_close = market_data["close"].iloc[-1]
            
            # Calculate monthly pivots
            pivot = (monthly_high + monthly_low + monthly_close) / 3
            
            # Resistance levels
            r1 = 2 * pivot - monthly_low
            r2 = pivot + (monthly_high - monthly_low)
            
            # Support levels
            s1 = 2 * pivot - monthly_high
            s2 = pivot - (monthly_high - monthly_low)
            
            monthly_pivots = [
                (r2, "M_R2", "resistance", 0.65),
                (r1, "M_R1", "resistance", 0.8),
                (pivot, "M_PP", "neutral", 0.85),
                (s1, "M_S1", "support", 0.8),
                (s2, "M_S2", "support", 0.65)
            ]
            
            for price, label, level_type, strength in monthly_pivots:
                levels.append({
                    "price": price,
                    "source": f"monthly_{label}",
                    "type": level_type,
                    "strength": strength,
                    "method": "monthly_pivots",
                    "timeframe": "monthly"
                })
        
        return levels
    
    def _calculate_fibonacci_levels(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Calculate Fibonacci retracement levels
        
        Returns:
            List of Fibonacci levels
        """
        levels = []
        
        if len(market_data) >= 50:  # Need sufficient data for swing detection
            # Find recent swing high and low
            lookback = min(100, len(market_data))
            recent_data = market_data.iloc[-lookback:]
            
            swing_high = recent_data["high"].max()
            swing_low = recent_data["low"].min()
            current_price = market_data["close"].iloc[-1]
            
            # Determine trend direction
            high_idx = recent_data["high"].idxmax()
            low_idx = recent_data["low"].idxmin()
            
            if high_idx > low_idx:  # Uptrend - retracement from high
                for fib_level in self.fib_levels:
                    price = swing_high - (swing_high - swing_low) * fib_level
                    
                    # Fib levels act as support in uptrend
                    levels.append({
                        "price": price,
                        "source": f"fib_{int(fib_level * 100)}",
                        "type": "support" if price < current_price else "resistance",
                        "strength": 0.7 if fib_level in [0.382, 0.5, 0.618] else 0.5,
                        "method": "fibonacci",
                        "timeframe": "swing"
                    })
            else:  # Downtrend - retracement from low
                for fib_level in self.fib_levels:
                    price = swing_low + (swing_high - swing_low) * fib_level
                    
                    # Fib levels act as resistance in downtrend
                    levels.append({
                        "price": price,
                        "source": f"fib_{int(fib_level * 100)}",
                        "type": "resistance" if price > current_price else "support",
                        "strength": 0.7 if fib_level in [0.382, 0.5, 0.618] else 0.5,
                        "method": "fibonacci",
                        "timeframe": "swing"
                    })
        
        return levels
    
    def _get_monthly_extremes(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Get monthly high and low levels
        
        Returns:
            List of monthly extreme levels
        """
        levels = []
        
        if len(market_data) >= 20:
            monthly_high = market_data["high"].iloc[-20:].max()
            monthly_low = market_data["low"].iloc[-20:].min()
            
            levels.append({
                "price": monthly_high,
                "source": "monthly_high",
                "type": "resistance",
                "strength": 0.8,
                "method": "monthly_pivots",
                "timeframe": "monthly"
            })
            
            levels.append({
                "price": monthly_low,
                "source": "monthly_low",
                "type": "support",
                "strength": 0.8,
                "method": "monthly_pivots",
                "timeframe": "monthly"
            })
        
        return levels
    
    def validate_historical_levels(
        self,
        levels: List[Dict[str, Any]],
        market_data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Validate levels against historical price action
        
        Args:
            levels: List of detected levels
            market_data: Historical market data
            
        Returns:
            Validated levels with touch counts
        """
        validated_levels = []
        
        for level in levels:
            price = level["price"]
            touches = 0
            bounces = 0
            
            # Count touches and bounces
            for i in range(len(market_data)):
                high = market_data["high"].iloc[i]
                low = market_data["low"].iloc[i]
                close = market_data["close"].iloc[i]
                
                # Check if price touched the level
                if (abs(high - price) / price < self.touch_tolerance or
                    abs(low - price) / price < self.touch_tolerance):
                    touches += 1
                    
                    # Check if price bounced from level
                    if i < len(market_data) - 1:
                        next_close = market_data["close"].iloc[i + 1]
                        
                        if level["type"] == "support" and close > price and next_close > close:
                            bounces += 1
                        elif level["type"] == "resistance" and close < price and next_close < close:
                            bounces += 1
            
            # Add validation metrics
            level["touches"] = touches
            level["bounces"] = bounces
            level["bounce_rate"] = bounces / max(touches, 1)
            
            # Only include levels with minimum touches
            if touches >= self.min_touches:
                validated_levels.append(level)
        
        return validated_levels