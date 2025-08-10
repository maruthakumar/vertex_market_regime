"""
Option Data Manager for Rolling ATM Tracking
===========================================

Manages option data for technical indicators that need consistent ATM tracking
and option-based price calculations.

Author: Market Regime Refactoring Team
Date: 2025-07-06
Version: 2.0.0 - Refactored Architecture
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass
class ATMData:
    """ATM option data structure"""
    timestamp: datetime
    spot_price: float
    atm_strike: float
    atm_ce_price: float
    atm_pe_price: float
    atm_straddle_price: float
    dte: int
    distance_from_spot: float
    confidence: float = 1.0

@dataclass
class OptionPriceData:
    """Option price data for technical analysis"""
    timestamp: datetime
    ce_price: float
    pe_price: float
    straddle_price: float
    underlying_price: float
    dte: int
    strike: float
    data_quality: float = 1.0

class RollingATMTracker:
    """
    Tracks rolling ATM option data for technical indicators
    
    Maintains a consistent view of ATM option prices over time,
    essential for option-based technical indicators like RSI, MACD, etc.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ATM tracker"""
        self.config = config or {}
        
        # Configuration parameters
        self.max_distance_threshold = self.config.get('max_distance_threshold', 0.02)  # 2%
        self.history_length = self.config.get('history_length', 1000)
        self.price_smoothing = self.config.get('price_smoothing', True)
        self.smoothing_alpha = self.config.get('smoothing_alpha', 0.1)
        
        # Internal data storage
        self.atm_history: List[ATMData] = []
        self.price_history: List[OptionPriceData] = []
        self.current_atm: Optional[ATMData] = None
        
        # Price smoothing state
        self.smoothed_prices = {
            'ce': None,
            'pe': None,
            'straddle': None
        }
        
        logger.info("RollingATMTracker initialized")
    
    def update(self, 
               market_data: pd.DataFrame, 
               spot_price: float,
               timestamp: Optional[datetime] = None) -> Optional[ATMData]:
        """
        Update ATM tracking with new market data
        
        Args:
            market_data: Market data DataFrame
            spot_price: Current spot price
            timestamp: Data timestamp (default: now)
            
        Returns:
            ATMData: Updated ATM data or None if failed
        """
        try:
            timestamp = timestamp or datetime.now()
            
            # Find ATM strike
            atm_data = self._find_atm_strike(market_data, spot_price, timestamp)
            
            if atm_data:
                # Apply price smoothing if enabled
                if self.price_smoothing:
                    atm_data = self._apply_price_smoothing(atm_data)
                
                # Update history
                self.atm_history.append(atm_data)
                self.current_atm = atm_data
                
                # Create price data for technical indicators
                price_data = OptionPriceData(
                    timestamp=timestamp,
                    ce_price=atm_data.atm_ce_price,
                    pe_price=atm_data.atm_pe_price,
                    straddle_price=atm_data.atm_straddle_price,
                    underlying_price=spot_price,
                    dte=atm_data.dte,
                    strike=atm_data.atm_strike,
                    data_quality=atm_data.confidence
                )
                
                self.price_history.append(price_data)
                
                # Maintain history length
                if len(self.atm_history) > self.history_length:
                    self.atm_history = self.atm_history[-self.history_length:]
                
                if len(self.price_history) > self.history_length:
                    self.price_history = self.price_history[-self.history_length:]
                
                logger.debug(f"ATM updated: strike={atm_data.atm_strike}, straddle={atm_data.atm_straddle_price:.2f}")
                
                return atm_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error updating ATM tracker: {e}")
            return None
    
    def get_price_series(self, 
                        price_type: str = 'straddle',
                        lookback: Optional[int] = None) -> pd.Series:
        """
        Get price series for technical indicators
        
        Args:
            price_type: Type of price ('ce', 'pe', 'straddle', 'underlying')
            lookback: Number of periods to look back
            
        Returns:
            pd.Series: Price series with timestamps
        """
        try:
            if not self.price_history:
                return pd.Series(dtype=float)
            
            # Get data
            data = self.price_history[-lookback:] if lookback else self.price_history
            
            # Extract prices and timestamps
            timestamps = [d.timestamp for d in data]
            
            if price_type == 'ce':
                prices = [d.ce_price for d in data]
            elif price_type == 'pe':
                prices = [d.pe_price for d in data]
            elif price_type == 'straddle':
                prices = [d.straddle_price for d in data]
            elif price_type == 'underlying':
                prices = [d.underlying_price for d in data]
            else:
                logger.error(f"Unknown price type: {price_type}")
                return pd.Series(dtype=float)
            
            return pd.Series(prices, index=timestamps)
            
        except Exception as e:
            logger.error(f"Error getting price series: {e}")
            return pd.Series(dtype=float)
    
    def get_current_atm(self) -> Optional[ATMData]:
        """Get current ATM data"""
        return self.current_atm
    
    def get_atm_history(self, lookback: Optional[int] = None) -> List[ATMData]:
        """Get ATM history"""
        if lookback:
            return self.atm_history[-lookback:]
        return self.atm_history.copy()
    
    def _find_atm_strike(self, 
                        market_data: pd.DataFrame, 
                        spot_price: float,
                        timestamp: datetime) -> Optional[ATMData]:
        """Find the best ATM strike from market data"""
        try:
            if 'strike' not in market_data.columns:
                logger.error("Strike column not found in market data")
                return None
            
            # Get unique strikes
            strikes = market_data['strike'].unique()
            
            # Find closest strike to spot
            closest_strike = min(strikes, key=lambda x: abs(x - spot_price))
            distance = abs(closest_strike - spot_price) / spot_price
            
            # Check if distance is within threshold
            if distance > self.max_distance_threshold:
                logger.warning(f"ATM strike distance too large: {distance:.1%} > {self.max_distance_threshold:.1%}")
                confidence = max(0.1, 1.0 - distance * 5)  # Reduced confidence
            else:
                confidence = 1.0
            
            # Get option data for this strike
            strike_data = market_data[market_data['strike'] == closest_strike]
            
            # Get CE and PE prices
            ce_data = strike_data[strike_data['option_type'] == 'CE']
            pe_data = strike_data[strike_data['option_type'] == 'PE']
            
            if ce_data.empty or pe_data.empty:
                logger.warning(f"Missing CE or PE data for strike {closest_strike}")
                return None
            
            # Extract prices (prefer LTP, fall back to close)
            ce_price = self._get_option_price(ce_data.iloc[0])
            pe_price = self._get_option_price(pe_data.iloc[0])
            
            if ce_price <= 0 or pe_price <= 0:
                logger.warning(f"Invalid option prices: CE={ce_price}, PE={pe_price}")
                return None
            
            # Calculate straddle price
            straddle_price = ce_price + pe_price
            
            # Get DTE
            dte = ce_data.iloc[0].get('dte', 30) if 'dte' in ce_data.columns else 30
            
            return ATMData(
                timestamp=timestamp,
                spot_price=spot_price,
                atm_strike=closest_strike,
                atm_ce_price=ce_price,
                atm_pe_price=pe_price,
                atm_straddle_price=straddle_price,
                dte=dte,
                distance_from_spot=distance,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error finding ATM strike: {e}")
            return None
    
    def _get_option_price(self, option_row: pd.Series) -> float:
        """Extract option price from data row"""
        # Priority: LTP > close > (bid+ask)/2 > 0
        
        if 'ltp' in option_row and pd.notna(option_row['ltp']) and option_row['ltp'] > 0:
            return float(option_row['ltp'])
        
        if 'close' in option_row and pd.notna(option_row['close']) and option_row['close'] > 0:
            return float(option_row['close'])
        
        if 'bid' in option_row and 'ask' in option_row:
            bid = option_row.get('bid', 0)
            ask = option_row.get('ask', 0)
            if bid > 0 and ask > 0:
                return (bid + ask) / 2
        
        return 0.0
    
    def _apply_price_smoothing(self, atm_data: ATMData) -> ATMData:
        """Apply exponential smoothing to prices"""
        try:
            alpha = self.smoothing_alpha
            
            # Initialize smoothed prices if first time
            if self.smoothed_prices['ce'] is None:
                self.smoothed_prices['ce'] = atm_data.atm_ce_price
                self.smoothed_prices['pe'] = atm_data.atm_pe_price
                self.smoothed_prices['straddle'] = atm_data.atm_straddle_price
                return atm_data
            
            # Apply exponential smoothing
            smoothed_ce = alpha * atm_data.atm_ce_price + (1 - alpha) * self.smoothed_prices['ce']
            smoothed_pe = alpha * atm_data.atm_pe_price + (1 - alpha) * self.smoothed_prices['pe']
            smoothed_straddle = smoothed_ce + smoothed_pe
            
            # Update smoothed state
            self.smoothed_prices['ce'] = smoothed_ce
            self.smoothed_prices['pe'] = smoothed_pe
            self.smoothed_prices['straddle'] = smoothed_straddle
            
            # Return smoothed data
            return ATMData(
                timestamp=atm_data.timestamp,
                spot_price=atm_data.spot_price,
                atm_strike=atm_data.atm_strike,
                atm_ce_price=smoothed_ce,
                atm_pe_price=smoothed_pe,
                atm_straddle_price=smoothed_straddle,
                dte=atm_data.dte,
                distance_from_spot=atm_data.distance_from_spot,
                confidence=atm_data.confidence
            )
            
        except Exception as e:
            logger.error(f"Error applying price smoothing: {e}")
            return atm_data

class OptionDataManager:
    """
    Comprehensive option data manager
    
    Manages multiple ATM trackers, handles data validation,
    and provides unified interface for option-based indicators
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize option data manager"""
        self.config = config or {}
        
        # Multiple ATM trackers for different timeframes/purposes
        self.atm_trackers: Dict[str, RollingATMTracker] = {}
        
        # Default tracker
        self.default_tracker = RollingATMTracker(self.config.get('default_tracker', {}))
        self.atm_trackers['default'] = self.default_tracker
        
        # Data validation parameters
        self.min_data_quality = self.config.get('min_data_quality', 0.5)
        self.max_price_change = self.config.get('max_price_change', 0.5)  # 50% max change
        
        logger.info("OptionDataManager initialized")
    
    def add_tracker(self, name: str, config: Optional[Dict[str, Any]] = None) -> RollingATMTracker:
        """Add new ATM tracker"""
        tracker = RollingATMTracker(config)
        self.atm_trackers[name] = tracker
        logger.info(f"Added ATM tracker: {name}")
        return tracker
    
    def update_all_trackers(self, 
                           market_data: pd.DataFrame, 
                           spot_price: float,
                           timestamp: Optional[datetime] = None) -> Dict[str, Optional[ATMData]]:
        """Update all ATM trackers"""
        results = {}
        
        for name, tracker in self.atm_trackers.items():
            try:
                result = tracker.update(market_data, spot_price, timestamp)
                results[name] = result
            except Exception as e:
                logger.error(f"Error updating tracker {name}: {e}")
                results[name] = None
        
        return results
    
    def get_price_series_for_indicator(self, 
                                     indicator_name: str,
                                     price_type: str = 'straddle',
                                     lookback: Optional[int] = None) -> pd.Series:
        """Get price series for specific indicator"""
        # Use indicator-specific tracker if available, otherwise default
        tracker_name = self.config.get('indicator_trackers', {}).get(indicator_name, 'default')
        
        if tracker_name in self.atm_trackers:
            return self.atm_trackers[tracker_name].get_price_series(price_type, lookback)
        else:
            logger.warning(f"Tracker {tracker_name} not found, using default")
            return self.default_tracker.get_price_series(price_type, lookback)
    
    def validate_option_data(self, market_data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate option market data"""
        errors = []
        
        # Check required columns
        required_columns = ['strike', 'option_type']
        missing_columns = [col for col in required_columns if col not in market_data.columns]
        if missing_columns:
            errors.append(f"Missing columns: {missing_columns}")
        
        # Check data types
        if 'strike' in market_data.columns:
            if not pd.api.types.is_numeric_dtype(market_data['strike']):
                errors.append("Strike column must be numeric")
        
        # Check option types
        if 'option_type' in market_data.columns:
            valid_types = {'CE', 'PE'}
            invalid_types = set(market_data['option_type'].unique()) - valid_types
            if invalid_types:
                errors.append(f"Invalid option types: {invalid_types}")
        
        # Check for minimum data
        if len(market_data) < 2:
            errors.append("Insufficient option data (need at least 2 rows)")
        
        return len(errors) == 0, errors
    
    def get_data_quality_metrics(self) -> Dict[str, Any]:
        """Get data quality metrics for all trackers"""
        metrics = {}
        
        for name, tracker in self.atm_trackers.items():
            current_atm = tracker.get_current_atm()
            
            if current_atm:
                metrics[name] = {
                    'confidence': current_atm.confidence,
                    'distance_from_spot': current_atm.distance_from_spot,
                    'history_length': len(tracker.atm_history),
                    'data_age_seconds': (datetime.now() - current_atm.timestamp).total_seconds(),
                    'health': 'good' if current_atm.confidence > 0.8 else 'poor'
                }
            else:
                metrics[name] = {
                    'confidence': 0.0,
                    'health': 'no_data'
                }
        
        return metrics