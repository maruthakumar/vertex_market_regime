"""
Strike Selection Base Classes
============================

Provides base classes and implementations for different strike selection
strategies used by market regime indicators.

Author: Market Regime Refactoring Team
Date: 2025-07-06
Version: 2.0.0 - Refactored Architecture
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class StrikeSelectionStrategy(Enum):
    """Strike selection strategy types"""
    FULL_CHAIN = "full_chain"
    DYNAMIC_RANGE = "dynamic_range"
    ROLLING_ATM = "rolling_atm"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    VOLUME_WEIGHTED = "volume_weighted"

@dataclass
class StrikeInfo:
    """Information about a selected strike"""
    strike: float
    distance_from_atm: float
    weight: float
    option_type: str  # 'CE' or 'PE'
    dte: int
    selection_reason: str
    confidence: float = 1.0
    
    def __post_init__(self):
        """Validate strike info after initialization"""
        if self.weight < 0:
            raise ValueError("Strike weight cannot be negative")
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError("Confidence must be between 0 and 1")

class BaseStrikeSelector(ABC):
    """
    Base class for strike selection strategies
    
    Provides common interface for selecting strikes based on different criteria
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize strike selector"""
        self.config = config or {}
        self.name = self.__class__.__name__
        
        # Common configuration parameters
        self.max_strikes = self.config.get('max_strikes', 20)
        self.min_strikes = self.config.get('min_strikes', 3)
        self.weight_decay_factor = self.config.get('weight_decay_factor', 0.8)
        
        logger.debug(f"Initialized {self.name} with config: {self.config}")
    
    @abstractmethod
    def select_strikes(self, 
                      market_data: pd.DataFrame, 
                      spot_price: float,
                      dte: Optional[int] = None,
                      **kwargs) -> List[StrikeInfo]:
        """
        Select strikes based on strategy
        
        Args:
            market_data: Market data DataFrame
            spot_price: Current underlying spot price
            dte: Days to expiry (optional)
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            List[StrikeInfo]: Selected strikes with weights and metadata
        """
        pass
    
    def validate_selection(self, strikes: List[StrikeInfo]) -> bool:
        """
        Validate strike selection
        
        Args:
            strikes: List of selected strikes
            
        Returns:
            bool: True if selection is valid
        """
        if len(strikes) < self.min_strikes:
            logger.warning(f"Insufficient strikes selected: {len(strikes)} < {self.min_strikes}")
            return False
        
        if len(strikes) > self.max_strikes:
            logger.warning(f"Too many strikes selected: {len(strikes)} > {self.max_strikes}")
            return False
        
        # Check weight normalization
        total_weight = sum(strike.weight for strike in strikes)
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Strike weights not normalized: total = {total_weight}")
            return False
        
        return True
    
    def normalize_weights(self, strikes: List[StrikeInfo]) -> List[StrikeInfo]:
        """
        Normalize strike weights to sum to 1.0
        
        Args:
            strikes: List of strikes with weights
            
        Returns:
            List[StrikeInfo]: Strikes with normalized weights
        """
        total_weight = sum(strike.weight for strike in strikes)
        
        if total_weight <= 0:
            # Equal weights if no valid weights
            equal_weight = 1.0 / len(strikes) if strikes else 0.0
            for strike in strikes:
                strike.weight = equal_weight
        else:
            # Normalize existing weights
            for strike in strikes:
                strike.weight /= total_weight
        
        return strikes

class FullChainStrikeSelector(BaseStrikeSelector):
    """
    Select all available strikes with distance-based weighting
    
    Good for Greek sentiment analysis where all strikes matter
    """
    
    def select_strikes(self, 
                      market_data: pd.DataFrame, 
                      spot_price: float,
                      dte: Optional[int] = None,
                      **kwargs) -> List[StrikeInfo]:
        """Select all available strikes with distance-based weighting"""
        try:
            strikes = []
            
            # Get unique strikes from market data
            if 'strike' in market_data.columns:
                available_strikes = market_data['strike'].unique()
            else:
                logger.error("No strike column found in market data")
                return []
            
            for strike_price in sorted(available_strikes):
                # Calculate distance from ATM
                distance = abs(strike_price - spot_price) / spot_price
                
                # Distance-based weight (closer = higher weight)
                base_weight = np.exp(-distance * 5.0)  # Exponential decay
                
                # Check if we have both CE and PE data
                strike_data = market_data[market_data['strike'] == strike_price]
                
                if not strike_data.empty:
                    # Get DTE if available
                    strike_dte = dte or strike_data['dte'].iloc[0] if 'dte' in strike_data.columns else 30
                    
                    # Add CE strike if available
                    ce_data = strike_data[strike_data['option_type'] == 'CE']
                    if not ce_data.empty:
                        strikes.append(StrikeInfo(
                            strike=strike_price,
                            distance_from_atm=distance,
                            weight=base_weight,
                            option_type='CE',
                            dte=strike_dte,
                            selection_reason='full_chain_ce',
                            confidence=1.0 - distance  # Higher confidence for ATM
                        ))
                    
                    # Add PE strike if available
                    pe_data = strike_data[strike_data['option_type'] == 'PE']
                    if not pe_data.empty:
                        strikes.append(StrikeInfo(
                            strike=strike_price,
                            distance_from_atm=distance,
                            weight=base_weight,
                            option_type='PE',
                            dte=strike_dte,
                            selection_reason='full_chain_pe',
                            confidence=1.0 - distance
                        ))
            
            # Normalize weights
            strikes = self.normalize_weights(strikes)
            
            # Limit to max_strikes if needed
            if len(strikes) > self.max_strikes:
                # Sort by weight (descending) and take top strikes
                strikes.sort(key=lambda x: x.weight, reverse=True)
                strikes = strikes[:self.max_strikes]
                strikes = self.normalize_weights(strikes)  # Re-normalize
            
            logger.debug(f"FullChainStrikeSelector selected {len(strikes)} strikes")
            return strikes
            
        except Exception as e:
            logger.error(f"Error in FullChainStrikeSelector: {e}")
            return []

class DynamicRangeStrikeSelector(BaseStrikeSelector):
    """
    Select strikes within dynamic range based on volatility and market conditions
    
    Good for OI/PA analysis and volatility-sensitive indicators
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.base_range = self.config.get('base_range', 0.05)  # 5% base range
        self.volatility_multiplier = self.config.get('volatility_multiplier', 2.0)
        self.min_range = self.config.get('min_range', 0.02)  # 2% minimum
        self.max_range = self.config.get('max_range', 0.15)  # 15% maximum
    
    def select_strikes(self, 
                      market_data: pd.DataFrame, 
                      spot_price: float,
                      dte: Optional[int] = None,
                      **kwargs) -> List[StrikeInfo]:
        """Select strikes within dynamic volatility-adjusted range"""
        try:
            # Calculate dynamic range based on volatility
            volatility = kwargs.get('volatility', 0.2)  # Default 20% IV
            dynamic_range = self.base_range * (1 + volatility * self.volatility_multiplier)
            dynamic_range = np.clip(dynamic_range, self.min_range, self.max_range)
            
            # Calculate strike range
            lower_bound = spot_price * (1 - dynamic_range)
            upper_bound = spot_price * (1 + dynamic_range)
            
            logger.debug(f"Dynamic range: {dynamic_range:.1%}, bounds: [{lower_bound:.1f}, {upper_bound:.1f}]")
            
            strikes = []
            
            # Filter strikes within range
            if 'strike' in market_data.columns:
                available_strikes = market_data['strike'].unique()
                in_range_strikes = [s for s in available_strikes if lower_bound <= s <= upper_bound]
            else:
                logger.error("No strike column found in market data")
                return []
            
            for strike_price in sorted(in_range_strikes):
                # Calculate distance and weight
                distance = abs(strike_price - spot_price) / spot_price
                weight = np.exp(-distance / dynamic_range * 3.0)  # Scaled exponential decay
                
                strike_data = market_data[market_data['strike'] == strike_price]
                if not strike_data.empty:
                    strike_dte = dte or strike_data['dte'].iloc[0] if 'dte' in strike_data.columns else 30
                    
                    # Add both CE and PE
                    for option_type in ['CE', 'PE']:
                        type_data = strike_data[strike_data['option_type'] == option_type]
                        if not type_data.empty:
                            strikes.append(StrikeInfo(
                                strike=strike_price,
                                distance_from_atm=distance,
                                weight=weight,
                                option_type=option_type,
                                dte=strike_dte,
                                selection_reason=f'dynamic_range_{option_type.lower()}',
                                confidence=np.exp(-distance / dynamic_range)
                            ))
            
            # Normalize weights
            strikes = self.normalize_weights(strikes)
            
            logger.debug(f"DynamicRangeStrikeSelector selected {len(strikes)} strikes in range {dynamic_range:.1%}")
            return strikes
            
        except Exception as e:
            logger.error(f"Error in DynamicRangeStrikeSelector: {e}")
            return []

class RollingATMStrikeSelector(BaseStrikeSelector):
    """
    Select strikes focused around rolling ATM with time-based adjustments
    
    Good for technical indicators that need consistent ATM tracking
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.atm_strikes = self.config.get('atm_strikes', 3)  # Number of ATM strikes per side
        self.weight_falloff = self.config.get('weight_falloff', 0.7)  # Weight reduction per strike
    
    def select_strikes(self, 
                      market_data: pd.DataFrame, 
                      spot_price: float,
                      dte: Optional[int] = None,
                      **kwargs) -> List[StrikeInfo]:
        """Select strikes focused around ATM"""
        try:
            strikes = []
            
            if 'strike' in market_data.columns:
                available_strikes = sorted(market_data['strike'].unique())
            else:
                logger.error("No strike column found in market data")
                return []
            
            # Find closest ATM strike
            atm_strike = min(available_strikes, key=lambda x: abs(x - spot_price))
            atm_index = available_strikes.index(atm_strike)
            
            # Select strikes around ATM
            start_idx = max(0, atm_index - self.atm_strikes)
            end_idx = min(len(available_strikes), atm_index + self.atm_strikes + 1)
            
            selected_strikes = available_strikes[start_idx:end_idx]
            
            for i, strike_price in enumerate(selected_strikes):
                # Calculate distance from ATM strike (not spot)
                atm_distance = abs(available_strikes.index(strike_price) - atm_index)
                
                # Weight based on distance from ATM strike
                weight = self.weight_falloff ** atm_distance
                
                # Distance from spot price
                spot_distance = abs(strike_price - spot_price) / spot_price
                
                strike_data = market_data[market_data['strike'] == strike_price]
                if not strike_data.empty:
                    strike_dte = dte or strike_data['dte'].iloc[0] if 'dte' in strike_data.columns else 30
                    
                    # Add both CE and PE with higher weight for ATM
                    for option_type in ['CE', 'PE']:
                        type_data = strike_data[strike_data['option_type'] == option_type]
                        if not type_data.empty:
                            strikes.append(StrikeInfo(
                                strike=strike_price,
                                distance_from_atm=spot_distance,
                                weight=weight,
                                option_type=option_type,
                                dte=strike_dte,
                                selection_reason=f'rolling_atm_{option_type.lower()}',
                                confidence=1.0 / (1.0 + atm_distance)  # Higher confidence for closer to ATM
                            ))
            
            # Normalize weights
            strikes = self.normalize_weights(strikes)
            
            logger.debug(f"RollingATMStrikeSelector selected {len(strikes)} strikes around ATM {atm_strike}")
            return strikes
            
        except Exception as e:
            logger.error(f"Error in RollingATMStrikeSelector: {e}")
            return []

def create_strike_selector(strategy: Union[str, StrikeSelectionStrategy], 
                          config: Optional[Dict[str, Any]] = None) -> BaseStrikeSelector:
    """
    Factory function to create strike selectors
    
    Args:
        strategy: Strategy type (string or enum)
        config: Configuration parameters
        
    Returns:
        BaseStrikeSelector: Configured strike selector
    """
    if isinstance(strategy, str):
        strategy = StrikeSelectionStrategy(strategy)
    
    selector_map = {
        StrikeSelectionStrategy.FULL_CHAIN: FullChainStrikeSelector,
        StrikeSelectionStrategy.DYNAMIC_RANGE: DynamicRangeStrikeSelector,
        StrikeSelectionStrategy.ROLLING_ATM: RollingATMStrikeSelector,
    }
    
    if strategy not in selector_map:
        logger.warning(f"Unknown strategy {strategy}, defaulting to DynamicRangeStrikeSelector")
        strategy = StrikeSelectionStrategy.DYNAMIC_RANGE
    
    selector_class = selector_map[strategy]
    return selector_class(config)