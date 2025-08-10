"""
Simplified Models for POS (Positional) Strategy
For initial implementation and testing
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import date, time
from enum import Enum


class SimpleLegModel(BaseModel):
    """Simplified leg model for initial implementation"""
    leg_id: int
    leg_name: str
    option_type: str  # CE/PE or CALL/PUT
    position_type: str  # BUY/SELL
    strike_selection: str  # ATM/ITM/OTM/STRIKE_PRICE
    strike_offset: float = 0
    strike_price: Optional[float] = None
    expiry_type: str = "CURRENT_WEEK"  # CURRENT_WEEK/NEXT_WEEK/CURRENT_MONTH/NEXT_MONTH
    lots: int = 1
    lot_size: int = 50
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_time: time = time(9, 20)
    exit_time: time = time(15, 20)
    is_active: bool = True
    
    @validator('option_type')
    def normalize_option_type(cls, v):
        v = v.upper()
        if v in ['CALL', 'CE', 'C']:
            return 'CE'
        elif v in ['PUT', 'PE', 'P']:
            return 'PE'
        else:
            raise ValueError(f"Invalid option type: {v}")
    
    @validator('position_type')
    def normalize_position_type(cls, v):
        v = v.upper()
        if v in ['BUY', 'LONG', 'B']:
            return 'BUY'
        elif v in ['SELL', 'SHORT', 'S']:
            return 'SELL'
        else:
            raise ValueError(f"Invalid position type: {v}")
    
    @validator('strike_selection')
    def validate_strike_selection(cls, v, values):
        v = v.upper()
        if v == 'STRIKE_PRICE' and 'strike_price' in values and values['strike_price'] is None:
            raise ValueError("strike_price required when strike_selection is STRIKE_PRICE")
        return v


class SimplePortfolioModel(BaseModel):
    """Simplified portfolio model"""
    portfolio_name: str
    start_date: date
    end_date: date
    index_name: str = "NIFTY"
    position_size_value: float = 100000
    transaction_costs: float = 0.001
    slippage_value: float = 0.001
    use_intraday_data: bool = False
    calculate_greeks: bool = True
    enable_adjustments: bool = False
    max_portfolio_risk: float = 0.02
    strategy_type: str = "CUSTOM"
    
    @validator('end_date')
    def validate_dates(cls, v, values):
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError("end_date must be after start_date")
        return v


class SimplePOSStrategy(BaseModel):
    """Complete simplified strategy"""
    portfolio: SimplePortfolioModel
    legs: List[SimpleLegModel]
    strategy_type: str = "CUSTOM"
    
    @validator('legs')
    def validate_legs(cls, v):
        if not v:
            raise ValueError("At least one leg is required")
        if len(v) > 20:
            raise ValueError("Maximum 20 legs allowed")
        return v
    
    def get_summary(self) -> Dict[str, Any]:
        """Get strategy summary"""
        return {
            'portfolio_name': self.portfolio.portfolio_name,
            'strategy_type': self.strategy_type,
            'start_date': str(self.portfolio.start_date),
            'end_date': str(self.portfolio.end_date),
            'index': self.portfolio.index_name,
            'num_legs': len(self.legs),
            'legs': [
                {
                    'name': leg.leg_name,
                    'type': f"{leg.position_type} {leg.option_type}",
                    'strike': f"{leg.strike_selection} {leg.strike_offset}" if leg.strike_offset else leg.strike_selection,
                    'lots': leg.lots
                }
                for leg in self.legs
            ]
        }


# Trade result models
class TradeResult(BaseModel):
    """Single trade result"""
    trade_id: str
    trade_date: date
    trade_time: time
    trade_type: str  # ENTRY/EXIT/ADJUSTMENT
    leg_name: str
    option_type: str
    position_type: str
    strike_price: float
    expiry_date: date
    quantity: int
    price: float
    premium: float
    pnl: float = 0
    transaction_cost: float = 0
    slippage: float = 0
    underlying_price: float = 0
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None


class BacktestResult(BaseModel):
    """Complete backtest result"""
    strategy_summary: Dict[str, Any]
    trades: List[TradeResult]
    metrics: Dict[str, float]
    daily_pnl: Optional[List[Dict[str, Any]]] = None
    errors: List[str] = []
    warnings: List[str] = []