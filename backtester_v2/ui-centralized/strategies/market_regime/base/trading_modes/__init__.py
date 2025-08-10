"""
Trading Modes Module for Market Regime Analysis
===============================================

This module handles intraday/positional optimization and trading mode
management for the market regime system, integrating the existing
adaptive timeframe manager into the modular structure.

Components:
- intraday_optimizer: Optimizes parameters for intraday trading
- positional_optimizer: Optimizes parameters for positional trading  
- hybrid_mode_manager: Manages hybrid trading modes
- timeframe_coordinator: Integrates adaptive timeframe management
- trading_mode_orchestrator: Coordinates all trading modes

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 1.0.0
"""

from .intraday_optimizer import IntradayOptimizer
from .positional_optimizer import PositionalOptimizer
from .hybrid_mode_manager import HybridModeManager
from .timeframe_coordinator import TimeframeCoordinator
from .trading_mode_orchestrator import TradingModeOrchestrator

__all__ = [
    'IntradayOptimizer',
    'PositionalOptimizer',
    'HybridModeManager', 
    'TimeframeCoordinator',
    'TradingModeOrchestrator'
]