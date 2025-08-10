#!/usr/bin/env python3
"""
ORB (Opening Range Breakout) Strategy Module

This module implements the Opening Range Breakout strategy for the backtester.
ORB strategies trade based on price breakouts above or below the opening range
established during the initial market period.
"""

from .parser import ORBParser
from .models import (
    ORBSettingModel, 
    ORBLegModel, 
    ORBRange, 
    ORBSignal,
    ProcessedORBSignal,
    ORBBreakoutType,
    ORBSignalDirection
)
from .range_calculator import RangeCalculator
from .signal_generator import SignalGenerator, BreakoutType
from .query_builder import ORBQueryBuilder
from .processor import ORBProcessor
from .executor import ORBExecutor

__all__ = [
    'ORBParser',
    'ORBSettingModel',
    'ORBLegModel',
    'ORBRange',
    'ORBSignal',
    'ProcessedORBSignal',
    'ORBBreakoutType',
    'ORBSignalDirection',
    'RangeCalculator',
    'SignalGenerator',
    'BreakoutType',
    'ORBQueryBuilder',
    'ORBProcessor',
    'ORBExecutor'
]

__version__ = '1.0.0'