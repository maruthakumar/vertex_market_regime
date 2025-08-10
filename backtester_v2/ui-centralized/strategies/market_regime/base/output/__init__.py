"""
Output Module for Market Regime Analysis
========================================

This module handles all output generation for the market regime system,
including CSV time series, parameter injection, and format management.

Components:
- csv_output_manager: Manages CSV output generation
- time_series_generator: Generates 1-minute time series CSV
- parameter_injector: Injects Excel parameters into output
- output_orchestrator: Coordinates all output formats

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 1.0.0
"""

from .csv_output_manager import CSVOutputManager
from .time_series_generator import TimeSeriesGenerator
from .parameter_injector import ParameterInjector
from .output_orchestrator import OutputOrchestrator

__all__ = [
    'CSVOutputManager',
    'TimeSeriesGenerator', 
    'ParameterInjector',
    'OutputOrchestrator'
]