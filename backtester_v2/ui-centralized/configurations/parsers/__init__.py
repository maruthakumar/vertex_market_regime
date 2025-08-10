"""
Configuration parsers for different file formats and strategy types
"""

from .base_parser import BaseParser
from .excel_parser import ExcelParser
from .ml_triple_straddle_parser import MLTripleStraddleParser
from .indicator_parser import IndicatorParser

# Strategy-specific parsers will be imported as they are created
_parsers = {}

def register_parser(strategy_type: str, parser_class):
    """Register a parser for a strategy type"""
    _parsers[strategy_type.lower()] = parser_class

def get_parser(strategy_type: str):
    """Get parser for a strategy type"""
    strategy_type = strategy_type.lower()
    
    if strategy_type not in _parsers:
        # Try to import the parser dynamically
        try:
            module_name = f"{strategy_type}_parser"
            module = __import__(f"configurations.parsers.{module_name}", fromlist=[f"{strategy_type.upper()}Parser"])
            parser_class = getattr(module, f"{strategy_type.upper()}Parser")
            register_parser(strategy_type, parser_class)
        except (ImportError, AttributeError):
            # Fall back to generic Excel parser
            register_parser(strategy_type, ExcelParser)
    
    parser_class = _parsers[strategy_type]
    return parser_class()

__all__ = [
    "BaseParser",
    "ExcelParser",
    "MLTripleStraddleParser",
    "IndicatorParser",
    "register_parser",
    "get_parser"
]

# Register specialized parsers
register_parser('ml_triple_straddle', MLTripleStraddleParser)
register_parser('indicator', IndicatorParser)