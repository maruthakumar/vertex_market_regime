#!/usr/bin/env python3
"""
OI Strategy Wrapper to adapt OIExecutor to BaseStrategy interface
"""
from typing import Dict, Any, List
import logging
from datetime import datetime
import pandas as pd

from backtester_v2.core.base_strategy import BaseStrategy
from .executor import OIExecutor

logger = logging.getLogger(__name__)


class OIStrategy(BaseStrategy):
    """Wrapper to make OIExecutor compatible with BaseStrategy interface"""
    
    def __init__(self, db_config: Dict[str, Any] = None):
        """Initialize OI strategy wrapper"""
        super().__init__()
        self.executor = OIExecutor(db_config)
        
    def parse_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse OI input files"""
        portfolio_file = input_data.get('portfolio_file')
        max_oi_file = input_data.get('max_oi_file')
        
        if not portfolio_file:
            raise ValueError("portfolio_file is required")
            
        # Use executor's parser
        portfolio_data = self.executor.parser.parse_portfolio_excel(portfolio_file)
        
        # Parse OI specific files if provided
        if max_oi_file:
            oi_data = self.executor.parser.parse_oi_excel(max_oi_file)
            portfolio_data.update(oi_data)
        
        # Add dates from params
        portfolio_data['start_date'] = input_data.get('start_date', '2024-01-01')
        portfolio_data['end_date'] = input_data.get('end_date', '2024-12-31')
        portfolio_data['portfolio_file'] = portfolio_file
        
        return portfolio_data
        
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data"""
        return bool(input_data.get('portfolio_file'))
        
    def generate_query(self, parsed_data: Dict[str, Any]) -> List[str]:
        """Generate queries - OI uses internal query generation"""
        # OI executor handles query generation internally
        # Return empty list as queries are generated during execution
        return []
        
    def process_results(self, results: List[pd.DataFrame], parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process results by executing OI backtest"""
        # Convert string dates to date objects
        from datetime import datetime
        start_date = datetime.strptime(parsed_data['start_date'], '%Y-%m-%d').date()
        end_date = datetime.strptime(parsed_data['end_date'], '%Y-%m-%d').date()
        
        # Execute backtest through executor
        result = self.executor.execute_oi_backtest(
            portfolio_file=parsed_data['portfolio_file'],
            start_date=start_date,
            end_date=end_date,
            max_oi_file=parsed_data.get('max_oi_file'),
            premium_shift_file=parsed_data.get('premium_shift_file'),
            previous_close_file=parsed_data.get('previous_close_file')
        )
        
        # Convert executor result to standard format
        if result.get('status') == 'success':
            return {
                'trades': result.get('trades_df', pd.DataFrame()),
                'metrics': result.get('summary', {}),
                'daily_pnl': result.get('daily_pnl_df', pd.DataFrame()),
                'strategy_summary': result.get('strategy_summary', {})
            }
        else:
            raise RuntimeError(f"OI execution failed: {result.get('message', 'Unknown error')}")