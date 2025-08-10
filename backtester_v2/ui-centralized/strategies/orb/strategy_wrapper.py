#!/usr/bin/env python3
"""
ORB Strategy Wrapper to adapt ORBExecutor to BaseStrategy interface
"""
from typing import Dict, Any, List
import logging
from datetime import datetime
import pandas as pd
import sys
import os

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
backtester_v2_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, backtester_v2_dir)

from core.base_strategy import BaseStrategy
from core.sentry_config import capture_exception, add_breadcrumb, track_errors, capture_message, set_tag, set_context
from .executor import ORBExecutor

logger = logging.getLogger(__name__)


class ORBStrategy(BaseStrategy):
    """Wrapper to make ORBExecutor compatible with BaseStrategy interface"""
    
    def __init__(self, db_config: Dict[str, Any] = None):
        """Initialize ORB strategy wrapper"""
        try:
            set_tag("module", "orb_strategy")
            set_tag("strategy_type", "ORB")
            
            super().__init__()
            self.executor = ORBExecutor(db_config)
            
            add_breadcrumb(
                message="ORBStrategy initialized",
                category="strategy.orb",
                level="info"
            )
            
            capture_message(
                "ORBStrategy initialized successfully",
                level="info",
                module="orb_strategy"
            )
            
        except Exception as e:
            capture_exception(
                e,
                context="Failed to initialize ORBStrategy",
                module="orb_strategy",
                db_config=str(db_config)
            )
            raise
        
    @track_errors
    def parse_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse ORB input files"""
        set_tag("operation", "parse_input")
        
        portfolio_file = input_data.get('portfolio_file')
        orb_file = input_data.get('orb_file')
        
        set_context("orb_parse_input", {
            "portfolio_file": portfolio_file,
            "orb_file": orb_file,
            "has_portfolio": bool(portfolio_file),
            "has_orb": bool(orb_file),
            "start_date": input_data.get('start_date'),
            "end_date": input_data.get('end_date')
        })
        
        add_breadcrumb(
            message="Parsing ORB input files",
            category="strategy.orb",
            data={
                "portfolio_file": portfolio_file,
                "orb_file": orb_file
            }
        )
        
        if not portfolio_file or not orb_file:
            error_msg = "Both portfolio_file and orb_file are required"
            capture_message(
                error_msg,
                level="error",
                input_data=input_data
            )
            raise ValueError(error_msg)
        
        try:
            # Use executor's parser
            orb_data = self.executor.parser.parse_orb_excel(orb_file)
            
            add_breadcrumb(
                message="ORB file parsed successfully",
                category="strategy.orb",
                data={
                    "settings_count": len(orb_data.get('settings', [])),
                    "has_entry_params": bool(orb_data.get('entry_params')),
                    "has_exit_params": bool(orb_data.get('exit_params'))
                }
            )
            
        except Exception as e:
            capture_exception(
                e,
                context="Failed to parse ORB file",
                orb_file=orb_file
            )
            raise
        
        # Add dates from params
        orb_data['start_date'] = input_data.get('start_date', '2024-01-01')
        orb_data['end_date'] = input_data.get('end_date', '2024-12-31')
        orb_data['portfolio_file'] = portfolio_file
        orb_data['orb_file'] = orb_file
        
        capture_message(
            "ORB input parsing completed successfully",
            level="info",
            portfolio_file=portfolio_file,
            orb_file=orb_file,
            start_date=orb_data['start_date'],
            end_date=orb_data['end_date']
        )
        
        return orb_data
        
    @track_errors
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data"""
        set_tag("operation", "validate_input")
        
        has_portfolio = bool(input_data.get('portfolio_file'))
        has_orb = bool(input_data.get('orb_file'))
        
        is_valid = has_portfolio and has_orb
        
        add_breadcrumb(
            message="ORB input validation",
            category="strategy.orb",
            data={
                "has_portfolio": has_portfolio,
                "has_orb": has_orb,
                "is_valid": is_valid
            }
        )
        
        if not is_valid:
            capture_message(
                "ORB validation failed",
                level="warning",
                has_portfolio=has_portfolio,
                has_orb=has_orb
            )
        
        return is_valid
        
    @track_errors
    def generate_query(self, parsed_data: Dict[str, Any]) -> List[str]:
        """Generate queries - ORB uses internal query generation"""
        set_tag("operation", "generate_query")
        
        add_breadcrumb(
            message="ORB query generation (internal)",
            category="strategy.orb",
            data={
                "note": "ORB uses internal query generation during execution"
            }
        )
        
        # ORB executor handles query generation internally
        # Return empty list as queries are generated during execution
        return []
        
    @track_errors
    def process_results(self, results: List[pd.DataFrame], parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process results by executing ORB backtest"""
        set_tag("operation", "process_results")
        
        # Convert string dates to date objects
        from datetime import datetime
        start_date = datetime.strptime(parsed_data['start_date'], '%Y-%m-%d').date()
        end_date = datetime.strptime(parsed_data['end_date'], '%Y-%m-%d').date()
        
        set_context("orb_process_results", {
            "start_date": str(start_date),
            "end_date": str(end_date),
            "orb_file": parsed_data.get('orb_file'),
            "portfolio_file": parsed_data.get('portfolio_file')
        })
        
        add_breadcrumb(
            message="Starting ORB backtest execution",
            category="strategy.orb",
            data={
                "start_date": str(start_date),
                "end_date": str(end_date)
            }
        )
        
        try:
            # Execute backtest through executor
            result = self.executor.execute_orb_backtest(
                input_file=parsed_data['orb_file'],
                portfolio_file=parsed_data['portfolio_file'],
                start_date=start_date,
                end_date=end_date
            )
            
            add_breadcrumb(
                message="ORB backtest execution completed",
                category="strategy.orb",
                data={
                    "status": result.get('status'),
                    "has_trades": bool(result.get('trades_df') is not None),
                    "has_summary": bool(result.get('summary'))
                }
            )
            
        except Exception as e:
            capture_exception(
                e,
                context="Failed to execute ORB backtest",
                start_date=str(start_date),
                end_date=str(end_date),
                orb_file=parsed_data.get('orb_file')
            )
            raise
        
        # Convert executor result to standard format
        if result.get('status') == 'success':
            output = {
                'trades': result.get('trades_df', pd.DataFrame()),
                'metrics': result.get('summary', {}),
                'daily_pnl': result.get('daily_pnl_df', pd.DataFrame()),
                'strategy_summary': result.get('strategy_summary', {})
            }
            
            # Track metrics
            metrics = output.get('metrics', {})
            capture_message(
                "ORB backtest completed successfully",
                level="info",
                total_trades=metrics.get('total_trades', 0),
                winning_trades=metrics.get('winning_trades', 0),
                losing_trades=metrics.get('losing_trades', 0),
                total_pnl=metrics.get('total_pnl', 0),
                win_rate=metrics.get('win_rate', 0)
            )
            
            return output
        else:
            error_msg = f"ORB execution failed: {result.get('message', 'Unknown error')}"
            capture_message(
                error_msg,
                level="error",
                status=result.get('status'),
                message=result.get('message')
            )
            raise RuntimeError(error_msg)