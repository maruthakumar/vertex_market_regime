#!/usr/bin/env python3
"""
TBS Strategy - Fixed to work with actual input file structure
"""

import os
import logging
from typing import Dict, List, Any, Optional
import pandas as pd

import sys

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
backtester_v2_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, backtester_v2_dir)

from core.base_strategy import BaseStrategy
from core.database.connection_manager import DatabaseManager, get_database_manager
from .parser import TBSParser
from .query_builder import TBSQueryBuilder
from .processor import TBSProcessor

logger = logging.getLogger(__name__)


class TBSStrategy(BaseStrategy):
    """Trade Builder Strategy implementation - Fixed for actual file structure"""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """Initialize TBS strategy"""
        super().__init__()
        self.parser = TBSParser()
        self.query_builder = TBSQueryBuilder()
        self.processor = TBSProcessor()
        self.db_manager = db_manager or get_database_manager()
        
    def parse_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse TBS input files
        
        Args:
            input_data: Dictionary containing paths to input files
                - portfolio_excel: Path to portfolio Excel file
                - tbs_excel: Optional path to TBS strategy Excel file
                
        Returns:
            Parsed strategy parameters
        """
        portfolio_excel = input_data.get('portfolio_excel')
        tbs_excel = input_data.get('tbs_excel')
        
        if not portfolio_excel:
            raise ValueError("Portfolio Excel path is required")
        
        logger.info(f"Parsing TBS input files: {portfolio_excel}")
        
        # Parse portfolio Excel
        portfolio_data = self.parser.parse_portfolio_excel(portfolio_excel)
        
        # Find all TBS strategies
        all_strategies = []
        
        # If specific TBS excel provided, use it
        if tbs_excel and os.path.exists(tbs_excel):
            strategy_data = self.parser.parse_multi_leg_excel(tbs_excel)
            all_strategies.extend(strategy_data['strategies'])
        else:
            # Load strategy files referenced in portfolio
            for portfolio_strategy in portfolio_data.get('strategies', []):
                strategy_file = portfolio_strategy.get('strategy_excel_file_path')
                
                # Try to resolve the path
                if strategy_file and not os.path.exists(strategy_file):
                    # Try relative to portfolio file
                    strategy_file = os.path.join(
                        os.path.dirname(portfolio_excel), 
                        os.path.basename(strategy_file)
                    )
                
                if strategy_file and os.path.exists(strategy_file):
                    try:
                        strategy_data = self.parser.parse_multi_leg_excel(strategy_file)
                        # Add portfolio reference to each strategy
                        for strategy in strategy_data['strategies']:
                            strategy['portfolio_ref'] = portfolio_strategy
                        all_strategies.extend(strategy_data['strategies'])
                    except Exception as e:
                        logger.warning(f"Failed to load strategy file {strategy_file}: {e}")
        
        # Build final structure
        result = {
            'portfolio': portfolio_data['portfolio'],
            'strategies': all_strategies,
            'source_files': {
                'portfolio': portfolio_excel,
                'strategies': [s.get('source_file') for s in all_strategies]
            }
        }
        
        # Validate the parsed data
        validation_errors = self.validate_input(result)
        if validation_errors:
            raise ValueError(f"Input validation failed: {validation_errors}")
        
        return result
    
    def generate_query(self, params: Dict[str, Any]) -> List[str]:
        """Generate SQL queries for TBS execution"""
        logger.info("Generating TBS queries")
        
        # Get strategies with legs
        strategies = params.get('strategies', [])
        strategies_with_legs = [s for s in strategies if s.get('legs')]
        
        if not strategies_with_legs:
            logger.warning("No strategies with legs found")
            return []
        
        # Build query parameters
        query_params = {
            'portfolio_settings': params['portfolio'],
            'strategies': strategies_with_legs
        }
        
        # Generate queries
        queries = self.query_builder.build_queries(query_params)
        
        logger.info(f"Generated {len(queries)} queries for TBS execution")
        return queries
    
    def process_results(self, results: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process query results into backtest output"""
        logger.info("Processing TBS results")
        
        # Process results
        output = self.processor.process_results(
            query_results=[results],
            strategy_params=params
        )
        
        return output
    
    def validate_input(self, params: Dict[str, Any]) -> List[str]:
        """Validate TBS input parameters"""
        errors = []
        
        # Validate portfolio settings
        portfolio = params.get('portfolio', {})
        if not portfolio:
            errors.append("Portfolio settings are missing")
        else:
            if not portfolio.get('start_date'):
                errors.append("Start date is required")
            if not portfolio.get('end_date'):
                errors.append("End date is required")
            if portfolio.get('start_date') and portfolio.get('end_date'):
                if portfolio['start_date'] > portfolio['end_date']:
                    errors.append("Start date must be before end date")
        
        # Validate strategies
        strategies = params.get('strategies', [])
        if not strategies:
            errors.append("At least one strategy is required")
        
        # Check for strategies with legs
        strategies_with_legs = [s for s in strategies if s.get('legs')]
        if not strategies_with_legs:
            errors.append("At least one strategy with legs is required")
        
        for i, strategy in enumerate(strategies):
            strategy_name = strategy.get('strategy_name', f'Strategy_{i+1}')
            
            # Validate legs
            legs = strategy.get('legs', [])
            if not legs:
                logger.warning(f"{strategy_name}: No legs defined")
                continue
            
            for j, leg in enumerate(legs):
                leg_errors = self._validate_leg(leg, f"{strategy_name}_Leg{j+1}")
                errors.extend(leg_errors)
        
        return errors
    
    def _validate_leg(self, leg: Dict[str, Any], leg_name: str) -> List[str]:
        """Validate a single leg configuration"""
        errors = []
        
        # Required fields - using actual column names
        if not leg.get('option_type'):
            errors.append(f"{leg_name}: Option type is required")
        elif leg['option_type'] not in ['CE', 'PE', 'FUT']:
            errors.append(f"{leg_name}: Invalid option type '{leg['option_type']}'")
        
        if not leg.get('strike_selection'):
            errors.append(f"{leg_name}: Strike selection is required")
        
        if not leg.get('expiry_rule'):
            errors.append(f"{leg_name}: Expiry rule is required")
        
        if leg.get('quantity', 0) <= 0:
            errors.append(f"{leg_name}: Quantity must be positive")
        
        if not leg.get('transaction_type'):
            errors.append(f"{leg_name}: Transaction type is required")
        elif leg['transaction_type'] not in ['BUY', 'SELL']:
            errors.append(f"{leg_name}: Invalid transaction type '{leg['transaction_type']}'")
        
        return errors