#!/usr/bin/env python3
"""
TBS Strategy - Main TBS strategy implementation
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

import sys
import os

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
    """Trade Builder Strategy implementation"""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """Initialize TBS strategy
        
        Args:
            db_manager: Database connection manager instance
        """
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
                - multi_leg_excel: Optional path to multi-leg Excel file
                
        Returns:
            Parsed strategy parameters
        """
        portfolio_excel = input_data.get('portfolio_excel')
        multi_leg_excel = input_data.get('multi_leg_excel')
        
        if not portfolio_excel:
            raise ValueError("Portfolio Excel path is required")
        
        logger.info(f"Parsing TBS input files: {portfolio_excel}")
        
        # Parse portfolio Excel
        portfolio_data = self.parser.parse_portfolio_excel(portfolio_excel)
        
        # Parse multi-leg Excel if provided
        if multi_leg_excel and os.path.exists(multi_leg_excel):
            multi_leg_data = self.parser.parse_multi_leg_excel(multi_leg_excel)
            
            # Merge multi-leg data with portfolio data
            portfolio_data = self._merge_multi_leg_data(portfolio_data, multi_leg_data)
        
        # Validate the parsed data
        validation_errors = self.validate_input(portfolio_data)
        if validation_errors:
            raise ValueError(f"Input validation failed: {validation_errors}")
        
        return portfolio_data
    
    def generate_query(self, params: Dict[str, Any]) -> List[str]:
        """Generate SQL queries for TBS execution
        
        Args:
            params: Strategy parameters from parse_input
            
        Returns:
            List of SQL queries to execute
        """
        logger.info("Generating TBS queries")
        
        # Extract enabled strategies
        enabled_strategies = [
            strategy for strategy in params.get('strategies', [])
            if strategy.get('enabled', True)
        ]
        
        if not enabled_strategies:
            logger.warning("No enabled strategies found")
            return []
        
        all_queries = []
        
        # Generate queries for each strategy
        for strategy in enabled_strategies:
            strategy_params = {
                'portfolio': params['portfolio'],
                'legs': strategy.get('legs', [strategy])  # Single leg if not multi-leg
            }
            
            queries = self.query_builder.build_queries(strategy_params)
            all_queries.extend(queries)
        
        logger.info(f"Generated {len(all_queries)} queries for TBS execution")
        return all_queries
    
    def process_results(self, results: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process query results into backtest output
        
        Args:
            results: Combined results from all queries
            params: Original strategy parameters
            
        Returns:
            Processed backtest results
        """
        logger.info("Processing TBS results")
        
        # Group results by strategy
        strategy_results = self._group_results_by_strategy(results, params)
        
        # Process each strategy's results
        all_results = []
        for strategy_name, strategy_data in strategy_results.items():
            strategy_output = self.processor.process_results(
                query_results=[strategy_data['results']],
                strategy_params=strategy_data['params']
            )
            strategy_output['strategy_name'] = strategy_name
            all_results.append(strategy_output)
        
        # Combine results from all strategies
        combined_results = self._combine_strategy_results(all_results)
        
        logger.info(f"Processed {len(all_results)} strategies")
        return combined_results
    
    def validate_input(self, params: Dict[str, Any]) -> List[str]:
        """Validate TBS input parameters
        
        Args:
            params: Parsed strategy parameters
            
        Returns:
            List of validation errors (empty if valid)
        """
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
            if portfolio.get('capital', 0) <= 0:
                errors.append("Capital must be positive")
        
        # Validate strategies
        strategies = params.get('strategies', [])
        if not strategies:
            errors.append("At least one strategy is required")
        
        for i, strategy in enumerate(strategies):
            strategy_name = strategy.get('strategy_name', f'Strategy_{i+1}')
            
            # Validate legs
            legs = strategy.get('legs', [strategy])
            if not legs:
                errors.append(f"{strategy_name}: At least one leg is required")
            
            for j, leg in enumerate(legs):
                leg_errors = self._validate_leg(leg, f"{strategy_name}_Leg{j+1}")
                errors.extend(leg_errors)
        
        return errors
    
    def _validate_leg(self, leg: Dict[str, Any], leg_name: str) -> List[str]:
        """Validate a single leg configuration"""
        errors = []
        
        # Required fields
        if not leg.get('option_type'):
            errors.append(f"{leg_name}: Option type is required")
        elif leg['option_type'] not in ['CE', 'PE', 'FUT']:
            errors.append(f"{leg_name}: Invalid option type '{leg['option_type']}'")
        
        if not leg.get('strike_selection'):
            errors.append(f"{leg_name}: Strike selection is required")
        
        if leg.get('strike_selection') == 'FIXED' and not leg.get('strike_value'):
            errors.append(f"{leg_name}: Strike value is required for FIXED selection")
        
        if not leg.get('expiry_rule'):
            errors.append(f"{leg_name}: Expiry rule is required")
        
        if leg.get('quantity', 0) <= 0:
            errors.append(f"{leg_name}: Quantity must be positive")
        
        # Validate times
        entry_time = leg.get('entry_time')
        exit_time = leg.get('exit_time')
        if entry_time and exit_time and entry_time >= exit_time:
            errors.append(f"{leg_name}: Entry time must be before exit time")
        
        # Validate SL/Target
        sl_percent = leg.get('sl_percent')
        if sl_percent is not None and sl_percent < 0:
            errors.append(f"{leg_name}: Stop loss percent cannot be negative")
        
        target_percent = leg.get('target_percent')
        if target_percent is not None and target_percent < 0:
            errors.append(f"{leg_name}: Target percent cannot be negative")
        
        return errors
    
    def _merge_multi_leg_data(self, portfolio_data: Dict[str, Any], 
                             multi_leg_data: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multi-leg data with portfolio data"""
        # Get general parameters
        general_params = multi_leg_data.get('general_parameters', [])
        leg_params = multi_leg_data.get('leg_parameters', [])
        
        # Group legs by strategy name
        legs_by_strategy = {}
        for leg in leg_params:
            strategy_name = leg.get('strategy_name')
            if strategy_name:
                if strategy_name not in legs_by_strategy:
                    legs_by_strategy[strategy_name] = []
                legs_by_strategy[strategy_name].append(leg)
        
        # Create multi-leg strategies
        multi_leg_strategies = []
        for general in general_params:
            strategy_name = general.get('strategy_name')
            if strategy_name in legs_by_strategy:
                strategy = {
                    'strategy_name': strategy_name,
                    'enabled': general.get('enabled', True),
                    'legs': legs_by_strategy[strategy_name]
                }
                # Add general parameters to strategy
                for key, value in general.items():
                    if key not in strategy:
                        strategy[key] = value
                
                multi_leg_strategies.append(strategy)
        
        # Merge with existing strategies
        if multi_leg_strategies:
            portfolio_data['strategies'].extend(multi_leg_strategies)
        
        return portfolio_data
    
    def _group_results_by_strategy(self, results: pd.DataFrame, 
                                  params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Group query results by strategy"""
        # For now, return a simple grouping
        # In a real implementation, this would parse the results based on strategy identifiers
        grouped = {}
        
        for strategy in params.get('strategies', []):
            strategy_name = strategy.get('strategy_name', 'Unknown')
            grouped[strategy_name] = {
                'results': results,  # In practice, filter by strategy
                'params': {
                    'portfolio': params['portfolio'],
                    'legs': strategy.get('legs', [strategy])
                }
            }
        
        return grouped
    
    def _combine_strategy_results(self, strategy_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine results from multiple strategies"""
        if not strategy_results:
            return {
                'success': False,
                'message': 'No strategy results to combine'
            }
        
        # Combine all trades
        all_trades = []
        for result in strategy_results:
            if result.get('success') and not result.get('trades_df', pd.DataFrame()).empty:
                trades = result['trades_df'].copy()
                trades['StrategyName'] = result.get('strategy_name', 'Unknown')
                all_trades.append(trades)
        
        if not all_trades:
            return {
                'success': False,
                'message': 'No successful trades from any strategy'
            }
        
        # Combine trades
        combined_trades = pd.concat(all_trades, ignore_index=True)
        combined_trades = combined_trades.sort_values('EntryDateTime').reset_index(drop=True)
        
        # Recalculate cumulative P&L
        combined_trades['CumulativePnL'] = combined_trades['PnL'].cumsum()
        
        # Get initial capital from first result
        initial_capital = strategy_results[0]['metrics'].get('InitialCapital', 1000000)
        combined_trades['Capital'] = initial_capital + combined_trades['CumulativePnL']
        
        # Combine metrics
        combined_metrics = self._combine_metrics(strategy_results, combined_trades, initial_capital)
        
        # Combine daily P&L
        daily_pnls = [r['daily_pnl'] for r in strategy_results if not r.get('daily_pnl', pd.DataFrame()).empty]
        if daily_pnls:
            combined_daily_pnl = pd.concat(daily_pnls).groupby('Date').sum().reset_index()
            combined_daily_pnl['CumulativePnL'] = combined_daily_pnl['NetPnL'].cumsum()
        else:
            combined_daily_pnl = pd.DataFrame()
        
        # Combine monthly P&L
        monthly_pnls = [r['monthly_pnl'] for r in strategy_results if not r.get('monthly_pnl', pd.DataFrame()).empty]
        if monthly_pnls:
            combined_monthly_pnl = pd.concat(monthly_pnls).groupby('Month').sum().reset_index()
            combined_monthly_pnl['CumulativePnL'] = combined_monthly_pnl['NetPnL'].cumsum()
        else:
            combined_monthly_pnl = pd.DataFrame()
        
        return {
            'success': True,
            'trades_df': combined_trades,
            'metrics': combined_metrics,
            'daily_pnl': combined_daily_pnl,
            'monthly_pnl': combined_monthly_pnl,
            'strategy_results': strategy_results,
            'summary': {
                'total_strategies': len(strategy_results),
                'successful_strategies': len([r for r in strategy_results if r.get('success')]),
                'total_trades': len(combined_trades),
                'net_pnl': combined_metrics.get('NetPnL', 0)
            }
        }
    
    def _combine_metrics(self, strategy_results: List[Dict[str, Any]], 
                        combined_trades: pd.DataFrame,
                        initial_capital: float) -> Dict[str, Any]:
        """Combine metrics from multiple strategies"""
        # Recalculate metrics based on combined trades
        processor = TBSProcessor()
        combined_metrics = processor._calculate_metrics(combined_trades, initial_capital)
        
        # Add strategy-level breakdown
        strategy_metrics = []
        for result in strategy_results:
            if result.get('success'):
                strategy_metrics.append({
                    'strategy_name': result.get('strategy_name'),
                    'metrics': result.get('metrics', {})
                })
        
        combined_metrics['strategy_breakdown'] = strategy_metrics
        
        return combined_metrics