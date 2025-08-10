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
from core.sentry_config import capture_exception, add_breadcrumb, track_errors, capture_message, set_tag, set_context
from .parser import TBSParser
from .query_builder import TBSQueryBuilder
from .processor import TBSProcessor

logger = logging.getLogger(__name__)


class TBSStrategy(BaseStrategy):
    """Trade Builder Strategy implementation - Fixed for actual file structure"""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """Initialize TBS strategy"""
        try:
            set_tag("module", "tbs_strategy")
            set_tag("strategy_type", "TBS")
            
            super().__init__()
            self.parser = TBSParser()
            self.query_builder = TBSQueryBuilder()
            self.processor = TBSProcessor()
            self.db_manager = db_manager or get_database_manager()
            
            add_breadcrumb(
                message="TBSStrategy initialized",
                category="strategy.tbs",
                level="info"
            )
            
            capture_message(
                "TBSStrategy initialized successfully",
                level="info",
                module="tbs_strategy"
            )
            
        except Exception as e:
            capture_exception(
                e,
                context="Failed to initialize TBSStrategy",
                module="tbs_strategy"
            )
            raise
        
    @track_errors
    def parse_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse TBS input files
        
        Args:
            input_data: Dictionary containing paths to input files
                - portfolio_excel: Path to portfolio Excel file
                - tbs_excel: Optional path to TBS strategy Excel file
                
        Returns:
            Parsed strategy parameters
        """
        set_tag("operation", "parse_input")
        
        portfolio_excel = input_data.get('portfolio_excel')
        tbs_excel = input_data.get('tbs_excel')
        
        set_context("tbs_parse_input", {
            "portfolio_excel": portfolio_excel,
            "tbs_excel": tbs_excel,
            "has_portfolio": bool(portfolio_excel),
            "has_tbs": bool(tbs_excel)
        })
        
        add_breadcrumb(
            message="Parsing TBS input files",
            category="strategy.tbs",
            data={
                "portfolio_excel": portfolio_excel,
                "tbs_excel": tbs_excel
            }
        )
        
        if not portfolio_excel:
            error_msg = "Portfolio Excel path is required"
            capture_message(
                error_msg,
                level="error",
                input_data=input_data
            )
            raise ValueError(error_msg)
        
        logger.info(f"Parsing TBS input files: {portfolio_excel}")
        
        # Parse portfolio Excel
        try:
            portfolio_data = self.parser.parse_portfolio_excel(portfolio_excel)
            add_breadcrumb(
                message="Portfolio Excel parsed successfully",
                category="strategy.tbs",
                data={"portfolio_strategies": len(portfolio_data.get('strategies', []))}
            )
        except Exception as e:
            capture_exception(
                e,
                context="Failed to parse portfolio Excel",
                portfolio_excel=portfolio_excel
            )
            raise
        
        # Find all TBS strategies
        all_strategies = []
        strategy_load_errors = []
        
        # If specific TBS excel provided, use it
        if tbs_excel and os.path.exists(tbs_excel):
            try:
                strategy_data = self.parser.parse_multi_leg_excel(tbs_excel)
                all_strategies.extend(strategy_data['strategies'])
                add_breadcrumb(
                    message=f"Loaded {len(strategy_data['strategies'])} strategies from TBS Excel",
                    category="strategy.tbs"
                )
            except Exception as e:
                capture_exception(
                    e,
                    context="Failed to parse TBS Excel",
                    tbs_excel=tbs_excel
                )
                raise
        else:
            # Load strategy files referenced in portfolio
            for idx, portfolio_strategy in enumerate(portfolio_data.get('strategies', [])):
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
                        
                        add_breadcrumb(
                            message=f"Loaded strategy file {idx+1}: {strategy_file}",
                            category="strategy.tbs",
                            data={"strategies_count": len(strategy_data['strategies'])}
                        )
                    except Exception as e:
                        error_msg = f"Failed to load strategy file {strategy_file}: {e}"
                        logger.warning(error_msg)
                        strategy_load_errors.append(error_msg)
                        capture_exception(
                            e,
                            context="Failed to load strategy file",
                            strategy_file=strategy_file,
                            portfolio_index=idx
                        )
        
        # Build final structure
        result = {
            'portfolio': portfolio_data['portfolio'],
            'strategies': all_strategies,
            'source_files': {
                'portfolio': portfolio_excel,
                'strategies': [s.get('source_file') for s in all_strategies]
            }
        }
        
        # Report loading summary
        if strategy_load_errors:
            capture_message(
                f"Some strategy files failed to load: {len(strategy_load_errors)} errors",
                level="warning",
                errors=strategy_load_errors[:5]  # First 5 errors
            )
        
        # Validate the parsed data
        validation_errors = self.validate_input(result)
        if validation_errors:
            error_msg = f"Input validation failed: {validation_errors}"
            capture_message(
                error_msg,
                level="error",
                validation_errors=validation_errors,
                strategies_count=len(all_strategies)
            )
            raise ValueError(error_msg)
        
        # Success tracking
        capture_message(
            f"TBS input parsing completed successfully",
            level="info",
            portfolio_strategies=len(portfolio_data.get('strategies', [])),
            loaded_strategies=len(all_strategies),
            had_errors=len(strategy_load_errors) > 0
        )
        
        return result
    
    @track_errors
    def generate_query(self, params: Dict[str, Any]) -> List[str]:
        """Generate SQL queries for TBS execution"""
        set_tag("operation", "generate_query")
        
        logger.info("Generating TBS queries")
        
        # Get strategies with legs
        strategies = params.get('strategies', [])
        strategies_with_legs = [s for s in strategies if s.get('legs')]
        
        set_context("tbs_query_generation", {
            "total_strategies": len(strategies),
            "strategies_with_legs": len(strategies_with_legs),
            "portfolio_start": params.get('portfolio', {}).get('start_date'),
            "portfolio_end": params.get('portfolio', {}).get('end_date')
        })
        
        add_breadcrumb(
            message="Starting TBS query generation",
            category="strategy.tbs",
            data={
                "total_strategies": len(strategies),
                "strategies_with_legs": len(strategies_with_legs)
            }
        )
        
        if not strategies_with_legs:
            warning_msg = "No strategies with legs found"
            logger.warning(warning_msg)
            capture_message(
                warning_msg,
                level="warning",
                total_strategies=len(strategies)
            )
            return []
        
        # Build query parameters
        query_params = {
            'portfolio_settings': params['portfolio'],
            'strategies': strategies_with_legs
        }
        
        # Generate queries
        try:
            queries = self.query_builder.build_queries(query_params)
            
            add_breadcrumb(
                message=f"Generated {len(queries)} queries",
                category="strategy.tbs",
                data={
                    "query_count": len(queries),
                    "avg_query_length": sum(len(q) for q in queries) / len(queries) if queries else 0
                }
            )
            
            capture_message(
                f"TBS query generation successful",
                level="info",
                query_count=len(queries),
                strategies_count=len(strategies_with_legs)
            )
            
            logger.info(f"Generated {len(queries)} queries for TBS execution")
            return queries
            
        except Exception as e:
            capture_exception(
                e,
                context="Failed to generate TBS queries",
                strategies_count=len(strategies_with_legs),
                portfolio_settings=str(params.get('portfolio', {}))
            )
            raise
    
    @track_errors
    def process_results(self, results: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process query results into backtest output"""
        set_tag("operation", "process_results")
        
        logger.info("Processing TBS results")
        
        set_context("tbs_process_results", {
            "results_rows": len(results) if results is not None else 0,
            "results_columns": len(results.columns) if results is not None else 0,
            "strategies_count": len(params.get('strategies', [])),
            "has_results": results is not None and not results.empty
        })
        
        add_breadcrumb(
            message="Starting TBS results processing",
            category="strategy.tbs",
            data={
                "rows": len(results) if results is not None else 0,
                "columns": len(results.columns) if results is not None else 0
            }
        )
        
        if results is None or results.empty:
            warning_msg = "No results to process"
            capture_message(
                warning_msg,
                level="warning",
                params_strategies=len(params.get('strategies', []))
            )
            logger.warning(warning_msg)
        
        try:
            # Process results
            output = self.processor.process_results(
                query_results=[results],
                strategy_params=params
            )
            
            # Track output metrics
            if output:
                metrics = output.get('metrics', {})
                add_breadcrumb(
                    message="TBS results processed successfully",
                    category="strategy.tbs",
                    data={
                        "total_trades": metrics.get('total_trades', 0),
                        "winning_trades": metrics.get('winning_trades', 0),
                        "losing_trades": metrics.get('losing_trades', 0),
                        "total_pnl": metrics.get('total_pnl', 0)
                    }
                )
                
                capture_message(
                    "TBS results processing completed",
                    level="info",
                    total_trades=metrics.get('total_trades', 0),
                    total_pnl=metrics.get('total_pnl', 0),
                    win_rate=metrics.get('win_rate', 0)
                )
            
            return output
            
        except Exception as e:
            capture_exception(
                e,
                context="Failed to process TBS results",
                results_shape=(len(results), len(results.columns)) if results is not None else None,
                strategies_count=len(params.get('strategies', []))
            )
            raise
    
    @track_errors
    def validate_input(self, params: Dict[str, Any]) -> List[str]:
        """Validate TBS input parameters"""
        set_tag("operation", "validate_input")
        
        errors = []
        
        add_breadcrumb(
            message="Starting TBS input validation",
            category="strategy.tbs",
            data={
                "has_portfolio": bool(params.get('portfolio')),
                "strategies_count": len(params.get('strategies', []))
            }
        )
        
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
        
        # Track validation metrics
        total_legs = 0
        invalid_legs = 0
        
        for i, strategy in enumerate(strategies):
            strategy_name = strategy.get('strategy_name', f'Strategy_{i+1}')
            
            # Validate legs
            legs = strategy.get('legs', [])
            if not legs:
                logger.warning(f"{strategy_name}: No legs defined")
                capture_message(
                    f"Strategy has no legs: {strategy_name}",
                    level="warning",
                    strategy_index=i
                )
                continue
            
            total_legs += len(legs)
            
            for j, leg in enumerate(legs):
                leg_errors = self._validate_leg(leg, f"{strategy_name}_Leg{j+1}")
                if leg_errors:
                    invalid_legs += 1
                errors.extend(leg_errors)
        
        # Report validation summary
        if errors:
            capture_message(
                f"TBS validation found {len(errors)} errors",
                level="warning",
                error_count=len(errors),
                total_strategies=len(strategies),
                total_legs=total_legs,
                invalid_legs=invalid_legs,
                first_errors=errors[:5]  # First 5 errors
            )
        else:
            add_breadcrumb(
                message="TBS validation passed",
                category="strategy.tbs",
                data={
                    "strategies": len(strategies),
                    "total_legs": total_legs
                }
            )
        
        return errors
    
    def _validate_leg(self, leg: Dict[str, Any], leg_name: str) -> List[str]:
        """Validate a single leg configuration"""
        errors = []
        
        # Add validation breadcrumb
        add_breadcrumb(
            message=f"Validating leg: {leg_name}",
            category="strategy.tbs.validation",
            data={
                "leg_name": leg_name,
                "option_type": leg.get('option_type'),
                "transaction_type": leg.get('transaction_type'),
                "quantity": leg.get('quantity', 0)
            }
        )
        
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
        
        # Track validation issues
        if errors:
            capture_message(
                f"Leg validation failed: {leg_name}",
                level="debug",
                leg_name=leg_name,
                error_count=len(errors),
                errors=errors
            )
        
        return errors