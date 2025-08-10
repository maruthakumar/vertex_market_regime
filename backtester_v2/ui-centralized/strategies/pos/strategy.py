"""
Main Strategy Class for POS (Positional) Strategy
Implements the BaseStrategy interface for multi-leg options strategies
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from datetime import datetime
import os

# Import from parent directory structure
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.base_strategy import BaseStrategy
from core.database.connection_manager import DatabaseManager, get_database_manager
from core.sentry_config import capture_exception, add_breadcrumb, track_errors, capture_message, set_tag, set_context

from .parser import POSParser
from .query_builder import POSQueryBuilder
from .processor import POSProcessor
from .excel_output_generator import POSExcelOutputGenerator
from .models import POSStrategyModel
from .constants import (
    STRATEGY_TYPES,
    DB_COLUMN_MAPPINGS,
    ERROR_MESSAGES,
    MAX_LEGS
)

logger = logging.getLogger(__name__)


class POSStrategy(BaseStrategy):
    """
    Positional (POS) Strategy Implementation
    
    Supports complex multi-leg options strategies including:
    - Iron Condors
    - Iron Flies
    - Calendar Spreads
    - Custom multi-leg strategies
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, 
                 db_manager: Optional[DatabaseManager] = None):
        """
        Initialize POS Strategy
        
        Args:
            config: Optional configuration dictionary
            db_manager: Optional database manager instance
        """
        try:
            set_tag("module", "pos_strategy")
            set_tag("strategy_type", "POS")
            
            super().__init__(config)
            
            # Initialize components
            self.parser = POSParser()
            self.query_builder = POSQueryBuilder(
                table_name=config.get('table_name', 'nifty_option_chain') if config else 'nifty_option_chain'
            )
            self.processor = POSProcessor()
            self.excel_generator = POSExcelOutputGenerator()
            
            # Database manager
            self.db_manager = db_manager or get_database_manager()
            
            # Set strategy metadata
            self.name = "POS_Strategy"
            self.version = "1.0.0"
            
            add_breadcrumb(
                message="POSStrategy initialized",
                category="strategy.pos",
                level="info",
                data={
                    "version": self.version,
                    "table_name": self.query_builder.table_name
                }
            )
            
            # Define required columns
            self.required_columns = [
                'trade_date', 'trade_time', 'index_name', 'expiry_date',
                'strike_price', 'option_type', 'open_price', 'high_price',
                'low_price', 'close_price', 'volume', 'open_interest',
                'implied_volatility', 'underlying_value',
                'delta', 'gamma', 'theta', 'vega', 'rho'
            ]
            
            # Define output columns
            self.output_columns = [
                'position_id', 'trade_date', 'trade_time', 'trade_type',
                'leg_id', 'leg_name', 'option_type', 'position_type',
                'strike_price', 'expiry_date', 'quantity', 'price',
                'premium', 'delta', 'gamma', 'theta', 'vega',
                'underlying_price', 'position_gross_pnl', 'position_net_pnl',
                'transaction_cost', 'slippage'
            ]
            
            # Parsed data storage
            self.parsed_data = None
            self.strategy_model = None
            
            capture_message(
                "POSStrategy initialized successfully",
                level="info",
                module="pos_strategy",
                version=self.version
            )
            
        except Exception as e:
            capture_exception(
                e,
                context="Failed to initialize POSStrategy",
                module="pos_strategy",
                config=str(config)
            )
            raise
        
    @track_errors
    def parse_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse and validate input data for POS strategy
        
        Args:
            input_data: Dictionary containing:
                - portfolio_file: Path to portfolio Excel file
                - strategy_file: Path to strategy Excel file
                - adjustment_file: Optional path to adjustment rules
                - start_date: Optional override start date
                - end_date: Optional override end date
                
        Returns:
            Parsed and validated data ready for processing
            
        Raises:
            ValueError: If input data is invalid
        """
        set_tag("operation", "parse_input")
        
        try:
            logger.info("Parsing POS strategy input data")
            
            set_context("pos_parse_input", {
                "portfolio_file": input_data.get('portfolio_file'),
                "strategy_file": input_data.get('strategy_file'),
                "has_adjustment_file": bool(input_data.get('adjustment_file')),
                "has_date_override": bool(input_data.get('start_date') or input_data.get('end_date'))
            })
            
            add_breadcrumb(
                message="Parsing POS input files",
                category="strategy.pos",
                data={
                    "portfolio_file": input_data.get('portfolio_file'),
                    "strategy_file": input_data.get('strategy_file')
                }
            )
            
            # Validate required files
            if 'portfolio_file' not in input_data:
                error_msg = "portfolio_file is required"
                capture_message(error_msg, level="error", input_data=input_data)
                raise ValueError(error_msg)
            if 'strategy_file' not in input_data:
                error_msg = "strategy_file is required"
                capture_message(error_msg, level="error", input_data=input_data)
                raise ValueError(error_msg)
                
            # Parse files
            try:
                parsed = self.parser.parse_input(
                    portfolio_file=input_data['portfolio_file'],
                    strategy_file=input_data['strategy_file'],
                    adjustment_file=input_data.get('adjustment_file')
                )
                
                add_breadcrumb(
                    message="POS files parsed successfully",
                    category="strategy.pos",
                    data={
                        "has_errors": bool(parsed.get('errors')),
                        "strategy_type": parsed.get('portfolio', {}).get('strategy_type')
                    }
                )
                
            except Exception as e:
                capture_exception(
                    e,
                    context="Failed to parse POS files",
                    portfolio_file=input_data['portfolio_file'],
                    strategy_file=input_data['strategy_file']
                )
                raise
            
            # Check for parsing errors
            if parsed.get('errors'):
                error_msg = "Parsing errors: " + "; ".join(parsed['errors'])
                capture_message(
                    error_msg,
                    level="error",
                    errors=parsed['errors']
                )
                raise ValueError(error_msg)
                
            # Override dates if provided
            if 'start_date' in input_data and input_data['start_date']:
                parsed['portfolio']['start_date'] = pd.to_datetime(input_data['start_date']).date()
            if 'end_date' in input_data and input_data['end_date']:
                parsed['portfolio']['end_date'] = pd.to_datetime(input_data['end_date']).date()
                
            # Store parsed data and model
            self.parsed_data = parsed
            if 'model' in parsed:
                self.strategy_model = parsed['model']
            else:
                # Create model from parsed data
                from .models import POSStrategyModel, POSPortfolioModel, POSLegModel
                
                portfolio = POSPortfolioModel(**parsed['portfolio'])
                legs = [POSLegModel(**leg_data) for leg_data in parsed['strategy']['legs']]
                
                self.strategy_model = POSStrategyModel(
                    portfolio=portfolio,
                    legs=legs,
                    **parsed['strategy'].get('settings', {})
                )
                
            # Log summary
            logger.info(f"Parsed POS strategy: {self.strategy_model.portfolio.strategy_name}")
            logger.info(f"Strategy type: {self.strategy_model.portfolio.strategy_type}")
            logger.info(f"Number of legs: {len(self.strategy_model.legs)}")
            logger.info(f"Date range: {self.strategy_model.portfolio.start_date} to {self.strategy_model.portfolio.end_date}")
            
            # Capture success metrics
            capture_message(
                "POS input parsing completed successfully",
                level="info",
                strategy_name=self.strategy_model.portfolio.strategy_name,
                strategy_type=self.strategy_model.portfolio.strategy_type,
                num_legs=len(self.strategy_model.legs),
                start_date=str(self.strategy_model.portfolio.start_date),
                end_date=str(self.strategy_model.portfolio.end_date)
            )
            
            return parsed
            
        except Exception as e:
            logger.error(f"Error parsing POS input: {str(e)}")
            capture_exception(
                e,
                context="Failed to parse POS input",
                module="pos_strategy"
            )
            raise ValueError(f"Failed to parse input: {str(e)}")
    
    def generate_query(self, params: Dict[str, Any]) -> List[str]:
        """
        Generate SQL queries for POS strategy
        
        Args:
            params: Parameters from parsed input data
            
        Returns:
            List of SQL query strings to execute
        """
        try:
            logger.info("Generating queries for POS strategy")
            
            # Use strategy model if available
            if self.strategy_model:
                queries = self.query_builder.build_queries(self.strategy_model)
            else:
                # Fallback to building from params
                from .models import POSStrategyModel, POSPortfolioModel, POSLegModel
                
                portfolio = POSPortfolioModel(**params['portfolio'])
                legs = [POSLegModel(**leg_data) for leg_data in params['strategy']['legs']]
                
                strategy_model = POSStrategyModel(
                    portfolio=portfolio,
                    legs=legs,
                    **params['strategy'].get('settings', {})
                )
                
                queries = self.query_builder.build_queries(strategy_model)
                
            logger.info(f"Generated {len(queries)} queries for execution")
            
            # Log query summaries
            for i, query in enumerate(queries):
                query_type = "Main" if i == 0 else "Auxiliary"
                logger.debug(f"{query_type} query length: {len(query)} characters")
                
            return queries
            
        except Exception as e:
            logger.error(f"Error generating queries: {str(e)}")
            raise ValueError(f"Query generation failed: {str(e)}")
    
    def process_results(self, results: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw query results into final output format
        
        Args:
            results: Raw results from database query (or list of DataFrames)
            params: Original parameters for context
            
        Returns:
            Processed results ready for output
        """
        try:
            logger.info("Processing POS strategy results")
            
            # Handle both single DataFrame and list of DataFrames
            if isinstance(results, pd.DataFrame):
                results_list = [results]
            else:
                results_list = results
                
            # Use stored strategy model or create from params
            if self.strategy_model:
                strategy_model = self.strategy_model
            else:
                from .models import POSStrategyModel, POSPortfolioModel, POSLegModel
                
                portfolio = POSPortfolioModel(**params['portfolio'])
                legs = [POSLegModel(**leg_data) for leg_data in params['strategy']['legs']]
                
                strategy_model = POSStrategyModel(
                    portfolio=portfolio,
                    legs=legs,
                    **params['strategy'].get('settings', {})
                )
                
            # Process results
            processed = self.processor.process_results(results_list, strategy_model)
            
            # Log summary
            if 'metrics' in processed:
                logger.info(f"Total trades: {processed['metrics'].get('total_trades', 0)}")
                logger.info(f"Total P&L: {processed['metrics'].get('total_pnl', 0):.2f}")
                logger.info(f"Win rate: {processed['metrics'].get('win_rate', 0):.2%}")
                
            return processed
            
        except Exception as e:
            logger.error(f"Error processing results: {str(e)}")
            raise ValueError(f"Result processing failed: {str(e)}")
    
    def generate_excel_output(self, processed_results: Dict[str, Any], 
                             output_path: str, params: Dict[str, Any]) -> str:
        """
        Generate Excel output in legacy format for Strategy Consolidator compatibility
        
        Args:
            processed_results: Results from process_results method
            output_path: Path for output Excel file
            params: Original parameters for context
            
        Returns:
            Path to generated Excel file
        """
        try:
            logger.info(f"Generating POS Excel output: {output_path}")
            
            # Prepare strategy parameters for Excel generator
            strategy_params = {
                'portfolio_name': params.get('portfolio', {}).get('portfolio_name', 'POS_PORTFOLIO'),
                'strategy_name': params.get('strategy', {}).get('strategy_name', 'POS_STRATEGY'),
                'symbol': params.get('portfolio', {}).get('symbol', 'NIFTY'),
                'initial_capital': params.get('portfolio', {}).get('position_size_value', 100000)
            }
            
            # Generate Excel output using the standard format
            from pathlib import Path
            output_file = self.excel_generator.generate_excel_output(
                processed_results, 
                strategy_params, 
                Path(output_path)
            )
            
            logger.info(f"✅ POS Excel output generated successfully: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"❌ Error generating Excel output: {e}")
            capture_exception(
                e,
                context="Failed to generate Excel output",
                module="pos_strategy"
            )
            raise ValueError(f"Excel generation failed: {str(e)}")
    
    def validate_input(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate input data before processing
        
        Args:
            data: Input data to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required fields
        if 'portfolio_file' not in data:
            errors.append("portfolio_file is required")
        elif not os.path.exists(data['portfolio_file']):
            errors.append(f"Portfolio file not found: {data['portfolio_file']}")
            
        if 'strategy_file' not in data:
            errors.append("strategy_file is required")
        elif not os.path.exists(data['strategy_file']):
            errors.append(f"Strategy file not found: {data['strategy_file']}")
            
        # Check optional adjustment file
        if 'adjustment_file' in data and data['adjustment_file']:
            if not os.path.exists(data['adjustment_file']):
                errors.append(f"Adjustment file not found: {data['adjustment_file']}")
                
        # Validate dates if provided
        if 'start_date' in data and 'end_date' in data:
            try:
                start = pd.to_datetime(data['start_date'])
                end = pd.to_datetime(data['end_date'])
                if start >= end:
                    errors.append("start_date must be before end_date")
            except:
                errors.append("Invalid date format")
                
        # If we have parsed data, perform additional validation
        if hasattr(self, 'parsed_data') and self.parsed_data:
            validation_errors = self.parser.validate_parsed_data(self.parsed_data)
            errors.extend(validation_errors)
            
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get comprehensive strategy information
        
        Returns:
            Dictionary containing strategy metadata and configuration
        """
        info = super().get_strategy_info()
        
        # Add POS-specific information
        info.update({
            'strategy_types': list(STRATEGY_TYPES.keys()),
            'max_legs': MAX_LEGS,
            'supports_greeks': True,
            'supports_adjustments': True,
            'supports_multi_expiry': True,
            'features': {
                'multi_leg': True,
                'greeks_calculation': True,
                'dynamic_adjustments': True,
                'market_regime_filter': True,
                'position_sizing': ['FIXED', 'KELLY', 'VOLATILITY_BASED'],
                'strike_selection': ['ATM', 'ITM', 'OTM', 'STRIKE_PRICE', 'DELTA_BASED', 'PERCENTAGE_BASED']
            }
        })
        
        # Add current strategy details if available
        if self.strategy_model:
            info['current_strategy'] = {
                'name': self.strategy_model.portfolio.strategy_name,
                'type': self.strategy_model.portfolio.strategy_type,
                'legs': len(self.strategy_model.legs),
                'start_date': str(self.strategy_model.portfolio.start_date),
                'end_date': str(self.strategy_model.portfolio.end_date),
                'index': self.strategy_model.portfolio.index_name
            }
            
        return info
    
    def supports_distributed(self) -> bool:
        """
        Check if strategy supports distributed processing
        
        Returns:
            True - POS strategies can be distributed by date
        """
        return True
    
    def get_partitioning_key(self) -> Optional[str]:
        """
        Get the key to use for distributed partitioning
        
        Returns:
            'trade_date' for date-based partitioning
        """
        return 'trade_date'
    
    def merge_distributed_results(self, results: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Merge results from distributed processing
        
        Args:
            results: List of DataFrames from different nodes
            
        Returns:
            Merged DataFrame maintaining position integrity
        """
        if not results:
            return pd.DataFrame()
            
        # Concatenate results
        merged = pd.concat(results, ignore_index=True)
        
        # Sort by date and time to maintain chronological order
        if 'trade_date' in merged.columns and 'trade_time' in merged.columns:
            merged = merged.sort_values(['trade_date', 'trade_time'])
            
        # Recalculate position-level metrics if needed
        if 'position_id' in merged.columns:
            # Group by position and recalculate cumulative metrics
            position_groups = merged.groupby('position_id')
            
            # Ensure P&L is calculated correctly across partitions
            for position_id, group in position_groups:
                # Recalculate position P&L if needed
                pass
                
        return merged
    
    def get_performance_metrics(self, results: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics
        
        Args:
            results: Processed results DataFrame
            
        Returns:
            Dictionary of performance metrics including Greeks
        """
        metrics = super().get_performance_metrics(results)
        
        # Add POS-specific metrics
        if not results.empty:
            # Position-level metrics
            if 'position_id' in results.columns:
                unique_positions = results['position_id'].nunique()
                metrics['total_positions'] = unique_positions
                
            # Greek metrics
            greek_columns = ['delta', 'gamma', 'theta', 'vega']
            for greek in greek_columns:
                if greek in results.columns:
                    metrics[f'avg_{greek}'] = results[greek].mean()
                    metrics[f'max_{greek}'] = results[greek].max()
                    metrics[f'min_{greek}'] = results[greek].min()
                    
            # Strategy-specific metrics
            if hasattr(self, 'strategy_model') and self.strategy_model:
                metrics['strategy_type'] = self.strategy_model.portfolio.strategy_type
                metrics['total_legs'] = len(self.strategy_model.legs)
                
            # Adjustment metrics
            if 'adjustment_count' in results.columns:
                metrics['total_adjustments'] = results['adjustment_count'].sum()
                
        return metrics