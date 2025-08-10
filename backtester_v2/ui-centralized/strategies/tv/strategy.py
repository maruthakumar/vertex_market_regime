#!/usr/bin/env python3
"""
TV Strategy - TradingView signal-based strategy implementation
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
from .parser import TVParser
from .signal_processor import SignalProcessor
from .query_builder import TVQueryBuilder
from .processor import TVProcessor

# Import TBS parser for loading strategy files
from strategies.tbs.parser import TBSParser

logger = logging.getLogger(__name__)


class TVStrategy(BaseStrategy):
    """TradingView signal-based strategy implementation"""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """Initialize TV strategy"""
        try:
            set_tag("module", "tv_strategy")
            set_tag("strategy_type", "TV")
            
            super().__init__()
            self.parser = TVParser()
            self.signal_processor = SignalProcessor()
            self.query_builder = TVQueryBuilder()
            self.processor = TVProcessor()
            self.tbs_parser = TBSParser()  # For loading portfolio/strategy files
            self.db_manager = db_manager or get_database_manager()
            
            add_breadcrumb(
                message="TVStrategy initialized",
                category="strategy.tv",
                level="info"
            )
            
            capture_message(
                "TVStrategy initialized successfully",
                level="info",
                module="tv_strategy"
            )
            
        except Exception as e:
            capture_exception(
                e,
                context="Failed to initialize TVStrategy",
                module="tv_strategy"
            )
            raise
        
    @track_errors
    def parse_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse TV input files
        
        Args:
            input_data: Dictionary containing paths to input files
                - tv_excel: Path to TV settings Excel file
                
        Returns:
            Parsed TV parameters with signals and strategies
        """
        set_tag("operation", "parse_input")
        
        tv_excel = input_data.get('tv_excel')
        
        set_context("tv_parse_input", {
            "tv_excel": tv_excel,
            "has_tv_excel": bool(tv_excel)
        })
        
        add_breadcrumb(
            message="Parsing TV input files",
            category="strategy.tv",
            data={"tv_excel": tv_excel}
        )
        
        if not tv_excel:
            error_msg = "TV Excel path is required"
            capture_message(
                error_msg,
                level="error",
                input_data=input_data
            )
            raise ValueError(error_msg)
        
        logger.info(f"Parsing TV input files: {tv_excel}")
        
        # Parse TV settings
        try:
            tv_data = self.parser.parse_tv_settings(tv_excel)
            add_breadcrumb(
                message="TV settings parsed successfully",
                category="strategy.tv",
                data={"settings_count": len(tv_data.get('settings', []))}
            )
        except Exception as e:
            capture_exception(
                e,
                context="Failed to parse TV settings",
                tv_excel=tv_excel
            )
            raise
        
        # Process each enabled TV setting
        all_processed_data = []
        processing_errors = []
        
        for idx, tv_setting in enumerate(tv_data['settings']):
            try:
                setting_name = tv_setting.get('name', f'Setting_{idx+1}')
                add_breadcrumb(
                    message=f"Processing TV setting: {setting_name}",
                    category="strategy.tv",
                    data={"setting_index": idx, "setting_name": setting_name}
                )
                
                # Process this TV setting
                processed = self._process_tv_setting(tv_setting, tv_excel)
                all_processed_data.append(processed)
                
            except Exception as e:
                error_msg = f"Failed to process TV setting '{tv_setting.get('name')}': {e}"
                logger.error(error_msg)
                processing_errors.append(error_msg)
                capture_exception(
                    e,
                    context="Failed to process TV setting",
                    setting_name=tv_setting.get('name'),
                    setting_index=idx
                )
                continue
        
        # Report processing summary
        if processing_errors:
            capture_message(
                f"Some TV settings failed to process: {len(processing_errors)} errors",
                level="warning",
                errors=processing_errors[:5]  # First 5 errors
            )
        
        if not all_processed_data:
            error_msg = "No TV settings could be processed successfully"
            capture_message(
                error_msg,
                level="error",
                total_settings=len(tv_data.get('settings', [])),
                processing_errors=processing_errors
            )
            raise ValueError(error_msg)
        
        # For now, return the first processed setting
        # In future, could support multiple TV backtests
        result = all_processed_data[0]
        
        # Validate the parsed data
        validation_errors = self.validate_input(result)
        if validation_errors:
            error_msg = f"Input validation failed: {validation_errors}"
            capture_message(
                error_msg,
                level="error",
                validation_errors=validation_errors,
                signals_count=len(result.get('signals', []))
            )
            raise ValueError(error_msg)
        
        # Success tracking
        capture_message(
            "TV input parsing completed successfully",
            level="info",
            settings_processed=len(all_processed_data),
            signals_count=len(result.get('signals', [])),
            had_errors=len(processing_errors) > 0
        )
        
        return result
    
    @track_errors
    def _process_tv_setting(self, tv_setting: Dict[str, Any], tv_excel_path: str) -> Dict[str, Any]:
        """Process a single TV setting"""
        setting_name = tv_setting.get('name', 'Unknown')
        
        add_breadcrumb(
            message=f"Processing TV setting: {setting_name}",
            category="strategy.tv.processing",
            data={"setting_name": setting_name}
        )
        
        # Load and parse signals
        signal_file = tv_setting.get('signal_file_path')
        if not signal_file:
            error_msg = f"Signal file path missing for TV setting '{setting_name}'"
            capture_message(error_msg, level="error", tv_setting=tv_setting)
            raise ValueError(error_msg)
        
        # Resolve signal file path
        signal_file = self._resolve_file_path(signal_file, tv_excel_path)
        
        # Parse signals
        try:
            date_format = tv_setting.get('signal_date_format', '%Y%m%d %H%M%S')
            raw_signals = self.parser.parse_signals(signal_file, date_format)
            add_breadcrumb(
                message=f"Parsed {len(raw_signals)} signals from {signal_file}",
                category="strategy.tv.signals",
                data={"signal_count": len(raw_signals), "date_format": date_format}
            )
        except Exception as e:
            capture_exception(
                e,
                context="Failed to parse signals",
                signal_file=signal_file,
                date_format=date_format
            )
            raise
        
        # Process signals
        try:
            processed_signals = self.signal_processor.process_signals(raw_signals, tv_setting)
            add_breadcrumb(
                message=f"Processed {len(processed_signals)} signals",
                category="strategy.tv.signals"
            )
        except Exception as e:
            capture_exception(
                e,
                context="Failed to process signals",
                raw_signals_count=len(raw_signals)
            )
            raise
        
        # Load portfolio and strategy files for each signal
        enriched_signals = []
        enrichment_errors = 0
        
        for idx, signal in enumerate(processed_signals):
            try:
                enriched = self._enrich_signal_with_strategy(signal, tv_setting, tv_excel_path)
                if enriched:
                    enriched_signals.append(enriched)
            except Exception as e:
                enrichment_errors += 1
                capture_exception(
                    e,
                    context="Failed to enrich signal",
                    signal_index=idx,
                    trade_no=signal.get('trade_no')
                )
        
        # Report enrichment summary
        if enrichment_errors > 0:
            capture_message(
                f"Some signals failed enrichment: {enrichment_errors} errors",
                level="warning",
                total_signals=len(processed_signals),
                enriched_signals=len(enriched_signals)
            )
        
        capture_message(
            f"TV setting processed successfully: {setting_name}",
            level="info",
            raw_signals=len(raw_signals),
            processed_signals=len(processed_signals),
            enriched_signals=len(enriched_signals),
            enrichment_errors=enrichment_errors
        )
        
        return {
            'tv_settings': tv_setting,
            'signals': enriched_signals,
            'source_files': {
                'tv_settings': tv_excel_path,
                'signal_file': signal_file
            }
        }
    
    def _enrich_signal_with_strategy(
        self, 
        signal: Dict[str, Any], 
        tv_setting: Dict[str, Any],
        tv_excel_path: str
    ) -> Optional[Dict[str, Any]]:
        """Load portfolio and strategy for a signal"""
        
        # Get portfolio file based on signal direction
        portfolio_file = signal.get('portfolio_file')
        if not portfolio_file:
            logger.warning(f"No portfolio file for signal {signal['trade_no']}")
            return None
        
        # Resolve portfolio file path
        portfolio_file = self._resolve_file_path(portfolio_file, tv_excel_path)
        
        try:
            # Load portfolio
            portfolio_data = self.tbs_parser.parse_portfolio_excel(portfolio_file)
            
            # Load TBS strategies referenced in portfolio
            strategies = []
            for portfolio_strategy in portfolio_data.get('strategies', []):
                strategy_file = portfolio_strategy.get('strategy_excel_file_path')
                if strategy_file:
                    strategy_file = self._resolve_file_path(strategy_file, portfolio_file)
                    if os.path.exists(strategy_file):
                        strategy_data = self.tbs_parser.parse_multi_leg_excel(strategy_file)
                        strategies.extend(strategy_data.get('strategies', []))
            
            # Use first strategy with legs
            strategy_with_legs = None
            for strategy in strategies:
                if strategy.get('legs'):
                    strategy_with_legs = strategy
                    break
            
            if not strategy_with_legs:
                logger.warning(f"No strategy with legs found for signal {signal['trade_no']}")
                return None
            
            # Merge signal with portfolio and strategy
            signal['portfolio'] = portfolio_data['portfolio']
            signal['strategy'] = strategy_with_legs
            
            return signal
            
        except Exception as e:
            logger.error(f"Failed to load portfolio/strategy for signal {signal['trade_no']}: {e}")
            return None
    
    def _resolve_file_path(self, file_path: str, reference_path: str) -> str:
        """Resolve file path relative to reference"""
        
        # If absolute path exists, use it
        if os.path.exists(file_path):
            return file_path
        
        # Try relative to reference file directory
        ref_dir = os.path.dirname(reference_path)
        relative_path = os.path.join(ref_dir, os.path.basename(file_path))
        if os.path.exists(relative_path):
            return relative_path
        
        # Try in input_sheets/tv directory
        tv_dir = '/srv/samba/shared/bt/backtester_stable/BTRUN/input_sheets/tv'
        tv_path = os.path.join(tv_dir, os.path.basename(file_path))
        if os.path.exists(tv_path):
            return tv_path
        
        # Return original path
        return file_path
    
    @track_errors
    def generate_query(self, params: Dict[str, Any]) -> List[str]:
        """Generate SQL queries for TV execution"""
        set_tag("operation", "generate_query")
        
        logger.info("Generating TV queries")
        
        signals = params.get('signals', [])
        tv_settings = params.get('tv_settings', {})
        
        set_context("tv_query_generation", {
            "total_signals": len(signals),
            "tv_setting_name": tv_settings.get('name'),
            "start_date": tv_settings.get('start_date'),
            "end_date": tv_settings.get('end_date')
        })
        
        add_breadcrumb(
            message="Starting TV query generation",
            category="strategy.tv",
            data={"signal_count": len(signals)}
        )
        
        if not signals:
            warning_msg = "No signals to process"
            logger.warning(warning_msg)
            capture_message(warning_msg, level="warning")
            return []
        
        queries = []
        skipped_signals = []
        
        # Generate query for each signal
        for idx, signal in enumerate(signals):
            trade_no = signal.get('trade_no', f'Signal_{idx+1}')
            
            if not signal.get('portfolio') or not signal.get('strategy'):
                skip_msg = f"Skipping signal {trade_no} - missing portfolio/strategy"
                logger.warning(skip_msg)
                skipped_signals.append(trade_no)
                add_breadcrumb(
                    message=skip_msg,
                    category="strategy.tv.query",
                    data={"trade_no": trade_no, "has_portfolio": bool(signal.get('portfolio')), 
                          "has_strategy": bool(signal.get('strategy'))}
                )
                continue
            
            try:
                query = self.query_builder.build_signal_query(
                    signal,
                    signal['portfolio'],
                    signal['strategy'],
                    tv_settings
                )
                
                if query:
                    queries.append(query)
                    add_breadcrumb(
                        message=f"Generated query for signal {trade_no}",
                        category="strategy.tv.query",
                        data={"trade_no": trade_no, "query_length": len(query)}
                    )
                    
            except Exception as e:
                capture_exception(
                    e,
                    context="Failed to generate query for signal",
                    trade_no=trade_no,
                    signal_index=idx
                )
                skipped_signals.append(trade_no)
        
        # Report generation summary
        if skipped_signals:
            capture_message(
                f"Skipped {len(skipped_signals)} signals during query generation",
                level="warning",
                skipped_signals=skipped_signals[:10]  # First 10
            )
        
        capture_message(
            "TV query generation completed",
            level="info",
            total_signals=len(signals),
            queries_generated=len(queries),
            signals_skipped=len(skipped_signals)
        )
        
        logger.info(f"Generated {len(queries)} queries for TV execution")
        return queries
    
    @track_errors
    def process_results(self, results: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process query results into backtest output"""
        set_tag("operation", "process_results")
        
        logger.info("Processing TV results")
        
        set_context("tv_process_results", {
            "results_rows": len(results) if results is not None else 0,
            "results_columns": len(results.columns) if results is not None else 0,
            "signals_count": len(params.get('signals', [])),
            "has_results": results is not None and not results.empty
        })
        
        add_breadcrumb(
            message="Starting TV results processing",
            category="strategy.tv",
            data={
                "rows": len(results) if results is not None else 0,
                "columns": len(results.columns) if results is not None else 0
            }
        )
        
        # Prepare parameters for processor
        tv_params = {
            'name': params.get('tv_settings', {}).get('name', 'TV_Backtest'),
            'start_date': params.get('tv_settings', {}).get('start_date'),
            'end_date': params.get('tv_settings', {}).get('end_date'),
            'signals': params.get('signals', [])
        }
        
        try:
            # Process results
            output = self.processor.process_tv_results([results], tv_params)
            
            # Track output metrics
            if output:
                metrics = output.get('metrics', {})
                add_breadcrumb(
                    message="TV results processed successfully",
                    category="strategy.tv",
                    data={
                        "total_trades": metrics.get('total_trades', 0),
                        "winning_trades": metrics.get('winning_trades', 0),
                        "losing_trades": metrics.get('losing_trades', 0),
                        "total_pnl": metrics.get('total_pnl', 0)
                    }
                )
                
                capture_message(
                    "TV results processing completed",
                    level="info",
                    total_trades=metrics.get('total_trades', 0),
                    total_pnl=metrics.get('total_pnl', 0),
                    win_rate=metrics.get('win_rate', 0),
                    signals_processed=len(tv_params['signals'])
                )
            
            return output
            
        except Exception as e:
            capture_exception(
                e,
                context="Failed to process TV results",
                results_shape=(len(results), len(results.columns)) if results is not None else None,
                signals_count=len(params.get('signals', []))
            )
            raise
    
    @track_errors
    def validate_input(self, params: Dict[str, Any]) -> List[str]:
        """Validate TV input parameters"""
        set_tag("operation", "validate_input")
        
        errors = []
        
        add_breadcrumb(
            message="Starting TV input validation",
            category="strategy.tv",
            data={
                "has_tv_settings": bool(params.get('tv_settings')),
                "signals_count": len(params.get('signals', []))
            }
        )
        
        # Validate TV settings
        tv_settings = params.get('tv_settings', {})
        if not tv_settings:
            errors.append("TV settings are missing")
        else:
            if not tv_settings.get('start_date'):
                errors.append("Start date is required")
            if not tv_settings.get('end_date'):
                errors.append("End date is required")
            if tv_settings.get('start_date') and tv_settings.get('end_date'):
                if tv_settings['start_date'] > tv_settings['end_date']:
                    errors.append("Start date must be before end date")
        
        # Validate signals
        signals = params.get('signals', [])
        if not signals:
            errors.append("At least one signal is required")
        
        # Validate enriched signals
        valid_signals = 0
        invalid_signals = []
        
        for i, signal in enumerate(signals):
            trade_no = signal.get('trade_no', f'Signal_{i+1}')
            
            if signal.get('portfolio') and signal.get('strategy'):
                if signal['strategy'].get('legs'):
                    valid_signals += 1
                else:
                    invalid_signals.append(f"{trade_no}: No legs in strategy")
            else:
                if not signal.get('portfolio'):
                    invalid_signals.append(f"{trade_no}: Missing portfolio")
                if not signal.get('strategy'):
                    invalid_signals.append(f"{trade_no}: Missing strategy")
        
        if valid_signals == 0:
            errors.append("No valid signals with portfolio and strategy found")
        
        # Report validation summary
        if errors or invalid_signals:
            capture_message(
                f"TV validation found issues",
                level="warning",
                error_count=len(errors),
                total_signals=len(signals),
                valid_signals=valid_signals,
                invalid_signal_count=len(invalid_signals),
                errors=errors,
                invalid_signals=invalid_signals[:5]  # First 5
            )
        else:
            add_breadcrumb(
                message="TV validation passed",
                category="strategy.tv",
                data={
                    "signals": len(signals),
                    "valid_signals": valid_signals
                }
            )
        
        return errors