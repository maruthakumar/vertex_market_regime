#!/usr/bin/env python3
"""
OI Processor - Processes OI strategy signals and manages execution flow
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import date, time, datetime, timedelta
import logging
import pandas as pd

from .models import (
    OISettingModel, OILegModel, OIRanking, OISignal, 
    ProcessedOISignal, OIMethod, COIBasedOn
)
from .oi_analyzer import OIAnalyzer
from .query_builder import OIQueryBuilder

logger = logging.getLogger(__name__)


class OIProcessor:
    """Processes OI strategy signals and manages execution"""
    
    def __init__(self, db_connection):
        """
        Initialize OI processor
        
        Args:
            db_connection: HeavyDB connection
        """
        self.conn = db_connection
        self.oi_analyzer = OIAnalyzer(db_connection)
        self.query_builder = OIQueryBuilder()
        
        # Track state
        self._active_positions: Dict[str, Dict[str, Any]] = {}
        self._executed_trades: List[Dict[str, Any]] = []
        self._oi_rankings: Optional[OIRanking] = None
        self._monitoring_strikes: List[int] = []
    
    def process_oi_strategy(
        self,
        settings: OISettingModel,
        trade_date: date
    ) -> Dict[str, Any]:
        """
        Process complete OI strategy for a given date
        
        Args:
            settings: OI strategy settings
            trade_date: Date to process
            
        Returns:
            Dictionary with execution results
        """
        try:
            # Step 1: Calculate OI/COI rankings at strike selection time
            rankings = self._calculate_rankings(settings, trade_date)
            if not rankings:
                logger.warning(f"No valid OI rankings for {trade_date}")
                return self._create_empty_result(settings.strategy_name, trade_date)
            
            # Store rankings for later use
            self._oi_rankings = rankings
            
            # Step 2: Generate initial entry signals based on OI rankings
            entry_signals = self._generate_initial_signals(settings, rankings, trade_date)
            if not entry_signals:
                logger.info(f"No entry signals generated for {trade_date}")
                return self._create_empty_result(settings.strategy_name, trade_date)
            
            # Step 3: Execute initial trades
            execution_results = self._execute_trades(settings, entry_signals, trade_date)
            
            # Step 4: Monitor for position changes and dynamic switching
            monitoring_results = self._monitor_positions(settings, execution_results, trade_date)
            
            # Step 5: Handle exits (time-based, SL/TP, or OI rank changes)
            exit_results = self._handle_exits(settings, execution_results + monitoring_results, trade_date)
            
            # Step 6: Calculate final P&L
            final_results = self._calculate_final_results(
                settings, execution_results + monitoring_results, exit_results, trade_date
            )
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error processing OI strategy: {e}")
            return self._create_error_result(settings.strategy_name, trade_date, str(e))
    
    def _calculate_rankings(
        self,
        settings: OISettingModel,
        trade_date: date
    ) -> Optional[OIRanking]:
        """Calculate OI or COI rankings based on strategy settings"""
        
        # Determine if this is a COI-based strategy
        uses_coi = any(leg.strike_method.startswith('MAXCOI_') for leg in settings.legs)
        
        if uses_coi:
            # Calculate COI rankings
            return self.oi_analyzer.calculate_coi_rankings(
                index=settings.index,
                trade_date=trade_date,
                analysis_time=settings.strike_selection_time,
                coi_based_on=settings.coi_based_on,
                strike_count=settings.strike_count,
                underlying_type=settings.underlying
            )
        else:
            # Calculate standard OI rankings
            return self.oi_analyzer.calculate_oi_rankings(
                index=settings.index,
                trade_date=trade_date,
                analysis_time=settings.strike_selection_time,
                strike_count=settings.strike_count,
                underlying_type=settings.underlying
            )
    
    def _generate_initial_signals(
        self,
        settings: OISettingModel,
        rankings: OIRanking,
        trade_date: date
    ) -> List[ProcessedOISignal]:
        """Generate initial entry signals based on OI rankings"""
        
        processed_signals = []
        
        for leg in settings.legs:
            # Check if this leg uses OI-based selection
            if leg.strike_method.startswith(('MAXOI_', 'MAXCOI_')):
                signal = self._generate_oi_signal(leg, rankings, trade_date, settings)
                if signal:
                    processed_signals.append(signal)
            else:
                # Handle non-OI methods (ATM, ITM, OTM, FIXED)
                signal = self._generate_non_oi_signal(leg, trade_date, settings)
                if signal:
                    processed_signals.append(signal)
        
        return processed_signals
    
    def _generate_oi_signal(
        self,
        leg: OILegModel,
        rankings: OIRanking,
        trade_date: date,
        settings: OISettingModel
    ) -> Optional[ProcessedOISignal]:
        """Generate signal for OI-based leg"""
        
        # Extract rank from strike method (e.g., MAXOI_1 -> rank 1)
        try:
            rank = int(leg.strike_method.split('_')[1])
        except (IndexError, ValueError):
            logger.error(f"Invalid OI strike method: {leg.strike_method}")
            return None
        
        # Get strike for this rank and instrument type
        strike = rankings.get_strike_for_rank(rank, leg.instrument)
        if strike is None:
            logger.warning(f"No strike available for rank {rank} {leg.instrument}")
            return None
        
        # Get OI value for validation
        oi_value = rankings.get_oi_for_rank(rank, leg.instrument)
        if oi_value is None or oi_value < leg.oi_threshold:
            logger.warning(f"OI threshold not met: {oi_value} < {leg.oi_threshold}")
            return None
        
        # Get underlying price at signal time
        underlying_price = self._get_underlying_price(
            settings.index, trade_date, settings.start_time, settings.underlying
        )
        
        if underlying_price is None:
            logger.warning(f"Could not get underlying price for signal generation")
            return None
        
        return ProcessedOISignal(
            entrydate=trade_date,
            entrytime=settings.start_time,
            exitdate=trade_date,
            exittime=settings.end_time,
            lots=leg.lots,
            leg_id=leg.leg_id,
            instrument=leg.instrument,
            strike=strike,
            oi_rank=rank,
            oi_value=oi_value,
            selection_method=leg.strike_method,
            underlying_at_entry=underlying_price
        )
    
    def _generate_non_oi_signal(
        self,
        leg: OILegModel,
        trade_date: date,
        settings: OISettingModel
    ) -> Optional[ProcessedOISignal]:
        """Generate signal for non-OI based leg (ATM, ITM, OTM, FIXED)"""
        
        # Get ATM strike for reference
        atm_strike = self._get_atm_strike(
            settings.index, trade_date, settings.start_time
        )
        
        if atm_strike is None:
            logger.warning("Could not get ATM strike for non-OI signal")
            return None
        
        # Calculate strike based on method
        if leg.strike_method == 'ATM':
            strike = atm_strike
        elif leg.strike_method == 'FIXED':
            strike = int(leg.strike_value)
        elif leg.strike_method.startswith('ITM'):
            steps = int(leg.strike_method[3:]) if len(leg.strike_method) > 3 else 1
            if leg.instrument == 'CE':
                strike = atm_strike - (steps * 50)  # Assuming 50 point step
            else:  # PE
                strike = atm_strike + (steps * 50)
        elif leg.strike_method.startswith('OTM'):
            steps = int(leg.strike_method[3:]) if len(leg.strike_method) > 3 else 1
            if leg.instrument == 'CE':
                strike = atm_strike + (steps * 50)
            else:  # PE
                strike = atm_strike - (steps * 50)
        else:
            logger.warning(f"Unknown strike method: {leg.strike_method}")
            return None
        
        # Get underlying price
        underlying_price = self._get_underlying_price(
            settings.index, trade_date, settings.start_time, settings.underlying
        )
        
        return ProcessedOISignal(
            entrydate=trade_date,
            entrytime=settings.start_time,
            exitdate=trade_date,
            exittime=settings.end_time,
            lots=leg.lots,
            leg_id=leg.leg_id,
            instrument=leg.instrument,
            strike=strike,
            oi_rank=0,  # Not applicable for non-OI methods
            oi_value=0,  # Not applicable
            selection_method=leg.strike_method,
            underlying_at_entry=underlying_price or 0
        )
    
    def _execute_trades(
        self,
        settings: OISettingModel,
        entry_signals: List[ProcessedOISignal],
        trade_date: date
    ) -> List[Dict[str, Any]]:
        """Execute trades based on entry signals"""
        
        execution_results = []
        
        for signal in entry_signals:
            # Build signal data for query
            signal_data = {
                'leg_id': signal.leg_id,
                'instrument': signal.instrument,
                'strike': signal.strike,
                'signal_time': signal.entrytime,
                'lots': signal.lots,
                'transaction': self._get_leg_transaction(settings, signal.leg_id),
                'expiry': self._get_leg_expiry(settings, signal.leg_id),
                'selection_method': signal.selection_method,
                'oi_rank': signal.oi_rank,
                'oi_value': signal.oi_value
            }
            
            # Build and execute query
            query = self.query_builder.build_oi_entry_query(
                strategy=settings.__dict__,
                signals=[signal_data],
                trade_date=trade_date
            )
            
            try:
                cursor = self.conn.execute(query)
                results = cursor.fetchall()
                
                for row in results:
                    execution = self._parse_execution_result(row, signal)
                    execution_results.append(execution)
                    
                    # Track active position
                    position_id = f"{signal.entrydate}_{execution['leg_id']}_{execution['strike']}"
                    self._active_positions[position_id] = execution
                    
                    # Track strikes for monitoring
                    if execution['strike'] not in self._monitoring_strikes:
                        self._monitoring_strikes.append(execution['strike'])
                    
            except Exception as e:
                logger.error(f"Error executing OI trades: {e}")
        
        return execution_results
    
    def _monitor_positions(
        self,
        settings: OISettingModel,
        execution_results: List[Dict[str, Any]],
        trade_date: date
    ) -> List[Dict[str, Any]]:
        """Monitor positions for OI rank changes and dynamic switching"""
        
        # For now, implement basic monitoring
        # In a full implementation, this would check OI rankings periodically
        # and switch strikes if ranks change significantly
        
        monitoring_results = []
        
        # This is where dynamic OI switching logic would go
        # For the initial implementation, we'll keep it simple
        
        return monitoring_results
    
    def _handle_exits(
        self,
        settings: OISettingModel,
        all_positions: List[Dict[str, Any]],
        trade_date: date
    ) -> List[Dict[str, Any]]:
        """Handle position exits"""
        
        exit_results = []
        
        if self._active_positions:
            # Build exit query
            positions = list(self._active_positions.values())
            exit_query = self.query_builder.build_exit_query(
                strategy=settings.__dict__,
                positions=positions,
                exit_time=settings.end_time,
                trade_date=trade_date
            )
            
            try:
                cursor = self.conn.execute(exit_query)
                results = cursor.fetchall()
                
                for row in results:
                    exit_result = self._parse_exit_result(row)
                    exit_results.append(exit_result)
                    
                    # Remove from active positions
                    position_id = exit_result['position_id']
                    if position_id in self._active_positions:
                        del self._active_positions[position_id]
                        
            except Exception as e:
                logger.error(f"Error executing exits: {e}")
        
        return exit_results
    
    def _calculate_final_results(
        self,
        settings: OISettingModel,
        execution_results: List[Dict[str, Any]],
        exit_results: List[Dict[str, Any]],
        trade_date: date
    ) -> Dict[str, Any]:
        """Calculate final P&L and format results"""
        
        # Match entries and exits
        trades = []
        exit_map = {exit['position_id']: exit for exit in exit_results}
        
        for entry in execution_results:
            position_id = f"{entry['entry_date']}_{entry['leg_id']}_{entry['strike']}"
            exit = exit_map.get(position_id)
            
            if exit:
                # Calculate P&L
                if entry['transaction_type'] == 'BUY':
                    pnl = (exit['exit_price'] - entry['entry_price']) * entry['lots'] * entry['lot_size']
                else:  # SELL
                    pnl = (entry['entry_price'] - exit['exit_price']) * entry['lots'] * entry['lot_size']
                
                trade = {
                    'strategy_name': settings.strategy_name,
                    'trade_date': trade_date,
                    'leg_id': entry['leg_id'],
                    'instrument': entry['instrument'],
                    'transaction': entry['transaction_type'],
                    'strike': entry['strike'],
                    'expiry': entry['expiry_date'],
                    'entry_time': entry['entry_time'],
                    'entry_price': entry['entry_price'],
                    'exit_time': exit.get('exit_time', settings.end_time),
                    'exit_price': exit.get('exit_price', 0),
                    'lots': entry['lots'],
                    'pnl': pnl,
                    'selection_method': entry['selection_method'],
                    'oi_rank': entry['oi_rank'],
                    'oi_value': entry['oi_value']
                }
                trades.append(trade)
        
        # Calculate totals
        total_pnl = sum(trade['pnl'] for trade in trades)
        
        return {
            'strategy_name': settings.strategy_name,
            'trade_date': trade_date,
            'trades': trades,
            'total_pnl': total_pnl,
            'trade_count': len(trades),
            'status': 'completed'
        }
    
    def _get_underlying_price(
        self,
        index: str,
        trade_date: date,
        target_time: time,
        underlying_type: str
    ) -> Optional[float]:
        """Get underlying price at specific time"""
        
        table_name = self.oi_analyzer._get_table_name(index)
        price_column = 'spot' if underlying_type == 'SPOT' else 'fut_close'
        
        query = f"""
        SELECT {price_column}
        FROM {table_name}
        WHERE trade_date = DATE '{trade_date}'
            AND trade_time = TIME '{target_time}'
            AND {price_column} > 0
        ORDER BY trade_time DESC
        LIMIT 1
        """
        
        try:
            cursor = self.conn.execute(query)
            result = cursor.fetchone()
            
            if result and result[0] is not None:
                return float(result[0])
        except Exception as e:
            logger.error(f"Error getting underlying price: {e}")
        
        return None
    
    def _get_atm_strike(self, index: str, trade_date: date, target_time: time) -> Optional[int]:
        """Get ATM strike at specific time"""
        
        table_name = self.oi_analyzer._get_table_name(index)
        
        query = f"""
        SELECT atm_strike
        FROM {table_name}
        WHERE trade_date = DATE '{trade_date}'
            AND trade_time = TIME '{target_time}'
        LIMIT 1
        """
        
        try:
            cursor = self.conn.execute(query)
            result = cursor.fetchone()
            
            if result and result[0] is not None:
                return int(result[0])
        except Exception as e:
            logger.error(f"Error getting ATM strike: {e}")
        
        return None
    
    def _get_leg_transaction(self, settings: OISettingModel, leg_id: str) -> str:
        """Get transaction type for leg"""
        for leg in settings.legs:
            if leg.leg_id == leg_id:
                return leg.transaction
        return 'BUY'
    
    def _get_leg_expiry(self, settings: OISettingModel, leg_id: str) -> str:
        """Get expiry rule for leg"""
        for leg in settings.legs:
            if leg.leg_id == leg_id:
                return leg.expiry
        return 'CW'
    
    def _parse_execution_result(self, row: Any, signal: ProcessedOISignal) -> Dict[str, Any]:
        """Parse execution result from database row"""
        
        return {
            'leg_id': row[0],
            'entry_time': row[1],
            'underlying_at_entry': float(row[2]),
            'strike': int(row[3]),
            'symbol': row[4],
            'entry_price': float(row[5]),
            'open_interest': int(row[6]),
            'expiry_date': row[7],
            'instrument': row[8],
            'transaction_type': row[9],
            'lots': int(row[10]),
            'selection_method': row[11],
            'oi_rank': int(row[12]),
            'oi_value': float(row[13]),
            'lot_size': 50,  # Default lot size
            'entry_date': signal.entrydate
        }
    
    def _parse_exit_result(self, row: Any) -> Dict[str, Any]:
        """Parse exit result from database row"""
        
        return {
            'position_id': row[0],
            'leg_id': row[1],
            'exit_price': float(row[2]),
            'exit_oi': float(row[3]),
            'underlying_at_exit': float(row[4])
        }
    
    def _create_empty_result(self, strategy_name: str, trade_date: date) -> Dict[str, Any]:
        """Create empty result when no trades executed"""
        
        return {
            'strategy_name': strategy_name,
            'trade_date': trade_date,
            'trades': [],
            'total_pnl': 0,
            'trade_count': 0,
            'status': 'no_trades'
        }
    
    def _create_error_result(self, strategy_name: str, trade_date: date, error: str) -> Dict[str, Any]:
        """Create error result"""
        
        return {
            'strategy_name': strategy_name,
            'trade_date': trade_date,
            'trades': [],
            'total_pnl': 0,
            'trade_count': 0,
            'status': 'error',
            'error': error
        }
    
    def clear_state(self):
        """Clear processor state"""
        self._active_positions.clear()
        self._executed_trades.clear()
        self._oi_rankings = None
        self._monitoring_strikes.clear()
        self.oi_analyzer.clear_cache()