#!/usr/bin/env python3
"""
ORB Processor - Processes ORB signals and manages execution flow
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import date, time, datetime, timedelta
import logging
import pandas as pd

from .models import (
    ORBSettingModel, ORBLegModel, ORBRange, ORBSignal, 
    ProcessedORBSignal, ORBBreakoutType, ORBSignalDirection
)
from .range_calculator import RangeCalculator
from .signal_generator import SignalGenerator, BreakoutType
from .query_builder import ORBQueryBuilder

logger = logging.getLogger(__name__)


class ORBProcessor:
    """Processes ORB strategy signals and manages execution"""
    
    def __init__(self, db_connection):
        """
        Initialize ORB processor
        
        Args:
            db_connection: HeavyDB connection
        """
        self.conn = db_connection
        self.range_calculator = RangeCalculator()
        self.signal_generator = SignalGenerator()
        self.query_builder = ORBQueryBuilder()
        
        # Track state
        self._active_positions: Dict[str, Dict[str, Any]] = {}
        self._executed_trades: List[Dict[str, Any]] = []
        self._reentry_counts: Dict[str, int] = {}
    
    def process_orb_strategy(
        self,
        settings: ORBSettingModel,
        trade_date: date
    ) -> Dict[str, Any]:
        """
        Process complete ORB strategy for a given date
        
        Args:
            settings: ORB strategy settings
            trade_date: Date to process
            
        Returns:
            Dictionary with execution results
        """
        try:
            # Step 1: Calculate opening range
            orb_range = self._calculate_opening_range(settings, trade_date)
            if not orb_range:
                logger.warning(f"No valid opening range for {trade_date}")
                return self._create_empty_result(settings.strategy_name, trade_date)
            
            # Step 2: Detect breakout
            breakout_signal = self._detect_breakout(settings, trade_date, orb_range)
            if not breakout_signal:
                logger.info(f"No breakout detected for {trade_date}")
                return self._create_empty_result(settings.strategy_name, trade_date)
            
            # Step 3: Generate entry signals
            entry_signals = self._generate_entry_signals(settings, breakout_signal, trade_date)
            if not entry_signals:
                logger.warning(f"No entry signals generated for breakout")
                return self._create_empty_result(settings.strategy_name, trade_date)
            
            # Step 4: Execute trades
            execution_results = self._execute_trades(settings, entry_signals, trade_date)
            
            # Step 5: Monitor and exit positions
            exit_results = self._monitor_and_exit(settings, execution_results, trade_date)
            
            # Step 6: Calculate final P&L
            final_results = self._calculate_final_results(
                settings, execution_results, exit_results, trade_date
            )
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error processing ORB strategy: {e}")
            return self._create_error_result(settings.strategy_name, trade_date, str(e))
    
    def _calculate_opening_range(
        self,
        settings: ORBSettingModel,
        trade_date: date
    ) -> Optional[Dict[str, float]]:
        """Calculate opening range using range calculator"""
        
        return self.range_calculator.calculate_opening_range(
            db_connection=self.conn,
            index=settings.index,
            trade_date=trade_date,
            range_start=settings.orb_range_start,
            range_end=settings.orb_range_end,
            underlying_type=settings.underlying
        )
    
    def _detect_breakout(
        self,
        settings: ORBSettingModel,
        trade_date: date,
        orb_range: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """Detect breakout using signal generator"""
        
        return self.signal_generator.detect_first_breakout(
            db_connection=self.conn,
            index=settings.index,
            trade_date=trade_date,
            range_high=orb_range['range_high'],
            range_low=orb_range['range_low'],
            range_end_time=settings.orb_range_end,
            last_entry_time=settings.last_entry_time,
            underlying_type=settings.underlying
        )
    
    def _generate_entry_signals(
        self,
        settings: ORBSettingModel,
        breakout_signal: Dict[str, Any],
        trade_date: date
    ) -> List[ProcessedORBSignal]:
        """Generate processed entry signals for all legs"""
        
        processed_signals = []
        
        # Convert breakout type from string to enum
        breakout_type_str = breakout_signal['breakout_type']
        if isinstance(breakout_type_str, str):
            breakout_type = ORBBreakoutType[breakout_type_str]
        else:
            breakout_type = breakout_type_str
        
        # Get legs that should execute based on breakout type
        legs_to_execute = settings.get_entry_legs(breakout_type)
        
        if not legs_to_execute:
            logger.warning(f"No legs configured for {breakout_type}")
            return []
        
        # Create processed signal
        signal_direction = ORBSignalDirection.BULLISH if breakout_type == ORBBreakoutType.HIGHBREAKOUT else ORBSignalDirection.BEARISH
        
        processed_signal = ProcessedORBSignal(
            entrydate=breakout_signal['breakout_time'].date() if isinstance(breakout_signal['breakout_time'], datetime) else trade_date,
            entrytime=breakout_signal['breakout_time'] if isinstance(breakout_signal['breakout_time'], time) else breakout_signal['breakout_time'].time(),
            exitdate=settings.end_time.date() if isinstance(settings.end_time, datetime) else trade_date,
            exittime=settings.end_time if isinstance(settings.end_time, time) else settings.end_time.time(),
            lots=sum(leg.lots for leg in legs_to_execute),
            signal_direction=signal_direction,
            breakout_type=breakout_type,
            range_high=breakout_signal['range_high'],
            range_low=breakout_signal['range_low'],
            breakout_price=breakout_signal['breakout_price'],
            legs_to_execute=legs_to_execute
        )
        
        processed_signals.append(processed_signal)
        
        return processed_signals
    
    def _execute_trades(
        self,
        settings: ORBSettingModel,
        entry_signals: List[ProcessedORBSignal],
        trade_date: date
    ) -> List[Dict[str, Any]]:
        """Execute trades based on entry signals"""
        
        execution_results = []
        
        for signal in entry_signals:
            # Build and execute query for each signal
            signal_dict = {
                'signal_time': datetime.combine(signal.entrydate, signal.entrytime),
                'legs': []
            }
            
            for leg in signal.legs_to_execute:
                signal_dict['legs'].append({
                    'leg_id': leg.leg_id,
                    'instrument': leg.instrument,
                    'transaction': leg.transaction,
                    'expiry': leg.expiry,
                    'strike_method': leg.strike_method,
                    'strike_value': leg.strike_value,
                    'lots': leg.lots,
                    'signal_time': datetime.combine(signal.entrydate, signal.entrytime)
                })
            
            # Build query
            query = self.query_builder.build_orb_query(
                strategy=settings.__dict__,
                signals=signal_dict['legs'],
                trade_date=trade_date
            )
            
            # Execute query
            try:
                cursor = self.conn.execute(query)
                results = cursor.fetchall()
                
                for row in results:
                    execution = self._parse_execution_result(row, signal)
                    execution_results.append(execution)
                    
                    # Track active position
                    position_id = f"{signal.entrydate}_{execution['leg_id']}_{execution['strike']}"
                    self._active_positions[position_id] = execution
                    
            except Exception as e:
                logger.error(f"Error executing trades: {e}")
        
        return execution_results
    
    def _monitor_and_exit(
        self,
        settings: ORBSettingModel,
        execution_results: List[Dict[str, Any]],
        trade_date: date
    ) -> List[Dict[str, Any]]:
        """Monitor positions and generate exit signals"""
        
        exit_results = []
        
        # For simplicity, we'll use time-based exit at end_time
        # In a real implementation, this would monitor SL/TP levels tick by tick
        
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
        settings: ORBSettingModel,
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
                    'breakout_type': entry['breakout_type'],
                    'breakout_strength': entry['breakout_strength_pct']
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
    
    def _parse_execution_result(self, row: Any, signal: ProcessedORBSignal) -> Dict[str, Any]:
        """Parse execution result from database row"""
        
        return {
            'leg_id': row[0],
            'entry_time': row[1],
            'underlying_at_entry': float(row[2]),
            'breakout_type': row[3],
            'breakout_strength_pct': float(row[4]),
            'strike': int(row[5]),
            'symbol': row[6],
            'entry_price': float(row[7]),
            'open_interest': int(row[8]),
            'expiry_date': row[9],
            'instrument': row[10],
            'transaction_type': row[11],
            'lots': int(row[12]),
            'lot_size': 50,  # Default lot size, should come from config
            'entry_date': signal.entrydate
        }
    
    def _parse_exit_result(self, row: Any) -> Dict[str, Any]:
        """Parse exit result from database row"""
        
        return {
            'position_id': row[0],
            'leg_id': row[1],
            'exit_price': float(row[2]),
            'underlying_at_exit': float(row[3])
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
        self._reentry_counts.clear()
        self.range_calculator.clear_cache()
        self.signal_generator.clear_cache()