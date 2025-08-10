#!/usr/bin/env python3
"""
TV Signal Processor - Processes raw TV signals into executable trades
"""

from datetime import datetime, timedelta, date, time as datetime_time
from typing import List, Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class SignalProcessor:
    """Processes raw TV signals into executable trades"""
    
    def process_signals(
        self, 
        raw_signals: List[Dict[str, Any]], 
        tv_settings: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Process and pair signals, apply time adjustments
        
        Args:
            raw_signals: List of raw signal dictionaries
            tv_settings: TV settings dictionary
            
        Returns:
            List of processed signal dictionaries
        """
        
        # Filter by date range
        filtered_signals = self._filter_by_date_range(
            raw_signals,
            tv_settings.get('start_date'),
            tv_settings.get('end_date')
        )
        
        # Add synthetic manual trades if configured
        if tv_settings.get('manual_trade_entry_time') and tv_settings.get('manual_trade_lots'):
            manual_signals = self._generate_manual_signals(tv_settings)
            filtered_signals.extend(manual_signals)
        
        # Sort signals by datetime
        filtered_signals.sort(key=lambda x: x['datetime'])
        
        # Pair entry/exit signals
        paired_signals = self._pair_signals(filtered_signals)
        
        # Process each pair
        processed_signals = []
        for entry_signal, exit_signal in paired_signals:
            processed = self._process_signal_pair(
                entry_signal,
                exit_signal,
                tv_settings
            )
            if processed:
                processed_signals.append(processed)
        
        # Handle rollover if enabled
        if tv_settings.get('do_rollover'):
            processed_signals = self._apply_rollover(processed_signals, tv_settings)
        
        return processed_signals
    
    def _filter_by_date_range(
        self, 
        signals: List[Dict[str, Any]], 
        start_date: Optional[date],
        end_date: Optional[date]
    ) -> List[Dict[str, Any]]:
        """Filter signals by date range"""
        if not start_date and not end_date:
            return signals
        
        filtered = []
        for signal in signals:
            signal_date = signal['datetime'].date()
            
            if start_date and signal_date < start_date:
                continue
            if end_date and signal_date > end_date:
                continue
            
            filtered.append(signal)
        
        return filtered
    
    def _generate_manual_signals(self, tv_settings: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate synthetic manual signals for each trading day"""
        manual_signals = []
        
        entry_time = tv_settings.get('manual_trade_entry_time')
        exit_time = tv_settings.get('intraday_exit_time', datetime_time(15, 15))
        lots = tv_settings.get('manual_trade_lots', 1)
        
        # Generate signals for each day in range
        start_date = tv_settings.get('start_date')
        end_date = tv_settings.get('end_date')
        
        if not start_date or not end_date:
            return manual_signals
        
        current_date = start_date
        trade_no = 0
        
        while current_date <= end_date:
            # Skip weekends (simplified - should check trading calendar)
            if current_date.weekday() < 5:  # Monday = 0, Friday = 4
                trade_no += 1
                
                # Create entry signal
                entry_signal = {
                    'trade_no': f'MANUAL_{trade_no}',
                    'signal_type': 'MANUAL_ENTRY',
                    'datetime': datetime.combine(current_date, entry_time),
                    'contracts': lots
                }
                manual_signals.append(entry_signal)
                
                # Create exit signal
                exit_signal = {
                    'trade_no': f'MANUAL_{trade_no}',
                    'signal_type': 'MANUAL_EXIT',
                    'datetime': datetime.combine(current_date, exit_time),
                    'contracts': lots
                }
                manual_signals.append(exit_signal)
            
            current_date += timedelta(days=1)
        
        return manual_signals
    
    def _pair_signals(self, signals: List[Dict[str, Any]]) -> List[Tuple[Dict, Optional[Dict]]]:
        """Pair entry and exit signals by trade number"""
        pairs = []
        entries = {}
        
        for signal in signals:
            trade_no = signal['trade_no']
            signal_type = signal['signal_type'].upper()  # Normalize to uppercase
            
            if 'ENTRY' in signal_type:
                # Store entry signal
                entries[trade_no] = signal
            elif 'EXIT' in signal_type:
                # Find matching entry
                if trade_no in entries:
                    pairs.append((entries[trade_no], signal))
                    del entries[trade_no]
                else:
                    logger.warning(f"Exit signal {trade_no} has no matching entry")
        
        # Handle unpaired entries (no exit)
        for trade_no, entry in entries.items():
            logger.warning(f"Entry signal {trade_no} has no matching exit")
            pairs.append((entry, None))
        
        return pairs
    
    def _process_signal_pair(
        self,
        entry_signal: Dict[str, Any],
        exit_signal: Optional[Dict[str, Any]],
        tv_settings: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Process a paired entry/exit signal"""
        
        # Determine signal direction
        signal_type = entry_signal['signal_type'].upper()
        if 'LONG' in signal_type:
            direction = 'LONG'
        elif 'SHORT' in signal_type:
            direction = 'SHORT'
        elif 'MANUAL' in signal_type:
            direction = 'MANUAL'
        else:
            logger.warning(f"Unknown signal type: {signal_type}")
            return None
        
        # Apply time adjustments to entry
        entry_datetime = entry_signal['datetime']
        
        # First trade entry time override
        if tv_settings.get('first_trade_entry_time'):
            # Check if this is the first trade of the day
            # (simplified - should track by day)
            entry_datetime = datetime.combine(
                entry_datetime.date(),
                tv_settings['first_trade_entry_time']
            )
        
        # Add entry time offset
        if tv_settings.get('increase_entry_signal_time_by'):
            entry_datetime += timedelta(seconds=tv_settings['increase_entry_signal_time_by'])
        
        # Determine exit time
        if exit_signal and tv_settings.get('tv_exit_applicable'):
            exit_datetime = exit_signal['datetime']
            
            # Add exit time offset
            if tv_settings.get('increase_exit_signal_time_by'):
                exit_datetime += timedelta(seconds=tv_settings['increase_exit_signal_time_by'])
        else:
            # Use intraday exit time
            exit_datetime = datetime.combine(
                entry_datetime.date(),
                tv_settings.get('intraday_exit_time', datetime_time(15, 15))
            )
        
        # Check for intraday square off override
        if tv_settings.get('intraday_sqoff_applicable'):
            intraday_exit = datetime.combine(
                entry_datetime.date(),
                tv_settings.get('intraday_exit_time', datetime_time(15, 15))
            )
            if intraday_exit < exit_datetime:
                exit_datetime = intraday_exit
        
        # Build processed signal
        processed = {
            'trade_no': entry_signal['trade_no'],
            'signal_direction': direction,
            'entry_date': entry_datetime.date(),
            'entry_time': entry_datetime.time(),
            'exit_date': exit_datetime.date(),
            'exit_time': exit_datetime.time(),
            'lots': entry_signal.get('lots', entry_signal.get('contracts', 1)),
            'is_rollover_trade': False,
            'original_trade_no': entry_signal['trade_no']
        }
        
        # Determine portfolio file path
        if direction == 'LONG':
            processed['portfolio_file'] = tv_settings.get('long_portfolio_file_path')
        elif direction == 'SHORT':
            processed['portfolio_file'] = tv_settings.get('short_portfolio_file_path')
        elif direction == 'MANUAL':
            processed['portfolio_file'] = tv_settings.get('manual_portfolio_file_path')
        
        return processed
    
    def _apply_rollover(
        self, 
        signals: List[Dict[str, Any]], 
        tv_settings: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply rollover logic to signals spanning multiple expiries"""
        # This is a simplified implementation
        # Full implementation would need expiry calendar
        
        rollover_time = tv_settings.get('rollover_time', datetime_time(15, 15))
        processed_with_rollover = []
        
        for signal in signals:
            # Check if signal spans multiple days (simplified check)
            if signal['entry_date'] != signal['exit_date']:
                # Could be a rollover candidate
                # For now, just mark it
                signal['is_rollover_trade'] = True
            
            processed_with_rollover.append(signal)
        
        return processed_with_rollover