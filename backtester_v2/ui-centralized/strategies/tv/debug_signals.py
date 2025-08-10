#!/usr/bin/env python3
"""
TV Signal Debugging Utility
Analyzes and validates signal files for common issues
"""

import sys
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Any, Tuple

# Add parent directory to path
sys.path.insert(0, '.')

from parser import TVParser


class SignalDebugger:
    """Debug and analyze TV signal files"""
    
    def __init__(self):
        self.parser = TVParser()
        self.issues = []
        self.stats = {}
    
    def analyze_signal_file(self, file_path: str, date_format: str = '%Y%m%d %H%M%S') -> Dict[str, Any]:
        """Comprehensive signal file analysis"""
        print(f"\n🔍 Analyzing signal file: {file_path}")
        
        analysis = {
            'file_info': self._get_file_info(file_path),
            'structure': self._analyze_structure(file_path),
            'signals': self._analyze_signals(file_path, date_format),
            'pairing': self._analyze_pairing(file_path, date_format),
            'timing': self._analyze_timing(file_path, date_format),
            'issues': self.issues,
            'recommendations': self._generate_recommendations()
        }
        
        return analysis
    
    def _get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get basic file information"""
        path = Path(file_path)
        return {
            'name': path.name,
            'size': path.stat().st_size,
            'modified': datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            'exists': path.exists()
        }
    
    def _analyze_structure(self, file_path: str) -> Dict[str, Any]:
        """Analyze Excel file structure"""
        try:
            # Get all sheet names
            xl_file = pd.ExcelFile(file_path)
            sheets = xl_file.sheet_names
            
            # Try to find the signals sheet
            signal_sheet = None
            for sheet in sheets:
                if 'trade' in sheet.lower() or 'signal' in sheet.lower():
                    signal_sheet = sheet
                    break
            
            if not signal_sheet and sheets:
                signal_sheet = sheets[0]
            
            # Read the signal sheet
            df = pd.read_excel(file_path, sheet_name=signal_sheet)
            
            return {
                'sheets': sheets,
                'signal_sheet': signal_sheet,
                'columns': df.columns.tolist(),
                'row_count': len(df),
                'has_required_columns': self._check_required_columns(df)
            }
            
        except Exception as e:
            self.issues.append(f"Structure analysis failed: {e}")
            return {'error': str(e)}
    
    def _check_required_columns(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Check for required columns"""
        required = ['Trade #', 'Type', 'Date/Time', 'Contracts']
        found = {}
        
        for col in required:
            found[col] = col in df.columns
            if not found[col]:
                self.issues.append(f"Missing required column: {col}")
        
        return found
    
    def _analyze_signals(self, file_path: str, date_format: str) -> Dict[str, Any]:
        """Analyze signal content"""
        try:
            signals = self.parser.parse_signals(file_path, date_format)
            
            # Categorize signals
            signal_types = {}
            for signal in signals:
                sig_type = signal.get('signal_type', 'Unknown')
                signal_types[sig_type] = signal_types.get(sig_type, 0) + 1
            
            # Find unique trades
            trade_numbers = set(s['trade_no'] for s in signals)
            
            # Check for duplicates
            duplicates = []
            seen = set()
            for signal in signals:
                key = (signal['trade_no'], signal['signal_type'])
                if key in seen:
                    duplicates.append(key)
                seen.add(key)
            
            if duplicates:
                self.issues.append(f"Found {len(duplicates)} duplicate signals")
            
            return {
                'total_signals': len(signals),
                'signal_types': signal_types,
                'unique_trades': len(trade_numbers),
                'trade_numbers': sorted(list(trade_numbers)),
                'duplicates': len(duplicates),
                'date_range': self._get_date_range(signals)
            }
            
        except Exception as e:
            self.issues.append(f"Signal analysis failed: {e}")
            return {'error': str(e)}
    
    def _get_date_range(self, signals: List[Dict]) -> Dict[str, str]:
        """Get date range of signals"""
        if not signals:
            return {'start': None, 'end': None}
        
        dates = [s['datetime'] for s in signals if 'datetime' in s]
        if dates:
            return {
                'start': min(dates).isoformat(),
                'end': max(dates).isoformat()
            }
        return {'start': None, 'end': None}
    
    def _analyze_pairing(self, file_path: str, date_format: str) -> Dict[str, Any]:
        """Analyze signal pairing"""
        try:
            signals = self.parser.parse_signals(file_path, date_format)
            
            # Group by trade number
            trades = {}
            for signal in signals:
                trade_no = signal['trade_no']
                if trade_no not in trades:
                    trades[trade_no] = []
                trades[trade_no].append(signal)
            
            # Analyze each trade
            paired = 0
            unpaired_entries = []
            unpaired_exits = []
            
            for trade_no, trade_signals in trades.items():
                entries = [s for s in trade_signals if 'Entry' in s['signal_type']]
                exits = [s for s in trade_signals if 'Exit' in s['signal_type']]
                
                if len(entries) == 1 and len(exits) == 1:
                    paired += 1
                elif len(entries) > 0 and len(exits) == 0:
                    unpaired_entries.append(trade_no)
                    self.issues.append(f"Trade {trade_no} has entry but no exit")
                elif len(entries) == 0 and len(exits) > 0:
                    unpaired_exits.append(trade_no)
                    self.issues.append(f"Trade {trade_no} has exit but no entry")
                elif len(entries) > 1 or len(exits) > 1:
                    self.issues.append(f"Trade {trade_no} has multiple entries or exits")
            
            return {
                'total_trades': len(trades),
                'paired_trades': paired,
                'unpaired_entries': unpaired_entries,
                'unpaired_exits': unpaired_exits,
                'pairing_rate': (paired / len(trades) * 100) if trades else 0
            }
            
        except Exception as e:
            self.issues.append(f"Pairing analysis failed: {e}")
            return {'error': str(e)}
    
    def _analyze_timing(self, file_path: str, date_format: str) -> Dict[str, Any]:
        """Analyze signal timing"""
        try:
            signals = self.parser.parse_signals(file_path, date_format)
            
            # Group by trade and check timing
            trades = {}
            for signal in signals:
                trade_no = signal['trade_no']
                if trade_no not in trades:
                    trades[trade_no] = []
                trades[trade_no].append(signal)
            
            timing_issues = []
            
            for trade_no, trade_signals in trades.items():
                # Sort by datetime
                trade_signals.sort(key=lambda x: x['datetime'])
                
                # Check if exit comes after entry
                entries = [s for s in trade_signals if 'Entry' in s['signal_type']]
                exits = [s for s in trade_signals if 'Exit' in s['signal_type']]
                
                if entries and exits:
                    entry_time = entries[0]['datetime']
                    exit_time = exits[0]['datetime']
                    
                    if exit_time <= entry_time:
                        timing_issues.append({
                            'trade': trade_no,
                            'issue': 'Exit before entry',
                            'entry': entry_time.isoformat(),
                            'exit': exit_time.isoformat()
                        })
                        self.issues.append(f"Trade {trade_no}: Exit time before entry time")
            
            # Calculate average holding time
            holding_times = []
            for trade_no, trade_signals in trades.items():
                entries = [s for s in trade_signals if 'Entry' in s['signal_type']]
                exits = [s for s in trade_signals if 'Exit' in s['signal_type']]
                
                if entries and exits:
                    duration = (exits[0]['datetime'] - entries[0]['datetime']).total_seconds() / 60
                    holding_times.append(duration)
            
            return {
                'timing_issues': len(timing_issues),
                'timing_details': timing_issues[:5],  # Show first 5
                'avg_holding_time_minutes': sum(holding_times) / len(holding_times) if holding_times else 0,
                'min_holding_time_minutes': min(holding_times) if holding_times else 0,
                'max_holding_time_minutes': max(holding_times) if holding_times else 0
            }
            
        except Exception as e:
            self.issues.append(f"Timing analysis failed: {e}")
            return {'error': str(e)}
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        if any('Missing required column' in issue for issue in self.issues):
            recommendations.append("Ensure signal file has all required columns: Trade #, Type, Date/Time, Contracts")
        
        if any('no exit' in issue for issue in self.issues):
            recommendations.append("Add exit signals for all entry signals or configure intraday square-off")
        
        if any('Exit before entry' in issue for issue in self.issues):
            recommendations.append("Check date/time format and ensure exits come after entries")
        
        if any('duplicate signals' in issue for issue in self.issues):
            recommendations.append("Remove duplicate signals to avoid double execution")
        
        if not self.issues:
            recommendations.append("Signal file appears to be valid and well-formed")
        
        return recommendations
    
    def print_report(self, analysis: Dict[str, Any]):
        """Print formatted analysis report"""
        print("\n" + "="*80)
        print("SIGNAL FILE ANALYSIS REPORT")
        print("="*80)
        
        # File info
        info = analysis['file_info']
        print(f"\n📁 File Information:")
        print(f"   • Name: {info['name']}")
        print(f"   • Size: {info['size']:,} bytes")
        print(f"   • Modified: {info['modified']}")
        
        # Structure
        if 'structure' in analysis and 'error' not in analysis['structure']:
            struct = analysis['structure']
            print(f"\n📋 File Structure:")
            print(f"   • Sheets: {', '.join(struct['sheets'])}")
            print(f"   • Signal sheet: {struct['signal_sheet']}")
            print(f"   • Rows: {struct['row_count']}")
            print(f"   • Columns: {len(struct['columns'])}")
            
            # Required columns
            print(f"\n✅ Required Columns:")
            for col, found in struct['has_required_columns'].items():
                status = "✓" if found else "✗"
                print(f"   {status} {col}")
        
        # Signals
        if 'signals' in analysis and 'error' not in analysis['signals']:
            sigs = analysis['signals']
            print(f"\n📊 Signal Analysis:")
            print(f"   • Total signals: {sigs['total_signals']}")
            print(f"   • Unique trades: {sigs['unique_trades']}")
            print(f"   • Date range: {sigs['date_range']['start']} to {sigs['date_range']['end']}")
            
            print(f"\n   Signal Types:")
            for sig_type, count in sigs['signal_types'].items():
                print(f"   • {sig_type}: {count}")
        
        # Pairing
        if 'pairing' in analysis and 'error' not in analysis['pairing']:
            pair = analysis['pairing']
            print(f"\n🔗 Signal Pairing:")
            print(f"   • Total trades: {pair['total_trades']}")
            print(f"   • Paired trades: {pair['paired_trades']}")
            print(f"   • Pairing rate: {pair['pairing_rate']:.1f}%")
            
            if pair['unpaired_entries']:
                print(f"   • Unpaired entries: {len(pair['unpaired_entries'])}")
            if pair['unpaired_exits']:
                print(f"   • Unpaired exits: {len(pair['unpaired_exits'])}")
        
        # Timing
        if 'timing' in analysis and 'error' not in analysis['timing']:
            timing = analysis['timing']
            print(f"\n⏱️  Timing Analysis:")
            print(f"   • Timing issues: {timing['timing_issues']}")
            print(f"   • Avg holding time: {timing['avg_holding_time_minutes']:.1f} minutes")
            print(f"   • Min holding time: {timing['min_holding_time_minutes']:.1f} minutes")
            print(f"   • Max holding time: {timing['max_holding_time_minutes']:.1f} minutes")
        
        # Issues
        if analysis['issues']:
            print(f"\n⚠️  Issues Found ({len(analysis['issues'])}):")
            for issue in analysis['issues'][:10]:  # Show first 10
                print(f"   • {issue}")
            if len(analysis['issues']) > 10:
                print(f"   ... and {len(analysis['issues']) - 10} more")
        
        # Recommendations
        if analysis['recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in analysis['recommendations']:
                print(f"   • {rec}")
        
        print("\n" + "="*80)
        
        # Overall status
        if not analysis['issues']:
            print("✅ SIGNAL FILE IS VALID")
        else:
            print(f"⚠️  SIGNAL FILE HAS {len(analysis['issues'])} ISSUES")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Debug and analyze TV signal files')
    parser.add_argument('--signal-file', required=True, help='Path to signal file')
    parser.add_argument('--date-format', default='%Y%m%d %H%M%S', help='Date format in signal file')
    parser.add_argument('--output-json', help='Save analysis to JSON file')
    
    args = parser.parse_args()
    
    # Create debugger
    debugger = SignalDebugger()
    
    # Analyze signal file
    analysis = debugger.analyze_signal_file(args.signal_file, args.date_format)
    
    # Print report
    debugger.print_report(analysis)
    
    # Save JSON if requested
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"\n📄 Analysis saved to: {args.output_json}")
    
    # Exit with error code if issues found
    return 1 if analysis['issues'] else 0


if __name__ == "__main__":
    sys.exit(main())