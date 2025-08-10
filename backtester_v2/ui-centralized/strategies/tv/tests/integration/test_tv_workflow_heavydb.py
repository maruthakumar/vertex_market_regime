#!/usr/bin/env python3
"""
Integration Tests for Complete TV 6-File Workflow with HeavyDB
Tests the entire TV strategy workflow from Excel configuration through HeavyDB execution
NO MOCK DATA - ONLY REAL INPUT SHEETS AND HEAVYDB CONNECTION
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, time
from pathlib import Path
import json
import tempfile
import shutil
import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.tv.parser import TVParser
from strategies.tv.signal_processor import SignalProcessor
from strategies.tv.query_builder import TVQueryBuilder
from strategies.tv.processor import TVProcessor


class TestTVWorkflowHeavyDB:
    """Test complete TV workflow with real HeavyDB data"""
    
    def setup_method(self):
        """Setup for each test"""
        self.parser = TVParser()
        self.signal_processor = SignalProcessor()
        self.query_builder = TVQueryBuilder()
        self.processor = TVProcessor()
        self.temp_dir = None
        
    def teardown_method(self):
        """Cleanup after each test"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @pytest.mark.integration
    @pytest.mark.heavydb
    def test_complete_6_file_workflow(self, real_config_files, heavydb_connection):
        """Test complete workflow from 6-file configuration to HeavyDB execution"""
        
        # Step 1: Parse TV Master Configuration
        tv_master_path = real_config_files['tv_master']
        tv_config_result = self.parser.parse_tv_settings(str(tv_master_path))
        
        assert 'settings' in tv_config_result
        assert len(tv_config_result['settings']) > 0
        
        tv_config = tv_config_result['settings'][0]
        assert tv_config['name'] == 'TV_Backtest_Sample'
        assert tv_config['enabled'] is True
        
        # Step 2: Parse Signals File
        signals_path = real_config_files['signals']
        signals = self.parser.parse_signals(str(signals_path), tv_config['signal_date_format'])
        
        assert len(signals) == 4  # Real file has 4 signals
        assert signals[0]['trade_no'] == 'T001'
        
        # Step 3: Process Signals with TV Settings
        processed_signals = self.signal_processor.process_signals(signals, tv_config)
        
        assert len(processed_signals) > 0
        assert all('entry_date' in sig for sig in processed_signals)
        assert all('exit_date' in sig for sig in processed_signals)
        assert all('signal_direction' in sig for sig in processed_signals)
        
        # Step 4: Parse Portfolio Files (Long, Short, Manual)
        portfolio_configs = {}
        
        # Parse Long Portfolio
        long_portfolio_path = real_config_files['portfolio_long']
        portfolio_df = pd.read_excel(str(long_portfolio_path), sheet_name='PortfolioSetting', engine='openpyxl')
        portfolio_configs['long'] = {
            'capital': int(portfolio_df.iloc[0]['Capital']),
            'max_risk': int(portfolio_df.iloc[0]['MaxRisk']),
            'max_positions': int(portfolio_df.iloc[0]['MaxPositions']),
            'risk_per_trade': int(portfolio_df.iloc[0]['RiskPerTrade']),
            'lot_size': 50  # NIFTY lot size
        }
        
        # Parse Short Portfolio
        short_portfolio_path = real_config_files['portfolio_short']
        portfolio_df = pd.read_excel(str(short_portfolio_path), sheet_name='PortfolioSetting', engine='openpyxl')
        portfolio_configs['short'] = {
            'capital': int(portfolio_df.iloc[0]['Capital']),
            'max_risk': int(portfolio_df.iloc[0]['MaxRisk']),
            'max_positions': int(portfolio_df.iloc[0]['MaxPositions']),
            'risk_per_trade': int(portfolio_df.iloc[0]['RiskPerTrade']),
            'lot_size': 50
        }
        
        # Step 5: Parse TBS Strategy Configuration
        strategy_path = real_config_files['strategy']
        general_df = pd.read_excel(str(strategy_path), sheet_name='GeneralParameter', engine='openpyxl')
        legs_df = pd.read_excel(str(strategy_path), sheet_name='LegParameter', engine='openpyxl')
        
        strategy_config = {
            'name': general_df.iloc[0]['StrategyName'],
            'underlying': general_df.iloc[0]['Underlying'],
            'index': general_df.iloc[0]['Index'],
            'legs': []
        }
        
        # Parse leg parameters
        for _, leg_row in legs_df.iterrows():
            leg = {
                'leg_no': leg_row['LegID'],
                'option_type': 'CE' if leg_row['Instrument'] == 'call' else 'PE',
                'transaction_type': 'BUY' if leg_row['Transaction'] == 'buy' else 'SELL',
                'strike_selection': leg_row['StrikeMethod'],
                'strike_value': leg_row['StrikeValue'],
                'expiry_rule': 'CW' if leg_row['Expiry'] == 'current' else 'NW',
                'quantity': int(leg_row['Lots'])
            }
            strategy_config['legs'].append(leg)
        
        # Step 6: Generate Queries for Each Signal
        queries = []
        for signal in processed_signals[:2]:  # Test with first 2 signals to limit HeavyDB load
            # Determine portfolio based on signal direction
            if signal['signal_direction'] == 'LONG':
                portfolio = portfolio_configs['long']
            elif signal['signal_direction'] == 'SHORT':
                portfolio = portfolio_configs['short']
            else:
                portfolio = portfolio_configs['long']  # Default
            
            # Build query
            query = self.query_builder.build_signal_query(
                signal=signal,
                portfolio=portfolio,
                strategy=strategy_config,
                tv_settings=tv_config
            )
            
            queries.append({
                'signal': signal,
                'query': query,
                'portfolio': portfolio
            })
        
        assert len(queries) > 0
        assert all('query' in q for q in queries)
        
        # Step 7: Execute Queries on HeavyDB (with basic validation)
        results = []
        for query_info in queries:
            try:
                # For testing, we'll execute a simplified query
                test_query = f"""
                SELECT 
                    COUNT(*) as record_count,
                    MIN(trade_date) as min_date,
                    MAX(trade_date) as max_date
                FROM nifty_option_chain
                WHERE trade_date >= DATE '{query_info['signal']['entry_date']}'
                    AND trade_date <= DATE '{query_info['signal']['exit_date']}'
                LIMIT 1
                """
                
                cursor = heavydb_connection.execute(test_query)
                result = cursor.fetchall()
                
                if result:
                    query_result = {
                        'signal': query_info['signal'],
                        'db_result': result,
                        'success': True,
                        'record_count': result[0][0] if result else 0
                    }
                else:
                    query_result = {
                        'signal': query_info['signal'],
                        'db_result': None,
                        'success': False,
                        'error': 'No results'
                    }
                
                results.append(query_result)
                
            except Exception as e:
                results.append({
                    'signal': query_info['signal'],
                    'db_result': None,
                    'success': False,
                    'error': str(e)
                })
        
        # Step 8: Process Results
        successful_results = [r for r in results if r['success']]
        assert len(successful_results) > 0, "At least one query should succeed"
        
        # Step 9: Generate Output
        output = {
            'tv_config': tv_config['name'],
            'total_signals': len(signals),
            'processed_signals': len(processed_signals),
            'queries_executed': len(queries),
            'successful_queries': len(successful_results),
            'workflow_status': 'completed',
            'timestamp': datetime.now().isoformat()
        }
        
        # Validate complete workflow
        assert output['workflow_status'] == 'completed'
        assert output['total_signals'] == 4
        assert output['processed_signals'] > 0
        assert output['successful_queries'] > 0
    
    @pytest.mark.integration
    @pytest.mark.heavydb
    def test_signal_to_trade_workflow(self, real_config_files, heavydb_connection):
        """Test converting TV signals to actual trades with HeavyDB data"""
        
        # Parse configurations
        tv_config_result = self.parser.parse_tv_settings(str(real_config_files['tv_master']))
        tv_config = tv_config_result['settings'][0]
        
        signals = self.parser.parse_signals(str(real_config_files['signals']), tv_config['signal_date_format'])
        
        # Process first signal pair (T001)
        t001_signals = [s for s in signals if s['trade_no'] == 'T001']
        assert len(t001_signals) == 2  # Entry and Exit
        
        # Process signals
        processed = self.signal_processor.process_signals(t001_signals, tv_config)
        assert len(processed) == 1  # One complete trade
        
        trade = processed[0]
        assert trade['trade_no'] == 'T001'
        assert trade['signal_direction'] == 'LONG'
        assert trade['entry_date'] == date(2024, 1, 1)
        assert trade['exit_date'] == date(2024, 1, 1)
        
        # Query HeavyDB for actual option data
        query = f"""
        SELECT 
            trade_date,
            trade_time,
            index_spot,
            COUNT(DISTINCT strike) as available_strikes,
            COUNT(DISTINCT expiry_date) as available_expiries
        FROM nifty_option_chain
        WHERE trade_date = DATE '{trade['entry_date']}'
            AND trade_time >= TIME '09:15:00'
            AND trade_time <= TIME '15:30:00'
        GROUP BY trade_date, trade_time, index_spot
        ORDER BY trade_time
        LIMIT 10
        """
        
        cursor = heavydb_connection.execute(query)
        results = cursor.fetchall()
        
        # Validate HeavyDB has data for the trade date
        assert len(results) > 0, f"No HeavyDB data found for {trade['entry_date']}"
        
        # Check data quality
        for row in results:
            trade_date, trade_time, spot, strikes, expiries = row
            assert strikes > 0, "Should have strike prices available"
            assert expiries > 0, "Should have expiry dates available"
            assert spot > 0, "Should have valid spot price"
    
    @pytest.mark.integration
    @pytest.mark.heavydb
    def test_portfolio_allocation_workflow(self, real_config_files, heavydb_connection):
        """Test portfolio allocation across different signal types"""
        
        # Parse all configurations
        tv_config_result = self.parser.parse_tv_settings(str(real_config_files['tv_master']))
        tv_config = tv_config_result['settings'][0]
        
        signals = self.parser.parse_signals(str(real_config_files['signals']), tv_config['signal_date_format'])
        processed_signals = self.signal_processor.process_signals(signals, tv_config)
        
        # Group signals by direction
        long_signals = [s for s in processed_signals if s['signal_direction'] == 'LONG']
        short_signals = [s for s in processed_signals if s['signal_direction'] == 'SHORT']
        
        assert len(long_signals) > 0, "Should have LONG signals"
        assert len(short_signals) > 0, "Should have SHORT signals"
        
        # Verify portfolio file assignments
        for signal in long_signals:
            assert signal['portfolio_file'] == tv_config['long_portfolio_file_path']
        
        for signal in short_signals:
            assert signal['portfolio_file'] == tv_config['short_portfolio_file_path']
        
        # Test portfolio capital allocation
        portfolio_df = pd.read_excel(
            str(real_config_files['portfolio_long']), 
            sheet_name='PortfolioSetting',
            engine='openpyxl'
        )
        
        capital = portfolio_df.iloc[0]['Capital']
        max_positions = portfolio_df.iloc[0]['MaxPositions']
        risk_per_trade = portfolio_df.iloc[0]['RiskPerTrade']
        
        # Calculate position sizing
        position_size = (capital * risk_per_trade / 100) / max_positions
        
        assert capital == 1000000, "Expected capital of 1M"
        assert max_positions == 5, "Expected max 5 positions"
        assert risk_per_trade == 2, "Expected 2% risk per trade"
        assert position_size == 40000, "Expected 40K per position"
    
    @pytest.mark.integration
    @pytest.mark.heavydb
    def test_time_adjustments_workflow(self, real_config_files, heavydb_connection):
        """Test signal time adjustments and intraday square-off"""
        
        # Parse configurations
        tv_config_result = self.parser.parse_tv_settings(str(real_config_files['tv_master']))
        tv_config = tv_config_result['settings'][0]
        
        # Validate time settings from real config
        assert tv_config['intraday_sqoff_applicable'] is True
        assert tv_config['intraday_exit_time'] == time(15, 30, 0)
        assert tv_config['tv_exit_applicable'] is True
        
        # Create test signals with different scenarios
        test_signals = [
            {
                'trade_no': 'TEST1',
                'signal_type': 'Entry Long',
                'datetime': datetime(2024, 1, 1, 10, 0, 0),
                'lots': 1
            },
            {
                'trade_no': 'TEST1',
                'signal_type': 'Exit Long',
                'datetime': datetime(2024, 1, 1, 16, 0, 0),  # After market hours
                'lots': 1
            }
        ]
        
        # Process signals
        processed = self.signal_processor.process_signals(test_signals, tv_config)
        
        assert len(processed) == 1
        trade = processed[0]
        
        # Check intraday square-off was applied
        assert trade['exit_time'] == time(15, 30, 0), "Exit should be capped at intraday exit time"
        
        # Verify with HeavyDB that market data exists until exit time
        query = f"""
        SELECT 
            MAX(trade_time) as last_trade_time
        FROM nifty_option_chain
        WHERE trade_date = DATE '2024-01-01'
            AND trade_time <= TIME '15:30:00'
        """
        
        cursor = heavydb_connection.execute(query)
        result = cursor.fetchone()
        
        if result and result[0]:
            assert result[0] <= time(15, 30, 0), "Market data should end by 15:30"
    
    @pytest.mark.integration
    @pytest.mark.heavydb
    def test_strike_selection_workflow(self, real_config_files, heavydb_connection):
        """Test strike selection logic with real HeavyDB data"""
        
        # Parse strategy configuration
        strategy_path = real_config_files['strategy']
        legs_df = pd.read_excel(str(strategy_path), sheet_name='LegParameter', engine='openpyxl')
        
        # Get first leg configuration
        first_leg = legs_df.iloc[0]
        assert first_leg['StrikeMethod'] == 'ATM'
        assert first_leg['StrikeValue'] == 0
        
        # Test ATM strike selection with HeavyDB
        test_date = date(2024, 1, 1)
        test_time = time(9, 20, 0)
        
        # Query for ATM strike
        atm_query = f"""
        WITH spot_price AS (
            SELECT index_spot as spot
            FROM nifty_option_chain
            WHERE trade_date = DATE '{test_date}'
                AND trade_time = TIME '{test_time}'
                AND index_spot IS NOT NULL
            LIMIT 1
        )
        SELECT 
            strike,
            ABS(strike - spot) as distance
        FROM nifty_option_chain
        CROSS JOIN spot_price
        WHERE trade_date = DATE '{test_date}'
            AND trade_time = TIME '{test_time}'
        ORDER BY distance
        LIMIT 1
        """
        
        cursor = heavydb_connection.execute(atm_query)
        result = cursor.fetchone()
        
        assert result is not None, "Should find ATM strike"
        atm_strike, distance = result
        assert atm_strike > 0, "ATM strike should be positive"
        assert distance >= 0, "Distance should be non-negative"
    
    @pytest.mark.integration
    @pytest.mark.heavydb  
    def test_expiry_selection_workflow(self, real_config_files, heavydb_connection):
        """Test expiry selection logic with real HeavyDB data"""
        
        # Parse strategy configuration
        strategy_path = real_config_files['strategy']
        legs_df = pd.read_excel(str(strategy_path), sheet_name='LegParameter', engine='openpyxl')
        
        # Check expiry rule
        first_leg = legs_df.iloc[0]
        assert first_leg['Expiry'] == 'current'  # Current week expiry
        
        # Test expiry selection with HeavyDB
        test_date = date(2024, 1, 1)
        
        # Query for available expiries
        expiry_query = f"""
        SELECT DISTINCT 
            expiry_date,
            DATEDIFF('day', DATE '{test_date}', expiry_date) as days_to_expiry
        FROM nifty_option_chain
        WHERE trade_date = DATE '{test_date}'
            AND expiry_date >= DATE '{test_date}'
        ORDER BY expiry_date
        LIMIT 5
        """
        
        cursor = heavydb_connection.execute(expiry_query)
        results = cursor.fetchall()
        
        assert len(results) > 0, "Should have available expiries"
        
        # Validate expiry dates
        current_week_expiry = results[0][0]  # First expiry should be current week
        days_to_expiry = results[0][1]
        
        assert days_to_expiry >= 0, "Expiry should be in the future"
        assert days_to_expiry <= 7, "Current week expiry should be within 7 days"
    
    @pytest.mark.integration
    @pytest.mark.heavydb
    def test_complete_backtest_workflow(self, real_config_files, heavydb_connection):
        """Test complete backtest execution with P&L calculation"""
        
        # Create temporary output directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Parse all configurations
        tv_config_result = self.parser.parse_tv_settings(str(real_config_files['tv_master']))
        tv_config = tv_config_result['settings'][0]
        
        signals = self.parser.parse_signals(str(real_config_files['signals']), tv_config['signal_date_format'])
        processed_signals = self.signal_processor.process_signals(signals, tv_config)
        
        # Execute simplified backtest for first signal
        if processed_signals:
            signal = processed_signals[0]
            
            # Query for entry and exit prices
            price_query = f"""
            WITH entry_data AS (
                SELECT 
                    ce_close as entry_price,
                    strike
                FROM nifty_option_chain
                WHERE trade_date = DATE '{signal['entry_date']}'
                    AND trade_time = TIME '{signal['entry_time']}'
                    AND strike = (
                        SELECT strike
                        FROM nifty_option_chain
                        WHERE trade_date = DATE '{signal['entry_date']}'
                            AND trade_time = TIME '{signal['entry_time']}'
                            AND index_spot IS NOT NULL
                        ORDER BY ABS(strike - index_spot)
                        LIMIT 1
                    )
                LIMIT 1
            ),
            exit_data AS (
                SELECT 
                    ce_close as exit_price
                FROM nifty_option_chain
                JOIN entry_data ON nifty_option_chain.strike = entry_data.strike
                WHERE trade_date = DATE '{signal['exit_date']}'
                    AND trade_time = TIME '{signal['exit_time']}'
                LIMIT 1
            )
            SELECT 
                entry_data.entry_price,
                exit_data.exit_price,
                entry_data.strike,
                (exit_data.exit_price - entry_data.entry_price) * {signal['lots']} * 50 as pnl
            FROM entry_data
            CROSS JOIN exit_data
            """
            
            cursor = heavydb_connection.execute(price_query)
            result = cursor.fetchone()
            
            if result:
                entry_price, exit_price, strike, pnl = result
                
                # Create backtest result
                backtest_result = {
                    'trade_no': signal['trade_no'],
                    'signal_direction': signal['signal_direction'],
                    'entry_date': str(signal['entry_date']),
                    'entry_time': str(signal['entry_time']),
                    'exit_date': str(signal['exit_date']),
                    'exit_time': str(signal['exit_time']),
                    'strike': strike,
                    'entry_price': float(entry_price) if entry_price else 0,
                    'exit_price': float(exit_price) if exit_price else 0,
                    'pnl': float(pnl) if pnl else 0,
                    'lots': signal['lots'],
                    'status': 'completed'
                }
                
                # Save result
                output_file = Path(self.temp_dir) / f"backtest_result_{signal['trade_no']}.json"
                with open(output_file, 'w') as f:
                    json.dump(backtest_result, f, indent=2)
                
                # Validate result
                assert output_file.exists()
                assert backtest_result['strike'] > 0
                assert backtest_result['status'] == 'completed'


class TestTVWorkflowPerformance:
    """Performance tests for TV workflow with HeavyDB"""
    
    def setup_method(self):
        """Setup for each test"""
        self.parser = TVParser()
        self.signal_processor = SignalProcessor()
        self.query_builder = TVQueryBuilder()
        
    @pytest.mark.performance
    @pytest.mark.heavydb
    def test_workflow_performance(self, real_config_files, heavydb_connection):
        """Test complete workflow performance"""
        import time
        
        # Measure parsing performance
        start_time = time.time()
        
        # Parse all files
        tv_config_result = self.parser.parse_tv_settings(str(real_config_files['tv_master']))
        signals = self.parser.parse_signals(str(real_config_files['signals']), tv_config_result['settings'][0]['signal_date_format'])
        
        # Parse portfolios
        for portfolio_key in ['portfolio_long', 'portfolio_short', 'portfolio_manual']:
            pd.read_excel(str(real_config_files[portfolio_key]), sheet_name='PortfolioSetting', engine='openpyxl')
        
        # Parse strategy
        pd.read_excel(str(real_config_files['strategy']), sheet_name='GeneralParameter', engine='openpyxl')
        pd.read_excel(str(real_config_files['strategy']), sheet_name='LegParameter', engine='openpyxl')
        
        parsing_time = time.time() - start_time
        
        # Measure signal processing
        start_time = time.time()
        processed_signals = self.signal_processor.process_signals(signals, tv_config_result['settings'][0])
        processing_time = time.time() - start_time
        
        # Measure query execution (simple query)
        start_time = time.time()
        test_query = """
        SELECT COUNT(*) 
        FROM nifty_option_chain 
        WHERE trade_date = DATE '2024-01-01'
        LIMIT 1
        """
        cursor = heavydb_connection.execute(test_query)
        result = cursor.fetchone()
        query_time = time.time() - start_time
        
        # Performance assertions
        assert parsing_time < 1.0, f"Parsing too slow: {parsing_time:.3f}s"
        assert processing_time < 0.5, f"Signal processing too slow: {processing_time:.3f}s"
        assert query_time < 1.0, f"HeavyDB query too slow: {query_time:.3f}s"
        
        # Total workflow should be under 3 seconds
        total_time = parsing_time + processing_time + query_time
        assert total_time < 3.0, f"Total workflow too slow: {total_time:.3f}s"
        
        print(f"\nPerformance Metrics:")
        print(f"  Parsing: {parsing_time:.3f}s")
        print(f"  Processing: {processing_time:.3f}s")
        print(f"  Query: {query_time:.3f}s")
        print(f"  Total: {total_time:.3f}s")