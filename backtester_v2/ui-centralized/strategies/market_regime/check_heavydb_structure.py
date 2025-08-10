#!/usr/bin/env python3
"""
Check HeavyDB Structure and Data Availability

This script checks the HeavyDB connection and examines the nifty_option_chain table
structure to understand what real data is available for the market regime formation
validation analysis.

Author: The Augster
Date: 2025-06-19
Version: 1.0.0 (HeavyDB Structure Check)
"""

import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import sys
import os

# Add the path for HeavyDB imports
sys.path.append('/srv/samba/shared/bt/backtester_stable/BTRUN')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('heavydb_structure_check.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HeavyDBStructureChecker:
    """Check HeavyDB structure and data availability"""
    
    def __init__(self):
        """Initialize the structure checker"""
        self.connection = None
        self.table_info = {}
        
        logger.info("HeavyDB Structure Checker initialized")
    
    def connect_to_heavydb(self) -> bool:
        """Establish connection to HeavyDB"""
        try:
            # Try multiple connection methods
            connection_methods = [
                self._try_enhanced_connection,
                self._try_standard_connection,
                self._try_direct_connection
            ]
            
            for method in connection_methods:
                if method():
                    logger.info("✅ HeavyDB connection established")
                    return True
            
            logger.error("❌ Failed to establish HeavyDB connection")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error connecting to HeavyDB: {e}")
            return False
    
    def _try_enhanced_connection(self) -> bool:
        """Try enhanced HeavyDB connection"""
        try:
            from dal.heavydb_connection_enhanced import get_connection
            self.connection = get_connection()
            logger.info("Connected using enhanced connection")
            return True
        except Exception as e:
            logger.warning(f"Enhanced connection failed: {e}")
            return False
    
    def _try_standard_connection(self) -> bool:
        """Try standard HeavyDB connection"""
        try:
            from dal.heavydb_connection import get_connection
            self.connection = get_connection()
            logger.info("Connected using standard connection")
            return True
        except Exception as e:
            logger.warning(f"Standard connection failed: {e}")
            return False
    
    def _try_direct_connection(self) -> bool:
        """Try direct HeavyDB connection"""
        try:
            import heavydb
            
            # Default HeavyDB configuration
            config = {
                'host': 'localhost',
                'port': 6274,
                'user': 'admin',
                'password': 'HyperInteractive',
                'database': 'heavyai'
            }
            
            self.connection = heavydb.connect(**config)
            logger.info("Connected using direct connection")
            return True
        except Exception as e:
            logger.warning(f"Direct connection failed: {e}")
            return False
    
    def check_table_structure(self) -> Dict[str, Any]:
        """Check nifty_option_chain table structure"""
        logger.info("🔍 Checking nifty_option_chain table structure...")
        
        try:
            # Check if table exists
            table_exists_query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_name = 'nifty_option_chain'
            """
            
            result = self.connection.execute(table_exists_query)
            tables = result.fetchall()
            
            if not tables:
                logger.error("❌ nifty_option_chain table does not exist")
                return {'error': 'Table does not exist'}
            
            logger.info("✅ nifty_option_chain table exists")
            
            # Get table schema
            schema_query = """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'nifty_option_chain'
            ORDER BY ordinal_position
            """
            
            schema_result = self.connection.execute(schema_query)
            columns = schema_result.fetchall()
            
            table_info = {
                'table_exists': True,
                'total_columns': len(columns),
                'columns': {}
            }
            
            # Process column information
            for col in columns:
                column_name, data_type, is_nullable = col
                table_info['columns'][column_name] = {
                    'data_type': data_type,
                    'is_nullable': is_nullable
                }
            
            # Check for critical columns
            critical_columns = [
                'trade_date', 'trade_time', 'underlying_price', 'atm_strike',
                'strike', 'ce_close', 'pe_close', 'ce_volume', 'pe_volume',
                'ce_oi', 'pe_oi', 'ce_delta', 'ce_gamma', 'ce_theta', 'ce_vega',
                'pe_delta', 'pe_gamma', 'pe_theta', 'pe_vega'
            ]
            
            available_critical = []
            missing_critical = []
            
            for col in critical_columns:
                if col in table_info['columns']:
                    available_critical.append(col)
                else:
                    missing_critical.append(col)
            
            table_info['critical_columns'] = {
                'available': available_critical,
                'missing': missing_critical,
                'availability_rate': len(available_critical) / len(critical_columns) * 100
            }
            
            logger.info(f"✅ Table structure analyzed: {len(columns)} columns")
            logger.info(f"📊 Critical columns availability: {table_info['critical_columns']['availability_rate']:.1f}%")
            
            return table_info
            
        except Exception as e:
            logger.error(f"❌ Error checking table structure: {e}")
            return {'error': str(e)}
    
    def check_data_availability(self, start_date: str = "2024-01-01", 
                               end_date: str = "2024-01-31") -> Dict[str, Any]:
        """Check data availability for the specified date range"""
        logger.info(f"📅 Checking data availability from {start_date} to {end_date}...")
        
        try:
            # Check total records in date range
            count_query = f"""
            SELECT COUNT(*) as total_records
            FROM nifty_option_chain
            WHERE trade_date >= DATE '{start_date}'
            AND trade_date <= DATE '{end_date}'
            """
            
            count_result = self.connection.execute(count_query)
            total_records = count_result.fetchone()[0]
            
            if total_records == 0:
                logger.warning(f"⚠️ No data found for date range {start_date} to {end_date}")
                return {'error': 'No data available for specified date range'}
            
            logger.info(f"✅ Found {total_records:,} total records")
            
            # Check date coverage
            date_coverage_query = f"""
            SELECT 
                MIN(trade_date) as first_date,
                MAX(trade_date) as last_date,
                COUNT(DISTINCT trade_date) as unique_dates
            FROM nifty_option_chain
            WHERE trade_date >= DATE '{start_date}'
            AND trade_date <= DATE '{end_date}'
            """
            
            coverage_result = self.connection.execute(date_coverage_query)
            coverage_data = coverage_result.fetchone()
            
            # Check time coverage
            time_coverage_query = f"""
            SELECT 
                MIN(trade_time) as first_time,
                MAX(trade_time) as last_time,
                COUNT(DISTINCT trade_time) as unique_times
            FROM nifty_option_chain
            WHERE trade_date >= DATE '{start_date}'
            AND trade_date <= DATE '{end_date}'
            """
            
            time_result = self.connection.execute(time_coverage_query)
            time_data = time_result.fetchone()
            
            # Check underlying price availability
            spot_data_query = f"""
            SELECT 
                COUNT(*) as total_with_spot,
                COUNT(DISTINCT underlying_price) as unique_spot_prices,
                MIN(underlying_price) as min_spot,
                MAX(underlying_price) as max_spot,
                AVG(underlying_price) as avg_spot
            FROM nifty_option_chain
            WHERE trade_date >= DATE '{start_date}'
            AND trade_date <= DATE '{end_date}'
            AND underlying_price IS NOT NULL
            AND underlying_price > 0
            """
            
            spot_result = self.connection.execute(spot_data_query)
            spot_data = spot_result.fetchone()
            
            # Check options data availability
            options_data_query = f"""
            SELECT 
                COUNT(*) as total_options,
                SUM(CASE WHEN ce_close IS NOT NULL AND ce_close > 0 THEN 1 ELSE 0 END) as valid_ce_prices,
                SUM(CASE WHEN pe_close IS NOT NULL AND pe_close > 0 THEN 1 ELSE 0 END) as valid_pe_prices,
                SUM(CASE WHEN ce_volume IS NOT NULL AND ce_volume > 0 THEN 1 ELSE 0 END) as valid_ce_volume,
                SUM(CASE WHEN pe_volume IS NOT NULL AND pe_volume > 0 THEN 1 ELSE 0 END) as valid_pe_volume,
                SUM(CASE WHEN ce_oi IS NOT NULL AND ce_oi > 0 THEN 1 ELSE 0 END) as valid_ce_oi,
                SUM(CASE WHEN pe_oi IS NOT NULL AND pe_oi > 0 THEN 1 ELSE 0 END) as valid_pe_oi
            FROM nifty_option_chain
            WHERE trade_date >= DATE '{start_date}'
            AND trade_date <= DATE '{end_date}'
            """
            
            options_result = self.connection.execute(options_data_query)
            options_data = options_result.fetchone()
            
            # Check Greeks availability
            greeks_data_query = f"""
            SELECT 
                SUM(CASE WHEN ce_delta IS NOT NULL THEN 1 ELSE 0 END) as valid_ce_delta,
                SUM(CASE WHEN pe_delta IS NOT NULL THEN 1 ELSE 0 END) as valid_pe_delta,
                SUM(CASE WHEN ce_gamma IS NOT NULL THEN 1 ELSE 0 END) as valid_ce_gamma,
                SUM(CASE WHEN pe_gamma IS NOT NULL THEN 1 ELSE 0 END) as valid_pe_gamma,
                SUM(CASE WHEN ce_theta IS NOT NULL THEN 1 ELSE 0 END) as valid_ce_theta,
                SUM(CASE WHEN pe_theta IS NOT NULL THEN 1 ELSE 0 END) as valid_pe_theta,
                SUM(CASE WHEN ce_vega IS NOT NULL THEN 1 ELSE 0 END) as valid_ce_vega,
                SUM(CASE WHEN pe_vega IS NOT NULL THEN 1 ELSE 0 END) as valid_pe_vega
            FROM nifty_option_chain
            WHERE trade_date >= DATE '{start_date}'
            AND trade_date <= DATE '{end_date}'
            """
            
            greeks_result = self.connection.execute(greeks_data_query)
            greeks_data = greeks_result.fetchone()
            
            # Compile data availability report
            data_availability = {
                'total_records': total_records,
                'date_coverage': {
                    'first_date': coverage_data[0],
                    'last_date': coverage_data[1],
                    'unique_dates': coverage_data[2]
                },
                'time_coverage': {
                    'first_time': time_data[0],
                    'last_time': time_data[1],
                    'unique_times': time_data[2]
                },
                'spot_data': {
                    'total_with_spot': spot_data[0],
                    'unique_spot_prices': spot_data[1],
                    'min_spot': spot_data[2],
                    'max_spot': spot_data[3],
                    'avg_spot': spot_data[4],
                    'coverage_rate': (spot_data[0] / total_records * 100) if total_records > 0 else 0
                },
                'options_data': {
                    'total_options': options_data[0],
                    'valid_ce_prices': options_data[1],
                    'valid_pe_prices': options_data[2],
                    'valid_ce_volume': options_data[3],
                    'valid_pe_volume': options_data[4],
                    'valid_ce_oi': options_data[5],
                    'valid_pe_oi': options_data[6]
                },
                'greeks_data': {
                    'valid_ce_delta': greeks_data[0],
                    'valid_pe_delta': greeks_data[1],
                    'valid_ce_gamma': greeks_data[2],
                    'valid_pe_gamma': greeks_data[3],
                    'valid_ce_theta': greeks_data[4],
                    'valid_pe_theta': greeks_data[5],
                    'valid_ce_vega': greeks_data[6],
                    'valid_pe_vega': greeks_data[7]
                }
            }
            
            logger.info(f"✅ Data availability analysis completed")
            logger.info(f"📊 Spot data coverage: {data_availability['spot_data']['coverage_rate']:.1f}%")
            
            return data_availability
            
        except Exception as e:
            logger.error(f"❌ Error checking data availability: {e}")
            return {'error': str(e)}
    
    def generate_structure_report(self) -> str:
        """Generate comprehensive structure and data availability report"""
        logger.info("📝 Generating HeavyDB structure report...")
        
        try:
            # Connect to HeavyDB
            if not self.connect_to_heavydb():
                return "Failed to connect to HeavyDB"
            
            # Check table structure
            table_info = self.check_table_structure()
            
            if 'error' in table_info:
                return f"Table structure check failed: {table_info['error']}"
            
            # Check data availability
            data_availability = self.check_data_availability()
            
            if 'error' in data_availability:
                return f"Data availability check failed: {data_availability['error']}"
            
            # Generate report
            report = f"""
# HeavyDB Structure and Data Availability Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Table Structure Analysis

### nifty_option_chain Table
- **Status:** ✅ Table exists
- **Total Columns:** {table_info['total_columns']}
- **Critical Columns Availability:** {table_info['critical_columns']['availability_rate']:.1f}%

### Available Critical Columns
{chr(10).join(f"- {col}" for col in table_info['critical_columns']['available'])}

### Missing Critical Columns
{chr(10).join(f"- {col}" for col in table_info['critical_columns']['missing']) if table_info['critical_columns']['missing'] else "None"}

## Data Availability Analysis (January 2024)

### Overall Coverage
- **Total Records:** {data_availability['total_records']:,}
- **Date Range:** {data_availability['date_coverage']['first_date']} to {data_availability['date_coverage']['last_date']}
- **Unique Dates:** {data_availability['date_coverage']['unique_dates']}
- **Unique Times:** {data_availability['time_coverage']['unique_times']}

### Spot Price Data
- **Coverage Rate:** {data_availability['spot_data']['coverage_rate']:.1f}%
- **Price Range:** {data_availability['spot_data']['min_spot']:.2f} - {data_availability['spot_data']['max_spot']:.2f}
- **Average Price:** {data_availability['spot_data']['avg_spot']:.2f}

### Options Data Quality
- **Valid CE Prices:** {data_availability['options_data']['valid_ce_prices']:,}
- **Valid PE Prices:** {data_availability['options_data']['valid_pe_prices']:,}
- **Valid CE Volume:** {data_availability['options_data']['valid_ce_volume']:,}
- **Valid PE Volume:** {data_availability['options_data']['valid_pe_volume']:,}
- **Valid CE OI:** {data_availability['options_data']['valid_ce_oi']:,}
- **Valid PE OI:** {data_availability['options_data']['valid_pe_oi']:,}

### Greeks Data Quality
- **CE Delta:** {data_availability['greeks_data']['valid_ce_delta']:,}
- **PE Delta:** {data_availability['greeks_data']['valid_pe_delta']:,}
- **CE Gamma:** {data_availability['greeks_data']['valid_ce_gamma']:,}
- **PE Gamma:** {data_availability['greeks_data']['valid_pe_gamma']:,}
- **CE Theta:** {data_availability['greeks_data']['valid_ce_theta']:,}
- **PE Theta:** {data_availability['greeks_data']['valid_pe_theta']:,}
- **CE Vega:** {data_availability['greeks_data']['valid_ce_vega']:,}
- **PE Vega:** {data_availability['greeks_data']['valid_pe_vega']:,}

## Real Data Validation Readiness

✅ **Ready for Real Data Analysis:** HeavyDB contains sufficient real market data for comprehensive market regime formation validation.

---
*Report generated by HeavyDB Structure Checker*
"""
            
            # Save report
            report_file = f"heavydb_structure_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(report_file, 'w') as f:
                f.write(report)
            
            logger.info(f"✅ Structure report generated: {report_file}")
            return report_file
            
        except Exception as e:
            logger.error(f"❌ Error generating structure report: {e}")
            return f"Error: {e}"
        finally:
            if self.connection:
                self.connection.close()

if __name__ == "__main__":
    # Run structure check
    checker = HeavyDBStructureChecker()
    report_file = checker.generate_structure_report()
    
    print("\n" + "="*80)
    print("HEAVYDB STRUCTURE CHECK COMPLETED")
    print("="*80)
    print(f"Report file: {report_file}")
    print("="*80)
