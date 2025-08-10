#!/usr/bin/env python3
"""
Check Real HeavyDB Table Structure

Quick script to examine the actual column names and structure of the nifty_option_chain table
to fix the column name issues in the production validator.
"""

import heavydb
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_table_structure():
    """Check the actual table structure"""
    try:
        # Connect to HeavyDB
        config = {
            'host': 'localhost',
            'port': 6274,
            'user': 'admin',
            'password': 'HyperInteractive',
            'dbname': 'heavyai'
        }
        
        connection = heavydb.connect(**config)
        logger.info("✅ Connected to HeavyDB")
        
        # Get table schema
        try:
            schema_query = """
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'nifty_option_chain'
            ORDER BY ordinal_position
            """
            result = connection.execute(schema_query)
            columns = result.fetchall()
            
            print("\n" + "="*60)
            print("NIFTY_OPTION_CHAIN TABLE STRUCTURE")
            print("="*60)
            
            for i, (col_name, data_type) in enumerate(columns, 1):
                print(f"{i:2d}. {col_name:<25} {data_type}")
            
            print("="*60)
            print(f"Total columns: {len(columns)}")
            
        except Exception as e:
            logger.warning(f"Schema query failed: {e}")
            
            # Try alternative approach - sample data
            sample_query = "SELECT * FROM nifty_option_chain LIMIT 1"
            result = connection.execute(sample_query)
            
            # Get column names from cursor description
            if hasattr(result, 'description') and result.description:
                columns = [desc[0] for desc in result.description]
                print("\n" + "="*60)
                print("NIFTY_OPTION_CHAIN COLUMNS (from sample)")
                print("="*60)
                
                for i, col_name in enumerate(columns, 1):
                    print(f"{i:2d}. {col_name}")
                
                print("="*60)
                print(f"Total columns: {len(columns)}")
            else:
                # Try DESCRIBE approach
                try:
                    describe_query = "DESCRIBE nifty_option_chain"
                    result = connection.execute(describe_query)
                    columns = result.fetchall()
                    
                    print("\n" + "="*60)
                    print("NIFTY_OPTION_CHAIN TABLE DESCRIPTION")
                    print("="*60)
                    
                    for col_info in columns:
                        print(col_info)
                    
                except Exception as e2:
                    logger.error(f"DESCRIBE query also failed: {e2}")
        
        # Check for specific columns we need
        critical_columns = [
            'underlying_price', 'spot', 'underlying', 'nifty_price',
            'trade_date', 'trade_time', 'atm_strike',
            'ce_close', 'pe_close', 'ce_volume', 'pe_volume',
            'ce_oi', 'pe_oi', 'ce_delta', 'pe_delta'
        ]
        
        print("\n" + "="*60)
        print("CHECKING CRITICAL COLUMNS")
        print("="*60)
        
        for col in critical_columns:
            try:
                test_query = f"SELECT {col} FROM nifty_option_chain LIMIT 1"
                connection.execute(test_query)
                print(f"✅ {col}")
            except Exception:
                print(f"❌ {col}")
        
        # Sample data check
        print("\n" + "="*60)
        print("SAMPLE DATA (first 3 rows)")
        print("="*60)
        
        try:
            sample_query = "SELECT * FROM nifty_option_chain LIMIT 3"
            result = connection.execute(sample_query)
            rows = result.fetchall()
            
            for i, row in enumerate(rows, 1):
                print(f"Row {i}: {row[:10]}...")  # First 10 columns
                
        except Exception as e:
            logger.error(f"Sample data query failed: {e}")
        
        connection.close()
        
    except Exception as e:
        logger.error(f"Failed to check table structure: {e}")

if __name__ == "__main__":
    check_table_structure()
