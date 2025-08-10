"""
Test POS Strategy with Real HeavyDB Data
"""

import asyncio
from heavydb import connect
import pandas as pd
from datetime import date
import logging
import sys
import os

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from backtester_v2.strategies.pos.models_simple import SimplePOSStrategy, SimpleLegModel, SimplePortfolioModel
from backtester_v2.strategies.pos.query_builder_simple import SimplePOSQueryBuilder
from backtester_v2.strategies.pos.parser_fixed import POSParserFixed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def connect_to_heavydb():
    """Connect to HeavyDB"""
    try:
        conn = connect(
            host='localhost',
            port=6274,
            user='admin',
            password='HyperInteractive',
            dbname='heavyai'
        )
        logger.info("✓ Connected to HeavyDB successfully")
        return conn
    except Exception as e:
        logger.error(f"✗ Failed to connect to HeavyDB: {e}")
        raise


def test_data_availability(conn):
    """Test basic data availability in HeavyDB"""
    logger.info("\n=== Testing Data Availability ===")
    
    query = """
    SELECT 
        COUNT(DISTINCT trade_date) as num_days,
        MIN(trade_date) as start_date,
        MAX(trade_date) as end_date,
        COUNT(DISTINCT option_type) as option_types,
        COUNT(DISTINCT expiry_type) as expiry_types,
        COUNT(*) as total_rows
    FROM nifty_option_chain
    WHERE trade_date >= '2024-01-01'
    """
    
    df = pd.read_sql(query, conn)
    logger.info(f"Data Summary:\n{df}")
    
    # Check expiry types
    query2 = """
    SELECT DISTINCT expiry_type, COUNT(*) as count
    FROM nifty_option_chain
    WHERE trade_date = '2024-01-01'
    GROUP BY expiry_type
    """
    
    df2 = pd.read_sql(query2, conn)
    logger.info(f"\nExpiry Types Available:\n{df2}")
    
    return df


def test_strike_availability(conn, test_date='2024-01-01'):
    """Check available strikes for a specific date"""
    logger.info(f"\n=== Testing Strike Availability for {test_date} ===")
    
    query = f"""
    SELECT 
        option_type,
        expiry_type,
        COUNT(DISTINCT strike_price) as strike_count,
        MIN(strike_price) as min_strike,
        MAX(strike_price) as max_strike,
        AVG(volume) as avg_volume
    FROM nifty_option_chain
    WHERE trade_date = '{test_date}'
    AND trade_time = '09:20:00'
    AND option_type IN ('CE', 'PE')
    GROUP BY option_type, expiry_type
    ORDER BY option_type, expiry_type
    """
    
    df = pd.read_sql(query, conn)
    logger.info(f"Strike Availability:\n{df}")
    
    return df


def test_iron_condor_query(conn):
    """Test Iron Condor strategy query"""
    logger.info("\n=== Testing Iron Condor Query ===")
    
    # Create test strategy
    test_strategy = SimplePOSStrategy(
        portfolio=SimplePortfolioModel(
            portfolio_name="Test_Iron_Condor",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 5),  # Just 5 days for testing
            position_size_value=100000,
            transaction_costs=0.001
        ),
        legs=[
            SimpleLegModel(
                leg_id=1,
                leg_name="short_put",
                option_type="PE",
                position_type="SELL",
                strike_selection="OTM",
                strike_offset=100,
                expiry_type="CURRENT_WEEK",
                lots=1
            ),
            SimpleLegModel(
                leg_id=2,
                leg_name="long_put",
                option_type="PE",
                position_type="BUY",
                strike_selection="OTM",
                strike_offset=200,
                expiry_type="CURRENT_WEEK",
                lots=1
            ),
            SimpleLegModel(
                leg_id=3,
                leg_name="short_call",
                option_type="CE",
                position_type="SELL",
                strike_selection="OTM",
                strike_offset=100,
                expiry_type="CURRENT_WEEK",
                lots=1
            ),
            SimpleLegModel(
                leg_id=4,
                leg_name="long_call",
                option_type="CE",
                position_type="BUY",
                strike_selection="OTM",
                strike_offset=200,
                expiry_type="CURRENT_WEEK",
                lots=1
            )
        ],
        strategy_type="IRON_CONDOR"
    )
    
    # Build query
    query_builder = SimplePOSQueryBuilder()
    
    # First test simple query
    simple_query = query_builder.build_simple_test_query(test_strategy)
    logger.info(f"\nSimple Test Query:\n{simple_query[:500]}...")
    
    try:
        df = pd.read_sql(simple_query, conn)
        logger.info(f"\n✓ Simple query returned {len(df)} rows")
        if not df.empty:
            logger.info(f"Sample data:\n{df.head()}")
    except Exception as e:
        logger.error(f"✗ Simple query failed: {e}")
        return
    
    # Test single leg query
    single_leg_query = query_builder.build_single_leg_query(
        test_strategy.legs[0], 
        test_strategy.portfolio
    )
    logger.info(f"\nSingle Leg Query:\n{single_leg_query[:500]}...")
    
    try:
        df = pd.read_sql(single_leg_query, conn)
        logger.info(f"\n✓ Single leg query returned {len(df)} rows")
        if not df.empty:
            logger.info(f"Sample data:\n{df.head()}")
    except Exception as e:
        logger.error(f"✗ Single leg query failed: {e}")
    
    # Test full position query
    position_query = query_builder.build_position_query(test_strategy)
    logger.info(f"\nFull Position Query:\n{position_query[:1000]}...")
    
    try:
        df = pd.read_sql(position_query, conn)
        logger.info(f"\n✓ Position query returned {len(df)} rows")
        if not df.empty:
            logger.info(f"Columns: {df.columns.tolist()}")
            logger.info(f"First row:\n{df.iloc[0]}")
    except Exception as e:
        logger.error(f"✗ Position query failed: {e}")
        # Try to debug
        logger.info("Debugging query issues...")
        debug_query = """
        SELECT DISTINCT option_type, expiry_type, COUNT(*) 
        FROM nifty_option_chain 
        WHERE trade_date = '2024-01-01' 
        GROUP BY option_type, expiry_type
        """
        try:
            debug_df = pd.read_sql(debug_query, conn)
            logger.info(f"Available data:\n{debug_df}")
        except:
            pass


def test_parser_with_sample_data(conn):
    """Test parser with sample input files"""
    logger.info("\n=== Testing Parser with Sample Data ===")
    
    # Check if sample files exist
    portfolio_file = "/srv/samba/shared/bt/backtester_stable/BTRUN/input_sheets/pos/input_positional_portfolio.xlsx"
    strategy_file = "/srv/samba/shared/bt/backtester_stable/BTRUN/input_sheets/pos/input_positional_strategy.xlsx"
    
    if not os.path.exists(portfolio_file):
        logger.warning(f"Portfolio file not found: {portfolio_file}")
        return
    
    if not os.path.exists(strategy_file):
        logger.warning(f"Strategy file not found: {strategy_file}")
        return
    
    # Test parser
    parser = POSParserFixed()
    
    try:
        parsed_data = parser.parse_input(
            portfolio_file=portfolio_file,
            strategy_file=strategy_file
        )
        
        if parsed_data.get('errors'):
            logger.error(f"Parser errors: {parsed_data['errors']}")
        else:
            logger.info("✓ Parser succeeded")
            
            if 'model' in parsed_data:
                strategy_model = parsed_data['model']
                logger.info(f"Strategy Summary:\n{strategy_model.get_summary()}")
                
                # Test query generation
                query_builder = SimplePOSQueryBuilder()
                query = query_builder.build_position_query(strategy_model)
                logger.info(f"\nGenerated query length: {len(query)} characters")
                
                # Try to execute
                try:
                    df = pd.read_sql(query[:1000] + "... LIMIT 10", conn)  # Truncated for safety
                    logger.info(f"Query test returned {len(df)} rows")
                except Exception as e:
                    logger.error(f"Query execution failed: {e}")
                    
    except Exception as e:
        logger.error(f"Parser failed: {e}")


def test_specific_date_data(conn, test_date='2024-01-01'):
    """Test what data is actually available for a specific date"""
    logger.info(f"\n=== Testing Specific Date Data: {test_date} ===")
    
    query = f"""
    SELECT 
        option_type,
        expiry_type,
        expiry_date,
        strike_price,
        close_price,
        volume,
        delta,
        underlying_value
    FROM nifty_option_chain
    WHERE trade_date = '{test_date}'
    AND trade_time = '09:20:00'
    AND option_type IN ('CE', 'PE', 'XX')
    ORDER BY option_type, expiry_type, strike_price
    LIMIT 20
    """
    
    df = pd.read_sql(query, conn)
    logger.info(f"Sample data for {test_date}:\n{df}")
    
    # Check spot price
    spot_query = f"""
    SELECT 
        MAX(CASE WHEN option_type = 'XX' THEN close_price ELSE underlying_value END) as spot_price
    FROM nifty_option_chain
    WHERE trade_date = '{test_date}'
    AND trade_time = '09:20:00'
    """
    
    spot_df = pd.read_sql(spot_query, conn)
    logger.info(f"\nSpot price on {test_date}: {spot_df['spot_price'].iloc[0]}")


async def main():
    """Main test function"""
    logger.info("Starting POS Strategy HeavyDB Tests")
    
    # Connect to HeavyDB
    conn = connect_to_heavydb()
    
    try:
        # Run tests
        test_data_availability(conn)
        test_strike_availability(conn)
        test_specific_date_data(conn)
        test_iron_condor_query(conn)
        test_parser_with_sample_data(conn)
        
        logger.info("\n=== All Tests Completed ===")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()
        logger.info("Disconnected from HeavyDB")


if __name__ == "__main__":
    asyncio.run(main())