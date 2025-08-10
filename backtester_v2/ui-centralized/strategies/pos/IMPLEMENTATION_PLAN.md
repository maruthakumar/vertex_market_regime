# POS Strategy - Detailed Implementation Plan

## Phase 1: Parser Fix (Day 1-2)

### 1.1 Update Parser to Match Actual Input Format

```python
# pos/parser_fixed.py
def parse_portfolio_excel(self, excel_path: str) -> Dict[str, Any]:
    """Parse portfolio settings from actual Excel format"""
    try:
        # Read the actual format
        df = pd.read_excel(excel_path, sheet_name='PortfolioSetting')
        
        # Extract values from first row
        row = df.iloc[0]
        
        portfolio_data = {
            'portfolio_name': row.get('PortfolioName', 'POS_Portfolio'),
            'strategy_name': 'Positional_Strategy',
            'strategy_type': 'CUSTOM',  # Will be determined from legs
            'start_date': pd.to_datetime(row['StartDate']).date(),
            'end_date': pd.to_datetime(row['EndDate']).date(),
            'index_name': 'NIFTY',  # Default, can be parameterized
            'underlying_price_type': 'SPOT',
            'position_sizing': 'FIXED',
            'max_positions': 1,
            'position_size_value': row.get('Multiplier', 1) * 50 * 100,  # lots * lot_size * multiplier
            'max_portfolio_risk': row.get('PortfolioStoploss', 0.02),
            'transaction_costs': row.get('SlippagePercent', 0.1) / 100,
            'use_intraday_data': bool(row.get('IsTickBT', False)),
            'calculate_greeks': True,
            'enable_adjustments': False,  # Start simple
            'enabled': bool(row.get('Enabled', True))
        }
        
        return portfolio_data
        
    except Exception as e:
        logger.error(f"Error parsing portfolio Excel: {str(e)}")
        self.errors.append(f"Portfolio parsing error: {str(e)}")
        return {}

def parse_strategy_excel(self, excel_path: str) -> Dict[str, Any]:
    """Parse strategy legs from actual Excel format"""
    try:
        # Read LegParameter sheet
        leg_df = pd.read_excel(excel_path, sheet_name='LegParameter')
        
        legs = []
        for idx, row in leg_df.iterrows():
            # Map actual columns to expected format
            leg = {
                'leg_id': idx + 1,
                'leg_name': row['LegID'],
                'option_type': 'CE' if 'CALL' in str(row.get('Instrument', '')).upper() else 'PE',
                'position_type': row['Transaction'].upper(),  # BUY/SELL
                'strike_selection': self._map_strike_method(row['StrikeMethod']),
                'strike_offset': self._extract_strike_offset(row['StrikeMethod']),
                'lots': int(row['Lots']),
                'lot_size': 50,  # NIFTY lot size
                'entry_time': time(9, 20),  # Default
                'exit_time': time(15, 20),  # Default
                'stop_loss': float(row['StopLossValue']) if pd.notna(row.get('StopLossValue')) else None,
                'take_profit': float(row['TargetValue']) if pd.notna(row.get('TargetValue')) else None,
                'is_active': bool(row.get('IsActive', True))
            }
            
            # Determine expiry from leg name
            if 'weekly' in row['LegID'].lower():
                leg['expiry_type'] = 'CURRENT_WEEK'
            elif 'monthly' in row['LegID'].lower():
                leg['expiry_type'] = 'CURRENT_MONTH'
            else:
                leg['expiry_type'] = 'NEXT_WEEK'
                
            legs.append(leg)
            
        # Detect strategy type from legs
        strategy_type = self._detect_strategy_type(legs)
        
        return {
            'legs': legs,
            'strategy_type': strategy_type,
            'entry_logic': 'ALL',
            'exit_logic': 'ALL'
        }
        
    except Exception as e:
        logger.error(f"Error parsing strategy Excel: {str(e)}")
        self.errors.append(f"Strategy parsing error: {str(e)}")
        return {'legs': []}

def _map_strike_method(self, method: str) -> str:
    """Map Excel strike method to internal format"""
    method_upper = str(method).upper()
    if 'ATM' in method_upper:
        return 'ATM'
    elif 'ITM' in method_upper:
        return 'ITM'
    elif 'OTM' in method_upper:
        return 'OTM'
    elif 'STRIKE' in method_upper:
        return 'STRIKE_PRICE'
    else:
        return 'ATM'  # Default

def _extract_strike_offset(self, method: str) -> float:
    """Extract offset value from strike method string"""
    import re
    # Extract number from strings like 'OTM_100' or 'ITM_50'
    match = re.search(r'(\d+)', str(method))
    return float(match.group(1)) if match else 0
```

### 1.2 Create Simplified Models for Initial Implementation

```python
# pos/models_simple.py
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date, time

class SimpleLegModel(BaseModel):
    """Simplified leg model for initial implementation"""
    leg_id: int
    leg_name: str
    option_type: str  # CE/PE
    position_type: str  # BUY/SELL
    strike_method: str  # ATM/ITM/OTM
    strike_offset: float
    expiry_type: str  # CURRENT_WEEK/CURRENT_MONTH
    lots: int
    stop_loss: Optional[float]
    take_profit: Optional[float]
    
class SimplePortfolioModel(BaseModel):
    """Simplified portfolio model"""
    portfolio_name: str
    start_date: date
    end_date: date
    index_name: str = "NIFTY"
    position_size_value: float
    transaction_costs: float
    
class SimplePOSStrategy(BaseModel):
    """Complete simplified strategy"""
    portfolio: SimplePortfolioModel
    legs: List[SimpleLegModel]
    strategy_type: str
```

## Phase 2: HeavyDB Query Implementation (Day 3-4)

### 2.1 Create Working Query Builder

```python
# pos/query_builder_simple.py
class SimplePOSQueryBuilder:
    """Simplified query builder for POS strategies"""
    
    def build_position_query(self, strategy: SimplePOSStrategy) -> str:
        """Build query for multi-leg positions"""
        
        # Build expiry date subquery
        expiry_conditions = []
        for leg in strategy.legs:
            if leg.expiry_type == 'CURRENT_WEEK':
                expiry_conditions.append(f"""
                    MIN(CASE WHEN expiry_date >= trade_date 
                        AND EXTRACT(DOW FROM expiry_date) = 4
                        AND expiry_date < trade_date + INTERVAL '7' DAY
                        THEN expiry_date END) as {leg.leg_name}_expiry
                """)
            elif leg.expiry_type == 'CURRENT_MONTH':
                expiry_conditions.append(f"""
                    MIN(CASE WHEN expiry_date >= trade_date
                        AND EXTRACT(DAY FROM expiry_date) > 20
                        AND expiry_date < trade_date + INTERVAL '35' DAY
                        THEN expiry_date END) as {leg.leg_name}_expiry
                """)
        
        # Build main query
        query = f"""
        WITH daily_spot AS (
            SELECT 
                trade_date,
                MAX(CASE WHEN option_type = 'XX' THEN close_price END) as spot_price
            FROM nifty_option_chain
            WHERE trade_date BETWEEN '{strategy.portfolio.start_date}' 
                AND '{strategy.portfolio.end_date}'
            GROUP BY trade_date
        ),
        expiry_dates AS (
            SELECT 
                trade_date,
                {','.join(expiry_conditions)}
            FROM nifty_option_chain
            WHERE trade_date BETWEEN '{strategy.portfolio.start_date}' 
                AND '{strategy.portfolio.end_date}'
            GROUP BY trade_date
        ),
        """
        
        # Add CTEs for each leg
        leg_ctes = []
        for leg in strategy.legs:
            strike_calc = self._get_strike_calculation(leg)
            
            leg_cte = f"""
        {leg.leg_name}_data AS (
            SELECT 
                oc.trade_date,
                oc.trade_time,
                oc.expiry_date,
                oc.strike_price,
                oc.option_type,
                oc.open_price,
                oc.high_price,
                oc.low_price,
                oc.close_price,
                oc.volume,
                oc.open_interest,
                oc.implied_volatility,
                oc.delta,
                oc.gamma,
                oc.theta,
                oc.vega,
                ds.spot_price,
                ABS(oc.strike_price - {strike_calc}) as strike_distance
            FROM nifty_option_chain oc
            JOIN daily_spot ds ON oc.trade_date = ds.trade_date
            JOIN expiry_dates ed ON oc.trade_date = ed.trade_date
            WHERE oc.option_type = '{leg.option_type}'
            AND oc.expiry_date = ed.{leg.leg_name}_expiry
            AND oc.trade_time = '09:20:00'
        )"""
            leg_ctes.append(leg_cte)
        
        # Join all legs
        query += ',\n'.join(leg_ctes) + ',\n'
        
        # Final selection
        leg_joins = []
        leg_selects = []
        
        for i, leg in enumerate(strategy.legs):
            if i == 0:
                leg_joins.append(f"FROM {leg.leg_name}_data l{i+1}")
            else:
                leg_joins.append(f"""
            JOIN {leg.leg_name}_data l{i+1} 
                ON l1.trade_date = l{i+1}.trade_date 
                AND l1.trade_time = l{i+1}.trade_time""")
            
            leg_selects.extend([
                f"l{i+1}.strike_price as {leg.leg_name}_strike",
                f"l{i+1}.close_price as {leg.leg_name}_price",
                f"l{i+1}.delta as {leg.leg_name}_delta",
                f"l{i+1}.gamma as {leg.leg_name}_gamma",
                f"l{i+1}.theta as {leg.leg_name}_theta",
                f"l{i+1}.vega as {leg.leg_name}_vega"
            ])
        
        query += f"""
        final_positions AS (
            SELECT 
                l1.trade_date,
                l1.trade_time,
                l1.spot_price,
                {','.join(leg_selects)}
            {' '.join(leg_joins)}
            WHERE 1=1
        )
        SELECT * FROM final_positions
        ORDER BY trade_date, trade_time
        """
        
        return query
    
    def _get_strike_calculation(self, leg: SimpleLegModel) -> str:
        """Get strike price calculation based on method"""
        if leg.strike_method == 'ATM':
            return f"ROUND(ds.spot_price / 50) * 50 + {leg.strike_offset}"
        elif leg.strike_method == 'OTM':
            if leg.option_type == 'CE':
                return f"ROUND(ds.spot_price / 50) * 50 + {leg.strike_offset}"
            else:
                return f"ROUND(ds.spot_price / 50) * 50 - {leg.strike_offset}"
        elif leg.strike_method == 'ITM':
            if leg.option_type == 'CE':
                return f"ROUND(ds.spot_price / 50) * 50 - {leg.strike_offset}"
            else:
                return f"ROUND(ds.spot_price / 50) * 50 + {leg.strike_offset}"
        else:
            return "ROUND(ds.spot_price / 50) * 50"
```

### 2.2 Test Query with HeavyDB

```python
# pos/test_heavydb_connection.py
import asyncio
from heavydb import connect
import pandas as pd

async def test_pos_query():
    """Test POS query with real HeavyDB data"""
    
    # Connect to HeavyDB
    conn = connect(
        host='localhost',
        port=6274,
        user='admin',
        password='HyperInteractive',
        dbname='heavyai'
    )
    
    # Create test strategy
    test_strategy = SimplePOSStrategy(
        portfolio=SimplePortfolioModel(
            portfolio_name="Test_Iron_Condor",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            position_size_value=100000,
            transaction_costs=0.001
        ),
        legs=[
            SimpleLegModel(
                leg_id=1,
                leg_name="short_put",
                option_type="PE",
                position_type="SELL",
                strike_method="OTM",
                strike_offset=100,
                expiry_type="CURRENT_WEEK",
                lots=1,
                stop_loss=50,
                take_profit=30
            ),
            SimpleLegModel(
                leg_id=2,
                leg_name="long_put",
                option_type="PE",
                position_type="BUY",
                strike_method="OTM",
                strike_offset=200,
                expiry_type="CURRENT_WEEK",
                lots=1,
                stop_loss=None,
                take_profit=None
            ),
            SimpleLegModel(
                leg_id=3,
                leg_name="short_call",
                option_type="CE",
                position_type="SELL",
                strike_method="OTM",
                strike_offset=100,
                expiry_type="CURRENT_WEEK",
                lots=1,
                stop_loss=50,
                take_profit=30
            ),
            SimpleLegModel(
                leg_id=4,
                leg_name="long_call",
                option_type="CE",
                position_type="BUY",
                strike_method="OTM",
                strike_offset=200,
                expiry_type="CURRENT_WEEK",
                lots=1,
                stop_loss=None,
                take_profit=None
            )
        ],
        strategy_type="IRON_CONDOR"
    )
    
    # Build query
    query_builder = SimplePOSQueryBuilder()
    query = query_builder.build_position_query(test_strategy)
    
    print("Generated Query:")
    print(query[:500] + "...")
    
    # Execute query
    try:
        df = pd.read_sql(query, conn)
        print(f"\nQuery returned {len(df)} rows")
        print("\nFirst few rows:")
        print(df.head())
        print("\nColumns:")
        print(df.columns.tolist())
    except Exception as e:
        print(f"Query execution error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    asyncio.run(test_pos_query())
```

## Phase 3: Results Processing (Day 5-6)

### 3.1 Implement Simple P&L Calculation

```python
# pos/processor_simple.py
class SimplePOSProcessor:
    """Simplified processor for POS strategies"""
    
    def process_results(self, df: pd.DataFrame, strategy: SimplePOSStrategy) -> pd.DataFrame:
        """Process query results into trades"""
        
        trades = []
        
        for idx, row in df.iterrows():
            # Entry trades
            entry_trades = self._create_entry_trades(row, strategy)
            trades.extend(entry_trades)
            
            # Calculate position P&L
            position_pnl = self._calculate_position_pnl(row, strategy)
            
            # Exit trades (simplified - exit at end of day)
            exit_trades = self._create_exit_trades(row, strategy, position_pnl)
            trades.extend(exit_trades)
        
        trades_df = pd.DataFrame(trades)
        
        # Calculate cumulative P&L
        trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
        
        return trades_df
    
    def _create_entry_trades(self, row: pd.Series, strategy: SimplePOSStrategy) -> List[Dict]:
        """Create entry trades for all legs"""
        trades = []
        
        for leg in strategy.legs:
            trade = {
                'trade_date': row['trade_date'],
                'trade_time': row['trade_time'],
                'trade_type': 'ENTRY',
                'leg_name': leg.leg_name,
                'option_type': leg.option_type,
                'position_type': leg.position_type,
                'strike_price': row[f'{leg.leg_name}_strike'],
                'price': row[f'{leg.leg_name}_price'],
                'quantity': leg.lots * 50,  # lots * lot_size
                'premium': row[f'{leg.leg_name}_price'] * leg.lots * 50,
                'transaction_cost': abs(row[f'{leg.leg_name}_price'] * leg.lots * 50 * strategy.portfolio.transaction_costs)
            }
            
            # Adjust premium for buy/sell
            if leg.position_type == 'BUY':
                trade['premium'] = -trade['premium']
                
            trades.append(trade)
            
        return trades
    
    def _calculate_position_pnl(self, row: pd.Series, strategy: SimplePOSStrategy) -> float:
        """Calculate net position P&L"""
        total_pnl = 0
        
        for leg in strategy.legs:
            leg_pnl = 0
            entry_price = row[f'{leg.leg_name}_price']
            
            # For this simple version, assume exit at same price (can be enhanced)
            exit_price = entry_price * 0.8  # Simulate some price movement
            
            if leg.position_type == 'SELL':
                leg_pnl = (entry_price - exit_price) * leg.lots * 50
            else:
                leg_pnl = (exit_price - entry_price) * leg.lots * 50
                
            total_pnl += leg_pnl
            
        return total_pnl
```

## Phase 4: UI Integration (Day 7-8)

### 4.1 Add POS to UI Strategy Selector

```javascript
// Add to index_enterprise.html
<div class="strategy-tab" data-strategy="POS" onclick="selectStrategy('POS')">
    <i class="fas fa-calendar-alt"></i>
    <span>POS</span>
    <small>Positional</small>
</div>

<!-- POS Upload Section -->
<div id="POS-upload" class="upload-section">
    <h5><i class="fas fa-calendar-alt me-2"></i>Positional Strategy Files</h5>
    
    <div class="row">
        <div class="col-md-6">
            <div class="upload-zone" id="posPortfolioUploadZone">
                <i class="fas fa-file-excel fa-3x mb-3"></i>
                <h5>Portfolio Configuration</h5>
                <p>Drop portfolio Excel file here</p>
                <input type="file" id="posPortfolioFile" accept=".xlsx" style="display:none;">
            </div>
        </div>
        <div class="col-md-6">
            <div class="upload-zone" id="posStrategyUploadZone">
                <i class="fas fa-file-excel fa-3x mb-3"></i>
                <h5>Strategy Configuration</h5>
                <p>Drop strategy Excel file here</p>
                <input type="file" id="posStrategyFile" accept=".xlsx" style="display:none;">
            </div>
        </div>
    </div>
    
    <div class="mt-3">
        <h6>Strategy Templates:</h6>
        <div class="btn-group">
            <a href="/api/v1/templates/pos_iron_condor" class="btn btn-sm btn-outline-primary">
                <i class="fas fa-download"></i> Iron Condor
            </a>
            <a href="/api/v1/templates/pos_calendar_spread" class="btn btn-sm btn-outline-primary">
                <i class="fas fa-download"></i> Calendar Spread
            </a>
            <a href="/api/v1/templates/pos_iron_fly" class="btn btn-sm btn-outline-primary">
                <i class="fas fa-download"></i> Iron Fly
            </a>
        </div>
    </div>
</div>
```

### 4.2 Create API Endpoint

```python
# server/app/api/routes/pos.py
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from typing import List
import tempfile
import os

router = APIRouter(prefix="/pos", tags=["pos"])

@router.post("/submit")
async def submit_pos_backtest(
    portfolio_file: UploadFile = File(...),
    strategy_file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Submit POS strategy backtest"""
    
    # Save uploaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        portfolio_path = os.path.join(temp_dir, portfolio_file.filename)
        strategy_path = os.path.join(temp_dir, strategy_file.filename)
        
        with open(portfolio_path, "wb") as f:
            f.write(await portfolio_file.read())
        with open(strategy_path, "wb") as f:
            f.write(await strategy_file.read())
        
        # Initialize strategy
        from backtester_v2.strategies.pos import POSStrategy
        pos_strategy = POSStrategy()
        
        # Parse input
        input_data = {
            'portfolio_file': portfolio_path,
            'strategy_file': strategy_path
        }
        
        try:
            parsed_data = pos_strategy.parse_input(input_data)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Create backtest job
        job_id = str(uuid.uuid4())
        
        # Queue for execution
        await queue_manager.add_job({
            'job_id': job_id,
            'user_id': current_user.id,
            'strategy_type': 'POS',
            'config': parsed_data,
            'status': 'queued'
        })
        
        return {
            'job_id': job_id,
            'status': 'queued',
            'message': 'POS backtest submitted successfully'
        }

@router.get("/templates/{template_name}")
async def get_pos_template(template_name: str):
    """Get POS strategy template"""
    
    templates = {
        'pos_iron_condor': 'templates/pos_iron_condor.xlsx',
        'pos_calendar_spread': 'templates/pos_calendar_spread.xlsx',
        'pos_iron_fly': 'templates/pos_iron_fly.xlsx'
    }
    
    if template_name not in templates:
        raise HTTPException(status_code=404, detail="Template not found")
    
    return FileResponse(templates[template_name], filename=f"{template_name}.xlsx")
```

## Phase 5: Testing & Validation (Day 9-10)

### 5.1 Create Unit Tests

```python
# tests/test_pos_strategy.py
import pytest
from datetime import date
import pandas as pd
from backtester_v2.strategies.pos import POSStrategy, SimplePOSStrategy

def test_parser_portfolio():
    """Test portfolio parsing"""
    strategy = POSStrategy()
    
    # Create test data
    portfolio_df = pd.DataFrame({
        'StartDate': ['2024-01-01'],
        'EndDate': ['2024-12-31'],
        'PortfolioName': ['Test_Portfolio'],
        'Multiplier': [1],
        'SlippagePercent': [0.1]
    })
    
    # Test parsing
    result = strategy.parser.parse_portfolio_dataframe(portfolio_df)
    
    assert result['portfolio_name'] == 'Test_Portfolio'
    assert result['start_date'] == date(2024, 1, 1)
    assert result['transaction_costs'] == 0.001

def test_query_builder():
    """Test query generation"""
    from backtester_v2.strategies.pos.query_builder_simple import SimplePOSQueryBuilder
    
    builder = SimplePOSQueryBuilder()
    
    # Create test strategy
    strategy = SimplePOSStrategy(
        portfolio=SimplePortfolioModel(
            portfolio_name="Test",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            position_size_value=100000,
            transaction_costs=0.001
        ),
        legs=[
            SimpleLegModel(
                leg_id=1,
                leg_name="test_leg",
                option_type="CE",
                position_type="BUY",
                strike_method="ATM",
                strike_offset=0,
                expiry_type="CURRENT_WEEK",
                lots=1
            )
        ],
        strategy_type="TEST"
    )
    
    query = builder.build_position_query(strategy)
    
    assert "WITH daily_spot AS" in query
    assert "test_leg_data AS" in query
    assert "trade_date BETWEEN '2024-01-01' AND '2024-01-31'" in query

def test_processor_pnl():
    """Test P&L calculation"""
    from backtester_v2.strategies.pos.processor_simple import SimplePOSProcessor
    
    processor = SimplePOSProcessor()
    
    # Create test data
    test_data = pd.DataFrame({
        'trade_date': [date(2024, 1, 1)],
        'trade_time': ['09:20:00'],
        'spot_price': [20000],
        'test_leg_strike': [20000],
        'test_leg_price': [100],
        'test_leg_delta': [0.5]
    })
    
    strategy = SimplePOSStrategy(
        portfolio=SimplePortfolioModel(
            portfolio_name="Test",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            position_size_value=100000,
            transaction_costs=0.001
        ),
        legs=[
            SimpleLegModel(
                leg_id=1,
                leg_name="test_leg",
                option_type="CE",
                position_type="BUY",
                strike_method="ATM",
                strike_offset=0,
                expiry_type="CURRENT_WEEK",
                lots=1
            )
        ],
        strategy_type="TEST"
    )
    
    result = processor.process_results(test_data, strategy)
    
    assert len(result) > 0
    assert 'pnl' in result.columns
    assert 'cumulative_pnl' in result.columns
```

### 5.2 Integration Test Script

```python
# test_pos_integration.py
import asyncio
from datetime import date
from backtester_v2.strategies.pos import POSStrategy

async def test_full_integration():
    """Test complete POS strategy flow"""
    
    # Initialize strategy
    strategy = POSStrategy(config={
        'table_name': 'nifty_option_chain',
        'use_cache': False
    })
    
    # Test input files
    input_data = {
        'portfolio_file': 'input_sheets/pos/input_positional_portfolio.xlsx',
        'strategy_file': 'input_sheets/pos/input_iron_fly_strategy.xlsx',
        'start_date': '2024-01-01',
        'end_date': '2024-01-31'
    }
    
    try:
        # Parse input
        print("1. Parsing input files...")
        parsed = strategy.parse_input(input_data)
        print(f"   ✓ Parsed successfully: {parsed['portfolio']['portfolio_name']}")
        
        # Build queries
        print("2. Building queries...")
        queries = strategy.build_queries()
        print(f"   ✓ Generated {len(queries)} queries")
        
        # Execute queries
        print("3. Executing queries...")
        results = await strategy.execute_queries()
        print(f"   ✓ Query returned {len(results[0])} rows")
        
        # Process results
        print("4. Processing results...")
        final_results = strategy.process_results(results)
        print(f"   ✓ Generated {len(final_results['trades'])} trades")
        
        # Calculate metrics
        print("5. Metrics:")
        metrics = final_results['metrics']
        print(f"   - Total P&L: {metrics.get('total_pnl', 0):,.2f}")
        print(f"   - Win Rate: {metrics.get('win_rate', 0):.2%}")
        print(f"   - Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        
        return final_results
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(test_full_integration())
```

## Implementation Timeline

### Week 1
- **Day 1-2**: Fix parser and create simplified models
- **Day 3-4**: Implement HeavyDB queries and test
- **Day 5**: Complete basic processor

### Week 2  
- **Day 6-7**: Add UI integration
- **Day 8-9**: Create comprehensive tests
- **Day 10**: Bug fixes and optimization

### Week 3
- **Day 11-12**: Add advanced features (Greeks, adjustments)
- **Day 13-14**: Performance optimization
- **Day 15**: Documentation and deployment

## Key Success Metrics

1. **Parser Success**: Can parse actual input files without errors
2. **Query Execution**: Returns data from HeavyDB for date range
3. **P&L Accuracy**: Calculated P&L matches manual calculations
4. **UI Integration**: Can submit backtest from UI
5. **Performance**: Process 1 year of data in < 30 seconds

## Risk Mitigation

1. **Data Availability**: Test with small date ranges first
2. **Query Performance**: Add indexes if queries are slow
3. **Memory Usage**: Process data in chunks if needed
4. **Error Handling**: Add comprehensive logging
5. **Validation**: Compare results with manual calculations

This implementation plan provides a clear path to get POS strategy working with real HeavyDB data within 2-3 weeks.