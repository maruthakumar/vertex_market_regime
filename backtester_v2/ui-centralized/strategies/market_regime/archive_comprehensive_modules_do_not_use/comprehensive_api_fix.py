#!/usr/bin/env python3
"""
Comprehensive fix for Market Regime API to ensure ONLY real data is used
"""

import os
import shutil
from datetime import datetime

print("=" * 80)
print("COMPREHENSIVE MARKET REGIME API FIX")
print("=" * 80)

api_file = "/srv/samba/shared/bt/backtester_stable/BTRUN/server/app/api/routes/market_regime_api.py"

# Create a completely new fixed version
fixed_content = '''#!/usr/bin/env python3
"""
Market Regime API Routes - FIXED VERSION
========================================

CRITICAL: This version ONLY uses real HeavyDB data - NO MOCK DATA ALLOWED
"""

import os
import sys
import json
import io
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
import logging

# Setup logging
logger = logging.getLogger(__name__)

# Add paths for market regime modules
sys.path.append('/srv/samba/shared/bt/backtester_stable/BTRUN')
sys.path.append('/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/market_regime')

# Import market regime components
try:
    from backtester_v2.dal.heavydb_connection import get_connection, execute_query
    HEAVYDB_AVAILABLE = True
except ImportError:
    HEAVYDB_AVAILABLE = False
    logger.critical("❌ HeavyDB connection not available - CANNOT PROCEED")
    raise ImportError("HeavyDB is REQUIRED - no mock data allowed")

# Import Excel to YAML converter
try:
    from unified_excel_to_yaml_converter import UnifiedExcelToYAMLConverter as ExcelToYamlConverter
    CONVERTER_AVAILABLE = True
except ImportError:
    CONVERTER_AVAILABLE = False
    logger.warning("Excel to YAML converter not available")

# Import real market regime engines
try:
    from backtester_v2.strategies.market_regime.real_data_integration_engine import RealDataIntegrationEngine
    from backtester_v2.strategies.market_regime.comprehensive_market_regime_analyzer import ComprehensiveMarketRegimeAnalyzer
    REAL_ENGINE_AVAILABLE = True
except ImportError:
    REAL_ENGINE_AVAILABLE = False
    logger.warning("Real market regime engines not available")

# Import adapter for real engines
try:
    from backtester_v2.strategies.market_regime.real_data_engine_adapter import RealDataEngineAdapter
    ADAPTER_AVAILABLE = True
except ImportError:
    ADAPTER_AVAILABLE = False
    logger.warning("Real data engine adapter not available")

router = APIRouter()

# Pydantic models
class ConfigValidationRequest(BaseModel):
    validation_level: str = "comprehensive"
    include_sample_data: bool = False

class RegimeCalculationRequest(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    timeframe: str = "1min"
    include_confidence: bool = True

class CSVGenerationRequest(BaseModel):
    period: str = "1_day"
    format: str = "enhanced"
    include_metadata: bool = True

# Global configuration storage
current_config = {}
yaml_config_cache = {}

# Initialize real data engines
real_data_engine = None
market_regime_analyzer = None
engine_adapter = None

def initialize_real_engines():
    """Initialize real market regime engines - NO FALLBACKS"""
    global real_data_engine, market_regime_analyzer, engine_adapter
    
    if not HEAVYDB_AVAILABLE:
        raise RuntimeError("❌ HeavyDB not available - cannot proceed without real data")
    
    if ADAPTER_AVAILABLE:
        try:
            # Initialize adapter (required)
            engine_adapter = RealDataEngineAdapter(
                real_engine=real_data_engine,
                market_analyzer=market_regime_analyzer
            )
            logger.info("✅ Real data engine adapter initialized")
            
            # Try to initialize optional engines
            if REAL_ENGINE_AVAILABLE:
                try:
                    real_data_engine = RealDataIntegrationEngine()
                    market_regime_analyzer = ComprehensiveMarketRegimeAnalyzer()
                    logger.info("✅ Real engines initialized")
                except Exception as e:
                    logger.warning(f"Real engines not available: {e}")
                    
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize adapter: {e}")
            raise RuntimeError(f"Cannot initialize without adapter: {e}")
    else:
        raise RuntimeError("❌ Adapter not available - cannot proceed")

# Initialize engines on module load
try:
    REAL_ENGINES_INITIALIZED = initialize_real_engines()
except Exception as e:
    logger.critical(f"❌ CRITICAL: Cannot start without real engines: {e}")
    REAL_ENGINES_INITIALIZED = False

@router.post("/upload")
async def upload_config_file(
    background_tasks: BackgroundTasks,
    configFile: UploadFile = File(...)
):
    """Upload and validate Excel configuration file"""
    try:
        if not configFile.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload an Excel file"
            )
        
        content = await configFile.read()
        temp_dir = Path("/tmp/market_regime_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_file = temp_dir / f"config_{timestamp}_{configFile.filename}"
        
        with open(temp_file, "wb") as f:
            f.write(content)
        
        # Convert Excel to YAML
        validation_results = {}
        if CONVERTER_AVAILABLE:
            converter = ExcelToYamlConverter()
            yaml_config = converter.convert_excel_to_yaml(str(temp_file))
            yaml_validation = converter.validate_yaml_structure(yaml_config)
            
            global yaml_config_cache
            yaml_config_cache = yaml_config
            
            validation_results = {
                'excel_validation': {'success': True},
                'yaml_conversion': {
                    'success': 'error' not in yaml_config,
                    'sheets_processed': yaml_config.get('conversion_summary', {}).get('sheets_processed', 0),
                    'success_rate': yaml_config.get('conversion_summary', {}).get('success_rate', 0),
                    'yaml_validation': yaml_validation
                }
            }
        
        global current_config
        current_config = {
            'file_path': str(temp_file),
            'filename': configFile.filename,
            'upload_time': datetime.now().isoformat(),
            'validation_results': validation_results,
            'file_size': len(content)
        }
        
        return JSONResponse({
            "status": "success",
            "message": "Configuration uploaded successfully",
            "validation_results": validation_results,
            "file_info": {
                "filename": configFile.filename,
                "size": len(content),
                "upload_time": current_config['upload_time']
            }
        })
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_market_regime_status():
    """Get current market regime status - REAL DATA ONLY"""
    try:
        # Calculate current regime
        current_regime = await calculate_current_regime()
        
        # Check database status
        db_status = await check_database_status()
        
        health_status = {
            "heavydb_connected": HEAVYDB_AVAILABLE,
            "config_loaded": bool(current_config),
            "yaml_config_available": bool(yaml_config_cache),
            "real_engines_initialized": REAL_ENGINES_INITIALIZED,
            "engine_adapter_available": engine_adapter is not None,
            "last_calculation": datetime.now().isoformat(),
            "system_status": "operational" if REAL_ENGINES_INITIALIZED else "error_no_engines",
            "data_source": "real_heavydb" if current_regime.get('data_source') == 'real_heavydb' else "error_no_real_data"
        }
        health_status.update(db_status)
        
        return JSONResponse({
            "status": "success",
            "current_regime": current_regime,
            "system_health": health_status,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Status check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/calculate")
async def calculate_market_regime(request: RegimeCalculationRequest):
    """Calculate market regime - REAL DATA ONLY"""
    try:
        if not current_config:
            raise HTTPException(
                status_code=400,
                detail="No configuration loaded"
            )
        
        end_date = request.end_date or datetime.now().strftime('%Y-%m-%d')
        start_date = request.start_date or (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        regime_results = await perform_regime_calculation(
            start_date=start_date,
            end_date=end_date,
            timeframe=request.timeframe,
            include_confidence=request.include_confidence
        )
        
        return JSONResponse({
            "status": "success",
            "calculation_period": {
                "start_date": start_date,
                "end_date": end_date,
                "timeframe": request.timeframe
            },
            "regime_results": regime_results,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Calculation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-csv")
async def generate_csv_export(request: CSVGenerationRequest):
    """Generate CSV export from REAL DATA"""
    try:
        if not current_config:
            raise HTTPException(status_code=400, detail="No configuration loaded")
        
        # Must use real data for CSV
        csv_data = await create_real_csv_export(
            period=request.period,
            format=request.format,
            include_metadata=request.include_metadata
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"market_regime_real_{request.period}_{timestamp}.csv"
        
        return StreamingResponse(
            io.BytesIO(csv_data.encode('utf-8')),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        logger.error(f"CSV generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config")
async def get_current_config():
    """Get current configuration summary"""
    if not current_config:
        return JSONResponse({
            "status": "no_config",
            "message": "No configuration loaded",
            "timestamp": datetime.now().isoformat()
        })
    
    return JSONResponse({
        "status": "success",
        "config_summary": {
            "filename": current_config.get('filename'),
            "upload_time": current_config.get('upload_time'),
            "file_size": current_config.get('file_size')
        },
        "timestamp": datetime.now().isoformat()
    })

# Helper functions

async def calculate_current_regime() -> Dict[str, Any]:
    """Calculate current market regime - REAL DATA ONLY"""
    try:
        global engine_adapter, yaml_config_cache
        
        if not HEAVYDB_AVAILABLE:
            raise RuntimeError("HeavyDB not available")
            
        if not engine_adapter:
            raise RuntimeError("Engine adapter not initialized")
        
        # Get HeavyDB connection
        conn = get_connection()
        if not conn:
            raise RuntimeError("Cannot connect to HeavyDB")
        
        # Get latest date first to avoid sorting issues
        max_date_query = "SELECT MAX(trade_date) FROM nifty_option_chain"
        max_date_result = execute_query(conn, max_date_query)
        
        if max_date_result.empty:
            raise RuntimeError("No data in HeavyDB")
            
        max_date = max_date_result.iloc[0][0]
        
        # Get data for latest date
        query = f"""
        SELECT * FROM nifty_option_chain 
        WHERE trade_date = '{max_date}'
        AND atm_strike IS NOT NULL
        LIMIT 1000
        """
        
        latest_data = execute_query(conn, query)
        
        if latest_data.empty:
            raise RuntimeError("No recent data available")
        
        logger.info(f"✅ Retrieved {len(latest_data)} records from HeavyDB")
        
        # Calculate regime using adapter
        regime_result = engine_adapter.calculate_regime_from_data(
            market_data=latest_data,
            config=yaml_config_cache if yaml_config_cache else None
        )
        
        if regime_result and regime_result.get('success', False):
            return {
                "regime": regime_result.get('regime', 'NEUTRAL'),
                "confidence": regime_result.get('confidence', 0.8),
                "sub_regimes": regime_result.get('sub_regimes', {}),
                "indicators": regime_result.get('indicators', {}),
                "data_source": "real_heavydb",
                "data_points_used": len(latest_data),
                "calculation_timestamp": datetime.now().isoformat()
            }
        else:
            raise RuntimeError("Regime calculation failed")
            
    except Exception as e:
        logger.error(f"❌ Regime calculation error: {e}")
        # NO FALLBACK TO SIMULATION - Return error
        return {
            "regime": "ERROR",
            "confidence": 0.0,
            "error": str(e),
            "data_source": "error_no_real_data",
            "calculation_timestamp": datetime.now().isoformat()
        }

async def check_database_status() -> Dict[str, Any]:
    """Check HeavyDB status"""
    if not HEAVYDB_AVAILABLE:
        return {"database_status": "unavailable"}
    
    try:
        conn = get_connection()
        if conn:
            # Simple count query without alias
            result = execute_query(conn, "SELECT COUNT(*) FROM nifty_option_chain")
            
            if not result.empty:
                count = result.iloc[0][0]
                return {
                    "database_status": "connected",
                    "data_available": True,
                    "record_count": count,
                    "last_check": datetime.now().isoformat()
                }
            else:
                return {"database_status": "empty"}
        else:
            return {"database_status": "connection_failed"}
            
    except Exception as e:
        return {
            "database_status": "error",
            "error": str(e)
        }

async def perform_regime_calculation(start_date: str, end_date: str, 
                                   timeframe: str, include_confidence: bool) -> Dict[str, Any]:
    """Perform historical regime calculation - REAL DATA ONLY"""
    try:
        if not HEAVYDB_AVAILABLE or not engine_adapter:
            raise RuntimeError("Real data system not available")
        
        conn = get_connection()
        if not conn:
            raise RuntimeError("Cannot connect to HeavyDB")
        
        # Query historical data
        query = f"""
        SELECT trade_date, trade_time, spot, ce_iv, pe_iv, 
               ce_delta, pe_delta, ce_gamma, pe_gamma, 
               ce_theta, pe_theta, ce_vega, pe_vega,
               strike, dte, ce_close, pe_close
        FROM nifty_option_chain 
        WHERE trade_date >= '{start_date}'
          AND trade_date <= '{end_date}'
        ORDER BY trade_date ASC, trade_time ASC
        LIMIT 10000
        """
        
        historical_data = execute_query(conn, query)
        
        if historical_data.empty:
            raise RuntimeError(f"No data for period {start_date} to {end_date}")
        
        logger.info(f"✅ Retrieved {len(historical_data)} historical records")
        
        # Analyze using adapter
        regime_results = engine_adapter.analyze_time_series(
            market_data=historical_data,
            config=yaml_config_cache,
            timeframe=timeframe,
            include_confidence=include_confidence
        )
        
        if regime_results and regime_results.get('success', False):
            return {
                "time_series": regime_results.get('time_series', []),
                "summary": {
                    "total_points": len(regime_results.get('time_series', [])),
                    "data_period": {
                        "start": start_date,
                        "end": end_date,
                        "timeframe": timeframe
                    },
                    "data_points_analyzed": len(historical_data)
                },
                "data_source": "real_heavydb",
                "calculation_timestamp": datetime.now().isoformat()
            }
        else:
            raise RuntimeError("Time series analysis failed")
            
    except Exception as e:
        logger.error(f"❌ Historical calculation error: {e}")
        return {"error": str(e), "data_source": "error_no_real_data"}

async def create_real_csv_export(period: str, format: str, include_metadata: bool) -> str:
    """Create CSV from REAL DATA only"""
    try:
        # Calculate real regime data for period
        period_days = {
            "1_day": 1, "7_days": 7, "30_days": 30, 
            "90_days": 90, "1_year": 365
        }.get(period, 1)
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=period_days)).strftime('%Y-%m-%d')
        
        # Get real historical data
        regime_data = await perform_regime_calculation(
            start_date=start_date,
            end_date=end_date,
            timeframe="1H" if period_days > 7 else "5min",
            include_confidence=True
        )
        
        if "error" in regime_data:
            raise RuntimeError(regime_data["error"])
        
        # Convert to DataFrame
        time_series = regime_data.get("time_series", [])
        if not time_series:
            raise RuntimeError("No data available for export")
            
        df = pd.DataFrame(time_series)
        
        # Create CSV
        csv_content = ""
        if include_metadata:
            csv_content += f"# Market Regime Analysis Export (REAL DATA)\\n"
            csv_content += f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n"
            csv_content += f"# Period: {period}\\n"
            csv_content += f"# Data Source: HeavyDB (Real Data)\\n"
            csv_content += f"# Records: {len(df)}\\n\\n"
        
        csv_content += df.to_csv(index=False)
        return csv_content
        
    except Exception as e:
        raise Exception(f"CSV generation failed: {str(e)}")

# Add router tags
router.tags = ["market-regime"]
'''

# Backup current file
backup_file = api_file + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
shutil.copy(api_file, backup_file)
print(f"✅ Created backup: {backup_file}")

# Write the fixed version
with open(api_file, 'w') as f:
    f.write(fixed_content)

print("✅ Written comprehensive fix to market_regime_api.py")

# Also create a simple test script
test_script = """#!/usr/bin/env python3
import requests
import json

# Test the API
try:
    response = requests.get("http://localhost:8000/api/market-regime/status")
    data = response.json()
    
    print("=" * 60)
    print("MARKET REGIME API STATUS CHECK")
    print("=" * 60)
    
    regime = data.get('current_regime', {})
    health = data.get('system_health', {})
    
    print(f"\\nCurrent Regime: {regime.get('regime')}")
    print(f"Data Source: {regime.get('data_source')}")
    print(f"Confidence: {regime.get('confidence')}")
    
    print(f"\\nSystem Health:")
    print(f"  HeavyDB Connected: {health.get('heavydb_connected')}")
    print(f"  Database Status: {health.get('database_status')}")
    print(f"  Record Count: {health.get('record_count', 'N/A')}")
    print(f"  System Status: {health.get('system_status')}")
    
    if regime.get('data_source') == 'real_heavydb':
        print("\\n✅ SUCCESS: Using REAL HeavyDB data!")
    else:
        print(f"\\n❌ ERROR: Not using real data - source is {regime.get('data_source')}")
        
except Exception as e:
    print(f"❌ Error testing API: {e}")
"""

test_file = "/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/market_regime/test_real_data_api.py"
with open(test_file, 'w') as f:
    f.write(test_script)
os.chmod(test_file, 0o755)

print(f"✅ Created test script: {test_file}")
print("\nNext steps:")
print("1. Restart the API server completely")
print("2. Run: python3 test_real_data_api.py")
print("3. Verify data_source shows 'real_heavydb'")