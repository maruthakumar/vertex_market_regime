"""
Market Regime API Routes
=======================

This module defines API endpoints for the market regime strategy.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime

from .archive_enhanced_modules_do_not_use.enhanced_market_regime_engine import EnhancedMarketRegimeEngine
from .archive_enhanced_modules_do_not_use.enhanced_configurable_excel_manager import EnhancedConfigurableExcelManager
from .time_series_regime_storage import TimeSeriesRegimeStorage

router = APIRouter(prefix="/api/strategies/market-regime", tags=["market-regime"])

@router.post("/analyze")
async def analyze_market_regime(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    config_file: Optional[UploadFile] = File(None)
):
    """Analyze market regime for given parameters"""
    try:
        # Initialize engine
        if config_file:
            # Save uploaded config
            config_path = f"temp_config_{datetime.now().timestamp()}.xlsx"
            with open(config_path, "wb") as f:
                f.write(await config_file.read())
            engine = EnhancedMarketRegimeEngine(config_path)
        else:
            engine = EnhancedMarketRegimeEngine()
        
        # Run analysis
        results = engine.analyze_comprehensive_market_regime(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        return {"status": "success", "results": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config")
async def get_configuration():
    """Get current market regime configuration"""
    try:
        manager = EnhancedConfigurableExcelManager()
        config = manager.get_current_configuration()
        return {"status": "success", "config": config}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/export")
async def export_results(
    format: str = "csv",
    include_params: bool = True
):
    """Export market regime analysis results"""
    try:
        storage = TimeSeriesRegimeStorage()
        
        if format == "csv":
            file_path = storage.export_to_csv(include_params=include_params)
        elif format == "excel":
            file_path = storage.export_to_excel(include_params=include_params)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return {"status": "success", "file_path": file_path}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
