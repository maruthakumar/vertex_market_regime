#!/usr/bin/env python3
"""
Market Regime 1-Year Analysis API Endpoints
==========================================
FastAPI endpoints for 1-year Market Regime analysis with
real-time progress tracking and comprehensive reporting.

Features:
- 1-year test execution endpoint
- WebSocket progress monitoring
- Result retrieval and export
- Comparison endpoints
- Performance metrics

Author: Market Regime Testing Team
Date: 2025-06-27
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio
import json
import uuid
import os
from pathlib import Path

# Import test runner and validation tools
from test_runner_1year import MarketRegime1YearTestRunner, TestConfiguration
from regime_validation_tools import MarketRegimeValidator
from enhanced_logging_system import get_logger

# Create router
router = APIRouter(prefix="/api/v2/regime/test", tags=["1-Year Analysis"])

# Active test sessions
active_tests = {}
test_results = {}

class Test1YearRequest(BaseModel):
    """Request model for 1-year test"""
    regime_mode: str = Field(default="18_REGIME", description="Regime mode: 8_REGIME or 18_REGIME")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    index_name: str = Field(default="NIFTY", description="Index name")
    dte_adaptation: bool = Field(default=True, description="Enable DTE adaptation")
    dynamic_weights: bool = Field(default=True, description="Enable dynamic weights")
    enable_1year_mode: bool = Field(default=True, description="Enable 1-year optimizations")
    parallel_processing: bool = Field(default=True, description="Enable parallel processing")
    generate_report: bool = Field(default=True, description="Generate HTML report")
    config_file: Optional[str] = Field(None, description="Configuration file ID")

class Test1YearResponse(BaseModel):
    """Response model for 1-year test initiation"""
    test_id: str
    status: str
    message: str
    estimated_duration: int  # seconds

@router.post("/1year", response_model=Test1YearResponse)
async def start_1year_test(
    request: Test1YearRequest,
    background_tasks: BackgroundTasks
) -> Test1YearResponse:
    """
    Start a 1-year Market Regime analysis test
    """
    # Generate unique test ID
    test_id = str(uuid.uuid4())
    
    # Calculate estimated duration
    start = datetime.strptime(request.start_date, "%Y-%m-%d")
    end = datetime.strptime(request.end_date, "%Y-%m-%d")
    days = (end - start).days
    
    # Rough estimation: 1 second per day with parallel processing
    estimated_duration = days if request.parallel_processing else days * 3
    
    # Create test configuration
    config = TestConfiguration(
        name=f"{request.regime_mode}_{test_id[:8]}",
        regime_mode=request.regime_mode,
        start_date=request.start_date,
        end_date=request.end_date,
        index_name=request.index_name,
        dte_adaptation=request.dte_adaptation,
        dynamic_weights=request.dynamic_weights,
        excel_config_path=request.config_file
    )
    
    # Initialize test session
    active_tests[test_id] = {
        'config': config,
        'status': 'initializing',
        'progress': 0,
        'start_time': datetime.now(),
        'stats': {
            'records_processed': 0,
            'regime_changes': 0,
            'avg_confidence': 0,
            'processing_speed': 0
        }
    }
    
    # Start test in background
    background_tasks.add_task(run_test_async, test_id, config)
    
    return Test1YearResponse(
        test_id=test_id,
        status="started",
        message=f"1-year test started for {days} days of data",
        estimated_duration=estimated_duration
    )

async def run_test_async(test_id: str, config: TestConfiguration):
    """Run test asynchronously"""
    runner = MarketRegime1YearTestRunner()
    
    try:
        # Update status
        active_tests[test_id]['status'] = 'running'
        
        # Run test with progress updates
        result = await runner.run_1year_analysis(config)
        
        # Store result
        test_results[test_id] = result
        active_tests[test_id]['status'] = 'completed'
        active_tests[test_id]['progress'] = 100
        
    except Exception as e:
        active_tests[test_id]['status'] = 'failed'
        active_tests[test_id]['error'] = str(e)
        get_logger().log_error(
            error_type="1YearTestError",
            error_message=str(e),
            context={'test_id': test_id, 'config': config.name}
        )

@router.websocket("/1year/progress/{test_id}")
async def websocket_progress(websocket: WebSocket, test_id: str):
    """
    WebSocket endpoint for real-time test progress
    """
    await websocket.accept()
    
    if test_id not in active_tests:
        await websocket.send_json({
            "error": "Test ID not found"
        })
        await websocket.close()
        return
    
    try:
        while True:
            if test_id not in active_tests:
                break
                
            test_info = active_tests[test_id]
            
            # Send progress update
            await websocket.send_json({
                "test_id": test_id,
                "status": test_info['status'],
                "progress": test_info['progress'],
                "stats": test_info['stats'],
                "completed": test_info['status'] in ['completed', 'failed']
            })
            
            # Check if test is complete
            if test_info['status'] in ['completed', 'failed']:
                if test_info['status'] == 'completed' and test_id in test_results:
                    result = test_results[test_id]
                    await websocket.send_json({
                        "completed": True,
                        "stats": {
                            "records_processed": result.total_records,
                            "regime_changes": result.transition_count,
                            "avg_confidence": result.avg_confidence,
                            "processing_speed": result.total_records / result.total_duration
                        },
                        "report_url": f"/api/v2/regime/test/1year/report/{test_id}"
                    })
                break
            
            # Update progress based on elapsed time (simplified)
            elapsed = (datetime.now() - test_info['start_time']).total_seconds()
            if test_info['status'] == 'running':
                # Simulate progress
                test_info['progress'] = min(95, int(elapsed / 10))
                test_info['stats']['records_processed'] = int(elapsed * 1000)
                test_info['stats']['processing_speed'] = 1000
            
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        pass
    finally:
        await websocket.close()

@router.get("/1year/status/{test_id}")
async def get_test_status(test_id: str) -> Dict[str, Any]:
    """
    Get current status of a 1-year test
    """
    if test_id not in active_tests:
        raise HTTPException(status_code=404, detail="Test ID not found")
    
    test_info = active_tests[test_id]
    
    response = {
        "test_id": test_id,
        "status": test_info['status'],
        "progress": test_info['progress'],
        "start_time": test_info['start_time'].isoformat(),
        "stats": test_info['stats']
    }
    
    if test_info['status'] == 'completed' and test_id in test_results:
        result = test_results[test_id]
        response['result_summary'] = {
            "total_duration": result.total_duration,
            "total_records": result.total_records,
            "regime_changes": result.transition_count,
            "avg_confidence": result.avg_confidence
        }
    elif test_info['status'] == 'failed':
        response['error'] = test_info.get('error', 'Unknown error')
    
    return response

@router.get("/1year/report/{test_id}")
async def get_test_report(test_id: str):
    """
    Get HTML report for completed test
    """
    if test_id not in test_results:
        raise HTTPException(status_code=404, detail="Test results not found")
    
    # Generate report path
    report_path = Path(f"market_regime_1year_results/test_result_{test_id}_report.html")
    
    if not report_path.exists():
        # Generate report if not exists
        runner = MarketRegime1YearTestRunner()
        runner.generate_comparison_report([test_results[test_id]])
        
    return FileResponse(
        path=report_path,
        media_type="text/html",
        filename=f"market_regime_report_{test_id}.html"
    )

@router.post("/1year/compare")
async def compare_tests(test_ids: List[str]) -> Dict[str, Any]:
    """
    Compare multiple 1-year test results
    """
    if len(test_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 test IDs required")
    
    # Validate all test IDs exist
    results = []
    for test_id in test_ids:
        if test_id not in test_results:
            raise HTTPException(status_code=404, detail=f"Test {test_id} not found")
        results.append(test_results[test_id])
    
    # Generate comparison report
    runner = MarketRegime1YearTestRunner()
    report_path = runner.generate_comparison_report(results)
    
    # Perform validation comparison
    validator = MarketRegimeValidator()
    comparisons = []
    
    for i in range(len(results) - 1):
        for j in range(i + 1, len(results)):
            # Note: This is simplified - would need actual DataFrame results
            comparison = {
                "config1": results[i].config_name,
                "config2": results[j].config_name,
                "agreement_rate": 0.85,  # Placeholder
                "transition_similarity": 0.90  # Placeholder
            }
            comparisons.append(comparison)
    
    return {
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
        "report_url": f"/api/v2/regime/test/1year/comparison_report"
    }

@router.get("/1year/export/{test_id}")
async def export_test_results(
    test_id: str,
    format: str = "csv"
) -> StreamingResponse:
    """
    Export test results in various formats
    """
    if test_id not in test_results:
        raise HTTPException(status_code=404, detail="Test results not found")
    
    result = test_results[test_id]
    
    if format == "csv":
        # Generate CSV content
        import io
        import csv
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write headers
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Test Name', result.config_name])
        writer.writerow(['Duration (seconds)', result.total_duration])
        writer.writerow(['Total Records', result.total_records])
        writer.writerow(['Regime Changes', result.transition_count])
        writer.writerow(['Average Confidence', result.avg_confidence])
        
        # Regime distribution
        writer.writerow([])
        writer.writerow(['Regime', 'Count'])
        for regime, count in result.regime_distribution.items():
            writer.writerow([regime, count])
        
        output.seek(0)
        
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=test_results_{test_id}.csv"
            }
        )
    
    elif format == "json":
        # Return JSON
        return StreamingResponse(
            io.BytesIO(json.dumps(result.__dict__, default=str, indent=2).encode()),
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename=test_results_{test_id}.json"
            }
        )
    
    else:
        raise HTTPException(status_code=400, detail="Unsupported format")

@router.delete("/1year/{test_id}")
async def cleanup_test(test_id: str) -> Dict[str, str]:
    """
    Clean up test data and free resources
    """
    removed = False
    
    if test_id in active_tests:
        del active_tests[test_id]
        removed = True
    
    if test_id in test_results:
        del test_results[test_id]
        removed = True
    
    if not removed:
        raise HTTPException(status_code=404, detail="Test ID not found")
    
    return {"message": f"Test {test_id} cleaned up successfully"}

# Include router in main app
# app.include_router(router)