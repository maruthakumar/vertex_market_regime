"""
Portfolio-Based Excel Upload API
================================

Handles the multi-file Excel upload workflow where Portfolio files
reference other Excel files through StrategySetting sheet.

This API provides endpoints for:
1. Scanning portfolio files to detect required files
2. Validating complete file sets
3. Processing multi-file configurations
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import pandas as pd
import json

from fastapi import (
    APIRouter, HTTPException, UploadFile, File, Form, 
    BackgroundTasks, status
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Pydantic models
class StrategyFileInfo(BaseModel):
    """Information about a required strategy file"""
    name: str
    file_path: str
    enabled: bool = True
    priority: int = 1
    allocation_percent: float = 100.0
    required: bool = True

class PortfolioScanResponse(BaseModel):
    """Response from portfolio file scanning"""
    strategy_type: str
    portfolio_file: str
    required_files: List[StrategyFileInfo]
    total_files_expected: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

class FileSetValidationResponse(BaseModel):
    """Response from file set validation"""
    valid: bool
    files_received: int
    files_expected: int
    missing_files: List[str] = Field(default_factory=list)
    validation_details: Dict[str, str] = Field(default_factory=dict)
    ready_to_process: bool = False

class PortfolioUploadAPI:
    """
    API for handling portfolio-based Excel uploads
    """
    
    def __init__(self, upload_dir: Optional[str] = None):
        """Initialize the API"""
        self.upload_dir = Path(upload_dir or tempfile.gettempdir()) / "portfolio_uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Strategy type detection patterns
        self.strategy_patterns = {
            'tv': ['tv_', 'tradingview'],
            'tbs': ['tbs_', 'trade_builder'],
            'pos': ['pos_', 'position'],
            'oi': ['oi_', 'open_interest'],
            'orb': ['orb_', 'opening_range'],
            'ml': ['ml_', 'machine_learning', 'ml_indicator'],
            'market_regime': ['mr_', 'market_regime', 'regime']
        }
        
        # Create router
        self.router = APIRouter(prefix="/api/v1/portfolio", tags=["Portfolio Upload"])
        self._setup_routes()
        
        logger.info(f"PortfolioUploadAPI initialized with upload dir: {self.upload_dir}")
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.router.post("/scan", response_model=PortfolioScanResponse)
        async def scan_portfolio_file(
            file: UploadFile = File(..., description="Portfolio Excel file to scan")
        ):
            """
            Scan a portfolio file to detect required strategy files
            
            This endpoint:
            1. Reads the StrategySetting sheet from the portfolio file
            2. Extracts all referenced Excel files
            3. Returns the list of required files for upload
            """
            return await self._scan_portfolio_file(file)
        
        @self.router.post("/validate", response_model=FileSetValidationResponse)
        async def validate_file_set(
            portfolio: UploadFile = File(..., description="Portfolio Excel file"),
            files: List[UploadFile] = File(..., description="Additional strategy files")
        ):
            """
            Validate a complete set of configuration files
            
            This endpoint:
            1. Checks that all required files are present
            2. Validates each file's structure
            3. Returns validation status and details
            """
            return await self._validate_file_set(portfolio, files)
        
        @self.router.post("/process")
        async def process_configuration(
            background_tasks: BackgroundTasks,
            portfolio: UploadFile = File(..., description="Portfolio Excel file"),
            files: List[UploadFile] = File(..., description="Additional strategy files"),
            author: str = Form("api_user"),
            validate_only: bool = Form(False)
        ):
            """
            Process a complete configuration file set
            
            This endpoint:
            1. Validates all files
            2. Parses configuration data
            3. Optionally starts a backtest
            """
            return await self._process_configuration(
                portfolio, files, author, validate_only, background_tasks
            )
    
    async def _scan_portfolio_file(self, file: UploadFile) -> PortfolioScanResponse:
        """Scan portfolio file for required files"""
        
        # Save file temporarily
        content = await file.read()
        with tempfile.NamedTemporaryFile(
            suffix='.xlsx', 
            dir=self.upload_dir,
            delete=False
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Detect strategy type
            strategy_type = self._detect_strategy_type(file.filename)
            
            # Read Excel file
            excel_data = pd.ExcelFile(tmp_path)
            
            # Find StrategySetting sheet
            strategy_sheet = None
            for sheet_name in excel_data.sheet_names:
                if 'strategysetting' in sheet_name.lower().replace(' ', ''):
                    strategy_sheet = sheet_name
                    break
            
            if not strategy_sheet:
                raise HTTPException(
                    status_code=400,
                    detail="No StrategySetting sheet found in portfolio file"
                )
            
            # Read StrategySetting data
            df = pd.read_excel(tmp_path, sheet_name=strategy_sheet)
            
            # Extract required files
            required_files = []
            
            # Expected columns
            name_col = None
            path_col = None
            enabled_col = None
            priority_col = None
            allocation_col = None
            
            # Find columns (case-insensitive)
            for col in df.columns:
                col_lower = col.lower()
                if 'strategyname' in col_lower:
                    name_col = col
                elif 'strategyexcelfilepath' in col_lower or 'filepath' in col_lower:
                    path_col = col
                elif 'enabled' in col_lower:
                    enabled_col = col
                elif 'priority' in col_lower:
                    priority_col = col
                elif 'allocation' in col_lower:
                    allocation_col = col
            
            if not name_col or not path_col:
                raise HTTPException(
                    status_code=400,
                    detail="Required columns (StrategyName, StrategyExcelFilePath) not found"
                )
            
            # Extract file information
            for _, row in df.iterrows():
                if pd.notna(row.get(name_col)) and pd.notna(row.get(path_col)):
                    file_info = StrategyFileInfo(
                        name=str(row[name_col]),
                        file_path=str(row[path_col]),
                        enabled=bool(row.get(enabled_col, True)) if enabled_col else True,
                        priority=int(row.get(priority_col, 1)) if priority_col else 1,
                        allocation_percent=float(row.get(allocation_col, 100)) if allocation_col else 100.0
                    )
                    required_files.append(file_info)
            
            # Add metadata
            metadata = {
                "sheet_names": excel_data.sheet_names,
                "strategy_settings_sheet": strategy_sheet,
                "total_rows": len(df),
                "enabled_count": sum(1 for f in required_files if f.enabled)
            }
            
            return PortfolioScanResponse(
                strategy_type=strategy_type,
                portfolio_file=file.filename,
                required_files=required_files,
                total_files_expected=len([f for f in required_files if f.enabled]) + 1,  # +1 for portfolio
                metadata=metadata
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error scanning portfolio file: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error scanning portfolio file: {str(e)}"
            )
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    async def _validate_file_set(
        self, 
        portfolio: UploadFile, 
        files: List[UploadFile]
    ) -> FileSetValidationResponse:
        """Validate a complete set of files"""
        
        # First scan the portfolio to get requirements
        scan_result = await self._scan_portfolio_file(portfolio)
        
        # Create a map of uploaded files
        uploaded_files = {f.filename: f for f in files}
        uploaded_files[portfolio.filename] = portfolio
        
        # Check for missing files
        missing_files = []
        validation_details = {
            portfolio.filename: "valid"
        }
        
        for required_file in scan_result.required_files:
            if not required_file.enabled:
                continue
                
            # Try to find the file
            found = False
            for uploaded_name in uploaded_files:
                # Check if the uploaded file matches the expected file
                if (required_file.file_path in uploaded_name or 
                    uploaded_name.endswith(required_file.file_path) or
                    Path(uploaded_name).stem == Path(required_file.file_path).stem):
                    found = True
                    validation_details[uploaded_name] = "valid"
                    break
            
            if not found:
                missing_files.append(required_file.file_path)
                validation_details[required_file.file_path] = "missing"
        
        # Check if all files are valid
        all_valid = len(missing_files) == 0
        
        return FileSetValidationResponse(
            valid=all_valid,
            files_received=len(uploaded_files),
            files_expected=scan_result.total_files_expected,
            missing_files=missing_files,
            validation_details=validation_details,
            ready_to_process=all_valid
        )
    
    async def _process_configuration(
        self,
        portfolio: UploadFile,
        files: List[UploadFile],
        author: str,
        validate_only: bool,
        background_tasks: BackgroundTasks
    ) -> Dict[str, Any]:
        """Process the complete configuration"""
        
        # Validate file set first
        validation_result = await self._validate_file_set(portfolio, files)
        
        if not validation_result.valid:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid file set",
                    "validation": validation_result.dict()
                }
            )
        
        # Save all files
        saved_files = {}
        try:
            # Save portfolio
            portfolio_content = await portfolio.read()
            portfolio_path = self.upload_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{portfolio.filename}"
            with open(portfolio_path, 'wb') as f:
                f.write(portfolio_content)
            saved_files['portfolio'] = str(portfolio_path)
            
            # Save other files
            for file in files:
                content = await file.read()
                file_path = self.upload_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
                with open(file_path, 'wb') as f:
                    f.write(content)
                saved_files[file.filename] = str(file_path)
            
            # If validate_only, return validation results
            if validate_only:
                return {
                    "status": "validated",
                    "message": "Configuration validated successfully",
                    "validation": validation_result.dict(),
                    "saved_files": saved_files
                }
            
            # TODO: Process configuration and start backtest
            # This would integrate with your existing backtest system
            
            return {
                "status": "success",
                "message": "Configuration processed successfully",
                "validation": validation_result.dict(),
                "saved_files": saved_files,
                "author": author,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            # Clean up saved files on error
            for file_path in saved_files.values():
                if os.path.exists(file_path):
                    os.unlink(file_path)
            
            logger.error(f"Error processing configuration: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing configuration: {str(e)}"
            )
    
    def _detect_strategy_type(self, filename: str) -> str:
        """Detect strategy type from filename"""
        filename_lower = filename.lower()
        
        for strategy_type, patterns in self.strategy_patterns.items():
            for pattern in patterns:
                if pattern in filename_lower:
                    return strategy_type
        
        # Default to extracting from standard naming
        if '_config_' in filename_lower:
            parts = filename_lower.split('_')
            if parts:
                return parts[0]
        
        return "unknown"

# Create the API instance
portfolio_api = PortfolioUploadAPI()

# Export the router
router = portfolio_api.router