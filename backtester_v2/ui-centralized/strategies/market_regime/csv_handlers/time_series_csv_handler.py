"""
Time Series CSV Output Handler
==============================

This module handles CSV output generation for market regime analysis
with proper parameter inclusion and time series formatting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import json
import logging

logger = logging.getLogger(__name__)

class TimeSeriesCSVHandler:
    """Handles CSV output generation for market regime time series data"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_csv(
        self,
        data: pd.DataFrame,
        parameters: Dict[str, Any],
        filename_prefix: str = "market_regime"
    ) -> str:
        """
        Generate CSV file with time series data and parameters
        
        Args:
            data: DataFrame with regime analysis results
            parameters: Input parameters used for analysis
            filename_prefix: Prefix for output filename
            
        Returns:
            Path to generated CSV file
        """
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{parameters.get('symbol', 'NIFTY')}_{timestamp}.csv"
        filepath = self.output_dir / filename
        
        # Create enhanced dataframe with parameters
        enhanced_df = data.copy()
        
        # Add parameter columns
        for key, value in parameters.items():
            if isinstance(value, (list, dict)):
                enhanced_df[f'param_{key}'] = json.dumps(value)
            else:
                enhanced_df[f'param_{key}'] = value
        
        # Add metadata columns
        enhanced_df['analysis_timestamp'] = datetime.now()
        enhanced_df['version'] = '2.0'
        
        # Format numeric columns
        numeric_columns = enhanced_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            enhanced_df[col] = enhanced_df[col].round(4)
        
        # Save to CSV with proper formatting
        enhanced_df.to_csv(
            filepath,
            index=False,
            date_format='%Y-%m-%d %H:%M:%S'
        )
        
        # Also create a metadata file
        metadata = {
            'parameters': parameters,
            'analysis_date': datetime.now().isoformat(),
            'row_count': len(data),
            'columns': list(data.columns),
            'file_path': str(filepath)
        }
        
        metadata_path = filepath.with_suffix('.meta.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ CSV generated: {filepath}")
        logger.info(f"✅ Metadata saved: {metadata_path}")
        
        return str(filepath)
    
    def generate_summary_csv(
        self,
        regime_data: List[Dict[str, Any]],
        output_filename: str = "regime_summary.csv"
    ) -> str:
        """Generate summary CSV with regime statistics"""
        # Convert to DataFrame
        df = pd.DataFrame(regime_data)
        
        # Calculate summary statistics
        summary_stats = df.groupby('regime_name').agg({
            'confidence_score': ['mean', 'std', 'count'],
            'final_score': ['mean', 'std'],
            'timestamp': ['min', 'max']
        })
        
        # Flatten column names
        summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
        
        # Save summary
        summary_path = self.output_dir / output_filename
        summary_stats.to_csv(summary_path)
        
        logger.info(f"✅ Summary CSV generated: {summary_path}")
        return str(summary_path)
