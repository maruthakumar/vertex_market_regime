"""
Archive Compatibility Adapter for Indicator Strategy
Ensures modern implementation follows archive patterns
"""

import json
from typing import Dict, List, Any, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class ArchiveIndicatorAdapter:
    """Adapts modern indicator system to follow archive patterns"""
    
    # Archive indicator mappings
    ARCHIVE_INDICATOR_MAP = {
        'rsi': 'RSI',
        'ema': 'EMA', 
        'vwap': 'VWAP',
        'st': 'SUPERTREND',
        'vol_ema': 'VOLUME_EMA',
        'vol_sma': 'VOLUME_SMA',
        'sma': 'SMA',
        'bb': 'BBANDS',
        'macd': 'MACD',
        'stoch': 'STOCH'
    }
    
    def __init__(self):
        self.validation_errors = []
        
    def convert_archive_format(self, archive_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert archive JSON format to modern format
        
        Archive format:
        {
            "name": "rsi",
            "length": 14,
            "condition": "rsi < 30"
        }
        
        Modern format:
        {
            "IndicatorName": "RSI",
            "IndicatorType": "TALIB",
            "Period": 14,
            "Signal_Type": "THRESHOLD",
            "Threshold_Lower": 30
        }
        """
        modern_config = {}
        
        # Map indicator name
        archive_name = archive_config.get('name', '').lower()
        if archive_name in self.ARCHIVE_INDICATOR_MAP:
            modern_config['IndicatorName'] = self.ARCHIVE_INDICATOR_MAP[archive_name]
            modern_config['IndicatorType'] = 'TALIB'
        else:
            # Custom indicator
            modern_config['IndicatorName'] = archive_name.upper()
            modern_config['IndicatorType'] = 'CUSTOM'
            
        # Map period/length
        if 'length' in archive_config:
            modern_config['Period'] = archive_config['length']
            
        # Parse condition
        if 'condition' in archive_config:
            self._parse_archive_condition(archive_config['condition'], modern_config)
            
        # SuperTrend specific
        if archive_name == 'st' and 'multiplier' in archive_config:
            modern_config['Multiplier'] = archive_config['multiplier']
            
        return modern_config
        
    def _parse_archive_condition(self, condition: str, config: Dict[str, Any]):
        """Parse archive-style condition string"""
        # Examples:
        # "rsi < 30" -> Threshold_Lower = 30
        # "close > ema" -> Signal_Type = CROSSOVER
        # "vwap > close" -> Signal_Type = ABOVE
        
        condition = condition.strip()
        
        # Simple threshold conditions
        if '<' in condition:
            parts = condition.split('<')
            if len(parts) == 2:
                try:
                    threshold = float(parts[1].strip())
                    config['Threshold_Lower'] = threshold
                    config['Signal_Type'] = 'THRESHOLD'
                except ValueError:
                    pass
                    
        elif '>' in condition:
            parts = condition.split('>')
            if len(parts) == 2:
                try:
                    threshold = float(parts[1].strip())
                    config['Threshold_Upper'] = threshold
                    config['Signal_Type'] = 'THRESHOLD'
                except ValueError:
                    # It's a comparison between indicators
                    config['Signal_Type'] = 'CROSSOVER'
                    
    def create_archive_compatible_json(self, modern_config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert modern config back to archive JSON format for compatibility"""
        archive_json = {
            "strategy_type": "indicator",
            "indicators": {
                "entry": [],
                "exit": []
            }
        }
        
        # Convert indicators
        if 'indicators' in modern_config:
            for ind in modern_config['indicators']:
                archive_ind = self._modern_to_archive_indicator(ind)
                if ind.get('Usage', 'ENTRY') == 'ENTRY':
                    archive_json['indicators']['entry'].append(archive_ind)
                else:
                    archive_json['indicators']['exit'].append(archive_ind)
                    
        # Add portfolio settings
        if 'portfolio' in modern_config:
            portfolio = modern_config['portfolio']
            archive_json.update({
                "index_name": portfolio.get('IndexName', 'NIFTY'),
                "start_date": str(portfolio.get('StartDate', '')),
                "end_date": str(portfolio.get('EndDate', '')),
                "start_time": str(portfolio.get('StartTime', '09:20:00')),
                "end_time": str(portfolio.get('EndTime', '15:15:00')),
                "square_off": str(portfolio.get('SquareOffTime', '15:15:00')),
                "capital": portfolio.get('Capital', 1000000)
            })
            
        return archive_json
        
    def _modern_to_archive_indicator(self, modern_ind: Dict[str, Any]) -> Dict[str, Any]:
        """Convert single modern indicator to archive format"""
        # Reverse mapping
        reverse_map = {v: k for k, v in self.ARCHIVE_INDICATOR_MAP.items()}
        
        indicator_name = modern_ind.get('IndicatorName', '')
        archive_name = reverse_map.get(indicator_name, indicator_name.lower())
        
        archive_ind = {
            "name": archive_name,
            "length": modern_ind.get('Period', 14)
        }
        
        # Build condition string
        signal_type = modern_ind.get('Signal_Type', '')
        if signal_type == 'THRESHOLD':
            if 'Threshold_Lower' in modern_ind:
                archive_ind['condition'] = f"{archive_name} < {modern_ind['Threshold_Lower']}"
            elif 'Threshold_Upper' in modern_ind:
                archive_ind['condition'] = f"{archive_name} > {modern_ind['Threshold_Upper']}"
        elif signal_type == 'CROSSOVER':
            archive_ind['condition'] = f"close > {archive_name}"
            
        return archive_ind
        
    def validate_archive_compatibility(self, config: Dict[str, Any]) -> List[str]:
        """Validate if config is compatible with archive system"""
        errors = []
        
        # Check supported indicators
        if 'indicators' in config:
            for ind in config['indicators']:
                name = ind.get('IndicatorName', '')
                if name not in ['RSI', 'EMA', 'SMA', 'VWAP', 'SUPERTREND', 
                               'VOLUME_EMA', 'VOLUME_SMA', 'BBANDS', 'MACD', 'STOCH']:
                    if ind.get('IndicatorType') != 'CUSTOM':
                        errors.append(f"Indicator {name} not supported in archive system")
                        
        # Check required fields
        required_fields = ['portfolio', 'indicators']
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
                
        return errors
        
    def get_archive_evaluator_type(self, indicators: List[Dict[str, Any]]) -> str:
        """Determine which archive evaluator to use based on indicators"""
        indicator_names = [ind.get('IndicatorName', '').upper() for ind in indicators]
        
        # Check for specific combinations
        if 'VWAP' in indicator_names and 'EMA' in indicator_names:
            return 'vwap_ema'
        elif 'EMA' in indicator_names and len([n for n in indicator_names if n == 'EMA']) >= 5:
            return 'five_ema'
        elif 'RSI' in indicator_names and ('EMA' in indicator_names or 'SUPERTREND' in indicator_names):
            return 'heiken'
        else:
            return 'generic'


def create_archive_compatible_input(excel_file: str, output_json: str):
    """
    Create archive-compatible JSON from modern Excel input
    
    Args:
        excel_file: Path to modern indicator Excel file
        output_json: Path to save archive-compatible JSON
    """
    adapter = ArchiveIndicatorAdapter()
    
    # Read Excel file
    xl = pd.ExcelFile(excel_file)
    
    config = {
        'portfolio': {},
        'indicators': []
    }
    
    # Parse sheets
    if 'Portfolio' in xl.sheet_names:
        portfolio_df = pd.read_excel(xl, 'Portfolio')
        config['portfolio'] = portfolio_df.to_dict('records')[0] if len(portfolio_df) > 0 else {}
        
    if 'IndicatorConfiguration' in xl.sheet_names:
        indicator_df = pd.read_excel(xl, 'IndicatorConfiguration')
        config['indicators'] = indicator_df.to_dict('records')
        
    # Convert to archive format
    archive_json = adapter.create_archive_compatible_json(config)
    
    # Save JSON
    with open(output_json, 'w') as f:
        json.dump(archive_json, f, indent=2)
        
    logger.info(f"Created archive-compatible JSON at {output_json}")
    
    return archive_json