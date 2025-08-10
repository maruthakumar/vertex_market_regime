"""
ML Triple Straddle configuration parser

Handles the 26-sheet Excel configuration structure for ML Triple Rolling Straddle strategy
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import logging

from .excel_parser import ExcelParser
from ..core.exceptions import ParsingError

logger = logging.getLogger(__name__)

class MLTripleStraddleParser(ExcelParser):
    """
    Specialized parser for ML Triple Straddle configuration files
    
    This parser handles the complex 26-sheet structure of ML Triple Straddle
    configurations including:
    - 5 ML model configurations (LightGBM, CatBoost, TabNet, LSTM, Transformer)
    - 6 feature engineering categories
    - Risk management settings
    - Signal generation parameters
    - Database connections
    """
    
    def __init__(self):
        """Initialize ML Triple Straddle parser"""
        super().__init__()
        
        # Define expected sheet structure
        self.sheet_mapping = {
            # ML Models (Sheets 1-6)
            'lightgbm_config': 'ml_models.lightgbm',
            'catboost_config': 'ml_models.catboost',
            'tabnet_config': 'ml_models.tabnet',
            'lstm_config': 'ml_models.lstm',
            'transformer_config': 'ml_models.transformer',
            'ensemble_config': 'ml_models.ensemble',
            
            # Feature Engineering (Sheets 7-12)
            'market_regime_features': 'features.market_regime',
            'greek_features': 'features.greeks',
            'iv_features': 'features.iv',
            'oi_features': 'features.oi',
            'technical_features': 'features.technical',
            'microstructure_features': 'features.microstructure',
            
            # Risk Management (Sheets 13-16)
            'position_sizing': 'risk_management.position_sizing',
            'risk_limits': 'risk_management.risk_limits',
            'stop_loss': 'risk_management.stop_loss',
            'circuit_breaker': 'risk_management.circuit_breaker',
            
            # Signal Generation (Sheets 17-19)
            'straddle_config': 'signal_generation.straddle_config',
            'signal_filters': 'signal_generation.signal_filters',
            'signal_processing': 'signal_generation.signal_processing',
            
            # Training & Testing (Sheets 20-22)
            'training_config': 'training.config',
            'model_training': 'training.model_training',
            'backtesting': 'training.backtesting',
            
            # Database (Sheets 23-24)
            'heavydb_connection': 'database.heavydb',
            'data_source': 'database.data_source',
            
            # System Status (Sheets 25-26)
            'overview': 'system.overview',
            'performance_targets': 'system.performance_targets'
        }
        
        # Model weight keys for validation
        self.model_weights = {
            'lightgbm': 0.30,
            'catboost': 0.25,
            'tabnet': 0.25,
            'lstm': 0.10,
            'transformer': 0.10
        }
    
    def parse(self, file_path: str) -> Dict[str, Any]:
        """
        Parse ML Triple Straddle configuration file
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            Dictionary containing parsed configuration data
        """
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            raw_data = {}
            
            # Parse each sheet
            for sheet_name in excel_file.sheet_names:
                # Skip index sheet (Sheet 00)
                if sheet_name.lower() in ['sheet_index', 'index', 'sheet 00', '00']:
                    continue
                
                # Read sheet
                df = pd.read_excel(excel_file, sheet_name=sheet_name, header=0)
                
                # Skip empty sheets
                if df.empty:
                    continue
                
                # Parse sheet based on structure
                sheet_data = self._parse_sheet(df, sheet_name)
                
                if sheet_data:
                    normalized_name = self._normalize_sheet_name(sheet_name)
                    raw_data[normalized_name] = sheet_data
            
            # Transform raw data into hierarchical structure
            structured_data = self._structure_data(raw_data)
            
            # Validate the structured data
            if not self._validate_ml_structure(structured_data):
                raise ParsingError("Invalid ML Triple Straddle configuration structure", 
                                 file_path=file_path, errors=self.errors)
            
            return structured_data
            
        except Exception as e:
            raise ParsingError(f"Failed to parse ML Triple Straddle file: {str(e)}", 
                             file_path=file_path)
    
    def _structure_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform flat sheet data into hierarchical structure
        
        Args:
            raw_data: Flat dictionary of sheet data
            
        Returns:
            Hierarchically structured configuration
        """
        structured = {
            'ml_models': {},
            'features': {},
            'risk_management': {},
            'signal_generation': {},
            'training': {},
            'database': {},
            'system': {}
        }
        
        # Map sheets to hierarchical structure
        for sheet_name, sheet_data in raw_data.items():
            # Find matching mapping
            for pattern, path in self.sheet_mapping.items():
                if pattern in sheet_name:
                    # Navigate to the correct nested location
                    parts = path.split('.')
                    target = structured
                    
                    # Create nested structure
                    for i, part in enumerate(parts[:-1]):
                        if part not in target:
                            target[part] = {}
                        target = target[part]
                    
                    # Set the data
                    target[parts[-1]] = sheet_data
                    break
        
        # Add metadata
        structured['_metadata'] = {
            'strategy_type': 'ml_triple_straddle',
            'version': '1.0',
            'sheet_count': len(raw_data)
        }
        
        return structured
    
    def _validate_ml_structure(self, data: Dict[str, Any]) -> bool:
        """
        Validate ML Triple Straddle configuration structure
        
        Args:
            data: Structured configuration data
            
        Returns:
            True if valid, False otherwise
        """
        # Check required sections
        required_sections = ['ml_models', 'features', 'risk_management', 
                           'signal_generation', 'database']
        
        for section in required_sections:
            if section not in data:
                self.add_error(f"Missing required section: {section}")
                return False
        
        # Validate ML models
        if not self._validate_ml_models(data.get('ml_models', {})):
            return False
        
        # Validate features
        if not self._validate_features(data.get('features', {})):
            return False
        
        # Validate risk management
        if not self._validate_risk_management(data.get('risk_management', {})):
            return False
        
        # Validate signal generation
        if not self._validate_signal_generation(data.get('signal_generation', {})):
            return False
        
        # Validate database configuration
        if not self._validate_database(data.get('database', {})):
            return False
        
        return True
    
    def _validate_ml_models(self, models: Dict[str, Any]) -> bool:
        """Validate ML model configurations"""
        required_models = ['lightgbm', 'catboost', 'tabnet', 'ensemble']
        
        for model in required_models:
            if model not in models:
                self.add_error(f"Missing required ML model: {model}")
                return False
        
        # Validate ensemble weights
        ensemble = models.get('ensemble', {})
        if isinstance(ensemble, dict):
            # Check if weights sum to 1.0
            total_weight = 0
            for model_name, expected_weight in self.model_weights.items():
                weight_key = f"{model_name}_weight"
                if weight_key in ensemble:
                    total_weight += ensemble[weight_key]
            
            if abs(total_weight - 1.0) > 0.01:
                self.add_error(f"Model weights must sum to 1.0, got {total_weight}")
                return False
        
        return True
    
    def _validate_features(self, features: Dict[str, Any]) -> bool:
        """Validate feature configurations"""
        required_features = ['market_regime', 'greeks', 'iv', 'oi', 'technical', 
                           'microstructure']
        
        for feature_type in required_features:
            if feature_type not in features:
                self.add_error(f"Missing required feature category: {feature_type}")
                return False
        
        # Validate feature counts match documentation
        expected_counts = {
            'market_regime': 38,
            'greeks': 21,
            'iv': 30,
            'oi': 20,
            'technical': 31,
            'microstructure': 20
        }
        
        for feature_type, expected_count in expected_counts.items():
            feature_data = features.get(feature_type, {})
            if isinstance(feature_data, list):
                actual_count = len(feature_data)
                if actual_count != expected_count:
                    logger.warning(f"{feature_type} has {actual_count} features, "
                                 f"expected {expected_count}")
        
        return True
    
    def _validate_risk_management(self, risk_mgmt: Dict[str, Any]) -> bool:
        """Validate risk management configurations"""
        required_components = ['position_sizing', 'risk_limits', 'stop_loss']
        
        for component in required_components:
            if component not in risk_mgmt:
                self.add_error(f"Missing required risk component: {component}")
                return False
        
        # Validate position sizing
        pos_sizing = risk_mgmt.get('position_sizing', {})
        if isinstance(pos_sizing, dict):
            method = pos_sizing.get('position_sizing_method', '').lower()
            if method == 'kelly':
                kelly_fraction = pos_sizing.get('kelly_fraction', 0)
                if not 0 < kelly_fraction <= 1:
                    self.add_error(f"Invalid Kelly fraction: {kelly_fraction}")
                    return False
        
        return True
    
    def _validate_signal_generation(self, signal_gen: Dict[str, Any]) -> bool:
        """Validate signal generation configurations"""
        required_components = ['straddle_config', 'signal_filters']
        
        for component in required_components:
            if component not in signal_gen:
                self.add_error(f"Missing required signal component: {component}")
                return False
        
        # Validate straddle configuration
        straddle_config = signal_gen.get('straddle_config', {})
        if isinstance(straddle_config, dict):
            # Check straddle weights
            weights = {
                'atm_weight': straddle_config.get('atm_weight', 0),
                'itm_weight': straddle_config.get('itm_weight', 0),
                'otm_weight': straddle_config.get('otm_weight', 0)
            }
            
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 0.01:
                self.add_error(f"Straddle weights must sum to 1.0, got {total_weight}")
                return False
        
        return True
    
    def _validate_database(self, database: Dict[str, Any]) -> bool:
        """Validate database configurations"""
        # Check HeavyDB configuration
        heavydb = database.get('heavydb', {})
        if not heavydb:
            self.add_error("Missing HeavyDB configuration")
            return False
        
        if isinstance(heavydb, dict):
            required_fields = ['host', 'port', 'user', 'database']
            for field in required_fields:
                if field not in heavydb:
                    self.add_error(f"Missing HeavyDB field: {field}")
                    return False
        
        return True
    
    def _parse_model_sheet(self, df: pd.DataFrame, model_name: str) -> Dict[str, Any]:
        """Parse ML model configuration sheet"""
        # Model sheets are typically in key-value format
        if self._is_key_value_sheet(df):
            return self._parse_key_value_sheet(df)
        else:
            # Handle as parameter table
            params = {}
            for _, row in df.iterrows():
                if 'parameter' in df.columns and 'value' in df.columns:
                    param = str(row['parameter']).strip()
                    value = self._convert_value(row['value'])
                    if param and value is not None:
                        params[self._normalize_key(param)] = value
            return params
    
    def _parse_feature_sheet(self, df: pd.DataFrame, feature_type: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Parse feature configuration sheet"""
        # Feature sheets can be either feature lists or configuration
        if 'feature_name' in df.columns or 'feature' in df.columns:
            # Parse as feature list
            features = []
            for _, row in df.iterrows():
                feature = {}
                for col in df.columns:
                    value = self._convert_value(row[col])
                    if value is not None:
                        feature[col] = value
                if feature:
                    features.append(feature)
            return features
        else:
            # Parse as configuration
            return self._parse_key_value_sheet(df) if self._is_key_value_sheet(df) else self._parse_table_sheet(df)
    
    def extract_feature_list(self, data: Dict[str, Any]) -> List[str]:
        """
        Extract complete list of features from configuration
        
        Args:
            data: Parsed configuration data
            
        Returns:
            List of all feature names
        """
        all_features = []
        
        features_section = data.get('features', {})
        for category, feature_data in features_section.items():
            if isinstance(feature_data, list):
                # Extract feature names from list
                for feature in feature_data:
                    if isinstance(feature, dict):
                        name = feature.get('feature_name') or feature.get('feature') or feature.get('name')
                        if name:
                            all_features.append(name)
            elif isinstance(feature_data, dict):
                # Extract enabled features from config
                for key, value in feature_data.items():
                    if key.endswith('_enabled') and value:
                        feature_name = key.replace('_enabled', '')
                        all_features.append(feature_name)
        
        return all_features
    
    def get_model_weights(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract model weights from configuration
        
        Args:
            data: Parsed configuration data
            
        Returns:
            Dictionary of model weights
        """
        weights = {}
        
        ensemble = data.get('ml_models', {}).get('ensemble', {})
        if isinstance(ensemble, dict):
            for model_name in self.model_weights.keys():
                weight_key = f"{model_name}_weight"
                if weight_key in ensemble:
                    weights[model_name] = ensemble[weight_key]
                elif 'model_weight' in ensemble and model_name in str(ensemble):
                    # Alternative format
                    weights[model_name] = ensemble.get('model_weight', 0)
        
        return weights
    
    def __repr__(self) -> str:
        return "MLTripleStraddleParser(sheets=26, models=5, features=160+)"