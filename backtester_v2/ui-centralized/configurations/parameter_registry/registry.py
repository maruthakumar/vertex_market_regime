"""
Parameter Registry

Central registry for all strategy parameters. Provides storage, retrieval,
and management capabilities for parameter definitions across all strategies.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
import sqlite3
import threading
from collections import defaultdict

from .models import (
    ParameterDefinition, 
    ParameterCategory, 
    StrategyMetadata,
    ParameterType,
    WidgetType
)

logger = logging.getLogger(__name__)

class ParameterRegistry:
    """
    Central registry for strategy parameters
    
    Provides a unified interface for storing, retrieving, and managing
    parameter definitions across all strategy types.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, db_path: Optional[str] = None):
        """Ensure singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize parameter registry"""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self.db_path = db_path or "/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/configurations/parameter_registry/registry.db"
            self.db_path = Path(self.db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # In-memory caches
            self._parameters: Dict[str, ParameterDefinition] = {}
            self._categories: Dict[str, ParameterCategory] = {}
            self._strategies: Dict[str, StrategyMetadata] = {}
            self._strategy_parameters: Dict[str, List[str]] = defaultdict(list)
            
            # Initialize database
            self._init_database()
            
            # Load existing data
            self._load_from_database()
            
            logger.info(f"ParameterRegistry initialized with {len(self._parameters)} parameters")
    
    def _init_database(self):
        """Initialize SQLite database tables"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Parameters table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS parameters (
                    parameter_id TEXT PRIMARY KEY,
                    strategy_type TEXT NOT NULL,
                    category TEXT NOT NULL,
                    name TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    default_value TEXT,
                    validation_rules TEXT,
                    ui_hints TEXT,
                    enum_values TEXT,
                    description TEXT,
                    version TEXT DEFAULT '1.0',
                    created_at TEXT,
                    updated_at TEXT,
                    UNIQUE(strategy_type, category, name)
                )
            ''')
            
            # Categories table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS categories (
                    category_id TEXT PRIMARY KEY,
                    strategy_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    description TEXT,
                    order_index INTEGER DEFAULT 0,
                    icon TEXT,
                    collapsible BOOLEAN DEFAULT 1,
                    collapsed_by_default BOOLEAN DEFAULT 0,
                    UNIQUE(strategy_type, name)
                )
            ''')
            
            # Strategies table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategies (
                    strategy_type TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    version TEXT DEFAULT '1.0',
                    excel_template_sheets TEXT,
                    parameter_count INTEGER DEFAULT 0,
                    category_count INTEGER DEFAULT 0,
                    complexity_level TEXT DEFAULT 'intermediate',
                    documentation_url TEXT,
                    icon TEXT
                )
            ''')
            
            # Create indices for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_parameters_strategy ON parameters(strategy_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_parameters_category ON parameters(category)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_categories_strategy ON categories(strategy_type)')
            
            conn.commit()
    
    def _load_from_database(self):
        """Load existing data from database into memory"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Load parameters
            cursor.execute('SELECT * FROM parameters')
            for row in cursor.fetchall():
                param_data = {
                    'parameter_id': row[0],
                    'strategy_type': row[1],
                    'category': row[2],
                    'name': row[3],
                    'data_type': row[4],
                    'default_value': json.loads(row[5]) if row[5] else None,
                    'validation_rules': json.loads(row[6]) if row[6] else [],
                    'ui_hints': json.loads(row[7]) if row[7] else None,
                    'enum_values': json.loads(row[8]) if row[8] else None,
                    'description': row[9],
                    'version': row[10],
                    'created_at': row[11],
                    'updated_at': row[12]
                }
                
                param = ParameterDefinition.from_dict(param_data)
                self._parameters[param.parameter_id] = param
                self._strategy_parameters[param.strategy_type].append(param.parameter_id)
            
            # Load categories
            cursor.execute('SELECT * FROM categories')
            for row in cursor.fetchall():
                category = ParameterCategory(
                    category_id=row[0],
                    strategy_type=row[1],
                    name=row[2],
                    display_name=row[3],
                    description=row[4],
                    order=row[5],
                    icon=row[6],
                    collapsible=bool(row[7]),
                    collapsed_by_default=bool(row[8])
                )
                self._categories[category.category_id] = category
            
            # Load strategies
            cursor.execute('SELECT * FROM strategies')
            for row in cursor.fetchall():
                strategy = StrategyMetadata(
                    strategy_type=row[0],
                    display_name=row[1],
                    description=row[2],
                    version=row[3],
                    excel_template_sheets=json.loads(row[4]) if row[4] else [],
                    parameter_count=row[5],
                    category_count=row[6],
                    complexity_level=row[7],
                    documentation_url=row[8],
                    icon=row[9]
                )
                self._strategies[strategy.strategy_type] = strategy
    
    def register_parameter(self, parameter: ParameterDefinition) -> bool:
        """
        Register a parameter definition
        
        Args:
            parameter: Parameter definition to register
            
        Returns:
            True if registered successfully
        """
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                # Insert or update parameter
                cursor.execute('''
                    INSERT OR REPLACE INTO parameters (
                        parameter_id, strategy_type, category, name, data_type,
                        default_value, validation_rules, ui_hints, enum_values,
                        description, version, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    parameter.parameter_id,
                    parameter.strategy_type,
                    parameter.category,
                    parameter.name,
                    parameter.data_type.value,
                    json.dumps(parameter.default_value),
                    json.dumps([rule.to_dict() for rule in parameter.validation_rules]),
                    json.dumps(parameter.ui_hints.to_dict()) if parameter.ui_hints else None,
                    json.dumps(parameter.enum_values),
                    parameter.description,
                    parameter.version,
                    parameter.created_at.isoformat(),
                    parameter.updated_at.isoformat()
                ))
                
                conn.commit()
            
            # Update memory cache
            self._parameters[parameter.parameter_id] = parameter
            if parameter.parameter_id not in self._strategy_parameters[parameter.strategy_type]:
                self._strategy_parameters[parameter.strategy_type].append(parameter.parameter_id)
            
            logger.debug(f"Registered parameter: {parameter.parameter_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register parameter {parameter.parameter_id}: {e}")
            return False
    
    def register_category(self, category: ParameterCategory) -> bool:
        """Register a parameter category"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO categories (
                        category_id, strategy_type, name, display_name, description,
                        order_index, icon, collapsible, collapsed_by_default
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    category.category_id,
                    category.strategy_type,
                    category.name,
                    category.display_name,
                    category.description,
                    category.order,
                    category.icon,
                    category.collapsible,
                    category.collapsed_by_default
                ))
                
                conn.commit()
            
            self._categories[category.category_id] = category
            return True
            
        except Exception as e:
            logger.error(f"Failed to register category {category.category_id}: {e}")
            return False
    
    def register_strategy(self, strategy: StrategyMetadata) -> bool:
        """Register strategy metadata"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO strategies (
                        strategy_type, display_name, description, version,
                        excel_template_sheets, parameter_count, category_count,
                        complexity_level, documentation_url, icon
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    strategy.strategy_type,
                    strategy.display_name,
                    strategy.description,
                    strategy.version,
                    json.dumps(strategy.excel_template_sheets),
                    strategy.parameter_count,
                    strategy.category_count,
                    strategy.complexity_level,
                    strategy.documentation_url,
                    strategy.icon
                ))
                
                conn.commit()
            
            self._strategies[strategy.strategy_type] = strategy
            return True
            
        except Exception as e:
            logger.error(f"Failed to register strategy {strategy.strategy_type}: {e}")
            return False
    
    def get_parameter(self, parameter_id: str) -> Optional[ParameterDefinition]:
        """Get parameter by ID"""
        return self._parameters.get(parameter_id)
    
    def get_parameters_by_strategy(self, strategy_type: str) -> List[ParameterDefinition]:
        """Get all parameters for a strategy"""
        param_ids = self._strategy_parameters.get(strategy_type, [])
        return [self._parameters[pid] for pid in param_ids if pid in self._parameters]
    
    def get_parameters_by_category(self, strategy_type: str, category: str) -> List[ParameterDefinition]:
        """Get parameters by strategy and category"""
        return [
            param for param in self.get_parameters_by_strategy(strategy_type)
            if param.category == category
        ]
    
    def get_category(self, category_id: str) -> Optional[ParameterCategory]:
        """Get category by ID"""
        return self._categories.get(category_id)
    
    def get_categories_by_strategy(self, strategy_type: str) -> List[ParameterCategory]:
        """Get all categories for a strategy"""
        return [
            cat for cat in self._categories.values()
            if cat.strategy_type == strategy_type
        ]
    
    def get_strategy(self, strategy_type: str) -> Optional[StrategyMetadata]:
        """Get strategy metadata"""
        return self._strategies.get(strategy_type)
    
    def get_all_strategies(self) -> List[StrategyMetadata]:
        """Get all registered strategies"""
        return list(self._strategies.values())
    
    def search_parameters(self, query: str, strategy_type: Optional[str] = None) -> List[ParameterDefinition]:
        """
        Search parameters by name, description, or ID
        
        Args:
            query: Search query
            strategy_type: Optional strategy type filter
            
        Returns:
            List of matching parameters
        """
        query = query.lower()
        results = []
        
        for param in self._parameters.values():
            if strategy_type and param.strategy_type != strategy_type:
                continue
                
            if (query in param.name.lower() or 
                query in param.parameter_id.lower() or
                (param.description and query in param.description.lower())):
                results.append(param)
        
        return results
    
    def get_schema_for_strategy(self, strategy_type: str) -> Dict[str, Any]:
        """
        Generate complete JSON schema for a strategy
        
        Args:
            strategy_type: Strategy type
            
        Returns:
            JSON schema for the strategy
        """
        parameters = self.get_parameters_by_strategy(strategy_type)
        categories = self.get_categories_by_strategy(strategy_type)
        
        # Group parameters by category
        properties = {}
        required = []
        
        for category in sorted(categories, key=lambda c: c.order):
            cat_params = [p for p in parameters if p.category == category.name]
            if not cat_params:
                continue
                
            cat_properties = {}
            cat_required = []
            
            for param in sorted(cat_params, key=lambda p: p.ui_hints.order if p.ui_hints else 0):
                cat_properties[param.name] = param.get_validation_schema()
                if param.is_required():
                    cat_required.append(param.name)
            
            properties[category.name] = {
                "type": "object",
                "properties": cat_properties,
                "required": cat_required
            }
            
            if cat_required:
                required.append(category.name)
        
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": properties,
            "required": required
        }
    
    def export_to_json(self, file_path: str) -> bool:
        """Export registry to JSON file"""
        try:
            data = {
                "parameters": [param.to_dict() for param in self._parameters.values()],
                "categories": [cat.to_dict() for cat in self._categories.values()],
                "strategies": [strat.to_dict() for strat in self._strategies.values()],
                "exported_at": datetime.now().isoformat()
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to export registry: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        strategy_stats = {}
        for strategy_type in self._strategies.keys():
            params = self.get_parameters_by_strategy(strategy_type)
            categories = self.get_categories_by_strategy(strategy_type)
            
            strategy_stats[strategy_type] = {
                "parameter_count": len(params),
                "category_count": len(categories),
                "required_parameters": len([p for p in params if p.is_required()]),
                "enum_parameters": len([p for p in params if p.data_type == ParameterType.ENUM])
            }
        
        return {
            "total_parameters": len(self._parameters),
            "total_categories": len(self._categories),
            "total_strategies": len(self._strategies),
            "strategy_breakdown": strategy_stats
        }