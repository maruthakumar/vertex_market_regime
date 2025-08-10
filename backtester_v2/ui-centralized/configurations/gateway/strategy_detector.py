"""
Strategy Detector

Automatically detects strategy type from Excel file structure and content.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Any, Set
import re

logger = logging.getLogger(__name__)

class StrategyDetector:
    """
    Automatic strategy type detection from Excel files
    
    Analyzes Excel file structure, sheet names, and content patterns
    to automatically determine the strategy type.
    """
    
    def __init__(self):
        """Initialize strategy detector with detection patterns"""
        
        # Strategy detection patterns
        self.strategy_patterns = {
            "tbs": {
                "required_sheets": ["portfolio", "strategy"],
                "optional_sheets": ["enhancement", "risk"],
                "key_columns": ["capital", "entry_time", "exit_time", "strike_selection_method"],
                "sheet_count_range": (2, 6),
                "identifier_keywords": ["time", "entry", "exit", "tbs"]
            },
            
            "tv": {
                "required_sheets": ["config", "alerts"],
                "optional_sheets": ["risk", "webhook"],
                "key_columns": ["webhook_url", "alert_message", "symbol"],
                "sheet_count_range": (2, 5),
                "identifier_keywords": ["tradingview", "webhook", "alert", "tv"]
            },
            
            "orb": {
                "required_sheets": ["setup", "rules"],
                "optional_sheets": ["risk", "filters"],
                "key_columns": ["opening_range_start", "opening_range_end", "breakout_threshold"],
                "sheet_count_range": (2, 5),
                "identifier_keywords": ["opening", "range", "breakout", "orb"]
            },
            
            "oi": {
                "required_sheets": ["config", "analysis"],
                "optional_sheets": ["rules", "pcr"],
                "key_columns": ["pcr_threshold", "oi_change", "strike_analysis"],
                "sheet_count_range": (2, 5),
                "identifier_keywords": ["open", "interest", "pcr", "oi"]
            },
            
            "ml": {
                "required_sheets": ["models", "features", "training"],
                "optional_sheets": ["risk", "validation", "hyperparameters"],
                "key_columns": ["model_type", "feature_list", "training_data", "ml_algorithm"],
                "sheet_count_range": (3, 10),
                "identifier_keywords": ["machine", "learning", "model", "feature", "ml", "algorithm"]
            },
            
            "pos": {
                "required_sheets": ["position", "greeks"],
                "optional_sheets": ["risk", "hedging"],
                "key_columns": ["delta", "gamma", "theta", "vega", "position_size"],
                "sheet_count_range": (2, 6),
                "identifier_keywords": ["position", "greeks", "delta", "gamma", "theta", "pos"]
            },
            
            "market_regime": {
                "required_sheets": ["indicators", "regimes", "analysis"],
                "optional_sheets": ["transitions", "straddle", "risk", "status"],
                "key_columns": ["regime_type", "indicator_weight", "transition_probability", "regime_classification"],
                "sheet_count_range": (3, 8),
                "identifier_keywords": ["regime", "market", "indicator", "transition", "classification"]
            },
            
            "ml_triple_straddle": {
                "required_sheets": ["lightgbm", "catboost", "tabnet", "ensemble", "features"],
                "optional_sheets": ["lstm", "transformer", "training", "risk", "database"],
                "key_columns": ["model_weight", "feature_engineering", "straddle_config", "ensemble_method"],
                "sheet_count_range": (10, 30),
                "identifier_keywords": ["straddle", "ensemble", "lightgbm", "catboost", "tabnet", "ml_triple"]
            },
            
            "indicator": {
                "required_sheets": ["indicatorconfiguration", "signalconditions"],
                "optional_sheets": ["riskmanagement", "timeframesettings"],
                "key_columns": ["indicator_name", "indicator_type", "signal_type", "threshold"],
                "sheet_count_range": (2, 6),
                "identifier_keywords": ["indicator", "signal", "technical", "rsi", "macd", "smc"]
            },
            
            "strategy_consolidation": {
                "required_sheets": ["strategies", "consolidation", "optimization"],
                "optional_sheets": ["portfolio", "weights", "allocation"],
                "key_columns": ["strategy_weight", "allocation_method", "optimization_method", "consolidation_rules"],
                "sheet_count_range": (3, 8),
                "identifier_keywords": ["consolidation", "portfolio", "allocation", "optimization", "weight"]
            }
        }
        
        # Common column patterns for fallback detection
        self.common_patterns = {
            "time_based": ["entry_time", "exit_time", "square_off_time"],
            "options_trading": ["strike_price", "expiry", "option_type", "premium"],
            "risk_management": ["stop_loss", "target", "max_loss", "position_size"],
            "technical_analysis": ["rsi", "macd", "ema", "sma", "bollinger"],
            "machine_learning": ["model", "feature", "training", "prediction", "algorithm"]
        }
    
    def detect_strategy_type(self, file_path: str) -> Optional[str]:
        """
        Detect strategy type from Excel file
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            Detected strategy type or None if cannot determine
        """
        try:
            file_path = Path(file_path)
            
            # Step 1: Try filename detection
            filename_strategy = self._detect_from_filename(file_path.name)
            if filename_strategy:
                logger.debug(f"Detected {filename_strategy} from filename: {file_path.name}")
                return filename_strategy
            
            # Step 2: Analyze Excel structure
            structure_strategy = self._detect_from_structure(file_path)
            if structure_strategy:
                logger.debug(f"Detected {structure_strategy} from structure")
                return structure_strategy
            
            # Step 3: Content analysis
            content_strategy = self._detect_from_content(file_path)
            if content_strategy:
                logger.debug(f"Detected {content_strategy} from content")
                return content_strategy
            
            logger.warning(f"Could not detect strategy type for {file_path}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to detect strategy type for {file_path}: {e}")
            return None
    
    def _detect_from_filename(self, filename: str) -> Optional[str]:
        """Detect strategy from filename patterns"""
        filename_lower = filename.lower()
        
        # Direct matches
        for strategy_type in self.strategy_patterns.keys():
            if strategy_type in filename_lower:
                return strategy_type
        
        # Special patterns
        patterns = {
            "tbs": ["time_based", "timebase", "time.based"],
            "tv": ["tradingview", "trading_view", "tv_"],
            "orb": ["opening_range", "breakout", "orb_"],
            "oi": ["open_interest", "oi_", "pcr"],
            "ml": ["machine_learning", "ml_", "model"],
            "pos": ["positional", "position", "greeks"],
            "market_regime": ["regime", "market_regime", "market.regime"],
            "ml_triple_straddle": ["triple_straddle", "ml_triple", "triple.straddle"],
            "indicator": ["indicator", "technical", "ta_lib"],
            "strategy_consolidation": ["consolidation", "portfolio", "multi_strategy"]
        }
        
        for strategy_type, keywords in patterns.items():
            if any(keyword in filename_lower for keyword in keywords):
                return strategy_type
        
        return None
    
    def _detect_from_structure(self, file_path: Path) -> Optional[str]:
        """Detect strategy from Excel structure (sheet names, count)"""
        try:
            excel_file = pd.ExcelFile(str(file_path))
            sheet_names = [name.lower().replace(' ', '').replace('_', '') for name in excel_file.sheet_names]
            sheet_count = len(sheet_names)
            
            # Score each strategy based on structure match
            scores = {}
            
            for strategy_type, pattern in self.strategy_patterns.items():
                score = 0
                
                # Check sheet count range
                min_sheets, max_sheets = pattern["sheet_count_range"]
                if min_sheets <= sheet_count <= max_sheets:
                    score += 2
                
                # Check required sheets
                required_matches = 0
                for required_sheet in pattern["required_sheets"]:
                    if any(required_sheet in sheet_name for sheet_name in sheet_names):
                        required_matches += 1
                        score += 5
                
                # Must have at least some required sheets
                if required_matches == 0:
                    continue
                
                # Check optional sheets
                for optional_sheet in pattern["optional_sheets"]:
                    if any(optional_sheet in sheet_name for sheet_name in sheet_names):
                        score += 2
                
                # Check for strategy keywords in sheet names
                for keyword in pattern["identifier_keywords"]:
                    if any(keyword in sheet_name for sheet_name in sheet_names):
                        score += 3
                
                scores[strategy_type] = score
            
            # Return strategy with highest score if above threshold
            if scores:
                best_strategy = max(scores, key=scores.get)
                if scores[best_strategy] >= 7:  # Minimum confidence threshold
                    return best_strategy
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to analyze structure for {file_path}: {e}")
            return None
    
    def _detect_from_content(self, file_path: Path) -> Optional[str]:
        """Detect strategy from Excel content analysis"""
        try:
            excel_file = pd.ExcelFile(str(file_path))
            
            # Analyze first few sheets for content patterns
            all_columns = set()
            all_content = []
            
            for sheet_name in excel_file.sheet_names[:5]:  # Limit to first 5 sheets
                try:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name, nrows=20)
                    
                    # Collect column names
                    if not df.empty:
                        all_columns.update([col.lower().replace(' ', '_') for col in df.columns])
                        
                        # Collect cell content
                        for col in df.columns:
                            all_content.extend([str(val).lower() for val in df[col].dropna().head(10)])
                
                except Exception as e:
                    logger.debug(f"Failed to read sheet {sheet_name}: {e}")
                    continue
            
            # Score strategies based on content patterns
            scores = {}
            
            for strategy_type, pattern in self.strategy_patterns.items():
                score = 0
                
                # Check for key columns
                for key_column in pattern["key_columns"]:
                    if any(key_column in col for col in all_columns):
                        score += 5
                
                # Check for identifier keywords in content
                for keyword in pattern["identifier_keywords"]:
                    if any(keyword in content for content in all_content):
                        score += 3
                
                scores[strategy_type] = score
            
            # Return best match if above threshold
            if scores:
                best_strategy = max(scores, key=scores.get)
                if scores[best_strategy] >= 8:  # Content threshold
                    return best_strategy
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to analyze content for {file_path}: {e}")
            return None
    
    def get_detection_confidence(self, file_path: str, detected_strategy: str) -> float:
        """
        Get confidence score for detected strategy
        
        Args:
            file_path: Path to Excel file
            detected_strategy: Detected strategy type
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        try:
            if detected_strategy not in self.strategy_patterns:
                return 0.0
            
            file_path = Path(file_path)
            pattern = self.strategy_patterns[detected_strategy]
            
            total_score = 0
            max_possible_score = 0
            
            # Filename confidence
            filename_match = self._detect_from_filename(file_path.name) == detected_strategy
            total_score += 10 if filename_match else 0
            max_possible_score += 10
            
            # Structure confidence
            try:
                excel_file = pd.ExcelFile(str(file_path))
                sheet_names = [name.lower() for name in excel_file.sheet_names]
                sheet_count = len(sheet_names)
                
                # Sheet count match
                min_sheets, max_sheets = pattern["sheet_count_range"]
                if min_sheets <= sheet_count <= max_sheets:
                    total_score += 20
                max_possible_score += 20
                
                # Required sheets match
                required_matches = sum(1 for req in pattern["required_sheets"] 
                                     if any(req in sheet for sheet in sheet_names))
                total_score += required_matches * 15
                max_possible_score += len(pattern["required_sheets"]) * 15
                
                # Optional sheets match
                optional_matches = sum(1 for opt in pattern["optional_sheets"]
                                     if any(opt in sheet for sheet in sheet_names))
                total_score += optional_matches * 5
                max_possible_score += len(pattern["optional_sheets"]) * 5
                
            except Exception:
                pass
            
            return min(total_score / max_possible_score if max_possible_score > 0 else 0, 1.0)
            
        except Exception as e:
            logger.warning(f"Failed to calculate confidence: {e}")
            return 0.0
    
    def suggest_strategy_types(self, file_path: str, top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Get top N strategy type suggestions with confidence scores
        
        Args:
            file_path: Path to Excel file
            top_n: Number of suggestions to return
            
        Returns:
            List of strategy suggestions with confidence scores
        """
        suggestions = []
        
        for strategy_type in self.strategy_patterns.keys():
            # Temporarily detect as this strategy
            confidence = self.get_detection_confidence(file_path, strategy_type)
            
            if confidence > 0.1:  # Minimum threshold
                suggestions.append({
                    "strategy_type": strategy_type,
                    "confidence": confidence,
                    "display_name": self._get_strategy_display_name(strategy_type),
                    "description": self._get_strategy_description(strategy_type)
                })
        
        # Sort by confidence and return top N
        suggestions.sort(key=lambda x: x["confidence"], reverse=True)
        return suggestions[:top_n]
    
    def _get_strategy_display_name(self, strategy_type: str) -> str:
        """Get human-readable strategy name"""
        display_names = {
            "tbs": "Time-Based Strategy",
            "tv": "TradingView Strategy",
            "orb": "Opening Range Breakout",
            "oi": "Open Interest Strategy", 
            "ml": "Machine Learning Strategy",
            "pos": "Positional Strategy",
            "market_regime": "Market Regime Strategy",
            "ml_triple_straddle": "ML Triple Rolling Straddle",
            "indicator": "Technical Indicator Strategy",
            "strategy_consolidation": "Strategy Consolidation"
        }
        
        return display_names.get(strategy_type, strategy_type.upper())
    
    def _get_strategy_description(self, strategy_type: str) -> str:
        """Get strategy description"""
        descriptions = {
            "tbs": "Simple time-based entry and exit strategy",
            "tv": "Strategy based on TradingView alerts",
            "orb": "Breakout strategy based on opening range",
            "oi": "Strategy based on open interest analysis",
            "ml": "Machine learning based trading strategy",
            "pos": "Multi-day positional strategy with Greeks",
            "market_regime": "18-regime market classification strategy",
            "ml_triple_straddle": "Advanced ML-based options strategy",
            "indicator": "Technical indicator based strategy",
            "strategy_consolidation": "Portfolio-level strategy management"
        }
        
        return descriptions.get(strategy_type, f"{strategy_type} trading strategy")
    
    def validate_detection(self, file_path: str, expected_strategy: str) -> bool:
        """
        Validate that detection matches expected strategy
        
        Args:
            file_path: Path to Excel file
            expected_strategy: Expected strategy type
            
        Returns:
            True if detection matches expected strategy
        """
        detected = self.detect_strategy_type(file_path)
        return detected == expected_strategy