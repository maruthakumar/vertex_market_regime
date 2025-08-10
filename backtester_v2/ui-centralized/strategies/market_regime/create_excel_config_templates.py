#!/usr/bin/env python3
"""
Excel Configuration Template Generator
Phase 2 Day 5: Excel Configuration Integration

This module creates comprehensive Excel configuration templates for the
DTE Enhanced Triple Straddle Rolling Analysis Framework.

Author: The Augster
Date: 2025-06-20
Version: 5.0.0 (Phase 2 Day 5 Excel Integration)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_dte_enhanced_configuration_template():
    """Create comprehensive Excel configuration template for DTE learning system"""
    
    logger.info("🚀 Creating DTE Enhanced Configuration Template...")
    
    # Create output directory
    output_dir = Path("excel_config_templates")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize Excel writer
    excel_file = output_dir / "DTE_ENHANCED_CONFIGURATION_TEMPLATE.xlsx"
    
    try:
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            
            # Sheet 1: DTE Learning Configuration
            logger.info("📊 Creating DTE Learning Configuration sheet...")
            dte_config = create_dte_learning_config_sheet()
            dte_config.to_excel(writer, sheet_name='DTE_Learning_Config', index=False)
            
            # Sheet 2: ML Model Configuration
            logger.info("🧠 Creating ML Model Configuration sheet...")
            ml_config = create_ml_model_config_sheet()
            ml_config.to_excel(writer, sheet_name='ML_Model_Config', index=False)
            
            # Sheet 3: Strategy Type Configuration
            logger.info("⚙️ Creating Strategy Type Configuration sheet...")
            strategy_config = create_strategy_type_config_sheet()
            strategy_config.to_excel(writer, sheet_name='Strategy_Config', index=False)
            
            # Sheet 4: Performance Parameters
            logger.info("⚡ Creating Performance Configuration sheet...")
            performance_config = create_performance_config_sheet()
            performance_config.to_excel(writer, sheet_name='Performance_Config', index=False)
            
            # Sheet 5: Progressive Disclosure Settings
            logger.info("🎯 Creating Progressive Disclosure Configuration sheet...")
            ui_config = create_progressive_disclosure_config_sheet()
            ui_config.to_excel(writer, sheet_name='UI_Config', index=False)
            
            # Sheet 6: Historical Validation Parameters
            logger.info("📈 Creating Historical Validation Configuration sheet...")
            validation_config = create_historical_validation_config_sheet()
            validation_config.to_excel(writer, sheet_name='Validation_Config', index=False)
            
            # Sheet 7: Rolling Analysis Configuration
            logger.info("🔄 Creating Rolling Analysis Configuration sheet...")
            rolling_config = create_rolling_analysis_config_sheet()
            rolling_config.to_excel(writer, sheet_name='Rolling_Config', index=False)
            
            # Sheet 8: Regime Classification Configuration
            logger.info("🎭 Creating Regime Classification Configuration sheet...")
            regime_config = create_regime_classification_config_sheet()
            regime_config.to_excel(writer, sheet_name='Regime_Config', index=False)
        
        logger.info(f"✅ Excel configuration template created: {excel_file}")
        return excel_file
        
    except Exception as e:
        logger.error(f"❌ Error creating Excel template: {e}")
        raise

def create_dte_learning_config_sheet():
    """Create DTE Learning Configuration sheet"""
    
    config_data = {
        'Parameter': [
            'DTE_LEARNING_ENABLED',
            'DTE_RANGE_MIN',
            'DTE_RANGE_MAX', 
            'DTE_FOCUS_RANGE_MIN',
            'DTE_FOCUS_RANGE_MAX',
            'HISTORICAL_YEARS_REQUIRED',
            'MIN_SAMPLE_SIZE_PER_DTE',
            'CONFIDENCE_LEVEL',
            'SIGNIFICANCE_THRESHOLD',
            'ATM_BASE_WEIGHT',
            'ITM1_BASE_WEIGHT',
            'OTM1_BASE_WEIGHT',
            'DTE_0_ATM_MULTIPLIER',
            'DTE_1_ATM_MULTIPLIER',
            'DTE_2_4_ATM_MULTIPLIER',
            'DTE_5_PLUS_ATM_MULTIPLIER',
            'WEIGHT_BOUNDS_MIN',
            'WEIGHT_BOUNDS_MAX',
            'LEARNING_RATE',
            'MOMENTUM_FACTOR',
            'DECAY_FACTOR',
            'MARKET_SIMILARITY_THRESHOLD',
            'ADAPTATION_RATE_LIMIT',
            'CONVERGENCE_THRESHOLD'
        ],
        'Value': [
            True,
            0,
            30,
            0,
            4,
            3,
            100,
            0.95,
            0.05,
            0.50,
            0.30,
            0.20,
            1.30,
            1.20,
            1.00,
            0.90,
            0.05,
            0.80,
            0.01,
            0.90,
            0.95,
            0.70,
            0.10,
            0.001
        ],
        'Description': [
            'Enable/disable DTE learning framework',
            'Minimum DTE value for analysis',
            'Maximum DTE value for analysis',
            'Minimum DTE for focus range (0-4 DTE)',
            'Maximum DTE for focus range (0-4 DTE)',
            'Years of historical data required',
            'Minimum sample size per DTE value',
            'Statistical confidence level',
            'P-value significance threshold',
            'Base weight for ATM straddle',
            'Base weight for ITM1 straddle',
            'Base weight for OTM1 straddle',
            'ATM weight multiplier for DTE 0',
            'ATM weight multiplier for DTE 1',
            'ATM weight multiplier for DTE 2-4',
            'ATM weight multiplier for DTE 5+',
            'Minimum allowed weight',
            'Maximum allowed weight',
            'ML learning rate',
            'Momentum factor for adaptation',
            'Exponential decay factor',
            'Market similarity threshold',
            'Maximum adaptation rate per update',
            'Convergence threshold for optimization'
        ],
        'Skill_Level': [
            'Novice',
            'Novice',
            'Novice',
            'Novice',
            'Novice',
            'Intermediate',
            'Intermediate',
            'Expert',
            'Expert',
            'Novice',
            'Novice',
            'Novice',
            'Intermediate',
            'Intermediate',
            'Intermediate',
            'Intermediate',
            'Intermediate',
            'Intermediate',
            'Intermediate',
            'Intermediate',
            'Expert',
            'Expert',
            'Expert',
            'Expert'
        ],
        'Category': [
            'Core',
            'Core',
            'Core',
            'Core',
            'Core',
            'Validation',
            'Validation',
            'Statistical',
            'Statistical',
            'Weights',
            'Weights',
            'Weights',
            'DTE_Specific',
            'DTE_Specific',
            'DTE_Specific',
            'DTE_Specific',
            'Constraints',
            'Constraints',
            'ML_Learning',
            'ML_Learning',
            'ML_Learning',
            'Market_Analysis',
            'Adaptation',
            'Optimization'
        ],
        'Hot_Reload': [
            True,
            False,
            False,
            True,
            True,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True
        ]
    }
    
    return pd.DataFrame(config_data)

def create_ml_model_config_sheet():
    """Create ML Model Configuration sheet"""
    
    config_data = {
        'Model_Type': [
            'RandomForest',
            'RandomForest',
            'RandomForest',
            'RandomForest',
            'RandomForest',
            'RandomForest',
            'NeuralNetwork',
            'NeuralNetwork',
            'NeuralNetwork',
            'NeuralNetwork',
            'NeuralNetwork',
            'NeuralNetwork',
            'Ensemble',
            'Ensemble',
            'Ensemble',
            'Ensemble',
            'Ensemble',
            'Performance',
            'Performance',
            'Performance'
        ],
        'Parameter': [
            'n_estimators',
            'max_depth',
            'min_samples_split',
            'random_state',
            'enabled',
            'feature_importance_threshold',
            'hidden_layer_sizes',
            'max_iter',
            'random_state',
            'early_stopping',
            'validation_fraction',
            'enabled',
            'weighting_method',
            'confidence_threshold',
            'fallback_strategy',
            'enabled',
            'model_selection_criteria',
            'training_data_window',
            'retraining_frequency',
            'performance_threshold'
        ],
        'Value': [
            100,
            10,
            5,
            42,
            True,
            0.01,
            '(100,50)',
            500,
            42,
            True,
            0.2,
            True,
            'weighted_average',
            0.60,
            'base_weights',
            True,
            'cross_validation',
            252,
            'daily',
            0.85
        ],
        'Description': [
            'Number of trees in Random Forest',
            'Maximum depth of trees',
            'Minimum samples required to split node',
            'Random seed for reproducibility',
            'Enable Random Forest model',
            'Minimum feature importance threshold',
            'Hidden layer architecture (tuple)',
            'Maximum training iterations',
            'Random seed for reproducibility',
            'Enable early stopping',
            'Fraction of data for validation',
            'Enable Neural Network model',
            'Method for combining model predictions',
            'Minimum confidence for ML predictions',
            'Strategy when ML models fail',
            'Enable ensemble approach',
            'Criteria for selecting best model',
            'Training data window size (days)',
            'How often to retrain models',
            'Minimum performance threshold'
        ],
        'Skill_Level': [
            'Intermediate',
            'Expert',
            'Expert',
            'Expert',
            'Intermediate',
            'Expert',
            'Expert',
            'Expert',
            'Expert',
            'Expert',
            'Expert',
            'Intermediate',
            'Expert',
            'Intermediate',
            'Expert',
            'Intermediate',
            'Expert',
            'Intermediate',
            'Intermediate',
            'Intermediate'
        ],
        'Category': [
            'RandomForest',
            'RandomForest',
            'RandomForest',
            'RandomForest',
            'RandomForest',
            'RandomForest',
            'NeuralNetwork',
            'NeuralNetwork',
            'NeuralNetwork',
            'NeuralNetwork',
            'NeuralNetwork',
            'NeuralNetwork',
            'Ensemble',
            'Ensemble',
            'Ensemble',
            'Ensemble',
            'Ensemble',
            'Performance',
            'Performance',
            'Performance'
        ],
        'Hot_Reload': [
            False,
            False,
            False,
            False,
            True,
            True,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            True
        ]
    }
    
    return pd.DataFrame(config_data)

def create_strategy_type_config_sheet():
    """Create Strategy Type Configuration sheet"""
    
    config_data = {
        'Strategy_Type': [
            'TBS',
            'TBS',
            'TBS',
            'TBS',
            'TV',
            'TV',
            'TV',
            'TV',
            'ORB',
            'ORB',
            'ORB',
            'ORB',
            'OI',
            'OI',
            'OI',
            'OI',
            'Indicator',
            'Indicator',
            'Indicator',
            'Indicator',
            'POS',
            'POS',
            'POS',
            'POS'
        ],
        'Parameter': [
            'dte_learning_enabled',
            'default_dte_focus',
            'weight_optimization',
            'performance_target',
            'dte_learning_enabled',
            'default_dte_focus',
            'weight_optimization',
            'performance_target',
            'dte_learning_enabled',
            'default_dte_focus',
            'weight_optimization',
            'performance_target',
            'dte_learning_enabled',
            'default_dte_focus',
            'weight_optimization',
            'performance_target',
            'dte_learning_enabled',
            'default_dte_focus',
            'weight_optimization',
            'performance_target',
            'dte_learning_enabled',
            'default_dte_focus',
            'weight_optimization',
            'performance_target'
        ],
        'Value': [
            True,
            3,
            'ml_enhanced',
            0.85,
            True,
            2,
            'ml_enhanced',
            0.80,
            True,
            1,
            'ml_enhanced',
            0.75,
            True,
            7,
            'statistical',
            0.70,
            True,
            14,
            'ml_enhanced',
            0.65,
            True,
            0,
            'conservative',
            0.90
        ],
        'Description': [
            'Enable DTE learning for TBS strategy',
            'Default DTE focus for TBS',
            'Weight optimization method for TBS',
            'Performance target for TBS',
            'Enable DTE learning for TV strategy',
            'Default DTE focus for TV',
            'Weight optimization method for TV',
            'Performance target for TV',
            'Enable DTE learning for ORB strategy',
            'Default DTE focus for ORB',
            'Weight optimization method for ORB',
            'Performance target for ORB',
            'Enable DTE learning for OI strategy',
            'Default DTE focus for OI',
            'Weight optimization method for OI',
            'Performance target for OI',
            'Enable DTE learning for Indicator strategy',
            'Default DTE focus for Indicator',
            'Weight optimization method for Indicator',
            'Performance target for Indicator',
            'Enable DTE learning for POS strategy',
            'Default DTE focus for POS',
            'Weight optimization method for POS',
            'Performance target for POS'
        ],
        'Skill_Level': [
            'Intermediate',
            'Novice',
            'Expert',
            'Intermediate',
            'Intermediate',
            'Novice',
            'Expert',
            'Intermediate',
            'Intermediate',
            'Novice',
            'Expert',
            'Intermediate',
            'Intermediate',
            'Novice',
            'Expert',
            'Intermediate',
            'Intermediate',
            'Novice',
            'Expert',
            'Intermediate',
            'Intermediate',
            'Novice',
            'Expert',
            'Intermediate'
        ],
        'Enabled': [
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True
        ],
        'Hot_Reload': [
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True
        ]
    }
    
    return pd.DataFrame(config_data)

def create_performance_config_sheet():
    """Create Performance Configuration sheet"""

    config_data = {
        'Parameter': [
            'TARGET_PROCESSING_TIME',
            'PARALLEL_PROCESSING_ENABLED',
            'MAX_WORKERS',
            'ENABLE_CACHING',
            'CACHE_SIZE_LIMIT',
            'ENABLE_VECTORIZATION',
            'MEMORY_LIMIT_MB',
            'CPU_UTILIZATION_TARGET',
            'OPTIMIZATION_LEVEL',
            'PERFORMANCE_MONITORING',
            'LOGGING_LEVEL',
            'PROFILING_ENABLED',
            'BENCHMARK_MODE',
            'TIMEOUT_SECONDS'
        ],
        'Value': [
            3.0,
            True,
            72,
            True,
            1000,
            True,
            1024,
            80.0,
            'aggressive',
            True,
            'INFO',
            False,
            False,
            300
        ],
        'Description': [
            'Target processing time in seconds',
            'Enable parallel processing',
            'Maximum number of worker threads',
            'Enable result caching',
            'Maximum cache size (MB)',
            'Enable vectorized operations',
            'Memory usage limit (MB)',
            'Target CPU utilization percentage',
            'Optimization level (conservative/balanced/aggressive)',
            'Enable performance monitoring',
            'Logging level (DEBUG/INFO/WARNING/ERROR)',
            'Enable performance profiling',
            'Enable benchmark mode',
            'Operation timeout in seconds'
        ],
        'Skill_Level': [
            'Novice',
            'Intermediate',
            'Expert',
            'Intermediate',
            'Expert',
            'Expert',
            'Expert',
            'Expert',
            'Intermediate',
            'Intermediate',
            'Expert',
            'Expert',
            'Expert',
            'Expert'
        ],
        'Category': [
            'Performance',
            'Performance',
            'Performance',
            'Performance',
            'Performance',
            'Performance',
            'Performance',
            'Performance',
            'Performance',
            'Monitoring',
            'Monitoring',
            'Monitoring',
            'Monitoring',
            'System'
        ],
        'Hot_Reload': [
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True
        ]
    }

    return pd.DataFrame(config_data)

def create_progressive_disclosure_config_sheet():
    """Create Progressive Disclosure Configuration sheet"""

    config_data = {
        'Parameter': [
            'DTE_LEARNING_ENABLED',
            'DTE_FOCUS_RANGE_MIN',
            'DTE_FOCUS_RANGE_MAX',
            'ATM_BASE_WEIGHT',
            'ITM1_BASE_WEIGHT',
            'OTM1_BASE_WEIGHT',
            'TARGET_PROCESSING_TIME',
            'PARALLEL_PROCESSING_ENABLED',
            'DTE_RANGE_MIN',
            'DTE_RANGE_MAX',
            'HISTORICAL_YEARS_REQUIRED',
            'MIN_SAMPLE_SIZE_PER_DTE',
            'CONFIDENCE_THRESHOLD',
            'ML_MODEL_ENABLED_RF',
            'ML_MODEL_ENABLED_NN',
            'ROLLING_WINDOW_3MIN',
            'ROLLING_WINDOW_5MIN',
            'REGIME_ACCURACY_TARGET',
            'VALIDATION_ENABLED',
            'EXPORT_VALIDATION_CSV',
            'STATISTICAL_SIGNIFICANCE_MIN',
            'ENSEMBLE_WEIGHTING_METHOD',
            'FEATURE_IMPORTANCE_THRESHOLD',
            'N_ESTIMATORS',
            'MAX_DEPTH',
            'HIDDEN_LAYER_SIZES',
            'EARLY_STOPPING',
            'OPTIMIZATION_LEVEL',
            'MEMORY_LIMIT_MB',
            'CPU_UTILIZATION_TARGET'
        ],
        'Value': [
            True,
            0,
            4,
            0.50,
            0.30,
            0.20,
            3.0,
            True,
            0,
            30,
            3,
            100,
            0.70,
            True,
            True,
            20,
            12,
            0.85,
            True,
            True,
            0.05,
            'weighted_average',
            0.01,
            100,
            10,
            '(100,50)',
            True,
            'aggressive',
            1024,
            80.0
        ],
        'Skill_Level': [
            'Novice',
            'Novice',
            'Novice',
            'Novice',
            'Novice',
            'Novice',
            'Novice',
            'Novice',
            'Intermediate',
            'Intermediate',
            'Intermediate',
            'Intermediate',
            'Intermediate',
            'Intermediate',
            'Intermediate',
            'Intermediate',
            'Intermediate',
            'Intermediate',
            'Intermediate',
            'Intermediate',
            'Expert',
            'Expert',
            'Expert',
            'Expert',
            'Expert',
            'Expert',
            'Expert',
            'Expert',
            'Expert',
            'Expert'
        ],
        'Category': [
            'DTE_Basic',
            'DTE_Basic',
            'DTE_Basic',
            'Weights_Basic',
            'Weights_Basic',
            'Weights_Basic',
            'Performance_Basic',
            'Performance_Basic',
            'DTE_Advanced',
            'DTE_Advanced',
            'Validation_Basic',
            'Validation_Basic',
            'ML_Basic',
            'ML_Basic',
            'ML_Basic',
            'Rolling_Analysis',
            'Rolling_Analysis',
            'Regime_Basic',
            'Validation_Basic',
            'Output_Basic',
            'Statistical_Advanced',
            'ML_Advanced',
            'ML_Advanced',
            'ML_Advanced',
            'ML_Advanced',
            'ML_Advanced',
            'ML_Advanced',
            'Performance_Advanced',
            'System_Advanced',
            'System_Advanced'
        ],
        'Description': [
            'Enable/disable DTE learning framework',
            'Minimum DTE for focus range (0-4 DTE)',
            'Maximum DTE for focus range (0-4 DTE)',
            'Base weight for ATM straddle',
            'Base weight for ITM1 straddle',
            'Base weight for OTM1 straddle',
            'Target processing time in seconds',
            'Enable parallel processing',
            'Minimum DTE value for analysis',
            'Maximum DTE value for analysis',
            'Years of historical data required',
            'Minimum sample size per DTE value',
            'Minimum confidence for regime classification',
            'Enable Random Forest model',
            'Enable Neural Network model',
            'Rolling window size for 3min timeframe',
            'Rolling window size for 5min timeframe',
            'Target accuracy for regime detection',
            'Enable historical validation framework',
            'Export validation results to CSV',
            'Maximum p-value for statistical significance',
            'Method for combining ML model predictions',
            'Minimum feature importance threshold',
            'Number of trees in Random Forest',
            'Maximum depth of trees',
            'Hidden layer architecture for Neural Network',
            'Enable early stopping for Neural Network',
            'Optimization level (conservative/balanced/aggressive)',
            'Memory usage limit (MB)',
            'Target CPU utilization percentage'
        ],
        'Auto_Hide_Complex': [
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True
        ],
        'Show_Help_Text': [
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False
        ],
        'Hot_Reload': [
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            True,
            True,
            True,
            True
        ]
    }

    return pd.DataFrame(config_data)

def create_historical_validation_config_sheet():
    """Create Historical Validation Configuration sheet"""

    config_data = {
        'Parameter': [
            'VALIDATION_ENABLED',
            'HISTORICAL_DATA_PATH',
            'MIN_DATA_SPAN_DAYS',
            'REQUIRED_SAMPLE_SIZE',
            'ACCURACY_THRESHOLD',
            'SHARPE_RATIO_THRESHOLD',
            'MAX_DRAWDOWN_THRESHOLD',
            'VOLATILITY_THRESHOLD_MAX',
            'WIN_RATE_THRESHOLD',
            'STATISTICAL_SIGNIFICANCE_MIN',
            'CONFIDENCE_INTERVAL_LEVEL',
            'VALIDATION_FREQUENCY',
            'AUTO_REVALIDATION',
            'VALIDATION_REPORT_PATH',
            'EXPORT_VALIDATION_CSV',
            'REAL_DATA_ENFORCEMENT',
            'SYNTHETIC_DATA_ALLOWED',
            'DATA_QUALITY_CHECKS'
        ],
        'Value': [
            True,
            'historical_data/',
            1095,
            100,
            0.55,
            0.5,
            0.15,
            0.30,
            0.52,
            0.05,
            0.95,
            'daily',
            True,
            'validation_reports/',
            True,
            True,
            False,
            True
        ],
        'Description': [
            'Enable historical validation framework',
            'Path to historical data files',
            'Minimum data span required (days)',
            'Minimum sample size per DTE',
            'Minimum accuracy threshold',
            'Minimum Sharpe ratio threshold',
            'Maximum allowed drawdown',
            'Maximum volatility threshold',
            'Minimum win rate threshold',
            'Maximum p-value for significance',
            'Confidence interval level',
            'How often to run validation',
            'Enable automatic revalidation',
            'Path for validation reports',
            'Export validation results to CSV',
            'Enforce 100% real data usage',
            'Allow synthetic data fallbacks',
            'Enable data quality validation'
        ],
        'Skill_Level': [
            'Novice',
            'Intermediate',
            'Intermediate',
            'Intermediate',
            'Intermediate',
            'Expert',
            'Expert',
            'Expert',
            'Intermediate',
            'Expert',
            'Expert',
            'Intermediate',
            'Intermediate',
            'Intermediate',
            'Novice',
            'Expert',
            'Expert',
            'Expert'
        ],
        'Category': [
            'Core',
            'Data',
            'Requirements',
            'Requirements',
            'Performance',
            'Performance',
            'Risk',
            'Risk',
            'Performance',
            'Statistical',
            'Statistical',
            'Automation',
            'Automation',
            'Output',
            'Output',
            'Data_Quality',
            'Data_Quality',
            'Data_Quality'
        ],
        'Hot_Reload': [
            True,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            True,
            True,
            True,
            True
        ]
    }

    return pd.DataFrame(config_data)

def create_rolling_analysis_config_sheet():
    """Create Rolling Analysis Configuration sheet"""

    config_data = {
        'Timeframe': [
            '3min',
            '3min',
            '3min',
            '5min',
            '5min',
            '5min',
            '10min',
            '10min',
            '10min',
            '15min',
            '15min',
            '15min',
            'Global',
            'Global',
            'Global',
            'Global',
            'Global',
            'Global'
        ],
        'Parameter': [
            'window',
            'weight',
            'dte_adjustment',
            'window',
            'weight',
            'dte_adjustment',
            'window',
            'weight',
            'dte_adjustment',
            'window',
            'weight',
            'dte_adjustment',
            'rolling_percentage',
            'correlation_window',
            'parallel_enabled',
            'vectorized_enabled',
            'momentum_sensitivity',
            'volatility_sensitivity'
        ],
        'Value': [
            20,
            0.15,
            True,
            12,
            0.35,
            True,
            6,
            0.30,
            True,
            4,
            0.20,
            True,
            1.0,
            20,
            True,
            True,
            1.1,
            1.2
        ],
        'Description': [
            'Rolling window size for 3min timeframe',
            'Weight for 3min timeframe in ensemble',
            'Apply DTE-specific adjustments to 3min',
            'Rolling window size for 5min timeframe',
            'Weight for 5min timeframe in ensemble',
            'Apply DTE-specific adjustments to 5min',
            'Rolling window size for 10min timeframe',
            'Weight for 10min timeframe in ensemble',
            'Apply DTE-specific adjustments to 10min',
            'Rolling window size for 15min timeframe',
            'Weight for 15min timeframe in ensemble',
            'Apply DTE-specific adjustments to 15min',
            'Percentage of data for rolling analysis',
            'Window size for correlation calculations',
            'Enable parallel timeframe processing',
            'Enable vectorized calculations',
            'Momentum calculation sensitivity factor',
            'Volatility calculation sensitivity factor'
        ],
        'Skill_Level': [
            'Intermediate',
            'Novice',
            'Expert',
            'Intermediate',
            'Novice',
            'Expert',
            'Intermediate',
            'Novice',
            'Expert',
            'Intermediate',
            'Novice',
            'Expert',
            'Expert',
            'Intermediate',
            'Intermediate',
            'Expert',
            'Expert',
            'Expert'
        ],
        'Category': [
            'Timeframe',
            'Timeframe',
            'Timeframe',
            'Timeframe',
            'Timeframe',
            'Timeframe',
            'Timeframe',
            'Timeframe',
            'Timeframe',
            'Timeframe',
            'Timeframe',
            'Timeframe',
            'Global',
            'Global',
            'Performance',
            'Performance',
            'Sensitivity',
            'Sensitivity'
        ],
        'Hot_Reload': [
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True
        ]
    }

    return pd.DataFrame(config_data)

def create_regime_classification_config_sheet():
    """Create Regime Classification Configuration sheet"""

    config_data = {
        'Parameter': [
            'NUM_REGIMES',
            'CONFIDENCE_THRESHOLD',
            'ACCURACY_TARGET',
            'DTE_SPECIFIC_ACCURACY',
            'ML_ENHANCED_CLASSIFICATION',
            'MOMENTUM_THRESHOLD_HIGH',
            'MOMENTUM_THRESHOLD_LOW',
            'VOLATILITY_THRESHOLD_HIGH',
            'VOLATILITY_THRESHOLD_LOW',
            'REGIME_STABILITY_WINDOW',
            'RAPID_SWITCHING_THRESHOLD',
            'DTE_ADJUSTMENT_FACTOR_0_1',
            'DTE_ADJUSTMENT_FACTOR_2_4',
            'DTE_ADJUSTMENT_FACTOR_5_PLUS',
            'REGIME_CONFIDENCE_BOOST',
            'ENSEMBLE_REGIME_VOTING',
            'REGIME_PERSISTENCE_FACTOR',
            'CLASSIFICATION_TIMEOUT'
        ],
        'Value': [
            12,
            0.70,
            0.85,
            True,
            True,
            0.02,
            -0.02,
            0.03,
            0.01,
            10,
            0.10,
            1.3,
            1.1,
            0.9,
            0.1,
            True,
            0.8,
            5.0
        ],
        'Description': [
            'Number of market regimes to classify',
            'Minimum confidence for regime classification',
            'Target accuracy for regime detection',
            'Enable DTE-specific accuracy tracking',
            'Enable ML-enhanced classification',
            'High momentum threshold for bullish regimes',
            'Low momentum threshold for bearish regimes',
            'High volatility threshold',
            'Low volatility threshold',
            'Window for regime stability analysis',
            'Threshold for rapid regime switching detection',
            'DTE adjustment factor for 0-1 DTE',
            'DTE adjustment factor for 2-4 DTE',
            'DTE adjustment factor for 5+ DTE',
            'Confidence boost for DTE focus range',
            'Enable ensemble voting for regime classification',
            'Factor for regime persistence weighting',
            'Maximum time for regime classification (seconds)'
        ],
        'Skill_Level': [
            'Intermediate',
            'Novice',
            'Intermediate',
            'Expert',
            'Expert',
            'Expert',
            'Expert',
            'Expert',
            'Expert',
            'Expert',
            'Expert',
            'Expert',
            'Expert',
            'Expert',
            'Expert',
            'Expert',
            'Expert',
            'Expert'
        ],
        'Category': [
            'Core',
            'Core',
            'Performance',
            'DTE_Specific',
            'ML_Enhanced',
            'Thresholds',
            'Thresholds',
            'Thresholds',
            'Thresholds',
            'Stability',
            'Stability',
            'DTE_Adjustment',
            'DTE_Adjustment',
            'DTE_Adjustment',
            'Confidence',
            'Ensemble',
            'Persistence',
            'Performance'
        ],
        'Hot_Reload': [
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True
        ]
    }

    return pd.DataFrame(config_data)

if __name__ == "__main__":
    try:
        excel_file = create_dte_enhanced_configuration_template()
        logger.info(f"🎉 Excel configuration template creation completed successfully!")
        logger.info(f"📁 Template location: {excel_file}")

        # Create JSON version for programmatic access
        json_file = excel_file.parent / "DTE_ENHANCED_CONFIGURATION_TEMPLATE.json"

        # Read back the Excel file and convert to JSON
        config_dict = {}

        # Read each sheet
        sheet_names = [
            'DTE_Learning_Config',
            'ML_Model_Config',
            'Strategy_Config',
            'Performance_Config',
            'UI_Config',
            'Validation_Config',
            'Rolling_Config',
            'Regime_Config'
        ]

        for sheet_name in sheet_names:
            try:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                config_dict[sheet_name] = df.to_dict('records')
            except Exception as e:
                logger.warning(f"⚠️ Could not read sheet {sheet_name}: {e}")

        # Save JSON version
        with open(json_file, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)

        logger.info(f"📄 JSON configuration template created: {json_file}")

    except Exception as e:
        logger.error(f"❌ Failed to create Excel configuration template: {e}")
        raise
