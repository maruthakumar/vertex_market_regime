#!/usr/bin/env python3
"""
Enhanced Excel Configuration Template for Enhanced Triple Straddle Framework v2.0
Comprehensive unified input sheets design supporting all enhanced features

Author: The Augster
Date: 2025-06-20
Version: 2.0.0 (Enhanced Configuration Template)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class EnhancedMarketRegimeExcelTemplate:
    """
    Enhanced Excel configuration template for Enhanced Triple Straddle Framework v2.0
    
    Provides comprehensive unified input sheets supporting:
    - Delta-based strike selection parameters
    - Volume-weighting configuration
    - 18-regime classification thresholds
    - Hybrid system integration weights
    - Mathematical accuracy validation parameters
    - Performance monitoring configuration
    """
    
    def __init__(self):
        """Initialize enhanced Excel template"""
        
        self.template_structure = self._define_enhanced_template_structure()
        logger.info("🎯 Enhanced Market Regime Excel Template initialized")
    
    def _define_enhanced_template_structure(self) -> Dict[str, Dict[str, Any]]:
        """Define the enhanced Excel template structure"""
        
        return {
            # Enhanced Greek Sentiment Configuration
            'EnhancedGreekSentimentConfig': {
                'description': 'Volume-weighted Greek sentiment analysis with delta-based strike selection',
                'columns': [
                    'Parameter', 'Value', 'Description', 'DataType', 'MinValue', 'MaxValue', 'Category'
                ],
                'default_data': [
                    # Delta-based Strike Selection
                    ['CallDeltaRangeMin', 0.01, 'Minimum delta for CALL options inclusion', 'float', 0.001, 0.1, 'StrikeSelection'],
                    ['CallDeltaRangeMax', 0.5, 'Maximum delta for CALL options inclusion', 'float', 0.1, 0.8, 'StrikeSelection'],
                    ['PutDeltaRangeMin', -0.5, 'Minimum delta for PUT options inclusion', 'float', -0.8, -0.1, 'StrikeSelection'],
                    ['PutDeltaRangeMax', -0.01, 'Maximum delta for PUT options inclusion', 'float', -0.1, -0.001, 'StrikeSelection'],
                    ['MinVolumeThreshold', 10, 'Minimum volume for option inclusion', 'int', 1, 100, 'StrikeSelection'],
                    ['MinOpenInterest', 100, 'Minimum open interest for option inclusion', 'int', 10, 1000, 'StrikeSelection'],
                    
                    # Volume Weighting Configuration
                    ['EnableVolumeWeighting', 'YES', 'Enable volume-weighted Greek calculations', 'bool', 'YES', 'NO', 'VolumeWeighting'],
                    ['OptionMultiplier', 50, 'Contract multiplier for NIFTY options', 'int', 25, 100, 'VolumeWeighting'],
                    ['VolumeNormalizationMethod', 'MAX_VOLUME', 'Volume normalization method', 'str', 'MAX_VOLUME', 'PERCENTILE', 'VolumeWeighting'],
                    
                    # Expiry Weighting (Near Expiry Focus)
                    ['ExpiryWeight0DTE', 0.70, 'Weight for 0 DTE options (same day expiry)', 'float', 0.5, 0.9, 'ExpiryWeighting'],
                    ['ExpiryWeight1DTE', 0.20, 'Weight for 1 DTE options', 'float', 0.1, 0.4, 'ExpiryWeighting'],
                    ['ExpiryWeight2DTE', 0.07, 'Weight for 2 DTE options', 'float', 0.0, 0.2, 'ExpiryWeighting'],
                    ['ExpiryWeight3DTE', 0.03, 'Weight for 3 DTE options', 'float', 0.0, 0.1, 'ExpiryWeighting'],
                    
                    # Greek Component Weights
                    ['DeltaComponentWeight', 0.40, 'Delta component weight in sentiment score', 'float', 0.2, 0.6, 'GreekWeights'],
                    ['GammaComponentWeight', 0.30, 'Gamma component weight in sentiment score', 'float', 0.1, 0.5, 'GreekWeights'],
                    ['ThetaComponentWeight', 0.20, 'Theta component weight in sentiment score', 'float', 0.1, 0.4, 'GreekWeights'],
                    ['VegaComponentWeight', 0.10, 'Vega component weight in sentiment score', 'float', 0.0, 0.3, 'GreekWeights'],
                    
                    # Normalization Factors for Tanh
                    ['DeltaNormalizationFactor', 100000, 'Normalization factor for delta exposure', 'int', 50000, 200000, 'Normalization'],
                    ['GammaNormalizationFactor', 50000, 'Normalization factor for gamma exposure', 'int', 25000, 100000, 'Normalization'],
                    ['ThetaNormalizationFactor', 10000, 'Normalization factor for theta exposure', 'int', 5000, 20000, 'Normalization'],
                    ['VegaNormalizationFactor', 20000, 'Normalization factor for vega exposure', 'int', 10000, 40000, 'Normalization'],
                    
                    # Baseline Configuration
                    ['BaselineEstablishmentTime', '09:15:00', 'Time for establishing opening baseline', 'time', '09:15:00', '09:30:00', 'Baseline'],
                    ['BaselineUpdateFrequency', 60, 'Baseline update frequency in seconds', 'int', 30, 300, 'Baseline'],
                    ['EnableBaselineDrift', 'NO', 'Enable automatic baseline drift correction', 'bool', 'YES', 'NO', 'Baseline']
                ]
            },
            
            # Hybrid Classification System Configuration
            'HybridClassificationConfig': {
                'description': 'Hybrid classification system integration (Enhanced + Stable systems)',
                'columns': [
                    'Parameter', 'Value', 'Description', 'DataType', 'MinValue', 'MaxValue', 'Category'
                ],
                'default_data': [
                    # System Integration Weights
                    ['EnhancedSystemWeight', 0.70, 'Weight for Enhanced 18-regime system', 'float', 0.5, 0.9, 'SystemWeights'],
                    ['StableSystemWeight', 0.30, 'Weight for existing timeframe hierarchy system', 'float', 0.1, 0.5, 'SystemWeights'],
                    
                    # Enhanced System Component Weights
                    ['StraddleAnalysisWeight', 0.40, 'Enhanced Triple Straddle analysis weight', 'float', 0.2, 0.6, 'EnhancedComponents'],
                    ['GreekSentimentWeight', 0.30, 'Volume-weighted Greek sentiment weight', 'float', 0.2, 0.5, 'EnhancedComponents'],
                    ['OIPatternRecognitionWeight', 0.20, 'Advanced OI pattern recognition weight', 'float', 0.1, 0.4, 'EnhancedComponents'],
                    ['TechnicalAnalysisWeight', 0.10, 'Enhanced technical analysis weight', 'float', 0.0, 0.3, 'EnhancedComponents'],
                    
                    # Stable System Timeframe Weights
                    ['Timeframe1MinWeight', 0.05, 'Weight for 1-minute timeframe analysis', 'float', 0.0, 0.2, 'StableTimeframes'],
                    ['Timeframe5MinWeight', 0.15, 'Weight for 5-minute timeframe analysis', 'float', 0.1, 0.3, 'StableTimeframes'],
                    ['Timeframe15MinWeight', 0.25, 'Weight for 15-minute timeframe analysis', 'float', 0.1, 0.4, 'StableTimeframes'],
                    ['Timeframe30MinWeight', 0.35, 'Weight for 30-minute timeframe analysis', 'float', 0.2, 0.5, 'StableTimeframes'],
                    ['OpeningWeight', 0.10, 'Weight for opening analysis', 'float', 0.0, 0.2, 'StableTimeframes'],
                    ['PreviousDayWeight', 0.10, 'Weight for previous day analysis', 'float', 0.0, 0.2, 'StableTimeframes'],
                    
                    # Agreement and Confidence Configuration
                    ['AgreementWeight', 0.30, 'Weight for system agreement in confidence calculation', 'float', 0.1, 0.5, 'Confidence'],
                    ['MinConfidenceThreshold', 0.50, 'Minimum confidence threshold for regime classification', 'float', 0.3, 0.8, 'Confidence'],
                    ['TransitionProbabilityWeight', 0.40, 'Weight for transition probability calculation', 'float', 0.2, 0.6, 'Confidence'],
                    
                    # Integration Method Configuration
                    ['IntegrationMethod', 'WEIGHTED_AVERAGE', 'Method for integrating system results', 'str', 'WEIGHTED_AVERAGE', 'VOTING', 'Integration'],
                    ['EnableDualOutput', 'YES', 'Enable dual output (18-regime + timeframe)', 'bool', 'YES', 'NO', 'Integration'],
                    ['ConflictResolutionMethod', 'ENHANCED_PRIORITY', 'Method for resolving system conflicts', 'str', 'ENHANCED_PRIORITY', 'CONFIDENCE_BASED', 'Integration']
                ]
            },
            
            # 18-Regime Classification Thresholds
            'Regime18ClassificationConfig': {
                'description': 'Deterministic 18-regime classification thresholds and mappings',
                'columns': [
                    'RegimeID', 'RegimeName', 'ScoreThresholdMin', 'ScoreThresholdMax', 'VolatilityLevel', 'Description'
                ],
                'default_data': [
                    # Bullish Regimes (1-6)
                    [1, 'High_Volatile_Strong_Bullish', 0.5, 1.0, 'High', 'Strong bullish with high volatility'],
                    [2, 'Normal_Volatile_Strong_Bullish', 0.5, 1.0, 'Normal', 'Strong bullish with normal volatility'],
                    [3, 'Low_Volatile_Strong_Bullish', 0.5, 1.0, 'Low', 'Strong bullish with low volatility'],
                    [4, 'High_Volatile_Mild_Bullish', 0.2, 0.5, 'High', 'Mild bullish with high volatility'],
                    [5, 'Normal_Volatile_Mild_Bullish', 0.2, 0.5, 'Normal', 'Mild bullish with normal volatility'],
                    [6, 'Low_Volatile_Mild_Bullish', 0.2, 0.5, 'Low', 'Mild bullish with low volatility'],
                    
                    # Neutral Regimes (7-12)
                    [7, 'High_Volatile_Neutral', -0.1, 0.1, 'High', 'Neutral with high volatility'],
                    [8, 'Normal_Volatile_Neutral', -0.1, 0.1, 'Normal', 'Neutral with normal volatility'],
                    [9, 'Low_Volatile_Neutral', -0.1, 0.1, 'Low', 'Neutral with low volatility'],
                    [10, 'High_Volatile_Sideways', -0.2, 0.2, 'High', 'Sideways with high volatility'],
                    [11, 'Normal_Volatile_Sideways', -0.2, 0.2, 'Normal', 'Sideways with normal volatility'],
                    [12, 'Low_Volatile_Sideways', -0.2, 0.2, 'Low', 'Sideways with low volatility'],
                    
                    # Bearish Regimes (13-18)
                    [13, 'High_Volatile_Mild_Bearish', -0.5, -0.2, 'High', 'Mild bearish with high volatility'],
                    [14, 'Normal_Volatile_Mild_Bearish', -0.5, -0.2, 'Normal', 'Mild bearish with normal volatility'],
                    [15, 'Low_Volatile_Mild_Bearish', -0.5, -0.2, 'Low', 'Mild bearish with low volatility'],
                    [16, 'High_Volatile_Strong_Bearish', -1.0, -0.5, 'High', 'Strong bearish with high volatility'],
                    [17, 'Normal_Volatile_Strong_Bearish', -1.0, -0.5, 'Normal', 'Strong bearish with normal volatility'],
                    [18, 'Low_Volatile_Strong_Bearish', -1.0, -0.5, 'Low', 'Strong bearish with low volatility']
                ]
            },
            
            # Mathematical Accuracy Configuration
            'MathematicalAccuracyConfig': {
                'description': 'Mathematical accuracy validation and tolerance configuration',
                'columns': [
                    'Parameter', 'Value', 'Description', 'DataType', 'MinValue', 'MaxValue', 'Category'
                ],
                'default_data': [
                    # Tolerance Configuration
                    ['MathematicalTolerance', 0.001, 'Mathematical accuracy tolerance (±)', 'float', 0.0001, 0.01, 'Tolerance'],
                    ['VolumeWeightedCalculationTolerance', 0.001, 'Tolerance for volume-weighted calculations', 'float', 0.0001, 0.01, 'Tolerance'],
                    ['GreekExposureTolerance', 0.001, 'Tolerance for Greek exposure calculations', 'float', 0.0001, 0.01, 'Tolerance'],
                    ['PatternCorrelationTolerance', 0.001, 'Tolerance for pattern correlation calculations', 'float', 0.0001, 0.01, 'Tolerance'],
                    
                    # Validation Configuration
                    ['EnableContinuousValidation', 'YES', 'Enable continuous mathematical validation', 'bool', 'YES', 'NO', 'Validation'],
                    ['ValidationFrequencySeconds', 300, 'Frequency of validation checks in seconds', 'int', 60, 3600, 'Validation'],
                    ['ValidationSampleSizes', '100,500,1000,5000', 'Sample sizes for validation testing', 'str', '100', '10000', 'Validation'],
                    ['StressTestIterations', 10, 'Number of stress test iterations', 'int', 5, 50, 'Validation'],
                    
                    # Alert Configuration
                    ['AccuracyAlertThreshold', 0.80, 'Accuracy threshold for alerts', 'float', 0.5, 0.95, 'Alerts'],
                    ['ToleranceViolationAlertThreshold', 0.01, 'Tolerance violation threshold for alerts', 'float', 0.001, 0.1, 'Alerts'],
                    ['EnableAccuracyAlerts', 'YES', 'Enable accuracy violation alerts', 'bool', 'YES', 'NO', 'Alerts'],
                    ['AlertChannels', 'EMAIL,LOG', 'Alert delivery channels', 'str', 'LOG', 'EMAIL,SMS,TELEGRAM', 'Alerts']
                ]
            },
            
            # Enhanced OI Pattern Recognition Configuration
            'EnhancedOIPatternRecognitionConfig': {
                'description': 'Advanced OI pattern recognition with mathematical correlation',
                'columns': [
                    'Parameter', 'Value', 'Description', 'DataType', 'MinValue', 'MaxValue', 'Category'
                ],
                'default_data': [
                    # Mathematical Correlation Parameters
                    ['CorrelationThreshold', 0.80, 'Pearson correlation threshold for pattern similarity', 'float', 0.5, 0.95, 'Correlation'],
                    ['TimeDecayFactor', 0.1, 'Time decay factor λ for exp(-λ×(T-t))', 'float', 0.05, 0.5, 'Correlation'],
                    ['PatternLookbackPeriod', 20, 'Number of periods for pattern matching', 'int', 10, 50, 'Correlation'],
                    ['MinPatternLength', 5, 'Minimum pattern length for correlation', 'int', 3, 10, 'Correlation'],
                    ['MaxPatternAge', 100, 'Maximum pattern age in periods', 'int', 50, 500, 'Correlation'],

                    # Advanced Pattern Recognition (Preserve Existing)
                    ['StrikeRange', 7, 'Strike range around ATM (±7 strikes)', 'int', 3, 15, 'PatternRecognition'],
                    ['VolatilityAdjustment', 'YES', 'Enable volatility-based strike range adjustment', 'bool', 'YES', 'NO', 'PatternRecognition'],
                    ['DivergenceThreshold', 0.30, 'Threshold for divergence detection (30%)', 'float', 0.1, 0.5, 'PatternRecognition'],
                    ['DivergenceWindow', 10, 'Window for divergence analysis', 'int', 5, 20, 'PatternRecognition'],

                    # Multi-timeframe Analysis (Preserve Existing)
                    ['PrimaryTimeframe', 3, 'Primary timeframe in minutes', 'int', 1, 5, 'Timeframes'],
                    ['ConfirmationTimeframe', 15, 'Confirmation timeframe in minutes', 'int', 10, 30, 'Timeframes'],
                    ['TimeframeWeight3Min', 0.40, 'Weight for 3-minute timeframe', 'float', 0.2, 0.6, 'Timeframes'],
                    ['TimeframeWeight15Min', 0.60, 'Weight for 15-minute timeframe', 'float', 0.4, 0.8, 'Timeframes'],

                    # Institutional vs Retail Detection (Preserve Existing)
                    ['InstitutionalLotSize', 1000, 'Lot size threshold for institutional detection', 'int', 500, 5000, 'InstitutionalDetection'],
                    ['InstitutionalOIThreshold', 0.60, 'OI concentration threshold for institutional activity', 'float', 0.4, 0.8, 'InstitutionalDetection'],
                    ['VolumeWeightFactor', 0.30, 'Volume weighting factor', 'float', 0.1, 0.5, 'InstitutionalDetection'],
                    ['MinVolumeThreshold', 100, 'Minimum volume threshold for inclusion', 'int', 50, 500, 'InstitutionalDetection'],

                    # Session-based Weighting (Preserve Existing)
                    ['MarketOpenWeight', 1.2, 'Weight for market open session (9:15-10:30)', 'float', 1.0, 1.5, 'SessionWeights'],
                    ['MidSessionWeight', 1.0, 'Weight for mid session (10:30-14:30)', 'float', 0.8, 1.2, 'SessionWeights'],
                    ['MarketCloseWeight', 1.3, 'Weight for market close session (14:30-15:30)', 'float', 1.0, 1.5, 'SessionWeights'],

                    # Integration with Enhanced System
                    ['EnableMathematicalCorrelation', 'YES', 'Enable mathematical correlation analysis', 'bool', 'YES', 'NO', 'Integration'],
                    ['CorrelationWeight', 0.30, 'Weight for correlation analysis in final signal', 'float', 0.1, 0.5, 'Integration'],
                    ['PreserveExistingFeatures', 'YES', 'Preserve all existing advanced features', 'bool', 'YES', 'NO', 'Integration'],
                    ['OIPatternWeight', 0.20, 'Weight for OI pattern recognition in hybrid system', 'float', 0.1, 0.4, 'Integration']
                ]
            },

            # Performance Monitoring Configuration
            'PerformanceMonitoringConfig': {
                'description': 'Performance targets and monitoring configuration',
                'columns': [
                    'Parameter', 'Value', 'Description', 'DataType', 'MinValue', 'MaxValue', 'Category'
                ],
                'default_data': [
                    # Processing Time Targets
                    ['MaxProcessingTimeSeconds', 3.0, 'Maximum processing time per minute of data', 'float', 1.0, 10.0, 'ProcessingTime'],
                    ['ProcessingTimeAlertThreshold', 4.0, 'Processing time threshold for alerts', 'float', 2.0, 15.0, 'ProcessingTime'],
                    ['EnableProcessingTimeMonitoring', 'YES', 'Enable processing time monitoring', 'bool', 'YES', 'NO', 'ProcessingTime'],

                    # Accuracy Targets
                    ['MinAccuracyThreshold', 0.85, 'Minimum regime classification accuracy', 'float', 0.7, 0.95, 'Accuracy'],
                    ['AccuracyMeasurementWindow', 100, 'Number of classifications for accuracy measurement', 'int', 50, 500, 'Accuracy'],
                    ['EnableAccuracyTracking', 'YES', 'Enable accuracy tracking and validation', 'bool', 'YES', 'NO', 'Accuracy'],

                    # Memory and Resource Monitoring
                    ['MaxMemoryUsageGB', 2.0, 'Maximum memory usage in GB', 'float', 0.5, 8.0, 'Resources'],
                    ['MemoryAlertThreshold', 1.5, 'Memory usage threshold for alerts', 'float', 0.5, 6.0, 'Resources'],
                    ['EnableResourceMonitoring', 'YES', 'Enable resource usage monitoring', 'bool', 'YES', 'NO', 'Resources'],

                    # Real-time Monitoring
                    ['MonitoringIntervalSeconds', 60, 'Real-time monitoring interval in seconds', 'int', 30, 300, 'Monitoring'],
                    ['PerformanceHistoryLimit', 1000, 'Number of performance records to keep', 'int', 100, 5000, 'Monitoring'],
                    ['EnablePerformanceDashboard', 'YES', 'Enable performance monitoring dashboard', 'bool', 'YES', 'NO', 'Monitoring']
                ]
            }
        }
    
    def generate_enhanced_excel_template(self, output_path: str) -> str:
        """
        Generate enhanced Excel configuration template
        
        Args:
            output_path (str): Path where to save the template
            
        Returns:
            str: Path to generated template file
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                for sheet_name, sheet_config in self.template_structure.items():
                    # Create DataFrame from template data
                    df = pd.DataFrame(
                        sheet_config['default_data'],
                        columns=sheet_config['columns']
                    )
                    
                    # Write to Excel
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Add description as a comment (if supported)
                    worksheet = writer.sheets[sheet_name]
                    try:
                        from openpyxl.comments import Comment
                        worksheet.cell(1, 1).comment = Comment(sheet_config['description'], 'Enhanced System')
                    except:
                        # Skip comment if not supported
                        pass
            
            logger.info(f"📊 Enhanced Excel template generated: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"❌ Error generating enhanced Excel template: {e}")
            raise
    
    def validate_configuration(self, config_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Validate enhanced configuration data
        
        Args:
            config_data: Dictionary of configuration DataFrames
            
        Returns:
            Dict with validation results
        """
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Validate Greek component weights sum to 1.0
            if 'EnhancedGreekSentimentConfig' in config_data:
                df = config_data['EnhancedGreekSentimentConfig']
                weight_params = ['DeltaComponentWeight', 'GammaComponentWeight', 'ThetaComponentWeight', 'VegaComponentWeight']
                
                weights = {}
                for param in weight_params:
                    param_row = df[df['Parameter'] == param]
                    if not param_row.empty:
                        weights[param] = float(param_row.iloc[0]['Value'])
                
                total_weight = sum(weights.values())
                if abs(total_weight - 1.0) > 0.001:
                    validation_results['errors'].append(f"Greek component weights sum to {total_weight:.3f}, should be 1.0")
                    validation_results['valid'] = False
            
            # Validate hybrid system weights sum to 1.0
            if 'HybridClassificationConfig' in config_data:
                df = config_data['HybridClassificationConfig']
                enhanced_weight_row = df[df['Parameter'] == 'EnhancedSystemWeight']
                stable_weight_row = df[df['Parameter'] == 'StableSystemWeight']

                if not enhanced_weight_row.empty and not stable_weight_row.empty:
                    enhanced_weight = float(enhanced_weight_row.iloc[0]['Value'])
                    stable_weight = float(stable_weight_row.iloc[0]['Value'])
                    total_weight = enhanced_weight + stable_weight

                    if abs(total_weight - 1.0) > 0.001:
                        validation_results['errors'].append(f"Hybrid system weights sum to {total_weight:.3f}, should be 1.0")
                        validation_results['valid'] = False

            # Validate OI pattern recognition correlation threshold
            if 'EnhancedOIPatternRecognitionConfig' in config_data:
                df = config_data['EnhancedOIPatternRecognitionConfig']
                correlation_threshold_row = df[df['Parameter'] == 'CorrelationThreshold']

                if not correlation_threshold_row.empty:
                    correlation_threshold = float(correlation_threshold_row.iloc[0]['Value'])

                    if correlation_threshold < 0.5:
                        validation_results['warnings'].append(f"Correlation threshold {correlation_threshold:.3f} is below recommended minimum 0.5")
                    elif correlation_threshold < 0.8:
                        validation_results['warnings'].append(f"Correlation threshold {correlation_threshold:.3f} is below optimal value 0.8")

                # Validate timeframe weights sum to 1.0
                timeframe_3min_row = df[df['Parameter'] == 'TimeframeWeight3Min']
                timeframe_15min_row = df[df['Parameter'] == 'TimeframeWeight15Min']

                if not timeframe_3min_row.empty and not timeframe_15min_row.empty:
                    weight_3min = float(timeframe_3min_row.iloc[0]['Value'])
                    weight_15min = float(timeframe_15min_row.iloc[0]['Value'])
                    total_timeframe_weight = weight_3min + weight_15min

                    if abs(total_timeframe_weight - 1.0) > 0.001:
                        validation_results['errors'].append(f"OI timeframe weights sum to {total_timeframe_weight:.3f}, should be 1.0")
                        validation_results['valid'] = False
            
            # Validate 18-regime thresholds
            if 'Regime18ClassificationConfig' in config_data:
                df = config_data['Regime18ClassificationConfig']
                
                if len(df) != 18:
                    validation_results['errors'].append(f"Expected 18 regimes, found {len(df)}")
                    validation_results['valid'] = False
                
                # Check for overlapping thresholds
                for volatility in ['High', 'Normal', 'Low']:
                    vol_regimes = df[df['VolatilityLevel'] == volatility].sort_values('ScoreThresholdMin')
                    
                    for i in range(len(vol_regimes) - 1):
                        current_max = vol_regimes.iloc[i]['ScoreThresholdMax']
                        next_min = vol_regimes.iloc[i + 1]['ScoreThresholdMin']
                        
                        if current_max > next_min:
                            validation_results['warnings'].append(
                                f"Overlapping thresholds in {volatility} volatility regimes"
                            )
            
            logger.info(f"✅ Configuration validation: {'PASS' if validation_results['valid'] else 'FAIL'}")
            
        except Exception as e:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Validation error: {str(e)}")
            logger.error(f"❌ Configuration validation error: {e}")
        
        return validation_results

def main():
    """Main function for testing enhanced Excel template"""
    
    logger.info("🚀 Testing Enhanced Excel Configuration Template")
    
    # Initialize template
    template = EnhancedMarketRegimeExcelTemplate()
    
    # Generate template
    output_path = "enhanced_market_regime_config_template.xlsx"
    template_path = template.generate_enhanced_excel_template(output_path)
    
    logger.info("🎯 Enhanced Excel Template Testing Complete")
    
    return template_path

if __name__ == "__main__":
    main()
