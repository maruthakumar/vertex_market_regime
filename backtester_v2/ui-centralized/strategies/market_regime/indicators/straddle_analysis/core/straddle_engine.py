"""
Triple Straddle Analysis Engine

Main engine that orchestrates all straddle analysis components to provide
comprehensive market regime detection and trading signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from .calculation_engine import CalculationEngine
from .resistance_analyzer import ResistanceAnalyzer, ResistanceAnalysisResult
from ..rolling.window_manager import RollingWindowManager
from ..rolling.correlation_matrix import CorrelationMatrix
from ..config.excel_reader import StraddleConfigReader, StraddleConfig
from ..components.combined_straddle_analyzer import CombinedStraddleAnalyzer, CombinedStraddleResult

import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class TripleStraddleAnalysisResult:
    """Complete result of triple straddle analysis"""
    timestamp: pd.Timestamp
    analysis_time: float  # Time taken for analysis
    
    # Core analysis results
    straddle_result: CombinedStraddleResult
    resistance_result: ResistanceAnalysisResult
    correlation_result: Dict[str, Any]
    
    # Aggregated metrics
    market_regime: str
    regime_confidence: float
    regime_indicators: Dict[str, float]
    
    # Trading recommendations
    position_recommendations: Dict[str, Any]
    risk_parameters: Dict[str, float]
    
    # Performance metrics
    component_status: Dict[str, bool]
    warnings: List[str]


class TripleStraddleEngine:
    """
    Triple Straddle Analysis Engine
    
    Orchestrates all components for comprehensive straddle analysis:
    - 6 individual option components (ATM/ITM1/OTM1 x CE/PE)
    - 3 straddle combinations
    - Combined weighted analysis
    - Support/resistance integration
    - Correlation matrix analysis
    - Market regime detection
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Triple Straddle Engine
        
        Args:
            config_path: Path to Excel configuration file
        """
        self.start_time = time.time()
        
        # Load configuration
        self.config_reader = StraddleConfigReader(config_path)
        self.config = self.config_reader.read_config()
        
        # Initialize core components
        self.calculation_engine = CalculationEngine()
        self.window_manager = RollingWindowManager(
            window_sizes=self.config.rolling_windows,
            max_data_points=500
        )
        
        # Initialize analyzers
        self.resistance_analyzer = ResistanceAnalyzer(
            window_sizes=self.config.rolling_windows
        )
        self.correlation_analyzer = CorrelationMatrixAnalyzer(
            config=self.config,
            window_manager=self.window_manager
        )
        self.straddle_analyzer = CombinedStraddleAnalyzer(
            config=self.config,
            calculation_engine=self.calculation_engine,
            window_manager=self.window_manager
        )
        
        # Performance tracking
        self.analysis_history = []
        self.max_history_length = 100
        self.performance_target = 3.0  # 3 seconds target
        
        # Parallel execution settings
        self.use_parallel = True
        self.max_workers = 4
        
        # Validation flags
        self.validate_inputs = True
        self.validate_outputs = True
        
        init_time = time.time() - self.start_time
        self.logger = logging.getLogger(f"{__name__}.TripleStraddleEngine")
        self.logger.info(f"Triple Straddle Engine initialized in {init_time:.2f} seconds")
    
    def analyze(self, market_data: Dict[str, Any], 
                timestamp: Optional[pd.Timestamp] = None) -> Optional[TripleStraddleAnalysisResult]:
        """
        Perform complete triple straddle analysis
        
        Args:
            market_data: Dictionary containing market data
            timestamp: Analysis timestamp (default: now)
            
        Returns:
            TripleStraddleAnalysisResult or None if analysis fails
        """
        analysis_start = time.time()
        warnings_list = []
        
        try:
            # Set timestamp
            if timestamp is None:
                timestamp = pd.Timestamp.now()
            
            # Validate input data
            if self.validate_inputs:
                validation_result = self._validate_market_data(market_data)
                if not validation_result['valid']:
                    self.logger.error(f"Invalid market data: {validation_result['errors']}")
                    return None
                warnings_list.extend(validation_result.get('warnings', []))
            
            # Run analyses (parallel or sequential)
            if self.use_parallel:
                analysis_results = self._run_parallel_analysis(market_data, timestamp)
            else:
                analysis_results = self._run_sequential_analysis(market_data, timestamp)
            
            # Check component status
            component_status = self._check_component_status(analysis_results)
            
            # Aggregate results
            market_regime, regime_confidence, regime_indicators = self._determine_market_regime(
                analysis_results
            )
            
            # Generate trading recommendations
            position_recommendations = self._generate_position_recommendations(
                analysis_results, market_regime
            )
            
            # Calculate risk parameters
            risk_parameters = self._calculate_risk_parameters(
                analysis_results, position_recommendations
            )
            
            # Create result
            analysis_time = time.time() - analysis_start
            
            result = TripleStraddleAnalysisResult(
                timestamp=timestamp,
                analysis_time=analysis_time,
                straddle_result=analysis_results['straddle'],
                resistance_result=analysis_results['resistance'],
                correlation_result=analysis_results['correlation'],
                market_regime=market_regime,
                regime_confidence=regime_confidence,
                regime_indicators=regime_indicators,
                position_recommendations=position_recommendations,
                risk_parameters=risk_parameters,
                component_status=component_status,
                warnings=warnings_list
            )
            
            # Validate output
            if self.validate_outputs:
                output_validation = self._validate_output(result)
                if not output_validation['valid']:
                    self.logger.error(f"Invalid output: {output_validation['errors']}")
                    return None
            
            # Update history
            self._update_analysis_history(result)
            
            # Log performance
            if analysis_time > self.performance_target:
                self.logger.warning(
                    f"Analysis took {analysis_time:.2f}s, exceeding {self.performance_target}s target"
                )
            else:
                self.logger.debug(f"Analysis completed in {analysis_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in triple straddle analysis: {e}", exc_info=True)
            return None
    
    def _validate_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input market data"""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Required fields
        required_fields = ['underlying_price', 'timestamp']
        for field in required_fields:
            if field not in market_data:
                validation_result['valid'] = False
                validation_result['errors'].append(f"Missing required field: {field}")
        
        # Price validation
        if 'underlying_price' in market_data:
            price = market_data['underlying_price']
            if not isinstance(price, (int, float)) or price <= 0:
                validation_result['valid'] = False
                validation_result['errors'].append(f"Invalid underlying price: {price}")
        
        # Option data validation
        option_fields = ['ATM_CE', 'ATM_PE', 'ITM1_CE', 'ITM1_PE', 'OTM1_CE', 'OTM1_PE']
        missing_options = [f for f in option_fields if f not in market_data]
        
        if len(missing_options) > 3:  # Allow some missing, but not most
            validation_result['warnings'].append(f"Missing option data: {missing_options}")
        
        return validation_result
    
    def _run_parallel_analysis(self, market_data: Dict[str, Any], 
                              timestamp: pd.Timestamp) -> Dict[str, Any]:
        """Run analyses in parallel for performance"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks
            futures = {
                executor.submit(self.straddle_analyzer.analyze, market_data, timestamp): 'straddle',
                executor.submit(self.resistance_analyzer.analyze, market_data, timestamp): 'resistance',
                executor.submit(self._run_correlation_analysis, market_data, timestamp): 'correlation'
            }
            
            # Collect results
            for future in as_completed(futures):
                analysis_type = futures[future]
                try:
                    result = future.result(timeout=2.0)  # 2 second timeout per analysis
                    results[analysis_type] = result
                except Exception as e:
                    self.logger.error(f"Error in {analysis_type} analysis: {e}")
                    results[analysis_type] = None
        
        return results
    
    def _run_sequential_analysis(self, market_data: Dict[str, Any], 
                                timestamp: pd.Timestamp) -> Dict[str, Any]:
        """Run analyses sequentially (fallback)"""
        results = {}
        
        # Straddle analysis
        try:
            results['straddle'] = self.straddle_analyzer.analyze(market_data, timestamp)
        except Exception as e:
            self.logger.error(f"Error in straddle analysis: {e}")
            results['straddle'] = None
        
        # Resistance analysis
        try:
            results['resistance'] = self.resistance_analyzer.analyze(market_data, timestamp)
        except Exception as e:
            self.logger.error(f"Error in resistance analysis: {e}")
            results['resistance'] = None
        
        # Correlation analysis
        try:
            results['correlation'] = self._run_correlation_analysis(market_data, timestamp)
        except Exception as e:
            self.logger.error(f"Error in correlation analysis: {e}")
            results['correlation'] = None
        
        return results
    
    def _run_correlation_analysis(self, market_data: Dict[str, Any], 
                                 timestamp: pd.Timestamp) -> Dict[str, Any]:
        """Run correlation matrix analysis"""
        # Add component data to window manager first
        components = ['atm_ce', 'atm_pe', 'itm1_ce', 'itm1_pe', 'otm1_ce', 'otm1_pe']
        
        for component in components:
            price = market_data.get(component.upper(), 0)
            if price > 0:
                data_point = {
                    'timestamp': timestamp,
                    'close': price,
                    'volume': market_data.get(f'{component}_volume', 0)
                }
                self.window_manager.add_data_point(component, timestamp, data_point)
        
        # Run correlation analysis
        return self.correlation_analyzer.analyze(timestamp)
    
    def _check_component_status(self, analysis_results: Dict[str, Any]) -> Dict[str, bool]:
        """Check status of each component"""
        status = {
            'straddle_analysis': analysis_results.get('straddle') is not None,
            'resistance_analysis': analysis_results.get('resistance') is not None,
            'correlation_analysis': analysis_results.get('correlation') is not None
        }
        
        # Check individual straddles if available
        if analysis_results.get('straddle'):
            straddle_result = analysis_results['straddle']
            status['atm_straddle'] = straddle_result.atm_result is not None
            status['itm1_straddle'] = straddle_result.itm1_result is not None
            status['otm1_straddle'] = straddle_result.otm1_result is not None
        
        return status
    
    def _determine_market_regime(self, analysis_results: Dict[str, Any]) -> Tuple[str, float, Dict[str, float]]:
        """Determine overall market regime from all analyses"""
        regime_indicators = {}
        
        # Get regime indicators from each analysis
        if analysis_results.get('straddle'):
            straddle_indicators = analysis_results['straddle'].regime_indicators
            for key, value in straddle_indicators.items():
                regime_indicators[f'straddle_{key}'] = value
        
        if analysis_results.get('resistance'):
            resistance_indicators = self.resistance_analyzer.get_regime_contribution(
                analysis_results['resistance']
            )
            for key, value in resistance_indicators.items():
                regime_indicators[f'resistance_{key}'] = value
        
        if analysis_results.get('correlation'):
            correlation_indicators = analysis_results['correlation'].get('regime_indicators', {})
            for key, value in correlation_indicators.items():
                regime_indicators[f'correlation_{key}'] = value
        
        # Aggregate regime determination
        if analysis_results.get('straddle'):
            # Use straddle analyzer's regime as primary
            market_regime = analysis_results['straddle'].market_regime
            regime_confidence = analysis_results['straddle'].regime_confidence
        else:
            # Fallback regime determination
            market_regime = 'UNKNOWN'
            regime_confidence = 0.0
        
        return market_regime, regime_confidence, regime_indicators
    
    def _generate_position_recommendations(self, analysis_results: Dict[str, Any],
                                         market_regime: str) -> Dict[str, Any]:
        """Generate position recommendations based on analysis"""
        recommendations = {
            'primary_strategy': 'TRIPLE_STRADDLE',
            'position_size': 1.0,
            'adjustments': {}
        }
        
        # Get straddle recommendations
        if analysis_results.get('straddle'):
            straddle_result = analysis_results['straddle']
            
            # Use optimal weights
            recommendations['straddle_weights'] = straddle_result.optimal_weights
            
            # Get signals
            signals = straddle_result.strategy_signals
            recommendations['entry_signal'] = signals.get('combined_entry_signal', 0.5)
            recommendations['position_size'] = signals.get('position_size_recommendation', 1.0)
            recommendations['action'] = signals.get('recommendation', 'NEUTRAL')
        
        # Get resistance-based adjustments
        if analysis_results.get('resistance'):
            resistance_adjustments = self.resistance_analyzer.get_straddle_adjustments(
                analysis_results['resistance']
            )
            recommendations['adjustments'].update(resistance_adjustments)
        
        # Regime-specific recommendations
        recommendations['regime_adjustments'] = self._get_regime_specific_adjustments(market_regime)
        
        return recommendations
    
    def _get_regime_specific_adjustments(self, market_regime: str) -> Dict[str, Any]:
        """Get regime-specific position adjustments"""
        adjustments = {}
        
        # Parse regime components
        regime_parts = market_regime.split('_')
        
        if 'HIGH_VOL' in regime_parts:
            adjustments['volatility_adjustment'] = 'INCREASE_STRADDLE_SIZE'
            adjustments['preferred_straddle'] = 'OTM1'  # Benefits from high vol
        elif 'LOW_VOL' in regime_parts:
            adjustments['volatility_adjustment'] = 'REDUCE_STRADDLE_SIZE'
            adjustments['preferred_straddle'] = 'ITM1'  # More protected
        
        if 'TRENDING' in market_regime:
            adjustments['trend_adjustment'] = 'INCREASE_DIRECTIONAL_BIAS'
        elif 'RANGING' in market_regime:
            adjustments['trend_adjustment'] = 'MAINTAIN_NEUTRALITY'
        
        return adjustments
    
    def _calculate_risk_parameters(self, analysis_results: Dict[str, Any],
                                  position_recommendations: Dict[str, Any]) -> Dict[str, float]:
        """Calculate risk parameters for position"""
        risk_params = {
            'max_loss': 0.0,
            'daily_theta': 0.0,
            'vega_exposure': 0.0,
            'gamma_exposure': 0.0,
            'breakeven_points': [],
            'risk_score': 0.5
        }
        
        if analysis_results.get('straddle'):
            straddle_result = analysis_results['straddle']
            
            # Extract risk metrics
            risk_params['max_loss'] = straddle_result.combined_price
            risk_params['daily_theta'] = abs(straddle_result.combined_greeks.get('net_theta', 0))
            risk_params['vega_exposure'] = abs(straddle_result.combined_greeks.get('net_vega', 0))
            risk_params['gamma_exposure'] = abs(straddle_result.combined_greeks.get('net_gamma', 0))
            
            # Breakeven points
            metrics = straddle_result.combined_metrics
            if 'combined_upper_breakeven' in metrics:
                risk_params['breakeven_points'] = [
                    metrics.get('combined_lower_breakeven', 0),
                    metrics.get('combined_upper_breakeven', 0)
                ]
            
            # Calculate risk score
            risk_params['risk_score'] = self._calculate_risk_score(risk_params, straddle_result)
        
        return risk_params
    
    def _calculate_risk_score(self, risk_params: Dict[str, float], 
                             straddle_result: CombinedStraddleResult) -> float:
        """Calculate overall risk score (0-1, higher is riskier)"""
        risk_factors = []
        
        # Theta risk
        daily_decay_pct = straddle_result.combined_metrics.get('daily_decay_pct', 0)
        if daily_decay_pct > 5:
            risk_factors.append(1.0)
        elif daily_decay_pct > 2:
            risk_factors.append(0.7)
        else:
            risk_factors.append(0.3)
        
        # Vega risk
        vega = risk_params['vega_exposure']
        if vega > 200:
            risk_factors.append(0.8)
        elif vega > 100:
            risk_factors.append(0.5)
        else:
            risk_factors.append(0.2)
        
        # Efficiency risk
        efficiency = straddle_result.combined_metrics.get('weighted_efficiency', 0.5)
        risk_factors.append(1.0 - efficiency)
        
        return np.mean(risk_factors) if risk_factors else 0.5
    
    def _validate_output(self, result: TripleStraddleAnalysisResult) -> Dict[str, Any]:
        """Validate analysis output"""
        validation = {
            'valid': True,
            'errors': []
        }
        
        # Check for required components
        if result.straddle_result is None:
            validation['valid'] = False
            validation['errors'].append("Missing straddle analysis result")
        
        # Validate regime
        if result.regime_confidence < 0 or result.regime_confidence > 1:
            validation['valid'] = False
            validation['errors'].append(f"Invalid regime confidence: {result.regime_confidence}")
        
        # Validate risk parameters
        if result.risk_parameters['max_loss'] < 0:
            validation['valid'] = False
            validation['errors'].append("Invalid max loss calculation")
        
        return validation
    
    def _update_analysis_history(self, result: TripleStraddleAnalysisResult):
        """Update analysis history"""
        self.analysis_history.append(result)
        
        # Maintain history size
        if len(self.analysis_history) > self.max_history_length:
            self.analysis_history.pop(0)
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get engine status and performance metrics"""
        status = {
            'config': self.config.to_dict(),
            'components': {
                'straddle_analyzer': self.straddle_analyzer.get_analyzer_status(),
                'resistance_analyzer': self.resistance_analyzer.get_analyzer_status(),
                'window_manager': self.window_manager.get_window_status()
            },
            'performance': self._get_performance_metrics(),
            'history_length': len(self.analysis_history)
        }
        
        return status
    
    def _get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics from history"""
        if not self.analysis_history:
            return {}
        
        recent_analyses = self.analysis_history[-20:]  # Last 20 analyses
        
        analysis_times = [r.analysis_time for r in recent_analyses]
        
        return {
            'avg_analysis_time': np.mean(analysis_times),
            'max_analysis_time': np.max(analysis_times),
            'min_analysis_time': np.min(analysis_times),
            'analyses_within_target': sum(1 for t in analysis_times if t <= self.performance_target),
            'target_achievement_rate': sum(1 for t in analysis_times if t <= self.performance_target) / len(analysis_times)
        }
    
    def export_analysis(self, result: TripleStraddleAnalysisResult, 
                       filepath: str, format: str = 'json'):
        """Export analysis result to file"""
        try:
            if format == 'json':
                self._export_json(result, filepath)
            elif format == 'csv':
                self._export_csv(result, filepath)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Analysis exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error exporting analysis: {e}")
    
    def _export_json(self, result: TripleStraddleAnalysisResult, filepath: str):
        """Export to JSON format"""
        export_data = {
            'timestamp': result.timestamp.isoformat(),
            'analysis_time': result.analysis_time,
            'market_regime': result.market_regime,
            'regime_confidence': result.regime_confidence,
            'regime_indicators': result.regime_indicators,
            'position_recommendations': result.position_recommendations,
            'risk_parameters': result.risk_parameters,
            'warnings': result.warnings
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def _export_csv(self, result: TripleStraddleAnalysisResult, filepath: str):
        """Export to CSV format"""
        # Flatten the result for CSV
        flat_data = {
            'timestamp': result.timestamp,
            'analysis_time': result.analysis_time,
            'market_regime': result.market_regime,
            'regime_confidence': result.regime_confidence
        }
        
        # Add regime indicators
        for key, value in result.regime_indicators.items():
            flat_data[f'regime_{key}'] = value
        
        # Add risk parameters
        for key, value in result.risk_parameters.items():
            if not isinstance(value, list):
                flat_data[f'risk_{key}'] = value
        
        # Create DataFrame and save
        df = pd.DataFrame([flat_data])
        df.to_csv(filepath, index=False)