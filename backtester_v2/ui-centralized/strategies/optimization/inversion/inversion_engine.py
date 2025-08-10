"""
Inversion Engine - Master Controller for Strategy Inversions

Orchestrates the complete strategy inversion workflow including analysis, 
pattern detection, risk assessment, and execution of optimal inversions.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import json
import time
from pathlib import Path

from .strategy_inverter import StrategyInverter, InversionType, InversionResult
from .inversion_analyzer import InversionAnalyzer, InversionOpportunity
from ..base.base_optimizer import BaseOptimizer, OptimizationResult

logger = logging.getLogger(__name__)

@dataclass
class InversionEngineConfig:
    """Configuration for inversion engine"""
    enable_simple_inversion: bool = True
    enable_ulta_inversion: bool = True
    enable_conditional_inversion: bool = True
    enable_rolling_inversion: bool = True
    enable_risk_adjusted_inversion: bool = True
    enable_smart_inversion: bool = False
    
    # Thresholds
    min_improvement_threshold: float = 0.05
    confidence_threshold: float = 0.7
    risk_threshold: float = 0.8
    
    # Analysis parameters
    analysis_window: int = 252
    pattern_detection_threshold: float = 0.05
    enable_ml_analysis: bool = True
    
    # Execution parameters
    max_inversions_per_run: int = 10
    enable_batch_processing: bool = True
    save_intermediate_results: bool = True

@dataclass
class InversionExecutionResult:
    """Result from inversion execution"""
    strategy_name: str
    execution_status: str
    inversion_result: Optional[InversionResult]
    execution_time: float
    error_message: Optional[str]
    pre_execution_metrics: Dict[str, float]
    post_execution_metrics: Dict[str, float]

@dataclass
class InversionEngineResult:
    """Complete result from inversion engine"""
    engine_config: InversionEngineConfig
    execution_summary: Dict[str, Any]
    strategy_analyses: Dict[str, Any]
    inversion_opportunities: List[InversionOpportunity]
    execution_results: List[InversionExecutionResult]
    portfolio_analysis: Dict[str, Any]
    performance_improvement: Dict[str, float]
    risk_analysis: Dict[str, Any]
    recommendations: List[str]
    execution_metadata: Dict[str, Any]

class InversionEngine:
    """
    Master engine for strategy inversion operations
    
    Provides a unified interface for analyzing strategies, identifying inversion
    opportunities, and executing optimal inversions with comprehensive reporting.
    """
    
    def __init__(self,
                 config: Optional[InversionEngineConfig] = None,
                 output_directory: Optional[str] = None,
                 enable_logging: bool = True):
        """
        Initialize inversion engine
        
        Args:
            config: Engine configuration
            output_directory: Directory for saving results
            enable_logging: Enable detailed logging
        """
        self.config = config or InversionEngineConfig()
        self.output_directory = Path(output_directory) if output_directory else None
        self.enable_logging = enable_logging
        
        # Initialize components
        self.strategy_inverter = StrategyInverter(
            min_improvement_threshold=self.config.min_improvement_threshold,
            confidence_threshold=self.config.confidence_threshold,
            enable_advanced_inversions=any([
                self.config.enable_conditional_inversion,
                self.config.enable_rolling_inversion,
                self.config.enable_risk_adjusted_inversion,
                self.config.enable_smart_inversion
            ])
        )
        
        self.inversion_analyzer = InversionAnalyzer(
            analysis_window=self.config.analysis_window,
            pattern_detection_threshold=self.config.pattern_detection_threshold,
            enable_ml_analysis=self.config.enable_ml_analysis
        )
        
        # Execution tracking
        self.execution_history = []
        self.performance_metrics = {
            'total_strategies_analyzed': 0,
            'successful_inversions': 0,
            'total_improvement': 0.0,
            'average_execution_time': 0.0,
            'error_count': 0
        }
        
        # Create output directory if specified
        if self.output_directory:
            self.output_directory.mkdir(parents=True, exist_ok=True)
        
        logger.info("InversionEngine initialized")
    
    def analyze_and_invert_portfolio(self,
                                   strategy_data: pd.DataFrame,
                                   strategy_columns: List[str],
                                   market_data: Optional[pd.DataFrame] = None,
                                   force_inversions: List[str] = None,
                                   exclude_strategies: List[str] = None) -> InversionEngineResult:
        """
        Complete pipeline: analyze portfolio and execute optimal inversions
        
        Args:
            strategy_data: DataFrame with strategy returns
            strategy_columns: List of strategy column names
            market_data: Optional market data for context
            force_inversions: Strategies to force invert regardless of analysis
            exclude_strategies: Strategies to exclude from inversion
            
        Returns:
            Complete inversion engine result
        """
        start_time = time.time()
        logger.info(f"Starting portfolio inversion analysis for {len(strategy_columns)} strategies")
        
        # Filter strategies
        active_strategies = self._filter_strategies(
            strategy_columns, force_inversions or [], exclude_strategies or []
        )
        
        # Phase 1: Portfolio Analysis
        logger.info("Phase 1: Analyzing portfolio for inversion opportunities")
        portfolio_analysis = self.inversion_analyzer.analyze_strategy_portfolio(
            strategy_data, active_strategies, market_data
        )
        
        # Phase 2: Opportunity Prioritization
        logger.info("Phase 2: Prioritizing inversion opportunities")
        prioritized_opportunities = self._prioritize_opportunities(
            portfolio_analysis['inversion_opportunities']
        )
        
        # Phase 3: Risk Assessment
        logger.info("Phase 3: Assessing inversion risks")
        risk_analysis = self._comprehensive_risk_assessment(
            strategy_data, active_strategies, prioritized_opportunities
        )
        
        # Phase 4: Execution Planning
        logger.info("Phase 4: Planning inversion execution")
        execution_plan = self._create_execution_plan(
            prioritized_opportunities, risk_analysis, force_inversions or []
        )
        
        # Phase 5: Execute Inversions
        logger.info(f"Phase 5: Executing {len(execution_plan)} inversions")
        execution_results = self._execute_inversions(
            strategy_data, execution_plan, market_data
        )
        
        # Phase 6: Performance Analysis
        logger.info("Phase 6: Analyzing performance improvements")
        performance_improvement = self._analyze_performance_improvement(
            strategy_data, execution_results
        )
        
        # Phase 7: Generate Recommendations
        logger.info("Phase 7: Generating recommendations")
        recommendations = self._generate_comprehensive_recommendations(
            portfolio_analysis, execution_results, performance_improvement
        )
        
        # Create comprehensive result
        execution_time = time.time() - start_time
        result = InversionEngineResult(
            engine_config=self.config,
            execution_summary={
                'total_execution_time': execution_time,
                'strategies_analyzed': len(active_strategies),
                'opportunities_identified': len(prioritized_opportunities),
                'inversions_executed': len(execution_results),
                'successful_inversions': sum(1 for r in execution_results if r.execution_status == 'success'),
                'total_improvement_score': sum(performance_improvement.values())
            },
            strategy_analyses=portfolio_analysis['strategy_analyses'],
            inversion_opportunities=prioritized_opportunities,
            execution_results=execution_results,
            portfolio_analysis=portfolio_analysis,
            performance_improvement=performance_improvement,
            risk_analysis=risk_analysis,
            recommendations=recommendations,
            execution_metadata={
                'execution_timestamp': pd.Timestamp.now(),
                'engine_version': '1.0.0',
                'config_used': asdict(self.config)
            }
        )
        
        # Save results if output directory specified
        if self.output_directory:
            self._save_results(result)
        
        # Update performance metrics
        self._update_performance_metrics(result)
        
        logger.info(f"Portfolio inversion completed in {execution_time:.2f}s")
        logger.info(f"Executed {result.execution_summary['successful_inversions']} successful inversions")
        
        return result
    
    def invert_single_strategy(self,
                             strategy_returns: pd.Series,
                             strategy_name: str,
                             inversion_type: Optional[InversionType] = None,
                             market_data: Optional[pd.DataFrame] = None,
                             force_inversion: bool = False) -> InversionExecutionResult:
        """
        Invert a single strategy with comprehensive analysis
        
        Args:
            strategy_returns: Strategy return series
            strategy_name: Name of the strategy
            inversion_type: Specific inversion type to use
            market_data: Optional market data for context
            force_inversion: Force inversion regardless of analysis
            
        Returns:
            Execution result for the strategy
        """
        start_time = time.time()
        logger.info(f"Inverting single strategy: {strategy_name}")
        
        # Pre-execution metrics
        pre_metrics = self._calculate_strategy_metrics(strategy_returns)
        
        try:
            # Analyze strategy
            if not force_inversion and inversion_type is None:
                analysis = self.strategy_inverter.analyze_strategy_for_inversion(
                    strategy_returns, strategy_name, market_data
                )
                
                if not analysis['recommended_inversion']:
                    return InversionExecutionResult(
                        strategy_name=strategy_name,
                        execution_status='skipped',
                        inversion_result=None,
                        execution_time=time.time() - start_time,
                        error_message='No viable inversion found',
                        pre_execution_metrics=pre_metrics,
                        post_execution_metrics=pre_metrics
                    )
            
            # Execute inversion
            inversion_result = self.strategy_inverter.invert_strategy(
                strategy_returns, strategy_name, inversion_type, market_data, force_inversion
            )
            
            if inversion_result is None:
                return InversionExecutionResult(
                    strategy_name=strategy_name,
                    execution_status='failed',
                    inversion_result=None,
                    execution_time=time.time() - start_time,
                    error_message='Inversion execution failed',
                    pre_execution_metrics=pre_metrics,
                    post_execution_metrics=pre_metrics
                )
            
            # Post-execution metrics
            post_metrics = self._calculate_strategy_metrics(inversion_result.inverted_returns)
            
            return InversionExecutionResult(
                strategy_name=strategy_name,
                execution_status='success',
                inversion_result=inversion_result,
                execution_time=time.time() - start_time,
                error_message=None,
                pre_execution_metrics=pre_metrics,
                post_execution_metrics=post_metrics
            )
            
        except Exception as e:
            logger.error(f"Error inverting strategy {strategy_name}: {e}")
            return InversionExecutionResult(
                strategy_name=strategy_name,
                execution_status='error',
                inversion_result=None,
                execution_time=time.time() - start_time,
                error_message=str(e),
                pre_execution_metrics=pre_metrics,
                post_execution_metrics=pre_metrics
            )
    
    def create_inverted_portfolio(self,
                                strategy_data: pd.DataFrame,
                                inversion_results: List[InversionExecutionResult]) -> pd.DataFrame:
        """
        Create new portfolio DataFrame with inversions applied
        
        Args:
            strategy_data: Original strategy data
            inversion_results: Results from inversion execution
            
        Returns:
            New DataFrame with inversions applied
        """
        inverted_data = strategy_data.copy()
        
        for result in inversion_results:
            if (result.execution_status == 'success' and 
                result.inversion_result is not None and 
                result.strategy_name in inverted_data.columns):
                
                # Replace original strategy with inverted version
                inverted_returns = result.inversion_result.inverted_returns
                
                # Align indices
                aligned_returns = inverted_returns.reindex(inverted_data.index, fill_value=0)
                inverted_data[result.strategy_name] = aligned_returns
                
                logger.info(f"Applied inversion to {result.strategy_name}")
        
        return inverted_data
    
    def benchmark_inversion_methods(self,
                                  strategy_returns: pd.Series,
                                  strategy_name: str,
                                  market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Benchmark different inversion methods on a strategy
        
        Args:
            strategy_returns: Strategy return series
            strategy_name: Name of the strategy
            market_data: Optional market data for context
            
        Returns:
            Benchmark results for all inversion methods
        """
        logger.info(f"Benchmarking inversion methods for {strategy_name}")
        
        benchmark_results = {
            'strategy_name': strategy_name,
            'original_metrics': self._calculate_strategy_metrics(strategy_returns),
            'inversion_methods': {},
            'best_method': None,
            'benchmark_summary': {}
        }
        
        # Test all available inversion types
        inversion_types = [
            InversionType.SIMPLE,
            InversionType.ULTA,
            InversionType.CONDITIONAL,
            InversionType.ROLLING,
            InversionType.RISK_ADJUSTED
        ]
        
        if self.config.enable_smart_inversion:
            inversion_types.append(InversionType.SMART)
        
        for inversion_type in inversion_types:
            try:
                result = self.strategy_inverter.invert_strategy(
                    strategy_returns, strategy_name, inversion_type, market_data, force_inversion=True
                )
                
                if result is not None:
                    benchmark_results['inversion_methods'][inversion_type.value] = {
                        'inversion_result': result,
                        'improvement_score': result.improvement_score,
                        'confidence_score': result.confidence_score,
                        'risk_score': result.risk_score,
                        'inverted_metrics': result.inverted_metrics
                    }
                
            except Exception as e:
                logger.warning(f"Error benchmarking {inversion_type}: {e}")
                benchmark_results['inversion_methods'][inversion_type.value] = {
                    'error': str(e)
                }
        
        # Identify best method
        valid_methods = {
            k: v for k, v in benchmark_results['inversion_methods'].items()
            if 'improvement_score' in v
        }
        
        if valid_methods:
            best_method = max(
                valid_methods.items(),
                key=lambda x: x[1]['improvement_score'] * x[1]['confidence_score']
            )
            benchmark_results['best_method'] = {
                'method': best_method[0],
                'combined_score': best_method[1]['improvement_score'] * best_method[1]['confidence_score'],
                'details': best_method[1]
            }
        
        # Summary statistics
        if valid_methods:
            improvement_scores = [v['improvement_score'] for v in valid_methods.values()]
            confidence_scores = [v['confidence_score'] for v in valid_methods.values()]
            
            benchmark_results['benchmark_summary'] = {
                'methods_tested': len(inversion_types),
                'successful_methods': len(valid_methods),
                'avg_improvement': np.mean(improvement_scores),
                'best_improvement': np.max(improvement_scores),
                'avg_confidence': np.mean(confidence_scores),
                'method_success_rate': len(valid_methods) / len(inversion_types)
            }
        
        return benchmark_results
    
    def _filter_strategies(self,
                          strategy_columns: List[str],
                          force_inversions: List[str],
                          exclude_strategies: List[str]) -> List[str]:
        """Filter strategies based on inclusion/exclusion criteria"""
        
        # Start with all strategies
        active_strategies = strategy_columns.copy()
        
        # Remove excluded strategies
        active_strategies = [s for s in active_strategies if s not in exclude_strategies]
        
        # Add forced inversions (even if excluded)
        for strategy in force_inversions:
            if strategy in strategy_columns and strategy not in active_strategies:
                active_strategies.append(strategy)
        
        return active_strategies
    
    def _prioritize_opportunities(self,
                                opportunities: List[InversionOpportunity]) -> List[InversionOpportunity]:
        """Prioritize inversion opportunities based on multiple criteria"""
        
        # Calculate priority score for each opportunity
        for opp in opportunities:
            priority_score = (
                0.4 * opp.expected_improvement +
                0.3 * opp.confidence_score +
                0.2 * (1.0 if opp.risk_level == 'low' else 0.5 if opp.risk_level == 'medium' else 0.1) +
                0.1 * (1.0 if 'high' in opp.opportunity_type else 0.5)
            )
            opp.priority_score = priority_score
        
        # Sort by priority score
        prioritized = sorted(opportunities, key=lambda x: getattr(x, 'priority_score', 0), reverse=True)
        
        # Limit to max inversions per run
        return prioritized[:self.config.max_inversions_per_run]
    
    def _comprehensive_risk_assessment(self,
                                     strategy_data: pd.DataFrame,
                                     strategy_columns: List[str],
                                     opportunities: List[InversionOpportunity]) -> Dict[str, Any]:
        """Comprehensive risk assessment for proposed inversions"""
        
        risk_analysis = {
            'portfolio_risk_metrics': {},
            'individual_strategy_risks': {},
            'correlation_impact': {},
            'overall_risk_score': 0.0,
            'risk_recommendations': []
        }
        
        # Portfolio-level risk metrics
        portfolio_returns = strategy_data[strategy_columns].mean(axis=1)
        risk_analysis['portfolio_risk_metrics'] = {
            'portfolio_volatility': portfolio_returns.std() * np.sqrt(252),
            'portfolio_sharpe': self._calculate_sharpe_ratio(portfolio_returns),
            'portfolio_max_drawdown': self._calculate_max_drawdown(portfolio_returns),
            'portfolio_var_95': portfolio_returns.quantile(0.05) * np.sqrt(252)
        }
        
        # Individual strategy risk assessment
        for opp in opportunities:
            if opp.strategy_name in strategy_data.columns:
                strategy_returns = strategy_data[opp.strategy_name]
                
                # Simulate inversion impact
                inverted_returns = -strategy_returns  # Simple simulation
                
                risk_metrics = {
                    'pre_inversion_volatility': strategy_returns.std() * np.sqrt(252),
                    'post_inversion_volatility': inverted_returns.std() * np.sqrt(252),
                    'volatility_change': abs(inverted_returns.std() - strategy_returns.std()),
                    'correlation_with_portfolio': strategy_returns.corr(portfolio_returns),
                    'risk_contribution': self._calculate_risk_contribution(strategy_returns, portfolio_returns)
                }
                
                risk_analysis['individual_strategy_risks'][opp.strategy_name] = risk_metrics
        
        # Calculate overall risk score
        individual_risk_scores = []
        for risk_metrics in risk_analysis['individual_strategy_risks'].values():
            vol_change_score = min(1.0, risk_metrics['volatility_change'] * 5)  # Normalize
            individual_risk_scores.append(vol_change_score)
        
        risk_analysis['overall_risk_score'] = np.mean(individual_risk_scores) if individual_risk_scores else 0.0
        
        # Risk recommendations
        if risk_analysis['overall_risk_score'] > self.config.risk_threshold:
            risk_analysis['risk_recommendations'].append(
                "High overall risk detected - consider reducing number of inversions"
            )
        
        return risk_analysis
    
    def _create_execution_plan(self,
                             opportunities: List[InversionOpportunity],
                             risk_analysis: Dict[str, Any],
                             force_inversions: List[str]) -> List[Dict[str, Any]]:
        """Create execution plan for inversions"""
        
        execution_plan = []
        
        # Add forced inversions first
        for strategy in force_inversions:
            execution_plan.append({
                'strategy_name': strategy,
                'inversion_type': None,  # Will be determined by inverter
                'priority': 'forced',
                'risk_level': 'unknown',
                'force_execution': True
            })
        
        # Add opportunity-based inversions
        for opp in opportunities:
            if opp.strategy_name not in force_inversions:
                
                # Determine inversion type from recommended parameters
                inversion_type_str = opp.recommended_parameters.get('inversion_type', 'simple')
                inversion_type = getattr(InversionType, inversion_type_str.upper(), InversionType.SIMPLE)
                
                execution_plan.append({
                    'strategy_name': opp.strategy_name,
                    'inversion_type': inversion_type,
                    'priority': 'opportunity',
                    'risk_level': opp.risk_level,
                    'force_execution': False,
                    'opportunity': opp
                })
        
        return execution_plan
    
    def _execute_inversions(self,
                          strategy_data: pd.DataFrame,
                          execution_plan: List[Dict[str, Any]],
                          market_data: Optional[pd.DataFrame] = None) -> List[InversionExecutionResult]:
        """Execute the inversion plan"""
        
        execution_results = []
        
        for plan_item in execution_plan:
            strategy_name = plan_item['strategy_name']
            
            if strategy_name not in strategy_data.columns:
                logger.warning(f"Strategy {strategy_name} not found in data")
                continue
            
            strategy_returns = strategy_data[strategy_name]
            inversion_type = plan_item.get('inversion_type')
            force_execution = plan_item.get('force_execution', False)
            
            # Execute inversion
            result = self.invert_single_strategy(
                strategy_returns, strategy_name, inversion_type, market_data, force_execution
            )
            
            execution_results.append(result)
        
        return execution_results
    
    def _analyze_performance_improvement(self,
                                       strategy_data: pd.DataFrame,
                                       execution_results: List[InversionExecutionResult]) -> Dict[str, float]:
        """Analyze performance improvement from inversions"""
        
        performance_improvement = {}
        
        for result in execution_results:
            if result.execution_status == 'success' and result.inversion_result is not None:
                improvement_score = result.inversion_result.improvement_score
                performance_improvement[result.strategy_name] = improvement_score
            else:
                performance_improvement[result.strategy_name] = 0.0
        
        return performance_improvement
    
    def _generate_comprehensive_recommendations(self,
                                             portfolio_analysis: Dict[str, Any],
                                             execution_results: List[InversionExecutionResult],
                                             performance_improvement: Dict[str, float]) -> List[str]:
        """Generate comprehensive recommendations"""
        
        recommendations = []
        
        # Execution summary recommendations
        successful_inversions = sum(1 for r in execution_results if r.execution_status == 'success')
        total_attempted = len(execution_results)
        
        if successful_inversions > 0:
            avg_improvement = np.mean([imp for imp in performance_improvement.values() if imp > 0])
            recommendations.append(
                f"Successfully inverted {successful_inversions}/{total_attempted} strategies "
                f"with average improvement of {avg_improvement:.3f}"
            )
        
        # Strategy-specific recommendations
        top_improvements = sorted(
            performance_improvement.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        if top_improvements and top_improvements[0][1] > 0.1:
            recommendations.append(
                f"Top performing inversions: {', '.join([name for name, _ in top_improvements])}"
            )
        
        # Risk recommendations
        high_risk_inversions = [
            r.strategy_name for r in execution_results
            if (r.execution_status == 'success' and
                r.inversion_result is not None and
                r.inversion_result.risk_score > 0.7)
        ]
        
        if high_risk_inversions:
            recommendations.append(
                f"Monitor high-risk inversions: {', '.join(high_risk_inversions)}"
            )
        
        # Portfolio-level recommendations
        portfolio_recommendations = portfolio_analysis.get('recommendations', [])
        recommendations.extend(portfolio_recommendations[:3])  # Top 3 portfolio recommendations
        
        return recommendations
    
    def _calculate_strategy_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate basic strategy metrics"""
        if len(returns) == 0:
            return {}
        
        return {
            'total_return': returns.sum(),
            'mean_return': returns.mean(),
            'volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'max_drawdown': self._calculate_max_drawdown(returns),
            'win_rate': (returns > 0).mean()
        }
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns.mean() * 252 - risk_free_rate
        volatility = returns.std() * np.sqrt(252)
        return excess_returns / (volatility + 1e-8)
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())
    
    def _calculate_risk_contribution(self, strategy_returns: pd.Series, portfolio_returns: pd.Series) -> float:
        """Calculate strategy's contribution to portfolio risk"""
        correlation = strategy_returns.corr(portfolio_returns)
        strategy_vol = strategy_returns.std()
        portfolio_vol = portfolio_returns.std()
        
        # Simplified risk contribution
        return correlation * strategy_vol / portfolio_vol
    
    def _save_results(self, result: InversionEngineResult):
        """Save results to output directory"""
        if not self.output_directory:
            return
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main result as JSON
        result_dict = self._convert_result_to_dict(result)
        result_file = self.output_directory / f"inversion_result_{timestamp}.json"
        
        with open(result_file, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        
        # Save individual inversion results
        for exec_result in result.execution_results:
            if exec_result.inversion_result is not None:
                strategy_file = self.output_directory / f"inversion_{exec_result.strategy_name}_{timestamp}.json"
                
                strategy_dict = {
                    'strategy_name': exec_result.strategy_name,
                    'execution_status': exec_result.execution_status,
                    'execution_time': exec_result.execution_time,
                    'pre_metrics': exec_result.pre_execution_metrics,
                    'post_metrics': exec_result.post_execution_metrics,
                    'inversion_details': {
                        'inversion_type': exec_result.inversion_result.inversion_type.value,
                        'improvement_score': exec_result.inversion_result.improvement_score,
                        'confidence_score': exec_result.inversion_result.confidence_score,
                        'risk_score': exec_result.inversion_result.risk_score
                    }
                }
                
                with open(strategy_file, 'w') as f:
                    json.dump(strategy_dict, f, indent=2, default=str)
        
        logger.info(f"Results saved to {self.output_directory}")
    
    def _convert_result_to_dict(self, result: InversionEngineResult) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization"""
        result_dict = {
            'engine_config': asdict(result.engine_config),
            'execution_summary': result.execution_summary,
            'execution_results_summary': {
                'total_executions': len(result.execution_results),
                'successful_executions': sum(1 for r in result.execution_results if r.execution_status == 'success'),
                'failed_executions': sum(1 for r in result.execution_results if r.execution_status == 'failed'),
                'error_executions': sum(1 for r in result.execution_results if r.execution_status == 'error')
            },
            'performance_improvement': result.performance_improvement,
            'recommendations': result.recommendations,
            'execution_metadata': result.execution_metadata
        }
        
        return result_dict
    
    def _update_performance_metrics(self, result: InversionEngineResult):
        """Update engine performance metrics"""
        self.performance_metrics['total_strategies_analyzed'] += result.execution_summary['strategies_analyzed']
        self.performance_metrics['successful_inversions'] += result.execution_summary['successful_inversions']
        self.performance_metrics['total_improvement'] += result.execution_summary['total_improvement_score']
        
        # Update average execution time
        total_executions = len(self.execution_history) + 1
        current_avg = self.performance_metrics['average_execution_time']
        new_time = result.execution_summary['total_execution_time']
        self.performance_metrics['average_execution_time'] = (
            (current_avg * len(self.execution_history) + new_time) / total_executions
        )
        
        # Count errors
        error_count = sum(1 for r in result.execution_results if r.execution_status == 'error')
        self.performance_metrics['error_count'] += error_count
        
        # Add to execution history
        self.execution_history.append(result)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get engine performance metrics"""
        return {
            'performance_metrics': self.performance_metrics.copy(),
            'execution_count': len(self.execution_history),
            'average_improvement_per_execution': (
                self.performance_metrics['total_improvement'] / 
                max(1, self.performance_metrics['successful_inversions'])
            ),
            'success_rate': (
                self.performance_metrics['successful_inversions'] / 
                max(1, self.performance_metrics['total_strategies_analyzed'])
            )
        }
    
    def reset_performance_metrics(self):
        """Reset performance tracking"""
        self.performance_metrics = {
            'total_strategies_analyzed': 0,
            'successful_inversions': 0,
            'total_improvement': 0.0,
            'average_execution_time': 0.0,
            'error_count': 0
        }
        self.execution_history.clear()
        logger.info("Performance metrics reset")