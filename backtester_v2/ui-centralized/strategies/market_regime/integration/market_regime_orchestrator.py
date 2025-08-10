"""
Market Regime Orchestrator - Main System Orchestration
====================================================

Central orchestrator for the complete market regime analysis system.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import all major components
from ..indicators.straddle_analysis.straddle_engine import StraddleAnalysisEngine
from ..indicators.oi_pa_analysis.oi_pa_analyzer import OIPAAnalyzer
from ..indicators.greek_sentiment.greek_sentiment_analyzer import GreekSentimentAnalyzer
from ..indicators.market_breadth.market_breadth_analyzer import MarketBreadthAnalyzer
from ..indicators.iv_analytics.iv_analytics_analyzer import IVAnalyticsAnalyzer
from ..indicators.technical_indicators.technical_indicators_analyzer import TechnicalIndicatorsAnalyzer

# Import optimization components
from ..adaptive_optimization.core.historical_optimizer import HistoricalOptimizer
from ..adaptive_optimization.core.performance_evaluator import PerformanceEvaluator
from ..adaptive_optimization.core.weight_validator import WeightValidator

# Import base utilities
from ..base.common_utils import MathUtils, TimeUtils, ErrorHandler, CacheUtils

# Import new modular components
from ..base.output import OutputOrchestrator
from ..base.trading_modes import TradingModeOrchestrator

logger = logging.getLogger(__name__)


class MarketRegimeOrchestrator:
    """Central orchestrator for market regime analysis system"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Market Regime Orchestrator"""
        self.config = config
        self.execution_mode = config.get('execution_mode', 'parallel')  # parallel, sequential, adaptive
        self.max_workers = config.get('max_workers', 4)
        
        # Initialize all component analyzers
        self._initialize_components()
        
        # Initialize optimization components
        self._initialize_optimization_components()
        
        # Initialize new modular components
        self._initialize_output_system()
        self._initialize_trading_modes()
        
        # Performance tracking
        self.performance_metrics = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'avg_execution_time': 0.0,
            'component_performance': {}
        }
        
        # Execution state
        self.execution_history = {
            'analyses': [],
            'timestamps': [],
            'performance_scores': [],
            'regime_classifications': []
        }
        
        # Mathematical utilities
        self.math_utils = MathUtils()
        self.time_utils = TimeUtils()
        self.cache = CacheUtils(max_size=100)
        
        logger.info("MarketRegimeOrchestrator initialized with comprehensive system orchestration")
    
    def _initialize_components(self):
        """Initialize all analysis components"""
        try:
            # Core analysis components
            self.straddle_analyzer = StraddleAnalysisEngine(self.config.get('straddle_config', {}))
            self.oi_pa_analyzer = OIPAAnalyzer(self.config.get('oi_pa_config', {}))
            self.greek_sentiment_analyzer = GreekSentimentAnalyzer(self.config.get('greek_sentiment_config', {}))
            self.market_breadth_analyzer = MarketBreadthAnalyzer(self.config.get('market_breadth_config', {}))
            self.iv_analytics_analyzer = IVAnalyticsAnalyzer(self.config.get('iv_analytics_config', {}))
            self.technical_indicators_analyzer = TechnicalIndicatorsAnalyzer(self.config.get('technical_indicators_config', {}))
            
            # Component registry
            self.components = {
                'straddle_analysis': self.straddle_analyzer,
                'oi_pa_analysis': self.oi_pa_analyzer,
                'greek_sentiment': self.greek_sentiment_analyzer,
                'market_breadth': self.market_breadth_analyzer,
                'iv_analytics': self.iv_analytics_analyzer,
                'technical_indicators': self.technical_indicators_analyzer
            }
            
            # Component weights
            self.component_weights = self.config.get('component_weights', {
                'straddle_analysis': 0.25,
                'oi_pa_analysis': 0.20,
                'greek_sentiment': 0.15,
                'market_breadth': 0.25,
                'iv_analytics': 0.10,
                'technical_indicators': 0.05
            })
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            self.components = {}
    
    def _initialize_optimization_components(self):
        """Initialize optimization components"""
        try:
            opt_config = self.config.get('optimization_config', {})
            
            self.historical_optimizer = HistoricalOptimizer(opt_config.get('historical_optimizer', {}))
            self.performance_evaluator = PerformanceEvaluator(opt_config.get('performance_evaluator', {}))
            self.weight_validator = WeightValidator(opt_config.get('weight_validator', {}))
            
        except Exception as e:
            logger.error(f"Error initializing optimization components: {e}")
            self.historical_optimizer = None
            self.performance_evaluator = None
            self.weight_validator = None
    
    def _initialize_output_system(self):
        """Initialize output orchestrator"""
        try:
            output_config = self.config.get('output_config', {})
            self.output_orchestrator = OutputOrchestrator(output_config)
            logger.info("Output orchestrator initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing output system: {e}")
            self.output_orchestrator = None
    
    def _initialize_trading_modes(self):
        """Initialize trading mode orchestrator"""
        try:
            trading_mode_config = self.config.get('trading_mode_config', {})
            self.trading_mode_orchestrator = TradingModeOrchestrator(trading_mode_config)
            logger.info("Trading mode orchestrator initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing trading modes: {e}")
            self.trading_mode_orchestrator = None
    
    async def analyze_market_regime(self,
                                  option_data: pd.DataFrame,
                                  underlying_data: Optional[pd.DataFrame] = None,
                                  market_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Comprehensive market regime analysis
        
        Args:
            option_data: Option market data
            underlying_data: Optional underlying asset data
            market_context: Optional market context information
            
        Returns:
            Dict with complete market regime analysis
        """
        start_time = datetime.now()
        
        try:
            # Validate input data
            if not self._validate_input_data(option_data, underlying_data):
                return self._get_default_analysis_result()
            
            # Check cache
            cache_key = self._generate_cache_key(option_data, underlying_data, market_context)
            cached_result = self.cache.get(cache_key)
            if cached_result:
                return cached_result
            
            # Execute component analysis
            if self.execution_mode == 'parallel':
                component_results = await self._execute_parallel_analysis(option_data, underlying_data, market_context)
            else:
                component_results = await self._execute_sequential_analysis(option_data, underlying_data, market_context)
            
            # Aggregate results
            aggregated_result = self._aggregate_component_results(component_results)
            
            # Generate final regime classification
            regime_classification = self._classify_market_regime(aggregated_result, component_results)
            
            # Calculate confidence scores
            confidence_analysis = self._calculate_confidence_scores(component_results, regime_classification)
            
            # Performance optimization (if enabled)
            optimization_result = await self._perform_optimization_analysis(component_results, aggregated_result)
            
            # Compile final result
            final_result = {
                'analysis_timestamp': datetime.now(),
                'regime_classification': regime_classification,
                'component_results': component_results,
                'aggregated_analysis': aggregated_result,
                'confidence_analysis': confidence_analysis,
                'optimization_analysis': optimization_result,
                'execution_metadata': self._generate_execution_metadata(start_time, component_results)
            }
            
            # Cache result
            self.cache.set(cache_key, final_result)
            
            # Update performance metrics
            self._update_performance_metrics(start_time, True, component_results)
            
            # Update execution history
            self._update_execution_history(final_result)
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error in market regime analysis: {e}")
            self._update_performance_metrics(start_time, False, {})
            return self._get_default_analysis_result()
    
    async def _execute_parallel_analysis(self,
                                       option_data: pd.DataFrame,
                                       underlying_data: Optional[pd.DataFrame],
                                       market_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute component analysis in parallel"""
        try:
            component_results = {}
            
            # Create execution tasks
            tasks = []
            loop = asyncio.get_event_loop()
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Straddle Analysis
                if 'straddle_analysis' in self.components:
                    task = loop.run_in_executor(
                        executor,
                        self._safe_component_execution,
                        'straddle_analysis',
                        self.straddle_analyzer.analyze_straddle_patterns,
                        option_data, underlying_data, market_context
                    )
                    tasks.append(('straddle_analysis', task))
                
                # OI-PA Analysis
                if 'oi_pa_analysis' in self.components:
                    task = loop.run_in_executor(
                        executor,
                        self._safe_component_execution,
                        'oi_pa_analysis',
                        self.oi_pa_analyzer.analyze_oi_pa_patterns,
                        option_data, underlying_data, market_context
                    )
                    tasks.append(('oi_pa_analysis', task))
                
                # Greek Sentiment Analysis
                if 'greek_sentiment' in self.components:
                    task = loop.run_in_executor(
                        executor,
                        self._safe_component_execution,
                        'greek_sentiment',
                        self.greek_sentiment_analyzer.analyze_greek_sentiment,
                        option_data, underlying_data, market_context
                    )
                    tasks.append(('greek_sentiment', task))
                
                # Market Breadth Analysis
                if 'market_breadth' in self.components:
                    task = loop.run_in_executor(
                        executor,
                        self._safe_component_execution,
                        'market_breadth',
                        self.market_breadth_analyzer.analyze_market_breadth,
                        option_data, underlying_data, market_context
                    )
                    tasks.append(('market_breadth', task))
                
                # IV Analytics Analysis
                if 'iv_analytics' in self.components:
                    task = loop.run_in_executor(
                        executor,
                        self._safe_component_execution,
                        'iv_analytics',
                        self.iv_analytics_analyzer.analyze_iv_patterns,
                        option_data, underlying_data, market_context
                    )
                    tasks.append(('iv_analytics', task))
                
                # Technical Indicators Analysis
                if 'technical_indicators' in self.components:
                    task = loop.run_in_executor(
                        executor,
                        self._safe_component_execution,
                        'technical_indicators',
                        self.technical_indicators_analyzer.analyze_technical_patterns,
                        option_data, underlying_data, market_context
                    )
                    tasks.append(('technical_indicators', task))
                
                # Collect results
                for component_name, task in tasks:
                    try:
                        result = await task
                        component_results[component_name] = result
                    except Exception as e:
                        logger.error(f"Error in {component_name} analysis: {e}")
                        component_results[component_name] = self._get_default_component_result()
            
            return component_results
            
        except Exception as e:
            logger.error(f"Error in parallel execution: {e}")
            return {}
    
    async def _execute_sequential_analysis(self,
                                         option_data: pd.DataFrame,
                                         underlying_data: Optional[pd.DataFrame],
                                         market_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute component analysis sequentially"""
        try:
            component_results = {}
            
            # Execute each component sequentially
            for component_name, component in self.components.items():
                try:
                    if component_name == 'straddle_analysis':
                        result = self.straddle_analyzer.analyze_straddle_patterns(option_data, underlying_data, market_context)
                    elif component_name == 'oi_pa_analysis':
                        result = self.oi_pa_analyzer.analyze_oi_pa_patterns(option_data, underlying_data, market_context)
                    elif component_name == 'greek_sentiment':
                        result = self.greek_sentiment_analyzer.analyze_greek_sentiment(option_data, underlying_data, market_context)
                    elif component_name == 'market_breadth':
                        result = self.market_breadth_analyzer.analyze_market_breadth(option_data, underlying_data, market_context)
                    elif component_name == 'iv_analytics':
                        result = self.iv_analytics_analyzer.analyze_iv_patterns(option_data, underlying_data, market_context)
                    elif component_name == 'technical_indicators':
                        result = self.technical_indicators_analyzer.analyze_technical_patterns(option_data, underlying_data, market_context)
                    else:
                        result = self._get_default_component_result()
                    
                    component_results[component_name] = result
                    
                except Exception as e:
                    logger.error(f"Error in {component_name} analysis: {e}")
                    component_results[component_name] = self._get_default_component_result()
            
            return component_results
            
        except Exception as e:
            logger.error(f"Error in sequential execution: {e}")
            return {}
    
    def _safe_component_execution(self,
                                component_name: str,
                                analysis_function: callable,
                                option_data: pd.DataFrame,
                                underlying_data: Optional[pd.DataFrame],
                                market_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Safely execute component analysis with error handling"""
        try:
            return analysis_function(option_data, underlying_data, market_context)
        except Exception as e:
            logger.error(f"Error in {component_name} safe execution: {e}")
            return self._get_default_component_result()
    
    def _aggregate_component_results(self, component_results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results from all components"""
        try:
            aggregated = {
                'composite_score': 0.0,
                'regime_signals': [],
                'key_insights': [],
                'risk_indicators': [],
                'component_contributions': {}
            }
            
            total_weight = 0.0
            weighted_scores = []
            
            # Aggregate component scores and signals
            for component_name, result in component_results.items():
                if not result or 'status' in result and result['status'] == 'error':
                    continue
                
                weight = self.component_weights.get(component_name, 0.0)
                total_weight += weight
                
                # Extract component score
                component_score = self._extract_component_score(result)
                weighted_scores.append(component_score * weight)
                
                # Collect signals
                signals = self._extract_component_signals(result)
                aggregated['regime_signals'].extend(signals)
                
                # Collect insights
                insights = self._extract_component_insights(result)
                aggregated['key_insights'].extend(insights)
                
                # Collect risk indicators
                risks = self._extract_component_risks(result)
                aggregated['risk_indicators'].extend(risks)
                
                # Track component contribution
                aggregated['component_contributions'][component_name] = {
                    'score': float(component_score),
                    'weight': float(weight),
                    'weighted_contribution': float(component_score * weight),
                    'signal_count': len(signals)
                }
            
            # Calculate composite score
            if total_weight > 0:
                aggregated['composite_score'] = float(sum(weighted_scores) / total_weight)
            
            # Normalize weights if needed
            if total_weight != 1.0 and total_weight > 0:
                for component_name in aggregated['component_contributions']:
                    aggregated['component_contributions'][component_name]['normalized_weight'] = (
                        aggregated['component_contributions'][component_name]['weight'] / total_weight
                    )
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Error aggregating component results: {e}")
            return {'composite_score': 0.0, 'regime_signals': [], 'key_insights': [], 'risk_indicators': []}
    
    def _classify_market_regime(self,
                              aggregated_result: Dict[str, Any],
                              component_results: Dict[str, Any]) -> Dict[str, Any]:
        """Classify overall market regime"""
        try:
            classification = {
                'primary_regime': 'neutral',
                'regime_confidence': 0.0,
                'regime_characteristics': [],
                'regime_strength': 'moderate',
                'supporting_evidence': {},
                'regime_transitions': {}
            }
            
            composite_score = aggregated_result.get('composite_score', 0.0)
            regime_signals = aggregated_result.get('regime_signals', [])
            
            # Primary regime classification based on composite score
            if composite_score > 0.7:
                classification['primary_regime'] = 'strong_bullish'
                classification['regime_strength'] = 'strong'
            elif composite_score > 0.55:
                classification['primary_regime'] = 'moderate_bullish'
                classification['regime_strength'] = 'moderate'
            elif composite_score > 0.45:
                classification['primary_regime'] = 'neutral'
                classification['regime_strength'] = 'weak'
            elif composite_score > 0.3:
                classification['primary_regime'] = 'moderate_bearish'
                classification['regime_strength'] = 'moderate'
            else:
                classification['primary_regime'] = 'strong_bearish'
                classification['regime_strength'] = 'strong'
            
            # Calculate regime confidence
            classification['regime_confidence'] = self._calculate_regime_confidence(aggregated_result, component_results)
            
            # Extract regime characteristics
            classification['regime_characteristics'] = self._extract_regime_characteristics(regime_signals)
            
            # Analyze supporting evidence
            classification['supporting_evidence'] = self._analyze_supporting_evidence(component_results)
            
            # Detect regime transitions
            classification['regime_transitions'] = self._detect_regime_transitions(component_results)
            
            return classification
            
        except Exception as e:
            logger.error(f"Error classifying market regime: {e}")
            return {'primary_regime': 'neutral', 'regime_confidence': 0.0}
    
    def _calculate_confidence_scores(self,
                                   component_results: Dict[str, Any],
                                   regime_classification: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive confidence scores"""
        try:
            confidence = {
                'overall_confidence': 0.0,
                'component_confidence': {},
                'signal_consensus': 0.0,
                'data_quality_score': 0.0,
                'confidence_factors': {}
            }
            
            confidence_scores = []
            signal_alignment = []
            
            # Calculate component-level confidence
            for component_name, result in component_results.items():
                if not result or 'status' in result and result['status'] == 'error':
                    continue
                
                component_confidence = self._extract_component_confidence(result)
                confidence['component_confidence'][component_name] = float(component_confidence)
                confidence_scores.append(component_confidence)
                
                # Check signal alignment with primary regime
                component_signals = self._extract_component_signals(result)
                alignment = self._calculate_signal_alignment(component_signals, regime_classification['primary_regime'])
                signal_alignment.append(alignment)
            
            # Overall confidence
            if confidence_scores:
                confidence['overall_confidence'] = float(np.mean(confidence_scores))
            
            # Signal consensus
            if signal_alignment:
                confidence['signal_consensus'] = float(np.mean(signal_alignment))
            
            # Data quality assessment
            confidence['data_quality_score'] = self._assess_data_quality(component_results)
            
            # Confidence factors
            confidence['confidence_factors'] = {
                'component_agreement': confidence['signal_consensus'],
                'data_quality': confidence['data_quality_score'],
                'historical_accuracy': self._get_historical_accuracy_score(),
                'sample_size_adequacy': self._assess_sample_size_adequacy(component_results)
            }
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating confidence scores: {e}")
            return {'overall_confidence': 0.0}
    
    async def _perform_optimization_analysis(self,
                                           component_results: Dict[str, Any],
                                           aggregated_result: Dict[str, Any]) -> Dict[str, Any]:
        """Perform optimization analysis if enabled"""
        try:
            if not self.historical_optimizer:
                return {'optimization_enabled': False}
            
            optimization_result = {
                'optimization_enabled': True,
                'weight_validation': {},
                'performance_evaluation': {},
                'parameter_optimization': {}
            }
            
            # Weight validation
            if self.weight_validator:
                component_scores = {name: pd.Series([result.get('composite_score', 0.0)]) 
                                  for name, result in component_results.items()}
                weight_validation = self.weight_validator.validate_weights(
                    self.component_weights, component_scores
                )
                optimization_result['weight_validation'] = weight_validation
            
            # Performance evaluation
            if self.performance_evaluator and self.execution_history['performance_scores']:
                performance_series = pd.Series(self.execution_history['performance_scores'])
                regime_series = pd.Series(self.execution_history['regime_classifications'])
                
                performance_eval = self.performance_evaluator.evaluate_performance(
                    performance_series, regime_series
                )
                optimization_result['performance_evaluation'] = performance_eval
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Error in optimization analysis: {e}")
            return {'optimization_enabled': False, 'error': str(e)}
    
    def _validate_input_data(self, option_data: pd.DataFrame, underlying_data: Optional[pd.DataFrame]) -> bool:
        """Validate input data quality"""
        try:
            if option_data.empty:
                logger.warning("Empty option data provided")
                return False
            
            # Check required columns
            required_columns = ['strike', 'option_type', 'volume', 'oi']
            missing_columns = [col for col in required_columns if col not in option_data.columns]
            
            if missing_columns:
                logger.warning(f"Missing required columns: {missing_columns}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating input data: {e}")
            return False
    
    def _generate_cache_key(self,
                          option_data: pd.DataFrame,
                          underlying_data: Optional[pd.DataFrame],
                          market_context: Optional[Dict[str, Any]]) -> str:
        """Generate cache key for analysis results"""
        try:
            # Create hash from data characteristics
            data_hash = hash(str(option_data.shape) + str(option_data.columns.tolist()))
            
            if underlying_data is not None:
                data_hash += hash(str(underlying_data.shape))
            
            if market_context:
                data_hash += hash(str(sorted(market_context.items())))
            
            return f"market_regime_analysis_{abs(data_hash)}"
            
        except Exception as e:
            logger.error(f"Error generating cache key: {e}")
            return "default_cache_key"
    
    def _extract_component_score(self, result: Dict[str, Any]) -> float:
        """Extract normalized score from component result"""
        try:
            # Try multiple possible score keys
            score_keys = ['composite_score', 'overall_score', 'score', 'regime_score']
            
            for key in score_keys:
                if key in result:
                    return float(result[key])
            
            # If no direct score, try to calculate from sub-components
            if 'analysis_results' in result:
                sub_results = result['analysis_results']
                if isinstance(sub_results, dict):
                    scores = []
                    for sub_key, sub_value in sub_results.items():
                        if isinstance(sub_value, (int, float)):
                            scores.append(float(sub_value))
                        elif isinstance(sub_value, dict) and 'score' in sub_value:
                            scores.append(float(sub_value['score']))
                    
                    if scores:
                        return np.mean(scores)
            
            return 0.5  # Neutral default
            
        except Exception as e:
            logger.error(f"Error extracting component score: {e}")
            return 0.5
    
    def _extract_component_signals(self, result: Dict[str, Any]) -> List[str]:
        """Extract signals from component result"""
        try:
            signals = []
            
            # Try multiple possible signal keys
            signal_keys = ['regime_signals', 'signals', 'indicators', 'patterns']
            
            for key in signal_keys:
                if key in result:
                    signal_data = result[key]
                    if isinstance(signal_data, list):
                        signals.extend([str(s) for s in signal_data])
                    elif isinstance(signal_data, dict):
                        signals.extend([f"{k}:{v}" for k, v in signal_data.items()])
                    elif isinstance(signal_data, str):
                        signals.append(signal_data)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error extracting component signals: {e}")
            return []
    
    def _extract_component_insights(self, result: Dict[str, Any]) -> List[str]:
        """Extract insights from component result"""
        try:
            insights = []
            
            insight_keys = ['insights', 'key_findings', 'observations', 'conclusions']
            
            for key in insight_keys:
                if key in result:
                    insight_data = result[key]
                    if isinstance(insight_data, list):
                        insights.extend([str(i) for i in insight_data])
                    elif isinstance(insight_data, str):
                        insights.append(insight_data)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error extracting component insights: {e}")
            return []
    
    def _extract_component_risks(self, result: Dict[str, Any]) -> List[str]:
        """Extract risk indicators from component result"""
        try:
            risks = []
            
            risk_keys = ['risk_indicators', 'risks', 'warnings', 'alerts']
            
            for key in risk_keys:
                if key in result:
                    risk_data = result[key]
                    if isinstance(risk_data, list):
                        risks.extend([str(r) for r in risk_data])
                    elif isinstance(risk_data, str):
                        risks.append(risk_data)
            
            return risks
            
        except Exception as e:
            logger.error(f"Error extracting component risks: {e}")
            return []
    
    def _extract_component_confidence(self, result: Dict[str, Any]) -> float:
        """Extract confidence score from component result"""
        try:
            confidence_keys = ['confidence', 'confidence_score', 'reliability', 'certainty']
            
            for key in confidence_keys:
                if key in result:
                    return float(result[key])
            
            # Default confidence based on data completeness
            if 'status' in result and result['status'] == 'success':
                return 0.8
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error extracting component confidence: {e}")
            return 0.5
    
    def _calculate_regime_confidence(self,
                                   aggregated_result: Dict[str, Any],
                                   component_results: Dict[str, Any]) -> float:
        """Calculate overall regime classification confidence"""
        try:
            confidence_factors = []
            
            # Component agreement factor
            component_scores = [self._extract_component_score(result) for result in component_results.values()]
            if component_scores:
                score_std = np.std(component_scores)
                agreement_factor = max(0, 1 - score_std)  # Lower std = higher agreement
                confidence_factors.append(agreement_factor)
            
            # Signal consensus factor
            all_signals = []
            for result in component_results.values():
                all_signals.extend(self._extract_component_signals(result))
            
            if all_signals:
                bullish_signals = sum(1 for s in all_signals if any(term in s.lower() for term in ['bullish', 'positive', 'up']))
                bearish_signals = sum(1 for s in all_signals if any(term in s.lower() for term in ['bearish', 'negative', 'down']))
                total_directional = bullish_signals + bearish_signals
                
                if total_directional > 0:
                    consensus_factor = max(bullish_signals, bearish_signals) / total_directional
                    confidence_factors.append(consensus_factor)
            
            # Composite score extremity (more extreme = more confident)
            composite_score = aggregated_result.get('composite_score', 0.5)
            extremity_factor = abs(composite_score - 0.5) * 2  # Convert to 0-1 scale
            confidence_factors.append(extremity_factor)
            
            # Overall confidence
            if confidence_factors:
                return float(np.mean(confidence_factors))
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error calculating regime confidence: {e}")
            return 0.5
    
    def _extract_regime_characteristics(self, regime_signals: List[str]) -> List[str]:
        """Extract regime characteristics from signals"""
        try:
            characteristics = []
            
            # Analyze signal patterns
            signal_text = ' '.join(regime_signals).lower()
            
            # Volatility characteristics
            if any(term in signal_text for term in ['high_volatility', 'volatile', 'uncertainty']):
                characteristics.append('high_volatility')
            elif any(term in signal_text for term in ['low_volatility', 'stable', 'calm']):
                characteristics.append('low_volatility')
            
            # Trend characteristics
            if any(term in signal_text for term in ['trending', 'momentum', 'directional']):
                characteristics.append('trending')
            elif any(term in signal_text for term in ['ranging', 'sideways', 'consolidation']):
                characteristics.append('ranging')
            
            # Volume characteristics
            if any(term in signal_text for term in ['high_volume', 'active', 'participation']):
                characteristics.append('high_participation')
            elif any(term in signal_text for term in ['low_volume', 'quiet', 'thin']):
                characteristics.append('low_participation')
            
            # Sentiment characteristics
            if any(term in signal_text for term in ['bullish', 'optimistic', 'positive']):
                characteristics.append('bullish_sentiment')
            elif any(term in signal_text for term in ['bearish', 'pessimistic', 'negative']):
                characteristics.append('bearish_sentiment')
            
            return characteristics
            
        except Exception as e:
            logger.error(f"Error extracting regime characteristics: {e}")
            return []
    
    def _analyze_supporting_evidence(self, component_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze supporting evidence for regime classification"""
        try:
            evidence = {
                'strong_support': [],
                'moderate_support': [],
                'weak_support': [],
                'conflicting_signals': []
            }
            
            for component_name, result in component_results.items():
                component_score = self._extract_component_score(result)
                component_confidence = self._extract_component_confidence(result)
                
                # Classify evidence strength
                if component_confidence > 0.8 and abs(component_score - 0.5) > 0.3:
                    evidence['strong_support'].append({
                        'component': component_name,
                        'score': float(component_score),
                        'confidence': float(component_confidence)
                    })
                elif component_confidence > 0.6 and abs(component_score - 0.5) > 0.15:
                    evidence['moderate_support'].append({
                        'component': component_name,
                        'score': float(component_score),
                        'confidence': float(component_confidence)
                    })
                elif component_confidence > 0.4:
                    evidence['weak_support'].append({
                        'component': component_name,
                        'score': float(component_score),
                        'confidence': float(component_confidence)
                    })
                else:
                    evidence['conflicting_signals'].append({
                        'component': component_name,
                        'score': float(component_score),
                        'confidence': float(component_confidence)
                    })
            
            return evidence
            
        except Exception as e:
            logger.error(f"Error analyzing supporting evidence: {e}")
            return {'strong_support': [], 'moderate_support': [], 'weak_support': [], 'conflicting_signals': []}
    
    def _detect_regime_transitions(self, component_results: Dict[str, Any]) -> Dict[str, Any]:
        """Detect potential regime transitions"""
        try:
            transitions = {
                'transition_probability': 0.0,
                'transition_direction': 'stable',
                'transition_indicators': [],
                'transition_timeline': 'unknown'
            }
            
            # Check for transition indicators in component results
            transition_signals = []
            
            for component_name, result in component_results.items():
                signals = self._extract_component_signals(result)
                
                # Look for transition-related signals
                for signal in signals:
                    if any(term in signal.lower() for term in ['transition', 'change', 'shift', 'reversal']):
                        transition_signals.append(f"{component_name}: {signal}")
            
            # Calculate transition probability
            if transition_signals:
                transitions['transition_probability'] = min(len(transition_signals) / 10, 1.0)
                transitions['transition_indicators'] = transition_signals
                
                # Determine transition direction
                bullish_transitions = sum(1 for s in transition_signals if 'bullish' in s.lower())
                bearish_transitions = sum(1 for s in transition_signals if 'bearish' in s.lower())
                
                if bullish_transitions > bearish_transitions:
                    transitions['transition_direction'] = 'bullish'
                elif bearish_transitions > bullish_transitions:
                    transitions['transition_direction'] = 'bearish'
                else:
                    transitions['transition_direction'] = 'uncertain'
            
            return transitions
            
        except Exception as e:
            logger.error(f"Error detecting regime transitions: {e}")
            return {'transition_probability': 0.0, 'transition_direction': 'stable'}
    
    def _calculate_signal_alignment(self, signals: List[str], primary_regime: str) -> float:
        """Calculate how well signals align with primary regime"""
        try:
            if not signals:
                return 0.5
            
            aligned_signals = 0
            total_signals = len(signals)
            
            # Define regime keywords
            if 'bullish' in primary_regime:
                target_keywords = ['bullish', 'positive', 'up', 'strong', 'expansion']
            elif 'bearish' in primary_regime:
                target_keywords = ['bearish', 'negative', 'down', 'weak', 'contraction']
            else:
                # Neutral regime
                target_keywords = ['neutral', 'stable', 'range', 'consolidation']
            
            # Count aligned signals
            for signal in signals:
                if any(keyword in signal.lower() for keyword in target_keywords):
                    aligned_signals += 1
            
            return aligned_signals / total_signals
            
        except Exception as e:
            logger.error(f"Error calculating signal alignment: {e}")
            return 0.5
    
    def _assess_data_quality(self, component_results: Dict[str, Any]) -> float:
        """Assess overall data quality across components"""
        try:
            quality_scores = []
            
            for component_name, result in component_results.items():
                # Check for data quality indicators
                if 'data_quality' in result:
                    quality_scores.append(float(result['data_quality']))
                elif 'status' in result:
                    if result['status'] == 'success':
                        quality_scores.append(0.9)
                    elif result['status'] == 'warning':
                        quality_scores.append(0.6)
                    else:
                        quality_scores.append(0.3)
                else:
                    quality_scores.append(0.7)  # Default
            
            return float(np.mean(quality_scores)) if quality_scores else 0.5
            
        except Exception as e:
            logger.error(f"Error assessing data quality: {e}")
            return 0.5
    
    def _get_historical_accuracy_score(self) -> float:
        """Get historical accuracy score from past analyses"""
        try:
            if len(self.execution_history['performance_scores']) < 5:
                return 0.7  # Default for insufficient history
            
            recent_scores = self.execution_history['performance_scores'][-10:]
            return float(np.mean(recent_scores))
            
        except Exception as e:
            logger.error(f"Error getting historical accuracy score: {e}")
            return 0.7
    
    def _assess_sample_size_adequacy(self, component_results: Dict[str, Any]) -> float:
        """Assess if sample size is adequate for analysis"""
        try:
            # This is a simplified assessment
            # In practice, would check actual data sizes in components
            successful_components = sum(1 for result in component_results.values() 
                                      if result and result.get('status') != 'error')
            
            total_components = len(self.components)
            adequacy = successful_components / total_components if total_components > 0 else 0
            
            return float(adequacy)
            
        except Exception as e:
            logger.error(f"Error assessing sample size adequacy: {e}")
            return 0.5
    
    def _generate_execution_metadata(self,
                                   start_time: datetime,
                                   component_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate execution metadata"""
        try:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            return {
                'start_time': start_time,
                'end_time': end_time,
                'execution_time_seconds': float(execution_time),
                'execution_mode': self.execution_mode,
                'components_executed': len(component_results),
                'successful_components': sum(1 for r in component_results.values() if r and r.get('status') != 'error'),
                'failed_components': sum(1 for r in component_results.values() if not r or r.get('status') == 'error'),
                'cache_hit': False,  # Would be True if result came from cache
                'version': '2.0.0'
            }
            
        except Exception as e:
            logger.error(f"Error generating execution metadata: {e}")
            return {'execution_time_seconds': 0.0, 'version': '2.0.0'}
    
    def _update_performance_metrics(self,
                                  start_time: datetime,
                                  success: bool,
                                  component_results: Dict[str, Any]):
        """Update performance metrics"""
        try:
            self.performance_metrics['total_analyses'] += 1
            
            if success:
                self.performance_metrics['successful_analyses'] += 1
            else:
                self.performance_metrics['failed_analyses'] += 1
            
            # Update execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            current_avg = self.performance_metrics['avg_execution_time']
            total_analyses = self.performance_metrics['total_analyses']
            
            # Running average
            self.performance_metrics['avg_execution_time'] = (
                (current_avg * (total_analyses - 1) + execution_time) / total_analyses
            )
            
            # Update component performance
            for component_name, result in component_results.items():
                if component_name not in self.performance_metrics['component_performance']:
                    self.performance_metrics['component_performance'][component_name] = {
                        'total_executions': 0,
                        'successful_executions': 0,
                        'avg_execution_time': 0.0
                    }
                
                comp_perf = self.performance_metrics['component_performance'][component_name]
                comp_perf['total_executions'] += 1
                
                if result and result.get('status') != 'error':
                    comp_perf['successful_executions'] += 1
                    
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def _update_execution_history(self, final_result: Dict[str, Any]):
        """Update execution history"""
        try:
            self.execution_history['analyses'].append(final_result)
            self.execution_history['timestamps'].append(final_result['analysis_timestamp'])
            
            # Extract performance score
            composite_score = final_result.get('aggregated_analysis', {}).get('composite_score', 0.0)
            self.execution_history['performance_scores'].append(composite_score)
            
            # Extract regime classification
            primary_regime = final_result.get('regime_classification', {}).get('primary_regime', 'neutral')
            self.execution_history['regime_classifications'].append(primary_regime)
            
            # Trim history to reasonable size
            max_history = 100
            for key in self.execution_history.keys():
                if len(self.execution_history[key]) > max_history:
                    self.execution_history[key] = self.execution_history[key][-max_history:]
                    
        except Exception as e:
            logger.error(f"Error updating execution history: {e}")
    
    def _get_default_analysis_result(self) -> Dict[str, Any]:
        """Get default analysis result when analysis fails"""
        return {
            'analysis_timestamp': datetime.now(),
            'regime_classification': {
                'primary_regime': 'neutral',
                'regime_confidence': 0.0,
                'regime_characteristics': [],
                'regime_strength': 'unknown'
            },
            'component_results': {},
            'aggregated_analysis': {
                'composite_score': 0.0,
                'regime_signals': [],
                'key_insights': ['Analysis failed - insufficient data or system error'],
                'risk_indicators': ['High uncertainty due to analysis failure']
            },
            'confidence_analysis': {'overall_confidence': 0.0},
            'optimization_analysis': {'optimization_enabled': False},
            'execution_metadata': {
                'execution_time_seconds': 0.0,
                'components_executed': 0,
                'successful_components': 0,
                'failed_components': 0,
                'status': 'failed'
            }
        }
    
    def _get_default_component_result(self) -> Dict[str, Any]:
        """Get default component result when component fails"""
        return {
            'status': 'error',
            'composite_score': 0.0,
            'confidence': 0.0,
            'regime_signals': [],
            'insights': ['Component analysis failed'],
            'risk_indicators': ['Component unavailable']
        }
    
    async def generate_complete_output(
        self,
        regime_analysis_result: Dict[str, Any],
        option_data: pd.DataFrame,
        underlying_data: Optional[pd.DataFrame] = None,
        symbol: str = 'NIFTY',
        timeframe: str = '1min'
    ) -> Dict[str, Any]:
        """Generate complete output using integrated output and trading mode systems"""
        try:
            if not self.output_orchestrator or not self.trading_mode_orchestrator:
                return {
                    'success': False,
                    'error': 'Output or trading mode orchestrator not initialized'
                }
            
            # Optimize trading mode configuration
            current_mode = self.config.get('trading_mode', 'hybrid')
            trading_optimization = self.trading_mode_orchestrator.optimize_for_current_mode(
                underlying_data if underlying_data is not None else option_data,
                option_data,
                regime_analysis_result.get('performance_metrics')
            )
            
            # Prepare regime data for output generation
            regime_data = pd.DataFrame([
                {
                    'timestamp': datetime.now(),
                    'regime_name': regime_analysis_result.get('primary_regime', 'unknown'),
                    'confidence_score': regime_analysis_result.get('confidence_score', 0.5),
                    'final_score': regime_analysis_result.get('final_score', 0.0)
                }
            ])
            
            # Get optimized parameters
            optimized_params = {}
            if trading_optimization.get('success'):
                optimized_params = trading_optimization.get('optimized_parameters', {})
            
            # Add original analysis parameters
            optimized_params.update({
                'regime_analysis': regime_analysis_result,
                'trading_mode': current_mode,
                'symbol': symbol,
                'timeframe': timeframe,
                'generation_timestamp': datetime.now().isoformat()
            })
            
            # Generate complete output
            output_result = self.output_orchestrator.generate_complete_output(
                regime_data,
                underlying_data if underlying_data is not None else option_data,
                optimized_params,
                symbol,
                timeframe
            )
            
            return {
                'success': True,
                'regime_analysis': regime_analysis_result,
                'trading_mode_optimization': trading_optimization,
                'output_generation': output_result,
                'summary': {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'trading_mode': current_mode,
                    'regime_detected': regime_analysis_result.get('primary_regime', 'unknown'),
                    'confidence': regime_analysis_result.get('confidence_score', 0.5),
                    'output_files_generated': len(output_result.get('outputs', {})) if output_result.get('success') else 0
                }
            }
            
        except Exception as e:
            error_msg = f"Error generating complete output: {e}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'regime_analysis': regime_analysis_result
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            return {
                'orchestrator_status': 'operational',
                'components_status': {name: 'loaded' for name in self.components.keys()},
                'performance_metrics': self.performance_metrics.copy(),
                'execution_history_length': len(self.execution_history['analyses']),
                'cache_status': {
                    'cache_size': len(self.cache.cache),
                    'max_cache_size': self.cache.max_size
                },
                'configuration': {
                    'execution_mode': self.execution_mode,
                    'max_workers': self.max_workers,
                    'component_weights': self.component_weights
                },
                'optimization_status': {
                    'historical_optimizer': self.historical_optimizer is not None,
                    'performance_evaluator': self.performance_evaluator is not None,
                    'weight_validator': self.weight_validator is not None
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'orchestrator_status': 'error', 'error': str(e)}