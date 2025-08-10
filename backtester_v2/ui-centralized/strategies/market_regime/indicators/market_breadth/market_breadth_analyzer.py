"""
Market Breadth Analyzer - Main Orchestrator for Market Breadth V2
================================================================

Main orchestrator for comprehensive market breadth analysis combining
option and underlying breadth metrics with composite analysis.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

# Import option breadth components
from .option_breadth import (
    OptionVolumeFlow, OptionRatioAnalyzer, OptionMomentum, SectorBreadth
)

# Import underlying breadth components
from .underlying_breadth import (
    AdvanceDeclineAnalyzer, VolumeFlowIndicator, NewHighsLows, ParticipationRatio
)

# Import composite components
from .composite import (
    BreadthDivergenceDetector, RegimeBreadthClassifier, BreadthMomentumScorer
)

logger = logging.getLogger(__name__)


class MarketBreadthAnalyzer:
    """
    Main orchestrator for Market Breadth V2
    
    Coordinates all option and underlying breadth components to provide
    comprehensive market breadth analysis and regime detection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Market Breadth Analyzer"""
        self.config = config
        
        # Initialize option breadth components
        self.option_volume_flow = OptionVolumeFlow(config.get('option_volume_config', {}))
        self.option_ratio_analyzer = OptionRatioAnalyzer(config.get('option_ratio_config', {}))
        self.option_momentum = OptionMomentum(config.get('option_momentum_config', {}))
        self.sector_breadth = SectorBreadth(config.get('sector_breadth_config', {}))
        
        # Initialize underlying breadth components
        self.advance_decline_analyzer = AdvanceDeclineAnalyzer(config.get('advance_decline_config', {}))
        self.volume_flow_indicator = VolumeFlowIndicator(config.get('volume_flow_config', {}))
        self.new_highs_lows = NewHighsLows(config.get('new_highs_lows_config', {}))
        self.participation_ratio = ParticipationRatio(config.get('participation_ratio_config', {}))
        
        # Initialize composite components
        self.breadth_divergence_detector = BreadthDivergenceDetector(config.get('divergence_config', {}))
        self.regime_breadth_classifier = RegimeBreadthClassifier(config.get('regime_classifier_config', {}))
        self.breadth_momentum_scorer = BreadthMomentumScorer(config.get('momentum_scorer_config', {}))
        
        # Performance tracking
        self.performance_metrics = {
            'calculations': 0,
            'errors': 0,
            'avg_calculation_time': 0,
            'component_health': {}
        }
        
        logger.info("MarketBreadthAnalyzer initialized with comprehensive breadth analysis")
    
    def analyze_market_breadth(self,
                             option_data: pd.DataFrame,
                             underlying_data: Optional[pd.DataFrame] = None,
                             historical_data: Optional[Dict[str, pd.DataFrame]] = None,
                             market_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive market breadth analysis
        
        Args:
            option_data: DataFrame with option market data
            underlying_data: Optional underlying asset data
            historical_data: Optional historical data for trend analysis
            market_context: Optional market context information
            
        Returns:
            Dict with complete market breadth analysis results
        """
        start_time = datetime.now()
        
        try:
            results = {
                'timestamp': datetime.now(),
                'option_breadth_analysis': {},
                'underlying_breadth_analysis': {},
                'composite_analysis': {},
                'breadth_signals': {},
                'breadth_regime': {},
                'health_status': {}
            }
            
            # Option Breadth Analysis
            results['option_breadth_analysis'] = self._perform_option_breadth_analysis(
                option_data, historical_data, market_context
            )
            
            # Underlying Breadth Analysis
            results['underlying_breadth_analysis'] = self._perform_underlying_breadth_analysis(
                underlying_data, historical_data, market_context
            )
            
            # Composite Analysis
            results['composite_analysis'] = self._perform_composite_analysis(
                results['option_breadth_analysis'],
                results['underlying_breadth_analysis'],
                market_context
            )
            
            # Generate consolidated breadth signals
            results['breadth_signals'] = self._generate_breadth_signals(results)
            
            # Classify breadth regime
            results['breadth_regime'] = self._classify_breadth_regime(results)
            
            # Check component health
            results['health_status'] = self._check_component_health()
            
            # Update performance metrics
            self._update_performance_metrics(start_time)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in market breadth analysis: {e}")
            self.performance_metrics['errors'] += 1
            return self._get_default_results()
    
    def _perform_option_breadth_analysis(self, 
                                       option_data: pd.DataFrame,
                                       historical_data: Optional[Dict[str, pd.DataFrame]],
                                       market_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform comprehensive option breadth analysis"""
        try:
            option_analysis = {}
            
            # Option Volume Flow Analysis
            try:
                option_analysis['volume_flow'] = self.option_volume_flow.analyze_volume_flow(option_data)
                self.performance_metrics['component_health']['option_volume_flow'] = 'healthy'
            except Exception as e:
                logger.error(f"Error in option volume flow analysis: {e}")
                option_analysis['volume_flow'] = {'breadth_score': 0.5}
                self.performance_metrics['component_health']['option_volume_flow'] = 'error'
            
            # Option Ratio Analysis
            try:
                option_analysis['ratio_analysis'] = self.option_ratio_analyzer.analyze_option_ratios(option_data)
                self.performance_metrics['component_health']['option_ratio_analyzer'] = 'healthy'
            except Exception as e:
                logger.error(f"Error in option ratio analysis: {e}")
                option_analysis['ratio_analysis'] = {'breadth_score': 0.5}
                self.performance_metrics['component_health']['option_ratio_analyzer'] = 'error'
            
            # Option Momentum Analysis
            try:
                historical_option_data = historical_data.get('option_data') if historical_data else None
                option_analysis['momentum_analysis'] = self.option_momentum.analyze_option_momentum(
                    option_data, historical_option_data
                )
                self.performance_metrics['component_health']['option_momentum'] = 'healthy'
            except Exception as e:
                logger.error(f"Error in option momentum analysis: {e}")
                option_analysis['momentum_analysis'] = {'breadth_score': 0.5}
                self.performance_metrics['component_health']['option_momentum'] = 'error'
            
            # Sector Breadth Analysis
            try:
                underlying_data_for_sector = historical_data.get('underlying_data') if historical_data else None
                option_analysis['sector_analysis'] = self.sector_breadth.analyze_sector_breadth(
                    option_data, underlying_data_for_sector
                )
                self.performance_metrics['component_health']['sector_breadth'] = 'healthy'
            except Exception as e:
                logger.error(f"Error in sector breadth analysis: {e}")
                option_analysis['sector_analysis'] = {'breadth_score': 0.5}
                self.performance_metrics['component_health']['sector_breadth'] = 'error'
            
            # Calculate consolidated option breadth score
            option_analysis['consolidated_score'] = self._calculate_option_breadth_score(option_analysis)
            
            return option_analysis
            
        except Exception as e:
            logger.error(f"Error performing option breadth analysis: {e}")
            return {}
    
    def _perform_underlying_breadth_analysis(self,
                                           underlying_data: Optional[pd.DataFrame],
                                           historical_data: Optional[Dict[str, pd.DataFrame]],
                                           market_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform comprehensive underlying breadth analysis"""
        try:
            underlying_analysis = {}
            
            if underlying_data is None or underlying_data.empty:
                logger.warning("No underlying data provided for breadth analysis")
                return self._get_default_underlying_analysis()
            
            # Advance/Decline Analysis
            try:
                price_changes = historical_data.get('price_changes') if historical_data else None
                underlying_analysis['advance_decline'] = self.advance_decline_analyzer.analyze_advance_decline(
                    underlying_data, price_changes
                )
                self.performance_metrics['component_health']['advance_decline_analyzer'] = 'healthy'
            except Exception as e:
                logger.error(f"Error in advance/decline analysis: {e}")
                underlying_analysis['advance_decline'] = {'breadth_score': 0.5}
                self.performance_metrics['component_health']['advance_decline_analyzer'] = 'error'
            
            # Volume Flow Analysis
            try:
                price_data = historical_data.get('price_data') if historical_data else None
                underlying_analysis['volume_flow'] = self.volume_flow_indicator.analyze_volume_flow(
                    underlying_data, price_data
                )
                self.performance_metrics['component_health']['volume_flow_indicator'] = 'healthy'
            except Exception as e:
                logger.error(f"Error in volume flow analysis: {e}")
                underlying_analysis['volume_flow'] = {'breadth_score': 0.5}
                self.performance_metrics['component_health']['volume_flow_indicator'] = 'error'
            
            # New Highs/Lows Analysis
            try:
                historical_price_data = historical_data.get('historical_prices') if historical_data else None
                underlying_analysis['new_highs_lows'] = self.new_highs_lows.analyze_new_highs_lows(
                    underlying_data, historical_price_data
                )
                self.performance_metrics['component_health']['new_highs_lows'] = 'healthy'
            except Exception as e:
                logger.error(f"Error in new highs/lows analysis: {e}")
                underlying_analysis['new_highs_lows'] = {'breadth_score': 0.5}
                self.performance_metrics['component_health']['new_highs_lows'] = 'error'
            
            # Participation Ratio Analysis
            try:
                sector_data = self._extract_sector_data(underlying_data) if underlying_data is not None else None
                underlying_analysis['participation_ratio'] = self.participation_ratio.analyze_participation_ratio(
                    underlying_data, sector_data
                )
                self.performance_metrics['component_health']['participation_ratio'] = 'healthy'
            except Exception as e:
                logger.error(f"Error in participation ratio analysis: {e}")
                underlying_analysis['participation_ratio'] = {'breadth_score': 0.5}
                self.performance_metrics['component_health']['participation_ratio'] = 'error'
            
            # Calculate consolidated underlying breadth score
            underlying_analysis['consolidated_score'] = self._calculate_underlying_breadth_score(underlying_analysis)
            
            return underlying_analysis
            
        except Exception as e:
            logger.error(f"Error performing underlying breadth analysis: {e}")
            return {}
    
    def _perform_composite_analysis(self,
                                  option_breadth: Dict[str, Any],
                                  underlying_breadth: Dict[str, Any],
                                  market_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform composite breadth analysis"""
        try:
            composite_analysis = {}
            
            # Breadth Divergence Detection
            try:
                composite_analysis['divergence_analysis'] = self.breadth_divergence_detector.detect_breadth_divergences(
                    option_breadth, underlying_breadth, market_context
                )
                self.performance_metrics['component_health']['breadth_divergence_detector'] = 'healthy'
            except Exception as e:
                logger.error(f"Error in divergence detection: {e}")
                composite_analysis['divergence_analysis'] = {'breadth_alignment_score': 0.5}
                self.performance_metrics['component_health']['breadth_divergence_detector'] = 'error'
            
            # Breadth Momentum Scoring
            try:
                composite_analysis['momentum_scoring'] = self.breadth_momentum_scorer.calculate_momentum_scores(
                    option_breadth, underlying_breadth
                )
                self.performance_metrics['component_health']['breadth_momentum_scorer'] = 'healthy'
            except Exception as e:
                logger.error(f"Error in momentum scoring: {e}")
                composite_analysis['momentum_scoring'] = {'composite_momentum': {'score': 0.0}}
                self.performance_metrics['component_health']['breadth_momentum_scorer'] = 'error'
            
            # Regime Classification
            try:
                divergence_analysis = composite_analysis.get('divergence_analysis')
                composite_analysis['regime_classification'] = self.regime_breadth_classifier.classify_market_regime(
                    option_breadth, underlying_breadth, divergence_analysis, market_context
                )
                self.performance_metrics['component_health']['regime_breadth_classifier'] = 'healthy'
            except Exception as e:
                logger.error(f"Error in regime classification: {e}")
                composite_analysis['regime_classification'] = {'primary_regime': {'regime_type': 'neutral'}}
                self.performance_metrics['component_health']['regime_breadth_classifier'] = 'error'
            
            return composite_analysis
            
        except Exception as e:
            logger.error(f"Error performing composite analysis: {e}")
            return {}
    
    def _generate_breadth_signals(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate consolidated breadth signals"""
        try:
            signals = {
                'primary_signal': 'neutral',
                'signal_strength': 0.0,
                'breadth_signals': [],
                'component_signals': {},
                'signal_consensus': 'mixed'
            }
            
            # Extract component signals
            option_signals = results.get('option_breadth_analysis', {})
            underlying_signals = results.get('underlying_breadth_analysis', {})
            composite_signals = results.get('composite_analysis', {})
            
            # Collect all component signals
            component_signals = {}
            
            # Option component signals
            if 'volume_flow' in option_signals:
                vol_signal = option_signals['volume_flow'].get('flow_signals', {}).get('primary_signal', 'neutral')
                component_signals['option_volume_flow'] = vol_signal
                if vol_signal != 'neutral':
                    signals['breadth_signals'].append(f'option_{vol_signal}')
            
            if 'ratio_analysis' in option_signals:
                ratio_signal = option_signals['ratio_analysis'].get('breadth_signals', {}).get('primary_signal', 'neutral')
                component_signals['option_ratios'] = ratio_signal
                if ratio_signal != 'neutral':
                    signals['breadth_signals'].append(f'ratio_{ratio_signal}')
            
            if 'momentum_analysis' in option_signals:
                momentum_signal = option_signals['momentum_analysis'].get('momentum_signals', {}).get('primary_signal', 'neutral')
                component_signals['option_momentum'] = momentum_signal
                if momentum_signal != 'neutral':
                    signals['breadth_signals'].append(f'option_momentum_{momentum_signal}')
            
            if 'sector_analysis' in option_signals:
                sector_signal = option_signals['sector_analysis'].get('sector_signals', {}).get('primary_signal', 'neutral')
                component_signals['sector_breadth'] = sector_signal
                if sector_signal != 'neutral':
                    signals['breadth_signals'].append(f'sector_{sector_signal}')
            
            # Underlying component signals
            if 'advance_decline' in underlying_signals:
                ad_signal = underlying_signals['advance_decline'].get('breadth_signals', {}).get('primary_signal', 'neutral')
                component_signals['advance_decline'] = ad_signal
                if ad_signal != 'neutral':
                    signals['breadth_signals'].append(f'ad_{ad_signal}')
            
            if 'volume_flow' in underlying_signals:
                vol_flow_signal = underlying_signals['volume_flow'].get('flow_signals', {}).get('primary_signal', 'neutral')
                component_signals['underlying_volume_flow'] = vol_flow_signal
                if vol_flow_signal != 'neutral':
                    signals['breadth_signals'].append(f'underlying_{vol_flow_signal}')
            
            if 'new_highs_lows' in underlying_signals:
                hl_signal = underlying_signals['new_highs_lows'].get('hl_signals', {}).get('primary_signal', 'neutral')
                component_signals['new_highs_lows'] = hl_signal
                if hl_signal != 'neutral':
                    signals['breadth_signals'].append(f'hl_{hl_signal}')
            
            if 'participation_ratio' in underlying_signals:
                participation_signal = underlying_signals['participation_ratio'].get('participation_signals', {}).get('primary_signal', 'neutral')
                component_signals['participation'] = participation_signal
                if participation_signal != 'neutral':
                    signals['breadth_signals'].append(f'participation_{participation_signal}')
            
            # Composite signals
            if 'divergence_analysis' in composite_signals:
                divergence_signal = composite_signals['divergence_analysis'].get('divergence_signals', {}).get('primary_signal', 'neutral')
                component_signals['divergence'] = divergence_signal
                if divergence_signal != 'neutral':
                    signals['breadth_signals'].append(f'divergence_{divergence_signal}')
            
            if 'momentum_scoring' in composite_signals:
                composite_momentum_signal = composite_signals['momentum_scoring'].get('momentum_signals', {}).get('primary_signal', 'neutral')
                component_signals['composite_momentum'] = composite_momentum_signal
                if composite_momentum_signal != 'neutral':
                    signals['breadth_signals'].append(f'momentum_{composite_momentum_signal}')
            
            if 'regime_classification' in composite_signals:
                regime_signal = composite_signals['regime_classification'].get('regime_signals', {}).get('primary_signal', 'neutral')
                component_signals['regime'] = regime_signal
                if regime_signal != 'neutral':
                    signals['breadth_signals'].append(f'regime_{regime_signal}')
            
            signals['component_signals'] = component_signals
            
            # Calculate signal consensus
            non_neutral_signals = [sig for sig in component_signals.values() if sig != 'neutral']
            
            if len(non_neutral_signals) == 0:
                signals['signal_consensus'] = 'neutral'
                signals['primary_signal'] = 'neutral'
            else:
                # Count bullish vs bearish signals
                bullish_signals = sum(1 for sig in non_neutral_signals if any(word in sig.lower() for word in ['bullish', 'positive', 'expanding', 'thrust']))
                bearish_signals = sum(1 for sig in non_neutral_signals if any(word in sig.lower() for word in ['bearish', 'negative', 'contracting', 'decline']))
                
                total_signals = len(non_neutral_signals)
                
                # Calculate signal strength based on consensus
                if total_signals > 0:
                    consensus_ratio = max(bullish_signals, bearish_signals) / total_signals
                    signals['signal_strength'] = float(consensus_ratio)
                    
                    if consensus_ratio >= 0.8:
                        signals['signal_consensus'] = 'strong_consensus'
                    elif consensus_ratio >= 0.6:
                        signals['signal_consensus'] = 'moderate_consensus'
                    else:
                        signals['signal_consensus'] = 'mixed'
                    
                    # Determine primary signal
                    if bullish_signals > bearish_signals:
                        if consensus_ratio >= 0.8:
                            signals['primary_signal'] = 'strong_bullish_breadth'
                        elif consensus_ratio >= 0.6:
                            signals['primary_signal'] = 'moderate_bullish_breadth'
                        else:
                            signals['primary_signal'] = 'weak_bullish_breadth'
                    elif bearish_signals > bullish_signals:
                        if consensus_ratio >= 0.8:
                            signals['primary_signal'] = 'strong_bearish_breadth'
                        elif consensus_ratio >= 0.6:
                            signals['primary_signal'] = 'moderate_bearish_breadth'
                        else:
                            signals['primary_signal'] = 'weak_bearish_breadth'
                    else:
                        signals['primary_signal'] = 'mixed_breadth_signals'
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating breadth signals: {e}")
            return {'primary_signal': 'neutral', 'signal_strength': 0.0}
    
    def _classify_breadth_regime(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Classify overall breadth regime"""
        try:
            regime = {
                'breadth_regime': 'neutral',
                'regime_confidence': 0.0,
                'regime_characteristics': [],
                'breadth_quality': 'moderate'
            }
            
            # Get regime classification from composite analysis
            composite_analysis = results.get('composite_analysis', {})
            if 'regime_classification' in composite_analysis:
                regime_data = composite_analysis['regime_classification']
                
                regime['breadth_regime'] = regime_data.get('primary_regime', {}).get('regime_type', 'neutral')
                regime['regime_confidence'] = regime_data.get('regime_confidence', {}).get('overall_confidence', 0.0)
                regime['regime_characteristics'] = regime_data.get('secondary_characteristics', {})
                regime['breadth_quality'] = regime_data.get('breadth_quality', {}).get('overall_quality', 'moderate')
            
            # Enhance with additional breadth context
            option_score = results.get('option_breadth_analysis', {}).get('consolidated_score', 0.5)
            underlying_score = results.get('underlying_breadth_analysis', {}).get('consolidated_score', 0.5)
            
            regime['option_breadth_score'] = float(option_score)
            regime['underlying_breadth_score'] = float(underlying_score)
            regime['composite_breadth_score'] = float((option_score + underlying_score) / 2)
            
            # Add divergence context
            divergence_analysis = composite_analysis.get('divergence_analysis', {})
            if divergence_analysis:
                regime['breadth_alignment'] = divergence_analysis.get('breadth_alignment_score', 0.5)
                regime['divergence_risk'] = divergence_analysis.get('divergence_severity', {}).get('risk_level', 'low')
            
            return regime
            
        except Exception as e:
            logger.error(f"Error classifying breadth regime: {e}")
            return {'breadth_regime': 'neutral', 'regime_confidence': 0.0}
    
    def _calculate_option_breadth_score(self, option_analysis: Dict[str, Any]) -> float:
        """Calculate consolidated option breadth score"""
        try:
            scores = []
            
            # Extract individual component scores
            if 'volume_flow' in option_analysis:
                scores.append(option_analysis['volume_flow'].get('breadth_score', 0.5))
            
            if 'ratio_analysis' in option_analysis:
                scores.append(option_analysis['ratio_analysis'].get('breadth_score', 0.5))
            
            if 'momentum_analysis' in option_analysis:
                scores.append(option_analysis['momentum_analysis'].get('breadth_score', 0.5))
            
            if 'sector_analysis' in option_analysis:
                scores.append(option_analysis['sector_analysis'].get('breadth_score', 0.5))
            
            # Calculate weighted average (equal weights for now)
            if scores:
                return float(np.mean(scores))
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error calculating option breadth score: {e}")
            return 0.5
    
    def _calculate_underlying_breadth_score(self, underlying_analysis: Dict[str, Any]) -> float:
        """Calculate consolidated underlying breadth score"""
        try:
            scores = []
            
            # Extract individual component scores
            if 'advance_decline' in underlying_analysis:
                scores.append(underlying_analysis['advance_decline'].get('breadth_score', 0.5))
            
            if 'volume_flow' in underlying_analysis:
                scores.append(underlying_analysis['volume_flow'].get('breadth_score', 0.5))
            
            if 'new_highs_lows' in underlying_analysis:
                scores.append(underlying_analysis['new_highs_lows'].get('breadth_score', 0.5))
            
            if 'participation_ratio' in underlying_analysis:
                scores.append(underlying_analysis['participation_ratio'].get('breadth_score', 0.5))
            
            # Calculate weighted average (equal weights for now)
            if scores:
                return float(np.mean(scores))
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error calculating underlying breadth score: {e}")
            return 0.5
    
    def _extract_sector_data(self, underlying_data: pd.DataFrame) -> Optional[Dict[str, pd.DataFrame]]:
        """Extract sector data from underlying data if available"""
        try:
            if 'sector' in underlying_data.columns:
                sector_data = {}
                for sector in underlying_data['sector'].unique():
                    sector_data[sector] = underlying_data[underlying_data['sector'] == sector]
                return sector_data
            return None
            
        except Exception as e:
            logger.error(f"Error extracting sector data: {e}")
            return None
    
    def _check_component_health(self) -> Dict[str, Any]:
        """Check health status of all components"""
        try:
            health = {
                'overall_status': 'healthy',
                'component_status': self.performance_metrics['component_health'].copy(),
                'error_rate': 0.0,
                'recommendations': []
            }
            
            # Calculate error rate
            total_calcs = self.performance_metrics['calculations']
            if total_calcs > 0:
                health['error_rate'] = self.performance_metrics['errors'] / total_calcs
            
            # Check for unhealthy components
            unhealthy = [
                comp for comp, status in health['component_status'].items()
                if status != 'healthy'
            ]
            
            if unhealthy:
                health['overall_status'] = 'degraded' if len(unhealthy) < 5 else 'unhealthy'
                health['recommendations'].append(f"Check components: {', '.join(unhealthy)}")
            
            # Check error rate
            if health['error_rate'] > 0.2:
                health['overall_status'] = 'unhealthy'
                health['recommendations'].append("High error rate detected in breadth analysis")
            
            return health
            
        except Exception as e:
            logger.error(f"Error checking component health: {e}")
            return {'overall_status': 'unknown'}
    
    def _update_performance_metrics(self, start_time: datetime):
        """Update performance metrics"""
        try:
            # Update calculation count
            self.performance_metrics['calculations'] += 1
            
            # Update average calculation time
            calc_time = (datetime.now() - start_time).total_seconds()
            current_avg = self.performance_metrics['avg_calculation_time']
            count = self.performance_metrics['calculations']
            
            # Running average
            self.performance_metrics['avg_calculation_time'] = (
                (current_avg * (count - 1) + calc_time) / count
            )
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def _get_default_results(self) -> Dict[str, Any]:
        """Get default results structure"""
        return {
            'timestamp': datetime.now(),
            'option_breadth_analysis': {},
            'underlying_breadth_analysis': {},
            'composite_analysis': {},
            'breadth_signals': {'primary_signal': 'neutral', 'signal_strength': 0.0},
            'breadth_regime': {'breadth_regime': 'neutral', 'regime_confidence': 0.0},
            'health_status': {'overall_status': 'error'}
        }
    
    def _get_default_underlying_analysis(self) -> Dict[str, Any]:
        """Get default underlying analysis when no data available"""
        return {
            'advance_decline': {'breadth_score': 0.5},
            'volume_flow': {'breadth_score': 0.5},
            'new_highs_lows': {'breadth_score': 0.5},
            'participation_ratio': {'breadth_score': 0.5},
            'consolidated_score': 0.5
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of market breadth analysis system"""
        try:
            # Get summaries from all components
            component_summaries = {}
            
            try:
                component_summaries['option_volume_flow'] = self.option_volume_flow.get_flow_summary()
            except:
                component_summaries['option_volume_flow'] = {'status': 'error'}
            
            try:
                component_summaries['option_ratio_analyzer'] = self.option_ratio_analyzer.get_ratio_summary()
            except:
                component_summaries['option_ratio_analyzer'] = {'status': 'error'}
            
            try:
                component_summaries['option_momentum'] = self.option_momentum.get_momentum_summary()
            except:
                component_summaries['option_momentum'] = {'status': 'error'}
            
            try:
                component_summaries['sector_breadth'] = self.sector_breadth.get_sector_summary()
            except:
                component_summaries['sector_breadth'] = {'status': 'error'}
            
            try:
                component_summaries['advance_decline_analyzer'] = self.advance_decline_analyzer.get_ad_summary()
            except:
                component_summaries['advance_decline_analyzer'] = {'status': 'error'}
            
            try:
                component_summaries['volume_flow_indicator'] = self.volume_flow_indicator.get_volume_flow_summary()
            except:
                component_summaries['volume_flow_indicator'] = {'status': 'error'}
            
            try:
                component_summaries['new_highs_lows'] = self.new_highs_lows.get_hl_summary()
            except:
                component_summaries['new_highs_lows'] = {'status': 'error'}
            
            try:
                component_summaries['participation_ratio'] = self.participation_ratio.get_participation_summary()
            except:
                component_summaries['participation_ratio'] = {'status': 'error'}
            
            try:
                component_summaries['breadth_divergence_detector'] = self.breadth_divergence_detector.get_divergence_summary()
            except:
                component_summaries['breadth_divergence_detector'] = {'status': 'error'}
            
            try:
                component_summaries['regime_breadth_classifier'] = self.regime_breadth_classifier.get_regime_summary()
            except:
                component_summaries['regime_breadth_classifier'] = {'status': 'error'}
            
            try:
                component_summaries['breadth_momentum_scorer'] = self.breadth_momentum_scorer.get_momentum_summary()
            except:
                component_summaries['breadth_momentum_scorer'] = {'status': 'error'}
            
            return {
                'performance_metrics': self.performance_metrics.copy(),
                'component_summaries': component_summaries,
                'system_status': self._check_component_health()
            }
            
        except Exception as e:
            logger.error(f"Error getting market breadth summary: {e}")
            return {'status': 'error', 'error': str(e)}